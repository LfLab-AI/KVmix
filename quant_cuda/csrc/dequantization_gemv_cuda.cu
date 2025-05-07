#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "dequantization_gemv_cuda.h"

// Template-based kernel for quantized matrix-vector multiplication across various bit widths
template<int QUANT_BITS, int ELEMENTS_PER_PACK>
__global__ void execute_quantized_matvec(
    const half* input_vector,
    const uint32_t* packed_weights,
    const half* quantization_zeros,
    const half* quantization_scales,
    half* result_vector,
    const int input_size,
    const int output_size,
    const int quantization_group,
    const int total_heads,
    const int kv_heads
) {
    // Determine thread and block positions
    const int sample_index = blockIdx.x;
    const int packed_output_group = blockIdx.y * blockDim.y + threadIdx.y;
    const int output_start = packed_output_group * ELEMENTS_PER_PACK;
    const int group_index = output_start / quantization_group;

    // Establish data pointers with offsets
    const half* input_base = input_vector + sample_index * input_size;
    half* output_base = result_vector + sample_index * output_size;
    const int head_division = total_heads / kv_heads;
    const int weight_batch = sample_index / head_division;
    const uint32_t* weights_base = packed_weights + weight_batch * (output_size * input_size / ELEMENTS_PER_PACK);
    const half* scales_base = quantization_scales + weight_batch * (output_size * input_size / quantization_group);
    const half* zeros_base = quantization_zeros + weight_batch * (output_size * input_size / quantization_group);

    // Define processing chunk size
    const int CHUNK_SIZE = 128;

    // Initialize accumulation array
    float accumulators[ELEMENTS_PER_PACK] = {0.0f};

    // Calculate total chunks to process
    const int chunk_count = (input_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < chunk_count; chunk++) {
        // Temporary storage for loaded data
        uint32_t weight_chunks[4] = {0};
        half scale_chunks[4] = {0};
        half zero_chunks[4] = {0};
        half input_chunks[4] = {0};

        // Compute memory offsets
        int weight_base_offset = packed_output_group * input_size + chunk * CHUNK_SIZE + threadIdx.x * 4;
        int scale_base_offset = group_index * input_size + chunk * CHUNK_SIZE + threadIdx.x * 4;
        int input_base_offset = chunk * CHUNK_SIZE + threadIdx.x * 4;

        // Load data safely within bounds
        for (int idx = 0; idx < 4; idx++) {
            int w_offset = weight_base_offset + idx;
            if (w_offset < output_size * input_size / ELEMENTS_PER_PACK) {
                weight_chunks[idx] = weights_base[w_offset];
            }
            int s_offset = scale_base_offset + idx;
            if (s_offset < output_size * input_size / quantization_group) {
                scale_chunks[idx] = scales_base[s_offset];
                zero_chunks[idx] = zeros_base[s_offset];
            }
            int i_offset = input_base_offset + idx;
            if (i_offset < input_size) {
                input_chunks[idx] = input_base[i_offset];
            }
        }

        // Process each chunk of loaded data
        #pragma unroll
        for (int segment = 0; segment < 4; segment++) {
            uint32_t current_weight_pack = weight_chunks[segment];
            float current_input = __half2float(input_chunks[segment]);
            float current_scale = __half2float(scale_chunks[segment]);
            float current_zero = __half2float(zero_chunks[segment]);

            if (QUANT_BITS == 3) {
                // Handle 3-bit quantization with special unpacking (10x3-bit + 1x2-bit)
                for (int pos = 0; pos < 10; pos++) {
                    if (pos < ELEMENTS_PER_PACK) {
                        float weight_value = (float)(current_weight_pack & 0x7);
                        current_weight_pack >>= 3;
                        float restored_weight = current_scale * weight_value + current_zero;
                        accumulators[pos] += restored_weight * current_input;
                    }
                }
                if (ELEMENTS_PER_PACK > 10) {
                    float weight_value = (float)(current_weight_pack & 0x3);
                    float restored_weight = current_scale * weight_value + current_zero;
                    accumulators[10] += restored_weight * current_input;
                }
            } else {
                // Standard unpacking for 1, 2, 4-bit quantization
                const int extraction_mask = (1 << QUANT_BITS) - 1;
                for (int pos = 0; pos < ELEMENTS_PER_PACK; pos++) {
                    float weight_value = (float)(current_weight_pack & extraction_mask);
                    current_weight_pack >>= QUANT_BITS;
                    float restored_weight = current_scale * weight_value + current_zero;
                    accumulators[pos] += restored_weight * current_input;
                }
            }
        }
    }

    // Finalize results with warp-level reduction and output
    for (int pos = 0; pos < ELEMENTS_PER_PACK; pos++) {
        int output_index = output_start + pos;
        if (output_index < output_size) {
            // Inline warp-level reduction
            float reduced_value = accumulators[pos];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                reduced_value += __shfl_down_sync(0xffffffff, reduced_value, offset);
            }
            if (threadIdx.x == 0) {
                output_base[output_index] = __float2half(reduced_value);
            }
        }
    }
}


torch::Tensor quantized_gemm(
    int quantization_group_size,
    torch::Tensor input_matrix,       // (B, nh, M, K) float16
    torch::Tensor quantized_weights,  // (B, nh, K, N // feat_per_int) int32
    torch::Tensor quantization_scales,// (B, nh, K, G) float16
    torch::Tensor quantization_zeros, // (B, nh, K, G) float16
    int quantization_bits
) {
    // Check input tensor dimensions
    TORCH_CHECK(input_matrix.dim() == 4 && quantized_weights.dim() == 4, 
                "Input tensors must be 4-dimensional");
    
    // Extract dimensions from input matrix
    int batch_size = input_matrix.size(0);
    int num_heads = input_matrix.size(1);
    int seq_length = input_matrix.size(2);
    int input_dim = input_matrix.size(3);
    
    // Extract KV heads from quantized weights
    int kv_heads = quantized_weights.size(1);
    int elements_per_int = 32 / quantization_bits;
    int output_dim = quantized_weights.size(3) * elements_per_int;

    // Reshape input matrix to 3D for processing
    input_matrix = input_matrix.view({-1, seq_length, input_dim}).contiguous();
    
    // Reshape and transpose quantized weights
    quantized_weights = quantized_weights.view({-1, input_dim, quantized_weights.size(3)})
                                         .transpose(1, 2)
                                         .contiguous();
    
    // Calculate flattened batch size
    int flattened_batch = batch_size * kv_heads;
    
    // Reshape and transpose scales and zeros
    quantization_scales = quantization_scales.view({flattened_batch, 
                                                    quantization_scales.size(-2), 
                                                    quantization_scales.size(-1)})
                                             .transpose(1, 2)
                                             .contiguous();
    quantization_zeros = quantization_zeros.view({flattened_batch, 
                                                  quantization_zeros.size(-2), 
                                                  quantization_zeros.size(-1)})
                                           .transpose(1, 2)
                                           .contiguous();

    // Validate quantization bits and head divisibility
    TORCH_CHECK(quantization_bits == 1 || quantization_bits == 2 || 
                quantization_bits == 3 || quantization_bits == 4, 
                "Supported quantization bits: 1, 2, 3, 4");
    TORCH_CHECK(num_heads % kv_heads == 0, 
                "Total heads must be divisible by KV heads");

    // Retrieve tensor dimensions for kernel launch
    int effective_batch_size = input_matrix.size(0);
    int num_features = input_matrix.size(1);
    int input_size = input_matrix.size(2);
    int output_size = quantization_zeros.size(1) * quantization_group_size;

    // Access tensor data pointers
    auto input_data = reinterpret_cast<half*>(input_matrix.data_ptr<at::Half>());
    auto weight_data = reinterpret_cast<uint32_t*>(quantized_weights.data_ptr<int>());
    auto zeros_data = reinterpret_cast<half*>(quantization_zeros.data_ptr<at::Half>());
    auto scales_data = reinterpret_cast<half*>(quantization_scales.data_ptr<at::Half>());

    // Create output tensor
    auto output_options = torch::TensorOptions()
                            .dtype(input_matrix.dtype())
                            .device(input_matrix.device());
    at::Tensor result_matrix = torch::empty({effective_batch_size, num_features, output_size}, 
                                            output_options);
    auto result_data = reinterpret_cast<half*>(result_matrix.data_ptr<at::Half>());

    // Determine packing factor based on quantization bits
    int packing_factor = (quantization_bits == 3) ? 11 : 32 / quantization_bits;
    if (quantization_bits == 3) {
        TORCH_CHECK(quantization_group_size == 11, 
                    "For 3-bit quantization, group size must be 11");
    }

    // Set up kernel launch parameters
    dim3 grid_config(effective_batch_size, 
                     (output_size / packing_factor + 3) / 4, 
                     num_features);
    dim3 block_config(32, 4);

    // Launch appropriate kernel based on quantization bits
    switch (quantization_bits) {
        case 4:
            execute_quantized_matvec<4, 8><<<grid_config, block_config>>>(
                input_data, weight_data, zeros_data, scales_data, result_data,
                input_size, output_size, quantization_group_size, num_heads, kv_heads
            );
            break;
        case 2:
            execute_quantized_matvec<2, 16><<<grid_config, block_config>>>(
                input_data, weight_data, zeros_data, scales_data, result_data,
                input_size, output_size, quantization_group_size, num_heads, kv_heads
            );
            break;
        case 1:
            execute_quantized_matvec<1, 32><<<grid_config, block_config>>>(
                input_data, weight_data, zeros_data, scales_data, result_data,
                input_size, output_size, quantization_group_size, num_heads, kv_heads
            );
            break;
        case 3:
            execute_quantized_matvec<3, 11><<<grid_config, block_config>>>(
                input_data, weight_data, zeros_data, scales_data, result_data,
                input_size, output_size, quantization_group_size, num_heads, kv_heads
            );
            break;
        default:
            TORCH_CHECK(false, "Quantization bits not supported: ", quantization_bits);
    }

    // Reshape result back to original dimensions
    result_matrix = result_matrix.view({batch_size, num_heads, 
                                        result_matrix.size(-2), 
                                        result_matrix.size(-1)});
    return result_matrix;
}
// // Launches the quantized matrix-vector multiplication kernel
// torch::Tensor launch_quantized_gemv(
//     torch::Tensor input_features,
//     torch::Tensor weight_matrix,
//     torch::Tensor scale_values,
//     torch::Tensor zero_values,
//     const int quantization_bits,
//     const int group_size,
//     const int head_count,
//     const int kv_head_count
// ) {
//     // Retrieve tensor dimensions
//     int batch_size = input_features.size(0);
//     int feature_count = input_features.size(1);
//     int input_dimension = input_features.size(2);
//     int output_dimension = zero_values.size(1) * group_size;

//     // Access tensor data
//     auto inputs = reinterpret_cast<half*>(input_features.data_ptr<at::Half>());
//     auto weights = reinterpret_cast<uint32_t*>(weight_matrix.data_ptr<int>());
//     auto zeros = reinterpret_cast<half*>(zero_values.data_ptr<at::Half>());
//     auto scales = reinterpret_cast<half*>(scale_values.data_ptr<at::Half>());

//     // Allocate output tensor
//     auto tensor_options = torch::TensorOptions()
//                             .dtype(input_features.dtype())
//                             .device(input_features.device());
//     at::Tensor output_features = torch::empty({batch_size, feature_count, output_dimension}, tensor_options);
//     auto outputs = reinterpret_cast<half*>(output_features.data_ptr<at::Half>());

//     // Determine packing factor based on quantization bits
//     int pack_factor = (quantization_bits == 3) ? 11 : 32 / quantization_bits;
//     if (quantization_bits == 3) {
//         TORCH_CHECK(group_size == 11, "For 3-bit quantization, group size must be 11");
//     }

//     // Configure execution parameters
//     dim3 block_config(batch_size, (output_dimension / pack_factor + 3) / 4, feature_count);
//     dim3 thread_config(32, 4);

//     // Dispatch kernel based on quantization bits
//     switch (quantization_bits) {
//         case 4:
//             execute_quantized_matvec<4, 8><<<block_config, thread_config>>>(
//                 inputs, weights, zeros, scales, outputs,
//                 input_dimension, output_dimension, group_size, head_count, kv_head_count
//             );
//             break;
//         case 2:
//             execute_quantized_matvec<2, 16><<<block_config, thread_config>>>(
//                 inputs, weights, zeros, scales, outputs,
//                 input_dimension, output_dimension, group_size, head_count, kv_head_count
//             );
//             break;
//         case 1:
//             execute_quantized_matvec<1, 32><<<block_config, thread_config>>>(
//                 inputs, weights, zeros, scales, outputs,
//                 input_dimension, output_dimension, group_size, head_count, kv_head_count
//             );
//             break;
//         case 3:
//             execute_quantized_matvec<3, 11><<<block_config, thread_config>>>(
//                 inputs, weights, zeros, scales, outputs,
//                 input_dimension, output_dimension, group_size, head_count, kv_head_count
//             );
//             break;
//         default:
//             TORCH_CHECK(false, "Quantization bits not supported: ", quantization_bits);
//     }

//     return output_features;
// }

// // Python binding for quantized batch matrix multiplication
// torch::Tensor cuda_bmm_fA_qB_outer(
//     int group_size,
//     torch::Tensor fA,       // (B, nh, M, K) float16
//     torch::Tensor qB,       // (B, nh, K, N // feat_per_int) int32
//     torch::Tensor scales,   // (B, nh, K, G) float16
//     torch::Tensor zeros,    // (B, nh, K, G) float16
//     int bits
// ) {
//     // Validate input shapes
//     TORCH_CHECK(fA.dim() == 4 && qB.dim() == 4, "Input tensors must be 4-dimensional");
//     int B = fA.size(0);
//     int nh = fA.size(1);
//     int M = fA.size(2);
//     int K = fA.size(3);
//     int nh_kv = qB.size(1);
//     int feat_per_int = 32 / bits;
//     int N = qB.size(3) * feat_per_int;

//     // Flatten and transpose tensors for processing
//     fA = fA.view({-1, M, K}).contiguous();
//     qB = qB.view({-1, K, qB.size(3)}).transpose(1, 2).contiguous();
//     int flatten_B = B * nh_kv;
//     scales = scales.view({flatten_B, scales.size(-2), scales.size(-1)}).transpose(1, 2).contiguous();
//     zeros = zeros.view({flatten_B, zeros.size(-2), zeros.size(-1)}).transpose(1, 2).contiguous();

//     // Ensure valid bit width and head divisibility
//     TORCH_CHECK(bits == 1 || bits == 2 || bits == 3 || bits == 4, "Supported bits: 1, 2, 3, 4");
//     TORCH_CHECK(nh % nh_kv == 0, "Total heads must be divisible by KV heads");

//     // Execute the quantized GEMV operation
//     torch::Tensor c = launch_quantized_gemv(fA, qB, scales, zeros, bits, group_size, nh, nh_kv);

//     // Reshape the result back to (B, nh, M, N)
//     c = c.view({B, nh, c.size(-2), c.size(-1)});
//     return c;
// }
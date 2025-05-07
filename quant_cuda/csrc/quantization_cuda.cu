#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <tuple>
#include <cfloat>
#include "quantization_cuda.h"


// Kernel to calculate scale and min for key data
__global__ void calculate_scale_min_key_kernel(
    const float* __restrict__ input_data,
    const float* __restrict__ prev_scale,
    const float* __restrict__ prev_min,
    float* __restrict__ combined_scale,
    float* __restrict__ combined_min,
    int num_rows,
    int new_time_steps,
    int old_group_count,
    int group_length,
    int quant_bits
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y;

    int new_group_count = new_time_steps / group_length;
    int total_groups = old_group_count + new_group_count;

    if (row < num_rows && group < total_groups) {
        if (group < old_group_count) {
            combined_scale[row * total_groups + group] = prev_scale[row * old_group_count + group];
            combined_min[row * total_groups + group] = prev_min[row * old_group_count + group];
        } else {
            int new_group = group - old_group_count;
            int offset = row * new_time_steps + new_group * group_length;
            float min_value = FLT_MAX;
            float max_value = -FLT_MAX;

            for (int i = 0; i < group_length; i++) {
                int index = offset + i;
                if (index < row * new_time_steps + new_time_steps) {
                    float value = input_data[index];
                    min_value = fminf(min_value, value);
                    max_value = fmaxf(max_value, value);
                }
            }

            float quant_max = (1 << quant_bits) - 1;
            float scale = (max_value - min_value) / quant_max;
            combined_scale[row * total_groups + group] = scale;
            combined_min[row * total_groups + group] = min_value;
        }
    }
}

// Kernel to quantize and pack key data
__global__ void quant_pack_key_kernel(
    const float* __restrict__ input_data,
    const int* __restrict__ prev_code,
    const float* __restrict__ combined_scale,
    const float* __restrict__ combined_min,
    int* __restrict__ combined_code,
    int num_rows,
    int new_time_steps,
    int old_int_count,
    int old_group_count,
    int group_length,
    int quant_bits,
    int features_per_int
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int int_index = blockIdx.y;

    int new_int_count = new_time_steps / features_per_int;
    int total_ints = old_int_count + new_int_count;
    int total_groups = old_group_count + new_time_steps / group_length;

    if (row < num_rows && int_index < total_ints) {
        if (int_index < old_int_count) {
            combined_code[row * total_ints + int_index] = prev_code[row * old_int_count + int_index];
        } else {
            int new_int = int_index - old_int_count;
            int offset = row * new_time_steps + new_int * features_per_int;
            int packed_value = 0;

            float quant_max = (1 << quant_bits) - 1;
            int group_start = (new_int * features_per_int) / group_length;
            float scale = combined_scale[row * total_groups + old_group_count + group_start];
            float min_value = combined_min[row * total_groups + old_group_count + group_start];

            for (int i = 0; i < features_per_int; i++) {
                int index = offset + i;
                if (index < row * new_time_steps + new_time_steps) {
                    float value = input_data[index];
                    float normalized = (value - min_value) / scale;
                    int quantized = roundf(normalized);
                    quantized = max(0, min(quantized, static_cast<int>(quant_max)));
                    packed_value |= (quantized << (i * quant_bits));
                }
            }
            combined_code[row * total_ints + int_index] = packed_value;
        }
    }
}

// Kernel for 3-bit quantization and packing of key data
__global__ void calc_scale_quant_pack_key_3bit_kernel(
    const float* __restrict__ input_data,
    const int* __restrict__ prev_code,
    const float* __restrict__ prev_scale,
    const float* __restrict__ prev_min,
    int* __restrict__ combined_code,
    float* __restrict__ combined_scale,
    float* __restrict__ combined_min,
    int num_rows,
    int new_time_steps,
    int old_int_count,
    int old_group_count,
    int group_length,
    int quant_bits
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y;

    int new_group_count = new_time_steps / group_length;
    int total_groups = old_group_count + new_group_count;

    if (row >= num_rows || group >= total_groups) return;

    if (group < old_group_count) {
        combined_scale[row * total_groups + group] = prev_scale[row * old_group_count + group];
        combined_min[row * total_groups + group] = prev_min[row * old_group_count + group];
    } else {
        int new_group = group - old_group_count;
        int offset = row * new_time_steps + new_group * group_length;

        float min_value = FLT_MAX;
        float max_value = -FLT_MAX;
        for (int i = 0; i < group_length; i++) {
            int index = offset + i;
            if (index < row * new_time_steps + new_time_steps) {
                float value = input_data[index];
                min_value = fminf(min_value, value);
                max_value = fmaxf(max_value, value);
            }
        }

        float quant_max = 7.0f;  // Fixed for 3-bit
        float scale = (max_value - min_value) / quant_max;
        combined_scale[row * total_groups + group] = scale;
        combined_min[row * total_groups + group] = min_value;

        int quant_values[11];
        for (int i = 0; i < 10; i++) {
            int index = offset + i;
            if (index < row * new_time_steps + new_time_steps) {
                float value = input_data[index];
                float normalized = (value - min_value) / scale;
                int quant_value = roundf(normalized);
                quant_values[i] = max(0, min(quant_value, 7));
            } else {
                quant_values[i] = 0;
            }
        }
        int index = offset + 10;
        if (index < row * new_time_steps + new_time_steps) {
            float value = input_data[index];
            float normalized = (value - min_value) / scale;
            int quant_value = roundf(normalized);
            quant_values[10] = max(0, min(quant_value, 3));
        } else {
            quant_values[10] = 0;
        }

        int packed_value = 0;
        for (int i = 0; i < 10; i++) {
            packed_value |= (quant_values[i] << (i * 3));
        }
        packed_value |= (quant_values[10] << 30);

        int int_index = old_int_count + new_group;
        combined_code[row * (old_int_count + new_group_count) + int_index] = packed_value;
    }
}

// Kernel to calculate scale and min for value data
__global__ void calculate_scale_min_value_kernel(
    const float* __restrict__ input_data,
    const float* __restrict__ prev_scale,
    const float* __restrict__ prev_min,
    float* __restrict__ combined_scale,
    float* __restrict__ combined_min,
    int num_rows,
    int new_time_steps,
    int head_dimension,
    int old_time_steps,
    int group_length,
    int quant_bits
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int time = blockIdx.y;

    int total_time_steps = old_time_steps + new_time_steps;
    int groups_per_time = head_dimension / group_length;

    if (row < num_rows && time < total_time_steps) {
        if (time < old_time_steps) {
            for (int g = 0; g < groups_per_time; g++) {
                combined_scale[row * total_time_steps * groups_per_time + time * groups_per_time + g] =
                    prev_scale[row * old_time_steps * groups_per_time + time * groups_per_time + g];
                combined_min[row * total_time_steps * groups_per_time + time * groups_per_time + g] =
                    prev_min[row * old_time_steps * groups_per_time + time * groups_per_time + g];
            }
        } else {
            int new_time = time - old_time_steps;
            int offset = row * new_time_steps * head_dimension + new_time * head_dimension;
            float quant_max = (1 << quant_bits) - 1;

            for (int g = 0; g < groups_per_time; g++) {
                int group_offset = offset + g * group_length;
                float min_value = FLT_MAX;
                float max_value = -FLT_MAX;

                for (int i = 0; i < group_length; i++) {
                    int index = group_offset + i;
                    if (index < (row + 1) * new_time_steps * head_dimension) {
                        float value = input_data[index];
                        min_value = fminf(min_value, value);
                        max_value = fmaxf(max_value, value);
                    }
                }

                float scale = (max_value - min_value) / quant_max;
                combined_scale[row * total_time_steps * groups_per_time + time * groups_per_time + g] = scale;
                combined_min[row * total_time_steps * groups_per_time + time * groups_per_time + g] = min_value;
            }
        }
    }
}

// Kernel to quantize and pack value data
__global__ void quant_pack_value_kernel(
    const float* __restrict__ input_data,
    const int* __restrict__ prev_code,
    const float* __restrict__ combined_scale,
    const float* __restrict__ combined_min,
    int* __restrict__ combined_code,
    int num_rows,
    int new_time_steps,
    int head_dimension,
    int old_time_steps,
    int group_length,
    int quant_bits,
    int features_per_int
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int time = blockIdx.y;

    int total_time_steps = old_time_steps + new_time_steps;
    int packed_head_dim = head_dimension / features_per_int;
    int groups_per_time = head_dimension / group_length;

    if (row < num_rows && time < total_time_steps) {
        if (time < old_time_steps) {
            for (int i = 0; i < packed_head_dim; i++) {
                combined_code[row * total_time_steps * packed_head_dim + time * packed_head_dim + i] =
                    prev_code[row * old_time_steps * packed_head_dim + time * packed_head_dim + i];
            }
        } else {
            int new_time = time - old_time_steps;
            int offset = row * new_time_steps * head_dimension + new_time * head_dimension;
            float quant_max = (1 << quant_bits) - 1;

            for (int i = 0; i < packed_head_dim; i++) {
                int packed_value = 0;
                int feat_offset = offset + i * features_per_int;
                int group_start = (i * features_per_int) / group_length;
                float scale = combined_scale[row * total_time_steps * groups_per_time + time * groups_per_time + group_start];
                float min_value = combined_min[row * total_time_steps * groups_per_time + time * groups_per_time + group_start];

                for (int j = 0; j < features_per_int; j++) {
                    int index = feat_offset + j;
                    if (index < (row + 1) * new_time_steps * head_dimension) {
                        float value = input_data[index];
                        float normalized = (value - min_value) / scale;
                        int quantized = roundf(normalized);
                        quantized = max(0, min(quantized, static_cast<int>(quant_max)));
                        packed_value |= (quantized << (j * quant_bits));
                    }
                }
                combined_code[row * total_time_steps * packed_head_dim + time * packed_head_dim + i] = packed_value;
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
process_key_quantization_cuda(
    torch::Tensor input_tensor,
    torch::Tensor previous_code,
    torch::Tensor previous_scale,
    torch::Tensor previous_min,
    int group_length,
    int quant_bits
) {
    bool is_half_precision = (input_tensor.scalar_type() == torch::kHalf);
    if (is_half_precision) {
        input_tensor = input_tensor.to(torch::kFloat32);
        previous_scale = previous_scale.to(torch::kFloat32);
        previous_min = previous_min.to(torch::kFloat32);
    }

    auto dimensions = input_tensor.sizes();
    int batch_size = dimensions[0], num_heads = dimensions[1], depth = dimensions[2], new_steps = dimensions[3];

    if (quant_bits == 3) {
        TORCH_CHECK(group_length == 11, "For 3-bit quantization, group_length must be 11");
        TORCH_CHECK(new_steps % 11 == 0, "new_steps must be divisible by 11 for 3-bit quantization");
    } else {
        TORCH_CHECK(new_steps % group_length == 0, "new_steps must be divisible by group_length");
    }

    int new_group_count = (quant_bits == 3) ? (new_steps / 11) : (new_steps / group_length);
    int old_group_count = previous_scale.size(3);
    int features_per_int = (quant_bits == 3) ? 11 : (32 / quant_bits);
    int new_int_count = new_steps / features_per_int;
    int old_int_count = previous_code.size(3);

    int total_rows = batch_size * num_heads * depth;

    torch::Tensor reshaped_input = input_tensor.reshape({total_rows, new_steps}).contiguous();
    torch::Tensor reshaped_prev_code = previous_code.reshape({total_rows, old_int_count}).contiguous();
    torch::Tensor reshaped_prev_scale = previous_scale.reshape({total_rows, old_group_count}).contiguous();
    torch::Tensor reshaped_prev_min = previous_min.reshape({total_rows, old_group_count}).contiguous();

    auto float_opts = torch::TensorOptions().device(input_tensor.device()).dtype(input_tensor.dtype());
    auto int_opts = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt32);
    torch::Tensor output_code = torch::empty({total_rows, old_int_count + new_int_count}, int_opts);
    torch::Tensor output_scale = torch::empty({total_rows, old_group_count + new_group_count}, float_opts);
    torch::Tensor output_min = torch::empty({total_rows, old_group_count + new_group_count}, float_opts);

    const int BLOCK_SIZE = 128;
    int grid_x = (total_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (quant_bits == 3) {
        dim3 grid(grid_x, old_group_count + new_group_count);
        calc_scale_quant_pack_key_3bit_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reshaped_input.data_ptr<float>(),
            reshaped_prev_code.data_ptr<int>(),
            reshaped_prev_scale.data_ptr<float>(),
            reshaped_prev_min.data_ptr<float>(),
            output_code.data_ptr<int>(),
            output_scale.data_ptr<float>(),
            output_min.data_ptr<float>(),
            total_rows,
            new_steps,
            old_int_count,
            old_group_count,
            11,
            3
        );
    } else {
        dim3 grid_scale(grid_x, old_group_count + new_group_count);
        calculate_scale_min_key_kernel<<<grid_scale, BLOCK_SIZE, 0, stream>>>(
            reshaped_input.data_ptr<float>(),
            reshaped_prev_scale.data_ptr<float>(),
            reshaped_prev_min.data_ptr<float>(),
            output_scale.data_ptr<float>(),
            output_min.data_ptr<float>(),
            total_rows,
            new_steps,
            old_group_count,
            group_length,
            quant_bits
        );

        dim3 grid_code(grid_x, old_int_count + new_int_count);
        quant_pack_key_kernel<<<grid_code, BLOCK_SIZE, 0, stream>>>(
            reshaped_input.data_ptr<float>(),
            reshaped_prev_code.data_ptr<int>(),
            output_scale.data_ptr<float>(),
            output_min.data_ptr<float>(),
            output_code.data_ptr<int>(),
            total_rows,
            new_steps,
            old_int_count,
            old_group_count,
            group_length,
            quant_bits,
            features_per_int
        );
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    output_code = output_code.view({batch_size, num_heads, depth, old_int_count + new_int_count});
    output_scale = output_scale.view({batch_size, num_heads, depth, old_group_count + new_group_count});
    output_min = output_min.view({batch_size, num_heads, depth, old_group_count + new_group_count});

    if (is_half_precision) {
        output_scale = output_scale.to(torch::kHalf);
        output_min = output_min.to(torch::kHalf);
    }

    return std::make_tuple(output_code, output_scale, output_min);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
process_value_quantization_cuda(
    torch::Tensor input_tensor,
    torch::Tensor previous_code,
    torch::Tensor previous_scale,
    torch::Tensor previous_min,
    int group_length,
    int quant_bits
) {
    bool is_half_precision = (input_tensor.scalar_type() == torch::kHalf);
    if (is_half_precision) {
        input_tensor = input_tensor.to(torch::kFloat32);
        previous_scale = previous_scale.to(torch::kFloat32);
        previous_min = previous_min.to(torch::kFloat32);
    }

    auto dimensions = input_tensor.sizes();
    int batch_size = dimensions[0], num_heads = dimensions[1], new_steps = dimensions[2], head_dim = dimensions[3];

    if (quant_bits == 3) {
        TORCH_CHECK(group_length == 11, "For 3-bit quantization, group_length must be 11");
        TORCH_CHECK(head_dim % 11 == 0, "head_dim must be divisible by 11 for 3-bit quantization");
    } else {
        TORCH_CHECK(head_dim % group_length == 0, "head_dim must be divisible by group_length");
    }

    int old_steps = previous_code.size(2);
    int features_per_int = (quant_bits == 3) ? 11 : (32 / quant_bits);
    int packed_head_dim = head_dim / features_per_int;
    int groups_per_time = head_dim / group_length;

    int total_rows = batch_size * num_heads;

    torch::Tensor reshaped_input = input_tensor.reshape({total_rows, new_steps * head_dim}).contiguous();
    torch::Tensor reshaped_prev_code = previous_code.reshape({total_rows, old_steps * packed_head_dim}).contiguous();
    torch::Tensor reshaped_prev_scale = previous_scale.reshape({total_rows, old_steps * groups_per_time}).contiguous();
    torch::Tensor reshaped_prev_min = previous_min.reshape({total_rows, old_steps * groups_per_time}).contiguous();

    auto float_opts = torch::TensorOptions().device(input_tensor.device()).dtype(input_tensor.dtype());
    auto int_opts = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt32);
    torch::Tensor output_code = torch::empty({total_rows, (old_steps + new_steps) * packed_head_dim}, int_opts);
    torch::Tensor output_scale = torch::empty({total_rows, (old_steps + new_steps) * groups_per_time}, float_opts);
    torch::Tensor output_min = torch::empty({total_rows, (old_steps + new_steps) * groups_per_time}, float_opts);

    const int BLOCK_SIZE = 128;
    int grid_x = (total_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(grid_x, old_steps + new_steps);
    if (quant_bits == 3) {
        // Note: Original code references a missing 3-bit value kernel; using placeholder logic
        calculate_scale_min_value_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reshaped_input.data_ptr<float>(),
            reshaped_prev_scale.data_ptr<float>(),
            reshaped_prev_min.data_ptr<float>(),
            output_scale.data_ptr<float>(),
            output_min.data_ptr<float>(),
            total_rows,
            new_steps,
            head_dim,
            old_steps,
            11,
            3
        );
        quant_pack_value_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reshaped_input.data_ptr<float>(),
            reshaped_prev_code.data_ptr<int>(),
            output_scale.data_ptr<float>(),
            output_min.data_ptr<float>(),
            output_code.data_ptr<int>(),
            total_rows,
            new_steps,
            head_dim,
            old_steps,
            11,
            3,
            11
        );
    } else {
        calculate_scale_min_value_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reshaped_input.data_ptr<float>(),
            reshaped_prev_scale.data_ptr<float>(),
            reshaped_prev_min.data_ptr<float>(),
            output_scale.data_ptr<float>(),
            output_min.data_ptr<float>(),
            total_rows,
            new_steps,
            head_dim,
            old_steps,
            group_length,
            quant_bits
        );

        quant_pack_value_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            reshaped_input.data_ptr<float>(),
            reshaped_prev_code.data_ptr<int>(),
            output_scale.data_ptr<float>(),
            output_min.data_ptr<float>(),
            output_code.data_ptr<int>(),
            total_rows,
            new_steps,
            head_dim,
            old_steps,
            group_length,
            quant_bits,
            features_per_int
        );
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    output_code = output_code.view({batch_size, num_heads, old_steps + new_steps, packed_head_dim});
    output_scale = output_scale.view({batch_size, num_heads, old_steps + new_steps, groups_per_time});
    output_min = output_min.view({batch_size, num_heads, old_steps + new_steps, groups_per_time});

    if (is_half_precision) {
        output_scale = output_scale.to(torch::kHalf);
        output_min = output_min.to(torch::kHalf);
    }

    return std::make_tuple(output_code, output_scale, output_min);
}

// Kernel to compute min/max and quantize data
__global__ void min_max_quantize_kernel(
    const float* __restrict__ source_data,
    int* __restrict__ quantized_output,
    float* __restrict__ min_output,
    float* __restrict__ max_output,
    int element_count,
    int num_rows,
    int group_count,
    int group_length,
    int quant_bits
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = num_rows * group_count;
    if (group_idx < total_groups) {
        int offset = group_idx * group_length;
        float min_value = FLT_MAX;
        float max_value = -FLT_MAX;

        for (int i = 0; i < group_length; i++) {
            int idx = offset + i;
            if (idx < element_count) {
                float value = source_data[idx];
                min_value = fminf(min_value, value);
                max_value = fmaxf(max_value, value);
            }
        }

        min_output[group_idx] = min_value;
        max_output[group_idx] = max_value;

        float quant_max = (1 << quant_bits) - 1;
        float scale = (max_value - min_value) / quant_max;

        if (quant_bits == 3) {
            for (int i = 0; i < group_length; i++) {
                int idx = offset + i;
                if (idx < element_count) {
                    float value = source_data[idx];
                    float normalized = (value - min_value) / scale;
                    int quant_value = roundf(normalized);
                    int elem_max = (i < 10) ? 7 : 3;
                    quant_value = max(0, min(quant_value, elem_max));
                    quantized_output[idx] = quant_value;
                }
            }
        } else {
            for (int i = 0; i < group_length; i++) {
                int idx = offset + i;
                if (idx < element_count) {
                    float value = source_data[idx];
                    float normalized = (value - min_value) / scale;
                    int quant_value = roundf(normalized);
                    quant_value = max(0, min(quant_value, static_cast<int>(quant_max)));
                    quantized_output[idx] = quant_value;
                }
            }
        }
    }
}

// Kernel to pack quantized data
__global__ void pack_data_kernel(
    int quant_bits,
    const int* __restrict__ quantized_input,
    int* __restrict__ packed_output,
    int num_rows,
    int feature_count,
    int features_per_int,
    int int_count_per_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y;
    if (row < num_rows) {
        int out_idx = row * int_count_per_dim + group;
        int in_start = row * feature_count + group * features_per_int;
        int packed_value = 0;

        if (quant_bits == 3) {
            for (int i = 0; i < 10; ++i) {
                packed_value |= (quantized_input[in_start + i] << (i * 3));
            }
            packed_value |= (quantized_input[in_start + 10] << 30);
        } else {
            for (int i = 0; i < features_per_int; i++) {
                packed_value |= (quantized_input[in_start + i] << (i * quant_bits));
            }
        }
        packed_output[out_idx] = packed_value;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quantize_pack_last_dim_cuda(
    torch::Tensor input_tensor,
    int group_length,
    int quant_bits
) {
    if (quant_bits == 3) {
        TORCH_CHECK(group_length == 11, "For 3-bit quantization, group_length must be 11");
    }

    bool is_half_precision = (input_tensor.scalar_type() == torch::kHalf);
    if (is_half_precision) input_tensor = input_tensor.to(torch::kFloat32);

    TORCH_CHECK(input_tensor.dim() == 4, "input_tensor must be a 4D tensor");
    auto dimensions = input_tensor.sizes();
    int batch_size = dimensions[0], num_heads = dimensions[1], depth = dimensions[2], steps = dimensions[3];

    if (quant_bits == 3) {
        TORCH_CHECK(steps % 11 == 0, "For 3-bit quantization, steps must be divisible by 11");
        group_length = 11;
    } else {
        TORCH_CHECK(steps % group_length == 0, "steps must be divisible by group_length");
    }

    int group_count = steps / group_length;
    int total_rows = batch_size * num_heads * depth;

    torch::Tensor reshaped_input = input_tensor.reshape({total_rows, steps}).contiguous();
    auto opts = torch::TensorOptions().device(input_tensor.device()).dtype(input_tensor.dtype());
    torch::Tensor min_vals = torch::empty({total_rows, group_count}, opts);
    torch::Tensor max_vals = torch::empty({total_rows, group_count}, opts);
    torch::Tensor quantized_data = torch::empty({total_rows, steps}, torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt32));

    const int BLOCK_SIZE = 128;
    int total_groups = total_rows * group_count;
    int grid_size = (total_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    min_max_quantize_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reshaped_input.data_ptr<float>(),
        quantized_data.data_ptr<int>(),
        min_vals.data_ptr<float>(),
        max_vals.data_ptr<float>(),
        reshaped_input.numel(),
        total_rows,
        group_count,
        group_length,
        quant_bits
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    int features_per_int = (quant_bits == 3) ? 11 : (32 / quant_bits);
    if (quant_bits != 3) {
        TORCH_CHECK(steps % features_per_int == 0, "steps must be divisible by (32 / quant_bits)");
    }
    int int_count_per_dim = steps / features_per_int;

    torch::Tensor packed_code = torch::zeros({total_rows, int_count_per_dim}, torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt32));

    int grid_x = (total_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_x, int_count_per_dim);
    pack_data_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        quant_bits,
        quantized_data.data_ptr<int>(),
        packed_code.data_ptr<int>(),
        total_rows,
        steps,
        features_per_int,
        int_count_per_dim
    );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    packed_code = packed_code.view({batch_size, num_heads, depth, int_count_per_dim});
    torch::Tensor scale = (max_vals - min_vals) / ((1 << quant_bits) - 1);
    scale = scale.view({batch_size, num_heads, depth, group_count});
    min_vals = min_vals.view({batch_size, num_heads, depth, group_count});

    if (is_half_precision) {
        scale = scale.to(torch::kHalf);
        min_vals = min_vals.to(torch::kHalf);
    }

    return std::make_tuple(packed_code, scale, min_vals);
}
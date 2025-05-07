#pragma once
#include <torch/extension.h>

// torch::Tensor gemv_forward_cuda(
//     torch::Tensor _in_feats,
//     torch::Tensor _kernel,
//     torch::Tensor _scaling_factors,
//     torch::Tensor _zeros,
//     const int bit,
//     const int group_size);


// torch::Tensor gemv_forward_cuda_outer_dim(
//     torch::Tensor _in_feats,
//     torch::Tensor _kernel,
//     torch::Tensor _scaling_factors,
//     torch::Tensor _zeros,
//     const int bit,
//     const int group_size,
//     const int nh,
//     const int nh_kv);
torch::Tensor quantized_gemm(
    int group_size,
    torch::Tensor fA,       // (B, nh, M, K) float16
    torch::Tensor qB,       // (B, nh, K, N // feat_per_int) int32
    torch::Tensor scales,   // (B, nh, K, G) float16
    torch::Tensor zeros,    // (B, nh, K, G) float16
    int bits
);
// torch::Tensor dequantized_matmul(
//     int group_size,
//     torch::Tensor input_float,
//     torch::Tensor weights_quantized,
//     torch::Tensor scaling_factors,
//     torch::Tensor zero_points,
//     int quantization_bits);
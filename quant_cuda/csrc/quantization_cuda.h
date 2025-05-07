#pragma once
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
process_key_quantization_cuda(
    torch::Tensor input_tensor,
    torch::Tensor previous_code,
    torch::Tensor previous_scale,
    torch::Tensor previous_min,
    int group_length,
    int quant_bits
);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
process_value_quantization_cuda(
    torch::Tensor input_tensor,
    torch::Tensor previous_code,
    torch::Tensor previous_scale,
    torch::Tensor previous_min,
    int group_length,
    int quant_bits
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quantize_pack_last_dim_cuda(
    torch::Tensor input_tensor,
    int group_length,
    int quant_bits
); 

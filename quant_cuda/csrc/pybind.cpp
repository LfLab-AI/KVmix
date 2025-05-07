#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "dequantization_gemv_cuda.h"
#include "quantization_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("quantized_gemm", &quantized_gemm);
  m.def("quantize_pack_last_dim_cuda", &quantize_pack_last_dim_cuda,
    "Quantize and pack along last dim (CUDA)");
  m.def("process_value_quantization_cuda", &process_value_quantization_cuda,
      "Quantize and pack along last dim (CUDA)value");
  m.def("process_key_quantization_cuda", &process_key_quantization_cuda,
      "Quantize and pack along last dim (CUDA)key");
}


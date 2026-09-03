// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include <torch/extension.h>

void adamw_bf16_sr_cuda(
    int64_t chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    int64_t adam_step,
    uint64_t stochastic_step,
    double weight_decay,
    uint64_t seed);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "adamw",
      &adamw_bf16_sr_cuda,
      "Fused BF16 AdamW with stateless stochastic parameter writeback");
}

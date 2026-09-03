// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "multi_tensor_apply.cuh"

#ifndef SR_ADAMW_HASH32
#error "The authenticated AdamWBF16SR build requires SR_ADAMW_HASH32=1"
#endif

namespace {

constexpr int kBlockSize = 512;
constexpr int kInstructionLevelParallelism = 4;
constexpr uint64_t kMaximumTensorElements = UINT64_C(0xffffffff);

__device__ __forceinline__ uint32_t avalanche32(uint32_t value) {
  value ^= value >> 16;
  value *= UINT32_C(0x7feb352d);
  value ^= value >> 15;
  value *= UINT32_C(0x846ca68b);
  return value ^ (value >> 16);
}

__device__ __forceinline__ uint32_t stateless_random16(
    uint64_t seed,
    uint64_t step,
    uint64_t tensor_index,
    uint64_t element_index) {
  // The odd element multiplier makes the pre-avalanche mapping bijective for
  // every supported tensor. Tensor, step, and both seed halves select a stable
  // permutation offset without consulting any ambient CUDA RNG state.
  uint32_t counter = static_cast<uint32_t>(element_index) *
      UINT32_C(0x9e3779b1);
  counter ^= static_cast<uint32_t>(tensor_index) * UINT32_C(0x85ebca77);
  counter ^= static_cast<uint32_t>(step) * UINT32_C(0xc2b2ae3d);
  counter ^= static_cast<uint32_t>(step >> 32) * UINT32_C(0x27d4eb2f);
  counter ^= static_cast<uint32_t>(seed);
  counter ^= static_cast<uint32_t>(seed >> 32) * UINT32_C(0x165667b1);
  return avalanche32(counter) >> 16;
}

__device__ __forceinline__ c10::BFloat16 stochastic_bf16(
    float value,
    uint64_t seed,
    uint64_t step,
    uint64_t tensor_index,
    uint64_t element_index) {
  union {
    float floating;
    uint32_t bits;
  } representation{value};

  // Preserve infinities and canonicalize NaNs using PyTorch's deterministic
  // conversion. TorchAO's generic SR helper instead turns these into -0.
  if ((representation.bits & UINT32_C(0x7f800000)) ==
      UINT32_C(0x7f800000)) {
    return c10::BFloat16(value);
  }

  const uint32_t remainder = representation.bits & UINT32_C(0xffff);
  const uint32_t truncated = representation.bits & UINT32_C(0xffff0000);
  const uint32_t rounded =
      truncated +
      (stateless_random16(
           seed, step, tensor_index, element_index) < remainder
           ? UINT32_C(0x10000)
           : UINT32_C(0));
  return c10::BFloat16(
      static_cast<uint16_t>(rounded >> 16), c10::BFloat16::from_bits());
}

template <typename index_t>
struct AdamWBF16SRFunctor {
  __device__ __forceinline__ void operator()(
      index_t chunk_size,
      volatile int* noop_flag,
      TensorListMetadata<4>& tensor_list,
      float beta1,
      float beta2,
      float beta1_correction,
      float beta2_correction,
      float epsilon,
      float learning_rate,
      float weight_decay,
      uint64_t seed,
      uint64_t stochastic_step) {
    if (*noop_flag != 0) {
      return;
    }

    const index_t tensor_location = tensor_list.block_to_tensor[blockIdx.x];
    const index_t stable_tensor_index =
        static_cast<index_t>(tensor_list.start_tensor_this_launch) +
        tensor_location;
    const index_t chunk_index = tensor_list.block_to_chunk[blockIdx.x];
    const index_t chunk_offset = chunk_index * chunk_size;
    const index_t remaining =
        static_cast<index_t>(tensor_list.sizes[tensor_location]) - chunk_offset;

    auto* gradient =
        static_cast<c10::BFloat16*>(
            tensor_list.addresses[0][tensor_location]) +
        chunk_offset;
    auto* parameter =
        static_cast<c10::BFloat16*>(
            tensor_list.addresses[1][tensor_location]) +
        chunk_offset;
    auto* first_moment =
        static_cast<c10::BFloat16*>(
            tensor_list.addresses[2][tensor_location]) +
        chunk_offset;
    auto* second_moment =
        static_cast<c10::BFloat16*>(
            tensor_list.addresses[3][tensor_location]) +
        chunk_offset;

    for (index_t start = 0;
         start < remaining && start < chunk_size;
         start += blockDim.x * kInstructionLevelParallelism) {
      float gradient_register[kInstructionLevelParallelism];
      float parameter_register[kInstructionLevelParallelism];
      float first_moment_register[kInstructionLevelParallelism];
      float second_moment_register[kInstructionLevelParallelism];

#pragma unroll
      for (int item = 0; item < kInstructionLevelParallelism; ++item) {
        const index_t index = start + threadIdx.x + item * blockDim.x;
        if (index < remaining && index < chunk_size) {
          gradient_register[item] = static_cast<float>(gradient[index]);
          parameter_register[item] = static_cast<float>(parameter[index]);
          first_moment_register[item] =
              static_cast<float>(first_moment[index]);
          second_moment_register[item] =
              static_cast<float>(second_moment[index]);
        } else {
          gradient_register[item] = 0.0f;
          parameter_register[item] = 0.0f;
          first_moment_register[item] = 0.0f;
          second_moment_register[item] = 0.0f;
        }
      }

#pragma unroll
      for (int item = 0; item < kInstructionLevelParallelism; ++item) {
        first_moment_register[item] =
            beta1 * first_moment_register[item] +
            (1.0f - beta1) * gradient_register[item];
        second_moment_register[item] =
            beta2 * second_moment_register[item] +
            (1.0f - beta2) * gradient_register[item] *
                gradient_register[item];
        const float unbiased_first =
            first_moment_register[item] / beta1_correction;
        const float unbiased_second =
            second_moment_register[item] / beta2_correction;
        const float adam_update =
            unbiased_first / (sqrtf(unbiased_second) + epsilon);
        // Preserve AdamW's decoupled, sequential update semantics exactly.
        // Combining these expressions is equivalent only for finite values:
        // at lr=wd=0, a combined 0 * (0 * Inf) spuriously creates NaN.
        parameter_register[item] *=
            1.0f - learning_rate * weight_decay;
        parameter_register[item] -= learning_rate * adam_update;
      }

#pragma unroll
      for (int item = 0; item < kInstructionLevelParallelism; ++item) {
        const index_t index = start + threadIdx.x + item * blockDim.x;
        if (index < remaining && index < chunk_size) {
          const uint64_t element_index =
              static_cast<uint64_t>(chunk_offset + index);
          parameter[index] = stochastic_bf16(
              parameter_register[item],
              seed,
              stochastic_step,
              static_cast<uint64_t>(stable_tensor_index),
              element_index);
          // Moments deliberately use deterministic round-to-nearest BF16.
          first_moment[index] = c10::BFloat16(first_moment_register[item]);
          second_moment[index] = c10::BFloat16(second_moment_register[item]);
        }
      }
    }
  }
};

void validate_tensor_lists(
    const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  TORCH_CHECK(tensor_lists.size() == 4, "expected [grad, param, m, v]");
  TORCH_CHECK(!tensor_lists[0].empty(), "tensor lists must be nonempty");
  const size_t tensor_count = tensor_lists[0].size();
  const auto reference_device = tensor_lists[0][0].device();
  for (size_t list_index = 0; list_index < tensor_lists.size(); ++list_index) {
    TORCH_CHECK(
        tensor_lists[list_index].size() == tensor_count,
        "all tensor lists must have equal length");
    for (size_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
      const auto& tensor = tensor_lists[list_index][tensor_index];
      TORCH_CHECK(tensor.is_cuda(), "all tensors must be CUDA tensors");
      TORCH_CHECK(
          tensor.scalar_type() == at::kBFloat16,
          "all tensors must use bfloat16 storage");
      TORCH_CHECK(tensor.is_contiguous(), "all tensors must be contiguous");
      TORCH_CHECK(
          tensor.numel() == tensor_lists[0][tensor_index].numel(),
          "corresponding tensors must have equal numel");
      TORCH_CHECK(
          static_cast<uint64_t>(tensor.numel()) <= kMaximumTensorElements,
          "individual tensors may not exceed 2**32 - 1 elements");
      TORCH_CHECK(
          tensor.device() == reference_device,
          "all tensors must be on one CUDA device");
    }
  }
}

}  // namespace

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
    uint64_t seed) {
  validate_tensor_lists(tensor_lists);
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(adam_step > 0, "Adam step must be positive");
  TORCH_CHECK(noop_flag.is_cuda(), "noop flag must be a CUDA tensor");
  TORCH_CHECK(
      noop_flag.scalar_type() == at::kInt && noop_flag.numel() == 1,
      "noop flag must be one int32 value");
  TORCH_CHECK(
      noop_flag.device() == tensor_lists[0][0].device(),
      "noop flag and optimizer tensors must share a device");

  const c10::cuda::OptionalCUDAGuard device_guard(
      at::device_of(tensor_lists[0][0]));
  const float beta1_correction =
      1.0f - std::pow(static_cast<float>(beta1), adam_step);
  const float beta2_correction =
      1.0f - std::pow(static_cast<float>(beta2), adam_step);

  bool requires_64bit_indexing = false;
  for (const auto& tensor : tensor_lists[0]) {
    if (tensor.numel() >= std::numeric_limits<int32_t>::max()) {
      requires_64bit_indexing = true;
      break;
    }
  }

  if (requires_64bit_indexing) {
    multi_tensor_apply<4>(
        static_cast<int64_t>(kBlockSize),
        chunk_size,
        noop_flag,
        tensor_lists,
        AdamWBF16SRFunctor<int64_t>(),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        beta1_correction,
        beta2_correction,
        static_cast<float>(epsilon),
        static_cast<float>(learning_rate),
        static_cast<float>(weight_decay),
        seed,
        stochastic_step);
  } else {
    multi_tensor_apply<4>(
        kBlockSize,
        chunk_size,
        noop_flag,
        tensor_lists,
        AdamWBF16SRFunctor<int32_t>(),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        beta1_correction,
        beta2_correction,
        static_cast<float>(epsilon),
        static_cast<float>(learning_rate),
        static_cast<float>(weight_decay),
        seed,
        stochastic_step);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) 2026 Graphcore Ltd. All rights reserved.
// Vendored and adapted from NVIDIA Apex; see APEX_LICENSE and NOTICE.md.
#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <cassert>
#include <vector>

#include "compat.h"

// Keep these bounds aligned with the upstream Apex metadata ABI. The host
// dispatcher emits multiple kernels when either fixed-capacity table fills.
constexpr int depth_to_max_tensors[6] = {110, 64, 48, 36, 30, 24};
constexpr int depth_to_max_blocks[6] = {320, 320, 320, 320, 320, 320};

template <int n>
struct TensorListMetadata {
  void* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t sizes[depth_to_max_tensors[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

template <typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(
    int64_t chunk_size,
    volatile int* noop_flag,
    T tensor_list,
    U callable,
    ArgTypes... args) {
  callable(chunk_size, noop_flag, tensor_list, args...);
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    int64_t block_size,
    int64_t chunk_size,
    const at::Tensor& noop_flag,
    const std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(tensor_lists.size() == depth, "tensor_lists.size() != depth");
  const int tensor_count = tensor_lists[0].size();
  TORCH_CHECK(tensor_count > 0, "tensor_lists must be nonempty");
  const auto reference_device = tensor_lists[0][0].device();
  TORCH_CHECK(reference_device.type() == at::kCUDA, "expected CUDA tensors");

  for (int list_index = 0; list_index < tensor_lists.size(); ++list_index) {
    TORCH_CHECK(
        tensor_lists[list_index].size() == tensor_count,
        "tensor-list sizes differ");
    for (int tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
      const auto& tensor = tensor_lists[list_index][tensor_index];
      bool contiguous = tensor.is_contiguous();
#ifdef VERSION_GE_1_5
      contiguous = contiguous ||
          tensor.is_contiguous(at::MemoryFormat::ChannelsLast) ||
          tensor.is_contiguous(at::MemoryFormat::ChannelsLast3d);
#endif
      TORCH_CHECK(contiguous, "all tensors must be contiguous");
      TORCH_CHECK(
          tensor.device() == reference_device,
          "all tensors must be on one device");
      TORCH_CHECK(
          tensor.numel() == tensor_lists[0][tensor_index].numel(),
          "corresponding tensors must have equal numel");
    }
  }

  TensorListMetadata<depth> tensor_list;
  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(tensor_lists[0][0]));
  const auto stream = at::cuda::getCurrentCUDAStream();

  tensor_list.start_tensor_this_launch = 0;
  int local_block_count = 0;
  int local_tensor_count = 0;
  for (int tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
    tensor_list.sizes[local_tensor_count] =
        tensor_lists[0][tensor_index].numel();
    for (int list_index = 0; list_index < depth; ++list_index) {
      tensor_list.addresses[list_index][local_tensor_count] =
          tensor_lists[list_index][tensor_index].data_ptr();
    }
    ++local_tensor_count;

    const auto chunks_this_tensor =
        (tensor_lists[0][tensor_index].numel() + chunk_size - 1) / chunk_size;
    for (int64_t chunk = 0; chunk < chunks_this_tensor; ++chunk) {
      tensor_list.block_to_tensor[local_block_count] = local_tensor_count - 1;
      tensor_list.block_to_chunk[local_block_count] = chunk;
      ++local_block_count;

      const bool tensors_full =
          local_tensor_count == depth_to_max_tensors[depth - 1] &&
          chunk == chunks_this_tensor - 1;
      const bool blocks_full =
          local_block_count == depth_to_max_blocks[depth - 1];
      const bool last_chunk =
          tensor_index == tensor_count - 1 &&
          chunk == chunks_this_tensor - 1;
      if (tensors_full || blocks_full || last_chunk) {
        multi_tensor_apply_kernel<<<local_block_count, block_size, 0, stream>>>(
            chunk_size,
            noop_flag.DATA_PTR<int>(),
            tensor_list,
            callable,
            args...);
        AT_CUDA_CHECK(cudaGetLastError());

        local_block_count = 0;
        if (chunk == chunks_this_tensor - 1) {
          local_tensor_count = 0;
          tensor_list.start_tensor_this_launch = tensor_index + 1;
        } else {
          tensor_list.sizes[0] = tensor_list.sizes[local_tensor_count - 1];
          for (int list_index = 0; list_index < depth; ++list_index) {
            tensor_list.addresses[list_index][0] =
                tensor_list.addresses[list_index][local_tensor_count - 1];
          }
          local_tensor_count = 1;
          tensor_list.start_tensor_this_launch = tensor_index;
        }
      }
    }
  }
}

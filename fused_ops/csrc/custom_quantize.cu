/*
 * custom_quantize.cu — Separate TU that compiles the custom_quantisation/ kernels.
 *
 * This file is compiled by the JIT build with include paths set up so that
 * the custom_quantisation/ headers can resolve their relative includes.
 *
 * Required include paths (set up in fused_te_linear.py JIT build):
 *   - TE_INCLUDE  (transformer_engine/common/include — public API)
 *   - TE_COMMON   (transformer_engine/common — internal headers)
 *   - TE_PARENT   (transformer_engine/ — for #include "common/utils.cuh")
 *   - TE_CUSTOM   (cast/nvfp4/custom_quantisation — resolve headers)
 *   - TE_CAST_NVFP4 (cast/nvfp4 — for ../../common.h fallback)
 *
 * Uses NVTETensor C API at the interface, converts internally to TE Tensor*.
 */

// Undo PyTorch's restrictive operator disabling - TE kernels need them
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/recipe.h>

// TE internal header for convertNVTETensor
#include "common.h"

// Core TMA alignment function from cast/core/common.cuh
// We define it directly here to avoid pulling in CheckInputTensor etc.
// from the full core/common.cuh (those symbols aren't exported by libtransformer_engine.so).
#ifndef TMA_SHMEM_ALIGNMENT
#define TMA_SHMEM_ALIGNMENT 128
#endif
namespace transformer_engine { namespace dispatch { namespace common {
__device__ __forceinline__ unsigned char *align_smem_ptr_per_TMA_requirements(unsigned char *p) {
  size_t addr = reinterpret_cast<size_t>(p);
  addr = (addr + TMA_SHMEM_ALIGNMENT - 1) & ~(TMA_SHMEM_ALIGNMENT - 1);
  return reinterpret_cast<unsigned char *>(addr);
}
}}}  // namespace transformer_engine::dispatch::common

// Include the custom_quantisation kernels (resolved via include paths)
#include "quantize_transpose_nvfp4.cuh"
#include "group_quantize_transpose_nvfp4.cuh"

using namespace transformer_engine;

// ─── Single-tensor quantize+transpose ───────────────────────────────────
void custom_nvfp4_quantize_transpose(
    NVTETensor input, NVTETensor output, cudaStream_t stream) {
  Tensor *in_ptr = convertNVTETensor(input);
  Tensor *out_ptr = convertNVTETensor(output);
  Tensor noop_tensor;
  // Dispatch to the custom version (always 1D quantization for bf16)
  dispatch::nvfp4::quantize_transpose<false>(*in_ptr, &noop_tensor, out_ptr, nullptr, stream);
}

// ─── Grouped quantize+transpose ─────────────────────────────────────────
void custom_nvfp4_group_quantize_transpose(
    NVTETensor input, NVTETensor* output_list,
    const size_t* split_sections, size_t num_tensors,
    cudaStream_t stream) {
  Tensor *in_ptr = convertNVTETensor(input);
  Tensor noop_tensor;
  std::vector<Tensor *> out_ptrs(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    out_ptrs[i] = convertNVTETensor(output_list[i]);
  }
  dispatch::nvfp4::group_quantize_transpose<false>(
      *in_ptr, &noop_tensor, out_ptrs, split_sections, num_tensors, nullptr, stream);
}

// ─── Grouped quantize+transpose with TK sg/b_sg output ─────────────────
void custom_nvfp4_group_quantize_transpose_tk(
    NVTETensor input, NVTETensor* output_list,
    const size_t* split_sections, size_t num_tensors,
    float* sg_output, float* fwd_b_sg, float* dgrad_b_sg,
    int b_tile_size,
    cudaStream_t stream) {
  Tensor *in_ptr = convertNVTETensor(input);
  Tensor noop_tensor;
  std::vector<Tensor *> out_ptrs(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    out_ptrs[i] = convertNVTETensor(output_list[i]);
  }
  dispatch::nvfp4::group_quantize_transpose<false>(
      *in_ptr, &noop_tensor, out_ptrs, split_sections, num_tensors, nullptr, stream,
      sg_output, fwd_b_sg, dgrad_b_sg, b_tile_size);
}

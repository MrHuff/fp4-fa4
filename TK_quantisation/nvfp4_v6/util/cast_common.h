/*
 * Minimal standalone replacement for TE's common.h
 * 
 * ONLY provides symbols that utils.cuh does NOT define:
 *   - FP4_TYPE_SUPPORTED macro
 *   - TMA_SHMEM_ALIGNMENT
 *   - DIVUP / DIVUP_TO_MULTIPLE macros
 *   - TypeExtrema<> specializations
 *   - align_smem_ptr_per_TMA_requirements
 *   - NVTE_CHECK / NVTE_ERROR stubs (host-side, dead code under TK_STANDALONE)
 *
 * Everything else (Vec<>, type aliases, THREADS_PER_WARP, NVTE_DEVICE_ERROR,
 * Quantized_Limits, e8m0_t) comes from util/utils.cuh — do NOT duplicate here.
 */

#ifndef TK_QUANT_CAST_COMMON_H_
#define TK_QUANT_CAST_COMMON_H_

#include <cudaTypedefs.h>

// Detect FP4 support (CUDA 12.8+)
#ifndef FP4_TYPE_SUPPORTED
#define FP4_TYPE_SUPPORTED (CUDA_VERSION >= 12080)
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <type_traits>

// ─────────── TMA alignment ───────────
#ifndef TMA_SHMEM_ALIGNMENT
constexpr size_t TMA_SHMEM_ALIGNMENT = 128;
#endif

// ─────────── Macros ───────────
#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif
#ifndef DIVUP_TO_MULTIPLE
#define DIVUP_TO_MULTIPLE(x, y) (DIVUP(x, y) * (y))
#endif

// ─────────── Host-side stub macros (dead code under TK_STANDALONE) ───────────
#ifdef TK_STANDALONE
#ifndef NVTE_CHECK
#define NVTE_CHECK(...) do {} while(0)
#endif
#ifndef NVTE_CHECK_CUDA
#define NVTE_CHECK_CUDA(...) do {} while(0)
#endif
#ifndef NVTE_ERROR
#define NVTE_ERROR(...) do {} while(0)
#endif
#endif  // TK_STANDALONE

// ─────────── Type extrema (kernel-only, not in utils.cuh) ───────────
namespace transformer_engine {
namespace detail {

template <typename T>
struct TypeExtrema {};

template <>
struct TypeExtrema<float> {
  static constexpr float max = 3.402823466e+38f;
  static constexpr float min = -3.402823466e+38f;
};

template <>
struct TypeExtrema<__nv_bfloat16> {
  static constexpr float max = 3.3895314e+38f;
  static constexpr float min = -3.3895314e+38f;
};

template <>
struct TypeExtrema<__half> {
  static constexpr float max = 65504.0f;
  static constexpr float min = -65504.0f;
};

#if FP4_TYPE_SUPPORTED
template <>
struct TypeExtrema<__nv_fp8_e4m3> {
  static constexpr float max = 448.0f;
  static constexpr float min = -448.0f;
};

template <>
struct TypeExtrema<__nv_fp4_e2m1> {
  static constexpr float max = 6.0f;
  static constexpr float min = -6.0f;
};
#endif

}  // namespace detail

// ─────────── Type aliases needed by ptx.cuh (before utils.cuh) ───────────
using bf16 = __nv_bfloat16;
using fp16 = __half;
#if FP4_TYPE_SUPPORTED
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
using nvfp4_scale_t = fp8e4m3;
#endif

// ─────────── Common namespace ───────────
namespace common {

__device__ __forceinline__ unsigned char *
align_smem_ptr_per_TMA_requirements(unsigned char *ptr) {
  uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned = (base + TMA_SHMEM_ALIGNMENT - 1) &
                       ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));
  return reinterpret_cast<unsigned char *>(aligned);
}

}  // namespace common

}  // namespace transformer_engine

#endif  // TK_QUANT_CAST_COMMON_H_

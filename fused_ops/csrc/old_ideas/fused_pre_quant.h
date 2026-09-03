#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// Optimized version with inv_rms caching for backward pass
void launch_fused_pre_quant(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float* global_amax,
    float scale_override,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale_out,
    float* inv_rms_cache  // Output: cached inv_rms per row for backward reuse
);

// Legacy wrapper (backward compatible, allocates temp buffer internally)
void launch_fused_pre_quant(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float* global_amax,
    float scale_override,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale_out
);

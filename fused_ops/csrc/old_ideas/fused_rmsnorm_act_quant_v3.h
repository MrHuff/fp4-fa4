#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// V3 Optimized Fused RMSNorm + SiLU Activation + FP4 Quantization
// - Warp shuffle reductions instead of atomics
// - inv_rms factored into quantization scale (avoids per-element multiply)

void launch_fused_rmsnorm_act_quant_v3(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float scale_override,
    bool use_four_six,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale,
    float* inv_rms_cache,
    float* block_amax_scratch
);

void launch_fused_rmsnorm_act_quant_v3(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float scale_override,
    bool use_four_six,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale,
    float* inv_rms_cache
);

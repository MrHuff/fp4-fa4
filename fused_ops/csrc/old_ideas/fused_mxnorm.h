#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// Optimized Fused MXNorm + SiLU Activation + FP4 Quantization
// Supports multiple normalization estimation modes
// norm_mode:
// 0: Standard RMSNorm (sqrt(mean(x^2)))
// 1: AbsMax Estimate (alpha * max|x|)
// 2: Block-Max RMS (RMS(block_maxes))

void launch_fused_mxnorm(
    const nv_bfloat16* x,           // Input [rows, cols]
    const nv_bfloat16* w,           // Norm gain [cols]
    float epsilon,
    int rows, int cols,
    float scale_override,           // Quantization scale factor
    bool use_four_six,              // true = four-six search, false = RTN
    int norm_mode,                  // 0=RMS, 1=AbsMax, 2=BlockMax
    __nv_fp4x4_e2m1* y,             // Output quantized [rows * cols / 2]
    __nv_fp8_e4m3* scales,          // Block scales [rows * cols / 16]
    float* global_scale,            // Single global scale output
    float* inv_rms_cache,           // [rows] cached for backward pass
    float* block_amax_scratch       // [rows * cols / 16] scratch buffer
);

// Fast MXNorm (AbsMax Only)
void launch_fused_mxnorm_fast(
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

// Fused Backward AbsMax (Exact)
void launch_fused_backward_absmax(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    const float* inv_rms_cache,
    int rows, int cols,
    nv_bfloat16* grad_input
);

// Fast MXNorm (Block-Max Only)
void launch_fused_mxnorm_fast_block(
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

// Convenience wrapper for fast kernel
void launch_fused_mxnorm_fast(
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

// Convenience wrapper that allocates scratch internally
void launch_fused_mxnorm(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float scale_override,
    bool use_four_six,
    int norm_mode,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale,
    float* inv_rms_cache
);

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// Fused RMSNorm + SiLU Activation + FP4 Quantization
// Single cooperative kernel with grid-wide sync

// Full version with external scratch buffer
void launch_fused_rmsnorm_act_quant(
    const nv_bfloat16* x,           // Input [rows, cols]
    const nv_bfloat16* w,           // RMSNorm weight [cols]
    float epsilon,
    int rows, int cols,
    float scale_override,           // Quantization scale factor
    bool use_four_six,              // true = four-six search, false = RTN
    __nv_fp4x4_e2m1* y,             // Output quantized [rows * cols / 2]
    __nv_fp8_e4m3* scales,          // Block scales [rows * cols / 16]
    float* global_scale,            // Single global scale output
    float* inv_rms_cache,           // [rows] cached for backward pass
    float* block_amax_scratch       // [rows] scratch buffer
);

// Convenience wrapper that allocates scratch internally
void launch_fused_rmsnorm_act_quant(
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

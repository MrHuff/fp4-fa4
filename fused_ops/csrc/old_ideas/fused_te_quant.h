#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Fused RMSNorm/AbsMax/MXNorm + SiLU/GeLU/Identity + NVFP4 Quantization
// Output is TE-compatible: packed FP4 [M,K/2] + FP8 scales [M,K/16] + float global_scale
//
// norm_mode: 0=RMSNorm, 1=AbsMax, 2=MXNorm
// act_mode:  0=SiLU, 1=GeLU, 2=Identity

void launch_fused_te_quant(
    const nv_bfloat16* x,           // Input [rows, cols]
    const nv_bfloat16* w,           // Norm/gain weight [cols]
    float epsilon,
    int rows, int cols,
    int norm_mode,                  // 0=RMS, 1=AbsMax, 2=MXNorm
    int act_mode,                   // 0=SiLU, 1=GeLU, 2=Identity
    unsigned char* y,               // Packed FP4 output [rows, cols/2]
    __nv_fp8_e4m3* scales,          // Block scales [rows, cols/16]
    float* global_scale,            // Single float global scale
    float* inv_rms_cache,           // [rows] cached for backward
    float* block_amax_scratch       // [rows * cols/16] scratch buffer
);

// Convenience wrapper (allocates scratch internally)
void launch_fused_te_quant(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    int norm_mode,
    int act_mode,
    unsigned char* y,
    __nv_fp8_e4m3* scales,
    float* global_scale,
    float* inv_rms_cache
);

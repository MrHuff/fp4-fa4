#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// Fused dequant → transpose → requant (no Hadamard)
// Input:  FP4 tensor x[N, K/2] + scales[N, K/16] + global_scale
// Output: FP4 tensor y[K, N/2] + scales[K, N/16] + global_scale
void launch_dequant_transpose_quant(
    __nv_fp4x2_storage_t* y,
    __nv_fp8_e4m3* y_scales,
    float* y_global_scale,
    const __nv_fp4x2_storage_t* x,
    const __nv_fp8_e4m3* x_scales,
    const float* x_global_scale,
    float* scratch_amax,         // preallocated float for atomic max
    int N, int K,
    float scale_override,
    cudaStream_t stream = 0
);

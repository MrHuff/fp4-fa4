#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

void launch_fused_silu_rmsnorm_backward(
    const nv_bfloat16* grad_output,     // dx_proj from dgrad GEMM (M, K)
    const nv_bfloat16* input,            // x_raw (M, K)
    const nv_bfloat16* weight,           // gamma (K,)
    const float* cached_inv_rms,         // (M,) float
    int rows, int cols,
    nv_bfloat16* grad_input,             // dx output (M, K)
    float* grad_weight_accum,            // dgamma output (K,) float32
    cudaStream_t stream
);

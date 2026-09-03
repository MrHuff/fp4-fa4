#pragma once
#include <cuda_bf16.h>

// Optimized backward kernel with faster math intrinsics
void launch_fused_backward_opt(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    const float* cached_inv_rms,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* grad_input
);

void launch_fused_backward_opt(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* grad_input
);

#pragma once
#include <cuda_bf16.h>

// Optimized version with cached inv_rms from forward pass
void launch_fused_backward(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    const float* cached_inv_rms,  // From forward pass (can be nullptr for fallback)
    float epsilon,
    int rows, int cols,
    nv_bfloat16* grad_input
);

// Legacy version (backward compatible, recomputes inv_rms internally)
void launch_fused_backward(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* grad_input
);

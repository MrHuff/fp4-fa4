#pragma once
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// Activation types
enum class ActivationType : int {
    SILU = 0,
    RELU2 = 1,
    GELU = 2,
    ELU = 3
};

// Multi-activation fused kernel with runtime dispatch
void launch_fused_rmsnorm_act_quant_multiact_dispatch(
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    float epsilon,
    int rows, int cols,
    float scale_override,
    __nv_fp4x4_e2m1* output_fp4,
    __nv_fp8_e4m3* output_scales,
    float* global_scale,
    float* inv_rms_cache,
    bool use_four_six,
    int activation_type
);

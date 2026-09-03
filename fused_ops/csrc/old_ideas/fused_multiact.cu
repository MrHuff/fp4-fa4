// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + Multi-Activation + FP4 Quantization Kernel
// Supports SiLU, ReLU², GELU, ELU activations
// Simplified version without cooperative groups for testing

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>
#include <cub/cub.cuh>

#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;
using fp32x8 = GenericVector<float, 8>;
using fp4x8 = GenericVector<unsigned char, 4>;

// -------------------------------------------------------------------------
// Activation Types
// -------------------------------------------------------------------------

enum class ActivationType : int {
    SILU = 0,
    RELU2 = 1,
    GELU = 2,
    ELU = 3
};

// -------------------------------------------------------------------------
// Activation Functions (Forward)
// -------------------------------------------------------------------------

template<ActivationType ACT>
__device__ __forceinline__ float activation_fwd(float x);

template<>
__device__ __forceinline__ float activation_fwd<ActivationType::SILU>(float x) {
    return x / (1.0f + __expf(-x));
}

template<>
__device__ __forceinline__ float activation_fwd<ActivationType::RELU2>(float x) {
    float relu = fmaxf(0.0f, x);
    return relu * relu;
}

template<>
__device__ __forceinline__ float activation_fwd<ActivationType::GELU>(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

template<>
__device__ __forceinline__ float activation_fwd<ActivationType::ELU>(float x) {
    return (x > 0.0f) ? x : (__expf(x) - 1.0f);
}

// -------------------------------------------------------------------------
// Simple Forward Kernel (per-row processing, no cooperative groups)
// -------------------------------------------------------------------------

template<int BLOCK_SIZE = 256, ActivationType ACT = ActivationType::SILU>
__global__ void fused_multiact_kernel(
    const nv_bfloat16* __restrict__ input,
    const nv_bfloat16* __restrict__ weight,
    float epsilon,
    int rows, int cols,
    float scale,
    __nv_fp4x4_e2m1* __restrict__ output_fp4,
    __nv_fp8_e4m3* __restrict__ output_scales,
    float* __restrict__ inv_rms_cache
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const nv_bfloat16* row_input = input + row * cols;
    const int BLOCK_GROUP_SIZE = 16;
    
    // Phase 1: Compute RMSNorm sum-of-squares using cub reduction
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float sum_sq = 0.0f;
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 x_vec = *reinterpret_cast<const bf16x8*>(&row_input[i]);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(x_vec[k]);
            sum_sq += val * val;
        }
    }
    
    float total_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);
    __syncthreads();
    
    __shared__ float s_inv_rms;
    if (tid == 0) {
        float rms = sqrtf(total_sum_sq / cols + epsilon);
        s_inv_rms = 1.0f / rms;
        if (inv_rms_cache) inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    float inv_rms = s_inv_rms;
    
    // Phase 2: Apply RMSNorm + Activation + Quantize
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 x_vec = *reinterpret_cast<const bf16x8*>(&row_input[i]);
        bf16x8 w_vec = *reinterpret_cast<const bf16x8*>(&weight[i]);
        
        // Compute block absmax and activated values
        float block_amax = 0.0f;
        bf16x8 act_vec;
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(x_vec[k]) * inv_rms;
            float w = static_cast<float>(w_vec[k]);
            float act_val = activation_fwd<ACT>(val) * w;
            act_vec[k] = static_cast<nv_bfloat16>(act_val);
            block_amax = fmaxf(block_amax, fabsf(act_val));
        }
        
        // Quantize to FP4
        float val_max = 6.0f;
        float s_group = block_amax / val_max;
        __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / scale);
        float s_round = static_cast<float>(s_as_fp8);
        if (s_round == 0.0f) s_round = 1.0f;
        
        float factor = 1.0f / (s_round * scale);
        fp4x8 result;
        
        #pragma unroll
        for (int k = 0; k < 8; k += 2) {
            float2 src;
            src.x = static_cast<float>(act_vec[k]) * factor;
            src.y = static_cast<float>(act_vec[k+1]) * factor;
            unsigned char bits = __nv_cvt_float2_to_fp4x2(src, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
            result[k/2] = bits;
        }
        
        // Write output using byte offset (matching V1 pattern)
        int vec_idx = (row * cols + i) / 8;  
        result.store(reinterpret_cast<unsigned char*>(output_fp4) + 4 * vec_idx);
        
        // Write scale every 16 elements (every 2 vectors)
        if (vec_idx % 2 == 0) {
            output_scales[vec_idx / 2] = s_as_fp8;
        }
    }
}

// -------------------------------------------------------------------------
// Launch Wrappers
// -------------------------------------------------------------------------

template<ActivationType ACT>
void launch_fused_multiact(
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    float epsilon,
    int rows, int cols,
    float scale,
    __nv_fp4x4_e2m1* output_fp4,
    __nv_fp8_e4m3* output_scales,
    float* inv_rms_cache
) {
    constexpr int BLOCK_SIZE = 256;
    fused_multiact_kernel<BLOCK_SIZE, ACT><<<rows, BLOCK_SIZE>>>(
        input, weight, epsilon, rows, cols, scale,
        output_fp4, output_scales, inv_rms_cache
    );
    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template void launch_fused_multiact<ActivationType::SILU>(
    const nv_bfloat16*, const nv_bfloat16*, float, int, int, float,
    __nv_fp4x4_e2m1*, __nv_fp8_e4m3*, float*);
template void launch_fused_multiact<ActivationType::RELU2>(
    const nv_bfloat16*, const nv_bfloat16*, float, int, int, float,
    __nv_fp4x4_e2m1*, __nv_fp8_e4m3*, float*);
template void launch_fused_multiact<ActivationType::GELU>(
    const nv_bfloat16*, const nv_bfloat16*, float, int, int, float,
    __nv_fp4x4_e2m1*, __nv_fp8_e4m3*, float*);
template void launch_fused_multiact<ActivationType::ELU>(
    const nv_bfloat16*, const nv_bfloat16*, float, int, int, float,
    __nv_fp4x4_e2m1*, __nv_fp8_e4m3*, float*);

// Runtime dispatcher
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
) {
    // Note: For simplified version, scale must be pre-computed
    float scale = (scale_override > 0.0f) ? scale_override : 1.0f;
    
    switch (static_cast<ActivationType>(activation_type)) {
        case ActivationType::SILU:
            launch_fused_multiact<ActivationType::SILU>(
                input, weight, epsilon, rows, cols, scale,
                output_fp4, output_scales, inv_rms_cache);
            break;
        case ActivationType::RELU2:
            launch_fused_multiact<ActivationType::RELU2>(
                input, weight, epsilon, rows, cols, scale,
                output_fp4, output_scales, inv_rms_cache);
            break;
        case ActivationType::GELU:
            launch_fused_multiact<ActivationType::GELU>(
                input, weight, epsilon, rows, cols, scale,
                output_fp4, output_scales, inv_rms_cache);
            break;
        case ActivationType::ELU:
            launch_fused_multiact<ActivationType::ELU>(
                input, weight, epsilon, rows, cols, scale,
                output_fp4, output_scales, inv_rms_cache);
            break;
        default:
            throw std::runtime_error("Unknown activation type");
    }
    
    // Set global_scale to 1.0 for now (scale was used directly)
    if (global_scale) {
        cudaMemset(global_scale, 0, sizeof(float));
        float one = scale;
        cudaMemcpy(global_scale, &one, sizeof(float), cudaMemcpyHostToDevice);
    }
}

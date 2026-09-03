// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Optimized Fused Pre-Quantization Kernel
// - 2 kernels instead of 3 (merged amax reduce with atomic)
// - Caches inv_rms for backward pass reuse
// - No redundant RMSNorm recomputation

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

// SiLU Activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// RMSNorm helper - Sum of squares reduction within a block (row)
__device__ __forceinline__ float block_reduce_sum(float val) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    return BlockReduce(temp_storage).Sum(val);
}

struct MaxOp {
    __device__ __forceinline__ float operator()(const float &a, const float &b) const {
        return fmaxf(a, b);
    }
};

__device__ __forceinline__ float block_reduce_max(float val) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    return BlockReduce(temp_storage).Reduce(val, MaxOp());
}

// Atomic max for floats (positive values only)
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    // For positive floats, we can use atomicMax on the bit representation
    // since IEEE754 floats have the property that for positive values,
    // the integer representation preserves ordering
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_uint, assumed, __float_as_uint(val));
    } while (assumed != old);
}

// -------------------------------------------------------------------------
// Quantization Logic
// -------------------------------------------------------------------------

struct QuantResult {
    fp4x8 bits;
    float scale;
    __nv_fp8_e4m3 fp8s;
};

__device__ __forceinline__ QuantResult quantize_vec(float abs_max, float val_max, float scale, bf16x8& x) {
    float s_group = abs_max / val_max;
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0) s_round_fp8 = 1.f;

    float factor = 1.f / (s_round_fp8 * scale);
    fp4x8 result;
    for (int k = 0; k < bf16x8::size; k += 2) {
        float2 src;
        src.x = static_cast<float>(x[k+0]) * factor;
        src.y = static_cast<float>(x[k+1]) * factor;
        unsigned char bits = __nv_cvt_float2_to_fp4x2(src, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
        result[k/2] = bits;
    }

    return QuantResult{result, s_round_fp8, s_as_fp8};
}

__device__ __forceinline__ QuantResult quantize_rtn_vec(float abs_max, float val_max, float scale, bf16x8& x) {
     return quantize_vec(abs_max, val_max, scale, x);
}

// -------------------------------------------------------------------------
// Kernel 1: Fused RMSNorm + Activation + Absmax (with atomic global reduce)
// Also caches inv_rms for backward pass
// -------------------------------------------------------------------------

__global__ void fused_amax_kernel_v2(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ range_weight_ptr,
    float epsilon,
    int rows, int cols,
    float* __restrict__ global_amax_ptr,      // Single float output (atomic update)
    float* __restrict__ inv_rms_cache_ptr     // Per-row cache for backward
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;

    // 1. Compute RMSNorm stats (Sum of Squares)
    float sum_sq = 0.0f;
    for (int i = tid * 8; i < cols; i += blockDim.x * 8) {
        bf16x8 data = bf16x8::load(x_ptr + row * cols + i);
        #pragma unroll
        for(int k=0; k<8; ++k) {
             float val = static_cast<float>(data[k]);
             sum_sq += val * val;
        }
    }
    
    // Reduce sum_sq within block
    float row_sum_sq = block_reduce_sum(sum_sq);
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
        // Cache inv_rms for backward pass reuse
        inv_rms_cache_ptr[row] = inv_rms;
    }
    __syncthreads();
    
    // 2. Compute Max Abs of Act(RMSNorm(x)) in same pass
    float local_max = 0.0f;
    for (int i = tid * 8; i < cols; i += blockDim.x * 8) {
        bf16x8 data = bf16x8::load(x_ptr + row * cols + i);
        bf16x8 w_data = bf16x8::load(range_weight_ptr + i);
        
        #pragma unroll
        for(int k=0; k<8; ++k) {
             float val = static_cast<float>(data[k]);
             float w = static_cast<float>(w_data[k]);
             // RMSNorm + SiLU
             float norm_val = val * inv_rms * w;
             float act_val = silu(norm_val);
             local_max = fmaxf(local_max, fabsf(act_val));
        }
    }
    
    // Reduce to get row max
    float row_max = block_reduce_max(local_max);
    
    // Atomic update global max (eliminates separate reduce kernel)
    if (tid == 0) {
        atomicMaxFloat(global_amax_ptr, row_max);
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Quantize (using cached inv_rms - no recomputation!)
// -------------------------------------------------------------------------

__global__ void fused_quant_kernel_v2(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ range_weight_ptr,
    const float* __restrict__ inv_rms_cache_ptr,  // Cached from Kernel 1
    int rows, int cols,
    const float* __restrict__ global_amax_ptr,
    float scale_override,
    __nv_fp4x4_e2m1* __restrict__ y_ptr,
    __nv_fp8_e4m3* __restrict__ scale_ptr,
    float* __restrict__ global_scale_out_ptr
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    
    // Load cached inv_rms (NO RECOMPUTATION!)
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = inv_rms_cache_ptr[row];
    }
    __syncthreads();
    
    // Global Scaling
    float global_abs_max = *global_amax_ptr;
    float scales_max = 448.f;
    float val_max = 6.f * (1.0f / scale_override);
    float scale = (global_abs_max == 0) ? 1.f : global_abs_max / scales_max / val_max;
    
    if (tid == 0 && row == 0) {
        *global_scale_out_ptr = scale;
    }

    // Apply RMSNorm + SiLU + Quantize
    for (int i = tid * 8; i < cols; i += blockDim.x * 8) {
        bf16x8 data = bf16x8::load(x_ptr + row * cols + i);
        bf16x8 w_data = bf16x8::load(range_weight_ptr + i);
        
        bf16x8 act_vec;
        
        #pragma unroll
        for(int k=0; k<8; ++k) {
             float val = static_cast<float>(data[k]);
             float w = static_cast<float>(w_data[k]);
             float norm_val = val * inv_rms * w;
             float act_val = silu(norm_val);
             act_vec[k] = (nv_bfloat16)act_val;
        }
        
        // Quantize with micro-scaling
        nv_bfloat16 vec_abs_max = vecReduceAbsMax(act_vec); 
        nv_bfloat16 neighbor_abs_max = __shfl_xor_sync(0xffffffff, vec_abs_max, 1);
        float full_abs_max = static_cast<float>(__hmax(vec_abs_max, neighbor_abs_max));
        
        QuantResult res = quantize_rtn_vec(full_abs_max, val_max, scale, act_vec);
        
        // Store quantized values
        int vec_idx = (row * cols + i) / 8;
        res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * vec_idx);
        
        // Store Scale (every 2 vectors / 16 elements)
        if (vec_idx % 2 == 0) {
            scale_ptr[vec_idx / 2] = res.fp8s;
        }
    }
}

// -------------------------------------------------------------------------
// Host Wrapper - Now only 2 kernel calls!
// -------------------------------------------------------------------------

void launch_fused_pre_quant(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float* global_amax,
    float scale_override,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale_out,
    float* inv_rms_cache  // NEW: output buffer for backward reuse
) {
    int block_size = 256;
    int grid_size = rows;
    
    // Initialize global_amax to 0 for atomic max
    cudaMemsetAsync(global_amax, 0, sizeof(float), 0);
    
    // Kernel 1: RMSNorm stats + Activation + Absmax (with atomic global reduce)
    // Also caches inv_rms
    fused_amax_kernel_v2<<<grid_size, block_size>>>(
        x, w, epsilon, rows, cols, global_amax, inv_rms_cache
    );
    
    // Kernel 2: Quantize using cached inv_rms (NO RECOMPUTATION!)
    fused_quant_kernel_v2<<<grid_size, block_size>>>(
        x, w, inv_rms_cache, rows, cols, global_amax, scale_override, y, scales, global_scale_out
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Legacy wrapper for backward compatibility (allocates temp buffer)
void launch_fused_pre_quant(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float* global_amax,
    float scale_override,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale_out
) {
    // Allocate temp inv_rms cache (not ideal, but maintains compatibility)
    float* temp_inv_rms;
    cudaMallocAsync(&temp_inv_rms, rows * sizeof(float), 0);
    
    launch_fused_pre_quant(x, w, epsilon, rows, cols, global_amax, scale_override, 
                           y, scales, global_scale_out, temp_inv_rms);
    
    cudaFreeAsync(temp_inv_rms, 0);
}

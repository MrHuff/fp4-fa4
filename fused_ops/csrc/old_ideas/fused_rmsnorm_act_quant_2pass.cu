// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + SiLU Activation + FP4 Quantization (2-Pass Approach)
// Pass 1: Reduction (Compute inv_rms and global_amax)
// Pass 2: Quantization (Apply stats)
//
// Goal: Avoid global grid synchronization overhead of cooperative launch.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cub/cub.cuh>

#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;
using fp32x8 = GenericVector<float, 8>;
using fp4x8 = GenericVector<unsigned char, 4>;

constexpr int BLOCK_GROUP_SIZE = 16; 

// -------------------------------------------------------------------------
// Helper: Float Atomic Max
// -------------------------------------------------------------------------
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    // We assume value >= 0 (since it's absmax)
    // For positive floats, int-casting preserves ordering.
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
         __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

// -------------------------------------------------------------------------
// Activation
// -------------------------------------------------------------------------
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// -------------------------------------------------------------------------
// Pass 1: Reduction Kernel
// Computes inv_rms per row and global_amax across grid
// -------------------------------------------------------------------------
template<int BLOCK_SIZE=256>
__global__ void fused_reduction_kernel(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float* __restrict__ inv_rms_cache,   // Output: per-row inv_rms
    float* __restrict__ global_amax_ptr  // Output: Single global float for AMAX
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    const nv_bfloat16* row_ptr = x_ptr + row * cols;
    
    // 1. Compute sum_sq (for RMSNorm) and local_amax (for Global Scale)
    float sum_sq = 0.0f;
    float local_max = 0.0f;
    
    // Iterate over the row
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(row_ptr + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            
            // RMSNorm accumulation
            sum_sq += val * val;
            
            // AMAX accumulation
            // Only need "pre-RMS" amax? No, we need "post-activation" amax.
            // But we don't have inv_rms yet!
            // Wait, we can't compute post-act amax comfortably without inv_rms.
            // BUT: We can compute sum_sq, then reduce it to get inv_rms, 
            // THEN re-read (or keep in registers if small) to compute act_max.
            // Since Cols can be large (8k), register file won't hold it. 
            // We have to re-read or accept 2 passes over data within this kernel.
            // Re-reading is safer for register pressure.
        }
    }
    
    // 2. Reduce sum_sq -> inv_rms
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float row_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);
    
    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
        inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    
    float inv_rms = s_inv_rms;
    
    // 3. Compute AMAX (Post-Activation)
    // We must re-read data because we didn't have inv_rms before.
    // (Optimization: for small Sequence Length, we could maybe cache? But 2048*2 bytes is too big for shared)
    
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(row_ptr + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            
            float norm_val = val * inv_rms * w;
            float act_val = silu(norm_val);
            local_max = fmaxf(local_max, fabsf(act_val));
        }
    }
    
    // 4. Block Reduction of AMAX
    struct MaxOp {
        __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
    };
    float row_max = BlockReduce(temp_storage).Reduce(local_max, MaxOp());
    
    // 5. Atomic Grid Reduction
    if (tid == 0) {
        atomicMaxFloat(global_amax_ptr, row_max);
    }
}


// -------------------------------------------------------------------------
// Quantization Helpers
// -------------------------------------------------------------------------
struct QuantResult {
    fp4x8 bits;
    float scale;
    __nv_fp8_e4m3 fp8s;
};

__device__ __forceinline__ QuantResult quantize_block(float abs_max, float val_max, float scale, bf16x8& x) {
    float s_group = abs_max / val_max;
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0) s_round_fp8 = 1.f;

    float factor = 1.f / (s_round_fp8 * scale);
    fp4x8 result;
    #pragma unroll
    for (int k = 0; k < bf16x8::size; k += 2) {
        float2 src;
        src.x = static_cast<float>(x[k+0]) * factor;
        src.y = static_cast<float>(x[k+1]) * factor;
        unsigned char bits = __nv_cvt_float2_to_fp4x2(src, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
        result[k/2] = bits;
    }
    return QuantResult{result, s_round_fp8, s_as_fp8};
}

__forceinline__ __device__ float quant_error(bf16x8& x, const QuantResult& q, float scale) {
    const float descale = static_cast<float>(q.fp8s) * scale;
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 dq = __nv_cvt_fp4x2_to_float2(q.bits[i]);
        float2 xv = {static_cast<float>(x[2*i+0]), static_cast<float>(x[2*i+1])};
        float2 d = {dq.x * descale - xv.x, dq.y * descale - xv.y};
        sum += d.x * d.x + d.y * d.y;
    }
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}

template<bool USE_FOUR_SIX = true>
__device__ __forceinline__ QuantResult quantize_four_six(float abs_max, float inv_scale_override, float scale, bf16x8& x) {
    if constexpr (!USE_FOUR_SIX) {
        return quantize_block(abs_max, 6.f * inv_scale_override, scale, x);
    }
    QuantResult r6 = quantize_block(abs_max, 6.f * inv_scale_override, scale, x);
    QuantResult r4 = quantize_block(abs_max, 4.f * inv_scale_override, scale, x);
    float e6 = quant_error(x, r6, scale);
    float e4 = quant_error(x, r4, scale);
    return (e4 < e6) ? r4 : r6;
}

__device__ __forceinline__ float vecReduceAbsMax(bf16x8& v) {
    float m = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) m = fmaxf(m, fabsf(static_cast<float>(v[k])));
    return m;
}

// -------------------------------------------------------------------------
// Pass 2: Quantization Kernel
// -------------------------------------------------------------------------
template<int BLOCK_SIZE=256, bool USE_FOUR_SIX=true>
__global__ void fused_quant_kernel(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    int rows, int cols,
    float inv_scale_override,
    const float* __restrict__ inv_rms_cache,   // Input: from Pass 1
    const float* __restrict__ global_amax_ptr, // Input: from Pass 1
    __nv_fp4x4_e2m1* __restrict__ y_ptr,
    __nv_fp8_e4m3* __restrict__ scale_ptr,
    float* __restrict__ global_scale_ptr       // Output: final calculated scale
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    const nv_bfloat16* row_ptr = x_ptr + row * cols;
    
    // 1. Read Globals once
    // Thread 0 computes and writes the global scale derived from global amax
    __shared__ float s_global_scale;
    if (tid == 0 && blockIdx.x == 0) {
        float global_max = *global_amax_ptr;
        constexpr float scales_max = USE_FOUR_SIX ? 256.f : 448.f;
        float val_max = 6.f * inv_scale_override;
        float scale = (global_max == 0) ? 1.f : global_max / scales_max / val_max;
        
        *global_scale_ptr = scale; // Write it out for host/others
        s_global_scale = scale;
    }
    // Optimization: Other blocks need to read global_scale.
    // We can't share s_global_scale across blocks. 
    // We MUST read *global_scale_ptr. But we just wrote it.
    // Wait, race condition? No, Pass 2 strictly follows Pass 1 (stream order).
    // BUT Pass 2 block 0 writes it. Other blocks in Pass 2 might read it before it's written?
    // YES. We have a problem. "global_scale_ptr" is derived from "global_amax_ptr".
    // 
    // Solution: ALL blocks should read "global_amax_ptr" and derive "scale" locally.
    // It's just a few ops.
    __shared__ float s_scale;
    if (tid == 0) {
        float global_max = *global_amax_ptr;
        constexpr float scales_max = USE_FOUR_SIX ? 256.f : 448.f;
        float val_max = 6.f * inv_scale_override;
        float scale = (global_max == 0) ? 1.f : global_max / scales_max / val_max;
        s_scale = scale;
        
        // Write out global_scale solely for the user/debug/GEMM if needed
        if (row == 0) *global_scale_ptr = scale; 
    }
    __syncthreads();
    float global_scale = s_scale;
    
    // Load per-row inv_rms
    float inv_rms = inv_rms_cache[row];

    // 2. Quantization Loop
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(row_ptr + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        // Apply RMSNorm + SiLU
        bf16x8 act_vec;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            float norm_val = val * inv_rms * w;
            float act_val = silu(norm_val);
            act_vec[k] = (nv_bfloat16)act_val;
        }
        
        // Compute local absmax for this 8-element group
        nv_bfloat16 vec_abs_max = vecReduceAbsMax(act_vec);
        
        // Get neighbor's absmax for 16-element block scaling
        nv_bfloat16 neighbor_abs_max = __shfl_xor_sync(0xffffffff, vec_abs_max, 1);
        float full_abs_max = static_cast<float>(__hmax(vec_abs_max, neighbor_abs_max));
        
        // Quantize
        QuantResult res = quantize_four_six<USE_FOUR_SIX>(full_abs_max, inv_scale_override, global_scale, act_vec);
        
        // Store
        int vec_idx = (row * cols + i) / 8;
        res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * vec_idx);
        
        if (vec_idx % 2 == 0) {
            scale_ptr[vec_idx / 2] = res.fp8s;
        }
    }
}


// -------------------------------------------------------------------------
// Host Launcher
// -------------------------------------------------------------------------
void launch_fused_rmsnorm_act_quant_2pass(
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
    float* global_amax_scratch // Needs 1 float, initialized to 0
) {
    constexpr int BLOCK_SIZE = 256;
    float inv_scale_override = 1.0f / scale_override;
    
    // Initialize global amax to 0
    cudaMemsetAsync(global_amax_scratch, 0, sizeof(float), 0);
    
    // Pass 1: Reduction
    // Grid dim: One block per row (assuming rows is reasonable, e.g. < 65535)
    // For very large BS (e.g. 1M rows), max Grid Y is 65535, X is 2^31.
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);
    
    fused_reduction_kernel<BLOCK_SIZE><<<grid, block, 0, 0>>>(
        x, w, epsilon, rows, cols, inv_rms_cache, global_amax_scratch
    );
    
    // Pass 2: Quantization
    if (use_four_six) {
        fused_quant_kernel<BLOCK_SIZE, true><<<grid, block, 0, 0>>>(
            x, w, rows, cols, inv_scale_override, 
            inv_rms_cache, global_amax_scratch, 
            y, scales, global_scale
        );
    } else {
         fused_quant_kernel<BLOCK_SIZE, false><<<grid, block, 0, 0>>>(
            x, w, rows, cols, inv_scale_override, 
            inv_rms_cache, global_amax_scratch, 
            y, scales, global_scale
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// Wrapper to allocate temporary global amax
void launch_fused_rmsnorm_act_quant_2pass(
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
) {
    float* global_amax;
    cudaMallocAsync(&global_amax, sizeof(float), 0);
    
    launch_fused_rmsnorm_act_quant_2pass(
        x, w, epsilon, rows, cols, scale_override, use_four_six,
        y, scales, global_scale, inv_rms_cache, global_amax
    );
    
    cudaFreeAsync(global_amax, 0);
}

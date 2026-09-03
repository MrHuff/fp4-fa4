// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Optimized Fused RMSNorm + SiLU Activation + FP4 Quantization Kernel V3
// 
// Key optimizations over V2:
// 1. Warp shuffle reductions instead of atomics for block absmax
// 2. Factor inv_rms into quantization scale (avoid per-element multiply)
//
// Mathematical insight:
// - block_amax = max|act(x) * gain| * inv_rms (computed in phase 1)
// - Instead of: x_quant = act(x) * inv_rms * gain, then quantize with block_amax
// - We do: x_raw = act(x) * gain, then pass inv_rms to quantize func to adjust factor
// - factor = inv_rms / (s_round_fp8 * scale)  instead of  1 / (s_round_fp8 * scale)
// - This saves one multiply per element in phase 2!

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

#include "vec.cuh"
#include "utils.cuh"

namespace cg = cooperative_groups;

using bf16x8 = GenericVector<nv_bfloat16, 8>;
using fp4x8 = GenericVector<unsigned char, 4>;

constexpr int BLOCK_GROUP_SIZE = 16;  // FP4 micro-scaling block size
constexpr int WARP_SIZE = 32;

// -------------------------------------------------------------------------
// Activation Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// -------------------------------------------------------------------------
// Warp-Level Reduction (no atomics!)
// -------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// -------------------------------------------------------------------------
// Block Reductions
// -------------------------------------------------------------------------

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    return BlockReduce(temp_storage).Sum(val);
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    struct MaxOp {
        __device__ __forceinline__ float operator()(float a, float b) const {
            return fmaxf(a, b);
        }
    };
    return BlockReduce(temp_storage).Reduce(val, MaxOp());
}

// -------------------------------------------------------------------------
// Quantization Logic with inv_rms factored into scale
// -------------------------------------------------------------------------

struct QuantResult {
    fp4x8 bits;
    float scale;
    __nv_fp8_e4m3 fp8s;
};

// MODIFIED: Takes inv_rms as parameter, factors it into the quantization
__device__ __forceinline__ QuantResult quantize_block_with_rms(
    float abs_max,       // Already scaled by inv_rms
    float val_max, 
    float scale,         // Global scale
    float inv_rms,       // To factor into quantization
    bf16x8& x            // x = act(input) * gain, NOT scaled by inv_rms
) {
    // Scale computation uses abs_max (which has inv_rms baked in)
    float s_group = abs_max / val_max;
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0) s_round_fp8 = 1.f;

    // KEY OPTIMIZATION: Factor inv_rms into the quantization factor!
    // This avoids multiplying each element by inv_rms in the data path
    float factor = inv_rms / (s_round_fp8 * scale);
    
    fp4x8 result;
    #pragma unroll
    for (int k = 0; k < bf16x8::size; k += 2) {
        float2 src;
        src.x = static_cast<float>(x[k+0]) * factor;  // x is NOT scaled by inv_rms
        src.y = static_cast<float>(x[k+1]) * factor;  // factor includes inv_rms
        unsigned char bits = __nv_cvt_float2_to_fp4x2(src, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
        result[k/2] = bits;
    }

    return QuantResult{result, s_round_fp8, s_as_fp8};
}

// Error computation for four-six search
__forceinline__ __device__ float quant_error_with_rms(bf16x8 x, const QuantResult& q, float scale, float inv_rms) {
    const float descale = static_cast<float>(q.fp8s) * scale;
    float2 sum = {0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 dq = __nv_cvt_fp4x2_to_float2(q.bits[i]);
        // Compare against x * inv_rms (the true normalized value)
        float2 xv = {static_cast<float>(x[2*i+0]) * inv_rms, static_cast<float>(x[2*i+1]) * inv_rms};
        float2 d;
        d.x = dq.x * descale - xv.x;
        d.y = dq.y * descale - xv.y;
        sum.x += d.x * d.x;
        sum.y += d.y * d.y;
    }
    float local_error = sum.x + sum.y;
    local_error += __shfl_xor_sync(0xffffffff, local_error, 1);
    return local_error;
}

// Four-six quantization with inv_rms factoring
template<bool USE_FOUR_SIX = true>
__device__ __forceinline__ QuantResult quantize_four_six_with_rms(
    float abs_max, float inv_scale_override, float scale, float inv_rms, bf16x8& x
) {
    if constexpr (!USE_FOUR_SIX) {
        return quantize_block_with_rms(abs_max, 6.f * inv_scale_override, scale, inv_rms, x);
    }
    
    QuantResult r6 = quantize_block_with_rms(abs_max, 6.f * inv_scale_override, scale, inv_rms, x);
    QuantResult r4 = quantize_block_with_rms(abs_max, 4.f * inv_scale_override, scale, inv_rms, x);
    
    float e6 = quant_error_with_rms(x, r6, scale, inv_rms);
    float e4 = quant_error_with_rms(x, r4, scale, inv_rms);
    
    return (e4 < e6) ? r4 : r6;
}

// -------------------------------------------------------------------------
// Main V3 Kernel: Warp shuffles + inv_rms factored into quantization
// -------------------------------------------------------------------------

template<int BLOCK_SIZE = 256, bool USE_FOUR_SIX = true>
__global__ void fused_rmsnorm_act_quant_kernel_v3(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float inv_scale_override,
    __nv_fp4x4_e2m1* __restrict__ y_ptr,
    __nv_fp8_e4m3* __restrict__ scale_ptr,
    float* __restrict__ global_scale_ptr,
    float* __restrict__ block_amax_scratch,   // [rows * num_blocks_per_row]
    float* __restrict__ inv_rms_cache
) {
    cg::grid_group grid = cg::this_grid();
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    
    // Shared memory layout:
    // - s_block_amax[num_blocks_per_row]: block absmaxes
    // - s_warp_sum[num_warps]: warp-level sum_sq partials
    extern __shared__ float smem[];
    float* s_block_amax = smem;
    float* s_warp_sum = smem + num_blocks_per_row;
    
    // Initialize block amax (one thread per block)
    for (int b = tid; b < num_blocks_per_row; b += BLOCK_SIZE) {
        s_block_amax[b] = 0.0f;
    }
    __syncthreads();
    
    // ===== PHASE 1: Compute sum_sq and block amax using warp shuffles =====
    
    float sum_sq = 0.0f;
    
    // Each thread processes contiguous 8-element vectors
    // We assign threads to 16-element blocks: 2 threads per block
    // Use warp shuffles to combine the two halves
    
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(x_ptr + row * cols + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        float local_max = 0.0f;
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            
            float act_val = silu(val);
            sum_sq += act_val * act_val;  // For RMSNorm (no gain)
            
            float act_with_gain = fabsf(act_val * w);  // With gain for absmax
            local_max = fmaxf(local_max, act_with_gain);
        }
        
        // Warp shuffle to combine with neighbor (16-element block spans 2 threads)
        int block_id = i / BLOCK_GROUP_SIZE;
        
        // Pair up consecutive threads to form 16-element blocks
        float neighbor_max = __shfl_xor_sync(0xffffffff, local_max, 1);
        float block_max = fmaxf(local_max, neighbor_max);
        
        // Only even lanes write (one write per 16-element block)
        if ((lane & 1) == 0) {
            // Within a warp, different threads may still target same block
            // Use warp-level reduction for threads targeting same block
            atomicMax(reinterpret_cast<int*>(&s_block_amax[block_id]), __float_as_int(block_max));
        }
    }
    __syncthreads();
    
    // Block-reduce sum_sq
    float row_sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    
    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
        inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    float inv_rms = s_inv_rms;
    
    // Scale block_amax by inv_rms and write to global scratch
    for (int b = tid; b < num_blocks_per_row; b += BLOCK_SIZE) {
        // Fix atomicMax result (it stored as int)
        float amax = __int_as_float(reinterpret_cast<int*>(s_block_amax)[b]);
        float block_amax_final = amax * inv_rms;
        block_amax_scratch[row * num_blocks_per_row + b] = block_amax_final;
    }
    
    // ===== GRID SYNC =====
    grid.sync();
    
    // ===== GLOBAL REDUCTION =====
    if (blockIdx.x == 0) {
        float global_max = 0.0f;
        int total_blocks = rows * num_blocks_per_row;
        for (int b = tid; b < total_blocks; b += BLOCK_SIZE) {
            global_max = fmaxf(global_max, block_amax_scratch[b]);
        }
        
        global_max = block_reduce_max<BLOCK_SIZE>(global_max);
        
        if (tid == 0) {
            constexpr float scales_max = USE_FOUR_SIX ? 256.f : 448.f;
            float val_max = 6.f * inv_scale_override;
            float scale = (global_max == 0) ? 1.f : global_max / scales_max / val_max;
            *global_scale_ptr = scale;
        }
    }
    
    // ===== GRID SYNC =====
    grid.sync();
    
    float global_scale = *global_scale_ptr;
    
    // ===== PHASE 2: Quantize with inv_rms factored into scale =====
    
    // Reload block absmaxes
    for (int b = tid; b < num_blocks_per_row; b += BLOCK_SIZE) {
        s_block_amax[b] = block_amax_scratch[row * num_blocks_per_row + b];
    }
    __syncthreads();
    
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(x_ptr + row * cols + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        // Compute x_raw = act(x) * gain (NO inv_rms multiply!)
        bf16x8 x_raw;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            float act_val = silu(val);
            x_raw[k] = (nv_bfloat16)(act_val * w);  // NO inv_rms here!
        }
        
        int block_id = i / BLOCK_GROUP_SIZE;
        float block_amax = s_block_amax[block_id];
        
        // Quantize with inv_rms factored into the quantization factor
        QuantResult res = quantize_four_six_with_rms<USE_FOUR_SIX>(
            block_amax, inv_scale_override, global_scale, inv_rms, x_raw
        );
        
        // Store
        int vec_idx = (row * cols + i) / 8;
        res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * vec_idx);
        
        if ((i / 8) % 2 == 0) {
            scale_ptr[vec_idx / 2] = res.fp8s;
        }
    }
}

// -------------------------------------------------------------------------
// Host Launcher
// -------------------------------------------------------------------------

void launch_fused_rmsnorm_act_quant_v3(
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
    float* block_amax_scratch
) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    
    float inv_scale_override = 1.0f / scale_override;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    
    // Shared memory: block_amax + warp_sum
    size_t smem_size = (num_blocks_per_row + NUM_WARPS) * sizeof(float);
    
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("Device does not support cooperative launch");
    }
    
    int max_blocks_per_sm;
    if (use_four_six) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            fused_rmsnorm_act_quant_kernel_v3<BLOCK_SIZE, true>,
            BLOCK_SIZE, smem_size
        );
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            fused_rmsnorm_act_quant_kernel_v3<BLOCK_SIZE, false>,
            BLOCK_SIZE, smem_size
        );
    }
    int max_coop_blocks = max_blocks_per_sm * prop.multiProcessorCount;
    
    if (rows > max_coop_blocks) {
        throw std::runtime_error("Grid size exceeds cooperative launch limit.");
    }
    
    void* args[] = {
        (void*)&x, (void*)&w, (void*)&epsilon,
        (void*)&rows, (void*)&cols, (void*)&inv_scale_override,
        (void*)&y, (void*)&scales, (void*)&global_scale,
        (void*)&block_amax_scratch, (void*)&inv_rms_cache
    };
    
    if (use_four_six) {
        cudaLaunchCooperativeKernel(
            (void*)fused_rmsnorm_act_quant_kernel_v3<BLOCK_SIZE, true>,
            dim3(rows), dim3(BLOCK_SIZE),
            args, smem_size, nullptr
        );
    } else {
        cudaLaunchCooperativeKernel(
            (void*)fused_rmsnorm_act_quant_kernel_v3<BLOCK_SIZE, false>,
            dim3(rows), dim3(BLOCK_SIZE),
            args, smem_size, nullptr
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// Convenience wrapper
void launch_fused_rmsnorm_act_quant_v3(
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
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    float* block_amax_scratch;
    cudaMallocAsync(&block_amax_scratch, rows * num_blocks_per_row * sizeof(float), 0);
    
    launch_fused_rmsnorm_act_quant_v3(
        x, w, epsilon, rows, cols, scale_override, use_four_six,
        y, scales, global_scale, inv_rms_cache, block_amax_scratch
    );
    
    cudaFreeAsync(block_amax_scratch, 0);
}

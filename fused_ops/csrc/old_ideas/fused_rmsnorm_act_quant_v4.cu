// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + SiLU Activation + FP4 Quantization Kernel V4
// 
// FULLY LOCK-FREE DESIGN:
// - Each thread owns exclusive 16-element blocks (no atomics at all!)
// - Trade-off: 1 thread processes 16 elements (less parallelism)
// - But: zero contention, no atomics, pure warp-level operations
//
// Key insight: For cols=4096, we have 256 blocks per row
// With 256 threads per block, each thread owns exactly 1 block!
// Perfect assignment: thread_i owns block_i

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
// Quantization with inv_rms factored into scale
// -------------------------------------------------------------------------

struct QuantResult {
    fp4x8 bits;
    float scale;
    __nv_fp8_e4m3 fp8s;
};

// Takes 16 elements (as 2 bf16x8 vectors), returns quantized result
__device__ __forceinline__ QuantResult quantize_block16_with_rms(
    float abs_max,       // Already scaled by inv_rms
    float val_max, 
    float scale,         // Global scale
    float inv_rms,       // To factor into quantization
    bf16x8& x0,          // First 8 elements  (act(input) * gain, NOT scaled by inv_rms)
    bf16x8& x1           // Second 8 elements
) {
    // Scale computation uses abs_max (which has inv_rms baked in)
    float s_group = abs_max / val_max;
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0) s_round_fp8 = 1.f;

    // Factor inv_rms into quantization factor
    float factor = inv_rms / (s_round_fp8 * scale);
    
    fp4x8 result;
    
    // First 8 elements (from x0)
    #pragma unroll
    for (int k = 0; k < 8; k += 2) {
        float2 src;
        src.x = static_cast<float>(x0[k+0]) * factor;
        src.y = static_cast<float>(x0[k+1]) * factor;
        unsigned char bits = __nv_cvt_float2_to_fp4x2(src, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
        result[k/2] = bits;
    }
    
    // Second 8 elements (from x1) - but we don't need them for the first 8-element result
    // Actually, for 16-element block we should output 8 bytes (16 FP4 values)
    // Let's do this properly in the kernel
    
    return QuantResult{result, s_round_fp8, s_as_fp8};
}

// Alternative: quantize a single bf16x8 with known absmax
__device__ __forceinline__ QuantResult quantize_block8_with_rms(
    float abs_max, float val_max, float scale, float inv_rms, bf16x8& x
) {
    float s_group = abs_max / val_max;
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0) s_round_fp8 = 1.f;

    float factor = inv_rms / (s_round_fp8 * scale);
    
    fp4x8 result;
    #pragma unroll
    for (int k = 0; k < 8; k += 2) {
        float2 src;
        src.x = static_cast<float>(x[k+0]) * factor;
        src.y = static_cast<float>(x[k+1]) * factor;
        unsigned char bits = __nv_cvt_float2_to_fp4x2(src, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
        result[k/2] = bits;
    }

    return QuantResult{result, s_round_fp8, s_as_fp8};
}

// Error computation for four-six search
__forceinline__ __device__ float quant_error_8(bf16x8& x, const QuantResult& q, float scale, float inv_rms) {
    const float descale = static_cast<float>(q.fp8s) * scale;
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 dq = __nv_cvt_fp4x2_to_float2(q.bits[i]);
        float x0_norm = static_cast<float>(x[2*i+0]) * inv_rms;
        float x1_norm = static_cast<float>(x[2*i+1]) * inv_rms;
        float d0 = dq.x * descale - x0_norm;
        float d1 = dq.y * descale - x1_norm;
        sum += d0 * d0 + d1 * d1;
    }
    return sum;
}

// -------------------------------------------------------------------------
// V4 Kernel: Lock-free with exclusive block ownership
// -------------------------------------------------------------------------

template<int BLOCK_SIZE = 256, bool USE_FOUR_SIX = true>
__global__ void fused_rmsnorm_act_quant_kernel_v4(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float inv_scale_override,
    __nv_fp4x4_e2m1* __restrict__ y_ptr,
    __nv_fp8_e4m3* __restrict__ scale_ptr,
    float* __restrict__ global_scale_ptr,
    float* __restrict__ block_amax_scratch,
    float* __restrict__ inv_rms_cache
) {
    cg::grid_group grid = cg::this_grid();
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    
    // ===== PHASE 1: Each thread computes sum_sq and owns its block's absmax =====
    
    float sum_sq = 0.0f;
    float my_block_amax = 0.0f;  // Each thread owns one 16-element block
    
    // Thread tid owns block tid (when num_blocks_per_row == BLOCK_SIZE)
    // For other cases, threads may own multiple blocks
    
    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        int elem_start = block_id * BLOCK_GROUP_SIZE;
        
        // Load the 16 elements for this block (as 2 bf16x8 vectors)
        bf16x8 data0 = bf16x8::load(x_ptr + row * cols + elem_start);
        bf16x8 data1 = bf16x8::load(x_ptr + row * cols + elem_start + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + elem_start);
        bf16x8 w1 = bf16x8::load(w_ptr + elem_start + 8);
        
        float block_max = 0.0f;
        
        // Process first 8 elements
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data0[k]);
            float w = static_cast<float>(w0[k]);
            float act_val = silu(val);
            sum_sq += act_val * act_val;
            block_max = fmaxf(block_max, fabsf(act_val * w));
        }
        
        // Process second 8 elements
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data1[k]);
            float w = static_cast<float>(w1[k]);
            float act_val = silu(val);
            sum_sq += act_val * act_val;
            block_max = fmaxf(block_max, fabsf(act_val * w));
        }
        
        // Save this block's absmax (for later use)
        my_block_amax = fmaxf(my_block_amax, block_max);
    }
    
    // Now we have sum_sq distributed across threads
    // Each thread also has the max absmax of the blocks it owns
    
    // Block-reduce sum_sq
    float row_sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    
    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
        inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    float inv_rms = s_inv_rms;
    
    // Scale my_block_amax by inv_rms
    my_block_amax *= inv_rms;
    
    // Find row-max (for global scale computation)
    float row_amax = block_reduce_max<BLOCK_SIZE>(my_block_amax);
    
    // Write each thread's block amax to scratch  
    // (NO ATOMICS - each thread owns its blocks exclusively)
    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        // Recompute this block's absmax (we need to track per-block, not aggregate)
        int elem_start = block_id * BLOCK_GROUP_SIZE;
        bf16x8 data0 = bf16x8::load(x_ptr + row * cols + elem_start);
        bf16x8 data1 = bf16x8::load(x_ptr + row * cols + elem_start + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + elem_start);
        bf16x8 w1 = bf16x8::load(w_ptr + elem_start + 8);
        
        float block_max = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data0[k]);
            float w = static_cast<float>(w0[k]);
            float act_val = silu(val);
            block_max = fmaxf(block_max, fabsf(act_val * w));
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data1[k]);
            float w = static_cast<float>(w1[k]);
            float act_val = silu(val);
            block_max = fmaxf(block_max, fabsf(act_val * w));
        }
        
        block_amax_scratch[row * num_blocks_per_row + block_id] = block_max * inv_rms;
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
    
    // ===== PHASE 2: Quantize (each thread owns its blocks exclusively) =====
    
    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        int elem_start = block_id * BLOCK_GROUP_SIZE;
        
        // Reload data
        bf16x8 data0 = bf16x8::load(x_ptr + row * cols + elem_start);
        bf16x8 data1 = bf16x8::load(x_ptr + row * cols + elem_start + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + elem_start);
        bf16x8 w1 = bf16x8::load(w_ptr + elem_start + 8);
        
        // Compute x_raw = act(x) * gain (NO inv_rms multiply)
        bf16x8 x_raw0, x_raw1;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data0[k]);
            float w = static_cast<float>(w0[k]);
            x_raw0[k] = (nv_bfloat16)(silu(val) * w);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data1[k]);
            float w = static_cast<float>(w1[k]);
            x_raw1[k] = (nv_bfloat16)(silu(val) * w);
        }
        
        float block_amax = block_amax_scratch[row * num_blocks_per_row + block_id];
        
        // Quantize both halves with same scale (same 16-element block)
        float val_max = 6.f * inv_scale_override;
        
        if constexpr (USE_FOUR_SIX) {
            // Try val_max=6 and val_max=4, pick lower error
            QuantResult r6_0 = quantize_block8_with_rms(block_amax, 6.f * inv_scale_override, global_scale, inv_rms, x_raw0);
            QuantResult r4_0 = quantize_block8_with_rms(block_amax, 4.f * inv_scale_override, global_scale, inv_rms, x_raw0);
            float e6_0 = quant_error_8(x_raw0, r6_0, global_scale, inv_rms);
            float e4_0 = quant_error_8(x_raw0, r4_0, global_scale, inv_rms);
            
            QuantResult r6_1 = quantize_block8_with_rms(block_amax, 6.f * inv_scale_override, global_scale, inv_rms, x_raw1);
            QuantResult r4_1 = quantize_block8_with_rms(block_amax, 4.f * inv_scale_override, global_scale, inv_rms, x_raw1);
            float e6_1 = quant_error_8(x_raw1, r6_1, global_scale, inv_rms);
            float e4_1 = quant_error_8(x_raw1, r4_1, global_scale, inv_rms);
            
            // Total error for the 16-element block
            float e6_total = e6_0 + e6_1;
            float e4_total = e4_0 + e4_1;
            
            QuantResult res0 = (e4_total < e6_total) ? r4_0 : r6_0;
            QuantResult res1 = (e4_total < e6_total) ? r4_1 : r6_1;
            
            // Store
            int vec_idx0 = (row * cols + elem_start) / 8;
            int vec_idx1 = vec_idx0 + 1;
            res0.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * vec_idx0);
            res1.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * vec_idx1);
            scale_ptr[block_id + row * num_blocks_per_row] = res0.fp8s;  // Same scale for both halves
        } else {
            QuantResult res0 = quantize_block8_with_rms(block_amax, val_max, global_scale, inv_rms, x_raw0);
            QuantResult res1 = quantize_block8_with_rms(block_amax, val_max, global_scale, inv_rms, x_raw1);
            
            int vec_idx0 = (row * cols + elem_start) / 8;
            int vec_idx1 = vec_idx0 + 1;
            res0.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * vec_idx0);
            res1.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * vec_idx1);
            scale_ptr[block_id + row * num_blocks_per_row] = res0.fp8s;
        }
    }
}

// -------------------------------------------------------------------------
// Host Launcher
// -------------------------------------------------------------------------

void launch_fused_rmsnorm_act_quant_v4(
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
    
    float inv_scale_override = 1.0f / scale_override;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    
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
            fused_rmsnorm_act_quant_kernel_v4<BLOCK_SIZE, true>,
            BLOCK_SIZE, 0
        );
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            fused_rmsnorm_act_quant_kernel_v4<BLOCK_SIZE, false>,
            BLOCK_SIZE, 0
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
            (void*)fused_rmsnorm_act_quant_kernel_v4<BLOCK_SIZE, true>,
            dim3(rows), dim3(BLOCK_SIZE),
            args, 0, nullptr
        );
    } else {
        cudaLaunchCooperativeKernel(
            (void*)fused_rmsnorm_act_quant_kernel_v4<BLOCK_SIZE, false>,
            dim3(rows), dim3(BLOCK_SIZE),
            args, 0, nullptr
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// Convenience wrapper
void launch_fused_rmsnorm_act_quant_v4(
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
    
    launch_fused_rmsnorm_act_quant_v4(
        x, w, epsilon, rows, cols, scale_override, use_four_six,
        y, scales, global_scale, inv_rms_cache, block_amax_scratch
    );
    
    cudaFreeAsync(block_amax_scratch, 0);
}

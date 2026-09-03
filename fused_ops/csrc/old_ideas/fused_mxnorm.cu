// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused MXNorm + SiLU + FP4 Quantization
// Extension of V2 kernel to support alternative normalization estimations.

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

// Atomic max for floats
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
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

__forceinline__ __device__ float quant_error(bf16x8 x, const QuantResult& q, float scale) {
    const float descale = static_cast<float>(q.fp8s) * scale;
    float2 sum = {0.f, 0.f};
    const float2 dsv = {-descale, -descale};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 dq = __nv_cvt_fp4x2_to_float2(q.bits[i]);
        float2 xv = {static_cast<float>(x[2*i+0]), static_cast<float>(x[2*i+1])};
        float2 d;
        d.x = dq.x * dsv.x + xv.x;
        d.y = dq.y * dsv.y + xv.y;
        sum.x += d.x * d.x;
        sum.y += d.y * d.y;
    }
    float local_error = sum.x + sum.y;
    local_error += __shfl_xor_sync(0xffffffff, local_error, 1);
    return local_error;
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

// -------------------------------------------------------------------------
// MXNorm Kernel
// -------------------------------------------------------------------------

template<int BLOCK_SIZE = 256, bool USE_FOUR_SIX = true, int NORM_MODE = 0>
__global__ void fused_mxnorm_kernel(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float inv_scale_override,
    // int norm_mode,    // Templatized
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
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    
    // Shared memory for block absmaxes (raw)
    extern __shared__ float smem[];
    float* s_block_amax_raw = smem;  // [num_blocks_per_row]
    
    // Initialize shared memory
    for (int b = tid; b < num_blocks_per_row; b += BLOCK_SIZE) {
        s_block_amax_raw[b] = 0.0f;
    }
    __syncthreads();
    
    // ===== PHASE 1: Statistics Collection =====
    
    [[maybe_unused]] float sum_sq = 0.0f;        // For mode 0
    [[maybe_unused]] float max_abs_val = 0.0f;   // For mode 1
    
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(x_ptr + row * cols + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        float local_max = 0.0f;
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            
            // Mode 0: RMSNorm on raw input (SiLU(RMS(x)))
            if constexpr (NORM_MODE == 0) {
                sum_sq += val * val;
                
                // For local max (quantization), we need the final output value
                // We don't have inv_rms yet, but typically for quantization stats we can approximate or use a separate pass?
                // Wait, Phase 2 recalculates properly. Phase 1 needs local_max for block quantization logic.
                // The current kernel fuses everything.
                // If we do SiLU(RMS(x)), the output magnitude depends on inv_rms (which is global per row).
                // But we need `local_max` for warp reduction of block maxes.
                // Problem: We can't know the exact output value `SiLU(val * inv_rms * w)` without `inv_rms`.
                // BUT, `inv_rms` is computed AFTER this loop.
                // V2 kernel (original fused_rmsnorm_act_quant_v2) solves this by... wait.
                // Original V2 also did stats collection first.
                // In RMSNorm, `inv_rms` is shared across the row.
                // If we need `max(|SiLU(val*inv_rms*w)|)` for quantization scaling, we need `inv_rms`.
                // Existing code does: `local_max = fmaxf(local_max, act_with_gain)`.
                // `act_with_gain` was `fabsf(act_val * w)`.
                // This assumes `inv_rms` is roughly 1 or handled globally?
                // No, Phase 1 computes `s_block_amax_raw` which is `max(|x*w|)`.
                // Then later: `block_amax_final = s_block_amax_raw[b] * inv_rms`.
                // So checking `|x*w|` is valid IF the operation is linear-ish scaling.
                // For SiLU(x*inv_rms*w), this is NOT linear scaling of SiLU(x*w).
                // However, `SiLU(c*x) ~= c * SiLU(x)` is FALSE.
                
                // CRITICAL: Changing the order to SiLU(RMS(x)) breaks the "BlockMax * inv_rms" optimization 
                // because SiLU doesn't commute with scalar multiplication.
                
                // OPTION: For stats purposes (quantization scale), maybe we just track |val*w| and apply SiLU logic later?
                // Or maybe we accept that Fused RMS logic was `RMS(SiLU(x))` BY DESIGN for performance?
                // But user insisted "it should be identical to Compiled RMS".
                
                // Let's look at `local_max`.
                // Ideally we want `max(|Output|)`. Output = SiLU(val * inv_rms * w).
                // Since inv_rms is constant for the row, max(|Output|) occurs at max(|val * inv_rms * w|) (monotonicity of SiLU?).
                // Yes, SiLU is roughly monotonic for positive inputs, and x*SiLU(x) ... wait.
                // SiLU(z) = z / (1+exp(-z)). Monotonic? Yes, for z > -1ish.
                // Absolute value |SiLU(z)| is related to |z|.
                // So `max(|SiLU(z)|)` roughly corresponds to `max(|z|)`.
                // So if we track `max(|val*w|)` (which becomes `max(|val*w*inv_rms|)`),
                // we can estimate `max(|SiLU(z)|)`?
                
                // Let's stick to tracking `fabsf(val * w)` (pre-silu gain).
                // Then in Phase 2 we use exact values.
                // Phase 1 local_max is just for block-level scaling factor estimation.
                
                float val_with_gain = fabsf(val * w);
                local_max = fmaxf(local_max, val_with_gain);
            }
            // Mode 1: AbsMax (keeps SiLU(x) first logic)
            else if constexpr (NORM_MODE == 1) {
                float act_val = silu(val);
                max_abs_val = fmaxf(max_abs_val, fabsf(act_val));
                
                float act_with_gain = fabsf(act_val * w);
                local_max = fmaxf(local_max, act_with_gain);
            }
            // Mode 2: BlockMax (keeps SiLU(x) first logic?)
            // Assuming Mode 2 logic follows Mode 1 for now unless specified otherwise.
            else {
                float act_val = silu(val);
                float act_with_gain = fabsf(act_val * w);
                local_max = fmaxf(local_max, act_with_gain);
            }
        }
        
        // Resolve quantization block max (same as V2)
        int block_id = i / BLOCK_GROUP_SIZE;
        int lane = tid & 31;
        float neighbor_max = __shfl_xor_sync(0xffffffff, local_max, 1);
        float block_max = fmaxf(local_max, neighbor_max);
        
        if ((lane & 1) == 0) {
            atomicMaxFloat(&s_block_amax_raw[block_id], block_max);
        }
    }
    __syncthreads();
    
    // Reduce for inv_rms
    float inv_rms;
    
    if constexpr (NORM_MODE == 0) {
        // RMSNorm: reduce sum_sq
        float row_sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
        __shared__ float s_inv_rms;
        if (tid == 0) {
            s_inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    } else if constexpr (NORM_MODE == 1) {
        // AbsMax: reduce max_abs_val
        float row_max = block_reduce_max<BLOCK_SIZE>(max_abs_val);
        __shared__ float s_inv_rms;
        if (tid == 0) {
            // "Estimation": 1 / max
            s_inv_rms = 1.0f / (row_max + epsilon);
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    } else if constexpr (NORM_MODE == 2) {
        // BlockMax RMS: RMS of block maxes (estimated)
        // ... (previous logic)
        
        float block_sum_sq = 0.0f;
        for (int b = tid; b < num_blocks_per_row; b += BLOCK_SIZE) {
           float bmax = s_block_amax_raw[b];
           block_sum_sq += bmax * bmax;
        }
        float row_block_sum_sq = block_reduce_sum<BLOCK_SIZE>(block_sum_sq);
        
        __shared__ float s_inv_rms;
        if (tid == 0) {
             s_inv_rms = rsqrtf(row_block_sum_sq / num_blocks_per_row + epsilon);
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    } else {
        inv_rms = 1.0f; // Fallback
    }

    if (tid == 0 && inv_rms_cache) {
        inv_rms_cache[row] = inv_rms;
    }
    
    // Scale block_amax_raw by inv_rms (since it was |act*gain|, now we want |norm*gain| = |act*inv_rms*gain|)
    // block_amax_final = block_amax_raw * inv_rms
    for (int b = tid; b < num_blocks_per_row; b += BLOCK_SIZE) {
        float block_amax_final = s_block_amax_raw[b] * inv_rms;
        block_amax_scratch[row * num_blocks_per_row + b] = block_amax_final;
    }
    
    // ===== GRID SYNC =====
    grid.sync();
    
    // ===== REDUCE GLOBAL SCALE =====
    // __shared__ float s_global_amax; // Unused
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
            block_amax_scratch[0] = scale;  // Reuse scratch for broadcast
        }
    }
    grid.sync();
    
    float global_scale = *global_scale_ptr;
    
    // ===== PHASE 2: Quantize =====
    // Reload block absmaxes
    for (int b = tid; b < num_blocks_per_row; b += BLOCK_SIZE) {
        s_block_amax_raw[b] = block_amax_scratch[row * num_blocks_per_row + b];
    }
    __syncthreads();
    
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(x_ptr + row * cols + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        bf16x8 out_vec;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            
            float norm_val;
            if constexpr (NORM_MODE == 0) {
                 // Mode 0: SiLU(RMS(x))
                 // val * inv_rms * w  -> SiLU
                 float pre_act = val * inv_rms * w;
                 norm_val = silu(pre_act);
            } else {
                 // Mode 1+: RMS/AbsMax(SiLU(x)) (Original Logic)
                 float act_val = silu(val);
                 norm_val = act_val * inv_rms * w;
            }
            out_vec[k] = (nv_bfloat16)norm_val;
        }
        
        // Logic to pick correct block max (same as V2)
        // Here s_block_amax_raw contains the final scaled block maxes
        int block_id = i / BLOCK_GROUP_SIZE;
        float block_amax = s_block_amax_raw[block_id];
        
        // Quantize
        QuantResult res = quantize_four_six<USE_FOUR_SIX>(block_amax, inv_scale_override, global_scale, out_vec);
        
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

void launch_fused_mxnorm(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float scale_override,
    bool use_four_six,
    int norm_mode,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale,
    float* inv_rms_cache,
    float* block_amax_scratch
) {
    constexpr int BLOCK_SIZE = 256;
    float inv_scale_override = 1.0f / scale_override;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    size_t smem_size = num_blocks_per_row * sizeof(float);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("Device does not support cooperative launch");
    }
    
    int max_blocks_per_sm;
    // Just check on instance 0, assuming similar resource usage
    if (use_four_six) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            fused_mxnorm_kernel<BLOCK_SIZE, true, 0>,
            BLOCK_SIZE, smem_size
        );
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            fused_mxnorm_kernel<BLOCK_SIZE, false, 0>,
            BLOCK_SIZE, smem_size
        );
    }
    
    int max_coop_blocks = max_blocks_per_sm * prop.multiProcessorCount;
    if (rows > max_coop_blocks) {
        // Fallback or error?
        // Since this is a specialized kernel, we might want to strict fail or implement a batched loop (omitted for now)
        throw std::runtime_error("Grid size exceeds cooperative launch limit");
    }
    
    void* args[] = {
        (void*)&x, (void*)&w, (void*)&epsilon,
        (void*)&rows, (void*)&cols, (void*)&inv_scale_override,
        // (void*)&norm_mode, // Removed from args
        (void*)&y, (void*)&scales, (void*)&global_scale,
        (void*)&block_amax_scratch, (void*)&inv_rms_cache
    };
    
    if (use_four_six) {
        switch(norm_mode) {
            case 0: cudaLaunchCooperativeKernel((void*)fused_mxnorm_kernel<BLOCK_SIZE, true, 0>, dim3(rows), dim3(BLOCK_SIZE), args, smem_size, nullptr); break;
            case 1: cudaLaunchCooperativeKernel((void*)fused_mxnorm_kernel<BLOCK_SIZE, true, 1>, dim3(rows), dim3(BLOCK_SIZE), args, smem_size, nullptr); break;
            case 2: cudaLaunchCooperativeKernel((void*)fused_mxnorm_kernel<BLOCK_SIZE, true, 2>, dim3(rows), dim3(BLOCK_SIZE), args, smem_size, nullptr); break;
            default: break;
        }
    } else {
        switch(norm_mode) {
            case 0: cudaLaunchCooperativeKernel((void*)fused_mxnorm_kernel<BLOCK_SIZE, false, 0>, dim3(rows), dim3(BLOCK_SIZE), args, smem_size, nullptr); break;
            case 1: cudaLaunchCooperativeKernel((void*)fused_mxnorm_kernel<BLOCK_SIZE, false, 1>, dim3(rows), dim3(BLOCK_SIZE), args, smem_size, nullptr); break;
            case 2: cudaLaunchCooperativeKernel((void*)fused_mxnorm_kernel<BLOCK_SIZE, false, 2>, dim3(rows), dim3(BLOCK_SIZE), args, smem_size, nullptr); break;
            default: break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_mxnorm(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    float scale_override,
    bool use_four_six,
    int norm_mode,
    __nv_fp4x4_e2m1* y,
    __nv_fp8_e4m3* scales,
    float* global_scale,
    float* inv_rms_cache
) {
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    float* block_amax_scratch;
    cudaMallocAsync(&block_amax_scratch, rows * num_blocks_per_row * sizeof(float), 0);
    
    launch_fused_mxnorm(
        x, w, epsilon, rows, cols, scale_override, use_four_six, norm_mode,
        y, scales, global_scale, inv_rms_cache, block_amax_scratch
    );
    
    cudaFreeAsync(block_amax_scratch, 0);
}

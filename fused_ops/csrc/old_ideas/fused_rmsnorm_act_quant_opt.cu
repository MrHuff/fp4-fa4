// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Optimized Fused RMSNorm + SiLU Activation + FP4 Quantization Kernel
// 
// V1-OPT: Micro-optimizations on V1 structure
// - Maintain V1's optimal structure (no atomics, minimal storage)
// - Reduce redundant operations where possible
// - Better register utilization
//
// Note: Since SiLU is nonlinear, we can't factor inv_rms into quantization.
// silu(x * inv_rms) != silu(x) * inv_rms

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

#include "vec.cuh"
#include "utils.cuh"

namespace cg = cooperative_groups;

using bf16x8 = GenericVector<nv_bfloat16, 8>;
using fp4x8 = GenericVector<unsigned char, 4>;

constexpr int BLOCK_GROUP_SIZE = 16;

// -------------------------------------------------------------------------
// Activation and Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Fast silu approximation (optional, can test later)
__device__ __forceinline__ float silu_fast(float x) {
    // Approximation: x * sigmoid(x) ≈ x / (1 + exp(-x))
    // Using fast sigmoid: x * 0.5f * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)))
    return x / (1.0f + __expf(-x));  // __expf is faster than expf
}

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

__forceinline__ __device__ float quant_error(bf16x8& x, const QuantResult& q, float scale) {
    const float descale = static_cast<float>(q.fp8s) * scale;
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 dq = __nv_cvt_fp4x2_to_float2(q.bits[i]);
        float x0 = static_cast<float>(x[2*i+0]);
        float x1 = static_cast<float>(x[2*i+1]);
        float d0 = dq.x * descale - x0;
        float d1 = dq.y * descale - x1;
        sum += d0 * d0 + d1 * d1;
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

// Helper to reduce 8-element vector to absmax
__device__ __forceinline__ float vec8_absmax(bf16x8& v) {
    float m = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        m = fmaxf(m, fabsf(static_cast<float>(v[k])));
    }
    return m;
}

// -------------------------------------------------------------------------
// Main V1-OPT Kernel
// -------------------------------------------------------------------------

template<int BLOCK_SIZE = 256, bool USE_FOUR_SIX = true>
__global__ void fused_rmsnorm_act_quant_kernel_opt(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float inv_scale_override,
    __nv_fp4x4_e2m1* __restrict__ y_ptr,
    __nv_fp8_e4m3* __restrict__ scale_ptr,
    float* __restrict__ global_scale_ptr,
    float* __restrict__ row_amax_scratch,
    float* __restrict__ inv_rms_cache
) {
    cg::grid_group grid = cg::this_grid();
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    const nv_bfloat16* row_ptr = x_ptr + row * cols;
    
    // ===== PHASE 1A: Compute sum_sq for RMSNorm =====
    float sum_sq = 0.0f;
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(row_ptr + i);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            sum_sq += val * val;
        }
    }
    
    float row_sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    
    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
        inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    float inv_rms = s_inv_rms;
    
    // ===== PHASE 1B: Apply RMSNorm + SiLU, compute row absmax =====
    float local_max = 0.0f;
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(row_ptr + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            // Using __expf for faster computation
            float norm_val = val * inv_rms * w;
            float act_val = norm_val / (1.0f + __expf(-norm_val));
            local_max = fmaxf(local_max, fabsf(act_val));
        }
    }
    
    float row_max = block_reduce_max<BLOCK_SIZE>(local_max);
    if (tid == 0) {
        row_amax_scratch[row] = row_max;
    }
    
    // ===== GRID SYNC =====
    grid.sync();
    
    // ===== GLOBAL REDUCTION =====
    __shared__ float s_global_amax;
    if (blockIdx.x == 0 && tid == 0) {
        float global_max = 0.0f;
        for (int r = 0; r < rows; ++r) {
            global_max = fmaxf(global_max, row_amax_scratch[r]);
        }
        row_amax_scratch[0] = global_max;
        
        constexpr float scales_max = USE_FOUR_SIX ? 256.f : 448.f;
        float val_max = 6.f * inv_scale_override;
        float scale = (global_max == 0) ? 1.f : global_max / scales_max / val_max;
        *global_scale_ptr = scale;
    }
    
    // ===== GRID SYNC =====
    grid.sync();
    
    float global_scale = *global_scale_ptr;
    
    // ===== PHASE 2: Quantize =====
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 data = bf16x8::load(row_ptr + i);
        bf16x8 w_data = bf16x8::load(w_ptr + i);
        
        // Apply RMSNorm + SiLU
        bf16x8 act_vec;
        float local_amax = 0.f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = static_cast<float>(data[k]);
            float w = static_cast<float>(w_data[k]);
            float norm_val = val * inv_rms * w;
            float act_val = norm_val / (1.0f + __expf(-norm_val));
            act_vec[k] = (nv_bfloat16)act_val;
            local_amax = fmaxf(local_amax, fabsf(act_val));
        }
        
        // Get neighbor's absmax for 16-element block
        float neighbor_amax = __shfl_xor_sync(0xffffffff, local_amax, 1);
        float block_amax = fmaxf(local_amax, neighbor_amax);
        
        // Quantize
        QuantResult res = quantize_four_six<USE_FOUR_SIX>(block_amax, inv_scale_override, global_scale, act_vec);
        
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

void launch_fused_rmsnorm_act_quant_opt(
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
    float* row_amax_scratch
) {
    constexpr int BLOCK_SIZE = 256;
    float inv_scale_override = 1.0f / scale_override;
    
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
            fused_rmsnorm_act_quant_kernel_opt<BLOCK_SIZE, true>,
            BLOCK_SIZE, 0
        );
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            fused_rmsnorm_act_quant_kernel_opt<BLOCK_SIZE, false>,
            BLOCK_SIZE, 0
        );
    }
    int max_coop_blocks = max_blocks_per_sm * prop.multiProcessorCount;
    
    if (rows > max_coop_blocks) {
        // Process in batches (Copied from fused_rmsnorm_act_quant.cu)
        int batch_start = 0;
        while (batch_start < rows) {
            int batch_rows = std::min(max_coop_blocks, rows - batch_start);
            
            // Offset pointers for this batch
            const nv_bfloat16* x_batch = x + batch_start * cols;
            __nv_fp4x4_e2m1* y_batch = y + (batch_start * cols / 8);
            __nv_fp8_e4m3* scales_batch = scales + (batch_start * cols / 16);
            float* inv_rms_batch = inv_rms_cache + batch_start;
            float* scratch_batch = row_amax_scratch + batch_start;
            
            void* args[] = {
                (void*)&x_batch, (void*)&w, (void*)&epsilon,
                (void*)&batch_rows, (void*)&cols, (void*)&inv_scale_override,
                (void*)&y_batch, (void*)&scales_batch, (void*)&global_scale,
                (void*)&scratch_batch, (void*)&inv_rms_batch
            };
            
            if (use_four_six) {
                cudaLaunchCooperativeKernel(
                    (void*)fused_rmsnorm_act_quant_kernel_opt<BLOCK_SIZE, true>,
                    dim3(batch_rows), dim3(BLOCK_SIZE),
                    args, 0, nullptr
                );
            } else {
                cudaLaunchCooperativeKernel(
                    (void*)fused_rmsnorm_act_quant_kernel_opt<BLOCK_SIZE, false>,
                    dim3(batch_rows), dim3(BLOCK_SIZE),
                    args, 0, nullptr
                );
            }
            CUDA_CHECK(cudaGetLastError());
            
            batch_start += batch_rows;
        }
        return;
    }
    
    // Normal path: fits in single cooperative launch
    void* args[] = {
        (void*)&x, (void*)&w, (void*)&epsilon,
        (void*)&rows, (void*)&cols, (void*)&inv_scale_override,
        (void*)&y, (void*)&scales, (void*)&global_scale,
        (void*)&row_amax_scratch, (void*)&inv_rms_cache
    };
    
    if (use_four_six) {
        cudaLaunchCooperativeKernel(
            (void*)fused_rmsnorm_act_quant_kernel_opt<BLOCK_SIZE, true>,
            dim3(rows), dim3(BLOCK_SIZE),
            args, 0, nullptr
        );
    } else {
        cudaLaunchCooperativeKernel(
            (void*)fused_rmsnorm_act_quant_kernel_opt<BLOCK_SIZE, false>,
            dim3(rows), dim3(BLOCK_SIZE),
            args, 0, nullptr
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// Convenience wrapper
void launch_fused_rmsnorm_act_quant_opt(
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
    float* row_amax_scratch;
    cudaMallocAsync(&row_amax_scratch, rows * sizeof(float), 0);
    
    launch_fused_rmsnorm_act_quant_opt(
        x, w, epsilon, rows, cols, scale_override, use_four_six,
        y, scales, global_scale, inv_rms_cache, row_amax_scratch
    );
    
    cudaFreeAsync(row_amax_scratch, 0);
}

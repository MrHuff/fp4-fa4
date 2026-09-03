// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Optimized Fused Backward Kernel for AbsMax Norm
// Computes exact gradient for SiLU -> AbsMaxNorm -> Quant pipeline.
//
// Math:
// Forward: z = SiLU(x), s = max(|z|), y = z/s * w
// Backward:
// dx = (dy * w / s) - Correction
// Correction = (sum(dy * y) / s) * I(i == k_max) * sign(z_i)
// Finally backprop through SiLU: grad_input = dx * silu'(x)

#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include "vec.cuh"
#include "utils.cuh"

namespace cg = cooperative_groups;

using bf16x8 = GenericVector<nv_bfloat16, 8>;

constexpr int BLOCK_SIZE = 256;

// -------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float d_silu(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

// Struct to track Max Value AND Index
struct MaxWithIndex {
    float val;
    int idx;
};

struct MaxOp {
    __device__ __forceinline__ MaxWithIndex operator()(const MaxWithIndex& a, const MaxWithIndex& b) const {
        return (a.val >= b.val) ? a : b;
    }
};

// -------------------------------------------------------------------------
// Backward Kernel
// -------------------------------------------------------------------------

__global__ void fused_backward_absmax_kernel(
    const nv_bfloat16* __restrict__ grad_output,  // [rows, cols]
    const nv_bfloat16* __restrict__ input,        // [rows, cols] (original x)
    const nv_bfloat16* __restrict__ weight,       // [cols]
    const float* __restrict__ inv_rms_cache,      // [rows] (stores 1/s)
    int rows, int cols,
    nv_bfloat16* __restrict__ grad_input,         // [rows, cols] (dx)
    float* __restrict__ grad_weight             // [cols] (dw) - Optional, usually handled separately approx
) {
    // Note: grad_weight calculation in a fused kernel is tricky (atomic adds). 
    // Usually PyTorch handles dw separately or we do a separate kernel.
    // For this implementation, we focus on DX (grad_input), as that's where the stability issue lies.
    // We will assume grad_weight is computed separately or we output 'y' to let another kernel do it.
    // Actually, to fully match the "fused" expectation, we usually compute DX. DW is often separate.
    
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    float inv_s = inv_rms_cache[row];

    // Shared memory for reduction
    // We need to compute: 
    // 1. Dot product: sum(dy * y) = sum(dy * (z/s * w)) = sum(dy * w * z) / s
    //    Let dp_sum_accum = sum(dy * w * z)
    // 2. Max index: argmax(|z|)
    
    // Specialize BlockReduce for 256
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceSum;
    typedef cub::BlockReduce<MaxWithIndex, BLOCK_SIZE> BlockReduceMax;
    
    __shared__ union {
        typename BlockReduceSum::TempStorage sum;
        typename BlockReduceMax::TempStorage max;
    } temp_storage;

    // --- PHASE 1: Statistics Collection (DP Sum and Max Index) ---
    
    float thread_dp_sum = 0.0f;
    MaxWithIndex thread_max = {-1.0f, -1};
    
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 dy_vec = bf16x8::load(grad_output + row * cols + i);
        bf16x8 x_vec = bf16x8::load(input + row * cols + i);
        bf16x8 w_vec = bf16x8::load(weight + i);
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float dy = static_cast<float>(dy_vec[k]);
            float x = static_cast<float>(x_vec[k]);
            float w = static_cast<float>(w_vec[k]);
            
            float z = silu(x);
            
            // For Dot Product: dy * y_out = dy * (z * w * inv_s)
            // We accumulate (dy * w * z) and multiply by inv_s at the end
            thread_dp_sum += dy * w * z;
            
            // For Max Index we must match Forward: max(|z|)
            float abs_z = fabsf(z);
            if (abs_z > thread_max.val) {
                thread_max.val = abs_z;
                thread_max.idx = i + k;
            }
        }
    }
    
    // Reduce Sum
    float row_dp_sum_raw = BlockReduceSum(temp_storage.sum).Sum(thread_dp_sum);
    // ... (Reduction Syncs same as before) ...
    __syncthreads();
    
    // Reduce Max
    MaxWithIndex row_max = BlockReduceMax(temp_storage.max).Reduce(thread_max, MaxOp());
    __syncthreads();
    
    // Broadcast...
    __shared__ float s_correction_factor;
    __shared__ int s_max_idx;
    
    if (tid == 0) {
        float dp_sum = row_dp_sum_raw * inv_s; 
        s_correction_factor = dp_sum * inv_s; 
        s_max_idx = row_max.idx;
    }
    __syncthreads();
    
    float correction_factor = s_correction_factor;
    int max_idx = s_max_idx;
    
    // --- PHASE 2: Compute Gradients ---
    
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 dy_vec = bf16x8::load(grad_output + row * cols + i);
        bf16x8 x_vec = bf16x8::load(input + row * cols + i);
        bf16x8 w_vec = bf16x8::load(weight + i);
        bf16x8 dx_vec;
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int global_idx = i + k;
            float dy = static_cast<float>(dy_vec[k]);
            float x = static_cast<float>(x_vec[k]);
            float w = static_cast<float>(w_vec[k]);
            float z = silu(x);
            
            // 1. Backprop through Norm
            // dz = (dy * w * inv_s)
            float dz = dy * w * inv_s;
            
            // 2. Apply Sparse Correction
            // s = max(|z|). ds/dz_i = sign(z) * I(i=kmax)
            // correction term = (sum(dy*y)/s) * ds/dz
            if (global_idx == max_idx) {
                // sign of z
                float sign_z = (z > 0.0f) ? 1.0f : ((z < 0.0f) ? -1.0f : 0.0f);
                
                // term: correction_factor * sign_z (NO W HERE)
                dz -= correction_factor * sign_z;
            }
            
            // 3. Backprop through SiLU
            float dx = dz * d_silu(x);
            
            dx_vec[k] = (nv_bfloat16)dx;
        }
        
        dx_vec.store(grad_input + row * cols + i);
    }
}

// -------------------------------------------------------------------------
// Host Launcher
// -------------------------------------------------------------------------

void launch_fused_backward_absmax(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    const float* inv_rms_cache,
    int rows, int cols,
    nv_bfloat16* grad_input
) {
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);
    
    // Shared memory size? CUB might need some, but we used union in kernel.
    // Usually no dynamic smem needed if using CUB BlockReduce with explicitly typed Simple storage?
    // Actually CUB calls might use smem.
    // Let's check if we need dynamic smem. We declared __shared__ inside kernel.
    
    fused_backward_absmax_kernel<<<grid, block>>>(
        grad_output, input, weight, inv_rms_cache,
        rows, cols, grad_input, nullptr
    );
    CUDA_CHECK(cudaGetLastError());
}

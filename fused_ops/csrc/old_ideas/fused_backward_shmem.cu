// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused Backward Kernel using Shared Memory Cache
// 
// Optimization:
// For cols <= SHMEM_CAPACITY, we cache 'x' and 'd_z' in Shared Memory.
// This allows us to read inputs (dy, x, w) ONLY ONCE from Global Memory.
// 
// Memory Traffic Analysis:
// Standard 2-Pass:
//   Pass 1: Read dy, x, w. Write nothing.
//   Pass 2: Read dy, x, w. Write dx.
//   Total Reads: 2 * (3 * size)
// 
// Shmem 1-Pass:
//   Pass 1: Read dy, x, w. Store x, d_z to Shmem.
//   Pass 2: Read x, d_z from Shmem. Write dx.
//   Total Reads: 1 * (3 * size) -> ~50% reduction in read bandwidth.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;

// Activation derivatives
__device__ __forceinline__ float silu_backward_fast(float dy, float u) {
    float s = 1.0f / (1.0f + __expf(-u));
    return dy * s * (1.0f + u * (1.0f - s));
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    return BlockReduce(temp_storage).Sum(val);
}

// -------------------------------------------------------------------------
// Kernel
// -------------------------------------------------------------------------

template<int BLOCK_SIZE = 256>
__global__ void fused_backward_kernel_shmem(
    const nv_bfloat16* __restrict__ grad_output,
    const nv_bfloat16* __restrict__ input,
    const nv_bfloat16* __restrict__ weight,
    const float* __restrict__ cached_inv_rms,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* __restrict__ grad_input
) {
    // Shared Memory Layout:
    // We need to store X (bf16) and d_z (float or bf16?)
    // d_z = d_y * w. 
    // d_x = inv_rms * (d_z - x * ...)
    // So if we cache x and d_z, we can compute d_x without re-reading w or dy.
    // 
    // Cache x: bf16 (2 bytes) per element
    // Cache d_z: bf16 (2 bytes) per element is enough precision for backward usually? 
    // Or float? Let's use bf16 to save shmem.
    // Total Shmem Requirement: 4 bytes per col.
    // For 8192 cols: 32KB. Correct.
    
    extern __shared__ __align__(16) char smem[];
    nv_bfloat16* s_x = reinterpret_cast<nv_bfloat16*>(smem);
    // Align s_dz to 4 bytes. s_x is 2-byte aligned. cols might be odd? 
    // cols is usually multiple of 8 or 256. 
    // s_x size is cols * 2 bytes. 
    // We can just cast.
    float* s_dz = reinterpret_cast<float*>(s_x + cols); 
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    const nv_bfloat16* row_input = input + row * cols;
    const nv_bfloat16* row_grad = grad_output + row * cols;
    nv_bfloat16* row_dx = grad_input + row * cols;
    
    // 1. Get inv_rms
    float inv_rms;
    if (cached_inv_rms != nullptr) {
        inv_rms = cached_inv_rms[row];
    } else {
        // Fallback recompute (rarely used in this optimized kernel)
        float sum_sq = 0.0f;
        for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
            bf16x8 data = bf16x8::load(row_input + i);
            #pragma unroll
            for (int k = 0; k < 8; ++k) sum_sq += (float)data[k] * (float)data[k];
        }
        float row_sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
        __shared__ float s_inv_rms;
        if (tid == 0) s_inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
        __syncthreads();
        inv_rms = s_inv_rms;
    }
    
    // 2. Read Global, Compute Intermediates, Cache to Shmem, Accumulate Sum
    float local_sum_dy_y = 0.0f;
    
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        bf16x8 x_vec = bf16x8::load(row_input + i);
        bf16x8 w_vec = bf16x8::load(weight + i);
        bf16x8 dy_vec = bf16x8::load(row_grad + i);
        // We need to store floats now. Cannot use bf16x8.
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x_val = (float)x_vec[k];
            float w_val = (float)w_vec[k];
            float g_out = (float)dy_vec[k];
            
            float norm_in = x_val * inv_rms * w_val;
            float d_y = silu_backward_fast(g_out, norm_in);
            
            local_sum_dy_y += d_y * norm_in; // Accumulate reduction term
            
            float d_z = d_y * w_val;
            
            // Store to shared memory as float
            s_dz[i + k] = d_z;
        }
        
        // Cache X to shared memory
        x_vec.store(s_x + i);
        // d_z already stored manually loop above
    }
    
    // 3. Reduction
    float row_sum_dy_y = block_reduce_sum<BLOCK_SIZE>(local_sum_dy_y);
    __shared__ float mean_dy_y;
    if (tid == 0) mean_dy_y = row_sum_dy_y / cols;
    __syncthreads(); // Wait for mean_dy_y AND for Shmem to be populated
    
    float mean_val = mean_dy_y;

    // 4. Compute dX from Shmem (No Global Read!)
    for (int i = tid * 8; i < cols; i += BLOCK_SIZE * 8) {
        // Load from Shmem
        bf16x8 x_vec = bf16x8::load(s_x + i);
        // Load dz from float array
        
        bf16x8 dx_vec;
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x_val = (float)x_vec[k];
            float d_z = s_dz[i + k];
            
            // dx = inv_rms * (d_z - x * inv_rms * mean_dy_y)
            float d_x = inv_rms * (d_z - x_val * inv_rms * mean_val);
            dx_vec[k] = (nv_bfloat16)d_x;
        }
        
        // Store Result to Global
        dx_vec.store(row_dx + i);
    }
}

// -------------------------------------------------------------------------
// Host Launcher
// -------------------------------------------------------------------------

void launch_fused_backward_shmem(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    const float* cached_inv_rms,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* grad_input
) {
    constexpr int BLOCK_SIZE = 256;
    
    // Calculate Shared Memory Size: 
    // cols * sizeof(bf16) [for x] + cols * sizeof(float) [for dz]
    size_t shmem_size = cols * sizeof(nv_bfloat16) + cols * sizeof(float);
    
    // Check if fits limits (e.g. 48KB standard, can go up to ~100KB on A100/H100/Blackwell)
    // If > max shmem, we should fallback or error. 
    // Assuming python side checks or we rely on cuda launch failure/fallback logic.
    // For now, let's assume valid calls up to ~12k cols.
    
    // Check if fits limits.
    // Ensure we have enough Shared Memory.
    // For 8192 cols, we need ~48KB + CUB overhead.
    // Default is usually 48KB. We need to opt-in for more.
    // SM100 supports adequate shmem.
    if (shmem_size > 32 * 1024) { // Be aggressive about reserving
         CUDA_CHECK(cudaFuncSetAttribute(fused_backward_kernel_shmem<BLOCK_SIZE>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
    }
    
    fused_backward_kernel_shmem<BLOCK_SIZE><<<rows, BLOCK_SIZE, shmem_size>>>(
        grad_output, input, weight, cached_inv_rms,
        epsilon, rows, cols, grad_input
    );
    
    CUDA_CHECK(cudaGetLastError());
}

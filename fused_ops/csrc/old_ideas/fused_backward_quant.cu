// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused Backward Quantization: dequant → transpose → requant in a single kernel.
// Replaces the eden Hadamard backward path with direct FP4 quantization.
//
// For the backward pass, we need:
//   1. quant_fp4(grad_output)  — already exists (four_six_fp4_kernel)
//   2. dequant_transpose_quant(weight_fp4) — THIS kernel: FP4 → dequant → T → requant → FP4
//
// The key optimization vs eden: no 128×128 Hadamard MMA.

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include "utils.cuh"
#include "vec.cuh"

// -------------------------------------------------------------------------
// Kernel: Dequantize FP4, transpose, requantize to FP4
// -------------------------------------------------------------------------
// Input:  FP4 tensor x[N, K/2] + scales[N, K/16] + global_scale
// Output: FP4 tensor y[K, N/2] + scales[K, N/16] + global_scale
//
// Strategy: Process in TILE_N × TILE_K tiles.
//   1. Each block processes one tile
//   2. Load FP4 from x, dequant to BF16 in shared memory (row-major: [N_tile, K_tile])
//   3. Read transposed from shared memory (now: [K_tile, N_tile])
//   4. Compute group absmax (group of 16 along N dimension)
//   5. Quantize to FP4 and write to y[K, N/2]
//
// The kernel also computes the global absmax via atomicMax and writes
// scales + global_scale in the NVFP4 format expected by Quartet.
// -------------------------------------------------------------------------

constexpr int TILE_N = 64;   // Tile size along N (input rows)
constexpr int TILE_K = 64;   // Tile size along K (input cols, in original elements, not packed)
constexpr int THREADS = 256;

// Atomic max for positive floats (IEEE754 bit ordering preserves float ordering for positives)
__device__ __forceinline__ void atomicMaxFloat_bq(float* addr, float val) {
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
// Pass 1: Compute global absmax of the transposed dequantized tensor
// -------------------------------------------------------------------------
__global__ void dequant_transpose_amax_kernel(
    float* __restrict__ global_amax,           // output: single float
    const __nv_fp4x2_storage_t* __restrict__ x, // input FP4 packed [N, K/2]
    const __nv_fp8_e4m3* __restrict__ x_scales,  // input scales [N, K/16]
    const float* __restrict__ x_global_scale,    // input global scale
    int N, int K                                  // logical dimensions
) {
    // Each thread processes a chunk of elements
    // We just need the global absmax of the dequantized values
    // This is the same as the absmax of the original tensor (transpose doesn't change absmax)
    // So we can compute it directly from the input without actually transposing.
    
    // Each thread handles groups of 8 BF16 values (= 4 FP4x2 bytes)
    int n_groups = (N * K) / 16;  // number of FP4 groups (16 elements each)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = 0.f;
    float gs = *x_global_scale;
    
    for (int g = tid; g < n_groups; g += gridDim.x * blockDim.x) {
        // Each group is 16 elements → 8 bytes of FP4, 1 FP8 scale
        float group_scale = static_cast<float>(x_scales[g]) * gs;
        
        // Load 8 FP4x2 bytes (16 elements)
        const __nv_fp4x2_storage_t* base = x + g * 8;
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float2 dq = __nv_cvt_fp4x2_to_float2(base[i]);
            float abs1 = fabsf(dq.x * group_scale);
            float abs2 = fabsf(dq.y * group_scale);
            local_max = fmaxf(local_max, fmaxf(abs1, abs2));
        }
    }
    
    // Warp reduce
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 16));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 8));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 4));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 2));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 1));
    
    if (threadIdx.x % 32 == 0) {
        atomicMaxFloat_bq(global_amax, local_max);
    }
}

// -------------------------------------------------------------------------
// Pass 2: Dequant + Transpose + Requant with known global scale
// -------------------------------------------------------------------------
// 
// We process tiles of TILE_N × TILE_K from the input.
// Each block handles one tile:
//   - Load FP4 tile from x[n..n+TILE_N, k..k+TILE_K] → dequant to BF16 in smem
//   - Read transposed from smem: now BF16[TILE_K, TILE_N]
//   - Quantize in groups of 16 along the N dimension (TILE_N / 16 groups per K row)
//   - Write FP4 output to y[k..k+TILE_K, n..n+TILE_N]
// -------------------------------------------------------------------------
__global__ void dequant_transpose_quant_kernel(
    __nv_fp4x2_storage_t* __restrict__ y,        // output FP4 [K, N/2]
    __nv_fp8_e4m3* __restrict__ y_scales,        // output scales [K, N/16]
    float* __restrict__ y_global_scale,          // output global scale
    const __nv_fp4x2_storage_t* __restrict__ x,  // input FP4 [N, K/2]
    const __nv_fp8_e4m3* __restrict__ x_scales,  // input scales [N, K/16]
    const float* __restrict__ x_global_scale,    // input global scale
    const float* __restrict__ dq_amax,           // precomputed global absmax of dequantized tensor
    int N, int K,                                // logical dimensions  
    float scale_override                         // 4/6 scale override (typically 1.0)
) {
    // Shared memory for the dequantized BF16 tile: TILE_N × TILE_K
    // We pad by 1 column to avoid bank conflicts during transposed reads
    __shared__ float smem_tile[TILE_N][TILE_K + 1];  // +1 padding to avoid bank conflicts
    
    // Block ID determines which tile we process
    int tiles_n = (N + TILE_N - 1) / TILE_N;
    int tiles_k = (K + TILE_K - 1) / TILE_K;
    int tile_n = blockIdx.x % tiles_n;
    int tile_k = blockIdx.x / tiles_n;
    
    int n_start = tile_n * TILE_N;
    int k_start = tile_k * TILE_K;
    
    int tid = threadIdx.x;
    float gs_in = *x_global_scale;
    
    // ---- Step 1: Load + Dequant into shared memory [TILE_N, TILE_K] ----
    // Each thread handles multiple elements
    int total_elements = TILE_N * TILE_K;
    for (int i = tid; i < total_elements; i += THREADS) {
        int local_n = i / TILE_K;
        int local_k = i % TILE_K;
        int global_n = n_start + local_n;
        int global_k = k_start + local_k;
        
        float val = 0.f;
        if (global_n < N && global_k < K) {
            // The FP4 data is packed: x[n, k/2], where each byte has 2 FP4 values
            int fp4_idx = global_n * (K / 2) + global_k / 2;
            __nv_fp4x2_storage_t packed = x[fp4_idx];
            
            // Dequant: unpack the correct nibble
            float2 dq = __nv_cvt_fp4x2_to_float2(packed);
            float raw_val = (global_k % 2 == 0) ? dq.x : dq.y;
            
            // Apply scales: group_scale = fp8_scale * global_scale
            int group_idx = global_n * (K / 16) + global_k / 16;
            float group_scale = static_cast<float>(x_scales[group_idx]) * gs_in;
            val = raw_val * group_scale;
        }
        smem_tile[local_n][local_k] = val;
    }
    __syncthreads();
    
    // ---- Step 2: Read transposed + Quantize + Write ----
    // Now smem_tile[n][k] contains the dequantized BF16 value at (n, k).
    // We want to produce y[k, n] in FP4 format.
    // Groups of 16 along the N dimension: each group is y[k, n:n+16]
    
    float global_abs_max = *dq_amax;
    if (global_abs_max == 0.f) global_abs_max = 1.f;
    
    // Scale parameters matching four_six_fp4_kernel
    constexpr float scales_max = 256.f;  // 4/6 mode uses 256
    float val_max = 6.f / scale_override;
    float out_scale = global_abs_max / scales_max / val_max;
    
    // Write global scale (only once across all blocks)
    if (tid == 0 && blockIdx.x == 0) {
        *y_global_scale = out_scale;
    }
    
    // Each thread processes groups of 16 elements from the transposed view
    // Transposed view: y[k][n], k is the row, n is the column
    // Group = 16 consecutive n values at a given k
    int n_groups_per_row = TILE_N / 16;  // groups per K-row within this tile
    int total_groups = TILE_K * n_groups_per_row;
    
    for (int gi = tid; gi < total_groups; gi += THREADS) {
        int local_k = gi / n_groups_per_row;
        int group_in_row = gi % n_groups_per_row;
        int local_n_start = group_in_row * 16;
        
        int global_k = k_start + local_k;
        int global_n_start = n_start + local_n_start;
        
        if (global_k >= K || global_n_start >= N) continue;
        
        // Load 16 values from transposed tile: smem_tile[n][k] for n in [local_n_start..+16]
        float vals[16];
        float abs_max = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int n_idx = local_n_start + i;
            vals[i] = (n_idx < TILE_N && (global_n_start + i) < N) ? smem_tile[n_idx][local_k] : 0.f;
            abs_max = fmaxf(abs_max, fabsf(vals[i]));
        }
        
        // Compute FP8 group scale
        float s_group = abs_max / val_max;
        __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / out_scale);
        float s_round_fp8 = static_cast<float>(s_as_fp8);
        if (s_round_fp8 == 0.f) s_round_fp8 = 1.f;
        float factor = 1.f / (s_round_fp8 * out_scale);
        
        // Quantize 16 values → 8 FP4x2 bytes
        __nv_fp4x2_storage_t packed[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float2 src = make_float2(vals[2*i] * factor, vals[2*i+1] * factor);
            packed[i] = __nv_cvt_float2_to_fp4x2(src, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
        }
        
        // Write FP4 output: y[global_k, global_n/2]
        // Output layout: y is [K, N/2], where N is the output column dimension
        int out_base = global_k * (N / 2) + global_n_start / 2;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (global_n_start + 2*i + 1 < N) {
                y[out_base + i] = packed[i];
            }
        }
        
        // Write scale: y_scales[global_k, group_index_in_N]
        int scale_idx = global_k * (N / 16) + global_n_start / 16;
        y_scales[scale_idx] = s_as_fp8;
    }
}

// -------------------------------------------------------------------------
// Host Launchers
// -------------------------------------------------------------------------

void launch_dequant_transpose_quant(
    __nv_fp4x2_storage_t* y,
    __nv_fp8_e4m3* y_scales,
    float* y_global_scale,
    const __nv_fp4x2_storage_t* x,
    const __nv_fp8_e4m3* x_scales,
    const float* x_global_scale,
    float* scratch_amax,         // preallocated float for atomic max
    int N, int K,
    float scale_override,
    cudaStream_t stream
) {
    // Pass 1: compute global absmax (reuse scratch_amax)
    CUDA_CHECK(cudaMemsetAsync(scratch_amax, 0, sizeof(float), stream));
    {
        int n_groups = (N * K) / 16;
        int threads = 256;
        int blocks = min((n_groups + threads - 1) / threads, 1024);
        dequant_transpose_amax_kernel<<<blocks, threads, 0, stream>>>(
            scratch_amax, x, x_scales, x_global_scale, N, K
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Pass 2: dequant + transpose + requant
    {
        int tiles_n = (N + TILE_N - 1) / TILE_N;
        int tiles_k = (K + TILE_K - 1) / TILE_K;
        int blocks = tiles_n * tiles_k;
        int smem = 0;  // Using static shared memory
        
        dequant_transpose_quant_kernel<<<blocks, THREADS, smem, stream>>>(
            y, y_scales, y_global_scale,
            x, x_scales, x_global_scale, scratch_amax,
            N, K, scale_override
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

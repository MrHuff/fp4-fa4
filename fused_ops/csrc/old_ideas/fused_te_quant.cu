// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + Activation + NVFP4 Quantization — TE-Compatible Output
// 2-PASS DESIGN: No cooperative launch needed (works with torch.utils.cpp_extension)
//
// Pass 1: Compute RMSNorm stats + per-block amax → write global amax
// Pass 2: Read global scale → normalize + activate + quantize
//
// Output format (TE/cuBLASLt compatible):
//   - Packed FP4 data: [M, K/2] as uint8
//   - Block scales:    [M, K/16] as fp8e4m3
//   - Global scale:    float scalar = global_amax / (448 * 6)

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>
#include <cub/cub.cuh>

#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;

// Portable bf16→float (works with -D__CUDA_NO_BFLOAT16_CONVERSIONS__)
__device__ __forceinline__ float bf16_to_f32(nv_bfloat16 v) {
    return __bfloat162float(v);
}

constexpr int BLOCK_GROUP_SIZE = 16;
constexpr int WARP_SIZE = 32;

// =========================================================================
// Activation Functions
// =========================================================================

__device__ __forceinline__ float act_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float act_gelu(float x) {
    constexpr float k = 0.7978845608f;
    constexpr float c = 0.044715f;
    float inner = k * (x + c * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

template<int ACT_MODE>
__device__ __forceinline__ float apply_activation(float x) {
    if constexpr (ACT_MODE == 0) return act_silu(x);
    else if constexpr (ACT_MODE == 1) return act_gelu(x);
    else return x;
}

// =========================================================================
// Block Reductions
// =========================================================================

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

// =========================================================================
// TE Scale Computation
// =========================================================================

__device__ __forceinline__ float compute_te_global_scale(float global_amax) {
    constexpr float fp8_max = 448.0f;
    constexpr float fp4_max = 6.0f;
    if (global_amax == 0.0f) return 1.0f;
    return global_amax / (fp8_max * fp4_max);
}

__device__ __forceinline__ __nv_fp8_e4m3 compute_te_block_scale(
    float block_amax, float global_scale
) {
    constexpr float fp4_max = 6.0f;
    float s_dec_b = block_amax / (fp4_max * global_scale);
    s_dec_b = fminf(s_dec_b, 448.0f);
    return static_cast<__nv_fp8_e4m3>(s_dec_b);
}

// =========================================================================
// Pass 1: Compute RMSNorm stats + per-block amax + row amax
//   → Also writes inv_rms_cache and atomicMax into global_amax_ptr
// =========================================================================

template<int BLOCK_SIZE = 256, int NORM_MODE = 0, int ACT_MODE = 0>
__global__ void fused_te_quant_pass1(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float* __restrict__ block_amax_scratch,   // [rows * cols/16]
    float* __restrict__ inv_rms_cache,        // [rows]
    unsigned int* __restrict__ global_amax_bits // Single uint32 for atomicMax
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;

    float stat = 0.0f;
    float my_row_amax = 0.0f;

    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        int elem_start = block_id * BLOCK_GROUP_SIZE;
        bf16x8 data0 = bf16x8::load(x_ptr + row * cols + elem_start);
        bf16x8 data1 = bf16x8::load(x_ptr + row * cols + elem_start + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + elem_start);
        bf16x8 w1 = bf16x8::load(w_ptr + elem_start + 8);

        float block_max = 0.0f;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = bf16_to_f32(data0[k]);
            float wv = bf16_to_f32(w0[k]);
            if constexpr (NORM_MODE == 0) stat += val * val;
            else stat = fmaxf(stat, fabsf(val));
            float act_val = apply_activation<ACT_MODE>(val);
            block_max = fmaxf(block_max, fabsf(act_val * wv));
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = bf16_to_f32(data1[k]);
            float wv = bf16_to_f32(w1[k]);
            if constexpr (NORM_MODE == 0) stat += val * val;
            else stat = fmaxf(stat, fabsf(val));
            float act_val = apply_activation<ACT_MODE>(val);
            block_max = fmaxf(block_max, fabsf(act_val * wv));
        }

        // Save per-block amax (PRE-rms-scaling)
        block_amax_scratch[row * num_blocks_per_row + block_id] = block_max;
        my_row_amax = fmaxf(my_row_amax, block_max);
    }

    // Compute inv_rms
    float inv_rms;
    if constexpr (NORM_MODE == 0) {
        float row_sum_sq = block_reduce_sum<BLOCK_SIZE>(stat);
        __shared__ float s_inv_rms;
        if (tid == 0) {
            s_inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
            inv_rms_cache[row] = s_inv_rms;
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    } else {
        float row_max = block_reduce_max<BLOCK_SIZE>(stat);
        __shared__ float s_inv_rms;
        if (tid == 0) {
            s_inv_rms = (row_max > 0.0f) ? (1.0f / row_max) : 1.0f;
            inv_rms_cache[row] = s_inv_rms;
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    }

    // Scale block amaxes by inv_rms and find row max
    my_row_amax = 0.0f;
    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        float scaled = block_amax_scratch[row * num_blocks_per_row + block_id] * inv_rms;
        block_amax_scratch[row * num_blocks_per_row + block_id] = scaled;
        my_row_amax = fmaxf(my_row_amax, scaled);
    }

    float row_amax = block_reduce_max<BLOCK_SIZE>(my_row_amax);

    // atomicMax on global amax (using float-as-uint trick for positive floats)
    if (tid == 0 && row_amax > 0.0f) {
        atomicMax(global_amax_bits, __float_as_uint(row_amax));
    }
}

// =========================================================================
// Pass 2: Read global scale → normalize + activate + quantize
// =========================================================================

template<int BLOCK_SIZE = 256, int ACT_MODE = 0>
__global__ void fused_te_quant_pass2(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    int rows, int cols,
    const float* __restrict__ block_amax_scratch,
    const float* __restrict__ inv_rms_cache,
    const float* __restrict__ global_scale_ptr,
    unsigned char*    __restrict__ y_ptr,
    __nv_fp8_e4m3*    __restrict__ scale_ptr
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;

    float inv_rms = inv_rms_cache[row];
    float global_scale = *global_scale_ptr;

    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        int elem_start = block_id * BLOCK_GROUP_SIZE;

        bf16x8 data0 = bf16x8::load(x_ptr + row * cols + elem_start);
        bf16x8 data1 = bf16x8::load(x_ptr + row * cols + elem_start + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + elem_start);
        bf16x8 w1 = bf16x8::load(w_ptr + elem_start + 8);

        // Compute normalized + activated values
        float vals[16];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = bf16_to_f32(data0[k]);
            float wv = bf16_to_f32(w0[k]);
            vals[k] = apply_activation<ACT_MODE>(val) * wv * inv_rms;
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = bf16_to_f32(data1[k]);
            float wv = bf16_to_f32(w1[k]);
            vals[8 + k] = apply_activation<ACT_MODE>(val) * wv * inv_rms;
        }

        // Block scale
        float block_amax = block_amax_scratch[row * num_blocks_per_row + block_id];
        __nv_fp8_e4m3 s_fp8 = compute_te_block_scale(block_amax, global_scale);
        float s_dec_b_float = float(s_fp8);
        if (s_dec_b_float == 0.0f) s_dec_b_float = 1.0f;

        float block_scale_inv = 1.0f / (s_dec_b_float * global_scale);

        // Quantize 16 values → 8 bytes
        unsigned char fp4_bytes[8];
        #pragma unroll
        for (int k = 0; k < 16; k += 2) {
            float2 src;
            src.x = vals[k + 0] * block_scale_inv;
            src.y = vals[k + 1] * block_scale_inv;
            fp4_bytes[k / 2] = __nv_cvt_float2_to_fp4x2(
                src, __nv_fp4_interpretation_t::__NV_E2M1,
                cudaRoundMode::cudaRoundNearest
            );
        }

        // Store packed FP4
        int byte_offset = (row * cols + elem_start) / 2;
        *reinterpret_cast<uint32_t*>(y_ptr + byte_offset) = *reinterpret_cast<uint32_t*>(fp4_bytes);
        *reinterpret_cast<uint32_t*>(y_ptr + byte_offset + 4) = *reinterpret_cast<uint32_t*>(fp4_bytes + 4);

        // Store block scale
        scale_ptr[row * num_blocks_per_row + block_id] = s_fp8;
    }
}

// =========================================================================
// Tiny kernel: convert atomicMax result → global_scale
// =========================================================================

__global__ void compute_global_scale_kernel(
    const unsigned int* __restrict__ amax_bits,
    float* __restrict__ global_scale_ptr
) {
    float amax = __uint_as_float(*amax_bits);
    if (amax == 0.0f) amax = 1.0f;
    *global_scale_ptr = amax / (448.0f * 6.0f);
}

// =========================================================================
// Host Launcher (2-pass, no cooperative launch)
// =========================================================================

void launch_fused_te_quant(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    int norm_mode,
    int act_mode,
    unsigned char* y,
    __nv_fp8_e4m3* scales,
    float* global_scale,
    float* inv_rms_cache
) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;

    // Allocate scratch
    float* block_amax_scratch;
    unsigned int* global_amax_bits;
    cudaMallocAsync(&block_amax_scratch, rows * num_blocks_per_row * sizeof(float), 0);
    cudaMallocAsync(reinterpret_cast<void**>(&global_amax_bits), sizeof(unsigned int), 0);
    cudaMemsetAsync(global_amax_bits, 0, sizeof(unsigned int), 0);

    // === PASS 1: Stats + amax ===
    #define LAUNCH_PASS1(NM, AM) \
        fused_te_quant_pass1<BLOCK_SIZE, NM, AM><<<rows, BLOCK_SIZE>>>( \
            x, w, epsilon, rows, cols, block_amax_scratch, inv_rms_cache, global_amax_bits);

    switch (norm_mode * 3 + act_mode) {
        case 0: LAUNCH_PASS1(0, 0); break;
        case 1: LAUNCH_PASS1(0, 1); break;
        case 2: LAUNCH_PASS1(0, 2); break;
        case 3: LAUNCH_PASS1(1, 0); break;
        case 4: LAUNCH_PASS1(1, 1); break;
        case 5: LAUNCH_PASS1(1, 2); break;
        case 6: LAUNCH_PASS1(2, 0); break;
        case 7: LAUNCH_PASS1(2, 1); break;
        case 8: LAUNCH_PASS1(2, 2); break;
    }
    #undef LAUNCH_PASS1

    // Compute global scale from atomic max
    compute_global_scale_kernel<<<1, 1>>>(global_amax_bits, global_scale);

    // === PASS 2: Quantize ===
    #define LAUNCH_PASS2(AM) \
        fused_te_quant_pass2<BLOCK_SIZE, AM><<<rows, BLOCK_SIZE>>>( \
            x, w, rows, cols, block_amax_scratch, inv_rms_cache, global_scale, y, scales);

    switch (act_mode) {
        case 0: LAUNCH_PASS2(0); break;
        case 1: LAUNCH_PASS2(1); break;
        case 2: LAUNCH_PASS2(2); break;
    }
    #undef LAUNCH_PASS2

    CUDA_CHECK(cudaGetLastError());
    cudaFreeAsync(block_amax_scratch, 0);
    cudaFreeAsync(global_amax_bits, 0);
}

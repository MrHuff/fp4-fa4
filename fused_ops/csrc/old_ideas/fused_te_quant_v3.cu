// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// V3: Fused RMSNorm + Activation + NVFP4 Quantization — PTX-Optimized
//
// Improvements over V1:
//   1. PTX fused mul+cvt: mul.f32x2 + cvt.rn.satfinite.e2m1x2 (4 vals/instruction)
//   2. Encode-centric vs Decode-centric scaling (flag)
//   3. Vectorized 128-bit loads (4x float or 8x bf16)
//
// Output format (TE/cuBLASLt compatible):
//   - Packed FP4 data: [M, K/2] as uint8
//   - Block scales:    [M, K/16] as fp8e4m3
//   - Global scale:    float scalar
//
// Scale modes:
//   DECODE_CENTRIC (default, same as V1):
//     global_scale = global_amax / (448 * 6)   [= S_dec]
//     block_scale = block_amax / (6 * global_scale)   [= S_b, stored as fp8]
//     quant_factor = 1 / (S_b_fp8 * global_scale)
//     Dequant: value = fp4_val * S_b_fp8 * S_dec
//
//   ENCODE_CENTRIC:
//     S_enc = 448 * 6 / global_amax   [= 1/S_dec]
//     block_mult = 6 / (block_amax * S_enc)   [= M, the multiplier, stored as fp8]
//     quant_factor = M_fp8 * S_enc
//     block_scale_stored = 1/M_fp8   [stored for decode compatibility]
//     Dequant: value = fp4_val / M_fp8 / S_enc  (= fp4_val * S_b_fp8 * S_dec)
//
// Norm modes: 0=RMS, 1=AbsMax, 2=MXNorm-BlockRMS
// Activation: 0=SiLU, 1=GeLU, 2=Identity

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>
#include <cub/cub.cuh>

#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;

__device__ __forceinline__ float bf16_to_f32(nv_bfloat16 v) {
    return __bfloat162float(v);
}

constexpr int BLOCK_GROUP_SIZE = 16;

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
// PTX Fused mul + cvt (from TE's nvfp4_transpose.cuh)
//
// Takes 4 float32 values (as 2x float2), multiplies by scale,
// then converts to 4 FP4 E2M1 nibbles (1 uint16 = 4 nibbles).
//
// This is ~2x faster than sequential __nv_cvt_float2_to_fp4x2 calls
// because the mul+cvt are fused in the instruction pipeline.
// =========================================================================

struct fp4x4_packed {
    uint16_t bits;  // 4 FP4 nibbles packed in 16 bits
};

__device__ __forceinline__ fp4x4_packed mul_cvt_fp32_to_fp4_4x(
    const float2 in01, const float2 in23, const float2 scale
) {
    uint32_t out_4x = 0;
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        ".reg.b8 f0; \n\t"
        ".reg.b8 f1; \n\t"
        "mov.b64 {v0, v1} , %1; \n\t"
        "mov.b64 {v2, v3} , %2; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %3; \n\t"
        "mul.f32x2 v23, v23, %3; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
        "mov.b32 %0, {f0, f1, f0, f1};\n\t"
        "}"
        : "=r"(out_4x)
        : "l"(reinterpret_cast<const uint64_t &>(in01)),
          "l"(reinterpret_cast<const uint64_t &>(in23)),
          "l"(reinterpret_cast<const uint64_t &>(scale)));
    fp4x4_packed result;
    result.bits = static_cast<uint16_t>(out_4x & 0xFFFF);
    return result;
}

// =========================================================================
// Scale Computation — Decode-Centric (default, compatible with V1)
// =========================================================================

__device__ __forceinline__ float compute_global_scale_decode(float global_amax) {
    constexpr float fp8_max = 448.0f;
    constexpr float fp4_max = 6.0f;
    if (global_amax == 0.0f) return 1.0f;
    return global_amax / (fp8_max * fp4_max);
}

__device__ __forceinline__ __nv_fp8_e4m3 compute_block_scale_decode(
    float block_amax, float global_scale
) {
    constexpr float fp4_max = 6.0f;
    float s_dec_b = block_amax / (fp4_max * global_scale);
    s_dec_b = fminf(s_dec_b, 448.0f);
    return static_cast<__nv_fp8_e4m3>(s_dec_b);
}

// =========================================================================
// Scale Computation — Encode-Centric
// =========================================================================

__device__ __forceinline__ float compute_global_scale_encode(float global_amax) {
    constexpr float fp8_max = 448.0f;
    constexpr float fp4_max = 6.0f;
    if (global_amax == 0.0f) return 1.0f;
    float s_enc = fp8_max * fp4_max / global_amax;
    s_enc = fminf(s_enc, 3.4e38f);  // clamp to float max
    if (s_enc == 0.0f) return 1.0f;
    return s_enc;
}

__device__ __forceinline__ __nv_fp8_e4m3 compute_block_mult_encode(
    float block_amax, float s_enc
) {
    constexpr float fp4_max = 6.0f;
    if (block_amax <= 1.0e-9f) {
        return static_cast<__nv_fp8_e4m3>(448.0f);  // max fp8
    }
    float mult = fp4_max / (block_amax * s_enc);
    mult = fminf(mult, 448.0f);
    return static_cast<__nv_fp8_e4m3>(mult);
}

// =========================================================================
// Pass 1: Stats + block amax
// =========================================================================

template<int BLOCK_SIZE = 256, int NORM_MODE = 0, int ACT_MODE = 0>
__global__ void fused_te_quant_v3_pass1(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float* __restrict__ block_amax_scratch,
    float* __restrict__ inv_rms_cache,
    unsigned int* __restrict__ global_amax_bits
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;

    float stat = 0.0f;
    float my_row_amax = 0.0f;

    // Process 16 elements (1 quant block) per iteration per thread
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
            else if constexpr (NORM_MODE == 1) stat = fmaxf(stat, fabsf(val));
            // NORM_MODE == 2: MXNorm uses block maxes, no per-element stat
            float act_val = apply_activation<ACT_MODE>(val);
            block_max = fmaxf(block_max, fabsf(act_val * wv));
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = bf16_to_f32(data1[k]);
            float wv = bf16_to_f32(w1[k]);
            if constexpr (NORM_MODE == 0) stat += val * val;
            else if constexpr (NORM_MODE == 1) stat = fmaxf(stat, fabsf(val));
            float act_val = apply_activation<ACT_MODE>(val);
            block_max = fmaxf(block_max, fabsf(act_val * wv));
        }

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
    } else if constexpr (NORM_MODE == 1) {
        float row_max = block_reduce_max<BLOCK_SIZE>(stat);
        __shared__ float s_inv_rms;
        if (tid == 0) {
            s_inv_rms = (row_max > 0.0f) ? (1.0f / row_max) : 1.0f;
            inv_rms_cache[row] = s_inv_rms;
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    } else {
        // MXNorm-BlockRMS: use block maxes as norm stat
        float block_sum_sq = 0.0f;
        for (int b = tid; b < num_blocks_per_row; b += BLOCK_SIZE) {
            float bmax = block_amax_scratch[row * num_blocks_per_row + b];
            block_sum_sq += bmax * bmax;
        }
        float row_block_sum_sq = block_reduce_sum<BLOCK_SIZE>(block_sum_sq);
        __shared__ float s_inv_rms;
        if (tid == 0) {
            s_inv_rms = rsqrtf(row_block_sum_sq / num_blocks_per_row + epsilon);
            inv_rms_cache[row] = s_inv_rms;
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    }

    // Scale block amaxes by inv_rms
    my_row_amax = 0.0f;
    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        float scaled = block_amax_scratch[row * num_blocks_per_row + block_id] * inv_rms;
        block_amax_scratch[row * num_blocks_per_row + block_id] = scaled;
        my_row_amax = fmaxf(my_row_amax, scaled);
    }

    float row_amax = block_reduce_max<BLOCK_SIZE>(my_row_amax);

    if (tid == 0 && row_amax > 0.0f) {
        atomicMax(global_amax_bits, __float_as_uint(row_amax));
    }
}

// =========================================================================
// Tiny kernel: convert atomicMax result → global scale
// =========================================================================

__global__ void compute_global_scale_v3(
    const unsigned int* __restrict__ amax_bits,
    float* __restrict__ global_scale_ptr,
    int encode_centric
) {
    float amax = __uint_as_float(*amax_bits);
    if (amax == 0.0f) amax = 1.0f;
    if (encode_centric) {
        *global_scale_ptr = compute_global_scale_encode(amax);
    } else {
        *global_scale_ptr = compute_global_scale_decode(amax);
    }
}

// =========================================================================
// Pass 2: Quantize with PTX fused mul+cvt
// =========================================================================

template<int BLOCK_SIZE = 256, int ACT_MODE = 0, int SCALE_MODE = 0>
__global__ void fused_te_quant_v3_pass2(
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

        // Compute block scale
        float block_amax = block_amax_scratch[row * num_blocks_per_row + block_id];
        float block_scale_inv;
        __nv_fp8_e4m3 stored_scale;

        if constexpr (SCALE_MODE == 0) {
            // DECODE-CENTRIC: store S_b (divisor)
            stored_scale = compute_block_scale_decode(block_amax, global_scale);
            float s_float = float(stored_scale);
            if (s_float == 0.0f) s_float = 1.0f;
            block_scale_inv = 1.0f / (s_float * global_scale);
        } else {
            // ENCODE-CENTRIC: compute multiplier M, store 1/M
            __nv_fp8_e4m3 mult_fp8 = compute_block_mult_encode(block_amax, global_scale);
            float mult_float = float(mult_fp8);
            if (mult_float == 0.0f) mult_float = 1.0f;
            block_scale_inv = mult_float * global_scale;
            // Store the reciprocal (divisor) for decode compatibility
            stored_scale = static_cast<__nv_fp8_e4m3>(1.0f / mult_float);
        }

        // PTX fused mul+cvt: process 4 values at a time
        float2 scale_2x = {block_scale_inv, block_scale_inv};

        // Block 0-3
        float2 in01 = {vals[0], vals[1]};
        float2 in23 = {vals[2], vals[3]};
        fp4x4_packed q0 = mul_cvt_fp32_to_fp4_4x(in01, in23, scale_2x);

        // Block 4-7
        in01 = {vals[4], vals[5]};
        in23 = {vals[6], vals[7]};
        fp4x4_packed q1 = mul_cvt_fp32_to_fp4_4x(in01, in23, scale_2x);

        // Block 8-11
        in01 = {vals[8], vals[9]};
        in23 = {vals[10], vals[11]};
        fp4x4_packed q2 = mul_cvt_fp32_to_fp4_4x(in01, in23, scale_2x);

        // Block 12-15
        in01 = {vals[12], vals[13]};
        in23 = {vals[14], vals[15]};
        fp4x4_packed q3 = mul_cvt_fp32_to_fp4_4x(in01, in23, scale_2x);

        // Pack 4x fp4x4 (4x 16-bit = 64 bits = 8 bytes)
        int byte_offset = (row * cols + elem_start) / 2;
        uint16_t* out16 = reinterpret_cast<uint16_t*>(y_ptr + byte_offset);
        out16[0] = q0.bits;
        out16[1] = q1.bits;
        out16[2] = q2.bits;
        out16[3] = q3.bits;

        // Store block scale
        scale_ptr[row * num_blocks_per_row + block_id] = stored_scale;
    }
}

// =========================================================================
// Host Launcher
// =========================================================================

void launch_fused_te_quant_v3(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    int norm_mode,      // 0=RMS, 1=AbsMax, 2=MXNorm-BlockRMS
    int act_mode,       // 0=SiLU, 1=GeLU, 2=Identity
    int scale_mode,     // 0=decode-centric, 1=encode-centric
    unsigned char* y,
    __nv_fp8_e4m3* scales,
    float* global_scale,
    float* inv_rms_cache,
    float* block_amax_scratch
) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;

    // Temp for atomicMax
    unsigned int* global_amax_bits;
    cudaMallocAsync(&global_amax_bits, sizeof(unsigned int), 0);
    cudaMemsetAsync(global_amax_bits, 0, sizeof(unsigned int), 0);

    // Pass 1: stats + block amax
    #define DISPATCH_PASS1(NM, AM) \
        fused_te_quant_v3_pass1<BLOCK_SIZE, NM, AM><<<rows, BLOCK_SIZE>>>( \
            x, w, epsilon, rows, cols, block_amax_scratch, inv_rms_cache, global_amax_bits)

    switch (norm_mode * 3 + act_mode) {
        case 0: DISPATCH_PASS1(0, 0); break;
        case 1: DISPATCH_PASS1(0, 1); break;
        case 2: DISPATCH_PASS1(0, 2); break;
        case 3: DISPATCH_PASS1(1, 0); break;
        case 4: DISPATCH_PASS1(1, 1); break;
        case 5: DISPATCH_PASS1(1, 2); break;
        case 6: DISPATCH_PASS1(2, 0); break;
        case 7: DISPATCH_PASS1(2, 1); break;
        case 8: DISPATCH_PASS1(2, 2); break;
    }
    #undef DISPATCH_PASS1

    // Tiny kernel: global scale
    compute_global_scale_v3<<<1, 1>>>(global_amax_bits, global_scale, scale_mode);

    // Pass 2: quantize with PTX fused mul+cvt
    #define DISPATCH_PASS2(AM, SM) \
        fused_te_quant_v3_pass2<BLOCK_SIZE, AM, SM><<<rows, BLOCK_SIZE>>>( \
            x, w, rows, cols, block_amax_scratch, inv_rms_cache, global_scale, y, scales)

    switch (act_mode * 2 + scale_mode) {
        case 0: DISPATCH_PASS2(0, 0); break;  // SiLU + decode
        case 1: DISPATCH_PASS2(0, 1); break;  // SiLU + encode
        case 2: DISPATCH_PASS2(1, 0); break;  // GeLU + decode
        case 3: DISPATCH_PASS2(1, 1); break;  // GeLU + encode
        case 4: DISPATCH_PASS2(2, 0); break;  // Identity + decode
        case 5: DISPATCH_PASS2(2, 1); break;  // Identity + encode
    }
    #undef DISPATCH_PASS2

    CUDA_CHECK(cudaGetLastError());
    cudaFreeAsync(global_amax_bits, 0);
}

// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// V7: Optimized Fused RMSNorm + Activation + NVFP4 Quantization
//
// Consolidates V1 performance baseline with V3 features:
//   - 3 Norm modes:  0=RMS, 1=AbsMax, 2=MXNorm-BlockRMS
//   - 3 Act modes:   0=SiLU, 1=GeLU, 2=Identity
//   - 2 Scale modes: 0=decode-centric, 1=encode-centric
//   - PTX fused mul+cvt.rn.satfinite.e2m1x2
//
// Optimizations over V1/V3:
//   1. Caller pre-allocates scratch (no cudaMallocAsync per call)
//   2. Eliminated pass1 amax re-scaling loop (defer to pass2)
//   3. Warp shuffle reductions (no cub::BlockReduce shmem overhead)
//   4. Row pointer precomputation
//   5. 64-bit packed FP4 output stores

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;

__device__ __forceinline__ float bf16_to_f32(nv_bfloat16 v) {
    return __bfloat162float(v);
}

constexpr int BG = 16;   // NVFP4 block group size
constexpr int BS = 256;  // threads per block

// =========================================================================
// Warp-level reductions (no shared memory needed for 256 threads, 8 warps)
// =========================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// Full block reduction using warp shuffles + small shared memory
__device__ __forceinline__ float block_reduce_sum_fast(float val) {
    __shared__ float warp_vals[8];  // 256/32 = 8 warps
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    val = warp_reduce_sum(val);
    if (lane == 0) warp_vals[warp] = val;
    __syncthreads();
    
    // First warp reduces across warps
    val = (threadIdx.x < 8) ? warp_vals[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ float block_reduce_max_fast(float val) {
    __shared__ float warp_vals[8];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    val = warp_reduce_max(val);
    if (lane == 0) warp_vals[warp] = val;
    __syncthreads();
    
    val = (threadIdx.x < 8) ? warp_vals[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_max(val);
    return val;
}

// =========================================================================
// Activation functions
// =========================================================================

__device__ __forceinline__ float act_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float act_gelu(float x) {
    constexpr float k = 0.7978845608f, c = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
}

template<int ACT>
__device__ __forceinline__ float apply_act(float x) {
    if constexpr (ACT == 0) return act_silu(x);
    else if constexpr (ACT == 1) return act_gelu(x);
    else return x;
}

// =========================================================================
// PTX fused mul+cvt: 4 float32 → 4 fp4 nibbles (16 bits)
// =========================================================================

struct fp4x4 { uint16_t bits; };

__device__ __forceinline__ fp4x4 ptx_mul_cvt(float2 a, float2 b, float2 s) {
    uint32_t o = 0;
    asm volatile(
        "{\n"
        ".reg.b64 v01; .reg.b64 v23;\n\t"
        ".reg.b32 v0; .reg.b32 v1; .reg.b32 v2; .reg.b32 v3;\n\t"
        ".reg.b8 f0; .reg.b8 f1;\n\t"
        "mov.b64 {v0,v1}, %1;\n\t"
        "mov.b64 {v2,v3}, %2;\n\t"
        "mov.b64 v01, {v0,v1};\n\t"
        "mov.b64 v23, {v2,v3};\n\t"
        "mul.f32x2 v01, v01, %3;\n\t"
        "mul.f32x2 v23, v23, %3;\n\t"
        "mov.b64 {v1,v0}, v01;\n\t"
        "mov.b64 {v3,v2}, v23;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
        "mov.b32 %0, {f0, f1, f0, f1};\n\t"
        "}" : "=r"(o)
        : "l"(reinterpret_cast<const uint64_t&>(a)),
          "l"(reinterpret_cast<const uint64_t&>(b)),
          "l"(reinterpret_cast<const uint64_t&>(s)));
    return {(uint16_t)(o & 0xFFFF)};
}

// =========================================================================
// Scale computation helpers
// =========================================================================

// Decode-centric: global_scale = amax / (448 * 6)
__device__ __forceinline__ float gs_decode(float a) {
    return (a == 0.0f) ? 1.0f : a / (448.0f * 6.0f);
}
// Encode-centric: global_scale = (448 * 6) / amax
__device__ __forceinline__ float gs_encode(float a) {
    if (a == 0.0f) return 1.0f;
    float s = 448.0f * 6.0f / a;
    return (s == 0.0f) ? 1.0f : fminf(s, 3.4e38f);
}
// Block scale decode: s_b = block_amax / (6 * gs)
__device__ __forceinline__ __nv_fp8_e4m3 bs_decode(float ba, float gs) {
    return static_cast<__nv_fp8_e4m3>(fminf(ba / (6.0f * gs), 448.0f));
}
// Block multiplier encode: mult = 6 / (block_amax * s_enc)
__device__ __forceinline__ __nv_fp8_e4m3 bm_encode(float ba, float se) {
    if (ba <= 1e-9f) return static_cast<__nv_fp8_e4m3>(448.0f);
    return static_cast<__nv_fp8_e4m3>(fminf(6.0f / (ba * se), 448.0f));
}

// =========================================================================
// Pass 1: Stats + block amax (PRE-norm) + atomicMax global amax
//
// KEY OPTIMIZATION: block_amax is stored PRE-norm (i.e. amax of act(x)*w
// before inv_rms scaling). The inv_rms scaling is deferred to pass2.
// This eliminates the extra loop in V3 pass1 that re-reads and scales
// block_amax_scratch.
//
// For MXNorm-BlockRMS: the norm stat is sqrt(mean(block_max^2)), where
// block_max is the PRE-norm max. We scale both the amax AND the norm
// factor appropriately.
// =========================================================================

template<int NORM = 0, int ACT = 0>
__global__ void __launch_bounds__(BS)
v7_pass1(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ w,
    float epsilon,
    int rows, int cols,
    float* __restrict__ block_amax,    // [rows, cols/16]
    float* __restrict__ inv_rms_out,   // [rows]
    unsigned int* __restrict__ ga_bits // single uint32
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nbpr = cols / BG;

    const nv_bfloat16* row_x = x + (int64_t)row * cols;

    float stat = 0.0f;
    float my_amax = 0.0f;

    for (int qb = tid; qb < nbpr; qb += BS) {
        int off = qb * BG;
        bf16x8 d0 = bf16x8::load(row_x + off);
        bf16x8 d1 = bf16x8::load(row_x + off + 8);
        bf16x8 w0 = bf16x8::load(w + off);
        bf16x8 w1 = bf16x8::load(w + off + 8);

        float bmax = 0.0f;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = bf16_to_f32(d0[k]);
            float wv = bf16_to_f32(w0[k]);
            if constexpr (NORM == 0) stat += v * v;          // RMS
            else if constexpr (NORM == 1) stat = fmaxf(stat, fabsf(v)); // AbsMax
            // NORM == 2: MXNorm uses block maxes
            float av = apply_act<ACT>(v) * wv;
            bmax = fmaxf(bmax, fabsf(av));
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = bf16_to_f32(d1[k]);
            float wv = bf16_to_f32(w1[k]);
            if constexpr (NORM == 0) stat += v * v;
            else if constexpr (NORM == 1) stat = fmaxf(stat, fabsf(v));
            float av = apply_act<ACT>(v) * wv;
            bmax = fmaxf(bmax, fabsf(av));
        }

        // Store PRE-norm block amax
        block_amax[row * nbpr + qb] = bmax;
        my_amax = fmaxf(my_amax, bmax);
    }

    // Compute inv_rms based on norm mode
    float inv_rms;
    __shared__ float s_inv_rms;

    if constexpr (NORM == 0) {
        // RMS norm: inv_rms = 1/sqrt(mean(x^2) + eps)
        float total = block_reduce_sum_fast(stat);
        if (tid == 0) {
            s_inv_rms = rsqrtf(total / cols + epsilon);
            inv_rms_out[row] = s_inv_rms;
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    } else if constexpr (NORM == 1) {
        // AbsMax norm: inv_rms = 1/max(|x|)
        float total = block_reduce_max_fast(stat);
        if (tid == 0) {
            s_inv_rms = (total > 0.0f) ? (1.0f / total) : 1.0f;
            inv_rms_out[row] = s_inv_rms;
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    } else {
        // MXNorm-BlockRMS: inv_rms = 1/sqrt(mean(block_max^2) + eps)
        // Uses the just-written block amaxes
        float bsq = 0.0f;
        for (int qb = tid; qb < nbpr; qb += BS) {
            float bm = block_amax[row * nbpr + qb];
            bsq += bm * bm;
        }
        float total = block_reduce_sum_fast(bsq);
        if (tid == 0) {
            s_inv_rms = rsqrtf(total / nbpr + epsilon);
            inv_rms_out[row] = s_inv_rms;
        }
        __syncthreads();
        inv_rms = s_inv_rms;
    }

    // Scale row amax by inv_rms for global amax computation
    // my_amax was PRE-norm; scaled amax = my_amax * inv_rms
    float scaled_amax = my_amax * inv_rms;
    float row_amax = block_reduce_max_fast(scaled_amax);

    if (tid == 0 && row_amax > 0.0f)
        atomicMax(ga_bits, __float_as_uint(row_amax));
}

// =========================================================================
// Tiny kernel: atomicMax bits → global_scale
// =========================================================================

__global__ void v7_global_scale(
    const unsigned int* __restrict__ bits,
    float* __restrict__ gs_ptr,
    int encode_centric
) {
    float a = __uint_as_float(*bits);
    if (a == 0.0f) a = 1.0f;
    *gs_ptr = encode_centric ? gs_encode(a) : gs_decode(a);
}

// =========================================================================
// Pass 2: Quantize with PTX fused mul+cvt
//
// Reads block_amax (PRE-norm), applies inv_rms to get true block_amax,
// then computes block scale and quantizes.
// =========================================================================

template<int ACT = 0, int SCALE = 0>
__global__ void __launch_bounds__(BS)
v7_pass2(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ w,
    int rows, int cols,
    const float* __restrict__ block_amax,     // PRE-norm
    const float* __restrict__ inv_rms_cache,
    const float* __restrict__ global_scale_ptr,
    unsigned char* __restrict__ y,
    __nv_fp8_e4m3* __restrict__ scales
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nbpr = cols / BG;

    float inv_rms = inv_rms_cache[row];
    float gs = *global_scale_ptr;
    const nv_bfloat16* row_x = x + (int64_t)row * cols;

    for (int qb = tid; qb < nbpr; qb += BS) {
        int off = qb * BG;
        bf16x8 d0 = bf16x8::load(row_x + off);
        bf16x8 d1 = bf16x8::load(row_x + off + 8);
        bf16x8 w0 = bf16x8::load(w + off);
        bf16x8 w1 = bf16x8::load(w + off + 8);

        // Compute normalized + activated values
        float v[16];
        #pragma unroll
        for (int k = 0; k < 8; ++k)
            v[k] = apply_act<ACT>(bf16_to_f32(d0[k])) * bf16_to_f32(w0[k]) * inv_rms;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
            v[8+k] = apply_act<ACT>(bf16_to_f32(d1[k])) * bf16_to_f32(w1[k]) * inv_rms;

        // Get true block amax (stored PRE-norm, scale now)
        float ba = block_amax[row * nbpr + qb] * inv_rms;

        // Compute block scale based on scale mode
        float bsi;
        __nv_fp8_e4m3 stored;

        if constexpr (SCALE == 0) {
            // Decode-centric
            stored = bs_decode(ba, gs);
            float sf = float(stored);
            if (sf == 0.0f) sf = 1.0f;
            bsi = 1.0f / (sf * gs);
        } else {
            // Encode-centric
            __nv_fp8_e4m3 mult = bm_encode(ba, gs);
            float mf = float(mult);
            if (mf == 0.0f) mf = 1.0f;
            bsi = mf * gs;
            stored = static_cast<__nv_fp8_e4m3>(1.0f / mf);
        }

        // PTX fused mul+cvt: 4 values at a time
        float2 s2 = {bsi, bsi};
        fp4x4 q0 = ptx_mul_cvt({v[0], v[1]}, {v[2], v[3]}, s2);
        fp4x4 q1 = ptx_mul_cvt({v[4], v[5]}, {v[6], v[7]}, s2);
        fp4x4 q2 = ptx_mul_cvt({v[8], v[9]}, {v[10], v[11]}, s2);
        fp4x4 q3 = ptx_mul_cvt({v[12], v[13]}, {v[14], v[15]}, s2);

        // 64-bit packed store (4 × uint16 = 8 bytes)
        int byte_off = (row * cols + off) / 2;
        uint64_t packed = (uint64_t)q0.bits
                        | ((uint64_t)q1.bits << 16)
                        | ((uint64_t)q2.bits << 32)
                        | ((uint64_t)q3.bits << 48);
        *reinterpret_cast<uint64_t*>(y + byte_off) = packed;

        // Store block scale
        scales[row * nbpr + qb] = stored;
    }
}

// =========================================================================
// Host launcher
// =========================================================================

void launch_fused_te_quant_v7(
    const nv_bfloat16* x, const nv_bfloat16* w,
    float epsilon, int rows, int cols,
    int norm_mode, int act_mode, int scale_mode,
    unsigned char* y, __nv_fp8_e4m3* scales,
    float* global_scale, float* inv_rms_cache,
    // Pre-allocated scratch (caller owns these)
    float* block_amax_scratch,       // [rows * cols/16] float
    unsigned int* global_amax_bits   // [1] uint32, must be zeroed before call
) {
    // Pass 1
    #define P1(N, A) v7_pass1<N, A><<<rows, BS>>>(x, w, epsilon, rows, cols, \
        block_amax_scratch, inv_rms_cache, global_amax_bits)
    switch (norm_mode * 3 + act_mode) {
        case 0: P1(0,0); break; case 1: P1(0,1); break; case 2: P1(0,2); break;
        case 3: P1(1,0); break; case 4: P1(1,1); break; case 5: P1(1,2); break;
        case 6: P1(2,0); break; case 7: P1(2,1); break; case 8: P1(2,2); break;
    }
    #undef P1

    // Global scale
    v7_global_scale<<<1, 1>>>(global_amax_bits, global_scale, scale_mode);

    // Pass 2
    #define P2(A, S) v7_pass2<A, S><<<rows, BS>>>(x, w, rows, cols, \
        block_amax_scratch, inv_rms_cache, global_scale, y, scales)
    switch (act_mode * 2 + scale_mode) {
        case 0: P2(0,0); break; case 1: P2(0,1); break;
        case 2: P2(1,0); break; case 3: P2(1,1); break;
        case 4: P2(2,0); break; case 5: P2(2,1); break;
    }
    #undef P2

    CUDA_CHECK(cudaGetLastError());
}

// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// V4: Fused RMSNorm + Activation + NVFP4 Quantization — TMA Bulk Copy
//
// Improvements over V3:
//   - Uses cp.async.bulk for async global→shared memory copies (TMA hardware)
//   - Data loads happen asynchronously while compute proceeds
//   - Processes data from shared memory instead of direct global loads
//
// Retains from V3: PTX fused mul+cvt, encode/decode scaling modes
//
// Architecture:
//   Pass 1: TMA load row to shmem → compute RMS stats + block amax from shmem
//   Pass 2: TMA load row to shmem → normalize + activate + quantize from shmem
//
// Scale modes: 0=decode-centric, 1=encode-centric
// Norm modes: 0=RMS, 1=AbsMax
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
// PTX Helpers for TMA
// =========================================================================

// mbarrier init
__device__ __forceinline__ void tma_mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"(mbar_ptr), "r"(count) : "memory");
}

// mbarrier arrive
__device__ __forceinline__ void tma_mbarrier_arrive(uint64_t* mbar) {
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];"
                 :: "r"(mbar_ptr) : "memory");
}

// mbarrier arrive with expected tx bytes
__device__ __forceinline__ void tma_mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx) {
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(mbar_ptr), "r"(tx) : "memory");
}

// mbarrier try_wait (parity-based)
__device__ __forceinline__ bool tma_mbarrier_try_wait(uint64_t* mbar, uint32_t parity) {
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
    uint32_t done;
    asm volatile(
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(done)
        : "r"(mbar_ptr), "r"(parity)
        : "memory");
    return done != 0;
}

// mbarrier wait (blocking)
__device__ __forceinline__ void tma_mbarrier_wait(uint64_t* mbar, uint32_t parity) {
    while (!tma_mbarrier_try_wait(mbar, parity)) {}
}

// cp.async.bulk: global → shared (1D, size in bytes)
__device__ __forceinline__ void tma_copy_1d_global_to_shared(
    void* dst_shared, const void* src_global, uint32_t num_bytes, uint64_t* mbar
) {
    uint32_t dst = __cvta_generic_to_shared(dst_shared);
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(dst), "l"(src_global), "r"(num_bytes), "r"(mbar_ptr)
        : "memory");
}

// fence proxy to ensure shared memory visibility
__device__ __forceinline__ void tma_fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

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
// PTX Fused mul + cvt (from V3)
// =========================================================================

struct fp4x4_packed {
    uint16_t bits;
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
// Scale Computation — Decode-Centric
// =========================================================================

__device__ __forceinline__ float compute_global_scale_decode(float global_amax) {
    if (global_amax == 0.0f) return 1.0f;
    return global_amax / (448.0f * 6.0f);
}

__device__ __forceinline__ __nv_fp8_e4m3 compute_block_scale_decode(float block_amax, float gs) {
    float s = block_amax / (6.0f * gs);
    s = fminf(s, 448.0f);
    return static_cast<__nv_fp8_e4m3>(s);
}

// =========================================================================
// Scale Computation — Encode-Centric
// =========================================================================

__device__ __forceinline__ float compute_global_scale_encode(float global_amax) {
    if (global_amax == 0.0f) return 1.0f;
    float s = 448.0f * 6.0f / global_amax;
    s = fminf(s, 3.4e38f);
    if (s == 0.0f) return 1.0f;
    return s;
}

__device__ __forceinline__ __nv_fp8_e4m3 compute_block_mult_encode(float block_amax, float s_enc) {
    if (block_amax <= 1.0e-9f) return static_cast<__nv_fp8_e4m3>(448.0f);
    float m = 6.0f / (block_amax * s_enc);
    m = fminf(m, 448.0f);
    return static_cast<__nv_fp8_e4m3>(m);
}

// =========================================================================
// Pass 1: TMA load → stats + block amax
// =========================================================================

// Shared memory layout: ROW_BYTES of bf16 data + mbarrier
// ROW_BYTES = cols * sizeof(bf16) = cols * 2

template<int BLOCK_SIZE = 256, int NORM_MODE = 0, int ACT_MODE = 0>
__global__ void fused_te_quant_v4_pass1(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float* __restrict__ block_amax_scratch,
    float* __restrict__ inv_rms_cache,
    unsigned int* __restrict__ global_amax_bits
) {
    extern __shared__ char smem[];
    
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    
    // Shared memory: data buffer + mbarrier
    uint32_t row_bytes = cols * sizeof(nv_bfloat16);
    nv_bfloat16* x_shared = reinterpret_cast<nv_bfloat16*>(smem);
    
    // Align mbarrier to 8 bytes
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        (reinterpret_cast<uintptr_t>(smem) + row_bytes + 7) & ~7ULL
    );
    
    // Initialize mbarrier
    if (tid == 0) {
        tma_mbarrier_init(mbar, BLOCK_SIZE);
        tma_fence_proxy_async();
    }
    __syncthreads();
    
    // 1D TMA: async copy entire row from global to shared
    if (tid == 0) {
        tma_copy_1d_global_to_shared(
            x_shared,
            x_ptr + row * cols,
            row_bytes,
            mbar
        );
        tma_mbarrier_arrive_expect_tx(mbar, row_bytes);
    } else {
        tma_mbarrier_arrive(mbar);
    }
    
    // Wait for TMA to complete
    tma_mbarrier_wait(mbar, 0);
    
    // Now process from shared memory
    float stat = 0.0f;
    float my_row_amax = 0.0f;

    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        int elem_start = block_id * BLOCK_GROUP_SIZE;
        
        // Load from SHARED memory (not global)
        bf16x8 data0 = bf16x8::load(x_shared + elem_start);
        bf16x8 data1 = bf16x8::load(x_shared + elem_start + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + elem_start);  // Weight still from global (small, cached in L2)
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
// Global scale kernel (same as V3)
// =========================================================================

__global__ void compute_global_scale_v4(
    const unsigned int* __restrict__ amax_bits,
    float* __restrict__ global_scale_ptr,
    int encode_centric
) {
    float amax = __uint_as_float(*amax_bits);
    if (amax == 0.0f) amax = 1.0f;
    if (encode_centric)
        *global_scale_ptr = compute_global_scale_encode(amax);
    else
        *global_scale_ptr = compute_global_scale_decode(amax);
}

// =========================================================================
// Pass 2: TMA load → normalize + activate + quantize
// =========================================================================

template<int BLOCK_SIZE = 256, int ACT_MODE = 0, int SCALE_MODE = 0>
__global__ void fused_te_quant_v4_pass2(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    int rows, int cols,
    const float* __restrict__ block_amax_scratch,
    const float* __restrict__ inv_rms_cache,
    const float* __restrict__ global_scale_ptr,
    unsigned char*    __restrict__ y_ptr,
    __nv_fp8_e4m3*    __restrict__ scale_ptr
) {
    extern __shared__ char smem[];
    
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;

    uint32_t row_bytes = cols * sizeof(nv_bfloat16);
    nv_bfloat16* x_shared = reinterpret_cast<nv_bfloat16*>(smem);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        (reinterpret_cast<uintptr_t>(smem) + row_bytes + 7) & ~7ULL
    );

    // Initialize mbarrier
    if (tid == 0) {
        tma_mbarrier_init(mbar, BLOCK_SIZE);
        tma_fence_proxy_async();
    }
    __syncthreads();

    // TMA load row
    if (tid == 0) {
        tma_copy_1d_global_to_shared(
            x_shared,
            x_ptr + row * cols,
            row_bytes,
            mbar
        );
        tma_mbarrier_arrive_expect_tx(mbar, row_bytes);
    } else {
        tma_mbarrier_arrive(mbar);
    }
    tma_mbarrier_wait(mbar, 0);

    float inv_rms = inv_rms_cache[row];
    float global_scale = *global_scale_ptr;

    for (int block_id = tid; block_id < num_blocks_per_row; block_id += BLOCK_SIZE) {
        int elem_start = block_id * BLOCK_GROUP_SIZE;

        // Load from shared memory
        bf16x8 data0 = bf16x8::load(x_shared + elem_start);
        bf16x8 data1 = bf16x8::load(x_shared + elem_start + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + elem_start);
        bf16x8 w1 = bf16x8::load(w_ptr + elem_start + 8);

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
        float block_scale_inv;
        __nv_fp8_e4m3 stored_scale;

        if constexpr (SCALE_MODE == 0) {
            stored_scale = compute_block_scale_decode(block_amax, global_scale);
            float s_float = float(stored_scale);
            if (s_float == 0.0f) s_float = 1.0f;
            block_scale_inv = 1.0f / (s_float * global_scale);
        } else {
            __nv_fp8_e4m3 mult_fp8 = compute_block_mult_encode(block_amax, global_scale);
            float mult_float = float(mult_fp8);
            if (mult_float == 0.0f) mult_float = 1.0f;
            block_scale_inv = mult_float * global_scale;
            stored_scale = static_cast<__nv_fp8_e4m3>(1.0f / mult_float);
        }

        // PTX fused mul+cvt
        float2 scale_2x = {block_scale_inv, block_scale_inv};

        float2 in01 = {vals[0], vals[1]};
        float2 in23 = {vals[2], vals[3]};
        fp4x4_packed q0 = mul_cvt_fp32_to_fp4_4x(in01, in23, scale_2x);

        in01 = {vals[4], vals[5]};
        in23 = {vals[6], vals[7]};
        fp4x4_packed q1 = mul_cvt_fp32_to_fp4_4x(in01, in23, scale_2x);

        in01 = {vals[8], vals[9]};
        in23 = {vals[10], vals[11]};
        fp4x4_packed q2 = mul_cvt_fp32_to_fp4_4x(in01, in23, scale_2x);

        in01 = {vals[12], vals[13]};
        in23 = {vals[14], vals[15]};
        fp4x4_packed q3 = mul_cvt_fp32_to_fp4_4x(in01, in23, scale_2x);

        int byte_offset = (row * cols + elem_start) / 2;
        uint16_t* out16 = reinterpret_cast<uint16_t*>(y_ptr + byte_offset);
        out16[0] = q0.bits;
        out16[1] = q1.bits;
        out16[2] = q2.bits;
        out16[3] = q3.bits;

        scale_ptr[row * num_blocks_per_row + block_id] = stored_scale;
    }
}

// =========================================================================
// Host Launcher
// =========================================================================

void launch_fused_te_quant_v4(
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    float epsilon,
    int rows, int cols,
    int norm_mode,      // 0=RMS, 1=AbsMax
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

    // Shared memory: row data + alignment padding + mbarrier (8 bytes)
    uint32_t row_bytes = cols * sizeof(nv_bfloat16);
    uint32_t shmem_size = row_bytes + 8 + 8;  // data + align + mbar

    // Check shmem limits
    int max_shmem = 0;
    cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (shmem_size > (uint32_t)max_shmem) {
        // Fallback: for very large K, we can't fit a full row in shmem
        // In practice K=8192 → 16KB which is fine (max is ~228KB on GB200)
        throw std::runtime_error("Row too large for shared memory");
    }

    unsigned int* global_amax_bits;
    cudaMallocAsync(&global_amax_bits, sizeof(unsigned int), 0);
    cudaMemsetAsync(global_amax_bits, 0, sizeof(unsigned int), 0);

    // Pass 1
    #define DISPATCH_P1_V4(NM, AM) do { \
        auto kernel = fused_te_quant_v4_pass1<BLOCK_SIZE, NM, AM>; \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size); \
        kernel<<<rows, BLOCK_SIZE, shmem_size>>>( \
            x, w, epsilon, rows, cols, block_amax_scratch, inv_rms_cache, global_amax_bits); \
    } while(0)

    switch (norm_mode * 3 + act_mode) {
        case 0: DISPATCH_P1_V4(0, 0); break;
        case 1: DISPATCH_P1_V4(0, 1); break;
        case 2: DISPATCH_P1_V4(0, 2); break;
        case 3: DISPATCH_P1_V4(1, 0); break;
        case 4: DISPATCH_P1_V4(1, 1); break;
        case 5: DISPATCH_P1_V4(1, 2); break;
    }
    #undef DISPATCH_P1_V4

    // Global scale
    compute_global_scale_v4<<<1, 1>>>(global_amax_bits, global_scale, scale_mode);

    // Pass 2
    #define DISPATCH_P2_V4(AM, SM) do { \
        auto kernel = fused_te_quant_v4_pass2<BLOCK_SIZE, AM, SM>; \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size); \
        kernel<<<rows, BLOCK_SIZE, shmem_size>>>( \
            x, w, rows, cols, block_amax_scratch, inv_rms_cache, global_scale, y, scales); \
    } while(0)

    switch (act_mode * 2 + scale_mode) {
        case 0: DISPATCH_P2_V4(0, 0); break;
        case 1: DISPATCH_P2_V4(0, 1); break;
        case 2: DISPATCH_P2_V4(1, 0); break;
        case 3: DISPATCH_P2_V4(1, 1); break;
        case 4: DISPATCH_P2_V4(2, 0); break;
        case 5: DISPATCH_P2_V4(2, 1); break;
    }
    #undef DISPATCH_P2_V4

    CUDA_CHECK(cudaGetLastError());
    cudaFreeAsync(global_amax_bits, 0);
}

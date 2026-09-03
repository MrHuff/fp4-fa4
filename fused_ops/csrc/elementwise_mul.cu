// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused elementwise multiply + amax kernel: out = h1 * h3, amax = max(|out|)
//
// The amax is computed in the SAME pass as the multiply,
// avoiding a second kernel launch for nvte_compute_amax_with_config.
// NVFP4 quantization needs amax as a global tensor scale:
//   tensor_scale = amax / (6.0 * 448.0)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------- Warp/Block reductions (same pattern as te_fused_pass1.cu) ----------

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float warp_vals[8]; // up to 256/32 = 8 warps
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    __syncthreads(); // Prevent race if called multiple times

    val = warp_reduce_max(val);
    if (lane == 0) warp_vals[warp] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? warp_vals[threadIdx.x] : 0.0f;
    if (warp == 0) val = warp_reduce_max(val);
    return val;
}

// Positive-float atomicMax (works because positive floats have same ordering as uint32)
__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    if (value >= 0) {
        atomicMax(reinterpret_cast<unsigned int*>(addr),
                  __float_as_uint(value));
    }
}

// ---------- Fused mul + amax kernel ----------

__global__ void elementwise_mul_amax_kernel(
    const nv_bfloat16* __restrict__ h1,
    const nv_bfloat16* __restrict__ h3,
    nv_bfloat16* __restrict__ out,
    float* __restrict__ global_amax,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    float my_amax = 0.0f;

    // Vectorized path: process 4 bf16 at a time (8 bytes = one int2)
    int64_t vec_numel = numel / 4;
    const int2* h1_vec = reinterpret_cast<const int2*>(h1);
    const int2* h3_vec = reinterpret_cast<const int2*>(h3);
    int2* out_vec = reinterpret_cast<int2*>(out);

    for (int64_t i = idx; i < vec_numel; i += stride) {
        int2 a = h1_vec[i];
        int2 b = h3_vec[i];

        nv_bfloat162 a0 = *reinterpret_cast<nv_bfloat162*>(&a.x);
        nv_bfloat162 a1 = *reinterpret_cast<nv_bfloat162*>(&a.y);
        nv_bfloat162 b0 = *reinterpret_cast<nv_bfloat162*>(&b.x);
        nv_bfloat162 b1 = *reinterpret_cast<nv_bfloat162*>(&b.y);

        nv_bfloat162 c0 = __hmul2(a0, b0);
        nv_bfloat162 c1 = __hmul2(a1, b1);

        // Track amax of output values
        float2 f0 = __bfloat1622float2(c0);
        float2 f1 = __bfloat1622float2(c1);
        my_amax = fmaxf(my_amax, fmaxf(fabsf(f0.x), fabsf(f0.y)));
        my_amax = fmaxf(my_amax, fmaxf(fabsf(f1.x), fabsf(f1.y)));

        int2 result;
        result.x = *reinterpret_cast<int*>(&c0);
        result.y = *reinterpret_cast<int*>(&c1);
        out_vec[i] = result;
    }

    // Handle remainder elements (if numel not divisible by 4)
    int64_t base = vec_numel * 4;
    for (int64_t i = base + idx; i < numel; i += stride) {
        float v1 = __bfloat162float(h1[i]);
        float v3 = __bfloat162float(h3[i]);
        float prod = v1 * v3;
        my_amax = fmaxf(my_amax, fabsf(prod));
        out[i] = __float2bfloat16_rn(prod);
    }

    // Reduce amax across the block and atomicMax into global
    float block_amax = block_reduce_max(my_amax);
    if (threadIdx.x == 0 && block_amax > 0.0f) {
        atomicMaxFloat(global_amax, block_amax);
    }
}

// ---------- Legacy launcher (no amax) for backward compat ----------

extern "C" void launch_elementwise_mul(
    const void* h1, const void* h3, void* out,
    int64_t numel, cudaStream_t stream
) {
    // This wrapper is kept for backward compatibility but should not be used
    // for the fused path. Use launch_elementwise_mul_amax instead.
    constexpr int BLOCK = 256;
    int64_t grid = (numel / 4 + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    // Allocate a dummy amax (not used, just satisfies the kernel signature)
    // For the legacy path, we just ignore amax
    float* dummy_amax = nullptr;
    cudaMallocAsync(&dummy_amax, sizeof(float), stream);
    cudaMemsetAsync(dummy_amax, 0, sizeof(float), stream);

    elementwise_mul_amax_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const nv_bfloat16*>(h1),
        static_cast<const nv_bfloat16*>(h3),
        static_cast<nv_bfloat16*>(out),
        dummy_amax,
        numel
    );

    cudaFreeAsync(dummy_amax, stream);
}

// ---------- New launcher: fused mul + amax ----------

extern "C" void launch_elementwise_mul_amax(
    const void* h1, const void* h3, void* out,
    float* global_amax,
    int64_t numel, cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int64_t grid = (numel / 4 + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    elementwise_mul_amax_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const nv_bfloat16*>(h1),
        static_cast<const nv_bfloat16*>(h3),
        static_cast<nv_bfloat16*>(out),
        global_amax,
        numel
    );
}

// ==========================================================================
// Dual elementwise multiply + amax kernel
//
// Given dh, h3, h1  (all bf16, same shape), computes:
//   out1 = dh * h3   (with global amax1)
//   out2 = dh * h1   (with global amax2)
//
// This fuses 2 element-wise multiplies + 2 amax reductions into 1 kernel.
// ==========================================================================

__global__ void dual_elementwise_mul_amax_kernel(
    const nv_bfloat16* __restrict__ dh,
    const nv_bfloat16* __restrict__ h3,
    const nv_bfloat16* __restrict__ h1,
    nv_bfloat16* __restrict__ out1,   // dh * h3
    nv_bfloat16* __restrict__ out2,   // dh * h1
    float* __restrict__ global_amax1,
    float* __restrict__ global_amax2,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    float my_amax1 = 0.0f;
    float my_amax2 = 0.0f;

    // Vectorized path: process 4 bf16 at a time (8 bytes = one int2)
    int64_t vec_numel = numel / 4;
    const int2* dh_vec  = reinterpret_cast<const int2*>(dh);
    const int2* h3_vec  = reinterpret_cast<const int2*>(h3);
    const int2* h1_vec  = reinterpret_cast<const int2*>(h1);
    int2* out1_vec = reinterpret_cast<int2*>(out1);
    int2* out2_vec = reinterpret_cast<int2*>(out2);

    for (int64_t i = idx; i < vec_numel; i += stride) {
        int2 d = dh_vec[i];
        int2 a = h3_vec[i];
        int2 b = h1_vec[i];

        nv_bfloat162 d0 = *reinterpret_cast<nv_bfloat162*>(&d.x);
        nv_bfloat162 d1 = *reinterpret_cast<nv_bfloat162*>(&d.y);
        nv_bfloat162 a0 = *reinterpret_cast<nv_bfloat162*>(&a.x);
        nv_bfloat162 a1 = *reinterpret_cast<nv_bfloat162*>(&a.y);
        nv_bfloat162 b0 = *reinterpret_cast<nv_bfloat162*>(&b.x);
        nv_bfloat162 b1 = *reinterpret_cast<nv_bfloat162*>(&b.y);

        // out1 = dh * h3
        nv_bfloat162 c0 = __hmul2(d0, a0);
        nv_bfloat162 c1 = __hmul2(d1, a1);
        
        // Compute silu(h1)
        float2 b0_f2 = __bfloat1622float2(b0);
        float2 b1_f2 = __bfloat1622float2(b1);
        b0_f2.x = b0_f2.x / (1.0f + expf(-b0_f2.x));
        b0_f2.y = b0_f2.y / (1.0f + expf(-b0_f2.y));
        b1_f2.x = b1_f2.x / (1.0f + expf(-b1_f2.x));
        b1_f2.y = b1_f2.y / (1.0f + expf(-b1_f2.y));
        nv_bfloat162 b0_silu = __float22bfloat162_rn(b0_f2);
        nv_bfloat162 b1_silu = __float22bfloat162_rn(b1_f2);
        
        // out2 = dh * silu(h1)
        nv_bfloat162 e0 = __hmul2(d0, b0_silu);
        nv_bfloat162 e1 = __hmul2(d1, b1_silu);

        // Track amax for output 1
        float2 f0 = __bfloat1622float2(c0);
        float2 f1 = __bfloat1622float2(c1);
        my_amax1 = fmaxf(my_amax1, fmaxf(fabsf(f0.x), fabsf(f0.y)));
        my_amax1 = fmaxf(my_amax1, fmaxf(fabsf(f1.x), fabsf(f1.y)));

        // Track amax for output 2
        float2 g0 = __bfloat1622float2(e0);
        float2 g1 = __bfloat1622float2(e1);
        my_amax2 = fmaxf(my_amax2, fmaxf(fabsf(g0.x), fabsf(g0.y)));
        my_amax2 = fmaxf(my_amax2, fmaxf(fabsf(g1.x), fabsf(g1.y)));

        int2 r1, r2;
        r1.x = *reinterpret_cast<int*>(&c0);
        r1.y = *reinterpret_cast<int*>(&c1);
        r2.x = *reinterpret_cast<int*>(&e0);
        r2.y = *reinterpret_cast<int*>(&e1);
        out1_vec[i] = r1;
        out2_vec[i] = r2;
    }

    // Handle remainder elements (if numel not divisible by 4)
    int64_t base = vec_numel * 4;
    for (int64_t i = base + idx; i < numel; i += stride) {
        float vd = __bfloat162float(dh[i]);
        float v3 = __bfloat162float(h3[i]);
        float v1 = __bfloat162float(h1[i]);
        
        float silu_v1 = v1 / (1.0f + expf(-v1));
        
        float prod1 = vd * v3;
        float prod2 = vd * silu_v1;
        my_amax1 = fmaxf(my_amax1, fabsf(prod1));
        my_amax2 = fmaxf(my_amax2, fabsf(prod2));
        out1[i] = __float2bfloat16_rn(prod1);
        out2[i] = __float2bfloat16_rn(prod2);
    }

    // Reduce amax across the block and atomicMax into global
    float block_amax1 = block_reduce_max(my_amax1);
    float block_amax2 = block_reduce_max(my_amax2);
    if (threadIdx.x == 0) {
        if (block_amax1 > 0.0f) atomicMaxFloat(global_amax1, block_amax1);
        if (block_amax2 > 0.0f) atomicMaxFloat(global_amax2, block_amax2);
    }
}

extern "C" void launch_dual_elementwise_mul_amax(
    const void* dh, const void* h3, const void* h1,
    void* out1, void* out2,
    float* global_amax1, float* global_amax2,
    int64_t numel, cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int64_t grid = (numel / 4 + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    dual_elementwise_mul_amax_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const nv_bfloat16*>(dh),
        static_cast<const nv_bfloat16*>(h3),
        static_cast<const nv_bfloat16*>(h1),
        static_cast<nv_bfloat16*>(out1),
        static_cast<nv_bfloat16*>(out2),
        global_amax1,
        global_amax2,
        numel
    );
}

// ==========================================================================
// Fused SiLU + multiply + amax kernel (FORWARD)
//
// Computes: out = silu(h1_raw) * h3, amax = max(|out|)
//
// For the corrected SwiGLU forward:
//   h1_raw = W1 @ rmsnorm(x)   (raw projection, no activation)
//   h3     = W3 @ rmsnorm(x)   (gate projection)
//   h      = silu(h1_raw) * h3  (this kernel)
// ==========================================================================

__device__ __forceinline__ float device_silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_silu_mul_amax_kernel(
    const nv_bfloat16* __restrict__ h1_raw,  // raw W1 output (no activation)
    const nv_bfloat16* __restrict__ h3,      // W3 output (gate)
    nv_bfloat16* __restrict__ out,           // silu(h1_raw) * h3
    float* __restrict__ global_amax,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    float my_amax = 0.0f;

    // Vectorized path: process 4 bf16 at a time
    int64_t vec_numel = numel / 4;
    const int2* h1_vec = reinterpret_cast<const int2*>(h1_raw);
    const int2* h3_vec = reinterpret_cast<const int2*>(h3);
    int2* out_vec = reinterpret_cast<int2*>(out);

    for (int64_t i = idx; i < vec_numel; i += stride) {
        int2 a = h1_vec[i];
        int2 b = h3_vec[i];

        // Unpack h1_raw to float, apply silu
        nv_bfloat162 a0 = *reinterpret_cast<nv_bfloat162*>(&a.x);
        nv_bfloat162 a1 = *reinterpret_cast<nv_bfloat162*>(&a.y);
        float2 a0_f = __bfloat1622float2(a0);
        float2 a1_f = __bfloat1622float2(a1);

        a0_f.x = device_silu_f(a0_f.x);
        a0_f.y = device_silu_f(a0_f.y);
        a1_f.x = device_silu_f(a1_f.x);
        a1_f.y = device_silu_f(a1_f.y);

        // Unpack h3 to float
        nv_bfloat162 b0 = *reinterpret_cast<nv_bfloat162*>(&b.x);
        nv_bfloat162 b1 = *reinterpret_cast<nv_bfloat162*>(&b.y);
        float2 b0_f = __bfloat1622float2(b0);
        float2 b1_f = __bfloat1622float2(b1);

        // Compute silu(h1_raw) * h3 in float, track amax
        float2 c0_f, c1_f;
        c0_f.x = a0_f.x * b0_f.x;
        c0_f.y = a0_f.y * b0_f.y;
        c1_f.x = a1_f.x * b1_f.x;
        c1_f.y = a1_f.y * b1_f.y;

        my_amax = fmaxf(my_amax, fmaxf(fabsf(c0_f.x), fabsf(c0_f.y)));
        my_amax = fmaxf(my_amax, fmaxf(fabsf(c1_f.x), fabsf(c1_f.y)));

        // Convert back to bf16 and store
        nv_bfloat162 c0 = __float22bfloat162_rn(c0_f);
        nv_bfloat162 c1 = __float22bfloat162_rn(c1_f);
        int2 result;
        result.x = *reinterpret_cast<int*>(&c0);
        result.y = *reinterpret_cast<int*>(&c1);
        out_vec[i] = result;
    }

    // Remainder
    int64_t base = vec_numel * 4;
    for (int64_t i = base + idx; i < numel; i += stride) {
        float v1 = device_silu_f(__bfloat162float(h1_raw[i]));
        float v3 = __bfloat162float(h3[i]);
        float prod = v1 * v3;
        my_amax = fmaxf(my_amax, fabsf(prod));
        out[i] = __float2bfloat16_rn(prod);
    }

    float block_amax = block_reduce_max(my_amax);
    if (threadIdx.x == 0 && block_amax > 0.0f) {
        atomicMaxFloat(global_amax, block_amax);
    }
}

extern "C" void launch_fused_silu_mul_amax(
    const void* h1_raw, const void* h3, void* out,
    float* global_amax,
    int64_t numel, cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int64_t grid = (numel / 4 + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    fused_silu_mul_amax_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const nv_bfloat16*>(h1_raw),
        static_cast<const nv_bfloat16*>(h3),
        static_cast<nv_bfloat16*>(out),
        global_amax,
        numel
    );
}


// ==========================================================================
// Fused SiLU-derivative dual multiply + amax kernel (BACKWARD)
//
// For the corrected SwiGLU backward where h = silu(h1_raw) * h3:
//   out1 = dh * h3 * silu'(h1_raw)     — gradient w.r.t. h1_raw (before silu)
//   out2 = dh * silu(h1_raw)           — gradient w.r.t. h3 (gate)
//
// silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//          = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
// ==========================================================================

__global__ void fused_silu_deriv_dual_mul_amax_kernel(
    const nv_bfloat16* __restrict__ dh,       // (M, H) upstream gradient
    const nv_bfloat16* __restrict__ h3,       // (M, H) gate output (saved)
    const nv_bfloat16* __restrict__ h1_raw,   // (M, H) raw W1 output (saved, pre-silu)
    nv_bfloat16* __restrict__ out1,           // dh * h3 * silu'(h1_raw)
    nv_bfloat16* __restrict__ out2,           // dh * silu(h1_raw)
    float* __restrict__ global_amax1,
    float* __restrict__ global_amax2,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    float my_amax1 = 0.0f;
    float my_amax2 = 0.0f;

    int64_t vec_numel = numel / 4;
    const int2* dh_vec  = reinterpret_cast<const int2*>(dh);
    const int2* h3_vec  = reinterpret_cast<const int2*>(h3);
    const int2* h1_vec  = reinterpret_cast<const int2*>(h1_raw);
    int2* out1_vec = reinterpret_cast<int2*>(out1);
    int2* out2_vec = reinterpret_cast<int2*>(out2);

    for (int64_t i = idx; i < vec_numel; i += stride) {
        int2 d = dh_vec[i];
        int2 a = h3_vec[i];
        int2 b = h1_vec[i];

        // Unpack to float
        nv_bfloat162 d0 = *reinterpret_cast<nv_bfloat162*>(&d.x);
        nv_bfloat162 d1 = *reinterpret_cast<nv_bfloat162*>(&d.y);
        float2 d0_f = __bfloat1622float2(d0);
        float2 d1_f = __bfloat1622float2(d1);

        nv_bfloat162 a0 = *reinterpret_cast<nv_bfloat162*>(&a.x);
        nv_bfloat162 a1 = *reinterpret_cast<nv_bfloat162*>(&a.y);
        float2 a0_f = __bfloat1622float2(a0);
        float2 a1_f = __bfloat1622float2(a1);

        nv_bfloat162 b0 = *reinterpret_cast<nv_bfloat162*>(&b.x);
        nv_bfloat162 b1 = *reinterpret_cast<nv_bfloat162*>(&b.y);
        float2 b0_f = __bfloat1622float2(b0);
        float2 b1_f = __bfloat1622float2(b1);

        // Compute sigmoid, silu, silu' for each h1_raw element
        // sigmoid(x) = 1 / (1 + exp(-x))
        // silu(x) = x * sigmoid(x)
        // silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        //          = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

        float sig0x = 1.0f / (1.0f + expf(-b0_f.x));
        float sig0y = 1.0f / (1.0f + expf(-b0_f.y));
        float sig1x = 1.0f / (1.0f + expf(-b1_f.x));
        float sig1y = 1.0f / (1.0f + expf(-b1_f.y));

        float silu0x = b0_f.x * sig0x;
        float silu0y = b0_f.y * sig0y;
        float silu1x = b1_f.x * sig1x;
        float silu1y = b1_f.y * sig1y;

        // silu'(x) = sig(x) * (1 + x - silu(x))  [equivalent, more stable form]
        float silup0x = sig0x * (1.0f + b0_f.x - silu0x);
        float silup0y = sig0y * (1.0f + b0_f.y - silu0y);
        float silup1x = sig1x * (1.0f + b1_f.x - silu1x);
        float silup1y = sig1y * (1.0f + b1_f.y - silu1y);

        // out1 = dh * h3 * silu'(h1_raw)
        float2 c0_f, c1_f;
        c0_f.x = d0_f.x * a0_f.x * silup0x;
        c0_f.y = d0_f.y * a0_f.y * silup0y;
        c1_f.x = d1_f.x * a1_f.x * silup1x;
        c1_f.y = d1_f.y * a1_f.y * silup1y;

        // out2 = dh * silu(h1_raw)
        float2 e0_f, e1_f;
        e0_f.x = d0_f.x * silu0x;
        e0_f.y = d0_f.y * silu0y;
        e1_f.x = d1_f.x * silu1x;
        e1_f.y = d1_f.y * silu1y;

        // Track amaxes
        my_amax1 = fmaxf(my_amax1, fmaxf(fabsf(c0_f.x), fabsf(c0_f.y)));
        my_amax1 = fmaxf(my_amax1, fmaxf(fabsf(c1_f.x), fabsf(c1_f.y)));
        my_amax2 = fmaxf(my_amax2, fmaxf(fabsf(e0_f.x), fabsf(e0_f.y)));
        my_amax2 = fmaxf(my_amax2, fmaxf(fabsf(e1_f.x), fabsf(e1_f.y)));

        // Store
        nv_bfloat162 c0 = __float22bfloat162_rn(c0_f);
        nv_bfloat162 c1 = __float22bfloat162_rn(c1_f);
        nv_bfloat162 e0 = __float22bfloat162_rn(e0_f);
        nv_bfloat162 e1 = __float22bfloat162_rn(e1_f);

        int2 r1, r2;
        r1.x = *reinterpret_cast<int*>(&c0);
        r1.y = *reinterpret_cast<int*>(&c1);
        r2.x = *reinterpret_cast<int*>(&e0);
        r2.y = *reinterpret_cast<int*>(&e1);
        out1_vec[i] = r1;
        out2_vec[i] = r2;
    }

    // Remainder
    int64_t base = vec_numel * 4;
    for (int64_t i = base + idx; i < numel; i += stride) {
        float vd = __bfloat162float(dh[i]);
        float v3 = __bfloat162float(h3[i]);
        float v1 = __bfloat162float(h1_raw[i]);

        float sig = 1.0f / (1.0f + expf(-v1));
        float silu_v1 = v1 * sig;
        float silup_v1 = sig * (1.0f + v1 - silu_v1);

        float prod1 = vd * v3 * silup_v1;
        float prod2 = vd * silu_v1;
        my_amax1 = fmaxf(my_amax1, fabsf(prod1));
        my_amax2 = fmaxf(my_amax2, fabsf(prod2));
        out1[i] = __float2bfloat16_rn(prod1);
        out2[i] = __float2bfloat16_rn(prod2);
    }

    float block_amax1 = block_reduce_max(my_amax1);
    float block_amax2 = block_reduce_max(my_amax2);
    if (threadIdx.x == 0) {
        if (block_amax1 > 0.0f) atomicMaxFloat(global_amax1, block_amax1);
        if (block_amax2 > 0.0f) atomicMaxFloat(global_amax2, block_amax2);
    }
}

extern "C" void launch_fused_silu_deriv_dual_mul_amax(
    const void* dh, const void* h3, const void* h1_raw,
    void* out1, void* out2,
    float* global_amax1, float* global_amax2,
    int64_t numel, cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int64_t grid = (numel / 4 + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    fused_silu_deriv_dual_mul_amax_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const nv_bfloat16*>(dh),
        static_cast<const nv_bfloat16*>(h3),
        static_cast<const nv_bfloat16*>(h1_raw),
        static_cast<nv_bfloat16*>(out1),
        static_cast<nv_bfloat16*>(out2),
        global_amax1,
        global_amax2,
        numel
    );
}


// ==========================================================================
// STRIDED variants: read h1_raw and h3 from a single (M, 2H) buffer
// without requiring .contiguous() copies.
//
// h13 layout: (M, 2H) row-major, h1_raw = h13[:, 0:H], h3 = h13[:, H:2H]
// ==========================================================================

// Forward: out = silu(h13[:, 0:H]) * h13[:, H:2H]
__global__ void fused_silu_mul_strided_amax_kernel(
    const nv_bfloat16* __restrict__ h13,  // (M, 2H) contiguous
    nv_bfloat16* __restrict__ out,         // (M, H) contiguous output
    float* __restrict__ global_amax,
    int64_t M, int64_t H
) {
    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    float my_amax = 0.0f;

    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        const nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;

        nv_bfloat162 h1_val = *reinterpret_cast<const nv_bfloat162*>(h1_ptr);
        nv_bfloat162 h3_val = *reinterpret_cast<const nv_bfloat162*>(h3_ptr);

        float2 h1_f = __bfloat1622float2(h1_val);
        h1_f.x = device_silu_f(h1_f.x);
        h1_f.y = device_silu_f(h1_f.y);

        float2 h3_f = __bfloat1622float2(h3_val);

        float2 r_f;
        r_f.x = h1_f.x * h3_f.x;
        r_f.y = h1_f.y * h3_f.y;

        my_amax = fmaxf(my_amax, fmaxf(fabsf(r_f.x), fabsf(r_f.y)));

        *reinterpret_cast<nv_bfloat162*>(out + elem) = __float22bfloat162_rn(r_f);
    }

    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float h1_f = device_silu_f(__bfloat162float(h13[row * row_stride + col]));
        float h3_f = __bfloat162float(h13[row * row_stride + H + col]);
        float prod = h1_f * h3_f;
        my_amax = fmaxf(my_amax, fabsf(prod));
        out[i] = __float2bfloat16_rn(prod);
    }

    float block_amax = block_reduce_max(my_amax);
    if (threadIdx.x == 0 && block_amax > 0.0f) {
        atomicMaxFloat(global_amax, block_amax);
    }
}

extern "C" void launch_fused_silu_mul_strided_amax(
    const void* h13, void* out,
    float* global_amax,
    int64_t M, int64_t H, cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int64_t total_pairs = M * H / 2;
    int64_t grid = (total_pairs + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    fused_silu_mul_strided_amax_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const nv_bfloat16*>(h13),
        static_cast<nv_bfloat16*>(out),
        global_amax,
        M, H
    );
}

// Backward strided: dh contiguous + h1_raw/h3 from h13
// out1 = dh * h3 * silu'(h1_raw)
// out2 = dh * silu(h1_raw)
__global__ void fused_silu_deriv_dual_mul_strided_amax_kernel(
    const nv_bfloat16* __restrict__ dh,    // (M, H) contiguous
    const nv_bfloat16* __restrict__ h13,   // (M, 2H) contiguous
    nv_bfloat16* __restrict__ out1,
    nv_bfloat16* __restrict__ out2,
    float* __restrict__ global_amax1,
    float* __restrict__ global_amax2,
    int64_t M, int64_t H
) {
    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    float my_amax1 = 0.0f, my_amax2 = 0.0f;

    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        nv_bfloat162 dh_val = *reinterpret_cast<const nv_bfloat162*>(dh + elem);
        float2 dh_f = __bfloat1622float2(dh_val);

        const nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;
        float2 h1_f = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(h1_ptr));
        float2 h3_f = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(h3_ptr));

        float sigx = 1.0f / (1.0f + expf(-h1_f.x));
        float sigy = 1.0f / (1.0f + expf(-h1_f.y));
        float silux = h1_f.x * sigx;
        float siluy = h1_f.y * sigy;
        float silupx = sigx * (1.0f + h1_f.x - silux);
        float silupy = sigy * (1.0f + h1_f.y - siluy);

        float2 c_f = { dh_f.x * h3_f.x * silupx, dh_f.y * h3_f.y * silupy };
        float2 e_f = { dh_f.x * silux, dh_f.y * siluy };

        my_amax1 = fmaxf(my_amax1, fmaxf(fabsf(c_f.x), fabsf(c_f.y)));
        my_amax2 = fmaxf(my_amax2, fmaxf(fabsf(e_f.x), fabsf(e_f.y)));

        *reinterpret_cast<nv_bfloat162*>(out1 + elem) = __float22bfloat162_rn(c_f);
        *reinterpret_cast<nv_bfloat162*>(out2 + elem) = __float22bfloat162_rn(e_f);
    }

    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float vd = __bfloat162float(dh[i]);
        float v1 = __bfloat162float(h13[row * row_stride + col]);
        float v3 = __bfloat162float(h13[row * row_stride + H + col]);
        float sig = 1.0f / (1.0f + expf(-v1));
        float silu_v1 = v1 * sig;
        float silup_v1 = sig * (1.0f + v1 - silu_v1);
        out1[i] = __float2bfloat16_rn(vd * v3 * silup_v1);
        out2[i] = __float2bfloat16_rn(vd * silu_v1);
    }

    float ba1 = block_reduce_max(my_amax1);
    float ba2 = block_reduce_max(my_amax2);
    if (threadIdx.x == 0) {
        if (ba1 > 0.0f) atomicMaxFloat(global_amax1, ba1);
        if (ba2 > 0.0f) atomicMaxFloat(global_amax2, ba2);
    }
}

extern "C" void launch_fused_silu_deriv_dual_mul_strided_amax(
    const void* dh, const void* h13,
    void* out1, void* out2,
    float* global_amax1, float* global_amax2,
    int64_t M, int64_t H, cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int64_t total_pairs = M * H / 2;
    int64_t grid = (total_pairs + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    fused_silu_deriv_dual_mul_strided_amax_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const nv_bfloat16*>(dh),
        static_cast<const nv_bfloat16*>(h13),
        static_cast<nv_bfloat16*>(out1),
        static_cast<nv_bfloat16*>(out2),
        global_amax1,
        global_amax2,
        M, H
    );
}


// ── Backward strided: reads h13 strided, writes dh13 interleaved (M,2H) ──
// Eliminates torch.cat by writing dh1 to dh13[:,0:H] and dh3 to dh13[:,H:2H]
__global__ void fused_silu_deriv_dual_mul_strided_interleaved_amax_kernel(
    const nv_bfloat16* __restrict__ dh,    // (M, H) contiguous
    const nv_bfloat16* __restrict__ h13,   // (M, 2H) contiguous
    nv_bfloat16* __restrict__ dh13,        // (M, 2H) contiguous output
    float* __restrict__ global_amax1,
    float* __restrict__ global_amax2,
    int64_t M, int64_t H
) {
    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    float my_amax1 = 0.0f, my_amax2 = 0.0f;

    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        nv_bfloat162 dh_val = *reinterpret_cast<const nv_bfloat162*>(dh + elem);
        float2 dh_f = __bfloat1622float2(dh_val);

        const nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;
        float2 h1_f = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(h1_ptr));
        float2 h3_f = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(h3_ptr));

        float sigx = 1.0f / (1.0f + expf(-h1_f.x));
        float sigy = 1.0f / (1.0f + expf(-h1_f.y));
        float silux = h1_f.x * sigx;
        float siluy = h1_f.y * sigy;
        float silupx = sigx * (1.0f + h1_f.x - silux);
        float silupy = sigy * (1.0f + h1_f.y - siluy);

        float2 c_f = { dh_f.x * h3_f.x * silupx, dh_f.y * h3_f.y * silupy };
        float2 e_f = { dh_f.x * silux, dh_f.y * siluy };

        my_amax1 = fmaxf(my_amax1, fmaxf(fabsf(c_f.x), fabsf(c_f.y)));
        my_amax2 = fmaxf(my_amax2, fmaxf(fabsf(e_f.x), fabsf(e_f.y)));

        // Write interleaved: dh1 to dh13[:,0:H], dh3 to dh13[:,H:2H]
        *reinterpret_cast<nv_bfloat162*>(dh13 + row * row_stride + col) = __float22bfloat162_rn(c_f);
        *reinterpret_cast<nv_bfloat162*>(dh13 + row * row_stride + H + col) = __float22bfloat162_rn(e_f);
    }

    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float vd = __bfloat162float(dh[i]);
        float v1 = __bfloat162float(h13[row * row_stride + col]);
        float v3 = __bfloat162float(h13[row * row_stride + H + col]);
        float sig = 1.0f / (1.0f + expf(-v1));
        float silu_v1 = v1 * sig;
        float silup_v1 = sig * (1.0f + v1 - silu_v1);
        dh13[row * row_stride + col] = __float2bfloat16_rn(vd * v3 * silup_v1);
        dh13[row * row_stride + H + col] = __float2bfloat16_rn(vd * silu_v1);
    }

    float ba1 = block_reduce_max(my_amax1);
    float ba2 = block_reduce_max(my_amax2);
    if (threadIdx.x == 0) {
        if (ba1 > 0.0f) atomicMaxFloat(global_amax1, ba1);
        if (ba2 > 0.0f) atomicMaxFloat(global_amax2, ba2);
    }
}

extern "C" void launch_fused_silu_deriv_dual_mul_strided_interleaved_amax(
    const void* dh, const void* h13,
    void* dh13,
    float* global_amax1, float* global_amax2,
    int64_t M, int64_t H, cudaStream_t stream
) {
    constexpr int BLOCK = 256;
    int64_t total_pairs = M * H / 2;
    int64_t grid = (total_pairs + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    fused_silu_deriv_dual_mul_strided_interleaved_amax_kernel<<<(int)grid, BLOCK, 0, stream>>>(
        static_cast<const nv_bfloat16*>(dh),
        static_cast<const nv_bfloat16*>(h13),
        static_cast<nv_bfloat16*>(dh13),
        global_amax1,
        global_amax2,
        M, H
    );
}

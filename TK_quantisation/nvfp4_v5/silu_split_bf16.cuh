#pragma once

namespace tk_silu_split {

constexpr int THREADS = 256;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float warp_max[THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) warp_max[wid] = val;
    __syncthreads();

    float block_val = (wid == 0 && lane < THREADS / 32) ? warp_max[lane] : 0.0f;
    if (wid == 0) {
        block_val = warp_reduce_max(block_val);
    }
    __syncthreads();
    return block_val;
}

__global__ void silu_mul_split_kernel(
    const __nv_bfloat16* __restrict__ h1_raw,
    const __nv_bfloat16* __restrict__ h3,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ global_amax,
    int64_t M,
    int64_t H
) {
    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t vec_numel = total / 4;
    const int2* h1_vec = reinterpret_cast<const int2*>(h1_raw);
    const int2* h3_vec = reinterpret_cast<const int2*>(h3);
    int2* out_vec = reinterpret_cast<int2*>(out);
    const bool track_amax = global_amax != nullptr;
    float local_max = 0.0f;

    for (int64_t i = idx; i < vec_numel; i += stride) {
        const int2 a = h1_vec[i];
        const int2 b = h3_vec[i];

        __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
        __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
        float2 a0_f = __bfloat1622float2(a0);
        float2 a1_f = __bfloat1622float2(a1);
        a0_f.x = silu(a0_f.x);
        a0_f.y = silu(a0_f.y);
        a1_f.x = silu(a1_f.x);
        a1_f.y = silu(a1_f.y);

        __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
        __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);
        const float2 b0_f = __bfloat1622float2(b0);
        const float2 b1_f = __bfloat1622float2(b1);

        const float2 c0_f = {a0_f.x * b0_f.x, a0_f.y * b0_f.y};
        const float2 c1_f = {a1_f.x * b1_f.x, a1_f.y * b1_f.y};
        const __nv_bfloat162 c0 = __float22bfloat162_rn(c0_f);
        const __nv_bfloat162 c1 = __float22bfloat162_rn(c1_f);
        if (track_amax) {
            const float2 c0_bf = __bfloat1622float2(c0);
            const float2 c1_bf = __bfloat1622float2(c1);
            local_max = fmaxf(local_max, fabsf(c0_bf.x));
            local_max = fmaxf(local_max, fabsf(c0_bf.y));
            local_max = fmaxf(local_max, fabsf(c1_bf.x));
            local_max = fmaxf(local_max, fabsf(c1_bf.y));
        }
        int2 packed;
        packed.x = *reinterpret_cast<const int*>(&c0);
        packed.y = *reinterpret_cast<const int*>(&c1);
        out_vec[i] = packed;
    }

    for (int64_t i = vec_numel * 4 + idx; i < total; i += stride) {
        const float prod = silu(__bfloat162float(h1_raw[i])) * __bfloat162float(h3[i]);
        out[i] = __float2bfloat16_rn(prod);
        if (track_amax) {
            local_max = fmaxf(local_max, fabsf(__bfloat162float(out[i])));
        }
    }

    if (track_amax) {
        const float block_max = block_reduce_max(local_max);
        if (threadIdx.x == 0 && block_max > 0.0f) {
            transformer_engine::atomicMaxFloat(global_amax, block_max);
        }
    }
}

__global__ void silu_mul_split_amax_only_kernel(
    const __nv_bfloat16* __restrict__ h1_raw,
    const __nv_bfloat16* __restrict__ h3,
    float* __restrict__ global_amax,
    int64_t M,
    int64_t H
) {
    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t vec_numel = total / 4;
    const int2* h1_vec = reinterpret_cast<const int2*>(h1_raw);
    const int2* h3_vec = reinterpret_cast<const int2*>(h3);
    float local_max = 0.0f;

    for (int64_t i = idx; i < vec_numel; i += stride) {
        const int2 a = h1_vec[i];
        const int2 b = h3_vec[i];

        __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
        __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
        float2 a0_f = __bfloat1622float2(a0);
        float2 a1_f = __bfloat1622float2(a1);
        a0_f.x = silu(a0_f.x);
        a0_f.y = silu(a0_f.y);
        a1_f.x = silu(a1_f.x);
        a1_f.y = silu(a1_f.y);

        __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
        __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);
        const float2 b0_f = __bfloat1622float2(b0);
        const float2 b1_f = __bfloat1622float2(b1);

        local_max = fmaxf(local_max, fabsf(a0_f.x * b0_f.x));
        local_max = fmaxf(local_max, fabsf(a0_f.y * b0_f.y));
        local_max = fmaxf(local_max, fabsf(a1_f.x * b1_f.x));
        local_max = fmaxf(local_max, fabsf(a1_f.y * b1_f.y));
    }

    for (int64_t i = vec_numel * 4 + idx; i < total; i += stride) {
        const float prod = silu(__bfloat162float(h1_raw[i])) * __bfloat162float(h3[i]);
        local_max = fmaxf(local_max, fabsf(prod));
    }

    const float block_max = block_reduce_max(local_max);
    if (threadIdx.x == 0 && block_max > 0.0f) {
        transformer_engine::atomicMaxFloat(global_amax, block_max);
    }
}

__global__ void silu_deriv_dual_split_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    float* __restrict__ global_amax1,
    float* __restrict__ global_amax2,
    int64_t M,
    int64_t H
) {
    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t vec_numel = total / 4;
    const int2* dh_vec = reinterpret_cast<const int2*>(dh);
    const int2* h3_vec = reinterpret_cast<const int2*>(h3);
    const int2* h1_vec = reinterpret_cast<const int2*>(h1_raw);
    int2* out1_vec = reinterpret_cast<int2*>(out1);
    int2* out2_vec = reinterpret_cast<int2*>(out2);
    const bool track_amax = global_amax1 != nullptr && global_amax2 != nullptr;
    float local_max1 = 0.0f;
    float local_max2 = 0.0f;

    for (int64_t i = idx; i < vec_numel; i += stride) {
        const int2 d = dh_vec[i];
        const int2 a = h3_vec[i];
        const int2 b = h1_vec[i];

        __nv_bfloat162 d0 = *reinterpret_cast<const __nv_bfloat162*>(&d.x);
        __nv_bfloat162 d1 = *reinterpret_cast<const __nv_bfloat162*>(&d.y);
        const float2 d0_f = __bfloat1622float2(d0);
        const float2 d1_f = __bfloat1622float2(d1);

        __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
        __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
        const float2 a0_f = __bfloat1622float2(a0);
        const float2 a1_f = __bfloat1622float2(a1);

        __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
        __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);
        const float2 b0_f = __bfloat1622float2(b0);
        const float2 b1_f = __bfloat1622float2(b1);

        const float sig0x = 1.0f / (1.0f + __expf(-b0_f.x));
        const float sig0y = 1.0f / (1.0f + __expf(-b0_f.y));
        const float sig1x = 1.0f / (1.0f + __expf(-b1_f.x));
        const float sig1y = 1.0f / (1.0f + __expf(-b1_f.y));

        const float silu0x = b0_f.x * sig0x;
        const float silu0y = b0_f.y * sig0y;
        const float silu1x = b1_f.x * sig1x;
        const float silu1y = b1_f.y * sig1y;

        const float silup0x = sig0x * (1.0f + b0_f.x - silu0x);
        const float silup0y = sig0y * (1.0f + b0_f.y - silu0y);
        const float silup1x = sig1x * (1.0f + b1_f.x - silu1x);
        const float silup1y = sig1y * (1.0f + b1_f.y - silu1y);

        const float2 c0_f = {d0_f.x * a0_f.x * silup0x, d0_f.y * a0_f.y * silup0y};
        const float2 c1_f = {d1_f.x * a1_f.x * silup1x, d1_f.y * a1_f.y * silup1y};
        const float2 e0_f = {d0_f.x * silu0x, d0_f.y * silu0y};
        const float2 e1_f = {d1_f.x * silu1x, d1_f.y * silu1y};
        const __nv_bfloat162 c0 = __float22bfloat162_rn(c0_f);
        const __nv_bfloat162 c1 = __float22bfloat162_rn(c1_f);
        const __nv_bfloat162 e0 = __float22bfloat162_rn(e0_f);
        const __nv_bfloat162 e1 = __float22bfloat162_rn(e1_f);
        if (track_amax) {
            const float2 c0_bf = __bfloat1622float2(c0);
            const float2 c1_bf = __bfloat1622float2(c1);
            const float2 e0_bf = __bfloat1622float2(e0);
            const float2 e1_bf = __bfloat1622float2(e1);
            local_max1 = fmaxf(local_max1, fabsf(c0_bf.x));
            local_max1 = fmaxf(local_max1, fabsf(c0_bf.y));
            local_max1 = fmaxf(local_max1, fabsf(c1_bf.x));
            local_max1 = fmaxf(local_max1, fabsf(c1_bf.y));
            local_max2 = fmaxf(local_max2, fabsf(e0_bf.x));
            local_max2 = fmaxf(local_max2, fabsf(e0_bf.y));
            local_max2 = fmaxf(local_max2, fabsf(e1_bf.x));
            local_max2 = fmaxf(local_max2, fabsf(e1_bf.y));
        }
        int2 r1, r2;
        r1.x = *reinterpret_cast<const int*>(&c0);
        r1.y = *reinterpret_cast<const int*>(&c1);
        r2.x = *reinterpret_cast<const int*>(&e0);
        r2.y = *reinterpret_cast<const int*>(&e1);
        out1_vec[i] = r1;
        out2_vec[i] = r2;
    }

    for (int64_t i = vec_numel * 4 + idx; i < total; i += stride) {
        const float vd = __bfloat162float(dh[i]);
        const float v1 = __bfloat162float(h1_raw[i]);
        const float v3 = __bfloat162float(h3[i]);
        const float sig = 1.0f / (1.0f + __expf(-v1));
        const float silu_v1 = v1 * sig;
        const float silup_v1 = sig * (1.0f + v1 - silu_v1);
        const float dh1_val = vd * v3 * silup_v1;
        const float dh3_val = vd * silu_v1;
        out1[i] = __float2bfloat16_rn(dh1_val);
        out2[i] = __float2bfloat16_rn(dh3_val);
        if (track_amax) {
            local_max1 = fmaxf(local_max1, fabsf(__bfloat162float(out1[i])));
            local_max2 = fmaxf(local_max2, fabsf(__bfloat162float(out2[i])));
        }
    }

    if (track_amax) {
        const float block_max1 = block_reduce_max(local_max1);
        if (threadIdx.x == 0 && block_max1 > 0.0f) {
            transformer_engine::atomicMaxFloat(global_amax1, block_max1);
        }
        const float block_max2 = block_reduce_max(local_max2);
        if (threadIdx.x == 0 && block_max2 > 0.0f) {
            transformer_engine::atomicMaxFloat(global_amax2, block_max2);
        }
    }
}

__global__ void silu_deriv_dual_split_amax_only_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h3,
    const __nv_bfloat16* __restrict__ h1_raw,
    float* __restrict__ global_amax1,
    float* __restrict__ global_amax2,
    int64_t M,
    int64_t H
) {
    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t vec_numel = total / 4;
    const int2* dh_vec = reinterpret_cast<const int2*>(dh);
    const int2* h3_vec = reinterpret_cast<const int2*>(h3);
    const int2* h1_vec = reinterpret_cast<const int2*>(h1_raw);
    float local_max1 = 0.0f;
    float local_max2 = 0.0f;

    for (int64_t i = idx; i < vec_numel; i += stride) {
        const int2 d = dh_vec[i];
        const int2 a = h3_vec[i];
        const int2 b = h1_vec[i];

        __nv_bfloat162 d0 = *reinterpret_cast<const __nv_bfloat162*>(&d.x);
        __nv_bfloat162 d1 = *reinterpret_cast<const __nv_bfloat162*>(&d.y);
        const float2 d0_f = __bfloat1622float2(d0);
        const float2 d1_f = __bfloat1622float2(d1);

        __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
        __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
        const float2 a0_f = __bfloat1622float2(a0);
        const float2 a1_f = __bfloat1622float2(a1);

        __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
        __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);
        const float2 b0_f = __bfloat1622float2(b0);
        const float2 b1_f = __bfloat1622float2(b1);

        const float sig0x = 1.0f / (1.0f + __expf(-b0_f.x));
        const float sig0y = 1.0f / (1.0f + __expf(-b0_f.y));
        const float sig1x = 1.0f / (1.0f + __expf(-b1_f.x));
        const float sig1y = 1.0f / (1.0f + __expf(-b1_f.y));

        const float silu0x = b0_f.x * sig0x;
        const float silu0y = b0_f.y * sig0y;
        const float silu1x = b1_f.x * sig1x;
        const float silu1y = b1_f.y * sig1y;

        const float silup0x = sig0x * (1.0f + b0_f.x - silu0x);
        const float silup0y = sig0y * (1.0f + b0_f.y - silu0y);
        const float silup1x = sig1x * (1.0f + b1_f.x - silu1x);
        const float silup1y = sig1y * (1.0f + b1_f.y - silu1y);

        local_max1 = fmaxf(local_max1, fabsf(d0_f.x * a0_f.x * silup0x));
        local_max1 = fmaxf(local_max1, fabsf(d0_f.y * a0_f.y * silup0y));
        local_max1 = fmaxf(local_max1, fabsf(d1_f.x * a1_f.x * silup1x));
        local_max1 = fmaxf(local_max1, fabsf(d1_f.y * a1_f.y * silup1y));

        local_max2 = fmaxf(local_max2, fabsf(d0_f.x * silu0x));
        local_max2 = fmaxf(local_max2, fabsf(d0_f.y * silu0y));
        local_max2 = fmaxf(local_max2, fabsf(d1_f.x * silu1x));
        local_max2 = fmaxf(local_max2, fabsf(d1_f.y * silu1y));
    }

    for (int64_t i = vec_numel * 4 + idx; i < total; i += stride) {
        const float vd = __bfloat162float(dh[i]);
        const float v1 = __bfloat162float(h1_raw[i]);
        const float v3 = __bfloat162float(h3[i]);
        const float sig = 1.0f / (1.0f + __expf(-v1));
        const float silu_v1 = v1 * sig;
        const float silup_v1 = sig * (1.0f + v1 - silu_v1);
        local_max1 = fmaxf(local_max1, fabsf(vd * v3 * silup_v1));
        local_max2 = fmaxf(local_max2, fabsf(vd * silu_v1));
    }

    const float block_max1 = block_reduce_max(local_max1);
    if (threadIdx.x == 0 && block_max1 > 0.0f) {
        transformer_engine::atomicMaxFloat(global_amax1, block_max1);
    }
    const float block_max2 = block_reduce_max(local_max2);
    if (threadIdx.x == 0 && block_max2 > 0.0f) {
        transformer_engine::atomicMaxFloat(global_amax2, block_max2);
    }
}

inline int64_t launch_grid(int64_t total_pairs) {
    int64_t grid = (total_pairs + THREADS - 1) / THREADS;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    return grid;
}

inline void launch_forward(
    const __nv_bfloat16* h1_raw,
    const __nv_bfloat16* h3,
    __nv_bfloat16* out,
    int64_t M,
    int64_t H,
    cudaStream_t stream
) {
    silu_mul_split_kernel<<<(int)launch_grid((M * H) / 4), THREADS, 0, stream>>>(
        h1_raw, h3, out, nullptr, M, H);
}

inline void launch_forward_with_amax(
    const __nv_bfloat16* h1_raw,
    const __nv_bfloat16* h3,
    __nv_bfloat16* out,
    float* global_amax,
    int64_t M,
    int64_t H,
    cudaStream_t stream
) {
    silu_mul_split_kernel<<<(int)launch_grid((M * H) / 4), THREADS, 0, stream>>>(
        h1_raw, h3, out, global_amax, M, H);
}

inline void launch_forward_amax_only(
    const __nv_bfloat16* h1_raw,
    const __nv_bfloat16* h3,
    float* global_amax,
    int64_t M,
    int64_t H,
    cudaStream_t stream
) {
    silu_mul_split_amax_only_kernel<<<(int)launch_grid((M * H) / 4), THREADS, 0, stream>>>(
        h1_raw, h3, global_amax, M, H);
}

inline void launch_backward(
    const __nv_bfloat16* dh,
    const __nv_bfloat16* h3,
    const __nv_bfloat16* h1_raw,
    __nv_bfloat16* out1,
    __nv_bfloat16* out2,
    int64_t M,
    int64_t H,
    cudaStream_t stream
) {
    silu_deriv_dual_split_kernel<<<(int)launch_grid((M * H) / 4), THREADS, 0, stream>>>(
        dh, h3, h1_raw, out1, out2, nullptr, nullptr, M, H);
}

inline void launch_backward_with_amax(
    const __nv_bfloat16* dh,
    const __nv_bfloat16* h3,
    const __nv_bfloat16* h1_raw,
    __nv_bfloat16* out1,
    __nv_bfloat16* out2,
    float* global_amax1,
    float* global_amax2,
    int64_t M,
    int64_t H,
    cudaStream_t stream
) {
    silu_deriv_dual_split_kernel<<<(int)launch_grid((M * H) / 4), THREADS, 0, stream>>>(
        dh, h3, h1_raw, out1, out2, global_amax1, global_amax2, M, H);
}

inline void launch_backward_amax_only(
    const __nv_bfloat16* dh,
    const __nv_bfloat16* h3,
    const __nv_bfloat16* h1_raw,
    float* global_amax1,
    float* global_amax2,
    int64_t M,
    int64_t H,
    cudaStream_t stream
) {
    silu_deriv_dual_split_amax_only_kernel<<<(int)launch_grid((M * H) / 4), THREADS, 0, stream>>>(
        dh, h3, h1_raw, global_amax1, global_amax2, M, H);
}

}  // namespace tk_silu_split

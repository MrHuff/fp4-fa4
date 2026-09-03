/*
 * Fused amax computation for bf16 tensors — optimized version.
 *
 * Single CUDA kernel that computes max(abs(input)) for a bf16 tensor.
 * Uses 32-byte vectorized loads (16 bf16 per load) to match TE's throughput.
 * Keeps reduction in bf16 domain until final atomicMax.
 *
 * The caller must zero *amax before launching this kernel
 * (cudaMemsetAsync is cheapest for this).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

// atomicMax for float (positive values only — IEEE 754 sorting property)
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

constexpr int AMAX_THREADS = 512;
// 16 bf16 per vectorized load = 32 bytes (matches TE's nvec=16)
constexpr int AMAX_NVEC = 16;

// Struct for 32-byte aligned loads (2x uint4)
struct alignas(32) uint8x4 {
    uint4 a, b;
};

__launch_bounds__(AMAX_THREADS)
__global__ void fused_amax_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ amax,
    const size_t N  // total number of bf16 elements
) {
    const size_t num_vec = N / AMAX_NVEC;
    const size_t vec_stride = (size_t)gridDim.x * blockDim.x;

    // Keep max in bf16 domain — avoid bf16→float conversion in the hot loop
    __nv_bfloat16 local_max_bf16 = __float2bfloat16(0.0f);

    // 32-byte vectorized loads: 16 bf16 per iteration
    const uint8x4* input_vec = reinterpret_cast<const uint8x4*>(input);

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_vec;
         i += vec_stride)
    {
        uint8x4 data = input_vec[i];

        // Unpack 16 bf16 from 8 uint32s (2 bf16 per uint32)
        __nv_bfloat162 v0 = *reinterpret_cast<__nv_bfloat162*>(&data.a.x);
        __nv_bfloat162 v1 = *reinterpret_cast<__nv_bfloat162*>(&data.a.y);
        __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&data.a.z);
        __nv_bfloat162 v3 = *reinterpret_cast<__nv_bfloat162*>(&data.a.w);
        __nv_bfloat162 v4 = *reinterpret_cast<__nv_bfloat162*>(&data.b.x);
        __nv_bfloat162 v5 = *reinterpret_cast<__nv_bfloat162*>(&data.b.y);
        __nv_bfloat162 v6 = *reinterpret_cast<__nv_bfloat162*>(&data.b.z);
        __nv_bfloat162 v7 = *reinterpret_cast<__nv_bfloat162*>(&data.b.w);

        // Abs + tree reduction on bf16x2 pairs
        v0 = __habs2(v0); v1 = __habs2(v1);
        v2 = __habs2(v2); v3 = __habs2(v3);
        v4 = __habs2(v4); v5 = __habs2(v5);
        v6 = __habs2(v6); v7 = __habs2(v7);

        __nv_bfloat162 m01 = __hmax2(v0, v1);
        __nv_bfloat162 m23 = __hmax2(v2, v3);
        __nv_bfloat162 m45 = __hmax2(v4, v5);
        __nv_bfloat162 m67 = __hmax2(v6, v7);

        __nv_bfloat162 m0123 = __hmax2(m01, m23);
        __nv_bfloat162 m4567 = __hmax2(m45, m67);
        __nv_bfloat162 m_all = __hmax2(m0123, m4567);

        // Reduce bf16x2 to scalar bf16
        __nv_bfloat16 hi = __high2bfloat16(m_all);
        __nv_bfloat16 lo = __low2bfloat16(m_all);
        __nv_bfloat16 iter_max = __hmax(hi, lo);
        local_max_bf16 = __hmax(local_max_bf16, iter_max);
    }

    // Handle tail elements (N not divisible by 16)
    size_t tail_start = num_vec * AMAX_NVEC;
    for (size_t i = tail_start + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += vec_stride)
    {
        local_max_bf16 = __hmax(local_max_bf16, __habs(input[i]));
    }

    // Convert to float for warp/block reduction (only once per thread)
    float local_max = __bfloat162float(local_max_bf16);

    // Warp-level reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
    }

    // Block-level reduction via shared memory
    __shared__ float warp_maxes[AMAX_THREADS / 32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (lane == 0) {
        warp_maxes[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        local_max = (lane < (AMAX_THREADS / 32)) ? warp_maxes[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
        }
        if (lane == 0) {
            atomicMaxFloat(amax, local_max);
        }
    }
}

// (anonymous namespace continues — grouped kernel below)

constexpr int MAX_SPLITS = 8;  // max number of splits supported

__launch_bounds__(AMAX_THREADS)
__global__ void grouped_amax_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ amaxes,
    const int64_t* __restrict__ split_elem_offsets,
    int num_splits,
    size_t total_elems
) {
    __shared__ int64_t s_offsets[MAX_SPLITS + 1];
    if (threadIdx.x < num_splits + 1 && threadIdx.x < MAX_SPLITS + 1) {
        s_offsets[threadIdx.x] = split_elem_offsets[threadIdx.x];
    }
    __syncthreads();

    const size_t num_vec = total_elems / AMAX_NVEC;
    const size_t vec_stride = (size_t)gridDim.x * blockDim.x;

    float local_max[MAX_SPLITS];
    #pragma unroll
    for (int s = 0; s < MAX_SPLITS; ++s) local_max[s] = 0.0f;

    const uint8x4* input_vec = reinterpret_cast<const uint8x4*>(input);

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_vec;
         i += vec_stride)
    {
        size_t elem_idx = i * AMAX_NVEC;
        int split_id = 0;
        #pragma unroll
        for (int s = 1; s < MAX_SPLITS; ++s) {
            if (s < num_splits && elem_idx >= (size_t)s_offsets[s]) {
                split_id = s;
            }
        }

        uint8x4 data = input_vec[i];

        __nv_bfloat162 v0 = __habs2(*reinterpret_cast<__nv_bfloat162*>(&data.a.x));
        __nv_bfloat162 v1 = __habs2(*reinterpret_cast<__nv_bfloat162*>(&data.a.y));
        __nv_bfloat162 v2 = __habs2(*reinterpret_cast<__nv_bfloat162*>(&data.a.z));
        __nv_bfloat162 v3 = __habs2(*reinterpret_cast<__nv_bfloat162*>(&data.a.w));
        __nv_bfloat162 v4 = __habs2(*reinterpret_cast<__nv_bfloat162*>(&data.b.x));
        __nv_bfloat162 v5 = __habs2(*reinterpret_cast<__nv_bfloat162*>(&data.b.y));
        __nv_bfloat162 v6 = __habs2(*reinterpret_cast<__nv_bfloat162*>(&data.b.z));
        __nv_bfloat162 v7 = __habs2(*reinterpret_cast<__nv_bfloat162*>(&data.b.w));

        __nv_bfloat162 m_all = __hmax2(__hmax2(__hmax2(v0, v1), __hmax2(v2, v3)),
                                        __hmax2(__hmax2(v4, v5), __hmax2(v6, v7)));
        float iter_max = __bfloat162float(__hmax(__high2bfloat16(m_all),
                                                 __low2bfloat16(m_all)));
        local_max[split_id] = fmaxf(local_max[split_id], iter_max);
    }

    __shared__ float warp_maxes[MAX_SPLITS][AMAX_THREADS / 32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    for (int s = 0; s < num_splits && s < MAX_SPLITS; ++s) {
        float val = local_max[s];
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
        if (lane == 0) {
            warp_maxes[s][warp_id] = val;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int s = 0; s < num_splits && s < MAX_SPLITS; ++s) {
            float val = (lane < (AMAX_THREADS / 32)) ? warp_maxes[s][lane] : 0.0f;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            }
            if (lane == 0) {
                atomicMaxFloat(&amaxes[s], val);
            }
        }
    }
}
// scalar scale kernels (used by launchers below)
__global__ void scalar_scale_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    float scale) {
    dst[0] = src[0] * scale;
}

__global__ void dual_scalar_scale_kernel(
    const float* __restrict__ src1, float* __restrict__ dst1,
    const float* __restrict__ src2, float* __restrict__ dst2,
    float scale) {
    dst1[0] = src1[0] * scale;
    dst2[0] = src2[0] * scale;
}

}  // anonymous namespace

// Host-side launcher (single tensor)
void launch_fused_amax_bf16(
    const void* input,
    float* amax,
    size_t N,
    cudaStream_t stream
) {
    if (N == 0) return;
    constexpr int threads = AMAX_THREADS;
    size_t num_blocks = (N / AMAX_NVEC + threads - 1) / threads;
    num_blocks = std::min(num_blocks, (size_t)65535);
    fused_amax_bf16_kernel<<<num_blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input), amax, N);
}

// Combined: fused amax + scalar scale in one host call (saves one cudaLaunchKernel)
void launch_fused_amax_bf16_with_scale(
    const void* input,
    float* amax,
    float* sg_out,
    float scale,
    size_t N,
    cudaStream_t stream
) {
    if (N == 0) return;
    constexpr int threads = AMAX_THREADS;
    size_t num_blocks = (N / AMAX_NVEC + threads - 1) / threads;
    num_blocks = std::min(num_blocks, (size_t)65535);
    fused_amax_bf16_kernel<<<num_blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input), amax, N);
    scalar_scale_kernel<<<1, 1, 0, stream>>>(amax, sg_out, scale);
}

// Host-side launcher (grouped: all per-split amaxes in 1 kernel)
void launch_grouped_amax_bf16(
    const void* input,
    float* amaxes,
    const int64_t* split_elem_offsets,
    int num_splits,
    size_t total_elems,
    cudaStream_t stream
) {
    if (total_elems == 0) return;
    constexpr int threads = AMAX_THREADS;
    size_t num_blocks = (total_elems / AMAX_NVEC + threads - 1) / threads;
    num_blocks = std::min(num_blocks, (size_t)65535);
    grouped_amax_bf16_kernel<<<num_blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        amaxes, split_elem_offsets, num_splits, total_elems);
}

void launch_scalar_scale(
    const float* src, float* dst, float scale, cudaStream_t stream
) {
    scalar_scale_kernel<<<1, 1, 0, stream>>>(src, dst, scale);
}

// Compute two scales in a single kernel launch (saves one cudaLaunchKernel)
void launch_dual_scalar_scale(
    const float* src1, float* dst1,
    const float* src2, float* dst2,
    float scale, cudaStream_t stream
) {
    dual_scalar_scale_kernel<<<1, 1, 0, stream>>>(src1, dst1, src2, dst2, scale);
}


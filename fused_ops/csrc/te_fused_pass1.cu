/*
 * Pass 1: Compute inv_rms (per-row) and global_amax (for NVFP4 quantization)
 *
 * OPTIMIZED: Single-loop interleaved computation
 *   - Computes sum_sq (for inv_rms) and pre-norm amax simultaneously
 *   - Uses bf16x8 vectorized loads (128-bit)
 *   - Uses warp shuffle reductions (no cub::BlockReduce overhead)
 *   - Defers inv_rms scaling of amax to post-reduction
 *
 * The pre-norm amax trick: We compute max(|silu(x)*w|) in the same loop
 * as sum_sq, then scale by inv_rms after reduction. This gives us
 * amax ≈ max(|silu(x*IR*w)|) without a second data pass.
 *
 * One block per row. The global_amax_ptr must be zero-initialized before launch.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "vec.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;

constexpr int PASS1_BLOCK_SIZE = 256;
constexpr int BG = 16;  // NVFP4 block group size (matches V7)

// =========================================================================
// Warp-level reductions (no shared memory needed per reduction step)
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

__device__ __forceinline__ float block_reduce_sum_fast(float val) {
    __shared__ float warp_vals[8];  // 256/32 = 8 warps
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    val = warp_reduce_sum(val);
    if (lane == 0) warp_vals[warp] = val;
    __syncthreads();

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

// Positive-float atomicMax (works because positive floats have same ordering as ints)
__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    if (value >= 0) {
        atomicMax(reinterpret_cast<unsigned int*>(addr),
                  __float_as_uint(value));
    }
}

__device__ __forceinline__ float device_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =========================================================================
// Optimized Pass 1 kernel: 2-loop exact amax, vectorized loads, warp shuffles
//
// Loop 1: compute sum_sq → reduce → inv_rms
// Loop 2: compute exact silu(x * inv_rms * w) → reduce → global_amax
//
// This avoids the pre-norm amax approximation which caused parity issues:
//   OLD: silu(x) * w * inv_rms  (APPROXIMATE — SiLU is nonlinear)
//   NEW: silu(x * inv_rms * w)  (EXACT)
// =========================================================================

template<int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_reduction_pass1_kernel(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float* __restrict__ inv_rms_cache,
    float* __restrict__ global_amax_ptr
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int nbpr = cols / BG;  // number of BG-element blocks per row

    const nv_bfloat16* row_x = x_ptr + (int64_t)row * cols;

    // === LOOP 1: Compute sum_sq for inv_rms ===
    float sum_sq = 0.0f;
    for (int qb = tid; qb < nbpr; qb += BLOCK_SIZE) {
        int off = qb * BG;
        bf16x8 d0 = bf16x8::load(row_x + off);
        bf16x8 d1 = bf16x8::load(row_x + off + 8);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d0[k]);
            sum_sq += v * v;
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d1[k]);
            sum_sq += v * v;
        }
    }

    // Reduce sum_sq → inv_rms
    float total_sq = block_reduce_sum_fast(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(total_sq / cols + epsilon);
        inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // === LOOP 2: Compute exact silu(x * inv_rms * w) → amax ===
    // Data should be in L2 cache from Loop 1, so this is fast
    float my_amax = 0.0f;
    for (int qb = tid; qb < nbpr; qb += BLOCK_SIZE) {
        int off = qb * BG;
        bf16x8 d0 = bf16x8::load(row_x + off);
        bf16x8 d1 = bf16x8::load(row_x + off + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + off);
        bf16x8 w1 = bf16x8::load(w_ptr + off + 8);

        float bmax = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d0[k]);
            float wv = __bfloat162float(w0[k]);
            // Exact: silu(bf16(x * inv_rms * w)), rounded to bf16 to match quantizer input
            float normed = __bfloat162float(__float2bfloat16_rn(v * inv_rms * wv));
            float sv = __bfloat162float(__float2bfloat16_rn(device_silu(normed)));
            bmax = fmaxf(bmax, fabsf(sv));
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d1[k]);
            float wv = __bfloat162float(w1[k]);
            float normed = __bfloat162float(__float2bfloat16_rn(v * inv_rms * wv));
            float sv = __bfloat162float(__float2bfloat16_rn(device_silu(normed)));
            bmax = fmaxf(bmax, fabsf(sv));
        }
        my_amax = fmaxf(my_amax, bmax);
    }

    // Reduce amax → global amax
    float row_amax = block_reduce_max_fast(my_amax);

    if (tid == 0 && row_amax > 0.0f) {
        atomicMaxFloat(global_amax_ptr, row_amax);
    }
}

extern "C" void launch_fused_reduction_pass1(
    const void* x_ptr,
    const void* w_ptr,
    float epsilon,
    int rows, int cols,
    float* inv_rms_cache,
    float* global_amax_ptr,
    cudaStream_t stream
) {
    dim3 grid(rows);
    dim3 block(PASS1_BLOCK_SIZE);

    fused_reduction_pass1_kernel<PASS1_BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(x_ptr),
        reinterpret_cast<const nv_bfloat16*>(w_ptr),
        epsilon, rows, cols,
        inv_rms_cache, global_amax_ptr
    );
}

// =========================================================================
// No-SiLU variant: computes amax from |x * inv_rms * w| (exact, no approx)
// Used for attention QKV and w3 path. 2-loop approach matching SiLU variant.
// =========================================================================

template<int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_reduction_pass1_no_silu_kernel(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float* __restrict__ inv_rms_cache,
    float* __restrict__ global_amax_ptr
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int nbpr = cols / BG;

    const nv_bfloat16* row_x = x_ptr + (int64_t)row * cols;

    // === LOOP 1: Compute sum_sq for inv_rms ===
    float sum_sq = 0.0f;
    for (int qb = tid; qb < nbpr; qb += BLOCK_SIZE) {
        int off = qb * BG;
        bf16x8 d0 = bf16x8::load(row_x + off);
        bf16x8 d1 = bf16x8::load(row_x + off + 8);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d0[k]);
            sum_sq += v * v;
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d1[k]);
            sum_sq += v * v;
        }
    }

    // Reduce sum_sq → inv_rms
    float total_sq = block_reduce_sum_fast(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(total_sq / cols + epsilon);
        inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // === LOOP 2: Compute exact |x * inv_rms * w| → amax ===
    float my_amax = 0.0f;
    for (int qb = tid; qb < nbpr; qb += BLOCK_SIZE) {
        int off = qb * BG;
        bf16x8 d0 = bf16x8::load(row_x + off);
        bf16x8 d1 = bf16x8::load(row_x + off + 8);
        bf16x8 w0 = bf16x8::load(w_ptr + off);
        bf16x8 w1 = bf16x8::load(w_ptr + off + 8);

        float bmax = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d0[k]);
            float wv = __bfloat162float(w0[k]);
            // Exact: |x * inv_rms * w|, rounded to bf16 to match quantizer input
            float normed = __bfloat162float(__float2bfloat16_rn(v * inv_rms * wv));
            bmax = fmaxf(bmax, fabsf(normed));
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d1[k]);
            float wv = __bfloat162float(w1[k]);
            float normed = __bfloat162float(__float2bfloat16_rn(v * inv_rms * wv));
            bmax = fmaxf(bmax, fabsf(normed));
        }
        my_amax = fmaxf(my_amax, bmax);
    }

    // Reduce amax → global amax
    float row_amax = block_reduce_max_fast(my_amax);

    if (tid == 0 && row_amax > 0.0f) {
        atomicMaxFloat(global_amax_ptr, row_amax);
    }
}

extern "C" void launch_fused_reduction_pass1_no_silu(
    const void* x_ptr,
    const void* w_ptr,
    float epsilon,
    int rows, int cols,
    float* inv_rms_cache,
    float* global_amax_ptr,
    cudaStream_t stream
) {
    dim3 grid(rows);
    dim3 block(PASS1_BLOCK_SIZE);

    fused_reduction_pass1_no_silu_kernel<PASS1_BLOCK_SIZE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(x_ptr),
        reinterpret_cast<const nv_bfloat16*>(w_ptr),
        epsilon, rows, cols,
        inv_rms_cache, global_amax_ptr
    );
}

// =========================================================================
// Fused RMSNorm + SiLU (bf16 output + amaxes, no quantization)
//
// Two-pass kernel:
//   Pass 1: one block per row, compute sum_sq → inv_rms
//   Pass 2: read x, apply x*inv_rms*gamma → normed (bf16)
//                            silu(normed) → silu_normed (bf16)
//           writes BOTH outputs + reduces amaxes for both in the same pass
//
// Amaxes enable direct nvte_quantize_v2 calls downstream (no separate amax pass).
// =========================================================================

template<int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rmsnorm_silu_dual_kernel(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* __restrict__ out_silu,   // silu(normed) for w1 path
    nv_bfloat16* __restrict__ out_normed, // normed for w3 path
    float* __restrict__ inv_rms_cache,
    float* __restrict__ global_amax_silu, // global amax for silu output (may be nullptr)
    float* __restrict__ global_amax_norm  // global amax for normed output (may be nullptr)
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const int elems_per_thread = 8;  // bf16x8 vectors
    int vec_count = cols / elems_per_thread;

    // Dynamic shared memory: cache x row to avoid reading from GMEM twice
    extern __shared__ nv_bfloat16 smem[];
    // smem layout: [0..cols-1] = x_row cached

    const nv_bfloat16* row_x = x_ptr + (int64_t)row * cols;
    nv_bfloat16* row_silu = out_silu + (int64_t)row * cols;
    nv_bfloat16* row_norm = out_normed + (int64_t)row * cols;

    // === Pass 1: load x → smem + compute sum_sq → inv_rms ===
    float sum_sq = 0.0f;
    for (int i = tid; i < vec_count; i += BLOCK_SIZE) {
        int off = i * elems_per_thread;
        bf16x8 d = bf16x8::load(row_x + off);
        // Cache in shared memory
        d.store(smem + off);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d[k]);
            sum_sq += v * v;
        }
    }

    float total_sq = block_reduce_sum_fast(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(total_sq / cols + epsilon);
        inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // === Pass 2: read x from SMEM (not GMEM!), apply rmsnorm + silu ===
    float local_max_silu = 0.0f;
    float local_max_norm = 0.0f;

    for (int i = tid; i < vec_count; i += BLOCK_SIZE) {
        int off = i * elems_per_thread;
        bf16x8 d = bf16x8::load(smem + off);  // read from shared memory!
        bf16x8 w = bf16x8::load(w_ptr + off);

        bf16x8 normed_out;
        bf16x8 silu_out;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d[k]);
            float wv = __bfloat162float(w[k]);
            float normed_val = v * inv_rms * wv;
            nv_bfloat16 normed_bf16 = __float2bfloat16_rn(normed_val);
            normed_out[k] = normed_bf16;
            float n = __bfloat162float(normed_bf16);
            float silu_val = n / (1.0f + expf(-n));
            silu_out[k] = __float2bfloat16_rn(silu_val);

            local_max_silu = fmaxf(local_max_silu, fabsf(silu_val));
            local_max_norm = fmaxf(local_max_norm, fabsf(n));
        }

        normed_out.store(row_norm + off);
        silu_out.store(row_silu + off);
    }

    // Warp-level max reduction + atomic to global
    if (global_amax_silu != nullptr) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max_silu = fmaxf(local_max_silu, __shfl_xor_sync(0xFFFFFFFF, local_max_silu, offset));
            local_max_norm = fmaxf(local_max_norm, __shfl_xor_sync(0xFFFFFFFF, local_max_norm, offset));
        }
        if ((tid & 31) == 0) {
            atomicMax(reinterpret_cast<int*>(global_amax_silu), __float_as_int(local_max_silu));
            atomicMax(reinterpret_cast<int*>(global_amax_norm), __float_as_int(local_max_norm));
        }
    }
}

extern "C" void launch_fused_rmsnorm_silu_dual(
    const void* x_ptr,
    const void* w_ptr,
    float epsilon,
    int rows, int cols,
    void* out_silu,
    void* out_normed,
    float* inv_rms_cache,
    cudaStream_t stream
) {
    dim3 grid(rows);
    dim3 block(PASS1_BLOCK_SIZE);
    size_t smem_bytes = cols * sizeof(nv_bfloat16);  // cache x row in smem

    fused_rmsnorm_silu_dual_kernel<PASS1_BLOCK_SIZE><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(x_ptr),
        reinterpret_cast<const nv_bfloat16*>(w_ptr),
        epsilon, rows, cols,
        reinterpret_cast<nv_bfloat16*>(out_silu),
        reinterpret_cast<nv_bfloat16*>(out_normed),
        inv_rms_cache,
        nullptr, nullptr
    );
}

// Launcher with amax computation
extern "C" void launch_fused_rmsnorm_silu_dual_with_amax(
    const void* x_ptr,
    const void* w_ptr,
    float epsilon,
    int rows, int cols,
    void* out_silu,
    void* out_normed,
    float* inv_rms_cache,
    float* global_amax_silu,
    float* global_amax_norm,
    cudaStream_t stream
) {
    dim3 grid(rows);
    dim3 block(PASS1_BLOCK_SIZE);
    size_t smem_bytes = cols * sizeof(nv_bfloat16);  // cache x row in smem

    fused_rmsnorm_silu_dual_kernel<PASS1_BLOCK_SIZE><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(x_ptr),
        reinterpret_cast<const nv_bfloat16*>(w_ptr),
        epsilon, rows, cols,
        reinterpret_cast<nv_bfloat16*>(out_silu),
        reinterpret_cast<nv_bfloat16*>(out_normed),
        inv_rms_cache,
        global_amax_silu,
        global_amax_norm
    );
}


// =========================================================================
// Fused RMSNorm ONLY (no SiLU) + optional amax
//
// Kernel for QKV path:
//   normed = x * inv_rms * gamma  (bf16)
//   inv_rms = rsqrt(mean(x^2) + eps)  (float32)
//   amax = max(|normed|)  (float32, optional — computed for free during Pass 2)
//
// Two-pass within one kernel launch:
//   Pass 1: load x → smem, compute sum_sq → reduce → inv_rms
//   Pass 2: read x from smem (L1 hit), apply rmsnorm → normed + optional amax
//
// When global_amax is nullptr: amax reduction is skipped (fused_rmsnorm_only path)
// When global_amax is non-null: amax is computed for free → enables fused quantize
// =========================================================================

template<int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rmsnorm_only_kernel(
    const nv_bfloat16* __restrict__ x_ptr,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* __restrict__ out_normed,
    float* __restrict__ inv_rms_cache,
    float* __restrict__ global_amax  // nullptr → skip amax
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const int elems_per_thread = 8;  // bf16x8 vectors
    int vec_count = cols / elems_per_thread;

    // Dynamic shared memory: cache x row to avoid reading from GMEM twice
    extern __shared__ nv_bfloat16 smem[];

    const nv_bfloat16* row_x = x_ptr + (int64_t)row * cols;
    nv_bfloat16* row_out = out_normed + (int64_t)row * cols;

    // === Pass 1: load x → smem + compute sum_sq → inv_rms ===
    float sum_sq = 0.0f;
    for (int i = tid; i < vec_count; i += BLOCK_SIZE) {
        int off = i * elems_per_thread;
        bf16x8 d = bf16x8::load(row_x + off);
        // Cache in shared memory
        d.store(smem + off);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d[k]);
            sum_sq += v * v;
        }
    }

    float total_sq = block_reduce_sum_fast(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(total_sq / cols + epsilon);
        inv_rms_cache[row] = s_inv_rms;
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // === Pass 2: read x from SMEM, apply rmsnorm + optional amax ===
    float local_max = 0.0f;
    for (int i = tid; i < vec_count; i += BLOCK_SIZE) {
        int off = i * elems_per_thread;
        bf16x8 d = bf16x8::load(smem + off);  // read from shared memory!
        bf16x8 w = bf16x8::load(w_ptr + off);

        bf16x8 normed_out;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __bfloat162float(d[k]);
            float wv = __bfloat162float(w[k]);
            float normed_val = v * inv_rms * wv;
            nv_bfloat16 normed_bf16 = __float2bfloat16_rn(normed_val);
            normed_out[k] = normed_bf16;
            // Amax of the bf16-rounded value (matches what quantizer will see)
            local_max = fmaxf(local_max, fabsf(__bfloat162float(normed_bf16)));
        }
        normed_out.store(row_out + off);
    }

    // Reduce amax → global (only if requested)
    if (global_amax != nullptr) {
        for (int offset = 16; offset > 0; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        if ((tid & 31) == 0)
            atomicMax(reinterpret_cast<int*>(global_amax), __float_as_int(local_max));
    }
}

extern "C" void launch_fused_rmsnorm_only(
    const void* x_ptr,
    const void* w_ptr,
    float epsilon,
    int rows, int cols,
    void* out_normed,
    float* inv_rms_cache,
    cudaStream_t stream
) {
    dim3 grid(rows);
    dim3 block(PASS1_BLOCK_SIZE);
    size_t smem_bytes = cols * sizeof(nv_bfloat16);

    fused_rmsnorm_only_kernel<PASS1_BLOCK_SIZE><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(x_ptr),
        reinterpret_cast<const nv_bfloat16*>(w_ptr),
        epsilon, rows, cols,
        reinterpret_cast<nv_bfloat16*>(out_normed),
        inv_rms_cache,
        nullptr  // no amax
    );
}

extern "C" void launch_fused_rmsnorm_only_with_amax(
    const void* x_ptr,
    const void* w_ptr,
    float epsilon,
    int rows, int cols,
    void* out_normed,
    float* inv_rms_cache,
    float* global_amax,
    cudaStream_t stream
) {
    dim3 grid(rows);
    dim3 block(PASS1_BLOCK_SIZE);
    size_t smem_bytes = cols * sizeof(nv_bfloat16);

    fused_rmsnorm_only_kernel<PASS1_BLOCK_SIZE><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(x_ptr),
        reinterpret_cast<const nv_bfloat16*>(w_ptr),
        epsilon, rows, cols,
        reinterpret_cast<nv_bfloat16*>(out_normed),
        inv_rms_cache,
        global_amax
    );
}


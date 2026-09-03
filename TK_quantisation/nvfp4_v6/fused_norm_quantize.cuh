/*************************************************************************
 * Fused RMSNorm + optional SiLU + Quantize Kernel
 *
 * Eliminates the intermediate bf16 GMEM write between rmsnorm and quantize.
 *
 * Architecture (2 phases, like V5 fused):
 *   Phase 1: TMA load raw x → SMEM, apply transform in registers,
 *            write transformed bf16 back to SMEM, scan amax
 *   Barrier: grid-level global amax
 *   Phase 2: Data is STILL in SMEM (already transformed) — quantize
 *
 * Prerequisites:
 *   - inv_rms[M] must be pre-computed by a separate lightweight kernel
 *   - gamma[K] (RMSNorm weight) passed as kernel param
 *
 * Template params:
 *   WITH_SILU: if true, applies silu(x * inv_rms * gamma)
 *              if false, applies x * inv_rms * gamma
 *   RETURN_TRANSPOSE: if true, produces transposed fp4 + colwise scales
 *
 * Data savings vs old approach:
 *   OLD: read x, write bf16_norm, read bf16_norm, write fp4 = 4 passes
 *   NEW: read x, write fp4 = 2 passes (transform applied in-register)
 *************************************************************************/

#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"

namespace fused_norm_quant {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::ptx;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

// ─── Device helper: silu(x) = x / (1 + exp(-x)) ───
__device__ __forceinline__
float device_silu(float x) {
    return x / (1.0f + expf(-x));
}

// ─── Scan tile amax WITH inline transform ───
// Reads raw x from SMEM, applies rmsnorm ± silu, writes transformed back,
// and returns the tile's max absolute value of the transformed data.
template <bool WITH_SILU>
__device__ __forceinline__
float scan_and_transform_tile(
    IType* sIn_ptr,
    const IType* gamma_smem,    // gamma loaded to SMEM, length V3_TILE_DIM_X
    const float* inv_rms_row,   // pointer to inv_rms for each row in this tile
    int tile_idx,
    int chunk_row_start,        // global starting row of this chunk
    int tile_row_offset,        // row offset within chunk for this tile
    int rows                    // total rows in matrix
) {
    auto& sIn = *reinterpret_cast<V3_IType3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int tid_X = threadIdx.x % (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    IType2 tile_max_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    constexpr int THREADS_X = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
    constexpr int THREADS_Y = V3_THREADS / THREADS_X;
    constexpr int ITERS = V3_TILE_DIM_Y / THREADS_Y;

    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int local_row = tid_Y + it * THREADS_Y;
        const int global_row = chunk_row_start + tile_row_offset + local_row;

        float row_inv_rms = 0.0f;
        if (global_row < rows) {
            row_inv_rms = inv_rms_row[local_row + tile_row_offset];
        }

        #pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; e += 2) {
            const int col_in_tile = thread_offset_X + e;
            float x0 = __bfloat162float(sIn[tile_idx][local_row][col_in_tile + 0]);
            float x1 = __bfloat162float(sIn[tile_idx][local_row][col_in_tile + 1]);
            float g0 = __bfloat162float(gamma_smem[col_in_tile + 0]);
            float g1 = __bfloat162float(gamma_smem[col_in_tile + 1]);

            float transformed0 = x0 * row_inv_rms * g0;
            float transformed1 = x1 * row_inv_rms * g1;
            if constexpr (WITH_SILU) {
                transformed0 = device_silu(transformed0);
                transformed1 = device_silu(transformed1);
            }

            const IType2 transformed_pair = {
                __float2bfloat16_rn(transformed0),
                __float2bfloat16_rn(transformed1)
            };
            *reinterpret_cast<IType2*>(&sIn[tile_idx][local_row][col_in_tile]) = transformed_pair;
            ptx::abs_max_2x(tile_max_2x, tile_max_2x, transformed_pair);
        }
    }

    return get_amax_of_pair(tile_max_2x);
}


// ═══════════════════════════════════════════════════════════════════
// Main kernel: fused rmsnorm + optional silu + quantize
// ═══════════════════════════════════════════════════════════════════

template <bool WITH_SILU, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
fused_norm_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    nvfp4_scale_t* const scales_ptr,
    nvfp4_scale_t* const scales_t_ptr,
    const float* __restrict__ inv_rms,        // pre-computed (M,)
    const IType* __restrict__ gamma,          // rmsnorm weight (K,)
    float* __restrict__ global_amax,
    float* __restrict__ sg_out,
    unsigned int* __restrict__ done_counter,
    unsigned int* __restrict__ ready_flag,
    const size_t rows, const size_t cols,
    const size_t scale_stride, const size_t scale_stride_t,
    const int total_blocks
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // ─── SMEM layout: same as V5 fused + gamma cache ───
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;
    // Extra: gamma cache for tile-width gamma values
    constexpr int gamma_bytes = DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*         sIn_ptr        = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*     sOut_ptr       = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);
    IType*         gamma_cache    = reinterpret_cast<IType*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes);

    auto& sIn        = *reinterpret_cast<V3_IType3D*>(sIn_ptr);
    auto& sSFrowwise = *reinterpret_cast<V3_ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<V3_ScalesTypeTr2D*>(sSFcolwise_ptr);

    // ─── This CTA's chunk ───
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
    const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

    // Load gamma slice into SMEM (128 elements starting at block_offset_X)
    for (int i = threadIdx.x; i < V3Config::CHUNK_DIM_X; i += V3_THREADS) {
        int gc = block_offset_X + i;
        gamma_cache[i] = (gc < (int)cols) ? gamma[gc] : static_cast<IType>(0.0f);
    }
    __syncthreads();

    // ─── TMA barriers ───
    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_init(&in_mbar[t], 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // ═════════════════════════════════════════════════════
    // PHASE 1: Load raw x → SMEM, apply transform, scan amax
    // ═════════════════════════════════════════════════════
    float cta_max = 0.0f;

    // Launch TMA for all 4 tiles
    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int ty = t / V3_TILES_X;
        const int tx = t % V3_TILES_X;
        const int gy = block_offset_Y + ty * V3_TILE_DIM_Y;
        const int gx = block_offset_X + tx * V3_TILE_DIM_X;
        if (leading) {
            ptx::mbarrier_arrive_expect_tx(&in_mbar[t], shmem_tile_bytes);
            ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[t]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                gx, gy, &in_mbar[t]);
        }
    }

    // Wait for each tile, transform in-place, scan amax
    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], 0);

        const int ty = t / V3_TILES_X;
        const int tx = t % V3_TILES_X;
        const int tile_col_offset = tx * V3_TILE_DIM_X;
        const int tile_row_offset = ty * V3_TILE_DIM_Y;

        // Get gamma for this tile's columns (gamma_cache is chunk-wide, tile is 64 cols)
        IType* tile_gamma = gamma_cache + tile_col_offset;

        float tile_max = scan_and_transform_tile<WITH_SILU>(
            sIn_ptr, tile_gamma, inv_rms + block_offset_Y,
            t, block_offset_Y, tile_row_offset, (int)rows);
        cta_max = fmaxf(cta_max, tile_max);
    }

    // Block-level reduction of cta_max
    {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));

        __shared__ float warp_max[V3_THREADS / 32];
        int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) warp_max[wid] = cta_max;
        __syncthreads();

        if (wid == 0) {
            cta_max = (lane < V3_THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (V3_THREADS / 32) / 2; mask > 0; mask >>= 1)
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
        }
    }

    // ═════════════════════════════════════════════════════
    // BARRIER: atomicMax + spin-wait
    // ═════════════════════════════════════════════════════
    grid_barrier(cta_max, global_amax, done_counter, ready_flag, total_blocks);

    // ═════════════════════════════════════════════════════
    // PHASE 2: Quantize — data is STILL IN SMEM (already transformed!)
    // This is IDENTICAL to V5 fused Phase 2
    // ═════════════════════════════════════════════════════
    const float amax_val = *global_amax;
    const float S_enc = compute_global_encode_scaling_factor_FP4(amax_val);

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        *sg_out = amax_val / 2688.0f;
        *ready_flag = 0;
    }

    const int block_offset_Y_tr = ctaid_X * V3Config::CHUNK_DIM_X;
    const int block_offset_X_tr = ctaid_Y * V3Config::CHUNK_DIM_Y;
    const int chunk_rows = rows - block_offset_Y;
    const int chunk_cols = cols - block_offset_X;

    const int sc_block_Y_row = ctaid_Y * V3Config::CHUNK_DIM_Y;
    const int sc_block_X_row = ctaid_X * V3_SCALES_PER_CHUNK_X;
    const int sc_block_Y_tr  = ctaid_X * V3Config::CHUNK_DIM_X;
    const int sc_block_X_tr  = ctaid_Y * V3_SCALES_PER_CHUNK_Y;

    auto& sOut    = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    int buff_out = 0;
    int buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        if (t > 0) {
            ptx::cp_async_bulk_wait_group_read<1>();
        }

        // Quantize: data for tile t is in sIn[t] — already transformed in Phase 1!
        v3_rowwise_scaling(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                           S_enc, stage_Y, stage_X, t, buff_out);

        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                               S_enc, stage_Y, stage_X, t, buff_out_tr);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store
        if (leading) {
            const int gy = block_offset_Y + stage_offset_Y;
            const int gx = block_offset_X + stage_offset_X;

            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                gx, gy, reinterpret_cast<uint64_t*>(&sOut[buff_out]));

            if constexpr (RETURN_TRANSPOSE) {
                const int gy_tr = block_offset_Y_tr + stage_offset_X;
                const int gx_tr = block_offset_X_tr + stage_offset_Y;
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    gx_tr, gy_tr, reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
        buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
    }

    // Wait for all TMA stores
    ptx::cp_async_bulk_wait_group_read<0>();

    // ─── Store scales to global (TK swizzle format) ───
    {
        const int ntk = static_cast<int>(scale_stride) / 4;
        for (size_t row = threadIdx.x; row < V3Config::CHUNK_DIM_Y; row += V3_THREADS) {
            const size_t rg = sc_block_Y_row + row;
            if (rg < rows) {
                const int tm = rg / 128, rit = rg % 128;
                const int j = rit % 32, grp = rit / 32;
                const int cnt = min((int)V3_SCALES_PER_CHUNK_X, (int)(chunk_cols / V3_SCALE_DIM));
                for (int kg = 0; kg < cnt / 4; ++kg) {
                    const int kb = kg * 4, kgb = sc_block_X_row + kb;
                    const int ts = (tm * ntk + kgb / 4) * 512 + j * 16 + grp * 4;
                    uint32_t pk; uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
                    auto& SF = sSFrowwise;
                    for (int kk = 0; kk < 4; ++kk)
                        p[kk] = reinterpret_cast<const uint8_t&>(SF[row][kb + kk]);
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(scales_ptr) + ts) = pk;
                }
            }
        }
    }

    if constexpr (RETURN_TRANSPOSE) {
        const int ntk_t = static_cast<int>(scale_stride_t) / 4;
        for (size_t col = threadIdx.x; col < V3Config::CHUNK_DIM_X; col += V3_THREADS) {
            const size_t cg = sc_block_Y_tr + col;
            if (cg < cols) {
                const int tm = cg / 128, rit = cg % 128;
                const int j = rit % 32, grp = rit / 32;
                const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, (int)(chunk_rows / V3_SCALE_DIM));
                for (int kg = 0; kg < cnt / 4; ++kg) {
                    const int kb = kg * 4, kgb = sc_block_X_tr + kb;
                    const int ts = (tm * ntk_t + kgb / 4) * 512 + j * 16 + grp * 4;
                    uint32_t pk; uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
                    auto& SF = sSFcolwise;
                    for (int kk = 0; kk < 4; ++kk)
                        p[kk] = reinterpret_cast<const uint8_t&>(SF[col][kb + kk]);
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(scales_t_ptr) + ts) = pk;
                }
            }
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}




// ─── SMEM size for fused kernel ───
static inline int fused_norm_quant_smem_size(bool return_transpose) {
    int in_bytes  = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    int out_tr_bytes = return_transpose ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    int sc_col_bytes = return_transpose ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;
    int gamma_bytes = DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    return in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes + gamma_bytes + TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace fused_norm_quant

// ─── Standalone inv_rms kernel (no FP4 dependency) ───
// Must be run before fused_norm_quantize_kernel.
// Uses explicit __nv_bfloat16 to avoid any namespace ambiguity.
namespace inv_rms_kernel_ns {

template <int BLOCK_SIZE = 256>
__global__ void compute_inv_rms_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ inv_rms_out,
    float epsilon,
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* row_x = x + (int64_t)row * cols;
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float v = __bfloat162float(row_x[i]);
        sum_sq += v * v;
    }

    // Warp reduce
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);

    // Block reduce via SMEM
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();

    if (wid == 0) {
        sum_sq = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int mask = (BLOCK_SIZE / 32) / 2; mask > 0; mask >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    }

    if (threadIdx.x == 0) {
        inv_rms_out[row] = rsqrtf(sum_sq / cols + epsilon);
    }
}

}  // namespace inv_rms_kernel_ns

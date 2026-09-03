// persistent_norm_quantize.cuh — Persistent fused RMSNorm + Quantize
//
// Applies rmsnorm (x * inv_rms * gamma) inline during both phases:
//   Phase 1: TMA load raw x → SMEM, transform in-place, scan amax
//   Phase 2: TMA load raw x AGAIN → SMEM, transform in-place, quantize
//
// No intermediate bf16 buffer needed. Works for all shapes.
// inv_rms must be pre-computed by compute_inv_rms_kernel.
//
#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"
#include "persistent_quantize.cuh"

namespace tk_v5 {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;

// ─── scan_and_transform_tile: load tile from SMEM, apply rmsnorm,
//     write transformed bf16 back to same SMEM location, return tile max ───
template <bool WITH_SILU, bool USE_ILP_AMAX>
__device__ __forceinline__
float scan_and_norm_tile(
    IType* sIn_ptr,
    const IType* gamma_smem,      // gamma[V3_TILE_DIM_X] for this tile's columns
    const float* inv_rms,         // global inv_rms array
    int tile_idx,
    int block_offset_Y,           // global row start of this chunk
    int tile_row_offset           // row offset within chunk for this tile
) {
    auto& sIn = *reinterpret_cast<V3_IType3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int tid_X = threadIdx.x % (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    IType2 tile_max_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    float tile_max = 0.0f;

    constexpr int THREADS_X = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
    constexpr int THREADS_Y = V3_THREADS / THREADS_X;
    constexpr int ITERS = V3_TILE_DIM_Y / THREADS_Y;

    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int local_row = tid_Y + it * THREADS_Y;
        const int global_row = block_offset_Y + tile_row_offset + local_row;
        const float row_inv_rms = inv_rms[global_row];

        #pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; e += 2) {
            const int col = thread_offset_X + e;
            float x0 = __bfloat162float(sIn[tile_idx][local_row][col + 0]);
            float x1 = __bfloat162float(sIn[tile_idx][local_row][col + 1]);
            float g0 = __bfloat162float(gamma_smem[col + 0]);
            float g1 = __bfloat162float(gamma_smem[col + 1]);

            float val0 = x0 * row_inv_rms * g0;
            float val1 = x1 * row_inv_rms * g1;
            if constexpr (WITH_SILU) {
                val0 = val0 / (1.0f + expf(-val0));
                val1 = val1 / (1.0f + expf(-val1));
            }

            const IType2 transformed_pair = {
                __float2bfloat16_rn(val0),
                __float2bfloat16_rn(val1)
            };
            *reinterpret_cast<IType2*>(&sIn[tile_idx][local_row][col]) = transformed_pair;
            if constexpr (USE_ILP_AMAX) {
                ptx::abs_max_2x_int(tile_max_2x, tile_max_2x, transformed_pair);
            } else {
                tile_max = fmaxf(tile_max, fabsf(__bfloat162float(transformed_pair.x)));
                tile_max = fmaxf(tile_max, fabsf(__bfloat162float(transformed_pair.y)));
            }
        }
    }
    if constexpr (USE_ILP_AMAX) {
        return get_amax_of_pair(tile_max_2x);
    } else {
        return tile_max;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Persistent fused norm+quantize kernel
// ═══════════════════════════════════════════════════════════════════
template <bool WITH_SILU, bool RETURN_TRANSPOSE, bool USE_ILP_AMAX = true>
__global__ void __launch_bounds__(V3_THREADS)
persistent_norm_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    nvfp4_scale_t* const scales_ptr,
    const float* __restrict__ inv_rms,
    const IType* __restrict__ gamma,   // (K,) rmsnorm weight
    const size_t rows, const size_t cols,
    const size_t scale_stride,
    PersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // SMEM layout: same as v5 persistent + gamma cache
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;
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
    auto& sOut       = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr    = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_init(&in_mbar[t], 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float block_max = 0.0f;
    int mbar_phase = 0;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Load raw x, apply rmsnorm, scan amax (work-stealing)
    // ═══════════════════════════════════════════════════════════════
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter_phase1, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        // Load gamma slice for this chunk's columns
        for (int i = threadIdx.x; i < (int)V3Config::CHUNK_DIM_X; i += V3_THREADS) {
            int gc = block_offset_X + i;
            gamma_cache[i] = (gc < (int)cols) ? gamma[gc] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Prefetch first 2 tiles
        #pragma unroll
        for (int pre = 0; pre < min(2, (int)V3_NUM_TILES); ++pre) {
            const int ty = pre / V3_TILES_X, tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &in_mbar[pre]);
            }
        }

        // Process each tile: wait, transform+scan, prefetch next
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            if (t + 2 < V3_NUM_TILES) {
                const int next = t + 2;
                const int ty = next / V3_TILES_X, tx = next % V3_TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[next]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        block_offset_X + tx * V3_TILE_DIM_X,
                        block_offset_Y + ty * V3_TILE_DIM_Y,
                        &in_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

            const int ty = t / V3_TILES_X;
            const int tx = t % V3_TILES_X;
            IType* tile_gamma = gamma_cache + tx * V3_TILE_DIM_X;

            // Apply rmsnorm inline and scan amax
            block_max = fmaxf(block_max,
                scan_and_norm_tile<WITH_SILU, USE_ILP_AMAX>(
                    sIn_ptr, tile_gamma, inv_rms,
                    t, block_offset_Y, ty * V3_TILE_DIM_Y));
        }
        mbar_phase ^= 1;
    }

    // Block reduction
    {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, mask));

        __shared__ float warp_max[V3_THREADS / 32];
        int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) warp_max[wid] = block_max;
        __syncthreads();
        if (wid == 0) {
            block_max = (lane < V3_THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (V3_THREADS / 32) / 2; mask > 0; mask >>= 1)
                block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, mask));
        }
    }

    // BARRIER
    grid_barrier(block_max, args.global_amax,
                 args.done_counter, args.ready_flag,
                 args.num_persistent);

    const float amax_val = args.global_amax[0];
    const float S_enc = compute_global_encode_scaling_factor_FP4(amax_val);

    if (leading && blockIdx.x == 0) {
        if (args.sg_output) args.sg_output[0] = amax_val / 2688.0f;
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Re-load, re-transform, quantize (work-stealing)
    // Re-computing rmsnorm is cheap (2 muls/element) vs saving GMEM
    // ═══════════════════════════════════════════════════════════════
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id2;
        if (leading) s_chunk_id2 = atomicAdd(args.work_counter_phase2, 1);
        __syncthreads();
        if (s_chunk_id2 >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id2 % args.tiles_X;
        const int ctaid_Y = s_chunk_id2 / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        // Reload gamma for this chunk's columns
        for (int i = threadIdx.x; i < (int)V3Config::CHUNK_DIM_X; i += V3_THREADS) {
            int gc = block_offset_X + i;
            gamma_cache[i] = (gc < (int)cols) ? gamma[gc] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Prefetch first 2 tiles
        #pragma unroll
        for (int pre = 0; pre < min(2, (int)V3_NUM_TILES); ++pre) {
            const int ty = pre / V3_TILES_X, tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &in_mbar[pre]);
            }
        }

        // For each tile: wait, re-transform in SMEM, quantize, store
        const bool leading2 = (threadIdx.x == 0);
        const int chunk_rows = (int)rows - block_offset_Y;
        const int chunk_cols = (int)cols - block_offset_X;
        int buff_out = 0, buff_out_tr = 0;

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            const int stage_Y = t / V3_TILES_X;
            const int stage_X = t % V3_TILES_X;
            const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
            const int stage_offset_X = stage_X * V3_TILE_DIM_X;

            // Prefetch tile t+2
            if (t + 2 < V3_NUM_TILES) {
                const int next = t + 2;
                const int nty = next / V3_TILES_X, ntx = next % V3_TILES_X;
                if (leading2) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[next]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        block_offset_X + ntx * V3_TILE_DIM_X,
                        block_offset_Y + nty * V3_TILE_DIM_Y,
                        &in_mbar[next]);
                }
            }

            // Wait for current tile
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

            // Re-apply rmsnorm transform in SMEM (cheap: 2 muls/element)
            IType* tile_gamma = gamma_cache + stage_X * V3_TILE_DIM_X;
            scan_and_norm_tile<WITH_SILU, USE_ILP_AMAX>(
                sIn_ptr, tile_gamma, inv_rms,
                t, block_offset_Y, stage_Y * V3_TILE_DIM_Y);
            __syncthreads();

            // Quantize from SMEM (data is now transformed)
            v3_rowwise_scaling(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                               S_enc, stage_Y, stage_X, t, buff_out);

            if constexpr (RETURN_TRANSPOSE) {
                v3_colwise_scaling(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                   S_enc, stage_Y, stage_X, t, buff_out_tr);
            }

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            // TMA store quantized tile
            if (leading2) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output),
                    block_offset_X + stage_offset_X,
                    block_offset_Y + stage_offset_Y,
                    reinterpret_cast<uint64_t*>(&sOut[buff_out]));

                if constexpr (RETURN_TRANSPOSE) {
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                        block_offset_Y + stage_offset_Y,  // transpose: row→col
                        block_offset_X + stage_offset_X,   // transpose: col→row
                        reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                }
                ptx::cp_async_bulk_commit_group();
            }

            buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
            buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
        }

        // Wait for all FP4 TMA stores
        if (leading2) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        // TMA scale stores (same as v5 persistent)
        {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
            swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading2) {
                const int tm = block_offset_Y / 128;
                const int tma_x_base = ctaid_X * 2 * 256;
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_scale_row),
                    tma_x_base, tm,
                    reinterpret_cast<uint64_t*>(sSFrowwise_ptr));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_scale_row),
                    tma_x_base + 256, tm,
                    reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFrowwise_ptr) + 512));
                ptx::cp_async_bulk_commit_group();
            }
        }

        if constexpr (RETURN_TRANSPOSE) {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
            swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading2) {
                const int tm_col = block_offset_X / 128;
                const int tma_x_base = ctaid_Y * 2 * 256;
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_scale_col),
                    tma_x_base, tm_col,
                    reinterpret_cast<uint64_t*>(sSFcolwise_ptr));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_scale_col),
                    tma_x_base + 256, tm_col,
                    reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFcolwise_ptr) + 512));
                ptx::cp_async_bulk_commit_group();
            }
        }

        if (leading2) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        mbar_phase ^= 1;
    }

    // Cleanup
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_invalid(&in_mbar[t]);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

// ─── SMEM size for persistent norm+quantize kernel ───
static inline int persistent_norm_quant_smem_size(bool return_transpose) {
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

} // namespace tk_v5

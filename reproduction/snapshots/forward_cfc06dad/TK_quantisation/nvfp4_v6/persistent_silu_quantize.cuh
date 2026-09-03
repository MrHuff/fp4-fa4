// persistent_silu_quantize.cuh — Persistent fused silu(h1)*h3 + quantize
//
// Two-phase work-stealing, same as persistent_quantize.cuh but:
//   Phase 1: Load h1+h3 via dual strided TMA, compute silu(h1)*h3 in-register,
//            scan amax. Data NOT kept in SMEM between phases.
//   Phase 2: Re-load h1+h3, recompute silu(h1)*h3 (cheap arithmetic vs GMEM
//            round-trip), quantize with known global amax.
//
// Inputs:  h13 (M, 2H) bf16 — h1 = columns [0,H), h3 = columns [H,2H)
// Outputs: fp4 (M, H) + TK-swizzled scales (+ optional transpose)
//
// SMEM budget: same as persistent_quantize (uses single sIn buffer for
//   transformed data). Phase 1 needs h1+h3 loaded into same buffer
//   sequentially per tile (load h1 tile → load h3 tile → silu+amax → next).
//   Actually we load BOTH h1 and h3 into separate halves of the 4-tile
//   input buffer, then process them pairwise.
//
// Actually, for the persistent approach with work-stealing, we only need
// the standard v3 sIn buffer (4 tiles). The trick:
//   - In phase 1, for each chunk: load h1 tiles into sIn[0..3], process them
//     to get amax of silu(h1)*h3. But we need h3 too!
//   - Solution: Use 8 mbarriers (4 for h1, 4 for h3). Load h1 and h3 tiles
//     interleaved into the SAME sIn buffer using double-buffering across tiles.
//
// Simplest approach: 2 separate 4-tile buffers (h1 and h3), ~64KB total.
// But that doubles SMEM → fewer CTAs/SM → defeats persistent purpose.
//
// Better approach: Sequential loading within a chunk:
//   For each chunk, process tiles one at a time:
//     1. Load h1[t] into sIn[0], load h3[t] into sIn[1]
//     2. Wait both, compute silu(h1[0])*h3[1] in sIn[0], scan amax
//     3. Repeat for next tile pair using sIn[2]/sIn[3]
//   This uses the existing 4-tile buffer (32KB) with no SMEM increase!
//
// Even simpler: just use 2 sIn buffers (h1 and h3), each 4 tiles.
// SMEM = 2×32KB + output + scales ≈ 72KB. On GB200 (228KB): 3 CTAs/SM.
// With 132 SMs: 396 persistent CTAs. For M=65536, H=5632: 22528 tiles,
// each CTA processes ~57 chunks. This is fine for persistent.

#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"
#include "persistent_quantize.cuh"    // for PersistentArgs, swizzle helpers
#include "fused_silu_quantize.cuh"    // for scan_and_silu_tile, device_silu

namespace persistent_silu_quant {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::ptx;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

// ─── Device helper: apply silu(h1)*h3 transform to a tile already in sIn ───
// h1 is in sIn_h1[tile_idx], h3 is in sIn_h3[tile_idx].
// Result is written to sIn_h1[tile_idx] (overwrites h1 data).
// Returns tile amax.
__device__ __forceinline__
float silu_transform_tile(
    IType* sIn_h1_ptr,
    const IType* sIn_h3_ptr,
    int tile_idx
) {
    // Reuse scan_and_silu_tile from fused_silu_quantize.cuh
    return fused_silu_quant::scan_and_silu_tile(sIn_h1_ptr, sIn_h3_ptr, tile_idx);
}

// ─── Phase 2 helper: load h1+h3, apply silu, quantize, store ───
// Like quantize_and_store_chunk_pipelined_v5 but with dual TMA loads
// and silu transform before quantize.
template <bool RETURN_TRANSPOSE>
__device__ __forceinline__ void silu_quantize_and_store_chunk_pipelined(
    IType* sIn_h1_ptr, IType* sIn_h3_ptr,
    fp4e2m1x2* sOut_ptr, fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr, nvfp4_scale_t* sSFcolwise_ptr,
    V3_OType2x3D& sOut, V3_OType2xt3D& sOut_tr,
    const CUtensorMap& tensor_map_output,
    const CUtensorMap& tensor_map_output_t,
    const CUtensorMap& tmap_scale_row,
    const CUtensorMap& tmap_scale_col,
    float S_enc,
    int block_offset_Y, int block_offset_X,
    int rows, int cols,
    int ctaid_X, int ctaid_Y,
    uint64_t* h1_mbar, uint64_t* h3_mbar, int mbar_phase,
    const CUtensorMap* tmap_h1_ptr,
    const CUtensorMap* tmap_h3_ptr
) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_rows = (int)rows - block_offset_Y;
    const int chunk_cols = (int)cols - block_offset_X;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    auto& sIn_h1 = *reinterpret_cast<V3_IType3D*>(sIn_h1_ptr);
    auto& sIn_h3 = *reinterpret_cast<V3_IType3D*>(sIn_h3_ptr);

    int buff_out = 0, buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        // Prefetch tile t+2 if it exists
        if (t + 2 < V3_NUM_TILES) {
            const int next = t + 2;
            const int nty = next / V3_TILES_X, ntx = next % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h1[next]),
                    reinterpret_cast<const uint64_t*>(tmap_h1_ptr),
                    block_offset_X + ntx * V3_TILE_DIM_X,
                    block_offset_Y + nty * V3_TILE_DIM_Y,
                    &h1_mbar[next]);
                ptx::mbarrier_arrive_expect_tx(&h3_mbar[next], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h3[next]),
                    reinterpret_cast<const uint64_t*>(tmap_h3_ptr),
                    block_offset_X + ntx * V3_TILE_DIM_X,
                    block_offset_Y + nty * V3_TILE_DIM_Y,
                    &h3_mbar[next]);
            }
        }

        // Wait for current h1 and h3 tiles
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], mbar_phase);

        // Apply silu(h1)*h3 in-register, result to sIn_h1[t]
        silu_transform_tile(sIn_h1_ptr, sIn_h3_ptr, t);
        __syncthreads();

        // Quantize from sIn_h1 (now contains transformed data)
        v3_rowwise_scaling(sIn_h1_ptr, sOut_ptr, sSFrowwise_ptr,
                           S_enc, stage_Y, stage_X, t, buff_out);

        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling(sIn_h1_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                               S_enc, stage_Y, stage_X, t, buff_out_tr);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store quantized tile
        if (leading) {
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));

            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
        buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
    }

    // Wait for all FP4 TMA stores
    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    // ─── TMA scale stores ───
    {
        const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
        tk_v5::swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
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
        tk_v5::swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
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

    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}


// ═══════════════════════════════════════════════════════════════════
// Persistent fused silu+quantize kernel
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
persistent_silu_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_h1,      // h13[:,0:H] stride=2H
    const __grid_constant__ CUtensorMap tensor_map_h3,      // h13[:,H:2H] stride=2H
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const size_t rows, const size_t cols,    // cols = H (output width)
    const size_t scale_stride,
    tk_v5::PersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // SMEM layout: [h1: 4 tiles] [h3: 4 tiles] [output] [output_tr] [scales_row] [scales_col]
    constexpr int in_h1_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int in_h3_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*         sIn_h1_ptr     = reinterpret_cast<IType*>(dshmem);
    IType*         sIn_h3_ptr     = reinterpret_cast<IType*>(dshmem + in_h1_bytes);
    fp4e2m1x2*     sOut_ptr       = reinterpret_cast<fp4e2m1x2*>(dshmem + in_h1_bytes + in_h3_bytes);
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + in_h1_bytes + in_h3_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_h1_bytes + in_h3_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_h1_bytes + in_h3_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn_h1    = *reinterpret_cast<V3_IType3D*>(sIn_h1_ptr);
    auto& sIn_h3    = *reinterpret_cast<V3_IType3D*>(sIn_h3_ptr);
    auto& sOut      = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr   = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    // Dual mbarriers: 4 for h1, 4 for h3
    __shared__ uint64_t h1_mbar[V3_NUM_TILES];
    __shared__ uint64_t h3_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_init(&h1_mbar[t], 1);
            ptx::mbarrier_init(&h3_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float block_max = 0.0f;
    int mbar_phase = 0;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Scan amax with SiLU transform (work-stealing)
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

        // Prefetch first 2 tile pairs
        #pragma unroll
        for (int pre = 0; pre < min(2, (int)V3_NUM_TILES); ++pre) {
            const int ty = pre / V3_TILES_X, tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h1[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &h1_mbar[pre]);
                ptx::mbarrier_arrive_expect_tx(&h3_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h3[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &h3_mbar[pre]);
            }
        }

        // Process each tile: wait, apply silu+scan, prefetch next
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            // Prefetch tile t+2 if it exists
            if (t + 2 < V3_NUM_TILES) {
                const int next = t + 2;
                const int ty = next / V3_TILES_X, tx = next % V3_TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_h1[next]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                        block_offset_X + (next % V3_TILES_X) * V3_TILE_DIM_X,
                        block_offset_Y + (next / V3_TILES_X) * V3_TILE_DIM_Y,
                        &h1_mbar[next]);
                    ptx::mbarrier_arrive_expect_tx(&h3_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_h3[next]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                        block_offset_X + (next % V3_TILES_X) * V3_TILE_DIM_X,
                        block_offset_Y + (next / V3_TILES_X) * V3_TILE_DIM_Y,
                        &h3_mbar[next]);
                }
            }

            // Wait for current tiles
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], mbar_phase);

            // Apply silu(h1)*h3 in-register, scan amax
            block_max = fmaxf(block_max, silu_transform_tile(sIn_h1_ptr, sIn_h3_ptr, t));
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
    // PHASE 2: Re-load h1+h3, recompute silu, quantize (work-stealing)
    // ═══════════════════════════════════════════════════════════════
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&h1_mbar[t]);
            ptx::mbarrier_init(&h1_mbar[t], 1);
            ptx::mbarrier_invalid(&h3_mbar[t]);
            ptx::mbarrier_init(&h3_mbar[t], 1);
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

        // Prefetch first 2 tile pairs
        #pragma unroll
        for (int pre = 0; pre < min(2, (int)V3_NUM_TILES); ++pre) {
            const int ty = pre / V3_TILES_X, tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h1[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &h1_mbar[pre]);
                ptx::mbarrier_arrive_expect_tx(&h3_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h3[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &h3_mbar[pre]);
            }
        }

        // Quantize chunk
        silu_quantize_and_store_chunk_pipelined<RETURN_TRANSPOSE>(
            sIn_h1_ptr, sIn_h3_ptr,
            sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            sOut, sOut_tr,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row, tmap_scale_col,
            S_enc,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y,
            h1_mbar, h3_mbar, mbar_phase,
            &tensor_map_h1, &tensor_map_h3);
        mbar_phase ^= 1;
    }

    // Cleanup
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&h1_mbar[t]);
            ptx::mbarrier_invalid(&h3_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}


// SMEM size for persistent silu+quantize (dual input buffers)
template <bool RETURN_TRANSPOSE>
inline int persistent_silu_quant_smem_size() {
    return fused_silu_quant::fused_silu_quant_smem_size<RETURN_TRANSPOSE>();
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace persistent_silu_quant

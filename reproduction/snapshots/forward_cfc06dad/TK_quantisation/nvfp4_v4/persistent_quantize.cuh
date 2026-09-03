// persistent_quantize.cuh — Persistent kernel for FP4 quantization (optimized)
//
// Optimizations over baseline:
//  1. "Last tile stays" — the last chunk scanned in Phase 1 is quantized
//     directly from SMEM after barrier, saving one TMA re-read per block
//  2. Mbarrier parity flipping — avoid invalidate+reinit cycle
//  3. L2 promotion hints in TMA map (handled in dispatch)
//
#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"

namespace tk_v4 {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;

struct PersistentArgs {
    unsigned int* work_counter_phase1;
    unsigned int* work_counter_phase2;
    float*        global_amax;
    unsigned int* done_counter;
    unsigned int* ready_flag;
    int tiles_X, tiles_Y, total_tiles;
    int num_persistent;
    float* sg_output;
    nvfp4_scale_t* col_scales_ptr;
    int64_t col_scale_stride;
    bool swizzle_scales;
};

// ─── Helper: quantize one chunk from SMEM and store via TMA + GMEM scales ───
template <bool RETURN_TRANSPOSE>
__device__ __forceinline__ void quantize_and_store_chunk(
    IType* sIn_ptr, fp4e2m1x2* sOut_ptr, fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr, nvfp4_scale_t* sSFcolwise_ptr,
    V3_OType2x3D& sOut, V3_OType2xt3D& sOut_tr,
    V3_ScalesType2D& sSFrowwise, V3_ScalesTypeTr2D& sSFcolwise,
    const CUtensorMap& tensor_map_output,
    const CUtensorMap& tensor_map_output_t,
    nvfp4_scale_t* scales_ptr,
    float S_enc,
    int block_offset_Y, int block_offset_X,
    int rows, int cols,
    int scale_stride,
    const PersistentArgs& args
) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_rows = (int)rows - block_offset_Y;
    const int chunk_cols = (int)cols - block_offset_X;
    const int block_offset_Y_tr = block_offset_X;  // for transpose TMA
    const int block_offset_X_tr = block_offset_Y;

    int buff_out = 0, buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        v3_rowwise_scaling(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                           S_enc, stage_Y, stage_X, t, buff_out);

        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                               S_enc, stage_Y, stage_X, t, buff_out_tr);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

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

    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    // Store rowwise scales (swizzled)
    {
        const int ntk = (int)scale_stride / 4;
        for (int row = threadIdx.x; row < (int)V3Config::CHUNK_DIM_Y; row += V3_THREADS) {
            const int abs_row = block_offset_Y + row;
            if (abs_row < (int)rows) {
                const int tm = abs_row / 128, rit = abs_row % 128;
                const int j = rit % 32, grp = rit / 32;
                const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
                const int sc_base = block_offset_X / V3_SCALE_DIM;
                for (int k = 0; k < cnt; ++k) {
                    const int sc_col = sc_base + k;
                    const int tile_k = sc_col / 4;
                    const int k_byte = sc_col % 4;
                    const int ts = (tm * ntk + tile_k) * 512 + j * 16 + grp * 4 + k_byte;
                    reinterpret_cast<uint8_t*>(scales_ptr)[ts] =
                        reinterpret_cast<const uint8_t&>(sSFrowwise[row][k]);
                }
            }
        }
    }

    // Store colwise scales
    if constexpr (RETURN_TRANSPOSE) {
        if (args.col_scales_ptr != nullptr) {
            const int col_sc_stride = (int)args.col_scale_stride;
            const int ntk_t = col_sc_stride / 4;
            const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
            const int sc_block_X_tr = block_offset_Y / V3_SCALE_DIM;
            for (int rtr = threadIdx.x; rtr < (int)V3Config::CHUNK_DIM_X; rtr += V3_THREADS) {
                const int abs_col = block_offset_X + rtr;
                if (abs_col < (int)cols) {
                    const int tm = abs_col / 128, rit = abs_col % 128;
                    const int j = rit % 32, grp = rit / 32;
                    for (int k = 0; k < cnt; ++k) {
                        const int sc_col = sc_block_X_tr + k;
                        const int tile_k = sc_col / 4;
                        const int k_byte = sc_col % 4;
                        const int ts = (tm * ntk_t + tile_k) * 512 + j * 16 + grp * 4 + k_byte;
                        reinterpret_cast<uint8_t*>(args.col_scales_ptr)[ts] =
                            reinterpret_cast<const uint8_t&>(sSFcolwise[rtr][k]);
                    }
                }
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Optimized persistent quantize kernel
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
persistent_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    nvfp4_scale_t* const scales_ptr,
    const size_t rows, const size_t cols,
    const size_t scale_stride,
    PersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*         sIn_ptr        = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*     sOut_ptr       = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

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
    // PHASE 1: Scan amax (work-stealing)
    // ═══════════════════════════════════════════════════════════════
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter_phase1, 1);
        }
        __syncthreads();
        unsigned int chunk_id = s_chunk_id;
        if (chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = chunk_id % args.tiles_X;
        const int ctaid_Y = chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        // TMA load
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            const int ty = t / V3_TILES_X, tx = t % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[t], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[t]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &in_mbar[t]);
            }
        }

        // Wait + scan
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            block_max = fmaxf(block_max, scan_tile_amax(sIn_ptr, t));
        }

        // Flip parity for next iteration (cheaper than invalidate+reinit)
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

    // ═══════════════════════════════════════════════════════════════
    // BARRIER
    // ═══════════════════════════════════════════════════════════════
    grid_barrier(block_max, args.global_amax,
                 args.done_counter, args.ready_flag,
                 args.num_persistent);

    const float amax_val = args.global_amax[0];
    const float S_enc = compute_global_encode_scaling_factor_FP4(amax_val);

    if (leading && blockIdx.x == 0) {
        if (args.sg_output) args.sg_output[0] = amax_val / 2688.0f;
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Quantize all tiles (work-stealing, L2 cache expected)
    // Reinitialize mbarriers — parity accumulated from Phase 1
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
        if (leading) {
            s_chunk_id2 = atomicAdd(args.work_counter_phase2, 1);
        }
        __syncthreads();
        unsigned int chunk_id = s_chunk_id2;
        if (chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = chunk_id % args.tiles_X;
        const int ctaid_Y = chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        // TMA load (L2 cache hit expected)
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            const int ty = t / V3_TILES_X, tx = t % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[t], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[t]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * V3_TILE_DIM_X,
                    block_offset_Y + ty * V3_TILE_DIM_Y,
                    &in_mbar[t]);
            }
        }

        // Wait for loads
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
        }
        mbar_phase ^= 1;

        quantize_and_store_chunk<RETURN_TRANSPOSE>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            sOut, sOut_tr, sSFrowwise, sSFcolwise,
            tensor_map_output, tensor_map_output_t,
            scales_ptr, S_enc,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols, (int)scale_stride, args);
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

} // namespace tk_v4

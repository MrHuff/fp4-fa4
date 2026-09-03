// persistent_quantize.cuh — v5: Persistent kernel with TMA scale output
//
// Changes from v4:
//  1. Scales are rearranged in-place in SMEM to swizzled layout, then
//     TMA bulk-stored instead of byte-level GMEM writes.
//  2. TMA tensor maps for row/col scales passed as __grid_constant__ params.
//
#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"

namespace tk_v5 {

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


// ─── Rearrange sSFrowwise[128][8] in-place to swizzled [1024] ───
// Swizzled layout: koffset*512 + j*16 + grp*4 + k_byte
// where j = row%32, grp = row/32, koffset = k/4, k_byte = k%4
//
__device__ __forceinline__ void swizzle_scales_row_inplace(
    nvfp4_scale_t* sSFrowwise_ptr, int num_scales_x
) {
    uint8_t my_scales[V3_SCALES_PER_CHUNK_X];

    for (int row = threadIdx.x; row < (int)V3Config::CHUNK_DIM_Y; row += V3_THREADS) {
        const int j   = row % 32;
        const int grp = row / 32;

        #pragma unroll
        for (int k = 0; k < V3_SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x)
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFrowwise_ptr[row * V3_SCALES_PER_CHUNK_X + k]);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < V3_SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                const int koffset = k / 4;
                const int k_byte  = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFrowwise_ptr)[dest] = my_scales[k];
            }
        }

        __syncthreads();
    }
}

// Same for colwise scales
__device__ __forceinline__ void swizzle_scales_col_inplace(
    nvfp4_scale_t* sSFcolwise_ptr, int num_scales_y
) {
    uint8_t my_scales[V3_SCALES_PER_CHUNK_Y];

    for (int col = threadIdx.x; col < (int)V3Config::CHUNK_DIM_X; col += V3_THREADS) {
        const int j   = col % 32;
        const int grp = col / 32;

        #pragma unroll
        for (int k = 0; k < V3_SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y)
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFcolwise_ptr[col * V3_SCALES_PER_CHUNK_Y + k]);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < V3_SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                const int koffset = k / 4;
                const int k_byte  = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFcolwise_ptr)[dest] = my_scales[k];
            }
        }

        __syncthreads();
    }
}

// ─── TMA scale store helper (2×512B stores per chunk) ───
__device__ __forceinline__
void tma_store_scales_2x512(
    const CUtensorMap& tmap, nvfp4_scale_t* smem_ptr,
    int tm_row, int tma_x_base
) {
    ptx::cp_async_bulk_tensor_2d_shared_to_global(
        reinterpret_cast<const uint64_t*>(&tmap),
        tma_x_base, tm_row,
        reinterpret_cast<uint64_t*>(smem_ptr));
    ptx::cp_async_bulk_tensor_2d_shared_to_global(
        reinterpret_cast<const uint64_t*>(&tmap),
        tma_x_base + 256, tm_row,
        reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_ptr) + 512));
    ptx::cp_async_bulk_commit_group();
}


// ─── Helper: pipelined quantize — waits for tiles individually, prefetches next ───
// Tiles 0 and 1 are already prefetched by the caller.
// For each tile t: wait mbar[t], quantize, prefetch t+2 (for next chunk via caller).
template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void quantize_and_store_chunk_pipelined_v5(
    IType* sIn_ptr, fp4e2m1x2* sOut_ptr, fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr, nvfp4_scale_t* sSFcolwise_ptr,
    V3_OType2x3D& sOut, V3_OType2xt3D& sOut_tr,
    V3_ScalesType2D& sSFrowwise, V3_ScalesTypeTr2D& sSFcolwise,
    const CUtensorMap& tensor_map_output,
    const CUtensorMap& tensor_map_output_t,
    const CUtensorMap& tmap_scale_row,
    const CUtensorMap& tmap_scale_col,
    float S_enc,
    int block_offset_Y, int block_offset_X,
    int rows, int cols,
    int ctaid_X, int ctaid_Y,
    uint64_t* in_mbar, int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_rows = (int)rows - block_offset_Y;
    const int chunk_cols = (int)cols - block_offset_X;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;

    int buff_out = 0, buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        auto& sIn = *reinterpret_cast<V3_IType3D*>(sIn_ptr);

        // Prefetch tile t+2 if it exists (overlaps with quantize of tile t)
        if (t + 2 < V3_NUM_TILES) {
            const int next = t + 2;
            const int nty = next / V3_TILES_X, ntx = next % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[next],
                    V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType));
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * V3_TILE_DIM_X,
                    block_offset_Y + nty * V3_TILE_DIM_Y,
                    &in_mbar[next]);
            }
        }

        // Wait for current tile
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        // Quantize tile t
        v3_rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                           S_enc, stage_Y, stage_X, t, buff_out);

        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
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
        swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);

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
        swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);

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


// ─── Helper: quantize one chunk from SMEM and store via TMA ───
template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void quantize_and_store_chunk_v5(
    IType* sIn_ptr, fp4e2m1x2* sOut_ptr, fp4e2m1x2* sOut_tr_ptr,
    nvfp4_scale_t* sSFrowwise_ptr, nvfp4_scale_t* sSFcolwise_ptr,
    V3_OType2x3D& sOut, V3_OType2xt3D& sOut_tr,
    V3_ScalesType2D& sSFrowwise, V3_ScalesTypeTr2D& sSFcolwise,
    const CUtensorMap& tensor_map_output,
    const CUtensorMap& tensor_map_output_t,
    const CUtensorMap& tmap_scale_row,
    const CUtensorMap& tmap_scale_col,
    float S_enc,
    int block_offset_Y, int block_offset_X,
    int rows, int cols,
    int ctaid_X, int ctaid_Y
) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_rows = (int)rows - block_offset_Y;
    const int chunk_cols = (int)cols - block_offset_X;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;

    int buff_out = 0, buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        v3_rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                           S_enc, stage_Y, stage_X, t, buff_out);

        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
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

    // Wait for all FP4 TMA stores
    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    // ─── TMA scale stores ───
    // Row scales: rearrange sSFrowwise[128][8] → swizzled [1024], then 2× TMA store
    {
        const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
        swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm = block_offset_Y / 128;
            // Two 512-byte TMA stores (256 BF16 elements each)
            const int tma_x_base = ctaid_X * 2 * 256;  // BF16 elements
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

    // Col scales: same approach
    if constexpr (RETURN_TRANSPOSE) {
        const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
        swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);

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

    // Wait for scale TMA stores to complete before reusing SMEM
    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}


// ═══════════════════════════════════════════════════════════════════
// v5 persistent quantize kernel with TMA scale output
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(V3_THREADS)
persistent_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
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
    // PHASE 1: Scan amax (work-stealing) — PIPELINED
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

        // Prefetch first 2 tiles (0 and 1)
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

        // Process each tile: wait for it, scan amax, prefetch next
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            // Prefetch tile t+2 if it exists
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

            // Wait for current tile and scan it
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            block_max = fmaxf(block_max, scan_tile_amax(sIn_ptr, t));
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
    // PHASE 2: Quantize (work-stealing) — PIPELINED
    // Overlap TMA loads with quantize computation
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

        // Process each tile: wait, quantize, prefetch next, TMA store
        quantize_and_store_chunk_pipelined_v5<RETURN_TRANSPOSE, ENCODE_CENTRIC>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            sOut, sOut_tr, sSFrowwise, sSFcolwise,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row, tmap_scale_col,
            S_enc,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y,
            in_mbar, mbar_phase,
            &tensor_map_input);
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

} // namespace tk_v5

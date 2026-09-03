/*
 * fused_group_amax_quantize.cuh — Fused single-pass grouped quantize kernel
 *
 * Same architecture as fused_amax_quantize.cuh:
 *   Phase 1: TMA-load all 4 tiles into SMEM → scan for per-group amax
 *   Barrier: per-group atomicMax + per-group spin-wait
 *   Phase 2: quantize from SMEM (data never left!)
 *
 * Supports both dim=0 (row splits) and dim=1 (column splits).
 * Group ID is determined by blockIdx.y (dim=0) or blockIdx.x (dim=1).
 *
 * Uses the same helpers (v3_rowwise_scaling, v3_colwise_scaling, scan_tile_amax)
 * from fused_amax_quantize.cuh.
 */

#ifndef TK_V3_FUSED_GROUP_AMAX_QUANTIZE_CUH_
#define TK_V3_FUSED_GROUP_AMAX_QUANTIZE_CUH_

#include "fused_amax_quantize.cuh"
#include "persistent_quantize.cuh"   // for tk_v5::swizzle_scales_*_inplace, tma_store_scales_2x512

namespace tk_v3 {

#if FP4_TYPE_SUPPORTED

constexpr int MAX_GROUPS = 8;

// ═══════════════════════════════════════════════════════════════════
// Per-group spin-wait barrier
// ═══════════════════════════════════════════════════════════════════
// Each group has its own global_amax, done_counter, ready_flag.
// Only CTAs belonging to the same group participate in the barrier.

__device__ __forceinline__
void group_grid_barrier(float cta_max,
                        float* __restrict__ group_amax,        // &global_amax[group_id]
                        unsigned int* __restrict__ group_done,  // &done_counter[group_id]
                        unsigned int* __restrict__ group_ready, // &ready_flag[group_id]
                        int blocks_in_group) {
    // Same logic as grid_barrier but per-group
    if (cta_max > 0.0f) {
        atomic_max_float(group_amax, cta_max);
    }

    __threadfence();

    __shared__ unsigned int s_ready;
    if (threadIdx.x == 0) {
        unsigned int prev = atomicAdd(group_done, 1);
        if (prev == (unsigned int)(blocks_in_group - 1)) {
            __threadfence();
            volatile unsigned int* vflag = (volatile unsigned int*)group_ready;
            *vflag = 1;
        }
        s_ready = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        volatile unsigned int* vflag = (volatile unsigned int*)group_ready;
        while (*vflag == 0) { /* spin */ }
        s_ready = 1;
    }
    __syncthreads();
}


// ═══════════════════════════════════════════════════════════════════
// Args struct for grouped fused kernel
// ═══════════════════════════════════════════════════════════════════
struct FusedGroupArgs {
    // Per-group sync buffers
    float*        global_amax;     // [num_groups] — must be zeroed
    unsigned int* done_counter;    // [num_groups] — must be zeroed
    unsigned int* ready_flag;      // [num_groups] — must be zeroed

    // Per-group output pointers
    nvfp4_scale_t* row_scale_ptrs[MAX_GROUPS];   // per-group row scale buffers
    fp4e2m1x2*     col_data_ptrs[MAX_GROUPS];    // per-group col FP4 buffers
    nvfp4_scale_t* col_scale_ptrs[MAX_GROUPS];   // per-group col scale buffers
    int            col_scale_stride[MAX_GROUPS];  // per-group col scale stride

    // Group boundaries (prefix sum)
    // For dim=0: row split boundaries. split_range[g+1] = sum of rows for splits 0..g
    // For dim=1: col split boundaries. split_range[g+1] = sum of cols for groups 0..g
    int split_range[MAX_GROUPS + 1];
    int num_groups;
    int blocks_per_group[MAX_GROUPS];  // number of CTAs in each group

    // sg / b_sg output
    float* sg_output;   // [num_groups]
    float* fwd_b_sg;    // per-tile b_sg for forward GEMM (dim=0 only)
    float* dgrad_b_sg;  // per-tile b_sg for dgrad GEMM (dim=0 only)
    int b_tile_size;    // typically 256
    int total_cols;     // K dimension

    // Whether to use swizzled scale layout
    bool swizzle_scales;

    // Device pointer to TMA maps for scale outputs (allocated on device, not in struct)
    // Layout: [0..num_groups-1] = row scale maps, [num_groups..2*num_groups-1] = col scale maps
    const CUtensorMap* scale_tma_maps;  // device pointer, 8 bytes
    bool use_tma_scales;  // if true, use TMA stores; if false, byte-level writes
    bool skip_amax;       // if true, amax is pre-computed in global_amax[], skip scan+barrier
};


// ═══════════════════════════════════════════════════════════════════
// Dim=0 grouped fused kernel
// ═══════════════════════════════════════════════════════════════════
// Group ID = split based on blockIdx.y position in row range

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(V3_THREADS)
fused_group_quantize_kernel_dim0(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    nvfp4_scale_t* const scales_ptr,    // unused (per-group ptrs in args)
    nvfp4_scale_t* const scales_t_ptr,  // unused for dim0
    const size_t rows, const size_t cols,
    const size_t scale_stride, const size_t scale_stride_t,
    FusedGroupArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // ─── SMEM layout (same as fused_amax_quantize.cuh) ───
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

    // ─── This CTA's chunk ───
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
    const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

    // ─── Determine group (split) ID from row position ───
    int group_id = 0;
    for (int g = 1; g < args.num_groups; ++g) {
        if (block_offset_Y >= args.split_range[g]) group_id = g;
    }
    const int split_start = args.split_range[group_id];
    const int split_end   = args.split_range[group_id + 1];

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
    // PHASE 1: Load ALL tiles + scan for amax (unless pre-computed)
    // ═════════════════════════════════════════════════════
    float cta_max = 0.0f;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int ty = t / V3_TILES_X, tx = t % V3_TILES_X;
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

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], 0);
        if (!args.skip_amax) {
            float tile_max = scan_tile_amax(sIn_ptr, t);
            cta_max = fmaxf(cta_max, tile_max);
        }
    }

    if (!args.skip_amax) {
        // Block-level reduction
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
        // BARRIER: per-group atomicMax + spin-wait
        // ═════════════════════════════════════════════════════
        group_grid_barrier(cta_max,
                           &args.global_amax[group_id],
                           &args.done_counter[group_id],
                           &args.ready_flag[group_id],
                           args.blocks_per_group[group_id]);
    }

    // ═════════════════════════════════════════════════════
    // PHASE 2: Quantize — data is STILL IN SMEM
    // ═════════════════════════════════════════════════════
    const float amax_val = args.global_amax[group_id];
    const float S_enc = compute_global_encode_scaling_factor_FP4(amax_val);
    const float sg_val = amax_val / 2688.0f;

    // Write sg, fwd_b_sg, dgrad_b_sg (first CTA in group)
    if (leading && ctaid_X == 0 && block_offset_Y == split_start) {
        if (args.sg_output) args.sg_output[group_id] = sg_val;

        if (args.fwd_b_sg) {
            int fwd_offset = 0;
            for (int s = 0; s < group_id; ++s) {
                fwd_offset += (args.split_range[s+1] - args.split_range[s]) / args.b_tile_size;
            }
            int fwd_tiles = (split_end - split_start) / args.b_tile_size;
            for (int t = 0; t < fwd_tiles; ++t)
                args.fwd_b_sg[fwd_offset + t] = sg_val;
        }
        if (args.dgrad_b_sg) {
            int dgrad_tiles_per = args.total_cols / args.b_tile_size;
            int dgrad_offset = group_id * dgrad_tiles_per;
            for (int t = 0; t < dgrad_tiles_per; ++t)
                args.dgrad_b_sg[dgrad_offset + t] = sg_val;
        }

        // Reset ready_flag for next invocation
        args.ready_flag[group_id] = 0;
    }

    // Get per-split output pointers
    nvfp4_scale_t* split_row_sc = args.row_scale_ptrs[group_id];
    const int local_row_offset_Y = block_offset_Y - split_start;

    const int block_offset_Y_tr = ctaid_X * V3Config::CHUNK_DIM_X;
    const int block_offset_X_tr = block_offset_Y;  // global row offset — TMA map covers contiguous buffer
    const int chunk_rows = (int)rows - block_offset_Y;
    const int chunk_cols = (int)cols - block_offset_X;

    int buff_out = 0, buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        if (t > 0) ptx::cp_async_bulk_wait_group_read<1>();

        // Quantize rowwise from SMEM
        v3_rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                           S_enc, stage_Y, stage_X, t, buff_out);

        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                               S_enc, stage_Y, stage_X, t, buff_out_tr);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store rowwise FP4
        if (leading) {
            const int gy = block_offset_Y + stage_offset_Y;
            const int gx = block_offset_X + stage_offset_X;
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                gx, gy, reinterpret_cast<uint64_t*>(&sOut[buff_out]));

            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    block_offset_X_tr + stage_offset_Y,  // col in transpose
                    block_offset_Y_tr + stage_offset_X,  // row in transpose
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
        buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
    }

    ptx::cp_async_bulk_wait_group_read<0>();

    // ─── Store scales ───
    if (args.use_tma_scales) {
        // TMA path: swizzle in-place then bulk store via device-pointer TMA maps
        {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
            tk_v5::swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                const int local_row = block_offset_Y - split_start;
                const int tm = local_row / 128;
                const int tma_x_base = ctaid_X * 2 * 256;
                tk_v5::tma_store_scales_2x512(args.scale_tma_maps[group_id], sSFrowwise_ptr,
                                              tm, tma_x_base);
            }
        }

        if constexpr (RETURN_TRANSPOSE) {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
            tk_v5::swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                const int tm_col = block_offset_X / 128;
                const int split_ctaid_Y = (block_offset_Y - split_start) / V3Config::CHUNK_DIM_Y;
                const int tma_x_base = split_ctaid_Y * 2 * 256;
                tk_v5::tma_store_scales_2x512(args.scale_tma_maps[args.num_groups + group_id], sSFcolwise_ptr,
                                              tm_col, tma_x_base);
            }
        }

        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();
    } else {
        // Byte-level path
        {
            const int ntk = (int)scale_stride / 4;
            for (int row = threadIdx.x; row < (int)V3Config::CHUNK_DIM_Y; row += V3_THREADS) {
                const int abs_row = block_offset_Y + row;
                if (abs_row < (int)rows) {
                    const int local_row = abs_row - split_start;
                    const int tm = local_row / 128, rit = local_row % 128;
                    const int j = rit % 32, grp = rit / 32;
                    const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
                    const int sc_block_X_row = ctaid_X * V3_SCALES_PER_CHUNK_X;
                    for (int kg = 0; kg < cnt / 4; ++kg) {
                        const int kb = kg * 4, kgb = sc_block_X_row + kb;
                        const int ts = (tm * ntk + kgb / 4) * 512 + j * 16 + grp * 4;
                        uint32_t pk; uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
                        p[0] = reinterpret_cast<const uint8_t&>(sSFrowwise[row][kb]);
                        p[1] = reinterpret_cast<const uint8_t&>(sSFrowwise[row][kb+1]);
                        p[2] = reinterpret_cast<const uint8_t&>(sSFrowwise[row][kb+2]);
                        p[3] = reinterpret_cast<const uint8_t&>(sSFrowwise[row][kb+3]);
                        *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(split_row_sc) + ts) = pk;
                    }
                    for (int k = (cnt/4)*4; k < cnt; ++k) {
                        const int kg2 = sc_block_X_row + k;
                        const int ts = (tm * ntk + kg2/4) * 512 + j*16 + grp*4 + kg2%4;
                        reinterpret_cast<uint8_t*>(split_row_sc)[ts] =
                            reinterpret_cast<const uint8_t&>(sSFrowwise[row][k]);
                    }
                }
            }
        }

        if constexpr (RETURN_TRANSPOSE) {
            nvfp4_scale_t* split_col_sc = args.col_scale_ptrs[group_id];
            const int col_sc_stride = args.col_scale_stride[group_id];
            if (split_col_sc != nullptr) {
                const int ntk_t = col_sc_stride / 4;
                const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
                const int sc_block_Y_tr = ctaid_X * V3Config::CHUNK_DIM_X;
                const int sc_block_X_tr = (block_offset_Y - split_start) / V3_SCALE_DIM;
                for (int rtr = threadIdx.x; rtr < (int)V3Config::CHUNK_DIM_X; rtr += V3_THREADS) {
                    const int rtg = sc_block_Y_tr + rtr;
                    if (rtg < (int)cols) {
                        const int tm = rtg / 128, rit = rtg % 128;
                        const int j = rit % 32, grp = rit / 32;
                        for (int kg = 0; kg < cnt / 4; ++kg) {
                            const int kb = kg * 4, kgb = sc_block_X_tr + kb;
                            const int ts = (tm * ntk_t + kgb / 4) * 512 + j * 16 + grp * 4;
                            uint32_t pk; uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
                            p[0] = reinterpret_cast<const uint8_t&>(sSFcolwise[rtr][kb]);
                            p[1] = reinterpret_cast<const uint8_t&>(sSFcolwise[rtr][kb+1]);
                            p[2] = reinterpret_cast<const uint8_t&>(sSFcolwise[rtr][kb+2]);
                            p[3] = reinterpret_cast<const uint8_t&>(sSFcolwise[rtr][kb+3]);
                            *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(split_col_sc) + ts) = pk;
                        }
                        for (int k = (cnt/4)*4; k < cnt; ++k) {
                            const int kg2 = sc_block_X_tr + k;
                            const int ts = (tm * ntk_t + kg2/4) * 512 + j*16 + grp*4 + kg2%4;
                            reinterpret_cast<uint8_t*>(split_col_sc)[ts] =
                                reinterpret_cast<const uint8_t&>(sSFcolwise[rtr][k]);
                        }
                    }
                }
            }
        }
    }

    // Clean up
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_invalid(&in_mbar[t]);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Dim=1 grouped fused kernel
// ═══════════════════════════════════════════════════════════════════
// Group ID = column group based on blockIdx.x position

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(V3_THREADS)
fused_group_quantize_kernel_dim1(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    nvfp4_scale_t* const scales_ptr,    // unused
    const size_t rows, const size_t cols,
    const size_t scale_stride, const size_t scale_stride_t,
    FusedGroupArgs args
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

    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
    const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

    // ─── Determine column group ID ───
    int group_id = 0;
    for (int g = 1; g < args.num_groups; ++g) {
        if (block_offset_X >= args.split_range[g]) group_id = g;
    }
    const int col_group_start = args.split_range[group_id];

    // ─── TMA barriers ───
    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_init(&in_mbar[t], 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // PHASE 1: Load + scan (unless pre-computed)
    float cta_max = 0.0f;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int ty = t / V3_TILES_X, tx = t % V3_TILES_X;
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

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], 0);
        if (!args.skip_amax) {
            cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
        }
    }

    if (!args.skip_amax) {
        // Block reduction
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

        // Per-group barrier
        group_grid_barrier(cta_max,
                           &args.global_amax[group_id],
                           &args.done_counter[group_id],
                           &args.ready_flag[group_id],
                           args.blocks_per_group[group_id]);
    }

    // PHASE 2: Quantize
    const float amax_val = args.global_amax[group_id];
    const float S_enc = compute_global_encode_scaling_factor_FP4(amax_val);

    if (leading && ctaid_Y == 0 && block_offset_X == col_group_start) {
        if (args.sg_output) args.sg_output[group_id] = amax_val / 2688.0f;
        args.ready_flag[group_id] = 0;
    }

    nvfp4_scale_t* split_row_sc = args.row_scale_ptrs[group_id];

    // Transpose coordinates — global, TMA writes to contiguous [cols, rows] buffer
    const int block_offset_Y_tr = ctaid_X * V3Config::CHUNK_DIM_X;
    const int block_offset_X_tr = block_offset_Y;  // global row offset
    const int chunk_rows = (int)rows - block_offset_Y;
    const int chunk_cols = (int)cols - block_offset_X;

    int buff_out = 0, buff_out_tr = 0;

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        if (t > 0) ptx::cp_async_bulk_wait_group_read<1>();

        v3_rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                           S_enc, stage_Y, stage_X, t, buff_out);

        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                               S_enc, stage_Y, stage_X, t, buff_out_tr);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store rowwise FP4
        if (leading) {
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));

            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                    block_offset_X_tr + stage_offset_Y,  // col in transpose = original row
                    block_offset_Y_tr + stage_offset_X,  // row in transpose = original col
                    reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
        buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
    }

    ptx::cp_async_bulk_wait_group_read<0>();

    // ─── Store scales ───
    if (args.use_tma_scales) {
        // TMA path via device-pointer maps
        {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
            tk_v5::swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                const int tm = block_offset_Y / 128;
                const int local_col_base = block_offset_X - col_group_start;
                const int local_ctaid_X = local_col_base / (int)V3Config::CHUNK_DIM_X;
                const int tma_x_base = local_ctaid_X * 2 * 256;
                tk_v5::tma_store_scales_2x512(args.scale_tma_maps[group_id], sSFrowwise_ptr,
                                              tm, tma_x_base);
            }
        }

        if constexpr (RETURN_TRANSPOSE) {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
            tk_v5::swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);

            ptx::fence_proxy_async_shared_cta();
            __syncthreads();

            if (leading) {
                const int local_col = block_offset_X - col_group_start;
                const int tm_col = local_col / 128;
                const int tma_x_base = ctaid_Y * 2 * 256;
                tk_v5::tma_store_scales_2x512(args.scale_tma_maps[args.num_groups + group_id], sSFcolwise_ptr,
                                              tm_col, tma_x_base);
            }
        }

        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();
    } else {
        // Byte-level path
        {
            const int local_col_base = block_offset_X - col_group_start;
            const int col_group_size = args.split_range[group_id + 1] - col_group_start;
            const int row_sc_ntk = ((col_group_size / 16 + 3) / 4);

            for (int row = threadIdx.x; row < (int)V3Config::CHUNK_DIM_Y; row += V3_THREADS) {
                const int abs_row = block_offset_Y + row;
                if (abs_row < (int)rows) {
                    const int tm = abs_row / 128, rit = abs_row % 128;
                    const int j = rit % 32, grp = rit / 32;
                    const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
                    const int sc_base = local_col_base / V3_SCALE_DIM;
                    for (int k = 0; k < cnt; ++k) {
                        const int local_sc_col = sc_base + k;
                        const int tile_k = local_sc_col / 4;
                        const int k_byte = local_sc_col % 4;
                        const int ts = (tm * row_sc_ntk + tile_k) * 512 + j * 16 + grp * 4 + k_byte;
                        reinterpret_cast<uint8_t*>(split_row_sc)[ts] =
                            reinterpret_cast<const uint8_t&>(sSFrowwise[row][k]);
                    }
                }
            }
        }

        if constexpr (RETURN_TRANSPOSE) {
            nvfp4_scale_t* col_sc = args.col_scale_ptrs[group_id];
            const int col_sc_stride_val = args.col_scale_stride[group_id];
            if (col_sc != nullptr && col_sc_stride_val > 0) {
                const int ntk_t = col_sc_stride_val / 4;
                const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
                const int local_col_base = block_offset_X - col_group_start;
                const int sc_block_X_tr = block_offset_Y / V3_SCALE_DIM;

                for (int rtr = threadIdx.x; rtr < (int)V3Config::CHUNK_DIM_X; rtr += V3_THREADS) {
                    const int local_col_t = local_col_base + rtr;
                    const int col_group_size = args.split_range[group_id + 1] - col_group_start;
                    if (local_col_t < col_group_size) {
                        const int tm = local_col_t / 128, rit = local_col_t % 128;
                        const int j = rit % 32, grp = rit / 32;
                        for (int k = 0; k < cnt; ++k) {
                            const int sc_col = sc_block_X_tr + k;
                            const int tile_k = sc_col / 4;
                            const int k_byte = sc_col % 4;
                            const int ts = (tm * ntk_t + tile_k) * 512 + j * 16 + grp * 4 + k_byte;
                            reinterpret_cast<uint8_t*>(col_sc)[ts] =
                                reinterpret_cast<const uint8_t&>(sSFcolwise[rtr][k]);
                        }
                    }
                }
            }
        }
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_invalid(&in_mbar[t]);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}


#endif  // FP4_TYPE_SUPPORTED
}  // namespace tk_v3

#endif  // TK_V3_FUSED_GROUP_AMAX_QUANTIZE_CUH_

// persistent_group_quantize.cuh — v5: Persistent grouped quantize
//
// TMA scale maps are on device memory, passed via pointer (not embedded in struct).
// This avoids kernel arg size overflow that caused illegal memory access.
//
#pragma once
#include "fused_group_amax_quantize.cuh"
#include "persistent_quantize.cuh"   // for swizzle_scales_*_inplace, tma_store_scales_2x512

namespace tk_v5 {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;


// ═══════════════════════════════════════════════════════════════════
// Args struct for persistent grouped kernel
// ═══════════════════════════════════════════════════════════════════
struct PersistentGroupArgs {
    unsigned int* work_counter_phase1;
    unsigned int* work_counter_phase2;

    float* global_amax;
    unsigned int* done_counter;
    unsigned int* ready_flag;

    int tiles_X, tiles_Y, total_tiles;
    int num_persistent;

    int num_groups;
    int split_range[MAX_GROUPS + 1];
    int blocks_per_group[MAX_GROUPS];

    nvfp4_scale_t* row_scale_ptrs[MAX_GROUPS];
    fp4e2m1x2*     col_data_ptrs[MAX_GROUPS];
    nvfp4_scale_t* col_scale_ptrs[MAX_GROUPS];
    int            col_scale_stride[MAX_GROUPS];

    float* sg_output;
    float* fwd_b_sg;
    float* dgrad_b_sg;
    int    b_tile_size;
    int    total_cols;

    bool swizzle_scales;

    // Device pointer to TMA maps for scale outputs (allocated on device, not embedded)
    // Layout: [0..num_groups-1] = row scale maps, [num_groups..2*num_groups-1] = col scale maps
    const CUtensorMap* scale_tma_maps;  // device pointer, 8 bytes
};


// ─── Helper: find group for a given position ───
__device__ __forceinline__
int find_group_v5(int pos, const int* split_range, int num_groups) {
    int g = 0;
    #pragma unroll
    for (int i = 1; i < MAX_GROUPS; ++i) {
        if (i < num_groups && pos >= split_range[i]) g = i;
    }
    return g;
}

// ─── Grid barrier (count-based sync) ───
__device__ __forceinline__
void grid_barrier_sync_v5(unsigned int* done_counter, unsigned int* ready_flag,
                          int num_persistent) {
    __threadfence();
    __shared__ unsigned int s_ready;
    if (threadIdx.x == 0) {
        unsigned int prev = atomicAdd(done_counter, 1);
        if (prev == (unsigned int)(num_persistent - 1)) {
            __threadfence();
            volatile unsigned int* vflag = (volatile unsigned int*)ready_flag;
            *vflag = 1;
        }
        s_ready = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        volatile unsigned int* vflag = (volatile unsigned int*)ready_flag;
        while (*vflag == 0) { /* spin */ }
        s_ready = 1;
    }
    __syncthreads();
}


// ═══════════════════════════════════════════════════════════════════
// Dim=0 persistent grouped kernel with TMA scale output (device-pointer)
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(V3_THREADS)
persistent_group_quantize_kernel_dim0(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    nvfp4_scale_t* const scales_ptr,
    const size_t rows, const size_t cols,
    const size_t scale_stride, const size_t scale_stride_t,
    PersistentGroupArgs args
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

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*         sIn_ptr        = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*     sOut_ptr       = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn    = *reinterpret_cast<V3_IType3D*>(sIn_ptr);
    auto& sOut   = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr= *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_init(&in_mbar[t], 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float group_max[MAX_GROUPS];
    #pragma unroll
    for (int g = 0; g < MAX_GROUPS; ++g) group_max[g] = 0.0f;

    int mbar_phase = 0;

    // PHASE 1: Scan amax (work-stealing)
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter_phase1, 1);
        __syncthreads();
        unsigned int chunk_id = s_chunk_id;
        if (chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = chunk_id % args.tiles_X;
        const int ctaid_Y = chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;
        int gid = find_group_v5(block_offset_Y, args.split_range, args.num_groups);

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

        float chunk_max = 0.0f;
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            chunk_max = fmaxf(chunk_max, scan_tile_amax(sIn_ptr, t));
        }
        mbar_phase ^= 1;
        group_max[gid] = fmaxf(group_max[gid], chunk_max);
    }

    // Block reduction per-group
    {
        #pragma unroll
        for (int g = 0; g < MAX_GROUPS; ++g) {
            if (g >= args.num_groups) break;
            float val = group_max[g];
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            __shared__ float warp_gmax[V3_THREADS / 32];
            int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
            if (lane == 0) warp_gmax[wid] = val;
            __syncthreads();
            if (wid == 0) {
                val = (lane < V3_THREADS / 32) ? warp_gmax[lane] : 0.0f;
                #pragma unroll
                for (int mask = (V3_THREADS / 32) / 2; mask > 0; mask >>= 1)
                    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            }
            if (threadIdx.x == 0 && val > 0.0f)
                atomic_max_float(&args.global_amax[g], val);
            __syncthreads();
        }
    }

    grid_barrier_sync_v5(args.done_counter, args.ready_flag, args.num_persistent);

    if (leading && blockIdx.x == 0) {
        for (int g = 0; g < args.num_groups; ++g) {
            float amax_g = args.global_amax[g];
            float sg_val = amax_g / 2688.0f;
            if (args.sg_output) args.sg_output[g] = sg_val;
            if (args.fwd_b_sg) {
                int fwd_offset = 0;
                for (int s = 0; s < g; ++s)
                    fwd_offset += (args.split_range[s+1] - args.split_range[s]) / args.b_tile_size;
                int fwd_tiles = (args.split_range[g+1] - args.split_range[g]) / args.b_tile_size;
                for (int t = 0; t < fwd_tiles; ++t)
                    args.fwd_b_sg[fwd_offset + t] = sg_val;
            }
            if (args.dgrad_b_sg) {
                int dgrad_tiles_per = args.total_cols / args.b_tile_size;
                int dgrad_offset = g * dgrad_tiles_per;
                for (int t = 0; t < dgrad_tiles_per; ++t)
                    args.dgrad_b_sg[dgrad_offset + t] = sg_val;
            }
        }
    }

    // PHASE 2: Quantize (work-stealing)
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
        unsigned int chunk_id = s_chunk_id2;
        if (chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = chunk_id % args.tiles_X;
        const int ctaid_Y = chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;
        int gid = find_group_v5(block_offset_Y, args.split_range, args.num_groups);
        int split_start = args.split_range[gid];
        float S_enc = compute_global_encode_scaling_factor_FP4(args.global_amax[gid]);

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

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
        mbar_phase ^= 1;

        const int chunk_rows = (int)rows - block_offset_Y;
        const int chunk_cols = (int)cols - block_offset_X;
        int buff_out = 0, buff_out_tr = 0;

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            const int stage_Y = t / V3_TILES_X, stage_X = t % V3_TILES_X;
            v3_rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                               S_enc, stage_Y, stage_X, t, buff_out);
            if constexpr (RETURN_TRANSPOSE)
                v3_colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                   S_enc, stage_Y, stage_X, t, buff_out_tr);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output),
                    block_offset_X + stage_X * V3_TILE_DIM_X,
                    block_offset_Y + stage_Y * V3_TILE_DIM_Y,
                    reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                if constexpr (RETURN_TRANSPOSE)
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                        block_offset_Y + stage_Y * V3_TILE_DIM_Y,
                        block_offset_X + stage_X * V3_TILE_DIM_X,
                        reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                ptx::cp_async_bulk_commit_group();
            }
            buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
            buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
        }

        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        // TMA scale stores: row (via device-pointer TMA map)
        {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
            swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                const int tm = (block_offset_Y - split_start) / 128;
                const int tma_x_base = ctaid_X * 2 * 256;
                tma_store_scales_2x512(args.scale_tma_maps[gid], sSFrowwise_ptr, tm, tma_x_base);
            }
        }

        // TMA scale stores: col
        if constexpr (RETURN_TRANSPOSE) {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
            swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                const int tm_col = block_offset_X / 128;
                const int split_ctaid_Y = (block_offset_Y - split_start) / V3Config::CHUNK_DIM_Y;
                const int tma_x_base = split_ctaid_Y * 2 * 256;
                tma_store_scales_2x512(args.scale_tma_maps[args.num_groups + gid], sSFcolwise_ptr, tm_col, tma_x_base);
            }
        }

        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();
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


// ═══════════════════════════════════════════════════════════════════
// Dim=1 persistent grouped kernel with TMA scale output (device-pointer)
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(V3_THREADS)
persistent_group_quantize_kernel_dim1(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    nvfp4_scale_t* const scales_ptr,
    const size_t rows, const size_t cols,
    const size_t scale_stride, const size_t scale_stride_t,
    PersistentGroupArgs args
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

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*         sIn_ptr        = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*     sOut_ptr       = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn    = *reinterpret_cast<V3_IType3D*>(sIn_ptr);
    auto& sOut   = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr= *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_init(&in_mbar[t], 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float group_max[MAX_GROUPS];
    #pragma unroll
    for (int g = 0; g < MAX_GROUPS; ++g) group_max[g] = 0.0f;
    int mbar_phase = 0;

    // PHASE 1: amax
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter_phase1, 1);
        __syncthreads();
        unsigned int chunk_id = s_chunk_id;
        if (chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = chunk_id % args.tiles_X;
        const int ctaid_Y = chunk_id / args.tiles_X;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        int gid = find_group_v5(block_offset_X, args.split_range, args.num_groups);

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

        float chunk_max = 0.0f;
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            chunk_max = fmaxf(chunk_max, scan_tile_amax(sIn_ptr, t));
        }
        mbar_phase ^= 1;
        group_max[gid] = fmaxf(group_max[gid], chunk_max);
    }

    // Block reduction
    {
        #pragma unroll
        for (int g = 0; g < MAX_GROUPS; ++g) {
            if (g >= args.num_groups) break;
            float val = group_max[g];
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            __shared__ float warp_gmax[V3_THREADS / 32];
            int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
            if (lane == 0) warp_gmax[wid] = val;
            __syncthreads();
            if (wid == 0) {
                val = (lane < V3_THREADS / 32) ? warp_gmax[lane] : 0.0f;
                #pragma unroll
                for (int mask = (V3_THREADS / 32) / 2; mask > 0; mask >>= 1)
                    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            }
            if (threadIdx.x == 0 && val > 0.0f)
                atomic_max_float(&args.global_amax[g], val);
            __syncthreads();
        }
    }

    grid_barrier_sync_v5(args.done_counter, args.ready_flag, args.num_persistent);

    if (leading && blockIdx.x == 0) {
        for (int g = 0; g < args.num_groups; ++g)
            if (args.sg_output) args.sg_output[g] = args.global_amax[g] / 2688.0f;
    }

    // PHASE 2: quantize
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
        unsigned int chunk_id = s_chunk_id2;
        if (chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = chunk_id % args.tiles_X;
        const int ctaid_Y = chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;
        int gid = find_group_v5(block_offset_X, args.split_range, args.num_groups);
        int col_group_start = args.split_range[gid];
        float S_enc = compute_global_encode_scaling_factor_FP4(args.global_amax[gid]);

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

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
        mbar_phase ^= 1;

        const int chunk_rows = (int)rows - block_offset_Y;
        const int chunk_cols = (int)cols - block_offset_X;
        int buff_out = 0, buff_out_tr = 0;

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            const int stage_Y = t / V3_TILES_X, stage_X = t % V3_TILES_X;
            v3_rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                               S_enc, stage_Y, stage_X, t, buff_out);
            if constexpr (RETURN_TRANSPOSE)
                v3_colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
                                   S_enc, stage_Y, stage_X, t, buff_out_tr);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output),
                    block_offset_X + stage_X * V3_TILE_DIM_X,
                    block_offset_Y + stage_Y * V3_TILE_DIM_Y,
                    reinterpret_cast<uint64_t*>(&sOut[buff_out]));
                if constexpr (RETURN_TRANSPOSE)
                    ptx::cp_async_bulk_tensor_2d_shared_to_global(
                        reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
                        block_offset_Y + stage_Y * V3_TILE_DIM_Y,
                        block_offset_X + stage_X * V3_TILE_DIM_X,
                        reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
                ptx::cp_async_bulk_commit_group();
            }
            buff_out = (buff_out + 1) % V3_BUFFS_NUM_OUT;
            buff_out_tr = (buff_out_tr + 1) % V3_BUFFS_NUM_OUT_TR;
        }

        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        // TMA scale stores: row (via device-pointer TMA map)
        {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
            swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                const int tm = block_offset_Y / 128;
                const int local_col = block_offset_X - col_group_start;
                const int local_ctaid_X = local_col / (int)V3Config::CHUNK_DIM_X;
                const int tma_x_base = local_ctaid_X * 2 * 256;
                tma_store_scales_2x512(args.scale_tma_maps[gid], sSFrowwise_ptr, tm, tma_x_base);
            }
        }

        // TMA scale stores: col
        if constexpr (RETURN_TRANSPOSE) {
            const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
            swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                const int local_col = block_offset_X - col_group_start;
                const int tm_col = local_col / 128;
                const int tma_x_base = ctaid_Y * 2 * 256;
                tma_store_scales_2x512(args.scale_tma_maps[args.num_groups + gid], sSFcolwise_ptr, tm_col, tma_x_base);
            }
        }

        if (leading) ptx::cp_async_bulk_wait_group_read<0>();
        __syncthreads();
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

} // namespace tk_v5

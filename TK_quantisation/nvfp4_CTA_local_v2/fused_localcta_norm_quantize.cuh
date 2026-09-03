#ifndef TK_LOCALCTA_FUSED_NORM_QUANTIZE_CUH_
#define TK_LOCALCTA_FUSED_NORM_QUANTIZE_CUH_

#include "fused_localcta_quantize.cuh"

namespace tk_localcta {

#if FP4_TYPE_SUPPORTED

__device__ __forceinline__ float localcta_device_silu(float x) {
    return x / (1.0f + expf(-x));
}

template <bool WITH_SILU, bool USE_ILP_AMAX>
__device__ __forceinline__ float scan_and_transform_tile_localcta(
    IType* sIn_ptr,
    const IType* gamma_smem,
    const float* inv_rms_row,
    int tile_idx,
    int chunk_row_start,
    int tile_row_offset,
    int rows
) {
    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);

    const int tid_Y = threadIdx.x / (TILE_DIM_X / ELTS_PER_THREAD);
    const int tid_X = threadIdx.x % (TILE_DIM_X / ELTS_PER_THREAD);
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    IType2 tile_max_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    float tile_max = 0.0f;

    constexpr int THREADS_X = TILE_DIM_X / ELTS_PER_THREAD;
    constexpr int THREADS_Y = THREADS / THREADS_X;
    constexpr int ITERS = TILE_DIM_Y / THREADS_Y;

    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int local_row = tid_Y + it * THREADS_Y;
        const int global_row = chunk_row_start + tile_row_offset + local_row;

        float row_inv_rms = 0.0f;
        if (global_row < rows) {
            row_inv_rms = inv_rms_row[local_row + tile_row_offset];
        }

        #pragma unroll
        for (int e = 0; e < ELTS_PER_THREAD; e += 2) {
            const int col = thread_offset_X + e;
            float x0 = __bfloat162float(sIn[tile_idx][local_row][col + 0]);
            float x1 = __bfloat162float(sIn[tile_idx][local_row][col + 1]);
            float g0 = __bfloat162float(gamma_smem[col + 0]);
            float g1 = __bfloat162float(gamma_smem[col + 1]);

            float v0 = x0 * row_inv_rms * g0;
            float v1 = x1 * row_inv_rms * g1;
            if constexpr (WITH_SILU) {
                v0 = localcta_device_silu(v0);
                v1 = localcta_device_silu(v1);
            }

            const IType2 transformed_pair = {
                __float2bfloat16_rn(v0),
                __float2bfloat16_rn(v1)
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

template <bool WITH_SILU, bool RETURN_TRANSPOSE, bool USE_ILP_AMAX = true, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS)
fused_localcta_norm_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const float* __restrict__ inv_rms,
    const IType* __restrict__ gamma,
    const size_t rows,
    const size_t cols,
    LocalCTAPersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int gamma_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_X * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);
    IType* gamma_cache =
        reinterpret_cast<IType*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes);

    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    __shared__ float warp_max[THREADS / 32];
    __shared__ float cta_amax_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter, 1);
        }
        __syncthreads();
        if (s_chunk_id >= static_cast<unsigned int>(args.total_tiles)) {
            break;
        }

        const int ctaid_X = static_cast<int>(s_chunk_id % args.tiles_X);
        const int ctaid_Y = static_cast<int>(s_chunk_id / args.tiles_X);
        const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
        const int chunk_cols = static_cast<int>(cols) - block_offset_X;

        for (int i = threadIdx.x; i < LocalCTAConfig::CHUNK_DIM_X; i += THREADS) {
            const int gc = block_offset_X + i;
            gamma_cache[i] = (gc < static_cast<int>(cols)) ? gamma[gc] : static_cast<IType>(0.0f);
        }
        __syncthreads();

        float cta_max = 0.0f;

        #pragma unroll
        for (int pre = 0; pre < min(2, (int)NUM_TILES); ++pre) {
            const int ty = pre / TILES_X;
            const int tx = pre % TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&in_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &in_mbar[pre]);
            }
        }

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            if (t + 2 < NUM_TILES) {
                const int next = t + 2;
                const int ty = next / TILES_X;
                const int tx = next % TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn[next]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

            const int ty = t / TILES_X;
            const int tx = t % TILES_X;
            cta_max = fmaxf(
                cta_max,
                scan_and_transform_tile_localcta<WITH_SILU, USE_ILP_AMAX>(
                    sIn_ptr,
                    gamma_cache + tx * TILE_DIM_X,
                    inv_rms + block_offset_Y,
                    t,
                    block_offset_Y,
                    ty * TILE_DIM_Y,
                    static_cast<int>(rows)));
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
        }

        if (lane == 0) {
            warp_max[wid] = cta_max;
        }
        __syncthreads();

        if (wid == 0) {
            cta_max = (lane < THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (THREADS / 32) / 2; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                cta_amax_shared = cta_max;
            }
        }
        __syncthreads();

        const float amax_val = cta_amax_shared;
        const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
        const float sg_val = amax_val / localcta_global_scale_num();

        if (leading) {
            row_sg_chunks[ctaid_Y * args.tiles_X + ctaid_X] = sg_val;
            if constexpr (RETURN_TRANSPOSE) {
                const int tiles_Y = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                col_sg_chunks[ctaid_X * tiles_Y + ctaid_Y] = sg_val;
            }
        }

        int buff_out = 0;
        int buff_out_tr = 0;

        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            const int stage_Y = t / TILES_X;
            const int stage_X = t % TILES_X;
            const int stage_offset_Y = stage_Y * TILE_DIM_Y;
            const int stage_offset_X = stage_X * TILE_DIM_X;

            if (t > 0) {
                ptx::cp_async_bulk_wait_group_read<1>();
            }

            rowwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_ptr, sSFrowwise_ptr,
                                            S_enc, stage_Y, stage_X, t, buff_out);

            if constexpr (RETURN_TRANSPOSE) {
                colwise_scaling<ENCODE_CENTRIC>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr,
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

            buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
            buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
        }

        if (leading) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        {
            const int cnt = min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM);
            swizzle_scales_row_inplace(sSFrowwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
            }
        }

        if constexpr (RETURN_TRANSPOSE) {
            const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
            swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (leading) {
                tma_store_scales_2x512(tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
            }
        }

        if (leading) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
        mbar_phase ^= 1;
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
inline int norm_shmem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t),
        TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t),
                          TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int gamma_bytes = DIVUP_TO_MULTIPLE(
        LocalCTAConfig::CHUNK_DIM_X * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);

    return in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes + gamma_bytes +
           TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace tk_localcta

#endif  // TK_LOCALCTA_FUSED_NORM_QUANTIZE_CUH_

/*************************************************************************
 * Persistent Fused SiLU Derivative + Dual Quantize Kernel
 *
 * Implements the 2-phase work-stealing loop for fused_silu_deriv_quantize.
 *
 * Designed to process `dh` and `h13` (where h13 is effectively two matrices)
 * and output two quantized tensors: `dh1_fp4` and `dh3_fp4`.
 *
 * Template RETURN_TRANSPOSE: when true, also produces colwise-quantized
 * (transposed) outputs for both dh1 and dh3 (needed for wgrad GEMMs).
 *************************************************************************/

#pragma once
#include <cuda_fp4.h>
#include "fused_silu_deriv_quantize.cuh"
#include "persistent_quantize.cuh"

namespace persistent_silu_deriv_quant {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

// ─── Helper: pipelined quantize and store for dual outputs ───
// Reads dh, h1, h3 from SMEM, computes SiLU deriv + quantizes to dh1_fp4 and dh3_fp4,
// directly writes to GMEM via TMA.
template <bool RETURN_TRANSPOSE>
__device__ __forceinline__ void silu_deriv_quantize_and_store_chunk_pipelined(
    IType* sIn_dh_ptr, IType* sIn_h1_ptr, IType* sIn_h3_ptr,
    fp4e2m1x2* sOut1_ptr, fp4e2m1x2* sOut2_ptr,
    fp4e2m1x2* sOut1_tr_ptr, fp4e2m1x2* sOut2_tr_ptr,
    nvfp4_scale_t* sSFrowwise1_ptr, nvfp4_scale_t* sSFrowwise2_ptr,
    nvfp4_scale_t* sSFcolwise1_ptr, nvfp4_scale_t* sSFcolwise2_ptr,
    V3_OType2x3D& sOut1, V3_OType2x3D& sOut2,
    V3_OType2xt3D& sOut1_tr, V3_OType2xt3D& sOut2_tr,
    const CUtensorMap& tmap_out1, const CUtensorMap& tmap_out2,
    const CUtensorMap& tmap_out1_t, const CUtensorMap& tmap_out2_t,
    const CUtensorMap& tmap_scale_row1, const CUtensorMap& tmap_scale_row2,
    const CUtensorMap& tmap_scale_col1, const CUtensorMap& tmap_scale_col2,
    float S_enc1, float S_enc2,
    int block_offset_Y, int block_offset_X,
    int rows, int cols,
    int ctaid_X, int ctaid_Y,
    uint64_t* dh_mbar, uint64_t* h1_mbar, uint64_t* h3_mbar, int mbar_phase,
    const CUtensorMap* tmap_dh, const CUtensorMap* tmap_h1, const CUtensorMap* tmap_h3
) {
    const bool leading = (threadIdx.x == 0);
    const int chunk_cols = (int)cols - block_offset_X;
    const int chunk_rows = (int)rows - block_offset_Y;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;
    int buff_out1 = 0, buff_out2 = 0;
    int buff_out1_tr = 0, buff_out2_tr = 0;

    auto& sIn_dh = *reinterpret_cast<V3_IType3D*>(sIn_dh_ptr);
    auto& sIn_h1 = *reinterpret_cast<V3_IType3D*>(sIn_h1_ptr);
    auto& sIn_h3 = *reinterpret_cast<V3_IType3D*>(sIn_h3_ptr);

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
            if (leading) {
                constexpr int bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[next], bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_dh[next]), reinterpret_cast<const uint64_t*>(tmap_dh),
                    block_offset_X + ntx * V3_TILE_DIM_X, block_offset_Y + nty * V3_TILE_DIM_Y, &dh_mbar[next]);

                ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h1[next]), reinterpret_cast<const uint64_t*>(tmap_h1),
                    block_offset_X + ntx * V3_TILE_DIM_X, block_offset_Y + nty * V3_TILE_DIM_Y, &h1_mbar[next]);
                
                ptx::mbarrier_arrive_expect_tx(&h3_mbar[next], bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_h3[next]), reinterpret_cast<const uint64_t*>(tmap_h3),
                    block_offset_X + ntx * V3_TILE_DIM_X, block_offset_Y + nty * V3_TILE_DIM_Y, &h3_mbar[next]);
            }
        }

        // Wait for tiles
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[t], mbar_phase);
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], mbar_phase);

        // Compute silu_deriv: results written to h1 (=dh1) and h3 (=dh3) buffers
        fused_silu_deriv_quant::scan_and_silu_deriv_tile(sIn_dh_ptr, sIn_h1_ptr, sIn_h3_ptr, t);
        __syncthreads();

        // Row quantize for dh1 (from sIn_h1) and dh3 (from sIn_h3)
        v3_rowwise_scaling(sIn_h1_ptr, sOut1_ptr, sSFrowwise1_ptr, S_enc1, stage_Y, stage_X, t, buff_out1);
        v3_rowwise_scaling(sIn_h3_ptr, sOut2_ptr, sSFrowwise2_ptr, S_enc2, stage_Y, stage_X, t, buff_out2);

        // Col quantize (transposed) for dh1 and dh3
        if constexpr (RETURN_TRANSPOSE) {
            v3_colwise_scaling(sIn_h1_ptr, sOut1_tr_ptr, sSFcolwise1_ptr, S_enc1, stage_Y, stage_X, t, buff_out1_tr);
            v3_colwise_scaling(sIn_h3_ptr, sOut2_tr_ptr, sSFcolwise2_ptr, S_enc2, stage_Y, stage_X, t, buff_out2_tr);
        }

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA Stores for row outputs
        if (leading) {
            // dh1 row
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_out1),
                block_offset_X + stage_offset_X, block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut1[buff_out1]));
            // dh3 row
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_out2),
                block_offset_X + stage_offset_X, block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut2[buff_out2]));

            // Col (transposed) outputs
            if constexpr (RETURN_TRANSPOSE) {
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_out1_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut1_tr[buff_out1_tr]));
                ptx::cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tmap_out2_t),
                    block_offset_X_tr + stage_offset_Y,
                    block_offset_Y_tr + stage_offset_X,
                    reinterpret_cast<uint64_t*>(&sOut2_tr[buff_out2_tr]));
            }
            ptx::cp_async_bulk_commit_group();
        }

        buff_out1 = (buff_out1 + 1) % V3_BUFFS_NUM_OUT;
        buff_out2 = (buff_out2 + 1) % V3_BUFFS_NUM_OUT;
        if constexpr (RETURN_TRANSPOSE) {
            buff_out1_tr = (buff_out1_tr + 1) % V3_BUFFS_NUM_OUT_TR;
            buff_out2_tr = (buff_out2_tr + 1) % V3_BUFFS_NUM_OUT_TR;
        }
    }

    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    // ─── TMA Scale Stores (row) ─── 
    {
        const int cnt = min((int)V3_SCALES_PER_CHUNK_X, chunk_cols / (int)V3_SCALE_DIM);
        tk_v5::swizzle_scales_row_inplace(sSFrowwise1_ptr, cnt);
        tk_v5::swizzle_scales_row_inplace(sSFrowwise2_ptr, cnt);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm = block_offset_Y / 128;
            const int tma_x_base = ctaid_X * 2 * 256;
            
            // out 1 row scales
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row1),
                tma_x_base, tm, reinterpret_cast<uint64_t*>(sSFrowwise1_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row1),
                tma_x_base + 256, tm, reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFrowwise1_ptr) + 512));
                
            // out 2 row scales
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row2),
                tma_x_base, tm, reinterpret_cast<uint64_t*>(sSFrowwise2_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_row2),
                tma_x_base + 256, tm, reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFrowwise2_ptr) + 512));
                
            ptx::cp_async_bulk_commit_group();
        }
    }

    // ─── TMA Scale Stores (col) ───
    if constexpr (RETURN_TRANSPOSE) {
        const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, chunk_rows / (int)V3_SCALE_DIM);
        tk_v5::swizzle_scales_col_inplace(sSFcolwise1_ptr, cnt);
        tk_v5::swizzle_scales_col_inplace(sSFcolwise2_ptr, cnt);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            const int tm_col = block_offset_X / 128;
            const int tma_x_base = ctaid_Y * 2 * 256;

            // out 1 col scales
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col1),
                tma_x_base, tm_col, reinterpret_cast<uint64_t*>(sSFcolwise1_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col1),
                tma_x_base + 256, tm_col, reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFcolwise1_ptr) + 512));

            // out 2 col scales
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col2),
                tma_x_base, tm_col, reinterpret_cast<uint64_t*>(sSFcolwise2_ptr));
            ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tmap_scale_col2),
                tma_x_base + 256, tm_col, reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sSFcolwise2_ptr) + 512));

            ptx::cp_async_bulk_commit_group();
        }
    }

    if (leading) ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}

// ═══════════════════════════════════════════════════════════════════
// Persistent Fused SiLU Deriv + Quantize Kernel
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
persistent_silu_deriv_quantize_kernel(
    const __grid_constant__ CUtensorMap tmap_dh,      // dh[:,:] stride=H
    const __grid_constant__ CUtensorMap tmap_h1,      // h13[:,0:H] stride=2H
    const __grid_constant__ CUtensorMap tmap_h3,      // h13[:,H:2H] stride=2H
    const __grid_constant__ CUtensorMap tmap_out1,    // dh1_fp4 row
    const __grid_constant__ CUtensorMap tmap_out2,    // dh3_fp4 row
    const __grid_constant__ CUtensorMap tmap_out1_t,  // dh1_fp4 col (transposed)
    const __grid_constant__ CUtensorMap tmap_out2_t,  // dh3_fp4 col (transposed)
    const __grid_constant__ CUtensorMap tmap_scale_row1, // dh1 row scales
    const __grid_constant__ CUtensorMap tmap_scale_row2, // dh3 row scales
    const __grid_constant__ CUtensorMap tmap_scale_col1, // dh1 col scales
    const __grid_constant__ CUtensorMap tmap_scale_col2, // dh3 col scales
    const size_t rows, const size_t cols,             // cols = H
    const size_t scale_stride,
    tk_v5::PersistentArgs args,
    float* global_amax2  // args.global_amax covers out1, this covers out2
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // SMEM layout
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    // 3 input buffers
    IType* sIn_dh_ptr = reinterpret_cast<IType*>(dshmem);
    IType* sIn_h1_ptr = reinterpret_cast<IType*>(dshmem + in_bytes);
    IType* sIn_h3_ptr = reinterpret_cast<IType*>(dshmem + 2 * in_bytes);
    
    // 2 row output buffers + 2 col output buffers
    int off = 3 * in_bytes;
    fp4e2m1x2* sOut1_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_bytes;
    fp4e2m1x2* sOut2_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_bytes;
    fp4e2m1x2* sOut1_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_tr_bytes;
    fp4e2m1x2* sOut2_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + off); off += out_tr_bytes;
    
    // 2 row scale buffers + 2 col scale buffers
    nvfp4_scale_t* sSFrowwise1_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_row_bytes;
    nvfp4_scale_t* sSFrowwise2_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_row_bytes;
    nvfp4_scale_t* sSFcolwise1_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off); off += sc_col_bytes;
    nvfp4_scale_t* sSFcolwise2_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + off);

    auto& sDh = *reinterpret_cast<V3_IType3D*>(sIn_dh_ptr);
    auto& sH1 = *reinterpret_cast<V3_IType3D*>(sIn_h1_ptr);
    auto& sH3 = *reinterpret_cast<V3_IType3D*>(sIn_h3_ptr);
    auto& sOut1 = *reinterpret_cast<V3_OType2x3D*>(sOut1_ptr);
    auto& sOut2 = *reinterpret_cast<V3_OType2x3D*>(sOut2_ptr);
    auto& sOut1_tr = *reinterpret_cast<V3_OType2xt3D*>(sOut1_tr_ptr);
    auto& sOut2_tr = *reinterpret_cast<V3_OType2xt3D*>(sOut2_tr_ptr);
    
    __shared__ uint64_t dh_mbar[V3_NUM_TILES];
    __shared__ uint64_t h1_mbar[V3_NUM_TILES];
    __shared__ uint64_t h3_mbar[V3_NUM_TILES];

    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_init(&dh_mbar[t], 1);
            ptx::mbarrier_init(&h1_mbar[t], 1);
            ptx::mbarrier_init(&h3_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float block_max1 = 0.0f;
    float block_max2 = 0.0f;
    int mbar_phase = 0;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Scan amax1 and amax2 (work-stealing)
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

        // Prefetch first 2 tiles
        #pragma unroll
        for (int pre = 0; pre < min(2, (int)V3_NUM_TILES); ++pre) {
            const int ty = pre / V3_TILES_X, tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[pre]), reinterpret_cast<const uint64_t*>(&tmap_dh),
                    block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &dh_mbar[pre]);
                    
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[pre]), reinterpret_cast<const uint64_t*>(&tmap_h1),
                    block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &h1_mbar[pre]);
                    
                ptx::mbarrier_arrive_expect_tx(&h3_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH3[pre]), reinterpret_cast<const uint64_t*>(&tmap_h3),
                    block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &h3_mbar[pre]);
            }
        }

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            if (t + 2 < V3_NUM_TILES) {
                const int next = t + 2;
                const int ty = next / V3_TILES_X, tx = next % V3_TILES_X;
                if (leading) {
                    ptx::mbarrier_arrive_expect_tx(&dh_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sDh[next]), reinterpret_cast<const uint64_t*>(&tmap_dh),
                        block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &dh_mbar[next]);
                        
                    ptx::mbarrier_arrive_expect_tx(&h1_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH1[next]), reinterpret_cast<const uint64_t*>(&tmap_h1),
                        block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &h1_mbar[next]);
                        
                    ptx::mbarrier_arrive_expect_tx(&h3_mbar[next], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sH3[next]), reinterpret_cast<const uint64_t*>(&tmap_h3),
                        block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &h3_mbar[next]);
                }
            }

            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&dh_mbar[t], mbar_phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], mbar_phase);
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], mbar_phase);

            float2 tile_maxes = fused_silu_deriv_quant::scan_and_silu_deriv_tile(sIn_dh_ptr, sIn_h1_ptr, sIn_h3_ptr, t);
            block_max1 = fmaxf(block_max1, tile_maxes.x);
            block_max2 = fmaxf(block_max2, tile_maxes.y);
        }
        mbar_phase ^= 1;
    }

    // Block reduction for both amaxes
    {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            block_max1 = fmaxf(block_max1, __shfl_xor_sync(0xffffffff, block_max1, mask));
            block_max2 = fmaxf(block_max2, __shfl_xor_sync(0xffffffff, block_max2, mask));
        }

        __shared__ float warp_max1[V3_THREADS / 32];
        __shared__ float warp_max2[V3_THREADS / 32];
        int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) {
            warp_max1[wid] = block_max1;
            warp_max2[wid] = block_max2;
        }
        __syncthreads();
        if (wid == 0) {
            block_max1 = (lane < V3_THREADS / 32) ? warp_max1[lane] : 0.0f;
            block_max2 = (lane < V3_THREADS / 32) ? warp_max2[lane] : 0.0f;
            #pragma unroll
            for (int mask = (V3_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                block_max1 = fmaxf(block_max1, __shfl_xor_sync(0xffffffff, block_max1, mask));
                block_max2 = fmaxf(block_max2, __shfl_xor_sync(0xffffffff, block_max2, mask));
            }
        }
    }

    // Grid barrier for both amaxes
    __shared__ bool is_last_block;
    if (leading) {
        if (block_max1 > 0.0f) atomicMax((unsigned int*)args.global_amax, __float_as_uint(block_max1));
        if (block_max2 > 0.0f) atomicMax((unsigned int*)global_amax2, __float_as_uint(block_max2));
        
        __threadfence();
        unsigned int cnt = atomicAdd(args.done_counter, 1);
        is_last_block = (cnt == args.num_persistent - 1);
    }
    __syncthreads();

    if (is_last_block) {
        if (leading) {
            atomicExch(args.ready_flag, 1);
            *args.done_counter = 0;
        }
    } else {
        while (*(volatile unsigned int*)args.ready_flag != 1) {}
    }
    __syncthreads();

    const float amax_val1 = __uint_as_float(*(volatile unsigned int*)args.global_amax);
    const float amax_val2 = __uint_as_float(*(volatile unsigned int*)global_amax2);
    const float S_enc1 = compute_global_encode_scaling_factor_FP4(amax_val1);
    const float S_enc2 = compute_global_encode_scaling_factor_FP4(amax_val2);

    if (leading && blockIdx.x == 0) {
        if (args.sg_output) {
            args.sg_output[0] = amax_val1 / 2688.0f;
            args.sg_output[1] = amax_val2 / 2688.0f;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Re-load dh, h1, h3, recompute, quantize to dh1/dh3 FP4
    // ═══════════════════════════════════════════════════════════════
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&dh_mbar[t]);
            ptx::mbarrier_init(&dh_mbar[t], 1);
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

        // Prefetch first 2 tiles
        #pragma unroll
        for (int pre = 0; pre < min(2, (int)V3_NUM_TILES); ++pre) {
            const int ty = pre / V3_TILES_X, tx = pre % V3_TILES_X;
            if (leading) {
                ptx::mbarrier_arrive_expect_tx(&dh_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sDh[pre]), reinterpret_cast<const uint64_t*>(&tmap_dh),
                    block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &dh_mbar[pre]);
                    
                ptx::mbarrier_arrive_expect_tx(&h1_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH1[pre]), reinterpret_cast<const uint64_t*>(&tmap_h1),
                    block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &h1_mbar[pre]);
                    
                ptx::mbarrier_arrive_expect_tx(&h3_mbar[pre], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sH3[pre]), reinterpret_cast<const uint64_t*>(&tmap_h3),
                    block_offset_X + tx * V3_TILE_DIM_X, block_offset_Y + ty * V3_TILE_DIM_Y, &h3_mbar[pre]);
            }
        }

        silu_deriv_quantize_and_store_chunk_pipelined<RETURN_TRANSPOSE>(
            sIn_dh_ptr, sIn_h1_ptr, sIn_h3_ptr,
            sOut1_ptr, sOut2_ptr,
            sOut1_tr_ptr, sOut2_tr_ptr,
            sSFrowwise1_ptr, sSFrowwise2_ptr,
            sSFcolwise1_ptr, sSFcolwise2_ptr,
            sOut1, sOut2, sOut1_tr, sOut2_tr,
            tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
            tmap_scale_row1, tmap_scale_row2,
            tmap_scale_col1, tmap_scale_col2,
            S_enc1, S_enc2,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y,
            dh_mbar, h1_mbar, h3_mbar, mbar_phase,
            &tmap_dh, &tmap_h1, &tmap_h3);
        mbar_phase ^= 1;
    }

    // Cleanup
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&dh_mbar[t]);
            ptx::mbarrier_invalid(&h1_mbar[t]);
            ptx::mbarrier_invalid(&h3_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

// SMEM size for persistent dual silu deriv+quantize
template <bool RETURN_TRANSPOSE>
inline int fused_silu_deriv_quant_smem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;

    // 3 inputs + 2 row outputs + 2 col outputs + 2 row scales + 2 col scales
    return 3 * in_bytes + 2 * out_bytes + 2 * out_tr_bytes
         + 2 * sc_row_bytes + 2 * sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

// Keep non-templated overload for backward compatibility
inline int persistent_silu_deriv_quant_smem_size() {
    return fused_silu_deriv_quant_smem_size<false>();
}

} // namespace persistent_silu_deriv_quant
#endif

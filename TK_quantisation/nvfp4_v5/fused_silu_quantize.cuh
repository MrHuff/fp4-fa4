/*************************************************************************
 * Fused Strided-SiLU + Quantize Kernel
 *
 * Eliminates the intermediate bf16 GMEM write between SiLU and quantize.
 *
 * Architecture (2 phases, same as fused_amax_quantize / fused_norm_quantize):
 *   Phase 1: TMA load h1 tile (from h13[:,0:H]) and h3 tile (from h13[:,H:2H])
 *            into SMEM. In registers: compute silu(h1)*h3, write bf16 back
 *            to SMEM slot, scan amax.
 *   Barrier: grid-level global amax
 *   Phase 2: Data is STILL in SMEM (already transformed) — quantize + TMA store
 *
 * Input:  h13 (M, 2H) bf16 — h1 in columns [0, H), h3 in columns [H, 2H)
 * Output: fp4 (M, H) + scales — same format as fused_amax_quantize
 *
 * Data savings vs old approach:
 *   OLD: read h13(M,2H), write h(M,H), read h(M,H), write fp4 = 4 passes
 *   NEW: read h13(M,2H), write fp4 = 2 passes (SiLU applied in-register)
 *
 * SMEM budget (128x128 chunk with dual input tiles):
 *   h1 tiles: 4 × 64×64 × 2B = 32 KB
 *   h3 tiles: 4 × 64×64 × 2B = 32 KB
 *   Output:   2 × 64×32 × 1B =  4 KB
 *   Out_tr:   2 × 64×16 × 1B =  2 KB
 *   Scales:                   ≈  2 KB
 *   Total:                    ≈ 72 KB → ~3 CTAs/SM on GB200 (228KB)
 *************************************************************************/

#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"

namespace fused_silu_quant {

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

// ─── Scan tile amax WITH SiLU transform ───
// Reads h1 from sIn_h1[tile_idx] and h3 from sIn_h3[tile_idx],
// computes silu(h1)*h3, writes result to sIn_h1[tile_idx] (reuses buffer),
// returns tile max |silu(h1)*h3|.
__device__ __forceinline__
float scan_and_silu_tile(
    IType* sIn_h1_ptr,     // SMEM buffer for h1 tiles
    const IType* sIn_h3_ptr, // SMEM buffer for h3 tiles
    int tile_idx
) {
    auto& sH1 = *reinterpret_cast<V3_IType3D*>(sIn_h1_ptr);
    const auto& sH3 = *reinterpret_cast<const V3_IType3D*>(sIn_h3_ptr);

    const int tid_Y = threadIdx.x / (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int tid_X = threadIdx.x % (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    float tile_max = 0.0f;

    constexpr int THREADS_X = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
    constexpr int THREADS_Y = V3_THREADS / THREADS_X;
    constexpr int ITERS = V3_TILE_DIM_Y / THREADS_Y;

    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int local_row = tid_Y + it * THREADS_Y;

        #pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_X + e;
            float h1_val = __bfloat162float(sH1[tile_idx][local_row][col]);
            float h3_val = __bfloat162float(sH3[tile_idx][local_row][col]);

            float result = device_silu(h1_val) * h3_val;

            // Write transformed bf16 back to h1's SMEM slot (for Phase 2 quantize)
            sH1[tile_idx][local_row][col] = __float2bfloat16_rn(result);

            tile_max = fmaxf(tile_max, fabsf(result));
        }
    }

    return tile_max;
}


// ═══════════════════════════════════════════════════════════════════
// Main kernel: fused silu(h1)*h3 + quantize
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(V3_THREADS)
fused_silu_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_h1,     // h13[:,0:H]
    const __grid_constant__ CUtensorMap tensor_map_h3,     // h13[:,H:2H]
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    nvfp4_scale_t* const scales_ptr,
    nvfp4_scale_t* const scales_t_ptr,
    float* __restrict__ global_amax,
    float* __restrict__ sg_out,
    unsigned int* __restrict__ done_counter,
    unsigned int* __restrict__ ready_flag,
    const size_t rows, const size_t cols,    // cols = H (output width)
    const size_t scale_stride, const size_t scale_stride_t,
    const int total_blocks
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // ─── SMEM layout ───
    // [h1 input: 4 tiles]  [h3 input: 4 tiles]  [output: 2]  [output_tr: 2]  [scales_row]  [scales_col]
    constexpr int in_bytes_h1  = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int in_bytes_h3  = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
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
    IType*         sIn_h3_ptr     = reinterpret_cast<IType*>(dshmem + in_bytes_h1);
    fp4e2m1x2*     sOut_ptr       = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes_h1 + in_bytes_h3);
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes_h1 + in_bytes_h3 + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes_h1 + in_bytes_h3 + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes_h1 + in_bytes_h3 + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn_h1    = *reinterpret_cast<V3_IType3D*>(sIn_h1_ptr);
    auto& sIn_h3    = *reinterpret_cast<V3_IType3D*>(sIn_h3_ptr);

    // ─── This CTA's chunk ───
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
    const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

    // ─── TMA barriers: one per h1 tile + one per h3 tile ───
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

    // ═════════════════════════════════════════════════════
    // PHASE 1: Load h1+h3 tiles, apply SiLU, scan amax
    // ═════════════════════════════════════════════════════
    float cta_max = 0.0f;

    // Launch TMA for all 4 tiles of h1 AND h3
    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int ty = t / V3_TILES_X;
        const int tx = t % V3_TILES_X;
        const int gy = block_offset_Y + ty * V3_TILE_DIM_Y;
        const int gx = block_offset_X + tx * V3_TILE_DIM_X;
        if (leading) {
            // h1 tile
            ptx::mbarrier_arrive_expect_tx(&h1_mbar[t], shmem_tile_bytes);
            ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn_h1[t]),
                reinterpret_cast<const uint64_t*>(&tensor_map_h1),
                gx, gy, &h1_mbar[t]);
            // h3 tile (same row positions, same column positions — the TMA map
            // was created with column offset H, so gx indexes into h3's columns)
            ptx::mbarrier_arrive_expect_tx(&h3_mbar[t], shmem_tile_bytes);
            ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn_h3[t]),
                reinterpret_cast<const uint64_t*>(&tensor_map_h3),
                gx, gy, &h3_mbar[t]);
        }
    }

    // Wait for each tile pair and apply SiLU transform
    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h1_mbar[t], 0);
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&h3_mbar[t], 0);

        float tile_max = scan_and_silu_tile(sIn_h1_ptr, sIn_h3_ptr, t);
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
    // PHASE 2: Quantize — transformed data is IN sIn_h1 SMEM
    // This is IDENTICAL to fused_amax_quantize Phase 2
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

    // Quantize from sIn_h1 (which now contains silu(h1)*h3 bf16 data)
    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        if (t > 0) {
            ptx::cp_async_bulk_wait_group_read<1>();
        }

        // Quantize: data for tile t is in sIn_h1[t] — SiLU-transformed in Phase 1
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

    // Wait for all TMA stores to complete
    ptx::cp_async_bulk_wait_group_read<0>();

    // ─── Store scales to global (TK swizzle format) ───
    {
        auto& sSFrowwise = *reinterpret_cast<V3_ScalesType2D*>(sSFrowwise_ptr);
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
                    p[0] = reinterpret_cast<const uint8_t&>(sSFrowwise[row][kb]);
                    p[1] = reinterpret_cast<const uint8_t&>(sSFrowwise[row][kb+1]);
                    p[2] = reinterpret_cast<const uint8_t&>(sSFrowwise[row][kb+2]);
                    p[3] = reinterpret_cast<const uint8_t&>(sSFrowwise[row][kb+3]);
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(scales_ptr) + ts) = pk;
                }
                for (int k = (cnt/4)*4; k < cnt; ++k) {
                    const int kg2 = sc_block_X_row + k;
                    const int ts = (tm * ntk + kg2/4) * 512 + j*16 + grp*4 + kg2%4;
                    reinterpret_cast<uint8_t*>(scales_ptr)[ts] =
                        reinterpret_cast<const uint8_t&>(sSFrowwise[row][k]);
                }
            }
        }
    }

    if constexpr (RETURN_TRANSPOSE) {
        auto& sSFcolwise = *reinterpret_cast<V3_ScalesTypeTr2D*>(sSFcolwise_ptr);
        const int ntk_t = static_cast<int>(scale_stride_t) / 4;
        const int cnt = min((int)V3_SCALES_PER_CHUNK_Y, (int)(chunk_rows / V3_SCALE_DIM));
        for (size_t rtr = threadIdx.x; rtr < V3Config::CHUNK_DIM_X; rtr += V3_THREADS) {
            const size_t rtg = sc_block_Y_tr + rtr;
            if (rtg < cols) {
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
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(scales_t_ptr) + ts) = pk;
                }
                for (int k = (cnt/4)*4; k < cnt; ++k) {
                    const int kg2 = sc_block_X_tr + k;
                    const int ts = (tm * ntk_t + kg2/4) * 512 + j*16 + grp*4 + kg2%4;
                    reinterpret_cast<uint8_t*>(scales_t_ptr)[ts] =
                        reinterpret_cast<const uint8_t&>(sSFcolwise[rtr][k]);
                }
            }
        }
    }

    // Clean up mbarriers
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


// ─── SMEM size for fused silu+quantize kernel ───
template <bool RETURN_TRANSPOSE>
inline int fused_silu_quant_smem_size() {
    constexpr int in_h1 = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int in_h3 = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;

    return in_h1 + in_h3 + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace fused_silu_quant

/*************************************************************************
 * Fused SiLU Derivative + Dual Quantize Kernel
 *
 * Eliminates the intermediate bf16 GMEM writes between SiLU deriv and 
 * dual quantizations (dh1, dh3).
 *
 * Architecture (2 phases, same as fused_silu_quantize):
 *   Phase 1: TMA load dh, h1, h3 tiles into SMEM. 
 *            In registers: 
 *                sig(x)   = 1 / (1 + exp(-x))
 *                silu(x)  = x * sig(x)
 *                silup(x) = sig(x) * (1 + x - silu(x))
 *                dh1 = dh * h3 * silup(h1)
 *                dh3 = dh * silu(h1)
 *            Write bf16 results back to SMEM (re-using h1/h3 slots).
 *            Scan amax1 (from dh1) and amax2 (from dh3).
 *   Barrier: grid-level global amax for both outputs.
 *   Phase 2: Data is STILL in SMEM (already transformed). 
 *            Quantize dh1 and dh3 + TMA store.
 *
 * Input:  dh (M, H) bf16
 *         h13 (M, 2H) bf16 — h1 in columns [0, H), h3 in columns [H, 2H)
 * Output: dh1_fp4 (M, H) + dh1_scales
 *         dh3_fp4 (M, H) + dh3_scales
 *
 * Data savings vs old approach:
 *   OLD: read dh(M,H), read h13(M,2H), write dh13(M,2H), 
 *        read dh13(M,2H) twice, write fp4(M,H)*2 = 9 passes
 *   NEW: read dh(M,H), read h13(M,2H), write fp4(M,H)*2 = 5 passes
 *
 * SMEM budget (128x128 chunk):
 *   dh  tiles: 4 × 64×64 × 2B = 32 KB
 *   h1  tiles: 4 × 64×64 × 2B = 32 KB
 *   h3  tiles: 4 × 64×64 × 2B = 32 KB
 *   Out dh1:   2 × 64×32 × 1B =  4 KB
 *   Out dh3:   2 × 64×32 × 1B =  4 KB
 *   Scales:                   ≈  4 KB
 *   Total:                    ≈ 108 KB → ~2 CTAs/SM on GB200 (228KB)
 *************************************************************************/

#pragma once
#include <cuda_fp4.h>
#include "fused_amax_quantize.cuh"

namespace fused_silu_deriv_quant {

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::ptx;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

// ─── Device helpers for SiLU and SiLU derivative ───
__device__ __forceinline__ void compute_silu_and_deriv(float x, float& silu_out, float& silup_out) {
    float sig = 1.0f / (1.0f + expf(-x));
    silu_out = x * sig;
    silup_out = sig * (1.0f + x - silu_out);
}

// ─── Scan tile amax WITH SiLU deriv transform ───
// Reads dh from sIn_dh, h1 from sIn_h1, h3 from sIn_h3.
// Computes dh1 = dh * h3 * silup(h1) and dh3 = dh * silu(h1).
// Writes dh1 back to sIn_h1 (reuses buffer) and dh3 to sIn_h3 (reuses buffer).
// Returns a float2 containing (tile_max_dh1, tile_max_dh3).
__device__ __forceinline__ float2 scan_and_silu_deriv_tile(
    const IType* sIn_dh_ptr,   // SMEM buffer for dh tiles
    IType* sIn_h1_ptr,         // SMEM buffer for h1 tiles (will store dh1)
    IType* sIn_h3_ptr,         // SMEM buffer for h3 tiles (will store dh3)
    int tile_idx
) {
    const auto& sDh = *reinterpret_cast<const V3_IType3D*>(sIn_dh_ptr);
    auto& sH1       = *reinterpret_cast<V3_IType3D*>(sIn_h1_ptr);
    auto& sH3       = *reinterpret_cast<V3_IType3D*>(sIn_h3_ptr);

    const int tid_Y = threadIdx.x / (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int tid_X = threadIdx.x % (V3_TILE_DIM_X / V3_ELTS_PER_THREAD);
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    float tile_max1 = 0.0f;
    float tile_max2 = 0.0f;

    constexpr int THREADS_X = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
    constexpr int THREADS_Y = V3_THREADS / THREADS_X;
    constexpr int ITERS = V3_TILE_DIM_Y / THREADS_Y;

    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int local_row = tid_Y + it * THREADS_Y;

        #pragma unroll
        for (int e = 0; e < V3_ELTS_PER_THREAD; ++e) {
            const int col = thread_offset_X + e;
            float dh_val = __bfloat162float(sDh[tile_idx][local_row][col]);
            float h1_val = __bfloat162float(sH1[tile_idx][local_row][col]);
            float h3_val = __bfloat162float(sH3[tile_idx][local_row][col]);

            float silu_v1, silup_v1;
            compute_silu_and_deriv(h1_val, silu_v1, silup_v1);

            float dh1_val = dh_val * h3_val * silup_v1;
            float dh3_val = dh_val * silu_v1;

            // Write transformed bf16 back to SMEM slots (for Phase 2 quantize)
            sH1[tile_idx][local_row][col] = __float2bfloat16_rn(dh1_val);
            sH3[tile_idx][local_row][col] = __float2bfloat16_rn(dh3_val);

            tile_max1 = fmaxf(tile_max1, fabsf(dh1_val));
            tile_max2 = fmaxf(tile_max2, fabsf(dh3_val));
        }
    }

    return make_float2(tile_max1, tile_max2);
}

// ─── SMEM size for fused silu_deriv+quantize kernel ───
// We do not need a transposed FP4 output buffer because backwards is typically
// only row-major layout (to match weight gradient requirements).
inline int fused_silu_deriv_quant_smem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    
    int total_in = in_bytes * 3;  // dh, h1, h3
    
    // Outputs: 2 (dh1, dh3) * 2 buffers
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    int total_out = out_bytes * 2;
    
    // Scales: 2 scale row buffers (dh1, dh3)
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    int total_sc = sc_row_bytes * 2;
    
    return total_in + total_out + total_sc + TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace fused_silu_deriv_quant

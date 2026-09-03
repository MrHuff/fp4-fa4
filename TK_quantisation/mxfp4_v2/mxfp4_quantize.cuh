/*************************************************************************
 * MXFP4 Quantize Kernel — Persistent (v2)
 *
 * Architecture:
 *   - Persistent CTAs (gridDim = num_SMs), work-stealing via atomicAdd
 *   - Single phase (no global amax needed for MXFP4)
 *   - TMA bulk load + TMA bulk store for FP4 output
 *   - Scalar global writes for scales (512B per chunk — negligible)
 *   - Each thread processes 1 row of 128 (4 × 32-element MX blocks)
 *   - E8M0 scales use round-ties-to-even
 *
 * SMEM layout:
 *   Input:  128×128×2B = 32 KB
 *   FP4:    128×64×1B  =  8 KB
 *   Total: ~40 KB
 *************************************************************************/

#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cstdio>

#define TK_STANDALONE
#include "../nvfp4_v5/util/cast_common.h"
#include "../nvfp4_v5/util/ptx.cuh"
#include "../nvfp4_v5/util/utils.cuh"

namespace mxfp4_v2 {

using namespace transformer_engine;
using namespace transformer_engine::ptx;

// ═══════════════════════════════════════════════════════════════════
// Configuration constants
// ═══════════════════════════════════════════════════════════════════

static constexpr int MX_CHUNK_DIM   = 128;     // 128×128 bf16 tile
static constexpr int MX_THREADS     = 128;     // 1 thread per row
static constexpr int MX_SCALE_BLOCK = 32;      // MXFP4 block size
static constexpr int MX_NUM_BLOCKS  = MX_CHUNK_DIM / MX_SCALE_BLOCK; // 4

// ═══════════════════════════════════════════════════════════════════
// Persistent args
// ═══════════════════════════════════════════════════════════════════
struct PersistentArgs {
    unsigned int* work_counter;
    int tiles_X, tiles_Y, total_tiles;
};


// ═══════════════════════════════════════════════════════════════════
// Helper: get scalar amax from a bf16x2 pair
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ float get_amax_of_bf16x2(const bf16x2& pair) {
    return __bfloat162float(__hmax(__habs(pair.x), __habs(pair.y)));
}


// ═══════════════════════════════════════════════════════════════════
// E8M0 conversion with round-ties-to-even (RN)
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ e8m0_t float_to_e8m0_rn(float val) {
    if (isnan(val)) return 0xFF;
    if (isinf(val)) return 0xFE;
    if (val == 0.0f) return 0x00;
    uint32_t val_u32 = *reinterpret_cast<uint32_t*>(&val);
    e8m0_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
    uint32_t mantissa = val_u32 & 0x7FFFFF;
    constexpr uint32_t half = 1u << (FP32_MANTISSA_BITS - 1);
    bool round_up = (mantissa > half) || (mantissa == half && (exponent & 1));
    if (round_up && exponent < 0xFE) {
        ++exponent;
    }
    return exponent;
}


// ═══════════════════════════════════════════════════════════════════
// Per-row quantize: bf16[128] → fp4x2[64] + 4 E8M0 scales
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ void quantize_row(
    const __nv_bfloat16* __restrict__ smem_in,
    __nv_fp4x2_e2m1* __restrict__ smem_out,
    uint8_t e8m0_out[MX_NUM_BLOCKS],
    const int row
) {
    #pragma unroll
    for (int b = 0; b < MX_NUM_BLOCKS; b++) {
        const int col_start = b * MX_SCALE_BLOCK;

        // Load 32 bf16 from SMEM using 4× 128-bit loads
        __uint128_t d[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t addr = __cvta_generic_to_shared(&smem_in[row * MX_CHUNK_DIM + col_start + i * 8]);
            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(reinterpret_cast<uint32_t*>(&d[i])[0]),
                  "=r"(reinterpret_cast<uint32_t*>(&d[i])[1]),
                  "=r"(reinterpret_cast<uint32_t*>(&d[i])[2]),
                  "=r"(reinterpret_cast<uint32_t*>(&d[i])[3])
                : "r"(addr));
        }

        // Vectorized amax over 32 elements
        bf16x2 amax = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const bf16x2* pairs = reinterpret_cast<const bf16x2*>(&d[i]);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                abs_max_2x(amax, amax, pairs[j]);
            }
        }

        float block_amax = get_amax_of_bf16x2(amax);

        uint8_t e8m0_val;
        if (block_amax <= 1e-9f) {
            e8m0_val = 0;
        } else {
            e8m0_val = float_to_e8m0_rn(block_amax);
        }
        e8m0_out[b] = e8m0_val;

        float scale_rcp = exp2f_rcp(e8m0_val);
        float coeff = 6.0f * scale_rcp;

        // Vectorized FP4 quantize (8 elements at a time)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint64_t* e = reinterpret_cast<const uint64_t*>(&d[i]);
            uint32_t q = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e[0], e[1], coeff);
            uint32_t out_addr = __cvta_generic_to_shared(
                reinterpret_cast<uint8_t*>(smem_out) + row * (MX_CHUNK_DIM / 2) + col_start / 2 + i * 4);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_addr), "r"(q) : "memory");
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Persistent single-phase quantize kernel
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(MX_THREADS)
mxfp4_v2_persistent_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M, const int64_t K,
    PersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int tid = threadIdx.x;
    const int ntk = K / 128;

    // ─── SMEM layout ───
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(MX_CHUNK_DIM * MX_CHUNK_DIM * (int)sizeof(__nv_bfloat16), TMA_SHMEM_ALIGNMENT);

    __nv_bfloat16*   smem_in  = reinterpret_cast<__nv_bfloat16*>(dshmem);
    __nv_fp4x2_e2m1* smem_fp4 = reinterpret_cast<__nv_fp4x2_e2m1*>(dshmem + in_bytes);

    __shared__ uint64_t in_mbar;
    if (leading) {
        mbarrier_init(&in_mbar, 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int mbar_phase = 0;

    // ═══════════════════════════════════════════════════════════════
    // Persistent work-stealing loop
    // ═══════════════════════════════════════════════════════════════
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int gY = ctaid_Y * MX_CHUNK_DIM;
        const int gX = ctaid_X * MX_CHUNK_DIM;

        // ─── TMA load 128×128 bf16 ───
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar, MX_CHUNK_DIM * MX_CHUNK_DIM * sizeof(__nv_bfloat16));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(smem_in),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                gX, gY, &in_mbar);
        }
        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar, mbar_phase);

        // ─── Quantize: each thread handles 1 row ───
        uint8_t e8m0_vals[MX_NUM_BLOCKS];
        quantize_row(smem_in, smem_fp4, e8m0_vals, tid);

        fence_proxy_async_shared_cta();
        __syncthreads();

        // ─── TMA store FP4 output ───
        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                gX / 2, gY,
                reinterpret_cast<uint64_t*>(smem_fp4));
            cp_async_bulk_commit_group();
        }

        // ─── Write E8M0 scales to global (swizzled layout) ───
        {
            const int j = tid % 32;
            const int grp = tid / 32;
            const int base = (ctaid_Y * ntk + ctaid_X) * 512 + j * 16 + grp * 4;
            uint32_t pk;
            uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
            p[0] = e8m0_vals[0]; p[1] = e8m0_vals[1];
            p[2] = e8m0_vals[2]; p[3] = e8m0_vals[3];
            *reinterpret_cast<uint32_t*>(scales_out + base) = pk;
        }

        // Wait for FP4 TMA store before reusing SMEM
        if (leading) cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        mbar_phase ^= 1;
    }

    // Cleanup
    if (leading) mbarrier_invalid(&in_mbar);
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Legacy non-persistent kernel (for group quantize)
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(MX_THREADS)
mxfp4_v2_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M, const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int tid = threadIdx.x;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int gY = ctaid_Y * MX_CHUNK_DIM;
    const int gX = ctaid_X * MX_CHUNK_DIM;

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(MX_CHUNK_DIM * MX_CHUNK_DIM * (int)sizeof(__nv_bfloat16), TMA_SHMEM_ALIGNMENT);

    __nv_bfloat16* smem_in = reinterpret_cast<__nv_bfloat16*>(dshmem);
    __nv_fp4x2_e2m1* smem_fp4 = reinterpret_cast<__nv_fp4x2_e2m1*>(dshmem + in_bytes);

    __shared__ uint64_t in_mbar;
    if (leading) {
        mbarrier_init(&in_mbar, 1);
        fence_proxy_async_shared_cta();
        mbarrier_arrive_expect_tx(&in_mbar, MX_CHUNK_DIM * MX_CHUNK_DIM * sizeof(__nv_bfloat16));
        cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(smem_in),
            reinterpret_cast<const uint64_t*>(&tensor_map_input),
            gX, gY, &in_mbar);
    }
    __syncthreads();
    mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar, 0);

    uint8_t e8m0_vals[MX_NUM_BLOCKS];
    if (tid < MX_CHUNK_DIM) {
        quantize_row(smem_in, smem_fp4, e8m0_vals, tid);
    }
    __syncthreads();

    if (leading) {
        fence_proxy_async_shared_cta();
        cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t*>(&tensor_map_output),
            gX / 2, gY,
            reinterpret_cast<uint64_t*>(smem_fp4));
        cp_async_bulk_commit_group();
    }

    if (tid < MX_CHUNK_DIM) {
        const int j = tid % 32;
        const int grp = tid / 32;
        const int ntk = K / 128;
        const int base = (ctaid_Y * ntk + ctaid_X) * 512 + j * 16 + grp * 4;
        uint32_t pk;
        uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
        p[0] = e8m0_vals[0]; p[1] = e8m0_vals[1];
        p[2] = e8m0_vals[2]; p[3] = e8m0_vals[3];
        *reinterpret_cast<uint32_t*>(scales_out + base) = pk;
    }

    if (leading) {
        cp_async_bulk_wait_group_read<0>();
        mbarrier_invalid(&in_mbar);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// SMEM size helper
// ═══════════════════════════════════════════════════════════════════
inline int v2_shmem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(MX_CHUNK_DIM * MX_CHUNK_DIM * (int)sizeof(__nv_bfloat16), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(MX_CHUNK_DIM * (MX_CHUNK_DIM / 2), TMA_SHMEM_ALIGNMENT);
    return in_bytes + out_bytes + TMA_SHMEM_ALIGNMENT;
}

} // namespace mxfp4_v2

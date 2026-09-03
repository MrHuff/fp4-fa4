/*************************************************************************
 * MXFP4 v3 Quantize — Single-phase pipelined kernel
 *
 * Based on NVFP4 v5 architecture but simplified:
 *   - NO global amax phase (MXFP4 uses per-32-element E8M0 scales)
 *   - Single-phase: TMA load → quantize → TMA store, pipelined
 *   - 4 sub-tiles of 64×64 within each 128×128 chunk
 *   - Double-buffered FP4 output (write one while TMA-storing the other)
 *   - Prefetch tile t+2 while processing tile t
 *   - Persistent work-stealing for large tensors
 *
 * SMEM budget (128×128 chunk):
 *   Input:  4 × 64×64 × 2B = 32 KB  (all sub-tiles)
 *   FP4:    2 × 64×32 × 1B =  4 KB  (double-buffered)
 *   Scales: 128×4 uint8     = 512 B  (E8M0, 4 blocks of 32 per 128)
 *   Total:                  ≈ 37 KB  → ~6 CTAs/SM on GB200 (228KB)
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
#include "../nvfp4_v5/util/math.h"

namespace mxfp4_v3 {

using namespace transformer_engine;
using namespace transformer_engine::ptx;

// ═══════════════════════════════════════════════════════════════════
// Configuration — mirrors NV v5 sub-tile layout
// ═══════════════════════════════════════════════════════════════════

static constexpr int CHUNK_DIM   = 128;
static constexpr int TILE_DIM    = 64;       // 64×64 sub-tiles
static constexpr int TILES_Y     = 2;        // 128/64
static constexpr int TILES_X     = 2;
static constexpr int NUM_TILES   = 4;
static constexpr int THREADS     = 128;
static constexpr int MX_BLOCK    = 32;       // MXFP4 scale block (32 elements)
static constexpr int MX_BLOCKS_PER_CHUNK = CHUNK_DIM / MX_BLOCK;  // 4

// Thread layout for rowwise quantization
static constexpr int ELTS_PER_THREAD = 16;                        // 16 bf16 per thread
static constexpr int THREADS_X   = TILE_DIM / ELTS_PER_THREAD;   // 4
static constexpr int THREADS_Y   = THREADS / THREADS_X;          // 32
static constexpr int ITERATIONS  = TILE_DIM / THREADS_Y;         // 2

static constexpr int PACK_SIZE   = 8;        // 8 elements per vectorised load
static constexpr int WAVES       = ELTS_PER_THREAD / PACK_SIZE;  // 2

// Double-buffered output
static constexpr int BUFFS_OUT   = 2;
static constexpr int OUT_DIM_Y   = TILE_DIM;
static constexpr int OUT_DIM_X   = (TILE_DIM * 4) / 8;  // FP4 packed: 64 elts -> 32 bytes
static constexpr int OUT_SIZE    = OUT_DIM_Y * OUT_DIM_X;

// Scales: 128 rows × 4 blocks (per chunk)
static constexpr int SCALES_PER_CHUNK = CHUNK_DIM / MX_BLOCK;  // 4

// SMEM bank conflict avoidance
static constexpr int TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;
static constexpr int THREADS_PER_BANK  = TOTAL_BANKS_WIDTH / ELTS_PER_THREAD;

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════
using IType  = bf16;
using IType2 = typename ptx::FPx2<IType>;
using InputBuf3D   = IType[NUM_TILES][TILE_DIM][TILE_DIM];
using OutputBuf3D  = fp4e2m1x2[BUFFS_OUT][OUT_DIM_Y][OUT_DIM_X];

// Persistent kernel args
struct PersistentArgs {
    unsigned int* work_counter;
    int tiles_X, tiles_Y, total_tiles;
};

// Group quantize args (max 16 groups)
static constexpr int MAX_GROUPS = 16;
struct GroupArgs {
    int num_groups;
    int boundaries[MAX_GROUPS + 1];
    uint8_t* scale_ptrs[MAX_GROUPS];
};


// ═══════════════════════════════════════════════════════════════════
// Quantization mode: how to round E8M0 scale exponent
// ═══════════════════════════════════════════════════════════════════
enum class QuantMode : int {
    RTE = 0,       // Round-ties-to-even (default)
    ENCODE = 1,    // Ceil exponent: scale >= amax, no clipping
    DECODE = 2,    // Floor exponent: fill range, allow clipping
    UP_1_8 = 3,    // Round up above 1/8 of the exponent bin
    UP_1_4 = 4,    // Round up above 1/4 of the exponent bin
    UP_3_8 = 5     // Round up above 3/8 of the exponent bin
};

// ═══════════════════════════════════════════════════════════════════
// E8M0 scale computation
// ═══════════════════════════════════════════════════════════════════
// Round-ties-to-even (original)
__device__ __forceinline__ uint8_t float_to_e8m0_rte(float val) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    constexpr uint32_t half = 1u << 22;
    bool round_up = (mant > half) || (mant == half && (exp & 1));
    if (round_up && exp < 0xFE) ++exp;
    return exp;
}

// Encode-centric: ceil exponent → 2^exp >= val always
__device__ __forceinline__ uint8_t float_to_e8m0_ceil(float val) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    // If there's any mantissa, round UP so scale >= val
    if (mant > 0 && exp < 0xFE) ++exp;
    return exp;
}

// Decode-centric: floor exponent → 2^exp <= val always
__device__ __forceinline__ uint8_t float_to_e8m0_floor(float val) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    // Just truncate mantissa → always rounds down
    return exp;
}

template<uint32_t MANTISSA_THRESHOLD>
__device__ __forceinline__ uint8_t float_to_e8m0_threshold(float val) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    bool round_up = mant > MANTISSA_THRESHOLD;
    if (round_up && exp < 0xFE) ++exp;
    return exp;
}

// Dispatch helper
template<QuantMode MODE>
__device__ __forceinline__ uint8_t float_to_e8m0(float val) {
    if constexpr (MODE == QuantMode::ENCODE) return float_to_e8m0_ceil(val);
    else if constexpr (MODE == QuantMode::DECODE) return float_to_e8m0_floor(val);
    else if constexpr (MODE == QuantMode::UP_1_8) {
        return float_to_e8m0_threshold<(1u << 20)>(val);
    } else if constexpr (MODE == QuantMode::UP_1_4) {
        return float_to_e8m0_threshold<(1u << 21)>(val);
    } else if constexpr (MODE == QuantMode::UP_3_8) {
        return float_to_e8m0_threshold<(3u << 20)>(val);
    }
    else return float_to_e8m0_rte(val);
}


// ═══════════════════════════════════════════════════════════════════
// Per-row quantize for one sub-tile: 16 bf16 → 8 fp4x2 + scale update
//
// Each thread handles 16 elements in a row (= half of MX_BLOCK=32).
// Scales are keyed by MX block index, so threads in the same block
// cooperate via __shfl to share the block amax.
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void mx_rowwise_quantize(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    uint8_t* __restrict__ scale_buf,     // [CHUNK_DIM × SCALES_PER_CHUNK]
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out
) {
    const auto& sIn = *reinterpret_cast<const InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    const int thread_lane = threadIdx.x % 32;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / THREADS_X;
    const int tid_X = threadIdx.x % THREADS_X;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    // Scale indices: each thread covers 16 elements, MX block = 32
    // So 2 threads per MX block in X direction
    const int mx_block_in_tile = thread_offset_X / MX_BLOCK;
    const int global_scale_x = stage_X * (TILE_DIM / MX_BLOCK) + mx_block_in_tile;

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
        const int row = tid_Y + it * THREADS_Y;
        const int global_row = stage_Y * TILE_DIM + row;

        // Load 16 bf16 from SMEM using vectorized loads (2 waves × 8)
        __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];
        IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e)
                abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
        }

        // Compute per-16-element amax, then reduce within MX block (32 elements)
        float my_amax = fmaxf(
            __bfloat162float(__habs(amax_2x.x)),
            __bfloat162float(__habs(amax_2x.y))
        );

        // Reduce with the other half of the MX block (the other thread covers the other 16 elements)
        // tid_X % 2 tells us which half we are; pair with XOR mask
        float pair_amax = __shfl_xor_sync(0xffffffff, my_amax, 1);
        float block_amax = fmaxf(my_amax, pair_amax);

        // Compute E8M0 scale using selected rounding mode
        uint8_t e8m0_val = float_to_e8m0<MODE>(block_amax);

        // Store scale (only one thread per MX block)
        if ((tid_X % 2) == 0) {
            scale_buf[global_row * SCALES_PER_CHUNK + global_scale_x] = e8m0_val;
        }

        // Compute quantization coefficient: 6.0 / 2^(e8m0 - 127)
        float scale_rcp = exp2f_rcp(e8m0_val);
        float coeff = 6.0f * scale_rcp;

        // Quantize and pack to FP4
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Pipelined quantize-and-store for one chunk (4 sub-tiles)
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void quantize_chunk_pipelined(
    IType* sIn_ptr, fp4e2m1x2* sOut_ptr,
    uint8_t* scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_output,
    int block_offset_Y, int block_offset_X,
    uint64_t* in_mbar, int mbar_phase,
    const CUtensorMap* tensor_map_input_ptr
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    int buff_out = 0;

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int stage_offset_Y = stage_Y * TILE_DIM;
        const int stage_offset_X = stage_X * TILE_DIM;

        // Prefetch tile t+2 if it exists
        if (t + 2 < NUM_TILES) {
            const int next = t + 2;
            const int nty = next / TILES_X, ntx = next % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[next],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[next]),
                    reinterpret_cast<const uint64_t*>(tensor_map_input_ptr),
                    block_offset_X + ntx * TILE_DIM,
                    block_offset_Y + nty * TILE_DIM,
                    &in_mbar[next]);
            }
        }

        // Wait for current tile
        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        // Quantize
        mx_rowwise_quantize<MODE>(sIn_ptr, sOut_ptr, scale_buf,
                           stage_Y, stage_X, t, buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        // TMA store quantized tile
        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_output),
                block_offset_X + stage_offset_X,
                block_offset_Y + stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[buff_out]));
            cp_async_bulk_commit_group();
        }

        buff_out = (buff_out + 1) % BUFFS_OUT;
    }

    // Wait for all FP4 TMA stores
    if (leading) cp_async_bulk_wait_group_read<0>();
    __syncthreads();
}


// ═══════════════════════════════════════════════════════════════════
// Write scales to global memory (swizzled layout for TK GEMM)
// Layout: [M/128, K/128, 32, 16] packed as uint8
//   Scale for row r, block b goes to:
//     base = (r/128 * ntk + chunk_x) * 512 + (r%128)%32 * 16 + (r%128)/32 * 4 + b
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ void write_scales_swizzled(
    const uint8_t* scale_buf,   // [CHUNK_DIM × SCALES_PER_CHUNK]
    uint8_t* __restrict__ global_scales,
    int ctaid_X, int ctaid_Y,
    int ntk
) {
    // Each thread processes one row of scales (4 values)
    for (int row = threadIdx.x; row < CHUNK_DIM; row += THREADS) {
        const int j = row % 32;
        const int grp = row / 32;
        const int base = (ctaid_Y * ntk + ctaid_X) * 512 + j * 16 + grp * 4;
        uint32_t pk;
        uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
        p[0] = scale_buf[row * SCALES_PER_CHUNK + 0];
        p[1] = scale_buf[row * SCALES_PER_CHUNK + 1];
        p[2] = scale_buf[row * SCALES_PER_CHUNK + 2];
        p[3] = scale_buf[row * SCALES_PER_CHUNK + 3];
        *reinterpret_cast<uint32_t*>(global_scales + base) = pk;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Persistent single-phase quantize kernel
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v3_persistent_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M, const int64_t K,
    PersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;

    // ─── SMEM layout ───
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();


    // ═══════════════════════════════════════════════════════════════
    // Persistent work-stealing loop — SINGLE PHASE
    // ═══════════════════════════════════════════════════════════════
    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) s_chunk_id = atomicAdd(args.work_counter, 1);
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) break;

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * CHUNK_DIM;
        const int block_offset_X = ctaid_X * CHUNK_DIM;

        // Reinitialize mbarriers for this chunk (clean state)
        if (leading) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES; ++t) {
                mbarrier_invalid(&in_mbar[t]);
                mbarrier_init(&in_mbar[t], 1);
            }
            fence_proxy_async_shared_cta();
        }
        __syncthreads();

        // Prefetch first 2 tiles
        #pragma unroll
        for (int pre = 0; pre < 2; ++pre) {
            const int ty = pre / TILES_X, tx = pre % TILES_X;
            if (leading) {
                mbarrier_arrive_expect_tx(&in_mbar[pre],
                    TILE_DIM * TILE_DIM * sizeof(IType));
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn[pre]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * TILE_DIM,
                    block_offset_Y + ty * TILE_DIM,
                    &in_mbar[pre]);
            }
        }

        // Pipelined quantize of all 4 sub-tiles (always phase 0)
        quantize_chunk_pipelined<MODE>(
            sIn_ptr, sOut_ptr, scale_buf, sOut,
            tensor_map_output,
            block_offset_Y, block_offset_X,
            in_mbar, 0,
            &tensor_map_input);

        // Write scales to global
        write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

        // Sync before next chunk to ensure all threads finished reading scale_buf
        __syncthreads();
    }

    // Cleanup
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Non-persistent (fused) kernel — one CTA per chunk
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v3_fused_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales_out,
    const int64_t M, const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Prefetch first 2 tiles
    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X, tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    // Pipelined quantize
    quantize_chunk_pipelined<MODE>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        block_offset_Y, block_offset_X,
        in_mbar, 0,
        &tensor_map_input);

    // Write scales
    write_scales_swizzled(scale_buf, scales_out, ctaid_X, ctaid_Y, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Write scales to per-group buffers (grouped quantize)
//
// Determines which group this chunk belongs to from split_range[],
// computes local row offset within that group, then writes scales
// to the group's scale buffer in swizzled layout.
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ void write_scales_grouped(
    const uint8_t* scale_buf,   // [CHUNK_DIM × SCALES_PER_CHUNK]
    const GroupArgs& args,
    int block_offset_Y,         // global row offset of this chunk
    int ctaid_X,
    int ntk                     // K / CHUNK_DIM
) {
    // Determine group ID from row position
    int group_id = 0;
    for (int g = 1; g < args.num_groups; ++g) {
        if (block_offset_Y >= args.boundaries[g]) group_id = g;
    }
    const int group_start = args.boundaries[group_id];
    uint8_t* __restrict__ group_scales = args.scale_ptrs[group_id];

    // Local ctaid_Y within this group
    const int local_ctaid_Y = (block_offset_Y - group_start) / CHUNK_DIM;

    for (int row = threadIdx.x; row < CHUNK_DIM; row += THREADS) {
        const int j = row % 32;
        const int grp = row / 32;
        const int base = (local_ctaid_Y * ntk + ctaid_X) * 512 + j * 16 + grp * 4;
        uint32_t pk;
        uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
        p[0] = scale_buf[row * SCALES_PER_CHUNK + 0];
        p[1] = scale_buf[row * SCALES_PER_CHUNK + 1];
        p[2] = scale_buf[row * SCALES_PER_CHUNK + 2];
        p[3] = scale_buf[row * SCALES_PER_CHUNK + 3];
        *reinterpret_cast<uint32_t*>(group_scales + base) = pk;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Grouped fused kernel — single kernel launch for all groups
//
// Identical to mxfp4_v3_fused_kernel but uses GroupArgs to write
// scales to per-group buffers. FP4 output is written to a single
// contiguous buffer (no per-group FP4 needed for MXFP4).
// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v3_fused_group_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const int64_t M, const int64_t K,
    GroupArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int ntk = K / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*      sIn_ptr  = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*  sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t*    scale_buf = dshmem + in_bytes + out_bytes;

    auto& sIn  = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_init(&in_mbar[t], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Prefetch first 2 tiles
    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        const int ty = pre / TILES_X, tx = pre % TILES_X;
        if (leading) {
            mbarrier_arrive_expect_tx(&in_mbar[pre],
                TILE_DIM * TILE_DIM * sizeof(IType));
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&sIn[pre]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_X + tx * TILE_DIM,
                block_offset_Y + ty * TILE_DIM,
                &in_mbar[pre]);
        }
    }

    // Pipelined quantize
    quantize_chunk_pipelined<MODE>(
        sIn_ptr, sOut_ptr, scale_buf, sOut,
        tensor_map_output,
        block_offset_Y, block_offset_X,
        in_mbar, 0,
        &tensor_map_input);

    // Write scales to per-group buffers
    write_scales_grouped(scale_buf, args, block_offset_Y, ctaid_X, ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t)
            mbarrier_invalid(&in_mbar[t]);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// SMEM size helper
// ═══════════════════════════════════════════════════════════════════
inline int v3_shmem_size() {
    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);
    return in_bytes + out_bytes + sc_bytes + TMA_SHMEM_ALIGNMENT;
}

} // namespace mxfp4_v3

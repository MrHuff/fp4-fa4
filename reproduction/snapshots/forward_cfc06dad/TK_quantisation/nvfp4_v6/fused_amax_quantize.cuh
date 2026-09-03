/*************************************************************************
 * v3: True Single-Pass Fused Amax+Quantize Kernel
 *
 * Architecture:
 *   1. TMA-load ALL tiles of this CTA's chunk into SMEM (4 buffers)
 *   2. Scan each tile: compute per-16-element block amaxes, track CTA max
 *   3. atomicMax to global + spin-wait barrier (all CTAs)
 *   4. S_enc is now known. Data is STILL IN SMEM.
 *   5. Quantize from SMEM (rowwise + colwise) → TMA store
 *
 * ONE HBM read per element. No cooperative groups needed.
 *
 * SMEM budget (128×128 chunk, 4 tiles of 64×64):
 *   Input:  4 × 64×64 × 2B = 32 KB
 *   Output: 2 × 64×32 × 1B =  4 KB (double-buffered FP4)
 *   Out_tr: 2 × 64×16 × 1B =  2 KB (double-buffered transpose FP4)
 *   Scales: 128×8 + 128×8  =  2 KB
 *   Misc:                  ≈  4 KB
 *   Total:                 ≈ 44 KB  →  ~5 CTAs/SM on GB200 (228KB)
 *
 * Target: SM100 (GB200), compiled with -arch sm_100a
 *************************************************************************/

#ifndef TK_V3_FUSED_AMAX_QUANTIZE_CUH_
#define TK_V3_FUSED_AMAX_QUANTIZE_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include "util/cast_common.h"
#include "util/math.h"
#include "util/ptx.cuh"
#include "util/utils.cuh"
#include "core.cuh"

namespace tk_v3 {

using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel;
using namespace transformer_engine::dispatch::nvfp4::core;
using namespace transformer_engine;
using namespace transformer_engine::ptx;

#if FP4_TYPE_SUPPORTED

// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════

struct V3Config {
    static constexpr int CHUNK_DIM_Y = 128;
    static constexpr int CHUNK_DIM_X = 128;
};

static constexpr int V3_SCALE_DIM = 16;
static constexpr int V3_THREADS = 128;
static constexpr int V3_ELTS_PER_THREAD = 16;
static constexpr int V3_TILE_DIM_Y = 64;
static constexpr int V3_TILE_DIM_X = 64;

static constexpr int V3_TILES_Y = V3Config::CHUNK_DIM_Y / V3_TILE_DIM_Y;  // 2
static constexpr int V3_TILES_X = V3Config::CHUNK_DIM_X / V3_TILE_DIM_X;  // 2
static constexpr int V3_NUM_TILES = V3_TILES_Y * V3_TILES_X;              // 4

// ALL 4 input tiles kept in SMEM simultaneously
static constexpr int V3_BUFFS_NUM_IN = V3_NUM_TILES;  // 4

// Output: double-buffered (write one, TMA-store the other)
static constexpr int V3_BUFFS_NUM_OUT = 2;
static constexpr int V3_BUFFS_NUM_OUT_TR = 2;

static constexpr int V3_BUFF_DIM_Y = V3_TILE_DIM_Y;
static constexpr int V3_BUFF_DIM_X = V3_TILE_DIM_X;
static constexpr int V3_BUFF_IN_ELEMS = V3_BUFF_DIM_Y * V3_BUFF_DIM_X;

static constexpr int V3_BUFF_OUT_DIM_Y = V3_BUFF_DIM_Y;
static constexpr int V3_BUFF_OUT_DIM_X = (V3_BUFF_DIM_X * 4) / 8;
static constexpr int V3_BUFF_OUT_SIZE = V3_BUFF_OUT_DIM_Y * V3_BUFF_OUT_DIM_X;

static constexpr int V3_BUFF_OUT_TR_DIM_Y = V3_BUFF_DIM_X;
static constexpr int V3_BUFF_OUT_TR_DIM_X = (V3_BUFF_DIM_Y * 4) / 8;
static constexpr int V3_BUFF_OUT_TR_SIZE = V3_BUFF_OUT_TR_DIM_Y * V3_BUFF_OUT_TR_DIM_X;

static constexpr int V3_SCALES_PER_CHUNK_Y = V3Config::CHUNK_DIM_Y / V3_SCALE_DIM;
static constexpr int V3_SCALES_PER_CHUNK_X = V3Config::CHUNK_DIM_X / V3_SCALE_DIM;
static constexpr int V3_SCALES_PER_TILE_Y = V3_TILE_DIM_Y / V3_SCALE_DIM;
static constexpr int V3_SCALES_PER_TILE_X = V3_TILE_DIM_X / V3_SCALE_DIM;

static constexpr int V3_PACK_SIZE = 8;
static constexpr int V3_WAVES = V3_ELTS_PER_THREAD / V3_PACK_SIZE;

static constexpr int V3_THREADS_X_ROWWISE = V3_TILE_DIM_X / V3_ELTS_PER_THREAD;
static constexpr int V3_THREADS_Y_ROWWISE = V3_THREADS / V3_THREADS_X_ROWWISE;
static constexpr int V3_ITERATIONS_NORMAL = V3_BUFF_DIM_Y / V3_THREADS_Y_ROWWISE;

static constexpr int V3_THREADS_PER_SCALE_ROWWISE = V3_SCALE_DIM / V3_ELTS_PER_THREAD;

static constexpr int V3_TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;
static constexpr int V3_THREADS_PER_BANK = V3_TOTAL_BANKS_WIDTH / V3_ELTS_PER_THREAD;

// Type aliases matching v2
using IType = bf16;
using IType2 = typename ptx::FPx2<IType>;
using V3_IType3D = IType[V3_BUFFS_NUM_IN][V3_BUFF_DIM_Y][V3_BUFF_DIM_X];
using V3_IType2x3D = IType2[V3_BUFFS_NUM_IN][V3_BUFF_DIM_Y][V3_BUFF_DIM_X / 2];
using V3_OType2x3D = fp4e2m1x2[V3_BUFFS_NUM_OUT][V3_BUFF_OUT_DIM_Y][V3_BUFF_OUT_DIM_X];
using V3_OType2xt3D = fp4e2m1x2[V3_BUFFS_NUM_OUT_TR][V3_BUFF_OUT_TR_DIM_Y][V3_BUFF_OUT_TR_DIM_X];
using V3_ScalesType2D = nvfp4_scale_t[V3Config::CHUNK_DIM_Y][V3_SCALES_PER_CHUNK_X];
using V3_ScalesTypeTr2D = nvfp4_scale_t[V3Config::CHUNK_DIM_X][V3_SCALES_PER_CHUNK_Y];


// ═══════════════════════════════════════════════════════════════════
// Spin-wait barrier using global atomics
// ═══════════════════════════════════════════════════════════════════

// global_amax:       float*, must be zeroed before launch
// done_counter:      unsigned int*, must be zeroed before launch
// ready_flag:        unsigned int*, must be zeroed before launch
// total_blocks:      total grid blocks
//
// Flow:
//   1. Each CTA: atomicMax(global_amax, cta_max)
//   2. Each CTA: __threadfence() to ensure visibility
//   3. Thread 0: atomicAdd(done_counter, 1)
//   4. Thread 0: if we're the last block, set ready_flag = 1
//   5. All threads: spin on ready_flag until it's 1
//   6. __syncthreads() to ensure all threads see the flag

__device__ __forceinline__
void atomic_max_float(float* addr, float val) {
    unsigned int* p = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *p;
    unsigned int want = __float_as_uint(val);
    while (want > old) {
        old = atomicCAS(p, old, want);
    }
}

__device__ __forceinline__
void grid_barrier(float cta_max,
                  float* __restrict__ global_amax,
                  unsigned int* __restrict__ done_counter,
                  unsigned int* __restrict__ ready_flag,
                  int total_blocks) {
    // Step 1: contribute this CTA's max
    if (cta_max > 0.0f) {
        atomic_max_float(global_amax, cta_max);
    }

    // Step 2: ensure the atomicMax is globally visible
    __threadfence();

    // Step 3: count completed CTAs, last one signals
    __shared__ unsigned int s_ready;
    if (threadIdx.x == 0) {
        unsigned int prev = atomicAdd(done_counter, 1);
        if (prev == (unsigned int)(total_blocks - 1)) {
            // We're the last block — all amaxes are committed
            __threadfence();  // ensure all atomicMax writes are visible
            // Signal all blocks (use volatile store)
            volatile unsigned int* vflag = (volatile unsigned int*)ready_flag;
            *vflag = 1;
        }
        s_ready = 0;
    }
    __syncthreads();

    // Step 4: spin-wait for ready flag using volatile read
    if (threadIdx.x == 0) {
        volatile unsigned int* vflag = (volatile unsigned int*)ready_flag;
        while (*vflag == 0) {
            // spin — volatile ensures the read isn't cached in registers
        }
        s_ready = 1;
    }
    __syncthreads();
    // All threads now know global_amax is finalized
}


// ═══════════════════════════════════════════════════════════════════
// Phase 2 helpers: rowwise + colwise quantization from SMEM
// (Same logic as v2, but reads from the persistent SMEM buffers)
// ═══════════════════════════════════════════════════════════════════

template <bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void v3_rowwise_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;  // USE_FAST_MATH = false

    const auto& sIn = *reinterpret_cast<const V3_IType3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sSFrowwise = *reinterpret_cast<V3_ScalesType2D*>(sSFrowwise_ptr);

    const int thread_lane = threadIdx.x % THREADS_PER_WARP;
    const int bank_group = thread_lane / V3_THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / V3_THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % V3_THREADS_X_ROWWISE;
    const int thread_offset_X = tid_X * V3_ELTS_PER_THREAD;

    const int SF_tid_Y = tid_Y;
    const int SF_tid_X = tid_X / V3_THREADS_PER_SCALE_ROWWISE;
    const bool SF_storing = (tid_X % V3_THREADS_PER_SCALE_ROWWISE == 0);
    const int stage_sc_Y = SF_tid_Y + stage_Y * V3_TILE_DIM_Y;
    const int stage_sc_X = SF_tid_X + stage_X * V3_SCALES_PER_TILE_X;

    #pragma unroll
    for (int it = 0; it < V3_ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * V3_THREADS_Y_ROWWISE;

        __align__(16) IType2 rIn[V3_WAVES][V3_PACK_SIZE / 2];
        IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < V3_WAVES; ++w) {
            const int sw = ((w + bank_group) * V3_PACK_SIZE) % V3_ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ptx::ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            #pragma unroll
            for (int e = 0; e < V3_PACK_SIZE / 2; ++e)
                ptx::abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
        }

        const float block_amax = get_amax_of_pair(amax_2x);

        float coeff;
        nvfp4_scale_t S_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const nvfp4_scale_t S_mult_fp8 = compute_encoding_scaling_factor_nv(block_amax, S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }

        if (SF_storing) {
            sSFrowwise[stage_sc_Y + it * V3_THREADS_Y_ROWWISE][stage_sc_X] = S_b_fp8;
        }

        #pragma unroll
        for (int w = 0; w < V3_WAVES; ++w) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            const int sw = ((w + bank_group) * V3_PACK_SIZE) % V3_ELTS_PER_THREAD;
            ptx::st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}

template <bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void v3_colwise_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out_tr
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    const auto& sIn2x = *reinterpret_cast<const V3_IType2x3D*>(sIn_ptr);
    auto& sOut_tr = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);
    auto& sSFcolwise = *reinterpret_cast<V3_ScalesTypeTr2D*>(sSFcolwise_ptr);

    const int warp = threadIdx.x / THREADS_PER_WARP;
    const int lane = threadIdx.x % THREADS_PER_WARP;

    const int tid_Y = (lane % 4 + warp) % 4;
    const int tid_X = lane;
    const int off_Y = tid_Y * V3_SCALE_DIM;
    const int off_X = tid_X * 2;
    const int in_Y = off_Y, in_X = off_X / 2;
    const int out_tr_Y = off_X, out_tr_X = off_Y / 2;
    const int sc_tr_Y = (stage_X * V3_TILE_DIM_X) + 2 * tid_X;
    const int sc_tr_X = (stage_Y * V3_SCALES_PER_TILE_Y) + tid_Y;

    __align__(8) IType rIn[2][V3_SCALE_DIM];
    IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    #pragma unroll
    for (int i = 0; i < V3_SCALE_DIM; ++i) {
        const IType2 pair = ptx::ld_shared_b32(&sIn2x[buff_in][in_Y + i][in_X]);
        rIn[0][i] = pair.x;
        rIn[1][i] = pair.y;
        ptx::abs_max_2x(amax_2x, amax_2x, pair);
    }

    const float bmax[2] = {
        __bfloat162float(__habs(amax_2x.x)),
        __bfloat162float(__habs(amax_2x.y))
    };

    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        float coeff;
        nvfp4_scale_t S_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const nvfp4_scale_t S_mult_fp8 = compute_encoding_scaling_factor_nv(bmax[w], S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(bmax[w], S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }
        sSFcolwise[sc_tr_Y + w][sc_tr_X] = S_b_fp8;

        __align__(8) uint32_t rOut[V3_SCALE_DIM / 8];
        #pragma unroll
        for (int e = 0; e < V3_SCALE_DIM / 8; ++e) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e + 4]);
            rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
        }
        ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_Y + w][out_tr_X],
                           *reinterpret_cast<uint64_t*>(rOut));
    }
}


// ═══════════════════════════════════════════════════════════════════
// Amax scan helper: scan one tile in SMEM for max(|x|)
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__
float scan_tile_amax(const IType* __restrict__ sIn_ptr, int buff_in) {
    const auto& sIn = *reinterpret_cast<const V3_IType3D*>(sIn_ptr);
    const int lane = threadIdx.x % THREADS_PER_WARP;
    const int bank_group = lane / V3_THREADS_PER_BANK;
    const int tid_Y = threadIdx.x / V3_THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % V3_THREADS_X_ROWWISE;
    const int off_X = tid_X * V3_ELTS_PER_THREAD;
    IType2 tile_max_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    #pragma unroll
    for (int it = 0; it < V3_ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * V3_THREADS_Y_ROWWISE;
        #pragma unroll
        for (int w = 0; w < V3_WAVES; ++w) {
            const int sw = ((w + bank_group) * V3_PACK_SIZE) % V3_ELTS_PER_THREAD;
            __uint128_t elts = ptx::ld_shared_b128(&sIn[buff_in][row][off_X + sw]);
            const IType2* pairs = reinterpret_cast<const IType2*>(&elts);
            #pragma unroll
            for (int e = 0; e < V3_PACK_SIZE / 2; ++e)
                ptx::abs_max_2x(tile_max_2x, tile_max_2x, pairs[e]);
        }
    }
    return get_amax_of_pair(tile_max_2x);
}


// ═══════════════════════════════════════════════════════════════════
// Main kernel: true single-pass fused amax+quantize
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(V3_THREADS)
fused_amax_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    nvfp4_scale_t* const scales_ptr,
    nvfp4_scale_t* const scales_t_ptr,
    float* __restrict__ global_amax,    // must be zeroed
    float* __restrict__ sg_out,
    unsigned int* __restrict__ done_counter, // must be zeroed
    unsigned int* __restrict__ ready_flag,   // must be zeroed
    const size_t rows, const size_t cols,
    const size_t scale_stride, const size_t scale_stride_t,
    const int total_blocks
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);

    // ─── SMEM layout ───
    // [input buffers: 4 tiles]  [output buffers: 2]  [output_tr: 2]  [scales_row]  [scales_col]
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

    // ─── This CTA's chunk ───
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
    const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

    // ─── TMA barriers: one per input tile ───
    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_init(&in_mbar[t], 1);
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // ═════════════════════════════════════════════════════
    // PHASE 1: Load ALL tiles + scan for amax
    // ═════════════════════════════════════════════════════
    float cta_max = 0.0f;

    // Launch TMA for all 4 tiles
    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int ty = t / V3_TILES_X;
        const int tx = t % V3_TILES_X;
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

    // Wait for each tile and scan it
    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], 0);

        float tile_max = scan_tile_amax(sIn_ptr, t);
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
    // PHASE 2: Quantize — data is STILL IN SMEM
    // ═════════════════════════════════════════════════════
    const float amax_val = *global_amax;
    const float S_enc = compute_global_encode_scaling_factor_FP4(amax_val);

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        *sg_out = amax_val / 2688.0f;
        // Reset ready_flag for next invocation
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

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        // Wait for any prior TMA store to finish reading from SMEM
        if (t > 0) {
            ptx::cp_async_bulk_wait_group_read<1>();
        }

        // Quantize: data for tile t is in sIn[t] — still there from Phase 1!
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
        for (int t = 0; t < V3_NUM_TILES; ++t)
            ptx::mbarrier_invalid(&in_mbar[t]);
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

template <bool RETURN_TRANSPOSE>
inline int v3_shmem_size() {
    constexpr int in_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(
        V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;

    return in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes + TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED
}  // namespace tk_v3

#endif  // TK_V3_FUSED_AMAX_QUANTIZE_CUH_

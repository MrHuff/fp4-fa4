/*************************************************************************
 * CTA-local single-pass NVFP4 quantization.
 *
 * This is a fork of the v5 fused amax+quantize kernel with the tensor-wide
 * reduction removed. Each CTA quantizes one 128x128 chunk using its own
 * chunk-local amax and writes one decode scale per chunk.
 *************************************************************************/

#ifndef TK_LOCALCTA_FUSED_QUANTIZE_CUH_
#define TK_LOCALCTA_FUSED_QUANTIZE_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include "../nvfp4_v5/util/cast_common.h"
#include "../nvfp4_v5/util/math.h"
#include "../nvfp4_v5/util/ptx.cuh"
#include "../nvfp4_v5/util/utils.cuh"
#include "../nvfp4_v5/core.cuh"

namespace tk_localcta {

using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::dispatch::nvfp4::core;
using namespace transformer_engine;
using namespace transformer_engine::ptx;

#if FP4_TYPE_SUPPORTED

struct LocalCTAConfig {
    static constexpr int CHUNK_DIM_Y = 128;
    static constexpr int CHUNK_DIM_X = 128;
};

static constexpr float LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM = 1493.0f;
static __device__ __constant__ float kLocalCTAGlobalScaleNum = LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM;

static constexpr int SCALE_DIM = 16;
static constexpr int THREADS = 128;
static constexpr int ELTS_PER_THREAD = 16;
static constexpr int TILE_DIM_Y = 64;
static constexpr int TILE_DIM_X = 64;

static constexpr int TILES_Y = LocalCTAConfig::CHUNK_DIM_Y / TILE_DIM_Y;
static constexpr int TILES_X = LocalCTAConfig::CHUNK_DIM_X / TILE_DIM_X;
static constexpr int NUM_TILES = TILES_Y * TILES_X;

static constexpr int BUFFS_NUM_IN = NUM_TILES;
static constexpr int BUFFS_NUM_OUT = 2;
static constexpr int BUFFS_NUM_OUT_TR = 2;

static constexpr int BUFF_DIM_Y = TILE_DIM_Y;
static constexpr int BUFF_DIM_X = TILE_DIM_X;
static constexpr int BUFF_IN_ELEMS = BUFF_DIM_Y * BUFF_DIM_X;

static constexpr int BUFF_OUT_DIM_Y = BUFF_DIM_Y;
static constexpr int BUFF_OUT_DIM_X = (BUFF_DIM_X * 4) / 8;
static constexpr int BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

static constexpr int BUFF_OUT_TR_DIM_Y = BUFF_DIM_X;
static constexpr int BUFF_OUT_TR_DIM_X = (BUFF_DIM_Y * 4) / 8;
static constexpr int BUFF_OUT_TR_SIZE = BUFF_OUT_TR_DIM_Y * BUFF_OUT_TR_DIM_X;

static constexpr int SCALES_PER_CHUNK_Y = LocalCTAConfig::CHUNK_DIM_Y / SCALE_DIM;
static constexpr int SCALES_PER_CHUNK_X = LocalCTAConfig::CHUNK_DIM_X / SCALE_DIM;
static constexpr int SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;
static constexpr int SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;

static constexpr int PACK_SIZE = 8;
static constexpr int WAVES = ELTS_PER_THREAD / PACK_SIZE;

static constexpr int THREADS_X_ROWWISE = TILE_DIM_X / ELTS_PER_THREAD;
static constexpr int THREADS_Y_ROWWISE = THREADS / THREADS_X_ROWWISE;
static constexpr int ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;

static constexpr int THREADS_PER_SCALE_ROWWISE = SCALE_DIM / ELTS_PER_THREAD;

static constexpr int TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;
static constexpr int THREADS_PER_BANK = TOTAL_BANKS_WIDTH / ELTS_PER_THREAD;

using IType = bf16;
using IType2 = typename ptx::FPx2<IType>;
using IType3D = IType[BUFFS_NUM_IN][BUFF_DIM_Y][BUFF_DIM_X];
template <int PIPE_DEPTH>
using ITypeRing4D = IType[PIPE_DEPTH][BUFFS_NUM_IN][BUFF_DIM_Y][BUFF_DIM_X];
using IType2x3D = IType2[BUFFS_NUM_IN][BUFF_DIM_Y][BUFF_DIM_X / 2];
using OType2x3D = fp4e2m1x2[BUFFS_NUM_OUT][BUFF_OUT_DIM_Y][BUFF_OUT_DIM_X];
using OType2xt3D = fp4e2m1x2[BUFFS_NUM_OUT_TR][BUFF_OUT_TR_DIM_Y][BUFF_OUT_TR_DIM_X];
using ScalesType2D = nvfp4_scale_t[LocalCTAConfig::CHUNK_DIM_Y][SCALES_PER_CHUNK_X];
using ScalesTypeTr2D = nvfp4_scale_t[LocalCTAConfig::CHUNK_DIM_X][SCALES_PER_CHUNK_Y];

struct LocalCTAPersistentArgs {
    unsigned int* work_counter;
    int tiles_X;
    int total_tiles;
};

struct LocalCTA2ClusterArgs {
    unsigned int* work_counter;
    int tiles_X;
    int tiles_Y;
    int total_macro_tiles;
};

__device__ __forceinline__ void cluster_sync_aligned() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");
}

template <typename T>
__device__ __forceinline__ uint32_t map_shared_cluster_addr(const T* ptr, int dst_cta) {
    uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(remote_addr)
                 : "r"(local_addr), "r"(dst_cta));
    return remote_addr;
}

__device__ __forceinline__ float cluster_load_shared_f32(const float* ptr, int src_cta) {
    const uint32_t remote_addr = map_shared_cluster_addr(ptr, src_cta);
    uint32_t bits;
    asm volatile("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(bits) : "r"(remote_addr));
    return *reinterpret_cast<const float*>(&bits);
}

__device__ __forceinline__ float localcta_global_scale_num() {
    return kLocalCTAGlobalScaleNum;
}

__device__ __forceinline__ float compute_localcta_encode_scaling_factor_FP4(const float local_amax) {
    float local_encode_scale = localcta_global_scale_num() / local_amax;
    local_encode_scale = fminf(local_encode_scale, transformer_engine::detail::TypeExtrema<float>::max);
    if (local_amax == 0.0f || local_encode_scale == 0.0f) {
        return 1.0f;
    }
    return local_encode_scale;
}

__device__ __forceinline__ uint32_t cluster_load_shared_u32(const uint32_t* ptr, int src_cta) {
    const uint32_t remote_addr = map_shared_cluster_addr(ptr, src_cta);
    uint32_t value;
    asm volatile("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(value) : "r"(remote_addr));
    return value;
}

template <int GROUP_THREADS, uint32_t BARRIER_ID = 1u>
__device__ __forceinline__ void subgroup_barrier_sync() {
    if (threadIdx.x < GROUP_THREADS) {
        ptx::numbered_barrier_sync(GROUP_THREADS, BARRIER_ID);
    }
}

template <int GROUP_THREADS>
__device__ __forceinline__ void swizzle_scales_row_inplace_group(
    nvfp4_scale_t* sSFrowwise_ptr, int num_scales_x, int tid
) {
    uint8_t my_scales[SCALES_PER_CHUNK_X];

    for (int row = tid; row < (int)LocalCTAConfig::CHUNK_DIM_Y; row += GROUP_THREADS) {
        const int j = row % 32;
        const int grp = row / 32;

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFrowwise_ptr[row * SCALES_PER_CHUNK_X + k]);
            }
        }

        subgroup_barrier_sync<GROUP_THREADS>();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFrowwise_ptr)[dest] = my_scales[k];
            }
        }

        subgroup_barrier_sync<GROUP_THREADS>();
    }
}

template <int GROUP_THREADS>
__device__ __forceinline__ void swizzle_scales_col_inplace_group(
    nvfp4_scale_t* sSFcolwise_ptr, int num_scales_y, int tid
) {
    uint8_t my_scales[SCALES_PER_CHUNK_Y];

    for (int col = tid; col < (int)LocalCTAConfig::CHUNK_DIM_X; col += GROUP_THREADS) {
        const int j = col % 32;
        const int grp = col / 32;

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFcolwise_ptr[col * SCALES_PER_CHUNK_Y + k]);
            }
        }

        subgroup_barrier_sync<GROUP_THREADS>();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFcolwise_ptr)[dest] = my_scales[k];
            }
        }

        subgroup_barrier_sync<GROUP_THREADS>();
    }
}

template <int GROUP_THREADS>
__device__ __forceinline__ void scale_swizzled_scales_inplace_group(
    nvfp4_scale_t* scales_ptr,
    int num_elements,
    float global_scale,
    int tid
) {
    for (int idx = tid; idx < num_elements; idx += GROUP_THREADS) {
        const float v = static_cast<float>(scales_ptr[idx]) * global_scale;
        scales_ptr[idx] = static_cast<nvfp4_scale_t>(v);
    }
}

__device__ __forceinline__ void swizzle_scales_row_inplace(
    nvfp4_scale_t* sSFrowwise_ptr, int num_scales_x
) {
    uint8_t my_scales[SCALES_PER_CHUNK_X];

    for (int row = threadIdx.x; row < (int)LocalCTAConfig::CHUNK_DIM_Y; row += THREADS) {
        const int j = row % 32;
        const int grp = row / 32;

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFrowwise_ptr[row * SCALES_PER_CHUNK_X + k]);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_X; ++k) {
            if (k < num_scales_x) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFrowwise_ptr)[dest] = my_scales[k];
            }
        }

        __syncthreads();
    }
}

__device__ __forceinline__ void swizzle_scales_col_inplace(
    nvfp4_scale_t* sSFcolwise_ptr, int num_scales_y
) {
    uint8_t my_scales[SCALES_PER_CHUNK_Y];

    for (int col = threadIdx.x; col < (int)LocalCTAConfig::CHUNK_DIM_X; col += THREADS) {
        const int j = col % 32;
        const int grp = col / 32;

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                my_scales[k] = reinterpret_cast<const uint8_t&>(
                    sSFcolwise_ptr[col * SCALES_PER_CHUNK_Y + k]);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SCALES_PER_CHUNK_Y; ++k) {
            if (k < num_scales_y) {
                const int koffset = k / 4;
                const int k_byte = k % 4;
                const int dest = koffset * 512 + j * 16 + grp * 4 + k_byte;
                reinterpret_cast<uint8_t*>(sSFcolwise_ptr)[dest] = my_scales[k];
            }
        }

        __syncthreads();
    }
}

__device__ __forceinline__ void tma_store_scales_2x512(
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

__device__ __forceinline__ void scale_swizzled_scales_inplace(
    nvfp4_scale_t* scales_ptr,
    int num_elements,
    float global_scale
) {
    for (int idx = threadIdx.x; idx < num_elements; idx += THREADS) {
        const float v = static_cast<float>(scales_ptr[idx]) * global_scale;
        scales_ptr[idx] = static_cast<nvfp4_scale_t>(v);
    }
}

__device__ __forceinline__ float get_amax_of_pair(const IType2 pair) {
    const float ax = __bfloat162float(__habs(pair.x));
    const float ay = __bfloat162float(__habs(pair.y));
    return fmaxf(ax, ay);
}

template <bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void rowwise_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);

    const int thread_lane = threadIdx.x % THREADS_PER_WARP;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = threadIdx.x / THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % THREADS_X_ROWWISE;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    const int SF_tid_Y = tid_Y;
    const int SF_tid_X = tid_X / THREADS_PER_SCALE_ROWWISE;
    const bool SF_storing = (tid_X % THREADS_PER_SCALE_ROWWISE == 0);
    const int stage_sc_Y = SF_tid_Y + stage_Y * TILE_DIM_Y;
    const int stage_sc_X = SF_tid_X + stage_X * SCALES_PER_TILE_X;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;

        __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];
        IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ptx::ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                ptx::abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
            }
        }

        const float block_amax = get_amax_of_pair(amax_2x);

        float coeff;
        nvfp4_scale_t S_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const nvfp4_scale_t S_mult_fp8 =
                compute_encoding_scaling_factor_nv(block_amax, S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }

        if (SF_storing) {
            sSFrowwise[stage_sc_Y + it * THREADS_Y_ROWWISE][stage_sc_X] = S_b_fp8;
        }

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            ptx::st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}

template <bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void colwise_scaling(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out_tr
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    const auto& sIn2x = *reinterpret_cast<const IType2x3D*>(sIn_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);

    const int warp = threadIdx.x / THREADS_PER_WARP;
    const int lane = threadIdx.x % THREADS_PER_WARP;

    const int tid_Y = (lane % 4 + warp) % 4;
    const int tid_X = lane;
    const int off_Y = tid_Y * SCALE_DIM;
    const int off_X = tid_X * 2;
    const int in_Y = off_Y, in_X = off_X / 2;
    const int out_tr_Y = off_X, out_tr_X = off_Y / 2;
    const int sc_tr_Y = (stage_X * TILE_DIM_X) + 2 * tid_X;
    const int sc_tr_X = (stage_Y * SCALES_PER_TILE_Y) + tid_Y;

    __align__(8) IType rIn[2][SCALE_DIM];
    IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    #pragma unroll
    for (int i = 0; i < SCALE_DIM; ++i) {
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
            const nvfp4_scale_t S_mult_fp8 =
                compute_encoding_scaling_factor_nv(bmax[w], S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(bmax[w], S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }
        sSFcolwise[sc_tr_Y + w][sc_tr_X] = S_b_fp8;

        __align__(8) uint32_t rOut[SCALE_DIM / 8];
        #pragma unroll
        for (int e = 0; e < SCALE_DIM / 8; ++e) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e + 4]);
            rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
        }
        ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_Y + w][out_tr_X],
                           *reinterpret_cast<uint64_t*>(rOut));
    }
}

template <int GROUP_THREADS, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void rowwise_scaling_group(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out,
    int tid
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    static_assert(GROUP_THREADS == 128, "rowwise_scaling_group currently expects 128 consumer threads");

    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);

    const int thread_lane = tid % THREADS_PER_WARP;
    const int bank_group = thread_lane / THREADS_PER_BANK;

    const int tid_Y = tid / THREADS_X_ROWWISE;
    const int tid_X = tid % THREADS_X_ROWWISE;
    const int thread_offset_X = tid_X * ELTS_PER_THREAD;

    const int SF_tid_Y = tid_Y;
    const int SF_tid_X = tid_X / THREADS_PER_SCALE_ROWWISE;
    const bool SF_storing = (tid_X % THREADS_PER_SCALE_ROWWISE == 0);
    const int stage_sc_Y = SF_tid_Y + stage_Y * TILE_DIM_Y;
    const int stage_sc_X = SF_tid_X + stage_X * SCALES_PER_TILE_X;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;

        __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];
        IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t& elts = *reinterpret_cast<__uint128_t*>(&rIn[w]);
            elts = ptx::ld_shared_b128(&sIn[buff_in][row][thread_offset_X + sw]);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                ptx::abs_max_2x(amax_2x, amax_2x, rIn[w][e]);
            }
        }

        const float block_amax = get_amax_of_pair(amax_2x);

        float coeff;
        nvfp4_scale_t S_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const nvfp4_scale_t S_mult_fp8 =
                compute_encoding_scaling_factor_nv(block_amax, S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }

        if (SF_storing) {
            sSFrowwise[stage_sc_Y + it * THREADS_Y_ROWWISE][stage_sc_X] = S_b_fp8;
        }

        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][2]);
            uint32_t out = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            ptx::st_shared_b32(&sOut[buff_out][row][(sw + thread_offset_X) / 2], out);
        }
    }
}

template <int GROUP_THREADS, bool ENCODE_CENTRIC = true>
__device__ __forceinline__ void colwise_scaling_group(
    const IType* __restrict__ sIn_ptr,
    fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float S_enc,
    const int stage_Y, const int stage_X,
    const int buff_in, const int buff_out_tr,
    int tid
) {
    using namespace quantization_and_transposition_SF;
    using scaling_coeff_type = float;

    static_assert(GROUP_THREADS == 128, "colwise_scaling_group currently expects 128 consumer threads");

    const auto& sIn2x = *reinterpret_cast<const IType2x3D*>(sIn_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);

    const int warp = tid / THREADS_PER_WARP;
    const int lane = tid % THREADS_PER_WARP;

    const int tid_Y = (lane % 4 + warp) % 4;
    const int tid_X = lane;
    const int off_Y = tid_Y * SCALE_DIM;
    const int off_X = tid_X * 2;
    const int in_Y = off_Y, in_X = off_X / 2;
    const int out_tr_Y = off_X, out_tr_X = off_Y / 2;
    const int sc_tr_Y = (stage_X * TILE_DIM_X) + 2 * tid_X;
    const int sc_tr_X = (stage_Y * SCALES_PER_TILE_Y) + tid_Y;

    __align__(8) IType rIn[2][SCALE_DIM];
    IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

    #pragma unroll
    for (int i = 0; i < SCALE_DIM; ++i) {
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
            const nvfp4_scale_t S_mult_fp8 =
                compute_encoding_scaling_factor_nv(bmax[w], S_enc);
            coeff = static_cast<float>(S_mult_fp8) * S_enc;
            S_b_fp8 = static_cast<nvfp4_scale_t>(1.0f / static_cast<float>(S_mult_fp8));
        } else {
            S_b_fp8 = compute_decoding_scaling_factor(bmax[w], S_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float S_dec = 1.0f / S_enc;
            coeff = fminf(1.0f / (static_cast<float>(S_b_fp8) * S_dec), float_max);
        }
        sSFcolwise[sc_tr_Y + w][sc_tr_X] = S_b_fp8;

        __align__(8) uint32_t rOut[SCALE_DIM / 8];
        #pragma unroll
        for (int e = 0; e < SCALE_DIM / 8; ++e) {
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&rIn[w][8 * e + 4]);
            rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
        }
        ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_Y + w][out_tr_X],
                           *reinterpret_cast<uint64_t*>(rOut));
    }
}

template <int GROUP_THREADS>
__device__ __forceinline__ float scan_tile_amax_group(
    const IType* __restrict__ sIn_ptr,
    int buff_in,
    int tid
) {
    static_assert(GROUP_THREADS == 128, "scan_tile_amax_group currently expects 128 consumer threads");
    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    const int lane = tid % THREADS_PER_WARP;
    const int bank_group = lane / THREADS_PER_BANK;
    const int tid_Y = tid / THREADS_X_ROWWISE;
    const int tid_X = tid % THREADS_X_ROWWISE;
    const int off_X = tid_X * ELTS_PER_THREAD;
    float tile_max = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t elts = ptx::ld_shared_b128(&sIn[buff_in][row][off_X + sw]);
            const IType2* pairs = reinterpret_cast<const IType2*>(&elts);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                float a = __bfloat162float(__habs(pairs[e].x));
                float b = __bfloat162float(__habs(pairs[e].y));
                tile_max = fmaxf(tile_max, fmaxf(a, b));
            }
        }
    }
    return tile_max;
}

__device__ __forceinline__ float scan_tile_amax(const IType* __restrict__ sIn_ptr, int buff_in) {
    const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
    const int lane = threadIdx.x % THREADS_PER_WARP;
    const int bank_group = lane / THREADS_PER_BANK;
    const int tid_Y = threadIdx.x / THREADS_X_ROWWISE;
    const int tid_X = threadIdx.x % THREADS_X_ROWWISE;
    const int off_X = tid_X * ELTS_PER_THREAD;
    float tile_max = 0.0f;

    #pragma unroll
    for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
        const int row = tid_Y + it * THREADS_Y_ROWWISE;
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const int sw = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
            __uint128_t elts = ptx::ld_shared_b128(&sIn[buff_in][row][off_X + sw]);
            const IType2* pairs = reinterpret_cast<const IType2*>(&elts);
            #pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
                float a = __bfloat162float(__habs(pairs[e].x));
                float b = __bfloat162float(__habs(pairs[e].y));
                tile_max = fmaxf(tile_max, fmaxf(a, b));
            }
        }
    }
    return tile_max;
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS)
fused_localcta_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    LocalCTAPersistentArgs args,
    bool write_raw_scales,
    bool write_prepared
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
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);
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
            cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
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
            if (write_raw_scales && leading) {
                tma_store_scales_2x512(tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
            }
            if (write_prepared) {
                if (write_raw_scales && leading) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
                __syncthreads();
                scale_swizzled_scales_inplace(
                    sSFrowwise_ptr,
                    LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
                    sg_val);
                ptx::fence_proxy_async_shared_cta();
                __syncthreads();
                if (leading) {
                    tma_store_scales_2x512(
                        tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                }
            }
        }

        if constexpr (RETURN_TRANSPOSE) {
            const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
            swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
            ptx::fence_proxy_async_shared_cta();
            __syncthreads();
            if (write_raw_scales && leading) {
                tma_store_scales_2x512(tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
            }
            if (write_prepared) {
                if (write_raw_scales && leading) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
                __syncthreads();
                scale_swizzled_scales_inplace(
                    sSFcolwise_ptr,
                    LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                    sg_val);
                ptx::fence_proxy_async_shared_cta();
                __syncthreads();
                if (leading) {
                    tma_store_scales_2x512(
                        tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                }
            }
        }

        if (leading && (write_raw_scales || write_prepared)) {
            ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
        mbar_phase ^= 1;
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(THREADS) __cluster_dims__(2, 1, 1)
fused_localcta_quantize_kernel_2cta(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    float* __restrict__ cluster_amax_scratch,
    const size_t rows, const size_t cols,
    LocalCTA2ClusterArgs args,
    bool write_raw_scales,
    bool write_prepared
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
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);
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

    const int cta_rank = blockIdx.x & 1;
    const int owner_block = blockIdx.x & ~1;
    __shared__ float warp_max[THREADS / 32];
    __shared__ float cluster_amax_shared;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int mbar_phase = 0;

    while (true) {
        if (leading) {
            if (cta_rank == 0) {
                cluster_amax_scratch[owner_block] = static_cast<float>(atomicAdd(args.work_counter, 1));
            } else {
                cluster_amax_scratch[blockIdx.x] = 0.0f;
            }
            __threadfence();
        }
        cluster_sync_aligned();

        const int macro_tile_id = static_cast<int>(cluster_amax_scratch[owner_block]);
        if (macro_tile_id >= args.total_macro_tiles) {
            break;
        }

        const int macro_ctaid_Y = macro_tile_id / args.tiles_X;
        const int ctaid_X = macro_tile_id % args.tiles_X;
        const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
        const bool cta_active = ctaid_Y < args.tiles_Y;

        const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
        const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
        const int chunk_rows = cta_active ? static_cast<int>(rows) - block_offset_Y : 0;
        const int chunk_cols = cta_active ? static_cast<int>(cols) - block_offset_X : 0;

        float cta_max = 0.0f;
        if (cta_active) {
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
                cta_max = fmaxf(cta_max, scan_tile_amax(sIn_ptr, t));
            }

            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
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
                cluster_amax_scratch[blockIdx.x] = cta_max;
            }
        }
        __syncthreads();

        if (leading) {
            __threadfence();
        }
        cluster_sync_aligned();

        if (leading) {
            cluster_amax_shared = fmaxf(cluster_amax_scratch[owner_block],
                                        cluster_amax_scratch[owner_block + 1]);
        }
        __syncthreads();

        if (cta_active) {
            const float amax_val = cluster_amax_shared;
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
                if (write_raw_scales && leading) {
                    tma_store_scales_2x512(tmap_scale_row, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                }
                if (write_prepared) {
                    if (write_raw_scales && leading) {
                        ptx::cp_async_bulk_wait_group_read<0>();
                    }
                    __syncthreads();
                    scale_swizzled_scales_inplace(
                        sSFrowwise_ptr,
                        LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
                        sg_val);
                    ptx::fence_proxy_async_shared_cta();
                    __syncthreads();
                    if (leading) {
                        tma_store_scales_2x512(
                            tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                    }
                }
            }

            if constexpr (RETURN_TRANSPOSE) {
                const int cnt = min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);
                swizzle_scales_col_inplace(sSFcolwise_ptr, cnt);
                ptx::fence_proxy_async_shared_cta();
                __syncthreads();
                if (write_raw_scales && leading) {
                    tma_store_scales_2x512(tmap_scale_col, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                }
                if (write_prepared) {
                    if (write_raw_scales && leading) {
                        ptx::cp_async_bulk_wait_group_read<0>();
                    }
                    __syncthreads();
                    scale_swizzled_scales_inplace(
                        sSFcolwise_ptr,
                        LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                        sg_val);
                    ptx::fence_proxy_async_shared_cta();
                    __syncthreads();
                    if (leading) {
                        tma_store_scales_2x512(
                            tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
                }
            }

            if (leading && (write_raw_scales || write_prepared)) {
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            __syncthreads();
        }

        mbar_phase ^= 1;
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int PIPE_DEPTH, bool RETURN_TRANSPOSE>
inline int prepared_2cta_tuned_shmem_size() {
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int input_ring_bytes = PIPE_DEPTH * slot_in_bytes;
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
    return input_ring_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <int PIPE_DEPTH, bool RETURN_TRANSPOSE>
inline int prepared_1cta_tuned_shmem_size() {
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
        BUFFS_NUM_IN * BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int input_ring_bytes = PIPE_DEPTH * slot_in_bytes;
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
    return input_ring_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

template <int TOTAL_THREADS, int PIPE_DEPTH, bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(TOTAL_THREADS)
fused_localcta_quantize_kernel_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    int tiles_X,
    int total_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS > 128 && TOTAL_THREADS <= 256 && TOTAL_THREADS % 32 == 0,
                  "tuned 1CTA prepared kernel expects 128 consumer threads plus at least one producer warp");

    constexpr int CONSUMER_THREADS = 128;
    constexpr int NUM_TILES_PER_CHUNK = BUFFS_NUM_IN;
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
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
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ring_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn_ring = *reinterpret_cast<ITypeRing4D<PIPE_DEPTH>*>(sIn_ring_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[PIPE_DEPTH][NUM_TILES_PER_CHUNK];
    __shared__ int slot_ready[PIPE_DEPTH];
    __shared__ int slot_phase[PIPE_DEPTH];
    __shared__ int slot_tile_id[PIPE_DEPTH];
    __shared__ float slot_amax[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_phase[s] = 0;
            slot_tile_id[s] = -1;
            slot_amax[s] = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_init(&in_mbar[s][t], 1);
            }
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int block_iters = (blockIdx.x < total_tiles)
        ? ((total_tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1)
        : 0;

    if (producer_leader) {
        for (int fill_iter = 0; fill_iter < block_iters; ++fill_iter) {
            const int slot = fill_iter % PIPE_DEPTH;
            while (slot_ready[slot] != 0) {
                __nanosleep(64);
            }

            const int tile_id = (int)blockIdx.x + fill_iter * (int)gridDim.x;
            slot_tile_id[slot] = tile_id;

            const int ctaid_Y = tile_id / tiles_X;
            const int ctaid_X = tile_id % tiles_X;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                const int ty = t / TILES_X;
                const int tx = t % TILES_X;
                ptx::mbarrier_arrive_expect_tx(&in_mbar[slot][t], shmem_tile_bytes);
                ptx::cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sIn_ring[slot][t]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    block_offset_X + tx * TILE_DIM_X,
                    block_offset_Y + ty * TILE_DIM_Y,
                    &in_mbar[slot][t]);
            }

            __threadfence_block();
            slot_ready[slot] = 1;
        }
    }

    if (is_consumer) {
        __shared__ float warp_max[CONSUMER_THREADS / 32];

        for (int consume_iter = 0; consume_iter < block_iters; ++consume_iter) {
            const int slot = consume_iter % PIPE_DEPTH;
            while (slot_ready[slot] == 0) {
                __nanosleep(64);
            }

            const int tile_id = slot_tile_id[slot];
            const int ctaid_Y = tile_id / tiles_X;
            const int ctaid_X = tile_id % tiles_X;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
            const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
            const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
            const int chunk_cols = static_cast<int>(cols) - block_offset_X;

            float cta_max = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[slot][t], slot_phase[slot]);
                cta_max = fmaxf(cta_max, scan_tile_amax_group<CONSUMER_THREADS>(&sIn_ring[slot][0][0][0], t, consumer_tid));
            }

            const int lane = consumer_tid % 32;
            const int wid = consumer_tid / 32;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
            }
            if (lane == 0) {
                warp_max[wid] = cta_max;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            if (wid == 0) {
                cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                #pragma unroll
                for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                    cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                }
                if (lane == 0) {
                    slot_amax[slot] = cta_max;
                    row_sg_chunks[ctaid_Y * tiles_X + ctaid_X] = cta_max / localcta_global_scale_num();
                    if constexpr (RETURN_TRANSPOSE) {
                        const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                        col_sg_chunks[ctaid_X * tiles_Y_full + ctaid_Y] = cta_max / localcta_global_scale_num();
                    }
                }
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            const float amax_val = slot_amax[slot];
            const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
            const float sg_val = amax_val / localcta_global_scale_num();

            int buff_out = 0;
            int buff_out_tr = 0;

            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                const int stage_Y = t / TILES_X;
                const int stage_X = t % TILES_X;
                const int stage_offset_Y = stage_Y * TILE_DIM_Y;
                const int stage_offset_X = stage_X * TILE_DIM_X;

                if (t > 0 && consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<1>();
                }

                rowwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                    &sIn_ring[slot][0][0][0], sOut_ptr, sSFrowwise_ptr,
                    S_enc, stage_Y, stage_X, t, buff_out, consumer_tid);

                if constexpr (RETURN_TRANSPOSE) {
                    colwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                        &sIn_ring[slot][0][0][0], sOut_tr_ptr, sSFcolwise_ptr,
                        S_enc, stage_Y, stage_X, t, buff_out_tr, consumer_tid);
                }

                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();

                if (consumer_leader) {
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

            if (consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<0>();
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();

            swizzle_scales_row_inplace_group<CONSUMER_THREADS>(
                sSFrowwise_ptr,
                min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM),
                consumer_tid);
            scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                sSFrowwise_ptr,
                LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
                sg_val,
                consumer_tid);
            subgroup_barrier_sync<CONSUMER_THREADS>();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            subgroup_barrier_sync<CONSUMER_THREADS>();
            if (consumer_leader) {
                tma_store_scales_2x512(
                    tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
            }

            if constexpr (RETURN_TRANSPOSE) {
                if (consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
                swizzle_scales_col_inplace_group<CONSUMER_THREADS>(
                    sSFcolwise_ptr,
                    min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM),
                    consumer_tid);
                scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                    sSFcolwise_ptr,
                    LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                    sg_val,
                    consumer_tid);
                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (consumer_leader) {
                    tma_store_scales_2x512(
                        tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                }
            }

            if (consumer_leader) {
                ptx::cp_async_bulk_wait_group_read<0>();
                slot_phase[slot] ^= 1;
                __threadfence_block();
                slot_ready[slot] = 0;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_invalid(&in_mbar[s][t]);
            }
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX,
          bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(TOTAL_THREADS) __cluster_dims__(2, 1, 1)
fused_localcta_quantize_kernel_2cta_prepared_tuned(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row_prepared,
    const __grid_constant__ CUtensorMap tmap_scale_col_prepared,
    float* __restrict__ row_sg_chunks,
    float* __restrict__ col_sg_chunks,
    const size_t rows, const size_t cols,
    int tiles_X,
    int tiles_Y,
    int total_macro_tiles
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(TOTAL_THREADS >= 160 && TOTAL_THREADS <= 512 && TOTAL_THREADS % 32 == 0,
                  "tuned 2CTA prepared kernel expects at least 128 consumer threads plus producer warp(s)");

    constexpr int CONSUMER_THREADS = 128;
    constexpr int NUM_TILES_PER_CHUNK = BUFFS_NUM_IN;
    constexpr int slot_in_bytes = DIVUP_TO_MULTIPLE(
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
    constexpr int shmem_tile_bytes = BUFF_DIM_Y * BUFF_DIM_X * sizeof(IType);

    const bool is_consumer = (threadIdx.x < CONSUMER_THREADS);
    const int consumer_tid = threadIdx.x;
    const bool producer_leader = (threadIdx.x == CONSUMER_THREADS);
    const bool consumer_leader = (threadIdx.x == 0);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ring_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes);
    fp4e2m1x2* sOut_tr_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr =
        reinterpret_cast<nvfp4_scale_t*>(dshmem + PIPE_DEPTH * slot_in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn_ring = *reinterpret_cast<ITypeRing4D<PIPE_DEPTH>*>(sIn_ring_ptr);
    auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
    auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);

    __shared__ uint64_t in_mbar[PIPE_DEPTH][NUM_TILES_PER_CHUNK];
    __shared__ int slot_ready[PIPE_DEPTH];
    __shared__ int slot_phase[PIPE_DEPTH];
    __shared__ int slot_macro_id[PIPE_DEPTH];
    __shared__ float slot_local_amax[PIPE_DEPTH];
    __shared__ uint32_t slot_local_ready[PIPE_DEPTH];
    __shared__ float slot_combined_amax[PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            slot_ready[s] = 0;
            slot_phase[s] = 0;
            slot_macro_id[s] = -1;
            slot_local_amax[s] = 0.0f;
            slot_local_ready[s] = 0;
            slot_combined_amax[s] = 0.0f;
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_init(&in_mbar[s][t], 1);
            }
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int cta_rank = blockIdx.x & 1;
    const int peer_cta = cta_rank ^ 1;
    const int cluster_id = blockIdx.x >> 1;
    const int num_clusters = max(1, (int)(gridDim.x >> 1));
    const int cluster_iters = (cluster_id < total_macro_tiles)
        ? ((total_macro_tiles - 1 - cluster_id) / num_clusters + 1)
        : 0;

    if (producer_leader) {
        for (int fill_iter = 0; fill_iter < cluster_iters; ++fill_iter) {
            const int slot = fill_iter % PIPE_DEPTH;
            while (slot_ready[slot] != 0) {
                __nanosleep(64);
            }

            const int macro_tile_id = cluster_id + fill_iter * num_clusters;
            slot_macro_id[slot] = macro_tile_id;
            slot_local_ready[slot] = 0;

            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;
            const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
            const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;

            if (cta_active) {
                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    const int ty = t / TILES_X;
                    const int tx = t % TILES_X;
                    ptx::mbarrier_arrive_expect_tx(&in_mbar[slot][t], shmem_tile_bytes);
                    ptx::cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&sIn_ring[slot][t]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        block_offset_X + tx * TILE_DIM_X,
                        block_offset_Y + ty * TILE_DIM_Y,
                        &in_mbar[slot][t]);
                }
            }

            __threadfence_block();
            slot_ready[slot] = 1;
        }
    }

    if (is_consumer) {
        __shared__ float warp_max[CONSUMER_THREADS / 32];

        for (int consume_iter = 0; consume_iter < cluster_iters; ++consume_iter) {
            const int slot = consume_iter % PIPE_DEPTH;
            while (slot_ready[slot] == 0) {
                __nanosleep(64);
            }

            const int macro_tile_id = slot_macro_id[slot];
            const int macro_ctaid_Y = macro_tile_id / tiles_X;
            const int ctaid_X = macro_tile_id % tiles_X;
            const int ctaid_Y = macro_ctaid_Y * 2 + cta_rank;
            const bool cta_active = ctaid_Y < tiles_Y;

            if (cta_active) {
                const int block_offset_Y = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int block_offset_X = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                const int block_offset_Y_tr = ctaid_X * LocalCTAConfig::CHUNK_DIM_X;
                const int block_offset_X_tr = ctaid_Y * LocalCTAConfig::CHUNK_DIM_Y;
                const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
                const int chunk_cols = static_cast<int>(cols) - block_offset_X;

                float cta_max = 0.0f;
                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[slot][t], slot_phase[slot]);
                    cta_max = fmaxf(cta_max, scan_tile_amax_group<CONSUMER_THREADS>(&sIn_ring[slot][0][0][0], t, consumer_tid));
                }

                const int lane = consumer_tid % 32;
                const int wid = consumer_tid / 32;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                }
                if (lane == 0) {
                    warp_max[wid] = cta_max;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();

                if (wid == 0) {
                    cta_max = (lane < CONSUMER_THREADS / 32) ? warp_max[lane] : 0.0f;
                    #pragma unroll
                    for (int mask = (CONSUMER_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                        cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
                    }
                    if (lane == 0) {
                        slot_local_amax[slot] = cta_max;
                        __threadfence_block();
                        slot_local_ready[slot] = 1;
                    }
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();

                float amax_val = slot_local_amax[slot];
                if constexpr (SHARED_AMAX) {
                    if (consumer_leader) {
                        while (cluster_load_shared_u32(&slot_local_ready[slot], peer_cta) == 0u) {
                            __nanosleep(64);
                        }
                        const float peer_amax = cluster_load_shared_f32(&slot_local_amax[slot], peer_cta);
                        slot_combined_amax[slot] = fmaxf(slot_local_amax[slot], peer_amax);
                    }
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    amax_val = slot_combined_amax[slot];
                }

                const float S_enc = compute_localcta_encode_scaling_factor_FP4(amax_val);
                const float sg_val = amax_val / localcta_global_scale_num();
                if (consumer_leader) {
                    row_sg_chunks[ctaid_Y * tiles_X + ctaid_X] = sg_val;
                    if constexpr (RETURN_TRANSPOSE) {
                        const int tiles_Y_full = static_cast<int>(rows / LocalCTAConfig::CHUNK_DIM_Y);
                        col_sg_chunks[ctaid_X * tiles_Y_full + ctaid_Y] = sg_val;
                    }
                }

                int buff_out = 0;
                int buff_out_tr = 0;

                #pragma unroll
                for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                    const int stage_Y = t / TILES_X;
                    const int stage_X = t % TILES_X;
                    const int stage_offset_Y = stage_Y * TILE_DIM_Y;
                    const int stage_offset_X = stage_X * TILE_DIM_X;

                    if (t > 0 && consumer_leader) {
                        ptx::cp_async_bulk_wait_group_read<1>();
                    }

                    rowwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                        &sIn_ring[slot][0][0][0], sOut_ptr, sSFrowwise_ptr,
                        S_enc, stage_Y, stage_X, t, buff_out, consumer_tid);

                    if constexpr (RETURN_TRANSPOSE) {
                        colwise_scaling_group<CONSUMER_THREADS, ENCODE_CENTRIC>(
                            &sIn_ring[slot][0][0][0], sOut_tr_ptr, sSFcolwise_ptr,
                            S_enc, stage_Y, stage_X, t, buff_out_tr, consumer_tid);
                    }

                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();

                    if (consumer_leader) {
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

                if (consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();

                swizzle_scales_row_inplace_group<CONSUMER_THREADS>(
                    sSFrowwise_ptr,
                    min((int)SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM),
                    consumer_tid);
                scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                    sSFrowwise_ptr,
                    LocalCTAConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X,
                    sg_val,
                    consumer_tid);
                subgroup_barrier_sync<CONSUMER_THREADS>();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                subgroup_barrier_sync<CONSUMER_THREADS>();
                if (consumer_leader) {
                    tma_store_scales_2x512(
                        tmap_scale_row_prepared, sSFrowwise_ptr, ctaid_Y, ctaid_X * 2 * 256);
                }

                if constexpr (RETURN_TRANSPOSE) {
                    if (consumer_leader) {
                        ptx::cp_async_bulk_wait_group_read<0>();
                    }
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    swizzle_scales_col_inplace_group<CONSUMER_THREADS>(
                        sSFcolwise_ptr,
                        min((int)SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM),
                        consumer_tid);
                    scale_swizzled_scales_inplace_group<CONSUMER_THREADS>(
                        sSFcolwise_ptr,
                        LocalCTAConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y,
                        sg_val,
                        consumer_tid);
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    subgroup_barrier_sync<CONSUMER_THREADS>();
                    if (consumer_leader) {
                        tma_store_scales_2x512(
                            tmap_scale_col_prepared, sSFcolwise_ptr, ctaid_X, ctaid_Y * 2 * 256);
                    }
                }

                if (consumer_leader) {
                    ptx::cp_async_bulk_wait_group_read<0>();
                    slot_local_ready[slot] = 0;
                }
                subgroup_barrier_sync<CONSUMER_THREADS>();
            }

            if (consumer_leader) {
                slot_phase[slot] ^= 1;
                __threadfence_block();
                slot_ready[slot] = 0;
            }
            subgroup_barrier_sync<CONSUMER_THREADS>();
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            #pragma unroll
            for (int t = 0; t < NUM_TILES_PER_CHUNK; ++t) {
                ptx::mbarrier_invalid(&in_mbar[s][t]);
            }
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_TRANSPOSE>
inline int shmem_size() {
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

    return in_bytes + out_bytes + out_tr_bytes + sc_row_bytes + sc_col_bytes +
           TMA_SHMEM_ALIGNMENT;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace tk_localcta

#endif  // TK_LOCALCTA_FUSED_QUANTIZE_CUH_

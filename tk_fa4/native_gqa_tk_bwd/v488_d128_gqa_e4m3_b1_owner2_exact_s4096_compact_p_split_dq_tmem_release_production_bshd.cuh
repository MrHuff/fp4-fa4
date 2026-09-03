#pragma once

#include "v438_d128_gqa_e4m3_b2_owner2_exact_s4096_production_bshd.cuh"

// B1/S4096 sequential owner-2 experiment reusing v439's retained K/V,
// FP32 dK/dV ownership, exact sequence schedule, runtime first E4M3
// gradient-MMA accumulate predicate, and additive dQ/dK/dV publication.
// Preserve v445's exact rounded-E4M3 compact P words in registers from dV
// publication through dS.  Split dQ's aliased TMEM lifetime from its shared
// publication lifetime: capture the complete exact BF16 payload in registers,
// release TMEM for the next dP, then publish the shared tile.  All other shapes
// retain v439's v438 fallback.
namespace tkfa4::native_gqa_tk_bwd::v488_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_split_dq_tmem_release_production_bshd {

namespace exact =
    tkfa4::native_gqa_tk_bwd::v436_d128_gqa_e4m3_exact_s4096_runtime_accumulate_production_bshd;
namespace fallback =
    tkfa4::native_gqa_tk_bwd::v438_d128_gqa_e4m3_b2_owner2_exact_s4096_production_bshd;
namespace prior =
    tkfa4::native_gqa_tk_bwd::v433_d128_gqa_e4m3_head_fast_raster_production_bshd;
namespace x32 =
    tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd;
namespace core =
    tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd;
namespace d64 =
    tkfa4::native_gqa_tk_bwd::v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds;

using prior::attention_tmem_tile;
using prior::globals;
using prior::gradient_tmem_tile;
using prior::gradient_full_tile;
using prior::kComputeWarps;
using prior::kColumnHalf;
using prior::kDkTmemOffset;
using prior::kDpDqTmemOffset;
using prior::kDvTmemOffset;
using prior::kHeadRatio;
using prior::kInputStages;
using prior::kKeyTile;
using prior::kKvHeads;
using prior::kLoaderWarp;
using prior::kQueryHeads;
using prior::kQueryTile;
using prior::kReduceWarpBase;
using prior::kReduceWarps;
using prior::kScoreTmemOffset;
using prior::kTensorIssueWarp;
using prior::kThreads;
using prior::operand_tile;
using prior::shared_storage;
using prior::stats_tile;

constexpr int kHeadsPerOwner = 2;
constexpr int kHeadPairs = kQueryHeads / kHeadsPerOwner;
constexpr int kPairOwnersPerKvHead = kHeadRatio / kHeadsPerOwner;
constexpr int kExactSequence = exact::kExactSequence;
constexpr int kExactQueryTiles = exact::kExactQueryTiles;
static_assert(kQueryHeads == 32 && kKvHeads == 8);
static_assert(kHeadsPerOwner == 2 && kHeadPairs == 16);
static_assert(kPairOwnersPerKvHead == 2);
static_assert(kExactSequence == 4096 && kExactQueryTiles == 32);

// Each compute lane owns 32 probability elements for one 16x64 half tile.
// Preserve those elements in the exact E4M3 words published for dV, rather
// than retaining the pre-rounding FP32 probability or loading the rounded
// payload back from shared memory before dS.  Two halves therefore cost only
// sixteen 32-bit registers per lane across the dP MMA.
constexpr int kCompactProbabilityWords = kColumnHalf / 8;
struct compact_probability_half {
    uint32_t words[kCompactProbabilityWords];
};
static_assert(kCompactProbabilityWords == 8);
static_assert(sizeof(compact_probability_half) == 8 * sizeof(uint32_t));

template <typename SharedTile>
__device__ __forceinline__ void publish_compact_probability_half(
    compact_probability_half &compact,
    SharedTile &destination,
    const d64::owner_aligned_fp32_half &source
) {
    static_assert(SharedTile::rows == 16);
    static_assert(SharedTile::cols == kColumnHalf);
    static_assert(sizeof(typename SharedTile::dtype) == 1);
    static_assert(SharedTile::swizzle);
    static_assert(SharedTile::swizzle_bytes == 128);

    const int lane_row = laneid() & 15;
    const int lane_column_base = 32 * (laneid() >> 4);
    const uint32_t shared_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&destination.data[0])
    );
#pragma unroll
    for (int word_pair = 0; word_pair < kCompactProbabilityWords / 2;
         ++word_pair) {
        const int first_word = 2 * word_pair;
        const int second_word = first_word + 1;
        const uint32_t packed_first = d64::pack_owner_e4m3_word(
            source.pairs[2 * first_word],
            source.pairs[2 * first_word + 1]
        );
        const uint32_t packed_second = d64::pack_owner_e4m3_word(
            source.pairs[2 * second_word],
            source.pairs[2 * second_word + 1]
        );
        compact.words[first_word] = packed_first;
        compact.words[second_word] = packed_second;
        const int column = lane_column_base + 4 * first_word;
        const uint32_t destination_address = destination.idx(
            shared_address,
            {lane_row, column}
        );
        asm volatile(
            "st.shared.v2.b32 [%0], {%1, %2};\n"
            :
            : "r"(destination_address),
              "r"(packed_first),
              "r"(packed_second)
            : "memory"
        );
    }
}

__device__ __forceinline__ void make_probability_half_compact(
    compact_probability_half &compact,
    const attention_tmem_tile &score_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int input_stage,
    bool diagonal,
    float beta_log2e,
    semaphore *score_consumed
) {
    d64::owner_aligned_fp32_half probability;
    d64::load_owner_aligned_fp32_half(
        probability,
        score_tmem,
        output_subtile,
        column_half
    );
    tensor_load_wait();

    if (score_consumed != nullptr) {
        tensor_before_thread_sync();
        __syncwarp();
        if (laneid() == 0) {
            arrive(*score_consumed);
        }
    }

    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    const int lane_row = laneid() & 15;
    const int lane_column_base = 32 * (laneid() >> 4);
    const int key_row = output_subtile * 16 + lane_row;
    const int query_column_base =
        column_half * kColumnHalf + lane_column_base;

#pragma unroll
    for (int pair = 0; pair < kColumnHalf / 4; ++pair) {
        const int local_column = 2 * pair;
        const float2 statistic = *reinterpret_cast<const float2 *>(
            &storage.lstat[input_stage][
                query_column_base + local_column
            ]
        );
        float2 value = probability.pairs[pair];
        value.x = value.x * beta_log2e + statistic.x;
        value.y = value.y * beta_log2e + statistic.y;
        if (diagonal) {
            if (key_row > query_column_base + local_column) {
                value.x = kNegInf;
            }
            if (key_row > query_column_base + local_column + 1) {
                value.y = kNegInf;
            }
        }
        value = d64::clamp_probability_log2(value);
        probability.pairs[pair] = d64::exp2_native_f32x2(value);
    }

    auto destination =
        storage.probability.template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    publish_compact_probability_half(compact, destination, probability);
}

__device__ __forceinline__ float4 expand_compact_e4m3_word(
    uint32_t packed
) {
    const fp8e4m3_4 probability = std::bit_cast<fp8e4m3_4>(packed);
    return kittens::base_types::convertor<float4, fp8e4m3_4>::convert(
        probability
    );
}

__device__ __forceinline__ uint32_t make_ds_compact_word(
    uint32_t probability_word,
    d64::owner_aligned_fp32_half &dp,
    const stats_tile &dstat,
    int query_column_base,
    int word,
    float beta
) {
    const int first_pair = 2 * word;
    const int first_column = 2 * first_pair;
    const float4 probability = expand_compact_e4m3_word(
        probability_word
    );
    const float2 statistic_first = *reinterpret_cast<const float2 *>(
        &dstat[query_column_base + first_column]
    );
    const float2 statistic_second = *reinterpret_cast<const float2 *>(
        &dstat[query_column_base + first_column + 2]
    );
    float2 first = dp.pairs[first_pair];
    float2 second = dp.pairs[first_pair + 1];
    first.x = (first.x + statistic_first.x) * probability.x;
    first.y = (first.y + statistic_first.y) * probability.y;
    second.x = (second.x + statistic_second.x) * probability.z;
    second.y = (second.y + statistic_second.y) * probability.w;
    first.x *= beta;
    first.y *= beta;
    second.x *= beta;
    second.y *= beta;
    return d64::pack_owner_e4m3_word(first, second);
}

__device__ __forceinline__ void make_ds_half_compact(
    const attention_tmem_tile &dp_tmem,
    const compact_probability_half &compact,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int input_stage,
    float beta
) {
    d64::owner_aligned_fp32_half dp;
    d64::load_owner_aligned_fp32_half(
        dp,
        dp_tmem,
        output_subtile,
        column_half
    );
    tensor_load_wait();

    auto destination = storage.ds.template subtile<16, kColumnHalf>(
        {output_subtile, column_half}
    );
    const int lane_row = laneid() & 15;
    const int lane_column_base = 32 * (laneid() >> 4);
    const int query_column_base =
        column_half * kColumnHalf + lane_column_base;
    const uint32_t shared_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&destination.data[0])
    );
#pragma unroll
    for (int word_pair = 0; word_pair < kCompactProbabilityWords / 2;
         ++word_pair) {
        const int first_word = 2 * word_pair;
        const int second_word = first_word + 1;
        const uint32_t ds_first = make_ds_compact_word(
            compact.words[first_word],
            dp,
            storage.dstat[input_stage],
            query_column_base,
            first_word,
            beta
        );
        const uint32_t ds_second = make_ds_compact_word(
            compact.words[second_word],
            dp,
            storage.dstat[input_stage],
            query_column_base,
            second_word,
            beta
        );
        const int column = lane_column_base + 4 * first_word;
        const uint32_t destination_address = destination.idx(
            shared_address,
            {lane_row, column}
        );
        asm volatile(
            "st.shared.v2.b32 [%0], {%1, %2};\n"
            :
            : "r"(destination_address), "r"(ds_first), "r"(ds_second)
            : "memory"
        );
    }
}

// Separate the TMEM source lifetime from the shared publication lifetime.
// D0 and D1 keep the exact scale/round operation but retain their thirty-two
// packed BF16 words in registers.  The paired D2/D3 loads then advertise
// complete TMEM capture before ordered D0/D1/D2/D3 shared stores execute.
__device__ __forceinline__ void drain_dq_full_owner_x32_split_release(
    const gradient_tmem_tile &source,
    gradient_full_tile &destination,
    semaphore &tmem_drained,
    semaphore &shared_ready,
    int logical_warp,
    int lane
) {
    constexpr float kOutputScale = 1.0f / 256.0f;
    const float2 output_scale = make_float2(kOutputScale, kOutputScale);
    const int physical_row = logical_warp * 32 + lane;
    const uint32_t source_row =
        source.addr +
        (static_cast<uint32_t>(logical_warp * 32) << 16);
    const uint32_t destination_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(destination.data)
    );
    static_assert(prior::kDepthChunks == 4);
    constexpr int kFirstDeferredPackedDepthChunk = 0;
    constexpr int kDeferredPackedDepthChunk =
        prior::kDepthChunks - 3;
    constexpr int kPairedDepthChunk = prior::kDepthChunks - 2;
    constexpr int kFinalDepthChunk = prior::kDepthChunks - 1;
    uint32_t values[prior::kDepthChunk];
    uint32_t deferred_packed_d0_values[prior::kDepthChunk / 2];

#pragma unroll 1
    for (
        int depth_chunk = 0;
        depth_chunk < kDeferredPackedDepthChunk;
        ++depth_chunk
    ) {
        x32::load_tmem_owner_x32(
            values,
            source_row + depth_chunk * prior::kDepthChunk
        );
        tensor_load_wait();

#pragma unroll
        for (
            int column = 0;
            column < prior::kDepthChunk;
            column += 8
        ) {
            const float2 value0 = make_float2(
                __uint_as_float(values[column + 0]),
                __uint_as_float(values[column + 1])
            );
            const float2 value1 = make_float2(
                __uint_as_float(values[column + 2]),
                __uint_as_float(values[column + 3])
            );
            const float2 value2 = make_float2(
                __uint_as_float(values[column + 4]),
                __uint_as_float(values[column + 5])
            );
            const float2 value3 = make_float2(
                __uint_as_float(values[column + 6]),
                __uint_as_float(values[column + 7])
            );
            const bf16_2 packed0 = __float22bfloat162_rn(
                __fmul2_rn(value0, output_scale)
            );
            const bf16_2 packed1 = __float22bfloat162_rn(
                __fmul2_rn(value1, output_scale)
            );
            const bf16_2 packed2 = __float22bfloat162_rn(
                __fmul2_rn(value2, output_scale)
            );
            const bf16_2 packed3 = __float22bfloat162_rn(
                __fmul2_rn(value3, output_scale)
            );
            deferred_packed_d0_values[column / 2 + 0] =
                *reinterpret_cast<const uint32_t *>(&packed0);
            deferred_packed_d0_values[column / 2 + 1] =
                *reinterpret_cast<const uint32_t *>(&packed1);
            deferred_packed_d0_values[column / 2 + 2] =
                *reinterpret_cast<const uint32_t *>(&packed2);
            deferred_packed_d0_values[column / 2 + 3] =
                *reinterpret_cast<const uint32_t *>(&packed3);
        }
    }

    uint32_t deferred_packed_d1_values[prior::kDepthChunk / 2];
    x32::load_tmem_owner_x32(
        values,
        source_row +
            kDeferredPackedDepthChunk * prior::kDepthChunk
    );
    tensor_load_wait();

#pragma unroll
    for (
        int column = 0;
        column < prior::kDepthChunk;
        column += 8
    ) {
        const float2 value0 = make_float2(
            __uint_as_float(values[column + 0]),
            __uint_as_float(values[column + 1])
        );
        const float2 value1 = make_float2(
            __uint_as_float(values[column + 2]),
            __uint_as_float(values[column + 3])
        );
        const float2 value2 = make_float2(
            __uint_as_float(values[column + 4]),
            __uint_as_float(values[column + 5])
        );
        const float2 value3 = make_float2(
            __uint_as_float(values[column + 6]),
            __uint_as_float(values[column + 7])
        );
        const bf16_2 packed0 = __float22bfloat162_rn(
            __fmul2_rn(value0, output_scale)
        );
        const bf16_2 packed1 = __float22bfloat162_rn(
            __fmul2_rn(value1, output_scale)
        );
        const bf16_2 packed2 = __float22bfloat162_rn(
            __fmul2_rn(value2, output_scale)
        );
        const bf16_2 packed3 = __float22bfloat162_rn(
            __fmul2_rn(value3, output_scale)
        );
        deferred_packed_d1_values[column / 2 + 0] =
            *reinterpret_cast<const uint32_t *>(&packed0);
        deferred_packed_d1_values[column / 2 + 1] =
            *reinterpret_cast<const uint32_t *>(&packed1);
        deferred_packed_d1_values[column / 2 + 2] =
            *reinterpret_cast<const uint32_t *>(&packed2);
        deferred_packed_d1_values[column / 2 + 3] =
            *reinterpret_cast<const uint32_t *>(&packed3);
    }

    uint32_t final_values[prior::kDepthChunk];
    x32::load_tmem_owner_x32(
        values,
        source_row + kPairedDepthChunk * prior::kDepthChunk
    );
    x32::load_tmem_owner_x32(
        final_values,
        source_row + kFinalDepthChunk * prior::kDepthChunk
    );
    tensor_load_wait();

    tensor_before_thread_sync();
    __syncwarp();
    if (lane == 0) {
        arrive(tmem_drained);
    }

#pragma unroll
    for (
        int packed_column = 0;
        packed_column < prior::kDepthChunk / 2;
        packed_column += 4
    ) {
        asm volatile(
            "st.shared.v4.b32 [%4], {%0, %1, %2, %3};\n"
            :
            : "r"(deferred_packed_d0_values[packed_column + 0]),
              "r"(deferred_packed_d0_values[packed_column + 1]),
              "r"(deferred_packed_d0_values[packed_column + 2]),
              "r"(deferred_packed_d0_values[packed_column + 3]),
              "r"(gradient_full_tile::idx(
                  destination_base,
                  {
                      physical_row,
                      kFirstDeferredPackedDepthChunk *
                              prior::kDepthChunk +
                          2 * packed_column,
                  }
              ))
            : "memory"
        );
    }

#pragma unroll
    for (
        int packed_column = 0;
        packed_column < prior::kDepthChunk / 2;
        packed_column += 4
    ) {
        asm volatile(
            "st.shared.v4.b32 [%4], {%0, %1, %2, %3};\n"
            :
            : "r"(deferred_packed_d1_values[packed_column + 0]),
              "r"(deferred_packed_d1_values[packed_column + 1]),
              "r"(deferred_packed_d1_values[packed_column + 2]),
              "r"(deferred_packed_d1_values[packed_column + 3]),
              "r"(gradient_full_tile::idx(
                  destination_base,
                  {
                      physical_row,
                      kDeferredPackedDepthChunk *
                              prior::kDepthChunk +
                          2 * packed_column,
                  }
              ))
            : "memory"
        );
    }

#pragma unroll
    for (
        int column = 0;
        column < prior::kDepthChunk;
        column += 8
    ) {
        const float2 value0 = make_float2(
            __uint_as_float(values[column + 0]),
            __uint_as_float(values[column + 1])
        );
        const float2 value1 = make_float2(
            __uint_as_float(values[column + 2]),
            __uint_as_float(values[column + 3])
        );
        const float2 value2 = make_float2(
            __uint_as_float(values[column + 4]),
            __uint_as_float(values[column + 5])
        );
        const float2 value3 = make_float2(
            __uint_as_float(values[column + 6]),
            __uint_as_float(values[column + 7])
        );
        const bf16_2 packed0 = __float22bfloat162_rn(
            __fmul2_rn(value0, output_scale)
        );
        const bf16_2 packed1 = __float22bfloat162_rn(
            __fmul2_rn(value1, output_scale)
        );
        const bf16_2 packed2 = __float22bfloat162_rn(
            __fmul2_rn(value2, output_scale)
        );
        const bf16_2 packed3 = __float22bfloat162_rn(
            __fmul2_rn(value3, output_scale)
        );
        x32::store_bf16_x8(
            gradient_full_tile::idx(
                destination_base,
                {
                    physical_row,
                    kPairedDepthChunk * prior::kDepthChunk + column,
                }
            ),
            packed0,
            packed1,
            packed2,
            packed3
        );
    }

#pragma unroll
    for (
        int column = 0;
        column < prior::kDepthChunk;
        column += 8
    ) {
        const float2 value0 = make_float2(
            __uint_as_float(final_values[column + 0]),
            __uint_as_float(final_values[column + 1])
        );
        const float2 value1 = make_float2(
            __uint_as_float(final_values[column + 2]),
            __uint_as_float(final_values[column + 3])
        );
        const float2 value2 = make_float2(
            __uint_as_float(final_values[column + 4]),
            __uint_as_float(final_values[column + 5])
        );
        const float2 value3 = make_float2(
            __uint_as_float(final_values[column + 6]),
            __uint_as_float(final_values[column + 7])
        );
        const bf16_2 packed0 = __float22bfloat162_rn(
            __fmul2_rn(value0, output_scale)
        );
        const bf16_2 packed1 = __float22bfloat162_rn(
            __fmul2_rn(value1, output_scale)
        );
        const bf16_2 packed2 = __float22bfloat162_rn(
            __fmul2_rn(value2, output_scale)
        );
        const bf16_2 packed3 = __float22bfloat162_rn(
            __fmul2_rn(value3, output_scale)
        );
        x32::store_bf16_x8(
            gradient_full_tile::idx(
                destination_base,
                {
                    physical_row,
                    kFinalDepthChunk * prior::kDepthChunk + column,
                }
            ),
            packed0,
            packed1,
            packed2,
            packed3
        );
    }

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    tensor_before_thread_sync();
    __syncwarp();
    if (lane == 0) {
        arrive(shared_ready);
    }
}

__global__ __launch_bounds__(kThreads, 1)
void b1_owner2_exact_s4096_compact_p_reuse_kernel(const __grid_constant__ globals g) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready[kInputStages];
    __shared__ alignas(16) semaphore stats_ready[kInputStages];
    __shared__ alignas(16) semaphore operand_consumed[kInputStages];
    __shared__ alignas(16) semaphore stats_consumed[kInputStages];
    __shared__ alignas(16) semaphore score_ready;
    __shared__ alignas(16) semaphore score_consumed;
    __shared__ alignas(16) semaphore probability_ready;
    __shared__ alignas(16) semaphore probability_consumed;
    __shared__ alignas(16) semaphore dp_ready;
    __shared__ alignas(16) semaphore dv_ready;
    __shared__ alignas(16) semaphore ds_ready;
    __shared__ alignas(16) semaphore dq_ready;
    __shared__ alignas(16) semaphore dk_ready;
    __shared__ alignas(16) semaphore dq_tmem_drained;
    __shared__ alignas(16) semaphore dq_drained;
    __shared__ alignas(16) semaphore full_gradient_ready;
    __shared__ alignas(16) semaphore full_gradient_reusable;
    __shared__ alignas(16) semaphore kernel_complete;

    const int physical_warp = warpid();
    const int lane = laneid();
    const int linear_owner = static_cast<int>(blockIdx.x);
    const int batch = linear_owner / kHeadPairs;
    const int head_pair = linear_owner - batch * kHeadPairs;
    const int key_tile = static_cast<int>(blockIdx.y);
    const int kv_head = head_pair / kPairOwnersPerKvHead;
    const int iterations_per_head = kExactQueryTiles - key_tile;
    const int total_work = kHeadsPerOwner * iterations_per_head;

    if (physical_warp < kComputeWarps) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;" ::: "memory");
    } else if (physical_warp < kTensorIssueWarp) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;" ::: "memory");
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;" ::: "memory");
    }

    if (threadIdx.x == 0) {
        init_semaphore(persistent_ready, 0, 1);
        for (int stage = 0; stage < kInputStages; ++stage) {
            init_semaphore(query_ready[stage], 0, 1);
            init_semaphore(stats_ready[stage], 0, 1);
            init_semaphore(operand_consumed[stage], 0, 1);
            init_semaphore(stats_consumed[stage], 0, kComputeWarps);
        }
        init_semaphore(score_ready, 0, 1);
        init_semaphore(score_consumed, 0, kComputeWarps);
        init_semaphore(probability_ready, 0, kComputeWarps);
        init_semaphore(probability_consumed, 0, kComputeWarps);
        init_semaphore(dp_ready, 0, 1);
        init_semaphore(dv_ready, 0, 1);
        init_semaphore(ds_ready, 0, kComputeWarps);
        init_semaphore(dq_ready, 0, 1);
        init_semaphore(dk_ready, 0, 1);
        init_semaphore(dq_tmem_drained, 0, kReduceWarps);
        init_semaphore(dq_drained, 0, kReduceWarps);
        init_semaphore(full_gradient_ready, 0, kReduceWarps);
        init_semaphore(full_gradient_reusable, 0, 1);
        init_semaphore(kernel_complete, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tmem_allocator{};
    attention_tmem_tile dp_tmem =
        tmem_allocator.template allocate<attention_tmem_tile>(
            kDpDqTmemOffset
        );
    gradient_tmem_tile dq_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(
            kDpDqTmemOffset
        );
    gradient_tmem_tile dk_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(
            kDkTmemOffset
        );
    gradient_tmem_tile dv_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(
            kDvTmemOffset
        );
    attention_tmem_tile score_tmem =
        tmem_allocator.template allocate<attention_tmem_tile>(
            kScoreTmemOffset
        );

    if (physical_warp == kLoaderWarp && lane == 0) {
        tma::expect_bytes(
            persistent_ready,
            sizeof(storage.k) + sizeof(storage.v)
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k,
            g.k,
            coord<operand_tile>{batch, key_tile, kv_head, 0},
            persistent_ready
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.v,
            g.v,
            coord<operand_tile>{batch, key_tile, kv_head, 0},
            persistent_ready
        );

        int work = 0;
#pragma unroll 1
        for (int local_head = 0; local_head < kHeadsPerOwner; ++local_head) {
            const int query_head =
                kHeadsPerOwner * head_pair + local_head;
            for (
                int iteration = 0;
                iteration < iterations_per_head;
                ++iteration, ++work
            ) {
                const int stage = work & (kInputStages - 1);
                if (work >= kInputStages) {
                    const int old_phase =
                        x32::previous_input_epoch_phase(work);
                    wait(operand_consumed[stage], old_phase);
                    wait(stats_consumed[stage], old_phase);
                    tensor_after_thread_sync();
                }
                const int query_tile = key_tile + iteration;
                const coord<operand_tile> operand_coordinate{
                    batch,
                    query_tile,
                    query_head,
                    0,
                };
                tma::expect_bytes(
                    query_ready[stage],
                    sizeof(storage.q[stage]) +
                        sizeof(storage.dout[stage])
                );
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    storage.q[stage],
                    g.q,
                    operand_coordinate,
                    query_ready[stage]
                );
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    storage.dout[stage],
                    g.dout,
                    operand_coordinate,
                    query_ready[stage]
                );

                const coord<stats_tile> stats_coordinate{
                    batch,
                    query_head,
                    0,
                    query_tile,
                };
                tma::expect_bytes(
                    stats_ready[stage],
                    sizeof(storage.lstat[stage]) +
                        sizeof(storage.dstat[stage])
                );
                tma::load_async(
                    storage.lstat[stage],
                    g.lstat,
                    stats_coordinate,
                    stats_ready[stage]
                );
                tma::load_async(
                    storage.dstat[stage],
                    g.dstat,
                    stats_coordinate,
                    stats_ready[stage]
                );
            }
        }
    } else if (physical_warp < kComputeWarps) {
        const int output_subtile =
            x32::output_subtile_for_warp(physical_warp);
        int work = 0;
#pragma unroll 1
        for (int local_head = 0; local_head < kHeadsPerOwner; ++local_head) {
            for (
                int iteration = 0;
                iteration < iterations_per_head;
                ++iteration, ++work
            ) {
                const int stage = work & (kInputStages - 1);
                const int phase = x32::iteration_phase(work);
                const int input_phase =
                    x32::input_stage_epoch_phase(work);
                if (work > 0) {
                    const int old_phase =
                        x32::iteration_phase(work - 1);
                    wait(dv_ready, old_phase);
                    wait(probability_consumed, old_phase);
                }
                wait(score_ready, phase);
                wait(stats_ready[stage], input_phase);
                tensor_after_thread_sync();
                compact_probability_half probability_compact[2];
                make_probability_half_compact(
                    probability_compact[0],
                    score_tmem,
                    storage,
                    output_subtile,
                    0,
                    stage,
                    iteration == 0,
                    g.beta_log2e,
                    nullptr
                );
                make_probability_half_compact(
                    probability_compact[1],
                    score_tmem,
                    storage,
                    output_subtile,
                    1,
                    stage,
                    iteration == 0,
                    g.beta_log2e,
                    &score_consumed
                );
                tensor_before_thread_sync();
                __syncwarp();
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
                if (lane == 0) {
                    arrive(probability_ready);
                }

                wait(dp_ready, phase);
                if (work > 0) {
                    const int old_phase =
                        x32::iteration_phase(work - 1);
                    wait(dq_ready, old_phase);
                    wait(dk_ready, old_phase);
                }
                tensor_after_thread_sync();
                make_ds_half_compact(
                    dp_tmem,
                    probability_compact[0],
                    storage,
                    output_subtile,
                    0,
                    stage,
                    g.beta
                );
                make_ds_half_compact(
                    dp_tmem,
                    probability_compact[1],
                    storage,
                    output_subtile,
                    1,
                    stage,
                    g.beta
                );
                tensor_before_thread_sync();
                __syncwarp();
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
                if (lane == 0) {
                    arrive(probability_consumed);
                    arrive(stats_consumed[stage]);
                    arrive(ds_ready);
                }
            }
        }
        if (physical_warp == 0) {
            wait(kernel_complete, 0);
            tensor_after_thread_sync();
        }
    } else if (physical_warp == kTensorIssueWarp && lane == 0) {
        wait(persistent_ready, 0);

        int work = 0;
#pragma unroll 1
        for (int local_head = 0; local_head < kHeadsPerOwner; ++local_head) {
            if (local_head > 0) {
                const int previous_phase =
                    x32::iteration_phase(work - 1);
                wait(score_consumed, previous_phase);
                wait(dq_tmem_drained, previous_phase);
            }
            const int first_stage = work & (kInputStages - 1);
            wait(
                query_ready[first_stage],
                x32::input_stage_epoch_phase(work)
            );
            tensor_after_thread_sync();
            core::issue_score_or_dp(
                score_tmem,
                storage.k,
                storage.q[first_stage],
                score_ready
            );
            core::issue_score_or_dp(
                dp_tmem,
                storage.v,
                storage.dout[first_stage],
                dp_ready
            );

            for (
                int iteration = 0;
                iteration < iterations_per_head;
                ++iteration, ++work
            ) {
                const int stage = work & (kInputStages - 1);
                const int phase = x32::iteration_phase(work);
                const bool has_next =
                    iteration + 1 < iterations_per_head;
                const int next_work = work + 1;
                const int next_stage =
                    next_work & (kInputStages - 1);

                if (has_next) {
                    wait(score_consumed, phase);
                    wait(
                        query_ready[next_stage],
                        x32::input_stage_epoch_phase(next_work)
                    );
                    tensor_after_thread_sync();
                    core::issue_score_or_dp(
                        score_tmem,
                        storage.k,
                        storage.q[next_stage],
                        score_ready
                    );
                }

                wait(probability_ready, phase);
                tensor_after_thread_sync();
                exact::issue_gradient_ab_runtime_accumulate(
                    dv_tmem,
                    storage.probability,
                    storage.dout[stage],
                    dv_ready,
                    work != 0
                );

                wait(ds_ready, phase);
                tensor_after_thread_sync();
                core::issue_gradient_atb(
                    dq_tmem,
                    storage.ds,
                    storage.k,
                    dq_ready
                );
                exact::issue_gradient_ab_runtime_accumulate(
                    dk_tmem,
                    storage.ds,
                    storage.q[stage],
                    dk_ready,
                    work != 0
                );

                if (has_next) {
                    wait(dq_tmem_drained, phase);
                    tensor_after_thread_sync();
                    core::issue_score_or_dp(
                        dp_tmem,
                        storage.v,
                        storage.dout[next_stage],
                        dp_ready
                    );
                }

                wait(dk_ready, phase);
                wait(dv_ready, phase);
                tensor_after_thread_sync();
                arrive(operand_consumed[stage]);
            }
        }
    } else if (
        physical_warp >= kReduceWarpBase &&
        physical_warp < kReduceWarpBase + kReduceWarps
    ) {
        const int logical_warp = physical_warp - kReduceWarpBase;
        int work = 0;
#pragma unroll 1
        for (int local_head = 0; local_head < kHeadsPerOwner; ++local_head) {
            const int query_head =
                kHeadsPerOwner * head_pair + local_head;
            for (
                int iteration = 0;
                iteration < iterations_per_head;
                ++iteration, ++work
            ) {
                const int phase = x32::iteration_phase(work);
                wait(dq_ready, phase);
                tensor_after_thread_sync();
                if (work > 0) {
                    prior::wait_full_gradient_reuse(
                        full_gradient_reusable,
                        x32::iteration_phase(work - 1),
                        logical_warp,
                        lane
                    );
                }
                drain_dq_full_owner_x32_split_release(
                    dq_tmem,
                    storage.gradient,
                    dq_tmem_drained,
                    dq_drained,
                    logical_warp,
                    lane
                );
                prior::publish_gradient_full(
                    g.dq,
                    storage.gradient,
                    dq_drained,
                    phase,
                    batch,
                    key_tile + iteration,
                    query_head,
                    logical_warp,
                    lane
                );
            }
        }

        const int last_phase =
            x32::iteration_phase(total_work - 1);
        wait(dk_ready, last_phase);
        tensor_after_thread_sync();
        prior::wait_full_gradient_reuse(
            full_gradient_reusable,
            x32::iteration_phase(total_work - 1),
            logical_warp,
            lane
        );
        prior::drain_gradient_full_owner_x32(
            dk_tmem,
            storage.gradient,
            full_gradient_ready,
            logical_warp,
            lane
        );
        prior::publish_gradient_full(
            g.dk,
            storage.gradient,
            full_gradient_ready,
            0,
            batch,
            key_tile,
            kv_head,
            logical_warp,
            lane
        );

        wait(dv_ready, last_phase);
        tensor_after_thread_sync();
        prior::wait_full_gradient_reuse(
            full_gradient_reusable,
            x32::iteration_phase(total_work),
            logical_warp,
            lane
        );
        prior::drain_gradient_full_owner_x32(
            dv_tmem,
            storage.gradient,
            full_gradient_ready,
            logical_warp,
            lane
        );
        prior::publish_gradient_full(
            g.dv,
            storage.gradient,
            full_gradient_ready,
            1,
            batch,
            key_tile,
            kv_head,
            logical_warp,
            lane
        );
        if (logical_warp == 0 && lane == 0) {
            warp::tma::store_async_wait<0>();
            arrive(kernel_complete);
        }
    }
}

inline globals make_globals(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lstat,
    at::Tensor &dstat,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float softmax_scale
) {
    const float beta = softmax_scale / 16.0f;
    return globals{
        kittens::py::tensor_to_gl<globals::operand_gl>(q),
        kittens::py::tensor_to_gl<globals::operand_gl>(k),
        kittens::py::tensor_to_gl<globals::operand_gl>(v),
        kittens::py::tensor_to_gl<globals::operand_gl>(dout),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dq),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dk),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dv),
        kittens::py::tensor_to_gl<globals::stats_gl>(
            lstat,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        kittens::py::tensor_to_gl<globals::stats_gl>(
            dstat,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        beta,
        beta * kLog2E,
        static_cast<int>(q.size(1)),
    };
}

inline void launch(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lstat,
    at::Tensor &dstat,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float softmax_scale,
    cudaStream_t stream
) {
    if (q.size(0) != 1 || q.size(1) != kExactSequence) {
        fallback::launch(
            q,
            k,
            v,
            dout,
            lstat,
            dstat,
            dq,
            dk,
            dv,
            softmax_scale,
            stream
        );
        return;
    }

    const globals g = make_globals(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        softmax_scale
    );
    const dim3 grid(
        static_cast<unsigned int>(kHeadPairs * q.size(0)),
        static_cast<unsigned int>(q.size(1) / kKeyTile),
        1
    );
    b1_owner2_exact_s4096_compact_p_reuse_kernel<<<grid, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v488_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_split_dq_tmem_release_production_bshd

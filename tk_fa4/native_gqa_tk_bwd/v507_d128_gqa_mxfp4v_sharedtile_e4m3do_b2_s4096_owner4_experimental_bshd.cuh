#pragma once

#include "v436_d128_gqa_e4m3_exact_s4096_runtime_accumulate_production_bshd.cuh"
#include "v437_d128_gqa_e4m3_b2_owner2_production_bshd.cuh"

// Experimental B2/S4096-only exact shared-tile MXFP4-V x E4M3-dO dP fork of
// v502.  The required producer quantizes each D32xS32 forward V tile once,
// register-transposes the exact E2M1 nibbles, and publishes the matching E8M0
// anchor into the backward physical scale page.  Preserve all four independent
// D32 anchors with block-scaled MMA; do not collapse or requantize them here.
// The tensor issuer overlaps the next score with current P1/dS work as soon as
// dp_ready proves the score-aliased scale TMEM dead.  This file is deliberately
// fail-closed to the shared-tile producer ABI and B2/S4096.
namespace tkfa4::native_gqa_tk_bwd::v507_d128_gqa_mxfp4v_sharedtile_e4m3do_b2_s4096_owner4_experimental_bshd {

namespace prior =
    tkfa4::native_gqa_tk_bwd::v436_d128_gqa_e4m3_exact_s4096_runtime_accumulate_production_bshd;
namespace fallback_owner2 =
    tkfa4::native_gqa_tk_bwd::v437_d128_gqa_e4m3_b2_owner2_production_bshd;
namespace x32 =
    tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd;
namespace core =
    tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd;
namespace d64 =
    tkfa4::native_gqa_tk_bwd::v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds;

using prior::attention_tmem_tile;
using prior::attention_tile;
using prior::gradient_tmem_tile;
using prior::gradient_full_tile;
using prior::kComputeWarps;
using prior::kColumnHalf;
using prior::kDkTmemOffset;
using prior::kDepth;
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

constexpr int kHeadsPerOwner = 4;
constexpr int kHeadOwners = kQueryHeads / kHeadsPerOwner;
constexpr int kOwnersPerKvHead = kHeadRatio / kHeadsPerOwner;
static_assert(kQueryHeads == 32 && kKvHeads == 8);
static_assert(kHeadsPerOwner == 4 && kHeadOwners == 8);
static_assert(kOwnersPerKvHead == 1);
static_assert(prior::kExactSequence == 4096);
static_assert(prior::kExactQueryTiles == 32);
constexpr int kGradientPublisherWarp = 14;
static_assert(kGradientPublisherWarp > kLoaderWarp);
static_assert(kGradientPublisherWarp < kThreads / 32);
static_assert(sizeof(shared_storage) == 162 * 1024);

using byte_gl = gl<char, -1, -1, -1, -1>;
using mx_scale_tile = st_fp8e8m0<32, 16, false>;
using mx_scale_tmem_tile = full_tt_fp8e8m0<16>;

// mxf8f6f4 consumes aligned narrow operands: each eight bytes of packed E2M1
// payload occupies the first half of a sixteen-byte shared segment.  The
// existing 128x128 E4M3 V slot is exactly large enough for that representation.
static_assert(sizeof(operand_tile) == 16 * 1024);
static_assert(sizeof(mx_scale_tile) == 512);

struct globals {
    using operand_gl = core::globals::operand_gl;
    using gradient_gl = prior::globals::gradient_gl;
    using stats_gl = core::globals::stats_gl;

    operand_gl q;
    operand_gl k;
    byte_gl v_backward_mxfp4;
    byte_gl v_backward_mxfp4_scales;
    operand_gl dout;
    gradient_gl dq;
    gradient_gl dk;
    gradient_gl dv;
    stats_gl lstat;
    stats_gl dstat;
    float beta;
    float beta_log2e;
    int sequence;
};

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

// One x16 TMEM load yields sixteen FP32 values per lane.  Together, chunks 0
// and 1 exactly cover the x32 owner's first 64-column half: chunk 0 maps
// [0,16)/[32,48), chunk 1 maps [16,32)/[48,64) for lane groups 0/1.
struct owner_aligned_fp32_quarter {
    float2 pairs[kColumnHalf / 8];
};
static_assert(kColumnHalf == 64);
static_assert(
    sizeof(owner_aligned_fp32_quarter) ==
        (kColumnHalf / 4) * sizeof(float)
);

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
    // Projection MXFP4 reconstructs V in the width-six domain while dO is
    // resident E4M3(x4).  Convert raw x24 dP to v490's x16 dstat domain before
    // centering.  FFMA keeps this correction fused with the existing add.
    constexpr float kMxDpRawToX16 = 2.0f / 3.0f;
    first.x = fmaf(first.x, kMxDpRawToX16, statistic_first.x) * probability.x;
    first.y = fmaf(first.y, kMxDpRawToX16, statistic_first.y) * probability.y;
    second.x =
        fmaf(second.x, kMxDpRawToX16, statistic_second.x) * probability.z;
    second.y =
        fmaf(second.y, kMxDpRawToX16, statistic_second.y) * probability.w;
    first.x *= beta;
    first.y *= beta;
    second.x *= beta;
    second.y *= beta;
    return d64::pack_owner_e4m3_word(first, second);
}

// Split only the first-half dP TMEM load into issue and consume operations.
__device__ __forceinline__ void issue_dp_half_compact(
    d64::owner_aligned_fp32_half &dp,
    const attention_tmem_tile &dp_tmem,
    int output_subtile,
    int column_half
) {
    d64::load_owner_aligned_fp32_half(
        dp,
        dp_tmem,
        output_subtile,
        column_half
    );
}

__device__ __forceinline__ void consume_ds_half_compact(
    d64::owner_aligned_fp32_half &dp,
    const compact_probability_half &compact,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int input_stage,
    float beta
) {
    // The wait remains at v443's original first dS consumer point.
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
    issue_dp_half_compact(dp, dp_tmem, output_subtile, column_half);
    consume_ds_half_compact(
        dp,
        compact,
        storage,
        output_subtile,
        column_half,
        input_stage,
        beta
    );
}

// v409 authenticates both the x16 TMEM address increment and lane ownership:
// offset Chunk*16 from first-half column zero, with each lane retaining eight
// float2 values.  Keep issue and wait separate so chunk-0 arithmetic can hide
// the second load without changing any numerical operation.
template <int Chunk>
__device__ __forceinline__ void issue_dp_first_half_quarter(
    owner_aligned_fp32_quarter &destination,
    const attention_tmem_tile &source,
    int output_subtile
) {
    static_assert(Chunk == 0 || Chunk == 1);
    const uint32_t address =
        source.addr +
        (static_cast<uint32_t>(output_subtile * 16) << 16) +
        Chunk * (kColumnHalf / 4);
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, "
        "[%16], 32;\n"
        : "=f"(destination.pairs[0].x),
          "=f"(destination.pairs[0].y),
          "=f"(destination.pairs[1].x),
          "=f"(destination.pairs[1].y),
          "=f"(destination.pairs[2].x),
          "=f"(destination.pairs[2].y),
          "=f"(destination.pairs[3].x),
          "=f"(destination.pairs[3].y),
          "=f"(destination.pairs[4].x),
          "=f"(destination.pairs[4].y),
          "=f"(destination.pairs[5].x),
          "=f"(destination.pairs[5].y),
          "=f"(destination.pairs[6].x),
          "=f"(destination.pairs[6].y),
          "=f"(destination.pairs[7].x),
          "=f"(destination.pairs[7].y)
        : "r"(address)
        : "memory"
    );
}

__device__ __forceinline__ uint32_t make_ds_quarter_compact_word(
    uint32_t probability_word,
    const owner_aligned_fp32_quarter &dp,
    const stats_tile &dstat,
    int query_column_base,
    int local_word,
    float beta
) {
    const int first_pair = 2 * local_word;
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
    constexpr float kMxDpRawToX16 = 2.0f / 3.0f;
    first.x = fmaf(first.x, kMxDpRawToX16, statistic_first.x) * probability.x;
    first.y = fmaf(first.y, kMxDpRawToX16, statistic_first.y) * probability.y;
    second.x =
        fmaf(second.x, kMxDpRawToX16, statistic_second.x) * probability.z;
    second.y =
        fmaf(second.y, kMxDpRawToX16, statistic_second.y) * probability.w;
    first.x *= beta;
    first.y *= beta;
    second.x *= beta;
    second.y *= beta;
    return d64::pack_owner_e4m3_word(first, second);
}

template <int Chunk>
__device__ __forceinline__ void consume_ds_first_half_quarter(
    const owner_aligned_fp32_quarter &dp,
    const compact_probability_half &compact,
    shared_storage &storage,
    int output_subtile,
    int input_stage,
    float beta
) {
    static_assert(Chunk == 0 || Chunk == 1);
    auto destination = storage.ds.template subtile<16, kColumnHalf>(
        {output_subtile, 0}
    );
    const int lane_row = laneid() & 15;
    const int lane_column_base = 32 * (laneid() >> 4);
    constexpr int kChunkColumns = kColumnHalf / 4;
    constexpr int kWordsPerChunk = kCompactProbabilityWords / 2;
    const int query_column_base =
        lane_column_base + Chunk * kChunkColumns;
    const uint32_t shared_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&destination.data[0])
    );
#pragma unroll
    for (int word_pair = 0; word_pair < kWordsPerChunk / 2;
         ++word_pair) {
        const int first_word = 2 * word_pair;
        const int second_word = first_word + 1;
        constexpr int kCompactWordBase = Chunk * kWordsPerChunk;
        const uint32_t ds_first = make_ds_quarter_compact_word(
            compact.words[kCompactWordBase + first_word],
            dp,
            storage.dstat[input_stage],
            query_column_base,
            first_word,
            beta
        );
        const uint32_t ds_second = make_ds_quarter_compact_word(
            compact.words[kCompactWordBase + second_word],
            dp,
            storage.dstat[input_stage],
            query_column_base,
            second_word,
            beta
        );
        const int column =
            lane_column_base + Chunk * kChunkColumns + 4 * first_word;
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

// Split the four K32 shared/shared dV commands without changing their
// descriptor mapping.  Half 0 owns chunks 0-1 and uses v436's runtime
// accumulate predicate on the first command; half 1 owns chunks 2-3, always
// accumulates, and is the sole dv_ready completion commit.  The issue loop may
// commit the next score MMA between halves; tensor-command ordering puts the
// already-issued half 0 before that group and the later half 1/dv_ready group.
template <int Half>
__device__ __forceinline__ void issue_gradient_ab_runtime_accumulate_half(
    gradient_tmem_tile &destination,
    const attention_tile &lhs,
    const operand_tile &rhs,
    semaphore &completion,
    bool accumulate
) {
    static_assert(Half == 0 || Half == 1);
    using input_type = typename attention_tile::T;
    using output_type = typename gradient_tmem_tile::T;
    static_assert(std::is_same_v<input_type, fp8e4m3>);
    static_assert(std::is_same_v<input_type, typename operand_tile::T>);
    static_assert(attention_tile::rows == gradient_tmem_tile::rows);
    static_assert(operand_tile::cols == gradient_tmem_tile::cols);
    static_assert(attention_tile::cols == operand_tile::rows);
    static_assert(attention_tile::cols == 4 * 32);
    constexpr uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            attention_tile::rows,
            operand_tile::cols,
            transpose::N,
            transpose::T,
            false
        >();

    if (warpgroup::laneid() == 0) {
        ::kittens::st_descriptor<attention_tile, transpose::N> lhs_desc(
            lhs
        );
        ::kittens::st_descriptor<operand_tile, transpose::T> rhs_desc(rhs);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        constexpr int kFirstChunk = Half * 2;
        if constexpr (Half == 0) {
            const uint32_t accumulate_value = accumulate ? 1u : 0u;
            asm volatile(
                "{\n\t"
                ".reg .pred accumulate_pred;\n\t"
                "setp.ne.u32 accumulate_pred, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f8f6f4 "
                "[%0], %1, %2, %3, accumulate_pred;\n\t"
                "}\n"
                :: "r"(destination.addr),
                   "l"(lhs_desc.chunk_descriptor(kFirstChunk)),
                   "l"(rhs_desc.chunk_descriptor(2 * kFirstChunk)),
                   "r"(instruction),
                   "r"(accumulate_value)
                : "memory"
            );
        } else {
            ::kittens::detail::tcgen05::template st_st<
                input_type,
                1,
                1
            >(
                destination.addr,
                lhs_desc.chunk_descriptor(kFirstChunk),
                rhs_desc.chunk_descriptor(2 * kFirstChunk),
                instruction
            );
        }

        constexpr int kSecondChunk = kFirstChunk + 1;
        ::kittens::detail::tcgen05::template st_st<input_type, 1, 1>(
            destination.addr,
            lhs_desc.chunk_descriptor(kSecondChunk),
            rhs_desc.chunk_descriptor(2 * kSecondChunk),
            instruction
        );
    }
    if constexpr (Half == 1) {
        tensor_commit<1>(completion);
    }
}

// The drain path has already multiplied every FP32 accumulator by 1/256 and
// packed it to BF16 in `source`.  A plain TMA store therefore preserves the
// exact v440 epilogue scale; unlike store_add_async, it does not require a
// pre-zeroed destination.  This helper is valid only for the owner-4 dK/dV
// publications, whose destination tiles each have one globally unique CTA.
__device__ __forceinline__ void publish_gradient_full_direct(
    const globals::gradient_gl &destination,
    gradient_full_tile &source,
    semaphore &ready,
    int phase,
    int batch,
    int sequence_tile,
    int head,
    int logical_warp,
    int lane
) {
    if (logical_warp != 0) {
        return;
    }
    wait(ready, phase);
    tensor_after_thread_sync();
    if (lane == 0) {
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            destination,
            source,
            coord<gradient_full_tile>{
                batch,
                sequence_tile,
                head,
                0,
            }
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

// Warp 14 owns the complete TMA bulk-group stream and signals this barrier
// only after its source read has retired.  Reducers are consumers only: they
// must not execute a wait instruction for a bulk group issued by another
// thread, and they must not contribute an additional barrier arrival.
__device__ __forceinline__ void wait_gradient_publisher_reuse(
    semaphore &reusable,
    int phase
) {
    wait(reusable, phase);
    tensor_after_thread_sync();
}

// Expand one packed [S128,D/2] MXFP4 tile into the aligned E2M1 byte-container
// layout required by mxf8f6f4.  Every 8-byte payload segment is followed by an
// 8-byte hole.  All 512 CTA threads cooperate; V is persistent for the CTA.
__device__ __forceinline__ void stage_persistent_mxfp4_v_and_scales(
    const globals &g,
    shared_storage &storage,
    mx_scale_tile &v_scale,
    mx_scale_tile &dout_unity_scale,
    int batch,
    int key_tile,
    int kv_head
) {
    constexpr int kPackedBytesPerRow = kDepth / 2;
    constexpr int kPayloadBytesPerSegment = 8;
    constexpr int kAlignedBytesPerSegment = 16;
    constexpr int kSegmentsPerRow =
        kPackedBytesPerRow / kPayloadBytesPerSegment;
    constexpr int kSegmentCount = kKeyTile * kSegmentsPerRow;
    static_assert(kPackedBytesPerRow == 64);
    static_assert(kSegmentsPerRow == 8);
    static_assert(kSegmentCount == 1024);

    const auto *v_bytes = g.v_backward_mxfp4.raw_ptr;
    const size_t first_row =
        (static_cast<size_t>(batch) * g.sequence + key_tile * kKeyTile) *
            kKvHeads +
        kv_head;
    for (int segment = static_cast<int>(threadIdx.x);
         segment < kSegmentCount;
         segment += kThreads) {
        const int row = segment / kSegmentsPerRow;
        const int row_segment = segment - row * kSegmentsPerRow;
        const size_t source_offset =
            (first_row + static_cast<size_t>(row) * kKvHeads) *
                kPackedBytesPerRow +
            row_segment * kPayloadBytesPerSegment;
        const uint64_t payload = *reinterpret_cast<const uint64_t *>(
            v_bytes + source_offset
        );
        auto *destination = operand_tile::idx(
            &storage.v.data[0],
            int2{row, row_segment * kAlignedBytesPerSegment}
        );
        *reinterpret_cast<uint64_t *>(destination) = payload;
    }

    if (threadIdx.x < 128) {
        const size_t scale_page =
            (static_cast<size_t>(batch) * (g.sequence / kKeyTile) + key_tile) *
                kKvHeads +
            kv_head;
        const auto *source_words = reinterpret_cast<const uint32_t *>(
            g.v_backward_mxfp4_scales.raw_ptr + scale_page * 512
        );
        reinterpret_cast<uint32_t *>(&v_scale.data[0])[threadIdx.x] =
            source_words[threadIdx.x];
        reinterpret_cast<uint32_t *>(&dout_unity_scale.data[0])[threadIdx.x] =
            0x7f7f7f7fu;
    }
}

// Load the physical 512-byte SFA/SFB pages into two temporarily aliased TMEM
// slots.  The caller must prove the score tile at columns [384,416) is dead.
__device__ __forceinline__ void stage_mixed_dp_scale_tmem(
    mx_scale_tmem_tile &v_scale_tmem,
    mx_scale_tmem_tile &dout_scale_tmem,
    const mx_scale_tile &v_scale,
    const mx_scale_tile &dout_scale
) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    load_mxnv_scale_async<1>(v_scale_tmem, v_scale);
    load_mxnv_scale_async<1>(dout_scale_tmem, dout_scale);
    tensor_load_wait();
    // Match the proven block-scaled issue paths: the completed asynchronous
    // TMEM loads must cross the tensor-proxy before-thread-sync fence before
    // a subsequent MMA consumes their scale addresses.
    tensor_before_thread_sync();
}

// Native SM100 mixed block-scaled dP: aligned E2M1 V times resident E4M3 dO.
// Descriptor formats are independent (A=5 E2M1, B=0 E4M3), scales are E8M0,
// and dense reduction is K32.  dP is ABt, so both operands use K-major
// descriptors and advance one K32 chunk per instruction.  SFIDs 0..3 select
// the four D128 scale groups.
__device__ __forceinline__ void issue_mxfp4v_e4m3do_dp(
    attention_tmem_tile &destination,
    const operand_tile &v_aligned,
    const operand_tile &dout,
    const mx_scale_tmem_tile &v_scale,
    const mx_scale_tmem_tile &dout_scale,
    semaphore &completion
) {
    constexpr uint32_t kInstructionBase =
        (5u << 7) |       // A format: E2M1
        (0u << 10) |      // B format: E4M3
        (16u << 17) |     // N / 8 = 16
        (1u << 23) |      // E8M0 scale factors
        (8u << 24);       // M / 16 = 8
    static_assert(kInstructionBase == 0x08A00280u);

    ::kittens::st_descriptor<operand_tile, transpose::N> v_desc(v_aligned);
    ::kittens::st_descriptor<operand_tile, transpose::N> dout_desc(dout);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        const uint32_t instruction =
            kInstructionBase |
            (static_cast<uint32_t>(chunk) << 29) |
            (static_cast<uint32_t>(chunk) << 4);
        const uint32_t accumulate = chunk != 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
            "}\n"
            :: "r"(destination.addr),
               "l"(v_desc.chunk_descriptor(chunk)),
               "l"(dout_desc.chunk_descriptor(chunk)),
               "r"(instruction),
               "r"(v_scale.addr),
               "r"(dout_scale.addr),
               "r"(accumulate)
            : "memory"
        );
    }
    tensor_commit<1>(completion);
}

__global__ __launch_bounds__(kThreads, 1)
void owner4_kernel(const __grid_constant__ globals g) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) mx_scale_tile v_scale_shared;
    __shared__ alignas(16) mx_scale_tile dout_unity_scale_shared;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready[kInputStages];
    __shared__ alignas(16) semaphore stats_ready[kInputStages];
    __shared__ alignas(16) semaphore operand_consumed[kInputStages];
    __shared__ alignas(16) semaphore score_ready;
    __shared__ alignas(16) semaphore score_consumed;
    __shared__ alignas(16) semaphore probability_half_ready[2];
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
    const int batch = linear_owner / kHeadOwners;
    const int head_owner = linear_owner - batch * kHeadOwners;
    const int key_tile = static_cast<int>(blockIdx.y);
    const int kv_head = head_owner / kOwnersPerKvHead;
    const int iterations_per_head =
        prior::kExactQueryTiles - key_tile;
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
        }
        init_semaphore(score_ready, 0, 1);
        init_semaphore(score_consumed, 0, kComputeWarps);
        init_semaphore(probability_half_ready[0], 0, kComputeWarps);
        init_semaphore(probability_half_ready[1], 0, kComputeWarps);
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

    stage_persistent_mxfp4_v_and_scales(
        g,
        storage,
        v_scale_shared,
        dout_unity_scale_shared,
        batch,
        key_tile,
        kv_head
    );
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
    // These scale pages intentionally alias the first 32 columns of score.
    // Every use is bracketed by score_consumed and dp_ready below.
    mx_scale_tmem_tile v_scale_tmem(kScoreTmemOffset);
    mx_scale_tmem_tile dout_scale_tmem(kScoreTmemOffset + 16);

    if (physical_warp == kLoaderWarp && lane == 0) {
        tma::expect_bytes(
            persistent_ready,
            sizeof(storage.k)
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k,
            g.k,
            coord<operand_tile>{batch, key_tile, kv_head, 0},
            persistent_ready
        );
        int work = 0;
#pragma unroll 1
        for (int local_head = 0; local_head < kHeadsPerOwner; ++local_head) {
            const int query_head =
                kHeadsPerOwner * head_owner + local_head;
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
                    // dV half 1 commits dv_ready after all four K32 commands,
                    // so completion proves no tensor consumer still reads the
                    // old shared-P tile.  dS reuses compact registers instead.
                    wait(dv_ready, old_phase);
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
                tensor_before_thread_sync();
                __syncwarp();
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
                if (lane == 0) {
                    arrive(probability_half_ready[0]);
                }
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
                    arrive(probability_half_ready[1]);
                }

                owner_aligned_fp32_quarter dp_first_quarter;
                wait(dp_ready, phase);
                // PTX requires the consumer-side tensor-proxy acquire after
                // the MMA completion wait and before an asynchronous TMEM
                // load issued by this different warp.
                tensor_after_thread_sync();
                issue_dp_first_half_quarter<0>(
                    dp_first_quarter,
                    dp_tmem,
                    output_subtile
                );
                if (work > 0) {
                    const int old_phase =
                        x32::iteration_phase(work - 1);
                    // dQ then dK commands and commits use the same tensor lane.
                    // Retained dK completion therefore proves dQ also stopped
                    // reading old shared dS; reducers still wait on dq_ready.
                    wait(dk_ready, old_phase);
                }
                // Chunk 0 is now available.  Launch chunk 1 before consuming
                // it so the first quarter's affine/E4M3 publication overlaps
                // the second x16 TMEM load.
                tensor_load_wait();
                owner_aligned_fp32_quarter dp_second_quarter;
                issue_dp_first_half_quarter<1>(
                    dp_second_quarter,
                    dp_tmem,
                    output_subtile
                );
                // Make the chunk-1 asynchronous TMEM load visible to the
                // thread before beginning chunk-0 scalar work.  This is only
                // tcgen05.fence::before_thread_sync; no warp barrier is needed.
                tensor_before_thread_sync();
                consume_ds_first_half_quarter<0>(
                    dp_first_quarter,
                    probability_compact[0],
                    storage,
                    output_subtile,
                    stage,
                    g.beta
                );
                tensor_load_wait();
                consume_ds_first_half_quarter<1>(
                    dp_second_quarter,
                    probability_compact[0],
                    storage,
                    output_subtile,
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
                // Scale factors alias score columns [384,416).  The final dP
                // of the prior head has no within-head successor to wait for
                // it, so prove that read dead before the next head replaces
                // those columns with a score tile.
                wait(dp_ready, previous_phase);
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
            if (local_head > 0) {
                const int previous_phase =
                    x32::iteration_phase(work - 1);
                wait(dq_tmem_drained, previous_phase);
                tensor_after_thread_sync();
            }
            // Score owns [384,512).  Wait until all compute warps have loaded
            // it, then reuse [384,416) for the two mixed-dP scale pages.
            const int first_phase = x32::iteration_phase(work);
            wait(score_consumed, first_phase);
            tensor_after_thread_sync();
            stage_mixed_dp_scale_tmem(
                v_scale_tmem,
                dout_scale_tmem,
                v_scale_shared,
                dout_unity_scale_shared
            );
            issue_mxfp4v_e4m3do_dp(
                dp_tmem,
                storage.v,
                storage.dout[first_stage],
                v_scale_tmem,
                dout_scale_tmem,
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

                wait(probability_half_ready[0], phase);
                tensor_after_thread_sync();
                issue_gradient_ab_runtime_accumulate_half<0>(
                    dv_tmem,
                    storage.probability,
                    storage.dout[stage],
                    dv_ready,
                    work != 0
                );

                // The current dP completion is the earliest proof that its
                // scale pages at score columns [384,416) are dead.  Refill the
                // complete score page immediately, while compute finishes P1
                // publication and dS.  Unlike v503, dp_ready is mandatory:
                // four independent D32 anchors remain live in block-scale MMA.
                if (has_next) {
                    wait(dp_ready, phase);
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

                wait(probability_half_ready[1], phase);
                tensor_after_thread_sync();
                issue_gradient_ab_runtime_accumulate_half<1>(
                    dv_tmem,
                    storage.probability,
                    storage.dout[stage],
                    dv_ready,
                    true
                );

                wait(ds_ready, phase);
                tensor_after_thread_sync();
                core::issue_gradient_atb(
                    dq_tmem,
                    storage.ds,
                    storage.k,
                    dq_ready
                );
                prior::issue_gradient_ab_runtime_accumulate(
                    dk_tmem,
                    storage.ds,
                    storage.q[stage],
                    dk_ready,
                    work != 0
                );

                if (has_next) {
                    const int next_phase = x32::iteration_phase(next_work);
                    wait(score_consumed, next_phase);
                    tensor_after_thread_sync();
                    // The next score is dead, so restore its exact four-anchor
                    // scale pages while reducers are still draining current
                    // dQ from the aliased dP/dQ destination columns.
                    stage_mixed_dp_scale_tmem(
                        v_scale_tmem,
                        dout_scale_tmem,
                        v_scale_shared,
                        dout_unity_scale_shared
                    );
                    wait(dq_tmem_drained, phase);
                    tensor_after_thread_sync();
                    issue_mxfp4v_e4m3do_dp(
                        dp_tmem,
                        storage.v,
                        storage.dout[next_stage],
                        v_scale_tmem,
                        dout_scale_tmem,
                        dp_ready
                    );
                }

                // The dK commit tracks both earlier dV halves because all
                // commands and commits are issued by this same tensor lane.
                wait(dk_ready, phase);
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
            for (
                int iteration = 0;
                iteration < iterations_per_head;
                ++iteration, ++work
            ) {
                const int phase = x32::iteration_phase(work);
                wait(dq_ready, phase);
                tensor_after_thread_sync();
                if (work > 0) {
                    wait_gradient_publisher_reuse(
                        full_gradient_reusable,
                        x32::iteration_phase(work - 1)
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
            }
        }

        const int last_phase =
            x32::iteration_phase(total_work - 1);
        wait(dk_ready, last_phase);
        tensor_after_thread_sync();
        wait_gradient_publisher_reuse(
            full_gradient_reusable,
            x32::iteration_phase(total_work - 1)
        );
        prior::drain_gradient_full_owner_x32(
            dk_tmem,
            storage.gradient,
            full_gradient_ready,
            logical_warp,
            lane
        );

        wait(dv_ready, last_phase);
        tensor_after_thread_sync();
        wait_gradient_publisher_reuse(
            full_gradient_reusable,
            x32::iteration_phase(total_work)
        );
        prior::drain_gradient_full_owner_x32(
            dv_tmem,
            storage.gradient,
            full_gradient_ready,
            logical_warp,
            lane
        );
    } else if (physical_warp == kGradientPublisherWarp && lane == 0) {
        int work = 0;
#pragma unroll 1
        for (int local_head = 0; local_head < kHeadsPerOwner; ++local_head) {
            const int query_head =
                kHeadsPerOwner * head_owner + local_head;
            for (
                int iteration = 0;
                iteration < iterations_per_head;
                ++iteration, ++work
            ) {
                const int phase = x32::iteration_phase(work);
                prior::publish_gradient_full(
                    g.dq,
                    storage.gradient,
                    dq_drained,
                    phase,
                    batch,
                    key_tile + iteration,
                    query_head,
                    0,
                    0
                );
                warp::tma::store_async_read_wait<0>();
                arrive(full_gradient_reusable);
            }
        }

        publish_gradient_full_direct(
            g.dk,
            storage.gradient,
            full_gradient_ready,
            0,
            batch,
            key_tile,
            kv_head,
            0,
            0
        );
        warp::tma::store_async_read_wait<0>();
        arrive(full_gradient_reusable);

        publish_gradient_full_direct(
            g.dv,
            storage.gradient,
            full_gradient_ready,
            1,
            batch,
            key_tile,
            kv_head,
            0,
            0
        );
        warp::tma::store_async_wait<0>();
        arrive(kernel_complete);
    }
}

inline globals make_globals(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v_backward_mxfp4,
    at::Tensor &v_backward_mxfp4_scales,
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
        kittens::py::tensor_to_gl<byte_gl, false>(v_backward_mxfp4),
        kittens::py::tensor_to_gl<byte_gl, false>(
            v_backward_mxfp4_scales
        ),
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
    at::Tensor &v_backward_mxfp4,
    at::Tensor &v_backward_mxfp4_scales,
    at::Tensor &dout,
    at::Tensor &lstat,
    at::Tensor &dstat,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float softmax_scale,
    cudaStream_t stream
) {
    TORCH_CHECK(
        q.size(0) == 2 && q.size(1) == prior::kExactSequence,
        "v507 shared-tile MXFP4-V backward is fail-closed to B2/S4096"
    );

    const globals g = make_globals(
        q,
        k,
        v_backward_mxfp4,
        v_backward_mxfp4_scales,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        softmax_scale
    );
    const dim3 grid(
        static_cast<unsigned int>(kHeadOwners * q.size(0)),
        static_cast<unsigned int>(q.size(1) / kKeyTile),
        1
    );
    owner4_kernel<<<grid, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v507_d128_gqa_mxfp4v_sharedtile_e4m3do_b2_s4096_owner4_experimental_bshd

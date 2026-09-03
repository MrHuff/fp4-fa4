#pragma once

#include "v438_d128_gqa_e4m3_b2_owner2_exact_s4096_production_bshd.cuh"

// Fail-closed B1/S4096 hybrid experiment reusing v488's owner-2 schedule,
// E4M3 Q/K gradient operands, and E4M3 V/dO. Score recomputation alone
// consumes the exact forward NVFP4 Q/K payload, row-K16 E4M3 scale pages,
// and per-head global scales. There is deliberately no shape fallback and
// no production dispatcher includes this file.
//
// The inherited schedule reuses v439's retained K/V,
// FP32 dK/dV ownership, exact sequence schedule, runtime first E4M3
// gradient-MMA accumulate predicate, and additive dQ/dK/dV publication.
// Preserve v445's exact rounded-E4M3 compact P words in registers from dV
// publication through dS.  Split dQ's aliased TMEM lifetime from its shared
// publication lifetime: capture the complete exact BF16 payload in registers,
// release TMEM for the next dP, then publish the shared tile. All other shapes
// are rejected by the standalone wrapper.
namespace tkfa4::native_gqa_tk_bwd::v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_experimental_bshd {

namespace exact =
    tkfa4::native_gqa_tk_bwd::v436_d128_gqa_e4m3_exact_s4096_runtime_accumulate_production_bshd;
namespace prior =
    tkfa4::native_gqa_tk_bwd::v433_d128_gqa_e4m3_head_fast_raster_production_bshd;
namespace x32 =
    tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd;
namespace core =
    tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd;
namespace d64 =
    tkfa4::native_gqa_tk_bwd::v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds;

using prior::attention_tmem_tile;
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

// Exact forward publications are BHSD FP4 payloads. Each 128-row Q/K tile
// contains 64 packed bytes per row. The two 512-byte scale pages correspond
// to the two D64 chunks and retain the projection epilogue's row-K16 E4M3
// scale representation without an FP4 -> E4M3 value lift.
using native_qk_tile = st_fp4e2m1_2<kKeyTile, core::kDepth / 2>;
using native_qk_scale_tile = st_hf<core::kDepth / 64, 256, false>;
static_assert(sizeof(native_qk_tile) == 8 * 1024);
static_assert(sizeof(native_qk_scale_tile) == 1024);

struct globals {
    using operand_gl = prior::globals::operand_gl;
    using gradient_gl = prior::globals::gradient_gl;
    using stats_gl = prior::globals::stats_gl;
    using native_qk_gl = gl<
        fp4e2m1_2,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<native_qk_tile, dim::ROW>
    >;
    // Scale publications are byte-valued E4M3, but TMA moves each physical
    // 512-byte page through a 256-half carrier exactly as the forward kernel.
    using native_qk_scale_gl = gl<
        half,
        -1,
        -1,
        -1,
        256,
        tma::descriptor<native_qk_scale_tile, dim::ROW>
    >;
    using global_scale_gl = gl<float, -1, -1, -1, -1>;

    // Existing represented-E4 operands remain the gradient path.
    operand_gl q;
    operand_gl k;
    operand_gl v;
    operand_gl dout;
    gradient_gl dq;
    gradient_gl dk;
    gradient_gl dv;
    stats_gl lstat;
    stats_gl dstat;

    // Exact forward score operands and scale metadata.
    native_qk_gl q_native;
    native_qk_gl k_native;
    native_qk_scale_gl q_native_scale;
    native_qk_scale_gl k_native_scale;
    global_scale_gl q_global_scale;
    global_scale_gl k_global_scale;

    // beta retains v488's E4x4 dP/dS convention. Score has no /16 lift;
    // its per-head global scales are applied separately before softmax.
    float beta;
    float softmax_scale_log2e;
    int sequence;
};

struct native_score_shared_storage {
    native_qk_tile k;
    native_qk_scale_tile k_scale;
    native_qk_tile q[kInputStages];
    native_qk_scale_tile q_scale[kInputStages];
};
static_assert(sizeof(shared_storage) == 162 * 1024);
static_assert(
    sizeof(shared_storage) + sizeof(native_score_shared_storage) < 232448
);

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

template <typename ST>
__device__ __forceinline__ uint32_t native_fp4_smem_descriptor_low(
    const ST &tile
) {
    static_assert(ST::swizzle);
    const uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&tile.data[0])
    );
    return ((smem_addr & 0x3ffffu) >> 4) | (1u << 16);
}

template <typename ST>
__device__ __forceinline__ uint32_t native_scale_smem_descriptor_low(
    const ST &tile
) {
    static_assert(!ST::swizzle);
    const uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&tile.data[0])
    );
    return ((smem_addr & 0x3ffffu) >> 4) | (8u << 16);
}

template <typename ST>
consteval uint32_t native_fp4_smem_descriptor_high() {
    constexpr uint32_t stride_bytes =
        ST::swizzle_bytes == 128 ? 1024 :
        ST::swizzle_bytes == 64 ? 512 : 256;
    constexpr uint32_t swizzle_mode =
        ST::swizzle_bytes == 128 ? 1 :
        ST::swizzle_bytes == 64 ? 2 : 3;
    return (stride_bytes >> 4) |
           (1u << (46 - 32)) |
           (swizzle_mode << (62 - 32));
}

template <typename ST>
consteval uint32_t native_fp4_k64_descriptor_offset() {
    if constexpr (
        ST::swizzle_bytes == 128 || ST::swizzle_bytes == 64
    ) {
        return 32 >> 4;
    } else {
        return (ST::rows / kittens::TILE_ROW_DIM<typename ST::T>) *
               (512 >> 4);
    }
}

// Verbatim group-1 scale-copy + two-K64 issue shape from the authenticated
// HAO forward path, specialized to score = K @ Q^T so v488's key-row/query-
// column score orientation remains unchanged. Scale TMEM aliases dP/dQ at
// columns [0,16); the issue schedule below waits for score completion before
// allowing dP to overwrite that range.
__device__ __forceinline__ void issue_native_nvfp4_score(
    attention_tmem_tile &score,
    native_qk_tile &lhs,
    native_qk_tile &rhs,
    native_qk_scale_tile &lhs_scale_smem,
    native_qk_scale_tile &rhs_scale_smem,
    semaphore &finished
) {
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e4m3;
    using T_D = typename attention_tmem_tile::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, kKeyTile, kQueryTile, false, 0>();
    static_assert(native_qk_tile::swizzle_bytes == native_qk_tile::swizzle_bytes);
    constexpr uint32_t DESC_HI =
        native_fp4_smem_descriptor_high<native_qk_tile>();
    constexpr uint32_t K64_DESC_OFFSET =
        native_fp4_k64_descriptor_offset<native_qk_tile>();
    const uint32_t lhs_desc_low = native_fp4_smem_descriptor_low(lhs);
    const uint32_t rhs_desc_low = native_fp4_smem_descriptor_low(rhs);
    const uint32_t lhs_scale_desc_low =
        native_scale_smem_descriptor_low(lhs_scale_smem);
    const uint32_t rhs_scale_desc_low =
        native_scale_smem_descriptor_low(rhs_scale_smem);
    constexpr uint32_t SCALE_DESC_HI = 8u | (1u << (46 - 32));
    const uint32_t scale_tmem_base = kDpDqTmemOffset;

    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, lhs_lo, rhs_lo, idesc, sfa, sfb, "
        "scale_hi, scale_lo, scale_dst;\n\t"
        ".reg .b64 lhs_desc, rhs_desc, scale_desc;\n\t"
        "mov.b32 scale_hi, %9;\n\t"
        "mov.b32 scale_lo, %7;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%3], scale_desc;\n\t"
        "add.u32 scale_lo, %7, 32;\n\t"
        "add.u32 scale_dst, %3, 4;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 scale_lo, %8;\n\t"
        "add.u32 scale_dst, %3, 8;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "add.u32 scale_lo, %8, 32;\n\t"
        "add.u32 scale_dst, %3, 12;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 lhs_desc, {%1, desc_hi};\n\t"
        "mov.b64 rhs_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], lhs_desc, rhs_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 lhs_lo, %1, %6;\n\t"
        "add.u32 rhs_lo, %2, %6;\n\t"
        "add.u32 sfa, %3, 4;\n\t"
        "add.u32 sfb, %3, 12;\n\t"
        "mov.b64 lhs_desc, {lhs_lo, desc_hi};\n\t"
        "mov.b64 rhs_desc, {rhs_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], lhs_desc, rhs_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(lhs_desc_low),
           "r"(rhs_desc_low),
           "r"(scale_tmem_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET),
           "r"(lhs_scale_desc_low),
           "r"(rhs_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory"
    );
    detail::tcgen05::commit<1>(finished);
}

__device__ __forceinline__ float score_beta_log2e(
    const globals &g,
    int batch,
    int query_head,
    int kv_head
) {
    const float q_scale = g.q_global_scale[{batch, query_head, 0, 0}];
    const float k_scale = g.k_global_scale[{batch, kv_head, 0, 0}];
    return g.softmax_scale_log2e * q_scale * k_scale;
}

__global__ __launch_bounds__(kThreads, 1)
void b1_native_nvfp4_score_e4m3_gradient_exact_s4096_kernel(
    const __grid_constant__ globals g
) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(1024) native_score_shared_storage native_score;
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
            sizeof(storage.k) + sizeof(storage.v) +
                sizeof(native_score.k) + sizeof(native_score.k_scale)
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
        tma::load_async<dim::ROW, cache_policy::NORMAL>(
            native_score.k,
            g.k_native,
            coord<native_qk_tile>{batch, kv_head, key_tile, 0},
            persistent_ready
        );
        // K scale pages are physically published every 64 sequence rows. The
        // first duplicated page for this 128-row key tile is selected here;
        // the scale TMA's two-row tile spans the two D64 chunks.
        tma::load_async<dim::ROW, cache_policy::NORMAL>(
            native_score.k_scale,
            g.k_native_scale,
            coord<native_qk_scale_tile>{
                batch, 2 * key_tile, kv_head, 0
            },
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
                        sizeof(storage.dout[stage]) +
                        sizeof(native_score.q[stage]) +
                        sizeof(native_score.q_scale[stage])
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
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    native_score.q[stage],
                    g.q_native,
                    coord<native_qk_tile>{
                        batch, query_head, query_tile, 0
                    },
                    query_ready[stage]
                );
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    native_score.q_scale[stage],
                    g.q_native_scale,
                    coord<native_qk_scale_tile>{
                        batch, query_tile, query_head, 0
                    },
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
            const int query_head =
                kHeadsPerOwner * head_pair + local_head;
            const float native_score_beta_log2e = score_beta_log2e(
                g, batch, query_head, kv_head
            );
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
                    native_score_beta_log2e,
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
                    native_score_beta_log2e,
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
            issue_native_nvfp4_score(
                score_tmem,
                native_score.k,
                native_score.q[first_stage],
                native_score.k_scale,
                native_score.q_scale[first_stage],
                score_ready
            );
            // Native scale factors alias columns [0,16) of dP/dQ TMEM.
            // Complete score before the dense-E4 dP issue overwrites them.
            wait(score_ready, x32::iteration_phase(work));
            tensor_after_thread_sync();
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
                    wait(score_consumed, phase);
                    wait(
                        query_ready[next_stage],
                        x32::input_stage_epoch_phase(next_work)
                    );
                    tensor_after_thread_sync();
                    issue_native_nvfp4_score(
                        score_tmem,
                        native_score.k,
                        native_score.q[next_stage],
                        native_score.k_scale,
                        native_score.q_scale[next_stage],
                        score_ready
                    );
                    wait(
                        score_ready,
                        x32::iteration_phase(next_work)
                    );
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
    at::Tensor &q_native,
    at::Tensor &k_native,
    at::Tensor &q_native_scale,
    at::Tensor &k_native_scale,
    at::Tensor &q_global_scale,
    at::Tensor &k_global_scale,
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
        kittens::py::tensor_to_gl<globals::native_qk_gl>(q_native),
        kittens::py::tensor_to_gl<globals::native_qk_gl>(k_native),
        kittens::py::tensor_to_gl<globals::native_qk_scale_gl, false>(
            q_native_scale,
            1,
            kExactSequence / kQueryTile,
            kQueryHeads * (core::kDepth / 64),
            256
        ),
        kittens::py::tensor_to_gl<globals::native_qk_scale_gl, false>(
            k_native_scale,
            1,
            kExactSequence / 64,
            kKvHeads * (core::kDepth / 64),
            256
        ),
        kittens::py::tensor_to_gl<globals::global_scale_gl>(
            q_global_scale, 1, kQueryHeads, 1, 1
        ),
        kittens::py::tensor_to_gl<globals::global_scale_gl>(
            k_global_scale, 1, kKvHeads, 1, 1
        ),
        beta,
        softmax_scale * kLog2E,
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
    at::Tensor &q_native,
    at::Tensor &k_native,
    at::Tensor &q_native_scale,
    at::Tensor &k_native_scale,
    at::Tensor &q_global_scale,
    at::Tensor &k_global_scale,
    float softmax_scale,
    cudaStream_t stream
) {
    TORCH_CHECK(
        q.size(0) == 1 && q.size(1) == kExactSequence,
        "v508 is fail-closed to B1/S4096"
    );

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
        q_native,
        k_native,
        q_native_scale,
        k_native_scale,
        q_global_scale,
        k_global_scale,
        softmax_scale
    );
    const dim3 grid(
        static_cast<unsigned int>(kHeadPairs * q.size(0)),
        static_cast<unsigned int>(q.size(1) / kKeyTile),
        1
    );
    b1_native_nvfp4_score_e4m3_gradient_exact_s4096_kernel<<<
        grid, kThreads, 0, stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_experimental_bshd

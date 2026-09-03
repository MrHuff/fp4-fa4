#pragma once

#include "v436_d128_gqa_e4m3_exact_s4096_runtime_accumulate_production_bshd.cuh"
#include "v437_d128_gqa_e4m3_b2_owner2_production_bshd.cuh"

// B2/S4096 quarter-dP scheduling experiment derived only from frozen v455.
// Preserve its owner-4/direct-dK-dV topology, compact exact-rounded E4M3 P
// reuse, split-dV publication, score(next), and all second-half/dQ/dK work.
// Only first-half dP changes: preissue owner-aligned x16 chunk 0 across the old
// dQ/dK waits, wait for it, issue x16 chunk 1, warp-sync, compute/publish
// chunk 0 while chunk 1 is in flight, then wait for and publish chunk 1.
// B1 falls back to v436; other B2 sequence lengths fall back to v437.
namespace tkfa4::native_gqa_tk_bwd::v461_d128_gqa_e4m3_b2_s4096_owner4_quarter_dp_split_dv_compact_p_reuse_production_bshd {

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

constexpr int kHeadsPerOwner = 4;
constexpr int kHeadOwners = kQueryHeads / kHeadsPerOwner;
constexpr int kOwnersPerKvHead = kHeadRatio / kHeadsPerOwner;
static_assert(kQueryHeads == 32 && kKvHeads == 8);
static_assert(kHeadsPerOwner == 4 && kHeadOwners == 8);
static_assert(kOwnersPerKvHead == 1);
static_assert(prior::kExactSequence == 4096);
static_assert(prior::kExactQueryTiles == 32);

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

__global__ __launch_bounds__(kThreads, 1)
void owner4_kernel(const __grid_constant__ globals g) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready[kInputStages];
    __shared__ alignas(16) semaphore stats_ready[kInputStages];
    __shared__ alignas(16) semaphore operand_consumed[kInputStages];
    __shared__ alignas(16) semaphore stats_consumed[kInputStages];
    __shared__ alignas(16) semaphore score_ready;
    __shared__ alignas(16) semaphore score_consumed;
    __shared__ alignas(16) semaphore probability_half_ready[2];
    __shared__ alignas(16) semaphore probability_consumed;
    __shared__ alignas(16) semaphore dp_ready;
    __shared__ alignas(16) semaphore dv_ready;
    __shared__ alignas(16) semaphore ds_ready;
    __shared__ alignas(16) semaphore dq_ready;
    __shared__ alignas(16) semaphore dk_ready;
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
            init_semaphore(stats_consumed[stage], 0, kComputeWarps);
        }
        init_semaphore(score_ready, 0, 1);
        init_semaphore(score_consumed, 0, kComputeWarps);
        init_semaphore(probability_half_ready[0], 0, kComputeWarps);
        init_semaphore(probability_half_ready[1], 0, kComputeWarps);
        init_semaphore(probability_consumed, 0, kComputeWarps);
        init_semaphore(dp_ready, 0, 1);
        init_semaphore(dv_ready, 0, 1);
        init_semaphore(ds_ready, 0, kComputeWarps);
        init_semaphore(dq_ready, 0, 1);
        init_semaphore(dk_ready, 0, 1);
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
                    wait(dq_ready, old_phase);
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
                // Scheduling-only warp barrier: deliberately no tensor-
                // proxy before-thread fence and no numerical operation.
                __syncwarp();
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
                wait(dq_drained, previous_phase);
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

                wait(probability_half_ready[0], phase);
                tensor_after_thread_sync();
                issue_gradient_ab_runtime_accumulate_half<0>(
                    dv_tmem,
                    storage.probability,
                    storage.dout[stage],
                    dv_ready,
                    work != 0
                );

                // P1 releases the score page immediately after its TMEM load.
                // Refill that page while compute finishes P1 math/publication;
                // dS0 still remains after the complete P1 producer call.
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
                    wait(dq_drained, phase);
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
                kHeadsPerOwner * head_owner + local_head;
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
                prior::drain_gradient_full_owner_x32(
                    dq_tmem,
                    storage.gradient,
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
        publish_gradient_full_direct(
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
        publish_gradient_full_direct(
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
    if (q.size(0) != 2) {
        prior::launch(
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
    if (q.size(1) != prior::kExactSequence) {
        fallback_owner2::launch(
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
        static_cast<unsigned int>(kHeadOwners * q.size(0)),
        static_cast<unsigned int>(q.size(1) / kKeyTile),
        1
    );
    owner4_kernel<<<grid, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v461_d128_gqa_e4m3_b2_s4096_owner4_quarter_dp_split_dv_compact_p_reuse_production_bshd

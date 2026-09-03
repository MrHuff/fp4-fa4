#pragma once

#include "v387_d64_gqa_e4m3_async_pipeline.cuh"
#include "v389_d64_gqa_e4m3_alu_exp2_period2.cuh"

// v403 plus v404's independently validated dQ-first tensor-issue schedule.
// Retain v399's owner-aligned score/dP load,
// packed-P TMEM store, and dS shared publication.  Preserve v400's exact FP32
// P lifetime, per-half dV overlap, clamp/native-EX2 policy, raw-stat TMA ABI,
// fused stat scaling, async gradient publication, and sole score/P reuse point.
// The only deliberate change from v406 is the authenticated CuTe public
// degree-1/native period-2 exp2 policy already encoded in v389.  dQ aliases
// the retired dP TMEM allocation, dK remains disjoint, and dq_drained remains
// the sole gate before the next dP overwrite.
namespace tkfa4::native_gqa_tk_bwd::v413_d64_gqa_e4m3_owner_aligned_dq_first_period2 {

namespace base =
    tkfa4::native_gqa_tk_bwd::v385_d64_gqa_e4m3_k128q128;
namespace half =
    tkfa4::native_gqa_tk_bwd::v386_d64_gqa_e4m3_k128q128_halfcols;
namespace predecessor =
    tkfa4::native_gqa_tk_bwd::v387_d64_gqa_e4m3_async_pipeline;
namespace exp2_policy =
    tkfa4::native_gqa_tk_bwd::v389_d64_gqa_e4m3_alu_exp2_period2;

using base::attention_tile;
using base::attention_tmem_tile;
using base::gradient_stage_tile;
using base::gradient_tmem_tile;
using base::kDepth;
using base::kHeadRatio;
using base::kKeyTile;
using base::kKvHeads;
using base::kOperandScale;
using base::kQueryHeads;
using base::kQueryTile;
using base::operand_tile;
using predecessor::drain_gradient_to_bf16;
using predecessor::previous_stage_phase;
using predecessor::score_stage;
using predecessor::stage_phase;

constexpr int kColumnHalf = half::kColumnHalf;
constexpr int kThreads = predecessor::kThreads;
constexpr int kComputeWarps = predecessor::kComputeWarps;
constexpr int kReduceWarpBase = predecessor::kReduceWarpBase;
constexpr int kReduceWarps = predecessor::kReduceWarps;
constexpr int kTensorIssueWarp = predecessor::kTensorIssueWarp;
constexpr int kLoaderWarp = predecessor::kLoaderWarp;
constexpr int kStatsWarp = predecessor::kStatsWarp;
constexpr int kStages = predecessor::kStages;
constexpr int kGradientPublicationStages = 2;

constexpr int kDpDqTmemOffset = predecessor::kDpDqTmemOffset;
constexpr int kDkTmemOffset = predecessor::kDkTmemOffset;
constexpr int kDvTmemOffset = predecessor::kDvTmemOffset;
constexpr int kScoreTmemOffset = predecessor::kScoreTmemOffset;

using probability_tmem_tile = full_tt_fp8e4m3<kQueryTile>;
using probability_tmem_fragment = full_tt_fp8e4m3<kColumnHalf>;
using stats_tile = sv_fl<kQueryTile>;

struct globals {
    using operand_gl = base::globals::operand_gl;
    using gradient_gl = base::globals::gradient_gl;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    operand_gl q;
    operand_gl k;
    operand_gl v;
    operand_gl dout;
    gradient_gl dq;
    gradient_gl dk;
    gradient_gl dv;
    stats_gl l_aux;
    stats_gl delta;
    float beta;
    float beta_log2e;
    float l_aux_scale;
    int sequence;
};

struct shared_storage {
    operand_tile k;
    operand_tile v;
    operand_tile q[kStages];
    operand_tile dout[kStages];
    attention_tile ds[kStages];
    gradient_stage_tile gradient[kGradientPublicationStages];
    stats_tile lstat[kStages];
    stats_tile dstat[kStages];
};

// v394 is 100,352 bytes.  The sole storage increase is v392's second 16-KiB
// publication tile; P remains TMEM-resident and dS remains the same two-stage
// shared allocation.
static_assert(sizeof(shared_storage) <= 116736);
static_assert(sizeof(shared_storage) < 128 * 1024);

struct owner_aligned_fp32_half {
    float2 pairs[kColumnHalf / 2];
};

static_assert(sizeof(owner_aligned_fp32_half) == kColumnHalf * sizeof(float));

// The 16-data-path x32 FP32 load and x8 packed-E4M3 store have the same
// logical ownership: lane L owns row L%16 and columns 32*(L/16)+[0,32).
// Preserve v400's half-resident FP32-P lifetime while removing the generic
// fragment's cross-lane ownership exchange.
__device__ __forceinline__ void load_owner_aligned_fp32_half(
    owner_aligned_fp32_half &destination,
    const attention_tmem_tile &source,
    int output_subtile,
    int column_half
) {
    const uint32_t address =
        source.addr +
        (static_cast<uint32_t>(output_subtile * 16) << 16) +
        column_half * kColumnHalf;
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, "
        "[%32], 32;\n"
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
          "=f"(destination.pairs[7].y),
          "=f"(destination.pairs[8].x),
          "=f"(destination.pairs[8].y),
          "=f"(destination.pairs[9].x),
          "=f"(destination.pairs[9].y),
          "=f"(destination.pairs[10].x),
          "=f"(destination.pairs[10].y),
          "=f"(destination.pairs[11].x),
          "=f"(destination.pairs[11].y),
          "=f"(destination.pairs[12].x),
          "=f"(destination.pairs[12].y),
          "=f"(destination.pairs[13].x),
          "=f"(destination.pairs[13].y),
          "=f"(destination.pairs[14].x),
          "=f"(destination.pairs[14].y),
          "=f"(destination.pairs[15].x),
          "=f"(destination.pairs[15].y)
        : "r"(address)
        : "memory"
    );
}

__device__ __forceinline__ uint32_t pack_owner_e4m3_word(
    const float2 &lower,
    const float2 &upper
) {
    const uint32_t lower_e4m3 =
        base::donor::convert_f32_pair_to_e4m3(lower);
    const uint32_t upper_e4m3 =
        base::donor::convert_f32_pair_to_e4m3(upper);
    uint32_t packed;
    asm volatile(
        "shl.b32 %0, %2, 16;\n"
        "or.b32 %0, %0, %1;\n"
        : "=&r"(packed)
        : "r"(lower_e4m3), "r"(upper_e4m3)
    );
    return packed;
}

__device__ __forceinline__ float2 exp2_native_f32x2(float2 value) {
    float2 output;
    asm(
        "ex2.approx.ftz.f32 %0, %2;\n\t"
        "ex2.approx.ftz.f32 %1, %3;\n"
        : "=f"(output.x), "=f"(output.y)
        : "f"(value.x), "f"(value.y)
    );
    return output;
}

__device__ __forceinline__ float2 clamp_probability_log2(float2 value) {
    float2 output;
    asm(
        "min.ftz.f32 %0, %2, 0f00000000;\n\t"
        "min.ftz.f32 %1, %3, 0f00000000;\n"
        : "=f"(output.x), "=f"(output.y)
        : "f"(value.x), "f"(value.y)
    );
    return output;
}

__device__ __forceinline__ void make_probability_half(
    owner_aligned_fp32_half &probability,
    const attention_tmem_tile &score_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int stage,
    bool diagonal,
    float beta_log2e,
    float l_aux_scale
) {
    load_owner_aligned_fp32_half(
        probability, score_tmem, output_subtile, column_half
    );
    tensor_load_wait();

    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    const int lane_row = kittens::laneid() & 15;
    const int lane_column_base = 32 * (kittens::laneid() >> 4);
    const int key_row = output_subtile * 16 + lane_row;
    const int query_column_base =
        column_half * kColumnHalf + lane_column_base;
    const float2 l_aux_scale_pair{l_aux_scale, l_aux_scale};

    // Preserve v400's raw-stat TMA ABI and fused scaling, but consume the
    // statistic in the owner-aligned column order.
#pragma unroll
    for (int pair = 0; pair < kColumnHalf / 2; ++pair) {
        const int local_column = 2 * pair;
        const float2 statistic = *reinterpret_cast<const float2 *>(
            &storage.lstat[stage][query_column_base + local_column]
        );
        float2 value = probability.pairs[pair];
        value.x *= beta_log2e;
        value.y *= beta_log2e;
        probability.pairs[pair] =
            base_ops::fma_AxBtC::template op<float2>(
                statistic, l_aux_scale_pair, value
            );
    }

    // Keep the branch mask-only so the common exp2 body is emitted once.
    if (diagonal) {
#pragma unroll
        for (int pair = 0; pair < kColumnHalf / 2; ++pair) {
            const int local_column = 2 * pair;
            float2 value = probability.pairs[pair];
            if (key_row > query_column_base + local_column) {
                value.x = kNegInf;
            }
            if (key_row > query_column_base + local_column + 1) {
                value.y = kNegInf;
            }
            probability.pairs[pair] = value;
        }
    }
#pragma unroll
    for (int pair = 0; pair < kColumnHalf / 2; ++pair) {
        float2 value = clamp_probability_log2(probability.pairs[pair]);
        // CuTe's flat loop uses (i / 2) % 2 == 0.  One owner float2 is one
        // such pair, so even pairs use the degree-1 ALU port and odd pairs
        // retain native EX2.  Full unrolling folds this branch statically.
        if ((pair & 1) == 0) {
            value = exp2_policy::exp2_alu_degree1_f32x2(value);
        } else {
            value = exp2_native_f32x2(value);
        }
        value.x *= 256.0f;
        value.y *= 256.0f;
        probability.pairs[pair] = value;
    }
}

__device__ __forceinline__ void store_probability_half(
    probability_tmem_tile &destination,
    const owner_aligned_fp32_half &probability,
    int output_subtile,
    int column_half
) {
    uint32_t packed[kColumnHalf / 8];
#pragma unroll
    for (int word = 0; word < kColumnHalf / 8; ++word) {
        packed[word] = pack_owner_e4m3_word(
            probability.pairs[2 * word],
            probability.pairs[2 * word + 1]
        );
    }
    const probability_tmem_fragment destination_half =
        destination.template subtile<probability_tmem_fragment>(
            0,
            column_half * kColumnHalf
        );
    const uint32_t address =
        destination_half.addr +
        (static_cast<uint32_t>(output_subtile * 16) << 16);
    asm volatile(
        "tcgen05.st.sync.aligned.16x32bx2.x8.b32 "
        "[%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};\n"
        :: "r"(address),
           "r"(packed[0]),
           "r"(packed[1]),
           "r"(packed[2]),
           "r"(packed[3]),
           "r"(packed[4]),
           "r"(packed[5]),
           "r"(packed[6]),
           "r"(packed[7])
        : "memory"
    );
}

template <typename SharedTile>
__device__ __forceinline__ void store_owner_aligned_shared_half(
    SharedTile &destination,
    const owner_aligned_fp32_half &source
) {
    static_assert(SharedTile::rows == 16);
    static_assert(SharedTile::cols == kColumnHalf);
    static_assert(sizeof(typename SharedTile::dtype) == 1);
    static_assert(SharedTile::swizzle);
    static_assert(SharedTile::swizzle_bytes == 128);

    const int lane_row = kittens::laneid() & 15;
    const int lane_column_base = 32 * (kittens::laneid() >> 4);
    const uint32_t shared_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&destination.data[0])
    );
#pragma unroll
    for (int word = 0; word < kColumnHalf / 8; ++word) {
        const int column = lane_column_base + 4 * word;
        const uint32_t packed = pack_owner_e4m3_word(
            source.pairs[2 * word],
            source.pairs[2 * word + 1]
        );
        const uint32_t destination_address = destination.idx(
            shared_address,
            {lane_row, column}
        );
        asm volatile(
            "st.shared.b32 [%0], %1;\n"
            :
            : "r"(destination_address), "r"(packed)
            : "memory"
        );
    }
}

__device__ __forceinline__ void make_ds_half(
    const attention_tmem_tile &dp_tmem,
    const owner_aligned_fp32_half &probability_scaled,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int stage,
    float beta,
    float delta_scale
) {
    owner_aligned_fp32_half dp;
    load_owner_aligned_fp32_half(
        dp, dp_tmem, output_subtile, column_half
    );
    tensor_load_wait();

    const int lane_column_base = 32 * (kittens::laneid() >> 4);
    const int query_column_base =
        column_half * kColumnHalf + lane_column_base;
    const float2 delta_scale_pair{delta_scale, delta_scale};
#pragma unroll
    for (int pair = 0; pair < kColumnHalf / 2; ++pair) {
        const int local_column = 2 * pair;
        const float2 statistic = *reinterpret_cast<const float2 *>(
            &storage.dstat[stage][query_column_base + local_column]
        );
        float2 value = base_ops::fma_AxBtC::template op<float2>(
            statistic, delta_scale_pair, dp.pairs[pair]
        );
        value.x *= probability_scaled.pairs[pair].x;
        value.y *= probability_scaled.pairs[pair].y;
        value.x *= beta;
        value.y *= beta;
        dp.pairs[pair] = value;
    }
    auto destination = storage.ds[stage].template subtile<16, kColumnHalf>(
        {output_subtile, column_half}
    );
    store_owner_aligned_shared_half(destination, dp);
}

// Dense FP8 dV consumes K32.  TK's generic TMEM/shared helper advances the
// MN-major shared descriptor by one K16 tile per command, so use the proven
// project-local 2*chunk correction while retaining native TMEM-A addressing.
template <int Half, int Accumulate>
__device__ __forceinline__ void issue_gradient_tmem_shared_ab_half(
    gradient_tmem_tile &destination,
    const probability_tmem_tile &probability,
    const operand_tile &dout,
    semaphore &completion
) {
    static_assert(Half == 0 || Half == 1);
    static_assert(Accumulate == 0 || Accumulate == 1);
    static_assert(Half == 0 || Accumulate == 1);
    using input_type = fp8e4m3;
    using output_type = typename gradient_tmem_tile::T;
    constexpr uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kKeyTile,
            kDepth,
            transpose::N,
            transpose::T,
            false
        >();
    ::kittens::st_descriptor<operand_tile, transpose::T> dout_descriptor(
        dout
    );
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    constexpr int kFirstChunk = Half * 2;
    ::kittens::detail::tcgen05::template tt_st<input_type, Accumulate, 1>(
        destination.addr,
        probability.template chunk_addr<transpose::N>(kFirstChunk),
        dout_descriptor.chunk_descriptor(2 * kFirstChunk),
        instruction
    );
    constexpr int kSecondChunk = kFirstChunk + 1;
    ::kittens::detail::tcgen05::template tt_st<input_type, 1, 1>(
        destination.addr,
        probability.template chunk_addr<transpose::N>(kSecondChunk),
        dout_descriptor.chunk_descriptor(2 * kSecondChunk),
        instruction
    );
    if constexpr (Half == 1) {
        // This commit covers both previously issued half-0 commands and these
        // final half-1 commands; score/P TMEM reuse waits on this exact event.
        tensor_commit<1>(completion);
    }
}

// All four reducer warps execute the same publication sequence.  The issuer
// owns the TMA group state; the barrier makes shared-stage reuse exact.
__device__ __forceinline__ void acquire_gradient_publication_stage(
    int publication_sequence,
    int logical_warp,
    barrier<kReduceWarps> reusable
) {
    if (publication_sequence >= kGradientPublicationStages &&
        logical_warp == 0) {
        warp::tma::store_async_read_wait<1>();
    }
    arrive_and_wait(reusable);
}

__device__ __forceinline__ void publish_gradient_async(
    const globals::gradient_gl &destination,
    gradient_stage_tile &stage,
    const coord<gradient_stage_tile> &coordinate,
    int logical_warp,
    barrier<kReduceWarps> staged
) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    arrive_and_wait(staged);
    if (logical_warp == 0) {
        warp::tma::store_add_async(destination, stage, coordinate);
    }
}

__device__ __forceinline__ void finish_gradient_publications(
    int logical_warp,
    barrier<kReduceWarps> reusable
) {
    if (logical_warp == 0) {
        warp::tma::store_async_wait<0>();
    }
    arrive_and_wait(reusable);
}

__global__ __launch_bounds__(kThreads, 1)
void main_kernel(const __grid_constant__ globals g) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready[kStages];
    __shared__ alignas(16) semaphore stats_ready[kStages];
    __shared__ alignas(16) semaphore stats_consumed[kStages];
    __shared__ alignas(16) semaphore score_ready[kStages];
    __shared__ alignas(16) semaphore probability_half_ready[kStages][2];
    __shared__ alignas(16) semaphore dp_ready[kStages];
    __shared__ alignas(16) semaphore dv_done[kStages];
    __shared__ alignas(16) semaphore ds_ready[kStages];
    __shared__ alignas(16) semaphore dk_done[kStages];
    __shared__ alignas(16) semaphore dq_ready[kStages];
    __shared__ alignas(16) semaphore dq_drained[kStages];
    __shared__ alignas(16) semaphore kernel_complete;

    const int physical_warp = warpid();
    const int lane = laneid();
    const int key_tile = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int kv_head = query_head / kHeadRatio;
    const int iteration_count = g.sequence / kQueryTile - key_tile;

    // The compute warps retain one FP32 P half across dP.  Rebalance the
    // 65,536-register CTA budget exactly: 8*136 + 4*144 + 4*96 = 2,048
    // registers/lane across sixteen warps.
    if (physical_warp < kComputeWarps) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;" ::: "memory");
    } else if (physical_warp >= kReduceWarpBase &&
        physical_warp < kTensorIssueWarp) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;" ::: "memory");
    } else if (physical_warp >= kTensorIssueWarp) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;" ::: "memory");
    }

    if (threadIdx.x == 0) {
        init_semaphore(persistent_ready, 0, 1);
        for (int stage = 0; stage < kStages; ++stage) {
            init_semaphore(query_ready[stage], 0, 1);
            init_semaphore(stats_ready[stage], 0, 1);
            init_semaphore(stats_consumed[stage], 0, kComputeWarps);
            init_semaphore(score_ready[stage], 0, 1);
            init_semaphore(
                probability_half_ready[stage][0], 0, kComputeWarps
            );
            init_semaphore(
                probability_half_ready[stage][1], 0, kComputeWarps
            );
            init_semaphore(dp_ready[stage], 0, 1);
            init_semaphore(dv_done[stage], 0, 1);
            init_semaphore(ds_ready[stage], 0, kComputeWarps);
            init_semaphore(dk_done[stage], 0, 1);
            init_semaphore(dq_ready[stage], 0, 1);
            init_semaphore(dq_drained[stage], 0, kReduceWarps);
        }
        init_semaphore(kernel_complete, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tmem_allocator{};
    attention_tmem_tile dp_dq_tmem =
        tmem_allocator.template allocate<attention_tmem_tile>(
            kDpDqTmemOffset
        );
    gradient_tmem_tile dq_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(
            kDpDqTmemOffset
        );
    gradient_tmem_tile dk_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(kDkTmemOffset);
    gradient_tmem_tile dv_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(kDvTmemOffset);
    attention_tmem_tile score_tmem_base =
        tmem_allocator.template allocate<attention_tmem_tile>(
            kScoreTmemOffset
        );
    probability_tmem_tile probability_tmem_base{score_tmem_base.addr};

    if (physical_warp == kLoaderWarp && lane == 0) {
        tma::expect_bytes(
            persistent_ready,
            sizeof(storage.k) + sizeof(storage.v)
        );
        tma::load_async(
            storage.k,
            g.k,
            coord<operand_tile>{batch, kv_head, key_tile, 0},
            persistent_ready
        );
        tma::load_async(
            storage.v,
            g.v,
            coord<operand_tile>{batch, kv_head, key_tile, 0},
            persistent_ready
        );

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            if (iteration >= kStages) {
                const int old_phase = previous_stage_phase(iteration);
                wait(score_ready[stage], old_phase);
                wait(dk_done[stage], old_phase);
                wait(dp_ready[stage], old_phase);
                wait(dv_done[stage], old_phase);
                // TMA may overwrite raw statistics only after every compute
                // warp has consumed both lstat for P and dstat for dS.
                wait(stats_consumed[stage], old_phase);
            }
            tma::expect_bytes(
                query_ready[stage],
                sizeof(storage.q[stage]) + sizeof(storage.dout[stage])
            );
            const int query_tile = key_tile + iteration;
            const coord<operand_tile> coordinate{
                batch,
                query_head,
                query_tile,
                0,
            };
            tma::load_async(
                storage.q[stage], g.q, coordinate, query_ready[stage]
            );
            tma::load_async(
                storage.dout[stage], g.dout, coordinate, query_ready[stage]
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
                g.l_aux,
                stats_coordinate,
                stats_ready[stage]
            );
            tma::load_async(
                storage.dstat[stage],
                g.delta,
                stats_coordinate,
                stats_ready[stage]
            );
        }
    } else if (physical_warp < kComputeWarps) {
        const int output_subtile =
            half::output_subtile_for_warp(physical_warp);
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            const int phase = stage_phase(iteration);
            wait(score_ready[stage], phase);
            wait(stats_ready[stage], phase);
            tensor_after_thread_sync();

            // Retain one exact FP32 P half at a time through the matching dP
            // load.  The two halves have disjoint score/P TMEM columns, and
            // output_subtile_for_warp is a bijection over the eight 16-row
            // slices, so each warp owns every address it overwrites.
            owner_aligned_fp32_half probability;
            const attention_tmem_tile active_score =
                score_stage(score_tmem_base, stage);
            make_probability_half(
                probability,
                active_score,
                storage,
                output_subtile,
                0,
                stage,
                iteration == 0,
                g.beta_log2e,
                g.l_aux_scale
            );
            probability_tmem_tile active_probability{
                probability_tmem_base.addr + stage * kQueryTile
            };
            store_probability_half(
                active_probability,
                probability,
                output_subtile,
                0
            );
            tensor_store_wait();
            tensor_before_thread_sync();
            __syncwarp();
            if (lane == 0) {
                arrive(probability_half_ready[stage][0]);
            }

            wait(dp_ready[stage], phase);
            if (iteration >= kStages) {
                const int old_phase = previous_stage_phase(iteration);
                // dK and dQ are independent readers of the recycled dS tile.
                wait(dk_done[stage], old_phase);
                wait(dq_ready[stage], old_phase);
            }
            tensor_after_thread_sync();
            make_ds_half(
                dp_dq_tmem,
                probability,
                storage,
                output_subtile,
                0,
                stage,
                g.beta,
                -16.0f
            );

            make_probability_half(
                probability,
                active_score,
                storage,
                output_subtile,
                1,
                stage,
                iteration == 0,
                g.beta_log2e,
                g.l_aux_scale
            );
            store_probability_half(
                active_probability,
                probability,
                output_subtile,
                1
            );
            tensor_store_wait();
            tensor_before_thread_sync();
            __syncwarp();
            if (lane == 0) {
                arrive(probability_half_ready[stage][1]);
            }
            make_ds_half(
                dp_dq_tmem,
                probability,
                storage,
                output_subtile,
                1,
                stage,
                g.beta,
                -16.0f
            );
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile(
                "fence.proxy.async.shared::cta;" ::: "memory"
            );
            if (lane == 0) {
                arrive(stats_consumed[stage]);
                arrive(ds_ready[stage]);
            }
        }
        if (physical_warp == 0) {
            wait(kernel_complete, 0);
            tensor_after_thread_sync();
        }
    } else if (physical_warp == kTensorIssueWarp && lane == 0) {
        wait(persistent_ready, 0);

        wait(query_ready[0], 0);
        base::issue_score_or_dp(
            score_tmem_base, storage.k, storage.q[0], score_ready[0]
        );
        // dP is independent of P publication and has a disjoint TMEM stage;
        // issue it immediately so each compute warp can retire its first FP32
        // P half before producing the second.
        base::issue_score_or_dp(
            dp_dq_tmem, storage.v, storage.dout[0], dp_ready[0]
        );
        wait(probability_half_ready[0][0], 0);
        tensor_after_thread_sync();
        issue_gradient_tmem_shared_ab_half<0, 0>(
            dv_tmem,
            probability_tmem_base,
            storage.dout[0],
            dv_done[0]
        );
        wait(probability_half_ready[0][1], 0);
        tensor_after_thread_sync();
        issue_gradient_tmem_shared_ab_half<1, 1>(
            dv_tmem,
            probability_tmem_base,
            storage.dout[0],
            dv_done[0]
        );

        for (int iteration = 0; iteration + 1 < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            const int phase = stage_phase(iteration);
            const int next_iteration = iteration + 1;
            const int next_stage = next_iteration & 1;
            const int next_phase = stage_phase(next_iteration);

            wait(query_ready[next_stage], next_phase);
            if (next_iteration >= kStages) {
                // score_stage[next_stage] contains the old FP8 P payload until
                // its true TMEM-A dV read commits.  This is the sole legal
                // reuse point for the shared score/P allocation.
                wait(
                    dv_done[next_stage],
                    previous_stage_phase(next_iteration)
                );
                tensor_after_thread_sync();
            }
            attention_tmem_tile next_score_tmem =
                score_stage(score_tmem_base, next_stage);
            base::issue_score_or_dp(
                next_score_tmem,
                storage.k,
                storage.q[next_stage],
                score_ready[next_stage]
            );

            wait(ds_ready[stage], phase);
            tensor_after_thread_sync();
            // Let the reducer drain aliased dQ while this warp submits dK.
            base::issue_gradient_atb(
                dq_tmem,
                storage.ds[stage],
                storage.k,
                dq_ready[stage]
            );
            if (iteration == 0) {
                base::issue_gradient_ab<0>(
                    dk_tmem,
                    storage.ds[stage],
                    storage.q[stage],
                    dk_done[stage]
                );
            } else {
                base::issue_gradient_ab<1>(
                    dk_tmem,
                    storage.ds[stage],
                    storage.q[stage],
                    dk_done[stage]
                );
            }

            wait(dq_drained[stage], phase);
            tensor_after_thread_sync();
            base::issue_score_or_dp(
                dp_dq_tmem,
                storage.v,
                storage.dout[next_stage],
                dp_ready[next_stage]
            );
            wait(
                probability_half_ready[next_stage][0], next_phase
            );
            tensor_after_thread_sync();
            probability_tmem_tile next_probability{
                probability_tmem_base.addr + next_stage * kQueryTile
            };
            issue_gradient_tmem_shared_ab_half<0, 1>(
                dv_tmem,
                next_probability,
                storage.dout[next_stage],
                dv_done[next_stage]
            );
            wait(
                probability_half_ready[next_stage][1], next_phase
            );
            tensor_after_thread_sync();
            issue_gradient_tmem_shared_ab_half<1, 1>(
                dv_tmem,
                next_probability,
                storage.dout[next_stage],
                dv_done[next_stage]
            );
        }

        const int last_iteration = iteration_count - 1;
        const int last_stage = last_iteration & 1;
        const int last_phase = stage_phase(last_iteration);
        wait(ds_ready[last_stage], last_phase);
        tensor_after_thread_sync();
        base::issue_gradient_atb(
            dq_tmem,
            storage.ds[last_stage],
            storage.k,
            dq_ready[last_stage]
        );
        if (last_iteration == 0) {
            base::issue_gradient_ab<0>(
                dk_tmem,
                storage.ds[last_stage],
                storage.q[last_stage],
                dk_done[last_stage]
            );
        } else {
            base::issue_gradient_ab<1>(
                dk_tmem,
                storage.ds[last_stage],
                storage.q[last_stage],
                dk_done[last_stage]
            );
        }
    } else if (
        physical_warp >= kReduceWarpBase &&
        physical_warp < kTensorIssueWarp
    ) {
        const int logical_warp = physical_warp - kReduceWarpBase;
        barrier<kReduceWarps> gradient_reusable(1);
        barrier<kReduceWarps> gradient_staged(2);
        int publication_sequence = 0;

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            const int phase = stage_phase(iteration);
            wait(dq_ready[stage], phase);
            tensor_after_thread_sync();
            const int publication_stage =
                publication_sequence & (kGradientPublicationStages - 1);
            acquire_gradient_publication_stage(
                publication_sequence, logical_warp, gradient_reusable
            );
            drain_gradient_to_bf16(
                dq_tmem,
                storage.gradient[publication_stage],
                logical_warp
            );
            tensor_before_thread_sync();
            __syncwarp();
            if (lane == 0) {
                arrive(dq_drained[stage]);
            }
            publish_gradient_async(
                g.dq,
                storage.gradient[publication_stage],
                coord<gradient_stage_tile>{
                    batch,
                    query_head,
                    key_tile + iteration,
                    0,
                },
                logical_warp,
                gradient_staged
            );
            ++publication_sequence;
        }

        const int last_iteration = iteration_count - 1;
        const int last_stage = last_iteration & 1;
        const int last_phase = stage_phase(last_iteration);
        wait(dk_done[last_stage], last_phase);
        tensor_after_thread_sync();
        int publication_stage =
            publication_sequence & (kGradientPublicationStages - 1);
        acquire_gradient_publication_stage(
            publication_sequence, logical_warp, gradient_reusable
        );
        drain_gradient_to_bf16(
            dk_tmem,
            storage.gradient[publication_stage],
            logical_warp
        );
        publish_gradient_async(
            g.dk,
            storage.gradient[publication_stage],
            coord<gradient_stage_tile>{batch, kv_head, key_tile, 0},
            logical_warp,
            gradient_staged
        );
        ++publication_sequence;

        wait(dv_done[last_stage], last_phase);
        tensor_after_thread_sync();
        publication_stage =
            publication_sequence & (kGradientPublicationStages - 1);
        acquire_gradient_publication_stage(
            publication_sequence, logical_warp, gradient_reusable
        );
        drain_gradient_to_bf16(
            dv_tmem,
            storage.gradient[publication_stage],
            logical_warp
        );
        tensor_before_thread_sync();
        __syncwarp();
        publish_gradient_async(
            g.dv,
            storage.gradient[publication_stage],
            coord<gradient_stage_tile>{batch, kv_head, key_tile, 0},
            logical_warp,
            gradient_staged
        );
        ++publication_sequence;
        finish_gradient_publications(logical_warp, gradient_reusable);
        if (logical_warp == 0 && lane == 0) {
            arrive(kernel_complete);
        }
    }
}

inline void launch(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float softmax_scale,
    cudaStream_t stream
) {
    const float beta = softmax_scale / 16.0f;
    const globals g{
        kittens::py::tensor_to_gl<globals::operand_gl>(q),
        kittens::py::tensor_to_gl<globals::operand_gl>(k),
        kittens::py::tensor_to_gl<globals::operand_gl>(v),
        kittens::py::tensor_to_gl<globals::operand_gl>(dout),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dq),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dk),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dv),
        kittens::py::tensor_to_gl<globals::stats_gl>(
            l_aux,
            q.size(0),
            q.size(1),
            1,
            q.size(2)
        ),
        kittens::py::tensor_to_gl<globals::stats_gl>(
            delta,
            q.size(0),
            q.size(1),
            1,
            q.size(2)
        ),
        beta,
        beta * kLog2E,
        softmax_scale * kLog2E,
        static_cast<int>(q.size(2)),
    };
    const dim3 grid(
        static_cast<unsigned int>(q.size(2) / kKeyTile),
        static_cast<unsigned int>(q.size(1)),
        static_cast<unsigned int>(q.size(0))
    );
    v413_d64_gqa_e4m3_owner_aligned_dq_first_period2::main_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v413_d64_gqa_e4m3_owner_aligned_dq_first_period2

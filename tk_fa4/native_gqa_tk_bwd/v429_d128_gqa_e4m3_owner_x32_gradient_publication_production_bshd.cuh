#pragma once

#include "v421_d128_gqa_e4m3_async_production_bshd.cuh"

// D128 exact-native production backward derived from v424.  Its reducer uses
// the owner-aligned tcgen05 x32 load to materialize one physical 32x32 slice,
// immediately scales/packs that slice to BF16, and writes it directly into the
// existing double-buffered D32 publication tile.  This removes the generic TK
// register-tile drain uniformly for dQ, dK, and dV without adding a full-tile
// FP32 shared-memory round trip.  All v424 score, dP, and lifetime scheduling
// remains unchanged.
namespace tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd {

namespace core =
    tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd;
namespace async =
    tkfa4::native_gqa_tk_bwd::v421_d128_gqa_e4m3_async_production_bshd;
namespace d64 =
    tkfa4::native_gqa_tk_bwd::v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds;
namespace half =
    tkfa4::native_gqa_tk_bwd::v386_d64_gqa_e4m3_k128q128_halfcols;
namespace mma = tkfa4::native_gqa_tk_bwd::pipelined;

using core::attention_tile;
using core::attention_tmem_fragment;
using core::attention_tmem_tile;
using core::globals;
using core::gradient_chunk_tile;
using core::gradient_tmem_tile;
using core::kColumnHalf;
using core::kComputeWarps;
using core::kDepth;
using core::kDepthChunk;
using core::kDepthChunks;
using core::kDkTmemOffset;
using core::kDpDqTmemOffset;
using core::kDvTmemOffset;
using core::kHeadRatio;
using core::kKeyTile;
using core::kKvHeads;
using core::kLoaderWarp;
using core::kOperandScale;
using core::kQueryHeads;
using core::kQueryTile;
using core::kReduceWarpBase;
using core::kReduceWarps;
using core::kScoreTmemOffset;
using core::kTensorIssueWarp;
using core::kThreads;
using core::operand_tile;
using core::stats_tile;

constexpr int kInputStages = 2;
constexpr int kGradientPublicationStages = 2;

struct shared_storage {
    operand_tile k;
    operand_tile v;
    operand_tile q[kInputStages];
    operand_tile dout[kInputStages];
    attention_tile probability;
    attention_tile ds;
    gradient_chunk_tile gradient[kGradientPublicationStages];
    stats_tile lstat[kInputStages];
    stats_tile dstat[kInputStages];
};

// K/V=32 KiB, Q/dO x2=64 KiB, P/dS=32 KiB, D32 publication
// x2=16 KiB, lstat/dstat x2=2 KiB: 146 KiB before small semaphore state.
static_assert(sizeof(shared_storage) == 146 * 1024);
static_assert(sizeof(shared_storage) < 150528);

__device__ __forceinline__ int iteration_phase(int iteration) {
    return iteration & 1;
}

__device__ __forceinline__ int input_stage_epoch_phase(int iteration) {
    return (iteration / kInputStages) & 1;
}

__device__ __forceinline__ int previous_input_epoch_phase(int iteration) {
    return ((iteration - kInputStages) / kInputStages) & 1;
}

__device__ __forceinline__ int output_subtile_for_warp(
    int physical_warp
) {
    return 2 * (physical_warp & 3) + (physical_warp >> 2);
}

__device__ __forceinline__ void make_probability_half(
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

    // Only the second-half call supplies this gate.  At this point all score
    // values needed by this warp are resident in one half-sized FP32 owner
    // fragment.  The collective fence makes it safe for the issuer to reuse
    // the single score TMEM page while native EX2 and the shared store proceed.
    if (score_consumed != nullptr) {
        tensor_before_thread_sync();
        __syncwarp();
        if (kittens::laneid() == 0) {
            arrive(*score_consumed);
        }
    }

    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    const int lane_row = kittens::laneid() & 15;
    const int lane_column_base = 32 * (kittens::laneid() >> 4);
    const int key_row = output_subtile * 16 + lane_row;
    const int query_column_base =
        column_half * kColumnHalf + lane_column_base;

#pragma unroll
    // Each lane owns 32 of this 64-column half (16 float2 pairs); the
    // companion lane for the same row owns the other 32 columns.
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
    d64::store_owner_aligned_shared_half(destination, probability);
}

__device__ __forceinline__ void make_ds_half(
    const attention_tmem_tile &dp_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int input_stage,
    float beta
) {
    half::attention_fp32_fragment probability;
    half::attention_fp32_fragment dp;
    half::attention_e4m3_fragment probability_lowp;
    half::attention_e4m3_fragment ds_lowp;

    auto probability_source =
        storage.probability.template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    half::load_e4m3_half(probability_lowp, probability_source);
    warp::copy(probability, probability_lowp);

    const attention_tmem_fragment dp_half =
        dp_tmem.template subtile<attention_tmem_fragment>(
            0,
            column_half * kColumnHalf
        );
    group<kComputeWarps>::load_async(dp, dp_half);
    tensor_load_wait();
    half::add_shared_row_vector_half(
        dp,
        storage.dstat[input_stage],
        column_half
    );
    warp::mul(dp, probability, dp);
    warp::mul(dp, dp, beta);
    mma::convert_f32_to_e4m3(ds_lowp, dp);

    auto destination = storage.ds.template subtile<16, kColumnHalf>(
        {output_subtile, column_half}
    );
    warp::store(destination, ds_lowp);
}

// A reducer warp owns 32 consecutive physical accumulator rows.  With the
// source row encoded in TMEM address bits [22:16], x32 returns the 32 D-column
// values for this lane's physical row in source order.  This is the audited
// owner mapping used by v428, but the values are converted and published
// directly instead of taking an FP32 shared-memory detour.
__device__ __forceinline__ void load_tmem_owner_x32(
    uint32_t (&destination)[kDepthChunk],
    uint32_t source_address
) {
    static_assert(kDepthChunk == 32);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
        : "=r"(destination[0]), "=r"(destination[1]),
          "=r"(destination[2]), "=r"(destination[3]),
          "=r"(destination[4]), "=r"(destination[5]),
          "=r"(destination[6]), "=r"(destination[7]),
          "=r"(destination[8]), "=r"(destination[9]),
          "=r"(destination[10]), "=r"(destination[11]),
          "=r"(destination[12]), "=r"(destination[13]),
          "=r"(destination[14]), "=r"(destination[15]),
          "=r"(destination[16]), "=r"(destination[17]),
          "=r"(destination[18]), "=r"(destination[19]),
          "=r"(destination[20]), "=r"(destination[21]),
          "=r"(destination[22]), "=r"(destination[23]),
          "=r"(destination[24]), "=r"(destination[25]),
          "=r"(destination[26]), "=r"(destination[27]),
          "=r"(destination[28]), "=r"(destination[29]),
          "=r"(destination[30]), "=r"(destination[31])
        : "r"(source_address)
        : "memory"
    );
}

__device__ __forceinline__ void store_bf16_x8(
    uint32_t destination,
    const bf16_2 &value0,
    const bf16_2 &value1,
    const bf16_2 &value2,
    const bf16_2 &value3
) {
    asm volatile(
        "st.shared.v4.b32 [%4], {%0, %1, %2, %3};\n"
        :
        : "r"(*reinterpret_cast<const uint32_t *>(&value0)),
          "r"(*reinterpret_cast<const uint32_t *>(&value1)),
          "r"(*reinterpret_cast<const uint32_t *>(&value2)),
          "r"(*reinterpret_cast<const uint32_t *>(&value3)),
          "r"(destination)
        : "memory"
    );
}

// Drain one 128x32 accumulator slice directly to the existing swizzled BF16
// publication stage.  The tensor-load wait must immediately follow the raw
// tcgen05 load: no accumulator value is inspected or repurposed before it.
__device__ __forceinline__ void drain_gradient_chunk_owner_x32_to_bf16(
    const gradient_tmem_tile &source,
    gradient_chunk_tile &destination,
    int logical_warp,
    int depth_chunk,
    int lane
) {
    constexpr float kOutputScale = 1.0f / 256.0f;
    const float2 output_scale = make_float2(kOutputScale, kOutputScale);
    uint32_t values[kDepthChunk];
    const uint32_t source_row =
        source.addr + (static_cast<uint32_t>(logical_warp * 32) << 16);
    load_tmem_owner_x32(
        values,
        source_row + depth_chunk * kDepthChunk
    );
    tensor_load_wait();

    const int physical_row = logical_warp * 32 + lane;
    const uint32_t destination_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(destination.data)
    );
#pragma unroll
    for (int column = 0; column < kDepthChunk; column += 8) {
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
        store_bf16_x8(
            gradient_chunk_tile::idx(
                destination_base,
                {physical_row, column}
            ),
            packed0,
            packed1,
            packed2,
            packed3
        );
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// This is v424/v421 publication with only the drain primitive replaced.  The
// source-drained arrival, ready/reusable phases, additive D32 TMA descriptor,
// coordinates, and publication sequence are intentionally equivalent at
// source level.
__device__ __forceinline__ void publish_gradient_tile_owner_x32(
    const gradient_tmem_tile &source,
    const globals::gradient_gl &destination,
    gradient_chunk_tile (&publication)[kGradientPublicationStages],
    semaphore (&ready)[kGradientPublicationStages],
    semaphore (&reusable)[kGradientPublicationStages],
    semaphore *source_drained,
    int &publication_sequence,
    int batch,
    int sequence_tile,
    int head,
    int logical_warp,
    int lane
) {
#pragma unroll
    for (int depth_chunk = 0; depth_chunk < kDepthChunks; ++depth_chunk) {
        const int stage =
            publication_sequence & (kGradientPublicationStages - 1);
        const int ready_phase =
            (publication_sequence / kGradientPublicationStages) & 1;
        if (publication_sequence >= kGradientPublicationStages) {
            const int reuse_phase =
                ((publication_sequence - kGradientPublicationStages) /
                 kGradientPublicationStages) & 1;
            if (logical_warp == 0 && lane == 0) {
                warp::tma::store_async_read_wait<1>();
                arrive(reusable[stage]);
            }
            wait(reusable[stage], reuse_phase);
        }

        drain_gradient_chunk_owner_x32_to_bf16(
            source,
            publication[stage],
            logical_warp,
            depth_chunk,
            lane
        );
        tensor_before_thread_sync();
        __syncwarp();
        if (source_drained != nullptr &&
            depth_chunk == kDepthChunks - 1 && lane == 0) {
            arrive(*source_drained);
        }
        if (lane == 0) {
            arrive(ready[stage]);
        }
        if (logical_warp == 0) {
            wait(ready[stage], ready_phase);
            if (lane == 0) {
                warp::tma::store_add_async<
                    dim::DEPTH,
                    cache_policy::NORMAL
                >(
                    destination,
                    publication[stage],
                    coord<gradient_chunk_tile>{
                        batch,
                        sequence_tile,
                        head,
                        depth_chunk,
                    }
                );
            }
        }
        ++publication_sequence;
    }
}

__global__ __launch_bounds__(kThreads, 1)
void main_kernel(const __grid_constant__ globals g) {
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
    __shared__ alignas(16) semaphore dq_drained;
    __shared__ alignas(16) semaphore publication_ready[
        kGradientPublicationStages
    ];
    __shared__ alignas(16) semaphore publication_reusable[
        kGradientPublicationStages
    ];
    __shared__ alignas(16) semaphore kernel_complete;

    const int physical_warp = warpid();
    const int lane = laneid();
    const int key_tile = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int kv_head = query_head / kHeadRatio;
    const int iteration_count = g.sequence / kQueryTile - key_tile;

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
            init_semaphore(
                stats_consumed[stage],
                0,
                kComputeWarps
            );
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
        init_semaphore(dq_drained, 0, kReduceWarps);
        for (int stage = 0; stage < kGradientPublicationStages; ++stage) {
            init_semaphore(publication_ready[stage], 0, kReduceWarps);
            init_semaphore(publication_reusable[stage], 0, 1);
        }
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

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            if (iteration >= kInputStages) {
                const int old_phase = previous_input_epoch_phase(iteration);
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
                sizeof(storage.q[stage]) + sizeof(storage.dout[stage])
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
    } else if (physical_warp < kComputeWarps) {
        const int output_subtile = output_subtile_for_warp(physical_warp);
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            const int phase = iteration_phase(iteration);
            const int input_phase = input_stage_epoch_phase(iteration);
            if (iteration > 0) {
                const int old_phase = iteration_phase(iteration - 1);
                wait(dv_ready, old_phase);
                wait(probability_consumed, old_phase);
            }
            wait(score_ready, phase);
            wait(stats_ready[stage], input_phase);
            tensor_after_thread_sync();
            make_probability_half(
                score_tmem,
                storage,
                output_subtile,
                0,
                stage,
                iteration == 0,
                g.beta_log2e,
                nullptr
            );
            make_probability_half(
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
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(probability_ready);
            }

            wait(dp_ready, phase);
            if (iteration > 0) {
                const int old_phase = iteration_phase(iteration - 1);
                wait(dq_ready, old_phase);
                wait(dk_ready, old_phase);
            }
            tensor_after_thread_sync();
            make_ds_half(
                dp_tmem,
                storage,
                output_subtile,
                0,
                stage,
                g.beta
            );
            make_ds_half(
                dp_tmem,
                storage,
                output_subtile,
                1,
                stage,
                g.beta
            );
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(probability_consumed);
                arrive(stats_consumed[stage]);
                arrive(ds_ready);
            }
        }
        if (physical_warp == 0) {
            wait(kernel_complete, 0);
            tensor_after_thread_sync();
        }
    } else if (physical_warp == kTensorIssueWarp && lane == 0) {
        wait(persistent_ready, 0);

        // Prime score and dP for iteration zero.  They target disjoint TMEM
        // pages, and dP depends only on persistent V plus the staged dO.
        wait(query_ready[0], 0);
        core::issue_score_or_dp(
            score_tmem,
            storage.k,
            storage.q[0],
            score_ready
        );
        core::issue_score_or_dp(
            dp_tmem,
            storage.v,
            storage.dout[0],
            dp_ready
        );

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            const int phase = iteration_phase(iteration);
            const bool has_next = iteration + 1 < iteration_count;
            const int next_iteration = iteration + 1;
            const int next_stage = next_iteration & (kInputStages - 1);

            // Every compute warp has completed its second-half score load, but
            // continues native EX2/E4M3/shared-P work from that one live owner
            // fragment.  Reuse score TMEM now instead of waiting for P.
            if (has_next) {
                wait(score_consumed, phase);
                wait(
                    query_ready[next_stage],
                    input_stage_epoch_phase(next_iteration)
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
            if (iteration == 0) {
                core::issue_gradient_ab<0>(
                    dv_tmem,
                    storage.probability,
                    storage.dout[stage],
                    dv_ready
                );
            } else {
                core::issue_gradient_ab<1>(
                    dv_tmem,
                    storage.probability,
                    storage.dout[stage],
                    dv_ready
                );
            }

            wait(ds_ready, phase);
            tensor_after_thread_sync();
            core::issue_gradient_atb(
                dq_tmem,
                storage.ds,
                storage.k,
                dq_ready
            );
            if (iteration == 0) {
                core::issue_gradient_ab<0>(
                    dk_tmem,
                    storage.ds,
                    storage.q[stage],
                    dk_ready
                );
            } else {
                core::issue_gradient_ab<1>(
                    dk_tmem,
                    storage.ds,
                    storage.q[stage],
                    dk_ready
                );
            }

            // dP(next) is independent of next P/score, but aliases current dQ
            // in TMEM [0,128).  Launch it at the exact collective reducer
            // release point, before waiting on current dK/dV solely for input
            // stage recycling.  ds_ready already proves dP(current) was fully
            // consumed, so no redundant dp_ready wait belongs on this path.
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

            // score_consumed retired Q's score reader, dK retires its remaining
            // Q reader, ds_ready retired dP's dO reader, and dV retires the
            // remaining dO reader.  Only dK/dV completion is still required.
            wait(dk_ready, phase);
            wait(dv_ready, phase);
            tensor_after_thread_sync();
            arrive(operand_consumed[stage]);
        }
    } else if (
        physical_warp >= kReduceWarpBase &&
        physical_warp < kReduceWarpBase + kReduceWarps
    ) {
        const int logical_warp = physical_warp - kReduceWarpBase;
        int publication_sequence = 0;
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int phase = iteration_phase(iteration);
            wait(dq_ready, phase);
            tensor_after_thread_sync();
            publish_gradient_tile_owner_x32(
                dq_tmem,
                g.dq,
                storage.gradient,
                publication_ready,
                publication_reusable,
                &dq_drained,
                publication_sequence,
                batch,
                key_tile + iteration,
                query_head,
                logical_warp,
                lane
            );
        }

        const int last_phase = iteration_phase(iteration_count - 1);
        wait(dk_ready, last_phase);
        tensor_after_thread_sync();
        publish_gradient_tile_owner_x32(
            dk_tmem,
            g.dk,
            storage.gradient,
            publication_ready,
            publication_reusable,
            nullptr,
            publication_sequence,
            batch,
            key_tile,
            kv_head,
            logical_warp,
            lane
        );

        wait(dv_ready, last_phase);
        tensor_after_thread_sync();
        publish_gradient_tile_owner_x32(
            dv_tmem,
            g.dv,
            storage.gradient,
            publication_ready,
            publication_reusable,
            nullptr,
            publication_sequence,
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
    const dim3 grid(
        static_cast<unsigned int>(q.size(1) / kKeyTile),
        static_cast<unsigned int>(q.size(2)),
        static_cast<unsigned int>(q.size(0))
    );
    v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd::main_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd

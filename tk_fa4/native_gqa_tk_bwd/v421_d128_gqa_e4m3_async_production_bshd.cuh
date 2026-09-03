#pragma once

#include "v420_d128_gqa_e4m3_shared_p_production_bshd.cuh"

// Role-specialized asynchronous evolution of v420.  D128 still consumes all
// 512 TMEM columns, so the single score page and four gradient pages remain
// unchanged.  The steady-state CTA-wide barriers are replaced by phase-
// counted mbarriers between the loader, tensor issuer, eight score/dS warps,
// and four reducer warps.  Gradient publication uses two D32 shared stages;
// read_wait<1> makes each stage reusable without serializing the other stage.
namespace tkfa4::native_gqa_tk_bwd::v421_d128_gqa_e4m3_async_production_bshd {

namespace base =
    tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd;
namespace d64 =
    tkfa4::native_gqa_tk_bwd::v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds;
namespace half =
    tkfa4::native_gqa_tk_bwd::v386_d64_gqa_e4m3_k128q128_halfcols;
namespace mma = tkfa4::native_gqa_tk_bwd::pipelined;

using base::attention_tile;
using base::attention_tmem_fragment;
using base::attention_tmem_tile;
using base::globals;
using base::gradient_chunk_tile;
using base::gradient_chunk_tmem_tile;
using base::gradient_tmem_tile;
using base::kColumnHalf;
using base::kComputeWarps;
using base::kDepth;
using base::kDepthChunk;
using base::kDepthChunks;
using base::kDkTmemOffset;
using base::kDpDqTmemOffset;
using base::kDvTmemOffset;
using base::kHeadRatio;
using base::kKeyTile;
using base::kKvHeads;
using base::kLoaderWarp;
using base::kOperandScale;
using base::kQueryHeads;
using base::kQueryTile;
using base::kReduceWarpBase;
using base::kReduceWarps;
using base::kScoreTmemOffset;
using base::kTensorIssueWarp;
using base::kThreads;
using base::operand_tile;
using base::stats_tile;

constexpr int kGradientPublicationStages = 2;

struct shared_storage {
    operand_tile k;
    operand_tile v;
    operand_tile q;
    operand_tile dout;
    attention_tile probability;
    attention_tile ds;
    gradient_chunk_tile gradient[kGradientPublicationStages];
    stats_tile lstat;
    stats_tile dstat;
};

// K/V/Q/dO=64 KiB, P/dS=32 KiB, two D32 BF16 publication
// stages=16 KiB, statistics=1 KiB.  This bounded single-input-stage control
// keeps an exact consumer gate; v422 evaluates the larger two-stage layout.
static_assert(sizeof(shared_storage) == 113 * 1024);
static_assert(sizeof(shared_storage) < 227 * 1024);

__device__ __forceinline__ int stage_phase(int sequence) {
    return sequence & 1;
}

__device__ __forceinline__ int previous_stage_phase(int sequence) {
    return (sequence - 1) & 1;
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
    bool diagonal,
    float beta_log2e
) {
    d64::owner_aligned_fp32_half probability;
    d64::load_owner_aligned_fp32_half(
        probability,
        score_tmem,
        output_subtile,
        column_half
    );
    tensor_load_wait();

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
            &storage.lstat[query_column_base + local_column]
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
    half::add_shared_row_vector_half(dp, storage.dstat, column_half);
    warp::mul(dp, probability, dp);
    warp::mul(dp, dp, beta);
    mma::convert_f32_to_e4m3(ds_lowp, dp);

    auto destination = storage.ds.template subtile<16, kColumnHalf>(
        {output_subtile, column_half}
    );
    warp::store(destination, ds_lowp);
}

// Four reducer warps cooperatively drain 128 rows x 32 D columns.  No full
// D128 register tile is ever materialized.
__device__ __forceinline__ void drain_gradient_chunk_to_bf16(
    const gradient_tmem_tile &source,
    gradient_chunk_tile &destination,
    int logical_warp,
    int depth_chunk
) {
    rt_fl<32, kDepthChunk> values_fp32;
    rt_bf<32, kDepthChunk> values_bf16;
    const gradient_chunk_tmem_tile source_chunk =
        source.template subtile<gradient_chunk_tmem_tile>(
            0,
            depth_chunk * kDepthChunk
        );
    group<kReduceWarps>::load_async(values_fp32, source_chunk);
    tensor_load_wait();
    warp::mul(values_fp32, values_fp32, 1.0f / 256.0f);
    warp::copy(values_bf16, values_fp32);
    auto destination_slice =
        destination.template subtile<32, kDepthChunk>(
            {logical_warp, 0}
        );
    warp::store(destination_slice, values_bf16);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Reducer-only publication.  ready[] counts four writer warps.  reusable[]
// is advanced by the publisher only after read_wait<1>, so the stage from two
// publications ago is no longer an async-TMA reader.  Other reducer warps can
// prepare the alternate stage while logical warp zero submits the current one.
__device__ __forceinline__ void publish_gradient_tile(
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

        drain_gradient_chunk_to_bf16(
            source,
            publication[stage],
            logical_warp,
            depth_chunk
        );
        // The source page may be overwritten as soon as source_drained
        // advances.  Make every lane's TMEM read and shared publication
        // visible before lane zero releases either consumer gate.
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
    __shared__ alignas(16) semaphore query_ready;
    __shared__ alignas(16) semaphore stats_ready;
    __shared__ alignas(16) semaphore stats_consumed;
    __shared__ alignas(16) semaphore score_ready;
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
        init_semaphore(query_ready, 0, 1);
        init_semaphore(stats_ready, 0, 1);
        init_semaphore(stats_consumed, 0, kComputeWarps);
        init_semaphore(score_ready, 0, 1);
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
            const int phase = stage_phase(iteration);
            if (iteration > 0) {
                const int old_phase = previous_stage_phase(iteration);
                // Q remains live through dK; dO remains live through both dP
                // and dV.  Statistics remain live through dS construction.
                wait(dk_ready, old_phase);
                wait(dp_ready, old_phase);
                wait(dv_ready, old_phase);
                wait(stats_consumed, old_phase);
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
                query_ready,
                sizeof(storage.q) + sizeof(storage.dout)
            );
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.q,
                g.q,
                operand_coordinate,
                query_ready
            );
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.dout,
                g.dout,
                operand_coordinate,
                query_ready
            );

            const coord<stats_tile> stats_coordinate{
                batch,
                query_head,
                0,
                query_tile,
            };
            tma::expect_bytes(
                stats_ready,
                sizeof(storage.lstat) + sizeof(storage.dstat)
            );
            tma::load_async(
                storage.lstat,
                g.lstat,
                stats_coordinate,
                stats_ready
            );
            tma::load_async(
                storage.dstat,
                g.dstat,
                stats_coordinate,
                stats_ready
            );
            (void)phase;
        }
    } else if (physical_warp < kComputeWarps) {
        const int output_subtile = output_subtile_for_warp(physical_warp);
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int phase = stage_phase(iteration);
            if (iteration > 0) {
                const int old_phase = previous_stage_phase(iteration);
                // P is read independently by dV and by these dS warps.
                wait(dv_ready, old_phase);
                wait(probability_consumed, old_phase);
            }
            wait(score_ready, phase);
            wait(stats_ready, phase);
            tensor_after_thread_sync();
            make_probability_half(
                score_tmem,
                storage,
                output_subtile,
                0,
                iteration == 0,
                g.beta_log2e
            );
            make_probability_half(
                score_tmem,
                storage,
                output_subtile,
                1,
                iteration == 0,
                g.beta_log2e
            );
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(probability_ready);
            }

            wait(dp_ready, phase);
            if (iteration > 0) {
                const int old_phase = previous_stage_phase(iteration);
                // Both dQ and dK must retire the old dS payload before reuse.
                wait(dq_ready, old_phase);
                wait(dk_ready, old_phase);
            }
            tensor_after_thread_sync();
            make_ds_half(
                dp_tmem,
                storage,
                output_subtile,
                0,
                g.beta
            );
            make_ds_half(
                dp_tmem,
                storage,
                output_subtile,
                1,
                g.beta
            );
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(probability_consumed);
                arrive(stats_consumed);
                arrive(ds_ready);
            }
        }
        if (physical_warp == 0) {
            wait(kernel_complete, 0);
            tensor_after_thread_sync();
        }
    } else if (physical_warp == kTensorIssueWarp && lane == 0) {
        wait(persistent_ready, 0);
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int phase = stage_phase(iteration);
            wait(query_ready, phase);
            base::issue_score_or_dp(
                score_tmem,
                storage.k,
                storage.q,
                score_ready
            );

            wait(probability_ready, phase);
            tensor_after_thread_sync();
            if (iteration > 0) {
                wait(dq_drained, previous_stage_phase(iteration));
                tensor_after_thread_sync();
            }
            if (iteration == 0) {
                base::issue_gradient_ab<0>(
                    dv_tmem,
                    storage.probability,
                    storage.dout,
                    dv_ready
                );
            } else {
                base::issue_gradient_ab<1>(
                    dv_tmem,
                    storage.probability,
                    storage.dout,
                    dv_ready
                );
            }
            base::issue_score_or_dp(
                dp_tmem,
                storage.v,
                storage.dout,
                dp_ready
            );

            wait(ds_ready, phase);
            tensor_after_thread_sync();
            base::issue_gradient_atb(
                dq_tmem,
                storage.ds,
                storage.k,
                dq_ready
            );
            if (iteration == 0) {
                base::issue_gradient_ab<0>(
                    dk_tmem,
                    storage.ds,
                    storage.q,
                    dk_ready
                );
            } else {
                base::issue_gradient_ab<1>(
                    dk_tmem,
                    storage.ds,
                    storage.q,
                    dk_ready
                );
            }
        }
    } else if (
        physical_warp >= kReduceWarpBase &&
        physical_warp < kReduceWarpBase + kReduceWarps
    ) {
        const int logical_warp = physical_warp - kReduceWarpBase;
        int publication_sequence = 0;
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int phase = stage_phase(iteration);
            wait(dq_ready, phase);
            tensor_after_thread_sync();
            publish_gradient_tile(
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

        const int last_phase = stage_phase(iteration_count - 1);
        wait(dk_ready, last_phase);
        tensor_after_thread_sync();
        publish_gradient_tile(
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
        publish_gradient_tile(
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
    v421_d128_gqa_e4m3_async_production_bshd::main_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v421_d128_gqa_e4m3_async_production_bshd

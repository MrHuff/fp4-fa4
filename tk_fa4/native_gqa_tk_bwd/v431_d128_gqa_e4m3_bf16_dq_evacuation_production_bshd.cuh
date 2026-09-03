#pragma once

#include "v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd.cuh"

// D128 exact-native production-backward experiment derived from v429/v424.
// Four reducer warps evacuate all of dQ from its aliased TMEM page directly
// into one descriptor-compatible 128x128 BF16 shared buffer before releasing
// dP(next).  The buffer is four independent D32 tiles, so the existing
// additive gradient TMA descriptor and coordinates remain unchanged.  dK/dV
// retain v429's two-stage direct owner-x32 publication path.
namespace tkfa4::native_gqa_tk_bwd::v431_d128_gqa_e4m3_bf16_dq_evacuation_production_bshd {

namespace prior =
    tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd;
using namespace prior;

struct shared_storage {
    prior::shared_storage common;
    gradient_chunk_tile dq_evacuation[kDepthChunks];
};

// v429 common storage is 146 KiB.  One full BF16 dQ tile contributes 32 KiB.
static_assert(sizeof(prior::shared_storage) == 146 * 1024);
static_assert(sizeof(gradient_chunk_tile) == 8 * 1024);
static_assert(sizeof(shared_storage) == 178 * 1024);
static_assert(sizeof(shared_storage) < 232448);

// Every reducer warp owns one physical 32-row stripe.  Each depth chunk uses
// v429's audited owner x32 load, immediate tensor-load wait, vectorized 1/256
// BF16 pack, and direct swizzled shared store.  No raw accumulator payload is
// live across chunks, which keeps this path spill-proof by construction.
__device__ __forceinline__ void evacuate_dq_to_bf16(
    const gradient_tmem_tile &source,
    gradient_chunk_tile (&destination)[kDepthChunks],
    semaphore &source_drained,
    int logical_warp,
    int lane
) {
#pragma unroll
    for (int depth_chunk = 0; depth_chunk < kDepthChunks; ++depth_chunk) {
        prior::drain_gradient_chunk_owner_x32_to_bf16(
            source,
            destination[depth_chunk],
            logical_warp,
            depth_chunk,
            lane
        );
    }

    // Every lane has completed all four direct stores and each store path has
    // issued its async-proxy fence before this warp contributes one arrival.
    tensor_before_thread_sync();
    __syncwarp();
    if (lane == 0) {
        arrive(source_drained);
    }
}

// Only logical reducer warp zero publishes dQ.  Waiting on the same collective
// dq_drained phase used by the tensor issuer proves all four reducer warps have
// finished the full BF16 evacuation before the first TMA can be issued.
__device__ __forceinline__ void publish_evacuated_dq(
    const globals::gradient_gl &destination,
    gradient_chunk_tile (&source)[kDepthChunks],
    semaphore &source_drained,
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
    wait(source_drained, phase);
    tensor_after_thread_sync();
    if (lane == 0) {
#pragma unroll
        for (int depth_chunk = 0; depth_chunk < kDepthChunks; ++depth_chunk) {
            warp::tma::store_add_async<
                dim::DEPTH,
                cache_policy::NORMAL
            >(
                destination,
                source[depth_chunk],
                coord<gradient_chunk_tile>{
                    batch,
                    sequence_tile,
                    head,
                    depth_chunk,
                }
            );
        }
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
    __shared__ alignas(16) semaphore dq_evacuation_reusable;
    __shared__ alignas(16) semaphore publication_ready[
        kGradientPublicationStages
    ];
    __shared__ alignas(16) semaphore publication_reusable[
        kGradientPublicationStages
    ];
    __shared__ alignas(16) semaphore kernel_complete;

    const int physical_warp = warpid();
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
        init_semaphore(dq_evacuation_reusable, 0, 1);
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

    if (physical_warp == kLoaderWarp && laneid() == 0) {
        tma::expect_bytes(
            persistent_ready,
            sizeof(storage.common.k) + sizeof(storage.common.v)
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.common.k,
            g.k,
            coord<operand_tile>{batch, key_tile, kv_head, 0},
            persistent_ready
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.common.v,
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
                sizeof(storage.common.q[stage]) +
                    sizeof(storage.common.dout[stage])
            );
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.common.q[stage],
                g.q,
                operand_coordinate,
                query_ready[stage]
            );
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.common.dout[stage],
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
                sizeof(storage.common.lstat[stage]) +
                    sizeof(storage.common.dstat[stage])
            );
            tma::load_async(
                storage.common.lstat[stage],
                g.lstat,
                stats_coordinate,
                stats_ready[stage]
            );
            tma::load_async(
                storage.common.dstat[stage],
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
            prior::make_probability_half(
                score_tmem,
                storage.common,
                output_subtile,
                0,
                stage,
                iteration == 0,
                g.beta_log2e,
                nullptr
            );
            prior::make_probability_half(
                score_tmem,
                storage.common,
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
            if (laneid() == 0) {
                arrive(probability_ready);
            }

            wait(dp_ready, phase);
            if (iteration > 0) {
                const int old_phase = iteration_phase(iteration - 1);
                wait(dq_ready, old_phase);
                wait(dk_ready, old_phase);
            }
            tensor_after_thread_sync();
            prior::make_ds_half(
                dp_tmem,
                storage.common,
                output_subtile,
                0,
                stage,
                g.beta
            );
            prior::make_ds_half(
                dp_tmem,
                storage.common,
                output_subtile,
                1,
                stage,
                g.beta
            );
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (laneid() == 0) {
                arrive(probability_consumed);
                arrive(stats_consumed[stage]);
                arrive(ds_ready);
            }
        }
        if (physical_warp == 0) {
            wait(kernel_complete, 0);
            tensor_after_thread_sync();
        }
    } else if (physical_warp == kTensorIssueWarp && laneid() == 0) {
        wait(persistent_ready, 0);

        wait(query_ready[0], 0);
        core::issue_score_or_dp(
            score_tmem,
            storage.common.k,
            storage.common.q[0],
            score_ready
        );
        core::issue_score_or_dp(
            dp_tmem,
            storage.common.v,
            storage.common.dout[0],
            dp_ready
        );

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            const int phase = iteration_phase(iteration);
            const bool has_next = iteration + 1 < iteration_count;
            const int next_iteration = iteration + 1;
            const int next_stage = next_iteration & (kInputStages - 1);

            if (has_next) {
                wait(score_consumed, phase);
                wait(
                    query_ready[next_stage],
                    input_stage_epoch_phase(next_iteration)
                );
                tensor_after_thread_sync();
                core::issue_score_or_dp(
                    score_tmem,
                    storage.common.k,
                    storage.common.q[next_stage],
                    score_ready
                );
            }

            wait(probability_ready, phase);
            tensor_after_thread_sync();
            if (iteration == 0) {
                core::issue_gradient_ab<0>(
                    dv_tmem,
                    storage.common.probability,
                    storage.common.dout[stage],
                    dv_ready
                );
            } else {
                core::issue_gradient_ab<1>(
                    dv_tmem,
                    storage.common.probability,
                    storage.common.dout[stage],
                    dv_ready
                );
            }

            wait(ds_ready, phase);
            tensor_after_thread_sync();
            core::issue_gradient_atb(
                dq_tmem,
                storage.common.ds,
                storage.common.k,
                dq_ready
            );
            if (iteration == 0) {
                core::issue_gradient_ab<0>(
                    dk_tmem,
                    storage.common.ds,
                    storage.common.q[stage],
                    dk_ready
                );
            } else {
                core::issue_gradient_ab<1>(
                    dk_tmem,
                    storage.common.ds,
                    storage.common.q[stage],
                    dk_ready
                );
            }

            // dP(next) may reuse the dQ TMEM page as soon as every reducer
            // warp has evacuated all four D32 chunks to the full BF16 buffer.
            if (has_next) {
                wait(dq_drained, phase);
                tensor_after_thread_sync();
                core::issue_score_or_dp(
                    dp_tmem,
                    storage.common.v,
                    storage.common.dout[next_stage],
                    dp_ready
                );
            }

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
        const int lane = laneid();
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int phase = iteration_phase(iteration);
            wait(dq_ready, phase);
            tensor_after_thread_sync();

            // A single full shared buffer is reused across dQ iterations.  Its
            // previous four TMA groups must all have stopped reading before
            // any reducer warp can overwrite even one chunk.
            if (iteration > 0) {
                const int reuse_phase = iteration_phase(iteration - 1);
                if (logical_warp == 0 && lane == 0) {
                    warp::tma::store_async_read_wait<0>();
                    arrive(dq_evacuation_reusable);
                }
                wait(dq_evacuation_reusable, reuse_phase);
                tensor_after_thread_sync();
            }

            evacuate_dq_to_bf16(
                dq_tmem,
                storage.dq_evacuation,
                dq_drained,
                logical_warp,
                lane
            );
            publish_evacuated_dq(
                g.dq,
                storage.dq_evacuation,
                dq_drained,
                phase,
                batch,
                key_tile + iteration,
                query_head,
                logical_warp,
                lane
            );
        }

        // This sequence is only needed by the later dK/dV publication path;
        // keep it dead across the register-critical dQ evacuation loop.
        int publication_sequence = 0;
        const int last_phase = iteration_phase(iteration_count - 1);
        wait(dk_ready, last_phase);
        tensor_after_thread_sync();
        prior::publish_gradient_tile_owner_x32(
            dk_tmem,
            g.dk,
            storage.common.gradient,
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
        prior::publish_gradient_tile_owner_x32(
            dv_tmem,
            g.dv,
            storage.common.gradient,
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
    v431_d128_gqa_e4m3_bf16_dq_evacuation_production_bshd::main_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v431_d128_gqa_e4m3_bf16_dq_evacuation_production_bshd

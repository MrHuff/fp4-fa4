#pragma once

#include "v433_d128_gqa_e4m3_head_fast_raster_production_bshd.cuh"

// B2-targeted sequential owner-2 experiment derived from final v433.  One CTA
// retains K/V and FP32 dK/dV while traversing two adjacent query heads.  dQ is
// still evacuated and published per head.  B1 keeps the exact v433 launch.
namespace tkfa4::native_gqa_tk_bwd::v437_d128_gqa_e4m3_b2_owner2_production_bshd {

namespace prior =
    tkfa4::native_gqa_tk_bwd::v433_d128_gqa_e4m3_head_fast_raster_production_bshd;
namespace x32 =
    tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd;
namespace core =
    tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd;

using prior::attention_tmem_tile;
using prior::globals;
using prior::gradient_tmem_tile;
using prior::kComputeWarps;
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
static_assert(kQueryHeads == 32 && kKvHeads == 8);
static_assert(kHeadsPerOwner == 2 && kHeadPairs == 16);
static_assert(kPairOwnersPerKvHead == 2);

__global__ __launch_bounds__(kThreads, 1)
void owner2_kernel(const __grid_constant__ globals g) {
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
    const int iterations_per_head =
        g.sequence / kQueryTile - key_tile;
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
                prior::make_probability_half(
                    score_tmem,
                    storage,
                    output_subtile,
                    0,
                    stage,
                    iteration == 0,
                    g.beta_log2e,
                    nullptr
                );
                prior::make_probability_half(
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
                prior::make_ds_half(
                    dp_tmem,
                    storage,
                    output_subtile,
                    0,
                    stage,
                    g.beta
                );
                prior::make_ds_half(
                    dp_tmem,
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
                if (work == 0) {
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
                if (work == 0) {
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
    owner2_kernel<<<grid, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v437_d128_gqa_e4m3_b2_owner2_production_bshd

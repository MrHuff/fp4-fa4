#pragma once

#include "v424_d128_gqa_e4m3_score_release_async_production_bshd.cuh"
#include "../fp4_fa4_fwd/stage2_ex2_alu_helpers.cuh"

// Combined D128 candidate: preserve v424's early score-TMEM release and
// dP(next) submission schedule, while replaying each probability owner word
// with v423's native-first/degree-2 period-2 packed exponent policy.
namespace tkfa4::native_gqa_tk_bwd::v426_d128_gqa_e4m3_score_release_d2p2_async_production_bshd {

namespace predecessor =
    tkfa4::native_gqa_tk_bwd::v424_d128_gqa_e4m3_score_release_async_production_bshd;
namespace core =
    tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd;
namespace async =
    tkfa4::native_gqa_tk_bwd::v421_d128_gqa_e4m3_async_production_bshd;
namespace d64 =
    tkfa4::native_gqa_tk_bwd::v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds;

using namespace predecessor;

__device__ __forceinline__ float2 probability_log2_pair(
    const d64::owner_aligned_fp32_half &score,
    shared_storage &storage,
    int pair,
    int key_row,
    int query_column_base,
    int input_stage,
    bool diagonal,
    float beta_log2e
) {
    const int local_column = 2 * pair;
    const float2 statistic = *reinterpret_cast<const float2 *>(
        &storage.lstat[input_stage][query_column_base + local_column]
    );
    float2 value = score.pairs[pair];
    value.x = value.x * beta_log2e + statistic.x;
    value.y = value.y * beta_log2e + statistic.y;
    if (diagonal) {
        constexpr float kNegInf =
            kittens::base_types::constants<float>::neg_infty();
        if (key_row > query_column_base + local_column) {
            value.x = kNegInf;
        }
        if (key_row > query_column_base + local_column + 1) {
            value.y = kNegInf;
        }
    }
    return d64::clamp_probability_log2(value);
}

__device__ __forceinline__ uint32_t probability_word_d2p2(
    const d64::owner_aligned_fp32_half &score,
    shared_storage &storage,
    int word,
    int key_row,
    int query_column_base,
    int input_stage,
    bool diagonal,
    float beta_log2e
) {
    const int native_pair = 2 * word;
    const int degree2_pair = native_pair + 1;
    const float2 native_log2 = probability_log2_pair(
        score,
        storage,
        native_pair,
        key_row,
        query_column_base,
        input_stage,
        diagonal,
        beta_log2e
    );
    const float2 native_probability =
        d64::exp2_native_f32x2(native_log2);
    const float2 degree2_log2 = probability_log2_pair(
        score,
        storage,
        degree2_pair,
        key_row,
        query_column_base,
        input_stage,
        diagonal,
        beta_log2e
    );
    const float2 degree2_probability =
        fp4pv_ex2_alu_emulation_degree2_f32x2(degree2_log2);
    return d64::pack_owner_e4m3_word(
        native_probability,
        degree2_probability
    );
}

__device__ __forceinline__ void make_probability_half_d2p2(
    const attention_tmem_tile &score_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int input_stage,
    bool diagonal,
    float beta_log2e,
    semaphore *score_consumed
) {
    d64::owner_aligned_fp32_half score;
    d64::load_owner_aligned_fp32_half(
        score,
        score_tmem,
        output_subtile,
        column_half
    );
    tensor_load_wait();

    // This is v424's reviewed release point: all score values used by this
    // warp are in registers before the issuer is allowed to reuse score TMEM.
    if (score_consumed != nullptr) {
        tensor_before_thread_sync();
        __syncwarp();
        if (kittens::laneid() == 0) {
            arrive(*score_consumed);
        }
    }

    const int lane_row = kittens::laneid() & 15;
    const int lane_column_base = 32 * (kittens::laneid() >> 4);
    const int key_row = output_subtile * 16 + lane_row;
    const int query_column_base =
        column_half * kColumnHalf + lane_column_base;
    auto destination =
        storage.probability.template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    const uint32_t shared_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&destination.data[0])
    );

#pragma unroll
    for (int word_pair = 0; word_pair < kColumnHalf / 16; ++word_pair) {
        const int first_word = 2 * word_pair;
        const int second_word = first_word + 1;
        const uint32_t packed_first = probability_word_d2p2(
            score,
            storage,
            first_word,
            key_row,
            query_column_base,
            input_stage,
            diagonal,
            beta_log2e
        );
        const uint32_t packed_second = probability_word_d2p2(
            score,
            storage,
            second_word,
            key_row,
            query_column_base,
            input_stage,
            diagonal,
            beta_log2e
        );
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
            make_probability_half_d2p2(
                score_tmem,
                storage,
                output_subtile,
                0,
                stage,
                iteration == 0,
                g.beta_log2e,
                nullptr
            );
            make_probability_half_d2p2(
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
            async::publish_gradient_tile(
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
        async::publish_gradient_tile(
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
        async::publish_gradient_tile(
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
    v426_d128_gqa_e4m3_score_release_d2p2_async_production_bshd::main_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


}  // namespace tkfa4::native_gqa_tk_bwd::v426_d128_gqa_e4m3_score_release_d2p2_async_production_bshd

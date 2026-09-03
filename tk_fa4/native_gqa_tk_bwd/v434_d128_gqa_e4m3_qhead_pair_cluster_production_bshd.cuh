#pragma once

#include "v432_d128_gqa_e4m3_full_gradient_tma_production_bshd.cuh"

// Bounded D128 GQA cluster experiment derived from final v432.  The two CTAs
// in a cluster own adjacent query heads from the same KV head.  Rank 0
// multicasts K/V to rank 1, while Q/dO/stats and dQ remain private.  At the
// epilogue, rank 1 exposes its rounded BF16 dK/dV tile through DSM and rank 0
// performs the only full-width additive TMA publication for the pair.
namespace tkfa4::native_gqa_tk_bwd::v434_d128_gqa_e4m3_qhead_pair_cluster_production_bshd {

namespace prior =
    tkfa4::native_gqa_tk_bwd::v432_d128_gqa_e4m3_full_gradient_tma_production_bshd;
namespace x32 =
    tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd;
namespace core =
    tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd;

using prior::attention_tmem_tile;
using prior::globals;
using prior::gradient_full_tile;
using prior::gradient_tmem_tile;
using prior::kComputeWarps;
using prior::kDepthChunk;
using prior::kDepthChunks;
using prior::kDkTmemOffset;
using prior::kDpDqTmemOffset;
using prior::kDvTmemOffset;
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

constexpr int kClusterCtas = 2;
constexpr int kHeadPairs = kQueryHeads / kClusterCtas;
constexpr uint16_t kClusterMask = 0x3u;
static_assert(kQueryHeads == 32 && kKvHeads == 8);
static_assert(kHeadPairs == 16);

// Read a packed BF16 pair from the same shared-tile address in a peer CTA.
// The producer/consumer cluster mbarriers around this load define the source
// lifetime; this helper deliberately performs no implicit synchronization.
__device__ __forceinline__ void cluster_load_bf16_2x4(
    uint32_t local_address,
    int source_rank,
    bf16_2 &value0,
    bf16_2 &value1,
    bf16_2 &value2,
    bf16_2 &value3
) {
    uint32_t remote_address = 0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(remote_address)
        : "r"(local_address), "r"(source_rank)
    );
    uint32_t bits0 = 0;
    uint32_t bits1 = 0;
    uint32_t bits2 = 0;
    uint32_t bits3 = 0;
    asm volatile(
        "ld.shared::cluster.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(bits0), "=r"(bits1), "=r"(bits2), "=r"(bits3)
        : "r"(remote_address)
        : "memory"
    );
    value0 = *reinterpret_cast<const bf16_2 *>(&bits0);
    value1 = *reinterpret_cast<const bf16_2 *>(&bits1);
    value2 = *reinterpret_cast<const bf16_2 *>(&bits2);
    value3 = *reinterpret_cast<const bf16_2 *>(&bits3);
}

// Rank 0 drains its FP32 accumulator, first rounds each head independently to
// BF16 (matching v432's publication boundary), adds rank 1's BF16 DSM payload,
// rounds once more, and fills its local descriptor-compatible output tile.
__device__ __forceinline__ void drain_gradient_pair_owner_x32(
    const gradient_tmem_tile &source,
    gradient_full_tile &destination,
    semaphore &source_drained,
    int logical_warp,
    int lane
) {
    constexpr float kOutputScale = 1.0f / 256.0f;
    const float2 output_scale = make_float2(kOutputScale, kOutputScale);
    const int physical_row = logical_warp * 32 + lane;
    const uint32_t source_row =
        source.addr + (static_cast<uint32_t>(logical_warp * 32) << 16);
    const uint32_t destination_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(destination.data)
    );

#pragma unroll 1
    for (int depth_chunk = 0; depth_chunk < kDepthChunks; ++depth_chunk) {
        uint32_t values[kDepthChunk];
        x32::load_tmem_owner_x32(
            values,
            source_row + depth_chunk * kDepthChunk
        );
        tensor_load_wait();

#pragma unroll
        for (int column = 0; column < kDepthChunk; column += 8) {
            const int output_column =
                depth_chunk * kDepthChunk + column;
            const uint32_t output_address = gradient_full_tile::idx(
                destination_base,
                {physical_row, output_column}
            );
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
            const bf16_2 own0 = __float22bfloat162_rn(
                __fmul2_rn(value0, output_scale)
            );
            const bf16_2 own1 = __float22bfloat162_rn(
                __fmul2_rn(value1, output_scale)
            );
            const bf16_2 own2 = __float22bfloat162_rn(
                __fmul2_rn(value2, output_scale)
            );
            const bf16_2 own3 = __float22bfloat162_rn(
                __fmul2_rn(value3, output_scale)
            );
            bf16_2 peer0;
            bf16_2 peer1;
            bf16_2 peer2;
            bf16_2 peer3;
            cluster_load_bf16_2x4(
                output_address,
                1,
                peer0,
                peer1,
                peer2,
                peer3
            );
            const float2 own_float0 = __bfloat1622float2(own0);
            const float2 own_float1 = __bfloat1622float2(own1);
            const float2 own_float2 = __bfloat1622float2(own2);
            const float2 own_float3 = __bfloat1622float2(own3);
            const float2 peer_float0 = __bfloat1622float2(peer0);
            const float2 peer_float1 = __bfloat1622float2(peer1);
            const float2 peer_float2 = __bfloat1622float2(peer2);
            const float2 peer_float3 = __bfloat1622float2(peer3);
            const bf16_2 packed0 = __float22bfloat162_rn(make_float2(
                own_float0.x + peer_float0.x,
                own_float0.y + peer_float0.y
            ));
            const bf16_2 packed1 = __float22bfloat162_rn(make_float2(
                own_float1.x + peer_float1.x,
                own_float1.y + peer_float1.y
            ));
            const bf16_2 packed2 = __float22bfloat162_rn(make_float2(
                own_float2.x + peer_float2.x,
                own_float2.y + peer_float2.y
            ));
            const bf16_2 packed3 = __float22bfloat162_rn(make_float2(
                own_float3.x + peer_float3.x,
                own_float3.y + peer_float3.y
            ));
            x32::store_bf16_x8(
                output_address,
                packed0,
                packed1,
                packed2,
                packed3
            );
        }
    }

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    tensor_before_thread_sync();
    __syncwarp();
    if (lane == 0) {
        arrive(source_drained);
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
    __shared__ alignas(16) semaphore full_gradient_ready;
    __shared__ alignas(16) semaphore full_gradient_reusable;
    __shared__ alignas(16) semaphore peer_gradient_ready;
    __shared__ alignas(16) semaphore peer_gradient_consumed;
    __shared__ alignas(16) semaphore kernel_complete;

    const int physical_warp = warpid();
    const int lane = laneid();
    const int cluster_rank = cluster_ctarank();
    const int3 cluster_index = clusterIdx();
    const int head_pair = cluster_index.x;
    const int key_tile = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int query_head = 2 * head_pair + cluster_rank;
    const int kv_head = head_pair / 2;
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
        init_semaphore(peer_gradient_ready, 0, kReduceWarps);
        init_semaphore(peer_gradient_consumed, 0, kReduceWarps);
        init_semaphore(kernel_complete, 0, 1);
        tma::cluster::expect_bytes(
            persistent_ready,
            sizeof(storage.k) + sizeof(storage.v)
        );
    }
    __syncthreads();
    everyone::tma::cluster::sync();

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
        if (cluster_rank == 0) {
            tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.k,
                g.k,
                coord<operand_tile>{batch, key_tile, kv_head, 0},
                persistent_ready,
                kClusterMask
            );
            tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.v,
                g.v,
                coord<operand_tile>{batch, key_tile, kv_head, 0},
                persistent_ready,
                kClusterMask
            );
        }

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            if (iteration >= kInputStages) {
                const int old_phase =
                    x32::previous_input_epoch_phase(iteration);
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
                sizeof(storage.lstat[stage]) + sizeof(storage.dstat[stage])
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
        const int output_subtile =
            x32::output_subtile_for_warp(physical_warp);
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            const int phase = x32::iteration_phase(iteration);
            const int input_phase =
                x32::input_stage_epoch_phase(iteration);
            if (iteration > 0) {
                const int old_phase =
                    x32::iteration_phase(iteration - 1);
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
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(probability_ready);
            }

            wait(dp_ready, phase);
            if (iteration > 0) {
                const int old_phase =
                    x32::iteration_phase(iteration - 1);
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
            const int phase = x32::iteration_phase(iteration);
            const bool has_next = iteration + 1 < iteration_count;
            const int next_iteration = iteration + 1;
            const int next_stage = next_iteration & (kInputStages - 1);

            if (has_next) {
                wait(score_consumed, phase);
                wait(
                    query_ready[next_stage],
                    x32::input_stage_epoch_phase(next_iteration)
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
    } else if (
        physical_warp >= kReduceWarpBase &&
        physical_warp < kReduceWarpBase + kReduceWarps
    ) {
        const int logical_warp = physical_warp - kReduceWarpBase;
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int phase = x32::iteration_phase(iteration);
            wait(dq_ready, phase);
            tensor_after_thread_sync();
            if (iteration > 0) {
                prior::wait_full_gradient_reuse(
                    full_gradient_reusable,
                    x32::iteration_phase(iteration - 1),
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

        const int last_phase =
            x32::iteration_phase(iteration_count - 1);
        wait(dk_ready, last_phase);
        tensor_after_thread_sync();
        prior::wait_full_gradient_reuse(
            full_gradient_reusable,
            x32::iteration_phase(iteration_count - 1),
            logical_warp,
            lane
        );
        if (cluster_rank == 1) {
            prior::drain_gradient_full_owner_x32(
                dk_tmem,
                storage.gradient,
                full_gradient_ready,
                logical_warp,
                lane
            );
            asm volatile(
                "fence.proxy.async.shared::cluster;" ::: "memory"
            );
            if (lane == 0) {
                tma::cluster::arrive(peer_gradient_ready, 0);
            }
            tma::cluster::wait(peer_gradient_consumed, 0);
            tensor_after_thread_sync();
        } else {
            tma::cluster::wait(peer_gradient_ready, 0);
            tensor_after_thread_sync();
            drain_gradient_pair_owner_x32(
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
            if (lane == 0) {
                tma::cluster::arrive(peer_gradient_consumed, 1);
            }
        }

        wait(dv_ready, last_phase);
        tensor_after_thread_sync();
        prior::wait_full_gradient_reuse(
            full_gradient_reusable,
            x32::iteration_phase(iteration_count),
            logical_warp,
            lane
        );
        if (cluster_rank == 1) {
            prior::drain_gradient_full_owner_x32(
                dv_tmem,
                storage.gradient,
                full_gradient_ready,
                logical_warp,
                lane
            );
            asm volatile(
                "fence.proxy.async.shared::cluster;" ::: "memory"
            );
            if (lane == 0) {
                tma::cluster::arrive(peer_gradient_ready, 0);
            }
            tma::cluster::wait(peer_gradient_consumed, 1);
            tensor_after_thread_sync();
            if (logical_warp == 0 && lane == 0) {
                arrive(kernel_complete);
            }
        } else {
            tma::cluster::wait(peer_gradient_ready, 1);
            tensor_after_thread_sync();
            drain_gradient_pair_owner_x32(
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
            if (lane == 0) {
                tma::cluster::arrive(peer_gradient_consumed, 1);
            }
            if (logical_warp == 0 && lane == 0) {
                warp::tma::store_async_wait<0>();
                arrive(kernel_complete);
            }
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
    kittens::LaunchConfig<true, false> launch_config(
        dim3(
            static_cast<unsigned int>(kQueryHeads),
            static_cast<unsigned int>(q.size(1) / kKeyTile),
            static_cast<unsigned int>(q.size(0))
        ),
        dim3(kThreads, 1, 1),
        0,
        stream,
        dim3(kClusterCtas, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel, g));
}

}  // namespace tkfa4::native_gqa_tk_bwd::v434_d128_gqa_e4m3_qhead_pair_cluster_production_bshd

#pragma once

// Owner-major, two-Hq-head specialization of the register-resident
// score/probability/dP D64 GQA backward.
//
// The correctness control materializes both score and dP as FP32 shared
// tiles and then walks 16K elements with scalar CUDA cores.  This variant
// follows the retained V382 reducer path instead: the eight reducer warps
// drain their native 16x128 TMEM fragments directly into registers, apply
// column statistics and causal masking there, and retain rounded BF16 P in
// registers until dS is formed.  Shared memory therefore only sees the two
// tensor-core operands P and dS, not the intermediate FP32 tiles.

#include "v382_d64_gqa_hkv_register_pd.cuh"
#include "v382_d64_gqa_hkv2_partial.cuh"

namespace tkfa4::native_gqa_tk_bwd::v382_d64_hkv2_register_pd {

namespace register_pd = v382_d64_hkv_register_pd;
namespace partial = v382_d64_hkv2_partial;
namespace control = v382_d64_owner;

using control::dq_lhs_tile;
using control::dq_rhs_tile;
using control::dq_stage_tile;
using control::dq_tmem_tile;
using control::gradient_tmem_tile;
using control::kDepth;
using control::kHeadRatio;
using control::kKvHeads;
using control::kOwnerClusters;
using control::kQueryHeads;
using control::kQueryTile;
using control::kSequence;
using control::p_ds_stage_tile;
using control::query_operands;
using control::score_tmem_tile;
using partial::kHeadsPerSplit;
using partial::kPartialHeads;
using partial::partial_globals;

using register_pd::attention_fp32_register;
using register_pd::attention_bf16_register;
using register_pd::attention_fp32_quarter_register;
using register_pd::attention_bf16_quarter_register;
using register_pd::attention_quarter_stats_register;
using register_pd::apply_diagonal_causal_mask;
using register_pd::make_ds_quarter;
using register_pd::make_probability_quarter;

constexpr int kRegisterPdThreads = 384;
constexpr int kTensorIssueWarp = 8;
constexpr int kLoaderWarp = 9;
constexpr int kExchangeWarp = 10;
constexpr int kStatsWarp = 11;

// score_dp is intentionally absent.  Keep dq_lhs separate from q_dk so its
// cross-CTA gather can overlap dK, while the final drains still share one
// epilogue arena.
struct hkv2_register_pd_storage {
    control::score_k_tile k;
    control::value_tile v;
    dq_rhs_tile k_dq;
    p_ds_stage_tile p_ds;
    query_operands query;
    dq_lhs_tile dq_lhs;
    control::epilogue_arena epilogue;
    sv_fl<kQueryTile> lse_log2;
    sv_fl<kQueryTile> delta;
};

static_assert(sizeof(hkv2_register_pd_storage) < 208 * 1024);
static_assert(kHeadsPerSplit == 2 && kPartialHeads == 16);
static_assert(
    attention_fp32_register::rows == 16 &&
    attention_fp32_register::cols == kQueryTile
);

// Do not set minBlocksPerMultiprocessor here: that caps every thread at 128
// registers and defeats the warp-specialized setmaxnreg redistribution below.
// The retained V382 role-split kernel likewise uses the one-argument form.
__global__ __launch_bounds__(kRegisterPdThreads)
void hkv2_register_pd_bf16_kernel(
    const __grid_constant__ partial_globals globals
) {
    __shared__ alignas(1024) hkv2_register_pd_storage storage;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready;
    __shared__ alignas(16) semaphore score_done;
    __shared__ alignas(16) semaphore dp_done;
    __shared__ alignas(16) semaphore dv_done;
    __shared__ alignas(16) semaphore dk_done;
    __shared__ alignas(16) semaphore dq_done;

    const int physical_warp = warpid();
    const int lane = laneid();
    constexpr int kOwnerHeads = kQueryHeads / kHeadsPerSplit;
    constexpr int kOwnersPerKvHead = kHeadRatio / kHeadsPerSplit;
    const int cta_rank = static_cast<int>(blockIdx.x) & 1;
    const int cluster_linear = static_cast<int>(blockIdx.x) >> 1;
    const int owner_idx = cluster_linear / kOwnerHeads;
    const int partial_head = cluster_linear % kOwnerHeads;
    const int kv_head = partial_head / kOwnersPerKvHead;
    const int ratio_base =
        (partial_head % kOwnersPerKvHead) * kHeadsPerSplit;
    const int batch_idx = static_cast<int>(blockIdx.z);

    // Only warps 0..7 retain P while loading dP.  Concentrate the register
    // budget there and leave sufficient producer/exchange capacity elsewhere.
    if (physical_warp < 8) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;" ::: "memory");
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 152;" ::: "memory");
    }

    if (threadIdx.x == 0) {
        init_semaphore(persistent_ready, 0, 1);
        init_semaphore(query_ready, 0, 1);
        init_semaphore(score_done, 0, 1);
        init_semaphore(dp_done, 0, 1);
        init_semaphore(dv_done, 0, 1);
        init_semaphore(dk_done, 0, 1);
        init_semaphore(dq_done, 0, 1);
    }
    __syncthreads();
    everyone::tma::cluster::sync();

    tensor_allocator<1, 2> tmem_allocator{};
    gradient_tmem_tile dk_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(0);
    gradient_tmem_tile dv_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(64);
    dq_tmem_tile dq_tmem =
        tmem_allocator.template allocate<dq_tmem_tile>(0, 128);
    score_tmem_tile score_dp_tmem =
        tmem_allocator.template allocate<score_tmem_tile>(192);

    if (physical_warp == kLoaderWarp && lane == 0) {
        tma::expect_bytes(
            persistent_ready,
            sizeof(storage.k) + sizeof(storage.v) + sizeof(storage.k_dq)
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k,
            globals.k,
            coord<control::score_k_tile>{
                batch_idx,
                owner_idx * 2 + cta_rank,
                kv_head,
                0
            },
            persistent_ready
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.v,
            globals.v,
            coord<control::value_tile>{
                batch_idx,
                owner_idx * 2 + cta_rank,
                kv_head,
                0
            },
            persistent_ready
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k_dq,
            globals.k,
            coord<dq_rhs_tile>{batch_idx, owner_idx, kv_head, cta_rank},
            persistent_ready
        );
    }
    wait(persistent_ready, 0);
    __syncthreads();
    everyone::tma::cluster::sync();

    bool first_accumulation = true;
    int global_iteration = 0;

#pragma unroll
    for (int local_head = 0; local_head < kHeadsPerSplit; ++local_head) {
        const int query_head =
            kv_head * kHeadRatio + ratio_base + local_head;
        for (
            int query_tile_idx = 2 * owner_idx;
            query_tile_idx < kSequence / kQueryTile;
            ++query_tile_idx, ++global_iteration
        ) {
            const int phase = global_iteration & 1;
            const int output_subtile =
                2 * (physical_warp & 3) + (physical_warp >> 2);

            if (physical_warp == kLoaderWarp && lane == 0) {
                tma::expect_bytes(query_ready, sizeof(query_operands));
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    storage.query.q_score,
                    globals.q,
                    coord<control::score_q_tile>{
                        batch_idx,
                        query_tile_idx * 2 + cta_rank,
                        query_head,
                        0
                    },
                    query_ready
                );
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    storage.query.dout_dp,
                    globals.dout,
                    coord<control::dp_dout_tile>{
                        batch_idx,
                        query_tile_idx * 2 + cta_rank,
                        query_head,
                        0
                    },
                    query_ready
                );
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    storage.query.q_dk,
                    globals.q,
                    coord<control::dkdv_rhs_tile>{
                        batch_idx,
                        query_tile_idx,
                        query_head,
                        cta_rank
                    },
                    query_ready
                );
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    storage.query.dout_dv,
                    globals.dout,
                    coord<control::dkdv_rhs_tile>{
                        batch_idx,
                        query_tile_idx,
                        query_head,
                        cta_rank
                    },
                    query_ready
                );
            }
            if (physical_warp == kStatsWarp) {
                const int stats_base =
                    (batch_idx * kQueryHeads + query_head) * kSequence +
                    query_tile_idx * kQueryTile;
                for (int column = lane; column < kQueryTile; column += 32) {
                    storage.lse_log2[column] =
                        globals.lse_log2[stats_base + column];
                    storage.delta[column] = globals.delta[stats_base + column];
                }
            }
            wait(query_ready, phase);
            __syncthreads();
            everyone::tma::cluster::sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                mm2_ABt(
                    score_dp_tmem,
                    storage.k,
                    storage.query.q_score,
                    score_done
                );
            }
            wait(score_done, phase);
            tensor_after_thread_sync();

            // This object deliberately spans the dP MMA.  It is the rounded
            // P used by dV, so dS and dV observe exactly the same probability.
            attention_bf16_register probability;
            if (physical_warp < 8) {
                attention_fp32_register score;
                group<8>::load_async(score, score_dp_tmem);
                tensor_load_wait();

                const int causal_iteration = query_tile_idx - 2 * owner_idx;
                if (causal_iteration < cta_rank) {
                    warp::neg_infty(score);
                } else if (causal_iteration == cta_rank) {
                    apply_diagonal_causal_mask(
                        score,
                        output_subtile,
                        0
                    );
                }
                make_probability_quarter<0>(
                    probability,
                    score,
                    storage.lse_log2,
                    globals.scale_log2e
                );
                make_probability_quarter<1>(
                    probability,
                    score,
                    storage.lse_log2,
                    globals.scale_log2e
                );
                make_probability_quarter<2>(
                    probability,
                    score,
                    storage.lse_log2,
                    globals.scale_log2e
                );
                make_probability_quarter<3>(
                    probability,
                    score,
                    storage.lse_log2,
                    globals.scale_log2e
                );

                auto p_destination =
                    storage.p_ds.template subtile<16, 128>(
                        {output_subtile, 0}
                    );
                warp::store(p_destination, probability);
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
            }
            __syncthreads();
            everyone::tma::cluster::sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                mm2_ABt(
                    score_dp_tmem,
                    storage.v,
                    storage.query.dout_dp,
                    dp_done
                );
            }
            wait(dp_done, phase);
            tensor_after_thread_sync();

            // Start dV before reducer warps form dS.  P remains untouched in
            // shared memory until dV signals completion.
            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                if (first_accumulation) {
                    mm2_AB(
                        dv_tmem,
                        storage.p_ds,
                        storage.query.dout_dv,
                        dv_done
                    );
                } else {
                    mma2_AB(
                        dv_tmem,
                        storage.p_ds,
                        storage.query.dout_dv,
                        dv_done
                    );
                }
            }

            if (physical_warp < 8) {
                attention_fp32_register dp;
                group<8>::load_async(dp, score_dp_tmem);
                tensor_load_wait();
                make_ds_quarter<0>(
                    probability,
                    dp,
                    storage.delta,
                    globals.scale
                );
                make_ds_quarter<1>(
                    probability,
                    dp,
                    storage.delta,
                    globals.scale
                );
                make_ds_quarter<2>(
                    probability,
                    dp,
                    storage.delta,
                    globals.scale
                );
                make_ds_quarter<3>(
                    probability,
                    dp,
                    storage.delta,
                    globals.scale
                );
            }

            wait(dv_done, phase);
            tensor_after_thread_sync();
            if (physical_warp < 8) {
                auto ds_destination =
                    storage.p_ds.template subtile<16, 128>(
                        {output_subtile, 0}
                    );
                warp::store(ds_destination, probability);
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
            }
            __syncthreads();
            everyone::tma::cluster::sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                if (first_accumulation) {
                    mm2_AB(
                        dk_tmem,
                        storage.p_ds,
                        storage.query.q_dk,
                        dk_done
                    );
                } else {
                    mma2_AB(
                        dk_tmem,
                        storage.p_ds,
                        storage.query.q_dk,
                        dk_done
                    );
                }
            }
            // dq_lhs is deliberately disjoint from the live q_dk operand, so
            // the exchange warp can gather dS while TCGEN computes dK.
            if (physical_warp == kExchangeWarp) {
                p_ds_stage_tile *peer_p_ds =
                    control::cluster_map_shared_ptr(
                        &storage.p_ds,
                        cta_rank ^ 1
                    );
                for (int linear = lane; linear < 256 * 64; linear += 32) {
                    const int key_row_256 = linear / 64;
                    const int query_column_local = linear % 64;
                    const int source_rank = key_row_256 / 128;
                    const int source_row = key_row_256 % 128;
                    const int source_column =
                        cta_rank * 64 + query_column_local;
                    const p_ds_stage_tile &source =
                        source_rank == cta_rank
                            ? storage.p_ds
                            : *peer_p_ds;
                    storage.dq_lhs[{
                        key_row_256,
                        query_column_local
                    }] = source[{source_row, source_column}];
                }
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
            }
            wait(dk_done, phase);
            tensor_after_thread_sync();
            __syncthreads();
            everyone::tma::cluster::sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                mm2_AtB(
                    dq_tmem,
                    storage.dq_lhs,
                    storage.k_dq,
                    dq_done
                );
            }
            wait(dq_done, phase);
            tensor_after_thread_sync();

            if (physical_warp < 4) {
                control::drain_dq(
                    dq_tmem,
                    storage.epilogue.dq,
                    physical_warp
                );
            }
            __syncthreads();
            if (physical_warp == 0) {
                warp::tma::store_add_async<
                    dim::DEPTH,
                    cache_policy::NORMAL
                >(
                    globals.dq_accum,
                    storage.epilogue.dq,
                    coord<dq_stage_tile>{
                        batch_idx,
                        query_tile_idx * 2 + cta_rank,
                        query_head,
                        0
                    }
                );
                warp::tma::store_async_wait();
            }
            __syncthreads();
            everyone::tma::cluster::sync();
            first_accumulation = false;
        }
    }

    if (physical_warp < 4) {
        control::drain_dkdv(
            dk_tmem,
            storage.epilogue.dkdv,
            physical_warp
        );
    }
    __syncthreads();
    if (physical_warp == 0) {
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dk_partial,
            storage.epilogue.dkdv,
            coord<control::dkdv_stage_tile>{
                batch_idx,
                owner_idx * 2 + cta_rank,
                partial_head,
                0
            }
        );
        warp::tma::store_async_wait();
    }
    __syncthreads();

    if (physical_warp < 4) {
        control::drain_dkdv(
            dv_tmem,
            storage.epilogue.dkdv,
            physical_warp
        );
    }
    __syncthreads();
    if (physical_warp == 0) {
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dv_partial,
            storage.epilogue.dkdv,
            coord<control::dkdv_stage_tile>{
                batch_idx,
                owner_idx * 2 + cta_rank,
                partial_head,
                0
            }
        );
        warp::tma::store_async_wait();
    }
    __syncthreads();
    everyone::tma::cluster::sync();
}

inline void launch_hkv2_register_pd_bf16(
    const partial_globals &globals,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(kOwnerClusters * kPartialHeads * 2, 1, globals.batch),
        dim3(kRegisterPdThreads, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        hkv2_register_pd_bf16_kernel,
        globals
    ));
}

inline void launch_finalize(
    const at::Tensor &dq_accum,
    at::Tensor &dq,
    const at::Tensor &dk_partial,
    const at::Tensor &dv_partial,
    at::Tensor &dk,
    at::Tensor &dv,
    cudaStream_t stream
) {
    partial::launch_finalize(
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv,
        stream
    );
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_hkv2_register_pd

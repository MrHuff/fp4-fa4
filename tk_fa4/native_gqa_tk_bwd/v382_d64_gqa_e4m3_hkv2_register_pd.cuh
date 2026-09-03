#pragma once

// Production fixed-scale E4M3(x4) D64/GQA backward using the V382
// owner-major, two-query-head register-P/dS schedule.
//
// This is deliberately separate from the validated one-query-head E4M3
// control.  It preserves that control's BSHD ABI, corrected dense-E4M3
// descriptor stepping, async-proxy publication fences, and encoded 4*dX
// outputs.  Each owner accumulates two Hq heads and publishes one FP32
// dK/dV partial; a pairwise epilogue is the sole dK/dV cross-owner reduction.

#include "v382_d64_gqa_e4m3_owner.cuh"

namespace tkfa4::native_gqa_tk_bwd::v382_d64_e4m3_hkv2_register_pd {

namespace base = v382_d64_e4m3_owner;

using base::dkdv_stage_tile;
using base::dq_lhs_tile;
using base::dq_rhs_tile;
using base::dq_stage_tile;
using base::dq_tmem_tile;
using base::gradient_tmem_tile;
using base::kDepth;
using base::kGradientOutputScale;
using base::kHeadRatio;
using base::kKvHeads;
using base::kOwnerClusters;
using base::kQueryHeads;
using base::kQueryTile;
using base::kSequence;
using base::main_globals;
using base::p_ds_stage_tile;
using base::query_operands;
using base::score_tmem_tile;

constexpr int kHeadsPerOwner = 2;
constexpr int kOwnersPerKvHead = kHeadRatio / kHeadsPerOwner;
constexpr int kPartialHeads = kKvHeads * kOwnersPerKvHead;
constexpr int kThreads = 384;
constexpr int kTensorIssueWarp = 8;
constexpr int kLoaderWarp = 9;
constexpr int kExchangeWarp = 10;
constexpr int kStatsWarp = 11;

static_assert(kHeadRatio == 4);
static_assert(kHeadsPerOwner == 2 && kPartialHeads == 16);

using attention_fp32_register = rt_fl<16, 128>;
using attention_e4m3_register = rt_fp8e4m3<16, 128>;
using attention_fp32_quarter_register = rt_fl<16, 32>;
using attention_e4m3_quarter_register = rt_fp8e4m3<16, 32>;
using attention_quarter_stats_register =
    typename attention_fp32_quarter_register::row_vec;

// score/dP stay in TMEM and reducer registers.  Keeping query and dq_lhs
// disjoint lets the exchange warp gather dS while the tensor warp issues dK.
struct shared_storage {
    base::score_k_tile k;
    base::value_tile v;
    dq_rhs_tile k_dq;
    p_ds_stage_tile p_ds;
    query_operands query;
    dq_lhs_tile dq_lhs;
    base::epilogue_arena epilogue;
    sv_fl<kQueryTile> lstat;
    sv_fl<kQueryTile> dstat;
};

static_assert(sizeof(shared_storage) < 208 * 1024);

template <typename Tile>
__device__ __forceinline__ void apply_diagonal_causal_mask(
    Tile &scores,
    int output_subtile
) {
    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    const int key_row_base = output_subtile * 16;
    warp::apply(scores, scores, [=](int row, int column, float value) {
        return key_row_base + row > column ? kNegInf : value;
    });
}

template <int Quarter>
__device__ __forceinline__ void make_probability_quarter(
    attention_e4m3_register &probability_ds,
    attention_fp32_register &score,
    const sv_fl<kQueryTile> &lstat,
    float beta_log2e
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    auto &score_quarter = *reinterpret_cast<
        attention_fp32_quarter_register *
    >(&score.tiles[0][Quarter * 2]);
    auto &probability_quarter = *reinterpret_cast<
        attention_e4m3_quarter_register *
    >(&probability_ds.tiles[0][Quarter]);
    attention_quarter_stats_register lstat_quarter;
    warp::load(lstat_quarter, lstat.template subvec<32>(Quarter));
    warp::mul(score_quarter, score_quarter, beta_log2e);
    warp::add_col(score_quarter, score_quarter, lstat_quarter);
    warp::exp2(score_quarter, score_quarter);
    warp::copy(probability_quarter, score_quarter);
}

template <int Quarter>
__device__ __forceinline__ void make_ds_quarter(
    attention_e4m3_register &probability_ds,
    attention_fp32_register &dp,
    const sv_fl<kQueryTile> &dstat,
    float beta
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    auto &dp_quarter = *reinterpret_cast<
        attention_fp32_quarter_register *
    >(&dp.tiles[0][Quarter * 2]);
    auto &probability_ds_quarter = *reinterpret_cast<
        attention_e4m3_quarter_register *
    >(&probability_ds.tiles[0][Quarter]);
    attention_quarter_stats_register dstat_quarter;
    attention_fp32_quarter_register probability_fp32;
    warp::load(dstat_quarter, dstat.template subvec<32>(Quarter));
    warp::copy(probability_fp32, probability_ds_quarter);
    warp::add_col(dp_quarter, dp_quarter, dstat_quarter);
    warp::mul(dp_quarter, probability_fp32, dp_quarter);
    warp::mul(dp_quarter, dp_quarter, beta);
    warp::copy(probability_ds_quarter, dp_quarter);
}

__global__ __launch_bounds__(kThreads)
void hkv2_register_pd_e4m3_kernel(
    const __grid_constant__ main_globals globals
) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready;
    __shared__ alignas(16) semaphore score_done;
    __shared__ alignas(16) semaphore dp_done;
    __shared__ alignas(16) semaphore dv_done;
    __shared__ alignas(16) semaphore dk_done;
    __shared__ alignas(16) semaphore dq_done;

    const int physical_warp = warpid();
    const int lane = laneid();
    constexpr int kOwnerHeads = kQueryHeads / kHeadsPerOwner;
    const int cta_rank = static_cast<int>(blockIdx.x) & 1;
    const int cluster_linear = static_cast<int>(blockIdx.x) >> 1;
    const int owner_idx = cluster_linear / kOwnerHeads;
    const int partial_head = cluster_linear % kOwnerHeads;
    const int kv_head = partial_head / kOwnersPerKvHead;
    const int ratio_base =
        (partial_head % kOwnersPerKvHead) * kHeadsPerOwner;
    const int batch_idx = static_cast<int>(blockIdx.z);

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
            coord<base::score_k_tile>{
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
            coord<base::value_tile>{
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
    for (int local_head = 0; local_head < kHeadsPerOwner; ++local_head) {
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
                    coord<base::score_q_tile>{
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
                    coord<base::dp_dout_tile>{
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
                    coord<base::dkdv_rhs_tile>{
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
                    coord<base::dkdv_rhs_tile>{
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
                    storage.lstat[column] = globals.lstat[stats_base + column];
                    storage.dstat[column] = globals.dstat[stats_base + column];
                }
            }
            wait(query_ready, phase);
            __syncthreads();
            everyone::tma::cluster::sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                base::fp8_mma2<transpose::N, transpose::N, 0>(
                    score_dp_tmem,
                    storage.k,
                    storage.query.q_score,
                    score_done
                );
            }
            wait(score_done, phase);
            tensor_after_thread_sync();

            // Retain the rounded E4M3 probability across the dP MMA so dV and
            // dS consume exactly the same quantized value.
            attention_e4m3_register probability;
            if (physical_warp < 8) {
                attention_fp32_register score;
                group<8>::load_async(score, score_dp_tmem);
                tensor_load_wait();

                const int causal_iteration = query_tile_idx - 2 * owner_idx;
                if (causal_iteration < cta_rank) {
                    warp::neg_infty(score);
                } else if (causal_iteration == cta_rank) {
                    apply_diagonal_causal_mask(score, output_subtile);
                }
                make_probability_quarter<0>(
                    probability, score, storage.lstat, globals.beta_log2e
                );
                make_probability_quarter<1>(
                    probability, score, storage.lstat, globals.beta_log2e
                );
                make_probability_quarter<2>(
                    probability, score, storage.lstat, globals.beta_log2e
                );
                make_probability_quarter<3>(
                    probability, score, storage.lstat, globals.beta_log2e
                );

                auto p_destination =
                    storage.p_ds.template subtile<16, 128>(
                        {output_subtile, 0}
                    );
                warp::store(p_destination, probability);
                base::async_proxy_fence();
            }
            __syncthreads();
            everyone::tma::cluster::sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                base::fp8_mma2<transpose::N, transpose::N, 0>(
                    score_dp_tmem,
                    storage.v,
                    storage.query.dout_dp,
                    dp_done
                );
            }
            wait(dp_done, phase);
            tensor_after_thread_sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                if (first_accumulation) {
                    base::fp8_mma2<transpose::N, transpose::T, 0>(
                        dv_tmem,
                        storage.p_ds,
                        storage.query.dout_dv,
                        dv_done
                    );
                } else {
                    base::fp8_mma2<transpose::N, transpose::T, 1>(
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
                    probability, dp, storage.dstat, globals.beta
                );
                make_ds_quarter<1>(
                    probability, dp, storage.dstat, globals.beta
                );
                make_ds_quarter<2>(
                    probability, dp, storage.dstat, globals.beta
                );
                make_ds_quarter<3>(
                    probability, dp, storage.dstat, globals.beta
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
                base::async_proxy_fence();
            }
            __syncthreads();
            everyone::tma::cluster::sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                if (first_accumulation) {
                    base::fp8_mma2<transpose::N, transpose::T, 0>(
                        dk_tmem,
                        storage.p_ds,
                        storage.query.q_dk,
                        dk_done
                    );
                } else {
                    base::fp8_mma2<transpose::N, transpose::T, 1>(
                        dk_tmem,
                        storage.p_ds,
                        storage.query.q_dk,
                        dk_done
                    );
                }
            }

            if (physical_warp == kExchangeWarp) {
                p_ds_stage_tile *peer_p_ds = base::cluster_map_shared_ptr(
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
                        source_rank == cta_rank ? storage.p_ds : *peer_p_ds;
                    storage.dq_lhs[{
                        key_row_256,
                        query_column_local
                    }] = source[{source_row, source_column}];
                }
                base::async_proxy_fence();
            }
            wait(dk_done, phase);
            tensor_after_thread_sync();
            __syncthreads();
            everyone::tma::cluster::sync();

            if (
                physical_warp == kTensorIssueWarp &&
                cta_rank == 0 && lane == 0
            ) {
                base::fp8_mma2<transpose::T, transpose::T, 0>(
                    dq_tmem,
                    storage.dq_lhs,
                    storage.k_dq,
                    dq_done
                );
            }
            wait(dq_done, phase);
            tensor_after_thread_sync();

            if (physical_warp < 4) {
                base::drain_dq(
                    dq_tmem,
                    storage.epilogue.dq,
                    physical_warp
                );
            }
            __syncthreads();
            if (physical_warp == 0) {
                base::async_proxy_fence();
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
        base::drain_dkdv(
            dk_tmem,
            storage.epilogue.dkdv,
            physical_warp
        );
    }
    __syncthreads();
    if (physical_warp == 0) {
        base::async_proxy_fence();
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dk_partial,
            storage.epilogue.dkdv,
            coord<dkdv_stage_tile>{
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
        base::drain_dkdv(
            dv_tmem,
            storage.epilogue.dkdv,
            physical_warp
        );
    }
    __syncthreads();
    if (physical_warp == 0) {
        base::async_proxy_fence();
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dv_partial,
            storage.epilogue.dkdv,
            coord<dkdv_stage_tile>{
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

__global__ __launch_bounds__(256, 2)
void finalize_dq_kernel(
    const float *__restrict__ dq_accum,
    bf16 *__restrict__ dq,
    int64_t elements
) {
    for (
        int64_t index =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x
    ) {
        dq[index] = __float2bfloat16_rn(
            dq_accum[index] * kGradientOutputScale
        );
    }
}

__global__ __launch_bounds__(256, 2)
void reduce_pair_partials_kernel(
    const float *__restrict__ dk_partial,
    const float *__restrict__ dv_partial,
    bf16 *__restrict__ dk,
    bf16 *__restrict__ dv,
    int64_t elements
) {
    for (
        int64_t index =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x
    ) {
        const int depth = static_cast<int>(index % kDepth);
        const int64_t row = index / kDepth;
        const int kv_head = static_cast<int>(row % kKvHeads);
        const int64_t batch_sequence = row / kKvHeads;
        const int64_t sequence = batch_sequence % kSequence;
        const int64_t batch = batch_sequence / kSequence;
        const int partial_head = kv_head * kOwnersPerKvHead;
        const int64_t first =
            ((batch * kSequence + sequence) * kPartialHeads + partial_head) *
                kDepth +
            depth;
        const int64_t second = first + kDepth;
        dk[index] = __float2bfloat16_rn(
            (dk_partial[first] + dk_partial[second]) *
            kGradientOutputScale
        );
        dv[index] = __float2bfloat16_rn(
            (dv_partial[first] + dv_partial[second]) *
            kGradientOutputScale
        );
    }
}

inline void launch_hkv2_register_pd_e4m3(
    const main_globals &globals,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(kOwnerClusters * kPartialHeads * 2, 1, batch_size),
        dim3(kThreads, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        hkv2_register_pd_e4m3_kernel,
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
    constexpr int kBlock = 256;
    const int64_t dq_elements = dq.numel();
    const int64_t kv_elements = dk.numel();
    finalize_dq_kernel<<<
        static_cast<unsigned int>((dq_elements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dq_accum.data_ptr()),
        reinterpret_cast<bf16 *>(dq.data_ptr()),
        dq_elements
    );
    CUDACHECK(cudaGetLastError());
    reduce_pair_partials_kernel<<<
        static_cast<unsigned int>((kv_elements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dk_partial.data_ptr()),
        reinterpret_cast<const float *>(dv_partial.data_ptr()),
        reinterpret_cast<bf16 *>(dk.data_ptr()),
        reinterpret_cast<bf16 *>(dv.data_ptr()),
        kv_elements
    );
    CUDACHECK(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_e4m3_hkv2_register_pd

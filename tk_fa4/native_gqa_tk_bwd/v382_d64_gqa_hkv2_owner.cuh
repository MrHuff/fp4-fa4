#pragma once

// Two-Hq-head split-owner variant of the production BSHD V382 D64/GQA
// backward.  grid.y has two owners per Hkv head.  Each cluster retains K/V
// and FP32 dK/dV while walking one pair of Hq heads, then atomically merges
// the pair result into Hkv-shaped FP32 output accumulators.

#include "v382_d64_gqa_hkv_owner.cuh"

namespace tkfa4::native_gqa_tk_bwd::v382_d64_hkv2_owner {

namespace control = v382_d64_owner;
namespace hkv4 = v382_d64_hkv_owner;

using control::dkdv_stage_tile;
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
using control::kThreads;
using control::p_ds_stage_tile;
using control::query_operands;
using control::score_tmem_tile;
using control::shared_storage;

constexpr int kHeadsPerSplit = 2;
constexpr int kSplitsPerKvHead = kHeadRatio / kHeadsPerSplit;
static_assert(kHeadsPerSplit == 2 && kSplitsPerKvHead == 2);

struct split_globals {
    using q_gl = hkv4::hkv_globals::q_gl;
    using k_gl = hkv4::hkv_globals::k_gl;
    using v_gl = hkv4::hkv_globals::v_gl;
    using dout_gl = hkv4::hkv_globals::dout_gl;
    using dq_gl = hkv4::hkv_globals::dq_gl;
    using hkv_accum_gl = gl<
        float,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dkdv_stage_tile, dim::DEPTH>
    >;

    q_gl q;
    k_gl k;
    v_gl v;
    dout_gl dout;
    hkv_accum_gl dk_accum;
    hkv_accum_gl dv_accum;
    dq_gl dq_accum;
    const float *lse_log2;
    const float *delta;
    float scale;
    float scale_log2e;
    int batch;
};

__global__ __launch_bounds__(kThreads, 1)
void hkv2_owner_bf16_kernel(
    const __grid_constant__ split_globals globals
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
    const int cta_rank = static_cast<int>(blockIdx.x) & 1;
    const int owner_idx = static_cast<int>(blockIdx.x) >> 1;
    const int split_owner = static_cast<int>(blockIdx.y);
    const int kv_head = split_owner / kSplitsPerKvHead;
    const int ratio_base =
        (split_owner % kSplitsPerKvHead) * kHeadsPerSplit;
    const int batch_idx = static_cast<int>(blockIdx.z);

    if (physical_warp < 12) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;" ::: "memory");
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 104;" ::: "memory");
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

    if (physical_warp == 13 && lane == 0) {
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

            if (physical_warp == 13 && lane == 0) {
                tma::expect_bytes(query_ready, sizeof(query_operands));
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    storage.phase.query.q_score,
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
                    storage.phase.query.dout_dp,
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
                    storage.phase.query.q_dk,
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
                    storage.phase.query.dout_dv,
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
            if (physical_warp == 15) {
                const int query_base = query_tile_idx * kQueryTile;
                const int stats_base =
                    (batch_idx * kQueryHeads + query_head) * kSequence +
                    query_base;
                for (int column = lane; column < kQueryTile; column += 32) {
                    storage.lse_log2[column] =
                        globals.lse_log2[stats_base + column];
                    storage.delta[column] = globals.delta[stats_base + column];
                }
            }
            wait(query_ready, phase);
            __syncthreads();
            everyone::tma::cluster::sync();

            if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
                mm2_ABt(
                    score_dp_tmem,
                    storage.k,
                    storage.phase.query.q_score,
                    score_done
                );
            }
            wait(score_done, phase);
            tensor_after_thread_sync();
            if (physical_warp < 8) {
                control::drain_score_or_dp(
                    score_dp_tmem,
                    storage.work.score_dp,
                    physical_warp
                );
            }
            __syncthreads();

            if (physical_warp < 8) {
                const int worker = physical_warp * 32 + lane;
                for (
                    int linear = worker;
                    linear < 128 * 128;
                    linear += 256
                ) {
                    const int key_row = linear / 128;
                    const int query_column = linear % 128;
                    const int key_position =
                        owner_idx * 256 + cta_rank * 128 + key_row;
                    const int query_position =
                        query_tile_idx * 128 + query_column;
                    float probability = 0.0f;
                    if (key_position <= query_position) {
                        const float exponent =
                            storage.work.score_dp[{key_row, query_column}] *
                                globals.scale_log2e -
                            storage.lse_log2[query_column];
                        probability = exp2f(exponent);
                    }
                    storage.p_ds[{key_row, query_column}] =
                        __float2bfloat16_rn(probability);
                }
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
            }
            __syncthreads();
            everyone::tma::cluster::sync();

            if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
                mm2_ABt(
                    score_dp_tmem,
                    storage.v,
                    storage.phase.query.dout_dp,
                    dp_done
                );
            }
            wait(dp_done, phase);
            tensor_after_thread_sync();
            if (physical_warp < 8) {
                control::drain_score_or_dp(
                    score_dp_tmem,
                    storage.work.score_dp,
                    physical_warp
                );
            }
            __syncthreads();

            if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
                if (first_accumulation) {
                    mm2_AB(
                        dv_tmem,
                        storage.p_ds,
                        storage.phase.query.dout_dv,
                        dv_done
                    );
                } else {
                    mma2_AB(
                        dv_tmem,
                        storage.p_ds,
                        storage.phase.query.dout_dv,
                        dv_done
                    );
                }
            }
            wait(dv_done, phase);
            tensor_after_thread_sync();

            if (physical_warp < 8) {
                const int worker = physical_warp * 32 + lane;
                for (
                    int linear = worker;
                    linear < 128 * 128;
                    linear += 256
                ) {
                    const int key_row = linear / 128;
                    const int query_column = linear % 128;
                    const float probability = __bfloat162float(
                        storage.p_ds[{key_row, query_column}]
                    );
                    const float centered_dp =
                        storage.work.score_dp[{key_row, query_column}] -
                        storage.delta[query_column];
                    storage.p_ds[{key_row, query_column}] =
                        __float2bfloat16_rn(
                            probability * centered_dp * globals.scale
                        );
                }
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
            }
            __syncthreads();
            everyone::tma::cluster::sync();

            if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
                if (first_accumulation) {
                    mm2_AB(
                        dk_tmem,
                        storage.p_ds,
                        storage.phase.query.q_dk,
                        dk_done
                    );
                } else {
                    mma2_AB(
                        dk_tmem,
                        storage.p_ds,
                        storage.phase.query.q_dk,
                        dk_done
                    );
                }
            }
            wait(dk_done, phase);
            tensor_after_thread_sync();
            __syncthreads();
            everyone::tma::cluster::sync();

            if (physical_warp == 14) {
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
                    storage.phase.dq_lhs[{
                        key_row_256,
                        query_column_local
                    }] = source[{source_row, source_column}];
                }
                asm volatile(
                    "fence.proxy.async.shared::cta;" ::: "memory"
                );
            }
            __syncthreads();
            everyone::tma::cluster::sync();

            if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
                mm2_AtB(
                    dq_tmem,
                    storage.phase.dq_lhs,
                    storage.k_dq,
                    dq_done
                );
            }
            wait(dq_done, phase);
            tensor_after_thread_sync();

            if (physical_warp < 4) {
                control::drain_dq(
                    dq_tmem,
                    storage.work.epilogue.dq,
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
                    storage.work.epilogue.dq,
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
            storage.work.epilogue.dkdv,
            physical_warp
        );
    }
    __syncthreads();
    if (physical_warp == 0) {
        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dk_accum,
            storage.work.epilogue.dkdv,
            coord<dkdv_stage_tile>{
                batch_idx,
                owner_idx * 2 + cta_rank,
                kv_head,
                0
            }
        );
        warp::tma::store_async_wait();
    }
    __syncthreads();

    if (physical_warp < 4) {
        control::drain_dkdv(
            dv_tmem,
            storage.work.epilogue.dkdv,
            physical_warp
        );
    }
    __syncthreads();
    if (physical_warp == 0) {
        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dv_accum,
            storage.work.epilogue.dkdv,
            coord<dkdv_stage_tile>{
                batch_idx,
                owner_idx * 2 + cta_rank,
                kv_head,
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
        dq[index] = __float2bfloat16_rn(dq_accum[index]);
    }
}

__global__ __launch_bounds__(256, 2)
void finalize_dkdv_kernel(
    const float *__restrict__ dk_accum,
    const float *__restrict__ dv_accum,
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
        dk[index] = __float2bfloat16_rn(dk_accum[index]);
        dv[index] = __float2bfloat16_rn(dv_accum[index]);
    }
}

inline void launch_hkv2_owner_bf16(
    const split_globals &globals,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(
            kOwnerClusters * 2,
            kKvHeads * kSplitsPerKvHead,
            globals.batch
        ),
        dim3(kThreads, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        hkv2_owner_bf16_kernel,
        globals
    ));
}

inline void launch_finalize(
    const at::Tensor &dq_accum,
    at::Tensor &dq,
    const at::Tensor &dk_accum,
    const at::Tensor &dv_accum,
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
    finalize_dkdv_kernel<<<
        static_cast<unsigned int>((kv_elements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dk_accum.data_ptr()),
        reinterpret_cast<const float *>(dv_accum.data_ptr()),
        reinterpret_cast<bf16 *>(dk.data_ptr()),
        reinterpret_cast<bf16 *>(dv.data_ptr()),
        kv_elements
    );
    CUDACHECK(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_hkv2_owner

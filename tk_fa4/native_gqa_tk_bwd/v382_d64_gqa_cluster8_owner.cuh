#pragma once

// Cluster-8 D64/GQA specialization of the validated V382 BF16 control.
//
// One 2x4 CTA cluster owns a K256 tile for one Hkv head.  Four independent
// CTA-group-2 pairs process the four associated Hq heads concurrently.  K/V
// are multicast by pair parity, dQ remains an FP32 store-add, and the four
// FP32 dK/dV pair partials are reduced through DSM before one direct BF16
// BSHD store per K128 half.

#include "v382_d64_gqa_hkv_owner.cuh"

namespace tkfa4::native_gqa_tk_bwd::v382_d64_cluster8_owner {

namespace control = v382_d64_owner;

using control::dkdv_stage_tile;
using control::dq_lhs_tile;
using control::dq_rhs_tile;
using control::dq_stage_tile;
using control::dq_tmem_tile;
using control::gradient_tmem_tile;
using control::kDepth;
using control::kHeadRatio;
using control::kKeyRowsPerCluster;
using control::kKvHeads;
using control::kOwnerClusters;
using control::kQueryHeads;
using control::kQueryTile;
using control::kSequence;
using control::kThreads;
using control::p_ds_stage_tile;
using control::query_operands;
using control::score_tmem_tile;

constexpr int kClusterCtas = 8;
constexpr int kPairs = kClusterCtas / 2;
constexpr uint16_t kEvenPairMask = 0x55u;
constexpr uint16_t kOddPairMask = 0xaau;

using dkdv_bf16_stage_tile = st_bf<128, 64>;

// Keep the BF16 reduction destination disjoint from every pair's DSM-visible
// FP32 source.  An in-place reinterpretation can overwrite FP32 rows before a
// sibling reducer warp has loaded them.
struct shared_storage {
    control::score_k_tile k;
    control::value_tile v;
    dq_rhs_tile k_dq;
    p_ds_stage_tile p_ds;
    control::phase_arena phase;
    control::work_arena work;
    float lse_log2[kQueryTile];
    float delta[kQueryTile];
    dkdv_bf16_stage_tile reduced_bf16;
};

static_assert(sizeof(shared_storage) < 208 * 1024);

struct cluster8_globals {
    using q_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<control::score_q_tile, dim::DEPTH>,
        tma::descriptor<control::dkdv_rhs_tile, dim::DEPTH>
    >;
    using k_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<control::score_k_tile, dim::DEPTH>,
        tma::descriptor<dq_rhs_tile, dim::DEPTH>
    >;
    using v_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<control::value_tile, dim::DEPTH>
    >;
    using dout_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<control::dp_dout_tile, dim::DEPTH>,
        tma::descriptor<control::dkdv_rhs_tile, dim::DEPTH>
    >;
    using hkv_out_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dkdv_bf16_stage_tile, dim::DEPTH>
    >;
    using dq_gl = gl<
        float,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dq_stage_tile, dim::DEPTH>
    >;

    q_gl q;
    k_gl k;
    v_gl v;
    dout_gl dout;
    hkv_out_gl dk;
    hkv_out_gl dv;
    dq_gl dq_accum;
    const float *lse_log2;
    const float *delta;
    float scale;
    float scale_log2e;
    int batch;
};

__device__ __forceinline__ uint16_t pair_mask(int pair_base) {
    return static_cast<uint16_t>(0x3u << pair_base);
}

// The sem-taking TK mm2 helpers commit to the hard-coded mask 0b11.  A
// cluster-8 pair must issue the no-semaphore operation and then commit to its
// own absolute two-CTA mask with this helper.
__device__ __forceinline__ void commit_pair(
    semaphore &barrier,
    int pair_base
) {
    tensor_commit<2>(barrier, pair_mask(pair_base));
}

// A generic pointer produced from mapa.shared::cluster is sufficient for the
// CTA-group-2 peer accesses used by the existing control, but it lowers to an
// ordinary ld.shared.  Ranks in the other CTA-group-2 pairs require the DSM
// instruction explicitly.  Keep that distinction local to the four-way
// dK/dV reduction instead of changing the validated pair-local dQ exchange.
__device__ __forceinline__ float cluster_load_shared_f32(
    const float *pointer,
    int source_rank
) {
    const uint32_t local_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    uint32_t remote_address = 0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(remote_address)
        : "r"(local_address), "r"(source_rank)
    );
    uint32_t bits = 0;
    asm volatile(
        "ld.shared::cluster.b32 %0, [%1];\n"
        : "=r"(bits)
        : "r"(remote_address)
        : "memory"
    );
    return __uint_as_float(bits);
}

__device__ __forceinline__ void reduce_pair_partials_to_bf16(
    shared_storage &storage,
    int pair_id,
    int pair_lane,
    int physical_warp
) {
    if (pair_id != 0 || physical_warp >= 8) {
        return;
    }

    for (int linear = laneid(); linear < 16 * 64; linear += 32) {
        const int row = physical_warp * 16 + linear / 64;
        const int column = linear % 64;
        float sum = storage.work.epilogue.dkdv[{row, column}];
#pragma unroll
        for (int source_pair = 1; source_pair < kPairs; ++source_pair) {
            const int source_rank = pair_lane + 2 * source_pair;
            sum += cluster_load_shared_f32(
                &storage.work.epilogue.dkdv[{row, column}],
                source_rank
            );
        }
        storage.reduced_bf16[{row, column}] =
            __float2bfloat16_rn(sum);
    }
}

__global__ __launch_bounds__(kThreads, 1)
void cluster8_owner_bf16_kernel(
    const __grid_constant__ cluster8_globals globals
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
    const int cluster_rank = cluster_ctarank();
    const int pair_lane = cluster_rank & 1;
    const int pair_id = cluster_rank >> 1;
    const int pair_base = cluster_rank - pair_lane;
    const int3 cluster_index = clusterIdx();
    const int owner_idx = cluster_index.x;
    const int kv_head = cluster_index.y;
    const int batch_idx = cluster_index.z;
    const int query_head = kv_head * kHeadRatio + pair_id;

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
        // Each CTA receives three multicast tiles even though ranks 0/1 issue
        // six transactions across the complete cluster.
        tma::cluster::expect_bytes(
            persistent_ready,
            sizeof(storage.k) + sizeof(storage.v) + sizeof(storage.k_dq)
        );
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

    // Rank 0 feeds the even pair lanes; rank 1 feeds the odd pair lanes.  The
    // destination shared-memory offsets and semaphore offsets are identical
    // in all CTAs.
    if (physical_warp == 13 && lane == 0 && cluster_rank < 2) {
        const uint16_t multicast_mask =
            pair_lane == 0 ? kEvenPairMask : kOddPairMask;
        tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k,
            globals.k,
            coord<control::score_k_tile>{
                batch_idx,
                owner_idx * 2 + pair_lane,
                kv_head,
                0
            },
            persistent_ready,
            multicast_mask
        );
        tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.v,
            globals.v,
            coord<control::value_tile>{
                batch_idx,
                owner_idx * 2 + pair_lane,
                kv_head,
                0
            },
            persistent_ready,
            multicast_mask
        );
        tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k_dq,
            globals.k,
            coord<dq_rhs_tile>{
                batch_idx,
                owner_idx,
                kv_head,
                pair_lane
            },
            persistent_ready,
            multicast_mask
        );
    }
    wait(persistent_ready, 0);
    __syncthreads();
    everyone::tma::cluster::sync();

    bool first_accumulation = true;
    int iteration = 0;
    for (
        int query_tile_idx = 2 * owner_idx;
        query_tile_idx < kSequence / kQueryTile;
        ++query_tile_idx, ++iteration
    ) {
        const int phase = iteration & 1;

        if (physical_warp == 13 && lane == 0) {
            tma::expect_bytes(query_ready, sizeof(query_operands));
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.phase.query.q_score,
                globals.q,
                coord<control::score_q_tile>{
                    batch_idx,
                    query_tile_idx * 2 + pair_lane,
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
                    query_tile_idx * 2 + pair_lane,
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
                    pair_lane
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
                    pair_lane
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

        if (physical_warp == 12 && pair_lane == 0 && lane == 0) {
            mm2_ABt(
                score_dp_tmem,
                storage.k,
                storage.phase.query.q_score
            );
            commit_pair(score_done, pair_base);
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
            for (int linear = worker; linear < 128 * 128; linear += 256) {
                const int key_row = linear / 128;
                const int query_column = linear % 128;
                const int key_position =
                    owner_idx * 256 + pair_lane * 128 + key_row;
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
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
        everyone::tma::cluster::sync();

        if (physical_warp == 12 && pair_lane == 0 && lane == 0) {
            mm2_ABt(
                score_dp_tmem,
                storage.v,
                storage.phase.query.dout_dp
            );
            commit_pair(dp_done, pair_base);
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

        if (physical_warp == 12 && pair_lane == 0 && lane == 0) {
            if (first_accumulation) {
                mm2_AB(
                    dv_tmem,
                    storage.p_ds,
                    storage.phase.query.dout_dv
                );
            } else {
                mma2_AB(
                    dv_tmem,
                    storage.p_ds,
                    storage.phase.query.dout_dv
                );
            }
            commit_pair(dv_done, pair_base);
        }
        wait(dv_done, phase);
        tensor_after_thread_sync();

        if (physical_warp < 8) {
            const int worker = physical_warp * 32 + lane;
            for (int linear = worker; linear < 128 * 128; linear += 256) {
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
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
        everyone::tma::cluster::sync();

        if (physical_warp == 12 && pair_lane == 0 && lane == 0) {
            if (first_accumulation) {
                mm2_AB(
                    dk_tmem,
                    storage.p_ds,
                    storage.phase.query.q_dk
                );
            } else {
                mma2_AB(
                    dk_tmem,
                    storage.p_ds,
                    storage.phase.query.q_dk
                );
            }
            commit_pair(dk_done, pair_base);
        }
        wait(dk_done, phase);
        tensor_after_thread_sync();
        __syncthreads();
        everyone::tma::cluster::sync();

        if (physical_warp == 14) {
            p_ds_stage_tile *peer_p_ds = control::cluster_map_shared_ptr(
                &storage.p_ds,
                cluster_rank ^ 1
            );
            for (int linear = lane; linear < 256 * 64; linear += 32) {
                const int key_row_256 = linear / 64;
                const int query_column_local = linear % 64;
                const int source_lane = key_row_256 / 128;
                const int source_row = key_row_256 % 128;
                const int source_column =
                    pair_lane * 64 + query_column_local;
                const p_ds_stage_tile &source =
                    source_lane == pair_lane ? storage.p_ds : *peer_p_ds;
                storage.phase.dq_lhs[{
                    key_row_256,
                    query_column_local
                }] = source[{source_row, source_column}];
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
        everyone::tma::cluster::sync();

        if (physical_warp == 12 && pair_lane == 0 && lane == 0) {
            mm2_AtB(
                dq_tmem,
                storage.phase.dq_lhs,
                storage.k_dq
            );
            commit_pair(dq_done, pair_base);
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
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                globals.dq_accum,
                storage.work.epilogue.dq,
                coord<dq_stage_tile>{
                    batch_idx,
                    query_tile_idx * 2 + pair_lane,
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

    // dK: publish every head's pair-local FP32 accumulator to DSM.
    if (physical_warp < 4) {
        control::drain_dkdv(
            dk_tmem,
            storage.work.epilogue.dkdv,
            physical_warp
        );
    }
    __syncthreads();
    everyone::tma::cluster::sync();
    reduce_pair_partials_to_bf16(
        storage,
        pair_id,
        pair_lane,
        physical_warp
    );
    __syncthreads();
    if (pair_id == 0 && physical_warp == 0) {
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dk,
            storage.reduced_bf16,
            coord<dkdv_bf16_stage_tile>{
                batch_idx,
                owner_idx * 2 + pair_lane,
                kv_head,
                0
            }
        );
        warp::tma::store_async_wait();
    }
    __syncthreads();
    // No pair may overwrite its DSM source before ranks 0/1 complete the
    // reduction and direct store.
    everyone::tma::cluster::sync();

    // dV reuses the FP32 DSM source arena and the BF16 reducer stage only
    // after the preceding full-cluster lifetime boundary.
    if (physical_warp < 4) {
        control::drain_dkdv(
            dv_tmem,
            storage.work.epilogue.dkdv,
            physical_warp
        );
    }
    __syncthreads();
    everyone::tma::cluster::sync();
    reduce_pair_partials_to_bf16(
        storage,
        pair_id,
        pair_lane,
        physical_warp
    );
    __syncthreads();
    if (pair_id == 0 && physical_warp == 0) {
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dv,
            storage.reduced_bf16,
            coord<dkdv_bf16_stage_tile>{
                batch_idx,
                owner_idx * 2 + pair_lane,
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
void finalize_dq_batched_kernel(
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

inline void launch_cluster8_owner_bf16(
    const cluster8_globals &globals,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(kOwnerClusters * 2, kKvHeads * kPairs, globals.batch),
        dim3(kThreads, 1, 1),
        0,
        stream,
        dim3(2, kPairs, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        cluster8_owner_bf16_kernel,
        globals
    ));
}

inline void launch_dq_finalize(
    const at::Tensor &dq_accum,
    at::Tensor &dq,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t elements = dq.numel();
    finalize_dq_batched_kernel<<<
        static_cast<unsigned int>((elements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dq_accum.data_ptr()),
        reinterpret_cast<bf16 *>(dq.data_ptr()),
        elements
    );
    CUDACHECK(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_cluster8_owner

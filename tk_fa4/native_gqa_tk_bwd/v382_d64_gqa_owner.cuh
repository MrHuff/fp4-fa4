#pragma once

// Project-owned D64/GQA extraction of the V382 two-CTA owner topology.
//
// This is deliberately narrow: B1, S4096, Hq32/Hkv8, D64 and BF16
// operands.  It preserves the V382 K256 owner cluster, Q128 causal walk,
// 512-thread role split and CTA-group-2 tensor-core shapes without including
// the historical D192 candidate header.

#include "../deprecated/fa4_common.cuh"

#include <cstdint>
#include <type_traits>

namespace tkfa4::native_gqa_tk_bwd::v382_d64_owner {

constexpr int kBatch = 1;
constexpr int kSequence = 4096;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kHeadRatio = kQueryHeads / kKvHeads;
constexpr int kDepth = 64;
constexpr int kQueryTile = 128;
constexpr int kKeyRowsPerCta = 128;
constexpr int kKeyRowsPerCluster = 256;
constexpr int kOwnerClusters = kSequence / kKeyRowsPerCluster;
constexpr int kThreads = 512;

static_assert(kHeadRatio == 4);
static_assert(kSequence % kKeyRowsPerCluster == 0);

using score_k_tile = st_bf<128, 64>;
using score_q_tile = st_bf<64, 64>;
using value_tile = st_bf<128, 64>;
using dp_dout_tile = st_bf<64, 64>;
using dkdv_rhs_tile = st_bf<128, 32>;
using dq_lhs_tile = st_bf<256, 64>;
using dq_rhs_tile = st_bf<256, 32>;

using score_stage_tile = st_fl<128, 128>;
using p_ds_stage_tile = st_bf<128, 128>;
using dkdv_stage_tile = st_fl<128, 64>;
using dq_stage_tile = st_fl<64, 64>;

using score_tmem_tile = full_tt_fl<128>;
using gradient_tmem_tile = full_tt_fl<64>;
using dq_tmem_tile = half_tt_fl<64>;

struct query_operands {
    score_q_tile q_score;
    dp_dout_tile dout_dp;
    dkdv_rhs_tile q_dk;
    dkdv_rhs_tile dout_dv;
};

static_assert(sizeof(query_operands) == sizeof(dq_lhs_tile));

union phase_arena {
    query_operands query;
    dq_lhs_tile dq_lhs;
};

union epilogue_arena {
    dkdv_stage_tile dkdv;
    dq_stage_tile dq;
};

// score_dp is dead after dS has been formed.  The dQ and final dK/dV drains
// happen strictly after that boundary, so the two workspaces can share the
// same 64 KiB allocation.
union work_arena {
    score_stage_tile score_dp;
    epilogue_arena epilogue;
};

struct shared_storage {
    score_k_tile k;
    value_tile v;
    dq_rhs_tile k_dq;
    p_ds_stage_tile p_ds;
    phase_arena phase;
    work_arena work;
    float lse_log2[kQueryTile];
    float delta[kQueryTile];
};

// SM100 provides enough shared memory for this correctness-first, non-aliased
// D64 map.  Keeping the limit explicit prevents an accidental return to the
// 226 KiB D192 macro layout.
static_assert(sizeof(shared_storage) < 208 * 1024);

struct main_globals {
    using q_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<score_q_tile, dim::ROW>,
        tma::descriptor<dkdv_rhs_tile, dim::ROW>
    >;
    using k_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<score_k_tile, dim::ROW>,
        tma::descriptor<dq_rhs_tile, dim::ROW>
    >;
    using v_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<value_tile, dim::ROW>
    >;
    using dout_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dp_dout_tile, dim::ROW>,
        tma::descriptor<dkdv_rhs_tile, dim::ROW>
    >;
    using partial_gl = gl<
        float,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dkdv_stage_tile, dim::ROW>
    >;
    using dq_gl = gl<
        float,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dq_stage_tile, dim::ROW>
    >;

    q_gl q;
    k_gl k;
    v_gl v;
    dout_gl dout;
    partial_gl dk_partial;
    partial_gl dv_partial;
    dq_gl dq_accum;
    const float *lse_log2;
    const float *delta;
    float scale;
    float scale_log2e;
};

template <typename T>
__device__ __forceinline__ T *cluster_map_shared_ptr(T *pointer, int rank) {
    const uint32_t shared_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    uint32_t mapped_address = 0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(mapped_address)
        : "r"(shared_address), "r"(rank)
    );
    const unsigned long long mapped64 =
        static_cast<unsigned long long>(mapped_address);
    unsigned long long generic_address = 0;
    asm volatile(
        "cvta.shared.u64 %0, %1;\n"
        : "=l"(generic_address)
        : "l"(mapped64)
    );
    return reinterpret_cast<T *>(generic_address);
}

__device__ __forceinline__ void drain_score_or_dp(
    const score_tmem_tile &source,
    score_stage_tile &destination,
    int physical_warp
) {
    const int compute_warp = physical_warp;
    const int output_subtile =
        2 * (compute_warp & 3) + (compute_warp >> 2);
    rt_fl<16, 128> values;
    // Preserve the donor's CTA-group-2 full-TMEM lane mapping.  The eight
    // compute warps jointly load the result; manually slicing by logical row
    // does not match the physical DP-lane layout.
    group<8>::load_async(values, source);
    tensor_load_wait();
    auto destination_slice = destination.template subtile<16, 128>(
        {output_subtile, 0}
    );
    warp::store(destination_slice, values);
}

__device__ __forceinline__ void drain_dq(
    const dq_tmem_tile &source,
    dq_stage_tile &destination,
    int physical_warp
) {
    using slice_tmem = tt_fl<32, 32>;
    rt_fl<32, 32> values;
    const int row_half = physical_warp & 1;
    const int column_half = physical_warp >> 1;
    // V382 Layout-B: each reducer warp owns one logical 32x32 quadrant,
    // while its physical TMEM rows are +{0,32,64,96}.  A 16x64 view mixes
    // the two independent row/column ownership dimensions.
    const slice_tmem source_slice{
        source.addr + ((32 * physical_warp) << 16)
    };
    group<1>::load_async(values, source_slice);
    tensor_load_wait();
    auto destination_slice = destination.template subtile<32, 32>(
        {row_half, column_half}
    );
    warp::store(destination_slice, values);
}

__device__ __forceinline__ void drain_dkdv(
    const gradient_tmem_tile &source,
    dkdv_stage_tile &destination,
    int physical_warp
) {
    rt_fl<32, 64> values;
    // As in V382, the four reducer warps collectively materialize the
    // rank-local K/V rows from the CTA-group-2 full tile.
    group<4>::load_async(values, source);
    tensor_load_wait();
    auto destination_slice = destination.template subtile<32, 64>(
        {physical_warp, 0}
    );
    warp::store(destination_slice, values);
}

__global__ __launch_bounds__(kThreads, 1)
void owner_bf16_kernel(const __grid_constant__ main_globals globals) {
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
    const int query_head = static_cast<int>(blockIdx.y);
    const int kv_head = query_head / kHeadRatio;

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
        tma::load_async<dim::ROW, cache_policy::NORMAL>(
            storage.k,
            globals.k,
            coord<score_k_tile>{
                0,
                kv_head,
                owner_idx * 2 + cta_rank,
                0
            },
            persistent_ready
        );
        tma::load_async<dim::ROW, cache_policy::NORMAL>(
            storage.v,
            globals.v,
            coord<value_tile>{
                0,
                kv_head,
                owner_idx * 2 + cta_rank,
                0
            },
            persistent_ready
        );
        tma::load_async<dim::ROW, cache_policy::NORMAL>(
            storage.k_dq,
            globals.k,
            coord<dq_rhs_tile>{0, kv_head, owner_idx, cta_rank},
            persistent_ready
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
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                storage.phase.query.q_score,
                globals.q,
                coord<score_q_tile>{
                    0,
                    query_head,
                    query_tile_idx * 2 + cta_rank,
                    0
                },
                query_ready
            );
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                storage.phase.query.dout_dp,
                globals.dout,
                coord<dp_dout_tile>{
                    0,
                    query_head,
                    query_tile_idx * 2 + cta_rank,
                    0
                },
                query_ready
            );
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                storage.phase.query.q_dk,
                globals.q,
                coord<dkdv_rhs_tile>{
                    0,
                    query_head,
                    query_tile_idx,
                    cta_rank
                },
                query_ready
            );
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                storage.phase.query.dout_dv,
                globals.dout,
                coord<dkdv_rhs_tile>{
                    0,
                    query_head,
                    query_tile_idx,
                    cta_rank
                },
                query_ready
            );
        }
        if (physical_warp == 15) {
            const int query_base = query_tile_idx * kQueryTile;
            const int stats_base = query_head * kSequence + query_base;
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
            drain_score_or_dp(
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
            drain_score_or_dp(
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
        }
        __syncthreads();
        everyone::tma::cluster::sync();

        // Consume q_dk before phase_arena is repurposed for the gathered dQ
        // left operand.  The TCGEN completion wait is an operand-lifetime
        // boundary: after it returns, warp 14 may safely overwrite q_dk.
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

        // The exchange warp gathers K256 x Q64 dS for the rank-local dQ
        // result.  This is the only cross-CTA operand materialization in the
        // first control implementation.
        if (physical_warp == 14) {
            p_ds_stage_tile *peer_p_ds = cluster_map_shared_ptr(
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
                storage.phase.dq_lhs[{
                    key_row_256,
                    query_column_local
                }] = source[{source_row, source_column}];
            }
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
            drain_dq(dq_tmem, storage.work.epilogue.dq, physical_warp);
        }
        __syncthreads();
        if (physical_warp == 0) {
            warp::tma::store_add_async<dim::ROW, cache_policy::NORMAL>(
                globals.dq_accum,
                storage.work.epilogue.dq,
                coord<dq_stage_tile>{
                    0,
                    query_head,
                    query_tile_idx * 2 + cta_rank,
                    0
                }
            );
            warp::tma::store_async_wait();
        }
        __syncthreads();
        everyone::tma::cluster::sync();
        first_accumulation = false;
    }

    if (physical_warp < 4) {
        drain_dkdv(dk_tmem, storage.work.epilogue.dkdv, physical_warp);
    }
    __syncthreads();
    if (physical_warp == 0) {
        warp::tma::store_async<dim::ROW, cache_policy::NORMAL>(
            globals.dk_partial,
            storage.work.epilogue.dkdv,
            coord<dkdv_stage_tile>{
                0,
                query_head,
                owner_idx * 2 + cta_rank,
                0
            }
        );
        warp::tma::store_async_wait();
    }
    __syncthreads();

    if (physical_warp < 4) {
        drain_dkdv(dv_tmem, storage.work.epilogue.dkdv, physical_warp);
    }
    __syncthreads();
    if (physical_warp == 0) {
        warp::tma::store_async<dim::ROW, cache_policy::NORMAL>(
            globals.dv_partial,
            storage.work.epilogue.dkdv,
            coord<dkdv_stage_tile>{
                0,
                query_head,
                owner_idx * 2 + cta_rank,
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
    bf16 *__restrict__ dq
) {
    constexpr int64_t kElements =
        static_cast<int64_t>(kQueryHeads) * kSequence * kDepth;
    for (
        int64_t index =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < kElements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x
    ) {
        dq[index] = __float2bfloat16_rn(dq_accum[index]);
    }
}

__global__ __launch_bounds__(256, 2)
void reduce_dkdv_kernel(
    const float *__restrict__ dk_partial,
    const float *__restrict__ dv_partial,
    bf16 *__restrict__ dk,
    bf16 *__restrict__ dv
) {
    constexpr int64_t kOutputElements =
        static_cast<int64_t>(kKvHeads) * kSequence * kDepth;
    for (
        int64_t index =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < kOutputElements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x
    ) {
        const int depth = static_cast<int>(index % kDepth);
        const int64_t row = index / kDepth;
        const int sequence = static_cast<int>(row % kSequence);
        const int kv_head = static_cast<int>(row / kSequence);
        const int first_query_head = kv_head * kHeadRatio;
        float dk_sum = 0.0f;
        float dv_sum = 0.0f;
#pragma unroll
        for (int ratio = 0; ratio < kHeadRatio; ++ratio) {
            const int query_head = first_query_head + ratio;
            const int64_t partial_index =
                (static_cast<int64_t>(query_head) * kSequence + sequence) *
                    kDepth +
                depth;
            dk_sum += dk_partial[partial_index];
            dv_sum += dv_partial[partial_index];
        }
        dk[index] = __float2bfloat16_rn(dk_sum);
        dv[index] = __float2bfloat16_rn(dv_sum);
    }
}

inline void launch_owner_bf16(
    const main_globals &globals,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(kOwnerClusters * 2, kQueryHeads, kBatch),
        dim3(kThreads, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        owner_bf16_kernel,
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
    constexpr int64_t kDqElements =
        static_cast<int64_t>(kQueryHeads) * kSequence * kDepth;
    constexpr int64_t kKvElements =
        static_cast<int64_t>(kKvHeads) * kSequence * kDepth;
    finalize_dq_kernel<<<
        static_cast<unsigned int>((kDqElements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dq_accum.data_ptr()),
        reinterpret_cast<bf16 *>(dq.data_ptr())
    );
    CUDACHECK(cudaGetLastError());
    reduce_dkdv_kernel<<<
        static_cast<unsigned int>((kKvElements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dk_partial.data_ptr()),
        reinterpret_cast<const float *>(dv_partial.data_ptr()),
        reinterpret_cast<bf16 *>(dk.data_ptr()),
        reinterpret_cast<bf16 *>(dv.data_ptr())
    );
    CUDACHECK(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_owner

#pragma once

// Production-layout dense-E4M3 D64/GQA extraction of the V382 two-CTA
// owner topology.  The tensor ABI is BSHD, while the forward-produced
// additive statistics retain their established [B,Hq,1,S] layout.
//
// Q/K/V/dO contain the fixed-scale E4M3 encoding of 4*x.  The statistics
// are lstat = 8 - LSE*log2(e) and dstat = -16*sum(O*dO).  The public scale
// argument is the ordinary attention scale (the kernel folds in /16), and
// the BF16 outputs intentionally remain encoded as 4*dX for the existing
// inverse-RoPE/projection consumer to decode.

#include "../deprecated/fa4_common.cuh"

#include <cstdint>
#include <type_traits>

namespace tkfa4::native_gqa_tk_bwd::v382_d64_e4m3_owner {

constexpr int kSequence = 4096;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kHeadRatio = kQueryHeads / kKvHeads;
constexpr int kDepth = 64;
constexpr int kQueryTile = 128;
constexpr int kKeyRowsPerCluster = 256;
constexpr int kOwnerClusters = kSequence / kKeyRowsPerCluster;
constexpr int kThreads = 512;
constexpr float kGradientOutputScale = 1.0f / 256.0f;

static_assert(kHeadRatio == 4);
static_assert(kSequence % kKeyRowsPerCluster == 0);

// Score and dP operands are rank-local M128 x K64 and N64 x K64 tiles.
using score_k_tile = st_fp8e4m3<128, 64>;
using score_q_tile = st_fp8e4m3<64, 64>;
using value_tile = st_fp8e4m3<128, 64>;
using dp_dout_tile = st_fp8e4m3<64, 64>;

// A CTA-group-2 N64 output consumes a rank-local D32 operand slice.  Dense
// E4M3 needs custom MN-major descriptor stepping below, but no duplicated
// N128 operand or post-MMA half selection.
using dkdv_rhs_tile = st_fp8e4m3<128, 32>;

// dQ consumes K256 x Q64 dS and K256 x D32 K per rank.
using dq_lhs_tile = st_fp8e4m3<256, 64>;
using dq_rhs_tile = st_fp8e4m3<256, 32>;

using score_stage_tile = st_fl<128, 128>;
using p_ds_stage_tile = st_fp8e4m3<128, 128>;
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

static_assert(sizeof(query_operands) >= sizeof(dq_lhs_tile));

union phase_arena {
    query_operands query;
    dq_lhs_tile dq_lhs;
};

union epilogue_arena {
    dkdv_stage_tile dkdv;
    dq_stage_tile dq;
};

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
    float lstat[kQueryTile];
    float dstat[kQueryTile];
};

static_assert(sizeof(shared_storage) < 160 * 1024);

struct main_globals {
    using q_gl = gl<
        fp8e4m3,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<score_q_tile, dim::DEPTH>,
        tma::descriptor<dkdv_rhs_tile, dim::DEPTH>
    >;
    using k_gl = gl<
        fp8e4m3,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<score_k_tile, dim::DEPTH>,
        tma::descriptor<dq_rhs_tile, dim::DEPTH>
    >;
    using v_gl = gl<
        fp8e4m3,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<value_tile, dim::DEPTH>
    >;
    using dout_gl = gl<
        fp8e4m3,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dp_dout_tile, dim::DEPTH>,
        tma::descriptor<dkdv_rhs_tile, dim::DEPTH>
    >;
    using partial_gl = gl<
        float,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dkdv_stage_tile, dim::DEPTH>
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
    partial_gl dk_partial;
    partial_gl dv_partial;
    dq_gl dq_accum;
    const float *lstat;
    const float *dstat;
    float beta;
    float beta_log2e;
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

__device__ __forceinline__ void async_proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <
    int TransA,
    int TransB,
    int Accumulate,
    ducks::tt::all D,
    ducks::st_descriptor::input A,
    ducks::st_descriptor::input B
>
__device__ __forceinline__ void fp8_mma2(
    D &destination,
    const A &a,
    const B &b,
    semaphore &completion
) {
    // TK's generic dense SS helper mishandles the MN-major side of E4M3
    // K32 issues.  K-major descriptors already step in 32-byte units; the
    // MN-major descriptor needs two physical 16-row advances.  Advancing it
    // by one silently overlaps half of each K step and never consumes the
    // tail of Q/dO.
    constexpr int kCtaGroup = 2;
    constexpr int kM = (TransA ? A::cols : A::rows) * kCtaGroup;
    constexpr int kN = (TransB ? B::cols : B::rows) * kCtaGroup;
    constexpr int kK = TransA ? A::rows : A::cols;
    constexpr int kBK = TransB ? B::rows : B::cols;
    using input_type = typename A::T;
    using output_type = typename D::T;
    static_assert(Accumulate == 0 || Accumulate == 1);
    static_assert(std::is_same_v<input_type, fp8e4m3>);
    static_assert(std::is_same_v<input_type, typename B::T>);
    static_assert(kM == D::rows * kCtaGroup);
    static_assert(kN == D::cols);
    static_assert(kK == kBK && kK % 32 == 0);

    constexpr uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kM,
            kN,
            TransA,
            TransB,
            false
        >();
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<A>,
        TransA
    > a_descriptor(a);
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<B>,
        TransB
    > b_descriptor(b);

    async_proxy_fence();
    ::kittens::detail::tcgen05::template st_st<
        input_type,
        Accumulate,
        kCtaGroup
    >(
        destination.addr,
        a_descriptor.chunk_descriptor(0),
        b_descriptor.chunk_descriptor(0),
        instruction
    );
#pragma unroll
    for (int chunk = 1; chunk < kK / 32; ++chunk) {
        constexpr int kAChunkScale = TransA ? 2 : 1;
        constexpr int kBChunkScale = TransB ? 2 : 1;
        ::kittens::detail::tcgen05::template st_st<
            input_type,
            1,
            kCtaGroup
        >(
            destination.addr,
            a_descriptor.chunk_descriptor(kAChunkScale * chunk),
            b_descriptor.chunk_descriptor(kBChunkScale * chunk),
            instruction
        );
    }
    ::kittens::detail::tcgen05::commit<kCtaGroup>(completion);
}

__device__ __forceinline__ void drain_score_or_dp(
    const score_tmem_tile &source,
    score_stage_tile &destination,
    int physical_warp
) {
    const int output_subtile =
        2 * (physical_warp & 3) + (physical_warp >> 2);
    rt_fl<16, 128> values;
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
    group<4>::load_async(values, source);
    tensor_load_wait();
    auto destination_slice = destination.template subtile<32, 64>(
        {physical_warp, 0}
    );
    warp::store(destination_slice, values);
}

__global__ __launch_bounds__(kThreads, 1)
void owner_e4m3_kernel(const __grid_constant__ main_globals globals) {
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
    const int batch_idx = static_cast<int>(blockIdx.z);
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
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k,
            globals.k,
            coord<score_k_tile>{
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
            coord<value_tile>{
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
            coord<dq_rhs_tile>{
                batch_idx,
                owner_idx,
                kv_head,
                cta_rank
            },
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
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.phase.query.q_score,
                globals.q,
                coord<score_q_tile>{
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
                coord<dp_dout_tile>{
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
                coord<dkdv_rhs_tile>{
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
                coord<dkdv_rhs_tile>{
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
                storage.lstat[column] =
                    globals.lstat[stats_base + column];
                storage.dstat[column] =
                    globals.dstat[stats_base + column];
            }
        }
        wait(query_ready, phase);
        __syncthreads();
        everyone::tma::cluster::sync();

        if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
            fp8_mma2<transpose::N, transpose::N, 0>(
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
                float probability_e = 0.0f;
                if (key_position <= query_position) {
                    const float exponent =
                        storage.work.score_dp[{key_row, query_column}] *
                            globals.beta_log2e +
                        storage.lstat[query_column];
                    probability_e = exp2f(exponent);
                }
                storage.p_ds[{key_row, query_column}] =
                    static_cast<fp8e4m3>(probability_e);
            }
            // Every writer must publish its generic shared-memory stores to
            // the async proxy used by the CTA-group-2 tensor operation.  A
            // cluster barrier orders CTAs but does not perform this proxy
            // transition for the partner CTA.
            async_proxy_fence();
        }
        __syncthreads();

        everyone::tma::cluster::sync();

        if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
            fp8_mma2<transpose::N, transpose::N, 0>(
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
                fp8_mma2<transpose::N, transpose::T, 0>(
                    dv_tmem,
                    storage.p_ds,
                    storage.phase.query.dout_dv,
                    dv_done
                );
            } else {
                fp8_mma2<transpose::N, transpose::T, 1>(
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
                const float probability_e = static_cast<float>(
                    storage.p_ds[{key_row, query_column}]
                );
                const float centered_dp_raw =
                    storage.work.score_dp[{key_row, query_column}] +
                    storage.dstat[query_column];
                const float ds_e =
                    probability_e * centered_dp_raw * globals.beta;
                storage.p_ds[{key_row, query_column}] =
                    static_cast<fp8e4m3>(ds_e);
            }
            async_proxy_fence();
        }
        __syncthreads();
        everyone::tma::cluster::sync();

        if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
            if (first_accumulation) {
                fp8_mma2<transpose::N, transpose::T, 0>(
                    dk_tmem,
                    storage.p_ds,
                    storage.phase.query.q_dk,
                    dk_done
                );
            } else {
                fp8_mma2<transpose::N, transpose::T, 1>(
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
            async_proxy_fence();
        }
        __syncthreads();
        everyone::tma::cluster::sync();

        if (physical_warp == 12 && cta_rank == 0 && lane == 0) {
            fp8_mma2<transpose::T, transpose::T, 0>(
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
            async_proxy_fence();
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
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

    if (physical_warp < 4) {
        drain_dkdv(dk_tmem, storage.work.epilogue.dkdv, physical_warp);
    }
    __syncthreads();
    if (physical_warp == 0) {
        async_proxy_fence();
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dk_partial,
            storage.work.epilogue.dkdv,
            coord<dkdv_stage_tile>{
                batch_idx,
                owner_idx * 2 + cta_rank,
                query_head,
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
        async_proxy_fence();
        warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
            globals.dv_partial,
            storage.work.epilogue.dkdv,
            coord<dkdv_stage_tile>{
                batch_idx,
                owner_idx * 2 + cta_rank,
                query_head,
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
void reduce_dkdv_kernel(
    const float *__restrict__ dk_partial,
    const float *__restrict__ dv_partial,
    bf16 *__restrict__ dk,
    bf16 *__restrict__ dv,
    int batch_size
) {
    const int64_t output_elements =
        static_cast<int64_t>(batch_size) * kSequence * kKvHeads * kDepth;
    for (
        int64_t index =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < output_elements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x
    ) {
        const int depth = static_cast<int>(index % kDepth);
        const int64_t head_row = index / kDepth;
        const int kv_head = static_cast<int>(head_row % kKvHeads);
        const int64_t token_row = head_row / kKvHeads;
        const int sequence = static_cast<int>(token_row % kSequence);
        const int batch_idx = static_cast<int>(token_row / kSequence);
        const int first_query_head = kv_head * kHeadRatio;
        float dk_sum = 0.0f;
        float dv_sum = 0.0f;
#pragma unroll
        for (int ratio = 0; ratio < kHeadRatio; ++ratio) {
            const int query_head = first_query_head + ratio;
            const int64_t partial_index =
                (((static_cast<int64_t>(batch_idx) * kSequence + sequence) *
                    kQueryHeads + query_head) * kDepth) +
                depth;
            dk_sum += dk_partial[partial_index];
            dv_sum += dv_partial[partial_index];
        }
        dk[index] = __float2bfloat16_rn(
            dk_sum * kGradientOutputScale
        );
        dv[index] = __float2bfloat16_rn(
            dv_sum * kGradientOutputScale
        );
    }
}

inline void launch_owner_e4m3(
    const main_globals &globals,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(kOwnerClusters * 2, kQueryHeads, batch_size),
        dim3(kThreads, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        owner_e4m3_kernel,
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
    int batch_size,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t dq_elements =
        static_cast<int64_t>(batch_size) * kSequence * kQueryHeads * kDepth;
    const int64_t kv_elements =
        static_cast<int64_t>(batch_size) * kSequence * kKvHeads * kDepth;
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
    reduce_dkdv_kernel<<<
        static_cast<unsigned int>((kv_elements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dk_partial.data_ptr()),
        reinterpret_cast<const float *>(dv_partial.data_ptr()),
        reinterpret_cast<bf16 *>(dk.data_ptr()),
        reinterpret_cast<bf16 *>(dv.data_ptr()),
        batch_size
    );
    CUDACHECK(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_e4m3_owner

#pragma once

#include "v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds.cuh"

// Correctness-first D128 causal-GQA backward built from the validated v416
// production ABI and numerical path.  One CTA owns a K128 tile for one query
// head.  P and dS have distinct shared tiles, while the four FP32 gradient
// accumulators exactly fill SM100 TMEM:
//
//   dP/dQ [0,128), dK [128,256), dV [256,384), score [384,512).
//
// In particular, there is only one score page.  D128 cannot mechanically
// inherit v416's two score stages without aliasing a live gradient page.
namespace tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd {

namespace d64 =
    tkfa4::native_gqa_tk_bwd::v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds;
namespace half =
    tkfa4::native_gqa_tk_bwd::v386_d64_gqa_e4m3_k128q128_halfcols;
namespace mma = tkfa4::native_gqa_tk_bwd::pipelined;

constexpr int kDepth = 128;
constexpr int kKeyTile = 128;
constexpr int kQueryTile = 128;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kHeadRatio = kQueryHeads / kKvHeads;
constexpr int kThreads = 512;
constexpr int kComputeWarps = 8;
constexpr int kReduceWarpBase = 8;
constexpr int kReduceWarps = 4;
constexpr int kTensorIssueWarp = 12;
constexpr int kLoaderWarp = 13;
constexpr int kColumnHalf = 64;
constexpr int kDepthChunk = 32;
constexpr int kDepthChunks = kDepth / kDepthChunk;
constexpr float kOperandScale = 4.0f;
constexpr float kProbabilityScale = 256.0f;
constexpr float kGradientOutputScale = 1.0f / kProbabilityScale;

constexpr int kDpDqTmemOffset = 0;
constexpr int kDkTmemOffset = 128;
constexpr int kDvTmemOffset = 256;
constexpr int kScoreTmemOffset = 384;

static_assert(kHeadRatio == 4);
static_assert(kDepthChunks == 4);
static_assert(kDpDqTmemOffset + kDepth == kDkTmemOffset);
static_assert(kDkTmemOffset + kDepth == kDvTmemOffset);
static_assert(kDvTmemOffset + kDepth == kScoreTmemOffset);
static_assert(kScoreTmemOffset + kQueryTile == 512);

using operand_tile = st_fp8e4m3<kKeyTile, kDepth>;
using attention_tile = st_fp8e4m3<kKeyTile, kQueryTile>;
using gradient_chunk_tile = st_bf<kKeyTile, kDepthChunk>;
using attention_tmem_tile = full_tt_fl<kQueryTile>;
using gradient_tmem_tile = full_tt_fl<kDepth>;
using gradient_chunk_tmem_tile = full_tt_fl<kDepthChunk>;
using attention_tmem_fragment = full_tt_fl<kColumnHalf>;
using stats_tile = sv_fl<kQueryTile>;

struct globals {
    using operand_gl = gl<
        fp8e4m3,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<operand_tile, dim::DEPTH>
    >;
    using gradient_gl = gl<
        bf16,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<gradient_chunk_tile, dim::DEPTH>
    >;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    operand_gl q;
    operand_gl k;
    operand_gl v;
    operand_gl dout;
    gradient_gl dq;
    gradient_gl dk;
    gradient_gl dv;
    stats_gl lstat;
    stats_gl dstat;
    float beta;
    float beta_log2e;
    int sequence;
};

struct shared_storage {
    operand_tile k;
    operand_tile v;
    operand_tile q;
    operand_tile dout;
    attention_tile probability;
    attention_tile ds;
    gradient_chunk_tile gradient;
    stats_tile lstat;
    stats_tile dstat;
};

// 4*16 KiB operands + 2*16 KiB attention payloads + 8 KiB gradient
// publication + 1 KiB statistics.  Semaphores remain outside this object.
static_assert(sizeof(shared_storage) == 105 * 1024);
static_assert(sizeof(shared_storage) < 128 * 1024);

__device__ __forceinline__ int output_subtile_for_warp(
    int physical_warp
) {
    return 2 * (physical_warp & 3) + (physical_warp >> 2);
}

// Score ownership is identical to v416: lane L owns row L%16 and columns
// 32*(L/16)+[0,32).  Reuse its exact native-EX2 and vector shared-store path,
// but publish P to shared because all 512 TMEM columns are live at D128.
__device__ __forceinline__ void make_probability_half(
    const attention_tmem_tile &score_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    bool diagonal,
    float beta_log2e
) {
    d64::owner_aligned_fp32_half probability;
    d64::load_owner_aligned_fp32_half(
        probability,
        score_tmem,
        output_subtile,
        column_half
    );
    tensor_load_wait();

    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    const int lane_row = kittens::laneid() & 15;
    const int lane_column_base = 32 * (kittens::laneid() >> 4);
    const int key_row = output_subtile * 16 + lane_row;
    const int query_column_base =
        column_half * kColumnHalf + lane_column_base;

#pragma unroll
    // Each lane owns 32 of this 64-column half (16 float2 pairs); the
    // companion lane for the same row owns the other 32 columns.
    for (int pair = 0; pair < kColumnHalf / 4; ++pair) {
        const int local_column = 2 * pair;
        const float2 statistic = *reinterpret_cast<const float2 *>(
            &storage.lstat[query_column_base + local_column]
        );
        float2 value = probability.pairs[pair];
        value.x = value.x * beta_log2e + statistic.x;
        value.y = value.y * beta_log2e + statistic.y;
        if (diagonal) {
            if (key_row > query_column_base + local_column) {
                value.x = kNegInf;
            }
            if (key_row > query_column_base + local_column + 1) {
                value.y = kNegInf;
            }
        }
        value = d64::clamp_probability_log2(value);
        probability.pairs[pair] = d64::exp2_native_f32x2(value);
    }

    auto destination =
        storage.probability.template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    d64::store_owner_aligned_shared_half(destination, probability);
}

// P is deliberately reloaded from its coordinate-correct shared E4M3
// publication.  With production lstat, that payload is Pscaled=256*P.  dP
// and dstat are both lifted by 16, so beta=softmax_scale/16 produces an E4M3
// dS payload lifted by 256, matching the direct-BF16 epilogue scale.
__device__ __forceinline__ void make_ds_half(
    const attention_tmem_tile &dp_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    float beta
) {
    half::attention_fp32_fragment probability;
    half::attention_fp32_fragment dp;
    half::attention_e4m3_fragment probability_lowp;
    half::attention_e4m3_fragment ds_lowp;

    auto probability_source =
        storage.probability.template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    half::load_e4m3_half(probability_lowp, probability_source);
    warp::copy(probability, probability_lowp);

    const attention_tmem_fragment dp_half =
        dp_tmem.template subtile<attention_tmem_fragment>(
            0,
            column_half * kColumnHalf
        );
    group<8>::load_async(dp, dp_half);
    tensor_load_wait();
    half::add_shared_row_vector_half(dp, storage.dstat, column_half);
    warp::mul(dp, probability, dp);
    warp::mul(dp, dp, beta);
    mma::convert_f32_to_e4m3(ds_lowp, dp);

    auto destination = storage.ds.template subtile<16, kColumnHalf>(
        {output_subtile, column_half}
    );
    warp::store(destination, ds_lowp);
}

__device__ __forceinline__ void issue_score_or_dp(
    attention_tmem_tile &destination,
    const operand_tile &lhs,
    const operand_tile &rhs,
    semaphore &completion
) {
    warpgroup::mm_ABt(destination, lhs, rhs, completion);
}

template <int Accumulate>
__device__ __forceinline__ void issue_gradient_ab(
    gradient_tmem_tile &destination,
    const attention_tile &lhs,
    const operand_tile &rhs,
    semaphore &completion
) {
    mma::fp8_mm_ab_corrected<Accumulate>(destination, lhs, rhs);
    tensor_commit<1>(completion);
}

__device__ __forceinline__ void issue_gradient_atb(
    gradient_tmem_tile &destination,
    const attention_tile &lhs,
    const operand_tile &rhs,
    semaphore &completion
) {
    mma::fp8_mm_atb_corrected<0>(destination, lhs, rhs);
    tensor_commit<1>(completion);
}

// Each reducer warp owns 32 rows.  Restricting the live register tile to
// 32x32 avoids the D128 full-width drain that previously forced ptxas local
// arrays/spills.  Four warps collectively materialize one 128x32 tile.
__device__ __forceinline__ void drain_gradient_chunk_to_bf16(
    const gradient_tmem_tile &source,
    gradient_chunk_tile &destination,
    int logical_warp,
    int depth_chunk
) {
    rt_fl<32, kDepthChunk> values_fp32;
    rt_bf<32, kDepthChunk> values_bf16;
    const gradient_chunk_tmem_tile source_chunk =
        source.template subtile<gradient_chunk_tmem_tile>(
            0,
            depth_chunk * kDepthChunk
        );
    group<kReduceWarps>::load_async(values_fp32, source_chunk);
    tensor_load_wait();
    warp::mul(
        values_fp32,
        values_fp32,
        1.0f / 256.0f
    );
    warp::copy(values_bf16, values_fp32);
    auto destination_slice =
        destination.template subtile<32, kDepthChunk>(
            {logical_warp, 0}
        );
    warp::store(destination_slice, values_bf16);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// This routine contains CTA barriers and must be called by every thread.
// The TMA descriptor is D32, so the final coordinate selects one of four
// physical depth chunks in the contiguous BSHD destination.
__device__ __forceinline__ void drain_and_publish_gradient(
    const gradient_tmem_tile &source,
    const globals::gradient_gl &destination,
    gradient_chunk_tile &stage,
    int batch,
    int sequence_tile,
    int head,
    int physical_warp,
    int lane
) {
    const int logical_warp = physical_warp - kReduceWarpBase;
#pragma unroll
    for (int depth_chunk = 0; depth_chunk < kDepthChunks; ++depth_chunk) {
        if (physical_warp >= kReduceWarpBase &&
            physical_warp < kReduceWarpBase + kReduceWarps) {
            drain_gradient_chunk_to_bf16(
                source,
                stage,
                logical_warp,
                depth_chunk
            );
        }
        __syncthreads();
        if (physical_warp == kReduceWarpBase && lane == 0) {
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                destination,
                stage,
                coord<gradient_chunk_tile>{
                    batch,
                    sequence_tile,
                    head,
                    depth_chunk,
                }
            );
            warp::tma::store_async_wait<0>();
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kThreads, 1)
void main_kernel(const __grid_constant__ globals g) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready;
    __shared__ alignas(16) semaphore stats_ready;
    __shared__ alignas(16) semaphore score_ready;
    __shared__ alignas(16) semaphore dp_ready;
    __shared__ alignas(16) semaphore dv_ready;
    __shared__ alignas(16) semaphore dq_ready;
    __shared__ alignas(16) semaphore dk_ready;

    const int physical_warp = warpid();
    const int lane = laneid();
    const int key_tile = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int kv_head = query_head / kHeadRatio;
    const int iteration_count = g.sequence / kQueryTile - key_tile;

    // Keep the register budget below one complete SM100 register file.  The
    // compute warps retain one 64-column FP32 score/P fragment; reducers only
    // materialize a 32x32 gradient chunk.
    if (physical_warp < kComputeWarps) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;" ::: "memory");
    } else if (physical_warp < kTensorIssueWarp) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;" ::: "memory");
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;" ::: "memory");
    }

    if (threadIdx.x == 0) {
        init_semaphore(persistent_ready, 0, 1);
        init_semaphore(query_ready, 0, 1);
        init_semaphore(stats_ready, 0, 1);
        init_semaphore(score_ready, 0, 1);
        init_semaphore(dp_ready, 0, 1);
        init_semaphore(dv_ready, 0, 1);
        init_semaphore(dq_ready, 0, 1);
        init_semaphore(dk_ready, 0, 1);
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
    }
    wait(persistent_ready, 0);
    __syncthreads();

    for (int iteration = 0; iteration < iteration_count; ++iteration) {
        const int phase = iteration & 1;
        const int query_tile = key_tile + iteration;

        if (physical_warp == kLoaderWarp && lane == 0) {
            const coord<operand_tile> operand_coordinate{
                batch,
                query_tile,
                query_head,
                0,
            };
            tma::expect_bytes(
                query_ready,
                sizeof(storage.q) + sizeof(storage.dout)
            );
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.q,
                g.q,
                operand_coordinate,
                query_ready
            );
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.dout,
                g.dout,
                operand_coordinate,
                query_ready
            );

            const coord<stats_tile> stats_coordinate{
                batch,
                query_head,
                0,
                query_tile,
            };
            tma::expect_bytes(
                stats_ready,
                sizeof(storage.lstat) + sizeof(storage.dstat)
            );
            tma::load_async(
                storage.lstat,
                g.lstat,
                stats_coordinate,
                stats_ready
            );
            tma::load_async(
                storage.dstat,
                g.dstat,
                stats_coordinate,
                stats_ready
            );
        }
        wait(query_ready, phase);
        wait(stats_ready, phase);
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            issue_score_or_dp(
                score_tmem,
                storage.k,
                storage.q,
                score_ready
            );
        }
        wait(score_ready, phase);
        tensor_after_thread_sync();

        if (physical_warp < kComputeWarps) {
            const int output_subtile =
                output_subtile_for_warp(physical_warp);
            make_probability_half(
                score_tmem,
                storage,
                output_subtile,
                0,
                iteration == 0,
                g.beta_log2e
            );
            make_probability_half(
                score_tmem,
                storage,
                output_subtile,
                1,
                iteration == 0,
                g.beta_log2e
            );
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            if (iteration == 0) {
                issue_gradient_ab<0>(
                    dv_tmem,
                    storage.probability,
                    storage.dout,
                    dv_ready
                );
            } else {
                issue_gradient_ab<1>(
                    dv_tmem,
                    storage.probability,
                    storage.dout,
                    dv_ready
                );
            }
            issue_score_or_dp(
                dp_tmem,
                storage.v,
                storage.dout,
                dp_ready
            );
        }
        wait(dv_ready, phase);
        wait(dp_ready, phase);
        tensor_after_thread_sync();

        if (physical_warp < kComputeWarps) {
            const int output_subtile =
                output_subtile_for_warp(physical_warp);
            make_ds_half(
                dp_tmem,
                storage,
                output_subtile,
                0,
                g.beta
            );
            make_ds_half(
                dp_tmem,
                storage,
                output_subtile,
                1,
                g.beta
            );
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            issue_gradient_atb(
                dq_tmem,
                storage.ds,
                storage.k,
                dq_ready
            );
            if (iteration == 0) {
                issue_gradient_ab<0>(
                    dk_tmem,
                    storage.ds,
                    storage.q,
                    dk_ready
                );
            } else {
                issue_gradient_ab<1>(
                    dk_tmem,
                    storage.ds,
                    storage.q,
                    dk_ready
                );
            }
        }
        wait(dq_ready, phase);
        wait(dk_ready, phase);
        tensor_after_thread_sync();

        drain_and_publish_gradient(
            dq_tmem,
            g.dq,
            storage.gradient,
            batch,
            query_tile,
            query_head,
            physical_warp,
            lane
        );
    }

    // dK and dV aggregate all causal query tiles for this key-tile/query-head
    // CTA.  Additive stores combine the four query heads sharing each KV head.
    drain_and_publish_gradient(
        dk_tmem,
        g.dk,
        storage.gradient,
        batch,
        key_tile,
        kv_head,
        physical_warp,
        lane
    );
    drain_and_publish_gradient(
        dv_tmem,
        g.dv,
        storage.gradient,
        batch,
        key_tile,
        kv_head,
        physical_warp,
        lane
    );
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
    main_kernel<<<grid, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v420_d128_gqa_e4m3_shared_p_production_bshd

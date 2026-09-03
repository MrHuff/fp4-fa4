#pragma once

#include "native_gqa_tk_bwd_pipelined.cuh"

// Bounded K128 x Q128 experiment for the D64/GQA E4M3 backward.
//
// Unlike the K64-owner donor, one CTA owns a complete K128 tile.  K/V are
// loaded once, Q/dO are staged in Q128 tiles, and row statistics are produced
// by a dedicated warp with ordinary global loads.  A tensor-issue warp emits
// full score/dP and gradient MMAs while eight compute warps perform the
// softmax/dS work in registers.  Gradients are reduced directly into compact
// BF16 destinations; callers must clear all three destinations before launch.
namespace tkfa4::native_gqa_tk_bwd::v385_d64_gqa_e4m3_k128q128 {

namespace donor = tkfa4::native_gqa_tk_bwd::pipelined;

constexpr int kDepth = 64;
constexpr int kKeyTile = 128;
constexpr int kQueryTile = 128;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kHeadRatio = kQueryHeads / kKvHeads;
constexpr int kThreads = 512;
constexpr int kComputeWarps = 8;
constexpr int kTensorIssueWarp = 12;
constexpr int kLoaderWarp = 13;
constexpr int kStatsWarp = 14;
constexpr float kOperandScale = 4.0f;
constexpr float kProbabilityScale = 256.0f;
constexpr float kGradientOutputScale = 1.0f / kProbabilityScale;

static_assert(kHeadRatio == 4);

using operand_tile = st_fp8e4m3<kKeyTile, kDepth>;
using attention_tile = st_fp8e4m3<kKeyTile, kQueryTile>;
using gradient_stage_tile = st_bf<kKeyTile, kDepth>;
using attention_tmem_tile = full_tt_fl<kQueryTile>;
using gradient_tmem_tile = full_tt_fl<kDepth>;
using attention_fp32_register = rt_fl<16, kQueryTile>;
using attention_e4m3_register = rt_fp8e4m3<16, kQueryTile>;
using attention_fp32_quarter_register = rt_fl<16, 32>;
using attention_e4m3_quarter_register = rt_fp8e4m3<16, 32>;
using attention_quarter_stats_register =
    typename attention_fp32_quarter_register::row_vec;

struct globals {
    using operand_gl = gl<fp8e4m3, -1, -1, -1, -1, operand_tile>;
    using gradient_gl =
        gl<bf16, -1, -1, -1, -1, gradient_stage_tile>;

    operand_gl q;
    operand_gl k;
    operand_gl v;
    operand_gl dout;
    gradient_gl dq;
    gradient_gl dk;
    gradient_gl dv;
    const float *l_aux;
    const float *delta;
    float beta;
    float beta_log2e;
    float l_aux_scale;
    int sequence;
};

struct shared_storage {
    operand_tile k;
    operand_tile v;
    operand_tile q;
    operand_tile dout;
    attention_tile probability_ds;
    gradient_stage_tile gradient;
    sv_fl<kQueryTile> lstat;
    sv_fl<kQueryTile> dstat;
};

static_assert(sizeof(shared_storage) < 80 * 1024);

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
    warp::mul(
        score_quarter,
        score_quarter,
        256.0f
    );
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
    warp::mul(
        probability_fp32,
        probability_fp32,
        1.0f / 256.0f
    );
    warp::add_col(dp_quarter, dp_quarter, dstat_quarter);
    warp::mul(dp_quarter, probability_fp32, dp_quarter);
    warp::mul(dp_quarter, dp_quarter, beta * 256.0f);
    warp::copy(probability_ds_quarter, dp_quarter);
}

__device__ __forceinline__ void apply_diagonal_causal_mask(
    attention_fp32_register &score,
    int output_subtile
) {
    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    const int key_row_base = output_subtile * 16;
    warp::apply(score, score, [=](int row, int column, float value) {
        return key_row_base + row > column ? kNegInf : value;
    });
}

__device__ __forceinline__ void drain_gradient_to_bf16(
    const gradient_tmem_tile &source,
    gradient_stage_tile &destination,
    int physical_warp
) {
    rt_fl<32, kDepth> values_fp32;
    rt_bf<32, kDepth> values_bf16;
    group<4>::load_async(values_fp32, source);
    tensor_load_wait();
    warp::mul(
        values_fp32,
        values_fp32,
        1.0f / 256.0f
    );
    warp::copy(values_bf16, values_fp32);
    auto destination_slice = destination.template subtile<32, kDepth>(
        {physical_warp, 0}
    );
    warp::store(destination_slice, values_bf16);
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
    donor::fp8_mm_ab_corrected<Accumulate>(destination, lhs, rhs);
    tensor_commit<1>(completion);
}

__device__ __forceinline__ void issue_gradient_atb(
    gradient_tmem_tile &destination,
    const attention_tile &lhs,
    const operand_tile &rhs,
    semaphore &completion
) {
    donor::fp8_mm_atb_corrected<0>(destination, lhs, rhs);
    tensor_commit<1>(completion);
}

__global__ __launch_bounds__(kThreads, 1)
void main_kernel(const __grid_constant__ globals g) {
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
    const int key_tile = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int kv_head = query_head / kHeadRatio;

    // Eight compute warps retain P/dS while the issue/load/control half of the
    // CTA surrenders registers.  The two values exactly consume one SM100
    // register file at full occupancy for a 512-thread CTA.
    if (physical_warp < kComputeWarps) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;" ::: "memory");
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;" ::: "memory");
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

    tensor_allocator<1, 1> tmem_allocator{};
    attention_tmem_tile score_dp_tmem =
        tmem_allocator.template allocate<attention_tmem_tile>(0);
    gradient_tmem_tile dq_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(0);
    gradient_tmem_tile dk_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(128);
    gradient_tmem_tile dv_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(192);

    if (physical_warp == kLoaderWarp && lane == 0) {
        tma::expect_bytes(
            persistent_ready,
            sizeof(storage.k) + sizeof(storage.v)
        );
        tma::load_async(
            storage.k,
            g.k,
            coord<operand_tile>{batch, kv_head, key_tile, 0},
            persistent_ready
        );
        tma::load_async(
            storage.v,
            g.v,
            coord<operand_tile>{batch, kv_head, key_tile, 0},
            persistent_ready
        );
    }
    wait(persistent_ready, 0);
    __syncthreads();

    int iteration = 0;
    for (
        int query_tile = key_tile;
        query_tile < g.sequence / kQueryTile;
        ++query_tile, ++iteration
    ) {
        const int phase = iteration & 1;

        if (physical_warp == kLoaderWarp && lane == 0) {
            tma::expect_bytes(
                query_ready,
                sizeof(storage.q) + sizeof(storage.dout)
            );
            const coord<operand_tile> coordinate{
                batch,
                query_head,
                query_tile,
                0,
            };
            tma::load_async(storage.q, g.q, coordinate, query_ready);
            tma::load_async(storage.dout, g.dout, coordinate, query_ready);
        }
        if (physical_warp == kStatsWarp) {
            const int stats_base =
                (batch * kQueryHeads + query_head) * g.sequence +
                query_tile * kQueryTile;
            for (int column = lane; column < kQueryTile; column += 32) {
                storage.lstat[column] =
                    g.l_aux[stats_base + column] * g.l_aux_scale;
                storage.dstat[column] =
                    -16.0f * g.delta[stats_base + column];
            }
        }
        wait(query_ready, phase);
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            issue_score_or_dp(
                score_dp_tmem,
                storage.k,
                storage.q,
                score_done
            );
        }
        wait(score_done, phase);
        tensor_after_thread_sync();

        attention_e4m3_register probability_ds;
        if (physical_warp < kComputeWarps) {
            attention_fp32_register score;
            group<8>::load_async(score, score_dp_tmem);
            tensor_load_wait();
            const int output_subtile =
                2 * (physical_warp & 3) + (physical_warp >> 2);
            if (query_tile == key_tile) {
                apply_diagonal_causal_mask(score, output_subtile);
            }
            make_probability_quarter<0>(
                probability_ds,
                score,
                storage.lstat,
                g.beta_log2e
            );
            make_probability_quarter<1>(
                probability_ds,
                score,
                storage.lstat,
                g.beta_log2e
            );
            make_probability_quarter<2>(
                probability_ds,
                score,
                storage.lstat,
                g.beta_log2e
            );
            make_probability_quarter<3>(
                probability_ds,
                score,
                storage.lstat,
                g.beta_log2e
            );
            auto probability_destination =
                storage.probability_ds.template subtile<16, kQueryTile>(
                    {output_subtile, 0}
                );
            warp::store(probability_destination, probability_ds);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            if (iteration == 0) {
                issue_gradient_ab<0>(
                    dv_tmem,
                    storage.probability_ds,
                    storage.dout,
                    dv_done
                );
            } else {
                issue_gradient_ab<1>(
                    dv_tmem,
                    storage.probability_ds,
                    storage.dout,
                    dv_done
                );
            }
            issue_score_or_dp(
                score_dp_tmem,
                storage.v,
                storage.dout,
                dp_done
            );
        }
        wait(dv_done, phase);
        wait(dp_done, phase);
        tensor_after_thread_sync();

        if (physical_warp < kComputeWarps) {
            attention_fp32_register dp;
            group<8>::load_async(dp, score_dp_tmem);
            tensor_load_wait();
            make_ds_quarter<0>(
                probability_ds,
                dp,
                storage.dstat,
                g.beta
            );
            make_ds_quarter<1>(
                probability_ds,
                dp,
                storage.dstat,
                g.beta
            );
            make_ds_quarter<2>(
                probability_ds,
                dp,
                storage.dstat,
                g.beta
            );
            make_ds_quarter<3>(
                probability_ds,
                dp,
                storage.dstat,
                g.beta
            );
            const int output_subtile =
                2 * (physical_warp & 3) + (physical_warp >> 2);
            auto ds_destination =
                storage.probability_ds.template subtile<16, kQueryTile>(
                    {output_subtile, 0}
                );
            warp::store(ds_destination, probability_ds);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            if (iteration == 0) {
                issue_gradient_ab<0>(
                    dk_tmem,
                    storage.probability_ds,
                    storage.q,
                    dk_done
                );
            } else {
                issue_gradient_ab<1>(
                    dk_tmem,
                    storage.probability_ds,
                    storage.q,
                    dk_done
                );
            }
            issue_gradient_atb(
                dq_tmem,
                storage.probability_ds,
                storage.k,
                dq_done
            );
        }
        wait(dk_done, phase);
        wait(dq_done, phase);
        tensor_after_thread_sync();

        if (physical_warp < 4) {
            drain_gradient_to_bf16(
                dq_tmem,
                storage.gradient,
                physical_warp
            );
        }
        __syncthreads();
        if (physical_warp == 0) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            warp::tma::store_add_async(
                g.dq,
                storage.gradient,
                coord<gradient_stage_tile>{
                    batch,
                    query_head,
                    query_tile,
                    0,
                }
            );
            warp::tma::store_async_wait();
        }
        __syncthreads();
    }

    if (physical_warp < 4) {
        drain_gradient_to_bf16(dk_tmem, storage.gradient, physical_warp);
    }
    __syncthreads();
    if (physical_warp == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        warp::tma::store_add_async(
            g.dk,
            storage.gradient,
            coord<gradient_stage_tile>{batch, kv_head, key_tile, 0}
        );
        warp::tma::store_async_wait();
    }
    __syncthreads();

    if (physical_warp < 4) {
        drain_gradient_to_bf16(dv_tmem, storage.gradient, physical_warp);
    }
    __syncthreads();
    if (physical_warp == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        warp::tma::store_add_async(
            g.dv,
            storage.gradient,
            coord<gradient_stage_tile>{batch, kv_head, key_tile, 0}
        );
        warp::tma::store_async_wait();
    }
}

inline void launch(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
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
        reinterpret_cast<const float *>(l_aux.data_ptr()),
        reinterpret_cast<const float *>(delta.data_ptr()),
        beta,
        beta * kLog2E,
        softmax_scale * kLog2E,
        static_cast<int>(q.size(2)),
    };
    const dim3 grid(
        static_cast<unsigned int>(q.size(2) / kKeyTile),
        static_cast<unsigned int>(q.size(1)),
        static_cast<unsigned int>(q.size(0))
    );
    main_kernel<<<grid, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v385_d64_gqa_e4m3_k128q128

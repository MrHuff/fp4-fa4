#pragma once

#include "v385_d64_gqa_e4m3_k128q128.cuh"

// K128 x Q128 ownership with the attention post-processing serialized through
// one reusable 16 x 64 register fragment.  This preserves v385's bulk-TMA and
// compact-BF16 topology while avoiding its 16 x 128 register lifetime and
// template-expanded quarter paths.
namespace tkfa4::native_gqa_tk_bwd::v386_d64_gqa_e4m3_k128q128_halfcols {

namespace base =
    tkfa4::native_gqa_tk_bwd::v385_d64_gqa_e4m3_k128q128;
namespace donor = tkfa4::native_gqa_tk_bwd::pipelined;

using base::attention_e4m3_register;
using base::attention_tile;
using base::attention_tmem_tile;
using base::globals;
using base::gradient_stage_tile;
using base::gradient_tmem_tile;
using base::kComputeWarps;
using base::kDepth;
using base::kHeadRatio;
using base::kKeyTile;
using base::kKvHeads;
using base::kLoaderWarp;
using base::kOperandScale;
using base::kQueryHeads;
using base::kQueryTile;
using base::kStatsWarp;
using base::kTensorIssueWarp;
using base::kThreads;
using base::operand_tile;
using base::shared_storage;

constexpr int kColumnHalf = 64;
using attention_fp32_fragment = rt_fl<16, kColumnHalf>;
using attention_e4m3_fragment = rt_fp8e4m3<16, kColumnHalf>;
using attention_tmem_fragment = full_tt_fl<kColumnHalf>;

__device__ __forceinline__ int output_subtile_for_warp(int physical_warp) {
    return 2 * (physical_warp & 3) + (physical_warp >> 2);
}

__device__ __forceinline__ void apply_diagonal_causal_mask(
    attention_fp32_fragment &score,
    int output_subtile,
    int column_half
) {
    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    const int key_row_base = output_subtile * 16;
    const int query_column_base = column_half * kColumnHalf;
    warp::apply(score, score, [=](int row, int column, float value) {
        return key_row_base + row > query_column_base + column
            ? kNegInf
            : value;
    });
}

// The generic shared-vector loader materializes its temporary packed values as
// a local array when the 64-column half is selected at runtime.  Add the row
// vector directly in the register-tile layout instead, keeping the two halves
// on the same instruction path without a compiler-generated stack frame.
__device__ __forceinline__ void add_shared_row_vector_half(
    attention_fp32_fragment &tile,
    const sv_fl<kQueryTile> &values,
    int column_half
) {
    const int half_base = column_half * kColumnHalf;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const int base_column =
            half_base + 16 * index + 2 * (kittens::laneid() & 3);
        const float2 values_lo = *reinterpret_cast<const float2 *>(
            &values[base_column]
        );
        const float2 values_hi = *reinterpret_cast<const float2 *>(
            &values[base_column + 8]
        );
        tile.tiles[0][index].data[0] =
            base_ops::sum::template op<float2>(
                tile.tiles[0][index].data[0],
                values_lo
            );
        tile.tiles[0][index].data[1] =
            base_ops::sum::template op<float2>(
                tile.tiles[0][index].data[1],
                values_lo
            );
        tile.tiles[0][index].data[2] =
            base_ops::sum::template op<float2>(
                tile.tiles[0][index].data[2],
                values_hi
            );
        tile.tiles[0][index].data[3] =
            base_ops::sum::template op<float2>(
                tile.tiles[0][index].data[3],
                values_hi
            );
    }
}

// Likewise, spell out the two LDSM operations for a 16 x 64 E4M3 fragment.
// ThunderKittens' generic loader uses a four-element temporary array here;
// ptxas retained that array in local memory for the runtime-half loop.
template <typename SharedTile>
__device__ __forceinline__ void load_e4m3_half(
    attention_e4m3_fragment &destination,
    const SharedTile &source
) {
    const int lane = kittens::laneid();
    const int row = lane & 15;
    const int column_group = (lane >> 4) * 16;
    const uint32_t shared_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&source.data[0])
    );
#pragma unroll
    for (int tile_column = 0; tile_column < 2; ++tile_column) {
        fp8e4m3_4 value0;
        fp8e4m3_4 value1;
        fp8e4m3_4 value2;
        fp8e4m3_4 value3;
        move<fp8e4m3_4>::ldsm4(
            value0,
            value1,
            value2,
            value3,
            source.idx(
                shared_address,
                {row, tile_column * 32 + column_group}
            )
        );
        destination.tiles[0][tile_column].data[0] = value0;
        destination.tiles[0][tile_column].data[1] = value1;
        destination.tiles[0][tile_column].data[2] = value2;
        destination.tiles[0][tile_column].data[3] = value3;
    }
}

__device__ __forceinline__ void make_probability_half(
    const attention_tmem_tile &score_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    bool diagonal,
    float beta_log2e
) {
    attention_fp32_fragment score;
    attention_e4m3_fragment probability;
    const attention_tmem_fragment score_half =
        score_tmem.template subtile<attention_tmem_fragment>(
            0,
            column_half * kColumnHalf
        );
    group<8>::load_async(score, score_half);
    tensor_load_wait();
    warp::mul(score, score, beta_log2e);
    add_shared_row_vector_half(score, storage.lstat, column_half);
    if (diagonal) {
        apply_diagonal_causal_mask(score, output_subtile, column_half);
    }
    warp::exp2(score, score);
    warp::mul(score, score, 256.0f);
    donor::convert_f32_to_e4m3(probability, score);
    auto destination =
        storage.probability_ds.template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    warp::store(destination, probability);
}

__device__ __forceinline__ void make_ds_half(
    const attention_tmem_tile &dp_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    float beta
) {
    attention_fp32_fragment probability;
    attention_fp32_fragment dp;
    attention_e4m3_fragment probability_lowp;
    attention_e4m3_fragment ds_lowp;

    auto source = storage.probability_ds.template subtile<16, kColumnHalf>(
        {output_subtile, column_half}
    );
    load_e4m3_half(probability_lowp, source);
    warp::copy(probability, probability_lowp);
    warp::mul(probability, probability, 1.0f / 256.0f);

    const attention_tmem_fragment dp_half =
        dp_tmem.template subtile<attention_tmem_fragment>(
            0,
            column_half * kColumnHalf
        );
    group<8>::load_async(dp, dp_half);
    tensor_load_wait();
    add_shared_row_vector_half(dp, storage.dstat, column_half);
    warp::mul(dp, probability, dp);
    warp::mul(dp, dp, beta * 256.0f);
    donor::convert_f32_to_e4m3(ds_lowp, dp);
    auto destination =
        storage.probability_ds.template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    warp::store(destination, ds_lowp);
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
            base::issue_score_or_dp(
                score_dp_tmem,
                storage.k,
                storage.q,
                score_done
            );
        }
        wait(score_done, phase);
        tensor_after_thread_sync();

        if (physical_warp < kComputeWarps) {
            const int output_subtile =
                output_subtile_for_warp(physical_warp);
#pragma unroll 1
            for (int column_half = 0; column_half < 2; ++column_half) {
                make_probability_half(
                    score_dp_tmem,
                    storage,
                    output_subtile,
                    column_half,
                    query_tile == key_tile,
                    g.beta_log2e
                );
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            if (iteration == 0) {
                base::issue_gradient_ab<0>(
                    dv_tmem,
                    storage.probability_ds,
                    storage.dout,
                    dv_done
                );
            } else {
                base::issue_gradient_ab<1>(
                    dv_tmem,
                    storage.probability_ds,
                    storage.dout,
                    dv_done
                );
            }
            base::issue_score_or_dp(
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
            const int output_subtile =
                output_subtile_for_warp(physical_warp);
#pragma unroll 1
            for (int column_half = 0; column_half < 2; ++column_half) {
                make_ds_half(
                    score_dp_tmem,
                    storage,
                    output_subtile,
                    column_half,
                    g.beta
                );
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            if (iteration == 0) {
                base::issue_gradient_ab<0>(
                    dk_tmem,
                    storage.probability_ds,
                    storage.q,
                    dk_done
                );
            } else {
                base::issue_gradient_ab<1>(
                    dk_tmem,
                    storage.probability_ds,
                    storage.q,
                    dk_done
                );
            }
            base::issue_gradient_atb(
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
            base::drain_gradient_to_bf16(
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
        base::drain_gradient_to_bf16(
            dk_tmem,
            storage.gradient,
            physical_warp
        );
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
        base::drain_gradient_to_bf16(
            dv_tmem,
            storage.gradient,
            physical_warp
        );
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
    v386_d64_gqa_e4m3_k128q128_halfcols::main_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v386_d64_gqa_e4m3_k128q128_halfcols

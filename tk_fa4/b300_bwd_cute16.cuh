#pragma once

#include "b300_bwd_cute16_kernel.cuh"
#include "b300_common.cuh"

namespace tkfa4::bwd_cute16 {

template <int Dvo>
struct preprocess_config {
    static_assert(Dvo == kB300VDim, "Exact B300 CuTe16 preprocess only supports Dvo=128");
    static constexpr int DvoDim = Dvo;
    static constexpr int ClusterSize = 2;
};

template <typename C>
struct preprocess_globals {
    using stats_tile = col_vec<st_fl<kRefTileM, C::DvoDim>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using dq_out_gl = gl<float, -1, -1, -1, -1, dq_chunk_tile>;

    const bf16 *o_ptr;
    const bf16 *dout_ptr;
    const float *lse_ptr;
    stats_gl dpsum;
    stats_gl lse_log2;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    int seq_len;
    int heads;
};

template <typename C>
__global__ __launch_bounds__(2 * kWarpThreads, 8)
void preprocess_kernel(const __grid_constant__ preprocess_globals<C> g) {
    constexpr int RowThreads = 4;
    constexpr int RowsPerWarp = kWarpThreads / RowThreads;
    const int q_tile_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = laneid();
    const int row_in_warp = lane / RowThreads;
    const int lane_in_row = lane % RowThreads;
    const int row = warp * RowsPerWarp + row_in_warp;
    if (row < kRefTileM) {
        const int seq_idx = q_tile_idx * kRefTileM + row;
        float dpsum = 0.0f;
        const size_t base = (((size_t)batch_idx * g.seq_len + seq_idx) * g.heads + head_idx) * C::DvoDim;
        #pragma unroll
        for (int d = lane_in_row; d < C::DvoDim; d += RowThreads) {
            const float o = __bfloat162float(g.o_ptr[base + d]);
            const float dout = __bfloat162float(g.dout_ptr[base + d]);
            dpsum += o * dout;
        }
        #pragma unroll
        for (int offset = RowThreads / 2; offset > 0; offset >>= 1) {
            dpsum += __shfl_down_sync(0xffffffff, dpsum, offset, RowThreads);
        }
        if (lane_in_row == 0) {
            const size_t lse_offset = ((size_t)batch_idx * g.seq_len + seq_idx) * g.heads + head_idx;
            g.dpsum[{batch_idx, head_idx, 0, seq_idx}] = dpsum;
            g.lse_log2[{batch_idx, head_idx, 0, seq_idx}] = g.lse_ptr[lse_offset] * kLog2E;
        }

        if (warp == 0) {
            rt_fl<kRefTileM, 64> zero_chunk;
            warp::zero(zero_chunk);
            warp::store(g.dq0, zero_chunk, {batch_idx, q_tile_idx, head_idx, 0});
            warp::store(g.dq1, zero_chunk, {batch_idx, q_tile_idx, head_idx, 1});
            warp::store(g.dq2, zero_chunk, {batch_idx, q_tile_idx, head_idx, 2});
        }
    }

}

template <typename C>
inline void launch_preprocess(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &lse,
    at::Tensor &dpsum,
    at::Tensor &lse_log2,
    at::Tensor &dq
) {
    using G = preprocess_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, out.size(0), out.size(2), 1, out.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, out.size(0), out.size(2), 1, out.size(1)),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / kRefTileM, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_kernel<C><<<grid, 2 * kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    [[maybe_unused]] bool deterministic,
    bool apply_causal_patches = true,
    int causal_q_start_offset_blocks = 1,
    bool full_causal_patch_coverage = false,
    bool use_exact_bulk_math = false
) {
    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor dqacc_dummy = at::empty({1}, lse.options());
    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2, dq);
    bwd_cute16_kernel::launch_backward<C>(
        q,
        k,
        v,
        dout,
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk,
        dv,
        dqacc_dummy,
        dqacc_dummy,
        dqacc_dummy,
        nullptr,
        causal,
        scale,
        deterministic,
        apply_causal_patches,
        causal_q_start_offset_blocks,
        full_causal_patch_coverage,
        use_exact_bulk_math
    );
}

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
using config = bwd_cute16_kernel::config<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;

}  // namespace tkfa4::bwd_cute16

#pragma once

#include "b300_common.cuh"

namespace tkfa4::bwd_fa4 {

template <int Dvo>
struct preprocess_config {
    static_assert(Dvo == kB300VDim, "Exact B300 backward preprocess only supports Dvo=128");
    static constexpr int DvoDim = Dvo;
};

template <typename C>
struct preprocess_globals {
    using o_gl = gl<bf16, -1, -1, -1, C::DvoDim>;
    using do_gl = gl<bf16, -1, -1, -1, C::DvoDim>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::DvoDim>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    o_gl o;
    do_gl dout;
    stats_gl lse;
    stats_gl dpsum;
    stats_gl lse_log2;
};

template <typename C>
__global__ __launch_bounds__(kWarpThreads, 8)
void preprocess_kernel(const __grid_constant__ preprocess_globals<C> g) {
    const int q_tile_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    rt_bf<kRefTileM, C::DvoDim> o_reg, do_reg;
    rt_fl<kRefTileM, C::DvoDim> o_fl, do_fl, prod;
    using vec_t = typename rt_fl<kRefTileM, C::DvoDim>::col_vec;
    vec_t dpsum_vec, lse_vec, lse_log2_vec;

    warp::load(o_reg, g.o, {batch_idx, head_idx, q_tile_idx, 0});
    warp::load(do_reg, g.dout, {batch_idx, head_idx, q_tile_idx, 0});
    warp::copy(o_fl, o_reg);
    warp::copy(do_fl, do_reg);
    warp::mul(prod, o_fl, do_fl);
    warp::row_sum(dpsum_vec, prod);
    warp::load(lse_vec, g.lse, {batch_idx, head_idx, 0, q_tile_idx});
    warp::mul(lse_log2_vec, lse_vec, 1.4426950408889634f);
    warp::store(g.dpsum, dpsum_vec, {batch_idx, head_idx, 0, q_tile_idx});
    warp::store(g.lse_log2, lse_log2_vec, {batch_idx, head_idx, 0, q_tile_idx});
}

template <typename C>
inline void launch_preprocess(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &lse,
    at::Tensor &dpsum,
    at::Tensor &lse_log2,
    at::Tensor &dq_accum,
    at::Tensor &dq_semaphore
) {
    dq_accum.zero_();
    if (dq_semaphore.defined()) {
        dq_semaphore.zero_();
    }

    using G = preprocess_globals<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::o_gl>(out),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse, out.size(0), out.size(1), 1, out.size(2)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, out.size(0), out.size(1), 1, out.size(2)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, out.size(0), out.size(1), 1, out.size(2)),
    };
    dim3 grid(out.size(2) / kRefTileM, out.size(1), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_kernel<C><<<grid, kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd_fa4

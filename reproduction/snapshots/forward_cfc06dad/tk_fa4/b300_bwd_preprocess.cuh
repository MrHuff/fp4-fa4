#pragma once

#include "b300_common.cuh"

namespace tkfa4::bwd {

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
    using d_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    o_gl o;
    do_gl dout;
    d_gl delta;
};

template <typename C>
__global__ __launch_bounds__(kWarpThreads, 8)
void preprocess_kernel(const __grid_constant__ preprocess_globals<C> g) {
    const int q_tile_idx = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    rt_bf<kRefTileM, C::DvoDim> o_reg, do_reg;
    rt_fl<kRefTileM, C::DvoDim> o_fl, do_fl, prod;
    using vec_t = typename rt_fl<kRefTileM, C::DvoDim>::col_vec;
    vec_t delta_vec;

    warp::load(o_reg, g.o, {batch_idx, q_head_idx, q_tile_idx, 0});
    warp::load(do_reg, g.dout, {batch_idx, q_head_idx, q_tile_idx, 0});
    warp::copy(o_fl, o_reg);
    warp::copy(do_fl, do_reg);
    warp::mul(prod, o_fl, do_fl);
    warp::row_sum(delta_vec, prod);
    warp::store(g.delta, delta_vec, {batch_idx, q_head_idx, 0, q_tile_idx});
}

template <typename C>
inline void launch_preprocess(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &delta
) {
    using G = preprocess_globals<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::o_gl>(out),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::d_gl>(delta, out.size(0), out.size(1), 1, out.size(2)),
    };
    dim3 grid(out.size(2) / kRefTileM, out.size(1), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_kernel<C><<<grid, kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd


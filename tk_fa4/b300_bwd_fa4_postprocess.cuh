#pragma once

#include "b300_common.cuh"

namespace tkfa4::bwd_fa4 {

template <typename C>
struct postprocess_globals {
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
    using dq_gl = gl<float, -1, -1, -1, C::Dqk>;

    dqacc_gl dq_accum;
    dq_gl dq;
    int *dq_semaphore;
    int heads;
    int q_tiles;
    int cluster_groups;
    int deterministic;
};

namespace detail {

template <typename C>
__device__ inline int dq_semaphore_index(
    const postprocess_globals<C> &g,
    int batch_idx,
    int head_idx,
    int q_tile_idx,
    int cluster_rank
) {
    return (((batch_idx * g.heads + head_idx) * g.q_tiles + q_tile_idx) * C::ClusterSize) + cluster_rank;
}

template <typename C>
__global__ __launch_bounds__(kWarpThreads, 8)
void postprocess_kernel(const __grid_constant__ postprocess_globals<C> g) {
    constexpr int kQSubtilesPerTile = C::Mb / kRefTileM;
    const int q_tile_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_base =
        (q_tile_group_idx * C::ClusterSize) * kQSubtilesPerTile + q_subtile_in_group;

    if (g.deterministic && laneid() == 0) {
        for (int cluster_rank = 0; cluster_rank < C::ClusterSize; ++cluster_rank) {
            int *sem = g.dq_semaphore + dq_semaphore_index(g, batch_idx, head_idx, q_tile_idx, cluster_rank);
            while (atomicAdd(sem, 0) < g.cluster_groups) {
            }
        }
    }
    __syncwarp();

    rt_fl<kRefTileM, C::Dqk> dq_reg, dq_partial;
    warp::load(dq_reg, g.dq_accum, {batch_idx, head_idx, scratch_tile_base, 0});
    if constexpr (C::ClusterSize == 2) {
        warp::load(dq_partial, g.dq_accum, {batch_idx, head_idx, scratch_tile_base + kQSubtilesPerTile, 0});
        warp::add(dq_reg, dq_reg, dq_partial);
    }
    warp::store(g.dq, dq_reg, {batch_idx, head_idx, q_tile_idx, 0});
}

}  // namespace detail

template <typename C>
inline void launch_postprocess(
    at::Tensor &dq_accum,
    at::Tensor &dq,
    at::Tensor &dq_semaphore,
    bool deterministic
) {
    using G = postprocess_globals<C>;
    using dqacc_gl = typename G::dqacc_gl;
    const int q_tile_groups = static_cast<int>(dq_accum.size(2));
    const int q_tiles = static_cast<int>(dq.size(2) / kRefTileM);
    const int dqacc_rows = q_tile_groups * C::ClusterSize * C::Mb;
    const int cluster_groups = static_cast<int>(dq.size(2) / (kForwardTileN * C::ClusterSize));

    G g{
        ::kittens::make_gl<dqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(dq.size(0)),
            static_cast<int>(dq.size(1)),
            dqacc_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
        dq_semaphore.defined() ? reinterpret_cast<int *>(dq_semaphore.data_ptr<int>()) : nullptr,
        static_cast<int>(dq.size(1)),
        q_tiles,
        cluster_groups,
        deterministic ? 1 : 0,
    };

    dim3 grid(q_tiles, dq.size(1), dq.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    detail::postprocess_kernel<C><<<grid, kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd_fa4

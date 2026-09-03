#pragma once

#include "b300_common.cuh"

namespace tkfa4::bwd {

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
struct config {
    static_assert(_Mb == kForwardTileM, "Exact B300 backward requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 backward requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 backward requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 backward requires Dvo=128");
    static_assert(_ClusterSize == 1 || _ClusterSize == 2, "Unsupported exact B300 backward cluster size");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = _ClusterSize;
    static constexpr int WarpTiles = 8;
    static constexpr int BlockThreads = 256;
    static constexpr int MinBlocksPerSm = 1;
};

template <typename C>
struct main_globals {
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using q_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using l_gl = gl<float, -1, -1, -1, -1, stats_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dqacc_gl dq_accum;
    dk_gl dk;
    dv_gl dv;
    l_gl l_aux;
    d_gl delta;
    float scale;
    float scale_log2e;
    int seq_len;
    int actual_seq_len;
};

template <typename C>
struct dv_only_globals {
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using q_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using do_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using l_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    do_gl dout;
    dv_gl dv;
    l_gl l_aux;
    float scale_log2e;
    int seq_len;
    int actual_seq_len;
};

template <typename C>
struct reduce_globals {
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
    using dq_gl = gl<float, -1, -1, -1, C::Dqk>;

    dqacc_gl dq_accum;
    dq_gl dq;
};

namespace detail {

template <typename C>
__device__ inline void reconstruct_probability_tile(
    rt_fl<kRefTileM, kRefTileN> &p,
    const rt_bf<kRefTileM, C::Dqk> &q_reg,
    const rt_bf<kRefTileM, C::Dqk> &k_reg,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &l_aux,
    float scale_log2e,
    int q_tile_idx,
    int k_tile_idx,
    int actual_seq_len,
    bool causal
) {
    warp::broadcast_row(p, l_aux);
    warp::mma_ABt(p, q_reg, k_reg, p);
    warp::mul(p, p, scale_log2e);
    apply_reference_mask(p, q_tile_idx, k_tile_idx, actual_seq_len, causal);
    warp::exp2(p, p);
}

template <typename C>
__device__ inline void reconstruct_probability_tile_dense(
    rt_fl<kRefTileM, kRefTileN> &p,
    const rt_bf<kRefTileM, C::Dqk> &q_reg,
    const rt_bf<kRefTileM, C::Dqk> &k_reg,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &l_aux,
    float scale_log2e
) {
    warp::broadcast_row(p, l_aux);
    warp::mma_ABt(p, q_reg, k_reg, p);
    warp::mul(p, p, scale_log2e);
    warp::exp2(p, p);
}

template <typename C, bool CAUSAL>
__device__ inline void backward_tile_step(
    rt_fl<kRefTileM, kRefTileN> &p,
    rt_fl<kRefTileM, kRefTileN> &dp,
    rt_fl<kRefTileM, kRefTileN> &ds,
    rt_fl<kRefTileM, C::Dqk> &dq_accum,
    rt_fl<kRefTileM, C::Dqk> &dk_accum,
    rt_fl<kRefTileM, C::Dvo> &dv_accum,
    const rt_bf<kRefTileM, C::Dqk> &q_reg,
    const rt_bf<kRefTileM, C::Dqk> &k_reg,
    const rt_bf<kRefTileM, C::Dvo> &v_reg,
    const rt_bf<kRefTileM, C::Dvo> &do_reg,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &l_aux_vec,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &delta_vec,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_subtile_idx,
    int actual_seq_len,
    bool dense_unmasked
) {
    if (dense_unmasked) {
        reconstruct_probability_tile_dense<C>(p, q_reg, k_reg, l_aux_vec, scale_log2e);
    } else {
        reconstruct_probability_tile<C>(
            p,
            q_reg,
            k_reg,
            l_aux_vec,
            scale_log2e,
            q_tile_idx,
            kv_subtile_idx,
            actual_seq_len,
            CAUSAL
        );
    }

    {
        rt_bf<kRefTileM, kRefTileN> p_bf;
        rt_bf<kRefTileM, C::Dvo, ducks::rt_layout::col> do_col;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col;
        warp::copy(p_bf, p);
        warp::swap_layout(p_col, p_bf);
        warp::swap_layout(do_col, do_reg);
        warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);
    }

    warp::zero(dp);
    warp::mma_ABt(dp, do_reg, v_reg, dp);
    warp::sub_row(dp, dp, delta_vec);
    warp::mul(ds, p, dp);
    warp::mul(ds, ds, scale);
    {
        rt_bf<kRefTileM, kRefTileN> ds_bf;
        rt_bf<kRefTileM, C::Dqk, ducks::rt_layout::col> q_col;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
        rt_bf<kRefTileM, C::Dqk, ducks::rt_layout::col> k_col;
        warp::copy(ds_bf, ds);
        warp::swap_layout(ds_col, ds_bf);
        warp::swap_layout(q_col, q_reg);
        warp::swap_layout(k_col, k_reg);
        warp::mma_AtB(dk_accum, ds_col, q_col, dk_accum);

        warp::zero(dq_accum);
        warp::mma_AB(dq_accum, ds_bf, k_col, dq_accum);
    }
}

template <typename C, bool CAUSAL>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel(const __grid_constant__ main_globals<C> g) {
    constexpr int kQSubtilesPerTile = C::Mb / kRefTileM;
    constexpr int q_tiles_buffered = 4;

    using qk_bf_tile = st_bf<kRefTileM, C::Dqk>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo>;
    using qk_fl_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) qk_fl_tile dq_smem[C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile l_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile delta_smem[q_tiles_buffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = C::ClusterSize == 2 ? cluster_ctarank() : 0;
    const int kv_block_idx = C::ClusterSize == 2
        ? static_cast<int>(clusterIdx().x) * C::ClusterSize + cluster_rank
        : static_cast<int>(blockIdx.x);
    const int num_k_blocks = g.seq_len / (kRefTileN * C::WarpTiles);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int num_q_blocks = g.seq_len / (kRefTileM * q_tiles_buffered);
    const bool dense_unmasked = !CAUSAL && g.actual_seq_len == g.seq_len;

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, C::Dqk> dk_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load(k_reg, g.k, {batch_idx, head_idx, kv_subtile_idx, 0});
    warp::load(v_reg, g.v, {batch_idx, head_idx, kv_subtile_idx, 0});
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;

        if (warp < q_tiles_buffered) {
            rt_bf<kRefTileM, C::Dqk> q_stage_reg;
            rt_bf<kRefTileM, C::Dvo> do_stage_reg;
            stats_vec l_stage_vec, delta_stage_vec;
            warp::load(q_stage_reg, g.q, {batch_idx, head_idx, q_tile_base + warp, 0});
            warp::store(q_smem[warp], q_stage_reg);
            warp::load(do_stage_reg, g.dout, {batch_idx, head_idx, q_tile_base + warp, 0});
            warp::store(do_smem[warp], do_stage_reg);
            warp::load(l_stage_vec, g.l_aux, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(l_smem[warp], l_stage_vec);
            warp::load(delta_stage_vec, g.delta, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(delta_smem[warp], delta_stage_vec);
        }
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, kRefTileN> p, dp, ds;
            rt_fl<kRefTileM, C::Dqk> dq_accum;
            stats_vec l_aux_vec, delta_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(l_aux_vec, l_smem[subtile]);
            warp::load(delta_vec, delta_smem[subtile]);

            backward_tile_step<C, CAUSAL>(
                p,
                dp,
                ds,
                dq_accum,
                dk_accum,
                dv_accum,
                q_reg,
                k_reg,
                v_reg,
                do_reg,
                l_aux_vec,
                delta_vec,
                g.scale,
                g.scale_log2e,
                q_tile_idx,
                kv_subtile_idx,
                g.actual_seq_len,
                dense_unmasked
            );
            warp::store(dq_smem[warp], dq_accum);
            __syncwarp();

            const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
            const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
            const int scratch_tile_idx =
                ((q_tile_group_idx * C::ClusterSize) + cluster_rank) * kQSubtilesPerTile + q_subtile_in_group;
            warp::tma::store_add_async(g.dq_accum, dq_smem[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
            warp::tma::store_async_wait();
        }
        __syncthreads();
    }

    warp::store(g.dk, dk_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
    warp::store(g.dv, dv_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
}

template <typename C, bool CAUSAL>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void dv_only_kernel(const __grid_constant__ dv_only_globals<C> g) {
    constexpr int q_tiles_buffered = 4;

    using qk_bf_tile = st_bf<kRefTileM, C::Dqk>;
    using do_bf_tile = st_bf<kRefTileM, C::Dvo>;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) do_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile l_smem[q_tiles_buffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = C::ClusterSize == 2 ? cluster_ctarank() : 0;
    const int kv_block_idx = C::ClusterSize == 2
        ? static_cast<int>(clusterIdx().x) * C::ClusterSize + cluster_rank
        : static_cast<int>(blockIdx.x);
    const int num_k_blocks = g.seq_len / (kRefTileN * C::WarpTiles);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int num_q_blocks = g.seq_len / (kRefTileM * q_tiles_buffered);
    const bool dense_unmasked = !CAUSAL && g.actual_seq_len == g.seq_len;

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load(k_reg, g.k, {batch_idx, head_idx, kv_subtile_idx, 0});
    warp::zero(dv_accum);

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;

        if (warp < q_tiles_buffered) {
            rt_bf<kRefTileM, C::Dqk> q_stage_reg;
            rt_bf<kRefTileM, C::Dvo> do_stage_reg;
            stats_vec l_stage_vec;
            warp::load(q_stage_reg, g.q, {batch_idx, head_idx, q_tile_base + warp, 0});
            warp::store(q_smem[warp], q_stage_reg);
            warp::load(do_stage_reg, g.dout, {batch_idx, head_idx, q_tile_base + warp, 0});
            warp::store(do_smem[warp], do_stage_reg);
            warp::load(l_stage_vec, g.l_aux, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(l_smem[warp], l_stage_vec);
        }
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, kRefTileN> p;
            stats_vec l_aux_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(l_aux_vec, l_smem[subtile]);

            if (dense_unmasked) {
                reconstruct_probability_tile_dense<C>(p, q_reg, k_reg, l_aux_vec, g.scale_log2e);
            } else {
                reconstruct_probability_tile<C>(
                    p,
                    q_reg,
                    k_reg,
                    l_aux_vec,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.actual_seq_len,
                    CAUSAL
                );
            }

            rt_bf<kRefTileM, kRefTileN> p_bf;
            rt_bf<kRefTileM, C::Dvo, ducks::rt_layout::col> do_col;
            rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col;
            warp::copy(p_bf, p);
            warp::swap_layout(p_col, p_bf);
            warp::swap_layout(do_col, do_reg);
            warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);
        }
        __syncthreads();
    }

    warp::store(g.dv, dv_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
}

template <typename C>
__global__ __launch_bounds__(kWarpThreads, 8)
void reduce_kernel(const __grid_constant__ reduce_globals<C> g) {
    constexpr int kQSubtilesPerTile = C::Mb / kRefTileM;
    const int q_tile_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_base =
        (q_tile_group_idx * C::ClusterSize) * kQSubtilesPerTile + q_subtile_in_group;

    rt_fl<kRefTileM, C::Dqk> dq_reg, dq_partial;
    warp::load(dq_reg, g.dq_accum, {batch_idx, head_idx, scratch_tile_base, 0});
    if constexpr (C::ClusterSize == 2) {
        warp::load(dq_partial, g.dq_accum, {batch_idx, head_idx, scratch_tile_base + kQSubtilesPerTile, 0});
        warp::add(dq_reg, dq_reg, dq_partial);
    }
    warp::store(g.dq, dq_reg, {batch_idx, head_idx, q_tile_idx, 0});
}

template <typename C>
inline void launch_main_kernel(
    const main_globals<C> &g,
    bool causal,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    if constexpr (C::ClusterSize == 2) {
        kittens::LaunchConfig<true, false> launch_config(
            dim3(num_k_blocks, heads, batch_size),
            dim3(C::BlockThreads, 1, 1),
            0,
            stream,
            dim3(C::ClusterSize, 1, 1)
        );
        if (causal) {
            CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel<C, true>, g));
        } else {
            CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel<C, false>, g));
        }
        return;
    }

    dim3 grid(num_k_blocks, heads, batch_size);
    if (causal) {
        main_kernel<C, true><<<grid, C::BlockThreads, 0, stream>>>(g);
    } else {
        main_kernel<C, false><<<grid, C::BlockThreads, 0, stream>>>(g);
    }
}

template <typename C>
inline void launch_dv_only_kernel(
    const dv_only_globals<C> &g,
    bool causal,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    if constexpr (C::ClusterSize == 2) {
        kittens::LaunchConfig<true, false> launch_config(
            dim3(num_k_blocks, heads, batch_size),
            dim3(C::BlockThreads, 1, 1),
            0,
            stream,
            dim3(C::ClusterSize, 1, 1)
        );
        if (causal) {
            CUDACHECK(cudaLaunchKernelEx(launch_config, dv_only_kernel<C, true>, g));
        } else {
            CUDACHECK(cudaLaunchKernelEx(launch_config, dv_only_kernel<C, false>, g));
        }
        return;
    }

    dim3 grid(num_k_blocks, heads, batch_size);
    if (causal) {
        dv_only_kernel<C, true><<<grid, C::BlockThreads, 0, stream>>>(g);
    } else {
        dv_only_kernel<C, false><<<grid, C::BlockThreads, 0, stream>>>(g);
    }
}

template <typename C>
inline void launch_reduce_kernel(
    const reduce_globals<C> &g,
    int q_tiles,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid(q_tiles, heads, batch_size);
    reduce_kernel<C><<<grid, kWarpThreads, 0, stream>>>(g);
}

}  // namespace detail

template <typename C>
inline int select_backward_cluster_size(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len,
    bool deterministic
) {
    (void)q;
    (void)k;
    (void)causal;
    (void)actual_seq_len;
    (void)deterministic;
    return C::ClusterSize;
}

template <typename C>
inline void launch_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &dq_accum,
    bool causal,
    float scale,
    int actual_seq_len,
    bool deterministic
) {
    (void)deterministic;

    using G = main_globals<C>;
    using dqacc_gl = typename G::dqacc_gl;
    const int q_tile_groups = static_cast<int>(dq_accum.size(2));
    const int q_tiles = static_cast<int>(q.size(2) / kRefTileM);
    const int dqacc_rows = q_tile_groups * C::ClusterSize * C::Mb;

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        ::kittens::make_gl<dqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(1)),
            dqacc_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        kittens::py::tensor_to_gl<typename G::d_gl>(delta, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(2)),
        actual_seq_len,
    };

    using R = reduce_globals<C>;
    using rdqacc_gl = typename R::dqacc_gl;
    R rg{
        ::kittens::make_gl<rdqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(1)),
            dqacc_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename R::dq_gl>(dq),
    };

    const int num_k_blocks = static_cast<int>(q.size(2) / (kRefTileN * C::WarpTiles));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    detail::launch_main_kernel<C>(
        g,
        causal,
        num_k_blocks,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    detail::launch_reduce_kernel<C>(
        rg,
        q_tiles,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_backward_dv_only(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &dv,
    bool causal,
    float scale,
    int actual_seq_len,
    bool deterministic
) {
    (void)deterministic;

    using G = dv_only_globals<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        scale * kLog2E,
        static_cast<int>(q.size(2)),
        actual_seq_len,
    };

    const int num_k_blocks = static_cast<int>(q.size(2) / (kRefTileN * C::WarpTiles));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    detail::launch_dv_only_kernel<C>(
        g,
        causal,
        num_k_blocks,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd

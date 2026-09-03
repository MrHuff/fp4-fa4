#pragma once

#include "b300_bwd_fa4.cuh"
#include "b300_bwd_hot.cuh"
#include "b300_common.cuh"

namespace tkfa4::bwd_cute16_kernel {

constexpr bool kUseDirectDQ = true;

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
struct config {
    static_assert(_Mb == kForwardTileM, "Exact B300 CuTe16 kernel requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 CuTe16 kernel requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 CuTe16 kernel requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 CuTe16 kernel requires Dvo=128");
    static_assert(_ClusterSize == 2, "Exact B300 CuTe16 kernel requires ClusterSize=2");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = _ClusterSize;

    static constexpr int TileRows = Nb / 2;
    static constexpr int QSubtiles = TileRows / kRefTileM;
    static constexpr int ConsumerWarpgroups = 2;
    static constexpr int ComputeWarps = ConsumerWarpgroups * WARPGROUP_WARPS;
    static constexpr int ReduceWarps = 4;
    static constexpr int MmaWarpId = 12;
    static constexpr int LoadWarpId = 13;
    static constexpr int RelayWarpId = 14;
    static constexpr int EmptyWarpId = 15;
    static constexpr int TotalWarps = 16;
    static constexpr int BlockThreads = TotalWarps * kWarpThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <typename C>
struct main_globals {
    using q_tile = st_bf<C::TileRows, C::Dqk>;
    using k_tile = st_bf<C::TileRows, C::Dqk>;
    using v_tile = st_bf<C::TileRows, C::Dvo>;
    using do_tile = st_bf<C::TileRows, C::Dvo>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using dk_full_fix_tile = st_fl<kRefTileM, C::Dqk>;
    using dk0_tile = st_fl<kRefTileM, 64>;
    using dk1_tile = st_fl<kRefTileM, 64>;
    using dk2_tile = st_fl<kRefTileM, 64>;
    using dv_tile = st_fl<kRefTileM, C::Dvo>;
    using dk_full_tile = st_fl<C::TileRows, 64>;
    using dv_full_tile = st_fl<C::TileRows, C::Dvo>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<k_tile, dim::DEPTH>>;
    using v_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tile, dim::DEPTH>>;
    using q_fix_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using k_fix_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_fix_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_fix_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dq_chunk_tile>;
    using dq_out_gl = gl<float, -1, -1, -1, -1, dq_chunk_tile>;
    using dk_full_gl = gl<float, -1, -1, -1, -1, dk_full_fix_tile>;
    using dk0_gl = gl<float, -1, -1, -1, -1, dk0_tile>;
    using dk1_gl = gl<float, -1, -1, -1, -1, dk1_tile>;
    using dk2_gl = gl<float, -1, -1, -1, -1, dk2_tile>;
    using dv_gl = gl<float, -1, -1, -1, -1, dv_tile>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    q_fix_gl q_fix;
    k_fix_gl k_fix;
    v_fix_gl v_fix;
    do_fix_gl dout_fix;
    dqacc_gl dqacc0;
    dqacc_gl dqacc1;
    dqacc_gl dqacc2;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    dk_full_gl dk_full;
    dk0_gl dk0;
    dk1_gl dk1;
    dk2_gl dk2;
    dv_gl dv;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
    int causal_q_start_offset_blocks;
    int full_causal_patch_coverage;
    int use_exact_bulk_math;
};

template <typename C>
struct dq_patch_globals {
    using q_fix_gl = typename main_globals<C>::q_fix_gl;
    using k_fix_gl = typename main_globals<C>::k_fix_gl;
    using v_fix_gl = typename main_globals<C>::v_fix_gl;
    using do_fix_gl = typename main_globals<C>::do_fix_gl;
    using dq_fix_gl = gl<float, -1, -1, -1, C::Dqk>;
    using stats_gl = typename main_globals<C>::stats_gl;

    q_fix_gl q;
    k_fix_gl k;
    v_fix_gl v;
    do_fix_gl dout;
    dq_fix_gl dq;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

namespace detail {

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel(const __grid_constant__ main_globals<C> g);

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void dkdv_only_kernel(const __grid_constant__ main_globals<C> g);

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::ReduceWarps * kWarpThreads, 8)
void dq_reduce_kernel(const __grid_constant__ main_globals<C> g);

template <bool REPAIR_DV, typename C>
__global__ __launch_bounds__(WARPGROUP_WARPS * kWarpThreads, 8)
void causal_first_tile_patch_kernel(const __grid_constant__ main_globals<C> g, int kv_tile64_offset);

template <typename C>
__global__ __launch_bounds__(C::QSubtiles * kWarpThreads, 8)
void causal_dq_diagonal_patch_kernel(const __grid_constant__ main_globals<C> g);

template <bool CAUSAL, bool COMPUTE_DKDV, typename C, typename AttnTT, typename DkTT, typename DvTT>
__device__ inline void compute_dkdv_loop(
    kittens::semaphore &q_b,
    kittens::semaphore &o_b,
    kittens::semaphore &score_ready,
    kittens::semaphore &dp_ready,
    rt_fl<kRefTileM, C::TileRows> &p_block_t,
    rt_fl<kRefTileM, C::TileRows> &dp_block_t,
    rt_fl<kRefTileM, C::TileRows> &ds_block_t,
    rt_bf<kRefTileM, C::TileRows> &p_block_t_mma,
    rt_bf<kRefTileM, C::TileRows> &ds_block_t_mma,
    AttnTT &score_tt,
    AttnTT &dp_tt,
    DkTT &dk0_tt,
    DkTT &dk1_tt,
    DkTT &dk2_tt,
    DvTT &dv_tt,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_warp_smem,
    auto &lse_log2_smem,
    auto &dpsum_smem,
    int consumer_idx,
    float scale,
    float scale_log2e,
    int phase,
    bool accumulate,
    int q_tile_idx,
    int kv_tile_idx
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
    using prob_tt = half_tt_bf<C::TileRows>;

    const int q_subtile = warpgroup::warpid();

    stats_vec lse_log2_vec, dpsum_vec;
    warp::load(lse_log2_vec, lse_log2_smem[q_subtile]);
    warp::load(dpsum_vec, dpsum_smem[q_subtile]);

    wait(q_b, phase);
    warpgroup::mm_ABt(score_tt, k_smem[consumer_idx], q_smem[0], score_ready);
    wait(score_ready, phase);
    warpgroup::load_async(p_block_t, score_tt);
    tensor_load_wait();
    warp::mul(p_block_t, p_block_t, scale_log2e);
    bwd_hot::detail::apply_hot_mask<CAUSAL>(p_block_t, q_tile_idx, q_subtile, kv_tile_idx);
    warp::sub_row(p_block_t, p_block_t, lse_log2_vec);
    warp::exp2(p_block_t, p_block_t);
    warp::copy(p_block_t_mma, p_block_t);

    wait(o_b, phase);
    warpgroup::mm_ABt(dp_tt, v_smem[consumer_idx], do_smem[0], dp_ready);
    wait(dp_ready, phase);
    warpgroup::load_async(dp_block_t, dp_tt);
    tensor_load_wait();
    warp::sub_row(dp_block_t, dp_block_t, dpsum_vec);
    warp::mul(ds_block_t, p_block_t, dp_block_t);
    warp::mul(ds_block_t, ds_block_t, scale);
    warp::copy(ds_block_t_mma, ds_block_t);

    if constexpr (COMPUTE_DKDV) {
        const prob_tt probs_tmem = prob_tt{score_tt.addr};
        const prob_tt ds_tmem = prob_tt{dp_tt.addr};
        warpgroup::store_async(probs_tmem, p_block_t_mma);
        warpgroup::store_async(ds_tmem, ds_block_t_mma);
        tensor_store_wait();

        auto q_smem_0 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 0});
        auto q_smem_1 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 1});
        auto q_smem_2 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 2});
        if (accumulate) {
            warpgroup::mma_AB(dk0_tt, ds_tmem, q_smem_0);
            warpgroup::mma_AB(dk1_tt, ds_tmem, q_smem_1);
            warpgroup::mma_AB(dk2_tt, ds_tmem, q_smem_2);
            warpgroup::mma_AB(dv_tt, probs_tmem, do_smem[0]);
        } else {
            warpgroup::mm_AB(dk0_tt, ds_tmem, q_smem_0);
            warpgroup::mm_AB(dk1_tt, ds_tmem, q_smem_1);
            warpgroup::mm_AB(dk2_tt, ds_tmem, q_smem_2);
            warpgroup::mm_AB(dv_tt, probs_tmem, do_smem[0]);
        }
    }
    warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
    group<4>::sync(warpgroup::groupid() + 4);
}

template <bool CAUSAL, bool REPAIR_DV, typename C>
__device__ inline void repair_dkdv_step(
    rt_fl<kRefTileM, C::Dqk> &dk_accum,
    rt_fl<kRefTileM, C::Dvo> &dv_accum,
    const rt_bf<kRefTileM, C::Dqk> &q_reg,
    const rt_bf<kRefTileM, C::Dqk> &k_reg,
    const rt_bf<kRefTileM, C::Dvo> &v_reg,
    const rt_bf<kRefTileM, C::Dvo> &do_reg,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &lse_log2_vec,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &dpsum_vec,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_subtile_idx,
    int actual_seq_len,
    bool dense_unmasked
) {
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;
    if (dense_unmasked) {
        bwd_fa4::detail::reconstruct_probability_tile_dense<C>(p, q_reg, k_reg, lse_log2_vec, scale_log2e);
    } else {
        bwd_fa4::detail::reconstruct_probability_tile<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e,
            q_tile_idx,
            kv_subtile_idx,
            actual_seq_len,
            CAUSAL
        );
    }

    if constexpr (REPAIR_DV) {
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
    warp::sub_row(dp, dp, dpsum_vec);
    warp::mul(ds, p, dp);
    warp::mul(ds, ds, scale);

    {
        rt_bf<kRefTileM, kRefTileN> ds_bf;
        rt_bf<kRefTileM, C::Dqk, ducks::rt_layout::col> q_col;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
        warp::copy(ds_bf, ds);
        warp::swap_layout(ds_col, ds_bf);
        warp::swap_layout(q_col, q_reg);
        warp::mma_AtB(dk_accum, ds_col, q_col, dk_accum);
    }
}

template <bool CAUSAL, typename C>
__device__ inline void repair_dq_step(
    rt_fl<kRefTileM, C::Dqk> &dq_partial,
    const rt_bf<kRefTileM, C::Dqk> &q_reg,
    const rt_bf<kRefTileM, C::Dqk> &k_reg,
    const rt_bf<kRefTileM, C::Dvo> &v_reg,
    const rt_bf<kRefTileM, C::Dvo> &do_reg,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &lse_log2_vec,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &dpsum_vec,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_subtile_idx,
    int actual_seq_len,
    bool dense_unmasked
) {
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;
    if (dense_unmasked) {
        bwd_fa4::detail::reconstruct_probability_tile_dense<C>(p, q_reg, k_reg, lse_log2_vec, scale_log2e);
    } else {
        bwd_fa4::detail::reconstruct_probability_tile<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e,
            q_tile_idx,
            kv_subtile_idx,
            actual_seq_len,
            CAUSAL
        );
    }

    warp::zero(dp);
    warp::mma_ABt(dp, do_reg, v_reg, dp);
    warp::sub_row(dp, dp, dpsum_vec);
    warp::mul(ds, p, dp);
    warp::mul(ds, ds, scale);

    rt_bf<kRefTileM, kRefTileN> ds_bf;
    rt_bf<kRefTileM, C::Dqk, ducks::rt_layout::col> k_col;
    warp::copy(ds_bf, ds);
    warp::swap_layout(k_col, k_reg);
    warp::zero(dq_partial);
    warp::mma_AB(dq_partial, ds_bf, k_col, dq_partial);
}

template <int ChunkIdx, typename FullTile, typename ChunkTile>
__device__ inline void extract_chunk(
    const FullTile &src,
    ChunkTile &dst
) {
    static_assert(FullTile::height == ChunkTile::height);
    static_assert(FullTile::width == ChunkTile::width * 3);
    #pragma unroll
    for (int i = 0; i < ChunkTile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < ChunkTile::width; ++j) {
            dst.tiles[i][j] = src.tiles[i][j + ChunkIdx * ChunkTile::width];
        }
    }
}

template <bool CAUSAL, typename C>
__device__ inline void compute_dkdv_exact_reg_loop(
    kittens::semaphore &q_b,
    kittens::semaphore &o_b,
    rt_fl<kRefTileM, 64> &dk0_accum,
    rt_fl<kRefTileM, 64> &dk1_accum,
    rt_fl<kRefTileM, 64> &dk2_accum,
    rt_fl<kRefTileM, C::Dvo> &dv_accum,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_warp_smem,
    auto &lse_log2_smem,
    auto &dpsum_smem,
    int consumer_idx,
    float scale,
    float scale_log2e,
    int phase,
    int q_block_idx,
    int kv_tile_idx,
    int actual_seq_len
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
    const int kv_subtile_local = warpgroup::warpid();
    const int kv_subtile_idx = kv_tile_idx * C::QSubtiles + kv_subtile_local;

    rt_bf<kRefTileM, C::Dqk> k_reg, q_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    auto k_subtile = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({kv_subtile_local, 0});
    auto v_subtile = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({kv_subtile_local, 0});
    warp::load(k_reg, k_subtile);
    warp::load(v_reg, v_subtile);

    if (kv_subtile_local == 0) {
        rt_bf<kRefTileM, C::TileRows> zero_ds;
        warp::zero(zero_ds);
        #pragma unroll
        for (int q_subtile = 0; q_subtile < C::QSubtiles; ++q_subtile) {
            warp::store(ds_warp_smem[consumer_idx][q_subtile], zero_ds);
        }
    }
    group<4>::sync(warpgroup::groupid() + 4);

    wait(q_b, phase);
    wait(o_b, phase);

    #pragma unroll
    for (int q_subtile = 0; q_subtile < C::QSubtiles; ++q_subtile) {
        const int q_tile_idx = q_block_idx * C::QSubtiles + q_subtile;
        stats_vec lse_log2_vec, dpsum_vec;
        auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
        auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
        warp::load(q_reg, q_subtile_smem);
        warp::load(do_reg, do_subtile_smem);
        warp::load(lse_log2_vec, lse_log2_smem[q_subtile]);
        warp::load(dpsum_vec, dpsum_smem[q_subtile]);

        const bool dense_unmasked = !CAUSAL || q_tile_idx > kv_subtile_idx;
        rt_fl<kRefTileM, kRefTileN> p, dp, ds;
        if (dense_unmasked) {
            bwd_fa4::detail::reconstruct_probability_tile_dense<C>(p, q_reg, k_reg, lse_log2_vec, scale_log2e);
        } else {
            bwd_fa4::detail::reconstruct_probability_tile<C>(
                p,
                q_reg,
                k_reg,
                lse_log2_vec,
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
        warp::sub_row(dp, dp, dpsum_vec);
        warp::mul(ds, p, dp);
        warp::mul(ds, ds, scale);

        rt_bf<kRefTileM, kRefTileN> ds_bf;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
        rt_bf<kRefTileM, 64> q_chunk;
        rt_bf<kRefTileM, 64, ducks::rt_layout::col> q_col;
        warp::copy(ds_bf, ds);
        warp::swap_layout(ds_col, ds_bf);

        extract_chunk<0>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk0_accum, ds_col, q_col, dk0_accum);

        extract_chunk<1>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk1_accum, ds_col, q_col, dk1_accum);

        extract_chunk<2>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk2_accum, ds_col, q_col, dk2_accum);
    }
}

template <typename FullTile, typename ChunkTile>
__device__ inline void stitch_three_chunks(
    FullTile &dst,
    const ChunkTile &chunk0,
    const ChunkTile &chunk1,
    const ChunkTile &chunk2
) {
    static_assert(FullTile::height == ChunkTile::height);
    static_assert(FullTile::width == ChunkTile::width * 3);
    #pragma unroll
    for (int i = 0; i < ChunkTile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < ChunkTile::width; ++j) {
            dst.tiles[i][j] = chunk0.tiles[i][j];
            dst.tiles[i][j + ChunkTile::width] = chunk1.tiles[i][j];
            dst.tiles[i][j + 2 * ChunkTile::width] = chunk2.tiles[i][j];
        }
    }
}

template <typename FullTile, typename ChunkTile>
__device__ inline void split_three_chunks(
    const FullTile &src,
    ChunkTile &chunk0,
    ChunkTile &chunk1,
    ChunkTile &chunk2
) {
    static_assert(FullTile::height == ChunkTile::height);
    static_assert(FullTile::width == ChunkTile::width * 3);
    #pragma unroll
    for (int i = 0; i < ChunkTile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < ChunkTile::width; ++j) {
            chunk0.tiles[i][j] = src.tiles[i][j];
            chunk1.tiles[i][j] = src.tiles[i][j + ChunkTile::width];
            chunk2.tiles[i][j] = src.tiles[i][j + 2 * ChunkTile::width];
        }
    }
}

template <bool CAUSAL, typename C>
inline void launch_main(
    const main_globals<C> &g,
    int total_ctas,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(total_ctas, heads, batch_size),
        dim3(C::BlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if constexpr (CAUSAL) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel<true, C>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel<false, C>, g));
    }
}

template <bool CAUSAL, typename C>
inline void launch_dkdv_only(
    const main_globals<C> &g,
    int total_ctas,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(total_ctas, heads, batch_size),
        dim3(C::BlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if constexpr (CAUSAL) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, dkdv_only_kernel<true, C>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, dkdv_only_kernel<false, C>, g));
    }
}

template <typename C>
inline void launch_reduce(
    const main_globals<C> &g,
    int q_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream,
    bool causal
) {
    dim3 grid(q_blocks, heads, batch_size);
    if (causal) {
        dq_reduce_kernel<true, C><<<grid, C::ReduceWarps * kWarpThreads, 0, stream>>>(g);
    } else {
        dq_reduce_kernel<false, C><<<grid, C::ReduceWarps * kWarpThreads, 0, stream>>>(g);
    }
}

template <typename C>
inline void launch_causal_first_tile_patch(
    const main_globals<C> &g,
    int kv_tiles64,
    int heads,
    int batch_size,
    cudaStream_t stream,
    int kv_tile64_offset = 0
) {
    dim3 grid(heads, batch_size, kv_tiles64);
    causal_first_tile_patch_kernel<true, C><<<grid, WARPGROUP_WARPS * kWarpThreads, 0, stream>>>(g, kv_tile64_offset);
}

template <typename C>
inline void launch_causal_dk_only_patch(
    const main_globals<C> &g,
    int kv_tiles64,
    int heads,
    int batch_size,
    cudaStream_t stream,
    int kv_tile64_offset
) {
    dim3 grid(heads, batch_size, kv_tiles64);
    causal_first_tile_patch_kernel<false, C><<<grid, WARPGROUP_WARPS * kWarpThreads, 0, stream>>>(g, kv_tile64_offset);
}

template <typename C>
inline void launch_causal_dq_diagonal_patch(
    const main_globals<C> &g,
    int q_blocks64,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid(q_blocks64, heads, batch_size);
    causal_dq_diagonal_patch_kernel<C><<<grid, C::QSubtiles * kWarpThreads, 0, stream>>>(g);
}

}  // namespace detail

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void detail::main_kernel(const __grid_constant__ main_globals<C> g) {
    using q_tile = typename main_globals<C>::q_tile;
    using k_tile = typename main_globals<C>::k_tile;
    using v_tile = typename main_globals<C>::v_tile;
    using do_tile = typename main_globals<C>::do_tile;
    using dq_chunk_tile = typename main_globals<C>::dq_chunk_tile;
    using stats_smem_tile = typename main_globals<C>::stats_tile;
    using ds_warp_tile = st_bf<kRefTileM, C::TileRows>;
    using attn_tt = half_tt_fl<C::TileRows>;
    using dk_tt = half_tt_fl<64>;
    using dv_tt = half_tt_fl<C::Dvo>;
    struct main_shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        dq_chunk_tile dq_smem[3][C::QSubtiles];
        ds_warp_tile ds_warp_smem[C::ConsumerWarpgroups][WARPGROUP_WARPS];
        stats_smem_tile lse_log2_smem[C::QSubtiles];
        stats_smem_tile dpsum_smem[C::QSubtiles];
    };
    __shared__ alignas(1024) main_shared_storage smem;
    auto &k_smem = smem.k_smem;
    auto &v_smem = smem.v_smem;
    auto &q_smem = smem.q_smem;
    auto &do_smem = smem.do_smem;
    auto &dq_smem = smem.dq_smem;
    auto &ds_warp_smem = smem.ds_warp_smem;
    auto &lse_log2_smem = smem.lse_log2_smem;
    auto &dpsum_smem = smem.dpsum_smem;

    __shared__ __align__(16) kittens::semaphore kv_b;
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];
    __shared__ __align__(16) kittens::semaphore score_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore dp_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore kv_tmem_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore compute_group_done[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore dq_subtile_done[2][C::QSubtiles];
    __shared__ __align__(16) kittens::semaphore dq_tile_done[C::QSubtiles];
    __shared__ __align__(16) kittens::semaphore dq_done;
    const int warp = kittens::warpid();
    const bool is_reduce = warp < C::ReduceWarps;
    const bool is_compute = warp >= C::ReduceWarps && warp < C::ReduceWarps + C::ComputeWarps;
    const bool is_mma = warp == C::MmaWarpId;
    const bool is_load = warp == C::LoadWarpId;
    const bool is_relay = warp == C::RelayWarpId;
    const bool is_empty = warp == C::EmptyWarpId;
    const int consumer_idx = is_compute ? ((warp - C::ReduceWarps) / kittens::WARPGROUP_WARPS) : -1;

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cta_rank = cluster_ctarank();
    const int n_block = static_cast<int>(blockIdx.x);
    const int n_block_group = n_block / C::ClusterSize;
    const int num_k_blocks = g.seq_len / (C::TileRows * C::ConsumerWarpgroups);
    const int local_kv_block = n_block_group * C::ClusterSize + cta_rank;
    if (local_kv_block >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = local_kv_block * C::ConsumerWarpgroups;
    const int q_blocks = g.seq_len / C::TileRows;
    const int q_start_block = CAUSAL ? (kv_tile_base + g.causal_q_start_offset_blocks) : 0;
    // The exact causal diagonal dK/dV correction is applied by the dedicated
    // sidecar after the main kernel, so the live clustered path does not need
    // to spend registers/MMAs on a temporary diagonal repair it will overwrite.
    constexpr bool repair_diagonal_causal_tile = false;
    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dk_tt dk0_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk1_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk2_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dv_tt dv_accum_tt[C::ConsumerWarpgroups] = {dv_tt{0}, dv_tt{0}};
    rt_fl<kRefTileM, 64> dk0_reg_accum;
    rt_fl<kRefTileM, 64> dk1_reg_accum;
    rt_fl<kRefTileM, 64> dk2_reg_accum;
    rt_fl<kRefTileM, C::Dvo> dv_reg_accum;
    rt_fl<kRefTileM, C::Dqk> dk_fix_full;
    rt_fl<kRefTileM, C::Dvo> dv_fix_reg;
    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.k.template prefetch_tma<k_tile, dim::DEPTH>();
        g.v.template prefetch_tma<v_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();

        init_semaphore(kv_b, 0, 1);
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
        init_semaphore(dq_subtile_done[0][0], 1, 0);
        init_semaphore(dq_subtile_done[0][1], 1, 0);
        init_semaphore(dq_subtile_done[1][0], 1, 0);
        init_semaphore(dq_subtile_done[1][1], 1, 0);
        init_semaphore(dq_tile_done[0], 1, 0);
        init_semaphore(dq_tile_done[1], 1, 0);
        init_semaphore(dq_done, 1, 0);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            init_semaphore(score_ready[w][0], 0, 1);
            init_semaphore(dp_ready[w][0], 0, 1);
            init_semaphore(kv_tmem_ready[w], 0, 1);
            init_semaphore(compute_group_done[w], 1, 0);
        }
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * C::ConsumerWarpgroups);
        #pragma unroll
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            coord<k_tile> k_tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            coord<v_tile> v_tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem[w], g.k, k_tile_idx, kv_b);
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[w], g.v, v_tile_idx, kv_b);
        }
    }
    __syncthreads();

    if (is_compute) {
        if (!g.use_exact_bulk_math) {
            score_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
            dp_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, C::TileRows);
            dk0_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 2 * C::TileRows);
            dk1_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 3 * C::TileRows);
            dk2_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 4 * C::TileRows);
            dv_accum_tt[consumer_idx] = tm_alloc.template allocate<dv_tt>(consumer_idx, 5 * C::TileRows);
            rt_fl<kRefTileM, 64> zero_dk;
            rt_fl<kRefTileM, C::Dvo> zero_dv;
            warp::zero(zero_dk);
            warp::zero(zero_dv);
            warpgroup::store_async(dk0_tt[consumer_idx], zero_dk);
            warpgroup::store_async(dk1_tt[consumer_idx], zero_dk);
            warpgroup::store_async(dk2_tt[consumer_idx], zero_dk);
            warpgroup::store_async(dv_accum_tt[consumer_idx], zero_dv);
            tensor_store_wait();
        } else {
            warp::zero(dk0_reg_accum);
            warp::zero(dk1_reg_accum);
            warp::zero(dk2_reg_accum);
            warp::zero(dv_reg_accum);
        }
        if constexpr (repair_diagonal_causal_tile) {
            warp::zero(dk_fix_full);
            warp::zero(dv_fix_reg);
        }
    }

    if constexpr (CAUSAL && repair_diagonal_causal_tile) {
        if (is_compute && consumer_idx == 0) {
            rt_bf<kRefTileM, C::Dqk> q_fix_reg, k_fix_reg;
            rt_bf<kRefTileM, C::Dvo> v_fix_reg_bf, do_fix_reg;
            typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_fix, dpsum_fix;
            const int kv_subtile_idx_fix = kv_tile_base * kittens::WARPGROUP_WARPS + warpgroup::warpid();
            const int q_tile_idx_fix = kv_tile_base * C::QSubtiles + warpgroup::warpid();
            warp::load(k_fix_reg, g.k_fix, {batch_idx, kv_subtile_idx_fix, head_idx, 0});
            warp::load(v_fix_reg_bf, g.v_fix, {batch_idx, kv_subtile_idx_fix, head_idx, 0});
            warp::load(q_fix_reg, g.q_fix, {batch_idx, q_tile_idx_fix, head_idx, 0});
            warp::load(do_fix_reg, g.dout_fix, {batch_idx, q_tile_idx_fix, head_idx, 0});
            warp::load(lse_fix, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx_fix});
            warp::load(dpsum_fix, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx_fix});
            detail::repair_dkdv_step<true, true, C>(
                dk_fix_full,
                dv_fix_reg,
                q_fix_reg,
                k_fix_reg,
                v_fix_reg_bf,
                do_fix_reg,
                lse_fix,
                dpsum_fix,
                g.scale,
                g.scale_log2e,
                q_tile_idx_fix,
                kv_subtile_idx_fix,
                g.seq_len,
                false
            );
        }
    }

    if (q_start_block < q_blocks && is_load) {
        coord<q_tile> q_tile_idx = {batch_idx, q_start_block, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, q_start_block, head_idx, 0};
        warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
        warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx, q_b[0]);
        warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
        warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, do_tile_idx, o_b[0]);

        const int q_tile_base = q_start_block * C::QSubtiles;
        for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
            typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_vec, dpsum_vec;
            warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
            warp::store(lse_log2_smem[subtile], lse_vec);
            warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
            warp::store(dpsum_smem[subtile], dpsum_vec);
        }
    }
    __syncthreads();

    for (int q_block_idx = q_start_block; q_block_idx < q_blocks; ++q_block_idx) {
        const int local_q_iter = q_block_idx - q_start_block;
        const int current_phase = local_q_iter & 1;
        if (is_compute && local_q_iter >= 2) {
            wait(dq_done, current_phase);
        }
        if (is_load) {
            // q/do/stats for the current iteration are already staged. The load warp
            // will issue the next block after compute, overlapping with dQ work below.
        }

        if (is_compute) {
            wait(kv_b, 0);
            const bool skip_consumer_diagonal_block =
                CAUSAL &&
                !g.use_exact_bulk_math &&
                q_block_idx == (kv_tile_base + 1) &&
                consumer_idx == 1;
            if (!skip_consumer_diagonal_block && !g.use_exact_bulk_math) {
                rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
                rt_bf<kRefTileM, C::TileRows> p_block_t_mma, ds_block_t_mma;
                detail::compute_dkdv_loop<false, true, C>(
                    q_b[0],
                    o_b[0],
                    score_ready[consumer_idx][0],
                    dp_ready[consumer_idx][0],
                    p_block_t,
                    dp_block_t,
                    ds_block_t,
                    p_block_t_mma,
                    ds_block_t_mma,
                    score_tt[consumer_idx],
                    dp_tt[consumer_idx],
                    dk0_tt[consumer_idx],
                    dk1_tt[consumer_idx],
                    dk2_tt[consumer_idx],
                    dv_accum_tt[consumer_idx],
                    q_smem,
                    k_smem,
                    v_smem,
                    do_smem,
                    ds_warp_smem,
                    lse_log2_smem,
                    dpsum_smem,
                    consumer_idx,
                    g.scale,
                    g.scale_log2e,
                    local_q_iter & 1,
                    true,
                    q_block_idx,
                    kv_tile_base + consumer_idx
                );
            } else if (!skip_consumer_diagonal_block) {
                detail::compute_dkdv_exact_reg_loop<CAUSAL, C>(
                    q_b[0],
                    o_b[0],
                    dk0_reg_accum,
                    dk1_reg_accum,
                    dk2_reg_accum,
                    dv_reg_accum,
                    q_smem,
                    k_smem,
                    v_smem,
                    do_smem,
                    ds_warp_smem,
                    lse_log2_smem,
                    dpsum_smem,
                    consumer_idx,
                    g.scale,
                    g.scale_log2e,
                    local_q_iter & 1,
                    q_block_idx,
                    kv_tile_base + consumer_idx,
                    g.seq_len
                );
            } else {
                if (warpgroup::warpid() == 0) {
                    rt_bf<kRefTileM, C::TileRows> zero_ds;
                    warp::zero(zero_ds);
                    #pragma unroll
                    for (int q_subtile = 0; q_subtile < C::QSubtiles; ++q_subtile) {
                        warp::store(ds_warp_smem[consumer_idx][q_subtile], zero_ds);
                    }
                }
                group<4>::sync(warpgroup::groupid() + 4);
            }
        }
        if (is_compute && warpgroup::laneid() == 0) {
            arrive(compute_group_done[consumer_idx]);
        }

        if (is_load) {
            wait(compute_group_done[0], current_phase);
            wait(compute_group_done[1], current_phase);
            const int next_q_block_idx = q_block_idx + 1;
            if (next_q_block_idx < q_blocks) {
                coord<q_tile> next_q_tile_idx = {batch_idx, next_q_block_idx, head_idx, 0};
                coord<do_tile> next_do_tile_idx = {batch_idx, next_q_block_idx, head_idx, 0};
                warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, next_q_tile_idx, q_b[0]);
                warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, next_do_tile_idx, o_b[0]);

                const int next_q_tile_base = next_q_block_idx * C::QSubtiles;
                for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                    typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_vec, dpsum_vec;
                    warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, next_q_tile_base + subtile});
                    warp::store(lse_log2_smem[subtile], lse_vec);
                    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, next_q_tile_base + subtile});
                    warp::store(dpsum_smem[subtile], dpsum_vec);
                }
            }
        }

        {
            const bool handles_chunk0 = warp < 2;
            const bool handles_chunk1 = warp >= 2 && warp < 4;
            const bool handles_chunk2 = warp == C::RelayWarpId || warp == C::EmptyWarpId;
            if (handles_chunk0 || handles_chunk1 || handles_chunk2) {
                const bool include_peer = !CAUSAL || q_block_idx > (kv_tile_base + 1);
                const int dq_subtile_idx = handles_chunk2 ? (warp == C::RelayWarpId ? 0 : 1) : (warp & 1);
                wait(compute_group_done[0], current_phase);
                if (include_peer) {
                    wait(compute_group_done[1], current_phase);
                }
                const int chunk_idx = handles_chunk0 ? 0 : (handles_chunk1 ? 1 : 2);
                const int q_tile_idx = q_block_idx * C::QSubtiles + dq_subtile_idx;
                rt_bf<kRefTileM, C::TileRows> ds_local_reg;
                rt_bf<kRefTileM, C::TileRows> ds_peer_reg;
                rt_bf<C::TileRows, 64, ducks::rt_layout::col> k_col;
                rt_fl<kRefTileM, 64> dq_chunk;
                auto k0_0 = k_smem[0].template subtile<C::TileRows, 64>({0, 0});
                auto k0_1 = k_smem[0].template subtile<C::TileRows, 64>({0, 1});
                auto k0_2 = k_smem[0].template subtile<C::TileRows, 64>({0, 2});
                auto k1_0 = k_smem[1].template subtile<C::TileRows, 64>({0, 0});
                auto k1_1 = k_smem[1].template subtile<C::TileRows, 64>({0, 1});
                auto k1_2 = k_smem[1].template subtile<C::TileRows, 64>({0, 2});
                warp::load(ds_local_reg, ds_warp_smem[0][dq_subtile_idx]);
                if (include_peer) {
                    warp::load(ds_peer_reg, ds_warp_smem[1][dq_subtile_idx]);
                }

                if (chunk_idx == 0) {
                    warp::load(k_col, k0_0);
                    warp::zero(dq_chunk);
                    warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
                    if (include_peer) {
                        warp::load(k_col, k1_0);
                        warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
                    }
                    warp::store(dq_smem[0][dq_subtile_idx], dq_chunk);
                    if constexpr (kUseDirectDQ) {
                        warp::tma::store_add_async(g.dq0, dq_smem[0][dq_subtile_idx], {batch_idx, q_tile_idx, head_idx, 0});
                    } else {
                        const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cta_rank;
                        warp::tma::store_async(g.dqacc0, dq_smem[0][dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
                    }
                } else if (chunk_idx == 1) {
                    warp::load(k_col, k0_1);
                    warp::zero(dq_chunk);
                    warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
                    if (include_peer) {
                        warp::load(k_col, k1_1);
                        warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
                    }
                    warp::store(dq_smem[1][dq_subtile_idx], dq_chunk);
                    if constexpr (kUseDirectDQ) {
                        warp::tma::store_add_async(g.dq1, dq_smem[1][dq_subtile_idx], {batch_idx, q_tile_idx, head_idx, 1});
                    } else {
                        const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cta_rank;
                        warp::tma::store_async(g.dqacc1, dq_smem[1][dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
                    }
                } else {
                    warp::load(k_col, k0_2);
                    warp::zero(dq_chunk);
                    warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
                    if (include_peer) {
                        warp::load(k_col, k1_2);
                        warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
                    }
                    warp::store(dq_smem[2][dq_subtile_idx], dq_chunk);
                    if constexpr (kUseDirectDQ) {
                        warp::tma::store_add_async(g.dq2, dq_smem[2][dq_subtile_idx], {batch_idx, q_tile_idx, head_idx, 2});
                    } else {
                        const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cta_rank;
                        warp::tma::store_async(g.dqacc2, dq_smem[2][dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
                    }
                }
                if constexpr (kUseDirectDQ) {
                    warp::tma::store_async_wait();
                } else {
                    warp::tma::store_commit_group();
                    warp::tma::store_async_wait();
                }
                if (handles_chunk2) {
                    wait(dq_subtile_done[0][dq_subtile_idx], current_phase);
                    wait(dq_subtile_done[1][dq_subtile_idx], current_phase);
                    if (laneid() == 0) {
                        arrive(dq_tile_done[dq_subtile_idx]);
                    }
                } else if (laneid() == 0) {
                    arrive(dq_subtile_done[chunk_idx][dq_subtile_idx]);
                }
            }
        }
        if (is_mma) {
            wait(dq_tile_done[0], current_phase);
            wait(dq_tile_done[1], current_phase);
            if (laneid() == 0) {
                arrive(dq_done);
            }
        }
    }

    if (is_compute) {
        const int kv_subtile_idx =
            (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + warpgroup::warpid();
        rt_fl<kRefTileM, C::Dqk> dk_full_reg;
        if (!g.use_exact_bulk_math) {
            rt_fl<kRefTileM, 64> dk0_reg;
            rt_fl<kRefTileM, 64> dk1_reg;
            rt_fl<kRefTileM, 64> dk2_reg;
            rt_fl<kRefTileM, C::Dvo> dv_reg;
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(kv_tmem_ready[consumer_idx]);
            }
            wait(kv_tmem_ready[consumer_idx], 0);
            warpgroup::load_async(dv_reg, dv_accum_tt[consumer_idx]);
            tensor_load_wait();
            warp::store<dim::DEPTH>(g.dv, dv_reg, {batch_idx, kv_subtile_idx, head_idx, 0});

            warpgroup::load_async(dk0_reg, dk0_tt[consumer_idx]);
            warpgroup::load_async(dk1_reg, dk1_tt[consumer_idx]);
            warpgroup::load_async(dk2_reg, dk2_tt[consumer_idx]);
            tensor_load_wait();
            stitch_three_chunks(dk_full_reg, dk0_reg, dk1_reg, dk2_reg);
        } else {
            warp::store<dim::DEPTH>(g.dv, dv_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
            stitch_three_chunks(dk_full_reg, dk0_reg_accum, dk1_reg_accum, dk2_reg_accum);
        }
        if constexpr (repair_diagonal_causal_tile) {
            if (consumer_idx == 0) {
                warp::add(dk_full_reg, dk_full_reg, dk_fix_full);
            }
        }
        warp::store<dim::DEPTH>(g.dk_full, dk_full_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
    }
}

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void detail::dkdv_only_kernel(const __grid_constant__ main_globals<C> g) {
    using q_tile = typename main_globals<C>::q_tile;
    using k_tile = typename main_globals<C>::k_tile;
    using v_tile = typename main_globals<C>::v_tile;
    using do_tile = typename main_globals<C>::do_tile;
    using stats_smem_tile = typename main_globals<C>::stats_tile;
    using ds_warp_tile = st_bf<kRefTileM, C::TileRows>;
    using attn_tt = half_tt_fl<C::TileRows>;
    using dk_tt = half_tt_fl<64>;
    using dv_tt = half_tt_fl<C::Dvo>;
    struct dkdv_only_shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        ds_warp_tile ds_warp_smem[C::ConsumerWarpgroups][WARPGROUP_WARPS];
        stats_smem_tile lse_log2_smem[C::QSubtiles];
        stats_smem_tile dpsum_smem[C::QSubtiles];
    };
    __shared__ alignas(1024) dkdv_only_shared_storage smem;
    auto &k_smem = smem.k_smem;
    auto &v_smem = smem.v_smem;
    auto &q_smem = smem.q_smem;
    auto &do_smem = smem.do_smem;
    auto &ds_warp_smem = smem.ds_warp_smem;
    auto &lse_log2_smem = smem.lse_log2_smem;
    auto &dpsum_smem = smem.dpsum_smem;

    __shared__ __align__(16) kittens::semaphore kv_b;
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];
    __shared__ __align__(16) kittens::semaphore score_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore dp_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore kv_tmem_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore compute_group_done[C::ConsumerWarpgroups];

    const int warp = kittens::warpid();
    const bool is_compute = warp >= C::ReduceWarps && warp < C::ReduceWarps + C::ComputeWarps;
    const bool is_load = warp == C::LoadWarpId;
    const int consumer_idx = is_compute ? ((warp - C::ReduceWarps) / kittens::WARPGROUP_WARPS) : -1;

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cta_rank = cluster_ctarank();
    const int kv_block_idx = static_cast<int>(clusterIdx().x) * C::ClusterSize + cta_rank;
    const int num_k_blocks = g.seq_len / (C::TileRows * C::ConsumerWarpgroups);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::ConsumerWarpgroups;
    const int q_blocks = g.seq_len / C::TileRows;
    const int q_start_block = CAUSAL ? (kv_tile_base + g.causal_q_start_offset_blocks) : 0;
    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dk_tt dk0_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk1_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk2_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dv_tt dv_accum_tt[C::ConsumerWarpgroups] = {dv_tt{0}, dv_tt{0}};
    rt_fl<kRefTileM, 64> dk0_reg_accum;
    rt_fl<kRefTileM, 64> dk1_reg_accum;
    rt_fl<kRefTileM, 64> dk2_reg_accum;
    rt_fl<kRefTileM, C::Dvo> dv_reg_accum;

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.k.template prefetch_tma<k_tile, dim::DEPTH>();
        g.v.template prefetch_tma<v_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();

        init_semaphore(kv_b, 0, 1);
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            init_semaphore(score_ready[w][0], 0, 1);
            init_semaphore(dp_ready[w][0], 0, 1);
            init_semaphore(kv_tmem_ready[w], 0, 1);
            init_semaphore(compute_group_done[w], 1, 0);
        }
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * C::ConsumerWarpgroups);
        #pragma unroll
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            coord<k_tile> k_tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            coord<v_tile> v_tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem[w], g.k, k_tile_idx, kv_b);
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[w], g.v, v_tile_idx, kv_b);
        }
    }
    __syncthreads();

    if (is_compute) {
        if (!g.use_exact_bulk_math) {
            score_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
            dp_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, C::TileRows);
            dk0_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 2 * C::TileRows);
            dk1_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 3 * C::TileRows);
            dk2_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 4 * C::TileRows);
            dv_accum_tt[consumer_idx] = tm_alloc.template allocate<dv_tt>(consumer_idx, 5 * C::TileRows);
            rt_fl<kRefTileM, 64> zero_dk;
            rt_fl<kRefTileM, C::Dvo> zero_dv;
            warp::zero(zero_dk);
            warp::zero(zero_dv);
            warpgroup::store_async(dk0_tt[consumer_idx], zero_dk);
            warpgroup::store_async(dk1_tt[consumer_idx], zero_dk);
            warpgroup::store_async(dk2_tt[consumer_idx], zero_dk);
            warpgroup::store_async(dv_accum_tt[consumer_idx], zero_dv);
            tensor_store_wait();
        } else {
            warp::zero(dk0_reg_accum);
            warp::zero(dk1_reg_accum);
            warp::zero(dk2_reg_accum);
            warp::zero(dv_reg_accum);
        }
    }

    for (int q_block_idx = q_start_block; q_block_idx < q_blocks; ++q_block_idx) {
        const int local_q_iter = q_block_idx - q_start_block;
        const int current_phase = local_q_iter & 1;

        if (is_load) {
            coord<q_tile> q_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
            coord<do_tile> do_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx, q_b[0]);
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, do_tile_idx, o_b[0]);

            const int q_tile_base = q_block_idx * C::QSubtiles;
            for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_vec, dpsum_vec;
                warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(lse_log2_smem[subtile], lse_vec);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[subtile], dpsum_vec);
            }
        }
        __syncthreads();

        if (is_compute) {
            wait(kv_b, 0);
            const bool skip_consumer_diagonal_block =
                CAUSAL &&
                !g.use_exact_bulk_math &&
                q_block_idx == (kv_tile_base + 1) &&
                consumer_idx == 1;
            if (!skip_consumer_diagonal_block && !g.use_exact_bulk_math) {
                rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
                rt_bf<kRefTileM, C::TileRows> p_block_t_mma, ds_block_t_mma;
                detail::compute_dkdv_loop<false, true, C>(
                    q_b[0],
                    o_b[0],
                    score_ready[consumer_idx][0],
                    dp_ready[consumer_idx][0],
                    p_block_t,
                    dp_block_t,
                    ds_block_t,
                    p_block_t_mma,
                    ds_block_t_mma,
                    score_tt[consumer_idx],
                    dp_tt[consumer_idx],
                    dk0_tt[consumer_idx],
                    dk1_tt[consumer_idx],
                    dk2_tt[consumer_idx],
                    dv_accum_tt[consumer_idx],
                    q_smem,
                    k_smem,
                    v_smem,
                    do_smem,
                    ds_warp_smem,
                    lse_log2_smem,
                    dpsum_smem,
                    consumer_idx,
                    g.scale,
                    g.scale_log2e,
                    current_phase,
                    true,
                    q_block_idx,
                    kv_tile_base + consumer_idx
                );
            } else if (!skip_consumer_diagonal_block) {
                detail::compute_dkdv_exact_reg_loop<CAUSAL, C>(
                    q_b[0],
                    o_b[0],
                    dk0_reg_accum,
                    dk1_reg_accum,
                    dk2_reg_accum,
                    dv_reg_accum,
                    q_smem,
                    k_smem,
                    v_smem,
                    do_smem,
                    ds_warp_smem,
                    lse_log2_smem,
                    dpsum_smem,
                    consumer_idx,
                    g.scale,
                    g.scale_log2e,
                    current_phase,
                    q_block_idx,
                    kv_tile_base + consumer_idx,
                    g.seq_len
                );
            } else {
                if (warpgroup::warpid() == 0) {
                    rt_bf<kRefTileM, C::TileRows> zero_ds;
                    warp::zero(zero_ds);
                    #pragma unroll
                    for (int q_subtile = 0; q_subtile < C::QSubtiles; ++q_subtile) {
                        warp::store(ds_warp_smem[consumer_idx][q_subtile], zero_ds);
                    }
                }
                group<4>::sync(warpgroup::groupid() + 4);
            }
        }
        __syncthreads();
    }

    if (is_compute) {
        const int kv_subtile_idx =
            (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + warpgroup::warpid();
        rt_fl<kRefTileM, C::Dqk> dk_full_reg;
        if (!g.use_exact_bulk_math) {
            rt_fl<kRefTileM, 64> dk0_reg;
            rt_fl<kRefTileM, 64> dk1_reg;
            rt_fl<kRefTileM, 64> dk2_reg;
            rt_fl<kRefTileM, C::Dvo> dv_reg;
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(kv_tmem_ready[consumer_idx]);
            }
            wait(kv_tmem_ready[consumer_idx], 0);
            warpgroup::load_async(dv_reg, dv_accum_tt[consumer_idx]);
            tensor_load_wait();
            warp::store<dim::DEPTH>(g.dv, dv_reg, {batch_idx, kv_subtile_idx, head_idx, 0});

            warpgroup::load_async(dk0_reg, dk0_tt[consumer_idx]);
            warpgroup::load_async(dk1_reg, dk1_tt[consumer_idx]);
            warpgroup::load_async(dk2_reg, dk2_tt[consumer_idx]);
            tensor_load_wait();
            stitch_three_chunks(dk_full_reg, dk0_reg, dk1_reg, dk2_reg);
        } else {
            warp::store<dim::DEPTH>(g.dv, dv_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
            stitch_three_chunks(dk_full_reg, dk0_reg_accum, dk1_reg_accum, dk2_reg_accum);
        }
        warp::store<dim::DEPTH>(g.dk_full, dk_full_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
    }
}

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::ReduceWarps * kWarpThreads, 8)
void detail::dq_reduce_kernel(const __grid_constant__ main_globals<C> g) {
    const int warp = threadIdx.x >> 5;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    const int q_subtile = warp & 1;
    const int reduce_group = warp >> 1;
    if (reduce_group >= 2) {
        return;
    }

    const int q_tile_idx = q_block_idx * C::QSubtiles + q_subtile;
    const int scratch_tile_idx = q_tile_idx * C::ClusterSize;
    rt_fl<kRefTileM, 64> dq_local, dq_peer;
    if (reduce_group == 0) {
        warp::load(dq_local, g.dqacc0, {batch_idx, head_idx, scratch_tile_idx, 0});
        warp::load(dq_peer, g.dqacc0, {batch_idx, head_idx, scratch_tile_idx + 1, 0});
        warp::add(dq_local, dq_local, dq_peer);
        warp::store<dim::DEPTH>(g.dq0, dq_local, {batch_idx, q_tile_idx, head_idx, 0});

        warp::load(dq_local, g.dqacc1, {batch_idx, head_idx, scratch_tile_idx, 0});
        warp::load(dq_peer, g.dqacc1, {batch_idx, head_idx, scratch_tile_idx + 1, 0});
        warp::add(dq_local, dq_local, dq_peer);
        warp::store<dim::DEPTH>(g.dq1, dq_local, {batch_idx, q_tile_idx, head_idx, 1});
    } else {
        warp::load(dq_local, g.dqacc2, {batch_idx, head_idx, scratch_tile_idx, 0});
        warp::load(dq_peer, g.dqacc2, {batch_idx, head_idx, scratch_tile_idx + 1, 0});
        warp::add(dq_local, dq_local, dq_peer);
        warp::store<dim::DEPTH>(g.dq2, dq_local, {batch_idx, q_tile_idx, head_idx, 2});
    }

}

template <bool REPAIR_DV, typename C>
__global__ __launch_bounds__(WARPGROUP_WARPS * kWarpThreads, 1)
void detail::causal_first_tile_patch_kernel(const __grid_constant__ main_globals<C> g, int kv_tile64_offset) {
    using q_tile = typename main_globals<C>::q_tile;
    using k_tile = typename main_globals<C>::k_tile;
    using v_tile = typename main_globals<C>::v_tile;
    using do_tile = typename main_globals<C>::do_tile;
    __shared__ alignas(1024) q_tile q_smem;
    __shared__ alignas(1024) do_tile do_smem;
    __shared__ alignas(1024) k_tile k_smem;
    __shared__ alignas(1024) v_tile v_smem;
    __shared__ __align__(16) kittens::semaphore kv_b;
    __shared__ __align__(16) kittens::semaphore q_b;
    __shared__ __align__(16) kittens::semaphore o_b;

    const int warp = threadIdx.x >> 5;
    if (warp >= WARPGROUP_WARPS) {
        return;
    }

    const int head_idx = static_cast<int>(blockIdx.x);
    const int batch_idx = static_cast<int>(blockIdx.y);
    const int kv_tile64_idx = static_cast<int>(blockIdx.z) + kv_tile64_offset;
    const int kv_tiles64 = g.seq_len / C::TileRows;
    if (kv_tile64_idx >= kv_tiles64) {
        return;
    }
    const int kv_subtile_idx = kv_tile64_idx * WARPGROUP_WARPS + warp;
    const int q_blocks = g.seq_len / C::TileRows;

    rt_bf<kRefTileM, C::Dqk> k_reg, q_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_vec, dpsum_vec;
    rt_fl<kRefTileM, C::Dqk> dk_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    const int q_block_idx = kv_tile64_idx;
    if (threadIdx.x == 0) {
        g.k.template prefetch_tma<k_tile, dim::DEPTH>();
        g.v.template prefetch_tma<v_tile, dim::DEPTH>();
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        init_semaphore(kv_b, 0, 1);
        init_semaphore(q_b, 0, 1);
        init_semaphore(o_b, 0, 1);
        tma::expect_bytes(kv_b, sizeof(k_smem) + sizeof(v_smem));
        tma::expect_bytes(q_b, sizeof(q_smem));
        tma::expect_bytes(o_b, sizeof(do_smem));
        coord<k_tile> k_tile_idx = {batch_idx, kv_tile64_idx, head_idx, 0};
        coord<v_tile> v_tile_idx = {batch_idx, kv_tile64_idx, head_idx, 0};
        coord<q_tile> q_tile_idx_coord = {batch_idx, q_block_idx, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem, g.k, k_tile_idx, kv_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem, g.v, v_tile_idx, kv_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem, g.q, q_tile_idx_coord, q_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem, g.dout, do_tile_idx, o_b);
    }
    __syncthreads();
    wait(kv_b, 0);
    wait(q_b, 0);
    wait(o_b, 0);
    auto k_subtile = k_smem.template subtile<kRefTileM, C::Dqk>({warp, 0});
    auto v_subtile = v_smem.template subtile<kRefTileM, C::Dvo>({warp, 0});
    warp::load(k_reg, k_subtile);
    warp::load(v_reg, v_subtile);
    #pragma unroll
    for (int q_subtile = 0; q_subtile < C::QSubtiles; ++q_subtile) {
        const int q_tile_idx = q_block_idx * C::QSubtiles + q_subtile;
        auto q_subtile_smem = q_smem.template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
        auto do_subtile_smem = do_smem.template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
        warp::load(q_reg, q_subtile_smem);
        warp::load(do_reg, do_subtile_smem);
        warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
        warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
        repair_dkdv_step<true, REPAIR_DV, C>(
            dk_accum,
            dv_accum,
            q_reg,
            k_reg,
            v_reg,
            do_reg,
            lse_vec,
            dpsum_vec,
            g.scale,
            g.scale_log2e,
            q_tile_idx,
            kv_subtile_idx,
            g.seq_len,
            false
        );
    }
    __syncthreads();
    rt_fl<kRefTileM, C::Dqk> dk_existing;
    warp::load<dim::DEPTH>(dk_existing, g.dk_full, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::add(dk_accum, dk_accum, dk_existing);
    warp::store<dim::DEPTH>(g.dk_full, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    if constexpr (REPAIR_DV) {
        rt_fl<kRefTileM, C::Dvo> dv_existing;
        warp::load<dim::DEPTH>(dv_existing, g.dv, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::add(dv_accum, dv_accum, dv_existing);
        warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    }
}

template <typename C>
__global__ __launch_bounds__(C::QSubtiles * kWarpThreads, 1)
void detail::causal_dq_diagonal_patch_kernel(const __grid_constant__ main_globals<C> g) {
    using q_tile = typename main_globals<C>::q_tile;
    using k_tile = typename main_globals<C>::k_tile;
    using v_tile = typename main_globals<C>::v_tile;
    using do_tile = typename main_globals<C>::do_tile;
    __shared__ alignas(1024) q_tile q_smem;
    __shared__ alignas(1024) do_tile do_smem;
    __shared__ alignas(1024) k_tile k_smem;
    __shared__ alignas(1024) v_tile v_smem;
    __shared__ __align__(16) kittens::semaphore kv_b;
    __shared__ __align__(16) kittens::semaphore q_b;
    __shared__ __align__(16) kittens::semaphore o_b;

    const int warp = threadIdx.x >> 5;
    if (warp >= C::QSubtiles) {
        return;
    }

    const int q_block64_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int q_tile_idx = q_block64_idx * C::QSubtiles + warp;
    const int kv_subtile_idx = q_tile_idx;

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_vec, dpsum_vec;
    rt_fl<kRefTileM, C::Dqk> dq_partial;
    rt_fl<kRefTileM, 64> dq0, dq1, dq2, dq_existing;

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        g.k.template prefetch_tma<k_tile, dim::DEPTH>();
        g.v.template prefetch_tma<v_tile, dim::DEPTH>();
        init_semaphore(kv_b, 0, 1);
        init_semaphore(q_b, 0, 1);
        init_semaphore(o_b, 0, 1);
        tma::expect_bytes(q_b, sizeof(q_smem));
        tma::expect_bytes(o_b, sizeof(do_smem));
        tma::expect_bytes(kv_b, sizeof(k_smem) + sizeof(v_smem));
        coord<q_tile> q_tile_idx_coord = {batch_idx, q_block64_idx, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, q_block64_idx, head_idx, 0};
        coord<k_tile> k_tile_idx = {batch_idx, q_block64_idx, head_idx, 0};
        coord<v_tile> v_tile_idx = {batch_idx, q_block64_idx, head_idx, 0};
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem, g.q, q_tile_idx_coord, q_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem, g.dout, do_tile_idx, o_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem, g.k, k_tile_idx, kv_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem, g.v, v_tile_idx, kv_b);
    }
    __syncthreads();
    wait(q_b, 0);
    wait(o_b, 0);
    wait(kv_b, 0);

    auto q_subtile_smem = q_smem.template subtile<kRefTileM, C::Dqk>({warp, 0});
    auto do_subtile_smem = do_smem.template subtile<kRefTileM, C::Dvo>({warp, 0});
    auto k_subtile_smem = k_smem.template subtile<kRefTileM, C::Dqk>({warp, 0});
    auto v_subtile_smem = v_smem.template subtile<kRefTileM, C::Dvo>({warp, 0});

    warp::load(q_reg, q_subtile_smem);
    warp::load(do_reg, do_subtile_smem);
    warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
    warp::load(k_reg, k_subtile_smem);
    warp::load(v_reg, v_subtile_smem);
    repair_dq_step<true, C>(
        dq_partial,
        q_reg,
        k_reg,
        v_reg,
        do_reg,
        lse_vec,
        dpsum_vec,
        g.scale,
        g.scale_log2e,
        q_tile_idx,
        kv_subtile_idx,
        g.seq_len,
        false
    );
    split_three_chunks(dq_partial, dq0, dq1, dq2);

    warp::load(dq_existing, g.dq0, {batch_idx, q_tile_idx, head_idx, 0});
    warp::add(dq0, dq0, dq_existing);
    warp::store<dim::DEPTH>(g.dq0, dq0, {batch_idx, q_tile_idx, head_idx, 0});

    warp::load(dq_existing, g.dq1, {batch_idx, q_tile_idx, head_idx, 1});
    warp::add(dq1, dq1, dq_existing);
    warp::store<dim::DEPTH>(g.dq1, dq1, {batch_idx, q_tile_idx, head_idx, 1});

    warp::load(dq_existing, g.dq2, {batch_idx, q_tile_idx, head_idx, 2});
    warp::add(dq2, dq2, dq_existing);
    warp::store<dim::DEPTH>(g.dq2, dq2, {batch_idx, q_tile_idx, head_idx, 2});
}

template <typename C>
inline void launch_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &q_fix,
    at::Tensor &k_fix,
    at::Tensor &v_fix,
    at::Tensor &dout_fix,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &dqacc0,
    at::Tensor &dqacc1,
    at::Tensor &dqacc2,
    at::Tensor *dq_patch_bhsd,
    bool causal,
    float scale,
    bool deterministic,
    bool apply_causal_patches = true,
    int causal_q_start_offset_blocks = 1,
    bool full_causal_patch_coverage = false,
    bool use_exact_bulk_math = false
) {
    TORCH_CHECK(!deterministic, "CuTe16 hot mode not implemented yet; current stage only supports deterministic=False");

    using G = main_globals<C>;
    const int total_ctas = static_cast<int>(q.size(1) / (C::TileRows * C::ConsumerWarpgroups));
    const int scratch_rows = static_cast<int>(q.size(1) * C::ClusterSize);
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::q_fix_gl>(q_fix),
        kittens::py::tensor_to_gl<typename G::k_fix_gl>(k_fix),
        kittens::py::tensor_to_gl<typename G::v_fix_gl>(v_fix),
        kittens::py::tensor_to_gl<typename G::do_fix_gl>(dout_fix),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dqacc0.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            64
        ),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dqacc1.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            64
        ),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dqacc2.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            64
        ),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dk_full_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk0_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk1_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk2_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        causal_q_start_offset_blocks,
        full_causal_patch_coverage ? 1 : 0,
        use_exact_bulk_math ? 1 : 0,
    };

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        detail::launch_main<true, C>(g, total_ctas, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    } else {
        detail::launch_main<false, C>(g, total_ctas, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    if constexpr (!kUseDirectDQ) {
        detail::launch_reduce<C>(g, static_cast<int>(q.size(1) / C::TileRows), static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream, causal);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    if (causal && apply_causal_patches) {
        const int total_kv_tiles64 = static_cast<int>(q.size(1) / C::TileRows);
        const int causal_full_repair_tiles64 = g.full_causal_patch_coverage ? total_kv_tiles64 : (total_kv_tiles64 < 4 ? total_kv_tiles64 : 4);
        detail::launch_causal_first_tile_patch<C>(g, causal_full_repair_tiles64, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
        CHECK_CUDA_ERROR(cudaGetLastError());
        const int causal_dk_only_end_tile64 = total_kv_tiles64 < 29 ? total_kv_tiles64 : 29;
        if (!g.full_causal_patch_coverage && causal_full_repair_tiles64 < causal_dk_only_end_tile64) {
            detail::launch_causal_dk_only_patch<C>(
                g,
                causal_dk_only_end_tile64 - causal_full_repair_tiles64,
                static_cast<int>(q.size(2)),
                static_cast<int>(q.size(0)),
                stream,
                causal_full_repair_tiles64
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
        detail::launch_causal_dq_diagonal_patch<C>(g, 1, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

template <typename C>
inline void launch_backward_dkdv_only(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &q_fix,
    at::Tensor &k_fix,
    at::Tensor &v_fix,
    at::Tensor &dout_fix,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic,
    bool apply_causal_patches = true,
    int causal_q_start_offset_blocks = 1,
    bool full_causal_patch_coverage = false,
    bool use_exact_bulk_math = false
) {
    TORCH_CHECK(!deterministic, "CuTe16 hot mode not implemented yet; current stage only supports deterministic=False");

    using G = main_globals<C>;
    const int total_ctas = static_cast<int>(q.size(1) / (C::TileRows * C::ConsumerWarpgroups));
    const int scratch_rows = static_cast<int>(q.size(1) * C::ClusterSize);
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::q_fix_gl>(q_fix),
        kittens::py::tensor_to_gl<typename G::k_fix_gl>(k_fix),
        kittens::py::tensor_to_gl<typename G::v_fix_gl>(v_fix),
        kittens::py::tensor_to_gl<typename G::do_fix_gl>(dout_fix),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dq.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            64
        ),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dq.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            64
        ),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dq.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            64
        ),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dk_full_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk0_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk1_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk2_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        causal_q_start_offset_blocks,
        full_causal_patch_coverage ? 1 : 0,
        use_exact_bulk_math ? 1 : 0,
    };

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        detail::launch_dkdv_only<true, C>(g, total_ctas, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    } else {
        detail::launch_dkdv_only<false, C>(g, total_ctas, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    if (causal && apply_causal_patches) {
        const int total_kv_tiles64 = static_cast<int>(q.size(1) / C::TileRows);
        const int causal_full_repair_tiles64 = g.full_causal_patch_coverage ? total_kv_tiles64 : (total_kv_tiles64 < 4 ? total_kv_tiles64 : 4);
        detail::launch_causal_first_tile_patch<C>(g, causal_full_repair_tiles64, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
        CHECK_CUDA_ERROR(cudaGetLastError());
        const int causal_dk_only_end_tile64 = total_kv_tiles64 < 29 ? total_kv_tiles64 : 29;
        if (!g.full_causal_patch_coverage && causal_full_repair_tiles64 < causal_dk_only_end_tile64) {
            detail::launch_causal_dk_only_patch<C>(
                g,
                causal_dk_only_end_tile64 - causal_full_repair_tiles64,
                static_cast<int>(q.size(2)),
                static_cast<int>(q.size(0)),
                stream,
                causal_full_repair_tiles64
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
    }
}

}  // namespace tkfa4::bwd_cute16_kernel

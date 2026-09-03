#pragma once

#include "b300_common.cuh"

namespace tkfa4::bwd_hot {

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
struct config {
    static_assert(_Mb == kForwardTileM, "Exact B300 hot backward requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 hot backward requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 hot backward requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 hot backward requires Dvo=128");
    static_assert(_ClusterSize == 2, "Exact B300 hot backward requires ClusterSize=2");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = _ClusterSize;

    static constexpr int TileRows = Nb / 2;
    static constexpr int QSubtiles = TileRows / kRefTileM;
    static constexpr int ConsumerWarpgroups = 2;
    static constexpr int ComputeWarps = ConsumerWarpgroups * WARPGROUP_WARPS;
    static constexpr int ReduceWarps = QSubtiles;
    static constexpr int DkdvBlockThreads = (ComputeWarps + 1) * kWarpThreads;
    static constexpr int DqBlockThreads = (ComputeWarps + 1) * kWarpThreads;
    static constexpr int ReduceBlockThreads = ReduceWarps * kWarpThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <typename C>
struct dkdv_globals {
    using q_tile = st_bf<C::TileRows, C::Dqk>;
    using k_tile = st_bf<C::TileRows, C::Dqk>;
    using v_tile = st_bf<C::TileRows, C::Dvo>;
    using do_tile = st_bf<C::TileRows, C::Dvo>;
    using dk0_tile = st_fl<kRefTileM, 64>;
    using dk1_tile = st_fl<kRefTileM, 64>;
    using dk2_tile = st_fl<kRefTileM, 64>;
    using dq_tile = st_fl<kRefTileM, 64>;
    using dv_tile = st_fl<kRefTileM, C::Dvo>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using do_gl = gl<bf16, -1, -1, -1, -1, do_tile>;
    using dk0_gl = gl<float, -1, -1, -1, -1, dk0_tile>;
    using dk1_gl = gl<float, -1, -1, -1, -1, dk1_tile>;
    using dk2_gl = gl<float, -1, -1, -1, -1, dk2_tile>;
    using dq_gl = gl<float, -1, -1, -1, -1, dq_tile>;
    using dv_gl = gl<float, -1, -1, -1, -1, dv_tile>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk0_gl dk0;
    dk1_gl dk1;
    dk2_gl dk2;
    dq_gl dq;
    dv_gl dv;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

template <typename C>
struct dq_globals {
    using q_tile = st_bf<C::TileRows, C::Dqk>;
    using k_tile = st_bf<C::TileRows, C::Dqk>;
    using v_tile = st_bf<C::TileRows, C::Dvo>;
    using do_tile = st_bf<C::TileRows, C::Dvo>;
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using do_gl = gl<bf16, -1, -1, -1, -1, do_tile>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dqacc_gl dq_accum;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

namespace detail {

template <bool CAUSAL, typename Tile>
__device__ inline void apply_hot_mask(Tile &scores, int q_tile_base, int q_subtile, int kv_tile_idx) {
    if constexpr (!CAUSAL) {
        return;
    }
    constexpr float neg_inf = kittens::base_types::constants<float>::neg_infty();
    const int q_base = q_tile_base * config<kForwardTileM, kForwardTileN, kB300QKDim, kB300VDim, 2>::TileRows +
                       q_subtile * kRefTileM;
    const int k_base = kv_tile_idx * config<kForwardTileM, kForwardTileN, kB300QKDim, kB300VDim, 2>::TileRows;
    warp::apply(scores, scores, [=](int row, int col, float value) {
        if (k_base + col > q_base + row) {
            return neg_inf;
        }
        return value;
    });
}

template <typename FullTile, typename ChunkTile>
__device__ inline void stitch_four_chunks(
    FullTile &dst,
    const ChunkTile &chunk0,
    const ChunkTile &chunk1,
    const ChunkTile &chunk2,
    const ChunkTile &chunk3
) {
    static_assert(FullTile::height == ChunkTile::height);
    static_assert(FullTile::width == ChunkTile::width * 4);
    #pragma unroll
    for (int i = 0; i < ChunkTile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < ChunkTile::width; ++j) {
            dst.tiles[i][j] = chunk0.tiles[i][j];
            dst.tiles[i][j + ChunkTile::width] = chunk1.tiles[i][j];
            dst.tiles[i][j + 2 * ChunkTile::width] = chunk2.tiles[i][j];
            dst.tiles[i][j + 3 * ChunkTile::width] = chunk3.tiles[i][j];
        }
    }
}

template <typename FullTile, typename SubTile>
__device__ inline void insert_block(
    FullTile &dst,
    const SubTile &src,
    int col_block
) {
    static_assert(FullTile::height == SubTile::height);
    static_assert(FullTile::width % SubTile::width == 0);
    #pragma unroll
    for (int i = 0; i < SubTile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < SubTile::width; ++j) {
            dst.tiles[i][j + col_block * SubTile::width] = src.tiles[i][j];
        }
    }
}

template <typename ChunkTile, typename FullTile>
__device__ inline void split_four_chunks(
    ChunkTile &chunk0,
    ChunkTile &chunk1,
    ChunkTile &chunk2,
    ChunkTile &chunk3,
    const FullTile &src
) {
    static_assert(FullTile::height == ChunkTile::height);
    static_assert(FullTile::width == ChunkTile::width * 4);
    #pragma unroll
    for (int i = 0; i < ChunkTile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < ChunkTile::width; ++j) {
            chunk0.tiles[i][j] = src.tiles[i][j];
            chunk1.tiles[i][j] = src.tiles[i][j + ChunkTile::width];
            chunk2.tiles[i][j] = src.tiles[i][j + 2 * ChunkTile::width];
            chunk3.tiles[i][j] = src.tiles[i][j + 3 * ChunkTile::width];
        }
    }
}

template <typename C, typename DvAccum, typename ProbChunk, typename DoReg>
__device__ inline void accumulate_dv_from_prob_chunks(
    DvAccum &dv_accum,
    const ProbChunk &p0,
    const ProbChunk &p1,
    const ProbChunk &p2,
    const ProbChunk &p3,
    const DoReg &do_reg
) {
    rt_bf<kRefTileM, kRefTileN> p_bf;
    rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col;
    rt_bf<kRefTileM, C::Dvo, ducks::rt_layout::col> do_col;
    warp::swap_layout(do_col, do_reg);

    warp::copy(p_bf, p0);
    warp::swap_layout(p_col, p_bf);
    warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);

    warp::copy(p_bf, p1);
    warp::swap_layout(p_col, p_bf);
    warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);

    warp::copy(p_bf, p2);
    warp::swap_layout(p_col, p_bf);
    warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);

    warp::copy(p_bf, p3);
    warp::swap_layout(p_col, p_bf);
    warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);
}

template <bool CAUSAL, typename C, typename AttnTT, typename DkTT, typename DvTT>
__device__ inline void hot_compute_dkdv_loop(
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
    float scale,
    float scale_log2e,
    int phase,
    bool accumulate,
    int q_tile_idx,
    int kv_tile_idx
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
    using prob_tt = half_tt_bf<C::TileRows>;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
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
    apply_hot_mask<CAUSAL>(p_block_t, q_tile_idx, q_subtile, kv_tile_idx);
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
    const prob_tt probs_tmem = prob_tt{score_tt.addr};
    const prob_tt ds_tmem = prob_tt{dp_tt.addr};
    warpgroup::store_async(probs_tmem, p_block_t_mma);
    warpgroup::store_async(ds_tmem, ds_block_t_mma);
    tensor_store_wait();

    auto q_smem_0 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 0});
    auto q_smem_1 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 1});
    auto q_smem_2 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 2});
    if (accumulate) {
        warpgroup::mma_AB(dv_tt, probs_tmem, do_smem[0]);
        warpgroup::mma_AB(dk0_tt, ds_tmem, q_smem_0);
        warpgroup::mma_AB(dk1_tt, ds_tmem, q_smem_1);
        warpgroup::mma_AB(dk2_tt, ds_tmem, q_smem_2);
    } else {
        warpgroup::mm_AB(dv_tt, probs_tmem, do_smem[0]);
        warpgroup::mm_AB(dk0_tt, ds_tmem, q_smem_0);
        warpgroup::mm_AB(dk1_tt, ds_tmem, q_smem_1);
        warpgroup::mm_AB(dk2_tt, ds_tmem, q_smem_2);
    }
    group<8>::sync(10);
}

template <bool CAUSAL, typename C>
__device__ inline void hot_compute_exact_prob_ds_chunk(
    rt_fl<kRefTileM, kRefTileN> &p,
    rt_fl<kRefTileM, kRefTileN> &ds,
    const rt_bf<kRefTileM, C::Dqk> &q_reg,
    const rt_bf<kRefTileM, C::Dqk> &k_reg,
    const rt_bf<kRefTileM, C::Dvo> &v_reg,
    const rt_bf<kRefTileM, C::Dvo> &do_reg,
    const typename rt_fl<kRefTileM, C::TileRows>::col_vec &lse_log2_vec,
    const typename rt_fl<kRefTileM, C::TileRows>::col_vec &dpsum_vec,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_subtile_idx,
    int actual_seq_len
) {
    rt_fl<kRefTileM, kRefTileN> dp;

    if constexpr (!CAUSAL) {
        bwd_fa4::detail::reconstruct_probability_tile_dense<C>(p, q_reg, k_reg, lse_log2_vec, scale_log2e);
    } else {
        if (q_tile_idx > kv_subtile_idx) {
            bwd_fa4::detail::reconstruct_probability_tile_dense<C>(p, q_reg, k_reg, lse_log2_vec, scale_log2e);
        } else if (q_tile_idx < kv_subtile_idx) {
            warp::zero(p);
            warp::zero(ds);
            return;
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
                true
            );
        }
    }

    warp::zero(dp);
    warp::mma_ABt(dp, do_reg, v_reg, dp);
    warp::sub_row(dp, dp, dpsum_vec);
    warp::mul(ds, p, dp);
    warp::mul(ds, ds, scale);
}

template <bool CAUSAL, typename C, typename AttnTT, typename DkTT, typename DvTT>
__device__ inline void hot_compute_dkdv_only_exact_loop(
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
    auto &lse_log2_smem,
    auto &dpsum_smem,
    float scale,
    float scale_log2e,
    int phase,
    bool accumulate,
    int q_tile_idx,
    int kv_tile_idx
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
    using prob_tt = half_tt_bf<C::TileRows>;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
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
    apply_hot_mask<CAUSAL>(p_block_t, q_tile_idx, q_subtile, kv_tile_idx);
    warp::sub_row(p_block_t, p_block_t, lse_log2_vec);
    warp::exp2(p_block_t, p_block_t);

    wait(o_b, phase);
    warpgroup::mm_ABt(dp_tt, v_smem[consumer_idx], do_smem[0], dp_ready);
    wait(dp_ready, phase);
    warpgroup::load_async(dp_block_t, dp_tt);
    tensor_load_wait();
    warp::sub_row(dp_block_t, dp_block_t, dpsum_vec);
    warp::mul(ds_block_t, p_block_t, dp_block_t);
    warp::mul(ds_block_t, ds_block_t, scale);
    warp::copy(p_block_t_mma, p_block_t);
    warp::copy(ds_block_t_mma, ds_block_t);
    const prob_tt probs_tmem = prob_tt{score_tt.addr};
    const prob_tt ds_tmem = prob_tt{dp_tt.addr};
    warpgroup::store_async(probs_tmem, p_block_t_mma);
    warpgroup::store_async(ds_tmem, ds_block_t_mma);
    tensor_store_wait();

    auto q_smem_0 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 0});
    auto q_smem_1 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 1});
    auto q_smem_2 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 2});
    if (accumulate) {
        warpgroup::mma_AB(dv_tt, probs_tmem, do_smem[0]);
        warpgroup::mma_AB(dk0_tt, ds_tmem, q_smem_0);
        warpgroup::mma_AB(dk1_tt, ds_tmem, q_smem_1);
        warpgroup::mma_AB(dk2_tt, ds_tmem, q_smem_2);
    } else {
        warpgroup::mm_AB(dv_tt, probs_tmem, do_smem[0]);
        warpgroup::mm_AB(dk0_tt, ds_tmem, q_smem_0);
        warpgroup::mm_AB(dk1_tt, ds_tmem, q_smem_1);
        warpgroup::mm_AB(dk2_tt, ds_tmem, q_smem_2);
    }
    group<8>::sync(10);
}

template <bool CAUSAL, bool UseTmemDk, bool UseTmemDv, typename C, typename AttnTT, typename DkTT, typename DvTT>
__device__ inline void hot_compute_dkdv_tmem_exact_loop(
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
    rt_fl<kRefTileM, 64> &dk0_reg,
    rt_fl<kRefTileM, 64> &dk1_reg,
    rt_fl<kRefTileM, 64> &dk2_reg,
    rt_fl<kRefTileM, C::Dvo> &dv_reg,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &lse_log2_smem,
    auto &dpsum_smem,
    float scale,
    float scale_log2e,
    int phase,
    bool accumulate,
    int q_tile_idx,
    int kv_tile_idx,
    int actual_seq_len
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
    using prob_tt = half_tt_bf<C::TileRows>;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
    const int q_subtile = warpgroup::warpid();

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    rt_fl<kRefTileM, kRefTileN> p0, p1, p2, p3;
    rt_fl<kRefTileM, kRefTileN> ds0, ds1, ds2, ds3;
    stats_vec lse_log2_vec, dpsum_vec;

    const int q_subtile_idx = q_tile_idx * C::QSubtiles + q_subtile;

    wait(q_b, phase);
    wait(o_b, phase);

    auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
    auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
    warp::load(q_reg, q_subtile_smem);
    warp::load(do_reg, do_subtile_smem);
    warp::load(lse_log2_vec, lse_log2_smem[q_subtile]);
    warp::load(dpsum_vec, dpsum_smem[q_subtile]);

    auto k_subtile0 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({0, 0});
    auto k_subtile1 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({1, 0});
    auto k_subtile2 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({2, 0});
    auto k_subtile3 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({3, 0});
    auto v_subtile0 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({0, 0});
    auto v_subtile1 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({1, 0});
    auto v_subtile2 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({2, 0});
    auto v_subtile3 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({3, 0});

    warp::load(k_reg, k_subtile0);
    warp::load(v_reg, v_subtile0);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p0, ds0, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 0, actual_seq_len);

    warp::load(k_reg, k_subtile1);
    warp::load(v_reg, v_subtile1);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p1, ds1, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 1, actual_seq_len);

    warp::load(k_reg, k_subtile2);
    warp::load(v_reg, v_subtile2);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p2, ds2, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 2, actual_seq_len);

    warp::load(k_reg, k_subtile3);
    warp::load(v_reg, v_subtile3);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p3, ds3, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 3, actual_seq_len);

    warp::zero(p_block_t);
    warp::zero(ds_block_t);
    insert_block(p_block_t, p0, 0);
    insert_block(p_block_t, p1, 1);
    insert_block(p_block_t, p2, 2);
    insert_block(p_block_t, p3, 3);
    insert_block(ds_block_t, ds0, 0);
    insert_block(ds_block_t, ds1, 1);
    insert_block(ds_block_t, ds2, 2);
    insert_block(ds_block_t, ds3, 3);
    warp::zero(dp_block_t);
    warp::copy(p_block_t_mma, p_block_t);
    warp::copy(ds_block_t_mma, ds_block_t);

    const prob_tt probs_tmem = prob_tt{score_tt.addr};
    const prob_tt ds_tmem = prob_tt{dp_tt.addr};
    warpgroup::store_async(probs_tmem, p_block_t_mma);
    warpgroup::store_async(ds_tmem, ds_block_t_mma);
    tensor_store_wait();

    rt_fl<kRefTileM, kRefTileN> dv_p0, dv_p1, dv_p2, dv_p3;
    if constexpr (!UseTmemDv) {
        warpgroup::load_async(dp_block_t, probs_tmem);
        tensor_load_wait();
        split_four_chunks(dv_p0, dv_p1, dv_p2, dv_p3, dp_block_t);
    }

    auto q_smem_0 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 0});
    auto q_smem_1 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 1});
    auto q_smem_2 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 2});
    if (accumulate) {
        if constexpr (UseTmemDv) {
            warpgroup::mma_AB(dv_tt, probs_tmem, do_smem[0]);
        } else {
            accumulate_dv_from_prob_chunks<C>(dv_reg, dv_p0, dv_p1, dv_p2, dv_p3, do_reg);
        }
        if constexpr (UseTmemDk) {
            warpgroup::mma_AB(dk0_tt, ds_tmem, q_smem_0);
            warpgroup::mma_AB(dk1_tt, ds_tmem, q_smem_1);
            warpgroup::mma_AB(dk2_tt, ds_tmem, q_smem_2);
        } else {
            warpgroup::mma_AB(dk0_reg, ds_block_t_mma, q_smem_0);
            warpgroup::mma_AB(dk1_reg, ds_block_t_mma, q_smem_1);
            warpgroup::mma_AB(dk2_reg, ds_block_t_mma, q_smem_2);
        }
    } else {
        if constexpr (UseTmemDv) {
            warpgroup::mm_AB(dv_tt, probs_tmem, do_smem[0]);
        } else {
            accumulate_dv_from_prob_chunks<C>(dv_reg, dv_p0, dv_p1, dv_p2, dv_p3, do_reg);
        }
        if constexpr (UseTmemDk) {
            warpgroup::mm_AB(dk0_tt, ds_tmem, q_smem_0);
            warpgroup::mm_AB(dk1_tt, ds_tmem, q_smem_1);
            warpgroup::mm_AB(dk2_tt, ds_tmem, q_smem_2);
        } else {
            warpgroup::mm_AB(dk0_reg, ds_block_t_mma, q_smem_0);
            warpgroup::mm_AB(dk1_reg, ds_block_t_mma, q_smem_1);
            warpgroup::mm_AB(dk2_reg, ds_block_t_mma, q_smem_2);
        }
    }
    group<8>::sync(10);
}

template <bool CAUSAL, bool UseTmemDk, bool UseTmemDv, typename C, typename AttnTT, typename DkTT, typename DvTT, typename QGL, typename DOGL, typename StatsGL>
__device__ inline void hot_compute_dkdv_tmem_exact_loop_direct_qdo(
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
    rt_fl<kRefTileM, 64> &dk0_reg,
    rt_fl<kRefTileM, 64> &dk1_reg,
    rt_fl<kRefTileM, 64> &dk2_reg,
    rt_fl<kRefTileM, C::Dvo> &dv_reg,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    const QGL &q_gl,
    const DOGL &do_gl,
    const StatsGL &lse_gl,
    const StatsGL &dpsum_gl,
    int batch_idx,
    int head_idx,
    float scale,
    float scale_log2e,
    bool accumulate,
    int q_tile_idx,
    int kv_tile_idx,
    int actual_seq_len
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
    using prob_tt = half_tt_bf<C::TileRows>;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
    const int q_subtile = warpgroup::warpid();

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    rt_fl<kRefTileM, kRefTileN> p0, p1, p2, p3;
    rt_fl<kRefTileM, kRefTileN> ds0, ds1, ds2, ds3;
    stats_vec lse_log2_vec, dpsum_vec;

    const int q_subtile_idx = q_tile_idx * C::QSubtiles + q_subtile;
    warp::load<dim::DEPTH>(q_reg, q_gl, {batch_idx, q_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(do_reg, do_gl, {batch_idx, q_subtile_idx, head_idx, 0});
    warp::load(lse_log2_vec, lse_gl, {batch_idx, head_idx, 0, q_subtile_idx});
    warp::load(dpsum_vec, dpsum_gl, {batch_idx, head_idx, 0, q_subtile_idx});

    auto k_subtile0 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({0, 0});
    auto k_subtile1 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({1, 0});
    auto k_subtile2 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({2, 0});
    auto k_subtile3 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({3, 0});
    auto v_subtile0 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({0, 0});
    auto v_subtile1 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({1, 0});
    auto v_subtile2 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({2, 0});
    auto v_subtile3 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({3, 0});

    warp::load(k_reg, k_subtile0);
    warp::load(v_reg, v_subtile0);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p0, ds0, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 0, actual_seq_len);

    warp::load(k_reg, k_subtile1);
    warp::load(v_reg, v_subtile1);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p1, ds1, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 1, actual_seq_len);

    warp::load(k_reg, k_subtile2);
    warp::load(v_reg, v_subtile2);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p2, ds2, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 2, actual_seq_len);

    warp::load(k_reg, k_subtile3);
    warp::load(v_reg, v_subtile3);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p3, ds3, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 3, actual_seq_len);

    warp::zero(p_block_t);
    warp::zero(ds_block_t);
    insert_block(p_block_t, p0, 0);
    insert_block(p_block_t, p1, 1);
    insert_block(p_block_t, p2, 2);
    insert_block(p_block_t, p3, 3);
    insert_block(ds_block_t, ds0, 0);
    insert_block(ds_block_t, ds1, 1);
    insert_block(ds_block_t, ds2, 2);
    insert_block(ds_block_t, ds3, 3);
    warp::zero(dp_block_t);
    warp::copy(p_block_t_mma, p_block_t);
    warp::copy(ds_block_t_mma, ds_block_t);

    const prob_tt probs_tmem = prob_tt{score_tt.addr};
    const prob_tt ds_tmem = prob_tt{dp_tt.addr};
    warpgroup::store_async(probs_tmem, p_block_t_mma);
    warpgroup::store_async(ds_tmem, ds_block_t_mma);
    tensor_store_wait();

    auto q_smem_0 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 0});
    auto q_smem_1 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 1});
    auto q_smem_2 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 2});
    if (accumulate) {
        if constexpr (UseTmemDv) {
            warpgroup::mma_AB(dv_tt, probs_tmem, do_smem[0]);
        } else {
            accumulate_dv_from_prob_chunks<C>(dv_reg, p0, p1, p2, p3, do_reg);
        }
        if constexpr (UseTmemDk) {
            warpgroup::mma_AB(dk0_tt, ds_tmem, q_smem_0);
            warpgroup::mma_AB(dk1_tt, ds_tmem, q_smem_1);
            warpgroup::mma_AB(dk2_tt, ds_tmem, q_smem_2);
        } else {
            warpgroup::mma_AB(dk0_reg, ds_block_t_mma, q_smem_0);
            warpgroup::mma_AB(dk1_reg, ds_block_t_mma, q_smem_1);
            warpgroup::mma_AB(dk2_reg, ds_block_t_mma, q_smem_2);
        }
    } else {
        if constexpr (UseTmemDv) {
            warpgroup::mm_AB(dv_tt, probs_tmem, do_smem[0]);
        } else {
            accumulate_dv_from_prob_chunks<C>(dv_reg, p0, p1, p2, p3, do_reg);
        }
        if constexpr (UseTmemDk) {
            warpgroup::mm_AB(dk0_tt, ds_tmem, q_smem_0);
            warpgroup::mm_AB(dk1_tt, ds_tmem, q_smem_1);
            warpgroup::mm_AB(dk2_tt, ds_tmem, q_smem_2);
        } else {
            warpgroup::mm_AB(dk0_reg, ds_block_t_mma, q_smem_0);
            warpgroup::mm_AB(dk1_reg, ds_block_t_mma, q_smem_1);
            warpgroup::mm_AB(dk2_reg, ds_block_t_mma, q_smem_2);
        }
    }
    group<8>::sync(10);
}

template <bool CAUSAL, typename C>
__device__ inline void hot_compute_dkdv_reg_loop(
    kittens::semaphore &q_b,
    kittens::semaphore &o_b,
    kittens::semaphore &score_ready,
    kittens::semaphore &dp_ready,
    rt_fl<kRefTileM, C::TileRows> &p_block_t,
    rt_fl<kRefTileM, C::TileRows> &dp_block_t,
    rt_fl<kRefTileM, C::TileRows> &ds_block_t,
    rt_bf<kRefTileM, C::TileRows> &p_block_t_mma,
    rt_bf<kRefTileM, C::TileRows> &ds_block_t_mma,
    rt_fl<kRefTileM, 64> &dk0_reg,
    rt_fl<kRefTileM, 64> &dk1_reg,
    rt_fl<kRefTileM, 64> &dk2_reg,
    rt_fl<kRefTileM, C::Dvo> &dv_reg,
    half_tt_fl<C::TileRows> &score_tt,
    half_tt_fl<C::TileRows> &dp_tt,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_warp_smem,
    auto &lse_log2_smem,
    auto &dpsum_smem,
    float scale,
    float scale_log2e,
    int phase,
    bool accumulate,
    int q_tile_idx,
    int kv_tile_idx
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
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
    apply_hot_mask<CAUSAL>(p_block_t, q_tile_idx, q_subtile, kv_tile_idx);
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

    auto q_smem_0 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 0});
    auto q_smem_1 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 1});
    auto q_smem_2 = q_smem[0].template subtile<kForwardTileN / 2, 64>({0, 2});
    if (accumulate) {
        warpgroup::mma_AB(dv_reg, p_block_t_mma, do_smem[0]);
        warpgroup::mma_AB(dk0_reg, ds_block_t_mma, q_smem_0);
        warpgroup::mma_AB(dk1_reg, ds_block_t_mma, q_smem_1);
        warpgroup::mma_AB(dk2_reg, ds_block_t_mma, q_smem_2);
    } else {
        warpgroup::mm_AB(dv_reg, p_block_t_mma, do_smem[0]);
        warpgroup::mm_AB(dk0_reg, ds_block_t_mma, q_smem_0);
        warpgroup::mm_AB(dk1_reg, ds_block_t_mma, q_smem_1);
        warpgroup::mm_AB(dk2_reg, ds_block_t_mma, q_smem_2);
    }
    warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
    group<8>::sync(10);
}

template <bool CAUSAL, typename C, typename AttnTT>
__device__ inline void hot_compute_dq_loop(
    kittens::semaphore &q_b,
    kittens::semaphore &o_b,
    kittens::semaphore &score_ready,
    kittens::semaphore &dp_ready,
    rt_fl<kRefTileM, C::TileRows> &p_block_t,
    rt_fl<kRefTileM, C::TileRows> &dp_block_t,
    rt_fl<kRefTileM, C::TileRows> &ds_block_t,
    rt_bf<kRefTileM, C::TileRows> &ds_block_t_mma,
    AttnTT &score_tt,
    AttnTT &dp_tt,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_warp_smem,
    auto &lse_log2_smem,
    auto &dpsum_smem,
    float scale,
    float scale_log2e,
    int phase,
    int q_tile_idx,
    int kv_tile_idx
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
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
    apply_hot_mask<CAUSAL>(p_block_t, q_tile_idx, q_subtile, kv_tile_idx);
    warp::sub_row(p_block_t, p_block_t, lse_log2_vec);
    warp::exp2(p_block_t, p_block_t);

    wait(o_b, phase);
    warpgroup::mm_ABt(dp_tt, v_smem[consumer_idx], do_smem[0], dp_ready);
    wait(dp_ready, phase);
    warpgroup::load_async(dp_block_t, dp_tt);
    tensor_load_wait();
    warp::sub_row(dp_block_t, dp_block_t, dpsum_vec);
    warp::mul(ds_block_t, p_block_t, dp_block_t);
    warp::mul(ds_block_t, ds_block_t, scale);
    warp::copy(ds_block_t_mma, ds_block_t);
    warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
    group<8>::sync(10);
}

template <bool CAUSAL, typename C, typename AttnTT>
__device__ inline void hot_compute_dq_qmajor_loop(
    kittens::semaphore &q_b,
    kittens::semaphore &o_b,
    kittens::semaphore &score_ready,
    kittens::semaphore &dp_ready,
    rt_fl<kRefTileM, C::TileRows> &p_block_t,
    rt_fl<kRefTileM, C::TileRows> &dp_block_t,
    rt_fl<kRefTileM, C::TileRows> &ds_block_t,
    rt_bf<kRefTileM, C::TileRows> &ds_block_t_mma,
    AttnTT &score_tt,
    AttnTT &dp_tt,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_warp_smem,
    auto &lse_log2_smem,
    auto &dpsum_smem,
    float scale,
    float scale_log2e,
    int phase,
    int q_tile_idx,
    int kv_tile_idx,
    int input_stage = 0,
    int input_phase = -1
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
    const int q_subtile = warpgroup::warpid();

    stats_vec lse_log2_vec, dpsum_vec;
    warp::load(lse_log2_vec, lse_log2_smem[q_subtile]);
    warp::load(dpsum_vec, dpsum_smem[q_subtile]);

    const int load_phase = input_phase < 0 ? phase : input_phase;
    wait(q_b, load_phase);
    warpgroup::mm_ABt(score_tt, q_smem[input_stage], k_smem[consumer_idx], score_ready);
    wait(score_ready, phase);
    using attn_warp_tt = tt_fl<kRefTileM, C::TileRows>;
    const uint32_t warp_row_offset = (32 * q_subtile) << 16;
    const attn_warp_tt score_warp_tt{score_tt.addr + warp_row_offset};
    warp::load_async(p_block_t, score_warp_tt);
    tensor_load_wait();
    warp::mul(p_block_t, p_block_t, scale_log2e);
    apply_hot_mask<CAUSAL>(p_block_t, q_tile_idx, q_subtile, kv_tile_idx);
    warp::sub_row(p_block_t, p_block_t, lse_log2_vec);
    warp::exp2(p_block_t, p_block_t);

    wait(o_b, load_phase);
    warpgroup::mm_ABt(dp_tt, do_smem[input_stage], v_smem[consumer_idx], dp_ready);
    wait(dp_ready, phase);
    const attn_warp_tt dp_warp_tt{dp_tt.addr + warp_row_offset};
    warp::load_async(dp_block_t, dp_warp_tt);
    tensor_load_wait();
    warp::sub_row(dp_block_t, dp_block_t, dpsum_vec);
    warp::mul(ds_block_t, p_block_t, dp_block_t);
    warp::mul(ds_block_t, ds_block_t, scale);
    warp::copy(ds_block_t_mma, ds_block_t);
    warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
    group<8>::sync(10);
}

template <bool CAUSAL, typename C>
__device__ inline void hot_compute_dq_exact_loop(
    kittens::semaphore &q_b,
    kittens::semaphore &o_b,
    rt_fl<kRefTileM, C::TileRows> &ds_block_t,
    rt_bf<kRefTileM, C::TileRows> &ds_block_t_mma,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_warp_smem,
    auto &lse_log2_smem,
    auto &dpsum_smem,
    float scale,
    float scale_log2e,
    int phase,
    int q_tile_idx,
    int kv_tile_idx,
    int actual_seq_len
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
    const int q_subtile = warpgroup::warpid();
    const int q_subtile_idx = q_tile_idx * C::QSubtiles + q_subtile;

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    rt_fl<kRefTileM, kRefTileN> p0, p1, p2, p3;
    rt_fl<kRefTileM, kRefTileN> ds0, ds1, ds2, ds3;
    stats_vec lse_log2_vec, dpsum_vec;

    wait(q_b, phase);
    wait(o_b, phase);

    auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
    auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
    warp::load(q_reg, q_subtile_smem);
    warp::load(do_reg, do_subtile_smem);
    warp::load(lse_log2_vec, lse_log2_smem[q_subtile]);
    warp::load(dpsum_vec, dpsum_smem[q_subtile]);

    auto k_subtile0 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({0, 0});
    auto k_subtile1 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({1, 0});
    auto k_subtile2 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({2, 0});
    auto k_subtile3 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({3, 0});
    auto v_subtile0 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({0, 0});
    auto v_subtile1 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({1, 0});
    auto v_subtile2 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({2, 0});
    auto v_subtile3 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({3, 0});

    warp::load(k_reg, k_subtile0);
    warp::load(v_reg, v_subtile0);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p0, ds0, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 0, actual_seq_len);

    warp::load(k_reg, k_subtile1);
    warp::load(v_reg, v_subtile1);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p1, ds1, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 1, actual_seq_len);

    warp::load(k_reg, k_subtile2);
    warp::load(v_reg, v_subtile2);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p2, ds2, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 2, actual_seq_len);

    warp::load(k_reg, k_subtile3);
    warp::load(v_reg, v_subtile3);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p3, ds3, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 3, actual_seq_len);

    warp::zero(ds_block_t);
    insert_block(ds_block_t, ds0, 0);
    insert_block(ds_block_t, ds1, 1);
    insert_block(ds_block_t, ds2, 2);
    insert_block(ds_block_t, ds3, 3);
    warp::copy(ds_block_t_mma, ds_block_t);
    warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
    group<8>::sync(10);
}

template <bool CAUSAL, typename C>
__device__ inline void hot_overwrite_dq_exact_from_loaded_nosync(
    rt_fl<kRefTileM, C::TileRows> &ds_block_t,
    rt_bf<kRefTileM, C::TileRows> &ds_block_t_mma,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_warp_smem,
    auto &lse_log2_smem,
    auto &dpsum_smem,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_tile_idx,
    int actual_seq_len
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
    const int q_subtile = warpgroup::warpid();
    const int q_subtile_idx = q_tile_idx * C::QSubtiles + q_subtile;

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    rt_fl<kRefTileM, kRefTileN> p0, p1, p2, p3;
    rt_fl<kRefTileM, kRefTileN> ds0, ds1, ds2, ds3;
    stats_vec lse_log2_vec, dpsum_vec;

    auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
    auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
    warp::load(q_reg, q_subtile_smem);
    warp::load(do_reg, do_subtile_smem);
    warp::load(lse_log2_vec, lse_log2_smem[q_subtile]);
    warp::load(dpsum_vec, dpsum_smem[q_subtile]);

    auto k_subtile0 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({0, 0});
    auto k_subtile1 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({1, 0});
    auto k_subtile2 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({2, 0});
    auto k_subtile3 = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({3, 0});
    auto v_subtile0 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({0, 0});
    auto v_subtile1 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({1, 0});
    auto v_subtile2 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({2, 0});
    auto v_subtile3 = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({3, 0});

    warp::load(k_reg, k_subtile0);
    warp::load(v_reg, v_subtile0);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p0, ds0, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 0, actual_seq_len);

    warp::load(k_reg, k_subtile1);
    warp::load(v_reg, v_subtile1);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p1, ds1, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 1, actual_seq_len);

    warp::load(k_reg, k_subtile2);
    warp::load(v_reg, v_subtile2);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p2, ds2, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 2, actual_seq_len);

    warp::load(k_reg, k_subtile3);
    warp::load(v_reg, v_subtile3);
    hot_compute_exact_prob_ds_chunk<CAUSAL, C>(
        p3, ds3, q_reg, k_reg, v_reg, do_reg, lse_log2_vec, dpsum_vec,
        scale, scale_log2e, q_subtile_idx, kv_tile_idx * C::QSubtiles + 3, actual_seq_len);

    warp::zero(ds_block_t);
    insert_block(ds_block_t, ds0, 0);
    insert_block(ds_block_t, ds1, 1);
    insert_block(ds_block_t, ds2, 2);
    insert_block(ds_block_t, ds3, 3);
    warp::copy(ds_block_t_mma, ds_block_t);
    warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
}

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::DkdvBlockThreads, C::MinBlocksPerSm)
void dkdv_kernel(const __grid_constant__ dkdv_globals<C> g) {
    using q_tile = typename dkdv_globals<C>::q_tile;
    using k_tile = typename dkdv_globals<C>::k_tile;
    using v_tile = typename dkdv_globals<C>::v_tile;
    using do_tile = typename dkdv_globals<C>::do_tile;
    using stats_smem_tile = typename dkdv_globals<C>::stats_tile;
    using dq_tile = typename dkdv_globals<C>::dq_tile;
    using ds_warp_tile = st_bf<kRefTileM, C::TileRows>;
    using attn_tt = half_tt_fl<C::TileRows>;
    using dk_tt = half_tt_fl<64>;
    using dv_tt = half_tt_fl<C::Dvo>;

    struct shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        dq_tile dq_smem[C::QSubtiles];
        ds_warp_tile ds_warp_smem[C::ConsumerWarpgroups][WARPGROUP_WARPS];
        stats_smem_tile lse_log2_smem[C::QSubtiles];
        stats_smem_tile dpsum_smem[C::QSubtiles];
    };

    __shared__ alignas(1024) shared_storage smem;
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

    const int warp = kittens::warpid();
    const bool is_compute = warp < C::ComputeWarps;
    const bool is_load = warp == C::ComputeWarps;
    const int consumer_idx = is_compute ? (warp / kittens::WARPGROUP_WARPS) : -1;

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = cluster_ctarank();
    const int cluster_idx = static_cast<int>(blockIdx.x) / C::ClusterSize;
    const int num_k_blocks = g.seq_len / (C::TileRows * C::ConsumerWarpgroups);
    const int kv_block_idx = cluster_idx * C::ClusterSize + cluster_rank;
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::ConsumerWarpgroups;
    const int q_blocks = g.seq_len / C::TileRows;
    const int q_start_block = CAUSAL ? kv_tile_base : 0;
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;

    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dk_tt dk0_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk1_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk2_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dv_tt dv_accum_tt[C::ConsumerWarpgroups] = {dv_tt{0}, dv_tt{0}};

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile>();
        g.k.template prefetch_tma<k_tile>();
        g.v.template prefetch_tma<v_tile>();
        g.dout.template prefetch_tma<do_tile>();

        init_semaphore(kv_b, 0, 1);
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            init_semaphore(score_ready[w][0], 0, 1);
            init_semaphore(dp_ready[w][0], 0, 1);
            init_semaphore(kv_tmem_ready[w], 0, 1);
        }
    }
    __syncthreads();

    if (is_compute) {
        score_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
        dp_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, C::TileRows);
        dk0_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 2 * C::TileRows);
        dk1_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 3 * C::TileRows);
        dk2_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 4 * C::TileRows);
        dv_accum_tt[consumer_idx] = tm_alloc.template allocate<dv_tt>(consumer_idx, 5 * C::TileRows);
    }

    if (threadIdx.x == 0) {
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * C::ConsumerWarpgroups);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            coord<k_tile> tile_idx = {batch_idx, head_idx, kv_tile_base + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }
    }
    for (int q_block_idx = q_start_block; q_block_idx < q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * C::QSubtiles;
        if (is_load) {
            coord<q_tile> q_tile_idx = {batch_idx, head_idx, q_block_idx, 0};
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
            warp::tma::load_async(q_smem[0], g.q, q_tile_idx, q_b[0]);
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
            warp::tma::load_async(do_smem[0], g.dout, coord<do_tile>{batch_idx, head_idx, q_block_idx, 0}, o_b[0]);
            for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_stage_vec, dpsum_stage_vec;
                warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(lse_log2_smem[subtile], lse_stage_vec);
                warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[subtile], dpsum_stage_vec);
            }
        }
        __syncthreads();

        if (is_compute) {
            rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
            rt_bf<kRefTileM, C::TileRows> p_block_t_mma, ds_block_t_mma;
            const int phase = q_block_idx & 1;

            wait(kv_b, 0);
            hot_compute_dkdv_loop<CAUSAL, C>(
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
                g.scale,
                g.scale_log2e,
                phase,
                q_block_idx > q_start_block,
                q_block_idx,
                kv_tile_base + consumer_idx
            );
        }
        __syncthreads();

        if (is_compute && consumer_idx == 0) {
            const int dq_subtile_idx = warpgroup::warpid();
            const int q_tile_idx = q_tile_base + dq_subtile_idx;
            rt_bf<kRefTileM, C::TileRows> ds_local_reg;
            rt_bf<C::TileRows, 64> k_reg;
            rt_bf<C::TileRows, 64, ducks::rt_layout::col> k_col;
            rt_fl<kRefTileM, 64> dq_chunk;

            auto k0_0 = k_smem[0].template subtile<C::TileRows, 64>({0, 0});
            auto k0_1 = k_smem[0].template subtile<C::TileRows, 64>({0, 1});
            auto k0_2 = k_smem[0].template subtile<C::TileRows, 64>({0, 2});
            auto k1_0 = k_smem[1].template subtile<C::TileRows, 64>({0, 0});
            auto k1_1 = k_smem[1].template subtile<C::TileRows, 64>({0, 1});
            auto k1_2 = k_smem[1].template subtile<C::TileRows, 64>({0, 2});

            rt_bf<kRefTileM, C::TileRows> ds_peer_reg;
            warp::load(ds_local_reg, ds_warp_smem[0][dq_subtile_idx]);
            warp::load(ds_peer_reg, ds_warp_smem[1][dq_subtile_idx]);

            warp::zero(dq_chunk);
            warp::load(k_reg, k0_0);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
            warp::load(k_reg, k1_0);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
            warp::store(dq_smem[dq_subtile_idx], dq_chunk);
            __syncwarp();
            warp::tma::store_add_async(
                g.dq,
                dq_smem[dq_subtile_idx],
                {batch_idx, head_idx, q_tile_idx, 0}
            );
            warp::tma::store_async_read_wait();

            warp::zero(dq_chunk);
            warp::load(k_reg, k0_1);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
            warp::load(k_reg, k1_1);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
            warp::store(dq_smem[dq_subtile_idx], dq_chunk);
            __syncwarp();
            warp::tma::store_add_async(
                g.dq,
                dq_smem[dq_subtile_idx],
                {batch_idx, head_idx, q_tile_idx, 1}
            );
            warp::tma::store_async_read_wait();

            warp::zero(dq_chunk);
            warp::load(k_reg, k0_2);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
            warp::load(k_reg, k1_2);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
            warp::store(dq_smem[dq_subtile_idx], dq_chunk);
            __syncwarp();
            warp::tma::store_add_async(
                g.dq,
                dq_smem[dq_subtile_idx],
                {batch_idx, head_idx, q_tile_idx, 2}
            );
            warp::tma::store_async_read_wait();
        }
        __syncthreads();
    }

    if (is_compute) {
        rt_fl<kRefTileM, 64> dk0_reg;
        rt_fl<kRefTileM, 64> dk1_reg;
        rt_fl<kRefTileM, 64> dk2_reg;
        rt_fl<kRefTileM, C::Dvo> dv_reg;
        if (warpgroup::laneid() == 0) {
            tensor_commit<1>(kv_tmem_ready[consumer_idx]);
        }
        wait(kv_tmem_ready[consumer_idx], 0);
        warpgroup::load_async(dk0_reg, dk0_tt[consumer_idx]);
        warpgroup::load_async(dk1_reg, dk1_tt[consumer_idx]);
        warpgroup::load_async(dk2_reg, dk2_tt[consumer_idx]);
        warpgroup::load_async(dv_reg, dv_accum_tt[consumer_idx]);
        tensor_load_wait();
        const int kv_subtile_idx =
            (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + warpgroup::warpid();
        warp::store(g.dk0, dk0_reg, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::store(g.dk1, dk1_reg, {batch_idx, head_idx, kv_subtile_idx, 1});
        warp::store(g.dk2, dk2_reg, {batch_idx, head_idx, kv_subtile_idx, 2});
        warp::store(g.dv, dv_reg, {batch_idx, head_idx, kv_subtile_idx, 0});
    }
}

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::DqBlockThreads, C::MinBlocksPerSm)
void dq_kernel(const __grid_constant__ dq_globals<C> g) {
    using q_tile = typename dq_globals<C>::q_tile;
    using k_tile = typename dq_globals<C>::k_tile;
    using v_tile = typename dq_globals<C>::v_tile;
    using do_tile = typename dq_globals<C>::do_tile;
    using dqacc_tile = typename dq_globals<C>::dqacc_tile;
    using stats_smem_tile = typename dq_globals<C>::stats_tile;
    using ds_warp_tile = st_bf<kRefTileM, C::TileRows>;
    using attn_tt = half_tt_fl<C::TileRows>;

    struct shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        dqacc_tile dq_smem[C::QSubtiles];
        ds_warp_tile ds_warp_smem[C::ConsumerWarpgroups][WARPGROUP_WARPS];
        stats_smem_tile lse_log2_smem[C::QSubtiles];
        stats_smem_tile dpsum_smem[C::QSubtiles];
    };

    __shared__ alignas(1024) shared_storage smem;
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

    const int warp = kittens::warpid();
    constexpr int kLoadWarp = C::ComputeWarps;
    const bool is_compute = warp < C::ComputeWarps;
    const bool is_load = warp == kLoadWarp;
    const int consumer_idx = is_compute ? (warp / kittens::WARPGROUP_WARPS) : -1;

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = cluster_ctarank();
    const int cluster_idx = static_cast<int>(blockIdx.x) / C::ClusterSize;
    const int num_k_blocks = g.seq_len / (C::TileRows * C::ConsumerWarpgroups);
    const int kv_block_idx = cluster_idx * C::ClusterSize + cluster_rank;
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::ConsumerWarpgroups;
    const int q_blocks = g.seq_len / C::TileRows;
    const int q_start_block = CAUSAL ? kv_tile_base : 0;
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;

    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile>();
        g.k.template prefetch_tma<k_tile>();
        g.v.template prefetch_tma<v_tile>();
        g.dout.template prefetch_tma<do_tile>();

        init_semaphore(kv_b, 0, 1);
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            init_semaphore(score_ready[w][0], 0, 1);
            init_semaphore(dp_ready[w][0], 0, 1);
        }
    }
    __syncthreads();

    if (is_compute) {
        score_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
        dp_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, C::TileRows);
    }

    if (threadIdx.x == 0) {
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * C::ConsumerWarpgroups);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            coord<k_tile> tile_idx = {batch_idx, head_idx, kv_tile_base + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }
    }
    __syncthreads();

    for (int q_block_idx = q_start_block; q_block_idx < q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * C::QSubtiles;
        if (is_load) {
            coord<q_tile> q_tile_idx = {batch_idx, head_idx, q_block_idx, 0};
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
            warp::tma::load_async(q_smem[0], g.q, q_tile_idx, q_b[0]);
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
            warp::tma::load_async(do_smem[0], g.dout, q_tile_idx, o_b[0]);

            for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_stage_vec, dpsum_stage_vec;
                warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(lse_log2_smem[subtile], lse_stage_vec);
                warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[subtile], dpsum_stage_vec);
            }
        }
        __syncthreads();

        if (is_compute) {
            rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
            rt_bf<kRefTileM, C::TileRows> ds_block_t_mma;
            const int phase = q_block_idx & 1;

            wait(kv_b, 0);
            hot_compute_dq_loop<CAUSAL, C>(
                q_b[0],
                o_b[0],
                score_ready[consumer_idx][0],
                dp_ready[consumer_idx][0],
                p_block_t,
                dp_block_t,
                ds_block_t,
                ds_block_t_mma,
                score_tt[consumer_idx],
                dp_tt[consumer_idx],
                q_smem,
                k_smem,
                v_smem,
                do_smem,
                ds_warp_smem,
                lse_log2_smem,
                dpsum_smem,
                g.scale,
                g.scale_log2e,
                phase,
                q_block_idx,
                kv_tile_base + consumer_idx
            );
        }
        __syncthreads();

        if (is_compute && consumer_idx == 0) {
            const int dq_subtile_idx = warpgroup::warpid();
            const int q_tile_idx = q_tile_base + dq_subtile_idx;
            const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
            const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
            const int scratch_tile_idx =
                ((q_tile_group_idx * C::ClusterSize) + cluster_rank) * kQSubtilesPerTile + q_subtile_in_group;

            rt_fl<kRefTileM, C::Dqk> dq_partial;
            rt_bf<kRefTileM, C::TileRows> ds_local_reg;
            rt_bf<kRefTileM, C::TileRows> ds_peer_reg;
            rt_bf<C::TileRows, C::Dqk> k_local_reg;
            rt_bf<C::TileRows, C::Dqk, ducks::rt_layout::col> k_local_col;

            warp::zero(dq_partial);
            warp::load(ds_local_reg, ds_warp_smem[0][dq_subtile_idx]);
            warp::load(k_local_reg, k_smem[0]);
            warp::swap_layout(k_local_col, k_local_reg);
            warp::mma_AB(dq_partial, ds_local_reg, k_local_col, dq_partial);

            warp::load(ds_peer_reg, ds_warp_smem[1][dq_subtile_idx]);
            warp::load(k_local_reg, k_smem[1]);
            warp::swap_layout(k_local_col, k_local_reg);
            warp::mma_AB(dq_partial, ds_peer_reg, k_local_col, dq_partial);
            warp::store(dq_smem[dq_subtile_idx], dq_partial);
            __syncwarp();
            warp::tma::store_add_async(g.dq_accum, dq_smem[dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
            warp::tma::store_async_wait();
        }
        __syncthreads();
    }
}

template <bool CAUSAL, typename C>
inline void launch_hot_dkdv(
    const dkdv_globals<C> &g,
    int total_ctas,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(total_ctas, heads, batch_size),
        dim3(C::DkdvBlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if constexpr (CAUSAL) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, dkdv_kernel<true, C>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, dkdv_kernel<false, C>, g));
    }
}

template <bool CAUSAL, typename C>
inline void launch_hot_dq(
    const dq_globals<C> &g,
    int total_ctas,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(total_ctas, heads, batch_size),
        dim3(C::DqBlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if constexpr (CAUSAL) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, dq_kernel<true, C>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, dq_kernel<false, C>, g));
    }
}

}  // namespace detail

template <typename C>
inline void launch_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale
) {
    using DkdvG = dkdv_globals<C>;
    const int total_ctas = static_cast<int>((q.size(2) / (C::TileRows * C::ConsumerWarpgroups)) * C::ClusterSize);

    DkdvG dkdv_g{
        kittens::py::tensor_to_gl<typename DkdvG::q_gl>(q),
        kittens::py::tensor_to_gl<typename DkdvG::k_gl>(k),
        kittens::py::tensor_to_gl<typename DkdvG::v_gl>(v),
        kittens::py::tensor_to_gl<typename DkdvG::do_gl>(dout),
        kittens::py::tensor_to_gl<typename DkdvG::dk0_gl>(dk),
        kittens::py::tensor_to_gl<typename DkdvG::dk1_gl>(dk),
        kittens::py::tensor_to_gl<typename DkdvG::dk2_gl>(dk),
        ::kittens::make_gl<typename DkdvG::dq_gl>(
            reinterpret_cast<uint64_t>(dq.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(1)),
            static_cast<int>(q.size(2)),
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename DkdvG::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename DkdvG::stats_gl>(lse_log2, q.size(0), q.size(1), 1, q.size(2)),
        kittens::py::tensor_to_gl<typename DkdvG::stats_gl>(dpsum, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(2)),
    };

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        detail::launch_hot_dkdv<true, C>(
            dkdv_g,
            total_ctas,
            static_cast<int>(q.size(1)),
            static_cast<int>(q.size(0)),
            stream
        );
    } else {
        detail::launch_hot_dkdv<false, C>(
            dkdv_g,
            total_ctas,
            static_cast<int>(q.size(1)),
            static_cast<int>(q.size(0)),
            stream
        );
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd_hot

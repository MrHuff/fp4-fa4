#pragma once

#include "b300_bwd_fa4.cuh"
#include "b300_bwd_hot.cuh"
#include "b300_common.cuh"

namespace tkfa4::bwd_cute {

inline constexpr bool kUseRegDkDvHybrid = false;

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
struct config {
    static_assert(_Mb == kForwardTileM, "Exact B300 CuTe-style backward requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 CuTe-style backward requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 CuTe-style backward requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 CuTe-style backward requires Dvo=128");
    static_assert(_ClusterSize == 2, "Exact B300 CuTe-style backward requires ClusterSize=2");

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
    using dk0_tile = st_fl<kRefTileM, 64>;
    using dk1_tile = st_fl<kRefTileM, 64>;
    using dk2_tile = st_fl<kRefTileM, 64>;
    using dk_full_tile = st_fl<C::TileRows, 64>;
    using dv_tile = st_fl<kRefTileM, C::Dvo>;
    using dv_full_tile = st_fl<C::TileRows, C::Dvo>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<k_tile, dim::DEPTH>>;
    using v_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tile, dim::DEPTH>>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dq_chunk_tile>;
    using dq_out_gl = gl<float, -1, -1, -1, -1, dq_chunk_tile>;
    using dk0_gl = gl<float, -1, -1, -1, -1, dk0_tile>;
    using dk1_gl = gl<float, -1, -1, -1, -1, dk1_tile>;
    using dk2_gl = gl<float, -1, -1, -1, -1, dk2_tile>;
    using dk_full_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dk_full_tile, dim::DEPTH>>;
    using dv_gl = gl<float, -1, -1, -1, -1, dv_tile>;
    using dv_full_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dv_full_tile, dim::DEPTH>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dqacc_gl dqacc0;
    dqacc_gl dqacc1;
    dqacc_gl dqacc2;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    dk0_gl dk0;
    dk1_gl dk1;
    dk2_gl dk2;
    dk_full_gl dk0_full;
    dk_full_gl dk1_full;
    dk_full_gl dk2_full;
    dv_gl dv;
    dv_full_gl dv_full;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
    int *dq_semaphore;
    int heads;
    int q_tiles;
    int cluster_groups;
    int deterministic;
};

namespace detail {

template <typename C>
__device__ inline int dq_semaphore_index(
    const main_globals<C> &g,
    int batch_idx,
    int head_idx,
    int q_tile_idx,
    int cluster_rank
) {
    return (((batch_idx * g.heads + head_idx) * g.q_tiles + q_tile_idx) * C::ClusterSize) + cluster_rank;
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

template <typename C, typename RT>
__device__ inline void apply_reference_mask_block(
    RT &scores,
    int q_block_idx,
    int q_subtile_idx,
    int kv_tile_idx,
    int actual_seq_len,
    bool causal
) {
    constexpr float neg_inf = kittens::base_types::constants<float>::neg_infty();
    const int q_base = q_block_idx * C::TileRows + q_subtile_idx * kRefTileM;
    const int k_base = kv_tile_idx * C::TileRows;
    warp::apply(scores, scores, [=](int row, int col, float value) {
        const int q_idx = q_base + row;
        const int k_idx = k_base + col;
        if (q_idx >= actual_seq_len || k_idx >= actual_seq_len) {
            return neg_inf;
        }
        if (causal && k_idx > q_idx) {
            return neg_inf;
        }
        return value;
    });
}

template <bool CAUSAL, typename C, typename AttnTT, typename DkTT, typename DvTT>
__device__ inline void cute_compute_dkdv_loop(
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
    int kv_tile_idx,
    int actual_seq_len
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
    using prob_tt = half_tt_bf<C::TileRows>;

    const int q_subtile = warpgroup::warpid();
    const int q_tile16_idx = q_tile_idx * C::QSubtiles + q_subtile;

    stats_vec lse_log2_vec, dpsum_vec;
    warp::load(lse_log2_vec, lse_log2_smem[q_subtile]);
    warp::load(dpsum_vec, dpsum_smem[q_subtile]);

    wait(q_b, phase);
    wait(o_b, phase);
    {
        rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
        rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
        auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
        auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
        warp::load(q_reg, q_subtile_smem);
        warp::load(do_reg, do_subtile_smem);

        warp::zero(p_block_t);
        warp::zero(dp_block_t);
        warp::zero(ds_block_t);
        #pragma unroll
        for (int kv_subtile = 0; kv_subtile < C::QSubtiles; ++kv_subtile) {
            const int kv_tile16_idx = kv_tile_idx * C::QSubtiles + kv_subtile;
            const bool dense_unmasked = !CAUSAL || (kv_tile16_idx < q_tile16_idx);
            rt_fl<kRefTileM, kRefTileN> p_sub, dp_sub, ds_sub;
            auto k_subtile_smem = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({kv_subtile, 0});
            auto v_subtile_smem = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({kv_subtile, 0});
            warp::load(k_reg, k_subtile_smem);
            warp::load(v_reg, v_subtile_smem);

            if (dense_unmasked) {
                bwd_fa4::detail::reconstruct_probability_tile_dense<C>(
                    p_sub,
                    q_reg,
                    k_reg,
                    lse_log2_vec,
                    scale_log2e
                );
            } else {
                bwd_fa4::detail::reconstruct_probability_tile<C>(
                    p_sub,
                    q_reg,
                    k_reg,
                    lse_log2_vec,
                    scale_log2e,
                    q_tile16_idx,
                    kv_tile16_idx,
                    actual_seq_len,
                    CAUSAL
                );
            }

            warp::zero(dp_sub);
            warp::mma_ABt(dp_sub, do_reg, v_reg, dp_sub);
            warp::sub_row(dp_sub, dp_sub, dpsum_vec);
            warp::mul(ds_sub, p_sub, dp_sub);
            warp::mul(ds_sub, ds_sub, scale);

            insert_block(p_block_t, p_sub, kv_subtile);
            insert_block(dp_block_t, dp_sub, kv_subtile);
            insert_block(ds_block_t, ds_sub, kv_subtile);
        }
        warp::copy(p_block_t_mma, p_block_t);
        warp::copy(ds_block_t_mma, ds_block_t);
    }

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
    warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
    group<8>::sync(10);
}

template <bool CAUSAL, typename C>
__device__ inline void cute_compute_dkdv_reg_loop(
    kittens::semaphore &q_b,
    kittens::semaphore &o_b,
    rt_fl<kRefTileM, C::TileRows> &p_block_t,
    rt_fl<kRefTileM, C::TileRows> &dp_block_t,
    rt_fl<kRefTileM, C::TileRows> &ds_block_t,
    rt_bf<kRefTileM, C::TileRows> &p_block_t_mma,
    rt_bf<kRefTileM, C::TileRows> &ds_block_t_mma,
    rt_fl<kRefTileM, 64> &dk0_reg,
    rt_fl<kRefTileM, 64> &dk1_reg,
    rt_fl<kRefTileM, 64> &dk2_reg,
    rt_fl<kRefTileM, C::Dvo> &dv_reg,
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
    int kv_tile_idx,
    int actual_seq_len
) {
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
    (void)accumulate;

    const int q_subtile = warpgroup::warpid();
    const int q_tile16_idx = q_tile_idx * C::QSubtiles + q_subtile;

    stats_vec lse_log2_vec, dpsum_vec;
    warp::load(lse_log2_vec, lse_log2_smem[q_subtile]);
    warp::load(dpsum_vec, dpsum_smem[q_subtile]);

    wait(q_b, phase);
    wait(o_b, phase);
    {
        rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
        rt_bf<kRefTileM, 64> q_chunk;
        rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
        rt_bf<kRefTileM, kRefTileN> p_bf, ds_bf;
        rt_bf<kRefTileM, 64, ducks::rt_layout::col> q_chunk_col;
        rt_bf<kRefTileM, C::Dvo, ducks::rt_layout::col> do_col;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col, ds_col;
        auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
        auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
        warp::load(q_reg, q_subtile_smem);
        warp::load(do_reg, do_subtile_smem);
        warp::swap_layout(do_col, do_reg);

        warp::zero(p_block_t);
        warp::zero(dp_block_t);
        warp::zero(ds_block_t);
        #pragma unroll
        for (int kv_subtile = 0; kv_subtile < C::QSubtiles; ++kv_subtile) {
            const int kv_tile16_idx = kv_tile_idx * C::QSubtiles + kv_subtile;
            const bool dense_unmasked = !CAUSAL || (kv_tile16_idx < q_tile16_idx);
            rt_fl<kRefTileM, kRefTileN> p_sub, dp_sub, ds_sub;
            auto k_subtile_smem = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({kv_subtile, 0});
            auto v_subtile_smem = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({kv_subtile, 0});
            warp::load(k_reg, k_subtile_smem);
            warp::load(v_reg, v_subtile_smem);

            if (dense_unmasked) {
                bwd_fa4::detail::reconstruct_probability_tile_dense<C>(
                    p_sub,
                    q_reg,
                    k_reg,
                    lse_log2_vec,
                    scale_log2e
                );
            } else {
                bwd_fa4::detail::reconstruct_probability_tile<C>(
                    p_sub,
                    q_reg,
                    k_reg,
                    lse_log2_vec,
                    scale_log2e,
                    q_tile16_idx,
                    kv_tile16_idx,
                    actual_seq_len,
                    CAUSAL
                );
            }

            warp::copy(p_bf, p_sub);
            warp::swap_layout(p_col, p_bf);
            warp::mma_AtB(dv_reg, p_col, do_col, dv_reg);

            warp::zero(dp_sub);
            warp::mma_ABt(dp_sub, do_reg, v_reg, dp_sub);
            warp::sub_row(dp_sub, dp_sub, dpsum_vec);
            warp::mul(ds_sub, p_sub, dp_sub);
            warp::mul(ds_sub, ds_sub, scale);

            warp::copy(ds_bf, ds_sub);
            warp::swap_layout(ds_col, ds_bf);

            extract_chunk<0>(q_reg, q_chunk);
            warp::swap_layout(q_chunk_col, q_chunk);
            warp::mma_AtB(dk0_reg, ds_col, q_chunk_col, dk0_reg);

            extract_chunk<1>(q_reg, q_chunk);
            warp::swap_layout(q_chunk_col, q_chunk);
            warp::mma_AtB(dk1_reg, ds_col, q_chunk_col, dk1_reg);

            extract_chunk<2>(q_reg, q_chunk);
            warp::swap_layout(q_chunk_col, q_chunk);
            warp::mma_AtB(dk2_reg, ds_col, q_chunk_col, dk2_reg);

            insert_block(p_block_t, p_sub, kv_subtile);
            insert_block(dp_block_t, dp_sub, kv_subtile);
            insert_block(ds_block_t, ds_sub, kv_subtile);
        }
        warp::copy(p_block_t_mma, p_block_t);
        warp::copy(ds_block_t_mma, ds_block_t);
    }
    warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
    group<8>::sync(10);
}

}  // namespace detail

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel(const __grid_constant__ main_globals<C> g) {
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
    struct epilogue_shared_storage {
        typename main_globals<C>::dk_full_tile dk0_smem[C::ConsumerWarpgroups];
        typename main_globals<C>::dk_full_tile dk1_smem[C::ConsumerWarpgroups];
        typename main_globals<C>::dk_full_tile dk2_smem[C::ConsumerWarpgroups];
        typename main_globals<C>::dv_full_tile dv_smem[C::ConsumerWarpgroups];
    };
    union shared_storage {
        main_shared_storage main;
        epilogue_shared_storage epilogue;
    };

    __shared__ alignas(1024) shared_storage smem;
    auto &k_smem = smem.main.k_smem;
    auto &v_smem = smem.main.v_smem;
    auto &q_smem = smem.main.q_smem;
    auto &do_smem = smem.main.do_smem;
    auto &dq_smem = smem.main.dq_smem;
    auto &ds_warp_smem = smem.main.ds_warp_smem;
    auto &lse_log2_smem = smem.main.lse_log2_smem;
    auto &dpsum_smem = smem.main.dpsum_smem;

    __shared__ __align__(16) kittens::semaphore kv_b;
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];
    __shared__ __align__(16) kittens::semaphore score_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore dp_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore kv_tmem_ready[C::ConsumerWarpgroups];

    const int warp = kittens::warpid();
    const bool is_reduce = warp < C::ReduceWarps;
    const bool is_compute = warp >= C::ReduceWarps && warp < C::ReduceWarps + C::ComputeWarps;
    const bool is_mma = warp == C::MmaWarpId;
    const bool is_load = warp == C::LoadWarpId;
    const bool is_relay = warp == C::RelayWarpId;
    const int consumer_idx = is_compute ? ((warp - C::ReduceWarps) / kittens::WARPGROUP_WARPS) : -1;

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
    const int q_start_block = 0;
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
        }
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * C::ConsumerWarpgroups);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            coord<k_tile> k_tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            coord<v_tile> v_tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem[w], g.k, k_tile_idx, kv_b);
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[w], g.v, v_tile_idx, kv_b);
        }
    }
    __syncthreads();

    if (is_compute) {
        if constexpr (kUseRegDkDvHybrid) {
            warp::zero(dk0_reg_accum);
            warp::zero(dk1_reg_accum);
            warp::zero(dk2_reg_accum);
            warp::zero(dv_reg_accum);
        } else {
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
        }
    }

    for (int q_block_idx = q_start_block; q_block_idx < q_blocks; ++q_block_idx) {
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
            rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
            rt_bf<kRefTileM, C::TileRows> p_block_t_mma, ds_block_t_mma;
            wait(kv_b, 0);
            if constexpr (kUseRegDkDvHybrid) {
                detail::cute_compute_dkdv_reg_loop<CAUSAL, C>(
                    q_b[0],
                    o_b[0],
                    p_block_t,
                    dp_block_t,
                    ds_block_t,
                    p_block_t_mma,
                    ds_block_t_mma,
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
                    q_block_idx & 1,
                    q_block_idx > (CAUSAL ? (kv_tile_base + consumer_idx) : 0),
                    q_block_idx,
                    kv_tile_base + consumer_idx,
                    g.seq_len
                );
            } else {
                detail::cute_compute_dkdv_loop<CAUSAL, C>(
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
                    q_block_idx & 1,
                    q_block_idx > (CAUSAL ? (kv_tile_base + consumer_idx) : 0),
                    q_block_idx,
                    kv_tile_base + consumer_idx,
                    g.seq_len
                );
            }
        }
        __syncthreads();

        if (is_reduce) {
            const int dq_subtile_idx = warp;
            const int q_tile_idx = q_block_idx * C::QSubtiles + dq_subtile_idx;
            const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cluster_rank;
            const int cluster_group_idx = kv_block_idx / C::ClusterSize;

            rt_bf<kRefTileM, C::TileRows> ds_local_reg, ds_peer_reg;
            rt_bf<C::TileRows, 64> k_reg;
            rt_bf<C::TileRows, 64, ducks::rt_layout::col> k_col;
            rt_fl<kRefTileM, 64> dq_chunk;
            rt_fl<kRefTileM, 64> dq_existing;
            auto k0_0 = k_smem[0].template subtile<C::TileRows, 64>({0, 0});
            auto k0_1 = k_smem[0].template subtile<C::TileRows, 64>({0, 1});
            auto k0_2 = k_smem[0].template subtile<C::TileRows, 64>({0, 2});
            auto k1_0 = k_smem[1].template subtile<C::TileRows, 64>({0, 0});
            auto k1_1 = k_smem[1].template subtile<C::TileRows, 64>({0, 1});
            auto k1_2 = k_smem[1].template subtile<C::TileRows, 64>({0, 2});

            warp::load(ds_local_reg, ds_warp_smem[0][dq_subtile_idx]);
            warp::load(ds_peer_reg, ds_warp_smem[1][dq_subtile_idx]);

            warp::load(k_reg, k0_0);
            warp::swap_layout(k_col, k_reg);
            warp::zero(dq_chunk);
            warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
            warp::load(k_reg, k1_0);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
            warp::store(dq_smem[0][dq_subtile_idx], dq_chunk);

            warp::load(k_reg, k0_1);
            warp::swap_layout(k_col, k_reg);
            warp::zero(dq_chunk);
            warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
            warp::load(k_reg, k1_1);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
            warp::store(dq_smem[1][dq_subtile_idx], dq_chunk);

            warp::load(k_reg, k0_2);
            warp::swap_layout(k_col, k_reg);
            warp::zero(dq_chunk);
            warp::mma_AB(dq_chunk, ds_local_reg, k_col, dq_chunk);
            warp::load(k_reg, k1_2);
            warp::swap_layout(k_col, k_reg);
            warp::mma_AB(dq_chunk, ds_peer_reg, k_col, dq_chunk);
            warp::store(dq_smem[2][dq_subtile_idx], dq_chunk);

            if (!g.deterministic) {
                warp::tma::store_add_async(g.dqacc0, dq_smem[0][dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
                warp::tma::store_add_async(g.dqacc1, dq_smem[1][dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
                warp::tma::store_add_async(g.dqacc2, dq_smem[2][dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
                warp::tma::store_commit_group();
                warp::tma::store_async_read_wait();
            } else {
                if (laneid() == 0) {
                    int *sem = g.dq_semaphore + detail::dq_semaphore_index(g, batch_idx, head_idx, q_tile_idx, cluster_rank);
                    while (atomicAdd(sem, 0) != cluster_group_idx) {
                    }
                }
                __syncwarp();

                warp::load(dq_existing, g.dqacc0, {batch_idx, head_idx, scratch_tile_idx, 0});
                warp::load(dq_chunk, dq_smem[0][dq_subtile_idx]);
                warp::add(dq_chunk, dq_chunk, dq_existing);
                warp::store(g.dqacc0, dq_chunk, {batch_idx, head_idx, scratch_tile_idx, 0});

                warp::load(dq_existing, g.dqacc1, {batch_idx, head_idx, scratch_tile_idx, 0});
                warp::load(dq_chunk, dq_smem[1][dq_subtile_idx]);
                warp::add(dq_chunk, dq_chunk, dq_existing);
                warp::store(g.dqacc1, dq_chunk, {batch_idx, head_idx, scratch_tile_idx, 0});

                warp::load(dq_existing, g.dqacc2, {batch_idx, head_idx, scratch_tile_idx, 0});
                warp::load(dq_chunk, dq_smem[2][dq_subtile_idx]);
                warp::add(dq_chunk, dq_chunk, dq_existing);
                warp::store(g.dqacc2, dq_chunk, {batch_idx, head_idx, scratch_tile_idx, 0});

                __syncwarp();
                if (laneid() == 0) {
                    __threadfence();
                    int *sem = g.dq_semaphore + detail::dq_semaphore_index(g, batch_idx, head_idx, q_tile_idx, cluster_rank);
                    atomicExch(sem, cluster_group_idx + 1);
                }
            }
        }
        __syncthreads();
    }

    if (is_compute) {
        rt_fl<kRefTileM, 64> dk0_reg;
        rt_fl<kRefTileM, 64> dk1_reg;
        rt_fl<kRefTileM, 64> dk2_reg;
        rt_fl<kRefTileM, C::Dvo> dv_reg;
        if constexpr (kUseRegDkDvHybrid) {
            const int kv_subtile_idx =
                (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + warpgroup::warpid();
            warp::store(g.dk0, dk0_reg_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
            warp::store(g.dk1, dk1_reg_accum, {batch_idx, head_idx, kv_subtile_idx, 1});
            warp::store(g.dk2, dk2_reg_accum, {batch_idx, head_idx, kv_subtile_idx, 2});
            warp::store(g.dv, dv_reg_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
        } else {
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(kv_tmem_ready[consumer_idx]);
            }
            wait(kv_tmem_ready[consumer_idx], 0);
            warpgroup::load_async(dk0_reg, dk0_tt[consumer_idx]);
            warpgroup::load_async(dk1_reg, dk1_tt[consumer_idx]);
            warpgroup::load_async(dk2_reg, dk2_tt[consumer_idx]);
            warpgroup::load_async(dv_reg, dv_accum_tt[consumer_idx]);
            tensor_load_wait();
            warpgroup::store(smem.epilogue.dk0_smem[consumer_idx], dk0_reg);
            warpgroup::store(smem.epilogue.dk1_smem[consumer_idx], dk1_reg);
            warpgroup::store(smem.epilogue.dk2_smem[consumer_idx], dk2_reg);
            warpgroup::store(smem.epilogue.dv_smem[consumer_idx], dv_reg);
            group<4>::sync(warpgroup::groupid() + 4);
            if (warpgroup::warpid() == 0) {
                coord<typename main_globals<C>::dk_full_tile> dk0_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 0};
                coord<typename main_globals<C>::dk_full_tile> dk1_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 1};
                coord<typename main_globals<C>::dk_full_tile> dk2_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 2};
                coord<typename main_globals<C>::dv_full_tile> dv_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 0};
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
                    g.dk0_full,
                    smem.epilogue.dk0_smem[consumer_idx],
                    dk0_tile_idx
                );
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
                    g.dk1_full,
                    smem.epilogue.dk1_smem[consumer_idx],
                    dk1_tile_idx
                );
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
                    g.dk2_full,
                    smem.epilogue.dk2_smem[consumer_idx],
                    dk2_tile_idx
                );
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(
                    g.dv_full,
                    smem.epilogue.dv_smem[consumer_idx],
                    dv_tile_idx
                );
                warp::tma::store_commit_group();
                warp::tma::store_async_wait();
            }
        }
    }
}

template <typename C>
__global__ __launch_bounds__(C::ReduceWarps * kWarpThreads, 8)
void dq_reduce_kernel(const __grid_constant__ main_globals<C> g) {
    const int warp = threadIdx.x >> 5;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    if (warp >= C::QSubtiles) {
        return;
    }

    const int q_tile_idx = q_block_idx * C::QSubtiles + warp;
    const int scratch_tile_idx = q_tile_idx * C::ClusterSize;
    rt_fl<kRefTileM, 64> dq_local, dq_peer;

    if (g.deterministic && laneid() == 0) {
        for (int cluster_rank = 0; cluster_rank < C::ClusterSize; ++cluster_rank) {
            int *sem = g.dq_semaphore + detail::dq_semaphore_index(g, batch_idx, head_idx, q_tile_idx, cluster_rank);
            while (atomicAdd(sem, 0) < g.cluster_groups) {
            }
        }
    }
    __syncwarp();

    warp::load(dq_local, g.dqacc0, {batch_idx, head_idx, scratch_tile_idx, 0});
    warp::load(dq_peer, g.dqacc0, {batch_idx, head_idx, scratch_tile_idx + 1, 0});
    warp::add(dq_local, dq_local, dq_peer);
    warp::store<dim::DEPTH>(g.dq0, dq_local, {batch_idx, q_tile_idx, head_idx, 0});

    warp::load(dq_local, g.dqacc1, {batch_idx, head_idx, scratch_tile_idx, 0});
    warp::load(dq_peer, g.dqacc1, {batch_idx, head_idx, scratch_tile_idx + 1, 0});
    warp::add(dq_local, dq_local, dq_peer);
    warp::store<dim::DEPTH>(g.dq1, dq_local, {batch_idx, q_tile_idx, head_idx, 1});

    warp::load(dq_local, g.dqacc2, {batch_idx, head_idx, scratch_tile_idx, 0});
    warp::load(dq_peer, g.dqacc2, {batch_idx, head_idx, scratch_tile_idx + 1, 0});
    warp::add(dq_local, dq_local, dq_peer);
    warp::store<dim::DEPTH>(g.dq2, dq_local, {batch_idx, q_tile_idx, head_idx, 2});
}

namespace detail {

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

template <typename C>
inline void launch_reduce(
    const main_globals<C> &g,
    int q_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid(q_blocks, heads, batch_size);
    dq_reduce_kernel<C><<<grid, C::ReduceWarps * kWarpThreads, 0, stream>>>(g);
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
    at::Tensor &dqacc0,
    at::Tensor &dqacc1,
    at::Tensor &dqacc2,
    at::Tensor &dq_semaphore,
    bool causal,
    float scale,
    bool deterministic
) {
    using G = main_globals<C>;
    const int total_ctas = static_cast<int>((q.size(1) / (C::TileRows * C::ConsumerWarpgroups)) * C::ClusterSize);
    const int scratch_rows = static_cast<int>(q.size(1) * C::ClusterSize);
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
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
        kittens::py::tensor_to_gl<typename G::dk0_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk1_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk2_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk_full_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk_full_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk_full_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::dv_full_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        dq_semaphore.defined() ? reinterpret_cast<int *>(dq_semaphore.data_ptr<int>()) : nullptr,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(1) / kRefTileM),
        static_cast<int>(q.size(1) / (kForwardTileN * C::ClusterSize)),
        deterministic ? 1 : 0,
    };

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        detail::launch_main<true, C>(g, total_ctas, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    } else {
        detail::launch_main<false, C>(g, total_ctas, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    detail::launch_reduce<C>(g, static_cast<int>(q.size(1) / C::TileRows), static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd_cute

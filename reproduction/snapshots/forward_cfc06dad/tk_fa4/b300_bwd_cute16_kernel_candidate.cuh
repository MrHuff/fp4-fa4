#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "b300_bwd_cute16_kernel.cuh"

namespace tkfa4::bwd_cute16_kernel_candidate {

namespace detail {

inline bool clustered_dq_timing_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("TK_FA4_CLUSTERED_DQ_TIMING");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

}  // namespace detail

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
struct config {
    static_assert(_Mb == kForwardTileM, "Exact B300 candidate kernel requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 candidate kernel requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 candidate kernel requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 candidate kernel requires Dvo=128");
    static_assert(_ClusterSize == 1 || _ClusterSize == 2, "Exact B300 candidate kernel requires ClusterSize=1 or 2");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = _ClusterSize;
    static constexpr int WarpTiles = 8;
    static constexpr int BlockThreads = WarpTiles * kWarpThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
struct dq_only_dedicated_load_config {
    static_assert(_Mb == kForwardTileM, "Exact B300 dq-only config requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 dq-only config requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 dq-only config requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 dq-only config requires Dvo=128");
    static_assert(_ClusterSize == 1 || _ClusterSize == 2, "Exact B300 dq-only config requires ClusterSize=1 or 2");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = _ClusterSize;
    static constexpr int WarpTiles = 8;
    static constexpr int BlockThreads = (WarpTiles + 2) * kWarpThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <typename C>
struct main_globals {
    using q_tma_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using do_tma_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tma_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tma_tile, dim::DEPTH>>;
    using dqacc_chunk_gl = gl<float, -1, -1, -1, -1, dq_chunk_tile>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
    using dq_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dqacc_chunk_gl dqacc_chunks;
    dqacc_gl dqacc;
    dq_gl dq;
    dk_gl dk;
    dv_gl dv;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
    int actual_seq_len;
    int *dq_semaphore;
    int heads;
    int q_tiles;
    int cluster_groups;
    int deterministic;
};

template <typename C, typename DkdvOutT = float>
struct dkdv_only_globals {
    using q_tma_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using do_tma_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tma_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tma_tile, dim::DEPTH>>;
    using dk_gl = gl<DkdvOutT, -1, -1, -1, C::Dqk>;
    using dv_gl = gl<DkdvOutT, -1, -1, -1, C::Dvo>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk_gl dk;
    dv_gl dv;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

template <typename C>
struct dkdv_only_ds_globals {
    using q_tma_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using do_tma_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using ds_tile = st_bf<kRefTileM, kRefTileN>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tma_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tma_tile, dim::DEPTH>>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using ds_gl = gl<bf16, -1, -1, -1, -1, ds_tile>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk_gl dk;
    dv_gl dv;
    ds_gl ds;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

template <typename C>
struct shared_ds_monolithic_globals {
    using q_tma_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using do_tma_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tma_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tma_tile, dim::DEPTH>>;
    using dq_out_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dq_chunk_tile, dim::DEPTH>>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    dk_gl dk;
    dv_gl dv;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

template <typename C>
struct dq_only_globals {
    using q_tma_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using do_tma_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tma_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tma_tile, dim::DEPTH>>;
    using dqacc_chunk_gl = gl<float, -1, -1, -1, -1, dq_chunk_tile>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
    using dq_out_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dq_chunk_tile, dim::DEPTH>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dqacc_chunk_gl dqacc_chunks;
    dqacc_gl dqacc;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

template <typename C>
struct dq_from_ds_globals {
    using ds_tile = st_bf<kRefTileM, kRefTileN>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using ds_gl = gl<bf16, -1, -1, -1, -1, ds_tile>;
    using dq_out_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dq_chunk_tile, dim::DEPTH>>;

    k_gl k;
    ds_gl ds;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    int seq_len;
};

template <typename C>
struct dq_reduce_globals {
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
    using dq_gl = gl<float, -1, -1, -1, C::Dqk>;

    dqacc_gl dqacc;
    dq_gl dq;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo>
struct seq2048_exact_config {
    static_assert(_Mb == kForwardTileM, "Exact B300 seq2048 split kernel requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 seq2048 split kernel requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 seq2048 split kernel requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 seq2048 split kernel requires Dvo=128");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = 2;

    static constexpr int TileRows = Nb / 2;
    static constexpr int QSubtiles = TileRows / kRefTileM;
    static constexpr int ConsumerWarpgroups = 2;
    static constexpr int ComputeWarps = ConsumerWarpgroups * WARPGROUP_WARPS;
    static constexpr int ReduceWarps = QSubtiles;
    static constexpr int LoadWarpId = ComputeWarps;
    static constexpr int DkdvBlockThreads = ComputeWarps * kWarpThreads;
    static constexpr int DqBlockThreads = (ComputeWarps + 1) * kWarpThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <typename C>
struct seq2048_exact_dkdv_globals {
    using q_tile = st_bf<C::TileRows, C::Dqk>;
    using do_tile = st_bf<C::TileRows, C::Dvo>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tile, dim::DEPTH>>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk_gl dk;
    dv_gl dv;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo>
struct dense_tmem_frontier_config {
    static_assert(_Mb == kForwardTileM, "Dense TMEM frontier kernel requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Dense TMEM frontier kernel requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Dense TMEM frontier kernel requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Dense TMEM frontier kernel requires Dvo=128");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = 2;
    static constexpr int TileRows = Nb / 2;
    static constexpr int QSubtiles = TileRows / kRefTileM;
    static constexpr int ConsumerWarpgroups = 2;
    static constexpr int ComputeWarps = ConsumerWarpgroups * WARPGROUP_WARPS;
    static constexpr int DenseBlockThreads = (ComputeWarps + 1) * kWarpThreads;
    static constexpr int FusedDqReduceWarps = 4;
    static constexpr int FusedDqReduceWarpBase = ComputeWarps + 1;
    static constexpr int FusedDqBlockThreads =
        (ComputeWarps + 1 + FusedDqReduceWarps) * kWarpThreads;
    static constexpr int FusedDqLoadOverlapReduceWarpBase = ComputeWarps;
    static constexpr int FusedDqLoadOverlapBlockThreads =
        (ComputeWarps + FusedDqReduceWarps) * kWarpThreads;
    static constexpr int FrontierBlockThreads = ComputeWarps * kWarpThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <typename C>
struct dense_tmem_frontier_globals {
    using q_tile = st_bf<C::TileRows, C::Dqk>;
    using k_tile = st_bf<C::TileRows, C::Dqk>;
    using v_tile = st_bf<C::TileRows, C::Dvo>;
    using do_tile = st_bf<C::TileRows, C::Dvo>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<k_tile, dim::DEPTH>>;
    using v_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tile, dim::DEPTH>>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using dq_out_gl = gl<
        float,
        -1,
        -1,
        -1,
        -1,
        tma::descriptor<dq_chunk_tile, dim::DEPTH>
    >;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk_gl dk;
    dv_gl dv;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
    int batch_size;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo>
struct dq_only_clustered_config {
    static_assert(_Mb == kForwardTileM, "Exact B300 clustered dq-only kernel requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 clustered dq-only kernel requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 clustered dq-only kernel requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 clustered dq-only kernel requires Dvo=128");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = 2;

    static constexpr int TileRows = Nb / 2;
    static constexpr int QSubtiles = TileRows / kRefTileM;
    static constexpr int ConsumerWarpgroups = 2;
    static constexpr int ComputeWarps = ConsumerWarpgroups * WARPGROUP_WARPS;
    static constexpr int ReduceWarps = QSubtiles;
    static constexpr int DqBlockThreads = (ComputeWarps + 1) * kWarpThreads;
    static constexpr int BlockThreads = DqBlockThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo>
struct dq_only_clustered_cluster1_config {
    static_assert(_Mb == kForwardTileM, "Exact B300 clustered dq-only kernel requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 clustered dq-only kernel requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 clustered dq-only kernel requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 clustered dq-only kernel requires Dvo=128");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = 1;

    static constexpr int TileRows = Nb / 2;
    static constexpr int QSubtiles = TileRows / kRefTileM;
    static constexpr int ConsumerWarpgroups = 2;
    static constexpr int ComputeWarps = ConsumerWarpgroups * WARPGROUP_WARPS;
    static constexpr int ReduceWarps = QSubtiles;
    static constexpr int DqBlockThreads = (ComputeWarps + 1) * kWarpThreads;
    static constexpr int BlockThreads = DqBlockThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo>
struct dq_only_clustered_pipelined_config {
    static_assert(_Mb == kForwardTileM, "Exact B300 pipelined dq-only kernel requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 pipelined dq-only kernel requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 pipelined dq-only kernel requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 pipelined dq-only kernel requires Dvo=128");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = 1;

    static constexpr int TileRows = Nb / 2;
    static constexpr int QSubtiles = TileRows / kRefTileM;
    static constexpr int ConsumerWarpgroups = 2;
    static constexpr int ComputeWarps = ConsumerWarpgroups * WARPGROUP_WARPS;
    static constexpr int ReduceWarps = QSubtiles;
    static constexpr int ReduceWarpBase = ComputeWarps;
    static constexpr int LoadWarp = ReduceWarpBase + ReduceWarps;
    static constexpr int BlockThreads = (LoadWarp + 1) * kWarpThreads;
    static constexpr int MinBlocksPerSm = 1;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo>
struct dq_only_compact_cluster2_config {
    static_assert(_Mb == kForwardTileM, "Exact B300 compact clustered dq-only kernel requires Mb=128");
    static_assert(_Nb == kForwardTileN, "Exact B300 compact clustered dq-only kernel requires Nb=128");
    static_assert(_Dqk == kB300QKDim, "Exact B300 compact clustered dq-only kernel requires Dqk=192");
    static_assert(_Dvo == kB300VDim, "Exact B300 compact clustered dq-only kernel requires Dvo=128");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr int ClusterSize = 2;
    static constexpr int WarpTiles = 4;
    static constexpr int BlockThreads = WarpTiles * kWarpThreads;
    static constexpr int MinBlocksPerSm = 1;
};

enum class dq_only_clustered_mode : int {
    LegacyPatched,
    DonorBulkOnly,
};

constexpr dq_only_clustered_mode kDqOnlyClusteredMode = dq_only_clustered_mode::LegacyPatched;
inline constexpr bool kUseFusedPatchReduceClusteredDq = false;
inline constexpr bool kUseChunkedClusteredDqReduce = true;
inline constexpr bool kUseDirectFinalClusteredDq = true;
inline constexpr bool kUseMainFirstBlockClusteredDq = true;

template <typename C, typename DqOutT = float>
struct dq_only_clustered_globals {
    using q_tile = st_bf<C::TileRows, C::Dqk>;
    using k_tile = st_bf<C::TileRows, C::Dqk>;
    using v_tile = st_bf<C::TileRows, C::Dvo>;
    using do_tile = st_bf<C::TileRows, C::Dvo>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tile, dim::DEPTH>>;
    using dqacc_chunk_gl = gl<float, -1, -1, -1, -1, dq_chunk_tile>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
    using dq_out_gl = gl<DqOutT, -1, -1, -1, -1, tma::descriptor<dq_chunk_tile, dim::DEPTH>>;
    using dq_full_out_gl = gl<DqOutT, -1, -1, -1, -1, tma::descriptor<dqacc_tile, dim::DEPTH>>;
    using dq_gl = gl<DqOutT, -1, -1, -1, C::Dqk>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dqacc_chunk_gl dqacc_chunks;
    dqacc_gl dqacc;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    dq_full_out_gl dq_full;
    dq_gl dq;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
    int heads;
    int q_tiles;
};

namespace detail {

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

template <bool CAUSAL, bool DENSE_UNMASKED, typename C>
__device__ inline void repair_dq_step_accumulate(
    rt_fl<kRefTileM, C::Dqk> &dq_accum,
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
    int actual_seq_len
) {
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;
    if constexpr (DENSE_UNMASKED) {
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
    warp::mma_AB(dq_accum, ds_bf, k_col, dq_accum);
}

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

template <typename C, bool CAUSAL, bool DENSE_UNMASKED>
__device__ inline void backward_tile_step_compact(
    rt_fl<kRefTileM, C::Dqk> &dq_partial,
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
    int actual_seq_len
) {
    rt_fl<kRefTileM, kRefTileN> p, ds;

    if constexpr (DENSE_UNMASKED) {
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
        warp::zero(ds);
        warp::mma_ABt(ds, do_reg, v_reg, ds);
        warp::sub_row(ds, ds, dpsum_vec);
        warp::mul(ds, p, ds);
        warp::mul(ds, ds, scale);
    }

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

        warp::zero(dq_partial);
        warp::mma_AB(dq_partial, ds_bf, k_col, dq_partial);
    }
}

template <typename C, bool CAUSAL, bool DENSE_UNMASKED>
__device__ inline void backward_tile_step_compact_chunked(
    rt_fl<kRefTileM, 64> &dq0,
    rt_fl<kRefTileM, 64> &dq1,
    rt_fl<kRefTileM, 64> &dq2,
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
    int actual_seq_len
) {
    rt_fl<kRefTileM, kRefTileN> p, ds;

    if constexpr (DENSE_UNMASKED) {
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
        warp::zero(ds);
        warp::mma_ABt(ds, do_reg, v_reg, ds);
        warp::sub_row(ds, ds, dpsum_vec);
        warp::mul(ds, p, ds);
        warp::mul(ds, ds, scale);
    }

    {
        rt_bf<kRefTileM, kRefTileN> ds_bf;
        warp::copy(ds_bf, ds);
        {
            rt_bf<kRefTileM, C::Dqk, ducks::rt_layout::col> q_col;
            rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
            warp::swap_layout(ds_col, ds_bf);
            warp::swap_layout(q_col, q_reg);
            warp::mma_AtB(dk_accum, ds_col, q_col, dk_accum);
        }
        {
            rt_bf<kRefTileM, 64> k_chunk;
            rt_bf<kRefTileM, 64, ducks::rt_layout::col> k_col;

            extract_chunk<0>(k_reg, k_chunk);
            warp::swap_layout(k_col, k_chunk);
            warp::zero(dq0);
            warp::mma_AB(dq0, ds_bf, k_col, dq0);

            extract_chunk<1>(k_reg, k_chunk);
            warp::swap_layout(k_col, k_chunk);
            warp::zero(dq1);
            warp::mma_AB(dq1, ds_bf, k_col, dq1);

            extract_chunk<2>(k_reg, k_chunk);
            warp::swap_layout(k_col, k_chunk);
            warp::zero(dq2);
            warp::mma_AB(dq2, ds_bf, k_col, dq2);
        }
    }
}

template <typename C, bool CAUSAL, bool DENSE_UNMASKED>
__device__ inline void backward_tile_step_compact_fully_chunked(
    rt_fl<kRefTileM, 64> &dq0,
    rt_fl<kRefTileM, 64> &dq1,
    rt_fl<kRefTileM, 64> &dq2,
    rt_fl<kRefTileM, 64> &dk0_accum,
    rt_fl<kRefTileM, 64> &dk1_accum,
    rt_fl<kRefTileM, 64> &dk2_accum,
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
    int actual_seq_len
) {
    rt_fl<kRefTileM, kRefTileN> p, ds;

    if constexpr (DENSE_UNMASKED) {
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
        warp::zero(ds);
        warp::mma_ABt(ds, do_reg, v_reg, ds);
        warp::sub_row(ds, ds, dpsum_vec);
        warp::mul(ds, p, ds);
        warp::mul(ds, ds, scale);
    }

    rt_bf<kRefTileM, kRefTileN> ds_bf;
    warp::copy(ds_bf, ds);

    {
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
        rt_bf<kRefTileM, 64> q_chunk;
        rt_bf<kRefTileM, 64, ducks::rt_layout::col> q_col;

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

    {
        rt_bf<kRefTileM, 64> k_chunk;
        rt_bf<kRefTileM, 64, ducks::rt_layout::col> k_col;

        extract_chunk<0>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq0);
        warp::mma_AB(dq0, ds_bf, k_col, dq0);

        extract_chunk<1>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq1);
        warp::mma_AB(dq1, ds_bf, k_col, dq1);

        extract_chunk<2>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq2);
        warp::mma_AB(dq2, ds_bf, k_col, dq2);
    }
}

template <
    typename C,
    bool CAUSAL,
    bool DENSE_UNMASKED,
    bool COMPUTE_DK = true,
    bool COMPUTE_DV = true
>
__device__ inline void repair_dkdv_step_chunked(
    rt_fl<kRefTileM, 64> &dk0_accum,
    rt_fl<kRefTileM, 64> &dk1_accum,
    rt_fl<kRefTileM, 64> &dk2_accum,
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
    int actual_seq_len
) {
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;

    if constexpr (DENSE_UNMASKED) {
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

    if constexpr (COMPUTE_DV) {
        rt_bf<kRefTileM, kRefTileN> p_bf;
        rt_bf<kRefTileM, C::Dvo, ducks::rt_layout::col> do_col;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col;
        warp::copy(p_bf, p);
        warp::swap_layout(p_col, p_bf);
        warp::swap_layout(do_col, do_reg);
        warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);
    }

    if constexpr (COMPUTE_DK) {
        warp::zero(dp);
        warp::mma_ABt(dp, do_reg, v_reg, dp);
        warp::sub_row(dp, dp, dpsum_vec);
        warp::mul(ds, p, dp);
        warp::mul(ds, ds, scale);

        rt_bf<kRefTileM, kRefTileN> ds_bf;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
        warp::copy(ds_bf, ds);
        warp::swap_layout(ds_col, ds_bf);

        rt_bf<kRefTileM, 64> q_chunk;
        rt_bf<kRefTileM, 64, ducks::rt_layout::col> q_col;

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

template <typename C, bool CAUSAL, bool DENSE_UNMASKED>
__device__ inline void repair_dk_step_one_chunk(
    rt_fl<kRefTileM, 64> &dk_accum,
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
    int dk_chunk_idx
) {
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;
    if constexpr (DENSE_UNMASKED) {
        bwd_fa4::detail::reconstruct_probability_tile_dense<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e
        );
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
    rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
    rt_bf<kRefTileM, 64> q_chunk;
    rt_bf<kRefTileM, 64, ducks::rt_layout::col> q_col;
    warp::copy(ds_bf, ds);
    warp::swap_layout(ds_col, ds_bf);
    if (dk_chunk_idx == 0) {
        extract_chunk<0>(q_reg, q_chunk);
    } else if (dk_chunk_idx == 1) {
        extract_chunk<1>(q_reg, q_chunk);
    } else {
        extract_chunk<2>(q_reg, q_chunk);
    }
    warp::swap_layout(q_col, q_chunk);
    warp::mma_AtB(dk_accum, ds_col, q_col, dk_accum);
}

template <typename C, bool CAUSAL, bool DENSE_UNMASKED>
__device__ inline void repair_dkdv_step_chunked_store_ds(
    rt_fl<kRefTileM, 64> &dk0_accum,
    rt_fl<kRefTileM, 64> &dk1_accum,
    rt_fl<kRefTileM, 64> &dk2_accum,
    rt_fl<kRefTileM, C::Dvo> &dv_accum,
    rt_bf<kRefTileM, kRefTileN> &ds_bf_out,
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
    int actual_seq_len
) {
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;

    if constexpr (DENSE_UNMASKED) {
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

    rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
    warp::copy(ds_bf_out, ds);
    warp::swap_layout(ds_col, ds_bf_out);

    rt_bf<kRefTileM, 64> q_chunk;
    rt_bf<kRefTileM, 64, ducks::rt_layout::col> q_col;

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

template <
    typename C,
    int QTilesBuffered,
    typename QSmemTile = st_bf<kRefTileM, C::Dqk>,
    typename DoSmemTile = st_bf<kRefTileM, C::Dvo>,
    typename StatsSmemTile = typename main_globals<C>::stats_tile
>
__device__ inline void preload_q_block(
    const main_globals<C> &g,
    int batch_idx,
    int head_idx,
    int q_tile_base,
    int warp,
    QSmemTile (&q_smem)[QTilesBuffered],
    DoSmemTile (&do_smem)[QTilesBuffered],
    StatsSmemTile (&lse_log2_smem)[QTilesBuffered],
    StatsSmemTile (&dpsum_smem)[QTilesBuffered]
) {
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    if constexpr (C::WarpTiles >= 2 * QTilesBuffered) {
        if (warp < QTilesBuffered) {
            rt_bf<kRefTileM, C::Dqk> q_stage_reg;
            stats_vec lse_stage_vec;
            warp::load<dim::DEPTH>(q_stage_reg, g.q, {batch_idx, q_tile_base + warp, head_idx, 0});
            warp::store(q_smem[warp], q_stage_reg);
            warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_stage_vec);
        } else if (warp < 2 * QTilesBuffered) {
            const int subtile = warp - QTilesBuffered;
            rt_bf<kRefTileM, C::Dvo> do_stage_reg;
            stats_vec dpsum_stage_vec;
            warp::load<dim::DEPTH>(do_stage_reg, g.dout, {batch_idx, q_tile_base + subtile, head_idx, 0});
            warp::store(do_smem[subtile], do_stage_reg);
            warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
            warp::store(dpsum_smem[subtile], dpsum_stage_vec);
        }
    } else {
        if (warp < QTilesBuffered) {
            rt_bf<kRefTileM, C::Dqk> q_stage_reg;
            rt_bf<kRefTileM, C::Dvo> do_stage_reg;
            stats_vec lse_stage_vec, dpsum_stage_vec;
            warp::load<dim::DEPTH>(q_stage_reg, g.q, {batch_idx, q_tile_base + warp, head_idx, 0});
            warp::store(q_smem[warp], q_stage_reg);
            warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_stage_vec);
            warp::load<dim::DEPTH>(do_stage_reg, g.dout, {batch_idx, q_tile_base + warp, head_idx, 0});
            warp::store(do_smem[warp], do_stage_reg);
            warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_stage_vec);
        }
    }
}

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel(const __grid_constant__ main_globals<C> g) {
    constexpr int q_tiles_buffered = 2;
    constexpr int dense_q_block_lag = (C::WarpTiles + q_tiles_buffered - 1) / q_tiles_buffered;
    using qk_bf_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo>;
    using dqacc_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_smem_tile = typename main_globals<C>::stats_tile;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) dqacc_tile dq_smem[C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];

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
    const bool full_seq = g.actual_seq_len == g.seq_len;
    const int q_block_start = (CAUSAL && full_seq) ? (kv_tile_base / q_tiles_buffered) : 0;
    const bool globally_dense_unmasked = !CAUSAL && full_seq;

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, C::Dqk> dk_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    if constexpr (CAUSAL) {
        if (full_seq) {
            const int dense_q_block_start = min(num_q_blocks, q_block_start + dense_q_block_lag);

            for (int q_block_idx = q_block_start; q_block_idx < dense_q_block_start; ++q_block_idx) {
                const int q_tile_base = q_block_idx * q_tiles_buffered;
                preload_q_block<C, q_tiles_buffered>(
                    g,
                    batch_idx,
                    head_idx,
                    q_tile_base,
                    warp,
                    q_smem,
                    do_smem,
                    lse_log2_smem,
                    dpsum_smem
                );
                __syncthreads();

                #pragma unroll
                for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                    const int q_tile_idx = q_tile_base + subtile;
                    if (kv_subtile_idx > q_tile_idx) {
                        continue;
                    }

                    const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cluster_rank;
                    rt_bf<kRefTileM, C::Dqk> q_reg;
                    rt_bf<kRefTileM, C::Dvo> do_reg;
                    rt_fl<kRefTileM, C::Dqk> dq_partial;
                    stats_vec lse_log2_vec, dpsum_vec;

                    warp::load(q_reg, q_smem[subtile]);
                    warp::load(do_reg, do_smem[subtile]);
                    warp::load(lse_log2_vec, lse_log2_smem[subtile]);
                    warp::load(dpsum_vec, dpsum_smem[subtile]);

                    if (kv_subtile_idx < q_tile_idx) {
                        backward_tile_step_compact<C, true, true>(
                            dq_partial,
                            dk_accum,
                            dv_accum,
                            q_reg,
                            k_reg,
                            v_reg,
                            do_reg,
                            lse_log2_vec,
                            dpsum_vec,
                            g.scale,
                            g.scale_log2e,
                            q_tile_idx,
                            kv_subtile_idx,
                            g.actual_seq_len
                        );
                    } else {
                        backward_tile_step_compact<C, true, false>(
                            dq_partial,
                            dk_accum,
                            dv_accum,
                            q_reg,
                            k_reg,
                            v_reg,
                            do_reg,
                            lse_log2_vec,
                            dpsum_vec,
                            g.scale,
                            g.scale_log2e,
                            q_tile_idx,
                            kv_subtile_idx,
                            g.actual_seq_len
                        );
                    }

                    if (q_block_idx > 0 || subtile > 0) {
                        warp::tma::store_async_read_wait();
                    }
                    warp::store(dq_smem[warp], dq_partial);
                    warp::tma::store_add_async(g.dqacc, dq_smem[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
                    warp::tma::store_commit_group();
                }
                __syncthreads();
            }

            for (int q_block_idx = dense_q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
                const int q_tile_base = q_block_idx * q_tiles_buffered;
                preload_q_block<C, q_tiles_buffered>(
                    g,
                    batch_idx,
                    head_idx,
                    q_tile_base,
                    warp,
                    q_smem,
                    do_smem,
                    lse_log2_smem,
                    dpsum_smem
                );
                __syncthreads();

                #pragma unroll 1
                for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                    const int q_tile_idx = q_tile_base + subtile;
                    const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cluster_rank;
                    rt_bf<kRefTileM, C::Dqk> q_reg;
                    rt_bf<kRefTileM, C::Dvo> do_reg;
                    rt_fl<kRefTileM, C::Dqk> dq_partial;
                    stats_vec lse_log2_vec, dpsum_vec;

                    warp::load(q_reg, q_smem[subtile]);
                    warp::load(do_reg, do_smem[subtile]);
                    warp::load(lse_log2_vec, lse_log2_smem[subtile]);
                    warp::load(dpsum_vec, dpsum_smem[subtile]);

                    backward_tile_step_compact<C, true, true>(
                        dq_partial,
                        dk_accum,
                        dv_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        dpsum_vec,
                        g.scale,
                        g.scale_log2e,
                        q_tile_idx,
                        kv_subtile_idx,
                        g.actual_seq_len
                    );

                    if (q_block_idx > 0 || subtile > 0) {
                        warp::tma::store_async_read_wait();
                    }
                    warp::store(dq_smem[warp], dq_partial);
                    warp::tma::store_add_async(g.dqacc, dq_smem[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
                    warp::tma::store_commit_group();
                }
                __syncthreads();
            }

            warp::tma::store_async_read_wait();
            warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
            warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
            return;
        }
    }

    for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const bool full_seq_dense_block = CAUSAL && full_seq && (q_tile_base >= kv_tile_base + C::WarpTiles);
        preload_q_block<C, q_tiles_buffered>(
            g,
            batch_idx,
            head_idx,
            q_tile_base,
            warp,
            q_smem,
            do_smem,
            lse_log2_smem,
            dpsum_smem
        );
        __syncthreads();

        #pragma unroll
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cluster_rank;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, C::Dqk> dq_partial;
            stats_vec lse_log2_vec, dpsum_vec;
            bool tile_dense_unmasked = globally_dense_unmasked;

            if constexpr (CAUSAL) {
                if (full_seq_dense_block) {
                    tile_dense_unmasked = true;
                } else if (full_seq) {
                    if (kv_subtile_idx > q_tile_idx) {
                        continue;
                    }
                    tile_dense_unmasked = kv_subtile_idx < q_tile_idx;
                } else {
                    const int q_tile_row = q_tile_idx * kRefTileM;
                    const int kv_tile_col = kv_subtile_idx * kRefTileN;
                    if (q_tile_row >= g.actual_seq_len || kv_tile_col >= g.actual_seq_len || kv_subtile_idx > q_tile_idx) {
                        continue;
                    }
                    tile_dense_unmasked =
                        (q_tile_row + kRefTileM <= g.actual_seq_len) &&
                        (kv_tile_col + kRefTileN <= g.actual_seq_len) &&
                        kv_subtile_idx < q_tile_idx;
                }
            } else if (!full_seq) {
                const int q_tile_row = q_tile_idx * kRefTileM;
                const int kv_tile_col = kv_subtile_idx * kRefTileN;
                if (q_tile_row >= g.actual_seq_len || kv_tile_col >= g.actual_seq_len) {
                    continue;
                }
            }

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            if (tile_dense_unmasked) {
                backward_tile_step_compact<C, CAUSAL, true>(
                    dq_partial,
                    dk_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.actual_seq_len
                );
            } else {
                backward_tile_step_compact<C, CAUSAL, false>(
                    dq_partial,
                    dk_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.actual_seq_len
                );
            }

                    if (q_block_idx > 0 || subtile > 0) {
                        warp::tma::store_async_read_wait();
                    }
                    warp::store(dq_smem[warp], dq_partial);
                    warp::tma::store_add_async(g.dqacc, dq_smem[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
                    warp::tma::store_commit_group();
        }
        __syncthreads();
    }

    warp::tma::store_async_read_wait();
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_fullseq(const __grid_constant__ main_globals<C> g) {
    constexpr int q_tiles_buffered = 2;
    constexpr int dense_q_block_lag = (C::WarpTiles + q_tiles_buffered - 1) / q_tiles_buffered;
    using qk_bf_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using dq_chunk_tile = typename main_globals<C>::dq_chunk_tile;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo, true, 64>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) dq_chunk_tile dq_smem[3][C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];
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
    const int q_block_start = kv_tile_base / q_tiles_buffered;
    const int dense_q_block_start = min(num_q_blocks, q_block_start + dense_q_block_lag);

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, C::Dqk> dk_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    for (int q_block_idx = q_block_start; q_block_idx < dense_q_block_start; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        if constexpr (C::WarpTiles >= 2 * q_tiles_buffered) {
            if (warp < q_tiles_buffered) {
                rt_bf<kRefTileM, C::Dqk> q_stage_reg;
                stats_vec lse_log2_vec;
                warp::load<dim::DEPTH>(q_stage_reg, g.q, {batch_idx, q_tile_base + warp, head_idx, 0});
                warp::store(q_smem[warp], q_stage_reg);
                warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
                warp::store(lse_log2_smem[warp], lse_log2_vec);
            } else if (warp < 2 * q_tiles_buffered) {
                const int subtile = warp - q_tiles_buffered;
                rt_bf<kRefTileM, C::Dvo> do_stage_reg;
                stats_vec dpsum_vec;
                warp::load<dim::DEPTH>(do_stage_reg, g.dout, {batch_idx, q_tile_base + subtile, head_idx, 0});
                warp::store(do_smem[subtile], do_stage_reg);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[subtile], dpsum_vec);
            }
        } else {
            if (warp < q_tiles_buffered) {
                rt_bf<kRefTileM, C::Dqk> q_stage_reg;
                rt_bf<kRefTileM, C::Dvo> do_stage_reg;
                stats_vec lse_log2_vec, dpsum_vec;
                warp::load<dim::DEPTH>(q_stage_reg, g.q, {batch_idx, q_tile_base + warp, head_idx, 0});
                warp::store(q_smem[warp], q_stage_reg);
                warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
                warp::store(lse_log2_smem[warp], lse_log2_vec);
                warp::load<dim::DEPTH>(do_stage_reg, g.dout, {batch_idx, q_tile_base + warp, head_idx, 0});
                warp::store(do_smem[warp], do_stage_reg);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
                warp::store(dpsum_smem[warp], dpsum_vec);
            }
        }
        __syncthreads();
        const int first_active_subtile = max(0, kv_subtile_idx - q_tile_base);
        const int diagonal_subtile = kv_subtile_idx - q_tile_base;
        #pragma unroll
        for (int subtile = first_active_subtile; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cluster_rank;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, 64> dq0, dq1, dq2;
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);
            if (subtile != diagonal_subtile) {
                backward_tile_step_compact_chunked<C, true, true>(
                    dq0,
                    dq1,
                    dq2,
                    dk_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len
                );
            } else {
                backward_tile_step_compact_chunked<C, true, false>(
                    dq0,
                    dq1,
                    dq2,
                    dk_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len
                );
            }
            if (q_block_idx != q_block_start || subtile != first_active_subtile) {
                warp::tma::store_async_read_wait();
            }
            warp::store(dq_smem[0][warp], dq0);
            warp::store(dq_smem[1][warp], dq1);
            warp::store(dq_smem[2][warp], dq2);
            warp::tma::store_add_async(g.dqacc_chunks, dq_smem[0][warp], {batch_idx, head_idx, scratch_tile_idx, 0});
            warp::tma::store_add_async(g.dqacc_chunks, dq_smem[1][warp], {batch_idx, head_idx, scratch_tile_idx, 1});
            warp::tma::store_add_async(g.dqacc_chunks, dq_smem[2][warp], {batch_idx, head_idx, scratch_tile_idx, 2});
        }
        warp::tma::store_commit_group();
        __syncthreads();
    }

    for (int q_block_idx = dense_q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        if constexpr (C::WarpTiles >= 2 * q_tiles_buffered) {
            if (warp < q_tiles_buffered) {
                rt_bf<kRefTileM, C::Dqk> q_stage_reg;
                stats_vec lse_log2_vec;
                warp::load<dim::DEPTH>(q_stage_reg, g.q, {batch_idx, q_tile_base + warp, head_idx, 0});
                warp::store(q_smem[warp], q_stage_reg);
                warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
                warp::store(lse_log2_smem[warp], lse_log2_vec);
            } else if (warp < 2 * q_tiles_buffered) {
                const int subtile = warp - q_tiles_buffered;
                rt_bf<kRefTileM, C::Dvo> do_stage_reg;
                stats_vec dpsum_vec;
                warp::load<dim::DEPTH>(do_stage_reg, g.dout, {batch_idx, q_tile_base + subtile, head_idx, 0});
                warp::store(do_smem[subtile], do_stage_reg);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[subtile], dpsum_vec);
            }
        } else {
            if (warp < q_tiles_buffered) {
                rt_bf<kRefTileM, C::Dqk> q_stage_reg;
                rt_bf<kRefTileM, C::Dvo> do_stage_reg;
                stats_vec lse_log2_vec, dpsum_vec;
                warp::load<dim::DEPTH>(q_stage_reg, g.q, {batch_idx, q_tile_base + warp, head_idx, 0});
                warp::store(q_smem[warp], q_stage_reg);
                warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
                warp::store(lse_log2_smem[warp], lse_log2_vec);
                warp::load<dim::DEPTH>(do_stage_reg, g.dout, {batch_idx, q_tile_base + warp, head_idx, 0});
                warp::store(do_smem[warp], do_stage_reg);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
                warp::store(dpsum_smem[warp], dpsum_vec);
            }
        }
        __syncthreads();
        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cluster_rank;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, 64> dq0, dq1, dq2;
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            backward_tile_step_compact_chunked<C, true, true>(
                dq0,
                dq1,
                dq2,
                dk_accum,
                dv_accum,
                q_reg,
                k_reg,
                v_reg,
                do_reg,
                lse_log2_vec,
                dpsum_vec,
                g.scale,
                g.scale_log2e,
                q_tile_idx,
                kv_subtile_idx,
                g.seq_len
            );
            if (q_block_idx != dense_q_block_start || subtile != 0) {
                warp::tma::store_async_read_wait();
            }
            warp::store(dq_smem[0][warp], dq0);
            warp::store(dq_smem[1][warp], dq1);
            warp::store(dq_smem[2][warp], dq2);
            warp::tma::store_add_async(g.dqacc_chunks, dq_smem[0][warp], {batch_idx, head_idx, scratch_tile_idx, 0});
            warp::tma::store_add_async(g.dqacc_chunks, dq_smem[1][warp], {batch_idx, head_idx, scratch_tile_idx, 1});
            warp::tma::store_add_async(g.dqacc_chunks, dq_smem[2][warp], {batch_idx, head_idx, scratch_tile_idx, 2});
        }
        warp::tma::store_commit_group();
        __syncthreads();
    }

    warp::tma::store_async_read_wait();
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_fullseq_shared_ds_exact(const __grid_constant__ shared_ds_monolithic_globals<C> g) {
    constexpr int q_tiles_buffered = 4;
    constexpr int dense_q_block_lag = (C::WarpTiles + q_tiles_buffered - 1) / q_tiles_buffered;
    using qk_bf_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using dq_chunk_tile = typename shared_ds_monolithic_globals<C>::dq_chunk_tile;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo, true, 64>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) dq_chunk_tile dq_smem[3][C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];
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
    const int q_block_start = kv_tile_base / q_tiles_buffered;
    const int dense_q_block_start = min(num_q_blocks, q_block_start + dense_q_block_lag);

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, 64> dk0_accum;
    rt_fl<kRefTileM, 64> dk1_accum;
    rt_fl<kRefTileM, 64> dk2_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk0_accum);
    warp::zero(dk1_accum);
    warp::zero(dk2_accum);
    warp::zero(dv_accum);

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<qk_bf_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<v_bf_tile, dim::DEPTH>();
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
    }
    __syncthreads();

    for (int q_block_idx = q_block_start; q_block_idx < dense_q_block_start; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - q_block_start;
        const int phase = local_q_iter & 1;
        if (warp == C::WarpTiles - 2) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == C::WarpTiles - 1) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < q_tiles_buffered) {
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_log2_vec);
            warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_vec);
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        const int first_active_subtile = max(0, kv_subtile_idx - q_tile_base);
        const int diagonal_subtile = kv_subtile_idx - q_tile_base;
        #pragma unroll
        for (int subtile = first_active_subtile; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, 64> dq0, dq1, dq2;
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            if (subtile != diagonal_subtile) {
                backward_tile_step_compact_fully_chunked<C, true, true>(
                    dq0,
                    dq1,
                    dq2,
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len
                );
            } else {
                backward_tile_step_compact_fully_chunked<C, true, false>(
                    dq0,
                    dq1,
                    dq2,
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len
                );
            }
            if (q_block_idx != q_block_start || subtile != first_active_subtile) {
                warp::tma::store_async_read_wait();
            }
            warp::store(dq_smem[0][warp], dq0);
            warp::store(dq_smem[1][warp], dq1);
            warp::store(dq_smem[2][warp], dq2);
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq0, dq_smem[0][warp], {batch_idx, q_tile_idx, head_idx, 0});
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq1, dq_smem[1][warp], {batch_idx, q_tile_idx, head_idx, 1});
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq2, dq_smem[2][warp], {batch_idx, q_tile_idx, head_idx, 2});
        }
        warp::tma::store_commit_group();
        __syncthreads();
    }

    for (int q_block_idx = dense_q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - q_block_start;
        const int phase = local_q_iter & 1;
        if (warp == C::WarpTiles - 2) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == C::WarpTiles - 1) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < q_tiles_buffered) {
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_log2_vec);
            warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_vec);
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, 64> dq0, dq1, dq2;
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            backward_tile_step_compact_fully_chunked<C, true, true>(
                dq0,
                dq1,
                dq2,
                dk0_accum,
                dk1_accum,
                dk2_accum,
                dv_accum,
                q_reg,
                k_reg,
                v_reg,
                do_reg,
                lse_log2_vec,
                dpsum_vec,
                g.scale,
                g.scale_log2e,
                q_tile_idx,
                kv_subtile_idx,
                g.seq_len
            );
            if (q_block_idx != dense_q_block_start || subtile != 0) {
                warp::tma::store_async_read_wait();
            }
            warp::store(dq_smem[0][warp], dq0);
            warp::store(dq_smem[1][warp], dq1);
            warp::store(dq_smem[2][warp], dq2);
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq0, dq_smem[0][warp], {batch_idx, q_tile_idx, head_idx, 0});
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq1, dq_smem[1][warp], {batch_idx, q_tile_idx, head_idx, 1});
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq2, dq_smem[2][warp], {batch_idx, q_tile_idx, head_idx, 2});
        }
        warp::tma::store_commit_group();
        __syncthreads();
    }

    warp::tma::store_async_read_wait();
    rt_fl<kRefTileM, C::Dqk> dk_accum;
    tkfa4::bwd_cute16_kernel::detail::stitch_three_chunks(dk_accum, dk0_accum, dk1_accum, dk2_accum);
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C, typename DkdvOutT = float>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_fullseq_dkdv_only(const __grid_constant__ dkdv_only_globals<C, DkdvOutT> g) {
    constexpr int q_tiles_buffered = (C::WarpTiles == 8 ? 8 : 4);
    constexpr int dense_q_block_lag = (C::WarpTiles + q_tiles_buffered - 1) / q_tiles_buffered;
    using qk_bf_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo, true, 64>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];

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
    const int q_block_start = kv_tile_base / q_tiles_buffered;
    const int dense_q_block_start = min(num_q_blocks, q_block_start + dense_q_block_lag);

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, 64> dk0_accum;
    rt_fl<kRefTileM, 64> dk1_accum;
    rt_fl<kRefTileM, 64> dk2_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk0_accum);
    warp::zero(dk1_accum);
    warp::zero(dk2_accum);
    warp::zero(dv_accum);

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<qk_bf_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<v_bf_tile, dim::DEPTH>();
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
    }
    __syncthreads();

    for (int q_block_idx = q_block_start; q_block_idx < dense_q_block_start; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - q_block_start;
        const int phase = local_q_iter & 1;
        if (warp == C::WarpTiles - 2) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == C::WarpTiles - 1) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < C::WarpTiles) {
            #pragma unroll
            for (int stats_tile = warp; stats_tile < q_tiles_buffered; stats_tile += C::WarpTiles) {
                stats_vec lse_log2_vec, dpsum_vec;
                warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + stats_tile});
                warp::store(lse_log2_smem[stats_tile], lse_log2_vec);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + stats_tile});
                warp::store(dpsum_smem[stats_tile], dpsum_vec);
            }
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        const int first_active_subtile = max(0, kv_subtile_idx - q_tile_base);
        const int diagonal_subtile = kv_subtile_idx - q_tile_base;
        #pragma unroll
        for (int subtile = first_active_subtile; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);
            if (subtile != diagonal_subtile) {
                repair_dkdv_step_chunked<C, true, true>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len
                );
            } else {
                repair_dkdv_step_chunked<C, true, false>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len
                );
            }
        }
        __syncthreads();
    }

    for (int q_block_idx = dense_q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - q_block_start;
        const int phase = local_q_iter & 1;
        if (warp == C::WarpTiles - 2) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == C::WarpTiles - 1) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < C::WarpTiles) {
            #pragma unroll
            for (int stats_tile = warp; stats_tile < q_tiles_buffered; stats_tile += C::WarpTiles) {
                stats_vec lse_log2_vec, dpsum_vec;
                warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + stats_tile});
                warp::store(lse_log2_smem[stats_tile], lse_log2_vec);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + stats_tile});
                warp::store(dpsum_smem[stats_tile], dpsum_vec);
            }
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);
            repair_dkdv_step_chunked<C, true, true>(
                dk0_accum,
                dk1_accum,
                dk2_accum,
                dv_accum,
                q_reg,
                k_reg,
                v_reg,
                do_reg,
                lse_log2_vec,
                dpsum_vec,
                g.scale,
                g.scale_log2e,
                q_tile_idx,
                kv_subtile_idx,
                g.seq_len
            );
        }
        __syncthreads();
    }

    rt_fl<kRefTileM, C::Dqk> dk_accum;
    tkfa4::bwd_cute16_kernel::detail::stitch_three_chunks(dk_accum, dk0_accum, dk1_accum, dk2_accum);
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_fullseq_dkdv_only_store_ds(const __grid_constant__ dkdv_only_ds_globals<C> g) {
    constexpr int q_tiles_buffered = (C::WarpTiles == 8 ? 8 : 4);
    constexpr int dense_q_block_lag = (C::WarpTiles + q_tiles_buffered - 1) / q_tiles_buffered;
    using qk_bf_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo, true, 64>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;
    using ds_tile = typename dkdv_only_ds_globals<C>::ds_tile;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];

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
    const int q_block_start = kv_tile_base / q_tiles_buffered;
    const int dense_q_block_start = min(num_q_blocks, q_block_start + dense_q_block_lag);

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, 64> dk0_accum;
    rt_fl<kRefTileM, 64> dk1_accum;
    rt_fl<kRefTileM, 64> dk2_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk0_accum);
    warp::zero(dk1_accum);
    warp::zero(dk2_accum);
    warp::zero(dv_accum);

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<qk_bf_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<v_bf_tile, dim::DEPTH>();
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
    }
    __syncthreads();

    for (int q_block_idx = q_block_start; q_block_idx < dense_q_block_start; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - q_block_start;
        const int phase = local_q_iter & 1;
        if (warp == C::WarpTiles - 2) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == C::WarpTiles - 1) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < C::WarpTiles) {
            #pragma unroll
            for (int stats_tile = warp; stats_tile < q_tiles_buffered; stats_tile += C::WarpTiles) {
                stats_vec lse_log2_vec, dpsum_vec;
                warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + stats_tile});
                warp::store(lse_log2_smem[stats_tile], lse_log2_vec);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + stats_tile});
                warp::store(dpsum_smem[stats_tile], dpsum_vec);
            }
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        const int first_active_subtile = max(0, kv_subtile_idx - q_tile_base);
        const int diagonal_subtile = kv_subtile_idx - q_tile_base;
        #pragma unroll
        for (int subtile = first_active_subtile; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_bf<kRefTileM, kRefTileN> ds_bf;
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);
            if (subtile != diagonal_subtile) {
                repair_dkdv_step_chunked_store_ds<C, true, true>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    ds_bf,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len
                );
            } else {
                repair_dkdv_step_chunked_store_ds<C, true, false>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    ds_bf,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len
                );
            }
            warp::store<dim::DEPTH>(g.ds, ds_bf, {batch_idx, q_tile_idx, head_idx, kv_subtile_idx});
        }
        __syncthreads();
    }

    for (int q_block_idx = dense_q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - q_block_start;
        const int phase = local_q_iter & 1;
        if (warp == C::WarpTiles - 2) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == C::WarpTiles - 1) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < C::WarpTiles) {
            #pragma unroll
            for (int stats_tile = warp; stats_tile < q_tiles_buffered; stats_tile += C::WarpTiles) {
                stats_vec lse_log2_vec, dpsum_vec;
                warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + stats_tile});
                warp::store(lse_log2_smem[stats_tile], lse_log2_vec);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + stats_tile});
                warp::store(dpsum_smem[stats_tile], dpsum_vec);
            }
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_bf<kRefTileM, kRefTileN> ds_bf;
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);
            repair_dkdv_step_chunked_store_ds<C, true, true>(
                dk0_accum,
                dk1_accum,
                dk2_accum,
                dv_accum,
                ds_bf,
                q_reg,
                k_reg,
                v_reg,
                do_reg,
                lse_log2_vec,
                dpsum_vec,
                g.scale,
                g.scale_log2e,
                q_tile_idx,
                kv_subtile_idx,
                g.seq_len
            );
            warp::store<dim::DEPTH>(g.ds, ds_bf, {batch_idx, q_tile_idx, head_idx, kv_subtile_idx});
        }
        __syncthreads();
    }

    rt_fl<kRefTileM, C::Dqk> dk_accum;
    tkfa4::bwd_cute16_kernel::detail::stitch_three_chunks(dk_accum, dk0_accum, dk1_accum, dk2_accum);
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C>
__global__ __launch_bounds__(C::DkdvBlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_seq2048_exact_dkdv_only(const __grid_constant__ seq2048_exact_dkdv_globals<C> g) {
    constexpr int q_tiles_buffered = 2;
    constexpr int q_subtiles_buffered = q_tiles_buffered * C::QSubtiles;
    using q_tile = typename seq2048_exact_dkdv_globals<C>::q_tile;
    using do_tile = typename seq2048_exact_dkdv_globals<C>::do_tile;
    using stats_smem_tile = typename seq2048_exact_dkdv_globals<C>::stats_tile;
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    struct shared_storage {
        q_tile q_smem[q_tiles_buffered];
        do_tile do_smem[q_tiles_buffered];
        stats_smem_tile lse_log2_smem[q_subtiles_buffered];
        stats_smem_tile dpsum_smem[q_subtiles_buffered];
    };

    __shared__ alignas(1024) shared_storage smem;
    auto &q_smem = smem.q_smem;
    auto &do_smem = smem.do_smem;
    auto &lse_log2_smem = smem.lse_log2_smem;
    auto &dpsum_smem = smem.dpsum_smem;

    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];

    const int warp = kittens::warpid();
    const bool is_compute = warp < C::ComputeWarps;
    constexpr int kQLoadWarpId = C::ComputeWarps - 2;
    constexpr int kDoLoadWarpId = C::ComputeWarps - 1;
    const bool is_q_load = warp == kQLoadWarpId;
    const bool is_do_load = warp == kDoLoadWarpId;
    const int consumer_idx = is_compute ? (warp / kittens::WARPGROUP_WARPS) : -1;
    const int consumer_warp = is_compute ? (warp % kittens::WARPGROUP_WARPS) : -1;

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
    const int q_start_block = kv_tile_base;
    const int kv_subtile_idx = is_compute
        ? (kv_tile_base + consumer_idx) * C::QSubtiles + consumer_warp
        : -1;

    rt_bf<kRefTileM, C::Dqk> k_reg, q_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    rt_fl<kRefTileM, 64> dk0_accum, dk1_accum, dk2_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();

        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
    }
    __syncthreads();

    if (is_compute) {
        warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::zero(dk0_accum);
        warp::zero(dk1_accum);
        warp::zero(dk2_accum);
        warp::zero(dv_accum);
    }

    if (is_q_load) {
        warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
        #pragma unroll
        for (int tile_slot = 0; tile_slot < q_tiles_buffered; ++tile_slot) {
            const int q_block_idx = q_start_block + tile_slot;
            coord<q_tile> q_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[tile_slot], g.q, q_tile_idx, q_b[0]);
        }
    }
    if (is_do_load) {
        warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
        #pragma unroll
        for (int tile_slot = 0; tile_slot < q_tiles_buffered; ++tile_slot) {
            const int q_block_idx = q_start_block + tile_slot;
            coord<do_tile> do_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[tile_slot], g.dout, do_tile_idx, o_b[0]);
        }
    }
    if (warp < q_subtiles_buffered) {
        const int tile_slot = warp / C::QSubtiles;
        const int subtile = warp % C::QSubtiles;
        const int q_block_idx = q_start_block + tile_slot;
        const int q_tile_base = q_block_idx * C::QSubtiles;
        typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_vec, dpsum_vec;
        warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
        warp::store(lse_log2_smem[warp], lse_vec);
        warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
        warp::store(dpsum_smem[warp], dpsum_vec);
    }
    __syncthreads();

    auto prefetch_q_block_group = [&](int block_base) {
        if (block_base >= q_blocks) {
            return;
        }
        if (is_q_load) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int tile_slot = 0; tile_slot < q_tiles_buffered; ++tile_slot) {
                const int q_block_load_idx = block_base + tile_slot;
                coord<q_tile> next_q_tile_idx = {batch_idx, q_block_load_idx, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[tile_slot], g.q, next_q_tile_idx, q_b[0]);
            }
        }
        if (is_do_load) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int tile_slot = 0; tile_slot < q_tiles_buffered; ++tile_slot) {
                const int q_block_load_idx = block_base + tile_slot;
                coord<do_tile> next_do_tile_idx = {batch_idx, q_block_load_idx, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[tile_slot], g.dout, next_do_tile_idx, o_b[0]);
            }
        }
        if (warp < q_subtiles_buffered) {
            const int tile_slot = warp / C::QSubtiles;
            const int subtile = warp % C::QSubtiles;
            const int q_block_load_idx = block_base + tile_slot;
            const int next_q_tile_base = q_block_load_idx * C::QSubtiles;
            typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_vec, dpsum_vec;
            warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, next_q_tile_base + subtile});
            warp::store(lse_log2_smem[warp], lse_vec);
            warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, next_q_tile_base + subtile});
            warp::store(dpsum_smem[warp], dpsum_vec);
        }
    };

    const int current_phase = 0;
    if (is_compute) {
        wait(q_b[0], current_phase);
        wait(o_b[0], current_phase);

        #pragma unroll 1
        for (int tile_slot = consumer_idx; tile_slot < q_tiles_buffered; ++tile_slot) {
            const int q_block_tile_idx = q_start_block + tile_slot;
            int q_subtile_begin = 0;
            if (tile_slot == consumer_idx) {
                const int q_subtile = consumer_warp;
                const int q_tile_idx = q_block_tile_idx * C::QSubtiles + q_subtile;
                auto q_subtile_smem = q_smem[tile_slot].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
                auto do_subtile_smem = do_smem[tile_slot].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
                stats_vec lse_vec, dpsum_vec;
                const int smem_stats_idx = tile_slot * C::QSubtiles + q_subtile;
                warp::load(q_reg, q_subtile_smem);
                warp::load(do_reg, do_subtile_smem);
                warp::load(lse_vec, lse_log2_smem[smem_stats_idx]);
                warp::load(dpsum_vec, dpsum_smem[smem_stats_idx]);

                repair_dkdv_step_chunked<C, true, false>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
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
                    g.seq_len
                );
                q_subtile_begin = consumer_warp + 1;
            }
            #pragma unroll 1
            for (int q_subtile = q_subtile_begin; q_subtile < C::QSubtiles; ++q_subtile) {
                const int q_tile_idx = q_block_tile_idx * C::QSubtiles + q_subtile;
                auto q_subtile_smem = q_smem[tile_slot].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
                auto do_subtile_smem = do_smem[tile_slot].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
                stats_vec lse_vec, dpsum_vec;
                const int smem_stats_idx = tile_slot * C::QSubtiles + q_subtile;
                warp::load(q_reg, q_subtile_smem);
                warp::load(do_reg, do_subtile_smem);
                warp::load(lse_vec, lse_log2_smem[smem_stats_idx]);
                warp::load(dpsum_vec, dpsum_smem[smem_stats_idx]);

                repair_dkdv_step_chunked<C, true, true>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
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
                    g.seq_len
                );
            }
        }
    }
    __syncthreads();

    const int first_dense_q_block = q_start_block + q_tiles_buffered;
    prefetch_q_block_group(first_dense_q_block);
    __syncthreads();

    for (int q_block_idx = first_dense_q_block; q_block_idx < q_blocks; q_block_idx += q_tiles_buffered) {
        const int local_q_iter = 1 + ((q_block_idx - first_dense_q_block) / q_tiles_buffered);
        const int current_phase = local_q_iter & 1;

        if (is_compute) {
            wait(q_b[0], current_phase);
            wait(o_b[0], current_phase);

            #pragma unroll 1
            for (int tile_slot = 0; tile_slot < q_tiles_buffered; ++tile_slot) {
                const int q_block_tile_idx = q_block_idx + tile_slot;
                #pragma unroll 1
                for (int q_subtile = 0; q_subtile < C::QSubtiles; ++q_subtile) {
                    const int q_tile_idx = q_block_tile_idx * C::QSubtiles + q_subtile;
                    auto q_subtile_smem = q_smem[tile_slot].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
                    auto do_subtile_smem = do_smem[tile_slot].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
                    stats_vec lse_vec, dpsum_vec;
                    const int smem_stats_idx = tile_slot * C::QSubtiles + q_subtile;
                    warp::load(q_reg, q_subtile_smem);
                    warp::load(do_reg, do_subtile_smem);
                    warp::load(lse_vec, lse_log2_smem[smem_stats_idx]);
                    warp::load(dpsum_vec, dpsum_smem[smem_stats_idx]);

                    repair_dkdv_step_chunked<C, true, true>(
                        dk0_accum,
                        dk1_accum,
                        dk2_accum,
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
                        g.seq_len
                    );
                }
            }
        }
        __syncthreads();

        prefetch_q_block_group(q_block_idx + q_tiles_buffered);
        __syncthreads();
    }

    if (is_compute) {
        rt_fl<kRefTileM, C::Dqk> dk_full_reg;
        tkfa4::bwd_cute16_kernel::detail::stitch_three_chunks(dk_full_reg, dk0_accum, dk1_accum, dk2_accum);
        warp::store<dim::DEPTH>(g.dk, dk_full_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    }
}

template <
    typename C,
    int DenseSplitCount = 1,
    bool UseTmemScoreDp = false,
    bool FrontierOnly = false,
    bool FuseDenseDq = false,
    bool AdaptiveLastQuarter = false,
    bool OverlapLoadAndDqReduce = false,
    bool SkipAdaptiveTailScratch = false,
    bool UseLdsmTransposeDs = false,
    bool DoubleBufferFusedDqTma = false,
    bool ReleaseTmemOperandsEachIteration = false
>
__global__ __launch_bounds__(
    FuseDenseDq
        ? (OverlapLoadAndDqReduce
            ? C::FusedDqLoadOverlapBlockThreads
            : C::FusedDqBlockThreads)
        : C::DenseBlockThreads,
    C::MinBlocksPerSm
)
void main_kernel_causal_dense_tmem_dkdv(
    const __grid_constant__ dense_tmem_frontier_globals<C> g,
    typename dense_tmem_frontier_globals<C>::dk_gl split_dk,
    typename dense_tmem_frontier_globals<C>::dv_gl split_dv
) {
    constexpr bool RunFusedDqProducer = FuseDenseDq;
    constexpr bool RunFusedDqReducer = FuseDenseDq;
    static_assert(
        DenseSplitCount == 1 || DenseSplitCount == 2 || DenseSplitCount == 3 ||
            DenseSplitCount == 4 || DenseSplitCount == 8,
        "Dense split count must be 1, 2, 3, 4, or 8"
    );
    static_assert(!FrontierOnly || DenseSplitCount == 1, "Frontier-only TMEM does not use splits");
    static_assert(!FrontierOnly || UseTmemScoreDp, "Frontier-only TMEM requires TMEM score/dP");
    static_assert(!FuseDenseDq || UseTmemScoreDp, "Fused dQ requires the validated TMEM score/dP path");
    static_assert(
        !AdaptiveLastQuarter || (DenseSplitCount == 2 && !FrontierOnly),
        "Adaptive last-quarter ownership requires split-2 dense main"
    );
    static_assert(
        !OverlapLoadAndDqReduce || FuseDenseDq,
        "Load/reducer overlap requires fused dQ"
    );
    static_assert(
        !SkipAdaptiveTailScratch || AdaptiveLastQuarter,
        "Skipping tail scratch requires adaptive last-quarter ownership"
    );
    static_assert(
        !DoubleBufferFusedDqTma || FuseDenseDq,
        "Double-buffered dQ TMA drain requires fused dQ"
    );
    using q_tile = typename dense_tmem_frontier_globals<C>::q_tile;
    using k_tile = typename dense_tmem_frontier_globals<C>::k_tile;
    using v_tile = typename dense_tmem_frontier_globals<C>::v_tile;
    using do_tile = typename dense_tmem_frontier_globals<C>::do_tile;
    using stats_smem_tile = typename dense_tmem_frontier_globals<C>::stats_tile;
    using qk_block_smem_tile = st_bf<kRefTileM, C::TileRows>;
    using qk_full_smem_tile = st_bf<C::TileRows, C::TileRows>;
    using dq_chunk_tile = typename dense_tmem_frontier_globals<C>::dq_chunk_tile;
    using attn_tt = half_tt_fl<C::TileRows>;
    using dk_tt = half_tt_fl<64>;
    using dv_tt = half_tt_fl<C::Dvo>;
    using dq_tt = half_tt_fl<C::Dqk>;

    union ds_store_smem {
        qk_full_smem_tile full[C::ConsumerWarpgroups];
        qk_block_smem_tile warp[C::ConsumerWarpgroups][WARPGROUP_WARPS];
    };

    struct shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[2];
        do_tile do_smem[2];
        qk_block_smem_tile p_q_smem[C::ConsumerWarpgroups][WARPGROUP_WARPS];
        ds_store_smem ds_smem;
        stats_smem_tile lse_log2_smem[2][C::QSubtiles];
        stats_smem_tile dpsum_smem[2][C::QSubtiles];
    };

    __shared__ alignas(1024) shared_storage smem;
    auto &k_smem = smem.k_smem;
    auto &v_smem = smem.v_smem;
    auto &q_smem = smem.q_smem;
    auto &do_smem = smem.do_smem;
    auto &p_q_smem = smem.p_q_smem;
    auto &ds_q_smem = smem.ds_smem.warp;
    auto &ds_full_smem = smem.ds_smem.full;
    auto &lse_log2_smem = smem.lse_log2_smem;
    auto &dpsum_smem = smem.dpsum_smem;

    __shared__ __align__(16) kittens::semaphore kv_b;
    __shared__ __align__(16) kittens::semaphore q_b[2];
    __shared__ __align__(16) kittens::semaphore o_b[2];
    __shared__ __align__(16) kittens::semaphore output_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore score_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore dp_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore operand_consumed[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore fused_dq_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore fused_dq_empty;

    const int warp = kittens::warpid();
    const bool is_compute = warp < C::ComputeWarps;
    const bool is_load = warp == C::ComputeWarps;
    constexpr int FusedDqReduceWarpBase = OverlapLoadAndDqReduce
        ? C::FusedDqLoadOverlapReduceWarpBase
        : C::FusedDqReduceWarpBase;
    const bool is_fused_dq_reduce =
        RunFusedDqReducer &&
        warp >= FusedDqReduceWarpBase &&
        warp < FusedDqReduceWarpBase + C::FusedDqReduceWarps;
    const int consumer_idx = is_compute ? warp / WARPGROUP_WARPS : -1;

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = cluster_ctarank();
    const int cluster_group = static_cast<int>(blockIdx.x) / C::ClusterSize;
    const int num_k_blocks = g.seq_len / (C::TileRows * C::ConsumerWarpgroups);
    int split_idx;
    int kv_cluster_group;
    int active_dense_split_count;
    if constexpr (AdaptiveLastQuarter) {
        const int num_k_cluster_groups = num_k_blocks / C::ClusterSize;
        const int split_cluster_groups = (num_k_cluster_groups * 3) / 4;
        const int split_launch_groups = split_cluster_groups * DenseSplitCount;
        if (cluster_group < split_launch_groups) {
            split_idx = cluster_group % DenseSplitCount;
            kv_cluster_group = cluster_group / DenseSplitCount;
            active_dense_split_count = DenseSplitCount;
        } else {
            split_idx = 0;
            kv_cluster_group = split_cluster_groups + cluster_group - split_launch_groups;
            active_dense_split_count = 1;
        }
    } else {
        split_idx = cluster_group % DenseSplitCount;
        kv_cluster_group = cluster_group / DenseSplitCount;
        active_dense_split_count = DenseSplitCount;
    }
    const int kv_block_idx = kv_cluster_group * C::ClusterSize + cluster_rank;
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::ConsumerWarpgroups;
    const int q_blocks = g.seq_len / C::TileRows;
    const int dense_q_start = kv_tile_base + C::ConsumerWarpgroups;
    const int first_dense_q_block = FrontierOnly ? kv_tile_base : dense_q_start + split_idx;
    const int dense_q_stride = FrontierOnly ? 1 : active_dense_split_count;
    const int q_block_limit = FrontierOnly
        ? ((kv_tile_base + C::ConsumerWarpgroups) < q_blocks
            ? kv_tile_base + C::ConsumerWarpgroups
            : q_blocks)
        : q_blocks;

    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dk_tt dk0_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk1_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk2_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dv_tt dv_tt_accum[C::ConsumerWarpgroups] = {dv_tt{0}, dv_tt{0}};
    dq_tt fused_dq_tt[C::ConsumerWarpgroups] = {dq_tt{0}, dq_tt{0}};

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.k.template prefetch_tma<k_tile, dim::DEPTH>();
        g.v.template prefetch_tma<v_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        init_semaphore(kv_b, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(q_b[stage], 0, 1);
            init_semaphore(o_b[stage], 0, 1);
        }
        #pragma unroll
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            init_semaphore(output_ready[w], 0, 1);
            if constexpr (UseTmemScoreDp) {
                init_semaphore(score_ready[w], 0, 1);
                init_semaphore(dp_ready[w], 0, 1);
            }
            if constexpr (ReleaseTmemOperandsEachIteration) {
                init_semaphore(operand_consumed[w], 0, 1);
            }
            if constexpr (RunFusedDqProducer) {
                init_semaphore(fused_dq_ready[w], 0, 1);
            }
        }
        if constexpr (RunFusedDqReducer) {
            init_semaphore(fused_dq_empty, 0, C::FusedDqReduceWarps);
        }
    }
    __syncthreads();

    if (is_compute) {
        score_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
        dp_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, C::TileRows);
        if constexpr (FuseDenseDq) {
            fused_dq_tt[consumer_idx] = tm_alloc.template allocate<dq_tt>(consumer_idx, 0);
            dk0_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 3 * C::TileRows);
            dk1_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 4 * C::TileRows);
            dk2_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 5 * C::TileRows);
            dv_tt_accum[consumer_idx] = tm_alloc.template allocate<dv_tt>(consumer_idx, 6 * C::TileRows);
        } else {
            dk0_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 2 * C::TileRows);
            dk1_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 3 * C::TileRows);
            dk2_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 4 * C::TileRows);
            dv_tt_accum[consumer_idx] = tm_alloc.template allocate<dv_tt>(consumer_idx, 5 * C::TileRows);
        }

        rt_fl<kRefTileM, 64> zero_dk;
        rt_fl<kRefTileM, C::Dvo> zero_dv;
        warp::zero(zero_dk);
        warp::zero(zero_dv);
        warpgroup::store_async(dk0_tt[consumer_idx], zero_dk);
        warpgroup::store_async(dk1_tt[consumer_idx], zero_dk);
        warpgroup::store_async(dk2_tt[consumer_idx], zero_dk);
        warpgroup::store_async(dv_tt_accum[consumer_idx], zero_dv);
        tensor_store_wait();
    }

    if (threadIdx.x == 0) {
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

    if (is_load && first_dense_q_block < q_blocks) {
        coord<q_tile> q_tile_idx = {batch_idx, first_dense_q_block, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, first_dense_q_block, head_idx, 0};
        warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
        warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx, q_b[0]);
        warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
        warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, do_tile_idx, o_b[0]);

        const int q_tile_base = first_dense_q_block * C::QSubtiles;
        #pragma unroll
        for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
            typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_vec, dpsum_vec;
            warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
            warp::store(lse_log2_smem[0][subtile], lse_vec);
            warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
            warp::store(dpsum_smem[0][subtile], dpsum_vec);
        }
    }
    __syncthreads();

    for (
        int q_block_idx = first_dense_q_block;
        q_block_idx < q_block_limit;
        q_block_idx += dense_q_stride
    ) {
        const int iteration = (q_block_idx - first_dense_q_block) / dense_q_stride;
        const int stage = iteration & 1;
        const int phase = (iteration >> 1) & 1;
        const int next_q_block_idx = q_block_idx + dense_q_stride;
        const int next_stage = stage ^ 1;
        if (is_load && next_q_block_idx < q_block_limit) {
            coord<q_tile> q_tile_idx = {batch_idx, next_q_block_idx, head_idx, 0};
            coord<do_tile> do_tile_idx = {batch_idx, next_q_block_idx, head_idx, 0};
            warp::tma::expect_bytes(q_b[next_stage], sizeof(q_smem[next_stage]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                q_smem[next_stage],
                g.q,
                q_tile_idx,
                q_b[next_stage]
            );
            warp::tma::expect_bytes(o_b[next_stage], sizeof(do_smem[next_stage]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                do_smem[next_stage],
                g.dout,
                do_tile_idx,
                o_b[next_stage]
            );

            const int q_tile_base = next_q_block_idx * C::QSubtiles;
            #pragma unroll
            for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_vec, dpsum_vec;
                warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(lse_log2_smem[next_stage][subtile], lse_vec);
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[next_stage][subtile], dpsum_vec);
            }
        }

        if (is_compute) {
            if constexpr (RunFusedDqReducer) {
                if (iteration > 0) {
                    wait(fused_dq_empty, (iteration - 1) & 1);
                }
            }
            rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
            rt_bf<kRefTileM, C::TileRows> p_block_t_mma, ds_block_t_mma;
            using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
            using prob_tt = half_tt_bf<C::TileRows>;
            const int q_subtile = warpgroup::warpid();
            wait(kv_b, 0);
            if constexpr (UseTmemScoreDp) {
                const int kv_tile_idx = kv_tile_base + consumer_idx;
                if constexpr (FrontierOnly) {
                    if (q_block_idx > kv_tile_idx) {
                        bwd_hot::detail::hot_compute_dq_qmajor_loop<false, C>(
                            q_b[stage], o_b[stage], score_ready[consumer_idx],
                            dp_ready[consumer_idx], p_block_t, dp_block_t, ds_block_t,
                            ds_block_t_mma, score_tt[consumer_idx], dp_tt[consumer_idx],
                            q_smem, k_smem, v_smem, do_smem, ds_q_smem,
                            lse_log2_smem[stage], dpsum_smem[stage], g.scale,
                            g.scale_log2e, iteration & 1, q_block_idx, kv_tile_idx,
                            stage, phase
                        );
                    } else {
                        bwd_hot::detail::hot_compute_dq_qmajor_loop<true, C>(
                            q_b[stage], o_b[stage], score_ready[consumer_idx],
                            dp_ready[consumer_idx], p_block_t, dp_block_t, ds_block_t,
                            ds_block_t_mma, score_tt[consumer_idx], dp_tt[consumer_idx],
                            q_smem, k_smem, v_smem, do_smem, ds_q_smem,
                            lse_log2_smem[stage], dpsum_smem[stage], g.scale,
                            g.scale_log2e, iteration & 1, q_block_idx, kv_tile_idx,
                            stage, phase
                        );
                    }
                } else {
                    bwd_hot::detail::hot_compute_dq_qmajor_loop<false, C>(
                        q_b[stage], o_b[stage], score_ready[consumer_idx],
                        dp_ready[consumer_idx], p_block_t, dp_block_t, ds_block_t,
                        ds_block_t_mma, score_tt[consumer_idx], dp_tt[consumer_idx],
                        q_smem, k_smem, v_smem, do_smem, ds_q_smem,
                        lse_log2_smem[stage], dpsum_smem[stage], g.scale,
                        g.scale_log2e, iteration & 1, q_block_idx, kv_tile_idx,
                        stage, phase
                    );
                }
                warp::copy(p_block_t_mma, p_block_t);
                warp::store(p_q_smem[consumer_idx][q_subtile], p_block_t_mma);
                group<8>::sync(9);
            } else {
                wait(q_b[stage], phase);
                wait(o_b[stage], phase);

                rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
                rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
                stats_vec lse_vec, dpsum_vec;
                auto q_subtile_smem = q_smem[stage].template subtile<kRefTileM, C::Dqk>({q_subtile, 0});
                auto do_subtile_smem = do_smem[stage].template subtile<kRefTileM, C::Dvo>({q_subtile, 0});
                warp::load(q_reg, q_subtile_smem);
                warp::load(do_reg, do_subtile_smem);
                warp::load(lse_vec, lse_log2_smem[stage][q_subtile]);
                warp::load(dpsum_vec, dpsum_smem[stage][q_subtile]);

                warp::zero(p_block_t);
                warp::zero(dp_block_t);
                warp::zero(ds_block_t);
                #pragma unroll
                for (int kv_subtile = 0; kv_subtile < C::QSubtiles; ++kv_subtile) {
                    rt_fl<kRefTileM, kRefTileN> p_sub, dp_sub, ds_sub;
                    auto k_subtile_smem = k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({kv_subtile, 0});
                    auto v_subtile_smem = v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({kv_subtile, 0});
                    warp::load(k_reg, k_subtile_smem);
                    warp::load(v_reg, v_subtile_smem);
                    bwd_fa4::detail::reconstruct_probability_tile_dense<C>(
                        p_sub,
                        q_reg,
                        k_reg,
                        lse_vec,
                        g.scale_log2e
                    );
                    warp::zero(dp_sub);
                    warp::mma_ABt(dp_sub, do_reg, v_reg, dp_sub);
                    warp::sub_row(dp_sub, dp_sub, dpsum_vec);
                    warp::mul(ds_sub, p_sub, dp_sub);
                    warp::mul(ds_sub, ds_sub, g.scale);
                    bwd_hot::detail::insert_block(p_block_t, p_sub, kv_subtile);
                    bwd_hot::detail::insert_block(dp_block_t, dp_sub, kv_subtile);
                    bwd_hot::detail::insert_block(ds_block_t, ds_sub, kv_subtile);
                }
                warp::copy(p_block_t_mma, p_block_t);
                warp::copy(ds_block_t_mma, ds_block_t);
                warp::store(p_q_smem[consumer_idx][q_subtile], p_block_t_mma);
                warp::store(ds_q_smem[consumer_idx][q_subtile], ds_block_t_mma);
                group<8>::sync(9);
            }

            warp::zero(p_block_t_mma);
            warp::zero(ds_block_t_mma);
            #pragma unroll
            for (int source_q_subtile = 0; source_q_subtile < C::QSubtiles; ++source_q_subtile) {
                auto p_qk_subtile = p_q_smem[consumer_idx][source_q_subtile]
                    .template subtile<kRefTileM, kRefTileN>({0, q_subtile});
                auto ds_qk_subtile = ds_q_smem[consumer_idx][source_q_subtile]
                    .template subtile<kRefTileM, kRefTileN>({0, q_subtile});
                if constexpr (UseLdsmTransposeDs) {
                    using col_subtile = rt_bf<
                        kRefTileM,
                        kRefTileN,
                        ducks::rt_layout::col
                    >;
                    using row_subtile = rt_bf<kRefTileM, kRefTileN>;
                    static_assert(sizeof(col_subtile) == sizeof(row_subtile));
                    row_subtile p_sub;
                    col_subtile ds_sub_qmajor;
                    warp::load(p_sub, p_qk_subtile);
                    warp::load(ds_sub_qmajor, ds_qk_subtile);
                    warp::transpose_inplace(p_sub);

                    // LDSM.MT produces the row-layout representation of transposed dS.
                    const auto &ds_sub_t = reinterpret_cast<const row_subtile &>(ds_sub_qmajor);
                    bwd_hot::detail::insert_block(
                        p_block_t_mma,
                        p_sub,
                        source_q_subtile
                    );
                    bwd_hot::detail::insert_block(
                        ds_block_t_mma,
                        ds_sub_t,
                        source_q_subtile
                    );
                } else {
                    rt_bf<kRefTileM, kRefTileN> p_sub, ds_sub;
                    warp::load(p_sub, p_qk_subtile);
                    warp::load(ds_sub, ds_qk_subtile);
                    warp::transpose_inplace(p_sub);
                    warp::transpose_inplace(ds_sub);
                    bwd_hot::detail::insert_block(p_block_t_mma, p_sub, source_q_subtile);
                    bwd_hot::detail::insert_block(ds_block_t_mma, ds_sub, source_q_subtile);
                }
            }

            const prob_tt probs_tmem = prob_tt{score_tt[consumer_idx].addr};
            const prob_tt ds_tmem = prob_tt{dp_tt[consumer_idx].addr};
            if constexpr (ReleaseTmemOperandsEachIteration) {
                if (iteration > 0) {
                    wait(operand_consumed[consumer_idx], (iteration - 1) & 1);
                }
            }
            warpgroup::store_async(probs_tmem, p_block_t_mma);
            warpgroup::store_async(ds_tmem, ds_block_t_mma);
            tensor_store_wait();

            auto &q_smem_0 = q_smem[stage].template subtile<64>(0);
            auto &q_smem_1 = q_smem[stage].template subtile<64>(1);
            auto &q_smem_2 = q_smem[stage].template subtile<64>(2);
            warpgroup::mma_AB(dv_tt_accum[consumer_idx], probs_tmem, do_smem[stage]);
            warpgroup::mma_AB(dk0_tt[consumer_idx], ds_tmem, q_smem_0);
            warpgroup::mma_AB(dk1_tt[consumer_idx], ds_tmem, q_smem_1);
            warpgroup::mma_AB(dk2_tt[consumer_idx], ds_tmem, q_smem_2);
            if constexpr (RunFusedDqProducer) {
                warpgroup::mm_AB(
                    fused_dq_tt[consumer_idx],
                    ds_full_smem[consumer_idx],
                    k_smem[consumer_idx],
                    fused_dq_ready[consumer_idx]
                );
            }
            group<8>::sync(10);
            if constexpr (ReleaseTmemOperandsEachIteration) {
                if (
                    next_q_block_idx < q_block_limit &&
                    warpgroup::laneid() == 0
                ) {
                    tensor_commit<1>(operand_consumed[consumer_idx]);
                }
            }
        }
        if constexpr (RunFusedDqReducer) {
            if (is_fused_dq_reduce) {
                const int dq_subtile_idx = warp - FusedDqReduceWarpBase;
                const int fused_dq_phase = iteration & 1;
                wait(fused_dq_ready[0], fused_dq_phase);
                wait(fused_dq_ready[1], fused_dq_phase);

                const dq_tt local_dq = tm_alloc.template allocate<dq_tt>(0, 0);
                const dq_tt peer_dq = tm_alloc.template allocate<dq_tt>(1, 0);
                auto *dq_smem_stage0 = reinterpret_cast<dq_chunk_tile *>(&p_q_smem[0][0]);
                auto *dq_smem_stage1 = reinterpret_cast<dq_chunk_tile *>(&ds_full_smem[0]);
                const int q_output_subtile = OverlapLoadAndDqReduce
                    ? dq_subtile_idx
                    : (dq_subtile_idx + 1) % C::QSubtiles;
                const int q_tile_idx = q_block_idx * C::QSubtiles + q_output_subtile;
                const uint32_t warp_row_offset = (32 * dq_subtile_idx) << 16;
                #pragma unroll
                for (int chunk = 0; chunk < 3; ++chunk) {
                    rt_fl<kRefTileM, 64> dq_chunk, dq_peer_chunk;
                    using dq_warp_chunk_tt = tt_fl<kRefTileM, 64>;
                    const dq_warp_chunk_tt dq_local_tt{
                        local_dq.addr + warp_row_offset + chunk * 64
                    };
                    const dq_warp_chunk_tt dq_peer_tt{
                        peer_dq.addr + warp_row_offset + chunk * 64
                    };
                    warp::load_async(dq_chunk, dq_local_tt);
                    warp::load_async(dq_peer_chunk, dq_peer_tt);
                    tensor_load_wait();
                    warp::add(dq_chunk, dq_chunk, dq_peer_chunk);
                    if constexpr (DoubleBufferFusedDqTma) {
                        if (chunk >= 2) {
                            warp::tma::store_async_wait<1>();
                        }
                    } else if (chunk > 0) {
                        warp::tma::store_async_wait();
                    }
                    auto *dq_smem = DoubleBufferFusedDqTma && (chunk & 1)
                        ? dq_smem_stage1
                        : dq_smem_stage0;
                    warp::store(dq_smem[dq_subtile_idx], dq_chunk);
                    if (chunk == 0) {
                        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                            g.dq0,
                            dq_smem[dq_subtile_idx],
                            {batch_idx, q_tile_idx, head_idx, 0}
                        );
                    } else if (chunk == 1) {
                        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                            g.dq1,
                            dq_smem[dq_subtile_idx],
                            {batch_idx, q_tile_idx, head_idx, 1}
                        );
                    } else {
                        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                            g.dq2,
                            dq_smem[dq_subtile_idx],
                            {batch_idx, q_tile_idx, head_idx, 2}
                        );
                    }
                }
                warp::tma::store_async_wait();
                warp::arrive(fused_dq_empty);
            } else {
                if constexpr (OverlapLoadAndDqReduce) {
                    group<8>::sync(13);
                } else {
                    group<9>::sync(13);
                }
            }
        } else {
            if constexpr (RunFusedDqProducer) {
                if (is_compute) {
                    wait(fused_dq_ready[consumer_idx], iteration & 1);
                }
            }
            __syncthreads();
        }
    }

    if (is_compute) {
        rt_fl<kRefTileM, 64> dk0_reg, dk1_reg, dk2_reg;
        rt_fl<kRefTileM, C::Dqk> dk_reg;
        rt_fl<kRefTileM, C::Dvo> dv_reg;
        if (warpgroup::laneid() == 0) {
            tensor_commit<1>(output_ready[consumer_idx]);
        }
        wait(output_ready[consumer_idx], 0);
        warpgroup::load_async(dk0_reg, dk0_tt[consumer_idx]);
        warpgroup::load_async(dk1_reg, dk1_tt[consumer_idx]);
        warpgroup::load_async(dk2_reg, dk2_tt[consumer_idx]);
        warpgroup::load_async(dv_reg, dv_tt_accum[consumer_idx]);
        tensor_load_wait();

        const int kv_subtile_idx =
            (kv_tile_base + consumer_idx) * C::QSubtiles + warpgroup::warpid();
        tkfa4::bwd_cute16_kernel::detail::stitch_three_chunks(dk_reg, dk0_reg, dk1_reg, dk2_reg);
        if constexpr (FrontierOnly) {
            warp::store<dim::DEPTH>(g.dk, dk_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
            warp::store<dim::DEPTH>(g.dv, dv_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
        } else if constexpr (DenseSplitCount > 1) {
            if (split_idx == 0) {
                warp::store<dim::DEPTH>(g.dk, dk_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
                warp::store<dim::DEPTH>(g.dv, dv_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
                if constexpr (AdaptiveLastQuarter && !SkipAdaptiveTailScratch) {
                    if (active_dense_split_count == 1) {
                        warp::zero(dk_reg);
                        warp::zero(dv_reg);
                        warp::store<dim::DEPTH>(
                            split_dk,
                            dk_reg,
                            {batch_idx, kv_subtile_idx, head_idx, 0}
                        );
                        warp::store<dim::DEPTH>(
                            split_dv,
                            dv_reg,
                            {batch_idx, kv_subtile_idx, head_idx, 0}
                        );
                    }
                }
            } else {
                const int split_batch_idx = (split_idx - 1) * g.batch_size + batch_idx;
                warp::store<dim::DEPTH>(
                    split_dk,
                    dk_reg,
                    {split_batch_idx, kv_subtile_idx, head_idx, 0}
                );
                warp::store<dim::DEPTH>(
                    split_dv,
                    dv_reg,
                    {split_batch_idx, kv_subtile_idx, head_idx, 0}
                );
            }
        } else {
            warp::store<dim::DEPTH>(g.dk, dk_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
            warp::store<dim::DEPTH>(g.dv, dv_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
        }
    }
}

template <typename C, bool COMPUTE_DK, bool COMPUTE_DV, bool ACCUMULATE_EXISTING = true>
__global__ __launch_bounds__(C::FrontierBlockThreads, C::MinBlocksPerSm)
void causal_dense_tmem_frontier_patch_kernel(const __grid_constant__ dense_tmem_frontier_globals<C> g) {
    static_assert(COMPUTE_DK != COMPUTE_DV, "frontier patch must produce exactly one output");
    const int warp = kittens::warpid();
    const int consumer_idx = warp / WARPGROUP_WARPS;
    const int consumer_warp = warp % WARPGROUP_WARPS;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int patch_block_idx = static_cast<int>(blockIdx.x);
    const int dk_chunk_idx = COMPUTE_DK ? patch_block_idx % 3 : 0;
    const int kv_block_idx = COMPUTE_DK ? patch_block_idx / 3 : patch_block_idx;
    const int kv_tile_base = kv_block_idx * C::ConsumerWarpgroups;
    const int kv_subtile_idx =
        (kv_tile_base + consumer_idx) * C::QSubtiles + consumer_warp;

    rt_bf<kRefTileM, C::Dqk> k_reg, q_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    rt_fl<kRefTileM, 64> dk_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    if constexpr (COMPUTE_DK) {
        warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::zero(dk_accum);
    }
    if constexpr (COMPUTE_DV) {
        warp::zero(dv_accum);
    }

    const int diagonal_q_block = kv_tile_base + consumer_idx;
    #pragma unroll 1
    for (int q_subtile = consumer_warp; q_subtile < C::QSubtiles; ++q_subtile) {
        const int q_tile_idx = diagonal_q_block * C::QSubtiles + q_subtile;
        stats_vec lse_vec, dpsum_vec;
        warp::load<dim::DEPTH>(q_reg, g.q, {batch_idx, q_tile_idx, head_idx, 0});
        warp::load<dim::DEPTH>(do_reg, g.dout, {batch_idx, q_tile_idx, head_idx, 0});
        warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
        if constexpr (COMPUTE_DK) {
            warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
        }
        if (q_subtile == consumer_warp) {
            if constexpr (COMPUTE_DK) {
                repair_dk_step_one_chunk<C, true, false>(
                    dk_accum,
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
                    dk_chunk_idx
                );
            } else {
                repair_dkdv_step_chunked<C, true, false, false, true>(
                    dk_accum,
                    dk_accum,
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
                    g.seq_len
                );
            }
        } else {
            if constexpr (COMPUTE_DK) {
                repair_dk_step_one_chunk<C, true, true>(
                    dk_accum,
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
                    dk_chunk_idx
                );
            } else {
                repair_dkdv_step_chunked<C, true, true, false, true>(
                    dk_accum,
                    dk_accum,
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
                    g.seq_len
                );
            }
        }
    }

    if (consumer_idx == 0) {
        const int dense_frontier_q_block = kv_tile_base + 1;
        #pragma unroll 1
        for (int q_subtile = 0; q_subtile < C::QSubtiles; ++q_subtile) {
            const int q_tile_idx = dense_frontier_q_block * C::QSubtiles + q_subtile;
            stats_vec lse_vec, dpsum_vec;
            warp::load<dim::DEPTH>(q_reg, g.q, {batch_idx, q_tile_idx, head_idx, 0});
            warp::load<dim::DEPTH>(do_reg, g.dout, {batch_idx, q_tile_idx, head_idx, 0});
            warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
            if constexpr (COMPUTE_DK) {
                warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
            }
            if constexpr (COMPUTE_DK) {
                repair_dk_step_one_chunk<C, true, true>(
                    dk_accum,
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
                    dk_chunk_idx
                );
            } else {
                repair_dkdv_step_chunked<C, true, true, false, true>(
                    dk_accum,
                    dk_accum,
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
                    g.seq_len
                );
            }
        }
    }

    if constexpr (COMPUTE_DK) {
        if constexpr (ACCUMULATE_EXISTING) {
            rt_fl<kRefTileM, 64> dk_existing;
            warp::load<dim::DEPTH>(
                dk_existing,
                g.dk,
                {batch_idx, kv_subtile_idx, head_idx, dk_chunk_idx}
            );
            warp::add(dk_accum, dk_accum, dk_existing);
        }
        warp::store<dim::DEPTH>(
            g.dk,
            dk_accum,
            {batch_idx, kv_subtile_idx, head_idx, dk_chunk_idx}
        );
    }
    if constexpr (COMPUTE_DV) {
        if constexpr (ACCUMULATE_EXISTING) {
            rt_fl<kRefTileM, C::Dvo> dv_existing;
            warp::load<dim::DEPTH>(dv_existing, g.dv, {batch_idx, kv_subtile_idx, head_idx, 0});
            warp::add(dv_accum, dv_accum, dv_existing);
        }
        warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    }
}

__global__ __launch_bounds__(256, 2)
void dense_tmem_frontier_add_kernel(
    float4 *dk,
    const float4 *frontier_dk,
    int64_t dk_vecs,
    float4 *dv,
    const float4 *frontier_dv,
    int64_t dv_vecs
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dk_vecs) {
        const float4 lhs = dk[idx];
        const float4 rhs = frontier_dk[idx];
        dk[idx] = make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }
    if (idx < dv_vecs) {
        const float4 lhs = dv[idx];
        const float4 rhs = frontier_dv[idx];
        dv[idx] = make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }
}

__global__ __launch_bounds__(256, 2)
void dense_tmem_frontier_add_bf16_kernel(
    bf16_2 *dk,
    const float4 *frontier_dk,
    int64_t dk_vecs,
    bf16_2 *dv,
    const float4 *frontier_dv,
    int64_t dv_vecs
) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dk_vecs) {
        const float2 lo =
            base_types::convertor<float2, bf16_2>::convert(dk[2 * idx]);
        const float2 hi =
            base_types::convertor<float2, bf16_2>::convert(dk[2 * idx + 1]);
        const float4 rhs = frontier_dk[idx];
        dk[2 * idx] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(lo.x + rhs.x, lo.y + rhs.y)
        );
        dk[2 * idx + 1] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(hi.x + rhs.z, hi.y + rhs.w)
        );
    }
    if (idx < dv_vecs) {
        const float2 lo =
            base_types::convertor<float2, bf16_2>::convert(dv[2 * idx]);
        const float2 hi =
            base_types::convertor<float2, bf16_2>::convert(dv[2 * idx + 1]);
        const float4 rhs = frontier_dv[idx];
        dv[2 * idx] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(lo.x + rhs.x, lo.y + rhs.y)
        );
        dv[2 * idx + 1] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(hi.x + rhs.z, hi.y + rhs.w)
        );
    }
}

__global__ __launch_bounds__(256, 2)
void dense_tmem_two_scratch_add_kernel(
    float4 *dk,
    const float4 *dense_split_dk,
    const float4 *frontier_dk,
    int64_t dk_vecs,
    float4 *dv,
    const float4 *dense_split_dv,
    const float4 *frontier_dv,
    int64_t dv_vecs
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dk_vecs) {
        const float4 base = dk[idx];
        const float4 split = dense_split_dk[idx];
        const float4 frontier = frontier_dk[idx];
        dk[idx] = make_float4(
            base.x + split.x + frontier.x,
            base.y + split.y + frontier.y,
            base.z + split.z + frontier.z,
            base.w + split.w + frontier.w
        );
    }
    if (idx < dv_vecs) {
        const float4 base = dv[idx];
        const float4 split = dense_split_dv[idx];
        const float4 frontier = frontier_dv[idx];
        dv[idx] = make_float4(
            base.x + split.x + frontier.x,
            base.y + split.y + frontier.y,
            base.z + split.z + frontier.z,
            base.w + split.w + frontier.w
        );
    }
}

__global__ __launch_bounds__(256, 2)
void dense_tmem_two_scratch_add_to_bf16_kernel(
    bf16_2 *out_dk,
    const float4 *base_dk,
    const float4 *dense_split_dk,
    const float4 *frontier_dk,
    int64_t dk_vecs,
    bf16_2 *out_dv,
    const float4 *base_dv,
    const float4 *dense_split_dv,
    const float4 *frontier_dv,
    int64_t dv_vecs
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dk_vecs) {
        const float4 base = base_dk[idx];
        const float4 split = dense_split_dk[idx];
        const float4 frontier = frontier_dk[idx];
        out_dk[2 * idx] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(
                base.x + split.x + frontier.x,
                base.y + split.y + frontier.y
            )
        );
        out_dk[2 * idx + 1] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(
                base.z + split.z + frontier.z,
                base.w + split.w + frontier.w
            )
        );
    }
    if (idx < dv_vecs) {
        const float4 base = base_dv[idx];
        const float4 split = dense_split_dv[idx];
        const float4 frontier = frontier_dv[idx];
        out_dv[2 * idx] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(
                base.x + split.x + frontier.x,
                base.y + split.y + frontier.y
            )
        );
        out_dv[2 * idx + 1] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(
                base.z + split.z + frontier.z,
                base.w + split.w + frontier.w
            )
        );
    }
}

__global__ __launch_bounds__(256, 2)
void cta2_owner_q_split_add_to_bf16_kernel(
    bf16_2 *out_dk,
    const float4 *partial0_dk,
    const float4 *partial1_dk,
    int64_t dk_vecs,
    bf16_2 *out_dv,
    const float4 *partial0_dv,
    const float4 *partial1_dv,
    int64_t dv_vecs
) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dk_vecs) {
        const float4 lhs = partial0_dk[idx];
        const float4 rhs = partial1_dk[idx];
        out_dk[2 * idx] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(lhs.x + rhs.x, lhs.y + rhs.y)
        );
        out_dk[2 * idx + 1] =
            base_types::convertor<bf16_2, float2>::convert(
                make_float2(lhs.z + rhs.z, lhs.w + rhs.w)
            );
    }
    if (idx < dv_vecs) {
        const float4 lhs = partial0_dv[idx];
        const float4 rhs = partial1_dv[idx];
        out_dv[2 * idx] = base_types::convertor<bf16_2, float2>::convert(
            make_float2(lhs.x + rhs.x, lhs.y + rhs.y)
        );
        out_dv[2 * idx + 1] =
            base_types::convertor<bf16_2, float2>::convert(
                make_float2(lhs.z + rhs.z, lhs.w + rhs.w)
            );
    }
}

__global__ __launch_bounds__(256, 2)
void dense_tmem_adaptive_two_scratch_add_kernel(
    float4 *dk,
    const float4 *dense_split_dk,
    const float4 *frontier_dk,
    int64_t dk_vecs,
    int64_t dk_split_vecs,
    float4 *dv,
    const float4 *dense_split_dv,
    const float4 *frontier_dv,
    int64_t dv_vecs,
    int64_t dv_split_vecs
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dk_vecs) {
        const float4 base = dk[idx];
        const float4 frontier = frontier_dk[idx];
        float4 sum = make_float4(
            base.x + frontier.x,
            base.y + frontier.y,
            base.z + frontier.z,
            base.w + frontier.w
        );
        if (idx < dk_split_vecs) {
            const float4 split = dense_split_dk[idx];
            sum.x += split.x;
            sum.y += split.y;
            sum.z += split.z;
            sum.w += split.w;
        }
        dk[idx] = sum;
    }
    if (idx < dv_vecs) {
        const float4 base = dv[idx];
        const float4 frontier = frontier_dv[idx];
        float4 sum = make_float4(
            base.x + frontier.x,
            base.y + frontier.y,
            base.z + frontier.z,
            base.w + frontier.w
        );
        if (idx < dv_split_vecs) {
            const float4 split = dense_split_dv[idx];
            sum.x += split.x;
            sum.y += split.y;
            sum.z += split.z;
            sum.w += split.w;
        }
        dv[idx] = sum;
    }
}

template <int DenseSplitCount>
__global__ __launch_bounds__(256, 2)
void dense_tmem_multi_scratch_add_kernel(
    float4 *dk,
    const float4 *dense_split_dk,
    const float4 *frontier_dk,
    int64_t dk_vecs,
    float4 *dv,
    const float4 *dense_split_dv,
    const float4 *frontier_dv,
    int64_t dv_vecs
) {
    static_assert(
        DenseSplitCount == 3 || DenseSplitCount == 4 || DenseSplitCount == 8,
        "Multi-scratch merge is only used by split-3, split-4, or split-8"
    );
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < dk_vecs) {
        float4 sum = dk[idx];
        #pragma unroll
        for (int split = 0; split < DenseSplitCount - 1; ++split) {
            const float4 partial = dense_split_dk[static_cast<int64_t>(split) * dk_vecs + idx];
            sum.x += partial.x;
            sum.y += partial.y;
            sum.z += partial.z;
            sum.w += partial.w;
        }
        const float4 frontier = frontier_dk[idx];
        sum.x += frontier.x;
        sum.y += frontier.y;
        sum.z += frontier.z;
        sum.w += frontier.w;
        dk[idx] = sum;
    }
    if (idx < dv_vecs) {
        float4 sum = dv[idx];
        #pragma unroll
        for (int split = 0; split < DenseSplitCount - 1; ++split) {
            const float4 partial = dense_split_dv[static_cast<int64_t>(split) * dv_vecs + idx];
            sum.x += partial.x;
            sum.y += partial.y;
            sum.z += partial.z;
            sum.w += partial.w;
        }
        const float4 frontier = frontier_dv[idx];
        sum.x += frontier.x;
        sum.y += frontier.y;
        sum.z += frontier.z;
        sum.w += frontier.w;
        dv[idx] = sum;
    }
}

template <bool REPAIR_DV, typename C>
__global__ __launch_bounds__(WARPGROUP_WARPS * kWarpThreads, 8)
void seq2048_exact_causal_first_tile_patch_kernel(
    const __grid_constant__ seq2048_exact_dkdv_globals<C> g,
    int kv_tile64_offset
) {
    using q_tile = typename seq2048_exact_dkdv_globals<C>::q_tile;
    using k_tile = typename seq2048_exact_dkdv_globals<C>::k_tile;
    using v_tile = typename seq2048_exact_dkdv_globals<C>::v_tile;
    using do_tile = typename seq2048_exact_dkdv_globals<C>::do_tile;

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
    const int kv_tile64_idx = (static_cast<int>(blockIdx.z) + kv_tile64_offset) * 2;
    const int kv_tiles64 = g.seq_len / C::TileRows;
    if (kv_tile64_idx >= kv_tiles64) {
        return;
    }
    const int kv_subtile_idx = kv_tile64_idx * WARPGROUP_WARPS + warp;
    const int q_block_idx = kv_tile64_idx;

    rt_bf<kRefTileM, C::Dqk> k_reg, q_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_vec, dpsum_vec;
    rt_fl<kRefTileM, C::Dqk> dk_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;
    warp::zero(dk_accum);
    warp::zero(dv_accum);

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
        tkfa4::bwd_cute16_kernel::detail::repair_dkdv_step<true, REPAIR_DV, C>(
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
            g.seq_len
        );
    }

    rt_fl<kRefTileM, C::Dqk> dk_existing;
    warp::load<dim::DEPTH>(dk_existing, g.dk, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::add(dk_accum, dk_accum, dk_existing);
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    if constexpr (REPAIR_DV) {
        rt_fl<kRefTileM, C::Dvo> dv_existing;
        warp::load<dim::DEPTH>(dv_existing, g.dv, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::add(dv_accum, dv_accum, dv_existing);
        warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    }
}

template <typename C, bool kUseDirectDq>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_fullseq_dq_only(const __grid_constant__ dq_only_globals<C> g) {
    constexpr int TotalWarps = C::BlockThreads / kWarpThreads;
    constexpr int q_tiles_buffered = (C::WarpTiles == 8 ? 8 : 4);
    constexpr int dense_q_block_lag = (C::WarpTiles + q_tiles_buffered - 1) / q_tiles_buffered;
    static_assert(TotalWarps >= C::WarpTiles, "dq_only requires at least WarpTiles warps");
    using qk_bf_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using dq_chunk_tile = typename dq_only_globals<C>::dq_chunk_tile;
    using dqacc_tile = typename dq_only_globals<C>::dqacc_tile;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo, true, 64>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;
    union dq_smem_union {
        dq_chunk_tile chunks[3][C::WarpTiles];
        dqacc_tile full[C::WarpTiles];
    };

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) dq_smem_union dq_smem;
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];

    const int warp = threadIdx.x >> 5;
    const bool is_compute = warp < C::WarpTiles;
    constexpr int q_load_warp = TotalWarps - 2;
    constexpr int do_load_warp = TotalWarps - 1;
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
    const int q_block_start = kv_tile_base / q_tiles_buffered;
    const int dense_q_block_start = min(num_q_blocks, q_block_start + dense_q_block_lag);

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;

    if (is_compute) {
        warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    }

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<qk_bf_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<v_bf_tile, dim::DEPTH>();
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
    }
    __syncthreads();

    for (int q_block_idx = q_block_start; q_block_idx < dense_q_block_start; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - q_block_start;
        const int phase = local_q_iter & 1;
        if (warp == q_load_warp) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == do_load_warp) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < q_tiles_buffered) {
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_log2_vec);
            warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_vec);
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        if (is_compute) {
            const int first_active_subtile = max(0, kv_subtile_idx - q_tile_base);
            const int diagonal_subtile = kv_subtile_idx - q_tile_base;
            #pragma unroll
            for (int subtile = first_active_subtile; subtile < q_tiles_buffered; ++subtile) {
                const int q_tile_idx = q_tile_base + subtile;
                rt_bf<kRefTileM, C::Dqk> q_reg;
                rt_bf<kRefTileM, C::Dvo> do_reg;
                rt_fl<kRefTileM, C::Dqk> dq_partial;
                rt_fl<kRefTileM, 64> dq0, dq1, dq2;
                stats_vec lse_log2_vec, dpsum_vec;
                warp::load(q_reg, q_smem[subtile]);
                warp::load(do_reg, do_smem[subtile]);
                warp::load(lse_log2_vec, lse_log2_smem[subtile]);
                warp::load(dpsum_vec, dpsum_smem[subtile]);
                tkfa4::bwd_cute16_kernel::detail::repair_dq_step<true, C>(
                    dq_partial,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len,
                    subtile != diagonal_subtile
                );
                if (q_block_idx != q_block_start || subtile != first_active_subtile) {
                    warp::tma::store_async_read_wait();
                }
                extract_chunk<0>(dq_partial, dq0);
                extract_chunk<1>(dq_partial, dq1);
                extract_chunk<2>(dq_partial, dq2);
                if constexpr (kUseDirectDq) {
                    warp::store(dq_smem.chunks[0][warp], dq0);
                    warp::store(dq_smem.chunks[1][warp], dq1);
                    warp::store(dq_smem.chunks[2][warp], dq2);
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq0, dq_smem.chunks[0][warp], {batch_idx, q_tile_idx, head_idx, 0});
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq1, dq_smem.chunks[1][warp], {batch_idx, q_tile_idx, head_idx, 1});
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq2, dq_smem.chunks[2][warp], {batch_idx, q_tile_idx, head_idx, 2});
                } else {
                    const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cluster_rank;
                    warp::store(dq_smem.full[warp], dq_partial);
                    __syncwarp();
                    warp::tma::store_add_async(g.dqacc, dq_smem.full[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
                    warp::tma::store_async_wait();
                }
            }
        }
        if constexpr (kUseDirectDq) {
            if (is_compute) {
                warp::tma::store_commit_group();
            }
        }
        __syncthreads();
    }

    for (int q_block_idx = dense_q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - q_block_start;
        const int phase = local_q_iter & 1;
        if (warp == q_load_warp) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == do_load_warp) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < q_tiles_buffered) {
            stats_vec lse_log2_vec, dpsum_vec;
            warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_log2_vec);
            warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_vec);
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        if (is_compute) {
            #pragma unroll 1
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                const int q_tile_idx = q_tile_base + subtile;
                rt_bf<kRefTileM, C::Dqk> q_reg;
                rt_bf<kRefTileM, C::Dvo> do_reg;
                rt_fl<kRefTileM, C::Dqk> dq_partial;
                rt_fl<kRefTileM, 64> dq0, dq1, dq2;
                stats_vec lse_log2_vec, dpsum_vec;
                warp::load(q_reg, q_smem[subtile]);
                warp::load(do_reg, do_smem[subtile]);
                warp::load(lse_log2_vec, lse_log2_smem[subtile]);
                warp::load(dpsum_vec, dpsum_smem[subtile]);
                tkfa4::bwd_cute16_kernel::detail::repair_dq_step<true, C>(
                    dq_partial,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    g.scale,
                    g.scale_log2e,
                    q_tile_idx,
                    kv_subtile_idx,
                    g.seq_len,
                    true
                );
                if (q_block_idx != dense_q_block_start || subtile != 0) {
                    warp::tma::store_async_read_wait();
                }
                extract_chunk<0>(dq_partial, dq0);
                extract_chunk<1>(dq_partial, dq1);
                extract_chunk<2>(dq_partial, dq2);
                if constexpr (kUseDirectDq) {
                    warp::store(dq_smem.chunks[0][warp], dq0);
                    warp::store(dq_smem.chunks[1][warp], dq1);
                    warp::store(dq_smem.chunks[2][warp], dq2);
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq0, dq_smem.chunks[0][warp], {batch_idx, q_tile_idx, head_idx, 0});
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq1, dq_smem.chunks[1][warp], {batch_idx, q_tile_idx, head_idx, 1});
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq2, dq_smem.chunks[2][warp], {batch_idx, q_tile_idx, head_idx, 2});
                } else {
                    const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cluster_rank;
                    warp::store(dq_smem.full[warp], dq_partial);
                    __syncwarp();
                    warp::tma::store_add_async(g.dqacc, dq_smem.full[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
                    warp::tma::store_async_wait();
                }
            }
        }
        if constexpr (kUseDirectDq) {
            if (is_compute) {
                warp::tma::store_commit_group();
            }
        }
        __syncthreads();
    }

    if (is_compute) {
        warp::tma::store_async_read_wait();
    }
}

template <typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_fullseq_dq_from_ds(const __grid_constant__ dq_from_ds_globals<C> g) {
    using dq_chunk_tile = typename dq_from_ds_globals<C>::dq_chunk_tile;
    using ds_tile = typename dq_from_ds_globals<C>::ds_tile;

    __shared__ alignas(1024) dq_chunk_tile dq_smem[3][C::WarpTiles];

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
    const int q_tiles = g.seq_len / kRefTileM;
    const int q_block_start = kv_tile_base / C::WarpTiles;

    rt_bf<kRefTileM, C::Dqk> k_reg;
    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});

    for (int q_tile_idx = q_block_start; q_tile_idx < q_tiles; ++q_tile_idx) {
        rt_bf<kRefTileM, kRefTileN> ds_bf;
        rt_fl<kRefTileM, 64> dq0, dq1, dq2;
        rt_bf<kRefTileM, 64> k_chunk;
        rt_bf<kRefTileM, 64, ducks::rt_layout::col> k_col;

        warp::load<dim::DEPTH>(ds_bf, g.ds, {batch_idx, q_tile_idx, head_idx, kv_subtile_idx});

        if (q_tile_idx != q_block_start) {
            warp::tma::store_async_read_wait();
        }

        extract_chunk<0>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq0);
        warp::mma_AB(dq0, ds_bf, k_col, dq0);

        extract_chunk<1>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq1);
        warp::mma_AB(dq1, ds_bf, k_col, dq1);

        extract_chunk<2>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq2);
        warp::mma_AB(dq2, ds_bf, k_col, dq2);

        warp::store(dq_smem[0][warp], dq0);
        warp::store(dq_smem[1][warp], dq1);
        warp::store(dq_smem[2][warp], dq2);
        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq0, dq_smem[0][warp], {batch_idx, q_tile_idx, head_idx, 0});
        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq1, dq_smem[1][warp], {batch_idx, q_tile_idx, head_idx, 1});
        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq2, dq_smem[2][warp], {batch_idx, q_tile_idx, head_idx, 2});
    }

    warp::tma::store_commit_group();
    warp::tma::store_async_read_wait();
}

template <
    dq_only_clustered_mode Mode,
    typename C,
    typename DqOutT = float,
    bool UseChunkedTmemDq = false
>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_fullseq_dq_only_clustered(const __grid_constant__ dq_only_clustered_globals<C, DqOutT> g) {
    using q_tile = typename dq_only_clustered_globals<C, DqOutT>::q_tile;
    using k_tile = typename dq_only_clustered_globals<C, DqOutT>::k_tile;
    using v_tile = typename dq_only_clustered_globals<C, DqOutT>::v_tile;
    using do_tile = typename dq_only_clustered_globals<C, DqOutT>::do_tile;
    using dq_chunk_tile = typename dq_only_clustered_globals<C, DqOutT>::dq_chunk_tile;
    using dqacc_tile = typename dq_only_clustered_globals<C, DqOutT>::dqacc_tile;
    using stats_smem_tile = typename dq_only_clustered_globals<C, DqOutT>::stats_tile;
    using ds_warp_tile = st_bf<kRefTileM, C::TileRows>;
    using ds_full_tile = st_bf<C::TileRows, C::TileRows>;
    using attn_tt = half_tt_fl<C::TileRows>;
    using dq_tt = half_tt_fl<C::Dqk>;
    using dq_chunk_tt = half_tt_fl<64>;
    union dq_store_smem {
        dqacc_tile full[C::QSubtiles];
        dq_chunk_tile chunks[3][C::QSubtiles];
    };
    union ds_store_smem {
        ds_full_tile full[C::ConsumerWarpgroups];
        ds_warp_tile warp[C::ConsumerWarpgroups][WARPGROUP_WARPS];
    };

    struct shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        dq_store_smem dq_smem;
        ds_store_smem ds_smem;
        stats_smem_tile lse_log2_smem[C::QSubtiles];
        stats_smem_tile dpsum_smem[C::QSubtiles];
    };

    __shared__ alignas(1024) shared_storage smem;
    auto &k_smem = smem.k_smem;
    auto &v_smem = smem.v_smem;
    auto &q_smem = smem.q_smem;
    auto &do_smem = smem.do_smem;
    auto &dq_smem = smem.dq_smem;
    auto &ds_warp_smem = smem.ds_smem.warp;
    auto &ds_full_smem = smem.ds_smem.full;
    auto &lse_log2_smem = smem.lse_log2_smem;
    auto &dpsum_smem = smem.dpsum_smem;

    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];
    __shared__ __align__(16) kittens::semaphore score_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore dp_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore dq_ready[C::ConsumerWarpgroups];

    const int warp = kittens::warpid();
    constexpr int kLoadWarp = C::ComputeWarps;
    const bool is_compute = warp < C::ComputeWarps;
    const bool is_load = warp == kLoadWarp;
    const int consumer_idx = is_compute ? (warp / kittens::WARPGROUP_WARPS) : -1;

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int kv_block_idx = static_cast<int>(blockIdx.x);
    const int cluster_rank = kv_block_idx % C::ClusterSize;
    const int cluster_idx = kv_block_idx / C::ClusterSize;
    const int num_k_blocks = g.seq_len / (C::TileRows * C::ConsumerWarpgroups);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::ConsumerWarpgroups;
    const int q_blocks = g.seq_len / C::TileRows;
    const int q_start_block =
        Mode == dq_only_clustered_mode::DonorBulkOnly || kUseMainFirstBlockClusteredDq
            ? kv_tile_base
            : (kv_tile_base + 1);

    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dq_tt dq_accum_tt0{0};
    dq_tt dq_accum_tt1{0};

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();

        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            init_semaphore(score_ready[w][0], 0, 1);
            init_semaphore(dp_ready[w][0], 0, 1);
            init_semaphore(dq_ready[w], 0, 1);
        }
    }
    __syncthreads();

    if (q_start_block < q_blocks) {
        const int q_tile_base = q_start_block * C::QSubtiles;
        if (is_load) {
            coord<q_tile> q_tile_idx = {batch_idx, q_start_block, head_idx, 0};
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx, q_b[0]);
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, q_tile_idx, o_b[0]);

            #pragma unroll
            for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                typename rt_fl<kRefTileM, C::TileRows>::col_vec stats_stage_vec;
                warp::load(stats_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(lse_log2_smem[subtile], stats_stage_vec);
                warp::load(stats_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[subtile], stats_stage_vec);
            }
        }
    }

    if (is_compute) {
        score_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
        dp_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, C::TileRows);
        if constexpr (UseChunkedTmemDq) {
            dq_accum_tt0 = tm_alloc.template allocate<dq_tt>(0, 2 * C::TileRows);
            dq_accum_tt1 = tm_alloc.template allocate<dq_tt>(1, 2 * C::TileRows);
        }
        rt_bf<kRefTileM, C::Dqk> k_reg;
        rt_bf<kRefTileM, C::Dvo> v_reg;
        warpgroup::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_tile_base + consumer_idx, head_idx, 0});
        warpgroup::store(k_smem[consumer_idx], k_reg);
        warpgroup::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_tile_base + consumer_idx, head_idx, 0});
        warpgroup::store(v_smem[consumer_idx], v_reg);
    }
    __syncthreads();
    bool dq_store_outstanding = false;
    for (int q_block_idx = q_start_block; q_block_idx < q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * C::QSubtiles;
        const int local_q_iter = q_block_idx - q_start_block;
        const int phase = local_q_iter & 1;

        if (is_compute) {
            rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
            rt_bf<kRefTileM, C::TileRows> ds_block_t_mma;

            if constexpr (Mode == dq_only_clustered_mode::LegacyPatched) {
                const int kv_tile_idx = kv_tile_base + consumer_idx;
                if (q_block_idx > kv_tile_idx) {
                    tkfa4::bwd_hot::detail::hot_compute_dq_qmajor_loop<false, C>(
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
                        kv_tile_idx
                    );
                } else {
                    tkfa4::bwd_hot::detail::hot_compute_dq_qmajor_loop<true, C>(
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
                        kv_tile_idx
                    );
                }
                if (false && q_block_idx <= q_start_block + 3) {
                    tkfa4::bwd_hot::detail::hot_overwrite_dq_exact_from_loaded_nosync<true, C>(
                        ds_block_t,
                        ds_block_t_mma,
                        q_smem,
                        k_smem,
                        v_smem,
                        do_smem,
                        ds_warp_smem,
                        lse_log2_smem,
                        dpsum_smem,
                        g.scale,
                        g.scale_log2e,
                        q_block_idx,
                        kv_tile_base + consumer_idx,
                        g.seq_len
                    );
                }
            } else {
                tkfa4::bwd_hot::detail::hot_compute_dq_exact_loop<true, C>(
                    q_b[0],
                    o_b[0],
                    ds_block_t,
                    ds_block_t_mma,
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
                    kv_tile_base + consumer_idx,
                    g.seq_len
                );
            }
            if constexpr (UseChunkedTmemDq) {
                if (consumer_idx == 0) {
                    warpgroup::mm_AB(dq_accum_tt0, ds_full_smem[0], k_smem[0], dq_ready[0]);
                } else {
                    warpgroup::mm_AB(dq_accum_tt1, ds_full_smem[1], k_smem[1], dq_ready[1]);
                }
            }
        }
        __syncthreads();
        const int next_q_block_idx = q_block_idx + 1;
        if (next_q_block_idx < q_blocks) {
            const int next_q_tile_base = next_q_block_idx * C::QSubtiles;
            if (is_load) {
                coord<q_tile> next_q_tile_idx = {batch_idx, next_q_block_idx, head_idx, 0};
                warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, next_q_tile_idx, q_b[0]);
                warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, next_q_tile_idx, o_b[0]);

                #pragma unroll
                for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                    typename rt_fl<kRefTileM, C::TileRows>::col_vec stats_stage_vec;
                    warp::load(stats_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, next_q_tile_base + subtile});
                    warp::store(lse_log2_smem[subtile], stats_stage_vec);
                    warp::load(stats_stage_vec, g.dpsum, {batch_idx, head_idx, 0, next_q_tile_base + subtile});
                    warp::store(dpsum_smem[subtile], stats_stage_vec);
                }
            }
        }
        if (is_compute && consumer_idx == 0) {
            const int dq_subtile_idx = warpgroup::warpid();
            if (dq_subtile_idx < C::QSubtiles) {
                const bool include_peer =
                    Mode == dq_only_clustered_mode::DonorBulkOnly ? true : (q_block_idx > kv_tile_base);
                constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;
                const int q_tile_idx = q_tile_base + dq_subtile_idx;
                const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
                const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
                const int scratch_tile_idx =
                    ((q_tile_group_idx * C::ClusterSize) + cluster_rank) * kQSubtilesPerTile + q_subtile_in_group;

                if constexpr (UseChunkedTmemDq) {
                    wait(dq_ready[0], phase);
                    wait(dq_ready[1], phase);
                    #pragma unroll
                    for (int chunk = 0; chunk < 3; ++chunk) {
                        rt_fl<kRefTileM, 64> dq_chunk, dq_peer_chunk;
                        const dq_chunk_tt dq_local_tt =
                            dq_accum_tt0.template subtile<dq_chunk_tt>(chunk * 64);
                        const dq_chunk_tt dq_peer_tt =
                            dq_accum_tt1.template subtile<dq_chunk_tt>(chunk * 64);
                        warpgroup::load_async(dq_chunk, dq_local_tt);
                        if (include_peer) {
                            warpgroup::load_async(dq_peer_chunk, dq_peer_tt);
                        }
                        tensor_load_wait();
                        if (include_peer) {
                            warp::add(dq_chunk, dq_chunk, dq_peer_chunk);
                        }
                        if (dq_store_outstanding) {
                            warp::tma::store_async_wait();
                        }
                        warp::store(dq_smem.chunks[chunk][dq_subtile_idx], dq_chunk);
                        if (chunk == 0) {
                            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                                g.dq0,
                                dq_smem.chunks[0][dq_subtile_idx],
                                {batch_idx, q_tile_idx, head_idx, 0}
                            );
                        } else if (chunk == 1) {
                            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                                g.dq1,
                                dq_smem.chunks[1][dq_subtile_idx],
                                {batch_idx, q_tile_idx, head_idx, 1}
                            );
                        } else {
                            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                                g.dq2,
                                dq_smem.chunks[2][dq_subtile_idx],
                                {batch_idx, q_tile_idx, head_idx, 2}
                            );
                        }
                        dq_store_outstanding = true;
                    }
                } else {
                    rt_fl<kRefTileM, C::Dqk> dq_partial;
                    rt_bf<kRefTileM, C::TileRows> ds_reg;
                    rt_bf<C::TileRows, C::Dqk> k_local_reg;
                    rt_bf<C::TileRows, C::Dqk, ducks::rt_layout::col> k_local_col;

                    warp::zero(dq_partial);
                    warp::load(ds_reg, ds_warp_smem[0][dq_subtile_idx]);
                    warp::load(k_local_reg, k_smem[0]);
                    warp::swap_layout(k_local_col, k_local_reg);
                    warp::mma_AB(dq_partial, ds_reg, k_local_col, dq_partial);

                    if (include_peer) {
                        warp::load(ds_reg, ds_warp_smem[1][dq_subtile_idx]);
                        warp::load(k_local_reg, k_smem[1]);
                        warp::swap_layout(k_local_col, k_local_reg);
                        warp::mma_AB(dq_partial, ds_reg, k_local_col, dq_partial);
                    }
                    if (dq_store_outstanding) {
                        warp::tma::store_async_wait();
                    }
                    if constexpr (kUseDirectFinalClusteredDq) {
                        warp::store(dq_smem.full[dq_subtile_idx], dq_partial);
                        warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                            g.dq_full,
                            dq_smem.full[dq_subtile_idx],
                            {batch_idx, q_tile_idx, head_idx, 0}
                        );
                    } else {
                        warp::store(dq_smem.full[dq_subtile_idx], dq_partial);
                        __syncwarp();
                        warp::tma::store_add_async(g.dqacc, dq_smem.full[dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
                    }
                    dq_store_outstanding = true;
                }
            }
        }
        if (next_q_block_idx < q_blocks) {
            __syncthreads();
        }
    }
    if (is_compute && consumer_idx == 0 && dq_store_outstanding) {
        warp::tma::store_async_wait();
    }
}

template <
    typename C,
    typename DqOutT = float,
    bool DoubleBufferInputs = false,
    int DqReplaySplitCount = 1
>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_fullseq_dq_only_clustered_pipelined(
    const __grid_constant__ dq_only_clustered_globals<C, DqOutT> g
) {
    static_assert(
        DqReplaySplitCount == 1 || DqReplaySplitCount == 2,
        "Pipelined dQ replay split count must be 1 or 2"
    );
    using G = dq_only_clustered_globals<C, DqOutT>;
    using q_tile = typename G::q_tile;
    using k_tile = typename G::k_tile;
    using v_tile = typename G::v_tile;
    using do_tile = typename G::do_tile;
    using dq_chunk_tile = typename G::dq_chunk_tile;
    using stats_smem_tile = typename G::stats_tile;
    using ds_warp_tile = st_bf<kRefTileM, C::TileRows>;
    using ds_full_tile = st_bf<C::TileRows, C::TileRows>;
    using attn_tt = half_tt_fl<C::TileRows>;
    using dq_tt = half_tt_fl<C::Dqk>;
    using dq_chunk_tt = half_tt_fl<64>;
    static constexpr int kInputStages = DoubleBufferInputs ? 2 : 1;

    union ds_store_smem {
        ds_full_tile full[C::ConsumerWarpgroups];
        ds_warp_tile warp[C::ConsumerWarpgroups][WARPGROUP_WARPS];
    };
    struct shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[kInputStages];
        do_tile do_smem[kInputStages];
        dq_chunk_tile dq_smem[C::QSubtiles];
        ds_store_smem ds_smem[2];
        stats_smem_tile lse_log2_smem[kInputStages][C::QSubtiles];
        stats_smem_tile dpsum_smem[kInputStages][C::QSubtiles];
    };

    __shared__ alignas(1024) shared_storage smem;
    auto &k_smem = smem.k_smem;
    auto &v_smem = smem.v_smem;
    auto &q_smem = smem.q_smem;
    auto &do_smem = smem.do_smem;
    auto &dq_smem = smem.dq_smem;
    auto &ds_warp_smem = smem.ds_smem;
    auto &ds_full_smem = smem.ds_smem;
    auto &lse_log2_smem = smem.lse_log2_smem;
    auto &dpsum_smem = smem.dpsum_smem;

    __shared__ __align__(16) kittens::semaphore q_b[kInputStages];
    __shared__ __align__(16) kittens::semaphore o_b[kInputStages];
    __shared__ __align__(16) kittens::semaphore stats_ready[kInputStages];
    __shared__ __align__(16) kittens::semaphore input_released[kInputStages];
    __shared__ __align__(16) kittens::semaphore score_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore dp_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore dq_ready[C::ConsumerWarpgroups][2];
    __shared__ __align__(16) kittens::semaphore dq_empty[2];

    const int warp = kittens::warpid();
    const bool is_compute = warp < C::ComputeWarps;
    const bool is_reduce = warp >= C::ReduceWarpBase && warp < C::LoadWarp;
    const bool is_load = warp == C::LoadWarp;
    const int consumer_idx = is_compute ? (warp / kittens::WARPGROUP_WARPS) : -1;

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int work_idx = static_cast<int>(blockIdx.x);
    const int replay_split_idx = work_idx % DqReplaySplitCount;
    const int kv_block_idx = work_idx / DqReplaySplitCount;
    const int num_k_blocks = g.seq_len / (C::TileRows * C::ConsumerWarpgroups);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::ConsumerWarpgroups;
    const int q_blocks = g.seq_len / C::TileRows;
    const int q_start_block = kv_tile_base + replay_split_idx;

    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dq_tt dq_stage00 = tm_alloc.template allocate<dq_tt>(0, 2 * C::TileRows);
    dq_tt dq_stage10 = tm_alloc.template allocate<dq_tt>(1, 2 * C::TileRows);
    dq_tt dq_stage01 = tm_alloc.template allocate<dq_tt>(0, 2 * C::TileRows + C::Dqk);
    dq_tt dq_stage11 = tm_alloc.template allocate<dq_tt>(1, 2 * C::TileRows + C::Dqk);

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        for (int input_stage = 0; input_stage < kInputStages; ++input_stage) {
            init_semaphore(q_b[input_stage], 0, 1);
            init_semaphore(o_b[input_stage], 0, 1);
            init_semaphore(stats_ready[input_stage], 0, 1);
            init_semaphore(
                input_released[input_stage],
                0,
                C::ConsumerWarpgroups
            );
        }
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            init_semaphore(score_ready[w], 0, 1);
            init_semaphore(dp_ready[w], 0, 1);
            for (int stage = 0; stage < 2; ++stage) {
                init_semaphore(dq_ready[w][stage], 0, 1);
            }
        }
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(dq_empty[stage], 0, C::ReduceWarps);
        }
    }
    __syncthreads();

    if (is_compute) {
        score_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
        dp_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, C::TileRows);
        rt_bf<kRefTileM, C::Dqk> k_reg;
        rt_bf<kRefTileM, C::Dvo> v_reg;
        warpgroup::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_tile_base + consumer_idx, head_idx, 0});
        warpgroup::store(k_smem[consumer_idx], k_reg);
        warpgroup::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_tile_base + consumer_idx, head_idx, 0});
        warpgroup::store(v_smem[consumer_idx], v_reg);
    }
    __syncthreads();

    if (is_load) {
        for (
            int q_block_idx = q_start_block;
            q_block_idx < q_blocks;
            q_block_idx += DqReplaySplitCount
        ) {
            const int local_q_iter =
                (q_block_idx - q_start_block) / DqReplaySplitCount;
            const int input_stage = local_q_iter % kInputStages;
            if (local_q_iter >= kInputStages) {
                wait(
                    input_released[input_stage],
                    ((local_q_iter / kInputStages) - 1) & 1
                );
            }
            const int q_tile_base = q_block_idx * C::QSubtiles;
            coord<q_tile> q_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
            warp::tma::expect_bytes(q_b[input_stage], sizeof(q_smem[input_stage]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                q_smem[input_stage], g.q, q_tile_idx, q_b[input_stage]
            );
            warp::tma::expect_bytes(o_b[input_stage], sizeof(do_smem[input_stage]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                do_smem[input_stage], g.dout, q_tile_idx, o_b[input_stage]
            );

            #pragma unroll
            for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                typename rt_fl<kRefTileM, C::TileRows>::col_vec stats_stage_vec;
                warp::load(stats_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(lse_log2_smem[input_stage][subtile], stats_stage_vec);
                warp::load(stats_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[input_stage][subtile], stats_stage_vec);
            }
            warp::arrive(stats_ready[input_stage]);
        }
    } else if (is_compute) {
        for (
            int q_block_idx = q_start_block;
            q_block_idx < q_blocks;
            q_block_idx += DqReplaySplitCount
        ) {
            const int local_q_iter =
                (q_block_idx - q_start_block) / DqReplaySplitCount;
            const int stage = local_q_iter & 1;
            const int input_stage = local_q_iter % kInputStages;
            const int input_phase = (local_q_iter / kInputStages) & 1;
            if (local_q_iter >= 2) {
                const int previous_stage_phase = ((local_q_iter / 2) - 1) & 1;
                wait(dq_ready[consumer_idx][stage], previous_stage_phase);
                wait(dq_empty[stage], previous_stage_phase);
            }
            wait(stats_ready[input_stage], input_phase);

            rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
            rt_bf<kRefTileM, C::TileRows> ds_block_t_mma;
            const int kv_tile_idx = kv_tile_base + consumer_idx;
            if (q_block_idx > kv_tile_idx) {
                tkfa4::bwd_hot::detail::hot_compute_dq_qmajor_loop<false, C>(
                    q_b[input_stage], o_b[input_stage], score_ready[consumer_idx], dp_ready[consumer_idx],
                    p_block_t, dp_block_t, ds_block_t, ds_block_t_mma,
                    score_tt[consumer_idx], dp_tt[consumer_idx], q_smem, k_smem,
                    v_smem, do_smem, ds_warp_smem[stage].warp,
                    lse_log2_smem[input_stage], dpsum_smem[input_stage],
                    g.scale, g.scale_log2e, local_q_iter & 1, q_block_idx,
                    kv_tile_idx, input_stage, input_phase
                );
            } else {
                tkfa4::bwd_hot::detail::hot_compute_dq_qmajor_loop<true, C>(
                    q_b[input_stage], o_b[input_stage], score_ready[consumer_idx], dp_ready[consumer_idx],
                    p_block_t, dp_block_t, ds_block_t, ds_block_t_mma,
                    score_tt[consumer_idx], dp_tt[consumer_idx], q_smem, k_smem,
                    v_smem, do_smem, ds_warp_smem[stage].warp,
                    lse_log2_smem[input_stage], dpsum_smem[input_stage],
                    g.scale, g.scale_log2e, local_q_iter & 1, q_block_idx,
                    kv_tile_idx, input_stage, input_phase
                );
            }

            const dq_tt dq_stage = consumer_idx == 0
                ? (stage == 0 ? dq_stage00 : dq_stage01)
                : (stage == 0 ? dq_stage10 : dq_stage11);
            warpgroup::mm_AB(
                dq_stage,
                ds_full_smem[stage].full[consumer_idx],
                k_smem[consumer_idx],
                dq_ready[consumer_idx][stage]
            );
            if (warpgroup::warpid() == 0) {
                warp::arrive(input_released[input_stage]);
            }
        }
    } else if (is_reduce) {
        const int dq_subtile_idx = warp - C::ReduceWarpBase;
        for (
            int q_block_idx = q_start_block;
            q_block_idx < q_blocks;
            q_block_idx += DqReplaySplitCount
        ) {
            const int local_q_iter =
                (q_block_idx - q_start_block) / DqReplaySplitCount;
            const int stage = local_q_iter & 1;
            const int stage_phase = (local_q_iter / 2) & 1;
            wait(dq_ready[0][stage], stage_phase);
            wait(dq_ready[1][stage], stage_phase);

            const bool include_peer = q_block_idx > kv_tile_base;
            const int q_tile_idx = q_block_idx * C::QSubtiles + dq_subtile_idx;
            const dq_tt dq_local_stage = stage == 0 ? dq_stage00 : dq_stage01;
            const dq_tt dq_peer_stage = stage == 0 ? dq_stage10 : dq_stage11;

            #pragma unroll
            for (int chunk = 0; chunk < 3; ++chunk) {
                rt_fl<kRefTileM, 64> dq_chunk, dq_peer_chunk;
                using dq_warp_chunk_tt = tt_fl<kRefTileM, 64>;
                const uint32_t warp_row_offset = (32 * dq_subtile_idx) << 16;
                const dq_warp_chunk_tt dq_local_tt{
                    dq_local_stage.addr + warp_row_offset + chunk * 64
                };
                const dq_warp_chunk_tt dq_peer_tt{
                    dq_peer_stage.addr + warp_row_offset + chunk * 64
                };
                warp::load_async(dq_chunk, dq_local_tt);
                if (include_peer) {
                    warp::load_async(dq_peer_chunk, dq_peer_tt);
                }
                tensor_load_wait();
                if (include_peer) {
                    warp::add(dq_chunk, dq_chunk, dq_peer_chunk);
                }
                if (chunk > 0) {
                    warp::tma::store_async_wait();
                }
                warp::store(dq_smem[dq_subtile_idx], dq_chunk);
                if (chunk == 0) {
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                        g.dq0, dq_smem[dq_subtile_idx], {batch_idx, q_tile_idx, head_idx, 0}
                    );
                } else if (chunk == 1) {
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                        g.dq1, dq_smem[dq_subtile_idx], {batch_idx, q_tile_idx, head_idx, 1}
                    );
                } else {
                    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                        g.dq2, dq_smem[dq_subtile_idx], {batch_idx, q_tile_idx, head_idx, 2}
                    );
                }
            }
            warp::tma::store_async_wait();
            warp::arrive(dq_empty[stage]);
        }
    }
    __syncthreads();
}

template <typename C>
__global__ __launch_bounds__(kWarpThreads, 8)
void dq_only_clustered_reduce_kernel(const __grid_constant__ dq_only_clustered_globals<C> g) {
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;
    const int q_tile_idx = static_cast<int>(blockIdx.x);
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_idx = (q_tile_group_idx * C::ClusterSize) * kQSubtilesPerTile + q_subtile_in_group;
    const int peer_scratch_tile_idx = scratch_tile_idx + kQSubtilesPerTile;

    rt_fl<kRefTileM, C::Dqk> dq_local, dq_peer;
    warp::load(dq_local, g.dqacc, {batch_idx, head_idx, scratch_tile_idx, 0});
    warp::load(dq_peer, g.dqacc, {batch_idx, head_idx, peer_scratch_tile_idx, 0});
    warp::add(dq_local, dq_local, dq_peer);
    warp::store<dim::DEPTH>(g.dq, dq_local, {batch_idx, q_tile_idx, head_idx, 0});
}

template <typename C>
__global__ __launch_bounds__(3 * kWarpThreads, 4)
void dq_only_clustered_reduce_chunks_kernel(const __grid_constant__ dq_only_clustered_globals<C> g) {
    const int warp = threadIdx.x >> 5;
    if (warp >= 3) {
        return;
    }

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;
    const int q_tile_idx = static_cast<int>(blockIdx.x);
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_idx = (q_tile_group_idx * C::ClusterSize) * kQSubtilesPerTile + q_subtile_in_group;
    const int peer_scratch_tile_idx = scratch_tile_idx + kQSubtilesPerTile;

    rt_fl<kRefTileM, 64> dq_local, dq_peer;
    warp::load(dq_local, g.dqacc_chunks, {batch_idx, head_idx, scratch_tile_idx, warp});
    warp::load(dq_peer, g.dqacc_chunks, {batch_idx, head_idx, peer_scratch_tile_idx, warp});
    warp::add(dq_local, dq_local, dq_peer);
    if (warp == 0) {
        warp::store<dim::DEPTH>(g.dq0, dq_local, {batch_idx, q_tile_idx, head_idx, 0});
    } else if (warp == 1) {
        warp::store<dim::DEPTH>(g.dq1, dq_local, {batch_idx, q_tile_idx, head_idx, 1});
    } else {
        warp::store<dim::DEPTH>(g.dq2, dq_local, {batch_idx, q_tile_idx, head_idx, 2});
    }
}

template <typename C>
__global__ __launch_bounds__(C::QSubtiles * kWarpThreads, 1)
void dq_only_clustered_patch_reduce_kernel(const __grid_constant__ dq_only_clustered_globals<C> g) {
    using q_tile = typename dq_only_clustered_globals<C>::q_tile;
    using do_tile = typename dq_only_clustered_globals<C>::do_tile;
    using dqacc_tile = typename dq_only_clustered_globals<C>::dqacc_tile;
    using stats_smem_tile = typename dq_only_clustered_globals<C>::stats_tile;
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    __shared__ alignas(1024) q_tile q_smem[1];
    __shared__ alignas(1024) do_tile do_smem[1];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[C::QSubtiles];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[C::QSubtiles];
    __shared__ __align__(16) kittens::semaphore q_b;
    __shared__ __align__(16) kittens::semaphore o_b;

    const int warp = threadIdx.x >> 5;
    if (warp >= C::QSubtiles) {
        return;
    }

    const int q_block_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;
    const int q_tile_idx = q_block_idx * C::QSubtiles + warp;
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_idx = (q_tile_group_idx * C::ClusterSize) * kQSubtilesPerTile + q_subtile_in_group;
    const int peer_scratch_tile_idx = scratch_tile_idx + kQSubtilesPerTile;
    const bool needs_patch = (q_block_idx & 1) == 0;

    rt_fl<kRefTileM, C::Dqk> dq_local, dq_peer;
    warp::load(dq_local, g.dqacc, {batch_idx, head_idx, scratch_tile_idx, 0});
    warp::load(dq_peer, g.dqacc, {batch_idx, head_idx, peer_scratch_tile_idx, 0});
    warp::add(dq_local, dq_local, dq_peer);

    if (needs_patch && threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        init_semaphore(q_b, 0, 1);
        init_semaphore(o_b, 0, 1);
        tma::expect_bytes(q_b, sizeof(q_smem[0]));
        tma::expect_bytes(o_b, sizeof(do_smem[0]));
        coord<q_tile> q_tile_idx_coord = {batch_idx, q_block_idx, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx_coord, q_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, do_tile_idx, o_b);
    }
    __syncthreads();

    if (needs_patch) {
        wait(q_b, 0);
        wait(o_b, 0);
        stats_vec lse_stage_vec, dpsum_stage_vec;
        warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
        warp::store(lse_log2_smem[warp], lse_stage_vec);
        warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
        warp::store(dpsum_smem[warp], dpsum_stage_vec);
    }
    __syncthreads();

    if (needs_patch) {
        const int kv_first_subtile_idx = q_block_idx * C::QSubtiles;
        rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
        rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
        typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_vec, dpsum_vec;
        rt_fl<kRefTileM, C::Dqk> dq_patch, dq_contrib;

        auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({warp, 0});
        auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({warp, 0});
        warp::load(q_reg, q_subtile_smem);
        warp::load(do_reg, do_subtile_smem);
        warp::load(lse_vec, lse_log2_smem[warp]);
        warp::load(dpsum_vec, dpsum_smem[warp]);
        warp::zero(dq_patch);
        #pragma unroll
        for (int kv_offset = 0; kv_offset < C::QSubtiles; ++kv_offset) {
            const int kv_subtile_idx = kv_first_subtile_idx + kv_offset;
            warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
            warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
            tkfa4::bwd_cute16_kernel::detail::repair_dq_step<true, C>(
                dq_contrib,
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
            warp::add(dq_patch, dq_patch, dq_contrib);
        }
        warp::add(dq_local, dq_local, dq_patch);
    }

    warp::store<dim::DEPTH>(g.dq, dq_local, {batch_idx, q_tile_idx, head_idx, 0});
}

template <typename C>
__global__ __launch_bounds__(C::QSubtiles * kWarpThreads, 1)
void dq_only_clustered_first_block_patch_kernel(const __grid_constant__ dq_only_clustered_globals<C> g) {
    using q_tile = typename dq_only_clustered_globals<C>::q_tile;
    using do_tile = typename dq_only_clustered_globals<C>::do_tile;
    using dqacc_tile = typename dq_only_clustered_globals<C>::dqacc_tile;

    __shared__ alignas(1024) q_tile q_smem[1];
    __shared__ alignas(1024) do_tile do_smem[1];
    __shared__ alignas(1024) dqacc_tile dq_smem[C::QSubtiles];
    __shared__ __align__(16) kittens::semaphore q_b;
    __shared__ __align__(16) kittens::semaphore o_b;

    const int warp = threadIdx.x >> 5;
    if (warp >= C::QSubtiles) {
        return;
    }
    const int q_block_idx = static_cast<int>(blockIdx.x) * 2;
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;
    const int q_tile_idx = q_block_idx * C::QSubtiles + warp;
    const int kv_first_subtile_idx = q_block_idx * C::QSubtiles;
    const int kv_block_idx = q_block_idx / C::ConsumerWarpgroups;
    const int cluster_rank = kv_block_idx % C::ClusterSize;
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_idx =
        ((q_tile_group_idx * C::ClusterSize) + cluster_rank) * kQSubtilesPerTile + q_subtile_in_group;

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_vec, dpsum_vec;
    rt_fl<kRefTileM, C::Dqk> dq_partial;

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        init_semaphore(q_b, 0, 1);
        init_semaphore(o_b, 0, 1);
        tma::expect_bytes(q_b, sizeof(q_smem[0]));
        tma::expect_bytes(o_b, sizeof(do_smem[0]));
        coord<q_tile> q_tile_idx_coord = {batch_idx, q_block_idx, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx_coord, q_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, do_tile_idx, o_b);
    }
    __syncthreads();
    wait(q_b, 0);
    wait(o_b, 0);

    auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({warp, 0});
    auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({warp, 0});
    warp::load(q_reg, q_subtile_smem);
    warp::load(do_reg, do_subtile_smem);
    warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
    warp::zero(dq_partial);
    #pragma unroll
    for (int kv_offset = 0; kv_offset < C::QSubtiles; ++kv_offset) {
        if (kv_offset > warp) {
            continue;
        }
        const int kv_subtile_idx = kv_first_subtile_idx + kv_offset;
        warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
        repair_dq_step_accumulate<true, false, C>(
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
            g.seq_len
        );
    }

    warp::store(dq_smem[warp], dq_partial);
    __syncwarp();
    warp::tma::store_add_async(g.dqacc, dq_smem[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
    warp::tma::store_async_wait();
}

template <typename C>
__global__ __launch_bounds__(C::QSubtiles * kWarpThreads, 1)
void dq_only_clustered_first_block_patch_direct_kernel(const __grid_constant__ dq_only_clustered_globals<C> g) {
    using q_tile = typename dq_only_clustered_globals<C>::q_tile;
    using do_tile = typename dq_only_clustered_globals<C>::do_tile;
    using dqacc_tile = typename dq_only_clustered_globals<C>::dqacc_tile;

    __shared__ alignas(1024) q_tile q_smem[1];
    __shared__ alignas(1024) do_tile do_smem[1];
    __shared__ alignas(1024) dqacc_tile dq_smem[C::QSubtiles];
    __shared__ __align__(16) kittens::semaphore q_b;
    __shared__ __align__(16) kittens::semaphore o_b;

    const int warp = threadIdx.x >> 5;
    if (warp >= C::QSubtiles) {
        return;
    }
    const int q_block_idx = static_cast<int>(blockIdx.x) * 2;
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int q_tile_idx = q_block_idx * C::QSubtiles + warp;
    const int kv_first_subtile_idx = q_block_idx * C::QSubtiles;

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_vec, dpsum_vec;
    rt_fl<kRefTileM, C::Dqk> dq_partial;

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        init_semaphore(q_b, 0, 1);
        init_semaphore(o_b, 0, 1);
        tma::expect_bytes(q_b, sizeof(q_smem[0]));
        tma::expect_bytes(o_b, sizeof(do_smem[0]));
        coord<q_tile> q_tile_idx_coord = {batch_idx, q_block_idx, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx_coord, q_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, do_tile_idx, o_b);
    }
    __syncthreads();
    wait(q_b, 0);
    wait(o_b, 0);

    auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({warp, 0});
    auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({warp, 0});
    warp::load(q_reg, q_subtile_smem);
    warp::load(do_reg, do_subtile_smem);
    warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
    warp::zero(dq_partial);
    #pragma unroll
    for (int kv_offset = 0; kv_offset < C::QSubtiles; ++kv_offset) {
        if (kv_offset > warp) {
            continue;
        }
        const int kv_subtile_idx = kv_first_subtile_idx + kv_offset;
        warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
        repair_dq_step_accumulate<true, false, C>(
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
            g.seq_len
        );
    }

    warp::store(dq_smem[warp], dq_partial);
    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq_full, dq_smem[warp], {batch_idx, q_tile_idx, head_idx, 0});
    warp::tma::store_async_wait();
}

template <typename C>
__global__ __launch_bounds__(C::QSubtiles * kWarpThreads, 1)
void dq_only_clustered_second_block_patch_kernel(const __grid_constant__ dq_only_clustered_globals<C> g) {
    using q_tile = typename dq_only_clustered_globals<C>::q_tile;
    using do_tile = typename dq_only_clustered_globals<C>::do_tile;
    using dqacc_tile = typename dq_only_clustered_globals<C>::dqacc_tile;

    __shared__ alignas(1024) q_tile q_smem;
    __shared__ alignas(1024) do_tile do_smem;
    __shared__ alignas(1024) dqacc_tile dq_smem[C::QSubtiles];
    __shared__ __align__(16) kittens::semaphore q_b;
    __shared__ __align__(16) kittens::semaphore o_b;

    const int warp = threadIdx.x >> 5;
    if (warp >= C::QSubtiles) {
        return;
    }
    const int q_block_idx = static_cast<int>(blockIdx.x) * 2 + 1;
    if ((q_block_idx & 2) == 0) {
        return;
    }
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;
    const int q_tile_idx = q_block_idx * C::QSubtiles + warp;
    const int kv_subtile_idx = q_tile_idx;
    const int kv_block_idx = q_block_idx / C::ConsumerWarpgroups;
    const int cluster_rank = kv_block_idx % C::ClusterSize;
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_idx =
        ((q_tile_group_idx * C::ClusterSize) + cluster_rank) * kQSubtilesPerTile + q_subtile_in_group;

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_vec, dpsum_vec;
    rt_fl<kRefTileM, C::Dqk> dq_partial;

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        init_semaphore(q_b, 0, 1);
        init_semaphore(o_b, 0, 1);
        tma::expect_bytes(q_b, sizeof(q_smem));
        tma::expect_bytes(o_b, sizeof(do_smem));
        coord<q_tile> q_tile_idx_coord = {batch_idx, q_block_idx, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem, g.q, q_tile_idx_coord, q_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem, g.dout, do_tile_idx, o_b);
    }
    __syncthreads();
    wait(q_b, 0);
    wait(o_b, 0);

    auto q_subtile_smem = q_smem.template subtile<kRefTileM, C::Dqk>({warp, 0});
    auto do_subtile_smem = do_smem.template subtile<kRefTileM, C::Dvo>({warp, 0});
    warp::load(q_reg, q_subtile_smem);
    warp::load(do_reg, do_subtile_smem);
    warp::load(lse_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
    warp::zero(dq_partial);
    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    tkfa4::bwd_cute16_kernel::detail::repair_dq_step<true, C>(
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
    warp::store(dq_smem[warp], dq_partial);
    __syncwarp();
    warp::tma::store_add_async(g.dqacc, dq_smem[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
    warp::tma::store_async_wait();
}

template <typename C>
__global__ __launch_bounds__(C::QSubtiles * kWarpThreads, 1)
void dq_only_clustered_second_block_patch_kernel_full(const __grid_constant__ dq_only_clustered_globals<C> g) {
    using q_tile = typename dq_only_clustered_globals<C>::q_tile;
    using do_tile = typename dq_only_clustered_globals<C>::do_tile;
    using dqacc_tile = typename dq_only_clustered_globals<C>::dqacc_tile;
    using stats_smem_tile = typename dq_only_clustered_globals<C>::stats_tile;

    __shared__ alignas(1024) q_tile q_smem[1];
    __shared__ alignas(1024) do_tile do_smem[1];
    __shared__ alignas(1024) dqacc_tile dq_smem[C::QSubtiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[C::QSubtiles];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[C::QSubtiles];
    __shared__ __align__(16) kittens::semaphore q_b;
    __shared__ __align__(16) kittens::semaphore o_b;

    const int warp = threadIdx.x >> 5;
    if (warp >= C::QSubtiles) {
        return;
    }
    const int q_block_idx = static_cast<int>(blockIdx.x) * 2 + 1;
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;
    const int q_tile_idx = q_block_idx * C::QSubtiles + warp;
    const int kv_first_subtile_idx = (q_block_idx - 1) * C::QSubtiles;
    const int kv_block_idx = q_block_idx / C::ConsumerWarpgroups;
    const int cluster_rank = kv_block_idx % C::ClusterSize;
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_idx =
        ((q_tile_group_idx * C::ClusterSize) + cluster_rank) * kQSubtilesPerTile + q_subtile_in_group;

    rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
    typename rt_fl<kRefTileM, kRefTileN>::col_vec lse_vec, dpsum_vec;
    rt_fl<kRefTileM, C::Dqk> dq_partial, dq_contrib;
    using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();
        init_semaphore(q_b, 0, 1);
        init_semaphore(o_b, 0, 1);
        tma::expect_bytes(q_b, sizeof(q_smem[0]));
        tma::expect_bytes(o_b, sizeof(do_smem[0]));
        coord<q_tile> q_tile_idx_coord = {batch_idx, q_block_idx, head_idx, 0};
        coord<do_tile> do_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx_coord, q_b);
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, do_tile_idx, o_b);
    }
    __syncthreads();
    wait(q_b, 0);
    wait(o_b, 0);

    if (warp < C::QSubtiles) {
        stats_vec lse_stage_vec, dpsum_stage_vec;
        warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
        warp::store(lse_log2_smem[warp], lse_stage_vec);
        warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
        warp::store(dpsum_smem[warp], dpsum_stage_vec);
    }
    __syncthreads();

    auto q_subtile_smem = q_smem[0].template subtile<kRefTileM, C::Dqk>({warp, 0});
    auto do_subtile_smem = do_smem[0].template subtile<kRefTileM, C::Dvo>({warp, 0});
    warp::load(q_reg, q_subtile_smem);
    warp::load(do_reg, do_subtile_smem);
    warp::load(lse_vec, lse_log2_smem[warp]);
    warp::load(dpsum_vec, dpsum_smem[warp]);
    warp::zero(dq_partial);
    #pragma unroll
    for (int kv_offset = 0; kv_offset < C::QSubtiles * C::ConsumerWarpgroups; ++kv_offset) {
        const int kv_subtile_idx = kv_first_subtile_idx + kv_offset;
        warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
        warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
        tkfa4::bwd_cute16_kernel::detail::repair_dq_step<true, C>(
            dq_contrib,
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
        warp::add(dq_partial, dq_partial, dq_contrib);
    }

    warp::store(dq_smem[warp], dq_partial);
    __syncwarp();
    warp::tma::store_add_async(g.dqacc, dq_smem[warp], {batch_idx, head_idx, scratch_tile_idx, 0});
    warp::tma::store_async_wait();
}

template <typename C>
__global__ __launch_bounds__(kWarpThreads, 8)
void dq_reduce_kernel(const __grid_constant__ dq_reduce_globals<C> g) {
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int q_tile_idx = static_cast<int>(blockIdx.x);
    const int scratch_tile_idx = q_tile_idx * C::ClusterSize;

    rt_fl<kRefTileM, C::Dqk> dq_local, dq_peer;
    warp::load(dq_local, g.dqacc, {batch_idx, head_idx, scratch_tile_idx, 0});
    if constexpr (C::ClusterSize == 2) {
        warp::load(dq_peer, g.dqacc, {batch_idx, head_idx, scratch_tile_idx + 1, 0});
        warp::add(dq_local, dq_local, dq_peer);
    }
    warp::store<dim::DEPTH>(g.dq, dq_local, {batch_idx, q_tile_idx, head_idx, 0});
}

template <bool CAUSAL, typename C>
inline void launch_main(
    const main_globals<C> &g,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream,
    bool use_fullseq_specialization = false
) {
    if constexpr (C::ClusterSize == 2) {
        kittens::LaunchConfig<true, false> launch_config(
            dim3(num_k_blocks, heads, batch_size),
            dim3(C::BlockThreads, 1, 1),
            0,
            stream,
            dim3(C::ClusterSize, 1, 1)
        );
        if constexpr (CAUSAL) {
            if (use_fullseq_specialization) {
                CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel_causal_fullseq<C>, g));
                return;
            }
            CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel<true, C>, g));
        } else {
            CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel<false, C>, g));
        }
        return;
    }

    dim3 grid(num_k_blocks, heads, batch_size);
    if constexpr (CAUSAL) {
        if (use_fullseq_specialization) {
            main_kernel_causal_fullseq<C><<<grid, C::BlockThreads, 0, stream>>>(g);
        } else {
            main_kernel<true, C><<<grid, C::BlockThreads, 0, stream>>>(g);
        }
    } else {
        main_kernel<false, C><<<grid, C::BlockThreads, 0, stream>>>(g);
    }
}

template <typename C>
inline void launch_main_shared_ds_exact(
    const shared_ds_monolithic_globals<C> &g,
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
        CUDACHECK(cudaLaunchKernelEx(launch_config, main_kernel_causal_fullseq_shared_ds_exact<C>, g));
        return;
    }

    dim3 grid(num_k_blocks, heads, batch_size);
    main_kernel_causal_fullseq_shared_ds_exact<C><<<grid, C::BlockThreads, 0, stream>>>(g);
}

template <typename C>
inline void launch_reduce(
    const main_globals<C> &g,
    int q_tiles,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    const dq_reduce_globals<C> reduce_g{g.dqacc, g.dq};
    dim3 grid(q_tiles, heads, batch_size);
    dq_reduce_kernel<C><<<grid, kWarpThreads, 0, stream>>>(reduce_g);
}

template <typename C>
inline void launch_reduce(
    const dq_reduce_globals<C> &g,
    int q_tiles,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid(q_tiles, heads, batch_size);
    dq_reduce_kernel<C><<<grid, kWarpThreads, 0, stream>>>(g);
}

}  // namespace detail

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
    at::Tensor &dqacc,
    at::Tensor *dq_semaphore,
    bool causal,
    float scale,
    bool deterministic,
    [[maybe_unused]] bool apply_causal_patches = true
) {
    TORCH_CHECK(!deterministic, "Candidate exact clustered path currently supports deterministic=False only");

    using G = main_globals<C>;
    const int q_tiles = static_cast<int>(q.size(1) / kRefTileM);
    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));
    const int scratch_rows = q_tiles * C::ClusterSize;

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q_fix),
        kittens::py::tensor_to_gl<typename G::k_gl>(k_fix),
        kittens::py::tensor_to_gl<typename G::v_gl>(v_fix),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout_fix),
        kittens::py::tensor_to_gl<typename G::dqacc_chunk_gl>(dqacc),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dqacc.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(1)),
        dq_semaphore != nullptr && dq_semaphore->defined() ? reinterpret_cast<int *>(dq_semaphore->data_ptr<int>()) : nullptr,
        static_cast<int>(q.size(2)),
        q_tiles,
        num_k_blocks / C::ClusterSize,
        deterministic ? 1 : 0,
    };

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool use_fullseq_specialization = causal;
    if (causal) {
        detail::launch_main<true, C>(
            g,
            num_k_blocks,
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(0)),
            stream,
            use_fullseq_specialization
        );
    } else {
        detail::launch_main<false, C>(g, num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    detail::launch_reduce<C>(g, q_tiles, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_backward_shared_ds_exact(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream = nullptr
) {
    using G = shared_ds_monolithic_globals<C>;
    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
    };

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
    CUDACHECK(cudaMemsetAsync(
        dq.data_ptr<float>(),
        0,
        static_cast<size_t>(dq.numel()) * sizeof(float),
        stream
    ));
    detail::launch_main_shared_ds_exact<C>(
        g,
        num_k_blocks,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C, typename DkdvOutT = float>
inline void launch_backward_dkdv_only(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream = nullptr
) {
    using G = dkdv_only_globals<C, DkdvOutT>;
    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
    };

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
    if constexpr (C::ClusterSize == 2) {
        kittens::LaunchConfig<true, false> launch_config(
            dim3(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0))),
            dim3(C::BlockThreads, 1, 1),
            0,
            stream,
            dim3(C::ClusterSize, 1, 1)
        );
        CUDACHECK(cudaLaunchKernelEx(launch_config, ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dkdv_only<C, DkdvOutT>, g));
    } else {
        dim3 grid(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
        ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dkdv_only<C, DkdvOutT><<<grid, C::BlockThreads, 0, stream>>>(g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_backward_dkdv_only_store_ds(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &ds,
    float scale,
    cudaStream_t stream = nullptr
) {
    using G = dkdv_only_ds_globals<C>;
    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::ds_gl>(ds),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
    };

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
    if constexpr (C::ClusterSize == 2) {
        kittens::LaunchConfig<true, false> launch_config(
            dim3(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0))),
            dim3(C::BlockThreads, 1, 1),
            0,
            stream,
            dim3(C::ClusterSize, 1, 1)
        );
        CUDACHECK(cudaLaunchKernelEx(launch_config, ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dkdv_only_store_ds<C>, g));
    } else {
        dim3 grid(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
        ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dkdv_only_store_ds<C><<<grid, C::BlockThreads, 0, stream>>>(g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_backward_seq2048_exact_dkdv_only(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream = nullptr
) {
    using G = seq2048_exact_dkdv_globals<C>;
    const int num_k_blocks = static_cast<int>(q.size(1) / (C::TileRows * C::ConsumerWarpgroups));

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
    };

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
    kittens::LaunchConfig<true, false> launch_config(
        dim3(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0))),
        dim3(C::DkdvBlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_seq2048_exact_dkdv_only<C>,
        g
    ));
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <
    typename C,
    int DenseSplitCount = 1,
    bool UseTmemScoreDp = false,
    bool UseTmemFrontier = false,
    bool FuseDenseDq = false,
    bool AdaptiveLastQuarter = false,
    bool OverlapLoadAndDqReduce = false,
    bool SkipAdaptiveTailScratch = false,
    bool UseLdsmTransposeDs = false,
    bool DoubleBufferFusedDqTma = false,
    bool MaterializeDkdvBf16 = false,
    bool ReleaseTmemOperandsEachIteration = false,
    bool SerializeDenseFrontier = false
>
inline void launch_backward_dense_tmem_frontier_dkdv(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream = nullptr,
    cudaStream_t dv_frontier_stream = nullptr,
    cudaEvent_t dense_main_done = nullptr,
    cudaEvent_t dv_frontier_done = nullptr,
    at::Tensor *dk_frontier_scratch = nullptr,
    at::Tensor *dv_frontier_scratch = nullptr,
    at::Tensor *dense_split_dk_scratch = nullptr,
    at::Tensor *dense_split_dv_scratch = nullptr,
    at::Tensor *fused_dq = nullptr,
    at::Tensor *materialized_dk = nullptr,
    at::Tensor *materialized_dv = nullptr
) {
    static_assert(
        DenseSplitCount == 1 || DenseSplitCount == 2 || DenseSplitCount == 3 ||
            DenseSplitCount == 4 || DenseSplitCount == 8,
        "Dense split count must be 1, 2, 3, 4, or 8"
    );
    static_assert(
        !AdaptiveLastQuarter || DenseSplitCount == 2,
        "Adaptive last-quarter ownership requires split-2 dense main"
    );
    static_assert(
        !OverlapLoadAndDqReduce || FuseDenseDq,
        "Load/reducer overlap requires fused dQ"
    );
    static_assert(
        !SkipAdaptiveTailScratch || AdaptiveLastQuarter,
        "Skipping tail scratch requires adaptive last-quarter ownership"
    );
    static_assert(
        !MaterializeDkdvBf16 || (DenseSplitCount == 2 && !SkipAdaptiveTailScratch),
        "BF16 materialization currently supports the complete split-2 merge"
    );
    using G = dense_tmem_frontier_globals<C>;
    const int num_k_blocks = static_cast<int>(
        q.size(1) / (C::TileRows * C::ConsumerWarpgroups)
    );
    at::Tensor *dq_tensor = &dk;
    if constexpr (FuseDenseDq) {
        TORCH_CHECK(fused_dq != nullptr && fused_dq->defined(), "fused dQ output is required");
        TORCH_CHECK(fused_dq->sizes() == q.sizes(), "fused dQ output shape mismatch");
        dq_tensor = fused_dq;
    }
    typename G::dq_out_gl dq_gl =
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(*dq_tensor);

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        dq_gl,
        dq_gl,
        dq_gl,
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(0)),
    };
    typename G::dk_gl dense_split_dk_gl = g.dk;
    typename G::dv_gl dense_split_dv_gl = g.dv;
    if constexpr (DenseSplitCount > 1) {
        TORCH_CHECK(
            dense_split_dk_scratch != nullptr && dense_split_dk_scratch->defined(),
            "dense split dK scratch is required"
        );
        TORCH_CHECK(
            dense_split_dv_scratch != nullptr && dense_split_dv_scratch->defined(),
            "dense split dV scratch is required"
        );
        TORCH_CHECK(
            dense_split_dk_scratch->numel() >= (DenseSplitCount - 1) * dk.numel(),
            "dense split dK scratch shape mismatch"
        );
        TORCH_CHECK(
            dense_split_dv_scratch->numel() >= (DenseSplitCount - 1) * dv.numel(),
            "dense split dV scratch shape mismatch"
        );
        dense_split_dk_gl = kittens::py::tensor_to_gl<typename G::dk_gl>(*dense_split_dk_scratch);
        dense_split_dv_gl = kittens::py::tensor_to_gl<typename G::dv_gl>(*dense_split_dv_scratch);
    }

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }

    const bool overlap_frontier =
        dk_frontier_scratch != nullptr &&
        dv_frontier_scratch != nullptr &&
        dk_frontier_scratch->defined() &&
        dv_frontier_scratch->defined() &&
        dv_frontier_stream != nullptr &&
        dv_frontier_stream != stream &&
        dense_main_done != nullptr &&
        dv_frontier_done != nullptr;
    const bool serialize_dense_frontier =
        SerializeDenseFrontier &&
        overlap_frontier &&
        q.size(1) >= 8192 &&
        q.size(2) >= 2;
    G frontier_g = g;
    if (overlap_frontier) {
        TORCH_CHECK(dk_frontier_scratch->numel() == dk.numel(), "frontier dK scratch shape mismatch");
        TORCH_CHECK(dv_frontier_scratch->numel() == dv.numel(), "frontier dV scratch shape mismatch");
        frontier_g.dk = kittens::py::tensor_to_gl<typename G::dk_gl>(*dk_frontier_scratch);
        frontier_g.dv = kittens::py::tensor_to_gl<typename G::dv_gl>(*dv_frontier_scratch);
        if (!serialize_dense_frontier) {
            CUDACHECK(cudaEventRecord(dense_main_done, stream));
            CUDACHECK(cudaStreamWaitEvent(dv_frontier_stream, dense_main_done));
        }
    }
    if constexpr (UseTmemFrontier) {
        TORCH_CHECK(overlap_frontier, "TMEM frontier requires split frontier scratch and stream");
    }
    if constexpr (SkipAdaptiveTailScratch) {
        TORCH_CHECK(q.size(0) == 1, "Skipping adaptive tail scratch requires batch size 1");
    }
    if constexpr (MaterializeDkdvBf16) {
        TORCH_CHECK(
            overlap_frontier,
            "BF16 materialization requires distinct-stream frontier scratch"
        );
        TORCH_CHECK(
            materialized_dk != nullptr && materialized_dk->defined() &&
                materialized_dv != nullptr && materialized_dv->defined(),
            "BF16 materialized dK/dV outputs are required"
        );
        TORCH_CHECK(
            materialized_dk->sizes() == dk.sizes() && materialized_dv->sizes() == dv.sizes(),
            "BF16 materialized dK/dV output shape mismatch"
        );
        TORCH_CHECK(
            materialized_dk->scalar_type() == at::kBFloat16 &&
                materialized_dv->scalar_type() == at::kBFloat16,
            "BF16 materialized dK/dV outputs must use bfloat16"
        );
        TORCH_CHECK(
            materialized_dk->is_contiguous() && materialized_dv->is_contiguous(),
            "BF16 materialized dK/dV outputs must be contiguous"
        );
        TORCH_CHECK(
            materialized_dk->device() == dk.device() &&
                materialized_dv->device() == dv.device(),
            "BF16 materialized outputs and FP32 accumulators must share a device"
        );
        TORCH_CHECK(
            materialized_dk->data_ptr() != dk.data_ptr() &&
                materialized_dv->data_ptr() != dv.data_ptr(),
            "BF16 materialized outputs must not alias FP32 accumulators"
        );
    }

    int dense_grid_x = num_k_blocks * DenseSplitCount;
    if constexpr (AdaptiveLastQuarter) {
        dense_grid_x = num_k_blocks + (num_k_blocks * 3) / 4;
    }
    kittens::LaunchConfig<true, false> dense_launch_config(
        dim3(
            dense_grid_x,
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(0))
        ),
        dim3(
            FuseDenseDq
                ? (OverlapLoadAndDqReduce
                    ? C::FusedDqLoadOverlapBlockThreads
                    : C::FusedDqBlockThreads)
                : C::DenseBlockThreads,
            1,
            1
        ),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        dense_launch_config,
        ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_dense_tmem_dkdv<
            C,
            DenseSplitCount,
            UseTmemScoreDp,
            false,
            FuseDenseDq,
            AdaptiveLastQuarter,
            OverlapLoadAndDqReduce,
            SkipAdaptiveTailScratch,
            UseLdsmTransposeDs,
            DoubleBufferFusedDqTma,
            ReleaseTmemOperandsEachIteration
        >,
        g,
        dense_split_dk_gl,
        dense_split_dv_gl
    ));
    if (serialize_dense_frontier) {
        CUDACHECK(cudaEventRecord(dense_main_done, stream));
        CUDACHECK(cudaStreamWaitEvent(dv_frontier_stream, dense_main_done));
    }

    const bool parallel_frontier = !overlap_frontier &&
        dv_frontier_stream != nullptr &&
        dv_frontier_stream != stream &&
        dense_main_done != nullptr &&
        dv_frontier_done != nullptr;
    if (parallel_frontier) {
        CUDACHECK(cudaEventRecord(dense_main_done, stream));
        CUDACHECK(cudaStreamWaitEvent(dv_frontier_stream, dense_main_done));
    } else if (!overlap_frontier) {
        dv_frontier_stream = stream;
    }

    dim3 dk_frontier_grid(
        num_k_blocks * 3,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0))
    );
    dim3 dv_frontier_grid(
        num_k_blocks,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0))
    );
    if (overlap_frontier) {
        if constexpr (UseTmemFrontier) {
            kittens::LaunchConfig<true, false> frontier_launch_config(
                dv_frontier_grid,
                dim3(
                    FuseDenseDq
                        ? (OverlapLoadAndDqReduce
                            ? C::FusedDqLoadOverlapBlockThreads
                            : C::FusedDqBlockThreads)
                        : C::DenseBlockThreads,
                    1,
                    1
                ),
                0,
                dv_frontier_stream,
                dim3(C::ClusterSize, 1, 1)
            );
            CUDACHECK(cudaLaunchKernelEx(
                frontier_launch_config,
                ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_dense_tmem_dkdv<
                    C,
                    1,
                    true,
                    true,
                    FuseDenseDq,
                    false,
                    OverlapLoadAndDqReduce,
                    false,
                    UseLdsmTransposeDs,
                    DoubleBufferFusedDqTma,
                    ReleaseTmemOperandsEachIteration
                >,
                frontier_g,
                frontier_g.dk,
                frontier_g.dv
            ));
        } else {
            ::tkfa4::bwd_cute16_kernel_candidate::detail::causal_dense_tmem_frontier_patch_kernel<
                C,
                true,
                false,
                false
            ><<<dk_frontier_grid, C::FrontierBlockThreads, 0, dv_frontier_stream>>>(frontier_g);
            ::tkfa4::bwd_cute16_kernel_candidate::detail::causal_dense_tmem_frontier_patch_kernel<
                C,
                false,
                true,
                false
            ><<<dv_frontier_grid, C::FrontierBlockThreads, 0, dv_frontier_stream>>>(frontier_g);
        }
        CUDACHECK(cudaEventRecord(dv_frontier_done, dv_frontier_stream));
        CUDACHECK(cudaStreamWaitEvent(stream, dv_frontier_done));
        constexpr int kThreads = 256;
        TORCH_CHECK(dk.numel() % 4 == 0 && dv.numel() % 4 == 0, "frontier add requires float4-aligned shapes");
        const int64_t dk_vecs = dk.numel() / 4;
        const int64_t dv_vecs = dv.numel() / 4;
        const int64_t max_vecs = dk_vecs > dv_vecs ? dk_vecs : dv_vecs;
        const int blocks = static_cast<int>((max_vecs + kThreads - 1) / kThreads);
        if constexpr (DenseSplitCount == 2) {
            if constexpr (MaterializeDkdvBf16) {
                ::tkfa4::bwd_cute16_kernel_candidate::detail::dense_tmem_two_scratch_add_to_bf16_kernel
                    <<<blocks, kThreads, 0, stream>>>(
                        reinterpret_cast<bf16_2 *>(materialized_dk->data_ptr()),
                        reinterpret_cast<const float4 *>(dk.data_ptr()),
                        reinterpret_cast<const float4 *>(dense_split_dk_scratch->data_ptr()),
                        reinterpret_cast<const float4 *>(dk_frontier_scratch->data_ptr()),
                        dk_vecs,
                        reinterpret_cast<bf16_2 *>(materialized_dv->data_ptr()),
                        reinterpret_cast<const float4 *>(dv.data_ptr()),
                        reinterpret_cast<const float4 *>(dense_split_dv_scratch->data_ptr()),
                        reinterpret_cast<const float4 *>(dv_frontier_scratch->data_ptr()),
                        dv_vecs
                    );
            } else if constexpr (SkipAdaptiveTailScratch) {
                ::tkfa4::bwd_cute16_kernel_candidate::detail::dense_tmem_adaptive_two_scratch_add_kernel
                    <<<blocks, kThreads, 0, stream>>>(
                        reinterpret_cast<float4 *>(dk.data_ptr()),
                        reinterpret_cast<const float4 *>(dense_split_dk_scratch->data_ptr()),
                        reinterpret_cast<const float4 *>(dk_frontier_scratch->data_ptr()),
                        dk_vecs,
                        (dk_vecs * 3) / 4,
                        reinterpret_cast<float4 *>(dv.data_ptr()),
                        reinterpret_cast<const float4 *>(dense_split_dv_scratch->data_ptr()),
                        reinterpret_cast<const float4 *>(dv_frontier_scratch->data_ptr()),
                        dv_vecs,
                        (dv_vecs * 3) / 4
                    );
            } else {
                ::tkfa4::bwd_cute16_kernel_candidate::detail::dense_tmem_two_scratch_add_kernel
                    <<<blocks, kThreads, 0, stream>>>(
                        reinterpret_cast<float4 *>(dk.data_ptr()),
                        reinterpret_cast<const float4 *>(dense_split_dk_scratch->data_ptr()),
                        reinterpret_cast<const float4 *>(dk_frontier_scratch->data_ptr()),
                        dk_vecs,
                        reinterpret_cast<float4 *>(dv.data_ptr()),
                        reinterpret_cast<const float4 *>(dense_split_dv_scratch->data_ptr()),
                        reinterpret_cast<const float4 *>(dv_frontier_scratch->data_ptr()),
                        dv_vecs
                    );
            }
        } else if constexpr (DenseSplitCount > 2) {
            ::tkfa4::bwd_cute16_kernel_candidate::detail::dense_tmem_multi_scratch_add_kernel<
                DenseSplitCount
            ><<<blocks, kThreads, 0, stream>>>(
                reinterpret_cast<float4 *>(dk.data_ptr()),
                reinterpret_cast<const float4 *>(dense_split_dk_scratch->data_ptr()),
                reinterpret_cast<const float4 *>(dk_frontier_scratch->data_ptr()),
                dk_vecs,
                reinterpret_cast<float4 *>(dv.data_ptr()),
                reinterpret_cast<const float4 *>(dense_split_dv_scratch->data_ptr()),
                reinterpret_cast<const float4 *>(dv_frontier_scratch->data_ptr()),
                dv_vecs
            );
        } else {
            ::tkfa4::bwd_cute16_kernel_candidate::detail::dense_tmem_frontier_add_kernel
                <<<blocks, kThreads, 0, stream>>>(
                    reinterpret_cast<float4 *>(dk.data_ptr()),
                    reinterpret_cast<const float4 *>(dk_frontier_scratch->data_ptr()),
                    dk_vecs,
                    reinterpret_cast<float4 *>(dv.data_ptr()),
                    reinterpret_cast<const float4 *>(dv_frontier_scratch->data_ptr()),
                    dv_vecs
                );
        }
    } else {
        ::tkfa4::bwd_cute16_kernel_candidate::detail::causal_dense_tmem_frontier_patch_kernel<C, true, false>
            <<<dk_frontier_grid, C::FrontierBlockThreads, 0, stream>>>(g);
        ::tkfa4::bwd_cute16_kernel_candidate::detail::causal_dense_tmem_frontier_patch_kernel<C, false, true>
            <<<dv_frontier_grid, C::FrontierBlockThreads, 0, dv_frontier_stream>>>(g);
    }
    if (parallel_frontier) {
        CUDACHECK(cudaEventRecord(dv_frontier_done, dv_frontier_stream));
        CUDACHECK(cudaStreamWaitEvent(stream, dv_frontier_done));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C, bool kUseDirectDq = true>
inline void launch_backward_dq_only(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    float scale,
    cudaStream_t stream = nullptr,
    at::Tensor *dqacc_opt = nullptr
) {
    using MainG = dq_only_globals<C>;
    using ReduceG = dq_reduce_globals<C>;
    const int q_tiles = static_cast<int>(q.size(1) / kRefTileM);
    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));
    const int scratch_rows = static_cast<int>(q.size(1) * C::ClusterSize);
    at::Tensor dqacc;
    if constexpr (!kUseDirectDq) {
        dqacc = dqacc_opt ? *dqacc_opt : at::zeros({q.size(0), q.size(2), scratch_rows, C::Dqk}, lse_log2.options());
    }

    MainG main_g{
        kittens::py::tensor_to_gl<typename MainG::q_gl>(q),
        kittens::py::tensor_to_gl<typename MainG::k_gl>(k),
        kittens::py::tensor_to_gl<typename MainG::v_gl>(v),
        kittens::py::tensor_to_gl<typename MainG::do_gl>(dout),
        kUseDirectDq
            ? kittens::py::make_fake_gl<typename MainG::dqacc_chunk_gl>(1, 1, 1, 64)
            : kittens::py::tensor_to_gl<typename MainG::dqacc_chunk_gl>(dqacc),
        kUseDirectDq
            ? kittens::py::make_fake_gl<typename MainG::dqacc_gl>(1, 1, 1, C::Dqk)
            : ::kittens::make_gl<typename MainG::dqacc_gl>(
                reinterpret_cast<uint64_t>(dqacc.data_ptr<float>()),
                static_cast<int>(q.size(0)),
                static_cast<int>(q.size(2)),
                scratch_rows,
                C::Dqk
            ),
        kittens::py::tensor_to_gl<typename MainG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename MainG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename MainG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename MainG::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename MainG::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
    };
    if constexpr (!kUseDirectDq) {
        ReduceG reduce_g{
            ::kittens::make_gl<typename ReduceG::dqacc_gl>(
                reinterpret_cast<uint64_t>(dqacc.data_ptr<float>()),
                static_cast<int>(q.size(0)),
                static_cast<int>(q.size(2)),
                scratch_rows,
                C::Dqk
            ),
            kittens::py::tensor_to_gl<typename ReduceG::dq_gl>(dq),
        };
        if (stream == nullptr) {
            stream = at::cuda::getCurrentCUDAStream().stream();
        }
        if constexpr (C::ClusterSize == 2) {
            kittens::LaunchConfig<true, false> launch_config(
                dim3(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0))),
                dim3(C::BlockThreads, 1, 1),
                0,
                stream,
                dim3(C::ClusterSize, 1, 1)
            );
            CUDACHECK(cudaLaunchKernelEx(launch_config, ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only<C, kUseDirectDq>, main_g));
        } else {
            dim3 grid(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
            ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only<C, kUseDirectDq><<<grid, C::BlockThreads, 0, stream>>>(main_g);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        detail::launch_reduce<C>(reduce_g, q_tiles, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
        CHECK_CUDA_ERROR(cudaGetLastError());
        return;
    }

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
    if constexpr (C::ClusterSize == 2) {
        kittens::LaunchConfig<true, false> launch_config(
            dim3(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0))),
            dim3(C::BlockThreads, 1, 1),
            0,
            stream,
            dim3(C::ClusterSize, 1, 1)
        );
        CUDACHECK(cudaLaunchKernelEx(launch_config, ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only<C, kUseDirectDq>, main_g));
    } else {
        dim3 grid(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
        ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only<C, kUseDirectDq><<<grid, C::BlockThreads, 0, stream>>>(main_g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_backward_dq_from_ds(
    at::Tensor &k,
    at::Tensor &ds,
    at::Tensor &dq,
    cudaStream_t stream = nullptr
) {
    using G = dq_from_ds_globals<C>;
    const int num_k_blocks = static_cast<int>(k.size(1) / (kRefTileN * C::WarpTiles));

    G g{
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::ds_gl>(ds),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        static_cast<int>(k.size(1)),
    };

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
    if constexpr (C::ClusterSize == 2) {
        kittens::LaunchConfig<true, false> launch_config(
            dim3(num_k_blocks, static_cast<int>(k.size(2)), static_cast<int>(k.size(0))),
            dim3(C::BlockThreads, 1, 1),
            0,
            stream,
            dim3(C::ClusterSize, 1, 1)
        );
        CUDACHECK(cudaLaunchKernelEx(launch_config, ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_from_ds<C>, g));
    } else {
        dim3 grid(num_k_blocks, static_cast<int>(k.size(2)), static_cast<int>(k.size(0)));
        ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_from_ds<C><<<grid, C::BlockThreads, 0, stream>>>(g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C, typename DqOutT = float, bool UseChunkedTmemDq = false>
inline void launch_backward_dq_only_clustered(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    float scale,
    cudaStream_t stream = nullptr,
    at::Tensor *dqacc_opt = nullptr
) {
    using G = dq_only_clustered_globals<C, DqOutT>;
    const int q_tiles = static_cast<int>(q.size(1) / kRefTileM);
    const int num_k_blocks = static_cast<int>(q.size(1) / (C::TileRows * C::ConsumerWarpgroups));
    const int scratch_rows = static_cast<int>(q.size(1) * C::ClusterSize);
    at::Tensor dqacc = dqacc_opt ? *dqacc_opt : at::zeros({q.size(0), q.size(2), scratch_rows, C::Dqk}, lse_log2.options());

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dqacc_chunk_gl>(dqacc),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dqacc.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_full_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(2)),
        q_tiles,
    };

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
    const bool enable_timing = detail::clustered_dq_timing_enabled();
    cudaEvent_t dq_main_start = nullptr;
    cudaEvent_t dq_main_end = nullptr;
    cudaEvent_t dq_patch_end = nullptr;
    cudaEvent_t dq_reduce_end = nullptr;
    if (enable_timing) {
        CUDACHECK(cudaEventCreate(&dq_main_start));
        CUDACHECK(cudaEventCreate(&dq_main_end));
        CUDACHECK(cudaEventCreate(&dq_patch_end));
        CUDACHECK(cudaEventCreate(&dq_reduce_end));
        CUDACHECK(cudaEventRecord(dq_main_start, stream));
    }
    cudaLaunchAttribute launch_attrs[3];
    launch_attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    launch_attrs[0].val.preferredClusterDim.x = C::ClusterSize;
    launch_attrs[0].val.preferredClusterDim.y = 1;
    launch_attrs[0].val.preferredClusterDim.z = 1;
    launch_attrs[1].id = cudaLaunchAttributeClusterDimension;
    launch_attrs[1].val.clusterDim.x = C::ClusterSize;
    launch_attrs[1].val.clusterDim.y = 1;
    launch_attrs[1].val.clusterDim.z = 1;
    launch_attrs[2].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    launch_attrs[2].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = dim3(num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
    launch_config.blockDim = dim3(C::BlockThreads, 1, 1);
    launch_config.dynamicSmemBytes = 0;
    launch_config.stream = stream;
    launch_config.attrs = launch_attrs;
    launch_config.numAttrs = 3;
    if constexpr (kDqOnlyClusteredMode == dq_only_clustered_mode::DonorBulkOnly) {
        if constexpr (C::ClusterSize == 1) {
            ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only_clustered<
                dq_only_clustered_mode::DonorBulkOnly,
                C,
                DqOutT,
                UseChunkedTmemDq
            ><<<launch_config.gridDim, launch_config.blockDim, 0, stream>>>(g);
        } else {
            CUDACHECK(cudaLaunchKernelEx(
                &launch_config,
                ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only_clustered<
                    dq_only_clustered_mode::DonorBulkOnly,
                    C,
                    DqOutT,
                    UseChunkedTmemDq
                >,
                g
            ));
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        if (enable_timing) {
            CUDACHECK(cudaEventRecord(dq_main_end, stream));
        }
        dim3 reduce_grid(q_tiles, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
        if constexpr (kUseChunkedClusteredDqReduce) {
            ::tkfa4::bwd_cute16_kernel_candidate::detail::dq_only_clustered_reduce_chunks_kernel<C><<<
                reduce_grid,
                3 * kWarpThreads,
                0,
                stream
            >>>(g);
        } else {
            ::tkfa4::bwd_cute16_kernel_candidate::detail::dq_only_clustered_reduce_kernel<C><<<reduce_grid, kWarpThreads, 0, stream>>>(g);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        if (enable_timing) {
            float main_ms = 0.0f, reduce_ms = 0.0f, total_ms = 0.0f;
            CUDACHECK(cudaEventRecord(dq_reduce_end, stream));
            CUDACHECK(cudaEventSynchronize(dq_reduce_end));
            CUDACHECK(cudaEventElapsedTime(&main_ms, dq_main_start, dq_main_end));
            CUDACHECK(cudaEventElapsedTime(&reduce_ms, dq_main_end, dq_reduce_end));
            CUDACHECK(cudaEventElapsedTime(&total_ms, dq_main_start, dq_reduce_end));
            std::fprintf(
                stderr,
                "clustered_dq_timing_us main=%.2f patch=0.00 reduce=%.2f total=%.2f\n",
                main_ms * 1000.0f,
                reduce_ms * 1000.0f,
                total_ms * 1000.0f
            );
            cudaEventDestroy(dq_main_start);
            cudaEventDestroy(dq_main_end);
            cudaEventDestroy(dq_patch_end);
            cudaEventDestroy(dq_reduce_end);
        }
        return;
    }

    if constexpr (C::ClusterSize == 1) {
        ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only_clustered<
            dq_only_clustered_mode::LegacyPatched,
            C,
            DqOutT,
            UseChunkedTmemDq
        ><<<launch_config.gridDim, launch_config.blockDim, 0, stream>>>(g);
    } else {
        CUDACHECK(cudaLaunchKernelEx(
            &launch_config,
            ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only_clustered<
                dq_only_clustered_mode::LegacyPatched,
                C,
                DqOutT,
                UseChunkedTmemDq
            >,
            g
        ));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    if (enable_timing) {
        CUDACHECK(cudaEventRecord(dq_main_end, stream));
    }
    if constexpr (kUseDirectFinalClusteredDq) {
        if constexpr (!kUseMainFirstBlockClusteredDq) {
            dim3 patch_grid(static_cast<int>(q.size(1) / (2 * C::TileRows)), static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
            ::tkfa4::bwd_cute16_kernel_candidate::detail::dq_only_clustered_first_block_patch_direct_kernel<C><<<
                patch_grid,
                dim3(C::QSubtiles * kWarpThreads, 1, 1),
                0,
                stream
            >>>(g);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
        if (enable_timing) {
            float main_ms = 0.0f, patch_ms = 0.0f, total_ms = 0.0f;
            CUDACHECK(cudaEventRecord(dq_patch_end, stream));
            CUDACHECK(cudaEventSynchronize(dq_patch_end));
            CUDACHECK(cudaEventElapsedTime(&main_ms, dq_main_start, dq_main_end));
            CUDACHECK(cudaEventElapsedTime(&patch_ms, dq_main_end, dq_patch_end));
            CUDACHECK(cudaEventElapsedTime(&total_ms, dq_main_start, dq_patch_end));
            std::fprintf(
                stderr,
                "clustered_dq_timing_us main=%.2f patch=%.2f reduce=0.00 total=%.2f\n",
                main_ms * 1000.0f,
                patch_ms * 1000.0f,
                total_ms * 1000.0f
            );
            cudaEventDestroy(dq_main_start);
            cudaEventDestroy(dq_main_end);
            cudaEventDestroy(dq_patch_end);
            cudaEventDestroy(dq_reduce_end);
        }
        return;
    }
    if constexpr (kUseFusedPatchReduceClusteredDq) {
        dim3 patch_reduce_grid(q_tiles / C::QSubtiles, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
        ::tkfa4::bwd_cute16_kernel_candidate::detail::dq_only_clustered_patch_reduce_kernel<C><<<
            patch_reduce_grid,
            dim3(C::QSubtiles * kWarpThreads, 1, 1),
            0,
            stream
        >>>(g);
        CHECK_CUDA_ERROR(cudaGetLastError());
        if (enable_timing) {
            float main_ms = 0.0f, reduce_ms = 0.0f, total_ms = 0.0f;
            CUDACHECK(cudaEventRecord(dq_reduce_end, stream));
            CUDACHECK(cudaEventSynchronize(dq_reduce_end));
            CUDACHECK(cudaEventElapsedTime(&main_ms, dq_main_start, dq_main_end));
            CUDACHECK(cudaEventElapsedTime(&reduce_ms, dq_main_end, dq_reduce_end));
            CUDACHECK(cudaEventElapsedTime(&total_ms, dq_main_start, dq_reduce_end));
            std::fprintf(
                stderr,
                "clustered_dq_timing_us main=%.2f patch=0.00 reduce=%.2f total=%.2f\n",
                main_ms * 1000.0f,
                reduce_ms * 1000.0f,
                total_ms * 1000.0f
            );
            cudaEventDestroy(dq_main_start);
            cudaEventDestroy(dq_main_end);
            cudaEventDestroy(dq_patch_end);
            cudaEventDestroy(dq_reduce_end);
        }
    } else {
        if constexpr (!kUseMainFirstBlockClusteredDq) {
            dim3 patch_grid(static_cast<int>(q.size(1) / (2 * C::TileRows)), static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
            ::tkfa4::bwd_cute16_kernel_candidate::detail::dq_only_clustered_first_block_patch_kernel<C><<<
                patch_grid,
                dim3(C::QSubtiles * kWarpThreads, 1, 1),
                0,
                stream
            >>>(g);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
        if (enable_timing) {
            CUDACHECK(cudaEventRecord(dq_patch_end, stream));
        }
        dim3 reduce_grid(q_tiles, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
        if constexpr (kUseChunkedClusteredDqReduce) {
            ::tkfa4::bwd_cute16_kernel_candidate::detail::dq_only_clustered_reduce_chunks_kernel<C><<<
                reduce_grid,
                3 * kWarpThreads,
                0,
                stream
            >>>(g);
        } else {
            ::tkfa4::bwd_cute16_kernel_candidate::detail::dq_only_clustered_reduce_kernel<C><<<reduce_grid, kWarpThreads, 0, stream>>>(g);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        if (enable_timing) {
            float main_ms = 0.0f, patch_ms = 0.0f, reduce_ms = 0.0f, total_ms = 0.0f;
            CUDACHECK(cudaEventRecord(dq_reduce_end, stream));
            CUDACHECK(cudaEventSynchronize(dq_reduce_end));
            CUDACHECK(cudaEventElapsedTime(&main_ms, dq_main_start, dq_main_end));
            CUDACHECK(cudaEventElapsedTime(&patch_ms, dq_main_end, dq_patch_end));
            CUDACHECK(cudaEventElapsedTime(&reduce_ms, dq_patch_end, dq_reduce_end));
            CUDACHECK(cudaEventElapsedTime(&total_ms, dq_main_start, dq_reduce_end));
            std::fprintf(
                stderr,
                "clustered_dq_timing_us main=%.2f patch=%.2f reduce=%.2f total=%.2f\n",
                main_ms * 1000.0f,
                patch_ms * 1000.0f,
                reduce_ms * 1000.0f,
                total_ms * 1000.0f
            );
            cudaEventDestroy(dq_main_start);
            cudaEventDestroy(dq_main_end);
            cudaEventDestroy(dq_patch_end);
            cudaEventDestroy(dq_reduce_end);
        }
    }
}

template <
    typename C,
    typename DqOutT = float,
    bool DoubleBufferInputs = false,
    int DqReplaySplitCount = 1
>
inline void launch_backward_dq_only_clustered_pipelined(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    float scale,
    cudaStream_t stream = nullptr,
    at::Tensor *dqacc_opt = nullptr
) {
    static_assert(C::ClusterSize == 1, "Pipelined clustered dQ currently requires ClusterSize=1");
    static_assert(
        DqReplaySplitCount == 1 || DqReplaySplitCount == 2,
        "Pipelined dQ replay split count must be 1 or 2"
    );
    using G = dq_only_clustered_globals<C, DqOutT>;
    const int q_tiles = static_cast<int>(q.size(1) / kRefTileM);
    const int num_k_blocks = static_cast<int>(q.size(1) / (C::TileRows * C::ConsumerWarpgroups));
    const int scratch_rows = static_cast<int>(q.size(1) * C::ClusterSize);
    at::Tensor dqacc = dqacc_opt
        ? *dqacc_opt
        : at::zeros({q.size(0), q.size(2), scratch_rows, C::Dqk}, lse_log2.options());

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dqacc_chunk_gl>(dqacc),
        ::kittens::make_gl<typename G::dqacc_gl>(
            reinterpret_cast<uint64_t>(dqacc.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            scratch_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_full_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(2)),
        q_tiles,
    };

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    if (detail::clustered_dq_timing_enabled()) {
        CUDACHECK(cudaEventCreate(&start));
        CUDACHECK(cudaEventCreate(&end));
        CUDACHECK(cudaEventRecord(start, stream));
    }

    dim3 grid(
        num_k_blocks * DqReplaySplitCount,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0))
    );
    ::tkfa4::bwd_cute16_kernel_candidate::detail::main_kernel_causal_fullseq_dq_only_clustered_pipelined<
        C,
        DqOutT,
        DoubleBufferInputs,
        DqReplaySplitCount
    ><<<grid, C::BlockThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());

    if (detail::clustered_dq_timing_enabled()) {
        float elapsed_ms = 0.0f;
        CUDACHECK(cudaEventRecord(end, stream));
        CUDACHECK(cudaEventSynchronize(end));
        CUDACHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
        std::fprintf(stderr, "clustered_dq_pipelined_timing_us main=%.2f\n", elapsed_ms * 1000.0f);
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
}

namespace detail {

using cta2_fused_dense_stats_gl = gl<float, -1, -1, -1, -1>;
template <typename T>
using cta2_fused_dense_output_gl_t = gl<T, -1, -1, -1, -1>;
using cta2_fused_dense_output_gl = cta2_fused_dense_output_gl_t<float>;
using cta2_fused_dense_dq_stage = st_fl<32, 32>;
using cta2_fused_dense_dq_gl = gl<
    float,
    -1,
    -1,
    -1,
    -1,
    tma::descriptor<cta2_fused_dense_dq_stage, dim::DEPTH>
>;

using cta2_fused_dense_k_tile = st_bf<128, kB300QKDim>;
using cta2_fused_dense_q_tile = st_bf<64, kB300QKDim>;
using cta2_fused_dense_q_normal0_tile = st_bf<128, 64>;
using cta2_fused_dense_q_normal1_tile = st_bf<128, 32>;
using cta2_fused_dense_q_normal_wide_tile = st_bf<128, 96>;
using cta2_fused_dense_v_tile = st_bf<128, kB300VDim>;
using cta2_fused_dense_do_tile = st_bf<64, kB300VDim>;
using cta2_fused_dense_k_gl = gl<
    bf16,
    -1,
    -1,
    -1,
    -1,
    tma::descriptor<cta2_fused_dense_k_tile, dim::DEPTH>
>;
using cta2_fused_dense_q_gl = gl<
    bf16,
    -1,
    -1,
    -1,
    -1,
    tma::descriptor<cta2_fused_dense_q_tile, dim::DEPTH>,
    tma::descriptor<st_bf<64, 64>, dim::DEPTH>
>;
using cta2_fused_dense_q_dk_tma_gl = gl<
    bf16,
    -1,
    -1,
    -1,
    -1,
    tma::descriptor<cta2_fused_dense_q_tile, dim::DEPTH>,
    tma::descriptor<st_bf<64, 64>, dim::DEPTH>,
    tma::descriptor<cta2_fused_dense_q_normal0_tile, dim::DEPTH>,
    tma::descriptor<cta2_fused_dense_q_normal1_tile, dim::DEPTH>,
    tma::descriptor<cta2_fused_dense_q_normal_wide_tile, dim::DEPTH>
>;
using cta2_fused_dense_v_gl = gl<
    bf16,
    -1,
    -1,
    -1,
    -1,
    tma::descriptor<cta2_fused_dense_v_tile, dim::DEPTH>
>;
using cta2_fused_dense_do_gl = gl<
    bf16,
    -1,
    -1,
    -1,
    -1,
    tma::descriptor<cta2_fused_dense_do_tile, dim::DEPTH>,
    tma::descriptor<st_bf<64, 64>, dim::DEPTH>
>;

template <
    typename DkdvOutT,
    bool DirectTmaDkQ = false
>
struct cta2_fused_dense_globals_t {
    using q_gl = std::conditional_t<
        DirectTmaDkQ,
        cta2_fused_dense_q_dk_tma_gl,
        cta2_fused_dense_q_gl
    >;
    cta2_fused_dense_k_gl k;
    q_gl q;
    cta2_fused_dense_v_gl v;
    cta2_fused_dense_do_gl dout;
    cta2_fused_dense_output_gl_t<DkdvOutT> dk;
    cta2_fused_dense_output_gl_t<DkdvOutT> dv;
    cta2_fused_dense_dq_gl dq;
    cta2_fused_dense_stats_gl lse_log2;
    cta2_fused_dense_stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

using cta2_fused_dense_globals = cta2_fused_dense_globals_t<float>;

template <typename QGL>
__device__ __forceinline__ void cta2_fused_dense_load_qdo_elected_pair(
    cta2_fused_dense_q_tile &q_dst,
    const QGL &q_src,
    cta2_fused_dense_do_tile &do_dst,
    const cta2_fused_dense_do_gl &do_src,
    const coord<cta2_fused_dense_q_tile> &tile_coord,
    semaphore &bar,
    uint32_t additional_expected_bytes = 0
) {
    static_assert(
        cta2_fused_dense_q_tile::rows ==
            cta2_fused_dense_do_tile::rows
    );
    const uint64_t q_tma = reinterpret_cast<uint64_t>(
        q_src.template get_tma<cta2_fused_dense_q_tile, dim::DEPTH>()
    );
    const uint64_t do_tma = reinterpret_cast<uint64_t>(
        do_src.template get_tma<cta2_fused_dense_do_tile, dim::DEPTH>()
    );
    const uint32_t q_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&q_dst)
    );
    const uint32_t do_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&do_dst)
    );
    const uint32_t bar_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&bar)
    );
    const auto unit_coord =
        tile_coord.template unit_coord<dim::DEPTH, 3>();
    const int4 tma_coord = ::kittens::tma::detail::tma_coords<
        cta2_fused_dense_q_tile,
        dim::DEPTH
    >(unit_coord);
    const uint32_t expected_bytes =
        sizeof(cta2_fused_dense_q_tile) +
        sizeof(cta2_fused_dense_do_tile) +
        additional_expected_bytes;

    asm volatile(
        "{\n"
        ".reg .pred leader;\n"
        "elect.sync _|leader, -1;\n"
        "@leader mbarrier.arrive.expect_tx.shared::cta.b64 "
        "_, [%4], %9;\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %5, %6, %7, %8}], [%4];\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%2], [%3, {0, %5, %6, %7, %8}], [%4];\n"
        "}\n"
        :
        : "r"(q_smem),
          "l"(q_tma),
          "r"(do_smem),
          "l"(do_tma),
          "r"(bar_smem),
          "r"(tma_coord.x),
          "r"(tma_coord.y),
          "r"(tma_coord.z),
          "r"(tma_coord.w),
          "r"(expected_bytes)
        : "memory"
    );
}

__device__ __forceinline__ void
cta2_fused_dense_arm_elected_transaction(
    semaphore &bar,
    uint32_t expected_bytes
) {
    const uint32_t bar_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&bar)
    );
    asm volatile(
        "{\n"
        ".reg .pred leader;\n"
        "elect.sync _|leader, -1;\n"
        "@leader mbarrier.arrive.expect_tx.shared::cta.b64 "
        "_, [%0], %1;\n"
        "}\n"
        :
        : "r"(bar_smem), "r"(expected_bytes)
        : "memory"
    );
}

template <typename QGL>
__device__ __forceinline__ void
cta2_fused_dense_load_qdo_elected_pair_no_arm(
    cta2_fused_dense_q_tile &q_dst,
    const QGL &q_src,
    cta2_fused_dense_do_tile &do_dst,
    const cta2_fused_dense_do_gl &do_src,
    const coord<cta2_fused_dense_q_tile> &tile_coord,
    semaphore &bar
) {
    static_assert(
        cta2_fused_dense_q_tile::rows ==
            cta2_fused_dense_do_tile::rows
    );
    const uint64_t q_tma = reinterpret_cast<uint64_t>(
        q_src.template get_tma<cta2_fused_dense_q_tile, dim::DEPTH>()
    );
    const uint64_t do_tma = reinterpret_cast<uint64_t>(
        do_src.template get_tma<cta2_fused_dense_do_tile, dim::DEPTH>()
    );
    const uint32_t q_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&q_dst)
    );
    const uint32_t do_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&do_dst)
    );
    const uint32_t bar_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&bar)
    );
    const auto unit_coord =
        tile_coord.template unit_coord<dim::DEPTH, 3>();
    const int4 tma_coord = ::kittens::tma::detail::tma_coords<
        cta2_fused_dense_q_tile,
        dim::DEPTH
    >(unit_coord);

    asm volatile(
        "{\n"
        ".reg .pred leader;\n"
        "elect.sync _|leader, -1;\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %5, %6, %7, %8}], [%4];\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%2], [%3, {0, %5, %6, %7, %8}], [%4];\n"
        "}\n"
        :
        : "r"(q_smem),
          "l"(q_tma),
          "r"(do_smem),
          "l"(do_tma),
          "r"(bar_smem),
          "r"(tma_coord.x),
          "r"(tma_coord.y),
          "r"(tma_coord.z),
          "r"(tma_coord.w)
        : "memory"
    );
}

template <typename QGL>
__device__ __forceinline__ void cta2_fused_dense_load_dk_q_wide_elected(
    cta2_fused_dense_q_normal_wide_tile &q_dst,
    const QGL &q_src,
    const coord<cta2_fused_dense_q_normal_wide_tile> &tile_coord,
    semaphore &bar
) {
    const uint64_t q_tma = reinterpret_cast<uint64_t>(
        q_src.template get_tma<
            cta2_fused_dense_q_normal_wide_tile,
            dim::DEPTH
        >()
    );
    const uint32_t q_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&q_dst)
    );
    const uint32_t bar_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&bar)
    );
    const auto unit_coord =
        tile_coord.template unit_coord<dim::DEPTH, 3>();
    const int4 tma_coord = ::kittens::tma::detail::tma_coords<
        cta2_fused_dense_q_normal_wide_tile,
        dim::DEPTH
    >(unit_coord);
    const uint32_t expected_bytes =
        sizeof(cta2_fused_dense_q_normal_wide_tile);

    asm volatile(
        "{\n"
        ".reg .pred leader;\n"
        "elect.sync _|leader, -1;\n"
        "@leader mbarrier.arrive.expect_tx.shared::cta.b64 "
        "_, [%2], %7;\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %3, %4, %5, %6}], [%2];\n"
        "}\n"
        :
        : "r"(q_smem),
          "l"(q_tma),
          "r"(bar_smem),
          "r"(tma_coord.x),
          "r"(tma_coord.y),
          "r"(tma_coord.z),
          "r"(tma_coord.w),
          "r"(expected_bytes)
        : "memory"
    );
}

__device__ __forceinline__ void cta2_fused_dense_load_score_k_elected(
    cta2_fused_dense_k_tile &k_dst,
    const cta2_fused_dense_k_gl &k_src,
    const coord<cta2_fused_dense_k_tile> &tile_coord,
    semaphore &bar
) {
    const uint64_t k_tma = reinterpret_cast<uint64_t>(
        k_src.template get_tma<cta2_fused_dense_k_tile, dim::DEPTH>()
    );
    const uint32_t k_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&k_dst)
    );
    const uint32_t bar_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&bar)
    );
    const auto unit_coord =
        tile_coord.template unit_coord<dim::DEPTH, 3>();
    const int4 tma_coord = ::kittens::tma::detail::tma_coords<
        cta2_fused_dense_k_tile,
        dim::DEPTH
    >(unit_coord);

    asm volatile(
        "{\n"
        ".reg .pred leader;\n"
        "elect.sync _|leader, -1;\n"
        "@leader mbarrier.arrive.expect_tx.shared::cta.b64 "
        "_, [%2], %7;\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %3, %4, %5, %6}], [%2];\n"
        "}\n"
        :
        : "r"(k_smem),
          "l"(k_tma),
          "r"(bar_smem),
          "r"(tma_coord.x),
          "r"(tma_coord.y),
          "r"(tma_coord.z),
          "r"(tma_coord.w),
          "r"(static_cast<uint32_t>(sizeof(cta2_fused_dense_k_tile)))
        : "memory"
    );
}

__device__ __forceinline__ void cta2_fused_dense_load_score_k_v_elected(
    cta2_fused_dense_k_tile &k_dst,
    const cta2_fused_dense_k_gl &k_src,
    const coord<cta2_fused_dense_k_tile> &k_tile_coord,
    cta2_fused_dense_v_tile &v_dst,
    const cta2_fused_dense_v_gl &v_src,
    const coord<cta2_fused_dense_v_tile> &v_tile_coord,
    semaphore &bar
) {
    const uint64_t k_tma = reinterpret_cast<uint64_t>(
        k_src.template get_tma<cta2_fused_dense_k_tile, dim::DEPTH>()
    );
    const uint64_t v_tma = reinterpret_cast<uint64_t>(
        v_src.template get_tma<cta2_fused_dense_v_tile, dim::DEPTH>()
    );
    const uint32_t k_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&k_dst)
    );
    const uint32_t v_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&v_dst)
    );
    const uint32_t bar_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&bar)
    );
    const int4 k_tma_coord = ::kittens::tma::detail::tma_coords<
        cta2_fused_dense_k_tile,
        dim::DEPTH
    >(k_tile_coord.template unit_coord<dim::DEPTH, 3>());
    const int4 v_tma_coord = ::kittens::tma::detail::tma_coords<
        cta2_fused_dense_v_tile,
        dim::DEPTH
    >(v_tile_coord.template unit_coord<dim::DEPTH, 3>());

    asm volatile(
        "{\n"
        ".reg .pred leader;\n"
        "elect.sync _|leader, -1;\n"
        "@leader mbarrier.arrive.expect_tx.shared::cta.b64 "
        "_, [%4], %13;\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %5, %6, %7, %8}], [%4];\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%2], [%3, {0, %9, %10, %11, %12}], [%4];\n"
        "}\n"
        :
        : "r"(k_smem),
          "l"(k_tma),
          "r"(v_smem),
          "l"(v_tma),
          "r"(bar_smem),
          "r"(k_tma_coord.x),
          "r"(k_tma_coord.y),
          "r"(k_tma_coord.z),
          "r"(k_tma_coord.w),
          "r"(v_tma_coord.x),
          "r"(v_tma_coord.y),
          "r"(v_tma_coord.z),
          "r"(v_tma_coord.w),
          "r"(
              static_cast<uint32_t>(
                  sizeof(cta2_fused_dense_k_tile) +
                  sizeof(cta2_fused_dense_v_tile)
              )
          )
        : "memory"
    );
}

using cta2_fused_dense_p_tile = st_bf<128, 128>;
using cta2_fused_dense_ds_tile = st_bf<128, 128>;
using cta2_fused_dense_q_t0_tile = st_bf<64, 128>;
using cta2_fused_dense_q_t1_tile = st_bf<32, 128>;
using cta2_fused_dense_q_t_wide_tile = st_bf<96, 128>;
using cta2_fused_dense_dq_a_tile = st_bf<256, 64>;
using cta2_fused_dense_dq_b_tile = st_bf<256, 96>;

__device__ __forceinline__ void
cta2_fused_dense_load_scaled_dq_k(
    cta2_fused_dense_dq_b_tile &dst,
    const cta2_fused_dense_k_gl &src,
    const coord<cta2_fused_dense_dq_b_tile> &tile_coord,
    float scale
) {
    constexpr int kGroupThreads = 8 * WARP_THREADS;
    constexpr int kElementsPerVector =
        sizeof(float4) / sizeof(bf16);
    constexpr int kVectorsPerRow =
        cta2_fused_dense_dq_b_tile::cols / kElementsPerVector;
    constexpr int kVectorCalls =
        (
            cta2_fused_dense_dq_b_tile::rows *
                cta2_fused_dense_dq_b_tile::cols +
            kGroupThreads * kElementsPerVector - 1
        ) /
        (kGroupThreads * kElementsPerVector);

    const auto unit_coord =
        tile_coord.template unit_coord<dim::DEPTH, 3>();
    bf16 *src_ptr = reinterpret_cast<bf16 *>(&src[unit_coord]);
    const int row_stride = src.template stride<dim::DEPTH>();
    const uint32_t dst_ptr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&dst.data[0])
    );
    const int group_lane = threadIdx.x % kGroupThreads;
    const bf16_2 scale_pair =
        __floats2bfloat162_rn(scale, scale);

    #pragma unroll
    for (int call = 0; call < kVectorCalls; ++call) {
        const int load_idx = call * kGroupThreads + group_lane;
        const int row = load_idx / kVectorsPerRow;
        const int col =
            (load_idx * kElementsPerVector) %
            cta2_fused_dense_dq_b_tile::cols;
        float4 payload;
        move<float4>::ldg(
            payload,
            reinterpret_cast<float4 *>(
                &src_ptr[row * row_stride + col]
            )
        );
        bf16_2 *pairs = reinterpret_cast<bf16_2 *>(&payload);
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            pairs[pair] = __hmul2(pairs[pair], scale_pair);
        }
        move<float4>::sts(
            cta2_fused_dense_dq_b_tile::idx(
                dst_ptr,
                {row, col}
            ),
            payload
        );
    }
}

using cta2_fused_dense_dq_exchange_tile = st_bf<128, 64>;
using cta2_fused_dense_k_exchange_tile = st_bf<128, 96>;
using cta2_fused_dense_q0_exchange_tile = st_bf<64, 64>;
using cta2_fused_dense_q1_exchange_tile = st_bf<64, 64>;
using cta2_fused_dense_do_exchange_tile = st_bf<64, 64>;
using cta2_fused_dense_attn_tt = full_tt_fl<128>;
using cta2_fused_dense_ds_tt = full_tt_bf<128>;
using cta2_fused_dense_dk_tail_tt = full_tt_fl<64>;
using cta2_fused_dense_dk_wide_tt = full_tt_fl<kB300QKDim>;
using cta2_fused_dense_dq_tt = half_tt_fl<kB300QKDim>;
using cta2_fused_dense_attn_reg = rt_fl<16, 128>;
using cta2_fused_dense_attn_half_reg = rt_fl<16, 64>;
using cta2_fused_dense_attn_quarter_reg = rt_fl<16, 32>;
using cta2_fused_dense_dk_tail_reg = rt_fl<16, 64>;
using cta2_fused_dense_attn_bf_reg = rt_bf<16, 128>;
using cta2_fused_dense_attn_half_bf_reg = rt_bf<16, 64>;
using cta2_fused_dense_attn_quarter_bf_reg = rt_bf<16, 32>;
using cta2_fused_dense_attn_half_tt = full_tt_fl<64>;
using cta2_fused_dense_ds_half_tt = full_tt_bf<64>;
using cta2_fused_dense_attn_quarter_tt = full_tt_fl<32>;
using cta2_fused_dense_ds_quarter_tt = full_tt_bf<32>;
using cta2_fused_dense_stats_vec = typename cta2_fused_dense_attn_reg::row_vec;
using cta2_fused_dense_half_stats_vec =
    typename cta2_fused_dense_attn_half_reg::row_vec;
using cta2_fused_dense_quarter_stats_vec =
    typename cta2_fused_dense_attn_quarter_reg::row_vec;

struct cta2_fused_dense_q_transposed_storage {
    cta2_fused_dense_q_t0_tile q0;
    cta2_fused_dense_q_t1_tile q1;
};

struct cta2_fused_dense_q_normal_storage {
    cta2_fused_dense_q_normal0_tile q0;
    cta2_fused_dense_q_normal1_tile q1;
};

union cta2_fused_dense_q_operand_storage {
    cta2_fused_dense_q_tile q;
    cta2_fused_dense_q_transposed_storage transposed;
    cta2_fused_dense_q_t_wide_tile transposed_wide;
    cta2_fused_dense_q_normal_storage normal;
    cta2_fused_dense_q_normal_wide_tile normal_wide;
};

static_assert(
    sizeof(cta2_fused_dense_q_operand_storage) == sizeof(cta2_fused_dense_q_tile)
);

union cta2_fused_dense_p_operand_storage {
    cta2_fused_dense_p_tile p;
    cta2_fused_dense_do_tile next_dout;
};

struct cta2_fused_dense_qdo_exchange_storage {
    cta2_fused_dense_q0_exchange_tile q0;
    cta2_fused_dense_q1_exchange_tile q1;
    cta2_fused_dense_do_exchange_tile dout;
};

union cta2_fused_dense_qdo_phase_storage {
    cta2_fused_dense_qdo_exchange_storage qdo;
    cta2_fused_dense_dq_exchange_tile ds;
};

template <bool CacheLse>
struct cta2_fused_dense_stats_storage;

template <>
struct cta2_fused_dense_stats_storage<false> {
    sv_fl<128> dpsum;
};

template <>
struct cta2_fused_dense_stats_storage<true> {
    sv_fl<128> lse_log2;
    sv_fl<128> dpsum;
};

struct cta2_fused_dense_main_phase_storage {
    cta2_fused_dense_q_operand_storage q;
    cta2_fused_dense_do_tile dout;
    cta2_fused_dense_p_operand_storage p;
};

union cta2_fused_dense_exchange_storage {
    cta2_fused_dense_dq_exchange_tile ds;
    cta2_fused_dense_k_exchange_tile k;
    cta2_fused_dense_q_tile next_q;
};

struct cta2_fused_dense_dq_operands {
    cta2_fused_dense_dq_a_tile a;
    cta2_fused_dense_exchange_storage exchange;
};

struct cta2_fused_dense_dq_reduce {
    cta2_fused_dense_dq_stage stage[2][4];
};

static_assert(
    sizeof(cta2_fused_dense_dq_reduce) <=
        sizeof(cta2_fused_dense_p_operand_storage)
);

union cta2_fused_dense_dq_storage {
    cta2_fused_dense_dq_operands operands;
    cta2_fused_dense_dq_reduce reduce;
};

union cta2_fused_dense_phase_storage {
    cta2_fused_dense_main_phase_storage main;
    cta2_fused_dense_dq_storage dq;
};

template <bool CacheLse>
struct cta2_fused_dense_shared_storage {
    cta2_fused_dense_k_tile k;
    cta2_fused_dense_dq_b_tile dq_b;
    cta2_fused_dense_phase_storage phase;
    cta2_fused_dense_ds_tile ds;
    cta2_fused_dense_qdo_phase_storage qdo_phase;
    cta2_fused_dense_stats_storage<CacheLse> stats;
};

template <
    bool UseWarpStatsCache,
    bool PipelineLsePrefetch,
    bool UseX32StatsCache = false
>
struct cta2_fused_dense_role_split_shared_storage;

template <>
struct cta2_fused_dense_role_split_shared_storage<false, false, false>
    : cta2_fused_dense_shared_storage<false> {};

template <>
struct cta2_fused_dense_role_split_shared_storage<true, false, false>
    : cta2_fused_dense_shared_storage<true> {
    alignas(16) semaphore stats_ready;
    alignas(16) semaphore stats_consumed;
};

template <>
struct cta2_fused_dense_role_split_shared_storage<true, true, false>
    : cta2_fused_dense_shared_storage<true> {
    sv_fl<128> lse_log2_next;
    alignas(16) semaphore stats_ready;
    alignas(16) semaphore stats_consumed;
    alignas(16) semaphore dk_done;
};

template <>
struct cta2_fused_dense_role_split_shared_storage<false, false, true>
    : cta2_fused_dense_shared_storage<true> {
    alignas(16) semaphore stats_consumed;
    alignas(16) semaphore dk_done;
};

template <>
struct cta2_fused_dense_role_split_shared_storage<true, false, true>
    : cta2_fused_dense_shared_storage<true> {
    cta2_fused_dense_stats_storage<true> stats_next;
    alignas(16) semaphore dk_done;
};

static_assert(
    sizeof(cta2_fused_dense_role_split_shared_storage<false, false, false>) ==
        sizeof(cta2_fused_dense_shared_storage<false>)
);

template <
    bool UseWideDkN192 = false,
    bool LoadPeerQ = true,
    typename DkdvOutT = float,
    bool DirectTmaDkQ = false
>
__device__ __forceinline__ void cta2_fused_dense_load_peer_qdo(
    const cta2_fused_dense_globals_t<DkdvOutT, DirectTmaDkQ> &g,
    cta2_fused_dense_qdo_exchange_storage &dst,
    semaphore &barrier,
    int batch_idx,
    int q_tile_idx,
    int head_idx,
    int cta_rank
) {
    const int peer_rank = cta_rank ^ 1;
    tma::expect_bytes(
        barrier,
        LoadPeerQ ? sizeof(dst) : sizeof(dst.dout)
    );
    if constexpr (LoadPeerQ) {
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            dst.q0,
            g.q,
            coord<cta2_fused_dense_q0_exchange_tile>{
                batch_idx,
                q_tile_idx * 2 + peer_rank,
                head_idx,
                cta_rank
            },
            barrier
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            dst.q1,
            g.q,
            coord<cta2_fused_dense_q1_exchange_tile>{
                batch_idx,
                q_tile_idx * 2 + peer_rank,
                head_idx,
                UseWideDkN192 ? 1 + cta_rank : 2
            },
            barrier
        );
    }
    tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
        dst.dout,
        g.dout,
        coord<cta2_fused_dense_do_exchange_tile>{
            batch_idx,
            q_tile_idx * 2 + peer_rank,
            head_idx,
            cta_rank
        },
        barrier
    );
}

template <
    typename DkdvOutT,
    bool DirectTmaDkQ
>
__device__ __forceinline__ void cta2_fused_dense_load_peer_do_elected(
    const cta2_fused_dense_globals_t<DkdvOutT, DirectTmaDkQ> &g,
    cta2_fused_dense_qdo_exchange_storage &dst,
    semaphore &barrier,
    int batch_idx,
    int q_tile_idx,
    int head_idx,
    int cta_rank
) {
    const int peer_rank = cta_rank ^ 1;
    const uint64_t do_tma = reinterpret_cast<uint64_t>(
        g.dout.template get_tma<
            cta2_fused_dense_do_exchange_tile,
            dim::DEPTH
        >()
    );
    const uint32_t do_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&dst.dout)
    );
    const uint32_t bar_smem = static_cast<uint32_t>(
        __cvta_generic_to_shared(&barrier)
    );
    const coord<cta2_fused_dense_do_exchange_tile> tile_coord{
        batch_idx,
        q_tile_idx * 2 + peer_rank,
        head_idx,
        cta_rank
    };
    const auto unit_coord =
        tile_coord.template unit_coord<dim::DEPTH, 3>();
    const int4 tma_coord = ::kittens::tma::detail::tma_coords<
        cta2_fused_dense_do_exchange_tile,
        dim::DEPTH
    >(unit_coord);

    asm volatile(
        "{\n"
        ".reg .pred leader;\n"
        "elect.sync _|leader, -1;\n"
        "@leader mbarrier.arrive.expect_tx.shared::cta.b64 "
        "_, [%2], %7;\n"
        "@leader cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %3, %4, %5, %6}], [%2];\n"
        "}\n"
        :
        : "r"(do_smem),
          "l"(do_tma),
          "r"(bar_smem),
          "r"(tma_coord.x),
          "r"(tma_coord.y),
          "r"(tma_coord.z),
          "r"(tma_coord.w),
          "r"(static_cast<uint32_t>(sizeof(dst.dout)))
        : "memory"
    );
}

template <
    bool CacheLse,
    typename DkdvOutT,
    bool DirectTmaDkQ
>
__device__ __forceinline__ void cta2_fused_dense_load_stats(
    const cta2_fused_dense_globals_t<DkdvOutT, DirectTmaDkQ> &g,
    cta2_fused_dense_stats_storage<CacheLse> &dst,
    semaphore &barrier,
    int batch_idx,
    int q_tile_idx,
    int head_idx
) {
    const size_t offset =
        (static_cast<size_t>(batch_idx) * g.lse_log2.depth() + head_idx) *
            g.seq_len +
        q_tile_idx * 128;
    if constexpr (CacheLse) {
        tma::load_async(
            reinterpret_cast<void *>(&dst.lse_log2),
            reinterpret_cast<void *>(g.lse_log2.raw_ptr + offset),
            sizeof(dst.lse_log2),
            barrier
        );
    }
    tma::load_async(
        reinterpret_cast<void *>(&dst.dpsum),
        reinterpret_cast<void *>(g.dpsum.raw_ptr + offset),
        sizeof(dst.dpsum),
        barrier
    );
}

template <
    typename DkdvOutT,
    bool DirectTmaDkQ
>
__device__ __forceinline__ void cta2_fused_dense_load_dpsum(
    const cta2_fused_dense_globals_t<DkdvOutT, DirectTmaDkQ> &g,
    cta2_fused_dense_stats_storage<true> &dst,
    semaphore &barrier,
    int batch_idx,
    int q_tile_idx,
    int head_idx
) {
    const size_t offset =
        (static_cast<size_t>(batch_idx) * g.dpsum.depth() + head_idx) *
            g.seq_len +
        q_tile_idx * 128;
    tma::load_async(
        reinterpret_cast<void *>(&dst.dpsum),
        reinterpret_cast<void *>(g.dpsum.raw_ptr + offset),
        sizeof(dst.dpsum),
        barrier
    );
}

template <
    bool CacheLse,
    typename DkdvOutT,
    bool DirectTmaDkQ
>
__device__ __forceinline__ void cta2_fused_dense_prefetch_next_qdo(
    const cta2_fused_dense_globals_t<DkdvOutT, DirectTmaDkQ> &g,
    cta2_fused_dense_shared_storage<CacheLse> &storage,
    semaphore &barrier,
    int batch_idx,
    int q_tile_idx,
    int q_tile_count,
    int head_idx,
    int cta_rank
) {
    if (q_tile_idx + 1 >= q_tile_count || threadIdx.x != 0) {
        return;
    }
    auto &next_q = storage.phase.dq.operands.exchange.next_q;
    auto &next_dout =
        *reinterpret_cast<cta2_fused_dense_do_tile *>(&storage.ds);
    tma::expect_bytes(
        barrier,
        sizeof(next_q) + sizeof(next_dout) + sizeof(storage.stats)
    );
    coord<cta2_fused_dense_q_tile> next_q_tile_coord = {
        batch_idx,
        (q_tile_idx + 1) * 2 + cta_rank,
        head_idx,
        0
    };
    coord<cta2_fused_dense_do_tile> next_do_tile_coord = {
        batch_idx,
        (q_tile_idx + 1) * 2 + cta_rank,
        head_idx,
        0
    };
    tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
        next_q,
        g.q,
        next_q_tile_coord,
        barrier
    );
    tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
        next_dout,
        g.dout,
        next_do_tile_coord,
        barrier
    );
    cta2_fused_dense_load_stats<CacheLse>(
        g,
        storage.stats,
        barrier,
        batch_idx,
        q_tile_idx + 1,
        head_idx
    );
}

__device__ __forceinline__ void cta2_fused_dense_load_tmem_x32(
    uint32_t (&dst)[32],
    uint32_t src_addr
) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
          "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7]),
          "=r"(dst[8]), "=r"(dst[9]), "=r"(dst[10]), "=r"(dst[11]),
          "=r"(dst[12]), "=r"(dst[13]), "=r"(dst[14]), "=r"(dst[15]),
          "=r"(dst[16]), "=r"(dst[17]), "=r"(dst[18]), "=r"(dst[19]),
          "=r"(dst[20]), "=r"(dst[21]), "=r"(dst[22]), "=r"(dst[23]),
          "=r"(dst[24]), "=r"(dst[25]), "=r"(dst[26]), "=r"(dst[27]),
          "=r"(dst[28]), "=r"(dst[29]), "=r"(dst[30]), "=r"(dst[31])
        : "r"(src_addr)
    );
}

template <int Chunk, typename Globals>
__device__ __forceinline__ void
cta2_fused_dense_store_dq_chunk_cached_unscaled(
    const Globals &g,
    cta2_fused_dense_dq_stage &stage,
    uint32_t lane_row,
    const uint32_t (&values)[32],
    int batch_idx,
    int q_row_block,
    int head_idx,
    int col_half
) {
    static_assert(Chunk >= 0 && Chunk < 3);
    #pragma unroll
    for (int col = 0; col < 32; col += 4) {
        move<float4>::sts(
            lane_row ^ static_cast<uint32_t>(col * sizeof(float)),
            make_float4(
                __uint_as_float(values[col + 0]),
                __uint_as_float(values[col + 1]),
                __uint_as_float(values[col + 2]),
                __uint_as_float(values[col + 3])
            )
        );
    }
    __syncwarp();
    warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
        g.dq,
        stage,
        {
            batch_idx,
            q_row_block,
            head_idx,
            col_half * 3 + Chunk
        }
    );
}

__device__ __forceinline__ void cta2_role_split_store_tmem_x16(
    uint32_t dst_addr,
    const uint32_t (&src)[16]
) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
        "{%1, %2, %3, %4, %5, %6, %7, %8, "
        "%9, %10, %11, %12, %13, %14, %15, %16};\n"
        ::
          "r"(dst_addr),
          "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
          "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]),
          "r"(src[8]), "r"(src[9]), "r"(src[10]), "r"(src[11]),
          "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15])
        : "memory"
    );
}

__device__ __forceinline__ void cta2_role_split_load_stats_x32_stage(
    float (&dst)[32],
    const float *src
) {
    #pragma unroll
    for (int element = 0; element < 32; element += 4) {
        float4 values;
        move<float4>::ldg(
            values,
            reinterpret_cast<float4 *>(
                const_cast<float *>(src + element)
            )
        );
        dst[element + 0] = values.x;
        dst[element + 1] = values.y;
        dst[element + 2] = values.z;
        dst[element + 3] = values.w;
    }
}

__device__ __forceinline__ void cta2_role_split_load_stats_x32_stage_shared(
    float (&dst)[32],
    const float *src
) {
    const uint32_t src_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(src));
    #pragma unroll
    for (int element = 0; element < 32; element += 4) {
        float4 values;
        move<float4>::lds(
            values,
            src_ptr + sizeof(float) * element
        );
        dst[element + 0] = values.x;
        dst[element + 1] = values.y;
        dst[element + 2] = values.z;
        dst[element + 3] = values.w;
    }
}

__device__ __forceinline__ void cta2_role_split_load_p_tmem_exact(
    cta2_fused_dense_attn_bf_reg &dst,
    const cta2_fused_dense_ds_tt &src
) {
    const int group_warp = group<8>::warpid();
    const int physical_row =
        32 * (group_warp % 4) + 16 * (group_warp / 4);
    tt_bf<16, 128> src_subtile{
        src.addr + (static_cast<uint32_t>(physical_row) << 16)
    };
    #pragma unroll
    for (int j = 0; j < cta2_fused_dense_attn_bf_reg::width; j += 4) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x128b.x8.b32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, "
            "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
            : "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 0].data[0]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 0].data[1]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 0].data[2]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 0].data[3]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 1].data[0]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 1].data[1]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 1].data[2]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 1].data[3]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 2].data[0]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 2].data[1]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 2].data[2]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 2].data[3]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 3].data[0]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 3].data[1]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 3].data[2]
              )),
              "=r"(*reinterpret_cast<uint32_t *>(
                  &dst.tiles[0][j + 3].data[3]
              ))
            : "r"(src_subtile.addr + j * 8)
        );
    }
}

template <ducks::rt::all Dst, ducks::rt::all Src>
__device__ __forceinline__ void cta2_role_split_expand_bf16_bits(
    Dst &dst,
    const Src &src
) {
    static_assert(
        Dst::rows == Src::rows && Dst::cols == Src::cols
    );
    static_assert(
        std::is_same_v<typename Dst::dtype, float2> &&
        std::is_same_v<typename Src::dtype, bf16_2>
    );
    #pragma unroll
    for (int tile_row = 0; tile_row < src.height; ++tile_row) {
        #pragma unroll
        for (int tile_col = 0; tile_col < src.width; ++tile_col) {
            #pragma unroll
            for (int packed = 0; packed < src.packed_per_tile; ++packed) {
                const uint32_t word = *reinterpret_cast<const uint32_t *>(
                    &src.tiles[tile_row][tile_col].data[packed]
                );
                float2 value;
                value.x = __uint_as_float(word << 16);
                value.y = __uint_as_float(word & 0xffff0000u);
                dst.tiles[tile_row][tile_col].data[packed] = value;
            }
        }
    }
}

__device__ __forceinline__ uint32_t
cta2_fused_dense_map_cluster_semaphore(semaphore &barrier, int dst_cta) {
    const uint32_t local_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&barrier)
    );
    uint32_t mapped_address;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(mapped_address)
        : "r"(local_address), "r"(dst_cta)
    );
    return mapped_address;
}

__device__ __forceinline__ void cta2_fused_dense_cluster_arrive_mapped(
    uint32_t mapped_address,
    uint32_t count = 1
) {
    asm volatile(
        "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(mapped_address), "r"(count)
        : "memory"
    );
}

template <bool UseTmemDs, bool PrepareDoNormal = false>
__device__ __forceinline__ void cta2_fused_dense_prepare_q_operand(
    cta2_fused_dense_main_phase_storage &main,
    cta2_fused_dense_qdo_exchange_storage &qdo_exchange,
    int warp,
    int cta_rank
) {
    static_assert(!PrepareDoNormal || UseTmemDs);
    if constexpr (UseTmemDs) {
        rt_bf<16, 64> q0;
        rt_bf<16, 32> q1;
        rt_bf<16, 64> do_normal_reg;
        const int source_row = warp & 3;
        if ((warp >> 2) == cta_rank) {
            auto q0_source = main.q.q.template subtile<16, 64>(
                {source_row, cta_rank}
            );
            auto q1_source = main.q.q.template subtile<16, 32>(
                {source_row, 4 + cta_rank}
            );
            warp::load(q0, q0_source);
            warp::load(q1, q1_source);
            if constexpr (PrepareDoNormal) {
                auto do_source = main.dout.template subtile<16, 64>(
                    {source_row, cta_rank}
                );
                warp::load(do_normal_reg, do_source);
            }
        } else {
            auto q0_source = qdo_exchange.q0.template subtile<16, 64>(
                {source_row, 0}
            );
            auto q1_source = qdo_exchange.q1.template subtile<16, 32>(
                {source_row, cta_rank}
            );
            warp::load(q0, q0_source);
            warp::load(q1, q1_source);
            if constexpr (PrepareDoNormal) {
                auto do_source = qdo_exchange.dout.template subtile<16, 64>(
                    {source_row, 0}
                );
                warp::load(do_normal_reg, do_source);
            }
        }
        __syncthreads();
        auto q0_destination =
            main.q.normal.q0.template subtile<16, 64>({warp, 0});
        auto q1_destination =
            main.q.normal.q1.template subtile<16, 32>({warp, 0});
        warp::store(q0_destination, q0);
        warp::store(q1_destination, q1);
        if constexpr (PrepareDoNormal) {
            auto &do_normal =
                *reinterpret_cast<st_bf<128, 64> *>(&main.p.p);
            auto do_destination =
                do_normal.template subtile<16, 64>({warp, 0});
            warp::store(do_destination, do_normal_reg);
        }
    } else {
        rt_bf<16, 64> q0;
        rt_bf<64, 16> q0_transposed;
        rt_bf<16, 32> q1;
        rt_bf<32, 16> q1_transposed;
        const int source_row = warp & 3;
        if ((warp >> 2) == cta_rank) {
            auto q0_source = main.q.q.template subtile<16, 64>(
                {source_row, cta_rank}
            );
            auto q1_source = main.q.q.template subtile<16, 32>(
                {source_row, 4 + cta_rank}
            );
            warp::load(q0, q0_source);
            warp::load(q1, q1_source);
        } else {
            auto q0_source = qdo_exchange.q0.template subtile<16, 64>(
                {source_row, 0}
            );
            auto q1_source = qdo_exchange.q1.template subtile<16, 32>(
                {source_row, cta_rank}
            );
            warp::load(q0, q0_source);
            warp::load(q1, q1_source);
        }
        warp::transpose_sep(q0_transposed, q0);
        warp::transpose_sep(q1_transposed, q1);
        __syncthreads();
        auto q0_smem =
            main.q.transposed.q0.template subtile<64, 16>({0, warp});
        auto q1_smem =
            main.q.transposed.q1.template subtile<32, 16>({0, warp});
        warp::store(q0_smem, q0_transposed);
        warp::store(q1_smem, q1_transposed);
    }
    __syncthreads();
}

__device__ __forceinline__ void cta2_fused_dense_commit(semaphore &barrier) {
    const uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&barrier)
    );
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(address), "h"(uint16_t{0b11})
        : "memory"
    );
}

__device__ __forceinline__ void cta2_fused_dense_commit_mapped(
    uint32_t mapped_address
) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(mapped_address), "h"(uint16_t{0b11})
        : "memory"
    );
}

template <bool UseMappedAddress>
__device__ __forceinline__ void cta2_fused_dense_commit_selected(
    semaphore &barrier,
    uint32_t mapped_address
) {
    if constexpr (UseMappedAddress) {
        cta2_fused_dense_commit_mapped(mapped_address);
    } else {
        cta2_fused_dense_commit(barrier);
    }
}

__device__ __forceinline__ void cta2_fused_dense_wait_timeout(
    semaphore &barrier,
    int phase
) {
    const uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(&barrier)
    );
    asm volatile(
        "{\n\t"
        ".reg .pred ready;\n\t"
        "WAIT_TIMEOUT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1, 10000000;\n\t"
        "@ready bra.uni WAIT_DONE;\n\t"
        "bra.uni WAIT_TIMEOUT;\n\t"
        "WAIT_DONE:\n\t"
        "}\n"
        :: "r"(address), "r"(phase)
        : "memory"
    );
}

template <bool UseTimeout>
__device__ __forceinline__ void cta2_fused_dense_role_wait(
    semaphore &barrier,
    int phase
) {
    if constexpr (UseTimeout) {
        cta2_fused_dense_wait_timeout(barrier, phase);
    } else {
        wait(barrier, phase);
    }
}

template <int Accumulate = 0, ducks::tt::all D,
          ducks::st_descriptor::input A,
          ducks::st_descriptor::input B>
__device__ __forceinline__ void cta2_fused_dense_mm2_abt_no_fence(
    D &d,
    const A &a,
    const B &b
) {
    constexpr int kCtaGroup = 2;
    constexpr int kM = A::rows * kCtaGroup;
    constexpr int kN = B::rows * kCtaGroup;
    constexpr int kK = A::cols;
    static_assert(kK == B::cols && kK % 16 == 0);
    static_assert(kM == D::rows * kCtaGroup && kN == D::cols);

    using input_type = typename A::T;
    using output_type = typename D::T;
    static_assert(std::is_same_v<input_type, typename B::T>);
    const uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kM,
            kN,
            transpose::N,
            transpose::N,
            false
        >();
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<A>,
        transpose::N
    > a_desc(a);
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<B>,
        transpose::N
    > b_desc(b);

    ::kittens::detail::tcgen05::template st_st<
        input_type,
        Accumulate,
        kCtaGroup
    >(
        d.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        instruction
    );
    #pragma unroll
    for (int chunk = 1; chunk < kK / 16; ++chunk) {
        ::kittens::detail::tcgen05::template st_st<
            input_type,
            1,
            kCtaGroup
        >(
            d.addr,
            a_desc.chunk_descriptor(chunk),
            b_desc.chunk_descriptor(chunk),
            instruction
        );
    }
}

__device__ __forceinline__ void
cta2_fused_dense_score_mma_commit_compact(
    uint32_t d_addr,
    uint64_t a_base_desc,
    uint64_t b_base_desc,
    uint32_t commit_address
) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, zero_accumulate;\n\t"
        ".reg .b64 a_desc, b_desc;\n\t"
        "elect.sync _|leader, 0x1;\n\t"
        "mov.b64 a_desc, %1;\n\t"
        "mov.b64 b_desc, %2;\n\t"
        "setp.eq.u32 zero_accumulate, 0, 1;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, zero_accumulate;\n\t"
        "add.u64 a_desc, %1, 0x2;\n\t"
        "add.u64 b_desc, %2, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x4;\n\t"
        "add.u64 b_desc, %2, 0x4;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x6;\n\t"
        "add.u64 b_desc, %2, 0x6;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x400;\n\t"
        "add.u64 b_desc, %2, 0x200;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x402;\n\t"
        "add.u64 b_desc, %2, 0x202;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x404;\n\t"
        "add.u64 b_desc, %2, 0x204;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x406;\n\t"
        "add.u64 b_desc, %2, 0x206;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x800;\n\t"
        "add.u64 b_desc, %2, 0x400;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x802;\n\t"
        "add.u64 b_desc, %2, 0x402;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x804;\n\t"
        "add.u64 b_desc, %2, 0x404;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x806;\n\t"
        "add.u64 b_desc, %2, 0x406;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "@leader tcgen05.commit.cta_group::2."
            "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
            "[%3], %4;\n\t"
        "}\n"
        ::
        "r"(d_addr),
        "l"(a_base_desc),
        "l"(b_base_desc),
        "r"(commit_address),
        "h"(uint16_t{0b11})
        : "memory"
    );
}

__device__ __forceinline__ void
cta2_fused_dense_dp_mma_commit_compact(
    uint32_t d_addr,
    uint64_t a_base_desc,
    uint64_t b_base_desc,
    uint32_t commit_address
) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, zero_accumulate;\n\t"
        ".reg .b64 a_desc, b_desc;\n\t"
        "elect.sync _|leader, 0x1;\n\t"
        "mov.b64 a_desc, %1;\n\t"
        "mov.b64 b_desc, %2;\n\t"
        "setp.eq.u32 zero_accumulate, 0, 1;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, zero_accumulate;\n\t"
        "add.u64 a_desc, %1, 0x2;\n\t"
        "add.u64 b_desc, %2, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x4;\n\t"
        "add.u64 b_desc, %2, 0x4;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x6;\n\t"
        "add.u64 b_desc, %2, 0x6;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x400;\n\t"
        "add.u64 b_desc, %2, 0x200;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x402;\n\t"
        "add.u64 b_desc, %2, 0x202;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x404;\n\t"
        "add.u64 b_desc, %2, 0x204;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "add.u64 a_desc, %1, 0x406;\n\t"
        "add.u64 b_desc, %2, 0x206;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%0], a_desc, b_desc, 0x10200490, 1;\n\t"
        "@leader tcgen05.commit.cta_group::2."
            "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
            "[%3], %4;\n\t"
        "}\n"
        ::
        "r"(d_addr),
        "l"(a_base_desc),
        "l"(b_base_desc),
        "r"(commit_address),
        "h"(uint16_t{0b11})
        : "memory"
    );
}

__device__ __forceinline__ void
cta2_fused_dense_score_mma_commit_compact_rw(
    uint32_t d_addr,
    uint64_t a_base_desc,
    uint64_t b_base_desc,
    uint32_t commit_address
) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, zero_accumulate;\n\t"
        "elect.sync _|leader, 0x1;\n\t"
        "setp.eq.u32 zero_accumulate, 0, 1;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, zero_accumulate;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x3fa;\n\t"
        "add.u64 %1, %1, 0x1fa;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x3fa;\n\t"
        "add.u64 %1, %1, 0x1fa;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "add.u64 %0, %0, 0x2;\n\t"
        "add.u64 %1, %1, 0x2;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 "
            "[%2], %0, %1, 0x10200490, 1;\n\t"
        "@leader tcgen05.commit.cta_group::2."
            "mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
            "[%3], %4;\n\t"
        "}\n"
        : "+l"(a_base_desc),
          "+l"(b_base_desc)
        : "r"(d_addr),
          "r"(commit_address),
          "h"(uint16_t{0b11})
        : "memory"
    );
}

template <int Accumulate = 0, ducks::tt::all D,
          ducks::st_descriptor::input A,
          ducks::st_descriptor::input B>
__device__ __forceinline__ void cta2_fused_dense_mm2_ab_no_fence(
    D &d,
    const A &a,
    const B &b
) {
    constexpr int kCtaGroup = 2;
    constexpr int kM = A::rows * kCtaGroup;
    constexpr int kN = B::cols * kCtaGroup;
    constexpr int kK = A::cols;
    static_assert(kK == B::rows && kK % 16 == 0);
    static_assert(kM == D::rows * kCtaGroup && kN == D::cols);

    using input_type = typename A::T;
    using output_type = typename D::T;
    static_assert(std::is_same_v<input_type, typename B::T>);
    const uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kM,
            kN,
            transpose::N,
            transpose::T,
            false
        >();
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<A>,
        transpose::N
    > a_desc(a);
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<B>,
        transpose::T
    > b_desc(b);

    ::kittens::detail::tcgen05::template st_st<
        input_type,
        Accumulate,
        kCtaGroup
    >(
        d.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        instruction
    );
    #pragma unroll
    for (int chunk = 1; chunk < kK / 16; ++chunk) {
        ::kittens::detail::tcgen05::template st_st<
            input_type,
            1,
            kCtaGroup
        >(
            d.addr,
            a_desc.chunk_descriptor(chunk),
            b_desc.chunk_descriptor(chunk),
            instruction
        );
    }
}

template <ducks::tt::all D, ducks::st_descriptor::input A,
          ducks::st_descriptor::input B>
__device__ __forceinline__ void
cta2_fused_dense_mm2_ab_no_fence_runtime_accumulate(
    D &d,
    const A &a,
    const B &b,
    bool accumulate
) {
    constexpr int kCtaGroup = 2;
    constexpr int kM = A::rows * kCtaGroup;
    constexpr int kN = B::cols * kCtaGroup;
    constexpr int kK = A::cols;
    static_assert(kK == B::rows && kK % 16 == 0);
    static_assert(kM == D::rows * kCtaGroup && kN == D::cols);

    using input_type = typename A::T;
    using output_type = typename D::T;
    static_assert(std::is_same_v<input_type, typename B::T>);
    static_assert(std::is_same_v<input_type, bf16>);
    const uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kM,
            kN,
            transpose::N,
            transpose::T,
            false
        >();
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<A>,
        transpose::N
    > a_desc(a);
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<B>,
        transpose::T
    > b_desc(b);

    const uint64_t first_a = a_desc.chunk_descriptor(0);
    const uint64_t first_b = b_desc.chunk_descriptor(0);
    const uint32_t accumulate_value = accumulate ? 1u : 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred accumulate_pred;\n\t"
        "setp.ne.u32 accumulate_pred, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 "
        "[%0], %1, %2, %3, accumulate_pred;\n\t"
        "}\n"
        :: "r"(d.addr),
           "l"(first_a),
           "l"(first_b),
           "r"(instruction),
           "r"(accumulate_value)
        : "memory"
    );
    #pragma unroll
    for (int chunk = 1; chunk < kK / 16; ++chunk) {
        ::kittens::detail::tcgen05::template st_st<
            input_type,
            1,
            kCtaGroup
        >(
            d.addr,
            a_desc.chunk_descriptor(chunk),
            b_desc.chunk_descriptor(chunk),
            instruction
        );
    }
}

template <ducks::tt::all D, ducks::tt::all A,
          ducks::st_descriptor::input B>
__device__ __forceinline__ void
cta2_fused_dense_tmem_a_mm2_ab_no_fence_runtime_accumulate(
    D &d,
    const A &a,
    const B &b,
    bool accumulate
) {
    constexpr int kCtaGroup = 2;
    constexpr int kM = A::rows * kCtaGroup;
    constexpr int kN = B::cols * kCtaGroup;
    constexpr int kK = A::cols;
    static_assert(kK == B::rows && kK % 16 == 0);
    static_assert(kM == D::rows * kCtaGroup && kN == D::cols);

    using input_type = typename A::T;
    using output_type = typename D::T;
    static_assert(std::is_same_v<input_type, typename B::T>);
    static_assert(std::is_same_v<input_type, bf16>);
    const uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kM,
            kN,
            transpose::N,
            transpose::T,
            false
        >();
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<B>,
        transpose::T
    > b_desc(b);

    const uint32_t first_a = a.template chunk_addr<transpose::N>(0);
    const uint64_t first_b = b_desc.chunk_descriptor(0);
    const uint32_t accumulate_value = accumulate ? 1u : 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred accumulate_pred;\n\t"
        "setp.ne.u32 accumulate_pred, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 "
        "[%0], [%1], %2, %3, accumulate_pred;\n\t"
        "}\n"
        :: "r"(d.addr),
           "r"(first_a),
           "l"(first_b),
           "r"(instruction),
           "r"(accumulate_value)
        : "memory"
    );
    #pragma unroll
    for (int chunk = 1; chunk < kK / 16; ++chunk) {
        ::kittens::detail::tcgen05::template tt_st<
            input_type,
            1,
            kCtaGroup
        >(
            d.addr,
            a.template chunk_addr<transpose::N>(chunk),
            b_desc.chunk_descriptor(chunk),
            instruction
        );
    }
}

template <ducks::tt::all D, ducks::st_descriptor::input A,
          ducks::st_descriptor::input B>
__device__ __forceinline__ void cta2_fused_dense_mm2_atb_no_fence(
    D &d,
    const A &a,
    const B &b
) {
    constexpr int kCtaGroup = 2;
    constexpr int kM = A::cols * kCtaGroup;
    constexpr int kN = B::cols * kCtaGroup;
    constexpr int kK = A::rows;
    static_assert(kK == B::rows && kK % 16 == 0);
    static_assert(kM == D::rows * kCtaGroup && kN == D::cols);

    using input_type = typename A::T;
    using output_type = typename D::T;
    static_assert(std::is_same_v<input_type, typename B::T>);
    const uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kM,
            kN,
            transpose::T,
            transpose::T,
            false
        >();
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<A>,
        transpose::T
    > a_desc(a);
    ::kittens::st_descriptor<
        ducks::st_descriptor::detail::get_st<B>,
        transpose::T
    > b_desc(b);

    ::kittens::detail::tcgen05::template st_st<input_type, 0, kCtaGroup>(
        d.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        instruction
    );
    #pragma unroll
    for (int chunk = 1; chunk < kK / 16; ++chunk) {
        ::kittens::detail::tcgen05::template st_st<
            input_type,
            1,
            kCtaGroup
        >(
            d.addr,
            a_desc.chunk_descriptor(chunk),
            b_desc.chunk_descriptor(chunk),
            instruction
        );
    }
}

__device__ __forceinline__ void cta2_role_split_load_q_transposed(
    cta2_fused_dense_main_phase_storage &main,
    cta2_fused_dense_qdo_exchange_storage &exchange,
    int logical_warp,
    int cta_rank,
    rt_bf<64, 16> &q0_transposed,
    rt_bf<32, 16> &q1_transposed
) {
    rt_bf<16, 64> q0;
    rt_bf<16, 32> q1;
    const int source_row = logical_warp & 3;
    if ((logical_warp >> 2) == cta_rank) {
        auto q0_source = main.q.q.template subtile<16, 64>(
            {source_row, cta_rank}
        );
        auto q1_source = main.q.q.template subtile<16, 32>(
            {source_row, 4 + cta_rank}
        );
        warp::load(q0, q0_source);
        warp::load(q1, q1_source);
    } else {
        auto q0_source = exchange.q0.template subtile<16, 64>(
            {source_row, 0}
        );
        auto q1_source = exchange.q1.template subtile<16, 32>(
            {source_row, cta_rank}
        );
        warp::load(q0, q0_source);
        warp::load(q1, q1_source);
    }
    warp::transpose_sep(q0_transposed, q0);
    warp::transpose_sep(q1_transposed, q1);
}

template <int Rows>
__device__ __forceinline__ void cta2_role_split_store_q_transposed_wide(
    cta2_fused_dense_q_t_wide_tile &dst,
    const rt_bf<Rows, 16> &src,
    int row_tile_offset,
    int col_tile
) {
    static_assert(Rows % 16 == 0);
    #pragma unroll
    for (int row_tile = 0; row_tile < Rows / 16; ++row_tile) {
        rt_bf<16, 16> fragment;
        fragment.tiles[0][0] = src.tiles[row_tile][0];
        auto dst_fragment = dst.template subtile<16, 16>({
            row_tile_offset + row_tile,
            col_tile
        });
        warp::store(dst_fragment, fragment);
    }
}

__device__ __forceinline__ void cta2_role_split_load_q_transposed_wide(
    cta2_fused_dense_main_phase_storage &main,
    cta2_fused_dense_qdo_exchange_storage &exchange,
    int logical_warp,
    int cta_rank,
    rt_bf<32, 16> &q0_transposed,
    rt_bf<32, 16> &q1_transposed,
    rt_bf<32, 16> &q2_transposed
) {
    rt_bf<16, 32> q0;
    rt_bf<16, 32> q1;
    rt_bf<16, 32> q2;
    const int source_row = logical_warp & 3;
    if ((logical_warp >> 2) == cta_rank) {
        auto q0_source = main.q.q.template subtile<16, 32>({
            source_row,
            cta_rank * 3
        });
        auto q1_source = main.q.q.template subtile<16, 32>({
            source_row,
            cta_rank * 3 + 1
        });
        auto q2_source = main.q.q.template subtile<16, 32>({
            source_row,
            cta_rank * 3 + 2
        });
        warp::load(q0, q0_source);
        warp::load(q1, q1_source);
        warp::load(q2, q2_source);
    } else if (cta_rank == 0) {
        auto q0_source = exchange.q0.template subtile<16, 32>({
            source_row,
            0
        });
        auto q1_source = exchange.q0.template subtile<16, 32>({
            source_row,
            1
        });
        auto q2_source = exchange.q1.template subtile<16, 32>({
            source_row,
            0
        });
        warp::load(q0, q0_source);
        warp::load(q1, q1_source);
        warp::load(q2, q2_source);
    } else {
        auto q0_source = exchange.q0.template subtile<16, 32>({
            source_row,
            1
        });
        auto q1_source = exchange.q1.template subtile<16, 32>({
            source_row,
            0
        });
        auto q2_source = exchange.q1.template subtile<16, 32>({
            source_row,
            1
        });
        warp::load(q0, q0_source);
        warp::load(q1, q1_source);
        warp::load(q2, q2_source);
    }
    warp::transpose_sep(q0_transposed, q0);
    warp::transpose_sep(q1_transposed, q1);
    warp::transpose_sep(q2_transposed, q2);
}

template <
    bool CacheLse,
    bool UseTmemDs,
    bool OverlapDsExchange,
    bool OverlapQWithDp,
    bool UseTmemP = false,
    bool OverlapDoWithDp = false,
    bool UseDpOperandReadyMbar = false,
    bool UseDqOperandReadyMbar = false,
    bool OverlapDvWithDs = false,
    bool PipelineNextScore = false,
    bool PreloadDqA = false,
    bool UseScoreOperandReadyMbar = false,
    bool UseDsWarpMulticastMbar = false
>
__global__ __launch_bounds__(256, 1)
void main_kernel_causal_cta2_fused_dense(
    const __grid_constant__ cta2_fused_dense_globals g
) {
    static_assert(
        !OverlapDvWithDs || (UseTmemP && OverlapDoWithDp),
        "early dV issue requires TMEM P and the normalized dO operand"
    );
    static_assert(
        !PipelineNextScore || UseDqOperandReadyMbar,
        "score lookahead requires the validated dQ-ready route"
    );
    static_assert(
        !PreloadDqA || OverlapDsExchange,
        "dQ-A preload requires the peer dS exchange"
    );
    static_assert(
        !UseDsWarpMulticastMbar || (UseTmemDs && OverlapDsExchange),
        "multicast dS readiness requires TMEM dS and the peer exchange"
    );
    __shared__ alignas(1024) cta2_fused_dense_shared_storage<CacheLse> storage;
    __shared__ alignas(16) semaphore score_done;
    __shared__ alignas(16) semaphore dp_done;
    __shared__ alignas(16) semaphore dkdv_done;
    __shared__ alignas(16) semaphore kv_load_done;
    __shared__ alignas(16) semaphore qdo_load_done;
    __shared__ alignas(16) semaphore qdo_prefetch_done;
    __shared__ alignas(16) semaphore qdo_exchange_done;
    __shared__ alignas(16) semaphore dq_exchange_done;
    __shared__ alignas(16) semaphore k_exchange_done;
    __shared__ alignas(16) semaphore dq_done;
    __shared__ alignas(16) semaphore dp_operands_ready;
    __shared__ alignas(16) semaphore dq_operands_ready;
    __shared__ alignas(16) semaphore score_lookahead_ready;
    __shared__ alignas(16) semaphore score_operands_ready;
    __shared__ alignas(16) semaphore ds_warp_multicast_ready;

    const int warp = warpid();
    const int cta_rank = cluster_ctarank();
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int owner_pair_idx = static_cast<int>(clusterIdx().x);
    const int q_tile_count = g.seq_len / 128;
    const int owner_count = q_tile_count / 2;
    const int output_subtile = 2 * (warp % 4) + warp / 4;

    if (threadIdx.x == 0) {
        g.k.template prefetch_tma<cta2_fused_dense_k_tile, dim::DEPTH>();
        g.q.template prefetch_tma<cta2_fused_dense_q_tile, dim::DEPTH>();
        g.q.template prefetch_tma<
            cta2_fused_dense_q0_exchange_tile,
            dim::DEPTH
        >();
        g.v.template prefetch_tma<cta2_fused_dense_v_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<cta2_fused_dense_do_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<
            cta2_fused_dense_do_exchange_tile,
            dim::DEPTH
        >();
        g.dq.template prefetch_tma<cta2_fused_dense_dq_stage, dim::DEPTH>();
        init_semaphore(score_done, 0, 1);
        init_semaphore(dp_done, 0, 1);
        init_semaphore(dkdv_done, 0, 1);
        init_semaphore(kv_load_done, 0, 1);
        init_semaphore(qdo_load_done, 0, 1);
        init_semaphore(qdo_prefetch_done, 0, 1);
        init_semaphore(qdo_exchange_done, 0, 1);
        init_semaphore(dq_exchange_done, 0, 1);
        init_semaphore(k_exchange_done, 0, 1);
        init_semaphore(dq_done, 0, 1);
        if constexpr (UseDpOperandReadyMbar) {
            init_semaphore(dp_operands_ready, 0, 2);
        }
        if constexpr (UseDqOperandReadyMbar) {
            init_semaphore(dq_operands_ready, 0, 2);
        }
        if constexpr (PipelineNextScore) {
            init_semaphore(score_lookahead_ready, 0, 10);
        }
        if constexpr (UseScoreOperandReadyMbar) {
            init_semaphore(score_operands_ready, 0, 2);
        }
        if constexpr (UseDsWarpMulticastMbar) {
            init_semaphore(ds_warp_multicast_ready, 0, 16);
        }
    }
    __syncthreads();
    everyone::tma::cluster::sync();

    tensor_allocator<1, 2> tm_alloc{};
    cta2_fused_dense_attn_tt dk_main_tmem =
        tm_alloc.template allocate<cta2_fused_dense_attn_tt>(0);
    cta2_fused_dense_dk_tail_tt dk_tail_tmem =
        tm_alloc.template allocate<cta2_fused_dense_dk_tail_tt>(128);
    cta2_fused_dense_attn_tt dv_tmem =
        tm_alloc.template allocate<cta2_fused_dense_attn_tt>(192);
    cta2_fused_dense_attn_tt score_dp_tmem =
        tm_alloc.template allocate<cta2_fused_dense_attn_tt>(320);
    cta2_fused_dense_ds_tt ds_tmem =
        tm_alloc.template allocate<cta2_fused_dense_ds_tt>(320);
    cta2_fused_dense_ds_tt p_tmem =
        tm_alloc.template allocate<cta2_fused_dense_ds_tt>(448);
    cta2_fused_dense_dq_tt dq_tmem =
        tm_alloc.template allocate<cta2_fused_dense_dq_tt>(0, 320);
    int score_lookahead_count = 0;

    #pragma unroll
    for (int owner_pass = 0; owner_pass < 2; ++owner_pass) {
    const int owner_idx = owner_pass == 0
        ? owner_pair_idx
        : owner_count - 1 - owner_pair_idx;
    const int owner_phase = owner_pass & 1;
    const int first_dense_q_tile = 2 * owner_idx + 1;
    rt_bf<16, kB300VDim> v_persistent;

    bool first_accumulation = true;
    int iteration = 0;
    if (threadIdx.x == 0) {
        tma::expect_bytes(
            kv_load_done,
            sizeof(storage.k) + sizeof(storage.ds)
        );
        coord<cta2_fused_dense_k_tile> k_tile_idx = {
            batch_idx,
            owner_idx * 2 + cta_rank,
            head_idx,
            0
        };
        coord<cta2_fused_dense_v_tile> v_tile_idx = {
            batch_idx,
            owner_idx * 2 + cta_rank,
            head_idx,
            0
        };
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k,
            g.k,
            k_tile_idx,
            kv_load_done
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.ds,
            g.v,
            v_tile_idx,
            kv_load_done
        );
    }
    wait(kv_load_done, owner_phase);
    __syncthreads();
    {
        auto v_source = storage.ds.template subtile<16, kB300VDim>(
            {warp, 0}
        );
        warp::load(v_persistent, v_source);
    }
    __syncthreads();
    everyone::tma::cluster::sync();

    {
        rt_bf<16, 96> k_local;
        rt_bf<16, 96> k_peer;
        auto k_local_smem =
            storage.k.template subtile<16, 96>({warp, cta_rank});
        auto k_peer_smem =
            storage.k.template subtile<16, 96>({warp, cta_rank ^ 1});
        warp::load(k_local, k_local_smem);
        warp::load(k_peer, k_peer_smem);
        auto b_local_smem =
            storage.dq_b.template subtile<16, 96>(
                {cta_rank * 8 + warp, 0}
            );
        auto k_exchange_smem =
            storage.phase.dq.operands.exchange.k.template subtile<16, 96>(
                {warp, 0}
            );
        warp::store(b_local_smem, k_local);
        warp::store(k_exchange_smem, k_peer);
    }
    __syncthreads();
    everyone::tma::cluster::sync();

    if (threadIdx.x == 0) {
        const int peer_rank = cta_rank ^ 1;
        constexpr int kChunkElements = 128 * 32;
        ::kittens::tma::cluster::expect_bytes(
            k_exchange_done,
            sizeof(storage.phase.dq.operands.exchange.k),
            peer_rank
        );
        #pragma unroll
        for (int chunk = 0; chunk < 3; ++chunk) {
            void *destination = reinterpret_cast<void *>(
                &storage.dq_b.data[
                    chunk * 2 * kChunkElements +
                    cta_rank * kChunkElements
                ]
            );
            void *source = reinterpret_cast<void *>(
                &storage.phase.dq.operands.exchange.k.data[
                    chunk * kChunkElements
                ]
            );
            ::kittens::tma::cluster::store_async(
                destination,
                source,
                kChunkElements * sizeof(bf16),
                peer_rank,
                k_exchange_done
            );
        }
    }
    ::kittens::tma::cluster::wait(k_exchange_done, owner_phase);
    __syncthreads();
    everyone::tma::cluster::sync();

    for (
        int q_tile_idx = first_dense_q_tile;
        q_tile_idx < q_tile_count;
        ++q_tile_idx, ++iteration
    ) {
        auto &main = storage.phase.main;
        auto &qdo_exchange = storage.qdo_phase.qdo;
        const int local_phase = iteration & 1;
        const int phase = (iteration + owner_phase) & 1;

        if (iteration == 0) {
            if (threadIdx.x == 0) {
                tma::expect_bytes(
                    qdo_load_done,
                    sizeof(main.q.q) + sizeof(main.dout) +
                        sizeof(storage.stats)
                );
                coord<cta2_fused_dense_q_tile> q_tile_coord = {
                    batch_idx,
                    q_tile_idx * 2 + cta_rank,
                    head_idx,
                    0
                };
                coord<cta2_fused_dense_do_tile> do_tile_coord = {
                    batch_idx,
                    q_tile_idx * 2 + cta_rank,
                    head_idx,
                    0
                };
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    main.q.q,
                    g.q,
                    q_tile_coord,
                    qdo_load_done
                );
                tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                    main.dout,
                    g.dout,
                    do_tile_coord,
                    qdo_load_done
                );
                cta2_fused_dense_load_stats<CacheLse>(
                    g,
                    storage.stats,
                    qdo_load_done,
                    batch_idx,
                    q_tile_idx,
                    head_idx
                );
                cta2_fused_dense_load_peer_qdo(
                    g,
                    qdo_exchange,
                    qdo_exchange_done,
                    batch_idx,
                    q_tile_idx,
                    head_idx,
                    cta_rank
                );
            }
            wait(qdo_load_done, owner_phase);
        } else {
            wait(qdo_prefetch_done, local_phase ^ 1);
            if (warp < 4) {
                auto &next_q =
                    storage.phase.dq.operands.exchange.next_q;
                rt_bf<16, kB300QKDim> q_reg;
                auto q_source = next_q.template subtile<16, kB300QKDim>(
                    {warp, 0}
                );
                auto q_destination =
                    main.q.q.template subtile<16, kB300QKDim>({warp, 0});
                warp::load(q_reg, q_source);
                warp::store(q_destination, q_reg);

                rt_bf<16, kB300VDim> do_reg;
                auto &next_dout =
                    *reinterpret_cast<cta2_fused_dense_do_tile *>(
                        &storage.ds
                    );
                auto do_source =
                    next_dout.template subtile<16, kB300VDim>({warp, 0});
                auto do_destination =
                    main.dout.template subtile<16, kB300VDim>({warp, 0});
                warp::load(do_reg, do_source);
                warp::store(do_destination, do_reg);
            }
            if (threadIdx.x == 0) {
                cta2_fused_dense_load_peer_qdo(
                    g,
                    qdo_exchange,
                    qdo_exchange_done,
                    batch_idx,
                    q_tile_idx,
                    head_idx,
                    cta_rank
                );
            }
        }
        __syncthreads();
        if constexpr (UseScoreOperandReadyMbar) {
            if ((!PipelineNextScore || iteration == 0) && threadIdx.x == 0) {
                ::kittens::tma::cluster::arrive(score_operands_ready, 0);
            }
        } else {
            everyone::tma::cluster::sync();
        }

        if constexpr (PipelineNextScore) {
            if (iteration == 0 && cta_rank == 0) {
                if constexpr (UseScoreOperandReadyMbar) {
                    if (threadIdx.x == 0) {
                        wait(score_operands_ready, owner_phase);
                    }
                }
                group<8>::mm2_ABt(score_dp_tmem, storage.k, main.q.q);
                if (threadIdx.x == 0) {
                    cta2_fused_dense_commit(score_done);
                }
            }
        } else {
            if (cta_rank == 0) {
                if constexpr (UseScoreOperandReadyMbar) {
                    if (threadIdx.x == 0) {
                        wait(score_operands_ready, phase);
                    }
                }
                group<8>::mm2_ABt(score_dp_tmem, storage.k, main.q.q);
                if (threadIdx.x == 0) {
                    cta2_fused_dense_commit(score_done);
                }
            }
        }
        wait(score_done, phase);

        cta2_fused_dense_attn_bf_reg p_bf;
        {
            cta2_fused_dense_attn_reg p_reg;
            cta2_fused_dense_stats_vec lse_vec;
            group<8>::load_async(p_reg, score_dp_tmem);
            if constexpr (CacheLse) {
                warp::load(lse_vec, storage.stats.lse_log2);
            } else {
                warp::load(
                    lse_vec,
                    g.lse_log2,
                    {batch_idx, head_idx, 0, q_tile_idx}
                );
            }
            tensor_load_wait();
            warp::mul(p_reg, p_reg, g.scale_log2e);
            warp::sub_col(p_reg, p_reg, lse_vec);
            warp::exp2(p_reg, p_reg);
            if (q_tile_idx == first_dense_q_tile && cta_rank == 1) {
                // The frontier owns rank 1's diagonal half. Rank 0 alone owns
                // this K128-by-Q128 cross-half tile.
                warp::zero(p_reg);
            }
            warp::copy(p_bf, p_reg);
            if constexpr (UseTmemP) {
                group<8>::store_async(p_tmem, p_bf);
                tensor_store_wait();
            } else {
                auto p_smem =
                    main.p.p.template subtile<16, 128>({output_subtile, 0});
                warp::store(p_smem, p_bf);
            }
            auto v_smem = storage.ds.template subtile<16, kB300VDim>(
                {warp, 0}
            );
            warp::store(v_smem, v_persistent);
        }
        __syncthreads();
        if constexpr (UseDpOperandReadyMbar) {
            if (threadIdx.x == 0) {
                ::kittens::tma::cluster::arrive(dp_operands_ready, 0);
            }
        } else {
            everyone::tma::cluster::sync();
        }

        if constexpr (UseDpOperandReadyMbar) {
            if (cta_rank == 0 && threadIdx.x == 0) {
                wait(dp_operands_ready, phase);
                group<8>::mm2_ABt(score_dp_tmem, storage.ds, main.dout);
                cta2_fused_dense_commit(dp_done);
            }
        } else if (cta_rank == 0) {
            group<8>::mm2_ABt(score_dp_tmem, storage.ds, main.dout);
            if (threadIdx.x == 0) {
                cta2_fused_dense_commit(dp_done);
            }
        }
        if constexpr (OverlapQWithDp) {
            wait(qdo_exchange_done, phase);
            cta2_fused_dense_prepare_q_operand<
                UseTmemDs,
                UseTmemP && OverlapDoWithDp
            >(
                main,
                qdo_exchange,
                warp,
                cta_rank
            );
        }
        wait(dp_done, phase);

        if constexpr (OverlapDvWithDs) {
            if (cta_rank == 0) {
                auto &do_normal =
                    *reinterpret_cast<st_bf<128, 64> *>(&main.p.p);
                if (first_accumulation) {
                    group<8>::mm2_AB(dv_tmem, p_tmem, do_normal);
                } else {
                    group<8>::mma2_AB(dv_tmem, p_tmem, do_normal);
                }
            }
        }

        {
            cta2_fused_dense_attn_reg dp_reg;
            cta2_fused_dense_stats_vec dpsum_vec;
            group<8>::load_async(dp_reg, score_dp_tmem);
            warp::load(dpsum_vec, storage.stats.dpsum);
            tensor_load_wait();
            warp::sub_col(dp_reg, dp_reg, dpsum_vec);
            cta2_fused_dense_attn_reg ds_reg;
            warp::copy(ds_reg, p_bf);
            warp::mul(ds_reg, ds_reg, dp_reg);
            warp::mul(ds_reg, ds_reg, g.scale);
            cta2_fused_dense_attn_bf_reg ds_bf;
            warp::copy(ds_bf, ds_reg);
            if constexpr (UseTmemDs) {
                group<8>::store_async(ds_tmem, ds_bf);
                tensor_store_wait();
            }
            auto ds_smem = storage.ds.template subtile<16, 128>(
                {output_subtile, 0}
            );
            warp::store(ds_smem, ds_bf);
        }
        if constexpr (UseDsWarpMulticastMbar) {
            __syncwarp();
            if (laneid() == 0) {
                ::kittens::tma::cluster::arrive(
                    ds_warp_multicast_ready,
                    0
                );
                ::kittens::tma::cluster::arrive(
                    ds_warp_multicast_ready,
                    1
                );
            }
            ::kittens::tma::cluster::wait(
                ds_warp_multicast_ready,
                phase
            );
        } else {
            __syncthreads();
            everyone::tma::cluster::sync();
        }

        if constexpr (!OverlapQWithDp) {
            wait(qdo_exchange_done, phase);
            cta2_fused_dense_prepare_q_operand<
                UseTmemDs,
                UseTmemP && OverlapDoWithDp
            >(
                main,
                qdo_exchange,
                warp,
                cta_rank
            );
        }
        if constexpr (UseTmemP && !OverlapDoWithDp) {
            rt_bf<16, 64> do_half;
            const int source_row = warp & 3;
            if ((warp >> 2) == cta_rank) {
                auto do_source = main.dout.template subtile<16, 64>(
                    {source_row, cta_rank}
                );
                warp::load(do_half, do_source);
            } else {
                auto do_source = qdo_exchange.dout.template subtile<16, 64>(
                    {source_row, 0}
                );
                warp::load(do_half, do_source);
            }
            __syncthreads();
            auto &do_normal =
                *reinterpret_cast<st_bf<128, 64> *>(&main.dout);
            auto do_destination =
                do_normal.template subtile<16, 64>({warp, 0});
            warp::store(do_destination, do_half);
        } else if constexpr (!UseTmemP) {
            rt_bf<16, 64> do_half;
            rt_bf<64, 16> do_transposed;
            const int source_row = warp & 3;
            if ((warp >> 2) == cta_rank) {
                auto do_source = main.dout.template subtile<16, 64>(
                    {source_row, cta_rank}
                );
                warp::load(do_half, do_source);
            } else {
                auto do_source = qdo_exchange.dout.template subtile<16, 64>(
                    {source_row, 0}
                );
                warp::load(do_half, do_source);
            }
            warp::transpose_sep(do_transposed, do_half);
            __syncthreads();
            auto do_smem = main.dout.template subtile<64, 16>({0, warp});
            warp::store(do_smem, do_transposed);
        }
        if constexpr (!UseTmemP || !OverlapDoWithDp) {
            __syncthreads();
            everyone::tma::cluster::sync();
        }

        if constexpr (OverlapDsExchange) {
            if (threadIdx.x == 0) {
                const int peer_rank = cta_rank ^ 1;
                constexpr int kDsHalfElements = 128 * 64;
                ::kittens::tma::cluster::expect_bytes(
                    dq_exchange_done,
                    sizeof(storage.qdo_phase.ds),
                    peer_rank
                );
                ::kittens::tma::cluster::store_async(
                    reinterpret_cast<void *>(&storage.qdo_phase.ds),
                    reinterpret_cast<void *>(
                        &storage.ds.data[peer_rank * kDsHalfElements]
                    ),
                    sizeof(storage.qdo_phase.ds),
                    peer_rank,
                    dq_exchange_done
                );
            }
        }

        if (cta_rank == 0) {
            if (first_accumulation) {
                if constexpr (!OverlapDvWithDs) {
                    if constexpr (UseTmemP) {
                        auto &do_normal = OverlapDoWithDp
                            ? *reinterpret_cast<st_bf<128, 64> *>(&main.p.p)
                            : *reinterpret_cast<st_bf<128, 64> *>(&main.dout);
                        group<8>::mm2_AB(dv_tmem, p_tmem, do_normal);
                    } else {
                        group<8>::mm2_ABt(dv_tmem, main.p.p, main.dout);
                    }
                }
                if constexpr (UseTmemDs) {
                    group<8>::mm2_AB(
                        dk_main_tmem,
                        ds_tmem,
                        main.q.normal.q0
                    );
                    group<8>::mm2_AB(
                        dk_tail_tmem,
                        ds_tmem,
                        main.q.normal.q1
                    );
                } else {
                    group<8>::mm2_ABt(
                        dk_main_tmem,
                        storage.ds,
                        main.q.transposed.q0
                    );
                    group<8>::mm2_ABt(
                        dk_tail_tmem,
                        storage.ds,
                        main.q.transposed.q1
                    );
                }
            } else {
                if constexpr (!OverlapDvWithDs) {
                    if constexpr (UseTmemP) {
                        auto &do_normal = OverlapDoWithDp
                            ? *reinterpret_cast<st_bf<128, 64> *>(&main.p.p)
                            : *reinterpret_cast<st_bf<128, 64> *>(&main.dout);
                        group<8>::mma2_AB(dv_tmem, p_tmem, do_normal);
                    } else {
                        group<8>::mma2_ABt(dv_tmem, main.p.p, main.dout);
                    }
                }
                if constexpr (UseTmemDs) {
                    group<8>::mma2_AB(
                        dk_main_tmem,
                        ds_tmem,
                        main.q.normal.q0
                    );
                    group<8>::mma2_AB(
                        dk_tail_tmem,
                        ds_tmem,
                        main.q.normal.q1
                    );
                } else {
                    group<8>::mma2_ABt(
                        dk_main_tmem,
                        storage.ds,
                        main.q.transposed.q0
                    );
                    group<8>::mma2_ABt(
                        dk_tail_tmem,
                        storage.ds,
                        main.q.transposed.q1
                    );
                }
            }
            if (threadIdx.x == 0) {
                cta2_fused_dense_commit(dkdv_done);
            }
        }

        rt_bf<16, 64> dq_a_local_preload;
        rt_bf<16, 64> dq_a_peer_preload;
        if constexpr (PreloadDqA) {
            auto ds_local_smem =
                storage.ds.template subtile<16, 64>({warp, cta_rank});
            warp::load(dq_a_local_preload, ds_local_smem);
            ::kittens::tma::cluster::wait(dq_exchange_done, phase);
            auto ds_peer_smem =
                storage.qdo_phase.ds.template subtile<16, 64>({warp, 0});
            warp::load(dq_a_peer_preload, ds_peer_smem);
        }
        wait(dkdv_done, phase);

        auto &dq_operands = storage.phase.dq.operands;
        if constexpr (PreloadDqA) {
            auto a_local_smem =
                dq_operands.a.template subtile<16, 64>(
                    {cta_rank * 8 + warp, 0}
                );
            warp::store(a_local_smem, dq_a_local_preload);
            const int peer_rank = cta_rank ^ 1;
            auto a_peer_smem =
                dq_operands.a.template subtile<16, 64>(
                    {peer_rank * 8 + warp, 0}
                );
            warp::store(a_peer_smem, dq_a_peer_preload);
        } else {
            rt_bf<16, 64> ds_local;
            auto ds_local_smem =
                storage.ds.template subtile<16, 64>({warp, cta_rank});
            warp::load(ds_local, ds_local_smem);
            auto a_local_smem =
                dq_operands.a.template subtile<16, 64>(
                    {cta_rank * 8 + warp, 0}
                );
            warp::store(a_local_smem, ds_local);
        }

        if constexpr (!PreloadDqA) {
            if constexpr (OverlapDsExchange) {
                ::kittens::tma::cluster::wait(dq_exchange_done, phase);
                rt_bf<16, 64> ds_peer;
                auto ds_peer_smem =
                    storage.qdo_phase.ds.template subtile<16, 64>({warp, 0});
                warp::load(ds_peer, ds_peer_smem);
                const int peer_rank = cta_rank ^ 1;
                auto a_peer_smem =
                    dq_operands.a.template subtile<16, 64>(
                        {peer_rank * 8 + warp, 0}
                    );
                warp::store(a_peer_smem, ds_peer);
            } else {
                if (threadIdx.x == 0) {
                    const int peer_rank = cta_rank ^ 1;
                    constexpr int kDsHalfElements = 128 * 64;
                    void *destination = reinterpret_cast<void *>(
                        &dq_operands.a.data[cta_rank * 128 * 64]
                    );
                    ::kittens::tma::cluster::expect_bytes(
                        dq_exchange_done,
                        sizeof(dq_operands.exchange.ds),
                        peer_rank
                    );
                    ::kittens::tma::cluster::store_async(
                        destination,
                        reinterpret_cast<void *>(
                            &storage.ds.data[peer_rank * kDsHalfElements]
                        ),
                        sizeof(dq_operands.exchange.ds),
                        peer_rank,
                        dq_exchange_done
                    );
                }
                ::kittens::tma::cluster::wait(dq_exchange_done, phase);
            }
        }
        __syncthreads();
        if constexpr (UseDqOperandReadyMbar) {
            if (threadIdx.x == 0) {
                ::kittens::tma::cluster::arrive(dq_operands_ready, 0);
            }
        } else {
            everyone::tma::cluster::sync();
        }

        if constexpr (UseDqOperandReadyMbar) {
            if (cta_rank == 0 && threadIdx.x == 0) {
                wait(dq_operands_ready, phase);
                group<8>::mm2_AtB(
                    dq_tmem,
                    dq_operands.a,
                    storage.dq_b
                );
                cta2_fused_dense_commit(dq_done);
            }
        } else if (cta_rank == 0) {
            group<8>::mm2_AtB(dq_tmem, dq_operands.a, storage.dq_b);
            if (threadIdx.x == 0) {
                cta2_fused_dense_commit(dq_done);
            }
        }
        wait(dq_done, phase);

        cta2_fused_dense_prefetch_next_qdo<CacheLse>(
            g,
            storage,
            qdo_prefetch_done,
            batch_idx,
            q_tile_idx,
            q_tile_count,
            head_idx,
            cta_rank
        );

        const bool has_next_q_tile = q_tile_idx + 1 < q_tile_count;
        if constexpr (PipelineNextScore) {
            if (has_next_q_tile && threadIdx.x == 128) {
                wait(qdo_prefetch_done, local_phase);
                ::kittens::tma::cluster::arrive(score_lookahead_ready, 0);
            }
        }

        if (warp < 4) {
            auto &dq_reduce = storage.phase.dq.reduce;
            const int row_half = warp & 1;
            const int col_half = warp >> 1;
            const int q_row_block =
                q_tile_idx * 4 + cta_rank * 2 + row_half;
            const uint32_t physical_row = static_cast<uint32_t>(warp * 32);
            #pragma unroll 1
            for (int chunk = 0; chunk < 3; ++chunk) {
                uint32_t values[32];
                cta2_fused_dense_load_tmem_x32(
                    values,
                    dq_tmem.addr + (physical_row << 16) + chunk * 32
                );
                if constexpr (PipelineNextScore) {
                    if (
                        has_next_q_tile && chunk == 2 &&
                        (threadIdx.x & 31) == 0
                    ) {
                        ::kittens::tma::cluster::arrive(
                            score_lookahead_ready,
                            0
                        );
                    }
                }
                if (chunk >= 2) {
                    warp::tma::store_async_wait<1>();
                }
                auto &stage = dq_reduce.stage[chunk & 1][warp];
                #pragma unroll
                for (int col = 0; col < 32; col += 4) {
                    *reinterpret_cast<float4 *>(&stage[{laneid(), col}]) =
                        make_float4(
                            __uint_as_float(values[col + 0]),
                            __uint_as_float(values[col + 1]),
                            __uint_as_float(values[col + 2]),
                            __uint_as_float(values[col + 3])
                        );
                }
                __syncwarp();
                warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(
                    g.dq,
                    stage,
                    {
                        batch_idx,
                        q_row_block,
                        head_idx,
                        col_half * 3 + chunk
                    }
                );
            }
            warp::tma::store_async_wait();
        }
        if constexpr (PipelineNextScore) {
            if (
                has_next_q_tile && cta_rank == 0 && threadIdx.x == 128
            ) {
                wait(score_lookahead_ready, score_lookahead_count & 1);
                auto &next_q = storage.phase.dq.operands.exchange.next_q;
                ::kittens::mm2<transpose::N, transpose::T>(
                    score_dp_tmem,
                    storage.k,
                    next_q
                );
                cta2_fused_dense_commit(score_done);
            }
            if (has_next_q_tile) {
                ++score_lookahead_count;
            }
        }
        __syncthreads();
        everyone::tma::cluster::sync();
        first_accumulation = false;
    }

    const int kv_depth_tile = owner_idx * 16 + cta_rank * 8 + output_subtile;
    if (first_dense_q_tile < q_tile_count) {
        cta2_fused_dense_attn_reg dk_main;
        cta2_fused_dense_dk_tail_reg dk_tail;
        cta2_fused_dense_attn_reg dv_reg;
        group<8>::load_async(dk_main, dk_main_tmem);
        group<8>::load_async(dk_tail, dk_tail_tmem);
        group<8>::load_async(dv_reg, dv_tmem);
        tensor_load_wait();
        warp::store<dim::DEPTH>(
            g.dk,
            dk_main,
            {batch_idx, kv_depth_tile, head_idx, 0}
        );
        warp::store<dim::DEPTH>(
            g.dk,
            dk_tail,
            {batch_idx, kv_depth_tile, head_idx, 2}
        );
        warp::store<dim::DEPTH>(
            g.dv,
            dv_reg,
            {batch_idx, kv_depth_tile, head_idx, 0}
        );
    } else {
        cta2_fused_dense_attn_reg dk_main;
        cta2_fused_dense_dk_tail_reg dk_tail;
        cta2_fused_dense_attn_reg dv_reg;
        warp::zero(dk_main);
        warp::zero(dk_tail);
        warp::zero(dv_reg);
        warp::store<dim::DEPTH>(
            g.dk,
            dk_main,
            {batch_idx, kv_depth_tile, head_idx, 0}
        );
        warp::store<dim::DEPTH>(
            g.dk,
            dk_tail,
            {batch_idx, kv_depth_tile, head_idx, 2}
        );
        warp::store<dim::DEPTH>(
            g.dv,
            dv_reg,
            {batch_idx, kv_depth_tile, head_idx, 0}
        );
    }
    __syncthreads();
    everyone::tma::cluster::sync();
    }
}

__device__ __forceinline__ void cta2_role_split_select_ds_half(
    rt_bf<16, 64> &dst,
    const cta2_fused_dense_attn_bf_reg &src,
    int half
) {
    if (half == 0) {
        #pragma unroll
        for (int col = 0; col < dst.width; ++col) {
            dst.tiles[0][col] = src.tiles[0][col];
        }
    } else {
        #pragma unroll
        for (int col = 0; col < dst.width; ++col) {
            dst.tiles[0][col] = src.tiles[0][dst.width + col];
        }
    }
}

template <ducks::st::all ST>
__device__ __forceinline__ void cta2_role_split_store_ds_x32_stage(
    ST &dst,
    const uint32_t (&src)[16],
    int row_base,
    int col_base
) {
    static_assert(ST::cols == 64);
    static_assert(std::is_same_v<typename ST::dtype, bf16>);
    const int row = row_base + laneid();
    const uint32_t shared_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&dst.data[0])
    );
    #pragma unroll
    for (int packed = 0; packed < 16; packed += 4) {
        const uint32_t address = dst.idx(
            shared_addr,
            {row, col_base + 2 * packed}
        );
        asm volatile(
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
            ::
              "r"(address),
              "r"(src[packed + 0]),
              "r"(src[packed + 1]),
              "r"(src[packed + 2]),
              "r"(src[packed + 3])
            : "memory"
        );
    }
}

template <int Half, ducks::st::all ST>
__device__ __forceinline__ void cta2_role_split_store_ds_half_direct(
    ST &dst,
    const cta2_fused_dense_attn_bf_reg &src
) {
    static_assert(Half == 0 || Half == 1);
    static_assert(ST::cols == 64);
    using U = typename ST::dtype;
    using U2 = base_types::packing<U>::packed_type;
    static_assert(std::is_same_v<U2, typename cta2_fused_dense_attn_bf_reg::dtype>);

    const int lane = laneid();
    const int row = lane % 16;
    const uint32_t shared_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&dst.data[0])
    );
    #pragma unroll
    for (int tile_col = 0; tile_col < 4; ++tile_col) {
        const int col = tile_col * 16 + (lane / 16) * 8;
        const auto &tile = src.tiles[0][Half * 4 + tile_col];
        U2 value0 = tile.data[0];
        U2 value1 = tile.data[1];
        U2 value2 = tile.data[2];
        U2 value3 = tile.data[3];
        move<U2>::stsm4(
            dst.idx(shared_addr, {row, col}),
            value0,
            value1,
            value2,
            value3
        );
    }
}

template <ducks::st::all ST>
__device__ __forceinline__ void cta2_role_split_store_ds_stage_direct(
    ST &dst,
    const cta2_fused_dense_attn_half_bf_reg &src
) {
    static_assert(ST::cols == 64);
    using U = typename ST::dtype;
    using U2 = base_types::packing<U>::packed_type;
    static_assert(
        std::is_same_v<
            U2,
            typename cta2_fused_dense_attn_half_bf_reg::dtype
        >
    );

    const int lane = laneid();
    const int row = lane % 16;
    const uint32_t shared_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&dst.data[0])
    );
    #pragma unroll
    for (int tile_col = 0; tile_col < 4; ++tile_col) {
        const int col = tile_col * 16 + (lane / 16) * 8;
        const auto &tile = src.tiles[0][tile_col];
        U2 value0 = tile.data[0];
        U2 value1 = tile.data[1];
        U2 value2 = tile.data[2];
        U2 value3 = tile.data[3];
        move<U2>::stsm4(
            dst.idx(shared_addr, {row, col}),
            value0,
            value1,
            value2,
            value3
        );
    }
}

template <int Half, ducks::st::all ST>
__device__ __forceinline__ void cta2_role_split_store_ds_half_remote_async(
    ST &dst,
    const cta2_fused_dense_attn_bf_reg &src,
    int peer_rank,
    semaphore &completion
) {
    static_assert(Half == 0 || Half == 1);
    static_assert(ST::rows == 16 && ST::cols == 64);
    using U = typename ST::dtype;
    using U2 = base_types::packing<U>::packed_type;
    static_assert(
        std::is_same_v<U2, typename cta2_fused_dense_attn_bf_reg::dtype>
    );

    const int lane = laneid();
    const int quad_lane = lane & 3;
    const int quad_base = lane & ~3;
    const int row = lane >> 2;
    const uint32_t local_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(&dst.data[0])
    );
    const uint32_t local_completion = static_cast<uint32_t>(
        __cvta_generic_to_shared(&completion)
    );
    uint32_t remote_completion;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(remote_completion)
        : "r"(local_completion), "r"(peer_rank)
    );

    #pragma unroll
    for (int tile_col = 0; tile_col < 4; ++tile_col) {
        const auto &tile = src.tiles[0][Half * 4 + tile_col];
        #pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) {
            const uint32_t word = *reinterpret_cast<const uint32_t *>(
                &tile.data[fragment]
            );
            const uint32_t word0 = __shfl_sync(
                0xffffffff,
                word,
                quad_base + 0
            );
            const uint32_t word1 = __shfl_sync(
                0xffffffff,
                word,
                quad_base + 1
            );
            const uint32_t word2 = __shfl_sync(
                0xffffffff,
                word,
                quad_base + 2
            );
            const uint32_t word3 = __shfl_sync(
                0xffffffff,
                word,
                quad_base + 3
            );
            if (quad_lane == 0) {
                const int dst_row = row + (fragment & 1) * 8;
                const int dst_col = tile_col * 16 + (fragment >> 1) * 8;
                const uint32_t local_destination = dst.idx(
                    local_base,
                    {dst_row, dst_col}
                );
                uint32_t remote_destination;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(remote_destination)
                    : "r"(local_destination), "r"(peer_rank)
                );
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
                    "[%0], {%1, %2, %3, %4}, [%5];\n"
                    :
                    : "r"(remote_destination),
                      "r"(word0),
                      "r"(word1),
                      "r"(word2),
                      "r"(word3),
                      "r"(remote_completion)
                    : "memory"
                );
            }
        }
    }
}

__device__ __forceinline__ void
cta2_role_split_store_peer_bulk_after_cta_fence(
    void *destination,
    void *source,
    uint32_t size_bytes,
    int peer_rank,
    semaphore &completion
) {
    const uint32_t destination_shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    const uint32_t source_shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(source));
    const uint32_t completion_shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(&completion));
    uint32_t remote_destination;
    uint32_t remote_completion;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(remote_destination)
        : "r"(destination_shared), "r"(peer_rank)
    );
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(remote_completion)
        : "r"(completion_shared), "r"(peer_rank)
    );
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :
        : "r"(remote_destination),
          "r"(source_shared),
          "r"(size_bytes),
          "r"(remote_completion)
        : "memory"
    );
}

__device__ __forceinline__ float cta2_role_split_exp2_approx(float value) {
    float result;
    asm volatile(
        "ex2.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(value)
    );
    return result;
}

template <int ColumnBase, bool ApplyDiagonalMask>
__device__ __forceinline__ void cta2_role_split_make_p_x32_stage(
    uint32_t (&dst)[16],
    const uint32_t (&scores)[32],
    const float (&lse)[32],
    float scale_log2e,
    int key_row
) {
    static_assert(
        ColumnBase == 0 || ColumnBase == 32 ||
        ColumnBase == 64 || ColumnBase == 96
    );

    const float2 scale2{scale_log2e, scale_log2e};
    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    #pragma unroll
    for (int pair = 0; pair < 16; ++pair) {
        float2 score{
            __uint_as_float(scores[2 * pair + 0]),
            __uint_as_float(scores[2 * pair + 1])
        };
        const float2 neg_lse{
            lse[2 * pair + 0],
            lse[2 * pair + 1]
        };
        score = base_ops::mul::op<float2>(score, scale2);
        score = base_ops::sub::op<float2>(score, neg_lse);
        if constexpr (ApplyDiagonalMask) {
            if (key_row > ColumnBase + 2 * pair + 0) {
                score.x = kNegInf;
            }
            if (key_row > ColumnBase + 2 * pair + 1) {
                score.y = kNegInf;
            }
        }
        score.x = cta2_role_split_exp2_approx(score.x);
        score.y = cta2_role_split_exp2_approx(score.y);
        const bf16_2 packed =
            base_types::convertor<bf16_2, float2>::convert(score);
        dst[pair] = *reinterpret_cast<const uint32_t *>(&packed);
    }
}

__device__ __forceinline__ void cta2_role_split_make_ds_x32_stage(
    uint32_t (&p_ds)[16],
    const uint32_t (&dp_words)[32],
    const float (&dpsum)[32]
) {
    #pragma unroll
    for (int pair = 0; pair < 16; ++pair) {
        const uint32_t p_word = p_ds[pair];
        const float2 p{
            __uint_as_float(p_word << 16),
            __uint_as_float(p_word & 0xffff0000u)
        };
        const float2 dp{
            __uint_as_float(dp_words[2 * pair + 0]),
            __uint_as_float(dp_words[2 * pair + 1])
        };
        const float2 dpsum_pair{
            dpsum[2 * pair + 0],
            dpsum[2 * pair + 1]
        };
        const float2 centered =
            base_ops::sub::op<float2>(dp, dpsum_pair);
        const float2 ds = base_ops::mul::op<float2>(p, centered);
        const bf16_2 packed =
            base_types::convertor<bf16_2, float2>::convert(ds);
        p_ds[pair] = *reinterpret_cast<const uint32_t *>(&packed);
    }
}

__device__ __forceinline__ void cta2_role_split_exp2_approx(
    cta2_fused_dense_attn_reg &tile
) {
    #pragma unroll
    for (int tile_row = 0; tile_row < tile.height; ++tile_row) {
        #pragma unroll
        for (int tile_col = 0; tile_col < tile.width; ++tile_col) {
            #pragma unroll
            for (int packed = 0; packed < tile.packed_per_tile; ++packed) {
                float2 &value = tile.tiles[tile_row][tile_col].data[packed];
                value.x = cta2_role_split_exp2_approx(value.x);
                value.y = cta2_role_split_exp2_approx(value.y);
            }
        }
    }
}

template <bool RetainFp32 = false>
__device__ __forceinline__ void cta2_role_split_exp2_approx_pack_bf16(
    cta2_fused_dense_attn_bf_reg &dst,
    cta2_fused_dense_attn_reg &src
) {
    if constexpr (!RetainFp32) {
        #pragma unroll
        for (int tile_row = 0; tile_row < src.height; ++tile_row) {
            #pragma unroll
            for (int tile_col = 0; tile_col < src.width; ++tile_col) {
                #pragma unroll
                for (int packed = 0; packed < src.packed_per_tile; ++packed) {
                    const float2 &input =
                        src.tiles[tile_row][tile_col].data[packed];
                    const float2 value{
                        cta2_role_split_exp2_approx(input.x),
                        cta2_role_split_exp2_approx(input.y)
                    };
                    dst.tiles[tile_row][tile_col].data[packed] =
                        base_types::convertor<bf16_2, float2>::convert(value);
                }
            }
        }
    }
}

template <bool RetainFp32 = false>
__device__ __forceinline__ void
cta2_role_split_exp2_approx_pack_bf16_fragment4_first(
    cta2_fused_dense_attn_bf_reg &dst,
    cta2_fused_dense_attn_reg &src
) {
    if constexpr (!RetainFp32) {
        {
            const float2 &input = src.tiles[0][1].data[0];
            const float2 value{
                cta2_role_split_exp2_approx(input.x),
                cta2_role_split_exp2_approx(input.y)
            };
            dst.tiles[0][1].data[0] =
                base_types::convertor<bf16_2, float2>::convert(value);
        }
        #pragma unroll
        for (int tile_row = 0; tile_row < src.height; ++tile_row) {
            #pragma unroll
            for (int tile_col = 0; tile_col < src.width; ++tile_col) {
                #pragma unroll
                for (int packed = 0; packed < src.packed_per_tile; ++packed) {
                    if (tile_row != 0 || tile_col != 1 || packed != 0) {
                        const float2 &input =
                            src.tiles[tile_row][tile_col].data[packed];
                        const float2 value{
                            cta2_role_split_exp2_approx(input.x),
                            cta2_role_split_exp2_approx(input.y)
                        };
                        dst.tiles[tile_row][tile_col].data[packed] =
                            base_types::convertor<bf16_2, float2>::convert(
                                value
                            );
                        }
                }
            }
        }
    }
}

__device__ __forceinline__ void cta2_role_split_load_stats_direct(
    cta2_fused_dense_stats_vec &dst,
    const sv_fl<128> &src
) {
    const int column_pair = laneid() & 3;
    const uint32_t src_ptr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&src.data[0])
    );
    __syncwarp();
    #pragma unroll
    for (int fragment = 0; fragment < dst.outer_dim; ++fragment) {
        move<float2>::lds(
            dst[fragment][0],
            src_ptr + sizeof(float) *
                (fragment * 16 + 2 * column_pair)
        );
        move<float2>::lds(
            dst[fragment][1],
            src_ptr + sizeof(float) *
                (fragment * 16 + 8 + 2 * column_pair)
        );
    }
    __syncwarp();
}

template <int ColumnBase>
__device__ __forceinline__ void
cta2_role_split_load_stats_direct_quarter(
    cta2_fused_dense_quarter_stats_vec &dst,
    const sv_fl<128> &src
) {
    static_assert(
        ColumnBase == 0 || ColumnBase == 32 ||
        ColumnBase == 64 || ColumnBase == 96
    );
    const int column_pair = laneid() & 3;
    const uint32_t src_ptr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&src.data[0])
    );
    __syncwarp();
    #pragma unroll
    for (int fragment = 0; fragment < dst.outer_dim; ++fragment) {
        move<float2>::lds(
            dst[fragment][0],
            src_ptr + sizeof(float) *
                (ColumnBase + fragment * 16 + 2 * column_pair)
        );
        move<float2>::lds(
            dst[fragment][1],
            src_ptr + sizeof(float) *
                (ColumnBase + fragment * 16 + 8 + 2 * column_pair)
        );
    }
    __syncwarp();
}

template <int Half>
__device__ __forceinline__ void
cta2_role_split_issue_dp_fp32_half(
    cta2_fused_dense_attn_half_reg &dp_half,
    const cta2_fused_dense_attn_tt &dp_tmem
) {
    static_assert(Half == 0 || Half == 1);
    auto dp_half_tmem =
        dp_tmem.template subtile<cta2_fused_dense_attn_half_tt>(
            Half * 64
        );
    group<8>::load_async(dp_half, dp_half_tmem);
}

template <int Half>
__device__ __forceinline__ void
cta2_role_split_load_dp_fp32_half(
    cta2_fused_dense_attn_half_reg &dp_half,
    const cta2_fused_dense_attn_tt &dp_tmem
) {
    cta2_role_split_issue_dp_fp32_half<Half>(dp_half, dp_tmem);
    tensor_load_wait();
}

template <int Quarter>
__device__ __forceinline__ void
cta2_role_split_make_ds_fp32_quarter(
    cta2_fused_dense_attn_quarter_bf_reg &ds_quarter,
    const cta2_fused_dense_attn_quarter_reg &p_quarter,
    cta2_fused_dense_attn_quarter_reg &dp_quarter,
    const cta2_fused_dense_quarter_stats_vec &dpsum
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    warp::sub_col(dp_quarter, dp_quarter, dpsum);
    warp::mul(dp_quarter, p_quarter, dp_quarter);
    warp::copy(ds_quarter, dp_quarter);
}

template <int Quarter>
__device__ __forceinline__ void
cta2_role_split_pack_p_fp32_quarter(
    cta2_fused_dense_attn_reg &p_fp32,
    cta2_fused_dense_attn_quarter_bf_reg &p_bf_quarter
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    constexpr int kGlobalTileOffset =
        Quarter * cta2_fused_dense_attn_quarter_reg::width;
    auto &p_quarter = *reinterpret_cast<
        cta2_fused_dense_attn_quarter_reg *
    >(&p_fp32.tiles[0][kGlobalTileOffset]);
    if constexpr (Quarter == 0) {
        float2 &input = p_quarter.tiles[0][1].data[0];
        const float2 value{
            cta2_role_split_exp2_approx(input.x),
            cta2_role_split_exp2_approx(input.y)
        };
        input = value;
        p_bf_quarter.tiles[0][1].data[0] =
            base_types::convertor<bf16_2, float2>::convert(value);
    }
    #pragma unroll
    for (int tile_col = 0; tile_col < p_quarter.width; ++tile_col) {
        #pragma unroll
        for (
            int packed = 0;
            packed < p_quarter.packed_per_tile;
            ++packed
        ) {
            if (
                Quarter != 0 || tile_col != 1 || packed != 0
            ) {
                float2 &input =
                    p_quarter.tiles[0][tile_col].data[packed];
                const float2 value{
                    cta2_role_split_exp2_approx(input.x),
                    cta2_role_split_exp2_approx(input.y)
                };
                input = value;
                p_bf_quarter.tiles[0][tile_col].data[packed] =
                    base_types::convertor<bf16_2, float2>::convert(value);
            }
        }
    }
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ void cta2_role_split_load_stats_global_direct(
    cta2_fused_dense_stats_vec &dst,
    float *src
) {
    const int column_pair = laneid() & 3;
    #pragma unroll
    for (int fragment = 0; fragment < dst.outer_dim; ++fragment) {
        move<float2>::ldg(
            dst[fragment][0],
            reinterpret_cast<float2 *>(
                src + fragment * 16 + 2 * column_pair
            )
        );
        move<float2>::ldg(
            dst[fragment][1],
            reinterpret_cast<float2 *>(
                src + fragment * 16 + 8 + 2 * column_pair
            )
        );
    }
}

template <int FirstFragment, int EndFragment>
__device__ __forceinline__ void
cta2_role_split_load_stats_global_direct_range(
    cta2_fused_dense_stats_vec &dst,
    float *src
) {
    static_assert(0 <= FirstFragment && FirstFragment < EndFragment);
    static_assert(EndFragment <= cta2_fused_dense_stats_vec::outer_dim);
    const int column_pair = laneid() & 3;
    #pragma unroll
    for (
        int fragment = FirstFragment;
        fragment < EndFragment;
        ++fragment
    ) {
        move<float2>::ldg(
            dst[fragment][0],
            reinterpret_cast<float2 *>(
                src + fragment * 16 + 2 * column_pair
            )
        );
        move<float2>::ldg(
            dst[fragment][1],
            reinterpret_cast<float2 *>(
                src + fragment * 16 + 8 + 2 * column_pair
            )
        );
    }
}

__device__ __forceinline__ uint2 cta2_fused_dense_pack_bf16x4(
    const float2 &low_values,
    const float2 &high_values
) {
    const bf16_2 low =
        base_types::convertor<bf16_2, float2>::convert(low_values);
    const bf16_2 high =
        base_types::convertor<bf16_2, float2>::convert(high_values);
    const uint32_t low_word = *reinterpret_cast<const uint32_t *>(&low);
    const uint32_t high_word = *reinterpret_cast<const uint32_t *>(&high);
    const int lane_in_row = laneid() & 3;
    const int first_lane = (laneid() & ~3) + 2 * (lane_in_row & 1);
    const uint2 low_pair{
        __shfl_sync(0xffffffff, low_word, first_lane + 0),
        __shfl_sync(0xffffffff, low_word, first_lane + 1)
    };
    const uint2 high_pair{
        __shfl_sync(0xffffffff, high_word, first_lane + 0),
        __shfl_sync(0xffffffff, high_word, first_lane + 1)
    };
    return lane_in_row < 2 ? low_pair : high_pair;
}

__device__ __forceinline__ void cta2_fused_dense_store_bf16_coalesced(
    const cta2_fused_dense_output_gl_t<bf16> &dst,
    const rt_fl<32, 64> &src,
    int batch_idx,
    int kv_tile,
    int head_idx,
    int chunk
) {
    bf16 *dst_ptr = &dst[coord<>(
        batch_idx,
        kv_tile * 128,
        head_idx,
        chunk * 64
    )];
    const int row_stride = dst.template stride<dim::DEPTH>();
    const int lane = laneid();
    const int lane_in_row = lane & 3;
    const int row_offset = 32 * (warpid() & 3);

    #pragma unroll
    for (int i = 0; i < rt_fl<32, 64>::height; ++i) {
        const int row = row_offset + i * 16 + lane / 4;
        #pragma unroll
        for (int j = 0; j < rt_fl<32, 64>::width; ++j) {
            const uint2 packed = cta2_fused_dense_pack_bf16x4(
                src.tiles[i][j].data[0],
                src.tiles[i][j].data[2]
            );
            *reinterpret_cast<uint2 *>(
                &dst_ptr[row * row_stride + j * 16 + lane_in_row * 4]
            ) = packed;
        }
        #pragma unroll
        for (int j = 0; j < rt_fl<32, 64>::width; ++j) {
            const uint2 packed = cta2_fused_dense_pack_bf16x4(
                src.tiles[i][j].data[1],
                src.tiles[i][j].data[3]
            );
            *reinterpret_cast<uint2 *>(
                &dst_ptr[
                    (row + 8) * row_stride + j * 16 + lane_in_row * 4
                ]
            ) = packed;
        }
    }
}

__device__ __forceinline__ void cta2_fused_dense_store_bf16_coalesced_wide(
    const cta2_fused_dense_output_gl_t<bf16> &dst,
    const rt_fl<32, 64> &src,
    int batch_idx,
    int kv_tile,
    int head_idx,
    int chunk
) {
    bf16 *dst_ptr = &dst[coord<>(
        batch_idx,
        kv_tile * 128,
        head_idx,
        chunk * 64
    )];
    const int row_stride = dst.template stride<dim::DEPTH>();
    const int lane = laneid();
    const int lane_in_row = lane & 3;
    const int row_offset = 32 * (warpid() & 3);

    #pragma unroll
    for (int i = 0; i < rt_fl<32, 64>::height; ++i) {
        const int row = row_offset + i * 16 + lane / 4;
        #pragma unroll
        for (int j = 0; j < rt_fl<32, 64>::width; ++j) {
            const uint2 packed = cta2_fused_dense_pack_bf16x4(
                src.tiles[i][j].data[0],
                src.tiles[i][j].data[2]
            );
            const int peer_lane = lane | 1;
            const uint4 packed_wide{
                packed.x,
                packed.y,
                __shfl_sync(0xffffffff, packed.x, peer_lane),
                __shfl_sync(0xffffffff, packed.y, peer_lane)
            };
            if ((lane_in_row & 1) == 0) {
                *reinterpret_cast<uint4 *>(
                    &dst_ptr[
                        row * row_stride + j * 16 + lane_in_row * 4
                    ]
                ) = packed_wide;
            }
        }
        #pragma unroll
        for (int j = 0; j < rt_fl<32, 64>::width; ++j) {
            const uint2 packed = cta2_fused_dense_pack_bf16x4(
                src.tiles[i][j].data[1],
                src.tiles[i][j].data[3]
            );
            const int peer_lane = lane | 1;
            const uint4 packed_wide{
                packed.x,
                packed.y,
                __shfl_sync(0xffffffff, packed.x, peer_lane),
                __shfl_sync(0xffffffff, packed.y, peer_lane)
            };
            if ((lane_in_row & 1) == 0) {
                *reinterpret_cast<uint4 *>(
                    &dst_ptr[
                        (row + 8) * row_stride + j * 16 + lane_in_row * 4
                    ]
                ) = packed_wide;
            }
        }
    }
}

template <typename Tile>
__device__ inline void cta2_role_split_apply_diagonal_causal_mask(
    Tile &scores,
    int output_subtile
) {
    constexpr float kNegInf =
        kittens::base_types::constants<float>::neg_infty();
    const int k_row_base = output_subtile * 16;
    warp::apply(scores, scores, [=](int row, int col, float value) {
        return k_row_base + row > col ? kNegInf : value;
    });
}

template <typename LocalST, typename PeerST>
__device__ __forceinline__ void cta2_role_split_load_do_half_branchless(
    rt_bf<16, 64> &dst,
    const LocalST &local_src,
    const PeerST &peer_src,
    int use_local
) {
    static_assert(LocalST::rows == 16 && LocalST::cols == 64);
    static_assert(PeerST::rows == 16 && PeerST::cols == 64);
    using packed_bf16 = typename rt_bf<16, 64>::dtype;
    static_assert(std::is_same_v<packed_bf16, bf16_2>);

    const uint32_t local_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(&local_src.data[0])
    );
    const uint32_t peer_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(&peer_src.data[0])
    );
    const int row = laneid() & 15;
    const int lane_col = (laneid() >> 4) * 8;
    #pragma unroll
    for (int tile_col = 0; tile_col < 4; ++tile_col) {
        const int col = tile_col * 16 + lane_col;
        const uint32_t local_addr = local_src.idx(
            local_base,
            {row, col}
        );
        const uint32_t peer_addr = peer_src.idx(
            peer_base,
            {row, col}
        );
        uint32_t selected_addr;
        asm volatile(
            "{\n"
            "  .reg .pred use_local_pred;\n"
            "  setp.ne.u32 use_local_pred, %3, 0;\n"
            "  selp.u32 %0, %1, %2, use_local_pred;\n"
            "}\n"
            : "=r"(selected_addr)
            : "r"(local_addr), "r"(peer_addr), "r"(use_local)
        );
        packed_bf16 tmp[4];
        move<packed_bf16>::ldsm4(
            tmp[0],
            tmp[1],
            tmp[2],
            tmp[3],
            selected_addr
        );
        dst.tiles[0][tile_col].data[0] = tmp[0];
        dst.tiles[0][tile_col].data[1] = tmp[1];
        dst.tiles[0][tile_col].data[2] = tmp[2];
        dst.tiles[0][tile_col].data[3] = tmp[3];
    }
}

template <typename LocalST, typename PeerST>
__device__ __forceinline__ void
cta2_role_split_load_do_half_branchless_base_select(
    rt_bf<16, 64> &dst,
    const LocalST &local_src,
    const PeerST &peer_src,
    int use_local
) {
    static_assert(LocalST::rows == 16 && LocalST::cols == 64);
    static_assert(PeerST::rows == 16 && PeerST::cols == 64);
    static_assert(std::is_same_v<typename LocalST::T, typename PeerST::T>);
    static_assert(LocalST::underlying_rows == 64);
    static_assert(PeerST::underlying_rows == 64);
    static_assert(LocalST::swizzle_bytes == 128);
    static_assert(PeerST::swizzle_bytes == 128);
    using value_type = typename LocalST::T;
    using packed_bf16 = typename rt_bf<16, 64>::dtype;
    static_assert(sizeof(value_type) == sizeof(bf16));
    static_assert(std::is_same_v<packed_bf16, bf16_2>);
    constexpr uint32_t kSwizzleColumns =
        LocalST::swizzle_bytes / sizeof(value_type);
    constexpr uint32_t kSwizzleRepeat = LocalST::swizzle_bytes * 8;
    static_assert(kSwizzleColumns == 64);

    const uint32_t local_parent_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(&local_src.data[0])
    );
    const uint32_t peer_parent_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(&peer_src.data[0])
    );
    const uint32_t local_tile_base = local_parent_base + sizeof(value_type) * (
        (local_src.col_offset / kSwizzleColumns) *
            LocalST::underlying_rows * kSwizzleColumns +
        local_src.row_offset * kSwizzleColumns +
        local_src.col_offset % kSwizzleColumns
    );
    const uint32_t peer_tile_base = peer_parent_base + sizeof(value_type) * (
        (peer_src.col_offset / kSwizzleColumns) *
            PeerST::underlying_rows * kSwizzleColumns +
        peer_src.row_offset * kSwizzleColumns +
        peer_src.col_offset % kSwizzleColumns
    );
    uint32_t selected_tile_base;
    asm volatile(
        "{\n"
        "  .reg .pred use_local_pred;\n"
        "  setp.ne.u32 use_local_pred, %3, 0;\n"
        "  selp.u32 %0, %1, %2, use_local_pred;\n"
        "}\n"
        : "=r"(selected_tile_base)
        : "r"(local_tile_base), "r"(peer_tile_base), "r"(use_local)
    );

    const int row = laneid() & 15;
    const int lane_col = (laneid() >> 4) * 8;
    #pragma unroll
    for (int tile_col = 0; tile_col < 4; ++tile_col) {
        const int col = tile_col * 16 + lane_col;
        const uint32_t linear_addr = selected_tile_base +
            sizeof(value_type) * (row * kSwizzleColumns + col);
        const uint32_t swizzle =
            ((linear_addr % kSwizzleRepeat) >> 7) << 4;
        const uint32_t selected_addr = linear_addr ^ swizzle;
        packed_bf16 tmp[4];
        move<packed_bf16>::ldsm4(
            tmp[0],
            tmp[1],
            tmp[2],
            tmp[3],
            selected_addr
        );
        dst.tiles[0][tile_col].data[0] = tmp[0];
        dst.tiles[0][tile_col].data[1] = tmp[1];
        dst.tiles[0][tile_col].data[2] = tmp[2];
        dst.tiles[0][tile_col].data[3] = tmp[3];
    }
}

template <
    bool PreloadDpsum,
    bool RetainDsExchange = false,
    bool RetainDsLocal = false,
    bool UseNormalDoDv = false,
    bool UseTmaScoreK = false,
    bool DirectNextQdoDuringDqDrain = false,
    bool SingleOwnerCluster = false,
    bool UseFastExp2 = false,
    bool UseWarpStatsCache = false,
    bool PipelineLsePrefetch = false,
    bool UseDirectStatsLoads = false,
    bool SplitDvDkReady = false,
    bool StageDqAfterDv = false,
    bool StageDqPeerBeforeDv = false,
    bool UseWideDkN192 = false,
    bool DirectDsHalfStore = false,
    bool AsymmetricDvPublish = false,
    typename DkdvOutT = float,
    bool CoalescedBf16Store = false,
    bool DirectAsyncPeerDs = false,
    bool ProducerBulkPeerDs = false,
    bool ProducerBulkPeerDsCtaFenceOnly = false,
    bool DqReadHandoffBeforeCompletion = false,
    bool AggregateScoreConsumed = false,
    bool DirectTmaDkQ = false,
    bool TimeoutDqWait = false,
    bool TimeoutAllRoleWaits = false,
    bool UseNamedDoSourceBarrier = false,
    bool UseComputeScoreFanout = false,
    bool UseRuntimeAccumulationPredicate = false,
    bool UseReducerDqFanout = false,
    bool UseReducerDqLeaderArrive = false,
    bool MergeScoreDpReady = false,
    bool WideCoalescedBf16Store = false,
    bool BulkPeerDsFromFullTile = false,
    bool CoalescedPeerDsBulk = false,
    bool WideDqKGlobalToShared = false,
    bool IntegrateCausalFrontier = false,
    int ExactQTileCount = 0,
    bool FenceDsBeforeDkdvReady = false,
    bool UseNamedDkdvLocalFanIn = false,
    bool LeaderOnlyQdoPublishFence = false,
    bool CacheQdoReadyClusterAddress = false,
    bool GroupQdoTmaLoads = false,
    bool ElectedWideDkQTmaLoad = false,
    bool ElectedPeerDoTmaLoad = false,
    bool CacheRoleClusterAddresses = false,
    bool CacheTensorCommitAddresses = false,
    bool EnsureReducerOutputDrain = false,
    bool ElectedScoreKTmaLoad = false,
    bool UseExactClusterCoordinates = false,
    bool EnforceDpTmemConsumerRelease = false,
    bool SplitDpTmemConsumerRelease = false,
    bool UseIterationCausalMask = false,
    bool UseFusedTmemPAndDs = false,
    bool OverlapFusedDqAPublication = false,
    bool PrefetchNextQdoAfterDkdv = false,
    bool PrefetchNextOwnerQdo = false,
    bool UseFusedTmemRuntimeAccumulationPredicate = false,
    bool UseBitwisePExpansion = false,
    bool UseFusedExp2Pack = false,
    bool PeelCausalPrefix = false,
    bool BranchlessDoSourceLoad = false,
    bool BranchlessDoSourceBaseSelect = false,
    bool PublishVOncePerOwner = false,
    int OwnerQWorkSplitId = -1,
    bool BulkDoDvStage = false,
    bool LoaderOwnedDkQ = false,
    bool FuseScoreScaleLse = false,
    bool RetainPackedP = false,
    bool SplitDirectDpsumAcrossDpDoneWait = false,
    bool FusedExp2Fragment4First = false,
    bool CarryDirectStatsOffset = false,
    bool CarryAllRolePhases = false,
    bool UseExactDefaultScaleLog2e = false,
    bool ReverseDkTailTmemLoadIssue = false,
    bool PrearmNextQdoBeforeDkDone = false,
    bool UseX32TmemComputeLayout = false,
    bool UseLongSeqStatsCache = false,
    bool UseCompactScoreMma = false,
    bool UseCompactDpMma = false,
    bool UsePackedBf16DsProduct = false,
    bool SplitDqTmemAndSharedHandoff = false,
    bool DistributedDqSharedReadWait = false,
    bool CacheDqStageLanePointers = false,
    bool UseSlicedFp32PForDs = false,
    bool UseTmaVWithScoreK = false,
    bool UseStatsWarpScoreFanout = false,
    bool UseBatchedDqTmemLoads = false,
    bool UseDynamicDpReleaseBarrierId = false,
    bool PreissueFirstDpHalfBeforeQdoWait = false,
    bool OverlapSecondDpLoadWithReleaseBarrier = false
>
__global__ __launch_bounds__(512)
void main_kernel_causal_cta2_fused_dense_role_split(
    const __grid_constant__
        cta2_fused_dense_globals_t<DkdvOutT, DirectTmaDkQ> g
) {
    constexpr bool UseRoleStatsCache =
        UseWarpStatsCache || UseLongSeqStatsCache;
    static_assert(
        std::is_same_v<DkdvOutT, float> || std::is_same_v<DkdvOutT, bf16>
    );
    static_assert(
        OwnerQWorkSplitId == -1 || OwnerQWorkSplitId == 0 ||
            OwnerQWorkSplitId == 1
    );
    static_assert(
        OwnerQWorkSplitId < 0 ||
            ((ExactQTileCount == 16 || ExactQTileCount == 32 ||
              ExactQTileCount == 64) &&
             IntegrateCausalFrontier &&
             !UseWarpStatsCache && !SingleOwnerCluster)
    );
    static_assert(!CoalescedBf16Store || std::is_same_v<DkdvOutT, bf16>);
    static_assert(
        !PeelCausalPrefix || (UseIterationCausalMask && UseFusedExp2Pack)
    );
    static_assert(
        !BranchlessDoSourceBaseSelect || BranchlessDoSourceLoad
    );
    static_assert(
        !PublishVOncePerOwner ||
            (UseFusedTmemPAndDs && DirectNextQdoDuringDqDrain &&
             UseTmaScoreK && SplitDvDkReady && UseNormalDoDv &&
             MergeScoreDpReady && IntegrateCausalFrontier &&
             PrefetchNextQdoAfterDkdv)
    );
    static_assert(
        !UseTmaVWithScoreK ||
            (UseTmaScoreK && ElectedScoreKTmaLoad &&
             UseFusedTmemPAndDs && MergeScoreDpReady &&
             SingleOwnerCluster)
    );
    static_assert(
        !UseStatsWarpScoreFanout ||
            (UseComputeScoreFanout && UseWarpStatsCache &&
             PipelineLsePrefetch && SingleOwnerCluster)
    );
    static_assert(
        !UseBatchedDqTmemLoads ||
            (SplitDqTmemAndSharedHandoff &&
             DqReadHandoffBeforeCompletion &&
             CacheDqStageLanePointers && UseSlicedFp32PForDs)
    );
    static_assert(
        !UseDynamicDpReleaseBarrierId ||
            (EnforceDpTmemConsumerRelease &&
             SplitDpTmemConsumerRelease)
    );
    static_assert(
        !PreissueFirstDpHalfBeforeQdoWait ||
            (UseSlicedFp32PForDs && BulkDoDvStage &&
             SplitDvDkReady && PreloadDpsum)
    );
    static_assert(
        !OverlapSecondDpLoadWithReleaseBarrier ||
            (PreissueFirstDpHalfBeforeQdoWait &&
             EnforceDpTmemConsumerRelease &&
             SplitDpTmemConsumerRelease)
    );
    static_assert(
        !BulkDoDvStage ||
            (ExactQTileCount == 512 && SplitDvDkReady && UseNormalDoDv &&
             UseFusedTmemPAndDs && StageDqAfterDv &&
             DirectNextQdoDuringDqDrain && OverlapFusedDqAPublication &&
             IntegrateCausalFrontier && DirectTmaDkQ &&
             ElectedPeerDoTmaLoad && UseNamedDoSourceBarrier &&
             OwnerQWorkSplitId < 0)
    );
    static_assert(
        !LoaderOwnedDkQ ||
            (BulkDoDvStage && ExactQTileCount == 512 &&
             DirectTmaDkQ && UseWideDkN192 &&
             ElectedWideDkQTmaLoad && GroupQdoTmaLoads &&
             DirectNextQdoDuringDqDrain &&
             (PrefetchNextOwnerQdo || SingleOwnerCluster))
    );
    static_assert(
        !FuseScoreScaleLse ||
            (LoaderOwnedDkQ && PeelCausalPrefix &&
             UseIterationCausalMask && UseFusedExp2Pack)
    );
    static_assert(
        !RetainPackedP ||
            (FuseScoreScaleLse && UseFusedTmemPAndDs && PreloadDpsum &&
             SplitDvDkReady && BulkDoDvStage)
    );
    static_assert(
        !SplitDirectDpsumAcrossDpDoneWait ||
            (ExactQTileCount == 512 && PreloadDpsum &&
             UseDirectStatsLoads && !UseWarpStatsCache &&
             IntegrateCausalFrontier && OwnerQWorkSplitId < 0)
    );
    static_assert(
        !FusedExp2Fragment4First ||
            (FuseScoreScaleLse && UseFusedExp2Pack &&
             ExactQTileCount == 512 && IntegrateCausalFrontier &&
             OwnerQWorkSplitId < 0)
    );
    static_assert(
        !CarryDirectStatsOffset ||
            (FusedExp2Fragment4First &&
             SplitDirectDpsumAcrossDpDoneWait &&
             ExactQTileCount == 512 && PreloadDpsum &&
             UseDirectStatsLoads && !UseWarpStatsCache &&
             IntegrateCausalFrontier && OwnerQWorkSplitId < 0)
    );
    static_assert(
        !CarryAllRolePhases ||
            (ExactQTileCount == 512 && IntegrateCausalFrontier &&
             OwnerQWorkSplitId < 0)
    );
    static_assert(
        !UseExactDefaultScaleLog2e ||
            (FuseScoreScaleLse && ExactQTileCount == 512 &&
             IntegrateCausalFrontier && OwnerQWorkSplitId < 0)
    );
    static_assert(
        !ReverseDkTailTmemLoadIssue ||
            (WideCoalescedBf16Store && std::is_same_v<DkdvOutT, bf16>)
    );
    static_assert(
        !PrearmNextQdoBeforeDkDone ||
            (GroupQdoTmaLoads && PrefetchNextQdoAfterDkdv &&
             DirectNextQdoDuringDqDrain)
    );
    static_assert(
        !UseX32TmemComputeLayout ||
            (ExactQTileCount == 512 && FuseScoreScaleLse &&
             RetainPackedP && SplitDirectDpsumAcrossDpDoneWait &&
             UseFusedTmemPAndDs && OverlapFusedDqAPublication &&
             DirectDsHalfStore && UseBitwisePExpansion &&
             EnforceDpTmemConsumerRelease &&
             SplitDpTmemConsumerRelease &&
             UseComputeScoreFanout &&
             IntegrateCausalFrontier && OwnerQWorkSplitId < 0)
    );
    static_assert(
        !UseLongSeqStatsCache ||
            (ExactQTileCount == 512 && PreloadDpsum &&
             UseDirectStatsLoads && UseComputeScoreFanout &&
             IntegrateCausalFrontier && OwnerQWorkSplitId < 0 &&
             !UseWarpStatsCache && !UseX32TmemComputeLayout &&
             LoaderOwnedDkQ && GroupQdoTmaLoads &&
             DirectNextQdoDuringDqDrain && PrefetchNextQdoAfterDkdv &&
             PrefetchNextOwnerQdo && PrearmNextQdoBeforeDkDone)
    );
    static_assert(
        !UseCompactScoreMma ||
            (ExactQTileCount == 512 && CacheTensorCommitAddresses &&
             UseComputeScoreFanout && IntegrateCausalFrontier &&
             OwnerQWorkSplitId < 0 && !UseX32TmemComputeLayout)
    );
    static_assert(
        !UseCompactDpMma ||
            (UseCompactScoreMma && ExactQTileCount == 512 &&
             CacheTensorCommitAddresses && PreloadDpsum &&
             IntegrateCausalFrontier && OwnerQWorkSplitId < 0 &&
             !UseX32TmemComputeLayout)
    );
    static_assert(
        !UsePackedBf16DsProduct ||
            (ExactQTileCount == 512 && RetainPackedP && PreloadDpsum &&
             UseBitwisePExpansion && UseFusedTmemPAndDs &&
             IntegrateCausalFrontier && OwnerQWorkSplitId < 0 &&
             !UseX32TmemComputeLayout)
    );
    static_assert(
        !SplitDqTmemAndSharedHandoff ||
            (ExactQTileCount == 512 && DqReadHandoffBeforeCompletion &&
             DirectNextQdoDuringDqDrain && OverlapFusedDqAPublication &&
             UseReducerDqFanout && CarryAllRolePhases &&
             OwnerQWorkSplitId < 0 && !UseX32TmemComputeLayout)
    );
    static_assert(
        !DistributedDqSharedReadWait || SplitDqTmemAndSharedHandoff
    );
    static_assert(
        !CacheDqStageLanePointers ||
            (ExactQTileCount == 512 && UseReducerDqFanout)
    );
    static_assert(
        !UseSlicedFp32PForDs ||
            (ExactQTileCount == 512 && RetainPackedP &&
             UseFusedTmemPAndDs && UseWarpStatsCache &&
             UseDirectStatsLoads && !UseLongSeqStatsCache &&
             FuseScoreScaleLse && UseFusedExp2Pack &&
             FusedExp2Fragment4First &&
             UsePackedBf16DsProduct && DirectDsHalfStore &&
             OverlapFusedDqAPublication && !UseX32TmemComputeLayout)
    );
    static_assert(!WideCoalescedBf16Store || CoalescedBf16Store);
    static_assert(!RetainDsLocal || RetainDsExchange);
    static_assert(
        !UseNormalDoDv ||
            (PreloadDpsum && RetainDsExchange && RetainDsLocal)
    );
    static_assert(!UseTmaScoreK || UseNormalDoDv);
    static_assert(!DirectNextQdoDuringDqDrain || UseTmaScoreK);
    static_assert(!SingleOwnerCluster || DirectNextQdoDuringDqDrain);
    static_assert(!UseFastExp2 || SingleOwnerCluster || DirectDsHalfStore);
    static_assert(
        !UseWarpStatsCache ||
            (PreloadDpsum && SingleOwnerCluster && UseFastExp2)
    );
    static_assert(!PipelineLsePrefetch || UseWarpStatsCache);
    static_assert(
        !UseDirectStatsLoads || !UseWarpStatsCache || PipelineLsePrefetch
    );
    static_assert(
        !SplitDvDkReady ||
            (PreloadDpsum && UseNormalDoDv)
    );
    static_assert(!StageDqAfterDv || SplitDvDkReady);
    static_assert(!StageDqPeerBeforeDv || StageDqAfterDv);
    static_assert(!UseWideDkN192 || PreloadDpsum);
    static_assert(
        !DirectDsHalfStore || (RetainDsExchange && RetainDsLocal)
    );
    static_assert(
        !AsymmetricDvPublish ||
            (SplitDvDkReady && UseNormalDoDv &&
             (!SingleOwnerCluster || BulkDoDvStage))
    );
    static_assert(!TimeoutDqWait || DirectNextQdoDuringDqDrain);
    static_assert(
        !DirectAsyncPeerDs ||
            (DirectDsHalfStore && StageDqAfterDv &&
             !StageDqPeerBeforeDv && !SingleOwnerCluster)
    );
    static_assert(
        !ProducerBulkPeerDs ||
            (DirectDsHalfStore && StageDqAfterDv &&
             !StageDqPeerBeforeDv &&
             (!SingleOwnerCluster || BulkDoDvStage))
    );
    static_assert(!DirectAsyncPeerDs || !ProducerBulkPeerDs);
    static_assert(
        !ProducerBulkPeerDsCtaFenceOnly || ProducerBulkPeerDs
    );
    static_assert(!DirectTmaDkQ || PreloadDpsum);
    static_assert(!UseReducerDqLeaderArrive || UseReducerDqFanout);
    static_assert(!MergeScoreDpReady || AggregateScoreConsumed);
    static_assert(!BulkPeerDsFromFullTile || ProducerBulkPeerDs);
    static_assert(
        !CoalescedPeerDsBulk || BulkPeerDsFromFullTile
    );
    static_assert(
        !WideDqKGlobalToShared || CoalescedPeerDsBulk
    );
    static_assert(
        !IntegrateCausalFrontier ||
            (PreloadDpsum && SplitDvDkReady && StageDqAfterDv &&
             DirectTmaDkQ)
    );
    static_assert(!FenceDsBeforeDkdvReady || IntegrateCausalFrontier);
    static_assert(!UseNamedDkdvLocalFanIn || FenceDsBeforeDkdvReady);
    static_assert(
        !LeaderOnlyQdoPublishFence ||
            (DirectNextQdoDuringDqDrain && DirectTmaDkQ)
    );
    static_assert(
        !CacheQdoReadyClusterAddress || LeaderOnlyQdoPublishFence
    );
    static_assert(
        !GroupQdoTmaLoads ||
            (IntegrateCausalFrontier &&
             (ExactQTileCount == 128 || ExactQTileCount == 256 ||
              ExactQTileCount == 512) &&
             DirectNextQdoDuringDqDrain && DirectTmaDkQ &&
             LeaderOnlyQdoPublishFence && CacheQdoReadyClusterAddress)
    );
    static_assert(
        !ElectedWideDkQTmaLoad ||
            (GroupQdoTmaLoads && DirectTmaDkQ && UseWideDkN192)
    );
    static_assert(
        !ElectedPeerDoTmaLoad ||
            DirectTmaDkQ
    );
    static_assert(
        !CacheRoleClusterAddresses ||
            (IntegrateCausalFrontier && SplitDvDkReady && StageDqAfterDv &&
             UseNamedDkdvLocalFanIn && ProducerBulkPeerDs &&
             (!SingleOwnerCluster || BulkDoDvStage))
    );
    static_assert(
        !CacheTensorCommitAddresses ||
            (CacheRoleClusterAddresses && IntegrateCausalFrontier &&
             (!SingleOwnerCluster || BulkDoDvStage))
    );
    static_assert(
        !EnsureReducerOutputDrain ||
            (CacheTensorCommitAddresses && IntegrateCausalFrontier &&
             (!SingleOwnerCluster || BulkDoDvStage))
    );
    static_assert(
        !ElectedScoreKTmaLoad ||
            (EnsureReducerOutputDrain && UseTmaScoreK &&
             IntegrateCausalFrontier &&
             (ExactQTileCount == 128 || ExactQTileCount == 256 ||
              ExactQTileCount == 512))
    );
    static_assert(
        !UseExactClusterCoordinates ||
            (IntegrateCausalFrontier && ExactQTileCount > 0)
    );
    static_assert(
        !EnforceDpTmemConsumerRelease ||
            (UseExactClusterCoordinates && PreloadDpsum &&
             AggregateScoreConsumed && MergeScoreDpReady)
    );
    static_assert(
        !SplitDpTmemConsumerRelease || EnforceDpTmemConsumerRelease
    );
    static_assert(
        !UseIterationCausalMask ||
            (IntegrateCausalFrontier && ExactQTileCount > 0)
    );
    static_assert(
        !UseFusedTmemPAndDs ||
            (UseIterationCausalMask && SplitDpTmemConsumerRelease &&
             UseFastExp2 && SplitDvDkReady && UseNormalDoDv &&
             DirectDsHalfStore && ProducerBulkPeerDs &&
             BulkPeerDsFromFullTile && CoalescedPeerDsBulk &&
             UseWideDkN192)
    );
    static_assert(
        !OverlapFusedDqAPublication ||
            (UseFusedTmemPAndDs && StageDqAfterDv &&
             DirectDsHalfStore && ProducerBulkPeerDs &&
             ProducerBulkPeerDsCtaFenceOnly &&
             BulkPeerDsFromFullTile && CoalescedPeerDsBulk &&
             !StageDqPeerBeforeDv &&
             (!SingleOwnerCluster || BulkDoDvStage))
    );
    static_assert(
        !PrefetchNextQdoAfterDkdv ||
            (OverlapFusedDqAPublication &&
             DirectNextQdoDuringDqDrain && StageDqAfterDv &&
             UseDirectStatsLoads &&
             (!UseWarpStatsCache || PipelineLsePrefetch))
    );
    static_assert(
        !PrefetchNextOwnerQdo ||
            (PrefetchNextQdoAfterDkdv && IntegrateCausalFrontier &&
             (ExactQTileCount == 128 || ExactQTileCount == 256 ||
              ExactQTileCount == 512) &&
             !SingleOwnerCluster)
    );
    static_assert(
        !UseFusedTmemRuntimeAccumulationPredicate ||
            (UseFusedTmemPAndDs && UseRuntimeAccumulationPredicate)
    );
    static_assert(
        !UseBitwisePExpansion ||
            (UseFusedTmemPAndDs && UseFusedTmemRuntimeAccumulationPredicate &&
             IntegrateCausalFrontier &&
             (ExactQTileCount == 16 || ExactQTileCount == 32 ||
              ExactQTileCount == 64 ||
              ExactQTileCount == 128 ||
              ExactQTileCount == 256 || ExactQTileCount == 512))
    );
    static_assert(
        !UseFusedExp2Pack ||
            (UseBitwisePExpansion && UseFastExp2 &&
             UseIterationCausalMask &&
             (ExactQTileCount == 128 || ExactQTileCount == 256 ||
              ExactQTileCount == 512))
    );
    static_assert(
        ExactQTileCount == 0 ||
            (ExactQTileCount > 0 && ExactQTileCount % 2 == 0)
    );
    __shared__ alignas(1024)
        cta2_fused_dense_role_split_shared_storage<
            UseRoleStatsCache,
            PipelineLsePrefetch,
            UseX32TmemComputeLayout || UseLongSeqStatsCache
        > storage;
    __shared__ alignas(16) semaphore k_local_ready;
    __shared__ alignas(16) semaphore k_ready;
    __shared__ alignas(16) semaphore qdo_ready;
    __shared__ alignas(16) semaphore qdo_load_done;
    __shared__ alignas(16) semaphore qdo_prefetch_done;
    __shared__ alignas(16) semaphore qdo_exchange_done;
    __shared__ alignas(16) semaphore score_done;
    __shared__ alignas(16) semaphore score_consumed;
    __shared__ alignas(16) semaphore score_consumed_local;
    __shared__ alignas(16) semaphore v_local_ready;
    __shared__ alignas(16) semaphore dp_operands_ready;
    __shared__ alignas(16) semaphore dv_operands_ready;
    __shared__ alignas(16) semaphore dp_done;
    __shared__ alignas(16) semaphore p_tmem_ready;
    __shared__ alignas(16) semaphore q_source_loaded;
    __shared__ alignas(16) semaphore do_source_loaded;
    __shared__ alignas(16) semaphore dkdv_local_ready;
    __shared__ alignas(16) semaphore dkdv_operands_ready;
    __shared__ alignas(16) semaphore dkdv_done;
    __shared__ alignas(16) semaphore dq_local_ready;
    __shared__ alignas(16) semaphore dq_exchange_done;
    __shared__ alignas(16) semaphore dq_operands_ready;
    __shared__ alignas(16) semaphore dq_done;
    __shared__ alignas(16) semaphore dq_reduce_done;
    __shared__ alignas(16) semaphore dq_shared_read_done;
    __shared__ alignas(16) semaphore owner_output_loaded;
    auto &dq_peer_ready = *reinterpret_cast<semaphore *>(
        &storage.phase.dq.operands.a
    );
    auto &dk_done = [&]() -> semaphore & {
        if constexpr (
            UseX32TmemComputeLayout || UseLongSeqStatsCache ||
            (UseWarpStatsCache && PipelineLsePrefetch)
        ) {
            return storage.dk_done;
        } else {
            return *reinterpret_cast<semaphore *>(&storage.stats.dpsum);
        }
    }();

    const int physical_warp = warpid();
    const int lane = laneid();
    const int cta_rank = UseExactClusterCoordinates
        ? (static_cast<int>(blockIdx.x) & 1)
        : cluster_ctarank();
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int owner_pair_idx = UseExactClusterCoordinates
        ? (static_cast<int>(blockIdx.x) >> 1)
        : static_cast<int>(clusterIdx().x);
    const int q_tile_count = ExactQTileCount == 0
        ? g.seq_len / 128
        : ExactQTileCount;
    constexpr int q_tile_step = OwnerQWorkSplitId < 0 ? 1 : 2;
    const int owner_count = q_tile_count / 2;
    const bool is_compute = physical_warp >= 4 && physical_warp < 12;
    const int compute_warp = physical_warp & 7;

    if (physical_warp == 13 && lane == 0) {
        if constexpr (UseTmaScoreK) {
            g.k.template prefetch_tma<cta2_fused_dense_k_tile, dim::DEPTH>();
        }
        if constexpr (UseTmaVWithScoreK) {
            g.v.template prefetch_tma<cta2_fused_dense_v_tile, dim::DEPTH>();
        }
        g.q.template prefetch_tma<cta2_fused_dense_q_tile, dim::DEPTH>();
        if constexpr (DirectTmaDkQ) {
            if constexpr (UseWideDkN192) {
                g.q.template prefetch_tma<
                    cta2_fused_dense_q_normal_wide_tile,
                    dim::DEPTH
                >();
            } else {
                g.q.template prefetch_tma<
                    cta2_fused_dense_q_normal0_tile,
                    dim::DEPTH
                >();
                g.q.template prefetch_tma<
                    cta2_fused_dense_q_normal1_tile,
                    dim::DEPTH
                >();
            }
        }
        g.dout.template prefetch_tma<cta2_fused_dense_do_tile, dim::DEPTH>();
        init_semaphore(k_local_ready, 0, UseTmaScoreK ? 1 : 8);
        init_semaphore(k_ready, 0, 2);
        init_semaphore(qdo_ready, 0, 2);
        init_semaphore(qdo_load_done, 0, 1);
        init_semaphore(qdo_prefetch_done, 0, 1);
        init_semaphore(qdo_exchange_done, 0, 1);
        init_semaphore(score_done, 0, 1);
        init_semaphore(
            score_consumed,
            0,
            SplitDpTmemConsumerRelease
                ? 4
                : (AggregateScoreConsumed ? 2 : 16)
        );
        if constexpr (AggregateScoreConsumed) {
            init_semaphore(score_consumed_local, 0, 8);
        }
        init_semaphore(v_local_ready, 0, 4);
        init_semaphore(
            dp_operands_ready,
            0,
            MergeScoreDpReady ? 4 : 2
        );
        if constexpr (SplitDvDkReady && !SingleOwnerCluster) {
            init_semaphore(dv_operands_ready, 0, 2);
        }
        init_semaphore(dp_done, 0, 1);
        if constexpr (UseX32TmemComputeLayout) {
            init_semaphore(p_tmem_ready, 0, 8);
        }
        init_semaphore(q_source_loaded, 0, DirectTmaDkQ ? 1 : 4);
        init_semaphore(do_source_loaded, 0, BulkDoDvStage ? 1 : 8);
        if constexpr (!UseNamedDkdvLocalFanIn) {
            init_semaphore(dkdv_local_ready, 0, 12);
        }
        init_semaphore(dkdv_operands_ready, 0, 2);
        init_semaphore(dkdv_done, 0, 1);
        if constexpr (PrefetchNextQdoAfterDkdv) {
            init_semaphore(dk_done, 0, 1);
        }
        init_semaphore(dq_local_ready, 0, 8);
        init_semaphore(dq_exchange_done, 0, 1);
        init_semaphore(dq_operands_ready, 0, 2);
        init_semaphore(dq_done, 0, 1);
        init_semaphore(dq_reduce_done, 0, 4);
        if constexpr (SplitDqTmemAndSharedHandoff) {
            init_semaphore(dq_shared_read_done, 0, 4);
        }
        init_semaphore(owner_output_loaded, 0, 2);
        if constexpr (StageDqPeerBeforeDv) {
            init_semaphore(dq_peer_ready, 0, 8);
        }
        if constexpr (UseX32TmemComputeLayout) {
            init_semaphore(storage.stats_consumed, 0, 8);
        } else if constexpr (UseLongSeqStatsCache) {
        } else if constexpr (UseRoleStatsCache) {
            init_semaphore(storage.stats_ready, 0, 1);
            init_semaphore(storage.stats_consumed, 0, 8);
        }
    }
    __syncthreads();
    everyone::tma::cluster::sync();

    tensor_allocator<1, 2> tm_alloc{};
    cta2_fused_dense_attn_tt dk_main_tmem =
        tm_alloc.template allocate<cta2_fused_dense_attn_tt>(0);
    cta2_fused_dense_dk_tail_tt dk_tail_tmem =
        tm_alloc.template allocate<cta2_fused_dense_dk_tail_tt>(128);
    cta2_fused_dense_dk_wide_tt dk_wide_tmem(dk_main_tmem.addr);
    cta2_fused_dense_attn_tt dv_tmem =
        tm_alloc.template allocate<cta2_fused_dense_attn_tt>(192);
    cta2_fused_dense_attn_tt score_dp_tmem =
        tm_alloc.template allocate<cta2_fused_dense_attn_tt>(320);
    cta2_fused_dense_ds_tt p_tmem =
        tm_alloc.template allocate<cta2_fused_dense_ds_tt>(
            UseFusedTmemPAndDs ? 448 : 320
        );
    cta2_fused_dense_attn_tt dp_tmem =
        tm_alloc.template allocate<cta2_fused_dense_attn_tt>(384);
    cta2_fused_dense_ds_tt ds_tmem =
        tm_alloc.template allocate<cta2_fused_dense_ds_tt>(
            UseFusedTmemPAndDs ? 320 : 384
        );
    cta2_fused_dense_dq_tt dq_tmem =
        tm_alloc.template allocate<cta2_fused_dense_dq_tt>(0, 320);

    if (physical_warp < 4) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;" ::: "memory");
        #pragma unroll 1
        for (
            int owner_pass = 0;
            owner_pass < (SingleOwnerCluster ? 1 : 2);
            ++owner_pass
        ) {
            const int owner_idx = SingleOwnerCluster
                ? owner_pair_idx
                : (owner_pass == 0
                    ? owner_pair_idx
                    : owner_count - 1 - owner_pair_idx);
            const int owner_phase = OwnerQWorkSplitId >= 0
                ? (owner_pass == 0
                    ? 0
                    : ((owner_count - owner_pair_idx) & 1))
                : (SingleOwnerCluster || IntegrateCausalFrontier
                    ? 0
                    : (owner_pass & 1));
            const int owner_once_phase =
                SingleOwnerCluster ? 0 : (owner_pass & 1);
            const int first_dense_q_tile =
                2 * owner_idx + (IntegrateCausalFrontier ? 0 : 1) +
                (OwnerQWorkSplitId < 0 ? 0 : OwnerQWorkSplitId);
            rt_bf<32, kB300VDim> v_persistent;
            if constexpr (!UseTmaVWithScoreK) {
                warp::load<dim::DEPTH>(
                    v_persistent,
                    g.v,
                    {
                        batch_idx,
                        owner_idx * 8 + cta_rank * 4 + physical_warp,
                        head_idx,
                        0
                    }
                );
            }
            if constexpr (PublishVOncePerOwner && !UseTmaVWithScoreK) {
                auto v_smem = storage.ds.template subtile<32, kB300VDim>(
                    {physical_warp, 0}
                );
                warp::store(v_smem, v_persistent);
                warp::arrive(v_local_ready);
                if (physical_warp == 0 && lane == 0) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        v_local_ready,
                        owner_once_phase
                    );
                }
            }
            int iteration = 0;
            int reducer_score_phase = owner_phase;
            for (
                int q_tile_idx = first_dense_q_tile;
                q_tile_idx < q_tile_count;
                q_tile_idx += q_tile_step, ++iteration
            ) {
                const int phase = CarryAllRolePhases
                    ? reducer_score_phase
                    : ((iteration + owner_phase) & 1);
                if constexpr (UseComputeScoreFanout) {
                    if (physical_warp == 0) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            score_done,
                            phase
                        );
                        if constexpr (!UseStatsWarpScoreFanout) {
                            if constexpr (UseX32TmemComputeLayout) {
                                asm volatile(
                                    "bar.sync 5, 320;"
                                    ::: "memory"
                                );
                            } else {
                                asm volatile(
                                    "bar.sync 5, 288;"
                                    ::: "memory"
                                );
                            }
                        }
                    } else {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            score_done,
                            phase
                        );
                    }
                } else {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        score_done,
                        phase
                    );
                }
                auto &main = storage.phase.main;
                if constexpr (DirectTmaDkQ && !LoaderOwnedDkQ) {
                    if (physical_warp == 0) {
                        if constexpr (UseWideDkN192) {
                            if constexpr (ElectedWideDkQTmaLoad) {
                                cta2_fused_dense_load_dk_q_wide_elected(
                                    main.q.normal_wide,
                                    g.q,
                                    coord<
                                        cta2_fused_dense_q_normal_wide_tile
                                    >{
                                        batch_idx,
                                        q_tile_idx,
                                        head_idx,
                                        cta_rank
                                    },
                                    q_source_loaded
                                );
                            } else if (lane == 0) {
                                tma::expect_bytes(
                                    q_source_loaded,
                                    sizeof(main.q.normal_wide)
                                );
                                tma::load_async<
                                    dim::DEPTH,
                                    cache_policy::NORMAL
                                >(
                                    main.q.normal_wide,
                                    g.q,
                                    coord<
                                        cta2_fused_dense_q_normal_wide_tile
                                    >{
                                        batch_idx,
                                        q_tile_idx,
                                        head_idx,
                                        cta_rank
                                    },
                                    q_source_loaded
                                );
                            }
                        } else if (lane == 0) {
                            tma::expect_bytes(
                                q_source_loaded,
                                sizeof(main.q.normal)
                            );
                            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                                main.q.normal.q0,
                                g.q,
                                coord<cta2_fused_dense_q_normal0_tile>{
                                    batch_idx,
                                    q_tile_idx,
                                    head_idx,
                                    cta_rank
                                },
                                q_source_loaded
                            );
                            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                                main.q.normal.q1,
                                g.q,
                                coord<cta2_fused_dense_q_normal1_tile>{
                                    batch_idx,
                                    q_tile_idx,
                                    head_idx,
                                    4 + cta_rank
                                },
                                q_source_loaded
                            );
                        }
                    }
                }
                if constexpr (
                    !PublishVOncePerOwner && !UseTmaVWithScoreK
                ) {
                    auto v_smem = storage.ds.template subtile<32, kB300VDim>(
                        {physical_warp, 0}
                    );
                    warp::store(v_smem, v_persistent);
                    warp::arrive(v_local_ready);
                }
                if (physical_warp == 0 && lane == 0) {
                    if constexpr (
                        !PublishVOncePerOwner && !UseTmaVWithScoreK
                    ) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            v_local_ready,
                            phase
                        );
                    }
                    ::kittens::tma::cluster::arrive(
                        dp_operands_ready,
                        0
                    );
                }
                auto &qdo_exchange = storage.qdo_phase.qdo;
                if constexpr (!DirectTmaDkQ) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        qdo_exchange_done,
                        phase
                    );
                    if constexpr (UseWideDkN192) {
                        rt_bf<32, 16> q0_transposed[2];
                        rt_bf<32, 16> q1_transposed[2];
                        rt_bf<32, 16> q2_transposed[2];
                        cta2_role_split_load_q_transposed_wide(
                            main,
                            qdo_exchange,
                            physical_warp,
                            cta_rank,
                            q0_transposed[0],
                            q1_transposed[0],
                            q2_transposed[0]
                        );
                        cta2_role_split_load_q_transposed_wide(
                            main,
                            qdo_exchange,
                            physical_warp + 4,
                            cta_rank,
                            q0_transposed[1],
                            q1_transposed[1],
                            q2_transposed[1]
                        );
                        warp::arrive(q_source_loaded);
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            q_source_loaded,
                            phase
                        );
                        #pragma unroll
                        for (int source = 0; source < 2; ++source) {
                            const int col_tile = physical_warp + 4 * source;
                            cta2_role_split_store_q_transposed_wide(
                                main.q.transposed_wide,
                                q0_transposed[source],
                                0,
                                col_tile
                            );
                            cta2_role_split_store_q_transposed_wide(
                                main.q.transposed_wide,
                                q1_transposed[source],
                                2,
                                col_tile
                            );
                            cta2_role_split_store_q_transposed_wide(
                                main.q.transposed_wide,
                                q2_transposed[source],
                                4,
                                col_tile
                            );
                        }
                    } else {
                        rt_bf<64, 16> q0_transposed_0;
                        rt_bf<64, 16> q0_transposed_1;
                        rt_bf<32, 16> q1_transposed_0;
                        rt_bf<32, 16> q1_transposed_1;
                        cta2_role_split_load_q_transposed(
                            main,
                            qdo_exchange,
                            physical_warp,
                            cta_rank,
                            q0_transposed_0,
                            q1_transposed_0
                        );
                        cta2_role_split_load_q_transposed(
                            main,
                            qdo_exchange,
                            physical_warp + 4,
                            cta_rank,
                            q0_transposed_1,
                            q1_transposed_1
                        );
                        warp::arrive(q_source_loaded);
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            q_source_loaded,
                            phase
                        );
                        auto q0_smem_0 =
                            main.q.transposed.q0.template subtile<64, 16>(
                                {0, physical_warp}
                            );
                        auto q0_smem_1 =
                            main.q.transposed.q0.template subtile<64, 16>(
                                {0, physical_warp + 4}
                            );
                        auto q1_smem_0 =
                            main.q.transposed.q1.template subtile<32, 16>(
                                {0, physical_warp}
                            );
                        auto q1_smem_1 =
                            main.q.transposed.q1.template subtile<32, 16>(
                                {0, physical_warp + 4}
                            );
                        warp::store(q0_smem_0, q0_transposed_0);
                        warp::store(q0_smem_1, q0_transposed_1);
                        warp::store(q1_smem_0, q1_transposed_0);
                        warp::store(q1_smem_1, q1_transposed_1);
                    }
                } else {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        q_source_loaded,
                        phase
                    );
                }
                asm volatile(
                    "fence.proxy.async.shared::cta;"
                    ::: "memory"
                );
                if constexpr (UseNamedDkdvLocalFanIn) {
                    asm volatile("bar.arrive 7, 384;" ::: "memory");
                } else {
                    warp::arrive(dkdv_local_ready);
                }
                if constexpr (UseReducerDqFanout) {
                    if (physical_warp == 0) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            dq_done,
                            phase
                        );
                    }
                    if constexpr (UseReducerDqLeaderArrive) {
                        if (physical_warp == 0) {
                            asm volatile(
                                "bar.arrive 6, 128;"
                                ::: "memory"
                            );
                        } else {
                            asm volatile(
                                "bar.sync 6, 128;"
                                ::: "memory"
                            );
                        }
                    } else {
                        asm volatile(
                            "bar.sync 6, 128;"
                            ::: "memory"
                        );
                    }
                } else {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        dq_done,
                        phase
                    );
                }
                auto &dq_reduce = [&]() -> cta2_fused_dense_dq_reduce & {
                    if constexpr (DirectNextQdoDuringDqDrain) {
                        return *reinterpret_cast<cta2_fused_dense_dq_reduce *>(
                            &storage.phase.main.p
                        );
                    } else {
                        return storage.phase.dq.reduce;
                    }
                }();
                const float dq_fragment_scale =
                    UseSlicedFp32PForDs ? 1.0f : g.scale;
                    const int row_half = physical_warp & 1;
                    const int col_half = physical_warp >> 1;
                    const int q_row_block =
                        q_tile_idx * 4 + cta_rank * 2 + row_half;
                    const uint32_t physical_row =
                        static_cast<uint32_t>(physical_warp * 32);
                    uint32_t dq_stage_lane_pointer_even = 0;
                    uint32_t dq_stage_lane_pointer_odd = 0;
                    if constexpr (CacheDqStageLanePointers) {
                        auto &stage_even =
                            dq_reduce.stage[0][physical_warp];
                        auto &stage_odd =
                            dq_reduce.stage[1][physical_warp];
                        dq_stage_lane_pointer_even = static_cast<uint32_t>(
                            __cvta_generic_to_shared(
                                stage_even.data +
                                    lane *
                                        cta2_fused_dense_dq_stage::cols
                            )
                        );
                        dq_stage_lane_pointer_odd = static_cast<uint32_t>(
                            __cvta_generic_to_shared(
                                stage_odd.data +
                                    lane *
                                        cta2_fused_dense_dq_stage::cols
                            )
                        );
                        // Each 32-float row is 128-byte aligned, so the
                        // 128-byte swizzle selector is constant across all
                        // eight float4 stores.
                        dq_stage_lane_pointer_even ^=
                            (dq_stage_lane_pointer_even & 0x380u) >> 3;
                        dq_stage_lane_pointer_odd ^=
                            (dq_stage_lane_pointer_odd & 0x380u) >> 3;
                        asm volatile(
                            ""
                            : "+r"(dq_stage_lane_pointer_even),
                              "+r"(dq_stage_lane_pointer_odd)
                        );
                    }
                    if constexpr (UseBatchedDqTmemLoads) {
                        uint32_t values0[32];
                        uint32_t values1[32];
                        uint32_t values2[32];
                        const uint32_t dq_row_address =
                            dq_tmem.addr + (physical_row << 16);
                        cta2_fused_dense_load_tmem_x32(
                            values0,
                            dq_row_address
                        );
                        cta2_fused_dense_load_tmem_x32(
                            values1,
                            dq_row_address + 32
                        );
                        cta2_fused_dense_load_tmem_x32(
                            values2,
                            dq_row_address + 64
                        );
                        // Match CuTe's critical lifetime: release the
                        // score/dQ TMEM alias immediately after all three
                        // fragments have reached registers, while the shared
                        // stores and global reductions continue independently.
                        warp::arrive(dq_reduce_done);
                        cta2_fused_dense_store_dq_chunk_cached_unscaled<0>(
                            g,
                            dq_reduce.stage[0][physical_warp],
                            dq_stage_lane_pointer_even,
                            values0,
                            batch_idx,
                            q_row_block,
                            head_idx,
                            col_half
                        );
                        cta2_fused_dense_store_dq_chunk_cached_unscaled<1>(
                            g,
                            dq_reduce.stage[1][physical_warp],
                            dq_stage_lane_pointer_odd,
                            values1,
                            batch_idx,
                            q_row_block,
                            head_idx,
                            col_half
                        );
                        warp::tma::store_async_wait<1>();
                        cta2_fused_dense_store_dq_chunk_cached_unscaled<2>(
                            g,
                            dq_reduce.stage[0][physical_warp],
                            dq_stage_lane_pointer_even,
                            values2,
                            batch_idx,
                            q_row_block,
                            head_idx,
                            col_half
                        );
                        warp::tma::store_async_read_wait();
                        warp::arrive(dq_shared_read_done);
                        warp::tma::store_async_wait();
                    } else {
                        #pragma unroll 1
                        for (int chunk = 0; chunk < 3; ++chunk) {
                            uint32_t values[32];
                            cta2_fused_dense_load_tmem_x32(
                                values,
                                dq_tmem.addr +
                                    (physical_row << 16) + chunk * 32
                            );
                            if (chunk >= 2) {
                                if constexpr (SplitDqTmemAndSharedHandoff) {
                                    warp::arrive(dq_reduce_done);
                                }
                                warp::tma::store_async_wait<1>();
                            }
                            auto &stage =
                                dq_reduce.stage[chunk & 1][physical_warp];
                            #pragma unroll
                            for (int col = 0; col < 32; col += 4) {
                                if constexpr (CacheDqStageLanePointers) {
                                    const uint32_t lane_row =
                                        (chunk & 1)
                                            ? dq_stage_lane_pointer_odd
                                            : dq_stage_lane_pointer_even;
                                    move<float4>::sts(
                                        lane_row ^
                                            static_cast<uint32_t>(
                                                col * sizeof(float)
                                            ),
                                        make_float4(
                                            __uint_as_float(values[col + 0]) *
                                                dq_fragment_scale,
                                            __uint_as_float(values[col + 1]) *
                                                dq_fragment_scale,
                                            __uint_as_float(values[col + 2]) *
                                                dq_fragment_scale,
                                            __uint_as_float(values[col + 3]) *
                                                dq_fragment_scale
                                        )
                                    );
                                } else {
                                    *reinterpret_cast<float4 *>(
                                        &stage[{lane, col}]
                                    ) = make_float4(
                                        __uint_as_float(values[col + 0]) *
                                            dq_fragment_scale,
                                        __uint_as_float(values[col + 1]) *
                                            dq_fragment_scale,
                                        __uint_as_float(values[col + 2]) *
                                            dq_fragment_scale,
                                        __uint_as_float(values[col + 3]) *
                                            dq_fragment_scale
                                    );
                                }
                            }
                            __syncwarp();
                            warp::tma::store_add_async<
                                dim::DEPTH,
                                cache_policy::NORMAL
                            >(
                                g.dq,
                                stage,
                                {
                                    batch_idx,
                                    q_row_block,
                                    head_idx,
                                    col_half * 3 + chunk
                                }
                            );
                        }
                        if constexpr (DqReadHandoffBeforeCompletion) {
                            warp::tma::store_async_read_wait();
                            if constexpr (SplitDqTmemAndSharedHandoff) {
                                warp::arrive(dq_shared_read_done);
                            } else {
                                warp::arrive(dq_reduce_done);
                            }
                            warp::tma::store_async_wait();
                        } else {
                            warp::tma::store_async_wait();
                            warp::arrive(dq_reduce_done);
                        }
                    }
                if constexpr (CarryAllRolePhases) {
                    reducer_score_phase ^= 1;
                }
            }
            if (first_dense_q_tile < q_tile_count) {
                const int kv_tile = owner_idx * 2 + cta_rank;
                #pragma unroll
                for (int chunk = 0; chunk < 2; ++chunk) {
                    rt_fl<32, 64> output_chunk;
                    auto dv_chunk = dv_tmem.template subtile<
                        cta2_fused_dense_dk_tail_tt
                    >(chunk * 64);
                    group<4>::load_async(output_chunk, dv_chunk);
                    tensor_load_wait();
                    if constexpr (WideCoalescedBf16Store) {
                        cta2_fused_dense_store_bf16_coalesced_wide(
                            g.dv,
                            output_chunk,
                            batch_idx,
                            kv_tile,
                            head_idx,
                            chunk
                        );
                    } else if constexpr (CoalescedBf16Store) {
                        cta2_fused_dense_store_bf16_coalesced(
                            g.dv,
                            output_chunk,
                            batch_idx,
                            kv_tile,
                            head_idx,
                            chunk
                        );
                    } else {
                        group<4>::store<dim::DEPTH>(
                            g.dv,
                            output_chunk,
                            {batch_idx, kv_tile, head_idx, chunk}
                        );
                    }
                }
                #pragma unroll
                for (int chunk = 0; chunk < 2; ++chunk) {
                    rt_fl<32, 64> output_chunk;
                    auto dk_chunk = dk_main_tmem.template subtile<
                        cta2_fused_dense_dk_tail_tt
                    >(chunk * 64);
                    group<4>::load_async(output_chunk, dk_chunk);
                    tensor_load_wait();
                    warp::mul(output_chunk, output_chunk, g.scale);
                    if constexpr (WideCoalescedBf16Store) {
                        cta2_fused_dense_store_bf16_coalesced_wide(
                            g.dk,
                            output_chunk,
                            batch_idx,
                            kv_tile,
                            head_idx,
                            chunk
                        );
                    } else if constexpr (CoalescedBf16Store) {
                        cta2_fused_dense_store_bf16_coalesced(
                            g.dk,
                            output_chunk,
                            batch_idx,
                            kv_tile,
                            head_idx,
                            chunk
                        );
                    } else {
                        group<4>::store<dim::DEPTH>(
                            g.dk,
                            output_chunk,
                            {batch_idx, kv_tile, head_idx, chunk}
                        );
                    }
                }
                {
                    rt_fl<32, 64> output_chunk;
                    if constexpr (ReverseDkTailTmemLoadIssue) {
                        auto dk_tail_warp_tmem =
                            dk_tail_tmem.template subtile<tt_fl<32, 64>>(
                                32 * warpid(),
                                0
                            );
                        auto dk_tail_hi =
                            dk_tail_warp_tmem.template subtile<tt_fl<16, 64>>(
                                16,
                                0
                            );
                        auto dk_tail_lo =
                            dk_tail_warp_tmem.template subtile<tt_fl<16, 64>>(
                                0,
                                0
                            );
                        auto &output_hi =
                            group<1>::subtile_inplace<16>(output_chunk, 1);
                        auto &output_lo =
                            group<1>::subtile_inplace<16>(output_chunk, 0);
                        group<1>::load_async(output_hi, dk_tail_hi);
                        group<1>::load_async(output_lo, dk_tail_lo);
                    } else {
                        group<4>::load_async(output_chunk, dk_tail_tmem);
                    }
                    tensor_load_wait();
                    warp::mul(output_chunk, output_chunk, g.scale);
                    if constexpr (WideCoalescedBf16Store) {
                        cta2_fused_dense_store_bf16_coalesced_wide(
                            g.dk,
                            output_chunk,
                            batch_idx,
                            kv_tile,
                            head_idx,
                            2
                        );
                    } else if constexpr (CoalescedBf16Store) {
                        cta2_fused_dense_store_bf16_coalesced(
                            g.dk,
                            output_chunk,
                            batch_idx,
                            kv_tile,
                            head_idx,
                            2
                        );
                    } else {
                        group<4>::store<dim::DEPTH>(
                            g.dk,
                            output_chunk,
                            {batch_idx, kv_tile, head_idx, 2}
                        );
                    }
                }
                if constexpr (EnsureReducerOutputDrain) {
                    asm volatile("bar.sync 8, 128;" ::: "memory");
                }
                if (physical_warp == 0 && lane == 0) {
                    ::kittens::tma::cluster::arrive(
                        owner_output_loaded,
                        0
                    );
                }
            }
        }
        return;
    }

    if (is_compute) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;" ::: "memory");
        const int output_subtile =
            2 * (compute_warp % 4) + compute_warp / 4;
        uint32_t dv_operands_ready_cluster_address = 0;
        uint32_t dkdv_operands_ready_cluster_address = 0;
        if constexpr (
            CacheRoleClusterAddresses && !UseX32TmemComputeLayout
        ) {
            if (physical_warp == 4 && lane == 0) {
                dv_operands_ready_cluster_address =
                    cta2_fused_dense_map_cluster_semaphore(
                        dv_operands_ready,
                        0
                    );
                dkdv_operands_ready_cluster_address =
                    cta2_fused_dense_map_cluster_semaphore(
                        dkdv_operands_ready,
                        0
                );
            }
        }
        int compute_stats_iteration = 0;
        bool have_previous_dq_shared_read = false;
        int previous_dq_shared_read_phase = 0;
        #pragma unroll 1
        for (
            int owner_pass = 0;
            owner_pass < (SingleOwnerCluster ? 1 : 2);
            ++owner_pass
        ) {
            const int owner_idx = SingleOwnerCluster
                ? owner_pair_idx
                : (owner_pass == 0
                    ? owner_pair_idx
                    : owner_count - 1 - owner_pair_idx);
            const int owner_phase = OwnerQWorkSplitId >= 0
                ? (owner_pass == 0
                    ? 0
                    : ((owner_count - owner_pair_idx) & 1))
                : (SingleOwnerCluster || IntegrateCausalFrontier
                    ? 0
                    : (owner_pass & 1));
            const int owner_once_phase =
                SingleOwnerCluster ? 0 : (owner_pass & 1);
            const int first_dense_q_tile =
                2 * owner_idx + (IntegrateCausalFrontier ? 0 : 1) +
                (OwnerQWorkSplitId < 0 ? 0 : OwnerQWorkSplitId);
            size_t carried_direct_stats_offset = 0;
            if constexpr (CarryDirectStatsOffset) {
                carried_direct_stats_offset =
                    (static_cast<size_t>(batch_idx) *
                            g.lse_log2.depth() +
                        head_idx) *
                        g.seq_len +
                    first_dense_q_tile * 128;
            }
            if constexpr (UseTmaScoreK) {
                if constexpr (ElectedScoreKTmaLoad) {
                    if (physical_warp == 4) {
                        coord<cta2_fused_dense_k_tile> k_tile_coord = {
                            batch_idx,
                            owner_idx * 2 + cta_rank,
                            head_idx,
                            0
                        };
                        if constexpr (UseTmaVWithScoreK) {
                            coord<cta2_fused_dense_v_tile> v_tile_coord = {
                                batch_idx,
                                owner_idx * 2 + cta_rank,
                                head_idx,
                                0
                            };
                            cta2_fused_dense_load_score_k_v_elected(
                                storage.k,
                                g.k,
                                k_tile_coord,
                                storage.ds,
                                g.v,
                                v_tile_coord,
                                k_local_ready
                            );
                        } else {
                            cta2_fused_dense_load_score_k_elected(
                                storage.k,
                                g.k,
                                k_tile_coord,
                                k_local_ready
                            );
                        }
                    }
                } else if (physical_warp == 4 && lane == 0) {
                    tma::expect_bytes(k_local_ready, sizeof(storage.k));
                    coord<cta2_fused_dense_k_tile> k_tile_coord = {
                        batch_idx,
                        owner_idx * 2 + cta_rank,
                        head_idx,
                        0
                    };
                    tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                        storage.k,
                        g.k,
                        k_tile_coord,
                        k_local_ready
                    );
                }
            } else {
                rt_bf<16, kB300QKDim> k_reg;
                warp::load<dim::DEPTH>(
                    k_reg,
                    g.k,
                    {
                        batch_idx,
                        owner_idx * 16 + cta_rank * 8 + compute_warp,
                        head_idx,
                        0
                    }
                );
                auto k_smem = storage.k.template subtile<16, kB300QKDim>(
                    {compute_warp, 0}
                );
                warp::store(k_smem, k_reg);
            }
            if constexpr (WideDqKGlobalToShared) {
                const coord<cta2_fused_dense_dq_b_tile> dq_k_coord = {
                    batch_idx,
                    owner_idx,
                    head_idx,
                    cta_rank
                };
                if constexpr (UseSlicedFp32PForDs) {
                    cta2_fused_dense_load_scaled_dq_k(
                        storage.dq_b,
                        g.k,
                        dq_k_coord,
                        g.scale
                    );
                } else {
                    group<8>::load<dim::DEPTH, true>(
                        storage.dq_b,
                        g.k,
                        dq_k_coord
                    );
                }
            } else {
                rt_bf<32, 96> k_reg;
                warp::load<dim::DEPTH>(
                    k_reg,
                    g.k,
                    {
                        batch_idx,
                        owner_idx * 8 + compute_warp,
                        head_idx,
                        cta_rank
                    }
                );
                auto k_smem = storage.dq_b.template subtile<32, 96>(
                    {compute_warp, 0}
                );
                warp::store(k_smem, k_reg);
            }
            if constexpr (UseTmaScoreK) {
                if (physical_warp == 4) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        k_local_ready,
                        owner_once_phase
                    );
                }
            } else {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                warp::arrive(k_local_ready);
            }
            if (physical_warp == 4 && lane == 0) {
                if constexpr (!UseTmaScoreK) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        k_local_ready,
                        owner_once_phase
                    );
                }
                ::kittens::tma::cluster::arrive(k_ready, 0);
            }

            int iteration = 0;
            int compute_phase = owner_phase;
            for (
                int q_tile_idx = first_dense_q_tile;
                q_tile_idx < q_tile_count;
                q_tile_idx += q_tile_step,
                ++iteration,
                ++compute_stats_iteration
            ) {
                const int phase = CarryAllRolePhases
                    ? compute_phase
                    : ((iteration + owner_phase) & 1);
                const int causal_iteration = OwnerQWorkSplitId < 0
                    ? iteration
                    : q_tile_idx - 2 * owner_idx;
                auto &main = storage.phase.main;
                cta2_fused_dense_stats_vec lse_vec;
                float lse_x32_stage0[32];
                const int x32_column_group =
                    compute_warp < 4 ? 32 : 0;
                if constexpr (UseX32TmemComputeLayout) {
                } else if constexpr (UseLongSeqStatsCache) {
                    const size_t stats_offset =
                        CarryDirectStatsOffset
                            ? carried_direct_stats_offset
                            : (static_cast<size_t>(batch_idx) *
                                    g.lse_log2.depth() +
                                head_idx) *
                                    g.seq_len +
                                q_tile_idx * 128;
                    cta2_role_split_load_stats_global_direct(
                        lse_vec,
                        g.lse_log2.raw_ptr + stats_offset
                    );
                } else if constexpr (UseRoleStatsCache) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        storage.stats_ready,
                        phase
                    );
                    if constexpr (PipelineLsePrefetch) {
                        auto &lse_stage = phase == 0
                            ? storage.stats.lse_log2
                            : storage.lse_log2_next;
                        if constexpr (UseDirectStatsLoads) {
                            cta2_role_split_load_stats_direct(
                                lse_vec,
                                lse_stage
                            );
                        } else {
                            warp::load(lse_vec, lse_stage);
                        }
                    } else {
                        warp::load(lse_vec, storage.stats.lse_log2);
                    }
                } else {
                    if constexpr (UseDirectStatsLoads) {
                        const size_t stats_offset =
                            CarryDirectStatsOffset
                                ? carried_direct_stats_offset
                                : (static_cast<size_t>(batch_idx) *
                                        g.lse_log2.depth() +
                                    head_idx) *
                                        g.seq_len +
                                    q_tile_idx * 128;
                        cta2_role_split_load_stats_global_direct(
                            lse_vec,
                            g.lse_log2.raw_ptr + stats_offset
                        );
                    } else {
                        warp::load(
                            lse_vec,
                            g.lse_log2,
                            {batch_idx, head_idx, 0, q_tile_idx}
                        );
                    }
                }
                if constexpr (UseComputeScoreFanout) {
                    if constexpr (UseX32TmemComputeLayout) {
                        asm volatile("bar.sync 5, 320;" ::: "memory");
                    } else {
                        asm volatile("bar.sync 5, 288;" ::: "memory");
                    }
                } else {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        score_done,
                        phase
                    );
                }
                if constexpr (UseX32TmemComputeLayout) {
                    cta2_role_split_load_stats_x32_stage_shared(
                        lse_x32_stage0,
                        storage.stats.lse_log2.data +
                            x32_column_group
                    );
                }
                cta2_fused_dense_attn_bf_reg p_bf;
                cta2_fused_dense_attn_reg p_reg;
                uint32_t p_x32_stage0[16];
                uint32_t p_x32_stage1[16];
                if constexpr (UseX32TmemComputeLayout) {
                    const uint32_t physical_row =
                        static_cast<uint32_t>(
                            32 * (compute_warp & 3)
                        );
                    const uint32_t score_row_address =
                        score_dp_tmem.addr + (physical_row << 16);
                    const uint32_t p_row_address =
                        p_tmem.addr + (physical_row << 16);
                    const float score_scale_log2e =
                        UseExactDefaultScaleLog2e
                            ? 0x1.aa7728p-4f
                            : g.scale_log2e;
                    const int key_row =
                        static_cast<int>(physical_row) + lane;
                    uint32_t score_x32_stage0[32];
                    uint32_t score_x32_stage1[32];
                    cta2_fused_dense_load_tmem_x32(
                        score_x32_stage0,
                        score_row_address + x32_column_group
                    );
                    cta2_fused_dense_load_tmem_x32(
                        score_x32_stage1,
                        score_row_address + x32_column_group + 64
                    );
                    tensor_load_wait();
                    if constexpr (AggregateScoreConsumed) {
                        warp::arrive(score_consumed_local);
                        if (compute_warp == 0) {
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                score_consumed_local,
                                phase
                            );
                            if (lane == 0) {
                                if constexpr (MergeScoreDpReady) {
                                    ::kittens::tma::cluster::arrive(
                                        dp_operands_ready,
                                        0
                                    );
                                } else {
                                    ::kittens::tma::cluster::arrive(
                                        score_consumed,
                                        0
                                    );
                                    ::kittens::tma::cluster::arrive(
                                        score_consumed,
                                        1
                                    );
                                }
                            }
                        }
                    } else if (lane == 0) {
                        ::kittens::tma::cluster::arrive(
                            score_consumed,
                            0
                        );
                        ::kittens::tma::cluster::arrive(
                            score_consumed,
                            1
                        );
                    }
                    if (iteration >= 2) {
                        if (compute_warp < 4) {
                            cta2_role_split_make_p_x32_stage<32, false>(
                                p_x32_stage0,
                                score_x32_stage0,
                                lse_x32_stage0,
                                score_scale_log2e,
                                key_row
                            );
                        } else {
                            cta2_role_split_make_p_x32_stage<0, false>(
                                p_x32_stage0,
                                score_x32_stage0,
                                lse_x32_stage0,
                                score_scale_log2e,
                                key_row
                            );
                        }
                    } else if (causal_iteration < cta_rank) {
                        #pragma unroll
                        for (int pair = 0; pair < 16; ++pair) {
                            p_x32_stage0[pair] = 0;
                        }
                    } else if (causal_iteration == cta_rank) {
                        if (compute_warp < 4) {
                            cta2_role_split_make_p_x32_stage<32, true>(
                                p_x32_stage0,
                                score_x32_stage0,
                                lse_x32_stage0,
                                score_scale_log2e,
                                key_row
                            );
                        } else {
                            cta2_role_split_make_p_x32_stage<0, true>(
                                p_x32_stage0,
                                score_x32_stage0,
                                lse_x32_stage0,
                                score_scale_log2e,
                                key_row
                            );
                        }
                    } else {
                        if (compute_warp < 4) {
                            cta2_role_split_make_p_x32_stage<32, false>(
                                p_x32_stage0,
                                score_x32_stage0,
                                lse_x32_stage0,
                                score_scale_log2e,
                                key_row
                            );
                        } else {
                            cta2_role_split_make_p_x32_stage<0, false>(
                                p_x32_stage0,
                                score_x32_stage0,
                                lse_x32_stage0,
                                score_scale_log2e,
                                key_row
                            );
                        }
                    }
                    cta2_role_split_store_tmem_x16(
                        p_row_address + x32_column_group / 2,
                        p_x32_stage0
                    );
                    float lse_x32_stage1[32];
                    cta2_role_split_load_stats_x32_stage_shared(
                        lse_x32_stage1,
                        storage.stats.lse_log2.data +
                            x32_column_group + 64
                    );
                    if (iteration >= 2) {
                        if (compute_warp < 4) {
                            cta2_role_split_make_p_x32_stage<96, false>(
                                p_x32_stage1,
                                score_x32_stage1,
                                lse_x32_stage1,
                                score_scale_log2e,
                                key_row
                            );
                        } else {
                            cta2_role_split_make_p_x32_stage<64, false>(
                                p_x32_stage1,
                                score_x32_stage1,
                                lse_x32_stage1,
                                score_scale_log2e,
                                key_row
                            );
                        }
                    } else if (causal_iteration < cta_rank) {
                        #pragma unroll
                        for (int pair = 0; pair < 16; ++pair) {
                            p_x32_stage1[pair] = 0;
                        }
                    } else if (causal_iteration == cta_rank) {
                        if (compute_warp < 4) {
                            cta2_role_split_make_p_x32_stage<96, true>(
                                p_x32_stage1,
                                score_x32_stage1,
                                lse_x32_stage1,
                                score_scale_log2e,
                                key_row
                            );
                        } else {
                            cta2_role_split_make_p_x32_stage<64, true>(
                                p_x32_stage1,
                                score_x32_stage1,
                                lse_x32_stage1,
                                score_scale_log2e,
                                key_row
                            );
                        }
                    } else {
                        if (compute_warp < 4) {
                            cta2_role_split_make_p_x32_stage<96, false>(
                                p_x32_stage1,
                                score_x32_stage1,
                                lse_x32_stage1,
                                score_scale_log2e,
                                key_row
                            );
                        } else {
                            cta2_role_split_make_p_x32_stage<64, false>(
                                p_x32_stage1,
                                score_x32_stage1,
                                lse_x32_stage1,
                                score_scale_log2e,
                                key_row
                            );
                        }
                    }
                    cta2_role_split_store_tmem_x16(
                        p_row_address +
                            (x32_column_group + 64) / 2,
                        p_x32_stage1
                    );
                    tensor_store_wait();
                    warp::arrive(p_tmem_ready);
                } else {
                {
                    group<8>::load_async(p_reg, score_dp_tmem);
                    tensor_load_wait();
                    if constexpr (AggregateScoreConsumed) {
                        warp::arrive(score_consumed_local);
                        if (compute_warp == 0) {
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                score_consumed_local,
                                phase
                            );
                            if (lane == 0) {
                                if constexpr (MergeScoreDpReady) {
                                    ::kittens::tma::cluster::arrive(
                                        dp_operands_ready,
                                        0
                                    );
                                } else {
                                    ::kittens::tma::cluster::arrive(
                                        score_consumed,
                                        0
                                    );
                                    ::kittens::tma::cluster::arrive(
                                        score_consumed,
                                        1
                                    );
                                }
                            }
                        }
                    } else if (lane == 0) {
                        ::kittens::tma::cluster::arrive(
                            score_consumed,
                            0
                        );
                        ::kittens::tma::cluster::arrive(
                            score_consumed,
                            1
                        );
                    }
                    if constexpr (FuseScoreScaleLse) {
                        if (iteration >= 2) {
                            if constexpr (UseExactDefaultScaleLog2e) {
                                warp::mul(
                                    p_reg,
                                    p_reg,
                                    0x1.aa7728p-4f
                                );
                            } else {
                                warp::mul(p_reg, p_reg, g.scale_log2e);
                            }
                            warp::sub_col(p_reg, p_reg, lse_vec);
                            if constexpr (FusedExp2Fragment4First) {
                                cta2_role_split_exp2_approx_pack_bf16_fragment4_first<
                                    UseSlicedFp32PForDs
                                >(
                                    p_bf,
                                    p_reg
                                );
                            } else {
                                cta2_role_split_exp2_approx_pack_bf16<
                                    UseSlicedFp32PForDs
                                >(
                                    p_bf,
                                    p_reg
                                );
                            }
                        } else {
                            if constexpr (UseExactDefaultScaleLog2e) {
                                warp::mul(
                                    p_reg,
                                    p_reg,
                                    0x1.aa7728p-4f
                                );
                            } else {
                                warp::mul(p_reg, p_reg, g.scale_log2e);
                            }
                            warp::sub_col(p_reg, p_reg, lse_vec);
                            if (causal_iteration == cta_rank) {
                                cta2_role_split_apply_diagonal_causal_mask(
                                    p_reg,
                                    output_subtile
                                );
                            }
                            if (causal_iteration < cta_rank) {
                                if constexpr (UseSlicedFp32PForDs) {
                                    warp::neg_infty(p_reg);
                                } else {
                                    warp::zero(p_bf);
                                }
                            } else {
                                if constexpr (FusedExp2Fragment4First) {
                                    cta2_role_split_exp2_approx_pack_bf16_fragment4_first<
                                        UseSlicedFp32PForDs
                                    >(
                                        p_bf,
                                        p_reg
                                    );
                                } else {
                                    cta2_role_split_exp2_approx_pack_bf16<
                                        UseSlicedFp32PForDs
                                    >(
                                        p_bf,
                                        p_reg
                                    );
                                }
                            }
                        }
                    } else {
                        if constexpr (UseExactDefaultScaleLog2e) {
                            warp::mul(p_reg, p_reg, 0x1.aa7728p-4f);
                        } else {
                            warp::mul(p_reg, p_reg, g.scale_log2e);
                        }
                        if constexpr (PeelCausalPrefix) {
                            if (iteration >= 2) {
                                warp::sub_col(p_reg, p_reg, lse_vec);
                                cta2_role_split_exp2_approx_pack_bf16<
                                    UseSlicedFp32PForDs
                                >(
                                    p_bf,
                                    p_reg
                                );
                            } else {
                                if (causal_iteration == cta_rank) {
                                    cta2_role_split_apply_diagonal_causal_mask(
                                        p_reg,
                                        output_subtile
                                    );
                                }
                                warp::sub_col(p_reg, p_reg, lse_vec);
                                if (causal_iteration < cta_rank) {
                                    if constexpr (UseSlicedFp32PForDs) {
                                        warp::neg_infty(p_reg);
                                    } else {
                                        warp::zero(p_bf);
                                    }
                                } else {
                                    cta2_role_split_exp2_approx_pack_bf16<
                                        UseSlicedFp32PForDs
                                    >(
                                        p_bf,
                                        p_reg
                                    );
                                }
                            }
                        } else {
                            if constexpr (UseIterationCausalMask) {
                                if (causal_iteration == cta_rank) {
                                    cta2_role_split_apply_diagonal_causal_mask(
                                        p_reg,
                                        output_subtile
                                    );
                                }
                            } else if constexpr (IntegrateCausalFrontier) {
                                const int kv_tile_idx =
                                    2 * owner_idx + cta_rank;
                                if (q_tile_idx == kv_tile_idx) {
                                    cta2_role_split_apply_diagonal_causal_mask(
                                        p_reg,
                                        output_subtile
                                    );
                                }
                            }
                            warp::sub_col(p_reg, p_reg, lse_vec);
                            if constexpr (UseFusedExp2Pack) {
                                if (iteration < cta_rank) {
                                    if constexpr (UseSlicedFp32PForDs) {
                                        warp::neg_infty(p_reg);
                                    } else {
                                        warp::zero(p_bf);
                                    }
                                } else {
                                    cta2_role_split_exp2_approx_pack_bf16<
                                        UseSlicedFp32PForDs
                                    >(
                                        p_bf,
                                        p_reg
                                    );
                                }
                            } else {
                                if constexpr (UseFastExp2) {
                                    cta2_role_split_exp2_approx(p_reg);
                                } else {
                                    warp::exp2(p_reg, p_reg);
                                }
                                if constexpr (UseIterationCausalMask) {
                                    if (causal_iteration < cta_rank) {
                                        warp::zero(p_reg);
                                    }
                                } else if constexpr (
                                    IntegrateCausalFrontier
                                ) {
                                    if (
                                        q_tile_idx <
                                        2 * owner_idx + cta_rank
                                    ) {
                                        warp::zero(p_reg);
                                    }
                                } else if (
                                    q_tile_idx == first_dense_q_tile &&
                                    cta_rank == 1
                                ) {
                                    warp::zero(p_reg);
                                }
                                warp::copy(p_bf, p_reg);
                            }
                        }
                    }
                    if constexpr (UseFusedTmemPAndDs || !PreloadDpsum) {
                        if constexpr (UseSlicedFp32PForDs) {
                            {
                                auto &p_bf_quarter = *reinterpret_cast<
                                    cta2_fused_dense_attn_quarter_bf_reg *
                                >(&p_bf.tiles[0][0]);
                                cta2_role_split_pack_p_fp32_quarter<0>(
                                    p_reg,
                                    p_bf_quarter
                                );
                            }
                            {
                                auto &p_bf_quarter = *reinterpret_cast<
                                    cta2_fused_dense_attn_quarter_bf_reg *
                                >(&p_bf.tiles[0][2]);
                                cta2_role_split_pack_p_fp32_quarter<1>(
                                    p_reg,
                                    p_bf_quarter
                                );
                            }
                            {
                                auto &p_bf_half = *reinterpret_cast<
                                    cta2_fused_dense_attn_half_bf_reg *
                                >(&p_bf.tiles[0][0]);
                                auto p_half_tmem =
                                    p_tmem.template subtile<
                                        cta2_fused_dense_ds_half_tt
                                    >(0);
                                group<8>::store_async(
                                    p_half_tmem,
                                    p_bf_half
                                );
                            }
                            {
                                auto &p_bf_quarter = *reinterpret_cast<
                                    cta2_fused_dense_attn_quarter_bf_reg *
                                >(&p_bf.tiles[0][4]);
                                cta2_role_split_pack_p_fp32_quarter<2>(
                                    p_reg,
                                    p_bf_quarter
                                );
                            }
                            {
                                auto &p_bf_quarter = *reinterpret_cast<
                                    cta2_fused_dense_attn_quarter_bf_reg *
                                >(&p_bf.tiles[0][6]);
                                cta2_role_split_pack_p_fp32_quarter<3>(
                                    p_reg,
                                    p_bf_quarter
                                );
                            }
                            {
                                auto &p_bf_half = *reinterpret_cast<
                                    cta2_fused_dense_attn_half_bf_reg *
                                >(&p_bf.tiles[0][4]);
                                auto p_half_tmem =
                                    p_tmem.template subtile<
                                        cta2_fused_dense_ds_half_tt
                                    >(64);
                                group<8>::store_async(
                                    p_half_tmem,
                                    p_bf_half
                                );
                            }
                        } else {
                            group<8>::store_async(p_tmem, p_bf);
                        }
                        tensor_store_wait();
                    }
                    if constexpr (!UseFusedTmemPAndDs) {
                        auto p_smem = main.p.p.template subtile<16, 128>(
                            {output_subtile, 0}
                        );
                        warp::store(p_smem, p_bf);
                    }
                }
                }

                cta2_fused_dense_stats_vec dpsum_vec;
                cta2_fused_dense_quarter_stats_vec dpsum_quarter;
                cta2_fused_dense_attn_half_reg dp_half;
                float *split_direct_dpsum_src = nullptr;
                float dpsum_x32_stage0[32];
                if constexpr (PreloadDpsum) {
                    if constexpr (UseX32TmemComputeLayout) {
                        cta2_role_split_load_stats_x32_stage_shared(
                            dpsum_x32_stage0,
                            storage.stats.dpsum.data + x32_column_group
                        );
                    } else if constexpr (UseLongSeqStatsCache) {
                        const int stats_stage =
                            compute_stats_iteration & 1;
                        auto &stats = stats_stage == 0
                            ? storage.stats
                            : storage.stats_next;
                        cta2_role_split_load_stats_direct(
                            dpsum_vec,
                            stats.dpsum
                        );
                    } else if constexpr (UseRoleStatsCache) {
                        if constexpr (UseDirectStatsLoads) {
                            if constexpr (UseSlicedFp32PForDs) {
                                cta2_role_split_load_stats_direct_quarter<0>(
                                    dpsum_quarter,
                                    storage.stats.dpsum
                                );
                            } else {
                                cta2_role_split_load_stats_direct(
                                    dpsum_vec,
                                    storage.stats.dpsum
                                );
                            }
                        } else {
                            warp::load(dpsum_vec, storage.stats.dpsum);
                            warp::arrive(storage.stats_consumed);
                        }
                    } else {
                        if constexpr (UseDirectStatsLoads) {
                            const size_t stats_offset =
                                CarryDirectStatsOffset
                                    ? carried_direct_stats_offset
                                    : (static_cast<size_t>(batch_idx) *
                                            g.dpsum.depth() +
                                        head_idx) *
                                            g.seq_len +
                                        q_tile_idx * 128;
                            if constexpr (
                                SplitDirectDpsumAcrossDpDoneWait
                            ) {
                                split_direct_dpsum_src =
                                    g.dpsum.raw_ptr + stats_offset;
                                cta2_role_split_load_stats_global_direct_range<
                                    0,
                                    4
                                >(
                                    dpsum_vec,
                                    split_direct_dpsum_src
                                );
                            } else {
                                cta2_role_split_load_stats_global_direct(
                                    dpsum_vec,
                                    g.dpsum.raw_ptr + stats_offset
                                );
                            }
                        } else {
                            warp::load(
                                dpsum_vec,
                                g.dpsum,
                                {batch_idx, head_idx, 0, q_tile_idx}
                            );
                        }
                    }
                }
                cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                    dp_done,
                    phase
                );
                if constexpr (PreissueFirstDpHalfBeforeQdoWait) {
                    cta2_role_split_issue_dp_fp32_half<0>(
                        dp_half,
                        score_dp_tmem
                    );
                }
                if constexpr (
                    SplitDirectDpsumAcrossDpDoneWait &&
                    !UseX32TmemComputeLayout &&
                    !UseLongSeqStatsCache
                ) {
                    cta2_role_split_load_stats_global_direct_range<4, 8>(
                        dpsum_vec,
                        split_direct_dpsum_src
                    );
                }
                if constexpr (!PreloadDpsum) {
                    warp::load(
                        dpsum_vec,
                        g.dpsum,
                        {batch_idx, head_idx, 0, q_tile_idx}
                    );
                }
                if constexpr (SplitDvDkReady) {
                    if constexpr (BulkDoDvStage) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            qdo_exchange_done,
                            phase
                        );
                        if (physical_warp == 4 && lane == 0) {
                            constexpr uint32_t kHalfElements = 64 * 64;
                            constexpr uint32_t kHalfBytes =
                                kHalfElements * sizeof(bf16);
                            static_assert(
                                sizeof(cta2_fused_dense_do_exchange_tile) ==
                                    kHalfBytes
                            );
                            auto &do_normal =
                                *reinterpret_cast<st_bf<128, 64> *>(
                                    &main.dout
                                );
                            ::kittens::tma::cluster::expect_bytes(
                                do_source_loaded,
                                kHalfBytes,
                                cta_rank
                            );
                            cta2_role_split_store_peer_bulk_after_cta_fence(
                                reinterpret_cast<void *>(
                                    &do_normal.data[
                                        (cta_rank ^ 1) * kHalfElements
                                    ]
                                ),
                                reinterpret_cast<void *>(
                                    &storage.qdo_phase.qdo.dout
                                ),
                                kHalfBytes,
                                cta_rank,
                                do_source_loaded
                            );
                            cta2_fused_dense_role_wait<
                                TimeoutAllRoleWaits
                            >(
                                do_source_loaded,
                                phase
                            );
                            if constexpr (UseX32TmemComputeLayout) {
                                cta2_fused_dense_role_wait<
                                    TimeoutAllRoleWaits
                                >(
                                    p_tmem_ready,
                                    phase
                                );
                            }
                            if constexpr (SingleOwnerCluster) {
                                ::kittens::tma::cluster::arrive(
                                    owner_output_loaded,
                                    0
                                );
                            } else if constexpr (
                                CacheRoleClusterAddresses &&
                                !UseX32TmemComputeLayout
                            ) {
                                cta2_fused_dense_cluster_arrive_mapped(
                                    dv_operands_ready_cluster_address
                                );
                            } else {
                                ::kittens::tma::cluster::arrive(
                                    dv_operands_ready,
                                    0
                                );
                            }
                        }
                    } else {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            qdo_exchange_done,
                            phase
                        );
                        auto &qdo_exchange = storage.qdo_phase.qdo;
                        const int source_row = compute_warp & 3;
                        rt_bf<16, 64> do_half;
                        if constexpr (BranchlessDoSourceBaseSelect) {
                            auto local_source =
                                main.dout.template subtile<16, 64>(
                                    {source_row, cta_rank}
                                );
                            auto peer_source =
                                qdo_exchange.dout.template subtile<16, 64>(
                                    {source_row, 0}
                                );
                            cta2_role_split_load_do_half_branchless_base_select(
                                do_half,
                                local_source,
                                peer_source,
                                (compute_warp >> 2) == cta_rank
                            );
                        } else if constexpr (BranchlessDoSourceLoad) {
                            auto local_source =
                                main.dout.template subtile<16, 64>(
                                    {source_row, cta_rank}
                                );
                            auto peer_source =
                                qdo_exchange.dout.template subtile<16, 64>(
                                    {source_row, 0}
                                );
                            cta2_role_split_load_do_half_branchless(
                                do_half,
                                local_source,
                                peer_source,
                                (compute_warp >> 2) == cta_rank
                            );
                        } else if ((compute_warp >> 2) == cta_rank) {
                            auto do_source =
                                main.dout.template subtile<16, 64>(
                                    {source_row, cta_rank}
                                );
                            warp::load(do_half, do_source);
                        } else {
                            auto do_source =
                                qdo_exchange.dout.template subtile<16, 64>(
                                    {source_row, 0}
                                );
                            warp::load(do_half, do_source);
                        }
                        if constexpr (UseNamedDoSourceBarrier) {
                            asm volatile("bar.sync 4, 256;" ::: "memory");
                        } else {
                            warp::arrive(do_source_loaded);
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                do_source_loaded,
                                phase
                            );
                        }
                        auto &do_normal =
                            *reinterpret_cast<st_bf<128, 64> *>(
                                &main.dout
                            );
                        auto do_smem = do_normal.template subtile<16, 64>(
                            {compute_warp, 0}
                        );
                        warp::store(do_smem, do_half);
                        asm volatile(
                            "fence.proxy.async.shared::cta;"
                            ::: "memory"
                        );
                        if constexpr (AsymmetricDvPublish) {
                            if (physical_warp == 4) {
                                asm volatile(
                                    "bar.sync 1, 256;"
                                    ::: "memory"
                                );
                            } else {
                                asm volatile(
                                    "bar.arrive 1, 256;"
                                    ::: "memory"
                                );
                            }
                            if (physical_warp == 4 && lane == 0) {
                                if constexpr (
                                    CacheRoleClusterAddresses &&
                                    !UseX32TmemComputeLayout
                                ) {
                                    cta2_fused_dense_cluster_arrive_mapped(
                                        dv_operands_ready_cluster_address
                                    );
                                } else {
                                    ::kittens::tma::cluster::arrive(
                                        dv_operands_ready,
                                        0
                                    );
                                }
                            }
                        } else {
                            asm volatile("bar.sync 1, 256;" ::: "memory");
                            if (physical_warp == 4 && lane == 0) {
                                if constexpr (SingleOwnerCluster) {
                                    ::kittens::tma::cluster::arrive(
                                        owner_output_loaded,
                                        0
                                    );
                                } else if constexpr (
                                    CacheRoleClusterAddresses &&
                                    !UseX32TmemComputeLayout
                                ) {
                                    cta2_fused_dense_cluster_arrive_mapped(
                                        dv_operands_ready_cluster_address
                                    );
                                } else {
                                    ::kittens::tma::cluster::arrive(
                                        dv_operands_ready,
                                        0
                                    );
                                }
                            }
                        }
                    }
                }
                rt_bf<16, 64> dq_a_local_preload;
                rt_bf<16, 64> dq_a_peer_preload;
                cta2_fused_dense_attn_bf_reg ds_bf;
                if constexpr (UseX32TmemComputeLayout) {
                    const uint32_t physical_row =
                        static_cast<uint32_t>(
                            32 * (compute_warp & 3)
                        );
                    const uint32_t dp_row_address =
                        score_dp_tmem.addr + (physical_row << 16);
                    uint32_t dp_x32_stage0[32];
                    uint32_t dp_x32_stage1[32];
                    cta2_fused_dense_load_tmem_x32(
                        dp_x32_stage0,
                        dp_row_address + x32_column_group
                    );
                    cta2_fused_dense_load_tmem_x32(
                        dp_x32_stage1,
                        dp_row_address + x32_column_group + 64
                    );
                    tensor_load_wait();
                    asm volatile("bar.sync 11, 256;" ::: "memory");
                    if constexpr (EnforceDpTmemConsumerRelease) {
                        if constexpr (SplitDpTmemConsumerRelease) {
                            if (
                                (compute_warp == 0 || compute_warp == 4) &&
                                lane == 0
                            ) {
                                ::kittens::tma::cluster::arrive(
                                    score_consumed,
                                    0
                                );
                            }
                        } else {
                            asm volatile(
                                "bar.sync 9, 256;"
                                ::: "memory"
                            );
                            if (physical_warp == 4 && lane == 0) {
                                ::kittens::tma::cluster::arrive(
                                    score_consumed,
                                    0
                                );
                            }
                        }
                    }
                    cta2_role_split_make_ds_x32_stage(
                        p_x32_stage0,
                        dp_x32_stage0,
                        dpsum_x32_stage0
                    );
                    const uint32_t ds_row_address =
                        ds_tmem.addr + (physical_row << 16);
                    cta2_role_split_store_tmem_x16(
                        ds_row_address + x32_column_group / 2,
                        p_x32_stage0
                    );

                    float dpsum_x32_stage1[32];
                    cta2_role_split_load_stats_x32_stage_shared(
                        dpsum_x32_stage1,
                        storage.stats.dpsum.data +
                            x32_column_group + 64
                    );
                    warp::arrive(storage.stats_consumed);
                    cta2_role_split_make_ds_x32_stage(
                        p_x32_stage1,
                        dp_x32_stage1,
                        dpsum_x32_stage1
                    );
                    cta2_role_split_store_tmem_x16(
                        ds_row_address +
                            (x32_column_group + 64) / 2,
                        p_x32_stage1
                    );
                    tensor_store_wait();
                    tensor_before_thread_sync();
                } else if constexpr (UseSlicedFp32PForDs) {
                    if constexpr (PreissueFirstDpHalfBeforeQdoWait) {
                        tensor_load_wait();
                    } else {
                        cta2_role_split_load_dp_fp32_half<0>(
                            dp_half,
                            score_dp_tmem
                        );
                    }
                    {
                        auto &p_quarter = *reinterpret_cast<
                            const cta2_fused_dense_attn_quarter_reg *
                        >(&p_reg.tiles[0][0]);
                        auto &dp_quarter = *reinterpret_cast<
                            cta2_fused_dense_attn_quarter_reg *
                        >(&dp_half.tiles[0][0]);
                        auto &ds_quarter = *reinterpret_cast<
                            cta2_fused_dense_attn_quarter_bf_reg *
                        >(&ds_bf.tiles[0][0]);
                        cta2_role_split_make_ds_fp32_quarter<0>(
                            ds_quarter,
                            p_quarter,
                            dp_quarter,
                            dpsum_quarter
                        );
                    }
                    cta2_role_split_load_stats_direct_quarter<32>(
                        dpsum_quarter,
                        storage.stats.dpsum
                    );
                    {
                        auto &p_quarter = *reinterpret_cast<
                            const cta2_fused_dense_attn_quarter_reg *
                        >(&p_reg.tiles[0][2]);
                        auto &dp_quarter = *reinterpret_cast<
                            cta2_fused_dense_attn_quarter_reg *
                        >(&dp_half.tiles[0][2]);
                        auto &ds_quarter = *reinterpret_cast<
                            cta2_fused_dense_attn_quarter_bf_reg *
                        >(&ds_bf.tiles[0][2]);
                        cta2_role_split_make_ds_fp32_quarter<1>(
                            ds_quarter,
                            p_quarter,
                            dp_quarter,
                            dpsum_quarter
                        );
                    }
                    {
                        auto &ds_half = *reinterpret_cast<
                            cta2_fused_dense_attn_half_bf_reg *
                        >(&ds_bf.tiles[0][0]);
                        auto ds_half_tmem =
                            ds_tmem.template subtile<
                                cta2_fused_dense_ds_half_tt
                            >(0);
                        group<8>::store_async(ds_half_tmem, ds_half);
                        tensor_store_wait();
                    }
                    if constexpr (
                        OverlapSecondDpLoadWithReleaseBarrier
                    ) {
                        cta2_role_split_issue_dp_fp32_half<1>(
                            dp_half,
                            score_dp_tmem
                        );
                    } else {
                        cta2_role_split_load_dp_fp32_half<1>(
                            dp_half,
                            score_dp_tmem
                        );
                    }
                    if constexpr (EnforceDpTmemConsumerRelease) {
                        if constexpr (SplitDpTmemConsumerRelease) {
                            if constexpr (UseDynamicDpReleaseBarrierId) {
                                const int barrier_id =
                                    9 + (compute_warp >> 2);
                                asm volatile(
                                    "bar.sync %0, 128;"
                                    :: "r"(barrier_id)
                                    : "memory"
                                );
                            } else {
                                if (compute_warp < 4) {
                                    asm volatile(
                                        "bar.sync 9, 128;"
                                        ::: "memory"
                                    );
                                } else {
                                    asm volatile(
                                        "bar.sync 10, 128;"
                                        ::: "memory"
                                    );
                                }
                            }
                            if constexpr (
                                OverlapSecondDpLoadWithReleaseBarrier
                            ) {
                                tensor_load_wait();
                            }
                            if (
                                (compute_warp == 0 || compute_warp == 4) &&
                                lane == 0
                            ) {
                                ::kittens::tma::cluster::arrive(
                                    score_consumed,
                                    0
                                );
                            }
                        } else {
                            asm volatile("bar.sync 9, 256;" ::: "memory");
                            if (physical_warp == 4 && lane == 0) {
                                ::kittens::tma::cluster::arrive(
                                    score_consumed,
                                    0
                                );
                            }
                        }
                    }
                    cta2_role_split_load_stats_direct_quarter<64>(
                        dpsum_quarter,
                        storage.stats.dpsum
                    );
                    {
                        auto &p_quarter = *reinterpret_cast<
                            const cta2_fused_dense_attn_quarter_reg *
                        >(&p_reg.tiles[0][4]);
                        auto &dp_quarter = *reinterpret_cast<
                            cta2_fused_dense_attn_quarter_reg *
                        >(&dp_half.tiles[0][0]);
                        auto &ds_quarter = *reinterpret_cast<
                            cta2_fused_dense_attn_quarter_bf_reg *
                        >(&ds_bf.tiles[0][4]);
                        cta2_role_split_make_ds_fp32_quarter<2>(
                            ds_quarter,
                            p_quarter,
                            dp_quarter,
                            dpsum_quarter
                        );
                    }
                    cta2_role_split_load_stats_direct_quarter<96>(
                        dpsum_quarter,
                        storage.stats.dpsum
                    );
                    {
                        auto &p_quarter = *reinterpret_cast<
                            const cta2_fused_dense_attn_quarter_reg *
                        >(&p_reg.tiles[0][6]);
                        auto &dp_quarter = *reinterpret_cast<
                            cta2_fused_dense_attn_quarter_reg *
                        >(&dp_half.tiles[0][2]);
                        auto &ds_quarter = *reinterpret_cast<
                            cta2_fused_dense_attn_quarter_bf_reg *
                        >(&ds_bf.tiles[0][6]);
                        cta2_role_split_make_ds_fp32_quarter<3>(
                            ds_quarter,
                            p_quarter,
                            dp_quarter,
                            dpsum_quarter
                        );
                    }
                    {
                        auto &ds_half = *reinterpret_cast<
                            cta2_fused_dense_attn_half_bf_reg *
                        >(&ds_bf.tiles[0][4]);
                        auto ds_half_tmem =
                            ds_tmem.template subtile<
                                cta2_fused_dense_ds_half_tt
                            >(64);
                        group<8>::store_async(ds_half_tmem, ds_half);
                        tensor_store_wait();
                    }
                    warp::arrive(storage.stats_consumed);
                    tensor_before_thread_sync();
                } else {
                {
                    cta2_fused_dense_attn_reg dp_reg;
                    if constexpr (!PreloadDpsum) {
                        group<8>::load_async(dp_reg, dp_tmem);
                    } else {
                        group<8>::load_async(dp_reg, score_dp_tmem);
                    }
                    if constexpr (!RetainPackedP) {
                        if constexpr (UseFusedTmemPAndDs) {
                            cta2_role_split_load_p_tmem_exact(p_bf, p_tmem);
                        } else {
                            auto p_smem =
                                main.p.p.template subtile<16, 128>(
                                    {output_subtile, 0}
                                );
                            warp::load(p_bf, p_smem);
                        }
                    }
                    tensor_load_wait();
                    if constexpr (EnforceDpTmemConsumerRelease) {
                        if constexpr (SplitDpTmemConsumerRelease) {
                            if constexpr (UseDynamicDpReleaseBarrierId) {
                                const int barrier_id =
                                    9 + (compute_warp >> 2);
                                asm volatile(
                                    "bar.sync %0, 128;"
                                    :: "r"(barrier_id)
                                    : "memory"
                                );
                            } else {
                                if (compute_warp < 4) {
                                    asm volatile(
                                        "bar.sync 9, 128;"
                                        ::: "memory"
                                    );
                                } else {
                                    asm volatile(
                                        "bar.sync 10, 128;"
                                        ::: "memory"
                                    );
                                }
                            }
                            if (
                                (compute_warp == 0 || compute_warp == 4) &&
                                lane == 0
                            ) {
                                ::kittens::tma::cluster::arrive(
                                    score_consumed,
                                    0
                                );
                            }
                        } else {
                            asm volatile("bar.sync 9, 256;" ::: "memory");
                            if (physical_warp == 4 && lane == 0) {
                                ::kittens::tma::cluster::arrive(
                                    score_consumed,
                                    0
                                );
                            }
                        }
                    }
                    warp::sub_col(dp_reg, dp_reg, dpsum_vec);
                    if constexpr (
                        UseDirectStatsLoads && UseRoleStatsCache &&
                        !UseLongSeqStatsCache
                    ) {
                        warp::arrive(storage.stats_consumed);
                    }
                    if constexpr (UsePackedBf16DsProduct) {
                        cta2_fused_dense_attn_bf_reg dp_bf;
                        warp::copy(dp_bf, dp_reg);
                        warp::mul(ds_bf, p_bf, dp_bf);
                    } else {
                        cta2_fused_dense_attn_reg ds_reg;
                        if constexpr (UseBitwisePExpansion) {
                            cta2_role_split_expand_bf16_bits(ds_reg, p_bf);
                        } else {
                            warp::copy(ds_reg, p_bf);
                        }
                        warp::mul(ds_reg, ds_reg, dp_reg);
                        warp::copy(ds_bf, ds_reg);
                    }
                    if constexpr (RetainDsExchange && !DirectDsHalfStore) {
                        if constexpr (RetainDsLocal) {
                            cta2_role_split_select_ds_half(
                                dq_a_local_preload,
                                ds_bf,
                                cta_rank
                            );
                        }
                        cta2_role_split_select_ds_half(
                            dq_a_peer_preload,
                            ds_bf,
                            cta_rank ^ 1
                        );
                    }
                    if constexpr (UseFusedTmemPAndDs || !PreloadDpsum) {
                        group<8>::store_async(ds_tmem, ds_bf);
                        tensor_store_wait();
                        tensor_before_thread_sync();
                    }
                    if constexpr (!UseFusedTmemPAndDs) {
                        auto ds_smem = storage.ds.template subtile<16, 128>(
                            {output_subtile, 0}
                        );
                        warp::store(ds_smem, ds_bf);
                    }
                }
                }
                if constexpr (SplitDqTmemAndSharedHandoff) {
                    if (have_previous_dq_shared_read) {
                        if constexpr (DistributedDqSharedReadWait) {
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                dq_shared_read_done,
                                previous_dq_shared_read_phase
                            );
                        } else {
                            if (physical_warp == 4) {
                                cta2_fused_dense_role_wait<
                                    TimeoutAllRoleWaits
                                >(
                                    dq_shared_read_done,
                                    previous_dq_shared_read_phase
                                );
                            }
                            asm volatile(
                                "bar.sync 11, 256;"
                                ::: "memory"
                            );
                        }
                    }
                }
                if constexpr (OverlapFusedDqAPublication) {
                    if (physical_warp == 4 && lane == 0) {
                        ::kittens::tma::cluster::expect_bytes(
                            dq_exchange_done,
                            sizeof(storage.qdo_phase.ds),
                            cta_rank ^ 1
                        );
                    }
                    auto &dq_a = *reinterpret_cast<
                        cta2_fused_dense_dq_a_tile *
                    >(&main.p.p);
                    if constexpr (UseX32TmemComputeLayout) {
                        const int row_base =
                            32 * (compute_warp & 3);
                        if (cta_rank == 0) {
                            cta2_role_split_store_ds_x32_stage(
                                dq_a,
                                p_x32_stage0,
                                row_base,
                                x32_column_group
                            );
                            cta2_role_split_store_ds_x32_stage(
                                storage.qdo_phase.ds,
                                p_x32_stage1,
                                row_base,
                                x32_column_group
                            );
                        } else {
                            cta2_role_split_store_ds_x32_stage(
                                dq_a,
                                p_x32_stage1,
                                128 + row_base,
                                x32_column_group
                            );
                            cta2_role_split_store_ds_x32_stage(
                                storage.qdo_phase.ds,
                                p_x32_stage0,
                                row_base,
                                x32_column_group
                            );
                        }
                    } else {
                        auto a_local_smem =
                            dq_a.template subtile<16, 64>(
                                {cta_rank * 8 + output_subtile, 0}
                            );
                        auto a_peer_smem =
                            storage.qdo_phase.ds.template subtile<16, 64>(
                                {output_subtile, 0}
                            );
                        if (cta_rank == 0) {
                            cta2_role_split_store_ds_half_direct<0>(
                                a_local_smem,
                                ds_bf
                            );
                        } else {
                            cta2_role_split_store_ds_half_direct<1>(
                                a_local_smem,
                                ds_bf
                            );
                        }
                        if (cta_rank == 0) {
                            cta2_role_split_store_ds_half_direct<1>(
                                a_peer_smem,
                                ds_bf
                            );
                        } else {
                            cta2_role_split_store_ds_half_direct<0>(
                                a_peer_smem,
                                ds_bf
                            );
                        }
                    }
                    __syncwarp();
                    asm volatile(
                        "fence.proxy.async.shared::cta;"
                        ::: "memory"
                    );
                    asm volatile("bar.sync 3, 256;" ::: "memory");
                    if (physical_warp == 4 && lane == 0) {
                        constexpr int kHalfElements = 128 * 64;
                        void *destination = reinterpret_cast<void *>(
                            &dq_a.data[cta_rank * kHalfElements]
                        );
                        cta2_role_split_store_peer_bulk_after_cta_fence(
                            destination,
                            reinterpret_cast<void *>(
                                &storage.qdo_phase.ds
                            ),
                            kHalfElements * sizeof(bf16),
                            cta_rank ^ 1,
                            dq_exchange_done
                        );
                    }
                    asm volatile(
                        "fence.proxy.async.shared::cta;"
                        ::: "memory"
                    );
                    warp::arrive(dq_local_ready);
                }
                if constexpr (!SplitDvDkReady) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        qdo_exchange_done,
                        phase
                    );
                    auto &qdo_exchange = storage.qdo_phase.qdo;
                    const int source_row = compute_warp & 3;
                    rt_bf<16, 64> do_half;
                    if ((compute_warp >> 2) == cta_rank) {
                        auto do_source = main.dout.template subtile<16, 64>(
                            {source_row, cta_rank}
                        );
                        warp::load(do_half, do_source);
                    } else {
                        auto do_source =
                            qdo_exchange.dout.template subtile<16, 64>(
                                {source_row, 0}
                            );
                        warp::load(do_half, do_source);
                    }
                    if constexpr (UseNormalDoDv) {
                        warp::arrive(do_source_loaded);
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            do_source_loaded,
                            phase
                        );
                        auto &do_normal =
                            *reinterpret_cast<st_bf<128, 64> *>(
                                &main.dout
                            );
                        auto do_smem = do_normal.template subtile<16, 64>(
                            {compute_warp, 0}
                        );
                        warp::store(do_smem, do_half);
                    } else {
                        rt_bf<64, 16> do_transposed;
                        warp::transpose_sep(do_transposed, do_half);
                        warp::arrive(do_source_loaded);
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            do_source_loaded,
                            phase
                        );
                        auto do_smem = main.dout.template subtile<64, 16>(
                            {0, compute_warp}
                        );
                        warp::store(do_smem, do_transposed);
                    }
                }
                if constexpr (FenceDsBeforeDkdvReady) {
                    asm volatile(
                        "fence.proxy.async.shared::cta;"
                        ::: "memory"
                    );
                }
                if constexpr (UseNamedDkdvLocalFanIn) {
                    if (physical_warp == 4) {
                        asm volatile("bar.sync 7, 384;" ::: "memory");
                    } else {
                        asm volatile("bar.arrive 7, 384;" ::: "memory");
                    }
                    if (physical_warp == 4 && lane == 0) {
                        if constexpr (
                            CacheRoleClusterAddresses &&
                            !UseX32TmemComputeLayout
                        ) {
                            cta2_fused_dense_cluster_arrive_mapped(
                                dkdv_operands_ready_cluster_address
                            );
                        } else {
                            ::kittens::tma::cluster::arrive(
                                dkdv_operands_ready,
                                0
                            );
                        }
                    }
                } else {
                    warp::arrive(dkdv_local_ready);
                    if (physical_warp == 4 && lane == 0) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            dkdv_local_ready,
                            phase
                        );
                        if constexpr (
                            CacheRoleClusterAddresses &&
                            !UseX32TmemComputeLayout
                        ) {
                            cta2_fused_dense_cluster_arrive_mapped(
                                dkdv_operands_ready_cluster_address
                            );
                        } else {
                            ::kittens::tma::cluster::arrive(
                                dkdv_operands_ready,
                                0
                            );
                        }
                    }
                }
                if constexpr (RetainDsLocal) {
                    // Both dS halves remain in their producer warp registers.
                } else if constexpr (RetainDsExchange) {
                    auto ds_local_smem =
                        storage.ds.template subtile<16, 64>(
                            {compute_warp, cta_rank}
                        );
                    warp::load(dq_a_local_preload, ds_local_smem);
                } else {
                    auto ds_local_smem =
                        storage.ds.template subtile<16, 64>(
                            {compute_warp, cta_rank}
                        );
                    auto ds_peer_smem =
                        storage.ds.template subtile<16, 64>(
                            {compute_warp, cta_rank ^ 1}
                        );
                    warp::load(dq_a_local_preload, ds_local_smem);
                    warp::load(dq_a_peer_preload, ds_peer_smem);
                }
                if constexpr (StageDqAfterDv) {
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                }
                if constexpr (StageDqPeerBeforeDv) {
                    auto a_peer_smem =
                        storage.qdo_phase.ds.template subtile<16, 64>({
                            RetainDsExchange ? output_subtile : compute_warp,
                            0
                        });
                    warp::store(a_peer_smem, dq_a_peer_preload);
                    asm volatile(
                        "fence.proxy.async.shared::cta;"
                        ::: "memory"
                    );
                    warp::arrive(dq_peer_ready);
                }
                cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                    dkdv_done,
                    phase
                );

                if constexpr (
                    !OverlapFusedDqAPublication &&
                    (DirectAsyncPeerDs || ProducerBulkPeerDs)
                ) {
                    if (physical_warp == 4 && lane == 0) {
                        ::kittens::tma::cluster::expect_bytes(
                            dq_exchange_done,
                            sizeof(storage.qdo_phase.ds),
                            cta_rank ^ 1
                        );
                    }
                    if constexpr (!UseFusedTmemPAndDs) {
                        asm volatile("bar.sync 3, 256;" ::: "memory");
                    }
                }

                if constexpr (!OverlapFusedDqAPublication) {
                    auto &dq_operands = storage.phase.dq.operands;
                    auto &dq_a = [&]() -> cta2_fused_dense_dq_a_tile & {
                        if constexpr (StageDqAfterDv) {
                            return *reinterpret_cast<
                                cta2_fused_dense_dq_a_tile *
                            >(&main.p.p);
                        } else {
                            return dq_operands.a;
                        }
                    }();
                    auto a_local_smem =
                        dq_a.template subtile<16, 64>(
                            {
                                cta_rank * 8 +
                                    (RetainDsLocal
                                        ? output_subtile
                                        : compute_warp),
                                0
                            }
                        );
                    if constexpr (DirectDsHalfStore) {
                        if (cta_rank == 0) {
                            cta2_role_split_store_ds_half_direct<0>(
                                a_local_smem,
                                ds_bf
                            );
                        } else {
                            cta2_role_split_store_ds_half_direct<1>(
                                a_local_smem,
                                ds_bf
                            );
                        }
                    } else {
                        warp::store(a_local_smem, dq_a_local_preload);
                    }
                    if constexpr (!StageDqPeerBeforeDv) {
                        if constexpr (DirectAsyncPeerDs) {
                            auto a_peer_smem = dq_a.template subtile<16, 64>(
                                {cta_rank * 8 + output_subtile, 0}
                            );
                            asm volatile(
                                "fence.proxy.async.shared::cta;"
                                ::: "memory"
                            );
                            if (cta_rank == 0) {
                                cta2_role_split_store_ds_half_remote_async<1>(
                                    a_peer_smem,
                                    ds_bf,
                                    1,
                                    dq_exchange_done
                                );
                            } else {
                                cta2_role_split_store_ds_half_remote_async<0>(
                                    a_peer_smem,
                                    ds_bf,
                                    0,
                                    dq_exchange_done
                                );
                            }
                        } else {
                            if constexpr (
                                !BulkPeerDsFromFullTile ||
                                UseFusedTmemPAndDs
                            ) {
                                auto a_peer_smem =
                                    storage.qdo_phase.ds.template
                                        subtile<16, 64>({
                                            RetainDsExchange
                                                ? output_subtile
                                                : compute_warp,
                                            0
                                        });
                                if constexpr (DirectDsHalfStore) {
                                    if (cta_rank == 0) {
                                        cta2_role_split_store_ds_half_direct<1>(
                                            a_peer_smem,
                                            ds_bf
                                        );
                                    } else {
                                        cta2_role_split_store_ds_half_direct<0>(
                                            a_peer_smem,
                                            ds_bf
                                        );
                                    }
                                } else {
                                    warp::store(
                                        a_peer_smem,
                                        dq_a_peer_preload
                                    );
                                }
                            }
                            if constexpr (ProducerBulkPeerDs) {
                                __syncwarp();
                                asm volatile(
                                    "fence.proxy.async.shared::cta;"
                                    ::: "memory"
                                );
                                if constexpr (UseFusedTmemPAndDs) {
                                    asm volatile(
                                        "bar.sync 3, 256;"
                                        ::: "memory"
                                    );
                                }
                                if constexpr (CoalescedPeerDsBulk) {
                                    if (physical_warp == 4 && lane == 0) {
                                        constexpr int kHalfElements =
                                            128 * 64;
                                        void *destination =
                                            reinterpret_cast<void *>(
                                                &dq_a.data[
                                                    cta_rank * kHalfElements
                                                ]
                                            );
                                        void *source;
                                        if constexpr (UseFusedTmemPAndDs) {
                                            source = reinterpret_cast<void *>(
                                                &storage.qdo_phase.ds
                                            );
                                        } else {
                                            source = reinterpret_cast<void *>(
                                                &storage.ds.data[
                                                    (cta_rank ^ 1) *
                                                        kHalfElements
                                                ]
                                            );
                                        }
                                        cta2_role_split_store_peer_bulk_after_cta_fence(
                                            destination,
                                            source,
                                            kHalfElements * sizeof(bf16),
                                            cta_rank ^ 1,
                                            dq_exchange_done
                                        );
                                    }
                                } else if (lane == 0) {
                                    constexpr int kChunkElements = 16 * 64;
                                    void *destination = reinterpret_cast<void *>(
                                        &dq_a.data[
                                            (cta_rank * 128 +
                                             output_subtile * 16) * 64
                                        ]
                                    );
                                    void *source;
                                    if constexpr (BulkPeerDsFromFullTile) {
                                        constexpr int kHalfElements =
                                            128 * 64;
                                        source = reinterpret_cast<void *>(
                                            &storage.ds.data[
                                                (cta_rank ^ 1) *
                                                    kHalfElements +
                                                output_subtile *
                                                    kChunkElements
                                            ]
                                        );
                                    } else {
                                        source = reinterpret_cast<void *>(
                                            &storage.qdo_phase.ds.data[
                                                output_subtile *
                                                    kChunkElements
                                            ]
                                        );
                                    }
                                    if constexpr (
                                        ProducerBulkPeerDsCtaFenceOnly
                                    ) {
                                        cta2_role_split_store_peer_bulk_after_cta_fence(
                                            destination,
                                            source,
                                            kChunkElements * sizeof(bf16),
                                            cta_rank ^ 1,
                                            dq_exchange_done
                                        );
                                    } else {
                                        ::kittens::tma::cluster::store_async(
                                            destination,
                                            source,
                                            kChunkElements * sizeof(bf16),
                                            cta_rank ^ 1,
                                            dq_exchange_done
                                        );
                                    }
                                }
                            }
                        }
                    }
                    asm volatile(
                        "fence.proxy.async.shared::cta;"
                        ::: "memory"
                    );
                    warp::arrive(dq_local_ready);
                }
                if constexpr (CarryDirectStatsOffset) {
                    carried_direct_stats_offset += q_tile_step * 128;
                }
                if constexpr (SplitDqTmemAndSharedHandoff) {
                    have_previous_dq_shared_read = true;
                    previous_dq_shared_read_phase = phase;
                }
                if constexpr (CarryAllRolePhases) {
                    compute_phase ^= 1;
                }
            }
            if (first_dense_q_tile < q_tile_count) {
                const int iteration_count =
                    (q_tile_count - first_dense_q_tile + q_tile_step - 1) /
                    q_tile_step;
                const int last_phase =
                    (iteration_count - 1 + owner_phase) & 1;
                cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                    dq_done,
                    last_phase
                );
            }
        }
        return;
    }

    if (physical_warp == 12) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 104;" ::: "memory");
        if (cta_rank == 0 && lane == 0) {
            uint32_t score_done_commit_address = 0;
            uint32_t dp_done_commit_address = 0;
            uint32_t dkdv_done_commit_address = 0;
            uint32_t dk_done_commit_address = 0;
            uint32_t dq_done_commit_address = 0;
            if constexpr (CacheTensorCommitAddresses) {
                score_done_commit_address =
                    cta2_fused_dense_map_cluster_semaphore(score_done, 0);
                dp_done_commit_address =
                    cta2_fused_dense_map_cluster_semaphore(dp_done, 0);
                dkdv_done_commit_address =
                    cta2_fused_dense_map_cluster_semaphore(dkdv_done, 0);
                if constexpr (PrefetchNextQdoAfterDkdv) {
                    dk_done_commit_address =
                        cta2_fused_dense_map_cluster_semaphore(dk_done, 0);
                }
                dq_done_commit_address =
                    cta2_fused_dense_map_cluster_semaphore(dq_done, 0);
            }
            uint64_t compact_score_a_desc = 0;
            uint64_t compact_score_b_desc = 0;
            uint64_t compact_dp_a_desc = 0;
            uint64_t compact_dp_b_desc = 0;
            if constexpr (UseCompactScoreMma) {
                ::kittens::st_descriptor<
                    cta2_fused_dense_k_tile,
                    transpose::N
                > a_desc(storage.k);
                ::kittens::st_descriptor<
                    cta2_fused_dense_q_tile,
                    transpose::N
                > b_desc(storage.phase.main.q.q);
                compact_score_a_desc = a_desc.base_desc;
                compact_score_b_desc = b_desc.base_desc;
            }
            if constexpr (UseCompactDpMma) {
                ::kittens::st_descriptor<
                    cta2_fused_dense_ds_tile,
                    transpose::N
                > a_desc(storage.ds);
                ::kittens::st_descriptor<
                    cta2_fused_dense_do_tile,
                    transpose::N
                > b_desc(storage.phase.main.dout);
                compact_dp_a_desc = a_desc.base_desc;
                compact_dp_b_desc = b_desc.base_desc;
            }
            #pragma unroll 1
            for (
                int owner_pass = 0;
                owner_pass < (SingleOwnerCluster ? 1 : 2);
                ++owner_pass
            ) {
                const int owner_idx = SingleOwnerCluster
                    ? owner_pair_idx
                    : (owner_pass == 0
                        ? owner_pair_idx
                        : owner_count - 1 - owner_pair_idx);
                const int owner_phase = OwnerQWorkSplitId >= 0
                    ? (owner_pass == 0
                        ? 0
                        : ((owner_count - owner_pair_idx) & 1))
                    : (SingleOwnerCluster || IntegrateCausalFrontier
                        ? 0
                        : (owner_pass & 1));
                const int owner_once_phase =
                    SingleOwnerCluster ? 0 : (owner_pass & 1);
            const int first_dense_q_tile =
                2 * owner_idx + (IntegrateCausalFrontier ? 0 : 1) +
                (OwnerQWorkSplitId < 0 ? 0 : OwnerQWorkSplitId);
                cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                    k_ready,
                    owner_once_phase
                );
                bool first_accumulation = true;
                int iteration = 0;
                int tensor_phase = owner_phase;
                for (
                    int q_tile_idx = first_dense_q_tile;
                    q_tile_idx < q_tile_count;
                q_tile_idx += q_tile_step, ++iteration
                ) {
                    const int phase = CarryAllRolePhases
                        ? tensor_phase
                        : ((iteration + owner_phase) & 1);
                    auto &main = storage.phase.main;
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        qdo_ready,
                        phase
                    );
                    if constexpr (UseCompactScoreMma) {
                        cta2_fused_dense_score_mma_commit_compact(
                            score_dp_tmem.addr,
                            compact_score_a_desc,
                            compact_score_b_desc,
                            score_done_commit_address
                        );
                    } else {
                        cta2_fused_dense_mm2_abt_no_fence(
                            score_dp_tmem,
                            storage.k,
                            main.q.q
                        );
                        cta2_fused_dense_commit_selected<
                            CacheTensorCommitAddresses
                        >(score_done, score_done_commit_address);
                    }
                    if constexpr (!MergeScoreDpReady) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            score_consumed,
                            phase
                        );
                    }
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        dp_operands_ready,
                        phase
                    );
                    if constexpr (UseCompactDpMma) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;"
                            ::: "memory"
                        );
                        cta2_fused_dense_dp_mma_commit_compact(
                            score_dp_tmem.addr,
                            compact_dp_a_desc,
                            compact_dp_b_desc,
                            dp_done_commit_address
                        );
                    } else {
                        if constexpr (!PreloadDpsum) {
                            ::kittens::mm2_ABt(
                                dp_tmem,
                                storage.ds,
                                main.dout
                            );
                        } else {
                            ::kittens::mm2_ABt(
                                score_dp_tmem,
                                storage.ds,
                                main.dout
                            );
                        }
                        cta2_fused_dense_commit_selected<
                            CacheTensorCommitAddresses
                        >(dp_done, dp_done_commit_address);
                    }
                    if constexpr (SplitDvDkReady) {
                        if constexpr (SingleOwnerCluster) {
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                owner_output_loaded,
                                phase
                            );
                        } else {
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                dv_operands_ready,
                                phase
                            );
                        }
                        auto &do_normal =
                            *reinterpret_cast<st_bf<128, 64> *>(
                                &main.dout
                            );
                        if constexpr (UseFusedTmemPAndDs) {
                            if constexpr (
                                UseFusedTmemRuntimeAccumulationPredicate
                            ) {
                                cta2_fused_dense_tmem_a_mm2_ab_no_fence_runtime_accumulate(
                                    dv_tmem,
                                    p_tmem,
                                    do_normal,
                                    !first_accumulation
                                );
                            } else if (first_accumulation) {
                                ::kittens::mm2_AB(
                                    dv_tmem,
                                    p_tmem,
                                    do_normal
                                );
                            } else {
                                ::kittens::mma2_AB(
                                    dv_tmem,
                                    p_tmem,
                                    do_normal
                                );
                            }
                        } else if constexpr (
                            UseRuntimeAccumulationPredicate
                        ) {
                            cta2_fused_dense_mm2_ab_no_fence_runtime_accumulate(
                                dv_tmem,
                                main.p.p,
                                do_normal,
                                !first_accumulation
                            );
                        } else {
                            if (first_accumulation) {
                                ::kittens::mm2_AB(
                                    dv_tmem,
                                    main.p.p,
                                    do_normal
                                );
                            } else {
                                ::kittens::mma2_AB(
                                    dv_tmem,
                                    main.p.p,
                                    do_normal
                                );
                            }
                        }
                        if constexpr (StageDqAfterDv) {
                            cta2_fused_dense_commit_selected<
                                CacheTensorCommitAddresses
                            >(
                                dkdv_done,
                                dkdv_done_commit_address
                            );
                        }
                    }
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        dkdv_operands_ready,
                        phase
                    );
                    if constexpr (!PreloadDpsum) {
                        tensor_after_thread_sync();
                    }
                    if constexpr (!SingleOwnerCluster) {
                        if (first_accumulation && owner_pass != 0) {
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                owner_output_loaded,
                                owner_once_phase ^ 1
                            );
                        }
                    }
                    if constexpr (UseFusedTmemPAndDs) {
                        if constexpr (
                            UseFusedTmemRuntimeAccumulationPredicate
                        ) {
                            cta2_fused_dense_tmem_a_mm2_ab_no_fence_runtime_accumulate(
                                dk_wide_tmem,
                                ds_tmem,
                                main.q.normal_wide,
                                !first_accumulation
                            );
                        } else if (first_accumulation) {
                            ::kittens::mm2_AB(
                                dk_wide_tmem,
                                ds_tmem,
                                main.q.normal_wide
                            );
                        } else {
                            ::kittens::mma2_AB(
                                dk_wide_tmem,
                                ds_tmem,
                                main.q.normal_wide
                            );
                        }
                    } else if constexpr (
                        UseRuntimeAccumulationPredicate
                    ) {
                        static_assert(
                            PreloadDpsum && SplitDvDkReady && DirectTmaDkQ
                        );
                        if constexpr (UseWideDkN192) {
                            cta2_fused_dense_mm2_ab_no_fence_runtime_accumulate(
                                dk_wide_tmem,
                                storage.ds,
                                main.q.normal_wide,
                                !first_accumulation
                            );
                        } else {
                            cta2_fused_dense_mm2_ab_no_fence_runtime_accumulate(
                                dk_main_tmem,
                                storage.ds,
                                main.q.normal.q0,
                                !first_accumulation
                            );
                            cta2_fused_dense_mm2_ab_no_fence_runtime_accumulate(
                                dk_tail_tmem,
                                storage.ds,
                                main.q.normal.q1,
                                !first_accumulation
                            );
                        }
                    } else if (first_accumulation) {
                        if constexpr (!PreloadDpsum) {
                            ::kittens::mm2_ABt(
                                dv_tmem,
                                p_tmem,
                                main.dout
                            );
                            ::kittens::mm2_ABt(
                                dk_main_tmem,
                                ds_tmem,
                                main.q.transposed.q0
                            );
                            ::kittens::mm2_ABt(
                                dk_tail_tmem,
                                ds_tmem,
                                main.q.transposed.q1
                            );
                        } else {
                            if constexpr (
                                UseNormalDoDv && !SplitDvDkReady
                            ) {
                                auto &do_normal =
                                    *reinterpret_cast<st_bf<128, 64> *>(
                                        &main.dout
                                    );
                                ::kittens::mm2_AB(
                                    dv_tmem,
                                    main.p.p,
                                    do_normal
                                );
                            } else if constexpr (!SplitDvDkReady) {
                                ::kittens::mm2_ABt(
                                    dv_tmem,
                                    main.p.p,
                                    main.dout
                                );
                            }
                            if constexpr (UseWideDkN192) {
                                if constexpr (DirectTmaDkQ) {
                                    cta2_fused_dense_mm2_ab_no_fence<0>(
                                        dk_wide_tmem,
                                        storage.ds,
                                        main.q.normal_wide
                                    );
                                } else {
                                    cta2_fused_dense_mm2_abt_no_fence<0>(
                                        dk_wide_tmem,
                                        storage.ds,
                                        main.q.transposed_wide
                                    );
                                }
                            } else if constexpr (DirectTmaDkQ) {
                                ::kittens::mm2_AB(
                                    dk_main_tmem,
                                    storage.ds,
                                    main.q.normal.q0
                                );
                                ::kittens::mm2_AB(
                                    dk_tail_tmem,
                                    storage.ds,
                                    main.q.normal.q1
                                );
                            } else {
                                ::kittens::mm2_ABt(
                                    dk_main_tmem,
                                    storage.ds,
                                    main.q.transposed.q0
                                );
                                ::kittens::mm2_ABt(
                                    dk_tail_tmem,
                                    storage.ds,
                                    main.q.transposed.q1
                                );
                            }
                        }
                    } else {
                        if constexpr (!PreloadDpsum) {
                            ::kittens::mma2_ABt(
                                dv_tmem,
                                p_tmem,
                                main.dout
                            );
                            ::kittens::mma2_ABt(
                                dk_main_tmem,
                                ds_tmem,
                                main.q.transposed.q0
                            );
                            ::kittens::mma2_ABt(
                                dk_tail_tmem,
                                ds_tmem,
                                main.q.transposed.q1
                            );
                        } else {
                            if constexpr (
                                UseNormalDoDv && !SplitDvDkReady
                            ) {
                                auto &do_normal =
                                    *reinterpret_cast<st_bf<128, 64> *>(
                                        &main.dout
                                    );
                                ::kittens::mma2_AB(
                                    dv_tmem,
                                    main.p.p,
                                    do_normal
                                );
                            } else if constexpr (!SplitDvDkReady) {
                                ::kittens::mma2_ABt(
                                    dv_tmem,
                                    main.p.p,
                                    main.dout
                                );
                            }
                            if constexpr (UseWideDkN192) {
                                if constexpr (DirectTmaDkQ) {
                                    cta2_fused_dense_mm2_ab_no_fence<1>(
                                        dk_wide_tmem,
                                        storage.ds,
                                        main.q.normal_wide
                                    );
                                } else {
                                    cta2_fused_dense_mm2_abt_no_fence<1>(
                                        dk_wide_tmem,
                                        storage.ds,
                                        main.q.transposed_wide
                                    );
                                }
                            } else if constexpr (DirectTmaDkQ) {
                                ::kittens::mma2_AB(
                                    dk_main_tmem,
                                    storage.ds,
                                    main.q.normal.q0
                                );
                                ::kittens::mma2_AB(
                                    dk_tail_tmem,
                                    storage.ds,
                                    main.q.normal.q1
                                );
                            } else {
                                ::kittens::mma2_ABt(
                                    dk_main_tmem,
                                    storage.ds,
                                    main.q.transposed.q0
                                );
                                ::kittens::mma2_ABt(
                                    dk_tail_tmem,
                                    storage.ds,
                                    main.q.transposed.q1
                                );
                            }
                        }
                    }
                    if constexpr (PrefetchNextQdoAfterDkdv) {
                        cta2_fused_dense_commit_selected<
                            CacheTensorCommitAddresses
                        >(
                            dk_done,
                            dk_done_commit_address
                        );
                    }
                    if constexpr (!StageDqAfterDv) {
                        cta2_fused_dense_commit_selected<
                            CacheTensorCommitAddresses
                        >(
                            dkdv_done,
                            dkdv_done_commit_address
                        );
                    }
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        dq_operands_ready,
                        phase
                    );
                    auto &dq_a = [&]() -> cta2_fused_dense_dq_a_tile & {
                        if constexpr (StageDqAfterDv) {
                            return *reinterpret_cast<
                                cta2_fused_dense_dq_a_tile *
                            >(&main.p.p);
                        } else {
                            return storage.phase.dq.operands.a;
                        }
                    }();
                    cta2_fused_dense_mm2_atb_no_fence(
                        dq_tmem,
                        dq_a,
                        storage.dq_b
                    );
                    cta2_fused_dense_commit_selected<
                        CacheTensorCommitAddresses
                    >(dq_done, dq_done_commit_address);
                    if constexpr (EnforceDpTmemConsumerRelease) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            score_consumed,
                            phase
                        );
                    }
                    first_accumulation = false;
                    if constexpr (CarryAllRolePhases) {
                        tensor_phase ^= 1;
                    }
                }
            }
        }
        return;
    }

    if (physical_warp == 13) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 104;" ::: "memory");
        uint32_t qdo_ready_cluster_address = 0;
        if constexpr (CacheQdoReadyClusterAddress) {
            if (lane == 0) {
                qdo_ready_cluster_address =
                    cta2_fused_dense_map_cluster_semaphore(qdo_ready, 0);
            }
        }
        int loader_stats_iteration = 0;
        #pragma unroll 1
        for (
            int owner_pass = 0;
            owner_pass < (SingleOwnerCluster ? 1 : 2);
            ++owner_pass
        ) {
            const int owner_idx = SingleOwnerCluster
                ? owner_pair_idx
                : (owner_pass == 0
                    ? owner_pair_idx
                    : owner_count - 1 - owner_pair_idx);
            const int owner_phase = OwnerQWorkSplitId >= 0
                ? (owner_pass == 0
                    ? 0
                    : ((owner_count - owner_pair_idx) & 1))
                : (SingleOwnerCluster || IntegrateCausalFrontier
                    ? 0
                    : (owner_pass & 1));
            const int owner_once_phase =
                SingleOwnerCluster ? 0 : (owner_pass & 1);
            const int first_dense_q_tile =
                2 * owner_idx + (IntegrateCausalFrontier ? 0 : 1) +
                (OwnerQWorkSplitId < 0 ? 0 : OwnerQWorkSplitId);
            auto &main = storage.phase.main;
            semaphore *current_qdo_done = &qdo_load_done;
            int current_qdo_phase = owner_once_phase;
            if constexpr (LoaderOwnedDkQ) {
                if (owner_pass == 0) {
                    coord<cta2_fused_dense_q_tile> q_tile_coord = {
                        batch_idx,
                        first_dense_q_tile * 2 + cta_rank,
                        head_idx,
                        0
                    };
                    cta2_fused_dense_load_qdo_elected_pair(
                        main.q.q,
                        g.q,
                        main.dout,
                        g.dout,
                        q_tile_coord,
                        qdo_load_done,
                        UseLongSeqStatsCache
                            ? sizeof(sv_fl<128>)
                            : 0
                    );
                    if constexpr (UseLongSeqStatsCache) {
                        if (lane == 0) {
                            cta2_fused_dense_load_dpsum(
                                g,
                                storage.stats,
                                qdo_load_done,
                                batch_idx,
                                first_dense_q_tile,
                                head_idx
                            );
                        }
                    }
                }
            }
            int iteration = 0;
            int loader_phase = owner_phase;
            int loader_local_phase = OwnerQWorkSplitId >= 0
                ? (owner_pass == 0
                    ? 0
                    : ((owner_count - owner_pair_idx - 1) & 1))
                : (IntegrateCausalFrontier ? owner_pass : 0);
            for (
                int q_tile_idx = first_dense_q_tile;
                q_tile_idx < q_tile_count;
                q_tile_idx += q_tile_step,
                ++iteration,
                ++loader_stats_iteration
            ) {
                const int phase = CarryAllRolePhases
                    ? loader_phase
                    : ((iteration + owner_phase) & 1);
                const int local_phase = CarryAllRolePhases
                    ? loader_local_phase
                    : (OwnerQWorkSplitId >= 0
                        ? (iteration +
                           (owner_pass == 0
                                ? 0
                                : ((owner_count - owner_pair_idx - 1) & 1))) &
                            1
                        : (iteration +
                           (IntegrateCausalFrontier ? owner_pass : 0)) & 1);
                if constexpr (ElectedPeerDoTmaLoad) {
                    cta2_fused_dense_load_peer_do_elected(
                        g,
                        storage.qdo_phase.qdo,
                        qdo_exchange_done,
                        batch_idx,
                        q_tile_idx,
                        head_idx,
                        cta_rank
                    );
                } else if (lane == 0) {
                    cta2_fused_dense_load_peer_qdo<
                        UseWideDkN192,
                        !DirectTmaDkQ
                    >(
                        g,
                        storage.qdo_phase.qdo,
                        qdo_exchange_done,
                        batch_idx,
                        q_tile_idx,
                        head_idx,
                        cta_rank
                    );
                }
                if constexpr (LoaderOwnedDkQ) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        *current_qdo_done,
                        current_qdo_phase
                    );
                } else if (iteration == 0) {
                    const bool issue_initial_qdo =
                        !PrefetchNextOwnerQdo || owner_pass == 0;
                    if (issue_initial_qdo) {
                        if constexpr (GroupQdoTmaLoads) {
                            coord<cta2_fused_dense_q_tile> q_tile_coord = {
                                batch_idx,
                                q_tile_idx * 2 + cta_rank,
                                head_idx,
                                0
                            };
                            cta2_fused_dense_load_qdo_elected_pair(
                                main.q.q,
                                g.q,
                                main.dout,
                                g.dout,
                                q_tile_coord,
                                qdo_load_done
                            );
                        } else if (lane == 0) {
                            tma::expect_bytes(
                                qdo_load_done,
                                sizeof(main.q.q) + sizeof(main.dout)
                            );
                            coord<cta2_fused_dense_q_tile> q_tile_coord = {
                                batch_idx,
                                q_tile_idx * 2 + cta_rank,
                                head_idx,
                                0
                            };
                            coord<cta2_fused_dense_do_tile> do_tile_coord = {
                                batch_idx,
                                q_tile_idx * 2 + cta_rank,
                                head_idx,
                                0
                            };
                            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                                main.q.q,
                                g.q,
                                q_tile_coord,
                                qdo_load_done
                            );
                            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                                main.dout,
                                g.dout,
                                do_tile_coord,
                                qdo_load_done
                            );
                        }
                    }
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        qdo_load_done,
                        owner_once_phase
                    );
                } else {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        qdo_prefetch_done,
                        local_phase ^ 1
                    );
                    if constexpr (!DirectNextQdoDuringDqDrain) {
                        auto &next_q =
                            storage.phase.dq.operands.exchange.next_q;
                        auto &next_dout =
                            *reinterpret_cast<cta2_fused_dense_do_tile *>(
                                &storage.ds
                            );
                        #pragma unroll 1
                        for (
                            int logical_warp = 0;
                            logical_warp < 4;
                            ++logical_warp
                        ) {
                            rt_bf<16, kB300QKDim> q_reg;
                            auto q_source =
                                next_q.template subtile<16, kB300QKDim>(
                                    {logical_warp, 0}
                                );
                            auto q_destination =
                                main.q.q.template subtile<16, kB300QKDim>(
                                    {logical_warp, 0}
                                );
                            warp::load(q_reg, q_source);
                            warp::store(q_destination, q_reg);
                        }
                        #pragma unroll 1
                        for (
                            int logical_warp = 0;
                            logical_warp < 4;
                            ++logical_warp
                        ) {
                            rt_bf<16, kB300VDim> do_reg;
                            auto do_source =
                                next_dout.template subtile<16, kB300VDim>(
                                    {logical_warp, 0}
                                );
                            auto do_destination =
                                main.dout.template subtile<16, kB300VDim>(
                                    {logical_warp, 0}
                                );
                            warp::load(do_reg, do_source);
                            warp::store(do_destination, do_reg);
                        }
                    }
                }
                if constexpr (LeaderOnlyQdoPublishFence) {
                    if (lane == 0) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;"
                            ::: "memory"
                        );
                        if constexpr (CacheQdoReadyClusterAddress) {
                            cta2_fused_dense_cluster_arrive_mapped(
                                qdo_ready_cluster_address
                            );
                        } else {
                            ::kittens::tma::cluster::arrive(qdo_ready, 0);
                        }
                    }
                } else {
                    asm volatile(
                        "fence.proxy.async.shared::cta;"
                        ::: "memory"
                    );
                    __syncwarp();
                    if (lane == 0) {
                        ::kittens::tma::cluster::arrive(qdo_ready, 0);
                    }
                }
                if constexpr (LoaderOwnedDkQ) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        score_done,
                        phase
                    );
                    cta2_fused_dense_load_dk_q_wide_elected(
                        main.q.normal_wide,
                        g.q,
                        coord<cta2_fused_dense_q_normal_wide_tile>{
                            batch_idx,
                            q_tile_idx,
                            head_idx,
                            cta_rank
                        },
                        q_source_loaded
                    );
                }
                const bool has_next_q_tile =
                    q_tile_idx + q_tile_step < q_tile_count;
                const bool prefetch_next_owner =
                    PrefetchNextOwnerQdo && owner_pass == 0 && !has_next_q_tile;
                if (has_next_q_tile || prefetch_next_owner) {
                    if constexpr (DirectNextQdoDuringDqDrain) {
                        semaphore *prefetch_done = &qdo_prefetch_done;
                        if constexpr (PrefetchNextOwnerQdo) {
                            if (prefetch_next_owner) {
                                prefetch_done = &qdo_load_done;
                            }
                        }
                        if constexpr (PrearmNextQdoBeforeDkDone) {
                            cta2_fused_dense_arm_elected_transaction(
                                *prefetch_done,
                                sizeof(cta2_fused_dense_q_tile) +
                                    sizeof(cta2_fused_dense_do_tile) +
                                    (UseLongSeqStatsCache
                                        ? sizeof(sv_fl<128>)
                                        : 0)
                            );
                        }
                        if constexpr (PrefetchNextQdoAfterDkdv) {
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                dk_done,
                                phase
                            );
                        } else if constexpr (TimeoutDqWait) {
                            cta2_fused_dense_wait_timeout(dq_done, phase);
                        } else {
                            cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                                dq_done,
                                phase
                            );
                        }
                        int next_q_tile_idx = q_tile_idx + q_tile_step;
                        if constexpr (PrefetchNextOwnerQdo) {
                            if (prefetch_next_owner) {
                                const int next_owner_idx =
                                    owner_count - 1 - owner_pair_idx;
                                next_q_tile_idx = 2 * next_owner_idx;
                            }
                        }
                        if constexpr (GroupQdoTmaLoads) {
                            coord<cta2_fused_dense_q_tile>
                                next_q_tile_coord = {
                                    batch_idx,
                                    next_q_tile_idx * 2 + cta_rank,
                                    head_idx,
                                    0
                            };
                            if constexpr (PrearmNextQdoBeforeDkDone) {
                                cta2_fused_dense_load_qdo_elected_pair_no_arm(
                                    main.q.q,
                                    g.q,
                                    main.dout,
                                    g.dout,
                                    next_q_tile_coord,
                                    *prefetch_done
                                );
                            } else {
                                cta2_fused_dense_load_qdo_elected_pair(
                                    main.q.q,
                                    g.q,
                                    main.dout,
                                    g.dout,
                                    next_q_tile_coord,
                                    *prefetch_done,
                                    UseLongSeqStatsCache
                                        ? sizeof(sv_fl<128>)
                                        : 0
                                );
                            }
                        } else if (lane == 0) {
                            tma::expect_bytes(
                                *prefetch_done,
                                sizeof(main.q.q) + sizeof(main.dout) +
                                    (UseLongSeqStatsCache
                                        ? sizeof(sv_fl<128>)
                                        : 0)
                            );
                            coord<cta2_fused_dense_q_tile>
                                next_q_tile_coord = {
                                    batch_idx,
                                    next_q_tile_idx * 2 + cta_rank,
                                    head_idx,
                                    0
                                };
                            coord<cta2_fused_dense_do_tile>
                                next_do_tile_coord = {
                                    batch_idx,
                                    next_q_tile_idx * 2 + cta_rank,
                                    head_idx,
                                    0
                                };
                            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                                main.q.q,
                                g.q,
                                next_q_tile_coord,
                                *prefetch_done
                            );
                            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                                main.dout,
                                g.dout,
                                next_do_tile_coord,
                                *prefetch_done
                            );
                        }
                        if constexpr (UseLongSeqStatsCache) {
                            if (lane == 0) {
                                const int next_stats_stage =
                                    (loader_stats_iteration + 1) & 1;
                                auto &next_stats = next_stats_stage == 0
                                    ? storage.stats
                                    : storage.stats_next;
                                cta2_fused_dense_load_dpsum(
                                    g,
                                    next_stats,
                                    *prefetch_done,
                                    batch_idx,
                                    next_q_tile_idx,
                                    head_idx
                                );
                            }
                        }
                    } else {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            dkdv_done,
                            phase
                        );
                        auto &next_q =
                            storage.phase.dq.operands.exchange.next_q;
                        if (lane == 0) {
                            tma::expect_bytes(
                                qdo_prefetch_done,
                                sizeof(next_q) +
                                    sizeof(cta2_fused_dense_do_tile)
                            );
                            coord<cta2_fused_dense_q_tile>
                                next_q_tile_coord = {
                                    batch_idx,
                                    (q_tile_idx + q_tile_step) * 2 + cta_rank,
                                    head_idx,
                                    0
                                };
                            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                                next_q,
                                g.q,
                                next_q_tile_coord,
                                qdo_prefetch_done
                            );
                        }

                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            dq_done,
                            phase
                        );
                        auto &next_dout =
                            *reinterpret_cast<cta2_fused_dense_do_tile *>(
                                &storage.ds
                            );
                        if (lane == 0) {
                            coord<cta2_fused_dense_do_tile>
                                next_do_tile_coord = {
                                    batch_idx,
                                    (q_tile_idx + q_tile_step) * 2 + cta_rank,
                                    head_idx,
                                    0
                                };
                            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                                next_dout,
                                g.dout,
                                next_do_tile_coord,
                                qdo_prefetch_done
                            );
                        }
                    }
                }
                if constexpr (LoaderOwnedDkQ) {
                    current_qdo_done = &qdo_prefetch_done;
                    current_qdo_phase = local_phase;
                }
                cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                    dq_reduce_done,
                    phase
                );
                if constexpr (CarryAllRolePhases) {
                    loader_phase ^= 1;
                    loader_local_phase ^= 1;
                }
            }
        }
        return;
    }

    if (physical_warp == 14) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 104;" ::: "memory");
        uint32_t dq_operands_ready_cluster_address = 0;
        if constexpr (CacheRoleClusterAddresses) {
            if (lane == 0) {
                dq_operands_ready_cluster_address =
                    cta2_fused_dense_map_cluster_semaphore(
                        dq_operands_ready,
                        0
                    );
            }
        }
        #pragma unroll 1
        for (
            int owner_pass = 0;
            owner_pass < (SingleOwnerCluster ? 1 : 2);
            ++owner_pass
        ) {
            const int owner_idx = SingleOwnerCluster
                ? owner_pair_idx
                : (owner_pass == 0
                    ? owner_pair_idx
                    : owner_count - 1 - owner_pair_idx);
            const int owner_phase = OwnerQWorkSplitId >= 0
                ? (owner_pass == 0
                    ? 0
                    : ((owner_count - owner_pair_idx) & 1))
                : (SingleOwnerCluster || IntegrateCausalFrontier
                    ? 0
                    : (owner_pass & 1));
            const int first_dense_q_tile =
                2 * owner_idx + (IntegrateCausalFrontier ? 0 : 1) +
                (OwnerQWorkSplitId < 0 ? 0 : OwnerQWorkSplitId);
            int iteration = 0;
            int exchange_phase = owner_phase;
            for (
                int q_tile_idx = first_dense_q_tile;
                q_tile_idx < q_tile_count;
                q_tile_idx += q_tile_step, ++iteration
            ) {
                const int phase = CarryAllRolePhases
                    ? exchange_phase
                    : ((iteration + owner_phase) & 1);
                if constexpr (StageDqPeerBeforeDv) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        dq_peer_ready,
                        phase
                    );
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        dkdv_done,
                        phase
                    );
                } else {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        dq_local_ready,
                        phase
                    );
                }
                if (lane == 0) {
                    if constexpr (
                        !DirectAsyncPeerDs && !ProducerBulkPeerDs
                    ) {
                        auto &main = storage.phase.main;
                        auto &dq_operands = storage.phase.dq.operands;
                        auto &dq_a = [&]() -> cta2_fused_dense_dq_a_tile & {
                            if constexpr (StageDqAfterDv) {
                                return *reinterpret_cast<
                                    cta2_fused_dense_dq_a_tile *
                                >(&main.p.p);
                            } else {
                                return dq_operands.a;
                            }
                        }();
                        const int peer_rank = cta_rank ^ 1;
                        void *destination = reinterpret_cast<void *>(
                            &dq_a.data[cta_rank * 128 * 64]
                        );
                        ::kittens::tma::cluster::expect_bytes(
                            dq_exchange_done,
                            sizeof(storage.qdo_phase.ds),
                            peer_rank
                        );
                        ::kittens::tma::cluster::store_async(
                            destination,
                            reinterpret_cast<void *>(
                                &storage.qdo_phase.ds
                            ),
                            sizeof(storage.qdo_phase.ds),
                            peer_rank,
                            dq_exchange_done
                        );
                    }
                }
                ::kittens::tma::cluster::wait(
                    dq_exchange_done,
                    phase
                );
                if constexpr (StageDqPeerBeforeDv) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        dq_local_ready,
                        phase
                    );
                }
                if (lane == 0) {
                    if constexpr (CacheRoleClusterAddresses) {
                        cta2_fused_dense_cluster_arrive_mapped(
                            dq_operands_ready_cluster_address
                        );
                    } else {
                        ::kittens::tma::cluster::arrive(
                            dq_operands_ready,
                            0
                        );
                    }
                }
                if constexpr (CarryAllRolePhases) {
                    exchange_phase ^= 1;
                }
            }
        }
        return;
    }

    if (physical_warp == 15) {
        if constexpr (UseX32TmemComputeLayout) {
            asm volatile("setmaxnreg.dec.sync.aligned.u32 104;" ::: "memory");
            bool have_previous_stats = false;
            int previous_stats_phase = 0;
            #pragma unroll 1
            for (int owner_pass = 0; owner_pass < 2; ++owner_pass) {
                const int owner_idx = owner_pass == 0
                    ? owner_pair_idx
                    : owner_count - 1 - owner_pair_idx;
                const int first_dense_q_tile = 2 * owner_idx;
                #pragma unroll 1
                for (
                    int q_tile_idx = first_dense_q_tile;
                    q_tile_idx < q_tile_count;
                    ++q_tile_idx
                ) {
                    const int phase =
                        (q_tile_idx - first_dense_q_tile) & 1;
                    if (have_previous_stats) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            storage.stats_consumed,
                            previous_stats_phase
                        );
                    }
                    const size_t offset =
                        (static_cast<size_t>(batch_idx) *
                                g.lse_log2.depth() +
                            head_idx) *
                            g.seq_len +
                        q_tile_idx * 128;
                    #pragma unroll
                    for (
                        int element = lane;
                        element < 128;
                        element += 32
                    ) {
                        storage.stats.lse_log2[element] =
                            g.lse_log2.raw_ptr[offset + element];
                        storage.stats.dpsum[element] =
                            g.dpsum.raw_ptr[offset + element];
                    }
                    asm volatile(
                        "fence.proxy.async.shared::cta;"
                        ::: "memory"
                    );
                    __syncwarp();
                    asm volatile("bar.sync 5, 320;" ::: "memory");
                    have_previous_stats = true;
                    previous_stats_phase = phase;
                }
            }
            return;
        } else if constexpr (UseWarpStatsCache) {
            asm volatile("setmaxnreg.dec.sync.aligned.u32 104;" ::: "memory");
            const int first_dense_q_tile =
                2 * owner_pair_idx + (IntegrateCausalFrontier ? 0 : 1);
            int iteration = 0;
            #pragma unroll 1
            for (
                int q_tile_idx = first_dense_q_tile;
                q_tile_idx < q_tile_count;
                ++q_tile_idx, ++iteration
            ) {
                const int phase = iteration & 1;
                if (iteration != 0) {
                    if constexpr (!PipelineLsePrefetch) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            storage.stats_consumed,
                            phase ^ 1
                        );
                    }
                }
                const size_t offset =
                    (static_cast<size_t>(batch_idx) *
                            g.lse_log2.depth() +
                        head_idx) *
                            g.seq_len +
                    q_tile_idx * 128;
                if constexpr (PipelineLsePrefetch) {
                    auto &lse_stage = phase == 0
                        ? storage.stats.lse_log2
                        : storage.lse_log2_next;
                    #pragma unroll
                    for (int element = lane; element < 128; element += 32) {
                        lse_stage[element] =
                            g.lse_log2.raw_ptr[offset + element];
                    }
                    if (iteration != 0) {
                        cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                            storage.stats_consumed,
                            phase ^ 1
                        );
                    }
                    #pragma unroll
                    for (int element = lane; element < 128; element += 32) {
                        storage.stats.dpsum[element] =
                            g.dpsum.raw_ptr[offset + element];
                    }
                } else {
                    #pragma unroll
                    for (int element = lane; element < 128; element += 32) {
                        storage.stats.lse_log2[element] =
                            g.lse_log2.raw_ptr[offset + element];
                        storage.stats.dpsum[element] =
                            g.dpsum.raw_ptr[offset + element];
                    }
                }
                asm volatile(
                    "fence.proxy.async.shared::cta;"
                    ::: "memory"
                );
                __syncwarp();
                warp::arrive(storage.stats_ready);
                if constexpr (UseStatsWarpScoreFanout) {
                    cta2_fused_dense_role_wait<TimeoutAllRoleWaits>(
                        score_done,
                        phase
                    );
                    asm volatile("bar.sync 5, 288;" ::: "memory");
                }
            }
            return;
        }
    }

    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" ::: "memory");
}

}  // namespace detail

inline void launch_backward_cta2_fused_dense_with_frontier(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream,
    cudaStream_t frontier_stream,
    cudaEvent_t dense_ready,
    cudaEvent_t frontier_done,
    at::Tensor &frontier_dk,
    at::Tensor &frontier_dv,
    bool use_tmem_ds = true,
    bool overlap_ds_exchange = true,
    bool overlap_q_with_dp = true,
    bool use_tmem_p = false,
    bool overlap_do_with_dp = false,
    bool use_dp_operand_ready_mbar = false,
    bool use_dq_operand_ready_mbar = false,
    bool overlap_dv_with_ds = false,
    bool pipeline_next_score = false,
    bool preload_dq_a = false,
    bool use_score_operand_ready_mbar = false,
    bool use_ds_warp_multicast_mbar = false,
    bool use_role_split = false,
    bool retain_ds_exchange = false,
    bool retain_ds_local = false,
    bool use_normal_do_dv = false,
    bool use_tma_score_k = false,
    bool direct_next_qdo_during_dq_drain = false,
    bool single_owner_cluster = false,
    bool use_fast_exp2 = false,
    bool use_warp_stats_cache = false,
    bool pipeline_lse_prefetch = false,
    bool use_direct_stats_loads = false,
    bool split_dv_dk_ready = false,
    bool stage_dq_after_dv = false,
    bool stage_dq_peer_before_dv = false,
    bool use_wide_dk_n192 = false,
    bool direct_ds_half_store = false,
    bool asymmetric_dv_publish = false
) {
    using DenseG = detail::cta2_fused_dense_globals;
    using FrontierC = dense_tmem_frontier_config<
        kForwardTileM,
        kForwardTileN,
        kB300QKDim,
        kB300VDim
    >;
    using FrontierG = dense_tmem_frontier_globals<FrontierC>;

    TORCH_CHECK(q.size(0) == 1, "2-CTA fused dense route requires batch size 1");
    TORCH_CHECK(
        q.size(1) == 4096 || q.size(1) == 8192 || q.size(1) == 16384,
        "2-CTA fused dense route requires S4096, S8192, or S16384"
    );
    TORCH_CHECK(
        q.size(1) % 256 == 0,
        "2-CTA fused dense route requires sequence divisible by 256"
    );
    TORCH_CHECK(
        frontier_dk.sizes() == dk.sizes() && frontier_dv.sizes() == dv.sizes(),
        "2-CTA fused dense frontier scratch shape mismatch"
    );

    DenseG dense_g{
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_k_gl>(k),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_q_gl>(q),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_v_gl>(v),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_do_gl>(dout),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_output_gl>(dk),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_output_gl>(dv),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_dq_gl>(dq),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_stats_gl>(
            lse_log2,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_stats_gl>(
            dpsum,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
    };

    FrontierG frontier_g{
        kittens::py::tensor_to_gl<typename FrontierG::q_gl>(q),
        kittens::py::tensor_to_gl<typename FrontierG::k_gl>(k),
        kittens::py::tensor_to_gl<typename FrontierG::v_gl>(v),
        kittens::py::tensor_to_gl<typename FrontierG::do_gl>(dout),
        kittens::py::tensor_to_gl<typename FrontierG::dk_gl>(frontier_dk),
        kittens::py::tensor_to_gl<typename FrontierG::dv_gl>(frontier_dv),
        kittens::py::tensor_to_gl<typename FrontierG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename FrontierG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename FrontierG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename FrontierG::stats_gl>(
            lse_log2,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        kittens::py::tensor_to_gl<typename FrontierG::stats_gl>(
            dpsum,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(0)),
    };

    const int owner_clusters = static_cast<int>(q.size(1) / 256);
    TORCH_CHECK(
        owner_clusters % 2 == 0,
        "2-CTA fused dense paired ownership requires an even owner count"
    );
    const int owner_pair_clusters = owner_clusters / 2;
    const int dense_owner_clusters = single_owner_cluster
        ? owner_clusters
        : owner_pair_clusters;
    kittens::LaunchConfig<true, false> dense_config(
        dim3(
            dense_owner_clusters * 2,
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(0))
        ),
        dim3(use_role_split ? 512 : 256, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );

    CUDACHECK(cudaEventRecord(dense_ready, stream));
    CUDACHECK(cudaStreamWaitEvent(frontier_stream, dense_ready));
    if (use_role_split) {
        if (q.size(1) >= 8192) {
            if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && !single_owner_cluster &&
                use_fast_exp2 && !use_warp_stats_cache &&
                !pipeline_lse_prefetch && use_direct_stats_loads &&
                split_dv_dk_ready && stage_dq_after_dv &&
                !stage_dq_peer_before_dv && !use_wide_dk_n192 &&
                direct_ds_half_store && asymmetric_dv_publish
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        false,
                        true,
                        false,
                        false,
                        true,
                        true,
                        true,
                        false,
                        false,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && !single_owner_cluster &&
                use_fast_exp2 && !use_warp_stats_cache &&
                !pipeline_lse_prefetch && use_direct_stats_loads &&
                split_dv_dk_ready && stage_dq_after_dv &&
                !stage_dq_peer_before_dv && !use_wide_dk_n192 &&
                direct_ds_half_store && !asymmetric_dv_publish
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        false,
                        true,
                        false,
                        false,
                        true,
                        true,
                        true,
                        false,
                        false,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && !single_owner_cluster &&
                !use_fast_exp2 && !use_warp_stats_cache &&
                !pipeline_lse_prefetch && use_direct_stats_loads &&
                split_dv_dk_ready && stage_dq_after_dv &&
                !stage_dq_peer_before_dv && !use_wide_dk_n192 &&
                direct_ds_half_store
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        false,
                        false,
                        false,
                        false,
                        true,
                        true,
                        true,
                        false,
                        false,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && !single_owner_cluster &&
                !use_fast_exp2 && !use_warp_stats_cache &&
                !pipeline_lse_prefetch && use_direct_stats_loads &&
                split_dv_dk_ready && stage_dq_after_dv &&
                !stage_dq_peer_before_dv && !use_wide_dk_n192 &&
                !direct_ds_half_store
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        false,
                        false,
                        false,
                        false,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && !single_owner_cluster &&
                !use_fast_exp2 && !use_warp_stats_cache &&
                !pipeline_lse_prefetch && !use_direct_stats_loads &&
                split_dv_dk_ready && stage_dq_after_dv &&
                !stage_dq_peer_before_dv && !use_wide_dk_n192
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        false,
                        false,
                        false,
                        false,
                        false,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster &&
                use_fast_exp2 && use_warp_stats_cache &&
                pipeline_lse_prefetch && use_direct_stats_loads &&
                split_dv_dk_ready && stage_dq_after_dv &&
                stage_dq_peer_before_dv && use_wide_dk_n192
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster &&
                use_fast_exp2 && use_warp_stats_cache &&
                pipeline_lse_prefetch && use_direct_stats_loads &&
                split_dv_dk_ready && stage_dq_after_dv &&
                stage_dq_peer_before_dv
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster &&
                use_fast_exp2 && use_warp_stats_cache &&
                pipeline_lse_prefetch && use_direct_stats_loads &&
                split_dv_dk_ready && stage_dq_after_dv
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster &&
                use_fast_exp2 && use_warp_stats_cache &&
                pipeline_lse_prefetch && use_direct_stats_loads &&
                split_dv_dk_ready
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster &&
                use_fast_exp2 && use_warp_stats_cache &&
                pipeline_lse_prefetch && use_direct_stats_loads
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster &&
                use_fast_exp2 && use_warp_stats_cache &&
                pipeline_lse_prefetch
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster &&
                use_fast_exp2 && use_warp_stats_cache
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster &&
                use_fast_exp2
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain && single_owner_cluster
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (
                use_normal_do_dv && use_tma_score_k &&
                direct_next_qdo_during_dq_drain
            ) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (use_normal_do_dv && use_tma_score_k) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (use_normal_do_dv) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (retain_ds_local) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true,
                        true
                    >,
                    dense_g
                ));
            } else if (retain_ds_exchange) {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<
                        true,
                        true
                    >,
                    dense_g
                ));
            } else {
                CUDACHECK(cudaLaunchKernelEx(
                    dense_config,
                    detail::main_kernel_causal_cta2_fused_dense_role_split<true>,
                    dense_g
                ));
            }
        } else {
            CUDACHECK(cudaLaunchKernelEx(
                dense_config,
                detail::main_kernel_causal_cta2_fused_dense_role_split<false>,
                dense_g
            ));
        }
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p && overlap_do_with_dp &&
        use_dp_operand_ready_mbar && use_dq_operand_ready_mbar &&
        overlap_dv_with_ds && pipeline_next_score && preload_dq_a &&
        use_score_operand_ready_mbar && use_ds_warp_multicast_mbar
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p && overlap_do_with_dp &&
        use_dp_operand_ready_mbar && use_dq_operand_ready_mbar &&
        overlap_dv_with_ds && pipeline_next_score && preload_dq_a &&
        use_score_operand_ready_mbar
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p && overlap_do_with_dp &&
        use_dp_operand_ready_mbar && use_dq_operand_ready_mbar &&
        overlap_dv_with_ds && pipeline_next_score && preload_dq_a
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p && overlap_do_with_dp &&
        use_dp_operand_ready_mbar && use_dq_operand_ready_mbar &&
        overlap_dv_with_ds && pipeline_next_score
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p && overlap_do_with_dp &&
        use_dp_operand_ready_mbar && use_dq_operand_ready_mbar &&
        overlap_dv_with_ds
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p && overlap_do_with_dp &&
        use_dp_operand_ready_mbar && use_dq_operand_ready_mbar
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p && overlap_do_with_dp &&
        use_dp_operand_ready_mbar
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p && overlap_do_with_dp
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp && use_tmem_p
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange &&
        overlap_q_with_dp
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (q.size(1) == 8192 && use_tmem_ds && overlap_ds_exchange) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                true,
                false
            >,
            dense_g
        ));
    } else if (q.size(1) == 8192 && use_tmem_ds) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                true,
                false,
                false
            >,
            dense_g
        ));
    } else if (q.size(1) == 8192) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                true,
                false,
                false,
                false
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp &&
        use_tmem_p && overlap_do_with_dp && use_dp_operand_ready_mbar &&
        use_dq_operand_ready_mbar && overlap_dv_with_ds &&
        pipeline_next_score && preload_dq_a && use_score_operand_ready_mbar &&
        use_ds_warp_multicast_mbar
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp &&
        use_tmem_p && overlap_do_with_dp && use_dp_operand_ready_mbar &&
        use_dq_operand_ready_mbar && overlap_dv_with_ds &&
        pipeline_next_score && preload_dq_a && use_score_operand_ready_mbar
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp &&
        use_tmem_p && overlap_do_with_dp && use_dp_operand_ready_mbar &&
        use_dq_operand_ready_mbar && overlap_dv_with_ds &&
        pipeline_next_score && preload_dq_a
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp &&
        use_tmem_p && overlap_do_with_dp && use_dp_operand_ready_mbar &&
        use_dq_operand_ready_mbar && overlap_dv_with_ds &&
        pipeline_next_score
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp &&
        use_tmem_p && overlap_do_with_dp && use_dp_operand_ready_mbar &&
        use_dq_operand_ready_mbar && overlap_dv_with_ds
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp &&
        use_tmem_p && overlap_do_with_dp && use_dp_operand_ready_mbar &&
        use_dq_operand_ready_mbar
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp &&
        use_tmem_p && overlap_do_with_dp && use_dp_operand_ready_mbar
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp &&
        use_tmem_p && overlap_do_with_dp
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (
        use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp && use_tmem_p
    ) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (use_tmem_ds && overlap_ds_exchange && overlap_q_with_dp) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                true
            >,
            dense_g
        ));
    } else if (use_tmem_ds && overlap_ds_exchange) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                true,
                false
            >,
            dense_g
        ));
    } else if (use_tmem_ds) {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                true,
                false,
                false
            >,
            dense_g
        ));
    } else {
        CUDACHECK(cudaLaunchKernelEx(
            dense_config,
            detail::main_kernel_causal_cta2_fused_dense<
                false,
                false,
                false,
                false
            >,
            dense_g
        ));
    }

    const int frontier_grid_x = static_cast<int>(q.size(1) / 128);
    kittens::LaunchConfig<true, false> frontier_config(
        dim3(
            frontier_grid_x,
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(0))
        ),
        dim3(FrontierC::FusedDqLoadOverlapBlockThreads, 1, 1),
        0,
        frontier_stream,
        dim3(FrontierC::ClusterSize, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        frontier_config,
        detail::main_kernel_causal_dense_tmem_dkdv<
            FrontierC,
            1,
            true,
            true,
            true,
            false,
            true,
            false,
            true,
            true
        >,
        frontier_g,
        frontier_g.dk,
        frontier_g.dv
    ));
    CUDACHECK(cudaEventRecord(frontier_done, frontier_stream));
    CUDACHECK(cudaStreamWaitEvent(stream, frontier_done));

    constexpr int kThreads = 256;
    TORCH_CHECK(
        dk.numel() % 4 == 0 && dv.numel() % 4 == 0,
        "2-CTA fused dense frontier add requires float4-aligned shapes"
    );
    const int64_t dk_vecs = dk.numel() / 4;
    const int64_t dv_vecs = dv.numel() / 4;
    const int64_t max_vecs = dk_vecs > dv_vecs ? dk_vecs : dv_vecs;
    const int blocks = static_cast<int>((max_vecs + kThreads - 1) / kThreads);
    detail::dense_tmem_frontier_add_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4 *>(dk.data_ptr()),
        reinterpret_cast<const float4 *>(frontier_dk.data_ptr()),
        dk_vecs,
        reinterpret_cast<float4 *>(dv.data_ptr()),
        reinterpret_cast<const float4 *>(frontier_dv.data_ptr()),
        dv_vecs
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <
    bool CoalescedBf16Store = false,
    bool DirectAsyncPeerDs = false,
    bool ProducerBulkPeerDs = false,
    bool ProducerBulkPeerDsCtaFenceOnly = false,
    bool DqReadHandoffBeforeCompletion = false,
    bool AggregateScoreConsumed = false,
    bool DirectTmaDkQ = false,
    bool TimeoutDqWait = false,
    bool UseWideDkN192 = false,
    bool TimeoutAllRoleWaits = false,
    bool UseNamedDoSourceBarrier = false,
    bool UseComputeScoreFanout = false,
    bool UseRuntimeAccumulationPredicate = false,
    bool UseReducerDqFanout = false,
    bool UseReducerDqLeaderArrive = false,
    bool MergeScoreDpReady = false,
    bool WideCoalescedBf16Store = false,
    bool BulkPeerDsFromFullTile = false,
    bool CoalescedPeerDsBulk = false,
    bool WideDqKGlobalToShared = false,
    bool IntegrateCausalFrontier = false,
    int ExactQTileCount = 0,
    bool FenceDsBeforeDkdvReady = false,
    bool UseNamedDkdvLocalFanIn = false,
    bool LeaderOnlyQdoPublishFence = false,
    bool CacheQdoReadyClusterAddress = false,
    bool GroupQdoTmaLoads = false,
    bool ElectedWideDkQTmaLoad = false,
    bool ElectedPeerDoTmaLoad = false,
    bool CacheRoleClusterAddresses = false,
    bool CacheTensorCommitAddresses = false,
    bool EnsureReducerOutputDrain = false,
    bool ElectedScoreKTmaLoad = false,
    bool UseExactClusterCoordinates = false,
    bool EnforceDpTmemConsumerRelease = false,
    bool SplitDpTmemConsumerRelease = false,
    bool UseIterationCausalMask = false,
    bool UseFusedTmemPAndDs = false,
    bool OverlapFusedDqAPublication = false,
    bool PrefetchNextQdoAfterDkdv = false,
    bool PrefetchNextOwnerQdo = false,
    bool UseFusedTmemRuntimeAccumulationPredicate = false,
    bool UseBitwisePExpansion = false,
    bool UseFusedExp2Pack = false,
    bool PeelCausalPrefix = false,
    bool BranchlessDoSourceLoad = false,
    bool BranchlessDoSourceBaseSelect = false,
    bool PublishVOncePerOwner = false,
    bool BulkDoDvStage = false,
    bool LoaderOwnedDkQ = false,
    bool FuseScoreScaleLse = false,
    bool RetainPackedP = false,
    bool SplitDirectDpsumAcrossDpDoneWait = false,
    bool FusedExp2Fragment4First = false,
    bool CarryDirectStatsOffset = false,
    bool CarryAllRolePhases = false,
    bool UseExactDefaultScaleLog2e = false,
    bool ReverseDkTailTmemLoadIssue = false,
    bool PrearmNextQdoBeforeDkDone = false,
    bool UseX32TmemComputeLayout = false,
    bool UseLongSeqStatsCache = false,
    bool UseCompactScoreMma = false,
    bool UseCompactDpMma = false,
    bool UsePackedBf16DsProduct = false,
    bool SplitDqTmemAndSharedHandoff = false,
    bool DistributedDqSharedReadWait = false,
    bool BalancedSingleOwnerSchedule = false,
    bool UseSingleOwnerWarpStatsCache = false,
    bool CacheDqStageLanePointers = false,
    bool UseSlicedFp32PForDs = false,
    bool UseTmaVWithScoreK = false,
    bool UseStatsWarpScoreFanout = false,
    bool UseBatchedDqTmemLoads = false,
    bool UseDynamicDpReleaseBarrierId = false,
    bool PreissueFirstDpHalfBeforeQdoWait = false,
    bool OverlapSecondDpLoadWithReleaseBarrier = false
>
inline void launch_backward_cta2_fused_dense_bf16_with_frontier(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream,
    cudaStream_t frontier_stream,
    cudaEvent_t dense_ready,
    cudaEvent_t frontier_done,
    at::Tensor &frontier_dk,
    at::Tensor &frontier_dv
) {
    static_assert(
        !UseSingleOwnerWarpStatsCache ||
            (BalancedSingleOwnerSchedule && !UseLongSeqStatsCache)
    );
    using DenseG = detail::cta2_fused_dense_globals_t<bf16, DirectTmaDkQ>;
    using FrontierC = dense_tmem_frontier_config<
        kForwardTileM,
        kForwardTileN,
        kB300QKDim,
        kB300VDim
    >;
    using FrontierG = dense_tmem_frontier_globals<FrontierC>;

    const bool supported_shape =
        q.size(0) == 1 &&
        ((q.size(1) == 2048 && q.size(2) == 8) ||
         (q.size(1) == 4096 &&
          (q.size(2) == 4 || q.size(2) == 8)) ||
         (q.size(1) == 8192 &&
          (q.size(2) == 2 || q.size(2) == 4 || q.size(2) == 8 ||
           q.size(2) == 16)) ||
         (q.size(1) == 16384 &&
          (q.size(2) == 4 || q.size(2) == 8 || q.size(2) == 16 ||
           q.size(2) == 32 || q.size(2) == 64 || q.size(2) == 128)) ||
         (q.size(1) == 32768 &&
          (q.size(2) == 16 || q.size(2) == 32 || q.size(2) == 64 ||
           q.size(2) == 128)) ||
         (q.size(1) == 65536 &&
          (q.size(2) == 16 || q.size(2) == 32 || q.size(2) == 64 ||
           q.size(2) == 128)));
    TORCH_CHECK(
        supported_shape,
        "BF16 dK/dV 2-CTA route requires B1 S2048 H8, S4096 H4/H8, S8192 H2/H4/H8/H16, S16384 H4/H8/H16/H32/H64/H128, "
        "S32768 H16/H32/H64/H128, or S65536 H16/H32/H64/H128"
    );
    TORCH_CHECK(
        dk.scalar_type() == at::ScalarType::BFloat16 &&
            dv.scalar_type() == at::ScalarType::BFloat16,
        "BF16 dK/dV 2-CTA route requires BF16 dK and dV outputs"
    );
    TORCH_CHECK(
        frontier_dk.sizes() == dk.sizes() && frontier_dv.sizes() == dv.sizes(),
        "BF16 dK/dV 2-CTA frontier scratch shape mismatch"
    );

    DenseG dense_g{
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_k_gl>(k),
        kittens::py::tensor_to_gl<typename DenseG::q_gl>(q),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_v_gl>(v),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_do_gl>(dout),
        kittens::py::tensor_to_gl<
            detail::cta2_fused_dense_output_gl_t<bf16>
        >(dk),
        kittens::py::tensor_to_gl<
            detail::cta2_fused_dense_output_gl_t<bf16>
        >(dv),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_dq_gl>(dq),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_stats_gl>(
            lse_log2,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        kittens::py::tensor_to_gl<detail::cta2_fused_dense_stats_gl>(
            dpsum,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
    };

    FrontierG frontier_g{
        kittens::py::tensor_to_gl<typename FrontierG::q_gl>(q),
        kittens::py::tensor_to_gl<typename FrontierG::k_gl>(k),
        kittens::py::tensor_to_gl<typename FrontierG::v_gl>(v),
        kittens::py::tensor_to_gl<typename FrontierG::do_gl>(dout),
        kittens::py::tensor_to_gl<typename FrontierG::dk_gl>(frontier_dk),
        kittens::py::tensor_to_gl<typename FrontierG::dv_gl>(frontier_dv),
        kittens::py::tensor_to_gl<typename FrontierG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename FrontierG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename FrontierG::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename FrontierG::stats_gl>(
            lse_log2,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        kittens::py::tensor_to_gl<typename FrontierG::stats_gl>(
            dpsum,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(0)),
    };

    const int owner_clusters = static_cast<int>(q.size(1) / 256);
    const int dense_cluster_count = BalancedSingleOwnerSchedule
        ? owner_clusters
        : owner_clusters / 2;
    kittens::LaunchConfig<true, false> dense_config(
        dim3(dense_cluster_count * 2, q.size(2), q.size(0)),
        dim3(512, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );

    CUDACHECK(cudaEventRecord(dense_ready, stream));
    CUDACHECK(cudaStreamWaitEvent(frontier_stream, dense_ready));
    CUDACHECK(cudaLaunchKernelEx(
        dense_config,
        detail::main_kernel_causal_cta2_fused_dense_role_split<
            true,
            true,
            true,
            true,
            true,
            true,
            BalancedSingleOwnerSchedule,
            true,
            UseSingleOwnerWarpStatsCache,
            UseSingleOwnerWarpStatsCache,
            true,
            true,
            true,
            false,
            UseWideDkN192,
            true,
            true,
            bf16,
            CoalescedBf16Store,
            DirectAsyncPeerDs,
            ProducerBulkPeerDs,
            ProducerBulkPeerDsCtaFenceOnly,
            DqReadHandoffBeforeCompletion,
            AggregateScoreConsumed,
            DirectTmaDkQ,
            TimeoutDqWait,
            TimeoutAllRoleWaits,
            UseNamedDoSourceBarrier,
            UseComputeScoreFanout,
            UseRuntimeAccumulationPredicate,
            UseReducerDqFanout,
            UseReducerDqLeaderArrive,
            MergeScoreDpReady,
            WideCoalescedBf16Store,
            BulkPeerDsFromFullTile,
            CoalescedPeerDsBulk,
            WideDqKGlobalToShared,
            IntegrateCausalFrontier,
            ExactQTileCount,
            FenceDsBeforeDkdvReady,
            UseNamedDkdvLocalFanIn,
            LeaderOnlyQdoPublishFence,
            CacheQdoReadyClusterAddress,
            GroupQdoTmaLoads,
            ElectedWideDkQTmaLoad,
            ElectedPeerDoTmaLoad,
            CacheRoleClusterAddresses,
            CacheTensorCommitAddresses,
            EnsureReducerOutputDrain,
            ElectedScoreKTmaLoad,
            UseExactClusterCoordinates,
            EnforceDpTmemConsumerRelease,
            SplitDpTmemConsumerRelease,
            UseIterationCausalMask,
            UseFusedTmemPAndDs,
            OverlapFusedDqAPublication,
            PrefetchNextQdoAfterDkdv,
            PrefetchNextOwnerQdo && !BalancedSingleOwnerSchedule,
            UseFusedTmemRuntimeAccumulationPredicate,
            UseBitwisePExpansion,
            UseFusedExp2Pack,
            PeelCausalPrefix,
            BranchlessDoSourceLoad,
            BranchlessDoSourceBaseSelect,
            PublishVOncePerOwner,
            -1,
            BulkDoDvStage,
            LoaderOwnedDkQ,
            FuseScoreScaleLse,
            RetainPackedP,
            SplitDirectDpsumAcrossDpDoneWait &&
                !UseSingleOwnerWarpStatsCache,
            FusedExp2Fragment4First,
            CarryDirectStatsOffset && !UseSingleOwnerWarpStatsCache,
            CarryAllRolePhases,
            UseExactDefaultScaleLog2e,
            ReverseDkTailTmemLoadIssue,
            PrearmNextQdoBeforeDkDone,
            UseX32TmemComputeLayout,
            UseLongSeqStatsCache,
            UseCompactScoreMma,
            UseCompactDpMma,
            UsePackedBf16DsProduct,
            SplitDqTmemAndSharedHandoff,
            DistributedDqSharedReadWait,
            CacheDqStageLanePointers,
            UseSlicedFp32PForDs,
            UseTmaVWithScoreK,
            UseStatsWarpScoreFanout,
            UseBatchedDqTmemLoads,
            UseDynamicDpReleaseBarrierId,
            PreissueFirstDpHalfBeforeQdoWait,
            OverlapSecondDpLoadWithReleaseBarrier
        >,
        dense_g
    ));

    if constexpr (IntegrateCausalFrontier) {
        return;
    }

    const int frontier_grid_x = static_cast<int>(q.size(1) / 128);
    kittens::LaunchConfig<true, false> frontier_config(
        dim3(frontier_grid_x, q.size(2), q.size(0)),
        dim3(FrontierC::FusedDqLoadOverlapBlockThreads, 1, 1),
        0,
        frontier_stream,
        dim3(FrontierC::ClusterSize, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        frontier_config,
        detail::main_kernel_causal_dense_tmem_dkdv<
            FrontierC,
            1,
            true,
            true,
            true,
            false,
            true,
            false,
            true,
            true
        >,
        frontier_g,
        frontier_g.dk,
        frontier_g.dv
    ));
    CUDACHECK(cudaEventRecord(frontier_done, frontier_stream));
    CUDACHECK(cudaStreamWaitEvent(stream, frontier_done));

    constexpr int kThreads = 256;
    TORCH_CHECK(
        dk.numel() % 4 == 0 && dv.numel() % 4 == 0,
        "BF16 dK/dV frontier add requires four-element alignment"
    );
    const int64_t dk_vecs = dk.numel() / 4;
    const int64_t dv_vecs = dv.numel() / 4;
    const int64_t max_vecs = dk_vecs > dv_vecs ? dk_vecs : dv_vecs;
    const int blocks = static_cast<int>((max_vecs + kThreads - 1) / kThreads);
    detail::dense_tmem_frontier_add_bf16_kernel<<<
        blocks,
        kThreads,
        0,
        stream
    >>>(
        reinterpret_cast<bf16_2 *>(dk.data_ptr()),
        reinterpret_cast<const float4 *>(frontier_dk.data_ptr()),
        dk_vecs,
        reinterpret_cast<bf16_2 *>(dv.data_ptr()),
        reinterpret_cast<const float4 *>(frontier_dv.data_ptr()),
        dv_vecs
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <int ExactQTileCount, int ExpectedHeads>
inline void launch_backward_cta2_fused_dense_owner_q_split(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &partial0_dk,
    at::Tensor &partial0_dv,
    at::Tensor &partial1_dk,
    at::Tensor &partial1_dv,
    float scale,
    cudaStream_t stream,
    cudaStream_t split_stream,
    cudaEvent_t inputs_ready,
    cudaEvent_t split_done
) {
    static_assert(
        ExactQTileCount == 16 || ExactQTileCount == 32 ||
        ExactQTileCount == 64
    );
    static_assert(ExpectedHeads > 0);
    TORCH_CHECK(
        q.size(0) == 1 &&
            q.size(1) == ExactQTileCount * 128 &&
            q.size(2) == ExpectedHeads,
        "owner-Q split route received an unsupported exact shape"
    );
    TORCH_CHECK(
        stream != split_stream,
        "owner-Q split route requires a distinct auxiliary stream"
    );
    TORCH_CHECK(
        dk.scalar_type() == at::kBFloat16 && dv.scalar_type() == at::kBFloat16,
        "owner-Q split final dK/dV must be BF16"
    );
    TORCH_CHECK(
        partial0_dk.scalar_type() == at::kFloat &&
            partial0_dv.scalar_type() == at::kFloat &&
            partial1_dk.scalar_type() == at::kFloat &&
            partial1_dv.scalar_type() == at::kFloat,
        "owner-Q split partial dK/dV must be FP32"
    );
    TORCH_CHECK(
        partial0_dk.sizes() == dk.sizes() &&
            partial1_dk.sizes() == dk.sizes() &&
            partial0_dv.sizes() == dv.sizes() &&
            partial1_dv.sizes() == dv.sizes(),
        "owner-Q split partial shapes must match final dK/dV"
    );

    using DenseG = detail::cta2_fused_dense_globals_t<float, true>;
    auto make_globals = [&](at::Tensor &partial_dk, at::Tensor &partial_dv) {
        return DenseG{
            kittens::py::tensor_to_gl<detail::cta2_fused_dense_k_gl>(k),
            kittens::py::tensor_to_gl<typename DenseG::q_gl>(q),
            kittens::py::tensor_to_gl<detail::cta2_fused_dense_v_gl>(v),
            kittens::py::tensor_to_gl<detail::cta2_fused_dense_do_gl>(dout),
            kittens::py::tensor_to_gl<
                detail::cta2_fused_dense_output_gl_t<float>
            >(partial_dk),
            kittens::py::tensor_to_gl<
                detail::cta2_fused_dense_output_gl_t<float>
            >(partial_dv),
            kittens::py::tensor_to_gl<detail::cta2_fused_dense_dq_gl>(dq),
            kittens::py::tensor_to_gl<detail::cta2_fused_dense_stats_gl>(
                lse_log2,
                q.size(0),
                q.size(2),
                1,
                q.size(1)
            ),
            kittens::py::tensor_to_gl<detail::cta2_fused_dense_stats_gl>(
                dpsum,
                q.size(0),
                q.size(2),
                1,
                q.size(1)
            ),
            scale,
            scale * kLog2E,
            static_cast<int>(q.size(1)),
        };
    };
    DenseG split0_g = make_globals(partial0_dk, partial0_dv);
    DenseG split1_g = make_globals(partial1_dk, partial1_dv);

    constexpr int owner_pair_clusters = ExactQTileCount / 4;
    kittens::LaunchConfig<true, false> split0_config(
        dim3(owner_pair_clusters * 2, q.size(2), q.size(0)),
        dim3(512, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    kittens::LaunchConfig<true, false> split1_config(
        dim3(owner_pair_clusters * 2, q.size(2), q.size(0)),
        dim3(512, 1, 1),
        0,
        split_stream,
        dim3(2, 1, 1)
    );

    CUDACHECK(cudaEventRecord(inputs_ready, stream));
    CUDACHECK(cudaStreamWaitEvent(split_stream, inputs_ready));
    CUDACHECK(cudaLaunchKernelEx(
        split0_config,
        detail::main_kernel_causal_cta2_fused_dense_role_split<
            true, true, true, true, true, true, false, true, false, false,
            true, true, true, false, true, true, true, float,
            false, false, true, true, true, true, true, true, true, true,
            true, true, true, false, true, false, true, true, false, true,
            ExactQTileCount, true, true, false, false, false, false, true,
            false, false,
            false, false, true, true, true, true, true, true, true, false,
            true, true, false, false, true, true, true, 0
        >,
        split0_g
    ));
    CUDACHECK(cudaLaunchKernelEx(
        split1_config,
        detail::main_kernel_causal_cta2_fused_dense_role_split<
            true, true, true, true, true, true, false, true, false, false,
            true, true, true, false, true, true, true, float,
            false, false, true, true, true, true, true, true, true, true,
            true, true, true, false, true, false, true, true, false, true,
            ExactQTileCount, true, true, false, false, false, false, true,
            false, false,
            false, false, true, true, true, true, true, true, true, false,
            true, true, false, false, true, true, true, 1
        >,
        split1_g
    ));
    CUDACHECK(cudaEventRecord(split_done, split_stream));
    CUDACHECK(cudaStreamWaitEvent(stream, split_done));

    constexpr int kThreads = 256;
    TORCH_CHECK(
        dk.numel() % 4 == 0 && dv.numel() % 4 == 0,
        "owner-Q split merge requires four-element alignment"
    );
    const int64_t dk_vecs = dk.numel() / 4;
    const int64_t dv_vecs = dv.numel() / 4;
    const int64_t max_vecs = dk_vecs > dv_vecs ? dk_vecs : dv_vecs;
    const int blocks = static_cast<int>((max_vecs + kThreads - 1) / kThreads);
    detail::cta2_owner_q_split_add_to_bf16_kernel<<<
        blocks,
        kThreads,
        0,
        stream
    >>>(
        reinterpret_cast<bf16_2 *>(dk.data_ptr()),
        reinterpret_cast<const float4 *>(partial0_dk.data_ptr()),
        reinterpret_cast<const float4 *>(partial1_dk.data_ptr()),
        dk_vecs,
        reinterpret_cast<bf16_2 *>(dv.data_ptr()),
        reinterpret_cast<const float4 *>(partial0_dv.data_ptr()),
        reinterpret_cast<const float4 *>(partial1_dv.data_ptr()),
        dv_vecs
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd_cute16_kernel_candidate

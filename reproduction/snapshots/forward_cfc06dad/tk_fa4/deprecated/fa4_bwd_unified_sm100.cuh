#pragma once

#include "fa4_common.cuh"

#include <cstdlib>
#include <cstring>

namespace tkfa4::bwd {

template <int D>
struct unified_main_globals {
    using grad_tile = st_fl<kRefTileM, D>;
    using q_gl = gl<bf16, -1, -1, -1, D>;
    using k_gl = gl<bf16, -1, -1, -1, D>;
    using v_gl = gl<bf16, -1, -1, -1, D>;
    using do_gl = gl<bf16, -1, -1, -1, D>;
    using dk_gl = gl<float, -1, -1, -1, D>;
    using dv_gl = gl<float, -1, -1, -1, D>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, grad_tile>;
    using l_tile = col_vec<st_fl<kRefTileM, D>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, l_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk_gl dk;
    dv_gl dv;
    dqacc_gl dq_accum;
    l_gl l_aux;
    d_gl delta;
    float scale;
    float scale_log2e;
    int seq_len;
    int actual_seq_len;
    int q_heads;
    int kv_heads;
    int head_ratio;
};

template <int D>
struct unified_reduce_globals {
    using grad_tile = st_fl<kRefTileM, D>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, grad_tile>;
    using dq_gl = gl<float, -1, -1, -1, -1, grad_tile>;

    dqacc_gl dq_accum;
    dq_gl dq;
};

namespace detail {

template <int D>
struct fused_main_traits {
    static constexpr int k_warp_tiles = 8;
    static constexpr int k_block_threads = 256;
    static constexpr int k_min_blocks_per_sm = 1;
};

template <>
struct fused_main_traits<128> {
    static constexpr int k_warp_tiles = 8;
    static constexpr int k_block_threads = 256;
    static constexpr int k_min_blocks_per_sm = 1;
};

constexpr int kDenseHotThreads = 16 * kittens::WARP_THREADS;
constexpr int kDenseHotReduceWarps = 4;
constexpr int kDenseHotComputeWarps = 8;
constexpr int kDenseHotQSubtiles = 4;
constexpr int kDenseHotKSubtiles = 8;
constexpr int kDenseHotLoadWarp = 13;
constexpr int kDenseHotRelayWarp = 14;
constexpr int kDenseHotDqWarp = 12;
constexpr int kDenseHotEmptyWarp = 15;
constexpr int kDenseHotQTileRows = 64;
constexpr int kDenseHotClusterSize = 2;
constexpr int kDenseHotQStageCount = 1;
constexpr int kDenseHotDoStageCount = 1;
constexpr int kDenseHotSingleStageCount = 1;
constexpr int kDenseHotDkvPipeStages = 2;
constexpr int kDenseHotDqPipeStages = 1;
constexpr int kDenseHotDsPipeStages = 1;
constexpr int kDenseHotSdKvAccumStages = 2;

template <bool DETERMINISTIC>
struct dense_hot_reduce_traits {
    static constexpr int k_dq_reduce_ncol = DETERMINISTIC ? 16 : 8;
    static constexpr int k_sdqaccum_stage = DETERMINISTIC ? 2 : 4;
};

struct dense_hot_main_globals {
    using q_tile = st_bf<kRefTileM, 128>;
    using k_tile = st_bf<kRefTileM, 128>;
    using v_tile = st_bf<kRefTileM, 128>;
    using do_tile = st_bf<kRefTileM, 128>;
    using dq_tile = st_fl<kRefTileM, 128>;
    using dk_tile = st_fl<kRefTileM, 128>;
    using dv_tile = st_fl<kRefTileM, 128>;
    using l_tile = col_vec<st_fl<kRefTileM, 128>>;
    using d_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using do_gl = gl<bf16, -1, -1, -1, -1, do_tile>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dq_tile>;
    using dk_gl = gl<float, -1, -1, -1, -1, dk_tile>;
    using dv_gl = gl<float, -1, -1, -1, -1, dv_tile>;

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
};

constexpr int kWgHotConsumerWarpgroups = 2;
constexpr int kWgHotProducerWarpgroups = 1;
constexpr int kWgHotNumWarpgroups = kWgHotConsumerWarpgroups + kWgHotProducerWarpgroups;
constexpr int kWgHotNumWorkers = kWgHotNumWarpgroups * kittens::WARPGROUP_WARPS;

template <int D>
struct wg_hot_tile_dims;

template <>
struct wg_hot_tile_dims<128> {
    static constexpr int tile_width = 128;
    static constexpr int tile_h = 4 * 16;
    static constexpr int tile_h_qo = 4 * 16;
    static constexpr int blocks_sm = 1;
};

struct dense_clustered_wg_globals {
    using G = wg_hot_tile_dims<128>;
    using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile = st_bf<G::tile_h, G::tile_width>;
    using v_tile = st_bf<G::tile_h, G::tile_width>;
    using do_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using dq_tile = st_fl<kRefTileM, G::tile_width>;
    using dk_tile = st_fl<G::tile_h, G::tile_width>;
    using dv_tile = st_fl<G::tile_h, G::tile_width>;
    using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using do_gl = gl<bf16, -1, -1, -1, -1, do_tile>;
    using dqacc_gl = gl<float, -1, -1, -1, -1, dq_tile>;
    using dk_gl = gl<float, -1, -1, -1, -1, dk_tile>;
    using dv_gl = gl<float, -1, -1, -1, -1, dv_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

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
};

template <int D>
struct wg_hot_globals {
    using G = wg_hot_tile_dims<D>;
    using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile = st_bf<G::tile_h, G::tile_width>;
    using v_tile = st_bf<G::tile_h, G::tile_width>;
    using do_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using dq_tile = st_fl<G::tile_h_qo, G::tile_width>;
    using dk_tile = st_fl<G::tile_h, G::tile_width>;
    using dv_tile = st_fl<G::tile_h, G::tile_width>;
    using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using do_gl = gl<bf16, -1, -1, -1, -1, do_tile>;
    using dq_gl = gl<float, -1, -1, -1, -1, dq_tile>;
    using dk_gl = gl<float, -1, -1, -1, -1, dk_tile>;
    using dv_gl = gl<float, -1, -1, -1, -1, dv_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dq_gl dq;
    dk_gl dk;
    dv_gl dv;
    l_gl l_aux;
    d_gl delta;
    float scale;
    float scale_log2e;
    int seq_len;
    int head_ratio;
};
enum class backward_mode {
    Auto,
    Ref,
    Hot,
};

inline backward_mode resolve_backward_mode() {
    const char *mode = std::getenv("TK_FA4_BWD_MODE");
    if (mode == nullptr) {
        return backward_mode::Auto;
    }
    if (std::strcmp(mode, "ref") == 0) {
        return backward_mode::Ref;
    }
    if (std::strcmp(mode, "hot") == 0) {
        return backward_mode::Hot;
    }
    return backward_mode::Auto;
}

inline bool clustered_backward_supported(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len
) {
    return q.size(3) == 128 &&
           !causal &&
           q.size(1) == k.size(1) &&
           actual_seq_len == q.size(2) &&
           q.size(2) % 256 == 0;
}

inline bool dense_hot_gate_enabled() {
    const char *gate = std::getenv("TK_FA4_BWD_DENSE_HOT");
    if (gate == nullptr) {
        return false;
    }
    TORCH_CHECK(
        std::strcmp(gate, "0") == 0 || std::strcmp(gate, "1") == 0,
        "TK_FA4_BWD_DENSE_HOT only supports values 0 or 1"
    );
    return std::strcmp(gate, "1") == 0;
}

inline int backward_cluster_size(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len,
    bool deterministic
) {
    const bool dense_hot_supported =
        resolve_backward_mode() == backward_mode::Hot &&
        dense_hot_gate_enabled() &&
        clustered_backward_supported(q, k, causal, actual_seq_len) &&
        k.size(3) == 128 &&
        !deterministic;
    switch (resolve_backward_mode()) {
        case backward_mode::Ref:
            return 1;
        case backward_mode::Hot:
            return dense_hot_supported ? 2 : 1;
        case backward_mode::Auto:
            return 1;
    }
    return 1;
}

inline bool dense_hot_backward_supported(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len,
    bool deterministic
) {
    return resolve_backward_mode() == backward_mode::Hot &&
           dense_hot_gate_enabled() &&
           clustered_backward_supported(q, k, causal, actual_seq_len) &&
           k.size(3) == 128 &&
           !deterministic;
}

inline bool wg_hot_backward_supported(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len
) {
    return q.size(3) == 128 &&
           !causal &&
           q.size(1) == k.size(1) &&
           actual_seq_len == q.size(2) &&
           q.size(2) % 128 == 0;
}

inline bool use_wg_hot_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len
) {
    const char *override = std::getenv("TK_FA4_BWD_WG_HOT");
    const bool dense_shape_supported =
        q.size(3) == 128 &&
        !causal &&
        q.size(1) == k.size(1) &&
        actual_seq_len == q.size(2) &&
        q.size(2) % 128 == 0;
    if (override != nullptr) {
        if (std::strcmp(override, "0") == 0) {
            return false;
        }
        TORCH_CHECK(
            std::strcmp(override, "1") == 0,
            "TK_FA4_BWD_WG_HOT only supports values 0 or 1"
        );
    } else {
        if (resolve_backward_mode() == backward_mode::Ref) {
            return false;
        }
    }

    const bool supported = override != nullptr
        ? dense_shape_supported
        : wg_hot_backward_supported(q, k, causal, actual_seq_len);
    if (override != nullptr) {
        TORCH_CHECK(
            supported,
            "TK_FA4_BWD_WG_HOT=1 requires dense non-causal MHA with head_dim=128, equal Q/KV heads, and unpadded seqlen"
        );
    }
    return supported;
}

template <typename T>
__device__ inline T *cluster_map_shared_ptr(T *ptr, int dst_cta) {
    const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint32_t mapped_addr = 0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(mapped_addr)
        : "r"(shared_addr), "r"(dst_cta)
    );
    const unsigned long long mapped_addr64 = static_cast<unsigned long long>(mapped_addr);
    unsigned long long generic_addr = 0;
    asm volatile(
        "cvta.shared.u64 %0, %1;\n"
        : "=l"(generic_addr)
        : "l"(mapped_addr64)
    );
    return reinterpret_cast<T *>(generic_addr);
}

__device__ inline void cp_async_bulk_shared_to_cluster_shared(
    void *dst,
    const void *src,
    kittens::semaphore &dst_barrier,
    int dst_cta,
    uint32_t bytes
) {
    const uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    const uint32_t dst_addr_local = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const uint32_t bar_addr_local = static_cast<uint32_t>(__cvta_generic_to_shared(&dst_barrier));
    uint32_t dst_addr = 0;
    uint32_t bar_addr = 0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(dst_addr)
        : "r"(dst_addr_local), "r"(dst_cta)
    );
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(bar_addr)
        : "r"(bar_addr_local), "r"(dst_cta)
    );
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :
        : "r"(dst_addr), "r"(src_addr), "r"(bytes), "r"(bar_addr)
        : "memory"
    );
}

template <int D>
__device__ inline void reconstruct_probability_tile(
    rt_fl<kRefTileM, kRefTileN> &p,
    const rt_bf<kRefTileM, D> &q_reg,
    const rt_bf<kRefTileM, D> &k_reg,
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

template <int D>
__device__ inline void reconstruct_probability_tile_dense(
    rt_fl<kRefTileM, kRefTileN> &p,
    const rt_bf<kRefTileM, D> &q_reg,
    const rt_bf<kRefTileM, D> &k_reg,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &l_aux,
    float scale_log2e
) {
    warp::broadcast_row(p, l_aux);
    warp::mma_ABt(p, q_reg, k_reg, p);
    warp::mul(p, p, scale_log2e);
    warp::exp2(p, p);
}

template <typename RT, typename SMEM>
__device__ inline void wg_stream_tile(RT &reg_tile, SMEM &smem_vec, int stage) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int base_col = 16 * i + 2 * (kittens::laneid() % 4);
        reg_tile.tiles[0][i].data[0] = *reinterpret_cast<float2*>(&smem_vec[stage][base_col + 0]);
        reg_tile.tiles[0][i].data[1] = *reinterpret_cast<float2*>(&smem_vec[stage][base_col + 0]);
        reg_tile.tiles[0][i].data[2] = *reinterpret_cast<float2*>(&smem_vec[stage][base_col + 8]);
        reg_tile.tiles[0][i].data[3] = *reinterpret_cast<float2*>(&smem_vec[stage][base_col + 8]);
    }
}

template <typename RT, typename SMEM>
__device__ inline void wg_stream_sub_tile(RT &reg_tile, SMEM &smem_vec, int stage) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int base_col = 16 * i + 2 * (laneid() % 4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(
            reg_tile.tiles[0][i].data[0],
            *reinterpret_cast<float2*>(&smem_vec[stage][base_col + 0])
        );
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(
            reg_tile.tiles[0][i].data[1],
            *reinterpret_cast<float2*>(&smem_vec[stage][base_col + 0])
        );
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(
            reg_tile.tiles[0][i].data[2],
            *reinterpret_cast<float2*>(&smem_vec[stage][base_col + 8])
        );
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(
            reg_tile.tiles[0][i].data[3],
            *reinterpret_cast<float2*>(&smem_vec[stage][base_col + 8])
        );
    }
}

template <int TILE_H_QO, int TILE_H>
__device__ inline void wg_hot_causal_mask(auto &reg_tile, int qo_idx) {
    const int q_blk = qo_idx * (TILE_H_QO / kittens::TILE_ROW_DIM<bf16>);
    const int k_blk =
        (blockIdx.x * kWgHotConsumerWarpgroups * (TILE_H / kittens::TILE_ROW_DIM<bf16>)) +
        ((kittens::warpid() / kittens::WARPGROUP_WARPS) * (TILE_H / kittens::TILE_ROW_DIM<bf16>)) +
        (kittens::warpid() % kittens::WARPGROUP_WARPS);

    for (int j = 0; j < (TILE_H_QO / kittens::TILE_ROW_DIM<bf16>); ++j) {
        const int q_idx = q_blk + j;
        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);
        if (q_idx < k_blk) {
            warp::neg_infty(attn_subtile);
        } else if (q_idx == k_blk) {
            warp::make_causal_t(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
        }
    }
}

template <bool CAUSAL, int TILE_H_QO, int TILE_H, typename AttnTT, typename GradTT>
__device__ inline void wg_hot_compute_bwd_loop(
    kittens::semaphore *vec_b,
    kittens::semaphore *q_b,
    kittens::semaphore *o_b,
    kittens::semaphore &score_ready,
    kittens::semaphore &dp_ready,
    rt_fl<16, 64> &s_block_t,
    rt_fl<16, 64> &dp_block_t,
    rt_fl<16, 64> &p_block_t,
    rt_fl<16, 64> &ds_block_t,
    rt_bf<16, 64> &p_block_t_mma,
    rt_bf<16, 64> &ds_block_t_mma,
    AttnTT &score_tt,
    AttnTT &dp_tt,
    GradTT &dk_tt,
    GradTT &dv_tt,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_smem,
    auto &l_smem,
    auto &d_smem,
    int qo_idx,
    int q_start,
    int stage,
    float scale,
    float scale_log2e,
    bool accumulate
) {
    using prob_tt = half_tt_bf<TILE_H>;

    const int consumer_idx = kittens::warpid() / kittens::WARPGROUP_WARPS;
    const int phase = ((qo_idx - q_start) / 2) % 2;

    wait(vec_b[stage], phase);
    wg_stream_tile(s_block_t, l_smem, stage);
    wait(q_b[stage], phase);
    warpgroup::mm_ABt(score_tt, k_smem[consumer_idx], q_smem[stage], score_ready);
    wait(score_ready, phase);
    warpgroup::load_async(p_block_t, score_tt);
    tensor_load_wait();
    warp::add(s_block_t, s_block_t, p_block_t);

    wait(o_b[stage], phase);
    warpgroup::mm_ABt(dp_tt, v_smem[consumer_idx], do_smem[stage], dp_ready);
    wait(dp_ready, phase);
    warpgroup::load_async(dp_block_t, dp_tt);
    tensor_load_wait();

    warp::mul(s_block_t, s_block_t, scale_log2e);
    if constexpr (CAUSAL) {
        wg_hot_causal_mask<TILE_H_QO, TILE_H>(s_block_t, qo_idx);
    }
    warp::exp2(s_block_t, s_block_t);
    warp::copy(p_block_t, s_block_t);
    warp::copy(p_block_t_mma, s_block_t);
    wg_stream_sub_tile(dp_block_t, d_smem, stage);
    warp::mul(ds_block_t, p_block_t, dp_block_t);
    warp::mul(ds_block_t, ds_block_t, scale);
    warp::copy(ds_block_t_mma, ds_block_t);
    const prob_tt probs_tmem = prob_tt{score_tt.addr};
    const prob_tt ds_tmem = prob_tt{dp_tt.addr};

    warpgroup::store_async(probs_tmem, p_block_t_mma);
    warpgroup::store_async(ds_tmem, ds_block_t_mma);
    warpgroup::store(ds_smem[consumer_idx], ds_block_t_mma);
    tensor_store_wait();

    if (accumulate) {
        warpgroup::mma_AB(dv_tt, probs_tmem, do_smem[stage]);
        warpgroup::mma_AB(dk_tt, ds_tmem, q_smem[stage]);
    } else {
        warpgroup::mm_AB(dv_tt, probs_tmem, do_smem[stage]);
        warpgroup::mm_AB(dk_tt, ds_tmem, q_smem[stage]);
    }
    group<8>::sync(10);
}

template <bool CAUSAL, int TILE_H_QO, int TILE_H, typename AttnTT, typename GradTT>
__device__ inline void clustered_dense_compute_bwd_loop(
    kittens::semaphore *vec_b,
    kittens::semaphore *q_b,
    kittens::semaphore *o_b,
    kittens::semaphore &score_ready,
    kittens::semaphore &dp_ready,
    rt_fl<16, 64> &s_block_t,
    rt_fl<16, 64> &dp_block_t,
    rt_fl<16, 64> &p_block_t,
    rt_fl<16, 64> &ds_block_t,
    rt_bf<16, 64> &p_block_t_mma,
    rt_bf<16, 64> &ds_block_t_mma,
    AttnTT &score_tt,
    AttnTT &dp_tt,
    GradTT &dk_tt,
    GradTT &dv_tt,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &do_smem,
    auto &ds_smem,
    auto &l_smem,
    auto &d_smem,
    int consumer_idx,
    int qo_idx,
    int stage,
    float scale,
    float scale_log2e,
    bool accumulate
) {
    using prob_tt = half_tt_bf<TILE_H>;

    const int phase = qo_idx & 1;

    wait(vec_b[stage], phase);
    wg_stream_tile(s_block_t, l_smem, stage);
    wait(q_b[stage], phase);
    warpgroup::mm_ABt(score_tt, k_smem[consumer_idx], q_smem[stage], score_ready);
    wait(score_ready, phase);
    warpgroup::load_async(p_block_t, score_tt);
    tensor_load_wait();
    warp::add(s_block_t, s_block_t, p_block_t);

    wait(o_b[stage], phase);
    warpgroup::mm_ABt(dp_tt, v_smem[consumer_idx], do_smem[stage], dp_ready);
    wait(dp_ready, phase);
    warpgroup::load_async(dp_block_t, dp_tt);
    tensor_load_wait();

    warp::mul(s_block_t, s_block_t, scale_log2e);
    if constexpr (CAUSAL) {
        wg_hot_causal_mask<TILE_H_QO, TILE_H>(s_block_t, qo_idx);
    }
    warp::exp2(s_block_t, s_block_t);
    warp::copy(p_block_t, s_block_t);
    warp::copy(p_block_t_mma, s_block_t);
    wg_stream_sub_tile(dp_block_t, d_smem, stage);
    warp::mul(ds_block_t, p_block_t, dp_block_t);
    warp::mul(ds_block_t, ds_block_t, scale);
    warp::copy(ds_block_t_mma, ds_block_t);
    const prob_tt probs_tmem = prob_tt{score_tt.addr};
    const prob_tt ds_tmem = prob_tt{dp_tt.addr};

    warpgroup::store_async(probs_tmem, p_block_t_mma);
    warpgroup::store_async(ds_tmem, ds_block_t_mma);
    warpgroup::store(ds_smem[consumer_idx], ds_block_t_mma);
    tensor_store_wait();

    if (accumulate) {
        warpgroup::mma_AB(dv_tt, probs_tmem, do_smem[stage]);
        warpgroup::mma_AB(dk_tt, ds_tmem, q_smem[stage]);
    } else {
        warpgroup::mm_AB(dv_tt, probs_tmem, do_smem[stage]);
        warpgroup::mm_AB(dk_tt, ds_tmem, q_smem[stage]);
    }
    group<8>::sync(10);
}

template <typename DKTile, typename DVTile, typename Globals>
__device__ inline void wg_hot_kv_store(
    auto &dk_smem,
    auto &dk_reg,
    auto &dv_smem,
    auto &dv_reg,
    Globals &dst,
    kittens::semaphore &bar,
    int kv_head_idx,
    int stage
) {
    group<8>::sync(10);
    warpgroup::store(dk_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], dk_reg);
    group<4>::sync(warpgroup::groupid() + 4);
    if (kittens::warpid() % 4 == 0) {
        coord<DKTile> tile_idx = {
            blockIdx.z,
            kv_head_idx,
            (blockIdx.x * kWgHotConsumerWarpgroups) + (kittens::warpid() / kittens::WARPGROUP_WARPS),
            0
        };
        warp::tma::store_async(dst.dk, dk_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], tile_idx);
    }

    wait(bar, stage);
    warpgroup::store(dv_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], dv_reg);
    group<4>::sync(warpgroup::groupid() + 4);
    if (kittens::warpid() % 4 == 0) {
        coord<DVTile> tile_idx = {
            blockIdx.z,
            kv_head_idx,
            (blockIdx.x * kWgHotConsumerWarpgroups) + (kittens::warpid() / kittens::WARPGROUP_WARPS),
            0
        };
        warp::tma::store_add_async(dst.dv, dv_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], tile_idx);
        warp::tma::store_commit_group();
    }
    warp::tma::store_async_wait();
}

template <int D, bool CAUSAL>
__global__ __launch_bounds__(kWgHotNumWorkers * kittens::WARP_THREADS, wg_hot_tile_dims<D>::blocks_sm)
void wg_hot_backward_kernel(const __grid_constant__ wg_hot_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al(reinterpret_cast<int*>(&__shm[0]));

    using G = wg_hot_tile_dims<D>;
    using dk_tile = st_fl<G::tile_h, G::tile_width>;
    using dv_tile = st_fl<G::tile_h, G::tile_width>;
    using k_tile = st_bf<G::tile_h, G::tile_width>;
    using v_tile = st_bf<G::tile_h, G::tile_width>;
    using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using do_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using dq_tile = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tt = half_tt_fl<G::tile_h>;
    using prob_tt = half_tt_bf<G::tile_h>;
    using grad_tt = half_tt_fl<G::tile_width>;

    k_tile (&k_smem)[kWgHotConsumerWarpgroups] = al.allocate<k_tile, kWgHotConsumerWarpgroups>();
    v_tile (&v_smem)[kWgHotConsumerWarpgroups] = al.allocate<v_tile, kWgHotConsumerWarpgroups>();
    q_tile (&q_smem)[2] = al.allocate<q_tile, 2>();
    do_tile (&do_smem)[2] = al.allocate<do_tile, 2>();
    dq_tile (&dq_smem) = al.allocate<dq_tile>();
    l_tile (&l_smem)[2] = al.allocate<l_tile, 2>();
    d_tile (&d_smem)[2] = al.allocate<d_tile, 2>();
    dk_tile (*dk_smem) = reinterpret_cast<dk_tile*>(&k_smem[0].data[0]);
    dv_tile (*dv_smem) = reinterpret_cast<dv_tile*>(&q_smem[0].data[0]);
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;
    attn_tile (&ds_smem)[kWgHotConsumerWarpgroups] = al.allocate<attn_tile, kWgHotConsumerWarpgroups>();
    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[kWgHotConsumerWarpgroups][2];
    attn_tt dp_tt[kWgHotConsumerWarpgroups][2];
    grad_tt dk_tt[kWgHotConsumerWarpgroups];
    grad_tt dv_tt[kWgHotConsumerWarpgroups];
    grad_tt dq_tt[2];
    const int warpid = kittens::warpid();
    const int warpgroup_id = warpid / kittens::WARPGROUP_WARPS;
    const int qo_blocks = g.seq_len / G::tile_h_qo;
    const int kv_head_idx = blockIdx.y / g.head_ratio;

    __shared__ kittens::semaphore kv_b;
    __shared__ kittens::semaphore q_b[2];
    __shared__ kittens::semaphore o_b[2];
    __shared__ kittens::semaphore vec_b[2];
    __shared__ kittens::semaphore compute_done[2];
    __shared__ kittens::semaphore dq_ready;
    __shared__ kittens::semaphore score_ready[kWgHotConsumerWarpgroups][2];
    __shared__ kittens::semaphore dp_ready[kWgHotConsumerWarpgroups][2];
    __shared__ kittens::semaphore dq_tmem_ready[2];
    __shared__ kittens::semaphore kv_tmem_ready[kWgHotConsumerWarpgroups];

    int stage = 0;
    int next_stage = 1;
    const int q_start = CAUSAL ? (blockIdx.x * kWgHotConsumerWarpgroups) : 0;

    #pragma unroll
    for (int w = 0; w < kWgHotConsumerWarpgroups; ++w) {
        score_tt[w][0] = tm_alloc.template allocate<attn_tt>(w, 0);
        dp_tt[w][0] = tm_alloc.template allocate<attn_tt>(w, G::tile_h);
        score_tt[w][1] = tm_alloc.template allocate<attn_tt>(w, 2 * G::tile_h);
        dp_tt[w][1] = tm_alloc.template allocate<attn_tt>(w, 3 * G::tile_h);
        dk_tt[w] = tm_alloc.template allocate<grad_tt>(w, 4 * G::tile_h);
        dv_tt[w] = tm_alloc.template allocate<grad_tt>(w, 4 * G::tile_h + G::tile_width);
    }
    dq_tt[0] = grad_tt{tm_alloc.get_addr(0, 0)};
    dq_tt[1] = grad_tt{tm_alloc.get_addr(0, 2 * G::tile_h)};

    if (threadIdx.x == 0) {
        init_semaphore(kv_b, 0, 1);
        init_semaphore(dq_ready, 1, 0);
        for (int i = 0; i < 2; ++i) {
            init_semaphore(q_b[i], 0, 1);
            init_semaphore(o_b[i], 0, 1);
            init_semaphore(vec_b[i], 0, 1);
            init_semaphore(compute_done[i], 1, 0);
            init_semaphore(dq_tmem_ready[i], 0, 1);
            for (int w = 0; w < kWgHotConsumerWarpgroups; ++w) {
                init_semaphore(score_ready[w][i], 0, 1);
                init_semaphore(dp_ready[w][i], 0, 1);
            }
        }
        for (int w = 0; w < kWgHotConsumerWarpgroups; ++w) {
            init_semaphore(kv_tmem_ready[w], 0, 1);
        }

        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * kWgHotConsumerWarpgroups);
        for (int w = 0; w < kWgHotConsumerWarpgroups; ++w) {
            coord<k_tile> tile_idx = {
                blockIdx.z,
                kv_head_idx,
                (blockIdx.x * kWgHotConsumerWarpgroups) + w,
                0
            };
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }

        coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, q_start, 0};
        tma::expect_bytes(q_b[stage], sizeof(q_smem[0]));
        tma::load_async(q_smem[stage], g.q, q_tile_idx, q_b[stage]);
        tma::expect_bytes(o_b[stage], sizeof(do_smem[0]));
        tma::load_async(do_smem[stage], g.dout, q_tile_idx, o_b[stage]);

        coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, q_start};
        tma::expect_bytes(vec_b[stage], sizeof(l_smem[0]) + sizeof(d_smem[0]));
        tma::load_async(l_smem[stage], g.l_aux, vec_idx, vec_b[stage]);
        tma::load_async(d_smem[stage], g.delta, vec_idx, vec_b[stage]);
    }
    __syncthreads();

    if (warpgroup_id == kWgHotNumWarpgroups - 1) {
        warpgroup::decrease_registers<24>();
        
        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (int qo_idx = q_start; qo_idx < qo_blocks; ++qo_idx, stage ^= 1, next_stage ^= 1) {
                if (qo_idx + 1 < qo_blocks) {
                    coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, qo_idx + 1, 0};
                    warp::tma::expect_bytes(q_b[next_stage], sizeof(q_smem[0]));
                    warp::tma::load_async(q_smem[next_stage], g.q, q_tile_idx, q_b[next_stage]);
                    warp::tma::expect_bytes(o_b[next_stage], sizeof(do_smem[0]));
                    warp::tma::load_async(do_smem[next_stage], g.dout, q_tile_idx, o_b[next_stage]);

                    coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, qo_idx + 1};
                    warp::tma::expect_bytes(vec_b[next_stage], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                    warp::tma::load_async(l_smem[next_stage], g.l_aux, vec_idx, vec_b[next_stage]);
                    warp::tma::load_async(d_smem[next_stage], g.delta, vec_idx, vec_b[next_stage]);
                }
                wait(compute_done[stage], ((qo_idx - q_start) / 2) % 2);
            }
        } else if (warpid % kittens::WARPGROUP_WARPS == 1) {
            for (int qo_idx = q_start; qo_idx < qo_blocks; ++qo_idx, stage ^= 1, next_stage ^= 1) {
                wait(compute_done[stage], ((qo_idx - q_start) / 2) % 2);
                coord<dq_tile> tile_idx = {blockIdx.z, blockIdx.y, qo_idx, 0};
                warp::tma::store_add_async(g.dq, dq_smem, tile_idx);
                warp::tma::store_async_read_wait();
                if (laneid() == 0) {
                    arrive(dq_ready);
                }
            }
        }
    } else {
        rt_fl<16, G::tile_width> dk_reg, dv_reg;
        rt_fl<16, 64> s_block_t, p_block_t, dp_block_t, ds_block_t;
        rt_bf<16, 64> p_block_t_mma, ds_block_t_mma;

        if (warpgroup_id == 0) {
            warpgroup::increase_registers<240>();
            wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; ++qo_idx, stage ^= 1, next_stage ^= 1) {
                wg_hot_compute_bwd_loop<CAUSAL, G::tile_h_qo, G::tile_h>(
                    vec_b,
                    q_b,
                    o_b,
                    score_ready[0][stage],
                    dp_ready[0][stage],
                    s_block_t,
                    dp_block_t,
                    p_block_t,
                    ds_block_t,
                    p_block_t_mma,
                    ds_block_t_mma,
                    score_tt[0][stage],
                    dp_tt[0][stage],
                    dk_tt[0],
                    dv_tt[0],
                    q_smem,
                    k_smem,
                    v_smem,
                    do_smem,
                    ds_smem,
                    l_smem,
                    d_smem,
                    qo_idx,
                    q_start,
                    stage,
                    g.scale,
                    g.scale_log2e,
                    qo_idx > q_start
                );

                rt_fl<16, G::tile_width> dq_reg;
                warpgroup::mm_AtB(dq_tt[stage], ds_smem[0], k_smem[0]);
                warpgroup::mma_AtB(dq_tt[stage], ds_smem[1], k_smem[1]);
                if (warpgroup::laneid() == 0) {
                    tensor_commit<1>(dq_tmem_ready[stage]);
                }
                wait(dq_tmem_ready[stage], ((qo_idx - q_start) / 2) % 2);
                warpgroup::load_async(dq_reg, dq_tt[stage]);
                wait(dq_ready, next_stage);
                tensor_load_wait();
                warpgroup::store(dq_smem, dq_reg);
                group<4>::sync(warpgroup::groupid() + 4);
                if (warpgroup::laneid() == 0) {
                    arrive(compute_done[stage]);
                }
            }
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(kv_tmem_ready[0]);
            }
            wait(kv_tmem_ready[0], 0);
            warpgroup::load_async(dk_reg, dk_tt[0]);
            warpgroup::load_async(dv_reg, dv_tt[0]);
            tensor_load_wait();
            wg_hot_kv_store<dk_tile, dv_tile>(dk_smem, dk_reg, dv_smem, dv_reg, g, dq_ready, kv_head_idx, next_stage);
        } else {
            warpgroup::increase_registers<208>();
            wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; ++qo_idx, stage ^= 1, next_stage ^= 1) {
                wg_hot_compute_bwd_loop<CAUSAL, G::tile_h_qo, G::tile_h>(
                    vec_b,
                    q_b,
                    o_b,
                    score_ready[1][stage],
                    dp_ready[1][stage],
                    s_block_t,
                    dp_block_t,
                    p_block_t,
                    ds_block_t,
                    p_block_t_mma,
                    ds_block_t_mma,
                    score_tt[1][stage],
                    dp_tt[1][stage],
                    dk_tt[1],
                    dv_tt[1],
                    q_smem,
                    k_smem,
                    v_smem,
                    do_smem,
                    ds_smem,
                    l_smem,
                    d_smem,
                    qo_idx,
                    q_start,
                    stage,
                    g.scale,
                    g.scale_log2e,
                    qo_idx > q_start
                );
            }
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(kv_tmem_ready[1]);
            }
            wait(kv_tmem_ready[1], 0);
            warpgroup::load_async(dk_reg, dk_tt[1]);
            warpgroup::load_async(dv_reg, dv_tt[1]);
            tensor_load_wait();
            wg_hot_kv_store<dk_tile, dv_tile>(dk_smem, dk_reg, dv_smem, dv_reg, g, dq_ready, kv_head_idx, next_stage);
        }
    }
}

template <int D>
inline int wg_hot_dynamic_smem_bytes();

template <>
inline int wg_hot_dynamic_smem_bytes<128>() {
    return 183296;
}

template <bool CAUSAL>
__global__ __launch_bounds__(kDenseHotThreads, 1)
void dense_hot_main_kernel(const __grid_constant__ dense_clustered_wg_globals g) {
    static_assert(!CAUSAL, "Dense hot backward only supports dense non-causal BF16 D=128.");

    using G = wg_hot_tile_dims<128>;
    using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile = st_bf<G::tile_h, G::tile_width>;
    using v_tile = st_bf<G::tile_h, G::tile_width>;
    using do_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using dq_tile = st_fl<G::tile_h_qo, G::tile_width>;
    using dk_tile = st_fl<G::tile_h, G::tile_width>;
    using dv_tile = st_fl<G::tile_h, G::tile_width>;
    using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;
    using attn_tt = half_tt_fl<G::tile_h>;
    using prob_tt = half_tt_bf<G::tile_h>;
    using grad_tt = half_tt_fl<G::tile_width>;

    struct dense_hot_main_shared_tiles {
        k_tile k_smem[kWgHotConsumerWarpgroups];
        v_tile v_smem[kWgHotConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        st_bf<kRefTileM, G::tile_h> ds_warp_smem[kWgHotConsumerWarpgroups][kittens::WARPGROUP_WARPS];
        attn_tile ds_smem[kWgHotConsumerWarpgroups];
        l_tile l_smem[1];
        d_tile d_smem[1];
    };
    struct dense_hot_epilogue_shared_tiles {
        dk_tile dk_smem[kWgHotConsumerWarpgroups];
        dv_tile dv_smem[kWgHotConsumerWarpgroups];
    };
    union dense_hot_shared_storage {
        dense_hot_main_shared_tiles main;
        dense_hot_epilogue_shared_tiles epilogue;
    };
    __shared__ alignas(1024) dense_hot_shared_storage smem;
    auto &k_smem = smem.main.k_smem;
    auto &v_smem = smem.main.v_smem;
    auto &q_smem = smem.main.q_smem;
    auto &do_smem = smem.main.do_smem;
    auto &ds_warp_smem = smem.main.ds_warp_smem;
    auto &ds_smem = smem.main.ds_smem;
    auto &l_smem = smem.main.l_smem;
    auto &d_smem = smem.main.d_smem;
    auto &dk_smem = smem.epilogue.dk_smem;
    auto &dv_smem = smem.epilogue.dv_smem;

    __shared__ __align__(16) kittens::semaphore kv_b;
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];
    __shared__ __align__(16) kittens::semaphore vec_b[1];
    __shared__ __align__(16) kittens::semaphore score_ready[kWgHotConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore dp_ready[kWgHotConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore kv_tmem_ready[kWgHotConsumerWarpgroups];
    const int warp = kittens::warpid();
    const int warpgroup_id = warp / kittens::WARPGROUP_WARPS;
    constexpr int kDenseHotComputeWarpBegin = 0;
    constexpr int kDenseHotComputeWarpEnd = kDenseHotComputeWarps;
    constexpr int kDenseHotReduceWarpBegin = kDenseHotComputeWarpEnd;
    constexpr int kDenseHotReduceWarpEnd = kDenseHotReduceWarpBegin + kDenseHotReduceWarps;
    constexpr int kDenseHotLoadWarpDense = kDenseHotReduceWarpEnd;
    constexpr int kDenseHotRelayWarpDense = kDenseHotReduceWarpEnd + 1;

    const bool is_compute = warp >= kDenseHotComputeWarpBegin && warp < kDenseHotComputeWarpEnd;
    const bool is_reduce = warp >= kDenseHotReduceWarpBegin && warp < kDenseHotReduceWarpEnd;
    const bool is_load = warp == kDenseHotLoadWarpDense;
    const bool is_relay = warp == kDenseHotRelayWarpDense;
    const bool is_store = is_reduce;
    const int consumer_idx = is_compute ? warpgroup_id : -1;

    const int batch_idx = blockIdx.z;
    const int kv_head_idx = blockIdx.y;
    const int q_head_idx = kv_head_idx;
    const int cluster_rank = cluster_ctarank();
    const int cluster_idx = static_cast<int>(blockIdx.x) / kDenseHotClusterSize;
    const int num_k_blocks = g.seq_len / (G::tile_h * kWgHotConsumerWarpgroups);
    const int kv_block_idx = cluster_idx * kDenseHotClusterSize + cluster_rank;
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * kWgHotConsumerWarpgroups;
    const int q_blocks = g.seq_len / G::tile_h_qo;
    constexpr int kQSubtilesPerForwardTile = kForwardTileM / kRefTileM;
    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[kWgHotConsumerWarpgroups][1] = {
        {attn_tt{0}},
        {attn_tt{0}}
    };
    attn_tt dp_tt[kWgHotConsumerWarpgroups][1] = {
        {attn_tt{0}},
        {attn_tt{0}}
    };
    grad_tt dk_tt[kWgHotConsumerWarpgroups] = {grad_tt{0}, grad_tt{0}};
    grad_tt dv_tt[kWgHotConsumerWarpgroups] = {grad_tt{0}, grad_tt{0}};

    if (threadIdx.x == 0) {
        init_semaphore(kv_b, 0, 1);
        for (int stage = 0; stage < 1; ++stage) {
            init_semaphore(q_b[stage], 0, 1);
            init_semaphore(o_b[stage], 0, 1);
            init_semaphore(vec_b[stage], 0, 1);
            for (int w = 0; w < kWgHotConsumerWarpgroups; ++w) {
                init_semaphore(score_ready[w][stage], 0, 1);
                init_semaphore(dp_ready[w][stage], 0, 1);
            }
        }
        for (int w = 0; w < kWgHotConsumerWarpgroups; ++w) {
            init_semaphore(kv_tmem_ready[w], 0, 1);
        }
    }
    __syncthreads();

    if (is_compute) {
        score_tt[consumer_idx][0] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
        dp_tt[consumer_idx][0] = tm_alloc.template allocate<attn_tt>(consumer_idx, G::tile_h);
        dk_tt[consumer_idx] = tm_alloc.template allocate<grad_tt>(consumer_idx, 4 * G::tile_h);
        dv_tt[consumer_idx] = tm_alloc.template allocate<grad_tt>(consumer_idx, 4 * G::tile_h + G::tile_width);
    }

    if (threadIdx.x == 0) {
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * kWgHotConsumerWarpgroups);
        for (int w = 0; w < kWgHotConsumerWarpgroups; ++w) {
            coord<k_tile> tile_idx = {batch_idx, kv_head_idx, kv_tile_base + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }
    }
    __syncthreads();

    for (int qo_idx = 0; qo_idx < q_blocks; ++qo_idx) {
        constexpr int stage = 0;

        if (is_load) {
            coord<q_tile> q_tile_idx = {batch_idx, q_head_idx, qo_idx, 0};
            warp::tma::expect_bytes(q_b[stage], sizeof(q_smem[0]));
            warp::tma::load_async(q_smem[stage], g.q, q_tile_idx, q_b[stage]);
            warp::tma::expect_bytes(o_b[stage], sizeof(do_smem[0]));
            warp::tma::load_async(do_smem[stage], g.dout, q_tile_idx, o_b[stage]);

            coord<l_tile> vec_idx = {batch_idx, q_head_idx, 0, qo_idx};
            warp::tma::expect_bytes(vec_b[stage], sizeof(l_smem[0]) + sizeof(d_smem[0]));
            warp::tma::load_async(l_smem[stage], g.l_aux, vec_idx, vec_b[stage]);
            warp::tma::load_async(d_smem[stage], g.delta, vec_idx, vec_b[stage]);
        }
        __syncthreads();

        if (is_compute) {
            rt_fl<16, G::tile_width> dk_reg, dv_reg;
            rt_fl<16, 64> s_block_t, p_block_t, dp_block_t, ds_block_t;
            rt_bf<16, 64> p_block_t_mma, ds_block_t_mma;
            if (qo_idx == 0) {
                warpgroup::zero(dk_reg);
                warpgroup::zero(dv_reg);
            }

            wait(kv_b, 0);
            clustered_dense_compute_bwd_loop<CAUSAL, G::tile_h_qo, G::tile_h>(
                vec_b,
                q_b,
                o_b,
                score_ready[consumer_idx][stage],
                dp_ready[consumer_idx][stage],
                s_block_t,
                dp_block_t,
                p_block_t,
                ds_block_t,
                p_block_t_mma,
                ds_block_t_mma,
                score_tt[consumer_idx][0],
                dp_tt[consumer_idx][0],
                dk_tt[consumer_idx],
                dv_tt[consumer_idx],
                q_smem,
                k_smem,
                v_smem,
                do_smem,
                ds_smem,
                l_smem,
                d_smem,
                consumer_idx,
                qo_idx,
                stage,
                g.scale,
                g.scale_log2e,
                qo_idx > 0
            );
            warp::store(ds_warp_smem[consumer_idx][warpgroup::warpid()], ds_block_t_mma);
        }
        __syncthreads();

        if (is_store) {
            const int dq_subtile_idx = warp - kDenseHotReduceWarpBegin;
            rt_fl<16, G::tile_width> dq_partial;
            rt_bf<16, G::tile_h> ds_local_reg;
            rt_bf<G::tile_h, G::tile_width> k_local_reg;
            rt_bf<G::tile_h, G::tile_width, ducks::rt_layout::col> k_local_col;
            warp::zero(dq_partial);
            #pragma unroll
            for (int consumer = 0; consumer < kWgHotConsumerWarpgroups; ++consumer) {
                warp::load(ds_local_reg, ds_warp_smem[consumer][dq_subtile_idx]);
                warp::load(k_local_reg, k_smem[consumer]);
                warp::swap_layout(k_local_col, k_local_reg);
                warp::mma_AB(dq_partial, ds_local_reg, k_local_col, dq_partial);
            }
            const int q_tile_idx = qo_idx * kDenseHotReduceWarps + dq_subtile_idx;
            const int q_tile_group_idx = q_tile_idx / kQSubtilesPerForwardTile;
            const int q_subtile_in_group = q_tile_idx % kQSubtilesPerForwardTile;
            const int scratch_tile_idx =
                ((q_tile_group_idx * kDenseHotClusterSize) + cluster_rank) * kQSubtilesPerForwardTile +
                q_subtile_in_group;
            warp::store(g.dq_accum, dq_partial, {batch_idx, q_head_idx, scratch_tile_idx, 0});
        }
        __syncthreads();
    }

    if (is_compute) {
        rt_fl<16, G::tile_width> dk_reg, dv_reg;
        if (warpgroup::laneid() == 0) {
            tensor_commit<1>(kv_tmem_ready[consumer_idx]);
        }
        wait(kv_tmem_ready[consumer_idx], 0);
        warpgroup::load_async(dk_reg, dk_tt[consumer_idx]);
        warpgroup::load_async(dv_reg, dv_tt[consumer_idx]);
        tensor_load_wait();
        warpgroup::store(dk_smem[consumer_idx], dk_reg);
        warpgroup::store(dv_smem[consumer_idx], dv_reg);
    }
    __syncthreads();

    if (is_relay) {
        #pragma unroll
        for (int consumer = 0; consumer < kWgHotConsumerWarpgroups; ++consumer) {
            coord<dk_tile> tile_idx = {batch_idx, kv_head_idx, kv_tile_base + consumer, 0};
            warp::tma::store_async(g.dk, dk_smem[consumer], tile_idx);
            warp::tma::store_async(g.dv, dv_smem[consumer], tile_idx);
        }
        warp::tma::store_commit_group();
        warp::tma::store_async_wait();
    }
}

template <bool DETERMINISTIC>
__global__ __launch_bounds__(kDenseHotReduceWarps * kWarpThreads, 8)
void dense_hot_dqacc_reduce_kernel(const __grid_constant__ unified_reduce_globals<128> g) {
    using traits = dense_hot_reduce_traits<DETERMINISTIC>;
    static_assert(128 % traits::k_dq_reduce_ncol == 0, "Dense hot reduce columns must divide head_dim.");
    static_assert(traits::k_sdqaccum_stage >= 1, "Dense hot reducer requires at least one staging slot.");

    const int warp = threadIdx.x >> 5;
    const int batch_idx = blockIdx.z;
    const int q_head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    const int q_tile_base = q_block_idx * kDenseHotQSubtiles;

    if (warp >= kDenseHotReduceWarps) {
        return;
    }

    const int q_tile_idx = q_tile_base + warp;
    const int q_tile_group_idx = q_tile_idx / (kForwardTileM / kRefTileM);
    const int q_subtile_in_group = q_tile_idx % (kForwardTileM / kRefTileM);
    const int scratch_tile_idx =
        (q_tile_group_idx * kDenseHotClusterSize) * (kForwardTileM / kRefTileM) +
        q_subtile_in_group;

    rt_fl<kRefTileM, 128> dq_reg, dq_peer;
    warp::load(dq_reg, g.dq_accum, {batch_idx, q_head_idx, scratch_tile_idx, 0});
    warp::load(
        dq_peer,
        g.dq_accum,
        {
            batch_idx,
            q_head_idx,
            scratch_tile_idx + (kDenseHotClusterSize == 2 ? (kForwardTileM / kRefTileM) : 0),
            0
        }
    );
    warp::add(dq_reg, dq_reg, dq_peer);
    warp::store(g.dq, dq_reg, {batch_idx, q_head_idx, q_tile_idx, 0});
}

template <bool CAUSAL>
__global__ __launch_bounds__(256, 1)
void clustered_dsqacc_bringup_kernel(const __grid_constant__ unified_main_globals<128> g) {
    static_assert(!CAUSAL, "Clustered dS->dQaccum bringup only supports dense non-causal backward.");

    constexpr int D = 128;
    constexpr int kWarpTiles = fused_main_traits<D>::k_warp_tiles;
    constexpr int qTilesBuffered = 4;
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;

    using bf_tile = st_bf<kRefTileM, D>;
    using fl_tile = st_fl<kRefTileM, D>;
    using ds_tile = st_bf<kRefTileM, kRefTileN>;
    using sv_stats = col_vec<st_fl<kRefTileM, D>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) bf_tile q_smem[qTilesBuffered];
    __shared__ alignas(1024) bf_tile do_smem[qTilesBuffered];
    __shared__ alignas(1024) fl_tile dq_smem[kWarpTiles];
    __shared__ alignas(1024) ds_tile ds_xchg[kWarpTiles];
    __shared__ alignas(1024) ds_tile ds_peer[kWarpTiles];
    __shared__ alignas(64) sv_stats l_smem[qTilesBuffered];
    __shared__ alignas(64) sv_stats delta_smem[qTilesBuffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = blockIdx.z;
    const int kv_head_idx = blockIdx.y;
    const int cluster_rank = cluster_ctarank();
    const int kv_block_idx = clusterIdx().x * 2 + cluster_rank;
    const int num_k_blocks = g.seq_len / (kRefTileN * kWarpTiles);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * kWarpTiles;
    const int peer_kv_tile_base = kv_tile_base + kWarpTiles * (cluster_rank == 0 ? 1 : -1);
    const int kv_subtile_idx = kv_tile_base + warp;
    const int q_head_idx = kv_head_idx;
    const int num_q_blocks = g.seq_len / (kRefTileM * qTilesBuffered);
    const int peer_cta = cluster_rank ^ 1;
    bf16 *peer_ds_raw = cluster_map_shared_ptr(reinterpret_cast<bf16 *>(&ds_xchg[0].data[0]), peer_cta);
    bf16 *local_peer_ds_raw = reinterpret_cast<bf16 *>(&ds_peer[0].data[0]);

    rt_bf<kRefTileM, D> k_reg, v_reg, peer_k_reg;
    rt_bf<kRefTileM, D, ducks::rt_layout::col> k_col, peer_k_col;
    rt_fl<kRefTileM, D> dk_accum, dv_accum;

    warp::load(k_reg, g.k, {batch_idx, kv_head_idx, kv_subtile_idx, 0});
    warp::load(v_reg, g.v, {batch_idx, kv_head_idx, kv_subtile_idx, 0});
    warp::swap_layout(k_col, k_reg);
    if (cluster_rank == 0) {
        warp::load(peer_k_reg, g.k, {batch_idx, kv_head_idx, peer_kv_tile_base + warp, 0});
        warp::swap_layout(peer_k_col, peer_k_reg);
    }
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * qTilesBuffered;

        if (warp < qTilesBuffered) {
            rt_bf<kRefTileM, D> q_stage_reg, do_stage_reg;
            stats_vec l_stage_vec, delta_stage_vec;
            warp::load(q_stage_reg, g.q, {batch_idx, q_head_idx, q_tile_base + warp, 0});
            warp::store(q_smem[warp], q_stage_reg);
            warp::load(do_stage_reg, g.dout, {batch_idx, q_head_idx, q_tile_base + warp, 0});
            warp::store(do_smem[warp], do_stage_reg);
            warp::load(l_stage_vec, g.l_aux, {batch_idx, q_head_idx, 0, q_tile_base + warp});
            warp::store(l_smem[warp], l_stage_vec);
            warp::load(delta_stage_vec, g.delta, {batch_idx, q_head_idx, 0, q_tile_base + warp});
            warp::store(delta_smem[warp], delta_stage_vec);
        }
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < qTilesBuffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
            const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
            const int scratch_tile_idx = (q_tile_group_idx * 2) * kQSubtilesPerTile + q_subtile_in_group;

            rt_bf<kRefTileM, D> q_reg, do_reg;
            rt_bf<kRefTileM, D, ducks::rt_layout::col> q_col, do_col;
            rt_bf<kRefTileM, kRefTileN> p_bf, ds_bf, peer_ds_bf;
            rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col, ds_col;
            rt_fl<kRefTileM, kRefTileN> p, dp, ds;
            rt_fl<kRefTileM, D> dq_accum;
            stats_vec l_aux_vec, delta_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(l_aux_vec, l_smem[subtile]);
            warp::load(delta_vec, delta_smem[subtile]);

            reconstruct_probability_tile_dense(
                p,
                q_reg,
                k_reg,
                l_aux_vec,
                g.scale_log2e
            );

            warp::copy(p_bf, p);
            warp::swap_layout(p_col, p_bf);
            warp::swap_layout(do_col, do_reg);
            warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);

            warp::zero(dp);
            warp::mma_ABt(dp, do_reg, v_reg, dp);
            warp::sub_row(dp, dp, delta_vec);
            warp::mul(ds, p, dp);
            warp::mul(ds, ds, g.scale);
            warp::copy(ds_bf, ds);
            warp::swap_layout(ds_col, ds_bf);
            warp::swap_layout(q_col, q_reg);
            warp::mma_AtB(dk_accum, ds_col, q_col, dk_accum);

            warp::store(ds_xchg[warp], ds_bf);
            __syncthreads();
            everyone::tma::cluster::arrive_aligned();
            everyone::tma::cluster::wait_aligned();
            constexpr int kDsElems = kWarpTiles * kRefTileM * kRefTileN;
            for (int idx = threadIdx.x; idx < kDsElems; idx += blockDim.x) {
                local_peer_ds_raw[idx] = peer_ds_raw[idx];
            }
            __syncthreads();

            if (cluster_rank == 0) {
                warp::zero(dq_accum);
                warp::mma_AB(dq_accum, ds_bf, k_col, dq_accum);
                warp::load(peer_ds_bf, ds_peer[warp]);
                warp::mma_AB(dq_accum, peer_ds_bf, peer_k_col, dq_accum);
                warp::store(dq_smem[warp], dq_accum);
                __syncwarp();
                warp::tma::store_add_async(g.dq_accum, dq_smem[warp], {batch_idx, q_head_idx, scratch_tile_idx, 0});
                warp::tma::store_async_wait();
            }

            everyone::tma::cluster::arrive_aligned();
            everyone::tma::cluster::wait_aligned();
            __syncthreads();
        }
    }

    warp::store(g.dk, dk_accum, {batch_idx, kv_head_idx, kv_subtile_idx, 0});
    warp::store(g.dv, dv_accum, {batch_idx, kv_head_idx, kv_subtile_idx, 0});
}

template <int D, bool CAUSAL, int CLUSTER_SIZE>
__global__ __launch_bounds__(fused_main_traits<D>::k_block_threads, fused_main_traits<D>::k_min_blocks_per_sm)
void fused_main_kernel(const __grid_constant__ unified_main_globals<D> g) {
    static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2, "Unsupported cluster size.");
    constexpr int kWarpTiles = fused_main_traits<D>::k_warp_tiles;
    constexpr int qTilesBuffered = 4;
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;

    using bf_tile = st_bf<kRefTileM, D>;
    using fl_tile = st_fl<kRefTileM, D>;
    using sv_stats = col_vec<st_fl<kRefTileM, D>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) bf_tile q_smem[qTilesBuffered];
    __shared__ alignas(1024) bf_tile do_smem[qTilesBuffered];
    __shared__ alignas(1024) fl_tile dq_smem[kWarpTiles];
    __shared__ alignas(64) sv_stats l_smem[qTilesBuffered];
    __shared__ alignas(64) sv_stats delta_smem[qTilesBuffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = blockIdx.z;
    const int kv_head_idx = blockIdx.y;
    const int cluster_rank = CLUSTER_SIZE == 2 ? cluster_ctarank() : 0;
    const int kv_block_idx = CLUSTER_SIZE == 2 ? clusterIdx().x * CLUSTER_SIZE + cluster_rank : blockIdx.x;
    const int num_k_blocks = g.seq_len / (kRefTileN * kWarpTiles);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * kWarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;

    rt_bf<kRefTileM, D> k_reg, v_reg;
    rt_bf<kRefTileM, D, ducks::rt_layout::col> k_col;
    rt_fl<kRefTileM, D> dk_accum, dv_accum;
    using vec_t = stats_vec;

    warp::load(k_reg, g.k, {batch_idx, kv_head_idx, kv_subtile_idx, 0});
    warp::load(v_reg, g.v, {batch_idx, kv_head_idx, kv_subtile_idx, 0});
    warp::swap_layout(k_col, k_reg);
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    const int q_head_start = kv_head_idx * g.head_ratio;
    const int q_head_end = q_head_start + g.head_ratio;
    const int num_q_blocks = g.seq_len / (kRefTileM * qTilesBuffered);
    const bool dense_unmasked = !CAUSAL && g.actual_seq_len == g.seq_len;

    for (int q_head_idx = q_head_start; q_head_idx < q_head_end; ++q_head_idx) {
        for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
            const int q_tile_base = q_block_idx * qTilesBuffered;

            if (warp < qTilesBuffered) {
                rt_bf<kRefTileM, D> q_stage_reg, do_stage_reg;
                vec_t l_stage_vec, delta_stage_vec;
                warp::load(q_stage_reg, g.q, {batch_idx, q_head_idx, q_tile_base + warp, 0});
                warp::store(q_smem[warp], q_stage_reg);
                warp::load(do_stage_reg, g.dout, {batch_idx, q_head_idx, q_tile_base + warp, 0});
                warp::store(do_smem[warp], do_stage_reg);
                warp::load(l_stage_vec, g.l_aux, {batch_idx, q_head_idx, 0, q_tile_base + warp});
                warp::store(l_smem[warp], l_stage_vec);
                warp::load(delta_stage_vec, g.delta, {batch_idx, q_head_idx, 0, q_tile_base + warp});
                warp::store(delta_smem[warp], delta_stage_vec);
            }
            __syncthreads();

            #pragma unroll 1
            for (int subtile = 0; subtile < qTilesBuffered; ++subtile) {
                const int q_tile_idx = q_tile_base + subtile;
                rt_bf<kRefTileM, D> q_reg, do_reg;
                rt_fl<kRefTileM, kRefTileN> p, dp, ds;
                rt_fl<kRefTileM, D> dq_accum;
                vec_t l_aux_vec, delta_vec;

                warp::load(q_reg, q_smem[subtile]);
                warp::load(do_reg, do_smem[subtile]);
                warp::load(l_aux_vec, l_smem[subtile]);
                warp::load(delta_vec, delta_smem[subtile]);

                if (dense_unmasked) {
                    reconstruct_probability_tile_dense(
                        p,
                        q_reg,
                        k_reg,
                        l_aux_vec,
                        g.scale_log2e
                    );
                } else {
                    reconstruct_probability_tile(
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

                {
                    rt_bf<kRefTileM, kRefTileN> p_bf;
                    rt_bf<kRefTileM, D, ducks::rt_layout::col> do_col;
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
                warp::mul(ds, ds, g.scale);
                {
                    rt_bf<kRefTileM, kRefTileN> ds_bf;
                    rt_bf<kRefTileM, D, ducks::rt_layout::col> q_col;
                    rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> ds_col;
                    warp::copy(ds_bf, ds);
                    warp::swap_layout(ds_col, ds_bf);
                    warp::swap_layout(q_col, q_reg);
                    warp::mma_AtB(dk_accum, ds_col, q_col, dk_accum);

                    warp::zero(dq_accum);
                    warp::mma_AB(dq_accum, ds_bf, k_col, dq_accum);
                    warp::store(dq_smem[warp], dq_accum);
                }
                __syncwarp();
                const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
                const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
                const int scratch_tile_idx =
                    ((q_tile_group_idx * CLUSTER_SIZE) + cluster_rank) * kQSubtilesPerTile + q_subtile_in_group;
                warp::tma::store_add_async(g.dq_accum, dq_smem[warp], {batch_idx, q_head_idx, scratch_tile_idx, 0});
                warp::tma::store_async_wait();
            }
            __syncthreads();
        }
    }

    warp::store(g.dk, dk_accum, {batch_idx, kv_head_idx, kv_subtile_idx, 0});
    warp::store(g.dv, dv_accum, {batch_idx, kv_head_idx, kv_subtile_idx, 0});
}

template <int D, int CLUSTER_SIZE>
__global__ __launch_bounds__(kWarpThreads, 8)
void dqacc_reduce_kernel(const __grid_constant__ unified_reduce_globals<D> g) {
    const int q_tile_idx = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    constexpr int kQSubtilesPerTile = kForwardTileM / kRefTileM;
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_base =
        (q_tile_group_idx * CLUSTER_SIZE) * kQSubtilesPerTile + q_subtile_in_group;

    rt_fl<kRefTileM, D> dq_reg, dq_partial;
    warp::load(dq_reg, g.dq_accum, {batch_idx, q_head_idx, scratch_tile_base, 0});
    if constexpr (CLUSTER_SIZE == 2) {
        warp::load(dq_partial, g.dq_accum, {batch_idx, q_head_idx, scratch_tile_base + kQSubtilesPerTile, 0});
        warp::add(dq_reg, dq_reg, dq_partial);
    }
    warp::store(g.dq, dq_reg, {batch_idx, q_head_idx, q_tile_idx, 0});
}

template <int D>
inline void launch_main_kernel(
    const unified_main_globals<D> &g,
    bool causal,
    int num_k_blocks,
    int kv_heads,
    int batch_size,
    int cluster_size,
    cudaStream_t stream
) {
    if (cluster_size == 2) {
        kittens::LaunchConfig<true, false> launch_config(
            dim3(num_k_blocks, kv_heads, batch_size),
            dim3(fused_main_traits<D>::k_block_threads, 1, 1),
            0,
            stream,
            dim3(2, 1, 1)
        );
        if (causal) {
            CUDACHECK(cudaLaunchKernelEx(launch_config, fused_main_kernel<D, true, 2>, g));
        } else {
            CUDACHECK(cudaLaunchKernelEx(launch_config, fused_main_kernel<D, false, 2>, g));
        }
    } else {
        dim3 grid(num_k_blocks, kv_heads, batch_size);
        if (causal) {
            fused_main_kernel<D, true, 1><<<grid, fused_main_traits<D>::k_block_threads, 0, stream>>>(g);
        } else {
            fused_main_kernel<D, false, 1><<<grid, fused_main_traits<D>::k_block_threads, 0, stream>>>(g);
        }
    }
}

inline void launch_clustered_dsqacc_bringup_kernel(
    const unified_main_globals<128> &g,
    int num_k_blocks,
    int kv_heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(num_k_blocks, kv_heads, batch_size),
        dim3(256, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(launch_config, clustered_dsqacc_bringup_kernel<false>, g));
}

inline void launch_dense_hot_main_kernel(
    const dense_clustered_wg_globals &g,
    int /*num_k_blocks*/,
    int kv_heads,
    int batch_size,
    cudaStream_t stream
) {
    constexpr int kClusterKvRows =
        wg_hot_tile_dims<128>::tile_h * kWgHotConsumerWarpgroups * kDenseHotClusterSize;
    TORCH_CHECK(
        g.seq_len % kClusterKvRows == 0,
        "Dense hot backward requires sequence length divisible by clustered KV coverage"
    );
    const int num_cluster_blocks = g.seq_len / kClusterKvRows;
    const int total_ctas = num_cluster_blocks * kDenseHotClusterSize;
    kittens::LaunchConfig<true, false> launch_config(
        dim3(total_ctas, kv_heads, batch_size),
        dim3(kDenseHotThreads, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(launch_config, dense_hot_main_kernel<false>, g));
}

inline void launch_dense_hot_reduce_kernel(
    const unified_reduce_globals<128> &g,
    int q_tiles,
    int q_heads,
    int batch_size,
    bool deterministic,
    cudaStream_t stream
) {
    TORCH_CHECK(
        q_tiles % kDenseHotQSubtiles == 0,
        "Dense hot reducer requires q_tiles divisible by 4"
    );
    dim3 grid(q_tiles / kDenseHotQSubtiles, q_heads, batch_size);
    if (deterministic) {
        dense_hot_dqacc_reduce_kernel<true><<<grid, kDenseHotReduceWarps * kWarpThreads, 0, stream>>>(g);
    } else {
        dense_hot_dqacc_reduce_kernel<false><<<grid, kDenseHotReduceWarps * kWarpThreads, 0, stream>>>(g);
    }
}

template <int D>
inline void launch_reduce_kernel(
    const unified_reduce_globals<D> &g,
    int q_tiles,
    int q_heads,
    int batch_size,
    int cluster_size,
    cudaStream_t stream
) {
    dim3 grid(q_tiles, q_heads, batch_size);
    if (cluster_size == 2) {
        dqacc_reduce_kernel<D, 2><<<grid, kWarpThreads, 0, stream>>>(g);
    } else {
        dqacc_reduce_kernel<D, 1><<<grid, kWarpThreads, 0, stream>>>(g);
    }
}

template <int D>
inline void launch_wg_hot_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    cudaStream_t stream
) {
    static_assert(D == 128, "Warpgroup hot backward is specialized for head_dim=128.");

    using G = wg_hot_globals<D>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        kittens::py::tensor_to_gl<typename G::d_gl>(delta, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(1) / k.size(1)),
    };

    dim3 grid(
        static_cast<unsigned int>(q.size(2) / 128),
        static_cast<unsigned int>(q.size(1)),
        static_cast<unsigned int>(q.size(0))
    );
    const int dynamic_smem = wg_hot_dynamic_smem_bytes<D>();
    if (causal) {
        CUDACHECK(cudaFuncSetAttribute(
            wg_hot_backward_kernel<D, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_smem
        ));
        wg_hot_backward_kernel<D, true><<<grid, kWgHotNumWorkers * kittens::WARP_THREADS, dynamic_smem, stream>>>(g);
    } else {
        CUDACHECK(cudaFuncSetAttribute(
            wg_hot_backward_kernel<D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_smem
        ));
        wg_hot_backward_kernel<D, false><<<grid, kWgHotNumWorkers * kittens::WARP_THREADS, dynamic_smem, stream>>>(g);
    }
}

}  // namespace detail

inline int select_backward_cluster_size(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len,
    bool deterministic
) {
    return detail::backward_cluster_size(q, k, causal, actual_seq_len, deterministic);
}

inline bool use_wg_hot_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len
) {
    return detail::use_wg_hot_backward(q, k, causal, actual_seq_len);
}

inline bool use_dense_hot_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len,
    bool deterministic
) {
    return detail::dense_hot_backward_supported(q, k, causal, actual_seq_len, deterministic);
}

template <int D>
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
    at::Tensor &dq_semaphore,
    bool causal,
    float scale,
    int actual_seq_len,
    bool deterministic
) {
    (void)dq_semaphore;

    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    const bool dense_hot_enabled =
        D == 128 &&
        detail::dense_hot_backward_supported(q, k, causal, actual_seq_len, deterministic);

    if constexpr (D == 128) {
        if (!dense_hot_enabled && !deterministic && detail::use_wg_hot_backward(q, k, causal, actual_seq_len)) {
            detail::launch_wg_hot_backward<D>(
                q, k, v, dout, l_aux, delta, dq, dk, dv, causal, scale, stream
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
            return;
        }
    }

    using G = unified_main_globals<D>;
    using dqacc_gl = typename G::dqacc_gl;
    const int cluster_size = static_cast<int>(dq_accum.size(3));
    const int q_tile_groups = static_cast<int>(dq_accum.size(2));
    const int q_tiles = static_cast<int>(q.size(2) / kRefTileM);
    const int dqacc_rows = q_tile_groups * cluster_size * kForwardTileM;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        ::kittens::make_gl<dqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(1)),
            dqacc_rows,
            D
        ),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        kittens::py::tensor_to_gl<typename G::d_gl>(delta, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(2)),
        actual_seq_len,
        static_cast<int>(q.size(1)),
        static_cast<int>(k.size(1)),
        static_cast<int>(q.size(1) / k.size(1)),
    };

    using R = unified_reduce_globals<D>;
    using rdqacc_gl = typename R::dqacc_gl;
    R rg{
        ::kittens::make_gl<rdqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(1)),
            dqacc_rows,
            D
        ),
        kittens::py::tensor_to_gl<typename R::dq_gl>(dq),
    };

    const int num_k_blocks = static_cast<int>(q.size(2) / (kRefTileN * detail::fused_main_traits<D>::k_warp_tiles));

    if constexpr (D == 128) {
        if (dense_hot_enabled) {
            TORCH_CHECK(cluster_size == 2, "Dense hot backward requires cluster_size=2");
            const detail::dense_clustered_wg_globals dg{
                kittens::py::tensor_to_gl<detail::dense_clustered_wg_globals::q_gl>(q),
                kittens::py::tensor_to_gl<detail::dense_clustered_wg_globals::k_gl>(k),
                kittens::py::tensor_to_gl<detail::dense_clustered_wg_globals::v_gl>(v),
                kittens::py::tensor_to_gl<detail::dense_clustered_wg_globals::do_gl>(dout),
                ::kittens::make_gl<detail::dense_clustered_wg_globals::dqacc_gl>(
                    reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
                    static_cast<int>(q.size(0)),
                    static_cast<int>(q.size(1)),
                    dqacc_rows,
                    D
                ),
                kittens::py::tensor_to_gl<detail::dense_clustered_wg_globals::dk_gl>(dk),
                kittens::py::tensor_to_gl<detail::dense_clustered_wg_globals::dv_gl>(dv),
                kittens::py::tensor_to_gl<detail::dense_clustered_wg_globals::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
                kittens::py::tensor_to_gl<detail::dense_clustered_wg_globals::d_gl>(delta, q.size(0), q.size(1), 1, q.size(2)),
                scale,
                scale * kLog2E,
                static_cast<int>(q.size(2)),
            };
            detail::launch_dense_hot_main_kernel(
                dg,
                num_k_blocks,
                static_cast<int>(k.size(1)),
                static_cast<int>(q.size(0)),
                stream
            );
        } else {
            detail::launch_main_kernel(
                g,
                causal,
                num_k_blocks,
                static_cast<int>(k.size(1)),
                static_cast<int>(q.size(0)),
                cluster_size,
                stream
            );
        }
    } else {
        detail::launch_main_kernel(
            g,
            causal,
            num_k_blocks,
            static_cast<int>(k.size(1)),
            static_cast<int>(q.size(0)),
            cluster_size,
            stream
        );
    }
    CHECK_CUDA_ERROR(cudaGetLastError());

    if constexpr (D == 128) {
        if (detail::dense_hot_backward_supported(q, k, causal, actual_seq_len, deterministic)) {
            detail::launch_dense_hot_reduce_kernel(
                rg,
                q_tiles,
                static_cast<int>(q.size(1)),
                static_cast<int>(q.size(0)),
                deterministic,
                stream
            );
        } else {
            detail::launch_reduce_kernel(
                rg, q_tiles, static_cast<int>(q.size(1)), static_cast<int>(q.size(0)), cluster_size, stream
            );
        }
    } else {
        detail::launch_reduce_kernel(
            rg, q_tiles, static_cast<int>(q.size(1)), static_cast<int>(q.size(0)), cluster_size, stream
        );
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd

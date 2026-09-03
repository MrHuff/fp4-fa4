#pragma once

#include "fa4_common.cuh"
#include "fa4_scheduler.cuh"

#include <cstdlib>
#include <cstring>

namespace tkfa4::fwd {

enum class forward_mode {
    cluster,
    ref,
};

inline forward_mode resolve_forward_mode() {
    const char *mode = std::getenv("TK_FA4_FWD_MODE");
    if (mode == nullptr) {
        return forward_mode::cluster;
    }
    if (std::strcmp(mode, "ref") == 0) {
        return forward_mode::ref;
    }
    return forward_mode::cluster;
}

inline int resolve_force_2cta() {
    const char *value = std::getenv("TK_FA4_FORCE_2CTA");
    if (value == nullptr) {
        return -1;
    }
    if (std::strcmp(value, "0") == 0) {
        return 0;
    }
    if (std::strcmp(value, "1") == 0) {
        return 1;
    }
    return -1;
}

template <int D>
struct ref_globals {
    using q_gl = gl<bf16, -1, -1, -1, D>;
    using k_gl = gl<bf16, -1, -1, -1, D>;
    using v_gl = gl<bf16, -1, -1, -1, D>;
    using o_gl = gl<bf16, -1, -1, -1, D>;
    using l_tile = col_vec<st_fl<kRefTileM, D>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    l_gl l_aux;
    float scale;
    float scale_log2e;
    int actual_seq_len;
    int head_ratio;
};

template <int D, bool CAUSAL>
__global__ __launch_bounds__(kWarpThreads, 8)
void kernel_ref(const __grid_constant__ ref_globals<D> g) {
    const int q_tile_idx = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int kv_head_idx = scheduler::q_head_to_kv_head(q_head_idx, g.head_ratio);

    rt_bf<kRefTileM, D> q_reg, k_reg, v_reg, o_bf;
    rt_bf<kRefTileM, kRefTileN> probs_bf;
    rt_bf<kRefTileM, D, ducks::rt_layout::col> v_reg_col;
    rt_fl<kRefTileM, kRefTileN> scores;
    rt_fl<kRefTileM, D> out_accum;
    using vec_t = typename rt_fl<kRefTileM, kRefTileN>::col_vec;
    vec_t max_vec, max_prev, norm_vec, alpha, lse_vec;

    warp::load(q_reg, g.q, {batch_idx, q_head_idx, q_tile_idx, 0});
    warp::zero(out_accum);
    warp::zero(norm_vec);
    warp::neg_infty(max_vec);

    const int num_k_tiles = ceil_div(g.actual_seq_len, kRefTileN);
    const int n_block_max = CAUSAL && (q_tile_idx + 1) < num_k_tiles ? (q_tile_idx + 1) : num_k_tiles;
    for (int k_tile_idx = 0; k_tile_idx < n_block_max; ++k_tile_idx) {
        warp::load(k_reg, g.k, {batch_idx, kv_head_idx, k_tile_idx, 0});
        warp::load(v_reg, g.v, {batch_idx, kv_head_idx, k_tile_idx, 0});
        warp::swap_layout(v_reg_col, v_reg);

        warp::zero(scores);
        warp::mma_ABt(scores, q_reg, k_reg, scores);
        warp::mul(scores, scores, g.scale_log2e);
        apply_reference_mask(scores, q_tile_idx, k_tile_idx, g.actual_seq_len, CAUSAL);

        warp::copy(max_prev, max_vec);
        warp::row_max(max_vec, scores, max_vec);
        warp::sub(alpha, max_prev, max_vec);
        warp::exp2(alpha, alpha);
        warp::sub_row(scores, scores, max_vec);
        warp::exp2(scores, scores);

        warp::mul_row(out_accum, out_accum, alpha);
        warp::mul(norm_vec, norm_vec, alpha);
        warp::row_sum(norm_vec, scores, norm_vec);
        warp::copy(probs_bf, scores);
        warp::mma_AB(out_accum, probs_bf, v_reg_col, out_accum);
    }

    warp::div_row(out_accum, out_accum, norm_vec);
    warp::copy(o_bf, out_accum);
    warp::store(g.o, o_bf, {batch_idx, q_head_idx, q_tile_idx, 0});

    warp::mul(lse_vec, max_vec, 0.6931471805599453f);
    warp::log(norm_vec, norm_vec);
    warp::add(lse_vec, lse_vec, norm_vec);
    warp::mul(lse_vec, lse_vec, -1.0f / g.scale);
    warp::store(g.l_aux, lse_vec, {batch_idx, q_head_idx, 0, q_tile_idx});
}

constexpr int kPipeQStages = 2;
constexpr int kPipeNumWarpgroups = 4;
constexpr int kPipeNumWarps = kPipeNumWarpgroups * kittens::WARPGROUP_WARPS;
constexpr int kPipeNumThreads = kPipeNumWarps * kittens::WARP_THREADS;
constexpr bool kPipeEnableDispatch = true;

template <int D>
inline constexpr int pipe_kv_stages_v = 2;

template <int D>
struct pipe_globals {
    using q_tile = st_bf<kForwardSubtileM, D>;
    using kv_tile = st_bf<kForwardTileN, D>;
    using v_tile = st_bf<kForwardTileN, D>;
    using o_tile = st_bf<kForwardSubtileM, D>;
    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, kv_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using o_gl = gl<bf16, -1, -1, -1, D>;
    using l_tile = col_vec<st_fl<16, D>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    l_gl l_aux;
    float scale;
    float scale_log2e;
    int actual_seq_len;
    int padded_seq_len;
    int batch_size;
    int q_heads;
    int head_ratio;
};

template <int D>
struct pipe2cta_globals {
    using q_tile = st_bf<kForwardTileM, D>;
    using k_tile = st_bf<kForwardTileN / 2, D>;
    using v_tile = st_bf<D, kForwardTileN / 2>;
    using o_tile = st_bf<kForwardTileM, D>;
    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using o_gl = gl<bf16, -1, -1, -1, D>;
    using l_tile = col_vec<st_fl<32, D>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    l_gl l_aux;
    float scale;
    float scale_log2e;
    int actual_seq_len;
    int padded_seq_len;
    int batch_size;
    int q_heads;
    int head_ratio;
    int cluster_tile_idx;
};

template <bool CAUSAL>
__device__ inline void mask_score_stage(
    rt_fl<16, 128> &scores,
    int q_stage,
    int warp_in_group,
    int m_block,
    int n_block,
    int actual_seq_len
) {
    constexpr float neg_inf = kittens::base_types::constants<float>::neg_infty();
    const int row_base =
        m_block * kForwardTileM +
        q_stage * kForwardSubtileM +
        warp_in_group * 16;
    warp::apply(scores, scores, [=](int row, int col, float value) {
        const int q_idx = row_base + row;
        const int k_idx = n_block * kForwardTileN + col;
        if (k_idx >= actual_seq_len) {
            return neg_inf;
        }
        if constexpr (CAUSAL) {
            if (k_idx > q_idx) {
                return neg_inf;
            }
        }
        return value;
    });
}

template <bool CAUSAL>
__device__ inline void mask_score_tile_2cta(
    rt_fl<32, 128> &scores,
    int warp_in_group,
    int cta_m_block,
    int n_block,
    int actual_seq_len
) {
    constexpr float neg_inf = kittens::base_types::constants<float>::neg_infty();
    const int row_base = cta_m_block * kForwardTileM + warp_in_group * 32;
    warp::apply(scores, scores, [=](int row, int col, float value) {
        const int q_idx = row_base + row;
        const int k_idx = n_block * kForwardTileN + col;
        if (k_idx >= actual_seq_len) {
            return neg_inf;
        }
        if constexpr (CAUSAL) {
            if (k_idx > q_idx) {
                return neg_inf;
            }
        }
        return value;
    });
}

constexpr int kPipe2CtaClusterSize = 2;
constexpr int kPipe2CtaNumWarpgroups = 4;
constexpr int kPipe2CtaNumWarps = kPipe2CtaNumWarpgroups * kittens::WARPGROUP_WARPS;
constexpr int kPipe2CtaNumThreads = kPipe2CtaNumWarps * kittens::WARP_THREADS;
constexpr int kPipe2CtaSoftmax0WarpStart = 0;
constexpr int kPipe2CtaSoftmax1WarpStart = 4;
constexpr int kPipe2CtaCorrectionWarpStart = 8;
constexpr int kPipe2CtaMmaWarp = 12;
constexpr int kPipe2CtaEpilogueWarp = 13;
constexpr int kPipe2CtaLoadWarp = 14;
constexpr int kPipe2CtaEmptyWarp = 15;
constexpr int kPipe2CtaLaunchSmemSafetyBytes = 0;
constexpr int kPipe2CtaDebugStage = 0;
constexpr int kPipe2CtaDebugSubstage = 0;
constexpr bool kPipe2CtaTraceScores = false;

__host__ __device__ constexpr int pipe2cta_score_tmem_col(int stage) {
    return stage * kForwardTileN;
}

__host__ __device__ constexpr int pipe2cta_prob_tmem_col(int stage) {
    return pipe2cta_score_tmem_col(stage);
}

template <int D>
__host__ __device__ constexpr int pipe2cta_out_tmem_col(int stage) {
    return 2 * kForwardTileN + stage * D;
}

template <int D>
inline constexpr int pipe2cta_kv_stages_v =
    []() constexpr {
        using G = pipe2cta_globals<D>;
        constexpr int q_bytes = sizeof(typename G::q_tile) * kPipeQStages;
        constexpr int o_bytes = sizeof(typename G::o_tile);
        constexpr int base_bytes = q_bytes + o_bytes + 1024;
        constexpr int kv_bytes_per_stage =
            sizeof(typename G::k_tile) + sizeof(typename G::v_tile);
        constexpr int launchable_budget =
            MAX_SHARED_MEMORY - 1024 - kPipe2CtaLaunchSmemSafetyBytes;
        constexpr int max_stage_budget = (launchable_budget - base_bytes) / kv_bytes_per_stage;
        constexpr int clamped_stage_budget =
            max_stage_budget >= 3 ? 3 :
            (max_stage_budget >= 2 ? 2 : 1);
        return clamped_stage_budget;
    }();

template <int D>
inline constexpr uint32_t pipe2cta_kv_expected_bytes_total_v =
    kPipe2CtaClusterSize * (sizeof(typename pipe2cta_globals<D>::k_tile) +
                            sizeof(typename pipe2cta_globals<D>::v_tile));

template <int D, int KVStages>
inline int pipe2cta_dynamic_smem_bytes() {
    using G = pipe2cta_globals<D>;
    constexpr int bytes =
        sizeof(typename G::q_tile) * kPipeQStages +
        sizeof(typename G::o_tile) +
        (sizeof(typename G::k_tile) + sizeof(typename G::v_tile)) * KVStages +
        1024;
    static_assert(bytes <= MAX_SHARED_MEMORY - 1024, "2CTA pipe forward shared memory exceeds device limit");
    return bytes;
}

template <int D>
__device__ inline void zero_q_stage_tile(typename pipe2cta_globals<D>::q_tile &tile, int warp_in_group) {
    rt_bf<32, D> zero_chunk;
    warp::zero(zero_chunk);
    auto q_view = tile.template subtile<32, D>({warp_in_group, 0});
    warp::store(q_view, zero_chunk);
}

template <int D, int KVStages>
__device__ inline typename pipe2cta_globals<D>::k_tile &pipe2cta_k_stage(
    typename pipe2cta_globals<D>::k_tile (&kv_smem)[KVStages],
    int stage
) {
    return kv_smem[stage];
}

template <int D, int KVStages>
__device__ inline typename pipe2cta_globals<D>::v_tile &pipe2cta_v_stage(
    typename pipe2cta_globals<D>::k_tile (&kv_smem)[KVStages],
    int stage
) {
    return *reinterpret_cast<typename pipe2cta_globals<D>::v_tile *>(&kv_smem[stage]);
}

template <int D>
__device__ __noinline__ void pipe2cta_scale_out_tmem(
    full_tt_fl<D> &out_tt_tile,
    const typename rt_fl<32, kForwardTileN>::col_vec &alpha
) {
    constexpr int kOutChunkCols = 64;
    static_assert(D % kOutChunkCols == 0);
    using out_rt_chunk = rt_fl<32, kOutChunkCols>;
    using out_tt_chunk = full_tt_fl<kOutChunkCols>;

    out_rt_chunk out_partial;
    #pragma unroll
    for (int chunk = 0; chunk < D / kOutChunkCols; ++chunk) {
        auto out_tt_chunk_view = out_tt_tile.template subtile<out_tt_chunk>(0, kOutChunkCols * chunk);
        warp::zero(out_partial);
        warpgroup::load_async(out_partial, out_tt_chunk_view);
        tensor_load_wait();
        tensor_before_thread_sync();
        warp::mul_row(out_partial, out_partial, alpha);
        warpgroup::store_async(out_tt_chunk_view, out_partial);
        tensor_store_wait();
        tensor_before_thread_sync();
    }
}

template <int D>
__device__ __noinline__ void pipe2cta_load_out_tmem(
    rt_fl<32, D> &out_partial,
    full_tt_fl<D> &out_tt_tile
) {
    warp::zero(out_partial);
    warpgroup::load_async(out_partial, out_tt_tile);
    tensor_load_wait();
    tensor_before_thread_sync();
}

template <int D>
__device__ __noinline__ void pipe2cta_issue_pv_initial(
    full_tt_fl<D> &out_tt_tile,
    full_tt_bf<kForwardTileN> &probs_tt,
    typename pipe2cta_globals<D>::v_tile &v_smem,
    kittens::semaphore &inputs_finished
) {
    mm2_AB(out_tt_tile, probs_tt, v_smem, inputs_finished);
    tensor_after_thread_sync();
}

template <int D>
__device__ __noinline__ void pipe2cta_issue_pv_accum(
    full_tt_fl<D> &out_tt_tile,
    full_tt_bf<kForwardTileN> &probs_tt,
    typename pipe2cta_globals<D>::v_tile &v_smem,
    kittens::semaphore &inputs_finished
) {
    mma2_AB(out_tt_tile, probs_tt, v_smem, inputs_finished);
    tensor_after_thread_sync();
}

template <int KVStages>
__device__ inline int cluster_input_phase(int iter) {
    return (iter / KVStages) & 1;
}

template <int KVStages>
__device__ inline int cluster_finished_phase(int iter) {
    return ((iter / KVStages) + 1) & 1;
}

template <int KVStages>
__device__ inline int input_ring_slot(int iter) {
    return iter % KVStages;
}

__device__ inline void arrive_both_ctas(kittens::semaphore &sem) {
    tma::cluster::arrive(sem, 0);
    tma::cluster::arrive(sem, 1);
}

template <int D>
__device__ __noinline__ void pipe2cta_correction_rescale_role(
    int block_count,
    full_tt_fl<D> (&out_tt_tile)[kPipeQStages],
    const bool (&local_valid)[kPipeQStages],
    kittens::semaphore (&stats_outputs_arrived)[kPipeQStages],
    kittens::semaphore (&stats_outputs_finished)[kPipeQStages],
    kittens::semaphore (&out_outputs_finished)[kPipeQStages],
    sv_fl<kForwardTileM> (&scale_stats)[kPipeQStages][2]
) {
    using vec_t = typename rt_fl<32, kForwardTileN>::col_vec;

    const int warp_in_group = warpgroup::warpid();
    vec_t alpha;

    for (int iter = 0; iter < block_count; ++iter) {
        const int stage_phase = iter & 1;
        #pragma unroll
        for (int stage = 0; stage < kPipeQStages; ++stage) {
            if (!local_valid[stage]) {
                continue;
            }
            wait(stats_outputs_arrived[stage], stage_phase);
            kittens::group<kittens::WARPGROUP_WARPS>::load(alpha, scale_stats[stage][0]);
            if (iter >= 1) {
                pipe2cta_scale_out_tmem(out_tt_tile[stage], alpha);
            }
            warpgroup::sync(13);
            if (warp_in_group == 0 && laneid() == 0) {
                arrive(stats_outputs_finished[stage]);
                tma::cluster::arrive(out_outputs_finished[stage], 0);
            }
        }
    }
}

template <int D>
__device__ __noinline__ void pipe2cta_finalize_role(
    const pipe2cta_globals<D> &g,
    int batch_idx,
    int q_head_idx,
    const int (&cta_m_blocks)[kPipeQStages],
    const bool (&local_valid)[kPipeQStages],
    int block_count,
    full_tt_fl<D> (&out_tt_tile)[kPipeQStages],
    kittens::semaphore (&out_outputs_arrived)[kPipeQStages],
    kittens::semaphore (&final_out_ready)[kPipeQStages],
    kittens::semaphore &o_epi_ready,
    kittens::semaphore &o_epi_finished,
    sv_fl<kForwardTileM> (&scale_stats)[kPipeQStages][2],
    typename pipe2cta_globals<D>::o_tile &o_smem
) {
    using vec_t = typename rt_fl<32, kForwardTileN>::col_vec;
    constexpr int kOutChunkCols = 64;
    static_assert(D % kOutChunkCols == 0);
    using out_rt_chunk = rt_fl<32, kOutChunkCols>;
    using out_bf_chunk = rt_bf<32, kOutChunkCols>;
    using out_tt_chunk = full_tt_fl<kOutChunkCols>;

    const int warp_in_group = warpgroup::warpid();
    vec_t norm_vec, max_vec, lse_vec;

    #pragma unroll
    for (int stage = 0; stage < kPipeQStages; ++stage) {
        if (!local_valid[stage]) {
            continue;
        }
        wait(o_epi_finished, stage & 1);
        const int cta_m_block = cta_m_blocks[stage];
        wait(final_out_ready[stage], 0);
        if (block_count > 0) {
            wait(out_outputs_arrived[stage], (block_count - 1) & 1);
        }
        kittens::group<kittens::WARPGROUP_WARPS>::load(norm_vec, scale_stats[stage][0]);
        kittens::group<kittens::WARPGROUP_WARPS>::load(max_vec, scale_stats[stage][1]);
        #pragma unroll
        for (int chunk = 0; chunk < D / kOutChunkCols; ++chunk) {
            out_rt_chunk out_partial;
            out_bf_chunk out_store;
            auto out_tt_chunk_view = out_tt_tile[stage].template subtile<out_tt_chunk>(0, kOutChunkCols * chunk);
            warp::zero(out_partial);
            warpgroup::load_async(out_partial, out_tt_chunk_view);
            tensor_load_wait();
            tensor_before_thread_sync();
            warp::div_row(out_partial, out_partial, norm_vec);
            warp::copy(out_store, out_partial);
            auto o_view = o_smem.template subtile<32, kOutChunkCols>({warp_in_group, chunk});
            warp::store(o_view, out_store);
        }
        warp::mul(lse_vec, max_vec, 0.6931471805599453f);
        warp::log(norm_vec, norm_vec);
        warp::add(lse_vec, lse_vec, norm_vec);
        warp::mul(lse_vec, lse_vec, -1.0f / g.scale);
        warp::store(
            g.l_aux,
            lse_vec,
            {batch_idx, q_head_idx, 0, cta_m_block * kittens::WARPGROUP_WARPS + warp_in_group}
        );
        warpgroup::sync(13);
        if (warp_in_group == 0 && laneid() == 0) {
            arrive(o_epi_ready);
        }
    }
}

template <int D>
__device__ __noinline__ void pipe2cta_epilogue_role(
    const pipe2cta_globals<D> &g,
    int batch_idx,
    int q_head_idx,
    const int (&cta_m_blocks)[kPipeQStages],
    const bool (&local_valid)[kPipeQStages],
    kittens::semaphore &o_epi_ready,
    kittens::semaphore &o_epi_finished,
    typename pipe2cta_globals<D>::o_tile &o_smem
) {
    using out_bf = rt_bf<32, D>;

    out_bf out_store;
    for (int stage = 0; stage < kPipeQStages; ++stage) {
        if (!local_valid[stage]) {
            continue;
        }
        wait(o_epi_ready, stage & 1);
        const int cta_m_block = cta_m_blocks[stage];
        #pragma unroll
        for (int subtile = 0; subtile < kittens::WARPGROUP_WARPS; ++subtile) {
            auto o_view = o_smem.template subtile<32, D>({subtile, 0});
            warp::load(out_store, o_view);
            warp::store(
                g.o,
                out_store,
                {batch_idx, q_head_idx, cta_m_block * kittens::WARPGROUP_WARPS + subtile, 0}
            );
        }
        if (laneid() == 0) {
            arrive(o_epi_finished);
        }
    }
}

template <int D, bool CAUSAL, int QStageIdx>
__device__ __noinline__ void pipe2cta_softmax_step(
    const pipe2cta_globals<D> &g,
    int batch_idx,
    int q_head_idx,
    int cta_m_block,
    int n_block_max,
    int iter,
    full_tt_fl<kForwardTileN> &scores_tt,
    full_tt_bf<kForwardTileN> &probs_tt,
    kittens::semaphore &score_outputs_arrived,
    kittens::semaphore &score_outputs_finished,
    kittens::semaphore &stats_outputs_arrived,
    kittens::semaphore &stats_outputs_finished,
    kittens::semaphore &prob_outputs_arrived,
    kittens::semaphore &prob_outputs_finished,
    sv_fl<kForwardTileM> (&scale_stats)[2],
    sv_fl<kForwardTileM> (&softmax_running)[2]
) {
    static_assert(QStageIdx >= 0 && QStageIdx < kPipeQStages);
    using score_rt = rt_fl<32, kForwardTileN>;
    using prob_rt = rt_bf<32, kForwardTileN>;
    using vec_t = typename score_rt::col_vec;

    const int warp_in_group = warpgroup::warpid();
    const int n_block = n_block_max - 1 - iter;
    const int stage_phase = iter & 1;
    score_rt scores;
    prob_rt probs_bf;
    vec_t alpha, max_vec, norm_vec;

    warpgroup::sync(14 + QStageIdx);
    wait(score_outputs_arrived, stage_phase);
    warpgroup::load_async(scores, scores_tt);
    tensor_load_wait();
    tensor_before_thread_sync();
    kittens::group<kittens::WARPGROUP_WARPS>::load(norm_vec, softmax_running[0]);
    kittens::group<kittens::WARPGROUP_WARPS>::load(max_vec, softmax_running[1]);
    if (warp_in_group == 0 && laneid() == 0) {
        tma::cluster::arrive(score_outputs_finished, 0);
    }
    warp::mul(scores, scores, g.scale_log2e);

    const bool tail_tile = scheduler::forward_valid_cols(n_block, g.actual_seq_len) < kForwardTileN;
    if (tail_tile) {
        mask_score_tile_2cta<CAUSAL>(scores, warp_in_group, cta_m_block, n_block, g.actual_seq_len);
    }

    if constexpr (kPipe2CtaTraceScores) {
        rt_bf<32, D> out_store;
        warp::copy(out_store, scores);
        warp::store(
            g.o,
            out_store,
            {batch_idx, q_head_idx, cta_m_block * kittens::WARPGROUP_WARPS + warp_in_group, 0}
        );
        warp::row_max(max_vec, scores, max_vec);
        return;
    }

    warp::copy(alpha, max_vec);
    warp::row_max(max_vec, scores, max_vec);
    warp::sub(alpha, alpha, max_vec);
    warp::exp2(alpha, alpha);
    warp::sub_row(scores, scores, max_vec);
    warp::exp2(scores, scores);

    warp::mul(norm_vec, norm_vec, alpha);
    warp::row_sum(norm_vec, scores, norm_vec);
    if (iter >= 1) {
        wait(stats_outputs_finished, (iter - 1) & 1);
    }
    kittens::group<kittens::WARPGROUP_WARPS>::store(scale_stats[0], alpha);
    kittens::group<kittens::WARPGROUP_WARPS>::store(softmax_running[0], norm_vec);
    kittens::group<kittens::WARPGROUP_WARPS>::store(softmax_running[1], max_vec);
    warpgroup::sync(14 + QStageIdx);
    if (warp_in_group == 0 && laneid() == 0) {
        arrive(stats_outputs_arrived);
    }
    warp::copy(probs_bf, scores);
    if (iter >= 1) {
        wait(prob_outputs_finished, (iter - 1) & 1);
        tensor_before_thread_sync();
    }
    warpgroup::sync(14 + QStageIdx);
    warpgroup::store_async(probs_tt, probs_bf);
    tensor_store_wait();
    tensor_before_thread_sync();
    warpgroup::sync(14 + QStageIdx);
    if (warp_in_group == 0 && laneid() == 0) {
        tma::cluster::arrive(prob_outputs_arrived, 0);
    }
}

template <int D, bool CAUSAL, int QStageIdx>
__device__ __noinline__ void pipe2cta_softmax_role(
    const pipe2cta_globals<D> &g,
    int batch_idx,
    int q_head_idx,
    int cta_m_block,
    bool stage_valid,
    int block_count,
    int n_block_max,
    full_tt_fl<kForwardTileN> &scores_tt,
    full_tt_bf<kForwardTileN> &probs_tt,
    kittens::semaphore &score_outputs_arrived,
    kittens::semaphore &score_outputs_finished,
    kittens::semaphore &stats_outputs_arrived,
    kittens::semaphore &stats_outputs_finished,
    kittens::semaphore &prob_outputs_arrived,
    kittens::semaphore &prob_outputs_finished,
    kittens::semaphore &final_out_ready,
    sv_fl<kForwardTileM> (&scale_stats)[2],
    sv_fl<kForwardTileM> (&softmax_running)[2]
) {
    static_assert(QStageIdx >= 0 && QStageIdx < kPipeQStages);
    using score_rt = rt_fl<32, kForwardTileN>;
    using vec_t = typename score_rt::col_vec;

    const int warp_in_group = warpgroup::warpid();

    if (!stage_valid) {
        if (warp_in_group == 0 && laneid() == 0) {
            arrive(final_out_ready);
        }
        return;
    }

    vec_t init_norm, init_max, final_norm, final_max;
    warp::zero(init_norm);
    warp::neg_infty(init_max);
    kittens::group<kittens::WARPGROUP_WARPS>::store(softmax_running[0], init_norm);
    kittens::group<kittens::WARPGROUP_WARPS>::store(softmax_running[1], init_max);
    warpgroup::sync(14 + QStageIdx);

    for (int iter = 0; iter < block_count; ++iter) {
        pipe2cta_softmax_step<D, CAUSAL, QStageIdx>(
            g,
            batch_idx,
            q_head_idx,
            cta_m_block,
            n_block_max,
            iter,
            scores_tt,
            probs_tt,
            score_outputs_arrived,
            score_outputs_finished,
            stats_outputs_arrived,
            stats_outputs_finished,
            prob_outputs_arrived,
            prob_outputs_finished,
            scale_stats,
            softmax_running
        );
        if constexpr (kPipe2CtaTraceScores) {
            break;
        }
    }

    kittens::group<kittens::WARPGROUP_WARPS>::load(final_norm, softmax_running[0]);
    kittens::group<kittens::WARPGROUP_WARPS>::load(final_max, softmax_running[1]);
    if constexpr (kPipe2CtaTraceScores) {
        warp::store(
            g.l_aux,
            final_max,
            {batch_idx, q_head_idx, 0, cta_m_block * kittens::WARPGROUP_WARPS + warp_in_group}
        );
    } else {
        kittens::group<kittens::WARPGROUP_WARPS>::store(scale_stats[0], final_norm);
        kittens::group<kittens::WARPGROUP_WARPS>::store(scale_stats[1], final_max);
    }

    warpgroup::sync(14 + QStageIdx);
    if (warp_in_group == 0 && laneid() == 0) {
        arrive(final_out_ready);
    }
}
template <int D, int KVStages, bool CAUSAL>
__maxnreg__(128)
__global__
void kernel_pipe_2cta_unified(const __grid_constant__ pipe2cta_globals<D> g) {
    static_assert(D == 128, "kernel_pipe_2cta_unified is specialized for head_dim 128");

    extern __shared__ int __shm[];
    tma_swizzle_allocator al(reinterpret_cast<int*>(&__shm[0]));

    using q_tile = typename pipe2cta_globals<D>::q_tile;
    using k_tile = typename pipe2cta_globals<D>::k_tile;
    using v_tile = typename pipe2cta_globals<D>::v_tile;
    using o_tile = typename pipe2cta_globals<D>::o_tile;
    q_tile (&q_smem)[kPipeQStages] = al.allocate<q_tile, kPipeQStages>();
    k_tile (&k_smem)[KVStages] = al.allocate<k_tile, KVStages>();
    v_tile (&v_smem)[KVStages] = al.allocate<v_tile, KVStages>();
    o_tile &o_smem = al.allocate<o_tile>();

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile>();
        g.k.template prefetch_tma<k_tile>();
        g.v.template prefetch_tma<v_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int warp_in_group = warpgroup::warpid();
    const int warp_id = kittens::warpid();
    const int cta_rank = cluster_ctarank();
    const bool is_consumer = warp_id < kPipe2CtaCorrectionWarpStart;
    const bool is_correction = warp_id >= kPipe2CtaCorrectionWarpStart && warp_id < kPipe2CtaMmaWarp;
    const bool is_mma = warp_id == kPipe2CtaMmaWarp;
    const bool is_epilogue = warp_id == kPipe2CtaEpilogueWarp;
    const bool is_load = warp_id == kPipe2CtaLoadWarp;
    const int q_stage_idx = warp_id < kPipe2CtaSoftmax1WarpStart ? 0 : (warp_id < kPipe2CtaCorrectionWarpStart ? 1 : -1);
    (void)warpgroup_id;

    if constexpr (kPipe2CtaDebugStage == 3) {
        return;
    }

    __shared__ __align__(16) kittens::semaphore tmem_provisioned;
    __shared__ uint32_t tmem_addr;
    __shared__ __align__(16) kittens::semaphore q_arrived[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore kv_reload_ready[KVStages];
    __shared__ __align__(16) kittens::semaphore kv_inputs_arrived[KVStages];
    __shared__ __align__(16) kittens::semaphore inputs_finished[KVStages];
    __shared__ __align__(16) kittens::semaphore score_outputs_arrived[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore score_outputs_finished[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore stats_outputs_arrived[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore stats_outputs_finished[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore prob_outputs_arrived[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore prob_outputs_finished[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore out_outputs_arrived[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore out_outputs_finished[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore final_out_ready[kPipeQStages];
    __shared__ __align__(16) kittens::semaphore o_epi_ready;
    __shared__ __align__(16) kittens::semaphore o_epi_finished;
    __shared__ sv_fl<kForwardTileM> scale_stats[kPipeQStages][2];
    __shared__ sv_fl<kForwardTileM> softmax_running[kPipeQStages][2];

    tensor_allocator<1, 2, false> tm_alloc{};
    using score_tt = full_tt_fl<kForwardTileN>;
    using prob_tt = full_tt_bf<kForwardTileN>;
    using out_tt = full_tt_fl<D>;
    score_tt scores_tt[kPipeQStages] = {score_tt{0}, score_tt{0}};
    prob_tt probs_tt[kPipeQStages] = {prob_tt{0}, prob_tt{0}};
    out_tt out_tt_tile[kPipeQStages] = {out_tt{0}, out_tt{0}};

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
    }
    __syncthreads();

    if constexpr (kPipe2CtaDebugStage != 3) {
        everyone::tma::cluster::arrive_aligned();
        everyone::tma::cluster::wait_aligned();
        if (warp_id == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);

        if (is_consumer || is_mma || is_correction) {
            #pragma unroll
            for (int stage = 0; stage < kPipeQStages; ++stage) {
                scores_tt[stage] = tm_alloc.template allocate<score_tt>(pipe2cta_score_tmem_col(stage));
                probs_tt[stage] = tm_alloc.template allocate<prob_tt>(pipe2cta_prob_tmem_col(stage));
                out_tt_tile[stage] = tm_alloc.template allocate<out_tt>(pipe2cta_out_tmem_col<D>(stage));
            }
        }
    } else {
        __syncthreads();
    }

    const int num_m_blocks = g.padded_seq_len / kForwardTileM;
    const int kv_heads = g.q_heads / g.head_ratio;
    const int total_cluster_tiles = scheduler::forward_cluster_task_total_tiles(
        g.batch_size,
        kv_heads,
        g.head_ratio,
        num_m_blocks,
        kPipeQStages,
        kPipe2CtaClusterSize
    );

    using score_rt = rt_fl<32, kForwardTileN>;
    using prob_rt = rt_bf<32, kForwardTileN>;
    using out_rt = rt_fl<32, D>;
    using out_bf = rt_bf<32, D>;
    using vec_t = typename score_rt::col_vec;

    const int cluster_idx = static_cast<int>(blockIdx.x) / kPipe2CtaClusterSize;
    const int cluster_stride =
        static_cast<int>(gridDim.x) / kPipe2CtaClusterSize > 0
            ? static_cast<int>(gridDim.x) / kPipe2CtaClusterSize
            : 1;
    for (int cluster_tile = cluster_idx; cluster_tile < total_cluster_tiles; cluster_tile += cluster_stride) {
        const auto tile = scheduler::decode_forward_cluster_task(
            cluster_tile,
            g.batch_size,
            kv_heads,
            g.head_ratio,
            num_m_blocks,
            kPipeQStages,
            kPipe2CtaClusterSize,
            CAUSAL
        );
        const int batch_idx = tile.batch_idx;
        const int q_head_idx = tile.q_head_idx;
        const int kv_head_idx = tile.kv_head_idx;
        int cta_m_blocks[kPipeQStages];
        bool local_valid[kPipeQStages];
        #pragma unroll
        for (int stage = 0; stage < kPipeQStages; ++stage) {
            cta_m_blocks[stage] = scheduler::forward_cluster_stage_m_block(
                tile.m_block_base,
                stage,
                cta_rank,
                num_m_blocks,
                kPipeQStages,
                kPipe2CtaClusterSize,
                CAUSAL
            );
            local_valid[stage] = cta_m_blocks[stage] < num_m_blocks;
        }
        int valid_stage_count = 0;
        #pragma unroll
        for (int stage = 0; stage < kPipeQStages; ++stage) {
            valid_stage_count += local_valid[stage] ? 1 : 0;
        }
        int n_block_max = 0;
        #pragma unroll
        for (int stage = 0; stage < kPipeQStages; ++stage) {
            if (local_valid[stage]) {
                const int stage_n_block_max = scheduler::forward_cta_n_block_max(
                    cta_m_blocks[stage],
                    g.actual_seq_len,
                    g.actual_seq_len,
                    CAUSAL
                );
                n_block_max = stage_n_block_max > n_block_max ? stage_n_block_max : n_block_max;
            }
        }
        const int n_block_min = 0;
        const int block_count = n_block_max - n_block_min;

        if (threadIdx.x == 32) {
                for (int stage = 0; stage < kPipeQStages; ++stage) {
                    init_semaphore(q_arrived[stage], 0, 1);
                    init_semaphore(final_out_ready[stage], 0, 1);
                    init_semaphore(score_outputs_arrived[stage], 0, 1);
                    init_semaphore(score_outputs_finished[stage], 0, 2);
                    init_semaphore(stats_outputs_arrived[stage], 0, 1);
                    init_semaphore(stats_outputs_finished[stage], 0, 1);
                    init_semaphore(prob_outputs_arrived[stage], 0, 2);
                    init_semaphore(prob_outputs_finished[stage], 0, 1);
                    init_semaphore(out_outputs_arrived[stage], 0, 1);
                    init_semaphore(out_outputs_finished[stage], 0, 2);
                }
                init_semaphore(o_epi_ready, 0, 1);
                init_semaphore(o_epi_finished, 0, 1);
                arrive(o_epi_finished);
            for (int stage = 0; stage < KVStages; ++stage) {
                init_semaphore(kv_reload_ready[stage], 0, 1);
                init_semaphore(kv_inputs_arrived[stage], 0, 1);
                init_semaphore(inputs_finished[stage], 0, valid_stage_count > 0 ? valid_stage_count : 1);
            }
        }
        __syncthreads();

        if (is_load) {
            #pragma unroll
            for (int stage = 0; stage < kPipeQStages; ++stage) {
                if (!local_valid[stage]) {
                    zero_q_stage_tile<D>(q_smem[stage], warp_in_group);
                }
            }
        }

        if (is_load && laneid() == 0) {
            for (int stage = 0; stage < kPipeQStages; ++stage) {
                if constexpr (kPipe2CtaDebugStage != 3) {
                    if (local_valid[stage]) {
                        tma::expect_bytes(q_arrived[stage], sizeof(q_tile));
                        tma::load_async(
                            q_smem[stage],
                            g.q,
                            {batch_idx, q_head_idx, cta_m_blocks[stage], 0},
                            q_arrived[stage]
                        );
                    } else {
                        arrive(q_arrived[stage]);
                    }
                } else {
                    arrive(q_arrived[stage]);
                }
            }
        }

        if constexpr (kPipe2CtaDebugStage == 0 || kPipe2CtaDebugStage == 1) {
            everyone::tma::cluster::arrive_aligned();
            everyone::tma::cluster::wait_aligned();
        }

        if (is_load && laneid() == 0) {
            if constexpr (kPipe2CtaDebugStage == 0 || kPipe2CtaDebugStage == 1) {
                const int preload = block_count < KVStages ? block_count : KVStages;
                for (int iter = 0; iter < preload; ++iter) {
                    const int slot = input_ring_slot<KVStages>(iter);
                    const int n_block = n_block_max - 1 - iter;
                    tma::cluster::load_async(
                        k_smem[slot],
                        g.k,
                        {batch_idx, kv_head_idx, n_block * kPipe2CtaClusterSize + cta_rank, 0},
                        kv_inputs_arrived[slot],
                        static_cast<uint16_t>(1u << cta_rank),
                        0
                    );
                    tma::cluster::load_async(
                        v_smem[slot],
                        g.v,
                        {batch_idx, kv_head_idx, n_block, cta_rank},
                        kv_inputs_arrived[slot],
                        static_cast<uint16_t>(1u << cta_rank),
                        0
                    );
                }
                for (int iter = preload; iter < block_count; ++iter) {
                    const int slot = input_ring_slot<KVStages>(iter);
                    const int reload_phase = cluster_finished_phase<KVStages>(iter);
                    if (cta_rank == 0) {
                        wait(inputs_finished[slot], reload_phase);
                        arrive_both_ctas(kv_reload_ready[slot]);
                    } else {
                        wait(kv_reload_ready[slot], reload_phase);
                    }
                    const int n_block = n_block_max - 1 - iter;
                    tma::cluster::load_async(
                        k_smem[slot],
                        g.k,
                        {batch_idx, kv_head_idx, n_block * kPipe2CtaClusterSize + cta_rank, 0},
                        kv_inputs_arrived[slot],
                        static_cast<uint16_t>(1u << cta_rank),
                        0
                    );
                    tma::cluster::load_async(
                        v_smem[slot],
                        g.v,
                        {batch_idx, kv_head_idx, n_block, cta_rank},
                        kv_inputs_arrived[slot],
                        static_cast<uint16_t>(1u << cta_rank),
                        0
                    );
                }
            }
        }

        if (is_mma && cta_rank == 0) {
            if constexpr (kPipe2CtaDebugStage == 0) {
                if (laneid() == 0) {
                    for (int stage = 0; stage < kPipeQStages; ++stage) {
                        if (local_valid[stage]) {
                            wait(q_arrived[stage], 0);
                        }
                    }
                }
                for (int iter = 0; iter < block_count; ++iter) {
                    const int slot = input_ring_slot<KVStages>(iter);
                    const int phase = cluster_input_phase<KVStages>(iter);
                    if (laneid() == 0) {
                        if (local_valid[0] || local_valid[1]) {
                            tma::expect_bytes(
                                kv_inputs_arrived[slot],
                                pipe2cta_kv_expected_bytes_total_v<D>
                            );
                            wait(kv_inputs_arrived[slot], phase);
                        }
                        const int stage_phase = iter & 1;
                        for (int stage = 0; stage < kPipeQStages; ++stage) {
                            if (!local_valid[stage]) {
                                continue;
                            }
                            if (iter >= 1) {
                                wait(score_outputs_finished[stage], (iter - 1) & 1);
                                tensor_after_thread_sync();
                            }
                            mm2_ABt(scores_tt[stage], q_smem[stage], k_smem[slot]);
                            tensor_after_thread_sync();
                            tensor_commit<2>(score_outputs_arrived[stage]);
                            if constexpr (kPipe2CtaTraceScores) {
                                arrive(inputs_finished[slot]);
                                continue;
                            }
                        }
                        if constexpr (kPipe2CtaTraceScores) {
                            continue;
                        }
                        for (int stage = 0; stage < kPipeQStages; ++stage) {
                            if (!local_valid[stage]) {
                                continue;
                            }
                            wait(prob_outputs_arrived[stage], stage_phase);
                            wait(out_outputs_finished[stage], stage_phase);
                            if (iter == 0) {
                                pipe2cta_issue_pv_initial(
                                    out_tt_tile[stage],
                                    probs_tt[stage],
                                    v_smem[slot],
                                    inputs_finished[slot]
                                );
                            } else {
                                pipe2cta_issue_pv_accum(
                                    out_tt_tile[stage],
                                    probs_tt[stage],
                                    v_smem[slot],
                                    inputs_finished[slot]
                                );
                            }
                            tensor_commit<2>(out_outputs_arrived[stage]);
                            arrive_both_ctas(prob_outputs_finished[stage]);
                        }
                    }
                }
            }
        }

        if (is_consumer) {
            out_rt out_accum, out_partial;
            vec_t lse_vec;
            const int cta_m_block = cta_m_blocks[q_stage_idx];
            const bool stage_valid = local_valid[q_stage_idx];
            if constexpr (kPipe2CtaDebugStage != 3 && kPipe2CtaDebugStage != 0) {
                wait(q_arrived[q_stage_idx], 0);
            }
            if constexpr (kPipe2CtaDebugStage == 3 || kPipe2CtaDebugStage == 2) {
                warp::zero(out_accum);
                if (stage_valid) {
                    warp::zero(lse_vec);
                    warp::store(
                        g.l_aux,
                        lse_vec,
                        {batch_idx, q_head_idx, 0, cta_m_block * kittens::WARPGROUP_WARPS + warp_in_group}
                    );
                }
                if (warp_in_group == 0 && laneid() == 0) {
                    arrive(final_out_ready[q_stage_idx]);
                }
            } else if constexpr (kPipe2CtaDebugStage == 1) {
                if constexpr (kPipe2CtaDebugSubstage == 0 || kPipe2CtaDebugSubstage == 2) {
                    for (int iter = 0; iter < block_count; ++iter) {
                        const int slot = input_ring_slot<KVStages>(iter);
                        const int phase = cluster_input_phase<KVStages>(iter);
                        if (warp_in_group == 0 && laneid() == 0 && q_stage_idx == 0) {
                            tma::expect_bytes(
                                kv_inputs_arrived[slot],
                                pipe2cta_kv_expected_bytes_total_v<D>
                            );
                            careful_wait(kv_inputs_arrived[slot], phase);
                        }
                        warpgroup::sync(14 + q_stage_idx);
                        if (warp_in_group == 0 && laneid() == 0) {
                            arrive(inputs_finished[slot]);
                        }
                    }
                }
                warp::zero(out_accum);
                if (stage_valid) {
                    warp::zero(lse_vec);
                    warp::store(
                        g.l_aux,
                        lse_vec,
                        {batch_idx, q_head_idx, 0, cta_m_block * kittens::WARPGROUP_WARPS + warp_in_group}
                    );
                }
                if constexpr (kPipe2CtaDebugSubstage == 0) {
                    warpgroup::store_async(out_tt_tile[q_stage_idx], out_accum);
                    tensor_store_wait();
                    warpgroup::sync(14 + q_stage_idx);
                }
                if (warp_in_group == 0 && laneid() == 0) {
                    arrive(final_out_ready[q_stage_idx]);
                }
            } else {
                if (q_stage_idx == 0) {
                    pipe2cta_softmax_role<D, CAUSAL, 0>(
                        g,
                        batch_idx,
                        q_head_idx,
                        cta_m_block,
                        stage_valid,
                        block_count,
                        n_block_max,
                        scores_tt[0],
                        probs_tt[0],
                        score_outputs_arrived[0],
                        score_outputs_finished[0],
                        stats_outputs_arrived[0],
                        stats_outputs_finished[0],
                        prob_outputs_arrived[0],
                        prob_outputs_finished[0],
                        final_out_ready[0],
                        scale_stats[0],
                        softmax_running[0]
                    );
                } else {
                    pipe2cta_softmax_role<D, CAUSAL, 1>(
                        g,
                        batch_idx,
                        q_head_idx,
                        cta_m_block,
                        stage_valid,
                        block_count,
                        n_block_max,
                        scores_tt[1],
                        probs_tt[1],
                        score_outputs_arrived[1],
                        score_outputs_finished[1],
                        stats_outputs_arrived[1],
                        stats_outputs_finished[1],
                        prob_outputs_arrived[1],
                        prob_outputs_finished[1],
                        final_out_ready[1],
                        scale_stats[1],
                        softmax_running[1]
                    );
                }
            }
        }

        if (is_correction) {
            pipe2cta_correction_rescale_role(
                block_count,
                out_tt_tile,
                local_valid,
                stats_outputs_arrived,
                stats_outputs_finished,
                out_outputs_finished,
                scale_stats
            );
            pipe2cta_finalize_role(
                g,
                batch_idx,
                q_head_idx,
                cta_m_blocks,
                local_valid,
                block_count,
                out_tt_tile,
                out_outputs_arrived,
                final_out_ready,
                o_epi_ready,
                o_epi_finished,
                scale_stats,
                o_smem
            );
        }

        if (is_epilogue) {
            pipe2cta_epilogue_role(
                g,
                batch_idx,
                q_head_idx,
                cta_m_blocks,
                local_valid,
                o_epi_ready,
                o_epi_finished,
                o_smem
            );
        }

        __syncthreads();
        everyone::tma::cluster::arrive_aligned();
        everyone::tma::cluster::wait_aligned();
        __syncthreads();
    }

    if constexpr (kPipe2CtaDebugStage != 3) {
        __syncthreads();
        everyone::tma::cluster::arrive_aligned();
        everyone::tma::cluster::wait_aligned();
        if (is_load && laneid() == 0) {
            tm_alloc.deprovision();
        }
    }
}

template <int D, int KVStages, bool CAUSAL>
__global__ __launch_bounds__(kPipeNumThreads, 1)
void kernel_pipe(const __grid_constant__ pipe_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al(reinterpret_cast<int*>(&__shm[0]));

    using q_tile = typename pipe_globals<D>::q_tile;
    using kv_tile = typename pipe_globals<D>::kv_tile;
    using v_tile = typename pipe_globals<D>::v_tile;
    using o_tile = typename pipe_globals<D>::o_tile;
    q_tile (&q_smem)[kPipeQStages] = al.allocate<q_tile, kPipeQStages>();
    kv_tile (&k_smem)[KVStages] = al.allocate<kv_tile, KVStages>();
    v_tile (&v_smem)[KVStages] = al.allocate<v_tile, KVStages>();
    o_tile (&o_smem)[kPipeQStages] = al.allocate<o_tile, kPipeQStages>();

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile>();
        g.k.template prefetch_tma<kv_tile>();
        g.v.template prefetch_tma<v_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int warp_in_group = warpgroup::warpid();
    const bool is_producer = warpgroup_id == 0;
    const bool is_consumer = warpgroup_id == 1 || warpgroup_id == 2;
    const bool is_epilogue = warpgroup_id == 3;
    const int q_stage_idx = is_consumer ? warpgroup_id - 1 : -1;

    __shared__ kittens::semaphore tmem_provisioned;
    __shared__ uint32_t tmem_addr;
    __shared__ kittens::semaphore q_arrived[kPipeQStages];
    __shared__ kittens::semaphore k_arrived[KVStages];
    __shared__ kittens::semaphore v_arrived[KVStages];
    __shared__ kittens::semaphore stage_free[KVStages];
    __shared__ kittens::semaphore score_ready[kPipeQStages];
    __shared__ kittens::semaphore out_ready[kPipeQStages];

    tensor_allocator<1, 1, false> tm_alloc{};
    using score_tt = half_tt_fl<kForwardTileN>;
    using prob_tt = half_tt_bf<kForwardTileN>;
    using out_tt = half_tt_fl<D>;
    score_tt scores_tt = score_tt{0};
    prob_tt probs_tt = prob_tt{0};
    out_tt out_tt_tile = out_tt{0};

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
    }
    if (is_producer && warp_in_group == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);
    if (is_consumer) {
        const int superlane = q_stage_idx;
        scores_tt = tm_alloc.template allocate<score_tt>(superlane, 0);
        probs_tt = tm_alloc.template allocate<prob_tt>(superlane, kForwardTileN);
        out_tt_tile = tm_alloc.template allocate<out_tt>(superlane, 2 * kForwardTileN);
    }

    const int num_m_blocks = g.padded_seq_len / kForwardTileM;
    const int kv_heads = g.q_heads / g.head_ratio;
    const int total_tiles = scheduler::forward_total_tiles(g.batch_size, g.q_heads, num_m_blocks);

    using score_rt = rt_fl<16, kForwardTileN>;
    using prob_rt = rt_bf<16, kForwardTileN>;
    using out_rt = rt_fl<16, D>;
    using out_bf = rt_bf<16, D>;
    using vec_t = typename score_rt::col_vec;

    for (int linear_tile = static_cast<int>(blockIdx.x); linear_tile < total_tiles; linear_tile += static_cast<int>(gridDim.x)) {
        const auto tile = scheduler::decode_forward_tile(
            linear_tile,
            g.batch_size,
            g.q_heads,
            kv_heads,
            g.head_ratio,
            num_m_blocks,
            CAUSAL
        );
        const int m_block = tile.m_block;
        const int q_head_idx = tile.q_head_idx;
        const int batch_idx = tile.batch_idx;
        const int kv_head_idx = tile.kv_head_idx;
        const int n_block_min = 0;
        const int n_block_max =
            scheduler::forward_cta_n_block_max(m_block, g.actual_seq_len, g.actual_seq_len, CAUSAL);
        const int block_count = n_block_max - n_block_min;
        int stage_phase[KVStages] = {0};

        if (threadIdx.x == 32) {
            for (int stage = 0; stage < kPipeQStages; ++stage) {
                init_semaphore(q_arrived[stage], 0, 1);
                init_semaphore(score_ready[stage], 0, 1);
                init_semaphore(out_ready[stage], 0, 1);
            }
            for (int stage = 0; stage < KVStages; ++stage) {
                init_semaphore(k_arrived[stage], 0, 1);
                init_semaphore(v_arrived[stage], 0, 1);
                init_semaphore(stage_free[stage], 0, kPipeQStages);
            }
        }
        __syncthreads();

        if (is_producer && warp_in_group == 0 && laneid() == 0) {
            for (int stage = 0; stage < kPipeQStages; ++stage) {
                tma::expect_bytes(q_arrived[stage], sizeof(q_tile));
                tma::load_async(q_smem[stage], g.q, {batch_idx, q_head_idx, m_block * kPipeQStages + stage, 0}, q_arrived[stage]);
            }
            if (block_count > 0) {
                const int first_block = n_block_max - 1;
                coord<kv_tile> kv_coord = {batch_idx, kv_head_idx, first_block, 0};
                coord<v_tile> v_coord = {batch_idx, kv_head_idx, first_block, 0};
                tma::expect_bytes(k_arrived[0], sizeof(kv_tile));
                tma::load_async(k_smem[0], g.k, kv_coord, k_arrived[0]);
                tma::expect_bytes(v_arrived[0], sizeof(v_tile));
                tma::load_async(v_smem[0], g.v, v_coord, v_arrived[0]);
            }
        }

        score_rt scores;
        prob_rt probs_bf;
        out_rt out_accum, out_partial, zero_out;
        out_bf out_store;
        vec_t max_vec, max_prev, norm_vec, alpha, lse_vec;
        if (is_consumer) {
            score_rt zero_scores;
            wait(q_arrived[q_stage_idx], 0);
            warp::zero(zero_scores);
            warp::zero(zero_out);
            warpgroup::store_async(scores_tt, zero_scores);
            warpgroup::store_async(out_tt_tile, zero_out);
            tensor_store_wait();
            warp::zero(out_accum);
            warp::zero(norm_vec);
            warp::neg_infty(max_vec);
        }

        for (int iter = 0; iter < block_count; ++iter) {
            const int curr_slot = iter % KVStages;
            const int n_block = n_block_max - 1 - iter;
            const int future_iter = iter + 1;
            if (is_producer && warp_in_group == 0 && laneid() == 0 && future_iter < block_count) {
                const int future_slot = future_iter % KVStages;
                const int future_block = n_block_max - 1 - future_iter;
                if (future_iter >= KVStages) {
                    wait(stage_free[future_slot], stage_phase[future_slot]);
                }
                coord<kv_tile> kv_coord = {batch_idx, kv_head_idx, future_block, 0};
                coord<v_tile> v_coord = {batch_idx, kv_head_idx, future_block, 0};
                tma::expect_bytes(k_arrived[future_slot], sizeof(kv_tile));
                tma::load_async(k_smem[future_slot], g.k, kv_coord, k_arrived[future_slot]);
                tma::expect_bytes(v_arrived[future_slot], sizeof(v_tile));
                tma::load_async(v_smem[future_slot], g.v, v_coord, v_arrived[future_slot]);
                if (future_iter >= KVStages) {
                    stage_phase[future_slot] ^= 1;
                }
            }

            if (is_consumer) {
                const int curr_phase = (iter / KVStages) & 1;
                wait(k_arrived[curr_slot], curr_phase);
                wait(v_arrived[curr_slot], curr_phase);
                warpgroup::mm_ABt(scores_tt, q_smem[q_stage_idx], k_smem[curr_slot], score_ready[q_stage_idx]);
                wait(score_ready[q_stage_idx], iter & 1);
                warpgroup::sync(14 + q_stage_idx);
                warpgroup::load_async(scores, scores_tt);
                tensor_load_wait();
                warp::mul(scores, scores, g.scale_log2e);

                const bool tail_tile = scheduler::forward_valid_cols(n_block, g.actual_seq_len) < kForwardTileN;
                const bool diag_tile = CAUSAL && (n_block == m_block);
                if (tail_tile || diag_tile) {
                    mask_score_stage<CAUSAL>(scores, q_stage_idx, warp_in_group, m_block, n_block, g.actual_seq_len);
                }

                warp::copy(max_prev, max_vec);
                warp::row_max(max_vec, scores, max_vec);
                warp::sub(alpha, max_prev, max_vec);
                warp::exp2(alpha, alpha);
                warp::sub_row(scores, scores, max_vec);
                warp::exp2(scores, scores);

                warp::mul(norm_vec, norm_vec, alpha);
                warp::row_sum(norm_vec, scores, norm_vec);

                warp::copy(probs_bf, scores);
                warpgroup::sync(14 + q_stage_idx);
                warpgroup::store_async(probs_tt, probs_bf);
                tensor_store_wait();

                warpgroup::sync(14 + q_stage_idx);
                warpgroup::store_async(out_tt_tile, zero_out);
                tensor_store_wait();
                warpgroup::mm_AB(out_tt_tile, probs_tt, v_smem[curr_slot], out_ready[q_stage_idx]);
                warp::zero(out_partial);
                wait(out_ready[q_stage_idx], iter & 1);
                warpgroup::sync(14 + q_stage_idx);
                warpgroup::load_async(out_partial, out_tt_tile);
                tensor_load_wait();
                warpgroup::sync(14 + q_stage_idx);
                warp::mul_row(out_accum, out_accum, alpha);
                warp::add(out_accum, out_accum, out_partial);
                if (warp_in_group == 0 && laneid() == 0 && iter + KVStages < block_count) {
                    arrive(stage_free[curr_slot]);
                }
            }

            __syncthreads();
        }

        if (is_consumer) {
            warp::div_row(out_accum, out_accum, norm_vec);
            warp::copy(out_store, out_accum);
            warp::store(
                g.o,
                out_store,
                {batch_idx, q_head_idx, m_block * (kPipeQStages * kittens::WARPGROUP_WARPS) + q_stage_idx * kittens::WARPGROUP_WARPS + warp_in_group, 0}
            );

            warp::mul(lse_vec, max_vec, 0.6931471805599453f);
            warp::log(norm_vec, norm_vec);
            warp::add(lse_vec, lse_vec, norm_vec);
            warp::mul(lse_vec, lse_vec, -1.0f / g.scale);
            warp::store(
                g.l_aux,
                lse_vec,
                {batch_idx, q_head_idx, 0, m_block * (kPipeQStages * kittens::WARPGROUP_WARPS) + q_stage_idx * kittens::WARPGROUP_WARPS + warp_in_group}
            );
        }
        __syncthreads();

        if (is_epilogue) {
            rt_bf<16, D> out_chunk;
            for (int stage = 0; stage < kPipeQStages; ++stage) {
                (void)stage;
                warp::zero(out_chunk);
            }
        }
        __syncthreads();
    }

    if (is_producer && warp_in_group == 0) {
        tm_alloc.deprovision();
    }
}

constexpr int kFastConsumerGroups = 1;
constexpr int kFastProducerGroups = 1;
constexpr int kFastStages = 2;
constexpr int kFastNumWarps =
    (kFastConsumerGroups + kFastProducerGroups) * kittens::WARPGROUP_WARPS;
constexpr int kFastNumThreads = kFastNumWarps * kittens::WARP_THREADS;
constexpr bool kFastEnableDispatch = true;

template <int D>
struct fast_globals {
    using q_tile = st_bf<128, D>;
    using kv_tile = st_bf<128, D>;
    using v_tile = st_bf<128, D>;
    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, kv_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using o_gl = gl<bf16, -1, -1, -1, D>;
    using l_tile = col_vec<st_fl<32, D>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;
    l_gl l_aux;
    float scale;
    float scale_log2e;
    int actual_seq_len;
    int padded_seq_len;
    int batch_size;
    int q_heads;
    int head_ratio;
};

template <bool CAUSAL>
__device__ inline void mask_score_tile(
    rt_fl<32, 128> &scores,
    int warp_in_group,
    int m_block,
    int n_block,
    int actual_seq_len
) {
    constexpr float neg_inf = kittens::base_types::constants<float>::neg_infty();
    const int row_base = m_block * kForwardTileM + warp_in_group * 32;
    warp::apply(scores, scores, [=](int row, int col, float value) {
        const int q_idx = row_base + row;
        const int k_idx = n_block * kForwardTileN + col;
        if (k_idx >= actual_seq_len) {
            return neg_inf;
        }
        if constexpr (CAUSAL) {
            if (k_idx > q_idx) {
                return neg_inf;
            }
        }
        return value;
    });
}

template <int D, bool CAUSAL>
__global__ __launch_bounds__(kFastNumThreads, 1)
void kernel_fast(const __grid_constant__ fast_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al(reinterpret_cast<int*>(&__shm[0]));

    using q_tile = typename fast_globals<D>::q_tile;
    using kv_tile = typename fast_globals<D>::kv_tile;
    using v_tile = typename fast_globals<D>::v_tile;
    q_tile &q_smem = al.allocate<q_tile>();
    kv_tile (&k_smem)[kFastStages] = al.allocate<kv_tile, kFastStages>();
    v_tile (&v_smem)[kFastStages] = al.allocate<v_tile, kFastStages>();

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile>();
        g.k.template prefetch_tma<kv_tile>();
        g.v.template prefetch_tma<v_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int warp_in_group = warpgroup::warpid();
    const bool is_consumer = warpgroup_id < kFastConsumerGroups;
    __shared__ kittens::semaphore q_arrived;
    __shared__ kittens::semaphore k_arrived[kFastStages];
    __shared__ kittens::semaphore v_arrived[kFastStages];
    __shared__ kittens::semaphore stage_free[kFastStages];
    __shared__ kittens::semaphore score_ready;
    __shared__ kittens::semaphore out_ready;
    __shared__ kittens::semaphore tmem_provisioned;
    __shared__ uint32_t tmem_addr;

    tensor_allocator<1, 1, false> tm_alloc{};
    using score_tt = full_tt_fl<kForwardTileN>;
    using prob_tt = full_tt_bf<kForwardTileN>;
    using out_tt = full_tt_fl<D>;
    auto scores_tt = score_tt{0};
    auto probs_tt = prob_tt{0};
    auto out_tt_tile = out_tt{0};

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
    }
    if (is_consumer && warp_in_group == 0) {
        tm_alloc.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }
    __syncthreads();
    wait(tmem_provisioned, 0);
    tm_alloc.set_addr(tmem_addr);
    scores_tt = tm_alloc.template allocate<score_tt>(0);
    probs_tt = tm_alloc.template allocate<prob_tt>(kForwardTileN);
    out_tt_tile = tm_alloc.template allocate<out_tt>(2 * kForwardTileN);

    const int num_m_blocks = g.padded_seq_len / kForwardTileM;
    const int kv_heads = g.q_heads / g.head_ratio;
    const int total_tiles = scheduler::forward_total_tiles(g.batch_size, g.q_heads, num_m_blocks);

    using score_rt = rt_fl<32, kForwardTileN>;
    using prob_rt = rt_bf<32, kForwardTileN>;
    using out_rt = rt_fl<32, D>;
    using out_bf = rt_bf<32, D>;
    using vec_t = typename score_rt::col_vec;

    for (int linear_tile = static_cast<int>(blockIdx.x); linear_tile < total_tiles; linear_tile += static_cast<int>(gridDim.x)) {
        const auto tile = scheduler::decode_forward_tile(
            linear_tile,
            g.batch_size,
            g.q_heads,
            kv_heads,
            g.head_ratio,
            num_m_blocks,
            CAUSAL
        );
        const int m_block = tile.m_block;
        const int q_head_idx = tile.q_head_idx;
        const int batch_idx = tile.batch_idx;
        const int kv_head_idx = tile.kv_head_idx;
        const int n_block_max =
            scheduler::forward_cta_n_block_max(m_block, g.actual_seq_len, g.actual_seq_len, CAUSAL);

        if (threadIdx.x == 32) {
            init_semaphore(q_arrived, 0, 1);
            init_semaphore(score_ready, 0, 1);
            init_semaphore(out_ready, 0, 1);
            for (int stage = 0; stage < kFastStages; ++stage) {
                init_semaphore(k_arrived[stage], 0, 1);
                init_semaphore(v_arrived[stage], 0, 1);
                init_semaphore(stage_free[stage], 0, 1);
            }
        }
        __syncthreads();

        if (!is_consumer) {
            if (warp_in_group == 0 && laneid() == 0) {
                tma::expect_bytes(q_arrived, sizeof(q_tile));
                tma::load_async(q_smem, g.q, {batch_idx, q_head_idx, m_block, 0}, q_arrived);

                const int preload = n_block_max < kFastStages ? n_block_max : kFastStages;
                for (int stage = 0; stage < preload; ++stage) {
                    coord<kv_tile> kv_coord = {batch_idx, kv_head_idx, stage, 0};
                    coord<v_tile> v_coord = {batch_idx, kv_head_idx, stage, 0};
                    tma::expect_bytes(k_arrived[stage], sizeof(kv_tile));
                    tma::load_async(k_smem[stage], g.k, kv_coord, k_arrived[stage]);
                    tma::expect_bytes(v_arrived[stage], sizeof(v_tile));
                    tma::load_async(v_smem[stage], g.v, v_coord, v_arrived[stage]);
                }

                int stage_phase[kFastStages] = {0, 0};
                for (int n_block = preload; n_block < n_block_max; ++n_block) {
                    const int stage = n_block % kFastStages;
                    wait(stage_free[stage], stage_phase[stage]);
                    coord<kv_tile> kv_coord = {batch_idx, kv_head_idx, n_block, 0};
                    coord<v_tile> v_coord = {batch_idx, kv_head_idx, n_block, 0};
                    tma::expect_bytes(k_arrived[stage], sizeof(kv_tile));
                    tma::load_async(k_smem[stage], g.k, kv_coord, k_arrived[stage]);
                    tma::expect_bytes(v_arrived[stage], sizeof(v_tile));
                    tma::load_async(v_smem[stage], g.v, v_coord, v_arrived[stage]);
                    stage_phase[stage] ^= 1;
                }
            }
        } else {
            score_rt scores;
            prob_rt probs_bf;
            out_rt out_accum, out_partial, zero_out;
            out_bf out_store;
            vec_t max_vec, max_prev, norm_vec, alpha, lse_vec;

            wait(q_arrived, 0);
            warp::zero(out_accum);
            warp::zero(zero_out);
            warp::zero(norm_vec);
            warp::neg_infty(max_vec);

            for (int n_block = 0; n_block < n_block_max; ++n_block) {
                const int stage = n_block % kFastStages;
                const int phase = (n_block / kFastStages) & 1;
                wait(k_arrived[stage], phase);
                wait(v_arrived[stage], phase);

                warpgroup::mm_ABt(scores_tt, q_smem, k_smem[stage], score_ready);
                wait(score_ready, n_block & 1);
                warpgroup::sync(14);
                warpgroup::load_async(scores, scores_tt);
                tensor_load_wait();
                warp::mul(scores, scores, g.scale_log2e);

                const bool tail_tile = scheduler::forward_valid_cols(n_block, g.actual_seq_len) < kForwardTileN;
                const bool diag_tile = CAUSAL && (n_block == m_block);
                if (tail_tile || diag_tile) {
                    mask_score_tile<CAUSAL>(scores, warp_in_group, m_block, n_block, g.actual_seq_len);
                }

                warp::copy(max_prev, max_vec);
                warp::row_max(max_vec, scores, max_vec);
                warp::sub(alpha, max_prev, max_vec);
                warp::exp2(alpha, alpha);
                warp::sub_row(scores, scores, max_vec);
                    warp::exp2(scores, scores);

                    warp::mul(norm_vec, norm_vec, alpha);
                    warp::row_sum(norm_vec, scores, norm_vec);

                    warp::copy(probs_bf, scores);
                    warpgroup::sync(14);
                    warpgroup::store_async(probs_tt, probs_bf);
                    tensor_store_wait();

                    warpgroup::sync(14);
                    warpgroup::store_async(out_tt_tile, zero_out);
                    tensor_store_wait();
                    warpgroup::mm_AB(out_tt_tile, probs_tt, v_smem[stage], out_ready);
                    wait(out_ready, n_block & 1);
                    warpgroup::sync(14);
                    warpgroup::load_async(out_partial, out_tt_tile);
                    tensor_load_wait();
                    warpgroup::sync(14);
                warp::mul_row(out_accum, out_accum, alpha);
                warp::add(out_accum, out_accum, out_partial);

                if (warp_in_group == 0 && laneid() == 0 && n_block + kFastStages < n_block_max) {
                    arrive(stage_free[stage]);
                }
            }

            warp::div_row(out_accum, out_accum, norm_vec);
            warp::copy(out_store, out_accum);
            warp::store(g.o, out_store, {batch_idx, q_head_idx, m_block * kittens::WARPGROUP_WARPS + warp_in_group, 0});

            warp::mul(lse_vec, max_vec, 0.6931471805599453f);
            warp::log(norm_vec, norm_vec);
            warp::add(lse_vec, lse_vec, norm_vec);
            warp::mul(lse_vec, lse_vec, -1.0f / g.scale);
            warp::store(g.l_aux, lse_vec, {batch_idx, q_head_idx, 0, m_block * kittens::WARPGROUP_WARPS + warp_in_group});
        }

        __syncthreads();
    }

    if (is_consumer && warp_in_group == 0) {
        tm_alloc.deprovision();
    }
}

template <int D>
inline void launch_reference(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &o,
    at::Tensor &l_aux,
    bool causal,
    float scale,
    int actual_seq_len
) {
    using G = ref_globals<D>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::o_gl>(o),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        actual_seq_len,
        static_cast<int>(q.size(1) / k.size(1)),
    };
    dim3 grid(q.size(2) / kRefTileM, q.size(1), q.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        kernel_ref<D, true><<<grid, kWarpThreads, 0, stream>>>(g);
    } else {
        kernel_ref<D, false><<<grid, kWarpThreads, 0, stream>>>(g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <int D>
inline int fast_dynamic_smem_bytes() {
    using G = fast_globals<D>;
    constexpr int bytes =
        sizeof(typename G::q_tile) +
        sizeof(typename G::kv_tile) * kFastStages * 2 +
        1024;
    static_assert(bytes <= MAX_SHARED_MEMORY - 1024, "fast forward shared memory exceeds device limit");
    return bytes;
}

inline int select_q_stages(int actual_seq_len) {
    return actual_seq_len <= kForwardTileN ? 1 : 2;
}

inline int active_ctas_for_cluster(int total_tiles, int cluster_size) {
    const int sm_count = num_sms();
    if (cluster_size <= 1) {
        const int active_ctas = total_tiles < sm_count ? total_tiles : sm_count;
        return active_ctas > 0 ? active_ctas : 1;
    }
    const int max_clusters = sm_count / cluster_size > 0 ? sm_count / cluster_size : 1;
    const int requested_clusters = ceil_div(total_tiles, cluster_size);
    const int active_clusters = requested_clusters < max_clusters ? requested_clusters : max_clusters;
    return active_clusters * cluster_size;
}

template <typename Kernel, typename G>
inline void launch_kernel(
    Kernel kernel,
    const G &g,
    dim3 grid,
    dim3 block,
    int dynamic_smem,
    cudaStream_t stream,
    int cluster_size
) {
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem
    ));
    if (cluster_size > 1) {
        kittens::LaunchConfig<true, false> launch_config(
            grid,
            block,
            dynamic_smem,
            stream,
            dim3(cluster_size, 1, 1)
        );
        CHECK_CUDA_ERROR(cudaLaunchKernelEx(launch_config, kernel, g));
    } else {
        kittens::LaunchConfig<false, false> launch_config(
            grid,
            block,
            dynamic_smem,
            stream
        );
        CHECK_CUDA_ERROR(cudaLaunchKernelEx(launch_config, kernel, g));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <int D>
inline int resolve_cluster_size(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len
) {
    const bool flagship_dense_mha =
        D == 128 &&
        !causal &&
        q.size(1) == k.size(1) &&
        q.size(2) == k.size(2);
    const bool eligible_two_cta =
        flagship_dense_mha &&
        actual_seq_len > 256 &&
        q.size(1) == k.size(1);
    const int force_2cta = resolve_force_2cta();
    if (force_2cta == 0) {
        return 1;
    }
    if (force_2cta == 1) {
        return flagship_dense_mha ? 2 : 1;
    }
    return eligible_two_cta ? 2 : 1;
}

template <int D>
inline bool use_flagship_pipe_2cta(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len
) {
    if constexpr (D != 128) {
        return false;
    }
    const bool flagship_dense_mha =
        !causal &&
        q.size(1) == k.size(1) &&
        q.size(2) == k.size(2);
    const int force_2cta = resolve_force_2cta();
    if (force_2cta == 1) {
        return flagship_dense_mha;
    }
    if (!flagship_dense_mha) {
        return false;
    }
    if (force_2cta == 0 || actual_seq_len <= 256) {
        return false;
    }
    return resolve_cluster_size<D>(q, k, causal, actual_seq_len) == 2;
}

template <int D>
inline void launch_fast(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &o,
    at::Tensor &l_aux,
    bool causal,
    float scale,
    int actual_seq_len,
    int cluster_size
) {
    using G = fast_globals<D>;
    const int dynamic_smem = fast_dynamic_smem_bytes<D>();
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::o_gl>(o),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        actual_seq_len,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(1) / k.size(1)),
    };
    const int num_m_blocks = static_cast<int>(q.size(2) / kForwardTileM);
    const int total_tiles = scheduler::forward_total_tiles(
        static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)),
        num_m_blocks
    );
    const int active_ctas = active_ctas_for_cluster(total_tiles, cluster_size);
    dim3 grid(active_ctas > 0 ? active_ctas : 1);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        launch_kernel(
            kernel_fast<D, true>,
            g,
            grid,
            dim3(kFastNumThreads, 1, 1),
            dynamic_smem,
            stream,
            cluster_size
        );
    } else {
        launch_kernel(
            kernel_fast<D, false>,
            g,
            grid,
            dim3(kFastNumThreads, 1, 1),
            dynamic_smem,
            stream,
            cluster_size
        );
    }
}

template <int D>
inline void launch_pipe_2cta(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &o,
    at::Tensor &l_aux,
    bool causal,
    float scale,
    int actual_seq_len
) {
    static_assert(D == 128, "launch_pipe_2cta is only supported for head_dim 128");
    TORCH_CHECK(!causal, "2CTA forward is only enabled for dense noncausal head_dim=128");
    using G = pipe2cta_globals<D>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::o_gl>(o),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        actual_seq_len,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(1) / k.size(1)),
    };
    const int num_m_blocks = static_cast<int>(q.size(2) / kForwardTileM);
    const int cluster_tiles = scheduler::forward_cluster_task_total_tiles(
        static_cast<int>(q.size(0)),
        static_cast<int>(k.size(1)),
        static_cast<int>(q.size(1) / k.size(1)),
        num_m_blocks,
        kPipeQStages,
        kPipe2CtaClusterSize
    );
    const int total_ctas = cluster_tiles * kPipe2CtaClusterSize;
    const int active_ctas = active_ctas_for_cluster(total_ctas, kPipe2CtaClusterSize);
    dim3 grid(active_ctas > 0 ? active_ctas : kPipe2CtaClusterSize);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    g.cluster_tile_idx = 0;
    constexpr int kv_stages = pipe2cta_kv_stages_v<D>;
    if constexpr (kv_stages == 4) {
        launch_kernel(
            kernel_pipe_2cta_unified<D, 4, false>,
            g,
            grid,
            dim3(kPipe2CtaNumThreads, 1, 1),
            pipe2cta_dynamic_smem_bytes<D, 4>(),
            stream,
            kPipe2CtaClusterSize
        );
    } else if constexpr (kv_stages == 3) {
        launch_kernel(
            kernel_pipe_2cta_unified<D, 3, false>,
            g,
            grid,
            dim3(kPipe2CtaNumThreads, 1, 1),
            pipe2cta_dynamic_smem_bytes<D, 3>(),
            stream,
            kPipe2CtaClusterSize
        );
    } else {
        launch_kernel(
            kernel_pipe_2cta_unified<D, 2, false>,
            g,
            grid,
            dim3(kPipe2CtaNumThreads, 1, 1),
            pipe2cta_dynamic_smem_bytes<D, 2>(),
            stream,
            kPipe2CtaClusterSize
        );
    }
}

template <int D, int KVStages>
inline int pipe_dynamic_smem_bytes() {
    using G = pipe_globals<D>;
    constexpr int bytes =
        sizeof(typename G::q_tile) * kPipeQStages +
        sizeof(typename G::kv_tile) * KVStages * 2 +
        sizeof(typename G::o_tile) * kPipeQStages +
        1024;
    static_assert(bytes <= MAX_SHARED_MEMORY - 1024, "pipe forward shared memory exceeds device limit");
    return bytes;
}

template <int D>
inline void launch_pipe(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &o,
    at::Tensor &l_aux,
    bool causal,
    float scale,
    int actual_seq_len,
    int cluster_size
) {
    using G = pipe_globals<D>;
    constexpr int kv_stages = pipe_kv_stages_v<D>;
    const int dynamic_smem = pipe_dynamic_smem_bytes<D, kv_stages>();
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::o_gl>(o),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        actual_seq_len,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(1) / k.size(1)),
    };
    const int num_m_blocks = static_cast<int>(q.size(2) / kForwardTileM);
    const int total_tiles = scheduler::forward_total_tiles(
        static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)),
        num_m_blocks
    );
    const int active_ctas = active_ctas_for_cluster(total_tiles, cluster_size);
    dim3 grid(active_ctas > 0 ? active_ctas : 1);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        launch_kernel(
            kernel_pipe<D, kv_stages, true>,
            g,
            grid,
            dim3(kPipeNumThreads, 1, 1),
            dynamic_smem,
            stream,
            cluster_size
        );
    } else {
        launch_kernel(
            kernel_pipe<D, kv_stages, false>,
            g,
            grid,
            dim3(kPipeNumThreads, 1, 1),
            dynamic_smem,
            stream,
            cluster_size
        );
    }
}

template <int D>
inline void launch(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &o,
    at::Tensor &l_aux,
    bool causal,
    float scale,
    int actual_seq_len
) {
    if (resolve_forward_mode() == forward_mode::ref) {
        launch_reference<D>(q, k, v, o, l_aux, causal, scale, actual_seq_len);
        return;
    }

    if constexpr (D == 128) {
        if (use_flagship_pipe_2cta<D>(q, k, causal, actual_seq_len)) {
            launch_pipe_2cta<D>(q, k, v, o, l_aux, causal, scale, actual_seq_len);
            return;
        }
    }

    const int q_stages = select_q_stages(actual_seq_len);
    const int cluster_size = resolve_cluster_size<D>(q, k, causal, actual_seq_len);
    if (q_stages == 1) {
        launch_fast<D>(q, k, v, o, l_aux, causal, scale, actual_seq_len, 1);
    } else {
        launch_pipe<D>(q, k, v, o, l_aux, causal, scale, actual_seq_len, cluster_size);
    }
}

}  // namespace tkfa4::fwd

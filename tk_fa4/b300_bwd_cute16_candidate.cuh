#pragma once

#include <cstdio>
#include <cstdlib>

#include "b300_bwd_cute.cuh"
#include "b300_bwd_cute16.cuh"
#include "b300_bwd_cute16_candidate_cute.cuh"
#include "b300_bwd_cute16_kernel_candidate.cuh"
#include "b300_bwd_fa4.cuh"
#include "b300_bwd_fa4_postprocess.cuh"
#include "b300_bwd_fa4_preprocess.cuh"
#include "b300_bwd_hot.cuh"

namespace tkfa4::bwd_cute16_candidate {

namespace detail {

template <typename T>
concept hot_candidate_has_compute_warp_base = requires { T::ComputeWarpBase; };

template <typename T>
concept hot_candidate_has_load_warp_id = requires { T::LoadWarpId; };

struct split_hybrid_resources {
    int device = -1;
    uintptr_t owner_stream = 0;
    int batch = -1;
    int heads = -1;
    int seq_len = -1;
    int dense_split_capacity = 0;
    cudaStream_t dq_stream = nullptr;
    cudaStream_t dkdv_stream = nullptr;
    cudaEvent_t call_entry_ready = nullptr;
    cudaEvent_t preprocess_done = nullptr;
    cudaEvent_t dq_done = nullptr;
    cudaEvent_t dkdv_done = nullptr;
    cudaEvent_t dense_main_done = nullptr;
    cudaEvent_t frontier_dv_done = nullptr;
    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_end = nullptr;
    cudaEvent_t preprocess_start = nullptr;
    cudaEvent_t preprocess_end = nullptr;
    cudaEvent_t dkdv_start = nullptr;
    cudaEvent_t dkdv_end = nullptr;
    cudaEvent_t dq_start = nullptr;
    cudaEvent_t dq_zero_end = nullptr;
    cudaEvent_t dq_ready = nullptr;
    cudaEvent_t dq_end = nullptr;
    at::Tensor dpsum;
    at::Tensor lse_log2;
    at::Tensor dqacc;
    at::Tensor ds_scratch;
    at::Tensor frontier_dk;
    at::Tensor frontier_dv;
    at::Tensor dense_split_dk;
    at::Tensor dense_split_dv;
    at::Tensor cached_base_dk;
    at::Tensor cached_base_dv;
};

inline bool matches_fp32_base(
    const at::Tensor &tensor,
    const at::Tensor &q,
    int feature_dim
) {
    return tensor.defined() &&
        tensor.device() == q.device() &&
        tensor.scalar_type() == at::kFloat &&
        tensor.is_contiguous() &&
        tensor.dim() == 4 &&
        tensor.size(0) == q.size(0) &&
        tensor.size(1) == q.size(1) &&
        tensor.size(2) == q.size(2) &&
        tensor.size(3) == feature_dim;
}

inline bool split_timing_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("TK_FA4_SPLIT_TIMING");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

inline bool hot_clustered_dq_probe_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("TK_FA4_USE_HOT_CLUSTERED_DQ");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

inline split_hybrid_resources &get_split_hybrid_resources(
    int device,
    cudaStream_t current_stream,
    const at::Tensor &q,
    const at::Tensor &lse,
    int scratch_rows,
    int dqk_dim,
    bool need_ds_scratch = false,
    int frontier_dvo_dim = 0,
    int dense_split_count = 1,
    bool cache_dkdv_fp32_base = false
) {
    TORCH_CHECK(
        dense_split_count == 1 || dense_split_count == 2 || dense_split_count == 3 ||
            dense_split_count == 4 || dense_split_count == 8,
        "dense split count must be 1, 2, 3, 4, or 8"
    );
    TORCH_CHECK(
        dense_split_count == 1 || frontier_dvo_dim > 0,
        "dense split scratch requires a positive dV dimension"
    );
    TORCH_CHECK(
        !cache_dkdv_fp32_base || frontier_dvo_dim > 0,
        "cached FP32 dK/dV bases require a positive dV dimension"
    );
    static thread_local split_hybrid_resources cache;
    const bool device_changed = cache.device != device;
    if (device_changed) {
        if (cache.device >= 0 && cache.preprocess_done != nullptr) {
            const c10::cuda::CUDAGuard old_device_guard(
                static_cast<c10::DeviceIndex>(cache.device)
            );
            cudaEventDestroy(cache.call_entry_ready);
            cudaEventDestroy(cache.preprocess_done);
            cudaEventDestroy(cache.dq_done);
            cudaEventDestroy(cache.dkdv_done);
            cudaEventDestroy(cache.dense_main_done);
            cudaEventDestroy(cache.frontier_dv_done);
            cudaEventDestroy(cache.total_start);
            cudaEventDestroy(cache.total_end);
            cudaEventDestroy(cache.preprocess_start);
            cudaEventDestroy(cache.preprocess_end);
            cudaEventDestroy(cache.dkdv_start);
            cudaEventDestroy(cache.dkdv_end);
            cudaEventDestroy(cache.dq_start);
            cudaEventDestroy(cache.dq_zero_end);
            cudaEventDestroy(cache.dq_ready);
            cudaEventDestroy(cache.dq_end);
        }
        cache.device = device;
        cache.dq_stream = at::cuda::getStreamFromPool(false, device).stream();
        cache.dkdv_stream = at::cuda::getStreamFromPool(false, device).stream();
        CUDACHECK(cudaEventCreateWithFlags(&cache.call_entry_ready, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.preprocess_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.dq_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.dkdv_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.dense_main_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.frontier_dv_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreate(&cache.total_start));
        CUDACHECK(cudaEventCreate(&cache.total_end));
        CUDACHECK(cudaEventCreate(&cache.preprocess_start));
        CUDACHECK(cudaEventCreate(&cache.preprocess_end));
        CUDACHECK(cudaEventCreate(&cache.dkdv_start));
        CUDACHECK(cudaEventCreate(&cache.dkdv_end));
        CUDACHECK(cudaEventCreate(&cache.dq_start));
        CUDACHECK(cudaEventCreate(&cache.dq_zero_end));
        CUDACHECK(cudaEventCreate(&cache.dq_ready));
        CUDACHECK(cudaEventCreate(&cache.dq_end));
    }
    const uintptr_t stream_key = reinterpret_cast<uintptr_t>(current_stream);
    const bool needs_dqacc = scratch_rows > 0;
    const int dense_split_partials = dense_split_count - 1;
    const bool needs_scratch_refresh =
        device_changed ||
        cache.owner_stream != stream_key ||
        cache.batch != q.size(0) ||
        cache.heads != q.size(2) ||
        cache.seq_len != q.size(1) ||
        !cache.dpsum.defined() ||
        !cache.lse_log2.defined() ||
        (need_ds_scratch && (
            !cache.ds_scratch.defined() ||
            cache.ds_scratch.size(1) != q.size(1) ||
            cache.ds_scratch.size(3) != q.size(1)
        )) ||
        (frontier_dvo_dim > 0 && (
            !cache.frontier_dk.defined() ||
            !cache.frontier_dv.defined() ||
            cache.frontier_dk.size(3) != dqk_dim ||
            cache.frontier_dv.size(3) != frontier_dvo_dim
        )) ||
        (dense_split_count > 1 && (
            !cache.dense_split_dk.defined() ||
            !cache.dense_split_dv.defined() ||
            cache.dense_split_capacity < dense_split_partials ||
            cache.dense_split_dk.size(3) != dqk_dim ||
            cache.dense_split_dv.size(3) != frontier_dvo_dim
        )) ||
        (needs_dqacc && (
            !cache.dqacc.defined() ||
            cache.dqacc.size(2) != scratch_rows ||
            cache.dqacc.size(3) != dqk_dim
        )) ||
        (cache_dkdv_fp32_base && (
            !matches_fp32_base(cache.cached_base_dk, q, dqk_dim) ||
            !matches_fp32_base(cache.cached_base_dv, q, frontier_dvo_dim)
        ));
    if (needs_scratch_refresh) {
        cache.owner_stream = stream_key;
        cache.batch = static_cast<int>(q.size(0));
        cache.heads = static_cast<int>(q.size(2));
        cache.seq_len = static_cast<int>(q.size(1));
        cache.dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
        cache.lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
        if (need_ds_scratch) {
            cache.ds_scratch = at::empty(
                {q.size(0), q.size(1), q.size(2), q.size(1)},
                q.options()
            );
        } else {
            cache.ds_scratch = at::Tensor();
        }
        if (needs_dqacc) {
            cache.dqacc = at::empty({q.size(0), q.size(2), scratch_rows, dqk_dim}, lse.options());
        } else {
            cache.dqacc = at::Tensor();
        }
        if (frontier_dvo_dim > 0) {
            cache.frontier_dk = at::empty(
                {q.size(0), q.size(1), q.size(2), dqk_dim},
                lse.options()
            );
            cache.frontier_dv = at::empty(
                {q.size(0), q.size(1), q.size(2), frontier_dvo_dim},
                lse.options()
            );
        } else {
            cache.frontier_dk = at::Tensor();
            cache.frontier_dv = at::Tensor();
        }
        if (dense_split_count > 1) {
            cache.dense_split_capacity = dense_split_partials;
            cache.dense_split_dk = at::empty(
                {q.size(0) * dense_split_partials, q.size(1), q.size(2), dqk_dim},
                lse.options()
            );
            cache.dense_split_dv = at::empty(
                {q.size(0) * dense_split_partials, q.size(1), q.size(2), frontier_dvo_dim},
                lse.options()
            );
        } else {
            cache.dense_split_capacity = 0;
            cache.dense_split_dk = at::Tensor();
            cache.dense_split_dv = at::Tensor();
        }
        if (cache_dkdv_fp32_base) {
            cache.cached_base_dk = at::empty(
                {q.size(0), q.size(1), q.size(2), dqk_dim},
                lse.options().dtype(at::kFloat)
            );
            cache.cached_base_dv = at::empty(
                {q.size(0), q.size(1), q.size(2), frontier_dvo_dim},
                lse.options().dtype(at::kFloat)
            );
        } else {
            cache.cached_base_dk = at::Tensor();
            cache.cached_base_dv = at::Tensor();
        }
    }
    return cache;
}

}  // namespace detail

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
using config = bwd_cute16_kernel_candidate::config<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;

template <int Dvo>
struct preprocess_config {
    static_assert(Dvo == kB300VDim, "Exact B300 candidate preprocess only supports Dvo=128");
    static constexpr int DvoDim = Dvo;
};

template <typename C>
struct preprocess_globals {
    using stats_tile = col_vec<st_fl<kRefTileM, C::DvoDim>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    const bf16 *o_ptr;
    const bf16 *dout_ptr;
    const float *lse_ptr;
    stats_gl dpsum;
    stats_gl lse_log2;
    int seq_len;
    int heads;
};

template <typename C>
__global__ __launch_bounds__(4 * kWarpThreads, 8)
void preprocess_kernel(const __grid_constant__ preprocess_globals<C> g) {
    constexpr int RowThreads = 8;
    constexpr int RowsPerWarp = kWarpThreads / RowThreads;
    const int q_tile_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = laneid();
    const int row_in_warp = lane / RowThreads;
    const int lane_in_row = lane % RowThreads;
    const int row = warp * RowsPerWarp + row_in_warp;
    if (row < kRefTileM) {
        const int seq_idx = q_tile_idx * kRefTileM + row;
        float dpsum = 0.0f;
        const size_t base = (((size_t)batch_idx * g.seq_len + seq_idx) * g.heads + head_idx) * C::DvoDim;
        #pragma unroll
        for (int d = lane_in_row; d < C::DvoDim; d += RowThreads) {
            dpsum += __bfloat162float(g.o_ptr[base + d]) * __bfloat162float(g.dout_ptr[base + d]);
        }
        #pragma unroll
        for (int offset = RowThreads / 2; offset > 0; offset >>= 1) {
            dpsum += __shfl_down_sync(0xffffffff, dpsum, offset, RowThreads);
        }
        if (lane_in_row == 0) {
            const size_t lse_offset = ((size_t)batch_idx * g.seq_len + seq_idx) * g.heads + head_idx;
            g.dpsum[{batch_idx, head_idx, 0, seq_idx}] = dpsum;
            g.lse_log2[{batch_idx, head_idx, 0, seq_idx}] = g.lse_ptr[lse_offset] * kLog2E;
        }
    }
}

template <typename C>
inline void launch_preprocess(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &lse,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    using G = preprocess_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, out.size(0), out.size(2), 1, out.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, out.size(0), out.size(2), 1, out.size(1)),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / kRefTileM, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_kernel<C><<<grid, 4 * kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

namespace detail {

template <typename C>
struct fa4_bshd_direct_dq_globals {
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
    int actual_seq_len;
};

template <typename C>
struct hot_bshd_dkdv_globals {
    using q_tile = st_bf<C::TileRows, C::Dqk>;
    using k_tile = st_bf<C::TileRows, C::Dqk>;
    using v_tile = st_bf<C::TileRows, C::Dvo>;
    using do_tile = st_bf<C::TileRows, C::Dvo>;
    using dk_full_tile = st_fl<C::TileRows, 64>;
    using dv_full_tile = st_fl<C::TileRows, C::Dvo>;
    using dv_tile = st_fl<kRefTileM, C::Dvo>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<k_tile, dim::DEPTH>>;
    using v_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tile, dim::DEPTH>>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dk_full_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dk_full_tile, dim::DEPTH>>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using dv_full_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dv_full_tile, dim::DEPTH>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk_gl dk;
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
};

template <typename C, bool CAUSAL>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void fa4_bshd_main_kernel(const __grid_constant__ bwd_fa4::main_globals<C> g) {
    constexpr int q_tiles_buffered = 4;
    using qk_bf_tile = st_bf<kRefTileM, C::Dqk>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo>;
    using qk_fl_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) qk_fl_tile dq_smem[C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = cluster_ctarank();
    const int kv_block_idx = static_cast<int>(clusterIdx().x) * C::ClusterSize + cluster_rank;
    const int num_k_blocks = g.seq_len / (kRefTileN * C::WarpTiles);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }
    const int cluster_group_idx = kv_block_idx / C::ClusterSize;

    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int num_q_blocks = g.seq_len / (kRefTileM * q_tiles_buffered);
    const bool dense_unmasked = !CAUSAL && g.actual_seq_len == g.seq_len;

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, C::Dqk> dk_accum, dq_total, dq_reg, dq_existing;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;

        if (warp < q_tiles_buffered) {
            rt_bf<kRefTileM, C::Dqk> q_stage_reg;
            rt_bf<kRefTileM, C::Dvo> do_stage_reg;
            stats_vec lse_stage_vec, dpsum_stage_vec;
            warp::load<dim::DEPTH>(q_stage_reg, g.q, {batch_idx, q_tile_base + warp, head_idx, 0});
            warp::store(q_smem[warp], q_stage_reg);
            warp::load<dim::DEPTH>(do_stage_reg, g.dout, {batch_idx, q_tile_base + warp, head_idx, 0});
            warp::store(do_smem[warp], do_stage_reg);
            warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_stage_vec);
            warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_stage_vec);
        }
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, kRefTileN> p, dp, ds;
            rt_fl<kRefTileM, C::Dqk> dq_partial;
            stats_vec lse_log2_vec, dpsum_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            bwd_fa4::detail::backward_tile_step<C, CAUSAL>(
                p,
                dp,
                ds,
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
                g.actual_seq_len,
                dense_unmasked
            );

            bwd_fa4::detail::accumulate_dq_partial<C>(
                g,
                dq_partial,
                dq_total,
                dq_reg,
                dq_existing,
                dq_smem,
                batch_idx,
                head_idx,
                q_tile_idx,
                cluster_rank,
                cluster_group_idx,
                warp
            );
        }
        __syncthreads();
    }

    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C, bool CAUSAL>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void fa4_bshd_direct_dq_main_kernel(const __grid_constant__ fa4_bshd_direct_dq_globals<C> g) {
    constexpr int q_tiles_buffered = 4;
    using qk_bf_tile = typename fa4_bshd_direct_dq_globals<C>::q_tma_tile;
    using v_bf_tile = typename fa4_bshd_direct_dq_globals<C>::do_tma_tile;
    using qk_fl_tile = st_fl<kRefTileM, C::Dqk>;
    using dq_chunk_tile = typename fa4_bshd_direct_dq_globals<C>::dq_chunk_tile;
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
    const int cluster_rank = cluster_ctarank();
    const int kv_block_idx = static_cast<int>(clusterIdx().x) * C::ClusterSize + cluster_rank;
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

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<qk_bf_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<v_bf_tile, dim::DEPTH>();
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
    }
    __syncthreads();

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int phase = q_block_idx & 1;

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
            stats_vec lse_stage_vec, dpsum_stage_vec;
            warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_stage_vec);
            warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_stage_vec);
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, kRefTileN> p, dp, ds;
            rt_fl<kRefTileM, C::Dqk> dq_partial;
            rt_fl<kRefTileM, 64> dq0, dq1, dq2;
            stats_vec lse_log2_vec, dpsum_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            bwd_fa4::detail::backward_tile_step<C, CAUSAL>(
                p,
                dp,
                ds,
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
                g.actual_seq_len,
                dense_unmasked
            );

            if (q_block_idx != 0 || subtile != 0) {
                warp::tma::store_async_read_wait();
            }
            ::tkfa4::bwd_cute16_kernel_candidate::detail::extract_chunk<0>(dq_partial, dq0);
            ::tkfa4::bwd_cute16_kernel_candidate::detail::extract_chunk<1>(dq_partial, dq1);
            ::tkfa4::bwd_cute16_kernel_candidate::detail::extract_chunk<2>(dq_partial, dq2);
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
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C, bool CAUSAL>
__global__ __launch_bounds__(C::DkdvBlockThreads, C::MinBlocksPerSm)
void hot_bshd_dkdv_main_kernel(const __grid_constant__ hot_bshd_dkdv_globals<C> g) {
    constexpr bool kUseExactRegFrontier = false;
    constexpr bool kUseTmemDense = true;
    constexpr bool kUseTmemDenseDk = true;
    constexpr bool kUseTmemDenseDv = true;
    constexpr bool kUseDirectDenseReconstruct = false;
    constexpr bool kUseCuteDenseHelper = false;
    using q_tile = typename hot_bshd_dkdv_globals<C>::q_tile;
    using k_tile = typename hot_bshd_dkdv_globals<C>::k_tile;
    using v_tile = typename hot_bshd_dkdv_globals<C>::v_tile;
    using do_tile = typename hot_bshd_dkdv_globals<C>::do_tile;
    using stats_smem_tile = typename hot_bshd_dkdv_globals<C>::stats_tile;
    using ds_warp_tile = st_bf<kRefTileM, C::TileRows>;
    using attn_tt = half_tt_fl<C::TileRows>;
    using dk_tt = half_tt_fl<64>;
    using dv_tt = half_tt_fl<C::Dvo>;

    struct main_shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        ds_warp_tile ds_warp_smem[C::ConsumerWarpgroups][WARPGROUP_WARPS];
        stats_smem_tile lse_log2_smem[C::QSubtiles];
        stats_smem_tile dpsum_smem[C::QSubtiles];
    };
    struct epilogue_shared_storage {
        typename hot_bshd_dkdv_globals<C>::dk_full_tile dk0_smem[C::ConsumerWarpgroups];
        typename hot_bshd_dkdv_globals<C>::dk_full_tile dk1_smem[C::ConsumerWarpgroups];
        typename hot_bshd_dkdv_globals<C>::dk_full_tile dk2_smem[C::ConsumerWarpgroups];
        typename hot_bshd_dkdv_globals<C>::dv_full_tile dv_smem[C::ConsumerWarpgroups];
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
    const int first_dense_q_block =
        (CAUSAL && kUseExactRegFrontier) ? (q_start_block + C::ConsumerWarpgroups) : q_start_block;
    rt_fl<kRefTileM, 64> dk0_reg_accum, dk1_reg_accum, dk2_reg_accum;
    rt_fl<kRefTileM, C::Dvo> dv_reg_accum;
    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dk_tt dk0_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk1_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk2_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dv_tt dv_accum_tt[C::ConsumerWarpgroups] = {dv_tt{0}, dv_tt{0}};

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
            coord<k_tile> tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            coord<v_tile> v_tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[w], g.v, v_tile_idx, kv_b);
        }
    }
    __syncthreads();

    if (is_compute) {
        warp::zero(dk0_reg_accum);
        warp::zero(dk1_reg_accum);
        warp::zero(dk2_reg_accum);
        warp::zero(dv_reg_accum);
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

    if (is_compute) {
        if constexpr (kUseExactRegFrontier && CAUSAL) {
            wait(kv_b, 0);
            rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
            rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
            using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
            stats_vec lse_log2_vec, dpsum_vec;
            const int consumer_warp = warpgroup::warpid();
            warp::load(k_reg, k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({consumer_warp, 0}));
            warp::load(v_reg, v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({consumer_warp, 0}));

            for (int tile_offset = consumer_idx; tile_offset < C::ConsumerWarpgroups; ++tile_offset) {
                const int q_block_tile_idx = q_start_block + tile_offset;
                if (q_block_tile_idx >= q_blocks) {
                    break;
                }

                int q_subtile_begin = 0;
                if (tile_offset == consumer_idx) {
                    const int q_subtile = consumer_warp;
                    const int q_tile_idx = q_block_tile_idx * C::QSubtiles + q_subtile;
                    warp::load<dim::DEPTH>(q_reg, g.q, {batch_idx, q_tile_idx, head_idx, 0});
                    warp::load<dim::DEPTH>(do_reg, g.dout, {batch_idx, q_tile_idx, head_idx, 0});
                    warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
                    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
                    bwd_cute16_kernel_candidate::detail::repair_dkdv_step_chunked<C, true, false>(
                        dk0_reg_accum,
                        dk1_reg_accum,
                        dk2_reg_accum,
                        dv_reg_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        dpsum_vec,
                        g.scale,
                        g.scale_log2e,
                        q_tile_idx,
                        (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + consumer_warp,
                        g.seq_len
                    );
                    q_subtile_begin = consumer_warp + 1;
                }

                for (int q_subtile = q_subtile_begin; q_subtile < C::QSubtiles; ++q_subtile) {
                    const int q_tile_idx = q_block_tile_idx * C::QSubtiles + q_subtile;
                    warp::load<dim::DEPTH>(q_reg, g.q, {batch_idx, q_tile_idx, head_idx, 0});
                    warp::load<dim::DEPTH>(do_reg, g.dout, {batch_idx, q_tile_idx, head_idx, 0});
                    warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
                    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
                    bwd_cute16_kernel_candidate::detail::repair_dkdv_step_chunked<C, true, true>(
                        dk0_reg_accum,
                        dk1_reg_accum,
                        dk2_reg_accum,
                        dv_reg_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        dpsum_vec,
                        g.scale,
                        g.scale_log2e,
                        q_tile_idx,
                        (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + consumer_warp,
                        g.seq_len
                    );
                }
            }
        }
    }
    __syncthreads();

    for (int q_block_idx = first_dense_q_block; q_block_idx < q_blocks; ++q_block_idx) {
        const int phase = (q_block_idx - first_dense_q_block) & 1;
        if (is_load) {
            coord<q_tile> q_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx, q_b[0]);
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, q_tile_idx, o_b[0]);
            const int q_tile_base = q_block_idx * C::QSubtiles;
            for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_stage_vec, dpsum_stage_vec;
                warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(lse_log2_smem[subtile], lse_stage_vec);
                warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[subtile], dpsum_stage_vec);
            }
        }
        if (is_compute) {
            if constexpr (!kUseTmemDense || kUseDirectDenseReconstruct) {
                wait(q_b[0], phase);
                wait(o_b[0], phase);
            }
        }
        __syncthreads();

        if (is_compute) {
            if constexpr (kUseTmemDense) {
                rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
                rt_bf<kRefTileM, C::TileRows> p_block_t_mma, ds_block_t_mma;
                wait(kv_b, 0);
                if constexpr (kUseCuteDenseHelper) {
                    bwd_cute::detail::cute_compute_dkdv_loop<CAUSAL, C>(
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
                        phase,
                        q_block_idx > (CAUSAL ? (kv_tile_base + consumer_idx) : 0),
                        q_block_idx,
                        kv_tile_base + consumer_idx,
                        g.seq_len
                    );
                } else if constexpr (kUseDirectDenseReconstruct) {
                    bwd_hot::detail::hot_compute_dkdv_tmem_exact_loop_direct_qdo<
                        CAUSAL,
                        kUseTmemDenseDk,
                        kUseTmemDenseDv,
                        C
                    >(
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
                        dk0_reg_accum,
                        dk1_reg_accum,
                        dk2_reg_accum,
                        dv_reg_accum,
                        q_smem,
                        k_smem,
                        v_smem,
                        do_smem,
                        g.q,
                        g.dout,
                        g.lse_log2,
                        g.dpsum,
                        batch_idx,
                        head_idx,
                        g.scale,
                        g.scale_log2e,
                        q_block_idx > (CAUSAL ? (kv_tile_base + consumer_idx) : 0),
                        q_block_idx,
                        kv_tile_base + consumer_idx,
                        g.seq_len
                    );
                } else {
                    bwd_hot::detail::hot_compute_dkdv_tmem_exact_loop<
                        CAUSAL,
                        kUseTmemDenseDk,
                        kUseTmemDenseDv,
                        C
                    >(
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
                        dk0_reg_accum,
                        dk1_reg_accum,
                        dk2_reg_accum,
                        dv_reg_accum,
                        q_smem,
                        k_smem,
                        v_smem,
                        do_smem,
                        lse_log2_smem,
                        dpsum_smem,
                        g.scale,
                        g.scale_log2e,
                        phase,
                        q_block_idx > (CAUSAL ? (kv_tile_base + consumer_idx) : 0),
                        q_block_idx,
                        kv_tile_base + consumer_idx,
                        g.seq_len
                    );
                }
            }
        }
        __syncthreads();
    }

    if (is_compute) {
        const int kv_subtile_idx = (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + warpgroup::warpid();
        if constexpr (kUseTmemDense) {
            rt_fl<kRefTileM, 64> dk0_reg, dk1_reg, dk2_reg;
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
            warp::add(dk0_reg, dk0_reg, dk0_reg_accum);
            warp::add(dk1_reg, dk1_reg, dk1_reg_accum);
            warp::add(dk2_reg, dk2_reg, dk2_reg_accum);
            warp::add(dv_reg, dv_reg, dv_reg_accum);
            warpgroup::store(smem.epilogue.dk0_smem[consumer_idx], dk0_reg);
            warpgroup::store(smem.epilogue.dk1_smem[consumer_idx], dk1_reg);
            warpgroup::store(smem.epilogue.dk2_smem[consumer_idx], dk2_reg);
            warpgroup::store(smem.epilogue.dv_smem[consumer_idx], dv_reg);
            group<4>::sync(warpgroup::groupid() + 4);
            if (warpgroup::warpid() == 0) {
                coord<typename hot_bshd_dkdv_globals<C>::dk_full_tile> dk0_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 0};
                coord<typename hot_bshd_dkdv_globals<C>::dk_full_tile> dk1_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 1};
                coord<typename hot_bshd_dkdv_globals<C>::dk_full_tile> dk2_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 2};
                coord<typename hot_bshd_dkdv_globals<C>::dv_full_tile> dv_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 0};
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.dk0_full, smem.epilogue.dk0_smem[consumer_idx], dk0_tile_idx);
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.dk1_full, smem.epilogue.dk1_smem[consumer_idx], dk1_tile_idx);
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.dk2_full, smem.epilogue.dk2_smem[consumer_idx], dk2_tile_idx);
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.dv_full, smem.epilogue.dv_smem[consumer_idx], dv_tile_idx);
                warp::tma::store_commit_group();
                warp::tma::store_async_wait();
            }
        } else {
            warp::store<dim::DEPTH>(g.dk, dk0_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
            warp::store<dim::DEPTH>(g.dk, dk1_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 1});
            warp::store<dim::DEPTH>(g.dk, dk2_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 2});
            warp::store<dim::DEPTH>(g.dv, dv_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
        }
    }
}

template <typename C>
inline void launch_fa4_bshd_main_kernel(
    const bwd_fa4::main_globals<C> &g,
    bool causal,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(num_k_blocks, heads, batch_size),
        dim3(C::BlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if (causal) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, fa4_bshd_main_kernel<C, true>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, fa4_bshd_main_kernel<C, false>, g));
    }
}

template <typename C>
inline void launch_fa4_bshd_direct_dq_main_kernel(
    const fa4_bshd_direct_dq_globals<C> &g,
    bool causal,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(num_k_blocks, heads, batch_size),
        dim3(C::BlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if (causal) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, fa4_bshd_direct_dq_main_kernel<C, true>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, fa4_bshd_direct_dq_main_kernel<C, false>, g));
    }
}

template <typename C>
inline void launch_fa4_bshd_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &dq_accum,
    at::Tensor &dq_semaphore,
    bool causal,
    float scale,
    int actual_seq_len,
    bool deterministic
) {
    using G = bwd_fa4::main_globals<C>;
    using dqacc_gl = typename G::dqacc_gl;
    const int q_tile_groups = static_cast<int>(dq_accum.size(2));
    const int dq_tiles = static_cast<int>(q.size(1) / kRefTileM);
    const int dqacc_rows = q_tile_groups * C::ClusterSize * C::Mb;

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        ::kittens::make_gl<dqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            dqacc_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        actual_seq_len,
        dq_semaphore.defined() ? reinterpret_cast<int *>(dq_semaphore.data_ptr<int>()) : nullptr,
        static_cast<int>(q.size(2)),
        dq_tiles,
        deterministic ? 1 : 0,
    };

    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fa4_bshd_main_kernel<C>(
        g,
        causal,
        num_k_blocks,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_hot_bshd_dkdv_main_kernel(
    const hot_bshd_dkdv_globals<C> &g,
    bool causal,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(num_k_blocks, heads, batch_size),
        dim3(C::DkdvBlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if (causal) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, hot_bshd_dkdv_main_kernel<C, true>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, hot_bshd_dkdv_main_kernel<C, false>, g));
    }
}

template <typename C>
inline void launch_fa4_bshd_backward_direct_dq(
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
    float scale,
    int actual_seq_len
) {
    using G = fa4_bshd_direct_dq_globals<C>;

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
        actual_seq_len,
    };

    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    CUDACHECK(cudaMemsetAsync(
        dq.data_ptr<float>(),
        0,
        static_cast<size_t>(dq.numel()) * sizeof(float),
        stream
    ));
    launch_fa4_bshd_direct_dq_main_kernel<C>(
        g,
        causal,
        num_k_blocks,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_hot_bshd_dkdv_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale
) {
    using G = hot_bshd_dkdv_globals<C>;

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
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
    };

    const int num_k_blocks = static_cast<int>(q.size(1) / (C::TileRows * C::ConsumerWarpgroups));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_hot_bshd_dkdv_main_kernel<C>(
        g,
        causal,
        num_k_blocks,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
__global__ __launch_bounds__(kWarpThreads, 8)
void fa4_bshd_postprocess_kernel(const __grid_constant__ bwd_fa4::postprocess_globals<C> g) {
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
            int *sem = g.dq_semaphore + bwd_fa4::detail::dq_semaphore_index(g, batch_idx, head_idx, q_tile_idx, cluster_rank);
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
    warp::store<dim::DEPTH>(g.dq, dq_reg, {batch_idx, q_tile_idx, head_idx, 0});
}

template <typename C>
inline void launch_fa4_bshd_postprocess(
    at::Tensor &dq_accum,
    at::Tensor &dq,
    at::Tensor &dq_semaphore,
    bool deterministic
) {
    using G = bwd_fa4::postprocess_globals<C>;
    using dqacc_gl = typename G::dqacc_gl;
    const int q_tile_groups = static_cast<int>(dq_accum.size(2));
    const int q_tiles = static_cast<int>(dq.size(1) / kRefTileM);
    const int dqacc_rows = q_tile_groups * C::ClusterSize * C::Mb;
    const int cluster_groups = static_cast<int>(dq.size(1) / (kForwardTileN * C::ClusterSize));

    G g{
        ::kittens::make_gl<dqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(dq.size(0)),
            static_cast<int>(dq.size(2)),
            dqacc_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
        dq_semaphore.defined() ? reinterpret_cast<int *>(dq_semaphore.data_ptr<int>()) : nullptr,
        static_cast<int>(dq.size(2)),
        q_tiles,
        cluster_groups,
        deterministic ? 1 : 0,
    };

    dim3 grid(q_tiles, dq.size(2), dq.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    fa4_bshd_postprocess_kernel<C><<<grid, kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace detail

template <typename C>
inline void launch_fa4_exact_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    using ExactConfig = bwd_fa4::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;

    at::Tensor q_bhsd = q.permute({0, 2, 1, 3}).contiguous();
    at::Tensor k_bhsd = k.permute({0, 2, 1, 3}).contiguous();
    at::Tensor v_bhsd = v.permute({0, 2, 1, 3}).contiguous();
    at::Tensor out_bhsd = out.permute({0, 2, 1, 3}).contiguous();
    at::Tensor dout_bhsd = dout.permute({0, 2, 1, 3}).contiguous();
    at::Tensor lse_bh1s = lse.permute({0, 2, 1}).contiguous().unsqueeze(2);

    at::Tensor dq_bhsd = at::empty_like(q_bhsd, lse.options());
    at::Tensor dk_bhsd = at::empty_like(k_bhsd, lse.options());
    at::Tensor dv_bhsd = at::empty_like(v_bhsd, lse.options());
    at::Tensor dpsum = at::empty({q_bhsd.size(0), q_bhsd.size(1), 1, q_bhsd.size(2)}, lse.options());
    at::Tensor lse_log2 = at::empty({q_bhsd.size(0), q_bhsd.size(1), 1, q_bhsd.size(2)}, lse.options());
    at::Tensor dq_accum = at::zeros(
        {q_bhsd.size(0), q_bhsd.size(1), q_bhsd.size(2) / kForwardTileM, C::ClusterSize, kForwardTileM, kB300QKDim},
        lse.options()
    );
    at::Tensor dq_semaphore = at::empty(
        {q_bhsd.size(0), q_bhsd.size(1), q_bhsd.size(2) / kRefTileM, C::ClusterSize},
        q.options().dtype(at::kInt)
    );

    bwd_fa4::launch_preprocess<bwd_fa4::preprocess_config<kB300VDim>>(
        out_bhsd,
        dout_bhsd,
        lse_bh1s,
        dpsum,
        lse_log2,
        dq_accum,
        dq_semaphore
    );
    bwd_fa4::launch_backward<ExactConfig>(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        dout_bhsd,
        lse_log2,
        dpsum,
        dk_bhsd,
        dv_bhsd,
        dq_accum,
        dq_semaphore,
        causal,
        scale,
        static_cast<int>(q.size(1)),
        deterministic
    );
    bwd_fa4::launch_postprocess<ExactConfig>(dq_accum, dq_bhsd, dq_semaphore, deterministic);

    dq.copy_(dq_bhsd.permute({0, 2, 1, 3}).contiguous());
    dk.copy_(dk_bhsd.permute({0, 2, 1, 3}).contiguous());
    dv.copy_(dv_bhsd.permute({0, 2, 1, 3}).contiguous());
}

template <typename C>
inline void launch_fa4_exact_backward_bshd(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    using ExactConfig = bwd_fa4::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;

    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor dq_accum = at::zeros(
        {q.size(0), q.size(2), q.size(1) / kForwardTileM, ExactConfig::ClusterSize, kForwardTileM, kB300QKDim},
        lse.options()
    );
    at::Tensor dq_semaphore = at::zeros(
        {q.size(0), q.size(2), q.size(1) / kRefTileM, ExactConfig::ClusterSize},
        q.options().dtype(at::kInt)
    );

    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
    detail::launch_fa4_bshd_backward<ExactConfig>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dk,
        dv,
        dq_accum,
        dq_semaphore,
        causal,
        scale,
        static_cast<int>(q.size(1)),
        deterministic
    );
    detail::launch_fa4_bshd_postprocess<ExactConfig>(dq_accum, dq, dq_semaphore, deterministic);
}

template <typename C>
inline void launch_fa4_exact_backward_bshd_direct_dq(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    if (deterministic) {
        launch_fa4_exact_backward_bshd<C>(q, k, v, out, lse, dout, dq, dk, dv, causal, scale, deterministic);
        return;
    }

    using ExactConfig = bwd_fa4::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());

    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
    detail::launch_fa4_bshd_backward_direct_dq<ExactConfig>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk,
        dv,
        causal,
        scale,
        static_cast<int>(q.size(1))
    );
}

template <typename C, typename DkdvOutT = float, typename DqOutT = float>
inline void launch_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    if (deterministic) {
        launch_fa4_exact_backward<C>(q, k, v, out, lse, dout, dq, dk, dv, causal, scale, deterministic);
        return;
    }

    if (causal) {
        using Cluster1Config = config<C::Mb, C::Nb, C::Dqk, C::Dvo, 1>;
        using DedicatedLoadDqConfig = bwd_cute16_kernel_candidate::dq_only_dedicated_load_config<C::Mb, C::Nb, C::Dqk, C::Dvo, 1>;
        using Cluster2DqConfig = bwd_cute16_kernel_candidate::dq_only_compact_cluster2_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;
        using HotClusteredDqConfig = bwd_cute16_kernel_candidate::dq_only_clustered_cluster1_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;
        using Seq2048ExactConfig = bwd_cute16_kernel_candidate::seq2048_exact_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;
        using HotBshdConfig = bwd_hot::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
        using FullStockCuTeConfig = bwd_hot::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
        using FullStockCuTe16Config = tkfa4::bwd_cute16::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
        using CuTe16NativeExactConfig = tkfa4::bwd_cute16::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
        using SharedDsMonolithicConfig = config<C::Mb, C::Nb, C::Dqk, C::Dvo, 1>;
        constexpr bool kUseCuTe16NativeExact2048 = false;
        constexpr bool kUseFullStockCuTe2048 = false;
        constexpr bool kUseFullStockCuTe16Exact2048 = false;
        constexpr bool kUseCute16HotExact2048 = false;
        constexpr bool kUseSharedDsMonolithic2048 = false;
        constexpr bool kUseDsScratchDqOnly2048 = false;
        constexpr bool kUseClusteredDqOnly = false;
        constexpr bool kUseReducedDqOnly2048 = false;
        constexpr bool kUseHotClusteredDqOnly2048 = true;
        constexpr bool kUseDedicatedLoadDqOnly2048 = false;
        constexpr bool kUseSeq2048ExactOwnership = false;
        constexpr bool kUseDkdvPoolStream2048 = false;
        constexpr bool kUseScratchBackedDqOnly = kUseReducedDqOnly2048 || kUseHotClusteredDqOnly2048;
        constexpr bool kUseDirectDqOnly = !kUseClusteredDqOnly && !kUseScratchBackedDqOnly;
        const bool can_use_exact_split_concurrent =
            q.size(1) >= 2048 &&
            q.size(1) <= 32768 &&
            q.size(1) == k.size(1) &&
            q.size(1) == v.size(1) &&
            q.size(1) == out.size(1) &&
            q.size(1) == dout.size(1) &&
            (q.size(1) % (kForwardTileM * 2) == 0);
        if constexpr (kUseCuTe16NativeExact2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                bwd_cute16_candidate_cute::launch_causal_backward_cute16_native_exact<CuTe16NativeExactConfig>(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    dout,
                    dq,
                    dk,
                    dv,
                    scale
                );
                return;
            }
        }
        if constexpr (kUseFullStockCuTe2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
                bwd_cute16_candidate_cute::launch_causal_backward<FullStockCuTeConfig>(
                    q,
                    k,
                    v,
                    dout,
                    lse_log2,
                    dpsum,
                    dq,
                    dk,
                    dv,
                    scale
                );
                return;
            }
        }
        if constexpr (kUseFullStockCuTe16Exact2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                tkfa4::bwd_cute16::launch_backward<FullStockCuTe16Config>(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    dout,
                    dq,
                    dk,
                    dv,
                    causal,
                    scale,
                    false
                );
                return;
            }
        }
        if constexpr (kUseCute16HotExact2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                auto current_stream = at::cuda::getCurrentCUDAStream();
                at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
                bwd_cute16_kernel_candidate::launch_backward_dkdv_only<Cluster1Config, DkdvOutT>(
                    q,
                    k,
                    v,
                    dout,
                    lse_log2,
                    dpsum,
                    dk,
                    dv,
                    causal,
                    scale
                );
                CUDACHECK(cudaMemsetAsync(
                    dq.data_ptr(),
                    0,
                    dq.nbytes(),
                    current_stream.stream()
                ));
                bwd_cute16_kernel_candidate::launch_backward_dq_only<Cluster1Config>(
                    q,
                    k,
                    v,
                    dout,
                    lse_log2,
                    dpsum,
                    dq,
                    scale,
                    current_stream.stream()
                );
                return;
            }
        }
        if constexpr (kUseSharedDsMonolithic2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
                bwd_cute16_kernel_candidate::launch_backward_shared_ds_exact<SharedDsMonolithicConfig>(
                    q,
                    k,
                    v,
                    dout,
                    lse_log2,
                    dpsum,
                    dq,
                    dk,
                    dv,
                    scale,
                    at::cuda::getCurrentCUDAStream().stream()
                );
                return;
            }
        }
        if (can_use_exact_split_concurrent) {
            auto current_stream = at::cuda::getCurrentCUDAStream();
            const bool use_hot_clustered_dq_probe = detail::hot_clustered_dq_probe_enabled();
            const int split_dqacc_rows = use_hot_clustered_dq_probe
                ? static_cast<int>(q.size(1) * HotClusteredDqConfig::ClusterSize)
                : (kUseDirectDqOnly
                    ? 0
                    : static_cast<int>(q.size(1) * (kUseSeq2048ExactOwnership ? Seq2048ExactConfig::ClusterSize : (kUseHotClusteredDqOnly2048 ? HotClusteredDqConfig::ClusterSize : (kUseClusteredDqOnly ? Cluster2DqConfig::ClusterSize : Cluster1Config::ClusterSize)))));
            auto &split = detail::get_split_hybrid_resources(
                current_stream.device_index(),
                current_stream.stream(),
                q,
                lse,
                split_dqacc_rows,
                C::Dqk,
                kUseDsScratchDqOnly2048
            );
            CUDACHECK(cudaEventRecord(split.call_entry_ready, current_stream.stream()));
            CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.call_entry_ready));
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
                CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
            }
            launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, split.dpsum, split.lse_log2);
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
                CUDACHECK(cudaEventRecord(split.dq_start, split.dq_stream));
            }
            if (use_hot_clustered_dq_probe) {
                if constexpr (
                    bwd_cute16_kernel_candidate::kUseDirectFinalClusteredDq &&
                    kUseHotClusteredDqOnly2048
                ) {
                    CUDACHECK(cudaMemsetAsync(
                        dq.data_ptr(),
                        0,
                        dq.nbytes(),
                        split.dq_stream
                    ));
                } else {
                    CUDACHECK(cudaMemsetAsync(
                        split.dqacc.data_ptr<float>(),
                        0,
                        static_cast<size_t>(split.dqacc.numel()) * sizeof(float),
                        split.dq_stream
                    ));
                }
            } else if constexpr (
                bwd_cute16_kernel_candidate::kUseDirectFinalClusteredDq &&
                kUseHotClusteredDqOnly2048
            ) {
                CUDACHECK(cudaMemsetAsync(
                    dq.data_ptr(),
                    0,
                    dq.nbytes(),
                    split.dq_stream
                ));
            } else if constexpr (kUseDirectDqOnly) {
                CUDACHECK(cudaMemsetAsync(
                    dq.data_ptr(),
                    0,
                    dq.nbytes(),
                    split.dq_stream
                ));
            } else {
                CUDACHECK(cudaMemsetAsync(
                    split.dqacc.data_ptr<float>(),
                    0,
                    static_cast<size_t>(split.dqacc.numel()) * sizeof(float),
                    split.dq_stream
                ));
            }
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.dq_zero_end, split.dq_stream));
            }
            CUDACHECK(cudaEventRecord(split.preprocess_done, current_stream.stream()));
            CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.preprocess_done));
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.dq_ready, split.dq_stream));
            }
            if constexpr (kUseDkdvPoolStream2048) {
                CUDACHECK(cudaStreamWaitEvent(split.dkdv_stream, split.preprocess_done));
            } else if (use_hot_clustered_dq_probe) {
                CUDACHECK(cudaStreamWaitEvent(split.dkdv_stream, split.preprocess_done));
            }
            cudaStream_t dkdv_exec_stream = (kUseDkdvPoolStream2048 || use_hot_clustered_dq_probe)
                ? split.dkdv_stream
                : current_stream.stream();
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.dkdv_start, dkdv_exec_stream));
            }
            if constexpr (kUseSeq2048ExactOwnership) {
                bwd_cute16_kernel_candidate::launch_backward_seq2048_exact_dkdv_only<Seq2048ExactConfig>(
                    q,
                    k,
                    v,
                    dout,
                    split.lse_log2,
                    split.dpsum,
                    dk,
                    dv,
                    scale,
                    dkdv_exec_stream
                );
                bwd_cute16_kernel_candidate::launch_backward_dq_only<Cluster1Config>(
                    q,
                    k,
                    v,
                    dout,
                    split.lse_log2,
                    split.dpsum,
                    dq,
                    scale,
                    split.dq_stream
                );
            } else if constexpr (kUseDsScratchDqOnly2048) {
                bwd_cute16_kernel_candidate::launch_backward_dkdv_only_store_ds<Cluster1Config>(
                    q,
                    k,
                    v,
                    dout,
                    split.lse_log2,
                    split.dpsum,
                    dk,
                    dv,
                    split.ds_scratch,
                    scale,
                    dkdv_exec_stream
                );
                bwd_cute16_kernel_candidate::launch_backward_dq_from_ds<Cluster1Config>(
                    k,
                    split.ds_scratch,
                    dq,
                    split.dq_stream
                );
            } else {
                bwd_cute16_kernel_candidate::launch_backward_dkdv_only<Cluster1Config, DkdvOutT>(
                    q,
                    k,
                    v,
                    dout,
                    split.lse_log2,
                    split.dpsum,
                    dk,
                    dv,
                    scale,
                    dkdv_exec_stream
                );
                if (use_hot_clustered_dq_probe) {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<HotClusteredDqConfig, DqOutT>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream,
                        &split.dqacc
                    );
                } else if constexpr (kUseClusteredDqOnly) {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only<Cluster2DqConfig>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream
                    );
                } else if constexpr (kUseHotClusteredDqOnly2048) {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<HotClusteredDqConfig, DqOutT>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream,
                        &split.dqacc
                    );
                } else if constexpr (kUseDedicatedLoadDqOnly2048) {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only<DedicatedLoadDqConfig, kUseDirectDqOnly>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream,
                        kUseDirectDqOnly ? nullptr : &split.dqacc
                    );
                } else {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only<Cluster1Config, kUseDirectDqOnly>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream,
                        kUseDirectDqOnly ? nullptr : &split.dqacc
                    );
                }
            }
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.dkdv_end, dkdv_exec_stream));
                CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
            }
            if constexpr (kUseDkdvPoolStream2048) {
                CUDACHECK(cudaEventRecord(split.dkdv_done, dkdv_exec_stream));
                CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dkdv_done));
            } else if (use_hot_clustered_dq_probe) {
                CUDACHECK(cudaEventRecord(split.dkdv_done, dkdv_exec_stream));
                CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dkdv_done));
            }
            CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
            CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
            if (detail::split_timing_enabled()) {
                float total_ms = 0.0f;
                float preprocess_ms = 0.0f;
                float dkdv_ms = 0.0f;
                float dq_ms = 0.0f;
                float dq_zero_ms = 0.0f;
                float dq_wait_ms = 0.0f;
                float dq_kernel_ms = 0.0f;
                CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
                CUDACHECK(cudaEventSynchronize(split.total_end));
                CUDACHECK(cudaEventElapsedTime(&total_ms, split.total_start, split.total_end));
                CUDACHECK(cudaEventElapsedTime(&preprocess_ms, split.preprocess_start, split.preprocess_end));
                CUDACHECK(cudaEventElapsedTime(&dkdv_ms, split.dkdv_start, split.dkdv_end));
                CUDACHECK(cudaEventElapsedTime(&dq_ms, split.dq_start, split.dq_end));
                CUDACHECK(cudaEventElapsedTime(&dq_zero_ms, split.dq_start, split.dq_zero_end));
                CUDACHECK(cudaEventElapsedTime(&dq_wait_ms, split.dq_zero_end, split.dq_ready));
                CUDACHECK(cudaEventElapsedTime(&dq_kernel_ms, split.dq_ready, split.dq_end));
                std::fprintf(
                    stderr,
                    "split_timing_us preprocess=%.2f dkdv=%.2f dq=%.2f dq_zero=%.2f dq_wait=%.2f dq_kernel=%.2f total=%.2f\n",
                    preprocess_ms * 1000.0f,
                    dkdv_ms * 1000.0f,
                    dq_ms * 1000.0f,
                    dq_zero_ms * 1000.0f,
                    dq_wait_ms * 1000.0f,
                    dq_kernel_ms * 1000.0f,
                    total_ms * 1000.0f
                );
            }
            return;
        }

        at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
        at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
        launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
        at::Tensor dqacc = at::zeros({q.size(0), q.size(2), q.size(1) * Cluster1Config::ClusterSize, C::Dqk}, lse.options());
        bwd_cute16_kernel_candidate::launch_backward<Cluster1Config>(
            q,
            k,
            v,
            dout,
            q,
            k,
            v,
            dout,
            lse_log2,
            dpsum,
            dq,
            dk,
            dv,
            dqacc,
            nullptr,
            causal,
            scale,
            deterministic
        );
        return;
    }

    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor dqacc = at::zeros({q.size(0), q.size(2), q.size(1) * C::ClusterSize, C::Dqk}, lse.options());

    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
    bwd_cute16_kernel_candidate::launch_backward<C>(
        q,
        k,
        v,
        dout,
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk,
        dv,
        dqacc,
        nullptr,
        causal,
        scale,
        deterministic
    );
}

template <typename C, typename DqOutT = float>
inline void launch_backward_cute_parity_2cta_dkdv(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    TORCH_CHECK(causal, "CuTe parity 2CTA dK/dV candidate only supports causal=True");
    TORCH_CHECK(!deterministic, "CuTe parity 2CTA dK/dV candidate only supports deterministic=False");
    TORCH_CHECK(q.size(1) == 2048, "CuTe parity 2CTA dK/dV candidate is fixed to seqlen 2048");

    using Seq2048DkdvConfig = bwd_cute16_kernel_candidate::seq2048_exact_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;
    using HotClusteredDqConfig = bwd_cute16_kernel_candidate::dq_only_clustered_cluster1_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        static_cast<int>(q.size(1) * HotClusteredDqConfig::ClusterSize),
        C::Dqk,
        false
    );
    CUDACHECK(cudaEventRecord(split.call_entry_ready, current_stream.stream()));
    CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.call_entry_ready));

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
    }
    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, split.dpsum, split.lse_log2);
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.dq_start, split.dq_stream));
    }

    CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), split.dq_stream));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_zero_end, split.dq_stream));
    }

    CUDACHECK(cudaEventRecord(split.preprocess_done, current_stream.stream()));
    CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.preprocess_done));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_ready, split.dq_stream));
        CUDACHECK(cudaEventRecord(split.dkdv_start, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::launch_backward_seq2048_exact_dkdv_only<Seq2048DkdvConfig>(
        q,
        k,
        v,
        dout,
        split.lse_log2,
        split.dpsum,
        dk,
        dv,
        scale,
        current_stream.stream()
    );
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dkdv_end, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<HotClusteredDqConfig, DqOutT>(
        q,
        k,
        v,
        dout,
        split.lse_log2,
        split.dpsum,
        dq,
        scale,
        split.dq_stream,
        &split.dqacc
    );
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
    }

    CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
    CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
    if (detail::split_timing_enabled()) {
        float total_ms = 0.0f;
        float preprocess_ms = 0.0f;
        float dkdv_ms = 0.0f;
        float dq_ms = 0.0f;
        float dq_zero_ms = 0.0f;
        float dq_wait_ms = 0.0f;
        float dq_kernel_ms = 0.0f;
        CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
        CUDACHECK(cudaEventSynchronize(split.total_end));
        CUDACHECK(cudaEventElapsedTime(&total_ms, split.total_start, split.total_end));
        CUDACHECK(cudaEventElapsedTime(&preprocess_ms, split.preprocess_start, split.preprocess_end));
        CUDACHECK(cudaEventElapsedTime(&dkdv_ms, split.dkdv_start, split.dkdv_end));
        CUDACHECK(cudaEventElapsedTime(&dq_ms, split.dq_start, split.dq_end));
        CUDACHECK(cudaEventElapsedTime(&dq_zero_ms, split.dq_start, split.dq_zero_end));
        CUDACHECK(cudaEventElapsedTime(&dq_wait_ms, split.dq_zero_end, split.dq_ready));
        CUDACHECK(cudaEventElapsedTime(&dq_kernel_ms, split.dq_ready, split.dq_end));
        std::fprintf(
            stderr,
            "parity_2cta_split_timing_us preprocess=%.2f dkdv=%.2f dq=%.2f dq_zero=%.2f dq_wait=%.2f dq_kernel=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            dkdv_ms * 1000.0f,
            dq_ms * 1000.0f,
            dq_zero_ms * 1000.0f,
            dq_wait_ms * 1000.0f,
            dq_kernel_ms * 1000.0f,
            total_ms * 1000.0f
        );
    }
}

template <
    typename C,
    typename DqOutT = float,
    bool UseChunkedTmemDq = false,
    bool UsePipelinedTmemDq = false,
    bool DoubleBufferPipelinedInputs = false,
    bool EnqueuePipelinedDqEarly = true,
    int DenseSplitCount = 1,
    int DqReplaySplitCount = 1,
    bool UseTmemScoreDp = false,
    bool UseTmemFrontier = false,
    bool FuseDenseDq = false,
    bool AdaptiveLastQuarter = false,
    bool OverlapLoadAndDqReduce = false,
    bool SkipAdaptiveTailScratch = false,
    bool UseLdsmTransposeDs = false,
    bool DoubleBufferFusedDqTma = false,
    bool MaterializeDkdvBf16 = false,
    bool CacheDkdvFp32Base = false,
    bool ReleaseTmemOperandsEachIteration = false,
    bool SerializePipelinedDqBeforeDkdv = false,
    bool SerializeDenseFrontier = false
>
inline void launch_backward_dense_tmem_frontier_dkdv(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic,
    at::Tensor *materialized_dk = nullptr,
    at::Tensor *materialized_dv = nullptr
) {
    static_assert(!(UseChunkedTmemDq && UsePipelinedTmemDq), "Select only one opt-in TMEM dQ route");
    static_assert(
        !DoubleBufferPipelinedInputs || UsePipelinedTmemDq,
        "Double-buffered inputs require the pipelined TMEM dQ route"
    );
    static_assert(
        DenseSplitCount == 1 || DenseSplitCount == 2 || DenseSplitCount == 3 ||
            DenseSplitCount == 4 || DenseSplitCount == 8,
        "Dense split count must be 1, 2, 3, 4, or 8"
    );
    static_assert(
        DqReplaySplitCount == 1 || DqReplaySplitCount == 2,
        "dQ replay split count must be 1 or 2"
    );
    static_assert(
        DqReplaySplitCount == 1 || UsePipelinedTmemDq,
        "dQ replay splitting requires the pipelined TMEM dQ route"
    );
    static_assert(!FuseDenseDq || UseTmemScoreDp, "fused dQ requires TMEM score/dP");
    static_assert(!FuseDenseDq || UseTmemFrontier, "fused dQ requires the TMEM frontier owner");
    static_assert(!FuseDenseDq || DqReplaySplitCount == 1, "fused dQ does not use replay splits");
    static_assert(
        !AdaptiveLastQuarter || (FuseDenseDq && DenseSplitCount == 2),
        "Adaptive last-quarter ownership requires fused split-2"
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
    static_assert(
        !CacheDkdvFp32Base || MaterializeDkdvBf16,
        "cached FP32 dK/dV bases require BF16 materialization"
    );
    TORCH_CHECK(causal, "Dense TMEM frontier dK/dV candidate only supports causal=True");
    TORCH_CHECK(!deterministic, "Dense TMEM frontier dK/dV candidate only supports deterministic=False");
    if constexpr (DenseSplitCount > 1) {
        TORCH_CHECK(
            q.size(1) >= 512 && q.size(1) % 512 == 0,
            "Split dense TMEM dK/dV candidate requires seqlen >= 512 and divisible by 512"
        );
    } else {
        TORCH_CHECK(q.size(1) == 2048, "Dense TMEM frontier dK/dV candidate is fixed to seqlen 2048");
    }

    using DenseDkdvConfig = bwd_cute16_kernel_candidate::dense_tmem_frontier_config<
        C::Mb,
        C::Nb,
        C::Dqk,
        C::Dvo
    >;
    using HotClusteredDqConfig = bwd_cute16_kernel_candidate::dq_only_clustered_cluster1_config<
        C::Mb,
        C::Nb,
        C::Dqk,
        C::Dvo
    >;
    using PipelinedDqConfig = bwd_cute16_kernel_candidate::dq_only_clustered_pipelined_config<
        C::Mb,
        C::Nb,
        C::Dqk,
        C::Dvo
    >;

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        static_cast<int>(q.size(1) * HotClusteredDqConfig::ClusterSize),
        C::Dqk,
        false,
        C::Dvo,
        DenseSplitCount,
        CacheDkdvFp32Base
    );
    CUDACHECK(cudaEventRecord(split.call_entry_ready, current_stream.stream()));
    CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.call_entry_ready));

    at::Tensor &dk_base = CacheDkdvFp32Base ? split.cached_base_dk : dk;
    at::Tensor &dv_base = CacheDkdvFp32Base ? split.cached_base_dv : dv;

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
    }
    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, split.dpsum, split.lse_log2);
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.dq_start, split.dq_stream));
    }

    CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), split.dq_stream));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_zero_end, split.dq_stream));
    }

    CUDACHECK(cudaEventRecord(split.preprocess_done, current_stream.stream()));
    CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.preprocess_done));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_ready, split.dq_stream));
    }
    if constexpr (FuseDenseDq) {
        CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
        CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
    }

    if constexpr (!FuseDenseDq && UsePipelinedTmemDq && EnqueuePipelinedDqEarly) {
        bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered_pipelined<
            PipelinedDqConfig,
            DqOutT,
            DoubleBufferPipelinedInputs,
            DqReplaySplitCount
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            scale,
            split.dq_stream,
            &split.dqacc
        );
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
        }
    } else if constexpr (!FuseDenseDq && UseChunkedTmemDq) {
        bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<
            HotClusteredDqConfig,
            DqOutT,
            true
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            scale,
            split.dq_stream,
            &split.dqacc
        );
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
        }
    }

    if constexpr (
        !FuseDenseDq &&
        UsePipelinedTmemDq &&
        EnqueuePipelinedDqEarly &&
        SerializePipelinedDqBeforeDkdv
    ) {
        const bool serialize_clustered_dq =
            ((q.size(1) == 2048 || q.size(1) == 4096) && q.size(2) == 8) ||
            (q.size(1) >= 8192 && q.size(2) >= 2);
        if (serialize_clustered_dq) {
            CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
            CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
        }
    }

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dkdv_start, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::launch_backward_dense_tmem_frontier_dkdv<
        DenseDkdvConfig,
        DenseSplitCount,
        UseTmemScoreDp,
        UseTmemFrontier,
        FuseDenseDq,
        AdaptiveLastQuarter,
        OverlapLoadAndDqReduce,
        SkipAdaptiveTailScratch,
        UseLdsmTransposeDs,
        DoubleBufferFusedDqTma,
        MaterializeDkdvBf16,
        ReleaseTmemOperandsEachIteration,
        SerializeDenseFrontier
    >(
        q,
        k,
        v,
        dout,
        split.lse_log2,
        split.dpsum,
        dk_base,
        dv_base,
        scale,
        current_stream.stream(),
        split.dkdv_stream,
        split.dense_main_done,
        split.frontier_dv_done,
        &split.frontier_dk,
        &split.frontier_dv,
        DenseSplitCount > 1 ? &split.dense_split_dk : nullptr,
        DenseSplitCount > 1 ? &split.dense_split_dv : nullptr,
        FuseDenseDq ? &dq : nullptr,
        materialized_dk,
        materialized_dv
    );
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dkdv_end, current_stream.stream()));
        if constexpr (FuseDenseDq) {
            CUDACHECK(cudaEventRecord(split.dq_end, current_stream.stream()));
        }
    }

    if constexpr (!FuseDenseDq && UsePipelinedTmemDq && !EnqueuePipelinedDqEarly) {
        bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered_pipelined<
            PipelinedDqConfig,
            DqOutT,
            DoubleBufferPipelinedInputs,
            DqReplaySplitCount
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            scale,
            split.dq_stream,
            &split.dqacc
        );
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
        }
    } else if constexpr (!FuseDenseDq && !UseChunkedTmemDq && !UsePipelinedTmemDq) {
        bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<
            HotClusteredDqConfig,
            DqOutT,
            false
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            scale,
            split.dq_stream,
            &split.dqacc
        );
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
        }
    }

    if constexpr (!FuseDenseDq) {
        CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
        CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
    }
    if (detail::split_timing_enabled()) {
        float total_ms = 0.0f;
        float preprocess_ms = 0.0f;
        float dkdv_ms = 0.0f;
        float dq_ms = 0.0f;
        float dq_zero_ms = 0.0f;
        float dq_wait_ms = 0.0f;
        float dq_kernel_ms = 0.0f;
        CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
        CUDACHECK(cudaEventSynchronize(split.total_end));
        CUDACHECK(cudaEventElapsedTime(&total_ms, split.total_start, split.total_end));
        CUDACHECK(cudaEventElapsedTime(&preprocess_ms, split.preprocess_start, split.preprocess_end));
        CUDACHECK(cudaEventElapsedTime(&dkdv_ms, split.dkdv_start, split.dkdv_end));
        CUDACHECK(cudaEventElapsedTime(&dq_ms, split.dq_start, split.dq_end));
        CUDACHECK(cudaEventElapsedTime(&dq_zero_ms, split.dq_start, split.dq_zero_end));
        CUDACHECK(cudaEventElapsedTime(&dq_wait_ms, split.dq_zero_end, split.dq_ready));
        CUDACHECK(cudaEventElapsedTime(&dq_kernel_ms, split.dq_ready, split.dq_end));
        std::fprintf(
            stderr,
            "dense_tmem_frontier_split_timing_us preprocess=%.2f dkdv=%.2f dq=%.2f dq_zero=%.2f dq_wait=%.2f dq_kernel=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            dkdv_ms * 1000.0f,
            dq_ms * 1000.0f,
            dq_zero_ms * 1000.0f,
            dq_wait_ms * 1000.0f,
            dq_kernel_ms * 1000.0f,
            total_ms * 1000.0f
        );
    }
}

template <
    typename C,
    bool UseTmemDs = true,
    bool OverlapDsExchange = true,
    bool OverlapQWithDp = true,
    bool UseTmemP = false,
    bool OverlapDoWithDp = false,
    bool UseDpOperandReadyMbar = false,
    bool UseDqOperandReadyMbar = false,
    bool OverlapDvWithDs = false,
    bool PipelineNextScore = false,
    bool PreloadDqA = false,
    bool UseScoreOperandReadyMbar = false,
    bool UseDsWarpMulticastMbar = false,
    bool UseRoleSplit = false,
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
    bool AsymmetricDvPublish = false
>
inline void launch_backward_cta2_fused_dense(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    TORCH_CHECK(causal, "2-CTA fused dense route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA fused dense route supports deterministic=False only"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA fused dense route supports batch size 1 only");
    TORCH_CHECK(
        (q.size(1) == 4096 && q.size(2) == 8) ||
            (q.size(1) == 8192 &&
             (q.size(2) == 2 || q.size(2) == 4 || q.size(2) == 8 ||
              q.size(2) == 16)) ||
            (q.size(1) == 16384 &&
             (q.size(2) == 4 || q.size(2) == 8 || q.size(2) == 16)),
        "2-CTA fused dense route supports S4096 H8 and "
        "S8192 H2/H4/H8/H16 or S16384 H4/H8/H16"
    );

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        0,
        C::Dqk,
        false,
        C::Dvo,
        1
    );

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
    }
    launch_preprocess<preprocess_config<kB300VDim>>(
        out,
        dout,
        lse,
        split.dpsum,
        split.lse_log2
    );
    CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), current_stream.stream()));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.dkdv_start, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::launch_backward_cta2_fused_dense_with_frontier(
        q,
        k,
        v,
        dout,
        split.lse_log2,
        split.dpsum,
        dq,
        dk,
        dv,
        scale,
        current_stream.stream(),
        split.dkdv_stream,
        split.dense_main_done,
        split.frontier_dv_done,
        split.frontier_dk,
        split.frontier_dv,
        UseTmemDs,
        OverlapDsExchange,
        OverlapQWithDp,
        UseTmemP,
        OverlapDoWithDp,
        UseDpOperandReadyMbar,
        UseDqOperandReadyMbar,
        OverlapDvWithDs,
        PipelineNextScore,
        PreloadDqA,
        UseScoreOperandReadyMbar,
        UseDsWarpMulticastMbar,
        UseRoleSplit,
        RetainDsExchange,
        RetainDsLocal,
        UseNormalDoDv,
        UseTmaScoreK,
        DirectNextQdoDuringDqDrain,
        SingleOwnerCluster,
        UseFastExp2,
        UseWarpStatsCache,
        PipelineLsePrefetch,
        UseDirectStatsLoads,
        SplitDvDkReady,
        StageDqAfterDv,
        StageDqPeerBeforeDv,
        UseWideDkN192,
        DirectDsHalfStore,
        AsymmetricDvPublish
    );

    if (detail::split_timing_enabled()) {
        float preprocess_ms = 0.0f;
        float dense_frontier_ms = 0.0f;
        float total_ms = 0.0f;
        CUDACHECK(cudaEventRecord(split.dkdv_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
        CUDACHECK(cudaEventSynchronize(split.total_end));
        CUDACHECK(cudaEventElapsedTime(
            &preprocess_ms,
            split.preprocess_start,
            split.preprocess_end
        ));
        CUDACHECK(cudaEventElapsedTime(
            &dense_frontier_ms,
            split.dkdv_start,
            split.dkdv_end
        ));
        CUDACHECK(cudaEventElapsedTime(
            &total_ms,
            split.total_start,
            split.total_end
        ));
        std::fprintf(
            stderr,
            "cta2_fused_dense_timing_us preprocess=%.2f dense_frontier=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            dense_frontier_ms * 1000.0f,
            total_ms * 1000.0f
        );
    }
}

template <
    typename C,
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
    bool OverlapSecondDpLoadWithReleaseBarrier = false,
    bool RelayDoDvCompletionViaExchangeWarp = false,
    bool OverlapDqPeerCopyWithDoDvCompletion = false,
    bool OverlapLocalDqStoreWithPeerCopy = false,
    bool UseNonblockingDqPublicationFollowers = false,
    bool SplitDqAliasLifetimeWithCuteTmemMap = false,
    bool DeferFirstDsTmemStoreWait = false,
    bool OverlapFinalDsTmemStoreWithPeerSharedStores = false,
    bool DelayScoreAliasReleaseUntilFirstDqTailLoad = false,
    bool InterleaveSteadyScoreExpPairs = false,
    bool ShiftOverlappingScoreHalfBeforeDpRelease = false,
    bool BuildCompactDpDescriptorsAfterWait = false,
    int LateTensorCommitAddressSharedMask = 0,
    bool CacheCompactDpDescriptorsInShared = false,
    bool OverlapFirstDpsumQuarterWithSecondPStore = false,
    bool HoistReducerDpReadyBeforeScoreWait = false,
    bool PipelineFirstDpQuarterLoads = false,
    bool PublishNextQdoAtDqAliasRelease = false,
    bool JoinNextQdoWithDqAliasRelease = false,
    bool PrecomputePostScoreFanoutAddresses = false,
    bool PrecomputeScoreIterationDeltaUnderFanout = false
>
inline void launch_backward_cta2_fused_dense_bf16_dkdv(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    TORCH_CHECK(causal, "BF16 dK/dV 2-CTA route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "BF16 dK/dV 2-CTA route supports deterministic=False only"
    );
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
        "BF16 dK/dV 2-CTA route supports B1 S2048 H8, S4096 H4/H8, S8192 H2/H4/H8/H16, S16384 H4/H8/H16/H32/H64/H128, "
        "S32768 H16/H32/H64/H128, or S65536 H16/H32/H64/H128"
    );

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        0,
        C::Dqk,
        false,
        C::Dvo,
        1
    );

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
    }
    launch_preprocess<preprocess_config<kB300VDim>>(
        out,
        dout,
        lse,
        split.dpsum,
        split.lse_log2
    );
    CUDACHECK(cudaMemsetAsync(
        dq.data_ptr(),
        0,
        dq.nbytes(),
        current_stream.stream()
    ));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.dkdv_start, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::
        launch_backward_cta2_fused_dense_bf16_with_frontier<
            CoalescedBf16Store,
            DirectAsyncPeerDs,
            ProducerBulkPeerDs,
            ProducerBulkPeerDsCtaFenceOnly,
            DqReadHandoffBeforeCompletion,
            AggregateScoreConsumed,
            DirectTmaDkQ,
            TimeoutDqWait,
            UseWideDkN192,
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
            PrefetchNextOwnerQdo,
            UseFusedTmemRuntimeAccumulationPredicate,
            UseBitwisePExpansion,
            UseFusedExp2Pack,
            PeelCausalPrefix,
            BranchlessDoSourceLoad,
            BranchlessDoSourceBaseSelect,
            PublishVOncePerOwner,
            BulkDoDvStage,
            LoaderOwnedDkQ,
            FuseScoreScaleLse,
            RetainPackedP,
            SplitDirectDpsumAcrossDpDoneWait,
            FusedExp2Fragment4First,
            CarryDirectStatsOffset,
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
            BalancedSingleOwnerSchedule,
            UseSingleOwnerWarpStatsCache,
            CacheDqStageLanePointers,
            UseSlicedFp32PForDs,
            UseTmaVWithScoreK,
            UseStatsWarpScoreFanout,
            UseBatchedDqTmemLoads,
            UseDynamicDpReleaseBarrierId,
            PreissueFirstDpHalfBeforeQdoWait,
            OverlapSecondDpLoadWithReleaseBarrier,
            RelayDoDvCompletionViaExchangeWarp,
            OverlapDqPeerCopyWithDoDvCompletion,
            OverlapLocalDqStoreWithPeerCopy,
            UseNonblockingDqPublicationFollowers,
            SplitDqAliasLifetimeWithCuteTmemMap,
            DeferFirstDsTmemStoreWait,
            OverlapFinalDsTmemStoreWithPeerSharedStores,
            DelayScoreAliasReleaseUntilFirstDqTailLoad,
            InterleaveSteadyScoreExpPairs,
            ShiftOverlappingScoreHalfBeforeDpRelease,
            BuildCompactDpDescriptorsAfterWait,
            LateTensorCommitAddressSharedMask,
            CacheCompactDpDescriptorsInShared,
            OverlapFirstDpsumQuarterWithSecondPStore,
            HoistReducerDpReadyBeforeScoreWait,
            PipelineFirstDpQuarterLoads,
            PublishNextQdoAtDqAliasRelease,
            JoinNextQdoWithDqAliasRelease,
            PrecomputePostScoreFanoutAddresses,
            PrecomputeScoreIterationDeltaUnderFanout
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            dk,
            dv,
            scale,
            current_stream.stream(),
            split.dkdv_stream,
            split.dense_main_done,
            split.frontier_dv_done,
            split.frontier_dk,
            split.frontier_dv
        );

    if (detail::split_timing_enabled()) {
        float preprocess_ms = 0.0f;
        float dense_frontier_ms = 0.0f;
        float total_ms = 0.0f;
        CUDACHECK(cudaEventRecord(split.dkdv_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
        CUDACHECK(cudaEventSynchronize(split.total_end));
        CUDACHECK(cudaEventElapsedTime(
            &preprocess_ms,
            split.preprocess_start,
            split.preprocess_end
        ));
        CUDACHECK(cudaEventElapsedTime(
            &dense_frontier_ms,
            split.dkdv_start,
            split.dkdv_end
        ));
        CUDACHECK(cudaEventElapsedTime(
            &total_ms,
            split.total_start,
            split.total_end
        ));
        std::fprintf(
            stderr,
            "cta2_fused_dense_bf16_timing_us preprocess=%.2f "
            "dense_frontier=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            dense_frontier_ms * 1000.0f,
            total_ms * 1000.0f
        );
    }
}

template <typename C, int ExactQTileCount, int ExpectedHeads>
inline void launch_backward_cta2_fused_dense_owner_q_split(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    static_assert(
        ExactQTileCount == 16 || ExactQTileCount == 32 ||
        ExactQTileCount == 64
    );
    static_assert(ExpectedHeads > 0);
    TORCH_CHECK(causal, "owner-Q split route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "owner-Q split route supports deterministic=False only"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            q.size(1) == ExactQTileCount * 128 &&
            q.size(2) == ExpectedHeads,
        "owner-Q split route received an unsupported exact shape"
    );

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        0,
        C::Dqk,
        false,
        C::Dvo,
        1,
        true
    );
    launch_preprocess<preprocess_config<kB300VDim>>(
        out,
        dout,
        lse,
        split.dpsum,
        split.lse_log2
    );
    CUDACHECK(cudaMemsetAsync(
        dq.data_ptr(),
        0,
        dq.nbytes(),
        current_stream.stream()
    ));

    bwd_cute16_kernel_candidate::
        launch_backward_cta2_fused_dense_owner_q_split<
            ExactQTileCount,
            ExpectedHeads
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            dk,
            dv,
            split.cached_base_dk,
            split.cached_base_dv,
            split.frontier_dk,
            split.frontier_dv,
            scale,
            current_stream.stream(),
            split.dkdv_stream,
            split.dense_main_done,
            split.frontier_dv_done
        );
}

template <
    typename C,
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
    bool UseReducerDqLeaderArrive = true,
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
    bool OverlapSecondDpLoadWithReleaseBarrier = false,
    bool RelayDoDvCompletionViaExchangeWarp = false,
    bool OverlapDqPeerCopyWithDoDvCompletion = false,
    bool OverlapLocalDqStoreWithPeerCopy = false,
    bool UseNonblockingDqPublicationFollowers = false,
    bool SplitDqAliasLifetimeWithCuteTmemMap = false,
    bool DeferFirstDsTmemStoreWait = false,
    bool OverlapFinalDsTmemStoreWithPeerSharedStores = false,
    bool DelayScoreAliasReleaseUntilFirstDqTailLoad = false,
    bool InterleaveSteadyScoreExpPairs = false,
    bool ShiftOverlappingScoreHalfBeforeDpRelease = false,
    bool BuildCompactDpDescriptorsAfterWait = false,
    int LateTensorCommitAddressSharedMask = 0,
    bool CacheCompactDpDescriptorsInShared = false,
    bool OverlapFirstDpsumQuarterWithSecondPStore = false,
    bool HoistReducerDpReadyBeforeScoreWait = false,
    bool PipelineFirstDpQuarterLoads = false,
    bool PublishNextQdoAtDqAliasRelease = false,
    bool JoinNextQdoWithDqAliasRelease = false,
    bool PrecomputePostScoreFanoutAddresses = false,
    bool PrecomputeScoreIterationDeltaUnderFanout = false
>
inline void launch_backward_cta2_fused_dense_bf16_integrated_frontier(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    launch_backward_cta2_fused_dense_bf16_dkdv<
        C,
        true,   // CoalescedBf16Store
        false,  // DirectAsyncPeerDs
        true,   // ProducerBulkPeerDs
        true,   // ProducerBulkPeerDsCtaFenceOnly
        true,   // DqReadHandoffBeforeCompletion
        true,   // AggregateScoreConsumed
        true,   // DirectTmaDkQ
        true,   // TimeoutDqWait
        true,   // UseWideDkN192
        true,   // TimeoutAllRoleWaits
        true,   // UseNamedDoSourceBarrier
        true,   // UseComputeScoreFanout
        true,   // UseRuntimeAccumulationPredicate
        true,   // UseReducerDqFanout
        UseReducerDqLeaderArrive,
        true,   // MergeScoreDpReady
        true,   // WideCoalescedBf16Store
        true,   // BulkPeerDsFromFullTile
        true,   // CoalescedPeerDsBulk
        true,   // WideDqKGlobalToShared
        true,   // IntegrateCausalFrontier
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
        PrefetchNextOwnerQdo,
        UseFusedTmemRuntimeAccumulationPredicate,
        UseBitwisePExpansion,
        UseFusedExp2Pack,
        PeelCausalPrefix,
        BranchlessDoSourceLoad,
        BranchlessDoSourceBaseSelect,
        PublishVOncePerOwner,
        BulkDoDvStage,
        LoaderOwnedDkQ,
        FuseScoreScaleLse,
        RetainPackedP,
        SplitDirectDpsumAcrossDpDoneWait,
        FusedExp2Fragment4First,
        CarryDirectStatsOffset,
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
        BalancedSingleOwnerSchedule,
        UseSingleOwnerWarpStatsCache,
        CacheDqStageLanePointers,
        UseSlicedFp32PForDs,
        UseTmaVWithScoreK,
        UseStatsWarpScoreFanout,
        UseBatchedDqTmemLoads,
        UseDynamicDpReleaseBarrierId,
        PreissueFirstDpHalfBeforeQdoWait,
        OverlapSecondDpLoadWithReleaseBarrier,
        RelayDoDvCompletionViaExchangeWarp,
        OverlapDqPeerCopyWithDoDvCompletion,
        OverlapLocalDqStoreWithPeerCopy,
        UseNonblockingDqPublicationFollowers,
        SplitDqAliasLifetimeWithCuteTmemMap,
        DeferFirstDsTmemStoreWait,
        OverlapFinalDsTmemStoreWithPeerSharedStores,
        DelayScoreAliasReleaseUntilFirstDqTailLoad,
        InterleaveSteadyScoreExpPairs,
        ShiftOverlappingScoreHalfBeforeDpRelease,
        BuildCompactDpDescriptorsAfterWait,
        LateTensorCommitAddressSharedMask,
        CacheCompactDpDescriptorsInShared,
        OverlapFirstDpsumQuarterWithSecondPStore,
        HoistReducerDpReadyBeforeScoreWait,
        PipelineFirstDpQuarterLoads,
        PublishNextQdoAtDqAliasRelease,
        JoinNextQdoWithDqAliasRelease,
        PrecomputePostScoreFanoutAddresses,
        PrecomputeScoreIterationDeltaUnderFanout
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        scale,
        deterministic
    );
}

}  // namespace tkfa4::bwd_cute16_candidate

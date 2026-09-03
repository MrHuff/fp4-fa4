#include <ATen/cuda/CUDAContext.h>

#include "b300_bwd_cute16_candidate_seq2048.cuh"
#include "b300_bwd_cute16_kernel_candidate.cuh"

namespace tkfa4::bwd_cute16_candidate_seq2048 {

using C = tkfa4::bwd_cute16_kernel_candidate::config<
    tkfa4::kForwardTileM,
    tkfa4::kForwardTileN,
    tkfa4::kB300QKDim,
    tkfa4::kB300VDim,
    1
>;
using G = tkfa4::bwd_cute16_kernel_candidate::main_globals<C>;

namespace detail {

namespace kernel_detail = tkfa4::bwd_cute16_kernel_candidate::detail;

constexpr int q_tiles_buffered = 4;
constexpr int num_q_blocks = 2048 / (tkfa4::kRefTileM * q_tiles_buffered);
constexpr int dense_q_block_lag = (C::WarpTiles + q_tiles_buffered - 1) / q_tiles_buffered;
constexpr int num_k_blocks = 2048 / (tkfa4::kRefTileN * C::WarpTiles);

using qk_bf_tile = st_bf<tkfa4::kRefTileM, C::Dqk, true, 64>;
using v_bf_tile = st_bf<tkfa4::kRefTileM, C::Dvo, true, 64>;
using dqacc_tile = st_fl<tkfa4::kRefTileM, C::Dqk>;
using stats_smem_tile = col_vec<st_fl<tkfa4::kRefTileM, C::Dvo, true, 64>>;
using stats_vec = typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec;

static_assert(num_q_blocks == 32, "2048 specialization expects 32 q-blocks");
static_assert(dense_q_block_lag == 2, "2048 specialization expects two mixed q-blocks");
static_assert(num_k_blocks == 16, "2048 specialization expects 16 kv blocks");

template <typename DqSmemTile>
__device__ inline void commit_dq_partial(
    const G &g,
    int batch_idx,
    int head_idx,
    int q_block_idx,
    int q_tile_idx,
    int subtile,
    int warp,
    const rt_fl<tkfa4::kRefTileM, C::Dqk> &dq_partial,
    DqSmemTile (&dq_smem)[C::WarpTiles]
) {
    if (q_block_idx > 0 || subtile > 0) {
        warp::tma::store_async_read_wait();
    }
    warp::store(dq_smem[warp], dq_partial);
    warp::tma::store_add_async(g.dqacc, dq_smem[warp], {batch_idx, head_idx, q_tile_idx, 0});
}

template <int LocalQIter>
__device__ inline void preload_q_block(
    const G &g,
    int batch_idx,
    int head_idx,
    int q_tile_base,
    int warp,
    qk_bf_tile (&q_smem)[q_tiles_buffered],
    v_bf_tile (&do_smem)[q_tiles_buffered],
    stats_smem_tile (&lse_log2_smem)[q_tiles_buffered],
    stats_smem_tile (&dpsum_smem)[q_tiles_buffered],
    kittens::semaphore (&q_b)[1],
    kittens::semaphore (&o_b)[1]
) {
    constexpr int phase = LocalQIter & 1;

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
}

template <bool DenseUnmasked, typename DqSmemTile>
__device__ inline void process_subtile(
    const G &g,
    int batch_idx,
    int head_idx,
    int kv_subtile_idx,
    int q_block_idx,
    int subtile,
    int warp,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &v_reg,
    rt_fl<tkfa4::kRefTileM, C::Dqk> &dk_accum,
    rt_fl<tkfa4::kRefTileM, C::Dvo> &dv_accum,
    qk_bf_tile (&q_smem)[q_tiles_buffered],
    v_bf_tile (&do_smem)[q_tiles_buffered],
    stats_smem_tile (&lse_log2_smem)[q_tiles_buffered],
    stats_smem_tile (&dpsum_smem)[q_tiles_buffered],
    DqSmemTile (&dq_smem)[C::WarpTiles]
) {
    const int q_tile_idx = q_block_idx * q_tiles_buffered + subtile;
    rt_bf<tkfa4::kRefTileM, C::Dqk> q_reg;
    rt_bf<tkfa4::kRefTileM, C::Dvo> do_reg;
    rt_fl<tkfa4::kRefTileM, C::Dqk> dq_partial;
    stats_vec lse_log2_vec, dpsum_vec;

    warp::load(q_reg, q_smem[subtile]);
    warp::load(do_reg, do_smem[subtile]);
    warp::load(lse_log2_vec, lse_log2_smem[subtile]);
    warp::load(dpsum_vec, dpsum_smem[subtile]);

    kernel_detail::backward_tile_step_compact<C, true, DenseUnmasked>(
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
        g.seq_len
    );

    commit_dq_partial(g, batch_idx, head_idx, q_block_idx, q_tile_idx, subtile, warp, dq_partial, dq_smem);
}

template <int QBlockIdx, int LocalQIter, typename DqSmemTile>
__device__ inline void run_mixed_q_block(
    const G &g,
    int batch_idx,
    int head_idx,
    int kv_subtile_idx,
    int warp,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &v_reg,
    rt_fl<tkfa4::kRefTileM, C::Dqk> &dk_accum,
    rt_fl<tkfa4::kRefTileM, C::Dvo> &dv_accum,
    qk_bf_tile (&q_smem)[q_tiles_buffered],
    v_bf_tile (&do_smem)[q_tiles_buffered],
    stats_smem_tile (&lse_log2_smem)[q_tiles_buffered],
    stats_smem_tile (&dpsum_smem)[q_tiles_buffered],
    DqSmemTile (&dq_smem)[C::WarpTiles],
    kittens::semaphore (&q_b)[1],
    kittens::semaphore (&o_b)[1]
) {
    constexpr int q_tile_base = QBlockIdx * q_tiles_buffered;
    preload_q_block<LocalQIter>(
        g,
        batch_idx,
        head_idx,
        q_tile_base,
        warp,
        q_smem,
        do_smem,
        lse_log2_smem,
        dpsum_smem,
        q_b,
        o_b
    );

    #pragma unroll
    for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
        const int q_tile_idx = q_tile_base + subtile;
        if (kv_subtile_idx > q_tile_idx) {
            continue;
        }
        if (kv_subtile_idx < q_tile_idx) {
            process_subtile<true>(
                g, batch_idx, head_idx, kv_subtile_idx, QBlockIdx, subtile, warp, k_reg, v_reg,
                dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem
            );
        } else {
            process_subtile<false>(
                g, batch_idx, head_idx, kv_subtile_idx, QBlockIdx, subtile, warp, k_reg, v_reg,
                dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem
            );
        }
    }
    __syncthreads();
}

template <int QBlockStart>
__device__ inline void run_from_start(
    const G &g,
    int batch_idx,
    int head_idx,
    int kv_subtile_idx,
    int warp,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &v_reg,
    rt_fl<tkfa4::kRefTileM, C::Dqk> &dk_accum,
    rt_fl<tkfa4::kRefTileM, C::Dvo> &dv_accum,
    qk_bf_tile (&q_smem)[q_tiles_buffered],
    v_bf_tile (&do_smem)[q_tiles_buffered],
    stats_smem_tile (&lse_log2_smem)[q_tiles_buffered],
    stats_smem_tile (&dpsum_smem)[q_tiles_buffered],
    dqacc_tile (&dq_smem)[C::WarpTiles],
    kittens::semaphore (&q_b)[1],
    kittens::semaphore (&o_b)[1]
) {
    run_mixed_q_block<QBlockStart, 0>(
        g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum,
        q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b
    );
    run_mixed_q_block<QBlockStart + 1, 1>(
        g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum,
        q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b
    );

    for (int q_block_idx = QBlockStart + dense_q_block_lag; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int local_q_iter = q_block_idx - QBlockStart;
        if ((local_q_iter & 1) == 0) {
            preload_q_block<0>(
                g, batch_idx, head_idx, q_tile_base, warp, q_smem, do_smem, lse_log2_smem, dpsum_smem, q_b, o_b
            );
        } else {
            preload_q_block<1>(
                g, batch_idx, head_idx, q_tile_base, warp, q_smem, do_smem, lse_log2_smem, dpsum_smem, q_b, o_b
            );
        }

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            process_subtile<true>(
                g, batch_idx, head_idx, kv_subtile_idx, q_block_idx, subtile, warp, k_reg, v_reg,
                dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem
            );
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel_causal_seq2048_c1(const __grid_constant__ G g) {
    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) dqacc_tile dq_smem[C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int kv_block_idx = static_cast<int>(blockIdx.x);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int q_block_start = kv_tile_base / q_tiles_buffered;

    rt_bf<tkfa4::kRefTileM, C::Dqk> k_reg;
    rt_bf<tkfa4::kRefTileM, C::Dvo> v_reg;
    rt_fl<tkfa4::kRefTileM, C::Dqk> dk_accum;
    rt_fl<tkfa4::kRefTileM, C::Dvo> dv_accum;

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

    switch (q_block_start >> 1) {
        case 0:
            run_from_start<0>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 1:
            run_from_start<2>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 2:
            run_from_start<4>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 3:
            run_from_start<6>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 4:
            run_from_start<8>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 5:
            run_from_start<10>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 6:
            run_from_start<12>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 7:
            run_from_start<14>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 8:
            run_from_start<16>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 9:
            run_from_start<18>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 10:
            run_from_start<20>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 11:
            run_from_start<22>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 12:
            run_from_start<24>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 13:
            run_from_start<26>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 14:
            run_from_start<28>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
        case 15:
            run_from_start<30>(g, batch_idx, head_idx, kv_subtile_idx, warp, k_reg, v_reg, dk_accum, dv_accum, q_smem, do_smem, lse_log2_smem, dpsum_smem, dq_smem, q_b, o_b);
            break;
    }

    warp::tma::store_async_read_wait();
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

}  // namespace detail

void launch_backward_cluster1_seq2048(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &dqacc,
    float scale
) {
    const int q_tiles = static_cast<int>(q.size(1) / tkfa4::kRefTileM);
    const int scratch_rows = static_cast<int>(q.size(1));

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
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
        scale * tkfa4::kLog2E,
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(1)),
        nullptr,
        static_cast<int>(q.size(2)),
        q_tiles,
        detail::num_k_blocks,
        0,
    };

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(detail::num_k_blocks, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)));
    detail::main_kernel_causal_seq2048_c1<<<grid, C::BlockThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    tkfa4::bwd_cute16_kernel_candidate::detail::launch_reduce<C>(
        g,
        q_tiles,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd_cute16_candidate_seq2048

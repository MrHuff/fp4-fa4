#pragma once

#include "fa4_bwd_dq_sm100.cuh"

namespace tkfa4::bwd {

template <int D>
struct dkdv_globals {
    using q_gl = gl<bf16, -1, -1, -1, D>;
    using k_gl = gl<bf16, -1, -1, -1, D>;
    using v_gl = gl<bf16, -1, -1, -1, D>;
    using do_gl = gl<bf16, -1, -1, -1, D>;
    using dk_gl = gl<float, -1, -1, -1, D>;
    using dv_gl = gl<float, -1, -1, -1, D>;
    using l_tile = col_vec<st_fl<kRefTileM, D>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, l_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk_gl dk;
    dv_gl dv;
    l_gl l_aux;
    d_gl delta;
    float scale;
    float scale_log2e;
    int seq_len;
    int actual_seq_len;
    int head_ratio;
};

template <int D>
struct fused_hot_globals {
    using grad_tile = st_fl<kRefTileM, D>;
    using q_gl = gl<bf16, -1, -1, -1, D>;
    using k_gl = gl<bf16, -1, -1, -1, D>;
    using v_gl = gl<bf16, -1, -1, -1, D>;
    using do_gl = gl<bf16, -1, -1, -1, D>;
    using dq_gl = gl<float, -1, -1, -1, -1, grad_tile>;
    using dk_gl = gl<float, -1, -1, -1, -1, grad_tile>;
    using dv_gl = gl<float, -1, -1, -1, -1, grad_tile>;
    using l_tile = col_vec<st_fl<kRefTileM, D>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, l_tile>;

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
    int actual_seq_len;
    int head_ratio;
};

template <int D>
struct hot_bwd_tile_dims;

template <>
struct hot_bwd_tile_dims<128> {
    static constexpr int tile_width = 128;
    static constexpr int tile_h = 4 * 16;
    static constexpr int tile_h_qo = 4 * 16;
    static constexpr int blocks_sm = 1;
};

template <int D>
struct hot_bwd_globals {
    using G = hot_bwd_tile_dims<D>;
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

namespace detail {

constexpr int kHotBackwardConsumerWarpgroups = 2;
constexpr int kHotBackwardProducerWarpgroups = 1;
constexpr int kHotBackwardNumWarpgroups = kHotBackwardConsumerWarpgroups + kHotBackwardProducerWarpgroups;
constexpr int kHotBackwardNumWorkers = kHotBackwardNumWarpgroups * kittens::WARPGROUP_WARPS;

template <typename RT, typename SMEM>
__device__ inline void stream_tile(RT &reg_tile, SMEM &smem_vec, int stage) {
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
__device__ inline void stream_sub_tile(RT &reg_tile, SMEM &smem_vec, int stage) {
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

template <int tile_h_qo, int tile_h>
__device__ inline void causal_mask(auto &reg_tile, int qo_idx) {
    const int q_blk = qo_idx * (tile_h_qo / kittens::TILE_ROW_DIM<bf16>);
    const int k_blk =
        (blockIdx.x * kHotBackwardConsumerWarpgroups * (tile_h / kittens::TILE_ROW_DIM<bf16>)) +
        ((kittens::warpid() / kittens::WARPGROUP_WARPS) * (tile_h / kittens::TILE_ROW_DIM<bf16>)) +
        (kittens::warpid() % kittens::WARPGROUP_WARPS);

    for (int j = 0; j < (tile_h_qo / kittens::TILE_ROW_DIM<bf16>); ++j) {
        const int q_idx = q_blk + j;
        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);
        if (q_idx < k_blk) {
            warp::neg_infty(attn_subtile);
        } else if (q_idx == k_blk) {
            warp::make_causal_t(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
        }
    }
}

template <bool CAUSAL, int TILE_H_QO, int TILE_H, int TILE_WIDTH>
__device__ inline void hot_compute_bwd_loop(
    kittens::semaphore *vec_b,
    kittens::semaphore *q_b,
    kittens::semaphore *o_b,
    rt_fl<16, 64> &s_block_t,
    rt_fl<16, 64> &dp_block_t,
    rt_fl<16, 64> &p_block_t,
    rt_fl<16, 64> &ds_block_t,
    rt_bf<16, 64> &p_block_t_mma,
    rt_bf<16, 64> &ds_block_t_mma,
    rt_fl<16, TILE_WIDTH> &dk_reg,
    rt_fl<16, TILE_WIDTH> &dv_reg,
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
    int next_stage,
    float scale,
    float scale_log2e
) {
    wait(vec_b[stage], ((qo_idx - q_start) / 2) % 2);
    stream_tile(s_block_t, l_smem, stage);
    wait(q_b[stage], ((qo_idx - q_start) / 2) % 2);

    warpgroup::mma_ABt(s_block_t, k_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], q_smem[stage]);

    wait(o_b[stage], ((qo_idx - q_start) / 2) % 2);
    warpgroup::mm_ABt(dp_block_t, v_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], do_smem[stage]);

    warp::mul(s_block_t, s_block_t, scale_log2e);
    if constexpr (CAUSAL) {
        causal_mask<TILE_H_QO, TILE_H>(s_block_t, qo_idx);
    }

    warp::exp2(s_block_t, s_block_t);
    warp::copy(p_block_t, s_block_t);
    warp::copy(p_block_t_mma, s_block_t);
    stream_sub_tile(dp_block_t, d_smem, stage);
    warp::mul(ds_block_t, p_block_t, dp_block_t);
    warp::mul(ds_block_t, ds_block_t, scale);

    warpgroup::mma_AB(dv_reg, p_block_t_mma, do_smem[stage]);

    warp::copy(ds_block_t_mma, ds_block_t);
    warpgroup::store(ds_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], ds_block_t);
    warpgroup::mma_AB(dk_reg, ds_block_t_mma, q_smem[stage]);
    group<8>::sync(10);
}

template <typename DKTile, typename DVTile, typename Globals>
__device__ inline void hot_kv_store(
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
            (blockIdx.x * kHotBackwardConsumerWarpgroups) + (kittens::warpid() / kittens::WARPGROUP_WARPS),
            0
        };
        warp::tma::store_add_async(dst.dk, dk_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], tile_idx);
        warp::tma::store_commit_group();
    }

    wait(bar, stage);
    warpgroup::store(dv_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], dv_reg);
    group<4>::sync(warpgroup::groupid() + 4);

    if (kittens::warpid() % 4 == 0) {
        coord<DVTile> tile_idx = {
            blockIdx.z,
            kv_head_idx,
            (blockIdx.x * kHotBackwardConsumerWarpgroups) + (kittens::warpid() / kittens::WARPGROUP_WARPS),
            0
        };
        warp::tma::store_add_async(dst.dv, dv_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], tile_idx);
        warp::tma::store_commit_group();
    }
    warp::tma::store_async_wait();
}

template <int D, bool CAUSAL>
__global__ __launch_bounds__(kHotBackwardNumWorkers * kittens::WARP_THREADS, hot_bwd_tile_dims<D>::blocks_sm)
void hot_backward_kernel(const __grid_constant__ hot_bwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al(reinterpret_cast<int*>(&__shm[0]));

    using G = hot_bwd_tile_dims<D>;
    using dk_tile = st_fl<G::tile_h, G::tile_width>;
    using dv_tile = st_fl<G::tile_h, G::tile_width>;
    using k_tile = st_bf<G::tile_h, G::tile_width>;
    using v_tile = st_bf<G::tile_h, G::tile_width>;
    using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using do_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using dq_tile = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;

    k_tile (&k_smem)[kHotBackwardConsumerWarpgroups] = al.allocate<k_tile, kHotBackwardConsumerWarpgroups>();
    v_tile (&v_smem)[kHotBackwardConsumerWarpgroups] = al.allocate<v_tile, kHotBackwardConsumerWarpgroups>();
    q_tile (&q_smem)[2] = al.allocate<q_tile, 2>();
    do_tile (&do_smem)[2] = al.allocate<do_tile, 2>();
    dq_tile (&dq_smem) = al.allocate<dq_tile>();
    l_tile (&l_smem)[2] = al.allocate<l_tile, 2>();
    d_tile (&d_smem)[2] = al.allocate<d_tile, 2>();
    dk_tile (*dk_smem) = reinterpret_cast<dk_tile*>(&k_smem[0].data[0]);
    dv_tile (*dv_smem) = reinterpret_cast<dv_tile*>(&q_smem[0].data[0]);
    attn_tile (&ds_smem)[kHotBackwardConsumerWarpgroups] = al.allocate<attn_tile, kHotBackwardConsumerWarpgroups>();

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

    int stage = 0;
    int next_stage = 1;
    const int q_start = CAUSAL ? (blockIdx.x * kHotBackwardConsumerWarpgroups) : 0;

    if (threadIdx.x == 0) {
        init_semaphore(kv_b, 0, 1);
        init_semaphore(dq_ready, 1, 0);
        for (int i = 0; i < 2; ++i) {
            init_semaphore(q_b[i], 0, 1);
            init_semaphore(o_b[i], 0, 1);
            init_semaphore(vec_b[i], 0, 1);
            init_semaphore(compute_done[i], 1, 0);
        }

        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * kHotBackwardConsumerWarpgroups);
        for (int w = 0; w < kHotBackwardConsumerWarpgroups; ++w) {
            coord<k_tile> tile_idx = {
                blockIdx.z,
                kv_head_idx,
                (blockIdx.x * kHotBackwardConsumerWarpgroups) + w,
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

    if (warpgroup_id == kHotBackwardNumWarpgroups - 1) {
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
                warp::tma::store_async_wait();

                if (laneid() == 0) {
                    arrive(dq_ready);
                }
            }
        }
    } else {
        rt_fl<16, G::tile_width> dk_reg, dv_reg;
        rt_fl<16, 64> s_block_t, p_block_t, dp_block_t, ds_block_t;
        rt_bf<16, 64> p_block_t_mma, ds_block_t_mma;

        warp::zero(dk_reg);
        warp::zero(dv_reg);

        if (warpgroup_id == 0) {
            warpgroup::increase_registers<256>();
            wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; ++qo_idx, stage ^= 1, next_stage ^= 1) {
                hot_compute_bwd_loop<CAUSAL, G::tile_h_qo, G::tile_h, G::tile_width>(
                    vec_b,
                    q_b,
                    o_b,
                    s_block_t,
                    dp_block_t,
                    p_block_t,
                    ds_block_t,
                    p_block_t_mma,
                    ds_block_t_mma,
                    dk_reg,
                    dv_reg,
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
                    next_stage,
                    g.scale,
                    g.scale_log2e
                );

                rt_fl<16, G::tile_width> dq_reg;
                warpgroup::mm_AtB(dq_reg, ds_smem[0], k_smem[0]);
                warpgroup::mma_AtB(dq_reg, ds_smem[1], k_smem[1]);

                wait(dq_ready, next_stage);
                if (qo_idx > 0) {
                    warp::tma::store_async_wait();
                }

                warpgroup::store(dq_smem, dq_reg);
                group<4>::sync(warpgroup::groupid() + 4);

                if (warpgroup::laneid() == 0) {
                    arrive(compute_done[stage]);
                }
            }
            hot_kv_store<dk_tile, dv_tile>(dk_smem, dk_reg, dv_smem, dv_reg, g, dq_ready, kv_head_idx, next_stage);
        } else {
            warpgroup::increase_registers<224>();
            wait(kv_b, 0);
            for (int qo_idx = q_start; qo_idx < qo_blocks; ++qo_idx, stage ^= 1, next_stage ^= 1) {
                hot_compute_bwd_loop<CAUSAL, G::tile_h_qo, G::tile_h, G::tile_width>(
                    vec_b,
                    q_b,
                    o_b,
                    s_block_t,
                    dp_block_t,
                    p_block_t,
                    ds_block_t,
                    p_block_t_mma,
                    ds_block_t_mma,
                    dk_reg,
                    dv_reg,
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
                    next_stage,
                    g.scale,
                    g.scale_log2e
                );
            }
            hot_kv_store<dk_tile, dv_tile>(dk_smem, dk_reg, dv_smem, dv_reg, g, dq_ready, kv_head_idx, next_stage);
        }
    }
}

template <int D>
inline int hot_backward_dynamic_smem_bytes();

template <>
inline int hot_backward_dynamic_smem_bytes<128>() {
    return 183296;
}

template <int D>
__global__ __launch_bounds__(256, 1)
void fused_persistent_hot_kernel(const __grid_constant__ fused_hot_globals<D> g) {
    static_assert(D == 128, "Fused hot backward kernel is specialized for head_dim 128.");
    constexpr int kTilesPerBlock = 8;

    using bf_tile = st_bf<kRefTileM, D>;
    using fl_tile = st_fl<kRefTileM, D>;

    __shared__ alignas(1024) bf_tile q_smem[kTilesPerBlock];
    __shared__ alignas(1024) bf_tile do_smem[kTilesPerBlock];
    __shared__ alignas(1024) fl_tile dq_smem[kTilesPerBlock];

    const int batch_size = g.q.batch();
    const int num_heads = g.k.depth();
    const int seqlen = g.seq_len;
    const int warp = threadIdx.x >> 5;
    const int task_stride = gridDim.x;
    const int kv_tiles_per_head = seqlen / (kRefTileN * kTilesPerBlock);
    const int total_tasks = batch_size * num_heads * kv_tiles_per_head;

    for (int task = blockIdx.x; task < total_tasks; task += task_stride) {
        int tmp = task;
        const int kv_tile_idx = tmp % kv_tiles_per_head;
        tmp /= kv_tiles_per_head;
        const int head_idx = tmp % num_heads;
        const int batch_idx = tmp / num_heads;
        const int kv_tile_base = kv_tile_idx * kTilesPerBlock;
        const int kv_subtile_idx = kv_tile_base + warp;

        rt_bf<kRefTileM, D> q_reg, k_reg, v_reg, do_reg;
        rt_bf<kRefTileM, kRefTileN> p_bf, ds_bf;
        rt_bf<kRefTileM, D, ducks::rt_layout::col> q_col, k_col, do_col;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col, ds_col;
        rt_fl<kRefTileM, kRefTileN> p, dp, ds;
        rt_fl<kRefTileM, D> dk_accum, dv_accum, dq_accum;
        using vec_t = typename rt_fl<kRefTileM, kRefTileN>::col_vec;
        vec_t l_aux_vec, delta_vec;

        warp::load(k_reg, g.k, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::load(v_reg, g.v, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::swap_layout(k_col, k_reg);
        warp::zero(dk_accum);
        warp::zero(dv_accum);

        const int num_q_blocks = seqlen / (kRefTileM * kTilesPerBlock);
        for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
            const int q_tile_base = q_block_idx * kTilesPerBlock;
            warp::load(q_reg, g.q, {batch_idx, head_idx, q_tile_base + warp, 0});
            warp::store(q_smem[warp], q_reg);
            warp::load(do_reg, g.dout, {batch_idx, head_idx, q_tile_base + warp, 0});
            warp::store(do_smem[warp], do_reg);
            __syncthreads();

            #pragma unroll
            for (int subtile = 0; subtile < kTilesPerBlock; ++subtile) {
                const int q_tile_idx = q_tile_base + subtile;
                warp::load(q_reg, q_smem[subtile]);
                warp::load(do_reg, do_smem[subtile]);
                warp::load(l_aux_vec, g.l_aux, {batch_idx, head_idx, 0, q_tile_idx});
                warp::load(delta_vec, g.delta, {batch_idx, head_idx, 0, q_tile_idx});

                warp::broadcast_row(p, l_aux_vec);
                warp::mma_ABt(p, q_reg, k_reg, p);
                warp::mul(p, p, g.scale_log2e);
                warp::exp2(p, p);

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

                warp::zero(dq_accum);
                warp::mma_AB(dq_accum, ds_bf, k_col, dq_accum);
                warp::store(dq_smem[warp], dq_accum);
                __syncwarp();
                warp::tma::store_add_async(g.dq, dq_smem[warp], {batch_idx, head_idx, q_tile_idx, 0});
                warp::tma::store_async_wait();
            }
            __syncthreads();
        }

        warp::store(g.dk, dk_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::store(g.dv, dv_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
    }
}

template <int D>
inline void launch_hot_backward(
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
    float scale
) {
    static_assert(D == 128, "Hot backward launch is specialized for head_dim 128.");
    TORCH_CHECK(!causal, "Fused hot backward currently supports dense non-causal attention only");

    using G = fused_hot_globals<D>;
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
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(1) / k.size(1)),
    };

    const auto *prop = at::cuda::getCurrentDeviceProperties();
    const int kv_tiles_per_head = static_cast<int>(q.size(2)) / 128;
    const int total_tasks = static_cast<int>(q.size(0) * q.size(1) * kv_tiles_per_head);
    const int blocks_target = prop->multiProcessorCount * persistent_waves(static_cast<int>(q.size(2)));
    const int blocks = blocks_target < total_tasks ? blocks_target : total_tasks;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    fused_persistent_hot_kernel<D><<<blocks > 0 ? blocks : 1, 256, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <int D>
__global__ __launch_bounds__(256, 1)
void dkdv_persistent_hot_kernel(const __grid_constant__ dkdv_globals<D> g) {
    static_assert(D == 128, "Hot dKdV kernel is specialized for head_dim 128.");
    constexpr int kTilesPerBlock = 8;

    using bf_tile = st_bf<kRefTileM, D>;

    __shared__ alignas(1024) bf_tile q_smem[kTilesPerBlock];
    __shared__ alignas(1024) bf_tile do_smem[kTilesPerBlock];

    const int batch_size = g.q.batch();
    const int num_heads = g.k.depth();
    const int seqlen = g.seq_len;
    const int warp = threadIdx.x >> 5;
    const int task_stride = gridDim.x;
    const int kv_tiles_per_head = seqlen / (kRefTileN * kTilesPerBlock);
    const int total_tasks = batch_size * num_heads * kv_tiles_per_head;

    for (int task = blockIdx.x; task < total_tasks; task += task_stride) {
        int tmp = task;
        const int kv_tile_idx = tmp % kv_tiles_per_head;
        tmp /= kv_tiles_per_head;
        const int head_idx = tmp % num_heads;
        const int batch_idx = tmp / num_heads;
        const int kv_tile_base = kv_tile_idx * kTilesPerBlock;
        const int kv_subtile_idx = kv_tile_base + warp;

        rt_bf<kRefTileM, D> q_reg, k_reg, v_reg, do_reg;
        rt_bf<kRefTileM, kRefTileN> p_bf, ds_bf;
        rt_bf<kRefTileM, D, ducks::rt_layout::col> q_col, do_col;
        rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col, ds_col;
        rt_fl<kRefTileM, kRefTileN> p, dp, ds;
        rt_fl<kRefTileM, D> dk_accum, dv_accum;
        using vec_t = typename rt_fl<kRefTileM, kRefTileN>::col_vec;
        vec_t l_aux_vec, delta_vec;

        warp::load(k_reg, g.k, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::load(v_reg, g.v, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::zero(dk_accum);
        warp::zero(dv_accum);

        const int num_q_blocks = seqlen / (kRefTileM * kTilesPerBlock);
        for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
            const int q_tile_base = q_block_idx * kTilesPerBlock;
            warp::load(q_reg, g.q, {batch_idx, head_idx, q_tile_base + warp, 0});
            warp::store(q_smem[warp], q_reg);
            warp::load(do_reg, g.dout, {batch_idx, head_idx, q_tile_base + warp, 0});
            warp::store(do_smem[warp], do_reg);
            __syncthreads();

            #pragma unroll
            for (int subtile = 0; subtile < kTilesPerBlock; ++subtile) {
                const int q_tile_idx = q_tile_base + subtile;
                warp::load(q_reg, q_smem[subtile]);
                warp::load(do_reg, do_smem[subtile]);
                warp::load(l_aux_vec, g.l_aux, {batch_idx, head_idx, 0, q_tile_idx});
                warp::load(delta_vec, g.delta, {batch_idx, head_idx, 0, q_tile_idx});

                warp::broadcast_row(p, l_aux_vec);
                warp::mma_ABt(p, q_reg, k_reg, p);
                warp::mul(p, p, g.scale_log2e);
                warp::exp2(p, p);

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
            }
            __syncthreads();
        }

        warp::store(g.dk, dk_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::store(g.dv, dv_accum, {batch_idx, head_idx, kv_subtile_idx, 0});
    }
}

template <int D>
__global__ __launch_bounds__(128, 1)
void dkdv_hot_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const __nv_bfloat16 *__restrict__ dout,
    const float *__restrict__ l_aux,
    const float *__restrict__ delta,
    float *__restrict__ dk,
    float *__restrict__ dv,
    int num_heads,
    int seqlen,
    float scale
) {
    static_assert(D == 128, "Hot dKdV kernel is specialized for head_dim 128.");
    const int seq_k = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int d = threadIdx.x;

    __shared__ float k_row[128];
    __shared__ float v_row[128];
    __shared__ float warp_score[4];
    __shared__ float warp_dp[4];
    __shared__ float prob_shared;
    __shared__ float dS_shared;

    k_row[d] = __bfloat162float(k[q_offset_bhsd(batch_idx, head_idx, seq_k, d, num_heads, seqlen, D)]);
    v_row[d] = __bfloat162float(v[q_offset_bhsd(batch_idx, head_idx, seq_k, d, num_heads, seqlen, D)]);
    __syncthreads();

    float dk_acc = 0.0f;
    float dv_acc = 0.0f;

    for (int seq_q = 0; seq_q < seqlen; ++seq_q) {
        const float qv = __bfloat162float(q[q_offset_bhsd(batch_idx, head_idx, seq_q, d, num_heads, seqlen, D)]);
        const float do_val = __bfloat162float(dout[q_offset_bhsd(batch_idx, head_idx, seq_q, d, num_heads, seqlen, D)]);

        const float score = block_reduce_sum_128(qv * k_row[d], warp_score);
        const float dP = block_reduce_sum_128(do_val * v_row[d], warp_dp);

        if (d == 0) {
            const float row_lse = -l_aux[row_offset_bhs(batch_idx, head_idx, seq_q, num_heads, seqlen)] * scale;
            const float row_delta = delta[row_offset_bhs(batch_idx, head_idx, seq_q, num_heads, seqlen)];
            prob_shared = __expf(score * scale - row_lse);
            dS_shared = prob_shared * (dP - row_delta);
        }
        __syncthreads();

        dk_acc += scale * dS_shared * qv;
        dv_acc += prob_shared * do_val;
        __syncthreads();
    }

    dk[q_offset_bhsd(batch_idx, head_idx, seq_k, d, num_heads, seqlen, D)] = dk_acc;
    dv[q_offset_bhsd(batch_idx, head_idx, seq_k, d, num_heads, seqlen, D)] = dv_acc;
}

template <int D>
inline void launch_dkdv_hot(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale
) {
    static_assert(D == 128, "Hot dKdV launch is specialized for head_dim 128.");
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int seqlen = static_cast<int>(q.size(2));
    const bool use_persistent = []() {
        const char *value = std::getenv("TK_FA4_HOT_BACKWARD_DKDV_PERSISTENT");
        return value == nullptr || std::atoi(value) != 0;
    }();
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (use_persistent) {
        using G = dkdv_globals<D>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(q),
            kittens::py::tensor_to_gl<typename G::k_gl>(k),
            kittens::py::tensor_to_gl<typename G::v_gl>(v),
            kittens::py::tensor_to_gl<typename G::do_gl>(dout),
            kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
            kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
            kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
            kittens::py::tensor_to_gl<typename G::d_gl>(delta, q.size(0), q.size(1), 1, q.size(2)),
            scale,
            scale * kLog2E,
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(1) / k.size(1)),
        };
        const auto *prop = at::cuda::getCurrentDeviceProperties();
        const int kv_tiles_per_head = seqlen / 128;
        const int total_tasks = batch_size * num_heads * kv_tiles_per_head;
        const int blocks_target = prop->multiProcessorCount * persistent_waves(seqlen);
        const int blocks = blocks_target < total_tasks ? blocks_target : total_tasks;
        dkdv_persistent_hot_kernel<D><<<blocks > 0 ? blocks : 1, 256, 0, stream>>>(g);
    } else {
        dim3 grid(seqlen, num_heads, batch_size);
        dkdv_hot_kernel<D><<<grid, 128, 0, stream>>>(
            data_ptr<__nv_bfloat16>(q),
            data_ptr<__nv_bfloat16>(k),
            data_ptr<__nv_bfloat16>(v),
            data_ptr<__nv_bfloat16>(dout),
            data_ptr<float>(l_aux),
            data_ptr<float>(delta),
            data_ptr<float>(dk),
            data_ptr<float>(dv),
            num_heads,
            seqlen,
            scale
        );
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace detail

template <int D, bool CAUSAL>
__global__ __launch_bounds__(kWarpThreads, 8)
void dkdv_kernel(const __grid_constant__ dkdv_globals<D> g) {
    const int kv_tile_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    rt_bf<kRefTileM, D> k_reg, v_reg, q_reg, do_reg;
    rt_bf<kRefTileM, kRefTileN> p_bf, ds_bf;
    rt_bf<kRefTileM, D, ducks::rt_layout::col> q_col, do_col;
    rt_bf<kRefTileM, kRefTileN, ducks::rt_layout::col> p_col, ds_col;
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;
    rt_fl<kRefTileM, D> dk_accum, dv_accum;
    using vec_t = typename rt_fl<kRefTileM, kRefTileN>::col_vec;
    vec_t l_aux, delta_vec;

    warp::load(k_reg, g.k, {batch_idx, kv_head_idx, kv_tile_idx, 0});
    warp::load(v_reg, g.v, {batch_idx, kv_head_idx, kv_tile_idx, 0});
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    const int q_head_start = kv_head_idx * g.head_ratio;
    const int q_head_end = q_head_start + g.head_ratio;
    const int num_q_tiles = g.seq_len / kRefTileM;

    for (int q_head_idx = q_head_start; q_head_idx < q_head_end; ++q_head_idx) {
        for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
            warp::load(q_reg, g.q, {batch_idx, q_head_idx, q_tile_idx, 0});
            warp::load(do_reg, g.dout, {batch_idx, q_head_idx, q_tile_idx, 0});
            warp::load(l_aux, g.l_aux, {batch_idx, q_head_idx, 0, q_tile_idx});
            warp::load(delta_vec, g.delta, {batch_idx, q_head_idx, 0, q_tile_idx});

            reconstruct_probability_tile(
                p, q_reg, k_reg, l_aux, g.scale_log2e, q_tile_idx, kv_tile_idx, g.actual_seq_len, CAUSAL
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
        }
    }

    warp::store(g.dk, dk_accum, {batch_idx, kv_head_idx, kv_tile_idx, 0});
    warp::store(g.dv, dv_accum, {batch_idx, kv_head_idx, kv_tile_idx, 0});
}

template <int D>
inline void launch_dkdv(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    int actual_seq_len
) {
    if constexpr (D == 128) {
        if (detail::use_hot_backward(q, k, causal, actual_seq_len)) {
            detail::launch_dkdv_hot<D>(q, k, v, dout, l_aux, delta, dk, dv, scale);
            return;
        }
    }

    using G = dkdv_globals<D>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        kittens::py::tensor_to_gl<typename G::d_gl>(delta, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(2)),
        actual_seq_len,
        static_cast<int>(q.size(1) / k.size(1)),
    };

    dim3 grid(k.size(2) / kRefTileN, k.size(1), k.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        dkdv_kernel<D, true><<<grid, kWarpThreads, 0, stream>>>(g);
    } else {
        dkdv_kernel<D, false><<<grid, kWarpThreads, 0, stream>>>(g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd

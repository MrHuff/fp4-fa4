#pragma once

#include "b300_bwd_cute16_kernel.cuh"

namespace tkfa4::bwd_cute16_kernel_candidate2 {

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
using config = bwd_cute16_kernel::config<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;

template <typename C>
using main_globals = bwd_cute16_kernel::main_globals<C>;

namespace detail {

template <bool CAUSAL, typename C>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void main_kernel(const __grid_constant__ main_globals<C> g);

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
    using dk_subtile_tt = tt_fl<kRefTileM, 64>;
    using dv_tt = half_tt_fl<C::Dvo>;
    using dv_subtile_tt = tt_fl<kRefTileM, C::Dvo>;
    constexpr int kLiveDQSubtiles = 2;
    struct live_shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        dq_chunk_tile dq_smem[3][kLiveDQSubtiles];
        ds_warp_tile ds_warp_smem[C::ConsumerWarpgroups][WARPGROUP_WARPS];
        stats_smem_tile lse_log2_smem[C::QSubtiles];
        stats_smem_tile dpsum_smem[C::QSubtiles];
    };
    using main_shared_storage = live_shared_storage;
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
    __shared__ __align__(16) kittens::semaphore dq_subtile_done[2][kLiveDQSubtiles];
    __shared__ __align__(16) kittens::semaphore dq_tile_done[kLiveDQSubtiles];
    __shared__ __align__(16) kittens::semaphore dq_done;
    __shared__ __align__(16) kittens::semaphore dv_ready[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore dv_store_done[C::ConsumerWarpgroups];
    __shared__ __align__(16) kittens::semaphore dk_ready[C::ConsumerWarpgroups];
    __shared__ uint32_t dk0_tmem_addr[C::ConsumerWarpgroups];
    __shared__ uint32_t dk1_tmem_addr[C::ConsumerWarpgroups];
    __shared__ uint32_t dk2_tmem_addr[C::ConsumerWarpgroups];
    __shared__ uint32_t dv_tmem_addr[C::ConsumerWarpgroups];

    const int warp = kittens::warpid();
    const bool is_reduce = warp < C::ReduceWarps;
    const bool is_compute = warp >= C::ReduceWarps && warp < C::ReduceWarps + C::ComputeWarps;
    const bool is_mma = warp == C::MmaWarpId;
    const bool is_load = warp == C::LoadWarpId;
    const bool is_relay = warp == C::RelayWarpId;
    const bool is_empty = warp == C::EmptyWarpId;
    const bool is_dv_relay = is_relay || is_empty;
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
    const int q_start_block = CAUSAL ? (kv_tile_base + 1) : 0;
    constexpr bool repair_diagonal_causal_tile = false;
    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dk_tt dk0_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk1_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk2_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dv_tt dv_accum_tt[C::ConsumerWarpgroups] = {dv_tt{0}, dv_tt{0}};
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
            init_semaphore(dv_ready[w], 1, 0);
            init_semaphore(dv_store_done[w], 1, 0);
            init_semaphore(dk_ready[w], 1, 0);
            dk0_tmem_addr[w] = 0;
            dk1_tmem_addr[w] = 0;
            dk2_tmem_addr[w] = 0;
            dv_tmem_addr[w] = 0;
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
            bwd_cute16_kernel::detail::repair_dkdv_step<true, true, C>(
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

        if (is_compute) {
            rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
            rt_bf<kRefTileM, C::TileRows> p_block_t_mma, ds_block_t_mma;
            wait(kv_b, 0);
            bwd_cute16_kernel::detail::compute_dkdv_loop<false, true, C>(
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
                    if constexpr (bwd_cute16_kernel::kUseDirectDQ) {
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
                    if constexpr (bwd_cute16_kernel::kUseDirectDQ) {
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
                    if constexpr (bwd_cute16_kernel::kUseDirectDQ) {
                        warp::tma::store_add_async(g.dq2, dq_smem[2][dq_subtile_idx], {batch_idx, q_tile_idx, head_idx, 2});
                    } else {
                        const int scratch_tile_idx = q_tile_idx * C::ClusterSize + cta_rank;
                        warp::tma::store_async(g.dqacc2, dq_smem[2][dq_subtile_idx], {batch_idx, head_idx, scratch_tile_idx, 0});
                    }
                }
                if constexpr (bwd_cute16_kernel::kUseDirectDQ) {
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
    __syncthreads();

    if (is_compute) {
        if (warpgroup::laneid() == 0) {
            dk0_tmem_addr[consumer_idx] = dk0_tt[consumer_idx].addr;
            dk1_tmem_addr[consumer_idx] = dk1_tt[consumer_idx].addr;
            dk2_tmem_addr[consumer_idx] = dk2_tt[consumer_idx].addr;
            dv_tmem_addr[consumer_idx] = dv_accum_tt[consumer_idx].addr;
        }
        if (warpgroup::laneid() == 0) {
            tensor_commit<1>(kv_tmem_ready[consumer_idx]);
        }
    }
    if (is_mma) {
        wait(kv_tmem_ready[0], 0);
        if (laneid() == 0) {
            arrive(dv_ready[0]);
        }
        wait(kv_tmem_ready[1], 0);
        if (laneid() == 0) {
            arrive(dv_ready[1]);
        }
    }
    __syncthreads();

    if (is_dv_relay) {
        const int relay_consumer = is_relay ? 0 : 1;
        wait(dv_ready[relay_consumer], 0);
        const dv_tt dv_accum_relay = dv_tt{dv_tmem_addr[relay_consumer]};
        #pragma unroll
        for (int subtile = 0; subtile < kittens::WARPGROUP_WARPS; ++subtile) {
            rt_fl<kRefTileM, C::Dvo> dv_reg;
            const int kv_subtile_idx =
                (kv_tile_base + relay_consumer) * kittens::WARPGROUP_WARPS + subtile;
            const dv_subtile_tt dv_subtile = dv_accum_relay.template subtile<dv_subtile_tt>(kRefTileM * subtile, 0);
            group<1>::load_async(dv_reg, dv_subtile);
            tensor_load_wait();
            warp::store<dim::DEPTH>(g.dv, dv_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
        }
        if (laneid() == 0) {
            arrive(dv_store_done[relay_consumer]);
        }
    }
    if (is_mma) {
        wait(dv_store_done[0], 0);
        if (laneid() == 0) {
            arrive(dk_ready[0]);
        }
        wait(dv_store_done[1], 0);
        if (laneid() == 0) {
            arrive(dk_ready[1]);
        }
    }
    if (is_dv_relay) {
        const int relay_consumer = is_relay ? 0 : 1;
        wait(dk_ready[relay_consumer], 0);
        const dk_tt dk0_relay = dk_tt{dk0_tmem_addr[relay_consumer]};
        const dk_tt dk1_relay = dk_tt{dk1_tmem_addr[relay_consumer]};
        const dk_tt dk2_relay = dk_tt{dk2_tmem_addr[relay_consumer]};
        #pragma unroll
        for (int subtile = 0; subtile < kittens::WARPGROUP_WARPS; ++subtile) {
            rt_fl<kRefTileM, 64> dk0_reg;
            rt_fl<kRefTileM, 64> dk1_reg;
            rt_fl<kRefTileM, 64> dk2_reg;
            rt_fl<kRefTileM, C::Dqk> dk_full_reg;
            const int kv_subtile_idx =
                (kv_tile_base + relay_consumer) * kittens::WARPGROUP_WARPS + subtile;
            const dk_subtile_tt dk0_subtile = dk0_relay.template subtile<dk_subtile_tt>(kRefTileM * subtile, 0);
            const dk_subtile_tt dk1_subtile = dk1_relay.template subtile<dk_subtile_tt>(kRefTileM * subtile, 0);
            const dk_subtile_tt dk2_subtile = dk2_relay.template subtile<dk_subtile_tt>(kRefTileM * subtile, 0);
            group<1>::load_async(dk0_reg, dk0_subtile);
            group<1>::load_async(dk1_reg, dk1_subtile);
            group<1>::load_async(dk2_reg, dk2_subtile);
            tensor_load_wait();
            bwd_cute16_kernel::detail::stitch_three_chunks(dk_full_reg, dk0_reg, dk1_reg, dk2_reg);
            if constexpr (repair_diagonal_causal_tile) {
                if (relay_consumer == 0) {
                    warp::add(dk_full_reg, dk_full_reg, dk_fix_full);
                }
            }
            warp::store<dim::DEPTH>(g.dk_full, dk_full_reg, {batch_idx, kv_subtile_idx, head_idx, 0});
        }
    }
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
    bool deterministic
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
    };

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        detail::launch_main<true, C>(g, total_ctas, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    } else {
        detail::launch_main<false, C>(g, total_ctas, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    if constexpr (!bwd_cute16_kernel::kUseDirectDQ) {
        bwd_cute16_kernel::detail::launch_reduce<C>(g, static_cast<int>(q.size(1) / C::TileRows), static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream, causal);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    if (causal) {
        const int total_kv_tiles64 = static_cast<int>(q.size(1) / C::TileRows);
        const int causal_full_repair_tiles64 = total_kv_tiles64 < 4 ? total_kv_tiles64 : 4;
        bwd_cute16_kernel::detail::launch_causal_first_tile_patch<C>(g, causal_full_repair_tiles64, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
        CHECK_CUDA_ERROR(cudaGetLastError());
        const int causal_dk_only_end_tile64 = total_kv_tiles64 < 29 ? total_kv_tiles64 : 29;
        if (causal_full_repair_tiles64 < causal_dk_only_end_tile64) {
            bwd_cute16_kernel::detail::launch_causal_dk_only_patch<C>(
                g,
                causal_dk_only_end_tile64 - causal_full_repair_tiles64,
                static_cast<int>(q.size(2)),
                static_cast<int>(q.size(0)),
                stream,
                causal_full_repair_tiles64
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
        bwd_cute16_kernel::detail::launch_causal_dq_diagonal_patch<C>(g, 1, static_cast<int>(q.size(2)), static_cast<int>(q.size(0)), stream);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

}  // namespace tkfa4::bwd_cute16_kernel_candidate2

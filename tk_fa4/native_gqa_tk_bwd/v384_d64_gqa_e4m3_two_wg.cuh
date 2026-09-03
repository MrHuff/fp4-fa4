#pragma once

#include "native_gqa_tk_bwd_pipelined.cuh"

// Bounded residency experiment for the completion-correct native TK donor.
//
// This file intentionally specializes the donor to its E4M3 BHSD path.  The
// two K64 consumers remain unchanged, but the dedicated producer warpgroup is
// removed.  WG0 warp 0 issues the next-stage TMA loads and publishes dQ after
// both consumer warpgroups have completed the current query tile.
//
// Two independent changes are required for real two-CTA residency on SM100:
//   * 256 threads and a <=128-register launch bound;
//   * tensor_allocator<2, 1>, which gives each resident CTA 256 TMEM columns.
//
// The donor consumed all 512 TMEM columns.  This experiment fits in 256 by
// keeping one score/dP pair (columns 0/64), persistent dK/dV accumulators
// (128/192), and overlaying dQ on the dead score columns after P and dS have
// been published to shared memory.
namespace tkfa4::native_gqa_tk_bwd::v384_d64_gqa_e4m3_two_wg {

namespace donor = tkfa4::native_gqa_tk_bwd::pipelined;

constexpr int kDepth = donor::kDepth;
constexpr int kTileHeight = donor::kTileHeight;
constexpr int kConsumerWarpgroups = donor::kConsumerWarpgroups;
constexpr int kNumWarpgroups = kConsumerWarpgroups;
constexpr int kThreads =
    kNumWarpgroups * kittens::WARPGROUP_WARPS * kittens::WARP_THREADS;

// The E4M3 allocator high-water mark is 65 KiB.  Reserving 96 KiB still stays
// below the requested 112 KiB gate and prevents a third CTA from contending
// for a tensor-memory half on SM100 parts with >200 KiB shared memory.
constexpr int kDynamicSmemBytes = 96 * 1024;

using e4m3_globals = donor::e4m3_globals;

template <
    typename AttentionTmem,
    typename GradientTmem,
    typename LowpRegister
>
__device__ __forceinline__ void compute_loop(
    kittens::semaphore *stats_ready,
    kittens::semaphore *q_ready,
    kittens::semaphore *dout_ready,
    kittens::semaphore &score_ready,
    kittens::semaphore &dp_ready,
    kittens::semaphore &kv_step_done,
    rt_fl<16, 64> &score_or_probability,
    rt_fl<16, 64> &dp_or_ds,
    LowpRegister &probability_lowp,
    LowpRegister &ds_lowp,
    AttentionTmem &score_tmem,
    AttentionTmem &dp_tmem,
    GradientTmem &dk_tmem,
    GradientTmem &dv_tmem,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &dout_smem,
    auto &probability_smem,
    auto &ds_smem,
    auto &l_smem,
    auto &delta_smem,
    int query_block,
    int query_start,
    int load_stage,
    const e4m3_globals &g
) {
    const int phase = ((query_block - query_start) / 2) % 2;
    const int consumer = kittens::warpid() / kittens::WARPGROUP_WARPS;

    wait(stats_ready[load_stage], phase);
    donor::stream_tile<e4m3_globals::kStatsScale>(
        score_or_probability,
        l_smem,
        load_stage
    );
    wait(q_ready[load_stage], phase);
    warpgroup::mm_ABt(
        score_tmem,
        k_smem[consumer],
        q_smem[load_stage],
        score_ready
    );
    wait(score_ready, phase);
    tensor_after_thread_sync();

    // dP is not live until the second MMA, so use its register fragment to
    // materialize QK and immediately fold it into the row statistic.
    donor::load_half_tmem_async(dp_or_ds, score_tmem);
    tensor_load_wait();
    warp::add(
        score_or_probability,
        score_or_probability,
        dp_or_ds
    );

    wait(dout_ready[load_stage], phase);
    warpgroup::mm_ABt(
        dp_tmem,
        v_smem[consumer],
        dout_smem[load_stage],
        dp_ready
    );
    wait(dp_ready, phase);
    tensor_after_thread_sync();
    donor::load_half_tmem_async(dp_or_ds, dp_tmem);
    tensor_load_wait();

    warp::mul(score_or_probability, score_or_probability, g.scale_log2e);
    donor::causal_mask(score_or_probability, query_block);
    warp::exp2(score_or_probability, score_or_probability);
    donor::stream_sub_tile<e4m3_globals::kDeltaScale>(
        dp_or_ds,
        delta_smem,
        load_stage
    );

    // P remains in score_or_probability while centered dP is overwritten by
    // dS.  Convert both only after dS has consumed the unscaled P value.
    warp::mul(
        dp_or_ds,
        score_or_probability,
        dp_or_ds
    );
    warp::mul(dp_or_ds, dp_or_ds, g.scale * 256.0f);
    warp::mul(
        score_or_probability,
        score_or_probability,
        256.0f
    );
    donor::convert_f32_to_e4m3(probability_lowp, score_or_probability);
    donor::convert_f32_to_e4m3(ds_lowp, dp_or_ds);

    warpgroup::store(probability_smem[consumer], probability_lowp);
    warpgroup::store(ds_smem[consumer], ds_lowp);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    warpgroup::sync(12 + consumer);

    if (query_block == query_start) {
        donor::fp8_mm_ab_corrected<0>(
            dv_tmem,
            probability_smem[consumer],
            dout_smem[load_stage]
        );
        donor::fp8_mm_ab_corrected<0>(
            dk_tmem,
            ds_smem[consumer],
            q_smem[load_stage]
        );
    } else {
        donor::fp8_mm_ab_corrected<1>(
            dv_tmem,
            probability_smem[consumer],
            dout_smem[load_stage]
        );
        donor::fp8_mm_ab_corrected<1>(
            dk_tmem,
            ds_smem[consumer],
            q_smem[load_stage]
        );
    }
    if (warpgroup::laneid() == 0) {
        tensor_commit<1>(kv_step_done);
    }
    wait(kv_step_done, phase);
    tensor_after_thread_sync();
    group<8>::sync(10);
}

template <typename QTile, typename DoutTile, typename LTile>
__device__ __forceinline__ void prefetch_next_stage(
    typename e4m3_globals::q_gl const &q,
    typename e4m3_globals::dout_gl const &dout,
    typename e4m3_globals::l_gl const &l_aux,
    typename e4m3_globals::delta_gl const &delta,
    QTile (&q_smem)[2],
    DoutTile (&dout_smem)[2],
    LTile (&l_smem)[2],
    LTile (&delta_smem)[2],
    kittens::semaphore *q_ready,
    kittens::semaphore *dout_ready,
    kittens::semaphore *stats_ready,
    int query_block,
    int query_blocks,
    int next_stage
) {
    if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0 &&
        query_block + 1 < query_blocks) {
        const coord<QTile> coordinate = {
            static_cast<int>(blockIdx.z),
            static_cast<int>(blockIdx.y),
            query_block + 1,
            0,
        };
        warp::tma::expect_bytes(q_ready[next_stage], sizeof(q_smem[0]));
        warp::tma::load_async(
            q_smem[next_stage],
            q,
            coordinate,
            q_ready[next_stage]
        );
        warp::tma::expect_bytes(
            dout_ready[next_stage],
            sizeof(dout_smem[0])
        );
        warp::tma::load_async(
            dout_smem[next_stage],
            dout,
            coordinate,
            dout_ready[next_stage]
        );
        const coord<LTile> stats_coordinate = {
            static_cast<int>(blockIdx.z),
            static_cast<int>(blockIdx.y),
            0,
            query_block + 1,
        };
        warp::tma::expect_bytes(
            stats_ready[next_stage],
            sizeof(l_smem[0]) + sizeof(delta_smem[0])
        );
        warp::tma::load_async(
            l_smem[next_stage],
            l_aux,
            stats_coordinate,
            stats_ready[next_stage]
        );
        warp::tma::load_async(
            delta_smem[next_stage],
            delta,
            stats_coordinate,
            stats_ready[next_stage]
        );
    }
    // All WG0 warps must reach the compute path together after its elected
    // warp has issued the asynchronous producer work.
    group<4>::sync(8);
}

template <typename DqTile>
__device__ __forceinline__ void publish_dq(
    typename e4m3_globals::dq_gl const &dq_global,
    DqTile &dq_smem,
    int query_block
) {
    if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
        const coord<DqTile> coordinate = {
            static_cast<int>(blockIdx.z),
            static_cast<int>(blockIdx.y),
            query_block,
            0,
        };
        warp::tma::store_add_async(dq_global, dq_smem, coordinate);
    }
    // The elected warp drains this store immediately before dq_smem is reused
    // by the next query tile.  Keep all four WG0 warps aligned after issue.
    group<4>::sync(9);
}

__device__ __forceinline__ void complete_previous_dq_store(
    kittens::semaphore &dq_ready,
    bool has_previous_store
) {
    if (has_previous_store &&
        kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
        warp::tma::store_async_wait();
        if (kittens::laneid() == 0) {
            arrive(dq_ready);
        }
    }
    // No thread may overwrite dq_smem until the elected warp has completed
    // the previous asynchronous TMA store and advanced its parity semaphore.
    group<4>::sync(9);
}

__global__ __launch_bounds__(kThreads, 2)
void main_kernel(const __grid_constant__ e4m3_globals g) {
    extern __shared__ int dynamic_shared[];
    tma_swizzle_allocator allocator(dynamic_shared);

    using k_tile = e4m3_globals::k_tile;
    using v_tile = e4m3_globals::v_tile;
    using q_tile = e4m3_globals::q_tile;
    using dout_tile = e4m3_globals::dout_tile;
    using dq_tile = e4m3_globals::dq_tile;
    using dk_tile = e4m3_globals::dk_tile;
    using dv_tile = e4m3_globals::dv_tile;
    using l_tile = e4m3_globals::l_tile;
    using delta_tile = e4m3_globals::delta_tile;
    using probability_tile = e4m3_globals::probability_tile;
    using ds_tile = e4m3_globals::ds_tile;
    using lowp_register = e4m3_globals::lowp_register;
    using attention_tmem = half_tt_fl<kTileHeight>;
    using gradient_tmem = half_tt_fl<kDepth>;

    k_tile (&k_smem)[kConsumerWarpgroups] =
        allocator.allocate<k_tile, kConsumerWarpgroups>();
    v_tile (&v_smem)[kConsumerWarpgroups] =
        allocator.allocate<v_tile, kConsumerWarpgroups>();
    q_tile (&q_smem)[2] = allocator.allocate<q_tile, 2>();
    dout_tile (&dout_smem)[2] = allocator.allocate<dout_tile, 2>();
    dq_tile (&dq_smem) = allocator.allocate<dq_tile>();
    l_tile (&l_smem)[2] = allocator.allocate<l_tile, 2>();
    delta_tile (&delta_smem)[2] = allocator.allocate<delta_tile, 2>();

    // Four FP32 epilogue tiles alias the dead input/staging arena.  Their
    // 64-KiB span is the shared-memory high-water mark before the small P/dS
    // publication tiles allocated below.
    dk_tile *dk_smem = reinterpret_cast<dk_tile *>(&k_smem[0].data[0]);
    dv_tile *dv_smem = reinterpret_cast<dv_tile *>(
        dk_smem + kConsumerWarpgroups
    );
    ds_tile (&ds_smem)[kConsumerWarpgroups] =
        allocator.allocate<ds_tile, kConsumerWarpgroups>();
    probability_tile (&probability_smem)[kConsumerWarpgroups] =
        allocator.allocate<probability_tile, kConsumerWarpgroups>();

    tensor_allocator<2, 1> tmem_allocator{};
    static_assert(tensor_allocator<2, 1>::cols == 256);
    attention_tmem score_tmem[kConsumerWarpgroups];
    attention_tmem dp_tmem[kConsumerWarpgroups];
    gradient_tmem dk_tmem[kConsumerWarpgroups];
    gradient_tmem dv_tmem[kConsumerWarpgroups];

#pragma unroll
    for (int consumer = 0; consumer < kConsumerWarpgroups; ++consumer) {
        score_tmem[consumer] =
            tmem_allocator.template allocate<attention_tmem>(consumer, 0);
        dp_tmem[consumer] = tmem_allocator.template allocate<attention_tmem>(
            consumer,
            kTileHeight
        );
        dk_tmem[consumer] = tmem_allocator.template allocate<gradient_tmem>(
            consumer,
            2 * kTileHeight
        );
        dv_tmem[consumer] = tmem_allocator.template allocate<gradient_tmem>(
            consumer,
            2 * kTileHeight + kDepth
        );
    }
    // dQ starts only after both score and dP fragments have reached shared
    // memory, so it may reuse score's physical columns in superlane zero.
    gradient_tmem dq_tmem =
        tmem_allocator.template allocate<gradient_tmem>(0, 0);

    const int warp = kittens::warpid();
    const int warpgroup = warp / kittens::WARPGROUP_WARPS;
    const int query_blocks = g.sequence / kTileHeight;
    const int kv_head = static_cast<int>(blockIdx.y) / g.head_ratio;
    const int query_start = static_cast<int>(blockIdx.x) * 2;
    int load_stage = 0;
    int next_stage = 1;

    __shared__ kittens::semaphore kv_ready;
    __shared__ kittens::semaphore q_ready[2];
    __shared__ kittens::semaphore dout_ready[2];
    __shared__ kittens::semaphore stats_ready[2];
    __shared__ kittens::semaphore dq_ready;
    __shared__ kittens::semaphore score_done[kConsumerWarpgroups][2];
    __shared__ kittens::semaphore dp_done[kConsumerWarpgroups][2];
    __shared__ kittens::semaphore kv_step_done[kConsumerWarpgroups][2];
    __shared__ kittens::semaphore dq_tmem_done[2];
    __shared__ kittens::semaphore kv_tmem_done[kConsumerWarpgroups];

    if (threadIdx.x == 0) {
        init_semaphore(kv_ready, 0, 1);
        init_semaphore(dq_ready, 1, 0);
#pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(q_ready[stage], 0, 1);
            init_semaphore(dout_ready[stage], 0, 1);
            init_semaphore(stats_ready[stage], 0, 1);
            init_semaphore(dq_tmem_done[stage], 0, 1);
#pragma unroll
            for (int consumer = 0; consumer < kConsumerWarpgroups; ++consumer) {
                init_semaphore(score_done[consumer][stage], 0, 1);
                init_semaphore(dp_done[consumer][stage], 0, 1);
                init_semaphore(kv_step_done[consumer][stage], 0, 1);
            }
        }
#pragma unroll
        for (int consumer = 0; consumer < kConsumerWarpgroups; ++consumer) {
            init_semaphore(kv_tmem_done[consumer], 0, 1);
        }

        tma::expect_bytes(
            kv_ready,
            (sizeof(k_smem[0]) + sizeof(v_smem[0])) *
                kConsumerWarpgroups
        );
#pragma unroll
        for (int consumer = 0; consumer < kConsumerWarpgroups; ++consumer) {
            const coord<k_tile> coordinate = {
                static_cast<int>(blockIdx.z),
                kv_head,
                static_cast<int>(blockIdx.x) * kConsumerWarpgroups + consumer,
                0,
            };
            tma::load_async(k_smem[consumer], g.k, coordinate, kv_ready);
            tma::load_async(v_smem[consumer], g.v, coordinate, kv_ready);
        }

        const coord<q_tile> q_coordinate = {
            static_cast<int>(blockIdx.z),
            static_cast<int>(blockIdx.y),
            query_start,
            0,
        };
        tma::expect_bytes(q_ready[load_stage], sizeof(q_smem[0]));
        tma::load_async(
            q_smem[load_stage],
            g.q,
            q_coordinate,
            q_ready[load_stage]
        );
        tma::expect_bytes(
            dout_ready[load_stage],
            sizeof(dout_smem[0])
        );
        tma::load_async(
            dout_smem[load_stage],
            g.dout,
            q_coordinate,
            dout_ready[load_stage]
        );
        const coord<l_tile> stats_coordinate = {
            static_cast<int>(blockIdx.z),
            static_cast<int>(blockIdx.y),
            0,
            query_start,
        };
        tma::expect_bytes(
            stats_ready[load_stage],
            sizeof(l_smem[0]) + sizeof(delta_smem[0])
        );
        tma::load_async(
            l_smem[load_stage],
            g.l_aux,
            stats_coordinate,
            stats_ready[load_stage]
        );
        tma::load_async(
            delta_smem[load_stage],
            g.delta,
            stats_coordinate,
            stats_ready[load_stage]
        );
    }
    __syncthreads();

    rt_fl<16, 64> score_or_probability;
    rt_fl<16, 64> dp_or_ds;
    lowp_register probability_lowp;
    lowp_register ds_lowp;

    wait(kv_ready, 0);
    for (
        int query_block = query_start;
        query_block < query_blocks;
        ++query_block, load_stage ^= 1, next_stage ^= 1
    ) {
        if (warpgroup == 0) {
            prefetch_next_stage(
                g.q,
                g.dout,
                g.l_aux,
                g.delta,
                q_smem,
                dout_smem,
                l_smem,
                delta_smem,
                q_ready,
                dout_ready,
                stats_ready,
                query_block,
                query_blocks,
                next_stage
            );
        }

        compute_loop(
            stats_ready,
            q_ready,
            dout_ready,
            score_done[warpgroup][load_stage],
            dp_done[warpgroup][load_stage],
            kv_step_done[warpgroup][load_stage],
            score_or_probability,
            dp_or_ds,
            probability_lowp,
            ds_lowp,
            score_tmem[warpgroup],
            dp_tmem[warpgroup],
            dk_tmem[warpgroup],
            dv_tmem[warpgroup],
            q_smem,
            k_smem,
            v_smem,
            dout_smem,
            probability_smem,
            ds_smem,
            l_smem,
            delta_smem,
            query_block,
            query_start,
            load_stage,
            g
        );

        if (warpgroup == 0) {
            const int completion_phase =
                ((query_block - query_start) / 2) % 2;
            donor::fp8_mm_atb_corrected<0>(
                dq_tmem,
                ds_smem[0],
                k_smem[0]
            );
            donor::fp8_mm_atb_corrected<1>(
                dq_tmem,
                ds_smem[1],
                k_smem[1]
            );
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(dq_tmem_done[load_stage]);
            }
            wait(dq_tmem_done[load_stage], completion_phase);
            tensor_after_thread_sync();

            // WG1 cannot overwrite ds_smem[1] until dQ has consumed it.
            group<8>::sync(11);

            rt_fl<16, kDepth> dq_fragment;
            donor::load_half_tmem_async(dq_fragment, dq_tmem);
            complete_previous_dq_store(
                dq_ready,
                query_block != query_start
            );
            wait(dq_ready, next_stage);
            tensor_load_wait();
            warp::mul(dq_fragment, dq_fragment, 1.0f / 256.0f);
            warpgroup::store(dq_smem, dq_fragment);
            group<4>::sync(warpgroup::groupid() + 4);
            publish_dq(g.dq, dq_smem, query_block);
        } else {
            group<8>::sync(11);
        }
    }

    if (warpgroup == 0) {
        // Drain the final publication before the dK/dV epilogue aliases the
        // input/dQ shared arena.  This is the final dq_ready arrival, matching
        // the donor's one-arrival-per-query phase sequence exactly.
        complete_previous_dq_store(dq_ready, true);
    }

    rt_fl<16, kDepth> dk_fragment;
    rt_fl<16, kDepth> dv_fragment;
    if (warpgroup::laneid() == 0) {
        tensor_commit<1>(kv_tmem_done[warpgroup]);
    }
    wait(kv_tmem_done[warpgroup], 0);
    tensor_after_thread_sync();
    donor::load_half_tmem_async(dk_fragment, dk_tmem[warpgroup]);
    donor::load_half_tmem_async(dv_fragment, dv_tmem[warpgroup]);
    tensor_load_wait();
    donor::store_kv<e4m3_globals, dk_tile, dv_tile>(
        dk_smem,
        dk_fragment,
        dv_smem,
        dv_fragment,
        g,
        dq_ready,
        kv_head,
        next_stage
    );
}

inline void launch_e4m3(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream
) {
    const float kernel_scale = scale / 16.0f;
    const e4m3_globals g{
        kittens::py::tensor_to_gl<e4m3_globals::q_gl>(q),
        kittens::py::tensor_to_gl<e4m3_globals::k_gl>(k),
        kittens::py::tensor_to_gl<e4m3_globals::v_gl>(v),
        kittens::py::tensor_to_gl<e4m3_globals::dout_gl>(dout),
        kittens::py::tensor_to_gl<e4m3_globals::dq_gl>(dq),
        kittens::py::tensor_to_gl<e4m3_globals::dk_gl>(dk),
        kittens::py::tensor_to_gl<e4m3_globals::dv_gl>(dv),
        kittens::py::tensor_to_gl<e4m3_globals::l_gl>(
            l_aux,
            q.size(0),
            q.size(1),
            1,
            q.size(2)
        ),
        kittens::py::tensor_to_gl<e4m3_globals::delta_gl>(
            delta,
            q.size(0),
            q.size(1),
            1,
            q.size(2)
        ),
        kernel_scale,
        kernel_scale * kLog2E,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(1) / k.size(1)),
    };
    const dim3 grid(
        static_cast<unsigned int>(q.size(2) / 128),
        static_cast<unsigned int>(q.size(1)),
        static_cast<unsigned int>(q.size(0))
    );
    CUDACHECK(cudaFuncSetAttribute(
        main_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSmemBytes
    ));
    main_kernel<<<grid, kThreads, kDynamicSmemBytes, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v384_d64_gqa_e4m3_two_wg

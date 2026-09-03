#pragma once

// Persistent SM100 projection-backward consumer for head-complete BF16 dQ
// slices or a small hierarchy of BF16 dQ partials.
//
// Attention has many K/V-owner CTAs contribute to each 128-row dQ tile.  A
// producer CTA increments a per-head tile counter only after its BF16 TMA
// reductions retire.  This consumer occupies a bounded number of two-CTA
// clusters, leaving enough SMs to advance the attention grid.  Rather than
// waiting for every head before issuing the GEMM, it waits immediately before
// each head's K64 slices and accumulates those slices into one output tile.
// This folds head readiness into the projection reduction loop without extra
// tensor work, output atomics, or attention-side TMEM allocation.  Two private
// projection accumulators overlap adjacent output MMA and epilogue work.  In
// the ordinary specialization the BF16 scratch contains the completed
// cross-owner sum.  The hierarchical specialization instead gives each owner
// one of a small number of reduction lanes.  Projection consumes every lane
// into the same FP32 TMEM accumulator, so no completed standalone BF16 dQ
// tensor is ever published.  This is deliberately a separate specialization:
// its larger input stage must not reduce the load depth of the retained path.
namespace tkfa4_dq_projection {

constexpr int kTileRows = 128;

template <
    int _LOAD_PIPE_DEPTH = 4,
    int _MAX_CLUSTERS = 64,
    int _REDUCTION_LANES = 1
>
struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = false;
    static constexpr int NUM_THREADS = 256;
    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int MAX_CLUSTERS = _MAX_CLUSTERS;
    static constexpr int REDUCTION_LANES = _REDUCTION_LANES;
    static_assert(REDUCTION_LANES == 1 || REDUCTION_LANES == 2);
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 64;
    static constexpr int MMA_PIPE_DEPTH = 2;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH;
};

template <typename C>
struct globals {
    using A_tile = kittens::st_bf<C::Mb / 2, C::Kb>;
    using B_tile = kittens::st_bf<C::Nb / 2, C::Kb>;
    using D_tile = kittens::st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;
    using A_gl = kittens::gl<kittens::bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = kittens::gl<kittens::bf16, 1, 1, -1, -1, B_tile>;
    using D_gl = kittens::gl<kittens::bf16, 1, 1, -1, -1, D_tile>;

    A_gl A;
    B_gl B;
    D_gl D;
    const uint32_t *arrivals;
    int heads;
    int block_begin;
    int block_end;
    int cluster_cap;

    struct inputs_t {
        A_tile A[C::REDUCTION_LANES];
        B_tile B;
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline int cluster_count() const {
        const int column_blocks = D.cols() / C::Nb;
        // Measured overlap sweet spots on GB200: six clusters for the H8
        // projection and sixteen for H24.  Larger grids delay the remaining
        // attention owners more than they shorten the projection tail.
        int cap = cluster_cap > 0
            ? cluster_cap
            : (C::REDUCTION_LANES == 2
                ? (D.cols() <= 1024 ? 8 : 16)
                : (D.cols() <= 1024 ? 6 : 16));
        cap = min(cap, C::MAX_CLUSTERS);
        if (const char *value = std::getenv(
                "TK_FA4_DQ_PROJECTION_CLUSTERS"
            )) {
            const int requested = std::atoi(value);
            if (requested > 0) {
                cap = min(requested, C::MAX_CLUSTERS);
            }
        }
        // Columns alone expose too little parallelism for H8: four clusters
        // then serialize all 32 row pairs.  Consecutive persistent clusters
        // may safely own different rows of the same output-column block.
        return min(block_end - block_begin, cap);
    }
    __host__ inline dim3 grid() const {
        return dim3(cluster_count() * C::CLUSTER_SIZE);
    }
    __host__ inline dim3 block() const {
        return dim3(C::NUM_THREADS);
    }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int bytes = sizeof(inputs_t) * C::LOAD_PIPE_DEPTH +
                              sizeof(outputs_t) + 1024;
        static_assert(bytes <= kittens::MAX_SHARED_MEMORY - 4096);
        return bytes;
    }
};

__device__ __forceinline__ void wait_for_tile(
    const uint32_t *counter,
    uint32_t expected
) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> arrival(
        *const_cast<uint32_t *>(counter)
    );
    while (arrival.load(cuda::memory_order_acquire) < expected) {
        __nanosleep(256);
    }
}

template <typename C>
__global__ __launch_bounds__(C::NUM_THREADS, 1) void kernel(
    const __grid_constant__ globals<C> g
) {
    using G = globals<C>;
    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_tile>();
        g.B.template prefetch_tma<typename G::B_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = kittens::warpgroup::groupid();
    const int producer_warp =
        kittens::group<kittens::WARPGROUP_WARPS>::warpid();
    const int cta_id = kittens::cluster_ctarank();
    const int cluster_id = kittens::clusterIdx().x;
    const int cluster_count = gridDim.x / C::CLUSTER_SIZE;
    const int column_blocks = g.D.cols() / C::Nb;
    const int reduction_blocks = g.A.cols() / C::Kb;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    kittens::tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::inputs_t (&inputs)[C::LOAD_PIPE_DEPTH] =
        allocator.template allocate<typename G::inputs_t,
                                    C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &outputs =
        allocator.template allocate<typename G::outputs_t>();

    kittens::tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;
    __shared__ uint32_t tmem_addr;
    __shared__ kittens::semaphore tmem_provisioned;
    __shared__ kittens::semaphore tmem_finished;
    __shared__ kittens::semaphore inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ kittens::semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ kittens::semaphore outputs_arrived[C::MMA_PIPE_DEPTH];
    __shared__ kittens::semaphore outputs_finished[C::MMA_PIPE_DEPTH];
    if (threadIdx.x == 32) {
        kittens::init_semaphore(tmem_provisioned, 0, 1);
        kittens::init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            kittens::init_semaphore(inputs_arrived[i], 0, 1);
            kittens::init_semaphore(inputs_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::MMA_PIPE_DEPTH; ++i) {
            kittens::init_semaphore(outputs_arrived[i], 0, 1);
            kittens::init_semaphore(
                outputs_finished[i],
                0,
                C::CLUSTER_SIZE
            );
        }
    }
    kittens::everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id == 1) {
        if (producer_warp == 3 && kittens::warp::elect_leader()) {
            kittens::everyone::tma::cluster::wait();
            for (int block = g.block_begin + cluster_id;
                 block < g.block_end;
                 block += cluster_count) {
                const int row_block = block / column_blocks;
                const int column_block = block - row_block * column_blocks;
                const int q_tile = row_block * C::CLUSTER_SIZE + cta_id;
                const int q_tiles = g.D.rows() / kTileRows;
                const int reductions_per_head = reduction_blocks / g.heads;
                for (int reduction = 0; reduction < reduction_blocks;
                     ++reduction) {
                    if (reduction % reductions_per_head == 0) {
                        const int head = reduction / reductions_per_head;
                        const uint32_t expected = static_cast<uint32_t>(
                            2 * (row_block + 1)
                        );
                        wait_for_tile(
                            g.arrivals + head * q_tiles + q_tile,
                            expected
                        );
                    }
                    kittens::wait(
                        inputs_finished[stage],
                        kittens::get_phasebit<1>(phasebits, stage)
                    );
                    #pragma unroll
                    for (int reduction_lane = 0;
                         reduction_lane < C::REDUCTION_LANES;
                         ++reduction_lane) {
                        kittens::tma::cluster::load_async(
                            inputs[stage].A[reduction_lane],
                            g.A,
                            {
                                reduction_lane * q_tiles +
                                    row_block * 2 + cta_id,
                                reduction
                            },
                            inputs_arrived[stage],
                            static_cast<uint16_t>(1u << cta_id),
                            0
                        );
                    }
                    kittens::tma::cluster::load_async(
                        inputs[stage].B,
                        g.B,
                        {column_block * 2 + cta_id, reduction},
                        inputs_arrived[stage],
                        static_cast<uint16_t>(1u << cta_id),
                        0
                    );
                    kittens::update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        }
        if (
            cta_id == 0 && producer_warp == 0 &&
            kittens::warp::elect_leader()
        ) {
            kittens::everyone::tma::cluster::wait();
            kittens::wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto output_tm0 = tm_allocator.template allocate<
                kittens::full_tt_fl<C::Nb>
            >(0);
            auto output_tm1 = tm_allocator.template allocate<
                kittens::full_tt_fl<C::Nb>
            >(C::Nb);
            int task = 0;
            for (int block = g.block_begin + cluster_id;
                 block < g.block_end;
                 block += cluster_count, ++task) {
                const int output_slot = task % C::MMA_PIPE_DEPTH;
                kittens::wait(
                    outputs_finished[output_slot],
                    ((task + C::MMA_PIPE_DEPTH) /
                         C::MMA_PIPE_DEPTH) & 1
                );
                kittens::tensor_after_thread_sync();
                auto &output_tm = output_slot == 0
                    ? output_tm0
                    : output_tm1;
                for (int reduction = 0; reduction < reduction_blocks;
                     ++reduction) {
                    kittens::tma::expect_bytes(
                        inputs_arrived[stage],
                        2 * sizeof(typename G::inputs_t)
                    );
                    kittens::wait(
                        inputs_arrived[stage],
                        kittens::get_phasebit<0>(phasebits, stage)
                    );
                    if constexpr (C::REDUCTION_LANES == 1) {
                        if (reduction == 0) {
                            kittens::mm2_ABt(
                                output_tm,
                                inputs[stage].A[0],
                                inputs[stage].B,
                                inputs_finished[stage]
                            );
                        } else {
                            kittens::mma2_ABt(
                                output_tm,
                                inputs[stage].A[0],
                                inputs[stage].B,
                                inputs_finished[stage]
                            );
                        }
                    } else {
                        if (reduction == 0) {
                            kittens::mm2_ABt(
                                output_tm,
                                inputs[stage].A[0],
                                inputs[stage].B
                            );
                        } else {
                            kittens::mma2_ABt(
                                output_tm,
                                inputs[stage].A[0],
                                inputs[stage].B
                            );
                        }
                        kittens::mma2_ABt(
                            output_tm,
                            inputs[stage].A[1],
                            inputs[stage].B,
                            inputs_finished[stage]
                        );
                    }
                    kittens::update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                kittens::tensor_commit<2>(outputs_arrived[output_slot]);
            }
        }
    } else {
        kittens::everyone::tma::cluster::wait_aligned();
        if (kittens::warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            kittens::warp::arrive(tmem_provisioned);
        }
        kittens::wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto output_tm0 = tm_allocator.template allocate<
            kittens::full_tt_fl<C::Nb>
        >(0);
        auto output_tm1 = tm_allocator.template allocate<
            kittens::full_tt_fl<C::Nb>
        >(C::Nb);
        constexpr int kSlice = C::Nb / C::EPI_PIPE_DEPTH;
        using accum_rt = kittens::rt_fl<C::Mb / 8, kSlice>;
        using output_rt = kittens::rt_bf<C::Mb / 8, kSlice>;
        int task = 0;
        for (int block = g.block_begin + cluster_id;
             block < g.block_end;
             block += cluster_count, ++task) {
            const int row_block = block / column_blocks;
            const int column_block = block - row_block * column_blocks;
            const int output_slot = task % C::MMA_PIPE_DEPTH;
            auto &output_tm = output_slot == 0
                ? output_tm0
                : output_tm1;
            kittens::wait(
                outputs_arrived[output_slot],
                (task / C::MMA_PIPE_DEPTH) & 1
            );
            kittens::warpgroup::tma::store_async_read_wait<0>();
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                accum_rt accumulator;
                output_rt registers;
                kittens::warpgroup::load_async(
                    accumulator,
                    output_tm.template subtile<kittens::full_tt_fl<kSlice>>(
                        0,
                        epi * kSlice
                    )
                );
                kittens::tensor_load_wait();
                kittens::tensor_before_thread_sync();
                kittens::warpgroup::sync(1);
                kittens::warp::copy(registers, accumulator);
                kittens::warpgroup::store(outputs.D[epi], registers);
                kittens::warpgroup::sync(1);
                if (epi == C::EPI_PIPE_DEPTH - 1) {
                    kittens::warpgroup::tma::cluster::arrive(
                        outputs_finished[output_slot],
                        0,
                        1
                    );
                }
                kittens::warpgroup::tma::store_async<
                    kittens::dim::ROW,
                    kittens::cache_policy::EVICT_FIRST
                >(
                    g.D,
                    outputs.D[epi],
                    {
                        row_block * 2 + cta_id,
                        column_block * C::EPI_PIPE_DEPTH + epi
                    }
                );
            }
        }
        kittens::warpgroup::sync(1);
        kittens::warpgroup::tma::store_async_read_wait<0>();
        if (kittens::warpgroup::warpid() == 0) {
            if (kittens::warp::elect_leader()) {
                kittens::tma::cluster::arrive(tmem_finished, 1 - cta_id);
            }
            kittens::wait(tmem_finished, 0);
            tm_allocator.deprovision();
        }
    }
}

template <typename C>
inline void launch(const globals<C> &g, cudaStream_t stream) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()
    ));
    kittens::LaunchConfig<true, false> launch_config(
        g.grid(),
        g.block(),
        g.dynamic_shared_memory(),
        stream,
        dim3(C::CLUSTER_SIZE, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
}

} // namespace tkfa4_dq_projection

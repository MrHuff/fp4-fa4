#pragma once

// Query-owner NVFP4 projection topology for SM100.
//
// A conventional two-CTA projection cluster owns one M256 x N256 output
// tile.  Consequently, every N256 tile reloads the same two M128 A tiles.
// This specialization groups several independent two-CTA MMA pairs into one
// larger cluster.  CTA 0 multicasts the even and odd A halves by parity while
// each pair loads a private N256 B tile.  The tensor-core instruction and
// TMEM ownership remain the proven two-CTA form; only the projection-N data
// topology changes.

#include "kittens.cuh"

namespace tkfa4_projection_n_multicast {

template <int _CLUSTER_SIZE, int _LOAD_PIPE_DEPTH = 4>
struct config {
    static_assert(
        _CLUSTER_SIZE == 2 || _CLUSTER_SIZE == 4 ||
            _CLUSTER_SIZE == 8 || _CLUSTER_SIZE == 16,
        "projection-N multicast supports 2/4/8/16 CTA clusters"
    );
    static_assert((_CLUSTER_SIZE & 1) == 0);
    static constexpr int CLUSTER_SIZE = _CLUSTER_SIZE;
    static constexpr int PAIRS = CLUSTER_SIZE / 2;
    static constexpr int NUM_THREADS = 256;
    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 256;
    static constexpr int MMA_PER_TILE = Kb / 64;
    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH;
};

template <typename C>
struct globals {
    using A_tile = kittens::st_fp4e2m1_2<C::Mb / 2, C::Kb / 2>;
    using B_tile = kittens::st_fp4e2m1_2<C::Nb / 2, C::Kb / 2>;
    using A_sc_tile = kittens::st_hf<C::MMA_PER_TILE, 256, false>;
    using B_sc_tile = kittens::st_hf<C::MMA_PER_TILE, 256, false>;
    using D_tile = kittens::st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;
    using A_gl = kittens::gl<kittens::fp4e2m1_2, 1, 1, -1, -1, A_tile>;
    using B_gl = kittens::gl<kittens::fp4e2m1_2, 1, 1, -1, -1, B_tile>;
    using A_sc_gl = kittens::gl<kittens::half, 1, -1, -1, 256, A_sc_tile>;
    using B_sc_gl = kittens::gl<kittens::half, 1, -1, -1, 256, B_sc_tile>;
    using scale_gl = kittens::gl<float, 1, 1, 1, 1>;
    using D_gl = kittens::gl<kittens::bf16, 1, 1, -1, -1, D_tile>;

    A_gl A;
    A_sc_gl A_sc;
    scale_gl A_scale;
    B_gl B;
    B_sc_gl B_sc;
    scale_gl B_scale;
    D_gl D;
    int cluster_cap;

    struct input_tiles_t {
        A_tile A;
        B_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline int task_count() const {
        const int row_blocks = D.rows() / C::Mb;
        const int super_columns = D.cols() / (C::PAIRS * C::Nb);
        return row_blocks * super_columns;
    }

    __host__ inline int cluster_count() const {
        int clusters = min(task_count(), kittens::num_sms() / C::CLUSTER_SIZE);
        if (cluster_cap > 0) {
            clusters = min(clusters, cluster_cap);
        }
        return max(clusters, 1);
    }

    __host__ inline dim3 grid() const {
        return dim3(cluster_count() * 2, C::PAIRS, 1);
    }

    __host__ inline dim3 block() const {
        return dim3(C::NUM_THREADS, 1, 1);
    }

    __host__ inline int dynamic_shared_memory() const {
        constexpr int bytes =
            sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH +
            sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH +
            sizeof(outputs_t) + 1024;
        static_assert(bytes <= kittens::MAX_SHARED_MEMORY - 4096);
        return bytes;
    }
};

template <typename C>
__global__ __launch_bounds__(C::NUM_THREADS, 1) void kernel(
    const __grid_constant__ globals<C> g
) {
    using G = globals<C>;
    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = kittens::warpgroup::groupid();
    const int producer_warp =
        kittens::group<kittens::WARPGROUP_WARPS>::warpid();
    const int cta_id = kittens::cluster_ctarank();
    const int pair_id = cta_id >> 1;
    const int pair_lane = cta_id & 1;
    const int pair_base = cta_id - pair_lane;
    const bool pair_leader = pair_lane == 0;
    const int cluster_id = kittens::clusterIdx().x;
    const int cluster_count = gridDim.x / 2;
    const int row_blocks = g.D.rows() / C::Mb;
    const int super_columns = g.D.cols() / (C::PAIRS * C::Nb);
    const int tasks = row_blocks * super_columns;
    const int reduction_blocks = 2 * g.A.cols() / C::Kb;
    constexpr uint16_t kEvenMask = []() constexpr {
        uint16_t mask = 0;
        for (int cta = 0; cta < C::CLUSTER_SIZE; cta += 2) {
            mask |= static_cast<uint16_t>(1u << cta);
        }
        return mask;
    }();
    constexpr uint16_t kOddMask =
        static_cast<uint16_t>(kEvenMask << 1);

    extern __shared__ int __shm[];
    kittens::tma_swizzle_allocator allocator(reinterpret_cast<int *>(__shm));
    typename G::input_tiles_t (&inputs)[C::LOAD_PIPE_DEPTH] =
        allocator.template allocate<
            typename G::input_tiles_t,
            C::LOAD_PIPE_DEPTH
        >();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        allocator.template allocate<
            typename G::input_scales_t,
            C::LOAD_PIPE_DEPTH
        >();
    typename G::outputs_t &outputs =
        allocator.template allocate<typename G::outputs_t>();

    kittens::tensor_allocator<1, 2, false> tm_allocator;
    __shared__ uint32_t tmem_addr;
    __shared__ kittens::semaphore tmem_provisioned;
    __shared__ kittens::semaphore inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ kittens::semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ kittens::semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    // CTA 0 owns each A multicast.  A stage may therefore be reused only
    // after every independent MMA pair has released its local copy, not just
    // after pair 0 has released it.  Keep tiles and scales separate because
    // their producer warps advance independently.
    __shared__ kittens::semaphore tile_multicast_reusable[
        C::LOAD_PIPE_DEPTH
    ];
    __shared__ kittens::semaphore scale_multicast_reusable[
        C::LOAD_PIPE_DEPTH
    ];
    __shared__ kittens::semaphore outputs_arrived;
    __shared__ kittens::semaphore outputs_finished;
    if (threadIdx.x == 32) {
        kittens::init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < C::LOAD_PIPE_DEPTH; ++stage) {
            kittens::init_semaphore(inputs_arrived[stage], 0, 1);
            kittens::init_semaphore(scales_arrived[stage], 0, 1);
            kittens::init_semaphore(inputs_finished[stage], 0, 1);
            kittens::init_semaphore(
                tile_multicast_reusable[stage], C::PAIRS
            );
            kittens::init_semaphore(
                scale_multicast_reusable[stage], C::PAIRS
            );
        }
        kittens::init_semaphore(outputs_arrived, 0, 1);
        kittens::init_semaphore(outputs_finished, 0, 2);
    }
    __syncthreads();
    if (threadIdx.x == 0 && pair_leader) {
        #pragma unroll
        for (int stage = 0; stage < C::LOAD_PIPE_DEPTH; ++stage) {
            kittens::tma::cluster::expect_bytes(
                inputs_arrived[stage],
                2 * sizeof(typename G::input_tiles_t)
            );
            kittens::tma::cluster::expect_bytes(
                scales_arrived[stage],
                2 * sizeof(typename G::input_scales_t)
            );
        }
    }
    __syncthreads();
    kittens::everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id == 1) {
        if (producer_warp == 3 && kittens::warp::elect_leader()) {
            kittens::everyone::tma::cluster::wait();
            uint32_t stage = 0;
            uint32_t phasebits = 0xFFFF0000;
            uint32_t multicast_phasebits = 0;
            for (int task = cluster_id; task < tasks; task += cluster_count) {
                const int row_block = task / super_columns;
                const int super_column = task - row_block * super_columns;
                for (int reduction = 0; reduction < reduction_blocks;
                     ++reduction) {
                    kittens::wait(
                        inputs_finished[stage],
                        kittens::get_phasebit<1>(phasebits, stage)
                    );
                    if (pair_leader) {
                        kittens::warp::tma::cluster::arrive(
                            tile_multicast_reusable[stage], 0, 1
                        );
                    }
                    if (cta_id == 0) {
                        kittens::tma::cluster::wait(
                            tile_multicast_reusable[stage],
                            kittens::get_phasebit<0>(
                                multicast_phasebits, stage
                            )
                        );
                        kittens::tma::cluster::load_async(
                            inputs[stage].A,
                            g.A,
                            {row_block * 2, reduction},
                            inputs_arrived[stage],
                            kEvenMask,
                            0
                        );
                        kittens::tma::cluster::load_async(
                            inputs[stage].A,
                            g.A,
                            {row_block * 2 + 1, reduction},
                            inputs_arrived[stage],
                            kOddMask,
                            0
                        );
                        kittens::update_phasebit<0>(
                            multicast_phasebits, stage
                        );
                    }
                    const int b_tile =
                        super_column * C::CLUSTER_SIZE + cta_id;
                    kittens::tma::cluster::load_async(
                        inputs[stage].B,
                        g.B,
                        {b_tile, reduction},
                        inputs_arrived[stage],
                        static_cast<uint16_t>(1u << cta_id),
                        pair_base
                    );
                    kittens::update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (producer_warp == 2 && kittens::warp::elect_leader()) {
            kittens::everyone::tma::cluster::wait();
            uint32_t stage = 0;
            uint32_t phasebits = 0xFFFF0000;
            uint32_t multicast_phasebits = 0;
            for (int task = cluster_id; task < tasks; task += cluster_count) {
                const int row_block = task / super_columns;
                const int super_column = task - row_block * super_columns;
                for (int reduction = 0; reduction < reduction_blocks;
                     ++reduction) {
                    kittens::wait(
                        inputs_finished[stage],
                        kittens::get_phasebit<1>(phasebits, stage)
                    );
                    if (pair_leader) {
                        kittens::warp::tma::cluster::arrive(
                            scale_multicast_reusable[stage], 0, 1
                        );
                    }
                    if (cta_id == 0) {
                        kittens::tma::cluster::wait(
                            scale_multicast_reusable[stage],
                            kittens::get_phasebit<0>(
                                multicast_phasebits, stage
                            )
                        );
                        kittens::tma::cluster::load_async(
                            input_scales[stage].A,
                            g.A_sc,
                            {row_block * 2, reduction, 0},
                            scales_arrived[stage],
                            kEvenMask,
                            0
                        );
                        kittens::tma::cluster::load_async(
                            input_scales[stage].A,
                            g.A_sc,
                            {row_block * 2 + 1, reduction, 0},
                            scales_arrived[stage],
                            kOddMask,
                            0
                        );
                        kittens::update_phasebit<0>(
                            multicast_phasebits, stage
                        );
                    }
                    const int b_tile =
                        super_column * C::CLUSTER_SIZE + cta_id;
                    const uint16_t pair_mask = static_cast<uint16_t>(
                        0x3u << pair_base
                    );
                    kittens::tma::cluster::load_async(
                        input_scales[stage].B[pair_lane],
                        g.B_sc,
                        {b_tile, reduction, 0},
                        scales_arrived[stage],
                        pair_mask,
                        pair_base
                    );
                    kittens::update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (producer_warp == 0) {
            kittens::everyone::tma::cluster::wait_aligned();
            kittens::wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            if (pair_leader && kittens::warp::elect_leader()) {
                auto output_tm = tm_allocator.template allocate<
                    kittens::full_tt_fl<C::Nb>
                >(0);
                auto A_sc_tm = tm_allocator.template allocate<
                    kittens::full_tt_fp8e4m3<
                        16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH
                    >
                >(256);
                auto B_sc_tm = tm_allocator.template allocate<
                    kittens::full_tt_fp8e4m3<
                        32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH
                    >
                >(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                uint32_t stage = 0;
                uint32_t phasebits = 0;
                uint32_t output_phasebits = 0xFFFF0000;
                for (int task = cluster_id; task < tasks;
                     task += cluster_count) {
                    kittens::tma::cluster::wait(
                        outputs_finished,
                        kittens::get_phasebit<1>(output_phasebits, 0)
                    );
                    kittens::tensor_after_thread_sync();
                    for (int reduction = 0;
                         reduction < reduction_blocks;
                         ++reduction) {
                        kittens::wait(
                            inputs_arrived[stage],
                            kittens::get_phasebit<0>(phasebits, stage)
                        );
                        kittens::wait(
                            scales_arrived[stage],
                            kittens::get_phasebit<0>(phasebits, stage)
                        );
                        kittens::tma::cluster::expect_bytes(
                            inputs_arrived[stage],
                            2 * sizeof(typename G::input_tiles_t)
                        );
                        kittens::tma::cluster::expect_bytes(
                            scales_arrived[stage],
                            2 * sizeof(typename G::input_scales_t)
                        );
                        #pragma unroll
                        for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                            auto A_sc_subtile = A_sc_tm.template subtile<
                                kittens::full_tt_fp8e4m3<16>
                            >(
                                stage * C::MMA_PER_TILE * 16 + ii * 16
                            );
                            auto &A_sc_shared = *reinterpret_cast<
                                kittens::st_fp8e4m3<32, 16, false> *
                            >(
                                reinterpret_cast<uint64_t>(
                                    &input_scales[stage].A.data[0]
                                ) + 16 * 32 * ii
                            );
                            load_mxnv_scale_async2(
                                A_sc_subtile,
                                A_sc_shared
                            );
                            auto B_sc_subtile0 = B_sc_tm.template subtile<
                                kittens::full_tt_fp8e4m3<16>
                            >(
                                stage * C::MMA_PER_TILE * 32 +
                                ii * C::B_SC_SIZE * 16
                            );
                            auto &B_sc_shared0 = *reinterpret_cast<
                                kittens::st_fp8e4m3<32, 16, false> *
                            >(
                                reinterpret_cast<uint64_t>(
                                    &input_scales[stage].B[0].data[0]
                                ) + 16 * 32 * ii
                            );
                            load_mxnv_scale_async2(
                                B_sc_subtile0,
                                B_sc_shared0
                            );
                            auto B_sc_subtile1 = B_sc_tm.template subtile<
                                kittens::full_tt_fp8e4m3<16>
                            >(
                                stage * C::MMA_PER_TILE * 32 +
                                ii * C::B_SC_SIZE * 16 + 16
                            );
                            auto &B_sc_shared1 = *reinterpret_cast<
                                kittens::st_fp8e4m3<32, 16, false> *
                            >(
                                reinterpret_cast<uint64_t>(
                                    &input_scales[stage].B[1].data[0]
                                ) + 16 * 32 * ii
                            );
                            load_mxnv_scale_async2(
                                B_sc_subtile1,
                                B_sc_shared1
                            );
                        }
                        auto A_sc_stage = A_sc_tm.template subtile<
                            kittens::full_tt_fp8e4m3<
                                C::MMA_PER_TILE * 16
                            >
                        >(stage * C::MMA_PER_TILE * 16);
                        auto B_sc_stage = B_sc_tm.template subtile<
                            kittens::full_tt_fp8e4m3<
                                C::MMA_PER_TILE * 32
                            >
                        >(stage * C::MMA_PER_TILE * 32);
                        if (reduction == 0) {
                            kittens::mm2_ABt(
                                output_tm,
                                inputs[stage].A,
                                inputs[stage].B,
                                A_sc_stage,
                                B_sc_stage
                            );
                        } else {
                            kittens::mma2_ABt(
                                output_tm,
                                inputs[stage].A,
                                inputs[stage].B,
                                A_sc_stage,
                                B_sc_stage
                            );
                        }
                        kittens::tensor_commit<2>(
                            inputs_finished[stage],
                            static_cast<uint16_t>(0x3u << pair_base)
                        );
                        kittens::update_phasebit<0>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                    kittens::tensor_commit<2>(
                        outputs_arrived,
                        static_cast<uint16_t>(0x3u << pair_base)
                    );
                    kittens::update_phasebit<1>(output_phasebits, 0);
                }
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
        auto output_tm = tm_allocator.template allocate<
            kittens::full_tt_fl<C::Nb>
        >(0);
        constexpr int kSlice = C::Nb / C::EPI_PIPE_DEPTH;
        using accum_rt = kittens::rt_fl<C::Mb / 8, kSlice>;
        using output_rt = kittens::rt_bf<C::Mb / 8, kSlice>;
        const float global_scale = g.A_scale[{0}] * g.B_scale[{0}];
        uint32_t output_phasebits = 0;

        for (int task = cluster_id; task < tasks; task += cluster_count) {
            const int row_block = task / super_columns;
            const int super_column = task - row_block * super_columns;
            const int column_block = super_column * C::PAIRS + pair_id;
            kittens::wait(
                outputs_arrived,
                kittens::get_phasebit<0>(output_phasebits, 0)
            );
            kittens::warpgroup::tma::store_async_read_wait<0>();
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                accum_rt accumulator;
                output_rt registers;
                kittens::warpgroup::load_async(
                    accumulator,
                    output_tm.template subtile<
                        kittens::full_tt_fl<kSlice>
                    >(0, epi * kSlice)
                );
                kittens::tensor_load_wait();
                kittens::tensor_before_thread_sync();
                kittens::warpgroup::sync(1);
                kittens::warp::mul(
                    accumulator,
                    accumulator,
                    global_scale
                );
                kittens::warp::copy(registers, accumulator);
                kittens::warpgroup::store(outputs.D[epi], registers);
                kittens::warpgroup::sync(1);
                if (epi == C::EPI_PIPE_DEPTH - 1) {
                    kittens::warpgroup::tma::cluster::arrive(
                        outputs_finished,
                        pair_base,
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
                        row_block * 2 + pair_lane,
                        column_block * C::EPI_PIPE_DEPTH + epi
                    }
                );
            }
            kittens::update_phasebit<0>(output_phasebits, 0);
        }
        kittens::warpgroup::sync(1);
        kittens::warpgroup::tma::store_async_read_wait<0>();
        if (kittens::warpgroup::warpid() == 0) {
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
    if constexpr (C::CLUSTER_SIZE > 8) {
        CUDACHECK(cudaFuncSetAttribute(
            kernel<C>,
            cudaFuncAttributeNonPortableClusterSizeAllowed,
            1
        ));
    }
    kittens::LaunchConfig<true, false> launch_config(
        g.grid(),
        g.block(),
        g.dynamic_shared_memory(),
        stream,
        dim3(2, C::PAIRS, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
}

template <typename C>
__global__ __launch_bounds__(C::NUM_THREADS, 1) void tma_smoke_kernel(
    const __grid_constant__ globals<C> g,
    uint32_t *done
) {
    using G = globals<C>;
    const int cta_id = kittens::cluster_ctarank();
    const int pair_lane = cta_id & 1;
    const int pair_base = cta_id - pair_lane;
    constexpr uint16_t kAllMask = C::CLUSTER_SIZE == 16
        ? uint16_t{0xffffu}
        : static_cast<uint16_t>((1u << C::CLUSTER_SIZE) - 1u);

    extern __shared__ int __shm[];
    kittens::tma_swizzle_allocator allocator(reinterpret_cast<int *>(__shm));
    typename G::input_tiles_t &inputs =
        allocator.template allocate<typename G::input_tiles_t>();
    typename G::input_scales_t &input_scales =
        allocator.template allocate<typename G::input_scales_t>();
    __shared__ kittens::semaphore inputs_arrived;
    __shared__ kittens::semaphore scales_arrived;
    if (threadIdx.x == 0) {
        kittens::init_semaphore(inputs_arrived, 0, 1);
        kittens::init_semaphore(scales_arrived, 0, 1);
    }
    kittens::everyone::tma::cluster::sync();
    if (threadIdx.x == 0) {
        kittens::tma::cluster::expect_bytes(
            inputs_arrived,
            sizeof(typename G::input_tiles_t)
        );
        kittens::tma::cluster::expect_bytes(
            scales_arrived,
            sizeof(typename G::input_scales_t)
        );
    }
    kittens::everyone::tma::cluster::sync();
    if (threadIdx.x == 0) {
        if (cta_id == 0) {
            kittens::tma::cluster::load_async(
                inputs.A,
                g.A,
                {0, 0},
                inputs_arrived,
                kAllMask
            );
            kittens::tma::cluster::load_async(
                input_scales.A,
                g.A_sc,
                {0, 0, 0},
                scales_arrived,
                kAllMask
            );
        }
        kittens::tma::cluster::load_async(
            inputs.B,
            g.B,
            {cta_id, 0},
            inputs_arrived,
            static_cast<uint16_t>(1u << cta_id)
        );
        kittens::tma::cluster::load_async(
            input_scales.B[pair_lane],
            g.B_sc,
            {cta_id, 0, 0},
            scales_arrived,
            static_cast<uint16_t>(0x3u << pair_base)
        );
        kittens::wait(inputs_arrived, 0);
        kittens::wait(scales_arrived, 0);
        done[cta_id] = 1u;
    }
    __syncthreads();
}

template <typename C>
inline void launch_tma_smoke(
    const globals<C> &g,
    uint32_t *done,
    cudaStream_t stream
) {
    constexpr int dynamic_shared_memory =
        sizeof(typename globals<C>::input_tiles_t) +
        sizeof(typename globals<C>::input_scales_t) + 1024;
    CUDACHECK(cudaFuncSetAttribute(
        tma_smoke_kernel<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory
    ));
    if constexpr (C::CLUSTER_SIZE > 8) {
        CUDACHECK(cudaFuncSetAttribute(
            tma_smoke_kernel<C>,
            cudaFuncAttributeNonPortableClusterSizeAllowed,
            1
        ));
    }
    kittens::LaunchConfig<true, false> launch_config(
        dim3(2, C::PAIRS, 1),
        dim3(C::NUM_THREADS, 1, 1),
        dynamic_shared_memory,
        stream,
        dim3(2, C::PAIRS, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        tma_smoke_kernel<C>,
        g,
        done
    ));
}

} // namespace tkfa4_projection_n_multicast

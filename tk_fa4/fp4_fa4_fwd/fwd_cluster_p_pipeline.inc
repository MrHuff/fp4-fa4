#pragma once

#ifndef TK_FA4_CLUSTER_P_PIPELINE_PROBE_BUILD
#define TK_FA4_CLUSTER_P_PIPELINE_PROBE_BUILD 0
#endif

#ifndef TK_FA4_CLUSTER_P_PIPELINE_TRACE
#define TK_FA4_CLUSTER_P_PIPELINE_TRACE 1
#endif

#if TK_FA4_CLUSTER_P_PIPELINE_PROBE_BUILD

struct config_mxfp4_cluster_p_fake
    : public config_fp4pv<128, 64, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool TRACE = TK_FA4_CLUSTER_P_PIPELINE_TRACE != 0;
};

struct globals_mxfp4_cluster_p_fake {
    using p_tile = st_fp4e2m1_2<128, 32>;
    using p_scale_tile = st_fp8e8m0<32, 16, false>;
    using v_tile = st_fp4e2m1_2<128, 32>;
    using v_scale_tile = st_fp8e8m0<32, 16, false>;

    const uint8_t *p_fp4;
    const int32_t *p_scale_words;
    const uint8_t *v_fp4;
    const int32_t *v_scale_words;
    float *output;
    int64_t *trace;
    int epochs;
    int clusters;

    static constexpr int dynamic_shared_memory_bytes() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add(2 * (int)sizeof(p_tile));
        add(2 * (int)sizeof(p_scale_tile));
        add((int)sizeof(v_tile));
        add((int)sizeof(v_scale_tile));
        return align_up_constexpr(bytes, 1024) + 1024;
    }

    __host__ __inline__ dim3 grid() const { return dim3(clusters * 2); }
    __host__ __inline__ dim3 block() const {
        return dim3(config_mxfp4_cluster_p_fake::NUM_THREADS);
    }
    __host__ __inline__ int dynamic_shared_memory() const {
        return dynamic_shared_memory_bytes();
    }
};

template <typename Tile>
__device__ __forceinline__ void cluster_p_load_payload_wg(
    Tile &tile, const uint8_t *source) {
    static_assert(Tile::rows == 128 && Tile::cols == 32);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = warpgroup::laneid();
         linear < Tile::rows * STORES_PER_ROW;
         linear += 128) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            source + row * Tile::cols + col);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(Tile::idx(base, {row, col})),
            "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename ScaleTile>
__device__ __forceinline__ void cluster_p_load_scale_warp(
    ScaleTile &tile, const int32_t *source) {
    const int lane = warp::laneid();
    const int base = lane * 4;
    fp4pv_store_mxfp4_scale_word4_swizzled(
        tile, lane,
        static_cast<uint32_t>(source[base + 0]),
        static_cast<uint32_t>(source[base + 1]),
        static_cast<uint32_t>(source[base + 2]),
        static_cast<uint32_t>(source[base + 3]));
}

template <typename OutputTT>
__device__ __forceinline__ void cluster_p_store_output(
    OutputTT &output, float *destination) {
    static_assert(OutputTT::rows == 128 && OutputTT::cols == 128);
    const int row = warpgroup::laneid();
    #pragma unroll
    for (int col = 0; col < 128; col += 16) {
        float2 values[8];
        const uint32_t addr = output.addr +
            ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(values[0].x), "=f"(values[0].y),
              "=f"(values[1].x), "=f"(values[1].y),
              "=f"(values[2].x), "=f"(values[2].y),
              "=f"(values[3].x), "=f"(values[3].y),
              "=f"(values[4].x), "=f"(values[4].y),
              "=f"(values[5].x), "=f"(values[5].y),
              "=f"(values[6].x), "=f"(values[6].y),
              "=f"(values[7].x), "=f"(values[7].y)
            : "r"(addr));
        tensor_load_wait();
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            *reinterpret_cast<float2 *>(
                destination + row * 128 + col + pair * 2) = values[pair];
        }
    }
}

__device__ __forceinline__ unsigned long long cluster_p_globaltimer() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__cluster_dims__(2, 1, 1)
__launch_bounds__(config_mxfp4_cluster_p_fake::NUM_THREADS, 1)
__global__ void kernel_mxfp4_cluster_p_fake(
    const __grid_constant__ globals_mxfp4_cluster_p_fake g) {
    using C = config_mxfp4_cluster_p_fake;
    using G = globals_mxfp4_cluster_p_fake;
    using score_tt = tt<float, 128, 64>;
    using output_tt = tt<float, 128, 128>;
    using scale_tt = full_tt_fp8e8m0<16>;

    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = 64;
    constexpr int P_SCALE0_BASE = 192;
    constexpr int P_SCALE1_BASE = 208;
    constexpr int V_SCALE_BASE = 224;
    constexpr int LOGICAL_END = 240;
    constexpr uint32_t P_HANDOFF_BYTES =
        sizeof(G::p_tile) + sizeof(G::p_scale_tile);
    static_assert(C::Nb == 64 && C::Dvo == 128 && C::CLUSTER_SIZE == 1);
    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2);
    static_assert(LOGICAL_END <= tensor_allocator<2, 1>::cols);

    tensor_allocator<2, 1> tm_alloc{};
    score_tt score = tm_alloc.template allocate<score_tt>(SCORE_BASE);
    output_tt output = tm_alloc.template allocate<output_tt>(OUTPUT_BASE);
    scale_tt p_scale[2] = {
        tm_alloc.template allocate<scale_tt>(P_SCALE0_BASE),
        tm_alloc.template allocate<scale_tt>(P_SCALE1_BASE)};
    scale_tt v_scale = tm_alloc.template allocate<scale_tt>(V_SCALE_BASE);
    (void)score;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::p_tile (&p_smem)[2] =
        allocator.allocate<typename G::p_tile, 2>();
    typename G::p_scale_tile (&p_scale_smem)[2] =
        allocator.allocate<typename G::p_scale_tile, 2>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::v_scale_tile &v_scale_smem =
        allocator.allocate<typename G::v_scale_tile>();

    __shared__ semaphore p_ready[2];
    __shared__ semaphore p_free[2];
    __shared__ semaphore p_scale_done[2];
    __shared__ semaphore v_scale_done;
    __shared__ semaphore pv_src_done[2];
    __shared__ semaphore pv_done[2];
    __shared__ semaphore all_pv_done;

    const int cta_rank = cluster_ctarank();
    const int group = warpgroup::groupid();
    const bool issue_warp = cta_rank == 1 && group == 0 &&
        warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const int cluster_id = blockIdx.x >> 1;
    const int64_t trace_base = static_cast<int64_t>(blockIdx.x) * 32;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            init_semaphore(p_ready[slot], 0, 1);
            init_semaphore(p_free[slot], 0, 1);
            init_semaphore(p_scale_done[slot], 0, 1);
            init_semaphore(pv_src_done[slot], 0, 1);
            init_semaphore(pv_done[slot], 0, 1);
        }
        init_semaphore(v_scale_done, 0, 1);
        init_semaphore(all_pv_done, 0, 1);
        if constexpr (C::TRACE) {
            g.trace[trace_base + 0] = blockIdx.x;
            g.trace[trace_base + 1] = cluster_id;
            g.trace[trace_base + 2] = cta_rank;
            g.trace[trace_base + 3] = kittens::smid();
            g.trace[trace_base + 4] = tm_alloc.addr;
            g.trace[trace_base + 5] = LOGICAL_END;
            g.trace[trace_base + 6] = G::dynamic_shared_memory_bytes();
            g.trace[trace_base + 7] = cluster_p_globaltimer();
        }
    }
    everyone::tma::cluster::sync();

    if (cta_rank == 1 && group == 1) {
        cluster_p_load_payload_wg(v_smem, g.v_fp4);
        if (warpgroup::warpid() == 0) {
            cluster_p_load_scale_warp(v_scale_smem, g.v_scale_words);
        }
        warpgroup::sync(2);
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(2);
    }
    __syncthreads();

    if (cta_rank == 1 && threadIdx.x == 0) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            tma::cluster::expect_bytes(p_ready[slot], P_HANDOFF_BYTES);
            tma::cluster::arrive(p_free[slot], 0);
        }
    }

    if (issue_lane) {
        fp4pv_load_mxnv_scale_async_ranklocal(v_scale, v_scale_smem);
        detail::tcgen05::commit<1>(v_scale_done);
    }
    if (issue_warp) {
        wait(v_scale_done, 0);
        tensor_after_thread_sync();
    }

    if (cta_rank == 0 && group == 0) {
        #pragma unroll 1
        for (int epoch = 0; epoch < g.epochs; ++epoch) {
            const int slot = epoch & 1;
            const int phase = (epoch >> 1) & 1;
            if (warpgroup::warpid() == 0) {
                wait(p_free[slot], phase);
            }
            warpgroup::sync(1);
            cluster_p_load_payload_wg(
                p_smem[slot],
                g.p_fp4 + static_cast<int64_t>(epoch) * 128 * 32);
            if (warpgroup::warpid() == 0) {
                cluster_p_load_scale_warp(
                    p_scale_smem[slot],
                    g.p_scale_words + static_cast<int64_t>(epoch) * 32 * 4);
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                tma::cluster::store_async(
                    p_smem[slot], p_smem[slot], 1, p_ready[slot]);
                tma::cluster::store_async(
                    p_scale_smem[slot], p_scale_smem[slot], 1,
                    p_ready[slot]);
                if constexpr (C::TRACE) {
                    if (epoch == 0) {
                        g.trace[trace_base + 8] = cluster_p_globaltimer();
                    }
                    g.trace[trace_base + 10] = epoch + 1;
                }
            }
        }
    }

    if (issue_warp) {
        #pragma unroll 1
        for (int epoch = 0; epoch < g.epochs; ++epoch) {
            const int slot = epoch & 1;
            const int phase = (epoch >> 1) & 1;
            if (epoch >= 2) {
                wait(pv_src_done[slot], phase ^ 1);
                wait(pv_done[slot], phase ^ 1);
                tensor_after_thread_sync();
                if (issue_lane) {
                    tma::cluster::expect_bytes(
                        p_ready[slot], P_HANDOFF_BYTES);
                    tma::cluster::arrive(p_free[slot], 0);
                }
            }
            wait(p_ready[slot], phase);
            tensor_after_thread_sync();
            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(
                    p_scale[slot], p_scale_smem[slot]);
                detail::tcgen05::commit<1>(p_scale_done[slot]);
            }
            wait(p_scale_done[slot], phase);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
            if (issue_lane) {
                if (epoch == 0) {
                    fp4pv_mm_p_stage_ranklocal_ABt<C>(
                        output, p_smem[slot], v_smem,
                        p_scale[slot], v_scale, pv_src_done[slot]);
                } else {
                    fp4pv_mma_p_stage_ranklocal_ABt<C>(
                        output, p_smem[slot], v_smem,
                        p_scale[slot], v_scale, pv_src_done[slot]);
                }
                tensor_commit<1>(pv_done[slot]);
                if constexpr (C::TRACE) {
                    if (epoch == 0) {
                        g.trace[trace_base + 8] = cluster_p_globaltimer();
                    }
                    g.trace[trace_base + 10] = epoch + 1;
                }
            }
        }
        const int drain_begin = max(0, g.epochs - 2);
        for (int epoch = drain_begin; epoch < g.epochs; ++epoch) {
            const int slot = epoch & 1;
            const int phase = (epoch >> 1) & 1;
            wait(pv_src_done[slot], phase);
            wait(pv_done[slot], phase);
            tensor_after_thread_sync();
        }
        if (issue_lane) {
            arrive(all_pv_done);
        }
    }

    if (cta_rank == 1 && group == 1) {
        wait(all_pv_done, 0);
        tensor_after_thread_sync();
        cluster_p_store_output(
            output,
            g.output + static_cast<int64_t>(cluster_id) * 128 * 128);
        warpgroup::sync(2);
        if constexpr (C::TRACE) {
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                g.trace[trace_base + 9] = cluster_p_globaltimer();
                g.trace[trace_base + 11] = g.epochs;
            }
        }
    }

    __syncthreads();
    everyone::tma::cluster::sync();
    if constexpr (C::TRACE) {
        if (threadIdx.x == 0) {
            g.trace[trace_base + 12] = cluster_p_globaltimer();
            __threadfence_system();
            g.trace[trace_base + 15] = 1;
        }
    }
}

__launch_bounds__(config_mxfp4_cluster_p_fake::NUM_THREADS, 1)
__global__ void kernel_mxfp4_cluster_p_local_control(
    const __grid_constant__ globals_mxfp4_cluster_p_fake g) {
    using C = config_mxfp4_cluster_p_fake;
    using G = globals_mxfp4_cluster_p_fake;
    using score_tt = tt<float, 128, 64>;
    using output_tt = tt<float, 128, 128>;
    using scale_tt = full_tt_fp8e8m0<16>;
    constexpr int LOGICAL_END = 240;

    tensor_allocator<2, 1> tm_alloc{};
    score_tt score = tm_alloc.template allocate<score_tt>(0);
    output_tt output = tm_alloc.template allocate<output_tt>(64);
    scale_tt p_scale[2] = {
        tm_alloc.template allocate<scale_tt>(192),
        tm_alloc.template allocate<scale_tt>(208)};
    scale_tt v_scale = tm_alloc.template allocate<scale_tt>(224);
    (void)score;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::p_tile (&p_smem)[2] =
        allocator.allocate<typename G::p_tile, 2>();
    typename G::p_scale_tile (&p_scale_smem)[2] =
        allocator.allocate<typename G::p_scale_tile, 2>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::v_scale_tile &v_scale_smem =
        allocator.allocate<typename G::v_scale_tile>();
    __shared__ semaphore p_scale_done[2];
    __shared__ semaphore v_scale_done;
    __shared__ semaphore pv_src_done[2];
    __shared__ semaphore pv_done[2];
    __shared__ semaphore all_pv_done;

    const int group = warpgroup::groupid();
    const bool issue_warp = group == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const int64_t trace_base = static_cast<int64_t>(blockIdx.x) * 32;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            init_semaphore(p_scale_done[slot], 0, 1);
            init_semaphore(pv_src_done[slot], 0, 1);
            init_semaphore(pv_done[slot], 0, 1);
        }
        init_semaphore(v_scale_done, 0, 1);
        init_semaphore(all_pv_done, 0, 1);
        if constexpr (C::TRACE) {
            g.trace[trace_base + 0] = blockIdx.x;
            g.trace[trace_base + 1] = blockIdx.x;
            g.trace[trace_base + 2] = 0;
            g.trace[trace_base + 3] = kittens::smid();
            g.trace[trace_base + 4] = tm_alloc.addr;
            g.trace[trace_base + 5] = LOGICAL_END;
            g.trace[trace_base + 6] = G::dynamic_shared_memory_bytes();
            g.trace[trace_base + 7] = cluster_p_globaltimer();
        }
    }

    if (group == 1) {
        cluster_p_load_payload_wg(v_smem, g.v_fp4);
        if (warpgroup::warpid() == 0) {
            cluster_p_load_scale_warp(v_scale_smem, g.v_scale_words);
        }
        warpgroup::sync(2);
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(2);
    }
    __syncthreads();
    if (issue_warp) {
        tensor_after_thread_sync();
    }
    if (issue_lane) {
        fp4pv_load_mxnv_scale_async_ranklocal(v_scale, v_scale_smem);
        detail::tcgen05::commit<1>(v_scale_done);
    }
    if (issue_warp) {
        wait(v_scale_done, 0);
        tensor_after_thread_sync();
    }

    if (group == 0) {
        #pragma unroll 1
        for (int epoch = 0; epoch < g.epochs; ++epoch) {
            const int slot = epoch & 1;
            const int phase = (epoch >> 1) & 1;
            if (epoch >= 2 && issue_warp) {
                wait(pv_src_done[slot], phase ^ 1);
                wait(pv_done[slot], phase ^ 1);
                tensor_after_thread_sync();
            }
            warpgroup::sync(1);
            cluster_p_load_payload_wg(
                p_smem[slot],
                g.p_fp4 + static_cast<int64_t>(epoch) * 128 * 32);
            if (warpgroup::warpid() == 0) {
                cluster_p_load_scale_warp(
                    p_scale_smem[slot],
                    g.p_scale_words + static_cast<int64_t>(epoch) * 32 * 4);
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
            if (issue_warp) {
                tensor_after_thread_sync();
            }
            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(
                    p_scale[slot], p_scale_smem[slot]);
                detail::tcgen05::commit<1>(p_scale_done[slot]);
            }
            if (issue_warp) {
                wait(p_scale_done[slot], phase);
                tensor_after_thread_sync();
                tensor_before_thread_sync();
            }
            if (issue_lane) {
                if (epoch == 0) {
                    fp4pv_mm_p_stage_ranklocal_ABt<C>(
                        output, p_smem[slot], v_smem,
                        p_scale[slot], v_scale, pv_src_done[slot]);
                } else {
                    fp4pv_mma_p_stage_ranklocal_ABt<C>(
                        output, p_smem[slot], v_smem,
                        p_scale[slot], v_scale, pv_src_done[slot]);
                }
                tensor_commit<1>(pv_done[slot]);
                if constexpr (C::TRACE) {
                    g.trace[trace_base + 10] = epoch + 1;
                }
            }
        }
        if (issue_warp) {
            const int drain_begin = max(0, g.epochs - 2);
            for (int epoch = drain_begin; epoch < g.epochs; ++epoch) {
                const int slot = epoch & 1;
                const int phase = (epoch >> 1) & 1;
                wait(pv_src_done[slot], phase);
                wait(pv_done[slot], phase);
                tensor_after_thread_sync();
            }
            if (issue_lane) {
                arrive(all_pv_done);
            }
        }
    }

    if (group == 1) {
        wait(all_pv_done, 0);
        tensor_after_thread_sync();
        cluster_p_store_output(
            output,
            g.output + static_cast<int64_t>(blockIdx.x) * 128 * 128);
        warpgroup::sync(2);
        if constexpr (C::TRACE) {
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                g.trace[trace_base + 9] = cluster_p_globaltimer();
                g.trace[trace_base + 11] = g.epochs;
            }
        }
    }
    __syncthreads();
    if constexpr (C::TRACE) {
        if (threadIdx.x == 0) {
            g.trace[trace_base + 12] = cluster_p_globaltimer();
            __threadfence_system();
            g.trace[trace_base + 15] = 1;
        }
    }
}

void dispatch_mxfp4_cluster_p_fake(
    at::Tensor p_fp4,
    at::Tensor p_scale_words,
    at::Tensor v_fp4,
    at::Tensor v_scale_words,
    at::Tensor output,
    at::Tensor trace) {
    for (const at::Tensor *tensor : {
             &p_fp4, &p_scale_words, &v_fp4, &v_scale_words,
             &output, &trace}) {
        TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "cluster P probe tensors must be contiguous CUDA tensors");
    }
    TORCH_CHECK(p_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    p_fp4.dim() == 3 && p_fp4.size(1) == 128 &&
                    p_fp4.size(2) == 32,
                "P must be float4_e2m1fn_x2 (epochs,128,32)");
    TORCH_CHECK(p_scale_words.dtype() == at::ScalarType::Int &&
                    p_scale_words.dim() == 3 &&
                    p_scale_words.size(0) == p_fp4.size(0) &&
                    p_scale_words.size(1) == 32 &&
                    p_scale_words.size(2) == 4,
                "P scales must be int32 (epochs,32,4)");
    TORCH_CHECK(v_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    v_fp4.sizes() == at::IntArrayRef({128, 32}),
                "V must be float4_e2m1fn_x2 (128,32)");
    TORCH_CHECK(v_scale_words.dtype() == at::ScalarType::Int &&
                    v_scale_words.sizes() == at::IntArrayRef({32, 4}),
                "V scales must be int32 (32,4)");
    TORCH_CHECK(output.dtype() == at::ScalarType::Float &&
                    output.dim() == 3 && output.size(1) == 128 &&
                    output.size(2) == 128,
                "output must be float32 (clusters,128,128)");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long && trace.dim() == 2 &&
                    trace.size(0) == 2 * output.size(0) &&
                    trace.size(1) >= 32,
                "trace must be int64 (2*clusters,>=32)");
    TORCH_CHECK(p_fp4.size(0) >= 1 && p_fp4.size(0) <= 64,
                "cluster P probe requires 1..64 epochs");
    TORCH_CHECK(output.size(0) >= 1,
                "cluster P probe requires at least one cluster");

    globals_mxfp4_cluster_p_fake g{
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        p_scale_words.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t *>(v_fp4.data_ptr()),
        v_scale_words.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        static_cast<int>(p_fp4.size(0)),
        static_cast<int>(output.size(0)),
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_cluster_p_fake,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 2);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_mxfp4_cluster_p_fake, g));
}

void dispatch_mxfp4_cluster_p_local_control(
    at::Tensor p_fp4,
    at::Tensor p_scale_words,
    at::Tensor v_fp4,
    at::Tensor v_scale_words,
    at::Tensor output,
    at::Tensor trace) {
    for (const at::Tensor *tensor : {
             &p_fp4, &p_scale_words, &v_fp4, &v_scale_words,
             &output, &trace}) {
        TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "local P control tensors must be contiguous CUDA tensors");
    }
    TORCH_CHECK(p_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    p_fp4.dim() == 3 && p_fp4.size(1) == 128 &&
                    p_fp4.size(2) == 32,
                "P must be float4_e2m1fn_x2 (epochs,128,32)");
    TORCH_CHECK(p_scale_words.dtype() == at::ScalarType::Int &&
                    p_scale_words.sizes() == at::IntArrayRef(
                        {p_fp4.size(0), 32, 4}),
                "P scales must be int32 (epochs,32,4)");
    TORCH_CHECK(v_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    v_fp4.sizes() == at::IntArrayRef({128, 32}),
                "V must be float4_e2m1fn_x2 (128,32)");
    TORCH_CHECK(v_scale_words.dtype() == at::ScalarType::Int &&
                    v_scale_words.sizes() == at::IntArrayRef({32, 4}),
                "V scales must be int32 (32,4)");
    TORCH_CHECK(output.dtype() == at::ScalarType::Float &&
                    output.dim() == 3 && output.size(1) == 128 &&
                    output.size(2) == 128,
                "output must be float32 (blocks,128,128)");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long && trace.dim() == 2 &&
                    trace.size(0) == output.size(0) && trace.size(1) >= 32,
                "trace must be int64 (blocks,>=32)");
    TORCH_CHECK(p_fp4.size(0) >= 1 && p_fp4.size(0) <= 64,
                "local P control requires 1..64 epochs");

    globals_mxfp4_cluster_p_fake g{
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        p_scale_words.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t *>(v_fp4.data_ptr()),
        v_scale_words.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        static_cast<int>(p_fp4.size(0)),
        static_cast<int>(output.size(0)),
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_cluster_p_local_control,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        dim3(g.clusters), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_mxfp4_cluster_p_local_control, g));
}

#endif

#ifndef TK_FA4_MXFP4_QK_CTA_GROUP_AB_BUILD
#define TK_FA4_MXFP4_QK_CTA_GROUP_AB_BUILD 0
#endif

#ifndef TK_FA4_QK_CTA_GROUP_AB_PHYSICAL_CTAS
#define TK_FA4_QK_CTA_GROUP_AB_PHYSICAL_CTAS 152
#endif

#if TK_FA4_MXFP4_QK_CTA_GROUP_AB_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)

// Work-normalized Dqk192 QK-only control.  The group-1 launch performs one
// M128xN128 product in each of 152 independent CTAs.  The group-2 launch uses
// 76 two-CTA clusters; each cluster performs one collective M256xN128 product.
// Both therefore produce 152 logical M128xN128xK192 query products per outer
// iteration.  Payload/scale staging and the optional score store are outside
// the recorded issue interval.
template <int _ClusterSize, bool _WaitEveryIteration>
struct config_mxfp4_qk_cta_group_ab_probe {
    static_assert(_ClusterSize == 1 || _ClusterSize == 2,
                  "QK CTA-group probe supports group 1 or 2");

    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dqk = 192;
    static constexpr int CLUSTER_SIZE = _ClusterSize;
    static constexpr bool WAIT_EVERY_ITERATION = _WaitEveryIteration;
    static constexpr int QK_SCALE_CHUNKS = Dqk / 64;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int PHYSICAL_CTAS =
        TK_FA4_QK_CTA_GROUP_AB_PHYSICAL_CTAS;
    static constexpr int TRACE_FIELDS = 16;
};

template <typename C>
struct globals_mxfp4_qk_cta_group_ab_probe {
    using q_tile = st_fp4e2m1_2<C::Mb, C::Dqk / 2>;
    using k_tile = std::conditional_t<
        C::CLUSTER_SIZE == 1,
        st_fp4e2m1_2<C::Nb, C::Dqk / 2>,
        st_fp4e2m1_2<C::Nb / C::CLUSTER_SIZE, C::Dqk / 2>>;
    using q_sc_tile = st_hf<C::QK_SCALE_CHUNKS, 256, false>;
    using k_sc_tile = st_hf<C::QK_SCALE_CHUNKS, 256, false>;

    const uint8_t *q_fp4;
    const uint8_t *q_sc;
    const uint8_t *k_fp4;
    const uint8_t *k_sc;
    float *scores;
    unsigned long long *trace;
    int iterations;
    int store_scores;

    __host__ __inline__ dim3 grid() const { return dim3(C::PHYSICAL_CTAS); }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() const {
        return align_up_constexpr((int)sizeof(q_tile), 1024) +
               align_up_constexpr((int)sizeof(k_tile), 1024) +
               align_up_constexpr((int)sizeof(q_sc_tile), 1024) +
               align_up_constexpr((int)sizeof(k_sc_tile), 1024) + 4096;
    }
};

template <typename Tile>
__device__ __forceinline__ void mxfp4_qk_cta_group_copy_payload(
    Tile &tile, const uint8_t *src) {
    constexpr int BYTES_PER_STORE = 16;
    static_assert(Tile::cols % BYTES_PER_STORE == 0,
                  "QK probe payload rows must be vector-loadable");
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x; linear < TOTAL_STORES;
         linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value =
            *reinterpret_cast<const uint4 *>(src + row * Tile::cols + col);
        asm volatile(
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
            :
            : "r"(Tile::idx(smem_base, {row, col})), "r"(value.x),
              "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename Tile>
__device__ __forceinline__ void mxfp4_qk_cta_group_copy_scale(
    Tile &tile, const uint8_t *src) {
    static_assert(sizeof(Tile) % sizeof(uint4) == 0,
                  "QK probe scale record must be vector-loadable");
    constexpr int WORDS = sizeof(Tile) / sizeof(uint4);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int word = threadIdx.x; word < WORDS; word += blockDim.x) {
        const uint4 value = reinterpret_cast<const uint4 *>(src)[word];
        asm volatile(
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
            :
            : "r"(smem_base + static_cast<uint32_t>(word * sizeof(uint4))),
              "r"(value.x),
              "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename ScoreTT>
__device__ __forceinline__ void mxfp4_qk_cta_group_store_scores(
    const ScoreTT &tt_score, float *dst) {
    const int row = warpgroup::laneid();
    const uint32_t row_base =
        tt_score.addr + ((warpgroup::warpid() * 32) << 16);
    #pragma unroll
    for (int col = 0; col < 128; col += 16) {
        float2 values[8];
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
            : "r"(row_base + col));
        tensor_load_wait();
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            *reinterpret_cast<float2 *>(
                dst + row * 128 + col + pair * 2) = values[pair];
        }
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1)
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_qk_cta_group_ab_probe(
    const __grid_constant__ globals_mxfp4_qk_cta_group_ab_probe<C> g) {
    using G = globals_mxfp4_qk_cta_group_ab_probe<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_scale_all =
        full_tt_fp8e4m3<16 * C::QK_SCALE_CHUNKS>;
    using T_AB = typename G::q_tile::T;
    using T_SAB = typename d_tt_scale_all::T;
    using T_D = typename d_tt_score::T;

    static_assert(C::Mb == 128 && C::Nb == 128 && C::Dqk == 192,
                  "QK CTA-group probe is fixed to M128/N128/K192 rank tiles");
    static_assert(C::QK_SCALE_CHUNKS == 3,
                  "QK CTA-group probe requires three K64 scale chunks");
    static_assert(C::PHYSICAL_CTAS % C::CLUSTER_SIZE == 0,
                  "QK CTA-group grid must contain whole clusters");

    const int cta_rank = cluster_ctarank();
    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(0);
    d_tt_scale_all q_sc_tm =
        tm_alloc.template allocate<d_tt_scale_all>(128);
    d_tt_scale_all k_sc_tm = tm_alloc.template allocate<d_tt_scale_all>(176);
    static_assert(224 <= tensor_allocator<1, C::CLUSTER_SIZE>::cols,
                  "QK CTA-group score and scales must fit one 512-column allocation");

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile &q_smem = al.allocate<typename G::q_tile>();
    typename G::k_tile &k_smem = al.allocate<typename G::k_tile>();
    typename G::q_sc_tile &q_sc_smem = al.allocate<typename G::q_sc_tile>();
    typename G::k_sc_tile &k_sc_smem = al.allocate<typename G::k_sc_tile>();

    __shared__ semaphore qk_done;
    if (threadIdx.x == 0) {
        init_semaphore(qk_done, 0, 1);
    }
    __syncthreads();

    const uint8_t *q_src =
        g.q_fp4 + blockIdx.x * sizeof(typename G::q_tile);
    const uint8_t *k_src =
        g.k_fp4 + blockIdx.x * sizeof(typename G::k_tile);
    const uint8_t *q_sc_src =
        g.q_sc + blockIdx.x * sizeof(typename G::q_sc_tile);
    const uint8_t *k_sc_src =
        g.k_sc + blockIdx.x * sizeof(typename G::k_sc_tile);
    mxfp4_qk_cta_group_copy_payload(q_smem, q_src);
    mxfp4_qk_cta_group_copy_payload(k_smem, k_src);
    mxfp4_qk_cta_group_copy_scale(q_sc_smem, q_sc_src);
    mxfp4_qk_cta_group_copy_scale(k_sc_smem, k_sc_src);
    __syncthreads();
    fp4pv_publish_shared_backing_proxy_only();
    __syncthreads();
    fp4pv_cluster_sync<C>();

    const bool issue_warp =
        warpgroup::groupid() == 0 && warpgroup::warpid() == 0 &&
        (C::CLUSTER_SIZE == 1 || cta_rank == 0);
    const bool issue_lane = issue_warp && warp::elect_leader();
    if (issue_warp) {
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            auto q_sc_sub = q_sc_tm.template subtile<full_tt_fp8e4m3<16>>(
                chunk * 16);
            auto k_sc_sub = k_sc_tm.template subtile<full_tt_fp8e4m3<16>>(
                chunk * 16);
            fp4pv_load_q_scale_chunk_cluster<C>(
                q_sc_sub, q_sc_smem, chunk);
            fp4pv_load_k_scale_chunk_cluster<C>(
                k_sc_sub, k_sc_smem, chunk);
        }
        tensor_load_wait();
        tensor_before_thread_sync();
    }
    fp4pv_cluster_sync<C>();

    st_descriptor<typename G::q_tile, 0> q_desc(q_smem);
    st_descriptor<typename G::k_tile, 0> k_desc(k_smem);
    constexpr int LOGICAL_M = C::Mb * C::CLUSTER_SIZE;
    constexpr int LOGICAL_N =
        G::k_tile::rows * C::CLUSTER_SIZE;
    static_assert((C::CLUSTER_SIZE == 1 && LOGICAL_M == 128) ||
                      (C::CLUSTER_SIZE == 2 && LOGICAL_M == 256),
                  "QK CTA-group M descriptor mismatch");
    static_assert(LOGICAL_N == 128,
                  "QK CTA-group control must retain production N128");
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, LOGICAL_M, LOGICAL_N, false, 0>();

    unsigned long long global_start = 0;
    unsigned long long global_end = 0;
    unsigned long long clock_start = 0;
    unsigned long long clock_end = 0;
    if (issue_lane) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_start));
        clock_start = clock64();
    }

    int done_phase = 0;
    if (issue_warp) {
        for (int iteration = 0; iteration < g.iterations; ++iteration) {
            #pragma unroll
            for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                auto q_sc_sub =
                    q_sc_tm.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
                auto k_sc_sub =
                    k_sc_tm.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
                if (issue_lane) {
                    if (chunk == 0) {
                        detail::tcgen05::template st_st<
                            T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                            tt_score.addr,
                            q_desc.chunk_descriptor(chunk),
                            k_desc.chunk_descriptor(chunk), q_sc_sub.addr,
                            k_sc_sub.addr, IDESC);
                    } else {
                        detail::tcgen05::template st_st<
                            T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                            tt_score.addr,
                            q_desc.chunk_descriptor(chunk),
                            k_desc.chunk_descriptor(chunk), q_sc_sub.addr,
                            k_sc_sub.addr, IDESC);
                    }
                }
            }
            if constexpr (C::WAIT_EVERY_ITERATION) {
                if (issue_lane) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(qk_done);
                    wait(qk_done, done_phase);
                    done_phase ^= 1;
                }
            }
        }
        if constexpr (!C::WAIT_EVERY_ITERATION) {
            if (issue_lane) {
                detail::tcgen05::commit<C::CLUSTER_SIZE>(qk_done);
                wait(qk_done, 0);
            }
        }
        tensor_after_thread_sync();
    }
    if (issue_lane) {
        clock_end = clock64();
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_end));
    }

    // Keep rank 1 alive until the collective issue and completion are done,
    // then make the final score generation visible before optional TMEM reads.
    fp4pv_cluster_sync<C>();
    if (g.store_scores) {
        tensor_after_thread_sync();
        if (warpgroup::groupid() == 1) {
            mxfp4_qk_cta_group_store_scores(
                tt_score,
                g.scores + blockIdx.x * C::Mb * C::Nb);
        }
        warpgroup::sync(warpgroup::groupid() + 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long *record =
            g.trace + blockIdx.x * C::TRACE_FIELDS;
        record[0] = 0x514b433247525031ull;  // "QKC2GRP1"
        record[1] = C::CLUSTER_SIZE;
        record[2] = C::WAIT_EVERY_ITERATION ? 1 : 0;
        record[3] = g.iterations;
        record[4] = blockIdx.x;
        record[5] = cta_rank;
        record[6] = kittens::smid();
        record[7] = C::PHYSICAL_CTAS;
        record[8] = C::PHYSICAL_CTAS / C::CLUSTER_SIZE;
        record[9] = global_start;
        record[10] = global_end;
        record[11] = clock_start;
        record[12] = clock_end;
        record[13] = g.store_scores;
        record[14] = LOGICAL_M;
        record[15] = LOGICAL_N;
    }
}

template <int ClusterSize, bool WaitEveryIteration>
static void launch_mxfp4_qk_cta_group_ab_probe(
    at::Tensor q_fp4, at::Tensor q_sc, at::Tensor k_fp4,
    at::Tensor k_sc, at::Tensor scores, at::Tensor trace,
    int iterations, bool store_scores) {
    using C = config_mxfp4_qk_cta_group_ab_probe<
        ClusterSize, WaitEveryIteration>;
    using G = globals_mxfp4_qk_cta_group_ab_probe<C>;
    CHECK_INPUT(q_fp4);
    CHECK_INPUT(q_sc);
    CHECK_INPUT(k_fp4);
    CHECK_INPUT(k_sc);
    CHECK_INPUT(scores);
    CHECK_INPUT(trace);
    TORCH_CHECK(q_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "QK probe q_fp4 must use packed E2M1");
    TORCH_CHECK(k_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "QK probe k_fp4 must use packed E2M1");
    TORCH_CHECK(q_sc.dtype() == at::ScalarType::Byte &&
                    k_sc.dtype() == at::ScalarType::Byte,
                "QK probe scale records must be uint8 E4M3 bytes");
    TORCH_CHECK(scores.dtype() == at::ScalarType::Float,
                "QK probe scores must be float32");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long,
                "QK probe trace must be int64");
    TORCH_CHECK(iterations > 0 && iterations <= 4096,
                "QK probe iterations must be in [1,4096]");
    TORCH_CHECK(!store_scores || iterations == 1,
                "QK probe score verification requires one iteration");
    TORCH_CHECK(q_fp4.dim() == 3 &&
                    q_fp4.size(0) == C::PHYSICAL_CTAS &&
                    q_fp4.size(1) == C::Mb &&
                    q_fp4.size(2) == C::Dqk / 2,
                "QK probe q_fp4 must be [152,128,96]");
    TORCH_CHECK(k_fp4.dim() == 3 &&
                    k_fp4.size(0) == C::PHYSICAL_CTAS &&
                    k_fp4.size(1) == G::k_tile::rows &&
                    k_fp4.size(2) == C::Dqk / 2,
                "QK probe k_fp4 has the wrong group-specific shape");
    TORCH_CHECK(q_sc.numel() ==
                    C::PHYSICAL_CTAS * (int64_t)sizeof(typename G::q_sc_tile) &&
                    k_sc.numel() ==
                    C::PHYSICAL_CTAS * (int64_t)sizeof(typename G::k_sc_tile),
                "QK probe scale records have the wrong byte count");
    TORCH_CHECK(scores.numel() >=
                    C::PHYSICAL_CTAS * C::Mb * C::Nb,
                "QK probe scores must hold 152 M128xN128 tiles");
    TORCH_CHECK(trace.numel() >=
                    C::PHYSICAL_CTAS * C::TRACE_FIELDS,
                "QK probe trace must hold 152x16 int64 values");

    G g{
        reinterpret_cast<const uint8_t *>(q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(q_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(k_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(k_sc.data_ptr()),
        scores.data_ptr<float>(),
        reinterpret_cast<unsigned long long *>(trace.data_ptr<int64_t>()),
        iterations,
        store_scores ? 1 : 0,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_qk_cta_group_ab_probe<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_mxfp4_qk_cta_group_ab_probe<C>, g));
}

void dispatch_mxfp4_qk_group1_issue_ceiling(
    at::Tensor q_fp4, at::Tensor q_sc, at::Tensor k_fp4,
    at::Tensor k_sc, at::Tensor scores, at::Tensor trace,
    int iterations, bool store_scores) {
    launch_mxfp4_qk_cta_group_ab_probe<1, false>(
        q_fp4, q_sc, k_fp4, k_sc, scores, trace,
        iterations, store_scores);
}

void dispatch_mxfp4_qk_group1_production_cadence(
    at::Tensor q_fp4, at::Tensor q_sc, at::Tensor k_fp4,
    at::Tensor k_sc, at::Tensor scores, at::Tensor trace,
    int iterations, bool store_scores) {
    launch_mxfp4_qk_cta_group_ab_probe<1, true>(
        q_fp4, q_sc, k_fp4, k_sc, scores, trace,
        iterations, store_scores);
}

void dispatch_mxfp4_qk_group2_issue_ceiling(
    at::Tensor q_fp4, at::Tensor q_sc, at::Tensor k_fp4,
    at::Tensor k_sc, at::Tensor scores, at::Tensor trace,
    int iterations, bool store_scores) {
    launch_mxfp4_qk_cta_group_ab_probe<2, false>(
        q_fp4, q_sc, k_fp4, k_sc, scores, trace,
        iterations, store_scores);
}

void dispatch_mxfp4_qk_group2_production_cadence(
    at::Tensor q_fp4, at::Tensor q_sc, at::Tensor k_fp4,
    at::Tensor k_sc, at::Tensor scores, at::Tensor trace,
    int iterations, bool store_scores) {
    launch_mxfp4_qk_cta_group_ab_probe<2, true>(
        q_fp4, q_sc, k_fp4, k_sc, scores, trace,
        iterations, store_scores);
}

#endif  // TK_FA4_MXFP4_QK_CTA_GROUP_AB_BUILD

#if TK_FA4_OPTION_AB_PROBE_BUILD

template <typename C>
struct globals_mxfp4_option_b_nb64_dvo128 {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_tile = st_fp4e2m1_2<C::Mb, C::Nb / 2>;
    using p_sc_tile = st_fp8e8m0<32, 16, false>;
    using v_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using v_sc_tile = st_fp8e8m0<32, 16, false>;
    using q_gl = typename base::q_gl;
    using k_gl = gl<fp4e2m1_2, -1, -1, -1, -1,
                    tma::descriptor<k_tile, dim::ROW>>;
    using v_gl = gl<fp4e2m1_2, -1, -1, -1, -1,
                    tma::descriptor<v_tile, dim::ROW>>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_tile));
        add((int)sizeof(q_sc_tile));
        add((int)sizeof(k_tile));
        add((int)sizeof(k_sc_tile));
        add((int)sizeof(p_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_tile));
        add((int)sizeof(v_sc_tile));
        return align_up_constexpr(bytes, 1024) + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY =
        compute_dynamic_shared_memory();

    q_gl q;
    k_gl k;
    v_gl v;
    const uint8_t *q_sc;
    const float *q_sg;
    const uint8_t *k_sc;
    const float *k_sg;
    const uint8_t *v_sc;
    bf16 *o;
    float *lse;
    unsigned long long *trace;
    int batch;
    int heads;
    int seqlen;
    int trace_stride;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / C::Mb));
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() {
        return DYNAMIC_SHARED_MEMORY;
    }
};

template <typename C, typename OutputTT>
__device__ __forceinline__ void option_b_rescale_output(
    OutputTT &output, float correction) {
    const float effective = correction < 0.999f ? correction : 1.0f;
    const float2 correction2 = {effective, effective};
    #pragma unroll
    for (int col = 0; col < OutputTT::cols; col += 16) {
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
            values[pair] = __fmul2_rn(values[pair], correction2);
        }
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
            :: "f"(values[0].x), "f"(values[0].y),
               "f"(values[1].x), "f"(values[1].y),
               "f"(values[2].x), "f"(values[2].y),
               "f"(values[3].x), "f"(values[3].y),
               "f"(values[4].x), "f"(values[4].y),
               "f"(values[5].x), "f"(values[5].y),
               "f"(values[6].x), "f"(values[6].y),
               "f"(values[7].x), "f"(values[7].y),
               "r"(addr));
    }
    tensor_store_wait();
}

template <typename C, int MIN_BLOCKS_PER_SM, bool ENABLE_TRACE = true>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_option_b_nb64_dvo128(
    const __grid_constant__ globals_mxfp4_option_b_nb64_dvo128<C> g) {
    using G = globals_mxfp4_option_b_nb64_dvo128<C>;
    using score_tt = tt<float, C::Mb, C::Nb>;
    using output_tt = tt<float, C::Mb, C::Dvo>;
    using qk_scale_tt = full_tt_fp8e4m3<16>;
    using pv_scale_tt = full_tt_fp8e8m0<16>;

    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = 64;
    constexpr int SCALE_A_BASE = 192;
    constexpr int SCALE_B_BASE = 208;
    constexpr int LOGICAL_END = SCALE_B_BASE + 16;
    constexpr int PHYSICAL_COLS = tensor_allocator<2, 1>::cols;
    constexpr float LOG2E_OVER_SQRT_D = 0.10408340855860843f;
    constexpr float OUTPUT_CORRECTION =
        1.0f / (36.0f * FP4PV_MXFP4_ONLINE_P_SCALE);

    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2);
    static_assert(C::CLUSTER_SIZE == 1 && C::Mb == 128 && C::Nb == 64);
    static_assert(C::Dqk == 192 && C::Dvo == 128);
    static_assert(LOGICAL_END <= PHYSICAL_COLS);
    static_assert(PHYSICAL_COLS == 256);

    tensor_allocator<2, 1> tm_alloc{};
    score_tt score = tm_alloc.template allocate<score_tt>(SCORE_BASE);
    output_tt output = tm_alloc.template allocate<output_tt>(OUTPUT_BASE);
    qk_scale_tt q_scale =
        tm_alloc.template allocate<qk_scale_tt>(SCALE_A_BASE);
    qk_scale_tt k_scale =
        tm_alloc.template allocate<qk_scale_tt>(SCALE_B_BASE);
    pv_scale_tt p_scale =
        tm_alloc.template allocate<pv_scale_tt>(SCALE_A_BASE);
    pv_scale_tt v_scale =
        tm_alloc.template allocate<pv_scale_tt>(SCALE_B_BASE);

    const int task = blockIdx.x;
    const int tiles = g.seqlen / C::Mb;
    const int num_bh = g.batch * g.heads;
    const int query_tile = task / num_bh;
    const int bh = task - query_tile * num_bh;
    const int b = bh / g.heads;
    const int h = bh - b * g.heads;
    const int micro_iters = 2 * (query_tile + 1);

    auto globaltimer = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if constexpr (ENABLE_TRACE) {
            if (field < g.trace_stride) {
                g.trace[static_cast<unsigned long long>(task) * g.trace_stride +
                        field] = value;
            }
        }
    };
    if constexpr (ENABLE_TRACE) {
        if (threadIdx.x == 0) {
            trace_store(0, task);
            trace_store(1, kittens::smid());
            trace_store(2, tm_alloc.addr);
            trace_store(3, MIN_BLOCKS_PER_SM);
            trace_store(4, query_tile);
            trace_store(5, h);
            trace_store(6, micro_iters);
            trace_store(7, globaltimer());
            trace_store(14, PHYSICAL_COLS);
            trace_store(15, 0);
        }
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::q_tile &q_smem = allocator.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem =
        allocator.allocate<typename G::q_sc_tile>();
    typename G::k_tile &k_smem = allocator.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem =
        allocator.allocate<typename G::k_sc_tile>();
    typename G::p_tile &p_smem = allocator.allocate<typename G::p_tile>();
    typename G::p_sc_tile &p_sc_smem =
        allocator.allocate<typename G::p_sc_tile>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::v_sc_tile &v_sc_smem =
        allocator.allocate<typename G::v_sc_tile>();

    __shared__ semaphore q_payload_done;
    __shared__ semaphore k_payload_done;
    __shared__ semaphore v_payload_done;
    __shared__ semaphore qk_scale_done[C::QK_SCALE_CHUNKS];
    __shared__ semaphore qk_done;
    __shared__ semaphore p_ready;
    __shared__ semaphore pv_scale_done;
    __shared__ semaphore pv_src_done;
    __shared__ semaphore pv_done;
    if (threadIdx.x == 0) {
        init_semaphore(q_payload_done, 0, 1);
        init_semaphore(k_payload_done, 0, 1);
        init_semaphore(v_payload_done, 0, 1);
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            init_semaphore(qk_scale_done[chunk], 0, 1);
        }
        init_semaphore(qk_done, 0, 1);
        init_semaphore(p_ready, 0, 1);
        init_semaphore(pv_scale_done, 0, 1);
        init_semaphore(pv_src_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
    }
    __syncthreads();

    const bool producer_group = warpgroup::groupid() == 0;
    const bool output_group = warpgroup::groupid() == 1;
    const bool issue_warp = producer_group && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();

    auto copy_scale = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int WORD_BYTES = 4;
        constexpr int WORDS = sizeof(Tile) / WORD_BYTES;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int word = warpgroup::laneid(); word < WORDS;
             word += WARPGROUP_WARPS * WARP_THREADS) {
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + word * WORD_BYTES);
            asm volatile("st.shared.b32 [%0], %1;" ::
                         "r"(smem_base + word * WORD_BYTES), "r"(value));
        }
    };
    auto copy_v_scale_half = [&](int micro) {
        constexpr int WORDS = sizeof(typename G::v_sc_tile) / 4;
        const int key_tile = micro >> 1;
        const int key_half = micro & 1;
        const int scale_record =
            (((b * g.heads + h) * 2) * tiles + key_tile) * 32 * 16;
        const uint32_t smem_base =
            __cvta_generic_to_shared(&v_sc_smem.data[0]);
        for (int word = warpgroup::laneid(); word < WORDS;
             word += WARPGROUP_WARPS * WARP_THREADS) {
            const uint32_t packed = *reinterpret_cast<const uint32_t *>(
                g.v_sc + scale_record + word * 4);
            const uint32_t selected = key_half == 0
                ? packed & 0x0000ffffu
                : (packed >> 16) & 0x0000ffffu;
            asm volatile("st.shared.b32 [%0], %1;" ::
                         "r"(smem_base + word * 4), "r"(selected));
        }
    };

    if (producer_group) {
        const coord<typename G::q_tile> q_coord = {b, h, query_tile, 0};
        if (threadIdx.x == 0) {
            tma::expect_bytes(q_payload_done, sizeof(typename G::q_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                q_smem, g.q, q_coord, q_payload_done);
        }
        const uint8_t *q_sc_src = g.q_sc +
            (((b * tiles + query_tile) *
              (g.heads * C::QK_SCALE_CHUNKS) +
              h * C::QK_SCALE_CHUNKS) * 512);
        copy_scale(q_sc_smem, q_sc_src);
        warpgroup::sync(1);
        if (threadIdx.x == 0) {
            wait(q_payload_done, 0);
        }
        warpgroup::sync(1);
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(1);
    }

    const float scale_log2 = LOG2E_OVER_SQRT_D *
        g.q_sg[b * g.heads + h] * g.k_sg[b * g.heads + h];
    const bool zero_scale = scale_log2 == 0.0f;
    float row_max = base_types::constants<float>::neg_infty();
    float row_sum = 0.0f;
    unsigned long long qk_count = 0;
    unsigned long long pv_count = 0;

    #pragma unroll 1
    for (int micro = 0; micro < micro_iters; ++micro) {
        const int phase = micro & 1;
        if (producer_group) {
            if (micro > 0 && issue_warp) {
                wait(pv_src_done, (micro - 1) & 1);
                wait(pv_done, (micro - 1) & 1);
                tensor_after_thread_sync();
            }
            warpgroup::sync(1);

            const coord<typename G::k_tile> k_coord = {b, h, micro, 0};
            const coord<typename G::v_tile> v_coord = {b, h, 0, micro};
            if (threadIdx.x == 0) {
                tma::expect_bytes(k_payload_done, sizeof(typename G::k_tile));
                tma::expect_bytes(v_payload_done, sizeof(typename G::v_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    k_smem, g.k, k_coord, k_payload_done);
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    v_smem, g.v, v_coord, v_payload_done);
            }
            const uint8_t *k_sc_src = g.k_sc +
                (((b * (tiles * 2) + micro) *
                  (g.heads * C::QK_SCALE_CHUNKS) +
                  h * C::QK_SCALE_CHUNKS) * 512);
            copy_scale(k_sc_smem, k_sc_src);
            copy_v_scale_half(micro);
            warpgroup::sync(1);
            if (threadIdx.x == 0) {
                wait(k_payload_done, phase);
                wait(v_payload_done, phase);
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
        }

        if (issue_warp) {
            using T_AB = typename G::q_tile::T;
            using T_SAB = typename qk_scale_tt::T;
            using T_D = typename score_tt::T;
            constexpr uint32_t IDESC =
                detail::tcgen05::instruction_descriptor<
                    T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
            st_descriptor<typename G::q_tile, 0> q_desc(q_smem);
            st_descriptor<typename G::k_tile, 0> k_desc(k_smem);
            #pragma unroll
            for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                if (issue_lane) {
                    fp4pv_load_q_scale_chunk_ranklocal(
                        q_scale, q_sc_smem, chunk);
                    fp4pv_load_k_scale_chunk_ranklocal(
                        k_scale, k_sc_smem, chunk);
                    detail::tcgen05::commit<1>(qk_scale_done[chunk]);
                }
                wait(qk_scale_done[chunk], phase);
                tensor_after_thread_sync();
                tensor_before_thread_sync();
                if (issue_lane) {
                    if (chunk == 0) {
                        detail::tcgen05::template st_st<
                            T_AB, T_SAB, 0, 1, 16>(
                                score.addr, q_desc.chunk_descriptor(chunk),
                                k_desc.chunk_descriptor(chunk), q_scale.addr,
                                k_scale.addr, IDESC);
                    } else {
                        detail::tcgen05::template st_st<
                            T_AB, T_SAB, 1, 1, 16>(
                                score.addr, q_desc.chunk_descriptor(chunk),
                                k_desc.chunk_descriptor(chunk), q_scale.addr,
                                k_scale.addr, IDESC);
                    }
                }
            }
            if (issue_lane) {
                detail::tcgen05::commit<1>(qk_done);
                ++qk_count;
            }
        }

        float correction = 1.0f;
        if (output_group) {
            wait(qk_done, phase);
            tensor_after_thread_sync();
            const int row = warpgroup::laneid();
            const int query_col = query_tile * C::Mb + row;
            const uint32_t score_base = score.addr +
                ((warpgroup::warpid() * 32) << 16);
            float block_max[2];
            float tile_max = base_types::constants<float>::neg_infty();
            #pragma unroll
            for (int qid = 0; qid < 2; ++qid) {
                float2 scores[16];
                fp4pv_load_score_quarter_tmem(scores, score_base, qid);
                tensor_load_wait();
                float local_max = base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int key_col = micro * C::Nb + qid * 32 + pair * 2;
                    if (key_col > query_col) {
                        scores[pair].x =
                            base_types::constants<float>::neg_infty();
                    }
                    if (key_col + 1 > query_col) {
                        scores[pair].y =
                            base_types::constants<float>::neg_infty();
                    }
                    local_max = fmaxf(
                        local_max, fmaxf(scores[pair].x, scores[pair].y));
                }
                block_max[qid] = local_max;
                tile_max = fmaxf(tile_max, local_max);
            }

            const float old_max = row_max;
            row_max = fmaxf(row_max, tile_max);
            if (micro > 0 && isfinite(old_max) && isfinite(row_max) &&
                !zero_scale) {
                correction = fp4pv_exp2_approx(
                    (old_max - row_max) * scale_log2);
                option_b_rescale_output<C>(output, correction);
            }

            uint32_t packed_scales = 0;
            float tile_sum = 0.0f;
            #pragma unroll
            for (int qid = 0; qid < 2; ++qid) {
                uint8_t p_e8m0 = 0;
                if (isfinite(block_max[qid]) && isfinite(row_max)) {
                    const float block_scale_log2 =
                        FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                        (block_max[qid] - row_max) * scale_log2;
                    p_e8m0 =
                        fp4pv_log2_to_e8m0_floor_finite(block_scale_log2);
                }
                packed_scales |=
                    static_cast<uint32_t>(p_e8m0) << (qid * 8);
                const float p_log2_coeff = p_e8m0 == 0
                    ? 0.0f
                    : FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE -
                      static_cast<float>(static_cast<int>(p_e8m0) - 127);
                const float p_rcp_coeff =
                    fp4pv_exp2f_e8m0(p_e8m0) *
                    FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;
                const float bias = -row_max * scale_log2 + p_log2_coeff;
                const float2 scale2 = {scale_log2, scale_log2};
                const float2 bias2 = {bias, bias};
                float2 scores[16];
                fp4pv_load_score_quarter_tmem(scores, score_base, qid);
                tensor_load_wait();
                float2 pair_sum = {0.0f, 0.0f};
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int key_col = micro * C::Nb + qid * 32 + pair * 2;
                    if (key_col > query_col) {
                        scores[pair].x =
                            base_types::constants<float>::neg_infty();
                    }
                    if (key_col + 1 > query_col) {
                        scores[pair].y =
                            base_types::constants<float>::neg_infty();
                    }
                    if (!zero_scale) {
                        scores[pair] =
                            __ffma2_rn(scores[pair], scale2, bias2);
                    } else {
                        scores[pair].x = isfinite(scores[pair].x)
                            ? p_log2_coeff : scores[pair].x;
                        scores[pair].y = isfinite(scores[pair].y)
                            ? p_log2_coeff : scores[pair].y;
                    }
                    scores[pair].x = fp4pv_exp2_approx(scores[pair].x);
                    scores[pair].y = fp4pv_exp2_approx(scores[pair].y);
                    pair_sum = __fadd2_rn(pair_sum, scores[pair]);
                }
                tile_sum += (pair_sum.x + pair_sum.y) * p_rcp_coeff;
                uint32_t words[4];
                #pragma unroll
                for (int pack = 0; pack < 4; ++pack) {
                    words[pack] = p_e8m0 == 0
                        ? 0u
                        : fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(
                              &scores[pack * 4]);
                }
                fp4pv_store_quantized_scores_group_mxfp4(
                    p_smem, words, row, qid);
            }
            fp4pv_store_mxfp4_scale_word_swizzled(
                p_sc_smem, row, packed_scales);
            row_sum = row_sum * correction + tile_sum;
            warpgroup::sync(2);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(2);
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                arrive(p_ready);
            }
        }

        if (issue_warp) {
            wait(p_ready, phase);
            tensor_after_thread_sync();
            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(p_scale, p_sc_smem);
                fp4pv_load_mxnv_scale_async_ranklocal(v_scale, v_sc_smem);
                detail::tcgen05::commit<1>(pv_scale_done);
            }
            wait(pv_scale_done, phase);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
            if (issue_lane) {
                if (micro == 0) {
                    fp4pv_mm_p_stage_ranklocal_ABt<C>(
                        output, p_smem, v_smem, p_scale, v_scale,
                        pv_src_done);
                } else {
                    fp4pv_mma_p_stage_ranklocal_ABt<C>(
                        output, p_smem, v_smem, p_scale, v_scale,
                        pv_src_done);
                }
                tensor_commit<1>(pv_done);
                ++pv_count;
            }
        }
    }

    if (output_group) {
        wait(pv_done, (micro_iters - 1) & 1);
        tensor_after_thread_sync();
        const int row = warpgroup::laneid();
        const bool valid = row_sum > 0.0f && isfinite(row_sum);
        const float output_scale = valid ? OUTPUT_CORRECTION / row_sum : 0.0f;
        const float2 output_scale2 = {output_scale, output_scale};
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
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
                const bf16_2 packed = __float22bfloat162_rn(
                    __fmul2_rn(values[pair], output_scale2));
                const int out_offset =
                    ((((b * g.seqlen + query_tile * C::Mb + row) *
                       g.heads + h) * C::Dvo) + col + pair * 2);
                *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
            }
        }
        float lse_value = base_types::constants<float>::neg_infty();
        if (valid) {
            float log2_row_sum;
            asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum)
                : "f"(row_sum));
            constexpr float LN2 = 0.693147180559945f;
            lse_value = (row_max * scale_log2 + log2_row_sum) * LN2;
        }
        g.lse[((b * g.heads + h) * g.seqlen) +
              query_tile * C::Mb + row] = lse_value;
        warpgroup::sync(2);
        if constexpr (ENABLE_TRACE) {
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                trace_store(8, globaltimer());
            }
        }
    }

    __syncthreads();
    if constexpr (ENABLE_TRACE) {
        if (threadIdx.x == 0) {
            trace_store(9, qk_count);
            trace_store(10, pv_count);
            trace_store(11, micro_iters);
            trace_store(12, micro_iters);
            trace_store(13, globaltimer());
            __threadfence_system();
            trace_store(15, 1);
        }
    }
}

#endif

#if TK_FA4_TMEM224_PROBE_BUILD

template <typename C>
struct globals_mxfp4_tmem224_pair {
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
        add(2 * (int)sizeof(q_tile));
        add(2 * (int)sizeof(q_sc_tile));
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
    float *lse_aux;
    float *workspace;
    unsigned long long *trace;
    int batch;
    int heads;
    int seqlen;
    int trace_stride;

    __host__ __inline__ dim3 grid() const {
        const int query_tiles = seqlen / C::Mb;
        return dim3(batch * heads * ((query_tiles + 1) / 2));
    }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() const {
        return DYNAMIC_SHARED_MEMORY;
    }
};

template <typename Tile>
__device__ __forceinline__ void tmem224_copy_scale_tile(
    Tile &tile, const uint8_t *source) {
    constexpr int WORD_BYTES = 4;
    constexpr int WORDS = sizeof(Tile) / WORD_BYTES;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int word = warpgroup::laneid(); word < WORDS;
         word += WARPGROUP_WARPS * WARP_THREADS) {
        const uint32_t value = *reinterpret_cast<const uint32_t *>(
            source + word * WORD_BYTES);
        asm volatile("st.shared.b32 [%0], %1;" ::
                     "r"(smem_base + word * WORD_BYTES), "r"(value));
    }
}

template <typename Tile>
__device__ __forceinline__ void tmem224_copy_scale_half(
    Tile &tile, const uint8_t *source, int key_half) {
    constexpr int WORDS = sizeof(Tile) / 4;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int word = warpgroup::laneid(); word < WORDS;
         word += WARPGROUP_WARPS * WARP_THREADS) {
        const uint32_t packed = *reinterpret_cast<const uint32_t *>(
            source + word * 4);
        const uint32_t selected = key_half == 0
            ? packed & 0x0000ffffu
            : (packed >> 16) & 0x0000ffffu;
        asm volatile("st.shared.b32 [%0], %1;" ::
                     "r"(smem_base + word * 4), "r"(selected));
    }
}

template <typename C, typename OutputTT>
__device__ __forceinline__ void tmem224_rescale_output(
    OutputTT &output, float correction) {
    const float2 correction2 = {correction, correction};
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
    tensor_after_thread_sync();
}

template <typename C, typename G, typename OutputTT>
__device__ __forceinline__ void tmem224_store_output(
    const G &g, OutputTT &output, int b, int h, int query_tile,
    int output_column_base, float row_sum) {
    constexpr float OUTPUT_CORRECTION =
        1.0f / (36.0f * FP4PV_MXFP4_ONLINE_P_SCALE);
    const int row = warpgroup::laneid();
    const bool valid = row_sum > 0.0f && isfinite(row_sum);
    const float output_scale = valid ? OUTPUT_CORRECTION / row_sum : 0.0f;
    const float2 output_scale2 = {output_scale, output_scale};
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
            const bf16_2 packed = __float22bfloat162_rn(
                __fmul2_rn(values[pair], output_scale2));
            const int out_offset =
                ((((b * g.seqlen + query_tile * C::Mb + row) *
                    g.heads + h) * 128) + output_column_base +
                 col + pair * 2);
            *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
        }
    }
}

template <typename G>
__device__ __forceinline__ void tmem224_store_lse(
    const G &g, float *destination, int b, int h, int query_tile,
    float row_max, float row_sum, float scale_log2) {
    const int row = warpgroup::laneid();
    const bool valid = row_sum > 0.0f && isfinite(row_sum);
    float value = base_types::constants<float>::neg_infty();
    if (valid) {
        float log2_row_sum;
        asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum)
            : "f"(row_sum));
        constexpr float LN2 = 0.693147180559945f;
        value = (row_max * scale_log2 + log2_row_sum) * LN2;
    }
    destination[((b * g.heads + h) * g.seqlen) +
                query_tile * 128 + row] = value;
}

enum tmem224_external_mode : int {
    TMEM224_EXTERNAL_FP32_SMEM = 0,
    TMEM224_EXTERNAL_FP32_GLOBAL = 1,
    TMEM224_EXTERNAL_FP32_REGISTER = 2,
    TMEM224_EXTERNAL_BF16_SMEM = 3,
    TMEM224_EXTERNAL_FP16_SMEM = 4,
};

template <int EXTERNAL_MODE, typename OutputTT, typename ExternalTile>
__device__ __forceinline__ void tmem224_exchange_output_state(
    OutputTT &output, ExternalTile *external_smem, float *global_workspace,
    float *external_registers, int task, bool save_outgoing,
    bool restore_incoming) {
    static_assert(OutputTT::rows == 128 && OutputTT::cols == 128);
    const int row = warpgroup::laneid();
    #pragma unroll
    for (int col = 0; col < 128; col += 16) {
        float2 outgoing[8];
        float2 incoming[8];
        const uint32_t address = output.addr +
            ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(outgoing[0].x), "=f"(outgoing[0].y),
              "=f"(outgoing[1].x), "=f"(outgoing[1].y),
              "=f"(outgoing[2].x), "=f"(outgoing[2].y),
              "=f"(outgoing[3].x), "=f"(outgoing[3].y),
              "=f"(outgoing[4].x), "=f"(outgoing[4].y),
              "=f"(outgoing[5].x), "=f"(outgoing[5].y),
              "=f"(outgoing[6].x), "=f"(outgoing[6].y),
              "=f"(outgoing[7].x), "=f"(outgoing[7].y)
            : "r"(address));
        tensor_load_wait();
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const int element = col + pair * 2;
            if constexpr (EXTERNAL_MODE == TMEM224_EXTERNAL_FP32_SMEM) {
                float2 *slot = reinterpret_cast<float2 *>(
                    &(*external_smem)[int2{row, element}]);
                if (restore_incoming) incoming[pair] = *slot;
                if (save_outgoing) *slot = outgoing[pair];
            } else if constexpr (
                EXTERNAL_MODE == TMEM224_EXTERNAL_FP32_GLOBAL) {
                float2 *slot = reinterpret_cast<float2 *>(
                    global_workspace +
                    ((unsigned long long)task * 128 + row) * 128 + element);
                if (restore_incoming) incoming[pair] = *slot;
                if (save_outgoing) *slot = outgoing[pair];
            } else if constexpr (
                EXTERNAL_MODE == TMEM224_EXTERNAL_FP32_REGISTER) {
                if (restore_incoming) {
                    incoming[pair] = {
                        external_registers[element],
                        external_registers[element + 1]};
                }
                if (save_outgoing) {
                    external_registers[element] = outgoing[pair].x;
                    external_registers[element + 1] = outgoing[pair].y;
                }
            } else if constexpr (
                EXTERNAL_MODE == TMEM224_EXTERNAL_BF16_SMEM) {
                bf16_2 *slot = reinterpret_cast<bf16_2 *>(
                    &(*external_smem)[int2{row, element}]);
                if (restore_incoming) incoming[pair] = __bfloat1622float2(*slot);
                if (save_outgoing) {
                    *slot = __float22bfloat162_rn(outgoing[pair]);
                }
            } else {
                static_assert(EXTERNAL_MODE == TMEM224_EXTERNAL_FP16_SMEM);
                constexpr float CHECKPOINT_SCALE =
                    1.0f / (36.0f * FP4PV_MXFP4_ONLINE_P_SCALE);
                constexpr float CHECKPOINT_SCALE_INVERSE =
                    36.0f * FP4PV_MXFP4_ONLINE_P_SCALE;
                const float2 checkpoint_scale = {
                    CHECKPOINT_SCALE, CHECKPOINT_SCALE};
                const float2 checkpoint_scale_inverse = {
                    CHECKPOINT_SCALE_INVERSE, CHECKPOINT_SCALE_INVERSE};
                half_2 *slot = reinterpret_cast<half_2 *>(
                    &(*external_smem)[int2{row, element}]);
                if (restore_incoming) {
                    incoming[pair] = __fmul2_rn(
                        __half22float2(*slot), checkpoint_scale_inverse);
                }
                if (save_outgoing) {
                    *slot = __float22half2_rn(
                        __fmul2_rn(outgoing[pair], checkpoint_scale));
                }
            }
        }
        if (restore_incoming) {
            asm volatile(
                "{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                :: "f"(incoming[0].x), "f"(incoming[0].y),
                   "f"(incoming[1].x), "f"(incoming[1].y),
                   "f"(incoming[2].x), "f"(incoming[2].y),
                   "f"(incoming[3].x), "f"(incoming[3].y),
                   "f"(incoming[4].x), "f"(incoming[4].y),
                   "f"(incoming[5].x), "f"(incoming[5].y),
                   "f"(incoming[6].x), "f"(incoming[6].y),
                   "f"(incoming[7].x), "f"(incoming[7].y),
                   "r"(address));
        }
    }
    if (restore_incoming) {
        tensor_store_wait();
        tensor_after_thread_sync();
    }
}

template <typename C, typename G, typename ScoreTT, typename OutputTT,
          typename QScaleTT, typename PVScaleTT>
__device__ __forceinline__ void tmem224_run_loaded_query_micro(
    const G &g, int b, int h, int query_tile, int absolute_micro,
    bool initialize_output, typename G::q_tile &q_smem,
    typename G::q_sc_tile &q_sc_smem, typename G::k_tile &k_smem,
    typename G::k_sc_tile &k_sc_smem, typename G::p_tile &p_smem,
    typename G::p_sc_tile &p_sc_smem, typename G::v_tile &v_smem,
    typename G::v_sc_tile &v_sc_smem, ScoreTT &score, OutputTT &output,
    QScaleTT &q_scale, QScaleTT &k_scale, PVScaleTT &p_scale,
    PVScaleTT &v_scale,
    semaphore (&qk_scale_done)[C::QK_SCALE_CHUNKS],
    semaphore &qk_done, semaphore &p_ready, semaphore &pv_scale_done,
    semaphore &pv_src_done, semaphore &pv_done, int event,
    float &row_max, float &row_sum, unsigned long long &qk_count,
    unsigned long long &pv_count) {
    constexpr float LOG2E_OVER_SQRT_D = 0.10408340855860843f;
    const bool producer_group = warpgroup::groupid() == 0;
    const bool output_group = warpgroup::groupid() == 1;
    const bool issue_warp = producer_group && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const int phase = event & 1;
    const float scale_log2 = LOG2E_OVER_SQRT_D *
        g.q_sg[b * g.heads + h] * g.k_sg[b * g.heads + h];
    const bool zero_scale = scale_log2 == 0.0f;

    if (issue_warp) {
        using T_AB = typename G::q_tile::T;
        using T_SAB = typename QScaleTT::T;
        using T_D = typename ScoreTT::T;
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
        const int query_column = query_tile * C::Mb + row;
        const uint32_t score_base = score.addr +
            ((warpgroup::warpid() * 32) << 16);
        float block_max[2];
        float tile_max = base_types::constants<float>::neg_infty();
        #pragma unroll
        for (int quarter = 0; quarter < 2; ++quarter) {
            float2 scores[16];
            fp4pv_load_score_quarter_tmem(scores, score_base, quarter);
            tensor_load_wait();
            float local_max = base_types::constants<float>::neg_infty();
            #pragma unroll
            for (int pair = 0; pair < 16; ++pair) {
                const int key_column = absolute_micro * C::Nb +
                    quarter * 32 + pair * 2;
                if (key_column > query_column) {
                    scores[pair].x =
                        base_types::constants<float>::neg_infty();
                }
                if (key_column + 1 > query_column) {
                    scores[pair].y =
                        base_types::constants<float>::neg_infty();
                }
                local_max = fmaxf(
                    local_max, fmaxf(scores[pair].x, scores[pair].y));
            }
            block_max[quarter] = local_max;
            tile_max = fmaxf(tile_max, local_max);
        }

        const float old_max = row_max;
        row_max = fmaxf(row_max, tile_max);
        if (!initialize_output && isfinite(old_max) && isfinite(row_max) &&
            !zero_scale) {
            correction = fp4pv_exp2_approx(
                (old_max - row_max) * scale_log2);
            tmem224_rescale_output<C>(output, correction);
        }

        uint32_t packed_scales = 0;
        float tile_sum = 0.0f;
        #pragma unroll
        for (int quarter = 0; quarter < 2; ++quarter) {
            uint8_t p_e8m0 = 0;
            if (isfinite(block_max[quarter]) && isfinite(row_max)) {
                const float block_scale_log2 =
                    FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                    (block_max[quarter] - row_max) * scale_log2;
                p_e8m0 = fp4pv_log2_to_e8m0_floor_finite(
                    block_scale_log2);
            }
            packed_scales |=
                static_cast<uint32_t>(p_e8m0) << (quarter * 8);
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
            fp4pv_load_score_quarter_tmem(scores, score_base, quarter);
            tensor_load_wait();
            float2 pair_sum = {0.0f, 0.0f};
            #pragma unroll
            for (int pair = 0; pair < 16; ++pair) {
                const int key_column = absolute_micro * C::Nb +
                    quarter * 32 + pair * 2;
                if (key_column > query_column) {
                    scores[pair].x =
                        base_types::constants<float>::neg_infty();
                }
                if (key_column + 1 > query_column) {
                    scores[pair].y =
                        base_types::constants<float>::neg_infty();
                }
                if (!zero_scale) {
                    scores[pair] = __ffma2_rn(scores[pair], scale2, bias2);
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
                p_smem, words, row, quarter);
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
            if (initialize_output) {
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

template <typename C, typename G, typename ScoreTT, typename OutputTT,
          typename QScaleTT, typename PVScaleTT>
__device__ __forceinline__ void tmem224_run_query_walk(
    const G &g, int b, int h, int tiles, int query_tile, int dvo_half,
    int max_key_tiles, typename G::q_tile &q_smem,
    typename G::q_sc_tile &q_sc_smem, typename G::k_tile &k_smem,
    typename G::k_sc_tile &k_sc_smem, typename G::p_tile &p_smem,
    typename G::p_sc_tile &p_sc_smem, typename G::v_tile &v_smem,
    typename G::v_sc_tile &v_sc_smem, ScoreTT &score, OutputTT &output,
    QScaleTT &q_scale, QScaleTT &k_scale, PVScaleTT &p_scale,
    PVScaleTT &v_scale, semaphore &k_payload_done,
    semaphore &v_payload_done,
    semaphore (&qk_scale_done)[C::QK_SCALE_CHUNKS],
    semaphore &qk_done, semaphore &p_ready, semaphore &pv_scale_done,
    semaphore &pv_src_done, semaphore &pv_done, int &event,
    float &final_row_max, float &final_row_sum,
    unsigned long long &qk_count, unsigned long long &pv_count) {
    constexpr float LOG2E_OVER_SQRT_D = 0.10408340855860843f;
    const bool producer_group = warpgroup::groupid() == 0;
    const bool output_group = warpgroup::groupid() == 1;
    const bool issue_warp = producer_group && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const float scale_log2 = LOG2E_OVER_SQRT_D *
        g.q_sg[b * g.heads + h] * g.k_sg[b * g.heads + h];
    const bool zero_scale = scale_log2 == 0.0f;
    float row_max = base_types::constants<float>::neg_infty();
    float row_sum = 0.0f;
    const int full_micro_iters = 2 * (query_tile + 1);
    const int micro_iters = max_key_tiles == 0
        ? full_micro_iters : min(full_micro_iters, 2 * max_key_tiles);

    #pragma unroll 1
    for (int local_micro = 0; local_micro < micro_iters;
         ++local_micro, ++event) {
        const int phase = event & 1;
        if (producer_group) {
            if (local_micro > 0 && issue_warp) {
                wait(pv_src_done, (event - 1) & 1);
                wait(pv_done, (event - 1) & 1);
                tensor_after_thread_sync();
            }
            warpgroup::sync(1);

            const coord<typename G::k_tile> k_coord = {
                b, h, local_micro, 0};
            const coord<typename G::v_tile> v_coord = [](
                int batch, int head, int half, int micro) {
                if constexpr (C::Dvo == 64) {
                    return coord<typename G::v_tile>{
                        batch, 2 * head + half, 0, micro};
                } else {
                    return coord<typename G::v_tile>{batch, head, 0, micro};
                }
            }(b, h, dvo_half, local_micro);
            if (threadIdx.x == 0) {
                tma::expect_bytes(k_payload_done, sizeof(typename G::k_tile));
                tma::expect_bytes(v_payload_done, sizeof(typename G::v_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    k_smem, g.k, k_coord, k_payload_done);
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    v_smem, g.v, v_coord, v_payload_done);
            }
            const uint8_t *k_sc_source = g.k_sc +
                (((b * (tiles * 2) + local_micro) *
                  (g.heads * C::QK_SCALE_CHUNKS) +
                  h * C::QK_SCALE_CHUNKS) * 512);
            tmem224_copy_scale_tile(k_sc_smem, k_sc_source);
            const int key_tile = local_micro >> 1;
            const int key_half = local_micro & 1;
            const int v_record = C::Dvo == 64
                ? ((b * g.heads + h) * 2 + dvo_half)
                : ((b * g.heads + h) * 2);
            const uint8_t *v_sc_source = g.v_sc +
                ((v_record * tiles + key_tile) *
                 sizeof(typename G::v_sc_tile));
            tmem224_copy_scale_half(v_sc_smem, v_sc_source, key_half);
            warpgroup::sync(1);
            if (threadIdx.x == 0) {
                wait(k_payload_done, phase);
                wait(v_payload_done, phase);
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
        }

        tmem224_run_loaded_query_micro<C>(
            g, b, h, query_tile, local_micro, local_micro == 0,
            q_smem, q_sc_smem, k_smem, k_sc_smem, p_smem, p_sc_smem,
            v_smem, v_sc_smem, score, output, q_scale, k_scale, p_scale,
            v_scale, qk_scale_done, qk_done, p_ready, pv_scale_done,
            pv_src_done, pv_done, event, row_max, row_sum, qk_count,
            pv_count);
#if 0
        if (issue_warp) {
            using T_AB = typename G::q_tile::T;
            using T_SAB = typename QScaleTT::T;
            using T_D = typename ScoreTT::T;
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
            const int query_column = query_tile * C::Mb + row;
            const uint32_t score_base = score.addr +
                ((warpgroup::warpid() * 32) << 16);
            float block_max[2];
            float tile_max = base_types::constants<float>::neg_infty();
            #pragma unroll
            for (int quarter = 0; quarter < 2; ++quarter) {
                float2 scores[16];
                fp4pv_load_score_quarter_tmem(scores, score_base, quarter);
                tensor_load_wait();
                float local_max = base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int key_column = local_micro * C::Nb +
                        quarter * 32 + pair * 2;
                    if (key_column > query_column) {
                        scores[pair].x =
                            base_types::constants<float>::neg_infty();
                    }
                    if (key_column + 1 > query_column) {
                        scores[pair].y =
                            base_types::constants<float>::neg_infty();
                    }
                    local_max = fmaxf(
                        local_max, fmaxf(scores[pair].x, scores[pair].y));
                }
                block_max[quarter] = local_max;
                tile_max = fmaxf(tile_max, local_max);
            }

            const float old_max = row_max;
            row_max = fmaxf(row_max, tile_max);
            if (local_micro > 0 && isfinite(old_max) &&
                isfinite(row_max) && !zero_scale) {
                correction = fp4pv_exp2_approx(
                    (old_max - row_max) * scale_log2);
                tmem224_rescale_output<C>(output, correction);
            }

            uint32_t packed_scales = 0;
            float tile_sum = 0.0f;
            #pragma unroll
            for (int quarter = 0; quarter < 2; ++quarter) {
                uint8_t p_e8m0 = 0;
                if (isfinite(block_max[quarter]) && isfinite(row_max)) {
                    const float block_scale_log2 =
                        FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                        (block_max[quarter] - row_max) * scale_log2;
                    p_e8m0 = fp4pv_log2_to_e8m0_floor_finite(
                        block_scale_log2);
                }
                packed_scales |=
                    static_cast<uint32_t>(p_e8m0) << (quarter * 8);
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
                fp4pv_load_score_quarter_tmem(scores, score_base, quarter);
                tensor_load_wait();
                float2 pair_sum = {0.0f, 0.0f};
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int key_column = local_micro * C::Nb +
                        quarter * 32 + pair * 2;
                    if (key_column > query_column) {
                        scores[pair].x =
                            base_types::constants<float>::neg_infty();
                    }
                    if (key_column + 1 > query_column) {
                        scores[pair].y =
                            base_types::constants<float>::neg_infty();
                    }
                    if (!zero_scale) {
                        scores[pair] = __ffma2_rn(
                            scores[pair], scale2, bias2);
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
                    p_smem, words, row, quarter);
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
                if (local_micro == 0) {
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
#endif
    }

    if (micro_iters > 0) {
        const int final_phase = (event - 1) & 1;
        if (issue_warp) {
            wait(pv_src_done, final_phase);
            wait(pv_done, final_phase);
            tensor_after_thread_sync();
        }
        if (output_group) {
            wait(pv_done, final_phase);
            tensor_after_thread_sync();
        }
    }
    __syncthreads();
    final_row_max = row_max;
    final_row_sum = row_sum;
}

template <typename C, typename G, typename ScoreTT, typename OutputTT,
          typename QScaleTT, typename PVScaleTT>
__device__ __forceinline__ void tmem224_run_paired_d64_walk(
    const G &g, int b, int h, int tiles, const int (&query_tile)[2],
    const bool (&query_active)[2], int dvo_half, int max_key_tiles,
    typename G::q_tile (&q_smem)[2],
    typename G::q_sc_tile (&q_sc_smem)[2], typename G::k_tile &k_smem,
    typename G::k_sc_tile &k_sc_smem, typename G::p_tile &p_smem,
    typename G::p_sc_tile &p_sc_smem, typename G::v_tile &v_smem,
    typename G::v_sc_tile &v_sc_smem, ScoreTT &score,
    OutputTT (&output)[2], QScaleTT &q_scale, QScaleTT &k_scale,
    PVScaleTT &p_scale, PVScaleTT &v_scale,
    semaphore &k_payload_done, semaphore &v_payload_done,
    semaphore (&qk_scale_done)[C::QK_SCALE_CHUNKS],
    semaphore &qk_done, semaphore &p_ready, semaphore &pv_scale_done,
    semaphore &pv_src_done, semaphore &pv_done, int &event,
    float (&final_row_max)[2], float (&final_row_sum)[2],
    unsigned long long &kv_load_count, unsigned long long &qk_count,
    unsigned long long &pv_count) {
    const bool producer_group = warpgroup::groupid() == 0;
    const bool output_group = warpgroup::groupid() == 1;
    const bool issue_warp = producer_group && warpgroup::warpid() == 0;
    const int max_query_tile = query_active[1]
        ? query_tile[1] : query_tile[0];
    const int full_micro_iters = 2 * (max_query_tile + 1);
    const int micro_iters = max_key_tiles == 0
        ? full_micro_iters : min(full_micro_iters, 2 * max_key_tiles);
    float row_max[2] = {
        base_types::constants<float>::neg_infty(),
        base_types::constants<float>::neg_infty()};
    float row_sum[2] = {0.0f, 0.0f};
    int update_count[2] = {0, 0};

    #pragma unroll 1
    for (int micro = 0; micro < micro_iters; ++micro) {
        const int payload_phase = micro & 1;
        if (producer_group) {
            if (event > 0 && issue_warp) {
                wait(pv_src_done, (event - 1) & 1);
                wait(pv_done, (event - 1) & 1);
                tensor_after_thread_sync();
            }
            warpgroup::sync(1);
            const coord<typename G::k_tile> k_coord = {b, h, micro, 0};
            const coord<typename G::v_tile> v_coord = {
                b, 2 * h + dvo_half, 0, micro};
            if (threadIdx.x == 0) {
                tma::expect_bytes(k_payload_done, sizeof(typename G::k_tile));
                tma::expect_bytes(v_payload_done, sizeof(typename G::v_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    k_smem, g.k, k_coord, k_payload_done);
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    v_smem, g.v, v_coord, v_payload_done);
                ++kv_load_count;
            }
            const uint8_t *k_sc_source = g.k_sc +
                (((b * (tiles * 2) + micro) *
                  (g.heads * C::QK_SCALE_CHUNKS) +
                  h * C::QK_SCALE_CHUNKS) * 512);
            tmem224_copy_scale_tile(k_sc_smem, k_sc_source);
            const int key_tile = micro >> 1;
            const int key_half = micro & 1;
            const int v_record = ((b * g.heads + h) * 2 + dvo_half);
            const uint8_t *v_sc_source = g.v_sc +
                ((v_record * tiles + key_tile) *
                 sizeof(typename G::v_sc_tile));
            tmem224_copy_scale_half(v_sc_smem, v_sc_source, key_half);
            warpgroup::sync(1);
            if (threadIdx.x == 0) {
                wait(k_payload_done, payload_phase);
                wait(v_payload_done, payload_phase);
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
        }

        bool issued_in_micro = false;
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            const bool consumes = query_active[stage] &&
                micro < 2 * (query_tile[stage] + 1);
            if (!consumes) {
                continue;
            }
            if (issued_in_micro && producer_group) {
                if (issue_warp) {
                    wait(pv_src_done, (event - 1) & 1);
                    wait(pv_done, (event - 1) & 1);
                    tensor_after_thread_sync();
                }
                warpgroup::sync(1);
            }
            tmem224_run_loaded_query_micro<C>(
                g, b, h, query_tile[stage], micro,
                update_count[stage] == 0, q_smem[stage],
                q_sc_smem[stage], k_smem, k_sc_smem, p_smem, p_sc_smem,
                v_smem, v_sc_smem, score, output[stage], q_scale, k_scale,
                p_scale, v_scale, qk_scale_done, qk_done, p_ready,
                pv_scale_done, pv_src_done, pv_done, event, row_max[stage],
                row_sum[stage], qk_count, pv_count);
            ++update_count[stage];
            ++event;
            issued_in_micro = true;
        }
    }

    if (event > 0) {
        const int final_phase = (event - 1) & 1;
        if (issue_warp) {
            wait(pv_src_done, final_phase);
            wait(pv_done, final_phase);
            tensor_after_thread_sync();
        }
        if (output_group) {
            wait(pv_done, final_phase);
            tensor_after_thread_sync();
        }
    }
    __syncthreads();
    #pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
        final_row_max[stage] = row_max[stage];
        final_row_sum[stage] = row_sum[stage];
    }
}

template <typename C, typename G>
__device__ __forceinline__ void tmem224_load_query_pair(
    const G &g, int b, int h, int tiles, const int (&query_tile)[2],
    typename G::q_tile (&q_smem)[2],
    typename G::q_sc_tile (&q_sc_smem)[2],
    semaphore (&q_payload_done)[2]) {
    if (warpgroup::groupid() == 0) {
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            const coord<typename G::q_tile> q_coord = {
                b, h, query_tile[stage], 0};
            if (threadIdx.x == 0) {
                tma::expect_bytes(
                    q_payload_done[stage], sizeof(typename G::q_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    q_smem[stage], g.q, q_coord, q_payload_done[stage]);
            }
            const uint8_t *q_sc_source = g.q_sc +
                (((b * tiles + query_tile[stage]) *
                  (g.heads * C::QK_SCALE_CHUNKS) +
                  h * C::QK_SCALE_CHUNKS) * 512);
            tmem224_copy_scale_tile(q_sc_smem[stage], q_sc_source);
        }
        warpgroup::sync(1);
        if (threadIdx.x == 0) {
            wait(q_payload_done[0], 0);
            wait(q_payload_done[1], 0);
        }
        warpgroup::sync(1);
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(1);
    }
    __syncthreads();
}

template <typename C, int MIN_BLOCKS_PER_SM, int MAX_KEY_TILES,
          bool ENABLE_TRACE>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_tmem224_d64_stripmine(
    const __grid_constant__ globals_mxfp4_tmem224_pair<C> g) {
    using G = globals_mxfp4_tmem224_pair<C>;
    using score_tt = tt<float, C::Mb, C::Nb>;
    using output_tt = tt<float, C::Mb, C::Dvo>;
    using qk_scale_tt = full_tt_fp8e4m3<16>;
    using pv_scale_tt = full_tt_fp8e8m0<16>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT0_BASE = 64;
    constexpr int OUTPUT1_BASE = 128;
    constexpr int SCALE_A_BASE = 192;
    constexpr int SCALE_B_BASE = 208;
    constexpr int LOGICAL_END = 224;
    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2 &&
                  C::Mb == 128 && C::Nb == 64 && C::Dqk == 192 &&
                  C::Dvo == 64 && C::CLUSTER_SIZE == 1);
    static_assert(LOGICAL_END <= tensor_allocator<2, 1>::cols &&
                  tensor_allocator<2, 1>::cols == 256);

    tensor_allocator<2, 1> tm_alloc{};
    score_tt score = tm_alloc.template allocate<score_tt>(SCORE_BASE);
    output_tt output[2] = {
        tm_alloc.template allocate<output_tt>(OUTPUT0_BASE),
        tm_alloc.template allocate<output_tt>(OUTPUT1_BASE)};
    qk_scale_tt q_scale =
        tm_alloc.template allocate<qk_scale_tt>(SCALE_A_BASE);
    qk_scale_tt k_scale =
        tm_alloc.template allocate<qk_scale_tt>(SCALE_B_BASE);
    pv_scale_tt p_scale =
        tm_alloc.template allocate<pv_scale_tt>(SCALE_A_BASE);
    pv_scale_tt v_scale =
        tm_alloc.template allocate<pv_scale_tt>(SCALE_B_BASE);

    const int task = blockIdx.x;
    const int tiles = g.seqlen / 128;
    const int pairs_per_head = (tiles + 1) / 2;
    const int bh = task / pairs_per_head;
    const int pair = task - bh * pairs_per_head;
    const int b = bh / g.heads;
    const int h = bh - b * g.heads;
    const int query_tile[2] = {
        2 * pair, min(2 * pair + 1, tiles - 1)};
    const bool query_active[2] = {true, 2 * pair + 1 < tiles};

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::q_tile (&q_smem)[2] =
        allocator.allocate<typename G::q_tile, 2>();
    typename G::q_sc_tile (&q_sc_smem)[2] =
        allocator.allocate<typename G::q_sc_tile, 2>();
    typename G::k_tile &k_smem = allocator.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem =
        allocator.allocate<typename G::k_sc_tile>();
    typename G::p_tile &p_smem = allocator.allocate<typename G::p_tile>();
    typename G::p_sc_tile &p_sc_smem =
        allocator.allocate<typename G::p_sc_tile>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::v_sc_tile &v_sc_smem =
        allocator.allocate<typename G::v_sc_tile>();

    __shared__ semaphore q_payload_done[2];
    __shared__ semaphore k_payload_done, v_payload_done;
    __shared__ semaphore qk_scale_done[C::QK_SCALE_CHUNKS];
    __shared__ semaphore qk_done, p_ready, pv_scale_done;
    __shared__ semaphore pv_src_done, pv_done;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(q_payload_done[stage], 0, 1);
        }
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

    auto now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if constexpr (ENABLE_TRACE) {
            if (field < g.trace_stride) {
                g.trace[(unsigned long long)task * g.trace_stride + field] =
                    value;
            }
        }
    };
    if constexpr (ENABLE_TRACE) {
        if (threadIdx.x == 0) {
            trace_store(0, task); trace_store(1, kittens::smid());
            trace_store(2, tm_alloc.addr); trace_store(3, MIN_BLOCKS_PER_SM);
            trace_store(4, pair); trace_store(5, h); trace_store(6, now());
            trace_store(15, 0);
        }
    }

    tmem224_load_query_pair<C>(
        g, b, h, tiles, query_tile, q_smem, q_sc_smem, q_payload_done);
    int event = 0;
    unsigned long long kv_load_count = 0, qk_count = 0, pv_count = 0;
    #pragma unroll
    for (int dvo_half = 0; dvo_half < 2; ++dvo_half) {
        float row_max[2] = {
            base_types::constants<float>::neg_infty(),
            base_types::constants<float>::neg_infty()};
        float row_sum[2] = {0.0f, 0.0f};
        tmem224_run_paired_d64_walk<C>(
            g, b, h, tiles, query_tile, query_active, dvo_half,
            MAX_KEY_TILES, q_smem, q_sc_smem, k_smem, k_sc_smem,
            p_smem, p_sc_smem, v_smem, v_sc_smem, score, output,
            q_scale, k_scale, p_scale, v_scale, k_payload_done,
            v_payload_done, qk_scale_done, qk_done, p_ready,
            pv_scale_done, pv_src_done, pv_done, event, row_max, row_sum,
            kv_load_count, qk_count, pv_count);
        if (warpgroup::groupid() == 1) {
            #pragma unroll
            for (int stage = 0; stage < 2; ++stage) {
                if (query_active[stage]) {
                    tmem224_store_output<C>(
                        g, output[stage], b, h, query_tile[stage],
                        dvo_half * 64, row_sum[stage]);
                    const float scale_log2 = 0.10408340855860843f *
                        g.q_sg[b * g.heads + h] *
                        g.k_sg[b * g.heads + h];
                    tmem224_store_lse(
                        g, dvo_half == 0 ? g.lse : g.lse_aux,
                        b, h, query_tile[stage], row_max[stage],
                        row_sum[stage], scale_log2);
                }
            }
            warpgroup::sync(2);
        }
        __syncthreads();
    }

    if constexpr (ENABLE_TRACE) {
        if (threadIdx.x == 0) {
            trace_store(7, now()); trace_store(8, qk_count);
            trace_store(9, pv_count); trace_store(10, event);
            trace_store(11, LOGICAL_END); trace_store(12, 256);
            trace_store(13, query_active[1]);
            trace_store(14, kv_load_count); __threadfence_system();
            trace_store(15, 1);
        }
    }
}

template <typename C, int MIN_BLOCKS_PER_SM, bool ENABLE_TRACE>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_tmem224_serial_qpair(
    const __grid_constant__ globals_mxfp4_tmem224_pair<C> g) {
    using G = globals_mxfp4_tmem224_pair<C>;
    using score_tt = tt<float, C::Mb, C::Nb>;
    using output_tt = tt<float, C::Mb, C::Dvo>;
    using qk_scale_tt = full_tt_fp8e4m3<16>;
    using pv_scale_tt = full_tt_fp8e8m0<16>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = 64;
    constexpr int SCALE_A_BASE = 192;
    constexpr int SCALE_B_BASE = 208;
    constexpr int LOGICAL_END = 224;
    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2 &&
                  C::Mb == 128 && C::Nb == 64 && C::Dqk == 192 &&
                  C::Dvo == 128 && C::CLUSTER_SIZE == 1);
    static_assert(LOGICAL_END <= tensor_allocator<2, 1>::cols &&
                  tensor_allocator<2, 1>::cols == 256);

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
    const int tiles = g.seqlen / 128;
    const int pairs_per_head = (tiles + 1) / 2;
    const int bh = task / pairs_per_head;
    const int pair = task - bh * pairs_per_head;
    const int b = bh / g.heads;
    const int h = bh - b * g.heads;
    const int query_tile[2] = {
        2 * pair, min(2 * pair + 1, tiles - 1)};
    const bool query_active[2] = {true, 2 * pair + 1 < tiles};

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::q_tile (&q_smem)[2] =
        allocator.allocate<typename G::q_tile, 2>();
    typename G::q_sc_tile (&q_sc_smem)[2] =
        allocator.allocate<typename G::q_sc_tile, 2>();
    typename G::k_tile &k_smem = allocator.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem =
        allocator.allocate<typename G::k_sc_tile>();
    typename G::p_tile &p_smem = allocator.allocate<typename G::p_tile>();
    typename G::p_sc_tile &p_sc_smem =
        allocator.allocate<typename G::p_sc_tile>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::v_sc_tile &v_sc_smem =
        allocator.allocate<typename G::v_sc_tile>();

    __shared__ semaphore q_payload_done[2];
    __shared__ semaphore k_payload_done, v_payload_done;
    __shared__ semaphore qk_scale_done[C::QK_SCALE_CHUNKS];
    __shared__ semaphore qk_done, p_ready, pv_scale_done;
    __shared__ semaphore pv_src_done, pv_done;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(q_payload_done[stage], 0, 1);
        }
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

    auto now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if constexpr (ENABLE_TRACE) {
            if (field < g.trace_stride) {
                g.trace[(unsigned long long)task * g.trace_stride + field] =
                    value;
            }
        }
    };
    if constexpr (ENABLE_TRACE) {
        if (threadIdx.x == 0) {
            trace_store(0, task); trace_store(1, kittens::smid());
            trace_store(2, tm_alloc.addr); trace_store(3, MIN_BLOCKS_PER_SM);
            trace_store(4, pair); trace_store(5, h); trace_store(6, now());
            trace_store(15, 0);
        }
    }

    tmem224_load_query_pair<C>(
        g, b, h, tiles, query_tile, q_smem, q_sc_smem, q_payload_done);
    int event = 0;
    unsigned long long qk_count = 0, pv_count = 0;
    #pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
        if (!query_active[stage]) {
            continue;
        }
        float row_max = base_types::constants<float>::neg_infty();
        float row_sum = 0.0f;
        tmem224_run_query_walk<C>(
            g, b, h, tiles, query_tile[stage], 0, 0,
            q_smem[stage], q_sc_smem[stage], k_smem, k_sc_smem,
            p_smem, p_sc_smem, v_smem, v_sc_smem, score, output,
            q_scale, k_scale, p_scale, v_scale, k_payload_done,
            v_payload_done, qk_scale_done, qk_done, p_ready,
            pv_scale_done, pv_src_done, pv_done, event, row_max,
            row_sum, qk_count, pv_count);
        if (warpgroup::groupid() == 1) {
            tmem224_store_output<C>(
                g, output, b, h, query_tile[stage], 0, row_sum);
            const float scale_log2 = 0.10408340855860843f *
                g.q_sg[b * g.heads + h] * g.k_sg[b * g.heads + h];
            tmem224_store_lse(
                g, g.lse, b, h, query_tile[stage], row_max, row_sum,
                scale_log2);
            warpgroup::sync(2);
        }
        __syncthreads();
    }

    if constexpr (ENABLE_TRACE) {
        if (threadIdx.x == 0) {
            trace_store(7, now()); trace_store(8, qk_count);
            trace_store(9, pv_count); trace_store(10, event);
            trace_store(11, LOGICAL_END); trace_store(12, 256);
            trace_store(13, query_active[1]); __threadfence_system();
            trace_store(15, 1);
        }
    }
}

template <typename C, int EXTERNAL_MODE, int MIN_BLOCKS_PER_SM,
          bool ENABLE_TRACE>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_tmem224_output_evacuation(
    const __grid_constant__ globals_mxfp4_tmem224_pair<C> g) {
    using G = globals_mxfp4_tmem224_pair<C>;
    using score_tt = tt<float, C::Mb, C::Nb>;
    using output_tt = tt<float, C::Mb, C::Dvo>;
    using qk_scale_tt = full_tt_fp8e4m3<16>;
    using pv_scale_tt = full_tt_fp8e8m0<16>;
    using external_tile = std::conditional_t<
        EXTERNAL_MODE == TMEM224_EXTERNAL_FP32_SMEM,
        st_fl<128, 128, false>,
        std::conditional_t<
            EXTERNAL_MODE == TMEM224_EXTERNAL_BF16_SMEM,
            st_bf<128, 128, false>,
            std::conditional_t<
                EXTERNAL_MODE == TMEM224_EXTERNAL_FP16_SMEM,
                st_hf<128, 128, false>, st_fl<16, 16, false>>>>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = 64;
    constexpr int SCALE_A_BASE = 192;
    constexpr int SCALE_B_BASE = 208;
    constexpr int LOGICAL_END = 224;
    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2 &&
                  C::Mb == 128 && C::Nb == 64 && C::Dqk == 192 &&
                  C::Dvo == 128 && C::CLUSTER_SIZE == 1);
    static_assert(LOGICAL_END <= tensor_allocator<2, 1>::cols &&
                  tensor_allocator<2, 1>::cols == 256);
    static_assert(EXTERNAL_MODE >= TMEM224_EXTERNAL_FP32_SMEM &&
                  EXTERNAL_MODE <= TMEM224_EXTERNAL_FP16_SMEM);

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
    const int tiles = g.seqlen / 128;
    const int pairs_per_head = (tiles + 1) / 2;
    const int bh = task / pairs_per_head;
    const int pair = task - bh * pairs_per_head;
    const int b = bh / g.heads;
    const int h = bh - b * g.heads;
    const int query_tile[2] = {
        2 * pair, min(2 * pair + 1, tiles - 1)};
    const bool query_active[2] = {true, 2 * pair + 1 < tiles};

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::q_tile (&q_smem)[2] =
        allocator.allocate<typename G::q_tile, 2>();
    typename G::q_sc_tile (&q_sc_smem)[2] =
        allocator.allocate<typename G::q_sc_tile, 2>();
    typename G::k_tile &k_smem = allocator.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem =
        allocator.allocate<typename G::k_sc_tile>();
    typename G::p_tile &p_smem = allocator.allocate<typename G::p_tile>();
    typename G::p_sc_tile &p_sc_smem =
        allocator.allocate<typename G::p_sc_tile>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::v_sc_tile &v_sc_smem =
        allocator.allocate<typename G::v_sc_tile>();
    external_tile *external_smem = nullptr;
    if constexpr (EXTERNAL_MODE == TMEM224_EXTERNAL_FP32_SMEM ||
                  EXTERNAL_MODE == TMEM224_EXTERNAL_BF16_SMEM ||
                  EXTERNAL_MODE == TMEM224_EXTERNAL_FP16_SMEM) {
        external_smem = &allocator.allocate<external_tile>();
    }
    float external_registers[
        EXTERNAL_MODE == TMEM224_EXTERNAL_FP32_REGISTER ? 128 : 1];

    __shared__ semaphore q_payload_done[2];
    __shared__ semaphore k_payload_done, v_payload_done;
    __shared__ semaphore qk_scale_done[C::QK_SCALE_CHUNKS];
    __shared__ semaphore qk_done, p_ready, pv_scale_done;
    __shared__ semaphore pv_src_done, pv_done, swap_done;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(q_payload_done[stage], 0, 1);
        }
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
        init_semaphore(swap_done, 0, 1);
    }
    __syncthreads();

    auto now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if constexpr (ENABLE_TRACE) {
            if (field < g.trace_stride) {
                g.trace[(unsigned long long)task * g.trace_stride + field] =
                    value;
            }
        }
    };
    if constexpr (ENABLE_TRACE) {
        if (threadIdx.x == 0) {
            trace_store(0, task); trace_store(1, kittens::smid());
            trace_store(2, tm_alloc.addr); trace_store(3, MIN_BLOCKS_PER_SM);
            trace_store(4, pair); trace_store(5, h); trace_store(6, now());
            trace_store(15, 0);
        }
    }

    tmem224_load_query_pair<C>(
        g, b, h, tiles, query_tile, q_smem, q_sc_smem, q_payload_done);
    const bool producer_group = warpgroup::groupid() == 0;
    const bool output_group = warpgroup::groupid() == 1;
    const bool issue_warp = producer_group && warpgroup::warpid() == 0;
    int event = 0;
    int swap_event = 0;
    int current_stage = 0;
    int update_count[2] = {0, 0};
    bool output_stored[2] = {false, false};
    float row_max[2] = {
        base_types::constants<float>::neg_infty(),
        base_types::constants<float>::neg_infty()};
    float row_sum[2] = {0.0f, 0.0f};
    unsigned long long kv_load_count = 0, qk_count = 0, pv_count = 0;
    const int max_query_tile = query_active[1]
        ? query_tile[1] : query_tile[0];
    const int micro_iters = 2 * (max_query_tile + 1);

    #pragma unroll 1
    for (int micro = 0; micro < micro_iters; ++micro) {
        const int payload_phase = micro & 1;
        if (producer_group) {
            if (event > 0 && issue_warp) {
                wait(pv_src_done, (event - 1) & 1);
                wait(pv_done, (event - 1) & 1);
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
                ++kv_load_count;
            }
            const uint8_t *k_sc_source = g.k_sc +
                (((b * (tiles * 2) + micro) *
                  (g.heads * C::QK_SCALE_CHUNKS) +
                  h * C::QK_SCALE_CHUNKS) * 512);
            tmem224_copy_scale_tile(k_sc_smem, k_sc_source);
            const int key_tile = micro >> 1;
            const int key_half = micro & 1;
            const int v_record = (b * g.heads + h) * 2;
            const uint8_t *v_sc_source = g.v_sc +
                ((v_record * tiles + key_tile) *
                 sizeof(typename G::v_sc_tile));
            tmem224_copy_scale_half(v_sc_smem, v_sc_source, key_half);
            warpgroup::sync(1);
            if (threadIdx.x == 0) {
                wait(k_payload_done, payload_phase);
                wait(v_payload_done, payload_phase);
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
        }

        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            const bool consumes = query_active[stage] &&
                micro < 2 * (query_tile[stage] + 1);
            if (!consumes) {
                continue;
            }
            if (stage != current_stage) {
                if (output_group) {
                    wait(pv_done, (event - 1) & 1);
                    tensor_after_thread_sync();
                    tmem224_exchange_output_state<EXTERNAL_MODE>(
                        output, external_smem, g.workspace,
                        external_registers, task,
                        update_count[current_stage] > 0 &&
                            !output_stored[current_stage],
                        update_count[stage] > 0);
                    warpgroup::sync(2);
                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                        arrive(swap_done);
                    }
                }
                if (issue_warp) {
                    wait(pv_src_done, (event - 1) & 1);
                    wait(pv_done, (event - 1) & 1);
                    wait(swap_done, swap_event & 1);
                    tensor_after_thread_sync();
                }
                ++swap_event;
                current_stage = stage;
            }

            tmem224_run_loaded_query_micro<C>(
                g, b, h, query_tile[stage], micro,
                update_count[stage] == 0, q_smem[stage],
                q_sc_smem[stage], k_smem, k_sc_smem, p_smem, p_sc_smem,
                v_smem, v_sc_smem, score, output, q_scale, k_scale,
                p_scale, v_scale, qk_scale_done, qk_done, p_ready,
                pv_scale_done, pv_src_done, pv_done, event, row_max[stage],
                row_sum[stage], qk_count, pv_count);
            ++update_count[stage];
            ++event;

            const bool stage_final =
                micro + 1 == 2 * (query_tile[stage] + 1);
            if (stage_final) {
                if (output_group) {
                    wait(pv_done, (event - 1) & 1);
                    tensor_after_thread_sync();
                    tmem224_store_output<C>(
                        g, output, b, h, query_tile[stage], 0,
                        row_sum[stage]);
                    const float scale_log2 = 0.10408340855860843f *
                        g.q_sg[b * g.heads + h] *
                        g.k_sg[b * g.heads + h];
                    tmem224_store_lse(
                        g, g.lse, b, h, query_tile[stage],
                        row_max[stage], row_sum[stage], scale_log2);
                    warpgroup::sync(2);
                }
                output_stored[stage] = true;
            }
        }
    }

    if (event > 0) {
        const int phase = (event - 1) & 1;
        if (issue_warp) {
            wait(pv_src_done, phase);
            wait(pv_done, phase);
            tensor_after_thread_sync();
        }
        if (output_group) {
            wait(pv_done, phase);
            tensor_after_thread_sync();
        }
    }
    __syncthreads();
    if constexpr (ENABLE_TRACE) {
        if (threadIdx.x == 0) {
            trace_store(7, now()); trace_store(8, qk_count);
            trace_store(9, pv_count); trace_store(10, event);
            trace_store(11, LOGICAL_END); trace_store(12, 256);
            trace_store(13, EXTERNAL_MODE);
            trace_store(14, kv_load_count); __threadfence_system();
            trace_store(15, 1);
        }
    }
}

#endif

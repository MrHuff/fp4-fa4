#pragma once

__device__ __forceinline__ void hao_direct_wait(
    semaphore &sem,
    int phase
) {
#if TK_HAO_DIRECT_WAIT_HINT == 10000000
    const uint32_t mbar_ptr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&sem));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "HAO_DIRECT_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 "
        "P1, [%0], %1, 10000000;\n"
        "@P1 bra.uni HAO_DIRECT_WAIT_DONE;\n"
        "bra.uni HAO_DIRECT_WAIT;\n"
        "HAO_DIRECT_WAIT_DONE:\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(phase)
        : "memory");
#else
    wait(sem, phase);
#endif
}

template <typename C>
struct globals_fp8pv_hao_direct {
    using Parent = globals_fp4pv_upstream_fp8pv<C>;
    using q_tile = typename Parent::q_tile;
    using q_sc_tile = std::conditional_t<
        C::HAO_DIRECT_NVFP4_QK,
        typename Parent::q_sc_tile,
        st_hf<1, 256, false>>;
    using k_tile = typename Parent::k_tile;
    using k_sc_tile = std::conditional_t<
        C::HAO_DIRECT_NVFP4_QK,
        typename Parent::k_sc_tile,
        st_hf<1, 256, false>>;
    using o_tile = typename Parent::o_tile;
    using v_fp8_tile = typename Parent::v_fp8_tile;
    using q_gl = typename Parent::q_gl;
    using q_sc_gl = gl<
        half, -1, -1, -1, 256,
        tma::descriptor<q_sc_tile, dim::ROW>>;
    using q_sg_gl = typename Parent::q_sg_gl;
    using k_gl = typename Parent::k_gl;
    using k_sc_gl = gl<
        half, -1, -1, -1, 256,
        tma::descriptor<k_sc_tile, dim::ROW>>;
    using k_sg_gl = typename Parent::k_sg_gl;
    using v_fp8_gl = typename Parent::v_fp8_gl;
    using o_gl = typename Parent::o_gl;
    using lse_gl = typename Parent::lse_gl;

    q_gl q;
    q_sc_gl q_sc;
    q_sg_gl q_sg;
    k_gl k;
    k_sc_gl k_sc;
    k_sg_gl k_sg;
    v_fp8_gl v_fp8;
    o_gl o;
    lse_gl lse;
    int disable_lse_store = 0;
};

template <typename C>
__device__ __forceinline__ float qk_scale_log2(
    const globals_fp8pv_hao_direct<C> &g,
    int b,
    int h
) {
    static_assert(C::Dqk == 128);
    constexpr float LOG2E_OVER_SQRT_D = 0.1275170599133291f;
    float q_scale = 0.0f;
    float k_scale = 0.0f;
    if (warp::laneid() == 0) {
        q_scale = g.q_sg[{b, h, 0, 0}];
        k_scale = g.k_sg[{b, h, 0, 0}];
    }
    q_scale = __shfl_sync(0xffffffffu, q_scale, 0);
    k_scale = __shfl_sync(0xffffffffu, k_scale, 0);
    return LOG2E_OVER_SQRT_D * q_scale * k_scale;
}

template <typename C>
struct hao_direct_task {
    int batch;
    int head;
    int pair;
    int qtile0;
    int qtile1;
};

template <typename C>
__device__ __forceinline__ hao_direct_task<C>
hao_direct_decode_task(int logical_job) {
    constexpr int PAIRS_PER_BATCH =
        C::N_TILES / C::QUERY_STAGES;
    const int batch =
        logical_job / (C::HEADS * PAIRS_PER_BATCH);
    const int local =
        logical_job % (C::HEADS * PAIRS_PER_BATCH);
    const int head = local % C::HEADS;
    const int pair = local / C::HEADS;
    return {batch, head, pair, pair * 2, pair * 2 + 1};
}

template <typename C>
union hao_direct_fp8_kv_payload_slot {
    using G = globals_fp8pv_hao_direct<C>;
    typename G::k_tile k;
    typename G::v_fp8_tile v;
};

template <typename C>
struct hao_direct_fp8_k_scale_slot {
    using G = globals_fp8pv_hao_direct<C>;
    typename G::k_sc_tile k;
};

template <typename C>
struct hao_direct_pipeline_smem_layout {
    using G = globals_fp8pv_hao_direct<C>;
    using Payload = hao_direct_fp8_kv_payload_slot<C>;
    using Scale = hao_direct_fp8_k_scale_slot<C>;
    static constexpr int ALIGNMENT = 1024;
    static constexpr int align_up(int offset) {
        return ((offset + ALIGNMENT - 1) / ALIGNMENT) *
               ALIGNMENT;
    }
    static constexpr int append(int offset, int bytes) {
        return align_up(offset) + bytes;
    }
    static constexpr int bytes() {
        int offset = 0;
        offset = append(
            offset, 2 * int(sizeof(typename G::q_tile)));
        offset = append(
            offset, 2 * int(sizeof(typename G::q_sc_tile)));
        offset = append(
            offset, C::KV_STAGES * int(sizeof(Payload)));
        offset = append(
            offset, C::KV_STAGES * int(sizeof(Scale)));
        offset = append(
            offset, 2 * int(sizeof(typename G::o_tile)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        return offset + ALIGNMENT;
    }
};

template <
    typename C,
    typename Score,
    typename Q,
    typename K,
    typename QScaleSmem,
    typename KScaleSmem,
    typename QScaleTmem,
    typename KScaleTmem>
__device__ __forceinline__ void hao_direct_issue_qk_mxfp4_fp8pv(
    Score &score,
    Q &q,
    K &k,
    QScaleSmem &q_scale_smem,
    KScaleSmem &k_scale_smem,
    QScaleTmem &q_scale_tmem,
    KScaleTmem &k_scale_tmem,
    semaphore &finished
) {
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e8m0;
    using T_D = typename Score::T;
    constexpr uint32_t IDESC0 =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
    constexpr uint32_t IDESC1 =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 2>();
    auto &q_scale_source =
        *reinterpret_cast<const st_fp8e8m0<32, 16, false> *>(
            &q_scale_smem.data[0]);
    auto &k_scale_source =
        *reinterpret_cast<const st_fp8e8m0<32, 16, false> *>(
            &k_scale_smem.data[0]);
    fp4pv_load_mxnv_scale_async_ranklocal(
        q_scale_tmem, q_scale_source);
    fp4pv_load_mxnv_scale_async_ranklocal(
        k_scale_tmem, k_scale_source);
    st_descriptor<Q, 0> q_desc(q);
    st_descriptor<K, 0> k_desc(k);
    detail::tcgen05::template st_st<
        T_AB, T_SAB, 0, 1, 32>(
            score.addr,
            q_desc.chunk_descriptor(0),
            k_desc.chunk_descriptor(0),
            q_scale_tmem.addr,
            k_scale_tmem.addr,
            IDESC0);
    detail::tcgen05::template st_st<
        T_AB, T_SAB, 1, 1, 32>(
            score.addr,
            q_desc.chunk_descriptor(1),
            k_desc.chunk_descriptor(1),
            q_scale_tmem.addr,
            k_scale_tmem.addr,
            IDESC1);
    detail::tcgen05::commit<1>(finished);
}

template <typename C, typename D, typename P, typename V>
__device__ __forceinline__ void hao_direct_issue_pv_fp8_n128(
    D &out,
    P &p,
    V &v,
    semaphore &p_middle,
    semaphore &p_tail,
    int p_phase,
    bool accumulate) {
    using T_AB = fp8e4m3;
    using T_D = typename D::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, C::Mb, C::Dvo, false, false>();
    st_descriptor<V, 0> v_desc(v);
    #pragma unroll
    for (int chunk = 0; chunk < C::P_FIRST_CHUNKS; ++chunk) {
        const uint32_t p_addr = p.template chunk_addr<0>(chunk);
        const uint64_t v_descriptor =
            v_desc.chunk_descriptor(chunk);
        const int chunk_accumulate =
            chunk == 0 ? int(accumulate) : 1;
        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.eq.u32 p, 1, %4;\n\t"
            "tcgen05.mma.cta_group::1.kind::f8f6f4 "
            "[%0], [%1], %2, %3, p;}\n"
            :: "r"(out.addr),
               "r"(p_addr),
               "l"(v_descriptor),
               "r"(IDESC),
               "r"(chunk_accumulate));
    }

    if constexpr (C::HAO_DIRECT_TWO_CHUNK_PIPELINE) {
        hao_direct_wait(p_middle, p_phase);
        constexpr int chunk = 2;
        const uint32_t p_addr = p.template chunk_addr<0>(chunk);
        const uint64_t v_descriptor =
            v_desc.chunk_descriptor(chunk);
        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.eq.u32 p, 1, %4;\n\t"
            "tcgen05.mma.cta_group::1.kind::f8f6f4 "
            "[%0], [%1], %2, %3, p;}\n"
            :: "r"(out.addr),
               "r"(p_addr),
               "l"(v_descriptor),
               "r"(IDESC),
               "r"(1));
    }

    hao_direct_wait(p_tail, p_phase);

    constexpr int tail_start =
        C::HAO_DIRECT_TWO_CHUNK_PIPELINE
            ? 3
            : C::P_FIRST_CHUNKS;
    #pragma unroll
    for (int chunk = tail_start; chunk < 4; ++chunk) {
        const uint32_t p_addr = p.template chunk_addr<0>(chunk);
        const uint64_t v_descriptor =
            v_desc.chunk_descriptor(chunk);
        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.eq.u32 p, 1, %4;\n\t"
            "tcgen05.mma.cta_group::1.kind::f8f6f4 "
            "[%0], [%1], %2, %3, p;}\n"
            :: "r"(out.addr),
               "r"(p_addr),
               "l"(v_descriptor),
               "r"(IDESC),
               "r"(1));
    }
}

template <int Quarter, int Pair, typename C>
__device__ __forceinline__ float2 hao_direct_exp2_fp8_pair(
    float2 value,
    float row_max,
    float scale_log2);

template <int Quarter, int Slice, typename C>
__device__ __forceinline__ float
hao_direct_transform_pack_fp8_slice(
    float2 *scores,
    float row_max,
    float scale_log2,
    uint32_t &word0,
    uint32_t &word1
) {
    scores[0] = hao_direct_exp2_fp8_pair<
        Quarter, Slice * 4 + 0, C>(
            scores[0], row_max, scale_log2);
    scores[1] = hao_direct_exp2_fp8_pair<
        Quarter, Slice * 4 + 1, C>(
            scores[1], row_max, scale_log2);
    scores[2] = hao_direct_exp2_fp8_pair<
        Quarter, Slice * 4 + 2, C>(
            scores[2], row_max, scale_log2);
    scores[3] = hao_direct_exp2_fp8_pair<
        Quarter, Slice * 4 + 3, C>(
            scores[3], row_max, scale_log2);
    word0 = upstream_pack_fp8_four(scores[0], scores[1]);
    word1 = upstream_pack_fp8_four(scores[2], scores[3]);
    const float2 sum01 = __fadd2_rn(scores[0], scores[1]);
    const float2 sum23 = __fadd2_rn(scores[2], scores[3]);
    const float2 sum = __fadd2_rn(sum01, sum23);
    return sum.x + sum.y;
}

template <int Quarter, int Half, typename P>
__device__ __forceinline__ void
hao_direct_store_p_fp8_half_quarter_tmem(
    P &p,
    int reader_warp,
    uint32_t word0,
    uint32_t word1,
    uint32_t word2,
    uint32_t word3
) {
    static_assert(Half == 0 || Half == 1);
    const uint32_t addr =
        p.addr + (reader_warp * 32 << 16) +
        Quarter * 8 + Half * 4;
    asm volatile(
        "{tcgen05.st.sync.aligned.32x32b.x4.b32 "
        "[%0], {%1,%2,%3,%4};}"
        :: "r"(addr),
           "r"(word0),
           "r"(word1),
           "r"(word2),
           "r"(word3));
}

template <int Quarter, int Pair, typename C>
__device__ __forceinline__ float2 hao_direct_exp2_fp8_pair(
    float2 value,
    float row_max,
    float scale_log2
) {
    // Match HAO's FP8-PV e2e_frg_limit=1: only the first three x32
    // fragments use the mixed ALU/MUFU cadence.
    constexpr uint32_t ex2_emu_mask =
        Quarter < 3 ? C::ONLINE_STAGE2_EX2_EMU_MASK : 0;
    const float2 mul = {scale_log2, scale_log2};
    const float2 bias = {
        -row_max * scale_log2,
        -row_max * scale_log2};
    return fp4pv_stage2_exp2_pair<
        C::ONLINE_STAGE2_EX2_EMU_PERIOD,
        ex2_emu_mask,
        Quarter * 16 + Pair>(
            __ffma2_rn(value, mul, bias));
}

template <int Quarter, int Slice, typename C>
__device__ __forceinline__ float2 hao_direct_transform_pack_fp8_four(
    float2 &s0,
    float2 &s1,
    float2 &s2,
    float2 &s3,
    float row_max,
    float scale_log2,
    uint32_t &word0,
    uint32_t &word1
) {
    s0 = hao_direct_exp2_fp8_pair<
        Quarter, Slice * 4 + 0, C>(s0, row_max, scale_log2);
    s1 = hao_direct_exp2_fp8_pair<
        Quarter, Slice * 4 + 1, C>(s1, row_max, scale_log2);
    s2 = hao_direct_exp2_fp8_pair<
        Quarter, Slice * 4 + 2, C>(s2, row_max, scale_log2);
    s3 = hao_direct_exp2_fp8_pair<
        Quarter, Slice * 4 + 3, C>(s3, row_max, scale_log2);
    word0 = upstream_pack_fp8_four(s0, s1);
    word1 = upstream_pack_fp8_four(s2, s3);
    const float2 sum01 = __fadd2_rn(s0, s1);
    const float2 sum23 = __fadd2_rn(s2, s3);
    return __fadd2_rn(sum01, sum23);
}

template <int Quarter, typename P>
__device__ __forceinline__ float
hao_direct_store_fixed_p_fp8_quarter(
    P &p,
    int reader_warp
) {
    // E4M3 1.0 has bit pattern 0x38.
    constexpr uint32_t P_ONES = 0x38383838u;
    const uint32_t words[8] = {
        P_ONES, P_ONES, P_ONES, P_ONES,
        P_ONES, P_ONES, P_ONES, P_ONES};
    upstream_store_p_fp8_quarter_tmem(
        p, reader_warp, Quarter, words);
    return 32.0f;
}

template <int Quarter, typename P>
__device__ __forceinline__ float
hao_direct_pack_raw_score_fp8_quarter(
    P &p,
    int reader_warp,
    const score_quarter_regs &scores
) {
    const uint32_t words[8] = {
        upstream_pack_fp8_four(scores.v0, scores.v1),
        upstream_pack_fp8_four(scores.v2, scores.v3),
        upstream_pack_fp8_four(scores.v4, scores.v5),
        upstream_pack_fp8_four(scores.v6, scores.v7),
        upstream_pack_fp8_four(scores.v8, scores.v9),
        upstream_pack_fp8_four(scores.v10, scores.v11),
        upstream_pack_fp8_four(scores.v12, scores.v13),
        upstream_pack_fp8_four(scores.v14, scores.v15)};
    upstream_store_p_fp8_quarter_tmem(
        p, reader_warp, Quarter, words);
    // Keep the epilogue denominator finite without adding a reduction to
    // this conversion-only ceiling.
    return 32.0f;
}

__device__ __forceinline__ float
hao_direct_exp2_native_scalar(float value) {
    float result;
    asm volatile(
        "ex2.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(value));
    return result;
}

template <int Quarter, typename C, typename P>
__device__ __forceinline__ float
hao_direct_transform_pack_shiftless_fp8_quarter(
    P &p,
    int reader_warp,
    score_quarter_regs &scores,
    float quarter_max,
    float scale_log2
) {
    constexpr float CUBIC_A = 0.07839806f;
    constexpr float CUBIC_B = 0.28625049f;
    constexpr float CUBIC_C = 0.63145205f;
    constexpr float CUBIC_D = 0.99202336f;
    constexpr float AFFINE_A = 1.62330034f;
    constexpr float AFFINE_B = 0.92083546f;
    const float absolute_scale =
        hao_direct_exp2_native_scalar(quarter_max * scale_log2);
    const float2 relative_scale = {scale_log2, scale_log2};
    const float2 relative_bias = {
        -quarter_max * scale_log2,
        -quarter_max * scale_log2};
    const float2 absolute_scale2 = {
        absolute_scale, absolute_scale};
    const float2 affine_scale = {
        AFFINE_A * scale_log2 * absolute_scale,
        AFFINE_A * scale_log2 * absolute_scale};
    const float affine_bias_scalar =
        (AFFINE_B - AFFINE_A * quarter_max * scale_log2) *
        absolute_scale;
    const float2 affine_bias = {
        affine_bias_scalar, affine_bias_scalar};
    float2 *values = &scores.v0;

    fp4pv_stage2_static_for<0, 16>([&](auto pair_tag) {
        constexpr int Pair = decltype(pair_tag)::value;
        constexpr int sample_offset = Quarter & 3;
        constexpr bool native_sample =
            Pair == sample_offset ||
            Pair == sample_offset + 8 ||
            Pair == 4 ||
            Pair == 12;
        if constexpr (native_sample) {
            const float2 absolute_log = __fmul2_rn(
                values[Pair], relative_scale);
            values[Pair] = fp4pv_stage2_exp2_pair<
                16, 0, Quarter * 16 + Pair>(absolute_log);
        } else if constexpr (
            Quarter == 0 ||
            C::HAO_DIRECT_SHIFTLESS_MODE >= 3) {
            const float2 relative_log = __ffma2_rn(
                values[Pair], relative_scale, relative_bias);
            float2 approximation = __ffma2_rn(
                relative_log,
                float2{CUBIC_A, CUBIC_A},
                float2{CUBIC_B, CUBIC_B});
            approximation = __ffma2_rn(
                approximation,
                relative_log,
                float2{CUBIC_C, CUBIC_C});
            approximation = __ffma2_rn(
                approximation,
                relative_log,
                float2{CUBIC_D, CUBIC_D});
            approximation = __fmul2_rn(
                approximation, absolute_scale2);
            values[Pair] = {
                fmaxf(approximation.x, 0.0f),
                fmaxf(approximation.y, 0.0f)};
        } else {
            const float2 approximation = __ffma2_rn(
                values[Pair], affine_scale, affine_bias);
            values[Pair] = {
                fmaxf(approximation.x, 0.0f),
                fmaxf(approximation.y, 0.0f)};
        }
    });

    float quarter_sum;
    if constexpr (
        C::HAO_DIRECT_SHIFTLESS_MODE == 2 ||
        C::HAO_DIRECT_SHIFTLESS_MODE == 4) {
        constexpr int sample_offset = Quarter & 3;
        const float2 sample01 = __fadd2_rn(
            values[sample_offset], values[sample_offset + 8]);
        const float2 sample23 = __fadd2_rn(values[4], values[12]);
        const float2 sampled_sum = __fadd2_rn(sample01, sample23);
        quarter_sum =
            4.0f * (sampled_sum.x + sampled_sum.y);
    } else {
        const float2 group0 = __fadd2_rn(
            __fadd2_rn(values[0], values[1]),
            __fadd2_rn(values[2], values[3]));
        const float2 group1 = __fadd2_rn(
            __fadd2_rn(values[4], values[5]),
            __fadd2_rn(values[6], values[7]));
        const float2 group2 = __fadd2_rn(
            __fadd2_rn(values[8], values[9]),
            __fadd2_rn(values[10], values[11]));
        const float2 group3 = __fadd2_rn(
            __fadd2_rn(values[12], values[13]),
            __fadd2_rn(values[14], values[15]));
        const float2 sum = __fadd2_rn(
            __fadd2_rn(group0, group1),
            __fadd2_rn(group2, group3));
        quarter_sum = sum.x + sum.y;
    }

    const uint32_t words[8] = {
        upstream_pack_fp8_four(values[0], values[1]),
        upstream_pack_fp8_four(values[2], values[3]),
        upstream_pack_fp8_four(values[4], values[5]),
        upstream_pack_fp8_four(values[6], values[7]),
        upstream_pack_fp8_four(values[8], values[9]),
        upstream_pack_fp8_four(values[10], values[11]),
        upstream_pack_fp8_four(values[12], values[13]),
        upstream_pack_fp8_four(values[14], values[15])};
    upstream_store_p_fp8_quarter_tmem(
        p, reader_warp, Quarter, words);
    return quarter_sum;
}

template <int Quarter, typename C, typename P>
__device__ __forceinline__ float
hao_direct_transform_pack_retained_quarter(
    P &p,
    int reader_warp,
    score_quarter_regs &scores,
    float row_max,
    float scale_log2
) {
    uint32_t words[8];
    if constexpr (C::HAO_DIRECT_P_ILP_PIPELINE) {
        scores.v0 = hao_direct_exp2_fp8_pair<
            Quarter, 0, C>(scores.v0, row_max, scale_log2);
        scores.v1 = hao_direct_exp2_fp8_pair<
            Quarter, 1, C>(scores.v1, row_max, scale_log2);
        scores.v2 = hao_direct_exp2_fp8_pair<
            Quarter, 2, C>(scores.v2, row_max, scale_log2);
        scores.v3 = hao_direct_exp2_fp8_pair<
            Quarter, 3, C>(scores.v3, row_max, scale_log2);

        scores.v4 = hao_direct_exp2_fp8_pair<
            Quarter, 4, C>(scores.v4, row_max, scale_log2);
        scores.v5 = hao_direct_exp2_fp8_pair<
            Quarter, 5, C>(scores.v5, row_max, scale_log2);
        scores.v6 = hao_direct_exp2_fp8_pair<
            Quarter, 6, C>(scores.v6, row_max, scale_log2);
        scores.v7 = hao_direct_exp2_fp8_pair<
            Quarter, 7, C>(scores.v7, row_max, scale_log2);

        words[0] = upstream_pack_fp8_four(scores.v0, scores.v1);
        words[1] = upstream_pack_fp8_four(scores.v2, scores.v3);
        const float2 group0 = __fadd2_rn(
            __fadd2_rn(scores.v0, scores.v1),
            __fadd2_rn(scores.v2, scores.v3));

        scores.v8 = hao_direct_exp2_fp8_pair<
            Quarter, 8, C>(scores.v8, row_max, scale_log2);
        scores.v9 = hao_direct_exp2_fp8_pair<
            Quarter, 9, C>(scores.v9, row_max, scale_log2);
        scores.v10 = hao_direct_exp2_fp8_pair<
            Quarter, 10, C>(scores.v10, row_max, scale_log2);
        scores.v11 = hao_direct_exp2_fp8_pair<
            Quarter, 11, C>(scores.v11, row_max, scale_log2);

        words[2] = upstream_pack_fp8_four(scores.v4, scores.v5);
        words[3] = upstream_pack_fp8_four(scores.v6, scores.v7);
        const float2 group1 = __fadd2_rn(
            __fadd2_rn(scores.v4, scores.v5),
            __fadd2_rn(scores.v6, scores.v7));

        scores.v12 = hao_direct_exp2_fp8_pair<
            Quarter, 12, C>(scores.v12, row_max, scale_log2);
        scores.v13 = hao_direct_exp2_fp8_pair<
            Quarter, 13, C>(scores.v13, row_max, scale_log2);
        scores.v14 = hao_direct_exp2_fp8_pair<
            Quarter, 14, C>(scores.v14, row_max, scale_log2);
        scores.v15 = hao_direct_exp2_fp8_pair<
            Quarter, 15, C>(scores.v15, row_max, scale_log2);

        words[4] = upstream_pack_fp8_four(scores.v8, scores.v9);
        words[5] = upstream_pack_fp8_four(scores.v10, scores.v11);
        const float2 group2 = __fadd2_rn(
            __fadd2_rn(scores.v8, scores.v9),
            __fadd2_rn(scores.v10, scores.v11));

        words[6] = upstream_pack_fp8_four(scores.v12, scores.v13);
        words[7] = upstream_pack_fp8_four(scores.v14, scores.v15);
        const float2 group3 = __fadd2_rn(
            __fadd2_rn(scores.v12, scores.v13),
            __fadd2_rn(scores.v14, scores.v15));

        upstream_store_p_fp8_quarter_tmem(
            p, reader_warp, Quarter, words);
        const float2 sum01 = __fadd2_rn(group0, group1);
        const float2 sum23 = __fadd2_rn(group2, group3);
        const float2 quarter_sum = __fadd2_rn(sum01, sum23);
        return quarter_sum.x + quarter_sum.y;
    }

    uint32_t w0;
    uint32_t w1;
    const float2 sum0 =
        hao_direct_transform_pack_fp8_four<Quarter, 0, C>(
        scores.v0, scores.v1, scores.v2, scores.v3,
        row_max, scale_log2, w0, w1);
    words[0] = w0;
    words[1] = w1;
    const float2 sum1 =
        hao_direct_transform_pack_fp8_four<Quarter, 1, C>(
        scores.v4, scores.v5, scores.v6, scores.v7,
        row_max, scale_log2, w0, w1);
    words[2] = w0;
    words[3] = w1;
    const float2 sum2 =
        hao_direct_transform_pack_fp8_four<Quarter, 2, C>(
        scores.v8, scores.v9, scores.v10, scores.v11,
        row_max, scale_log2, w0, w1);
    words[4] = w0;
    words[5] = w1;
    const float2 sum3 =
        hao_direct_transform_pack_fp8_four<Quarter, 3, C>(
        scores.v12, scores.v13, scores.v14, scores.v15,
        row_max, scale_log2, w0, w1);
    words[6] = w0;
    words[7] = w1;
    upstream_store_p_fp8_quarter_tmem(
        p, reader_warp, Quarter, words);
    const float2 sum01 = __fadd2_rn(sum0, sum1);
    const float2 sum23 = __fadd2_rn(sum2, sum3);
    const float2 quarter_sum = __fadd2_rn(sum01, sum23);
    return quarter_sum.x + quarter_sum.y;
}

struct hao_direct_score_quarter_bf16_regs {
    bf16_2 v0;
    bf16_2 v1;
    bf16_2 v2;
    bf16_2 v3;
    bf16_2 v4;
    bf16_2 v5;
    bf16_2 v6;
    bf16_2 v7;
    bf16_2 v8;
    bf16_2 v9;
    bf16_2 v10;
    bf16_2 v11;
    bf16_2 v12;
    bf16_2 v13;
    bf16_2 v14;
    bf16_2 v15;
};

__device__ __forceinline__ void
hao_direct_compress_score_quarter_bf16(
    const score_quarter_regs &source,
    hao_direct_score_quarter_bf16_regs &destination
) {
    destination.v0 = __float22bfloat162_rn(source.v0);
    destination.v1 = __float22bfloat162_rn(source.v1);
    destination.v2 = __float22bfloat162_rn(source.v2);
    destination.v3 = __float22bfloat162_rn(source.v3);
    destination.v4 = __float22bfloat162_rn(source.v4);
    destination.v5 = __float22bfloat162_rn(source.v5);
    destination.v6 = __float22bfloat162_rn(source.v6);
    destination.v7 = __float22bfloat162_rn(source.v7);
    destination.v8 = __float22bfloat162_rn(source.v8);
    destination.v9 = __float22bfloat162_rn(source.v9);
    destination.v10 = __float22bfloat162_rn(source.v10);
    destination.v11 = __float22bfloat162_rn(source.v11);
    destination.v12 = __float22bfloat162_rn(source.v12);
    destination.v13 = __float22bfloat162_rn(source.v13);
    destination.v14 = __float22bfloat162_rn(source.v14);
    destination.v15 = __float22bfloat162_rn(source.v15);
}

template <int Quarter, typename C, typename P>
__device__ __forceinline__ float
hao_direct_transform_pack_compressed_quarter(
    P &p,
    int reader_warp,
    const hao_direct_score_quarter_bf16_regs &scores,
    float row_max,
    float scale_log2
) {
    uint32_t words[8];
    uint32_t w0;
    uint32_t w1;
    float2 s0 = __bfloat1622float2(scores.v0);
    float2 s1 = __bfloat1622float2(scores.v1);
    float2 s2 = __bfloat1622float2(scores.v2);
    float2 s3 = __bfloat1622float2(scores.v3);
    const float2 sum0 =
        hao_direct_transform_pack_fp8_four<Quarter, 0, C>(
            s0, s1, s2, s3, row_max, scale_log2, w0, w1);
    words[0] = w0;
    words[1] = w1;
    s0 = __bfloat1622float2(scores.v4);
    s1 = __bfloat1622float2(scores.v5);
    s2 = __bfloat1622float2(scores.v6);
    s3 = __bfloat1622float2(scores.v7);
    const float2 sum1 =
        hao_direct_transform_pack_fp8_four<Quarter, 1, C>(
            s0, s1, s2, s3, row_max, scale_log2, w0, w1);
    words[2] = w0;
    words[3] = w1;
    s0 = __bfloat1622float2(scores.v8);
    s1 = __bfloat1622float2(scores.v9);
    s2 = __bfloat1622float2(scores.v10);
    s3 = __bfloat1622float2(scores.v11);
    const float2 sum2 =
        hao_direct_transform_pack_fp8_four<Quarter, 2, C>(
            s0, s1, s2, s3, row_max, scale_log2, w0, w1);
    words[4] = w0;
    words[5] = w1;
    s0 = __bfloat1622float2(scores.v12);
    s1 = __bfloat1622float2(scores.v13);
    s2 = __bfloat1622float2(scores.v14);
    s3 = __bfloat1622float2(scores.v15);
    const float2 sum3 =
        hao_direct_transform_pack_fp8_four<Quarter, 3, C>(
            s0, s1, s2, s3, row_max, scale_log2, w0, w1);
    words[6] = w0;
    words[7] = w1;
    upstream_store_p_fp8_quarter_tmem(
        p, reader_warp, Quarter, words);
    const float2 sum01 = __fadd2_rn(sum0, sum1);
    const float2 sum23 = __fadd2_rn(sum2, sum3);
    const float2 quarter_sum = __fadd2_rn(sum01, sum23);
    return quarter_sum.x + quarter_sum.y;
}

template <typename C>
__cluster_dims__(1, 1, 1) __launch_bounds__(512, 1)
__global__ void kernel_fp4pv_hao_direct_fp8pv(
    const __grid_constant__ globals_fp8pv_hao_direct<C> g) {
    static_assert(C::HAO_DIRECT_PIPELINE);
    static_assert(C::NUM_THREADS == 512);
    static_assert(C::NUM_WARPS == 16);
    static_assert(C::CLUSTER_SIZE == 1);
    static_assert(C::Mb == 128 && C::Nb == 128);
    static_assert(C::Dqk == 128 && C::Dvo == 128);
    static_assert(C::QK_SCALE_CHUNKS == 2);
    static_assert(C::KV_STAGES == 4);
    static_assert(
        C::P_FIRST_CHUNKS == 2 ||
        C::P_FIRST_CHUNKS == 3);

    using G = globals_fp8pv_hao_direct<C>;
    using Score = tt<float, 128, 128>;
    using Output = tt<float, 128, 128>;
    using P = tt<fp8e4m3, 128, 128>;
    using QKScale = std::conditional_t<
        C::HAO_DIRECT_NVFP4_QK,
        full_tt_fp8e4m3<16>,
        full_tt_fp8e8m0<16>>;
    constexpr int S0 = 0;
    constexpr int S1 = 128;
    constexpr int O0 = 256;
    constexpr int O1 = 384;
    constexpr int P0 = S0 + 64;
    constexpr int P1 = S1 + 64;
    static_assert(P0 == 64 && P1 == 192);
    static_assert(O1 + 128 == C::TMEM_COLUMNS);
    static_assert(
        hao_direct_pipeline_smem_layout<C>::bytes() <=
        227 * 1024);

    tensor_allocator<1, 1> tm{};
    Score score0 = tm.template allocate<Score>(S0);
    Score score1 = tm.template allocate<Score>(S1);
    Output output0 = tm.template allocate<Output>(O0);
    Output output1 = tm.template allocate<Output>(O1);
    P p0 = tm.template allocate<P>(P0);
    P p1 = tm.template allocate<P>(P1);
    QKScale qk_q_sc0 =
        tm.template allocate<QKScale>(S1 + 0);
    QKScale qk_k_sc0 =
        tm.template allocate<QKScale>(S1 + 16);
    QKScale qk_q_sc1 =
        tm.template allocate<QKScale>(S0 + 0);
    QKScale qk_k_sc1 =
        tm.template allocate<QKScale>(S0 + 16);

    extern __shared__ int shm[];
    tma_swizzle_allocator al(shm);
    typename G::q_tile (&q_smem)[2] =
        al.allocate<typename G::q_tile, 2>();
    typename G::q_sc_tile (&q_sc_smem)[2] =
        al.allocate<typename G::q_sc_tile, 2>();
    hao_direct_fp8_kv_payload_slot<C>
        (&kv_payload)[C::KV_STAGES] =
            al.allocate<
                hao_direct_fp8_kv_payload_slot<C>,
                C::KV_STAGES>();
    hao_direct_fp8_k_scale_slot<C>
        (&kv_scale)[C::KV_STAGES] =
            al.allocate<
                hao_direct_fp8_k_scale_slot<C>,
                C::KV_STAGES>();
    typename G::o_tile (&o_smem)[2] =
        al.allocate<typename G::o_tile, 2>();
    sv_fl<128> (&row_max_smem)[2] =
        al.allocate<sv_fl<128>, 2>();
    sv_fl<128> (&row_sum_smem)[2] =
        al.allocate<sv_fl<128>, 2>();
    sv_fl<128> (&correction_smem)[2] =
        al.allocate<sv_fl<128>, 2>();

    __shared__ semaphore q_full[2];
    __shared__ semaphore q_empty[2];
    __shared__ semaphore kv_full[C::KV_STAGES];
    __shared__ semaphore kv_empty[C::KV_STAGES];
    __shared__ semaphore p_first_o_rescaled[2];
    __shared__ semaphore p_middle[2];
    __shared__ semaphore p_tail[2];
    __shared__ semaphore score_full[2];
    __shared__ semaphore output_full[2];
    __shared__ semaphore softmax_corr_full[2];
    __shared__ semaphore softmax_corr_empty[2];
    __shared__ semaphore corr_epi_full[2];
    __shared__ semaphore corr_epi_empty[2];
    __shared__ semaphore sfqk_load[2];
    __shared__ semaphore tmem_dealloc;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(q_full[stage], 0, 1);
            init_semaphore(q_empty[stage], 0, 1);
            init_semaphore(
                p_first_o_rescaled[stage], 0, 8);
            if constexpr (C::HAO_DIRECT_TWO_CHUNK_PIPELINE) {
                init_semaphore(p_middle[stage], 0, 4);
            }
            init_semaphore(p_tail[stage], 0, 4);
            init_semaphore(score_full[stage], 0, 1);
            init_semaphore(output_full[stage], 0, 1);
            init_semaphore(
                softmax_corr_full[stage], 0, 4);
            init_semaphore(
                softmax_corr_empty[stage], 0, 4);
            init_semaphore(corr_epi_full[stage], 0, 4);
            init_semaphore(corr_epi_empty[stage], 0, 1);
            init_semaphore(sfqk_load[stage], 0, 4);
        }
        #pragma unroll
        for (int stage = 0; stage < C::KV_STAGES; ++stage) {
            init_semaphore(kv_full[stage], 0, 1);
            init_semaphore(kv_empty[stage], 0, 1);
        }
        init_semaphore(tmem_dealloc, 0, 12);
    }
    __syncthreads();

    const int warp_id = int(threadIdx.x) >> 5;
    if (warpgroup::groupid() < 2) {
        warpgroup::increase_registers<C::SOFTMAX_REGS>();
    } else if (warpgroup::groupid() == 2) {
        warpgroup::decrease_registers<C::CORRECTION_REGS>();
    } else {
        warpgroup::decrease_registers<C::PRODUCER_REGS>();
    }

    if (warp_id == 14) {
        const bool leader = warp::elect_leader();
        int q_producer_phase = 1;
        int kv_producer_item = 0;

        auto load_k = [&](const hao_direct_task<C> &task, int n) {
            const int slot =
                kv_producer_item % C::KV_STAGES;
            const int producer_phase =
                1 ^ ((kv_producer_item / C::KV_STAGES) & 1);
            hao_direct_wait(kv_empty[slot], producer_phase);
            if (leader) {
                fp4pv_expect_bytes<C>(
                    kv_full[slot],
                    sizeof(typename G::k_tile) +
                        sizeof(typename G::k_sc_tile));
                tma::load_async<
                    dim::ROW, cache_policy::NORMAL>(
                    kv_payload[slot].k,
                    g.k,
                    {task.batch, task.head, n, 0},
                    kv_full[slot]);
                tma::load_async<
                    dim::ROW, cache_policy::NORMAL>(
                    kv_scale[slot].k,
                    g.k_sc,
                    {task.batch, 2 * n, task.head, 0},
                    kv_full[slot]);
            }
            ++kv_producer_item;
        };
        auto load_v = [&](const hao_direct_task<C> &task, int n) {
            const int slot =
                kv_producer_item % C::KV_STAGES;
            const int producer_phase =
                1 ^ ((kv_producer_item / C::KV_STAGES) & 1);
            hao_direct_wait(kv_empty[slot], producer_phase);
            if (leader) {
                fp4pv_expect_bytes<C>(
                    kv_full[slot],
                    sizeof(typename G::v_fp8_tile));
                tma::load_async<
                    dim::ROW, cache_policy::NORMAL>(
                    kv_payload[slot].v,
                    g.v_fp8,
                    {task.batch, task.head, 0, n},
                    kv_full[slot]);
            }
            ++kv_producer_item;
        };
        auto load_q = [&](
            const hao_direct_task<C> &task,
            int stage,
            int qtile) {
            hao_direct_wait(q_empty[stage], q_producer_phase);
            if (leader) {
                fp4pv_expect_bytes<C>(
                    q_full[stage],
                    sizeof(typename G::q_tile) +
                        sizeof(typename G::q_sc_tile));
                tma::load_async<
                    dim::ROW, cache_policy::NORMAL>(
                    q_smem[stage],
                    g.q,
                    {task.batch, task.head, qtile, 0},
                    q_full[stage]);
                tma::load_async<
                    dim::ROW, cache_policy::NORMAL>(
                    q_sc_smem[stage],
                    g.q_sc,
                    {task.batch, qtile, task.head, 0},
                    q_full[stage]);
            }
        };

        for (int logical_job = int(blockIdx.x);
             logical_job < C::LOGICAL_JOBS;
             logical_job += int(gridDim.x)) {
            const hao_direct_task task =
                hao_direct_decode_task<C>(logical_job);
            load_q(task, 0, task.qtile0);
            load_k(task, 0);
            load_q(task, 1, task.qtile1);
            q_producer_phase ^= 1;
            load_v(task, 0);
            #pragma unroll 1
            for (int n = 1; n < C::N_TILES; ++n) {
                load_k(task, n);
                load_v(task, n);
            }
        }
    } else if (warp_id == 12) {
        const bool leader = warp::elect_leader();
        int q_consumer_phase = 0;
        int kv_consumer_item = 0;
        int p_phase = 0;
        int sfqk_phase = 0;

        auto wait_kv_item = [&](int item) {
            const int slot = item % C::KV_STAGES;
            const int phase =
                (item / C::KV_STAGES) & 1;
            if (leader) {
                hao_direct_wait(kv_full[slot], phase);
            }
            return slot;
        };
        auto issue_qk = [&]<int Stage>(
            int k_slot, int sf_phase) {
            if (leader) {
                hao_direct_wait(sfqk_load[Stage], sf_phase);
                if constexpr (Stage == 0) {
                    if constexpr (C::HAO_DIRECT_NVFP4_QK) {
                        fp4pv_issue_qk_chunked_ranklocal<C>(
                            score0,
                            q_smem[0],
                            kv_payload[k_slot].k,
                            q_sc_smem[0],
                            kv_scale[k_slot].k,
                            qk_q_sc0,
                            qk_k_sc0,
                            score_full[0]);
                    } else {
                        hao_direct_issue_qk_mxfp4_fp8pv<C>(
                            score0,
                            q_smem[0],
                            kv_payload[k_slot].k,
                            q_sc_smem[0],
                            kv_scale[k_slot].k,
                            qk_q_sc0,
                            qk_k_sc0,
                            score_full[0]);
                    }
                } else {
                    if constexpr (C::HAO_DIRECT_NVFP4_QK) {
                        fp4pv_issue_qk_chunked_ranklocal<C>(
                            score1,
                            q_smem[1],
                            kv_payload[k_slot].k,
                            q_sc_smem[1],
                            kv_scale[k_slot].k,
                            qk_q_sc1,
                            qk_k_sc1,
                            score_full[1]);
                    } else {
                        hao_direct_issue_qk_mxfp4_fp8pv<C>(
                            score1,
                            q_smem[1],
                            kv_payload[k_slot].k,
                            q_sc_smem[1],
                            kv_scale[k_slot].k,
                            qk_q_sc1,
                            qk_k_sc1,
                            score_full[1]);
                    }
                }
            }
        };
        auto issue_pv = [&]<int Stage>(
            int v_slot, bool accumulate) {
            if (leader) {
                hao_direct_wait(
                    p_first_o_rescaled[Stage],
                    p_phase);
                if constexpr (Stage == 0) {
                    hao_direct_issue_pv_fp8_n128<C>(
                        output0,
                        p0,
                        kv_payload[v_slot].v,
                        p_middle[0],
                        p_tail[0],
                        p_phase,
                        accumulate);
                } else {
                    hao_direct_issue_pv_fp8_n128<C>(
                        output1,
                        p1,
                        kv_payload[v_slot].v,
                        p_middle[1],
                        p_tail[1],
                        p_phase,
                        accumulate);
                }
            }
        };

        for (int logical_job = int(blockIdx.x);
             logical_job < C::LOGICAL_JOBS;
             logical_job += int(gridDim.x)) {
            if (leader) {
                hao_direct_wait(q_full[0], q_consumer_phase);
            }
            const int first_k_slot =
                wait_kv_item(kv_consumer_item);
            issue_qk.template operator()<0>(
                first_k_slot, sfqk_phase);
            if (leader) {
                hao_direct_wait(q_full[1], q_consumer_phase);
            }
            issue_qk.template operator()<1>(
                first_k_slot, sfqk_phase);
            if (leader) {
                detail::tcgen05::commit<1>(
                    kv_empty[first_k_slot]);
            }
            ++kv_consumer_item;
            q_consumer_phase ^= 1;
            sfqk_phase ^= 1;

            #pragma unroll 1
            for (int n = 0; n + 1 < C::N_TILES; ++n) {
                const int v_slot =
                    wait_kv_item(kv_consumer_item);
                issue_pv.template operator()<0>(
                    v_slot, n != 0);

                const int next_k_item =
                    kv_consumer_item + 1;
                const int k_slot =
                    wait_kv_item(next_k_item);
                issue_qk.template operator()<0>(
                    k_slot, sfqk_phase);

                issue_pv.template operator()<1>(
                    v_slot, n != 0);
                if (leader) {
                    detail::tcgen05::commit<1>(
                        kv_empty[v_slot]);
                }
                issue_qk.template operator()<1>(
                    k_slot, sfqk_phase);
                if (leader) {
                    detail::tcgen05::commit<1>(
                        kv_empty[k_slot]);
                }

                kv_consumer_item += 2;
                p_phase ^= 1;
                sfqk_phase ^= 1;
            }

            if (leader) {
                detail::tcgen05::commit<1>(q_empty[0]);
                detail::tcgen05::commit<1>(q_empty[1]);
            }

            const int final_v_slot =
                wait_kv_item(kv_consumer_item);
            issue_pv.template operator()<0>(
                final_v_slot, C::N_TILES != 1);
            if (leader) {
                detail::tcgen05::commit<1>(output_full[0]);
            }
            issue_pv.template operator()<1>(
                final_v_slot, C::N_TILES != 1);
            if (leader) {
                detail::tcgen05::commit<1>(output_full[1]);
                detail::tcgen05::commit<1>(
                    kv_empty[final_v_slot]);
            }
            ++kv_consumer_item;
            p_phase ^= 1;
        }

        hao_direct_wait(tmem_dealloc, 0);
    } else if (warp_id < 4) {
#define HAO_DIRECT_SOFTMAX_STAGE 0
#include "hao_direct_softmax_reader.inc"
#undef HAO_DIRECT_SOFTMAX_STAGE
    } else if (warp_id < 8) {
#define HAO_DIRECT_SOFTMAX_STAGE 1
#include "hao_direct_softmax_reader.inc"
#undef HAO_DIRECT_SOFTMAX_STAGE
    } else if (warp_id >= 8 && warp_id < 12) {
        const int row_group = warp_id - 8;
        int softmax_consumer_phase = 0;
        int output_consumer_phase = 0;
        int epi_producer_phase = 1;

        // The first PV overwrites empty output accumulators.  These arrivals
        // provide the output-ready half of the first combined P/O barrier.
        if (warp::elect_leader()) {
            arrive(p_first_o_rescaled[0]);
            arrive(p_first_o_rescaled[1]);
        }

        for (int logical_job = int(blockIdx.x);
             logical_job < C::LOGICAL_JOBS;
             logical_job += int(gridDim.x)) {
            const hao_direct_task task =
                hao_direct_decode_task<C>(logical_job);

            hao_direct_wait(
                softmax_corr_full[0],
                softmax_consumer_phase);
            if (warp::elect_leader()) {
                arrive(softmax_corr_empty[0]);
            }
            hao_direct_wait(
                softmax_corr_full[1],
                softmax_consumer_phase);
            softmax_consumer_phase ^= 1;

            #pragma unroll 1
            for (int n = 0; n + 1 < C::N_TILES; ++n) {
                #pragma unroll
                for (int stage = 0; stage < 2; ++stage) {
                    hao_direct_wait(
                        softmax_corr_full[stage],
                        softmax_consumer_phase);
                    const int row =
                        row_group * 32 + warp::laneid();
                    const float correction = stats_load(
                        correction_smem[stage], row);
                    const bool should_rescale =
                        __any_sync(
                            0xffffffffu,
                            correction < 1.0f);
                    if (should_rescale) {
                        tensor_after_thread_sync();
                        if (stage == 0) {
                            upstream_correct_output_stripe<C>(
                                output0,
                                row_group,
                                correction,
                                false);
                        } else {
                            upstream_correct_output_stripe<C>(
                                output1,
                                row_group,
                                correction,
                                false);
                        }
                    }
                    if (warp::elect_leader()) {
                        arrive(
                            p_first_o_rescaled[stage]);
                        arrive(
                            softmax_corr_empty[1 - stage]);
                    }
                }
                softmax_consumer_phase ^= 1;
            }

            if (warp::elect_leader()) {
                arrive(softmax_corr_empty[1]);
            }

            #pragma unroll
            for (int stage = 0; stage < 2; ++stage) {
                hao_direct_wait(
                    softmax_corr_full[stage],
                    softmax_consumer_phase);
                if (warp::elect_leader()) {
                    arrive(softmax_corr_empty[stage]);
                }
                hao_direct_wait(
                    output_full[stage],
                    output_consumer_phase);
                hao_direct_wait(
                    corr_epi_empty[stage],
                    epi_producer_phase);
                tensor_after_thread_sync();
                const int query_tile =
                    stage == 0
                        ? task.qtile0
                        : task.qtile1;
                if (stage == 0) {
                    upstream_store_output_row_group_owned_fp8<C>(
                        output0,
                        o_smem[0],
                        g,
                        task.batch,
                        query_tile,
                        task.head,
                        row_max_smem[0],
                        row_sum_smem[0],
                        row_group);
                } else {
                    upstream_store_output_row_group_owned_fp8<C>(
                        output1,
                        o_smem[1],
                        g,
                        task.batch,
                        query_tile,
                        task.head,
                        row_max_smem[1],
                        row_sum_smem[1],
                        row_group);
                }
                if (warp::elect_leader()) {
                    arrive(corr_epi_full[stage]);
                    arrive(p_first_o_rescaled[stage]);
                }
            }

            output_consumer_phase ^= 1;
            softmax_consumer_phase ^= 1;
            epi_producer_phase ^= 1;
        }

        if (warp::elect_leader()) {
            arrive(tmem_dealloc);
        }
    } else if (warp_id == 13) {
        int epi_consumer_phase = 0;
        for (int logical_job = int(blockIdx.x);
             logical_job < C::LOGICAL_JOBS;
             logical_job += int(gridDim.x)) {
            const hao_direct_task task =
                hao_direct_decode_task<C>(logical_job);
            hao_direct_wait(
                corr_epi_full[0], epi_consumer_phase);
            upstream_store_output_tma_one_warp(
                o_smem[0],
                g,
                task.batch,
                task.qtile0,
                task.head);
            if (warp::elect_leader()) {
                arrive(corr_epi_empty[0]);
            }
            hao_direct_wait(
                corr_epi_full[1], epi_consumer_phase);
            upstream_store_output_tma_one_warp(
                o_smem[1],
                g,
                task.batch,
                task.qtile1,
                task.head);
            if (warp::elect_leader()) {
                arrive(corr_epi_empty[1]);
            }
            epi_consumer_phase ^= 1;
        }
    }
}

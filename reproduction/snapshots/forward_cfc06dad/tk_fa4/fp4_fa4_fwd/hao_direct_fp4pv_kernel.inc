#pragma once

struct alignas(16) hao_direct_q1_scale_slot {
    float raw_max[128];
};

template <typename C>
struct hao_direct_q1_max_config : C {
    static constexpr int HAO_DIRECT_MX_MAX_SAMPLE_STRIDE =
        C::HAO_DIRECT_MX_Q1_MAX_SAMPLE_STRIDE;
};

template <typename C, int Stage>
struct hao_direct_pair_scale_stage_config : C {
    static constexpr int HAO_DIRECT_MX_PAIR_SCALE_REUSE =
        ((C::HAO_DIRECT_MX_PAIR_SCALE_STAGE_MASK & (1 << Stage))
             ? C::HAO_DIRECT_MX_PAIR_SCALE_REUSE
             : 0) |
        (C::HAO_DIRECT_MX_LAGGED_Q1_SCALE ? 1 : 0) |
        (C::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD ? 1 : 0);
    static constexpr int HAO_DIRECT_MX_Q1_PAIR_PROBE =
        (C::HAO_DIRECT_MX_PAIR_SCALE_STAGE_MASK & (1 << Stage))
            ? C::HAO_DIRECT_MX_Q1_PAIR_PROBE
            : 0;
    static constexpr int HAO_DIRECT_MX_MODE23_NATIVE_DENSITY =
        (C::HAO_DIRECT_MX_Q1_SELF_MAX == 3 ||
         C::HAO_DIRECT_MX_Q1_SELF_MAX == 4) && Stage == 0
            ? 0
            : C::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY;
    static constexpr bool HAO_DIRECT_MX_MODE23_EARLY_NATIVE =
        C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE &&
        (C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_STAGE_MASK &
         (1 << Stage));
};

__device__ __forceinline__ float hao_direct_max3_f32(
    float a,
    float b,
    float c
) {
    float result;
    asm volatile(
        "{max.f32 %0, %1, %2, %3;}"
        : "=f"(result)
        : "f"(a), "f"(b), "f"(c));
    return result;
}

template <typename C>
__device__ __forceinline__ void hao_direct_reduce_q1_score_quarter(
    score_quarter_regs &scores,
    float &result
) {
    if constexpr (C::HAO_DIRECT_MX_Q1_MAX_SAMPLE_STRIDE == 2) {
        float p0 = hao_direct_max3_f32(
            result, scores.v0.x, scores.v0.y);
        float p1 = hao_direct_max3_f32(
            result, scores.v2.x, scores.v2.y);
        float p2 = hao_direct_max3_f32(
            result, scores.v4.x, scores.v4.y);
        float p3 = hao_direct_max3_f32(
            result, scores.v6.x, scores.v6.y);
        p0 = hao_direct_max3_f32(p0, scores.v8.x, scores.v8.y);
        p1 = hao_direct_max3_f32(p1, scores.v10.x, scores.v10.y);
        p2 = hao_direct_max3_f32(p2, scores.v12.x, scores.v12.y);
        p3 = hao_direct_max3_f32(p3, scores.v14.x, scores.v14.y);
        p0 = hao_direct_max3_f32(p0, p1, p2);
        result = hao_direct_max3_f32(result, p0, p3);
    } else if constexpr (
        C::HAO_DIRECT_MX_Q1_MAX_SAMPLE_STRIDE == 3) {
        float p0 = hao_direct_max3_f32(
            result, scores.v0.x, scores.v0.y);
        float p1 = hao_direct_max3_f32(
            result, scores.v1.x, scores.v1.y);
        float p2 = hao_direct_max3_f32(
            result, scores.v2.x, scores.v2.y);
        float p3 = hao_direct_max3_f32(
            result, scores.v4.x, scores.v4.y);
        p0 = hao_direct_max3_f32(p0, scores.v5.x, scores.v5.y);
        p1 = hao_direct_max3_f32(p1, scores.v6.x, scores.v6.y);
        p2 = hao_direct_max3_f32(p2, scores.v8.x, scores.v8.y);
        p3 = hao_direct_max3_f32(p3, scores.v9.x, scores.v9.y);
        p0 = hao_direct_max3_f32(p0, scores.v10.x, scores.v10.y);
        p1 = hao_direct_max3_f32(p1, scores.v12.x, scores.v12.y);
        p2 = hao_direct_max3_f32(p2, scores.v13.x, scores.v13.y);
        p3 = hao_direct_max3_f32(p3, scores.v14.x, scores.v14.y);
        p0 = hao_direct_max3_f32(p0, p1, p2);
        result = hao_direct_max3_f32(result, p0, p3);
    } else {
        upstream_reduce_score_quarter<hao_direct_q1_max_config<C>>(
            scores, result);
    }
}

template <int ProbeCount>
__device__ __forceinline__ float hao_direct_q1_pair_scale_probe(
    const score_quarter_regs &scores
) {
    static_assert(
        ProbeCount == 4 || ProbeCount == 8 || ProbeCount == 16);
    constexpr float NEG_INF =
        base_types::constants<float>::neg_infty();
    if constexpr (ProbeCount == 4) {
        const float p0 = hao_direct_max3_f32(
            scores.v0.x, scores.v4.x, scores.v8.x);
        return hao_direct_max3_f32(
            NEG_INF, p0, scores.v12.x);
    } else if constexpr (ProbeCount == 8) {
        const float p0 = hao_direct_max3_f32(
            NEG_INF, scores.v0.x, scores.v0.y);
        const float p1 = hao_direct_max3_f32(
            NEG_INF, scores.v4.x, scores.v4.y);
        const float p2 = hao_direct_max3_f32(
            NEG_INF, scores.v8.x, scores.v8.y);
        const float p3 = hao_direct_max3_f32(
            NEG_INF, scores.v12.x, scores.v12.y);
        return hao_direct_max3_f32(
            NEG_INF,
            hao_direct_max3_f32(p0, p1, p2),
            p3);
    } else {
        float p0 = hao_direct_max3_f32(
            NEG_INF, scores.v0.x, scores.v0.y);
        float p1 = hao_direct_max3_f32(
            NEG_INF, scores.v2.x, scores.v2.y);
        float p2 = hao_direct_max3_f32(
            NEG_INF, scores.v4.x, scores.v4.y);
        float p3 = hao_direct_max3_f32(
            NEG_INF, scores.v6.x, scores.v6.y);
        p0 = hao_direct_max3_f32(p0, scores.v8.x, scores.v8.y);
        p1 = hao_direct_max3_f32(p1, scores.v10.x, scores.v10.y);
        p2 = hao_direct_max3_f32(p2, scores.v12.x, scores.v12.y);
        p3 = hao_direct_max3_f32(p3, scores.v14.x, scores.v14.y);
        return hao_direct_max3_f32(
            NEG_INF,
            hao_direct_max3_f32(p0, p1, p2),
            p3);
    }
}

__device__ __forceinline__ void hao_direct_fp4pv_wait(
    semaphore &sem,
    int phase
) {
#if TK_HAO_DIRECT_WAIT_HINT == 10000000
    const uint32_t mbar_ptr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&sem));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "HAO_DIRECT_FP4PV_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 "
        "P1, [%0], %1, 10000000;\n"
        "@P1 bra.uni HAO_DIRECT_FP4PV_WAIT_DONE;\n"
        "bra.uni HAO_DIRECT_FP4PV_WAIT;\n"
        "HAO_DIRECT_FP4PV_WAIT_DONE:\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(phase)
        : "memory");
#else
    wait(sem, phase);
#endif
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
    int head;
    int pair;
    if constexpr (C::TASK_ORDER == 1) {
        head = local / PAIRS_PER_BATCH;
        pair = local % PAIRS_PER_BATCH;
    } else {
        head = local % C::HEADS;
        pair = local / C::HEADS;
    }
    return {batch, head, pair, pair * 2, pair * 2 + 1};
}

template <typename C>
struct globals_fp4pv_hao_direct {
    using Parent = globals_fp4pv_upstream_fp8pv<C>;
    using q_tile = typename Parent::q_tile;
    using q_sc_tile = std::conditional_t<
        C::HAO_DIRECT_NV_QK_COMPACT_FOLDED_SCALES,
        st_hf<1, 256, false>,
        std::conditional_t<
            C::HAO_DIRECT_NVFP4_QK,
            typename Parent::q_sc_tile,
            st_hf<1, 256, false>>>;
    using k_tile = typename Parent::k_tile;
    using k_sc_tile = std::conditional_t<
        C::HAO_DIRECT_NV_QK_COMPACT_FOLDED_SCALES,
        st_hf<1, 256, false>,
        std::conditional_t<
            C::HAO_DIRECT_NVFP4_QK,
            typename Parent::k_sc_tile,
            st_hf<1, 256, false>>>;
    using o_tile = typename Parent::o_tile;
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
    using o_gl = typename Parent::o_gl;
    using lse_gl = typename Parent::lse_gl;
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using v_sc_tile = std::conditional_t<
        C::HAO_DIRECT_NVFP4_PV,
        st_hf<C::QK_SCALE_CHUNKS, 256, false>,
        st_hf<1, 256, false>>;
    using p_sc_tile = v_sc_tile;
    using v_fp4_gl = gl<
        fp4e2m1_2, -1, -1, -1, -1,
        tma::descriptor<v_fp4_tile, dim::ROW>>;
    using v_sc_gl = gl<
        half, -1, -1, -1, 256,
        tma::descriptor<v_sc_tile, dim::ROW>>;

    q_gl q;
    q_sc_gl q_sc;
    q_sg_gl q_sg;
    k_gl k;
    k_sc_gl k_sc;
    k_sg_gl k_sg;
    v_fp4_gl v_fp4;
    v_sc_gl v_sc;
    float v_global_decode = 1.0f;
    o_gl o;
    lse_gl lse;
    int disable_lse_store = 0;
};

template <typename C>
__device__ __forceinline__ float qk_scale_log2(
    const globals_fp4pv_hao_direct<C> &g,
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
union hao_direct_fp4_kv_payload_slot {
    using G = globals_fp4pv_hao_direct<C>;
    typename G::k_tile k;
    typename G::v_fp4_tile v;
};

template <typename C>
union hao_direct_fp4_kv_scale_slot {
    using G = globals_fp4pv_hao_direct<C>;
    typename G::k_sc_tile k;
    typename G::v_sc_tile v;
};

template <typename C>
struct hao_direct_q3_smem_slot {
    uint32_t word[8][C::Mb];
};

struct hao_direct_q3_fp8_words {
    uint32_t word[8];
};

struct hao_direct_nv_denom_half_snapshot {
    uint32_t word[2][4];
    uint32_t scale_half;
};

struct hao_direct_nv_first_payload_smem {
    uint32_t word[128][8];
};

struct hao_direct_mx_denom_payload_smem {
    // Four padding words keep every row 16-byte aligned while rotating row
    // starts over eight shared-memory bank groups.
    uint32_t word[128][20];
};

using hao_direct_sparse_p_tile = st_fp4e2m1_2<128, 32>;

struct hao_direct_sparse_selector_lut {
    uint8_t data[256];
};

struct hao_direct_sparse_p {
    static constexpr bool SPARSE = true;
    uint32_t addr;
    hao_direct_sparse_p_tile *payload;
    uint32_t tail_addr;
    hao_direct_sparse_p_tile *tail_payload;
    sv_hf<128> *metadata_halves;
    hao_direct_sparse_selector_lut *selector_lut;
};

template <typename P>
concept hao_direct_sparse_p_type =
    std::is_same_v<
        std::remove_cv_t<P>,
        hao_direct_sparse_p>;

template <typename C>
struct hao_direct_pipeline_smem_layout {
    using G = globals_fp4pv_hao_direct<C>;
    using Payload = hao_direct_fp4_kv_payload_slot<C>;
    using Scale = hao_direct_fp4_kv_scale_slot<C>;
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
        if constexpr (C::HAO_DIRECT_SPARSE_PV) {
            offset = append(
                offset,
                (C::HAO_DIRECT_SPARSE_SPLIT_K128 ? 4 : 2) *
                    int(sizeof(hao_direct_sparse_p_tile)));
            offset = append(
                offset,
                4 * int(sizeof(sv_hf<C::Mb>)));
            if constexpr (
                C::HAO_DIRECT_SPARSE_SELECTOR_MODE >= 4) {
                offset = append(
                    offset,
                    int(sizeof(hao_direct_sparse_selector_lut)));
            }
        }
        offset = append(
            offset, 2 * int(sizeof(typename G::o_tile)));
        offset = append(
            offset, 2 * int(sizeof(typename G::p_sc_tile)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        if constexpr (
            C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF ||
            C::HAO_DIRECT_MX_DENOM_EARLY_SMEM) {
            offset = append(
                offset,
                2 * int(sizeof(hao_direct_nv_first_payload_smem)));
        }
        if constexpr (C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM) {
            offset = append(
                offset,
                2 * int(sizeof(hao_direct_mx_denom_payload_smem)));
        }
        if constexpr (C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG) {
            offset = append(
                offset,
                2 * int(sizeof(hao_direct_q3_smem_slot<C>)));
        }
        return offset + ALIGNMENT;
    }
};

template <typename C>
__device__ __forceinline__ void hao_direct_store_q3_fp8_smem(
    hao_direct_q3_smem_slot<C> &dst,
    int row_group,
    const score_quarter_regs &scores
) {
    static_assert(
        sizeof(score_quarter_regs) == 16 * sizeof(float2));
    const int row = row_group * 32 + warp::laneid();
    const float2 *values =
        reinterpret_cast<const float2 *>(&scores);
    #pragma unroll
    for (int word = 0; word < 8; ++word) {
        dst.word[word][row] =
            upstream_pack_fp8_four(
                values[2 * word],
                values[2 * word + 1]);
    }
}

template <typename C>
__device__ __forceinline__ void hao_direct_load_q3_fp8_smem(
    score_quarter_regs &scores,
    const hao_direct_q3_smem_slot<C> &src,
    int row_group
) {
    static_assert(
        sizeof(score_quarter_regs) == 16 * sizeof(float2));
    const int row = row_group * 32 + warp::laneid();
    hao_direct_q3_fp8_words packed_words;
    #pragma unroll
    for (int word = 0; word < 8; ++word) {
        packed_words.word[word] = src.word[word][row];
    }
    float2 *values = reinterpret_cast<float2 *>(&scores);
    #pragma unroll
    for (int word = 0; word < 8; ++word) {
        const uint32_t packed = packed_words.word[word];
        uint32_t half_lo;
        uint32_t half_hi;
        asm volatile(
            "{\n"
            ".reg .b16 lo, hi;\n"
            "mov.b32 {lo, hi}, %2;\n"
            "cvt.rn.f16x2.e4m3x2 %0, lo;\n"
            "cvt.rn.f16x2.e4m3x2 %1, hi;\n"
            "}\n"
            : "=r"(half_lo), "=r"(half_hi)
            : "r"(packed));
        values[2 * word] = __half22float2(
            *reinterpret_cast<const half_2 *>(&half_lo));
        values[2 * word + 1] = __half22float2(
            *reinterpret_cast<const half_2 *>(&half_hi));
    }
}

template <typename P>
__device__ __forceinline__ void hao_direct_store_q3_fp8_tmem(
    const P &p,
    int row_group,
    const score_quarter_regs &scores
) {
    static_assert(
        sizeof(score_quarter_regs) == 16 * sizeof(float2));
    const float2 *values =
        reinterpret_cast<const float2 *>(&scores);
    hao_direct_q3_fp8_words packed;
    #pragma unroll
    for (int word = 0; word < 8; ++word) {
        packed.word[word] =
            upstream_pack_fp8_four(
                values[2 * word],
                values[2 * word + 1]);
    }
    const uint32_t addr =
        p.addr + (row_group * 32 << 16) + 8;
    asm volatile(
        "{tcgen05.st.sync.aligned.32x32b.x8.b32 "
        "[%0], {%1,%2,%3,%4,%5,%6,%7,%8};}"
        :: "r"(addr),
           "r"(packed.word[0]), "r"(packed.word[1]),
           "r"(packed.word[2]), "r"(packed.word[3]),
           "r"(packed.word[4]), "r"(packed.word[5]),
           "r"(packed.word[6]), "r"(packed.word[7]));
}

template <typename P>
__device__ __forceinline__ void hao_direct_load_q3_fp8_tmem(
    hao_direct_q3_fp8_words &packed,
    const P &p,
    int row_group
) {
    const uint32_t addr =
        p.addr + (row_group * 32 << 16) + 8;
    asm volatile(
        "{tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];}"
        : "=r"(packed.word[0]), "=r"(packed.word[1]),
          "=r"(packed.word[2]), "=r"(packed.word[3]),
          "=r"(packed.word[4]), "=r"(packed.word[5]),
          "=r"(packed.word[6]), "=r"(packed.word[7])
        : "r"(addr));
}

__device__ __forceinline__ void hao_direct_unpack_q3_fp8_words(
    score_quarter_regs &scores,
    const hao_direct_q3_fp8_words &packed_words
) {
    float2 *values = reinterpret_cast<float2 *>(&scores);
    #pragma unroll
    for (int word = 0; word < 8; ++word) {
        const uint32_t packed = packed_words.word[word];
        uint32_t half_lo;
        uint32_t half_hi;
        asm volatile(
            "{\n"
            ".reg .b16 lo, hi;\n"
            "mov.b32 {lo, hi}, %2;\n"
            "cvt.rn.f16x2.e4m3x2 %0, lo;\n"
            "cvt.rn.f16x2.e4m3x2 %1, hi;\n"
            "}\n"
            : "=r"(half_lo), "=r"(half_hi)
            : "r"(packed));
        values[2 * word] = __half22float2(
            *reinterpret_cast<const half_2 *>(&half_lo));
        values[2 * word + 1] = __half22float2(
            *reinterpret_cast<const half_2 *>(&half_hi));
    }
}

template <typename TT, typename ST>
__device__ __forceinline__ void hao_direct_load_mxfp4_scale_ranklocal(
    TT &dst,
    const ST &src
) {
    auto &source =
        *reinterpret_cast<const st_fp8e8m0<32, 16, false> *>(
            &src.data[0]);
    fp4pv_load_mxnv_scale_async_ranklocal(dst, source);
}

template <int Half, bool Wait = true>
__device__ __forceinline__ void
hao_direct_store_mxfp4_p_scale_half_packed(
    uint32_t page_addr,
    int reader_warp,
    uint32_t packed
) {
    static_assert(Half == 0 || Half == 1);
    full_tt_fp8e8m0<16> page(page_addr);
    fp4pv_store_mxfp4_scale_tmem_32x32b_x1(
        page, reader_warp, packed);
    if constexpr (Wait) {
        fp4pv_tmem_store_wait();
    }
}

template <int Half, bool Wait = true>
__device__ __forceinline__ void
hao_direct_store_mxfp4_p_scale_half_register(
    uint32_t page_addr,
    int reader_warp,
    uint8_t first_quarter,
    uint8_t second_quarter
) {
    static_assert(Half == 0 || Half == 1);
    uint32_t packed =
        static_cast<uint32_t>(first_quarter) |
        (static_cast<uint32_t>(second_quarter) << 8);
    if constexpr (Half == 1) {
        packed <<= 16;
    }
    hao_direct_store_mxfp4_p_scale_half_packed<Half, Wait>(
        page_addr, reader_warp, packed);
}

template <typename PScale>
__device__ __forceinline__ void
hao_direct_store_mxfp4_scale_byte_linear(
    PScale &p_scale_smem,
    int row,
    int quarter,
    uint8_t e8m0
) {
    const uint32_t base = __cvta_generic_to_shared(
        &reinterpret_cast<uint8_t &>(p_scale_smem.data[0]));
    const uint32_t address =
        base + static_cast<uint32_t>((row & 127) * 4 + quarter);
    asm volatile(
        "st.shared.b8 [%0], %1;"
        :: "r"(address), "r"(static_cast<uint32_t>(e8m0)));
}

template <typename PScale>
__device__ __forceinline__ uint8_t
hao_direct_load_mxfp4_scale_byte_linear(
    PScale &p_scale_smem,
    int row,
    int quarter
) {
    const uint32_t base = __cvta_generic_to_shared(
        &reinterpret_cast<uint8_t &>(p_scale_smem.data[0]));
    const uint32_t address =
        base + static_cast<uint32_t>((row & 127) * 4 + quarter);
    uint32_t e8m0;
    asm volatile(
        "ld.shared.u8 %0, [%1];"
        : "=r"(e8m0)
        : "r"(address));
    return static_cast<uint8_t>(e8m0);
}

template <typename PScale>
__device__ __forceinline__ uint8_t
hao_direct_load_mxfp4_scale_byte_swizzled(
    PScale &p_scale_smem,
    int row,
    int quarter
) {
    const int source = fp4pv_scale_swizzle_idx(row & 127, quarter);
    const uint32_t base = __cvta_generic_to_shared(
        &reinterpret_cast<uint8_t &>(p_scale_smem.data[0]));
    uint32_t e8m0;
    asm volatile(
        "ld.shared.u8 %0, [%1];"
        : "=r"(e8m0)
        : "r"(base + static_cast<uint32_t>(source)));
    return static_cast<uint8_t>(e8m0);
}

template <typename PScale>
__device__ __forceinline__ uint32_t
hao_direct_load_mxfp4_scale_word_linear(
    PScale &p_scale_smem,
    int row
) {
    const uint32_t base = __cvta_generic_to_shared(
        &reinterpret_cast<uint8_t &>(p_scale_smem.data[0]));
    const uint32_t address =
        base + static_cast<uint32_t>((row & 127) * 4);
    uint32_t packed;
    asm volatile(
        "ld.shared.b32 %0, [%1];"
        : "=r"(packed)
        : "r"(address));
    return packed;
}

template <
    int Half,
    bool Wait = true,
    bool LinearScratch = false,
    typename PScale>
__device__ __forceinline__ void
hao_direct_store_mxfp4_p_scale_half_direct(
    uint32_t page_addr,
    PScale &p_scale_smem,
    int reader_warp
) {
    static_assert(Half == 0 || Half == 1);
    uint32_t packed;
    if constexpr (LinearScratch) {
        packed = hao_direct_load_mxfp4_scale_word_linear(
            p_scale_smem, reader_warp * 32 + warp::laneid());
    } else {
        packed = fp4pv_load_mxfp4_scale_word_swizzled(
            p_scale_smem, reader_warp * 32 + warp::laneid());
    }
    if constexpr (Half == 0) {
        packed &= 0x0000ffffu;
    } else {
        packed &= 0xffff0000u;
    }
    hao_direct_store_mxfp4_p_scale_half_packed<Half, Wait>(
        page_addr, reader_warp, packed);
}

template <int Chunk, typename ST>
__device__ __forceinline__ void hao_direct_load_nvfp4_scale_ranklocal(
    uint32_t dst_addr,
    const ST &src
) {
    static_assert(Chunk == 0 || Chunk == 1);
    full_tt_fp8e4m3<16> dst(dst_addr);
    fp4pv_load_q_scale_chunk_ranklocal(dst, src, Chunk);
}

template <int Half, typename PScale>
__device__ __forceinline__ void
hao_direct_store_nvfp4_p_scale_half_direct(
    uint32_t page_addr,
    PScale &p_scale_smem,
    int reader_warp
) {
    static_assert(Half == 0 || Half == 1);
    const uint32_t base = __cvta_generic_to_shared(
        &reinterpret_cast<uint8_t &>(p_scale_smem.data[0]));
    const uint32_t address =
        base + warp::laneid() * 16 + reader_warp * 4 +
        Half * 512;
    uint32_t packed;
    asm volatile(
        "ld.shared.b32 %0, [%1];"
        : "=r"(packed)
        : "r"(address));
    const uint32_t tmem_addr =
        page_addr + static_cast<uint32_t>(reader_warp);
    asm volatile(
        "{tcgen05.st.sync.aligned.32x32b.x1.b32 "
        "[%0], {%1};}"
        :: "r"(tmem_addr), "r"(packed));
}

template <typename PScale>
__device__ __forceinline__ void
hao_direct_store_sparse_nvfp4_p_scale_direct(
    uint32_t page_addr,
    PScale &p_scale_smem,
    int reader_warp
) {
    const uint32_t base = __cvta_generic_to_shared(
        &reinterpret_cast<uint8_t &>(p_scale_smem.data[0]));
    const uint32_t address =
        base + warp::laneid() * 16 + reader_warp * 4;
    const uint32_t tail_address = address + 512;
    uint32_t first;
    uint32_t tail;
    asm volatile(
        "ld.shared.b32 %0, [%2];\n\t"
        "ld.shared.b32 %1, [%3];"
        : "=r"(first), "=r"(tail)
        : "r"(address), "r"(tail_address));
    // Quarter-local scale encoding duplicates each block-32 scale. Select
    // bytes q0[0], q1[0], q2[0], q3[0] into one sparse K128 scale word.
    const uint32_t packed = __byte_perm(first, tail, 0x6420);
    const uint32_t tmem_addr =
        page_addr + static_cast<uint32_t>(reader_warp);
    asm volatile(
        "{tcgen05.st.sync.aligned.32x32b.x1.b32 "
        "[%0], {%1};}"
        :: "r"(tmem_addr), "r"(packed));
}

template <int Half, typename PScale>
__device__ __forceinline__ void
hao_direct_store_sparse_nvfp4_p_scale_half_direct(
    uint32_t page_addr,
    PScale &p_scale_smem,
    int reader_warp
) {
    static_assert(Half == 0 || Half == 1);
    const uint32_t base = __cvta_generic_to_shared(
        &reinterpret_cast<uint8_t &>(p_scale_smem.data[0]));
    const uint32_t address =
        base + warp::laneid() * 16 + reader_warp * 4 +
        Half * 512;
    uint32_t duplicated;
    asm volatile(
        "ld.shared.b32 %0, [%1];"
        : "=r"(duplicated)
        : "r"(address));
    // Keep the owned block-32 scales in their natural K64 half. Scale values
    // for the permanently zero half are irrelevant, so repeat the owned pair.
    const uint32_t packed =
        __byte_perm(duplicated, duplicated, 0x2020);
    const uint32_t tmem_addr =
        page_addr + static_cast<uint32_t>(reader_warp);
    asm volatile(
        "{tcgen05.st.sync.aligned.32x32b.x1.b32 "
        "[%0], {%1};}"
        :: "r"(tmem_addr), "r"(packed));
}

template <int Half>
__device__ __forceinline__ void
hao_direct_store_nvfp4_p_scale_half_register(
    uint32_t page_addr,
    int reader_warp,
    uint16_t first_quarter,
    uint16_t second_quarter
) {
    static_assert(Half == 0 || Half == 1);
    const uint32_t packed =
        static_cast<uint32_t>(first_quarter) |
        (static_cast<uint32_t>(second_quarter) << 16);
    const uint32_t tmem_addr =
        page_addr + static_cast<uint32_t>(reader_warp);
    asm volatile(
        "{tcgen05.st.sync.aligned.32x32b.x1.b32 "
        "[%0], {%1};}"
        :: "r"(tmem_addr), "r"(packed));
}

template <typename ST>
__device__ __forceinline__ uint32_t
hao_direct_fp4_smem_descriptor_low(const ST &tile) {
    static_assert(ST::swizzle);
    const uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&tile.data[0]));
    // K-major FP4 descriptors use a 16-byte leading dimension. The stride,
    // Blackwell-valid bit, and swizzle mode live in the constant high word
    // materialized inside each MMA assembly block.
    return ((smem_addr & 0x3ffffu) >> 4) | (1u << 16);
}

template <typename ST>
__device__ __forceinline__ uint32_t
hao_direct_scale_smem_descriptor_low(const ST &tile) {
    static_assert(!ST::swizzle);
    const uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&tile.data[0]));
    return ((smem_addr & 0x3ffffu) >> 4) | (8u << 16);
}

template <typename ST>
consteval uint32_t hao_direct_fp4_smem_descriptor_high() {
    constexpr uint32_t stride_bytes =
        ST::swizzle_bytes == 128 ? 1024 :
        ST::swizzle_bytes == 64 ? 512 : 256;
    constexpr uint32_t swizzle_mode =
        ST::swizzle_bytes == 128 ? 1 :
        ST::swizzle_bytes == 64 ? 2 : 3;
    return (stride_bytes >> 4) |
           (1u << (46 - 32)) |
           (swizzle_mode << (62 - 32));
}

template <typename ST>
consteval uint32_t hao_direct_fp4_k64_descriptor_offset() {
    if constexpr (
        ST::swizzle_bytes == 128 ||
        ST::swizzle_bytes == 64) {
        return 32 >> 4;
    } else {
        return (ST::rows / kittens::TILE_ROW_DIM<typename ST::T>) *
               (512 >> 4);
    }
}

template <typename C, typename Score, typename Q, typename K>
__device__ __forceinline__ void
hao_direct_issue_qk_nvfp4_one_folded_scale(
    Score &score,
    Q &q,
    K &k,
    uint32_t scale_base
) {
    static_assert(
        C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 1 ||
        C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 2);
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e4m3;
    using T_D = typename Score::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
    static_assert(Q::swizzle_bytes == K::swizzle_bytes);
    constexpr uint32_t DESC_HI =
        hao_direct_fp4_smem_descriptor_high<Q>();
    constexpr uint32_t K64_DESC_OFFSET =
        hao_direct_fp4_k64_descriptor_offset<Q>();
    const uint32_t q_desc_low =
        hao_direct_fp4_smem_descriptor_low(q);
    const uint32_t k_desc_low =
        hao_direct_fp4_smem_descriptor_low(k);
    if constexpr (C::HAO_DIRECT_NV_Q_FOLDED_K64_SCALES) {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb;\n\t"
        ".reg .b64 q_desc, k_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "add.u32 sfb, %3, 12;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET)
        : "memory");
    } else {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb;\n\t"
        ".reg .b64 q_desc, k_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "add.u32 sfa, %3, 4;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET)
        : "memory");
    }
}

template <
    typename C,
    typename Score,
    typename Q,
    typename K,
    typename QScale,
    typename KScale>
__device__ __forceinline__ void
hao_direct_issue_qk_nvfp4_compact_scales(
    Score &score,
    Q &q,
    K &k,
    QScale &q_scale_smem,
    KScale &k_scale_smem,
    uint32_t scale_base,
    semaphore &finished
) {
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e4m3;
    using T_D = typename Score::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
    static_assert(Q::swizzle_bytes == K::swizzle_bytes);
    constexpr uint32_t DESC_HI =
        hao_direct_fp4_smem_descriptor_high<Q>();
    constexpr uint32_t K64_DESC_OFFSET =
        hao_direct_fp4_k64_descriptor_offset<Q>();
    const uint32_t q_desc_low =
        hao_direct_fp4_smem_descriptor_low(q);
    const uint32_t k_desc_low =
        hao_direct_fp4_smem_descriptor_low(k);
    const uint32_t q_scale_desc_low =
        hao_direct_scale_smem_descriptor_low(q_scale_smem);
    const uint32_t k_scale_desc_low =
        hao_direct_scale_smem_descriptor_low(k_scale_smem);
    constexpr uint32_t SCALE_DESC_HI =
        8u | (1u << (46 - 32));
    if constexpr (C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 2) {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb, "
        "scale_hi, scale_lo, scale_dst;\n\t"
        ".reg .b64 q_desc, k_desc, scale_desc;\n\t"
        "mov.b32 scale_hi, %9;\n\t"
        "mov.b32 scale_lo, %7;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%3], scale_desc;\n\t"
        "add.u32 scale_lo, %7, 32;\n\t"
        "add.u32 scale_dst, %3, 4;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 scale_lo, %8;\n\t"
        "add.u32 scale_dst, %3, 8;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "add.u32 sfa, %3, 4;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET),
           "r"(q_scale_desc_low),
           "r"(k_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    } else if constexpr (
        C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 1) {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb, "
        "scale_hi, scale_lo, scale_dst;\n\t"
        ".reg .b64 q_desc, k_desc, scale_desc;\n\t"
        "mov.b32 scale_hi, %9;\n\t"
        "mov.b32 scale_lo, %7;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%3], scale_desc;\n\t"
        "mov.b32 scale_lo, %8;\n\t"
        "add.u32 scale_dst, %3, 8;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "add.u32 scale_lo, %8, 32;\n\t"
        "add.u32 scale_dst, %3, 12;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "add.u32 sfb, %3, 12;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET),
           "r"(q_scale_desc_low),
           "r"(k_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    } else if constexpr (C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES) {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb, "
        "scale_hi, scale_lo, scale_dst;\n\t"
        ".reg .b64 q_desc, k_desc, scale_desc;\n\t"
        "mov.b32 scale_hi, %9;\n\t"
        "mov.b32 scale_lo, %7;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%3], scale_desc;\n\t"
        "mov.b32 scale_lo, %8;\n\t"
        "add.u32 scale_dst, %3, 8;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET),
           "r"(q_scale_desc_low),
           "r"(k_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    } else {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb, "
        "scale_hi, scale_lo, scale_dst;\n\t"
        ".reg .b64 q_desc, k_desc, scale_desc;\n\t"
        "mov.b32 scale_hi, %9;\n\t"
        "mov.b32 scale_lo, %7;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%3], scale_desc;\n\t"
        "add.u32 scale_lo, %7, 32;\n\t"
        "add.u32 scale_dst, %3, 4;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 scale_lo, %8;\n\t"
        "add.u32 scale_dst, %3, 8;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "add.u32 scale_lo, %8, 32;\n\t"
        "add.u32 scale_dst, %3, 12;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "add.u32 sfa, %3, 4;\n\t"
        "add.u32 sfb, %3, 12;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET),
           "r"(q_scale_desc_low),
           "r"(k_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    }
    detail::tcgen05::commit<1>(finished);
}

template <typename C, typename QScale, typename KScale>
__device__ __forceinline__ void
hao_direct_preload_qk_nvfp4_compact_scales(
    QScale &q_scale_smem,
    KScale &k_scale_smem,
    uint32_t scale_base
) {
    const uint32_t q_scale_desc_low =
        hao_direct_scale_smem_descriptor_low(q_scale_smem);
    const uint32_t k_scale_desc_low =
        hao_direct_scale_smem_descriptor_low(k_scale_smem);
    constexpr uint32_t SCALE_DESC_HI =
        8u | (1u << (46 - 32));
    if constexpr (C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 2) {
        asm volatile(
        "{\n\t"
        ".reg .b32 scale_hi, scale_lo, scale_dst;\n\t"
        ".reg .b64 scale_desc;\n\t"
        "mov.b32 scale_hi, %3;\n\t"
        "mov.b32 scale_lo, %1;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%0], scale_desc;\n\t"
        "add.u32 scale_lo, %1, 32;\n\t"
        "add.u32 scale_dst, %0, 4;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 scale_lo, %2;\n\t"
        "add.u32 scale_dst, %0, 8;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "}\n"
        :: "r"(scale_base),
           "r"(q_scale_desc_low),
           "r"(k_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    } else if constexpr (
        C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 1) {
        asm volatile(
        "{\n\t"
        ".reg .b32 scale_hi, scale_lo, scale_dst;\n\t"
        ".reg .b64 scale_desc;\n\t"
        "mov.b32 scale_hi, %3;\n\t"
        "mov.b32 scale_lo, %1;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%0], scale_desc;\n\t"
        "mov.b32 scale_lo, %2;\n\t"
        "add.u32 scale_dst, %0, 8;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "add.u32 scale_lo, %2, 32;\n\t"
        "add.u32 scale_dst, %0, 12;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "}\n"
        :: "r"(scale_base),
           "r"(q_scale_desc_low),
           "r"(k_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    } else if constexpr (C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES) {
        if constexpr (C::HAO_DIRECT_NV_QK_PRELOAD_PAGE_MASK == 3) {
            asm volatile(
            "{\n\t"
            ".reg .b32 scale_hi, scale_lo, scale_dst;\n\t"
            ".reg .b64 scale_desc;\n\t"
            "mov.b32 scale_hi, %3;\n\t"
            "mov.b32 scale_lo, %1;\n\t"
            "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
            "tcgen05.cp.cta_group::1.32x128b.warpx4 "
            "[%0], scale_desc;\n\t"
            "mov.b32 scale_lo, %2;\n\t"
            "add.u32 scale_dst, %0, 8;\n\t"
            "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
            "tcgen05.cp.cta_group::1.32x128b.warpx4 "
            "[scale_dst], scale_desc;\n\t"
            "}\n"
            :: "r"(scale_base),
               "r"(q_scale_desc_low),
               "r"(k_scale_desc_low),
               "n"(SCALE_DESC_HI)
            : "memory");
        } else if constexpr (
            C::HAO_DIRECT_NV_QK_PRELOAD_PAGE_MASK == 1) {
            asm volatile(
            "{\n\t"
            ".reg .b32 scale_hi;\n\t"
            ".reg .b64 scale_desc;\n\t"
            "mov.b32 scale_hi, %2;\n\t"
            "mov.b64 scale_desc, {%1, scale_hi};\n\t"
            "tcgen05.cp.cta_group::1.32x128b.warpx4 "
            "[%0], scale_desc;\n\t"
            "}\n"
            :: "r"(scale_base),
               "r"(q_scale_desc_low),
               "n"(SCALE_DESC_HI)
            : "memory");
        } else {
            asm volatile(
            "{\n\t"
            ".reg .b32 scale_hi, scale_dst;\n\t"
            ".reg .b64 scale_desc;\n\t"
            "mov.b32 scale_hi, %2;\n\t"
            "add.u32 scale_dst, %0, 8;\n\t"
            "mov.b64 scale_desc, {%1, scale_hi};\n\t"
            "tcgen05.cp.cta_group::1.32x128b.warpx4 "
            "[scale_dst], scale_desc;\n\t"
            "}\n"
            :: "r"(scale_base),
               "r"(k_scale_desc_low),
               "n"(SCALE_DESC_HI)
            : "memory");
        }
    } else {
        asm volatile(
        "{\n\t"
        ".reg .b32 scale_hi, scale_lo, scale_dst;\n\t"
        ".reg .b64 scale_desc;\n\t"
        "mov.b32 scale_hi, %3;\n\t"
        "mov.b32 scale_lo, %1;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%0], scale_desc;\n\t"
        "add.u32 scale_lo, %1, 32;\n\t"
        "add.u32 scale_dst, %0, 4;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 scale_lo, %2;\n\t"
        "add.u32 scale_dst, %0, 8;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "add.u32 scale_lo, %2, 32;\n\t"
        "add.u32 scale_dst, %0, 12;\n\t"
        "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "}\n"
        :: "r"(scale_base),
           "r"(q_scale_desc_low),
           "r"(k_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    }
}

template <
    typename C,
    typename Score,
    typename Q,
    typename K,
    typename QScale,
    typename KScale>
__device__ __forceinline__ void
hao_direct_issue_qk_nvfp4_preloaded_scales(
    Score &score,
    Q &q,
    K &k,
    QScale &q_scale_smem,
    KScale &k_scale_smem,
    uint32_t scale_base,
    semaphore &finished
) {
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e4m3;
    using T_D = typename Score::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
    static_assert(Q::swizzle_bytes == K::swizzle_bytes);
    constexpr uint32_t DESC_HI =
        hao_direct_fp4_smem_descriptor_high<Q>();
    constexpr uint32_t K64_DESC_OFFSET =
        hao_direct_fp4_k64_descriptor_offset<Q>();
    const uint32_t q_desc_low =
        hao_direct_fp4_smem_descriptor_low(q);
    const uint32_t k_desc_low =
        hao_direct_fp4_smem_descriptor_low(k);
    const uint32_t q_scale_desc_low =
        hao_direct_scale_smem_descriptor_low(q_scale_smem);
    const uint32_t k_scale_desc_low =
        hao_direct_scale_smem_descriptor_low(k_scale_smem);
    constexpr uint32_t SCALE_DESC_HI =
        8u | (1u << (46 - 32));
    if constexpr (
        C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 1 ||
        C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 2) {
        hao_direct_issue_qk_nvfp4_one_folded_scale<C>(
            score,
            q,
            k,
            scale_base);
    } else if constexpr (
        C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES &&
        C::HAO_DIRECT_NV_QK_PRELOAD_PAGE_MASK == 1) {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb, "
        "scale_hi, scale_dst;\n\t"
        ".reg .b64 q_desc, k_desc, scale_desc;\n\t"
        "mov.b32 scale_hi, %8;\n\t"
        "add.u32 scale_dst, %3, 8;\n\t"
        "mov.b64 scale_desc, {%7, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[scale_dst], scale_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET),
           "r"(k_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    } else if constexpr (
        C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES &&
        C::HAO_DIRECT_NV_QK_PRELOAD_PAGE_MASK == 2) {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb, "
        "scale_hi;\n\t"
        ".reg .b64 q_desc, k_desc, scale_desc;\n\t"
        "mov.b32 scale_hi, %8;\n\t"
        "mov.b64 scale_desc, {%7, scale_hi};\n\t"
        "tcgen05.cp.cta_group::1.32x128b.warpx4 "
        "[%3], scale_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET),
           "r"(q_scale_desc_low),
           "n"(SCALE_DESC_HI)
        : "memory");
    } else if constexpr (C::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES) {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb;\n\t"
        ".reg .b64 q_desc, k_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET)
        : "memory");
    } else {
        asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 desc_hi, q_lo, k_lo, idesc, sfa, sfb;\n\t"
        ".reg .b64 q_desc, k_desc;\n\t"
        "mov.b32 desc_hi, %5;\n\t"
        "mov.b32 idesc, %4;\n\t"
        "mov.b32 sfa, %3;\n\t"
        "add.u32 sfb, %3, 8;\n\t"
        "mov.b64 q_desc, {%1, desc_hi};\n\t"
        "mov.b64 k_desc, {%2, desc_hi};\n\t"
        "setp.eq.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "add.u32 q_lo, %1, %6;\n\t"
        "add.u32 k_lo, %2, %6;\n\t"
        "add.u32 sfa, %3, 4;\n\t"
        "add.u32 sfb, %3, 12;\n\t"
        "mov.b64 q_desc, {q_lo, desc_hi};\n\t"
        "mov.b64 k_desc, {k_lo, desc_hi};\n\t"
        "setp.ne.u32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::4X [%0], q_desc, k_desc, idesc, "
        "[sfa], [sfb], p;\n\t"
        "}\n"
        :: "r"(score.addr),
           "r"(q_desc_low),
           "r"(k_desc_low),
           "r"(scale_base),
           "n"(IDESC),
           "n"(DESC_HI),
           "n"(K64_DESC_OFFSET)
        : "memory");
    }
    detail::tcgen05::commit<1>(finished);
}

template <
    typename C,
    typename Score,
    typename Q,
    typename K,
    typename QScale,
    typename KScale>
__device__ __forceinline__ void hao_direct_issue_qk_mxfp4_compact_scales(
    Score &score,
    Q &q,
    K &k,
    QScale &q_scale_smem,
    KScale &k_scale_smem,
    uint32_t scale_base,
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
    st_descriptor<Q, 0> q_desc(q);
    st_descriptor<K, 0> k_desc(k);
    full_tt_fp8e8m0<16> q_sc(scale_base);
    full_tt_fp8e8m0<16> k_sc(scale_base + 16);
    hao_direct_load_mxfp4_scale_ranklocal(q_sc, q_scale_smem);
    hao_direct_load_mxfp4_scale_ranklocal(k_sc, k_scale_smem);
    detail::tcgen05::template st_st<
        T_AB, T_SAB, 0, 1, 32>(
            score.addr,
            q_desc.chunk_descriptor(0),
            k_desc.chunk_descriptor(0),
            q_sc.addr,
            k_sc.addr,
            IDESC0);
    detail::tcgen05::template st_st<
        T_AB, T_SAB, 1, 1, 32>(
            score.addr,
            q_desc.chunk_descriptor(1),
            k_desc.chunk_descriptor(1),
            q_sc.addr,
            k_sc.addr,
            IDESC1);
    detail::tcgen05::commit<1>(finished);
}

template <
    typename C,
    int Stage,
    typename D,
    typename P,
    typename V,
    typename PScale,
    typename VScale>
__device__ __forceinline__ void
hao_direct_issue_pv_nvfp4_n128(
    D &out,
    P &p,
    V &v,
    PScale &p_scale_smem,
    VScale &v_scale_smem,
    uint32_t scale_base,
    uint32_t v_scale_base,
    semaphore &p_tail,
    int p_phase,
    bool accumulate
) {
    constexpr bool PREFETCH_V_SCALE =
        (C::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK &
         (1 << Stage)) != 0;
    constexpr bool LOAD_V_SCALE =
        !PREFETCH_V_SCALE &&
        (!C::HAO_DIRECT_NV_REUSE_V_SCALE || Stage == 0);
    constexpr bool LOAD_V_SCALE_FIRST =
        LOAD_V_SCALE &&
        (C::HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK & 0x1) == 0;
    constexpr bool LOAD_V_SCALE_TAIL =
        LOAD_V_SCALE &&
        (C::HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK & 0x2) == 0;
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e4m3;
    using T_D = typename D::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Dvo, false, 0>();
    constexpr uint32_t DESC_HI =
        hao_direct_fp4_smem_descriptor_high<V>();
    constexpr uint32_t K64_DESC_OFFSET =
        hao_direct_fp4_k64_descriptor_offset<V>();
    if constexpr (
        C::HAO_DIRECT_FUSED_PV_ISSUE &&
        !C::HAO_DIRECT_EARLY_P) {
        const uint32_t v_desc_low =
            hao_direct_fp4_smem_descriptor_low(v);
        const uint32_t p_scale_desc_low =
            hao_direct_scale_smem_descriptor_low(p_scale_smem);
        const uint32_t v_scale_desc_low =
            hao_direct_scale_smem_descriptor_low(v_scale_smem);
        const uint32_t p_tail_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&p_tail));
        constexpr uint32_t SCALE_DESC_HI =
            8u | (1u << (46 - 32));
        asm volatile(
            "{\n\t"
            ".reg .pred p, ready;\n\t"
            ".reg .b32 desc_hi, idesc, sfa, sfb, v_lo, p_lo, "
            "scale_hi, scale_lo, scale_dst;\n\t"
            ".reg .b64 v_desc, scale_desc;\n\t"
            "mov.b32 scale_hi, %10;\n\t"
            "mov.b32 scale_lo, %8;\n\t"
            "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
            "tcgen05.cp.cta_group::1.32x128b.warpx4 "
            "[%3], scale_desc;\n\t"
            "add.u32 scale_lo, %8, 32;\n\t"
            "add.u32 scale_dst, %3, 4;\n\t"
            "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
            "tcgen05.cp.cta_group::1.32x128b.warpx4 "
            "[scale_dst], scale_desc;\n\t"
            "mov.b32 scale_lo, %9;\n\t"
            "add.u32 scale_dst, %3, 8;\n\t"
            "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
            "tcgen05.cp.cta_group::1.32x128b.warpx4 "
            "[scale_dst], scale_desc;\n\t"
            "add.u32 scale_lo, %9, 32;\n\t"
            "add.u32 scale_dst, %3, 12;\n\t"
            "mov.b64 scale_desc, {scale_lo, scale_hi};\n\t"
            "tcgen05.cp.cta_group::1.32x128b.warpx4 "
            "[scale_dst], scale_desc;\n\t"
            "mov.b32 desc_hi, %7;\n\t"
            "mov.b32 idesc, %5;\n\t"
            "mov.b32 sfa, %3;\n\t"
            "add.u32 sfb, %3, 8;\n\t"
            "mov.b64 v_desc, {%2, desc_hi};\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4."
            "block_scale.scale_vec::4X "
            "[%0], [%1], v_desc, idesc, [sfa], [sfb], p;\n\t"
            "HAO_DIRECT_NVFP4_PV_WAIT:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 "
            "ready, [%11], %12, 10000000;\n\t"
            "@ready bra.uni HAO_DIRECT_NVFP4_PV_READY;\n\t"
            "bra.uni HAO_DIRECT_NVFP4_PV_WAIT;\n\t"
            "HAO_DIRECT_NVFP4_PV_READY:\n\t"
            "add.u32 v_lo, %2, %7;\n\t"
            "add.u32 p_lo, %1, 8;\n\t"
            "add.u32 sfa, %3, 4;\n\t"
            "add.u32 sfb, %3, 12;\n\t"
            "mov.b64 v_desc, {v_lo, desc_hi};\n\t"
            "setp.ne.b32 p, 1, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4."
            "block_scale.scale_vec::4X "
            "[%0], [p_lo], v_desc, idesc, [sfa], [sfb], p;\n\t"
            "}\n"
            :: "r"(out.addr),
               "r"(p.addr),
               "r"(v_desc_low),
               "r"(scale_base),
               "n"(IDESC),
               "r"(int(accumulate)),
               "n"(DESC_HI),
               "n"(K64_DESC_OFFSET),
               "r"(p_scale_desc_low),
               "r"(v_scale_desc_low),
               "n"(SCALE_DESC_HI),
               "r"(p_tail_addr),
               "r"(p_phase)
            : "memory");
    } else {
        const uint32_t p_scale_first =
            C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE
                ? scale_base + 32
                : scale_base;
        const uint32_t p_scale_tail =
            C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE
                ? scale_base + 36
                : scale_base + 4;
        if constexpr (!C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
            hao_direct_load_nvfp4_scale_ranklocal<0>(
                p_scale_first, p_scale_smem);
        }
        if constexpr (LOAD_V_SCALE_FIRST) {
            hao_direct_load_nvfp4_scale_ranklocal<0>(
                v_scale_base + 8, v_scale_smem);
        }
        if constexpr (
            LOAD_V_SCALE_TAIL &&
            C::HAO_DIRECT_NV_PRELOAD_V_TAIL_MODE == 1) {
            hao_direct_load_nvfp4_scale_ranklocal<1>(
                v_scale_base + 12, v_scale_smem);
        }
        const uint32_t v_desc_low =
            hao_direct_fp4_smem_descriptor_low(v);
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 desc_hi, idesc, sfa, sfb;\n\t"
            ".reg .b64 v_desc;\n\t"
            "mov.b32 desc_hi, %7;\n\t"
            "mov.b32 idesc, %5;\n\t"
            "mov.b32 sfa, %3;\n\t"
            "mov.b32 sfb, %4;\n\t"
            "mov.b64 v_desc, {%2, desc_hi};\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4."
            "block_scale.scale_vec::4X "
            "[%0], [%1], v_desc, idesc, [sfa], [sfb], p;\n\t"
            "}\n"
            :: "r"(out.addr),
               "r"(p.addr),
               "r"(v_desc_low),
               "r"(p_scale_first),
               "r"(v_scale_base + 8),
               "n"(IDESC),
               "r"(int(accumulate)),
               "n"(DESC_HI));

        if constexpr (
            LOAD_V_SCALE_TAIL &&
            C::HAO_DIRECT_NV_PRELOAD_V_TAIL_MODE == 2) {
            hao_direct_load_nvfp4_scale_ranklocal<1>(
                v_scale_base + 12, v_scale_smem);
        }
        hao_direct_fp4pv_wait(p_tail, p_phase);

        if constexpr (!C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
            hao_direct_load_nvfp4_scale_ranklocal<1>(
                p_scale_tail, p_scale_smem);
        }
        if constexpr (
            LOAD_V_SCALE_TAIL &&
            C::HAO_DIRECT_NV_PRELOAD_V_TAIL_MODE == 0) {
            hao_direct_load_nvfp4_scale_ranklocal<1>(
                v_scale_base + 12, v_scale_smem);
        }
        const uint32_t v_tail_desc_low =
            v_desc_low + K64_DESC_OFFSET;
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 desc_hi, idesc, sfa, sfb;\n\t"
            ".reg .b64 v_desc;\n\t"
            "mov.b32 desc_hi, %6;\n\t"
            "mov.b32 idesc, %5;\n\t"
            "mov.b32 sfa, %3;\n\t"
            "mov.b32 sfb, %4;\n\t"
            "mov.b64 v_desc, {%2, desc_hi};\n\t"
            "setp.ne.b32 p, 1, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
            "scale_vec::4X [%0], [%1], v_desc, idesc, "
            "[sfa], [sfb], p;\n\t"
            "}\n"
            :: "r"(out.addr),
               "r"(p.addr + 8),
               "r"(v_tail_desc_low),
               "r"(p_scale_tail),
               "r"(v_scale_base + 12),
               "n"(IDESC),
               "n"(DESC_HI));
    }
}

template <
    typename C,
    typename D,
    typename P,
    typename V,
    typename VScale>
__device__ __forceinline__ void
hao_direct_issue_pv_sparse_nvfp4_n128(
    D &out,
    P &p,
    V &v,
    VScale &v_scale_smem,
    uint32_t p_scale_base,
    uint32_t v_scale_base,
    semaphore &p_tail,
    int p_phase,
    bool accumulate
) {
    static_assert(C::HAO_DIRECT_SPARSE_PV);
    static_assert(hao_direct_sparse_p_type<P>);
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e4m3;
    using T_D = typename D::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Dvo, false, 0>() |
        (1u << 2);

    // Sparse NVFP4 consumes one block-32 scale word for the full logical
    // K128 V tile. Sparse benchmark inputs place that precompacted page in
    // the first rank-local scale chunk.
    hao_direct_load_nvfp4_scale_ranklocal<0>(
        v_scale_base, v_scale_smem);
    st_descriptor<V, transpose::N> v_desc(v);
    if constexpr (C::HAO_DIRECT_SPARSE_SPLIT_K128) {
        st_descriptor<hao_direct_sparse_p_tile, transpose::N> first_desc(
            *p.payload);
        asm volatile(
            "{\n\t"
            ".reg .pred pred;\n\t"
            "setp.ne.b32 pred, %7, 0;\n\t"
            "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4."
            "block_scale.block16 "
            "[%0], %1, %2, [%3], %4, [%5], [%6], pred;\n\t"
            "}\n"
            :
            : "r"(out.addr),
              "l"(first_desc.chunk_descriptor(0)),
              "l"(v_desc.chunk_descriptor(0)),
              "r"(p.addr),
              "r"(IDESC),
              "r"(p_scale_base),
              "r"(v_scale_base),
              "r"(int(accumulate))
            : "memory");

        hao_direct_fp4pv_wait(p_tail, p_phase);
        st_descriptor<hao_direct_sparse_p_tile, transpose::N> tail_desc(
            *p.tail_payload);
        asm volatile(
            "{\n\t"
            ".reg .pred pred;\n\t"
            "setp.ne.b32 pred, 1, 0;\n\t"
            "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4."
            "block_scale.block16 "
            "[%0], %1, %2, [%3], %4, [%5], [%6], pred;\n\t"
            "}\n"
            :
            : "r"(out.addr),
              "l"(tail_desc.chunk_descriptor(0)),
              "l"(v_desc.chunk_descriptor(0)),
              "r"(p.tail_addr),
              "r"(IDESC),
              "r"(p_scale_base + 4),
              "r"(v_scale_base)
            : "memory");
    } else {
        hao_direct_fp4pv_wait(p_tail, p_phase);
        st_descriptor<hao_direct_sparse_p_tile, transpose::N> p_desc(
            *p.payload);
        asm volatile(
            "{\n\t"
            ".reg .pred pred;\n\t"
            "setp.ne.b32 pred, %7, 0;\n\t"
            "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4."
            "block_scale.block16 "
            "[%0], %1, %2, [%3], %4, [%5], [%6], pred;\n\t"
            "}\n"
            :
            : "r"(out.addr),
              "l"(p_desc.chunk_descriptor(0)),
              "l"(v_desc.chunk_descriptor(0)),
              "r"(p.addr),
              "r"(IDESC),
              "r"(p_scale_base),
              "r"(v_scale_base),
              "r"(int(accumulate))
            : "memory");
    }
}

template <
    typename C,
    typename D,
    typename P,
    typename V,
    typename PScale,
    typename VScale>
__device__ __forceinline__ void hao_direct_issue_pv_mxfp4_first(
    D &out,
    P &p,
    V &v,
    PScale &p_scale_smem,
    VScale &v_scale_smem,
    uint32_t scale_base,
    bool accumulate
) {
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e8m0;
    using T_D = typename D::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Dvo, false, 0>();
    st_descriptor<V, 0> v_desc(v);
    full_tt_fp8e8m0<16> p_sc(
        C::HAO_DIRECT_MX_EARLY_P
            ? scale_base + 32
            : scale_base);
    full_tt_fp8e8m0<16> v_sc(scale_base + 16);
    if constexpr (!C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
        hao_direct_load_mxfp4_scale_ranklocal(
            p_sc, p_scale_smem);
    }
    hao_direct_load_mxfp4_scale_ranklocal(
        v_sc, v_scale_smem);
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::2X [%0], [%1], %2, %3, [%4], [%5], p;}\n"
        :: "r"(out.addr),
           "r"(p.addr),
           "l"(v_desc.chunk_descriptor(0)),
           "r"(IDESC),
           "r"(p_sc.addr),
           "r"(v_sc.addr),
           "r"(int(accumulate)));
}

template <
    typename C,
    typename D,
    typename P,
    typename V,
    typename PScale>
__device__ __forceinline__ void hao_direct_issue_pv_mxfp4_tail(
    D &out,
    P &p,
    V &v,
    PScale &p_scale_smem,
    uint32_t scale_base,
    semaphore &p_tail,
    int p_phase
) {
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e8m0;
    using T_D = typename D::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Dvo, false, 2>();
    st_descriptor<V, 0> v_desc(v);
    full_tt_fp8e8m0<16> p_sc(
        C::HAO_DIRECT_MX_EARLY_P
            ? scale_base + 48
            : scale_base);
    full_tt_fp8e8m0<16> v_sc(scale_base + 16);

    hao_direct_fp4pv_wait(p_tail, p_phase);
    if constexpr (
        C::HAO_DIRECT_MX_EARLY_P &&
        !C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
        hao_direct_load_mxfp4_scale_ranklocal(
            p_sc, p_scale_smem);
    }
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.ne.b32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::2X [%0], [%1], %2, %3, [%4], [%5], p;}\n"
        :: "r"(out.addr),
           "r"(p.addr + 8),
           "l"(v_desc.chunk_descriptor(1)),
           "r"(IDESC),
           "r"(p_sc.addr),
           "r"(v_sc.addr));
}

template <
    typename C,
    typename D,
    typename P,
    typename V,
    typename PScale,
    typename VScale>
__device__ __forceinline__ void hao_direct_issue_pv_mxfp4_n128(
    D &out,
    P &p,
    V &v,
    PScale &p_scale_smem,
    VScale &v_scale_smem,
    uint32_t scale_base,
    semaphore &p_tail,
    int p_phase,
    bool accumulate
) {
    using T_AB = fp4e2m1_2;
    using T_SAB = fp8e8m0;
    using T_D = typename D::T;
    constexpr uint32_t IDESC0 =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Dvo, false, 0>();
    constexpr uint32_t IDESC1 =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, C::Mb, C::Dvo, false, 2>();
    st_descriptor<V, 0> v_desc(v);
    // QK reuses scale_base + [0, 32) between the two stage-local PV calls.
    // Both direct P-scale pages therefore live in retired score quarter 1.
    full_tt_fp8e8m0<16> p_sc_first(
        C::HAO_DIRECT_MX_EARLY_P
            ? scale_base + 32
            : scale_base);
    full_tt_fp8e8m0<16> p_sc_tail(
        C::HAO_DIRECT_MX_EARLY_P
            ? scale_base + 48
            : scale_base);
    full_tt_fp8e8m0<16> v_sc(scale_base + 16);
    if constexpr (!C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
        hao_direct_load_mxfp4_scale_ranklocal(
            p_sc_first, p_scale_smem);
    }
    hao_direct_load_mxfp4_scale_ranklocal(
        v_sc, v_scale_smem);
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::2X [%0], [%1], %2, %3, [%4], [%5], p;}\n"
        :: "r"(out.addr),
           "r"(p.addr),
           "l"(v_desc.chunk_descriptor(0)),
           "r"(IDESC0),
           "r"(p_sc_first.addr),
           "r"(v_sc.addr),
           "r"(int(accumulate)));

    hao_direct_fp4pv_wait(p_tail, p_phase);

    if constexpr (
        C::HAO_DIRECT_MX_EARLY_P &&
        !C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
        hao_direct_load_mxfp4_scale_ranklocal(
            p_sc_tail, p_scale_smem);
    }

    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.ne.b32 p, 1, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::2X [%0], [%1], %2, %3, [%4], [%5], p;}\n"
        :: "r"(out.addr),
           "r"(p.addr + 8),
           "l"(v_desc.chunk_descriptor(1)),
           "r"(IDESC1),
           "r"(p_sc_tail.addr),
           "r"(v_sc.addr));
}

template <int Quarter, int Pair, typename C>
__device__ __forceinline__ float2 hao_direct_exp2_fp8_pair(
    float2 value,
    float row_max,
    float scale_log2
) {
    const float2 mul = {scale_log2, scale_log2};
    const float2 bias = {
        -row_max * scale_log2,
        -row_max * scale_log2};
    return fp4pv_stage2_exp2_pair<
        C::ONLINE_STAGE2_EX2_EMU_PERIOD,
        C::ONLINE_STAGE2_EX2_EMU_MASK,
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

template <int Quarter, int Pair, typename C>
__device__ __forceinline__ float2 hao_direct_exp2_nvfp4_pair(
    float2 value,
    float row_max,
    float scale_log2
) {
    const float2 mul = {scale_log2, scale_log2};
    const float2 bias = {
        -row_max * scale_log2,
        -row_max * scale_log2};
            constexpr uint32_t ex2_emu_mask =
                Quarter == 1
                    ? C::HAO_DIRECT_MX_EX2_Q1_MASK
                    : (Quarter == 2
                           ? C::HAO_DIRECT_MX_EX2_Q2_MASK
                           : 0);
    return fp4pv_stage2_exp2_pair<
        C::ONLINE_STAGE2_EX2_EMU_PERIOD,
        ex2_emu_mask,
        Quarter * 16 + Pair>(
            __ffma2_rn(value, mul, bias));
}

__device__ __forceinline__ float hao_direct_max8_pairs(
    const float2 &v0,
    const float2 &v1,
    const float2 &v2,
    const float2 &v3,
    const float2 &v4,
    const float2 &v5,
    const float2 &v6,
    const float2 &v7
) {
    float value = fmaxf(v0.x, v0.y);
    value = fmaxf(value, fmaxf(v1.x, v1.y));
    value = fmaxf(value, fmaxf(v2.x, v2.y));
    value = fmaxf(value, fmaxf(v3.x, v3.y));
    value = fmaxf(value, fmaxf(v4.x, v4.y));
    value = fmaxf(value, fmaxf(v5.x, v5.y));
    value = fmaxf(value, fmaxf(v6.x, v6.y));
    return fmaxf(value, fmaxf(v7.x, v7.y));
}

__device__ __forceinline__ float hao_direct_sum8_pairs(
    const float2 &v0,
    const float2 &v1,
    const float2 &v2,
    const float2 &v3,
    const float2 &v4,
    const float2 &v5,
    const float2 &v6,
    const float2 &v7
) {
    const float2 s01 = __fadd2_rn(v0, v1);
    const float2 s23 = __fadd2_rn(v2, v3);
    const float2 s45 = __fadd2_rn(v4, v5);
    const float2 s67 = __fadd2_rn(v6, v7);
    const float2 s0123 = __fadd2_rn(s01, s23);
    const float2 s4567 = __fadd2_rn(s45, s67);
    const float2 sum = __fadd2_rn(s0123, s4567);
    return sum.x + sum.y;
}

__device__ __forceinline__ float hao_direct_exp2_native_scalar(
    float value
) {
    float result;
    asm(
        "ex2.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(value));
    return result;
}

__device__ __forceinline__ float hao_direct_fma_sat_scalar(
    float a,
    float b,
    float c
) {
    float result;
    asm(
        "fma.rn.sat.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(a), "f"(b), "f"(c));
    return result;
}

__device__ __forceinline__ int hao_direct_float_to_s32_rn_sat(
    float value
) {
    int result;
    asm(
        "cvt.rni.sat.s32.f32 %0, %1;"
        : "=r"(result)
        : "f"(value));
    return result;
}

__device__ __forceinline__ float
hao_direct_pow2_integer_scalar(int exponent) {
    int biased = exponent + 127;
    asm(
        "max.s32 %0, %0, 0;\n\t"
        "min.s32 %0, %0, 254;"
        : "+r"(biased));
    return __int_as_float(biased << 23);
}

template <typename C>
__device__ __forceinline__ uint8_t
hao_direct_select_mxfp4_e8m0(float log2_value) {
    if constexpr (C::HAO_DIRECT_MX_SCALE_SELECT == 0) {
        return isfinite(log2_value)
            ? fp4pv_log2_to_e8m0_rte_finite(log2_value)
            : uint8_t{0};
    } else if constexpr (C::HAO_DIRECT_MX_SCALE_SELECT == 1) {
        // Rounding a positive value to its nearest power of two changes at
        // 1.5 * 2^e. In log2 space this is one shifted floor operation. Exact
        // midpoint ties round upward here; those inputs are vanishingly rare.
        constexpr float SHIFT =
            1.0f - 0.5849625007211562f;
        const int rounded =
            __float2int_rd(log2_value + SHIFT);
        int exponent;
        asm(
            "{\n\t"
            ".reg .s32 biased;\n\t"
            "add.s32 biased, %1, 127;\n\t"
            "max.s32 biased, biased, 0;\n\t"
            "min.s32 %0, biased, 254;\n\t"
            "}\n"
            : "=r"(exponent)
            : "r"(rounded));
        return static_cast<uint8_t>(exponent);
    } else if constexpr (C::HAO_DIRECT_MX_SCALE_SELECT == 3) {
        // floor(x + SHIFT) == round-to-nearest(x + SHIFT - 0.5).
        // The magic-bias conversion keeps the operation on the FP/INT ALUs
        // instead of introducing an F2I dependency into every P quarter.
        constexpr float FLOOR_OFFSET = -0.08496244f;
        constexpr float MAGIC = 12582912.0f;
        constexpr int MAGIC_BITS = 0x4b400000;
        const float shifted = __fadd_rn(log2_value, FLOOR_OFFSET);
        const int rounded =
            __float_as_int(__fadd_rn(shifted, MAGIC)) - MAGIC_BITS;
        int exponent;
        asm(
            "{\n\t"
            ".reg .s32 biased;\n\t"
            "add.s32 biased, %1, 127;\n\t"
            "max.s32 biased, biased, 0;\n\t"
            "min.s32 %0, biased, 254;\n\t"
            "}\n"
            : "=r"(exponent)
            : "r"(rounded));
        return static_cast<uint8_t>(exponent);
    } else {
        static_assert(C::HAO_DIRECT_MX_SCALE_SELECT == 2);
        const float scale =
            hao_direct_exp2_native_scalar(log2_value);
        uint32_t exponent;
        asm(
            "{\n\t"
            ".reg .pred gt_half, eq_half, odd, bump, can_bump;\n\t"
            ".reg .b32 bits, mantissa, parity, increment;\n\t"
            "mov.b32 bits, %1;\n\t"
            "bfe.u32 %0, bits, 23, 8;\n\t"
            "and.b32 mantissa, bits, 0x7fffff;\n\t"
            "and.b32 parity, %0, 1;\n\t"
            "setp.gt.u32 gt_half, mantissa, 0x400000;\n\t"
            "setp.eq.u32 eq_half, mantissa, 0x400000;\n\t"
            "setp.ne.u32 odd, parity, 0;\n\t"
            "and.pred eq_half, eq_half, odd;\n\t"
            "or.pred bump, gt_half, eq_half;\n\t"
            "setp.lt.u32 can_bump, %0, 254;\n\t"
            "and.pred bump, bump, can_bump;\n\t"
            "selp.u32 increment, 1, 0, bump;\n\t"
            "add.u32 %0, %0, increment;\n\t"
            "min.u32 %0, %0, 254;\n\t"
            "}\n"
            : "=&r"(exponent)
            : "f"(scale));
        return static_cast<uint8_t>(exponent);
    }
}

__device__ __forceinline__ uint32_t
hao_direct_pack_e2m1_pair_from_log_fast(float2 log_value) {
    const float2 affine_scale = {2.0f, 2.0f};
    const float2 affine_bias = {2.37f, 2.37f};
    const float2 transformed =
        __ffma2_rn(log_value, affine_scale, affine_bias);
    uint32_t packed;
    asm(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .s32 x, y, low;\n\t"
        "cvt.rmi.s32.f32 x, %1;\n\t"
        "max.s32 x, x, 0;\n\t"
        "min.s32 x, x, 7;\n\t"
        "setp.ge.f32 p, %3, 0fC0000000;\n\t"
        "selp.s32 low, 1, 0, p;\n\t"
        "max.s32 x, x, low;\n\t"
        "setp.ge.f32 p, %3, 0fBED47FCC;\n\t"
        "selp.s32 low, 2, 0, p;\n\t"
        "max.s32 x, x, low;\n\t"
        "cvt.rmi.s32.f32 y, %2;\n\t"
        "max.s32 y, y, 0;\n\t"
        "min.s32 y, y, 7;\n\t"
        "setp.ge.f32 p, %4, 0fC0000000;\n\t"
        "selp.s32 low, 1, 0, p;\n\t"
        "max.s32 y, y, low;\n\t"
        "setp.ge.f32 p, %4, 0fBED47FCC;\n\t"
        "selp.s32 low, 2, 0, p;\n\t"
        "max.s32 y, y, low;\n\t"
        "shl.b32 y, y, 4;\n\t"
        "or.b32 %0, x, y;\n\t"
        "}\n"
        : "=r"(packed)
        : "f"(transformed.x),
          "f"(transformed.y),
          "f"(log_value.x),
          "f"(log_value.y));
    return packed;
}

__device__ __forceinline__ uint32_t
hao_direct_pack_e2m1_pair_from_log_predicated(float2 log_value) {
    const float2 transformed = __ffma2_rn(
        log_value,
        float2{2.0f, 2.0f},
        float2{2.37f, 2.37f});
    uint32_t packed;
    asm(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .s32 x, y;\n\t"
        "cvt.rmi.s32.f32 x, %1;\n\t"
        "max.s32 x, x, 0;\n\t"
        "min.s32 x, x, 7;\n\t"
        "setp.ge.f32 p, %3, 0fC0000000;\n\t"
        "@p max.s32 x, x, 1;\n\t"
        "setp.ge.f32 p, %3, 0fBED47FCC;\n\t"
        "@p max.s32 x, x, 2;\n\t"
        "cvt.rmi.s32.f32 y, %2;\n\t"
        "max.s32 y, y, 0;\n\t"
        "min.s32 y, y, 7;\n\t"
        "setp.ge.f32 p, %4, 0fC0000000;\n\t"
        "@p max.s32 y, y, 1;\n\t"
        "setp.ge.f32 p, %4, 0fBED47FCC;\n\t"
        "@p max.s32 y, y, 2;\n\t"
        "shl.b32 y, y, 4;\n\t"
        "or.b32 %0, x, y;\n\t"
        "}\n"
        : "=r"(packed)
        : "f"(transformed.x),
          "f"(transformed.y),
          "f"(log_value.x),
          "f"(log_value.y));
    return packed;
}

__device__ __forceinline__ uint32_t
hao_direct_pack_e2m1_pair_from_log_lut16(float2 log_value) {
    // Uniform log2 bins approximate the seven positive E2M1 rounding
    // thresholds. Each nibble in the two immediates is one output code.
    const float2 transformed = __ffma2_rn(
        log_value,
        float2{3.0601740634f, 3.0601740634f},
        float2{7.1789968086f, 7.1789968086f});
    uint32_t packed;
    asm(
        "{\n\t"
        ".reg .pred high;\n\t"
        ".reg .s32 x, y;\n\t"
        ".reg .b32 lut, shift;\n\t"
        "cvt.rmi.s32.f32 x, %1;\n\t"
        "max.s32 x, x, 0;\n\t"
        "min.s32 x, x, 15;\n\t"
        "setp.ge.s32 high, x, 8;\n\t"
        "selp.b32 lut, 0x77655433, 0x22111110, high;\n\t"
        "and.b32 shift, x, 7;\n\t"
        "shl.b32 shift, shift, 2;\n\t"
        "shr.u32 lut, lut, shift;\n\t"
        "and.b32 x, lut, 0xf;\n\t"
        "cvt.rmi.s32.f32 y, %2;\n\t"
        "max.s32 y, y, 0;\n\t"
        "min.s32 y, y, 15;\n\t"
        "setp.ge.s32 high, y, 8;\n\t"
        "selp.b32 lut, 0x77655433, 0x22111110, high;\n\t"
        "and.b32 shift, y, 7;\n\t"
        "shl.b32 shift, shift, 2;\n\t"
        "shr.u32 lut, lut, shift;\n\t"
        "and.b32 y, lut, 0xf;\n\t"
        "shl.b32 y, y, 4;\n\t"
        "or.b32 %0, x, y;\n\t"
        "}\n"
        : "=r"(packed)
        : "f"(transformed.x), "f"(transformed.y));
    return packed;
}

template <int LowCorrectionCount>
__device__ __forceinline__ uint32_t
hao_direct_pack_e2m1_pair_from_log_bits(float2 log_value) {
    static_assert(
        LowCorrectionCount >= 0 && LowCorrectionCount <= 2);
    constexpr float CODE_SCALE =
        LowCorrectionCount <= 1
            ? 1.826841f
            : 1.891177f;
    // Adding 1.5 * 2^23 makes FP32 rounding place the rounded integer in
    // the low mantissa bits. The +2 bias is exactly representable at this
    // magnitude, so the packed FMA directly produces the candidate code.
    const float2 encoded = __ffma2_rn(
        log_value,
        float2{CODE_SCALE, CODE_SCALE},
        float2{12582914.0f, 12582914.0f});
    const uint32_t log_x = __float_as_uint(log_value.x);
    const uint32_t log_y = __float_as_uint(log_value.y);
    const uint32_t encoded_x = __float_as_uint(encoded.x);
    const uint32_t encoded_y = __float_as_uint(encoded.y);
    uint32_t packed;
    if constexpr (LowCorrectionCount == 0) {
        asm(
            "{\n\t"
            ".reg .s32 x, y;\n\t"
            "sub.s32 x, %1, 0x4b400000;\n\t"
            "max.s32 x, x, 0;\n\t"
            "min.s32 x, x, 7;\n\t"
            "sub.s32 y, %2, 0x4b400000;\n\t"
            "max.s32 y, y, 0;\n\t"
            "min.s32 y, y, 7;\n\t"
            "shl.b32 y, y, 4;\n\t"
            "or.b32 %0, x, y;\n\t"
            "}\n"
            : "=r"(packed)
            : "r"(encoded_x), "r"(encoded_y));
    } else if constexpr (LowCorrectionCount == 1) {
        asm(
            "{\n\t"
            ".reg .pred p;\n\t"
            ".reg .s32 x, y;\n\t"
            "sub.s32 x, %3, 0x4b400000;\n\t"
            "max.s32 x, x, 0;\n\t"
            "min.s32 x, x, 7;\n\t"
            "setp.le.u32 p, %1, 0xc0000000;\n\t"
            "@p max.s32 x, x, 1;\n\t"
            "sub.s32 y, %4, 0x4b400000;\n\t"
            "max.s32 y, y, 0;\n\t"
            "min.s32 y, y, 7;\n\t"
            "setp.le.u32 p, %2, 0xc0000000;\n\t"
            "@p max.s32 y, y, 1;\n\t"
            "shl.b32 y, y, 4;\n\t"
            "or.b32 %0, x, y;\n\t"
            "}\n"
            : "=r"(packed)
            : "r"(log_x),
              "r"(log_y),
              "r"(encoded_x),
              "r"(encoded_y));
    } else {
        asm(
            "{\n\t"
            ".reg .pred p;\n\t"
            ".reg .s32 x, y;\n\t"
            "sub.s32 x, %3, 0x4b400000;\n\t"
            "max.s32 x, x, 0;\n\t"
            "min.s32 x, x, 7;\n\t"
            "setp.le.u32 p, %1, 0xc0000000;\n\t"
            "@p max.s32 x, x, 1;\n\t"
            "setp.le.u32 p, %1, 0xbed47fcc;\n\t"
            "@p max.s32 x, x, 2;\n\t"
            "sub.s32 y, %4, 0x4b400000;\n\t"
            "max.s32 y, y, 0;\n\t"
            "min.s32 y, y, 7;\n\t"
            "setp.le.u32 p, %2, 0xc0000000;\n\t"
            "@p max.s32 y, y, 1;\n\t"
            "setp.le.u32 p, %2, 0xbed47fcc;\n\t"
            "@p max.s32 y, y, 2;\n\t"
            "shl.b32 y, y, 4;\n\t"
            "or.b32 %0, x, y;\n\t"
            "}\n"
            : "=r"(packed)
            : "r"(log_x),
              "r"(log_y),
              "r"(encoded_x),
              "r"(encoded_y));
    }
    return packed;
}

__device__ __forceinline__ uint32_t
hao_direct_pack_e2m1_pair_from_magic_encoded(float2 encoded) {
    const uint32_t encoded_x = __float_as_uint(encoded.x);
    const uint32_t encoded_y = __float_as_uint(encoded.y);
    uint32_t packed;
    asm(
        "{\n\t"
        ".reg .s32 x, y;\n\t"
        "sub.s32 x, %1, 0x4b400000;\n\t"
        "max.s32 x, x, 0;\n\t"
        "min.s32 x, x, 7;\n\t"
        "sub.s32 y, %2, 0x4b400000;\n\t"
        "max.s32 y, y, 0;\n\t"
        "min.s32 y, y, 7;\n\t"
        "shl.b32 y, y, 4;\n\t"
        "or.b32 %0, x, y;\n\t"
        "}\n"
        : "=r"(packed)
        : "r"(encoded_x), "r"(encoded_y));
    return packed;
}

__device__ __forceinline__ uint32_t
hao_direct_pack_e2m1_pair_from_log_bits_simd(float2 log_value) {
    constexpr float CODE_SCALE = 1.826841f;
    const float2 encoded = __ffma2_rn(
        log_value,
        float2{CODE_SCALE, CODE_SCALE},
        float2{12582914.0f, 12582914.0f});

    // The 0x4b400000 magic-bias base has zero low halfwords. Packing those
    // halfwords therefore yields the signed candidate codes directly, with
    // no integer subtraction. The high halfwords of the source scores are
    // sufficient for the exact -2.0 correction.
    uint32_t candidates = __byte_perm(
        __float_as_uint(encoded.x),
        __float_as_uint(encoded.y),
        0x5410);
    const uint32_t log_prefixes = __byte_perm(
        __float_as_uint(log_value.x),
        __float_as_uint(log_value.y),
        0x7632);
    const uint32_t code_one = __vsetleu2(
        log_prefixes, 0xc000c000u);
    candidates = __vmaxs2(candidates, code_one);
    candidates = __vmins2(candidates, 0x00070007u);

    const uint32_t high_code = candidates >> 12;
    return (candidates & 0x0fu) | (high_code & 0xf0u);
}

__device__ __forceinline__ int
hao_direct_qsum_x2_from_pair_byte(uint32_t packed_byte) {
    int sum = 0;
    asm(
        "{\n\t"
        ".reg .b32 weights;\n\t"
        "prmt.b32 weights, 0x03020100, 0x0c080604, %1;\n\t"
        "dp4a.s32.s32 %0, weights, 0x01010101, %0;\n\t"
        "}\n"
        : "+r"(sum)
        : "r"(packed_byte));
    return sum;
}

__device__ __forceinline__ void
hao_direct_accumulate_qsum_x2_word(
    int &sum,
    uint32_t packed_word
) {
    asm(
        "{\n\t"
        ".reg .b32 weights, selectors;\n\t"
        "prmt.b32 weights, 0x03020100, 0x0c080604, %1;\n\t"
        "dp4a.s32.s32 %0, weights, 0x01010101, %0;\n\t"
        "shr.u32 selectors, %1, 16;\n\t"
        "prmt.b32 weights, 0x03020100, 0x0c080604, selectors;\n\t"
        "dp4a.s32.s32 %0, weights, 0x01010101, %0;\n\t"
        "}\n"
        : "+r"(sum)
        : "r"(packed_word));
}

__device__ __forceinline__ void
hao_direct_decode_qsum_weight_bytes(
    uint32_t packed_word,
    uint32_t &weights_lo,
    uint32_t &weights_hi
) {
    asm(
        "{\n\t"
        ".reg .b32 selectors;\n\t"
        "prmt.b32 %0, 0x03020100, 0x0c080604, %2;\n\t"
        "shr.u32 selectors, %2, 16;\n\t"
        "prmt.b32 %1, 0x03020100, 0x0c080604, selectors;\n\t"
        "}\n"
        : "=r"(weights_lo), "=r"(weights_hi)
        : "r"(packed_word));
}

__device__ __forceinline__ int
hao_direct_qsum_x2_packed_bytes(const uint32_t (&words)[4]) {
    uint32_t w0_lo, w0_hi;
    uint32_t w1_lo, w1_hi;
    uint32_t w2_lo, w2_hi;
    uint32_t w3_lo, w3_hi;
    hao_direct_decode_qsum_weight_bytes(words[0], w0_lo, w0_hi);
    hao_direct_decode_qsum_weight_bytes(words[1], w1_lo, w1_hi);
    hao_direct_decode_qsum_weight_bytes(words[2], w2_lo, w2_hi);
    hao_direct_decode_qsum_weight_bytes(words[3], w3_lo, w3_hi);

    // Each byte sums to at most 96, so ordinary integer addition cannot
    // carry between the four independent byte lanes.
    const uint32_t partial0 = w0_lo + w0_hi + w1_lo;
    const uint32_t partial1 = w1_hi + w2_lo + w2_hi;
    const uint32_t partial2 = w3_lo + w3_hi;
    const uint32_t packed_sum = partial0 + partial1 + partial2;
    int qsum_x2;
    asm(
        "dp4a.s32.s32 %0, %1, 0x01010101, 0;"
        : "=r"(qsum_x2)
        : "r"(packed_sum));
    return qsum_x2;
}

template <int SkippedWord>
__device__ __forceinline__ int
hao_direct_qsum_x2_packed_bytes3(const uint32_t (&words)[4]) {
    static_assert(SkippedWord >= 0 && SkippedWord < 4);
    constexpr int W0 = SkippedWord == 0 ? 1 : 0;
    constexpr int W1 = SkippedWord <= 1 ? 2 : 1;
    constexpr int W2 = SkippedWord <= 2 ? 3 : 2;
    uint32_t w0_lo, w0_hi;
    uint32_t w1_lo, w1_hi;
    uint32_t w2_lo, w2_hi;
    hao_direct_decode_qsum_weight_bytes(words[W0], w0_lo, w0_hi);
    hao_direct_decode_qsum_weight_bytes(words[W1], w1_lo, w1_hi);
    hao_direct_decode_qsum_weight_bytes(words[W2], w2_lo, w2_hi);
    const uint32_t packed_sum =
        (w0_lo + w0_hi) +
        (w1_lo + w1_hi) +
        (w2_lo + w2_hi);
    int qsum_x2;
    asm(
        "dp4a.s32.s32 %0, %1, 0x01010101, 0;"
        : "=r"(qsum_x2)
        : "r"(packed_sum));
    return qsum_x2;
}

__device__ __forceinline__ int
hao_direct_qsum_x2_packed_bytes_pairwise(
    const uint32_t (&words)[4]
) {
    uint32_t packed_sum0;
    {
        uint32_t w0_lo, w0_hi;
        uint32_t w1_lo, w1_hi;
        hao_direct_decode_qsum_weight_bytes(words[0], w0_lo, w0_hi);
        hao_direct_decode_qsum_weight_bytes(words[1], w1_lo, w1_hi);
        packed_sum0 = (w0_lo + w0_hi) + (w1_lo + w1_hi);
    }
    uint32_t packed_sum1;
    {
        uint32_t w2_lo, w2_hi;
        uint32_t w3_lo, w3_hi;
        hao_direct_decode_qsum_weight_bytes(words[2], w2_lo, w2_hi);
        hao_direct_decode_qsum_weight_bytes(words[3], w3_lo, w3_hi);
        packed_sum1 = (w2_lo + w2_hi) + (w3_lo + w3_hi);
    }
    int qsum_x2 = 0;
    asm(
        "dp4a.s32.s32 %0, %1, 0x01010101, %0;\n\t"
        "dp4a.s32.s32 %0, %2, 0x01010101, %0;"
        : "+r"(qsum_x2)
        : "r"(packed_sum0), "r"(packed_sum1));
    return qsum_x2;
}

__device__ __forceinline__ bool
hao_direct_word_contains_e2m1_six(uint32_t packed_word) {
    const uint32_t low_three_bits =
        packed_word &
        (packed_word >> 1) &
        (packed_word >> 2) &
        0x11111111u;
    return low_three_bits != 0;
}

template <int LocalQuarter, int PQuarter>
__device__ __forceinline__ void
hao_direct_load_nv_denom_snapshot_quarter(
    hao_direct_nv_denom_half_snapshot &snapshot,
    uint32_t p_row_base
) {
    asm volatile(
        "{tcgen05.ld.sync.aligned.32x32b.x4.b32 "
        "{%0,%1,%2,%3}, [%4];}"
        : "=r"(snapshot.word[LocalQuarter][0]),
          "=r"(snapshot.word[LocalQuarter][1]),
          "=r"(snapshot.word[LocalQuarter][2]),
          "=r"(snapshot.word[LocalQuarter][3])
        : "r"(p_row_base + PQuarter * 4));
}

template <int Quarter>
__device__ __forceinline__ void
hao_direct_store_nv_first_payload_quarter_smem(
    hao_direct_nv_first_payload_smem &dst,
    int reader_warp,
    const uint32_t (&words)[4]
) {
    static_assert(Quarter == 0 || Quarter == 1);
    const int row = reader_warp * 32 + warp::laneid();
    const uint32_t address = __cvta_generic_to_shared(
        &dst.word[row][Quarter * 4]);
    asm volatile(
        "st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(address),
           "r"(words[0]),
           "r"(words[1]),
           "r"(words[2]),
           "r"(words[3]));
}

template <int Quarter>
__device__ __forceinline__ void
hao_direct_store_mx_denom_payload_quarter_smem(
    hao_direct_mx_denom_payload_smem &dst,
    int reader_warp,
    const uint32_t (&words)[4]
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    const int row = reader_warp * 32 + warp::laneid();
    const uint32_t address = __cvta_generic_to_shared(
        &dst.word[row][Quarter * 4]);
    asm volatile(
        "st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(address),
           "r"(words[0]),
           "r"(words[1]),
           "r"(words[2]),
           "r"(words[3]));
}

__device__ __forceinline__ void
hao_direct_load_mx_denom_payload_smem(
    hao_direct_nv_denom_half_snapshot &first,
    hao_direct_nv_denom_half_snapshot &tail,
    const hao_direct_mx_denom_payload_smem &src,
    int row_group
) {
    const int row = row_group * 32 + warp::laneid();
    const uint32_t address = __cvta_generic_to_shared(
        &src.word[row][0]);
    asm volatile(
        "ld.shared.v4.b32 {%0,%1,%2,%3}, [%16];\n\t"
        "ld.shared.v4.b32 {%4,%5,%6,%7}, [%16+16];\n\t"
        "ld.shared.v4.b32 {%8,%9,%10,%11}, [%16+32];\n\t"
        "ld.shared.v4.b32 {%12,%13,%14,%15}, [%16+48];"
        : "=r"(first.word[0][0]),
          "=r"(first.word[0][1]),
          "=r"(first.word[0][2]),
          "=r"(first.word[0][3]),
          "=r"(first.word[1][0]),
          "=r"(first.word[1][1]),
          "=r"(first.word[1][2]),
          "=r"(first.word[1][3]),
          "=r"(tail.word[0][0]),
          "=r"(tail.word[0][1]),
          "=r"(tail.word[0][2]),
          "=r"(tail.word[0][3]),
          "=r"(tail.word[1][0]),
          "=r"(tail.word[1][1]),
          "=r"(tail.word[1][2]),
          "=r"(tail.word[1][3])
        : "r"(address));
}

__device__ __forceinline__ void
hao_direct_load_nv_first_payload_smem(
    hao_direct_nv_denom_half_snapshot &snapshot,
    const hao_direct_nv_first_payload_smem &src,
    int row_group
) {
    const int row = row_group * 32 + warp::laneid();
    const uint32_t address = __cvta_generic_to_shared(
        &src.word[row][0]);
    asm volatile(
        "ld.shared.v4.b32 {%0,%1,%2,%3}, [%8];\n\t"
        "ld.shared.v4.b32 {%4,%5,%6,%7}, [%8+16];"
        : "=r"(snapshot.word[0][0]),
          "=r"(snapshot.word[0][1]),
          "=r"(snapshot.word[0][2]),
          "=r"(snapshot.word[0][3]),
          "=r"(snapshot.word[1][0]),
          "=r"(snapshot.word[1][1]),
          "=r"(snapshot.word[1][2]),
          "=r"(snapshot.word[1][3])
        : "r"(address));
}

__device__ __forceinline__ void
hao_direct_load_nv_first_scale(
    hao_direct_nv_denom_half_snapshot &snapshot,
    uint32_t score_base,
    int row_group
) {
    asm volatile(
        "{tcgen05.ld.sync.aligned.32x32b.x1.b32 "
        "{%0}, [%1];}"
        : "=r"(snapshot.scale_half)
        : "r"(score_base + 32 + row_group));
}

template <int Half, typename P>
__device__ __forceinline__ void
hao_direct_load_nv_denom_snapshot(
    hao_direct_nv_denom_half_snapshot &snapshot,
    const P &p,
    uint32_t score_base,
    int row_group
) {
    static_assert(Half == 0 || Half == 1);
    const uint32_t p_row_base =
        p.addr + (static_cast<uint32_t>(row_group * 32) << 16);
    hao_direct_load_nv_denom_snapshot_quarter<0, Half * 2>(
        snapshot, p_row_base);
    hao_direct_load_nv_denom_snapshot_quarter<1, Half * 2 + 1>(
        snapshot, p_row_base);
    asm volatile(
        "{tcgen05.ld.sync.aligned.32x32b.x1.b32 "
        "{%0}, [%1];}"
        : "=r"(snapshot.scale_half)
        : "r"(
              score_base +
              (Half == 0 ? 32 : 36) +
              row_group));
}

template <int Half, typename C, typename P>
__device__ __forceinline__ void
hao_direct_load_mx_denom_snapshot(
    hao_direct_nv_denom_half_snapshot &snapshot,
    const P &p,
    uint32_t score_base,
    int row_group
) {
    static_assert(Half == 0 || Half == 1);
    const uint32_t p_row_base =
        p.addr + (static_cast<uint32_t>(row_group * 32) << 16);
    if constexpr (C::HAO_DIRECT_MX_DENOM_LOAD_WIDTH == 4) {
        hao_direct_load_nv_denom_snapshot_quarter<0, Half * 2>(
            snapshot, p_row_base);
        hao_direct_load_nv_denom_snapshot_quarter<1, Half * 2 + 1>(
            snapshot, p_row_base);
    } else {
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];}"
            : "=r"(snapshot.word[0][0]),
              "=r"(snapshot.word[0][1]),
              "=r"(snapshot.word[0][2]),
              "=r"(snapshot.word[0][3]),
              "=r"(snapshot.word[1][0]),
              "=r"(snapshot.word[1][1]),
              "=r"(snapshot.word[1][2]),
              "=r"(snapshot.word[1][3])
            : "r"(p_row_base + Half * 8));
    }
    asm volatile(
        "{tcgen05.ld.sync.aligned.32x32b.x1.b32 "
        "{%0}, [%1];}"
        : "=r"(snapshot.scale_half)
        : "r"(
              score_base +
              (Half == 0 ? 32 : 48) +
              row_group));
}

template <typename C, typename P, typename PScale>
__device__ __forceinline__ void
hao_direct_load_mx_denom_snapshot_pair(
    hao_direct_nv_denom_half_snapshot &first,
    hao_direct_nv_denom_half_snapshot &tail,
    const P &p,
    PScale &p_scale_smem,
    uint32_t score_base,
    int row_group
) {
    if constexpr (C::HAO_DIRECT_MX_DENOM_LOAD_WIDTH == 16) {
        const uint32_t p_row_base =
            p.addr + (static_cast<uint32_t>(row_group * 32) << 16);
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
            "[%16];}"
            : "=r"(first.word[0][0]),
              "=r"(first.word[0][1]),
              "=r"(first.word[0][2]),
              "=r"(first.word[0][3]),
              "=r"(first.word[1][0]),
              "=r"(first.word[1][1]),
              "=r"(first.word[1][2]),
              "=r"(first.word[1][3]),
              "=r"(tail.word[0][0]),
              "=r"(tail.word[0][1]),
              "=r"(tail.word[0][2]),
              "=r"(tail.word[0][3]),
              "=r"(tail.word[1][0]),
              "=r"(tail.word[1][1]),
              "=r"(tail.word[1][2]),
              "=r"(tail.word[1][3])
            : "r"(p_row_base));
        if constexpr (C::HAO_DIRECT_MX_DENOM_SCALE_SMEM) {
            const int row = row_group * 32 + warp::laneid();
            const uint32_t packed_scale =
                fp4pv_load_mxfp4_scale_word_swizzled(
                    p_scale_smem, row);
            first.scale_half = packed_scale;
            tail.scale_half = packed_scale;
        } else {
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x1.b32 "
                "{%0}, [%2];}\n\t"
                "{tcgen05.ld.sync.aligned.32x32b.x1.b32 "
                "{%1}, [%3];}"
                : "=r"(first.scale_half), "=r"(tail.scale_half)
                : "r"(score_base + 32 + row_group),
                  "r"(score_base + 48 + row_group));
        }
    } else {
        hao_direct_load_mx_denom_snapshot<0, C>(
            first, p, score_base, row_group);
        hao_direct_load_mx_denom_snapshot<1, C>(
            tail, p, score_base, row_group);
    }
}

template <int FirstQuarter, int DenomWords, bool KnownMax>
__device__ __forceinline__ float
hao_direct_sum_nv_denom_snapshot(
    const hao_direct_nv_denom_half_snapshot &snapshot
) {
    static_assert(FirstQuarter == 0 || FirstQuarter == 2);
    static_assert(
        DenomWords == 1 || DenomWords == 2 || DenomWords == 4);
    float sum = 0.0f;
    #pragma unroll
    for (int quarter = 0; quarter < 2; ++quarter) {
        int qsum_x2 = 0;
        bool sampled_max = false;
        if constexpr (DenomWords == 4) {
            #pragma unroll
            for (int word = 0; word < 4; ++word) {
                hao_direct_accumulate_qsum_x2_word(
                    qsum_x2,
                    snapshot.word[quarter][word]);
                if constexpr (KnownMax) {
                    sampled_max =
                        sampled_max ||
                        hao_direct_word_contains_e2m1_six(
                            snapshot.word[quarter][word]);
                }
            }
        } else if constexpr (DenomWords == 2) {
            constexpr int first_word_base = FirstQuarter & 1;
            const int first_word = first_word_base ^ (quarter & 1);
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2,
                snapshot.word[quarter][first_word]);
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2,
                snapshot.word[quarter][first_word + 2]);
            if constexpr (KnownMax) {
                sampled_max =
                    hao_direct_word_contains_e2m1_six(
                        snapshot.word[quarter][first_word]) ||
                    hao_direct_word_contains_e2m1_six(
                        snapshot.word[quarter][first_word + 2]);
            }
        } else {
            constexpr int first_word = FirstQuarter & 3;
            const int word = (first_word + quarter) & 3;
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2,
                snapshot.word[quarter][word]);
            if constexpr (KnownMax) {
                sampled_max =
                    hao_direct_word_contains_e2m1_six(
                        snapshot.word[quarter][word]);
            }
        }
        const uint32_t scale_bits =
            (snapshot.scale_half >> (quarter * 16)) &
            0xffu;
        const float scale = __uint_as_float(
            (scale_bits << 20) + (120u << 23));
        float represented_sum;
        if constexpr (KnownMax) {
            constexpr float sampled_values =
                8.0f * static_cast<float>(DenomWords);
            const float residual_scale =
                sampled_max
                    ? 31.0f / (sampled_values - 1.0f)
                    : 31.0f / sampled_values;
            const float sampled_residual =
                static_cast<float>(qsum_x2) -
                (sampled_max ? 12.0f : 0.0f);
            represented_sum =
                0.5f *
                __fmaf_rn(
                    residual_scale,
                    sampled_residual,
                    12.0f);
        } else {
            constexpr float extrapolation =
                4.0f / static_cast<float>(DenomWords);
            represented_sum =
                (0.5f * extrapolation) *
                static_cast<float>(qsum_x2);
        }
        sum = __fmaf_rn(
            scale,
            represented_sum,
            sum);
    }
    return sum;
}

template <int Quarter, typename C>
__device__ __forceinline__ float
hao_direct_sum_mx_denom_quarter(
    const hao_direct_nv_denom_half_snapshot &snapshot
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    constexpr int LocalQuarter = Quarter & 1;
    int qsum_x2 = 0;
    if constexpr (C::HAO_DIRECT_MX_DENOM_WORDS == 4) {
        if constexpr (C::HAO_DIRECT_MX_DENOM_DECODE_MODE == 2) {
            qsum_x2 = hao_direct_qsum_x2_packed_bytes_pairwise(
                snapshot.word[LocalQuarter]);
        } else if constexpr (C::HAO_DIRECT_MX_DENOM_DECODE_MODE == 1) {
            qsum_x2 = hao_direct_qsum_x2_packed_bytes(
                snapshot.word[LocalQuarter]);
        } else if constexpr (C::HAO_DIRECT_MX_DENOM_SUM_LANES == 1) {
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, snapshot.word[LocalQuarter][0]);
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, snapshot.word[LocalQuarter][1]);
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, snapshot.word[LocalQuarter][2]);
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, snapshot.word[LocalQuarter][3]);
        } else if constexpr (C::HAO_DIRECT_MX_DENOM_SUM_LANES == 2) {
            int qsum1_x2 = 0;
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, snapshot.word[LocalQuarter][0]);
            hao_direct_accumulate_qsum_x2_word(
                qsum1_x2, snapshot.word[LocalQuarter][1]);
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, snapshot.word[LocalQuarter][2]);
            hao_direct_accumulate_qsum_x2_word(
                qsum1_x2, snapshot.word[LocalQuarter][3]);
            qsum_x2 += qsum1_x2;
        } else {
            int qsum1_x2 = 0;
            int qsum2_x2 = 0;
            int qsum3_x2 = 0;
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, snapshot.word[LocalQuarter][0]);
            hao_direct_accumulate_qsum_x2_word(
                qsum1_x2, snapshot.word[LocalQuarter][1]);
            hao_direct_accumulate_qsum_x2_word(
                qsum2_x2, snapshot.word[LocalQuarter][2]);
            hao_direct_accumulate_qsum_x2_word(
                qsum3_x2, snapshot.word[LocalQuarter][3]);
            qsum_x2 =
                (qsum_x2 + qsum1_x2) +
                (qsum2_x2 + qsum3_x2);
        }
    } else if constexpr (C::HAO_DIRECT_MX_DENOM_WORDS == 3) {
        constexpr int SkippedWord = Quarter & 3;
        if constexpr (C::HAO_DIRECT_MX_DENOM_DECODE_MODE == 1) {
            qsum_x2 = hao_direct_qsum_x2_packed_bytes3<SkippedWord>(
                snapshot.word[LocalQuarter]);
        } else {
            if constexpr (SkippedWord != 0) {
                hao_direct_accumulate_qsum_x2_word(
                    qsum_x2, snapshot.word[LocalQuarter][0]);
            }
            if constexpr (SkippedWord != 1) {
                hao_direct_accumulate_qsum_x2_word(
                    qsum_x2, snapshot.word[LocalQuarter][1]);
            }
            if constexpr (SkippedWord != 2) {
                hao_direct_accumulate_qsum_x2_word(
                    qsum_x2, snapshot.word[LocalQuarter][2]);
            }
            if constexpr (SkippedWord != 3) {
                hao_direct_accumulate_qsum_x2_word(
                    qsum_x2, snapshot.word[LocalQuarter][3]);
            }
        }
    } else if constexpr (C::HAO_DIRECT_MX_DENOM_WORDS == 2) {
        constexpr int FirstWord = Quarter & 1;
        hao_direct_accumulate_qsum_x2_word(
            qsum_x2, snapshot.word[LocalQuarter][FirstWord]);
        hao_direct_accumulate_qsum_x2_word(
            qsum_x2, snapshot.word[LocalQuarter][FirstWord + 2]);
    } else {
        constexpr int Word = Quarter & 3;
        hao_direct_accumulate_qsum_x2_word(
            qsum_x2, snapshot.word[LocalQuarter][Word]);
    }

    const uint32_t e8m0 =
        (snapshot.scale_half >> (Quarter * 8)) & 0xffu;
    float group_scale = __uint_as_float(e8m0 << 23);
    float represented_sum;
    if constexpr (C::HAO_DIRECT_MX_DENOM_MAGIC_I2F) {
        constexpr uint32_t MAGIC_BIAS_BITS = 0x4b000000u;
        const float biased_sum = __uint_as_float(
            MAGIC_BIAS_BITS + static_cast<uint32_t>(qsum_x2));
        const float negative_scaled_bias = __uint_as_float(
            0x80000000u | ((e8m0 + 23u) << 23));
        represented_sum = __fmaf_rn(
            biased_sum, group_scale, negative_scaled_bias);
        group_scale = 1.0f;
    } else {
        represented_sum = static_cast<float>(qsum_x2);
    }
    if constexpr (!C::HAO_DIRECT_MX_DENOM_FACTOR_HOIST) {
        group_scale *= 1.0f / 6.0f;
        represented_sum *=
            2.0f / static_cast<float>(C::HAO_DIRECT_MX_DENOM_WORDS);
    }
    return group_scale * represented_sum;
}

template <int FirstQuarter, typename C>
__device__ __forceinline__ float
hao_direct_sum_mx_denom_snapshot(
    const hao_direct_nv_denom_half_snapshot &snapshot
) {
    static_assert(FirstQuarter == 0 || FirstQuarter == 2);
    return
        hao_direct_sum_mx_denom_quarter<FirstQuarter, C>(snapshot) +
        hao_direct_sum_mx_denom_quarter<FirstQuarter + 1, C>(snapshot);
}

template <typename C>
__device__ __forceinline__ void
hao_direct_accumulate_mx_denom_snapshot_pair(
    float &sum,
    const hao_direct_nv_denom_half_snapshot &first,
    const hao_direct_nv_denom_half_snapshot &tail
) {
    if constexpr (C::HAO_DIRECT_MX_DENOM_INT_SCALE_FUSION) {
        const int q0 = hao_direct_qsum_x2_packed_bytes(first.word[0]);
        const int q1 = hao_direct_qsum_x2_packed_bytes(first.word[1]);
        const int q2 = hao_direct_qsum_x2_packed_bytes(tail.word[0]);
        const int q3 = hao_direct_qsum_x2_packed_bytes(tail.word[1]);
        const uint32_t e0 = first.scale_half & 0xffu;
        const uint32_t e1 = (first.scale_half >> 8) & 0xffu;
        const uint32_t e2 = (tail.scale_half >> 16) & 0xffu;
        const uint32_t e3 = (tail.scale_half >> 24) & 0xffu;
        const uint32_t n0 = q0 == 0 ? 255u : e0;
        const uint32_t n1 = q1 == 0 ? 255u : e1;
        const uint32_t n2 = q2 == 0 ? 255u : e2;
        const uint32_t n3 = q3 == 0 ? 255u : e3;
        const uint32_t emin = min(min(n0, n1), min(n2, n3));
        if (emin != 255u) {
            const uint32_t x0 = q0 == 0 ? emin : e0;
            const uint32_t x1 = q1 == 0 ? emin : e1;
            const uint32_t x2 = q2 == 0 ? emin : e2;
            const uint32_t x3 = q3 == 0 ? emin : e3;
            const uint32_t emax = max(max(x0, x1), max(x2, x3));
            if (emax - emin <= 20u) {
                const uint32_t aligned0 =
                    static_cast<uint32_t>(q0) << (x0 - emin);
                const uint32_t aligned1 =
                    static_cast<uint32_t>(q1) << (x1 - emin);
                const uint32_t aligned2 =
                    static_cast<uint32_t>(q2) << (x2 - emin);
                const uint32_t aligned3 =
                    static_cast<uint32_t>(q3) << (x3 - emin);
                const uint32_t aligned_sum =
                    (aligned0 + aligned1) + (aligned2 + aligned3);
                const float common_scale =
                    __uint_as_float(emin << 23);
                sum = __fmaf_rn(
                    common_scale,
                    static_cast<float>(aligned_sum),
                    sum);
            } else {
                sum = __fmaf_rn(
                    __uint_as_float(e0 << 23),
                    static_cast<float>(q0),
                    sum);
                sum = __fmaf_rn(
                    __uint_as_float(e1 << 23),
                    static_cast<float>(q1),
                    sum);
                sum = __fmaf_rn(
                    __uint_as_float(e2 << 23),
                    static_cast<float>(q2),
                    sum);
                sum = __fmaf_rn(
                    __uint_as_float(e3 << 23),
                    static_cast<float>(q3),
                    sum);
            }
        }
    } else {
        sum += hao_direct_sum_mx_denom_snapshot<0, C>(first);
        sum += hao_direct_sum_mx_denom_snapshot<2, C>(tail);
    }
}

template <typename C>
__device__ __forceinline__ float
hao_direct_finalize_mx_denom(float sum) {
    if constexpr (
        C::HAO_DIRECT_MX_DENOM_FACTOR_HOIST &&
        !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
        return sum *
            (1.0f /
             (3.0f * static_cast<float>(
                 C::HAO_DIRECT_MX_DENOM_WORDS)));
    }
    return sum;
}

__device__ __forceinline__ void
hao_direct_packed_float_to_e2m1_relu(
    float f0,
    float f1,
    float f2,
    float f3,
    float f4,
    float f5,
    float f6,
    float f7,
    uint32_t &out
) {
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.relu.satfinite.e2m1x2.f32 "
        "byte0, %2, %1;\n"
        "cvt.rn.relu.satfinite.e2m1x2.f32 "
        "byte1, %4, %3;\n"
        "cvt.rn.relu.satfinite.e2m1x2.f32 "
        "byte2, %6, %5;\n"
        "cvt.rn.relu.satfinite.e2m1x2.f32 "
        "byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(out)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3),
          "f"(f4), "f"(f5), "f"(f6), "f"(f7));
}

struct hao_direct_mx_log_word_result {
    uint32_t word;
    int pair_owned_qsum_x2;
};

template <int PairBase, typename C>
__device__ __forceinline__ hao_direct_mx_log_word_result
hao_direct_pack_mxfp4_log_word_predicated(
    score_quarter_regs &scores,
    float2 score_scale,
    float2 quant_bias
) {
    static_assert(
        PairBase == 0 || PairBase == 4 ||
        PairBase == 8 || PairBase == 12);
    uint32_t bytes[4];
    fp4pv_stage2_static_for<0, 4>([&](auto local_pair_tag) {
        constexpr int LocalPair =
            decltype(local_pair_tag)::value;
        constexpr int Pair = PairBase + LocalPair;
        const float2 log_value = __ffma2_rn(
            (&scores.v0)[Pair],
            score_scale,
            quant_bias);
        bytes[LocalPair] =
            hao_direct_pack_e2m1_pair_from_log_predicated(
                log_value);
    });
    const uint32_t word =
        bytes[0] |
        (bytes[1] << 8) |
        (bytes[2] << 16) |
        (bytes[3] << 24);
    int pair_owned_qsum_x2 = 0;
    if constexpr (
        C::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE == 4 &&
        !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
        const int sum0 =
            hao_direct_qsum_x2_from_pair_byte(bytes[0]);
        const int sum1 =
            hao_direct_qsum_x2_from_pair_byte(bytes[1]);
        const int sum2 =
            hao_direct_qsum_x2_from_pair_byte(bytes[2]);
        const int sum3 =
            hao_direct_qsum_x2_from_pair_byte(bytes[3]);
        pair_owned_qsum_x2 =
            (sum0 + sum1) + (sum2 + sum3);
    }
    return {word, pair_owned_qsum_x2};
}

template <typename C>
__device__ __forceinline__ void
hao_direct_pack_mxfp4_log_codes(
    score_quarter_regs &scores,
    float2 score_scale,
    float2 quant_bias,
    uint32_t (&words)[4]
) {
    words[0] = 0;
    words[1] = 0;
    words[2] = 0;
    words[3] = 0;
    fp4pv_stage2_static_for<0, 16>([&](auto pair_tag) {
        constexpr int Pair = decltype(pair_tag)::value;
        constexpr int Word = Pair / 4;
        constexpr int Shift = (Pair % 4) * 8;
        const float2 log_value = __ffma2_rn(
            (&scores.v0)[Pair],
            score_scale,
            quant_bias);
        uint32_t packed_byte;
        if constexpr (
            C::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE == 1) {
            const fp4pv_ex2_range_f32x2 range =
                fp4pv_ex2_alu_range_f32x2(log_value);
            packed_byte =
                fp4pv_pack_e2m1_pair_from_log(range);
        } else {
            static_assert(
                C::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE == 2);
            packed_byte =
                hao_direct_pack_e2m1_pair_from_log_fast(
                    log_value);
        }
        words[Word] |= packed_byte << Shift;
    });
}

__device__ __forceinline__ float hao_direct_quantize_nvfp4_group(
    float2 &v0,
    float2 &v1,
    float2 &v2,
    float2 &v3,
    float2 &v4,
    float2 &v5,
    float2 &v6,
    float2 &v7,
    uint32_t &word0,
    uint32_t &word1,
    float &scale
) {
    const float sum = hao_direct_sum8_pairs(
        v0, v1, v2, v3, v4, v5, v6, v7);
    const float group_max = hao_direct_max8_pairs(
        v0, v1, v2, v3, v4, v5, v6, v7);
    scale = group_max * (1.0f / 6.0f);
    const float inv_scale =
        group_max > 0.0f ? 6.0f / group_max : 0.0f;
    const float2 factor = {inv_scale, inv_scale};
    v0 = __fmul2_rn(v0, factor);
    v1 = __fmul2_rn(v1, factor);
    v2 = __fmul2_rn(v2, factor);
    v3 = __fmul2_rn(v3, factor);
    v4 = __fmul2_rn(v4, factor);
    v5 = __fmul2_rn(v5, factor);
    v6 = __fmul2_rn(v6, factor);
    v7 = __fmul2_rn(v7, factor);
    fp4pv_packed_float_to_e2m1(
        v0.x, v0.y, v1.x, v1.y,
        v2.x, v2.y, v3.x, v3.y,
        word0);
    fp4pv_packed_float_to_e2m1(
        v4.x, v4.y, v5.x, v5.y,
        v6.x, v6.y, v7.x, v7.y,
        word1);
    return sum;
}

template <
    int Quarter,
    int PairBase,
    typename C,
    bool ForceAffine = false>
__device__ __forceinline__ float
hao_direct_quantize_nvfp4_group_log2(
    float2 *values,
    float row_max,
    float scale_log2,
    float shared_group_max,
    float prepared_quant_bias,
    float prepared_scale,
    uint32_t &word0,
    uint32_t &word1,
    float &scale
) {
    static_assert(PairBase == 0 || PairBase == 8);
    constexpr float LOG2_FP4_MAX = 2.584962500721156f;
    constexpr bool FOLD_PWL_NORMALIZATION =
        C::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX &&
        C::HAO_DIRECT_NV_PWL_EXP2_MODE != 0;
    const float2 score_scale = {scale_log2, scale_log2};
    const float2 row_bias = {
        -row_max * scale_log2,
        -row_max * scale_log2};
    float group_max;
    if constexpr (C::HAO_DIRECT_NV_QUARTER_SCALE) {
        group_max = shared_group_max;
    } else if constexpr (FOLD_PWL_NORMALIZATION) {
        group_max = hao_direct_max8_pairs(
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7]);
    } else {
        fp4pv_stage2_static_for<0, 8>([&](auto pair_tag) {
            constexpr int Pair = decltype(pair_tag)::value;
            values[Pair] =
                __ffma2_rn(values[Pair], score_scale, row_bias);
        });
        group_max = hao_direct_max8_pairs(
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7]);
    }
    float quant_bias_scalar;
    if constexpr (C::HAO_DIRECT_NV_QUARTER_SCALE) {
        quant_bias_scalar = prepared_quant_bias;
        scale = prepared_scale;
    } else {
        const float normalized_group_max =
            FOLD_PWL_NORMALIZATION
                ? (group_max - row_max) * scale_log2
                : group_max;
        const float quant_log2_scale =
            fmaxf(normalized_group_max, -100.0f) -
            LOG2_FP4_MAX;
        quant_bias_scalar =
            FOLD_PWL_NORMALIZATION
                ? -row_max * scale_log2 - quant_log2_scale
                : -quant_log2_scale;
        scale = hao_direct_exp2_native_scalar(
            quant_log2_scale +
            C::HAO_DIRECT_NV_P_GLOBAL_LOG2);
    }
    const float2 quant_bias = {
        quant_bias_scalar, quant_bias_scalar};

    constexpr float CUBIC_A =
        TK_HAO_DIRECT_FP4PV_NV_CUBIC_A;
    constexpr float CUBIC_B =
        TK_HAO_DIRECT_FP4PV_NV_CUBIC_B;
    constexpr float CUBIC_C =
        TK_HAO_DIRECT_FP4PV_NV_CUBIC_C;
    constexpr float CUBIC_D =
        TK_HAO_DIRECT_FP4PV_NV_CUBIC_D;
    const float score_scale2 = scale_log2 * scale_log2;
    const float quant_bias2 =
        quant_bias_scalar * quant_bias_scalar;
    const float cubic_raw_a_scalar =
        CUBIC_A * score_scale2 * scale_log2;
    const float cubic_raw_b_scalar =
        score_scale2 * __fmaf_rn(
            3.0f * CUBIC_A,
            quant_bias_scalar,
            CUBIC_B);
    const float cubic_raw_c_scalar =
        scale_log2 * __fmaf_rn(
            3.0f * CUBIC_A,
            quant_bias2,
            __fmaf_rn(
                2.0f * CUBIC_B,
                quant_bias_scalar,
                CUBIC_C));
    const float cubic_raw_d_scalar = __fmaf_rn(
        CUBIC_A * quant_bias_scalar,
        quant_bias2,
        __fmaf_rn(
            CUBIC_B,
            quant_bias2,
            __fmaf_rn(
                CUBIC_C,
                quant_bias_scalar,
                CUBIC_D)));
    const float2 cubic_raw_a = {
        cubic_raw_a_scalar, cubic_raw_a_scalar};
    const float2 cubic_raw_b = {
        cubic_raw_b_scalar, cubic_raw_b_scalar};
    const float2 cubic_raw_c = {
        cubic_raw_c_scalar, cubic_raw_c_scalar};
    const float2 cubic_raw_d = {
        cubic_raw_d_scalar, cubic_raw_d_scalar};
    constexpr float REFIT_CUBIC_A =
        TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_A;
    constexpr float REFIT_CUBIC_B =
        TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_B;
    constexpr float REFIT_CUBIC_C =
        TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_C;
    constexpr float REFIT_CUBIC_D =
        TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_D;
    const float refit_cubic_raw_a_scalar =
        REFIT_CUBIC_A * score_scale2 * scale_log2;
    const float refit_cubic_raw_b_scalar =
        score_scale2 * __fmaf_rn(
            3.0f * REFIT_CUBIC_A,
            quant_bias_scalar,
            REFIT_CUBIC_B);
    const float refit_cubic_raw_c_scalar =
        scale_log2 * __fmaf_rn(
            3.0f * REFIT_CUBIC_A,
            quant_bias2,
            __fmaf_rn(
                2.0f * REFIT_CUBIC_B,
                quant_bias_scalar,
                REFIT_CUBIC_C));
    const float refit_cubic_raw_d_scalar = __fmaf_rn(
        REFIT_CUBIC_A * quant_bias_scalar,
        quant_bias2,
        __fmaf_rn(
            REFIT_CUBIC_B,
            quant_bias2,
            __fmaf_rn(
                REFIT_CUBIC_C,
                quant_bias_scalar,
                REFIT_CUBIC_D)));
    const float2 refit_cubic_raw_a = {
        refit_cubic_raw_a_scalar, refit_cubic_raw_a_scalar};
    const float2 refit_cubic_raw_b = {
        refit_cubic_raw_b_scalar, refit_cubic_raw_b_scalar};
    const float2 refit_cubic_raw_c = {
        refit_cubic_raw_c_scalar, refit_cubic_raw_c_scalar};
    const float2 refit_cubic_raw_d = {
        refit_cubic_raw_d_scalar, refit_cubic_raw_d_scalar};
    constexpr float AFFINE_A =
        TK_HAO_DIRECT_FP4PV_NV_AFFINE_A;
    constexpr float AFFINE_B =
        TK_HAO_DIRECT_FP4PV_NV_AFFINE_B;
    const float2 affine_raw_scale = {
        AFFINE_A * scale_log2,
        AFFINE_A * scale_log2};
    const float2 affine_raw_bias = {
        AFFINE_A * quant_bias_scalar + AFFINE_B,
        AFFINE_A * quant_bias_scalar + AFFINE_B};
    constexpr float HINGE_M =
        TK_HAO_DIRECT_FP4PV_NV_HINGE_M;
    constexpr float HINGE_B =
        TK_HAO_DIRECT_FP4PV_NV_HINGE_B;
    constexpr float HINGE_D =
        TK_HAO_DIRECT_FP4PV_NV_HINGE_D;
    constexpr float HINGE_T =
        TK_HAO_DIRECT_FP4PV_NV_HINGE_T;
    constexpr float HINGE_LOW_M = HINGE_M - HINGE_D;
    constexpr float HINGE_LOW_B =
        HINGE_B + HINGE_D * HINGE_T;
    constexpr float HINGE_HIGH_M = HINGE_M + HINGE_D;
    constexpr float HINGE_HIGH_B =
        HINGE_B - HINGE_D * HINGE_T;
    const float2 hinge_low_raw_scale = {
        HINGE_LOW_M * scale_log2,
        HINGE_LOW_M * scale_log2};
    const float2 hinge_low_raw_bias = {
        HINGE_LOW_M * quant_bias_scalar + HINGE_LOW_B,
        HINGE_LOW_M * quant_bias_scalar + HINGE_LOW_B};
    const float2 hinge_high_raw_scale = {
        HINGE_HIGH_M * scale_log2,
        HINGE_HIGH_M * scale_log2};
    const float2 hinge_high_raw_bias = {
        HINGE_HIGH_M * quant_bias_scalar + HINGE_HIGH_B,
        HINGE_HIGH_M * quant_bias_scalar + HINGE_HIGH_B};

    fp4pv_stage2_static_for<0, 8>([&](auto pair_tag) {
        constexpr int LocalPair = decltype(pair_tag)::value;
        constexpr int Pair = PairBase + LocalPair;
        constexpr int sample_offset = Quarter & 3;
        constexpr int extra_offset = (sample_offset + 2) & 3;
        constexpr int local_word_pair = Pair & 3;
        constexpr bool four_native =
            Pair == sample_offset ||
            Pair == sample_offset + 8 ||
            Pair == 4 ||
            Pair == 12;
        constexpr bool six_native =
            four_native ||
            Pair == extra_offset ||
            Pair == extra_offset + 8;
        constexpr bool eight_native =
            local_word_pair == sample_offset ||
            local_word_pair == ((sample_offset + 2) & 3);
        constexpr bool two_native =
            Pair == sample_offset ||
            Pair == sample_offset + 8;
        constexpr bool use_six_native =
            C::HAO_DIRECT_NV_NATIVE_PAIRS == 6 &&
            ((C::HAO_DIRECT_NV_NATIVE6_QUARTER_MASK >>
              Quarter) &
             1u);
        constexpr bool native_sample =
            (C::HAO_DIRECT_NV_FAST_APPROX_MODE == 2 &&
             Quarter == 0) ||
            ((C::HAO_DIRECT_NV_PWL_EXP2_MODE == 2 ||
              C::HAO_DIRECT_NV_PWL_EXP2_MODE == 5 ||
              C::HAO_DIRECT_NV_PWL_EXP2_MODE == 6 ||
              C::HAO_DIRECT_NV_PWL_EXP2_MODE == 7 ||
              C::HAO_DIRECT_NV_PWL_EXP2_MODE == 8 ||
              C::HAO_DIRECT_NV_PWL_EXP2_MODE == 9) &&
             (C::HAO_DIRECT_NV_PWL_EXP2_MODE == 9
                  ? (C::HAO_DIRECT_NV_NATIVE_PAIRS == 8
                         ? eight_native
                         : (use_six_native
                                ? six_native
                                : four_native))
                  : four_native)) ||
            (C::HAO_DIRECT_NV_PWL_EXP2_MODE == 3 &&
             eight_native) ||
            (C::HAO_DIRECT_NV_PWL_EXP2_MODE == 4 &&
             two_native);
        if constexpr (
            C::HAO_DIRECT_NV_PWL_EXP2_MODE == 0 ||
            native_sample) {
            float2 log_value;
            if constexpr (FOLD_PWL_NORMALIZATION) {
                log_value = __ffma2_rn(
                    values[LocalPair],
                    score_scale,
                    quant_bias);
            } else {
                log_value =
                    __fadd2_rn(values[LocalPair], quant_bias);
            }
            constexpr uint32_t ex2_emu_mask =
                Quarter == 1
                    ? C::HAO_DIRECT_MX_EX2_Q1_MASK
                    : (Quarter == 2
                           ? C::HAO_DIRECT_MX_EX2_Q2_MASK
                           : 0);
            values[LocalPair] = fp4pv_stage2_exp2_pair<
                16,
                ex2_emu_mask,
                Quarter * 16 + Pair>(log_value);
        } else {
            float2 approximation;
            if constexpr (
                ForceAffine ||
                ((C::HAO_DIRECT_NV_PWL_EXP2_MODE == 5 ||
                  C::HAO_DIRECT_NV_PWL_EXP2_MODE == 6 ||
                  C::HAO_DIRECT_NV_PWL_EXP2_MODE == 7 ||
                  C::HAO_DIRECT_NV_PWL_EXP2_MODE == 8 ||
                  C::HAO_DIRECT_NV_PWL_EXP2_MODE == 9) &&
                 Quarter == 0)) {
                if constexpr (FOLD_PWL_NORMALIZATION) {
                    if constexpr (
                        C::HAO_DIRECT_NV_FAST_APPROX_MODE != 1 &&
                        C::HAO_DIRECT_NV_FAST_APPROX_MODE != 3 &&
                        C::HAO_DIRECT_NV_FAST_APPROX_MODE != 4) {
                        static_assert(
                            C::HAO_DIRECT_NV_FAST_APPROX_MODE == 0 ||
                            C::HAO_DIRECT_NV_FAST_APPROX_MODE == 2);
                        approximation = __ffma2_rn(
                            values[LocalPair],
                            affine_raw_scale,
                            affine_raw_bias);
                    } else if constexpr (
                        C::HAO_DIRECT_NV_FAST_APPROX_MODE == 4) {
                        const float2 low_line = __ffma2_rn(
                            values[LocalPair],
                            hinge_low_raw_scale,
                            hinge_low_raw_bias);
                        const float2 high_line = __ffma2_rn(
                            values[LocalPair],
                            hinge_high_raw_scale,
                            hinge_high_raw_bias);
                        approximation = {
                            fmaxf(low_line.x, high_line.x),
                            fmaxf(low_line.y, high_line.y)};
                    } else {
                        static_assert(
                            C::HAO_DIRECT_NV_FAST_APPROX_MODE == 1 ||
                            C::HAO_DIRECT_NV_FAST_APPROX_MODE == 3);
                        approximation = __ffma2_rn(
                            values[LocalPair],
                            refit_cubic_raw_a,
                            refit_cubic_raw_b);
                        approximation = __ffma2_rn(
                            approximation,
                            values[LocalPair],
                            refit_cubic_raw_c);
                        approximation = __ffma2_rn(
                            approximation,
                            values[LocalPair],
                            refit_cubic_raw_d);
                    }
                } else {
                    const float2 log_value =
                        __fadd2_rn(values[LocalPair], quant_bias);
                    if constexpr (
                        C::HAO_DIRECT_NV_FAST_APPROX_MODE != 1 &&
                        C::HAO_DIRECT_NV_FAST_APPROX_MODE != 3 &&
                        C::HAO_DIRECT_NV_FAST_APPROX_MODE != 4) {
                        static_assert(
                            C::HAO_DIRECT_NV_FAST_APPROX_MODE == 0 ||
                            C::HAO_DIRECT_NV_FAST_APPROX_MODE == 2);
                        approximation = __ffma2_rn(
                            log_value,
                            float2{AFFINE_A, AFFINE_A},
                            float2{AFFINE_B, AFFINE_B});
                    } else if constexpr (
                        C::HAO_DIRECT_NV_FAST_APPROX_MODE == 4) {
                        const float2 low_line = __ffma2_rn(
                            log_value,
                            float2{HINGE_LOW_M, HINGE_LOW_M},
                            float2{HINGE_LOW_B, HINGE_LOW_B});
                        const float2 high_line = __ffma2_rn(
                            log_value,
                            float2{HINGE_HIGH_M, HINGE_HIGH_M},
                            float2{HINGE_HIGH_B, HINGE_HIGH_B});
                        approximation = {
                            fmaxf(low_line.x, high_line.x),
                            fmaxf(low_line.y, high_line.y)};
                    } else {
                        static_assert(
                            C::HAO_DIRECT_NV_FAST_APPROX_MODE == 1 ||
                            C::HAO_DIRECT_NV_FAST_APPROX_MODE == 3);
                        approximation = __ffma2_rn(
                            log_value,
                            float2{REFIT_CUBIC_A, REFIT_CUBIC_A},
                            float2{REFIT_CUBIC_B, REFIT_CUBIC_B});
                        approximation = __ffma2_rn(
                            approximation,
                            log_value,
                            float2{REFIT_CUBIC_C, REFIT_CUBIC_C});
                        approximation = __ffma2_rn(
                            approximation,
                            log_value,
                            float2{REFIT_CUBIC_D, REFIT_CUBIC_D});
                    }
                }
            } else if constexpr (
                C::HAO_DIRECT_NV_FAST_APPROX_MODE == 4 &&
                FOLD_PWL_NORMALIZATION) {
                const float2 low_line = __ffma2_rn(
                    values[LocalPair],
                    hinge_low_raw_scale,
                    hinge_low_raw_bias);
                const float2 high_line = __ffma2_rn(
                    values[LocalPair],
                    hinge_high_raw_scale,
                    hinge_high_raw_bias);
                approximation = {
                    fmaxf(low_line.x, high_line.x),
                    fmaxf(low_line.y, high_line.y)};
            } else if constexpr (
                C::HAO_DIRECT_NV_FAST_APPROX_MODE == 4) {
                const float2 log_value =
                    __fadd2_rn(values[LocalPair], quant_bias);
                const float2 low_line = __ffma2_rn(
                    log_value,
                    float2{HINGE_LOW_M, HINGE_LOW_M},
                    float2{HINGE_LOW_B, HINGE_LOW_B});
                const float2 high_line = __ffma2_rn(
                    log_value,
                    float2{HINGE_HIGH_M, HINGE_HIGH_M},
                    float2{HINGE_HIGH_B, HINGE_HIGH_B});
                approximation = {
                    fmaxf(low_line.x, high_line.x),
                    fmaxf(low_line.y, high_line.y)};
            } else if constexpr (FOLD_PWL_NORMALIZATION) {
                approximation = __ffma2_rn(
                    values[LocalPair],
                    cubic_raw_a,
                    cubic_raw_b);
                approximation = __ffma2_rn(
                    approximation,
                    values[LocalPair],
                    cubic_raw_c);
                approximation = __ffma2_rn(
                    approximation,
                    values[LocalPair],
                    cubic_raw_d);
            } else {
                const float2 log_value =
                    __fadd2_rn(values[LocalPair], quant_bias);
                approximation = __ffma2_rn(
                    log_value,
                    float2{CUBIC_A, CUBIC_A},
                    float2{CUBIC_B, CUBIC_B});
                approximation = __ffma2_rn(
                    approximation,
                    log_value,
                    float2{CUBIC_C, CUBIC_C});
                approximation = __ffma2_rn(
                    approximation,
                    log_value,
                    float2{CUBIC_D, CUBIC_D});
            }
            if constexpr (C::HAO_DIRECT_NV_RELU_PACK) {
                values[LocalPair] = approximation;
            } else {
                values[LocalPair] = {
                    fmaxf(approximation.x, 0.0f),
                    fmaxf(approximation.y, 0.0f)};
            }
        }
    });
    constexpr bool USE_QUANTIZED_DENOM =
        C::HAO_DIRECT_NV_QUANTIZED_DENOM ||
        (C::HAO_DIRECT_NV_HYBRID_DENOM &&
         (ForceAffine || Quarter == 0));
    constexpr bool USE_SAMPLED_DENOM =
        C::HAO_DIRECT_NV_SAMPLED_DENOM;
    float scaled_sum = 16.0f;
    if constexpr (
        !C::HAO_DIRECT_FAKE_DENOM_CEILING &&
        !USE_QUANTIZED_DENOM &&
        !USE_SAMPLED_DENOM) {
        scaled_sum = hao_direct_sum8_pairs(
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7]);
    } else if constexpr (
        !C::HAO_DIRECT_FAKE_DENOM_CEILING &&
        USE_SAMPLED_DENOM) {
        constexpr int SAMPLE_PAIR = Quarter & 3;
        if constexpr (C::HAO_DIRECT_NV_NATIVE_PAIRS == 8) {
            constexpr int EXTRA_PAIR = (SAMPLE_PAIR + 2) & 3;
            const float2 sampled_sum = __fadd2_rn(
                __fadd2_rn(
                    values[SAMPLE_PAIR],
                    values[EXTRA_PAIR]),
                __fadd2_rn(
                    values[SAMPLE_PAIR + 4],
                    values[EXTRA_PAIR + 4]));
            scaled_sum =
                2.0f * (sampled_sum.x + sampled_sum.y);
        } else if constexpr (
            C::HAO_DIRECT_NV_NATIVE_PAIRS == 6 &&
            ((C::HAO_DIRECT_NV_NATIVE6_QUARTER_MASK >>
              Quarter) &
             1u)) {
            constexpr int EXTRA_PAIR = (SAMPLE_PAIR + 2) & 3;
            const float2 sampled_sum = __fadd2_rn(
                __fadd2_rn(
                    values[SAMPLE_PAIR],
                    values[EXTRA_PAIR]),
                values[4]);
            scaled_sum =
                (8.0f / 3.0f) *
                (sampled_sum.x + sampled_sum.y);
        } else {
            const float2 sampled_sum = __fadd2_rn(
                values[SAMPLE_PAIR], values[4]);
            scaled_sum =
                4.0f * (sampled_sum.x + sampled_sum.y);
        }
    }
    if constexpr (C::HAO_DIRECT_NV_RELU_PACK) {
        hao_direct_packed_float_to_e2m1_relu(
            values[0].x, values[0].y,
            values[1].x, values[1].y,
            values[2].x, values[2].y,
            values[3].x, values[3].y,
            word0);
        hao_direct_packed_float_to_e2m1_relu(
            values[4].x, values[4].y,
            values[5].x, values[5].y,
            values[6].x, values[6].y,
            values[7].x, values[7].y,
            word1);
    } else {
        fp4pv_packed_float_to_e2m1(
            values[0].x, values[0].y,
            values[1].x, values[1].y,
            values[2].x, values[2].y,
            values[3].x, values[3].y,
            word0);
        fp4pv_packed_float_to_e2m1(
            values[4].x, values[4].y,
            values[5].x, values[5].y,
            values[6].x, values[6].y,
            values[7].x, values[7].y,
            word1);
    }
    if constexpr (C::HAO_DIRECT_FAKE_DENOM_CEILING) {
        return 16.0f;
    } else if constexpr (USE_QUANTIZED_DENOM) {
        return 0.0f;
    } else {
        return C::HAO_DIRECT_NV_SCALE_SATFINITE
            ? fmaxf(scaled_sum, 6.0f)
            : scaled_sum;
    }
}

template <typename C>
__device__ __forceinline__ float2
hao_direct_nv_sampled_denom_value(float2 value) {
    if constexpr (
        C::HAO_DIRECT_NV_SCALE_SATFINITE &&
        C::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX) {
        constexpr float E2M1_MAX = 6.0f;
        return {
            fminf(value.x, E2M1_MAX),
            fminf(value.y, E2M1_MAX)};
    } else {
        return value;
    }
}

struct hao_direct_noop_tail_hook {
    __device__ __forceinline__ void operator()() const {}
};

__device__ __forceinline__ float2
hao_direct_bf16_cubic_pair(
    float2 raw_score,
    float2 score_scale,
    float2 quant_bias,
    float a,
    float b,
    float c,
    float d
) {
    const float2 log_value =
        __ffma2_rn(raw_score, score_scale, quant_bias);
    const __nv_bfloat162 x =
        __floats2bfloat162_rn(log_value.x, log_value.y);
    __nv_bfloat162 approximation = __hfma2(
        x,
        __float2bfloat162_rn(a),
        __float2bfloat162_rn(b));
    approximation = __hfma2(
        approximation,
        x,
        __float2bfloat162_rn(c));
    approximation = __hfma2(
        approximation,
        x,
        __float2bfloat162_rn(d));
    const uint32_t packed =
        *reinterpret_cast<const uint32_t *>(&approximation);
    return {
        __uint_as_float(packed << 16),
        __uint_as_float(packed & 0xffff0000u)};
}

template <
    int Quarter,
    typename C,
    bool ForceAffine = false
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD != 0
    ,
    typename Hook = hao_direct_noop_tail_hook>
#else
    >
#endif
__device__ __forceinline__ float
hao_direct_quantize_nvfp4_quarter_log2_scheduled(
    score_quarter_regs &scores,
    float scale_log2,
    float quant_bias_scalar,
    uint32_t (&words)[4]
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD != 0
    ,
    Hook hook = {}
#endif
) {
    static_assert(
        C::HAO_DIRECT_NV_QUARTER_SCHEDULE_MODE >= 1 &&
        C::HAO_DIRECT_NV_QUARTER_SCHEDULE_MODE <= 2);
    static_assert(C::HAO_DIRECT_NV_PWL_EXP2_MODE == 9);

    const float2 score_scale = {scale_log2, scale_log2};
    const float2 quant_bias = {
        quant_bias_scalar, quant_bias_scalar};
    constexpr float CUBIC_A =
        TK_HAO_DIRECT_FP4PV_NV_CUBIC_A;
    constexpr float CUBIC_B =
        TK_HAO_DIRECT_FP4PV_NV_CUBIC_B;
    constexpr float CUBIC_C =
        TK_HAO_DIRECT_FP4PV_NV_CUBIC_C;
    constexpr float CUBIC_D =
        TK_HAO_DIRECT_FP4PV_NV_CUBIC_D;
    const float score_scale2 = scale_log2 * scale_log2;
    const float quant_bias2 =
        quant_bias_scalar * quant_bias_scalar;
    const float cubic_raw_a_scalar =
        CUBIC_A * score_scale2 * scale_log2;
    const float cubic_raw_b_scalar =
        score_scale2 * __fmaf_rn(
            3.0f * CUBIC_A,
            quant_bias_scalar,
            CUBIC_B);
    const float cubic_raw_c_scalar =
        scale_log2 * __fmaf_rn(
            3.0f * CUBIC_A,
            quant_bias2,
            __fmaf_rn(
                2.0f * CUBIC_B,
                quant_bias_scalar,
                CUBIC_C));
    const float cubic_raw_d_scalar = __fmaf_rn(
        CUBIC_A * quant_bias_scalar,
        quant_bias2,
        __fmaf_rn(
            CUBIC_B,
            quant_bias2,
            __fmaf_rn(
                CUBIC_C,
                quant_bias_scalar,
                CUBIC_D)));
    const float2 cubic_raw_a = {
        cubic_raw_a_scalar, cubic_raw_a_scalar};
    const float2 cubic_raw_b = {
        cubic_raw_b_scalar, cubic_raw_b_scalar};
    const float2 cubic_raw_c = {
        cubic_raw_c_scalar, cubic_raw_c_scalar};
    const float2 cubic_raw_d = {
        cubic_raw_d_scalar, cubic_raw_d_scalar};
    constexpr float AFFINE_A =
        TK_HAO_DIRECT_FP4PV_NV_AFFINE_A;
    constexpr float AFFINE_B =
        TK_HAO_DIRECT_FP4PV_NV_AFFINE_B;
    const float2 affine_raw_scale = {
        AFFINE_A * scale_log2,
        AFFINE_A * scale_log2};
    const float2 affine_raw_bias = {
        AFFINE_A * quant_bias_scalar + AFFINE_B,
        AFFINE_A * quant_bias_scalar + AFFINE_B};
    constexpr float HINGE_M =
        TK_HAO_DIRECT_FP4PV_NV_HINGE_M;
    constexpr float HINGE_B =
        TK_HAO_DIRECT_FP4PV_NV_HINGE_B;
    constexpr float HINGE_D =
        TK_HAO_DIRECT_FP4PV_NV_HINGE_D;
    constexpr float HINGE_T =
        TK_HAO_DIRECT_FP4PV_NV_HINGE_T;
    constexpr float HINGE_LOW_M = HINGE_M - HINGE_D;
    constexpr float HINGE_LOW_B =
        HINGE_B + HINGE_D * HINGE_T;
    constexpr float HINGE_HIGH_M = HINGE_M + HINGE_D;
    constexpr float HINGE_HIGH_B =
        HINGE_B - HINGE_D * HINGE_T;
    const float2 hinge_low_raw_scale = {
        HINGE_LOW_M * scale_log2,
        HINGE_LOW_M * scale_log2};
    const float2 hinge_low_raw_bias = {
        HINGE_LOW_M * quant_bias_scalar + HINGE_LOW_B,
        HINGE_LOW_M * quant_bias_scalar + HINGE_LOW_B};
    const float2 hinge_high_raw_scale = {
        HINGE_HIGH_M * scale_log2,
        HINGE_HIGH_M * scale_log2};
    const float2 hinge_high_raw_bias = {
        HINGE_HIGH_M * quant_bias_scalar + HINGE_HIGH_B,
        HINGE_HIGH_M * quant_bias_scalar + HINGE_HIGH_B};
    constexpr float REFIT_CUBIC_A =
        TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_A;
    constexpr float REFIT_CUBIC_B =
        TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_B;
    constexpr float REFIT_CUBIC_C =
        TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_C;
    constexpr float REFIT_CUBIC_D =
        TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_D;
    const float refit_cubic_raw_a_scalar =
        REFIT_CUBIC_A * score_scale2 * scale_log2;
    const float refit_cubic_raw_b_scalar =
        score_scale2 * __fmaf_rn(
            3.0f * REFIT_CUBIC_A,
            quant_bias_scalar,
            REFIT_CUBIC_B);
    const float refit_cubic_raw_c_scalar =
        scale_log2 * __fmaf_rn(
            3.0f * REFIT_CUBIC_A,
            quant_bias2,
            __fmaf_rn(
                2.0f * REFIT_CUBIC_B,
                quant_bias_scalar,
                REFIT_CUBIC_C));
    const float refit_cubic_raw_d_scalar = __fmaf_rn(
        REFIT_CUBIC_A * quant_bias_scalar,
        quant_bias2,
        __fmaf_rn(
            REFIT_CUBIC_B,
            quant_bias2,
            __fmaf_rn(
                REFIT_CUBIC_C,
                quant_bias_scalar,
                REFIT_CUBIC_D)));
    const float2 refit_cubic_raw_a = {
        refit_cubic_raw_a_scalar, refit_cubic_raw_a_scalar};
    const float2 refit_cubic_raw_b = {
        refit_cubic_raw_b_scalar, refit_cubic_raw_b_scalar};
    const float2 refit_cubic_raw_c = {
        refit_cubic_raw_c_scalar, refit_cubic_raw_c_scalar};
    const float2 refit_cubic_raw_d = {
        refit_cubic_raw_d_scalar, refit_cubic_raw_d_scalar};
    auto transform_pair = [&](auto pair_tag) {
        constexpr int Pair = decltype(pair_tag)::value;
        constexpr int sample_offset = Quarter & 3;
        constexpr int extra_offset = (sample_offset + 2) & 3;
        constexpr int local_word_pair = Pair & 3;
        constexpr bool four_pair_sample =
            Pair == sample_offset ||
            Pair == sample_offset + 8 ||
            Pair == 4 ||
            Pair == 12;
        constexpr bool six_pair_sample =
            four_pair_sample ||
            Pair == extra_offset ||
            Pair == extra_offset + 8;
        constexpr bool eight_pair_sample =
            local_word_pair == sample_offset ||
            local_word_pair == extra_offset;
        constexpr bool use_six_native =
            C::HAO_DIRECT_NV_NATIVE_PAIRS == 6 &&
            ((C::HAO_DIRECT_NV_NATIVE6_QUARTER_MASK >>
              Quarter) &
             1u);
        constexpr bool native_sample =
            (C::HAO_DIRECT_NV_FAST_APPROX_MODE == 2 &&
             Quarter == 0) ||
            (C::HAO_DIRECT_NV_NATIVE_PAIRS == 8
                ? eight_pair_sample
                : (use_six_native
                       ? six_pair_sample
                       : four_pair_sample));
        const float2 raw_score = (&scores.v0)[Pair];
        if constexpr (native_sample) {
            float2 log_value = __ffma2_rn(
                raw_score, score_scale, quant_bias);
            (&scores.v0)[Pair] = fp4pv_stage2_exp2_pair<
                16, 0, Quarter * 16 + Pair>(log_value);
        } else if constexpr (ForceAffine || Quarter == 0) {
            if constexpr (
                C::HAO_DIRECT_NV_FAST_APPROX_MODE == 0 ||
                C::HAO_DIRECT_NV_FAST_APPROX_MODE == 2) {
                static_assert(
                    C::HAO_DIRECT_NV_FAST_APPROX_MODE == 0 ||
                    C::HAO_DIRECT_NV_FAST_APPROX_MODE == 2);
                (&scores.v0)[Pair] = __ffma2_rn(
                    raw_score,
                    affine_raw_scale,
                    affine_raw_bias);
            } else if constexpr (
                C::HAO_DIRECT_NV_FAST_APPROX_MODE == 1) {
                static_assert(
                    C::HAO_DIRECT_NV_FAST_APPROX_MODE == 1);
                float2 approximation = __ffma2_rn(
                    raw_score,
                    refit_cubic_raw_a,
                    refit_cubic_raw_b);
                approximation = __ffma2_rn(
                    approximation,
                    raw_score,
                    refit_cubic_raw_c);
                (&scores.v0)[Pair] = __ffma2_rn(
                    approximation,
                    raw_score,
                    refit_cubic_raw_d);
            } else if constexpr (
                C::HAO_DIRECT_NV_FAST_APPROX_MODE == 4) {
                const float2 low_line = __ffma2_rn(
                    raw_score,
                    hinge_low_raw_scale,
                    hinge_low_raw_bias);
                const float2 high_line = __ffma2_rn(
                    raw_score,
                    hinge_high_raw_scale,
                    hinge_high_raw_bias);
                (&scores.v0)[Pair] = {
                    fmaxf(low_line.x, high_line.x),
                    fmaxf(low_line.y, high_line.y)};
            } else {
                static_assert(
                    C::HAO_DIRECT_NV_FAST_APPROX_MODE == 3);
                (&scores.v0)[Pair] = hao_direct_bf16_cubic_pair(
                    raw_score,
                    score_scale,
                    quant_bias,
                    REFIT_CUBIC_A,
                    REFIT_CUBIC_B,
                    REFIT_CUBIC_C,
                    REFIT_CUBIC_D);
            }
        } else {
            if constexpr (
                C::HAO_DIRECT_NV_FAST_APPROX_MODE == 4) {
                const float2 low_line = __ffma2_rn(
                    raw_score,
                    hinge_low_raw_scale,
                    hinge_low_raw_bias);
                const float2 high_line = __ffma2_rn(
                    raw_score,
                    hinge_high_raw_scale,
                    hinge_high_raw_bias);
                (&scores.v0)[Pair] = {
                    fmaxf(low_line.x, high_line.x),
                    fmaxf(low_line.y, high_line.y)};
            } else if constexpr (
                C::HAO_DIRECT_NV_FAST_APPROX_MODE == 3) {
                (&scores.v0)[Pair] = hao_direct_bf16_cubic_pair(
                    raw_score,
                    score_scale,
                    quant_bias,
                    CUBIC_A,
                    CUBIC_B,
                    CUBIC_C,
                    CUBIC_D);
            } else {
                float2 approximation = __ffma2_rn(
                    raw_score, cubic_raw_a, cubic_raw_b);
                approximation = __ffma2_rn(
                    approximation, raw_score, cubic_raw_c);
                (&scores.v0)[Pair] = __ffma2_rn(
                    approximation, raw_score, cubic_raw_d);
            }
        }
    };
    auto pack_word = [&]<int PairBase>(uint32_t &word) {
        const float2 *values = &scores.v0 + PairBase;
        hao_direct_packed_float_to_e2m1_relu(
            values[0].x, values[0].y,
            values[1].x, values[1].y,
            values[2].x, values[2].y,
            values[3].x, values[3].y,
            word);
    };

    constexpr int sample_offset = Quarter & 3;
    transform_pair(std::integral_constant<int, sample_offset>{});
    if constexpr (sample_offset != 0) {
        transform_pair(std::integral_constant<int, 0>{});
    }
    if constexpr (sample_offset != 1) {
        transform_pair(std::integral_constant<int, 1>{});
    }
    if constexpr (sample_offset != 2) {
        transform_pair(std::integral_constant<int, 2>{});
    }
    if constexpr (sample_offset != 3) {
        transform_pair(std::integral_constant<int, 3>{});
    }
    pack_word.template operator()<0>(words[0]);
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD == 1
    if constexpr (Quarter == 2) {
        hook();
    }
#endif

    transform_pair(std::integral_constant<int, 4>{});
    transform_pair(std::integral_constant<int, 5>{});
    transform_pair(std::integral_constant<int, 6>{});
    transform_pair(std::integral_constant<int, 7>{});
    pack_word.template operator()<4>(words[1]);
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD == 2
    if constexpr (Quarter == 2) {
        hook();
    }
#endif

    transform_pair(
        std::integral_constant<int, sample_offset + 8>{});
    if constexpr (sample_offset != 0) {
        transform_pair(std::integral_constant<int, 8>{});
    }
    if constexpr (sample_offset != 1) {
        transform_pair(std::integral_constant<int, 9>{});
    }
    if constexpr (sample_offset != 2) {
        transform_pair(std::integral_constant<int, 10>{});
    }
    if constexpr (sample_offset != 3) {
        transform_pair(std::integral_constant<int, 11>{});
    }
    pack_word.template operator()<8>(words[2]);
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD == 3
    if constexpr (Quarter == 2) {
        hook();
    }
#endif

    transform_pair(std::integral_constant<int, 12>{});
    transform_pair(std::integral_constant<int, 13>{});
    transform_pair(std::integral_constant<int, 14>{});
    transform_pair(std::integral_constant<int, 15>{});
    pack_word.template operator()<12>(words[3]);

    if constexpr (
        C::HAO_DIRECT_FAKE_DENOM_CEILING ||
        C::HAO_DIRECT_NV_DENOM_CORR_WG) {
        return 32.0f;
    } else if constexpr (C::HAO_DIRECT_NV_DENOM_SOURCE == 1) {
        int qsum_x2 = 0;
        bool sampled_max = false;
        if constexpr (C::HAO_DIRECT_NV_DENOM_WORDS == 4) {
            hao_direct_accumulate_qsum_x2_word(qsum_x2, words[0]);
            hao_direct_accumulate_qsum_x2_word(qsum_x2, words[1]);
            hao_direct_accumulate_qsum_x2_word(qsum_x2, words[2]);
            hao_direct_accumulate_qsum_x2_word(qsum_x2, words[3]);
        } else if constexpr (
            C::HAO_DIRECT_NV_DENOM_WORDS == 2) {
            constexpr int first_word = Quarter & 1;
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, words[first_word]);
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, words[first_word + 2]);
            if constexpr (C::HAO_DIRECT_NV_DENOM_KNOWN_MAX) {
                sampled_max =
                    hao_direct_word_contains_e2m1_six(
                        words[first_word]) ||
                    hao_direct_word_contains_e2m1_six(
                        words[first_word + 2]);
            }
        } else {
            constexpr int word = Quarter & 3;
            const uint32_t sampled_word = words[word];
            hao_direct_accumulate_qsum_x2_word(
                qsum_x2, sampled_word);
            if constexpr (C::HAO_DIRECT_NV_DENOM_KNOWN_MAX) {
                sampled_max =
                    hao_direct_word_contains_e2m1_six(sampled_word);
            }
        }
        if constexpr (C::HAO_DIRECT_NV_DENOM_KNOWN_MAX) {
            constexpr float sampled_values =
                8.0f *
                static_cast<float>(C::HAO_DIRECT_NV_DENOM_WORDS);
            const float residual_scale =
                sampled_max
                    ? 31.0f / (sampled_values - 1.0f)
                    : 31.0f / sampled_values;
            const float sampled_residual =
                static_cast<float>(qsum_x2) -
                (sampled_max ? 12.0f : 0.0f);
            return 0.5f *
                   __fmaf_rn(
                       residual_scale,
                       sampled_residual,
                       12.0f);
        }
        constexpr float extrapolation =
            4.0f /
            static_cast<float>(C::HAO_DIRECT_NV_DENOM_WORDS);
        return (0.5f * extrapolation) *
               static_cast<float>(qsum_x2);
    } else if constexpr (
        C::HAO_DIRECT_NV_NATIVE_PAIRS == 6 &&
        ((C::HAO_DIRECT_NV_NATIVE6_QUARTER_MASK >> Quarter) & 1u)) {
        constexpr int extra_offset = (sample_offset + 2) & 3;
        const float2 sample0 = __fadd2_rn(
            hao_direct_nv_sampled_denom_value<C>(
                (&scores.v0)[sample_offset]),
            hao_direct_nv_sampled_denom_value<C>(
                (&scores.v0)[extra_offset]));
        const float2 sample1 = __fadd2_rn(
            hao_direct_nv_sampled_denom_value<C>(
                (&scores.v0)[sample_offset + 8]),
            hao_direct_nv_sampled_denom_value<C>(
                (&scores.v0)[extra_offset + 8]));
        const float2 sample01 = __fadd2_rn(sample0, sample1);
        const float2 sample23 = __fadd2_rn(
            hao_direct_nv_sampled_denom_value<C>(scores.v4),
            hao_direct_nv_sampled_denom_value<C>(scores.v12));
        const float2 sample = __fadd2_rn(sample01, sample23);
        return (sample.x + sample.y) * (8.0f / 3.0f);
    } else if constexpr (
        C::HAO_DIRECT_NV_QUARTER_SCHEDULE_MODE == 2) {
        const float2 sample = __fadd2_rn(
            hao_direct_nv_sampled_denom_value<C>(
                (&scores.v0)[sample_offset]),
            hao_direct_nv_sampled_denom_value<C>(
                (&scores.v0)[sample_offset + 8]));
        return (sample.x + sample.y) * 8.0f;
    } else {
        const float2 sample0 = __fadd2_rn(
            hao_direct_nv_sampled_denom_value<C>(
                (&scores.v0)[sample_offset]),
            hao_direct_nv_sampled_denom_value<C>(scores.v4));
        const float2 sample1 = __fadd2_rn(
            hao_direct_nv_sampled_denom_value<C>(
                (&scores.v0)[sample_offset + 8]),
            hao_direct_nv_sampled_denom_value<C>(scores.v12));
        const float2 sample = __fadd2_rn(sample0, sample1);
        return (sample.x + sample.y) * 4.0f;
    }
}

struct hao_direct_sparse_word {
    uint16_t payload;
    uint8_t metadata;
};

__device__ __forceinline__ uint32_t
hao_direct_positive_e2m1_pair_score(uint32_t packed_byte) {
    const uint32_t lo = packed_byte & 0x7u;
    const uint32_t hi = (packed_byte >> 4) & 0x7u;
    return lo > hi ? lo : hi;
}

__device__ __forceinline__ hao_direct_sparse_word
hao_direct_compress_paired_4of8(uint32_t packed) {
    int best0 = 0;
    int best1 = 1;
    uint32_t score0 =
        hao_direct_positive_e2m1_pair_score(packed);
    uint32_t score1 =
        hao_direct_positive_e2m1_pair_score(packed >> 8);
    if (score1 > score0) {
        const int index_tmp = best0;
        best0 = best1;
        best1 = index_tmp;
        const uint32_t score_tmp = score0;
        score0 = score1;
        score1 = score_tmp;
    }
    #pragma unroll
    for (int index = 2; index < 4; ++index) {
        const uint32_t score =
            hao_direct_positive_e2m1_pair_score(
                packed >> (index * 8));
        if (score > score0) {
            best1 = best0;
            score1 = score0;
            best0 = index;
            score0 = score;
        } else if (score > score1) {
            best1 = index;
            score1 = score;
        }
    }
    const int first = best0 < best1 ? best0 : best1;
    const int second = best0 < best1 ? best1 : best0;
    const uint16_t payload =
        static_cast<uint16_t>((packed >> (first * 8)) & 0xffu) |
        static_cast<uint16_t>(
            ((packed >> (second * 8)) & 0xffu) << 8);
    const uint8_t metadata =
        static_cast<uint8_t>(first | (second << 2));
    return {payload, metadata};
}

__device__ __forceinline__ hao_direct_sparse_word
hao_direct_compress_paired_4of8_packed(uint32_t packed) {
    const uint32_t lo = packed & 0x07070707u;
    const uint32_t hi = (packed >> 4) & 0x07070707u;
    const uint32_t scores = __vmaxu4(lo, hi);
    // Each byte is ordered by (pair maximum, pair index), making ties
    // deterministic while retaining four independent SIMD lanes.
    const uint32_t ranked =
        (scores << 2) | 0x03020100u;
    const uint32_t paired =
        __byte_perm(ranked, 0, 0x2301);
    const uint32_t pair_winners =
        __vmaxu4(ranked, paired);
    const uint32_t pair_losers =
        __vminu4(ranked, paired);
    // Pack (loser, winner) for pair groups 01 and 23. Unsigned halfword
    // min/max then chooses the winning group and carries its loser with it.
    const uint32_t groups =
        __byte_perm(pair_losers, pair_winners, 0x6240);
    const uint32_t swapped_groups =
        __byte_perm(groups, 0, 0x1032);
    const uint32_t winning_group =
        __vmaxu2(groups, swapped_groups);
    const uint32_t losing_group =
        __vminu2(groups, swapped_groups);
    const uint32_t top = (winning_group >> 8) & 0xffu;
    const uint32_t runner_from_top_group =
        winning_group & 0xffu;
    const uint32_t runner_from_other_group =
        (losing_group >> 8) & 0xffu;
    const uint32_t runner =
        max(runner_from_top_group, runner_from_other_group);
    const uint32_t top_index = top & 0x3u;
    const uint32_t runner_index = runner & 0x3u;
    const uint32_t first =
        min(top_index, runner_index);
    const uint32_t second =
        max(top_index, runner_index);
    const uint32_t selected =
        __byte_perm(packed, 0, first | (second << 4));
    return {
        static_cast<uint16_t>(selected),
        static_cast<uint8_t>(first | (second << 2))};
}

__device__ __forceinline__ hao_direct_sparse_word
hao_direct_compress_paired_4of8_branchless(uint32_t packed) {
    const uint32_t score0 =
        max(packed & 0x7u, (packed >> 4) & 0x7u);
    const uint32_t score1 =
        max((packed >> 8) & 0x7u, (packed >> 12) & 0x7u);
    const uint32_t score2 =
        max((packed >> 16) & 0x7u, (packed >> 20) & 0x7u);
    const uint32_t score3 =
        max((packed >> 24) & 0x7u, (packed >> 28) & 0x7u);
    const uint32_t record0 = score0 << 2;
    const uint32_t record1 = (score1 << 2) | 1u;
    const uint32_t record2 = (score2 << 2) | 2u;
    const uint32_t record3 = (score3 << 2) | 3u;
    const uint32_t winner01 = max(record0, record1);
    const uint32_t loser01 = min(record0, record1);
    const uint32_t winner23 = max(record2, record3);
    const uint32_t loser23 = min(record2, record3);
    const bool first_group_wins = winner01 >= winner23;
    const uint32_t top =
        first_group_wins ? winner01 : winner23;
    const uint32_t paired_loser =
        first_group_wins ? loser01 : loser23;
    const uint32_t other_winner =
        first_group_wins ? winner23 : winner01;
    const uint32_t runner =
        max(paired_loser, other_winner);
    const uint32_t top_index = top & 0x3u;
    const uint32_t runner_index = runner & 0x3u;
    const uint32_t first =
        min(top_index, runner_index);
    const uint32_t second =
        max(top_index, runner_index);
    const uint32_t selected =
        __byte_perm(packed, 0, first | (second << 4));
    return {
        static_cast<uint16_t>(selected),
        static_cast<uint8_t>(first | (second << 2))};
}

__device__ __forceinline__ uint8_t
hao_direct_sparse_selector_lut_entry(uint32_t index) {
    const uint32_t record0 = (index & 0x3u) << 2;
    const uint32_t record1 =
        (((index >> 2) & 0x3u) << 2) | 1u;
    const uint32_t record2 =
        (((index >> 4) & 0x3u) << 2) | 2u;
    const uint32_t record3 =
        (((index >> 6) & 0x3u) << 2) | 3u;
    const uint32_t winner01 = max(record0, record1);
    const uint32_t loser01 = min(record0, record1);
    const uint32_t winner23 = max(record2, record3);
    const uint32_t loser23 = min(record2, record3);
    const bool first_group_wins = winner01 >= winner23;
    const uint32_t top =
        first_group_wins ? winner01 : winner23;
    const uint32_t runner = max(
        first_group_wins ? loser01 : loser23,
        first_group_wins ? winner23 : winner01);
    const uint32_t top_index = top & 0x3u;
    const uint32_t runner_index = runner & 0x3u;
    const uint32_t first =
        min(top_index, runner_index);
    const uint32_t second =
        max(top_index, runner_index);
    return static_cast<uint8_t>(first | (second << 2));
}

template <bool OrScore>
__device__ __forceinline__ hao_direct_sparse_word
hao_direct_compress_paired_4of8_lut(
    uint32_t packed,
    const hao_direct_sparse_selector_lut &lut
) {
    uint32_t index;
    if constexpr (OrScore) {
        const uint32_t two_bit_bytes =
            ((packed | (packed >> 4)) >> 1) &
            0x03030303u;
        index =
            (two_bit_bytes * 0x01041040u) >> 24;
    } else {
        const uint32_t score0 = max(
            (packed >> 1) & 0x3u,
            (packed >> 5) & 0x3u);
        const uint32_t score1 = max(
            (packed >> 9) & 0x3u,
            (packed >> 13) & 0x3u);
        const uint32_t score2 = max(
            (packed >> 17) & 0x3u,
            (packed >> 21) & 0x3u);
        const uint32_t score3 = max(
            (packed >> 25) & 0x3u,
            (packed >> 29) & 0x3u);
        index =
            score0 |
            (score1 << 2) |
            (score2 << 4) |
            (score3 << 6);
    }
    const uint32_t metadata = lut.data[index];
    const uint32_t first = metadata & 0x3u;
    const uint32_t second = metadata >> 2;
    const uint32_t selected =
        __byte_perm(packed, 0, first | (second << 4));
    return {
        static_cast<uint16_t>(selected),
        static_cast<uint8_t>(metadata)};
}

template <typename P>
__device__ __forceinline__ hao_direct_sparse_word
hao_direct_compress_paired_4of8_dispatch(
    uint32_t packed,
    const P &p
) {
    if constexpr (
        TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR == 2) {
        return hao_direct_compress_paired_4of8_packed(packed);
    } else if constexpr (
        TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR == 3) {
        return hao_direct_compress_paired_4of8_branchless(packed);
    } else if constexpr (
        TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR == 4) {
        return hao_direct_compress_paired_4of8_lut<false>(
            packed, *p.selector_lut);
    } else if constexpr (
        TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR == 5) {
        return hao_direct_compress_paired_4of8_lut<true>(
            packed, *p.selector_lut);
    } else {
        return hao_direct_compress_paired_4of8(packed);
    }
}

template <int Quarter, typename P>
__device__ __forceinline__ int hao_direct_store_p_nvfp4_quarter_tmem(
    P &p,
    int reader_warp,
    uint32_t word0,
    uint32_t word1,
    uint32_t word2,
    uint32_t word3
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    if constexpr (hao_direct_sparse_p_type<P>) {
        uint32_t payload0;
        uint32_t payload1;
        uint16_t metadata;
        if constexpr (
            TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR == 1) {
            payload0 =
                (word0 & 0xffffu) |
                ((word1 & 0xffffu) << 16);
            payload1 =
                (word2 & 0xffffu) |
                ((word3 & 0xffffu) << 16);
            metadata = 0x4444u;
        } else {
            const hao_direct_sparse_word sparse0 =
                hao_direct_compress_paired_4of8_dispatch(word0, p);
            const hao_direct_sparse_word sparse1 =
                hao_direct_compress_paired_4of8_dispatch(word1, p);
            const hao_direct_sparse_word sparse2 =
                hao_direct_compress_paired_4of8_dispatch(word2, p);
            const hao_direct_sparse_word sparse3 =
                hao_direct_compress_paired_4of8_dispatch(word3, p);
            payload0 =
                static_cast<uint32_t>(sparse0.payload) |
                (static_cast<uint32_t>(sparse1.payload) << 16);
            payload1 =
                static_cast<uint32_t>(sparse2.payload) |
                (static_cast<uint32_t>(sparse3.payload) << 16);
            metadata =
                static_cast<uint16_t>(sparse0.metadata) |
                (static_cast<uint16_t>(sparse1.metadata) << 4) |
                (static_cast<uint16_t>(sparse2.metadata) << 8) |
                (static_cast<uint16_t>(sparse3.metadata) << 12);
        }
        const int row =
            reader_warp * 32 + warp::laneid();
        int retained_qsum_x2 = 0;
        hao_direct_accumulate_qsum_x2_word(
            retained_qsum_x2, payload0);
        hao_direct_accumulate_qsum_x2_word(
            retained_qsum_x2, payload1);
        auto *payload =
            Quarter < 2 ? p.payload : p.tail_payload;
        const uint32_t payload_base =
            __cvta_generic_to_shared(&payload->data[0]);
        const uint32_t payload_addr =
            hao_direct_sparse_p_tile::idx(
                payload_base, {row, Quarter * 8});
        asm volatile(
            "st.shared.v2.b32 [%0], {%1, %2};"
            :: "r"(payload_addr), "r"(payload0), "r"(payload1));

        auto &metadata_half =
            p.metadata_halves[Quarter >> 1];
        const uint32_t metadata_smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(
                &metadata_half.data[row]));
        if constexpr ((Quarter & 1) == 0) {
            asm volatile(
                "st.shared.u16 [%0], %1;"
                :: "r"(metadata_smem_addr), "h"(metadata));
        } else {
            uint16_t first_metadata;
            asm volatile(
                "ld.shared.u16 %0, [%1];"
                : "=h"(first_metadata)
                : "r"(metadata_smem_addr));
            const uint32_t packed_metadata =
                static_cast<uint32_t>(first_metadata) |
                (static_cast<uint32_t>(metadata) << 16);
            const uint32_t metadata_base =
                Quarter < 2 ? p.addr : p.tail_addr;
            const uint32_t metadata_tmem_addr =
                metadata_base +
                (static_cast<uint32_t>(reader_warp * 32) << 16) +
                (Quarter >> 1);
            asm volatile(
                "{tcgen05.st.sync.aligned.32x32b.x1.b32 "
                "[%0], {%1};}"
                :: "r"(metadata_tmem_addr),
                   "r"(packed_metadata));
        }
        return retained_qsum_x2;
    } else {
        const uint32_t addr =
            p.addr + (reader_warp * 32 << 16) + Quarter * 4;
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x4.b32 "
            "[%0], {%1,%2,%3,%4};}"
            :: "r"(addr),
               "r"(word0),
               "r"(word1),
               "r"(word2),
               "r"(word3));
        return 0;
    }
}

template <int Quarter, typename PScale>
__device__ __forceinline__ void hao_direct_store_p_nvfp4_scales_smem(
    PScale &p_scale,
    int reader_warp,
    float scale0,
    float scale1
) {
    const fp8e4m3_2 packed =
        base_types::convertor<fp8e4m3_2, float2>::convert(
            float2{scale0, scale1});
    const uint16_t bits = std::bit_cast<uint16_t>(packed);
    const uint32_t base =
        __cvta_generic_to_shared(&p_scale.data[0]);
    const uint32_t offset =
        warp::laneid() * 16 +
        reader_warp * 4 +
        (Quarter & 1) * 2 +
        (Quarter >> 1) * 512;
    asm volatile(
        "st.shared.u16 [%0], %1;"
        :: "r"(base + offset), "h"(bits));
}

template <int Quarter, typename PScale>
__device__ __forceinline__ void
hao_direct_store_p_nvfp4_scale_bits_smem(
    PScale &p_scale,
    int reader_warp,
    uint8_t scale_bits
) {
    const uint16_t bits =
        static_cast<uint16_t>(scale_bits) |
        (static_cast<uint16_t>(scale_bits) << 8);
    const uint32_t base =
        __cvta_generic_to_shared(&p_scale.data[0]);
    const uint32_t offset =
        warp::laneid() * 16 +
        reader_warp * 4 +
        (Quarter & 1) * 2 +
        (Quarter >> 1) * 512;
    asm volatile(
        "st.shared.u16 [%0], %1;"
        :: "r"(base + offset), "h"(bits));
}

template <int Quarter, typename C, typename PScale>
__device__ __forceinline__ void
hao_direct_store_identity_p_scale(
    PScale &p_scale,
    int reader_warp
) {
    if constexpr (C::HAO_DIRECT_NVFP4_PV) {
        hao_direct_store_p_nvfp4_scales_smem<Quarter>(
            p_scale, reader_warp, 1.0f, 1.0f);
    } else {
        constexpr uint8_t E8M0_ONE = 127;
        const int row =
            reader_warp * 32 + warp::laneid();
        fp4pv_store_mxfp4_scale_byte_swizzled(
            p_scale, row, Quarter, E8M0_ONE);
    }
}

template <int Quarter, typename C, typename P, typename PScale>
__device__ __forceinline__ float
hao_direct_store_fixed_p_quarter(
    P &p,
    PScale &p_scale,
    int reader_warp
) {
    // Positive E2M1 code 0x2 is exactly 1.0.
    constexpr uint32_t P_ONES = 0x22222222u;
    hao_direct_store_identity_p_scale<Quarter, C>(
        p_scale, reader_warp);
    hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
        p,
        reader_warp,
        P_ONES,
        P_ONES,
        P_ONES,
        P_ONES);
    return 32.0f;
}

template <int Quarter, typename C, typename P, typename PScale>
__device__ __forceinline__ float
hao_direct_pack_raw_score_quarter(
    P &p,
    PScale &p_scale,
    int reader_warp,
    score_quarter_regs &scores
) {
    uint32_t words[4];
    fp4pv_packed_float_to_e2m1(
        scores.v0.x, scores.v0.y,
        scores.v1.x, scores.v1.y,
        scores.v2.x, scores.v2.y,
        scores.v3.x, scores.v3.y,
        words[0]);
    fp4pv_packed_float_to_e2m1(
        scores.v4.x, scores.v4.y,
        scores.v5.x, scores.v5.y,
        scores.v6.x, scores.v6.y,
        scores.v7.x, scores.v7.y,
        words[1]);
    fp4pv_packed_float_to_e2m1(
        scores.v8.x, scores.v8.y,
        scores.v9.x, scores.v9.y,
        scores.v10.x, scores.v10.y,
        scores.v11.x, scores.v11.y,
        words[2]);
    fp4pv_packed_float_to_e2m1(
        scores.v12.x, scores.v12.y,
        scores.v13.x, scores.v13.y,
        scores.v14.x, scores.v14.y,
        scores.v15.x, scores.v15.y,
        words[3]);
    hao_direct_store_identity_p_scale<Quarter, C>(
        p_scale, reader_warp);
    hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
        p,
        reader_warp,
        words[0],
        words[1],
        words[2],
        words[3]);
    return 32.0f;
}

template <
    int Quarter,
    typename C,
    bool ForceAffine = false,
    bool StoreScaleSmem = true,
    bool CapturePackedWords = false,
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD != 0
    typename Hook = hao_direct_noop_tail_hook,
#endif
    typename P,
    typename PScale>
__device__ __forceinline__ float
hao_direct_transform_pack_nvfp4_quarter(
    P &p,
    PScale &p_scale,
    int reader_warp,
    score_quarter_regs &scores,
    float row_max,
    float scale_log2,
    float raw_group_max,
    uint16_t *scale_bits_out = nullptr,
    float *unscaled_sum_out = nullptr,
    uint32_t *packed_words_out = nullptr
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD != 0
    ,
    Hook hook = {}
#endif
) {
    static_assert(
        sizeof(score_quarter_regs) == 16 * sizeof(float2));
    if constexpr (C::HAO_DIRECT_LOG2_QUANT) {
        uint32_t word0;
        uint32_t word1;
        uint32_t word2;
        uint32_t word3;
        float scale0;
        float scale1;
        float prepared_quant_bias = 0.0f;
        float prepared_scale = 0.0f;
        uint8_t prepared_scale_bits = 0;
        if constexpr (C::HAO_DIRECT_NV_QUARTER_SCALE) {
            constexpr float LOG2_FP4_MAX =
                2.584962500721156f;
            const float normalized_group_max =
                (raw_group_max - row_max) * scale_log2 +
                0.5f * static_cast<float>(
                    C::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2);
            const float quant_log2_scale =
                fmaxf(normalized_group_max, -100.0f) -
                LOG2_FP4_MAX;
            const float stored_log2_scale =
                quant_log2_scale +
                C::HAO_DIRECT_NV_P_GLOBAL_LOG2;
            if constexpr (
                C::HAO_DIRECT_NV_SCALE_ENCODE_MODE == 1) {
                int scale_exponent =
                    __float2int_ru(stored_log2_scale);
                scale_exponent =
                    scale_exponent < -9
                        ? -9
                        : (scale_exponent > 8 ? 8 : scale_exponent);
                prepared_quant_bias =
                    -row_max * scale_log2 -
                    static_cast<float>(scale_exponent) +
                    C::HAO_DIRECT_NV_P_GLOBAL_LOG2;
                prepared_scale = __uint_as_float(
                    static_cast<uint32_t>(
                        scale_exponent + 127) << 23);
                prepared_scale_bits =
                    scale_exponent >= -6
                        ? static_cast<uint8_t>(
                              (scale_exponent + 7) << 3)
                        : static_cast<uint8_t>(
                              1u << (scale_exponent + 9));
            } else if constexpr (
                C::HAO_DIRECT_NV_SCALE_ENCODE_MODE >= 2 &&
                C::HAO_DIRECT_NV_SCALE_ENCODE_MODE <= 5) {
                constexpr float MAGIC = 12582912.0f;
                constexpr float MAGIC_PLUS_E4M3_ONE = MAGIC + 56.0f;
                float encoded_log2;
                if constexpr (
                    C::HAO_DIRECT_NV_SCALE_ENCODE_MODE == 4) {
                    int scale_code =
                        hao_direct_float_to_s32_rn_sat(
                            __fmaf_rn(
                                stored_log2_scale,
                                8.0f,
                                56.0f));
                    asm(
                        "max.s32 %0, %0, 8;\n\t"
                        "min.s32 %0, %0, 126;"
                        : "+r"(scale_code));
                    prepared_scale_bits =
                        static_cast<uint8_t>(scale_code);
                    encoded_log2 =
                        static_cast<float>(scale_code - 56) *
                        0.125f;
                } else {
                    float carrier;
                    if constexpr (
                        C::HAO_DIRECT_NV_SCALE_ENCODE_MODE == 3) {
                        // Express clamp(x, -6, 8.75) in a normalized
                        // coordinate so saturation and the affine map share
                        // an FMA.
                        constexpr float LOG2_RANGE = 14.75f;
                        constexpr float INV_LOG2_RANGE =
                            1.0f / LOG2_RANGE;
                        constexpr float UNIT_BIAS =
                            6.0f / LOG2_RANGE;
                        const float bounded_unit =
                            hao_direct_fma_sat_scalar(
                                stored_log2_scale,
                                INV_LOG2_RANGE,
                                UNIT_BIAS);
                        carrier = __fmaf_rn(
                            bounded_unit,
                            8.0f * LOG2_RANGE,
                            MAGIC_PLUS_E4M3_ONE - 48.0f);
                    } else {
                        const float encode_log2 =
                            C::HAO_DIRECT_NV_SCALE_SATFINITE
                                ? fminf(
                                      fmaxf(
                                          stored_log2_scale,
                                          -6.0f),
                                      8.75f)
                                : stored_log2_scale;
                        carrier = __fmaf_rn(
                            encode_log2,
                            8.0f,
                            MAGIC_PLUS_E4M3_ONE);
                    }
                    encoded_log2 =
                        (carrier - MAGIC_PLUS_E4M3_ONE) *
                        0.125f;
                    prepared_scale_bits = static_cast<uint8_t>(
                        __float_as_uint(carrier));
                }
                const float payload_log2_scale =
                    C::HAO_DIRECT_NV_SCALE_ENCODE_MODE == 5
                        ? stored_log2_scale
                        : encoded_log2;
                prepared_quant_bias =
                    -row_max * scale_log2 -
                    payload_log2_scale +
                    C::HAO_DIRECT_NV_P_GLOBAL_LOG2;
                const uint32_t fp32_scale_bits =
                    (static_cast<uint32_t>(prepared_scale_bits) << 20) +
                    (120u << 23);
                prepared_scale = __uint_as_float(fp32_scale_bits);
            } else {
                prepared_quant_bias =
                    -row_max * scale_log2 -
                    quant_log2_scale;
                prepared_scale =
                    hao_direct_exp2_native_scalar(
                        stored_log2_scale);
            }
        }
        if constexpr (
            C::HAO_DIRECT_NV_QUARTER_SCHEDULE_MODE != 0) {
            uint32_t scheduled_words[4];
            const float scaled_sum =
                hao_direct_quantize_nvfp4_quarter_log2_scheduled<
                    Quarter,
                    C,
                    ForceAffine>(
                        scores,
                        scale_log2,
                        prepared_quant_bias,
                        scheduled_words
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD != 0
                        ,
                        hook);
#else
                        );
#endif
            if constexpr (CapturePackedWords) {
                #pragma unroll
                for (int word = 0; word < 4; ++word) {
                    packed_words_out[word] = scheduled_words[word];
                }
            }
            if (unscaled_sum_out != nullptr) {
                *unscaled_sum_out = scaled_sum;
            }
            if constexpr (StoreScaleSmem) {
                hao_direct_store_p_nvfp4_scale_bits_smem<Quarter>(
                    p_scale, reader_warp, prepared_scale_bits);
            } else {
                *scale_bits_out =
                    static_cast<uint16_t>(prepared_scale_bits) |
                    (static_cast<uint16_t>(
                         prepared_scale_bits) << 8);
            }
            const int retained_qsum_x2 =
                hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
                p,
                reader_warp,
                scheduled_words[0],
                scheduled_words[1],
                scheduled_words[2],
                scheduled_words[3]);
            if constexpr (C::HAO_DIRECT_SPARSE_PV) {
                return prepared_scale *
                       (0.5f * static_cast<float>(
                           retained_qsum_x2));
            }
            const float guarded_sum =
                C::HAO_DIRECT_NV_SCALE_SATFINITE &&
                        C::HAO_DIRECT_NV_DENOM_SOURCE == 0
                    ? fmaxf(scaled_sum, 6.0f)
                    : scaled_sum;
            return prepared_scale * guarded_sum;
        }
        const float sum0 =
            hao_direct_quantize_nvfp4_group_log2<
                Quarter, 0, C, ForceAffine>(
                    &scores.v0,
                    row_max,
                    scale_log2,
                    raw_group_max,
                    prepared_quant_bias,
                    prepared_scale,
                    word0,
                    word1,
                    scale0);
        const float sum1 =
            hao_direct_quantize_nvfp4_group_log2<
                Quarter, 8, C, ForceAffine>(
                    &scores.v8,
                    row_max,
                    scale_log2,
                    raw_group_max,
                    prepared_quant_bias,
                    prepared_scale,
                    word2,
                    word3,
                    scale1);
        if constexpr (
            C::HAO_DIRECT_NV_SCALE_ENCODE_MODE != 0) {
            if constexpr (StoreScaleSmem) {
                hao_direct_store_p_nvfp4_scale_bits_smem<Quarter>(
                    p_scale, reader_warp, prepared_scale_bits);
            } else {
                *scale_bits_out =
                    static_cast<uint16_t>(prepared_scale_bits) |
                    (static_cast<uint16_t>(prepared_scale_bits) << 8);
            }
        } else {
            if constexpr (StoreScaleSmem) {
                hao_direct_store_p_nvfp4_scales_smem<Quarter>(
                    p_scale, reader_warp, scale0, scale1);
            } else {
                const fp8e4m3_2 packed =
                    base_types::convertor<
                        fp8e4m3_2,
                        float2>::convert(float2{scale0, scale1});
                *scale_bits_out = std::bit_cast<uint16_t>(packed);
            }
        }
        hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
            p, reader_warp, word0, word1, word2, word3);
        constexpr bool USE_QUANTIZED_DENOM =
            C::HAO_DIRECT_NV_QUANTIZED_DENOM ||
            (C::HAO_DIRECT_NV_HYBRID_DENOM &&
             (ForceAffine || Quarter == 0));
        if constexpr (C::HAO_DIRECT_FAKE_DENOM_CEILING) {
            return sum0 + sum1;
        } else if constexpr (USE_QUANTIZED_DENOM) {
            int qsum0_x2 = 0;
            int qsum1_x2 = 0;
            hao_direct_accumulate_qsum_x2_word(qsum0_x2, word0);
            hao_direct_accumulate_qsum_x2_word(qsum0_x2, word1);
            hao_direct_accumulate_qsum_x2_word(qsum1_x2, word2);
            hao_direct_accumulate_qsum_x2_word(qsum1_x2, word3);
            if constexpr (C::HAO_DIRECT_NV_QUARTER_SCALE) {
                return prepared_scale *
                       (0.5f * static_cast<float>(
                           qsum0_x2 + qsum1_x2));
            } else {
                return 0.5f * (
                    scale0 * static_cast<float>(qsum0_x2) +
                    scale1 * static_cast<float>(qsum1_x2));
            }
        } else if constexpr (C::HAO_DIRECT_NV_QUARTER_SCALE) {
            return prepared_scale * (sum0 + sum1);
        } else {
            return scale0 * sum0 + scale1 * sum1;
        }
    }

    scores.v0 = hao_direct_exp2_nvfp4_pair<Quarter, 0, C>(
        scores.v0, row_max, scale_log2);
    scores.v1 = hao_direct_exp2_nvfp4_pair<Quarter, 1, C>(
        scores.v1, row_max, scale_log2);
    scores.v2 = hao_direct_exp2_nvfp4_pair<Quarter, 2, C>(
        scores.v2, row_max, scale_log2);
    scores.v3 = hao_direct_exp2_nvfp4_pair<Quarter, 3, C>(
        scores.v3, row_max, scale_log2);
    scores.v4 = hao_direct_exp2_nvfp4_pair<Quarter, 4, C>(
        scores.v4, row_max, scale_log2);
    scores.v5 = hao_direct_exp2_nvfp4_pair<Quarter, 5, C>(
        scores.v5, row_max, scale_log2);
    scores.v6 = hao_direct_exp2_nvfp4_pair<Quarter, 6, C>(
        scores.v6, row_max, scale_log2);
    scores.v7 = hao_direct_exp2_nvfp4_pair<Quarter, 7, C>(
        scores.v7, row_max, scale_log2);
    scores.v8 = hao_direct_exp2_nvfp4_pair<Quarter, 8, C>(
        scores.v8, row_max, scale_log2);
    scores.v9 = hao_direct_exp2_nvfp4_pair<Quarter, 9, C>(
        scores.v9, row_max, scale_log2);
    scores.v10 = hao_direct_exp2_nvfp4_pair<Quarter, 10, C>(
        scores.v10, row_max, scale_log2);
    scores.v11 = hao_direct_exp2_nvfp4_pair<Quarter, 11, C>(
        scores.v11, row_max, scale_log2);
    scores.v12 = hao_direct_exp2_nvfp4_pair<Quarter, 12, C>(
        scores.v12, row_max, scale_log2);
    scores.v13 = hao_direct_exp2_nvfp4_pair<Quarter, 13, C>(
        scores.v13, row_max, scale_log2);
    scores.v14 = hao_direct_exp2_nvfp4_pair<Quarter, 14, C>(
        scores.v14, row_max, scale_log2);
    scores.v15 = hao_direct_exp2_nvfp4_pair<Quarter, 15, C>(
        scores.v15, row_max, scale_log2);

    uint32_t word0;
    uint32_t word1;
    uint32_t word2;
    uint32_t word3;
    float scale0;
    float scale1;
    const float sum0 = hao_direct_quantize_nvfp4_group(
        scores.v0, scores.v1, scores.v2, scores.v3,
        scores.v4, scores.v5, scores.v6, scores.v7,
        word0, word1, scale0);
    const float sum1 = hao_direct_quantize_nvfp4_group(
        scores.v8, scores.v9, scores.v10, scores.v11,
        scores.v12, scores.v13, scores.v14, scores.v15,
        word2, word3, scale1);
    hao_direct_store_p_nvfp4_scales_smem<Quarter>(
        p_scale, reader_warp, scale0, scale1);
    hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
        p, reader_warp, word0, word1, word2, word3);
    return sum0 + sum1;
}

template <
    int Quarter,
    typename C,
    bool ForceAffine = false,
    bool CapturePackedWords = false,
    typename P,
    typename PScale>
__device__ __forceinline__ float
hao_direct_transform_pack_nvfp4_quarter_register_scale(
    P &p,
    PScale &p_scale,
    int reader_warp,
    score_quarter_regs &scores,
    float row_max,
    float scale_log2,
    float raw_group_max,
    uint16_t &scale_bits,
    float *unscaled_sum_out = nullptr,
    uint32_t *packed_words_out = nullptr
) {
    static_assert(C::HAO_DIRECT_NVFP4_PV);
    static_assert(!C::HAO_DIRECT_ROWMAX_PACK_CEILING);
    return hao_direct_transform_pack_nvfp4_quarter<
        Quarter,
        C,
        ForceAffine,
        false,
        CapturePackedWords>(
            p,
            p_scale,
            reader_warp,
            scores,
            row_max,
            scale_log2,
            raw_group_max,
            &scale_bits,
            unscaled_sum_out,
            packed_words_out);
}

template <
    int Quarter,
    typename C,
    bool ForceAffine = false,
    bool CapturePackedWords = false,
    bool PreparedAnchorAffine = false,
    bool CaptureScaleByte = false,
    bool InterleaveExactMax = false,
    bool RebaseInterleavedScale = false,
    typename P,
    typename PScale>
__device__ __forceinline__ float
hao_direct_transform_pack_mxfp4_quarter_log2(
    P &p,
    PScale &p_scale,
    int reader_warp,
    score_quarter_regs &scores,
    float row_max,
    float scale_log2,
    float raw_group_max,
    uint32_t *packed_words_out = nullptr,
    float prepared_quant_bias_base = 0.0f,
    float prepared_affine_scale = 0.0f,
    uint8_t *stored_e8m0_out = nullptr,
    uint8_t *pair_e8m0_io = nullptr,
    bool reuse_pair_scale_runtime = true,
    const score_quarter_regs *interleaved_max_scores = nullptr,
    float *interleaved_max_out = nullptr
) {
    static_assert(
        sizeof(score_quarter_regs) == 16 * sizeof(float2));
    static_assert(
        !InterleaveExactMax || C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 8);
    static_assert(!RebaseInterleavedScale || InterleaveExactMax);
    static_assert(
        !RebaseInterleavedScale ||
        C::HAO_DIRECT_MX_PWL_EXP2_MODE == 23);
    constexpr float LOG2_FP4_MAX = 2.584962500721156f;

    const float sample_bias =
        0.5f * static_cast<float>(
            C::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2);
    const float group_max =
        C::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2 != 0
            ? raw_group_max * scale_log2 -
                  static_cast<float>(
                      C::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2) +
                  sample_bias
            : PreparedAnchorAffine
            ? __fmaf_rn(
                  raw_group_max,
                  scale_log2,
                  prepared_quant_bias_base - LOG2_FP4_MAX +
                      sample_bias)
            : (raw_group_max - row_max) * scale_log2 + sample_bias;
    constexpr bool reuse_pair_scale =
        (C::HAO_DIRECT_MX_PAIR_SCALE_REUSE &
         (1 << (Quarter >> 1))) != 0;
    uint8_t e8m0;
    if constexpr (reuse_pair_scale && (Quarter & 1)) {
        if (reuse_pair_scale_runtime) {
            e8m0 = *pair_e8m0_io;
            if constexpr (
                C::HAO_DIRECT_MX_PAIR_SCALE_DELTA != 0) {
                int adjusted =
                    static_cast<int>(e8m0) +
                    C::HAO_DIRECT_MX_PAIR_SCALE_DELTA;
                asm(
                    "max.s32 %0, %0, 0;\n\t"
                    "min.s32 %0, %0, 254;"
                    : "+r"(adjusted));
                e8m0 = static_cast<uint8_t>(adjusted);
            }
        } else {
            e8m0 = hao_direct_select_mxfp4_e8m0<C>(group_max);
        }
    } else {
        e8m0 = hao_direct_select_mxfp4_e8m0<C>(group_max);
        if constexpr (reuse_pair_scale) {
            *pair_e8m0_io = e8m0;
        }
    }
    int working_e8m0 = static_cast<int>(e8m0);
    if constexpr (
        C::HAO_DIRECT_MX_SCALE_SELECT != 0 ||
        C::HAO_DIRECT_MX_DIRECT_LOG_CODE ||
        C::HAO_DIRECT_MX_LOG_LUT) {
        asm(
            "max.s32 %0, %0, 1;"
            : "+r"(working_e8m0));
    }
    const int scale_exponent = working_e8m0 - 127;
    const float2 score_scale = {scale_log2, scale_log2};
    const float quant_bias_base =
        C::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2 != 0
            ? LOG2_FP4_MAX - static_cast<float>(
                  C::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2)
            : PreparedAnchorAffine
            ? prepared_quant_bias_base
            : -row_max * scale_log2 + LOG2_FP4_MAX;
    const float quant_bias_scalar =
        quant_bias_base - static_cast<float>(scale_exponent);
    const float2 quant_bias = {
        quant_bias_scalar, quant_bias_scalar};

    uint32_t words[4] = {0, 0, 0, 0};
    float scaled_sum = 0.0f;
    float interleaved_p0 = 0.0f;
    float interleaved_p1 = 0.0f;
    float interleaved_p2 = 0.0f;
    float interleaved_p3 = 0.0f;
    float interleaved_p4 = 0.0f;
    float interleaved_p5 = 0.0f;
    float interleaved_p6 = 0.0f;
    float interleaved_p7 = 0.0f;
    float interleaved_p8 = 0.0f;
    float interleaved_p9 = 0.0f;
    float interleaved_p10 = 0.0f;
    if constexpr (C::HAO_DIRECT_MX_LOG_LUT) {
        static_assert(C::HAO_DIRECT_MX_LOG_LUT_BITS == 4);
        const uint32_t valid_mask =
            0u - static_cast<uint32_t>(e8m0 != 0);
        constexpr float BIT_CODE_SCALE =
            C::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE == 7
                ? 2.0f
                : 1.826841f;
        const float2 fused_bit_score_scale = __fmul2_rn(
            score_scale,
            float2{BIT_CODE_SCALE, BIT_CODE_SCALE});
        const float2 fused_bit_quant_bias = __ffma2_rn(
            quant_bias,
            float2{BIT_CODE_SCALE, BIT_CODE_SCALE},
            float2{12582914.0f, 12582914.0f});
        int quantized_sum_x2 = 0;
        fp4pv_stage2_static_for<0, 16>([&](auto pair_tag) {
            constexpr int Pair = decltype(pair_tag)::value;
            constexpr int Word = Pair / 4;
            constexpr int Shift = (Pair % 4) * 8;
            uint32_t packed_byte;
            if constexpr (
                C::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE == 5 ||
                C::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE == 7) {
                const float2 encoded = __ffma2_rn(
                    (&scores.v0)[Pair],
                    fused_bit_score_scale,
                    fused_bit_quant_bias);
                packed_byte =
                    hao_direct_pack_e2m1_pair_from_magic_encoded(
                        encoded);
            } else {
                const float2 log_value = __ffma2_rn(
                    (&scores.v0)[Pair],
                    score_scale,
                    quant_bias);
                if constexpr (
                    C::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE == 0) {
                    packed_byte =
                        hao_direct_pack_e2m1_pair_from_log_lut16(
                            log_value);
                } else if constexpr (
                    C::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE == 1) {
                    packed_byte =
                        hao_direct_pack_e2m1_pair_from_log_bits<1>(
                            log_value);
                } else if constexpr (
                    C::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE == 2) {
                    packed_byte =
                        hao_direct_pack_e2m1_pair_from_log_bits<2>(
                            log_value);
                } else if constexpr (
                    C::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE == 3) {
                    packed_byte =
                        hao_direct_pack_e2m1_pair_from_log_bits_simd(
                            log_value);
                } else {
                    static_assert(
                        C::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE == 4);
                    packed_byte =
                        hao_direct_pack_e2m1_pair_from_log_bits<0>(
                            log_value);
                }
            }
            words[Word] |= packed_byte << Shift;
        });
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] &= valid_mask;
            if constexpr (!C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                quantized_sum_x2 =
                    fp4pv_stage2_qsum_weights_from_word(
                        words[i], quantized_sum_x2);
            }
        }
        if constexpr (C::HAO_DIRECT_FAKE_DENOM_CEILING) {
            scaled_sum = 32.0f;
        } else {
            scaled_sum =
                static_cast<float>(quantized_sum_x2) * 0.5f;
        }
    } else if constexpr (C::HAO_DIRECT_MX_DIRECT_LOG_CODE) {
        const uint32_t valid_mask =
            0u - static_cast<uint32_t>(e8m0 != 0);
        int quantized_sum_x2 = 0;
        if constexpr (
            C::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE <= 2) {
            hao_direct_pack_mxfp4_log_codes<C>(
                scores,
                score_scale,
                quant_bias,
                words);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                words[i] &= valid_mask;
                if constexpr (!C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                    quantized_sum_x2 =
                        fp4pv_stage2_qsum_weights_from_word(
                            words[i], quantized_sum_x2);
                }
            }
        } else {
            const auto result0 =
                hao_direct_pack_mxfp4_log_word_predicated<0, C>(
                    scores, score_scale, quant_bias);
            const auto result1 =
                hao_direct_pack_mxfp4_log_word_predicated<4, C>(
                    scores, score_scale, quant_bias);
            words[0] = result0.word & valid_mask;
            words[1] = result1.word & valid_mask;
            if constexpr (
                C::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE == 3 &&
                !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                quantized_sum_x2 =
                    fp4pv_stage2_qsum_weights_from_word(
                        words[0], quantized_sum_x2);
            }

            const auto result2 =
                hao_direct_pack_mxfp4_log_word_predicated<8, C>(
                    scores, score_scale, quant_bias);
            words[2] = result2.word & valid_mask;
            if constexpr (
                C::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE == 3 &&
                !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                quantized_sum_x2 =
                    fp4pv_stage2_qsum_weights_from_word(
                        words[1], quantized_sum_x2);
            }

            const auto result3 =
                hao_direct_pack_mxfp4_log_word_predicated<12, C>(
                    scores, score_scale, quant_bias);
            words[3] = result3.word & valid_mask;
            if constexpr (
                C::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE == 3 &&
                !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                quantized_sum_x2 =
                    fp4pv_stage2_qsum_weights_from_word(
                        words[2], quantized_sum_x2);
                quantized_sum_x2 =
                    fp4pv_stage2_qsum_weights_from_word(
                        words[3], quantized_sum_x2);
            } else if constexpr (
                C::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE == 4 &&
                !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                const int pair_owned_sum =
                    (result0.pair_owned_qsum_x2 +
                     result1.pair_owned_qsum_x2) +
                    (result2.pair_owned_qsum_x2 +
                     result3.pair_owned_qsum_x2);
                quantized_sum_x2 =
                    e8m0 != 0 ? pair_owned_sum : 0;
            }
        }
        if constexpr (C::HAO_DIRECT_FAKE_DENOM_CEILING) {
            scaled_sum = 32.0f;
        } else {
            scaled_sum =
                static_cast<float>(quantized_sum_x2) * 0.5f;
        }
    } else {
        constexpr bool PWL_WEIGHT_TUNED =
            C::HAO_DIRECT_MX_PWL_EXP2_MODE == 7;
        constexpr bool PWL_EIGHTH_TUNED =
            C::HAO_DIRECT_MX_PWL_EXP2_MODE == 8;
        constexpr float PWL_LINEAR =
            PWL_EIGHTH_TUNED
                ? 0.47158433f
                : (PWL_WEIGHT_TUNED ? 0.48881574f : 0.5009f);
        constexpr float PWL_BIAS =
            PWL_EIGHTH_TUNED
                ? 1.0201012f
                : (PWL_WEIGHT_TUNED ? 1.0278957f : 1.1556f);
        constexpr float PWL_HIGH_SLOPE =
            PWL_EIGHTH_TUNED
                ? 2.1975632f
                : (PWL_WEIGHT_TUNED ? 2.2071569f : 2.4690f);
        constexpr float PWL_HINGE =
            PWL_HIGH_SLOPE - PWL_LINEAR;
        constexpr float PWL_THRESHOLD =
            PWL_EIGHTH_TUNED
                ? 0.8756810f
                : (PWL_WEIGHT_TUNED ? 0.8564587f : 1.0f);
        constexpr float CUBIC_A = 0.07839806f;
        constexpr float CUBIC_B = 0.28625049f;
        constexpr float CUBIC_C = 0.63145205f;
        constexpr float CUBIC_D = 0.99202336f;
        const float2 pwl_linear_scale = {
            PWL_LINEAR * scale_log2,
            PWL_LINEAR * scale_log2};
        const float2 pwl_hinge_scale = {
            PWL_HINGE * scale_log2,
            PWL_HINGE * scale_log2};
        const float pwl_threshold =
            (PWL_THRESHOLD - quant_bias_scalar) / scale_log2;
        const float pwl_linear_bias_scalar =
            (PWL_LINEAR + PWL_HINGE) * quant_bias_scalar +
            PWL_BIAS - PWL_HINGE * PWL_THRESHOLD;
        const float2 pwl_linear_bias = {
            pwl_linear_bias_scalar,
            pwl_linear_bias_scalar};
        const float cubic_score_scale2 =
            scale_log2 * scale_log2;
        const float cubic_quant_bias2 =
            quant_bias_scalar * quant_bias_scalar;
        const float cubic_raw_a_scalar =
            CUBIC_A * cubic_score_scale2 * scale_log2;
        const float cubic_raw_b_scalar =
            cubic_score_scale2 * __fmaf_rn(
                3.0f * CUBIC_A,
                quant_bias_scalar,
                CUBIC_B);
        const float cubic_raw_c_scalar =
            scale_log2 * __fmaf_rn(
                3.0f * CUBIC_A,
                cubic_quant_bias2,
                __fmaf_rn(
                    2.0f * CUBIC_B,
                    quant_bias_scalar,
                    CUBIC_C));
        const float cubic_raw_d_scalar = __fmaf_rn(
            CUBIC_A * quant_bias_scalar,
            cubic_quant_bias2,
            __fmaf_rn(
                CUBIC_B,
                cubic_quant_bias2,
                __fmaf_rn(
                    CUBIC_C,
                    quant_bias_scalar,
                    CUBIC_D)));
        const float2 cubic_raw_a = {
            cubic_raw_a_scalar, cubic_raw_a_scalar};
        const float2 cubic_raw_b = {
            cubic_raw_b_scalar, cubic_raw_b_scalar};
        const float2 cubic_raw_c = {
            cubic_raw_c_scalar, cubic_raw_c_scalar};
        const float2 cubic_raw_d = {
            cubic_raw_d_scalar, cubic_raw_d_scalar};
        constexpr float AFFINE_A = C::HAO_DIRECT_MX_AFFINE_A;
        constexpr float AFFINE_B = C::HAO_DIRECT_MX_AFFINE_B;
        const float affine_raw_scale_scalar =
            PreparedAnchorAffine
                ? prepared_affine_scale
                : AFFINE_A * scale_log2;
        const float2 affine_raw_scale = {
            affine_raw_scale_scalar,
            affine_raw_scale_scalar};
        const float2 affine_raw_bias = {
            AFFINE_A * quant_bias_scalar + AFFINE_B,
            AFFINE_A * quant_bias_scalar + AFFINE_B};
        auto transform_pair = [&](auto pair_tag) {
            constexpr int Pair = decltype(pair_tag)::value;
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 2) {
                const float2 raw_score = (&scores.v0)[Pair];
                float2 approximation;
                if constexpr (
                    C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 9) {
                    constexpr bool hybrid_native =
                        C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 12;
                    constexpr int local_pair = Pair & 3;
                    constexpr int packed_word = Pair >> 2;
                    constexpr bool scheduled_base4 =
                        C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 16 &&
                        C::HAO_DIRECT_MX_PWL_EXP2_MODE <= 23;
                    constexpr int sample_offset = Quarter & 3;
                    constexpr bool mode23_base_native =
                        Pair == sample_offset ||
                        Pair == sample_offset + 8 ||
                        Pair == 4 || Pair == 12;
                    constexpr bool mode23_opposite_native =
                        Pair == ((sample_offset + 2) & 3) ||
                        Pair == 4 + 2 ||
                        Pair == 8 + ((sample_offset + 2) & 3) ||
                        Pair == 12 + 2;
                    constexpr bool mode23_native =
                        (C::HAO_DIRECT_MX_MODE23_NATIVE_QUARTER_MASK &
                         (1u << Quarter)) != 0 &&
                        (C::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY >= 4 ||
                         (C::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY >= 1 &&
                          mode23_base_native) ||
                         (C::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY >= 2 &&
                          mode23_opposite_native) ||
                         (C::HAO_DIRECT_MX_Q1_SELF_MAX == 4 &&
                          Quarter == 1 &&
                          (Pair == sample_offset || Pair == 4)));
                    constexpr bool native_sample =
                        hybrid_native &&
                        (C::HAO_DIRECT_MX_PWL_EXP2_MODE != 23
                             ?
                        ((scheduled_base4 &&
                          (Pair == sample_offset ||
                           Pair == sample_offset + 8 ||
                           Pair == 4 ||
                           Pair == 12)) ||
                         C::HAO_DIRECT_MX_PWL_EXP2_MODE == 15 ||
                         (C::HAO_DIRECT_MX_PWL_EXP2_MODE == 14 &&
                          (local_pair == sample_offset ||
                           local_pair == ((sample_offset + 2) & 3))) ||
                         (C::HAO_DIRECT_MX_PWL_EXP2_MODE == 13 &&
                          local_pair == sample_offset) ||
                         (C::HAO_DIRECT_MX_PWL_EXP2_MODE == 12 &&
                          (Pair == sample_offset ||
                           Pair == sample_offset + 8)) ||
                         (C::HAO_DIRECT_MX_PWL_EXP2_MODE == 20 &&
                          Quarter == 0 &&
                          (Pair == 6 || Pair == 14)))
                             : mode23_native);
                    if constexpr (
                        RebaseInterleavedScale && !native_sample) {
                        // Keep affine pairs raw until the interleaved max has
                        // selected the exact exponent. Native exp2 pairs issue
                        // early and are repaired by a power-of-two rescale.
                        return;
                    }
                    if constexpr (native_sample) {
                        const float2 log_value = __ffma2_rn(
                            raw_score,
                            score_scale,
                            quant_bias);
                        approximation = {
                            hao_direct_exp2_native_scalar(
                                log_value.x),
                            hao_direct_exp2_native_scalar(
                                log_value.y)};
                    } else if constexpr (
                        ForceAffine ||
                        (Quarter == 0 &&
                        (C::HAO_DIRECT_MX_PWL_EXP2_MODE == 19 ||
                         C::HAO_DIRECT_MX_PWL_EXP2_MODE == 20 ||
                         C::HAO_DIRECT_MX_PWL_EXP2_MODE == 23 ||
                         (C::HAO_DIRECT_MX_PWL_EXP2_MODE == 21 &&
                          packed_word == 0) ||
                         (C::HAO_DIRECT_MX_PWL_EXP2_MODE == 22 &&
                          Pair == 3)))) {
                        approximation = __ffma2_rn(
                            raw_score,
                            affine_raw_scale,
                            affine_raw_bias);
                    } else {
                        approximation = __ffma2_rn(
                            raw_score,
                            cubic_raw_a,
                            cubic_raw_b);
                        approximation = __ffma2_rn(
                            approximation,
                            raw_score,
                            cubic_raw_c);
                        approximation = __ffma2_rn(
                            approximation,
                            raw_score,
                            cubic_raw_d);
                    }
                } else {
                    const float2 hinge_anchor = {
                        fmaxf(raw_score.x, pwl_threshold),
                        fmaxf(raw_score.y, pwl_threshold)};
                    approximation = __ffma2_rn(
                        raw_score,
                        pwl_linear_scale,
                        pwl_linear_bias);
                    approximation = __ffma2_rn(
                        hinge_anchor,
                        pwl_hinge_scale,
                        approximation);
                }
                if constexpr (C::HAO_DIRECT_MX_PWL_RELU_PACK) {
                    (&scores.v0)[Pair] = approximation;
                } else {
                    (&scores.v0)[Pair] = {
                        fmaxf(approximation.x, 0.0f),
                        fmaxf(approximation.y, 0.0f)};
                }
            } else {
                const float2 log_value = __ffma2_rn(
                    (&scores.v0)[Pair],
                    score_scale,
                    quant_bias);
                constexpr bool use_bit_exp2 =
                    Quarter >= 1 && Quarter < 3 &&
                    (C::HAO_DIRECT_MX_BIT_EXP2_MASK &
                        (uint32_t{1} << Pair)) != 0;
                if constexpr (C::HAO_DIRECT_MX_PWL_EXP2) {
                // Continuous two-segment fit to positive E2M1 rounding
                // boundaries in log2 space.
                    const float2 hinge = {
                        fmaxf(log_value.x - 1.0f, 0.0f),
                        fmaxf(log_value.y - 1.0f, 0.0f)};
                    float2 approximation = __ffma2_rn(
                        log_value,
                        float2{PWL_LINEAR, PWL_LINEAR},
                        float2{PWL_BIAS, PWL_BIAS});
                    approximation = __ffma2_rn(
                        hinge,
                        float2{PWL_HINGE, PWL_HINGE},
                        approximation);
                    (&scores.v0)[Pair] = {
                        fmaxf(approximation.x, 0.0f),
                        fmaxf(approximation.y, 0.0f)};
                } else if constexpr (use_bit_exp2) {
                    constexpr float BIT_SCALE = 8388608.0f;
                    constexpr float BIT_BIAS = 1065353216.0f;
                    const int x_bits = __float2int_rn(
                        __fmaf_rn(
                            log_value.x, BIT_SCALE, BIT_BIAS));
                    const int y_bits = __float2int_rn(
                        __fmaf_rn(
                            log_value.y, BIT_SCALE, BIT_BIAS));
                    (&scores.v0)[Pair] = {
                        __int_as_float(x_bits),
                        __int_as_float(y_bits)};
                } else {
                    constexpr uint32_t ex2_emu_mask =
                        Quarter == 1
                            ? C::HAO_DIRECT_MX_EX2_Q1_MASK
                            : (Quarter == 2
                                   ? C::HAO_DIRECT_MX_EX2_Q2_MASK
                                   : (Quarter == 3
                                          ? C::HAO_DIRECT_MX_EX2_Q3_MASK
                                          : 0));
                    (&scores.v0)[Pair] = fp4pv_stage2_exp2_pair<
                        16,
                        ex2_emu_mask,
                        Quarter * 16 + Pair>(log_value);
                }
            }
        };
        auto pack_word = [](
            const float2 &v0,
            const float2 &v1,
            const float2 &v2,
            const float2 &v3,
            uint32_t &word
        ) {
            if constexpr (C::HAO_DIRECT_MX_PWL_RELU_PACK) {
                hao_direct_packed_float_to_e2m1_relu(
                    v0.x, v0.y,
                    v1.x, v1.y,
                    v2.x, v2.y,
                    v3.x, v3.y,
                    word);
            } else {
                fp4pv_packed_float_to_e2m1(
                    v0.x, v0.y,
                    v1.x, v1.y,
                    v2.x, v2.y,
                    v3.x, v3.y,
                    word);
            }
        };
        auto sum_group = [](
            const float2 &v0,
            const float2 &v1,
            const float2 &v2,
            const float2 &v3
        ) {
            return __fadd2_rn(
                __fadd2_rn(v0, v1),
                __fadd2_rn(v2, v3));
        };
        auto denom_sum_group = [&](
            const float2 &v0,
            const float2 &v1,
            const float2 &v2,
            const float2 &v3
        ) {
            if constexpr (
                C::HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE == 1) {
                const float2 p0 = {
                    fmaxf(v0.x, 0.0f), fmaxf(v0.y, 0.0f)};
                const float2 p1 = {
                    fmaxf(v1.x, 0.0f), fmaxf(v1.y, 0.0f)};
                const float2 p2 = {
                    fmaxf(v2.x, 0.0f), fmaxf(v2.y, 0.0f)};
                const float2 p3 = {
                    fmaxf(v3.x, 0.0f), fmaxf(v3.y, 0.0f)};
                return sum_group(p0, p1, p2, p3);
            } else {
                const float2 raw = sum_group(v0, v1, v2, v3);
                if constexpr (
                    C::HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE == 2) {
                    return float2{
                        fmaxf(raw.x, 0.0f),
                        fmaxf(raw.y, 0.0f)};
                } else {
                    return raw;
                }
            }
        };

        float ilp_unquantized_sum = 0.0f;
        float full_approx_sum = 0.0f;
        float2 full_approx_group0 = {0.0f, 0.0f};
        float2 full_approx_group1 = {0.0f, 0.0f};
        float2 full_approx_group2 = {0.0f, 0.0f};
        float2 full_approx_group3 = {0.0f, 0.0f};
        float interleaved_sampled_sum = 0.0f;
        float interleaved_sample0_sum = 0.0f;
        float interleaved_sample4_sum = 0.0f;
        float interleaved_sample8_sum = 0.0f;
        if constexpr (
            C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 8) {
            constexpr int sample_offset = Quarter & 3;
            constexpr bool early_mode23_native =
                C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE &&
                (C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_QUARTER_MASK &
                 (1 << Quarter));
            constexpr int early_native_lookahead =
                early_mode23_native
                    ? C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_LOOKAHEAD
                    : 0;
            static_assert(
                !early_mode23_native ||
                (early_native_lookahead >= 1 &&
                 early_native_lookahead <= 4));
            static_assert(
                C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_ORDER >= 0 &&
                C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_ORDER <= 2);
            static_assert(
                !early_mode23_native ||
                C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_ORDER == 0 ||
                early_native_lookahead == 4);
            if constexpr (InterleaveExactMax) {
                const auto &s = *interleaved_max_scores;
                if constexpr (C::HAO_DIRECT_MX_MAX3_WIDE_REDUCE) {
                    interleaved_p0 = hao_direct_max3_f32(
                        *interleaved_max_out, s.v0.x, s.v0.y);
                    interleaved_p1 = hao_direct_max3_f32(
                        s.v1.x, s.v1.y, s.v2.x);
                    interleaved_p2 = hao_direct_max3_f32(
                        s.v2.y, s.v3.x, s.v3.y);
                } else {
                    interleaved_p0 = hao_direct_max3_f32(
                        *interleaved_max_out, s.v0.x, s.v0.y);
                    interleaved_p1 = hao_direct_max3_f32(
                        *interleaved_max_out, s.v1.x, s.v1.y);
                    interleaved_p2 = hao_direct_max3_f32(
                        *interleaved_max_out, s.v2.x, s.v2.y);
                    interleaved_p3 = hao_direct_max3_f32(
                        *interleaved_max_out, s.v3.x, s.v3.y);
                }
            }
            if constexpr (C::HAO_DIRECT_MX_PWL_EXP2_MODE == 18) {
                transform_pair(
                    std::integral_constant<int, sample_offset>{});
                transform_pair(std::integral_constant<int, 4>{});
                transform_pair(std::integral_constant<
                    int, sample_offset + 8>{});
                transform_pair(std::integral_constant<int, 12>{});
            } else if constexpr (early_mode23_native) {
                transform_pair(
                    std::integral_constant<int, sample_offset>{});
                if constexpr (
                    early_native_lookahead == 4 &&
                    C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_ORDER == 1) {
                    transform_pair(std::integral_constant<
                        int, sample_offset + 8>{});
                    transform_pair(std::integral_constant<int, 4>{});
                    transform_pair(std::integral_constant<int, 12>{});
                } else if constexpr (
                    early_native_lookahead == 4 &&
                    C::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_ORDER == 2) {
                    transform_pair(std::integral_constant<int, 4>{});
                    transform_pair(std::integral_constant<int, 12>{});
                    transform_pair(std::integral_constant<
                        int, sample_offset + 8>{});
                } else if constexpr (early_native_lookahead >= 2) {
                    transform_pair(std::integral_constant<int, 4>{});
                    if constexpr (early_native_lookahead >= 3) {
                        transform_pair(std::integral_constant<
                            int, sample_offset + 8>{});
                    }
                    if constexpr (early_native_lookahead >= 4) {
                        transform_pair(std::integral_constant<int, 12>{});
                    }
                }
            }
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 17 ||
                (C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 19 &&
                 !early_mode23_native)) {
                transform_pair(
                    std::integral_constant<int, sample_offset>{});
                if constexpr (sample_offset != 0) {
                    transform_pair(std::integral_constant<int, 0>{});
                }
                if constexpr (sample_offset != 1) {
                    transform_pair(std::integral_constant<int, 1>{});
                }
                if constexpr (sample_offset != 2) {
                    transform_pair(std::integral_constant<int, 2>{});
                }
                if constexpr (sample_offset != 3) {
                    transform_pair(std::integral_constant<int, 3>{});
                }
            } else if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE != 18 &&
                !early_mode23_native) {
                transform_pair(std::integral_constant<int, 0>{});
                transform_pair(std::integral_constant<int, 1>{});
                transform_pair(std::integral_constant<int, 2>{});
                transform_pair(std::integral_constant<int, 3>{});
            } else {
                if constexpr (sample_offset != 0) {
                    transform_pair(std::integral_constant<int, 0>{});
                }
                if constexpr (sample_offset != 1) {
                    transform_pair(std::integral_constant<int, 1>{});
                }
                if constexpr (sample_offset != 2) {
                    transform_pair(std::integral_constant<int, 2>{});
                }
                if constexpr (sample_offset != 3) {
                    transform_pair(std::integral_constant<int, 3>{});
                }
            }
            const float2 sample0_raw =
                (&scores.v0)[sample_offset];
            float2 sample0;
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 12) {
                sample0 = sample0_raw;
            } else {
                sample0 = {
                    fmaxf(sample0_raw.x, 0.0f),
                    fmaxf(sample0_raw.y, 0.0f)};
            }
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 10) {
                interleaved_sample0_sum =
                    sample0.x + sample0.y;
            } else if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 11) {
                interleaved_sampled_sum =
                    (sample0.x + sample0.y) * 16.0f;
            } else if constexpr (
                C::HAO_DIRECT_MX_PWL_FOUR_SAMPLE_DENOM) {
                interleaved_sample0_sum =
                    sample0.x + sample0.y;
            }
            if constexpr (C::HAO_DIRECT_MX_FULL_APPROX_DENOM) {
                full_approx_group0 = denom_sum_group(
                    scores.v0, scores.v1, scores.v2, scores.v3);
            }
            if constexpr (!RebaseInterleavedScale) {
                pack_word(
                    scores.v0, scores.v1, scores.v2, scores.v3,
                    words[0]);
            }
            if constexpr (InterleaveExactMax) {
                const auto &s = *interleaved_max_scores;
                if constexpr (C::HAO_DIRECT_MX_MAX3_WIDE_REDUCE) {
                    interleaved_p3 = hao_direct_max3_f32(
                        s.v4.x, s.v4.y, s.v5.x);
                    interleaved_p4 = hao_direct_max3_f32(
                        s.v5.y, s.v6.x, s.v6.y);
                    interleaved_p5 = hao_direct_max3_f32(
                        s.v7.x, s.v7.y, s.v8.x);
                } else {
                    interleaved_p0 = hao_direct_max3_f32(
                        interleaved_p0, s.v4.x, s.v4.y);
                    interleaved_p1 = hao_direct_max3_f32(
                        interleaved_p1, s.v5.x, s.v5.y);
                    interleaved_p2 = hao_direct_max3_f32(
                        interleaved_p2, s.v6.x, s.v6.y);
                    interleaved_p3 = hao_direct_max3_f32(
                        interleaved_p3, s.v7.x, s.v7.y);
                }
            }
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE != 18 &&
                (!early_mode23_native || early_native_lookahead < 2)) {
                transform_pair(std::integral_constant<int, 4>{});
            }
            transform_pair(std::integral_constant<int, 5>{});
            transform_pair(std::integral_constant<int, 6>{});
            transform_pair(std::integral_constant<int, 7>{});
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 10 ||
                C::HAO_DIRECT_MX_PWL_FOUR_SAMPLE_DENOM) {
                const float2 sample4_raw =
                    (&scores.v0)[sample_offset + 4];
                interleaved_sample4_sum =
                    fmaxf(sample4_raw.x, 0.0f) +
                    fmaxf(sample4_raw.y, 0.0f);
            }
            if constexpr (C::HAO_DIRECT_MX_FULL_APPROX_DENOM) {
                full_approx_group1 = denom_sum_group(
                    scores.v4, scores.v5, scores.v6, scores.v7);
            }
            if constexpr (!RebaseInterleavedScale) {
                pack_word(
                    scores.v4, scores.v5, scores.v6, scores.v7,
                    words[1]);
            }
            if constexpr (InterleaveExactMax) {
                const auto &s = *interleaved_max_scores;
                if constexpr (C::HAO_DIRECT_MX_MAX3_WIDE_REDUCE) {
                    interleaved_p6 = hao_direct_max3_f32(
                        s.v8.y, s.v9.x, s.v9.y);
                    interleaved_p7 = hao_direct_max3_f32(
                        s.v10.x, s.v10.y, s.v11.x);
                    interleaved_p8 = hao_direct_max3_f32(
                        s.v11.y, s.v12.x, s.v12.y);
                } else {
                    interleaved_p0 = hao_direct_max3_f32(
                        interleaved_p0, s.v8.x, s.v8.y);
                    interleaved_p1 = hao_direct_max3_f32(
                        interleaved_p1, s.v9.x, s.v9.y);
                    interleaved_p2 = hao_direct_max3_f32(
                        interleaved_p2, s.v10.x, s.v10.y);
                    interleaved_p3 = hao_direct_max3_f32(
                        interleaved_p3, s.v11.x, s.v11.y);
                }
            }
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 17 ||
                (C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 19 &&
                 !early_mode23_native)) {
                transform_pair(std::integral_constant<
                    int, sample_offset + 8>{});
                if constexpr (sample_offset != 0) {
                    transform_pair(std::integral_constant<int, 8>{});
                }
                if constexpr (sample_offset != 1) {
                    transform_pair(std::integral_constant<int, 9>{});
                }
                if constexpr (sample_offset != 2) {
                    transform_pair(std::integral_constant<int, 10>{});
                }
                if constexpr (sample_offset != 3) {
                    transform_pair(std::integral_constant<int, 11>{});
                }
            } else if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE != 18 &&
                !early_mode23_native) {
                transform_pair(std::integral_constant<int, 8>{});
                transform_pair(std::integral_constant<int, 9>{});
                transform_pair(std::integral_constant<int, 10>{});
                transform_pair(std::integral_constant<int, 11>{});
            } else {
                if constexpr (
                    early_mode23_native && early_native_lookahead < 3) {
                    transform_pair(std::integral_constant<
                        int, sample_offset + 8>{});
                }
                if constexpr (sample_offset != 0) {
                    transform_pair(std::integral_constant<int, 8>{});
                }
                if constexpr (sample_offset != 1) {
                    transform_pair(std::integral_constant<int, 9>{});
                }
                if constexpr (sample_offset != 2) {
                    transform_pair(std::integral_constant<int, 10>{});
                }
                if constexpr (sample_offset != 3) {
                    transform_pair(std::integral_constant<int, 11>{});
                }
            }
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE != 11) {
                const float2 sample8_raw =
                    (&scores.v0)[sample_offset + 8];
                float2 sample8;
                if constexpr (
                    C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 12) {
                    sample8 = sample8_raw;
                } else {
                    sample8 = {
                        fmaxf(sample8_raw.x, 0.0f),
                        fmaxf(sample8_raw.y, 0.0f)};
                }
                if constexpr (
                    C::HAO_DIRECT_MX_PWL_EXP2_MODE == 10) {
                    interleaved_sample8_sum =
                        sample8.x + sample8.y;
                } else if constexpr (
                    C::HAO_DIRECT_MX_PWL_FOUR_SAMPLE_DENOM) {
                    interleaved_sample8_sum =
                        sample8.x + sample8.y;
                } else {
                    const float2 sample =
                        __fadd2_rn(sample0, sample8);
                    interleaved_sampled_sum =
                        (sample.x + sample.y) * 8.0f;
                }
            }
            if constexpr (C::HAO_DIRECT_MX_FULL_APPROX_DENOM) {
                full_approx_group2 = denom_sum_group(
                    scores.v8, scores.v9, scores.v10, scores.v11);
            }
            if constexpr (!RebaseInterleavedScale) {
                pack_word(
                    scores.v8, scores.v9, scores.v10, scores.v11,
                    words[2]);
            }
            if constexpr (InterleaveExactMax) {
                const auto &s = *interleaved_max_scores;
                if constexpr (C::HAO_DIRECT_MX_MAX3_WIDE_REDUCE) {
                    interleaved_p9 = hao_direct_max3_f32(
                        s.v13.x, s.v13.y, s.v14.x);
                    interleaved_p10 = hao_direct_max3_f32(
                        s.v14.y, s.v15.x, s.v15.y);
                    interleaved_p0 = hao_direct_max3_f32(
                        interleaved_p0,
                        interleaved_p1,
                        interleaved_p2);
                    interleaved_p3 = hao_direct_max3_f32(
                        interleaved_p3,
                        interleaved_p4,
                        interleaved_p5);
                    interleaved_p6 = hao_direct_max3_f32(
                        interleaved_p6,
                        interleaved_p7,
                        interleaved_p8);
                    interleaved_p0 = hao_direct_max3_f32(
                        interleaved_p0,
                        interleaved_p3,
                        interleaved_p6);
                    *interleaved_max_out = hao_direct_max3_f32(
                        interleaved_p0,
                        interleaved_p9,
                        interleaved_p10);
                } else {
                    interleaved_p0 = hao_direct_max3_f32(
                        interleaved_p0, s.v12.x, s.v12.y);
                    interleaved_p1 = hao_direct_max3_f32(
                        interleaved_p1, s.v13.x, s.v13.y);
                    interleaved_p2 = hao_direct_max3_f32(
                        interleaved_p2, s.v14.x, s.v14.y);
                    interleaved_p3 = hao_direct_max3_f32(
                        interleaved_p3, s.v15.x, s.v15.y);
                    interleaved_p0 = hao_direct_max3_f32(
                        interleaved_p0,
                        interleaved_p1,
                        interleaved_p2);
                    *interleaved_max_out = hao_direct_max3_f32(
                        *interleaved_max_out,
                        interleaved_p0,
                        interleaved_p3);
                }
            }
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE != 18 &&
                (!early_mode23_native || early_native_lookahead < 4)) {
                transform_pair(std::integral_constant<int, 12>{});
            }
            transform_pair(std::integral_constant<int, 13>{});
            transform_pair(std::integral_constant<int, 14>{});
            transform_pair(std::integral_constant<int, 15>{});
            if constexpr (
                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 10 ||
                C::HAO_DIRECT_MX_PWL_FOUR_SAMPLE_DENOM) {
                const float2 sample12_raw =
                    (&scores.v0)[sample_offset + 12];
                const float interleaved_sample12_sum =
                    fmaxf(sample12_raw.x, 0.0f) +
                    fmaxf(sample12_raw.y, 0.0f);
                interleaved_sampled_sum =
                    ((interleaved_sample0_sum +
                      interleaved_sample4_sum) +
                     (interleaved_sample8_sum +
                      interleaved_sample12_sum)) * 4.0f;
            }
            if constexpr (C::HAO_DIRECT_MX_FULL_APPROX_DENOM) {
                full_approx_group3 = denom_sum_group(
                    scores.v12, scores.v13, scores.v14, scores.v15);
            }
            if constexpr (!RebaseInterleavedScale) {
                pack_word(
                    scores.v12, scores.v13, scores.v14, scores.v15,
                    words[3]);
            }

            if constexpr (C::HAO_DIRECT_MX_FULL_APPROX_DENOM) {
                const float2 half0_raw = __fadd2_rn(
                    full_approx_group0, full_approx_group1);
                const float2 half1_raw = __fadd2_rn(
                    full_approx_group2, full_approx_group3);
                const float2 half0 =
                    C::HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE == 3
                        ? float2{
                              fmaxf(half0_raw.x, 0.0f),
                              fmaxf(half0_raw.y, 0.0f)}
                        : half0_raw;
                const float2 half1 =
                    C::HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE == 3
                        ? float2{
                              fmaxf(half1_raw.x, 0.0f),
                              fmaxf(half1_raw.y, 0.0f)}
                        : half1_raw;
                const float2 total = __fadd2_rn(half0, half1);
                const float raw_sum = total.x + total.y;
                full_approx_sum =
                    C::HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE == 4
                        ? fmaxf(raw_sum, 0.0f)
                        : raw_sum;
            }
        } else if constexpr (C::HAO_DIRECT_P_ILP_PIPELINE) {
            transform_pair(std::integral_constant<int, 0>{});
            transform_pair(std::integral_constant<int, 1>{});
            transform_pair(std::integral_constant<int, 2>{});
            transform_pair(std::integral_constant<int, 3>{});

            transform_pair(std::integral_constant<int, 4>{});
            transform_pair(std::integral_constant<int, 5>{});
            transform_pair(std::integral_constant<int, 6>{});
            transform_pair(std::integral_constant<int, 7>{});

            pack_word(
                scores.v0, scores.v1, scores.v2, scores.v3,
                words[0]);
            const float2 group0 = sum_group(
                scores.v0, scores.v1, scores.v2, scores.v3);

            transform_pair(std::integral_constant<int, 8>{});
            transform_pair(std::integral_constant<int, 9>{});
            transform_pair(std::integral_constant<int, 10>{});
            transform_pair(std::integral_constant<int, 11>{});

            pack_word(
                scores.v4, scores.v5, scores.v6, scores.v7,
                words[1]);
            const float2 group1 = sum_group(
                scores.v4, scores.v5, scores.v6, scores.v7);
            const float2 half0 = __fadd2_rn(group0, group1);

            transform_pair(std::integral_constant<int, 12>{});
            transform_pair(std::integral_constant<int, 13>{});
            transform_pair(std::integral_constant<int, 14>{});
            transform_pair(std::integral_constant<int, 15>{});

            pack_word(
                scores.v8, scores.v9, scores.v10, scores.v11,
                words[2]);
            const float2 group2 = sum_group(
                scores.v8, scores.v9, scores.v10, scores.v11);
            pack_word(
                scores.v12, scores.v13, scores.v14, scores.v15,
                words[3]);
            const float2 group3 = sum_group(
                scores.v12, scores.v13, scores.v14, scores.v15);
            const float2 half1 = __fadd2_rn(group2, group3);
            const float half_sum0 = half0.x + half0.y;
            const float half_sum1 = half1.x + half1.y;
            ilp_unquantized_sum = half_sum0 + half_sum1;
        } else {
            fp4pv_stage2_static_for<0, 16>(transform_pair);
            pack_word(
                scores.v0, scores.v1, scores.v2, scores.v3,
                words[0]);
            pack_word(
                scores.v4, scores.v5, scores.v6, scores.v7,
                words[1]);
            pack_word(
                scores.v8, scores.v9, scores.v10, scores.v11,
                words[2]);
            pack_word(
                scores.v12, scores.v13, scores.v14, scores.v15,
                words[3]);
        }

        if constexpr (RebaseInterleavedScale) {
            const float exact_group_max =
                C::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2 != 0
                    ? *interleaved_max_out * scale_log2 -
                          static_cast<float>(
                              C::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2) +
                          sample_bias
                    : PreparedAnchorAffine
                    ? __fmaf_rn(
                          *interleaved_max_out,
                          scale_log2,
                          prepared_quant_bias_base - LOG2_FP4_MAX +
                              sample_bias)
                    : (*interleaved_max_out - row_max) * scale_log2 +
                          sample_bias;
            const uint8_t exact_e8m0 =
                hao_direct_select_mxfp4_e8m0<C>(exact_group_max);
            const int scale_delta =
                static_cast<int>(e8m0) -
                static_cast<int>(exact_e8m0);
            const float rebase_scalar =
                hao_direct_pow2_integer_scalar(scale_delta);
            const float2 rebase = {rebase_scalar, rebase_scalar};
            const float exact_affine_bias_scalar = __fmaf_rn(
                AFFINE_A,
                quant_bias_scalar + static_cast<float>(scale_delta),
                AFFINE_B);
            const float2 exact_affine_bias = {
                exact_affine_bias_scalar, exact_affine_bias_scalar};
            fp4pv_stage2_static_for<0, 16>([&](auto pair_tag) {
                constexpr int Pair = decltype(pair_tag)::value;
                constexpr int sample_offset = Quarter & 3;
                constexpr bool mode23_base_native =
                    Pair == sample_offset ||
                    Pair == sample_offset + 8 ||
                    Pair == 4 || Pair == 12;
                constexpr bool mode23_opposite_native =
                    Pair == ((sample_offset + 2) & 3) ||
                    Pair == 4 + 2 ||
                    Pair == 8 + ((sample_offset + 2) & 3) ||
                    Pair == 12 + 2;
                constexpr bool mode23_native =
                    (C::HAO_DIRECT_MX_MODE23_NATIVE_QUARTER_MASK &
                     (1u << Quarter)) != 0 &&
                    (C::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY >= 4 ||
                     (C::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY >= 1 &&
                      mode23_base_native) ||
                     (C::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY >= 2 &&
                      mode23_opposite_native) ||
                     (C::HAO_DIRECT_MX_Q1_SELF_MAX == 4 &&
                      Quarter == 1 &&
                      (Pair == sample_offset || Pair == 4)));
                if constexpr (mode23_native) {
                    (&scores.v0)[Pair] = __fmul2_rn(
                        (&scores.v0)[Pair], rebase);
                } else {
                    const float2 approximation = __ffma2_rn(
                        (&scores.v0)[Pair],
                        affine_raw_scale,
                        exact_affine_bias);
                    if constexpr (C::HAO_DIRECT_MX_PWL_RELU_PACK) {
                        (&scores.v0)[Pair] = approximation;
                    } else {
                        (&scores.v0)[Pair] = {
                            fmaxf(approximation.x, 0.0f),
                            fmaxf(approximation.y, 0.0f)};
                    }
                }
            });
            pack_word(
                scores.v0, scores.v1, scores.v2, scores.v3,
                words[0]);
            pack_word(
                scores.v4, scores.v5, scores.v6, scores.v7,
                words[1]);
            pack_word(
                scores.v8, scores.v9, scores.v10, scores.v11,
                words[2]);
            pack_word(
                scores.v12, scores.v13, scores.v14, scores.v15,
                words[3]);
            if constexpr (!C::HAO_DIRECT_DENOM_CORR_WG) {
                constexpr int sample_offset = Quarter & 3;
                auto positive_pair = [&](int pair) {
                    const float2 value = (&scores.v0)[pair];
                    return float2{
                        fmaxf(value.x, 0.0f),
                        fmaxf(value.y, 0.0f)};
                };
                const float2 sample0 = positive_pair(sample_offset);
                const float2 sample8 = positive_pair(sample_offset + 8);
                if constexpr (
                    C::HAO_DIRECT_MX_PWL_FOUR_SAMPLE_DENOM) {
                    const float2 sample4 = positive_pair(4);
                    const float2 sample12 = positive_pair(12);
                    const float2 sample04 =
                        __fadd2_rn(sample0, sample4);
                    const float2 sample812 =
                        __fadd2_rn(sample8, sample12);
                    const float2 sample =
                        __fadd2_rn(sample04, sample812);
                    interleaved_sampled_sum =
                        (sample.x + sample.y) * 4.0f;
                } else {
                    const float2 sample =
                        __fadd2_rn(sample0, sample8);
                    interleaved_sampled_sum =
                        (sample.x + sample.y) * 8.0f;
                }
            }
            e8m0 = exact_e8m0;
        }

        if constexpr (C::HAO_DIRECT_MX_SKIP_ZERO_SCALE_MASK) {
            // The E8M0 scale determines whether this payload contributes.
        } else if constexpr (C::HAO_DIRECT_MX_SCALE_SELECT == 0) {
            if (e8m0 == 0) {
                words[0] = 0;
                words[1] = 0;
                words[2] = 0;
                words[3] = 0;
            }
        } else {
            const uint32_t valid_mask =
                0u - static_cast<uint32_t>(e8m0 != 0);
            words[0] &= valid_mask;
            words[1] &= valid_mask;
            words[2] &= valid_mask;
            words[3] &= valid_mask;
        }

        if constexpr (C::HAO_DIRECT_FAKE_DENOM_CEILING) {
            scaled_sum = 32.0f;
        } else if constexpr (C::HAO_DIRECT_MX_QUANTIZED_DENOM) {
            int quantized_sum_x2;
            if constexpr (
                C::HAO_DIRECT_MX_DENOM_WORDS == 4 &&
                C::HAO_DIRECT_MX_DENOM_DECODE_MODE == 1) {
                quantized_sum_x2 =
                    hao_direct_qsum_x2_packed_bytes(words);
            } else {
                quantized_sum_x2 = 0;
                constexpr int word0 = Quarter & 3;
                quantized_sum_x2 =
                    fp4pv_stage2_qsum_weights_from_word(
                        words[word0], quantized_sum_x2);
                if constexpr (C::HAO_DIRECT_MX_DENOM_WORDS >= 2) {
                    constexpr int word1 = (word0 + 2) & 3;
                    quantized_sum_x2 =
                        fp4pv_stage2_qsum_weights_from_word(
                            words[word1], quantized_sum_x2);
                }
                if constexpr (C::HAO_DIRECT_MX_DENOM_WORDS >= 3) {
                    constexpr int word2 = (word0 + 1) & 3;
                    quantized_sum_x2 =
                        fp4pv_stage2_qsum_weights_from_word(
                            words[word2], quantized_sum_x2);
                }
                if constexpr (C::HAO_DIRECT_MX_DENOM_WORDS >= 4) {
                    constexpr int word3 = (word0 + 3) & 3;
                    quantized_sum_x2 =
                        fp4pv_stage2_qsum_weights_from_word(
                            words[word3], quantized_sum_x2);
                }
            }
            scaled_sum =
                static_cast<float>(quantized_sum_x2) *
                (2.0f / static_cast<float>(
                    C::HAO_DIRECT_MX_DENOM_WORDS));
        } else if constexpr (C::HAO_DIRECT_MX_FULL_APPROX_DENOM) {
            scaled_sum = full_approx_sum;
        } else if constexpr (
            C::HAO_DIRECT_MX_PWL_EXP2_MODE == 4) {
            constexpr int sample_offset = Quarter & 1;
            auto positive_sample_pair = [&](auto pair_tag) {
                constexpr int Pair = decltype(pair_tag)::value;
                const float2 value = (&scores.v0)[Pair];
                return float2{
                    fmaxf(value.x, 0.0f),
                    fmaxf(value.y, 0.0f)};
            };
            const float2 sample0 = positive_sample_pair(
                std::integral_constant<int, sample_offset>{});
            const float2 sample2 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 2>{});
            const float2 sample4 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 4>{});
            const float2 sample6 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 6>{});
            const float2 sample8 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 8>{});
            const float2 sample10 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 10>{});
            const float2 sample12 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 12>{});
            const float2 sample14 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 14>{});
            const float2 sample02 = __fadd2_rn(sample0, sample2);
            const float2 sample46 = __fadd2_rn(sample4, sample6);
            const float2 sample810 = __fadd2_rn(sample8, sample10);
            const float2 sample1214 = __fadd2_rn(sample12, sample14);
            const float2 sample06 = __fadd2_rn(sample02, sample46);
            const float2 sample814 = __fadd2_rn(sample810, sample1214);
            const float2 sample =
                __fadd2_rn(sample06, sample814);
            scaled_sum = (sample.x + sample.y) * 2.0f;
        } else if constexpr (
            C::HAO_DIRECT_MX_PWL_EXP2_MODE == 5 ||
            C::HAO_DIRECT_MX_PWL_EXP2_MODE == 7) {
            constexpr int sample_offset = Quarter & 3;
            auto positive_sample_pair = [&](auto pair_tag) {
                constexpr int Pair = decltype(pair_tag)::value;
                const float2 value = (&scores.v0)[Pair];
                return float2{
                    fmaxf(value.x, 0.0f),
                    fmaxf(value.y, 0.0f)};
            };
            const float2 sample0 = positive_sample_pair(
                std::integral_constant<int, sample_offset>{});
            const float2 sample4 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 4>{});
            const float2 sample8 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 8>{});
            const float2 sample12 = positive_sample_pair(
                std::integral_constant<int, sample_offset + 12>{});
            const float2 sample04 = __fadd2_rn(sample0, sample4);
            const float2 sample812 = __fadd2_rn(sample8, sample12);
            const float2 sample = __fadd2_rn(sample04, sample812);
            scaled_sum = (sample.x + sample.y) * 4.0f;
        } else if constexpr (
            C::HAO_DIRECT_MX_PWL_EXP2_MODE >= 8) {
            scaled_sum = interleaved_sampled_sum;
        } else if constexpr (
            C::HAO_DIRECT_MX_PWL_EXP2_MODE == 6) {
            constexpr int sample_word = Quarter & 3;
            const int sample_sum_x2 =
                fp4pv_stage2_qsum_weights_from_word(
                    words[sample_word], 0);
            scaled_sum =
                static_cast<float>(sample_sum_x2) * 2.0f;
        } else if constexpr (C::HAO_DIRECT_P_ILP_PIPELINE) {
            scaled_sum = ilp_unquantized_sum;
        } else if constexpr (
            C::HAO_DIRECT_MX_DENOM_SAMPLE_STRIDE == 2) {
            constexpr int sample_offset = Quarter & 1;
            const float2 sample02 =
                __fadd2_rn(
                    (&scores.v0)[sample_offset],
                    (&scores.v0)[sample_offset + 2]);
            const float2 sample46 =
                __fadd2_rn(
                    (&scores.v0)[sample_offset + 4],
                    (&scores.v0)[sample_offset + 6]);
            const float2 sample810 =
                __fadd2_rn(
                    (&scores.v0)[sample_offset + 8],
                    (&scores.v0)[sample_offset + 10]);
            const float2 sample1214 =
                __fadd2_rn(
                    (&scores.v0)[sample_offset + 12],
                    (&scores.v0)[sample_offset + 14]);
            const float2 sample06 =
                __fadd2_rn(sample02, sample46);
            const float2 sample814 =
                __fadd2_rn(sample810, sample1214);
            const float2 sample =
                __fadd2_rn(sample06, sample814);
            scaled_sum = (sample.x + sample.y) * 2.0f;
        } else if constexpr (
            C::HAO_DIRECT_MX_DENOM_SAMPLE_STRIDE == 4) {
            const float2 sample04 =
                __fadd2_rn(scores.v0, scores.v4);
            const float2 sample812 =
                __fadd2_rn(scores.v8, scores.v12);
            const float2 sample =
                __fadd2_rn(sample04, sample812);
            scaled_sum = (sample.x + sample.y) * 4.0f;
        } else {
            scaled_sum =
                hao_direct_sum8_pairs(
                    scores.v0, scores.v1, scores.v2, scores.v3,
                    scores.v4, scores.v5, scores.v6, scores.v7) +
                hao_direct_sum8_pairs(
                    scores.v8, scores.v9, scores.v10, scores.v11,
                    scores.v12, scores.v13, scores.v14, scores.v15);
        }
    }

    const int row =
        reader_warp * 32 + warp::laneid();
    uint8_t stored_e8m0 = e8m0;
    if constexpr (C::HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2 != 0) {
        constexpr int SHIFT =
            C::HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2;
        stored_e8m0 = e8m0 > SHIFT
            ? static_cast<uint8_t>(e8m0 - SHIFT)
            : uint8_t{0};
    }
    if constexpr (CaptureScaleByte) {
        *stored_e8m0_out = stored_e8m0;
    } else if constexpr (C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH) {
        hao_direct_store_mxfp4_scale_byte_linear(
            p_scale, row, Quarter, stored_e8m0);
    } else {
        fp4pv_store_mxfp4_scale_byte_swizzled(
            p_scale, row, Quarter, stored_e8m0);
    }
    if constexpr (CapturePackedWords) {
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            packed_words_out[word] = words[word];
        }
    }
    hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
        p,
        reader_warp,
        words[0],
        words[1],
        words[2],
        words[3]);
    if constexpr (C::HAO_DIRECT_DENOM_CORR_WG) {
        // Exact represented-sum correction owns normalization. The softmax
        // reader only needs to publish P and its scale in this mode.
        return 0.0f;
    }
    if constexpr (C::HAO_DIRECT_FAKE_DENOM_CEILING) {
        return 32.0f;
    }
    if constexpr (C::HAO_DIRECT_MX_SCALE_SELECT == 0 &&
                  !C::HAO_DIRECT_MX_DIRECT_LOG_CODE) {
        if (e8m0 == 0) {
            return 0.0f;
        }
    }
    const float group_scale =
        __uint_as_float(static_cast<uint32_t>(stored_e8m0) << 23) *
        (1.0f / 6.0f);
    return group_scale * scaled_sum;
}

template <typename C>
__device__ __forceinline__ void
hao_direct_pack_mxfp4_local_log_codes(
    score_quarter_regs &scores,
    float scale_log2,
    float local_max,
    uint32_t (&words)[4]
) {
    constexpr float LOG2_FP4_MAX = 2.584962500721156f;
    const float2 score_scale = {scale_log2, scale_log2};
    const float2 quant_bias = {
        -local_max * scale_log2 + LOG2_FP4_MAX,
        -local_max * scale_log2 + LOG2_FP4_MAX};
    words[0] = 0;
    words[1] = 0;
    words[2] = 0;
    words[3] = 0;
    hao_direct_pack_mxfp4_log_codes<C>(
        scores,
        score_scale,
        quant_bias,
        words);
}

template <int Quarter, typename C, typename P>
__device__ __forceinline__ int
hao_direct_store_mxfp4_local_log_quarter(
    P &p,
    int reader_warp,
    score_quarter_regs &scores,
    float scale_log2,
    float local_max
) {
    uint32_t words[4];
    hao_direct_pack_mxfp4_local_log_codes<C>(
        scores, scale_log2, local_max, words);
    hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
        p,
        reader_warp,
        words[0],
        words[1],
        words[2],
        words[3]);
    int quantized_sum_x2 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        quantized_sum_x2 =
            fp4pv_stage2_qsum_weights_from_word(
                words[i], quantized_sum_x2);
    }
    return quantized_sum_x2;
}

template <int Quarter, typename C, typename P>
__device__ __forceinline__ float
hao_direct_store_mxfp4_local_native_quarter(
    P &p,
    int reader_warp,
    score_quarter_regs &scores,
    float scale_log2,
    float local_max,
    int &absolute_e8m0
) {
    constexpr float LOG2_FP4_MAX = 2.584962500721156f;
    const uint8_t e8m0 =
        hao_direct_select_mxfp4_e8m0<C>(
            local_max * scale_log2);
    absolute_e8m0 = static_cast<int>(e8m0);
    int working_e8m0 = absolute_e8m0;
    asm(
        "max.s32 %0, %0, 1;"
        : "+r"(working_e8m0));
    const int scale_exponent = working_e8m0 - 127;
    const float2 score_scale = {scale_log2, scale_log2};
    const float2 quant_bias = {
        LOG2_FP4_MAX - static_cast<float>(scale_exponent),
        LOG2_FP4_MAX - static_cast<float>(scale_exponent)};

    fp4pv_stage2_static_for<0, 16>([&](auto pair_tag) {
        constexpr int Pair = decltype(pair_tag)::value;
        constexpr uint32_t ex2_emu_mask =
            Quarter >= 1 && Quarter < 3
                ? C::ONLINE_STAGE2_EX2_EMU_MASK
                : 0;
        const float2 local_log = __ffma2_rn(
            (&scores.v0)[Pair],
            score_scale,
            quant_bias);
        (&scores.v0)[Pair] = fp4pv_stage2_exp2_pair<
            16,
            ex2_emu_mask,
            Quarter * 16 + Pair>(local_log);
    });

    const float local_scaled_sum =
        hao_direct_sum8_pairs(
            scores.v0, scores.v1, scores.v2, scores.v3,
            scores.v4, scores.v5, scores.v6, scores.v7) +
        hao_direct_sum8_pairs(
            scores.v8, scores.v9, scores.v10, scores.v11,
            scores.v12, scores.v13, scores.v14, scores.v15);
    uint32_t words[4];
    fp4pv_packed_float_to_e2m1(
        scores.v0.x, scores.v0.y,
        scores.v1.x, scores.v1.y,
        scores.v2.x, scores.v2.y,
        scores.v3.x, scores.v3.y,
        words[0]);
    fp4pv_packed_float_to_e2m1(
        scores.v4.x, scores.v4.y,
        scores.v5.x, scores.v5.y,
        scores.v6.x, scores.v6.y,
        scores.v7.x, scores.v7.y,
        words[1]);
    fp4pv_packed_float_to_e2m1(
        scores.v8.x, scores.v8.y,
        scores.v9.x, scores.v9.y,
        scores.v10.x, scores.v10.y,
        scores.v11.x, scores.v11.y,
        words[2]);
    fp4pv_packed_float_to_e2m1(
        scores.v12.x, scores.v12.y,
        scores.v13.x, scores.v13.y,
        scores.v14.x, scores.v14.y,
        scores.v15.x, scores.v15.y,
        words[3]);
    const uint32_t valid_mask =
        0u - static_cast<uint32_t>(e8m0 != 0);
    words[0] &= valid_mask;
    words[1] &= valid_mask;
    words[2] &= valid_mask;
    words[3] &= valid_mask;
    hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
        p,
        reader_warp,
        words[0],
        words[1],
        words[2],
        words[3]);
    return e8m0 != 0 ? local_scaled_sum : 0.0f;
}

template <int Quarter, typename C, typename PScale>
__device__ __forceinline__ float
hao_direct_store_mxfp4_scale_and_quantized_sum(
    PScale &p_scale,
    int reader_warp,
    float local_max,
    float row_max,
    float scale_log2,
    int quantized_sum_x2
) {
    const float log2_group_amplitude =
        (local_max - row_max) * scale_log2;
    const uint8_t e8m0 =
        hao_direct_select_mxfp4_e8m0<C>(
            log2_group_amplitude);
    const int row =
        reader_warp * 32 + warp::laneid();
    fp4pv_store_mxfp4_scale_byte_swizzled(
        p_scale, row, Quarter, e8m0);
    const float group_amplitude =
        __uint_as_float(static_cast<uint32_t>(e8m0) << 23);
    return group_amplitude *
        static_cast<float>(quantized_sum_x2) *
        (1.0f / 12.0f);
}

template <int Quarter, typename C, typename PScale>
__device__ __forceinline__ float
hao_direct_store_mxfp4_scale_and_local_sum(
    PScale &p_scale,
    int reader_warp,
    int absolute_e8m0,
    int row_anchor_log2,
    float local_scaled_sum
) {
    int relative_e8m0 = absolute_e8m0 - row_anchor_log2;
    asm(
        "{\n\t"
        ".reg .pred valid;\n\t"
        "setp.ne.s32 valid, %1, 0;\n\t"
        "max.s32 %0, %0, 0;\n\t"
        "min.s32 %0, %0, 254;\n\t"
        "selp.s32 %0, %0, 0, valid;\n\t"
        "}\n"
        : "+r"(relative_e8m0)
        : "r"(absolute_e8m0));
    const uint8_t e8m0 =
        static_cast<uint8_t>(relative_e8m0);
    const int row =
        reader_warp * 32 + warp::laneid();
    fp4pv_store_mxfp4_scale_byte_swizzled(
        p_scale, row, Quarter, e8m0);
    const float group_amplitude =
        __uint_as_float(static_cast<uint32_t>(e8m0) << 23);
    return group_amplitude * local_scaled_sum * (1.0f / 6.0f);
}

template <
    int Quarter,
    typename C,
    bool ForceAffine = false,
    bool CapturePackedWords = false,
    bool PreparedAnchorAffine = false,
    bool CaptureScaleByte = false,
    bool InterleaveExactMax = false,
    bool RebaseInterleavedScale = false,
    typename P,
    typename PScale>
__device__ __forceinline__ float
hao_direct_transform_pack_mxfp4_quarter(
    P &p,
    PScale &p_scale,
    int reader_warp,
    score_quarter_regs &scores,
    float row_max,
    float scale_log2,
    float raw_group_max,
    uint32_t *packed_words_out = nullptr,
    float prepared_quant_bias_base = 0.0f,
    float prepared_affine_scale = 0.0f,
    uint8_t *stored_e8m0_out = nullptr,
    uint8_t *pair_e8m0_io = nullptr,
    bool reuse_pair_scale_runtime = true,
    const score_quarter_regs *interleaved_max_scores = nullptr,
    float *interleaved_max_out = nullptr
) {
    static_assert(
        sizeof(score_quarter_regs) == 16 * sizeof(float2));
    if constexpr (C::HAO_DIRECT_MX_LOG2_QUANT) {
        return hao_direct_transform_pack_mxfp4_quarter_log2<
            Quarter,
            C,
            ForceAffine,
            CapturePackedWords,
            PreparedAnchorAffine,
            CaptureScaleByte,
            InterleaveExactMax,
            RebaseInterleavedScale>(
                p,
                p_scale,
                reader_warp,
                scores,
                row_max,
                scale_log2,
                raw_group_max,
                packed_words_out,
                prepared_quant_bias_base,
                prepared_affine_scale,
                stored_e8m0_out,
                pair_e8m0_io,
                reuse_pair_scale_runtime,
                interleaved_max_scores,
                interleaved_max_out);
    }

    scores.v0 = hao_direct_exp2_nvfp4_pair<Quarter, 0, C>(
        scores.v0, row_max, scale_log2);
    scores.v1 = hao_direct_exp2_nvfp4_pair<Quarter, 1, C>(
        scores.v1, row_max, scale_log2);
    scores.v2 = hao_direct_exp2_nvfp4_pair<Quarter, 2, C>(
        scores.v2, row_max, scale_log2);
    scores.v3 = hao_direct_exp2_nvfp4_pair<Quarter, 3, C>(
        scores.v3, row_max, scale_log2);
    scores.v4 = hao_direct_exp2_nvfp4_pair<Quarter, 4, C>(
        scores.v4, row_max, scale_log2);
    scores.v5 = hao_direct_exp2_nvfp4_pair<Quarter, 5, C>(
        scores.v5, row_max, scale_log2);
    scores.v6 = hao_direct_exp2_nvfp4_pair<Quarter, 6, C>(
        scores.v6, row_max, scale_log2);
    scores.v7 = hao_direct_exp2_nvfp4_pair<Quarter, 7, C>(
        scores.v7, row_max, scale_log2);
    scores.v8 = hao_direct_exp2_nvfp4_pair<Quarter, 8, C>(
        scores.v8, row_max, scale_log2);
    scores.v9 = hao_direct_exp2_nvfp4_pair<Quarter, 9, C>(
        scores.v9, row_max, scale_log2);
    scores.v10 = hao_direct_exp2_nvfp4_pair<Quarter, 10, C>(
        scores.v10, row_max, scale_log2);
    scores.v11 = hao_direct_exp2_nvfp4_pair<Quarter, 11, C>(
        scores.v11, row_max, scale_log2);
    scores.v12 = hao_direct_exp2_nvfp4_pair<Quarter, 12, C>(
        scores.v12, row_max, scale_log2);
    scores.v13 = hao_direct_exp2_nvfp4_pair<Quarter, 13, C>(
        scores.v13, row_max, scale_log2);
    scores.v14 = hao_direct_exp2_nvfp4_pair<Quarter, 14, C>(
        scores.v14, row_max, scale_log2);
    scores.v15 = hao_direct_exp2_nvfp4_pair<Quarter, 15, C>(
        scores.v15, row_max, scale_log2);

    const float sum =
        hao_direct_sum8_pairs(
            scores.v0, scores.v1, scores.v2, scores.v3,
            scores.v4, scores.v5, scores.v6, scores.v7) +
        hao_direct_sum8_pairs(
            scores.v8, scores.v9, scores.v10, scores.v11,
            scores.v12, scores.v13, scores.v14, scores.v15);
    uint32_t words[4];
    uint8_t e8m0;
    fp4pv_quantize_mxfp4_scores_group_rte(
        &scores.v0, words, e8m0);
    const int row =
        reader_warp * 32 + warp::laneid();
    if constexpr (CaptureScaleByte) {
        *stored_e8m0_out = e8m0;
    } else {
        fp4pv_store_mxfp4_scale_byte_swizzled(
            p_scale, row, Quarter, e8m0);
    }
    if constexpr (CapturePackedWords) {
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            packed_words_out[word] = words[word];
        }
    }
    hao_direct_store_p_nvfp4_quarter_tmem<Quarter>(
        p,
        reader_warp,
        words[0],
        words[1],
        words[2],
        words[3]);
    return sum;
}

template <
    int Quarter,
    typename C,
    bool ForceAffine = false,
    bool CapturePackedWords = false,
    bool PreparedAnchorAffine = false,
    bool CaptureScaleByte = false,
    bool InterleaveExactMax = false,
    bool RebaseInterleavedScale = false,
    typename P,
    typename PScale>
__device__ __forceinline__ float
hao_direct_transform_pack_fp4_quarter(
    P &p,
    PScale &p_scale,
    int reader_warp,
    score_quarter_regs &scores,
    float row_max,
    float scale_log2,
    float raw_group_max,
    uint32_t *packed_words_out = nullptr,
    float prepared_quant_bias_base = 0.0f,
    float prepared_affine_scale = 0.0f,
    uint8_t *stored_e8m0_out = nullptr,
    uint8_t *pair_e8m0_io = nullptr,
    bool reuse_pair_scale_runtime = true,
    const score_quarter_regs *interleaved_max_scores = nullptr,
    float *interleaved_max_out = nullptr
) {
    static_assert(
        !CapturePackedWords || C::HAO_DIRECT_MXFP4_PV);
    static_assert(
        !PreparedAnchorAffine || C::HAO_DIRECT_MXFP4_PV);
    static_assert(
        !CaptureScaleByte || C::HAO_DIRECT_MXFP4_PV);
    static_assert(
        !InterleaveExactMax || C::HAO_DIRECT_MXFP4_PV);
    static_assert(
        !RebaseInterleavedScale || C::HAO_DIRECT_MXFP4_PV);
    if constexpr (C::HAO_DIRECT_ROWMAX_PACK_CEILING) {
        return hao_direct_pack_raw_score_quarter<Quarter, C>(
            p,
            p_scale,
            reader_warp,
            scores);
    }
    if constexpr (C::HAO_DIRECT_NVFP4_PV) {
        return hao_direct_transform_pack_nvfp4_quarter<
            Quarter, C, ForceAffine>(
                p,
                p_scale,
                reader_warp,
                scores,
                row_max,
                scale_log2,
                raw_group_max);
    } else {
        return hao_direct_transform_pack_mxfp4_quarter<
            Quarter,
            C,
            ForceAffine,
            CapturePackedWords,
            PreparedAnchorAffine,
            CaptureScaleByte,
            InterleaveExactMax,
            RebaseInterleavedScale>(
                p,
                p_scale,
                reader_warp,
                scores,
                row_max,
                scale_log2,
                raw_group_max,
                packed_words_out,
                prepared_quant_bias_base,
                prepared_affine_scale,
                stored_e8m0_out,
                pair_e8m0_io,
                reuse_pair_scale_runtime,
                interleaved_max_scores,
                interleaved_max_out);
    }
}

template <typename C>
__cluster_dims__(1, 1, 1) __launch_bounds__(512, 1)
__global__ void kernel_fp4pv_hao_direct_fp4pv(
    const __grid_constant__ globals_fp4pv_hao_direct<C> g) {
    static_assert(C::HAO_DIRECT_PIPELINE);
    static_assert(C::NUM_THREADS == 512);
    static_assert(C::NUM_WARPS == 16);
    static_assert(C::CLUSTER_SIZE == 1);
    static_assert(C::Mb == 128 && C::Nb == 128);
    static_assert(C::Dqk == 128 && C::Dvo == 128);
    static_assert(C::QK_SCALE_CHUNKS == 2);
    static_assert(C::KV_STAGES >= 4 && C::KV_STAGES <= 13);
    static_assert(C::P_FIRST_CHUNKS == 1);
    static_assert(
        !C::HAO_DIRECT_MX_EARLY_P ||
        C::HAO_DIRECT_MXFP4_PV);

    using G = globals_fp4pv_hao_direct<C>;
    using Score = tt<float, 128, 128>;
    using Output = tt<float, 128, 128>;
    using DenseP = tt<fp4e2m1_2, 128, 64>;
    using P = std::conditional_t<
        C::HAO_DIRECT_SPARSE_PV,
        hao_direct_sparse_p,
        DenseP>;
    constexpr int S0 = 0;
    constexpr int S1 = 128;
    constexpr int O0 = 256;
    constexpr int O1 = 384;
    constexpr int P0 = S0 + 64;
    constexpr int P1 = S1 + 64;
    static_assert(P0 == 64 && P1 == 192);
    static_assert(S0 + 48 <= P0);
    static_assert(S1 + 48 <= P1);
    static_assert(O1 + 128 == C::TMEM_COLUMNS);
    static_assert(
        hao_direct_pipeline_smem_layout<C>::bytes() <=
        227 * 1024);

    tensor_allocator<1, 1> tm{};
    Score score0 = tm.template allocate<Score>(S0);
    Score score1 = tm.template allocate<Score>(S1);
    Output output0 = tm.template allocate<Output>(O0);
    Output output1 = tm.template allocate<Output>(O1);

    extern __shared__ int shm[];
    tma_swizzle_allocator al(shm);
    typename G::q_tile (&q_smem)[2] =
        al.allocate<typename G::q_tile, 2>();
    typename G::q_sc_tile (&q_sc_smem)[2] =
        al.allocate<typename G::q_sc_tile, 2>();
    hao_direct_fp4_kv_payload_slot<C>
        (&kv_payload)[C::KV_STAGES] =
            al.allocate<
                hao_direct_fp4_kv_payload_slot<C>,
                C::KV_STAGES>();
    hao_direct_fp4_kv_scale_slot<C>
        (&kv_scale)[C::KV_STAGES] =
            al.allocate<
                hao_direct_fp4_kv_scale_slot<C>,
                C::KV_STAGES>();
    hao_direct_sparse_p_tile *sparse_p_payload = nullptr;
    sv_hf<C::Mb> *sparse_p_metadata = nullptr;
    hao_direct_sparse_selector_lut *sparse_selector_lut = nullptr;
    if constexpr (C::HAO_DIRECT_SPARSE_PV) {
        constexpr int SPARSE_PAYLOADS =
            C::HAO_DIRECT_SPARSE_SPLIT_K128 ? 4 : 2;
        auto &payload_storage =
            al.allocate<hao_direct_sparse_p_tile, SPARSE_PAYLOADS>();
        sparse_p_payload = &payload_storage[0];
        auto &metadata_storage =
            al.allocate<sv_hf<C::Mb>, 4>();
        sparse_p_metadata = &metadata_storage[0];
        if constexpr (
            C::HAO_DIRECT_SPARSE_SELECTOR_MODE >= 4) {
            sparse_selector_lut =
                &al.allocate<hao_direct_sparse_selector_lut>();
        }
    }
    typename G::o_tile (&o_smem)[2] =
        al.allocate<typename G::o_tile, 2>();
    typename G::p_sc_tile (&p_sc_smem)[2] =
        al.allocate<typename G::p_sc_tile, 2>();
    sv_fl<128> (&row_max_smem)[2] =
        al.allocate<sv_fl<128>, 2>();
    sv_fl<128> (&row_sum_smem)[2] =
        al.allocate<sv_fl<128>, 2>();
    sv_fl<128> (&correction_smem)[2] =
        al.allocate<sv_fl<128>, 2>();
    hao_direct_nv_first_payload_smem *nv_first_payload_smem =
        nullptr;
    if constexpr (
        C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF ||
        C::HAO_DIRECT_MX_DENOM_EARLY_SMEM) {
        auto &storage =
            al.allocate<hao_direct_nv_first_payload_smem, 2>();
        nv_first_payload_smem = &storage[0];
    }
    hao_direct_mx_denom_payload_smem *mx_denom_payload_smem = nullptr;
    if constexpr (C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM) {
        auto &storage =
            al.allocate<hao_direct_mx_denom_payload_smem, 2>();
        mx_denom_payload_smem = &storage[0];
    }
    hao_direct_q3_smem_slot<C> *q3_stage_smem = nullptr;
    if constexpr (C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG) {
        auto &q3_stage_storage =
            al.allocate<hao_direct_q3_smem_slot<C>, 2>();
        q3_stage_smem = &q3_stage_storage[0];
    }
    P p0 = [&]() {
        if constexpr (C::HAO_DIRECT_SPARSE_PV) {
            return P{
                P0,
                &sparse_p_payload[0],
                P0 + (C::HAO_DIRECT_SPARSE_SPLIT_K128 ? 2 : 0),
                &sparse_p_payload[
                    C::HAO_DIRECT_SPARSE_SPLIT_K128 ? 1 : 0],
                &sparse_p_metadata[0],
                sparse_selector_lut};
        } else {
            return tm.template allocate<DenseP>(P0);
        }
    }();
    P p1 = [&]() {
        if constexpr (C::HAO_DIRECT_SPARSE_PV) {
            return P{
                P1,
                &sparse_p_payload[
                    C::HAO_DIRECT_SPARSE_SPLIT_K128 ? 2 : 1],
                P1 + (C::HAO_DIRECT_SPARSE_SPLIT_K128 ? 2 : 0),
                &sparse_p_payload[
                    C::HAO_DIRECT_SPARSE_SPLIT_K128 ? 3 : 1],
                &sparse_p_metadata[2],
                sparse_selector_lut};
        } else {
            return tm.template allocate<DenseP>(P1);
        }
    }();
    auto *bypass_semaphores = reinterpret_cast<semaphore *>(
        &correction_smem[1]);
    semaphore (&output_empty)[2] =
        *reinterpret_cast<semaphore (*)[2]>(bypass_semaphores);
    semaphore &q3_sum_full = bypass_semaphores[2];
    semaphore &q3_sum_empty = bypass_semaphores[3];

    __shared__ semaphore q_full[2];
    __shared__ semaphore q_empty[2];
    __shared__ semaphore kv_full[C::KV_STAGES];
    __shared__ semaphore kv_empty[C::KV_STAGES];
    __shared__ semaphore p_first_o_rescaled[2];
    __shared__ semaphore p_tail[2];
    __shared__ semaphore score_full[2];
    __shared__ semaphore q1_scale_full[2];
    __shared__ hao_direct_q1_scale_slot q1_scale_smem[2];
    __shared__ semaphore output_full[2];
    __shared__ semaphore softmax_corr_full[2];
    __shared__ semaphore softmax_corr_empty[2];
    __shared__ semaphore nv_denom_loaded[2];
    __shared__ semaphore nv_denom_done[2];
    __shared__ semaphore corr_epi_full[2];
    __shared__ semaphore corr_epi_empty[2];
    __shared__ semaphore sfqk_load[2];
    __shared__ semaphore global_anchor_full[2];
    __shared__ semaphore dual_q3_sum_full[2];
    __shared__ semaphore dual_q3_sum_empty[2];
    __shared__ semaphore dual_q2_ready[2];
    __shared__ semaphore dual_q3_staged[2];
    __shared__ semaphore dual_q3_checkpoint_empty[2];
    __shared__ semaphore tmem_dealloc;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(q_full[stage], 0, 1);
            init_semaphore(q_empty[stage], 0, 1);
            init_semaphore(
                p_first_o_rescaled[stage],
                0,
                C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS ? 4 : 8);
            init_semaphore(p_tail[stage], 0, 4);
            init_semaphore(score_full[stage], 0, 1);
            init_semaphore(q1_scale_full[stage], 0, 4);
            init_semaphore(output_full[stage], 0, 1);
            init_semaphore(
                softmax_corr_full[stage], 0, 4);
            init_semaphore(
                softmax_corr_empty[stage], 0, 4);
            if constexpr (C::HAO_DIRECT_DENOM_CORR_WG) {
                init_semaphore(nv_denom_loaded[stage], 0, 4);
                init_semaphore(nv_denom_done[stage], 0, 4);
            }
            if constexpr (
                C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                init_semaphore(output_empty[stage], 0, 4);
            }
            init_semaphore(corr_epi_full[stage], 0, 4);
            init_semaphore(corr_epi_empty[stage], 0, 1);
            init_semaphore(
                sfqk_load[stage],
                0,
                (C::HAO_DIRECT_MX_DUAL_Q3_CORR_WG ||
                 C::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD)
                    ? 8
                    : 4);
            if constexpr (
                C::HAO_DIRECT_NV_GLOBAL_ANCHOR_CORR_WG) {
                init_semaphore(global_anchor_full[stage], 0, 4);
            }
            if constexpr (
                C::HAO_DIRECT_MX_Q3_CORR_WG ||
                C::HAO_DIRECT_MX_DUAL_Q3_CORR_WG ||
                C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG ||
                C::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG ||
                C::HAO_DIRECT_NV_Q01_CORR_WG) {
                init_semaphore(dual_q3_sum_full[stage], 0, 4);
                init_semaphore(dual_q3_sum_empty[stage], 0, 4);
                init_semaphore(dual_q2_ready[stage], 0, 4);
            }
            if constexpr (
                C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG ||
                C::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG) {
                init_semaphore(dual_q3_staged[stage], 0, 4);
            }
            if constexpr (C::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG) {
                init_semaphore(
                    dual_q3_checkpoint_empty[stage], 0, 4);
            }
        }
        if constexpr (
            C::HAO_DIRECT_MX_Q3_CORR_WG ||
            C::HAO_DIRECT_NV_Q01_CORR_WG) {
            init_semaphore(q3_sum_full, 0, 4);
            init_semaphore(q3_sum_empty, 0, 4);
        }
        #pragma unroll
        for (int stage = 0; stage < C::KV_STAGES; ++stage) {
            init_semaphore(kv_full[stage], 0, 1);
            init_semaphore(kv_empty[stage], 0, 1);
        }
        init_semaphore(tmem_dealloc, 0, 12);
    }
    if constexpr (C::HAO_DIRECT_SPARSE_SELECTOR_MODE >= 4) {
        if (threadIdx.x < 256) {
            sparse_selector_lut->data[threadIdx.x] =
                hao_direct_sparse_selector_lut_entry(threadIdx.x);
        }
    }
    if constexpr (C::HAO_DIRECT_SPARSE_SPLIT_K128) {
        constexpr int SPARSE_PAYLOADS = 4;
        constexpr int PAYLOAD_WORDS =
            SPARSE_PAYLOADS *
            int(sizeof(hao_direct_sparse_p_tile)) /
            int(sizeof(uint32_t));
        auto *payload_words =
            reinterpret_cast<uint32_t *>(sparse_p_payload);
        for (int word = int(threadIdx.x);
             word < PAYLOAD_WORDS;
             word += C::NUM_THREADS) {
            payload_words[word] = 0;
        }

        const int init_warp = int(threadIdx.x) >> 5;
        if (init_warp < 8) {
            const int reader_warp = init_warp & 3;
            auto &p_owned = init_warp < 4 ? p0 : p1;
            const uint32_t row_base =
                static_cast<uint32_t>(reader_warp * 32) << 16;
            const uint32_t first_zero_tail =
                p_owned.addr + row_base + 1;
            const uint32_t tail_zero_first =
                p_owned.tail_addr + row_base;
            constexpr uint32_t ZERO_METADATA = 0x44444444u;
            asm volatile(
                "{tcgen05.st.sync.aligned.32x32b.x1.b32 "
                "[%0], {%2};\n\t"
                "tcgen05.st.sync.aligned.32x32b.x1.b32 "
                "[%1], {%2};}"
                :: "r"(first_zero_tail),
                   "r"(tail_zero_first),
                   "r"(ZERO_METADATA));
        }
    }
    __syncthreads();
    if constexpr (C::HAO_DIRECT_SPARSE_SPLIT_K128) {
        fp4pv_shared_async_proxy_fence_cta();
        tensor_before_thread_sync();
        __syncthreads();
    }

    const int warp_id = int(threadIdx.x) >> 5;
    if (warp_id == 14) {
        warpgroup::decrease_registers<C::AUXILIARY_REGS>();
        const bool leader = warp::elect_leader();
        int q_producer_phase = 1;
        int kv_producer_item = 0;

        auto load_k = [&](const hao_direct_task<C> &task, int n) {
            const int slot =
                kv_producer_item % C::KV_STAGES;
            const int producer_phase =
                1 ^ ((kv_producer_item / C::KV_STAGES) & 1);
            hao_direct_fp4pv_wait(
                kv_empty[slot], producer_phase);
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
            hao_direct_fp4pv_wait(
                kv_empty[slot], producer_phase);
            if (leader) {
                fp4pv_expect_bytes<C>(
                    kv_full[slot],
                    sizeof(typename G::v_fp4_tile) +
                        sizeof(typename G::v_sc_tile));
                tma::load_async<
                    dim::ROW, cache_policy::NORMAL>(
                    kv_payload[slot].v,
                    g.v_fp4,
                    {task.batch, task.head, 0, n},
                    kv_full[slot]);
                tma::load_async<
                    dim::ROW, cache_policy::NORMAL>(
                    kv_scale[slot].v,
                    g.v_sc,
                    {task.batch, n, task.head, 0},
                    kv_full[slot]);
            }
            ++kv_producer_item;
        };
        auto load_q = [&](
            const hao_direct_task<C> &task,
            int stage,
            int qtile) {
            hao_direct_fp4pv_wait(
                q_empty[stage], q_producer_phase);
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
        warpgroup::decrease_registers<C::PRODUCER_REGS>();
        const bool leader = warp::elect_leader();
        int q_consumer_phase = 0;
        int kv_consumer_item = 0;
        int p_phase = 0;
        int sfqk_phase = 0;
        int output_empty_phase = 1;

        auto wait_kv_item = [&](int item) {
            const int slot = item % C::KV_STAGES;
            const int phase =
                (item / C::KV_STAGES) & 1;
            if (leader) {
                hao_direct_fp4pv_wait(kv_full[slot], phase);
            }
            return slot;
        };
        auto issue_qk = [&]<int Stage>(
            int k_slot, int sf_phase) {
            if (leader) {
                hao_direct_fp4pv_wait(
                    sfqk_load[Stage], sf_phase);
                if constexpr (Stage == 0) {
                    if constexpr (C::HAO_DIRECT_NVFP4_QK) {
                        hao_direct_issue_qk_nvfp4_compact_scales<C>(
                            score0,
                            q_smem[0],
                            kv_payload[k_slot].k,
                            q_sc_smem[0],
                            kv_scale[k_slot].k,
                            S1,
                            score_full[0]);
                    } else {
                        hao_direct_issue_qk_mxfp4_compact_scales<C>(
                            score0,
                            q_smem[0],
                            kv_payload[k_slot].k,
                            q_sc_smem[0],
                            kv_scale[k_slot].k,
                            S1,
                            score_full[0]);
                    }
                } else {
                    if constexpr (C::HAO_DIRECT_NVFP4_QK) {
                        hao_direct_issue_qk_nvfp4_compact_scales<C>(
                            score1,
                            q_smem[1],
                            kv_payload[k_slot].k,
                            q_sc_smem[1],
                            kv_scale[k_slot].k,
                            S0,
                            score_full[1]);
                    } else {
                        hao_direct_issue_qk_mxfp4_compact_scales<C>(
                            score1,
                            q_smem[1],
                            kv_payload[k_slot].k,
                            q_sc_smem[1],
                            kv_scale[k_slot].k,
                            S0,
                            score_full[1]);
                    }
                }
            }
        };
        auto preload_qk_scales = [&]<int Stage>(
            int k_slot, int sf_phase) {
            if constexpr (
                C::HAO_DIRECT_MX_QK_SCALE_PRELOAD &&
                (C::HAO_DIRECT_MX_QK_SCALE_PRELOAD_MASK &
                 (1 << Stage))) {
                if (leader) {
                    hao_direct_fp4pv_wait(
                        sfqk_load[Stage], sf_phase);
                    if constexpr (Stage == 0) {
                        hao_direct_preload_qk_nvfp4_compact_scales<C>(
                            q_sc_smem[0],
                            kv_scale[k_slot].k,
                            S1);
                    } else {
                        hao_direct_preload_qk_nvfp4_compact_scales<C>(
                            q_sc_smem[1],
                            kv_scale[k_slot].k,
                            S0);
                    }
                }
            }
        };
        auto issue_qk_preloaded = [&]<int Stage>(int k_slot) {
            if constexpr (
                C::HAO_DIRECT_MX_QK_SCALE_PRELOAD &&
                (C::HAO_DIRECT_MX_QK_SCALE_PRELOAD_MASK &
                 (1 << Stage))) {
                if (leader) {
                    if constexpr (Stage == 0) {
                        hao_direct_issue_qk_nvfp4_preloaded_scales<C>(
                            score0,
                            q_smem[0],
                            kv_payload[k_slot].k,
                            q_sc_smem[0],
                            kv_scale[k_slot].k,
                            S1,
                            score_full[0]);
                    } else {
                        hao_direct_issue_qk_nvfp4_preloaded_scales<C>(
                            score1,
                            q_smem[1],
                            kv_payload[k_slot].k,
                            q_sc_smem[1],
                            kv_scale[k_slot].k,
                            S0,
                            score_full[1]);
                    }
                }
            } else {
                issue_qk.template operator()<Stage>(
                    k_slot, sfqk_phase);
            }
        };
        auto wait_nv_denom_loaded = [&](int stage) {
            if constexpr (C::HAO_DIRECT_DENOM_CORR_WG) {
                if (leader) {
                    hao_direct_fp4pv_wait(
                        nv_denom_loaded[stage], p_phase);
                }
            }
        };
        auto issue_pv = [&]<int Stage>(
            int v_slot, bool accumulate) {
            if (leader) {
                if constexpr (
                    C::HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK != 0) {
                    constexpr uint32_t VScaleBase =
                        Stage == 0 ? S0 : S1;
                    if constexpr (
                        (C::HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK &
                         0x1) != 0) {
                        hao_direct_load_nvfp4_scale_ranklocal<0>(
                            VScaleBase + 8,
                            kv_scale[v_slot].v);
                    }
                    if constexpr (
                        (C::HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK &
                         0x2) != 0) {
                        hao_direct_load_nvfp4_scale_ranklocal<1>(
                            VScaleBase + 12,
                            kv_scale[v_slot].v);
                    }
                }
                hao_direct_fp4pv_wait(
                    p_first_o_rescaled[Stage],
                    p_phase);
                if constexpr (
                    (C::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK &
                     (1 << Stage)) != 0) {
                    tensor_after_thread_sync();
                }
                if constexpr (Stage == 0) {
                    if constexpr (C::HAO_DIRECT_SPARSE_PV) {
                        hao_direct_issue_pv_sparse_nvfp4_n128<C>(
                            output0,
                            p0,
                            kv_payload[v_slot].v,
                            kv_scale[v_slot].v,
                            S0 + 32,
                            S0 + 48,
                            p_tail[0],
                            p_phase,
                            accumulate);
                    } else if constexpr (C::HAO_DIRECT_NVFP4_PV) {
                        hao_direct_issue_pv_nvfp4_n128<C, 0>(
                            output0,
                            p0,
                            kv_payload[v_slot].v,
                            p_sc_smem[0],
                            kv_scale[v_slot].v,
                            S0,
                            S0,
                            p_tail[0],
                            p_phase,
                            accumulate);
                    } else {
                        hao_direct_issue_pv_mxfp4_n128<C>(
                            output0,
                            p0,
                            kv_payload[v_slot].v,
                            p_sc_smem[0],
                            kv_scale[v_slot].v,
                            S0,
                            p_tail[0],
                            p_phase,
                            accumulate);
                    }
                } else {
                    if constexpr (C::HAO_DIRECT_SPARSE_PV) {
                        hao_direct_issue_pv_sparse_nvfp4_n128<C>(
                            output1,
                            p1,
                            kv_payload[v_slot].v,
                            kv_scale[v_slot].v,
                            S1 + 32,
                            S1 + 48,
                            p_tail[1],
                            p_phase,
                            accumulate);
                    } else if constexpr (C::HAO_DIRECT_NVFP4_PV) {
                        hao_direct_issue_pv_nvfp4_n128<C, 1>(
                            output1,
                            p1,
                            kv_payload[v_slot].v,
                            p_sc_smem[1],
                            kv_scale[v_slot].v,
                            S1,
                            C::HAO_DIRECT_NV_REUSE_V_SCALE ? S0 : S1,
                            p_tail[1],
                            p_phase,
                            accumulate);
                        if constexpr (
                            C::HAO_DIRECT_NV_REUSE_V_SCALE) {
                            tensor_before_thread_sync();
                        }
                    } else {
                        hao_direct_issue_pv_mxfp4_n128<C>(
                            output1,
                            p1,
                            kv_payload[v_slot].v,
                            p_sc_smem[1],
                            kv_scale[v_slot].v,
                            S1,
                            p_tail[1],
                            p_phase,
                            accumulate);
                    }
                }
            }
        };
        auto issue_pv_first = [&]<int Stage>(
            int v_slot, bool accumulate) {
            if (leader) {
                hao_direct_fp4pv_wait(
                    p_first_o_rescaled[Stage],
                    p_phase);
                if constexpr (Stage == 0) {
                    hao_direct_issue_pv_mxfp4_first<C>(
                        output0,
                        p0,
                        kv_payload[v_slot].v,
                        p_sc_smem[0],
                        kv_scale[v_slot].v,
                        S0,
                        accumulate);
                } else {
                    hao_direct_issue_pv_mxfp4_first<C>(
                        output1,
                        p1,
                        kv_payload[v_slot].v,
                        p_sc_smem[1],
                        kv_scale[v_slot].v,
                        S1,
                        accumulate);
                }
            }
        };
        auto issue_pv_tail = [&]<int Stage>(int v_slot) {
            if (leader) {
                if constexpr (Stage == 0) {
                    hao_direct_issue_pv_mxfp4_tail<C>(
                        output0,
                        p0,
                        kv_payload[v_slot].v,
                        p_sc_smem[0],
                        S0,
                        p_tail[0],
                        p_phase);
                } else {
                    hao_direct_issue_pv_mxfp4_tail<C>(
                        output1,
                        p1,
                        kv_payload[v_slot].v,
                        p_sc_smem[1],
                        S1,
                        p_tail[1],
                        p_phase);
                }
            }
        };

        for (int logical_job = int(blockIdx.x);
             logical_job < C::LOGICAL_JOBS;
             logical_job += int(gridDim.x)) {
            if (leader) {
                hao_direct_fp4pv_wait(
                    q_full[0], q_consumer_phase);
            }
            const int first_k_slot =
                wait_kv_item(kv_consumer_item);
            issue_qk.template operator()<0>(
                first_k_slot, sfqk_phase);
            if (leader) {
                hao_direct_fp4pv_wait(
                    q_full[1], q_consumer_phase);
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
                if constexpr (
                    C::HAO_DIRECT_MX_QK_SCALE_PRELOAD) {
                    if constexpr (
                        C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                        if (n == 0 && leader) {
                            hao_direct_fp4pv_wait(
                                output_empty[0], output_empty_phase);
                        }
                    }
                    int k_slot;
                    if constexpr (
                        C::HAO_DIRECT_MX_QK_SCALE_PRELOAD_BEFORE_P) {
                        const int next_k_item =
                            kv_consumer_item + 1;
                        k_slot = wait_kv_item(next_k_item);
                        preload_qk_scales.template operator()<0>(
                            k_slot, sfqk_phase);
                    }
                    issue_pv_first.template operator()<0>(
                        v_slot, n != 0);

                    if constexpr (
                        !C::HAO_DIRECT_MX_QK_SCALE_PRELOAD_BEFORE_P) {
                        const int next_k_item =
                            kv_consumer_item + 1;
                        k_slot = wait_kv_item(next_k_item);
                        preload_qk_scales.template operator()<0>(
                            k_slot, sfqk_phase);
                    }
                    issue_pv_tail.template operator()<0>(v_slot);
                    issue_qk_preloaded.template operator()<0>(k_slot);

                    if constexpr (
                        C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                        if (n == 0 && leader) {
                            hao_direct_fp4pv_wait(
                                output_empty[1], output_empty_phase);
                        }
                    }
                    issue_pv_first.template operator()<1>(
                        v_slot, n != 0);
                    preload_qk_scales.template operator()<1>(
                        k_slot, sfqk_phase);
                    issue_pv_tail.template operator()<1>(v_slot);
                    issue_qk_preloaded.template operator()<1>(k_slot);
                    if (leader) {
                        detail::tcgen05::commit<1>(
                            kv_empty[v_slot]);
                    }
                    if (leader) {
                        detail::tcgen05::commit<1>(
                            kv_empty[k_slot]);
                    }
                } else if constexpr (
                    C::HAO_DIRECT_MX_SPLIT_STAGE_PV) {
                    if constexpr (
                        C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                        if (n == 0 && leader) {
                            hao_direct_fp4pv_wait(
                                output_empty[0], output_empty_phase);
                        }
                    }
                    issue_pv_first.template operator()<0>(
                        v_slot, n != 0);
                    if constexpr (
                        C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                        if (n == 0 && leader) {
                            hao_direct_fp4pv_wait(
                                output_empty[1], output_empty_phase);
                        }
                    }
                    issue_pv_first.template operator()<1>(
                        v_slot, n != 0);
                    issue_pv_tail.template operator()<0>(v_slot);

                    const int next_k_item =
                        kv_consumer_item + 1;
                    const int k_slot =
                        wait_kv_item(next_k_item);
                    wait_nv_denom_loaded(0);
                    issue_qk.template operator()<0>(
                        k_slot, sfqk_phase);
                    if (leader) {
                        full_tt_fp8e8m0<16> stage1_v_sc(S1 + 16);
                        hao_direct_load_mxfp4_scale_ranklocal(
                            stage1_v_sc,
                            kv_scale[v_slot].v);
                    }
                    issue_pv_tail.template operator()<1>(v_slot);
                    if (leader) {
                        detail::tcgen05::commit<1>(
                            kv_empty[v_slot]);
                    }
                    wait_nv_denom_loaded(1);
                    issue_qk.template operator()<1>(
                        k_slot, sfqk_phase);
                    if (leader) {
                        detail::tcgen05::commit<1>(
                            kv_empty[k_slot]);
                    }
                } else {
                    if constexpr (
                        C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                        if (n == 0 && leader) {
                            hao_direct_fp4pv_wait(
                                output_empty[0], output_empty_phase);
                        }
                    }
                    issue_pv.template operator()<0>(
                        v_slot, n != 0);

                    const int next_k_item =
                        kv_consumer_item + 1;
                    const int k_slot =
                        wait_kv_item(next_k_item);
                    wait_nv_denom_loaded(0);
                    issue_qk.template operator()<0>(
                        k_slot, sfqk_phase);

                    if constexpr (
                        C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                        if (n == 0 && leader) {
                            hao_direct_fp4pv_wait(
                                output_empty[1], output_empty_phase);
                        }
                    }
                    issue_pv.template operator()<1>(
                        v_slot, n != 0);
                    if (leader) {
                        detail::tcgen05::commit<1>(
                            kv_empty[v_slot]);
                    }
                    wait_nv_denom_loaded(1);
                    issue_qk.template operator()<1>(
                        k_slot, sfqk_phase);
                    if (leader) {
                        detail::tcgen05::commit<1>(
                            kv_empty[k_slot]);
                    }
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
            if constexpr (
                C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
                C::N_TILES == 1) {
                if (leader) {
                    hao_direct_fp4pv_wait(
                        output_empty[0], output_empty_phase);
                }
            }
            if constexpr (C::HAO_DIRECT_MX_SPLIT_STAGE_PV) {
                issue_pv_first.template operator()<0>(
                    final_v_slot, C::N_TILES != 1);
                if constexpr (
                    C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
                    C::N_TILES == 1) {
                    if (leader) {
                        hao_direct_fp4pv_wait(
                            output_empty[1], output_empty_phase);
                    }
                }
                issue_pv_first.template operator()<1>(
                    final_v_slot, C::N_TILES != 1);
                issue_pv_tail.template operator()<0>(final_v_slot);
                if (leader) {
                    detail::tcgen05::commit<1>(output_full[0]);
                }
                issue_pv_tail.template operator()<1>(final_v_slot);
                if (leader) {
                    detail::tcgen05::commit<1>(output_full[1]);
                    detail::tcgen05::commit<1>(
                        kv_empty[final_v_slot]);
                }
            } else {
                issue_pv.template operator()<0>(
                    final_v_slot, C::N_TILES != 1);
                if (leader) {
                    detail::tcgen05::commit<1>(output_full[0]);
                }
                if constexpr (
                    C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
                    C::N_TILES == 1) {
                    if (leader) {
                        hao_direct_fp4pv_wait(
                            output_empty[1], output_empty_phase);
                    }
                }
                issue_pv.template operator()<1>(
                    final_v_slot, C::N_TILES != 1);
                if (leader) {
                    detail::tcgen05::commit<1>(output_full[1]);
                    detail::tcgen05::commit<1>(
                        kv_empty[final_v_slot]);
                }
            }
            if constexpr (C::HAO_DIRECT_DENOM_CORR_WG) {
                if (leader) {
                    hao_direct_fp4pv_wait(
                        nv_denom_loaded[0], p_phase);
                    hao_direct_fp4pv_wait(
                        nv_denom_loaded[1], p_phase);
                }
            }
            ++kv_consumer_item;
            p_phase ^= 1;
            if constexpr (
                C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                output_empty_phase ^= 1;
            }
        }

        hao_direct_fp4pv_wait(tmem_dealloc, 0);
    } else if (warp_id < 4) {
        warpgroup::increase_registers<C::SOFTMAX_STAGE0_REGS>();
#define HAO_DIRECT_SOFTMAX_STAGE 0
#include "hao_direct_fp4pv_softmax_reader.inc"
#undef HAO_DIRECT_SOFTMAX_STAGE
    } else if (warp_id < 8) {
        warpgroup::increase_registers<C::SOFTMAX_REGS>();
#define HAO_DIRECT_SOFTMAX_STAGE 1
#include "hao_direct_fp4pv_softmax_reader.inc"
#undef HAO_DIRECT_SOFTMAX_STAGE
    } else if (warp_id >= 8 && warp_id < 12) {
        warpgroup::decrease_registers<C::CORRECTION_ACTIVE_REGS>();
        const int row_group = warp_id - 8;
        int softmax_consumer_phase = 0;
        int output_consumer_phase = 0;
        int epi_producer_phase = 1;
        int q3_score_phase = 0;
        int q3_empty_phase = 1;
        int q3_sfqk_phase = 0;
        int dual_q3_score_phase = 0;
        int dual_q2_phase = 0;
        int dual_q3_empty_phase = 1;
        int q01_score_phase = 0;
        int q01_ready_phase = 0;
        int q01_empty_phase = 1;
        int global_anchor_score_phase = 0;
        int nv_denom_consumer_phase = 0;
        int q1_scale_score_phase = 0;

        // The first PV overwrites empty output accumulators.  These arrivals
        // provide the output-ready half of the first combined P/O barrier.
        if constexpr (
            !C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
            if (warp::elect_leader()) {
                arrive(p_first_o_rescaled[0]);
                arrive(p_first_o_rescaled[1]);
            }
        }
        if constexpr (C::HAO_DIRECT_MX_DUAL_Q3_CORR_WG) {
            // Stage-0 QK initially borrows the still-empty stage-1 score
            // bank. Match stage-1 softmax's bootstrap lease contribution.
            if (warp::elect_leader()) {
                arrive(sfqk_load[0]);
            }
        }
        if constexpr (C::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD) {
            // Stage-0 QK borrows the not-yet-live stage-1 score bank for its
            // scales. Match the stage-1 reader's bootstrap contribution.
            if (warp::elect_leader()) {
                arrive(sfqk_load[0]);
            }
        }

        for (int logical_job = int(blockIdx.x);
             logical_job < C::LOGICAL_JOBS;
             logical_job += int(gridDim.x)) {
            const hao_direct_task task =
                hao_direct_decode_task<C>(logical_job);

            if constexpr (
                C::HAO_DIRECT_NV_GLOBAL_ANCHOR_CORR_WG) {
                hao_direct_fp4pv_wait(
                    score_full[0], global_anchor_score_phase);
                tensor_after_thread_sync();
                score_quarter_regs q1_scores;
                upstream_load_score_quarter_x32<1, C, false>(
                    q1_scores,
                    score0.addr + (row_group * 32 << 16),
                    0,
                    task.qtile0);
                tensor_load_wait();
                float q1_max =
                    base_types::constants<float>::neg_infty();
                upstream_reduce_score_quarter<C>(
                    q1_scores, q1_max);
                stats_store(
                    row_max_smem[0],
                    warpgroup::laneid(),
                    q1_max);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(global_anchor_full[0]);
                }
                global_anchor_score_phase ^=
                    C::N_TILES & 1;
            }

            if constexpr (C::HAO_DIRECT_DENOM_CORR_WG) {
                float represented_row_sum[2] = {
                    C::HAO_DIRECT_FAKE_DENOM_CEILING ? 1.0f : 0.0f,
                    C::HAO_DIRECT_FAKE_DENOM_CEILING ? 1.0f : 0.0f};
                if constexpr (
                    C::HAO_DIRECT_MX_DENOM_SPLIT_PREFETCH) {
                    #pragma unroll 1
                    for (int n = 0; n < C::N_TILES; ++n) {
                        hao_direct_nv_denom_half_snapshot first0;
                        hao_direct_nv_denom_half_snapshot first1;
                        hao_direct_nv_denom_half_snapshot tail0;
                        hao_direct_nv_denom_half_snapshot tail1;

                        hao_direct_fp4pv_wait(
                            p_first_o_rescaled[0],
                            nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot<0, C>(
                            first0, p0, S0, row_group);
                        hao_direct_fp4pv_wait(
                            p_tail[0], nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot<1, C>(
                            tail0, p0, S0, row_group);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(nv_denom_loaded[0]);
                        }
                        hao_direct_accumulate_mx_denom_snapshot_pair<C>(
                            represented_row_sum[0], first0, tail0);

                        hao_direct_fp4pv_wait(
                            p_first_o_rescaled[1],
                            nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot<0, C>(
                            first1, p1, S1, row_group);
                        hao_direct_fp4pv_wait(
                            p_tail[1], nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot<1, C>(
                            tail1, p1, S1, row_group);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(nv_denom_loaded[1]);
                        }
                        hao_direct_accumulate_mx_denom_snapshot_pair<C>(
                            represented_row_sum[1], first1, tail1);

                        if (n + 1 == C::N_TILES) {
                            const int row =
                                row_group * 32 + warp::laneid();
                            stats_store(
                                row_sum_smem[0],
                                row,
                                hao_direct_finalize_mx_denom<C>(
                                    represented_row_sum[0]));
                            stats_store(
                                row_sum_smem[1],
                                row,
                                hao_direct_finalize_mx_denom<C>(
                                    represented_row_sum[1]));
                            __threadfence_block();
                            if (warp::elect_leader()) {
                                arrive(nv_denom_done[0]);
                                arrive(nv_denom_done[1]);
                            }
                        }
                        nv_denom_consumer_phase ^= 1;
                    }
                } else if constexpr (
                    C::HAO_DIRECT_MX_DENOM_EARLY_SPLIT) {
                    #pragma unroll 1
                    for (int n = 0; n < C::N_TILES; ++n) {
                        hao_direct_nv_denom_half_snapshot first0;
                        hao_direct_nv_denom_half_snapshot first1;
                        hao_direct_nv_denom_half_snapshot tail0;
                        hao_direct_nv_denom_half_snapshot tail1;

                        hao_direct_fp4pv_wait(
                            p_first_o_rescaled[0],
                            nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot<0, C>(
                            first0, p0, S0, row_group);
                        tensor_load_wait();
                        represented_row_sum[0] +=
                            hao_direct_sum_mx_denom_snapshot<0, C>(
                                first0);

                        hao_direct_fp4pv_wait(
                            p_tail[0], nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot<1, C>(
                            tail0, p0, S0, row_group);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(nv_denom_loaded[0]);
                        }
                        represented_row_sum[0] +=
                            hao_direct_sum_mx_denom_snapshot<2, C>(
                                tail0);

                        hao_direct_fp4pv_wait(
                            p_first_o_rescaled[1],
                            nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot<0, C>(
                            first1, p1, S1, row_group);
                        tensor_load_wait();
                        represented_row_sum[1] +=
                            hao_direct_sum_mx_denom_snapshot<0, C>(
                                first1);

                        hao_direct_fp4pv_wait(
                            p_tail[1], nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot<1, C>(
                            tail1, p1, S1, row_group);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(nv_denom_loaded[1]);
                        }
                        represented_row_sum[1] +=
                            hao_direct_sum_mx_denom_snapshot<2, C>(
                                tail1);

                        if (n + 1 == C::N_TILES) {
                            const int row =
                                row_group * 32 + warp::laneid();
                            stats_store(
                                row_sum_smem[0],
                                row,
                                hao_direct_finalize_mx_denom<C>(
                                    represented_row_sum[0]));
                            stats_store(
                                row_sum_smem[1],
                                row,
                                hao_direct_finalize_mx_denom<C>(
                                    represented_row_sum[1]));
                            __threadfence_block();
                            if (warp::elect_leader()) {
                                arrive(nv_denom_done[0]);
                                arrive(nv_denom_done[1]);
                            }
                        }
                        nv_denom_consumer_phase ^= 1;
                    }
                } else if constexpr (
                    C::HAO_DIRECT_MX_DENOM_PIPELINED_SUM &&
                    !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                    hao_direct_nv_denom_half_snapshot first0;
                    hao_direct_nv_denom_half_snapshot first1;
                    hao_direct_nv_denom_half_snapshot tail0;
                    hao_direct_nv_denom_half_snapshot tail1;

                    // Prime stage 0. Each later load carries arithmetic from
                    // the preceding stage in its TMEM latency window.
                    hao_direct_fp4pv_wait(
                        p_tail[0], nv_denom_consumer_phase);
                    tensor_after_thread_sync();
                    hao_direct_load_mx_denom_snapshot_pair<C>(
                        first0, tail0, p0, p_sc_smem[0], S0, row_group);
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(nv_denom_loaded[0]);
                    }

                    #pragma unroll 1
                    for (int n = 0; n < C::N_TILES; ++n) {
                        hao_direct_fp4pv_wait(
                            p_tail[1], nv_denom_consumer_phase);
                        tensor_after_thread_sync();
                        hao_direct_load_mx_denom_snapshot_pair<C>(
                            first1, tail1, p1, p_sc_smem[1], S1, row_group);
                        hao_direct_accumulate_mx_denom_snapshot_pair<C>(
                            represented_row_sum[0], first0, tail0);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(nv_denom_loaded[1]);
                        }

                        const bool has_next = n + 1 < C::N_TILES;
                        if (has_next) {
                            nv_denom_consumer_phase ^= 1;
                            hao_direct_fp4pv_wait(
                                p_tail[0], nv_denom_consumer_phase);
                            tensor_after_thread_sync();
                            hao_direct_load_mx_denom_snapshot_pair<C>(
                                first0,
                                tail0,
                                p0,
                                p_sc_smem[0],
                                S0,
                                row_group);
                        }
                        hao_direct_accumulate_mx_denom_snapshot_pair<C>(
                            represented_row_sum[1], first1, tail1);
                        if (has_next) {
                            tensor_load_wait();
                            tensor_before_thread_sync();
                            if (warp::elect_leader()) {
                                arrive(nv_denom_loaded[0]);
                            }
                        }
                    }
                    nv_denom_consumer_phase ^= 1;

                    const int row =
                        row_group * 32 + warp::laneid();
                    stats_store(
                        row_sum_smem[0],
                        row,
                        hao_direct_finalize_mx_denom<C>(
                            represented_row_sum[0]));
                    stats_store(
                        row_sum_smem[1],
                        row,
                        hao_direct_finalize_mx_denom<C>(
                            represented_row_sum[1]));
                    __threadfence_block();
                    if (warp::elect_leader()) {
                        arrive(nv_denom_done[0]);
                        arrive(nv_denom_done[1]);
                    }
                } else {
                #pragma unroll 1
                for (int n = 0; n < C::N_TILES; ++n) {
                    if constexpr (
                        C::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD) {
                        auto publish_q1_scale = [&]<int ScaleStage>(
                            const Score &score,
                            int query_tile
                        ) {
                            hao_direct_fp4pv_wait(
                                score_full[ScaleStage],
                                q1_scale_score_phase);
                            tensor_after_thread_sync();

                            score_quarter_regs q1_scores;
                            upstream_load_score_quarter_x32<1, C, false>(
                                q1_scores,
                                score.addr +
                                    (row_group * 32 << 16),
                                n,
                                query_tile);
                            tensor_load_wait();
                            tensor_before_thread_sync();
                            if (warp::elect_leader()) {
                                arrive(sfqk_load[1 - ScaleStage]);
                            }

                            float q1_max =
                                base_types::constants<float>::neg_infty();
                            upstream_reduce_score_quarter<C>(
                                q1_scores, q1_max);
                            q1_scale_smem[ScaleStage].raw_max[
                                row_group * 32 + warp::laneid()] = q1_max;
                            __threadfence_block();
                            if (warp::elect_leader()) {
                                arrive(q1_scale_full[ScaleStage]);
                            }
                        };
                        publish_q1_scale.template operator()<0>(
                            score0, task.qtile0);
                        publish_q1_scale.template operator()<1>(
                            score1, task.qtile1);
                        q1_scale_score_phase ^= 1;
                    }
                    hao_direct_nv_denom_half_snapshot first0;
                    hao_direct_nv_denom_half_snapshot first1;
                    hao_direct_nv_denom_half_snapshot tail0;
                    hao_direct_nv_denom_half_snapshot tail1;
                    hao_direct_fp4pv_wait(
                        p_tail[0], nv_denom_consumer_phase);
                    if constexpr (
                        !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                        if constexpr (
                            C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM) {
                            hao_direct_load_mx_denom_payload_smem(
                                first0,
                                tail0,
                                mx_denom_payload_smem[0],
                                row_group);
                            const int row =
                                row_group * 32 + warp::laneid();
                            const uint32_t packed_scale =
                                fp4pv_load_mxfp4_scale_word_swizzled(
                                    p_sc_smem[0], row);
                            first0.scale_half = packed_scale;
                            tail0.scale_half = packed_scale;
                        } else {
                            tensor_after_thread_sync();
                            if constexpr (
                            C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF) {
                                hao_direct_load_nv_first_scale(
                                    first0,
                                    S0,
                                    row_group);
                            } else if constexpr (
                                C::HAO_DIRECT_MX_DENOM_CORR_WG) {
                                hao_direct_load_mx_denom_snapshot_pair<C>(
                                    first0,
                                    tail0,
                                    p0,
                                    p_sc_smem[0],
                                    S0,
                                    row_group);
                            } else {
                                hao_direct_load_nv_denom_snapshot<0>(
                                    first0, p0, S0, row_group);
                            }
                            if constexpr (
                                !C::HAO_DIRECT_MX_DENOM_CORR_WG) {
                                hao_direct_load_nv_denom_snapshot<1>(
                                    tail0, p0, S0, row_group);
                            }
                            tensor_load_wait();
                            tensor_before_thread_sync();
                            if constexpr (
                            C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF) {
                                // This release also permits the producer to overwrite
                                // the mirrored Q0/Q1 payload on the next score tile.
                                hao_direct_load_nv_first_payload_smem(
                                    first0,
                                    nv_first_payload_smem[0],
                                    row_group);
                            }
                        }
                    }
                    if (warp::elect_leader()) {
                        arrive(nv_denom_loaded[0]);
                    }

                    hao_direct_fp4pv_wait(
                        p_tail[1], nv_denom_consumer_phase);
                    if constexpr (
                        !C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                        if constexpr (
                            C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM) {
                            hao_direct_load_mx_denom_payload_smem(
                                first1,
                                tail1,
                                mx_denom_payload_smem[1],
                                row_group);
                            const int row =
                                row_group * 32 + warp::laneid();
                            const uint32_t packed_scale =
                                fp4pv_load_mxfp4_scale_word_swizzled(
                                    p_sc_smem[1], row);
                            first1.scale_half = packed_scale;
                            tail1.scale_half = packed_scale;
                        } else {
                            tensor_after_thread_sync();
                            if constexpr (
                            C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF) {
                                hao_direct_load_nv_first_scale(
                                    first1,
                                    S1,
                                    row_group);
                            } else if constexpr (
                                C::HAO_DIRECT_MX_DENOM_CORR_WG) {
                                hao_direct_load_mx_denom_snapshot_pair<C>(
                                    first1,
                                    tail1,
                                    p1,
                                    p_sc_smem[1],
                                    S1,
                                    row_group);
                            } else {
                                hao_direct_load_nv_denom_snapshot<0>(
                                    first1, p1, S1, row_group);
                            }
                            if constexpr (
                                !C::HAO_DIRECT_MX_DENOM_CORR_WG) {
                                hao_direct_load_nv_denom_snapshot<1>(
                                    tail1, p1, S1, row_group);
                            }
                            if constexpr (
                                C::HAO_DIRECT_MX_DENOM_PAIR_OVERLAP) {
                                represented_row_sum[0] +=
                                    hao_direct_sum_mx_denom_snapshot<0, C>(
                                        first0);
                                represented_row_sum[0] +=
                                    hao_direct_sum_mx_denom_snapshot<2, C>(
                                        tail0);
                            }
                            tensor_load_wait();
                            tensor_before_thread_sync();
                            if constexpr (
                            C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF) {
                                hao_direct_load_nv_first_payload_smem(
                                    first1,
                                    nv_first_payload_smem[1],
                                    row_group);
                            }
                        }
                    }
                    if (warp::elect_leader()) {
                        arrive(nv_denom_loaded[1]);
                    }
                    if constexpr (
                        C::HAO_DIRECT_FAKE_DENOM_CEILING) {
                    } else if constexpr (
                        C::HAO_DIRECT_MX_DENOM_CORR_WG) {
                        if constexpr (
                            !C::HAO_DIRECT_MX_DENOM_PAIR_OVERLAP) {
                            hao_direct_accumulate_mx_denom_snapshot_pair<C>(
                                represented_row_sum[0], first0, tail0);
                        }
                        hao_direct_accumulate_mx_denom_snapshot_pair<C>(
                            represented_row_sum[1], first1, tail1);
                    } else {
                        represented_row_sum[0] +=
                            hao_direct_sum_nv_denom_snapshot<
                                0,
                                C::HAO_DIRECT_NV_DENOM_WORDS,
                                C::HAO_DIRECT_NV_DENOM_KNOWN_MAX>(first0);
                        represented_row_sum[0] +=
                            hao_direct_sum_nv_denom_snapshot<
                                2,
                                C::HAO_DIRECT_NV_DENOM_WORDS,
                                C::HAO_DIRECT_NV_DENOM_KNOWN_MAX>(tail0);
                        represented_row_sum[1] +=
                            hao_direct_sum_nv_denom_snapshot<
                                0,
                                C::HAO_DIRECT_NV_DENOM_WORDS,
                                C::HAO_DIRECT_NV_DENOM_KNOWN_MAX>(first1);
                        represented_row_sum[1] +=
                            hao_direct_sum_nv_denom_snapshot<
                                2,
                                C::HAO_DIRECT_NV_DENOM_WORDS,
                                C::HAO_DIRECT_NV_DENOM_KNOWN_MAX>(tail1);
                    }
                    if (n + 1 == C::N_TILES) {
                        const int row =
                            row_group * 32 + warp::laneid();
                        stats_store(
                            row_sum_smem[0],
                            row,
                            hao_direct_finalize_mx_denom<C>(
                                represented_row_sum[0]));
                        stats_store(
                            row_sum_smem[1],
                            row,
                            hao_direct_finalize_mx_denom<C>(
                                represented_row_sum[1]));
                        __threadfence_block();
                        if (warp::elect_leader()) {
                            arrive(nv_denom_done[0]);
                            arrive(nv_denom_done[1]);
                        }
                    }
                    nv_denom_consumer_phase ^= 1;
                }
                }
            }

            if constexpr (C::HAO_DIRECT_NV_Q01_CORR_WG) {
                const float scale_log2 =
                    qk_scale_log2(g, task.batch, task.head);
                hao_direct_fp4pv_wait(
                    q3_sum_empty,
                    q01_empty_phase);
                q01_empty_phase ^= 1;

                float q01_row_sum = 0.0f;
                #pragma unroll 1
                for (int n = 0; n < C::N_TILES; ++n) {
                    hao_direct_fp4pv_wait(
                        score_full[0],
                        q01_score_phase);
                    tensor_after_thread_sync();

                    score_quarter_regs q0_scores;
                    score_quarter_regs q1_scores;
                    upstream_load_score_quarter_x32<0, C, false>(
                        q0_scores,
                        score0.addr + (row_group * 32 << 16),
                        n,
                        task.qtile0);
                    upstream_load_score_quarter_x32<1, C, false>(
                        q1_scores,
                        score0.addr + (row_group * 32 << 16),
                        n,
                        task.qtile0);
                    tensor_load_wait();

                    // The borrowed scale area may be reused once Q0/Q1 are
                    // resident. P cannot overlay Q2 until its owner confirms
                    // that both tail quarters have also been retained.
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(sfqk_load[1]);
                    }
                    hao_direct_fp4pv_wait(
                        dual_q2_ready[0],
                        q01_ready_phase);

                    float quarter_max =
                        base_types::constants<float>::neg_infty();
                    upstream_reduce_score_quarter<C>(
                        q0_scores, quarter_max);
                    float q01_sum =
                        hao_direct_transform_pack_fp4_quarter<
                            0,
                            C,
                            (C::HAO_DIRECT_STAGE0_AFFINE_MASK &
                             0x1u) != 0>(
                                p0,
                                p_sc_smem[0],
                                row_group,
                                q0_scores,
                                0.0f,
                                scale_log2,
                                quarter_max);

                    quarter_max =
                        base_types::constants<float>::neg_infty();
                    upstream_reduce_score_quarter<C>(
                        q1_scores, quarter_max);
                    q01_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            1,
                            C,
                            (C::HAO_DIRECT_STAGE0_AFFINE_MASK &
                             0x2u) != 0>(
                                p0,
                                p_sc_smem[0],
                                row_group,
                                q1_scores,
                                0.0f,
                                scale_log2,
                                quarter_max);
                    hao_direct_store_nvfp4_p_scale_half_direct<0>(
                        score0.addr + 32,
                        p_sc_smem[0],
                        row_group);
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(p_first_o_rescaled[0]);
                    }

                    q01_row_sum += q01_sum;
                    q01_score_phase ^= 1;
                    q01_ready_phase ^= 1;
                }

                stats_store(
                    correction_smem[0],
                    warpgroup::laneid(),
                    q01_row_sum);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(q3_sum_full);
                }
            }

            if constexpr (
                C::HAO_DIRECT_MX_DUAL_Q3_CORR_WG ||
                C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG ||
                C::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG) {
                const float scale_log2 =
                    qk_scale_log2(g, task.batch, task.head);
                hao_direct_fp4pv_wait(
                    dual_q3_sum_empty[0],
                    dual_q3_empty_phase);
                hao_direct_fp4pv_wait(
                    dual_q3_sum_empty[1],
                    dual_q3_empty_phase);
                dual_q3_empty_phase ^= 1;

                float q3_row_sum0 = 0.0f;
                float q3_row_sum1 = 0.0f;
                #pragma unroll 1
                for (int n = 0; n < C::N_TILES; ++n) {
                    score_quarter_regs q3_scores;
                    float q3_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (
                        C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG) {
                        hao_direct_fp4pv_wait(
                            dual_q3_staged[0],
                            dual_q3_score_phase);
                        hao_direct_load_q3_fp8_smem<C>(
                            q3_scores,
                            q3_stage_smem[0],
                            row_group);
                    } else if constexpr (
                        C::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG) {
                        hao_direct_fp4pv_wait(
                            dual_q3_staged[0],
                            dual_q3_score_phase);
                        tensor_after_thread_sync();
                        hao_direct_q3_fp8_words packed_q3;
                        hao_direct_load_q3_fp8_tmem(
                            packed_q3, p0, row_group);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(dual_q3_checkpoint_empty[0]);
                        }
                        hao_direct_unpack_q3_fp8_words(
                            q3_scores, packed_q3);
                    } else {
                        hao_direct_fp4pv_wait(
                            score_full[0],
                            dual_q3_score_phase);
                        tensor_after_thread_sync();
                        upstream_load_score_quarter_x32<3, C, false>(
                            q3_scores,
                            score0.addr + (row_group * 32 << 16),
                            n,
                            task.qtile0);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(sfqk_load[1]);
                        }
                    }
                    upstream_reduce_score_quarter<C>(
                        q3_scores,
                        q3_max);
                    q3_row_sum0 +=
                        hao_direct_transform_pack_fp4_quarter<3, C>(
                            p0,
                            p_sc_smem[0],
                            row_group,
                            q3_scores,
                            0.0f,
                            scale_log2,
                            q3_max);

                    q3_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (
                        C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG) {
                        hao_direct_fp4pv_wait(
                            dual_q3_staged[1],
                            dual_q3_score_phase);
                        hao_direct_load_q3_fp8_smem<C>(
                            q3_scores,
                            q3_stage_smem[1],
                            row_group);
                    } else if constexpr (
                        C::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG) {
                        hao_direct_fp4pv_wait(
                            dual_q3_staged[1],
                            dual_q3_score_phase);
                        tensor_after_thread_sync();
                        hao_direct_q3_fp8_words packed_q3;
                        hao_direct_load_q3_fp8_tmem(
                            packed_q3, p1, row_group);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(dual_q3_checkpoint_empty[1]);
                        }
                        hao_direct_unpack_q3_fp8_words(
                            q3_scores, packed_q3);
                    } else {
                        hao_direct_fp4pv_wait(
                            score_full[1],
                            dual_q3_score_phase);
                        tensor_after_thread_sync();
                        upstream_load_score_quarter_x32<3, C, false>(
                            q3_scores,
                            score1.addr + (row_group * 32 << 16),
                            n,
                            task.qtile1);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(sfqk_load[0]);
                        }
                    }

                    // Both score leases are now released. Publish stage 0's
                    // tail before spending the independent arithmetic window
                    // on the already-resident stage-1 Q3.
                    hao_direct_fp4pv_wait(
                        dual_q2_ready[0],
                        dual_q2_phase);
                    hao_direct_store_mxfp4_p_scale_half_direct<
                        1,
                        !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                        C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                            score0.addr + 48,
                            p_sc_smem[0],
                            row_group);
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(p_tail[0]);
                    }

                    upstream_reduce_score_quarter<C>(
                        q3_scores,
                        q3_max);
                    q3_row_sum1 +=
                        hao_direct_transform_pack_fp4_quarter<3, C>(
                            p1,
                            p_sc_smem[1],
                            row_group,
                            q3_scores,
                            0.0f,
                            scale_log2,
                            q3_max);

                    hao_direct_fp4pv_wait(
                        dual_q2_ready[1],
                        dual_q2_phase);
                    hao_direct_store_mxfp4_p_scale_half_direct<
                        1,
                        !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                        C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                            score1.addr + 48,
                            p_sc_smem[1],
                            row_group);
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(p_tail[1]);
                    }

                    dual_q3_score_phase ^= 1;
                    dual_q2_phase ^= 1;
                }

                stats_store(
                    row_sum_smem[0],
                    warpgroup::laneid(),
                    q3_row_sum0);
                stats_store(
                    row_sum_smem[1],
                    warpgroup::laneid(),
                    q3_row_sum1);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(dual_q3_sum_full[0]);
                    arrive(dual_q3_sum_full[1]);
                }
            }

            if constexpr (C::HAO_DIRECT_MX_Q3_CORR_WG) {
                const float scale_log2 =
                    qk_scale_log2(g, task.batch, task.head);
                hao_direct_fp4pv_wait(
                    q3_sum_empty,
                    q3_empty_phase);
                q3_empty_phase ^= 1;
                float q3_row_sum = 0.0f;
                #pragma unroll 1
                for (int n = 0; n < C::N_TILES; ++n) {
                    hao_direct_fp4pv_wait(
                        score_full[0],
                        q3_score_phase);
                    tensor_after_thread_sync();

                    score_quarter_regs q3_scores;
                    float q3_max =
                        base_types::constants<float>::neg_infty();
                    const uint32_t score_row_base =
                        score0.addr + (row_group * 32 << 16);
                    upstream_load_score_quarter_x32<3, C>(
                        q3_scores,
                        score_row_base,
                        n,
                        task.qtile0);
                    upstream_reduce_score_quarter<C>(
                        q3_scores,
                        q3_max);
                    const float q3_sum =
                        hao_direct_transform_pack_fp4_quarter<3, C>(
                            p0,
                            p_sc_smem[0],
                            row_group,
                            q3_scores,
                            0.0f,
                            scale_log2,
                            q3_max);
                    q3_row_sum += q3_sum;

                    hao_direct_fp4pv_wait(
                        sfqk_load[1],
                        q3_sfqk_phase);
                    hao_direct_store_mxfp4_p_scale_half_direct<
                        1,
                        !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                        C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                            score0.addr + 48,
                            p_sc_smem[0],
                            row_group);
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(p_tail[0]);
                    }
                    q3_score_phase ^= 1;
                    q3_sfqk_phase ^= 1;
                }
                stats_store(
                    correction_smem[0],
                    warpgroup::laneid(),
                    q3_row_sum);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(q3_sum_full);
                }
            }

            if constexpr (
                !C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                hao_direct_fp4pv_wait(
                    softmax_corr_full[0],
                    softmax_consumer_phase);
                if (warp::elect_leader()) {
                    arrive(softmax_corr_empty[0]);
                }
                hao_direct_fp4pv_wait(
                    softmax_corr_full[1],
                    softmax_consumer_phase);
                softmax_consumer_phase ^= 1;

                #pragma unroll 1
                for (int n = 0; n + 1 < C::N_TILES; ++n) {
                    #pragma unroll
                    for (int stage = 0; stage < 2; ++stage) {
                        hao_direct_fp4pv_wait(
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
            }

            #pragma unroll
            for (int stage = 0; stage < 2; ++stage) {
                hao_direct_fp4pv_wait(
                    softmax_corr_full[stage],
                    softmax_consumer_phase);
                if (warp::elect_leader()) {
                    arrive(softmax_corr_empty[stage]);
                }
                hao_direct_fp4pv_wait(
                    output_full[stage],
                    output_consumer_phase);
                hao_direct_fp4pv_wait(
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
                    if constexpr (
                        C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                        arrive(output_empty[stage]);
                    } else {
                        arrive(p_first_o_rescaled[stage]);
                    }
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
        warpgroup::decrease_registers<C::AUXILIARY_REGS>();
        int epi_consumer_phase = 0;
        for (int logical_job = int(blockIdx.x);
             logical_job < C::LOGICAL_JOBS;
             logical_job += int(gridDim.x)) {
            const hao_direct_task task =
                hao_direct_decode_task<C>(logical_job);
            hao_direct_fp4pv_wait(
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
            hao_direct_fp4pv_wait(
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
    } else {
        warpgroup::decrease_registers<C::EMPTY_REGS>();
    }
}

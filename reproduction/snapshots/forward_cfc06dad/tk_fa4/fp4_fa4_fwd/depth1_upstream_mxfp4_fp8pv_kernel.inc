#pragma once

// Isolated HAO-style NVFP4-QK + dense-E4M3-PV pipeline.  It preserves the
// retained depth-four issue-order ownership shell but removes every P/V
// block-scale data path.

template <typename C>
struct globals_fp4pv_upstream_fp8pv {
    using Parent = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename Parent::q_tile;
    using q_sc_tile = typename Parent::q_sc_tile;
    // The HAO topology always stages a complete K tile. Keep this descriptor
    // local instead of depending on experimental fwd_configs.inc traits.
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using k_sc_tile = typename Parent::k_sc_tile;
    using o_tile = typename Parent::o_tile;
    using q_gl = typename Parent::q_gl;
    using q_sc_gl = typename Parent::q_sc_gl;
    using q_sg_gl = typename Parent::q_sg_gl;
    using k_gl = gl<
        fp4e2m1_2, -1, -1, -1, -1,
        tma::descriptor<k_tile, dim::ROW>>;
    using k_sc_gl = typename Parent::k_sc_gl;
    using k_sg_gl = typename Parent::k_sg_gl;
    using o_gl = typename Parent::o_gl;
    using lse_gl = typename Parent::lse_gl;
    using v_fp8_tile = st_fp8e4m3<C::Dvo, C::Nb, true>;
    using v_fp8_gl = gl<
        fp8e4m3, -1, -1, -1, -1,
        tma::descriptor<v_fp8_tile, dim::ROW>>;

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
    const globals_fp4pv_upstream_fp8pv<C> &g, int b, int h
) {
    static_assert(C::Dqk == 128);
    constexpr float LOG2E_OVER_SQRT_D = 0.1275170599133291f;
    float q_scale = 0.0f;
    float k_scale = 0.0f;
    if (warp::laneid() == 0) {
        q_scale = g.q_sg[{b, h, 0, 0}];
        k_scale = g.k_sg[{b, h, 0, 0}];
    }
    q_scale = __shfl_sync(0xFFFFFFFFu, q_scale, 0);
    k_scale = __shfl_sync(0xFFFFFFFFu, k_scale, 0);
    return LOG2E_OVER_SQRT_D * q_scale * k_scale;
}

template <typename C>
union upstream_mxfp4_kv_payload_slot {
    using G = globals_fp4pv_upstream_fp8pv<C>;
    typename G::k_tile k;
    typename G::v_fp8_tile v;
};

template <typename C>
struct upstream_mxfp4_k_scale_slot {
    using G = globals_fp4pv_upstream_fp8pv<C>;
    typename G::k_sc_tile k;
};

template <typename C>
struct upstream_mxfp4_pipeline_smem_layout {
    using G = globals_fp4pv_upstream_fp8pv<C>;
    using Payload = upstream_mxfp4_kv_payload_slot<C>;
    using Scale = upstream_mxfp4_k_scale_slot<C>;
    static constexpr int ALIGNMENT = 1024;
    static constexpr int align_up(int offset) {
        return ((offset + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }
    static constexpr int append(int offset, int bytes) {
        return align_up(offset) + bytes;
    }
    static constexpr int bytes_for_depth(int depth) {
        int offset = 0;
        offset = append(offset, 2 * int(sizeof(typename G::q_tile)));
        offset = append(offset, 2 * int(sizeof(typename G::q_sc_tile)));
        offset = append(offset, depth * int(sizeof(Payload)));
        offset = append(offset, depth * int(sizeof(Scale)));
        offset = append(offset, int(sizeof(typename G::o_tile)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        offset = append(offset, 2 * int(sizeof(sv_fl<C::Mb>)));
        // Extern SMEM itself is not guaranteed to start 1024-byte aligned.
        // Match the established FA4 convention with one conservative guard.
        return offset + ALIGNMENT;
    }
    static constexpr int bytes() {
        return bytes_for_depth(C::KV_STAGES);
    }
};

template <int Quarter, int Slice, typename C, bool Wait = true>
__device__ __forceinline__ void upstream_load_score_sixteenth_tmem(
    float2 *scores, uint32_t score_row_base,
    int key_tile, int query_tile);

template <typename PTT>
__device__ __forceinline__ void upstream_store_p_fp8_word_tmem(
    PTT &p,
    int reader_warp,
    int quarter,
    int pack,
    uint32_t word
) {
    const uint32_t addr =
        p.addr + (reader_warp * 32 << 16) + quarter * 8 + pack;
    asm volatile(
        "{tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};}"
        :: "r"(addr), "r"(word));
}

__device__ __forceinline__ uint32_t upstream_pack_fp8_four(
    const float2 &lo, const float2 &hi
) {
    const float4 values = {lo.x, lo.y, hi.x, hi.y};
    const fp8e4m3_4 packed =
        base_types::convertor<fp8e4m3_4, float4>::convert(values);
    return std::bit_cast<uint32_t>(packed);
}

template <typename PTT>
__device__ __forceinline__ void upstream_store_p_fp8_quarter_tmem(
    PTT &p, int reader_warp, int quarter, const uint32_t (&words)[8]) {
    const uint32_t addr = p.addr + (reader_warp * 32 << 16) + quarter * 8;
    asm volatile("{tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};}"
        :: "r"(addr), "r"(words[0]), "r"(words[1]), "r"(words[2]),
           "r"(words[3]), "r"(words[4]), "r"(words[5]), "r"(words[6]),
           "r"(words[7]));
}

struct score_quarter_regs { float2 v0; float2 v1; float2 v2; float2 v3; float2 v4; float2 v5; float2 v6; float2 v7; float2 v8; float2 v9; float2 v10; float2 v11; float2 v12; float2 v13; float2 v14; float2 v15; };

template <int Quarter, int Slice, typename C, typename P>
__device__ __forceinline__ float upstream_transform_pack_fp8_sixteenth(
    P &p,
    int reader_warp,
    float2 (&scores)[4],
    float row_max,
    float scale_log2, uint32_t &word0, uint32_t &word1
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    static_assert(Slice >= 0 && Slice < 4);
    const float2 mul = {scale_log2, scale_log2};
    const float bias_scalar = -row_max * scale_log2;
    const float2 bias = {bias_scalar, bias_scalar};
    float2 sum = {0.0f, 0.0f};
    fp4pv_stage2_static_for<0, 4>([&](auto element_tag) {
        constexpr int Element = decltype(element_tag)::value;
        constexpr int PairIndex = Quarter * 16 + Slice * 4 + Element;
        scores[Element] = __ffma2_rn(scores[Element], mul, bias);
        scores[Element] = fp4pv_stage2_exp2_pair<
            C::ONLINE_STAGE2_EX2_EMU_PERIOD,
            C::ONLINE_STAGE2_EX2_EMU_MASK,
            PairIndex>(scores[Element]);
        sum = __fadd2_rn(sum, scores[Element]);
    });
    word0 = upstream_pack_fp8_four(scores[0], scores[1]);
    word1 = upstream_pack_fp8_four(scores[2], scores[3]);
    return sum.x + sum.y;
}

template <int Quarter, int Slice, typename C, typename P>
__device__ __forceinline__ float upstream_reread_pack_fp8_sixteenth(
    P &p,
    int reader_warp,
    uint32_t score_row_base,
    int key_tile,
    int query_tile,
    float row_max,
    float scale_log2
) {
    float2 scores[4];
    upstream_load_score_sixteenth_tmem<Quarter, Slice, C>(
        scores, score_row_base, key_tile, query_tile);
    uint32_t w0, w1;
    return upstream_transform_pack_fp8_sixteenth<Quarter, Slice, C>(
        p, reader_warp, scores, row_max, scale_log2, w0, w1);
}

template <int Quarter, typename C, typename P>
__device__ __forceinline__ float upstream_reread_pack_fp8_quarter(
    P &p, int reader_warp, uint32_t score_row_base, int key_tile,
    int query_tile, float row_max, float scale_log2) {
    static_assert(Quarter == 0 || Quarter == 1 || Quarter == 3);
    uint32_t words[8]; uint32_t w0, w1; float2 scores[4]; float sum = 0.0f;
#define UPSTREAM_QSLICE(S) \
    upstream_load_score_sixteenth_tmem<Quarter,S,C>(scores,score_row_base,key_tile,query_tile); \
    sum += upstream_transform_pack_fp8_sixteenth<Quarter,S,C>(p,reader_warp,scores,row_max,scale_log2,w0,w1); \
    words[2*S]=w0; words[2*S+1]=w1;
    UPSTREAM_QSLICE(0); UPSTREAM_QSLICE(1); UPSTREAM_QSLICE(2); UPSTREAM_QSLICE(3);
#undef UPSTREAM_QSLICE
    upstream_store_p_fp8_quarter_tmem(p,reader_warp,Quarter,words);
    return sum;
}

template <int Quarter, int Slice, typename C>
__device__ __forceinline__ float upstream_transform_pack_fp8_four(
    float2 &s0, float2 &s1, float2 &s2, float2 &s3,
    float row_max, float scale_log2, uint32_t &word0, uint32_t &word1) {
    const float2 mul = {scale_log2, scale_log2};
    const float2 bias = {-row_max * scale_log2, -row_max * scale_log2};
    s0 = fp4pv_stage2_exp2_pair<C::ONLINE_STAGE2_EX2_EMU_PERIOD, C::ONLINE_STAGE2_EX2_EMU_MASK, Quarter*16 + Slice*4 + 0>(__ffma2_rn(s0,mul,bias));
    s1 = fp4pv_stage2_exp2_pair<C::ONLINE_STAGE2_EX2_EMU_PERIOD, C::ONLINE_STAGE2_EX2_EMU_MASK, Quarter*16 + Slice*4 + 1>(__ffma2_rn(s1,mul,bias));
    s2 = fp4pv_stage2_exp2_pair<C::ONLINE_STAGE2_EX2_EMU_PERIOD, C::ONLINE_STAGE2_EX2_EMU_MASK, Quarter*16 + Slice*4 + 2>(__ffma2_rn(s2,mul,bias));
    s3 = fp4pv_stage2_exp2_pair<C::ONLINE_STAGE2_EX2_EMU_PERIOD, C::ONLINE_STAGE2_EX2_EMU_MASK, Quarter*16 + Slice*4 + 3>(__ffma2_rn(s3,mul,bias));
    word0 = upstream_pack_fp8_four(s0,s1);
    word1 = upstream_pack_fp8_four(s2,s3);
    return s0.x+s0.y+s1.x+s1.y+s2.x+s2.y+s3.x+s3.y;
}

template <int Quarter, typename C, typename P>
__device__ __forceinline__ float upstream_transform_pack_retained_quarter(
    P &p, int reader_warp, score_quarter_regs &scores,
    float row_max, float scale_log2) {
    uint32_t words[8]; float sum = 0.0f; uint32_t w0,w1;
    sum += upstream_transform_pack_fp8_four<Quarter,0,C>(scores.v0,scores.v1,scores.v2,scores.v3,row_max,scale_log2,w0,w1); words[0]=w0;words[1]=w1;
    sum += upstream_transform_pack_fp8_four<Quarter,1,C>(scores.v4,scores.v5,scores.v6,scores.v7,row_max,scale_log2,w0,w1); words[2]=w0;words[3]=w1;
    sum += upstream_transform_pack_fp8_four<Quarter,2,C>(scores.v8,scores.v9,scores.v10,scores.v11,row_max,scale_log2,w0,w1); words[4]=w0;words[5]=w1;
    sum += upstream_transform_pack_fp8_four<Quarter,3,C>(scores.v12,scores.v13,scores.v14,scores.v15,row_max,scale_log2,w0,w1); words[6]=w0;words[7]=w1;
    upstream_store_p_fp8_quarter_tmem(p,reader_warp,Quarter,words);
    return sum;
}

template <typename C, typename P>
__device__ __forceinline__ float upstream_reread_pack_fp8_quarter2(
    P &p,
    int reader_warp,
    uint32_t score_row_base,
    int key_tile,
    int query_tile,
    float row_max,
    float scale_log2
) {
    // P quarter 2 writes score columns 80..87.  Those columns are exactly
    // source slice 2 of score quarter 2, so acquire only that slice before any
    // dense-P store.  Slices 0/1 source columns 64..79 and slice 3 source
    // columns 88..95 remain disjoint from the quarter-2 destinations.
    float2 pending_slice2[4];
    upstream_load_score_sixteenth_tmem<2, 2, C>(
        pending_slice2, score_row_base, key_tile, query_tile);
    float sum = 0.0f;
    uint32_t words[8]; uint32_t w0, w1; float2 s[4];
    upstream_load_score_sixteenth_tmem<2,0,C>(s,score_row_base,key_tile,query_tile);
    sum += upstream_transform_pack_fp8_sixteenth<2,0,C>(p,reader_warp,s,row_max,scale_log2,w0,w1); words[0]=w0; words[1]=w1;
    upstream_load_score_sixteenth_tmem<2,1,C>(s,score_row_base,key_tile,query_tile);
    sum += upstream_transform_pack_fp8_sixteenth<2,1,C>(p,reader_warp,s,row_max,scale_log2,w0,w1); words[2]=w0; words[3]=w1;
    sum += upstream_transform_pack_fp8_sixteenth<2,2,C>(p,reader_warp,pending_slice2,row_max,scale_log2,w0,w1); words[4]=w0; words[5]=w1;
    upstream_load_score_sixteenth_tmem<2,3,C>(s,score_row_base,key_tile,query_tile);
    sum += upstream_transform_pack_fp8_sixteenth<2,3,C>(p,reader_warp,s,row_max,scale_log2,w0,w1); words[6]=w0; words[7]=w1;
    upstream_store_p_fp8_quarter_tmem(p,reader_warp,2,words);
    return sum;
}

// Dense E4M3 TS PV.  N128 uses four legal K32 f8f6f4 commands; P is the
// retired-score TMEM operand and V is the prequantized shared operand.
template <typename C, typename D, typename P, typename V>
__device__ __forceinline__ void upstream_issue_pv_fp8_tmem_n128(
    D &out, P &p, V &v, semaphore &done,
    bool accumulate
) {
    using T_AB = fp8e4m3;
    using T_D = typename D::T;
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, C::Mb, C::Dvo, false, false>();
    st_descriptor<V, 0> v_desc(v);
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        const uint32_t p_addr = p.template chunk_addr<0>(chunk);
        const uint64_t v_descriptor = v_desc.chunk_descriptor(chunk);
        const int chunk_accumulate = chunk == 0 ? int(accumulate) : 1;
        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.eq.u32 p, 1, %4;\n\t"
            "tcgen05.mma.cta_group::1.kind::f8f6f4 "
            "[%0], [%1], %2, %3, p;}\n"
            :: "r"(out.addr), "r"(p_addr), "l"(v_descriptor), "r"(IDESC),
               "r"(chunk_accumulate));
    }
    detail::tcgen05::commit<1>(done);
}

template <typename C, typename Output>
__device__ __forceinline__ void upstream_correct_output_stripe(
    Output &output, int row_group, float correction, bool clear
) {
    const uint32_t row_base =
        output.addr + (uint32_t(row_group * 32) << 16);
    const float2 scale = {correction, correction};
    #pragma unroll
    for (int col = 0; col < C::Dvo; col += 16) {
        float2 values[8];
        if (!clear) {
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
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) values[i] = {0.0f, 0.0f};
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) values[i] = __fmul2_rn(values[i], scale);
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
               "r"(row_base + col));
    }
    tensor_store_wait();
    tensor_after_thread_sync();
}

template <typename C, typename = void>
struct upstream_output_accumulator_scale {
    static constexpr float value = 1.0f;
};

template <typename C>
struct upstream_output_accumulator_scale<
    C,
    std::void_t<decltype(C::HAO_DIRECT_MXFP4_PV)>> {
    // E8M0 MXFP4 operands dequantize as payload * scale / 6. TCGEN
    // accumulates payload * scale for both P and V, so normalize once by 36.
    static constexpr float value =
        C::HAO_DIRECT_MXFP4_PV ? (1.0f / 36.0f) : 1.0f;
};

template <typename C, typename = void>
struct upstream_p_global_log2 {
    static constexpr float value = 0.0f;
};

template <typename C>
struct upstream_p_global_log2<
    C,
    std::void_t<decltype(C::HAO_DIRECT_NV_P_GLOBAL_LOG2)>> {
    static constexpr float value =
        C::HAO_DIRECT_NV_P_GLOBAL_LOG2;
};

template <typename C, typename G>
__device__ __forceinline__ float upstream_v_global_decode(
    const G &g
) {
    if constexpr (
        requires {
            C::HAO_DIRECT_NV_V_GLOBAL_DECODE;
            g.v_global_decode;
        }) {
        if constexpr (C::HAO_DIRECT_NV_V_GLOBAL_DECODE) {
            return g.v_global_decode;
        }
    }
    return 1.0f;
}

template <typename C, typename Output, typename OTile, typename G>
__device__ __forceinline__ void upstream_store_output_row_group_owned_fp8(
    Output output_owned, OTile &o_smem, const G &g,
    int batch, int query_tile_owned, int head,
    sv_fl<C::Mb> &row_max_owned,
    sv_fl<C::Mb> &row_sum_owned,
    int owned_row_group
) {
    constexpr int SUBTILE_COLS = OTile::swizzle_bytes / sizeof(bf16);
    constexpr uint32_t SWIZZLE_MASK = OTile::swizzle_bytes * 8 - 1;
    const uint32_t smem_base = __cvta_generic_to_shared(&o_smem.data[0]);
    const int row = owned_row_group * 32 + warp::laneid();
    const float sum = stats_load(row_sum_owned, row);
    const float mx = stats_load(row_max_owned, row);
    const bool invalid = !(sum > 0.0f) || !isfinite(sum) ||
                         !isfinite(mx);
    float inv_sum;
    asm volatile("rcp.approx.ftz.f32 %0, %1;"
                 : "=f"(inv_sum) : "f"(invalid ? 1.0f : sum));
    const float output_scale =
        invalid
            ? 0.0f
            : inv_sum *
                  upstream_output_accumulator_scale<C>::value *
                  upstream_v_global_decode<C>(g);
    const float2 scale = {output_scale, output_scale};
    #pragma unroll 1
    for (int col = 0; col < C::Dvo; col += 16) {
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
            : "r"(output_owned.addr +
                   (uint32_t(owned_row_group * 32) << 16) + col));
        tensor_load_wait();
        const uint32_t row_base = smem_base +
            (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16)) +
            row * SUBTILE_COLS * sizeof(bf16) +
            (col % SUBTILE_COLS) * sizeof(bf16);
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
            const bf16_2 a = __float22bfloat162_rn(__fmul2_rn(values[i], scale));
            const bf16_2 b = __float22bfloat162_rn(__fmul2_rn(values[i + 1], scale));
            const bf16_2 c = __float22bfloat162_rn(__fmul2_rn(values[i + 2], scale));
            const bf16_2 d = __float22bfloat162_rn(__fmul2_rn(values[i + 3], scale));
            const uint32_t addr = row_base + i * sizeof(bf16_2);
            const uint32_t swizzled =
                addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(swizzled),
                "r"(*reinterpret_cast<const uint32_t *>(&a)),
                "r"(*reinterpret_cast<const uint32_t *>(&b)),
                "r"(*reinterpret_cast<const uint32_t *>(&c)),
                "r"(*reinterpret_cast<const uint32_t *>(&d)));
        }
    }
    warp::sync();
    __threadfence_block();
    fp4pv_publish_shared_backing_proxy_only();
    if (g.disable_lse_store == 0) {
        if constexpr (
            requires { C::HAO_DIRECT_DEBUG_LSE_STAT; }) {
            if constexpr (C::HAO_DIRECT_DEBUG_LSE_STAT == 1) {
                g.lse[{
                    batch,
                    head,
                    0,
                    query_tile_owned * C::Mb + row}] = mx;
                return;
            } else if constexpr (
                C::HAO_DIRECT_DEBUG_LSE_STAT == 2) {
                g.lse[{
                    batch,
                    head,
                    0,
                    query_tile_owned * C::Mb + row}] = sum;
                return;
            }
        }
        if constexpr (
            requires { C::HAO_DIRECT_NV_DEBUG_LSE_ROWMAX; }) {
            if constexpr (C::HAO_DIRECT_NV_DEBUG_LSE_ROWMAX) {
                g.lse[{
                    batch,
                    head,
                    0,
                    query_tile_owned * C::Mb + row}] = mx;
                return;
            }
        }
        const float qk_scale = qk_scale_log2(g, batch, head);
        float lse = base_types::constants<float>::neg_infty();
        if (!invalid) {
            float log2_sum;
            asm("lg2.approx.ftz.f32 %0, %1;"
                : "=f"(log2_sum) : "f"(sum));
            constexpr float LN2 = 0.693147180559945f;
            lse = (
                mx * qk_scale +
                log2_sum -
                upstream_p_global_log2<C>::value) * LN2;
        }
        g.lse[{batch, head, 0, query_tile_owned * C::Mb + row}] = lse;
    }
}

template <typename OTile, typename G>
__device__ __forceinline__ void upstream_store_output_tma_one_warp(
    OTile &o_smem, const G &g, int batch, int query_tile_owned, int head
) {
    if (warp::elect_leader()) {
        warp::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
            g.o, o_smem, {batch, query_tile_owned, head, 0});
    }
    warp::tma::store_async_wait<0>();
}

template <int Quarter, int Slice, typename C, bool Wait>
__device__ __forceinline__ void upstream_load_score_sixteenth_tmem(
    float2 *scores,
    uint32_t score_row_base,
    int key_tile,
    int query_tile
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    static_assert(Slice >= 0 && Slice < 4);
    asm volatile(
        "{tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];}"
        : "=f"(scores[0].x), "=f"(scores[0].y),
          "=f"(scores[1].x), "=f"(scores[1].y),
          "=f"(scores[2].x), "=f"(scores[2].y),
          "=f"(scores[3].x), "=f"(scores[3].y)
        : "r"(score_row_base + Quarter * 32 + Slice * 8));
    if constexpr (Wait) {
        tensor_load_wait();
    }
    const int query_col =
        query_tile * C::Mb + static_cast<int>(warpgroup::laneid());
    const int tile_col_base =
        key_tile * C::Nb + Quarter * 32 + Slice * 8;
    constexpr bool Noncausal = [] {
        if constexpr (requires { C::UPSTREAM_NATIVE_NONCAUSAL; }) {
            return bool(C::UPSTREAM_NATIVE_NONCAUSAL);
        } else {
            return bool(C::HAO_DIRECT_NONCAUSAL);
        }
    }();
    if constexpr (!Noncausal) {
        if (tile_col_base + 7 > query_col) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col0 = tile_col_base + j * 2;
                if (col0 > query_col) {
                    scores[j].x = base_types::constants<float>::neg_infty();
                }
                if (col0 + 1 > query_col) {
                    scores[j].y = base_types::constants<float>::neg_infty();
                }
            }
        }
    }
}

template <int Pairs>
__device__ __forceinline__ void upstream_reduce_score_pairs(
    const float2 *scores,
    float &m
) {
    static_assert(Pairs == 4 || Pairs == 8);
    #pragma unroll
    for (int i = 0; i < Pairs; ++i) {
        m = fmaxf(m, scores[i].x);
        m = fmaxf(m, scores[i].y);
    }
}

template <typename C>
__device__ __forceinline__ void upstream_reduce_score_quarter(score_quarter_regs &s, float &m) {
    constexpr bool use_max3_reduce = [] {
        if constexpr (requires { C::HAO_DIRECT_P_MAX3_REDUCE; }) {
            if constexpr (
                requires { C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE; }) {
                return C::HAO_DIRECT_P_MAX3_REDUCE &&
                    C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE == 1;
            } else {
                return C::HAO_DIRECT_P_MAX3_REDUCE;
            }
        } else if constexpr (
            requires { C::HAO_DIRECT_NV_MAX3_REDUCE; }) {
            return C::HAO_DIRECT_NV_MAX3_REDUCE;
        } else {
            return false;
        }
    }();
    if constexpr (
        use_max3_reduce) {
        constexpr bool use_wide_max3_reduce = [] {
            if constexpr (
                requires { C::HAO_DIRECT_MX_MAX3_WIDE_REDUCE; }) {
                return C::HAO_DIRECT_MX_MAX3_WIDE_REDUCE;
            }
            return false;
        }();
        if constexpr (use_wide_max3_reduce) {
            float p0;
            float p1;
            float p2;
            float p3;
            float p4;
            float p5;
            float p6;
            float p7;
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p0) : "f"(m), "f"(s.v0.x), "f"(s.v0.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p1) : "f"(m), "f"(s.v2.x), "f"(s.v2.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p2) : "f"(m), "f"(s.v4.x), "f"(s.v4.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p3) : "f"(m), "f"(s.v6.x), "f"(s.v6.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p4) : "f"(m), "f"(s.v8.x), "f"(s.v8.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p5) : "f"(m), "f"(s.v10.x), "f"(s.v10.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p6) : "f"(m), "f"(s.v12.x), "f"(s.v12.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p7) : "f"(m), "f"(s.v14.x), "f"(s.v14.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p0) : "f"(p0), "f"(s.v1.x), "f"(s.v1.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p1) : "f"(p1), "f"(s.v3.x), "f"(s.v3.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p2) : "f"(p2), "f"(s.v5.x), "f"(s.v5.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p3) : "f"(p3), "f"(s.v7.x), "f"(s.v7.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p4) : "f"(p4), "f"(s.v9.x), "f"(s.v9.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p5) : "f"(p5), "f"(s.v11.x), "f"(s.v11.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p6) : "f"(p6), "f"(s.v13.x), "f"(s.v13.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p7) : "f"(p7), "f"(s.v15.x), "f"(s.v15.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p0) : "f"(p0), "f"(p1), "f"(p2));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p3) : "f"(p3), "f"(p4), "f"(p5));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p6) : "f"(m), "f"(p6), "f"(p7));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(m) : "f"(p0), "f"(p3), "f"(p6));
            return;
        }
        float p0;
            float p1;
            float p2;
            float p3;
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p0) : "f"(m), "f"(s.v0.x), "f"(s.v0.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p1) : "f"(m), "f"(s.v1.x), "f"(s.v1.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p2) : "f"(m), "f"(s.v2.x), "f"(s.v2.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p3) : "f"(m), "f"(s.v3.x), "f"(s.v3.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p0) : "f"(p0), "f"(s.v4.x), "f"(s.v4.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p1) : "f"(p1), "f"(s.v5.x), "f"(s.v5.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p2) : "f"(p2), "f"(s.v6.x), "f"(s.v6.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p3) : "f"(p3), "f"(s.v7.x), "f"(s.v7.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p0) : "f"(p0), "f"(s.v8.x), "f"(s.v8.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p1) : "f"(p1), "f"(s.v9.x), "f"(s.v9.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p2) : "f"(p2), "f"(s.v10.x), "f"(s.v10.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p3) : "f"(p3), "f"(s.v11.x), "f"(s.v11.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p0) : "f"(p0), "f"(s.v12.x), "f"(s.v12.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p1) : "f"(p1), "f"(s.v13.x), "f"(s.v13.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p2) : "f"(p2), "f"(s.v14.x), "f"(s.v14.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p3) : "f"(p3), "f"(s.v15.x), "f"(s.v15.y));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(p0) : "f"(p0), "f"(p1), "f"(p2));
            asm volatile(
                "{max.f32 %0, %1, %2, %3;}"
                : "=f"(m) : "f"(m), "f"(p0), "f"(p3));
            return;
    }
    if constexpr (
        requires { C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE; }) {
        if constexpr (C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE == 7) {
            m = fmaxf(m, fmaxf(s.v0.x, s.v8.x));
            return;
        }
        if constexpr (C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE == 6) {
            const float p0 = fmaxf(s.v0.x, s.v4.x);
            const float p1 = fmaxf(s.v8.x, s.v12.x);
            m = fmaxf(m, fmaxf(p0, p1));
            return;
        }
        if constexpr (C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE == 5) {
            m = fmaxf(m, s.v0.x);
            m = fmaxf(m, s.v0.y);
            m = fmaxf(m, s.v4.x);
            m = fmaxf(m, s.v4.y);
            m = fmaxf(m, s.v8.x);
            m = fmaxf(m, s.v8.y);
            m = fmaxf(m, s.v12.x);
            m = fmaxf(m, s.v12.y);
            return;
        }
        if constexpr (C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE == 4) {
            float p0 = fmaxf(s.v0.x, s.v0.y);
            float p1 = fmaxf(s.v1.x, s.v1.y);
            float p2 = fmaxf(s.v2.x, s.v2.y);
            float p3 = s.v3.x;
            float p4 = fmaxf(s.v4.x, s.v4.y);
            float p5 = fmaxf(s.v5.x, s.v5.y);
            float p6 = fmaxf(s.v6.x, s.v6.y);
            float p7 = s.v7.x;
            float p8 = fmaxf(s.v8.x, s.v8.y);
            float p9 = fmaxf(s.v9.x, s.v9.y);
            float p10 = fmaxf(s.v10.x, s.v10.y);
            float p11 = s.v11.x;
            float p12 = fmaxf(s.v12.x, s.v12.y);
            float p13 = fmaxf(s.v13.x, s.v13.y);
            float p14 = fmaxf(s.v14.x, s.v14.y);
            float p15 = s.v15.x;
            p0 = fmaxf(p0, p1);
            p2 = fmaxf(p2, p3);
            p4 = fmaxf(p4, p5);
            p6 = fmaxf(p6, p7);
            p8 = fmaxf(p8, p9);
            p10 = fmaxf(p10, p11);
            p12 = fmaxf(p12, p13);
            p14 = fmaxf(p14, p15);
            p0 = fmaxf(p0, p2);
            p4 = fmaxf(p4, p6);
            p8 = fmaxf(p8, p10);
            p12 = fmaxf(p12, p14);
            p0 = fmaxf(p0, p4);
            p8 = fmaxf(p8, p12);
            m = fmaxf(m, fmaxf(p0, p8));
            return;
        }
        if constexpr (C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE == 3) {
            float p0 = fmaxf(s.v0.x, s.v0.y);
            float p1 = s.v1.x;
            float p2 = fmaxf(s.v2.x, s.v2.y);
            float p3 = s.v3.x;
            float p4 = fmaxf(s.v4.x, s.v4.y);
            float p5 = s.v5.x;
            float p6 = fmaxf(s.v6.x, s.v6.y);
            float p7 = s.v7.x;
            float p8 = fmaxf(s.v8.x, s.v8.y);
            float p9 = s.v9.x;
            float p10 = fmaxf(s.v10.x, s.v10.y);
            float p11 = s.v11.x;
            float p12 = fmaxf(s.v12.x, s.v12.y);
            float p13 = s.v13.x;
            float p14 = fmaxf(s.v14.x, s.v14.y);
            float p15 = s.v15.x;
            p0 = fmaxf(p0, p1);
            p2 = fmaxf(p2, p3);
            p4 = fmaxf(p4, p5);
            p6 = fmaxf(p6, p7);
            p8 = fmaxf(p8, p9);
            p10 = fmaxf(p10, p11);
            p12 = fmaxf(p12, p13);
            p14 = fmaxf(p14, p15);
            p0 = fmaxf(p0, p2);
            p4 = fmaxf(p4, p6);
            p8 = fmaxf(p8, p10);
            p12 = fmaxf(p12, p14);
            p0 = fmaxf(p0, p4);
            p8 = fmaxf(p8, p12);
            m = fmaxf(m, fmaxf(p0, p8));
            return;
        }
        if constexpr (C::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE == 2) {
            float p0 = fmaxf(s.v0.x, s.v0.y);
            float p2 = fmaxf(s.v2.x, s.v2.y);
            float p4 = fmaxf(s.v4.x, s.v4.y);
            float p6 = fmaxf(s.v6.x, s.v6.y);
            float p8 = fmaxf(s.v8.x, s.v8.y);
            float p10 = fmaxf(s.v10.x, s.v10.y);
            float p12 = fmaxf(s.v12.x, s.v12.y);
            float p14 = fmaxf(s.v14.x, s.v14.y);
            p0 = fmaxf(p0, p2);
            p4 = fmaxf(p4, p6);
            p8 = fmaxf(p8, p10);
            p12 = fmaxf(p12, p14);
            p0 = fmaxf(p0, p4);
            p8 = fmaxf(p8, p12);
            m = fmaxf(m, fmaxf(p0, p8));
            return;
        }
    }
    if constexpr (requires { C::HAO_DIRECT_MX_TREE_MAX; }) {
        if constexpr (C::HAO_DIRECT_MX_TREE_MAX) {
            float p0 = fmaxf(s.v0.x, s.v0.y);
            float p1 = fmaxf(s.v1.x, s.v1.y);
            float p2 = fmaxf(s.v2.x, s.v2.y);
            float p3 = fmaxf(s.v3.x, s.v3.y);
            float p4 = fmaxf(s.v4.x, s.v4.y);
            float p5 = fmaxf(s.v5.x, s.v5.y);
            float p6 = fmaxf(s.v6.x, s.v6.y);
            float p7 = fmaxf(s.v7.x, s.v7.y);
            float p8 = fmaxf(s.v8.x, s.v8.y);
            float p9 = fmaxf(s.v9.x, s.v9.y);
            float p10 = fmaxf(s.v10.x, s.v10.y);
            float p11 = fmaxf(s.v11.x, s.v11.y);
            float p12 = fmaxf(s.v12.x, s.v12.y);
            float p13 = fmaxf(s.v13.x, s.v13.y);
            float p14 = fmaxf(s.v14.x, s.v14.y);
            float p15 = fmaxf(s.v15.x, s.v15.y);
            p0 = fmaxf(p0, p1);
            p2 = fmaxf(p2, p3);
            p4 = fmaxf(p4, p5);
            p6 = fmaxf(p6, p7);
            p8 = fmaxf(p8, p9);
            p10 = fmaxf(p10, p11);
            p12 = fmaxf(p12, p13);
            p14 = fmaxf(p14, p15);
            p0 = fmaxf(p0, p2);
            p4 = fmaxf(p4, p6);
            p8 = fmaxf(p8, p10);
            p12 = fmaxf(p12, p14);
            p0 = fmaxf(p0, p4);
            p8 = fmaxf(p8, p12);
            m = fmaxf(m, fmaxf(p0, p8));
            return;
        }
    }
    m = fmaxf(m, s.v0.x);
    m = fmaxf(m, s.v0.y);
    m = fmaxf(m, s.v1.x);
    m = fmaxf(m, s.v1.y);
    m = fmaxf(m, s.v2.x);
    m = fmaxf(m, s.v2.y);
    m = fmaxf(m, s.v3.x);
    m = fmaxf(m, s.v3.y);
    m = fmaxf(m, s.v4.x);
    m = fmaxf(m, s.v4.y);
    m = fmaxf(m, s.v5.x);
    m = fmaxf(m, s.v5.y);
    m = fmaxf(m, s.v6.x);
    m = fmaxf(m, s.v6.y);
    m = fmaxf(m, s.v7.x);
    m = fmaxf(m, s.v7.y);
    m = fmaxf(m, s.v8.x);
    m = fmaxf(m, s.v8.y);
    m = fmaxf(m, s.v9.x);
    m = fmaxf(m, s.v9.y);
    m = fmaxf(m, s.v10.x);
    m = fmaxf(m, s.v10.y);
    m = fmaxf(m, s.v11.x);
    m = fmaxf(m, s.v11.y);
    m = fmaxf(m, s.v12.x);
    m = fmaxf(m, s.v12.y);
    m = fmaxf(m, s.v13.x);
    m = fmaxf(m, s.v13.y);
    m = fmaxf(m, s.v14.x);
    m = fmaxf(m, s.v14.y);
    m = fmaxf(m, s.v15.x);
    m = fmaxf(m, s.v15.y);
}

template <int Quarter, typename C, bool Wait = true>
__device__ __forceinline__ void upstream_load_score_quarter_x32(
    score_quarter_regs &scores, uint32_t score_row_base, int key_tile, int query_tile) {
    const uint32_t addr = score_row_base + Quarter * 32;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 " "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(scores.v0.x),"=f"(scores.v0.y),"=f"(scores.v1.x),"=f"(scores.v1.y),"=f"(scores.v2.x),"=f"(scores.v2.y),"=f"(scores.v3.x),"=f"(scores.v3.y),"=f"(scores.v4.x),"=f"(scores.v4.y),"=f"(scores.v5.x),"=f"(scores.v5.y),"=f"(scores.v6.x),"=f"(scores.v6.y),"=f"(scores.v7.x),"=f"(scores.v7.y),"=f"(scores.v8.x),"=f"(scores.v8.y),"=f"(scores.v9.x),"=f"(scores.v9.y),"=f"(scores.v10.x),"=f"(scores.v10.y),"=f"(scores.v11.x),"=f"(scores.v11.y),"=f"(scores.v12.x),"=f"(scores.v12.y),"=f"(scores.v13.x),"=f"(scores.v13.y),"=f"(scores.v14.x),"=f"(scores.v14.y),"=f"(scores.v15.x),"=f"(scores.v15.y) : "r"(addr));
    if constexpr (Wait) {
        tensor_load_wait();
    }
}

template <int Quarter, int Half, typename C, bool Wait = true>
__device__ __forceinline__ void upstream_load_score_quarter_half_x16(
    score_quarter_regs &scores,
    uint32_t score_row_base,
    int key_tile,
    int query_tile
) {
    static_assert(Quarter >= 0 && Quarter < 4);
    static_assert(Half == 0 || Half == 1);
    float2 *values = &scores.v0 + Half * 8;
    const uint32_t addr =
        score_row_base + Quarter * 32 + Half * 16;
    asm volatile(
        "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
        "[%16];}"
        : "=f"(values[0].x), "=f"(values[0].y),
          "=f"(values[1].x), "=f"(values[1].y),
          "=f"(values[2].x), "=f"(values[2].y),
          "=f"(values[3].x), "=f"(values[3].y),
          "=f"(values[4].x), "=f"(values[4].y),
          "=f"(values[5].x), "=f"(values[5].y),
          "=f"(values[6].x), "=f"(values[6].y),
          "=f"(values[7].x), "=f"(values[7].y)
        : "r"(addr));
    if constexpr (Wait) {
        tensor_load_wait();
    }
}

#ifndef TK_HAO_DIRECT_HELPERS_ONLY
template <typename C>
__cluster_dims__(1, 1, 1) __launch_bounds__(512, 1)
__global__ void kernel_fp4pv_tk_hao_native_fullgrid(
    const __grid_constant__ globals_fp4pv_upstream_fp8pv<C> g
) {
    static_assert(C::UPSTREAM_OWNED_TWO_STAGE);
    static_assert(C::NUM_THREADS == 512 && C::NUM_WARPS == 16);
    static_assert(C::CLUSTER_SIZE == 1 && C::Mb == 128 && C::Nb == 128);
    static_assert(C::Dqk == 128 && C::Dvo == 128);
    static_assert(C::UPSTREAM_NATIVE_NONCAUSAL);
    static_assert(C::QK_SCALE_CHUNKS == 2);
    static_assert(C::LOAD_STAGES == 4);
    static_assert(C::LOAD_STAGES == C::K_SCALE_SMEM_SLOTS);
    using G = globals_fp4pv_upstream_fp8pv<C>;
    using Score = tt<float, 128, 128>;
    using Output = tt<float, 128, 128>;
    using P = tt<fp8e4m3, 128, 128>;
    using QKScale = full_tt_fp8e4m3<16>;
    constexpr int S0 = 0, S1 = 128, O0 = 256, O1 = 384;
    constexpr int P0 = S0 + 64;
    constexpr int P1 = S1 + 64;
    static_assert(S0 == 0 && S1 == 128 && O0 == 256 && O1 == 384);
    static_assert(P0 == 64 && P1 == 192);
    static_assert(P0 + 32 == 96 && P1 + 32 == 224);
    static_assert(O1 + 128 == 512);
    static_assert(sizeof(typename G::v_fp8_tile) == 16384);
    static_assert(sizeof(upstream_mxfp4_kv_payload_slot<C>) == 16384);
    static_assert(sizeof(upstream_mxfp4_k_scale_slot<C>) == 1024);
    static_assert(
        upstream_mxfp4_pipeline_smem_layout<C>::bytes() == 124928);

    const int logical_job = int(blockIdx.x);
    const int pairs_per_batch = C::N_TILES / C::QUERY_STAGES;
    const int batch = logical_job / (C::HEADS * pairs_per_batch);
    const int batch_job = logical_job % (C::HEADS * pairs_per_batch);
    const int head = batch_job % C::HEADS;
    const int pair = batch_job / C::HEADS;
    if (batch >= C::BATCH) return;
    const int qtile0 = pair * 2;
    const int qtile1 = qtile0 + 1;
    const int iters0 = C::N_TILES;
    const int iters1 = C::N_TILES;
    const float scale_log2 = qk_scale_log2(g, batch, head);

    tensor_allocator<1, 1> tm{};
    Score score0 = tm.template allocate<Score>(S0);
    Score score1 = tm.template allocate<Score>(S1);
    Output output0 = tm.template allocate<Output>(O0);
    Output output1 = tm.template allocate<Output>(O1);
    P p0 = tm.template allocate<P>(P0);
    P p1 = tm.template allocate<P>(P1);
    QKScale qk_q_sc0 = tm.template allocate<QKScale>(S1 + 0);
    QKScale qk_q_sc1 = tm.template allocate<QKScale>(S0 + 0);
    QKScale qk_k_sc0 = tm.template allocate<QKScale>(S1 + 16);
    QKScale qk_k_sc1 = tm.template allocate<QKScale>(S0 + 16);

    extern __shared__ int shm[];
    tma_swizzle_allocator al(shm);
    typename G::q_tile (&q_smem)[2] = al.allocate<typename G::q_tile, 2>();
    typename G::q_sc_tile (&q_sc_smem)[2] =
        al.allocate<typename G::q_sc_tile, 2>();
    upstream_mxfp4_kv_payload_slot<C> (&kv_payload)[C::KV_STAGES] =
        al.allocate<upstream_mxfp4_kv_payload_slot<C>, C::KV_STAGES>();
    upstream_mxfp4_k_scale_slot<C> (&kv_scale)[C::KV_STAGES] =
        al.allocate<upstream_mxfp4_k_scale_slot<C>, C::KV_STAGES>();
    typename G::o_tile &o_smem = al.allocate<typename G::o_tile>();
    sv_fl<128> (&row_max_smem)[2] = al.allocate<sv_fl<128>, 2>();
    sv_fl<128> (&row_sum_smem)[2] = al.allocate<sv_fl<128>, 2>();
    sv_fl<128> (&correction_smem)[2] = al.allocate<sv_fl<128>, 2>();

    __shared__ semaphore q_ready[2];
    __shared__ semaphore kv_full[C::KV_STAGES], kv_empty[C::KV_STAGES];
    __shared__ semaphore score_full[2], score_retired[2], corr_ready[2], p_ready[2];
    __shared__ semaphore pv_done[2], output_ready[2];
    __shared__ semaphore stats_ready[2];
    __shared__ semaphore output_smem_ready[2], output_smem_reusable;
    __shared__ semaphore tmem_dealloc;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            init_semaphore(q_ready[s], 0, 1);
            init_semaphore(score_full[s], 0, 1);
            init_semaphore(score_retired[s], 0, 4);
            init_semaphore(corr_ready[s], 0, 4);
            init_semaphore(p_ready[s], 0, 4);
            init_semaphore(pv_done[s], 0, 1);
            init_semaphore(output_ready[s], 0, 4);
            init_semaphore(stats_ready[s], 0, 4);
            init_semaphore(output_smem_ready[s], 0, 4);
        }
        #pragma unroll
        for (int slot = 0; slot < C::KV_STAGES; ++slot) {
            init_semaphore(kv_full[slot], 0, 1);
            init_semaphore(kv_empty[slot], 0, 1);
        }
        init_semaphore(output_smem_reusable, 0, 1);
        init_semaphore(tmem_dealloc, 0, C::NUM_WARPS);
    }
    // UPSTREAM_EVENT_INIT_ONLY_CTA_BARRIER
    __syncthreads();

    const int warp_id = int(threadIdx.x) >> 5;
    // UPSTREAM_EVENT_REGISTER_BUDGETS: 256*176 + 128*64 + 128*96
    // = 65,536 registers/CTA.  This bounded four-WG discriminator transfers
    // exactly 2,048 registers from the mapped correction role to the two
    // score readers; issue/epilogue ownership is unchanged.
    if (warpgroup::groupid() < 2) {
        warpgroup::increase_registers<176>();
    } else if (warpgroup::groupid() == 2) {
        warpgroup::decrease_registers<80>();
    } else {
        warpgroup::decrease_registers<80>();
    }
    if (warp_id == 14) {
        // UPSTREAM_ROLE_LOADER
        const bool leader = warp::elect_leader();
        if (leader) {
            fp4pv_expect_bytes<C>(q_ready[0],
                sizeof(typename G::q_tile) + sizeof(typename G::q_sc_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                q_smem[0], g.q, {batch, head, qtile0, 0}, q_ready[0]);
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                q_sc_smem[0], g.q_sc, {batch, qtile0, head, 0}, q_ready[0]);
            fp4pv_expect_bytes<C>(q_ready[1],
                sizeof(typename G::q_tile) + sizeof(typename G::q_sc_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                q_smem[1], g.q, {batch, head, qtile1, 0}, q_ready[1]);
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                q_sc_smem[1], g.q_sc, {batch, qtile1, head, 0}, q_ready[1]);
        }
        #pragma unroll 1
        for (int n = 0; n < iters1; ++n) {
            const int k_slot = upstream_kv_slot<C>(n, upstream_kv_kind::K);
            const int v_slot = upstream_kv_slot<C>(n, upstream_kv_kind::V);
            const int k_item = upstream_kv_item<C>(n, upstream_kv_kind::K);
            const int v_item = upstream_kv_item<C>(n, upstream_kv_kind::V);
            const int k_phase = upstream_kv_phase<C>(n, upstream_kv_kind::K);
            const int v_phase = upstream_kv_phase<C>(n, upstream_kv_kind::V);
            if (k_item >= C::KV_STAGES) wait(kv_empty[k_slot], k_phase ^ 1);
            if (v_item >= C::KV_STAGES) wait(kv_empty[v_slot], v_phase ^ 1);
            if (leader) {
                // UPSTREAM_EVENT_COMBINED_K_READY_ARM
                fp4pv_expect_bytes<C>(kv_full[k_slot],
                    sizeof(typename G::k_tile) + sizeof(typename G::k_sc_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    kv_payload[k_slot].k, g.k, {batch, head, n, 0}, kv_full[k_slot]);
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    kv_scale[k_slot].k, g.k_sc, {batch, 2 * n, head, 0}, kv_full[k_slot]);
                // UPSTREAM_EVENT_COMBINED_V_READY_ARM
                fp4pv_expect_bytes<C>(kv_full[v_slot],
                    sizeof(typename G::v_fp8_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    kv_payload[v_slot].v, g.v_fp8,
                    {batch, head, 0, n}, kv_full[v_slot]);
            }
        }
    } else if (warp_id == 12) {
        // UPSTREAM_ROLE_MMA
        const bool leader = warp::elect_leader();
        if (leader) { wait(q_ready[0], 0); wait(q_ready[1], 0); }
        auto issue_qk = [&]<int Stage>(int n) {
            static_assert(Stage == 0 || Stage == 1);
            const int slot = upstream_kv_slot<C>(n, upstream_kv_kind::K);
            const int phase = upstream_kv_phase<C>(n, upstream_kv_kind::K);
            // UPSTREAM_EVENT_ISSUE_ORDER_BANK_REUSE_CREDIT
            // W12 is the sole tensor-command issuer.  The fixed command
            // stream issues the previous same-stage PV before this QK, so
            // TCGEN command ordering protects the aliased P/PV-scale reads
            // without a completion-barrier round trip.  pv_done remains the
            // completion edge for output reads, V release, and the tail.
            wait(kv_full[slot], phase);
            if (leader) {
                if constexpr (Stage == 0) {
                    fp4pv_issue_qk_chunked_ranklocal<C>(
                        score0, q_smem[0], kv_payload[slot].k,
                        q_sc_smem[0], kv_scale[slot].k,
                        qk_q_sc0, qk_k_sc0, score_full[0]);
                } else {
                    fp4pv_issue_qk_chunked_ranklocal<C>(
                        score1, q_smem[1], kv_payload[slot].k,
                        q_sc_smem[1], kv_scale[slot].k,
                        qk_q_sc1, qk_k_sc1, score_full[1]);
                }
            }
        };
        auto issue_pv = [&]<int Stage>(int n) {
            static_assert(Stage == 0 || Stage == 1);
            const int slot = upstream_kv_slot<C>(n, upstream_kv_kind::V);
            const int phase = n & 1;
            // The explicit retired-score acquire is the bank ownership edge;
            // p_ready also dominates it, but is not used as an implicit lease.
            wait(score_retired[Stage], phase);
            wait(p_ready[Stage], phase);
            wait(output_ready[Stage], phase);
            // The first lower-K64 PV overwrites its stage's output accumulator;
            // the upper half and every later PV accumulate.  Same-stage reuse
            // remains ordered by the prior pv_done acquire in issue_qk.
            wait(kv_full[slot],
                 upstream_kv_phase<C>(n, upstream_kv_kind::V));
            if (leader) {
                if constexpr (Stage == 0) {
                    upstream_issue_pv_fp8_tmem_n128<C>(
                        output0, p0, kv_payload[slot].v,
                        pv_done[0], n != 0);
                } else {
                    upstream_issue_pv_fp8_tmem_n128<C>(
                        output1, p1, kv_payload[slot].v,
                        pv_done[1], n != 0);
                }
            }
        };

        issue_qk.template operator()<0>(0);
        // UPSTREAM_LOOP_BEGIN
        #pragma unroll 1
        for (int n = 0; n < iters0; ++n) {
            // UPSTREAM_EVENT_PV0_ISSUE
            issue_pv.template operator()<0>(n);
            // UPSTREAM_EVENT_QK1_ISSUE
            issue_qk.template operator()<1>(n);
            // UPSTREAM_EVENT_PV1_ISSUE
            issue_pv.template operator()<1>(n);
            if (n + 1 < iters0) {
                // UPSTREAM_EVENT_QK0_NEXT_ISSUE
                issue_qk.template operator()<0>(n + 1);
            }
        }
        // Both noncausal stages have exactly N_TILES generations.  There is
        // no causal unmatched-stage QK or PV tail.
        // UPSTREAM_EVENT_NONCAUSAL_PV1_FINAL_DRAIN
        wait(pv_done[1], (iters1 - 1) & 1);
        // UPSTREAM_LOOP_END
    } else if (warp_id < 4) {
        // UPSTREAM_ROLE_SOFTMAX_STAGE0
#define UPSTREAM_FP8PV_READER_STAGE 0
#include "depth1_upstream_mxfp4_fp8pv_quarter_reader.inc"
#undef UPSTREAM_FP8PV_READER_STAGE
    } else if (warp_id < 8) {
        // UPSTREAM_ROLE_SOFTMAX_STAGE1
#define UPSTREAM_FP8PV_READER_STAGE 1
#include "depth1_upstream_mxfp4_fp8pv_quarter_reader.inc"
#undef UPSTREAM_FP8PV_READER_STAGE
    } else if (warp_id >= 8 && warp_id < 12) {
        // UPSTREAM_ROLE_CORRECTION_FINAL_STATS.  Each local warp owns one
        // native 32-row TMEM stripe for both stages.
        const int owned_row_group = warp_id - 8;
        auto correct_one = [&]<int Stage>(int n) {
            static_assert(Stage == 0 || Stage == 1);
            auto &output_owned = Stage == 0 ? output0 : output1;
            wait(corr_ready[Stage], n & 1);
            if (n > 0) {
                wait(pv_done[Stage], (n - 1) & 1);
                tensor_after_thread_sync();
                const int row =
                    owned_row_group * 32 + warp::laneid();
                const float correction =
                    stats_load(correction_smem[Stage], row);
                upstream_correct_output_stripe<C>(
                    output_owned, owned_row_group, correction, false);
            }
            if (warp::elect_leader()) arrive(output_ready[Stage]);
        };
        #pragma unroll 1
        for (int n = 0; n < iters1; ++n) {
            if (n < iters0) {
                correct_one.template operator()<0>(n);
            }
            correct_one.template operator()<1>(n);
        }

        auto stage_final_output = [&]<int Stage>() {
            static_assert(Stage == 0 || Stage == 1);
            constexpr bool STAGE0 = Stage == 0;
            const int stage_iters = STAGE0 ? iters0 : iters1;
            const int query_tile = STAGE0 ? qtile0 : qtile1;
            auto &output_owned = STAGE0 ? output0 : output1;
            wait(stats_ready[Stage], 0);
            wait(pv_done[Stage], (stage_iters - 1) & 1);
            tensor_after_thread_sync();
            // UPSTREAM_EVENT_NATIVE_ROWGROUP_EPILOGUE_READ
            upstream_store_output_row_group_owned_fp8<C>(
                output_owned, o_smem, g, batch, query_tile, head,
                row_max_smem[Stage], row_sum_smem[Stage],
                owned_row_group);
            if (warp::elect_leader()) arrive(output_smem_ready[Stage]);
        };
        stage_final_output.template operator()<0>();
        wait(output_smem_reusable, 0);
        stage_final_output.template operator()<1>();
    } else if (warp_id == 13) {
        // UPSTREAM_ROLE_EPILOGUE
        // This warp owns no output TMEM rows.  It consumes only the four-warp
        // shared-memory publication and retains sole ordered TMA-store
        // ownership for the reusable output tile.
        wait(output_smem_ready[0], 0);
        upstream_store_output_tma_one_warp(o_smem, g, batch, qtile0, head);
        if (warp::elect_leader()) arrive(output_smem_reusable);
        wait(output_smem_ready[1], 0);
        upstream_store_output_tma_one_warp(o_smem, g, batch, qtile1, head);
    } else if (warp_id == 15) {
        // UPSTREAM_ROLE_COMPLETION_PROXY: completion-backed alternating-slot
        // releases. This warp never issues a tensor MMA.
        #pragma unroll 1
        for (int n = 0; n < iters1; ++n) {
            const int query_phase = n & 1;
            const int k_slot = upstream_kv_slot<C>(n, upstream_kv_kind::K);
            const int v_slot = upstream_kv_slot<C>(n, upstream_kv_kind::V);
            wait(score_full[1], query_phase);
            tensor_after_thread_sync();
            // UPSTREAM_EVENT_K_COMPLETION_RELEASE
            if (warp::elect_leader()) arrive(kv_empty[k_slot]);
            wait(pv_done[1], query_phase);
            tensor_after_thread_sync();
            // UPSTREAM_EVENT_V_COMPLETION_RELEASE
            if (warp::elect_leader()) arrive(kv_empty[v_slot]);
        }
    }
    // UPSTREAM_EVENT_TMEM_DEALLOC_ARRIVE
    if (warp::elect_leader()) arrive(tmem_dealloc);
    wait(tmem_dealloc, 0);
}
#endif

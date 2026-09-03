template <typename C, bool NEED_P_SCAN>
__device__ inline float qk_scale_log2(const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g, int b, int h) {
    constexpr float LOG2E_OVER_SQRT_D = 0.10408340855860843f;  // log2(e) / sqrt(192)
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

// A cluster-mapped shared-memory address does not by itself make an mbarrier
// arrival a cluster-scope release. Keep this local helper separate from the
// ThunderKittens default-scope arrival so the K-only attribution routes can
// pair a remote release.cluster with an acquire.cluster wait without patching
// the submodule or changing production defaults.
__device__ __forceinline__ void fp4pv_cluster_arrive_release(
    semaphore &bar, int dst_cta, uint32_t count = 1) {
    if (warp::laneid() == 0) {
        const uint32_t local_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        uint32_t remote_addr;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;\n"
            : "=r"(remote_addr)
            : "r"(local_addr), "r"(dst_cta));
        asm volatile(
            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0], %1;\n"
            :
            : "r"(remote_addr), "r"(count)
            : "memory");
    }
}

__device__ __forceinline__ uint8_t fp4pv_float_to_e8m0_rte(float val) {
    uint32_t u = __float_as_uint(val);
    uint32_t exp32 = (u >> 23) & 0xFFu;
    if (!(val > 1e-38f) || exp32 == 0xFFu) {
        return 0x00;
    }
    uint8_t exp = static_cast<uint8_t>(exp32);
    uint32_t mant = u & 0x7FFFFF;
    constexpr uint32_t half = 1u << 22;
    const bool round_up = (mant > half) || (mant == half && (exp & 1));
    if (round_up && exp < 0xFE) {
        ++exp;
    }
    return exp;
}

__device__ __forceinline__ uint8_t fp4pv_float_to_e8m0_ceil(float val) {
    uint32_t u = __float_as_uint(val);
    uint32_t exp32 = (u >> 23) & 0xFFu;
    if (!(val > 1e-38f) || exp32 == 0xFFu) {
        return 0x00;
    }
    uint8_t exp = static_cast<uint8_t>(exp32);
    const uint32_t mant = u & 0x7FFFFF;
    if (mant > 0 && exp < 0xFE) {
        ++exp;
    }
    return exp;
}

__device__ __forceinline__ uint8_t fp4pv_float_to_e8m0_floor(float val) {
    uint32_t u = __float_as_uint(val);
    uint32_t exp32 = (u >> 23) & 0xFFu;
    if (!(val > 1e-38f) || exp32 == 0xFFu) {
        return 0x00;
    }
    return static_cast<uint8_t>(exp32);
}

__device__ __forceinline__ uint8_t fp4pv_float_to_e8m0(float val, int mode) {
    if (mode == 1) {
        return fp4pv_float_to_e8m0_ceil(val);
    }
    if (mode == 2) {
        return fp4pv_float_to_e8m0_floor(val);
    }
    return fp4pv_float_to_e8m0_rte(val);
}

__device__ __forceinline__ float fp4pv_exp2f_rcp_e8m0(uint8_t e8m0) {
    if (e8m0 == 0) {
        return 0.0f;
    }
    return __uint_as_float((uint32_t)(254 - e8m0) << 23);
}

__device__ __forceinline__ float fp4pv_exp2f_e8m0(uint8_t e8m0) {
    if (e8m0 == 0) {
        return 0.0f;
    }
    return __uint_as_float(static_cast<uint32_t>(e8m0) << 23);
}

__device__ __forceinline__ uint8_t fp4pv_log2_to_e8m0(float log2_val, int mode) {
    if (!isfinite(log2_val)) {
        return 0x00;
    }
    const int floor_log2 = __float2int_rd(log2_val);
    int exp = floor_log2 + 127;
    if (exp <= 0) {
        return 0x00;
    }
    if (exp >= 0xff) {
        return 0xfe;
    }
    const float frac = log2_val - static_cast<float>(floor_log2);
    if (mode == 1) {
        if (frac > 0.0f && exp < 0xfe) {
            ++exp;
        }
    } else if (mode != 2) {
        constexpr float LOG2_1P5 = 0.5849625007211562f;
        if ((frac > LOG2_1P5 || (frac == LOG2_1P5 && (exp & 1))) && exp < 0xfe) {
            ++exp;
        }
    }
    return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ uint8_t fp4pv_log2_to_e8m0_rte_finite(float log2_val) {
    const int floor_log2 = __float2int_rd(log2_val);
    int exp = floor_log2 + 127;
    if (exp <= 0) {
        return 0x00;
    }
    if (exp >= 0xff) {
        return 0xfe;
    }
    const float frac = log2_val - static_cast<float>(floor_log2);
    constexpr float LOG2_1P5 = 0.5849625007211562f;
    if ((frac > LOG2_1P5 || (frac == LOG2_1P5 && (exp & 1))) && exp < 0xfe) {
        ++exp;
    }
    return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ uint8_t fp4pv_log2_to_e8m0_floor_finite(float log2_val) {
    const int floor_log2 = __float2int_rd(log2_val);
    const int exp = floor_log2 + 127;
    if (exp <= 0) {
        return 0x00;
    }
    if (exp >= 0xff) {
        return 0xfe;
    }
    return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ uint16_t fp4pv_bf16_pair_max_abs_bits(const bf16_2 &v) {
    const uint32_t bits = *reinterpret_cast<const uint32_t *>(&v);
    const uint16_t x = static_cast<uint16_t>(bits & 0x7fffu);
    const uint16_t y = static_cast<uint16_t>((bits >> 16) & 0x7fffu);
    return x > y ? x : y;
}

__device__ __forceinline__ uint8_t fp4pv_bf16_abs_bits_to_e8m0(uint16_t abs_bits, int mode) {
    const uint8_t exp = static_cast<uint8_t>((abs_bits >> 7) & 0xffu);
    if (exp == 0 || exp == 0xff) {
        return 0x00;
    }
    const uint16_t mant = abs_bits & 0x7fu;
    if (mode == 1) {
        return static_cast<uint8_t>((mant > 0 && exp < 0xfeu) ? exp + 1 : exp);
    }
    if (mode == 2) {
        return exp;
    }
    const bool round_up = (mant > 0x40u) || (mant == 0x40u && (exp & 1u));
    return static_cast<uint8_t>((round_up && exp < 0xfeu) ? exp + 1 : exp);
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
        "r"(smem_addr),
        "r"(words[0]),
        "r"(words[1]),
        "r"(words[2]),
        "r"(words[3]));
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_v2(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::
        "r"(smem_addr),
        "r"(words[0]),
        "r"(words[1]));
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::
        "r"(smem_addr + 8),
        "r"(words[2]),
        "r"(words[3]));
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_scalar(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 0), "r"(words[0]));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 4), "r"(words[1]));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 8), "r"(words[2]));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 12), "r"(words[3]));
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_striped(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    const int row_phase = warp::laneid() >> 3;
    #pragma unroll
    for (int phase = 0; phase < 4; ++phase) {
        if (row_phase == phase) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                "r"(smem_addr),
                "r"(words[0]),
                "r"(words[1]),
                "r"(words[2]),
                "r"(words[3]));
        }
    }
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_half_striped(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    const int row_phase = warp::laneid() >> 4;
    #pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        if (row_phase == phase) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                "r"(smem_addr),
                "r"(words[0]),
                "r"(words[1]),
                "r"(words[2]),
                "r"(words[3]));
        }
    }
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_staggered(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    const int row_phase = warp::laneid() >> 3;
    #pragma unroll
    for (int phase = 0; phase < 4; ++phase) {
        if (((row_phase + mx_block) & 3) == phase) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                "r"(smem_addr),
                "r"(words[0]),
                "r"(words[1]),
                "r"(words[2]),
                "r"(words[3]));
        }
    }
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_half_staggered(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    const int row_phase = warp::laneid() >> 4;
    #pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        if (((row_phase + (mx_block & 1)) & 1) == phase) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                "r"(smem_addr),
                "r"(words[0]),
                "r"(words[1]),
                "r"(words[2]),
                "r"(words[3]));
        }
    }
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_selected(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
}

template <typename PScTile>
__device__ inline void fp4pv_store_mxfp4_scale_byte_swizzled(
    PScTile &p_sc_tile,
    int global_row,
    int mx_block,
    uint8_t e8m0
) {
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, mx_block);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    asm volatile("st.shared.b8 [%0], %1;" :: "r"(smem_addr), "r"((uint32_t)e8m0));
}

template <typename PScTile>
__device__ inline void fp4pv_store_mxfp4_scale_word_swizzled(
    PScTile &p_sc_tile,
    int global_row,
    uint32_t packed
) {
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, 0);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr), "r"(packed));
}

template <typename PScTile>
__device__ inline void fp4pv_store_mxfp4_scale_word4_swizzled(
    PScTile &p_sc_tile,
    int row_in_32,
    uint32_t packed0,
    uint32_t packed1,
    uint32_t packed2,
    uint32_t packed3
) {
    const int dest = (row_in_32 & 31) * 16;
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
        "r"(smem_addr),
        "r"(packed0),
        "r"(packed1),
        "r"(packed2),
        "r"(packed3));
}

__device__ __forceinline__ void fp4pv_store_global_u32x4(
    uint8_t *__restrict__ dst,
    const uint32_t (&words)[4]
) {
    const uint4 packed = {words[0], words[1], words[2], words[3]};
    *reinterpret_cast<uint4 *>(dst) = packed;
}

__device__ __forceinline__ void fp4pv_store_global_u32x4_value(
    uint8_t *__restrict__ dst,
    uint32_t word
) {
    const uint4 packed = {word, word, word, word};
    *reinterpret_cast<uint4 *>(dst) = packed;
}

__device__ __forceinline__ void fp4pv_store_global_scale_word_swizzled(
    uint8_t *__restrict__ scale_tile,
    int row,
    uint32_t packed
) {
    uint8_t *dst = scale_tile + fp4pv_scale_swizzle_idx(row, 0);
    *reinterpret_cast<uint32_t *>(dst) = packed;
}

template <int HALF, typename PScTile>
__device__ inline void fp4pv_store_mxfp4_scale_halfword4_swizzled(
    PScTile &p_sc_tile,
    int row_in_32,
    uint32_t packed0,
    uint32_t packed1,
    uint32_t packed2,
    uint32_t packed3
) {
    static_assert(HALF == 0 || HALF == 1, "K64 half-scale shared store expects half 0 or 1");
    const int dest = (row_in_32 & 31) * 16 + HALF * 2;
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    asm volatile(
        "st.shared.b16 [%0], %1;\n"
        "st.shared.b16 [%2], %3;\n"
        "st.shared.b16 [%4], %5;\n"
        "st.shared.b16 [%6], %7;\n" ::
        "r"(smem_addr + 0u),
        "r"(packed0 & 0xffffu),
        "r"(smem_addr + 4u),
        "r"(packed1 & 0xffffu),
        "r"(smem_addr + 8u),
        "r"(packed2 & 0xffffu),
        "r"(smem_addr + 12u),
        "r"(packed3 & 0xffffu));
}

template <typename PScTile>
__device__ inline uint32_t fp4pv_load_mxfp4_scale_word_swizzled(
    PScTile &p_sc_tile,
    int global_row
) {
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, 0);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    uint32_t packed;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(packed) : "r"(smem_addr));
    return packed;
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_zero_invalid_causal_groups_mxfp4(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    int global_row,
    int valid_mx_blocks
) {
    uint32_t zero_words[4] = {0u, 0u, 0u, 0u};
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    for (int mx_block = valid_mx_blocks; mx_block < MX_BLOCKS; ++mx_block) {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, zero_words, global_row, mx_block);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, 0u);
    }
}

template <typename PTile>
__device__ inline void fp4pv_zero_invalid_causal_payload_groups_mxfp4(
    PTile &p_fp4_tile,
    int global_row,
    int valid_mx_blocks,
    int packed_col_offset = 0
) {
    uint32_t zero_words[4] = {0u, 0u, 0u, 0u};
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    for (int mx_block = valid_mx_blocks; mx_block < MX_BLOCKS; ++mx_block) {
        fp4pv_store_quantized_scores_group_mxfp4(
            p_fp4_tile, zero_words, global_row, mx_block, packed_col_offset);
    }
}

template <typename PScTT>
__device__ inline void fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
    PScTT &p_sc_tm,
    int row_offset,
    uint32_t row_tile0,
    uint32_t row_tile1,
    uint32_t row_tile2,
    uint32_t row_tile3
) {
    static_assert(std::is_same_v<typename PScTT::T, fp8e8m0>, "direct P-scale TMEM store expects e8m0 scales");
    static_assert(PScTT::rows == 128 && PScTT::cols == 16, "direct P-scale TMEM store expects a 128x16 scale tile");
    const uint32_t tmem_addr = p_sc_tm.addr + (static_cast<uint32_t>(row_offset) << 16);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::
        "r"(tmem_addr),
        "r"(row_tile0),
        "r"(row_tile1),
        "r"(row_tile2),
        "r"(row_tile3));
}

template <typename PScTT>
__device__ inline void fp4pv_store_mxfp4_scale_tmem_32x32b_x1(
    PScTT &p_sc_tm,
    int col_offset,
    uint32_t packed_scale
) {
    static_assert(std::is_same_v<typename PScTT::T, fp8e8m0>, "direct P-scale TMEM store expects e8m0 scales");
    asm volatile("{tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};}" ::
        "r"(p_sc_tm.addr + col_offset),
        "r"(packed_scale));
}

__device__ __forceinline__ void fp4pv_tmem_store_wait() {
    asm volatile("{tcgen05.wait::st.sync.aligned;}" ::);
}

template <typename ScaleTT>
__device__ __forceinline__ void fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(ScaleTT &scale_tm, uint32_t packed_scale) {
    #pragma unroll
    for (int row = 0; row < 128; row += 32) {
        fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
            scale_tm, row, packed_scale, packed_scale, packed_scale, packed_scale);
    }
}

template <typename C, typename TT, typename ST>
__device__ inline void fp4pv_load_mxnv_scale_async_cluster(TT &dst, const ST &src) {
    static_assert(C::CLUSTER_SIZE == 1 || C::CLUSTER_SIZE == 2, "FP4PV scale staging supports cluster size 1 or 2");
    load_mxnv_scale_async<C::CLUSTER_SIZE>(dst, src);
}

template <typename TT, typename ST>
__device__ inline void fp4pv_load_mxnv_scale_async_ranklocal(TT &dst, const ST &src) {
    load_mxnv_scale_async<1>(dst, src);
}

__device__ inline void fp4pv_publish_shared_backing_proxy_only() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ inline void fp4pv_publish_cta_shared_only() {
    __threadfence_block();
}

template <typename C, typename TT, typename ST>
__device__ inline void fp4pv_load_q_scale_chunk_cluster(TT &q_sc_tm, const ST &q_sc_smem, int chunk_idx) {
    auto &q_sc_sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
        reinterpret_cast<uint64_t>(&q_sc_smem.data[0]) + 16 * 32 * chunk_idx);
    fp4pv_load_mxnv_scale_async_cluster<C>(q_sc_tm, q_sc_sm_sub);
}

template <typename C, typename TT, typename ST>
__device__ inline void fp4pv_load_k_scale_chunk_cluster(TT &k_sc_tm, const ST &k_sc_smem, int chunk_idx) {
    auto &k_sc_sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
        reinterpret_cast<uint64_t>(&k_sc_smem.data[0]) + 16 * 32 * chunk_idx);
    fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm, k_sc_sm_sub);
}

template <typename TT, typename ST>
__device__ inline void fp4pv_load_q_scale_chunk_ranklocal(TT &q_sc_tm, const ST &q_sc_smem, int chunk_idx) {
    auto &q_sc_sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
        reinterpret_cast<uint64_t>(&q_sc_smem.data[0]) + 16 * 32 * chunk_idx);
    fp4pv_load_mxnv_scale_async_ranklocal(q_sc_tm, q_sc_sm_sub);
}

template <typename TT, typename ST>
__device__ inline void fp4pv_load_k_scale_chunk_ranklocal(TT &k_sc_tm, const ST &k_sc_smem, int chunk_idx) {
    auto &k_sc_sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
        reinterpret_cast<uint64_t>(&k_sc_smem.data[0]) + 16 * 32 * chunk_idx);
    fp4pv_load_mxnv_scale_async_ranklocal(k_sc_tm, k_sc_sm_sub);
}

template <typename C, typename QScTTFull, typename QScTile>
__device__ inline void fp4pv_stage_q_scale_tmem_cluster(QScTTFull &q_sc_tm_full, const QScTile &q_sc_smem) {
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, chunk);
    }
}

__device__ __forceinline__ float2 fp4pv_apply_backward_dp_delta_scale(
    float2 p_pair,
    bf16_2 dp_pair_bf,
    float2 scale_2,
    float2 bias_2
) {
    const float2 dp_pair = __bfloat1622float2(dp_pair_bf);
    const float2 ds_factor = __ffma2_rn(dp_pair, scale_2, bias_2);
    return {p_pair.x * ds_factor.x, p_pair.y * ds_factor.y};
}

__device__ __forceinline__ float2 fp4pv_apply_backward_dp_delta_scale(
    float2 p_pair,
    float2 dp_pair,
    float2 scale_2,
    float2 bias_2
) {
    const float2 ds_factor = __ffma2_rn(dp_pair, scale_2, bias_2);
    return {p_pair.x * ds_factor.x, p_pair.y * ds_factor.y};
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_cluster_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_ABt(d, a, b, sa, sb, sem);
    } else {
        mm2_ABt(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_cluster_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_ABt(d, a, b, sa, sb, sem);
    } else {
        mma2_ABt(d, a, b, sa, sb, sem);
    }
}

template <typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_ranklocal_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    mm_ABt(d, a, b, sa, sb, sem);
}

template <typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_ranklocal_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    mma_ABt(d, a, b, sa, sb, sem);
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_cluster_AtB(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_AtB(d, a, b, sa, sb, sem);
    } else {
        mm2_AtB(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_cluster_AtB(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_AtB(d, a, b, sa, sb, sem);
    } else {
        mma2_AtB(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_nosem_cluster_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_ABt(d, a, b, sa, sb);
    } else {
        mm2_ABt(d, a, b, sa, sb);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_nosem_cluster_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_ABt(d, a, b, sa, sb);
    } else {
        mma2_ABt(d, a, b, sa, sb);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mm_unscaled_cluster_ABt(
    D &d, A &a, B &b, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_ABt(d, a, b, sem);
    } else {
        mm2_ABt(d, a, b, sem);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mma_unscaled_cluster_ABt(
    D &d, A &a, B &b, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_ABt(d, a, b, sem);
    } else {
        mma2_ABt(d, a, b, sem);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mm_unscaled_nosem_cluster_ABt(
    D &d, A &a, B &b
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_ABt(d, a, b);
    } else {
        mm2_ABt(d, a, b);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mma_unscaled_nosem_cluster_ABt(
    D &d, A &a, B &b
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_ABt(d, a, b);
    } else {
        mma2_ABt(d, a, b);
    }
}

template <bool RANK_LOCAL, typename C>
__device__ __forceinline__ void fp4pv_expect_bytes_route(semaphore &bar, uint32_t bytes) {
    if constexpr (RANK_LOCAL) {
        tma::expect_bytes(bar, bytes);
    } else {
        fp4pv_expect_bytes<C>(bar, bytes);
    }
}

template <bool RANK_LOCAL, typename C>
__device__ __forceinline__ void fp4pv_wait_route(semaphore &bar, int phase) {
    if constexpr (RANK_LOCAL) {
        wait(bar, phase);
    } else {
        fp4pv_wait<C>(bar, phase);
    }
}

template <int axis, cache_policy policy, bool RANK_LOCAL, typename C, typename ST, typename GL, typename COORD>
__device__ __forceinline__ void fp4pv_load_async_route(
    ST &dst,
    const GL &src,
    const COORD &idx,
    semaphore &bar,
    uint16_t cluster_mask,
    int dst_mbar_cta = -1
) {
    if constexpr (RANK_LOCAL) {
        tma::load_async<axis, policy>(dst, src, idx, bar);
    } else {
        fp4pv_load_async<axis, policy, C>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
    }
}

template <bool RANK_LOCAL, typename C>
__device__ __forceinline__ void fp4pv_tensor_commit_route(semaphore &sem) {
    if constexpr (RANK_LOCAL) {
        tensor_commit<1>(sem);
    } else {
        tensor_commit<C::CLUSTER_SIZE>(sem);
    }
}

template <bool RANK_LOCAL, typename C>
__device__ __forceinline__ void fp4pv_tcgen_commit_route(semaphore &sem) {
    if constexpr (RANK_LOCAL) {
        detail::tcgen05::commit<1>(sem);
    } else {
        detail::tcgen05::commit<C::CLUSTER_SIZE>(sem);
    }
}

template <bool RANK_LOCAL, typename C, typename TT, typename ST>
__device__ __forceinline__ void fp4pv_load_mxnv_scale_async_route(TT &dst, ST &src) {
    if constexpr (RANK_LOCAL) {
        fp4pv_load_mxnv_scale_async_ranklocal(dst, src);
    } else {
        fp4pv_load_mxnv_scale_async_cluster<C>(dst, src);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mm_p_stage_unscaled_ABt(
    D &d, A &a, B &b, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mm_unscaled_cluster_ABt<C>(d, a_view, b, sem);
    } else {
        fp4pv_mm_unscaled_cluster_ABt<C>(d, a, b, sem);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mma_p_stage_unscaled_ABt(
    D &d, A &a, B &b, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mma_unscaled_cluster_ABt<C>(d, a_view, b, sem);
    } else {
        fp4pv_mma_unscaled_cluster_ABt<C>(d, a, b, sem);
    }
}

template <typename Tile>
__device__ __forceinline__ void fp4pv_zero_shared_tile_block(Tile &tile) {
    auto *raw = reinterpret_cast<unsigned char *>(&tile);
    for (int i = threadIdx.x; i < (int)sizeof(Tile); i += blockDim.x) {
        raw[i] = 0;
    }
}

template <typename Tile>
__device__ __forceinline__ void fp4pv_zero_shared_tile_warp(Tile &tile) {
    auto *raw = reinterpret_cast<unsigned char *>(&tile);
    for (int i = warp::laneid(); i < (int)sizeof(Tile); i += WARP_THREADS) {
        raw[i] = 0;
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_p_stage_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mm_cluster_ABt<C>(d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mm_cluster_ABt<C>(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_p_stage_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mma_cluster_ABt<C>(d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mma_cluster_ABt<C>(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_p_stage_ranklocal_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mm_ranklocal_ABt(d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mm_ranklocal_ABt(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_p_stage_ranklocal_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mma_ranklocal_ABt(d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mma_ranklocal_ABt(d, a, b, sa, sb, sem);
    }
}

template <int CHUNK, int ACC, int SCALE_FACTOR_ID, bool DO_COMMIT, typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_p_stage_ABt_k64_chunk_view(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    using T_D = typename D::T;
    using T_AB = typename A::T;
    using T_SAB = typename SA::T;
    static_assert(CHUNK == 0 || CHUNK == 1, "split-K64 FP4 PV only supports the two K64 chunks in a K128 tile");
    static_assert(ACC == 0 || ACC == 1, "tcgen05 accumulator predicate must be 0 or 1");
    static_assert(std::is_same_v<T_AB, fp4e2m1_2>, "split-K64 PV is only for FP4 payloads");
    static_assert(std::is_same_v<T_SAB, fp8e8m0>, "split-K64 PV expects MXFP4 e8m0 scales");
    constexpr int ncta = C::CLUSTER_SIZE;
    constexpr int trans_a = transpose::N;
    constexpr int trans_b = 1 - transpose::T;
    constexpr int M = A::rows * ncta;
    constexpr int N = B::rows * ncta;
    constexpr int K = A::cols * 2;
    constexpr int block_size = 32;
    static_assert(K == 128, "split-K64 FP4 PV expects a logical K128 source tile");
    static_assert(M == D::rows * ncta && ((ncta == 1 && M == 128) || (ncta == 2 && (M == 128 || M == 256))));
    static_assert(N == D::cols && ((ncta == 1 && N % 8 == 0) || (ncta == 2 && N % 16 == 0)));
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, SCALE_FACTOR_ID>();
    detail::tcgen05::template st_st<T_AB, T_SAB, ACC, ncta, block_size>(
        d.addr,
        a_desc.chunk_descriptor(CHUNK),
        b_desc.chunk_descriptor(CHUNK),
        sa.addr,
        sb.addr,
        idesc);
    if constexpr (DO_COMMIT) {
        detail::tcgen05::commit<ncta>(sem);
    }
}

template <typename C, bool CLEAR_ACC, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_p_stage_ABt_split_k64_view(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    fp4pv_mma_p_stage_ABt_k64_chunk_view<0, CLEAR_ACC ? 0 : 1, 0, false, C>(d, a, b, sa, sb, sem);
    fp4pv_mma_p_stage_ABt_k64_chunk_view<1, 1, 2, true, C>(d, a, b, sa, sb, sem);
}

template <int CHUNK, int ACC, int SCALE_FACTOR_ID, bool DO_COMMIT, typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_p_stage_ABt_k64_chunk(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mma_p_stage_ABt_k64_chunk_view<CHUNK, ACC, SCALE_FACTOR_ID, DO_COMMIT, C>(
            d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mma_p_stage_ABt_k64_chunk_view<CHUNK, ACC, SCALE_FACTOR_ID, DO_COMMIT, C>(
            d, a, b, sa, sb, sem);
    }
}

template <typename C, bool CLEAR_ACC, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_p_stage_ABt_split_k64(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mma_p_stage_ABt_split_k64_view<C, CLEAR_ACC>(d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mma_p_stage_ABt_split_k64_view<C, CLEAR_ACC>(d, a, b, sa, sb, sem);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group(
    const float2 *scores_group,
    int quant_mode,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const float2 pair_f = scores_group[jj];
        scores_bf[jj] = __float22bfloat162_rn(pair_f);
        block_amax = fmaxf(block_amax, fmaxf(pair_f.x, pair_f.y));
    }
    e8m0 = fp4pv_float_to_e8m0(block_amax, quant_mode);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0;
        }
        e8m0 = 0;
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_rte(
    const float2 *scores_group,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const float2 pair_f = scores_group[jj];
        scores_bf[jj] = __float22bfloat162_rn(pair_f);
        block_amax = fmaxf(block_amax, fmaxf(pair_f.x, pair_f.y));
    }
    e8m0 = fp4pv_float_to_e8m0_rte(block_amax);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0x77777777u;
        }
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_bf16_amax(
    const float2 *scores_group,
    int quant_mode,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        scores_bf[jj] = __float22bfloat162_rn(scores_group[jj]);
        block_amax = fmaxf(
            block_amax,
            fmaxf(fabsf(__bfloat162float(scores_bf[jj].x)), fabsf(__bfloat162float(scores_bf[jj].y))));
    }
    e8m0 = fp4pv_float_to_e8m0(block_amax, quant_mode);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0x77777777u;
        }
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_bf16_pairs_amax(
    const bf16_2 (&scores_bf)[16],
    int quant_mode,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    uint16_t block_amax_bits = 0;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const uint16_t pair_amax_bits = fp4pv_bf16_pair_max_abs_bits(scores_bf[jj]);
        block_amax_bits = pair_amax_bits > block_amax_bits ? pair_amax_bits : block_amax_bits;
    }
    e8m0 = fp4pv_bf16_abs_bits_to_e8m0(block_amax_bits, quant_mode);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0x77777777u;
        }
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_scaled_rte(
    const float2 *scores_group,
    const float2 p_scale,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const float2 pair_f = scores_group[jj];
        scores_bf[jj] = __float22bfloat162_rn(pair_f);
        block_amax = fmaxf(block_amax, fmaxf(pair_f.x, pair_f.y));
    }
    const float p_scale_scalar = p_scale.x;
    e8m0 = fp4pv_float_to_e8m0_rte(block_amax * p_scale_scalar);
    const float coeff = (6.0f * p_scale_scalar) * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0;
        }
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_fixed_scaled_rte(
    const float2 *scores_group,
    uint32_t (&words)[4]
) {
    __align__(8) bf16_2 scores_bf[16];
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        scores_bf[jj] = __float22bfloat162_rn(scores_group[jj]);
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(
            e03, e47, FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF);
    }
}

__device__ __forceinline__ uint32_t fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(
    const float2 *scores_group
) {
    const float2 coeff = {
        FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF,
        FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF
    };
    const auto lo = transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x_with_rn(
        scores_group[0], scores_group[1], coeff, 0);
    const auto hi = transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x_with_rn(
        scores_group[2], scores_group[3], coeff, 0);
    const uint16_t lo_bits = *reinterpret_cast<const uint16_t *>(&lo);
    const uint16_t hi_bits = *reinterpret_cast<const uint16_t *>(&hi);
    return static_cast<uint32_t>(lo_bits) | (static_cast<uint32_t>(hi_bits) << 16);
}

__device__ __forceinline__ uint32_t fp4pv_mul_cvt_fp32_to_fp4_8x_scaled_rte(
    const float2 *scores_group,
    float coeff_scalar
) {
    const float2 coeff = {coeff_scalar, coeff_scalar};
    const auto lo = transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x_with_rn(
        scores_group[0], scores_group[1], coeff, 0);
    const auto hi = transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x_with_rn(
        scores_group[2], scores_group[3], coeff, 0);
    const uint16_t lo_bits = *reinterpret_cast<const uint16_t *>(&lo);
    const uint16_t hi_bits = *reinterpret_cast<const uint16_t *>(&hi);
    return static_cast<uint32_t>(lo_bits) | (static_cast<uint32_t>(hi_bits) << 16);
}

// Pure-convert FP4 pack for scores that already carry the fixed payload
// coefficient folded into their exp2-space bias. This matches the byte layout
// used by fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte without applying the
// fixed-scale coefficient a second time.
__device__ __forceinline__ uint32_t fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(
    const float2 *scores_group
) {
    uint32_t word = 0;
    asm volatile(
        "{\n"
        ".reg.b8 b0; \n\t"
        ".reg.b8 b1; \n\t"
        ".reg.b8 b2; \n\t"
        ".reg.b8 b3; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n\t"
        "mov.b32 %0, {b0, b1, b2, b3};\n\t"
        "}"
        : "=r"(word)
        : "f"(scores_group[0].x), "f"(scores_group[0].y),
          "f"(scores_group[1].x), "f"(scores_group[1].y),
          "f"(scores_group[2].x), "f"(scores_group[2].y),
          "f"(scores_group[3].x), "f"(scores_group[3].y));
    return word;
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_with_amax(
    const float2 *scores_group,
    float block_amax,
    int quant_mode,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        scores_bf[jj] = __float22bfloat162_rn(scores_group[jj]);
    }
    e8m0 = fp4pv_float_to_e8m0(block_amax, quant_mode);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0;
        }
        e8m0 = 0;
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile>
__device__ inline uint8_t fp4pv_pack_scores_one_to_stage_mxfp4_no_scale(
    PTile &p_fp4_tile,
    const float2 (&scores_one)[16],
    int global_row,
    int mx_block,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t words[4];
    uint8_t e8m0 = 0;
    fp4pv_quantize_mxfp4_scores_group(scores_one, quant_mode, words, e8m0);
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
    return e8m0;
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile>
__device__ inline uint8_t fp4pv_pack_scores_one_to_stage_mxfp4_no_scale_with_amax(
    PTile &p_fp4_tile,
    const float2 (&scores_one)[16],
    float block_amax,
    int global_row,
    int mx_block,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t words[4];
    uint8_t e8m0 = 0;
    fp4pv_quantize_mxfp4_scores_group_with_amax(scores_one, block_amax, quant_mode, words, e8m0);
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
    return e8m0;
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group(&scores_reg[mx_block * 16], quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <typename PTile, typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_stage_mxfp4_marker(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t packed_scales = 0u;
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group(&scores_reg[mx_block * 16], quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    return packed_scales;
}

template <typename PTile, typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_stage_mxfp4_marker_bf16_amax(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t packed_scales = 0u;
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    return packed_scales;
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_bf16_amax(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <typename PTile, typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_stage_mxfp4_marker_bf16_amax_and_scan(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    fp4pv_p_scan3d &scan,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t packed_scales = 0u;
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, mx_block);
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    return packed_scales;
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_bf16_amax_and_scan(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    fp4pv_p_scan3d &scan,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, mx_block);
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_rte(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group_rte(&scores_reg[mx_block * 16], words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    bool STORE_SCALE_TO_SMEM = true,
    int NUM_SCORE_REGS,
    typename PTile,
    typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_stage_mxfp4_scaled_rte(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[NUM_SCORE_REGS],
    const float2 p_scale,
    int global_row,
    int packed_col_offset = 0
) {
    uint32_t packed_scales = 0u;
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    static_assert(NUM_SCORE_REGS >= MX_BLOCKS * 16,
        "MXFP4 P packer needs one float2 register per packed score pair");
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group_scaled_rte(&scores_reg[mx_block * 16], p_scale, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4_selected<
            SCALAR_P_PAYLOAD_STORE,
            STRIPED_P_PAYLOAD_STORE,
            HALF_STRIPED_P_PAYLOAD_STORE,
            V2_P_PAYLOAD_STORE,
            STAGGERED_P_PAYLOAD_STORE,
            HALF_STAGGERED_P_PAYLOAD_STORE>(
            p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    if constexpr (STORE_SCALE_TO_SMEM) {
        fp4pv_store_mxfp4_scale_word_swizzled(p_sc_tile, global_row, packed_scales);
    }
    return packed_scales;
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_fixed_scaled_rte(
    PTile &p_fp4_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        fp4pv_quantize_mxfp4_scores_group_fixed_scaled_rte(&scores_reg[mx_block * 16], words);
        fp4pv_store_quantized_scores_group_mxfp4_selected<
            SCALAR_P_PAYLOAD_STORE,
            STRIPED_P_PAYLOAD_STORE,
            HALF_STRIPED_P_PAYLOAD_STORE,
            V2_P_PAYLOAD_STORE,
            STAGGERED_P_PAYLOAD_STORE,
            HALF_STAGGERED_P_PAYLOAD_STORE>(
            p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
}

template <typename PScTile>
__device__ inline void fp4pv_fill_mxfp4_scale_stage_word_swizzled(
    PScTile &p_sc_tile,
    int rows,
    uint32_t packed_scales
) {
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        fp4pv_store_mxfp4_scale_word_swizzled(p_sc_tile, row, packed_scales);
    }
}

template <int MX_BLOCK_BEGIN, int MX_BLOCK_END, typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_range(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    #pragma unroll
    for (int mx_block = MX_BLOCK_BEGIN; mx_block < MX_BLOCK_END; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group(&scores_reg[mx_block * 16], quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <int MX_BLOCK_BEGIN, int MX_BLOCK_END, typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_half_to_stage_mxfp4_range(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_half)[32],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    #pragma unroll
    for (int mx_block = MX_BLOCK_BEGIN; mx_block < MX_BLOCK_END; ++mx_block) {
        const int local_block = mx_block - MX_BLOCK_BEGIN;
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group(&scores_half[local_block * 16], quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile,
    typename PScTile>
__device__ inline void fp4pv_pack_scores_one_to_stage_mxfp4(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_one)[16],
    int global_row,
    int mx_block,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t words[4];
    uint8_t e8m0 = 0;
    fp4pv_quantize_mxfp4_scores_group(scores_one, quant_mode, words, e8m0);
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
    fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile,
    typename PScTile>
__device__ inline void fp4pv_pack_scores_one_to_stage_mxfp4_with_amax(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_one)[16],
    float block_amax,
    int global_row,
    int mx_block,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t words[4];
    uint8_t e8m0 = 0;
    fp4pv_quantize_mxfp4_scores_group_with_amax(scores_one, block_amax, quant_mode, words, e8m0);
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
    fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
}

template <typename C, bool NEED_P_SCAN, typename VTile>
__device__ inline void load_v_fp4_tile_from_global(
    VTile &v_smem,
    const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int packed_col_offset = 0
) {
    const int packed_seqlen = g.k.rows() / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "Expected fp4 V load width divisible by vector width");
    static_assert(VTile::cols >= C::Nb / 2, "Expected fp4 V tile to hold at least one 128-column packed slice");
    static_assert(VTile::swizzle_bytes >= BYTES_PER_STORE, "Expected V tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.v_fp4[((((b * g.q.depth() + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(VTile::idx(smem_base, {row, packed_col_offset + col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <int PRODUCER_THREADS, typename C, bool NEED_P_SCAN, typename VTile>
__device__ inline void load_v_fp4_tile_from_global_producer_group(
    VTile &v_smem,
    const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int producer_lane,
    int packed_col_offset = 0
) {
    const int packed_seqlen = g.k.rows() / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "Expected fp4 V load width divisible by vector width");
    static_assert(VTile::cols >= C::Nb / 2, "Expected fp4 V tile to hold at least one 128-column packed slice");
    static_assert(VTile::swizzle_bytes >= BYTES_PER_STORE, "Expected V tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    if constexpr (PRODUCER_THREADS == 64 && C::Dvo == 128 && C::Nb == 128 && C::CLUSTER_SIZE == 2) {
        static_assert(TOTAL_STORES == 4 * PRODUCER_THREADS,
            "Expected four V FP4 vector chunks per producer lane");
        uint4 values[4];
        uint32_t smem_addrs[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int linear = producer_lane + u * PRODUCER_THREADS;
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            values[u] = *reinterpret_cast<const uint4 *>(
                &g.v_fp4[((((b * g.q.depth() + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
            smem_addrs[u] = VTile::idx(smem_base, {row, packed_col_offset + col});
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint4 value = values[u];
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(smem_addrs[u]),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    } else {
        for (int linear = producer_lane; linear < TOTAL_STORES; linear += PRODUCER_THREADS) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.v_fp4[((((b * g.q.depth() + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(VTile::idx(smem_base, {row, packed_col_offset + col})),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    }
}

template <typename C, bool NEED_P_SCAN, typename VScTile>
__device__ inline void load_v_sc_prepared_from_global(
    VScTile &v_sc_smem,
    const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g,
    int b,
    int h,
    int k_iter
) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&v_sc_smem.data[0]);
    static_assert(sizeof(VScTile) % sizeof(uint32_t) == 0, "Expected aligned MXFP4 V scale tile size");
    constexpr int TOTAL_WORDS = sizeof(VScTile) / sizeof(uint32_t);
    constexpr int SCALE_ITERS_PER_TILE = 128 / C::Nb;
    const int words_per_tile = 512 / sizeof(uint32_t);
    const int scale_tile = k_iter / SCALE_ITERS_PER_TILE;
    const int scale_half = k_iter - scale_tile * SCALE_ITERS_PER_TILE;
    const int output_half = C::Dvo == 64 ? g.dvo_half : 0;
    const int base_offset =
        (((((b * g.q.depth() + h) * g.v_scale_records_per_head + output_half) *
            (g.k.rows() / 128)) + scale_tile) * 32 * 16);
    if constexpr (C::Nb == 128) {
        constexpr int BYTES_PER_STORE = 16;
        static_assert(sizeof(VScTile) % BYTES_PER_STORE == 0, "Expected aligned MXFP4 V scale tile size");
        constexpr int TOTAL_STORES = sizeof(VScTile) / BYTES_PER_STORE;
        uint32_t smem_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);
        for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
            const int byte_offset = linear * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.v_sc_prepared[base_offset + byte_offset]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(smem_base + byte_offset),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    } else {
        for (int linear = warp::laneid(); linear < TOTAL_WORDS; linear += WARP_THREADS) {
            uint32_t packed = *reinterpret_cast<const uint32_t *>(
                &g.v_sc_prepared[base_offset + linear * sizeof(uint32_t)]);
            if constexpr (C::Nb == 64) {
                packed = scale_half == 0 ? (packed & 0x0000ffffu) : ((packed >> 16) & 0x0000ffffu);
            }
            dst[linear] = packed;
        }
    }
    (void)words_per_tile;
}

template <int PRODUCER_THREADS, typename C, bool NEED_P_SCAN, typename VScTile>
__device__ inline void load_v_sc_prepared_from_global_producer_group(
    VScTile &v_sc_smem,
    const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g,
    int b,
    int h,
    int k_iter,
    int producer_lane
) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&v_sc_smem.data[0]);
    static_assert(sizeof(VScTile) % sizeof(uint32_t) == 0, "Expected aligned MXFP4 V scale tile size");
    constexpr int TOTAL_WORDS = sizeof(VScTile) / sizeof(uint32_t);
    constexpr int SCALE_ITERS_PER_TILE = 128 / C::Nb;
    const int scale_tile = k_iter / SCALE_ITERS_PER_TILE;
    const int scale_half = k_iter - scale_tile * SCALE_ITERS_PER_TILE;
    const int output_half = C::Dvo == 64 ? g.dvo_half : 0;
    const int base_offset =
        (((((b * g.q.depth() + h) * g.v_scale_records_per_head + output_half) *
            (g.k.rows() / 128)) + scale_tile) * 32 * 16);
    if constexpr (C::Nb == 128) {
        constexpr int BYTES_PER_STORE = 16;
        static_assert(sizeof(VScTile) % BYTES_PER_STORE == 0, "Expected aligned MXFP4 V scale tile size");
        constexpr int TOTAL_STORES = sizeof(VScTile) / BYTES_PER_STORE;
        uint32_t smem_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);
        for (int linear = producer_lane; linear < TOTAL_STORES; linear += PRODUCER_THREADS) {
            const int byte_offset = linear * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.v_sc_prepared[base_offset + byte_offset]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(smem_base + byte_offset),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    } else {
        for (int linear = producer_lane; linear < TOTAL_WORDS; linear += PRODUCER_THREADS) {
            uint32_t packed = *reinterpret_cast<const uint32_t *>(
                &g.v_sc_prepared[base_offset + linear * sizeof(uint32_t)]);
            if constexpr (C::Nb == 64) {
                packed = scale_half == 0 ? (packed & 0x0000ffffu) : ((packed >> 16) & 0x0000ffffu);
            }
            dst[linear] = packed;
        }
    }
}

template <typename C>
struct globals_pv_from_p {
    using p_fp4_tile = typename globals_fp4pv<C>::p_fp4_tile;
    using p_sc_tile = typename globals_fp4pv<C>::p_sc_tile;
    using v_fp4_tile = typename globals_fp4pv<C>::v_fp4_tile;
    using v_sc_tile = typename globals_fp4pv<C>::v_sc_tile;
    using o_tile = typename globals_fp4pv<C>::o_tile;
    using o_gl = typename globals_fp4pv<C>::o_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(o_tile));
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc_prepared;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    o_gl o;
    int tile_begin;
    int tiles_to_run;
    int second_tile_mm2;
    int pv_wait_count;
    int early_tile_arrived;
    int batch;
    int heads;
    int seqlen;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / (C::Mb * C::CLUSTER_SIZE)) * C::CLUSTER_SIZE);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_pv_from_p_chunked_debug {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = typename base::v_fp4_tile;
    using v_sc_tile = typename base::v_sc_tile;
    using o_tile = typename base::o_tile;
    using o_gl = typename base::o_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        add((int)sizeof(o_tile));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    bf16 *o_direct;
    o_gl o;
    int batch;
    int heads;
    int seqlen;
    int p_cols;
    int v_scale_rows;
    int debug_stage;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / (C::Mb * C::CLUSTER_SIZE)) * C::CLUSTER_SIZE);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

#if TK_FA4_DVO64_TWO_PASS_BUILD
template <typename C>
struct globals_mxfp4_dvo64_lean2wg {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using v_sc_tile = typename base::v_sc_tile;
    using q_gl = typename base::q_gl;
    using k_gl = gl<fp4e2m1_2, -1, -1, -1, -1,
                    tma::descriptor<k_tile, dim::ROW>>;
    using v_fp4_gl = gl<fp4e2m1_2, -1, -1, -1, -1,
                        tma::descriptor<v_fp4_tile, dim::ROW>>;

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
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY =
        compute_dynamic_shared_memory();

    q_gl q;
    k_gl k;
    v_fp4_gl v_fp4_tma;
    const uint8_t *q_fp4;
    const uint8_t *q_sc;
    const float *q_sg;
    const uint8_t *k_fp4;
    const uint8_t *k_sc;
    const float *k_sg;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    bf16 *o;
    float *lse;
    int batch;
    int heads;
    int seqlen;
    int dvo_half;
    int disable_lse_store;
#if TK_FA4_DVO64_LEAN_TRACE_BUILD
    int trace_enabled;
#endif
    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / C::Mb));
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() {
        return DYNAMIC_SHARED_MEMORY;
    }
};

#endif

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
__device__ __forceinline__ void fp4pv_load_score_quarter_tmem(
    float2 (&scores)[16], uint32_t score_tt_base, int qid);
__device__ inline float softmax_rescale_factor_strict(
    float row_max, float row_max_old, float scale_log2);
template <typename C, typename OutputTT>
__device__ __forceinline__ void rescale_tt_output_inplace_direct_streaming(
    OutputTT &tt_output, float correction, bool lane_needs_rescale);

template <typename C> struct globals_mxfp4_nb128_dvo128_qstage2 {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = typename base::v_fp4_tile;
    using v_sc_tile = typename base::v_sc_tile;
    const uint8_t *q_fp4, *q_sc, *k_fp4, *k_sc;
    const float *q_sg, *k_sg;
    const uint8_t *v_fp4, *v_sc;
    bf16 *o; float *lse; unsigned long long *trace;
    int query_tiles, heads, seqlen;
    float softmax_scale_log2;
    __host__ __inline__ dim3 grid() const { return dim3(((query_tiles + 1) / 2) * heads); }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
private:
    static constexpr int smem_size() {
        int bytes = 0;
        auto add = [&](int n) constexpr { bytes = align_up_constexpr(bytes, 1024) + n; };
        add(sizeof(q_tile)); add(sizeof(q_tile));
        add(sizeof(q_sc_tile)); add(sizeof(q_sc_tile));
        add(sizeof(k_tile)); add(sizeof(k_sc_tile));
        add(sizeof(p_fp4_tile)); add(sizeof(p_sc_tile));
        add(sizeof(v_fp4_tile)); add(sizeof(v_sc_tile));
        return bytes + 1024;
    }
public:
    __host__ __inline__ int dynamic_shared_memory() const { return smem_size(); }
};

template <typename C> __launch_bounds__(512, 1)
__global__ void kernel_mxfp4_nb128_dvo128_qstage2(
    const __grid_constant__ globals_mxfp4_nb128_dvo128_qstage2<C> g) {
    using G = globals_mxfp4_nb128_dvo128_qstage2<C>;
    constexpr int SCORE_BASE=0, OQ0_BASE=128, OQ1_BASE=256;
    constexpr int Q_SCALE_BASE=384, K_SCALE_BASE=400, P_SCALE_BASE=416, V_SCALE_BASE=432;
    static_assert(C::NUM_THREADS == 512 && C::TOTAL_WGS == 4);
    static_assert(V_SCALE_BASE + 16 <= tensor_allocator<1, 1>::cols);
    tensor_allocator<1, 1> tm_alloc{};
    using score_t=tt<float,128,128>; using output_t=tt<float,128,128>;
    score_t tt_score = tm_alloc.template allocate<score_t>(SCORE_BASE);
    output_t tt_output0 = tm_alloc.template allocate<output_t>(OQ0_BASE);
    output_t tt_output1 = tm_alloc.template allocate<output_t>(OQ1_BASE);
    using qk_scale_t = full_tt_fp8e4m3<16>;
    using pv_scale_t = full_tt_fp8e8m0<16>;
    qk_scale_t q_sc_tm = tm_alloc.template allocate<qk_scale_t>(Q_SCALE_BASE);
    qk_scale_t k_sc_tm = tm_alloc.template allocate<qk_scale_t>(K_SCALE_BASE);
    pv_scale_t p_sc_tm = tm_alloc.template allocate<pv_scale_t>(P_SCALE_BASE);
    pv_scale_t v_sc_tm = tm_alloc.template allocate<pv_scale_t>(V_SCALE_BASE);
    extern __shared__ int qstage2_smem[];
    tma_swizzle_allocator al((int *)&qstage2_smem[0]);
    typename G::q_tile &q_smem0 = al.allocate<typename G::q_tile>();
    typename G::q_tile &q_smem1 = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem0 = al.allocate<typename G::q_sc_tile>();
    typename G::q_sc_tile &q_sc_smem1 = al.allocate<typename G::q_sc_tile>();
    using k_tile_t = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    k_tile_t &k_smem = al.allocate<k_tile_t>();
    typename G::k_sc_tile &k_sc_smem = al.allocate<typename G::k_sc_tile>();
    typename G::p_fp4_tile &p_smem = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_smem = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();
    auto load_payload = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int bytes = 4;
        static_assert(Tile::cols % bytes == 0);
        constexpr int stores = Tile::rows * (Tile::cols / bytes);
        const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
        for (int i=threadIdx.x; i<stores; i+=blockDim.x) {
            const int row=i/(Tile::cols/bytes), col=(i%(Tile::cols/bytes))*bytes;
            const uint32_t value=*reinterpret_cast<const uint32_t *>(src+row*Tile::cols+col);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(Tile::idx(base,{row,col})),"r"(value));
        }
    };
    auto load_scale_record = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        const uint32_t base=__cvta_generic_to_shared(&tile.data[0]);
        for (int i=threadIdx.x;i<(int)sizeof(Tile)/4;i+=blockDim.x) {
            const uint32_t value=*reinterpret_cast<const uint32_t *>(src+i*4);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(base+i*4),"r"(value));
        }
    };
    const int block_query=blockIdx.x/g.heads, h=blockIdx.x%g.heads;
    const float head_softmax_scale_log2 = g.softmax_scale_log2 * g.q_sg[h] * g.k_sg[h];
    const int query_tile[2]={2*block_query,2*block_query+1};
    const bool valid[2]={query_tile[0]<g.query_tiles,query_tile[1]<g.query_tiles};
    __shared__ semaphore qk_done[2][2];
    __shared__ semaphore pv_src_done[2][2];
    __shared__ semaphore pv_done[2][2];
    __shared__ float stage_acc_scale[2][128];
    __shared__ int k_load_count;
    __shared__ int v_load_count;
    __shared__ int qk_issue_count[2];
    __shared__ int pv_issue_count[2];
    if(threadIdx.x==0){for(int s=0;s<2;++s)for(int slot=0;slot<2;++slot){init_semaphore(qk_done[s][slot],0,1);init_semaphore(pv_src_done[s][slot],0,1);init_semaphore(pv_done[s][slot],0,1);}
        k_load_count=0; v_load_count=0; qk_issue_count[0]=0; qk_issue_count[1]=0; pv_issue_count[0]=0; pv_issue_count[1]=0;
        g.trace[blockIdx.x*32+0]=block_query; g.trace[blockIdx.x*32+1]=valid[0]; g.trace[blockIdx.x*32+2]=valid[1];}
    __syncthreads();
    for (int stage=0; stage<2; ++stage) if (valid[stage]) {
        auto load_q = [&](auto &qt, auto &qs) {
            load_payload(qt, g.q_fp4 + (query_tile[stage]*g.heads+h)*sizeof(typename G::q_tile));
            load_scale_record(qs, g.q_sc + (query_tile[stage]*g.heads+h)*sizeof(typename G::q_sc_tile));
        };
        if (stage == 0) load_q(q_smem0, q_sc_smem0); else load_q(q_smem1, q_sc_smem1);
    }
    __syncthreads();
    const int last_key=valid[1]?query_tile[1]:query_tile[0];
    float row_max_state[2] = {base_types::constants<float>::neg_infty(), base_types::constants<float>::neg_infty()};
    float row_sum_state[2] = {0.0f, 0.0f};
    auto load_v_once = [&](int key_tile) {
        using VTile = typename G::v_fp4_tile;
        constexpr int bytes = 4;
        constexpr int stores = VTile::rows * (VTile::cols / bytes);
        const int packed_seqlen = g.seqlen / 2;
        const uint32_t base = __cvta_generic_to_shared(&v_smem.data[0]);
        for (int linear=threadIdx.x; linear<stores; linear+=blockDim.x) {
            const int row=linear/(VTile::cols/bytes), col=(linear%(VTile::cols/bytes))*bytes;
            const int global_row=h*C::Dvo+row;
            const uint32_t value=*reinterpret_cast<const uint32_t *>(g.v_fp4+global_row*packed_seqlen+key_tile*64+col);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(VTile::idx(base,{row,col})),"r"(value));
        }
        load_scale_record(v_sc_smem, g.v_sc+(h*g.query_tiles+key_tile)*sizeof(typename G::v_sc_tile));
    };
    auto issue_stage_qk = [&](auto &q_tile_ref, auto &q_sc_ref, int stage, int slot, int phase) {
        using T_AB = typename G::q_tile::T;
        using T_SAB = typename qk_scale_t::T;
        using T_D = typename score_t::T;
        constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D,T_AB,T_SAB,C::Mb,C::Nb,false,0>();
        st_descriptor<typename G::q_tile,0> q_desc(q_tile_ref);
        st_descriptor<k_tile_t,0> k_desc(k_smem);
        const bool issue_warp = warpgroup::groupid()==0 && warpgroup::warpid()==0;
        const bool issue_lane = issue_warp && warp::elect_leader();
        for (int chunk=0; chunk<3; ++chunk) {
            if (issue_lane) {
                fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_ref, chunk);
                fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm, k_sc_smem, chunk);
            }
            if (issue_warp) { tensor_load_wait(); tensor_before_thread_sync(); }
            if (issue_lane) {
                if (chunk == 0) {
                detail::tcgen05::template st_st<T_AB,T_SAB,0,1,16>(
                    tt_score.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                    q_sc_tm.addr, k_sc_tm.addr, IDESC);
                } else {
                detail::tcgen05::template st_st<T_AB,T_SAB,1,1,16>(
                    tt_score.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                    q_sc_tm.addr, k_sc_tm.addr, IDESC);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done[stage][slot]);
            atomicAdd(&qk_issue_count[stage], 1);
        }
        (void)phase;
    };
    auto finish_output = [&](int stage, output_t &tt_output) {
        if (warpgroup::groupid()!=1) return;
        const int row = warpgroup::laneid();
        const bool finite = row_sum_state[stage] > 0.0f && isfinite(row_sum_state[stage]);
        const float scale = finite ? (1.0f/(36.0f * FP4PV_MXFP4_ONLINE_P_SCALE)) / row_sum_state[stage] : 0.0f;
        for (int col=0; col<128; col+=16) {
            float2 out[8];
            const uint32_t addr = tt_output.addr + ((warpgroup::warpid()*32)<<16) + col;
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(out[0].x),"=f"(out[0].y),"=f"(out[1].x),"=f"(out[1].y),"=f"(out[2].x),"=f"(out[2].y),"=f"(out[3].x),"=f"(out[3].y),"=f"(out[4].x),"=f"(out[4].y),"=f"(out[5].x),"=f"(out[5].y),"=f"(out[6].x),"=f"(out[6].y),"=f"(out[7].x),"=f"(out[7].y) : "r"(addr));
            tensor_load_wait(); tensor_before_thread_sync();
            for (int i=0;i<8;++i) {
                const bf16_2 packed=__float22bfloat162_rn(__fmul2_rn(out[i],float2{scale,scale}));
                const int out_offset=(((query_tile[stage]*128+row)*g.heads+h)*128)+col+i*2;
                *reinterpret_cast<bf16_2 *>(&g.o[out_offset])=packed;
            }
        }
        const bool lse_valid=row_sum_state[stage]>0.0f && isfinite(row_sum_state[stage]);
        float lse_value=base_types::constants<float>::neg_infty();
        if (lse_valid) { float log2_sum; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_sum) : "f"(row_sum_state[stage])); lse_value=(row_max_state[stage]*head_softmax_scale_log2+log2_sum)*0.693147180559945f; }
        g.lse[((h*g.seqlen) + query_tile[stage]*128 + row)] = lse_value;
    };
    for(int key_tile=0;key_tile<=last_key;++key_tile){
        const int slot=key_tile&1, phase=(key_tile>>1)&1;
        load_payload(k_smem, g.k_fp4 + (h*g.query_tiles+key_tile)*sizeof(k_tile_t));
        load_scale_record(k_sc_smem, g.k_sc + (h*g.query_tiles+key_tile)*sizeof(typename G::k_sc_tile));
        load_v_once(key_tile);
        __syncthreads();
        fp4pv_publish_shared_backing_proxy_only();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(&k_load_count, 1);
        if (threadIdx.x == 0) atomicAdd(&v_load_count, 1);
        for(int stage=0;stage<2;++stage) if(valid[stage]&&key_tile<=query_tile[stage]){
            if(stage==0) issue_stage_qk(q_smem0,q_sc_smem0,stage,slot,phase);
            else issue_stage_qk(q_smem1,q_sc_smem1,stage,slot,phase);
            if (warpgroup::groupid()==1) {
                wait(qk_done[stage][slot],phase);
                tensor_after_thread_sync();
                const uint32_t score_tt_base = tt_score.addr + ((warpgroup::warpid()*32)<<16);
                float block_max[4]; float tile_max=base_types::constants<float>::neg_infty();
                const bool diagonal = key_tile == query_tile[stage];
                for(int qid=0;qid<4;++qid){float2 s[16];fp4pv_load_score_quarter_tmem(s,score_tt_base,qid);tensor_load_wait();float m=base_types::constants<float>::neg_infty();for(int i=0;i<16;++i){int kc=key_tile*128+qid*32+i*2;if(diagonal&&kc>query_tile[stage]*128+warpgroup::laneid())s[i].x=base_types::constants<float>::neg_infty();if(diagonal&&kc+1>query_tile[stage]*128+warpgroup::laneid())s[i].y=base_types::constants<float>::neg_infty();m=fmaxf(m,fmaxf(s[i].x,s[i].y));}block_max[qid]=m;tile_max=fmaxf(tile_max,m);}
                const float old_max=row_max_state[stage]; row_max_state[stage]=fmaxf(old_max,tile_max); const float acc_scale=key_tile?softmax_rescale_factor_strict(row_max_state[stage],old_max,head_softmax_scale_log2):1.0f; stage_acc_scale[stage][warpgroup::laneid()]=acc_scale;
                uint32_t packed_scales=0; float tile_sum=0.0f;
                for(int qid=0;qid<4;++qid){uint8_t pe=0;if(isfinite(block_max[qid])&&isfinite(row_max_state[stage])) pe=fp4pv_log2_to_e8m0_floor_finite(FP4PV_MXFP4_ONLINE_LOG2_P_SCALE+(block_max[qid]-row_max_state[stage])*head_softmax_scale_log2);packed_scales|=(uint32_t)pe<<(qid*8);float rc=(pe==0)?0.0f:fp4pv_exp2f_e8m0(pe)*FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;float bias=-row_max_state[stage]*head_softmax_scale_log2+((pe==0)?0.0f:(FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE-(float)((int)pe-127)));float2 sscale={head_softmax_scale_log2,head_softmax_scale_log2},bias2={bias,bias};float2 s[16];fp4pv_load_score_quarter_tmem(s,score_tt_base,qid);tensor_load_wait();float2 sum={0,0};for(int i=0;i<16;++i){int kc=key_tile*128+qid*32+i*2;if(diagonal&&kc>query_tile[stage]*128+warpgroup::laneid())s[i].x=base_types::constants<float>::neg_infty();if(diagonal&&kc+1>query_tile[stage]*128+warpgroup::laneid())s[i].y=base_types::constants<float>::neg_infty();s[i]=__ffma2_rn(s[i],sscale,bias2);s[i].x=fp4pv_exp2_approx(s[i].x);s[i].y=fp4pv_exp2_approx(s[i].y);sum=__fadd2_rn(sum,s[i]);}tile_sum+=(sum.x+sum.y)*rc;uint32_t words[4];for(int p=0;p<4;++p)words[p]=(pe==0)?0u:fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(&s[p*4]);fp4pv_store_quantized_scores_group_mxfp4(p_smem,words,warpgroup::laneid(),qid);}
                fp4pv_store_mxfp4_scale_word_swizzled(p_sc_smem,warpgroup::laneid(),packed_scales); row_sum_state[stage]=row_sum_state[stage]*acc_scale+tile_sum;
            }
            __syncthreads();
            fp4pv_publish_shared_backing_proxy_only();
            __syncthreads();
            if (key_tile > 0 && warpgroup::groupid()==1) {
                rescale_tt_output_inplace_direct_streaming<C>(stage ? tt_output1 : tt_output0, stage_acc_scale[stage][warpgroup::laneid()], true);
            }
            __syncthreads();
            if (warpgroup::groupid()==0) {
                const bool issue_warp = warpgroup::warpid()==0;
                const bool issue_lane = issue_warp && warp::elect_leader();
                if (issue_lane) {
                    fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm,p_sc_smem);
                    fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm,v_sc_smem);
                }
                if (issue_warp) { tensor_load_wait(); tensor_before_thread_sync(); }
                if (issue_lane) {
                    if (key_tile == 0) {
                        fp4pv_p_stage_ABt_split_k64<C, true>(stage ? tt_output1 : tt_output0,
                            p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_done[stage][slot]);
                    } else {
                        fp4pv_p_stage_ABt_split_k64<C, false>(stage ? tt_output1 : tt_output0,
                            p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_done[stage][slot]);
                    }
                    tensor_commit<1>(pv_done[stage][slot]);
                    atomicAdd(&pv_issue_count[stage], 1);
                }
            }
            if (warpgroup::groupid()==1) {
                wait(pv_done[stage][slot],phase);
                tensor_after_thread_sync();
            }
            __syncthreads();
        }
    }
    if (valid[0]) finish_output(0,tt_output0);
    if (valid[1]) finish_output(1,tt_output1);
    __syncthreads();
    if(threadIdx.x==0){g.trace[blockIdx.x*32+3]=last_key; g.trace[blockIdx.x*32+6]=k_load_count; g.trace[blockIdx.x*32+7]=qk_issue_count[0]; g.trace[blockIdx.x*32+8]=qk_issue_count[1]; g.trace[blockIdx.x*32+9]=pv_issue_count[0]; g.trace[blockIdx.x*32+10]=pv_issue_count[1]; g.trace[blockIdx.x*32+11]=v_load_count;}
}

#endif

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
template <typename C> struct globals_mxfp4_nb128_dvo128_qstage2_pipeline {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = typename base::v_fp4_tile;
    using v_sc_tile = typename base::v_sc_tile;
    const uint8_t *q_fp4, *q_sc, *k_fp4, *k_sc;
    const float *q_sg, *k_sg;
    const uint8_t *v_fp4, *v_sc;
    bf16 *o; float *lse; unsigned long long *trace;
    int query_tiles, heads, seqlen;
    float softmax_scale_log2;
    __host__ __inline__ dim3 grid() const { return dim3(((query_tiles + 1) / 2) * heads); }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
private:
    static constexpr int smem_size() {
        int bytes = 0;
        auto add = [&](int n) constexpr { bytes = align_up_constexpr(bytes, 1024) + n; };
        add(sizeof(q_tile)); add(sizeof(q_tile));
        add(sizeof(q_sc_tile)); add(sizeof(q_sc_tile));
        add(sizeof(k_tile)); add(sizeof(k_tile));
        add(sizeof(k_sc_tile)); add(sizeof(k_sc_tile));
        add(sizeof(p_fp4_tile)); add(sizeof(p_fp4_tile));
        add(sizeof(p_sc_tile)); add(sizeof(p_sc_tile));
        add(sizeof(v_fp4_tile)); add(sizeof(v_fp4_tile));
        add(sizeof(v_sc_tile)); add(sizeof(v_sc_tile));
        return bytes + 1024;
    }
public:
    __host__ __inline__ int dynamic_shared_memory() const { return smem_size(); }
};

template <typename C> __launch_bounds__(512, 1)
__global__ void kernel_mxfp4_nb128_dvo128_qstage2_pipeline(
    const __grid_constant__ globals_mxfp4_nb128_dvo128_qstage2_pipeline<C> g) {
    using G = globals_mxfp4_nb128_dvo128_qstage2_pipeline<C>;
    constexpr int SCORE_BASE=0, OQ0_BASE=128, OQ1_BASE=256;
    constexpr int Q_SCALE_BASE=384, K_SCALE_BASE=400, P_SCALE_BASE=416, V_SCALE_BASE=432;
    static_assert(C::NUM_THREADS == 512 && C::TOTAL_WGS == 4);
    static_assert(V_SCALE_BASE + 16 <= tensor_allocator<1, 1>::cols);
    tensor_allocator<1, 1> tm_alloc{};
    using score_t=tt<float,128,128>; using output_t=tt<float,128,128>;
    score_t tt_score = tm_alloc.template allocate<score_t>(SCORE_BASE);
    output_t tt_output0 = tm_alloc.template allocate<output_t>(OQ0_BASE);
    output_t tt_output1 = tm_alloc.template allocate<output_t>(OQ1_BASE);
    using qk_scale_t = full_tt_fp8e4m3<16>;
    using pv_scale_t = full_tt_fp8e8m0<16>;
    qk_scale_t q_sc_tm = tm_alloc.template allocate<qk_scale_t>(Q_SCALE_BASE);
    qk_scale_t k_sc_tm = tm_alloc.template allocate<qk_scale_t>(K_SCALE_BASE);
    pv_scale_t p_sc_tm = tm_alloc.template allocate<pv_scale_t>(P_SCALE_BASE);
    pv_scale_t v_sc_tm = tm_alloc.template allocate<pv_scale_t>(V_SCALE_BASE);
    extern __shared__ int qstage2_smem[];
    tma_swizzle_allocator al((int *)&qstage2_smem[0]);
    typename G::q_tile &q_smem0 = al.allocate<typename G::q_tile>();
    typename G::q_tile &q_smem1 = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem0 = al.allocate<typename G::q_sc_tile>();
    typename G::q_sc_tile &q_sc_smem1 = al.allocate<typename G::q_sc_tile>();
    using k_tile_t = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    k_tile_t &k_smem0 = al.allocate<k_tile_t>();
    k_tile_t &k_smem1 = al.allocate<k_tile_t>();
    typename G::k_sc_tile &k_sc_smem0 = al.allocate<typename G::k_sc_tile>();
    typename G::k_sc_tile &k_sc_smem1 = al.allocate<typename G::k_sc_tile>();
    typename G::p_fp4_tile &p_smem0 = al.allocate<typename G::p_fp4_tile>();
    typename G::p_fp4_tile &p_smem1 = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem0 = al.allocate<typename G::p_sc_tile>();
    typename G::p_sc_tile &p_sc_smem1 = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_smem0 = al.allocate<typename G::v_fp4_tile>();
    typename G::v_fp4_tile &v_smem1 = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem0 = al.allocate<typename G::v_sc_tile>();
    typename G::v_sc_tile &v_sc_smem1 = al.allocate<typename G::v_sc_tile>();
    auto load_payload = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int bytes = 4;
        static_assert(Tile::cols % bytes == 0);
        constexpr int stores = Tile::rows * (Tile::cols / bytes);
        const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
        for (int i=warpgroup::laneid(); i<stores; i+=128) {
            const int row=i/(Tile::cols/bytes), col=(i%(Tile::cols/bytes))*bytes;
            const uint32_t value=*reinterpret_cast<const uint32_t *>(src+row*Tile::cols+col);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(Tile::idx(base,{row,col})),"r"(value));
        }
    };
    auto load_scale_record = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        const uint32_t base=__cvta_generic_to_shared(&tile.data[0]);
        for (int i=warpgroup::laneid();i<(int)sizeof(Tile)/4;i+=128) {
            const uint32_t value=*reinterpret_cast<const uint32_t *>(src+i*4);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(base+i*4),"r"(value));
        }
    };
    const int block_query=blockIdx.x/g.heads, h=blockIdx.x%g.heads;
    const float head_softmax_scale_log2 = g.softmax_scale_log2 * g.q_sg[h] * g.k_sg[h];
    const int query_tile[2]={2*block_query,2*block_query+1};
    const bool valid[2]={query_tile[0]<g.query_tiles,query_tile[1]<g.query_tiles};
    __shared__ semaphore kv_full[2], kv_empty[2];
    __shared__ semaphore qk_done[2][2];
    __shared__ semaphore score_released[2][2];
    __shared__ semaphore output_rescaled[2][2];
    __shared__ semaphore p_ready[2][2];
    __shared__ semaphore pv_src_done[2][2];
    __shared__ semaphore pv_done[2][2];
    __shared__ float stage_acc_scale[2][128];
    __shared__ int k_load_count;
    __shared__ int v_load_count;
    __shared__ int qk_issue_count[2];
    __shared__ int pv_issue_count[2];
    if(threadIdx.x==0){for(int slot=0;slot<2;++slot){init_semaphore(kv_full[slot],0,1);init_semaphore(kv_empty[slot],0,1);} for(int s=0;s<2;++s)for(int slot=0;slot<2;++slot){init_semaphore(qk_done[s][slot],0,1);init_semaphore(score_released[s][slot],0,1);init_semaphore(output_rescaled[s][slot],0,1);init_semaphore(p_ready[s][slot],0,1);init_semaphore(pv_src_done[s][slot],0,1);init_semaphore(pv_done[s][slot],0,1);}
        k_load_count=0; v_load_count=0; qk_issue_count[0]=0; qk_issue_count[1]=0; pv_issue_count[0]=0; pv_issue_count[1]=0;
        g.trace[blockIdx.x*32+0]=block_query; g.trace[blockIdx.x*32+1]=valid[0]; g.trace[blockIdx.x*32+2]=valid[1];}
    __syncthreads();
    for (int stage=0; stage<2; ++stage) if (valid[stage]) {
        auto load_q = [&](auto &qt, auto &qs) {
            load_payload(qt, g.q_fp4 + (query_tile[stage]*g.heads+h)*sizeof(typename G::q_tile));
            load_scale_record(qs, g.q_sc + (query_tile[stage]*g.heads+h)*sizeof(typename G::q_sc_tile));
        };
        if (warpgroup::groupid() == 3) { if (stage == 0) load_q(q_smem0, q_sc_smem0); else load_q(q_smem1, q_sc_smem1); }
    }
    __syncthreads();
    const int last_key=valid[1]?query_tile[1]:query_tile[0];
    float row_max_state[2] = {base_types::constants<float>::neg_infty(), base_types::constants<float>::neg_infty()};
    float row_sum_state[2] = {0.0f, 0.0f};
    auto load_v_once = [&](int key_tile, auto &v_slot, auto &v_sc_slot) {
        using VTile = typename G::v_fp4_tile;
        constexpr int bytes = 4;
        constexpr int stores = VTile::rows * (VTile::cols / bytes);
        const int packed_seqlen = g.seqlen / 2;
        const uint32_t base = __cvta_generic_to_shared(&v_slot.data[0]);
        for (int linear=warpgroup::laneid(); linear<stores; linear+=128) {
            const int row=linear/(VTile::cols/bytes), col=(linear%(VTile::cols/bytes))*bytes;
            const int global_row=h*C::Dvo+row;
            const uint32_t value=*reinterpret_cast<const uint32_t *>(g.v_fp4+global_row*packed_seqlen+key_tile*(C::Nb/2)+col);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(VTile::idx(base,{row,col})),"r"(value));
        }
        load_scale_record(v_sc_slot, g.v_sc+(h*g.query_tiles+key_tile)*sizeof(typename G::v_sc_tile));
    };
    auto issue_stage_qk = [&](auto &q_tile_ref, auto &q_sc_ref, auto &k_slot,
                              auto &k_sc_slot, int stage, int slot, int phase) {
        using T_AB = typename G::q_tile::T;
        using T_SAB = typename qk_scale_t::T;
        using T_D = typename score_t::T;
        constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D,T_AB,T_SAB,C::Mb,C::Nb,false,0>();
        st_descriptor<typename G::q_tile,0> q_desc(q_tile_ref);
        st_descriptor<k_tile_t,0> k_desc(k_slot);
        const bool issue_warp = warpgroup::groupid()==0 && warpgroup::warpid()==0;
        const bool issue_lane = issue_warp && warp::elect_leader();
        for (int chunk=0; chunk<3; ++chunk) {
            if (issue_lane) {
                fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_ref, chunk);
                fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm, k_sc_slot, chunk);
            }
            if (issue_warp) { tensor_load_wait(); tensor_before_thread_sync(); }
            if (issue_lane) {
                if (chunk == 0) {
                detail::tcgen05::template st_st<T_AB,T_SAB,0,1,16>(
                    tt_score.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                    q_sc_tm.addr, k_sc_tm.addr, IDESC);
                } else {
                detail::tcgen05::template st_st<T_AB,T_SAB,1,1,16>(
                    tt_score.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                    q_sc_tm.addr, k_sc_tm.addr, IDESC);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done[stage][slot]);
            atomicAdd(&qk_issue_count[stage], 1);
        }
        (void)phase;
    };
    auto finish_output = [&](int stage, output_t &tt_output) {
        if (warpgroup::groupid()!=stage+1) return;
        const int row = warpgroup::laneid();
        const bool finite = row_sum_state[stage] > 0.0f && isfinite(row_sum_state[stage]);
        const float scale = finite ? (1.0f/(36.0f * FP4PV_MXFP4_ONLINE_P_SCALE)) / row_sum_state[stage] : 0.0f;
        for (int col=0; col<128; col+=16) {
            float2 out[8];
            const uint32_t addr = tt_output.addr + ((warpgroup::warpid()*32)<<16) + col;
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(out[0].x),"=f"(out[0].y),"=f"(out[1].x),"=f"(out[1].y),"=f"(out[2].x),"=f"(out[2].y),"=f"(out[3].x),"=f"(out[3].y),"=f"(out[4].x),"=f"(out[4].y),"=f"(out[5].x),"=f"(out[5].y),"=f"(out[6].x),"=f"(out[6].y),"=f"(out[7].x),"=f"(out[7].y) : "r"(addr));
            tensor_load_wait(); tensor_before_thread_sync();
            for (int i=0;i<8;++i) {
                const bf16_2 packed=__float22bfloat162_rn(__fmul2_rn(out[i],float2{scale,scale}));
                const int out_offset=(((query_tile[stage]*128+row)*g.heads+h)*128)+col+i*2;
                *reinterpret_cast<bf16_2 *>(&g.o[out_offset])=packed;
            }
        }
        const bool lse_valid=row_sum_state[stage]>0.0f && isfinite(row_sum_state[stage]);
        float lse_value=base_types::constants<float>::neg_infty();
        if (lse_valid) { float log2_sum; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_sum) : "f"(row_sum_state[stage])); lse_value=(row_max_state[stage]*head_softmax_scale_log2+log2_sum)*0.693147180559945f; }
        g.lse[((h*g.seqlen) + query_tile[stage]*128 + row)] = lse_value;
    };
    for(int key_tile=0;key_tile<=last_key;++key_tile){
        auto &k_smem = (key_tile & 1) ? k_smem1 : k_smem0;
        auto &k_sc_smem = (key_tile & 1) ? k_sc_smem1 : k_sc_smem0;
        auto &v_smem = (key_tile & 1) ? v_smem1 : v_smem0;
        auto &v_sc_smem = (key_tile & 1) ? v_sc_smem1 : v_sc_smem0;
        const int slot=key_tile&1, phase=(key_tile>>1)&1;
        if (warpgroup::groupid() == 3) {
            if (key_tile >= 2 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                wait(kv_empty[slot], phase ^ 1);
            }
            warpgroup::sync(4);
            load_payload(k_smem, g.k_fp4 + (h*g.query_tiles+key_tile)*sizeof(k_tile_t));
            load_scale_record(k_sc_smem, g.k_sc + (h*g.query_tiles+key_tile)*sizeof(typename G::k_sc_tile));
            load_v_once(key_tile, v_smem, v_sc_smem);
            warpgroup::sync(4);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(4);
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                arrive(kv_full[slot]);
            }
        }
        if (threadIdx.x == 0) atomicAdd(&k_load_count, 1);
        if (threadIdx.x == 0) atomicAdd(&v_load_count, 1);
        if (warpgroup::groupid()==0 && warpgroup::warpid()==0 && warp::elect_leader()) {
            wait(kv_full[slot], phase);
        }
        for(int stage=0;stage<2;++stage) if(valid[stage]&&key_tile<=query_tile[stage]){
            auto &p_smem = stage == 0 ? p_smem0 : p_smem1;
            auto &p_sc_smem = stage == 0 ? p_sc_smem0 : p_sc_smem1;
            if(stage==0) issue_stage_qk(q_smem0,q_sc_smem0,k_smem,k_sc_smem,stage,slot,phase);
            else issue_stage_qk(q_smem1,q_sc_smem1,k_smem,k_sc_smem,stage,slot,phase);
            if (warpgroup::groupid()==stage+1) {
                wait(qk_done[stage][slot],phase);
                tensor_after_thread_sync();
                const uint32_t score_tt_base = tt_score.addr + ((warpgroup::warpid()*32)<<16);
                float block_max[4]; float tile_max=base_types::constants<float>::neg_infty();
                const bool diagonal = key_tile == query_tile[stage];
                for(int qid=0;qid<4;++qid){float2 s[16];fp4pv_load_score_quarter_tmem(s,score_tt_base,qid);tensor_load_wait();float m=base_types::constants<float>::neg_infty();for(int i=0;i<16;++i){int kc=key_tile*128+qid*32+i*2;if(diagonal&&kc>query_tile[stage]*128+warpgroup::laneid())s[i].x=base_types::constants<float>::neg_infty();if(diagonal&&kc+1>query_tile[stage]*128+warpgroup::laneid())s[i].y=base_types::constants<float>::neg_infty();m=fmaxf(m,fmaxf(s[i].x,s[i].y));}block_max[qid]=m;tile_max=fmaxf(tile_max,m);}
                const float old_max=row_max_state[stage]; row_max_state[stage]=fmaxf(old_max,tile_max); const float acc_scale=key_tile?softmax_rescale_factor_strict(row_max_state[stage],old_max,head_softmax_scale_log2):1.0f; stage_acc_scale[stage][warpgroup::laneid()]=acc_scale;
                uint32_t packed_scales=0; float tile_sum=0.0f;
                for(int qid=0;qid<4;++qid){uint8_t pe=0;if(isfinite(block_max[qid])&&isfinite(row_max_state[stage])) pe=fp4pv_log2_to_e8m0_floor_finite(FP4PV_MXFP4_ONLINE_LOG2_P_SCALE+(block_max[qid]-row_max_state[stage])*head_softmax_scale_log2);packed_scales|=(uint32_t)pe<<(qid*8);float rc=(pe==0)?0.0f:fp4pv_exp2f_e8m0(pe)*FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;float bias=-row_max_state[stage]*head_softmax_scale_log2+((pe==0)?0.0f:(FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE-(float)((int)pe-127)));float2 sscale={head_softmax_scale_log2,head_softmax_scale_log2},bias2={bias,bias};float2 s[16];fp4pv_load_score_quarter_tmem(s,score_tt_base,qid);tensor_load_wait();float2 sum={0,0};for(int i=0;i<16;++i){int kc=key_tile*128+qid*32+i*2;if(diagonal&&kc>query_tile[stage]*128+warpgroup::laneid())s[i].x=base_types::constants<float>::neg_infty();if(diagonal&&kc+1>query_tile[stage]*128+warpgroup::laneid())s[i].y=base_types::constants<float>::neg_infty();s[i]=__ffma2_rn(s[i],sscale,bias2);s[i].x=fp4pv_exp2_approx(s[i].x);s[i].y=fp4pv_exp2_approx(s[i].y);sum=__fadd2_rn(sum,s[i]);}tile_sum+=(sum.x+sum.y)*rc;uint32_t words[4];for(int p=0;p<4;++p)words[p]=(pe==0)?0u:fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(&s[p*4]);fp4pv_store_quantized_scores_group_mxfp4(p_smem,words,warpgroup::laneid(),qid);}
                fp4pv_store_mxfp4_scale_word_swizzled(p_sc_smem,warpgroup::laneid(),packed_scales); row_sum_state[stage]=row_sum_state[stage]*acc_scale+tile_sum;
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                fp4pv_publish_shared_backing_proxy_only();
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                    arrive(score_released[stage][slot]);
                    arrive(p_ready[stage][slot]);
                }
            }
            if (warpgroup::groupid()==stage+1) {
                if (key_tile > 0) {
                    rescale_tt_output_inplace_direct_streaming<C>(stage ? tt_output1 : tt_output0, stage_acc_scale[stage][warpgroup::laneid()], true);
                    tensor_load_wait();
                }
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                if (warpgroup::warpid() == 0 && warp::elect_leader()) arrive(output_rescaled[stage][slot]);
            }
            if (warpgroup::groupid()==0) {
                const bool issue_warp = warpgroup::warpid()==0;
                const bool issue_lane = issue_warp && warp::elect_leader();
                if (issue_lane) {
                    wait(score_released[stage][slot], phase);
                    wait(p_ready[stage][slot], phase);
                    wait(output_rescaled[stage][slot], phase);
                    fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm,p_sc_smem);
                    fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm,v_sc_smem);
                }
                if (issue_warp) { tensor_load_wait(); tensor_before_thread_sync(); }
                if (issue_lane) {
                    if (key_tile == 0) {
                        fp4pv_p_stage_ABt_split_k64<C, true>(stage ? tt_output1 : tt_output0,
                            p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_done[stage][slot]);
                    } else {
                        fp4pv_p_stage_ABt_split_k64<C, false>(stage ? tt_output1 : tt_output0,
                            p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_done[stage][slot]);
                    }
                    tensor_commit<1>(pv_done[stage][slot]);
                    atomicAdd(&pv_issue_count[stage], 1);
                }
            }
            if (warpgroup::groupid()==stage+1) {
                wait(pv_done[stage][slot],phase);
                tensor_after_thread_sync();
            }
        }
        if (warpgroup::groupid()==0 && warpgroup::warpid()==0 && warp::elect_leader()) {
            const int last_consumer_stage = valid[1] ? 1 : 0;
            wait(pv_src_done[last_consumer_stage][slot], phase);
            arrive(kv_empty[slot]);
        }
    }
    if (valid[0]) finish_output(0,tt_output0);
    if (valid[1]) finish_output(1,tt_output1);
    __syncthreads();
    if(threadIdx.x==0){g.trace[blockIdx.x*32+3]=last_key; g.trace[blockIdx.x*32+6]=k_load_count; g.trace[blockIdx.x*32+7]=qk_issue_count[0]; g.trace[blockIdx.x*32+8]=qk_issue_count[1]; g.trace[blockIdx.x*32+9]=pv_issue_count[0]; g.trace[blockIdx.x*32+10]=pv_issue_count[1]; g.trace[blockIdx.x*32+11]=v_load_count;}
}
#endif

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
#include "fwd_qstage2_two_n64.inc"
/* two_n64 implementation is isolated in fwd_qstage2_two_n64.inc */
#if 0  // Stale in-place clone; remove after the isolated include builds.
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using physical_k_tile = st_fp4e2m1_2<128, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using physical_v_tile = st_fp4e2m1_2<C::Dvo, 64>;
    using physical_k_sc_tile = typename base::k_sc_tile;
    using physical_v_sc_tile = typename base::v_sc_tile;
    const uint8_t *q_fp4, *q_sc, *k_fp4, *k_sc;
    const float *q_sg, *k_sg;
    const uint8_t *v_fp4, *v_sc;
    bf16 *o; float *lse; unsigned long long *trace;
    int query_tiles, heads, seqlen;
    float softmax_scale_log2;
    __host__ __inline__ dim3 grid() const { return dim3(((query_tiles + 1) / 2) * heads); }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
private:
    static constexpr int smem_size() {
        int bytes = 0;
        auto add = [&](int n) constexpr { bytes = align_up_constexpr(bytes, 1024) + n; };
        add(sizeof(q_tile)); add(sizeof(q_tile));
        add(sizeof(q_sc_tile)); add(sizeof(q_sc_tile));
        add(sizeof(physical_k_tile)); add(sizeof(physical_k_tile));
        add(sizeof(physical_k_sc_tile)); add(sizeof(physical_k_sc_tile));
        add(sizeof(p_fp4_tile)); add(sizeof(p_fp4_tile));
        add(sizeof(p_sc_tile)); add(sizeof(p_sc_tile));
        add(sizeof(physical_v_tile)); add(sizeof(physical_v_tile));
        add(sizeof(physical_v_sc_tile)); add(sizeof(physical_v_sc_tile));
        return bytes + 1024;
    }
public:
    __host__ __inline__ int dynamic_shared_memory() const { return smem_size(); }
};

template <typename C> __launch_bounds__(512, 1)
__global__ void kernel_mxfp4_nb128_dvo128_qstage2_two_n64(
    const __grid_constant__ globals_mxfp4_nb128_dvo128_qstage2_two_n64<C> g) {
    using G = globals_mxfp4_nb128_dvo128_qstage2_two_n64<C>;
    constexpr int SCORE_BASE=0, OQ0_BASE=128, OQ1_BASE=256;
    constexpr int Q_SCALE_BASE=384, K_SCALE_BASE=400, P_SCALE_BASE=416, V_SCALE_BASE=432;
    static_assert(C::NUM_THREADS == 512 && C::TOTAL_WGS == 4);
    static_assert(V_SCALE_BASE + 16 <= tensor_allocator<1, 1>::cols);
    tensor_allocator<1, 1> tm_alloc{};
    using score_t=tt<float,128,64>; using output_t=tt<float,128,128>;
    score_t tt_score0 = tm_alloc.template allocate<score_t>(0);
    score_t tt_score1 = tm_alloc.template allocate<score_t>(64);
    auto score_for_stage = [&](int stage) -> score_t & { return stage == 0 ? tt_score0 : tt_score1; };
    score_t &tt_score = tt_score0;
    output_t tt_output0 = tm_alloc.template allocate<output_t>(OQ0_BASE);
    output_t tt_output1 = tm_alloc.template allocate<output_t>(OQ1_BASE);
    using qk_scale_t = full_tt_fp8e4m3<16>;
    using pv_scale_t = full_tt_fp8e8m0<16>;
    qk_scale_t q_sc_tm = tm_alloc.template allocate<qk_scale_t>(Q_SCALE_BASE);
    qk_scale_t k_sc_tm = tm_alloc.template allocate<qk_scale_t>(K_SCALE_BASE);
    pv_scale_t p_sc_tm = tm_alloc.template allocate<pv_scale_t>(P_SCALE_BASE);
    pv_scale_t v_sc_tm = tm_alloc.template allocate<pv_scale_t>(V_SCALE_BASE);
    extern __shared__ int qstage2_smem[];
    tma_swizzle_allocator al((int *)&qstage2_smem[0]);
    typename G::q_tile &q_smem0 = al.allocate<typename G::q_tile>();
    typename G::q_tile &q_smem1 = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem0 = al.allocate<typename G::q_sc_tile>();
    typename G::q_sc_tile &q_sc_smem1 = al.allocate<typename G::q_sc_tile>();
    using k_tile_t = typename G::k_tile;
    using physical_k_tile_t = typename G::physical_k_tile;
    using physical_v_tile_t = typename G::physical_v_tile;
    physical_k_tile_t &k_full0 = al.allocate<physical_k_tile_t>();
    physical_k_tile_t &k_full1 = al.allocate<physical_k_tile_t>();
    typename G::physical_k_sc_tile &k_sc_full0 = al.allocate<typename G::physical_k_sc_tile>();
    typename G::physical_k_sc_tile &k_sc_full1 = al.allocate<typename G::physical_k_sc_tile>();
    typename G::p_fp4_tile &p_smem0 = al.allocate<typename G::p_fp4_tile>();
    typename G::p_fp4_tile &p_smem1 = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem0 = al.allocate<typename G::p_sc_tile>();
    typename G::p_sc_tile &p_sc_smem1 = al.allocate<typename G::p_sc_tile>();
    physical_v_tile_t &v_full0 = al.allocate<physical_v_tile_t>();
    physical_v_tile_t &v_full1 = al.allocate<physical_v_tile_t>();
    typename G::physical_v_sc_tile &v_sc_full0 = al.allocate<typename G::physical_v_sc_tile>();
    typename G::physical_v_sc_tile &v_sc_full1 = al.allocate<typename G::physical_v_sc_tile>();
    auto load_payload = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int bytes = 4;
        static_assert(Tile::cols % bytes == 0);
        constexpr int stores = Tile::rows * (Tile::cols / bytes);
        const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
        for (int i=warpgroup::laneid(); i<stores; i+=128) {
            const int row=i/(Tile::cols/bytes), col=(i%(Tile::cols/bytes))*bytes;
            const uint32_t value=*reinterpret_cast<const uint32_t *>(src+row*Tile::cols+col);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(Tile::idx(base,{row,col})),"r"(value));
        }
    };
    auto load_scale_record = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        const uint32_t base=__cvta_generic_to_shared(&tile.data[0]);
        for (int i=warpgroup::laneid();i<(int)sizeof(Tile)/4;i+=128) {
            const uint32_t value=*reinterpret_cast<const uint32_t *>(src+i*4);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(base+i*4),"r"(value));
        }
    };
    const int block_query=blockIdx.x/g.heads, h=blockIdx.x%g.heads;
    const float head_softmax_scale_log2 = g.softmax_scale_log2 * g.q_sg[h] * g.k_sg[h];
    const int query_tile[2]={2*block_query,2*block_query+1};
    const bool valid[2]={query_tile[0]<g.query_tiles,query_tile[1]<g.query_tiles};
    __shared__ semaphore kv_full[2], kv_empty[2];
    __shared__ semaphore qk_done[2][2];
    __shared__ semaphore score_released[2][2];
    __shared__ semaphore output_rescaled[2][2];
    __shared__ semaphore p_ready[2][2];
    __shared__ semaphore pv_src_done[2][2];
    __shared__ semaphore pv_done[2][2];
    __shared__ float stage_acc_scale[2][128];
    __shared__ int k_load_count;
    __shared__ int v_load_count;
    __shared__ int qk_issue_count[2];
    __shared__ int pv_issue_count[2];
    if(threadIdx.x==0){for(int slot=0;slot<2;++slot){init_semaphore(kv_full[slot],0,1);init_semaphore(kv_empty[slot],0,1);} for(int s=0;s<2;++s)for(int slot=0;slot<2;++slot){init_semaphore(qk_done[s][slot],0,1);init_semaphore(score_released[s][slot],0,1);init_semaphore(output_rescaled[s][slot],0,1);init_semaphore(p_ready[s][slot],0,1);init_semaphore(pv_src_done[s][slot],0,1);init_semaphore(pv_done[s][slot],0,1);}
        k_load_count=0; v_load_count=0; qk_issue_count[0]=0; qk_issue_count[1]=0; pv_issue_count[0]=0; pv_issue_count[1]=0;
        g.trace[blockIdx.x*32+0]=block_query; g.trace[blockIdx.x*32+1]=valid[0]; g.trace[blockIdx.x*32+2]=valid[1];}
    __syncthreads();
    for (int stage=0; stage<2; ++stage) if (valid[stage]) {
        auto load_q = [&](auto &qt, auto &qs) {
            load_payload(qt, g.q_fp4 + (query_tile[stage]*g.heads+h)*sizeof(typename G::q_tile));
            load_scale_record(qs, g.q_sc + (query_tile[stage]*g.heads+h)*sizeof(typename G::q_sc_tile));
        };
        if (warpgroup::groupid() == 3) { if (stage == 0) load_q(q_smem0, q_sc_smem0); else load_q(q_smem1, q_sc_smem1); }
    }
    __syncthreads();
    const int last_key=valid[1]?query_tile[1]:query_tile[0];
    float row_max_state[2] = {base_types::constants<float>::neg_infty(), base_types::constants<float>::neg_infty()};
    float row_sum_state[2] = {0.0f, 0.0f};
    auto load_v_once = [&](int key_tile, auto &v_slot, auto &v_sc_slot) {
        using VTile = typename G::v_fp4_tile;
        constexpr int bytes = 4;
        constexpr int stores = VTile::rows * (VTile::cols / bytes);
        const int packed_seqlen = g.seqlen / 2;
        const uint32_t base = __cvta_generic_to_shared(&v_slot.data[0]);
        for (int linear=warpgroup::laneid(); linear<stores; linear+=128) {
            const int row=linear/(VTile::cols/bytes), col=(linear%(VTile::cols/bytes))*bytes;
            const int global_row=h*C::Dvo+row;
            const uint32_t value=*reinterpret_cast<const uint32_t *>(g.v_fp4+global_row*packed_seqlen+key_tile*(C::Nb/2)+col);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(VTile::idx(base,{row,col})),"r"(value));
        }
        load_scale_record(v_sc_slot, g.v_sc+(h*g.query_tiles+key_tile)*sizeof(typename G::v_sc_tile));
    };
    auto issue_stage_qk = [&](auto &q_tile_ref, auto &q_sc_ref, auto &k_slot,
                              auto &k_sc_slot, int stage, int slot, int phase) {
        using T_AB = typename G::q_tile::T;
        using T_SAB = typename qk_scale_t::T;
        using T_D = typename score_t::T;
        constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D,T_AB,T_SAB,C::Mb,C::Nb,false,0>();
        st_descriptor<typename G::q_tile,0> q_desc(q_tile_ref);
        st_descriptor<k_tile_t,0> k_desc(k_slot);
        const bool issue_warp = warpgroup::groupid()==0 && warpgroup::warpid()==0;
        const bool issue_lane = issue_warp && warp::elect_leader();
        for (int chunk=0; chunk<3; ++chunk) {
            if (issue_lane) {
                fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_ref, chunk);
                fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm, k_sc_slot, chunk);
            }
            if (issue_warp) { tensor_load_wait(); tensor_before_thread_sync(); }
            if (issue_lane) {
                if (chunk == 0) {
                detail::tcgen05::template st_st<T_AB,T_SAB,0,1,16>(
                    score_for_stage(stage).addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                    q_sc_tm.addr, k_sc_tm.addr, IDESC);
                } else {
                detail::tcgen05::template st_st<T_AB,T_SAB,1,1,16>(
                    score_for_stage(stage).addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                    q_sc_tm.addr, k_sc_tm.addr, IDESC);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done[stage][slot]);
            atomicAdd(&qk_issue_count[stage], 1);
        }
        (void)phase;
    };
    auto finish_output = [&](int stage, output_t &tt_output) {
        if (warpgroup::groupid()!=stage+1) return;
        const int row = warpgroup::laneid();
        const bool finite = row_sum_state[stage] > 0.0f && isfinite(row_sum_state[stage]);
        const float scale = finite ? (1.0f/(36.0f * FP4PV_MXFP4_ONLINE_P_SCALE)) / row_sum_state[stage] : 0.0f;
        for (int col=0; col<128; col+=16) {
            float2 out[8];
            const uint32_t addr = tt_output.addr + ((warpgroup::warpid()*32)<<16) + col;
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(out[0].x),"=f"(out[0].y),"=f"(out[1].x),"=f"(out[1].y),"=f"(out[2].x),"=f"(out[2].y),"=f"(out[3].x),"=f"(out[3].y),"=f"(out[4].x),"=f"(out[4].y),"=f"(out[5].x),"=f"(out[5].y),"=f"(out[6].x),"=f"(out[6].y),"=f"(out[7].x),"=f"(out[7].y) : "r"(addr));
            tensor_load_wait(); tensor_before_thread_sync();
            for (int i=0;i<8;++i) {
                const bf16_2 packed=__float22bfloat162_rn(__fmul2_rn(out[i],float2{scale,scale}));
                const int out_offset=(((query_tile[stage]*128+row)*g.heads+h)*128)+col+i*2;
                *reinterpret_cast<bf16_2 *>(&g.o[out_offset])=packed;
            }
        }
        const bool lse_valid=row_sum_state[stage]>0.0f && isfinite(row_sum_state[stage]);
        float lse_value=base_types::constants<float>::neg_infty();
        if (lse_valid) { float log2_sum; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_sum) : "f"(row_sum_state[stage])); lse_value=(row_max_state[stage]*head_softmax_scale_log2+log2_sum)*0.693147180559945f; }
        g.lse[((h*g.seqlen) + query_tile[stage]*128 + row)] = lse_value;
    };
    for(int key_tile=0;key_tile<=last_key;++key_tile){
        auto &k_smem = (key_tile & 1) ? k_smem1 : k_smem0;
        auto &k_sc_smem = (key_tile & 1) ? k_sc_smem1 : k_sc_smem0;
        auto &v_smem = (key_tile & 1) ? v_smem1 : v_smem0;
        auto &v_sc_smem = (key_tile & 1) ? v_sc_smem1 : v_sc_smem0;
        const int slot=key_tile&1, phase=(key_tile>>1)&1;
        if (warpgroup::groupid() == 3) {
            if (key_tile >= 2 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                wait(kv_empty[slot], phase ^ 1);
            }
            warpgroup::sync(4);
            load_payload(k_smem, g.k_fp4 + (h*g.query_tiles+key_tile)*sizeof(k_tile_t));
            load_scale_record(k_sc_smem, g.k_sc + (h*g.query_tiles+key_tile)*sizeof(typename G::k_sc_tile));
            load_v_once(key_tile, v_smem, v_sc_smem);
            warpgroup::sync(4);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(4);
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                arrive(kv_full[slot]);
            }
        }
        if (threadIdx.x == 0) atomicAdd(&k_load_count, 1);
        if (threadIdx.x == 0) atomicAdd(&v_load_count, 1);
        if (warpgroup::groupid()==0 && warpgroup::warpid()==0 && warp::elect_leader()) {
            wait(kv_full[slot], phase);
        }
        for(int stage=0;stage<2;++stage) if(valid[stage]&&key_tile<=query_tile[stage]){
            auto &p_smem = stage == 0 ? p_smem0 : p_smem1;
            auto &p_sc_smem = stage == 0 ? p_sc_smem0 : p_sc_smem1;
            if(stage==0) issue_stage_qk(q_smem0,q_sc_smem0,k_smem,k_sc_smem,stage,slot,phase);
            else issue_stage_qk(q_smem1,q_sc_smem1,k_smem,k_sc_smem,stage,slot,phase);
            if (warpgroup::groupid()==stage+1) {
                wait(qk_done[stage][slot],phase);
                tensor_after_thread_sync();
                const uint32_t score_tt_base = tt_score.addr + ((warpgroup::warpid()*32)<<16);
                float block_max[4]; float tile_max=base_types::constants<float>::neg_infty();
                const bool diagonal = key_tile == query_tile[stage];
                for(int qid=0;qid<4;++qid){float2 s[16];fp4pv_load_score_quarter_tmem(s,score_tt_base,qid);tensor_load_wait();float m=base_types::constants<float>::neg_infty();for(int i=0;i<16;++i){int kc=key_tile*128+qid*32+i*2;if(diagonal&&kc>query_tile[stage]*128+warpgroup::laneid())s[i].x=base_types::constants<float>::neg_infty();if(diagonal&&kc+1>query_tile[stage]*128+warpgroup::laneid())s[i].y=base_types::constants<float>::neg_infty();m=fmaxf(m,fmaxf(s[i].x,s[i].y));}block_max[qid]=m;tile_max=fmaxf(tile_max,m);}
                const float old_max=row_max_state[stage]; row_max_state[stage]=fmaxf(old_max,tile_max); const float acc_scale=key_tile?softmax_rescale_factor_strict(row_max_state[stage],old_max,head_softmax_scale_log2):1.0f; stage_acc_scale[stage][warpgroup::laneid()]=acc_scale;
                uint32_t packed_scales=0; float tile_sum=0.0f;
                for(int qid=0;qid<4;++qid){uint8_t pe=0;if(isfinite(block_max[qid])&&isfinite(row_max_state[stage])) pe=fp4pv_log2_to_e8m0_floor_finite(FP4PV_MXFP4_ONLINE_LOG2_P_SCALE+(block_max[qid]-row_max_state[stage])*head_softmax_scale_log2);packed_scales|=(uint32_t)pe<<(qid*8);float rc=(pe==0)?0.0f:fp4pv_exp2f_e8m0(pe)*FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;float bias=-row_max_state[stage]*head_softmax_scale_log2+((pe==0)?0.0f:(FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE-(float)((int)pe-127)));float2 sscale={head_softmax_scale_log2,head_softmax_scale_log2},bias2={bias,bias};float2 s[16];fp4pv_load_score_quarter_tmem(s,score_tt_base,qid);tensor_load_wait();float2 sum={0,0};for(int i=0;i<16;++i){int kc=key_tile*128+qid*32+i*2;if(diagonal&&kc>query_tile[stage]*128+warpgroup::laneid())s[i].x=base_types::constants<float>::neg_infty();if(diagonal&&kc+1>query_tile[stage]*128+warpgroup::laneid())s[i].y=base_types::constants<float>::neg_infty();s[i]=__ffma2_rn(s[i],sscale,bias2);s[i].x=fp4pv_exp2_approx(s[i].x);s[i].y=fp4pv_exp2_approx(s[i].y);sum=__fadd2_rn(sum,s[i]);}tile_sum+=(sum.x+sum.y)*rc;uint32_t words[4];for(int p=0;p<4;++p)words[p]=(pe==0)?0u:fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(&s[p*4]);fp4pv_store_quantized_scores_group_mxfp4(p_smem,words,warpgroup::laneid(),qid);}
                fp4pv_store_mxfp4_scale_word_swizzled(p_sc_smem,warpgroup::laneid(),packed_scales); row_sum_state[stage]=row_sum_state[stage]*acc_scale+tile_sum;
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                fp4pv_publish_shared_backing_proxy_only();
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                    arrive(score_released[stage][slot]);
                    arrive(p_ready[stage][slot]);
                }
            }
            if (warpgroup::groupid()==stage+1) {
                if (key_tile > 0) {
                    rescale_tt_output_inplace_direct_streaming<C>(stage ? tt_output1 : tt_output0, stage_acc_scale[stage][warpgroup::laneid()], true);
                    tensor_load_wait();
                }
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                if (warpgroup::warpid() == 0 && warp::elect_leader()) arrive(output_rescaled[stage][slot]);
            }
            if (warpgroup::groupid()==0) {
                const bool issue_warp = warpgroup::warpid()==0;
                const bool issue_lane = issue_warp && warp::elect_leader();
                if (issue_lane) {
                    wait(score_released[stage][slot], phase);
                    wait(p_ready[stage][slot], phase);
                    wait(output_rescaled[stage][slot], phase);
                    fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm,p_sc_smem);
                    fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm,v_sc_smem);
                }
                if (issue_warp) { tensor_load_wait(); tensor_before_thread_sync(); }
                if (issue_lane) {
                    if (key_tile == 0) {
                        fp4pv_p_stage_ABt_split_k64<C, true>(stage ? tt_output1 : tt_output0,
                            p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_done[stage][slot]);
                    } else {
                        fp4pv_p_stage_ABt_split_k64<C, false>(stage ? tt_output1 : tt_output0,
                            p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_done[stage][slot]);
                    }
                    tensor_commit<1>(pv_done[stage][slot]);
                    atomicAdd(&pv_issue_count[stage], 1);
                }
            }
            if (warpgroup::groupid()==stage+1) {
                wait(pv_done[stage][slot],phase);
                tensor_after_thread_sync();
            }
        }
        if (warpgroup::groupid()==0 && warpgroup::warpid()==0 && warp::elect_leader()) {
            const int last_consumer_stage = valid[1] ? 1 : 0;
            wait(pv_src_done[last_consumer_stage][slot], phase);
            arrive(kv_empty[slot]);
        }
    }
    if (valid[0]) finish_output(0,tt_output0);
    if (valid[1]) finish_output(1,tt_output1);
    __syncthreads();
    if(threadIdx.x==0){g.trace[blockIdx.x*32+3]=last_key; g.trace[blockIdx.x*32+6]=k_load_count; g.trace[blockIdx.x*32+7]=qk_issue_count[0]; g.trace[blockIdx.x*32+8]=qk_issue_count[1]; g.trace[blockIdx.x*32+9]=pv_issue_count[0]; g.trace[blockIdx.x*32+10]=pv_issue_count[1]; g.trace[blockIdx.x*32+11]=v_load_count;}
}
#endif
#endif

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
template <typename C>
struct globals_mxfp4_pv_dvo64_residency_shell {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = typename base::v_fp4_tile;
    using v_sc_tile = typename base::v_sc_tile;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        // Static semaphores precede extern shared memory. Leave one allocator
        // alignment quantum for the dynamic base before its first 1 KiB tile.
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    bf16 *o;
    unsigned long long *trace;
    int batch;
    int heads;
    int seqlen;
    int p_cols;
    int v_scale_rows;
    int stage;
    int grid_blocks;

    __host__ __inline__ dim3 grid() { return dim3(grid_blocks); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_nb64_dvo128_residency_shell {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = typename base::k_tile;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = typename base::v_fp4_tile;
    using v_sc_tile = typename base::v_sc_tile;

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
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *q_fp4;
    const uint8_t *q_sc;
    const uint8_t *k_fp4;
    const uint8_t *k_sc;
    const uint8_t *p_fp4;
    const uint8_t *p_sc;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    bf16 *o;
    unsigned long long *trace;
    int batch;
    int heads;
    int seqlen;
    int p_cols;
    int v_scale_rows;
    int stage;
    int grid_blocks;

    __host__ __inline__ dim3 grid() { return dim3(grid_blocks); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_nb128_dual_dvo64_attention_shell {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    // Cluster-2 production combines two 64-row rank-local K tiles. This
    // cluster-1 shell must own the complete N128 operand in one CTA.
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = typename base::v_fp4_tile;
    using v_sc_tile = typename base::v_sc_tile;

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
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *q_fp4;
    const uint8_t *q_sc;
    const uint8_t *k_fp4;
    const uint8_t *k_sc;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    uint8_t *p_debug;
    uint8_t *p_sc_debug;
    bf16 *o;
    float *lse;
    float *row_sum;
    unsigned long long *trace;
    int batch;
    int heads;
    int seqlen;
    int v_scale_rows;
    float softmax_scale_log2;
    int stage;
    int grid_blocks;

    __host__ __inline__ dim3 grid() { return dim3(grid_blocks); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_nb128_dual_dvo64_multistep_shell {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = typename base::v_fp4_tile;
    using v_sc_tile = typename base::v_sc_tile;
    using o_backing_tile = std::conditional_t<
        C::REGISTER_ACCUM, st_fl<16, 16, false>,
        st_fl<C::Mb, C::Dvo, false>>;
    using o_backing_aux_tile = std::conditional_t<
        C::REGISTER_ACCUM || C::Dvo == 128,
        st_fl<16, 16, false>, o_backing_tile>;

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
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        add((int)sizeof(o_backing_tile));
        add((int)sizeof(o_backing_aux_tile));
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY =
        compute_dynamic_shared_memory();

    const uint8_t *q_fp4;
    const uint8_t *q_sc;
    const uint8_t *k_fp4;
    const uint8_t *k_sc;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    uint8_t *p_debug;
    uint8_t *p_sc_debug;
    float *acc_scale_debug;
    bf16 *o;
    float *lse;
    float *row_sum;
    unsigned long long *trace;
    int batch;
    int heads;
    int seqlen;
    int query_tiles;
    int key_tiles;
    float softmax_scale_log2;
    int stage;
    int capture_p;
    int grid_blocks;

    __host__ __inline__ dim3 grid() { return dim3(grid_blocks); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() {
        return DYNAMIC_SHARED_MEMORY;
    }
};
#endif

template <typename C>
struct globals_mxfp4_pv_from_p_k256_debug {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using p_fp4_tile = typename base::p_fp4_k256_tile;
    using p_sc_tile = typename base::p_sc_k256_tile;
    using v_fp4_tile = typename base::v_fp4_k256_tile;
    using v_sc_tile = typename base::v_sc_k256_tile;
    using o_tile = typename base::o_tile;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile) * 2);
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile) * 2);
        add((int)sizeof(o_tile));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    bf16 *o_direct;
    int batch;
    int heads;
    int seqlen;
    int p_cols;
    int v_scale_rows;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / (C::Mb * C::CLUSTER_SIZE)) * C::CLUSTER_SIZE);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <int _KTile, int _NTile, int _ClusterSize, int _ScaleLayout>
struct config_mxfp4_tcgen_shape_debug {
    static_assert(_KTile == 64 || _KTile == 128 || _KTile == 256, "KTile must be 64, 128, or 256");
    static_assert(_NTile == 128 || _NTile == 256 || _NTile == 512, "single-tile shape debug supports NTile 128, 256, or phased 512");
    static_assert(_ClusterSize == 1 || _ClusterSize == 2, "ClusterSize must be 1 or 2");
    static_assert(_ScaleLayout == 0 || _ScaleLayout == 1, "ScaleLayout must be 0(compact) or 1(gemm-like)");

    static constexpr int Mb = 128;
    static constexpr int Nb = _KTile;
    static constexpr int Dvo = _NTile;
    static constexpr int ACC_DVO = (_NTile == 512) ? 256 : _NTile;
    static constexpr int OUT_PHASES = _NTile / ACC_DVO;
    static constexpr int CLUSTER_SIZE = _ClusterSize;
    static constexpr int SCALE_LAYOUT = _ScaleLayout;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int K_SCALE_CHUNKS = (_KTile + 127) / 128;
    static constexpr int B_SCALE_GROUPS = _NTile / 128;
    static constexpr int ACTIVE_B_SCALE_GROUPS = ACC_DVO / 128;
};

template <typename C>
struct globals_mxfp4_tcgen_shape_debug {
    using a_fp4_tile = st_fp4e2m1_2<C::Mb, C::Nb / 2>;
    using b_fp4_tile = std::conditional_t<
        C::CLUSTER_SIZE == 1,
        st_fp4e2m1_2<C::ACC_DVO, C::Nb / 2>,
        st_fp4e2m1_2<C::ACC_DVO / C::CLUSTER_SIZE, C::Nb / 2>>;
    using sc_tile = st_fp8e8m0<32, 16, false>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_fp4_tile));
        add((int)sizeof(b_fp4_tile));
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS * C::B_SCALE_GROUPS);
        return align_up_constexpr(bytes, 1024) + 4096;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *a_fp4;
    const uint8_t *a_sc;
    const uint8_t *b_fp4;
    const uint8_t *b_sc;
    bf16 *o_direct;
    int rows;
    int a_packed_cols;
    int b_packed_cols;
    int a_scale_k_tiles;
    int b_scale_k_tiles;
    int debug_stage;

    __host__ __inline__ dim3 grid() {
        const int row_tiles = rows / C::Mb;
        if constexpr (C::CLUSTER_SIZE == 1) {
            return dim3(row_tiles);
        } else {
            return dim3(((row_tiles + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE) * C::CLUSTER_SIZE);
        }
    }
	    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
	    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
	};

template <int _QuarterTile, int _Quarters, int _OptMode = 0>
struct config_mxfp4_joint_qk_pv_quarter_debug {
    static_assert(_QuarterTile == 128, "joint QK/PV quarter debug currently supports N128 quarters");
    static_assert(_Quarters == 1 || _Quarters == 2 || _Quarters == 4,
                  "joint QK/PV quarter debug supports 1, 2, or 4 quarters");
    static_assert(_OptMode == 0 || _OptMode == 2,
                  "joint QK/PV quarter debug opt mode must be 0(baseline) or 2(hoist A scale)");

    static constexpr int Mb = 128;
    static constexpr int Nb = _QuarterTile;
    static constexpr int Dvo = 128;
    static constexpr int Dqk = _QuarterTile;
    static constexpr int QUARTERS = _Quarters;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int K_SCALE_CHUNKS = (_QuarterTile + 127) / 128;
    static constexpr int OPT_MODE = _OptMode;
    static constexpr bool HOIST_A_SCALE_TMEM = _OptMode == 2;
};

template <typename C>
struct globals_mxfp4_joint_qk_pv_quarter_debug {
    using a_fp4_tile = st_fp4e2m1_2<C::Mb, C::Nb / 2>;
    using b_fp4_tile = st_fp4e2m1_2<C::Nb, C::Nb / 2>;
    using sc_tile = st_fp8e8m0<32, 16, false>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_fp4_tile));
        add((int)sizeof(b_fp4_tile));
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        return align_up_constexpr(bytes, 1024) + 4096;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *a_fp4;
    const uint8_t *a_sc;
    const uint8_t *b_fp4;
    const uint8_t *b_sc;
    bf16 *o_direct;
    int rows;
    int a_packed_cols;
    int b_packed_cols;
    int a_scale_k_tiles;
    int b_scale_k_tiles;
    int debug_stage;

    __host__ __inline__ dim3 grid() { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

struct config_mxfp4_joint_qk_pv_q4_stage0_perf {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dvo = 128;
    static constexpr int Dqk = 128;
    static constexpr int QUARTERS = 4;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int K_SCALE_CHUNKS = 1;
    static constexpr int OPT_MODE = 2;
    static constexpr bool HOIST_A_SCALE_TMEM = true;
};

struct config_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf :
    config_mxfp4_joint_qk_pv_q4_stage0_perf {
    static constexpr int B_SCALE_SLOTS = 2;
};

template <typename C>
struct globals_mxfp4_joint_qk_pv_q4_stage0_perf {
    using a_fp4_tile = st_fp4e2m1_2<C::Mb, C::Nb / 2>;
    using b_fp4_tile = st_fp4e2m1_2<C::Nb, C::Nb / 2>;
    using sc_tile = st_fp8e8m0<32, 16, false>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_fp4_tile));
        add((int)sizeof(b_fp4_tile));
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        return align_up_constexpr(bytes, 1024) + 4096;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *a_fp4;
    const uint8_t *a_sc;
    const uint8_t *b_fp4;
    const uint8_t *b_sc;
    bf16 *o_direct;
    int rows;
    int a_packed_cols;
    int b_packed_cols;
    int a_scale_k_tiles;
    int b_scale_k_tiles;

    __host__ __inline__ dim3 grid() { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf {
    using a_fp4_tile = typename globals_mxfp4_joint_qk_pv_q4_stage0_perf<
        config_mxfp4_joint_qk_pv_q4_stage0_perf>::a_fp4_tile;
    using b_fp4_tile = typename globals_mxfp4_joint_qk_pv_q4_stage0_perf<
        config_mxfp4_joint_qk_pv_q4_stage0_perf>::b_fp4_tile;
    using sc_tile = typename globals_mxfp4_joint_qk_pv_q4_stage0_perf<
        config_mxfp4_joint_qk_pv_q4_stage0_perf>::sc_tile;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_fp4_tile));
        add((int)sizeof(b_fp4_tile));
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS * C::B_SCALE_SLOTS);
        return align_up_constexpr(bytes, 1024) + 4096;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *a_fp4;
    const uint8_t *a_sc;
    const uint8_t *b_fp4;
    const uint8_t *b_sc;
    bf16 *o_direct;
    int rows;
    int a_packed_cols;
    int b_packed_cols;
    int a_scale_k_tiles;
    int b_scale_k_tiles;

    __host__ __inline__ dim3 grid() { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

struct config_mxfp4_joint_qk_pv_half_stage0_perf {
    static constexpr int Mb = 128;
    static constexpr int KTile = 128;
    static constexpr int Nb = KTile;
    static constexpr int NTile = 256;
    static constexpr int Dvo = 128;
    static constexpr int HALVES = 2;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int K_SCALE_CHUNKS = KTile / 128;
    static constexpr int B_SCALE_GROUPS = NTile / 128;
    static constexpr bool HOIST_A_SCALE_TMEM = true;
};

template <typename C>
struct globals_mxfp4_joint_qk_pv_half_stage0_perf {
    using a_fp4_tile = st_fp4e2m1_2<C::Mb, C::KTile / 2>;
    using b_fp4_tile = st_fp4e2m1_2<C::NTile, C::KTile / 2>;
    using b_pv_fp4_tile = st_fp4e2m1_2<128, C::KTile / 2>;
    using sc_tile = st_fp8e8m0<32, 16, false>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_fp4_tile));
        add((int)sizeof(b_fp4_tile));
        add((int)sizeof(b_pv_fp4_tile) * C::B_SCALE_GROUPS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS * C::B_SCALE_GROUPS);
        return align_up_constexpr(bytes, 1024) + 4096;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *a_fp4;
    const uint8_t *a_sc;
    const uint8_t *b_fp4;
    const uint8_t *b_sc;
    bf16 *o_direct;
    int rows;
    int a_packed_cols;
    int b_packed_cols;
    int a_scale_k_tiles;
    int b_scale_k_tiles;

    __host__ __inline__ dim3 grid() { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr int Mb = 128;
    static constexpr int KTile = 128;
    static constexpr int Nb = 128;
    static constexpr int NTotal = 512;
    static constexpr int Dvo = 128;
    static constexpr int QUARTERS = NTotal / Nb;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int K_SCALE_CHUNKS = KTile / 128;
    static constexpr bool PRESTAGE_V_UNDER_QK = false;
    static constexpr bool PIPELINE_QK_PV = false;
    static constexpr int SCORE_TMEM_SLOTS = 1;
    static constexpr int P_SOURCE_SLOTS = 1;
    static constexpr int V_SOURCE_SLOTS = 1;
    static constexpr int P_SCALE_SMEM_SLOTS = 1;
    static constexpr int V_SCALE_SMEM_SLOTS = 1;
    static constexpr int PV_SCALE_TMEM_SLOTS = 1;
    static constexpr bool PV_SPLIT_K64 = false;
    static constexpr bool SPLIT_OUTPUT_ACCUM = false;
    static constexpr bool TRIM_PV_ISSUER_SYNCS = false;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_pingpong_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PIPELINE_QK_PV = true;
    static constexpr int SCORE_TMEM_SLOTS = 2;
    static constexpr int P_SOURCE_SLOTS = 2;
    static constexpr int V_SOURCE_SLOTS = 2;
    static constexpr int P_SCALE_SMEM_SLOTS = 2;
    static constexpr int V_SCALE_SMEM_SLOTS = 2;
    static constexpr int PV_SCALE_TMEM_SLOTS = 2;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_score_tmem_only_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PIPELINE_QK_PV = true;
    static constexpr int SCORE_TMEM_SLOTS = 2;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_scale_tmem_only_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PIPELINE_QK_PV = true;
    static constexpr int PV_SCALE_TMEM_SLOTS = 2;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_vonly_src_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PIPELINE_QK_PV = true;
    static constexpr int V_SOURCE_SLOTS = 2;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_ponly_src_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PIPELINE_QK_PV = true;
    static constexpr int P_SOURCE_SLOTS = 2;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_score_scale_tmem_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PIPELINE_QK_PV = true;
    static constexpr int SCORE_TMEM_SLOTS = 2;
    static constexpr int PV_SCALE_TMEM_SLOTS = 2;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_score_p_src_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PIPELINE_QK_PV = true;
    static constexpr int SCORE_TMEM_SLOTS = 2;
    static constexpr int P_SOURCE_SLOTS = 2;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_pv_split_k64_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PV_SPLIT_K64 = true;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_split_output_accum_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool PIPELINE_QK_PV = true;
    static constexpr int PV_SCALE_TMEM_SLOTS = 2;
    static constexpr bool SPLIT_OUTPUT_ACCUM = true;
};

struct config_mxfp4_joint_qk_pv_true_q4_stage0_barrier_trim_perf :
    config_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    static constexpr bool PRESTAGE_V_UNDER_QK = true;
    static constexpr bool TRIM_PV_ISSUER_SYNCS = true;
};

template <typename C>
struct globals_mxfp4_joint_qk_pv_true_q4_stage0_ref {
    using a_fp4_tile = st_fp4e2m1_2<C::Mb, C::KTile / 2>;
    using k_fp4_tile = st_fp4e2m1_2<C::Nb, C::KTile / 2>;
    using p_fp4_tile = st_fp4e2m1_2<C::Mb, C::Nb / 2>;
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using sc_tile = st_fp8e8m0<32, 16, false>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_fp4_tile));
        add((int)sizeof(k_fp4_tile));
        add((int)sizeof(p_fp4_tile) * C::P_SOURCE_SLOTS);
        add((int)sizeof(v_fp4_tile) * C::V_SOURCE_SLOTS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::P_SCALE_SMEM_SLOTS);
        add((int)sizeof(sc_tile) * C::V_SCALE_SMEM_SLOTS);
        return align_up_constexpr(bytes, 1024) + 4096;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *a_fp4;
    const uint8_t *a_sc;
    const uint8_t *b_fp4;
    const uint8_t *b_sc;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    bf16 *o_direct;
    int rows;
    int a_packed_cols;
    int b_packed_cols;
    int a_scale_k_tiles;
    int b_scale_k_tiles;
    int seqlen;
    int heads;
    int v_scale_rows;

    __host__ __inline__ dim3 grid() { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

struct config_mxfp4_joint_qk_pv_true_half_stage0_perf {
    static constexpr int Mb = 128;
    static constexpr int KTile = 128;
    static constexpr int Nb = 128;
    static constexpr int NTotal = 512;
    static constexpr int NTile = 256;
    static constexpr int Dvo = 128;
    static constexpr int HALVES = NTotal / NTile;
    static constexpr int HALF_SCALE_GROUPS = NTile / Nb;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int K_SCALE_CHUNKS = KTile / 128;
    static constexpr bool PIPELINE_NEXT_HALF_UNDER_PV = false;
    static constexpr bool TRIM_PV_ISSUER_SYNCS = false;
};

struct config_mxfp4_joint_qk_pv_true_half_stage0_srcwait_pipeline_perf :
    config_mxfp4_joint_qk_pv_true_half_stage0_perf {
    static constexpr bool PIPELINE_NEXT_HALF_UNDER_PV = true;
};

struct config_mxfp4_joint_qk_pv_true_half_stage0_barrier_trim_perf :
    config_mxfp4_joint_qk_pv_true_half_stage0_perf {
    static constexpr bool TRIM_PV_ISSUER_SYNCS = true;
};

template <typename C>
struct globals_mxfp4_joint_qk_pv_true_half_stage0_perf {
    using a_fp4_tile = st_fp4e2m1_2<C::Mb, C::KTile / 2>;
    using k_half_fp4_tile = st_fp4e2m1_2<C::NTile, C::KTile / 2>;
    using p_half_fp4_tile = st_fp4e2m1_2<C::Mb, C::NTile / 2>;
    using v_half_fp4_tile = st_fp4e2m1_2<C::Dvo, C::NTile / 2>;
    using sc_tile = st_fp8e8m0<32, 16, false>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_fp4_tile));
        add((int)sizeof(k_half_fp4_tile));
        add((int)sizeof(p_half_fp4_tile));
        add((int)sizeof(v_half_fp4_tile));
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::HALF_SCALE_GROUPS);
        add((int)sizeof(sc_tile) * C::HALF_SCALE_GROUPS);
        add((int)sizeof(sc_tile) * C::HALF_SCALE_GROUPS);
        return align_up_constexpr(bytes, 1024) + 4096;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *a_fp4;
    const uint8_t *a_sc;
    const uint8_t *b_fp4;
    const uint8_t *b_sc;
    const uint8_t *v_fp4;
    const uint8_t *v_sc;
    bf16 *o_direct;
    int rows;
    int a_packed_cols;
    int b_packed_cols;
    int a_scale_k_tiles;
    int b_scale_k_tiles;
    int seqlen;
    int heads;
    int v_scale_rows;

    __host__ __inline__ dim3 grid() { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_joint_qk_pv_quarter_handoff_debug {
    using a_fp4_tile = typename globals_mxfp4_joint_qk_pv_quarter_debug<C>::a_fp4_tile;
    using b_fp4_tile = typename globals_mxfp4_joint_qk_pv_quarter_debug<C>::b_fp4_tile;
    using sc_tile = typename globals_mxfp4_joint_qk_pv_quarter_debug<C>::sc_tile;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_fp4_tile));
        add((int)sizeof(b_fp4_tile));
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        add((int)sizeof(sc_tile) * C::K_SCALE_CHUNKS);
        return align_up_constexpr(bytes, 1024) + 4096;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *a_fp4;
    const uint8_t *a_sc;
    const uint8_t *b_fp4;
    const uint8_t *b_sc;
    bf16 *o_direct;
    int rows;
    int a_packed_cols;
    int b_packed_cols;
    int a_scale_k_tiles;
    int b_scale_k_tiles;
    int control_mode;

    __host__ __inline__ dim3 grid() { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_v_tma_tile_check_debug {
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using v_sc_tile = st_fp8e8m0<32, 16, false>;
    using v_fp4_gl = gl<fp4e2m1_2, -1, -1, -1, -1, tma::descriptor<v_fp4_tile, dim::ROW>>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        add((int)sizeof(v_sc_tile));
        add((int)sizeof(v_fp4_tile) * 2);
        add((int)sizeof(v_sc_tile) * 2);
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    v_fp4_gl v_fp4_tma;
    const uint8_t *v_fp4;
    int32_t *debug;
    int batch;
    int heads;
    int seqlen;
    int b;
    int h;
    int k_iter;
    int mode;
    const uint8_t *v_sc_prepared;
    int v_scale_rows;
    int check_scale;

    __host__ __inline__ dim3 grid() { return dim3(1); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_k_tma_multicast_debug {
    using base = globals_fp4pv<C, true>;
    using k_tile = typename base::k_tile;
    using k_sc_tile = typename base::k_sc_tile;
    using k_gl = typename base::k_gl;
    using k_sc_gl = typename base::k_sc_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(k_tile));
        add((int)sizeof(k_tile) * C::LOAD_STAGES);
        add((int)sizeof(k_sc_tile) * C::K_SCALE_SMEM_SLOTS);
        add((int)sizeof(k_sc_tile));
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    k_gl k;
    k_sc_gl k_sc;
    const uint8_t *k_fp4;
    const uint8_t *k_sc_raw;
    int32_t *debug;
    int batch;
    int heads;
    int seqlen;
    int b;
    int h;
    int k_iter;
    int mode;
    int check_scale;

    __host__ __inline__ dim3 grid() { return dim3(1); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_kv_tma_multicast_debug {
    using base = globals_fp4pv<C, true>;
    using k_tile = typename base::k_tile;
    using k_sc_tile = typename base::k_sc_tile;
    using k_gl = typename base::k_gl;
    using k_sc_gl = typename base::k_sc_gl;
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using v_sc_tile = st_fp8e8m0<32, 16, false>;
    using v_fp4_gl = gl<fp4e2m1_2, -1, -1, -1, -1, tma::descriptor<v_fp4_tile, dim::ROW>>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(k_tile));
        add((int)sizeof(k_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    k_gl k;
    k_sc_gl k_sc;
    v_fp4_gl v_fp4_tma;
    const uint8_t *k_fp4;
    const uint8_t *k_sc_raw;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    int32_t *debug;
    int batch;
    int heads;
    int seqlen;
    int b;
    int h;
    int k_iter;
    int mode;
    int check_scale;
    int v_scale_rows;

    __host__ __inline__ dim3 grid() { return dim3(C::CLUSTER_SIZE); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_mxfp4_qk_tma_multicast_debug {
    using base = globals_fp4pv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = typename base::k_tile;
    using k_sc_tile = typename base::k_sc_tile;
    using q_gl = typename base::q_gl;
    using q_sc_gl = typename base::q_sc_gl;
    using k_gl = typename base::k_gl;
    using k_sc_gl = typename base::k_sc_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_tile));
        add((int)sizeof(q_sc_tile));
        add((int)sizeof(k_tile) * C::LOAD_STAGES);
        add((int)sizeof(k_sc_tile) * C::K_SCALE_SMEM_SLOTS);
        return bytes + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    q_gl q;
    q_sc_gl q_sc;
    k_gl k;
    k_sc_gl k_sc;
    int32_t *debug;
    int batch;
    int heads;
    int seqlen;
    int b;
    int h;
    int q_iter;
    int k_iter;
    int mode;

    __host__ __inline__ dim3 grid() { return dim3(C::CLUSTER_SIZE); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_two_tile_pv_tmem_debug {
    using p_fp4_tile = typename globals_fp4pv<C>::p_fp4_tile;
    using p_sc_tile = typename globals_fp4pv<C>::p_sc_tile;
    using v_fp4_tile = typename globals_fp4pv<C>::v_fp4_tile;
    using v_sc_tile = typename globals_fp4pv<C>::v_sc_tile;
    using o_tile = typename globals_fp4pv<C>::o_tile;
    using o_gl = typename globals_fp4pv<C>::o_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        add((int)sizeof(o_tile));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc_prepared;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    o_gl o_snapshot;
    int tiles_to_run;
    int batch;
    int heads;
    int seqlen;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / (C::Mb * C::CLUSTER_SIZE)) * C::CLUSTER_SIZE);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_two_tile_pv_mainpath_debug {
    using p_fp4_tile = typename globals_fp4pv<C>::p_fp4_tile;
    using p_sc_tile = typename globals_fp4pv<C>::p_sc_tile;
    using p_sc_guard_slot = typename globals_fp4pv<C>::p_sc_guard_slot;
    using v_fp4_tile = typename globals_fp4pv<C>::v_fp4_tile;
    using v_sc_tile = typename globals_fp4pv<C>::v_sc_tile;
    using o_tile = typename globals_fp4pv<C>::o_tile;
    using o_gl = typename globals_fp4pv<C>::o_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(v_fp4_tile) * C::LOAD_STAGES);
        add((int)sizeof(v_sc_tile) * C::LOAD_STAGES);
        add((int)sizeof(o_tile));
        add((int)sizeof(p_fp4_tile) * 2);
        add((int)sizeof(p_sc_tile) * 2);
        add((int)sizeof(p_sc_guard_slot));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc_prepared;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    o_gl o;
    int tile_begin;
    int tiles_to_run;
    int second_tile_mm2;
    int pv_wait_count;
    int early_tile_arrived;
    int batch;
    int heads;
    int seqlen;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / (C::Mb * C::CLUSTER_SIZE)) * C::CLUSTER_SIZE);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_two_tile_correction_replay_debug {
    const float *pre_correction;
    const float *correction;
    const float *row_sum;
    bf16 *out;
    int batch;
    int heads;
    int seqlen;
};

template <typename C, typename PTile, typename G>
__device__ inline void load_p_fp4_tile_from_global_allthreads(
    PTile &p_smem,
    const G &g,
    int b,
    int h,
    int m_tile,
    int k_iter
) {
    const int packed_seqlen = g.seqlen / 2;
    const int row_base = m_tile * C::Mb;
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&p_smem.data[0]);
    constexpr int BYTES_PER_STORE = 4;
    static_assert(PTile::cols % BYTES_PER_STORE == 0, "Expected fp4 P tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = PTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = PTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint32_t value = *reinterpret_cast<const uint32_t *>(
            &g.p_fp4[((((b * g.heads + h) * g.seqlen) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.b32 [%0], %1;"
                     :
                     : "r"(PTile::idx(smem_base, {row, col})), "r"(value));
    }
}

template <typename C, typename PScTile, typename G>
__device__ inline void load_p_sc_prepared_from_global_allthreads(
    PScTile &p_sc_smem,
    const G &g,
    int b,
    int h,
    int m_tile,
    int k_iter
) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&p_sc_smem.data[0]);
    const int scale_k_base = k_iter * 2;
    static_assert(sizeof(PScTile) % sizeof(uint32_t) == 0, "Expected aligned P scale tile size");
    constexpr int WORDS_PER_SLAB = 512 / sizeof(uint32_t);
    constexpr int TOTAL_WORDS = sizeof(PScTile) / sizeof(uint32_t);
    for (int linear = threadIdx.x; linear < TOTAL_WORDS; linear += blockDim.x) {
        const int slab = linear / WORDS_PER_SLAB;
        const int word = linear % WORDS_PER_SLAB;
        dst[linear] = *reinterpret_cast<const uint32_t *>(
            &g.p_sc_prepared[((((b * g.heads + h) * (g.seqlen / 128) + m_tile) * (g.seqlen / 64)) + scale_k_base + slab) * 512 + word * sizeof(uint32_t)]);
    }
}

template <typename C, typename VTile, typename G>
__device__ inline void load_v_fp4_tile_from_global_allthreads(
    VTile &v_smem,
    const G &g,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int dq_half_override = -1
) {
    const int packed_seqlen = g.seqlen / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert(VTile::cols % BYTES_PER_STORE == 0, "Expected fp4 V tile cols divisible by vector width");
    static_assert(VTile::swizzle_bytes >= BYTES_PER_STORE, "Expected V tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = VTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.v_fp4[((((b * g.heads + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(VTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename VScTile, typename G>
__device__ inline void load_v_sc_prepared_from_global_allthreads(
    VScTile &v_sc_smem,
    const G &g,
    int b,
    int h,
    int k_iter
) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&v_sc_smem.data[0]);
    const int scale_k_base = k_iter * 2;
    static_assert(sizeof(VScTile) % sizeof(uint32_t) == 0, "Expected aligned V scale tile size");
    constexpr int WORDS_PER_SLAB = 512 / sizeof(uint32_t);
    constexpr int TOTAL_WORDS = sizeof(VScTile) / sizeof(uint32_t);
    for (int linear = threadIdx.x; linear < TOTAL_WORDS; linear += blockDim.x) {
        const int slab = linear / WORDS_PER_SLAB;
        const int word = linear % WORDS_PER_SLAB;
        dst[linear] = *reinterpret_cast<const uint32_t *>(
            &g.v_sc_prepared[((((b * g.heads + h) * (C::Dvo / 128) + 0) * (g.seqlen / 64)) + scale_k_base + slab) * 512 + word * sizeof(uint32_t)]);
    }
}

template <typename C, typename PTile, typename G>
__device__ inline void load_mxfp4_p_fp4_tile_from_global_allthreads(
    PTile &p_smem,
    const G &g,
    int b,
    int h,
    int m_tile,
    int p_iter
) {
    const int packed_p_cols = g.p_cols / 2;
    const int row_base = m_tile * C::Mb;
    const int col_base = p_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&p_smem.data[0]);
    constexpr int BYTES_PER_STORE = 4;
    constexpr int LOGICAL_PACKED_COLS = C::Nb / 2;
    static_assert(PTile::cols >= LOGICAL_PACKED_COLS, "Expected compact MXFP4 P tile to hold one packed K tile");
    static_assert(LOGICAL_PACKED_COLS % BYTES_PER_STORE == 0, "Expected compact MXFP4 P tile cols divisible by vector width");
    fp4pv_zero_shared_tile_block(p_smem);
    constexpr int STORES_PER_ROW = LOGICAL_PACKED_COLS / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = PTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint32_t value = *reinterpret_cast<const uint32_t *>(
            &g.p_fp4[((((b * g.heads + h) * g.seqlen) + row_base + row) * packed_p_cols) + col_base + col]);
        asm volatile("st.shared.b32 [%0], %1;"
                     :
                     : "r"(PTile::idx(smem_base, {row, col})), "r"(value));
    }
}

template <typename C, typename PTile, typename G>
__device__ inline void load_mxfp4_p_fp4_k256_tile_from_global_allthreads(
    PTile &p_smem,
    const G &g,
    int b,
    int h,
    int m_tile,
    int k_iter
) {
    const int packed_p_cols = g.p_cols / 2;
    const int row_base = m_tile * C::Mb;
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&p_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int LOGICAL_PACKED_COLS = C::Nb;
    static_assert(PTile::cols >= LOGICAL_PACKED_COLS, "Expected compact MXFP4 K256 P tile to hold two packed K128 tiles");
    static_assert(LOGICAL_PACKED_COLS % BYTES_PER_STORE == 0, "Expected compact MXFP4 K256 P tile cols divisible by vector width");
    static_assert(PTile::swizzle_bytes >= BYTES_PER_STORE, "Expected compact MXFP4 K256 P tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = LOGICAL_PACKED_COLS / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = PTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.p_fp4[((((b * g.heads + h) * g.seqlen) + row_base + row) * packed_p_cols) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(PTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename PScTile, typename G>
__device__ inline void load_mxfp4_p_sc_from_global_allthreads(
    PScTile &p_sc_smem,
    const G &g,
    int b,
    int h,
    int m_tile,
    int p_iter
) {
    constexpr int BYTES_PER_STORE = 16;
    static_assert(sizeof(PScTile) % BYTES_PER_STORE == 0, "Expected aligned MXFP4 P scale tile size");
    constexpr int TOTAL_STORES = sizeof(PScTile) / BYTES_PER_STORE;
    const int scale_cols = g.p_cols / 128;
    const int base_offset = (((((b * g.heads + h) * (g.seqlen / 128) + m_tile) * scale_cols) + p_iter) * 32 * 16);
    uint32_t smem_base = __cvta_generic_to_shared(&p_sc_smem.data[0]);
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int byte_offset = linear * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(&g.p_sc[base_offset + byte_offset]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(smem_base + byte_offset),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename VTile, typename G>
__device__ inline void load_mxfp4_v_fp4_k256_tile_from_global_allthreads(
    VTile &v_smem,
    const G &g,
    int b,
    int h,
    int k_iter,
    int cta_rank
) {
    const int packed_seqlen = g.seqlen / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int LOGICAL_PACKED_COLS = C::Nb;
    static_assert(LOGICAL_PACKED_COLS % BYTES_PER_STORE == 0, "Expected compact MXFP4 K256 V load width divisible by vector width");
    static_assert(VTile::cols >= LOGICAL_PACKED_COLS, "Expected compact MXFP4 K256 V tile to hold two packed K128 tiles");
    static_assert(VTile::swizzle_bytes >= BYTES_PER_STORE, "Expected compact MXFP4 K256 V tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = LOGICAL_PACKED_COLS / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.v_fp4[((((b * g.heads + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(VTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename VTile, typename G>
__device__ inline void load_mxfp4_v_fp4_tile_from_global_allthreads(
    VTile &v_smem,
    const G &g,
    int b,
    int h,
    int k_iter,
    int cta_rank
) {
    const int packed_seqlen = g.seqlen / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "Expected compact MXFP4 V load width divisible by vector width");
    static_assert(VTile::cols >= C::Nb / 2, "Expected compact MXFP4 V tile to hold one packed 128-column slice");
    static_assert(VTile::swizzle_bytes >= BYTES_PER_STORE, "Expected compact MXFP4 V tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.v_fp4[((((b * g.heads + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(VTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename VScTile, typename G>
__device__ inline void load_mxfp4_v_sc_from_global_allthreads(
    VScTile &v_sc_smem,
    const G &g,
    int b,
    int h,
    int k_iter
) {
    constexpr int BYTES_PER_STORE = 16;
    static_assert(sizeof(VScTile) % BYTES_PER_STORE == 0, "Expected aligned MXFP4 V scale tile size");
    constexpr int TOTAL_STORES = sizeof(VScTile) / BYTES_PER_STORE;
    const int base_offset = (((((b * g.heads + h) * g.v_scale_rows + 0) * (g.seqlen / 128)) + k_iter) * 32 * 16);
    uint32_t smem_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int byte_offset = linear * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(&g.v_sc[base_offset + byte_offset]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(smem_base + byte_offset),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename Tile, typename G>
__device__ inline void load_live_dp_dout_row_fp4_tile_warp(
    Tile &dst,
    const G &g,
    int b,
    int h,
    int m_tile
) {
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int packed_dvo = C::Dvo / 2;
    const int row_base = m_tile * C::Mb;
    uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert(Tile::cols % BYTES_PER_STORE == 0, "Expected dO row FP4 tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.backward_dout_row_fp4[
                ((((static_cast<int64_t>(b) * heads + h) * seqlen) + row_base + row) * packed_dvo) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(Tile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename Tile, typename G>
__device__ inline void load_live_dp_v_row_fp4_tile_warp(
    Tile &dst,
    const G &g,
    int b,
    int h,
    int k_iter,
    int cta_rank
) {
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int packed_dvo = C::Dvo / 2;
    const int row_base = k_iter * C::Nb + cta_rank * (C::Nb / C::CLUSTER_SIZE);
    uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert(Tile::cols % BYTES_PER_STORE == 0, "Expected V row FP4 tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.backward_v_row_fp4[
                ((((static_cast<int64_t>(b) * heads + h) * seqlen) + row_base + row) * packed_dvo) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(Tile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename ScTile, typename G>
__device__ inline void load_live_dp_dout_row_sc_tile_warp(
    ScTile &dst,
    const G &g,
    int b,
    int h,
    int m_tile
) {
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int row_tiles = seqlen / C::Mb;
    constexpr int TOTAL_BYTES = sizeof(ScTile);
    static_assert(TOTAL_BYTES == 32 * 16, "Expected dO row scale tile to be 32x16 bytes");
    uint32_t smem_base = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(dst.data[0]));
    const int64_t base =
        (((((static_cast<int64_t>(b) * heads + h) * row_tiles + m_tile) * (C::Dvo / 128)) + 0) * 32 * 16);
    // Byte staging is intentional here: wider stores have produced stale scale
    // bytes in clustered live-dP scale loads on multi-head launches.
    for (int byte_offset = warp::laneid(); byte_offset < TOTAL_BYTES; byte_offset += WARP_THREADS) {
        const uint32_t value = static_cast<uint32_t>(g.backward_dout_row_sc[base + byte_offset]);
        asm volatile("st.shared.b8 [%0], %1;" :: "r"(smem_base + byte_offset), "r"(value));
        if (g.live_dp_debug_mode == 17 && g.backward_dp_bf16 != nullptr) {
            uint32_t check;
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(check) : "r"(smem_base + byte_offset));
            const int out_row = 15 + byte_offset / seqlen;
            const int out_col = byte_offset - (out_row - 15) * seqlen;
            const int64_t debug_row_base =
                (((static_cast<int64_t>(b) * heads + h) * seqlen + m_tile * C::Mb) * seqlen);
            g.backward_dp_bf16[debug_row_base + out_row * seqlen + out_col] =
                __float2bfloat16(static_cast<float>(check));
        }
    }
}

template <typename C, typename ScTile, typename G>
__device__ inline void load_live_dp_v_row_sc_tile_warp(
    ScTile &dst,
    const G &g,
    int b,
    int h,
    int k_iter
) {
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int row_tiles = seqlen / C::Mb;
    constexpr int TOTAL_BYTES = sizeof(ScTile);
    static_assert(TOTAL_BYTES == 32 * 16, "Expected V row scale tile to be 32x16 bytes");
    uint32_t smem_base = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(dst.data[0]));
    const int64_t base =
        (((((static_cast<int64_t>(b) * heads + h) * row_tiles + k_iter) * (C::Dvo / 128)) + 0) * 32 * 16);
    // See the dO scale loader above; this path must preserve byte-granular
    // shared staging for the live-dP scale TMEM copy.
    for (int byte_offset = warp::laneid(); byte_offset < TOTAL_BYTES; byte_offset += WARP_THREADS) {
        const uint32_t value = static_cast<uint32_t>(g.backward_v_row_sc[base + byte_offset]);
        asm volatile("st.shared.b8 [%0], %1;" :: "r"(smem_base + byte_offset), "r"(value));
        if (g.live_dp_debug_mode == 17 && g.backward_dp_bf16 != nullptr) {
            uint32_t check;
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(check) : "r"(smem_base + byte_offset));
            const int out_row = 11 + byte_offset / seqlen;
            const int out_col = byte_offset - (out_row - 11) * seqlen;
            const int64_t debug_row_base = (((static_cast<int64_t>(b) * heads + h) * seqlen) * seqlen);
            g.backward_dp_bf16[debug_row_base + out_row * seqlen + out_col] =
                __float2bfloat16(static_cast<float>(check));
        }
    }
}

template <typename C, typename G>
__device__ inline void store_live_dp_source_debug_warp(
    const G &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int cta_rank
) {
    if (g.backward_dp_bf16 == nullptr) {
        return;
    }
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int packed_dvo = C::Dvo / 2;
    const int row_tiles = seqlen / C::Mb;
    const int dout_row_base = m_tile * C::Mb;
    const int v_row_base = k_iter * C::Nb + cta_rank * (C::Nb / C::CLUSTER_SIZE);
    if (dout_row_base + 7 >= seqlen) {
        return;
    }
    const int64_t debug_row_base =
        (((static_cast<int64_t>(b) * heads + h) * seqlen) + dout_row_base) * seqlen;
    const int64_t dout_payload_base =
        ((((static_cast<int64_t>(b) * heads + h) * seqlen) + dout_row_base) * packed_dvo);
    const int64_t v_payload_base =
        ((((static_cast<int64_t>(b) * heads + h) * seqlen) + v_row_base) * packed_dvo);
    const int64_t dout_scale_base =
        (((((static_cast<int64_t>(b) * heads + h) * row_tiles + m_tile) * (C::Dvo / 128)) + 0) * 32 * 16);
    const int64_t v_scale_base =
        (((((static_cast<int64_t>(b) * heads + h) * row_tiles + k_iter) * (C::Dvo / 128)) + 0) * 32 * 16);
    for (int col = warp::laneid(); col < 64; col += WARP_THREADS) {
        g.backward_dp_bf16[debug_row_base + col] =
            __float2bfloat16(static_cast<float>(g.backward_dout_row_fp4[dout_payload_base + col]));
        g.backward_dp_bf16[debug_row_base + seqlen + col] =
            __float2bfloat16(static_cast<float>(g.backward_v_row_fp4[v_payload_base + col]));
        g.backward_dp_bf16[debug_row_base + 2 * seqlen + col] =
            __float2bfloat16(static_cast<float>(g.backward_dout_row_sc[dout_scale_base + col]));
        g.backward_dp_bf16[debug_row_base + 3 * seqlen + col] =
            __float2bfloat16(static_cast<float>(g.backward_v_row_sc[v_scale_base + col]));
    }
}

template <typename C, typename DTile, typename VTile, typename ScTile, typename G>
__device__ inline void store_live_dp_shared_debug_warp(
    const G &g,
    const DTile &dout_smem,
    const ScTile &dout_sc_smem,
    const VTile &v_smem,
    const ScTile &v_sc_smem,
    int b,
    int h,
    int m_tile
) {
    if (g.backward_dp_bf16 == nullptr) {
        return;
    }
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int dout_row_base = m_tile * C::Mb;
    if (dout_row_base + 3 >= seqlen) {
        return;
    }
    const int64_t debug_row_base =
        (((static_cast<int64_t>(b) * heads + h) * seqlen) + dout_row_base) * seqlen;
    const uint32_t dout_base = __cvta_generic_to_shared(&dout_smem.data[0]);
    const uint32_t v_base = __cvta_generic_to_shared(&v_smem.data[0]);
    const uint32_t dout_sc_base = __cvta_generic_to_shared(&dout_sc_smem.data[0]);
    const uint32_t v_sc_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);
    for (int col = warp::laneid(); col < 64; col += WARP_THREADS) {
        uint32_t dout_byte;
        uint32_t v_byte;
        uint32_t dout_sc_byte;
        uint32_t v_sc_byte;
        asm volatile("ld.shared.u8 %0, [%1];"
                     : "=r"(dout_byte)
                     : "r"(DTile::idx(dout_base, {0, col})));
        asm volatile("ld.shared.u8 %0, [%1];"
                     : "=r"(v_byte)
                     : "r"(VTile::idx(v_base, {0, col})));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(dout_sc_byte) : "r"(dout_sc_base + col));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v_sc_byte) : "r"(v_sc_base + col));
        g.backward_dp_bf16[debug_row_base + col] =
            __float2bfloat16(static_cast<float>(dout_byte));
        g.backward_dp_bf16[debug_row_base + seqlen + col] =
            __float2bfloat16(static_cast<float>(v_byte));
        g.backward_dp_bf16[debug_row_base + 2 * seqlen + col] =
            __float2bfloat16(static_cast<float>(dout_sc_byte));
        g.backward_dp_bf16[debug_row_base + 3 * seqlen + col] =
            __float2bfloat16(static_cast<float>(v_sc_byte));
        uint32_t dout_phys_byte;
        uint32_t v_phys_byte;
        uint32_t dout_sc_phys_byte;
        uint32_t v_sc_phys_byte;
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(dout_phys_byte) : "r"(dout_base + col));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v_phys_byte) : "r"(v_base + col));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(dout_sc_phys_byte) : "r"(dout_sc_base + col));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v_sc_phys_byte) : "r"(v_sc_base + col));
        g.backward_dp_bf16[debug_row_base + 4 * seqlen + col] =
            __float2bfloat16(static_cast<float>(dout_phys_byte));
        g.backward_dp_bf16[debug_row_base + 5 * seqlen + col] =
            __float2bfloat16(static_cast<float>(v_phys_byte));
        g.backward_dp_bf16[debug_row_base + 6 * seqlen + col] =
            __float2bfloat16(static_cast<float>(dout_sc_phys_byte));
        g.backward_dp_bf16[debug_row_base + 7 * seqlen + col] =
            __float2bfloat16(static_cast<float>(v_sc_phys_byte));
    }
}

template <typename C, bool SECOND_TILE_MM2>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_two_tile_pv_tmem_debug(const __grid_constant__ globals_two_tile_pv_tmem_debug<C> g) {
    using G = globals_two_tile_pv_tmem_debug<C>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_p_sc = full_tt_fp8e4m3<C::P_SC_TMEM_WIDTH>;
    using d_tt_v_sc = full_tt_fp8e4m3<C::V_SC_TMEM_WIDTH>;

    const int cta_rank = cluster_ctarank();
    const int row_tiles = g.seqlen / C::Mb;
    const int row_clusters = row_tiles / C::CLUSTER_SIZE;
    const int cluster_linear = blockIdx.x / C::CLUSTER_SIZE;
    const int m_cluster = cluster_linear % row_clusters;
    const int hb = cluster_linear / row_clusters;
    const int h = hb % g.heads;
    const int b = hb / g.heads;
    const int m_tile = m_cluster * C::CLUSTER_SIZE + cta_rank;
    if (b >= g.batch || m_tile >= row_tiles || g.seqlen < 2 * C::Nb) {
        return;
    }

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(0);
    constexpr int P_SC_BASE = C::Dvo;
    constexpr int V_SC_BASE = P_SC_BASE + C::P_SC_TMEM_WIDTH;
    static_assert(V_SC_BASE + C::V_SC_TMEM_WIDTH <= MAX_TENSOR_COLS, "two-tile PV debug TMEM layout exceeds budget");
    d_tt_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE);
    d_tt_v_sc v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    auto (&o_smem)[1] = al.allocate<typename G::o_tile, 1>();
    typename G::p_fp4_tile &p_fp4_smem = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();
    (void)al.allocate<typename G::o_tile>();

    __shared__ semaphore src_finished[1];
    __shared__ semaphore pv_tmem_ready[1];
    __shared__ semaphore tile_arrived[1];

    if (threadIdx.x == 0) {
        g.o_snapshot.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        init_semaphore(src_finished[0], 0, 1);
        init_semaphore(pv_tmem_ready[0], 0, 1);
        init_semaphore(tile_arrived[0], 0, 1);
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0) && (cta_rank == 0);
    const bool issuer = issuer_warp && warp::elect_leader();
    int src_phase = 0;
    for (int idx = 0; idx < g.tiles_to_run; ++idx) {
        if (threadIdx.x == 0 && idx > 0) {
            wait(src_finished[0], src_phase);
            src_phase ^= 1;
        }
        __syncthreads();

        load_p_fp4_tile_from_global_allthreads<C>(p_fp4_smem, g, b, h, m_tile, idx);
        load_p_sc_prepared_from_global_allthreads<C>(p_sc_smem, g, b, h, m_tile, idx);
        load_v_fp4_tile_from_global_allthreads<C>(v_fp4_smem, g, b, h, idx, cta_rank);
        load_v_sc_prepared_from_global_allthreads<C>(v_sc_smem, g, b, h, idx);
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        fp4pv_cluster_sync<C>();

        if (issuer) {
            #pragma unroll
            for (int ii = 0; ii < C::PV_SCALE_CHUNKS; ++ii) {
                auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                auto v_sc_tm_sub = v_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                load_q_scale_chunk<C>(p_sc_tm_sub, p_sc_smem, ii);
                load_k_scale_chunk<C>(v_sc_tm_sub, v_sc_smem, ii);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            if (idx == 0 || SECOND_TILE_MM2) {
                mm2_ABt(tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
            } else {
                mma2_ABt(tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
            }
            tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[0]);
            if (idx == g.tiles_to_run - 1) {
                detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
            }
        }
        __syncthreads();
    }

    if (warpgroup::groupid() == 2) {
        int end_phase = 0;
        int pv_phase = 0;
        wait(tile_arrived[0], end_phase);
        for (int idx = 0; idx < g.tiles_to_run; ++idx) {
            wait(pv_tmem_ready[0], pv_phase);
            pv_phase ^= 1;
        }
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);

        constexpr int SUBTILE_COLS = G::o_tile::swizzle_bytes / sizeof(bf16);
        constexpr uint32_t SWIZZLE_MASK = G::o_tile::swizzle_bytes * 8 - 1;
        uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
        uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);

        for (int col = 0; col < C::Dvo; col += 16) {
            float2 o_reg[8];
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16))
                              + row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                bf16_2 tmp0 = __float22bfloat162_rn(o_reg[i]);
                bf16_2 tmp1 = __float22bfloat162_rn(o_reg[i + 1]);
                uint32_t addr = row_base + i * 4;
                uint32_t swizzled_addr = addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4);
                uint64_t packed = (uint64_t)(*(uint32_t *)&tmp0) | ((uint64_t)(*(uint32_t *)&tmp1) << 32);
                asm volatile("st.shared.b64 [%0], %1;" :: "r"(swizzled_addr), "l"(packed));
            }
        }
        warpgroup::sync(warpgroup::groupid() + 1);
        warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o_snapshot, o_smem[0], {b, m_tile, h, 0});
        warpgroup::tma::store_async_wait<0>();
    }
}

template <typename C>
__global__ void kernel_two_tile_correction_replay_debug(const __grid_constant__ globals_two_tile_correction_replay_debug<C> g) {
    const int row_tiles = g.seqlen / C::Mb;
    const int linear = blockIdx.x;
    const int m_tile = linear % row_tiles;
    const int hb = linear / row_tiles;
    const int h = hb % g.heads;
    const int b = hb / g.heads;
    const int row = threadIdx.x;
    if (b >= g.batch || row >= C::Mb) {
        return;
    }
    const int global_row = m_tile * C::Mb + row;
    float corr = g.correction[((b * g.heads + h) * g.seqlen) + global_row];
    float row_sum = g.row_sum[((b * g.heads + h) * g.seqlen) + global_row];
    const bool row_invalid = (row_sum == 0.0f) || !isfinite(row_sum);
    if (!(corr >= 0.0f) || !isfinite(corr)) {
        corr = 0.0f;
    }
    for (int col = 0; col < C::Dvo; ++col) {
        float value = g.pre_correction[(((b * g.seqlen + global_row) * g.heads + h) * C::Dvo) + col];
        if (row_invalid) {
            value = 0.0f;
        } else if (corr < 1.0f) {
            value *= corr;
        }
        g.out[(((b * g.seqlen + global_row) * g.heads + h) * C::Dvo) + col] = __float2bfloat16_rn(value);
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_two_tile_pv_mainpath_debug(const __grid_constant__ globals_two_tile_pv_mainpath_debug<C> g) {
    using G = globals_two_tile_pv_mainpath_debug<C>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_p_sc = full_tt_fp8e4m3<C::P_SC_TMEM_WIDTH>;
    using d_tt_v_sc = full_tt_fp8e4m3<C::V_SC_TMEM_WIDTH>;

    const int cta_rank = cluster_ctarank();
    const int row_tiles = g.seqlen / C::Mb;
    const int row_clusters = row_tiles / C::CLUSTER_SIZE;
    const int cluster_linear = blockIdx.x / C::CLUSTER_SIZE;
    const int m_cluster = cluster_linear % row_clusters;
    const int hb = cluster_linear / row_clusters;
    const int h = hb % g.heads;
    const int b = hb / g.heads;
    const int m_tile = m_cluster * C::CLUSTER_SIZE + cta_rank;
    if (b >= g.batch || m_tile >= row_tiles ||
        g.tile_begin < 0 ||
        g.tiles_to_run < 1 || g.tiles_to_run > 2 ||
        g.tile_begin + g.tiles_to_run > row_tiles ||
        g.pv_wait_count < 1 || g.pv_wait_count > g.tiles_to_run) {
        return;
    }

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::v_fp4_tile (&v_fp4_smem)[C::LOAD_STAGES] = al.allocate<typename G::v_fp4_tile, C::LOAD_STAGES>();
    typename G::v_sc_tile (&v_sc_smem)[C::LOAD_STAGES] = al.allocate<typename G::v_sc_tile, C::LOAD_STAGES>();
    auto (&o_smem)[1] = al.allocate<typename G::o_tile, 1>();
    auto (&p_fp4_stage)[2] = al.allocate<typename G::p_fp4_tile, 2>();
    auto (&p_sc_stage)[2] = al.allocate<typename G::p_sc_tile, 2>();
    auto (&p_sc_guard)[1] = al.allocate<typename G::p_sc_guard_slot, 1>();

    __shared__ semaphore v_finished[C::LOAD_STAGES];
    __shared__ semaphore pv_tmem_ready[1], tile_arrived[1];

    if (threadIdx.x == 0) {
        g.o.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        init_semaphore(pv_tmem_ready[0], 0, 1);
        init_semaphore(tile_arrived[0], 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_STAGES; ++i) {
            init_semaphore(v_finished[i], 0, 1);
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    for (int idx = 0; idx < g.tiles_to_run; ++idx) {
        const int k_iter = g.tile_begin + idx;
        load_p_fp4_tile_from_global_allthreads<C>(p_fp4_stage[idx], g, b, h, m_tile, k_iter);
        load_p_sc_prepared_from_global_allthreads<C>(p_sc_stage[idx], g, b, h, m_tile, k_iter);
        __syncthreads();
        publish_shared_backing();
        publish_cluster_shared_backing_if_needed<C>();
        fp4pv_cluster_sync<C>();
        __syncthreads();
    }

    for (int idx = 0; idx < g.tiles_to_run; ++idx) {
        const int k_iter = g.tile_begin + idx;
        load_v_fp4_tile_from_global_allthreads<C>(v_fp4_smem[idx], g, b, h, k_iter, cta_rank);
        load_v_sc_prepared_from_global_allthreads<C>(v_sc_smem[idx], g, b, h, k_iter);
        __syncthreads();
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        publish_cluster_shared_backing_if_needed<C>();
        fp4pv_cluster_sync<C>();
        __syncthreads();
    }

    if (warpgroup::groupid() == 3) {
        if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<128>();
        }
        if (warpgroup::warpid() == 0 && cta_rank == 0 && warp::elect_leader()) {
            d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(C::Nb);
            constexpr int P_SC_BASE = C::Nb + C::Dvo + C::Q_SC_TMEM_WIDTH + 16;
            constexpr int V_SC_BASE = P_SC_BASE + 2 * C::P_SC_TMEM_WIDTH;
            static_assert(V_SC_BASE + 2 * C::V_SC_TMEM_WIDTH <= MAX_TENSOR_COLS, "two-tile mainpath debug TMEM layout exceeds budget");
            d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * C::P_SC_TMEM_WIDTH);
            d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * C::P_SC_TMEM_WIDTH);
            d_tt_v_sc v_sc_tm0 = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE + 0 * C::V_SC_TMEM_WIDTH);
            d_tt_v_sc v_sc_tm1 = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE + 1 * C::V_SC_TMEM_WIDTH);

            auto stage_v_sc = [&](int tt_buf, int v_slot) {
                auto &v_sc_tm_cur = tt_buf == 0 ? v_sc_tm0 : v_sc_tm1;
                #pragma unroll
                for (int ii = 0; ii < C::PV_SCALE_CHUNKS; ++ii) {
                    auto v_sc_tm_sub = v_sc_tm_cur.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                    load_k_scale_chunk<C>(v_sc_tm_sub, v_sc_smem[v_slot], ii);
                }
            };

            auto stage_p_sc = [&](int tt_buf, int p_buf) {
                auto &p_sc_tm_cur = tt_buf == 0 ? p_sc_tm0 : p_sc_tm1;
                #pragma unroll
                for (int ii = 0; ii < C::PV_SCALE_CHUNKS; ++ii) {
                    auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                    load_q_scale_chunk<C>(p_sc_tm_sub, p_sc_stage[p_buf], ii);
                }
            };

            auto issue_pv = [&](int tt_buf, int p_buf, int v_slot, bool is_first, bool is_final) {
                auto &p_sc_tm_cur = tt_buf == 0 ? p_sc_tm0 : p_sc_tm1;
                auto &v_sc_tm_cur = tt_buf == 0 ? v_sc_tm0 : v_sc_tm1;
                tensor_load_wait();
                tensor_before_thread_sync();
                if (is_first) {
                    mm2_ABt(tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_slot], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_slot]);
                } else {
                    if (g.second_tile_mm2) {
                        mm2_ABt(tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_slot], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_slot]);
                    } else {
                        mma2_ABt(tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_slot], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_slot]);
                    }
                }
                tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[0]);
                if (g.early_tile_arrived && is_first) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                }
                if (is_final) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                }
            };

            zero_output_scratch_issue_lane<C>(tt_output);
            stage_v_sc(0, 0);
            stage_p_sc(0, 0);
            if (g.tiles_to_run == 1) {
                issue_pv(0, 0, 0, true, true);
            } else {
                issue_pv(0, 0, 0, true, false);
                stage_v_sc(1, 1);
                stage_p_sc(1, 1);
                issue_pv(1, 1, 1, false, true);
            }
        }
    } else if (warpgroup::groupid() == 2) {
        if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<48>();
        }
        d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(C::Nb);
        int end_phase = 0;
        int pv_phase = 0;
        for (int idx = 0; idx < g.pv_wait_count; ++idx) {
            wait(pv_tmem_ready[0], pv_phase);
            pv_phase ^= 1;
        }
        wait(tile_arrived[0], end_phase);
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);

        constexpr int SUBTILE_COLS = G::o_tile::swizzle_bytes / sizeof(bf16);
        constexpr uint32_t SWIZZLE_MASK = G::o_tile::swizzle_bytes * 8 - 1;
        uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
        uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);
        for (int col = 0; col < C::Dvo; col += 16) {
            float2 o_reg[8];
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16))
                              + row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                bf16_2 tmp0 = __float22bfloat162_rn(o_reg[i]);
                bf16_2 tmp1 = __float22bfloat162_rn(o_reg[i + 1]);
                uint32_t addr = row_base + i * 4;
                uint32_t swizzled_addr = addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4);
                uint64_t packed = (uint64_t)(*(uint32_t *)&tmp0) | ((uint64_t)(*(uint32_t *)&tmp1) << 32);
                asm volatile("st.shared.b64 [%0], %1;" :: "r"(swizzled_addr), "l"(packed));
            }
        }
        warpgroup::sync(warpgroup::groupid() + 1);
        warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, o_smem[0], {b, m_tile, h, 0});
        warpgroup::tma::store_async_wait<0>();
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_pv_from_p(const __grid_constant__ globals_pv_from_p<C> g) {
    using G = globals_pv_from_p<C>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_p_sc = full_tt_fp8e4m3<C::P_SC_TMEM_WIDTH>;
    using d_tt_v_sc = full_tt_fp8e4m3<C::V_SC_TMEM_WIDTH>;

    const int cta_rank = cluster_ctarank();
    const int row_tiles = g.seqlen / C::Mb;
    const int row_clusters = row_tiles / C::CLUSTER_SIZE;
    const int cluster_linear = blockIdx.x / C::CLUSTER_SIZE;
    const int m_cluster = cluster_linear % row_clusters;
    const int hb = cluster_linear / row_clusters;
    const int h = hb % g.heads;
    const int b = hb / g.heads;
    const int m_tile = m_cluster * C::CLUSTER_SIZE + cta_rank;
    if (b >= g.batch) {
        return;
    }

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(0);
    constexpr int P_SC_BASE = C::Dvo;
    constexpr int V_SC_BASE = P_SC_BASE + C::P_SC_TMEM_WIDTH;
    static_assert(V_SC_BASE + C::V_SC_TMEM_WIDTH <= MAX_TENSOR_COLS, "PV-only TMEM layout exceeds tensor memory budget");
    d_tt_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE);
    d_tt_v_sc v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::p_fp4_tile &p_fp4_smem = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();
    auto (&o_smem)[1] = al.allocate<typename G::o_tile, 1>();

    __shared__ semaphore src_finished[1];
    __shared__ semaphore tile_arrived[1];
    __shared__ semaphore pv_tmem_ready[1];

    if (threadIdx.x == 0) {
        g.o.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        init_semaphore(src_finished[0], 0, 1);
        init_semaphore(tile_arrived[0], 0, 1);
        init_semaphore(pv_tmem_ready[0], 0, 1);
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0) && (cta_rank == 0);
    const bool issuer = issuer_warp && warp::elect_leader();
    int src_phase = 0;
    const int k_tiles = g.seqlen / C::Nb;

    for (int k_iter = 0; k_iter < k_tiles; ++k_iter) {
        if (threadIdx.x == 0 && k_iter > 0) {
            wait(src_finished[0], src_phase);
            src_phase ^= 1;
        }
        __syncthreads();

        load_p_fp4_tile_from_global_allthreads<C>(p_fp4_smem, g, b, h, m_tile, k_iter);
        load_p_sc_prepared_from_global_allthreads<C>(p_sc_smem, g, b, h, m_tile, k_iter);
        load_v_fp4_tile_from_global_allthreads<C>(v_fp4_smem, g, b, h, k_iter, cta_rank);
        load_v_sc_prepared_from_global_allthreads<C>(v_sc_smem, g, b, h, k_iter);
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        fp4pv_cluster_sync<C>();

        if (issuer) {
            #pragma unroll
            for (int ii = 0; ii < C::PV_SCALE_CHUNKS; ++ii) {
                auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                auto v_sc_tm_sub = v_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                load_q_scale_chunk<C>(p_sc_tm_sub, p_sc_smem, ii);
                load_k_scale_chunk<C>(v_sc_tm_sub, v_sc_smem, ii);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            if (k_iter == 0) {
                mm2_ABt(tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
            } else {
                mma2_ABt(tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
            }
            tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[0]);
            if (k_iter == k_tiles - 1) {
                detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
            }
        }
        __syncthreads();
    }

    if (warpgroup::groupid() == 2) {
        int end_phase = 0;
        int pv_phase = 0;
        wait(tile_arrived[0], end_phase);
        #pragma unroll
        for (int k_iter = 0; k_iter < k_tiles; ++k_iter) {
            wait(pv_tmem_ready[0], pv_phase);
            pv_phase ^= 1;
        }
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);

        constexpr int SUBTILE_COLS = G::o_tile::swizzle_bytes / sizeof(bf16);
        constexpr uint32_t SWIZZLE_MASK = G::o_tile::swizzle_bytes * 8 - 1;
        uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
        uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);

        for (int col = 0; col < C::Dvo; col += 16) {
            float2 o_reg[8];
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16))
                              + row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                bf16_2 tmp0 = __float22bfloat162_rn(o_reg[i]);
                bf16_2 tmp1 = __float22bfloat162_rn(o_reg[i + 1]);
                uint32_t addr = row_base + i * 4;
                uint32_t swizzled_addr = addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4);
                uint64_t packed = (uint64_t)(*(uint32_t *)&tmp0) | ((uint64_t)(*(uint32_t *)&tmp1) << 32);
                asm volatile("st.shared.b64 [%0], %1;" :: "r"(swizzled_addr), "l"(packed));
            }
        }
        warpgroup::sync(warpgroup::groupid() + 1);
        warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, o_smem[0], {b, m_tile, h, 0});
        warpgroup::tma::store_async_wait<0>();
    }
}

#if (!defined(TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) || \
     !TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) && \
    (!defined(TK_FA4_TRACK_B_M256_TWO_QUERY_CLUSTER1_ONLY_BUILD) || \
     !TK_FA4_TRACK_B_M256_TWO_QUERY_CLUSTER1_ONLY_BUILD) && \
    (!defined(TK_FA4_LARGE_I73_N64_COMBINE_ONLY_BUILD) || \
     !TK_FA4_LARGE_I73_N64_COMBINE_ONLY_BUILD)
__global__ void kernel_mxfp4_control_plane_diag(int32_t *debug, int mode) {
    __shared__ semaphore task_ready;
    __shared__ semaphore task_done;
    __shared__ volatile int task_bid;
    __shared__ int role_ready_mask;
    __shared__ int observe_mask;
    __shared__ int ack_mask;
    __shared__ int done_mask;
    __shared__ int timeout_mask;
    __shared__ int epoch_observe_mask[2];
    __shared__ int epoch_ack_mask[2];
    __shared__ int epoch_done_mask[2];
    __shared__ int epoch_timeout_mask[2];

    constexpr int SPIN_LIMIT = 1000000;
    constexpr int ROLE_COUNT_MAX = 8;
    constexpr int ALL_ROLES_MASK = (1 << ROLE_COUNT_MAX) - 1;

    const int role_count = (mode < 2) ? 1 : ROLE_COUNT_MAX;
    const int expected_mask = (1 << role_count) - 1;
    const int active_role_mask = (mode == 9 || mode == 10) ? (expected_mask & ~0x80) : expected_mask;
    const int late_ready_mask = (mode == 11 || mode == 12) ? (expected_mask & 0x80) : 0;
    const int partial_skip_mask = (mode == 6 || mode == 14) ? (expected_mask & 0xF0) : 0;
    const int skip_ack_mask = (mode == 4 || mode == 5) ? expected_mask : partial_skip_mask;
    const int skip_done_mask = (mode == 4 || mode == 7 || mode == 13) ? expected_mask : partial_skip_mask;
    const bool should_publish = (mode != 1);
    const bool drain_request = (mode == 0 || mode == 3 || mode == 8 || mode == 10 || mode == 12 || mode == 13 || mode == 14 || mode == 18);
    const bool early_publish = (mode == 8 || mode == 15);
    const int published_task = drain_request ? -1 : 42;

    if (threadIdx.x == 0) {
        init_semaphore(task_ready, 0, 1);
        init_semaphore(task_done, role_count, 0);
        task_bid = 0;
        role_ready_mask = 0;
        observe_mask = 0;
        ack_mask = 0;
        done_mask = 0;
        timeout_mask = 0;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            epoch_observe_mask[i] = 0;
            epoch_ack_mask[i] = 0;
            epoch_done_mask[i] = 0;
            epoch_timeout_mask[i] = 0;
        }
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            debug[i] = 0;
        }
        debug[0] = 0x43504432; // CPD2
        debug[1] = mode;
        debug[9] = static_cast<int32_t>(blockDim.x);
        debug[10] = published_task;
        debug[16] = role_count;
        debug[18] = drain_request ? 1 : 0;
        debug[19] = skip_ack_mask;
        debug[20] = 0;
        debug[21] = skip_done_mask;
        __threadfence_block();
    }
    __syncthreads();

    if (mode == 16 || mode == 17 || mode == 18) {
        const int role_id16 = static_cast<int>(threadIdx.x) - 32;
        if (role_id16 >= 0 && role_id16 < role_count) {
            const int role_bit = 1 << role_id16;
            atomicOr(&role_ready_mask, role_bit);
            __threadfence_block();

            int ready_phase = 0;
            #pragma unroll 1
            for (int epoch = 0; epoch < 2; ++epoch) {
                int role_spin = 0;
                bool observed = false;
                int observed_bid = -999;
                #pragma unroll 1
                for (; role_spin < SPIN_LIMIT; ++role_spin) {
                    if (try_wait(task_ready, ready_phase)) {
                        observed_bid = task_bid;
                        ready_phase ^= 1;
                        observed = true;
                        break;
                    }
                }
                if (observed) {
                    const bool skip_mode17_done =
                        mode == 17 && epoch == 1 && role_id16 == role_count - 1;
                    atomicOr(&observe_mask, role_bit);
                    atomicOr(&ack_mask, role_bit);
                    atomicOr(&epoch_observe_mask[epoch], role_bit);
                    atomicOr(&epoch_ack_mask[epoch], role_bit);
                    if (!skip_mode17_done) {
                        atomicOr(&done_mask, role_bit);
                        atomicOr(&epoch_done_mask[epoch], role_bit);
                        __threadfence_block();
                        arrive(task_done);
                    } else {
                        __threadfence_block();
                    }
                } else {
                    atomicOr(&timeout_mask, role_bit);
                    atomicOr(&epoch_timeout_mask[epoch], role_bit);
                }
                if (role_id16 == 0) {
                    debug[24 + epoch] = observed_bid;
                }
            }
        }

        if (threadIdx.x == 0) {
            int sched_spin = 0;
            #pragma unroll 1
            for (; sched_spin < SPIN_LIMIT && (role_ready_mask & expected_mask) != expected_mask; ++sched_spin) {}

            int task_done_phase = 1;
            int publish_count = 0;
            int task_done_count = 0;
            int reinit_count = 0;
            int epoch_observe[2] = {0, 0};
            int epoch_ack[2] = {0, 0};
            int epoch_done[2] = {0, 0};
            int epoch_timeout[2] = {0, 0};
            int done_spin[2] = {0, 0};

            #pragma unroll 1
            for (int epoch = 0; epoch < 2; ++epoch) {
                const int expected_task = (mode == 18 && epoch == 1) ? -1 : 42 + epoch;
                task_bid = expected_task;
                __threadfence_block();
                arrive(task_ready);
                ++publish_count;

                bool task_done_observed = false;
                #pragma unroll 1
                for (; done_spin[epoch] < SPIN_LIMIT; ++done_spin[epoch]) {
                    const int done_now = atomicAdd(&epoch_done_mask[epoch], 0) & expected_mask;
                    const int timeout_now = atomicAdd(&epoch_timeout_mask[epoch], 0) & expected_mask;
                    const bool mode17_partial_done =
                        mode == 17 && epoch == 1 && done_now == (expected_mask & ~0x80);
                    if (done_now == expected_mask || timeout_now != 0 || mode17_partial_done) {
                        task_done_observed = true;
                        break;
                    }
                }
                if (task_done_observed && (atomicAdd(&epoch_done_mask[epoch], 0) & expected_mask) == expected_mask) {
                    ++task_done_count;
                }

                epoch_observe[epoch] = atomicAdd(&epoch_observe_mask[epoch], 0) & expected_mask;
                epoch_ack[epoch] = atomicAdd(&epoch_ack_mask[epoch], 0) & expected_mask;
                epoch_done[epoch] = atomicAdd(&epoch_done_mask[epoch], 0) & expected_mask;
                epoch_timeout[epoch] = atomicAdd(&epoch_timeout_mask[epoch], 0) & expected_mask;

                if (epoch == 0) {
                    init_semaphore(task_done, role_count, 0);
                    task_done_phase = 0;
                    observe_mask = 0;
                    ack_mask = 0;
                    done_mask = 0;
                    timeout_mask = 0;
                    ++reinit_count;
                    __threadfence_block();
                } else {
                    task_done_phase ^= 1;
                }
            }

            const int final_ready = role_ready_mask & expected_mask;
            const int timeout_union = epoch_timeout[0] | epoch_timeout[1];
            const int expected_epoch1_done = (mode == 17) ? (expected_mask & ~0x80) : expected_mask;
            const bool masks_ok =
                epoch_observe[0] == expected_mask &&
                epoch_ack[0] == expected_mask &&
                epoch_done[0] == expected_mask &&
                epoch_observe[1] == expected_mask &&
                epoch_ack[1] == expected_mask &&
                epoch_done[1] == expected_epoch1_done;
            const bool protocol_ok =
                final_ready == expected_mask &&
                publish_count == 2 &&
                task_done_count == (mode == 17 ? 1 : 2) &&
                reinit_count == 1 &&
                timeout_union == 0 &&
                masks_ok &&
                debug[24] == 42 &&
                debug[25] == ((mode == 18) ? -1 : 43);
            const int timeout_code = protocol_ok ? (mode == 17 ? 3 : 0) : 3;
            const int reuse_ready = (protocol_ok && mode != 17) ? 1 : 0;

            debug[2] = final_ready;
            debug[3] = publish_count;
            debug[4] = epoch_observe[0];
            debug[5] = epoch_ack[0];
            debug[6] = epoch_done[0];
            debug[7] = timeout_code;
            debug[8] = timeout_union;
            debug[10] = (mode == 18) ? -1 : 43;
            debug[11] = protocol_ok ? 1 : 0;
            debug[12] = sched_spin;
            debug[13] = done_spin[0];
            debug[14] = reuse_ready;
            debug[15] = protocol_ok ? 1 : 0;
            debug[17] = task_done_count;
            debug[19] = 42;
            debug[20] = done_spin[1];
            debug[21] = epoch_observe[1];
            debug[22] = epoch_ack[1];
            debug[23] = epoch_done[1];
            debug[26] = reinit_count;
            debug[27] = task_done_phase;
            debug[28] = publish_count & 1;
            debug[29] = epoch_timeout[0];
            debug[30] = epoch_timeout[1];
            __threadfence_block();
        }
        __syncthreads();
        return;
    }

    const int role_id = static_cast<int>(threadIdx.x) - 32;
    if (role_id >= 0 && role_id < role_count && ((active_role_mask & (1 << role_id)) != 0)) {
        const int role_bit = 1 << role_id;
        if ((late_ready_mask & role_bit) != 0) {
            int late_spin = 0;
            #pragma unroll 1
            for (; late_spin < SPIN_LIMIT * 8; ++late_spin) {
                asm volatile("");
            }
            debug[23] = late_spin;
            __threadfence_block();
        }
        atomicOr(&role_ready_mask, role_bit);
        __threadfence_block();
        int observed_bid = 0;
        int role_spin = 0;
        bool observed = false;
        #pragma unroll 1
        for (; role_spin < SPIN_LIMIT; ++role_spin) {
            if (try_wait(task_ready, 0)) {
                observed_bid = task_bid;
                observed = true;
                break;
            }
        }
        if (observed) {
            atomicOr(&observe_mask, role_bit);
            if ((skip_ack_mask & role_bit) == 0) {
                atomicOr(&ack_mask, role_bit);
            }
            if ((skip_done_mask & role_bit) == 0) {
                atomicOr(&done_mask, role_bit);
                __threadfence_block();
                arrive(task_done);
            }
        } else {
            atomicOr(&timeout_mask, role_bit);
        }
        if (role_id == 0) {
            debug[8] = role_spin;
            debug[22] = observed_bid;
        }
        __threadfence_block();
    }

    if (threadIdx.x == 0) {
        int sched_spin = 0;
        if (should_publish && early_publish) {
            task_bid = published_task;
            __threadfence_block();
            arrive(task_ready);
            debug[3] = 1;
        }
        __threadfence_block();
        #pragma unroll 1
        for (; sched_spin < SPIN_LIMIT && (role_ready_mask & expected_mask) != expected_mask; ++sched_spin) {}
        debug[12] = sched_spin;
        if ((role_ready_mask & expected_mask) != expected_mask && mode != 1) {
            debug[7] = 1;
        }
        if (should_publish && !early_publish) {
            task_bid = published_task;
            __threadfence_block();
            arrive(task_ready);
            debug[3] = 1;
        } else {
            debug[3] = early_publish ? 1 : 0;
        }
        __threadfence_block();

        int done_spin = 0;
        bool task_done_observed = false;
        if (should_publish) {
            #pragma unroll 1
            for (; done_spin < SPIN_LIMIT; ++done_spin) {
                if (try_wait(task_done, 1)) {
                    task_done_observed = true;
                    break;
                }
            }
            debug[13] = done_spin;
            debug[17] = task_done_observed ? 1 : 0;
            if (!task_done_observed && debug[7] == 0) {
                debug[7] = 3;
            }
        } else {
            debug[17] = 0;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const int final_ready = role_ready_mask & ALL_ROLES_MASK;
        const int final_observe = observe_mask & ALL_ROLES_MASK;
        const int final_ack = ack_mask & ALL_ROLES_MASK;
        const int final_done = done_mask & ALL_ROLES_MASK;
        const int final_timeout = (timeout_mask | (expected_mask & ~final_ready)) & ALL_ROLES_MASK;
        const int expected_ack = active_role_mask & ~skip_ack_mask;
        const int expected_done = active_role_mask & ~skip_done_mask;
        if (debug[7] == 1 && final_ready == expected_mask && mode != 11 && mode != 12) {
            debug[7] = 0;
        }
        if (mode == 11 || mode == 12) {
            debug[20] = debug[17];
            debug[17] = 0;
        } else if (mode == 4 || mode == 6 || mode == 7 || mode == 9 || mode == 10 || mode == 13 || mode == 14) {
            debug[20] = debug[17];
            debug[17] = (final_done == expected_mask) ? 1 : 0;
            if (final_done != expected_mask && debug[7] == 0) {
                debug[7] = 3;
            }
        }
        const bool publish_ok = debug[3] == (should_publish ? 1 : 0);
        const bool ready_ok = (final_ready & expected_mask) == active_role_mask;
        const bool observe_ok = final_observe == (should_publish ? active_role_mask : 0);
        const bool ack_ok = final_ack == (should_publish ? expected_ack : 0);
        const bool done_ok = final_done == (should_publish ? expected_done : 0);
        const bool timeout_ok =
            (mode == 1) ? (debug[7] == 0 && final_timeout == expected_mask) :
            (mode == 9 || mode == 10) ? (debug[7] == 1 && final_timeout == (expected_mask & ~active_role_mask) && debug[17] == 0) :
            (mode == 11 || mode == 12) ? (debug[7] == 1 && final_timeout == 0 && final_ready == expected_mask && final_done == expected_mask && debug[17] == 0) :
            (mode == 4 || mode == 6 || mode == 7 || mode == 13 || mode == 14) ? (debug[7] == 3 && final_timeout == 0 && debug[17] == 0) :
                                                    (debug[7] == 0 && final_timeout == 0 && debug[17] == 1);
        const bool reuse_ready = should_publish && final_done == expected_mask && debug[7] == 0;
        debug[2] = final_ready;
        debug[4] = final_observe;
        debug[5] = final_ack;
        debug[6] = final_done;
        debug[8] = final_timeout;
        debug[11] = timeout_ok ? 1 : 0;
        debug[14] = reuse_ready ? 1 : 0;
        debug[15] = (publish_ok && ready_ok && observe_ok && ack_ok && done_ok && timeout_ok) ? 1 : 0;
        __threadfence_block();
    }
}
#endif

template <typename C>
__global__ void kernel_mxfp4_pv_from_p_chunked_debug(
    const __grid_constant__ globals_mxfp4_pv_from_p_chunked_debug<C> g
) {
    using G = globals_mxfp4_pv_from_p_chunked_debug<C>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_p_sc = full_tt_fp8e8m0<16>;
    using d_tt_v_sc = full_tt_fp8e8m0<32>;
    constexpr int MAX_K_TILES = 512;

    const int cta_rank = cluster_ctarank();
    const int row_tiles = g.seqlen / C::Mb;
    const int row_clusters = row_tiles / C::CLUSTER_SIZE;
    const int cluster_linear = blockIdx.x / C::CLUSTER_SIZE;
    const int m_cluster = cluster_linear % row_clusters;
    const int hb = cluster_linear / row_clusters;
    const int h = hb % g.heads;
    const int b = hb / g.heads;
    const int m_tile = m_cluster * C::CLUSTER_SIZE + cta_rank;
    if (b >= g.batch) {
        return;
    }

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    constexpr int SCORE_TMEM_WIDTH = C::Nb;
    d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(SCORE_TMEM_WIDTH);
    constexpr int P_SC_BASE = SCORE_TMEM_WIDTH + C::Dvo;
    constexpr int V_SC_BASE = P_SC_BASE + 16;
    static_assert(V_SC_BASE + 32 <= MAX_TENSOR_COLS, "MXFP4 debug PV TMEM layout exceeds tensor memory budget");
    d_tt_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE);
    d_tt_v_sc v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::p_fp4_tile &p_fp4_smem = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();
    auto (&o_smem)[1] = al.allocate<typename G::o_tile, 1>();

    __shared__ semaphore src_finished[1];
    __shared__ semaphore tile_arrived[1];
    __shared__ semaphore pv_tmem_ready[MAX_K_TILES];

    const int k_tiles = g.seqlen / C::Nb;
    if (k_tiles > MAX_K_TILES) {
        return;
    }
    if (threadIdx.x == 0) {
        g.o.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        init_semaphore(src_finished[0], 0, 1);
        init_semaphore(tile_arrived[0], 0, 1);
    }
    for (int i = threadIdx.x; i < k_tiles && i < MAX_K_TILES; i += blockDim.x) {
        init_semaphore(pv_tmem_ready[i], 0, 1);
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    if (g.debug_stage == 20) {
        if (warpgroup::groupid() == 2 && warpgroup::laneid() == 0) {
            uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
            asm volatile("st.shared.b64 [%0], %1;" :: "r"(base_addr), "l"(0ull));
        }
        return;
    }
    if (g.debug_stage == 21) {
        if (warpgroup::groupid() == 2) {
            uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
            for (int col = 0; col < C::Dvo; col += 16) {
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    uint32_t addr = G::o_tile::idx(base_addr, {warpgroup::laneid(), col + i * 2});
                    asm volatile("st.shared.b64 [%0], %1;" :: "r"(addr), "l"(0ull));
                }
            }
        }
        return;
    }
    if (g.debug_stage == 22) {
        if (warpgroup::groupid() == 2) {
            constexpr int SUBTILE_COLS = G::o_tile::swizzle_bytes / sizeof(bf16);
            constexpr uint32_t SWIZZLE_MASK = G::o_tile::swizzle_bytes * 8 - 1;
            uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
            uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);
            for (int col = 0; col < C::Dvo; col += 16) {
                uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16))
                                  + row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    uint32_t addr = row_base + i * 4;
                    uint32_t swizzled_addr = addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4);
                    asm volatile("st.shared.b64 [%0], %1;" :: "r"(swizzled_addr), "l"(0ull));
                }
            }
        }
        return;
    }
    if (g.debug_stage == 23) {
        if (warpgroup::groupid() == 2 && warpgroup::laneid() == 0) {
            uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
            for (int col = 0; col < C::Dvo; col += 16) {
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    uint32_t addr = G::o_tile::idx(base_addr, {0, col + i * 2});
                    asm volatile("st.shared.b64 [%0], %1;" :: "r"(addr), "l"(0ull));
                }
            }
        }
        return;
    }
    if (g.debug_stage == 24) {
        if (warpgroup::groupid() == 2) {
            uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
            uint32_t addr = G::o_tile::idx(base_addr, {warpgroup::laneid(), 0});
            asm volatile("st.shared.b64 [%0], %1;" :: "r"(addr), "l"(0ull));
        }
        return;
    }
    if (g.debug_stage == 25 || g.debug_stage == 26) {
        const int row_limit = (g.debug_stage == 25) ? 32 : 64;
        if (warpgroup::groupid() == 2 && warpgroup::laneid() < row_limit) {
            uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
            for (int col = 0; col < C::Dvo; col += 16) {
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    uint32_t addr = G::o_tile::idx(base_addr, {warpgroup::laneid(), col + i * 2});
                    asm volatile("st.shared.b64 [%0], %1;" :: "r"(addr), "l"(0ull));
                }
            }
        }
        return;
    }

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0) && (cta_rank == 0);
    const bool issuer = issuer_warp && warp::elect_leader();
    if (g.debug_stage == 15) {
        if (issuer) {
            #pragma unroll
            for (int subtile = 0; subtile < C::Mb / 32; subtile++) {
                #pragma unroll
                for (int col = 0; col < C::Dvo; col += 16) {
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};}"
                        :: "r"(tt_output.addr + ((subtile * 32) << 16) + col),
                           "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                           "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                           "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                           "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                }
            }
            tensor_store_wait();
            detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
        }
        __syncthreads();
        if (warpgroup::groupid() == 2) {
            wait(tile_arrived[0], 0);
            warpgroup::tma::store_async_read_wait<0>();
            warpgroup::sync(warpgroup::groupid() + 1);
            for (int col = 0; col < C::Dvo; col += 16) {
                float2 o_reg[8];
                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                      "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                      "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                      "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                    : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                tensor_load_wait();
            }
        }
        return;
    }
    int src_phase = 0;
    const int p_tiles = g.p_cols / C::Nb;
    const bool one_tile_debug = (g.debug_stage == 17 || g.debug_stage == 18);
    const bool unscaled_debug = (g.debug_stage == 16 || g.debug_stage == 18);
    const int issue_k_tiles = one_tile_debug ? 1 : k_tiles;
    if (g.debug_stage == 1) {
        return;
    }

    for (int k_iter = 0; k_iter < issue_k_tiles; ++k_iter) {
        if (threadIdx.x == 0 && k_iter > 0) {
            wait(src_finished[0], src_phase);
            src_phase ^= 1;
        }
        __syncthreads();

        const int p_iter = k_iter - (k_iter / p_tiles) * p_tiles;
        load_mxfp4_p_fp4_tile_from_global_allthreads<C>(p_fp4_smem, g, b, h, m_tile, p_iter);
        load_mxfp4_p_sc_from_global_allthreads<C>(p_sc_smem, g, b, h, m_tile, p_iter);
        load_mxfp4_v_fp4_tile_from_global_allthreads<C>(v_fp4_smem, g, b, h, k_iter, cta_rank);
        load_mxfp4_v_sc_from_global_allthreads<C>(v_sc_smem, g, b, h, k_iter);
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        publish_cluster_shared_backing_if_needed<C>();
        __syncthreads();
        fp4pv_cluster_sync<C>();
        if (g.debug_stage == 2) {
            return;
        }

        if (issuer_warp && g.debug_stage != 3 && !unscaled_debug) {
            auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
            auto v_sc_tm_sub = v_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
            fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_smem);
            fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm_sub, v_sc_smem);
            tensor_load_wait();
        }
        __syncthreads();
        if (issuer) {
            if (g.debug_stage != 3) {
                tensor_before_thread_sync();
                if (unscaled_debug) {
                    if (k_iter == 0) {
                        fp4pv_mm_p_stage_unscaled_ABt<C>(tt_output, p_fp4_smem, v_fp4_smem, src_finished[0]);
                    } else {
                        fp4pv_mma_p_stage_unscaled_ABt<C>(tt_output, p_fp4_smem, v_fp4_smem, src_finished[0]);
                    }
                } else {
                    if (k_iter == 0) {
                        if constexpr (C::Nb == 128) {
                            fp4pv_p_stage_ABt_split_k64<C, true>(
                                tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
                        } else {
                            fp4pv_mm_p_stage_ABt<C>(tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
                        }
                    } else {
                        if constexpr (C::Nb == 128) {
                            fp4pv_p_stage_ABt_split_k64<C, false>(
                                tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
                        } else {
                            fp4pv_mma_p_stage_ABt<C>(tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
                        }
                    }
                }
                if (g.debug_stage == 28) {
                    if (k_iter == issue_k_tiles - 1) {
                        tensor_commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                    }
                } else {
                    tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[k_iter]);
                }
                if (g.debug_stage != 28 && k_iter == issue_k_tiles - 1) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                }
            }
        }
        __syncthreads();
        __syncthreads();
        if (g.debug_stage == 3) {
            return;
        }
        if (g.debug_stage == 4) {
            if (threadIdx.x == 0) {
                wait(src_finished[0], src_phase);
            }
            __syncthreads();
            return;
        }
    }

    if (g.debug_stage == 5) {
        return;
    }

    if (warpgroup::groupid() == 2) {
        int end_phase = 0;
        if (g.debug_stage == 6) {
            wait(tile_arrived[0], end_phase);
            return;
        }
        if (g.debug_stage == 7) {
            #pragma unroll
            for (int k_iter = 0; k_iter < issue_k_tiles; ++k_iter) {
                wait(pv_tmem_ready[k_iter], 0);
            }
            return;
        }
        if (g.debug_stage == 8) {
            #pragma unroll
            for (int k_iter = 0; k_iter < issue_k_tiles; ++k_iter) {
                wait(pv_tmem_ready[k_iter], 0);
            }
            wait(tile_arrived[0], end_phase);
            return;
        }
        wait(tile_arrived[0], end_phase);
        if (g.debug_stage != 9 && g.debug_stage != 28) {
        #pragma unroll
        for (int k_iter = 0; k_iter < issue_k_tiles; ++k_iter) {
            wait(pv_tmem_ready[k_iter], 0);
        }
        }
        if (g.debug_stage == 11) {
            return;
        }
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        if (g.debug_stage == 12) {
            return;
        }

        if (g.debug_stage == 0 || g.debug_stage == 27 || g.debug_stage == 28) {
            const int row = m_tile * C::Mb + warpgroup::laneid();
            constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
            for (int col = 0; col < C::Dvo; col += 16) {
                float2 o_reg[8];
                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                      "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                      "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                      "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                    : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                tensor_load_wait();
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o_reg[i].x *= MXFP4_GEMM_SCALE_CORRECTION;
                    o_reg[i].y *= MXFP4_GEMM_SCALE_CORRECTION;
                    bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
                    const int out_offset = (((b * g.seqlen + row) * g.heads + h) * C::Dvo) + col + i * 2;
                    *reinterpret_cast<bf16_2 *>(&g.o_direct[out_offset]) = packed;
                }
            }
            return;
        }

        constexpr int SUBTILE_COLS = G::o_tile::swizzle_bytes / sizeof(bf16);
        constexpr uint32_t SWIZZLE_MASK = G::o_tile::swizzle_bytes * 8 - 1;
        uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
        uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);

        if (g.debug_stage == 14) {
            for (int col = 0; col < C::Dvo; col += 16) {
                #pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    uint32_t addr = G::o_tile::idx(base_addr, {warpgroup::laneid(), col + i * 2});
                    asm volatile("st.shared.b64 [%0], %1;" :: "r"(addr), "l"(0ull));
                }
            }
            return;
        }

        for (int col = 0; col < C::Dvo; col += 16) {
            float2 o_reg[8];
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            if (g.debug_stage == 13 || g.debug_stage == 19) {
                continue;
            }
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                bf16_2 tmp0 = __float22bfloat162_rn(o_reg[i]);
                bf16_2 tmp1 = __float22bfloat162_rn(o_reg[i + 1]);
                uint32_t swizzled_addr = G::o_tile::idx(base_addr, {warpgroup::laneid(), col + i * 2});
                uint64_t packed = (uint64_t)(*(uint32_t *)&tmp0) | ((uint64_t)(*(uint32_t *)&tmp1) << 32);
                asm volatile("st.shared.b64 [%0], %1;" :: "r"(swizzled_addr), "l"(packed));
            }
        }
        if (g.debug_stage == 19) {
            return;
        }
        warpgroup::sync(warpgroup::groupid() + 1);
        if (g.debug_stage == 10) {
            return;
        }
        warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, o_smem[0], {b, m_tile, h, 0});
        warpgroup::tma::store_async_wait<0>();
    }
}

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
template <typename C, int MIN_BLOCKS_PER_SM>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_pv_dvo64_residency_shell(
    const __grid_constant__ globals_mxfp4_pv_dvo64_residency_shell<C> g
) {
    using G = globals_mxfp4_pv_dvo64_residency_shell<C>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_scale = full_tt_fp8e8m0<16>;
    constexpr int TRACE_FIELDS = 16;
    static_assert(C::NUM_THREADS == 256, "DVO64 residency shell requires two warp groups");
    static_assert(C::CLUSTER_SIZE == 1, "DVO64 residency shell is CTA-local");
    static_assert(C::Mb == 128 && C::Nb == 128 && C::Dvo == 64,
                  "DVO64 residency shell targets M128/K128/N64");

    auto now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if (field >= 0 && field < TRACE_FIELDS) {
            g.trace[blockIdx.x * TRACE_FIELDS + field] = value;
        }
    };

    tensor_allocator<2, 1> tm_alloc{};
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(0);
    d_tt_scale p_sc_tm = tm_alloc.template allocate<d_tt_scale>(64);
    d_tt_scale v_sc_tm = tm_alloc.template allocate<d_tt_scale>(80);
    static_assert(80 + 16 <= tensor_allocator<2, 1>::cols,
                  "DVO64 residency shell exceeds its 256-column allocation");

    if (threadIdx.x == 0) {
        trace_store(0, 1);
        trace_store(1, blockIdx.x);
        trace_store(2, kittens::smid());
        trace_store(3, g.stage);
        trace_store(4, blockIdx.x % g.heads);
        trace_store(5, MIN_BLOCKS_PER_SM);
        trace_store(6, now());
        trace_store(15, tm_alloc.addr);
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::p_fp4_tile &p_fp4_smem = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();

    __shared__ semaphore pv_src_done;
    __shared__ semaphore pv_done;
    if (threadIdx.x == 0) {
        init_semaphore(pv_src_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
    }
    __syncthreads();

    const int h = blockIdx.x % g.heads;
    const int b = blockIdx.x / g.heads;
    const bool copy_p_scale = g.stage == 1 || g.stage >= 3;
    const bool copy_v_scale = g.stage == 2 || g.stage >= 3;
    const bool issue_pv = g.stage >= 4;
    const bool load_output = g.stage >= 5;

    if (issue_pv) {
        load_mxfp4_p_fp4_tile_from_global_allthreads<C>(
            p_fp4_smem, g, b, h, 0, 0);
        load_mxfp4_v_fp4_tile_from_global_allthreads<C>(
            v_fp4_smem, g, b, h, 0, 0);
    }
    if (copy_p_scale) {
        load_mxfp4_p_sc_from_global_allthreads<C>(
            p_sc_smem, g, b, h, 0, 0);
    }
    if (copy_v_scale) {
        load_mxfp4_v_sc_from_global_allthreads<C>(
            v_sc_smem, g, b, h, 0);
    }
    __syncthreads();
    fp4pv_publish_shared_backing_proxy_only();
    __syncthreads();

    const bool issue_warp =
        warpgroup::groupid() == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    if (issue_warp) {
        if (copy_p_scale) {
            if (issue_lane) {
                trace_store(7, now());
                fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm, p_sc_smem);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            if (issue_lane) {
                trace_store(8, now());
            }
        }
        if (copy_v_scale) {
            if (issue_lane) {
                trace_store(9, now());
                fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm, v_sc_smem);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            if (issue_lane) {
                trace_store(10, now());
            }
        }
        if (issue_lane && issue_pv) {
            trace_store(11, now());
            fp4pv_p_stage_ABt_split_k64<C, true>(
                tt_output,
                p_fp4_smem,
                v_fp4_smem,
                p_sc_tm,
                v_sc_tm,
                pv_src_done);
            tensor_commit<1>(pv_done);
        }
    }

    if (issue_pv && warpgroup::groupid() == 1) {
        wait(pv_done, 0);
        tensor_after_thread_sync();
        if (load_output) {
            const int output_warp = (static_cast<int>(threadIdx.x) - 128) >> 5;
            const int row = output_warp * 32 + (static_cast<int>(threadIdx.x) & 31);
            constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
            for (int col = 0; col < C::Dvo; col += 16) {
                float2 o_reg[8];
                asm volatile(
                    "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y),
                      "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                      "=f"(o_reg[2].x), "=f"(o_reg[2].y),
                      "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                      "=f"(o_reg[4].x), "=f"(o_reg[4].y),
                      "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                      "=f"(o_reg[6].x), "=f"(o_reg[6].y),
                      "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                    : "r"(tt_output.addr +
                          ((output_warp * 32) << 16) + col));
                tensor_load_wait();
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o_reg[i].x *= MXFP4_GEMM_SCALE_CORRECTION;
                    o_reg[i].y *= MXFP4_GEMM_SCALE_CORRECTION;
                    const bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
                    const int out_offset =
                        (((b * g.seqlen + row) * g.heads + h) * C::Dvo) +
                        col + i * 2;
                    *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
                }
            }
        }
        warpgroup::sync(2);
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(13, now());
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        trace_store(12, now());
        trace_store(14, now());
    }
    __syncthreads();
}

template <typename C, int MIN_BLOCKS_PER_SM>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_nb64_dvo128_residency_shell(
    const __grid_constant__ globals_mxfp4_nb64_dvo128_residency_shell<C> g
) {
    using G = globals_mxfp4_nb64_dvo128_residency_shell<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_qk_scale = full_tt_fp8e4m3<16>;
    using d_tt_pv_scale = full_tt_fp8e8m0<16>;
    constexpr int TRACE_FIELDS = 16;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = 64;
    constexpr int Q_SCALE_BASE = 192;
    constexpr int K_SCALE_BASE = 208;
    constexpr int P_SCALE_BASE = 224;
    constexpr int V_SCALE_BASE = 240;
    static_assert(C::NUM_THREADS == 256, "Nb64/DVO128 shell requires two warp groups");
    static_assert(C::CLUSTER_SIZE == 1, "Nb64/DVO128 shell is CTA-local");
    static_assert(C::Mb == 128 && C::Nb == 64 && C::Dqk == 192 && C::Dvo == 128,
                  "Nb64/DVO128 shell targets M128/N64/K192/DVO128");
    static_assert(V_SCALE_BASE + 16 == tensor_allocator<2, 1>::cols,
                  "Nb64/DVO128 shell must exactly fill its 256-column allocation");

    auto now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if (field >= 0 && field < TRACE_FIELDS) {
            g.trace[blockIdx.x * TRACE_FIELDS + field] = value;
        }
    };

    tensor_allocator<2, 1> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_qk_scale q_sc_tm = tm_alloc.template allocate<d_tt_qk_scale>(Q_SCALE_BASE);
    d_tt_qk_scale k_sc_tm = tm_alloc.template allocate<d_tt_qk_scale>(K_SCALE_BASE);
    d_tt_pv_scale p_sc_tm = tm_alloc.template allocate<d_tt_pv_scale>(P_SCALE_BASE);
    d_tt_pv_scale v_sc_tm = tm_alloc.template allocate<d_tt_pv_scale>(V_SCALE_BASE);

    if (threadIdx.x == 0) {
        trace_store(0, 1);
        trace_store(1, blockIdx.x);
        trace_store(2, kittens::smid());
        trace_store(3, g.stage);
        trace_store(4, blockIdx.x % g.heads);
        trace_store(5, MIN_BLOCKS_PER_SM);
        trace_store(6, now());
        trace_store(15, tm_alloc.addr);
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile &q_smem = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem = al.allocate<typename G::q_sc_tile>();
    typename G::k_tile &k_smem = al.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem = al.allocate<typename G::k_sc_tile>();
    typename G::p_fp4_tile &p_fp4_smem = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();

    __shared__ semaphore qk_done;
    __shared__ semaphore pv_src_done;
    __shared__ semaphore pv_done;
    if (threadIdx.x == 0) {
        init_semaphore(qk_done, 0, 1);
        init_semaphore(pv_src_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
    }
    __syncthreads();

    const int h = blockIdx.x % g.heads;
    const int b = blockIdx.x / g.heads;
    const bool issue_qk = g.stage >= 1 && g.stage <= 3;
    const bool issue_pv = g.stage >= 2;
    const bool load_output = g.stage >= 3;

    auto load_payload = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int BYTES_PER_STORE = 4;
        static_assert(Tile::cols % BYTES_PER_STORE == 0,
                      "Residency shell payload tile must support word stores");
        constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
        constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + row * Tile::cols + col);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(Tile::idx(smem_base, {row, col})), "r"(value));
        }
    };
    auto load_scale = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int WORD_BYTES = 4;
        static_assert(sizeof(Tile) % WORD_BYTES == 0,
                      "Residency shell scale tile must be word aligned");
        constexpr int WORDS = sizeof(Tile) / WORD_BYTES;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int word = threadIdx.x; word < WORDS; word += blockDim.x) {
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + word * WORD_BYTES);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(smem_base + word * WORD_BYTES), "r"(value));
        }
    };

    if (issue_qk) {
        const uint8_t *q_src = g.q_fp4 + h * sizeof(typename G::q_tile);
        const uint8_t *k_src = g.k_fp4 + h * sizeof(typename G::k_tile);
        const uint8_t *q_sc_src = g.q_sc + h * sizeof(typename G::q_sc_tile);
        const uint8_t *k_sc_src = g.k_sc + h * sizeof(typename G::k_sc_tile);
        load_payload(q_smem, q_src);
        load_payload(k_smem, k_src);
        load_scale(q_sc_smem, q_sc_src);
        load_scale(k_sc_smem, k_sc_src);
    }
    if (issue_pv) {
        load_mxfp4_p_fp4_tile_from_global_allthreads<C>(
            p_fp4_smem, g, b, h, 0, 0);
        load_mxfp4_p_sc_from_global_allthreads<C>(
            p_sc_smem, g, b, h, 0, 0);
        load_mxfp4_v_fp4_tile_from_global_allthreads<C>(
            v_fp4_smem, g, b, h, 0, 0);
        load_mxfp4_v_sc_from_global_allthreads<C>(
            v_sc_smem, g, b, h, 0);
    }
    __syncthreads();
    fp4pv_publish_shared_backing_proxy_only();
    __syncthreads();

    const bool issue_warp =
        warpgroup::groupid() == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    if (issue_warp && issue_qk) {
        using T_AB = typename G::q_tile::T;
        using T_SAB = typename d_tt_qk_scale::T;
        using T_D = typename d_tt_score::T;
        constexpr int M = C::Mb;
        constexpr int N = C::Nb;
        constexpr uint32_t IDESC =
            detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
        st_descriptor<typename G::q_tile, 0> q_desc(q_smem);
        st_descriptor<typename G::k_tile, 0> k_desc(k_smem);
        tensor_before_thread_sync();
        if (issue_lane) {
            trace_store(7, now());
        }
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            if (issue_lane) {
                fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_smem, chunk);
                fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm, k_sc_smem, chunk);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            if (issue_lane) {
                if (chunk == 0) {
                    detail::tcgen05::template st_st<T_AB, T_SAB, 0, 1, 16>(
                        tt_score.addr, q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk), q_sc_tm.addr, k_sc_tm.addr, IDESC);
                } else {
                    detail::tcgen05::template st_st<T_AB, T_SAB, 1, 1, 16>(
                        tt_score.addr, q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk), q_sc_tm.addr, k_sc_tm.addr, IDESC);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done);
        }
    }
    if (issue_warp && issue_qk) {
        wait(qk_done, 0);
        tensor_after_thread_sync();
        if (issue_lane) {
            trace_store(8, now());
        }
    }
    if (issue_warp && issue_pv) {
        if (issue_lane) {
            fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm, p_sc_smem);
            fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm, v_sc_smem);
        }
        tensor_load_wait();
        tensor_before_thread_sync();
        if (issue_lane) {
            trace_store(9, now());
            fp4pv_mm_p_stage_ABt<C>(
                tt_output, p_fp4_smem, v_fp4_smem,
                p_sc_tm, v_sc_tm, pv_src_done);
            tensor_commit<1>(pv_done);
            trace_store(10, now());
        }
    }

    if (issue_pv && warpgroup::groupid() == 1) {
        wait(pv_done, 0);
        tensor_after_thread_sync();
        if (load_output) {
            const int output_warp = (static_cast<int>(threadIdx.x) - 128) >> 5;
            const int row = output_warp * 32 + (static_cast<int>(threadIdx.x) & 31);
            constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
            for (int col = 0; col < C::Dvo; col += 16) {
                float2 o_reg[8];
                asm volatile(
                    "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y),
                      "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                      "=f"(o_reg[2].x), "=f"(o_reg[2].y),
                      "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                      "=f"(o_reg[4].x), "=f"(o_reg[4].y),
                      "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                      "=f"(o_reg[6].x), "=f"(o_reg[6].y),
                      "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                    : "r"(tt_output.addr + ((output_warp * 32) << 16) + col));
                tensor_load_wait();
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o_reg[i].x *= MXFP4_GEMM_SCALE_CORRECTION;
                    o_reg[i].y *= MXFP4_GEMM_SCALE_CORRECTION;
                    const bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
                    const int out_offset =
                        (((b * g.seqlen + row) * g.heads + h) * C::Dvo) +
                        col + i * 2;
                    *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
                }
            }
        }
        warpgroup::sync(2);
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(11, now());
            trace_store(13, now());
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        trace_store(12, now());
        trace_store(14, now());
    }
    __syncthreads();
}

template <typename C, int MIN_BLOCKS_PER_SM>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_nb64_dual_dvo64_residency_shell(
    const __grid_constant__ globals_mxfp4_nb64_dvo128_residency_shell<C> g
) {
    using G = globals_mxfp4_nb64_dvo128_residency_shell<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_qk_scale = full_tt_fp8e4m3<16>;
    using d_tt_pv_scale = full_tt_fp8e8m0<16>;
    constexpr int TRACE_FIELDS = 20;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT0_BASE = 64;
    constexpr int OUTPUT1_BASE = 128;
    constexpr int Q_SCALE_BASE = 192;
    constexpr int K_SCALE_BASE = 208;
    constexpr int P_SCALE_BASE = 224;
    constexpr int V_SCALE_BASE = 240;
    static_assert(C::NUM_THREADS == 256, "dual-DVO64 shell requires two warp groups");
    static_assert(C::CLUSTER_SIZE == 1, "dual-DVO64 shell is CTA-local");
    static_assert(C::Mb == 128 && C::Nb == 64 && C::Dqk == 192 && C::Dvo == 64,
                  "dual-DVO64 shell targets M128/N64/K192/DVO64x2");
    static_assert(V_SCALE_BASE + 16 == tensor_allocator<2, 1>::cols,
                  "dual-DVO64 shell must exactly fill its 256-column allocation");

    auto now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if (field >= 0 && field < TRACE_FIELDS) {
            g.trace[blockIdx.x * TRACE_FIELDS + field] = value;
        }
    };

    tensor_allocator<2, 1> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output0 = tm_alloc.template allocate<d_tt_output>(OUTPUT0_BASE);
    d_tt_output tt_output1 = tm_alloc.template allocate<d_tt_output>(OUTPUT1_BASE);
    d_tt_qk_scale q_sc_tm = tm_alloc.template allocate<d_tt_qk_scale>(Q_SCALE_BASE);
    d_tt_qk_scale k_sc_tm = tm_alloc.template allocate<d_tt_qk_scale>(K_SCALE_BASE);
    d_tt_pv_scale p_sc_tm = tm_alloc.template allocate<d_tt_pv_scale>(P_SCALE_BASE);
    d_tt_pv_scale v_sc_tm = tm_alloc.template allocate<d_tt_pv_scale>(V_SCALE_BASE);

    if (threadIdx.x == 0) {
        trace_store(0, 1);
        trace_store(1, blockIdx.x);
        trace_store(2, kittens::smid());
        trace_store(3, g.stage);
        trace_store(4, blockIdx.x % g.heads);
        trace_store(5, MIN_BLOCKS_PER_SM);
        trace_store(6, now());
        trace_store(19, tm_alloc.addr);
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile &q_smem = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem = al.allocate<typename G::q_sc_tile>();
    typename G::k_tile &k_smem = al.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem = al.allocate<typename G::k_sc_tile>();
    typename G::p_fp4_tile &p_fp4_smem = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();

    __shared__ semaphore qk_done;
    __shared__ semaphore qk_scale_done[C::QK_SCALE_CHUNKS];
    __shared__ semaphore pv_scale_done0;
    __shared__ semaphore pv_scale_done1;
    __shared__ semaphore pv_src_done0;
    __shared__ semaphore pv_src_done1;
    __shared__ semaphore pv_done0;
    __shared__ semaphore pv_done1;
    if (threadIdx.x == 0) {
        init_semaphore(qk_done, 0, 1);
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            init_semaphore(qk_scale_done[chunk], 0, 1);
        }
        init_semaphore(pv_scale_done0, 0, 1);
        init_semaphore(pv_scale_done1, 0, 1);
        init_semaphore(pv_src_done0, 0, 1);
        init_semaphore(pv_src_done1, 0, 1);
        init_semaphore(pv_done0, 0, 1);
        init_semaphore(pv_done1, 0, 1);
    }
    __syncthreads();

    const int h = blockIdx.x % g.heads;
    const int b = blockIdx.x / g.heads;
    const bool issue_pv = g.stage >= 1;
    const bool issue_dual_pv = g.stage >= 2;
    const bool issue_qk = g.stage >= 3;

    auto load_payload = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int BYTES_PER_STORE = 4;
        constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
        constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + row * Tile::cols + col);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(Tile::idx(smem_base, {row, col})), "r"(value));
        }
    };
    auto load_scale_record = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int WORD_BYTES = 4;
        constexpr int WORDS = sizeof(Tile) / WORD_BYTES;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int word = threadIdx.x; word < WORDS; word += blockDim.x) {
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + word * WORD_BYTES);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(smem_base + word * WORD_BYTES), "r"(value));
        }
    };
    auto load_v_half = [&](int half) {
        using VTile = typename G::v_fp4_tile;
        constexpr int BYTES_PER_STORE = 4;
        constexpr int STORES_PER_ROW = VTile::cols / BYTES_PER_STORE;
        constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
        const int packed_seqlen = g.seqlen / 2;
        const uint32_t smem_base =
            __cvta_generic_to_shared(&v_fp4_smem.data[0]);
        for (int linear = threadIdx.x; linear < TOTAL_STORES;
             linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const int global_row =
                ((b * g.heads + h) * 2 + half) * C::Dvo + row;
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                g.v_fp4 + global_row * packed_seqlen + col);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(VTile::idx(smem_base, {row, col})),
                            "r"(value));
        }
        const uint8_t *v_sc_src = g.v_sc +
            (h * g.v_scale_rows + half) * sizeof(typename G::v_sc_tile);
        load_scale_record(v_sc_smem, v_sc_src);
    };

    if (issue_qk) {
        load_payload(q_smem, g.q_fp4 + h * sizeof(typename G::q_tile));
        load_payload(k_smem, g.k_fp4 + h * sizeof(typename G::k_tile));
        load_scale_record(q_sc_smem, g.q_sc + h * sizeof(typename G::q_sc_tile));
        load_scale_record(k_sc_smem, g.k_sc + h * sizeof(typename G::k_sc_tile));
    }
    if (issue_pv) {
        load_mxfp4_p_fp4_tile_from_global_allthreads<C>(
            p_fp4_smem, g, b, h, 0, 0);
        load_scale_record(
            p_sc_smem, g.p_sc + h * sizeof(typename G::p_sc_tile));
        load_v_half(0);
    }
    __syncthreads();
    fp4pv_publish_shared_backing_proxy_only();
    __syncthreads();

    const bool issue_warp =
        warpgroup::groupid() == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    if (issue_warp && issue_qk) {
        using T_AB = typename G::q_tile::T;
        using T_SAB = typename d_tt_qk_scale::T;
        using T_D = typename d_tt_score::T;
        constexpr uint32_t IDESC =
            detail::tcgen05::instruction_descriptor<
                T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
        st_descriptor<typename G::q_tile, 0> q_desc(q_smem);
        st_descriptor<typename G::k_tile, 0> k_desc(k_smem);
        if (issue_lane) {
            trace_store(7, now());
        }
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            if (issue_lane) {
                fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_smem, chunk);
                fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm, k_sc_smem, chunk);
                detail::tcgen05::commit<1>(qk_scale_done[chunk]);
            }
            wait(qk_scale_done[chunk], 0);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
            if (issue_lane) {
                if (chunk == 0) {
                    detail::tcgen05::template st_st<T_AB, T_SAB, 0, 1, 16>(
                        tt_score.addr, q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk), q_sc_tm.addr, k_sc_tm.addr, IDESC);
                } else {
                    detail::tcgen05::template st_st<T_AB, T_SAB, 1, 1, 16>(
                        tt_score.addr, q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk), q_sc_tm.addr, k_sc_tm.addr, IDESC);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done);
        }
        wait(qk_done, 0);
        tensor_after_thread_sync();
        if (issue_lane) {
            trace_store(8, now());
        }
    }

    if (issue_warp && issue_pv) {
        if (issue_lane) {
            fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm, p_sc_smem);
            fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm, v_sc_smem);
            detail::tcgen05::commit<1>(pv_scale_done0);
        }
        wait(pv_scale_done0, 0);
        tensor_after_thread_sync();
        tensor_before_thread_sync();
        if (issue_lane) {
            trace_store(9, now());
            fp4pv_mm_p_stage_ABt<C>(
                tt_output0, p_fp4_smem, v_fp4_smem,
                p_sc_tm, v_sc_tm, pv_src_done0);
            tensor_commit<1>(pv_done0);
        }
    }
    if (issue_pv) {
        wait(pv_done0, 0);
        tensor_after_thread_sync();
    }
    __syncthreads();
    if (threadIdx.x == 0 && issue_pv) {
        trace_store(10, now());
    }

    if (issue_dual_pv) {
        load_v_half(1);
        __syncthreads();
        fp4pv_publish_shared_backing_proxy_only();
        __syncthreads();
        if (threadIdx.x == 0) {
            trace_store(11, now());
        }
        if (issue_warp) {
            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm, v_sc_smem);
                detail::tcgen05::commit<1>(pv_scale_done1);
            }
            wait(pv_scale_done1, 0);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
            if (issue_lane) {
                trace_store(12, now());
                fp4pv_mm_p_stage_ABt<C>(
                    tt_output1, p_fp4_smem, v_fp4_smem,
                    p_sc_tm, v_sc_tm, pv_src_done1);
                tensor_commit<1>(pv_done1);
            }
        }
    }

    if (issue_pv && warpgroup::groupid() == 1) {
        if (issue_dual_pv) {
            wait(pv_done1, 0);
            tensor_after_thread_sync();
        }
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(13, now());
        }
        const int output_warp = (static_cast<int>(threadIdx.x) - 128) >> 5;
        const int row = output_warp * 32 + (static_cast<int>(threadIdx.x) & 31);
        const int output_halves = issue_dual_pv ? 2 : 1;
        constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
        for (int half = 0; half < output_halves; ++half) {
            const d_tt_output &tt_output = half == 0 ? tt_output0 : tt_output1;
            for (int col = 0; col < C::Dvo; col += 16) {
                float2 o_reg[8];
                asm volatile(
                    "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y),
                      "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                      "=f"(o_reg[2].x), "=f"(o_reg[2].y),
                      "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                      "=f"(o_reg[4].x), "=f"(o_reg[4].y),
                      "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                      "=f"(o_reg[6].x), "=f"(o_reg[6].y),
                      "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                    : "r"(tt_output.addr + ((output_warp * 32) << 16) + col));
                tensor_load_wait();
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o_reg[i].x *= MXFP4_GEMM_SCALE_CORRECTION;
                    o_reg[i].y *= MXFP4_GEMM_SCALE_CORRECTION;
                    const bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
                    const int out_offset =
                        (((b * g.seqlen + row) * g.heads + h) * 128) +
                        half * C::Dvo + col + i * 2;
                    *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
                }
            }
        }
        warpgroup::sync(2);
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(14, now());
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        trace_store(15, now());
        trace_store(18, now());
    }
    __syncthreads();
}

__device__ __forceinline__ void fp4pv_load_score_quarter_tmem(
    float2 (&scores)[16],
    uint32_t score_tt_base,
    int qid);

__device__ inline float softmax_rescale_factor_strict(
    float row_max,
    float row_max_old,
    float scale_log2);

template <typename C, typename OutputTT>
__device__ __forceinline__ void rescale_tt_output_inplace_direct_streaming(
    OutputTT &tt_output,
    float correction,
    bool lane_needs_rescale);

template <typename C, typename OutputTT, typename OTile>
__device__ __forceinline__ void restore_rescaled_output_direct(
    OutputTT &tt_output,
    OTile &o_backing,
    float correction) {
    const int row = warpgroup::laneid();
    const float2 correction2 = {correction, correction};
    #pragma unroll
    for (int col = 0; col < OutputTT::cols; col += 16) {
        float2 out[8];
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            out[pair].x = o_backing[{row, col + pair * 2}];
            out[pair].y = o_backing[{row, col + pair * 2 + 1}];
            out[pair] = __fmul2_rn(out[pair], correction2);
        }
        const uint32_t addr = tt_output.addr +
            ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
            :: "f"(out[0].x), "f"(out[0].y),
               "f"(out[1].x), "f"(out[1].y),
               "f"(out[2].x), "f"(out[2].y),
               "f"(out[3].x), "f"(out[3].y),
               "f"(out[4].x), "f"(out[4].y),
               "f"(out[5].x), "f"(out[5].y),
               "f"(out[6].x), "f"(out[6].y),
               "f"(out[7].x), "f"(out[7].y), "r"(addr));
    }
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OTile, typename OutputTT>
__device__ __forceinline__ void accumulate_output_backing_direct(
    OTile &o_backing,
    OutputTT &tt_output,
    float correction,
    bool has_prior) {
    const int row = warpgroup::laneid();
    #pragma unroll
    for (int col = 0; col < OutputTT::cols; col += 16) {
        float2 out[8];
        const uint32_t addr = tt_output.addr +
            ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(out[0].x), "=f"(out[0].y),
              "=f"(out[1].x), "=f"(out[1].y),
              "=f"(out[2].x), "=f"(out[2].y),
              "=f"(out[3].x), "=f"(out[3].y),
              "=f"(out[4].x), "=f"(out[4].y),
              "=f"(out[5].x), "=f"(out[5].y),
              "=f"(out[6].x), "=f"(out[6].y),
              "=f"(out[7].x), "=f"(out[7].y)
            : "r"(addr));
        tensor_load_wait();
        tensor_before_thread_sync();
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const float prior_x = has_prior
                ? o_backing[{row, col + pair * 2}] : 0.0f;
            const float prior_y = has_prior
                ? o_backing[{row, col + pair * 2 + 1}] : 0.0f;
            o_backing[{row, col + pair * 2}] =
                __fmaf_rn(prior_x, correction, out[pair].x);
            o_backing[{row, col + pair * 2 + 1}] =
                __fmaf_rn(prior_y, correction, out[pair].y);
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, int MIN_BLOCKS_PER_SM>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_nb128_dual_dvo64_attention_shell(
    const __grid_constant__ globals_mxfp4_nb128_dual_dvo64_attention_shell<C> g
) {
    using G = globals_mxfp4_nb128_dual_dvo64_attention_shell<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_qk_scale = full_tt_fp8e4m3<16>;
    using d_tt_pv_scale = full_tt_fp8e8m0<16>;
    constexpr int TRACE_FIELDS = 24;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT0_BASE = 0;
    constexpr int OUTPUT1_BASE = 64;
    constexpr int Q_OR_P_SCALE_BASE = 128;
    constexpr int K_OR_V_SCALE_BASE = 144;
    static_assert(C::NUM_THREADS == 256,
                  "Nb128 dual-DVO64 attention shell requires two warp groups");
    static_assert(C::CLUSTER_SIZE == 1,
                  "Nb128 dual-DVO64 attention shell is CTA-local");
    static_assert(C::Mb == 128 && C::Nb == 128 && C::Dqk == 192 && C::Dvo == 64,
                  "Nb128 dual-DVO64 attention shell targets M128/N128/K192/DVO64x2");
    static_assert(K_OR_V_SCALE_BASE + 16 <= tensor_allocator<2, 1>::cols,
                  "Nb128 dual-DVO64 attention shell exceeds 256 TMEM columns");

    auto now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if (field >= 0 && field < TRACE_FIELDS) {
            g.trace[blockIdx.x * TRACE_FIELDS + field] = value;
        }
    };

    tensor_allocator<2, 1> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output0 =
        tm_alloc.template allocate<d_tt_output>(OUTPUT0_BASE);
    d_tt_output tt_output1 =
        tm_alloc.template allocate<d_tt_output>(OUTPUT1_BASE);
    d_tt_qk_scale q_sc_tm =
        tm_alloc.template allocate<d_tt_qk_scale>(Q_OR_P_SCALE_BASE);
    d_tt_qk_scale k_sc_tm =
        tm_alloc.template allocate<d_tt_qk_scale>(K_OR_V_SCALE_BASE);
    d_tt_pv_scale p_sc_tm =
        tm_alloc.template allocate<d_tt_pv_scale>(Q_OR_P_SCALE_BASE);
    d_tt_pv_scale v_sc_tm =
        tm_alloc.template allocate<d_tt_pv_scale>(K_OR_V_SCALE_BASE);

    if (threadIdx.x == 0) {
        trace_store(0, 1);
        trace_store(1, blockIdx.x);
        trace_store(2, kittens::smid());
        trace_store(3, g.stage);
        trace_store(4, blockIdx.x % g.heads);
        trace_store(5, MIN_BLOCKS_PER_SM);
        trace_store(6, now());
        trace_store(21, tm_alloc.addr);
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile &q_smem = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem = al.allocate<typename G::q_sc_tile>();
    typename G::k_tile &k_smem = al.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem = al.allocate<typename G::k_sc_tile>();
    typename G::p_fp4_tile &p_fp4_smem = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem = al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();

    __shared__ semaphore qk_done;
    __shared__ semaphore score_consumed;
    __shared__ semaphore p_ready;
    __shared__ semaphore pv_src_done0;
    __shared__ semaphore pv_src_done1;
    __shared__ semaphore pv_done0;
    __shared__ semaphore pv_done1;
    if (threadIdx.x == 0) {
        init_semaphore(qk_done, 0, 1);
        init_semaphore(score_consumed, 0, 1);
        init_semaphore(p_ready, 0, 1);
        init_semaphore(pv_src_done0, 0, 1);
        init_semaphore(pv_src_done1, 0, 1);
        init_semaphore(pv_done0, 0, 1);
        init_semaphore(pv_done1, 0, 1);
    }
    __syncthreads();

    const int h = blockIdx.x % g.heads;
    const int b = blockIdx.x / g.heads;
    const bool issue_qk = g.stage >= 1;
    const bool produce_p = g.stage >= 2;
    const bool issue_pv = g.stage >= 3;
    const bool normalize_output = g.stage >= 4;

    auto load_payload = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int BYTES_PER_STORE = 4;
        constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
        constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int linear = threadIdx.x; linear < TOTAL_STORES;
             linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + row * Tile::cols + col);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(Tile::idx(smem_base, {row, col})), "r"(value));
        }
    };
    auto load_scale_record = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int WORD_BYTES = 4;
        constexpr int WORDS = sizeof(Tile) / WORD_BYTES;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int word = threadIdx.x; word < WORDS; word += blockDim.x) {
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + word * WORD_BYTES);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(smem_base + word * WORD_BYTES), "r"(value));
        }
    };
    auto load_v_half = [&](int half) {
        using VTile = typename G::v_fp4_tile;
        constexpr int BYTES_PER_STORE = 4;
        constexpr int STORES_PER_ROW = VTile::cols / BYTES_PER_STORE;
        constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
        const int packed_seqlen = g.seqlen / 2;
        const uint32_t smem_base =
            __cvta_generic_to_shared(&v_fp4_smem.data[0]);
        for (int linear = threadIdx.x; linear < TOTAL_STORES;
             linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const int global_row =
                ((b * g.heads + h) * 2 + half) * C::Dvo + row;
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                g.v_fp4 + global_row * packed_seqlen + col);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(VTile::idx(smem_base, {row, col})), "r"(value));
        }
        const uint8_t *v_sc_src = g.v_sc +
            (h * g.v_scale_rows + half) * sizeof(typename G::v_sc_tile);
        load_scale_record(v_sc_smem, v_sc_src);
    };

    if (issue_qk) {
        load_payload(q_smem, g.q_fp4 + h * sizeof(typename G::q_tile));
        load_payload(k_smem, g.k_fp4 + h * sizeof(typename G::k_tile));
        load_scale_record(q_sc_smem,
                          g.q_sc + h * sizeof(typename G::q_sc_tile));
        load_scale_record(k_sc_smem,
                          g.k_sc + h * sizeof(typename G::k_sc_tile));
    }
    if (issue_pv) {
        load_v_half(0);
    }
    __syncthreads();
    fp4pv_publish_shared_backing_proxy_only();
    __syncthreads();

    const bool issue_warp =
        warpgroup::groupid() == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    if (issue_warp && issue_qk) {
        using T_AB = typename G::q_tile::T;
        using T_SAB = typename d_tt_qk_scale::T;
        using T_D = typename d_tt_score::T;
        constexpr uint32_t IDESC =
            detail::tcgen05::instruction_descriptor<
                T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
        st_descriptor<typename G::q_tile, 0> q_desc(q_smem);
        st_descriptor<typename G::k_tile, 0> k_desc(k_smem);
        if (issue_lane) {
            trace_store(7, now());
        }
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            if (issue_lane) {
                fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_smem, chunk);
                fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm, k_sc_smem, chunk);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            if (issue_lane) {
                if (chunk == 0) {
                    detail::tcgen05::template st_st<T_AB, T_SAB, 0, 1, 16>(
                        tt_score.addr, q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk), q_sc_tm.addr,
                        k_sc_tm.addr, IDESC);
                } else {
                    detail::tcgen05::template st_st<T_AB, T_SAB, 1, 1, 16>(
                        tt_score.addr, q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk), q_sc_tm.addr,
                        k_sc_tm.addr, IDESC);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done);
        }
        wait(qk_done, 0);
        tensor_after_thread_sync();
        if (issue_lane) {
            trace_store(8, now());
        }
    }

    float row_sum_local = 1.0f;
    if (warpgroup::groupid() == 1 && issue_qk) {
        wait(qk_done, 0);
        tensor_after_thread_sync();
        const int global_row = warpgroup::laneid();
        const uint32_t score_tt_base =
            tt_score.addr + ((warpgroup::warpid() * 32) << 16);
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(9, now());
        }
        if (produce_p) {
            float block_max[4];
            float row_max = base_types::constants<float>::neg_infty();
            #pragma unroll
            for (int qid = 0; qid < 4; ++qid) {
                float2 score_chunk[16];
                fp4pv_load_score_quarter_tmem(score_chunk, score_tt_base, qid);
                tensor_load_wait();
                float local_max = base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int col = qid * 32 + pair * 2;
                    if (col > global_row) {
                        score_chunk[pair].x =
                            base_types::constants<float>::neg_infty();
                    }
                    if (col + 1 > global_row) {
                        score_chunk[pair].y =
                            base_types::constants<float>::neg_infty();
                    }
                    local_max = fmaxf(
                        local_max,
                        fmaxf(score_chunk[pair].x, score_chunk[pair].y));
                }
                block_max[qid] = local_max;
                row_max = fmaxf(row_max, local_max);
            }
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                trace_store(10, now());
            }

            uint32_t packed_scales = 0;
            row_sum_local = 0.0f;
            #pragma unroll
            for (int qid = 0; qid < 4; ++qid) {
                uint8_t p_e8m0 = 0;
                if (isfinite(block_max[qid]) && isfinite(row_max)) {
                    const float block_scale_log2 =
                        FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                        (block_max[qid] - row_max) * g.softmax_scale_log2;
                    p_e8m0 =
                        fp4pv_log2_to_e8m0_floor_finite(block_scale_log2);
                }
                packed_scales |= static_cast<uint32_t>(p_e8m0) << (qid * 8);
                const float p_log2_coeff = (p_e8m0 == 0)
                    ? 0.0f
                    : (FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE -
                       static_cast<float>(static_cast<int>(p_e8m0) - 127));
                const float p_rcp_coeff =
                    fp4pv_exp2f_e8m0(p_e8m0) *
                    FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;
                const float neg_max_scaled =
                    -row_max * g.softmax_scale_log2 + p_log2_coeff;
                const float2 scale_2 = {
                    g.softmax_scale_log2, g.softmax_scale_log2};
                const float2 bias_2 = {neg_max_scaled, neg_max_scaled};
                float2 score_chunk[16];
                fp4pv_load_score_quarter_tmem(score_chunk, score_tt_base, qid);
                tensor_load_wait();
                float2 chunk_sum = {0.0f, 0.0f};
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int col = qid * 32 + pair * 2;
                    if (col > global_row) {
                        score_chunk[pair].x =
                            base_types::constants<float>::neg_infty();
                    }
                    if (col + 1 > global_row) {
                        score_chunk[pair].y =
                            base_types::constants<float>::neg_infty();
                    }
                    score_chunk[pair] =
                        __ffma2_rn(score_chunk[pair], scale_2, bias_2);
                    score_chunk[pair].x =
                        fp4pv_exp2_approx(score_chunk[pair].x);
                    score_chunk[pair].y =
                        fp4pv_exp2_approx(score_chunk[pair].y);
                    chunk_sum = __fadd2_rn(chunk_sum, score_chunk[pair]);
                }
                row_sum_local +=
                    (chunk_sum.x + chunk_sum.y) * p_rcp_coeff;
                uint32_t words[4];
                #pragma unroll
                for (int pack = 0; pack < 4; ++pack) {
                    words[pack] = (p_e8m0 == 0)
                        ? 0u
                        : fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(
                              &score_chunk[pack * 4]);
                }
                fp4pv_store_quantized_scores_group_mxfp4(
                    p_fp4_smem, words, global_row, qid);
            }
            fp4pv_store_mxfp4_scale_word_swizzled(
                p_sc_smem, global_row, packed_scales);
            const bool row_valid =
                row_sum_local > 0.0f && isfinite(row_sum_local);
            float lse_value = base_types::constants<float>::neg_infty();
            if (row_valid) {
                constexpr float LN2 = 0.693147180559945f;
                float log2_row_sum;
                asm("lg2.approx.ftz.f32 %0, %1;"
                    : "=f"(log2_row_sum) : "f"(row_sum_local));
                lse_value =
                    (row_max * g.softmax_scale_log2 + log2_row_sum) * LN2;
            }
            g.lse[h * C::Mb + global_row] = lse_value;
            g.row_sum[h * C::Mb + global_row] = row_sum_local;
            warpgroup::sync(2);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(2);
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                trace_store(11, now());
                arrive(p_ready);
                trace_store(12, now());
            }
        }
        warpgroup::sync(2);
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            arrive(score_consumed);
            trace_store(22, now());
        }
    }

    if (issue_warp && issue_qk) {
        wait(score_consumed, 0);
        if (produce_p) {
            wait(p_ready, 0);
        }
    }
    __syncthreads();

    if (produce_p) {
        constexpr int P_BYTES =
            G::p_fp4_tile::rows * G::p_fp4_tile::cols;
        const uint32_t p_base =
            __cvta_generic_to_shared(&p_fp4_smem.data[0]);
        for (int linear = threadIdx.x; linear < P_BYTES;
             linear += blockDim.x) {
            const int row = linear / G::p_fp4_tile::cols;
            const int col = linear % G::p_fp4_tile::cols;
            uint32_t value;
            asm volatile("ld.shared.u8 %0, [%1];"
                         : "=r"(value)
                         : "r"(G::p_fp4_tile::idx(p_base, {row, col})));
            g.p_debug[(h * C::Mb + row) * G::p_fp4_tile::cols + col] =
                static_cast<uint8_t>(value);
        }
        constexpr int P_SC_BYTES =
            G::p_sc_tile::rows * G::p_sc_tile::cols;
        const uint32_t p_sc_base =
            __cvta_generic_to_shared(&p_sc_smem.data[0]);
        for (int linear = threadIdx.x; linear < P_SC_BYTES;
             linear += blockDim.x) {
            const int row = linear / G::p_sc_tile::cols;
            const int col = linear % G::p_sc_tile::cols;
            uint32_t value;
            asm volatile("ld.shared.u8 %0, [%1];"
                         : "=r"(value)
                         : "r"(G::p_sc_tile::idx(p_sc_base, {row, col})));
            g.p_sc_debug[h * P_SC_BYTES + linear] =
                static_cast<uint8_t>(value);
        }
    }
    __syncthreads();
    if (produce_p) {
        fp4pv_publish_shared_backing_proxy_only();
    }
    __syncthreads();

    if (issue_warp && issue_pv) {
        if (issue_lane) {
            fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm, p_sc_smem);
            fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm, v_sc_smem);
        }
        tensor_load_wait();
        tensor_before_thread_sync();
        if (issue_lane) {
            trace_store(13, now());
            fp4pv_p_stage_ABt_split_k64<C, true>(
                tt_output0, p_fp4_smem, v_fp4_smem,
                p_sc_tm, v_sc_tm, pv_src_done0);
            tensor_commit<1>(pv_done0);
        }
    }
    if (issue_pv) {
        wait(pv_done0, 0);
        tensor_after_thread_sync();
    }
    __syncthreads();
    if (threadIdx.x == 0 && issue_pv) {
        trace_store(14, now());
    }

    if (issue_pv) {
        load_v_half(1);
        __syncthreads();
        fp4pv_publish_shared_backing_proxy_only();
        __syncthreads();
        if (threadIdx.x == 0) {
            trace_store(15, now());
        }
        if (issue_warp) {
            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm, v_sc_smem);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            if (issue_lane) {
                trace_store(16, now());
                fp4pv_p_stage_ABt_split_k64<C, true>(
                    tt_output1, p_fp4_smem, v_fp4_smem,
                    p_sc_tm, v_sc_tm, pv_src_done1);
                tensor_commit<1>(pv_done1);
            }
        }
    }

    if (issue_pv && warpgroup::groupid() == 1) {
        wait(pv_done1, 0);
        tensor_after_thread_sync();
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(17, now());
        }
        const int output_warp = (static_cast<int>(threadIdx.x) - 128) >> 5;
        const int row = output_warp * 32 +
            (static_cast<int>(threadIdx.x) & 31);
        constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
        const float output_scale = normalize_output
            ? (MXFP4_GEMM_SCALE_CORRECTION / row_sum_local)
            : MXFP4_GEMM_SCALE_CORRECTION;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const d_tt_output &tt_output = half == 0 ? tt_output0 : tt_output1;
            #pragma unroll
            for (int col = 0; col < C::Dvo; col += 16) {
                float2 o_reg[8];
                asm volatile(
                    "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y),
                      "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                      "=f"(o_reg[2].x), "=f"(o_reg[2].y),
                      "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                      "=f"(o_reg[4].x), "=f"(o_reg[4].y),
                      "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                      "=f"(o_reg[6].x), "=f"(o_reg[6].y),
                      "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                    : "r"(tt_output.addr +
                          ((output_warp * 32) << 16) + col));
                tensor_load_wait();
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o_reg[i].x *= output_scale;
                    o_reg[i].y *= output_scale;
                    const bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
                    const int out_offset =
                        (((b * g.seqlen + row) * g.heads + h) * 128) +
                        half * C::Dvo + col + i * 2;
                    *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
                }
            }
        }
        warpgroup::sync(2);
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(18, now());
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        trace_store(19, now());
        trace_store(20, now());
    }
    __syncthreads();
}

template <typename C, int MIN_BLOCKS_PER_SM, bool DIAGNOSTICS,
          bool SHARED_ACCUM = false>
__launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_SM)
__global__ void kernel_mxfp4_nb128_dual_dvo64_multistep_shell(
    const __grid_constant__ globals_mxfp4_nb128_dual_dvo64_multistep_shell<C> g
) {
    using G = globals_mxfp4_nb128_dual_dvo64_multistep_shell<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_qk_scale = full_tt_fp8e4m3<16>;
    using d_tt_pv_scale = full_tt_fp8e8m0<16>;
    constexpr int TRACE_FIELDS = 32;
    constexpr int OUTPUT_TILES = 128 / C::Dvo;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT0_BASE = 0;
    constexpr int OUTPUT1_BASE = C::Dvo == 64 ? 64 : 0;
    constexpr int Q_OR_P_SCALE_BASE = 128;
    constexpr int K_OR_V_SCALE_BASE = 144;
    constexpr bool REGISTER_ACCUM = C::REGISTER_ACCUM;
    static_assert(C::NUM_THREADS == 256 && C::CLUSTER_SIZE == 1 &&
                  C::Mb == 128 && C::Nb == 128 && C::Dqk == 192 &&
                  (C::Dvo == 64 || C::Dvo == 128),
                  "multistep shell requires M128/N128/K192/D64x2 or D128");
    static_assert(OUTPUT_TILES * C::Dvo == 128,
                  "multistep output tiles must cover physical DVO128");
    static_assert(!REGISTER_ACCUM || C::Dvo == 128,
                  "register accumulation requires one physical DVO128 tile");
    static_assert(K_OR_V_SCALE_BASE + 16 <= tensor_allocator<2, 1>::cols,
                  "multistep dual-DVO64 shell exceeds 256 TMEM columns");

    auto now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if constexpr (DIAGNOSTICS) {
            if (field >= 0 && field < TRACE_FIELDS) {
                g.trace[blockIdx.x * TRACE_FIELDS + field] = value;
            }
        }
    };

    tensor_allocator<2, 1> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output0 =
        tm_alloc.template allocate<d_tt_output>(OUTPUT0_BASE);
    d_tt_output tt_output1 =
        tm_alloc.template allocate<d_tt_output>(OUTPUT1_BASE);
    d_tt_qk_scale q_sc_tm =
        tm_alloc.template allocate<d_tt_qk_scale>(Q_OR_P_SCALE_BASE);
    d_tt_qk_scale k_sc_tm =
        tm_alloc.template allocate<d_tt_qk_scale>(K_OR_V_SCALE_BASE);
    d_tt_pv_scale p_sc_tm =
        tm_alloc.template allocate<d_tt_pv_scale>(Q_OR_P_SCALE_BASE);
    d_tt_pv_scale v_sc_tm =
        tm_alloc.template allocate<d_tt_pv_scale>(K_OR_V_SCALE_BASE);

    const int task = blockIdx.x;
    const int h = task % g.heads;
    const int query_tile = task / g.heads;
    const int b = 0;
    const int iters = query_tile + 1;
    if (threadIdx.x == 0) {
        trace_store(0, 1);
        trace_store(1, task);
        trace_store(2, kittens::smid());
        trace_store(3, g.stage);
        trace_store(4, h);
        trace_store(5, query_tile);
        trace_store(6, MIN_BLOCKS_PER_SM);
        trace_store(7, now());
        trace_store(8, tm_alloc.addr);
        trace_store(9, iters);
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile &q_smem = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem = al.allocate<typename G::q_sc_tile>();
    typename G::k_tile &k_smem = al.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem = al.allocate<typename G::k_sc_tile>();
    typename G::p_fp4_tile &p_fp4_smem =
        al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem = al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem =
        al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem = al.allocate<typename G::v_sc_tile>();
    typename G::o_backing_tile &o_backing0 =
        al.allocate<typename G::o_backing_tile>();
    typename G::o_backing_aux_tile &o_backing1 =
        al.allocate<typename G::o_backing_aux_tile>();

    __shared__ semaphore qk_done[2];
    __shared__ semaphore pv_src_done0[2];
    __shared__ semaphore pv_src_done1[2];
    __shared__ semaphore pv_done0[2];
    __shared__ semaphore pv_done1[2];
    __shared__ float register_exchange[256];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            init_semaphore(qk_done[slot], 0, 1);
            init_semaphore(pv_src_done0[slot], 0, 1);
            init_semaphore(pv_src_done1[slot], 0, 1);
            init_semaphore(pv_done0[slot], 0, 1);
            init_semaphore(pv_done1[slot], 0, 1);
        }
    }
    __syncthreads();

    auto load_payload = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int BYTES_PER_STORE = 4;
        constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
        constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int linear = threadIdx.x; linear < TOTAL_STORES;
             linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + row * Tile::cols + col);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(Tile::idx(smem_base, {row, col})), "r"(value));
        }
    };
    auto load_scale_record = [&]<typename Tile>(Tile &tile,
                                                const uint8_t *src) {
        constexpr int WORD_BYTES = 4;
        constexpr int WORDS = sizeof(Tile) / WORD_BYTES;
        const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
        for (int word = threadIdx.x; word < WORDS; word += blockDim.x) {
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                src + word * WORD_BYTES);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(smem_base + word * WORD_BYTES), "r"(value));
        }
    };
    auto load_v_tile = [&](int key_tile, int output_tile) {
        using VTile = typename G::v_fp4_tile;
        constexpr int BYTES_PER_STORE = 4;
        constexpr int STORES_PER_ROW = VTile::cols / BYTES_PER_STORE;
        constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
        const int packed_seqlen = g.seqlen / 2;
        const uint32_t smem_base =
            __cvta_generic_to_shared(&v_fp4_smem.data[0]);
        for (int linear = threadIdx.x; linear < TOTAL_STORES;
             linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const int global_row =
                ((h * OUTPUT_TILES + output_tile) * C::Dvo) + row;
            const uint32_t value = *reinterpret_cast<const uint32_t *>(
                g.v_fp4 + global_row * packed_seqlen +
                key_tile * (C::Nb / 2) + col);
            asm volatile("st.shared.b32 [%0], %1;"
                         :: "r"(VTile::idx(smem_base, {row, col})),
                            "r"(value));
        }
        const uint8_t *v_sc_src;
        if constexpr (C::Dvo == 64) {
            v_sc_src = g.v_sc +
                (((h * g.key_tiles + key_tile) * 2 + output_tile) *
                 sizeof(typename G::v_sc_tile));
        } else {
            v_sc_src = g.v_sc +
                ((h * g.key_tiles + key_tile) *
                 sizeof(typename G::v_sc_tile));
        }
        load_scale_record(v_sc_smem, v_sc_src);
    };

    const uint8_t *q_src = g.q_fp4 +
        ((query_tile * g.heads + h) * sizeof(typename G::q_tile));
    const uint8_t *q_sc_src = g.q_sc +
        ((query_tile * g.heads + h) * sizeof(typename G::q_sc_tile));
    load_payload(q_smem, q_src);
    load_scale_record(q_sc_smem, q_sc_src);
    __syncthreads();
    fp4pv_publish_shared_backing_proxy_only();
    __syncthreads();

    const bool issue_warp =
        warpgroup::groupid() == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const bool issue_qk = DIAGNOSTICS ? g.stage >= 1 : true;
    const bool produce_p = DIAGNOSTICS ? g.stage >= 2 : true;
    const bool issue_pv = DIAGNOSTICS ? g.stage >= 3 : true;
    float row_max_state = base_types::constants<float>::neg_infty();
    float row_sum_state = 0.0f;
    float2 register_output[32];
    if constexpr (REGISTER_ACCUM) {
        #pragma unroll
        for (int pair = 0; pair < 32; ++pair) {
            register_output[pair] = {0.0f, 0.0f};
        }
    }

    #pragma unroll 1
    for (int key_tile = 0; key_tile < iters; ++key_tile) {
        const int slot = key_tile & 1;
        const int phase = (key_tile >> 1) & 1;
        const uint8_t *k_src = g.k_fp4 +
            ((h * g.key_tiles + key_tile) * sizeof(typename G::k_tile));
        const uint8_t *k_sc_src = g.k_sc +
            ((h * g.key_tiles + key_tile) * sizeof(typename G::k_sc_tile));
        load_payload(k_smem, k_src);
        load_scale_record(k_sc_smem, k_sc_src);
        if (issue_pv) {
            load_v_tile(key_tile, 0);
        }
        __syncthreads();
        fp4pv_publish_shared_backing_proxy_only();
        __syncthreads();

        if (issue_warp && issue_qk) {
            using T_AB = typename G::q_tile::T;
            using T_SAB = typename d_tt_qk_scale::T;
            using T_D = typename d_tt_score::T;
            constexpr uint32_t IDESC =
                detail::tcgen05::instruction_descriptor<
                    T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
            st_descriptor<typename G::q_tile, 0> q_desc(q_smem);
            st_descriptor<typename G::k_tile, 0> k_desc(k_smem);
            if (issue_lane && key_tile == 0) {
                trace_store(10, now());
            }
            #pragma unroll
            for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                if (issue_lane) {
                    fp4pv_load_q_scale_chunk_ranklocal(
                        q_sc_tm, q_sc_smem, chunk);
                    fp4pv_load_k_scale_chunk_ranklocal(
                        k_sc_tm, k_sc_smem, chunk);
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                if (issue_lane) {
                    if (chunk == 0) {
                        detail::tcgen05::template st_st<
                            T_AB, T_SAB, 0, 1, 16>(
                                tt_score.addr, q_desc.chunk_descriptor(chunk),
                                k_desc.chunk_descriptor(chunk), q_sc_tm.addr,
                                k_sc_tm.addr, IDESC);
                    } else {
                        detail::tcgen05::template st_st<
                            T_AB, T_SAB, 1, 1, 16>(
                                tt_score.addr, q_desc.chunk_descriptor(chunk),
                                k_desc.chunk_descriptor(chunk), q_sc_tm.addr,
                                k_sc_tm.addr, IDESC);
                    }
                }
            }
            if (issue_lane) {
                detail::tcgen05::commit<1>(qk_done[slot]);
            }
        }

        float acc_scale = 1.0f;
        if (warpgroup::groupid() == 1 && issue_qk) {
            wait(qk_done[slot], phase);
            tensor_after_thread_sync();
            const int row = warpgroup::laneid();
            const int query_col = query_tile * C::Mb + row;
            const uint32_t score_tt_base =
                tt_score.addr + ((warpgroup::warpid() * 32) << 16);
            if (key_tile == 0 && warpgroup::warpid() == 0 &&
                warp::elect_leader()) {
                trace_store(11, now());
            }
            if (produce_p) {
                float block_max[4];
                float tile_max = base_types::constants<float>::neg_infty();
                #pragma unroll 1
                for (int qid = 0; qid < 4; ++qid) {
                    float2 score_chunk[16];
                    fp4pv_load_score_quarter_tmem(
                        score_chunk, score_tt_base, qid);
                    tensor_load_wait();
                    float local_max =
                        base_types::constants<float>::neg_infty();
                    #pragma unroll
                    for (int pair = 0; pair < 16; ++pair) {
                        const int key_col = key_tile * C::Nb +
                            qid * 32 + pair * 2;
                        if (key_col > query_col) {
                            score_chunk[pair].x =
                                base_types::constants<float>::neg_infty();
                        }
                        if (key_col + 1 > query_col) {
                            score_chunk[pair].y =
                                base_types::constants<float>::neg_infty();
                        }
                        local_max = fmaxf(
                            local_max,
                            fmaxf(score_chunk[pair].x,
                                  score_chunk[pair].y));
                    }
                    block_max[qid] = local_max;
                    tile_max = fmaxf(tile_max, local_max);
                }
                const float old_max = row_max_state;
                row_max_state = fmaxf(row_max_state, tile_max);
                if (key_tile > 0) {
                    acc_scale = softmax_rescale_factor_strict(
                        row_max_state, old_max, g.softmax_scale_log2);
                }
                if constexpr (DIAGNOSTICS) {
                    if (g.capture_p != 0) {
                        const int scale_record =
                            ((query_tile * g.heads + h) * g.key_tiles +
                             key_tile);
                        g.acc_scale_debug[scale_record * C::Mb + row] =
                            acc_scale;
                    }
                }

                uint32_t packed_scales = 0;
                float tile_sum = 0.0f;
                #pragma unroll 1
                for (int qid = 0; qid < 4; ++qid) {
                    uint8_t p_e8m0 = 0;
                    if (isfinite(block_max[qid]) &&
                        isfinite(row_max_state)) {
                        const float block_scale_log2 =
                            FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                            (block_max[qid] - row_max_state) *
                                g.softmax_scale_log2;
                        p_e8m0 = fp4pv_log2_to_e8m0_floor_finite(
                            block_scale_log2);
                    }
                    packed_scales |=
                        static_cast<uint32_t>(p_e8m0) << (qid * 8);
                    const float p_log2_coeff = (p_e8m0 == 0)
                        ? 0.0f
                        : (FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE -
                           static_cast<float>(
                               static_cast<int>(p_e8m0) - 127));
                    const float p_rcp_coeff =
                        fp4pv_exp2f_e8m0(p_e8m0) *
                        FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;
                    const float bias =
                        -row_max_state * g.softmax_scale_log2 +
                        p_log2_coeff;
                    const float2 score_scale = {
                        g.softmax_scale_log2, g.softmax_scale_log2};
                    const float2 bias2 = {bias, bias};
                    float2 score_chunk[16];
                    fp4pv_load_score_quarter_tmem(
                        score_chunk, score_tt_base, qid);
                    tensor_load_wait();
                    float2 chunk_sum = {0.0f, 0.0f};
                    #pragma unroll
                    for (int pair = 0; pair < 16; ++pair) {
                        const int key_col = key_tile * C::Nb +
                            qid * 32 + pair * 2;
                        if (key_col > query_col) {
                            score_chunk[pair].x =
                                base_types::constants<float>::neg_infty();
                        }
                        if (key_col + 1 > query_col) {
                            score_chunk[pair].y =
                                base_types::constants<float>::neg_infty();
                        }
                        score_chunk[pair] = __ffma2_rn(
                            score_chunk[pair], score_scale, bias2);
                        score_chunk[pair].x =
                            fp4pv_exp2_approx(score_chunk[pair].x);
                        score_chunk[pair].y =
                            fp4pv_exp2_approx(score_chunk[pair].y);
                        chunk_sum = __fadd2_rn(
                            chunk_sum, score_chunk[pair]);
                    }
                    tile_sum +=
                        (chunk_sum.x + chunk_sum.y) * p_rcp_coeff;
                    uint32_t words[4];
                    #pragma unroll
                    for (int pack = 0; pack < 4; ++pack) {
                        words[pack] = (p_e8m0 == 0)
                            ? 0u
                            : fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(
                                  &score_chunk[pack * 4]);
                    }
                    fp4pv_store_quantized_scores_group_mxfp4(
                        p_fp4_smem, words, row, qid);
                }
                fp4pv_store_mxfp4_scale_word_swizzled(
                    p_sc_smem, row, packed_scales);
                row_sum_state = row_sum_state * acc_scale + tile_sum;
                if constexpr (REGISTER_ACCUM) {
                    register_exchange[row] = acc_scale;
                }
                if (key_tile + 1 == iters) {
                    const bool valid = row_sum_state > 0.0f &&
                        isfinite(row_sum_state);
                    float lse_value =
                        base_types::constants<float>::neg_infty();
                    if (valid) {
                        constexpr float LN2 = 0.693147180559945f;
                        float log2_row_sum;
                        asm("lg2.approx.ftz.f32 %0, %1;"
                            : "=f"(log2_row_sum) : "f"(row_sum_state));
                        lse_value =
                            (row_max_state * g.softmax_scale_log2 +
                             log2_row_sum) * LN2;
                    }
                    const int row_offset =
                        (query_tile * g.heads + h) * C::Mb + row;
                    g.lse[row_offset] = lse_value;
                    g.row_sum[row_offset] = row_sum_state;
                }
                warpgroup::sync(2);
                fp4pv_publish_shared_backing_proxy_only();
                warpgroup::sync(2);
                if (key_tile + 1 == iters &&
                    warpgroup::warpid() == 0 && warp::elect_leader()) {
                    trace_store(12, now());
                }
            }
        }
        __syncthreads();

        if constexpr (DIAGNOSTICS) {
            if (produce_p && g.capture_p != 0) {
                constexpr int P_BYTES =
                    G::p_fp4_tile::rows * G::p_fp4_tile::cols;
                const uint32_t p_base =
                    __cvta_generic_to_shared(&p_fp4_smem.data[0]);
                const int p_record =
                    ((query_tile * g.heads + h) * g.key_tiles + key_tile);
                for (int linear = threadIdx.x; linear < P_BYTES;
                     linear += blockDim.x) {
                    const int row = linear / G::p_fp4_tile::cols;
                    const int col = linear % G::p_fp4_tile::cols;
                    uint32_t value;
                    asm volatile("ld.shared.u8 %0, [%1];"
                                 : "=r"(value)
                                 : "r"(G::p_fp4_tile::idx(
                                       p_base, {row, col})));
                    g.p_debug[p_record * P_BYTES + linear] =
                        static_cast<uint8_t>(value);
                }
                constexpr int P_SC_BYTES =
                    G::p_sc_tile::rows * G::p_sc_tile::cols;
                const uint32_t p_sc_base =
                    __cvta_generic_to_shared(&p_sc_smem.data[0]);
                for (int linear = threadIdx.x; linear < P_SC_BYTES;
                     linear += blockDim.x) {
                    uint32_t value;
                    asm volatile("ld.shared.u8 %0, [%1];"
                                 : "=r"(value)
                                 : "r"(p_sc_base + linear));
                    g.p_sc_debug[p_record * P_SC_BYTES + linear] =
                        static_cast<uint8_t>(value);
                }
            }
        }
        __syncthreads();

        if constexpr (!SHARED_ACCUM && !REGISTER_ACCUM) {
            if (issue_pv && key_tile > 0 &&
                warpgroup::groupid() == 1) {
                load_output_backing<C>(tt_output0, o_backing0);
                rescale_tt_output_inplace_direct_streaming<C>(
                    tt_output0, acc_scale, true);
                if constexpr (C::Dvo == 64) {
                    load_output_backing<C>(tt_output1, o_backing1);
                    rescale_tt_output_inplace_direct_streaming<C>(
                        tt_output1, acc_scale, true);
                }
                if (key_tile + 1 == iters &&
                    warpgroup::warpid() == 0 && warp::elect_leader()) {
                    trace_store(13, now());
                }
            }
        }
        __syncthreads();
        if (produce_p) {
            fp4pv_publish_shared_backing_proxy_only();
        }
        __syncthreads();

        if (issue_warp && issue_pv) {
            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(
                    p_sc_tm, p_sc_smem);
                fp4pv_load_mxnv_scale_async_ranklocal(
                    v_sc_tm, v_sc_smem);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            if (issue_lane) {
                if constexpr (SHARED_ACCUM || REGISTER_ACCUM) {
                    fp4pv_p_stage_ABt_split_k64<C, true>(
                        tt_output0, p_fp4_smem, v_fp4_smem,
                        p_sc_tm, v_sc_tm, pv_src_done0[slot]);
                } else {
                    if (key_tile == 0) {
                        fp4pv_p_stage_ABt_split_k64<C, true>(
                            tt_output0, p_fp4_smem, v_fp4_smem,
                            p_sc_tm, v_sc_tm, pv_src_done0[slot]);
                    } else {
                        fp4pv_p_stage_ABt_split_k64<C, false>(
                            tt_output0, p_fp4_smem, v_fp4_smem,
                            p_sc_tm, v_sc_tm, pv_src_done0[slot]);
                    }
                }
                tensor_commit<1>(pv_done0[slot]);
            }
        }
        if constexpr (REGISTER_ACCUM) {
            if (issue_pv) {
                wait(pv_done0[slot], phase);
                tensor_after_thread_sync();
                const int output_col_base =
                    warpgroup::groupid() * 64;
                const float correction = register_exchange[
                    warpgroup::laneid()];
                #pragma unroll
                for (int col = 0; col < 64; col += 16) {
                    float2 out[8];
                    asm volatile(
                        "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                        : "=f"(out[0].x), "=f"(out[0].y),
                          "=f"(out[1].x), "=f"(out[1].y),
                          "=f"(out[2].x), "=f"(out[2].y),
                          "=f"(out[3].x), "=f"(out[3].y),
                          "=f"(out[4].x), "=f"(out[4].y),
                          "=f"(out[5].x), "=f"(out[5].y),
                          "=f"(out[6].x), "=f"(out[6].y),
                          "=f"(out[7].x), "=f"(out[7].y)
                        : "r"(tt_output0.addr +
                              ((warpgroup::warpid() * 32) << 16) +
                              output_col_base + col));
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    #pragma unroll
                    for (int pair = 0; pair < 8; ++pair) {
                        const int output_pair = col / 2 + pair;
                        if (key_tile == 0) {
                            register_output[output_pair] = out[pair];
                        } else {
                            register_output[output_pair].x = __fadd_rn(
                                __fmul_rn(register_output[output_pair].x,
                                          correction),
                                out[pair].x);
                            register_output[output_pair].y = __fadd_rn(
                                __fmul_rn(register_output[output_pair].y,
                                          correction),
                                out[pair].y);
                        }
                    }
                }
            }
        } else if (issue_pv && warpgroup::groupid() == 1) {
            wait(pv_done0[slot], phase);
            tensor_after_thread_sync();
            if constexpr (C::Dvo == 128) {
                if constexpr (SHARED_ACCUM) {
                    accumulate_output_backing_direct<C>(
                        o_backing0, tt_output0, acc_scale, key_tile > 0);
                } else if (key_tile + 1 < iters) {
                    spill_output_to_backing<C>(o_backing0, tt_output0);
                    publish_shared_backing();
                }
            }
        }
        __syncthreads();

        if constexpr (C::Dvo == 64) {
            if (issue_pv) {
                load_v_tile(key_tile, 1);
            }
            __syncthreads();
            if (issue_pv) {
                fp4pv_publish_shared_backing_proxy_only();
            }
            __syncthreads();
            if (issue_warp && issue_pv) {
                if (issue_lane) {
                    fp4pv_load_mxnv_scale_async_ranklocal(
                        v_sc_tm, v_sc_smem);
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                if (issue_lane) {
                    if constexpr (SHARED_ACCUM) {
                        fp4pv_p_stage_ABt_split_k64<C, true>(
                            tt_output1, p_fp4_smem, v_fp4_smem,
                            p_sc_tm, v_sc_tm, pv_src_done1[slot]);
                    } else {
                        if (key_tile == 0) {
                            fp4pv_p_stage_ABt_split_k64<C, true>(
                                tt_output1, p_fp4_smem, v_fp4_smem,
                                p_sc_tm, v_sc_tm, pv_src_done1[slot]);
                        } else {
                            fp4pv_p_stage_ABt_split_k64<C, false>(
                                tt_output1, p_fp4_smem, v_fp4_smem,
                                p_sc_tm, v_sc_tm, pv_src_done1[slot]);
                        }
                    }
                    tensor_commit<1>(pv_done1[slot]);
                }
            }
            if (issue_pv && warpgroup::groupid() == 1) {
                wait(pv_done1[slot], phase);
                tensor_after_thread_sync();
                if constexpr (SHARED_ACCUM) {
                    accumulate_output_backing_direct<C>(
                        o_backing0, tt_output0, acc_scale, key_tile > 0);
                    accumulate_output_backing_direct<C>(
                        o_backing1, tt_output1, acc_scale, key_tile > 0);
                } else if (key_tile + 1 < iters) {
                    spill_output_to_backing<C>(o_backing0, tt_output0);
                    spill_output_to_backing<C>(o_backing1, tt_output1);
                    publish_shared_backing();
                }
            }
            __syncthreads();
        }
        if (key_tile + 1 == iters && threadIdx.x == 0) {
            trace_store(14, now());
        }
    }

    if constexpr (REGISTER_ACCUM) {
        constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
        if (issue_pv && warpgroup::groupid() == 1) {
            const bool valid =
                row_sum_state > 0.0f && isfinite(row_sum_state);
            register_exchange[128 + warpgroup::laneid()] = valid
                ? MXFP4_GEMM_SCALE_CORRECTION / row_sum_state
                : 0.0f;
        }
        __syncthreads();
        if (issue_pv) {
            const int row = warpgroup::laneid();
            const int output_tile = warpgroup::groupid();
            const float output_scale = register_exchange[128 + row];
            #pragma unroll
            for (int pair = 0; pair < 32; ++pair) {
                const bf16_2 packed = __float22bfloat162_rn(
                    __fmul2_rn(
                        register_output[pair],
                        float2{output_scale, output_scale}));
                const int out_offset =
                    ((((b * g.seqlen + query_tile * C::Mb + row) *
                       g.heads + h) * 128) +
                     output_tile * 64 + pair * 2);
                *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
            }
            warpgroup::sync(warpgroup::groupid() + 1);
            if (warpgroup::groupid() == 1 &&
                warpgroup::warpid() == 0 && warp::elect_leader()) {
                trace_store(15, now());
            }
        }
    } else if (issue_pv && warpgroup::groupid() == 1) {
        const int output_warp =
            (static_cast<int>(threadIdx.x) - 128) >> 5;
        const int row = output_warp * 32 +
            (static_cast<int>(threadIdx.x) & 31);
        constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
        const bool valid = row_sum_state > 0.0f && isfinite(row_sum_state);
        const float output_scale = valid
            ? MXFP4_GEMM_SCALE_CORRECTION / row_sum_state
            : 0.0f;
        if constexpr (SHARED_ACCUM) {
            auto store_backing = [&]<typename Backing>(
                Backing &backing, int output_tile) {
                #pragma unroll
                for (int col = 0; col < C::Dvo; col += 16) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float2 out = {
                            backing[{row, col + i * 2}],
                            backing[{row, col + i * 2 + 1}]};
                        const bf16_2 packed = __float22bfloat162_rn(
                            __fmul2_rn(
                                out, float2{output_scale, output_scale}));
                        const int out_offset =
                            ((((b * g.seqlen +
                                query_tile * C::Mb + row) *
                               g.heads + h) * 128) +
                             output_tile * C::Dvo + col + i * 2);
                        *reinterpret_cast<bf16_2 *>(
                            &g.o[out_offset]) = packed;
                    }
                }
            };
            store_backing(o_backing0, 0);
            if constexpr (C::Dvo == 64) {
                store_backing(o_backing1, 1);
            }
        } else {
            #pragma unroll
            for (int output_tile = 0;
                 output_tile < OUTPUT_TILES; ++output_tile) {
                const d_tt_output &tt_output =
                    output_tile == 0 ? tt_output0 : tt_output1;
                #pragma unroll
                for (int col = 0; col < C::Dvo; col += 16) {
                    float2 out[8];
                    asm volatile(
                        "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                        : "=f"(out[0].x), "=f"(out[0].y),
                          "=f"(out[1].x), "=f"(out[1].y),
                          "=f"(out[2].x), "=f"(out[2].y),
                          "=f"(out[3].x), "=f"(out[3].y),
                          "=f"(out[4].x), "=f"(out[4].y),
                          "=f"(out[5].x), "=f"(out[5].y),
                          "=f"(out[6].x), "=f"(out[6].y),
                          "=f"(out[7].x), "=f"(out[7].y)
                        : "r"(tt_output.addr +
                              ((output_warp * 32) << 16) + col));
                    tensor_load_wait();
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const bf16_2 packed = __float22bfloat162_rn(
                            __fmul2_rn(
                                out[i],
                                float2{output_scale, output_scale}));
                        const int out_offset =
                            ((((b * g.seqlen +
                                query_tile * C::Mb + row) *
                               g.heads + h) * 128) +
                             output_tile * C::Dvo + col + i * 2);
                        *reinterpret_cast<bf16_2 *>(
                            &g.o[out_offset]) = packed;
                    }
                }
            }
        }
        warpgroup::sync(2);
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(15, now());
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        trace_store(16, now());
        trace_store(17, now());
    }
    __syncthreads();
}
#endif

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_v_tma_tile_check_debug(
    const __grid_constant__ globals_mxfp4_v_tma_tile_check_debug<C> g
) {
    using G = globals_mxfp4_v_tma_tile_check_debug<C>;
    using v_tile = typename G::v_fp4_tile;
    static_assert(C::CLUSTER_SIZE == 1, "V TMA tile check is a cluster1 debug kernel");
    static_assert(C::Dvo == 128 && C::Nb == 128, "V TMA tile check currently targets Dvo128/Nb128");

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    v_tile &manual_smem = al.allocate<v_tile>();
    v_tile &tma_smem = al.allocate<v_tile>();

    __shared__ semaphore tma_arrived;
    if (threadIdx.x == 0) {
        g.v_fp4_tma.template prefetch_tma<v_tile, dim::ROW>();
        init_semaphore(tma_arrived, 0, 1);
        for (int i = 0; i < 12; ++i) {
            g.debug[i] = 0;
        }
        g.debug[1] = -1;
        g.debug[2] = -1;
        g.debug[5] = C::Dvo * (C::Nb / 2);
        g.debug[6] = g.h;
        g.debug[7] = g.k_iter;
        g.debug[8] = g.mode;
        g.debug[11] = 1;
    }
    __syncthreads();
    if (g.mode == 0 || g.mode == 2 || g.mode == 3) {
        for (int delay = threadIdx.x; delay < 8192; delay += blockDim.x) {
            asm volatile("" ::: "memory");
        }
    }
    __syncthreads();

    const int packed_seqlen = g.seqlen / 2;
    const int col_base = g.k_iter * (C::Nb / 2);
    uint32_t manual_base = __cvta_generic_to_shared(&manual_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = v_tile::rows * STORES_PER_ROW;
    if (g.mode == 0 || g.mode == 1) {
        for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.v_fp4[((((g.b * g.heads + g.h) * C::Dvo) + row) * packed_seqlen) + col_base + col]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(v_tile::idx(manual_base, {row, col})),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        g.debug[11] = 2;
    }
    __syncthreads();

    if (threadIdx.x == 0 && (g.mode == 0 || g.mode == 2 || g.mode == 3)) {
        fp4pv_expect_bytes<C>(tma_arrived, sizeof(v_tile));
        const coord<v_tile> v_coord = {g.b, g.h, 0, g.k_iter};
        tma::load_async<dim::ROW, cache_policy::NORMAL>(tma_smem, g.v_fp4_tma, v_coord, tma_arrived);
        g.debug[11] = 3;
    }
    __syncthreads();
    if (threadIdx.x == 0 && (g.mode == 0 || g.mode == 2 || g.mode == 3)) {
        wait(tma_arrived, 0);
        fp4pv_publish_shared_backing_proxy_only();
        g.debug[11] = 4;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (g.mode == 3) {
            g.debug[11] = 5;
            return;
        }
        uint32_t tma_base = __cvta_generic_to_shared(&tma_smem.data[0]);
        int mismatches = 0;
        int first_row = -1;
        int first_col = -1;
        int first_manual = 0;
        int first_tma = 0;
        uint32_t manual_checksum = 0;
        uint32_t tma_checksum = 0;
        for (int row = 0; row < C::Dvo; ++row) {
            for (int col = 0; col < C::Nb / 2; ++col) {
                uint32_t manual_byte = 0;
                uint32_t tma_byte = 0;
                if (g.mode == 0 || g.mode == 1) {
                    asm volatile("ld.shared.u8 %0, [%1];"
                                 : "=r"(manual_byte)
                                 : "r"(v_tile::idx(manual_base, {row, col})));
                    manual_checksum = manual_checksum * 16777619u + manual_byte + 1u;
                }
                if (g.mode == 0 || g.mode == 2) {
                    asm volatile("ld.shared.u8 %0, [%1];"
                                 : "=r"(tma_byte)
                                 : "r"(v_tile::idx(tma_base, {row, col})));
                    tma_checksum = tma_checksum * 16777619u + tma_byte + 1u;
                }
                if (g.mode == 0 && manual_byte != tma_byte) {
                    if (mismatches == 0) {
                        first_row = row;
                        first_col = col;
                        first_manual = (int)manual_byte;
                        first_tma = (int)tma_byte;
                    }
                    ++mismatches;
                }
            }
        }
        g.debug[0] = mismatches;
        g.debug[1] = first_row;
        g.debug[2] = first_col;
        g.debug[3] = first_manual;
        g.debug[4] = first_tma;
        g.debug[5] = C::Dvo * (C::Nb / 2);
        g.debug[6] = g.h;
        g.debug[7] = g.k_iter;
        g.debug[8] = g.mode;
        g.debug[9] = (int)manual_checksum;
        g.debug[10] = (int)tma_checksum;
        g.debug[11] = 6;
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_k_tma_multicast_debug(
    const __grid_constant__ globals_mxfp4_k_tma_multicast_debug<C> g
) {
    using G = globals_mxfp4_k_tma_multicast_debug<C>;
    using k_tile = typename G::k_tile;
    using k_sc_tile = typename G::k_sc_tile;
    static_assert(C::CLUSTER_SIZE == 2, "K TMA multicast debug is a cluster2 diagnostic");
    static_assert(C::Dqk == 192 && C::Nb == 128, "K TMA multicast debug currently targets Dqk192/Nb128");

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    k_tile &manual_smem = al.allocate<k_tile>();
    k_tile &tma_smem = al.allocate<k_tile>();
    k_sc_tile &scale_manual_smem = al.allocate<k_sc_tile>();
    k_sc_tile &scale_tma_smem = al.allocate<k_sc_tile>();

    __shared__ semaphore tma_arrived;
    __shared__ semaphore scale_arrived;
    __shared__ semaphore owner_done;
    __shared__ int wait_failed;
    const int cta_rank = cluster_ctarank();
    constexpr int DEBUG_STRIDE = 32;
    int32_t *dbg = g.debug + cta_rank * DEBUG_STRIDE;
    constexpr int SPIN_LIMIT = 10000000;

    auto set_progress = [&](int progress) {
        if (threadIdx.x == 0) {
            dbg[1] = progress;
        }
    };
    auto wait_or_timeout = [&](semaphore &sem, int phase, int fail_code) -> bool {
        #pragma unroll 1
        for (int spin = 0; spin < SPIN_LIMIT; ++spin) {
            if (try_wait(sem, phase)) {
                return true;
            }
        }
        if (threadIdx.x == 0) {
            dbg[16] = fail_code;
            dbg[17] = phase;
        }
        return false;
    };

    if (threadIdx.x == 0) {
        g.k.template prefetch_tma<k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<k_sc_tile, dim::ROW>();
        init_semaphore(tma_arrived, 0, 1);
        init_semaphore(scale_arrived, 0, 1);
        init_semaphore(owner_done, 0, 1);
        wait_failed = 0;
        #pragma unroll
        for (int i = 0; i < DEBUG_STRIDE; ++i) {
            dbg[i] = 0;
        }
        dbg[0] = cta_rank;
        dbg[2] = g.mode;
        dbg[3] = 0;
        dbg[4] = 0b11;
        dbg[5] = (int)sizeof(k_tile);
        dbg[6] = g.b;
        dbg[7] = g.h;
        dbg[8] = g.k_iter;
        dbg[18] = g.check_scale;
        dbg[25] = (int)sizeof(k_sc_tile);
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    set_progress(1);

    const int packed_cols = C::Dqk / 2;
    const int row_base = g.k_iter * k_tile::rows;
    uint32_t manual_base = __cvta_generic_to_shared(&manual_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = k_tile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = k_tile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.k_fp4[((((g.b * g.heads + g.h) * g.seqlen) + row_base + row) * packed_cols) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(k_tile::idx(manual_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }

    if (g.check_scale) {
        const int k_sc_depth = g.seqlen / 64;
        const int scale_base_offset =
            (((g.b * k_sc_depth + g.k_iter) * (g.heads * C::QK_SCALE_CHUNKS) +
              g.h * C::QK_SCALE_CHUNKS) * 512);
        uint32_t scale_manual_base = __cvta_generic_to_shared(&scale_manual_smem.data[0]);
        for (int byte = threadIdx.x; byte < (int)sizeof(k_sc_tile); byte += blockDim.x) {
            const int row = byte / 512;
            const int row_byte = byte - row * 512;
            const int col = row_byte / 2;
            const int byte_in_half = row_byte & 1;
            const uint32_t value = (uint32_t)g.k_sc_raw[scale_base_offset + byte];
            asm volatile("st.shared.u8 [%0], %1;"
                         :
                         : "r"(k_sc_tile::idx(scale_manual_base, {row, col}) + byte_in_half),
                           "r"(value));
        }
    }
    __syncthreads();
    set_progress(2);

    const coord<k_tile> k_coord = {g.b, g.h, g.k_iter, 0};
    const coord<k_sc_tile> k_sc_coord = {g.b, g.k_iter, g.h, 0};
    if (g.mode == 2) {
        if (threadIdx.x == 0) {
            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                tma_smem, g.k, k_coord, tma_arrived, (uint16_t)(1 << cta_rank));
            dbg[4] = (1 << cta_rank);
            dbg[1] = 3;
            tma::cluster::expect_bytes(tma_arrived, sizeof(k_tile));
            if (g.check_scale) {
                tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                    scale_tma_smem, g.k_sc, k_sc_coord, scale_arrived, (uint16_t)(1 << cta_rank));
                tma::cluster::expect_bytes(scale_arrived, sizeof(k_sc_tile));
            }
            dbg[1] = 4;
        }
    } else if (g.mode == 1) {
        if (threadIdx.x == 0 && cta_rank == 0) {
            tma::cluster::expect_bytes(tma_arrived, sizeof(k_tile));
            if (g.check_scale) {
                tma::cluster::expect_bytes(scale_arrived, sizeof(k_sc_tile));
            }
            dbg[1] = 4;
            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                tma_smem, g.k, k_coord, tma_arrived, (uint16_t)0b11, 0);
            if (g.check_scale) {
                tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                    scale_tma_smem, g.k_sc, k_sc_coord, scale_arrived, (uint16_t)0b11, 0);
            }
            dbg[1] = 3;
        } else if (threadIdx.x == 0) {
            dbg[1] = 3;
        }
    } else {
        if (threadIdx.x == 0 && cta_rank == 0) {
            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                tma_smem, g.k, k_coord, tma_arrived, (uint16_t)0b11);
            if (g.check_scale) {
                tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                    scale_tma_smem, g.k_sc, k_sc_coord, scale_arrived, (uint16_t)0b11);
            }
            dbg[1] = 3;
        } else if (threadIdx.x == 0) {
            dbg[1] = 3;
        }
        if (threadIdx.x == 0) {
            tma::cluster::expect_bytes(tma_arrived, sizeof(k_tile));
            if (g.check_scale) {
                tma::cluster::expect_bytes(scale_arrived, sizeof(k_sc_tile));
            }
            dbg[1] = 4;
        }
    }
    __syncthreads();

    bool wait_ok = true;
    if (threadIdx.x == 0) {
        if (g.mode == 1) {
            if (cta_rank == 0) {
                wait_ok = wait_or_timeout(tma_arrived, 0, 101);
                if (wait_ok && g.check_scale) {
                    wait_ok = wait_or_timeout(scale_arrived, 0, 121);
                }
                if (wait_ok) {
                    tma::cluster::arrive(owner_done, 1);
                }
            } else {
                wait_ok = wait_or_timeout(owner_done, 0, 102);
            }
        } else {
            wait_ok = wait_or_timeout(tma_arrived, 0, 100 + cta_rank);
            if (wait_ok && g.check_scale) {
                wait_ok = wait_or_timeout(scale_arrived, 0, 120 + cta_rank);
            }
        }
        if (wait_ok) {
            fp4pv_publish_shared_backing_proxy_only();
            dbg[1] = 5;
        } else {
            wait_failed = 1;
        }
    }
    __syncthreads();
    if (wait_failed) {
        return;
    }

    if (threadIdx.x == 0) {
        uint32_t tma_base = __cvta_generic_to_shared(&tma_smem.data[0]);
        int mismatches = 0;
        int first_row = -1;
        int first_col = -1;
        int first_manual = 0;
        int first_tma = 0;
        uint32_t manual_checksum = 0;
        uint32_t tma_checksum = 0;
        #pragma unroll 1
        for (int row = 0; row < k_tile::rows; ++row) {
            #pragma unroll 1
            for (int col = 0; col < k_tile::cols; ++col) {
                uint32_t manual_byte = 0;
                uint32_t tma_byte = 0;
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(manual_byte)
                             : "r"(k_tile::idx(manual_base, {row, col})));
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(tma_byte)
                             : "r"(k_tile::idx(tma_base, {row, col})));
                manual_checksum = manual_checksum * 16777619u + manual_byte + 1u;
                tma_checksum = tma_checksum * 16777619u + tma_byte + 1u;
                if (manual_byte != tma_byte) {
                    if (mismatches == 0) {
                        first_row = row;
                        first_col = col;
                        first_manual = (int)manual_byte;
                        first_tma = (int)tma_byte;
                    }
                    ++mismatches;
                }
            }
        }
        dbg[1] = 6;
        dbg[9] = mismatches;
        dbg[10] = first_row;
        dbg[11] = first_col;
        dbg[12] = first_manual;
        dbg[13] = first_tma;
        dbg[14] = (int)manual_checksum;
        dbg[15] = (int)tma_checksum;
        if (g.check_scale) {
            uint32_t scale_manual_base = __cvta_generic_to_shared(&scale_manual_smem.data[0]);
            uint32_t scale_tma_base = __cvta_generic_to_shared(&scale_tma_smem.data[0]);
            int scale_mismatches = 0;
            int scale_first = -1;
            int scale_first_manual = 0;
            int scale_first_tma = 0;
            uint32_t scale_manual_checksum = 0;
            uint32_t scale_tma_checksum = 0;
            #pragma unroll 1
            for (int byte = 0; byte < (int)sizeof(k_sc_tile); ++byte) {
                const int row = byte / 512;
                const int row_byte = byte - row * 512;
                const int col = row_byte / 2;
                const int byte_in_half = row_byte & 1;
                uint32_t manual_byte = 0;
                uint32_t tma_byte = 0;
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(manual_byte)
                             : "r"(k_sc_tile::idx(scale_manual_base, {row, col}) + byte_in_half));
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(tma_byte)
                             : "r"(k_sc_tile::idx(scale_tma_base, {row, col}) + byte_in_half));
                scale_manual_checksum = scale_manual_checksum * 16777619u + manual_byte + 1u;
                scale_tma_checksum = scale_tma_checksum * 16777619u + tma_byte + 1u;
                if (manual_byte != tma_byte) {
                    if (scale_mismatches == 0) {
                        scale_first = byte;
                        scale_first_manual = (int)manual_byte;
                        scale_first_tma = (int)tma_byte;
                    }
                    ++scale_mismatches;
                }
            }
            dbg[19] = scale_mismatches;
            dbg[20] = scale_first;
            dbg[21] = scale_first_manual;
            dbg[22] = scale_first_tma;
            dbg[23] = (int)scale_manual_checksum;
            dbg[24] = (int)scale_tma_checksum;
        }
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_kv_tma_multicast_debug(
    const __grid_constant__ globals_mxfp4_kv_tma_multicast_debug<C> g
) {
    using G = globals_mxfp4_kv_tma_multicast_debug<C>;
    using k_tile = typename G::k_tile;
    using k_sc_tile = typename G::k_sc_tile;
    using v_tile = typename G::v_fp4_tile;
    using v_sc_tile = typename G::v_sc_tile;
    static_assert(C::CLUSTER_SIZE == 2, "K/V TMA multicast debug is a cluster2 diagnostic");
    static_assert(C::Dqk == 192 && C::Dvo == 128 && C::Nb == 128,
                  "K/V TMA multicast debug targets Dqk192/Dvo128/Nb128");

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    k_tile &k_smem = al.allocate<k_tile>();
    k_sc_tile &k_sc_smem = al.allocate<k_sc_tile>();
    v_tile &v_smem = al.allocate<v_tile>();
    v_sc_tile &v_sc_smem = al.allocate<v_sc_tile>();

    __shared__ semaphore k_arrived;
    __shared__ semaphore k_sc_arrived;
    __shared__ semaphore v_arrived;
    __shared__ semaphore v_sc_arrived;
    __shared__ int wait_failed;
    const int cta_rank = cluster_ctarank();
    constexpr int DEBUG_STRIDE = 64;
    int32_t *dbg = g.debug + cta_rank * DEBUG_STRIDE;
    constexpr int SPIN_LIMIT = 10000000;

    auto set_progress = [&](int progress) {
        if (threadIdx.x == 0) {
            dbg[1] = progress;
        }
    };
    auto wait_or_timeout = [&](semaphore &sem, int phase, int fail_code) -> bool {
        #pragma unroll 1
        for (int spin = 0; spin < SPIN_LIMIT; ++spin) {
            if (try_wait(sem, phase)) {
                return true;
            }
        }
        if (threadIdx.x == 0) {
            dbg[16] = fail_code;
            dbg[17] = phase;
        }
        return false;
    };
    auto expect_k = [&]() {
        tma::cluster::expect_bytes(k_arrived, sizeof(k_tile));
        if (g.check_scale) {
            tma::cluster::expect_bytes(k_sc_arrived, sizeof(k_sc_tile));
        }
    };
    auto expect_v = [&]() {
        tma::cluster::expect_bytes(v_arrived, sizeof(v_tile));
        if (g.check_scale) {
            tma::cluster::expect_bytes(v_sc_arrived, sizeof(v_sc_tile));
        }
    };
    auto issue_k = [&](uint16_t mask) {
        const coord<k_tile> k_coord = {g.b, g.h, g.k_iter, 0};
        const coord<k_sc_tile> k_sc_coord = {g.b, g.k_iter, g.h, 0};
        tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
            k_smem, g.k, k_coord, k_arrived, mask);
        if (g.check_scale) {
            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                k_sc_smem, g.k_sc, k_sc_coord, k_sc_arrived, mask);
        }
    };
    auto issue_v = [&](uint16_t mask) {
        const coord<v_tile> v_coord = {g.b, g.h, 0, g.k_iter};
        tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
            v_smem, g.v_fp4_tma, v_coord, v_arrived, mask);
        if (g.check_scale) {
            const int scale_base_offset =
                (((((g.b * g.heads + g.h) * g.v_scale_rows + 0) *
                   (g.seqlen / 128)) + g.k_iter) * 32 * 16);
            tma::cluster::load_async(
                reinterpret_cast<void *>(&v_sc_smem.data[0]),
                const_cast<uint8_t *>(&g.v_sc_prepared[scale_base_offset]),
                sizeof(v_sc_tile),
                v_sc_arrived,
                mask);
        }
    };
    auto wait_k = [&]() -> bool {
        if (!wait_or_timeout(k_arrived, 0, 100 + cta_rank)) {
            return false;
        }
        dbg[20] = 1;
        if (g.check_scale && !wait_or_timeout(k_sc_arrived, 0, 120 + cta_rank)) {
            return false;
        }
        dbg[21] = 1;
        return true;
    };
    auto wait_v = [&]() -> bool {
        if (!wait_or_timeout(v_arrived, 0, 140 + cta_rank)) {
            return false;
        }
        dbg[22] = 1;
        if (g.check_scale && !wait_or_timeout(v_sc_arrived, 0, 160 + cta_rank)) {
            return false;
        }
        dbg[23] = 1;
        return true;
    };

    if (threadIdx.x == 0) {
        g.k.template prefetch_tma<k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<k_sc_tile, dim::ROW>();
        g.v_fp4_tma.template prefetch_tma<v_tile, dim::ROW>();
        init_semaphore(k_arrived, 0, 1);
        init_semaphore(k_sc_arrived, 0, 1);
        init_semaphore(v_arrived, 0, 1);
        init_semaphore(v_sc_arrived, 0, 1);
        wait_failed = 0;
        #pragma unroll
        for (int i = 0; i < DEBUG_STRIDE; ++i) {
            dbg[i] = 0;
        }
        dbg[0] = cta_rank;
        dbg[2] = g.mode;
        dbg[3] = 0;
        dbg[4] = 0b11;
        dbg[5] = (int)sizeof(k_tile);
        dbg[6] = (int)sizeof(k_sc_tile);
        dbg[7] = (int)sizeof(v_tile);
        dbg[8] = (int)sizeof(v_sc_tile);
        dbg[9] = g.b;
        dbg[10] = g.h;
        dbg[11] = g.k_iter;
        dbg[12] = g.check_scale;
        dbg[13] = blockIdx.x;
        dbg[14] = cta_rank;
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    set_progress(1);

    if (threadIdx.x == 0) {
        if (g.mode == 1) {
            if (cta_rank == 0) {
                issue_k(0b11);
                issue_v(0b11);
                dbg[1] = 3;
            }
        } else if (g.mode != 3) {
            expect_k();
            expect_v();
            dbg[1] = 2;
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    if (threadIdx.x == 0) {
        if (g.mode == 1) {
            expect_k();
            expect_v();
            dbg[1] = 4;
        } else if (g.mode == 2) {
            issue_k((uint16_t)(1 << cta_rank));
            issue_v((uint16_t)(1 << cta_rank));
            dbg[1] = 3;
            dbg[4] = (1 << cta_rank);
        } else if (g.mode == 3) {
            expect_k();
            dbg[1] = 30;
        } else if (cta_rank == 0) {
            issue_k(0b11);
            issue_v(0b11);
            dbg[1] = 3;
        } else {
            dbg[1] = 3;
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    bool wait_ok = true;
    if (threadIdx.x == 0) {
        if (g.mode == 3) {
            if (cta_rank == 0) {
                issue_k(0b11);
                dbg[1] = 31;
            }
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (threadIdx.x == 0) {
        if (g.mode == 3) {
            wait_ok = wait_k();
            if (wait_ok) {
                fp4pv_publish_shared_backing_proxy_only();
                dbg[1] = 32;
                expect_v();
                dbg[1] = 33;
            }
        } else {
            wait_ok = wait_k();
            if (wait_ok) {
                wait_ok = wait_v();
            }
            if (wait_ok) {
                fp4pv_publish_shared_backing_proxy_only();
                dbg[1] = 5;
            }
        }
        if (!wait_ok) {
            wait_failed = 1;
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (threadIdx.x == 0 && g.mode == 3 && wait_ok) {
        if (cta_rank == 0) {
            issue_v(0b11);
            dbg[1] = 34;
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (threadIdx.x == 0 && g.mode == 3 && wait_ok) {
        wait_ok = wait_v();
        if (wait_ok) {
            fp4pv_publish_shared_backing_proxy_only();
            dbg[1] = 5;
        } else {
            wait_failed = 1;
        }
    }
    __syncthreads();
    if (wait_failed) {
        return;
    }

    if (threadIdx.x == 0) {
        const int packed_cols = C::Dqk / 2;
        const int row_base = g.k_iter * k_tile::rows;
        uint32_t k_base = __cvta_generic_to_shared(&k_smem.data[0]);
        int k_mismatches = 0;
        uint32_t k_global_checksum = 0;
        uint32_t k_tma_checksum = 0;
        #pragma unroll 1
        for (int row = 0; row < k_tile::rows; ++row) {
            #pragma unroll 1
            for (int col = 0; col < k_tile::cols; ++col) {
                const uint32_t global_byte =
                    (uint32_t)g.k_fp4[((((g.b * g.heads + g.h) * g.seqlen) + row_base + row) *
                                       packed_cols) + col];
                uint32_t tma_byte = 0;
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(tma_byte)
                             : "r"(k_tile::idx(k_base, {row, col})));
                k_global_checksum = k_global_checksum * 16777619u + global_byte + 1u;
                k_tma_checksum = k_tma_checksum * 16777619u + tma_byte + 1u;
                k_mismatches += (global_byte != tma_byte);
            }
        }

        uint32_t v_base = __cvta_generic_to_shared(&v_smem.data[0]);
        const int packed_seqlen = g.seqlen / 2;
        const int col_base = g.k_iter * (C::Nb / 2);
        int v_mismatches = 0;
        uint32_t v_global_checksum = 0;
        uint32_t v_tma_checksum = 0;
        #pragma unroll 1
        for (int row = 0; row < v_tile::rows; ++row) {
            #pragma unroll 1
            for (int col = 0; col < v_tile::cols; ++col) {
                const uint32_t global_byte =
                    (uint32_t)g.v_fp4[((((g.b * g.heads + g.h) * C::Dvo) + row) *
                                       packed_seqlen) + col_base + col];
                uint32_t tma_byte = 0;
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(tma_byte)
                             : "r"(v_tile::idx(v_base, {row, col})));
                v_global_checksum = v_global_checksum * 16777619u + global_byte + 1u;
                v_tma_checksum = v_tma_checksum * 16777619u + tma_byte + 1u;
                v_mismatches += (global_byte != tma_byte);
            }
        }

        dbg[24] = k_mismatches;
        dbg[25] = v_mismatches;
        dbg[28] = (int)k_global_checksum;
        dbg[29] = (int)k_tma_checksum;
        dbg[30] = (int)v_global_checksum;
        dbg[31] = (int)v_tma_checksum;
        if (g.check_scale) {
            const int k_sc_depth = g.seqlen / 64;
            const int k_scale_base_offset =
                (((g.b * k_sc_depth + g.k_iter) * (g.heads * C::QK_SCALE_CHUNKS) +
                  g.h * C::QK_SCALE_CHUNKS) * 512);
            uint32_t k_sc_base = __cvta_generic_to_shared(&k_sc_smem.data[0]);
            int k_sc_mismatches = 0;
            uint32_t k_sc_global_checksum = 0;
            uint32_t k_sc_tma_checksum = 0;
            #pragma unroll 1
            for (int byte = 0; byte < (int)sizeof(k_sc_tile); ++byte) {
                const int row = byte / 512;
                const int row_byte = byte - row * 512;
                const int col = row_byte / 2;
                const int byte_in_half = row_byte & 1;
                const uint32_t global_byte = (uint32_t)g.k_sc_raw[k_scale_base_offset + byte];
                uint32_t tma_byte = 0;
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(tma_byte)
                             : "r"(k_sc_tile::idx(k_sc_base, {row, col}) + byte_in_half));
                k_sc_global_checksum = k_sc_global_checksum * 16777619u + global_byte + 1u;
                k_sc_tma_checksum = k_sc_tma_checksum * 16777619u + tma_byte + 1u;
                k_sc_mismatches += (global_byte != tma_byte);
            }

            const int v_scale_base_offset =
                (((((g.b * g.heads + g.h) * g.v_scale_rows + 0) *
                   (g.seqlen / 128)) + g.k_iter) * 32 * 16);
            uint32_t v_sc_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);
            int v_sc_mismatches = 0;
            uint32_t v_sc_global_checksum = 0;
            uint32_t v_sc_tma_checksum = 0;
            #pragma unroll 1
            for (int byte = 0; byte < (int)sizeof(v_sc_tile); ++byte) {
                const uint32_t global_byte = (uint32_t)g.v_sc_prepared[v_scale_base_offset + byte];
                uint32_t tma_byte = 0;
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(tma_byte)
                             : "r"(v_sc_base + byte));
                v_sc_global_checksum = v_sc_global_checksum * 16777619u + global_byte + 1u;
                v_sc_tma_checksum = v_sc_tma_checksum * 16777619u + tma_byte + 1u;
                v_sc_mismatches += (global_byte != tma_byte);
            }
            (void)v_sc_base;
            dbg[26] = k_sc_mismatches;
            dbg[27] = v_sc_mismatches;
            dbg[32] = (int)k_sc_global_checksum;
            dbg[33] = (int)k_sc_tma_checksum;
            dbg[34] = (int)v_sc_global_checksum;
            dbg[35] = (int)v_sc_tma_checksum;
        }
        dbg[1] = 6;
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_qk_tma_multicast_debug(
    const __grid_constant__ globals_mxfp4_qk_tma_multicast_debug<C> g
) {
    using G = globals_mxfp4_qk_tma_multicast_debug<C>;
    using q_tile = typename G::q_tile;
    using q_sc_tile = typename G::q_sc_tile;
    using k_tile = typename G::k_tile;
    using k_sc_tile = typename G::k_sc_tile;
    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_sc16 = full_tt_fp8e4m3<16>;
    static_assert(C::CLUSTER_SIZE == 2, "QK TMA multicast debug is a cluster2 diagnostic");
    static_assert(C::Mb == 128 && C::Nb == 128 && C::Dqk == 192,
                  "QK TMA multicast debug targets Mb128/Nb128/Dqk192");

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    q_tile &q_smem = al.allocate<q_tile>();
    q_sc_tile &q_sc_smem = al.allocate<q_sc_tile>();
    k_tile (&k_smem)[C::LOAD_STAGES] = al.allocate<k_tile, C::LOAD_STAGES>();
    k_sc_tile (&k_sc_smem)[C::K_SCALE_SMEM_SLOTS] = al.allocate<k_sc_tile, C::K_SCALE_SMEM_SLOTS>();

    __shared__ semaphore q_arrived;
    __shared__ semaphore q_sc_arrived;
    __shared__ semaphore k_arrived[C::LOAD_STAGES];
    __shared__ semaphore k_sc_arrived[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore k_finished[C::LOAD_STAGES];
    __shared__ semaphore k_sc_finished[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore k_shared_finished[C::LOAD_STAGES];
    __shared__ semaphore k_sc_shared_finished[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore scores_arrived[2];
    __shared__ int wait_failed;

    const int cta_rank = cluster_ctarank();
    constexpr int DEBUG_STRIDE = 48;
    constexpr int SPIN_LIMIT = 10000000;
    int32_t *dbg = g.debug + cta_rank * DEBUG_STRIDE;

    auto set_progress = [&](int progress) {
        if (threadIdx.x == 0) {
            dbg[1] = progress;
        }
    };
    auto wait_local_or_timeout = [&](semaphore &sem, int phase, int fail_code) -> bool {
        #pragma unroll 1
        for (int spin = 0; spin < SPIN_LIMIT; ++spin) {
            if (try_wait(sem, phase)) {
                return true;
            }
        }
        if (threadIdx.x == 0) {
            dbg[9] = fail_code;
            dbg[10] = phase;
        }
        return false;
    };
    auto wait_cluster_or_timeout = [&](semaphore &sem, int phase, int fail_code) -> bool {
        #pragma unroll 1
        for (int spin = 0; spin < SPIN_LIMIT; ++spin) {
            if (tma::cluster::try_wait(sem, phase)) {
                return true;
            }
        }
        if (threadIdx.x == 0) {
            dbg[9] = fail_code;
            dbg[10] = phase;
        }
        return false;
    };

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<k_sc_tile, dim::ROW>();
        init_semaphore(q_arrived, 0, 1);
        init_semaphore(q_sc_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_STAGES; ++i) {
            init_semaphore(k_arrived[i], 0, 1);
            init_semaphore(k_finished[i], 0, 1);
            init_semaphore(k_shared_finished[i], 2, 0);
        }
        #pragma unroll
        for (int i = 0; i < C::K_SCALE_SMEM_SLOTS; ++i) {
            init_semaphore(k_sc_arrived[i], 0, 1);
            init_semaphore(k_sc_finished[i], 0, 1);
            init_semaphore(k_sc_shared_finished[i], 2, 0);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            init_semaphore(scores_arrived[i], 0, 1);
        }
        wait_failed = 0;
        #pragma unroll
        for (int i = 0; i < DEBUG_STRIDE; ++i) {
            dbg[i] = 0;
        }
        dbg[0] = cta_rank;
        dbg[2] = g.mode;
        dbg[3] = 0;
        dbg[4] = (g.mode == 1 || g.mode == 2) ? 0b11 : (1 << cta_rank);
        dbg[5] = (int)sizeof(k_tile);
        dbg[6] = (int)sizeof(k_sc_tile);
        dbg[7] = g.q_iter;
        dbg[8] = g.k_iter;
        dbg[11] = (int)sizeof(q_tile);
        dbg[12] = (int)sizeof(q_sc_tile);
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    set_progress(1);

    const coord<q_tile> q_coord = {g.b, g.h, g.q_iter, 0};
    const coord<q_sc_tile> q_sc_coord = {g.b, g.q_iter, g.h, 0};
    if (threadIdx.x == 0) {
        tma::load_async<dim::ROW, cache_policy::NORMAL>(q_smem, g.q, q_coord, q_arrived);
        tma::expect_bytes(q_arrived, sizeof(q_tile));
        tma::load_async<dim::ROW, cache_policy::NORMAL>(q_sc_smem, g.q_sc, q_sc_coord, q_sc_arrived);
        tma::expect_bytes(q_sc_arrived, sizeof(q_sc_tile));
        dbg[1] = 2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool q_ok = wait_local_or_timeout(q_arrived, 0, 100 + cta_rank);
        if (q_ok) {
            q_ok = wait_local_or_timeout(q_sc_arrived, 0, 120 + cta_rank);
        }
        if (q_ok) {
            fp4pv_publish_shared_backing_proxy_only();
            dbg[1] = 3;
        } else {
            wait_failed = 1;
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (wait_failed) {
        return;
    }

    if ((g.mode == 1 || g.mode == 2) && threadIdx.x == 0) {
        if (cta_rank == 0) {
            #pragma unroll
            for (int i = 0; i < C::LOAD_STAGES; ++i) {
                arrive(k_shared_finished[i]);
            }
            #pragma unroll
            for (int i = 0; i < C::K_SCALE_SMEM_SLOTS; ++i) {
                arrive(k_sc_shared_finished[i], 2);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < C::LOAD_STAGES; ++i) {
                tma::cluster::arrive(k_shared_finished[i], 0);
            }
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if ((g.mode == 1 || g.mode == 2) && threadIdx.x == 0 && cta_rank == 0) {
        bool reusable_ok = wait_local_or_timeout(k_shared_finished[0], 0, 200);
        if (reusable_ok) {
            reusable_ok = wait_local_or_timeout(k_sc_shared_finished[0], 0, 220);
        }
        dbg[13] = reusable_ok ? 1 : 0;
        if (!reusable_ok) {
            wait_failed = 1;
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (wait_failed) {
        return;
    }

    if (g.mode == 2 || g.mode == 3 || g.mode == 4) {
        tensor_allocator<1, 1> tm_alloc{};
        d_tt_scores tt_score_base = tm_alloc.template allocate<d_tt_scores>(0);
        d_tt_sc16 q_sc_tm = tm_alloc.template allocate<d_tt_sc16>(2 * C::Nb);
        d_tt_sc16 k_sc_tm = tm_alloc.template allocate<d_tt_sc16>(2 * C::Nb + 16);
        const bool qk_issue_leader =
            warpgroup::groupid() == 0 &&
            warpgroup::warpid() == 0 &&
            warp::elect_leader();
        const int max_iters = g.seqlen / 128;
        const int diag_iters = min(16, max_iters - g.k_iter);
        int k_idx = 0;
        int k_phase = 0;
        int k_sc_idx = 0;
        int k_sc_phase = 0;
        int score_phase_mask = 0;
        bool local_failed = false;
        auto record_first_fail = [&](int code, int slot, int phase, int iter) {
            if (threadIdx.x == 0 && dbg[20] == 0) {
                dbg[20] = code;
                dbg[21] = slot;
                dbg[22] = phase;
                dbg[23] = iter;
                dbg[24] = cta_rank;
            }
        };
        if (threadIdx.x == 0) {
            dbg[16] = diag_iters;
        }
        for (int idx = 0; idx < diag_iters; ++idx) {
            const int tile_idx = g.k_iter + idx;
            const coord<k_tile> ring_k_coord = {g.b, g.h, tile_idx, 0};
            const coord<k_sc_tile> ring_k_sc_coord = {g.b, tile_idx, g.h, 0};
            const int packed_state =
                (k_idx & 0xf) |
                ((k_phase & 0xf) << 4) |
                ((k_sc_idx & 0xf) << 8) |
                ((k_sc_phase & 0xf) << 12);
            const int score_slot = idx & 1;
            if (threadIdx.x == 0) {
                dbg[17] = idx;
                dbg[18] = packed_state;
                dbg[19] = score_slot;
            }
            __syncthreads();
            fp4pv_cluster_sync<C>();

            bool iter_ok = !local_failed;
            if (threadIdx.x == 0 && iter_ok) {
                const bool need_k_reuse_wait = (g.mode == 2 && cta_rank == 0) ||
                    (g.mode == 3 && idx >= C::LOAD_STAGES);
                const bool need_k_sc_reuse_wait = (g.mode == 2 && cta_rank == 0) ||
                    (g.mode == 3 && idx >= C::K_SCALE_SMEM_SLOTS);
                const int reuse_k_phase = (g.mode == 3) ? (k_phase ^ 1) : k_phase;
                const int reuse_k_sc_phase = (g.mode == 3) ? (k_sc_phase ^ 1) : k_sc_phase;
                semaphore &reuse_k_sem = (g.mode == 2) ? k_shared_finished[k_idx] : k_finished[k_idx];
                semaphore &reuse_k_sc_sem = (g.mode == 2) ? k_sc_shared_finished[k_sc_idx] : k_sc_finished[k_sc_idx];
                if (need_k_reuse_wait && !wait_local_or_timeout(reuse_k_sem, reuse_k_phase, 600 + k_idx)) {
                    record_first_fail(600 + k_idx, k_idx, reuse_k_phase, idx);
                    iter_ok = false;
                    local_failed = true;
                }
                if (iter_ok && need_k_sc_reuse_wait &&
                    !wait_local_or_timeout(reuse_k_sc_sem, reuse_k_sc_phase, 620 + k_sc_idx)) {
                    record_first_fail(620 + k_sc_idx, k_sc_idx, reuse_k_sc_phase, idx);
                    iter_ok = false;
                    local_failed = true;
                }
            }
            __syncthreads();
            fp4pv_cluster_sync<C>();

            if (threadIdx.x == 0 && iter_ok) {
                tma::cluster::expect_bytes(k_arrived[k_idx], sizeof(k_tile));
                tma::cluster::expect_bytes(k_sc_arrived[k_sc_idx], sizeof(k_sc_tile));
            }
            __syncthreads();
            fp4pv_cluster_sync<C>();

            if (threadIdx.x == 0 && iter_ok) {
                if (g.mode == 2) {
                    if (cta_rank == 0) {
                        tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                            k_smem[k_idx], g.k, ring_k_coord, k_arrived[k_idx], uint16_t(0b11));
                        tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                            k_sc_smem[k_sc_idx], g.k_sc, ring_k_sc_coord, k_sc_arrived[k_sc_idx], uint16_t(0b11));
                    }
                } else {
                    tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                        k_smem[k_idx], g.k, ring_k_coord, k_arrived[k_idx], uint16_t(1 << cta_rank));
                    tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                        k_sc_smem[k_sc_idx], g.k_sc, ring_k_sc_coord, k_sc_arrived[k_sc_idx], uint16_t(1 << cta_rank));
                }
            }
            __syncthreads();
            fp4pv_cluster_sync<C>();

            if (threadIdx.x == 0 && iter_ok) {
                if (!wait_cluster_or_timeout(k_arrived[k_idx], k_phase, 640 + k_idx)) {
                    record_first_fail(640 + k_idx, k_idx, k_phase, idx);
                    iter_ok = false;
                    local_failed = true;
                }
                if (iter_ok && !wait_cluster_or_timeout(k_sc_arrived[k_sc_idx], k_sc_phase, 660 + k_sc_idx)) {
                    record_first_fail(660 + k_sc_idx, k_sc_idx, k_sc_phase, idx);
                    iter_ok = false;
                    local_failed = true;
                }
                if (iter_ok) {
                    fp4pv_publish_shared_backing_proxy_only();
                    dbg[1] = 20 + idx;
                }
            }
            __syncthreads();
            fp4pv_cluster_sync<C>();

            if (qk_issue_leader && iter_ok) {
                tensor_before_thread_sync();
                semaphore &qk_k_finished = (g.mode == 2 && cta_rank == 0)
                    ? k_shared_finished[k_idx]
                    : k_finished[k_idx];
                d_tt_scores tt_score = d_tt_scores(tt_score_base.addr + score_slot * C::Nb);
                fp4pv_issue_qk_chunked_ranklocal<C>(
                    tt_score, q_smem, k_smem[k_idx], q_sc_smem, k_sc_smem[k_sc_idx],
                    q_sc_tm, k_sc_tm, qk_k_finished);
                detail::tcgen05::commit<1>(scores_arrived[score_slot]);
                if (g.mode == 4) {
                    arrive(k_sc_finished[k_sc_idx]);
                    const int score_phase = (score_phase_mask >> score_slot) & 1;
                    if (wait_local_or_timeout(scores_arrived[score_slot], score_phase, 740 + score_slot)) {
                        score_phase_mask ^= (1 << score_slot);
                    } else {
                        record_first_fail(740 + score_slot, score_slot, score_phase, idx);
                        local_failed = true;
                    }
                    const int k_done_phase = k_phase;
                    if (!local_failed && wait_local_or_timeout(k_finished[k_idx], k_done_phase, 680 + k_idx)) {
                        dbg[14] = idx + 1;
                    } else if (!local_failed) {
                        record_first_fail(680 + k_idx, k_idx, k_done_phase, idx);
                        local_failed = true;
                    }
                    if (!local_failed && !wait_local_or_timeout(k_sc_finished[k_sc_idx], k_sc_phase, 760 + k_sc_idx)) {
                        record_first_fail(760 + k_sc_idx, k_sc_idx, k_sc_phase, idx);
                        local_failed = true;
                    }
                } else if (g.mode == 2 && cta_rank == 0) {
                    arrive(k_sc_shared_finished[k_sc_idx]);
                } else if (g.mode == 2) {
                    tma::cluster::arrive(k_sc_shared_finished[k_sc_idx], 0);
                    const int k_done_phase = k_phase;
                    if (wait_local_or_timeout(k_finished[k_idx], k_done_phase, 680 + k_idx)) {
                        tma::cluster::arrive(k_shared_finished[k_idx], 0);
                        dbg[14] = idx + 1;
                    } else {
                        record_first_fail(680 + k_idx, k_idx, k_done_phase, idx);
                        local_failed = true;
                    }
                } else {
                    arrive(k_sc_finished[k_sc_idx]);
                }
            }
            __syncthreads();
            fp4pv_cluster_sync<C>();

            k_idx++;
            if (k_idx == C::LOAD_STAGES) {
                k_idx = 0;
                k_phase ^= 1;
            }
            k_sc_idx++;
            if (k_sc_idx == C::K_SCALE_SMEM_SLOTS) {
                k_sc_idx = 0;
                k_sc_phase ^= 1;
            }
        }
        constexpr int DRAIN_ITERS =
            C::LOAD_STAGES > C::K_SCALE_SMEM_SLOTS ? C::LOAD_STAGES : C::K_SCALE_SMEM_SLOTS;
        #pragma unroll
        for (int drain = 0; drain < DRAIN_ITERS; ++drain) {
            if (threadIdx.x == 0 && !local_failed &&
                (g.mode == 3 || cta_rank == 0)) {
                const int drain_k_phase = (g.mode == 3) ? (k_phase ^ 1) : k_phase;
                const int drain_k_sc_phase = (g.mode == 3) ? (k_sc_phase ^ 1) : k_sc_phase;
                semaphore &drain_k_sem = (g.mode == 2) ? k_shared_finished[k_idx] : k_finished[k_idx];
                semaphore &drain_k_sc_sem = (g.mode == 2) ? k_sc_shared_finished[k_sc_idx] : k_sc_finished[k_sc_idx];
                if (drain < C::LOAD_STAGES &&
                    !wait_local_or_timeout(drain_k_sem, drain_k_phase, 700 + k_idx)) {
                    record_first_fail(700 + k_idx, k_idx, drain_k_phase, diag_iters + drain);
                    local_failed = true;
                }
                if (drain < C::K_SCALE_SMEM_SLOTS &&
                    !wait_local_or_timeout(drain_k_sc_sem, drain_k_sc_phase, 720 + k_sc_idx)) {
                    record_first_fail(720 + k_sc_idx, k_sc_idx, drain_k_sc_phase, diag_iters + drain);
                    local_failed = true;
                }
            }
            __syncthreads();
            fp4pv_cluster_sync<C>();
            if (drain < C::LOAD_STAGES) {
                k_idx++;
                if (k_idx == C::LOAD_STAGES) {
                    k_idx = 0;
                    k_phase ^= 1;
                }
            }
            if (drain < C::K_SCALE_SMEM_SLOTS) {
                k_sc_idx++;
                if (k_sc_idx == C::K_SCALE_SMEM_SLOTS) {
                    k_sc_idx = 0;
                    k_sc_phase ^= 1;
                }
            }
        }
        __syncthreads();
        fp4pv_cluster_sync<C>();
        if (threadIdx.x == 0 && !local_failed) {
            dbg[1] = 7;
            dbg[15] = 1;
        }
        return;
    }

    const coord<k_tile> k_coord = {g.b, g.h, g.k_iter, 0};
    const coord<k_sc_tile> k_sc_coord = {g.b, g.k_iter, g.h, 0};
    if (threadIdx.x == 0) {
        if (g.mode == 1) {
            if (cta_rank == 0) {
                tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                    k_smem[0], g.k, k_coord, k_arrived[0], uint16_t(0b11));
                tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                    k_sc_smem[0], g.k_sc, k_sc_coord, k_sc_arrived[0], uint16_t(0b11));
            }
            tma::cluster::expect_bytes(k_arrived[0], sizeof(k_tile));
            tma::cluster::expect_bytes(k_sc_arrived[0], sizeof(k_sc_tile));
        } else {
            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                k_smem[0], g.k, k_coord, k_arrived[0], uint16_t(1 << cta_rank));
            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                k_sc_smem[0], g.k_sc, k_sc_coord, k_sc_arrived[0], uint16_t(1 << cta_rank));
            tma::cluster::expect_bytes(k_arrived[0], sizeof(k_tile));
            tma::cluster::expect_bytes(k_sc_arrived[0], sizeof(k_sc_tile));
        }
        dbg[1] = 4;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool k_ok = wait_cluster_or_timeout(k_arrived[0], 0, 300 + cta_rank);
        if (k_ok) {
            k_ok = wait_cluster_or_timeout(k_sc_arrived[0], 0, 320 + cta_rank);
        }
        if (k_ok) {
            fp4pv_publish_shared_backing_proxy_only();
            dbg[1] = 5;
        } else {
            wait_failed = 1;
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (wait_failed) {
        return;
    }

    tensor_allocator<1, 1> tm_alloc{};
    d_tt_scores tt_score = tm_alloc.template allocate<d_tt_scores>(0);
    d_tt_sc16 q_sc_tm = tm_alloc.template allocate<d_tt_sc16>(C::Nb);
    d_tt_sc16 k_sc_tm = tm_alloc.template allocate<d_tt_sc16>(C::Nb + 16);
    const bool qk_issue_leader =
        warpgroup::groupid() == 0 &&
        warpgroup::warpid() == 0 &&
        warp::elect_leader();
    if (qk_issue_leader) {
        tensor_before_thread_sync();
        semaphore &qk_k_finished = (g.mode == 1 && cta_rank == 0) ? k_shared_finished[0] : k_finished[0];
        fp4pv_issue_qk_chunked_ranklocal<C>(
            tt_score, q_smem, k_smem[0], q_sc_smem, k_sc_smem[0], q_sc_tm, k_sc_tm, qk_k_finished);
        detail::tcgen05::commit<1>(scores_arrived[0]);
        if (g.mode == 1) {
            if (cta_rank == 0) {
                arrive(k_sc_shared_finished[0]);
            } else {
                tma::cluster::arrive(k_sc_shared_finished[0], 0);
                const int k_done_phase = 1;
                if (wait_local_or_timeout(k_finished[0], k_done_phase, 400 + cta_rank)) {
                    tma::cluster::arrive(k_shared_finished[0], 0);
                    dbg[14] = 1;
                } else {
                    dbg[14] = 0;
                    wait_failed = 1;
                }
            }
        } else {
            arrive(k_sc_finished[0]);
        }
        dbg[1] = 6;
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (wait_failed) {
        return;
    }

    if (threadIdx.x == 0) {
        if (g.mode == 1) {
            if (cta_rank == 0) {
                bool shared_done = wait_local_or_timeout(k_shared_finished[0], 1, 500);
                if (shared_done) {
                    shared_done = wait_local_or_timeout(k_sc_shared_finished[0], 1, 520);
                }
                dbg[15] = shared_done ? 1 : 0;
                if (!shared_done) {
                    wait_failed = 1;
                }
            }
        } else {
            bool local_done = wait_local_or_timeout(k_finished[0], 0, 540 + cta_rank);
            if (local_done) {
                local_done = wait_local_or_timeout(k_sc_finished[0], 0, 560 + cta_rank);
            }
            dbg[15] = local_done ? 1 : 0;
            if (!local_done) {
                wait_failed = 1;
            }
        }
        if (!wait_failed) {
            dbg[1] = 7;
        }
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_v_tma_multicast_debug(
    const __grid_constant__ globals_mxfp4_v_tma_tile_check_debug<C> g
) {
    using G = globals_mxfp4_v_tma_tile_check_debug<C>;
    using v_tile = typename G::v_fp4_tile;
    static_assert(C::CLUSTER_SIZE == 2, "V TMA multicast debug is a cluster2 diagnostic");
    static_assert(C::Dvo == 128 && C::Nb == 128, "V TMA multicast debug currently targets Dvo128/Nb128");

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    v_tile &manual_smem = al.allocate<v_tile>();
    v_tile &tma_smem = al.allocate<v_tile>();
    typename G::v_sc_tile &scale_manual_smem = al.allocate<typename G::v_sc_tile>();
    typename G::v_sc_tile &scale_tma_smem = al.allocate<typename G::v_sc_tile>();
    v_tile (&ring_v_smem)[2] = al.allocate<v_tile, 2>();
    typename G::v_sc_tile (&ring_scale_smem)[2] = al.allocate<typename G::v_sc_tile, 2>();

    __shared__ semaphore tma_arrived;
    __shared__ semaphore scale_arrived;
    __shared__ semaphore tcgen_src_finished;
    __shared__ semaphore tcgen_done;
    __shared__ semaphore owner_done;
    __shared__ semaphore ring_tma_arrived[2];
    __shared__ semaphore ring_scale_arrived[2];
    __shared__ semaphore ring_tcgen_src_finished[2];
    __shared__ semaphore ring_tcgen_done[2];
    __shared__ semaphore ring_remote_finished[2];
    __shared__ int wait_failed;
    const int cta_rank = cluster_ctarank();
    constexpr int DEBUG_STRIDE = 32;
    int32_t *dbg = g.debug + cta_rank * DEBUG_STRIDE;
    constexpr int SPIN_LIMIT = 10000000;

    auto set_progress = [&](int progress) {
        if (threadIdx.x == 0) {
            dbg[1] = progress;
        }
    };
    auto wait_or_timeout = [&](semaphore &sem, int phase, int fail_code) -> bool {
        #pragma unroll 1
        for (int spin = 0; spin < SPIN_LIMIT; ++spin) {
            if (try_wait(sem, phase)) {
                return true;
            }
        }
        if (threadIdx.x == 0) {
            dbg[15] = fail_code;
            dbg[16] = phase;
        }
        return false;
    };

    if (threadIdx.x == 0) {
        g.v_fp4_tma.template prefetch_tma<v_tile, dim::ROW>();
        init_semaphore(tma_arrived, 0, 1);
        init_semaphore(scale_arrived, 0, 1);
        init_semaphore(tcgen_src_finished, 0, 1);
        init_semaphore(tcgen_done, 0, 1);
        init_semaphore(owner_done, 0, 1);
        init_semaphore(ring_tma_arrived[0], 0, 1);
        init_semaphore(ring_tma_arrived[1], 0, 1);
        init_semaphore(ring_scale_arrived[0], 0, 1);
        init_semaphore(ring_scale_arrived[1], 0, 1);
        init_semaphore(ring_tcgen_src_finished[0], 0, 1);
        init_semaphore(ring_tcgen_src_finished[1], 0, 1);
        init_semaphore(ring_tcgen_done[0], 0, 1);
        init_semaphore(ring_tcgen_done[1], 0, 1);
        init_semaphore(ring_remote_finished[0], 0, 1);
        init_semaphore(ring_remote_finished[1], 0, 1);
        wait_failed = 0;
        #pragma unroll
        for (int i = 0; i < DEBUG_STRIDE; ++i) {
            dbg[i] = 0;
        }
        dbg[0] = cta_rank;
        dbg[2] = g.mode;
        dbg[3] = 0;      // issuer rank
        dbg[4] = 0b11;   // multicast target mask
        dbg[5] = (int)sizeof(v_tile);
	        dbg[6] = g.h;
	        dbg[7] = g.k_iter;
	        dbg[8] = -1;
	        dbg[9] = -1;
	        dbg[17] = blockIdx.x;
            dbg[18] = g.check_scale;
            dbg[24] = 0;
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    set_progress(1);

    const int packed_seqlen = g.seqlen / 2;
    const int col_base = g.k_iter * (C::Nb / 2);
    uint32_t manual_base = __cvta_generic_to_shared(&manual_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = v_tile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.v_fp4[((((g.b * g.heads + g.h) * C::Dvo) + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(v_tile::idx(manual_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
	    }
	    __syncthreads();
	    set_progress(2);

	    if (g.check_scale) {
	        const int scale_tile = g.k_iter;
	        const int scale_base_offset =
	            (((((g.b * g.heads + g.h) * g.v_scale_rows + 0) *
	               (g.seqlen / 128)) + scale_tile) * 32 * 16);
	        constexpr int SCALE_BYTES_PER_STORE = 16;
	        constexpr int SCALE_TOTAL_STORES = sizeof(typename G::v_sc_tile) / SCALE_BYTES_PER_STORE;
	        uint32_t scale_manual_base = __cvta_generic_to_shared(&scale_manual_smem.data[0]);
	        for (int linear = threadIdx.x; linear < SCALE_TOTAL_STORES; linear += blockDim.x) {
	            const int byte_offset = linear * SCALE_BYTES_PER_STORE;
	            const uint4 value = *reinterpret_cast<const uint4 *>(
	                &g.v_sc_prepared[scale_base_offset + byte_offset]);
	            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
	                         :
	                         : "r"(scale_manual_base + byte_offset),
	                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
	        }
		    }
		    __syncthreads();

    if (g.mode == 5) {
        using d_tt_outputs = tt<float, C::Dvo, C::Dvo>;
        using d_tt_p_sc = full_tt_fp8e8m0<16>;
        using d_tt_v_sc = full_tt_fp8e8m0<32>;
        tensor_allocator<1, 1> tm_alloc{};
        d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(0);
        d_tt_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(C::Dvo);
        d_tt_v_sc v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(C::Dvo + 16);
        const bool tcgen_warp = (warpgroup::groupid() == C::TOTAL_WGS - 1 && warpgroup::warpid() == 0);
        const bool tcgen_issuer = tcgen_warp && warp::elect_leader();
        const int total_iters = g.seqlen / C::Nb;
        if (threadIdx.x == 0) {
            dbg[1] = 10;
            dbg[7] = total_iters;
            dbg[8] = 0;
            dbg[9] = -1;
            dbg[10] = -1;
            dbg[11] = 0;
            dbg[12] = (int)sizeof(v_tile);
            dbg[13] = g.check_scale ? (int)sizeof(typename G::v_sc_tile) : 0;
            dbg[14] = 0;
            dbg[15] = 0;
            dbg[16] = 0;
            dbg[19] = 0;
            dbg[20] = 0;
            dbg[21] = 0;
            dbg[22] = 0;
            dbg[23] = 0;
            dbg[24] = 0;
            dbg[25] = 0;
            dbg[26] = -1;
            dbg[27] = -1;
        }
        __syncthreads();

        #pragma unroll 1
        for (int iter = 0; iter < total_iters; ++iter) {
            const int slot = iter & 1;
            const int phase = (iter >> 1) & 1;
            const coord<v_tile> ring_v_coord = {g.b, g.h, 0, iter};
            if (threadIdx.x == 0) {
                dbg[1] = 20;
                dbg[25] = iter;
                dbg[26] = slot;
                dbg[27] = iter;
                if (cta_rank == 0) {
                    tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                        ring_v_smem[slot], g.v_fp4_tma, ring_v_coord, ring_tma_arrived[slot], (uint16_t)0b11);
                    if (g.check_scale) {
                        const int scale_base_offset =
                            (((((g.b * g.heads + g.h) * g.v_scale_rows + 0) *
                               (g.seqlen / 128)) + iter) * 32 * 16);
                        tma::cluster::load_async(
                            reinterpret_cast<void *>(&ring_scale_smem[slot].data[0]),
                            const_cast<uint8_t *>(&g.v_sc_prepared[scale_base_offset]),
                            sizeof(typename G::v_sc_tile),
                            ring_scale_arrived[slot],
                            (uint16_t)0b11);
                    }
                }
                tma::cluster::expect_bytes(ring_tma_arrived[slot], sizeof(v_tile));
                if (g.check_scale) {
                    tma::cluster::expect_bytes(ring_scale_arrived[slot], sizeof(typename G::v_sc_tile));
                }
                dbg[1] = 21;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                if (!wait_or_timeout(ring_tma_arrived[slot], phase, 600 + cta_rank)) {
                    dbg[8] = 600 + cta_rank;
                    dbg[9] = iter;
                    dbg[10] = slot;
                    dbg[11] = 1;
                    wait_failed = 1;
                } else {
                    dbg[19] += 1;
                }
                if (!wait_failed && g.check_scale) {
                    if (!wait_or_timeout(ring_scale_arrived[slot], phase, 620 + cta_rank)) {
                        dbg[8] = 620 + cta_rank;
                        dbg[9] = iter;
                        dbg[10] = slot;
                        dbg[11] = 2;
                        wait_failed = 1;
                    } else {
                        dbg[20] += 1;
                    }
                }
                if (!wait_failed) {
                    fp4pv_publish_shared_backing_proxy_only();
                    dbg[1] = 22;
                }
            }
            __syncthreads();
            if (wait_failed) {
                return;
            }

            if (tcgen_warp && g.check_scale) {
                auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
                auto v_sc_tm_sub = v_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
                fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm_sub, scale_manual_smem);
                fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm_sub, ring_scale_smem[slot]);
                if (tcgen_issuer) {
                    dbg[24] = 1;
                }
                tensor_load_wait();
            }
            __syncthreads();
            if (tcgen_issuer) {
                tensor_before_thread_sync();
                if (g.check_scale) {
                    mm_ABt(tt_output, manual_smem, ring_v_smem[slot], p_sc_tm, v_sc_tm,
                           ring_tcgen_src_finished[slot]);
                } else {
                    mm_ABt(tt_output, manual_smem, ring_v_smem[slot], ring_tcgen_src_finished[slot]);
                }
                detail::tcgen05::commit<1>(ring_tcgen_done[slot]);
                dbg[24] = 2;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                if (!wait_or_timeout(ring_tcgen_done[slot], phase, 640 + cta_rank)) {
                    dbg[8] = 640 + cta_rank;
                    dbg[9] = iter;
                    dbg[10] = slot;
                    dbg[11] = 3;
                    wait_failed = 1;
                } else {
                    dbg[21] += 1;
                }
                if (!wait_failed) {
                    if (!wait_or_timeout(ring_tcgen_src_finished[slot], phase, 660 + cta_rank)) {
                        dbg[8] = 660 + cta_rank;
                        dbg[9] = iter;
                        dbg[10] = slot;
                        dbg[11] = 4;
                        wait_failed = 1;
                    } else {
                        dbg[23] += 1;
                    }
                }
                if (!wait_failed && cta_rank == 1) {
                    tma::cluster::arrive(ring_remote_finished[slot], 0);
                    dbg[22] += 1;
                }
                if (!wait_failed && cta_rank == 0) {
                    if (!wait_or_timeout(ring_remote_finished[slot], phase, 680)) {
                        dbg[8] = 680;
                        dbg[9] = iter;
                        dbg[10] = slot;
                        dbg[11] = 5;
                        wait_failed = 1;
                    } else {
                        dbg[22] += 1;
                    }
                }
                if (!wait_failed) {
                    dbg[1] = 23;
                }
            }
            __syncthreads();
            if (wait_failed) {
                return;
            }
        }
        if (threadIdx.x == 0) {
            dbg[1] = 90;
            dbg[25] = total_iters;
        }
        return;
    }

		    const coord<v_tile> v_coord = {g.b, g.h, 0, g.k_iter};
		    if (g.mode == 2 || g.mode == 4 || g.mode == 6 || g.mode == 7) {
        if (threadIdx.x == 0) {
            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                tma_smem, g.v_fp4_tma, v_coord, tma_arrived, (uint16_t)(1 << cta_rank));
            dbg[4] = (1 << cta_rank);
            dbg[1] = 3;
	            tma::cluster::expect_bytes(tma_arrived, sizeof(v_tile));
	            dbg[1] = 4;
	            if (g.check_scale) {
	                tma::cluster::expect_bytes(scale_arrived, sizeof(typename G::v_sc_tile));
	                const int scale_tile = g.k_iter;
	                const int scale_base_offset =
	                    (((((g.b * g.heads + g.h) * g.v_scale_rows + 0) *
	                       (g.seqlen / 128)) + scale_tile) * 32 * 16);
	                tma::cluster::load_async(
	                    reinterpret_cast<void *>(&scale_tma_smem.data[0]),
	                    const_cast<uint8_t *>(&g.v_sc_prepared[scale_base_offset]),
	                    sizeof(typename G::v_sc_tile),
	                    scale_arrived,
	                    (uint16_t)(1 << cta_rank));
	            }
	        }
	    } else if (g.mode == 1) {
        if (threadIdx.x == 0 && cta_rank == 0) {
            tma::cluster::expect_bytes(tma_arrived, sizeof(v_tile));
            dbg[1] = 4;
            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                tma_smem, g.v_fp4_tma, v_coord, tma_arrived, (uint16_t)0b11, 0);
            dbg[1] = 3;
        } else if (threadIdx.x == 0) {
            dbg[1] = 3;
        }
	    } else {
	        if (threadIdx.x == 0 && cta_rank == 0) {
	            tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
	                tma_smem, g.v_fp4_tma, v_coord, tma_arrived, (uint16_t)0b11);
	            if (g.check_scale) {
	                const int scale_tile = g.k_iter;
	                const int scale_base_offset =
	                    (((((g.b * g.heads + g.h) * g.v_scale_rows + 0) *
	                       (g.seqlen / 128)) + scale_tile) * 32 * 16);
	                tma::cluster::load_async(
	                    reinterpret_cast<void *>(&scale_tma_smem.data[0]),
	                    const_cast<uint8_t *>(&g.v_sc_prepared[scale_base_offset]),
	                    sizeof(typename G::v_sc_tile),
	                    scale_arrived,
	                    (uint16_t)0b11);
	            }
	            dbg[1] = 3;
	        } else if (threadIdx.x == 0) {
	            dbg[1] = 3;
	        }
	        if (threadIdx.x == 0) {
	            tma::cluster::expect_bytes(tma_arrived, sizeof(v_tile));
	            if (g.check_scale) {
	                tma::cluster::expect_bytes(scale_arrived, sizeof(typename G::v_sc_tile));
	            }
	            dbg[1] = 4;
	        }
	    }
    __syncthreads();

    bool wait_ok = true;
    if (threadIdx.x == 0) {
        if (g.mode == 1) {
            if (cta_rank == 0) {
                wait_ok = wait_or_timeout(tma_arrived, 0, 101);
                if (wait_ok) {
                    tma::cluster::arrive(owner_done, 1);
                }
            } else {
                wait_ok = wait_or_timeout(owner_done, 0, 102);
            }
	        } else {
	            wait_ok = wait_or_timeout(tma_arrived, 0, 100 + cta_rank);
	            if (wait_ok && g.check_scale) {
	                wait_ok = wait_or_timeout(scale_arrived, 0, 120 + cta_rank);
	            }
	        }
        if (wait_ok) {
            fp4pv_publish_shared_backing_proxy_only();
            dbg[1] = 5;
        } else {
            wait_failed = 1;
        }
    }
    __syncthreads();
    if (wait_failed) {
        return;
    }

    if (threadIdx.x == 0) {
        uint32_t tma_base = __cvta_generic_to_shared(&tma_smem.data[0]);
        int mismatches = 0;
        int first_row = -1;
        int first_col = -1;
        int first_manual = 0;
        int first_tma = 0;
        uint32_t manual_checksum = 0;
        uint32_t tma_checksum = 0;
        for (int row = 0; row < C::Dvo; ++row) {
            for (int col = 0; col < C::Nb / 2; ++col) {
                uint32_t manual_byte = 0;
                uint32_t tma_byte = 0;
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(manual_byte)
                             : "r"(v_tile::idx(manual_base, {row, col})));
                asm volatile("ld.shared.u8 %0, [%1];"
                             : "=r"(tma_byte)
                             : "r"(v_tile::idx(tma_base, {row, col})));
                manual_checksum = manual_checksum * 16777619u + manual_byte + 1u;
                tma_checksum = tma_checksum * 16777619u + tma_byte + 1u;
                if (manual_byte != tma_byte) {
                    if (mismatches == 0) {
                        first_row = row;
                        first_col = col;
                        first_manual = (int)manual_byte;
                        first_tma = (int)tma_byte;
                    }
                    ++mismatches;
                }
            }
        }
        dbg[1] = 6;
        dbg[8] = mismatches;
        dbg[9] = first_row;
        dbg[10] = first_col;
        dbg[11] = first_manual;
        dbg[12] = first_tma;
	        dbg[13] = (int)manual_checksum;
	        dbg[14] = (int)tma_checksum;
	        if (g.check_scale) {
	            uint32_t scale_manual_base = __cvta_generic_to_shared(&scale_manual_smem.data[0]);
	            uint32_t scale_tma_base = __cvta_generic_to_shared(&scale_tma_smem.data[0]);
	            int scale_mismatches = 0;
	            int scale_first = -1;
	            int scale_first_manual = 0;
	            int scale_first_tma = 0;
	            uint32_t scale_manual_checksum = 0;
	            uint32_t scale_tma_checksum = 0;
	            #pragma unroll 1
	            for (int byte = 0; byte < (int)sizeof(typename G::v_sc_tile); ++byte) {
	                uint32_t manual_byte = 0;
	                uint32_t tma_byte = 0;
	                asm volatile("ld.shared.u8 %0, [%1];"
	                             : "=r"(manual_byte)
	                             : "r"(scale_manual_base + byte));
	                asm volatile("ld.shared.u8 %0, [%1];"
	                             : "=r"(tma_byte)
	                             : "r"(scale_tma_base + byte));
	                scale_manual_checksum = scale_manual_checksum * 16777619u + manual_byte + 1u;
	                scale_tma_checksum = scale_tma_checksum * 16777619u + tma_byte + 1u;
	                if (manual_byte != tma_byte) {
	                    if (scale_mismatches == 0) {
	                        scale_first = byte;
	                        scale_first_manual = (int)manual_byte;
	                        scale_first_tma = (int)tma_byte;
	                    }
	                    ++scale_mismatches;
	                }
	            }
	            dbg[18] = scale_mismatches;
	            dbg[19] = scale_first;
	            dbg[20] = scale_first_manual;
	            dbg[21] = scale_first_tma;
	            dbg[22] = (int)scale_manual_checksum;
            dbg[23] = (int)scale_tma_checksum;
        }
	    }
    __syncthreads();

    if (g.mode == 3 || g.mode == 4 || g.mode == 6 || g.mode == 7) {
        using d_tt_outputs = tt<float, C::Dvo, C::Dvo>;
        using d_tt_p_sc = full_tt_fp8e8m0<16>;
        using d_tt_v_sc = full_tt_fp8e8m0<32>;
        tensor_allocator<1, 1> tm_alloc{};
        d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(0);
        d_tt_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(C::Dvo);
        d_tt_v_sc v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(C::Dvo + 16);
        const bool tcgen_warp = (warpgroup::groupid() == C::TOTAL_WGS - 1 && warpgroup::warpid() == 0);
        const bool tcgen_issuer = tcgen_warp && warp::elect_leader();
        if (tcgen_issuer) {
            dbg[24] = 1;
        }
        if (tcgen_warp && g.mode != 6 && g.check_scale) {
                auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
                auto v_sc_tm_sub = v_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
                fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm_sub, scale_manual_smem);
                fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm_sub, scale_tma_smem);
                if (tcgen_issuer) {
                    dbg[24] = 2;
                }
                tensor_load_wait();
        }
        __syncthreads();
        if (tcgen_issuer) {
            if (g.mode != 6 && g.check_scale) {
                tensor_before_thread_sync();
                mm_ABt(tt_output, manual_smem, tma_smem, p_sc_tm, v_sc_tm, tcgen_src_finished);
            } else if (g.mode != 6) {
                tensor_before_thread_sync();
                mm_ABt(tt_output, manual_smem, tma_smem, tcgen_src_finished);
            }
            if (g.mode != 6) {
                detail::tcgen05::commit<1>(tcgen_done);
                dbg[24] = 3;
            }
        }
        __syncthreads();
        if (g.mode == 6 || g.mode == 7) {
            return;
        }
        if (threadIdx.x == 0) {
            if (wait_or_timeout(tcgen_done, 0, 130 + cta_rank)) {
                dbg[24] = 4;
            } else {
                wait_failed = 1;
            }
        }
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_v_tma_tile_check_fullg_debug(
    const __grid_constant__ globals_fp4pv_mxfp4_dv<C, true> g,
    int32_t *debug,
    int b,
    int h,
    int k_iter,
    int mode
) {
    using G = globals_fp4pv_mxfp4_dv<C, true>;
    using v_tile = typename G::v_fp4_tile;
    static_assert(C::CLUSTER_SIZE == 1, "V TMA full-g tile check is a cluster1 debug kernel");
    static_assert(C::Dvo == 128 && C::Nb == 128, "V TMA full-g tile check currently targets Dvo128/Nb128");

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    constexpr int P_STAGE_SLOTS = fp4pv_p_stage_slots<C>::value;
    typename G::q_tile (&q_smem)[C::NUM_SOFTMAXXERS] = al.allocate<typename G::q_tile, C::NUM_SOFTMAXXERS>();
    typename G::q_sc_smem_storage &q_sc_smem = al.allocate<typename G::q_sc_smem_storage>();
    typename G::k_tile (&k_smem)[C::LOAD_STAGES] = al.allocate<typename G::k_tile, C::LOAD_STAGES>();
    typename G::k_sc_smem_storage &k_sc_smem = al.allocate<typename G::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] = al.allocate<typename G::k_sc_guard_slot, 1>();
    typename G::v_fp4_tile (&v_fp4_smem)[2] = al.allocate<typename G::v_fp4_tile, 2>();
    typename G::v_sc_tile (&v_sc_smem)[2] = al.allocate<typename G::v_sc_tile, 2>();
    auto (&o_smem)[1] = al.allocate<typename G::o_tile, 1>();
    auto p_bf16_scan = fp4pv_allocate_p_scan<G::NEED_P_SCAN_STORAGE, 2>(al);
    auto (&p_fp4_stage)[P_STAGE_SLOTS] = al.allocate<typename G::p_fp4_tile, P_STAGE_SLOTS>();
    auto (&p_sc_stage)[P_STAGE_SLOTS] = al.allocate<typename G::p_sc_tile, P_STAGE_SLOTS>();
    auto (&p_sc_guard)[1] = al.allocate<typename G::p_sc_guard_slot, 1>();
    (void)q_smem;
    (void)q_sc_smem;
    (void)k_smem;
    (void)k_sc_smem;
    (void)k_sc_guard;
    (void)v_sc_smem;
    (void)o_smem;
    (void)p_bf16_scan;
    (void)p_fp4_stage;
    (void)p_sc_stage;
    (void)p_sc_guard;
    v_tile &manual_smem = v_fp4_smem[0];
    v_tile &tma_smem = v_fp4_smem[1];

    __shared__ semaphore tma_arrived;
    if (threadIdx.x == 0) {
        g.v_fp4_tma.template prefetch_tma<v_tile, dim::ROW>();
        init_semaphore(tma_arrived, 0, 1);
        for (int i = 0; i < 12; ++i) {
            debug[i] = 0;
        }
        debug[1] = -1;
        debug[2] = -1;
        debug[5] = C::Dvo * (C::Nb / 2);
        debug[6] = h;
        debug[7] = k_iter;
        debug[8] = mode;
        debug[11] = 1;
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    const int packed_seqlen = g.k.rows() / 2;
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t manual_base = __cvta_generic_to_shared(&manual_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = v_tile::rows * STORES_PER_ROW;
    if (mode == 0 || mode == 1) {
        for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.v_fp4[((((b * g.q.depth() + h) * C::Dvo) + row) * packed_seqlen) + col_base + col]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(v_tile::idx(manual_base, {row, col})),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        debug[11] = 2;
    }
    __syncthreads();

    if (threadIdx.x == 0 && (mode == 0 || mode == 2 || mode == 3)) {
        fp4pv_expect_bytes<C>(tma_arrived, sizeof(v_tile));
        const coord<v_tile> v_coord = {b, h, 0, k_iter};
        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(
            tma_smem, g.v_fp4_tma, v_coord, tma_arrived, (uint16_t)1, 0);
        debug[11] = 3;
    }
    __syncthreads();
    if (threadIdx.x == 0 && (mode == 0 || mode == 2 || mode == 3)) {
        fp4pv_wait<C>(tma_arrived, 0);
        fp4pv_publish_shared_backing_proxy_only();
        debug[11] = 4;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (mode == 3) {
            debug[11] = 5;
            return;
        }
        uint32_t tma_base = __cvta_generic_to_shared(&tma_smem.data[0]);
        int mismatches = 0;
        int first_row = -1;
        int first_col = -1;
        int first_manual = 0;
        int first_tma = 0;
        uint32_t manual_checksum = 0;
        uint32_t tma_checksum = 0;
        for (int row = 0; row < C::Dvo; ++row) {
            for (int col = 0; col < C::Nb / 2; ++col) {
                uint32_t manual_byte = 0;
                uint32_t tma_byte = 0;
                if (mode == 0 || mode == 1) {
                    asm volatile("ld.shared.u8 %0, [%1];"
                                 : "=r"(manual_byte)
                                 : "r"(v_tile::idx(manual_base, {row, col})));
                    manual_checksum = manual_checksum * 16777619u + manual_byte + 1u;
                }
                if (mode == 0 || mode == 2) {
                    asm volatile("ld.shared.u8 %0, [%1];"
                                 : "=r"(tma_byte)
                                 : "r"(v_tile::idx(tma_base, {row, col})));
                    tma_checksum = tma_checksum * 16777619u + tma_byte + 1u;
                }
                if (mode == 0 && manual_byte != tma_byte) {
                    if (mismatches == 0) {
                        first_row = row;
                        first_col = col;
                        first_manual = (int)manual_byte;
                        first_tma = (int)tma_byte;
                    }
                    ++mismatches;
                }
            }
        }
        debug[0] = mismatches;
        debug[1] = first_row;
        debug[2] = first_col;
        debug[3] = first_manual;
        debug[4] = first_tma;
        debug[5] = C::Dvo * (C::Nb / 2);
        debug[6] = h;
        debug[7] = k_iter;
        debug[8] = mode;
        debug[9] = (int)manual_checksum;
        debug[10] = (int)tma_checksum;
        debug[11] = 6;
    }
}

template <typename C>
__global__ void kernel_mxfp4_pv_from_p_k256_debug(
    const __grid_constant__ globals_mxfp4_pv_from_p_k256_debug<C> g
) {
    using G = globals_mxfp4_pv_from_p_k256_debug<C>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_p_sc = full_tt_fp8e8m0<32>;
    using d_tt_v_sc = full_tt_fp8e8m0<64>;
    constexpr int MAX_K_TILES = 512;

    const int cta_rank = cluster_ctarank();
    const int row_tiles = g.seqlen / C::Mb;
    const int row_clusters = row_tiles / C::CLUSTER_SIZE;
    const int cluster_linear = blockIdx.x / C::CLUSTER_SIZE;
    const int m_cluster = cluster_linear % row_clusters;
    const int hb = cluster_linear / row_clusters;
    const int h = hb % g.heads;
    const int b = hb / g.heads;
    const int m_tile = m_cluster * C::CLUSTER_SIZE + cta_rank;
    if (b >= g.batch) {
        return;
    }

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    constexpr int SCORE_TMEM_WIDTH = C::Nb;
    d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(SCORE_TMEM_WIDTH);
    constexpr int P_SC_BASE = SCORE_TMEM_WIDTH + C::Dvo;
    constexpr int V_SC_BASE = P_SC_BASE + 32;
    static_assert(V_SC_BASE + 64 <= MAX_TENSOR_COLS, "MXFP4 K256 debug PV TMEM layout exceeds tensor memory budget");
    d_tt_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE);
    d_tt_v_sc v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::p_fp4_tile &p_fp4_smem = al.allocate<typename G::p_fp4_tile>();
    auto (&p_sc_smem)[2] = al.allocate<typename G::p_sc_tile, 2>();
    typename G::v_fp4_tile &v_fp4_smem = al.allocate<typename G::v_fp4_tile>();
    auto (&v_sc_smem)[2] = al.allocate<typename G::v_sc_tile, 2>();

    __shared__ semaphore src_finished[1];
    __shared__ semaphore tile_arrived[1];

    const int k_tiles = g.seqlen / C::Nb;
    if (k_tiles > MAX_K_TILES || (k_tiles & 1)) {
        return;
    }
    if (threadIdx.x == 0) {
        init_semaphore(src_finished[0], 0, 1);
        init_semaphore(tile_arrived[0], 0, 1);
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0) && (cta_rank == 0);
    const bool issuer = issuer_warp && warp::elect_leader();
    int src_phase = 0;
    const int p_tiles = g.p_cols / C::Nb;
    for (int k_iter = 0; k_iter < k_tiles; k_iter += 2) {
        if (threadIdx.x == 0 && k_iter > 0) {
            wait(src_finished[0], src_phase);
            src_phase ^= 1;
        }
        __syncthreads();

        const int p_iter0 = k_iter - (k_iter / p_tiles) * p_tiles;
        const int p_iter1 = p_iter0 + 1;
        load_mxfp4_p_fp4_k256_tile_from_global_allthreads<C>(p_fp4_smem, g, b, h, m_tile, p_iter0);
        load_mxfp4_p_sc_from_global_allthreads<C>(p_sc_smem[0], g, b, h, m_tile, p_iter0);
        load_mxfp4_p_sc_from_global_allthreads<C>(p_sc_smem[1], g, b, h, m_tile, p_iter1);
        load_mxfp4_v_fp4_k256_tile_from_global_allthreads<C>(v_fp4_smem, g, b, h, k_iter, cta_rank);
        load_mxfp4_v_sc_from_global_allthreads<C>(v_sc_smem[0], g, b, h, k_iter);
        load_mxfp4_v_sc_from_global_allthreads<C>(v_sc_smem[1], g, b, h, k_iter + 1);
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        publish_cluster_shared_backing_if_needed<C>();
        __syncthreads();
        fp4pv_cluster_sync<C>();

        if (issuer_warp) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e8m0<16>>(half * 16);
                auto v_sc_tm_sub = v_sc_tm.template subtile<full_tt_fp8e8m0<16>>(half * 16);
                fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_smem[half]);
                fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm_sub, v_sc_smem[half]);
            }
            tensor_load_wait();
        }
        __syncthreads();
        if (issuer) {
            tensor_before_thread_sync();
            if (k_iter == 0) {
                fp4pv_mm_cluster_ABt<C>(
                    tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
            } else {
                fp4pv_mma_cluster_ABt<C>(
                    tt_output, p_fp4_smem, v_fp4_smem, p_sc_tm, v_sc_tm, src_finished[0]);
            }
            if (k_iter + 2 == k_tiles) {
                tensor_commit<C::CLUSTER_SIZE>(tile_arrived[0]);
            }
        }
        __syncthreads();
    }

    if (warpgroup::groupid() == 2) {
        wait(tile_arrived[0], 0);
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        const int row = m_tile * C::Mb + warpgroup::laneid();
        constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
        for (int col = 0; col < C::Dvo; col += 16) {
            float2 o_reg[8];
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            uint32_t packed_words[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                o_reg[i].x *= MXFP4_GEMM_SCALE_CORRECTION;
                o_reg[i].y *= MXFP4_GEMM_SCALE_CORRECTION;
                bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
                packed_words[i] = *reinterpret_cast<uint32_t *>(&packed);
            }
            const int out_offset = (((b * g.seqlen + row) * g.heads + h) * C::Dvo) + col;
            const uint4 packed0{packed_words[0], packed_words[1], packed_words[2], packed_words[3]};
            const uint4 packed1{packed_words[4], packed_words[5], packed_words[6], packed_words[7]};
            *reinterpret_cast<uint4 *>(&g.o_direct[out_offset]) = packed0;
            *reinterpret_cast<uint4 *>(&g.o_direct[out_offset + 8]) = packed1;
        }
    }
}

template <typename Tile>
__device__ inline void fp4pv_shape_zero_tile(Tile &tile) {
    uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int byte = threadIdx.x * 4; byte < (int)sizeof(Tile); byte += blockDim.x * 4) {
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(base + byte), "r"(0u));
    }
}

template <typename C, typename ATile, typename G>
__device__ inline void load_mxfp4_shape_a_tile(ATile &a_smem, const G &g, int row_tile) {
    fp4pv_shape_zero_tile(a_smem);
    const int row_base = row_tile * C::Mb;
    uint32_t smem_base = __cvta_generic_to_shared(&a_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int LOGICAL_PACKED_COLS = C::Nb / 2;
    static_assert(LOGICAL_PACKED_COLS % BYTES_PER_STORE == 0, "shape A tile packed K must be vector-loadable");
    constexpr int STORES_PER_ROW = LOGICAL_PACKED_COLS / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = ATile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.a_fp4[((row_base + row) * g.a_packed_cols) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(ATile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BTile, typename G>
__device__ inline void load_mxfp4_shape_b_tile(BTile &b_smem, const G &g, int cta_rank, int out_phase = 0) {
    fp4pv_shape_zero_tile(b_smem);
    const int row_base = out_phase * C::ACC_DVO + cta_rank * (C::ACC_DVO / C::CLUSTER_SIZE);
    uint32_t smem_base = __cvta_generic_to_shared(&b_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int LOGICAL_PACKED_COLS = C::Nb / 2;
    static_assert(LOGICAL_PACKED_COLS % BYTES_PER_STORE == 0, "shape B tile packed K must be vector-loadable");
    constexpr int STORES_PER_ROW = LOGICAL_PACKED_COLS / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = BTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.b_fp4[((row_base + row) * g.b_packed_cols) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(BTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BTile, typename G>
__device__ inline void load_mxfp4_joint_b_quarter_tile(BTile &b_smem, const G &g, int quarter) {
    fp4pv_shape_zero_tile(b_smem);
    const int row_base = quarter * C::Nb;
    uint32_t smem_base = __cvta_generic_to_shared(&b_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int LOGICAL_PACKED_COLS = C::Nb / 2;
    static_assert(LOGICAL_PACKED_COLS % BYTES_PER_STORE == 0, "joint B tile packed K must be vector-loadable");
    constexpr int STORES_PER_ROW = LOGICAL_PACKED_COLS / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = BTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.b_fp4[((row_base + row) * g.b_packed_cols) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(BTile::idx(smem_base, {row, col})),
	                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BTile, typename G>
__device__ inline void load_mxfp4_joint_b_half_tile(BTile &b_smem, const G &g, int half) {
    fp4pv_shape_zero_tile(b_smem);
    const int row_base = half * C::NTile;
    uint32_t smem_base = __cvta_generic_to_shared(&b_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int LOGICAL_PACKED_COLS = C::KTile / 2;
    static_assert(LOGICAL_PACKED_COLS % BYTES_PER_STORE == 0, "joint B half tile packed K must be vector-loadable");
    constexpr int STORES_PER_ROW = LOGICAL_PACKED_COLS / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = BTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.b_fp4[((row_base + row) * g.b_packed_cols) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(BTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BTile, typename G>
__device__ inline void load_mxfp4_joint_b_quarter_tile_producer_wg(
    BTile &b_smem,
    const G &g,
    int quarter
) {
    fp4pv_shape_zero_tile(b_smem);
    const int row_base = quarter * C::Nb;
    uint32_t smem_base = __cvta_generic_to_shared(&b_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    constexpr int LOGICAL_PACKED_COLS = C::Nb / 2;
    static_assert(LOGICAL_PACKED_COLS % BYTES_PER_STORE == 0, "joint B tile packed K must be vector-loadable");
    constexpr int STORES_PER_ROW = LOGICAL_PACKED_COLS / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = BTile::rows * STORES_PER_ROW;
    if (warpgroup::groupid() == 0) {
        for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += WARPGROUP_WARPS * WARP_THREADS) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.b_fp4[((row_base + row) * g.b_packed_cols) + col]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(BTile::idx(smem_base, {row, col})),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    }
}

template <typename ScTile>
__device__ inline void load_mxfp4_shape_sc_tile_producer_wg(ScTile &sc_smem, const uint8_t *src) {
    uint32_t smem_base = __cvta_generic_to_shared(&sc_smem.data[0]);
    constexpr int BYTES_PER_STORE = 4;
    static_assert(sizeof(ScTile) % BYTES_PER_STORE == 0, "shape scale tile must be word-loadable");
    constexpr int TOTAL_STORES = sizeof(ScTile) / BYTES_PER_STORE;
    if (warpgroup::groupid() == 0) {
        for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += WARPGROUP_WARPS * WARP_THREADS) {
            const int byte_offset = linear * BYTES_PER_STORE;
            const uint32_t value = *reinterpret_cast<const uint32_t *>(&src[byte_offset]);
            asm volatile("st.shared.b32 [%0], %1;"
                         :
                         : "r"(smem_base + byte_offset), "r"(value));
        }
    }
}

__device__ __forceinline__ void fp4pv_shared_async_proxy_fence_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void fp4pv_shared_membar_cta() {
    asm volatile("membar.cta;\n" ::: "memory");
}

__device__ __forceinline__ void fp4pv_shared_flag_publish(int *flag, int value, bool proxy_fence) {
    if (proxy_fence) {
        fp4pv_shared_async_proxy_fence_cta();
    }
    fp4pv_shared_membar_cta();
    *reinterpret_cast<volatile int *>(flag) = value;
}

__device__ __forceinline__ void fp4pv_shared_flag_wait_eq(int *flag, int value, bool proxy_fence_after) {
    volatile int *flag_v = reinterpret_cast<volatile int *>(flag);
    while (*flag_v != value) {
        __nanosleep(32);
    }
    fp4pv_shared_membar_cta();
    if (proxy_fence_after) {
        fp4pv_shared_async_proxy_fence_cta();
        fp4pv_shared_membar_cta();
    }
}

template <typename OutputTile>
__device__ inline void load_mxfp4_joint_output_col_regs(
    const OutputTile &tt_output,
    int row_warp,
    int col,
    float2 (&o_reg)[8]
) {
    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
        : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
          "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
          "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
          "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
        : "r"(tt_output.addr + ((row_warp * 32) << 16) + col));
    tensor_load_wait();
}

template <typename OutputTile>
__device__ inline void load_mxfp4_joint_output_col_regs(
    const OutputTile &tt_output,
    int col,
    float2 (&o_reg)[8]
) {
    load_mxfp4_joint_output_col_regs(tt_output, warpgroup::warpid(), col, o_reg);
}

template <typename OutputTile>
__device__ inline void load_mxfp4_joint_output_col_only(
    const OutputTile &tt_output,
    int col
) {
    float2 o_reg[8];
    load_mxfp4_joint_output_col_regs(tt_output, col, o_reg);
    asm volatile("" :: "f"(o_reg[0].x), "f"(o_reg[7].y) : "memory");
}

template <typename OutputTile>
__device__ inline void rescale_mxfp4_joint_output_col_only(
    const OutputTile &tt_output,
    int col
) {
    constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
    float2 o_reg[8];
    load_mxfp4_joint_output_col_regs(tt_output, col, o_reg);
    float sink = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        o_reg[i].x *= MXFP4_GEMM_SCALE_CORRECTION;
        o_reg[i].y *= MXFP4_GEMM_SCALE_CORRECTION;
        sink += o_reg[i].x + o_reg[i].y;
    }
    asm volatile("" :: "f"(sink) : "memory");
}

template <bool ApplyCorrection, typename C, typename OutputTile, typename G>
__device__ inline void store_mxfp4_joint_output_col_vec(
    const OutputTile &tt_output,
    const G &g,
    int row,
    int row_warp,
    int col
) {
    constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
    float2 o_reg[8];
    uint32_t packed_words[8];
    load_mxfp4_joint_output_col_regs(tt_output, row_warp, col, o_reg);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (ApplyCorrection) {
            o_reg[i].x *= MXFP4_GEMM_SCALE_CORRECTION;
            o_reg[i].y *= MXFP4_GEMM_SCALE_CORRECTION;
        }
        bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
        packed_words[i] = *reinterpret_cast<uint32_t *>(&packed);
    }
    const int out_offset = (row * C::Dvo) + col;
    const uint4 packed0{packed_words[0], packed_words[1], packed_words[2], packed_words[3]};
    const uint4 packed1{packed_words[4], packed_words[5], packed_words[6], packed_words[7]};
    *reinterpret_cast<uint4 *>(&g.o_direct[out_offset]) = packed0;
    *reinterpret_cast<uint4 *>(&g.o_direct[out_offset + 8]) = packed1;
}

template <bool ApplyCorrection, typename C, typename OutputTile, typename G>
__device__ inline void store_mxfp4_joint_output_col_vec(
    const OutputTile &tt_output,
    const G &g,
    int row,
    int col
) {
    store_mxfp4_joint_output_col_vec<ApplyCorrection, C>(tt_output, g, row, warpgroup::warpid(), col);
}

template <bool ApplyCorrection, typename C, typename OutputTile, typename G>
__device__ inline void store_mxfp4_joint_output_split_add_col_vec(
    const OutputTile &tt_output0,
    const OutputTile &tt_output1,
    const G &g,
    int row,
    int row_warp,
    int col
) {
    constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
    float2 o0_reg[8];
    float2 o1_reg[8];
    uint32_t packed_words[8];
    load_mxfp4_joint_output_col_regs(tt_output0, row_warp, col, o0_reg);
    load_mxfp4_joint_output_col_regs(tt_output1, row_warp, col, o1_reg);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float2 out_pair{o0_reg[i].x + o1_reg[i].x, o0_reg[i].y + o1_reg[i].y};
        if constexpr (ApplyCorrection) {
            out_pair.x *= MXFP4_GEMM_SCALE_CORRECTION;
            out_pair.y *= MXFP4_GEMM_SCALE_CORRECTION;
        }
        bf16_2 packed = __float22bfloat162_rn(out_pair);
        packed_words[i] = *reinterpret_cast<uint32_t *>(&packed);
    }
    const int out_offset = (row * C::Dvo) + col;
    const uint4 packed0{packed_words[0], packed_words[1], packed_words[2], packed_words[3]};
    const uint4 packed1{packed_words[4], packed_words[5], packed_words[6], packed_words[7]};
    *reinterpret_cast<uint4 *>(&g.o_direct[out_offset]) = packed0;
    *reinterpret_cast<uint4 *>(&g.o_direct[out_offset + 8]) = packed1;
}

template <bool ApplyCorrection, typename C, typename OutputTile, typename G>
__device__ inline void store_mxfp4_joint_output_split_add_col_vec(
    const OutputTile &tt_output0,
    const OutputTile &tt_output1,
    const G &g,
    int row,
    int col
) {
    store_mxfp4_joint_output_split_add_col_vec<ApplyCorrection, C>(
        tt_output0, tt_output1, g, row, warpgroup::warpid(), col);
}

template <typename C, typename G>
__device__ inline void store_mxfp4_joint_output_const_scalar(const G &g, int row) {
    const float2 value = make_float2(1.0f, 1.0f);
    const bf16_2 packed = __float22bfloat162_rn(value);
    #pragma unroll
    for (int col = 0; col < C::Dvo; col += 2) {
        *reinterpret_cast<bf16_2 *>(&g.o_direct[(row * C::Dvo) + col]) = packed;
    }
}

template <typename C, typename G>
__device__ inline void store_mxfp4_joint_output_const_vec(const G &g, int row) {
    const float2 value = make_float2(1.0f, 1.0f);
    const bf16_2 packed = __float22bfloat162_rn(value);
    const uint32_t word = *reinterpret_cast<const uint32_t *>(&packed);
    const uint4 packed4{word, word, word, word};
    #pragma unroll
    for (int col = 0; col < C::Dvo; col += 16) {
        const int out_offset = (row * C::Dvo) + col;
        *reinterpret_cast<uint4 *>(&g.o_direct[out_offset]) = packed4;
        *reinterpret_cast<uint4 *>(&g.o_direct[out_offset + 8]) = packed4;
    }
}

template <typename C, typename OutputTile, typename G>
__device__ inline void store_mxfp4_joint_output_col_raw(
    const OutputTile &tt_output,
    const G &g,
    int row,
    int col
) {
    float2 o_reg[8];
    load_mxfp4_joint_output_col_regs(tt_output, col, o_reg);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
        *reinterpret_cast<bf16_2 *>(&g.o_direct[(row * C::Dvo) + col + i * 2]) = packed;
    }
}

template <typename ScTile>
__device__ inline void load_mxfp4_shape_sc_tile(ScTile &sc_smem, const uint8_t *src) {
    uint32_t smem_base = __cvta_generic_to_shared(&sc_smem.data[0]);
    constexpr int BYTES_PER_STORE = 4;
    static_assert(sizeof(ScTile) % BYTES_PER_STORE == 0, "shape scale tile must be word-loadable");
    constexpr int TOTAL_STORES = sizeof(ScTile) / BYTES_PER_STORE;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int byte_offset = linear * BYTES_PER_STORE;
        const uint32_t value = *reinterpret_cast<const uint32_t *>(&src[byte_offset]);
        asm volatile("st.shared.b32 [%0], %1;"
                     :
                     : "r"(smem_base + byte_offset), "r"(value));
    }
}

template <typename ScTile>
__device__ inline void load_mxfp4_shape_sc_tile_first_word(ScTile &sc_smem, const uint8_t *src) {
    if (threadIdx.x == 0) {
        uint32_t smem_base = __cvta_generic_to_shared(&sc_smem.data[0]);
        const uint32_t value = *reinterpret_cast<const uint32_t *>(src);
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_base), "r"(value));
    }
}

template <typename C, int FixedDebugStage = -1>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_joint_qk_pv_quarter_debug(
    const __grid_constant__ globals_mxfp4_joint_qk_pv_quarter_debug<C> g
) {
    static_assert(FixedDebugStage == -1 || FixedDebugStage == 0,
                  "fixed joint-quarter stage is only used for production stage0");
    constexpr bool STAGE0_PRODUCTION = FixedDebugStage == 0;
    const int debug_stage = STAGE0_PRODUCTION ? 0 : g.debug_stage;
    using G = globals_mxfp4_joint_qk_pv_quarter_debug<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_a_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_b_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = C::Nb;
    constexpr int A_SC_BASE = OUTPUT_BASE + C::Dvo;
    constexpr int B_SC_BASE = A_SC_BASE + 16 * C::K_SCALE_CHUNKS;
    static_assert(B_SC_BASE + 16 * C::K_SCALE_CHUNKS <= 512,
                  "joint QK/PV debug TMEM map must fit score+output+scales");

    const int row_tile = blockIdx.x;
    const int row = row_tile * C::Mb + warpgroup::laneid();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_a_sc a_sc_tm = tm_alloc.template allocate<d_tt_a_sc>(A_SC_BASE);
    d_tt_b_sc b_sc_tm = tm_alloc.template allocate<d_tt_b_sc>(B_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_fp4_tile &a_smem = al.allocate<typename G::a_fp4_tile>();
    typename G::b_fp4_tile &b_smem = al.allocate<typename G::b_fp4_tile>();
    auto (&a_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&b_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();

    __shared__ semaphore qk_src_finished[C::QUARTERS];
    __shared__ semaphore qk_done[C::QUARTERS];
    __shared__ semaphore pv_src_finished[C::QUARTERS];
    __shared__ semaphore pv_done[C::QUARTERS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
            init_semaphore(qk_src_finished[quarter], 0, 1);
            init_semaphore(qk_done[quarter], 0, 1);
            init_semaphore(pv_src_finished[quarter], 0, 1);
            init_semaphore(pv_done[quarter], 0, 1);
        }
    }
    __syncthreads();
    if constexpr (!STAGE0_PRODUCTION) {
        if (debug_stage == 1) {
            return;
        }
    }

    if constexpr (!STAGE0_PRODUCTION) {
        if (debug_stage == 13 || debug_stage == 14) {
            if (warpgroup::groupid() == 2) {
                if (row < g.rows) {
                    if (debug_stage == 13) {
                        store_mxfp4_joint_output_const_scalar<C>(g, row);
                    } else {
                        store_mxfp4_joint_output_const_vec<C>(g, row);
                    }
                }
            }
            return;
        }
    }

    load_mxfp4_shape_a_tile<C>(a_smem, g, row_tile);
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
        load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
    }
    __syncthreads();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0);
    const bool issuer = issuer_warp && warp::elect_leader();

    if constexpr (C::HOIST_A_SCALE_TMEM) {
        if (issuer_warp) {
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
            }
            tensor_load_wait();
        }
        __syncthreads();
    }

    #pragma unroll
    for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
        load_mxfp4_joint_b_quarter_tile<C>(b_smem, g, quarter);
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            const uint8_t *b_sc_src = &g.b_sc[((quarter * g.b_scale_k_tiles + k) * 32 * 16)];
            load_mxfp4_shape_sc_tile(b_sc_smem[k], b_sc_src);
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        if constexpr (!STAGE0_PRODUCTION) {
            if (debug_stage == 3 && quarter == 0) {
                return;
            }
        }

        if (issuer_warp) {
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                auto b_sc_tm_sub = b_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                if constexpr (!C::HOIST_A_SCALE_TMEM) {
                    auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                    fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
                }
                fp4pv_load_mxnv_scale_async_cluster<C>(b_sc_tm_sub, b_sc_smem[k]);
            }
            tensor_load_wait();
        }
        __syncthreads();

        const bool skip_qk_issue = (!STAGE0_PRODUCTION && debug_stage == 6);
        if (issuer) {
            tensor_before_thread_sync();
            if (!skip_qk_issue) {
                fp4pv_mm_cluster_ABt<C>(tt_score, a_smem, b_smem, a_sc_tm, b_sc_tm, qk_src_finished[quarter]);
                tensor_commit<C::CLUSTER_SIZE>(qk_done[quarter]);
            }
            tensor_before_thread_sync();
            if (quarter == 0) {
                fp4pv_mm_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, pv_src_finished[quarter]);
            } else {
                fp4pv_mma_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, pv_src_finished[quarter]);
            }
            tensor_commit<C::CLUSTER_SIZE>(pv_done[quarter]);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            if (!skip_qk_issue) {
                wait(qk_done[quarter], 0);
            }
            wait(pv_done[quarter], 0);
        }
        __syncthreads();
        if constexpr (!STAGE0_PRODUCTION) {
            if (debug_stage == 2 && quarter == 0) {
                return;
            }
            if (debug_stage == 4 && quarter == 0) {
                return;
            }
            if (debug_stage == 5 && quarter == 0) {
                break;
            }
        }
    }

    if constexpr (!STAGE0_PRODUCTION) {
        if (debug_stage == 7) {
            return;
        }
    }

    if (warpgroup::groupid() == 2) {
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        if (row >= g.rows) {
            return;
        }
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            if constexpr (STAGE0_PRODUCTION) {
                store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
            } else {
                if (debug_stage == 8) {
                    load_mxfp4_joint_output_col_only(tt_output, col);
                } else if (debug_stage == 9) {
                    rescale_mxfp4_joint_output_col_only(tt_output, col);
                } else if (debug_stage == 10) {
                    store_mxfp4_joint_output_col_raw<C>(tt_output, g, row, col);
                } else if (debug_stage == 11) {
                    store_mxfp4_joint_output_col_vec<false, C>(tt_output, g, row, col);
                } else if (debug_stage == 12) {
                    store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
                } else {
                    store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
                }
            }
        }
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_joint_qk_pv_q4_stage0_perf(
    const __grid_constant__ globals_mxfp4_joint_qk_pv_q4_stage0_perf<C> g
) {
    static_assert(C::Nb == 128, "q4 stage0 perf path is fixed at quarter_tile=128");
    static_assert(C::QUARTERS == 4, "q4 stage0 perf path is fixed at four quarters");
    static_assert(C::HOIST_A_SCALE_TMEM, "q4 stage0 perf path is fixed to opt2 A-scale hoisting");
    using G = globals_mxfp4_joint_qk_pv_q4_stage0_perf<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_a_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_b_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = C::Nb;
    constexpr int A_SC_BASE = OUTPUT_BASE + C::Dvo;
    constexpr int B_SC_BASE = A_SC_BASE + 16 * C::K_SCALE_CHUNKS;
    static_assert(B_SC_BASE + 16 * C::K_SCALE_CHUNKS <= 512,
                  "q4 stage0 perf TMEM map must fit score+output+scales");

    const int row_tile = blockIdx.x;
    const int row = row_tile * C::Mb + warpgroup::laneid();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_a_sc a_sc_tm = tm_alloc.template allocate<d_tt_a_sc>(A_SC_BASE);
    d_tt_b_sc b_sc_tm = tm_alloc.template allocate<d_tt_b_sc>(B_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_fp4_tile &a_smem = al.allocate<typename G::a_fp4_tile>();
    typename G::b_fp4_tile &b_smem = al.allocate<typename G::b_fp4_tile>();
    auto (&a_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&b_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();

    __shared__ semaphore qk_src_finished[C::QUARTERS];
    __shared__ semaphore qk_done[C::QUARTERS];
    __shared__ semaphore pv_src_finished[C::QUARTERS];
    __shared__ semaphore pv_done[C::QUARTERS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
            init_semaphore(qk_src_finished[quarter], 0, 1);
            init_semaphore(qk_done[quarter], 0, 1);
            init_semaphore(pv_src_finished[quarter], 0, 1);
            init_semaphore(pv_done[quarter], 0, 1);
        }
    }
    __syncthreads();

    load_mxfp4_shape_a_tile<C>(a_smem, g, row_tile);
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
        load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
    }
    __syncthreads();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0);
    const bool issuer = issuer_warp && warp::elect_leader();

    if (issuer_warp) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
            fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
        }
        tensor_load_wait();
    }
    __syncthreads();

    #pragma unroll
    for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
        load_mxfp4_joint_b_quarter_tile<C>(b_smem, g, quarter);
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            const uint8_t *b_sc_src = &g.b_sc[((quarter * g.b_scale_k_tiles + k) * 32 * 16)];
            load_mxfp4_shape_sc_tile(b_sc_smem[k], b_sc_src);
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();

        if (issuer_warp) {
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                auto b_sc_tm_sub = b_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                fp4pv_load_mxnv_scale_async_cluster<C>(b_sc_tm_sub, b_sc_smem[k]);
            }
            tensor_load_wait();
        }
        __syncthreads();

        if (issuer) {
            tensor_before_thread_sync();
            fp4pv_mm_cluster_ABt<C>(tt_score, a_smem, b_smem, a_sc_tm, b_sc_tm, qk_src_finished[quarter]);
            tensor_commit<C::CLUSTER_SIZE>(qk_done[quarter]);
            tensor_before_thread_sync();
            if (quarter == 0) {
                fp4pv_mm_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, pv_src_finished[quarter]);
            } else {
                fp4pv_mma_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, pv_src_finished[quarter]);
            }
            tensor_commit<C::CLUSTER_SIZE>(pv_done[quarter]);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            wait(qk_done[quarter], 0);
            wait(pv_done[quarter], 0);
        }
        __syncthreads();
    }

    if (warpgroup::groupid() == 2) {
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        if (row >= g.rows) {
            return;
        }
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
        }
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf(
    const __grid_constant__ globals_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf<C> g
) {
    static_assert(C::Nb == 128, "q4 scale ping-pong perf path is fixed at quarter_tile=128");
    static_assert(C::QUARTERS == 4, "q4 scale ping-pong perf path is fixed at four quarters");
    static_assert(C::HOIST_A_SCALE_TMEM, "q4 scale ping-pong perf path is fixed to opt2 A-scale hoisting");
    static_assert(C::B_SCALE_SLOTS == 2, "q4 scale ping-pong perf path requires two B-scale slots");
    using G = globals_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_a_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_b_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = C::Nb;
    constexpr int A_SC_BASE = OUTPUT_BASE + C::Dvo;
    constexpr int B_SC_BASE = A_SC_BASE + 16 * C::K_SCALE_CHUNKS;
    constexpr int B_SC_STRIDE = 16 * C::K_SCALE_CHUNKS;
    static_assert(B_SC_BASE + C::B_SCALE_SLOTS * B_SC_STRIDE <= 512,
                  "q4 scale ping-pong TMEM map must fit score+output+scales");

    const int row_tile = blockIdx.x;
    const int row = row_tile * C::Mb + warpgroup::laneid();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_a_sc a_sc_tm = tm_alloc.template allocate<d_tt_a_sc>(A_SC_BASE);
    d_tt_b_sc b_sc_tm0 = tm_alloc.template allocate<d_tt_b_sc>(B_SC_BASE);
    d_tt_b_sc b_sc_tm1 = tm_alloc.template allocate<d_tt_b_sc>(B_SC_BASE + B_SC_STRIDE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_fp4_tile &a_smem = al.allocate<typename G::a_fp4_tile>();
    typename G::b_fp4_tile &b_smem = al.allocate<typename G::b_fp4_tile>();
    auto (&a_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&b_sc_smem)[C::B_SCALE_SLOTS][C::K_SCALE_CHUNKS] =
        al.allocate<typename G::sc_tile, C::B_SCALE_SLOTS, C::K_SCALE_CHUNKS>();

    __shared__ semaphore qk_src_finished[C::QUARTERS];
    __shared__ semaphore qk_done[C::QUARTERS];
    __shared__ semaphore pv_src_finished[C::QUARTERS];
    __shared__ semaphore pv_done[C::QUARTERS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
            init_semaphore(qk_src_finished[quarter], 0, 1);
            init_semaphore(qk_done[quarter], 0, 1);
            init_semaphore(pv_src_finished[quarter], 0, 1);
            init_semaphore(pv_done[quarter], 0, 1);
        }
    }
    __syncthreads();

    load_mxfp4_shape_a_tile<C>(a_smem, g, row_tile);
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
        load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
    }
    __syncthreads();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0);
    const bool issuer = issuer_warp && warp::elect_leader();

    if (issuer_warp) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
            fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
        }
        tensor_load_wait();
    }
    __syncthreads();

    load_mxfp4_joint_b_quarter_tile<C>(b_smem, g, 0);
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *b_sc_src = &g.b_sc[(k * 32 * 16)];
        load_mxfp4_shape_sc_tile(b_sc_smem[0][k], b_sc_src);
    }
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();
    if (issuer_warp) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            auto b_sc_tm_sub = b_sc_tm0.template subtile<full_tt_fp8e8m0<16>>(k * 16);
            fp4pv_load_mxnv_scale_async_cluster<C>(b_sc_tm_sub, b_sc_smem[0][k]);
        }
        tensor_load_wait();
    }
    __syncthreads();

    #pragma unroll
    for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
        const int scale_slot = quarter & 1;
        if (issuer) {
            auto &b_sc_tm = (scale_slot == 0) ? b_sc_tm0 : b_sc_tm1;
            tensor_before_thread_sync();
            fp4pv_mm_cluster_ABt<C>(tt_score, a_smem, b_smem, a_sc_tm, b_sc_tm, qk_src_finished[quarter]);
            tensor_commit<C::CLUSTER_SIZE>(qk_done[quarter]);
            tensor_before_thread_sync();
            if (quarter == 0) {
                fp4pv_mm_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, pv_src_finished[quarter]);
            } else {
                fp4pv_mma_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, pv_src_finished[quarter]);
            }
            tensor_commit<C::CLUSTER_SIZE>(pv_done[quarter]);
        }
        __syncthreads();

        if (quarter + 1 < C::QUARTERS) {
            const int next_quarter = quarter + 1;
            const int next_slot = next_quarter & 1;
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                const uint8_t *b_sc_src = &g.b_sc[((next_quarter * g.b_scale_k_tiles + k) * 32 * 16)];
                load_mxfp4_shape_sc_tile(b_sc_smem[next_slot][k], b_sc_src);
            }
            __syncthreads();
            fp4pv_shared_async_proxy_fence_cta();
            __syncthreads();
            if (issuer_warp) {
                if (next_slot == 0) {
                    #pragma unroll
                    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                        auto b_sc_tm_sub = b_sc_tm0.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                        fp4pv_load_mxnv_scale_async_cluster<C>(b_sc_tm_sub, b_sc_smem[0][k]);
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                        auto b_sc_tm_sub = b_sc_tm1.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                        fp4pv_load_mxnv_scale_async_cluster<C>(b_sc_tm_sub, b_sc_smem[1][k]);
                    }
                }
                tensor_load_wait();
            }
        }

        if (threadIdx.x == 0) {
            wait(qk_done[quarter], 0);
            wait(pv_done[quarter], 0);
        }
        __syncthreads();

        if (quarter + 1 < C::QUARTERS) {
            load_mxfp4_joint_b_quarter_tile<C>(b_smem, g, quarter + 1);
            __syncthreads();
            fp4pv_shared_async_proxy_fence_cta();
            __syncthreads();
        }
    }

    if (warpgroup::groupid() == 2) {
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        if (row >= g.rows) {
            return;
        }
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
        }
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_joint_qk_pv_half_stage0_perf(
    const __grid_constant__ globals_mxfp4_joint_qk_pv_half_stage0_perf<C> g
) {
    static_assert(C::KTile == 128, "half stage0 perf path is fixed at K=128 for fair q4 comparison");
    static_assert(C::NTile == 256, "half stage0 perf path uses two N256 half tiles");
    static_assert(C::HALVES == 2, "half stage0 perf path covers the q4 N512 problem with two halves");
    static_assert(C::B_SCALE_GROUPS == 2, "half stage0 perf path expects two q4 B-scale groups per half");
    using G = globals_mxfp4_joint_qk_pv_half_stage0_perf<C>;
    using d_tt_score = tt<float, C::Mb, C::NTile>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_a_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_b_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS * C::B_SCALE_GROUPS>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = C::NTile;
    constexpr int A_SC_BASE = OUTPUT_BASE + C::Dvo;
    constexpr int B_SC_BASE = A_SC_BASE + 16 * C::K_SCALE_CHUNKS;
    static_assert(B_SC_BASE + 16 * C::K_SCALE_CHUNKS * C::B_SCALE_GROUPS <= 512,
                  "half stage0 perf TMEM map must fit score+output+scales");

    const int row_tile = blockIdx.x;
    const int row = row_tile * C::Mb + warpgroup::laneid();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_a_sc a_sc_tm = tm_alloc.template allocate<d_tt_a_sc>(A_SC_BASE);
    d_tt_b_sc b_sc_tm = tm_alloc.template allocate<d_tt_b_sc>(B_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_fp4_tile &a_smem = al.allocate<typename G::a_fp4_tile>();
    typename G::b_fp4_tile &b_smem = al.allocate<typename G::b_fp4_tile>();
    auto (&b_pv_smem)[C::B_SCALE_GROUPS] =
        al.allocate<typename G::b_pv_fp4_tile, C::B_SCALE_GROUPS>();
    auto (&a_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&b_sc_smem)[C::K_SCALE_CHUNKS * C::B_SCALE_GROUPS] =
        al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS * C::B_SCALE_GROUPS>();

    __shared__ semaphore qk_src_finished[C::HALVES];
    __shared__ semaphore qk_done[C::HALVES];
    __shared__ semaphore pv_src_finished[C::HALVES * C::B_SCALE_GROUPS];
    __shared__ semaphore pv_done[C::HALVES * C::B_SCALE_GROUPS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int half = 0; half < C::HALVES; ++half) {
            init_semaphore(qk_src_finished[half], 0, 1);
            init_semaphore(qk_done[half], 0, 1);
        }
        #pragma unroll
        for (int idx = 0; idx < C::HALVES * C::B_SCALE_GROUPS; ++idx) {
            init_semaphore(pv_src_finished[idx], 0, 1);
            init_semaphore(pv_done[idx], 0, 1);
        }
    }
    __syncthreads();

    load_mxfp4_shape_a_tile<C>(a_smem, g, row_tile);
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
        load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
    }
    __syncthreads();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0);
    const bool issuer = issuer_warp && warp::elect_leader();

    if (issuer_warp) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
            fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
        }
        tensor_load_wait();
    }
    __syncthreads();

    #pragma unroll
    for (int half = 0; half < C::HALVES; ++half) {
        load_mxfp4_joint_b_half_tile<C>(b_smem, g, half);
        #pragma unroll
        for (int group = 0; group < C::B_SCALE_GROUPS; ++group) {
            load_mxfp4_joint_b_quarter_tile<C>(
                b_pv_smem[group],
                g,
                half * C::B_SCALE_GROUPS + group);
        }
        #pragma unroll
        for (int group = 0; group < C::B_SCALE_GROUPS; ++group) {
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                const int q4_scale_group = half * C::B_SCALE_GROUPS + group;
                const uint8_t *b_sc_src = &g.b_sc[((q4_scale_group * g.b_scale_k_tiles + k) * 32 * 16)];
                load_mxfp4_shape_sc_tile(b_sc_smem[group * C::K_SCALE_CHUNKS + k], b_sc_src);
            }
        }
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();

        if (issuer_warp) {
            #pragma unroll
            for (int group = 0; group < C::B_SCALE_GROUPS; ++group) {
                #pragma unroll
                for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                    const int offset = (group * C::K_SCALE_CHUNKS + k) * 16;
                    auto b_sc_tm_sub = b_sc_tm.template subtile<full_tt_fp8e8m0<16>>(offset);
                    fp4pv_load_mxnv_scale_async_cluster<C>(
                        b_sc_tm_sub, b_sc_smem[group * C::K_SCALE_CHUNKS + k]);
                }
            }
            tensor_load_wait();
        }
        __syncthreads();

        if (issuer) {
            tensor_before_thread_sync();
            fp4pv_mm_cluster_ABt<C>(tt_score, a_smem, b_smem, a_sc_tm, b_sc_tm, qk_src_finished[half]);
            tensor_commit<C::CLUSTER_SIZE>(qk_done[half]);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            wait(qk_done[half], 0);
        }
        __syncthreads();

        #pragma unroll
        for (int group = 0; group < C::B_SCALE_GROUPS; ++group) {
            if (issuer) {
                const int pv_idx = half * C::B_SCALE_GROUPS + group;
                auto b_sc_tm_sub = b_sc_tm.template subtile<full_tt_fp8e8m0<16>>(group * 16);
                tensor_before_thread_sync();
                if (pv_idx == 0) {
                    fp4pv_mm_cluster_ABt<C>(
                        tt_output, a_smem, b_pv_smem[group], a_sc_tm, b_sc_tm_sub, pv_src_finished[pv_idx]);
                } else {
                    fp4pv_mma_cluster_ABt<C>(
                        tt_output, a_smem, b_pv_smem[group], a_sc_tm, b_sc_tm_sub, pv_src_finished[pv_idx]);
                }
                tensor_commit<C::CLUSTER_SIZE>(pv_done[pv_idx]);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                wait(pv_done[half * C::B_SCALE_GROUPS + group], 0);
            }
            __syncthreads();
        }
    }

    if (warpgroup::groupid() == 2) {
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        if (row >= g.rows) {
            return;
        }
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
        }
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_joint_qk_pv_true_q4_stage0_ref(
    const __grid_constant__ globals_mxfp4_joint_qk_pv_true_q4_stage0_ref<C> g
) {
    static_assert(C::KTile == 128, "true q4 reference keeps QK feature K=128");
    static_assert(C::Nb == 128, "true q4 reference uses four logical N=128 quarters");
    static_assert(C::NTotal == 512, "true q4 reference covers the same logical N=512 problem");
    static_assert(C::QUARTERS == 4, "true q4 reference must use four q4 tiles");
    using G = globals_mxfp4_joint_qk_pv_true_q4_stage0_ref<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_a_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_k_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_p_sc = full_tt_fp8e8m0<16>;
    using d_tt_v_sc = full_tt_fp8e8m0<16>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = C::Nb;
    constexpr int A_SC_BASE = OUTPUT_BASE + C::Dvo;
    constexpr int K_SC_BASE = A_SC_BASE + 16 * C::K_SCALE_CHUNKS;
    constexpr int P_SC_BASE = K_SC_BASE + 16 * C::K_SCALE_CHUNKS;
    constexpr int PV_SCALE_SLOT_STRIDE = 32;
    constexpr int V_SC_BASE = P_SC_BASE + 16;
    constexpr int P_SC_BASE_1 = P_SC_BASE + PV_SCALE_SLOT_STRIDE;
    constexpr int V_SC_BASE_1 = P_SC_BASE_1 + 16;
    constexpr int SCALE_TMEM_END = P_SC_BASE + C::PV_SCALE_TMEM_SLOTS * PV_SCALE_SLOT_STRIDE;
    constexpr int OUTPUT_ALT_BASE = C::SPLIT_OUTPUT_ACCUM ? SCALE_TMEM_END : OUTPUT_BASE;
    constexpr int SCORE_PING_BASE = SCALE_TMEM_END + (C::SPLIT_OUTPUT_ACCUM ? C::Dvo : 0);
    constexpr int TMEM_END = SCORE_PING_BASE + ((C::SCORE_TMEM_SLOTS > 1) ? C::Nb : 0);
    static_assert(TMEM_END <= 512, "true q4 TMEM map must fit score/output/scale ping-pong");
    static_assert(!C::SPLIT_OUTPUT_ACCUM || C::SCORE_TMEM_SLOTS == 1,
                  "true q4 split-output accumulator leaves no room for a second score/P TMEM slot");
    static_assert(!C::SPLIT_OUTPUT_ACCUM || C::PV_SCALE_TMEM_SLOTS == 2,
                  "true q4 split-output accumulator needs two P/V scale TMEM slots to avoid serial reuse");
    static_assert(C::SCORE_TMEM_SLOTS == 1 || C::SCORE_TMEM_SLOTS == 2,
                  "true q4 supports one or two score TMEM slots");
    static_assert(C::P_SOURCE_SLOTS == 1 || C::P_SOURCE_SLOTS == 2,
                  "true q4 supports one or two P source slots");
    static_assert(C::V_SOURCE_SLOTS == 1 || C::V_SOURCE_SLOTS == 2,
                  "true q4 supports one or two V source slots");
    static_assert(C::P_SCALE_SMEM_SLOTS == 1 || C::P_SCALE_SMEM_SLOTS == 2,
                  "true q4 supports one or two P scale shared slots");
    static_assert(C::V_SCALE_SMEM_SLOTS == 1 || C::V_SCALE_SMEM_SLOTS == 2,
                  "true q4 supports one or two V scale shared slots");
    static_assert(C::PV_SCALE_TMEM_SLOTS == 1 || C::PV_SCALE_TMEM_SLOTS == 2,
                  "true q4 supports one or two P/V scale TMEM slots");

    const int row_tile = blockIdx.x;
    const int row = row_tile * C::Mb + warpgroup::laneid();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_output tt_output_alt = tm_alloc.template allocate<d_tt_output>(OUTPUT_ALT_BASE);
    d_tt_a_sc a_sc_tm = tm_alloc.template allocate<d_tt_a_sc>(A_SC_BASE);
    d_tt_k_sc k_sc_tm = tm_alloc.template allocate<d_tt_k_sc>(K_SC_BASE);
    d_tt_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE);
    d_tt_v_sc v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_fp4_tile &a_smem = al.allocate<typename G::a_fp4_tile>();
    typename G::k_fp4_tile &k_smem = al.allocate<typename G::k_fp4_tile>();
    auto (&p_smem_slots)[C::P_SOURCE_SLOTS] =
        al.allocate<typename G::p_fp4_tile, C::P_SOURCE_SLOTS>();
    auto (&v_smem_slots)[C::V_SOURCE_SLOTS] =
        al.allocate<typename G::v_fp4_tile, C::V_SOURCE_SLOTS>();
    auto (&a_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&k_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&p_sc_smem_slots)[C::P_SCALE_SMEM_SLOTS] =
        al.allocate<typename G::sc_tile, C::P_SCALE_SMEM_SLOTS>();
    auto (&v_sc_smem_slots)[C::V_SCALE_SMEM_SLOTS] =
        al.allocate<typename G::sc_tile, C::V_SCALE_SMEM_SLOTS>();

    __shared__ semaphore qk_src_finished[C::QUARTERS];
    __shared__ semaphore qk_done[C::QUARTERS];
    __shared__ semaphore pv_src_finished[C::QUARTERS];
    __shared__ semaphore pv_done[C::QUARTERS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
            init_semaphore(qk_src_finished[quarter], 0, 1);
            init_semaphore(qk_done[quarter], 0, 1);
            init_semaphore(pv_src_finished[quarter], 0, 1);
            init_semaphore(pv_done[quarter], 0, 1);
        }
    }
    __syncthreads();

    load_mxfp4_shape_a_tile<C>(a_smem, g, row_tile);
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
        load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
    }
    __syncthreads();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0);
    const bool issuer = issuer_warp && warp::elect_leader();
    if (issuer_warp) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
            fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
        }
        tensor_load_wait();
    }
    __syncthreads();

    if constexpr (C::PIPELINE_QK_PV) {
        d_tt_score tt_score_ping = tm_alloc.template allocate<d_tt_score>(SCORE_PING_BASE);
        d_tt_p_sc p_sc_tm_ping = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE_1);
        d_tt_v_sc v_sc_tm_ping = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE_1);

        #pragma unroll
        for (int preload_quarter = 0; preload_quarter < 1; ++preload_quarter) {
            load_mxfp4_joint_b_quarter_tile<C>(k_smem, g, preload_quarter);
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                const uint8_t *k_sc_src = &g.b_sc[((preload_quarter * g.b_scale_k_tiles + k) * 32 * 16)];
                load_mxfp4_shape_sc_tile(k_sc_smem[k], k_sc_src);
            }
            __syncthreads();
            fp4pv_shared_async_proxy_fence_cta();
            __syncthreads();

            if (issuer_warp) {
                #pragma unroll
                for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                    auto k_sc_tm_sub = k_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                    fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm_sub, k_sc_smem[k]);
                }
                tensor_load_wait();
            }
            __syncthreads();

            if (issuer) {
                tensor_before_thread_sync();
                fp4pv_mm_cluster_ABt<C>(tt_score, a_smem, k_smem, a_sc_tm, k_sc_tm, qk_src_finished[preload_quarter]);
                tensor_commit<C::CLUSTER_SIZE>(qk_done[preload_quarter]);
            }
            __syncthreads();
        }

        #pragma unroll
        for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
            const int p_slot = quarter % C::P_SOURCE_SLOTS;
            const int v_slot = quarter % C::V_SOURCE_SLOTS;
            const int p_scale_smem_slot = quarter % C::P_SCALE_SMEM_SLOTS;
            const int v_scale_smem_slot = quarter % C::V_SCALE_SMEM_SLOTS;
            const int scale_tmem_slot = quarter % C::PV_SCALE_TMEM_SLOTS;
            auto &p_smem = p_smem_slots[p_slot];
            auto &v_smem = v_smem_slots[v_slot];
            auto &p_sc_smem = p_sc_smem_slots[p_scale_smem_slot];
            auto &v_sc_smem = v_sc_smem_slots[v_scale_smem_slot];

            int waited_source_quarter = -1;
            const int v_source_wait_quarter = quarter - C::V_SOURCE_SLOTS;
            if constexpr (!(C::V_SOURCE_SLOTS == 1 && C::P_SOURCE_SLOTS == 2)) {
                if (v_source_wait_quarter >= 0) {
                    if (threadIdx.x == 0) {
                        wait(pv_src_finished[v_source_wait_quarter], 0);
                    }
                    __syncthreads();
                    waited_source_quarter = v_source_wait_quarter;
                }
                load_mxfp4_v_fp4_tile_from_global_allthreads<C>(v_smem, g, 0, 0, quarter, 0);
                load_mxfp4_v_sc_from_global_allthreads<C>(v_sc_smem, g, 0, 0, quarter);
            }

            if (threadIdx.x == 0) {
                wait(qk_done[quarter], 0);
            }
            __syncthreads();

            const int p_source_wait_quarter = quarter - C::P_SOURCE_SLOTS;
            if (p_source_wait_quarter >= 0 && p_source_wait_quarter != waited_source_quarter) {
                if (threadIdx.x == 0) {
                    wait(pv_src_finished[p_source_wait_quarter], 0);
                }
                __syncthreads();
                waited_source_quarter = p_source_wait_quarter;
            }

            if (warpgroup::groupid() == 0 || warpgroup::groupid() == 1) {
                float2 scores_half[32];
                auto score_addr = tt_score.addr;
                if constexpr (C::SCORE_TMEM_SLOTS == 2) {
                    if ((quarter & 1) != 0) {
                        score_addr = tt_score_ping.addr;
                    }
                }
                const int score_half = warpgroup::groupid();
                fp4pv_load_score_half_tmem(scores_half, score_addr, score_half);
                tensor_load_wait();
                if (warpgroup::groupid() == 0) {
                    fp4pv_pack_scores_half_to_stage_mxfp4_range<0, 2>(
                        p_smem, p_sc_smem, scores_half, row, 0);
                } else {
                    fp4pv_pack_scores_half_to_stage_mxfp4_range<2, 4>(
                        p_smem, p_sc_smem, scores_half, row, 0);
                }
            }

            if constexpr (C::V_SOURCE_SLOTS == 1 && C::P_SOURCE_SLOTS == 2) {
                if (v_source_wait_quarter >= 0 && v_source_wait_quarter != waited_source_quarter) {
                    if (threadIdx.x == 0) {
                        wait(pv_src_finished[v_source_wait_quarter], 0);
                    }
                    __syncthreads();
                    waited_source_quarter = v_source_wait_quarter;
                }
                load_mxfp4_v_fp4_tile_from_global_allthreads<C>(v_smem, g, 0, 0, quarter, 0);
                load_mxfp4_v_sc_from_global_allthreads<C>(v_sc_smem, g, 0, 0, quarter);
            }
            __syncthreads();
            fp4pv_shared_async_proxy_fence_cta();
            __syncthreads();

            int waited_pv_quarter = -1;
            if constexpr (C::PV_SCALE_TMEM_SLOTS == 1) {
                if (quarter > 0) {
                    if (threadIdx.x == 0) {
                        wait(pv_done[quarter - 1], 0);
                    }
                    __syncthreads();
                    waited_pv_quarter = quarter - 1;
                }
            }

            if (issuer_warp) {
                if constexpr (C::PV_SCALE_TMEM_SLOTS == 2) {
                    if (scale_tmem_slot == 0) {
                        fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm, p_sc_smem);
                        fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm, v_sc_smem);
                    } else {
                        fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_ping, p_sc_smem);
                        fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm_ping, v_sc_smem);
                    }
                } else {
                    fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm, p_sc_smem);
                    fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm, v_sc_smem);
                }
                tensor_load_wait();
            }
            __syncthreads();

            if constexpr (C::SPLIT_OUTPUT_ACCUM) {
                if (quarter >= 2 && waited_pv_quarter != quarter - 2) {
                    if (threadIdx.x == 0) {
                        wait(pv_done[quarter - 2], 0);
                    }
                    __syncthreads();
                    waited_pv_quarter = quarter - 2;
                }
            } else {
                if (quarter > 0 && waited_pv_quarter != quarter - 1) {
                    if (threadIdx.x == 0) {
                        wait(pv_done[quarter - 1], 0);
                    }
                    __syncthreads();
                    waited_pv_quarter = quarter - 1;
                }
            }

            if (issuer) {
                tensor_before_thread_sync();
                if constexpr (C::PV_SCALE_TMEM_SLOTS == 2) {
                    if (scale_tmem_slot == 0) {
                        if constexpr (C::SPLIT_OUTPUT_ACCUM) {
                            if ((quarter & 1) == 0) {
                                if (quarter < 2) {
                                    if constexpr (C::PV_SPLIT_K64) {
                                        fp4pv_p_stage_ABt_split_k64<C, true>(
                                            tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                    } else {
                                        fp4pv_mm_cluster_ABt<C>(
                                            tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                    }
                                } else {
                                    if constexpr (C::PV_SPLIT_K64) {
                                        fp4pv_p_stage_ABt_split_k64<C, false>(
                                            tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                    } else {
                                        fp4pv_mma_cluster_ABt<C>(
                                            tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                    }
                                }
                            } else {
                                if (quarter < 2) {
                                    if constexpr (C::PV_SPLIT_K64) {
                                        fp4pv_p_stage_ABt_split_k64<C, true>(
                                            tt_output_alt, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                    } else {
                                        fp4pv_mm_cluster_ABt<C>(
                                            tt_output_alt, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                    }
                                } else {
                                    if constexpr (C::PV_SPLIT_K64) {
                                        fp4pv_p_stage_ABt_split_k64<C, false>(
                                            tt_output_alt, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                    } else {
                                        fp4pv_mma_cluster_ABt<C>(
                                            tt_output_alt, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                    }
                                }
                            }
                        } else {
                            if (quarter == 0) {
                                if constexpr (C::PV_SPLIT_K64) {
                                    fp4pv_p_stage_ABt_split_k64<C, true>(
                                        tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                } else {
                                    fp4pv_mm_cluster_ABt<C>(
                                        tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                }
                            } else {
                                if constexpr (C::PV_SPLIT_K64) {
                                    fp4pv_p_stage_ABt_split_k64<C, false>(
                                        tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                } else {
                                    fp4pv_mma_cluster_ABt<C>(
                                        tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                                }
                            }
                        }
                    } else {
                        if constexpr (C::SPLIT_OUTPUT_ACCUM) {
                            if ((quarter & 1) == 0) {
                                if (quarter < 2) {
                                    if constexpr (C::PV_SPLIT_K64) {
                                        fp4pv_p_stage_ABt_split_k64<C, true>(
                                            tt_output, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                    } else {
                                        fp4pv_mm_cluster_ABt<C>(
                                            tt_output, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                    }
                                } else {
                                    if constexpr (C::PV_SPLIT_K64) {
                                        fp4pv_p_stage_ABt_split_k64<C, false>(
                                            tt_output, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                    } else {
                                        fp4pv_mma_cluster_ABt<C>(
                                            tt_output, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                    }
                                }
                            } else {
                                if (quarter < 2) {
                                    if constexpr (C::PV_SPLIT_K64) {
                                        fp4pv_p_stage_ABt_split_k64<C, true>(
                                            tt_output_alt, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                    } else {
                                        fp4pv_mm_cluster_ABt<C>(
                                            tt_output_alt, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                    }
                                } else {
                                    if constexpr (C::PV_SPLIT_K64) {
                                        fp4pv_p_stage_ABt_split_k64<C, false>(
                                            tt_output_alt, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                    } else {
                                        fp4pv_mma_cluster_ABt<C>(
                                            tt_output_alt, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                    }
                                }
                            }
                        } else {
                            if (quarter == 0) {
                                if constexpr (C::PV_SPLIT_K64) {
                                    fp4pv_p_stage_ABt_split_k64<C, true>(
                                        tt_output, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                } else {
                                    fp4pv_mm_cluster_ABt<C>(
                                        tt_output, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                }
                            } else {
                                if constexpr (C::PV_SPLIT_K64) {
                                    fp4pv_p_stage_ABt_split_k64<C, false>(
                                        tt_output, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                } else {
                                    fp4pv_mma_cluster_ABt<C>(
                                        tt_output, p_smem, v_smem, p_sc_tm_ping, v_sc_tm_ping, pv_src_finished[quarter]);
                                }
                            }
                        }
                    }
                } else {
                    if (quarter == 0) {
                        if constexpr (C::PV_SPLIT_K64) {
                            fp4pv_p_stage_ABt_split_k64<C, true>(
                                tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                        } else {
                            fp4pv_mm_cluster_ABt<C>(
                                tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                        }
                    } else {
                        if constexpr (C::PV_SPLIT_K64) {
                            fp4pv_p_stage_ABt_split_k64<C, false>(
                                tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                        } else {
                            fp4pv_mma_cluster_ABt<C>(
                                tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                        }
                    }
                }
                tensor_commit<C::CLUSTER_SIZE>(pv_done[quarter]);
            }
            __syncthreads();

            if (quarter + 1 < C::QUARTERS) {
                load_mxfp4_joint_b_quarter_tile<C>(k_smem, g, quarter + 1);
                #pragma unroll
                for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                    const uint8_t *k_sc_src = &g.b_sc[(((quarter + 1) * g.b_scale_k_tiles + k) * 32 * 16)];
                    load_mxfp4_shape_sc_tile(k_sc_smem[k], k_sc_src);
                }
                __syncthreads();
                fp4pv_shared_async_proxy_fence_cta();
                __syncthreads();

                if (issuer_warp) {
                    #pragma unroll
                    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                        auto k_sc_tm_sub = k_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                        fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm_sub, k_sc_smem[k]);
                    }
                    tensor_load_wait();
                }
                __syncthreads();

                if (issuer) {
                    tensor_before_thread_sync();
                    if constexpr (C::SCORE_TMEM_SLOTS == 2) {
                        if (((quarter + 1) & 1) == 0) {
                            fp4pv_mm_cluster_ABt<C>(
                                tt_score, a_smem, k_smem, a_sc_tm, k_sc_tm, qk_src_finished[quarter + 1]);
                        } else {
                            fp4pv_mm_cluster_ABt<C>(
                                tt_score_ping, a_smem, k_smem, a_sc_tm, k_sc_tm, qk_src_finished[quarter + 1]);
                        }
                    } else {
                        fp4pv_mm_cluster_ABt<C>(
                            tt_score, a_smem, k_smem, a_sc_tm, k_sc_tm, qk_src_finished[quarter + 1]);
                    }
                    tensor_commit<C::CLUSTER_SIZE>(qk_done[quarter + 1]);
                }
                __syncthreads();
            }
        }

        if (threadIdx.x == 0) {
            wait(pv_done[C::QUARTERS - 1], 0);
        }
        __syncthreads();
    } else {
        auto &p_smem = p_smem_slots[0];
        auto &v_smem = v_smem_slots[0];
        auto &p_sc_smem = p_sc_smem_slots[0];
        auto &v_sc_smem = v_sc_smem_slots[0];

        #pragma unroll
        for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
            load_mxfp4_joint_b_quarter_tile<C>(k_smem, g, quarter);
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                const uint8_t *k_sc_src = &g.b_sc[((quarter * g.b_scale_k_tiles + k) * 32 * 16)];
                load_mxfp4_shape_sc_tile(k_sc_smem[k], k_sc_src);
            }
            __syncthreads();
            fp4pv_shared_async_proxy_fence_cta();
            __syncthreads();

            if (issuer_warp) {
                #pragma unroll
                for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                    auto k_sc_tm_sub = k_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                    fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm_sub, k_sc_smem[k]);
                }
                tensor_load_wait();
            }
            __syncthreads();

            if (issuer) {
                tensor_before_thread_sync();
                fp4pv_mm_cluster_ABt<C>(tt_score, a_smem, k_smem, a_sc_tm, k_sc_tm, qk_src_finished[quarter]);
                tensor_commit<C::CLUSTER_SIZE>(qk_done[quarter]);
            }

            if constexpr (C::PRESTAGE_V_UNDER_QK) {
                load_mxfp4_v_fp4_tile_from_global_allthreads<C>(v_smem, g, 0, 0, quarter, 0);
                load_mxfp4_v_sc_from_global_allthreads<C>(v_sc_smem, g, 0, 0, quarter);
            } else {
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                wait(qk_done[quarter], 0);
            }
            __syncthreads();

            if (warpgroup::groupid() == 0 || warpgroup::groupid() == 1) {
                float2 scores_half[32];
                const int score_half = warpgroup::groupid();
                fp4pv_load_score_half_tmem(scores_half, tt_score.addr, score_half);
                tensor_load_wait();
                if (warpgroup::groupid() == 0) {
                    fp4pv_pack_scores_half_to_stage_mxfp4_range<0, 2>(
                        p_smem, p_sc_smem, scores_half, row, 0);
                } else {
                    fp4pv_pack_scores_half_to_stage_mxfp4_range<2, 4>(
                        p_smem, p_sc_smem, scores_half, row, 0);
                }
            }
            if constexpr (!C::PRESTAGE_V_UNDER_QK) {
                load_mxfp4_v_fp4_tile_from_global_allthreads<C>(v_smem, g, 0, 0, quarter, 0);
                load_mxfp4_v_sc_from_global_allthreads<C>(v_sc_smem, g, 0, 0, quarter);
            }
            __syncthreads();
            fp4pv_shared_async_proxy_fence_cta();
            __syncthreads();

            if (issuer_warp) {
                fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm, p_sc_smem);
                fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm, v_sc_smem);
                tensor_load_wait();
            }
            if constexpr (!C::TRIM_PV_ISSUER_SYNCS) {
                __syncthreads();
            }

            if (issuer) {
                tensor_before_thread_sync();
                if (quarter == 0) {
                    if constexpr (C::PV_SPLIT_K64) {
                        fp4pv_p_stage_ABt_split_k64<C, true>(
                            tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                    } else {
                        fp4pv_mm_cluster_ABt<C>(
                            tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                    }
                } else {
                    if constexpr (C::PV_SPLIT_K64) {
                        fp4pv_p_stage_ABt_split_k64<C, false>(
                            tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                    } else {
                        fp4pv_mma_cluster_ABt<C>(
                            tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[quarter]);
                    }
                }
                tensor_commit<C::CLUSTER_SIZE>(pv_done[quarter]);
            }
            if constexpr (!C::TRIM_PV_ISSUER_SYNCS) {
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                wait(pv_done[quarter], 0);
            }
            __syncthreads();
        }
    }

    if (warpgroup::groupid() == 2) {
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        if (row >= g.rows) {
            return;
        }
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            if constexpr (C::SPLIT_OUTPUT_ACCUM) {
                store_mxfp4_joint_output_split_add_col_vec<true, C>(
                    tt_output, tt_output_alt, g, row, col);
            } else {
                store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
            }
        }
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_joint_qk_pv_true_half_stage0_perf(
    const __grid_constant__ globals_mxfp4_joint_qk_pv_true_half_stage0_perf<C> g
) {
    static_assert(C::KTile == 128, "true half keeps QK feature K=128");
    static_assert(C::NTile == 256, "true half uses two logical N=256 halves");
    static_assert(C::NTotal == 512, "true half covers the same logical N=512 problem");
    static_assert(C::HALVES == 2, "true half must use exactly two halves");
    static_assert(C::HALF_SCALE_GROUPS == 2, "true half uses two K128 scale groups per logical half");
    using G = globals_mxfp4_joint_qk_pv_true_half_stage0_perf<C>;
    using d_tt_score = tt<float, C::Mb, C::NTile>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_a_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_k_sc = full_tt_fp8e8m0<16 * C::HALF_SCALE_GROUPS>;
    using d_tt_p_sc = full_tt_fp8e8m0<16 * C::HALF_SCALE_GROUPS>;
    using d_tt_v_sc = full_tt_fp8e8m0<16 * C::HALF_SCALE_GROUPS>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = C::NTile;
    constexpr int A_SC_BASE = OUTPUT_BASE + C::Dvo;
    constexpr int K_SC_BASE = A_SC_BASE + 16 * C::K_SCALE_CHUNKS;
    constexpr int P_SC_BASE = K_SC_BASE + 16 * C::HALF_SCALE_GROUPS;
    constexpr int V_SC_BASE = P_SC_BASE + 16 * C::HALF_SCALE_GROUPS;
    static_assert(V_SC_BASE + 16 * C::HALF_SCALE_GROUPS <= 512,
                  "true half TMEM map must fit score+output+scales");

    const int row_tile = blockIdx.x;
    const int row = row_tile * C::Mb + warpgroup::laneid();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_a_sc a_sc_tm = tm_alloc.template allocate<d_tt_a_sc>(A_SC_BASE);
    d_tt_k_sc k_sc_tm = tm_alloc.template allocate<d_tt_k_sc>(K_SC_BASE);
    d_tt_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE);
    d_tt_v_sc v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_fp4_tile &a_smem = al.allocate<typename G::a_fp4_tile>();
    typename G::k_half_fp4_tile &k_smem = al.allocate<typename G::k_half_fp4_tile>();
    typename G::p_half_fp4_tile &p_smem = al.allocate<typename G::p_half_fp4_tile>();
    typename G::v_half_fp4_tile &v_smem = al.allocate<typename G::v_half_fp4_tile>();
    auto (&a_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&k_sc_smem)[C::HALF_SCALE_GROUPS] = al.allocate<typename G::sc_tile, C::HALF_SCALE_GROUPS>();
    auto (&p_sc_smem)[C::HALF_SCALE_GROUPS] = al.allocate<typename G::sc_tile, C::HALF_SCALE_GROUPS>();
    auto (&v_sc_smem)[C::HALF_SCALE_GROUPS] = al.allocate<typename G::sc_tile, C::HALF_SCALE_GROUPS>();

    __shared__ semaphore qk_src_finished[C::HALVES];
    __shared__ semaphore qk_done[C::HALVES];
    __shared__ semaphore pv_src_finished[C::HALVES];
    __shared__ semaphore pv_done[C::HALVES];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int half = 0; half < C::HALVES; ++half) {
            init_semaphore(qk_src_finished[half], 0, 1);
            init_semaphore(qk_done[half], 0, 1);
            init_semaphore(pv_src_finished[half], 0, 1);
            init_semaphore(pv_done[half], 0, 1);
        }
    }
    __syncthreads();

    load_mxfp4_shape_a_tile<C>(a_smem, g, row_tile);
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
        load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
    }
    __syncthreads();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0);
    const bool issuer = issuer_warp && warp::elect_leader();
    if (issuer_warp) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
            fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
        }
        tensor_load_wait();
    }
    __syncthreads();

    #pragma unroll
    for (int half = 0; half < C::HALVES; ++half) {
        load_mxfp4_joint_b_half_tile<C>(k_smem, g, half);
        #pragma unroll
        for (int group = 0; group < C::HALF_SCALE_GROUPS; ++group) {
            const int q4_group = half * C::HALF_SCALE_GROUPS + group;
            const uint8_t *k_sc_src = &g.b_sc[((q4_group * g.b_scale_k_tiles) * 32 * 16)];
            load_mxfp4_shape_sc_tile(k_sc_smem[group], k_sc_src);
        }
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();

        if (issuer_warp) {
            #pragma unroll
            for (int group = 0; group < C::HALF_SCALE_GROUPS; ++group) {
                auto k_sc_tm_sub = k_sc_tm.template subtile<full_tt_fp8e8m0<16>>(group * 16);
                fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm_sub, k_sc_smem[group]);
            }
            tensor_load_wait();
        }
        __syncthreads();

        if (issuer) {
            tensor_before_thread_sync();
            fp4pv_mm_cluster_ABt<C>(tt_score, a_smem, k_smem, a_sc_tm, k_sc_tm, qk_src_finished[half]);
            tensor_commit<C::CLUSTER_SIZE>(qk_done[half]);
        }

        load_mxfp4_v_fp4_k256_tile_from_global_allthreads<C>(v_smem, g, 0, 0, half * C::HALF_SCALE_GROUPS, 0);
        #pragma unroll
        for (int group = 0; group < C::HALF_SCALE_GROUPS; ++group) {
            load_mxfp4_v_sc_from_global_allthreads<C>(v_sc_smem[group], g, 0, 0, half * C::HALF_SCALE_GROUPS + group);
        }

        if (threadIdx.x == 0) {
            wait(qk_done[half], 0);
        }
        __syncthreads();

        if (warpgroup::groupid() == 0 || warpgroup::groupid() == 1) {
            #pragma unroll
            for (int p_group = 0; p_group < C::HALF_SCALE_GROUPS; ++p_group) {
                float2 scores_half[32];
                const int quant_wg = warpgroup::groupid();
                const int score_half = p_group * 2 + quant_wg;
                fp4pv_load_score_half_tmem(scores_half, tt_score.addr, score_half);
                tensor_load_wait();
                const int packed_col_offset = p_group * (C::Nb / 2);
                if (quant_wg == 0) {
                    fp4pv_pack_scores_half_to_stage_mxfp4_range<0, 2>(
                        p_smem, p_sc_smem[p_group], scores_half, row, 0, packed_col_offset);
                } else {
                    fp4pv_pack_scores_half_to_stage_mxfp4_range<2, 4>(
                        p_smem, p_sc_smem[p_group], scores_half, row, 0, packed_col_offset);
                }
            }
        }
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();

        if constexpr (C::PIPELINE_NEXT_HALF_UNDER_PV) {
            if (half > 0) {
                if (threadIdx.x == 0) {
                    wait(pv_done[half - 1], 0);
                }
                __syncthreads();
            }
        }

        if (issuer_warp) {
            #pragma unroll
            for (int group = 0; group < C::HALF_SCALE_GROUPS; ++group) {
                auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e8m0<16>>(group * 16);
                auto v_sc_tm_sub = v_sc_tm.template subtile<full_tt_fp8e8m0<16>>(group * 16);
                fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_smem[group]);
                fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm_sub, v_sc_smem[group]);
            }
            tensor_load_wait();
        }
        if constexpr (!C::TRIM_PV_ISSUER_SYNCS) {
            __syncthreads();
        }

        if (issuer) {
            tensor_before_thread_sync();
            if (half == 0) {
                fp4pv_mm_cluster_ABt<C>(tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[half]);
            } else {
                fp4pv_mma_cluster_ABt<C>(tt_output, p_smem, v_smem, p_sc_tm, v_sc_tm, pv_src_finished[half]);
            }
            tensor_commit<C::CLUSTER_SIZE>(pv_done[half]);
        }
        if constexpr (!C::TRIM_PV_ISSUER_SYNCS) {
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            if constexpr (C::PIPELINE_NEXT_HALF_UNDER_PV) {
                if (half + 1 < C::HALVES) {
                    wait(pv_src_finished[half], 0);
                } else {
                    wait(pv_done[half], 0);
                }
            } else {
                wait(pv_done[half], 0);
            }
        }
        __syncthreads();
    }

    if (warpgroup::groupid() == 2) {
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        if (row >= g.rows) {
            return;
        }
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
        }
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_joint_qk_pv_quarter_handoff_debug(
    const __grid_constant__ globals_mxfp4_joint_qk_pv_quarter_handoff_debug<C> g
) {
    static_assert(C::QUARTERS == 4, "handoff diagnostic currently uses four quarters");
    static_assert(C::HOIST_A_SCALE_TMEM, "handoff diagnostic compares against opt2");
    using G = globals_mxfp4_joint_qk_pv_quarter_handoff_debug<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_a_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_b_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = C::Nb;
    constexpr int A_SC_BASE = OUTPUT_BASE + C::Dvo;
    constexpr int B_SC_BASE = A_SC_BASE + 16 * C::K_SCALE_CHUNKS;
    static_assert(B_SC_BASE + 16 * C::K_SCALE_CHUNKS <= 512,
                  "joint QK/PV handoff TMEM map must fit score+output+scales");

    const int row_tile = blockIdx.x;
    const int row = row_tile * C::Mb + warpgroup::laneid();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output = tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_a_sc a_sc_tm = tm_alloc.template allocate<d_tt_a_sc>(A_SC_BASE);
    d_tt_b_sc b_sc_tm = tm_alloc.template allocate<d_tt_b_sc>(B_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_fp4_tile &a_smem = al.allocate<typename G::a_fp4_tile>();
    typename G::b_fp4_tile &b_smem = al.allocate<typename G::b_fp4_tile>();
    auto (&a_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&b_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();

    __shared__ semaphore qk_src_finished[C::QUARTERS];
    __shared__ semaphore qk_done[C::QUARTERS];
    __shared__ semaphore pv_src_finished[C::QUARTERS];
    __shared__ semaphore pv_done[C::QUARTERS];
    __shared__ int handoff_ready;
    __shared__ int handoff_consumed;
    __shared__ int handoff_complete;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
            init_semaphore(qk_src_finished[quarter], 0, 1);
            init_semaphore(qk_done[quarter], 0, 1);
            init_semaphore(pv_src_finished[quarter], 0, 1);
            init_semaphore(pv_done[quarter], 0, 1);
        }
        handoff_ready = -1;
        handoff_consumed = -1;
        handoff_complete = -1;
    }
    __syncthreads();

    load_mxfp4_shape_a_tile<C>(a_smem, g, row_tile);
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
        load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
    }
    __syncthreads();

    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0);
    const bool issuer = issuer_warp && warp::elect_leader();
    const bool producer_wg = (warpgroup::groupid() == 0);
    const bool full_cta_control = (g.control_mode == 0);
    const bool issuer_proxy_fence = (g.control_mode == 2 || g.control_mode == 3 || g.control_mode == 4);
    const bool producer_proxy_fence = (g.control_mode == 1 || g.control_mode == 3 || g.control_mode == 4);
    const bool consumed_after_done = (g.control_mode == 3);

    if (issuer_warp) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
            fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
        }
        tensor_load_wait();
    }
    __syncthreads();

    #pragma unroll
    for (int quarter = 0; quarter < C::QUARTERS; ++quarter) {
        if (full_cta_control) {
            load_mxfp4_joint_b_quarter_tile<C>(b_smem, g, quarter);
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                const uint8_t *b_sc_src = &g.b_sc[((quarter * g.b_scale_k_tiles + k) * 32 * 16)];
                load_mxfp4_shape_sc_tile(b_sc_smem[k], b_sc_src);
            }
            __syncthreads();
            fp4pv_shared_async_proxy_fence_cta();
            __syncthreads();
        } else {
            if (producer_wg) {
                load_mxfp4_joint_b_quarter_tile_producer_wg<C>(b_smem, g, quarter);
                #pragma unroll
                for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                    const uint8_t *b_sc_src = &g.b_sc[((quarter * g.b_scale_k_tiles + k) * 32 * 16)];
                    load_mxfp4_shape_sc_tile_producer_wg(b_sc_smem[k], b_sc_src);
                }
                warpgroup::sync(warpgroup::groupid() + 1);
                if (threadIdx.x == 0) {
                    fp4pv_shared_flag_publish(&handoff_ready, quarter, producer_proxy_fence);
                }
                fp4pv_shared_flag_wait_eq(&handoff_consumed, quarter, false);
            }

            if (issuer_warp) {
                fp4pv_shared_flag_wait_eq(&handoff_ready, quarter, issuer_proxy_fence);
            }
        }

        if (issuer_warp) {
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                auto b_sc_tm_sub = b_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                fp4pv_load_mxnv_scale_async_cluster<C>(b_sc_tm_sub, b_sc_smem[k]);
            }
            tensor_load_wait();
        }
        if (full_cta_control) {
            __syncthreads();
        }

        if (issuer) {
            tensor_before_thread_sync();
            fp4pv_mm_cluster_ABt<C>(tt_score, a_smem, b_smem, a_sc_tm, b_sc_tm, qk_src_finished[quarter]);
            tensor_commit<C::CLUSTER_SIZE>(qk_done[quarter]);
            tensor_before_thread_sync();
            if (quarter == 0) {
                fp4pv_mm_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, pv_src_finished[quarter]);
            } else {
                fp4pv_mma_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, pv_src_finished[quarter]);
            }
            tensor_commit<C::CLUSTER_SIZE>(pv_done[quarter]);
        }

        if (full_cta_control) {
            __syncthreads();
            if (threadIdx.x == 0) {
                wait(qk_done[quarter], 0);
                wait(pv_done[quarter], 0);
            }
            __syncthreads();
        } else if (issuer) {
            wait(qk_src_finished[quarter], 0);
            wait(pv_src_finished[quarter], 0);
            if (!consumed_after_done) {
                fp4pv_shared_flag_publish(&handoff_consumed, quarter, false);
            }
            wait(qk_done[quarter], 0);
            wait(pv_done[quarter], 0);
            if (consumed_after_done) {
                fp4pv_shared_flag_publish(&handoff_consumed, quarter, false);
            }
            fp4pv_shared_flag_publish(&handoff_complete, quarter, false);
        }
    }

    if (warpgroup::groupid() == 2) {
        if (!full_cta_control) {
            fp4pv_shared_flag_wait_eq(&handoff_complete, C::QUARTERS - 1, false);
        }
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
        if (row >= g.rows) {
            return;
        }
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            store_mxfp4_joint_output_col_vec<true, C>(tt_output, g, row, col);
        }
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_mxfp4_tcgen_shape_debug(
    const __grid_constant__ globals_mxfp4_tcgen_shape_debug<C> g
) {
    using G = globals_mxfp4_tcgen_shape_debug<C>;
    using d_tt_outputs = tt<float, C::Mb, C::ACC_DVO>;
    using d_tt_a_sc = full_tt_fp8e8m0<16 * C::K_SCALE_CHUNKS>;
    using d_tt_b_sc = full_tt_fp8e8m0<32 * C::K_SCALE_CHUNKS>;
    constexpr int MAX_TENSOR_COLS_LOCAL = 512;
    constexpr int A_SC_BASE = (C::SCALE_LAYOUT == 0) ? C::ACC_DVO : 256;
    constexpr int B_SC_BASE = (C::SCALE_LAYOUT == 0)
        ? (A_SC_BASE + 16 * C::K_SCALE_CHUNKS)
        : (A_SC_BASE + 4 * C::K_SCALE_CHUNKS);
    constexpr int B_SC_END = B_SC_BASE + 32 * C::K_SCALE_CHUNKS;
    static_assert(B_SC_END <= MAX_TENSOR_COLS_LOCAL,
                  "shape debug output plus MXFP4 scale TMEM exceeds 512 tensor columns");

    const int cta_rank = cluster_ctarank();
    const int cluster_linear = blockIdx.x / C::CLUSTER_SIZE;
    const int row_tile = cluster_linear * C::CLUSTER_SIZE + cta_rank;
    const int row_tiles = g.rows / C::Mb;
    if (row_tile >= row_tiles) {
        return;
    }

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(0);
    d_tt_a_sc a_sc_tm = tm_alloc.template allocate<d_tt_a_sc>(A_SC_BASE);
    d_tt_b_sc b_sc_tm = tm_alloc.template allocate<d_tt_b_sc>(B_SC_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_fp4_tile &a_smem = al.allocate<typename G::a_fp4_tile>();
    typename G::b_fp4_tile &b_smem = al.allocate<typename G::b_fp4_tile>();
    auto (&a_sc_smem)[C::K_SCALE_CHUNKS] = al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS>();
    auto (&b_sc_smem)[C::K_SCALE_CHUNKS * C::B_SCALE_GROUPS] =
        al.allocate<typename G::sc_tile, C::K_SCALE_CHUNKS * C::B_SCALE_GROUPS>();

    __shared__ semaphore src_finished[C::OUT_PHASES];
    __shared__ semaphore tile_arrived[C::OUT_PHASES];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int phase = 0; phase < C::OUT_PHASES; ++phase) {
            init_semaphore(src_finished[phase], 0, 1);
            init_semaphore(tile_arrived[phase], 0, 1);
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (g.debug_stage == 1) {
        return;
    }

    load_mxfp4_shape_a_tile<C>(a_smem, g, row_tile);
    load_mxfp4_shape_b_tile<C>(b_smem, g, cta_rank, 0);
    __syncthreads();
    if (g.debug_stage == 2) {
        return;
    }
    if (g.debug_stage == 8) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles) * 32 * 16];
        load_mxfp4_shape_sc_tile_first_word(a_sc_smem[0], a_sc_src);
        __syncthreads();
        return;
    }
    if (g.debug_stage == 9) {
        const uint8_t *b_sc_src = &g.b_sc[0];
        load_mxfp4_shape_sc_tile_first_word(b_sc_smem[0], b_sc_src);
        __syncthreads();
        return;
    }
    if (g.debug_stage == 10) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
            load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
        }
        __syncthreads();
        return;
    }
    if (g.debug_stage == 11) {
        #pragma unroll
        for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
            #pragma unroll
            for (int n_group = 0; n_group < C::B_SCALE_GROUPS; ++n_group) {
                const uint8_t *b_sc_src = &g.b_sc[((n_group * g.b_scale_k_tiles + k) * 32 * 16)];
                load_mxfp4_shape_sc_tile(b_sc_smem[k * C::B_SCALE_GROUPS + n_group], b_sc_src);
            }
        }
        __syncthreads();
        return;
    }
    #pragma unroll
    for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
        const uint8_t *a_sc_src = &g.a_sc[(row_tile * g.a_scale_k_tiles + k) * 32 * 16];
        load_mxfp4_shape_sc_tile(a_sc_smem[k], a_sc_src);
        #pragma unroll
        for (int n_group = 0; n_group < C::B_SCALE_GROUPS; ++n_group) {
            const uint8_t *b_sc_src = &g.b_sc[((n_group * g.b_scale_k_tiles + k) * 32 * 16)];
            load_mxfp4_shape_sc_tile(b_sc_smem[k * C::B_SCALE_GROUPS + n_group], b_sc_src);
        }
    }
    __syncthreads();
    if (g.debug_stage == 6) {
        return;
    }
    const bool issuer_warp = (warpgroup::groupid() == 3) && (warpgroup::warpid() == 0) && (cta_rank == 0);
    const bool issuer = issuer_warp && warp::elect_leader();
    #pragma unroll
    for (int out_phase = 0; out_phase < C::OUT_PHASES; ++out_phase) {
        if constexpr (C::OUT_PHASES > 1) {
            load_mxfp4_shape_b_tile<C>(b_smem, g, cta_rank, out_phase);
            __syncthreads();
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        publish_cluster_shared_backing_if_needed<C>();
        __syncthreads();
        if (g.debug_stage == 7) {
            return;
        }
        fp4pv_cluster_sync<C>();
        if (g.debug_stage == 3) {
            return;
        }

        if (issuer_warp) {
            #pragma unroll
            for (int k = 0; k < C::K_SCALE_CHUNKS; ++k) {
                auto a_sc_tm_sub = a_sc_tm.template subtile<full_tt_fp8e8m0<16>>(k * 16);
                fp4pv_load_mxnv_scale_async_cluster<C>(a_sc_tm_sub, a_sc_smem[k]);
                #pragma unroll
                for (int n_group = 0; n_group < C::ACTIVE_B_SCALE_GROUPS; ++n_group) {
                    constexpr int ACTIVE_B_SCALE_GROUPS = C::ACTIVE_B_SCALE_GROUPS;
                    const int src_group = out_phase * ACTIVE_B_SCALE_GROUPS + n_group;
                    auto b_sc_tm_sub = b_sc_tm.template subtile<full_tt_fp8e8m0<16>>(
                        k * ACTIVE_B_SCALE_GROUPS * 16 + n_group * 16);
                    fp4pv_load_mxnv_scale_async_cluster<C>(
                        b_sc_tm_sub, b_sc_smem[k * C::B_SCALE_GROUPS + src_group]);
                }
            }
            tensor_load_wait();
        }
        __syncthreads();
        if (g.debug_stage == 4) {
            return;
        }
        if (issuer) {
            tensor_before_thread_sync();
            fp4pv_mm_cluster_ABt<C>(tt_output, a_smem, b_smem, a_sc_tm, b_sc_tm, src_finished[out_phase]);
            tensor_commit<C::CLUSTER_SIZE>(tile_arrived[out_phase]);
        }
        __syncthreads();
        if (g.debug_stage == 5) {
            if (threadIdx.x == 0) {
                wait(tile_arrived[out_phase], 0);
            }
            __syncthreads();
            return;
        }

        if (warpgroup::groupid() == 2) {
            wait(tile_arrived[out_phase], 0);
            warpgroup::tma::store_async_read_wait<0>();
            warpgroup::sync(warpgroup::groupid() + 1);
            const int row = row_tile * C::Mb + warpgroup::laneid();
            if (row >= g.rows) {
                return;
            }
            constexpr float MXFP4_GEMM_SCALE_CORRECTION = 1.0f / 36.0f;
            #pragma unroll
            for (int col = 0; col < C::ACC_DVO; col += 16) {
                float2 o_reg[8];
                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                      "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                      "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                      "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                    : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                tensor_load_wait();
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o_reg[i].x *= MXFP4_GEMM_SCALE_CORRECTION;
                    o_reg[i].y *= MXFP4_GEMM_SCALE_CORRECTION;
                    bf16_2 packed = __float22bfloat162_rn(o_reg[i]);
                    *reinterpret_cast<bf16_2 *>(
                        &g.o_direct[(row * C::Dvo) + out_phase * C::ACC_DVO + col + i * 2]) = packed;
                }
            }
        }
        __syncthreads();
        fp4pv_cluster_sync<C>();
    }
}

enum debug_stage_cut_mode : int {
    DEBUG_STAGE_CUT_FULL = 0,
    DEBUG_STAGE_CUT_ISSUE3_ONLY = 1,
    DEBUG_STAGE_CUT_CONSUME3_COPY = 2,
    DEBUG_STAGE_CUT_CONSUME3_ROWMAX = 3,
    DEBUG_STAGE_CUT_CONSUME3_EXP = 4,
    DEBUG_STAGE_CUT_CONSUME3_SCAN = 5,
    DEBUG_STAGE_CUT_PRODUCER3_POST_WAIT = 6,
    DEBUG_STAGE_CUT_PRODUCER3_POST_ISSUE = 7,
    DEBUG_STAGE_CUT_PRODUCER3_POST_COMMIT = 8,
    DEBUG_STAGE_CUT_PRODUCER3_POST_KSC_FINISH = 9,
    DEBUG_STAGE_CUT_CONSUME3_WAIT_ONLY = 10,
    DEBUG_STAGE_CUT_PRODUCER3_SKIP_COMMIT_TAIL = 11,
    DEBUG_STAGE_CUT_PRODUCER3_FRESH_COMMIT_TAIL = 12,
    DEBUG_STAGE_CUT_FRESH3_WAIT_ONLY = 13,
    DEBUG_STAGE_CUT_FRESH3_COPY = 14,
    DEBUG_STAGE_CUT_FRESH3_FULL = 15,
    DEBUG_STAGE_CUT_DUAL_SLOT_FULL = 16,
    DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL = 17,
    DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY = 100,
    DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY = 101,
    DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY = 102,
    DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE = 103,
    DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE = 104,
    DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY = 105,
    DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY = 106,
    DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE = 107,
    DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE = 108,
    DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM = 109,
    DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM = 110,
    DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL = 111,
};

enum debug_q_scale_mode : int {
    DEBUG_Q_SCALE_MODE_TMEM = 0,
    DEBUG_Q_SCALE_MODE_SMEM_CHUNKED = 1,
};

enum debug_tmem_layout_mode : int {
    DEBUG_TMEM_LAYOUT_COMPACT = 0,
    DEBUG_TMEM_LAYOUT_PRODUCTION_LIKE = 1,
};

enum debug_k_scale_depth_mode : int {
    DEBUG_K_SCALE_DEPTH_MODE_TILE128 = 0,
    DEBUG_K_SCALE_DEPTH_MODE_SLAB64 = 1,
};

enum debug_scale_probe_stage_mode : int {
    DEBUG_SCALE_PROBE_STAGE_SOURCE_ONLY = 0,
    DEBUG_SCALE_PROBE_STAGE_Q_LOAD_SYNC = 1,
    DEBUG_SCALE_PROBE_STAGE_Q_RT_LOAD_ONLY = 2,
    DEBUG_SCALE_PROBE_STAGE_Q_COPY_SHARED_ONLY = 3,
    DEBUG_SCALE_PROBE_STAGE_Q_LOAD_COPY = 4,
    DEBUG_SCALE_PROBE_STAGE_K_LOAD_SYNC = 5,
    DEBUG_SCALE_PROBE_STAGE_K_LOAD_COPY = 6,
    DEBUG_SCALE_PROBE_STAGE_FULL = 7,
    DEBUG_SCALE_PROBE_STAGE_Q_DIRECT_DUMP = 8,
    DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP = 9,
};

enum live_dp_debug_mode : int {
    LIVE_DP_DEBUG_FULL = 0,
    LIVE_DP_DEBUG_PROFILE_TC_HANDSHAKE_ZERO_DP = 21,
    LIVE_DP_DEBUG_X8_WAIT = 22,
    LIVE_DP_DEBUG_X8_ALIGNED_PROBE = 23,
    LIVE_DP_DEBUG_X16_WAIT = 24,
};

template <typename G>
__device__ __forceinline__ bool live_dp_mode_consumes_tmem(const G &g) {
    return g.live_dp_debug_mode == LIVE_DP_DEBUG_FULL ||
           g.live_dp_debug_mode == 13 ||
           g.live_dp_debug_mode == 18 ||
           g.live_dp_debug_mode == 19 ||
           g.live_dp_debug_mode == 20 ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_WAIT ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_ALIGNED_PROBE ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X16_WAIT ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_PROFILE_TC_HANDSHAKE_ZERO_DP;
}

template <typename G>
__device__ __forceinline__ bool live_dp_mode_applies_values(const G &g) {
    return g.live_dp_debug_mode == LIVE_DP_DEBUG_FULL ||
           g.live_dp_debug_mode == 13 ||
           g.live_dp_debug_mode == 18 ||
           g.live_dp_debug_mode == 19 ||
           g.live_dp_debug_mode == 20 ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_WAIT ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_ALIGNED_PROBE ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X16_WAIT;
}

template <typename G>
__device__ __forceinline__ bool live_dp_mode_uses_scaled_mma_values(const G &g) {
    return g.live_dp_debug_mode != 13;
}

template <typename C>
struct globals_quant_p_from_scores_debug {
    using q_tile = typename globals_fp4pv<C>::q_tile;
    using q_sc_tile = typename globals_fp4pv<C>::q_sc_tile;
    using k_tile = typename globals_fp4pv<C>::k_tile;
    using k_sc_tile = typename globals_fp4pv<C>::k_sc_tile;
    using p_fp4_tile = typename globals_fp4pv<C>::p_fp4_tile;
    using p_sc_tile = typename globals_fp4pv<C>::p_sc_tile;
    using bwd_dq_k_fp4_tile = std::conditional_t<
        C::CLUSTER_SIZE == 1,
        st_fp4e2m1_2<C::Dvo, C::Nb / 2>,
        typename globals_fp4pv<C>::v_fp4_tile>;
    using bwd_dq_sc_tile = st_fp8e8m0<32, 16, false>;
    using q_sc_smem_storage = typename globals_fp4pv<C>::q_sc_smem_storage;
    using k_sc_smem_storage = typename globals_fp4pv<C>::k_sc_smem_storage;
    using k_sc_guard_slot = typename globals_fp4pv<C>::k_sc_guard_slot;
    using p_sc_guard_slot = typename globals_fp4pv<C>::p_sc_guard_slot;
    using live_dp_dout_fp4_tile = st_fp4e2m1_2<C::Mb, C::Dvo / 2>;
    using live_dp_v_row_fp4_tile = st_fp4e2m1_2<C::Nb / 2, C::Dvo / 2>;
    using live_dp_sc_tile = st_fp8e8m0<32, 16, false>;

    using q_gl = typename globals_fp4pv<C>::q_gl;
    using q_sc_gl = typename globals_fp4pv<C>::q_sc_gl;
    using q_sg_gl = typename globals_fp4pv<C>::q_sg_gl;
    using k_gl = typename globals_fp4pv<C>::k_gl;
    using k_sc_gl = typename globals_fp4pv<C>::k_sc_gl;
    using k_sg_gl = typename globals_fp4pv<C>::k_sg_gl;
    using lse_gl = typename globals_fp4pv<C>::lse_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_tile) * C::NUM_SOFTMAXXERS);
        add((int)sizeof(q_sc_smem_storage));
        add((int)sizeof(k_tile) * C::LOAD_STAGES);
        add((int)sizeof(k_sc_smem_storage));
        add((int)sizeof(k_sc_guard_slot));
        add((int)sizeof(fp4pv_p_scan3d));
        add((int)sizeof(st_fp8e4m3<128, 32, false>));
        add((int)sizeof(p_fp4_tile) * 2);
        add((int)sizeof(p_sc_tile) * 2);
        add((int)sizeof(bwd_dq_k_fp4_tile) * 2);
        add((int)sizeof(bwd_dq_sc_tile) * 2);
        add((int)sizeof(bwd_dq_k_fp4_tile) * 2);
        add((int)sizeof(bwd_dq_sc_tile) * 2);
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(bwd_dq_k_fp4_tile));
        add((int)sizeof(p_sc_guard_slot));
        return bytes;
    }

    static constexpr int compute_dynamic_shared_memory_live_dp() {
        int bytes = compute_dynamic_shared_memory();
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(live_dp_dout_fp4_tile));
        add((int)sizeof(live_dp_sc_tile));
        add((int)sizeof(live_dp_v_row_fp4_tile) * 2);
        add((int)sizeof(live_dp_sc_tile) * 2);
        bytes += 4096;
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();
    static constexpr int DYNAMIC_SHARED_MEMORY_LIVE_DP = compute_dynamic_shared_memory_live_dp();

    q_gl q;
    q_sc_gl q_sc;
    q_sg_gl q_sg;
    k_gl k;
    k_sc_gl k_sc;
    k_sg_gl k_sg;
    uint8_t *p_fp4;
    uint8_t *p_sc_prepared;
    uint8_t *p_sc_rowmajor_live;
    uint8_t *p_sc_rowmajor_scan;
    bf16 *p_bf16_live;
    lse_gl lse;
    float *debug_row_max;
    float *debug_row_sum;
    float *debug_tile_amax;
    int selected_row_tile;
    int tile_begin;
    int tiles_to_run;
    int max_k_iters = -1;
    int selected_row_tile_span = 1;
    int compute_k_iter_base = 0;
    int output_full_layout = 0;
    int disable_lse_store = 0;
    int use_external_lse = 0;
    int debug_stage_cut = DEBUG_STAGE_CUT_FULL;
    int debug_q_scale_mode = DEBUG_Q_SCALE_MODE_TMEM;
    int debug_tmem_layout_mode = DEBUG_TMEM_LAYOUT_COMPACT;
    int debug_k_scale_depth_mode = DEBUG_K_SCALE_DEPTH_MODE_TILE128;
    int debug_qk_chunk_mask = (1 << C::QK_SCALE_CHUNKS) - 1;
    float *debug_tile_score_max = nullptr;
    float *debug_tile_exp_sum = nullptr;
    bf16 *debug_scores_bf16_live = nullptr;
    bf16 *debug_scores_bf16_scan_repacked = nullptr;
    bf16 *debug_scores_bf16_tmem_x16 = nullptr;
    uint8_t *debug_q_sc_loaded = nullptr;
    uint8_t *debug_k_sc_loaded = nullptr;
    uint8_t *debug_q_sc_tmem_loaded = nullptr;
    uint8_t *debug_k_sc_tmem_loaded = nullptr;
    bf16 *backward_dp_bf16 = nullptr;
    const uint8_t *backward_dout_row_fp4 = nullptr;
    const uint8_t *backward_dout_row_sc = nullptr;
    const uint8_t *backward_v_row_fp4 = nullptr;
    const uint8_t *backward_v_row_sc = nullptr;
    int live_dp_debug_mode = 0;
    int live_dp_selected_head = -1;
    float *backward_delta = nullptr;
    float backward_softmax_scale = 1.0f;
    int emit_backward_ds = 0;
    uint8_t *backward_ds_fp4_row = nullptr;
    uint8_t *backward_ds_sc_row = nullptr;
    uint8_t *backward_ds_fp4_col = nullptr;
    uint8_t *backward_ds_sc_col = nullptr;
    int emit_backward_ds_row_mxfp4 = 0;
    int emit_backward_ds_col_mxfp4 = 0;
    int backward_ds_row_skip_future_zero = 0;
    int backward_ds_col_skip_future_zero = 0;
    int backward_skip_global_ds_row_store = 0;
    int backward_ds_quant_mode = 0;
    uint32_t *backward_fused_stage_marker = nullptr;
    int backward_fused_stage_transpose = 0;
    const uint8_t *backward_dq_k_col_fp4 = nullptr;
    const uint8_t *backward_dq_k_col_sc = nullptr;
    bf16 *backward_dq_pad = nullptr;
    bf16 *backward_dq_bshd = nullptr;
    int backward_dq_half = 0;

    __host__ __inline__ int total_bids() {
        const int head_work = live_dp_selected_head >= 0 ? 1 : q.depth();
        if (selected_row_tile >= 0) {
            return q.batch() * head_work * C::CLUSTER_SIZE;
        }
        const int tiles_m = q.rows() / C::Mb;
        const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
        const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
        return q.batch() * head_work * num_block * C::CLUSTER_SIZE;
    }
    __host__ __inline__ dim3 grid() {
        const int total_bids = this->total_bids();
        constexpr int PERSISTENT_GRID_CAP = fp4pv_persistent_grid_cap<C>();
        int grid_x = total_bids < PERSISTENT_GRID_CAP ? total_bids : PERSISTENT_GRID_CAP;
        if (grid_x < C::CLUSTER_SIZE) {
            grid_x = C::CLUSTER_SIZE;
        } else {
            grid_x = ((grid_x + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        }
        return dim3(grid_x);
    }
    __host__ __inline__ dim3 grid_exact() {
        int grid_x = this->total_bids();
        if (grid_x < C::CLUSTER_SIZE) {
            grid_x = C::CLUSTER_SIZE;
        }
        return dim3(grid_x);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
    __host__ __inline__ int dynamic_shared_memory_live_dp() { return DYNAMIC_SHARED_MEMORY_LIVE_DP; }
};

template <typename C, typename BTile>
__device__ inline void load_bwd_dq_k_col_fp4_tile(
    BTile &dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int dq_half_override = -1
) {
    if (g.backward_dq_k_col_fp4 == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int BYTES_PER_STORE = 16;
    static_assert(BTile::cols >= C::Nb / 2, "dQ K-col tile must hold one packed K128 slice");
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "packed K dimension must be vector-loadable");
    static_assert(BTile::swizzle_bytes >= BYTES_PER_STORE, "dQ K-col tile swizzle must support vector stores");
    const int packed_seqlen = g.q.rows() / 2;
    const int rows_per_cta = C::Dvo / C::CLUSTER_SIZE;
    const int dq_half = dq_half_override >= 0 ? dq_half_override : g.backward_dq_half;
    const int row_base = dq_half * C::Dvo + cta_rank * rows_per_cta;
    const int col_base = k_iter * (C::Nb / 2);
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    const int total_stores = rows_per_cta * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < total_stores; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear - row * STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.backward_dq_k_col_fp4[((((b * g.q.depth() + h) * DQ_PAD) + row_base + row) * packed_seqlen) + col_base + col]);
        const uint32_t smem_addr = __cvta_generic_to_shared(dst.idx(&dst.data[0], {row, col}));
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(smem_addr),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BTile>
__device__ inline void load_bwd_dq_k_col_fp4_tail64_tile(
    BTile &dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int k_iter,
    int cta_rank
) {
    if (g.backward_dq_k_col_fp4 == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int BYTES_PER_STORE = 16;
    constexpr int TAIL_HALF = 1;
    constexpr int TAIL_COLS = C::Dvo / 2;
    constexpr int ROWS_PER_CTA = TAIL_COLS / C::CLUSTER_SIZE;
    static_assert(C::CLUSTER_SIZE == 2, "dQ tail64 loader is only for the cluster-size-2 production path");
    static_assert(C::Dvo == 128 && C::Dqk == 192, "dQ tail64 loader assumes a 64-column upper tail");
    static_assert(BTile::rows >= ROWS_PER_CTA, "dQ tail64 tile must hold the local output rows");
    static_assert(BTile::cols >= C::Nb / 2, "dQ tail64 K-col tile must hold one packed K128 slice");
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "packed K dimension must be vector-loadable");
    static_assert(BTile::swizzle_bytes >= BYTES_PER_STORE, "dQ tail64 K-col tile swizzle must support vector stores");
    const int packed_seqlen = g.q.rows() / 2;
    const int row_base = TAIL_HALF * C::Dvo + cta_rank * ROWS_PER_CTA;
    const int col_base = k_iter * (C::Nb / 2);
    const uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = ROWS_PER_CTA * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear - row * STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.backward_dq_k_col_fp4[((((b * g.q.depth() + h) * DQ_PAD) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(BTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BScTile>
__device__ inline void load_bwd_dq_k_col_sc_tile(
    BScTile &dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int k_iter,
    int dq_half_override = -1
) {
    if (g.backward_dq_k_col_sc == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int BYTES_PER_STORE = 16;
    static_assert(sizeof(BScTile) == 32 * 16, "dQ K-col MXFP4 scale tile must be one 32x16 E8M0 tile");
    static_assert(sizeof(BScTile) % BYTES_PER_STORE == 0, "scale tile must be vector-loadable");
    const int scale_tiles_k = g.q.rows() / C::Nb;
    const int scale_n_tile = dq_half_override >= 0 ? dq_half_override : g.backward_dq_half;
    const int base_offset =
        ((((b * g.q.depth() + h) * (DQ_PAD / C::Dvo) + scale_n_tile) * scale_tiles_k + k_iter) * 32 * 16);
    const uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int TOTAL_STORES = (32 * 16) / BYTES_PER_STORE;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int byte_offset = linear * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(&g.backward_dq_k_col_sc[base_offset + byte_offset]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(smem_base + byte_offset),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, int DQ_HALF_OVERRIDE = -1, typename OutputTT>
__device__ inline void store_bwd_dq_pad_tmem_tile(
    OutputTT &tt_output,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile
) {
    if (g.backward_dq_pad == nullptr && g.backward_dq_bshd == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int DQ_OUT = 192;
    const int row = warpgroup::laneid();
    const int global_row = m_tile * C::Mb + row;
    const int dq_half = DQ_HALF_OVERRIDE >= 0 ? DQ_HALF_OVERRIDE : g.backward_dq_half;
    if (global_row >= g.q.rows()) {
        return;
    }
    constexpr int static_tail_cols = DQ_OUT - DQ_HALF_OVERRIDE * C::Dvo;
    constexpr int static_valid_cols =
        static_tail_cols <= 0 ? 0 : (static_tail_cols < C::Dvo ? static_tail_cols : C::Dvo);
    const int valid_cols =
        DQ_HALF_OVERRIDE >= 0 ? static_valid_cols : max(0, min(C::Dvo, DQ_OUT - dq_half * C::Dvo));
    #pragma unroll
    for (int col = 0; col < valid_cols; col += 16) {
        float2 out_reg[8];
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(out_reg[0].x), "=f"(out_reg[0].y), "=f"(out_reg[1].x), "=f"(out_reg[1].y),
              "=f"(out_reg[2].x), "=f"(out_reg[2].y), "=f"(out_reg[3].x), "=f"(out_reg[3].y),
              "=f"(out_reg[4].x), "=f"(out_reg[4].y), "=f"(out_reg[5].x), "=f"(out_reg[5].y),
              "=f"(out_reg[6].x), "=f"(out_reg[6].y), "=f"(out_reg[7].x), "=f"(out_reg[7].y)
            : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
        tensor_load_wait();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            const int out_col = dq_half * C::Dvo + col + ii * 2;
            const bf16_2 packed = __float22bfloat162_rn(out_reg[ii]);
            if (g.backward_dq_bshd != nullptr) {
                const int64_t offset =
                    (((static_cast<int64_t>(b) * g.q.rows() + global_row) * g.q.depth() + h) * DQ_OUT) + out_col;
                *reinterpret_cast<bf16_2 *>(&g.backward_dq_bshd[offset]) = packed;
            } else {
                const int64_t offset =
                    (((static_cast<int64_t>(b) * g.q.depth() + h) * g.q.rows() + global_row) * DQ_PAD) + out_col;
                *reinterpret_cast<bf16_2 *>(&g.backward_dq_pad[offset]) = packed;
            }
        }
    }
}

template <typename C>
struct globals_loaded_qk_scale_chunk_debug {
    using q_sc_tile = typename globals_fp4pv<C>::q_sc_tile;
    using k_sc_tile = typename globals_fp4pv<C>::k_sc_tile;
    using q_sc_smem_storage = typename globals_fp4pv<C>::q_sc_smem_storage;
    using k_sc_smem_storage = typename globals_fp4pv<C>::k_sc_smem_storage;
    using k_sc_guard_slot = typename globals_fp4pv<C>::k_sc_guard_slot;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_sc_smem_storage));
        add((int)sizeof(k_sc_smem_storage));
        add((int)sizeof(k_sc_guard_slot));
        add(4 * (int)sizeof(st_fp8e4m3<32, 32, false>));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *q_sc_tile_bytes;
    const uint8_t *k_sc_tile_bytes;
    uint8_t *q_sc_source_chunk;
    uint8_t *k_sc_source_chunk;
    bf16 *q_sc_tmem_chunk;
    bf16 *k_sc_tmem_chunk;
    uint32_t *q_sc_tmem_raw_words;
    uint32_t *k_sc_tmem_raw_words;
    int chunk_idx;
    int stage_mode = DEBUG_SCALE_PROBE_STAGE_FULL;

    __host__ __inline__ dim3 grid() { return dim3(C::CLUSTER_SIZE); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
__device__ inline bool debug_tile_selected(const globals_quant_p_from_scores_debug<C> &g, int k_iter) {
    return k_iter >= g.tile_begin && k_iter < (g.tile_begin + g.tiles_to_run);
}

template <typename C>
__device__ inline int debug_window_k_iter(const globals_quant_p_from_scores_debug<C> &g, int k_iter) {
    return k_iter - g.tile_begin;
}

template <typename C>
__device__ inline int debug_selected_row_tile_span(const globals_quant_p_from_scores_debug<C> &g) {
    if (g.selected_row_tile < 0) {
        return g.q.rows() / C::Mb;
    }
    const int tiles_m = g.q.rows() / C::Mb;
    return min(max(g.selected_row_tile_span, 1), tiles_m - g.selected_row_tile);
}

template <typename C>
__device__ inline bool debug_output_row_selected(const globals_quant_p_from_scores_debug<C> &g, int global_row) {
    if (g.selected_row_tile < 0) {
        return true;
    }
    const int row_base = g.selected_row_tile * C::Mb;
    const int row_limit = row_base + debug_selected_row_tile_span<C>(g) * C::Mb;
    return global_row >= row_base && global_row < row_limit;
}

template <typename C>
__device__ inline int debug_output_rows(const globals_quant_p_from_scores_debug<C> &g) {
    if (g.output_full_layout) {
        return g.q.rows();
    }
    return g.selected_row_tile < 0 ? g.q.rows() : debug_selected_row_tile_span<C>(g) * C::Mb;
}

template <typename C>
__device__ inline int debug_output_row_index(const globals_quant_p_from_scores_debug<C> &g, int global_row) {
    if (g.output_full_layout) {
        return global_row;
    }
    return g.selected_row_tile < 0 ? global_row : (global_row - g.selected_row_tile * C::Mb);
}

template <typename C>
__device__ inline bool debug_output_m_tile_selected(const globals_quant_p_from_scores_debug<C> &g, int m_tile) {
    return g.selected_row_tile < 0 ||
           (m_tile >= g.selected_row_tile && m_tile < g.selected_row_tile + debug_selected_row_tile_span<C>(g));
}

template <typename C>
__device__ inline bool debug_use_q_scale_tmem(const globals_quant_p_from_scores_debug<C> &g) {
    return g.debug_q_scale_mode == DEBUG_Q_SCALE_MODE_TMEM;
}

template <typename C>
__device__ inline bool debug_qk_chunk_enabled(const globals_quant_p_from_scores_debug<C> &g, int chunk) {
    return ((g.debug_qk_chunk_mask >> chunk) & 1) != 0;
}

template <typename C>
__device__ inline int debug_first_enabled_qk_chunk(const globals_quant_p_from_scores_debug<C> &g) {
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        if (debug_qk_chunk_enabled<C>(g, chunk)) {
            return chunk;
        }
    }
    return 0;
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTile, typename KScTile, typename QScTT, typename KScTT>
__device__ inline void issue_qk_chunked_debug_masked(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    const QScTile &q_sc_smem,
    const KScTile &k_sc_smem,
    QScTT &q_sc_tm,
    KScTT &k_sc_tm,
    semaphore &finished,
    const globals_quant_p_from_scores_debug<C> &g
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N_PACKED = KTile::rows * C::CLUSTER_SIZE;
    constexpr int N = (fp4pv_online_qk_logical_n<C>::value && std::is_same_v<T_AB, fp4e2m1_2>)
        ? (N_PACKED * 2)
        : N_PACKED;
    static_assert(!fp4pv_online_qk_logical_n<C>::value ||
                  (std::is_same_v<T_AB, fp4e2m1_2> && N == ScoresTT::cols),
        "QK logical-N probe expects FP4-packed K rows and score columns to match the logical N");
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    bool first_issue = true;
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        if (!debug_qk_chunk_enabled<C>(g, chunk)) {
            continue;
        }
        fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, chunk);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (first_issue) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
            first_issue = false;
        } else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTTFull, typename KScTile, typename KScTT>
__device__ inline void issue_qk_chunked_qsc_tmem_debug_masked(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    QScTTFull &q_sc_tm_full,
    const KScTile &k_sc_smem,
    KScTT &k_sc_tm,
    semaphore &finished,
    const globals_quant_p_from_scores_debug<C> &g
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTTFull::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    bool first_issue = true;
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        if (!debug_qk_chunk_enabled<C>(g, chunk)) {
            continue;
        }
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (first_issue) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
            first_issue = false;
        } else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTile, typename KScTile, typename QScTT, typename KScTT>
__device__ inline void fp4pv_issue_qk_chunked_cluster(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    const QScTile &q_sc_smem,
    const KScTile &k_sc_smem,
    QScTT &q_sc_tm,
    KScTT &k_sc_tm,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N_PACKED = KTile::rows * C::CLUSTER_SIZE;
    constexpr int N = (fp4pv_online_qk_logical_n<C>::value && std::is_same_v<T_AB, fp4e2m1_2>)
        ? (N_PACKED * 2)
        : N_PACKED;
    static_assert(!fp4pv_online_qk_logical_n<C>::value ||
                  (std::is_same_v<T_AB, fp4e2m1_2> && N == ScoresTT::cols),
        "QK logical-N probe expects FP4-packed K rows and score columns to match the logical N");
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; chunk++) {
        fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, chunk);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (chunk == 0) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
        else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

// Diagnostic-only K-scale lifetime discriminator.  Q retains the compact
// production page, while each of the three K64 chunks is copied to and issued
// from a distinct immutable 16-column TMEM subtile before the single QK
// completion commit.  This changes no arithmetic and introduces no extra
// completion boundary.
template <typename C, typename ScoresTT, typename QTile, typename KTile,
          typename QScTile, typename KScTile, typename QScTT,
          typename KScTTFull>
__device__ inline void fp4pv_issue_qk_chunked_cluster_k_scale_three_page(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    const QScTile &q_sc_smem,
    const KScTile &k_sc_smem,
    QScTT &q_sc_tm,
    KScTTFull &k_sc_tm_full,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N_PACKED = KTile::rows * C::CLUSTER_SIZE;
    constexpr int N =
        (fp4pv_online_qk_logical_n<C>::value &&
         std::is_same_v<T_AB, fp4e2m1_2>)
            ? (N_PACKED * 2)
            : N_PACKED;
    static_assert(C::QK_SCALE_CHUNKS == 3 && QScTT::cols == 16 &&
                      KScTTFull::cols == 48,
                  "K-scale three-page QK requires Q16 plus three K16 pages");
    static_assert(C::CLUSTER_SIZE == 2,
                  "K-scale three-page discriminator is native group2 only");
    static_assert(!fp4pv_online_qk_logical_n<C>::value ||
                      (std::is_same_v<T_AB, fp4e2m1_2> &&
                       N == ScoresTT::cols),
                  "QK logical-N route requires packed K to match score N");
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    // Keep the three issue operands compile-time-distinct.  A loop/subtile
    // spelling generated the three correct TCGEN copies but NVCC folded the
    // MMA K-scale operands back to page zero.  The explicit physical TMEM
    // addresses below are part of this diagnostic route's static contract.
    full_tt_fp8e4m3<16> k_sc_tm0(k_sc_tm_full.addr);
    full_tt_fp8e4m3<16> k_sc_tm1(k_sc_tm_full.addr + 4);
    full_tt_fp8e4m3<16> k_sc_tm2(k_sc_tm_full.addr + 8);

    fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, 0);
    fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm0, k_sc_smem, 0);
    detail::tcgen05::template st_st<
        T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
        scores.addr, q_desc.chunk_descriptor(0),
        k_desc.chunk_descriptor(0), q_sc_tm.addr,
        k_sc_tm0.addr, IDESC);

    fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, 1);
    fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm1, k_sc_smem, 1);
    detail::tcgen05::template st_st<
        T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
        scores.addr, q_desc.chunk_descriptor(1),
        k_desc.chunk_descriptor(1), q_sc_tm.addr,
        k_sc_tm1.addr, IDESC);

    fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, 2);
    fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm2, k_sc_smem, 2);
    detail::tcgen05::template st_st<
        T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
        scores.addr, q_desc.chunk_descriptor(2),
        k_desc.chunk_descriptor(2), q_sc_tm.addr,
        k_sc_tm2.addr, IDESC);
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

// Issues one independently addressed score fragment but deliberately leaves
// source completion and score publication to the caller. Progressive score
// reuse uses two N64 fragments from one K epoch, then commits the shared K
// source and publishes the combined N128 score exactly once.
template <typename C, typename ScoresTT, typename QTile, typename KTile,
          typename QScTile, typename KScTile, typename QScTT, typename KScTT>
__device__ inline void fp4pv_issue_qk_chunked_cluster_no_finish(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    const QScTile &q_sc_smem,
    const KScTile &k_sc_smem,
    QScTT &q_sc_tm,
    KScTT &k_sc_tm,
    int k_scale_row_offset
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    static_assert(C::CLUSTER_SIZE == 1,
        "progressive N64 QK currently owns one CTA");
    static_assert(N == ScoresTT::cols,
        "progressive QK destination must match its independently staged K half");
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<
            T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    // st_descriptor's generic subtile constructor retains the parent data
    // pointer and does not apply st_subtile::row_offset. Re-root the K-major
    // descriptor explicitly so the N64 high-half view starts at row 64.
    auto *k_root = k.idx(
        const_cast<typename KTile::dtype *>(&k.data[0]), int2{0, 0});
    if constexpr (KTile::swizzle_bytes == 128) {
        k_desc.base_desc = detail::matrix_descriptor_raw(k_root, 16, 1024, 1);
    } else if constexpr (KTile::swizzle_bytes == 64) {
        k_desc.base_desc = detail::matrix_descriptor_raw(k_root, 16, 512, 2);
    } else {
        k_desc.base_desc = detail::matrix_descriptor_raw(k_root, 16, 256, 3);
    }
    const uint32_t k_sc_addr =
        k_sc_tm.addr + (static_cast<uint32_t>(k_scale_row_offset) << 16);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, chunk);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (chunk == 0) {
            detail::tcgen05::template st_st<
                T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk),
                k_desc.chunk_descriptor(chunk), q_sc_tm.addr, k_sc_addr,
                IDESC);
        } else {
            detail::tcgen05::template st_st<
                T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk),
                k_desc.chunk_descriptor(chunk), q_sc_tm.addr, k_sc_addr,
                IDESC);
        }
    }
}

template <typename KScTile>
__device__ inline void fp4pv_repack_k_scale_n64_high(
    KScTile &dst,
    const KScTile &src
) {
    static_assert(sizeof(KScTile) % (32 * 16) == 0,
        "K-scale page must contain 32x16-byte QK chunks");
    auto *dst_bytes = reinterpret_cast<uint8_t *>(&dst.data[0]);
    const auto *src_bytes =
        reinterpret_cast<const uint8_t *>(&src.data[0]);
    constexpr int CHUNKS = sizeof(KScTile) / (32 * 16);
    #pragma unroll
    for (int chunk = 0; chunk < CHUNKS; ++chunk) {
        #pragma unroll
        for (int row = 0; row < 32; ++row) {
            #pragma unroll
            for (int col = 0; col < 16; ++col) {
                const int dst_idx = chunk * 32 * 16 + row * 16 + col;
                const int src_idx =
                    chunk * 32 * 16 + row * 16 + 8 + (col & 7);
                dst_bytes[dst_idx] = src_bytes[src_idx];
            }
        }
    }
    fp4pv_publish_cta_shared_only();
    fp4pv_publish_shared_backing_proxy_only();
}

template <int QUARTER, typename KScTile>
__device__ inline void fp4pv_repack_k_scale_n32_quarter(
    KScTile &dst,
    const KScTile &src
) {
    static_assert(QUARTER >= 1 && QUARTER < 4,
        "quarter zero uses the production K-scale page directly");
    static_assert(sizeof(KScTile) % (32 * 16) == 0,
        "K-scale page must contain 32x16-byte QK chunks");
    auto *dst_bytes = reinterpret_cast<uint8_t *>(&dst.data[0]);
    const auto *src_bytes =
        reinterpret_cast<const uint8_t *>(&src.data[0]);
    constexpr int CHUNKS = sizeof(KScTile) / (32 * 16);
    #pragma unroll
    for (int chunk = 0; chunk < CHUNKS; ++chunk) {
        #pragma unroll
        for (int row = 0; row < 32; ++row) {
            #pragma unroll
            for (int col = 0; col < 16; ++col) {
                const int dst_idx = chunk * 32 * 16 + row * 16 + col;
                const int src_idx =
                    chunk * 32 * 16 + row * 16 + QUARTER * 4 +
                    (col & 3);
                dst_bytes[dst_idx] = src_bytes[src_idx];
            }
        }
    }
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTile, typename KScTile, typename QScTT, typename KScTT>
__device__ inline void fp4pv_issue_qk_chunked_ranklocal(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    const QScTile &q_sc_smem,
    const KScTile &k_sc_smem,
    QScTT &q_sc_tm,
    KScTT &k_sc_tm,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows;
    constexpr int N_PACKED = KTile::rows;
    constexpr int N = (fp4pv_online_qk_logical_n<C>::value && std::is_same_v<T_AB, fp4e2m1_2>)
        ? (N_PACKED * 2)
        : N_PACKED;
    static_assert(!fp4pv_online_qk_logical_n<C>::value ||
                  (std::is_same_v<T_AB, fp4e2m1_2> && N == ScoresTT::cols),
        "QK logical-N probe expects FP4-packed K rows and score columns to match the logical N");
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; chunk++) {
        fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_smem, chunk);
        fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm, k_sc_smem, chunk);
        if (chunk == 0) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, 1, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
        else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, 1, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<1>(finished);
}

template <typename C, typename ScoresTT, typename QTile, typename KPairTile,
          typename QScTile, typename KScTile, typename QScTT, typename KPairScTT>
__device__ inline void fp4pv_issue_qk_true_half_n256_ranklocal(
    ScoresTT &scores,
    const QTile &q,
    const KPairTile &k_pair,
    const QScTile &q_sc_smem,
    const KScTile &k_sc_smem0,
    const KScTile &k_sc_smem1,
    QScTT &q_sc_tm,
    KPairScTT &k_sc_pair_tm,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows;
    constexpr int N = KPairTile::rows;
    static_assert(C::CLUSTER_SIZE == 1, "true-half QK currently owns one CTA");
    static_assert(std::is_same_v<T_AB, fp4e2m1_2> && N == ScoresTT::cols,
                  "true-half K operand must cover both logical score slots");
    static_assert(KPairScTT::cols == 32, "true-half QK needs two 16-column K-scale groups");
    constexpr uint32_t IDESC =
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KPairTile, 0> k_desc(k_pair);
    auto k_sc_tm0 = k_sc_pair_tm.template subtile<full_tt_fp8e4m3<16>>(0);
    auto k_sc_tm1 = k_sc_pair_tm.template subtile<full_tt_fp8e4m3<16>>(16);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_smem, chunk);
        fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm0, k_sc_smem0, chunk);
        fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm1, k_sc_smem1, chunk);
        if (chunk == 0) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, 1, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_pair_tm.addr, IDESC);
        } else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, 1, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_pair_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<1>(finished);
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTTFull, typename KScTile, typename KScTT>
__device__ inline void fp4pv_issue_qk_chunked_qsc_tmem_cluster(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    QScTTFull &q_sc_tm_full,
    const KScTile &k_sc_smem,
    KScTT &k_sc_tm,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTTFull::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N_PACKED = KTile::rows * C::CLUSTER_SIZE;
    constexpr int N = (fp4pv_online_qk_logical_n<C>::value && std::is_same_v<T_AB, fp4e2m1_2>)
        ? (N_PACKED * 2)
        : N_PACKED;
    static_assert(!fp4pv_online_native_group2_q_scale_three_page<C>::value ||
                  (C::QK_SCALE_CHUNKS == 3 && QScTTFull::cols == 48),
        "native group2 three-page Q scales require three immutable 16-column subtiles");
    static_assert(!fp4pv_online_qk_logical_n<C>::value ||
                  (std::is_same_v<T_AB, fp4e2m1_2> && N == ScoresTT::cols),
        "QK logical-N probe expects FP4-packed K rows and score columns to match the logical N");
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; chunk++) {
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (chunk == 0) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
        else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C, typename KScTTFull, typename KScTile>
__device__ inline void stage_k_scale_tmem_full(KScTTFull &k_sc_tm_full, const KScTile &k_sc_smem) {
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        auto k_sc_tm = k_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
    }
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTTFull, typename KScTTFull>
__device__ inline void issue_qk_chunked_qsc_tmem_preloaded_ksc(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    QScTTFull &q_sc_tm_full,
    KScTTFull &k_sc_tm_full,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTTFull::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        auto k_sc_tm = k_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        if (chunk == 0) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        } else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C>
__device__ inline int debug_physical_k_iter(
    const globals_quant_p_from_scores_debug<C> &g,
    int local_k_iter
) {
    return g.compute_k_iter_base + local_k_iter;
}

template <typename C>
__device__ inline int debug_k_sc_depth_index(
    const globals_quant_p_from_scores_debug<C> &g,
    int idx,
    int cta_rank,
    int tiles_m
) {
    const int physical_idx = debug_physical_k_iter<C>(g, idx);
    if (g.debug_k_scale_depth_mode == DEBUG_K_SCALE_DEPTH_MODE_SLAB64) {
        const int k_sc_depth = g.k_sc.depth();
        return min(physical_idx * C::CLUSTER_SIZE + cta_rank, k_sc_depth - 1);
    }
    return min(physical_idx * C::CLUSTER_SIZE + cta_rank, tiles_m - 1);
}

template <typename C>
__device__ inline int debug_output_m_tiles(const globals_quant_p_from_scores_debug<C> &g) {
    if (g.output_full_layout) {
        return g.q.rows() / C::Mb;
    }
    return g.selected_row_tile < 0 ? (g.q.rows() / C::Mb) : debug_selected_row_tile_span<C>(g);
}

template <typename C>
__device__ inline int debug_output_m_tile_index(const globals_quant_p_from_scores_debug<C> &g, int m_tile) {
    if (g.output_full_layout) {
        return m_tile;
    }
    return g.selected_row_tile < 0 ? m_tile : (m_tile - g.selected_row_tile);
}

template <typename C>
__device__ inline int debug_executed_k_iters(
    const globals_quant_p_from_scores_debug<C> &g,
    int full_iters_per_task
) {
    int executed;
    if (g.max_k_iters < 0) {
        executed = full_iters_per_task;
    } else {
        const int remaining_iters = full_iters_per_task - g.compute_k_iter_base;
        if (remaining_iters <= 0) {
            return 0;
        }
        executed = min(remaining_iters, g.max_k_iters);
    }
    if (g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM) {
        executed = min(executed, 2);
    }
    return executed;
}

template <typename C>
__device__ inline int debug_valid_causal_k_iters_for_row(
    const globals_quant_p_from_scores_debug<C> &g,
    int executed_k_iters,
    int m_tile
) {
    const int remaining_valid = m_tile + 1 - g.compute_k_iter_base;
    if (remaining_valid <= 0) {
        return 0;
    }
    return min(executed_k_iters, remaining_valid);
}

template <typename ScaleTile>
__device__ inline void copy_scale_tile_bytes_to_shared(ScaleTile &dst, const uint8_t *src) {
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int TILE_BYTES = sizeof(ScaleTile);
    static_assert(TILE_BYTES % WORD_BYTES == 0, "scale tile byte size must be word aligned");
    uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(&dst.data[0]);
    for (int linear = threadIdx.x * WORD_BYTES; linear < TILE_BYTES; linear += blockDim.x * WORD_BYTES) {
        uint32_t value = *reinterpret_cast<const uint32_t *>(src + linear);
        *reinterpret_cast<uint32_t *>(dst_bytes + linear) = value;
    }
}

template <typename ScaleTile>
__device__ inline void store_scale_chunk_32x16_to_global(const ScaleTile &tile, uint8_t *dst, int chunk_idx) {
    if (dst == nullptr) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int CHUNK_BYTES = 32 * 16;
    constexpr int WORD_COUNT = CHUNK_BYTES / WORD_BYTES;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]) + static_cast<uint32_t>(chunk_idx * CHUNK_BYTES);
    for (int word = warp::laneid(); word < WORD_COUNT; word += WARP_THREADS) {
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(word * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(dst + word * WORD_BYTES) = value;
    }
}

template <typename ScaleTile>
__device__ inline void store_scale_tmem_chunk_128x16_to_global(const ScaleTile &tile, uint8_t *dst) {
    if (dst == nullptr) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_ROW = 16 / WORD_BYTES;
    constexpr int TOTAL_WORDS = 128 * WORDS_PER_ROW;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = warp::laneid(); linear < TOTAL_WORDS; linear += WARP_THREADS) {
        const int row = linear / WORDS_PER_ROW;
        const int word = linear - row * WORDS_PER_ROW;
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(row * ScaleTile::cols + word * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(dst + linear * WORD_BYTES) = value;
    }
}

template <typename ScaleTile>
__device__ inline void store_scale_bf16_chunk_32x16_to_global(const ScaleTile &tile, bf16 *dst) {
    if (dst == nullptr) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int BF16_BYTES_PER_ROW = 16 * (int)sizeof(bf16);
    constexpr int WORDS_PER_ROW = BF16_BYTES_PER_ROW / WORD_BYTES;
    constexpr int TOTAL_WORDS = 32 * WORDS_PER_ROW;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(dst);
    for (int linear = warp::laneid(); linear < TOTAL_WORDS; linear += WARP_THREADS) {
        const int row = linear / WORDS_PER_ROW;
        const int word = linear - row * WORDS_PER_ROW;
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(row * tile.cols * sizeof(bf16) + word * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(dst_bytes + row * BF16_BYTES_PER_ROW + word * WORD_BYTES) = value;
    }
}

template <typename ScaleTile>
__device__ inline void store_scale_fp8_chunk_32x16_to_global_bf16(const ScaleTile &tile, bf16 *dst) {
    if (dst == nullptr) {
        return;
    }
    static_assert(std::is_same_v<typename ScaleTile::dtype, fp8e4m3>, "shared tile must use fp8e4m3 elements");
    constexpr int HALF_COLS = 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = warp::laneid(); linear < 32 * (HALF_COLS / 4); linear += WARP_THREADS) {
        const int row = linear / (HALF_COLS / 4);
        const int pack_col = linear - row * (HALF_COLS / 4);
        uint32_t packed_u32;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(packed_u32)
                     : "r"(smem_base + static_cast<uint32_t>(row * tile.cols + pack_col * 4)));
        const fp8e4m3_4 packed = *reinterpret_cast<const fp8e4m3_4 *>(&packed_u32);
        const float4 vals = kittens::base_types::convertor<float4, fp8e4m3_4>::convert(packed);
        dst[row * HALF_COLS + pack_col * 4 + 0] = __float2bfloat16(vals.x);
        dst[row * HALF_COLS + pack_col * 4 + 1] = __float2bfloat16(vals.y);
        dst[row * HALF_COLS + pack_col * 4 + 2] = __float2bfloat16(vals.z);
        dst[row * HALF_COLS + pack_col * 4 + 3] = __float2bfloat16(vals.w);
    }
}

template <typename RT>
__device__ inline void store_fp8_rt_left_half_32x16_to_global_bf16(const RT &src, bf16 *dst) {
    static_assert(std::is_same_v<typename RT::dtype, fp8e4m3_4>, "register tile must be fp8-packed");
    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "register tile must be row-major");
    static_assert(RT::rows == 32 && RT::cols == 32, "register tile must be 32x32");
    constexpr int HALF_COLS = 16;
    const int lane = warp::laneid();
    if (lane >= 16) {
        return;
    }
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int k = 0; k < RT::packed_per_tile; ++k) {
            const int row = i * TILE_ROW_DIM<typename RT::T> + lane;
            const int col = k * 4;
            const float4 vals = kittens::base_types::convertor<float4, fp8e4m3_4>::convert(src.tiles[i][0].data[k]);
            dst[row * HALF_COLS + col + 0] = __float2bfloat16(vals.x);
            dst[row * HALF_COLS + col + 1] = __float2bfloat16(vals.y);
            dst[row * HALF_COLS + col + 2] = __float2bfloat16(vals.z);
            dst[row * HALF_COLS + col + 3] = __float2bfloat16(vals.w);
        }
    }
}

template <typename RT>
__device__ inline void store_fp8_rt_raw_32x32_to_global_u32(const RT &src, uint32_t *dst) {
    if (dst == nullptr) {
        return;
    }
    static_assert(std::is_same_v<typename RT::dtype, fp8e4m3_4>, "register tile must be fp8-packed");
    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "register tile must be row-major");
    static_assert(RT::rows == 32 && RT::cols == 32, "register tile must be 32x32");
    const int lane = warp::laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int k = 0; k < RT::packed_per_tile; ++k) {
            dst[((lane * RT::height) + i) * RT::packed_per_tile + k] =
                *reinterpret_cast<const uint32_t *>(&src.tiles[i][0].data[k]);
        }
    }
}

template <typename C>
__device__ inline int debug_consumer_k_iters(
    const globals_quant_p_from_scores_debug<C> &g,
    int producer_iters_per_task
) {
    const bool producer_only_cut =
        g.debug_stage_cut == DEBUG_STAGE_CUT_ISSUE3_ONLY ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_WAIT ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_ISSUE ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_COMMIT ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_KSC_FINISH ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_SKIP_COMMIT_TAIL ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_FRESH_COMMIT_TAIL;
    if (producer_only_cut && producer_iters_per_task == 3) {
        return 2;
    }
    return producer_iters_per_task;
}

template <typename C>
__device__ inline bool debug_dual_scores_slot_mode(
    const globals_quant_p_from_scores_debug<C> &g
) {
    return g.debug_stage_cut == DEBUG_STAGE_CUT_FULL ||
           g.debug_stage_cut == DEBUG_STAGE_CUT_DUAL_SLOT_FULL;
}

template <typename C>
__device__ inline bool debug_full_like_mode(
    const globals_quant_p_from_scores_debug<C> &g
) {
    return g.debug_stage_cut == DEBUG_STAGE_CUT_FULL ||
           g.debug_stage_cut == DEBUG_STAGE_CUT_DUAL_SLOT_FULL ||
           g.debug_stage_cut == DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL;
}

template <typename C>
__device__ inline semaphore &debug_scores_arrived_sem_for_iter(
    semaphore (&scores_arrived)[1],
    semaphore (&scores_arrived_fresh)[1],
    const globals_quant_p_from_scores_debug<C> &g,
    int k_iter
) {
    if (debug_dual_scores_slot_mode(g) && (k_iter & 1)) {
        return scores_arrived_fresh[0];
    }
    return scores_arrived[0];
}

template <typename C>
__device__ inline int debug_scores_arrived_phase_for_iter(
    const globals_quant_p_from_scores_debug<C> &g,
    int k_iter,
    int legacy_phase
) {
    if (debug_dual_scores_slot_mode(g)) {
        return (k_iter >> 1) & 1;
    }
    return legacy_phase;
}

template <typename C>
__device__ inline void store_lse_row_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    float lse_val
) {
    if (g.disable_lse_store) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    g.lse[{b, h, 0, debug_output_row_index(g, global_row)}] = lse_val;
}

template <typename C>
__device__ inline float qk_scale_log2(const globals_quant_p_from_scores_debug<C> &g, int b, int h) {
    constexpr float LOG2E_OVER_SQRT_D = 0.10408340855860843f;  // log2(e) / sqrt(192)
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

template <typename C, typename PTile>
__device__ inline void store_p_fp4_tile_to_global_warpgroup(
    const PTile &p_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int packed_seqlen = g.tiles_to_run * (C::Nb / 2);
    const int row_base = m_tile * C::Mb;
    const int col_base = debug_window_k_iter(g, k_iter) * (C::Nb / 2);
    const uint32_t smem_base = __cvta_generic_to_shared(&p_smem.data[0]);
    constexpr int BYTES_PER_STORE = 4;
    static_assert(PTile::cols % BYTES_PER_STORE == 0, "Expected fp4 P tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = PTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = PTile::rows * STORES_PER_ROW;
    for (int linear = lane; linear < TOTAL_STORES; linear += 128) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(PTile::idx(smem_base, {row, col})));
        *reinterpret_cast<uint32_t *>(
            &g.p_fp4[((((b * heads + h) * seqlen) + row_base + row) * packed_seqlen) + col_base + col]
        ) = value;
    }
}

template <typename C, typename PScTile>
__device__ inline void store_p_sc_prepared_to_global_warpgroup(
    const PScTile &p_sc_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const uint32_t smem_base = __cvta_generic_to_shared(&p_sc_smem.data[0]);
    const int scale_k_base = debug_window_k_iter(g, k_iter) * 2;
    const int prepared_cols = g.tiles_to_run * 2;
    constexpr int ROW_BYTES = 8;
    constexpr int ROWS_PER_SLAB = 512 / ROW_BYTES;
    constexpr int TOTAL_ROWS = sizeof(PScTile) / ROW_BYTES;
    static_assert(TOTAL_ROWS == 128, "Expected 128 raw scale rows per P scale tile");
    for (int row = lane; row < TOTAL_ROWS; row += 128) {
        const int slab = row / ROWS_PER_SLAB;
        const int row_in_slab = row % ROWS_PER_SLAB;
        uint64_t value;
        asm volatile("ld.shared.b64 %0, [%1];"
                     : "=l"(value)
                     : "r"(smem_base + static_cast<uint32_t>(row * ROW_BYTES)));
        *reinterpret_cast<uint64_t *>(
            &g.p_sc_prepared[((((b * heads + h) * (seqlen / 128) + m_tile) * prepared_cols) + scale_k_base + slab) * 512 + row_in_slab * ROW_BYTES]
        ) = value;
    }
}

template <typename C, typename PTile>
__device__ inline void store_backward_ds_fp4_row_tile_to_global_warpgroup(
    const PTile &p_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    if (g.backward_ds_fp4_row == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int packed_seqlen = g.output_full_layout ? (seqlen / 2) : (g.tiles_to_run * (C::Nb / 2));
    const int row_base = m_tile * C::Mb;
    const int col_base = (g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter)) * (C::Nb / 2);
    const uint32_t smem_base = __cvta_generic_to_shared(&p_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert(PTile::cols % BYTES_PER_STORE == 0, "Expected fp4 dS tile cols divisible by vector width");
    static_assert(PTile::swizzle_bytes >= BYTES_PER_STORE, "Expected fp4 dS tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = PTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = PTile::rows * STORES_PER_ROW;
    for (int linear = lane; linear < TOTAL_STORES; linear += 128) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        uint4 value;
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                     : "r"(PTile::idx(smem_base, {row, col})));
        *reinterpret_cast<uint4 *>(
            &g.backward_ds_fp4_row[((((b * heads + h) * seqlen) + row_base + row) * packed_seqlen) + col_base + col]
        ) = value;
    }
}

template <typename C, typename PScTile>
__device__ inline void store_backward_ds_sc_row_tile_to_global_warpgroup(
    const PScTile &p_sc_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    static_assert(C::Nb == 128, "Direct row-MXFP4 dS scale emission expects K128 score tiles");
    if (g.backward_ds_sc_row == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int row_tiles = seqlen / C::Mb;
    const int k_tiles = g.output_full_layout ? (seqlen / C::Nb) : g.tiles_to_run;
    const int scale_k = g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter);
    const uint32_t smem_base = __cvta_generic_to_shared(&p_sc_smem.data[0]);
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_MXFP4_SCALE_TILE = 512 / WORD_BYTES;
    for (int linear = lane; linear < WORDS_PER_MXFP4_SCALE_TILE; linear += 128) {
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(linear * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(
            &g.backward_ds_sc_row[((((b * heads + h) * row_tiles + m_tile) * k_tiles + scale_k) * 512) + linear * WORD_BYTES]
        ) = value;
    }
}

template <typename C, typename PScTile>
__device__ inline void store_backward_ds_sc_row_word_to_global_warpgroup(
    PScTile &p_sc_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int global_row
) {
    static_assert(C::Nb == 128, "Direct row-MXFP4 dS scale emission expects K128 score tiles");
    if (g.backward_ds_sc_row == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int row_tiles = seqlen / C::Mb;
    const int k_tiles = g.output_full_layout ? (seqlen / C::Nb) : g.tiles_to_run;
    const int scale_k = g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter);
    uint8_t *scale_tile = &g.backward_ds_sc_row[((((b * heads + h) * row_tiles + m_tile) * k_tiles + scale_k) * 512)];
    const uint32_t packed = fp4pv_load_mxfp4_scale_word_swizzled(p_sc_smem, global_row);
    *reinterpret_cast<uint32_t *>(&scale_tile[fp4pv_scale_swizzle_idx(global_row & 127, 0)]) = packed;
}

template <typename C>
__device__ inline void store_backward_ds_row_mxfp4_direct_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int k_iter,
    const float2 (&scores_reg)[64]
) {
    static_assert(C::Nb == 128, "Direct row-MXFP4 dS emission expects K128 score tiles");
    if (g.backward_ds_fp4_row == nullptr || g.backward_ds_sc_row == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int output_row = debug_output_row_index(g, global_row);
    const int packed_seqlen = g.output_full_layout ? (seqlen / 2) : (g.tiles_to_run * (C::Nb / 2));
    const int col_base = (g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter)) * (C::Nb / 2);
    const int row_tiles = seqlen / C::Mb;
    const int k_tiles = g.output_full_layout ? (seqlen / C::Nb) : g.tiles_to_run;
    const int m_tile = global_row / C::Mb;
    const int scale_k = g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter);
    uint8_t *payload_row = &g.backward_ds_fp4_row[((((b * heads + h) * debug_output_rows(g)) + output_row) * packed_seqlen) + col_base];
    uint8_t *scale_tile = &g.backward_ds_sc_row[((((b * heads + h) * row_tiles + m_tile) * k_tiles + scale_k) * 512)];
    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group_bf16_amax(
            &scores_reg[mx_block * 16],
            g.backward_ds_quant_mode,
            words,
            e8m0);
        uint8_t *payload_dst = payload_row + mx_block * 16;
        *reinterpret_cast<uint32_t *>(payload_dst + 0) = words[0];
        *reinterpret_cast<uint32_t *>(payload_dst + 4) = words[1];
        *reinterpret_cast<uint32_t *>(payload_dst + 8) = words[2];
        *reinterpret_cast<uint32_t *>(payload_dst + 12) = words[3];
        scale_tile[fp4pv_scale_swizzle_idx(global_row & 127, mx_block)] = e8m0;
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename PTile, typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_transposed_stage_mxfp4_marker_bf16_amax(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int m_tile,
    int k_iter,
    const float2 (&scores_reg)[64]
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Local transposed dS stage expects 128x128 score tiles");
    #pragma unroll
    for (int qid = 0; qid < 4; ++qid) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[qid * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, qid);
    }
    warpgroup::sync(warpgroup::groupid() + 1);

    const int local_col = warpgroup::laneid();
    const int src_tile_x = local_col >> 6;
    const int src_col_in_tile = local_col & 63;
    const bool diagonal_tile = (k_iter == m_tile);
    uint32_t packed_scales = 0u;

    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        if (diagonal_tile && (mx_block * 32 + 31) < local_col) {
            constexpr uint32_t ZERO_FP4_WORD = 0x77777777u;
            uint32_t zero_words[4] = {
                ZERO_FP4_WORD,
                ZERO_FP4_WORD,
                ZERO_FP4_WORD,
                ZERO_FP4_WORD,
            };
            fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, zero_words, local_col, mx_block);
            fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, local_col, mx_block, 0u);
            continue;
        }

        __align__(8) bf16_2 values_bf[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int src_row0 = mx_block * 32 + jj * 2 + 0;
            const int src_row1 = mx_block * 32 + jj * 2 + 1;
            const int tile_idx0 = ((src_row0 >> 6) * 2) + src_tile_x;
            const int tile_idx1 = ((src_row1 >> 6) * 2) + src_tile_x;
            values_bf[jj].x = scan[tile_idx0][src_row0 & 63][src_col_in_tile];
            values_bf[jj].y = scan[tile_idx1][src_row1 & 63][src_col_in_tile];
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(values_bf, g.backward_ds_quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, local_col, mx_block);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, local_col, mx_block, e8m0);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    warpgroup::sync(warpgroup::groupid() + 1);
    return packed_scales;
}

template <typename C>
__device__ inline void store_backward_ds_col_mxfp4_from_populated_scan_full(
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Direct dS^T MXFP4 emission expects 128x128 score tiles");
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int tiles = seqlen / 128;
    const int local_col = warpgroup::laneid();
    const int out_row = k_iter * C::Nb + local_col;
    const int m_tile = global_row / C::Mb;
    const int packed_seqlen = seqlen / 2;
    uint8_t *payload_row =
        &g.backward_ds_fp4_col[((((b * heads + h) * seqlen) + out_row) * packed_seqlen) + m_tile * (C::Mb / 2)];
    uint8_t *scale_tile =
        &g.backward_ds_sc_col[((((b * heads + h) * tiles + k_iter) * tiles + m_tile) * 512)];
    const int src_tile_x = local_col >> 6;
    const int src_col_in_tile = local_col & 63;
    const bool diagonal_tile = (k_iter == m_tile);

    uint32_t packed_scales = 0u;
    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        uint8_t e8m0 = 0;
        if (diagonal_tile && (mx_block * 32 + 31) < local_col) {
            constexpr uint32_t ZERO_FP4_WORD = 0x77777777u;
            uint8_t *payload_dst = payload_row + mx_block * 16;
            fp4pv_store_global_u32x4_value(payload_dst, ZERO_FP4_WORD);
            packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
            continue;
        }
        __align__(8) bf16_2 values_bf[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int src_row0 = mx_block * 32 + jj * 2 + 0;
            const int src_row1 = mx_block * 32 + jj * 2 + 1;
            const int tile_idx0 = ((src_row0 >> 6) * 2) + src_tile_x;
            const int tile_idx1 = ((src_row1 >> 6) * 2) + src_tile_x;
            values_bf[jj].x = scan[tile_idx0][src_row0 & 63][src_col_in_tile];
            values_bf[jj].y = scan[tile_idx1][src_row1 & 63][src_col_in_tile];
        }
        uint32_t words[4];
        fp4pv_quantize_mxfp4_bf16_pairs_amax(values_bf, g.backward_ds_quant_mode, words, e8m0);
        uint8_t *payload_dst = payload_row + mx_block * 16;
        fp4pv_store_global_u32x4(payload_dst, words);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    fp4pv_store_global_scale_word_swizzled(scale_tile, local_col, packed_scales);
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C>
__device__ __noinline__ void zero_backward_ds_row_mxfp4_future_scale_tiles(
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Future row dS scale zeroing expects 128x128 tiles");
    if (g.backward_ds_sc_row == nullptr || !g.emit_backward_ds_row_mxfp4 || g.backward_ds_row_skip_future_zero) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int tiles = seqlen / 128;
    const int local_row = warpgroup::laneid();
    for (int k_iter = m_tile + 1; k_iter < tiles; ++k_iter) {
        uint8_t *scale_tile =
            &g.backward_ds_sc_row[((((b * heads + h) * tiles + m_tile) * tiles + k_iter) * 512)];
        fp4pv_store_global_scale_word_swizzled(scale_tile, local_row, 0u);
    }
}

template <typename C>
__device__ inline void zero_backward_ds_col_mxfp4_future_scale_tiles(
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Future dS^T scale zeroing expects 128x128 tiles");
    if (g.backward_ds_sc_col == nullptr || !g.emit_backward_ds_col_mxfp4 || g.backward_ds_col_skip_future_zero) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int tiles = seqlen / 128;
    const int local_col = warpgroup::laneid();
    for (int k_iter = m_tile + 1; k_iter < tiles; ++k_iter) {
        uint8_t *scale_tile =
            &g.backward_ds_sc_col[((((b * heads + h) * tiles + k_iter) * tiles + m_tile) * 512)];
        fp4pv_store_global_scale_word_swizzled(scale_tile, local_col, 0u);
    }
}

template <typename C>
__device__ inline void store_backward_ds_col_mxfp4_from_populated_scan_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Direct dS^T MXFP4 emission expects 128x128 score tiles");
    if (g.backward_ds_fp4_col == nullptr || g.backward_ds_sc_col == nullptr || !g.emit_backward_ds_col_mxfp4) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    store_backward_ds_col_mxfp4_from_populated_scan_full<C>(
        g, scan, b, h, global_row, k_iter);
}

template <typename C>
__device__ inline void store_backward_ds_col_mxfp4_direct_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter,
    const float2 (&scores_reg)[64]
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Direct dS^T MXFP4 emission expects 128x128 score tiles");
    if (g.backward_ds_fp4_col == nullptr || g.backward_ds_sc_col == nullptr || !g.emit_backward_ds_col_mxfp4) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }

    #pragma unroll
    for (int qid = 0; qid < 4; ++qid) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[qid * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, qid);
    }
    store_backward_ds_col_mxfp4_from_populated_scan_if_needed<C>(
        g, scan, b, h, global_row, k_iter);
}

template <typename C>
__device__ inline void store_backward_ds_row_col_mxfp4_direct_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter,
    const float2 (&scores_reg)[64]
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Direct dS row+col MXFP4 emission expects 128x128 score tiles");
    if (g.backward_ds_fp4_row == nullptr || g.backward_ds_sc_row == nullptr ||
        g.backward_ds_fp4_col == nullptr || g.backward_ds_sc_col == nullptr ||
        !g.emit_backward_ds_col_mxfp4) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }

    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int output_row = debug_output_row_index(g, global_row);
    const int packed_seqlen = g.output_full_layout ? (seqlen / 2) : (g.tiles_to_run * (C::Nb / 2));
    const int col_base = (g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter)) * (C::Nb / 2);
    const int row_tiles = seqlen / C::Mb;
    const int k_tiles = g.output_full_layout ? (seqlen / C::Nb) : g.tiles_to_run;
    const int m_tile = global_row / C::Mb;
    const int scale_k = g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter);
    uint8_t *payload_row = &g.backward_ds_fp4_row[((((b * heads + h) * debug_output_rows(g)) + output_row) * packed_seqlen) + col_base];
    uint8_t *scale_row_tile = &g.backward_ds_sc_row[((((b * heads + h) * row_tiles + m_tile) * k_tiles + scale_k) * 512)];

    uint32_t packed_row_scales = 0u;
    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, mx_block);

        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, g.backward_ds_quant_mode, words, e8m0);
        uint8_t *payload_dst = payload_row + mx_block * 16;
        fp4pv_store_global_u32x4(payload_dst, words);
        packed_row_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    fp4pv_store_global_scale_word_swizzled(scale_row_tile, global_row & 127, packed_row_scales);
    warpgroup::sync(warpgroup::groupid() + 1);

    const int tiles = seqlen / 128;
    const int local_col = warpgroup::laneid();
    const int out_row = k_iter * C::Nb + local_col;
    uint8_t *payload_col_row =
        &g.backward_ds_fp4_col[((((b * heads + h) * seqlen) + out_row) * (seqlen / 2)) + m_tile * (C::Mb / 2)];
    uint8_t *scale_col_tile =
        &g.backward_ds_sc_col[((((b * heads + h) * tiles + k_iter) * tiles + m_tile) * 512)];
    const int src_tile_x = local_col >> 6;
    const int src_col_in_tile = local_col & 63;

    uint32_t packed_col_scales = 0u;
    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        __align__(8) bf16_2 values_bf[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int src_row0 = mx_block * 32 + jj * 2 + 0;
            const int src_row1 = mx_block * 32 + jj * 2 + 1;
            const int tile_idx0 = ((src_row0 >> 6) * 2) + src_tile_x;
            const int tile_idx1 = ((src_row1 >> 6) * 2) + src_tile_x;
            values_bf[jj].x = scan[tile_idx0][src_row0 & 63][src_col_in_tile];
            values_bf[jj].y = scan[tile_idx1][src_row1 & 63][src_col_in_tile];
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(values_bf, g.backward_ds_quant_mode, words, e8m0);
        uint8_t *payload_dst = payload_col_row + mx_block * 16;
        fp4pv_store_global_u32x4(payload_dst, words);
        packed_col_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    fp4pv_store_global_scale_word_swizzled(scale_col_tile, local_col, packed_col_scales);
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C>
__device__ inline void store_p_bf16_live_tile_if_needed(
    bf16 *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int k_iter,
    const float2 *scores_reg
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_rows = debug_output_rows(g);
    const int output_row = debug_output_row_index(g, global_row);
    const int col_base = (g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter)) * C::Nb;
    const int window_cols = g.output_full_layout ? g.q.rows() : g.tiles_to_run * C::Nb;
    uint32_t *row_dst = reinterpret_cast<uint32_t *>(
        &dst[((((b * heads + h) * output_rows) + output_row) * window_cols) + col_base]
    );
    #pragma unroll
    for (int q = 0; q < C::Nb / 2; ++q) {
        bf16_2 packed = __float22bfloat162_rn(scores_reg[q]);
        row_dst[q] = *reinterpret_cast<uint32_t *>(&packed);
    }
}

__device__ __forceinline__ void fp4pv_load_score_quarter_tmem(
    float2 (&scores)[16],
    uint32_t score_tt_base,
    int qid
) {
    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
        : "=f"(scores[0].x), "=f"(scores[0].y), "=f"(scores[1].x), "=f"(scores[1].y),
          "=f"(scores[2].x), "=f"(scores[2].y), "=f"(scores[3].x), "=f"(scores[3].y),
          "=f"(scores[4].x), "=f"(scores[4].y), "=f"(scores[5].x), "=f"(scores[5].y),
          "=f"(scores[6].x), "=f"(scores[6].y), "=f"(scores[7].x), "=f"(scores[7].y),
          "=f"(scores[8].x), "=f"(scores[8].y), "=f"(scores[9].x), "=f"(scores[9].y),
          "=f"(scores[10].x), "=f"(scores[10].y), "=f"(scores[11].x), "=f"(scores[11].y),
          "=f"(scores[12].x), "=f"(scores[12].y), "=f"(scores[13].x), "=f"(scores[13].y),
          "=f"(scores[14].x), "=f"(scores[14].y), "=f"(scores[15].x), "=f"(scores[15].y)
        : "r"(score_tt_base + qid * 32));
}

__device__ __forceinline__ void fp4pv_load_score_half_tmem(
    float2 (&scores)[32],
    uint32_t score_tt_base,
    int half
) {
    const int qid0 = half * 2;
    const int qid1 = qid0 + 1;
    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
        : "=f"(scores[0].x), "=f"(scores[0].y), "=f"(scores[1].x), "=f"(scores[1].y),
          "=f"(scores[2].x), "=f"(scores[2].y), "=f"(scores[3].x), "=f"(scores[3].y),
          "=f"(scores[4].x), "=f"(scores[4].y), "=f"(scores[5].x), "=f"(scores[5].y),
          "=f"(scores[6].x), "=f"(scores[6].y), "=f"(scores[7].x), "=f"(scores[7].y),
          "=f"(scores[8].x), "=f"(scores[8].y), "=f"(scores[9].x), "=f"(scores[9].y),
          "=f"(scores[10].x), "=f"(scores[10].y), "=f"(scores[11].x), "=f"(scores[11].y),
          "=f"(scores[12].x), "=f"(scores[12].y), "=f"(scores[13].x), "=f"(scores[13].y),
          "=f"(scores[14].x), "=f"(scores[14].y), "=f"(scores[15].x), "=f"(scores[15].y)
        : "r"(score_tt_base + qid0 * 32));
    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
        : "=f"(scores[16].x), "=f"(scores[16].y), "=f"(scores[17].x), "=f"(scores[17].y),
          "=f"(scores[18].x), "=f"(scores[18].y), "=f"(scores[19].x), "=f"(scores[19].y),
          "=f"(scores[20].x), "=f"(scores[20].y), "=f"(scores[21].x), "=f"(scores[21].y),
          "=f"(scores[22].x), "=f"(scores[22].y), "=f"(scores[23].x), "=f"(scores[23].y),
          "=f"(scores[24].x), "=f"(scores[24].y), "=f"(scores[25].x), "=f"(scores[25].y),
          "=f"(scores[26].x), "=f"(scores[26].y), "=f"(scores[27].x), "=f"(scores[27].y),
          "=f"(scores[28].x), "=f"(scores[28].y), "=f"(scores[29].x), "=f"(scores[29].y),
          "=f"(scores[30].x), "=f"(scores[30].y), "=f"(scores[31].x), "=f"(scores[31].y)
        : "r"(score_tt_base + qid1 * 32));
}

__device__ __forceinline__ void fp4pv_apply_live_dp_quarter_tmem_x16(
    float2 *scores_reg,
    uint32_t score_tt_base,
    int qid,
    float2 live_dp_scale_2,
    float2 neg_delta_scaled_2
) {
    #pragma unroll
    for (int seg = 0; seg < 2; ++seg) {
        float dp_half[16];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];}"
            : "=f"(dp_half[0]), "=f"(dp_half[1]), "=f"(dp_half[2]), "=f"(dp_half[3]),
              "=f"(dp_half[4]), "=f"(dp_half[5]), "=f"(dp_half[6]), "=f"(dp_half[7]),
              "=f"(dp_half[8]), "=f"(dp_half[9]), "=f"(dp_half[10]), "=f"(dp_half[11]),
              "=f"(dp_half[12]), "=f"(dp_half[13]), "=f"(dp_half[14]), "=f"(dp_half[15])
            : "r"(score_tt_base + qid * 32 + seg * 16)
        );
        #pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const int q = qid * 16 + seg * 8 + jj;
            const float2 dp_pair = {dp_half[jj * 2], dp_half[jj * 2 + 1]};
            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                scores_reg[q], dp_pair, live_dp_scale_2, neg_delta_scaled_2);
        }
    }
}

__device__ __forceinline__ void fp4pv_apply_live_dp_quarter_tmem_x8(
    float2 *scores_reg,
    uint32_t score_tt_base,
    int qid,
    float2 live_dp_scale_2,
    float2 neg_delta_scaled_2
) {
    #pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
        float dp_word[8];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];}"
            : "=f"(dp_word[0]), "=f"(dp_word[1]), "=f"(dp_word[2]), "=f"(dp_word[3]),
              "=f"(dp_word[4]), "=f"(dp_word[5]), "=f"(dp_word[6]), "=f"(dp_word[7])
            : "r"(score_tt_base + qid * 32 + seg * 8)
        );
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int q = qid * 16 + seg * 4 + jj;
            const float2 dp_pair = {dp_word[jj * 2], dp_word[jj * 2 + 1]};
            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                scores_reg[q], dp_pair, live_dp_scale_2, neg_delta_scaled_2);
        }
    }
}

__device__ __forceinline__ void fp4pv_apply_live_dp_quarter_tmem_x8_aligned_probe(
    float2 *scores_reg,
    uint32_t score_tt_base,
    int qid,
    float2 live_dp_scale_2,
    float2 neg_delta_scaled_2
) {
    #pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
        float dp_word[8];
        const bool aligned_base = (seg == 0 || seg == 2);
        if (aligned_base) {
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];}"
                : "=f"(dp_word[0]), "=f"(dp_word[1]), "=f"(dp_word[2]), "=f"(dp_word[3]),
                  "=f"(dp_word[4]), "=f"(dp_word[5]), "=f"(dp_word[6]), "=f"(dp_word[7])
                : "r"(score_tt_base + qid * 32 + seg * 8)
            );
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dp_word[i] = 0.0f;
            }
        }
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int q = qid * 16 + seg * 4 + jj;
            const float2 dp_pair = {dp_word[jj * 2], dp_word[jj * 2 + 1]};
            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                scores_reg[q], dp_pair, live_dp_scale_2, neg_delta_scaled_2);
        }
    }
}

template <typename C>
__device__ __forceinline__ void fp4pv_apply_online_causal_mask_score_quarter(
    float2 (&scores)[16],
    int qid,
    int idx,
    int m_tile
) {
    const int query_col = m_tile * C::Mb + static_cast<int>(warpgroup::laneid());
    const int tile_col_base = idx * C::Nb + qid * 32;
    if (tile_col_base + 31 > query_col) {
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int col0 = tile_col_base + jj * 2;
            if (col0 > query_col) {
                scores[jj].x = base_types::constants<float>::neg_infty();
            }
            if (col0 + 1 > query_col) {
                scores[jj].y = base_types::constants<float>::neg_infty();
            }
        }
    }
}

template <typename C>
__device__ __forceinline__ void fp4pv_apply_online_causal_mask_score_half(
    float2 (&scores)[32],
    int half,
    int idx,
    int m_tile
) {
    const int query_col = m_tile * C::Mb + static_cast<int>(warpgroup::laneid());
    const int tile_col_base = idx * C::Nb + half * 64;
    if (tile_col_base + 63 > query_col) {
        #pragma unroll
        for (int jj = 0; jj < 32; ++jj) {
            const int col0 = tile_col_base + jj * 2;
            if (col0 > query_col) {
                scores[jj].x = base_types::constants<float>::neg_infty();
            }
            if (col0 + 1 > query_col) {
                scores[jj].y = base_types::constants<float>::neg_infty();
            }
        }
    }
}

template <typename C>
__device__ inline void store_scores_bf16_scan_repacked_tile_if_needed(
    bf16 *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    const fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_rows = debug_output_rows(g);
    const int output_row = debug_output_row_index(g, global_row);
    const int col_base = debug_window_k_iter(g, k_iter) * C::Nb;
    const int window_cols = g.tiles_to_run * C::Nb;
    uint64_t *row_dst = reinterpret_cast<uint64_t *>(
        &dst[((((b * heads + h) * output_rows) + output_row) * window_cols) + col_base]
    );
    const int top_bottom = (global_row >> 6) & 1;
    const int local_row = global_row & 63;
    for (int quarter = 0; quarter < 4; ++quarter) {
        const int tile_x = quarter >> 1;
        const int local_col_base = (quarter & 1) * 32;
        const int tile_idx = top_bottom * 2 + tile_x;
        const uint32_t smem_base = __cvta_generic_to_shared(&scan[tile_idx][local_row][local_col_base]);
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            uint64_t packed;
            asm volatile("ld.shared.b64 %0, [%1];"
                         : "=l"(packed)
                         : "r"(smem_base + static_cast<uint32_t>(i * sizeof(bf16_2))));
            row_dst[quarter * 8 + i / 2] = packed;
        }
    }
}

template <typename C>
__device__ inline void store_scores_bf16_tmem_x16_tile_if_needed(
    bf16 *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int m_tile,
    int k_iter,
    uint32_t score_tt_base
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_rows = debug_output_rows(g);
    const int output_row = debug_output_row_index(g, global_row);
    const int col_base = debug_window_k_iter(g, k_iter) * C::Nb;
    const int window_cols = g.tiles_to_run * C::Nb;
    uint32_t *row_dst = reinterpret_cast<uint32_t *>(
        &dst[((((b * heads + h) * output_rows) + output_row) * window_cols) + col_base]
    );
    const int causal_col = (k_iter > m_tile) ? -1 : (int)warpgroup::laneid();
    const bool apply_causal = k_iter >= m_tile;
    #pragma unroll
    for (int seg = 0; seg < C::Nb / 16; ++seg) {
        float reg[16];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];}"
            : "=f"(reg[0]), "=f"(reg[1]), "=f"(reg[2]), "=f"(reg[3]),
              "=f"(reg[4]), "=f"(reg[5]), "=f"(reg[6]), "=f"(reg[7]),
              "=f"(reg[8]), "=f"(reg[9]), "=f"(reg[10]), "=f"(reg[11]),
              "=f"(reg[12]), "=f"(reg[13]), "=f"(reg[14]), "=f"(reg[15])
            : "r"(score_tt_base + seg * 16)
        );
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 pair = {reg[i * 2], reg[i * 2 + 1]};
            if (apply_causal) {
                const int col0 = seg * 16 + i * 2;
                const int col1 = col0 + 1;
                if (col0 > causal_col) {
                    pair.x = base_types::constants<float>::neg_infty();
                }
                if (col1 > causal_col) {
                    pair.y = base_types::constants<float>::neg_infty();
                }
            }
            const bf16_2 packed = __float22bfloat162_rn(pair);
            row_dst[seg * 8 + i] = *reinterpret_cast<const uint32_t *>(&packed);
        }
    }
}

template <typename C, typename PScTile>
__device__ inline void store_p_sc_rowmajor_live_tile_if_needed(
    const PScTile &p_sc_smem,
    uint8_t *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int k_iter
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    constexpr int GROUPS_PER_TILE = tk_localcta::SCALES_PER_CHUNK_X;
    static_assert(GROUPS_PER_TILE == 8, "Expected 8 rowwise scale groups per 128-column tile");
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int rowmajor_cols = g.tiles_to_run * GROUPS_PER_TILE;
    const int col_base = debug_window_k_iter(g, k_iter) * GROUPS_PER_TILE;
    auto &raw_scales = *reinterpret_cast<const fp4pv_p_raw_scales *>(&p_sc_smem.data[0]);
    const int row = global_row & 127;
    *reinterpret_cast<uint64_t *>(
        &dst[((((b * heads + h) * seqlen) + global_row) * rowmajor_cols) + col_base]
    ) = *reinterpret_cast<const uint64_t *>(&raw_scales[row][0]);
}

template <typename C, typename PScTile>
__device__ inline void store_p_sc_rowmajor_scan_tile_if_needed(
    const PScTile &p_sc_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    if (g.p_sc_rowmajor_scan == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int rowmajor_cols = g.tiles_to_run * tk_localcta::SCALES_PER_CHUNK_X;
    const int col_base = debug_window_k_iter(g, k_iter) * tk_localcta::SCALES_PER_CHUNK_X;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_sc_smem.data[0]);
    for (int row = lane; row < 128; row += 128) {
        uint64_t value;
        asm volatile("ld.shared.b64 %0, [%1];"
                     : "=l"(value)
                     : "r"(smem_base + static_cast<uint32_t>(row * sizeof(uint64_t))));
        *reinterpret_cast<uint64_t *>(
            &g.p_sc_rowmajor_scan[((((b * heads + h) * seqlen) + m_tile * C::Mb + row) * rowmajor_cols) + col_base]
        ) = value;
    }
}

template <typename C, typename PScTile>
__device__ inline void quantize_localcta_p_scales_from_scan_with_amax_and_store_raw(
    PScTile &p_sc_prepared_tile,
    fp4pv_p_scan3d &scan,
    float amax_val,
    int consumer_tid,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter
) {
    float s_enc = FP4PV_LOCALCTA_GLOBAL_SCALE_NUM / amax_val;
    s_enc = fminf(s_enc, transformer_engine::detail::TypeExtrema<float>::max);
    if (amax_val == 0.0f || s_enc == 0.0f) {
        s_enc = 1.0f;
    }

    #pragma unroll
    for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
        const int stage_y = t / tk_localcta::TILES_X;
        const int stage_x = t % tk_localcta::TILES_X;
        rowwise_scaling_scales_only_group_padded<128>(
            scan,
            reinterpret_cast<tk_localcta::nvfp4_scale_t *>(&p_sc_prepared_tile.data[0]),
            s_enc,
            stage_y,
            stage_x,
            t,
            consumer_tid);
    }
    tk_localcta::subgroup_barrier_sync<128>();
    store_p_sc_rowmajor_scan_tile_if_needed<C>(p_sc_prepared_tile, g, b, h, m_tile, k_iter, consumer_tid);
    tk_localcta::subgroup_barrier_sync<128>();
    finalize_localcta_p_scales_with_amax<C>(p_sc_prepared_tile, amax_val, consumer_tid);
}

template <typename C>
__device__ inline void store_debug_row_stat_if_needed(
    float *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int k_iter,
    float value
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_rows = debug_output_rows(g);
    dst[((((b * heads + h) * output_rows) + debug_output_row_index(g, global_row)) * g.tiles_to_run) + debug_window_k_iter(g, k_iter)] = value;
}

template <typename C>
__device__ inline void store_debug_tile_amax_if_needed(
    float *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    float value
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_m_tiles = debug_output_m_tiles(g);
    dst[((((b * heads + h) * output_m_tiles) + debug_output_m_tile_index(g, m_tile)) * g.tiles_to_run) + debug_window_k_iter(g, k_iter)] = value;
}

template <typename C, typename ScaleTile>
__device__ inline void store_debug_q_sc_tile_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    const ScaleTile &tile,
    int b,
    int h,
    int m_tile
) {
    if (g.debug_q_sc_loaded == nullptr) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_CHUNK = 512 / WORD_BYTES;
    constexpr int TOTAL_WORDS = C::QK_SCALE_CHUNKS * WORDS_PER_CHUNK;
    const int heads = g.q.depth();
    const int output_m_tile = debug_output_m_tile_index(g, m_tile);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = 0; linear < TOTAL_WORDS; ++linear) {
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(linear * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(
            &g.debug_q_sc_loaded[((((b * debug_output_m_tiles(g)) + output_m_tile) * (heads * C::QK_SCALE_CHUNKS) + h * C::QK_SCALE_CHUNKS) * 512) + linear * WORD_BYTES]
        ) = value;
    }
}

template <typename C, typename ScaleTile>
__device__ inline void store_debug_k_sc_tile_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    const ScaleTile &tile,
    int b,
    int h,
    int k_iter
) {
    if (g.debug_k_sc_loaded == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_CHUNK = 512 / WORD_BYTES;
    constexpr int TOTAL_WORDS = C::QK_SCALE_CHUNKS * WORDS_PER_CHUNK;
    const int heads = g.q.depth();
    const int out_k = debug_window_k_iter(g, k_iter);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = 0; linear < TOTAL_WORDS; ++linear) {
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(linear * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(
            &g.debug_k_sc_loaded[((((b * g.tiles_to_run) + out_k) * (heads * C::QK_SCALE_CHUNKS) + h * C::QK_SCALE_CHUNKS) * 512) + linear * WORD_BYTES]
        ) = value;
    }
}

template <typename C, typename ScaleTile>
__device__ inline void store_debug_q_sc_tmem_chunk_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    const ScaleTile &tile,
    int b,
    int h,
    int m_tile,
    int chunk
) {
    if (g.debug_q_sc_tmem_loaded == nullptr) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_ROW = 16 / WORD_BYTES;
    const int heads = g.q.depth();
    const int output_m_tile = debug_output_m_tile_index(g, m_tile);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int row = 0; row < 128; ++row) {
        for (int word = 0; word < WORDS_PER_ROW; ++word) {
            uint32_t value;
            asm volatile("ld.shared.b32 %0, [%1];"
                         : "=r"(value)
                         : "r"(smem_base + static_cast<uint32_t>(row * ScaleTile::cols + word * WORD_BYTES)));
            *reinterpret_cast<uint32_t *>(
                &g.debug_q_sc_tmem_loaded[((((b * debug_output_m_tiles(g)) + output_m_tile) * (heads * C::QK_SCALE_CHUNKS) + h * C::QK_SCALE_CHUNKS + chunk) * 512) + (row * 16 + word * WORD_BYTES)]
            ) = value;
        }
    }
}

template <typename C, typename ScaleTile>
__device__ inline void store_debug_k_sc_tmem_chunk_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    const ScaleTile &tile,
    int b,
    int h,
    int k_iter,
    int chunk
) {
    if (g.debug_k_sc_tmem_loaded == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_ROW = 16 / WORD_BYTES;
    const int heads = g.q.depth();
    const int out_k = debug_window_k_iter(g, k_iter);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int row = 0; row < 128; ++row) {
        for (int word = 0; word < WORDS_PER_ROW; ++word) {
            uint32_t value;
            asm volatile("ld.shared.b32 %0, [%1];"
                         : "=r"(value)
                         : "r"(smem_base + static_cast<uint32_t>(row * ScaleTile::cols + word * WORD_BYTES)));
            *reinterpret_cast<uint32_t *>(
                &g.debug_k_sc_tmem_loaded[((((b * g.tiles_to_run) + out_k) * (heads * C::QK_SCALE_CHUNKS) + h * C::QK_SCALE_CHUNKS + chunk) * 512) + (row * 16 + word * WORD_BYTES)]
            ) = value;
        }
    }
}

template <typename TT, typename ST>
__device__ inline void copy_scale_tt_to_shared(ST &dst, const TT &src) {
    rt_fp8e4m3<128, 32> scale_rt;
    kittens::group<1>::load_async(scale_rt, src);
    tensor_load_wait();
    tensor_before_thread_sync();
    kittens::group<1>::store(dst, scale_rt);
}

template <typename TT>
__device__ inline void load_scale_tt_subtile_to_rt(rt_fp8e4m3<32, 32> &dst, const TT &src, int row_offset) {
    auto src_subtile = src.template subtile<tt<typename TT::dtype, 32, 32>>(row_offset, 0);
    kittens::group<1>::load_async(dst, src_subtile);
    tensor_load_wait();
    tensor_before_thread_sync();
}

template <typename TT, typename ST>
__device__ inline void copy_scale_tt_subtile_to_shared(ST &dst, const TT &src, int row_offset) {
    rt_fp8e4m3<32, 32> scale_rt;
    load_scale_tt_subtile_to_rt(scale_rt, src, row_offset);
    kittens::group<1>::store(dst, scale_rt);
}

template <typename TT>
__device__ inline void dump_scale_tt_subtile_to_global_bf16(bf16 *dst, const TT &src, int row_offset) {
    rt_fp8e4m3<32, 32> scale_rt_fp8;
    load_scale_tt_subtile_to_rt(scale_rt_fp8, src, row_offset);
    store_fp8_rt_left_half_32x16_to_global_bf16(scale_rt_fp8, dst);
}

template <typename TT>
__device__ inline void dump_scale_tt_subtile_to_global_raw_u32(uint32_t *dst, const TT &src, int row_offset) {
    rt_fp8e4m3<32, 32> scale_rt_fp8;
    load_scale_tt_subtile_to_rt(scale_rt_fp8, src, row_offset);
    store_fp8_rt_raw_32x32_to_global_u32(scale_rt_fp8, dst);
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_probe_loaded_qk_scale_chunks_debug(const __grid_constant__ globals_loaded_qk_scale_chunk_debug<C> g) {
    const int cta_rank = cluster_ctarank();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    typename globals_loaded_qk_scale_chunk_debug<C>::q_sc_smem_storage &q_sc_smem =
        al.allocate<typename globals_loaded_qk_scale_chunk_debug<C>::q_sc_smem_storage>();
    typename globals_loaded_qk_scale_chunk_debug<C>::k_sc_smem_storage &k_sc_smem =
        al.allocate<typename globals_loaded_qk_scale_chunk_debug<C>::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] = al.allocate<typename globals_loaded_qk_scale_chunk_debug<C>::k_sc_guard_slot, 1>();
    (void)k_sc_guard;
    auto (&scale_dbg_smem)[4] = al.allocate<st_fp8e4m3<32, 32, false>, 4>();

    if (cta_rank == 0) {
        copy_scale_tile_bytes_to_shared(slot_tile_at(q_sc_smem, 0), g.q_sc_tile_bytes);
        copy_scale_tile_bytes_to_shared(slot_tile_at(k_sc_smem, 0), g.k_sc_tile_bytes);
    }
    __syncthreads();
    if (cta_rank == 0 && threadIdx.x == 0) {
        publish_shared_backing();
        publish_cluster_shared_backing_if_needed<C>();
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    if (warpgroup::groupid() != 3 || warpgroup::warpid() != 0 || cta_rank != 0) {
        return;
    }

    const bool issue_lane = warp::elect_leader();
    static_assert(64 <= MAX_TENSOR_COLS, "scale chunk probe exceeds TMEM budget");
    full_tt_fp8e4m3<32> q_sc_tm_dbg = tm_alloc.template allocate<full_tt_fp8e4m3<32>>(0);
    full_tt_fp8e4m3<32> k_sc_tm_dbg = tm_alloc.template allocate<full_tt_fp8e4m3<32>>(32);

    store_scale_chunk_32x16_to_global(slot_tile_at(q_sc_smem, 0), g.q_sc_source_chunk, g.chunk_idx);
    store_scale_chunk_32x16_to_global(slot_tile_at(k_sc_smem, 0), g.k_sc_source_chunk, g.chunk_idx);
    __syncwarp();
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_SOURCE_ONLY) {
        return;
    }

    auto q_sc_tm_dbg_lo = q_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(0);
    auto q_sc_tm_dbg_hi = q_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(16);
    if (issue_lane) {
        load_q_scale_chunk<C>(q_sc_tm_dbg_lo, slot_tile_at(q_sc_smem, 0), g.chunk_idx);
        load_q_scale_chunk<C>(q_sc_tm_dbg_hi, slot_tile_at(q_sc_smem, 0), g.chunk_idx);
    }
    __syncwarp();
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_LOAD_SYNC) {
        tensor_load_wait();
        tensor_before_thread_sync();
        return;
    }
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_RT_LOAD_ONLY) {
        #pragma unroll
        for (int row_block = 0; row_block < 4; ++row_block) {
            rt_fp8e4m3<32, 32> scale_rt;
            load_scale_tt_subtile_to_rt(scale_rt, q_sc_tm_dbg, row_block * 32);
        }
        return;
    }
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_DIRECT_DUMP || g.stage_mode == DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP) {
        #pragma unroll
        for (int row_block = 0; row_block < 4; ++row_block) {
            dump_scale_tt_subtile_to_global_bf16(g.q_sc_tmem_chunk + row_block * 32 * 16, q_sc_tm_dbg, row_block * 32);
            dump_scale_tt_subtile_to_global_raw_u32(g.q_sc_tmem_raw_words + row_block * 32 * 2 * 4, q_sc_tm_dbg, row_block * 32);
        }
        if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_DIRECT_DUMP) {
            return;
        }
    }
    if (g.stage_mode != DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP) {
        #pragma unroll
        for (int row_block = 0; row_block < 4; ++row_block) {
            copy_scale_tt_subtile_to_shared(scale_dbg_smem[row_block], q_sc_tm_dbg, row_block * 32);
        }
        __syncwarp();
        if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_COPY_SHARED_ONLY) {
            return;
        }
        for (int row_block = 0; row_block < 4; ++row_block) {
            store_scale_fp8_chunk_32x16_to_global_bf16(scale_dbg_smem[row_block], g.q_sc_tmem_chunk + row_block * 32 * 16);
        }
        __syncwarp();
        if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_LOAD_COPY) {
            return;
        }
    }

    auto k_sc_tm_dbg_lo = k_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(0);
    auto k_sc_tm_dbg_hi = k_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(16);
    if (issue_lane) {
        load_k_scale_chunk<C>(k_sc_tm_dbg_lo, slot_tile_at(k_sc_smem, 0), g.chunk_idx);
        load_k_scale_chunk<C>(k_sc_tm_dbg_hi, slot_tile_at(k_sc_smem, 0), g.chunk_idx);
    }
    __syncwarp();
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_K_LOAD_SYNC) {
        tensor_load_wait();
        tensor_before_thread_sync();
        return;
    }
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP) {
        #pragma unroll
        for (int row_block = 0; row_block < 4; ++row_block) {
            dump_scale_tt_subtile_to_global_bf16(g.k_sc_tmem_chunk + row_block * 32 * 16, k_sc_tm_dbg, row_block * 32);
            dump_scale_tt_subtile_to_global_raw_u32(g.k_sc_tmem_raw_words + row_block * 32 * 2 * 4, k_sc_tm_dbg, row_block * 32);
        }
        return;
    }
    #pragma unroll
    for (int row_block = 0; row_block < 4; ++row_block) {
        copy_scale_tt_subtile_to_shared(scale_dbg_smem[row_block], k_sc_tm_dbg, row_block * 32);
    }
    __syncwarp();
    for (int row_block = 0; row_block < 4; ++row_block) {
        store_scale_fp8_chunk_32x16_to_global_bf16(scale_dbg_smem[row_block], g.k_sc_tmem_chunk + row_block * 32 * 16);
    }
    __syncwarp();
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_K_LOAD_COPY) {
        return;
    }
}

template <
    typename C,
    bool NORMALIZE_TILES = true,
    bool QK_ONLY = false,
    bool SAVED_LSE_BWD_ONLY = false,
    bool SAVED_BWD_EMIT_ROW_MXFP4 = false,
    bool SAVED_BWD_EMIT_COL_MXFP4 = false,
    bool SAVED_BWD_STORE_BF16 = true,
    bool SAVED_BWD_LOCAL_STAGE_ONLY = false,
    bool SAVED_BWD_DQ_DUAL_HALVES = false,
    int FIXED_DEBUG_STAGE_CUT = -1,
    bool FIXED_FULL_PRODUCTION_PATH = false,
    bool LIVE_DP = false,
    bool LIVE_DP_MODE0_ONLY = false,
    bool LIVE_DP_MODE0_X8 = false>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_quantize_p_from_scores_debug(const __grid_constant__ globals_quant_p_from_scores_debug<C> g) {
    using G = globals_quant_p_from_scores_debug<C>;
    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_tail_outputs = tt<float, C::Mb, C::Dvo / 2>;
    using d_tt_q_sc = full_tt_fp8e4m3<C::Q_SC_TMEM_WIDTH>;
    using d_tt_sc16 = full_tt_fp8e4m3<16>;
    using d_tt_bwd_p_sc = full_tt_fp8e8m0<16>;
    using d_tt_bwd_k_sc = full_tt_fp8e8m0<16>;
    constexpr int COMPACT_Q_SC_BASE = C::Nb;
    constexpr int COMPACT_K_SC_BASE = COMPACT_Q_SC_BASE + C::Q_SC_TMEM_WIDTH;
    constexpr int QK_COMPACT_ACTIVE_TMEM_END = COMPACT_K_SC_BASE + 16;
    constexpr int BWD_DQ_P_SC_BASE = QK_COMPACT_ACTIVE_TMEM_END;
    constexpr int BWD_DQ_K_SC_BASE = BWD_DQ_P_SC_BASE + 16;
    constexpr int BWD_DQ_K1_SC_BASE = BWD_DQ_K_SC_BASE + 16;
    constexpr int BWD_DQ_OUTPUT_BASE = C::Nb + C::Dvo;
    constexpr int BWD_DQ_OUTPUT1_BASE = BWD_DQ_OUTPUT_BASE + C::Dvo;
    constexpr bool BWD_DQ_TAIL64_CLUSTER =
        FIXED_FULL_PRODUCTION_PATH && SAVED_BWD_DQ_DUAL_HALVES &&
        C::CLUSTER_SIZE == 2 && C::Dvo == 128 && C::Dqk == 192;
    constexpr int LIVE_DP_DOUT_SC_BASE =
        BWD_DQ_TAIL64_CLUSTER ? (BWD_DQ_OUTPUT1_BASE + C::Dvo / 2) : (BWD_DQ_K1_SC_BASE + 16);
    constexpr int LIVE_DP_V_SC_BASE =
        BWD_DQ_TAIL64_CLUSTER ? (LIVE_DP_DOUT_SC_BASE + 16) : COMPACT_K_SC_BASE;
    static_assert(BWD_DQ_K1_SC_BASE + 16 <= BWD_DQ_OUTPUT_BASE, "fused dQ scale TMEM layout overlaps dQ accumulators");
    static_assert(
        (!BWD_DQ_TAIL64_CLUSTER && LIVE_DP_V_SC_BASE + 16 <= BWD_DQ_OUTPUT_BASE) ||
            (BWD_DQ_TAIL64_CLUSTER &&
             BWD_DQ_OUTPUT1_BASE + C::Dvo / 2 <= LIVE_DP_DOUT_SC_BASE &&
             LIVE_DP_V_SC_BASE + 16 <= BWD_DQ_OUTPUT1_BASE + C::Dvo),
        "live dP scale TMEM layout overlaps active accumulators");
    static_assert(BWD_DQ_OUTPUT1_BASE + C::Dvo <= MAX_TENSOR_COLS, "dual-half fused dQ TMEM layout exceeds budget");
    static_assert(BWD_DQ_OUTPUT1_BASE + C::Dvo / 2 <= MAX_TENSOR_COLS, "tail64 fused dQ TMEM layout exceeds budget");

    const int cta_rank = cluster_ctarank();
    const int stage_cut = (FIXED_DEBUG_STAGE_CUT >= 0) ? FIXED_DEBUG_STAGE_CUT : g.debug_stage_cut;
    const int tiles_m = g.q.rows() / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int num_head = g.q.depth();
    const int num_batch = g.q.batch();
    const bool restrict_to_selected_head = g.live_dp_selected_head >= 0;
    const int head_work = restrict_to_selected_head ? 1 : num_head;
    const int size_one_head = g.k.rows() * (C::Dqk + C::Dvo) * 2;
    constexpr int size_l2 = 50 * 1024 * 1024;
    int swizzle = 1;
    if (size_l2 >= size_one_head) {
        swizzle = 1 << (31 - __clz(size_l2 / size_one_head));
    }
    const int num_hb = head_work * num_batch;
    const int num_hb_quotient = num_hb / swizzle;
    const int num_hb_remainder = num_hb - num_hb_quotient * swizzle;
    const int l2_major = swizzle * num_block;
    const bool restrict_to_selected_row_tile = g.selected_row_tile >= 0;
    const int total_bids = restrict_to_selected_row_tile
        ? num_batch * head_work * C::CLUSTER_SIZE
        : num_batch * head_work * num_block * C::CLUSTER_SIZE;

    auto get_tile_idx = [&](int block_idx) -> int3 {
        if (restrict_to_selected_row_tile) {
            const int bidhb_actual = block_idx / C::CLUSTER_SIZE;
            const int b = restrict_to_selected_head ? bidhb_actual : (bidhb_actual / num_head);
            const int h = restrict_to_selected_head ? g.live_dp_selected_head : (bidhb_actual - b * num_head);
            const int selected_row_cluster = g.selected_row_tile / C::CLUSTER_SIZE;
            const int m_tile_base_cluster = selected_row_cluster * C::CLUSTER_SIZE;
            return {b, m_tile_base_cluster, h};
        }
        int cluster_linear = block_idx / C::CLUSTER_SIZE;
        int bidhb = cluster_linear / l2_major;
        int l2_mod = cluster_linear - bidhb * l2_major;
        int m_cluster, bidhb_residual;
        if (bidhb < num_hb_quotient) {
            m_cluster = l2_mod / swizzle;
            bidhb_residual = l2_mod - m_cluster * swizzle;
        } else {
            int divisor = (num_hb_remainder > 0) ? num_hb_remainder : 1;
            m_cluster = l2_mod / divisor;
            bidhb_residual = l2_mod - m_cluster * divisor;
        }
        int bidhb_actual = bidhb * swizzle + bidhb_residual;
        int b = restrict_to_selected_head ? bidhb_actual : (bidhb_actual / num_head);
        int h = restrict_to_selected_head ? g.live_dp_selected_head : (bidhb_actual - b * num_head);
        m_cluster = num_block - 1 - m_cluster;
        int m_tile_base_cluster = m_cluster * (C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE);
        return {b, m_tile_base_cluster, h};
    };
    constexpr bool fixed_fresh3_full =
        FIXED_DEBUG_STAGE_CUT == DEBUG_STAGE_CUT_FRESH3_FULL;
    constexpr bool fixed_full_production_path = FIXED_FULL_PRODUCTION_PATH;
    static_assert(
        !fixed_full_production_path || fixed_fresh3_full,
        "fixed full production path must use DEBUG_STAGE_CUT_FRESH3_FULL");
    static_assert(
        !fixed_full_production_path ||
            (QK_ONLY &&
             SAVED_LSE_BWD_ONLY &&
             SAVED_BWD_LOCAL_STAGE_ONLY &&
             SAVED_BWD_DQ_DUAL_HALVES &&
	             ((SAVED_BWD_EMIT_COL_MXFP4 && !SAVED_BWD_EMIT_ROW_MXFP4) ||
	              (SAVED_BWD_EMIT_ROW_MXFP4 && !SAVED_BWD_EMIT_COL_MXFP4) ||
                  (!SAVED_BWD_EMIT_ROW_MXFP4 && !SAVED_BWD_EMIT_COL_MXFP4))),
        "fixed full production path is only valid for fused dQ + at most one dS MXFP4 layout");
    static_assert(
        !LIVE_DP ||
            (SAVED_BWD_LOCAL_STAGE_ONLY &&
             SAVED_BWD_DQ_DUAL_HALVES &&
             fixed_full_production_path &&
             C::Dvo == 128 &&
             C::Nb == 128 &&
             C::CLUSTER_SIZE == 2),
        "live dP is currently specialized for single-layout local-stage replay paths");
    static_assert(!LIVE_DP_MODE0_ONLY || LIVE_DP, "mode0-only live dP instantiation requires LIVE_DP");
    static_assert(!LIVE_DP_MODE0_X8 || LIVE_DP_MODE0_ONLY, "x8 live dP TMEM loads require mode0-only live dP");
    auto executed_k_iters = [&](int full_iters_per_task) -> int {
        if constexpr (fixed_fresh3_full) {
            return full_iters_per_task;
        } else {
            return debug_executed_k_iters<C>(g, full_iters_per_task);
        }
    };
    auto physical_k_iter_for = [&](int local_k_iter) -> int {
        if constexpr (fixed_fresh3_full) {
            return local_k_iter;
        } else {
            return debug_physical_k_iter<C>(g, local_k_iter);
        }
    };

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    auto allocate_tmem_tile = [&]<typename TT>(int col_offset) {
        if constexpr (TT::rows == 64) {
            return tm_alloc.template allocate<TT>(0, col_offset);
        } else {
            return tm_alloc.template allocate<TT>(col_offset);
        }
    };
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    typename G::q_tile (&q_smem)[C::NUM_SOFTMAXXERS] = al.allocate<typename G::q_tile, C::NUM_SOFTMAXXERS>();
    typename G::q_sc_smem_storage &q_sc_smem = al.allocate<typename G::q_sc_smem_storage>();
    typename G::k_tile (&k_smem)[C::LOAD_STAGES] = al.allocate<typename G::k_tile, C::LOAD_STAGES>();
    typename G::k_sc_smem_storage &k_sc_smem = al.allocate<typename G::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] = al.allocate<typename G::k_sc_guard_slot, 1>();
    (void)k_sc_guard;
    auto (&p_bf16_scan)[1] = al.allocate<fp4pv_p_scan3d, 1>();
    auto (&scale_dbg_smem)[1] = al.allocate<st_fp8e4m3<128, 32, false>, 1>();
    auto (&p_fp4_stage)[2] = al.allocate<typename G::p_fp4_tile, 2>();
    auto (&p_sc_stage)[2] = al.allocate<typename G::p_sc_tile, 2>();
    typename G::bwd_dq_k_fp4_tile (&bwd_dq_k_smem)[2] = al.allocate<typename G::bwd_dq_k_fp4_tile, 2>();
    typename G::bwd_dq_sc_tile (&bwd_dq_k_sc_smem)[2] = al.allocate<typename G::bwd_dq_sc_tile, 2>();
    typename G::bwd_dq_k_fp4_tile (&bwd_dq_k1_smem)[2] = al.allocate<typename G::bwd_dq_k_fp4_tile, 2>();
    typename G::bwd_dq_sc_tile (&bwd_dq_k1_sc_smem)[2] = al.allocate<typename G::bwd_dq_sc_tile, 2>();
    auto (&bwd_dq_p_dummy)[1] = al.allocate<typename G::p_fp4_tile, 1>();
    typename G::bwd_dq_k_fp4_tile (&bwd_dq_k_dummy)[1] = al.allocate<typename G::bwd_dq_k_fp4_tile, 1>();
    auto (&p_sc_guard)[1] = al.allocate<typename G::p_sc_guard_slot, 1>();
    (void)p_sc_guard;
    typename G::live_dp_dout_fp4_tile *live_dp_dout_smem = nullptr;
    typename G::live_dp_sc_tile *live_dp_dout_sc_smem = nullptr;
    typename G::live_dp_v_row_fp4_tile *live_dp_v_smem = nullptr;
    typename G::live_dp_sc_tile *live_dp_v_sc_smem = nullptr;
    if constexpr (LIVE_DP) {
        auto (&live_dp_dout_alloc)[1] = al.allocate<typename G::live_dp_dout_fp4_tile, 1>();
        auto (&live_dp_dout_sc_alloc)[1] = al.allocate<typename G::live_dp_sc_tile, 1>();
        auto (&live_dp_v_alloc)[2] = al.allocate<typename G::live_dp_v_row_fp4_tile, 2>();
        auto (&live_dp_v_sc_alloc)[2] = al.allocate<typename G::live_dp_sc_tile, 2>();
        live_dp_dout_smem = &live_dp_dout_alloc[0];
        live_dp_dout_sc_smem = &live_dp_dout_sc_alloc[0];
        live_dp_v_smem = &live_dp_v_alloc[0];
        live_dp_v_sc_smem = &live_dp_v_sc_alloc[0];
    }

    __shared__ semaphore q_arrived[1], q_sc_arrived[1], q_finished[1];
    __shared__ semaphore k_arrived[C::LOAD_STAGES], k_sc_arrived[C::K_SCALE_SMEM_SLOTS], k_finished[C::LOAD_STAGES], k_sc_finished[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore scores_arrived[1], scores_arrived_fresh[1], corr_arrived[1], rescale_finished[1], p_copy_done[2], p_pack_ready[2], p_store_done[2];
    __shared__ semaphore bwd_dq_stage_ready[2], bwd_dq_stage_remote_ready[2], bwd_dq_k_remote_ready[2], bwd_dq_issue_ready[2], bwd_dq_src_finished[2], bwd_dq_src1_finished[2], bwd_dq_tmem_ready[1], bwd_dq_accum_done[1];
    __shared__ semaphore live_dp_dout_ready[1], live_dp_dout_remote_ready[1], live_dp_dout_finished[1];
    __shared__ semaphore live_dp_v_ready[2], live_dp_v_remote_ready[2], live_dp_v_finished[2];
    __shared__ semaphore live_dp_arrived[1], live_dp_copy_done[1];
    __shared__ float p_softmax_warp_amax[2][4];
    __shared__ float p_cta_amax[2];

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<typename G::q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<typename G::k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<typename G::k_sc_tile, dim::ROW>();
        init_semaphore(q_arrived[0], 0, 1);
        init_semaphore(q_sc_arrived[0], 0, 1);
        init_semaphore(q_finished[0], 0, 1);
        init_semaphore(scores_arrived[0], 0, 1);
        init_semaphore(scores_arrived_fresh[0], 0, 1);
        init_semaphore(corr_arrived[0], 0, 4);
        init_semaphore(rescale_finished[0], 0, 1);
        init_semaphore(p_copy_done[0], 0, 1);
        init_semaphore(p_copy_done[1], 0, 1);
        init_semaphore(p_pack_ready[0], 0, 1);
        init_semaphore(p_pack_ready[1], 0, 1);
        init_semaphore(p_store_done[0], 0, 1);
        init_semaphore(p_store_done[1], 0, 1);
        init_semaphore(bwd_dq_stage_ready[0], 0, 1);
        init_semaphore(bwd_dq_stage_ready[1], 0, 1);
        init_semaphore(bwd_dq_stage_remote_ready[0], 0, 1);
        init_semaphore(bwd_dq_stage_remote_ready[1], 0, 1);
        init_semaphore(bwd_dq_k_remote_ready[0], 0, 1);
        init_semaphore(bwd_dq_k_remote_ready[1], 0, 1);
        init_semaphore(bwd_dq_issue_ready[0], 0, 1);
        init_semaphore(bwd_dq_issue_ready[1], 0, 1);
        init_semaphore(bwd_dq_src_finished[0], 0, 1);
        init_semaphore(bwd_dq_src_finished[1], 0, 1);
        init_semaphore(bwd_dq_src1_finished[0], 0, 1);
        init_semaphore(bwd_dq_src1_finished[1], 0, 1);
        init_semaphore(bwd_dq_tmem_ready[0], 0, 1);
        init_semaphore(bwd_dq_accum_done[0], 0, 1);
        init_semaphore(live_dp_dout_ready[0], 0, 1);
        init_semaphore(live_dp_dout_remote_ready[0], 0, 1);
        init_semaphore(live_dp_dout_finished[0], 0, 1);
        init_semaphore(live_dp_v_ready[0], 0, 1);
        init_semaphore(live_dp_v_ready[1], 0, 1);
        init_semaphore(live_dp_v_remote_ready[0], 0, 1);
        init_semaphore(live_dp_v_remote_ready[1], 0, 1);
        init_semaphore(live_dp_v_finished[0], 0, 1);
        init_semaphore(live_dp_v_finished[1], 0, 1);
        init_semaphore(live_dp_arrived[0], 0, 1);
        init_semaphore(live_dp_copy_done[0], 0, C::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < C::LOAD_STAGES; ++i) {
            init_semaphore(k_arrived[i], 0, 1);
            init_semaphore(k_finished[i], 0, 1);
            init_semaphore(k_sc_arrived[i], 0, 1);
            init_semaphore(k_sc_finished[i], 0, 1);
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM ||
        stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM) {
        fp4pv_zero_shared_tile_block(bwd_dq_p_dummy[0]);
        fp4pv_zero_shared_tile_block(bwd_dq_k_dummy[0]);
        __syncthreads();
        publish_shared_backing();
        publish_cluster_shared_backing_if_needed<C>();
        fp4pv_cluster_sync<C>();
        __syncthreads();
    }

    auto run_bwd_dq_issue_path = [&]() {
        if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
            if ((g.backward_dq_pad == nullptr && g.backward_dq_bshd == nullptr) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY)) {
                return;
            }
            const bool issue_warp = (cta_rank == 0) && (warpgroup::warpid() == 0);
            const bool issue_lane = issue_warp && warp::elect_leader();
            d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT_BASE);
            d_tt_bwd_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_bwd_p_sc>(BWD_DQ_P_SC_BASE);
            d_tt_bwd_k_sc k_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K_SC_BASE);
            int stage_phase_mask = 0;
            int src_phase_mask = 0;
            int stage_remote_phase_mask = 0;
            int k_remote_phase_mask = 0;
            int tmem_ready_count = 0;
            const bool bwd_dq_issue_tmem_active =
                (stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM);
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                const int m_tile = t_coord.y + cta_rank;
                if (m_tile >= tiles_m) {
                    continue;
                }
                const bool peer_cta_valid = (t_coord.y + 1) < tiles_m;
                const int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                if (issue_lane && bwd_dq_issue_tmem_active) {
                    zero_output_scratch_issue_lane<C>(tt_output);
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const int buf = idx & 1;
                    const int physical_k_iter = physical_k_iter_for(idx);
                    if (idx >= 2) {
                        const int src_phase = (src_phase_mask >> buf) & 1;
                        wait(bwd_dq_src_finished[buf], src_phase);
                        src_phase_mask ^= (1 << buf);
                    }
                    const int stage_phase = (stage_phase_mask >> buf) & 1;
                    wait(bwd_dq_stage_ready[buf], stage_phase);
                    stage_phase_mask ^= (1 << buf);
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            if (peer_cta_valid && (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
                                const int stage_remote_phase = (stage_remote_phase_mask >> buf) & 1;
                                tma::cluster::wait(bwd_dq_stage_remote_ready[buf], stage_remote_phase);
                                stage_remote_phase_mask ^= (1 << buf);
                            }
                        }
                    }
	                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
	                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
	                            warpgroup::arrive(bwd_dq_src_finished[buf]);
	                        }
	                        continue;
                    }
                    if (warpgroup::warpid() == 0) {
                        load_bwd_dq_k_col_fp4_tile<C>(bwd_dq_k_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                        load_bwd_dq_k_col_sc_tile<C>(bwd_dq_k_sc_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter);
                    }
                    warpgroup::sync(warpgroup::groupid() + 1);
                    fp4pv_publish_shared_backing_proxy_only();
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        publish_cluster_shared_backing_if_needed<C>();
                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(bwd_dq_k_remote_ready[buf], 0);
                        }
                    }
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            if (peer_cta_valid) {
                                const int k_remote_phase = (k_remote_phase_mask >> buf) & 1;
                                tma::cluster::wait(bwd_dq_k_remote_ready[buf], k_remote_phase);
                                k_remote_phase_mask ^= (1 << buf);
                            }
                        }
                    }
                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY)) {
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            warpgroup::arrive(bwd_dq_src_finished[buf]);
                        }
                        continue;
                    }
                    if (issue_warp) {
                        auto &p_sc_e8m0 = *reinterpret_cast<typename G::bwd_dq_sc_tile *>(&p_sc_stage[buf]);
                        fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm, p_sc_e8m0);
                        fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm, bwd_dq_k_sc_smem[buf]);
                        tensor_load_wait();
                        if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY)) {
                            tensor_before_thread_sync();
                            if (issue_lane) {
                                if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY)) {
                                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM)) {
                                        auto &dq_p_operand =
                                            ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                             (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM))
                                                ? bwd_dq_p_dummy[0]
                                                : p_fp4_stage[buf];
                                        auto &dq_k_operand =
                                            ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM))
                                                ? bwd_dq_k_dummy[0]
                                                : bwd_dq_k_smem[buf];
                                        if (idx == 0) {
                                            fp4pv_mm_unscaled_nosem_cluster_ABt<C>(tt_output, dq_p_operand, dq_k_operand);
                                        } else {
                                            fp4pv_mma_unscaled_nosem_cluster_ABt<C>(tt_output, dq_p_operand, dq_k_operand);
                                        }
                                    } else if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE)) {
                                        if (idx == 0) {
                                            fp4pv_mm_unscaled_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], bwd_dq_src_finished[buf]);
                                        } else {
                                            fp4pv_mma_unscaled_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], bwd_dq_src_finished[buf]);
                                        }
                                    } else {
                                        if (idx == 0) {
                                            fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
                                        } else {
                                            fp4pv_mma_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
                                        }
                                    }
                                    if ((stage_cut != DEBUG_STAGE_CUT_FRESH3_FULL)) {
                                        tensor_commit<C::CLUSTER_SIZE>(bwd_dq_tmem_ready[0]);
                                        ++tmem_ready_count;
                                    }
                                    if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) &&
                                        (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) &&
                                        idx == iters_per_task - 1) {
                                        detail::tcgen05::commit<C::CLUSTER_SIZE>(bwd_dq_accum_done[0]);
                                    }
                                }
                            }
                        }
                    }
                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY)) {
                        continue;
                    }
                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM)) {
                        continue;
                    }
                }
                (void)tmem_ready_count;
            }
        }
    };

    if (warpgroup::groupid() == 3) {
        if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<128>();
        }
        if (warpgroup::warpid() == 3) {
            const bool local_leader = warp::elect_leader();
            int k_idx = 0;
            int k_phase = 1;
            int k_sc_idx = 0;
            int k_sc_phase = 0;
            int q_phase = 1;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                const int q_tile_idx = min(t_coord.y + cta_rank, tiles_m - 1);
                if (local_leader) {
                    if (cta_rank < q_tiles_this_cluster) {
                        const coord<typename G::q_tile> q_coord = {t_coord.x, t_coord.z, q_tile_idx, 0};
                        const coord<typename G::q_sc_tile> q_sc_coord = {t_coord.x, q_tile_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0};
                        fp4pv_wait<C>(q_finished[0], q_phase);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(q_smem[0], g.q, q_coord, q_arrived[0], (uint16_t)(1 << cta_rank), 0);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(q_sc_smem, 0), g.q_sc, q_sc_coord, q_sc_arrived[0], (uint16_t)(1 << cta_rank), 0);
                    }
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const int physical_k_iter = physical_k_iter_for(idx);
                    const int k_tile_idx = min(physical_k_iter * C::CLUSTER_SIZE + cta_rank, tiles_m - 1);
                    if (local_leader) {
                        const coord<typename G::k_tile> k_coord = {t_coord.x, t_coord.z, k_tile_idx, 0};
                        const int k_sc_depth_idx = debug_k_sc_depth_index<C>(g, idx, cta_rank, tiles_m);
                        const coord<typename G::k_sc_tile> k_sc_coord = {t_coord.x, k_sc_depth_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0};
                        fp4pv_wait<C>(k_finished[k_idx], k_phase);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(k_smem[k_idx], g.k, k_coord, k_arrived[k_idx], (uint16_t)(1 << cta_rank), 0);
                        wait(k_sc_finished[k_sc_idx], k_sc_phase ^ 1);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(k_sc_smem, k_sc_idx), g.k_sc, k_sc_coord, k_sc_arrived[k_sc_idx], (uint16_t)(1 << cta_rank), 0);
                    }
                    k_idx++;
                    if (k_idx == C::LOAD_STAGES) {
                        k_idx = 0;
                        k_phase ^= 1;
                    }
                    k_sc_idx++;
                    if (k_sc_idx == C::K_SCALE_SMEM_SLOTS) {
                        k_sc_idx = 0;
                        k_sc_phase ^= 1;
                    }
                }
                q_phase ^= 1;
            }
        }
        if constexpr (LIVE_DP) {
            if constexpr (LIVE_DP_MODE0_ONLY) {
                if (warpgroup::warpid() == 1 && cta_rank != 0) {
                    const bool local_leader = warp::elect_leader();
                    int v_slot = 0;
                    int v_finished_phase_mask = 0;
                    int dout_finished_phase = 0;
                    for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, ++task_num) {
                        int3 t_coord = get_tile_idx(cur_bid);
                        int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                        if (iters_per_task <= 0) {
                            continue;
                        }
                        const int m_tile = t_coord.y + cta_rank;
                        const bool peer_row_valid = m_tile < tiles_m;
                        if (task_num > 0 && local_leader) {
                            wait(live_dp_dout_finished[0], dout_finished_phase);
                            dout_finished_phase ^= 1;
                        }
                        warp::sync();
                        if (peer_row_valid) {
                            load_live_dp_dout_row_fp4_tile_warp<C>(*live_dp_dout_smem, g, t_coord.x, t_coord.z, m_tile);
                            load_live_dp_dout_row_sc_tile_warp<C>(*live_dp_dout_sc_smem, g, t_coord.x, t_coord.z, m_tile);
                        } else {
                            fp4pv_zero_shared_tile_warp(*live_dp_dout_smem);
                            fp4pv_zero_shared_tile_warp(*live_dp_dout_sc_smem);
                        }
                        warp::sync();
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        publish_cluster_shared_backing_if_needed<C>();
                        warp::sync();
                        if (local_leader) {
                            arrive(live_dp_dout_ready[0]);
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if (cta_rank == 1) {
                                    tma::cluster::arrive(live_dp_dout_remote_ready[0], 0);
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int idx = 0; idx < iters_per_task; ++idx) {
                            const int physical_k_iter = physical_k_iter_for(idx);
                            if ((task_num > 0 || idx >= 2) && local_leader) {
                                const int v_finished_phase = (v_finished_phase_mask >> v_slot) & 1;
                                wait(live_dp_v_finished[v_slot], v_finished_phase);
                                v_finished_phase_mask ^= (1 << v_slot);
                            }
                            warp::sync();
                            load_live_dp_v_row_fp4_tile_warp<C>(live_dp_v_smem[v_slot], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                            load_live_dp_v_row_sc_tile_warp<C>(live_dp_v_sc_smem[v_slot], g, t_coord.x, t_coord.z, physical_k_iter);
                            warp::sync();
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            fp4pv_publish_shared_backing_proxy_only();
                            publish_cluster_shared_backing_if_needed<C>();
                            warp::sync();
                            if (local_leader) {
                                arrive(live_dp_v_ready[v_slot]);
                                if constexpr (C::CLUSTER_SIZE > 1) {
                                    if (cta_rank == 1) {
                                        tma::cluster::arrive(live_dp_v_remote_ready[v_slot], 0);
                                    }
                                }
                            }
                            v_slot ^= 1;
                        }
                    }
                }
            } else {
                if (warpgroup::warpid() == 1 && g.live_dp_debug_mode != 2 && cta_rank != 0) {
                    const bool local_leader = warp::elect_leader();
                    const bool local_only_debug =
                        (g.live_dp_debug_mode == 7 || g.live_dp_debug_mode == 8 ||
                         g.live_dp_debug_mode == 9 || g.live_dp_debug_mode == 10 ||
                         g.live_dp_debug_mode == 14 || g.live_dp_debug_mode == 17);
                    int v_slot = 0;
                    int v_finished_phase_mask = 0;
                    int dout_finished_phase = 0;
                    for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, ++task_num) {
                        int3 t_coord = get_tile_idx(cur_bid);
                        int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                        if (iters_per_task <= 0) {
                            continue;
                        }
                        const int m_tile = t_coord.y + cta_rank;
                        const bool peer_row_valid = m_tile < tiles_m;
                        const bool prepare_dout_source =
                            g.live_dp_debug_mode != 3 && g.live_dp_debug_mode != 9 &&
                            g.live_dp_debug_mode != 10;
                        if (task_num > 0 && prepare_dout_source && local_leader) {
                            wait(live_dp_dout_finished[0], dout_finished_phase);
                            dout_finished_phase ^= 1;
                        }
                        warp::sync();
                        if (peer_row_valid && prepare_dout_source) {
                            load_live_dp_dout_row_fp4_tile_warp<C>(*live_dp_dout_smem, g, t_coord.x, t_coord.z, m_tile);
                            load_live_dp_dout_row_sc_tile_warp<C>(*live_dp_dout_sc_smem, g, t_coord.x, t_coord.z, m_tile);
                        } else if (prepare_dout_source) {
                            fp4pv_zero_shared_tile_warp(*live_dp_dout_smem);
                            fp4pv_zero_shared_tile_warp(*live_dp_dout_sc_smem);
                        }
                        if (prepare_dout_source) {
                            warp::sync();
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            if (!local_only_debug) {
                                publish_cluster_shared_backing_if_needed<C>();
                            }
                            warp::sync();
                        }
                        if (prepare_dout_source && local_leader) {
                            arrive(live_dp_dout_ready[0]);
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if (!local_only_debug && cta_rank == 1) {
                                    tma::cluster::arrive(live_dp_dout_remote_ready[0], 0);
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int idx = 0; idx < iters_per_task; ++idx) {
                            const int physical_k_iter = physical_k_iter_for(idx);
                            if ((task_num > 0 || idx >= 2) && local_leader) {
                                const int v_finished_phase = (v_finished_phase_mask >> v_slot) & 1;
                                wait(live_dp_v_finished[v_slot], v_finished_phase);
                                v_finished_phase_mask ^= (1 << v_slot);
                            }
                            warp::sync();
                            if (g.live_dp_debug_mode != 4 && g.live_dp_debug_mode != 9) {
                                if (g.live_dp_debug_mode != 5 && g.live_dp_debug_mode != 7 &&
                                    g.live_dp_debug_mode != 10) {
                                    load_live_dp_v_row_fp4_tile_warp<C>(live_dp_v_smem[v_slot], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                                }
                                if (g.live_dp_debug_mode != 6 && g.live_dp_debug_mode != 8) {
                                    load_live_dp_v_row_sc_tile_warp<C>(live_dp_v_sc_smem[v_slot], g, t_coord.x, t_coord.z, physical_k_iter);
                                }
                            }
                            if (g.live_dp_debug_mode == 14) {
                                store_live_dp_source_debug_warp<C>(
                                    g, t_coord.x, t_coord.z, m_tile, physical_k_iter, cta_rank);
                            }
                            warp::sync();
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            fp4pv_publish_shared_backing_proxy_only();
                            if (!local_only_debug) {
                                publish_cluster_shared_backing_if_needed<C>();
                            }
                            warp::sync();
                            if (local_leader) {
                                arrive(live_dp_v_ready[v_slot]);
                                if constexpr (C::CLUSTER_SIZE > 1) {
                                    if (!local_only_debug && cta_rank == 1) {
                                        tma::cluster::arrive(live_dp_v_remote_ready[v_slot], 0);
                                    }
                                }
                            }
                            v_slot ^= 1;
                        }
                    }
                }
            }
        }
        if (warpgroup::warpid() == 0 && cta_rank == 0) {
            const bool issue_lane = warp::elect_leader();
            d_tt_scores tt_score = tm_alloc.template allocate<d_tt_scores>(0);
            constexpr int COMPACT_Q_SC_BASE = C::Nb;
            constexpr int COMPACT_K_SC_BASE = C::Nb + C::Q_SC_TMEM_WIDTH;
            constexpr int COMPACT_Q_SC_DBG_BASE = COMPACT_K_SC_BASE + 16;
            constexpr int COMPACT_K_SC_DBG_BASE = COMPACT_Q_SC_DBG_BASE + 32;
            constexpr int PRODUCTION_LIKE_Q_SC_BASE = C::Nb + C::Dvo;
            constexpr int PRODUCTION_LIKE_K_SC_BASE = C::Nb + C::Dvo + C::Q_SC_TMEM_WIDTH;
            constexpr int PRODUCTION_LIKE_Q_SC_DBG_BASE = PRODUCTION_LIKE_K_SC_BASE + 16;
            constexpr int PRODUCTION_LIKE_K_SC_DBG_BASE = PRODUCTION_LIKE_Q_SC_DBG_BASE + 32;
            static_assert(COMPACT_K_SC_DBG_BASE + 32 <= MAX_TENSOR_COLS, "debug compact TMEM layout exceeds TMEM budget");
            static_assert(PRODUCTION_LIKE_K_SC_DBG_BASE + 32 <= MAX_TENSOR_COLS, "debug production-like TMEM layout exceeds TMEM budget");
            const bool production_like_tmem_layout = g.debug_tmem_layout_mode == DEBUG_TMEM_LAYOUT_PRODUCTION_LIKE;
            d_tt_q_sc q_sc_tm = production_like_tmem_layout
                ? tm_alloc.template allocate<d_tt_q_sc>(PRODUCTION_LIKE_Q_SC_BASE)
                : tm_alloc.template allocate<d_tt_q_sc>(COMPACT_Q_SC_BASE);
            d_tt_sc16 k_sc_tm = production_like_tmem_layout
                ? tm_alloc.template allocate<d_tt_sc16>(PRODUCTION_LIKE_K_SC_BASE)
                : tm_alloc.template allocate<d_tt_sc16>(COMPACT_K_SC_BASE);
            d_tt_bwd_p_sc live_dp_dout_sc_tm = tm_alloc.template allocate<d_tt_bwd_p_sc>(LIVE_DP_DOUT_SC_BASE);
            d_tt_bwd_k_sc live_dp_v_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(LIVE_DP_V_SC_BASE);
            full_tt_fp8e4m3<32> q_sc_tm_dbg = production_like_tmem_layout
                ? tm_alloc.template allocate<full_tt_fp8e4m3<32>>(PRODUCTION_LIKE_Q_SC_DBG_BASE)
                : tm_alloc.template allocate<full_tt_fp8e4m3<32>>(COMPACT_Q_SC_DBG_BASE);
            full_tt_fp8e4m3<32> k_sc_tm_dbg = production_like_tmem_layout
                ? tm_alloc.template allocate<full_tt_fp8e4m3<32>>(PRODUCTION_LIKE_K_SC_DBG_BASE)
                : tm_alloc.template allocate<full_tt_fp8e4m3<32>>(COMPACT_K_SC_DBG_BASE);
            auto q_sc_tm_chunk = q_sc_tm.template subtile<full_tt_fp8e4m3<16>>(0);
            int k_idx = 0, k_phase = 0, k_sc_slot = 0;
            int k_sc_arrived_phase_mask = 0;
	            int current_b = 0;
	            int current_h = 0;
	            int current_m_tile = 0;
			            bool current_peer_cta_valid = false;
			            bool current_peer_live_dout_source_ready = false;
			            bool current_peer_live_v_valid = false;
            int live_dp_dout_ready_phase = 0;
            int live_dp_v_ready_phase_mask = 0;
            int live_dp_v_debug_done_phase_mask = 0;
            int live_dp_copy_phase = 0;
            // The debug softmax consumer serializes score-tile extraction through one live
            // reuse slot, so the producer must also treat p_copy_done as a single-buffer
            // handshake and wait for the first score copy before reissuing QK.
            int p_copy_phase_mask = 1;
            auto dump_q_sc_tmem_chunk = [&](int b, int h, int m_tile, int chunk) {
                if (g.debug_q_sc_tmem_loaded == nullptr) {
                    return;
                }
                auto q_sc_tm_dbg_lo = q_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(0);
                auto q_sc_tm_dbg_hi = q_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(16);
                if (issue_lane) {
                    load_q_scale_chunk<C>(q_sc_tm_dbg_lo, slot_tile_at(q_sc_smem, 0), chunk);
                    load_q_scale_chunk<C>(q_sc_tm_dbg_hi, slot_tile_at(q_sc_smem, 0), chunk);
                }
                __syncwarp();
                copy_scale_tt_to_shared(scale_dbg_smem[0], q_sc_tm_dbg);
                __syncwarp();
                if (issue_lane) {
                    store_debug_q_sc_tmem_chunk_if_needed<C>(g, scale_dbg_smem[0], b, h, m_tile, chunk);
                }
                __syncwarp();
            };
            auto dump_k_sc_tmem_chunk = [&](int b, int h, int k_iter, int chunk) {
                if (g.debug_k_sc_tmem_loaded == nullptr) {
                    return;
                }
                auto k_sc_tm_dbg_lo = k_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(0);
                auto k_sc_tm_dbg_hi = k_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(16);
                if (issue_lane) {
                    load_k_scale_chunk<C>(k_sc_tm_dbg_lo, slot_tile_at(k_sc_smem, k_sc_slot), chunk);
                    load_k_scale_chunk<C>(k_sc_tm_dbg_hi, slot_tile_at(k_sc_smem, k_sc_slot), chunk);
                }
                __syncwarp();
                copy_scale_tt_to_shared(scale_dbg_smem[0], k_sc_tm_dbg);
                __syncwarp();
                if (issue_lane) {
                    store_debug_k_sc_tmem_chunk_if_needed<C>(g, scale_dbg_smem[0], b, h, k_iter, chunk);
                }
                __syncwarp();
            };
            auto issue_live_dp_after_score_copy = [&](int prev_idx, bool final_live_dp_for_task) {
                if constexpr (LIVE_DP) {
                    if constexpr (LIVE_DP_MODE0_ONLY) {
                        const int v_slot = prev_idx & 1;
                        const int prev_physical_k_iter = physical_k_iter_for(prev_idx);
                        if (issue_lane) {
                            const int qk_done_slot = prev_idx % C::LOAD_STAGES;
                            const int qk_done_phase = (prev_idx / C::LOAD_STAGES) & 1;
                            fp4pv_wait<C>(k_finished[qk_done_slot], qk_done_phase);
                        }
                        if (prev_idx == 0) {
                            load_live_dp_dout_row_fp4_tile_warp<C>(
                                *live_dp_dout_smem, g, current_b, current_h, current_m_tile);
                            load_live_dp_dout_row_sc_tile_warp<C>(
                                *live_dp_dout_sc_smem, g, current_b, current_h, current_m_tile);
                        }
                        load_live_dp_v_row_fp4_tile_warp<C>(
                            live_dp_v_smem[v_slot], g, current_b, current_h,
                            prev_physical_k_iter, cta_rank);
                        load_live_dp_v_row_sc_tile_warp<C>(
                            live_dp_v_sc_smem[v_slot], g, current_b, current_h,
                            prev_physical_k_iter);
                        __syncwarp(__activemask());
                        __threadfence_block();
                        fp4pv_publish_shared_backing_proxy_only();
                        publish_cluster_shared_backing_if_needed<C>();
                        if (prev_idx == 0) {
                            if (issue_lane) {
                                if constexpr (C::CLUSTER_SIZE > 1) {
                                    if (current_peer_live_dout_source_ready) {
                                        tma::cluster::wait(live_dp_dout_remote_ready[0], live_dp_dout_ready_phase);
                                    }
                                }
                                live_dp_dout_ready_phase ^= 1;
                            }
                        }
                        if (issue_lane) {
                            const int v_ready_phase = (live_dp_v_ready_phase_mask >> v_slot) & 1;
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if (current_peer_live_v_valid) {
                                    tma::cluster::wait(live_dp_v_remote_ready[v_slot], v_ready_phase);
                                }
                            }
                            live_dp_v_ready_phase_mask ^= (1 << v_slot);
                        }
                        __syncwarp(__activemask());
                        if (issue_lane) {
                            if (prev_idx == 0) {
                                fp4pv_load_mxnv_scale_async_cluster<C>(live_dp_dout_sc_tm, *live_dp_dout_sc_smem);
                                tensor_load_wait();
                            }
                            fp4pv_load_mxnv_scale_async_cluster<C>(live_dp_v_sc_tm, live_dp_v_sc_smem[v_slot]);
                            tensor_load_wait();
                            tensor_before_thread_sync();
                            fp4pv_mm_cluster_ABt<C>(
                                tt_score,
                                *live_dp_dout_smem,
                                live_dp_v_smem[v_slot],
                                live_dp_dout_sc_tm,
                                live_dp_v_sc_tm,
                                live_dp_v_finished[v_slot]);
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(live_dp_arrived[0]);
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if (!current_peer_cta_valid) {
                                    arrive(live_dp_copy_done[0]);
                                }
                            }
                            wait(live_dp_copy_done[0], live_dp_copy_phase);
                            live_dp_copy_phase ^= 1;
                            if (final_live_dp_for_task) {
                                arrive(live_dp_dout_finished[0]);
                                fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
                            }
                        }
                        return;
                    }
                    if (g.live_dp_debug_mode == 2) {
                        if (issue_lane && final_live_dp_for_task) {
                            arrive(live_dp_dout_finished[0]);
                            fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
                        }
                        return;
                    }
                    const bool local_only_debug =
                        (g.live_dp_debug_mode == 7 || g.live_dp_debug_mode == 8 ||
                         g.live_dp_debug_mode == 9 || g.live_dp_debug_mode == 10 ||
                         g.live_dp_debug_mode == 14 || g.live_dp_debug_mode == 17);
                    const bool skip_live_ready_waits = (g.live_dp_debug_mode == 12);
                    const int v_slot = prev_idx & 1;
                    const bool skip_live_self_stage =
                        skip_live_ready_waits || g.live_dp_debug_mode == 9;
                    const int prev_physical_k_iter = physical_k_iter_for(prev_idx);
                    if (issue_lane) {
                        const int qk_done_slot = prev_idx % C::LOAD_STAGES;
                        const int qk_done_phase = (prev_idx / C::LOAD_STAGES) & 1;
                        fp4pv_wait<C>(k_finished[qk_done_slot], qk_done_phase);
                    }
                    if (!skip_live_self_stage) {
                        if (prev_idx == 0 &&
                            g.live_dp_debug_mode != 3 && g.live_dp_debug_mode != 9 &&
                            g.live_dp_debug_mode != 10) {
                            load_live_dp_dout_row_fp4_tile_warp<C>(
                                *live_dp_dout_smem, g, current_b, current_h, current_m_tile);
                            load_live_dp_dout_row_sc_tile_warp<C>(
                                *live_dp_dout_sc_smem, g, current_b, current_h, current_m_tile);
                        }
                        if (g.live_dp_debug_mode != 4 && g.live_dp_debug_mode != 9) {
                            if (g.live_dp_debug_mode != 5 && g.live_dp_debug_mode != 7 &&
                                g.live_dp_debug_mode != 10) {
                                load_live_dp_v_row_fp4_tile_warp<C>(
                                    live_dp_v_smem[v_slot], g, current_b, current_h,
                                    prev_physical_k_iter, cta_rank);
                            }
                            if (g.live_dp_debug_mode != 6 && g.live_dp_debug_mode != 8) {
                                load_live_dp_v_row_sc_tile_warp<C>(
                                    live_dp_v_sc_smem[v_slot], g, current_b, current_h,
                                    prev_physical_k_iter);
                            }
                        }
                        if (g.live_dp_debug_mode == 14) {
                            store_live_dp_source_debug_warp<C>(
                                g, current_b, current_h, current_m_tile, prev_physical_k_iter, cta_rank);
                        }
	                        __syncwarp(__activemask());
	                        __threadfence_block();
	                        fp4pv_publish_shared_backing_proxy_only();
                            if (!local_only_debug) {
                                publish_cluster_shared_backing_if_needed<C>();
                            }
	                        if (g.live_dp_debug_mode == 17) {
	                            store_live_dp_shared_debug_warp<C>(
                                g, *live_dp_dout_smem, *live_dp_dout_sc_smem,
                                live_dp_v_smem[v_slot], live_dp_v_sc_smem[v_slot],
                                current_b, current_h, current_m_tile);
                        }
                    }
			                    if (prev_idx == 0) {
		                        if (issue_lane && !skip_live_ready_waits) {
		                            if constexpr (C::CLUSTER_SIZE > 1) {
				                                if (!local_only_debug && current_peer_live_dout_source_ready) {
			                                    tma::cluster::wait(live_dp_dout_remote_ready[0], live_dp_dout_ready_phase);
			                                }
	                            }
	                            live_dp_dout_ready_phase ^= 1;
		                        }
		                    }
			                    if (issue_lane) {
			                        const int v_ready_phase = (live_dp_v_ready_phase_mask >> v_slot) & 1;
			                        if (!skip_live_ready_waits) {
			                            if constexpr (C::CLUSTER_SIZE > 1) {
				                                if (!local_only_debug && current_peer_live_v_valid) {
				                                    tma::cluster::wait(live_dp_v_remote_ready[v_slot], v_ready_phase);
				                                }
			                            }
			                            live_dp_v_ready_phase_mask ^= (1 << v_slot);
			                        }
	                                }
	                                __syncwarp(__activemask());
                                __syncwarp(__activemask());
				                    if (issue_lane) {
			                        if (g.live_dp_debug_mode == 1 || g.live_dp_debug_mode == 3 ||
			                            g.live_dp_debug_mode == 4 || g.live_dp_debug_mode == 5 ||
			                            g.live_dp_debug_mode == 6 || g.live_dp_debug_mode == 7 ||
			                            g.live_dp_debug_mode == 8 || g.live_dp_debug_mode == 9 ||
			                            g.live_dp_debug_mode == 10 || g.live_dp_debug_mode == 12 ||
	                                    g.live_dp_debug_mode == 14 || g.live_dp_debug_mode == 17) {
		                                arrive(live_dp_v_finished[v_slot]);
		                                fp4pv_remote_arrive_if_needed<C>(live_dp_v_finished[v_slot], 1);
		                                if (final_live_dp_for_task) {
		                                    arrive(live_dp_dout_finished[0]);
		                                    fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
			                            }
			                            return;
			                        }
			                        if (g.live_dp_debug_mode == 13) {
			                            tensor_before_thread_sync();
				                        fp4pv_mm_unscaled_cluster_ABt<C>(
			                                tt_score,
			                                *live_dp_dout_smem,
			                                live_dp_v_smem[v_slot],
			                                live_dp_v_finished[v_slot]);
			                        } else {
				                            constexpr uint32_t packed_unit_scale = 0x7f7f7f7fu;
                                            const bool unit_dout_scale =
                                                g.live_dp_debug_mode == 18 || g.live_dp_debug_mode == 19;
                                            const bool unit_v_scale =
                                                g.live_dp_debug_mode == 18 || g.live_dp_debug_mode == 20;
                                            if (unit_dout_scale) {
                                                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(live_dp_dout_sc_tm, packed_unit_scale);
                                            } else {
                                                fp4pv_load_mxnv_scale_async_cluster<C>(live_dp_dout_sc_tm, *live_dp_dout_sc_smem);
                                                tensor_load_wait();
                                            }
                                            if (unit_v_scale) {
                                                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(live_dp_v_sc_tm, packed_unit_scale);
                                            } else {
                                                fp4pv_load_mxnv_scale_async_cluster<C>(live_dp_v_sc_tm, live_dp_v_sc_smem[v_slot]);
                                                tensor_load_wait();
                                            }
                                            if (unit_dout_scale || unit_v_scale) {
                                                fp4pv_tmem_store_wait();
                                            }
				                            tensor_before_thread_sync();
			                            fp4pv_mm_cluster_ABt<C>(
			                                tt_score,
			                                *live_dp_dout_smem,
			                                live_dp_v_smem[v_slot],
			                                live_dp_dout_sc_tm,
			                                live_dp_v_sc_tm,
			                                live_dp_v_finished[v_slot]);
			                        }
                                if (g.live_dp_debug_mode == 15) {
                                    const int done_phase = (live_dp_v_debug_done_phase_mask >> v_slot) & 1;
                                    wait(live_dp_v_finished[v_slot], done_phase);
                                    live_dp_v_debug_done_phase_mask ^= (1 << v_slot);
                                    if (g.backward_dp_bf16 != nullptr) {
                                        const int heads = g.q.depth();
                                        const int seqlen = g.q.rows();
                                        const int row = current_m_tile * C::Mb + warp::laneid();
                                        const int col_base = prev_physical_k_iter * C::Nb;
                                        const int64_t dp_row_base =
                                            (((static_cast<int64_t>(current_b) * heads + current_h) * seqlen) + row) * seqlen;
                                        #pragma unroll
                                        for (int qid = 0; qid < 4; ++qid) {
                                            float2 dp_chunk[16];
                                            fp4pv_load_score_quarter_tmem(dp_chunk, tt_score.addr, qid);
                                            if (g.live_dp_debug_mode != 13) {
                                                constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                                const float2 alpha_2 = {MXFP4_ALPHA, MXFP4_ALPHA};
                                                #pragma unroll
                                                for (int jj = 0; jj < 16; ++jj) {
                                                    dp_chunk[jj] = __fmul2_rn(dp_chunk[jj], alpha_2);
                                                }
                                            }
                                            #pragma unroll
                                            for (int jj = 0; jj < 16; ++jj) {
                                                const int col = col_base + (qid * 16 + jj) * 2;
                                                const bf16_2 dp_pair_bf = __float22bfloat162_rn(dp_chunk[jj]);
                                                *reinterpret_cast<bf16_2 *>(&g.backward_dp_bf16[dp_row_base + col]) = dp_pair_bf;
                                            }
                                        }
                                    }
                                    if (final_live_dp_for_task) {
                                        arrive(live_dp_dout_finished[0]);
                                        fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
                                    }
                                    return;
                                }
		                        detail::tcgen05::commit<C::CLUSTER_SIZE>(live_dp_arrived[0]);
		                        if constexpr (C::CLUSTER_SIZE > 1) {
		                            if (!current_peer_cta_valid) {
		                                arrive(live_dp_copy_done[0]);
		                            }
		                        }
		                        wait(live_dp_copy_done[0], live_dp_copy_phase);
		                        live_dp_copy_phase ^= 1;
		                        if (final_live_dp_for_task) {
		                            arrive(live_dp_dout_finished[0]);
		                            fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
		                        }
		                    }
	                    (void)prev_physical_k_iter;
	                }
            };
            auto issue_next_qk = [&](bool final_q_for_task, int copy_buf, int k_iter, int iters_per_task) -> bool {
                (void)copy_buf;
                constexpr bool ENABLE_DEBUG_STAGE_CUTS = QK_ONLY && !SAVED_LSE_BWD_ONLY;
                const bool producer_third_issue_cut =
                    ENABLE_DEBUG_STAGE_CUTS &&
                    g.max_k_iters == 3 &&
                    final_q_for_task &&
                    stage_cut >= DEBUG_STAGE_CUT_PRODUCER3_POST_WAIT &&
                    stage_cut <= DEBUG_STAGE_CUT_FRESH3_FULL;
                constexpr int reuse_buf = 0;
                const int copy_phase = (p_copy_phase_mask >> reuse_buf) & 1;
                if (issue_lane) {
                    wait(p_copy_done[reuse_buf], copy_phase);
                }
                p_copy_phase_mask ^= (1 << reuse_buf);
                issue_live_dp_after_score_copy(k_iter - 1, false);
                if (producer_third_issue_cut && stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_WAIT) {
                    return true;
                }
                const int next_k_slot = k_idx;
                if (issue_lane) {
                    fp4pv_expect_bytes<C>(k_arrived[next_k_slot], C::CLUSTER_SIZE * sizeof(typename G::k_tile));
                    fp4pv_wait<C>(k_arrived[next_k_slot], k_phase);
                    fp4pv_expect_bytes<C>(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(typename G::k_sc_tile));
                }
                {
                    const int k_sc_phase = (k_sc_arrived_phase_mask >> k_sc_slot) & 1;
                    if (issue_lane) {
                        fp4pv_wait<C>(k_sc_arrived[k_sc_slot], k_sc_phase);
                    }
                    k_sc_arrived_phase_mask ^= (1 << k_sc_slot);
                }
                if (issue_lane) {
                    const int physical_k_iter = physical_k_iter_for(k_iter);
                    if constexpr (!fixed_fresh3_full) {
                        store_debug_k_sc_tile_if_needed<C>(g, slot_tile_at(k_sc_smem, k_sc_slot), current_b, current_h, physical_k_iter);
                    }
                }
                if constexpr (!fixed_fresh3_full) {
                    #pragma unroll
                    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                        if (debug_qk_chunk_enabled<C>(g, chunk)) {
                            dump_k_sc_tmem_chunk(current_b, current_h, physical_k_iter_for(k_iter), chunk);
                        }
                    }
                }
                if (issue_lane) {
                    if constexpr (fixed_fresh3_full) {
                        fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
                            tt_score, q_smem[0], k_smem[next_k_slot],
                            q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm,
                            k_finished[next_k_slot]);
                    } else {
                        if (debug_use_q_scale_tmem(g)) {
                            issue_qk_chunked_qsc_tmem_debug_masked<C>(tt_score, q_smem[0], k_smem[next_k_slot], q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[next_k_slot], g);
                        } else {
                            issue_qk_chunked_debug_masked<C>(tt_score, q_smem[0], k_smem[next_k_slot], slot_tile_at(q_sc_smem, 0), slot_tile_at(k_sc_smem, k_sc_slot), q_sc_tm_chunk, k_sc_tm, k_finished[next_k_slot], g);
                        }
                    }
                }
                if (producer_third_issue_cut && stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_ISSUE) {
                    return true;
                }
                const bool skip_scores_commit =
                    producer_third_issue_cut &&
                    stage_cut == DEBUG_STAGE_CUT_PRODUCER3_SKIP_COMMIT_TAIL;
                const bool fresh_scores_commit =
                    producer_third_issue_cut &&
                    (stage_cut == DEBUG_STAGE_CUT_PRODUCER3_FRESH_COMMIT_TAIL ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_WAIT_ONLY ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_COPY ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL);
                if (issue_lane) {
                    if (fresh_scores_commit) {
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived_fresh[0]);
                    } else if (!skip_scores_commit) {
                        if constexpr (SAVED_LSE_BWD_ONLY) {
                            if (k_iter & 1) {
                                detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived_fresh[0]);
                            } else {
                                detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                            }
                        } else {
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(
                                debug_scores_arrived_sem_for_iter<C>(scores_arrived, scores_arrived_fresh, g, k_iter)
                            );
                        }
                    }
                }
                if (producer_third_issue_cut && stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_COMMIT) {
                    return true;
                }
                if (issue_lane) {
                    arrive(k_sc_finished[k_sc_slot]);
                    fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1);
                }
                if (producer_third_issue_cut && stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_KSC_FINISH) {
                    return true;
                }
                if (issue_lane && final_q_for_task) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
                }
                const bool producer_fresh_tail_cut =
                    ENABLE_DEBUG_STAGE_CUTS &&
                    stage_cut == DEBUG_STAGE_CUT_PRODUCER3_FRESH_COMMIT_TAIL;
                if (skip_scores_commit || producer_fresh_tail_cut) {
                    return true;
                }
                k_idx ^= 1;
                if (k_idx == 0) {
                    k_phase ^= 1;
                }
                k_sc_slot ^= 1;
                return false;
            };

            for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, ++task_num) {
                int3 t_coord = get_tile_idx(cur_bid);
                current_b = t_coord.x;
	                current_h = t_coord.z;
	                current_m_tile = t_coord.y;
			                current_peer_cta_valid = (t_coord.y + 1) < tiles_m;
			                current_peer_live_dout_source_ready =
			                    g.live_dp_debug_mode != 3 && g.live_dp_debug_mode != 9 &&
			                    g.live_dp_debug_mode != 10;
			                current_peer_live_v_valid = true;
	                int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                int q_phase = task_num & 1;
                if (issue_lane) {
                    fp4pv_expect_bytes<C>(q_arrived[0], q_tiles_this_cluster * sizeof(typename G::q_tile));
                    fp4pv_wait<C>(q_arrived[0], q_phase);
                    fp4pv_expect_bytes<C>(q_sc_arrived[0], q_tiles_this_cluster * sizeof(typename G::q_sc_tile));
                    fp4pv_wait<C>(q_sc_arrived[0], q_phase);
                    if constexpr (!fixed_fresh3_full) {
                        store_debug_q_sc_tile_if_needed<C>(g, slot_tile_at(q_sc_smem, 0), current_b, current_h, t_coord.y);
                    }
                    if constexpr (fixed_fresh3_full) {
                        fp4pv_stage_q_scale_tmem_cluster<C>(q_sc_tm, slot_tile_at(q_sc_smem, 0));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                    } else if (debug_use_q_scale_tmem(g)) {
                        fp4pv_stage_q_scale_tmem_cluster<C>(q_sc_tm, slot_tile_at(q_sc_smem, 0));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                    }
                }
                if constexpr (!fixed_fresh3_full) {
                    #pragma unroll
                    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                        if (debug_qk_chunk_enabled<C>(g, chunk)) {
                            dump_q_sc_tmem_chunk(current_b, current_h, t_coord.y, chunk);
                        }
                    }
                }

                int k_slot = k_idx;
                if (issue_lane) {
                    fp4pv_expect_bytes<C>(k_arrived[k_slot], C::CLUSTER_SIZE * sizeof(typename G::k_tile));
                    fp4pv_wait<C>(k_arrived[k_slot], k_phase);
                    fp4pv_expect_bytes<C>(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(typename G::k_sc_tile));
                }
                {
                    const int k_sc_phase = (k_sc_arrived_phase_mask >> k_sc_slot) & 1;
                    if (issue_lane) {
                        fp4pv_wait<C>(k_sc_arrived[k_sc_slot], k_sc_phase);
                    }
                    k_sc_arrived_phase_mask ^= (1 << k_sc_slot);
                }
                if (issue_lane) {
                    if constexpr (!fixed_fresh3_full) {
                        store_debug_k_sc_tile_if_needed<C>(g, slot_tile_at(k_sc_smem, k_sc_slot), t_coord.x, t_coord.z, physical_k_iter_for(0));
                    }
                }
                if constexpr (!fixed_fresh3_full) {
                    #pragma unroll
                    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                        if (debug_qk_chunk_enabled<C>(g, chunk)) {
                            dump_k_sc_tmem_chunk(t_coord.x, t_coord.z, physical_k_iter_for(0), chunk);
                        }
                    }
                }
	                if (issue_lane) {
	                    if constexpr (fixed_fresh3_full) {
	                        fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
	                            tt_score, q_smem[0], k_smem[k_slot],
	                            q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm,
	                            k_finished[k_slot]);
	                    } else {
	                        if (debug_use_q_scale_tmem(g)) {
	                            issue_qk_chunked_qsc_tmem_debug_masked<C>(tt_score, q_smem[0], k_smem[k_slot], q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[k_slot], g);
	                        } else {
                                issue_qk_chunked_debug_masked<C>(tt_score, q_smem[0], k_smem[k_slot], slot_tile_at(q_sc_smem, 0), slot_tile_at(k_sc_smem, k_sc_slot), q_sc_tm_chunk, k_sc_tm, k_finished[k_slot], g);
                            }
                    }
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                    arrive(k_sc_finished[k_sc_slot]);
                    fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1);
	                    if (iters_per_task == 1) {
	                        detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
	                    }
	                }
	                k_idx ^= 1;
                if (k_idx == 0) {
                    k_phase ^= 1;
                }
                k_sc_slot ^= 1;

                bool stop_after_producer_stage_cut = false;
                if (iters_per_task > 1) {
                    stop_after_producer_stage_cut = issue_next_qk(iters_per_task == 2, 0, 1, iters_per_task);
                    for (int idx = 1; idx < iters_per_task - 1; ++idx) {
                        if (stop_after_producer_stage_cut) {
                            break;
                        }
                        stop_after_producer_stage_cut = issue_next_qk(idx == iters_per_task - 2, idx & 1, idx + 1, iters_per_task);
                    }
                }
                if (stop_after_producer_stage_cut) {
                    break;
                }
                if constexpr (LIVE_DP) {
                    const int reuse_buf = 0;
                    const int copy_phase = (p_copy_phase_mask >> reuse_buf) & 1;
                    if (issue_lane) {
                        wait(p_copy_done[reuse_buf], copy_phase);
                    }
                    p_copy_phase_mask ^= (1 << reuse_buf);
                    issue_live_dp_after_score_copy(iters_per_task - 1, true);
                }
            }
        }
    } else if (warpgroup::groupid() == 1) {
        warpgroup::increase_registers<168>();
        d_tt_scores tt_scores = tm_alloc.template allocate<d_tt_scores>(0);
        int scores_phase = 0;
        int rescale_phase = 1;
        int bwd_dq_src_phase_mask = 0;
        int bwd_dq_src1_phase_mask = 0;
        int live_dp_phase = 0;
        // The dump-only path is debug-focused, so we serialize P-stage reuse through a single
        // buffer to avoid multi-iteration phase hazards that do not exist in the fused kernel.
            int p_store_phase = 1;
            uint32_t score_tt_base = tt_scores.addr + ((warpgroup::warpid() * 32) << 16);
            auto arrive_live_dp_copy_done = [&]() {
                // This releases the producer to reuse tt_scores for live dP. If this warp
                // loaded live dP from TMEM, call tensor_load_wait() before this handoff.
                warpgroup::sync(warpgroup::groupid() + 1);
                if constexpr (C::CLUSTER_SIZE > 1) {
                    if (cta_rank == 0) {
                        warpgroup::arrive(live_dp_copy_done[0]);
                    } else if (cta_rank == 1) {
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(live_dp_copy_done[0], 0);
                        }
                    }
                } else {
                    warpgroup::arrive(live_dp_copy_done[0]);
                }
            };
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                int producer_iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (producer_iters_per_task <= 0) {
                    continue;
                }
                int m_tile = t_coord.y + cta_rank;
                if (m_tile >= tiles_m) {
                    continue;
                }
                if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY && SAVED_BWD_EMIT_COL_MXFP4) {
                    zero_backward_ds_col_mxfp4_future_scale_tiles<C>(
                        g, t_coord.x, t_coord.z, m_tile);
                }
                if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY && SAVED_BWD_EMIT_ROW_MXFP4) {
                    if (!g.backward_skip_global_ds_row_store) {
                        zero_backward_ds_row_mxfp4_future_scale_tiles<C>(
                            g, t_coord.x, t_coord.z, m_tile);
                    }
                }
                int consumer_iters_per_task = SAVED_LSE_BWD_ONLY
                    ? producer_iters_per_task
                    : debug_consumer_k_iters<C>(g, producer_iters_per_task);
            const float SCALE_LOG2 = qk_scale_log2(g, t_coord.x, t_coord.z);
            const bool ZERO_SCALE = (SCALE_LOG2 == 0.0f);
            float row_sum = 0.0f;
            float row_max = base_types::constants<float>::neg_infty();
            bool stop_after_stage_cut = false;
            const bool bwd_dq_stage_consumer_active =
                fixed_full_production_path
                    ? true
                    : ((g.backward_dq_pad != nullptr || g.backward_dq_bshd != nullptr) &&
                       (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY));
            const bool bwd_dq_nosem_full =
                stage_cut == DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL;
            constexpr bool bwd_dq_dual_halves = SAVED_BWD_DQ_DUAL_HALVES;
	            const bool bwd_dq_src_completion_active =
                    fixed_full_production_path
                        ? true
                        : (bwd_dq_stage_consumer_active &&
                           !bwd_dq_nosem_full &&
                           ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE)));
	            const bool bwd_dq_src1_completion_active =
                    fixed_full_production_path
                        ? bwd_dq_dual_halves
                        : (bwd_dq_dual_halves &&
                           bwd_dq_stage_consumer_active &&
                           !bwd_dq_nosem_full &&
                           ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE)));
            auto wait_bwd_dq_src_if_needed = [&](int bwd_dq_buf, int idx) {
                if constexpr (fixed_full_production_path) {
                    if (idx >= 2) {
                        const int src_phase = (bwd_dq_src_phase_mask >> bwd_dq_buf) & 1;
                        wait(bwd_dq_src_finished[bwd_dq_buf], src_phase);
                        bwd_dq_src_phase_mask ^= (1 << bwd_dq_buf);
                        if constexpr (SAVED_BWD_DQ_DUAL_HALVES) {
                            const int src1_phase = (bwd_dq_src1_phase_mask >> bwd_dq_buf) & 1;
                            wait(bwd_dq_src1_finished[bwd_dq_buf], src1_phase);
                            bwd_dq_src1_phase_mask ^= (1 << bwd_dq_buf);
                        }
                    }
                } else {
                    if (bwd_dq_src_completion_active && idx >= 2) {
                        const int src_phase = (bwd_dq_src_phase_mask >> bwd_dq_buf) & 1;
                        wait(bwd_dq_src_finished[bwd_dq_buf], src_phase);
                        bwd_dq_src_phase_mask ^= (1 << bwd_dq_buf);
                        if (bwd_dq_src1_completion_active) {
                            const int src1_phase = (bwd_dq_src1_phase_mask >> bwd_dq_buf) & 1;
                            wait(bwd_dq_src1_finished[bwd_dq_buf], src1_phase);
                            bwd_dq_src1_phase_mask ^= (1 << bwd_dq_buf);
                        }
                    }
                }
            };
            auto publish_bwd_dq_stage_ready = [&](int bwd_dq_buf) {
                if constexpr (fixed_full_production_path) {
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        publish_cluster_shared_backing_if_needed<C>();
                        if (cta_rank == 1 && warpgroup::laneid() == 0) {
                            tma::cluster::arrive(bwd_dq_stage_remote_ready[bwd_dq_buf], 0);
                        }
                    }
                    warpgroup::arrive(bwd_dq_stage_ready[bwd_dq_buf]);
                } else {
                    if (bwd_dq_stage_consumer_active) {
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
                                publish_cluster_shared_backing_if_needed<C>();
                                if (cta_rank == 1 && warpgroup::laneid() == 0) {
                                    tma::cluster::arrive(bwd_dq_stage_remote_ready[bwd_dq_buf], 0);
                                }
                            }
                        }
                        warpgroup::arrive(bwd_dq_stage_ready[bwd_dq_buf]);
                    }
                }
            };
            if constexpr (!QK_ONLY) {
                wait(rescale_finished[0], rescale_phase);
                rescale_phase ^= 1;
            }

            for (int idx = 0; idx < consumer_iters_per_task; ++idx) {
                constexpr int buf = 0;
                const int bwd_dq_buf = idx & 1;
                if constexpr (!QK_ONLY) {
                    wait(p_store_done[buf], p_store_phase);
                    p_store_phase ^= 1;
                }
                float2 scores_reg[C::Nb / 2];
                float tile_max = 0.0f;
                const int global_row = warpgroup::laneid();
                const int softmax_warp = warpgroup::laneid() / 32;
                const int physical_k_iter = physical_k_iter_for(idx);
                const bool third_iter_stage_cut =
                    QK_ONLY &&
                    !SAVED_LSE_BWD_ONLY &&
                    producer_iters_per_task == 3 &&
                    idx == 2 &&
                    !debug_full_like_mode(g);
                const bool fresh_consume_cut =
                    third_iter_stage_cut &&
                    (stage_cut == DEBUG_STAGE_CUT_FRESH3_WAIT_ONLY ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_COPY ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL);
                const int wait_phase = SAVED_LSE_BWD_ONLY
                    ? ((idx >> 1) & 1)
                    : debug_scores_arrived_phase_for_iter<C>(g, idx, scores_phase);
                if constexpr (SAVED_LSE_BWD_ONLY) {
                    if (idx & 1) {
                        wait(scores_arrived_fresh[0], wait_phase);
                    } else {
                        wait(scores_arrived[0], wait_phase);
                    }
                } else {
                    if (fresh_consume_cut) {
                        wait(scores_arrived_fresh[0], wait_phase);
                    } else {
                        wait(debug_scores_arrived_sem_for_iter<C>(scores_arrived, scores_arrived_fresh, g, idx), wait_phase);
                    }
                }
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_WAIT_ONLY) {
                    stop_after_stage_cut = true;
                    break;
                }
                if (fresh_consume_cut && stage_cut == DEBUG_STAGE_CUT_FRESH3_WAIT_ONLY) {
                    stop_after_stage_cut = true;
                    break;
                }
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 32; ++ii) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(scores_reg[ii * 16 + 0].x), "=f"(scores_reg[ii * 16 + 0].y), "=f"(scores_reg[ii * 16 + 1].x), "=f"(scores_reg[ii * 16 + 1].y), "=f"(scores_reg[ii * 16 + 2].x), "=f"(scores_reg[ii * 16 + 2].y), "=f"(scores_reg[ii * 16 + 3].x), "=f"(scores_reg[ii * 16 + 3].y),
                              "=f"(scores_reg[ii * 16 + 4].x), "=f"(scores_reg[ii * 16 + 4].y), "=f"(scores_reg[ii * 16 + 5].x), "=f"(scores_reg[ii * 16 + 5].y), "=f"(scores_reg[ii * 16 + 6].x), "=f"(scores_reg[ii * 16 + 6].y), "=f"(scores_reg[ii * 16 + 7].x), "=f"(scores_reg[ii * 16 + 7].y),
                              "=f"(scores_reg[ii * 16 + 8].x), "=f"(scores_reg[ii * 16 + 8].y), "=f"(scores_reg[ii * 16 + 9].x), "=f"(scores_reg[ii * 16 + 9].y), "=f"(scores_reg[ii * 16 + 10].x), "=f"(scores_reg[ii * 16 + 10].y), "=f"(scores_reg[ii * 16 + 11].x), "=f"(scores_reg[ii * 16 + 11].y),
                              "=f"(scores_reg[ii * 16 + 12].x), "=f"(scores_reg[ii * 16 + 12].y), "=f"(scores_reg[ii * 16 + 13].x), "=f"(scores_reg[ii * 16 + 13].y), "=f"(scores_reg[ii * 16 + 14].x), "=f"(scores_reg[ii * 16 + 14].y), "=f"(scores_reg[ii * 16 + 15].x), "=f"(scores_reg[ii * 16 + 15].y)
                            : "r"(score_tt_base + ii * 32));
                }
                warpgroup::arrive(p_copy_done[buf]);
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_COPY) {
                    stop_after_stage_cut = true;
                    break;
                }
                if (fresh_consume_cut && stage_cut == DEBUG_STAGE_CUT_FRESH3_COPY) {
                    stop_after_stage_cut = true;
                    break;
                }
                if constexpr (!fixed_fresh3_full) {
                    store_scores_bf16_tmem_x16_tile_if_needed<C>(
                        g.debug_scores_bf16_tmem_x16, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, m_tile, physical_k_iter, score_tt_base
                    );
                }
                if constexpr (fixed_full_production_path) {
                    if (physical_k_iter > m_tile) {
                        if constexpr (LIVE_DP) {
                            if constexpr (LIVE_DP_MODE0_ONLY) {
                                wait(live_dp_arrived[0], live_dp_phase);
                                arrive_live_dp_copy_done();
                                live_dp_phase ^= 1;
                            } else if (live_dp_mode_consumes_tmem(g)) {
                                wait(live_dp_arrived[0], live_dp_phase);
                                arrive_live_dp_copy_done();
                                live_dp_phase ^= 1;
                            }
                        }
                        wait_bwd_dq_src_if_needed(bwd_dq_buf, idx);
                        fp4pv_zero_invalid_causal_groups_mxfp4(
                            p_fp4_stage[bwd_dq_buf], p_sc_stage[bwd_dq_buf], global_row, 0);
                        tk_localcta::subgroup_barrier_sync<128>();
                        publish_shared_backing();
                        tk_localcta::subgroup_barrier_sync<128>();
                        publish_bwd_dq_stage_ready(bwd_dq_buf);
                        scores_phase ^= 1;
                        continue;
                    }
                }
	                if (physical_k_iter >= m_tile) {
	                    int causal_col = (physical_k_iter > m_tile) ? -1 : (int)warpgroup::laneid();
	                    #pragma unroll
	                    for (int q = 0; q < C::Nb / 2; ++q) {
	                        if (q * 2 > causal_col) {
	                            scores_reg[q].x = base_types::constants<float>::neg_infty();
	                        }
	                        if (q * 2 + 1 > causal_col) {
	                            scores_reg[q].y = base_types::constants<float>::neg_infty();
	                        }
	                    }
	                }
                if constexpr (!fixed_fresh3_full) {
                    store_p_bf16_live_tile_if_needed<C>(
                        g.debug_scores_bf16_live, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg
                    );
                    if (g.debug_scores_bf16_scan_repacked != nullptr) {
                        #pragma unroll
                        for (int qid = 0; qid < 4; ++qid) {
                            __align__(8) bf16_2 scores_bf_reg[16];
                            #pragma unroll
                            for (int jj = 0; jj < 16; ++jj) {
                                scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[qid * 16 + jj]);
                            }
                            store_scores_quarter_to_localcta_scan(p_bf16_scan[0], scores_bf_reg, qid);
                        }
                        store_scores_bf16_scan_repacked_tile_if_needed<C>(
                            g.debug_scores_bf16_scan_repacked, g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter
                        );
                    }
                }
                if constexpr (SAVED_LSE_BWD_ONLY) {
                    if constexpr (!fixed_full_production_path) {
                        if (physical_k_iter > m_tile) {
                            if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
                                if (g.backward_dq_pad != nullptr || g.backward_dq_bshd != nullptr) {
                                    wait_bwd_dq_src_if_needed(bwd_dq_buf, idx);
                                    fp4pv_zero_invalid_causal_groups_mxfp4(
                                        p_fp4_stage[bwd_dq_buf], p_sc_stage[bwd_dq_buf], global_row, 0);
                                    tk_localcta::subgroup_barrier_sync<128>();
                                    publish_shared_backing();
                                    tk_localcta::subgroup_barrier_sync<128>();
                                    publish_bwd_dq_stage_ready(bwd_dq_buf);
                                }
                            }
                            scores_phase ^= 1;
                            continue;
                        }
                    }
                    constexpr float LOG2E = 1.4426950408889634f;
                    const float saved_lse_log2 = g.lse[{t_coord.x, t_coord.z, 0, m_tile * C::Mb + global_row}] * LOG2E;
                    const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
	                    const float2 neg_lse_2 = {-saved_lse_log2, -saved_lse_log2};
	                    #pragma unroll
	                    for (int qid = 0; qid < 4; ++qid) {
	                        #pragma unroll
	                        for (int jj = 0; jj < 16; ++jj) {
	                            const int si = qid * 16 + jj;
	                            scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_lse_2);
	                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
	                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
	                        }
	                    }
                    const int heads = g.q.depth();
                    const int seqlen = g.q.rows();
                    const int row = m_tile * C::Mb + global_row;
                    const float delta_val = g.backward_delta[((t_coord.x * heads + t_coord.z) * seqlen) + row];
                    const float dp_scale = g.backward_softmax_scale;
                    const float neg_delta_scaled = -delta_val * dp_scale;
                    const float2 dp_scale_2 = {dp_scale, dp_scale};
                    const float2 neg_delta_scaled_2 = {neg_delta_scaled, neg_delta_scaled};
                    if constexpr (LIVE_DP) {
                        if constexpr (LIVE_DP_MODE0_ONLY) {
                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                            const float live_dp_scale = dp_scale * MXFP4_ALPHA;
                            const float2 live_dp_scale_2 = {live_dp_scale, live_dp_scale};
                            wait(live_dp_arrived[0], live_dp_phase);
                            #pragma unroll
                            for (int qid = 0; qid < 4; ++qid) {
                                if constexpr (LIVE_DP_MODE0_X8) {
                                    fp4pv_apply_live_dp_quarter_tmem_x8(
                                        scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                } else {
                                    fp4pv_apply_live_dp_quarter_tmem_x16(
                                        scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                }
                            }
                            tensor_load_wait();
                            arrive_live_dp_copy_done();
                            live_dp_phase ^= 1;
                        } else {
                            if (g.live_dp_debug_mode == LIVE_DP_DEBUG_FULL) {
                                wait(live_dp_arrived[0], live_dp_phase);
	                            #pragma unroll
	                            for (int qid = 0; qid < 4; ++qid) {
	                                float2 dp_chunk[16];
	                                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                                        : "=f"(dp_chunk[0].x), "=f"(dp_chunk[0].y), "=f"(dp_chunk[1].x), "=f"(dp_chunk[1].y), "=f"(dp_chunk[2].x), "=f"(dp_chunk[2].y), "=f"(dp_chunk[3].x), "=f"(dp_chunk[3].y),
                                          "=f"(dp_chunk[4].x), "=f"(dp_chunk[4].y), "=f"(dp_chunk[5].x), "=f"(dp_chunk[5].y), "=f"(dp_chunk[6].x), "=f"(dp_chunk[6].y), "=f"(dp_chunk[7].x), "=f"(dp_chunk[7].y),
                                          "=f"(dp_chunk[8].x), "=f"(dp_chunk[8].y), "=f"(dp_chunk[9].x), "=f"(dp_chunk[9].y), "=f"(dp_chunk[10].x), "=f"(dp_chunk[10].y), "=f"(dp_chunk[11].x), "=f"(dp_chunk[11].y),
	                                          "=f"(dp_chunk[12].x), "=f"(dp_chunk[12].y), "=f"(dp_chunk[13].x), "=f"(dp_chunk[13].y), "=f"(dp_chunk[14].x), "=f"(dp_chunk[14].y), "=f"(dp_chunk[15].x), "=f"(dp_chunk[15].y)
	                                        : "r"(score_tt_base + qid * 32));
	                                constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
	                                const float2 alpha_2 = {MXFP4_ALPHA, MXFP4_ALPHA};
	                                    #pragma unroll
	                                    for (int jj = 0; jj < 16; ++jj) {
	                                        const int q = qid * 16 + jj;
	                                        const float2 dp_val = __fmul2_rn(dp_chunk[jj], alpha_2);
	                                        scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
	                                        scores_reg[q], dp_val, dp_scale_2, neg_delta_scaled_2);
	                                }
	                            }
                                tensor_load_wait();
                                arrive_live_dp_copy_done();
                                live_dp_phase ^= 1;
                            } else {
                                if (!live_dp_mode_consumes_tmem(g)) {
                                    const float2 zero_dp = {0.0f, 0.0f};
                                    #pragma unroll
                                    for (int q = 0; q < C::Nb / 2; ++q) {
                                        scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                                            scores_reg[q], zero_dp, dp_scale_2, neg_delta_scaled_2);
                                    }
                                } else {
                                    wait(live_dp_arrived[0], live_dp_phase);
                                    if (live_dp_mode_applies_values(g)) {
                                        if (g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_WAIT &&
                                            g.backward_dp_bf16 == nullptr) {
                                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                            const float live_dp_scale = dp_scale * MXFP4_ALPHA;
                                            const float2 live_dp_scale_2 = {live_dp_scale, live_dp_scale};
                                            #pragma unroll
                                            for (int qid = 0; qid < 4; ++qid) {
                                                fp4pv_apply_live_dp_quarter_tmem_x8(
                                                    scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                            }
                                        } else if (g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_ALIGNED_PROBE &&
                                                   g.backward_dp_bf16 == nullptr) {
                                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                            const float live_dp_scale = dp_scale * MXFP4_ALPHA;
                                            const float2 live_dp_scale_2 = {live_dp_scale, live_dp_scale};
                                            #pragma unroll
                                            for (int qid = 0; qid < 4; ++qid) {
                                                fp4pv_apply_live_dp_quarter_tmem_x8_aligned_probe(
                                                    scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                            }
                                        } else if (g.live_dp_debug_mode == LIVE_DP_DEBUG_X16_WAIT &&
                                                   g.backward_dp_bf16 == nullptr) {
                                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                            const float live_dp_scale = dp_scale * MXFP4_ALPHA;
                                            const float2 live_dp_scale_2 = {live_dp_scale, live_dp_scale};
                                            #pragma unroll
                                            for (int qid = 0; qid < 4; ++qid) {
                                                fp4pv_apply_live_dp_quarter_tmem_x16(
                                                    scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                            }
                                        } else {
	                                        #pragma unroll
	                                        for (int qid = 0; qid < 4; ++qid) {
	                                            float2 dp_chunk[16];
	                                            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                                                : "=f"(dp_chunk[0].x), "=f"(dp_chunk[0].y), "=f"(dp_chunk[1].x), "=f"(dp_chunk[1].y), "=f"(dp_chunk[2].x), "=f"(dp_chunk[2].y), "=f"(dp_chunk[3].x), "=f"(dp_chunk[3].y),
                                                  "=f"(dp_chunk[4].x), "=f"(dp_chunk[4].y), "=f"(dp_chunk[5].x), "=f"(dp_chunk[5].y), "=f"(dp_chunk[6].x), "=f"(dp_chunk[6].y), "=f"(dp_chunk[7].x), "=f"(dp_chunk[7].y),
                                                  "=f"(dp_chunk[8].x), "=f"(dp_chunk[8].y), "=f"(dp_chunk[9].x), "=f"(dp_chunk[9].y), "=f"(dp_chunk[10].x), "=f"(dp_chunk[10].y), "=f"(dp_chunk[11].x), "=f"(dp_chunk[11].y),
	                                                  "=f"(dp_chunk[12].x), "=f"(dp_chunk[12].y), "=f"(dp_chunk[13].x), "=f"(dp_chunk[13].y), "=f"(dp_chunk[14].x), "=f"(dp_chunk[14].y), "=f"(dp_chunk[15].x), "=f"(dp_chunk[15].y)
	                                                : "r"(score_tt_base + qid * 32));
		                                        if (live_dp_mode_uses_scaled_mma_values(g)) {
		                                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
		                                            const float2 alpha_2 = {MXFP4_ALPHA, MXFP4_ALPHA};
	                                            #pragma unroll
	                                            for (int jj = 0; jj < 16; ++jj) {
	                                                dp_chunk[jj] = __fmul2_rn(dp_chunk[jj], alpha_2);
	                                            }
	                                        }
	                                        if (g.backward_dp_bf16 != nullptr) {
	                                            const int col_base = physical_k_iter * C::Nb;
	                                            const int64_t dp_row_base =
                                                (((static_cast<int64_t>(t_coord.x) * heads + t_coord.z) * seqlen) + row) * seqlen;
                                            #pragma unroll
                                            for (int jj = 0; jj < 16; ++jj) {
                                                const int col = col_base + (qid * 16 + jj) * 2;
                                                const bf16_2 dp_pair_bf = __float22bfloat162_rn(dp_chunk[jj]);
                                                *reinterpret_cast<bf16_2 *>(&g.backward_dp_bf16[dp_row_base + col]) = dp_pair_bf;
                                            }
                                        }
                                        #pragma unroll
                                        for (int jj = 0; jj < 16; ++jj) {
                                            const int q = qid * 16 + jj;
	                                            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
	                                                scores_reg[q], dp_chunk[jj], dp_scale_2, neg_delta_scaled_2);
	                                        }
	                                    }
                                        }
                                        tensor_load_wait();
                                    } else {
                                        const float2 zero_dp = {0.0f, 0.0f};
                                        #pragma unroll
                                        for (int q = 0; q < C::Nb / 2; ++q) {
                                            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                                                scores_reg[q], zero_dp, dp_scale_2, neg_delta_scaled_2);
                                        }
                                    }
	                                    arrive_live_dp_copy_done();
	                                    live_dp_phase ^= 1;
                                }
                            }
                        }
                    } else {
                        const int col_base = physical_k_iter * C::Nb;
                        const int64_t dp_row_base = (((static_cast<int64_t>(t_coord.x) * heads + t_coord.z) * seqlen) + row) * seqlen;
                        #pragma unroll
                        for (int qid = 0; qid < 4; ++qid) {
                            #pragma unroll
                            for (int jj = 0; jj < 16; ++jj) {
                                const int q = qid * 16 + jj;
                                const int col = col_base + q * 2;
                                const bf16_2 dp_pair_bf =
                                    *reinterpret_cast<const bf16_2 *>(&g.backward_dp_bf16[dp_row_base + col]);
                                scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                                    scores_reg[q], dp_pair_bf, dp_scale_2, neg_delta_scaled_2);
                            }
                        }
                    }
                    if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
                        // Keep this after live_dp_copy_done. Holding live-dP TMEM reuse while
                        // waiting for the dQ source stage can form a producer/consumer cycle.
                        wait_bwd_dq_src_if_needed(bwd_dq_buf, idx);
                        uint32_t packed_scale_marker = 0u;
	                        if constexpr (fixed_full_production_path) {
                                if constexpr (!SAVED_BWD_EMIT_ROW_MXFP4 && !SAVED_BWD_EMIT_COL_MXFP4) {
                                    if (g.backward_fused_stage_transpose) {
                                        packed_scale_marker = fp4pv_pack_scores_to_transposed_stage_mxfp4_marker_bf16_amax<C>(
                                            p_fp4_stage[bwd_dq_buf],
                                            p_sc_stage[bwd_dq_buf],
                                            g,
                                            p_bf16_scan[0],
                                            m_tile,
                                            physical_k_iter,
                                            scores_reg);
                                    } else {
                                        fp4pv_pack_scores_to_stage_mxfp4_bf16_amax(
                                            p_fp4_stage[bwd_dq_buf],
                                            p_sc_stage[bwd_dq_buf],
                                            scores_reg,
                                            global_row,
                                            g.backward_ds_quant_mode);
                                    }
	                            } else if constexpr (SAVED_BWD_EMIT_ROW_MXFP4 && !SAVED_BWD_EMIT_COL_MXFP4) {
	                                fp4pv_pack_scores_to_stage_mxfp4_bf16_amax(
	                                    p_fp4_stage[bwd_dq_buf],
	                                    p_sc_stage[bwd_dq_buf],
                                    scores_reg,
                                    global_row,
                                    g.backward_ds_quant_mode);
                            } else {
                                fp4pv_pack_scores_to_stage_mxfp4_bf16_amax_and_scan(
                                    p_fp4_stage[bwd_dq_buf],
                                    p_sc_stage[bwd_dq_buf],
                                    p_bf16_scan[0],
                                    scores_reg,
                                    global_row,
                                    g.backward_ds_quant_mode);
                            }
                        } else {
                            if (g.backward_fused_stage_transpose) {
                                packed_scale_marker = fp4pv_pack_scores_to_transposed_stage_mxfp4_marker_bf16_amax<C>(
                                    p_fp4_stage[bwd_dq_buf],
                                    p_sc_stage[bwd_dq_buf],
                                    g,
                                    p_bf16_scan[0],
                                    m_tile,
                                    physical_k_iter,
                                    scores_reg);
                            } else {
                                if constexpr (SAVED_BWD_EMIT_COL_MXFP4 && !SAVED_BWD_EMIT_ROW_MXFP4) {
                                    if constexpr (fixed_fresh3_full) {
                                        fp4pv_pack_scores_to_stage_mxfp4_bf16_amax_and_scan(
                                            p_fp4_stage[bwd_dq_buf],
                                            p_sc_stage[bwd_dq_buf],
                                            p_bf16_scan[0],
                                            scores_reg,
                                            global_row,
                                            g.backward_ds_quant_mode);
                                    } else {
                                        packed_scale_marker = fp4pv_pack_scores_to_stage_mxfp4_marker_bf16_amax_and_scan(
                                            p_fp4_stage[bwd_dq_buf],
                                            p_sc_stage[bwd_dq_buf],
                                            p_bf16_scan[0],
                                            scores_reg,
                                            global_row,
                                            g.backward_ds_quant_mode);
                                    }
                                } else {
                                    packed_scale_marker = fp4pv_pack_scores_to_stage_mxfp4_marker_bf16_amax(
                                        p_fp4_stage[bwd_dq_buf],
                                        p_sc_stage[bwd_dq_buf],
                                        scores_reg,
                                        global_row,
                                        g.backward_ds_quant_mode);
                                }
                            }
                        }
                        tk_localcta::subgroup_barrier_sync<128>();
                        publish_shared_backing();
                        tk_localcta::subgroup_barrier_sync<128>();
                        publish_bwd_dq_stage_ready(bwd_dq_buf);
                        if constexpr (!fixed_fresh3_full) {
                            if (g.backward_fused_stage_marker != nullptr) {
                                const int64_t marker_idx =
                                    ((((static_cast<int64_t>(t_coord.x) * heads + t_coord.z) * seqlen) + row) * (seqlen / C::Nb)) +
                                    physical_k_iter;
                                g.backward_fused_stage_marker[marker_idx] = packed_scale_marker;
                            }
                        }
	                            if constexpr (SAVED_BWD_EMIT_ROW_MXFP4) {
	                            if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
	                                store_backward_ds_row_col_mxfp4_direct_if_needed<C>(
	                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
	                            } else {
			                                if constexpr (fixed_full_production_path) {
			                                    if (!g.backward_skip_global_ds_row_store) {
			                                        store_backward_ds_fp4_row_tile_to_global_warpgroup<C>(
			                                            p_fp4_stage[bwd_dq_buf], g, t_coord.x, t_coord.z, m_tile, physical_k_iter, warpgroup::laneid());
			                                        store_backward_ds_sc_row_word_to_global_warpgroup<C>(
			                                            p_sc_stage[bwd_dq_buf], g, t_coord.x, t_coord.z, m_tile, physical_k_iter, m_tile * C::Mb + global_row);
			                                        warpgroup::sync(warpgroup::groupid() + 1);
			                                    }
			                                } else {
	                                    store_backward_ds_row_mxfp4_direct_if_needed<C>(
	                                        g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
	                                }
	                            }
	                        } else if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                            if constexpr (fixed_full_production_path) {
                                store_backward_ds_col_mxfp4_from_populated_scan_full<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter);
                            } else if (g.backward_fused_stage_transpose) {
                                store_backward_ds_col_mxfp4_direct_if_needed<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
                            } else if constexpr (!SAVED_BWD_EMIT_ROW_MXFP4) {
                                store_backward_ds_col_mxfp4_from_populated_scan_full<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter);
                            } else {
                                store_backward_ds_col_mxfp4_from_populated_scan_if_needed<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter);
                            }
                        }
                        scores_phase ^= 1;
                        continue;
                    }
                    if constexpr (SAVED_BWD_EMIT_ROW_MXFP4) {
                        if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                            store_backward_ds_row_col_mxfp4_direct_if_needed<C>(
                                g, p_bf16_scan[0], t_coord.x, t_coord.z, row, physical_k_iter, scores_reg);
                        } else {
                            store_backward_ds_row_mxfp4_direct_if_needed<C>(
                                g, t_coord.x, t_coord.z, row, physical_k_iter, scores_reg);
                        }
                    } else if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                        store_backward_ds_col_mxfp4_direct_if_needed<C>(
                            g, p_bf16_scan[0], t_coord.x, t_coord.z, row, physical_k_iter, scores_reg);
                    }
                    if constexpr (SAVED_BWD_STORE_BF16) {
                        store_p_bf16_live_tile_if_needed<C>(
                            g.p_bf16_live, g, t_coord.x, t_coord.z, row, physical_k_iter, scores_reg
                        );
                    }
                    scores_phase ^= 1;
                    continue;
                }
                if constexpr (QK_ONLY) {
                    if (g.use_external_lse) {
                        if (physical_k_iter > m_tile) {
                            // Full-layout dS outputs are pre-zeroed before launching this
                            // saved-LSE backward path, so future causal tiles require no
                            // explicit BF16 or MXFP4 zero stores.
                            scores_phase ^= 1;
                            continue;
                        }
                        constexpr float LOG2E = 1.4426950408889634f;
                        const float saved_lse_log2 = g.lse[{t_coord.x, t_coord.z, 0, m_tile * C::Mb + global_row}] * LOG2E;
                        const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                        const float2 neg_lse_2 = {-saved_lse_log2, -saved_lse_log2};
                        #pragma unroll
                        for (int qid = 0; qid < 4; ++qid) {
                            #pragma unroll
                            for (int jj = 0; jj < 16; ++jj) {
                                const int si = qid * 16 + jj;
                                scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_lse_2);
                                scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                                scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            }
                        }
                        if (g.emit_backward_ds) {
                            const int heads = g.q.depth();
                            const int seqlen = g.q.rows();
                            const int row = m_tile * C::Mb + global_row;
                            const float delta_val = g.backward_delta[((t_coord.x * heads + t_coord.z) * seqlen) + row];
                            const float dp_scale = g.backward_softmax_scale;
                            const float neg_delta_scaled = -delta_val * dp_scale;
                            const float2 dp_scale_2 = {dp_scale, dp_scale};
                            const float2 neg_delta_scaled_2 = {neg_delta_scaled, neg_delta_scaled};
                            const int col_base = physical_k_iter * C::Nb;
                            const int64_t dp_row_base = (((static_cast<int64_t>(t_coord.x) * heads + t_coord.z) * seqlen) + row) * seqlen;
                            #pragma unroll
                            for (int qid = 0; qid < 4; ++qid) {
                                #pragma unroll
                                for (int jj = 0; jj < 16; ++jj) {
                                    const int q = qid * 16 + jj;
                                    const int col = col_base + q * 2;
                                    const bf16_2 dp_pair_bf =
                                        *reinterpret_cast<const bf16_2 *>(&g.backward_dp_bf16[dp_row_base + col]);
                                    scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                                        scores_reg[q], dp_pair_bf, dp_scale_2, neg_delta_scaled_2);
                                }
                            }
                        }
                        if (g.emit_backward_ds_row_mxfp4) {
                            if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                                store_backward_ds_row_col_mxfp4_direct_if_needed<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
                            } else {
                                store_backward_ds_row_mxfp4_direct_if_needed<C>(
                                    g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
                            }
                        } else if (g.emit_backward_ds_col_mxfp4) {
                            if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                                store_backward_ds_col_mxfp4_direct_if_needed<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
                            }
                        }
                        store_p_bf16_live_tile_if_needed<C>(
                            g.p_bf16_live, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg
                        );
                        scores_phase ^= 1;
                        continue;
                    }
                }

                float row_max_old = row_max;
                float tile_score_max = base_types::constants<float>::neg_infty();
                float lm0 = base_types::constants<float>::neg_infty(), lm1 = base_types::constants<float>::neg_infty();
                float lm2 = base_types::constants<float>::neg_infty(), lm3 = base_types::constants<float>::neg_infty();
                float lm4 = base_types::constants<float>::neg_infty(), lm5 = base_types::constants<float>::neg_infty();
                float lm6 = base_types::constants<float>::neg_infty(), lm7 = base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int j = 0; j < C::Nb / 2; j += 8) {
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(scores_reg[j + 0].x), "f"(scores_reg[j + 0].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm1) : "f"(lm1), "f"(scores_reg[j + 1].x), "f"(scores_reg[j + 1].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm2) : "f"(lm2), "f"(scores_reg[j + 2].x), "f"(scores_reg[j + 2].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm3) : "f"(lm3), "f"(scores_reg[j + 3].x), "f"(scores_reg[j + 3].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(scores_reg[j + 4].x), "f"(scores_reg[j + 4].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm5) : "f"(lm5), "f"(scores_reg[j + 5].x), "f"(scores_reg[j + 5].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm6) : "f"(lm6), "f"(scores_reg[j + 6].x), "f"(scores_reg[j + 6].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm7) : "f"(lm7), "f"(scores_reg[j + 7].x), "f"(scores_reg[j + 7].y));
                }
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm1), "f"(lm2));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(lm5), "f"(lm6));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm3), "f"(lm4));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(tile_score_max) : "f"(tile_score_max), "f"(lm0), "f"(lm7));
                row_max = fmaxf(row_max, tile_score_max);
                float acc_scale = 1.0f;
                if (physical_k_iter >= m_tile || idx > 0) {
                    if (!ZERO_SCALE) {
                        constexpr float RESCALE_THRESHOLD = 8.f;
                        float acc_scale_log2 = (row_max_old - row_max) * SCALE_LOG2;
                        if (acc_scale_log2 >= -RESCALE_THRESHOLD) {
                            row_max = row_max_old;
                            acc_scale = 1.0f;
                        } else {
                            acc_scale = fp4pv_exp2_approx(acc_scale_log2);
                        }
                    }
                }
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_ROWMAX) {
                    stop_after_stage_cut = true;
                    break;
                }
                if constexpr (!QK_ONLY) {
                    warp::sync();
                    warp::arrive(corr_arrived[0]);
                }

                float2 tile_sum0 = {0.0f, 0.0f}, tile_sum1 = {0.0f, 0.0f};
                if (!ZERO_SCALE) {
                    float neg_max_scaled = row_max * (-SCALE_LOG2);
                    float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                    const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                    #pragma unroll
                    for (int qid = 0; qid < C::Nb / 32; ++qid) {
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj) {
                            int si = qid * 16 + jj;
                            scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_max_scaled_2);
                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            if ((jj & 1) == 0) {
                                tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                            } else {
                                tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int qid = 0; qid < C::Nb / 32; ++qid) {
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj) {
                            int si = qid * 16 + jj;
                            scores_reg[si].x = isfinite(scores_reg[si].x) ? 0.0f : scores_reg[si].x;
                            scores_reg[si].y = isfinite(scores_reg[si].y) ? 0.0f : scores_reg[si].y;
                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            if ((jj & 1) == 0) {
                                tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                            } else {
                                tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                            }
                        }
                    }
                }

                if constexpr (!QK_ONLY) {
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                }
                tile_sum0 = __fadd2_rn(tile_sum0, tile_sum1);
                const float tile_exp_sum = tile_sum0.x + tile_sum0.y;
                row_sum = row_sum * acc_scale + tile_exp_sum;
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_EXP) {
                    stop_after_stage_cut = true;
                    break;
                }
                if constexpr (NORMALIZE_TILES) {
                    const float inv_row_sum =
                        (row_sum > 0.0f && isfinite(row_sum)) ? __frcp_rn(row_sum) : 0.0f;
                    tile_max = 0.0f;
                    #pragma unroll
                    for (int qid = 0; qid < C::Nb / 32; ++qid) {
                        __align__(8) bf16_2 scores_bf_reg[16];
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj) {
                            const int si = qid * 16 + jj;
                            scores_reg[si].x *= inv_row_sum;
                            scores_reg[si].y *= inv_row_sum;
                            tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[si]);
                        }
                        store_scores_quarter_to_localcta_scan(p_bf16_scan[0], scores_bf_reg, qid);
                    }
                }
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_SCAN) {
                    stop_after_stage_cut = true;
                    break;
                }
                if constexpr (!QK_ONLY) {
                    warp::sync();
                    #pragma unroll
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, mask));
                    }
                    if (warp::laneid() == 0) {
                        p_softmax_warp_amax[buf][softmax_warp] = tile_max;
                    }
                    tk_localcta::subgroup_barrier_sync<128>();
                    if (warpgroup::laneid() == 0) {
                        float cta_amax = p_softmax_warp_amax[buf][0];
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][1]);
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][2]);
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][3]);
                        p_cta_amax[buf] = cta_amax;
                    }
                    tk_localcta::subgroup_barrier_sync<128>();
                    fp4pv_zero_raw_scale_stage(p_sc_stage[buf], warpgroup::laneid());
                    if constexpr (NORMALIZE_TILES) {
                        fp4pv_pack_scores_to_stage_payload_from_localcta_scan_row(
                            p_fp4_stage[buf], p_bf16_scan[0], p_cta_amax[buf], global_row
                        );
                        fp4pv_store_scales_from_localcta_scan_row(
                            p_sc_stage[buf], p_bf16_scan[0], p_cta_amax[buf], global_row
                        );
                    } else {
                        fp4pv_pack_scores_to_stage_and_scales(
                            p_fp4_stage[buf], p_sc_stage[buf], scores_reg, p_cta_amax[buf], global_row
                        );
                    }
                    int valid_groups = tk_localcta::SCALES_PER_CHUNK_X;
                    if (physical_k_iter >= m_tile) {
                        valid_groups = (physical_k_iter > m_tile) ? 0 : ((warpgroup::laneid() / 16) + 1);
                    }
                    fp4pv_zero_invalid_causal_groups(
                        p_fp4_stage[buf], p_sc_stage[buf], global_row, valid_groups
                    );
                    store_p_sc_rowmajor_live_tile_if_needed<C>(
                        p_sc_stage[buf], g.p_sc_rowmajor_live, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter
                    );
                    if (blockIdx.x == 0 && cta_rank == 1 && idx == 1 && warpgroup::laneid() == 32) {
                        auto &raw_scales_dbg = *reinterpret_cast<fp4pv_p_raw_scales *>(&p_sc_stage[buf].data[0]);
                        printf("raw_sc row=%d amax=%f :", global_row, p_cta_amax[buf]);
                        #pragma unroll
                        for (int gidx = 0; gidx < tk_localcta::SCALES_PER_CHUNK_X; ++gidx) {
                            printf(" %f", static_cast<float>(raw_scales_dbg[global_row][gidx]));
                        }
                        printf("\n");
                    }
                }
                store_p_bf16_live_tile_if_needed<C>(
                    g.p_bf16_live, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg
                );
                if constexpr (!QK_ONLY) {
                    tk_localcta::subgroup_barrier_sync<128>();
                    publish_shared_backing();
                    tk_localcta::subgroup_barrier_sync<128>();
                    warpgroup::arrive(p_pack_ready[buf]);
                }
                store_debug_row_stat_if_needed<C>(g.debug_row_max, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, row_max);
                store_debug_row_stat_if_needed<C>(g.debug_row_sum, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, row_sum);
                store_debug_row_stat_if_needed<C>(g.debug_tile_score_max, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, tile_score_max);
                store_debug_row_stat_if_needed<C>(g.debug_tile_exp_sum, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, tile_exp_sum);
                if (warpgroup::laneid() == 0) {
                    const float debug_tile_amax = QK_ONLY ? 0.0f : p_cta_amax[buf];
                    store_debug_tile_amax_if_needed<C>(g.debug_tile_amax, g, t_coord.x, t_coord.z, m_tile, physical_k_iter, debug_tile_amax);
	                }
	                scores_phase ^= 1;
	            }
	            if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
                if constexpr (fixed_full_production_path) {
                    const int tail_begin = consumer_iters_per_task > 2 ? consumer_iters_per_task - 2 : 0;
                    for (int tail_idx = tail_begin; tail_idx < consumer_iters_per_task; ++tail_idx) {
                        const int tail_buf = tail_idx & 1;
                        const int src_phase = (bwd_dq_src_phase_mask >> tail_buf) & 1;
                        wait(bwd_dq_src_finished[tail_buf], src_phase);
                        bwd_dq_src_phase_mask ^= (1 << tail_buf);
                        if constexpr (SAVED_BWD_DQ_DUAL_HALVES) {
                            const int src1_phase = (bwd_dq_src1_phase_mask >> tail_buf) & 1;
                            wait(bwd_dq_src1_finished[tail_buf], src1_phase);
                            bwd_dq_src1_phase_mask ^= (1 << tail_buf);
                        }
                    }
                } else if (bwd_dq_src_completion_active && !stop_after_stage_cut) {
	                    const int tail_begin = consumer_iters_per_task > 2 ? consumer_iters_per_task - 2 : 0;
	                    for (int tail_idx = tail_begin; tail_idx < consumer_iters_per_task; ++tail_idx) {
	                        const int tail_buf = tail_idx & 1;
	                        const int src_phase = (bwd_dq_src_phase_mask >> tail_buf) & 1;
	                        wait(bwd_dq_src_finished[tail_buf], src_phase);
	                        bwd_dq_src_phase_mask ^= (1 << tail_buf);
		                        if (bwd_dq_src1_completion_active) {
		                            const int src1_phase = (bwd_dq_src1_phase_mask >> tail_buf) & 1;
		                            wait(bwd_dq_src1_finished[tail_buf], src1_phase);
		                            bwd_dq_src1_phase_mask ^= (1 << tail_buf);
		                        }
	                    }
	                }
	            }
	            if (QK_ONLY && stage_cut == DEBUG_STAGE_CUT_ISSUE3_ONLY && producer_iters_per_task == 3) {
                stop_after_stage_cut = true;
            }
            if (stop_after_stage_cut) {
                break;
            }

            bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
            float lse_val;
            if (!row_invalid) {
                constexpr float LN2 = 0.693147180559945f;
                float log2_row_sum;
                asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                lse_val = (row_max * SCALE_LOG2 + log2_row_sum) * LN2;
            } else {
                lse_val = base_types::constants<float>::neg_infty();
            }
            store_lse_row_if_needed<C>(g, t_coord.x, t_coord.z, m_tile * C::Mb + warpgroup::laneid(), lse_val);
        }
    } else if (warpgroup::groupid() == 2) {
        if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<48>();
        }
        if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
            if constexpr (fixed_full_production_path) {
                d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT_BASE);
                int accum_phase = 0;
                for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    const int m_tile = t_coord.y + cta_rank;
                    if (m_tile >= tiles_m) {
                        continue;
                    }
                    const int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                    if (iters_per_task <= 0) {
                        continue;
                    }
                    wait(bwd_dq_accum_done[0], accum_phase);
                    accum_phase ^= 1;
                    warpgroup::tma::store_async_read_wait<0>();
                    warpgroup::sync(warpgroup::groupid() + 1);
                    store_bwd_dq_pad_tmem_tile<C, 0>(tt_output, g, t_coord.x, t_coord.z, m_tile);
                    if constexpr (BWD_DQ_TAIL64_CLUSTER) {
                        d_tt_tail_outputs tt_output1 = tm_alloc.template allocate<d_tt_tail_outputs>(BWD_DQ_OUTPUT1_BASE);
                        store_bwd_dq_pad_tmem_tile<C, 1>(tt_output1, g, t_coord.x, t_coord.z, m_tile);
                    } else {
                        d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                        store_bwd_dq_pad_tmem_tile<C, 1>(tt_output1, g, t_coord.x, t_coord.z, m_tile);
                    }
                }
            } else {
                const bool bwd_dq_wait_tmem_active =
                    (g.backward_dq_pad != nullptr || g.backward_dq_bshd != nullptr) &&
                    ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM));
                if (bwd_dq_wait_tmem_active) {
                    d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT_BASE);
                    constexpr bool bwd_dq_dual_halves = SAVED_BWD_DQ_DUAL_HALVES;
                    int accum_phase = 0;
                    int tmem_phase = 0;
                    for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                        int3 t_coord = get_tile_idx(cur_bid);
                        const int m_tile = t_coord.y + cta_rank;
                        if (m_tile >= tiles_m) {
                            continue;
                        }
                        const int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                        if (iters_per_task <= 0) {
                            continue;
                        }
                        if ((stage_cut != DEBUG_STAGE_CUT_FRESH3_FULL)) {
                            for (int idx = 0; idx < iters_per_task; ++idx) {
                                wait(bwd_dq_tmem_ready[0], tmem_phase);
                                tmem_phase ^= 1;
                            }
                        }
                        wait(bwd_dq_accum_done[0], accum_phase);
                        accum_phase ^= 1;
                        if ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL)) {
                            warpgroup::tma::store_async_read_wait<0>();
                            warpgroup::sync(warpgroup::groupid() + 1);
                            if constexpr (bwd_dq_dual_halves) {
                                d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                store_bwd_dq_pad_tmem_tile<C, 0>(tt_output, g, t_coord.x, t_coord.z, m_tile);
                                store_bwd_dq_pad_tmem_tile<C, 1>(tt_output1, g, t_coord.x, t_coord.z, m_tile);
                            } else {
                                store_bwd_dq_pad_tmem_tile<C>(tt_output, g, t_coord.x, t_coord.z, m_tile);
                            }
                        }
                    }
                }
            }
        } else if constexpr (!QK_ONLY) {
            int corr_phase = 0;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                const bool row_tile_valid = (t_coord.y + cta_rank) < tiles_m;
                if (!row_tile_valid) {
                    continue;
                }
                int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    wait(corr_arrived[0], corr_phase);
                    warpgroup::sync(warpgroup::groupid() + 1);
                    warpgroup::arrive(rescale_finished[0]);
                    corr_phase ^= 1;
                }
            }
        }
    } else if (warpgroup::groupid() == 0) {
        if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
            if ((g.backward_dq_pad != nullptr || g.backward_dq_bshd != nullptr) &&
                (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY)) {
                const bool issue_warp = (cta_rank == 0) && (warpgroup::warpid() == 0);
                const bool issue_lane = issue_warp && warp::elect_leader();
                d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT_BASE);
                d_tt_bwd_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_bwd_p_sc>(BWD_DQ_P_SC_BASE);
                d_tt_bwd_k_sc k_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K_SC_BASE);
                int stage_phase_mask = 0;
                int src_phase_mask = 0;
                int src1_phase_mask = 0;
                int stage_remote_phase_mask = 0;
	                int k_remote_phase_mask = 0;
	                int tmem_ready_count = 0;
		                constexpr bool bwd_dq_dual_halves = SAVED_BWD_DQ_DUAL_HALVES;
		                const bool bwd_dq_nosem_full =
		                    stage_cut == DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL;
		                const bool bwd_dq_src1_wait_active =
		                    bwd_dq_dual_halves &&
		                    !bwd_dq_nosem_full &&
		                    ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
		                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY) ||
		                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
		                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE));
	                constexpr bool bwd_dq_prepare_issue_active = false;
                for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    const int m_tile = t_coord.y + cta_rank;
                    if (m_tile >= tiles_m) {
                        continue;
                    }
                    const bool peer_cta_valid = (t_coord.y + 1) < tiles_m;
                    const int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                    if (iters_per_task <= 0) {
                        continue;
                    }
                    for (int idx = 0; idx < iters_per_task; ++idx) {
                        const int buf = idx & 1;
                        const int physical_k_iter = physical_k_iter_for(idx);
                        if (idx >= 2 && !bwd_dq_nosem_full) {
                            const int src_phase = (src_phase_mask >> buf) & 1;
                            wait(bwd_dq_src_finished[buf], src_phase);
                            src_phase_mask ^= (1 << buf);
	                            if (bwd_dq_src1_wait_active) {
	                                const int src1_phase = (src1_phase_mask >> buf) & 1;
	                                wait(bwd_dq_src1_finished[buf], src1_phase);
	                                src1_phase_mask ^= (1 << buf);
	                            }
                        }
                        const int stage_phase = (stage_phase_mask >> buf) & 1;
                        wait(bwd_dq_stage_ready[buf], stage_phase);
                        stage_phase_mask ^= (1 << buf);
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                if constexpr (fixed_fresh3_full) {
                                    if (peer_cta_valid) {
                                        const int stage_remote_phase = (stage_remote_phase_mask >> buf) & 1;
                                        tma::cluster::wait(bwd_dq_stage_remote_ready[buf], stage_remote_phase);
                                        stage_remote_phase_mask ^= (1 << buf);
                                    }
                                } else if (peer_cta_valid && (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
                                    const int stage_remote_phase = (stage_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(bwd_dq_stage_remote_ready[buf], stage_remote_phase);
                                    stage_remote_phase_mask ^= (1 << buf);
                                }
                            }
                        }
	                        if constexpr (!fixed_fresh3_full) {
	                        if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
	                            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
	                                warpgroup::arrive(bwd_dq_src_finished[buf]);
	                            }
	                            continue;
	                        }
	                        }
                        if (warpgroup::warpid() == 0) {
                            if constexpr (bwd_dq_dual_halves) {
                                load_bwd_dq_k_col_fp4_tile<C>(bwd_dq_k_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank, 0);
                                load_bwd_dq_k_col_sc_tile<C>(bwd_dq_k_sc_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, 0);
                                if constexpr (BWD_DQ_TAIL64_CLUSTER) {
                                    load_bwd_dq_k_col_fp4_tail64_tile<C>(bwd_dq_k1_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                                } else {
                                    load_bwd_dq_k_col_fp4_tile<C>(bwd_dq_k1_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank, 1);
                                }
                                load_bwd_dq_k_col_sc_tile<C>(bwd_dq_k1_sc_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, 1);
                            } else {
                                load_bwd_dq_k_col_fp4_tile<C>(bwd_dq_k_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                                load_bwd_dq_k_col_sc_tile<C>(bwd_dq_k_sc_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter);
                            }
                        }
                        warpgroup::sync(warpgroup::groupid() + 1);
                        publish_shared_backing();
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            publish_cluster_shared_backing_if_needed<C>();
                            if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                tma::cluster::arrive(bwd_dq_k_remote_ready[buf], 0);
                            }
                        }
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                if (peer_cta_valid) {
                                    const int k_remote_phase = (k_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(bwd_dq_k_remote_ready[buf], k_remote_phase);
                                    k_remote_phase_mask ^= (1 << buf);
                                }
                            }
                        }
                        if constexpr (!fixed_fresh3_full) {
                        if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY)) {
                            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                warpgroup::arrive(bwd_dq_src_finished[buf]);
                                if constexpr (bwd_dq_dual_halves) {
                                    warpgroup::arrive(bwd_dq_src1_finished[buf]);
                                }
                            }
                            continue;
                        }
                        }
                        if (bwd_dq_prepare_issue_active) {
                            if (issue_lane) {
                                warpgroup::arrive(bwd_dq_issue_ready[buf]);
                            }
                            continue;
                        }
                        if (issue_warp) {
                            auto &p_sc_e8m0 = *reinterpret_cast<typename G::bwd_dq_sc_tile *>(&p_sc_stage[buf]);
                            fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm, p_sc_e8m0);
                            fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm, bwd_dq_k_sc_smem[buf]);
                            if constexpr (bwd_dq_dual_halves) {
                                d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                fp4pv_load_mxnv_scale_async_cluster<C>(k1_sc_tm, bwd_dq_k1_sc_smem[buf]);
                            }
                            tensor_load_wait();
                            if constexpr (fixed_fresh3_full) {
                                tensor_before_thread_sync();
                                if (issue_lane) {
	                                    if (idx == 0) {
	                                        fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
	                                        if constexpr (bwd_dq_dual_halves) {
	                                            d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                                if constexpr (BWD_DQ_TAIL64_CLUSTER) {
                                                    d_tt_tail_outputs tt_output1 = tm_alloc.template allocate<d_tt_tail_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                    auto bwd_dq_k1_tail = bwd_dq_k1_smem[buf].template subtile<C::Dvo / 4, C::Nb / 2>(int2{0, 0});
                                                    fp4pv_mm_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_tail, p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                } else {
                                                    d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                    fp4pv_mm_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                }
	                                        }
	                                    } else {
	                                        fp4pv_mma_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
	                                        if constexpr (bwd_dq_dual_halves) {
	                                            d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                                if constexpr (BWD_DQ_TAIL64_CLUSTER) {
                                                    d_tt_tail_outputs tt_output1 = tm_alloc.template allocate<d_tt_tail_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                    auto bwd_dq_k1_tail = bwd_dq_k1_smem[buf].template subtile<C::Dvo / 4, C::Nb / 2>(int2{0, 0});
                                                    fp4pv_mma_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_tail, p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                } else {
                                                    d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                    fp4pv_mma_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                }
	                                        }
	                                    }
                                    if (idx == iters_per_task - 1) {
                                        detail::tcgen05::commit<C::CLUSTER_SIZE>(bwd_dq_accum_done[0]);
                                    }
                                }
                            } else {
                                if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY)) {
                                    tensor_before_thread_sync();
                                    if (issue_lane) {
                                        if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY)) {
                                            if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM)) {
                                                auto &dq_p_operand =
                                                    ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM))
                                                        ? bwd_dq_p_dummy[0]
                                                        : p_fp4_stage[buf];
                                                auto &dq_k_operand =
                                                    ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM))
                                                        ? bwd_dq_k_dummy[0]
                                                        : bwd_dq_k_smem[buf];
                                                if (idx == 0) {
                                                    fp4pv_mm_unscaled_nosem_cluster_ABt<C>(tt_output, dq_p_operand, dq_k_operand);
                                                } else {
                                                    fp4pv_mma_unscaled_nosem_cluster_ABt<C>(tt_output, dq_p_operand, dq_k_operand);
                                                }
                                            } else if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE)) {
                                                if (idx == 0) {
                                                    fp4pv_mm_unscaled_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], bwd_dq_src_finished[buf]);
                                                } else {
                                                    fp4pv_mma_unscaled_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], bwd_dq_src_finished[buf]);
                                                }
                                            } else {
                                                if (idx == 0) {
                                                    if (bwd_dq_nosem_full) {
                                                        fp4pv_mm_nosem_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm);
                                                    } else {
                                                        fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
                                                    }
                                                    if constexpr (bwd_dq_dual_halves) {
                                                        d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                        d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                                        if (bwd_dq_nosem_full) {
                                                            fp4pv_mm_nosem_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm);
                                                        } else {
                                                            fp4pv_mm_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                        }
                                                    }
                                                } else {
                                                    if (bwd_dq_nosem_full) {
                                                        fp4pv_mma_nosem_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm);
                                                    } else {
                                                        fp4pv_mma_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
                                                    }
                                                    if constexpr (bwd_dq_dual_halves) {
                                                        d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                        d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                                        if (bwd_dq_nosem_full) {
                                                            fp4pv_mma_nosem_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm);
                                                        } else {
                                                            fp4pv_mma_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                        }
                                                    }
                                                }
                                            }
                                            if ((stage_cut != DEBUG_STAGE_CUT_FRESH3_FULL) &&
                                                (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL)) {
                                                tensor_commit<C::CLUSTER_SIZE>(bwd_dq_tmem_ready[0]);
                                                ++tmem_ready_count;
                                            }
                                            if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) &&
                                                (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) &&
                                                idx == iters_per_task - 1) {
                                                detail::tcgen05::commit<C::CLUSTER_SIZE>(bwd_dq_accum_done[0]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if constexpr (!fixed_fresh3_full) {
                            if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY)) {
                                continue;
                            }
                            if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM)) {
                                continue;
                            }
                        }
                    }
                    (void)tmem_ready_count;
                }
            }
        } else if constexpr (!QK_ONLY) {
            int p_pack_phase = 0;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                int m_tile = t_coord.y + cta_rank;
                if (m_tile >= tiles_m) {
                    continue;
                }
                int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    constexpr int buf = 0;
                    const int consumer_tid = threadIdx.x;
                    wait(p_pack_ready[buf], p_pack_phase);
                    p_pack_phase ^= 1;
                    if constexpr (NORMALIZE_TILES) {
                        quantize_localcta_p_scales_from_scan_with_amax_and_store_raw<C>(
                            p_sc_stage[buf], p_bf16_scan[0], p_cta_amax[buf], consumer_tid,
                            g, t_coord.x, t_coord.z, m_tile, idx);
                        } else {
                            finalize_localcta_p_scales_with_amax<C>(
                                p_sc_stage[buf], p_cta_amax[buf], consumer_tid);
                        }
                    store_p_fp4_tile_to_global_warpgroup<C>(p_fp4_stage[buf], g, t_coord.x, t_coord.z, m_tile, idx, consumer_tid);
                    store_p_sc_prepared_to_global_warpgroup<C>(p_sc_stage[buf], g, t_coord.x, t_coord.z, m_tile, idx, consumer_tid);
                    tk_localcta::subgroup_barrier_sync<128>();
                    warpgroup::arrive(p_store_done[buf]);
                }
            }
        }
    }
}

__device__ inline float softmax_rescale_factor_strict(float row_max, float row_max_old, float scale_log2) {
    if (!isfinite(row_max_old) || !isfinite(row_max) || scale_log2 == 0.0f) {
        return 1.0f;
    }
    return fp4pv_exp2_approx((row_max_old - row_max) * scale_log2);
}

__device__ inline float softmax_rescale_factor_fast(float &row_max, float row_max_old, float scale_log2, float rescale_threshold = 8.f) {
    if (!isfinite(row_max_old) || !isfinite(row_max) || scale_log2 == 0.0f) {
        return 1.0f;
    }
    float acc_scale_log2 = (row_max_old - row_max) * scale_log2;
    if (!isfinite(acc_scale_log2) || acc_scale_log2 >= -rescale_threshold) {
        row_max = row_max_old;
        return 1.0f;
    }
    return fp4pv_exp2_approx(acc_scale_log2);
}

template <typename C, typename OTile>
__device__ inline void zero_output_tile_streaming(OTile &o_tile) {
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            #pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
                o_tile[{row, col + ii}] = __float2bfloat16(0.0f);
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OTile>
__device__ inline void rescale_output_tile_streaming(OTile &o_tile, float correction) {
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            #pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
                const float scaled = __bfloat162float(o_tile[{row, col + ii}]) * correction;
                o_tile[{row, col + ii}] = __float2bfloat16(scaled);
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OTile, typename OutputTT>
__device__ inline void accumulate_tt_output_into_tile_streaming(OTile &o_tile, OutputTT &tt_output) {
    const int row = warpgroup::laneid();
    const int subtile = row / 32;
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            float2 o_reg[8];
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((subtile * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                const float cur0 = __bfloat162float(o_tile[{row, col + ii * 2 + 0}]);
                const float cur1 = __bfloat162float(o_tile[{row, col + ii * 2 + 1}]);
                o_tile[{row, col + ii * 2 + 0}] = __float2bfloat16(cur0 + o_reg[ii].x);
                o_tile[{row, col + ii * 2 + 1}] = __float2bfloat16(cur1 + o_reg[ii].y);
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OTile>
__device__ inline void zero_fp32_output_tile_streaming(OTile &o_tile) {
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            #pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
                o_tile[{row, col + ii}] = 0.0f;
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OTile>
__device__ inline void rescale_fp32_output_tile_streaming(
    OTile &o_tile,
    float correction
) {
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            #pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
                o_tile[{row, col + ii}] *= correction;
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OTile, typename OutputTT>
__device__ inline void accumulate_tt_output_into_fp32_tile_streaming(
    OTile &o_tile,
    OutputTT &tt_output
) {
    const int row = warpgroup::laneid();
    const int subtile = row / 32;
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            float2 o_reg[8];
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y),
                  "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y),
                  "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y),
                  "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y),
                  "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((subtile * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                o_tile[{row, col + ii * 2 + 0}] += o_reg[ii].x;
                o_tile[{row, col + ii * 2 + 1}] += o_reg[ii].y;
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, int HALF, typename OTile>
__device__ __noinline__ void zero_fp32_output_half_tile_streaming(OTile &o_tile) {
    static_assert(C::Dvo == 128 && (HALF == 0 || HALF == 1));
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo / 2; col += 16) {
            #pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
                o_tile[{row, col + ii}] = 0.0f;
            }
        }
    }
    warpgroup::sync(HALF == 0 ? 3 : 4);
}

template <typename C, int HALF, typename OTile>
__device__ __noinline__ void rescale_fp32_output_half_tile_streaming(
    OTile &o_tile,
    float correction
) {
    static_assert(C::Dvo == 128 && (HALF == 0 || HALF == 1));
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo / 2; col += 16) {
            #pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
                o_tile[{row, col + ii}] *= correction;
            }
        }
    }
    warpgroup::sync(HALF == 0 ? 3 : 4);
}

template <typename C, int HALF, typename OTile, typename OutputTT>
__device__ __noinline__ void initialize_tt_output_half_to_fp32_tile_streaming(
    OTile &o_tile,
    OutputTT &tt_output
) {
    static_assert(C::Dvo == 128 && (HALF == 0 || HALF == 1));
    const int row = warpgroup::laneid();
    const int subtile = row / 32;
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo / 2; col += 16) {
            float2 o_reg[8];
            const uint32_t addr =
                tt_output.addr + ((subtile * 32) << 16) +
                HALF * (C::Dvo / 2) + col;
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y),
                  "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y),
                  "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y),
                  "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y),
                  "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(addr));
            tensor_load_wait();
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                o_tile[{row, col + ii * 2 + 0}] = o_reg[ii].x;
                o_tile[{row, col + ii * 2 + 1}] = o_reg[ii].y;
            }
        }
    }
    warpgroup::sync(HALF == 0 ? 3 : 4);
}

template <typename C, int HALF, typename OTile, typename OutputTT>
__device__ __noinline__ void fold_tt_output_half_into_fp32_tile_streaming(
    OTile &o_tile,
    OutputTT &tt_output,
    float correction
) {
    static_assert(C::Dvo == 128 && (HALF == 0 || HALF == 1));
    const int row = warpgroup::laneid();
    const int subtile = row / 32;
    const float2 correction2 = {correction, correction};
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo / 2; col += 16) {
            float2 o_reg[8];
            const uint32_t addr =
                tt_output.addr + ((subtile * 32) << 16) +
                HALF * (C::Dvo / 2) + col;
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y),
                  "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y),
                  "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y),
                  "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y),
                  "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(addr));
            tensor_load_wait();
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                float2 prior = {
                    o_tile[{row, col + ii * 2 + 0}],
                    o_tile[{row, col + ii * 2 + 1}],
                };
                const float2 folded = __fadd2_rn(
                    __fmul2_rn(prior, correction2), o_reg[ii]);
                o_tile[{row, col + ii * 2 + 0}] = folded.x;
                o_tile[{row, col + ii * 2 + 1}] = folded.y;
            }
        }
    }
    warpgroup::sync(HALF == 0 ? 3 : 4);
}

template <typename C, int HALF, typename OTile, typename Bf16Tile>
__device__ __noinline__ void store_fp32_output_half_to_bf16_tile_streaming(
    OTile &o_tile,
    Bf16Tile &o_bf16,
    float output_scale,
    bool row_invalid
) {
    static_assert(C::Dvo == 128 && (HALF == 0 || HALF == 1));
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        const float2 scale2 = {output_scale, output_scale};
        #pragma unroll
        for (int col = 0; col < C::Dvo / 2; col += 2) {
            float2 value = {
                o_tile[{row, col + 0}],
                o_tile[{row, col + 1}]};
            if (row_invalid) {
                value = {0.0f, 0.0f};
            }
            const bf16_2 packed =
                __float22bfloat162_rn(__fmul2_rn(value, scale2));
            o_bf16[{row, HALF * (C::Dvo / 2) + col + 0}] = packed.x;
            o_bf16[{row, HALF * (C::Dvo / 2) + col + 1}] = packed.y;
        }
    }
    warpgroup::sync(HALF == 0 ? 3 : 4);
}

template <typename C, typename OutputTT>
__device__ __forceinline__ void rescale_tt_output_inplace_direct_streaming(
    OutputTT &tt_output,
    float correction,
    bool lane_needs_rescale
) {
    if (warpgroup::laneid() >= C::Mb) {
        return;
    }
    const float2 corr_2 = {lane_needs_rescale ? correction : 1.0f,
                           lane_needs_rescale ? correction : 1.0f};
    #pragma unroll
    for (int col = 0; col < OutputTT::cols; col += 16) {
        float2 o_reg[8];
        const uint32_t addr = tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
              "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
              "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
              "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
            : "r"(addr));
        tensor_load_wait();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
        }
        asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
            :: "f"(o_reg[0].x), "f"(o_reg[0].y), "f"(o_reg[1].x), "f"(o_reg[1].y),
               "f"(o_reg[2].x), "f"(o_reg[2].y), "f"(o_reg[3].x), "f"(o_reg[3].y),
               "f"(o_reg[4].x), "f"(o_reg[4].y), "f"(o_reg[5].x), "f"(o_reg[5].y),
               "f"(o_reg[6].x), "f"(o_reg[6].y), "f"(o_reg[7].x), "f"(o_reg[7].y),
               "r"(addr));
    }
    tensor_store_wait();
}

template <typename C, typename OutputTT>
__device__ inline void merge_dual_tt_output_accum_streaming(
    OutputTT &tt_output_main,
    OutputTT &tt_output_spare,
    float correction,
    bool lane_needs_rescale
) {
    if (warpgroup::laneid() >= C::Mb) {
        warpgroup::sync(warpgroup::groupid() + 1);
        return;
    }
    const float2 corr_2 = {lane_needs_rescale ? correction : 1.0f,
                           lane_needs_rescale ? correction : 1.0f};
    #pragma unroll
    for (int col = 0; col < OutputTT::cols; col += 16) {
        float2 main_reg[8];
        float2 spare_reg[8];
        const uint32_t main_addr = tt_output_main.addr + ((warpgroup::warpid() * 32) << 16) + col;
        const uint32_t spare_addr = tt_output_spare.addr + ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(main_reg[0].x), "=f"(main_reg[0].y), "=f"(main_reg[1].x), "=f"(main_reg[1].y),
              "=f"(main_reg[2].x), "=f"(main_reg[2].y), "=f"(main_reg[3].x), "=f"(main_reg[3].y),
              "=f"(main_reg[4].x), "=f"(main_reg[4].y), "=f"(main_reg[5].x), "=f"(main_reg[5].y),
              "=f"(main_reg[6].x), "=f"(main_reg[6].y), "=f"(main_reg[7].x), "=f"(main_reg[7].y)
            : "r"(main_addr));
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(spare_reg[0].x), "=f"(spare_reg[0].y), "=f"(spare_reg[1].x), "=f"(spare_reg[1].y),
              "=f"(spare_reg[2].x), "=f"(spare_reg[2].y), "=f"(spare_reg[3].x), "=f"(spare_reg[3].y),
              "=f"(spare_reg[4].x), "=f"(spare_reg[4].y), "=f"(spare_reg[5].x), "=f"(spare_reg[5].y),
              "=f"(spare_reg[6].x), "=f"(spare_reg[6].y), "=f"(spare_reg[7].x), "=f"(spare_reg[7].y)
            : "r"(spare_addr));
        tensor_load_wait();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            main_reg[ii] = __fadd2_rn(__fmul2_rn(main_reg[ii], corr_2), spare_reg[ii]);
        }
        asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
            :: "f"(main_reg[0].x), "f"(main_reg[0].y), "f"(main_reg[1].x), "f"(main_reg[1].y),
               "f"(main_reg[2].x), "f"(main_reg[2].y), "f"(main_reg[3].x), "f"(main_reg[3].y),
               "f"(main_reg[4].x), "f"(main_reg[4].y), "f"(main_reg[5].x), "f"(main_reg[5].y),
               "f"(main_reg[6].x), "f"(main_reg[6].y), "f"(main_reg[7].x), "f"(main_reg[7].y),
               "r"(main_addr));
    }
    tensor_store_wait();
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename ColdRT, typename ColdChunk>
__device__ __forceinline__ void fp4pv_load_packed_fp16_cold(
    ColdRT &destination,
    const ColdChunk &source
) {
    static_assert(ColdRT::rows == 32 && ColdRT::cols == 32 &&
                  ColdRT::height == 2 && ColdRT::width == 2,
                  "packed FP16 cold load expects a 32x32 register tile");
    #pragma unroll
    for (int tile_row = 0; tile_row < 2; ++tile_row) {
        asm volatile(
            "{tcgen05.ld.sync.aligned.16x128b.x4.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];}"
            : "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][0].data[0])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][0].data[1])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][0].data[2])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][0].data[3])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][1].data[0])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][1].data[1])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][1].data[2])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][1].data[3]))
            : "r"(source.addr +
                    (((warpgroup::warpid() * 32 + tile_row * 16)) << 16)));
    }
}

template <typename OutputTT, typename ColdTT>
__device__ __forceinline__ void fp4pv_save_full_output_fp16_cold(
    OutputTT &source,
    ColdTT &cold
) {
    static_assert(OutputTT::cols == 128 && ColdTT::cols == 128,
                  "full-output cold checkpoint expects logical DVO128");
    constexpr float NORMALIZE =
        1.0f / (36.0f * FP4PV_MXFP4_ONLINE_P_SCALE);
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        rt_fl<32, 32> source_regs;
        rt_hf<32, 32> cold_regs;
        auto source_chunk =
            source.template subtile<full_tt_fl<32>>(chunk * 32);
        auto cold_chunk =
            cold.template subtile<full_tt_hf<32>>(chunk * 32);
        warpgroup::load_async(source_regs, source_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        warp::mul(source_regs, source_regs, NORMALIZE);
        warp::copy(cold_regs, source_regs);
        warpgroup::store_async(cold_chunk, cold_regs);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
}

template <typename C, typename OutputTT, typename ColdTT>
__device__ __forceinline__ void fp4pv_merge_fp16_cold_output_streaming(
    OutputTT &main_output,
    ColdTT &cold,
    float correction_row0,
    float correction_row8,
    float correction_row16,
    float correction_row24
) {
    static_assert(OutputTT::cols == 128 && ColdTT::cols == 128,
                  "full-output cold merge expects logical DVO128");
    if (warpgroup::laneid() >= C::Mb) {
        return;
    }
    constexpr float DENORMALIZE =
        36.0f * FP4PV_MXFP4_ONLINE_P_SCALE;
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        rt_fl<32, 32> main_regs;
        rt_hf<32, 32> cold_regs;
        rt_fl<32, 32> restored_regs;
        auto main_chunk =
            main_output.template subtile<full_tt_fl<32>>(chunk * 32);
        auto cold_chunk =
            cold.template subtile<full_tt_hf<32>>(chunk * 32);
        warpgroup::load_async(main_regs, main_chunk);
        fp4pv_load_packed_fp16_cold(cold_regs, cold_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        warp::copy(restored_regs, cold_regs);
        warp::mul(restored_regs, restored_regs, DENORMALIZE);
        #pragma unroll
        for (int tile_row = 0; tile_row < restored_regs.height; ++tile_row) {
            #pragma unroll
            for (int tile_col = 0; tile_col < restored_regs.width; ++tile_col) {
                #pragma unroll
                for (int word = 0; word < 4; ++word) {
                    // TK row-layout base tiles assign each lane four rows:
                    // tile-row selects +0/+16 and word parity selects +0/+8.
                    const float correction = tile_row == 0
                        ? ((word & 1) == 0
                               ? correction_row0
                               : correction_row8)
                        : ((word & 1) == 0
                               ? correction_row16
                               : correction_row24);
                    const float applied_correction =
                        correction < 0.999f ? correction : 1.0f;
                    const float2 corr2 = {
                        applied_correction, applied_correction};
                    float2 &main_value =
                        main_regs.tiles[tile_row][tile_col].data[word];
                    const float2 &cold_value =
                        restored_regs.tiles[tile_row][tile_col].data[word];
                    main_value = __fadd2_rn(
                        __fmul2_rn(main_value, corr2), cold_value);
                }
            }
        }
        warpgroup::store_async(main_chunk, main_regs);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
}

#if TK_FA4_DVO64_TWO_PASS_BUILD
template <typename ColdRT, typename ColdChunk>
__device__ __forceinline__ void fp4pv_dvo64_load_packed_16b_cold(
    ColdRT &destination,
    const ColdChunk &source
) {
    static_assert(ColdRT::rows == 32 && ColdRT::cols == 32 &&
                  ColdRT::height == 2 && ColdRT::width == 2,
                  "DVO64 cold load expects a packed 32x32 FP16 register tile");
    #pragma unroll
    for (int tile_row = 0; tile_row < 2; ++tile_row) {
        asm volatile(
            "{tcgen05.ld.sync.aligned.16x128b.x4.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];}"
            : "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][0].data[0])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][0].data[1])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][0].data[2])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][0].data[3])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][1].data[0])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][1].data[1])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][1].data[2])),
              "=r"(*reinterpret_cast<uint32_t *>(
                       &destination.tiles[tile_row][1].data[3]))
            : "r"(source.addr +
                    (((warpgroup::warpid() * 32 + tile_row * 16)) << 16)));
    }
}

template <int BARRIER_ID = 1, bool NORMALIZE_FP16 = false,
          typename OutputTT, typename ColdTT>
__device__ __forceinline__ void fp4pv_dvo64_save_fp16_cold(
    OutputTT &active,
    ColdTT &cold
) {
    static_assert(OutputTT::cols == 64 && ColdTT::cols == 64,
                  "DVO64 checkpoint expects logical width 64");
    #pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
        rt_fl<32, 32> active_regs;
        using cold_rt = std::conditional_t<
            std::is_same_v<typename ColdTT::dtype, half>,
            rt_hf<32, 32>, rt_bf<32, 32>>;
        using cold_chunk_t = std::conditional_t<
            std::is_same_v<typename ColdTT::dtype, half>,
            full_tt_hf<32>, full_tt_bf<32>>;
        cold_rt cold_regs;
        auto active_chunk =
            active.template subtile<full_tt_fl<32>>(chunk * 32);
        auto cold_chunk =
            cold.template subtile<cold_chunk_t>(chunk * 32);
        warpgroup::load_async(active_regs, active_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        if constexpr (NORMALIZE_FP16) {
            // Live MXFP4 PV accumulators carry both the GEMM factor and the
            // online P scale in their raw FP32 domain. Remove the complete
            // factor before FP16 packing to avoid exponent overflow.
            warp::mul(active_regs, active_regs,
                      1.0f / (36.0f * FP4PV_MXFP4_ONLINE_P_SCALE));
        }
        warp::copy(cold_regs, active_regs);
        warpgroup::store_async(cold_chunk, cold_regs);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
    warpgroup::sync(BARRIER_ID);
}

template <int BARRIER_ID = 1, bool NORMALIZE_FP16 = false,
          typename OutputTT, typename ColdTT>
__device__ __forceinline__ void fp4pv_dvo64_swap_fp16_cold(
    OutputTT &active,
    ColdTT &cold
) {
    static_assert(OutputTT::cols == 64 && ColdTT::cols == 64,
                  "DVO64 swap expects logical width 64");
    #pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
        rt_fl<32, 32> outgoing_active;
        rt_fl<32, 32> incoming_active;
        using cold_rt = std::conditional_t<
            std::is_same_v<typename ColdTT::dtype, half>,
            rt_hf<32, 32>, rt_bf<32, 32>>;
        using cold_chunk_t = std::conditional_t<
            std::is_same_v<typename ColdTT::dtype, half>,
            full_tt_hf<32>, full_tt_bf<32>>;
        cold_rt outgoing_cold;
        cold_rt incoming_cold;
        auto active_chunk =
            active.template subtile<full_tt_fl<32>>(chunk * 32);
        auto cold_chunk =
            cold.template subtile<cold_chunk_t>(chunk * 32);
        warpgroup::load_async(outgoing_active, active_chunk);
        fp4pv_dvo64_load_packed_16b_cold(incoming_cold, cold_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        if constexpr (NORMALIZE_FP16) {
            warp::mul(outgoing_active, outgoing_active,
                      1.0f / (36.0f * FP4PV_MXFP4_ONLINE_P_SCALE));
        }
        warp::copy(outgoing_cold, outgoing_active);
        warp::copy(incoming_active, incoming_cold);
        if constexpr (NORMALIZE_FP16) {
            warp::mul(incoming_active, incoming_active,
                      36.0f * FP4PV_MXFP4_ONLINE_P_SCALE);
        }
        warpgroup::store_async(cold_chunk, outgoing_cold);
        warpgroup::store_async(active_chunk, incoming_active);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
    warpgroup::sync(BARRIER_ID);
}

template <int BARRIER_ID = 1, bool NORMALIZE_FP16 = false,
          typename OutputTT, typename ColdTT>
__device__ __forceinline__ void fp4pv_dvo64_restore_fp16_cold(
    OutputTT &active,
    ColdTT &cold
) {
    static_assert(OutputTT::cols == 64 && ColdTT::cols == 64,
                  "DVO64 restore expects logical width 64");
    #pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
        using cold_rt = std::conditional_t<
            std::is_same_v<typename ColdTT::dtype, half>,
            rt_hf<32, 32>, rt_bf<32, 32>>;
        using cold_chunk_t = std::conditional_t<
            std::is_same_v<typename ColdTT::dtype, half>,
            full_tt_hf<32>, full_tt_bf<32>>;
        cold_rt cold_regs;
        rt_fl<32, 32> active_regs;
        auto active_chunk =
            active.template subtile<full_tt_fl<32>>(chunk * 32);
        auto cold_chunk =
            cold.template subtile<cold_chunk_t>(chunk * 32);
        fp4pv_dvo64_load_packed_16b_cold(cold_regs, cold_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        warp::copy(active_regs, cold_regs);
        if constexpr (NORMALIZE_FP16) {
            warp::mul(active_regs, active_regs,
                      36.0f * FP4PV_MXFP4_ONLINE_P_SCALE);
        }
        warpgroup::store_async(active_chunk, active_regs);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
    warpgroup::sync(BARRIER_ID);
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 2)
__global__ void kernel_mxfp4_dvo64_lean2wg(
    const __grid_constant__ globals_mxfp4_dvo64_lean2wg<C> g
) {
    using G = globals_mxfp4_dvo64_lean2wg<C>;
    using d_tt_score = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_output_tail = tt<float, C::Mb, 32>;
    using d_tt_qk_scale = full_tt_fp8e4m3<16>;
    using d_tt_pv_scale = full_tt_fp8e8m0<16>;
    constexpr bool PAIRED_CTA = C::CLUSTER_SIZE == 2;
    constexpr bool ONE_WALK_REGISTER_ACCUM = [] {
        if constexpr (requires {
            C::DVO64_ONE_WALK_REGISTER_ACCUM;
        }) {
            return C::DVO64_ONE_WALK_REGISTER_ACCUM;
        }
        return false;
    }();
    constexpr bool ONE_WALK_PARTIAL_OVERLAP = [] {
        if constexpr (requires {
            C::DVO64_ONE_WALK_PARTIAL_OVERLAP;
        }) {
            return C::DVO64_ONE_WALK_PARTIAL_OVERLAP;
        }
        return false;
    }();
    constexpr bool ONE_WALK_EAGER_SECOND_PV = [] {
        if constexpr (requires {
            C::DVO64_ONE_WALK_EAGER_SECOND_PV;
        }) {
            return C::DVO64_ONE_WALK_EAGER_SECOND_PV;
        }
        return false;
    }();
    constexpr bool ONE_WALK_DEFER_SECOND_PV_WAIT = [] {
        if constexpr (requires {
            C::DVO64_ONE_WALK_DEFER_SECOND_PV_WAIT;
        }) {
            return C::DVO64_ONE_WALK_DEFER_SECOND_PV_WAIT;
        }
        return false;
    }();
    constexpr bool ONE_WALK_FP16_COLD_SWAP = [] {
        if constexpr (requires {
            C::DVO64_ONE_WALK_FP16_COLD_SWAP;
        }) {
            return C::DVO64_ONE_WALK_FP16_COLD_SWAP;
        }
        return false;
    }();
    constexpr bool ONE_WALK_BF16_COLD_SWAP = [] {
        if constexpr (requires {
            C::DVO64_ONE_WALK_BF16_COLD_SWAP;
        }) {
            return C::DVO64_ONE_WALK_BF16_COLD_SWAP;
        }
        return false;
    }();
    constexpr bool ONE_WALK_PACKED_COLD_SWAP =
        ONE_WALK_FP16_COLD_SWAP || ONE_WALK_BF16_COLD_SWAP;
    using d_tt_output_cold = std::conditional_t<
        ONE_WALK_BF16_COLD_SWAP, full_tt_bf<64>, full_tt_hf<64>>;
    constexpr int SCORE_BASE = 0;
    constexpr int OUTPUT_BASE = ONE_WALK_PARTIAL_OVERLAP ? 96 : 128;
    constexpr int OUTPUT_HALF1_BASE =
        ONE_WALK_PARTIAL_OVERLAP ? 160 : OUTPUT_BASE;
    constexpr int OUTPUT_COLD_BASE = 192;
    constexpr int Q_SCALE_BASE =
        (ONE_WALK_PARTIAL_OVERLAP || ONE_WALK_PACKED_COLD_SWAP) ? 224 : 192;
    constexpr int K_SCALE_BASE =
        (ONE_WALK_PARTIAL_OVERLAP || ONE_WALK_PACKED_COLD_SWAP) ? 240 : 208;
    constexpr int P_SCALE_BASE =
        (ONE_WALK_PARTIAL_OVERLAP || ONE_WALK_PACKED_COLD_SWAP)
            ? Q_SCALE_BASE : 224;
    constexpr int V_SCALE_BASE =
        (ONE_WALK_PARTIAL_OVERLAP || ONE_WALK_PACKED_COLD_SWAP)
            ? K_SCALE_BASE : 240;
    using PVMmaConfig = std::conditional_t<
        PAIRED_CTA, config_fp4pv_dvo64_lean2wg_ex2e16pc, C>;
    constexpr float LOG2E_OVER_SQRT_D = 0.10408340855860843f;
    constexpr float MXFP4_GEMM_SCALE_CORRECTION =
        1.0f / (36.0f * FP4PV_MXFP4_ONLINE_P_SCALE);
    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2,
                  "lean DVO64 route requires two 128-thread warp groups");
    static_assert((C::CLUSTER_SIZE == 1 || C::CLUSTER_SIZE == 2) &&
                  C::Mb == 128 && C::Nb == 128 &&
                  C::Dqk == 192 && C::Dvo == 64,
                  "lean DVO64 route requires M128/N128/K192/D64");
    static_assert(!ONE_WALK_EAGER_SECOND_PV ||
                  ONE_WALK_PARTIAL_OVERLAP,
                  "queued second PV requires the partial-overlap map");
    static_assert(!ONE_WALK_DEFER_SECOND_PV_WAIT ||
                  ONE_WALK_EAGER_SECOND_PV,
                  "deferred second-PV completion requires queued issue");
    static_assert(!(ONE_WALK_FP16_COLD_SWAP && ONE_WALK_BF16_COLD_SWAP),
                  "cold checkpoint dtype must be unambiguous");
    static_assert(!ONE_WALK_PACKED_COLD_SWAP ||
                  (ONE_WALK_REGISTER_ACCUM &&
                   !ONE_WALK_PARTIAL_OVERLAP &&
                   !ONE_WALK_EAGER_SECOND_PV &&
                   !PAIRED_CTA),
                  "FP16 cold swap requires the serial one-walk DVO64 parent");
    static_assert(V_SCALE_BASE + 16 == tensor_allocator<2, 1>::cols,
                  "lean DVO64 route must exactly fill 256 TMEM columns");
    static_assert(!ONE_WALK_PACKED_COLD_SWAP || OUTPUT_COLD_BASE + 32 == Q_SCALE_BASE,
                  "packed cold checkpoint must occupy physical columns 192..223");

    tensor_allocator<2, 1> tm_alloc{};
    d_tt_score tt_score = tm_alloc.template allocate<d_tt_score>(SCORE_BASE);
    d_tt_output tt_output =
        tm_alloc.template allocate<d_tt_output>(OUTPUT_BASE);
    d_tt_output tt_output_half1 =
        tm_alloc.template allocate<d_tt_output>(OUTPUT_HALF1_BASE);
    d_tt_output_cold tt_output_cold =
        tm_alloc.template allocate<d_tt_output_cold>(OUTPUT_COLD_BASE);
    d_tt_output_tail tt_output0_tail =
        tm_alloc.template allocate<d_tt_output_tail>(OUTPUT_BASE + 32);
    d_tt_output tt_pv_temp(
        (ONE_WALK_PARTIAL_OVERLAP || ONE_WALK_PACKED_COLD_SWAP)
            ? tt_output.addr : tt_score.addr);
    d_tt_output tt_cold_restore(tt_score.addr);
    d_tt_qk_scale q_sc_tm =
        tm_alloc.template allocate<d_tt_qk_scale>(Q_SCALE_BASE);
    d_tt_qk_scale k_sc_tm =
        tm_alloc.template allocate<d_tt_qk_scale>(K_SCALE_BASE);
    d_tt_pv_scale p_sc_tm =
        tm_alloc.template allocate<d_tt_pv_scale>(P_SCALE_BASE);
    d_tt_pv_scale v_sc_tm =
        tm_alloc.template allocate<d_tt_pv_scale>(V_SCALE_BASE);
    d_tt_pv_scale v_sc_tm_half1 =
        tm_alloc.template allocate<d_tt_pv_scale>(
            ONE_WALK_EAGER_SECOND_PV ? SCORE_BASE : V_SCALE_BASE);

    const int cta_rank = PAIRED_CTA ? cluster_ctarank() : 0;
    const int logical_block = PAIRED_CTA ? blockIdx.x / 2 : blockIdx.x;
    const int dvo_half = PAIRED_CTA ? cta_rank :
        (ONE_WALK_REGISTER_ACCUM ? 0 : g.dvo_half);
    const bool score_owner = !PAIRED_CTA || cta_rank == 0;
    const int tiles = g.seqlen / C::Mb;
    const int num_hb = g.batch * g.heads;
    const int m_pos = logical_block / num_hb;
    const int hb = logical_block - m_pos * num_hb;
    const int b = hb / g.heads;
    const int h = hb - b * g.heads;
    const int query_tile = tiles - 1 - m_pos;
    const int iters = query_tile + 1;

#if TK_FA4_DVO64_LEAN_TRACE_BUILD
    constexpr int TRACE_FIELDS = 10;
    const unsigned long long trace_record =
        static_cast<unsigned long long>(dvo_half) * gridDim.x + blockIdx.x;
    auto *trace = reinterpret_cast<unsigned long long *>(
        g.o + static_cast<unsigned long long>(g.batch) * g.seqlen *
                  g.heads * 128) + trace_record * TRACE_FIELDS;
    auto trace_now = []() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    auto trace_store = [&](int field, unsigned long long value) {
        if (g.trace_enabled != 0) {
            trace[field] = value;
        }
    };
    if (threadIdx.x == 0 && g.trace_enabled != 0) {
        trace_store(0, 0);
        trace_store(1, blockIdx.x);
        trace_store(2, kittens::smid());
        trace_store(3, dvo_half);
        trace_store(4, tm_alloc.addr);
        trace_store(5, query_tile);
        trace_store(6, iters);
        trace_store(7, trace_now());
        trace_store(8, 0);
        trace_store(9, 0);
    }
#endif

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile &q_smem = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem =
        al.allocate<typename G::q_sc_tile>();
    typename G::k_tile &k_smem = al.allocate<typename G::k_tile>();
    typename G::k_sc_tile &k_sc_smem =
        al.allocate<typename G::k_sc_tile>();
    typename G::p_fp4_tile &p_fp4_smem =
        al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem =
        al.allocate<typename G::p_sc_tile>();
    typename G::v_fp4_tile &v_fp4_smem =
        al.allocate<typename G::v_fp4_tile>();
    typename G::v_sc_tile &v_sc_smem =
        al.allocate<typename G::v_sc_tile>();

    __shared__ semaphore qk_done[2];
    __shared__ semaphore qk_scale_done[C::QK_SCALE_CHUNKS];
    __shared__ semaphore p_ready[2];
    __shared__ semaphore pv_src_done[2];
    __shared__ semaphore pv_done[2];
    __shared__ semaphore pv_src_done_half1[2];
    __shared__ semaphore pv_done_half1[2];
    __shared__ semaphore v_scale_done;
    __shared__ semaphore q_payload_done;
    __shared__ semaphore k_payload_done;
    __shared__ semaphore v_payload_done;
    __shared__ semaphore v_payload_done_half1;
    __shared__ semaphore pair_p_ready;
    __shared__ semaphore cold_ready;
    __shared__ semaphore cold_reusable;
    __shared__ float pair_acc_scale[C::Mb];
    __shared__ float pair_row_sum[C::Mb];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            init_semaphore(qk_done[slot], 0, 1);
            init_semaphore(p_ready[slot], 0, 1);
            init_semaphore(pv_src_done[slot], 0, 1);
            init_semaphore(pv_done[slot], 0, 1);
            init_semaphore(pv_src_done_half1[slot], 0, 1);
            init_semaphore(pv_done_half1[slot], 0, 1);
        }
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            init_semaphore(qk_scale_done[chunk], 0, 1);
        }
        init_semaphore(v_scale_done, 0, 1);
        init_semaphore(q_payload_done, 0, 1);
        init_semaphore(k_payload_done, 0, 1);
        init_semaphore(v_payload_done, 0, 1);
        init_semaphore(v_payload_done_half1, 0, 1);
        init_semaphore(pair_p_ready, 0, 1);
        init_semaphore(cold_ready, 0, 1);
        init_semaphore(cold_reusable, 0, 1);
    }
    __syncthreads();
    if constexpr (PAIRED_CTA) {
        fp4pv_cluster_sync<C>();
    }
    if constexpr (ONE_WALK_PACKED_COLD_SWAP) {
        if (threadIdx.x == 0) {
            // The initial empty checkpoint slot is writable at phase zero.
            arrive(cold_reusable);
        }
        __syncthreads();
    }

    auto load_scale = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
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
    auto load_v_scale = [&](int key_tile, int output_half) {
        const uint8_t *v_sc_src = g.v_sc +
            ((((b * g.heads + h) * 2 + output_half) * tiles + key_tile) *
             sizeof(typename G::v_sc_tile));
        load_scale(v_sc_smem, v_sc_src);
    };

    const bool producer_group = warpgroup::groupid() == 0;
    const uint8_t *q_sc_src = g.q_sc +
        (((b * tiles + query_tile) * (g.heads * C::QK_SCALE_CHUNKS) +
          h * C::QK_SCALE_CHUNKS) * 512);
    const coord<typename G::q_tile> q_coord = {
        b, h, query_tile, 0
    };
    if (producer_group && score_owner) {
        if (threadIdx.x == 0) {
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                q_smem, g.q, q_coord, q_payload_done);
            tma::expect_bytes(q_payload_done, sizeof(typename G::q_tile));
        }
        load_scale(q_sc_smem, q_sc_src);
        warpgroup::sync(1);
        if (threadIdx.x == 0) {
            wait(q_payload_done, 0);
        }
        warpgroup::sync(1);
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(1);
    }

    const bool issue_warp =
        producer_group && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const float scale_log2 = LOG2E_OVER_SQRT_D *
        g.q_sg[b * g.heads + h] * g.k_sg[b * g.heads + h];
    const bool zero_scale = scale_log2 == 0.0f;
    float row_max = base_types::constants<float>::neg_infty();
    float row_sum = 0.0f;
    int v_scale_phase = 0;
    constexpr int REGISTER_OUTPUT_PAIRS = ONE_WALK_PACKED_COLD_SWAP
        ? 1 : (ONE_WALK_PARTIAL_OVERLAP ? 16 : 32);
    float2 register_output[REGISTER_OUTPUT_PAIRS];
    if constexpr (ONE_WALK_REGISTER_ACCUM && !ONE_WALK_PACKED_COLD_SWAP) {
        if (producer_group) {
        #pragma unroll
            for (int pair = 0; pair < REGISTER_OUTPUT_PAIRS; ++pair) {
                register_output[pair] = {0.0f, 0.0f};
            }
        }
    }

    if constexpr (PAIRED_CTA) {
        constexpr uint32_t PAIR_P_BYTES =
            sizeof(typename G::p_fp4_tile) +
            sizeof(typename G::p_sc_tile);
        constexpr uint32_t PAIR_ACC_BYTES = C::Mb * sizeof(float);
        constexpr uint32_t PAIR_ROW_SUM_BYTES = C::Mb * sizeof(float);
        const uint32_t first_pair_bytes = PAIR_P_BYTES +
            (iters == 1 ? PAIR_ROW_SUM_BYTES : 0);
        if (cta_rank == 1 && threadIdx.x == 0) {
            tma::cluster::expect_bytes(pair_p_ready, first_pair_bytes);
        }
        fp4pv_cluster_sync<C>();
    }

    #pragma unroll 1
    for (int key_tile = 0; key_tile < iters; ++key_tile) {
        const int slot = key_tile & 1;
        const int phase = (key_tile >> 1) & 1;
        const uint8_t *k_sc_src = g.k_sc +
            (((b * (tiles * 2) + key_tile * 2) *
              (g.heads * C::QK_SCALE_CHUNKS) +
              h * C::QK_SCALE_CHUNKS) * 512);
        const coord<typename G::k_tile> k_coord = {
            b, h, key_tile, 0
        };
        const coord<typename G::v_fp4_tile> v_coord = {
            b, 2 * h + dvo_half, 0, key_tile
        };
        if (producer_group) {
            if (threadIdx.x == 0) {
                if (score_owner) {
                    tma::load_async<dim::ROW, cache_policy::NORMAL>(
                        k_smem, g.k, k_coord, k_payload_done);
                    tma::expect_bytes(
                        k_payload_done, sizeof(typename G::k_tile));
                }
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    v_fp4_smem, g.v_fp4_tma, v_coord, v_payload_done);
                tma::expect_bytes(
                    v_payload_done, sizeof(typename G::v_fp4_tile));
            }
            if (score_owner) {
                load_scale(k_sc_smem, k_sc_src);
            }
            load_v_scale(key_tile, dvo_half);
            warpgroup::sync(1);
            if (threadIdx.x == 0) {
                if (score_owner) {
                    wait(k_payload_done, key_tile & 1);
                }
                wait(v_payload_done, key_tile & 1);
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
        }

        if (issue_warp && score_owner) {
            using T_AB = typename G::q_tile::T;
            using T_SAB = typename d_tt_qk_scale::T;
            using T_D = typename d_tt_score::T;
            constexpr uint32_t IDESC =
                detail::tcgen05::instruction_descriptor<
                    T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
            st_descriptor<typename G::q_tile, 0> q_desc(q_smem);
            st_descriptor<typename G::k_tile, 0> k_desc(k_smem);
            #pragma unroll
            for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                if (issue_lane) {
                    fp4pv_load_q_scale_chunk_ranklocal(
                        q_sc_tm, q_sc_smem, chunk);
                    fp4pv_load_k_scale_chunk_ranklocal(
                        k_sc_tm, k_sc_smem, chunk);
                    detail::tcgen05::commit<1>(qk_scale_done[chunk]);
                }
                wait(qk_scale_done[chunk], key_tile & 1);
                tensor_after_thread_sync();
                tensor_before_thread_sync();
                if (issue_lane) {
                    if (chunk == 0) {
                        detail::tcgen05::template st_st<
                            T_AB, T_SAB, 0, 1, 16>(
                                tt_score.addr,
                                q_desc.chunk_descriptor(chunk),
                                k_desc.chunk_descriptor(chunk),
                                q_sc_tm.addr, k_sc_tm.addr, IDESC);
                    } else {
                        detail::tcgen05::template st_st<
                            T_AB, T_SAB, 1, 1, 16>(
                                tt_score.addr,
                                q_desc.chunk_descriptor(chunk),
                                k_desc.chunk_descriptor(chunk),
                                q_sc_tm.addr, k_sc_tm.addr, IDESC);
                    }
                }
            }
            if (issue_lane) {
                detail::tcgen05::commit<1>(qk_done[slot]);
            }
        }

        float acc_scale = 1.0f;
        if (warpgroup::groupid() == 1 && score_owner) {
            wait(qk_done[slot], phase);
            tensor_after_thread_sync();
            const int row = warpgroup::laneid();
            const int query_col = query_tile * C::Mb + row;
            const uint32_t score_tt_base =
                tt_score.addr + ((warpgroup::warpid() * 32) << 16);
            float block_max[4];
            float tile_max = base_types::constants<float>::neg_infty();
            #pragma unroll
            for (int qid = 0; qid < 4; ++qid) {
                float2 score_quarter[16];
                fp4pv_load_score_quarter_tmem(
                    score_quarter, score_tt_base, qid);
                tensor_load_wait();
                float local_max = base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int pair = 0; pair < 16; ++pair) {
                    const int key_col = key_tile * C::Nb +
                        (qid * 16 + pair) * 2;
                    if (key_col > query_col) {
                        score_quarter[pair].x =
                            base_types::constants<float>::neg_infty();
                    }
                    if (key_col + 1 > query_col) {
                        score_quarter[pair].y =
                            base_types::constants<float>::neg_infty();
                    }
                    asm volatile("{max.f32 %0, %1, %2, %3;}"
                        : "=f"(local_max)
                        : "f"(local_max), "f"(score_quarter[pair].x),
                          "f"(score_quarter[pair].y));
                }
                block_max[qid] = local_max;
                tile_max = fmaxf(tile_max, local_max);
            }

            const float old_max = row_max;
            row_max = fmaxf(row_max, tile_max);
            if (key_tile > 0 && !zero_scale) {
                acc_scale = softmax_rescale_factor_strict(
                    row_max, old_max, scale_log2);
            }

            uint32_t packed_scales = 0;
            float tile_sum = 0.0f;
            #pragma unroll
            for (int qid = 0; qid < 4; ++qid) {
                uint8_t p_e8m0 = 0;
                if (isfinite(block_max[qid]) && isfinite(row_max)) {
                    const float block_scale_log2 =
                        FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                        (block_max[qid] - row_max) * scale_log2;
                    p_e8m0 =
                        fp4pv_log2_to_e8m0_floor_finite(block_scale_log2);
                }
                packed_scales |= static_cast<uint32_t>(p_e8m0) << (qid * 8);
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
                float2 sum0 = {0.0f, 0.0f};
                float2 sum1 = {0.0f, 0.0f};
                float2 score_quarter[16];
                fp4pv_load_score_quarter_tmem(
                    score_quarter, score_tt_base, qid);
                tensor_load_wait();
                uint32_t words[4];
                #pragma unroll
                for (int pack = 0; pack < 4; ++pack) {
                    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const int jj = pack * 4 + kk;
                        const int key_col = key_tile * C::Nb +
                            (qid * 16 + jj) * 2;
                        if (key_col > query_col) {
                            score_quarter[jj].x =
                                base_types::constants<float>::neg_infty();
                        }
                        if (key_col + 1 > query_col) {
                            score_quarter[jj].y =
                                base_types::constants<float>::neg_infty();
                        }
                        if (!zero_scale) {
                            score_quarter[jj] = __ffma2_rn(
                                score_quarter[jj], scale2, bias2);
                        } else {
                            score_quarter[jj].x =
                                isfinite(score_quarter[jj].x)
                                    ? p_log2_coeff : score_quarter[jj].x;
                            score_quarter[jj].y =
                                isfinite(score_quarter[jj].y)
                                    ? p_log2_coeff : score_quarter[jj].y;
                        }
                        score_quarter[jj] = jj >= 12
                            ? fp4pv_ex2_alu_emulation_f32x2(
                                  score_quarter[jj])
                            : fp4pv_ex2_native_f32x2(score_quarter[jj]);
                        if ((jj & 1) == 0) {
                            sum0 = __fadd2_rn(sum0, score_quarter[jj]);
                        } else {
                            sum1 = __fadd2_rn(sum1, score_quarter[jj]);
                        }
                    }
                    words[pack] = p_e8m0 == 0
                        ? 0u
                        : fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(
                              &score_quarter[pack * 4]);
                }
                sum0 = __fadd2_rn(sum0, sum1);
                tile_sum += (sum0.x + sum0.y) * p_rcp_coeff;
                fp4pv_store_quantized_scores_group_mxfp4(
                    p_fp4_smem, words, row, qid);
            }
            row_sum = row_sum * acc_scale + tile_sum;
            if constexpr (PAIRED_CTA) {
                fp4pv_store_mxfp4_scale_word_swizzled(
                    p_sc_smem, row, packed_scales);
                if (key_tile > 0) {
                    pair_acc_scale[row] = acc_scale;
                }
                if (key_tile + 1 == iters) {
                    pair_row_sum[row] = row_sum;
                }
            } else if constexpr (ONE_WALK_REGISTER_ACCUM) {
                fp4pv_store_mxfp4_scale_word_swizzled(
                    p_sc_smem, row, packed_scales);
                pair_acc_scale[row] = acc_scale;
                if (key_tile + 1 == iters) {
                    pair_row_sum[row] = row_sum;
                }
            } else {
                fp4pv_store_mxfp4_scale_tmem_32x32b_x1(
                    p_sc_tm, warpgroup::warpid(), packed_scales);
                fp4pv_tmem_store_wait();
            }

            if constexpr (ONE_WALK_DEFER_SECOND_PV_WAIT) {
                if (key_tile > 0) {
                    const int previous_key_tile = key_tile - 1;
                    wait(pv_done_half1[previous_key_tile & 1],
                         (previous_key_tile >> 1) & 1);
                    tensor_after_thread_sync();
                }
            }
            if constexpr (ONE_WALK_PACKED_COLD_SWAP) {
                if (key_tile > 0) {
                    // Event 2*k-2 (phase zero) contains half 0 from the
                    // previous key tile. WG1 consumes it, checkpoints the
                    // inactive half 1, and publishes event 2*k-1 (phase one).
                    if (warpgroup::warpid() == 0) {
                        wait(cold_ready, 0);
                    }
                    warpgroup::sync(2);
                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                        arrive(cold_reusable);
                    }
                    if (warpgroup::warpid() == 0) {
                        wait(cold_reusable, 1);
                    }
                    warpgroup::sync(2);
                    fp4pv_dvo64_swap_fp16_cold<
                        2, ONE_WALK_FP16_COLD_SWAP>(
                        tt_output, tt_output_cold);
                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                        arrive(cold_ready);
                    }
                }
            }
            if (key_tile > 0) {
                if constexpr (ONE_WALK_PARTIAL_OVERLAP) {
                    rescale_tt_output_inplace_direct_streaming<C>(
                        tt_output0_tail, acc_scale, acc_scale < 0.999f);
                    rescale_tt_output_inplace_direct_streaming<C>(
                        tt_output_half1, acc_scale, acc_scale < 0.999f);
                } else {
                    rescale_tt_output_inplace_direct_streaming<C>(
                        tt_output, acc_scale, acc_scale < 0.999f);
                }
            }
            warpgroup::sync(2);
            fp4pv_publish_shared_backing_proxy_only();
            tensor_before_thread_sync();
            if constexpr (PAIRED_CTA) {
                publish_cluster_shared_backing_if_needed<C>();
                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                    tma::cluster::store_async(
                        p_fp4_smem, p_fp4_smem, 1, pair_p_ready);
                    tma::cluster::store_async(
                        p_sc_smem, p_sc_smem, 1, pair_p_ready);
                    if (key_tile > 0) {
                        tma::cluster::store_async(
                            pair_acc_scale, pair_acc_scale, 1, pair_p_ready);
                    }
                    if (key_tile + 1 == iters) {
                        tma::cluster::store_async(
                            pair_row_sum, pair_row_sum, 1, pair_p_ready);
                    }
                }
            }
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                arrive(p_ready[slot]);
            }
        }

        if constexpr (PAIRED_CTA) {
            if (cta_rank == 1 && warpgroup::groupid() == 1) {
                wait(pair_p_ready, key_tile & 1);
                tensor_after_thread_sync();
                const int row = warpgroup::laneid();
                if (key_tile > 0) {
                    acc_scale = pair_acc_scale[row];
                    rescale_tt_output_inplace_direct_streaming<C>(
                        tt_output, acc_scale, acc_scale < 0.999f);
                }
                if (key_tile + 1 == iters) {
                    row_sum = pair_row_sum[row];
                }
            }
        }

        if (issue_warp) {
            if constexpr (PAIRED_CTA) {
                if (cta_rank == 0) {
                    wait(p_ready[slot], phase);
                } else {
                    wait(pair_p_ready, key_tile & 1);
                }
            } else {
                wait(p_ready[slot], phase);
            }
            tensor_after_thread_sync();
        }

        if constexpr (ONE_WALK_PARTIAL_OVERLAP) {
            if (producer_group) {
                // QK overwrites output-half0 columns 0..31. Restore only
                // that overlap after the score consumer publishes P.
                warpgroup::sync(1);
                if (key_tile > 0) {
                    const int row = warpgroup::laneid();
                    const float raw_correction = pair_acc_scale[row];
                    const float correction = raw_correction < 0.999f
                        ? raw_correction : 1.0f;
                    const float2 correction2 = {correction, correction};
                    #pragma unroll
                    for (int col = 0; col < 32; col += 16) {
                        float2 out[8];
                        #pragma unroll
                        for (int pair = 0; pair < 8; ++pair) {
                            out[pair] = __fmul2_rn(
                                register_output[col / 2 + pair],
                                correction2);
                        }
                        const uint32_t addr = tt_output.addr +
                            ((warpgroup::warpid() * 32) << 16) + col;
                        asm volatile(
                            "{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
                            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                            :: "f"(out[0].x), "f"(out[0].y),
                               "f"(out[1].x), "f"(out[1].y),
                               "f"(out[2].x), "f"(out[2].y),
                               "f"(out[3].x), "f"(out[3].y),
                               "f"(out[4].x), "f"(out[4].y),
                               "f"(out[5].x), "f"(out[5].y),
                               "f"(out[6].x), "f"(out[6].y),
                               "f"(out[7].x), "f"(out[7].y),
                               "r"(addr));
                    }
                    tensor_store_wait();
                }
                warpgroup::sync(1);
                tensor_before_thread_sync();
            }
        }

        if (issue_warp) {
            if (issue_lane) {
                if constexpr (PAIRED_CTA || ONE_WALK_REGISTER_ACCUM) {
                    fp4pv_load_mxnv_scale_async_ranklocal(
                        p_sc_tm, p_sc_smem);
                }
                fp4pv_load_mxnv_scale_async_ranklocal(
                    v_sc_tm, v_sc_smem);
                detail::tcgen05::commit<1>(v_scale_done);
            }
            wait(v_scale_done, v_scale_phase);
            v_scale_phase ^= 1;
            tensor_after_thread_sync();
            tensor_before_thread_sync();
            if (issue_lane) {
                if constexpr (ONE_WALK_REGISTER_ACCUM) {
                    if constexpr (ONE_WALK_PARTIAL_OVERLAP) {
                        if (key_tile == 0) {
                            fp4pv_p_stage_ABt_split_k64<
                                PVMmaConfig, true>(
                                    tt_pv_temp, p_fp4_smem, v_fp4_smem,
                                    p_sc_tm, v_sc_tm, pv_src_done[slot]);
                        } else {
                            fp4pv_p_stage_ABt_split_k64<
                                PVMmaConfig, false>(
                                    tt_pv_temp, p_fp4_smem, v_fp4_smem,
                                    p_sc_tm, v_sc_tm, pv_src_done[slot]);
                        }
                    } else if constexpr (ONE_WALK_PACKED_COLD_SWAP) {
                        if (key_tile == 0) {
                            fp4pv_p_stage_ABt_split_k64<PVMmaConfig, true>(
                                tt_pv_temp, p_fp4_smem, v_fp4_smem,
                                p_sc_tm, v_sc_tm, pv_src_done[slot]);
                        } else {
                            fp4pv_p_stage_ABt_split_k64<PVMmaConfig, false>(
                                tt_pv_temp, p_fp4_smem, v_fp4_smem,
                                p_sc_tm, v_sc_tm, pv_src_done[slot]);
                        }
                    } else {
                        fp4pv_p_stage_ABt_split_k64<PVMmaConfig, true>(
                            tt_pv_temp, p_fp4_smem, v_fp4_smem,
                            p_sc_tm, v_sc_tm, pv_src_done[slot]);
                    }
                } else {
                    if (key_tile == 0) {
                        fp4pv_p_stage_ABt_split_k64<PVMmaConfig, true>(
                            tt_output, p_fp4_smem, v_fp4_smem,
                            p_sc_tm, v_sc_tm, pv_src_done[slot]);
                    } else {
                        fp4pv_p_stage_ABt_split_k64<PVMmaConfig, false>(
                            tt_output, p_fp4_smem, v_fp4_smem,
                            p_sc_tm, v_sc_tm, pv_src_done[slot]);
                    }
                }
                tensor_commit<1>(pv_done[slot]);
            }
        }

        if constexpr (ONE_WALK_REGISTER_ACCUM) {
            if (producer_group) {
                if constexpr (ONE_WALK_EAGER_SECOND_PV) {
                    wait(pv_src_done[slot], phase);
                } else {
                    wait(pv_done[slot], phase);
                }
                tensor_after_thread_sync();
                warpgroup::sync(1);

                if constexpr (ONE_WALK_PACKED_COLD_SWAP) {
                    if (key_tile == 0) {
                        // Initial empty-slot grant is phase zero. The first
                        // PV creates half 0 in active FP32, then checkpoints it
                        // before the clear-form half-1 PV reuses active.
                        if (warpgroup::warpid() == 0) {
                            wait(cold_reusable, 0);
                        }
                        warpgroup::sync(1);
                        fp4pv_dvo64_save_fp16_cold<
                            1, ONE_WALK_FP16_COLD_SWAP>(
                            tt_output, tt_output_cold);
                    } else {
                        // Event 2*k-1 (phase one) contains the inactive half 1.
                        // Consume it only after half-0 PV completes, then put
                        // the updated half 0 back in the cold slot as event 2*k.
                        if (warpgroup::warpid() == 0) {
                            wait(cold_ready, 1);
                        }
                        warpgroup::sync(1);
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            arrive(cold_reusable);
                        }
                        if (warpgroup::warpid() == 0) {
                            wait(cold_reusable, 0);
                        }
                        warpgroup::sync(1);
                        fp4pv_dvo64_swap_fp16_cold<
                            1, ONE_WALK_FP16_COLD_SWAP>(
                            tt_output, tt_output_cold);
                        const int row = warpgroup::laneid();
                        const float raw_correction = pair_acc_scale[row];
                        const float correction = raw_correction < 0.999f
                            ? raw_correction : 1.0f;
                        rescale_tt_output_inplace_direct_streaming<C>(
                            tt_output, correction, correction < 0.999f);
                        warpgroup::sync(1);
                    }
                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                        arrive(cold_ready);
                    }
                }

                const coord<typename G::v_fp4_tile> v_half1_coord = {
                    b, 2 * h + 1, 0, key_tile
                };
                if (threadIdx.x == 0) {
                    tma::load_async<dim::ROW, cache_policy::NORMAL>(
                        v_fp4_smem, g.v_fp4_tma, v_half1_coord,
                        v_payload_done_half1);
                    tma::expect_bytes(
                        v_payload_done_half1,
                        sizeof(typename G::v_fp4_tile));
                }
                load_v_scale(key_tile, 1);
                warpgroup::sync(1);
                if (threadIdx.x == 0) {
                    wait(v_payload_done_half1, key_tile & 1);
                }
                warpgroup::sync(1);
                fp4pv_publish_shared_backing_proxy_only();
                warpgroup::sync(1);

                if (issue_warp) {
                    if (issue_lane) {
                        fp4pv_load_mxnv_scale_async_ranklocal(
                            v_sc_tm_half1, v_sc_smem);
                        detail::tcgen05::commit<1>(v_scale_done);
                    }
                    wait(v_scale_done, v_scale_phase);
                    v_scale_phase ^= 1;
                    tensor_after_thread_sync();
                    tensor_before_thread_sync();
                    if (issue_lane) {
                        if (key_tile == 0) {
                            fp4pv_p_stage_ABt_split_k64<
                                PVMmaConfig, true>(
                                    tt_output_half1,
                                    p_fp4_smem, v_fp4_smem,
                                    p_sc_tm, v_sc_tm_half1,
                                    pv_src_done_half1[slot]);
                        } else {
                            fp4pv_p_stage_ABt_split_k64<
                                PVMmaConfig, false>(
                                    tt_output_half1,
                                    p_fp4_smem, v_fp4_smem,
                                    p_sc_tm, v_sc_tm_half1,
                                    pv_src_done_half1[slot]);
                        }
                        tensor_commit<1>(pv_done_half1[slot]);
                    }
                }

                if constexpr (ONE_WALK_EAGER_SECOND_PV) {
                    wait(pv_done[slot], phase);
                    tensor_after_thread_sync();
                }

                if constexpr (!ONE_WALK_PACKED_COLD_SWAP) {
                    const int row = warpgroup::laneid();
                    const float raw_correction = pair_acc_scale[row];
                    const float correction = raw_correction < 0.999f
                        ? raw_correction : 1.0f;
                    const uint32_t output_tt_base = tt_pv_temp.addr +
                        ((warpgroup::warpid() * 32) << 16);
                    #pragma unroll
                    for (int col = 0;
                         col < REGISTER_OUTPUT_PAIRS * 2; col += 16) {
                        float2 out[8];
                        asm volatile(
                        "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                        : "=f"(out[0].x), "=f"(out[0].y),
                          "=f"(out[1].x), "=f"(out[1].y),
                          "=f"(out[2].x), "=f"(out[2].y),
                          "=f"(out[3].x), "=f"(out[3].y),
                          "=f"(out[4].x), "=f"(out[4].y),
                          "=f"(out[5].x), "=f"(out[5].y),
                          "=f"(out[6].x), "=f"(out[6].y),
                          "=f"(out[7].x), "=f"(out[7].y)
                        : "r"(output_tt_base + col));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        #pragma unroll
                        for (int pair = 0; pair < 8; ++pair) {
                            const int output_pair = col / 2 + pair;
                            if constexpr (ONE_WALK_PARTIAL_OVERLAP) {
                                register_output[output_pair] = out[pair];
                            } else if (key_tile == 0) {
                                register_output[output_pair] = out[pair];
                            } else {
                                register_output[output_pair].x = __fadd_rn(
                                    __fmul_rn(
                                        register_output[output_pair].x,
                                        correction),
                                    out[pair].x);
                                register_output[output_pair].y = __fadd_rn(
                                    __fmul_rn(
                                        register_output[output_pair].y,
                                        correction),
                                    out[pair].y);
                            }
                        }
                    }
                }
                warpgroup::sync(1);
            }

            if (warpgroup::groupid() == 1) {
                if constexpr (!ONE_WALK_DEFER_SECOND_PV_WAIT) {
                    wait(pv_done_half1[slot], phase);
                    tensor_after_thread_sync();
                    warpgroup::sync(2);
                } else if (key_tile + 1 == iters) {
                    wait(pv_done_half1[slot], phase);
                    tensor_after_thread_sync();
                    warpgroup::sync(2);
                }
            }
            if (producer_group) {
                if constexpr (ONE_WALK_DEFER_SECOND_PV_WAIT) {
                    wait(pv_src_done_half1[slot], phase);
                } else {
                    wait(pv_done_half1[slot], phase);
                }
                tensor_after_thread_sync();
                warpgroup::sync(1);
            }
        } else {
            if (warpgroup::groupid() == 1) {
                wait(pv_done[slot], phase);
                tensor_after_thread_sync();
            }
            if (producer_group) {
                if (issue_warp) {
                    wait(pv_done[slot], phase);
                    tensor_after_thread_sync();
                }
                warpgroup::sync(1);
            }
            if constexpr (PAIRED_CTA) {
                if (key_tile + 1 < iters) {
                    constexpr uint32_t PAIR_P_BYTES =
                        sizeof(typename G::p_fp4_tile) +
                        sizeof(typename G::p_sc_tile);
                    constexpr uint32_t PAIR_ACC_BYTES =
                        C::Mb * sizeof(float);
                    constexpr uint32_t PAIR_ROW_SUM_BYTES =
                        C::Mb * sizeof(float);
                    const int next_key_tile = key_tile + 1;
                    const uint32_t next_pair_bytes =
                        PAIR_P_BYTES + PAIR_ACC_BYTES +
                        (next_key_tile + 1 == iters
                            ? PAIR_ROW_SUM_BYTES : 0);
                    if (cta_rank == 1 && threadIdx.x == 0) {
                        tma::cluster::expect_bytes(
                            pair_p_ready, next_pair_bytes);
                    }
                    fp4pv_cluster_sync<C>();
                }
            }
        }
    }

    if constexpr (ONE_WALK_REGISTER_ACCUM) {
        const int row = warpgroup::laneid();
        const float final_row_sum = pair_row_sum[row];
        const bool valid =
            final_row_sum > 0.0f && isfinite(final_row_sum);
        const float output_scale = valid
            ? MXFP4_GEMM_SCALE_CORRECTION / final_row_sum : 0.0f;
        if (producer_group) {
            if constexpr (ONE_WALK_PACKED_COLD_SWAP) {
                // The final even cold event always contains output half 0.
                if (warpgroup::warpid() == 0) {
                    wait(cold_ready, 0);
                }
                warpgroup::sync(1);
                fp4pv_dvo64_restore_fp16_cold<
                    1, ONE_WALK_FP16_COLD_SWAP>(
                    tt_cold_restore, tt_output_cold);
                #pragma unroll
                for (int col = 0; col < C::Dvo; col += 16) {
                    float2 out[8];
                    asm volatile(
                        "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                        : "=f"(out[0].x), "=f"(out[0].y),
                          "=f"(out[1].x), "=f"(out[1].y),
                          "=f"(out[2].x), "=f"(out[2].y),
                          "=f"(out[3].x), "=f"(out[3].y),
                          "=f"(out[4].x), "=f"(out[4].y),
                          "=f"(out[5].x), "=f"(out[5].y),
                          "=f"(out[6].x), "=f"(out[6].y),
                          "=f"(out[7].x), "=f"(out[7].y)
                        : "r"(tt_cold_restore.addr +
                              ((warpgroup::warpid() * 32) << 16) + col));
                    tensor_load_wait();
                    #pragma unroll
                    for (int pair = 0; pair < 8; ++pair) {
                        const bf16_2 packed = __float22bfloat162_rn(
                            __fmul2_rn(
                                out[pair],
                                float2{output_scale, output_scale}));
                        const int out_offset =
                            ((((b * g.seqlen + query_tile * C::Mb + row) *
                               g.heads + h) * 128) +
                             col + pair * 2);
                        *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
                    }
                }
                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                    arrive(cold_reusable);
                }
            } else {
                #pragma unroll
                for (int pair = 0; pair < REGISTER_OUTPUT_PAIRS; ++pair) {
                    const bf16_2 packed = __float22bfloat162_rn(
                        __fmul2_rn(
                            register_output[pair],
                            float2{output_scale, output_scale}));
                    const int out_offset =
                        ((((b * g.seqlen + query_tile * C::Mb + row) *
                           g.heads + h) * 128) + pair * 2);
                    *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
                }
                if constexpr (ONE_WALK_PARTIAL_OVERLAP) {
                    #pragma unroll
                    for (int col = 32; col < C::Dvo; col += 16) {
                        float2 out[8];
                        asm volatile(
                            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                            : "=f"(out[0].x), "=f"(out[0].y),
                              "=f"(out[1].x), "=f"(out[1].y),
                              "=f"(out[2].x), "=f"(out[2].y),
                              "=f"(out[3].x), "=f"(out[3].y),
                              "=f"(out[4].x), "=f"(out[4].y),
                              "=f"(out[5].x), "=f"(out[5].y),
                              "=f"(out[6].x), "=f"(out[6].y),
                              "=f"(out[7].x), "=f"(out[7].y)
                            : "r"(tt_output.addr +
                                  ((warpgroup::warpid() * 32) << 16) + col));
                        tensor_load_wait();
                        #pragma unroll
                        for (int pair = 0; pair < 8; ++pair) {
                            const bf16_2 packed = __float22bfloat162_rn(
                                __fmul2_rn(
                                    out[pair],
                                    float2{output_scale, output_scale}));
                            const int out_offset =
                                ((((b * g.seqlen + query_tile * C::Mb + row) *
                                   g.heads + h) * 128) +
                                 col + pair * 2);
                            *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int col = 0; col < C::Dvo; col += 16) {
                float2 out[8];
                asm volatile(
                    "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                    : "=f"(out[0].x), "=f"(out[0].y),
                      "=f"(out[1].x), "=f"(out[1].y),
                      "=f"(out[2].x), "=f"(out[2].y),
                      "=f"(out[3].x), "=f"(out[3].y),
                      "=f"(out[4].x), "=f"(out[4].y),
                      "=f"(out[5].x), "=f"(out[5].y),
                      "=f"(out[6].x), "=f"(out[6].y),
                      "=f"(out[7].x), "=f"(out[7].y)
                    : "r"(tt_output_half1.addr +
                          ((warpgroup::warpid() * 32) << 16) + col));
                tensor_load_wait();
                #pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const bf16_2 packed = __float22bfloat162_rn(
                        __fmul2_rn(
                            out[pair],
                            float2{output_scale, output_scale}));
                    const int out_offset =
                        ((((b * g.seqlen + query_tile * C::Mb + row) *
                           g.heads + h) * 128) +
                         C::Dvo + col + pair * 2);
                    *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
                }
            }
            if (g.disable_lse_store == 0) {
                float lse_value =
                    base_types::constants<float>::neg_infty();
                if (valid) {
                    constexpr float LN2 = 0.693147180559945f;
                    float log2_row_sum;
                    asm("lg2.approx.ftz.f32 %0, %1;"
                        : "=f"(log2_row_sum) : "f"(final_row_sum));
                    lse_value =
                        (row_max * scale_log2 + log2_row_sum) * LN2;
                }
                g.lse[((b * g.heads + h) * g.seqlen) +
                      query_tile * C::Mb + row] = lse_value;
            }
        }
        warpgroup::sync(warpgroup::groupid() + 1);
#if TK_FA4_DVO64_LEAN_TRACE_BUILD
        if (warpgroup::groupid() == 1 &&
            warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(8, trace_now());
        }
#endif
    } else if (warpgroup::groupid() == 1) {
        const int row = warpgroup::laneid();
        const bool valid = row_sum > 0.0f && isfinite(row_sum);
        const float output_scale = valid
            ? MXFP4_GEMM_SCALE_CORRECTION / row_sum : 0.0f;
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            float2 out[8];
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(out[0].x), "=f"(out[0].y),
                  "=f"(out[1].x), "=f"(out[1].y),
                  "=f"(out[2].x), "=f"(out[2].y),
                  "=f"(out[3].x), "=f"(out[3].y),
                  "=f"(out[4].x), "=f"(out[4].y),
                  "=f"(out[5].x), "=f"(out[5].y),
                  "=f"(out[6].x), "=f"(out[6].y),
                  "=f"(out[7].x), "=f"(out[7].y)
                : "r"(tt_output.addr +
                      ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                const bf16_2 packed = __float22bfloat162_rn(
                    __fmul2_rn(out[pair],
                               float2{output_scale, output_scale}));
                const int out_offset =
                    ((((b * g.seqlen + query_tile * C::Mb + row) *
                     g.heads + h) * 128) +
                     dvo_half * C::Dvo + col + pair * 2);
                *reinterpret_cast<bf16_2 *>(&g.o[out_offset]) = packed;
            }
        }
        if (g.disable_lse_store == 0 && score_owner) {
            float lse_value = base_types::constants<float>::neg_infty();
            if (valid) {
                constexpr float LN2 = 0.693147180559945f;
                float log2_row_sum;
                asm("lg2.approx.ftz.f32 %0, %1;"
                    : "=f"(log2_row_sum) : "f"(row_sum));
                lse_value =
                    (row_max * scale_log2 + log2_row_sum) * LN2;
            }
            g.lse[((b * g.heads + h) * g.seqlen) +
                  query_tile * C::Mb + row] = lse_value;
        }
        warpgroup::sync(2);
#if TK_FA4_DVO64_LEAN_TRACE_BUILD
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            trace_store(8, trace_now());
        }
#endif
    }

    // All roles retire before global warp 0 runs tensor_allocator's destructor.
    __syncthreads();
    if constexpr (PAIRED_CTA) {
        fp4pv_cluster_sync<C>();
    }
#if TK_FA4_DVO64_LEAN_TRACE_BUILD
    if (threadIdx.x == 0 && g.trace_enabled != 0) {
        trace_store(9, trace_now());
        __threadfence_system();
        trace_store(0, 1);
    }
#endif
}
#endif

#pragma once

#ifndef TK_FA4_RAW_FP4_THROUGHPUT_PROBE_BUILD
#define TK_FA4_RAW_FP4_THROUGHPUT_PROBE_BUILD 0
#endif

#ifndef TK_FA4_RAW_FP4_MMA_THROUGHPUT_PRAGMA
#define TK_FA4_RAW_FP4_MMA_THROUGHPUT_PRAGMA 0
#endif

#if TK_FA4_RAW_FP4_THROUGHPUT_PROBE_BUILD

enum raw_fp4_throughput_mode : int {
    RAW_FP4_THROUGHPUT = 0,
    SCALED_MXFP4_THROUGHPUT = 1,
    BF16_THROUGHPUT = 2,
    SPARSE_MXFP4_THROUGHPUT = 3,
    SPARSE_NVFP4_THROUGHPUT = 4,
    SPLIT_SPARSE_NVFP4_THROUGHPUT = 5,
    DENSE_NVFP4_THROUGHPUT = 6,
    DEDICATED_MXFP4_THROUGHPUT = 7,
    DUAL_CHAIN_MXFP4_THROUGHPUT = 8,
    TRIPLE_CHAIN_MXFP4_THROUGHPUT = 9,
    DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT = 10,
    DUAL_CHAIN_K192_K96_MXFP4_THROUGHPUT = 11,
    DUAL_CHAIN_NVFP4_THROUGHPUT = 12,
    DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT = 13,
    DUAL_CHAIN_K192_K96_NVFP4_THROUGHPUT = 14,
    PARALLEL_DUAL_CHAIN_MXFP4_THROUGHPUT = 15,
    PARALLEL_TRIPLE_CHAIN_MXFP4_THROUGHPUT = 16,
    FOUR_CTA_N64_MXFP4_THROUGHPUT = 17,
    FOUR_CTA_N96_MXFP4_THROUGHPUT = 18,
    EIGHT_CTA_N32_MXFP4_THROUGHPUT = 19,
    ONE_CTA_N256_MXFP4_THROUGHPUT = 20,
    ONE_CTA_ATTENTION_N96_K96_MXFP4_THROUGHPUT = 21,
    TWO_CTA_ATTENTION_N96_K96_MXFP4_THROUGHPUT = 22,
    ONE_CTA_ATTENTION_N128_SPLIT_QK_MXFP4_THROUGHPUT = 23,
    TWO_CTA_ATTENTION_N128_SPLIT_QK_MXFP4_THROUGHPUT = 24,
    NVFP4_N128_SPLIT_QK_CORRECTNESS = 25,
    NVFP4_N128_SPLIT_QK_ALIASED_SCALE_CORRECTNESS = 26,
    ONE_CTA_ATTENTION_N96_NVQK_K96_MXPV_THROUGHPUT = 27,
    TWO_CTA_ATTENTION_N96_NVQK_K96_MXPV_THROUGHPUT = 28,
    ONE_CTA_ATTENTION_N96_PRODUCTION_SCALE_COPY = 29,
    TWO_CTA_ATTENTION_N96_PRODUCTION_SCALE_COPY = 30,
    TRIPLE_CHAIN_K192_K96_MXFP4_THROUGHPUT = 31,
    TRIPLE_CHAIN_K192_K96_NVFP4_THROUGHPUT = 32,
    DUAL_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT = 33,
    TRIPLE_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT = 34,
    DUAL_SCORE_COLLECTOR_MXFP4_THROUGHPUT = 35,
    DUAL_SCORE_COLLECTOR_NVFP4_THROUGHPUT = 36,
    ONE_CTA_ATTENTION_N192_ULTRA_NVQK_MXPV_THROUGHPUT = 37,
    ONE_CTA_ATTENTION_N256_ULTRA_NVQK_MXPV_THROUGHPUT = 38,
    ONE_CTA_ATTENTION_N256_DUAL_QUERY_PREFIX_SWAP_THROUGHPUT = 39,
};

__device__ __forceinline__ uint64_t raw_fp4_throughput_globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

template <int _MODE>
struct config_raw_fp4_throughput_probe {
    static constexpr int MODE = _MODE;
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dvo = 128;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_THREADS = 128;
    static constexpr bool DUAL_CHAIN =
        MODE == DUAL_CHAIN_MXFP4_THROUGHPUT ||
        MODE == DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT ||
        MODE == DUAL_CHAIN_K192_K96_MXFP4_THROUGHPUT ||
        MODE == DUAL_CHAIN_NVFP4_THROUGHPUT ||
        MODE == DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT ||
        MODE == DUAL_CHAIN_K192_K96_NVFP4_THROUGHPUT ||
        MODE == DUAL_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT ||
        MODE == DUAL_SCORE_COLLECTOR_MXFP4_THROUGHPUT ||
        MODE == DUAL_SCORE_COLLECTOR_NVFP4_THROUGHPUT ||
        MODE == PARALLEL_DUAL_CHAIN_MXFP4_THROUGHPUT;
    static constexpr bool TRIPLE_CHAIN =
        MODE == TRIPLE_CHAIN_MXFP4_THROUGHPUT ||
        MODE == PARALLEL_TRIPLE_CHAIN_MXFP4_THROUGHPUT ||
        MODE == TRIPLE_CHAIN_K192_K96_MXFP4_THROUGHPUT ||
        MODE == TRIPLE_CHAIN_K192_K96_NVFP4_THROUGHPUT ||
        MODE == TRIPLE_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT;
    static constexpr bool PARALLEL_CHAIN =
        MODE == PARALLEL_DUAL_CHAIN_MXFP4_THROUGHPUT ||
        MODE == PARALLEL_TRIPLE_CHAIN_MXFP4_THROUGHPUT;
    static constexpr bool MULTI_CHAIN = DUAL_CHAIN || TRIPLE_CHAIN;
    // Multi-chain routes model the full attention kernel: one CTA owns all
    // 512 columns and alternates independent N128 accumulators. Other routes
    // use two resident 256-column CTAs with one accumulator each.
    static constexpr int TMEM_RESIDENT_BLOCKS = MULTI_CHAIN ? 1 : 2;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        (MULTI_CHAIN ? 144 : 112) * 1024;
};

template <typename C>
struct globals_raw_fp4_throughput_probe {
    const uint8_t *a_fp4;
    const uint8_t *b_fp4;
    const bf16 *a_bf16;
    const bf16 *b_bf16;
    uint64_t *cycles;
    uint64_t *start_globaltimer;
    uint64_t *end_globaltimer;
    int32_t *smids;
    int iterations;
    int blocks;

    __host__ __inline__ dim3 grid() const { return dim3(blocks); }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() const {
        return C::DYNAMIC_SHARED_MEMORY;
    }
};

template <typename Tile>
__device__ inline void load_throughput_dense_fp4_tile(
    Tile &tile,
    const uint8_t *source
) {
    static_assert(Tile::rows >= 32 && Tile::rows <= 256
                      && Tile::rows % 32 == 0 && Tile::cols == 64,
                  "dense K128 FP4 tile must have 32-step rows up to 256");
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * STORES_PER_ROW;
         linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            source + (row % 128) * Tile::cols + col);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(Tile::idx(base, {row, col})),
            "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename Tile>
__device__ inline void load_throughput_raw_fp4_padded_tile(
    Tile &tile,
    const uint8_t *source
) {
    fp4pv_shape_zero_tile(tile);
    __syncthreads();
    constexpr int GROUP_LOGICAL_K = 16;
    constexpr int GROUP_DATA_BYTES = GROUP_LOGICAL_K / 2;
    constexpr int GROUP_SHARED_BYTES = 16;
    constexpr int GROUPS_PER_ROW = C::Nb / GROUP_LOGICAL_K;
    static_assert(Tile::cols == GROUPS_PER_ROW * GROUP_SHARED_BYTES);
    uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * GROUPS_PER_ROW;
         linear += blockDim.x) {
        const int row = linear / GROUPS_PER_ROW;
        const int group = linear - row * GROUPS_PER_ROW;
        const uint2 value = *reinterpret_cast<const uint2 *>(
            source + row * 64 + group * GROUP_DATA_BYTES);
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::
            "r"(Tile::idx(base, {row, group * GROUP_SHARED_BYTES})),
            "r"(value.x), "r"(value.y));
    }
}

template <typename Tile>
__device__ inline void load_throughput_sparse_fp4_tile(
    Tile &tile,
    const uint8_t *source
) {
    static_assert(Tile::rows == 128 && Tile::cols == 32,
                  "sparse K128 FP4 tile must be 128x32 packed bytes");
    uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int row = threadIdx.x; row < Tile::rows; row += blockDim.x) {
        #pragma unroll
        for (int group = 0; group < 16; ++group) {
            const uint16_t packed =
                static_cast<uint16_t>(source[row * 64 + group * 4]) |
                (static_cast<uint16_t>(
                     source[row * 64 + group * 4 + 1]) << 8);
            asm volatile("st.shared.b16 [%0], %1;" ::
                "r"(Tile::idx(base, {row, group * 2})),
                "h"(packed));
        }
    }
}

template <typename Tile>
__device__ inline void load_throughput_bf16_tile(
    Tile &tile,
    const bf16 *source
) {
    static_assert(Tile::rows == 128 && Tile::cols == 128,
                  "BF16 throughput tile must be 128x128");
    constexpr int ELEMENTS_PER_STORE = 8;
    constexpr int STORES_PER_ROW = Tile::cols / ELEMENTS_PER_STORE;
    uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * STORES_PER_ROW;
         linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * ELEMENTS_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            source + row * Tile::cols + col);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(Tile::idx(base, {row, col})),
            "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <int ACC, typename Output, typename ADesc, typename BDesc>
__device__ __forceinline__ void issue_raw_k128_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc
) {
    using fp4_t = fp4e2m1_2;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<float, fp4_t,
                                                 128, 128,
                                                 false, false, false>();
    detail::tcgen05::st_st<fp4_t, ACC, 1>(
        output.addr, a_desc.chunk_descriptor(0), b_desc.chunk_descriptor(0), idesc);
    #pragma unroll
    for (int chunk = 1; chunk < 4; ++chunk) {
        detail::tcgen05::st_st<fp4_t, 1, 1>(
            output.addr,
            a_desc.chunk_descriptor(chunk),
            b_desc.chunk_descriptor(chunk),
            idesc);
    }
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_scaled_k128_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc0 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>();
    constexpr uint32_t idesc2 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 2>();
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 32>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc0);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 32>(
        output.addr,
        a_desc.chunk_descriptor(1),
        b_desc.chunk_descriptor(1),
        a_scale.addr,
        b_scale.addr,
        idesc2);
}

template <int N, int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_scaled_n_k128_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc0 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, N, false, 0>();
    constexpr uint32_t idesc2 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, N, false, 2>();
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 32>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc0);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 32>(
        output.addr,
        a_desc.chunk_descriptor(1),
        b_desc.chunk_descriptor(1),
        a_scale.addr,
        b_scale.addr,
        idesc2);
}

template <typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_scaled_k128_chain(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale,
    int iterations,
    semaphore &done
) {
    tensor_before_thread_sync();
    issue_scaled_k128_iteration<0>(
        output, a_desc, b_desc, a_scale, b_scale);
    for (int iter = 1; iter < iterations; ++iter) {
        issue_scaled_k128_iteration<1>(
            output, a_desc, b_desc, a_scale, b_scale);
    }
    detail::tcgen05::commit<1>(done);
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_scaled_k192_k64_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc0 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>();
    constexpr uint32_t idesc2 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 2>();
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 32>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc0);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 32>(
        output.addr,
        a_desc.chunk_descriptor(1),
        b_desc.chunk_descriptor(1),
        a_scale.addr,
        b_scale.addr,
        idesc2);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 32>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc0);
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_scaled_k192_k96_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const auto make_k96_desc = [&](uint64_t current, uint64_t next) {
        // SM103 K96 consumes 32B from the current 128B-swizzled chunk and
        // 16B from the next chunk. Bit 52 changes the leading-byte-offset
        // field from a relative stride to the absolute address of that chunk.
        return (current & ~lbo_field_mask) |
               ((next & 0x3fffull) << 16) |
               absolute_lbo_mode;
    };
    const uint64_t a_desc_k96 = make_k96_desc(
        a_desc.chunk_descriptor(0), a_desc.chunk_descriptor(1));
    const uint64_t b_desc_k96 = make_k96_desc(
        b_desc.chunk_descriptor(0), b_desc.chunk_descriptor(1));
    constexpr uint32_t idesc_k96 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>() |
        (1u << 31);
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 32>(
        output.addr,
        a_desc_k96,
        b_desc_k96,
        a_scale.addr,
        b_scale.addr,
        idesc_k96);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 32>(
        output.addr,
        a_desc_k96,
        b_desc_k96,
        a_scale.addr,
        b_scale.addr,
        idesc_k96);
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_sm103_mxfp4_k192_k96_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const auto make_k96_desc = [&](uint64_t current, uint64_t next) {
        return (current & ~lbo_field_mask) |
               ((next & 0x3fffull) << 16) |
               absolute_lbo_mode;
    };
    const uint64_t a_desc_k96 = make_k96_desc(
        a_desc.chunk_descriptor(0), a_desc.chunk_descriptor(1));
    const uint64_t b_desc_k96 = make_k96_desc(
        b_desc.chunk_descriptor(0), b_desc.chunk_descriptor(1));
    constexpr uint32_t idesc_k96 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>() |
        (1u << 31);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "l"(a_desc_k96), "l"(b_desc_k96),
        "n"(idesc_k96), "r"(a_scale.addr), "r"(b_scale.addr),
        "n"(ACC));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, 1;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "l"(a_desc_k96), "l"(b_desc_k96),
        "n"(idesc_k96), "r"(a_scale.addr), "r"(b_scale.addr));
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_dedicated_mxfp4_k128_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc0 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>();
    constexpr uint32_t idesc2 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 2>();
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr),
        "l"(a_desc.chunk_descriptor(0)),
        "l"(b_desc.chunk_descriptor(0)),
        "n"(idesc0),
        "r"(a_scale.addr),
        "r"(b_scale.addr),
        "n"(ACC));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, 1;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr),
        "l"(a_desc.chunk_descriptor(1)),
        "l"(b_desc.chunk_descriptor(1)),
        "n"(idesc2),
        "r"(a_scale.addr),
        "r"(b_scale.addr));
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_collector_mxfp4_k128_pair(
    Output &output0,
    Output &output1,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc0 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>();
    constexpr uint32_t idesc2 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 2>();
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32."
        "collector::a::fill "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output0.addr), "l"(a_desc.chunk_descriptor(0)),
        "l"(b_desc.chunk_descriptor(0)), "n"(idesc0),
        "r"(a_scale.addr), "r"(b_scale.addr), "n"(ACC));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32."
        "collector::a::lastuse "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output1.addr), "l"(a_desc.chunk_descriptor(0)),
        "l"(b_desc.chunk_descriptor(0)), "n"(idesc0),
        "r"(a_scale.addr), "r"(b_scale.addr), "n"(ACC));
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32."
        "collector::a::fill "
        "[%0], %1, %2, %3, [%4], [%5], 1;\n" ::
        "r"(output0.addr), "l"(a_desc.chunk_descriptor(1)),
        "l"(b_desc.chunk_descriptor(1)), "n"(idesc2),
        "r"(a_scale.addr), "r"(b_scale.addr));
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32."
        "collector::a::lastuse "
        "[%0], %1, %2, %3, [%4], [%5], 1;\n" ::
        "r"(output1.addr), "l"(a_desc.chunk_descriptor(1)),
        "l"(b_desc.chunk_descriptor(1)), "n"(idesc2),
        "r"(a_scale.addr), "r"(b_scale.addr));
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_collector_nvfp4_k128_pair(
    Output &output0,
    Output &output1,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc0 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>();
    constexpr uint32_t idesc2 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 2>();
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16."
        "collector::a::fill "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output0.addr), "l"(a_desc.chunk_descriptor(0)),
        "l"(b_desc.chunk_descriptor(0)), "n"(idesc0),
        "r"(a_scale.addr), "r"(b_scale.addr), "n"(ACC));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16."
        "collector::a::lastuse "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output1.addr), "l"(a_desc.chunk_descriptor(0)),
        "l"(b_desc.chunk_descriptor(0)), "n"(idesc0),
        "r"(a_scale.addr), "r"(b_scale.addr), "n"(ACC));
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16."
        "collector::a::fill "
        "[%0], %1, %2, %3, [%4], [%5], 1;\n" ::
        "r"(output0.addr), "l"(a_desc.chunk_descriptor(1)),
        "l"(b_desc.chunk_descriptor(1)), "n"(idesc2),
        "r"(a_scale.addr), "r"(b_scale.addr));
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16."
        "collector::a::lastuse "
        "[%0], %1, %2, %3, [%4], [%5], 1;\n" ::
        "r"(output1.addr), "l"(a_desc.chunk_descriptor(1)),
        "l"(b_desc.chunk_descriptor(1)), "n"(idesc2),
        "r"(a_scale.addr), "r"(b_scale.addr));
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_nvfp4_k128_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>();
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 16>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 16>(
        output.addr,
        a_desc.chunk_descriptor(1),
        b_desc.chunk_descriptor(1),
        a_scale.addr,
        b_scale.addr,
        idesc);
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_nvfp4_k192_k64_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>();
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 16>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 16>(
        output.addr,
        a_desc.chunk_descriptor(1),
        b_desc.chunk_descriptor(1),
        a_scale.addr,
        b_scale.addr,
        idesc);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 16>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc);
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_nvfp4_k192_k96_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc_k96 =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>() |
        (1u << 31);
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 16>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc_k96);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 16>(
        output.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        a_scale.addr,
        b_scale.addr,
        idesc_k96);
}

template <int ACC, typename Output, typename ADesc, typename BDesc>
__device__ __forceinline__ void issue_bf16_k128_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc
) {
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<float, bf16,
                                                 128, 128,
                                                 false, false, false>();
    detail::tcgen05::st_st<bf16, ACC, 1>(
        output.addr, a_desc.chunk_descriptor(0), b_desc.chunk_descriptor(0), idesc);
    #pragma unroll
    for (int chunk = 1; chunk < 8; ++chunk) {
        detail::tcgen05::st_st<bf16, 1, 1>(
            output.addr,
            a_desc.chunk_descriptor(chunk),
            b_desc.chunk_descriptor(chunk),
            idesc);
    }
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_sparse_mxfp4_k128_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    uint32_t metadata_addr,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>() |
        (1u << 2);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %7;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 "
        "[%0], %1, %2, [%3], %4, [%5], [%6], p;\n\t"
        "}\n" ::
        "r"(output.addr),
        "l"(a_desc.chunk_descriptor(0)),
        "l"(b_desc.chunk_descriptor(0)),
        "r"(metadata_addr),
        "r"(idesc),
        "r"(a_scale.addr),
        "r"(b_scale.addr),
        "n"(ACC));
}

template <int ACC, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void issue_sparse_nvfp4_k128_iteration(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    uint32_t metadata_addr,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<float, fp4_t, scale_t,
                                                 128, 128, false, 0>() |
        (1u << 2);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %7;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, [%3], %4, [%5], [%6], p;\n\t"
        "}\n" ::
        "r"(output.addr),
        "l"(a_desc.chunk_descriptor(0)),
        "l"(b_desc.chunk_descriptor(0)),
        "r"(metadata_addr),
        "r"(idesc),
        "r"(a_scale.addr),
        "r"(b_scale.addr),
        "n"(ACC));
}

__device__ __forceinline__ void fill_throughput_tmem_128x16(
    uint32_t tmem_addr,
    uint32_t packed_value
) {
    #pragma unroll
    for (int row = 0; row < 128; row += 32) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32 "
            "[%0], {%1, %1, %1, %1};" ::
            "r"(tmem_addr + (static_cast<uint32_t>(row) << 16)),
            "r"(packed_value));
    }
}

// The general TK allocator intentionally models one or two resident CTAs.
// These probes use smaller power-of-two allocations to test independent CTA
// issue streams without changing the allocator used by production kernels.
template <int COLS>
struct raw_fp4_shape_tmem_allocator {
    static_assert(
        COLS == 64 || COLS == 128 || COLS == 256 || COLS == 512);
    static constexpr int cols = COLS;
    uint32_t addr;

    __device__ __forceinline__ raw_fp4_shape_tmem_allocator() {
        __shared__ uint32_t shared_addr;
        if (warpid() == 0) {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
                "[%0], %1;\n" ::
                "l"(reinterpret_cast<uint64_t>(&shared_addr)), "n"(COLS));
            asm volatile(
                "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
        }
        asm volatile("tcgen05.fence::before_thread_sync;\n");
        asm volatile("bar.sync 0;\n");
        asm volatile("tcgen05.fence::after_thread_sync;\n");
        addr = shared_addr;
    }

    template <typename Tile>
    __device__ __forceinline__ Tile allocate(int col_offset) const {
        return Tile(addr + col_offset);
    }

    __device__ __forceinline__ ~raw_fp4_shape_tmem_allocator() {
        if (warpid() == 0) {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::
                "r"(addr), "n"(COLS));
        }
    }
};

template <int N>
struct config_raw_fp4_shape_throughput_probe {
    static_assert(N == 32 || N == 64 || N == 96 || N == 256);
    static constexpr int OUTPUT_N = N;
    static constexpr int TMEM_COLS = N == 32 ? 64 : N <= 96 ? 128 : 512;
    static constexpr int RESIDENT_CTAS = 512 / TMEM_COLS;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 128;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        (RESIDENT_CTAS == 8 ? 24 : RESIDENT_CTAS == 4 ? 48 : 112) * 1024;
};

template <int N>
__launch_bounds__(128, config_raw_fp4_shape_throughput_probe<N>::RESIDENT_CTAS)
__global__ void kernel_raw_fp4_shape_throughput_probe(
    const __grid_constant__
        globals_raw_fp4_throughput_probe<
            config_raw_fp4_shape_throughput_probe<N>> g
) {
    using C = config_raw_fp4_shape_throughput_probe<N>;
    using output_t = tt<float, 128, N>;
    using a_tile_t = st_fp4e2m1_2<128, 64>;
    using b_tile_t = st_fp4e2m1_2<N, 64>;
    using scale_t = full_tt_fp8e8m0<16>;
    using tmem_allocator_t = raw_fp4_shape_tmem_allocator<C::TMEM_COLS>;
    constexpr int A_SCALE_BASE = N;
    constexpr int B_SCALE_BASE = N + 16;
    static_assert(B_SCALE_BASE + 16 <= tmem_allocator_t::cols);

    tmem_allocator_t tm_alloc{};
    output_t output = tm_alloc.template allocate<output_t>(0);
    scale_t a_scale = tm_alloc.template allocate<scale_t>(A_SCALE_BASE);
    scale_t b_scale = tm_alloc.template allocate<scale_t>(B_SCALE_BASE);
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    a_tile_t &a = allocator.allocate<a_tile_t>();
    b_tile_t &b = allocator.allocate<b_tile_t>();
    __shared__ semaphore done;
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    load_throughput_dense_fp4_tile(a, g.a_fp4);
    load_throughput_dense_fp4_tile(b, g.b_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    if (warpgroup::warpid() == 0) {
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            a_scale, 0x7f7f7f7fu);
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            b_scale, 0x7f7f7f7fu);
        fp4pv_tmem_store_wait();
    }
    __syncthreads();

    st_descriptor<a_tile_t, transpose::N> a_desc(a);
    st_descriptor<b_tile_t, transpose::N> b_desc(b);
    const bool issue_warp = warpgroup::warpid() == 0;
    const bool leader = issue_warp && warp::elect_leader();
    uint64_t start_cycle = 0;
    if (leader) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
        issue_scaled_n_k128_iteration<N, 0>(
            output, a_desc, b_desc, a_scale, b_scale);
        for (int iter = 1; iter < g.iterations; ++iter) {
            issue_scaled_n_k128_iteration<N, 1>(
                output, a_desc, b_desc, a_scale, b_scale);
        }
        detail::tcgen05::commit<1>(done);
    }
    if (issue_warp) {
        wait(done, 0);
        tensor_after_thread_sync();
    }
    if (leader) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
    __syncthreads();
}

template <int RESIDENT_CTAS>
struct config_raw_fp4_attention_n96_k96_probe {
    static_assert(RESIDENT_CTAS == 1 || RESIDENT_CTAS == 2);
    static constexpr int TMEM_COLS = 512 / RESIDENT_CTAS;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 128;
    static constexpr int DYNAMIC_SHARED_MEMORY = 32 * 1024;
};

template <typename ST>
__device__ __forceinline__ uint32_t
raw_fp4_scale_smem_descriptor_low(const ST &tile) {
    static_assert(!ST::swizzle);
    const uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&tile.data[0]));
    return ((smem_addr & 0x3ffffu) >> 4) | (8u << 16);
}

template <typename QScale, typename KScale>
__device__ __forceinline__ void
raw_fp4_preload_nvfp4_compact_scales(
    const QScale &q_scale_smem,
    const KScale &k_scale_smem,
    uint32_t scale_base
) {
    const uint32_t q_scale_desc_low =
        raw_fp4_scale_smem_descriptor_low(q_scale_smem);
    const uint32_t k_scale_desc_low =
        raw_fp4_scale_smem_descriptor_low(k_scale_smem);
    constexpr uint32_t scale_desc_hi =
        8u | (1u << (46 - 32));
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
           "n"(scale_desc_hi)
        : "memory");
}

template <typename TT, typename ST>
__device__ __forceinline__ void raw_fp4_load_mxfp4_scale_ranklocal(
    TT &dst,
    const ST &src
) {
    auto &source =
        *reinterpret_cast<const st_fp8e8m0<32, 16, false> *>(
            &src.data[0]);
    fp4pv_load_mxnv_scale_async_ranklocal(dst, source);
}

template <typename ScaleScratch>
__device__ __forceinline__ void raw_fp4_fill_scale_smem(
    ScaleScratch &scale,
    uint32_t packed
) {
    auto *words = reinterpret_cast<uint32_t *>(&scale.data[0]);
    constexpr int word_count =
        sizeof(ScaleScratch) / sizeof(uint32_t);
    for (int word = int(threadIdx.x);
         word < word_count;
         word += int(blockDim.x)) {
        words[word] = packed;
    }
}

template <int ACC, typename Output, typename VDesc,
          typename PScale, typename VScale>
__device__ __forceinline__ void issue_tmem_smem_k96_iteration(
    Output &output,
    uint32_t p_addr,
    VDesc &v_desc,
    const PScale &p_scale,
    const VScale &v_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const uint64_t current = v_desc.chunk_descriptor(0);
    const uint64_t next = v_desc.chunk_descriptor(1);
    const uint64_t v_desc_k96 =
        (current & ~lbo_field_mask) |
        ((next & 0x3fffull) << 16) |
        absolute_lbo_mode;
    constexpr uint32_t idesc_k96 =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 128, false, 0>() |
        (1u << 31);
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 "
        "[%0], [%1], %2, %3, [%4], [%5], p;}\n" ::
        "r"(output.addr),
        "r"(p_addr),
        "l"(v_desc_k96),
        "n"(idesc_k96),
        "r"(p_scale.addr),
        "r"(v_scale.addr),
        "n"(ACC) : "memory");
}

template <int ACC, typename Output, typename VDesc,
          typename PScale, typename VScale>
__device__ __forceinline__ void issue_tmem_smem_k64_iteration(
    Output &output,
    uint32_t p_addr,
    VDesc &v_desc,
    const PScale &p_scale,
    const VScale &v_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 128, false, 0>();
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 "
        "[%0], [%1], %2, %3, [%4], [%5], p;}\n" ::
        "r"(output.addr),
        "r"(p_addr),
        "l"(v_desc.chunk_descriptor(0)),
        "n"(idesc),
        "r"(p_scale.addr),
        "r"(v_scale.addr),
        "n"(ACC) : "memory");
}

template <int ACC, typename Score, typename QDesc, typename KDesc,
          typename QScale, typename KScale>
__device__ __forceinline__ void issue_nvfp4_n96_qk_iteration(
    Score &score,
    QDesc &q_desc,
    KDesc &k_desc,
    const QScale &q_scale,
    const KScale &k_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 96, false, 0>();
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 16>(
        score.addr,
        q_desc.chunk_descriptor(0),
        k_desc.chunk_descriptor(0),
        q_scale.addr,
        k_scale.addr,
        idesc);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 16>(
        score.addr,
        q_desc.chunk_descriptor(1),
        k_desc.chunk_descriptor(1),
        q_scale.addr,
        k_scale.addr,
        idesc);
}

template <int N, int ACC, typename Score, typename QDesc, typename KDesc,
          typename QScale, typename KScale>
__device__ __forceinline__ void issue_nvfp4_wide_qk_iteration(
    Score &score,
    QDesc &q_desc,
    KDesc &k_desc,
    const QScale &q_scale,
    const KScale &k_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, N, false, 0>();
    detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 16>(
        score.addr,
        q_desc.chunk_descriptor(0),
        k_desc.chunk_descriptor(0),
        q_scale.addr,
        k_scale.addr,
        idesc);
    detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 16>(
        score.addr,
        q_desc.chunk_descriptor(1),
        k_desc.chunk_descriptor(1),
        q_scale.addr,
        k_scale.addr,
        idesc);
}

// Models the proposed D128 production geometry. The N96 score/P operand and
// D128 output occupy [0, 224); the two scale pages live in the disjoint tail
// [224, 256). Comparing one and two resident CTAs isolates the benefit of the
// smaller TMEM lifetime without aliasing scale operands into a TCGEN result.
template <
    int RESIDENT_CTAS,
    bool NVFP4_QK = false,
    bool PRODUCTION_SCALE_COPY = false>
__launch_bounds__(128, RESIDENT_CTAS)
__global__ void kernel_raw_fp4_attention_n96_k96_probe(
    const __grid_constant__ globals_raw_fp4_throughput_probe<
        config_raw_fp4_attention_n96_k96_probe<RESIDENT_CTAS>> g
) {
    using C = config_raw_fp4_attention_n96_k96_probe<RESIDENT_CTAS>;
    using Score = tt<float, 128, 96>;
    using Output = tt<float, 128, 128>;
    using Q = st_fp4e2m1_2<128, 64>;
    using K = st_fp4e2m1_2<96, 64>;
    using Scale = full_tt_fp8e8m0<16>;
    using ScaleScratch = st_hf<1, 256, false>;
    using TmemAllocator = raw_fp4_shape_tmem_allocator<C::TMEM_COLS>;
    constexpr int SCORE_COL = 0;
    constexpr int OUTPUT_COL = 96;
    constexpr int P_SCALE_COL = 224;
    constexpr int V_SCALE_COL = 240;
    constexpr int Q_SCALE_COL = 224;
    constexpr int K_SCALE_COL = 232;
    static_assert(OUTPUT_COL + Output::cols == P_SCALE_COL);
    static_assert(V_SCALE_COL + Scale::cols <= C::TMEM_COLS);

    TmemAllocator tm_alloc{};
    Score score = tm_alloc.template allocate<Score>(SCORE_COL);
    Output output = tm_alloc.template allocate<Output>(OUTPUT_COL);
    Scale p_scale = tm_alloc.template allocate<Scale>(P_SCALE_COL);
    Scale v_scale = tm_alloc.template allocate<Scale>(V_SCALE_COL);
    Scale q_scale = tm_alloc.template allocate<Scale>(Q_SCALE_COL);
    Scale k_scale = tm_alloc.template allocate<Scale>(K_SCALE_COL);

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    Q &q_and_v = allocator.allocate<Q>();
    K &k = allocator.allocate<K>();
    ScaleScratch &q_scale_smem = allocator.allocate<ScaleScratch>();
    ScaleScratch &k_scale_smem = allocator.allocate<ScaleScratch>();
    ScaleScratch &p_scale_smem = allocator.allocate<ScaleScratch>();
    ScaleScratch &v_scale_smem = allocator.allocate<ScaleScratch>();
    __shared__ semaphore qk_done;
    __shared__ semaphore pv_done;
    if (threadIdx.x == 0) {
        init_semaphore(qk_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
    }
    load_throughput_dense_fp4_tile(q_and_v, g.a_fp4);
    load_throughput_dense_fp4_tile(k, g.b_fp4);
    if constexpr (PRODUCTION_SCALE_COPY) {
        raw_fp4_fill_scale_smem(q_scale_smem, 0x38383838u);
        raw_fp4_fill_scale_smem(k_scale_smem, 0x38383838u);
        raw_fp4_fill_scale_smem(p_scale_smem, 0x7f7f7f7fu);
        raw_fp4_fill_scale_smem(v_scale_smem, 0x7f7f7f7fu);
    }
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();

    st_descriptor<Q, transpose::N> q_desc(q_and_v);
    st_descriptor<K, transpose::N> k_desc(k);
    st_descriptor<Q, transpose::N> v_desc(q_and_v);
    const bool issue_warp = warpgroup::warpid() == 0;
    const bool leader = issue_warp && warp::elect_leader();
    if constexpr (!PRODUCTION_SCALE_COPY) {
        if (issue_warp) {
            if constexpr (NVFP4_QK) {
                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                    q_scale, 0x38383838u);
                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                    k_scale, 0x38383838u);
            } else {
                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                    p_scale, 0x7f7f7f7fu);
                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                    v_scale, 0x7f7f7f7fu);
            }
            fp4pv_tmem_store_wait();
        }
    }
    __syncthreads();
    uint64_t start_cycle = 0;
    int phase = 0;
    if (leader) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
    }

    for (int iter = 0; iter < g.iterations; ++iter) {
        if (leader) {
            if constexpr (NVFP4_QK) {
                if constexpr (PRODUCTION_SCALE_COPY) {
                    raw_fp4_preload_nvfp4_compact_scales(
                        q_scale_smem,
                        k_scale_smem,
                        q_scale.addr);
                }
                issue_nvfp4_n96_qk_iteration<0>(
                    score, q_desc, k_desc, q_scale, k_scale);
            } else {
                issue_scaled_n_k128_iteration<96, 0>(
                    score, q_desc, k_desc, p_scale, v_scale);
            }
            detail::tcgen05::commit<1>(qk_done);
        }
        if (issue_warp) {
            wait(qk_done, phase);
            tensor_after_thread_sync();
            // Packed E2M1 0x2 is 1.0. Sixteen physical TMEM columns cover
            // the twelve columns consumed by the K96 P operand.
            fill_throughput_tmem_128x16(score.addr, 0x22222222u);
            if constexpr (!PRODUCTION_SCALE_COPY) {
                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                    p_scale, 0x7f7f7f7fu);
                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                    v_scale, 0x7f7f7f7fu);
                fp4pv_tmem_store_wait();
            }
            tensor_before_thread_sync();
        }
        if (leader) {
            if constexpr (PRODUCTION_SCALE_COPY) {
                raw_fp4_load_mxfp4_scale_ranklocal(
                    p_scale, p_scale_smem);
                raw_fp4_load_mxfp4_scale_ranklocal(
                    v_scale, v_scale_smem);
            }
            if (iter == 0) {
                issue_tmem_smem_k96_iteration<0>(
                    output, score.addr, v_desc, p_scale, v_scale);
            } else {
                issue_tmem_smem_k96_iteration<1>(
                    output, score.addr, v_desc, p_scale, v_scale);
            }
            detail::tcgen05::commit<1>(pv_done);
        }
        if (issue_warp) {
            wait(pv_done, phase);
            tensor_after_thread_sync();
            if constexpr (NVFP4_QK && !PRODUCTION_SCALE_COPY) {
                // Restore the folded block16 Q/K scales after the K96 PV
                // scale pages have retired and before the next QK issue.
                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                    q_scale, 0x38383838u);
                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                    k_scale, 0x38383838u);
                fp4pv_tmem_store_wait();
            }
            tensor_before_thread_sync();
        }
        phase ^= 1;
    }

    if (leader) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
    __syncthreads();
}

template <int ACC, typename Output, typename VDesc,
          typename PScale, typename VScale>
__device__ __forceinline__ void issue_tmem_smem_k128_iteration(
    Output &output,
    uint32_t p_addr,
    VDesc &v_desc,
    const PScale &p_scale,
    const VScale &v_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc0 =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 128, false, 0>();
    constexpr uint32_t idesc2 =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 128, false, 2>();
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::2X [%0], [%1], %2, %3, [%4], [%5], p;}\n" ::
        "r"(output.addr),
        "r"(p_addr),
        "l"(v_desc.chunk_descriptor(0)),
        "n"(idesc0),
        "r"(p_scale.addr),
        "r"(v_scale.addr),
        "n"(ACC) : "memory");
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.eq.u32 p, 1, 1;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale."
        "scale_vec::2X [%0], [%1], %2, %3, [%4], [%5], p;}\n" ::
        "r"(output.addr),
        "r"(p_addr + 8),
        "l"(v_desc.chunk_descriptor(1)),
        "n"(idesc2),
        "r"(p_scale.addr),
        "r"(v_scale.addr) : "memory");
}

// Keeps the production N128 key tile while splitting QK into N96 + N32.
// Head-QK scales occupy the future N32 score columns. Once the N96 scores
// are conceptually retired, tail-QK and PV scales move to columns 16 and 32.
// This is the 256-column layout with the least duplicated arithmetic.
template <int RESIDENT_CTAS>
__launch_bounds__(128, RESIDENT_CTAS)
__global__ void kernel_raw_fp4_attention_n128_split_qk_probe(
    const __grid_constant__ globals_raw_fp4_throughput_probe<
        config_raw_fp4_attention_n96_k96_probe<RESIDENT_CTAS>> g
) {
    using C = config_raw_fp4_attention_n96_k96_probe<RESIDENT_CTAS>;
    using ScoreHead = tt<float, 128, 96>;
    using ScoreTail = tt<float, 128, 32>;
    using Output = tt<float, 128, 128>;
    using QV = st_fp4e2m1_2<128, 64>;
    using KHead = st_fp4e2m1_2<96, 64>;
    using KTail = st_fp4e2m1_2<32, 64>;
    using Scale = full_tt_fp8e8m0<16>;
    using TmemAllocator = raw_fp4_shape_tmem_allocator<C::TMEM_COLS>;

    TmemAllocator tm_alloc{};
    ScoreHead score_head = tm_alloc.template allocate<ScoreHead>(0);
    ScoreTail score_tail = tm_alloc.template allocate<ScoreTail>(96);
    Output output = tm_alloc.template allocate<Output>(128);
    Scale head_q_scale = tm_alloc.template allocate<Scale>(96);
    Scale head_k_scale = tm_alloc.template allocate<Scale>(112);
    Scale tail_p_scale = tm_alloc.template allocate<Scale>(16);
    Scale tail_v_scale = tm_alloc.template allocate<Scale>(32);

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    QV &q_and_v = allocator.allocate<QV>();
    KHead &k_head = allocator.allocate<KHead>();
    KTail &k_tail = allocator.allocate<KTail>();
    __shared__ semaphore qk_head_done;
    __shared__ semaphore qk_tail_done;
    __shared__ semaphore pv_done;
    if (threadIdx.x == 0) {
        init_semaphore(qk_head_done, 0, 1);
        init_semaphore(qk_tail_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
    }
    load_throughput_dense_fp4_tile(q_and_v, g.a_fp4);
    load_throughput_dense_fp4_tile(k_head, g.b_fp4);
    load_throughput_dense_fp4_tile(k_tail, g.b_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();

    st_descriptor<QV, transpose::N> q_desc(q_and_v);
    st_descriptor<KHead, transpose::N> k_head_desc(k_head);
    st_descriptor<KTail, transpose::N> k_tail_desc(k_tail);
    st_descriptor<QV, transpose::N> v_desc(q_and_v);
    const bool issue_warp = warpgroup::warpid() == 0;
    const bool leader = issue_warp && warp::elect_leader();
    if (issue_warp) {
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            head_q_scale, 0x7f7f7f7fu);
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            head_k_scale, 0x7f7f7f7fu);
        fp4pv_tmem_store_wait();
    }
    __syncthreads();

    uint64_t start_cycle = 0;
    int phase = 0;
    if (leader) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
    }
    for (int iter = 0; iter < g.iterations; ++iter) {
        if (leader) {
            issue_scaled_n_k128_iteration<96, 0>(
                score_head, q_desc, k_head_desc,
                head_q_scale, head_k_scale);
            detail::tcgen05::commit<1>(qk_head_done);
        }
        if (issue_warp) {
            wait(qk_head_done, phase);
            tensor_after_thread_sync();
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                tail_p_scale, 0x7f7f7f7fu);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                tail_v_scale, 0x7f7f7f7fu);
            fp4pv_tmem_store_wait();
            tensor_before_thread_sync();
        }
        if (leader) {
            issue_scaled_n_k128_iteration<32, 0>(
                score_tail, q_desc, k_tail_desc,
                tail_p_scale, tail_v_scale);
            detail::tcgen05::commit<1>(qk_tail_done);
        }
        if (issue_warp) {
            wait(qk_tail_done, phase);
            tensor_after_thread_sync();
            fill_throughput_tmem_128x16(score_head.addr, 0x22222222u);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                tail_p_scale, 0x7f7f7f7fu);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                tail_v_scale, 0x7f7f7f7fu);
            fp4pv_tmem_store_wait();
            tensor_before_thread_sync();
        }
        if (leader) {
            if (iter == 0) {
                issue_tmem_smem_k128_iteration<0>(
                    output, score_head.addr, v_desc,
                    tail_p_scale, tail_v_scale);
            } else {
                issue_tmem_smem_k128_iteration<1>(
                    output, score_head.addr, v_desc,
                    tail_p_scale, tail_v_scale);
            }
            detail::tcgen05::commit<1>(pv_done);
        }
        if (issue_warp) {
            wait(pv_done, phase);
            tensor_after_thread_sync();
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                head_q_scale, 0x7f7f7f7fu);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                head_k_scale, 0x7f7f7f7fu);
            fp4pv_tmem_store_wait();
            tensor_before_thread_sync();
        }
        phase ^= 1;
    }
    if (leader) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
    __syncthreads();
}

// SM103-specific attention geometry: keep a wide QK destination and consume
// the key tile with native K96 Ultra PV commands plus a K64 tail for N256.
// The score, persistent output, and four scale pages fit in 512 columns.
template <int KEY_TILE>
struct config_raw_fp4_attention_ultra_probe {
    static_assert(KEY_TILE == 192 || KEY_TILE == 256);
    static constexpr int TMEM_COLS = 512;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 128;
    static constexpr int DYNAMIC_SHARED_MEMORY = 64 * 1024;
};

template <int KEY_TILE>
__launch_bounds__(128, 1)
__global__ void kernel_raw_fp4_attention_ultra_probe(
    const __grid_constant__ globals_raw_fp4_throughput_probe<
        config_raw_fp4_attention_ultra_probe<KEY_TILE>> g
) {
    using C = config_raw_fp4_attention_ultra_probe<KEY_TILE>;
    using Score = tt<float, 128, KEY_TILE>;
    using Output = tt<float, 128, 128>;
    using Q = st_fp4e2m1_2<128, 64>;
    using K = st_fp4e2m1_2<KEY_TILE, 64>;
    using V96 = st_fp4e2m1_2<96, 64>;
    using V64 = st_fp4e2m1_2<64, 64>;
    using Scale = full_tt_fp8e8m0<16>;
    using TmemAllocator = raw_fp4_shape_tmem_allocator<C::TMEM_COLS>;
    constexpr int SCORE_COL = 0;
    constexpr int OUTPUT_COL = KEY_TILE;
    constexpr int Q_SCALE_COL = OUTPUT_COL + 128;
    constexpr int K_SCALE_COL = Q_SCALE_COL + 16;
    constexpr int P_SCALE_COL = K_SCALE_COL + 16;
    constexpr int V_SCALE_COL = P_SCALE_COL + 16;
    static_assert(OUTPUT_COL + Output::cols == Q_SCALE_COL);
    static_assert(V_SCALE_COL + Scale::cols <= C::TMEM_COLS);

    TmemAllocator tm_alloc{};
    Score score = tm_alloc.template allocate<Score>(SCORE_COL);
    Output output = tm_alloc.template allocate<Output>(OUTPUT_COL);
    Scale q_scale = tm_alloc.template allocate<Scale>(Q_SCALE_COL);
    Scale k_scale = tm_alloc.template allocate<Scale>(K_SCALE_COL);
    Scale p_scale = tm_alloc.template allocate<Scale>(P_SCALE_COL);
    Scale v_scale = tm_alloc.template allocate<Scale>(V_SCALE_COL);

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    Q &q = allocator.allocate<Q>();
    K &k = allocator.allocate<K>();
    V96 &v0 = allocator.allocate<V96>();
    V96 &v1 = allocator.allocate<V96>();
    V64 &v_tail = allocator.allocate<V64>();
    __shared__ semaphore qk_done;
    __shared__ semaphore pv_done;
    if (threadIdx.x == 0) {
        init_semaphore(qk_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
    }
    load_throughput_dense_fp4_tile(q, g.a_fp4);
    load_throughput_dense_fp4_tile(k, g.b_fp4);
    load_throughput_dense_fp4_tile(v0, g.a_fp4);
    load_throughput_dense_fp4_tile(v1, g.b_fp4);
    load_throughput_dense_fp4_tile(v_tail, g.a_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();

    st_descriptor<Q, transpose::N> q_desc(q);
    st_descriptor<K, transpose::N> k_desc(k);
    st_descriptor<V96, transpose::N> v0_desc(v0);
    st_descriptor<V96, transpose::N> v1_desc(v1);
    st_descriptor<V64, transpose::N> v_tail_desc(v_tail);
    const bool issue_warp = warpgroup::warpid() == 0;
    const bool leader = issue_warp && warp::elect_leader();
    if (issue_warp) {
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            q_scale, 0x38383838u);
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            k_scale, 0x38383838u);
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            p_scale, 0x7f7f7f7fu);
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            v_scale, 0x7f7f7f7fu);
        fp4pv_tmem_store_wait();
    }
    __syncthreads();

    uint64_t start_cycle = 0;
    int phase = 0;
    if (leader) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
    }
    for (int iter = 0; iter < g.iterations; ++iter) {
        if (leader) {
            issue_nvfp4_wide_qk_iteration<KEY_TILE, 0>(
                score, q_desc, k_desc, q_scale, k_scale);
            detail::tcgen05::commit<1>(qk_done);
        }
        if (issue_warp) {
            wait(qk_done, phase);
            tensor_after_thread_sync();
            // P192/P256 occupy 24/32 physical TMEM columns respectively.
            fill_throughput_tmem_128x16(score.addr, 0x22222222u);
            fill_throughput_tmem_128x16(score.addr + 16, 0x22222222u);
            tensor_before_thread_sync();
        }
        if (leader) {
            if (iter == 0) {
                issue_tmem_smem_k96_iteration<0>(
                    output, score.addr, v0_desc, p_scale, v_scale);
            } else {
                issue_tmem_smem_k96_iteration<1>(
                    output, score.addr, v0_desc, p_scale, v_scale);
            }
            issue_tmem_smem_k96_iteration<1>(
                output, score.addr + 12, v1_desc, p_scale, v_scale);
            if constexpr (KEY_TILE == 256) {
                issue_tmem_smem_k64_iteration<1>(
                    output, score.addr + 24, v_tail_desc,
                    p_scale, v_scale);
            }
            detail::tcgen05::commit<1>(pv_done);
        }
        if (issue_warp) {
            wait(pv_done, phase);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
        }
        phase ^= 1;
    }
    if (leader) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
    __syncthreads();
}

template <typename Scratch>
__device__ __forceinline__ void raw_fp4_save_tmem_x16_to_smem(
    uint32_t source_addr,
    Scratch &scratch
) {
    static_assert(Scratch::rows == 16 && Scratch::cols == 128);
    float2 values[8];
    const uint32_t address = source_addr +
        ((warpgroup::warpid() * 32) << 16);
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
        : "r"(address));
    tensor_load_wait();
    const int row = warpgroup::laneid();
    #pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        scratch[{pair * 2 + 0, row}] = values[pair].x;
        scratch[{pair * 2 + 1, row}] = values[pair].y;
    }
    tensor_after_thread_sync();
    warpgroup::sync(1);
}

template <typename Scratch>
__device__ __forceinline__ void raw_fp4_restore_smem_x16_to_tmem(
    uint32_t destination_addr,
    const Scratch &scratch
) {
    static_assert(Scratch::rows == 16 && Scratch::cols == 128);
    float2 values[8];
    const int row = warpgroup::laneid();
    #pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = scratch[{pair * 2 + 0, row}];
        values[pair].y = scratch[{pair * 2 + 1, row}];
    }
    const uint32_t address = destination_addr +
        ((warpgroup::warpid() * 32) << 16);
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
           "r"(address));
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(1);
}

__device__ __forceinline__ void fill_nvfp4_scale_page_distinct(
    uint32_t tmem_addr,
    uint32_t packed0,
    uint32_t packed1,
    uint32_t packed2,
    uint32_t packed3);

// D128 dual-query ceiling for SM103. Score and two persistent outputs consume
// all 512 TMEM columns. QK temporarily borrows the first 16 columns of output
// 1 for folded NVFP4 scales; only that prefix is saved to shared memory and
// restored before output 1 accumulates its PV result.
__launch_bounds__(128, 1)
__global__ void kernel_raw_fp4_attention_n256_dual_query_prefix_swap_probe(
    const __grid_constant__ globals_raw_fp4_throughput_probe<
        config_raw_fp4_attention_ultra_probe<256>> g
) {
    using C = config_raw_fp4_attention_ultra_probe<256>;
    using Score = tt<float, 128, 256>;
    using Output = tt<float, 128, 128>;
    using Q = st_fp4e2m1_2<128, 64>;
    using K = st_fp4e2m1_2<256, 64>;
    using V96 = st_fp4e2m1_2<96, 64>;
    using V64 = st_fp4e2m1_2<64, 64>;
    using PrefixScratch = st_fl<16, 128, false>;
    using NVScale = full_tt_fp8e4m3<16>;
    using MXScale = full_tt_fp8e8m0<16>;
    using TmemAllocator = raw_fp4_shape_tmem_allocator<C::TMEM_COLS>;
    constexpr int SCORE_COL = 0;
    constexpr int OUTPUT0_COL = 256;
    constexpr int OUTPUT1_COL = 384;
    constexpr int Q_SCALE_COL = OUTPUT1_COL;
    constexpr int K_SCALE_COL = OUTPUT1_COL + 8;
    constexpr int P_SCALE_COL = 64;
    constexpr int V_SCALE_COL = 80;
    static_assert(OUTPUT1_COL + Output::cols == C::TMEM_COLS);

    TmemAllocator tm_alloc{};
    Score score = tm_alloc.template allocate<Score>(SCORE_COL);
    Output output0 = tm_alloc.template allocate<Output>(OUTPUT0_COL);
    Output output1 = tm_alloc.template allocate<Output>(OUTPUT1_COL);
    // These logical wrappers intentionally alias live score/output storage.
    // The kernel orders every use so the physical columns have one owner.
    NVScale q_scale(Q_SCALE_COL);
    NVScale k_scale(K_SCALE_COL);
    MXScale p_scale(P_SCALE_COL);
    MXScale v_scale(V_SCALE_COL);

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    Q &q0 = allocator.allocate<Q>();
    Q &q1 = allocator.allocate<Q>();
    K &k = allocator.allocate<K>();
    V96 &v0 = allocator.allocate<V96>();
    V96 &v1 = allocator.allocate<V96>();
    V64 &v_tail = allocator.allocate<V64>();
    PrefixScratch &output1_prefix = allocator.allocate<PrefixScratch>();
    __shared__ semaphore qk0_done;
    __shared__ semaphore qk1_done;
    __shared__ semaphore pv1_done;
    if (threadIdx.x == 0) {
        init_semaphore(qk0_done, 0, 1);
        init_semaphore(qk1_done, 0, 1);
        init_semaphore(pv1_done, 0, 1);
    }
    load_throughput_dense_fp4_tile(q0, g.a_fp4);
    load_throughput_dense_fp4_tile(q1, g.b_fp4);
    load_throughput_dense_fp4_tile(k, g.b_fp4);
    load_throughput_dense_fp4_tile(v0, g.a_fp4);
    load_throughput_dense_fp4_tile(v1, g.b_fp4);
    load_throughput_dense_fp4_tile(v_tail, g.a_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();

    st_descriptor<Q, transpose::N> q0_desc(q0);
    st_descriptor<Q, transpose::N> q1_desc(q1);
    st_descriptor<K, transpose::N> k_desc(k);
    st_descriptor<V96, transpose::N> v0_desc(v0);
    st_descriptor<V96, transpose::N> v1_desc(v1);
    st_descriptor<V64, transpose::N> v_tail_desc(v_tail);
    const bool issue_warp = warpgroup::warpid() == 0;
    const bool leader = issue_warp && warp::elect_leader();

    uint64_t start_cycle = 0;
    int phase = 0;
    if (leader) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
    }
    for (int iter = 0; iter < g.iterations; ++iter) {
        if (iter != 0) {
            raw_fp4_save_tmem_x16_to_smem(
                output1.addr, output1_prefix);
        }
        if (issue_warp) {
            fill_nvfp4_scale_page_distinct(
                q_scale.addr, 0x38383838u, 0x38383838u,
                0x38383838u, 0x38383838u);
            fill_nvfp4_scale_page_distinct(
                k_scale.addr, 0x38383838u, 0x38383838u,
                0x38383838u, 0x38383838u);
            fp4pv_tmem_store_wait();
            tensor_before_thread_sync();
        }
        if (leader) {
            issue_nvfp4_wide_qk_iteration<256, 0>(
                score, q0_desc, k_desc, q_scale, k_scale);
            detail::tcgen05::commit<1>(qk0_done);
        }
        if (issue_warp) {
            wait(qk0_done, phase);
            tensor_after_thread_sync();
            fill_throughput_tmem_128x16(score.addr, 0x22222222u);
            fill_throughput_tmem_128x16(score.addr + 16, 0x22222222u);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                p_scale, 0x7f7f7f7fu);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                v_scale, 0x7f7f7f7fu);
            fp4pv_tmem_store_wait();
            tensor_before_thread_sync();
        }
        if (leader) {
            if (iter == 0) {
                issue_tmem_smem_k96_iteration<0>(
                    output0, score.addr, v0_desc, p_scale, v_scale);
            } else {
                issue_tmem_smem_k96_iteration<1>(
                    output0, score.addr, v0_desc, p_scale, v_scale);
            }
            issue_tmem_smem_k96_iteration<1>(
                output0, score.addr + 12, v1_desc, p_scale, v_scale);
            issue_tmem_smem_k64_iteration<1>(
                output0, score.addr + 24, v_tail_desc, p_scale, v_scale);
        }
        if (issue_warp) {
            // Model the query-local Q scale rewrite while PV0 is in flight.
            fill_nvfp4_scale_page_distinct(
                q_scale.addr, 0x38383838u, 0x38383838u,
                0x38383838u, 0x38383838u);
            fp4pv_tmem_store_wait();
            tensor_before_thread_sync();
        }
        if (leader) {
            issue_nvfp4_wide_qk_iteration<256, 0>(
                score, q1_desc, k_desc, q_scale, k_scale);
            detail::tcgen05::commit<1>(qk1_done);
        }
        if (issue_warp) {
            wait(qk1_done, phase);
            tensor_after_thread_sync();
            fill_throughput_tmem_128x16(score.addr, 0x22222222u);
            fill_throughput_tmem_128x16(score.addr + 16, 0x22222222u);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                p_scale, 0x7f7f7f7fu);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                v_scale, 0x7f7f7f7fu);
            fp4pv_tmem_store_wait();
            tensor_before_thread_sync();
        }
        warpgroup::sync(1);
        if (iter != 0) {
            raw_fp4_restore_smem_x16_to_tmem(
                output1.addr, output1_prefix);
        }
        if (leader) {
            if (iter == 0) {
                issue_tmem_smem_k96_iteration<0>(
                    output1, score.addr, v0_desc, p_scale, v_scale);
            } else {
                issue_tmem_smem_k96_iteration<1>(
                    output1, score.addr, v0_desc, p_scale, v_scale);
            }
            issue_tmem_smem_k96_iteration<1>(
                output1, score.addr + 12, v1_desc, p_scale, v_scale);
            issue_tmem_smem_k64_iteration<1>(
                output1, score.addr + 24, v_tail_desc, p_scale, v_scale);
            detail::tcgen05::commit<1>(pv1_done);
        }
        if (issue_warp) {
            wait(pv1_done, phase);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
        }
        warpgroup::sync(1);
        phase ^= 1;
    }
    if (leader) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_fp4_throughput_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
    __syncthreads();
}

__device__ __forceinline__ void fill_nvfp4_scale_page_distinct(
    uint32_t tmem_addr,
    uint32_t packed0,
    uint32_t packed1,
    uint32_t packed2,
    uint32_t packed3
) {
    #pragma unroll
    for (int row = 0; row < 128; row += 32) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32 "
            "[%0], {%1, %2, %3, %4};" ::
            "r"(tmem_addr + (static_cast<uint32_t>(row) << 16)),
            "r"(packed0), "r"(packed1), "r"(packed2), "r"(packed3));
    }
}

__device__ __forceinline__ void copy_nvfp4_qk_tail_scale_pages(
    uint32_t src_q_k0,
    uint32_t src_q_k1,
    uint32_t src_k_k0,
    uint32_t src_k_k1,
    uint32_t dst_q_k0,
    uint32_t dst_q_k1,
    uint32_t dst_k_k0,
    uint32_t dst_k_k1
) {
    #pragma unroll
    for (int row = 0; row < 128; row += 32) {
        const uint32_t row_offset = static_cast<uint32_t>(row) << 16;
        uint32_t q0[4];
        uint32_t q1[4];
        uint32_t k0_tail;
        uint32_t k1_tail;
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x4.b32 "
            "{%0,%1,%2,%3}, [%4];}"
            : "=r"(q0[0]), "=r"(q0[1]), "=r"(q0[2]), "=r"(q0[3])
            : "r"(src_q_k0 + row_offset));
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x4.b32 "
            "{%0,%1,%2,%3}, [%4];}"
            : "=r"(q1[0]), "=r"(q1[1]), "=r"(q1[2]), "=r"(q1[3])
            : "r"(src_q_k1 + row_offset));
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];}"
            : "=r"(k0_tail)
            : "r"(src_k_k0 + row_offset + 3));
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];}"
            : "=r"(k1_tail)
            : "r"(src_k_k1 + row_offset + 3));
        tensor_load_wait();
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x4.b32 "
            "[%0], {%1,%2,%3,%4};}"
            :: "r"(dst_q_k0 + row_offset),
               "r"(q0[0]), "r"(q0[1]), "r"(q0[2]), "r"(q0[3]));
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x4.b32 "
            "[%0], {%1,%2,%3,%4};}"
            :: "r"(dst_q_k1 + row_offset),
               "r"(q1[0]), "r"(q1[1]), "r"(q1[2]), "r"(q1[3]));
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};}"
            :: "r"(dst_k_k0 + row_offset), "r"(k0_tail));
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};}"
            :: "r"(dst_k_k1 + row_offset), "r"(k1_tail));
    }
    fp4pv_tmem_store_wait();
    tensor_before_thread_sync();
}

template <int N, int ACC, typename Score, typename QDesc, typename KDesc>
__device__ __forceinline__ void issue_nvfp4_split_qk_chunk(
    Score &score,
    QDesc &q_desc,
    KDesc &k_desc,
    int chunk,
    uint32_t q_scale_addr,
    uint32_t k_scale_addr
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, N, false, 0>();
    if (chunk == 0) {
        detail::tcgen05::st_st<fp4_t, scale_t, ACC, 1, 16>(
            score.addr,
            q_desc.chunk_descriptor(0),
            k_desc.chunk_descriptor(0),
            q_scale_addr,
            k_scale_addr,
            idesc);
    } else {
        detail::tcgen05::st_st<fp4_t, scale_t, 1, 1, 16>(
            score.addr,
            q_desc.chunk_descriptor(1),
            k_desc.chunk_descriptor(1),
            q_scale_addr,
            k_scale_addr,
            idesc);
    }
}

// Verifies that N96+N32 issue is arithmetically identical to N128 issue for
// NVFP4 block16 scales.  The four distinct physical K-scale columns make an
// incorrect tail-scale offset immediately visible rather than allowing a
// unit-scale test to pass accidentally.
template <bool ALIAS_SCALE_SCRATCH>
__launch_bounds__(128, 1)
__global__ void kernel_nvfp4_n128_split_qk_correctness(
    const __grid_constant__ globals_raw_fp4_throughput_probe<
        config_raw_fp4_attention_n96_k96_probe<1>> g
) {
    using C = config_raw_fp4_attention_n96_k96_probe<1>;
    using Score = tt<float, 128, 128>;
    using QKFull = st_fp4e2m1_2<128, 64>;
    using KHead = st_fp4e2m1_2<96, 64>;
    using KTail = st_fp4e2m1_2<32, 64>;
    using TmemAllocator = raw_fp4_shape_tmem_allocator<512>;
    constexpr int FULL_SCORE_COL = 0;
    constexpr int SPLIT_SCORE_COL = 128;
    constexpr int Q_SCALE_K0_COL = 256;
    constexpr int Q_SCALE_K1_COL = 260;
    constexpr int K_SCALE_K0_COL = 264;
    constexpr int K_SCALE_K1_COL = 268;
    constexpr int Q_TAIL_SCALE_K0_COL = 272;
    constexpr int Q_TAIL_SCALE_K1_COL = 276;
    constexpr int K_TAIL_SCALE_K0_COL = 280;
    constexpr int K_TAIL_SCALE_K1_COL = 284;
    constexpr int SPLIT_Q_SCALE_K0_COL =
        ALIAS_SCALE_SCRATCH ? SPLIT_SCORE_COL + 96 : Q_SCALE_K0_COL;
    constexpr int SPLIT_Q_SCALE_K1_COL =
        ALIAS_SCALE_SCRATCH ? SPLIT_SCORE_COL + 100 : Q_SCALE_K1_COL;
    constexpr int SPLIT_K_SCALE_K0_COL =
        ALIAS_SCALE_SCRATCH ? SPLIT_SCORE_COL + 104 : K_SCALE_K0_COL;
    constexpr int SPLIT_K_SCALE_K1_COL =
        ALIAS_SCALE_SCRATCH ? SPLIT_SCORE_COL + 108 : K_SCALE_K1_COL;
    constexpr int SPLIT_Q_TAIL_SCALE_K0_COL =
        ALIAS_SCALE_SCRATCH ? SPLIT_SCORE_COL : Q_TAIL_SCALE_K0_COL;
    constexpr int SPLIT_Q_TAIL_SCALE_K1_COL =
        ALIAS_SCALE_SCRATCH ? SPLIT_SCORE_COL + 4 : Q_TAIL_SCALE_K1_COL;
    constexpr int SPLIT_K_TAIL_SCALE_K0_COL =
        ALIAS_SCALE_SCRATCH ? SPLIT_SCORE_COL + 8 : K_TAIL_SCALE_K0_COL;
    constexpr int SPLIT_K_TAIL_SCALE_K1_COL =
        ALIAS_SCALE_SCRATCH ? SPLIT_SCORE_COL + 12 : K_TAIL_SCALE_K1_COL;

    TmemAllocator tm_alloc{};
    Score full_score = tm_alloc.template allocate<Score>(FULL_SCORE_COL);
    Score split_score = tm_alloc.template allocate<Score>(SPLIT_SCORE_COL);

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    QKFull &q = allocator.allocate<QKFull>();
    QKFull &k_full = allocator.allocate<QKFull>();
    KHead &k_head = allocator.allocate<KHead>();
    KTail &k_tail = *reinterpret_cast<KTail *>(
        &k_full.data[96 * QKFull::cols]);
    __shared__ semaphore done;
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    load_throughput_dense_fp4_tile(q, g.a_fp4);
    load_throughput_dense_fp4_tile(k_full, g.b_fp4);
    load_throughput_dense_fp4_tile(k_head, g.b_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();

    st_descriptor<QKFull, transpose::N> q_desc(q);
    st_descriptor<QKFull, transpose::N> k_full_desc(k_full);
    st_descriptor<KHead, transpose::N> k_head_desc(k_head);
    st_descriptor<KTail, transpose::N> k_tail_desc(k_tail);
    const bool issue_warp = warpgroup::warpid() == 0;
    const bool leader = issue_warp && warp::elect_leader();
    if (issue_warp) {
        // E4M3 encodings: 1.0, 1.125, 1.25, 1.5.  The second K64
        // chunk permutes them so chunk/address mistakes cannot cancel.
        fill_nvfp4_scale_page_distinct(
            tm_alloc.addr + Q_SCALE_K0_COL,
            0x38383838u, 0x39393939u, 0x3a3a3a3au, 0x3c3c3c3cu);
        fill_nvfp4_scale_page_distinct(
            tm_alloc.addr + Q_SCALE_K1_COL,
            0x3a3a3a3au, 0x3c3c3c3cu, 0x38383838u, 0x39393939u);
        fill_nvfp4_scale_page_distinct(
            tm_alloc.addr + K_SCALE_K0_COL,
            0x38383838u, 0x3a3a3a3au, 0x39393939u, 0x3c3c3c3cu);
        fill_nvfp4_scale_page_distinct(
            tm_alloc.addr + K_SCALE_K1_COL,
            0x3c3c3c3cu, 0x39393939u, 0x3a3a3a3au, 0x38383838u);
        if constexpr (ALIAS_SCALE_SCRATCH) {
            fill_nvfp4_scale_page_distinct(
                tm_alloc.addr + SPLIT_Q_SCALE_K0_COL,
                0x38383838u, 0x39393939u, 0x3a3a3a3au, 0x3c3c3c3cu);
            fill_nvfp4_scale_page_distinct(
                tm_alloc.addr + SPLIT_Q_SCALE_K1_COL,
                0x3a3a3a3au, 0x3c3c3c3cu, 0x38383838u, 0x39393939u);
            fill_nvfp4_scale_page_distinct(
                tm_alloc.addr + SPLIT_K_SCALE_K0_COL,
                0x38383838u, 0x3a3a3a3au, 0x39393939u, 0x3c3c3c3cu);
            fill_nvfp4_scale_page_distinct(
                tm_alloc.addr + SPLIT_K_SCALE_K1_COL,
                0x3c3c3c3cu, 0x39393939u, 0x3a3a3a3au, 0x38383838u);
        }
        fp4pv_tmem_store_wait();
        tensor_before_thread_sync();
    }
    __syncthreads();

    if (leader) {
        issue_nvfp4_split_qk_chunk<128, 0>(
            full_score, q_desc, k_full_desc, 0,
            tm_alloc.addr + Q_SCALE_K0_COL,
            tm_alloc.addr + K_SCALE_K0_COL);
        issue_nvfp4_split_qk_chunk<128, 1>(
            full_score, q_desc, k_full_desc, 1,
            tm_alloc.addr + Q_SCALE_K1_COL,
            tm_alloc.addr + K_SCALE_K1_COL);

        issue_nvfp4_split_qk_chunk<96, 0>(
            split_score, q_desc, k_head_desc, 0,
            tm_alloc.addr + SPLIT_Q_SCALE_K0_COL,
            tm_alloc.addr + SPLIT_K_SCALE_K0_COL);
        issue_nvfp4_split_qk_chunk<96, 1>(
            split_score, q_desc, k_head_desc, 1,
            tm_alloc.addr + SPLIT_Q_SCALE_K1_COL,
            tm_alloc.addr + SPLIT_K_SCALE_K1_COL);
        detail::tcgen05::commit<1>(done);
        wait(done, 0);
        tensor_after_thread_sync();
    }
    __syncthreads();
    if (issue_warp) {
        copy_nvfp4_qk_tail_scale_pages(
            tm_alloc.addr + Q_SCALE_K0_COL,
            tm_alloc.addr + Q_SCALE_K1_COL,
            tm_alloc.addr + K_SCALE_K0_COL,
            tm_alloc.addr + K_SCALE_K1_COL,
            tm_alloc.addr + SPLIT_Q_TAIL_SCALE_K0_COL,
            tm_alloc.addr + SPLIT_Q_TAIL_SCALE_K1_COL,
            tm_alloc.addr + SPLIT_K_TAIL_SCALE_K0_COL,
            tm_alloc.addr + SPLIT_K_TAIL_SCALE_K1_COL);
    }
    __syncthreads();
    if (leader) {
        Score split_tail(split_score.addr + 96);
        issue_nvfp4_split_qk_chunk<32, 0>(
            split_tail, q_desc, k_tail_desc, 0,
            tm_alloc.addr + SPLIT_Q_TAIL_SCALE_K0_COL,
            tm_alloc.addr + SPLIT_K_TAIL_SCALE_K0_COL);
        issue_nvfp4_split_qk_chunk<32, 1>(
            split_tail, q_desc, k_tail_desc, 1,
            tm_alloc.addr + SPLIT_Q_TAIL_SCALE_K1_COL,
            tm_alloc.addr + SPLIT_K_TAIL_SCALE_K1_COL);
        detail::tcgen05::commit<1>(done);
        wait(done, 1);
        tensor_after_thread_sync();
    }
    __syncthreads();

    uint64_t exact_mismatches = 0;
    uint64_t nonfinite_mismatches = 0;
    uint32_t max_abs_diff_bits = 0;
    const uint32_t row_base =
        static_cast<uint32_t>(warpgroup::warpid() * 32) << 16;
    #pragma unroll
    for (int col = 0; col < 128; col += 16) {
        float full[16];
        float split[16];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(full[0]), "=f"(full[1]), "=f"(full[2]), "=f"(full[3]),
              "=f"(full[4]), "=f"(full[5]), "=f"(full[6]), "=f"(full[7]),
              "=f"(full[8]), "=f"(full[9]), "=f"(full[10]), "=f"(full[11]),
              "=f"(full[12]), "=f"(full[13]), "=f"(full[14]), "=f"(full[15])
            : "r"(full_score.addr + row_base + col));
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(split[0]), "=f"(split[1]), "=f"(split[2]), "=f"(split[3]),
              "=f"(split[4]), "=f"(split[5]), "=f"(split[6]), "=f"(split[7]),
              "=f"(split[8]), "=f"(split[9]), "=f"(split[10]), "=f"(split[11]),
              "=f"(split[12]), "=f"(split[13]), "=f"(split[14]), "=f"(split[15])
            : "r"(split_score.addr + row_base + col));
        tensor_load_wait();
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint32_t full_bits = __float_as_uint(full[i]);
            const uint32_t split_bits = __float_as_uint(split[i]);
            exact_mismatches += full_bits != split_bits;
            const bool finite = isfinite(full[i]) && isfinite(split[i]);
            nonfinite_mismatches += !finite;
            if (finite) {
                max_abs_diff_bits = max(
                    max_abs_diff_bits,
                    __float_as_uint(fabsf(full[i] - split[i])));
            }
        }
    }
    if (exact_mismatches) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(&g.cycles[blockIdx.x]),
            static_cast<unsigned long long>(exact_mismatches));
    }
    if (nonfinite_mismatches) {
        atomicAdd(
            reinterpret_cast<unsigned long long *>(
                &g.start_globaltimer[blockIdx.x]),
            static_cast<unsigned long long>(nonfinite_mismatches));
    }
    atomicMax(
        reinterpret_cast<unsigned long long *>(&g.end_globaltimer[blockIdx.x]),
        static_cast<unsigned long long>(max_abs_diff_bits));
    if (threadIdx.x == 0) {
        g.smids[blockIdx.x] = C::TMEM_COLS;
    }
    __syncthreads();
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, C::TMEM_RESIDENT_BLOCKS)
__global__ void kernel_raw_fp4_throughput_probe(
    const __grid_constant__ globals_raw_fp4_throughput_probe<C> g
) {
#if TK_FA4_RAW_FP4_MMA_THROUGHPUT_PRAGMA
    asm volatile(".pragma \"mma_throughput\";\n");
#endif
    using output_t = tt<float, 128, 128>;
    using raw_tile = st_fp4e2m1_2<128, 128>;
    using scaled_tile = st_fp4e2m1_2<128, 64>;
    using sparse_tile = st_fp4e2m1_2<128, 32>;
    using bf16_tile = st_bf<128, 128>;
    using scale_t = full_tt_fp8e8m0<16>;
    using nv_scale_t = full_tt_fp8e4m3<16>;

    tensor_allocator<C::TMEM_RESIDENT_BLOCKS, 1> tm_alloc{};
    output_t output = tm_alloc.template allocate<output_t>(0);
    output_t output1 = tm_alloc.template allocate<output_t>(
        C::MULTI_CHAIN ? 128 : 0);
    output_t output2 = tm_alloc.template allocate<output_t>(
        C::TRIPLE_CHAIN ? 256 : 0);
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    __shared__ semaphore done;
    __shared__ semaphore parallel_done[3];
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    if (threadIdx.x < 3) {
        init_semaphore(parallel_done[threadIdx.x], 0, 1);
    }

    if constexpr (C::MODE == RAW_FP4_THROUGHPUT) {
        raw_tile &a = allocator.allocate<raw_tile>();
        raw_tile &b = allocator.allocate<raw_tile>();
        load_throughput_raw_fp4_padded_tile<C>(a, g.a_fp4);
        load_throughput_raw_fp4_padded_tile<C>(b, g.b_fp4);
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();
        st_descriptor<raw_tile, transpose::N> a_desc(a);
        st_descriptor<raw_tile, transpose::N> b_desc(b);
        const bool issue_warp = warpgroup::warpid() == 0;
        const bool leader = issue_warp && warp::elect_leader();
        uint64_t start_cycle = 0;
        if (leader) {
            tensor_before_thread_sync();
            start_cycle = clock64();
            g.start_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            issue_raw_k128_iteration<0>(output, a_desc, b_desc);
            for (int iter = 1; iter < g.iterations; ++iter) {
                issue_raw_k128_iteration<1>(output, a_desc, b_desc);
            }
            detail::tcgen05::commit<1>(done);
        }
        if (issue_warp) {
            wait(done, 0);
            tensor_after_thread_sync();
        }
        if (leader) {
            const uint64_t end_cycle = clock64();
            g.end_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            g.cycles[blockIdx.x] = end_cycle - start_cycle;
            g.smids[blockIdx.x] = smid();
        }
    } else if constexpr (
        C::MODE == SCALED_MXFP4_THROUGHPUT ||
        C::MODE == DEDICATED_MXFP4_THROUGHPUT ||
        C::MODE == DUAL_CHAIN_MXFP4_THROUGHPUT ||
        C::MODE == TRIPLE_CHAIN_MXFP4_THROUGHPUT ||
        C::MODE == DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT ||
        C::MODE == DUAL_CHAIN_K192_K96_MXFP4_THROUGHPUT ||
        C::MODE == TRIPLE_CHAIN_K192_K96_MXFP4_THROUGHPUT ||
        C::MODE == DUAL_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT ||
        C::MODE == TRIPLE_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT ||
        C::MODE == DUAL_SCORE_COLLECTOR_MXFP4_THROUGHPUT ||
        C::MODE == PARALLEL_DUAL_CHAIN_MXFP4_THROUGHPUT ||
        C::MODE == PARALLEL_TRIPLE_CHAIN_MXFP4_THROUGHPUT) {
        scaled_tile &a = allocator.allocate<scaled_tile>();
        scaled_tile &b = allocator.allocate<scaled_tile>();
        load_throughput_dense_fp4_tile(a, g.a_fp4);
        load_throughput_dense_fp4_tile(b, g.b_fp4);
        constexpr int SCALE_BASE =
            C::TRIPLE_CHAIN ? 384 : (C::DUAL_CHAIN ? 256 : 128);
        scale_t a_scale = tm_alloc.template allocate<scale_t>(SCALE_BASE);
        scale_t b_scale = tm_alloc.template allocate<scale_t>(SCALE_BASE + 16);
        const bool issue_warp = warpgroup::warpid() == 0;
        const bool leader = issue_warp && warp::elect_leader();
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        if (issue_warp) {
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(a_scale, 0x7f7f7f7fu);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(b_scale, 0x7f7f7f7fu);
            fp4pv_tmem_store_wait();
        }
        __syncthreads();
        st_descriptor<scaled_tile, transpose::N> a_desc(a);
        st_descriptor<scaled_tile, transpose::N> b_desc(b);
        if constexpr (C::PARALLEL_CHAIN) {
            constexpr int CHAINS = C::TRIPLE_CHAIN ? 3 : 2;
            const int chain = warpgroup::warpid();
            const bool active_warp = chain < CHAINS;
            const bool chain_leader = active_warp && warp::elect_leader();
            const int chain_iterations =
                g.iterations / CHAINS + (chain < (g.iterations % CHAINS));
            uint64_t start_cycle = 0;
            __syncthreads();
            if (threadIdx.x == 0) {
                start_cycle = clock64();
                g.start_globaltimer[blockIdx.x] =
                    raw_fp4_throughput_globaltimer();
            }
            __syncthreads();
            if (chain_leader) {
                if (chain == 0) {
                    issue_scaled_k128_chain(
                        output, a_desc, b_desc, a_scale, b_scale,
                        chain_iterations, parallel_done[0]);
                } else if (chain == 1) {
                    issue_scaled_k128_chain(
                        output1, a_desc, b_desc, a_scale, b_scale,
                        chain_iterations, parallel_done[1]);
                } else {
                    issue_scaled_k128_chain(
                        output2, a_desc, b_desc, a_scale, b_scale,
                        chain_iterations, parallel_done[2]);
                }
            }
            if (active_warp) {
                wait(parallel_done[chain], 0);
                tensor_after_thread_sync();
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const uint64_t end_cycle = clock64();
                g.end_globaltimer[blockIdx.x] =
                    raw_fp4_throughput_globaltimer();
                g.cycles[blockIdx.x] = end_cycle - start_cycle;
                g.smids[blockIdx.x] = smid();
            }
        } else {
        uint64_t start_cycle = 0;
        if (leader) {
            tensor_before_thread_sync();
            start_cycle = clock64();
            g.start_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            if constexpr (
                C::MODE ==
                TRIPLE_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT) {
                issue_sm103_mxfp4_k192_k96_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                issue_sm103_mxfp4_k192_k96_iteration<0>(
                    output1, a_desc, b_desc, a_scale, b_scale);
                issue_sm103_mxfp4_k192_k96_iteration<0>(
                    output2, a_desc, b_desc, a_scale, b_scale);
                int iter = 3;
                for (; iter + 2 < g.iterations; iter += 3) {
                    issue_sm103_mxfp4_k192_k96_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_sm103_mxfp4_k192_k96_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                    issue_sm103_mxfp4_k192_k96_iteration<1>(
                        output2, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_sm103_mxfp4_k192_k96_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    ++iter;
                }
                if (iter < g.iterations) {
                    issue_sm103_mxfp4_k192_k96_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (
                C::MODE == DUAL_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT) {
                issue_sm103_mxfp4_k192_k96_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                issue_sm103_mxfp4_k192_k96_iteration<0>(
                    output1, a_desc, b_desc, a_scale, b_scale);
                int iter = 2;
                for (; iter + 1 < g.iterations; iter += 2) {
                    issue_sm103_mxfp4_k192_k96_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_sm103_mxfp4_k192_k96_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_sm103_mxfp4_k192_k96_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (C::MODE == TRIPLE_CHAIN_MXFP4_THROUGHPUT) {
                issue_scaled_k128_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                issue_scaled_k128_iteration<0>(
                    output1, a_desc, b_desc, a_scale, b_scale);
                issue_scaled_k128_iteration<0>(
                    output2, a_desc, b_desc, a_scale, b_scale);
                int iter = 3;
                for (; iter + 2 < g.iterations; iter += 3) {
                    issue_scaled_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_scaled_k128_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                    issue_scaled_k128_iteration<1>(
                        output2, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_scaled_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    ++iter;
                }
                if (iter < g.iterations) {
                    issue_scaled_k128_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (
                C::MODE == TRIPLE_CHAIN_K192_K96_MXFP4_THROUGHPUT) {
                issue_scaled_k192_k96_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                issue_scaled_k192_k96_iteration<0>(
                    output1, a_desc, b_desc, a_scale, b_scale);
                issue_scaled_k192_k96_iteration<0>(
                    output2, a_desc, b_desc, a_scale, b_scale);
                int iter = 3;
                for (; iter + 2 < g.iterations; iter += 3) {
                    issue_scaled_k192_k96_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_scaled_k192_k96_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                    issue_scaled_k192_k96_iteration<1>(
                        output2, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_scaled_k192_k96_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    ++iter;
                }
                if (iter < g.iterations) {
                    issue_scaled_k192_k96_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (
                C::MODE == DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT ||
                C::MODE == DUAL_CHAIN_K192_K96_MXFP4_THROUGHPUT) {
                if constexpr (
                    C::MODE == DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT) {
                    issue_scaled_k192_k64_iteration<0>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_scaled_k192_k64_iteration<0>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                } else {
                    issue_scaled_k192_k96_iteration<0>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_scaled_k192_k96_iteration<0>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
                int iter = 2;
                for (; iter + 1 < g.iterations; iter += 2) {
                    if constexpr (
                        C::MODE == DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT) {
                        issue_scaled_k192_k64_iteration<1>(
                            output, a_desc, b_desc, a_scale, b_scale);
                        issue_scaled_k192_k64_iteration<1>(
                            output1, a_desc, b_desc, a_scale, b_scale);
                    } else {
                        issue_scaled_k192_k96_iteration<1>(
                            output, a_desc, b_desc, a_scale, b_scale);
                        issue_scaled_k192_k96_iteration<1>(
                            output1, a_desc, b_desc, a_scale, b_scale);
                    }
                }
                if (iter < g.iterations) {
                    if constexpr (
                        C::MODE == DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT) {
                        issue_scaled_k192_k64_iteration<1>(
                            output, a_desc, b_desc, a_scale, b_scale);
                    } else {
                        issue_scaled_k192_k96_iteration<1>(
                            output, a_desc, b_desc, a_scale, b_scale);
                    }
                }
            } else if constexpr (
                C::MODE == DUAL_SCORE_COLLECTOR_MXFP4_THROUGHPUT) {
                issue_collector_mxfp4_k128_pair<0>(
                    output, output1, a_desc, b_desc, a_scale, b_scale);
                int iter = 2;
                for (; iter + 1 < g.iterations; iter += 2) {
                    issue_collector_mxfp4_k128_pair<1>(
                        output, output1, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_dedicated_mxfp4_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (
                C::MODE == DUAL_CHAIN_MXFP4_THROUGHPUT) {
                issue_scaled_k128_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                issue_scaled_k128_iteration<0>(
                    output1, a_desc, b_desc, a_scale, b_scale);
                int iter = 2;
                for (; iter + 1 < g.iterations; iter += 2) {
                    issue_scaled_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_scaled_k128_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_scaled_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (
                C::MODE == DEDICATED_MXFP4_THROUGHPUT) {
                issue_dedicated_mxfp4_k128_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                for (int iter = 1; iter < g.iterations; ++iter) {
                    issue_dedicated_mxfp4_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                }
            } else {
                issue_scaled_k128_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                for (int iter = 1; iter < g.iterations; ++iter) {
                    issue_scaled_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                }
            }
            detail::tcgen05::commit<1>(done);
        }
        if (issue_warp) {
            wait(done, 0);
            tensor_after_thread_sync();
        }
        if (leader) {
            const uint64_t end_cycle = clock64();
            g.end_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            g.cycles[blockIdx.x] = end_cycle - start_cycle;
            g.smids[blockIdx.x] = smid();
        }
        }
    } else if constexpr (C::MODE == BF16_THROUGHPUT) {
        bf16_tile &a = allocator.allocate<bf16_tile>();
        bf16_tile &b = allocator.allocate<bf16_tile>();
        load_throughput_bf16_tile(a, g.a_bf16);
        load_throughput_bf16_tile(b, g.b_bf16);
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();
        st_descriptor<bf16_tile, transpose::N> a_desc(a);
        st_descriptor<bf16_tile, transpose::N> b_desc(b);
        const bool issue_warp = warpgroup::warpid() == 0;
        const bool leader = issue_warp && warp::elect_leader();
        uint64_t start_cycle = 0;
        if (leader) {
            tensor_before_thread_sync();
            start_cycle = clock64();
            g.start_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            issue_bf16_k128_iteration<0>(output, a_desc, b_desc);
            for (int iter = 1; iter < g.iterations; ++iter) {
                issue_bf16_k128_iteration<1>(output, a_desc, b_desc);
            }
            detail::tcgen05::commit<1>(done);
        }
        if (issue_warp) {
            wait(done, 0);
            tensor_after_thread_sync();
        }
        if (leader) {
            const uint64_t end_cycle = clock64();
            g.end_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            g.cycles[blockIdx.x] = end_cycle - start_cycle;
            g.smids[blockIdx.x] = smid();
        }
    } else if constexpr (
        C::MODE == DENSE_NVFP4_THROUGHPUT ||
        C::MODE == DUAL_CHAIN_NVFP4_THROUGHPUT ||
        C::MODE == DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT ||
        C::MODE == DUAL_CHAIN_K192_K96_NVFP4_THROUGHPUT ||
        C::MODE == TRIPLE_CHAIN_K192_K96_NVFP4_THROUGHPUT ||
        C::MODE == DUAL_SCORE_COLLECTOR_NVFP4_THROUGHPUT) {
        scaled_tile &a = allocator.allocate<scaled_tile>();
        scaled_tile &b = allocator.allocate<scaled_tile>();
        load_throughput_dense_fp4_tile(a, g.a_fp4);
        load_throughput_dense_fp4_tile(b, g.b_fp4);
        constexpr int SCALE_BASE =
            C::TRIPLE_CHAIN ? 384 : (C::DUAL_CHAIN ? 256 : 128);
        nv_scale_t a_scale = tm_alloc.template allocate<nv_scale_t>(SCALE_BASE);
        nv_scale_t b_scale = tm_alloc.template allocate<nv_scale_t>(SCALE_BASE + 16);
        const bool issue_warp = warpgroup::warpid() == 0;
        const bool leader = issue_warp && warp::elect_leader();
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        if (issue_warp) {
            fill_throughput_tmem_128x16(a_scale.addr, 0x38383838u);
            fill_throughput_tmem_128x16(b_scale.addr, 0x38383838u);
            fp4pv_tmem_store_wait();
        }
        __syncthreads();
        st_descriptor<scaled_tile, transpose::N> a_desc(a);
        st_descriptor<scaled_tile, transpose::N> b_desc(b);
        uint64_t start_cycle = 0;
        if (leader) {
            tensor_before_thread_sync();
            start_cycle = clock64();
            g.start_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            if constexpr (
                C::MODE == DUAL_SCORE_COLLECTOR_NVFP4_THROUGHPUT) {
                issue_collector_nvfp4_k128_pair<0>(
                    output, output1, a_desc, b_desc, a_scale, b_scale);
                int iter = 2;
                for (; iter + 1 < g.iterations; iter += 2) {
                    issue_collector_nvfp4_k128_pair<1>(
                        output, output1, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_nvfp4_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (C::MODE == DUAL_CHAIN_NVFP4_THROUGHPUT) {
                issue_nvfp4_k128_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                issue_nvfp4_k128_iteration<0>(
                    output1, a_desc, b_desc, a_scale, b_scale);
                int iter = 2;
                for (; iter + 1 < g.iterations; iter += 2) {
                    issue_nvfp4_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_nvfp4_k128_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_nvfp4_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (
                C::MODE == TRIPLE_CHAIN_K192_K96_NVFP4_THROUGHPUT) {
                issue_nvfp4_k192_k96_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                issue_nvfp4_k192_k96_iteration<0>(
                    output1, a_desc, b_desc, a_scale, b_scale);
                issue_nvfp4_k192_k96_iteration<0>(
                    output2, a_desc, b_desc, a_scale, b_scale);
                int iter = 3;
                for (; iter + 2 < g.iterations; iter += 3) {
                    issue_nvfp4_k192_k96_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_nvfp4_k192_k96_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                    issue_nvfp4_k192_k96_iteration<1>(
                        output2, a_desc, b_desc, a_scale, b_scale);
                }
                if (iter < g.iterations) {
                    issue_nvfp4_k192_k96_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    ++iter;
                }
                if (iter < g.iterations) {
                    issue_nvfp4_k192_k96_iteration<1>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
            } else if constexpr (
                C::MODE == DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT ||
                C::MODE == DUAL_CHAIN_K192_K96_NVFP4_THROUGHPUT) {
                if constexpr (
                    C::MODE == DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT) {
                    issue_nvfp4_k192_k64_iteration<0>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_nvfp4_k192_k64_iteration<0>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                } else {
                    issue_nvfp4_k192_k96_iteration<0>(
                        output, a_desc, b_desc, a_scale, b_scale);
                    issue_nvfp4_k192_k96_iteration<0>(
                        output1, a_desc, b_desc, a_scale, b_scale);
                }
                int iter = 2;
                for (; iter + 1 < g.iterations; iter += 2) {
                    if constexpr (
                        C::MODE == DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT) {
                        issue_nvfp4_k192_k64_iteration<1>(
                            output, a_desc, b_desc, a_scale, b_scale);
                        issue_nvfp4_k192_k64_iteration<1>(
                            output1, a_desc, b_desc, a_scale, b_scale);
                    } else {
                        issue_nvfp4_k192_k96_iteration<1>(
                            output, a_desc, b_desc, a_scale, b_scale);
                        issue_nvfp4_k192_k96_iteration<1>(
                            output1, a_desc, b_desc, a_scale, b_scale);
                    }
                }
                if (iter < g.iterations) {
                    if constexpr (
                        C::MODE == DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT) {
                        issue_nvfp4_k192_k64_iteration<1>(
                            output, a_desc, b_desc, a_scale, b_scale);
                    } else {
                        issue_nvfp4_k192_k96_iteration<1>(
                            output, a_desc, b_desc, a_scale, b_scale);
                    }
                }
            } else {
                issue_nvfp4_k128_iteration<0>(
                    output, a_desc, b_desc, a_scale, b_scale);
                for (int iter = 1; iter < g.iterations; ++iter) {
                    issue_nvfp4_k128_iteration<1>(
                        output, a_desc, b_desc, a_scale, b_scale);
                }
            }
            detail::tcgen05::commit<1>(done);
        }
        if (issue_warp) {
            wait(done, 0);
            tensor_after_thread_sync();
        }
        if (leader) {
            const uint64_t end_cycle = clock64();
            g.end_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            g.cycles[blockIdx.x] = end_cycle - start_cycle;
            g.smids[blockIdx.x] = smid();
        }
    } else if constexpr (C::MODE == SPARSE_MXFP4_THROUGHPUT) {
        sparse_tile &a = allocator.allocate<sparse_tile>();
        scaled_tile &b = allocator.allocate<scaled_tile>();
        load_throughput_sparse_fp4_tile(a, g.a_fp4);
        load_throughput_dense_fp4_tile(b, g.b_fp4);
        full_tt_fp8e8m0<16> metadata =
            tm_alloc.template allocate<full_tt_fp8e8m0<16>>(128);
        scale_t a_scale = tm_alloc.template allocate<scale_t>(132);
        scale_t b_scale = tm_alloc.template allocate<scale_t>(148);
        const bool issue_warp = warpgroup::warpid() == 0;
        const bool leader = issue_warp && warp::elect_leader();
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        if (issue_warp) {
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                metadata, 0x44444444u);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                a_scale, 0x7f7f7f7fu);
            fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
                b_scale, 0x7f7f7f7fu);
            fp4pv_tmem_store_wait();
        }
        __syncthreads();
        st_descriptor<sparse_tile, transpose::N> a_desc(a);
        st_descriptor<scaled_tile, transpose::N> b_desc(b);
        uint64_t start_cycle = 0;
        if (leader) {
            tensor_before_thread_sync();
            start_cycle = clock64();
            g.start_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            issue_sparse_mxfp4_k128_iteration<0>(
                output, a_desc, b_desc, metadata.addr, a_scale, b_scale);
            for (int iter = 1; iter < g.iterations; ++iter) {
                issue_sparse_mxfp4_k128_iteration<1>(
                    output, a_desc, b_desc, metadata.addr, a_scale, b_scale);
            }
            detail::tcgen05::commit<1>(done);
        }
        if (issue_warp) {
            wait(done, 0);
            tensor_after_thread_sync();
        }
        if (leader) {
            const uint64_t end_cycle = clock64();
            g.end_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            g.cycles[blockIdx.x] = end_cycle - start_cycle;
            g.smids[blockIdx.x] = smid();
        }
    } else {
        sparse_tile &a = allocator.allocate<sparse_tile>();
        scaled_tile &b = allocator.allocate<scaled_tile>();
        load_throughput_sparse_fp4_tile(a, g.a_fp4);
        load_throughput_dense_fp4_tile(b, g.b_fp4);
        full_tt_fp8e8m0<16> metadata =
            tm_alloc.template allocate<full_tt_fp8e8m0<16>>(128);
        nv_scale_t a_scale = tm_alloc.template allocate<nv_scale_t>(132);
        nv_scale_t b_scale = tm_alloc.template allocate<nv_scale_t>(148);
        const bool issue_warp = warpgroup::warpid() == 0;
        const bool leader = issue_warp && warp::elect_leader();
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        if (issue_warp) {
            fill_throughput_tmem_128x16(metadata.addr, 0x44444444u);
            fill_throughput_tmem_128x16(a_scale.addr, 0x38383838u);
            fill_throughput_tmem_128x16(b_scale.addr, 0x38383838u);
            fp4pv_tmem_store_wait();
        }
        __syncthreads();
        st_descriptor<sparse_tile, transpose::N> a_desc(a);
        st_descriptor<scaled_tile, transpose::N> b_desc(b);
        uint64_t start_cycle = 0;
        if (leader) {
            tensor_before_thread_sync();
            start_cycle = clock64();
            g.start_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            issue_sparse_nvfp4_k128_iteration<0>(
                output, a_desc, b_desc, metadata.addr, a_scale, b_scale);
            if constexpr (
                C::MODE == SPLIT_SPARSE_NVFP4_THROUGHPUT) {
                issue_sparse_nvfp4_k128_iteration<1>(
                    output, a_desc, b_desc,
                    metadata.addr, a_scale, b_scale);
            }
            for (int iter = 1; iter < g.iterations; ++iter) {
                issue_sparse_nvfp4_k128_iteration<1>(
                    output, a_desc, b_desc, metadata.addr, a_scale, b_scale);
                if constexpr (
                    C::MODE == SPLIT_SPARSE_NVFP4_THROUGHPUT) {
                    issue_sparse_nvfp4_k128_iteration<1>(
                        output, a_desc, b_desc,
                        metadata.addr, a_scale, b_scale);
                }
            }
            detail::tcgen05::commit<1>(done);
        }
        if (issue_warp) {
            wait(done, 0);
            tensor_after_thread_sync();
        }
        if (leader) {
            const uint64_t end_cycle = clock64();
            g.end_globaltimer[blockIdx.x] =
                raw_fp4_throughput_globaltimer();
            g.cycles[blockIdx.x] = end_cycle - start_cycle;
            g.smids[blockIdx.x] = smid();
        }
    }
    __syncthreads();
}

template <int MODE>
static void launch_raw_fp4_throughput_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor a_bf16,
    at::Tensor b_bf16,
    at::Tensor cycles,
    at::Tensor start_globaltimer,
    at::Tensor end_globaltimer,
    at::Tensor smids,
    int iterations,
    int blocks
) {
    using C = config_raw_fp4_throughput_probe<MODE>;
    using G = globals_raw_fp4_throughput_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        reinterpret_cast<const bf16 *>(a_bf16.data_ptr()),
        reinterpret_cast<const bf16 *>(b_bf16.data_ptr()),
        reinterpret_cast<uint64_t *>(cycles.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(start_globaltimer.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(end_globaltimer.data_ptr<int64_t>()),
        smids.data_ptr<int32_t>(),
        iterations,
        blocks,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_throughput_probe<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_raw_fp4_throughput_probe<C>, g));
}

template <int N>
static void launch_raw_fp4_shape_throughput_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor a_bf16,
    at::Tensor b_bf16,
    at::Tensor cycles,
    at::Tensor start_globaltimer,
    at::Tensor end_globaltimer,
    at::Tensor smids,
    int iterations,
    int blocks
) {
    using C = config_raw_fp4_shape_throughput_probe<N>;
    using G = globals_raw_fp4_throughput_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        reinterpret_cast<const bf16 *>(a_bf16.data_ptr()),
        reinterpret_cast<const bf16 *>(b_bf16.data_ptr()),
        reinterpret_cast<uint64_t *>(cycles.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(start_globaltimer.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(end_globaltimer.data_ptr<int64_t>()),
        smids.data_ptr<int32_t>(),
        iterations,
        blocks,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_shape_throughput_probe<N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_raw_fp4_shape_throughput_probe<N>, g));
}

template <
    int RESIDENT_CTAS,
    bool NVFP4_QK = false,
    bool PRODUCTION_SCALE_COPY = false>
static void launch_raw_fp4_attention_n96_k96_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor a_bf16,
    at::Tensor b_bf16,
    at::Tensor cycles,
    at::Tensor start_globaltimer,
    at::Tensor end_globaltimer,
    at::Tensor smids,
    int iterations,
    int blocks
) {
    using C = config_raw_fp4_attention_n96_k96_probe<RESIDENT_CTAS>;
    using G = globals_raw_fp4_throughput_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        reinterpret_cast<const bf16 *>(a_bf16.data_ptr()),
        reinterpret_cast<const bf16 *>(b_bf16.data_ptr()),
        reinterpret_cast<uint64_t *>(cycles.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(start_globaltimer.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(end_globaltimer.data_ptr<int64_t>()),
        smids.data_ptr<int32_t>(),
        iterations,
        blocks,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_attention_n96_k96_probe<
            RESIDENT_CTAS, NVFP4_QK, PRODUCTION_SCALE_COPY>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_raw_fp4_attention_n96_k96_probe<
            RESIDENT_CTAS, NVFP4_QK, PRODUCTION_SCALE_COPY>,
        g));
}

template <int RESIDENT_CTAS>
static void launch_raw_fp4_attention_n128_split_qk_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor a_bf16,
    at::Tensor b_bf16,
    at::Tensor cycles,
    at::Tensor start_globaltimer,
    at::Tensor end_globaltimer,
    at::Tensor smids,
    int iterations,
    int blocks
) {
    using C = config_raw_fp4_attention_n96_k96_probe<RESIDENT_CTAS>;
    using G = globals_raw_fp4_throughput_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        reinterpret_cast<const bf16 *>(a_bf16.data_ptr()),
        reinterpret_cast<const bf16 *>(b_bf16.data_ptr()),
        reinterpret_cast<uint64_t *>(cycles.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(start_globaltimer.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(end_globaltimer.data_ptr<int64_t>()),
        smids.data_ptr<int32_t>(),
        iterations,
        blocks,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_attention_n128_split_qk_probe<RESIDENT_CTAS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_raw_fp4_attention_n128_split_qk_probe<RESIDENT_CTAS>, g));
}

template <int KEY_TILE>
static void launch_raw_fp4_attention_ultra_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor a_bf16,
    at::Tensor b_bf16,
    at::Tensor cycles,
    at::Tensor start_globaltimer,
    at::Tensor end_globaltimer,
    at::Tensor smids,
    int iterations,
    int blocks
) {
    using C = config_raw_fp4_attention_ultra_probe<KEY_TILE>;
    using G = globals_raw_fp4_throughput_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        reinterpret_cast<const bf16 *>(a_bf16.data_ptr()),
        reinterpret_cast<const bf16 *>(b_bf16.data_ptr()),
        reinterpret_cast<uint64_t *>(cycles.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(start_globaltimer.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(end_globaltimer.data_ptr<int64_t>()),
        smids.data_ptr<int32_t>(),
        iterations,
        blocks,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_attention_ultra_probe<KEY_TILE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_raw_fp4_attention_ultra_probe<KEY_TILE>, g));
}

static void launch_raw_fp4_attention_n256_dual_query_prefix_swap_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor a_bf16,
    at::Tensor b_bf16,
    at::Tensor cycles,
    at::Tensor start_globaltimer,
    at::Tensor end_globaltimer,
    at::Tensor smids,
    int iterations,
    int blocks
) {
    using C = config_raw_fp4_attention_ultra_probe<256>;
    using G = globals_raw_fp4_throughput_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        reinterpret_cast<const bf16 *>(a_bf16.data_ptr()),
        reinterpret_cast<const bf16 *>(b_bf16.data_ptr()),
        reinterpret_cast<uint64_t *>(cycles.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(start_globaltimer.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(end_globaltimer.data_ptr<int64_t>()),
        smids.data_ptr<int32_t>(),
        iterations,
        blocks,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_attention_n256_dual_query_prefix_swap_probe,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_raw_fp4_attention_n256_dual_query_prefix_swap_probe,
        g));
}

template <bool ALIAS_SCALE_SCRATCH>
static void launch_nvfp4_n128_split_qk_correctness(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor a_bf16,
    at::Tensor b_bf16,
    at::Tensor cycles,
    at::Tensor start_globaltimer,
    at::Tensor end_globaltimer,
    at::Tensor smids,
    int iterations,
    int blocks
) {
    using C = config_raw_fp4_attention_n96_k96_probe<1>;
    using G = globals_raw_fp4_throughput_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        reinterpret_cast<const bf16 *>(a_bf16.data_ptr()),
        reinterpret_cast<const bf16 *>(b_bf16.data_ptr()),
        reinterpret_cast<uint64_t *>(cycles.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(start_globaltimer.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t *>(end_globaltimer.data_ptr<int64_t>()),
        smids.data_ptr<int32_t>(),
        iterations,
        blocks,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_nvfp4_n128_split_qk_correctness<ALIAS_SCALE_SCRATCH>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_nvfp4_n128_split_qk_correctness<ALIAS_SCALE_SCRATCH>,
        g));
}

void dispatch_raw_fp4_throughput_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor a_bf16,
    at::Tensor b_bf16,
    at::Tensor cycles,
    at::Tensor start_globaltimer,
    at::Tensor end_globaltimer,
    at::Tensor smids,
    int mode,
    int iterations,
    int blocks
) {
    CHECK_INPUT(a_fp4);
    CHECK_INPUT(b_fp4);
    CHECK_INPUT(a_bf16);
    CHECK_INPUT(b_bf16);
    CHECK_INPUT(cycles);
    CHECK_INPUT(start_globaltimer);
    CHECK_INPUT(end_globaltimer);
    CHECK_INPUT(smids);
    TORCH_CHECK(a_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "a_fp4 must be packed E2M1");
    TORCH_CHECK(b_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "b_fp4 must be packed E2M1");
    TORCH_CHECK(a_bf16.dtype() == at::ScalarType::BFloat16,
                "a_bf16 must be BF16");
    TORCH_CHECK(b_bf16.dtype() == at::ScalarType::BFloat16,
                "b_bf16 must be BF16");
    TORCH_CHECK(a_fp4.dim() == 2 && a_fp4.size(0) >= 192 && a_fp4.size(1) == 64,
                "a_fp4 must have at least 192 packed rows and 64 columns");
    TORCH_CHECK(b_fp4.sizes() == a_fp4.sizes(),
                "b_fp4 must match a_fp4");
    TORCH_CHECK(a_bf16.dim() == 2 && a_bf16.size(0) == 128 && a_bf16.size(1) == 128,
                "a_bf16 must have shape [128,128]");
    TORCH_CHECK(b_bf16.sizes() == a_bf16.sizes(),
                "b_bf16 must match a_bf16");
    TORCH_CHECK(iterations >= 16 && iterations <= 4096,
                "iterations must be in [16,4096]");
    TORCH_CHECK(blocks >= 1 && blocks <= 4096,
                "blocks must be in [1,4096]");
    TORCH_CHECK(cycles.dtype() == at::ScalarType::Long
                && start_globaltimer.dtype() == at::ScalarType::Long
                && end_globaltimer.dtype() == at::ScalarType::Long,
                "timing tensors must be int64");
    TORCH_CHECK(smids.dtype() == at::ScalarType::Int,
                "smids must be int32");
    TORCH_CHECK(cycles.numel() == blocks
                && start_globaltimer.numel() == blocks
                && end_globaltimer.numel() == blocks
                && smids.numel() == blocks,
                "trace tensors must contain one record per block");
    if (mode == RAW_FP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<RAW_FP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == SCALED_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<SCALED_MXFP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == BF16_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<BF16_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == SPARSE_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<SPARSE_MXFP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == SPARSE_NVFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<SPARSE_NVFP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == SPLIT_SPARSE_NVFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            SPLIT_SPARSE_NVFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer,
                smids, iterations, blocks);
    } else if (mode == DENSE_NVFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<DENSE_NVFP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DEDICATED_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<DEDICATED_MXFP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DUAL_CHAIN_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<DUAL_CHAIN_MXFP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == TRIPLE_CHAIN_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<TRIPLE_CHAIN_MXFP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            DUAL_CHAIN_K192_K64_MXFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DUAL_CHAIN_K192_K96_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            DUAL_CHAIN_K192_K96_MXFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == TRIPLE_CHAIN_K192_K96_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            TRIPLE_CHAIN_K192_K96_MXFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == DUAL_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            DUAL_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == TRIPLE_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            TRIPLE_CHAIN_K192_K96_SM103_MXFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DUAL_SCORE_COLLECTOR_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            DUAL_SCORE_COLLECTOR_MXFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DUAL_SCORE_COLLECTOR_NVFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            DUAL_SCORE_COLLECTOR_NVFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DUAL_CHAIN_NVFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<DUAL_CHAIN_NVFP4_THROUGHPUT>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            DUAL_CHAIN_K192_K64_NVFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == DUAL_CHAIN_K192_K96_NVFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            DUAL_CHAIN_K192_K96_NVFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == TRIPLE_CHAIN_K192_K96_NVFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            TRIPLE_CHAIN_K192_K96_NVFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == PARALLEL_DUAL_CHAIN_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            PARALLEL_DUAL_CHAIN_MXFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == PARALLEL_TRIPLE_CHAIN_MXFP4_THROUGHPUT) {
        launch_raw_fp4_throughput_probe<
            PARALLEL_TRIPLE_CHAIN_MXFP4_THROUGHPUT>(
                a_fp4, b_fp4, a_bf16, b_bf16, cycles,
                start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == FOUR_CTA_N64_MXFP4_THROUGHPUT) {
        launch_raw_fp4_shape_throughput_probe<64>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == FOUR_CTA_N96_MXFP4_THROUGHPUT) {
        launch_raw_fp4_shape_throughput_probe<96>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == EIGHT_CTA_N32_MXFP4_THROUGHPUT) {
        launch_raw_fp4_shape_throughput_probe<32>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == ONE_CTA_N256_MXFP4_THROUGHPUT) {
        launch_raw_fp4_shape_throughput_probe<256>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == ONE_CTA_ATTENTION_N96_K96_MXFP4_THROUGHPUT) {
        launch_raw_fp4_attention_n96_k96_probe<1>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == TWO_CTA_ATTENTION_N96_K96_MXFP4_THROUGHPUT) {
        launch_raw_fp4_attention_n96_k96_probe<2>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == ONE_CTA_ATTENTION_N128_SPLIT_QK_MXFP4_THROUGHPUT) {
        launch_raw_fp4_attention_n128_split_qk_probe<1>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == TWO_CTA_ATTENTION_N128_SPLIT_QK_MXFP4_THROUGHPUT) {
        launch_raw_fp4_attention_n128_split_qk_probe<2>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (mode == NVFP4_N128_SPLIT_QK_CORRECTNESS) {
        launch_nvfp4_n128_split_qk_correctness<false>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == NVFP4_N128_SPLIT_QK_ALIASED_SCALE_CORRECTNESS) {
        launch_nvfp4_n128_split_qk_correctness<true>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == ONE_CTA_ATTENTION_N96_NVQK_K96_MXPV_THROUGHPUT) {
        launch_raw_fp4_attention_n96_k96_probe<1, true>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == TWO_CTA_ATTENTION_N96_NVQK_K96_MXPV_THROUGHPUT) {
        launch_raw_fp4_attention_n96_k96_probe<2, true>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == ONE_CTA_ATTENTION_N96_PRODUCTION_SCALE_COPY) {
        launch_raw_fp4_attention_n96_k96_probe<1, true, true>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == TWO_CTA_ATTENTION_N96_PRODUCTION_SCALE_COPY) {
        launch_raw_fp4_attention_n96_k96_probe<2, true, true>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == ONE_CTA_ATTENTION_N192_ULTRA_NVQK_MXPV_THROUGHPUT) {
        launch_raw_fp4_attention_ultra_probe<192>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode == ONE_CTA_ATTENTION_N256_ULTRA_NVQK_MXPV_THROUGHPUT) {
        launch_raw_fp4_attention_ultra_probe<256>(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else if (
        mode ==
        ONE_CTA_ATTENTION_N256_DUAL_QUERY_PREFIX_SWAP_THROUGHPUT) {
        launch_raw_fp4_attention_n256_dual_query_prefix_swap_probe(
            a_fp4, b_fp4, a_bf16, b_bf16, cycles,
            start_globaltimer, end_globaltimer, smids, iterations, blocks);
    } else {
        TORCH_CHECK(
            false,
            "mode must be raw=0, scaled=1, bf16=2, "
            "sparse_mxfp4=3, sparse_nvfp4=4, or "
            "split_sparse_nvfp4=5, dense_nvfp4=6, or "
            "dedicated_mxfp4=7, dual_chain_mxfp4=8, or "
            "triple_chain_mxfp4=9, dual_chain_k192_k64_mxfp4=10, or "
            "dual_chain_k192_k96_mxfp4=11, dual_chain_nvfp4=12, "
            "dual_chain_k192_k64_nvfp4=13, or "
            "dual_chain_k192_k96_nvfp4=14, "
            "parallel_dual_chain_mxfp4=15, or "
            "parallel_triple_chain_mxfp4=16, or "
            "four_cta_n64_mxfp4=17, four_cta_n96_mxfp4=18, or "
            "eight_cta_n32_mxfp4=19, one_cta_n256_mxfp4=20, "
            "one_cta_attention_n96_k96_mxfp4=21, or "
            "two_cta_attention_n96_k96_mxfp4=22, "
            "one_cta_attention_n128_split_qk_mxfp4=23, or "
            "two_cta_attention_n128_split_qk_mxfp4=24, or "
            "nvfp4_n128_split_qk_correctness=25, or "
            "nvfp4_n128_split_qk_aliased_scale_correctness=26, "
            "one_cta_attention_n96_nvqk_k96_mxpv=27, or "
            "two_cta_attention_n96_nvqk_k96_mxpv=28, or "
            "one_cta_attention_n96_production_scale_copy=29, or "
            "two_cta_attention_n96_production_scale_copy=30, "
            "dual_chain_k192_k96_sm103_mxfp4=33, or "
            "triple_chain_k192_k96_sm103_mxfp4=34, "
            "dual_score_collector_mxfp4=35, or "
            "dual_score_collector_nvfp4=36, or "
            "one_cta_attention_n192_ultra_nvqk_mxpv=37, or "
            "one_cta_attention_n256_ultra_nvqk_mxpv=38, or "
            "one_cta_attention_n256_dual_query_prefix_swap=39");
    }
}

#endif

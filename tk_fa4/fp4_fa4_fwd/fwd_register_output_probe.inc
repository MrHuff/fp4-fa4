#pragma once

#ifndef TK_FA4_REGISTER_OUTPUT_PROBE_BUILD
#define TK_FA4_REGISTER_OUTPUT_PROBE_BUILD 0
#endif

#if TK_FA4_REGISTER_OUTPUT_PROBE_BUILD

struct config_mxfp4_register_output_probe {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dvo = 128;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 512;
};

struct globals_mxfp4_register_output_probe {
    using payload_tile = st_fp4e2m1_2<128, 64>;
    using scale_tile = st_fp8e8m0<32, 16, false>;

    const uint8_t *p_fp4;
    const uint8_t *v_fp4;
    const int32_t *scale_words;
    const float *corrections;
    float *output;
    int64_t *trace;
    int partials;
    int blocks;

    __host__ __inline__ dim3 grid() const { return dim3(blocks); }
    __host__ __inline__ dim3 block() const {
        return dim3(config_mxfp4_register_output_probe::NUM_THREADS);
    }
    __host__ __inline__ int dynamic_shared_memory() const { return 48 * 1024; }
};

template <typename Tile>
__device__ __forceinline__ void register_output_probe_load_payload(
    Tile &tile,
    const uint8_t *source
) {
    static_assert(Tile::rows == 128 && Tile::cols == 64,
                  "register output probe payload must be M128xK128 packed FP4");
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    const int loader_rank = warpgroup::laneid();
    for (int linear = loader_rank;
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
__device__ __forceinline__ void register_output_probe_write_scale(
    ScaleTile &tile,
    const int32_t *source
) {
    const int lane = warp::laneid();
    const int base = lane * 4;
    fp4pv_store_mxfp4_scale_word4_swizzled(
        tile, lane,
        static_cast<uint32_t>(source[base + 0]),
        static_cast<uint32_t>(source[base + 1]),
        static_cast<uint32_t>(source[base + 2]),
        static_cast<uint32_t>(source[base + 3]));
}

template <int HALF>
__device__ __noinline__ void register_output_probe_run_consumer(
    const float *corrections,
    float *output,
    int64_t *trace,
    const int partials,
    const uint32_t partial0_addr,
    const uint32_t partial1_addr,
    semaphore *pv_done,
    semaphore *bank_reusable
) {
    static_assert(HALF == 0 || HALF == 1, "invalid output consumer half");
    float2 r00, r01, r02, r03, r04, r05, r06, r07;
    float2 r08, r09, r10, r11, r12, r13, r14, r15;
    float2 r16, r17, r18, r19, r20, r21, r22, r23;
    float2 r24, r25, r26, r27, r28, r29, r30, r31;

    wait(pv_done[0], 0);
    tensor_after_thread_sync();
#define TK_FA4_LOAD_INIT_X16(CHUNK, A0, A1, A2, A3, A4, A5, A6, A7) do { \
        constexpr int COL = HALF * 64 + CHUNK * 16; \
        const uint32_t address = partial0_addr \
            + ((warpgroup::warpid() * 32) << 16) + COL; \
        asm volatile( \
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 " \
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}" \
            : "=f"(A0.x), "=f"(A0.y), "=f"(A1.x), "=f"(A1.y), \
              "=f"(A2.x), "=f"(A2.y), "=f"(A3.x), "=f"(A3.y), \
              "=f"(A4.x), "=f"(A4.y), "=f"(A5.x), "=f"(A5.y), \
              "=f"(A6.x), "=f"(A6.y), "=f"(A7.x), "=f"(A7.y) \
            : "r"(address)); \
        tensor_load_wait(); \
    } while (0)
    TK_FA4_LOAD_INIT_X16(0, r00, r01, r02, r03, r04, r05, r06, r07);
    TK_FA4_LOAD_INIT_X16(1, r08, r09, r10, r11, r12, r13, r14, r15);
    TK_FA4_LOAD_INIT_X16(2, r16, r17, r18, r19, r20, r21, r22, r23);
    TK_FA4_LOAD_INIT_X16(3, r24, r25, r26, r27, r28, r29, r30, r31);
#undef TK_FA4_LOAD_INIT_X16
    tensor_before_thread_sync();
    if constexpr (HALF == 0) {
        warpgroup::sync(6);
    } else {
        warpgroup::sync(7);
    }
    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
        arrive(bank_reusable[0]);
    }

    #pragma unroll 1
    for (int partial_index = 1; partial_index < partials; ++partial_index) {
        const int slot = partial_index & 1;
        const int phase = (partial_index >> 1) & 1;
        wait(pv_done[slot], phase);
        tensor_after_thread_sync();
        const float correction =
            corrections[partial_index * 128 + warpgroup::laneid()];
        const float2 correction2 = {correction, correction};
        const uint32_t partial_addr = slot == 0 ? partial0_addr : partial1_addr;
#define TK_FA4_LOAD_FOLD_X16(CHUNK, A0, A1, A2, A3, A4, A5, A6, A7) do { \
        float2 v0, v1, v2, v3, v4, v5, v6, v7; \
        constexpr int COL = HALF * 64 + CHUNK * 16; \
        const uint32_t address = partial_addr \
            + ((warpgroup::warpid() * 32) << 16) + COL; \
        asm volatile( \
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 " \
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}" \
            : "=f"(v0.x), "=f"(v0.y), "=f"(v1.x), "=f"(v1.y), \
              "=f"(v2.x), "=f"(v2.y), "=f"(v3.x), "=f"(v3.y), \
              "=f"(v4.x), "=f"(v4.y), "=f"(v5.x), "=f"(v5.y), \
              "=f"(v6.x), "=f"(v6.y), "=f"(v7.x), "=f"(v7.y) \
            : "r"(address)); \
        tensor_load_wait(); \
        A0 = __fadd2_rn(__fmul2_rn(A0, correction2), v0); \
        A1 = __fadd2_rn(__fmul2_rn(A1, correction2), v1); \
        A2 = __fadd2_rn(__fmul2_rn(A2, correction2), v2); \
        A3 = __fadd2_rn(__fmul2_rn(A3, correction2), v3); \
        A4 = __fadd2_rn(__fmul2_rn(A4, correction2), v4); \
        A5 = __fadd2_rn(__fmul2_rn(A5, correction2), v5); \
        A6 = __fadd2_rn(__fmul2_rn(A6, correction2), v6); \
        A7 = __fadd2_rn(__fmul2_rn(A7, correction2), v7); \
    } while (0)
        TK_FA4_LOAD_FOLD_X16(0, r00, r01, r02, r03, r04, r05, r06, r07);
        TK_FA4_LOAD_FOLD_X16(1, r08, r09, r10, r11, r12, r13, r14, r15);
        TK_FA4_LOAD_FOLD_X16(2, r16, r17, r18, r19, r20, r21, r22, r23);
        TK_FA4_LOAD_FOLD_X16(3, r24, r25, r26, r27, r28, r29, r30, r31);
#undef TK_FA4_LOAD_FOLD_X16
        tensor_before_thread_sync();
        if constexpr (HALF == 0) {
            warpgroup::sync(6);
        } else {
            warpgroup::sync(7);
        }
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            arrive(bank_reusable[slot]);
        }
    }
    float *block_output =
        output + static_cast<int64_t>(blockIdx.x) * 128 * 128;
    float2 *row_output = reinterpret_cast<float2 *>(
        block_output + warpgroup::laneid() * 128 + HALF * 64);
#define TK_FA4_STORE_OUTPUT_REG(INDEX, VALUE) row_output[INDEX] = VALUE
    TK_FA4_STORE_OUTPUT_REG(0, r00); TK_FA4_STORE_OUTPUT_REG(1, r01);
    TK_FA4_STORE_OUTPUT_REG(2, r02); TK_FA4_STORE_OUTPUT_REG(3, r03);
    TK_FA4_STORE_OUTPUT_REG(4, r04); TK_FA4_STORE_OUTPUT_REG(5, r05);
    TK_FA4_STORE_OUTPUT_REG(6, r06); TK_FA4_STORE_OUTPUT_REG(7, r07);
    TK_FA4_STORE_OUTPUT_REG(8, r08); TK_FA4_STORE_OUTPUT_REG(9, r09);
    TK_FA4_STORE_OUTPUT_REG(10, r10); TK_FA4_STORE_OUTPUT_REG(11, r11);
    TK_FA4_STORE_OUTPUT_REG(12, r12); TK_FA4_STORE_OUTPUT_REG(13, r13);
    TK_FA4_STORE_OUTPUT_REG(14, r14); TK_FA4_STORE_OUTPUT_REG(15, r15);
    TK_FA4_STORE_OUTPUT_REG(16, r16); TK_FA4_STORE_OUTPUT_REG(17, r17);
    TK_FA4_STORE_OUTPUT_REG(18, r18); TK_FA4_STORE_OUTPUT_REG(19, r19);
    TK_FA4_STORE_OUTPUT_REG(20, r20); TK_FA4_STORE_OUTPUT_REG(21, r21);
    TK_FA4_STORE_OUTPUT_REG(22, r22); TK_FA4_STORE_OUTPUT_REG(23, r23);
    TK_FA4_STORE_OUTPUT_REG(24, r24); TK_FA4_STORE_OUTPUT_REG(25, r25);
    TK_FA4_STORE_OUTPUT_REG(26, r26); TK_FA4_STORE_OUTPUT_REG(27, r27);
    TK_FA4_STORE_OUTPUT_REG(28, r28); TK_FA4_STORE_OUTPUT_REG(29, r29);
    TK_FA4_STORE_OUTPUT_REG(30, r30); TK_FA4_STORE_OUTPUT_REG(31, r31);
#undef TK_FA4_STORE_OUTPUT_REG
    if (warpgroup::laneid() == 0) {
        const int64_t trace_base = static_cast<int64_t>(blockIdx.x) * 64;
        trace[trace_base + 25 + HALF] = partials;
    }
}

template <typename OutputTT>
__device__ __forceinline__ void register_output_probe_store_tmem(
    const OutputTT &output,
    float *destination
) {
    const int row = warpgroup::laneid();
    #pragma unroll
    for (int col = 0; col < OutputTT::cols; col += 16) {
        float2 values[8];
        const uint32_t address =
            output.addr + ((warpgroup::warpid() * 32) << 16) + col;
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
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            reinterpret_cast<float2 *>(destination + row * OutputTT::cols + col)[pair] =
                values[pair];
        }
    }
}

template <typename OutputTT>
__device__ __forceinline__ void register_output_probe_zero_tmem(
    OutputTT &output
) {
    #pragma unroll
    for (int col = 0; col < 128; col += 16) {
        const uint32_t address =
            output.addr + ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
            "{0f00000000,0f00000000,0f00000000,0f00000000,"
            "0f00000000,0f00000000,0f00000000,0f00000000,"
            "0f00000000,0f00000000,0f00000000,0f00000000,"
            "0f00000000,0f00000000,0f00000000,0f00000000};}"
            :: "r"(address));
    }
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(6);
}

#endif

// Register-resident D64 accumulator used by the isolated full attention
// route. Named values keep the live state scalarized; each x16 TMEM load is
// folded before the next chunk is issued, so no full D64 partial coexists
// with the accumulator.
template <int HALF>
struct fp4pv_live_register_output_accum {
    static_assert(HALF == 0 || HALF == 1, "invalid output half");
    float2 r00, r01, r02, r03, r04, r05, r06, r07;
    float2 r08, r09, r10, r11, r12, r13, r14, r15;
    float2 r16, r17, r18, r19, r20, r21, r22, r23;
    float2 r24, r25, r26, r27, r28, r29, r30, r31;

    template <int QUARTER>
    __device__ __forceinline__ void load_initial_quarter(
        uint32_t bank_addr
    ) {
        static_assert(QUARTER == 0 || QUARTER == 1,
            "D64 register accumulator has two D32 quarters");
#define TK_FA4_LIVE_LOAD_INIT(CHUNK, A0, A1, A2, A3, A4, A5, A6, A7) do { \
        constexpr int COL = HALF * 64 + CHUNK * 16; \
        const uint32_t address = bank_addr + \
            ((warpgroup::warpid() * 32) << 16) + COL; \
        asm volatile( \
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 " \
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}" \
            : "=f"(A0.x), "=f"(A0.y), "=f"(A1.x), "=f"(A1.y), \
              "=f"(A2.x), "=f"(A2.y), "=f"(A3.x), "=f"(A3.y), \
              "=f"(A4.x), "=f"(A4.y), "=f"(A5.x), "=f"(A5.y), \
              "=f"(A6.x), "=f"(A6.y), "=f"(A7.x), "=f"(A7.y) \
            : "r"(address)); \
        tensor_load_wait(); \
    } while (0)
        if constexpr (QUARTER == 0) {
            TK_FA4_LIVE_LOAD_INIT(
                0, r00, r01, r02, r03, r04, r05, r06, r07);
            TK_FA4_LIVE_LOAD_INIT(
                1, r08, r09, r10, r11, r12, r13, r14, r15);
        } else {
            TK_FA4_LIVE_LOAD_INIT(
                2, r16, r17, r18, r19, r20, r21, r22, r23);
            TK_FA4_LIVE_LOAD_INIT(
                3, r24, r25, r26, r27, r28, r29, r30, r31);
        }
#undef TK_FA4_LIVE_LOAD_INIT
    }

    __device__ __forceinline__ void load_initial(uint32_t bank_addr) {
        load_initial_quarter<0>(bank_addr);
        load_initial_quarter<1>(bank_addr);
    }

    template <int QUARTER>
    __device__ __forceinline__ void fold_quarter(
        uint32_t bank_addr,
        float correction
    ) {
        static_assert(QUARTER == 0 || QUARTER == 1,
            "D64 register accumulator has two D32 quarters");
        const float2 correction2 = {correction, correction};
#define TK_FA4_LIVE_LOAD_FOLD(CHUNK, A0, A1, A2, A3, A4, A5, A6, A7) do { \
        float2 v0, v1, v2, v3, v4, v5, v6, v7; \
        constexpr int COL = HALF * 64 + CHUNK * 16; \
        const uint32_t address = bank_addr + \
            ((warpgroup::warpid() * 32) << 16) + COL; \
        asm volatile( \
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 " \
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}" \
            : "=f"(v0.x), "=f"(v0.y), "=f"(v1.x), "=f"(v1.y), \
              "=f"(v2.x), "=f"(v2.y), "=f"(v3.x), "=f"(v3.y), \
              "=f"(v4.x), "=f"(v4.y), "=f"(v5.x), "=f"(v5.y), \
              "=f"(v6.x), "=f"(v6.y), "=f"(v7.x), "=f"(v7.y) \
            : "r"(address)); \
        tensor_load_wait(); \
        A0 = __fadd2_rn(__fmul2_rn(A0, correction2), v0); \
        A1 = __fadd2_rn(__fmul2_rn(A1, correction2), v1); \
        A2 = __fadd2_rn(__fmul2_rn(A2, correction2), v2); \
        A3 = __fadd2_rn(__fmul2_rn(A3, correction2), v3); \
        A4 = __fadd2_rn(__fmul2_rn(A4, correction2), v4); \
        A5 = __fadd2_rn(__fmul2_rn(A5, correction2), v5); \
        A6 = __fadd2_rn(__fmul2_rn(A6, correction2), v6); \
        A7 = __fadd2_rn(__fmul2_rn(A7, correction2), v7); \
    } while (0)
        if constexpr (QUARTER == 0) {
            TK_FA4_LIVE_LOAD_FOLD(
                0, r00, r01, r02, r03, r04, r05, r06, r07);
            TK_FA4_LIVE_LOAD_FOLD(
                1, r08, r09, r10, r11, r12, r13, r14, r15);
        } else {
            TK_FA4_LIVE_LOAD_FOLD(
                2, r16, r17, r18, r19, r20, r21, r22, r23);
            TK_FA4_LIVE_LOAD_FOLD(
                3, r24, r25, r26, r27, r28, r29, r30, r31);
        }
#undef TK_FA4_LIVE_LOAD_FOLD
    }

    __device__ __forceinline__ void fold(
        uint32_t bank_addr,
        float correction
    ) {
        fold_quarter<0>(bank_addr, correction);
        fold_quarter<1>(bank_addr, correction);
    }

    template <typename OTile>
    __device__ __forceinline__ void store_bf16(
        OTile &output,
        float output_scale,
        bool row_invalid
    ) {
        const float2 scale2 = row_invalid
            ? float2{0.0f, 0.0f}
            : float2{output_scale, output_scale};
#define TK_FA4_LIVE_SCALE(A) A = __fmul2_rn(A, scale2)
        TK_FA4_LIVE_SCALE(r00); TK_FA4_LIVE_SCALE(r01);
        TK_FA4_LIVE_SCALE(r02); TK_FA4_LIVE_SCALE(r03);
        TK_FA4_LIVE_SCALE(r04); TK_FA4_LIVE_SCALE(r05);
        TK_FA4_LIVE_SCALE(r06); TK_FA4_LIVE_SCALE(r07);
        TK_FA4_LIVE_SCALE(r08); TK_FA4_LIVE_SCALE(r09);
        TK_FA4_LIVE_SCALE(r10); TK_FA4_LIVE_SCALE(r11);
        TK_FA4_LIVE_SCALE(r12); TK_FA4_LIVE_SCALE(r13);
        TK_FA4_LIVE_SCALE(r14); TK_FA4_LIVE_SCALE(r15);
        TK_FA4_LIVE_SCALE(r16); TK_FA4_LIVE_SCALE(r17);
        TK_FA4_LIVE_SCALE(r18); TK_FA4_LIVE_SCALE(r19);
        TK_FA4_LIVE_SCALE(r20); TK_FA4_LIVE_SCALE(r21);
        TK_FA4_LIVE_SCALE(r22); TK_FA4_LIVE_SCALE(r23);
        TK_FA4_LIVE_SCALE(r24); TK_FA4_LIVE_SCALE(r25);
        TK_FA4_LIVE_SCALE(r26); TK_FA4_LIVE_SCALE(r27);
        TK_FA4_LIVE_SCALE(r28); TK_FA4_LIVE_SCALE(r29);
        TK_FA4_LIVE_SCALE(r30); TK_FA4_LIVE_SCALE(r31);
#undef TK_FA4_LIVE_SCALE

        constexpr int SUBTILE_COLS = OTile::swizzle_bytes / sizeof(bf16);
        constexpr uint32_t SWIZZLE_MASK = OTile::swizzle_bytes * 8 - 1;
        const int row = warpgroup::laneid();
        const uint32_t base_addr = __cvta_generic_to_shared(&output.data[0]);
        const uint32_t row_offset = row * SUBTILE_COLS * sizeof(bf16);
#define TK_FA4_LIVE_STORE_CHUNK(CHUNK, A0, A1, A2, A3, A4, A5, A6, A7) do { \
        constexpr int COL = HALF * 64 + CHUNK * 16; \
        const uint32_t row_base = base_addr + \
            (COL / SUBTILE_COLS) * (128 * SUBTILE_COLS * sizeof(bf16)) + \
            row_offset + (COL % SUBTILE_COLS) * sizeof(bf16); \
        const bf16_2 b0 = __float22bfloat162_rn(A0); \
        const bf16_2 b1 = __float22bfloat162_rn(A1); \
        const bf16_2 b2 = __float22bfloat162_rn(A2); \
        const bf16_2 b3 = __float22bfloat162_rn(A3); \
        const bf16_2 b4 = __float22bfloat162_rn(A4); \
        const bf16_2 b5 = __float22bfloat162_rn(A5); \
        const bf16_2 b6 = __float22bfloat162_rn(A6); \
        const bf16_2 b7 = __float22bfloat162_rn(A7); \
        const uint32_t addr0 = row_base; \
        const uint32_t addr1 = row_base + 16; \
        const uint32_t swizzled0 = addr0 ^ (((addr0 & SWIZZLE_MASK) >> 7) << 4); \
        const uint32_t swizzled1 = addr1 ^ (((addr1 & SWIZZLE_MASK) >> 7) << 4); \
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: \
            "r"(swizzled0), "r"(*(const uint32_t *)&b0), \
            "r"(*(const uint32_t *)&b1), "r"(*(const uint32_t *)&b2), \
            "r"(*(const uint32_t *)&b3)); \
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: \
            "r"(swizzled1), "r"(*(const uint32_t *)&b4), \
            "r"(*(const uint32_t *)&b5), "r"(*(const uint32_t *)&b6), \
            "r"(*(const uint32_t *)&b7)); \
    } while (0)
        TK_FA4_LIVE_STORE_CHUNK(0, r00, r01, r02, r03, r04, r05, r06, r07);
        TK_FA4_LIVE_STORE_CHUNK(1, r08, r09, r10, r11, r12, r13, r14, r15);
        TK_FA4_LIVE_STORE_CHUNK(2, r16, r17, r18, r19, r20, r21, r22, r23);
        TK_FA4_LIVE_STORE_CHUNK(3, r24, r25, r26, r27, r28, r29, r30, r31);
#undef TK_FA4_LIVE_STORE_CHUNK
    }
};

#if TK_FA4_REGISTER_OUTPUT_PROBE_BUILD

template <int MODE>
__launch_bounds__(config_mxfp4_register_output_probe::NUM_THREADS, 1)
__global__ void kernel_mxfp4_register_output_probe(
    const __grid_constant__ globals_mxfp4_register_output_probe g
) {
    using C = config_mxfp4_register_output_probe;
    using G = globals_mxfp4_register_output_probe;
    using output_t = full_tt_fl<128>;
    using scale_t = full_tt_fp8e8m0<16>;
    static_assert(MODE == 0 || MODE == 1, "invalid register output probe mode");
    static_assert(C::NUM_THREADS == 512, "four-role probe requires four WGs");
    static_assert(2 * 128 + 16 + 16 <= 512,
                  "register primitive TMEM map exceeds one allocation");
    static_assert(3 * 128 + 16 + 16 <= 512,
                  "TMEM merge control map exceeds one allocation");

    const int group = warpgroup::groupid();
    tensor_allocator<1, 1> tm_alloc{};
    output_t partial0 = tm_alloc.template allocate<output_t>(0);
    output_t partial1 = tm_alloc.template allocate<output_t>(128);
    output_t main_output = tm_alloc.template allocate<output_t>(256);
    constexpr int SCALE_BASE = MODE == 0 ? 256 : 384;
    scale_t p_scale_tm = tm_alloc.template allocate<scale_t>(SCALE_BASE);
    scale_t v_scale_tm = tm_alloc.template allocate<scale_t>(SCALE_BASE + 16);

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::payload_tile &p_smem =
        allocator.allocate<typename G::payload_tile>();
    typename G::payload_tile &v_smem =
        allocator.allocate<typename G::payload_tile>();
    typename G::scale_tile &p_scale_smem =
        allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &v_scale_smem =
        allocator.allocate<typename G::scale_tile>();

    __shared__ semaphore scale_copy_done;
    __shared__ semaphore pv_done[2];
    __shared__ semaphore bank_reusable[2];
    if (threadIdx.x == 0) {
        init_semaphore(scale_copy_done, 0, 1);
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            init_semaphore(pv_done[slot], 0, 1);
            init_semaphore(bank_reusable[slot], MODE == 0 ? 2 : 1, 0);
        }
    }

    if (group == 1) {
        register_output_probe_load_payload(p_smem, g.p_fp4);
        register_output_probe_load_payload(v_smem, g.v_fp4);
        if (warpgroup::warpid() == 0) {
            register_output_probe_write_scale(p_scale_smem, g.scale_words);
        } else if (warpgroup::warpid() == 1) {
            register_output_probe_write_scale(
                v_scale_smem, g.scale_words + 32 * 4);
        }
    }
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();

    const bool issue_warp = group == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    if (issue_lane) {
        load_mxnv_scale_async<1>(p_scale_tm, p_scale_smem);
        load_mxnv_scale_async<1>(v_scale_tm, v_scale_smem);
        detail::tcgen05::commit<1>(scale_copy_done);
    }
    if (issue_warp) {
        wait(scale_copy_done, 0);
        tensor_after_thread_sync();
        tensor_before_thread_sync();
    }

    const int64_t trace_base = static_cast<int64_t>(blockIdx.x) * 64;
    if (threadIdx.x == 0) {
        g.trace[trace_base + 0] = MODE;
        g.trace[trace_base + 1] = 128;
        g.trace[trace_base + 2] = 128;
        g.trace[trace_base + 3] = g.partials;
        g.trace[trace_base + 4] = partial0.addr;
        g.trace[trace_base + 5] = partial1.addr;
        g.trace[trace_base + 6] = main_output.addr;
        g.trace[trace_base + 7] = p_scale_tm.addr;
        g.trace[trace_base + 8] = v_scale_tm.addr;
        g.trace[trace_base + 9] = MODE == 0 ? 288 : 416;
    }

    if constexpr (MODE == 1) {
        if (group == 2) {
            register_output_probe_zero_tmem(main_output);
        }
        __syncthreads();
    }

    if (group == 0) {
        #pragma unroll 1
        for (int partial_index = 0; partial_index < g.partials; ++partial_index) {
            const int slot = partial_index & 1;
            const int phase = (partial_index >> 1) & 1;
            if (partial_index >= 2 && issue_lane) {
                wait(bank_reusable[slot], phase ^ 1);
            }
            if (issue_lane) {
                output_t &partial = slot == 0 ? partial0 : partial1;
                fp4pv_p_stage_ABt_split_k64<C, true>(
                    partial, p_smem, v_smem,
                    p_scale_tm, v_scale_tm, pv_done[slot]);
            }
        }
        if (issue_lane) {
            g.trace[trace_base + 24] = g.partials;
        }
    } else if constexpr (MODE == 0) {
        if (group == 2) {
            register_output_probe_run_consumer<0>(
                g.corrections, g.output, g.trace, g.partials,
                partial0.addr, partial1.addr, pv_done, bank_reusable);
        } else if (group == 3) {
            register_output_probe_run_consumer<1>(
                g.corrections, g.output, g.trace, g.partials,
                partial0.addr, partial1.addr, pv_done, bank_reusable);
        }
    } else {
        if (group == 2) {
            #pragma unroll 1
            for (int partial_index = 0;
                 partial_index < g.partials;
                 ++partial_index) {
                const int slot = partial_index & 1;
                const int phase = (partial_index >> 1) & 1;
                wait(pv_done[slot], phase);
                tensor_after_thread_sync();
                const float correction =
                    g.corrections[partial_index * 128 + warpgroup::laneid()];
                merge_dual_tt_output_accum_streaming<C>(
                    main_output, slot == 0 ? partial0 : partial1,
                    correction, partial_index != 0);
                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                    arrive(bank_reusable[slot]);
                }
            }
            float *block_output =
                g.output + static_cast<int64_t>(blockIdx.x) * 128 * 128;
            register_output_probe_store_tmem(main_output, block_output);
            if (warpgroup::laneid() == 0) {
                g.trace[trace_base + 25] = g.partials;
            }
        }
    }

    __syncthreads();
}

template <int MODE>
static void launch_mxfp4_register_output_probe(
    const globals_mxfp4_register_output_probe &g
) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_register_output_probe<MODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_mxfp4_register_output_probe<MODE>, g));
}

void dispatch_mxfp4_register_output_probe(
    at::Tensor p_fp4,
    at::Tensor v_fp4,
    at::Tensor scale_words,
    at::Tensor corrections,
    at::Tensor output,
    at::Tensor trace,
    int mode,
    int budget
) {
    for (const at::Tensor *tensor : {
             &p_fp4, &v_fp4, &scale_words, &corrections, &output, &trace}) {
        TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "register output probe tensors must be contiguous CUDA tensors");
    }
    TORCH_CHECK(p_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2
                    && p_fp4.sizes() == at::IntArrayRef({128, 64}),
                "P must be float4_e2m1fn_x2 (128,64)");
    TORCH_CHECK(v_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2
                    && v_fp4.sizes() == at::IntArrayRef({128, 64}),
                "V must be float4_e2m1fn_x2 (128,64)");
    TORCH_CHECK(scale_words.dtype() == at::ScalarType::Int
                    && scale_words.sizes() == at::IntArrayRef({2, 32, 4}),
                "scale_words must be int32 (2,32,4)");
    TORCH_CHECK(corrections.dtype() == at::ScalarType::Float
                    && corrections.dim() == 2 && corrections.size(1) == 128,
                "corrections must be float32 (partials,128)");
    TORCH_CHECK(output.dtype() == at::ScalarType::Float && output.dim() == 3
                    && output.size(1) == 128 && output.size(2) == 128,
                "output must be float32 (blocks,128,128)");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long && trace.dim() == 2
                    && trace.size(0) == output.size(0) && trace.size(1) >= 64,
                "trace must be int64 (blocks,>=64)");
    TORCH_CHECK(corrections.size(0) >= 2 && corrections.size(0) <= 64,
                "probe requires 2..64 rolling partials");
    TORCH_CHECK(mode == 0 || mode == 1,
                "mode must be 0 (register) or 1 (TMEM merge control)");
    TORCH_CHECK(budget == 0,
                "optimized four-role probe exposes only fixed 128-register roles");

    globals_mxfp4_register_output_probe g{
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v_fp4.data_ptr()),
        scale_words.data_ptr<int32_t>(),
        corrections.data_ptr<float>(),
        output.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        static_cast<int>(corrections.size(0)),
        static_cast<int>(output.size(0)),
    };
    if (mode == 0) {
        launch_mxfp4_register_output_probe<0>(g);
    } else {
        launch_mxfp4_register_output_probe<1>(g);
    }
}

#endif

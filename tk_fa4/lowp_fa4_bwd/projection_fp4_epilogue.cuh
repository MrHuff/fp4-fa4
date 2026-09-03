#pragma once

// Persistent SM100 NVFP4 learned projection with consumer-native epilogues.
// A Q/K-only compatibility path publishes the adaptive E2M1 layouts used by
// FA4 backward.  The unified QKV path additionally publishes the scale pages
// consumed by FP4 FA4 forward and a transposed MXFP4 V operand directly from
// the same 32x32 BF16-rounded register fragment.

#include <cuda/atomic>

#include "kittens.cuh"

namespace tkfa4_projection {

// A 68-word row stride advances four shared-memory banks per sequence row.
// The D128 RoPE fragment maps eight row groups x four adjacent rotary pairs
// onto one warp, so this makes those 32 scalar reads bank-disjoint.
static constexpr int PACKED_ROPE_SHARED_ROWS = 128;
static constexpr int PACKED_ROPE_SHARED_PAIRS = 64;
static constexpr int PACKED_ROPE_SHARED_STRIDE = 68;
// BF16 publication fragments are written by an 8-row x 4-pair lane map, then
// read with one lane per row.  Row-major storage with an odd 23-word stride
// leaves only three colliding banks during staging and is conflict-free for
// the row-wise FP8/MXFP4 consumers.  The former pair-major [16][33] layout
// incurred up to 4.7-way conflicts on the staging stores.
static constexpr int BF16_PAIR_SHARED_ROWS = 32;
static constexpr int BF16_PAIR_SHARED_STRIDE = 23;

template <
    int _LOAD_PIPE_DEPTH = 4,
    int _SUPERGROUP_SIZE = 4,
    int _QK_DEPTH = 192,
    int _KB = 256,
    bool _DENSE_FP8 = false
>
struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = false;
    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_THREADS = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS *
                                       kittens::WARP_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = _KB;
    static constexpr bool DENSE_FP8 = _DENSE_FP8;
    static constexpr int MMA_PER_TILE = Kb / 64;
    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int NUM_D_TILES = EPI_PIPE_DEPTH;
    static constexpr int QK_DEPTH = _QK_DEPTH;
    static_assert(QK_DEPTH == 128 || QK_DEPTH == 192);
    static_assert(DENSE_FP8 ? (Kb >= 64 && Kb % 64 == 0) : Kb == 256);
};

template <typename C>
struct globals {
    using A_tile = std::conditional_t<
        C::DENSE_FP8,
        kittens::st_fp8e4m3<C::Mb / 2, C::Kb>,
        kittens::st_fp4e2m1_2<C::Mb / 2, C::Kb / 2>
    >;
    using B_tile = std::conditional_t<
        C::DENSE_FP8,
        kittens::st_fp8e4m3<C::Nb / 2, C::Kb>,
        kittens::st_fp4e2m1_2<C::Nb / 2, C::Kb / 2>
    >;
    using A_sc_tile = kittens::st_hf<C::MMA_PER_TILE, 256, false>;
    using B_sc_tile = kittens::st_hf<C::MMA_PER_TILE, 256, false>;
    using D_tile = kittens::st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;
    using A_gl = std::conditional_t<
        C::DENSE_FP8,
        kittens::gl<kittens::fp8e4m3, 1, 1, -1, -1, A_tile>,
        kittens::gl<kittens::fp4e2m1_2, 1, 1, -1, -1, A_tile>
    >;
    using B_gl = std::conditional_t<
        C::DENSE_FP8,
        kittens::gl<kittens::fp8e4m3, 1, 1, -1, -1, B_tile>,
        kittens::gl<kittens::fp4e2m1_2, 1, 1, -1, -1, B_tile>
    >;
    using A_sc_gl = std::conditional_t<
        C::DENSE_FP8,
        kittens::gl<float, 1, 1, 1, -1>,
        kittens::gl<kittens::half, 1, -1, -1, 256, A_sc_tile>
    >;
    using B_sc_gl = std::conditional_t<
        C::DENSE_FP8,
        kittens::gl<float, 1, 1, 1, -1>,
        kittens::gl<kittens::half, 1, -1, -1, 256, B_sc_tile>
    >;
    using scale_gl = kittens::gl<float, 1, 1, 1, 1>;
    using D_gl = kittens::gl<kittens::bf16, 1, 1, -1, -1, D_tile>;

    A_gl A;
    A_sc_gl A_sc;
    scale_gl A_scale;
    B_gl B;
    B_sc_gl B_sc;
    scale_gl B_scale;
    D_gl Q;
    D_gl K;
    D_gl V;
    D_gl D;
    uint8_t *q_sequence_aligned;
    uint8_t *q_depth_packed;
    uint8_t *k_depth_aligned;
    uint8_t *k_depth_packed;
    uint8_t *q_sequence_compact;
    uint8_t *k_sequence_compact;
    uint8_t *q_forward_scales;
    uint8_t *k_forward_scales;
    float *q_forward_global_scale;
    float *k_forward_global_scale;
    uint8_t *v_mxfp4;
    uint8_t *v_mxfp4_scales;
    uint8_t *v_backward_mxfp4;
    uint8_t *v_backward_mxfp4_scales;
    uint8_t *v_backward_fp8;
    // Optional feature-major [B,H,D,S] mirror consumed directly by the
    // FP8-PV forward. Publishing it here avoids a standalone transpose of
    // the row-major [B,S,H,D] operand retained by backward.
    uint8_t *v_forward_fp8;
    uint8_t *q_backward_fp8;
    uint8_t *k_backward_fp8;
    const kittens::bf16 *attention_output;
    const float *lse;
    bool lse_head_major = false;
    float *dpsum;
    float *lse_log2;
    // Optional probability lift owned by the fused dO publisher.  CuTe D128
    // applies its own FP8 dS lift and leaves this at zero, whereas native TK
    // reconstructs the lifted probability directly from lstat and requests
    // +8 (= log2(256)).  D64 retains its historical fixed +8 contract.
    float dout_probability_log2_lift = 0.0f;
    uint4 *dq_clear = nullptr;
    int64_t dq_clear_vectors = 0;
    const float *adaptive_scales;
    const kittens::bf16 *rope_cos;
    const kittens::bf16 *rope_sin;
    const uint32_t *rope_packed;
    int batch;
    int seq_len;
    int heads;
    int head_depth = 128;
    // A D128 projection tile may hold two adjacent logical D64 heads.  The
    // GEMM geometry remains D128, while compact Q/K payloads and V metadata
    // are published directly in logical-head order for the D64 FA4 kernels.
    bool paired_d64 = false;
    int v_width;
    int v_scale_rows;
    // Share one E8M0 scale over each 32-sequence x 32-depth V tile.  The
    // replicated 1x32 metadata remains directly consumable by tcgen05 while
    // the same quantized V representation can be used in either orientation.
    bool v_mxfp4_scale_2d = true;
    int output_width;
    // Optional release counters for a concurrently produced A operand.
    // One word covers one 128-row x K256 tile, exactly matching an A load
    // issued by each CTA in the two-CTA projection cluster.
    const uint32_t *A_ready = nullptr;
    int A_ready_reduction_tiles = 0;
    int cluster_cap = 0;
    uint32_t A_ready_expected = 1u;
    int block_begin = 0;
    int block_end = 0;

    struct input_tiles_t {
        A_tile A;
        B_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct empty_input_scales_t {
        uint8_t unused;
    };
    using staged_input_scales_t = std::conditional_t<
        C::DENSE_FP8,
        empty_input_scales_t,
        input_scales_t
    >;
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        const int total_width = output_width > 0
            ? output_width
            : Q.cols() + K.cols() + v_width;
        const int total_blocks = (A.rows() / C::Mb) *
                                 (total_width / C::Nb);
        const int begin = max(0, min(block_begin, total_blocks));
        const int end = block_end > begin
            ? min(block_end, total_blocks)
            : total_blocks;
        const int blocks = end - begin;
        int clusters = min(
            blocks,
            kittens::num_sms() / C::CLUSTER_SIZE
        );
        if (cluster_cap > 0) {
            clusters = min(clusters, cluster_cap);
        }
        return dim3(max(clusters, 1) * C::CLUSTER_SIZE);
    }
    __host__ inline dim3 block() const {
        return dim3(C::NUM_THREADS);
    }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int bytes = sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH +
                              sizeof(staged_input_scales_t) *
                                  C::LOAD_PIPE_DEPTH +
                              sizeof(outputs_t) + 1024;
        static_assert(bytes <= kittens::MAX_SHARED_MEMORY - 4096);
        return bytes;
    }
};

__device__ __forceinline__ void wait_for_a_operand(
    const uint32_t *counter,
    uint32_t expected
) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(
        *const_cast<uint32_t *>(counter)
    );
    while (ready.load(cuda::memory_order_acquire) < expected) {
        __nanosleep(256);
    }
}

__device__ __forceinline__ uint8_t quantize_pair(
    kittens::bf16_2 pair,
    float scale
) {
    float2 values = __bfloat1622float2(pair);
    values.x *= scale;
    values.y *= scale;
    return std::bit_cast<uint8_t>(
        kittens::base_types::convertor<kittens::fp4e2m1_2, float2>::convert(
            values
        )
    );
}

// Convert four already-packed BF16 pairs as one scheduling unit.  This is
// bit-equivalent to four quantize_pair calls: each BF16 lane is widened to
// FP32, multiplied with round-to-nearest, and converted with the same native
// saturating E2M1 instruction.  Keeping the four independent conversion
// chains in one asm block lets ptxas overlap them and emits the final byte
// pack directly instead of maintaining four serial OR dependency chains.
__device__ __forceinline__ uint32_t quantize_four_bf16_pairs(
    uint32_t pair01,
    uint32_t pair23,
    uint32_t pair45,
    uint32_t pair67,
    float scale
) {
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;\n"
        "mov.b32 {b0, b1}, %1;\n"
        "mov.b32 {b2, b3}, %2;\n"
        "mov.b32 {b4, b5}, %3;\n"
        "mov.b32 {b6, b7}, %4;\n"
        ".reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;\n"
        "cvt.f32.bf16 f0, b0;\n"
        "cvt.f32.bf16 f1, b1;\n"
        "cvt.f32.bf16 f2, b2;\n"
        "cvt.f32.bf16 f3, b3;\n"
        "cvt.f32.bf16 f4, b4;\n"
        "cvt.f32.bf16 f5, b5;\n"
        "cvt.f32.bf16 f6, b6;\n"
        "cvt.f32.bf16 f7, b7;\n"
        "mul.rn.f32 f0, f0, %5;\n"
        "mul.rn.f32 f1, f1, %5;\n"
        "mul.rn.f32 f2, f2, %5;\n"
        "mul.rn.f32 f3, f3, %5;\n"
        "mul.rn.f32 f4, f4, %5;\n"
        "mul.rn.f32 f5, f5, %5;\n"
        "mul.rn.f32 f6, f6, %5;\n"
        "mul.rn.f32 f7, f7, %5;\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, f1, f0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, f3, f2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, f5, f4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, f7, f6;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(packed)
        : "r"(pair01), "r"(pair23), "r"(pair45), "r"(pair67),
          "f"(scale)
    );
    return packed;
}

// Transpose one packed 32x32 E2M1 tile entirely in warp registers.  Before
// this helper, lane d owns four words containing sequence values [0, 32) for
// depth d.  Afterwards, lane s owns four words containing depth values
// [0, 32) for sequence s.  Nibble order is preserved, so the result can be
// written directly to the row-major backward-V payload without decoding or a
// second E2M1 conversion.
__device__ __forceinline__ void transpose_mxfp4_32x32_nibbles(
    uint32_t (&words)[4]
) {
    const int lane = kittens::warp::laneid();
    const int lane_in_eight = lane & 7;

    #pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint32_t value = words[quarter];
        uint32_t peer = __shfl_xor_sync(0xffffffffu, value, 1, 8);
        value = (lane_in_eight & 1)
            ? (value & 0xf0f0f0f0u) | ((peer & 0xf0f0f0f0u) >> 4)
            : (value & 0x0f0f0f0fu) | ((peer & 0x0f0f0f0fu) << 4);
        peer = __shfl_xor_sync(0xffffffffu, value, 2, 8);
        value = (lane_in_eight & 2)
            ? (value & 0xff00ff00u) | ((peer & 0xff00ff00u) >> 8)
            : (value & 0x00ff00ffu) | ((peer & 0x00ff00ffu) << 8);
        peer = __shfl_xor_sync(0xffffffffu, value, 4, 8);
        words[quarter] = (lane_in_eight & 4)
            ? (value & 0xffff0000u) | ((peer & 0xffff0000u) >> 16)
            : (value & 0x0000ffffu) | ((peer & 0x0000ffffu) << 16);
    }

    // Transpose the distributed 4x4 quarter matrix.  Four cross-quarter
    // shuffles are sufficient because each off-diagonal pair exchanges once.
    const int destination_quarter = lane >> 3;
    const bool quarter_bit0 = (destination_quarter & 1) != 0;
    const uint32_t exchange01 = __shfl_xor_sync(
        0xffffffffu,
        quarter_bit0 ? words[0] : words[1],
        8
    );
    const uint32_t exchange23 = __shfl_xor_sync(
        0xffffffffu,
        quarter_bit0 ? words[2] : words[3],
        8
    );
    if (quarter_bit0) {
        words[0] = exchange01;
        words[2] = exchange23;
    } else {
        words[1] = exchange01;
        words[3] = exchange23;
    }

    const bool quarter_bit1 = (destination_quarter & 2) != 0;
    const uint32_t exchange02 = __shfl_xor_sync(
        0xffffffffu,
        quarter_bit1 ? words[0] : words[2],
        16
    );
    const uint32_t exchange13 = __shfl_xor_sync(
        0xffffffffu,
        quarter_bit1 ? words[1] : words[3],
        16
    );
    if (quarter_bit1) {
        words[0] = exchange02;
        words[1] = exchange13;
    } else {
        words[2] = exchange02;
        words[3] = exchange13;
    }
}

// Re-quantize eight values from the exact E4M3x2 pairs retained by the
// backward V publisher.  Those bytes represent 4 * V, so the caller supplies
// one quarter of the ordinary MX encode multiplier.  Finite E4M3 values and
// the complete multiplier range are exact in FP16, allowing the conversion
// to stay in packed half2 arithmetic instead of widening through FP32.
__device__ __forceinline__ uint32_t quantize_four_e4m3_pairs_to_mxfp4(
    uint16_t pair01,
    uint16_t pair23,
    uint16_t pair45,
    uint16_t pair67,
    uint32_t multiplier_half2
) {
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b32 half01, half23, half45, half67;\n"
        "cvt.rn.f16x2.e4m3x2 half01, %1;\n"
        "cvt.rn.f16x2.e4m3x2 half23, %2;\n"
        "cvt.rn.f16x2.e4m3x2 half45, %3;\n"
        "cvt.rn.f16x2.e4m3x2 half67, %4;\n"
        "mul.rn.f16x2 half01, half01, %5;\n"
        "mul.rn.f16x2 half23, half23, %5;\n"
        "mul.rn.f16x2 half45, half45, %5;\n"
        "mul.rn.f16x2 half67, half67, %5;\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 byte0, half01;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 byte1, half23;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 byte2, half45;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 byte3, half67;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(packed)
        : "h"(pair01), "h"(pair23), "h"(pair45), "h"(pair67),
          "r"(multiplier_half2)
    );
    return packed;
}

__device__ __forceinline__ uint16_t bf16_pair_amax_bits(
    kittens::bf16_2 pair
) {
    const uint32_t bits = *reinterpret_cast<const uint32_t *>(&pair);
    const uint16_t low = static_cast<uint16_t>(bits) & 0x7fffu;
    const uint16_t high = static_cast<uint16_t>(bits >> 16) & 0x7fffu;
    return max(low, high);
}

__device__ __forceinline__ float positive_bf16_bits_to_float(
    uint16_t bits
) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

// Convert an already-published E2M1 pair into the fixed-x4 E4M3 convention
// consumed by the dense backward kernels. Decode the retained code instead
// of independently requantizing the BF16 producer value, so forward and
// backward differentiate through one represented Q/K operand.
__device__ __forceinline__ uint16_t lift_fp4_pair_to_fp8(
    uint8_t code,
    float represented_decode_scale
) {
    const kittens::fp4e2m1_2 packed =
        std::bit_cast<kittens::fp4e2m1_2>(code);
    float2 values =
        kittens::base_types::convertor<float2, kittens::fp4e2m1_2>::convert(
            packed
        );
    values.x *= represented_decode_scale;
    values.y *= represented_decode_scale;
    const kittens::fp8e4m3_2 lifted =
        kittens::base_types::convertor<kittens::fp8e4m3_2, float2>::convert(
            values
        );
    return std::bit_cast<uint16_t>(lifted);
}

__device__ __forceinline__ kittens::bf16_2 apply_rope_pair(
    kittens::bf16_2 pair,
    kittens::bf16 cos_value,
    kittens::bf16 sin_value
) {
    // The projection fragment has already crossed its BF16 publication
    // boundary.  Keep the rotation in packed BF16 so one multiply plus one
    // FMA handles both coordinates without expanding eight live pairs to
    // FP32 registers.  This is the same precision exposed to attention.
    const uint32_t pair_bits =
        *reinterpret_cast<const uint32_t *>(&pair);
    const uint32_t perpendicular_bits =
        ((pair_bits << 16) | (pair_bits >> 16)) ^ 0x00008000u;
    const uint16_t cosine_bits =
        *reinterpret_cast<const uint16_t *>(&cos_value);
    const uint16_t sine_bits =
        *reinterpret_cast<const uint16_t *>(&sin_value);
    const uint32_t cosine_pair_bits =
        static_cast<uint32_t>(cosine_bits) * 0x00010001u;
    const uint32_t sine_pair_bits =
        static_cast<uint32_t>(sine_bits) * 0x00010001u;
    const kittens::bf16_2 cosine_pair =
        *reinterpret_cast<const kittens::bf16_2 *>(&cosine_pair_bits);
    const kittens::bf16_2 sine_pair =
        *reinterpret_cast<const kittens::bf16_2 *>(&sine_pair_bits);
    const kittens::bf16_2 perpendicular =
        *reinterpret_cast<const kittens::bf16_2 *>(&perpendicular_bits);
    return __hfma2(
        perpendicular,
        sine_pair,
        __hmul2(pair, cosine_pair)
    );
}

// Q/K weights for this specialization are converted once from the standard
// split-half rotary order [x_0 ... x_{D/2-1}, y_0 ... y_{D/2-1}] to adjacent
// pairs [x_0, y_0, x_1, y_1, ...].  One bf16_2 register therefore owns a
// complete rotary plane.  The common permutation is invisible to QK^T, while
// avoiding cross-slice TMEM traffic in this epilogue.
template <typename C, typename RT>
__device__ __forceinline__ void apply_rope_tile(
    const globals<C> &g,
    RT &tile,
    int global_row_base,
    int local_col_start
) {
    constexpr int kDepth = C::QK_DEPTH;
    constexpr int kRotaryPairs = kDepth / 2;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int lane_pair = lane & 3;
    const int row_in_octet = lane >> 2;
    const int depth_base = local_col_start % kDepth;
    const int pair_base = depth_base / 2;
    const int warp_row_base = global_row_base + warp * 32;

    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int pair0 = pair_base + j * 8 + lane_pair;
            const int row0 = warp_row_base + i * 16 + row_in_octet;
            const size_t rope00 =
                static_cast<size_t>(row0) * kRotaryPairs + pair0;
            const size_t rope08 =
                static_cast<size_t>(row0 + 8) * kRotaryPairs + pair0;
            const size_t rope40 = rope00 + 4;
            const size_t rope48 = rope08 + 4;
            tile.tiles[i][j].data[0] = apply_rope_pair(
                tile.tiles[i][j].data[0],
                g.rope_cos[rope00],
                g.rope_sin[rope00]
            );
            tile.tiles[i][j].data[1] = apply_rope_pair(
                tile.tiles[i][j].data[1],
                g.rope_cos[rope08],
                g.rope_sin[rope08]
            );
            tile.tiles[i][j].data[2] = apply_rope_pair(
                tile.tiles[i][j].data[2],
                g.rope_cos[rope40],
                g.rope_sin[rope40]
            );
            tile.tiles[i][j].data[3] = apply_rope_pair(
                tile.tiles[i][j].data[3],
                g.rope_cos[rope48],
                g.rope_sin[rope48]
            );
        }
    }
}

template <
    bool REUSE,
    bool PACKED_ROPE,
    bool SHARED_PACKED_ROPE,
    typename C,
    int CACHE_SIZE
>
__device__ __forceinline__ uint32_t load_rope_pair_cached(
    const globals<C> &g,
    size_t offset,
    uint32_t (&cache)[CACHE_SIZE],
    int slot,
    const uint32_t *packed_rope_tile
) {
    if constexpr (REUSE) {
        return cache[slot];
    } else {
        uint32_t packed;
        if constexpr (PACKED_ROPE) {
            if constexpr (SHARED_PACKED_ROPE) {
                packed = packed_rope_tile[offset];
            } else {
                packed = g.rope_packed[offset];
            }
        } else {
            const uint16_t cosine =
                *reinterpret_cast<const uint16_t *>(g.rope_cos + offset);
            const uint16_t sine =
                *reinterpret_cast<const uint16_t *>(g.rope_sin + offset);
            packed = static_cast<uint32_t>(cosine) |
                (static_cast<uint32_t>(sine) << 16);
        }
        cache[slot] = packed;
        return packed;
    }
}

// A D128 output tile contains exactly two adjacent heads.  Pair the TMEM
// slices from those heads and retain each packed cosine/sine value in a
// register between them.  The second head therefore performs no global RoPE
// table loads, without spending the shared memory needed by the MMA pipeline.
template <
    bool REUSE,
    bool PACKED_ROPE,
    bool SHARED_PACKED_ROPE,
    typename C,
    typename RT
>
__device__ __forceinline__ void apply_rope_tile_head_pair_cached(
    const globals<C> &g,
    RT &tile,
    int global_row_base,
    int local_col_start,
    uint32_t (&cache)[RT::height * RT::width * 4],
    const uint32_t *packed_rope_tile
) {
    static_assert(C::QK_DEPTH == 128);
    constexpr int kRotaryPairs = C::QK_DEPTH / 2;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int lane_pair = lane & 3;
    const int row_in_octet = lane >> 2;
    const int depth_base = local_col_start & 127;
    const int pair_base = depth_base / 2;
    const int warp_row_base = global_row_base + warp * 32;
    const int warp_local_row_base = warp * 32;

    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int pair0 = pair_base + j * 8 + lane_pair;
            const int row0 = SHARED_PACKED_ROPE
                ? warp_local_row_base + i * 16 + row_in_octet
                : warp_row_base + i * 16 + row_in_octet;
            constexpr int kRopeRowStride = SHARED_PACKED_ROPE
                ? PACKED_ROPE_SHARED_STRIDE
                : kRotaryPairs;
            const size_t rope00 =
                static_cast<size_t>(row0) * kRopeRowStride + pair0;
            const size_t rope08 = rope00 + 8 * kRopeRowStride;
            const size_t rope40 = rope00 + 4;
            const size_t rope48 = rope08 + 4;
            const int slot = (i * RT::width + j) * 4;
            const uint32_t packed00 = load_rope_pair_cached<
                REUSE,
                PACKED_ROPE,
                SHARED_PACKED_ROPE
            >(
                g,
                rope00,
                cache,
                slot,
                packed_rope_tile
            );
            const uint32_t packed08 = load_rope_pair_cached<
                REUSE,
                PACKED_ROPE,
                SHARED_PACKED_ROPE
            >(
                g,
                rope08,
                cache,
                slot + 1,
                packed_rope_tile
            );
            const uint32_t packed40 = load_rope_pair_cached<
                REUSE,
                PACKED_ROPE,
                SHARED_PACKED_ROPE
            >(
                g,
                rope40,
                cache,
                slot + 2,
                packed_rope_tile
            );
            const uint32_t packed48 = load_rope_pair_cached<
                REUSE,
                PACKED_ROPE,
                SHARED_PACKED_ROPE
            >(
                g,
                rope48,
                cache,
                slot + 3,
                packed_rope_tile
            );
            const kittens::bf16 *rope00_values =
                reinterpret_cast<const kittens::bf16 *>(&packed00);
            const kittens::bf16 *rope08_values =
                reinterpret_cast<const kittens::bf16 *>(&packed08);
            const kittens::bf16 *rope40_values =
                reinterpret_cast<const kittens::bf16 *>(&packed40);
            const kittens::bf16 *rope48_values =
                reinterpret_cast<const kittens::bf16 *>(&packed48);
            tile.tiles[i][j].data[0] = apply_rope_pair(
                tile.tiles[i][j].data[0],
                rope00_values[0],
                rope00_values[1]
            );
            tile.tiles[i][j].data[1] = apply_rope_pair(
                tile.tiles[i][j].data[1],
                rope08_values[0],
                rope08_values[1]
            );
            tile.tiles[i][j].data[2] = apply_rope_pair(
                tile.tiles[i][j].data[2],
                rope40_values[0],
                rope40_values[1]
            );
            tile.tiles[i][j].data[3] = apply_rope_pair(
                tile.tiles[i][j].data[3],
                rope48_values[0],
                rope48_values[1]
            );
        }
    }
}

template <typename C>
__device__ __forceinline__ void publish_sequence_compact(
    const globals<C> &g,
    uint8_t (&codes)[kittens::WARPGROUP_WARPS][16][33],
    int global_row_base,
    int local_col_start,
    bool is_k
) {
    constexpr int kDepth = C::QK_DEPTH;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int warp_row_base = global_row_base + warp * 32;
    const int batch_idx = warp_row_base / g.seq_len;
    const int seq_base = warp_row_base - batch_idx * g.seq_len;
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int global_depth = depth_base + lane;
    const int pair = lane >> 1;
    const bool high_depth = (lane & 1) != 0;
    uint8_t compact_bytes[16];
    #pragma unroll
    for (int packed_seq = 0; packed_seq < 16; ++packed_seq) {
        const uint8_t row0 = codes[warp][pair][2 * packed_seq];
        const uint8_t row1 = codes[warp][pair][2 * packed_seq + 1];
        const uint8_t bits = high_depth
            ? static_cast<uint8_t>(
                  ((row0 >> 4) & 0x0fu) | (row1 & 0xf0u)
              )
            : static_cast<uint8_t>(
                  (row0 & 0x0fu) | ((row1 & 0x0fu) << 4)
              );
#if TK_FA4_BWD_PURE_MXFP4_B8_STMATRIX_TRANSPOSE
        const int output_seq = is_k
            ? ((packed_seq >> 1) | ((packed_seq & 1) << 3))
            : packed_seq;
#else
        const int output_seq = packed_seq;
#endif
        compact_bytes[output_seq] = bits;
    }
    const int q_compact_depth = C::QK_DEPTH == 192
        ? (global_depth < 64
              ? global_depth
              : global_depth < 128
                  ? global_depth + 32
                  : global_depth < 160
                      ? global_depth - 64
                      : global_depth)
        : global_depth;
    const int output_depth = is_k ? global_depth : q_compact_depth;
    const int head_count = is_k
        ? g.K.cols() / kDepth
        : g.Q.cols() / kDepth;
    const size_t output_base =
        ((static_cast<size_t>(batch_idx) * head_count + head_idx) * kDepth +
         output_depth) * (g.seq_len / 2) + seq_base / 2;
    const uint4 vector = *reinterpret_cast<const uint4 *>(compact_bytes);
    *reinterpret_cast<uint4 *>(
        (is_k ? g.k_sequence_compact : g.q_sequence_compact) + output_base
    ) = vector;
}

template <typename RT>
__device__ __forceinline__ void stage_bf16_pairs(
    const RT &tile,
    uint32_t (&pairs)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE]
) {
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int lane_pair = lane & 3;
    const int row_in_octet = lane >> 2;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int pair0 = j * 8 + lane_pair;
            const int row0 = i * 16 + row_in_octet;
            pairs[warp][row0][pair0] =
                *reinterpret_cast<const uint32_t *>(
                    &tile.tiles[i][j].data[0]
                );
            pairs[warp][row0 + 8][pair0] =
                *reinterpret_cast<const uint32_t *>(
                    &tile.tiles[i][j].data[1]
                );
            pairs[warp][row0][pair0 + 4] =
                *reinterpret_cast<const uint32_t *>(
                    &tile.tiles[i][j].data[2]
                );
            pairs[warp][row0 + 8][pair0 + 4] =
                *reinterpret_cast<const uint32_t *>(
                    &tile.tiles[i][j].data[3]
                );
        }
    }
}

template <uint32_t ExponentDelta>
__device__ __forceinline__ uint16_t convert_scaled_bf16_pair_to_fp8(
    const kittens::bf16_2 &values
) {
    const uint32_t source =
        *reinterpret_cast<const uint32_t *>(&values);
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b32 scaled;\n"
        ".reg .b32 value0;\n"
        ".reg .b32 value1;\n"
        ".reg .b16 result;\n"
        "add.u32 scaled, %1, %2;\n"
        "shl.b32 value0, scaled, 16;\n"
        "and.b32 value1, scaled, 0xffff0000;\n"
        "cvt.rn.satfinite.e4m3x2.f32 result, value1, value0;\n"
        "cvt.u32.u16 %0, result;\n"
        "}\n"
        : "=r"(packed)
        : "r"(source), "n"(ExponentDelta)
    );
    return static_cast<uint16_t>(packed);
}

__device__ __forceinline__ float2 decode_fp8_pair_to_float2(
    uint16_t packed
) {
    uint32_t half2_bits;
    asm volatile(
        "cvt.rn.f16x2.e4m3x2 %0, %1;\n"
        : "=r"(half2_bits)
        : "h"(packed)
    );
    return __half22float2(
        *reinterpret_cast<const kittens::half_2 *>(&half2_bits)
    );
}

// Authenticated by the standalone E5M2 producer microgate.  Keep the exact
// x4 encode and byte-decoded statistic contract local to the O-dgrad shared
// epilogue; Q/K/V publication remains fixed E4M3.
__device__ __forceinline__ uint16_t encode_e5m2_pair_x4(
    kittens::bf16_2 source
) {
    float2 values = __bfloat1622float2(source);
    values.x *= 4.0f;
    values.y *= 4.0f;
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b16 result;\n"
        "cvt.rn.satfinite.e5m2x2.f32 result, %2, %1;\n"
        "cvt.u32.u16 %0, result;\n"
        "}\n"
        : "=r"(packed)
        : "f"(values.x), "f"(values.y)
    );
    return static_cast<uint16_t>(packed);
}

__device__ __forceinline__ float2 decode_e5m2_pair(
    uint16_t packed
) {
    uint32_t half2_bits;
    asm volatile(
        "cvt.rn.f16x2.e5m2x2 %0, %1;\n"
        : "=r"(half2_bits)
        : "h"(packed)
    );
    return __half22float2(
        *reinterpret_cast<const kittens::half_2 *>(&half2_bits)
    );
}

template <
    typename C,
    bool PUBLISH_FORWARD_FP8 = true,
    bool STAGE_BACKWARD_FP8_FOR_MXFP4 = false
>
__device__ __forceinline__ void publish_v_fp8(
    const globals<C> &g,
    uint32_t (&pairs)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE],
    int block_batch_idx,
    int block_seq_base,
    int local_col_start
) {
    const int kDepth = g.head_depth;
    constexpr uint32_t kScaleBf16PairDelta =
        (2u << 7) * 0x00010001u;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int seq_idx = block_seq_base + warp * 32 + lane;
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int v_heads = g.v_width / kDepth;
    uint32_t words[8];
    #pragma unroll
    for (int pair = 0; pair < 16; pair += 2) {
        const kittens::bf16_2 values0 =
            *reinterpret_cast<const kittens::bf16_2 *>(
                &pairs[warp][lane][pair]
            );
        const kittens::bf16_2 values1 =
            *reinterpret_cast<const kittens::bf16_2 *>(
                &pairs[warp][lane][pair + 1]
            );
        words[pair / 2] =
            static_cast<uint32_t>(
                convert_scaled_bf16_pair_to_fp8<kScaleBf16PairDelta>(
                    values0
                )
            ) |
            (static_cast<uint32_t>(
                 convert_scaled_bf16_pair_to_fp8<kScaleBf16PairDelta>(
                     values1
                 )
             ) << 16);
    }
    const size_t output_base =
        ((static_cast<size_t>(block_batch_idx) * g.seq_len + seq_idx) *
             v_heads +
         head_idx) * kDepth + depth_base;
    *reinterpret_cast<uint4 *>(g.v_backward_fp8 + output_base) =
        make_uint4(words[0], words[1], words[2], words[3]);
    *reinterpret_cast<uint4 *>(g.v_backward_fp8 + output_base + 16) =
        make_uint4(words[4], words[5], words[6], words[7]);

    if constexpr (PUBLISH_FORWARD_FP8) {
        // Preserve the existing forward-FP8 null contract before touching
        // scratch.  The stage-only specialization has this flag false and
        // therefore proceeds independently of v_forward_fp8.
        if (g.v_forward_fp8 == nullptr) {
            return;
        }
    }
    if constexpr (PUBLISH_FORWARD_FP8 || STAGE_BACKWARD_FP8_FOR_MXFP4) {
        // Transpose this warp's 32 sequence rows x 32 depth bytes through the
        // staging fragment. The QKV path is finished with the BF16 fragment
        // at this point, so its first eight words per row can hold the exact
        // E4M3 bytes just published above. Each lane then reads one depth
        // column and emits 32 adjacent sequence bytes. This avoids both a
        // second conversion and a standalone transpose kernel.
        #pragma unroll
        for (int word = 0; word < 8; ++word) {
            pairs[warp][lane][word] = words[word];
        }
    }

    if constexpr (PUBLISH_FORWARD_FP8) {
        __syncwarp();
        const int depth = depth_base + lane;
        const int seq_base = block_seq_base + warp * 32;
        const size_t forward_base =
            ((static_cast<size_t>(block_batch_idx) * v_heads + head_idx) *
                 kDepth +
             depth) * g.seq_len + seq_base;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint32_t feature_words[4] = {0, 0, 0, 0};
            #pragma unroll
            for (int word = 0; word < 4; ++word) {
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const int row = half * 16 + word * 4 + byte;
                    const uint32_t source_word =
                        pairs[warp][row][lane >> 2];
                    const uint32_t value =
                        (source_word >> ((lane & 3) * 8)) & 0xffu;
                    feature_words[word] |= value << (byte * 8);
                }
            }
            *reinterpret_cast<uint4 *>(
                g.v_forward_fp8 + forward_base + half * 16
            ) = make_uint4(
                feature_words[0],
                feature_words[1],
                feature_words[2],
                feature_words[3]
            );
        }
    }
}

template <typename C>
__device__ __forceinline__ void publish_qk_fp8(
    const globals<C> &g,
    uint32_t (&pairs)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE],
    int block_batch_idx,
    int block_seq_base,
    int local_col_start,
    bool is_k
) {
    constexpr int kDepth = C::QK_DEPTH;
    constexpr uint32_t kScaleBf16PairDelta =
        (2u << 7) * 0x00010001u;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int seq_idx = block_seq_base + warp * 32 + lane;
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int head_count = is_k
        ? g.K.cols() / kDepth
        : g.Q.cols() / kDepth;
    uint32_t words[8];
    #pragma unroll
    for (int pair = 0; pair < 16; pair += 2) {
        const kittens::bf16_2 values0 =
            *reinterpret_cast<const kittens::bf16_2 *>(
                &pairs[warp][lane][pair]
            );
        const kittens::bf16_2 values1 =
            *reinterpret_cast<const kittens::bf16_2 *>(
                &pairs[warp][lane][pair + 1]
            );
        words[pair / 2] =
            static_cast<uint32_t>(
                convert_scaled_bf16_pair_to_fp8<kScaleBf16PairDelta>(
                    values0
                )
            ) |
            (static_cast<uint32_t>(
                 convert_scaled_bf16_pair_to_fp8<kScaleBf16PairDelta>(
                     values1
                 )
             ) << 16);
    }
    const size_t output_base =
        ((static_cast<size_t>(block_batch_idx) * g.seq_len + seq_idx) *
             head_count +
         head_idx) * kDepth + depth_base;
    uint8_t *output = is_k ? g.k_backward_fp8 : g.q_backward_fp8;
    *reinterpret_cast<uint4 *>(output + output_base) =
        make_uint4(words[0], words[1], words[2], words[3]);
    *reinterpret_cast<uint4 *>(output + output_base + 16) =
        make_uint4(words[4], words[5], words[6], words[7]);
}

template <typename C>
__device__ __forceinline__ void publish_qk_fp8_from_codes(
    const globals<C> &g,
    uint8_t (&codes)[kittens::WARPGROUP_WARPS][16][33],
    int block_batch_idx,
    int block_seq_base,
    int local_col_start,
    bool is_k,
    float quantize_scale
) {
    constexpr int kDepth = C::QK_DEPTH;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int seq_idx = block_seq_base + warp * 32 + lane;
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int head_count = is_k
        ? g.K.cols() / kDepth
        : g.Q.cols() / kDepth;

    // Forward reconstructs code / quantize_scale. Backward's fixed E4M3
    // descriptor applies 0.25, so publish the same value lifted by four.
    const float represented_decode_scale = 4.0f / quantize_scale;
    uint32_t words[8];
    #pragma unroll
    for (int pair = 0; pair < 16; pair += 2) {
        const uint16_t values0 = lift_fp4_pair_to_fp8(
            codes[warp][pair][lane],
            represented_decode_scale
        );
        const uint16_t values1 = lift_fp4_pair_to_fp8(
            codes[warp][pair + 1][lane],
            represented_decode_scale
        );
        words[pair / 2] = static_cast<uint32_t>(values0) |
            (static_cast<uint32_t>(values1) << 16);
    }
    const size_t output_base =
        ((static_cast<size_t>(block_batch_idx) * g.seq_len + seq_idx) *
             head_count +
         head_idx) * kDepth + depth_base;
    uint8_t *output = is_k ? g.k_backward_fp8 : g.q_backward_fp8;
    *reinterpret_cast<uint4 *>(output + output_base) =
        make_uint4(words[0], words[1], words[2], words[3]);
    *reinterpret_cast<uint4 *>(output + output_base + 16) =
        make_uint4(words[4], words[5], words[6], words[7]);
}

template <typename C>
__device__ __forceinline__ void publish_qk_fp8_from_perblock_codes(
    const globals<C> &g,
    uint8_t (&codes)[kittens::WARPGROUP_WARPS][16][33],
    uint8_t (&block_scales)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][2],
    int block_batch_idx,
    int block_seq_base,
    int local_col_start,
    bool is_k
) {
    constexpr int kDepth = C::QK_DEPTH;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int seq_idx = block_seq_base + warp * 32 + lane;
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int head_count = is_k
        ? g.K.cols() / kDepth
        : g.Q.cols() / kDepth;

    uint32_t words[8];
    #pragma unroll
    for (int pair = 0; pair < 16; pair += 2) {
        const uint8_t scale_bits = block_scales[warp][lane][pair >> 3];
        const kittens::fp8e4m3 encoded_scale =
            std::bit_cast<kittens::fp8e4m3>(scale_bits);
        const float represented_decode_scale = 4.0f *
            kittens::base_types::convertor<
                float,
                kittens::fp8e4m3
            >::convert(encoded_scale);
        const uint16_t values0 = lift_fp4_pair_to_fp8(
            codes[warp][pair][lane],
            represented_decode_scale
        );
        const uint16_t values1 = lift_fp4_pair_to_fp8(
            codes[warp][pair + 1][lane],
            represented_decode_scale
        );
        words[pair / 2] = static_cast<uint32_t>(values0) |
            (static_cast<uint32_t>(values1) << 16);
    }
    const size_t output_base =
        ((static_cast<size_t>(block_batch_idx) * g.seq_len + seq_idx) *
             head_count +
         head_idx) * kDepth + depth_base;
    uint8_t *output = is_k ? g.k_backward_fp8 : g.q_backward_fp8;
    *reinterpret_cast<uint4 *>(output + output_base) =
        make_uint4(words[0], words[1], words[2], words[3]);
    *reinterpret_cast<uint4 *>(output + output_base + 16) =
        make_uint4(words[4], words[5], words[6], words[7]);
}

template <
    typename C,
    bool PublishStats,
    bool NegateStats = false,
    bool PUBLISH_DOUT_E5M2 = false
>
__device__ __forceinline__ void publish_v_fp8_from_output_shared(
    const globals<C> &g,
    const typename globals<C>::D_tile &tile,
    int global_row_base,
    int local_col_start,
    float *stats_accumulators
) {
    const int kDepth = g.head_depth;
    constexpr uint32_t kScaleBf16PairDelta =
        (2u << 7) * 0x00010001u;
    static_assert(globals<C>::D_tile::cols == 32);
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int row_in_octet = lane >> 2;
    const int depth_chunk = lane & 3;
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int v_heads = g.v_width / kDepth;
    #pragma unroll
    for (int row_octet = 0; row_octet < 4; ++row_octet) {
        const int local_row =
            warp * 32 + row_octet * 8 + row_in_octet;
        const int row = global_row_base + local_row;
        const int batch_idx = row / g.seq_len;
        const int seq_idx = row - batch_idx * g.seq_len;
        const int2 source_coord = {local_row, depth_chunk * 8};
        const uint4 packed_bf16 =
            *reinterpret_cast<const uint4 *>(&tile[source_coord]);
        uint4 packed_output;
        if constexpr (PublishStats) {
            const size_t output_base =
                ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
                     v_heads +
                 head_idx) * kDepth +
                depth_base + depth_chunk * 8;
            packed_output = *reinterpret_cast<const uint4 *>(
                g.attention_output + output_base
            );
        }
        uint32_t words[2];
        float stats_partial = 0.0f;
        #pragma unroll
        for (int pair = 0; pair < 4; pair += 2) {
            const kittens::bf16_2 values0 =
                *reinterpret_cast<const kittens::bf16_2 *>(
                    reinterpret_cast<const uint32_t *>(&packed_bf16) + pair
                );
            const kittens::bf16_2 values1 =
                *reinterpret_cast<const kittens::bf16_2 *>(
                    reinterpret_cast<const uint32_t *>(&packed_bf16) + pair + 1
                );
            uint16_t fp8_values0;
            uint16_t fp8_values1;
            if constexpr (PUBLISH_DOUT_E5M2) {
                fp8_values0 = encode_e5m2_pair_x4(values0);
                fp8_values1 = encode_e5m2_pair_x4(values1);
            } else {
                fp8_values0 =
                    convert_scaled_bf16_pair_to_fp8<kScaleBf16PairDelta>(
                        values0
                    );
                fp8_values1 =
                    convert_scaled_bf16_pair_to_fp8<kScaleBf16PairDelta>(
                        values1
                    );
            }
            words[pair / 2] = static_cast<uint32_t>(fp8_values0) |
                (static_cast<uint32_t>(fp8_values1) << 16);
            if constexpr (PublishStats) {
                const kittens::bf16_2 output0 =
                    *reinterpret_cast<const kittens::bf16_2 *>(
                        reinterpret_cast<const uint32_t *>(&packed_output) +
                        pair
                    );
                const kittens::bf16_2 output1 =
                    *reinterpret_cast<const kittens::bf16_2 *>(
                        reinterpret_cast<const uint32_t *>(&packed_output) +
                        pair + 1
                    );
                // dP consumes these exact fixed-scale E4M3 values.  Deriving
                // dPsum from the pre-quantized BF16 fragment breaks softmax
                // centering as soon as E4M3 rounds or saturates.  Decode the
                // bytes we publish so both sides of dP - dPsum use one
                // numerical representation.
                float2 dout_values0;
                float2 dout_values1;
                if constexpr (PUBLISH_DOUT_E5M2) {
                    dout_values0 = decode_e5m2_pair(fp8_values0);
                    dout_values1 = decode_e5m2_pair(fp8_values1);
                } else {
                    dout_values0 = decode_fp8_pair_to_float2(fp8_values0);
                    dout_values1 = decode_fp8_pair_to_float2(fp8_values1);
                }
                const float2 output_values0 =
                    __bfloat1622float2(output0);
                const float2 output_values1 =
                    __bfloat1622float2(output1);
                stats_partial = fmaf(
                    output_values0.x,
                    dout_values0.x,
                    stats_partial
                );
                stats_partial = fmaf(
                    output_values0.y,
                    dout_values0.y,
                    stats_partial
                );
                stats_partial = fmaf(
                    output_values1.x,
                    dout_values1.x,
                    stats_partial
                );
                stats_partial = fmaf(
                    output_values1.y,
                    dout_values1.y,
                    stats_partial
                );
            }
        }
        const size_t output_base =
            ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
                 v_heads +
             head_idx) * kDepth +
            depth_base + depth_chunk * 8;
        // The internal globals field is an untyped byte destination. The
        // distinct v509 raw API supplies genuine Float8_e5m2 slot-7 storage;
        // every legacy call supplies the retained Float8_e4m3fn tensor.
        *reinterpret_cast<uint2 *>(g.v_backward_fp8 + output_base) =
            make_uint2(words[0], words[1]);
        if constexpr (PublishStats) {
            stats_accumulators[row_octet] += stats_partial;
            if (depth_base == kDepth - 32) {
                float row_sum = stats_accumulators[row_octet];
                row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, 4);
                row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, 4);
                if (depth_chunk == 0) {
                    const size_t stats_offset =
                        (static_cast<size_t>(batch_idx) * v_heads +
                         head_idx) * g.seq_len + seq_idx;
                    const size_t lse_offset = g.lse_head_major
                        ? (static_cast<size_t>(batch_idx) * v_heads +
                           head_idx) * g.seq_len + seq_idx
                        : (static_cast<size_t>(batch_idx) * g.seq_len +
                           seq_idx) * v_heads + head_idx;
                    // Published E4M3 or E5M2 dO already carries the x4
                    // operand lift. V carries the same x4 lift in dP, so the
                    // remaining factor matching the x16 accumulator is x4.
                    constexpr float dpsum_scale =
                        NegateStats ? -4.0f : 4.0f;
                    constexpr float lse_scale = NegateStats
                        ? -1.4426950408889634f
                        : 1.4426950408889634f;
                    const float dpsum_value = row_sum * dpsum_scale;
                    // D64 FP8 backward consumes a probability lifted by 2^8.
                    // Publish that offset in log space while the projection
                    // epilogue already owns the row statistic. The add folds
                    // into the existing multiply and disappears from every
                    // score pair in the attention mainloop.
                    const float probability_log2_lift = NegateStats
                        ? (kDepth == 64
                               ? 8.0f
                               : g.dout_probability_log2_lift)
                        : 0.0f;
                    const float lse_value = fmaf(
                        g.lse[lse_offset],
                        lse_scale,
                        probability_log2_lift
                    );
                    g.dpsum[stats_offset] = dpsum_value;
                    g.lse_log2[stats_offset] = lse_value;
                }
                stats_accumulators[row_octet] = 0.0f;
            }
        }
    }
}

__device__ __forceinline__ uint8_t bf16_amax_to_e8m0_rte(
    uint16_t absolute_bits
) {
    const uint8_t exponent =
        static_cast<uint8_t>((absolute_bits >> 7) & 0xffu);
    if (exponent == 0) {
        return 0;
    }
    const uint16_t mantissa = absolute_bits & 0x7fu;
    const bool round_up =
        (mantissa > 0x40u) ||
        (mantissa == 0x40u && (exponent & 1u));
    return static_cast<uint8_t>(
        round_up && exponent < 0xfeu ? exponent + 1 : exponent
    );
}

__device__ __forceinline__ uint8_t bf16_amax_to_e8m0_1d_mse(
    uint16_t absolute_bits
) {
    const uint8_t exponent =
        static_cast<uint8_t>((absolute_bits >> 7) & 0xffu);
    if (exponent == 0) {
        return 0;
    }
    // The 1x32 projected-V sweep selects the upper exponent beginning at
    // normalized BF16 amax 1.203125 (mantissa 0x1a).
    const bool round_up = (absolute_bits & 0x7fu) >= 0x1au;
    return static_cast<uint8_t>(
        round_up && exponent < 0xfeu ? exponent + 1 : exponent
    );
}

__device__ __forceinline__ float e8m0_encode_multiplier(uint8_t exponent) {
    if (exponent == 0) {
        return 0.0f;
    }
    const uint32_t reciprocal_bits =
        static_cast<uint32_t>(254 - exponent) << 23;
    return 6.0f * std::bit_cast<float>(reciprocal_bits);
}

// Convert the positive magnitude code selected from backward's E4M3(x4)
// publication to the exact BF16 bits of decode(E4M3) / 4.  Finite positive
// E4M3 codes are monotonic, so a whole MX scale group can select its amax on
// bytes before paying for this integer conversion.  The derived publisher
// handles E4M3FN's sole NaN magnitude with the E8M0 sentinel before calling
// this finite-code helper; retain a defensive satfinite clamp here.
__device__ __forceinline__ uint16_t e4m3_x4_amax_to_logical_bf16_bits(
    uint8_t magnitude_code
) {
    const uint32_t code = magnitude_code > 0x7eu
        ? 0x7eu
        : static_cast<uint32_t>(magnitude_code);
    const uint32_t exponent = code >> 3;
    const uint32_t mantissa = code & 7u;
    if (exponent != 0) {
        // E4M3 bias is seven. Undoing the publisher's x4 lift shifts the
        // decoded exponent by -2, hence BF16 biased exponent e + 118.
        return static_cast<uint16_t>(
            ((exponent + 118u) << 7) | (mantissa << 4)
        );
    }
    if (mantissa == 0) {
        return 0;
    }
    // E4M3 subnormals are m * 2^-9; after undoing x4 they are m * 2^-11
    // and are still normal BF16 values.
    const uint32_t highest_bit =
        31u - static_cast<uint32_t>(__clz(mantissa));
    return static_cast<uint16_t>(
        ((116u + highest_bit) << 7) |
        ((mantissa - (1u << highest_bit)) << (7u - highest_bit))
    );
}

__device__ __forceinline__ uint32_t
e8m0_e4m3_x4_encode_multiplier_half2(uint8_t exponent) {
    if (exponent == 0 || exponent == 0xffu) {
        return 0;
    }
    // Ordinary MX encoding uses 6 * 2^(127-e).  The source E4M3 byte already
    // represents 4 * V, so multiply it by exactly 1.5 * 2^(127-e).
    const uint32_t half_bits =
        ((142u - static_cast<uint32_t>(exponent)) << 10) | 0x200u;
    return half_bits | (half_bits << 16);
}

__device__ __forceinline__ float e8m0_pow2_encode_multiplier(
    uint8_t exponent
) {
    if (exponent == 0) {
        return 0.0f;
    }
    const uint32_t reciprocal_bits =
        static_cast<uint32_t>(254 - exponent) << 23;
    return 4.0f * std::bit_cast<float>(reciprocal_bits);
}

template <typename RT>
__device__ __forceinline__ void stage_codes(
    const RT &tile,
    uint8_t (&codes)[kittens::WARPGROUP_WARPS][16][33],
    float scale
) {
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int lane_pair = lane & 3;
    const int row_in_octet = lane >> 2;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int pair0 = j * 8 + lane_pair;
            const int row0 = i * 16 + row_in_octet;
            codes[warp][pair0][row0] =
                quantize_pair(tile.tiles[i][j].data[0], scale);
            codes[warp][pair0][row0 + 8] =
                quantize_pair(tile.tiles[i][j].data[1], scale);
            codes[warp][pair0 + 4][row0] =
                quantize_pair(tile.tiles[i][j].data[2], scale);
            codes[warp][pair0 + 4][row0 + 8] =
                quantize_pair(tile.tiles[i][j].data[3], scale);
        }
    }
}

template <bool STAGE_BF16_PAIRS, typename RT>
__device__ __forceinline__ void stage_codes_perblock_qk(
    const RT &tile,
    uint8_t (&codes)[kittens::WARPGROUP_WARPS][16][33],
    uint32_t (&pairs)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE],
    uint8_t (&block_scales)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][2]
) {
    static_assert(RT::width == 2);
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int lane_pair = lane & 3;
    const int row_in_octet = lane >> 2;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int pair0 = j * 8 + lane_pair;
            const int row0 = i * 16 + row_in_octet;
            if constexpr (STAGE_BF16_PAIRS) {
                pairs[warp][row0][pair0] =
                    *reinterpret_cast<const uint32_t *>(
                        &tile.tiles[i][j].data[0]
                    );
                pairs[warp][row0 + 8][pair0] =
                    *reinterpret_cast<const uint32_t *>(
                        &tile.tiles[i][j].data[1]
                    );
                pairs[warp][row0][pair0 + 4] =
                    *reinterpret_cast<const uint32_t *>(
                        &tile.tiles[i][j].data[2]
                    );
                pairs[warp][row0 + 8][pair0 + 4] =
                    *reinterpret_cast<const uint32_t *>(
                        &tile.tiles[i][j].data[3]
                    );
            }
            uint32_t row0_amax = max(
                static_cast<uint32_t>(bf16_pair_amax_bits(
                    tile.tiles[i][j].data[0]
                )),
                static_cast<uint32_t>(bf16_pair_amax_bits(
                    tile.tiles[i][j].data[2]
                ))
            );
            uint32_t row8_amax = max(
                static_cast<uint32_t>(bf16_pair_amax_bits(
                    tile.tiles[i][j].data[1]
                )),
                static_cast<uint32_t>(bf16_pair_amax_bits(
                    tile.tiles[i][j].data[3]
                ))
            );
            row0_amax = max(
                row0_amax,
                __shfl_xor_sync(0xffffffffu, row0_amax, 1)
            );
            row8_amax = max(
                row8_amax,
                __shfl_xor_sync(0xffffffffu, row8_amax, 1)
            );
            row0_amax = max(
                row0_amax,
                __shfl_xor_sync(0xffffffffu, row0_amax, 2)
            );
            row8_amax = max(
                row8_amax,
                __shfl_xor_sync(0xffffffffu, row8_amax, 2)
            );
            const kittens::fp8e4m3 row0_scale =
                kittens::base_types::convertor<
                    kittens::fp8e4m3,
                    float
                >::convert(
                    positive_bf16_bits_to_float(
                        static_cast<uint16_t>(row0_amax)
                    ) * (1.0f / 6.0f)
                );
            const kittens::fp8e4m3 row8_scale =
                kittens::base_types::convertor<
                    kittens::fp8e4m3,
                    float
                >::convert(
                    positive_bf16_bits_to_float(
                        static_cast<uint16_t>(row8_amax)
                    ) * (1.0f / 6.0f)
                );
            const float rounded_row0_scale =
                kittens::base_types::convertor<
                    float,
                    kittens::fp8e4m3
                >::convert(row0_scale);
            const float rounded_row8_scale =
                kittens::base_types::convertor<
                    float,
                    kittens::fp8e4m3
                >::convert(row8_scale);
            const float row0_multiplier = rounded_row0_scale > 0.0f
                ? 1.0f / rounded_row0_scale
                : 0.0f;
            const float row8_multiplier = rounded_row8_scale > 0.0f
                ? 1.0f / rounded_row8_scale
                : 0.0f;
            codes[warp][pair0][row0] = quantize_pair(
                tile.tiles[i][j].data[0], row0_multiplier
            );
            codes[warp][pair0][row0 + 8] = quantize_pair(
                tile.tiles[i][j].data[1], row8_multiplier
            );
            codes[warp][pair0 + 4][row0] = quantize_pair(
                tile.tiles[i][j].data[2], row0_multiplier
            );
            codes[warp][pair0 + 4][row0 + 8] = quantize_pair(
                tile.tiles[i][j].data[3], row8_multiplier
            );
            if (lane_pair == 0) {
                block_scales[warp][row0][j] =
                    std::bit_cast<uint8_t>(row0_scale);
                block_scales[warp][row0 + 8][j] =
                    std::bit_cast<uint8_t>(row8_scale);
            }
        }
    }
}

template <typename RT>
__device__ __forceinline__ void stage_codes_and_bf16_pairs(
    const RT &tile,
    uint8_t (&codes)[kittens::WARPGROUP_WARPS][16][33],
    uint32_t (&pairs)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE],
    float scale
) {
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int lane_pair = lane & 3;
    const int row_in_octet = lane >> 2;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int pair0 = j * 8 + lane_pair;
            const int row0 = i * 16 + row_in_octet;
            const uint32_t value0 = *reinterpret_cast<const uint32_t *>(
                &tile.tiles[i][j].data[0]
            );
            const uint32_t value1 = *reinterpret_cast<const uint32_t *>(
                &tile.tiles[i][j].data[1]
            );
            const uint32_t value2 = *reinterpret_cast<const uint32_t *>(
                &tile.tiles[i][j].data[2]
            );
            const uint32_t value3 = *reinterpret_cast<const uint32_t *>(
                &tile.tiles[i][j].data[3]
            );
            pairs[warp][row0][pair0] = value0;
            pairs[warp][row0 + 8][pair0] = value1;
            pairs[warp][row0][pair0 + 4] = value2;
            pairs[warp][row0 + 8][pair0 + 4] = value3;
            codes[warp][pair0][row0] =
                quantize_pair(tile.tiles[i][j].data[0], scale);
            codes[warp][pair0][row0 + 8] =
                quantize_pair(tile.tiles[i][j].data[1], scale);
            codes[warp][pair0 + 4][row0] =
                quantize_pair(tile.tiles[i][j].data[2], scale);
            codes[warp][pair0 + 4][row0 + 8] =
                quantize_pair(tile.tiles[i][j].data[3], scale);
        }
    }
}

template <
    typename C,
    bool PUBLISH_ALIGNED_QK = true,
    bool INTERLEAVE_CAUSAL_KV = false
>
__device__ __forceinline__ void publish_codes(
    const globals<C> &g,
    uint8_t (&codes)[kittens::WARPGROUP_WARPS][16][33],
    int block_batch_idx,
    int block_seq_base,
    int local_col_start,
    bool is_k
) {
    constexpr int kDepth = C::QK_DEPTH;
    constexpr int kPackedDepth = kDepth / 2;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int seq_base = block_seq_base + warp * 32;
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int physical_head_count = is_k
        ? g.K.cols() / kDepth
        : g.Q.cols() / kDepth;
    const int output_depth = g.paired_d64 ? 64 : kDepth;
    const int output_packed_depth = output_depth / 2;
    const int output_head_count = g.paired_d64
        ? physical_head_count * 2
        : physical_head_count;
    const int output_head_idx = g.paired_d64
        ? head_idx * 2 + depth_base / 64
        : head_idx;
    const int output_depth_base = g.paired_d64
        ? depth_base & 63
        : depth_base;

    // One lane owns one row.  Assemble the complete K32 compact segment in
    // registers and publish it with a single 128-bit store.
    uint32_t depth_words[4] = {0, 0, 0, 0};
    uint64_t aligned_low = 0;
    uint64_t aligned_high = 0;
    #pragma unroll
    for (int pair = 0; pair < 16; ++pair) {
        const uint8_t bits = codes[warp][pair][lane];
        depth_words[pair >> 2] |=
            static_cast<uint32_t>(bits) << ((pair & 3) * 8);
        if (pair < 8) {
            aligned_low |= static_cast<uint64_t>(bits) << (pair * 8);
        } else {
            aligned_high |=
                static_cast<uint64_t>(bits) << ((pair - 8) * 8);
        }
    }
    const int interleaved_local_seq = (lane & 3) * 32 +
        warp * 8 + (lane >> 2);
    const int seq_idx = INTERLEAVE_CAUSAL_KV && is_k
        ? block_seq_base + interleaved_local_seq
        : seq_base + lane;
    const size_t depth_row_base =
        ((static_cast<size_t>(block_batch_idx) * output_head_count +
          output_head_idx) *
             g.seq_len +
         seq_idx) * output_packed_depth + output_depth_base / 2;
    const uint4 compact = make_uint4(
        depth_words[0],
        depth_words[1],
        depth_words[2],
        depth_words[3]
    );
    if (is_k) {
        *reinterpret_cast<uint4 *>(g.k_depth_packed + depth_row_base) =
            compact;
        if constexpr (PUBLISH_ALIGNED_QK) {
        const size_t aligned_base =
            ((static_cast<size_t>(block_batch_idx) * g.seq_len + seq_idx) *
                 physical_head_count +
             head_idx) * kDepth +
            (depth_base / 16) * 16;
        *reinterpret_cast<uint64_t *>(
            g.k_depth_aligned + aligned_base
        ) = aligned_low;
        *reinterpret_cast<uint64_t *>(
            g.k_depth_aligned + aligned_base + 8
        ) = 0;
        *reinterpret_cast<uint64_t *>(
            g.k_depth_aligned + aligned_base + 16
        ) = aligned_high;
        *reinterpret_cast<uint64_t *>(
            g.k_depth_aligned + aligned_base + 24
        ) = 0;
        }
    } else {
        *reinterpret_cast<uint4 *>(g.q_depth_packed + depth_row_base) =
            compact;
    }

    if constexpr (PUBLISH_ALIGNED_QK) {
    if (!is_k) {
        // One lane owns one depth row and emits its two eight-byte sequence
        // segments plus their alignment holes as four 64-bit stores.
        const int local_depth = lane;
        const int pair = local_depth >> 1;
        const bool high_depth = (local_depth & 1) != 0;
        uint64_t sequence_low = 0;
        uint64_t sequence_high = 0;
        #pragma unroll
        for (int row_pair = 0; row_pair < 16; ++row_pair) {
            const uint8_t row0 = codes[warp][pair][2 * row_pair];
            const uint8_t row1 = codes[warp][pair][2 * row_pair + 1];
            const uint8_t bits = high_depth
                ? static_cast<uint8_t>(
                      ((row0 >> 4) & 0x0fu) | (row1 & 0xf0u)
                  )
                : static_cast<uint8_t>(
                      (row0 & 0x0fu) | ((row1 & 0x0fu) << 4)
                  );
            if (row_pair < 8) {
                sequence_low |=
                    static_cast<uint64_t>(bits) << (row_pair * 8);
            } else {
                sequence_high |= static_cast<uint64_t>(bits)
                    << ((row_pair - 8) * 8);
            }
        }
        const int global_depth = depth_base + local_depth;
        const size_t aligned_base =
            ((static_cast<size_t>(block_batch_idx) * physical_head_count +
              head_idx) *
                 kDepth +
             global_depth) * g.seq_len +
            (seq_base / 16) * 16;
        *reinterpret_cast<uint64_t *>(
            g.q_sequence_aligned + aligned_base
        ) = sequence_low;
        *reinterpret_cast<uint64_t *>(
            g.q_sequence_aligned + aligned_base + 8
        ) = 0;
        *reinterpret_cast<uint64_t *>(
            g.q_sequence_aligned + aligned_base + 16
        ) = sequence_high;
        *reinterpret_cast<uint64_t *>(
            g.q_sequence_aligned + aligned_base + 24
        ) = 0;
    }
    }
}

template <typename C, bool FIXED_SCALE_QK = false>
__device__ __forceinline__ void publish_forward_qk_scales(
    const globals<C> &g,
    int block_batch_idx,
    int block_seq_base,
    int local_col_start,
    bool is_k
) {
    constexpr int kDepth = C::QK_DEPTH;
    constexpr int kChunks = kDepth / 64;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    if (warp != 0) {
        return;
    }
    const int batch_idx = block_batch_idx;
    const int seq_base = block_seq_base;
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int head_count = is_k
        ? g.K.cols() / kDepth
        : g.Q.cols() / kDepth;
    const int metadata_head_stride = g.Q.cols() / kDepth;
    if ((depth_base & 63) != 0) {
        return;
    }
    const int chunk = head_idx * kChunks + depth_base / 64;
    const size_t scale_record =
        (static_cast<size_t>(batch_idx) * metadata_head_stride + head_idx) * 7;
    // The retained forward's score/softmax path is calibrated to the native
    // NVFP4 hierarchy: use the full E4M3 block-scale range and carry the
    // inverse factor in the tensor-wide scale.  Although moving this factor
    // algebraically between the two levels is exact, the fused score packing
    // is not numerically invariant to that move.  0x7e is +448 in E4M3.
    constexpr uint32_t kSaturatedScalePattern = 0x7e7e7e7eu;
    const uint32_t pattern = kSaturatedScalePattern;
    const float qk_quant_scale = FIXED_SCALE_QK
        ? 16.0f
        : g.adaptive_scales[scale_record + (is_k ? 1 : 0)];
    const float tensor_scale = 1.0f / (448.0f * qk_quant_scale);
    const uint4 vector = make_uint4(pattern, pattern, pattern, pattern);
    if (is_k) {
        const int scale_tiles = g.seq_len / 64;
        const int first_tile = seq_base / 64;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const size_t page =
                ((static_cast<size_t>(batch_idx) * scale_tiles +
                  first_tile + half) * (head_count * kChunks) + chunk) * 512;
            *reinterpret_cast<uint4 *>(
                g.k_forward_scales + page + lane * sizeof(uint4)
            ) = vector;
        }
        if (seq_base == 0 && lane == 0 &&
            (g.paired_d64 || depth_base == 0)) {
            const int output_head_count = g.paired_d64
                ? head_count * kChunks
                : head_count;
            const int output_head_idx = g.paired_d64 ? chunk : head_idx;
            g.k_forward_global_scale[
                static_cast<size_t>(batch_idx) * output_head_count +
                output_head_idx
            ] = tensor_scale;
        }
    } else {
        const int scale_tiles = g.seq_len / 128;
        const size_t page =
            ((static_cast<size_t>(batch_idx) * scale_tiles + seq_base / 128) *
                 (head_count * kChunks) +
             chunk) * 512;
        *reinterpret_cast<uint4 *>(
            g.q_forward_scales + page + lane * sizeof(uint4)
        ) = vector;
        if (seq_base == 0 && lane == 0 &&
            (g.paired_d64 || depth_base == 0)) {
            const int output_head_count = g.paired_d64
                ? head_count * kChunks
                : head_count;
            const int output_head_idx = g.paired_d64 ? chunk : head_idx;
            g.q_forward_global_scale[
                static_cast<size_t>(batch_idx) * output_head_count +
                output_head_idx
            ] = tensor_scale;
        }
    }
}

template <typename C, bool INTERLEAVE_CAUSAL_KV = false>
__device__ __forceinline__ void publish_forward_qk_perblock_scales(
    const globals<C> &g,
    uint8_t (&block_scales)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][2],
    int block_batch_idx,
    int block_seq_base,
    int local_col_start,
    bool is_k
) {
    constexpr int kDepth = C::QK_DEPTH;
    constexpr int kChunks = kDepth / 64;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - head_idx * kDepth;
    const int head_count = is_k
        ? g.K.cols() / kDepth
        : g.Q.cols() / kDepth;
    const int chunk = head_idx * kChunks + depth_base / 64;
    const int depth_block = (depth_base & 63) / 16;

    int page_quarter = warp;
    int page_row = lane;
    if constexpr (INTERLEAVE_CAUSAL_KV) {
        if (is_k) {
            const int interleaved_local_seq =
                (lane & 3) * 32 + warp * 8 + (lane >> 2);
            page_quarter = interleaved_local_seq >> 5;
            page_row = interleaved_local_seq & 31;
        }
    }
    const int page_offset =
        page_row * 16 + page_quarter * 4 + depth_block;
    const uint16_t scale_pair =
        static_cast<uint16_t>(block_scales[warp][lane][0]) |
        (static_cast<uint16_t>(block_scales[warp][lane][1]) << 8);

    if (is_k) {
        const int scale_tiles = g.seq_len / 64;
        const int first_tile = block_seq_base / 64;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const size_t page =
                ((static_cast<size_t>(block_batch_idx) * scale_tiles +
                  first_tile + half) * (head_count * kChunks) + chunk) * 512;
            *reinterpret_cast<uint16_t *>(
                g.k_forward_scales + page + page_offset
            ) = scale_pair;
        }
        if (block_seq_base == 0 && warp == 0 && lane == 0 &&
            (g.paired_d64 || depth_base == 0)) {
            const int output_head_count = g.paired_d64
                ? head_count * kChunks
                : head_count;
            const int output_head_idx = g.paired_d64 ? chunk : head_idx;
            g.k_forward_global_scale[
                static_cast<size_t>(block_batch_idx) *
                    output_head_count +
                output_head_idx
            ] = 1.0f;
        }
    } else {
        const int scale_tiles = g.seq_len / 128;
        const size_t page =
            ((static_cast<size_t>(block_batch_idx) * scale_tiles +
              block_seq_base / 128) * (head_count * kChunks) + chunk) * 512;
        *reinterpret_cast<uint16_t *>(
            g.q_forward_scales + page + page_offset
        ) = scale_pair;
        if (block_seq_base == 0 && warp == 0 && lane == 0 &&
            (g.paired_d64 || depth_base == 0)) {
            const int output_head_count = g.paired_d64
                ? head_count * kChunks
                : head_count;
            const int output_head_idx = g.paired_d64 ? chunk : head_idx;
            g.q_forward_global_scale[
                static_cast<size_t>(block_batch_idx) *
                    output_head_count +
                output_head_idx
            ] = 1.0f;
        }
    }
}

template <
    typename C,
    bool SEQUENCE_MAJOR_COLUMN_SCALES = false,
    bool INTERLEAVE_CAUSAL_KV = false
>
__device__ __forceinline__ void publish_v_mxfp4_from_backward_e4m3(
    const globals<C> &g,
    uint32_t (&pairs)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE],
    int block_batch_idx,
    int block_seq_base,
    int local_col_start
) {
    constexpr int kDepth = 128;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int batch_idx = block_batch_idx;
    const int seq_base = block_seq_base + warp * 32;
    const int physical_head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - physical_head_idx * kDepth;
    const int physical_v_heads = g.v_width / kDepth;
    const int physical_depth = depth_base + lane;
    const int output_head_depth = g.paired_d64 ? 64 : kDepth;
    const int v_heads = g.paired_d64
        ? physical_v_heads * 2
        : physical_v_heads;
    const int head_idx = g.paired_d64
        ? physical_head_idx * 2 + physical_depth / 64
        : physical_head_idx;
    const int depth = g.paired_d64
        ? physical_depth & 63
        : physical_depth;
    const int byte_word = lane >> 2;
    const int byte_shift = (lane & 3) * 8;

    // publish_v_fp8 has replaced words [0, 8) of every row with the exact
    // E4M3(x4) bytes written for backward.  Gather one feature column in the
    // forward consumer's causal order while retaining adjacent sequence
    // values as native E4M3x2 conversion operands.  Consuming the staged
    // bytes preserves universal bitwise identity with backward's depth-paired
    // encoder, including pathological BF16 bit patterns.
    uint16_t gathered_pairs[16];
    uint8_t amax_codes[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int packed_pair = 0; packed_pair < 16; ++packed_pair) {
        const int row0 = 2 * packed_pair;
        const int row1 = row0 + 1;
        const int source_warp0 = INTERLEAVE_CAUSAL_KV ? row0 >> 3 : warp;
        const int source_warp1 = INTERLEAVE_CAUSAL_KV ? row1 >> 3 : warp;
        const int source_row0 = INTERLEAVE_CAUSAL_KV
            ? warp + (row0 & 7) * 4
            : row0;
        const int source_row1 = INTERLEAVE_CAUSAL_KV
            ? warp + (row1 & 7) * 4
            : row1;
        const uint8_t value0 = static_cast<uint8_t>(
            pairs[source_warp0][source_row0][byte_word] >> byte_shift
        );
        const uint8_t value1 = static_cast<uint8_t>(
            pairs[source_warp1][source_row1][byte_word] >> byte_shift
        );
        gathered_pairs[packed_pair] = static_cast<uint16_t>(value0) |
            (static_cast<uint16_t>(value1) << 8);
        amax_codes[packed_pair & 3] = max(
            amax_codes[packed_pair & 3],
            max(
                static_cast<uint8_t>(value0 & 0x7fu),
                static_cast<uint8_t>(value1 & 0x7fu)
            )
        );
    }
    const uint8_t row_amax_code = max(
        max(amax_codes[0], amax_codes[1]),
        max(amax_codes[2], amax_codes[3])
    );
    uint32_t tile_amax_code = row_amax_code;
    if (g.v_mxfp4_scale_2d) {
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            tile_amax_code = max(
                tile_amax_code,
                __shfl_xor_sync(0xffffffffu, tile_amax_code, mask)
            );
        }
    }
    const uint8_t selected_amax_code = g.v_mxfp4_scale_2d
        ? static_cast<uint8_t>(tile_amax_code)
        : row_amax_code;
    const bool nan_group = selected_amax_code == 0x7fu;
    // Match the direct publisher's geometry-specific scale policy after the
    // E4M3 representation boundary: 2-D tiles use RTE, while 1x32 rows use
    // the empirically selected MSE cutoff.  E4M3FN magnitude 0x7f is NaN:
    // publish the E8M0 sentinel and zero the complete affected group, exactly
    // matching the standalone E4M3-to-MX converter's fail-closed policy.
    const uint16_t selected_amax_bits = nan_group
        ? 0
        : e4m3_x4_amax_to_logical_bf16_bits(selected_amax_code);
    const uint8_t e8m0 = nan_group
        ? 0xffu
        : (g.v_mxfp4_scale_2d
            ? bf16_amax_to_e8m0_rte(selected_amax_bits)
            : bf16_amax_to_e8m0_1d_mse(selected_amax_bits));
    const uint32_t multiplier_half2 =
        e8m0_e4m3_x4_encode_multiplier_half2(e8m0);
    const uint16_t finite_pair_mask = nan_group ? 0u : 0xffffu;
    uint32_t packed_words[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        packed_words[word] = quantize_four_e4m3_pairs_to_mxfp4(
            gathered_pairs[4 * word] & finite_pair_mask,
            gathered_pairs[4 * word + 1] & finite_pair_mask,
            gathered_pairs[4 * word + 2] & finite_pair_mask,
            gathered_pairs[4 * word + 3] & finite_pair_mask,
            multiplier_half2
        );
    }

    const size_t payload_base =
        ((static_cast<size_t>(batch_idx) * v_heads + head_idx) *
             output_head_depth +
         depth) * (g.seq_len / 2) + seq_base / 2;
    *reinterpret_cast<uint4 *>(g.v_mxfp4 + payload_base) = make_uint4(
        packed_words[0],
        packed_words[1],
        packed_words[2],
        packed_words[3]
    );

    const int sequence_tile = seq_base / 128;
    const int sequence_quarter = (seq_base & 127) / 32;
    const int depth_lane = depth & 31;
    const int depth_group = depth >> 5;
    const int sequence_tiles = g.seq_len / 128;
    const size_t scale_page = SEQUENCE_MAJOR_COLUMN_SCALES
        ? ((static_cast<size_t>(batch_idx) * sequence_tiles +
            sequence_tile) * v_heads + head_idx) * 512
        : ((static_cast<size_t>(batch_idx) * v_heads + head_idx) *
               g.v_scale_rows * sequence_tiles +
           sequence_tile) * 512;
    g.v_mxfp4_scales[
        scale_page + depth_lane * 16 + depth_group * 4 + sequence_quarter
    ] = e8m0;
    if constexpr (!SEQUENCE_MAJOR_COLUMN_SCALES) {
    if (g.v_scale_rows > 1) {
        const size_t upper_page = scale_page + sequence_tiles * 512;
        if (depth_group >= 2) {
            g.v_mxfp4_scales[
                upper_page + depth_lane * 16 + (depth_group - 2) * 4 +
                sequence_quarter
            ] = e8m0;
        } else {
            g.v_mxfp4_scales[
                upper_page + depth_lane * 16 + (depth_group + 2) * 4 +
                sequence_quarter
            ] = 0;
        }
    }
    }
}

template <
    typename C,
    bool SEQUENCE_MAJOR_COLUMN_SCALES = false,
    bool INTERLEAVE_CAUSAL_KV = false,
    bool PUBLISH_BACKWARD_MXFP4 = false,
    bool SHARE_MXFP4_TILE_WITH_BACKWARD = false
>
__device__ __noinline__ void publish_v_mxfp4_from_output_shared(
    const globals<C> &g,
    const typename globals<C>::D_tile &tile,
    int block_batch_idx,
    int block_seq_base,
    int local_col_start
) {
    static_assert(
        !SHARE_MXFP4_TILE_WITH_BACKWARD || PUBLISH_BACKWARD_MXFP4,
        "shared forward/backward MXFP4 tile requires backward publication"
    );
    // The projection output ring already holds this complete 128-row x
    // 32-depth BF16 slice.  Consume it in place instead of loading it into a
    // register tile and immediately restaging the same bits into a second
    // shared-memory fragment.  The addressing below is deliberately identical
    // to publish_v_mxfp4's BF16 gather after composing its
    // [source_warp][source_row] coordinates into one output-ring row.
    constexpr int kDepth = 128;
    static_assert(globals<C>::D_tile::cols == 32);
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int batch_idx = block_batch_idx;
    const int seq_base = block_seq_base + warp * 32;
    const int physical_head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - physical_head_idx * kDepth;
    const int physical_v_heads = g.v_width / kDepth;
    const int physical_depth = depth_base + lane;
    const int output_head_depth = g.paired_d64 ? 64 : kDepth;
    const int v_heads = g.paired_d64
        ? physical_v_heads * 2
        : physical_v_heads;
    const int head_idx = g.paired_d64
        ? physical_head_idx * 2 + physical_depth / 64
        : physical_head_idx;
    const int depth = g.paired_d64
        ? physical_depth & 63
        : physical_depth;
    const int depth_pair = lane & ~1;
    const int halfword_shift = (lane & 1) * 16;

    {
    uint32_t gathered_pairs[16];
    uint16_t amax_bits[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int packed_pair = 0; packed_pair < 16; ++packed_pair) {
        const int row0 = 2 * packed_pair;
        const int row1 = row0 + 1;
        const int source_local_row0 = INTERLEAVE_CAUSAL_KV
            ? (row0 >> 3) * 32 + warp + (row0 & 7) * 4
            : warp * 32 + row0;
        const int source_local_row1 = INTERLEAVE_CAUSAL_KV
            ? (row1 >> 3) * 32 + warp + (row1 & 7) * 4
            : warp * 32 + row1;
        const int2 source_coord0 = {source_local_row0, depth_pair};
        const int2 source_coord1 = {source_local_row1, depth_pair};
        const uint32_t source_pair0 =
            *reinterpret_cast<const uint32_t *>(&tile[source_coord0]);
        const uint32_t source_pair1 =
            *reinterpret_cast<const uint32_t *>(&tile[source_coord1]);
        const uint16_t value0 = static_cast<uint16_t>(
            source_pair0 >> halfword_shift
        );
        const uint16_t value1 = static_cast<uint16_t>(
            source_pair1 >> halfword_shift
        );
        const uint32_t gathered = static_cast<uint32_t>(value0) |
            (static_cast<uint32_t>(value1) << 16);
        gathered_pairs[packed_pair] = gathered;
        amax_bits[packed_pair & 3] = max(
            amax_bits[packed_pair & 3],
            max(
                static_cast<uint16_t>(value0 & 0x7fffu),
                static_cast<uint16_t>(value1 & 0x7fffu)
            )
        );
    }
    const uint16_t row_amax_bits = max(
        max(amax_bits[0], amax_bits[1]),
        max(amax_bits[2], amax_bits[3])
    );
    uint32_t tile_amax_bits = row_amax_bits;
    if (g.v_mxfp4_scale_2d) {
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            tile_amax_bits = max(
                tile_amax_bits,
                __shfl_xor_sync(0xffffffffu, tile_amax_bits, mask)
            );
        }
    }
    const uint8_t e8m0 = g.v_mxfp4_scale_2d
        ? bf16_amax_to_e8m0_rte(static_cast<uint16_t>(tile_amax_bits))
        : bf16_amax_to_e8m0_1d_mse(row_amax_bits);
    const float multiplier = e8m0_encode_multiplier(e8m0);
    uint32_t packed_words[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        packed_words[word] = quantize_four_bf16_pairs(
            gathered_pairs[4 * word],
            gathered_pairs[4 * word + 1],
            gathered_pairs[4 * word + 2],
            gathered_pairs[4 * word + 3],
            multiplier
        );
    }

    const size_t payload_base =
        ((static_cast<size_t>(batch_idx) * v_heads + head_idx) *
             output_head_depth +
         depth) * (g.seq_len / 2) + seq_base / 2;
    *reinterpret_cast<uint4 *>(g.v_mxfp4 + payload_base) = make_uint4(
        packed_words[0],
        packed_words[1],
        packed_words[2],
        packed_words[3]
    );

    const int sequence_tile = seq_base / 128;
    const int sequence_quarter = (seq_base & 127) / 32;
    const int depth_lane = depth & 31;
    const int depth_group = depth >> 5;
    const int sequence_tiles = g.seq_len / 128;
    const size_t scale_page = SEQUENCE_MAJOR_COLUMN_SCALES
        ? ((static_cast<size_t>(batch_idx) * sequence_tiles +
            sequence_tile) * v_heads + head_idx) * 512
        : ((static_cast<size_t>(batch_idx) * v_heads + head_idx) *
               g.v_scale_rows * sequence_tiles +
           sequence_tile) * 512;
    g.v_mxfp4_scales[
        scale_page + depth_lane * 16 + depth_group * 4 + sequence_quarter
    ] = e8m0;
    if constexpr (!SEQUENCE_MAJOR_COLUMN_SCALES) {
        if (g.v_scale_rows > 1) {
            const size_t upper_page = scale_page + sequence_tiles * 512;
            if (depth_group >= 2) {
                g.v_mxfp4_scales[
                    upper_page + depth_lane * 16 + (depth_group - 2) * 4 +
                    sequence_quarter
                ] = e8m0;
            } else {
                g.v_mxfp4_scales[
                    upper_page + depth_lane * 16 + (depth_group + 2) * 4 +
                    sequence_quarter
                ] = 0;
            }
        }
    }

    if constexpr (SHARE_MXFP4_TILE_WITH_BACKWARD) {
        // The checked host symbol requires one common E8M0 code over this
        // D32xS32 tile.  Forward has already stored the depth-major words;
        // transpose those exact nibbles in registers and publish the second
        // physical orientation.  There is no second BF16 scan, amax, or E2M1
        // conversion.
        transpose_mxfp4_32x32_nibbles(packed_words);
        const int seq_idx = block_seq_base + warp * 32 + lane;
        const size_t backward_payload_base =
            ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
                 v_heads +
             physical_head_idx) * (output_head_depth / 2) + depth_base / 2;
        *reinterpret_cast<uint4 *>(
            g.v_backward_mxfp4 + backward_payload_base
        ) = make_uint4(
            packed_words[0],
            packed_words[1],
            packed_words[2],
            packed_words[3]
        );

        const int row_in_32 = seq_idx & 31;
        const int tile_in_block = (seq_idx >> 5) & 3;
        const int k_block = depth_base >> 5;
        const size_t backward_scale_page =
            ((static_cast<size_t>(batch_idx) * (g.seq_len / 128) +
              seq_idx / 128) * v_heads + physical_head_idx) * 512;
        g.v_backward_mxfp4_scales[
            backward_scale_page + row_in_32 * 16 +
            tile_in_block * 4 + k_block
        ] = e8m0;
    }
    }

    if constexpr (
        PUBLISH_BACKWARD_MXFP4 && !SHARE_MXFP4_TILE_WITH_BACKWARD
    ) {
        // dP consumes V with feature depth as K, so its block-scaled payload
        // is row-major rather than the forward publisher's feature-major
        // layout. The complete BF16 128x32 output slice is still resident:
        // let each lane own one sequence row and quantize its 32 local depth
        // values directly, without a register-tile reload or shared restage.
        // This route is restricted by the kernel contract to ordinary-order
        // native D128. Local output columns are therefore exactly [0, 32),
        // while depth_base contributes only to global payload/scale offsets.
        uint32_t row_pairs[16];
        uint16_t row_amax_bits = 0;
        #pragma unroll
        for (int depth_pair_index = 0; depth_pair_index < 16;
             ++depth_pair_index) {
            const int2 source_coord = {
                warp * 32 + lane,
                2 * depth_pair_index
            };
            const uint32_t value_pair =
                *reinterpret_cast<const uint32_t *>(&tile[source_coord]);
            row_pairs[depth_pair_index] = value_pair;
            const uint16_t value0 = static_cast<uint16_t>(value_pair);
            const uint16_t value1 = static_cast<uint16_t>(value_pair >> 16);
            row_amax_bits = max(
                row_amax_bits,
                max(
                    static_cast<uint16_t>(value0 & 0x7fffu),
                    static_cast<uint16_t>(value1 & 0x7fffu)
                )
            );
        }
        const uint8_t row_e8m0 =
            bf16_amax_to_e8m0_1d_mse(row_amax_bits);
        const float row_multiplier = e8m0_encode_multiplier(row_e8m0);
        uint32_t row_words[4];
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int pair = word * 4;
            row_words[word] = quantize_four_bf16_pairs(
                row_pairs[pair],
                row_pairs[pair + 1],
                row_pairs[pair + 2],
                row_pairs[pair + 3],
                row_multiplier
            );
        }

        const int seq_idx = block_seq_base + warp * 32 + lane;
        const size_t backward_payload_base =
            ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
                 v_heads +
             physical_head_idx) * (output_head_depth / 2) + depth_base / 2;
        *reinterpret_cast<uint4 *>(
            g.v_backward_mxfp4 + backward_payload_base
        ) = make_uint4(
            row_words[0], row_words[1], row_words[2], row_words[3]
        );

        const int row_in_32 = seq_idx & 31;
        const int tile_in_block = (seq_idx >> 5) & 3;
        const int k_block = depth_base >> 5;
        const size_t backward_scale_page =
            ((static_cast<size_t>(batch_idx) * (g.seq_len / 128) +
              seq_idx / 128) * v_heads + physical_head_idx) * 512;
        g.v_backward_mxfp4_scales[
            backward_scale_page + row_in_32 * 16 +
            tile_in_block * 4 + k_block
        ] = row_e8m0;
    }
}

// Experimental native-D128 backward publication with one common E8M0 anchor
// per complete V row.  All eight N32 epilogue slices have already been copied
// to the resident output ring before the publication loop begins, so the four
// slices of one D128 head can be inspected together without another shared
// stage or a global-memory round trip.
//
// The payload retains the existing row-major [B,S,Hkv,D/2] geometry.  The
// common code is repeated in the four existing D32 scale slots so the physical
// [B,S/128,Hkv,512] ABI remains unchanged.  This direct-common experimental
// variant quantizes BF16 once under that D128 anchor.  It deliberately differs
// from v503's two-stage D32-quantize/re-anchor bytes and must be judged against
// the common-anchor numerical oracle rather than bytewise v503 equivalence.
template <typename C>
__device__ __noinline__ void
publish_v_common_rowscale_mxfp4_from_output_ring(
    const globals<C> &g,
    const typename globals<C>::outputs_t &outputs,
    int first_epi,
    int block_batch_idx,
    int block_seq_base,
    int local_head_col_start
) {
    constexpr int kDepth = 128;
    constexpr int kDepthBlocks = kDepth / 32;
    static_assert(globals<C>::D_tile::rows == 128);
    static_assert(globals<C>::D_tile::cols == 32);
    static_assert(kDepthBlocks == 4);

    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int batch_idx = block_batch_idx;
    const int seq_idx = block_seq_base + warp * 32 + lane;
    const int v_heads = g.v_width / kDepth;
    const int head_idx = local_head_col_start / kDepth;

    uint16_t row_amax_bits[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int k_block = 0; k_block < kDepthBlocks; ++k_block) {
        const auto &tile = outputs.D[first_epi + k_block];
        #pragma unroll
        for (int pair = 0; pair < 16; ++pair) {
            const int2 source_coord = {warp * 32 + lane, 2 * pair};
            const uint32_t value_pair =
                *reinterpret_cast<const uint32_t *>(&tile[source_coord]);
            const uint16_t value0 = static_cast<uint16_t>(value_pair);
            const uint16_t value1 = static_cast<uint16_t>(value_pair >> 16);
            row_amax_bits[pair & 3] = max(
                row_amax_bits[pair & 3],
                max(
                    static_cast<uint16_t>(value0 & 0x7fffu),
                    static_cast<uint16_t>(value1 & 0x7fffu)
                )
            );
        }
    }
    const uint16_t common_amax_bits = max(
        max(row_amax_bits[0], row_amax_bits[1]),
        max(row_amax_bits[2], row_amax_bits[3])
    );
    const uint8_t common_code =
        bf16_amax_to_e8m0_1d_mse(common_amax_bits);
    const float common_multiplier = e8m0_encode_multiplier(common_code);

    const size_t row_payload_base =
        ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) * v_heads +
         head_idx) * (kDepth / 2);
    #pragma unroll
    for (int k_block = 0; k_block < kDepthBlocks; ++k_block) {
        const auto &tile = outputs.D[first_epi + k_block];
        uint32_t row_pairs[16];
        #pragma unroll
        for (int pair = 0; pair < 16; ++pair) {
            const int2 source_coord = {warp * 32 + lane, 2 * pair};
            row_pairs[pair] =
                *reinterpret_cast<const uint32_t *>(&tile[source_coord]);
        }
        uint32_t row_words[4];
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int pair = word * 4;
            row_words[word] = quantize_four_bf16_pairs(
                row_pairs[pair],
                row_pairs[pair + 1],
                row_pairs[pair + 2],
                row_pairs[pair + 3],
                common_multiplier
            );
        }
        *reinterpret_cast<uint4 *>(
            g.v_backward_mxfp4 + row_payload_base + k_block * 16
        ) = make_uint4(
            row_words[0],
            row_words[1],
            row_words[2],
            row_words[3]
        );
    }

    const int row_in_32 = seq_idx & 31;
    const int tile_in_block = (seq_idx >> 5) & 3;
    const size_t scale_page =
        ((static_cast<size_t>(batch_idx) * (g.seq_len / 128) +
          seq_idx / 128) * v_heads + head_idx) * 512;
    const uint32_t repeated_common_code =
        static_cast<uint32_t>(common_code) * 0x01010101u;
    *reinterpret_cast<uint32_t *>(
        g.v_backward_mxfp4_scales + scale_page + row_in_32 * 16 +
        tile_in_block * 4
    ) = repeated_common_code;
}

template <
    typename C,
    bool SEQUENCE_MAJOR_COLUMN_SCALES = false,
    bool PUBLISH_BACKWARD_MXFP4 = true,
    bool INTERLEAVE_CAUSAL_KV = false,
    bool PUBLISH_REPRESENTED_V_FP8 = false
>
__device__ __forceinline__ void publish_v_mxfp4(
    const globals<C> &g,
    uint32_t (&pairs)[
        kittens::WARPGROUP_WARPS
    ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE],
    int block_batch_idx,
    int block_seq_base,
    int local_col_start
) {
    constexpr int kDepth = 128;
    const int warp = kittens::warpgroup::warpid();
    const int lane = kittens::warp::laneid();
    const int batch_idx = block_batch_idx;
    const int seq_base = block_seq_base + warp * 32;
    const int physical_head_idx = local_col_start / kDepth;
    const int depth_base = local_col_start - physical_head_idx * kDepth;
    const int physical_v_heads = g.v_width / kDepth;
    const int physical_depth = depth_base + lane;
    const int output_head_depth = g.paired_d64 ? 64 : kDepth;
    const int v_heads = g.paired_d64
        ? physical_v_heads * 2
        : physical_v_heads;
    const int head_idx = g.paired_d64
        ? physical_head_idx * 2 + physical_depth / 64
        : physical_head_idx;
    const int depth = g.paired_d64
        ? physical_depth & 63
        : physical_depth;
    const int pair_index = lane >> 1;
    const int halfword_shift = (lane & 1) * 16;

    // Retain the 32 sequence values as the 16 BF16 pairs consumed by E2M1,
    // rather than 32 separately promoted uint16_t values.  Four independent
    // amax chains remove the former 32-value serial dependency while keeping
    // the exact integer-max selector semantics (including Inf/NaN bit
    // patterns) and the causal-interleaved gather order.
    uint32_t gathered_pairs[16];
    uint16_t amax_bits[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int packed_pair = 0; packed_pair < 16; ++packed_pair) {
        const int row0 = 2 * packed_pair;
        const int row1 = row0 + 1;
        const int source_warp0 = INTERLEAVE_CAUSAL_KV ? row0 >> 3 : warp;
        const int source_warp1 = INTERLEAVE_CAUSAL_KV ? row1 >> 3 : warp;
        const int source_row0 = INTERLEAVE_CAUSAL_KV
            ? warp + (row0 & 7) * 4
            : row0;
        const int source_row1 = INTERLEAVE_CAUSAL_KV
            ? warp + (row1 & 7) * 4
            : row1;
        const uint16_t value0 = static_cast<uint16_t>(
            pairs[source_warp0][source_row0][pair_index] >> halfword_shift
        );
        const uint16_t value1 = static_cast<uint16_t>(
            pairs[source_warp1][source_row1][pair_index] >> halfword_shift
        );
        const uint32_t gathered = static_cast<uint32_t>(value0) |
            (static_cast<uint32_t>(value1) << 16);
        gathered_pairs[packed_pair] = gathered;
        amax_bits[packed_pair & 3] = max(
            amax_bits[packed_pair & 3],
            max(
                static_cast<uint16_t>(value0 & 0x7fffu),
                static_cast<uint16_t>(value1 & 0x7fffu)
            )
        );
    }
    const uint16_t row_amax_bits = max(
        max(amax_bits[0], amax_bits[1]),
        max(amax_bits[2], amax_bits[3])
    );
    uint32_t tile_amax_bits = row_amax_bits;
    if (g.v_mxfp4_scale_2d) {
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            tile_amax_bits = max(
                tile_amax_bits,
                __shfl_xor_sync(0xffffffffu, tile_amax_bits, mask)
            );
        }
    }
    // Selector policy is geometry-specific.  The 1.20 cutoff minimizes error
    // for 1x32 groups, while nearest-power RTE minimizes attention error for
    // the substantially larger 32x32 groups.
    const uint8_t e8m0 = g.v_mxfp4_scale_2d
        ? bf16_amax_to_e8m0_rte(static_cast<uint16_t>(tile_amax_bits))
        : bf16_amax_to_e8m0_1d_mse(row_amax_bits);
    const float multiplier = e8m0_encode_multiplier(e8m0);
    uint32_t packed_words[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        packed_words[word] = quantize_four_bf16_pairs(
            gathered_pairs[4 * word],
            gathered_pairs[4 * word + 1],
            gathered_pairs[4 * word + 2],
            gathered_pairs[4 * word + 3],
            multiplier
        );
    }

    const size_t payload_base =
        ((static_cast<size_t>(batch_idx) * v_heads + head_idx) *
             output_head_depth +
         depth) * (g.seq_len / 2) + seq_base / 2;
    *reinterpret_cast<uint4 *>(g.v_mxfp4 + payload_base) = make_uint4(
        packed_words[0],
        packed_words[1],
        packed_words[2],
        packed_words[3]
    );

    const int sequence_tile = seq_base / 128;
    const int sequence_quarter = (seq_base & 127) / 32;
    const int depth_lane = depth & 31;
    const int depth_group = depth >> 5;
    const int sequence_tiles = g.seq_len / 128;
    const size_t scale_page = SEQUENCE_MAJOR_COLUMN_SCALES
        ? ((static_cast<size_t>(batch_idx) * sequence_tiles +
            sequence_tile) * v_heads + head_idx) * 512
        : ((static_cast<size_t>(batch_idx) * v_heads + head_idx) *
               g.v_scale_rows * sequence_tiles +
           sequence_tile) * 512;
    g.v_mxfp4_scales[
        scale_page + depth_lane * 16 + depth_group * 4 + sequence_quarter
    ] = e8m0;
    if constexpr (!SEQUENCE_MAJOR_COLUMN_SCALES) {
    if (g.v_scale_rows > 1) {
        const size_t upper_page = scale_page + sequence_tiles * 512;
        if (depth_group >= 2) {
            g.v_mxfp4_scales[
                upper_page + depth_lane * 16 + (depth_group - 2) * 4 +
                sequence_quarter
            ] = e8m0;
        } else {
            g.v_mxfp4_scales[
                upper_page + depth_lane * 16 + (depth_group + 2) * 4 +
                sequence_quarter
            ] = 0;
        }
    }
    }

    if constexpr (PUBLISH_REPRESENTED_V_FP8) {
        // The forward MXFP4 payload is definitive. Decode its retained E2M1
        // codes with the exact E8M0 scale selected above, lift by four for
        // backward's fixed-0.25 E4M3 descriptor, and transpose back into
        // normal BSHD order through the scratch fragment. Do not revisit the
        // pre-quantized BF16 V values.
        kittens::warpgroup::sync(1);
        const float represented_scale = e8m0 == 0
            ? 0.0f
            : (4.0f / 6.0f) * std::bit_cast<float>(
                  static_cast<uint32_t>(e8m0) << 23
              );
        #pragma unroll
        for (int packed_pair = 0; packed_pair < 16; ++packed_pair) {
            const uint8_t code = static_cast<uint8_t>(
                packed_words[packed_pair >> 2] >>
                ((packed_pair & 3) * 8)
            );
            const uint16_t lifted = lift_fp4_pair_to_fp8(
                code,
                represented_scale
            );
            #pragma unroll
            for (int element = 0; element < 2; ++element) {
                const int physical_row = 2 * packed_pair + element;
                const int source_warp = INTERLEAVE_CAUSAL_KV
                    ? physical_row >> 3
                    : warp;
                const int source_row = INTERLEAVE_CAUSAL_KV
                    ? warp + (physical_row & 7) * 4
                    : physical_row;
                reinterpret_cast<uint8_t *>(
                    &pairs[source_warp][source_row][0]
                )[lane] = static_cast<uint8_t>(lifted >> (element * 8));
            }
        }
        kittens::warpgroup::sync(1);
        const int seq_idx = block_seq_base + warp * 32 + lane;
        const int output_depth_base = g.paired_d64
            ? depth_base & 63
            : depth_base;
        const size_t backward_output_base =
            ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
                 v_heads +
             head_idx) * output_head_depth + output_depth_base;
        // BF16_PAIR_SHARED_STRIDE is intentionally padded, so the shared row
        // is not generally 16-byte aligned. Assemble vectors from 32-bit
        // words and retain aligned global stores.
        const uint4 represented0 = make_uint4(
            pairs[warp][lane][0],
            pairs[warp][lane][1],
            pairs[warp][lane][2],
            pairs[warp][lane][3]
        );
        const uint4 represented1 = make_uint4(
            pairs[warp][lane][4],
            pairs[warp][lane][5],
            pairs[warp][lane][6],
            pairs[warp][lane][7]
        );
        *reinterpret_cast<uint4 *>(
            g.v_backward_fp8 + backward_output_base
        ) = represented0;
        *reinterpret_cast<uint4 *>(
            g.v_backward_fp8 + backward_output_base + 16
        ) = represented1;
        kittens::warpgroup::sync(1);
    }

    if constexpr (PUBLISH_BACKWARD_MXFP4) {
    // The backward dP MMA consumes V with depth as K.  Reuse the same staged
    // BF16 fragment to publish the complementary row-major K32 block: one
    // lane owns one sequence row, computes its exact E8M0 scale over the 32
    // depths in this epilogue slice, and emits one aligned 16-byte payload.
    uint16_t row_amax_bits = 0;
    #pragma unroll
    for (int depth_pair = 0; depth_pair < 16; ++depth_pair) {
        const uint32_t value_pair = pairs[warp][lane][depth_pair];
        const uint16_t value0 = static_cast<uint16_t>(value_pair);
        const uint16_t value1 = static_cast<uint16_t>(value_pair >> 16);
        const uint16_t absolute0 = value0 & 0x7fffu;
        const uint16_t absolute1 = value1 & 0x7fffu;
        row_amax_bits = max(row_amax_bits, max(absolute0, absolute1));
    }
    const uint8_t row_e8m0 = g.v_mxfp4_scale_2d
        ? e8m0
        : bf16_amax_to_e8m0_1d_mse(row_amax_bits);
    // Keep the backward-oriented payload on the standard MXFP4 width-six
    // contract used by tcgen05 block-scaled MMA.  The former rowwise width-four
    // special case disagreed with the independently published width-six dO
    // operand, so dP used mixed reconstruction factors and its centering path
    // could not apply one correct reciprocal.
    const float row_multiplier = e8m0_encode_multiplier(row_e8m0);
    uint32_t row_words[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        const int pair = word * 4;
        row_words[word] = quantize_four_bf16_pairs(
            pairs[warp][lane][pair],
            pairs[warp][lane][pair + 1],
            pairs[warp][lane][pair + 2],
            pairs[warp][lane][pair + 3],
            row_multiplier
        );
    }
    const int seq_idx = seq_base + lane;
    const size_t backward_payload_base =
        ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) * v_heads +
         head_idx) * (output_head_depth / 2) +
        (g.paired_d64 ? (depth_base & 63) : depth_base) / 2;
    *reinterpret_cast<uint4 *>(
        g.v_backward_mxfp4 + backward_payload_base
    ) = make_uint4(row_words[0], row_words[1], row_words[2], row_words[3]);

    const int row_in_32 = seq_idx & 31;
    const int tile_in_block = (seq_idx >> 5) & 3;
    const int k_block = depth_base >> 5;
    const size_t backward_scale_page =
        ((static_cast<size_t>(batch_idx) * (g.seq_len / 128) +
          seq_idx / 128) * v_heads + head_idx) * 512;
    g.v_backward_mxfp4_scales[
        backward_scale_page + row_in_32 * 16 + tile_in_block * 4 + k_block
    ] = row_e8m0;
    }
}

template <
    typename C,
    bool PUBLISH_FP4,
    bool PUBLISH_FORWARD_QK = false,
    bool PUBLISH_V_MXFP4 = false,
    bool STORE_BF16 = true,
    bool OUTPUT_IS_DOUT = false,
    bool PUBLISH_PURE_QK = false,
    bool PURE_QK_SINGLE_QUANT = false,
    bool SINGLE_OUTPUT = false,
    bool APPLY_ROPE = false,
    bool PUBLISH_V_FP8 = false,
    bool PUBLISH_V_BACKWARD_MXFP4 = true,
    bool PUBLISH_DOUT_STATS = true,
    bool PUBLISH_QK_FP8 = false,
    bool V_SEQUENCE_MAJOR_SCALES = false,
    bool PUBLISH_ALIGNED_QK = true,
    bool PACKED_ROPE = false,
    bool SHARED_PACKED_ROPE = false,
    bool CACHE_ADAPTIVE_QK_SCALE = false,
    bool NEGATE_DOUT_STATS = false,
    bool CLEAR_DQ = false,
    bool INTERLEAVE_CAUSAL_KV = false,
    bool PUBLISH_FORWARD_FP8 = true,
    bool PUBLISH_REPRESENTED_BACKWARD_FP8 = false,
    bool PER_BLOCK_QK_SCALES = false,
    bool EXPERIMENTAL_SPLIT_V_BACKWARD = false,
    bool EXPERIMENTAL_E4M3_DERIVED_MXFP4_V = false,
    bool EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V = false,
    bool EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V = false,
    bool EXPERIMENTAL_SHARED_TILE_MXFP4_V = false,
    bool PUBLISH_DOUT_E5M2 = false
>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;
    static_assert(!PURE_QK_SINGLE_QUANT || PUBLISH_PURE_QK);
    static_assert(!APPLY_ROPE || (!OUTPUT_IS_DOUT && !SINGLE_OUTPUT));
    static_assert(!PACKED_ROPE || (APPLY_ROPE && C::QK_DEPTH == 128));
    static_assert(!SHARED_PACKED_ROPE || PACKED_ROPE);
    static_assert(
        !PUBLISH_DOUT_E5M2 ||
            (
                OUTPUT_IS_DOUT && PUBLISH_V_FP8 && PUBLISH_DOUT_STATS &&
                NEGATE_DOUT_STATS && CLEAR_DQ && !STORE_BF16 &&
                !PUBLISH_FP4 && !PUBLISH_FORWARD_QK && !PUBLISH_V_MXFP4 &&
                !PUBLISH_QK_FP8 && !PUBLISH_FORWARD_FP8 &&
                !PUBLISH_REPRESENTED_BACKWARD_FP8
            ),
        "E5M2 dO publication is restricted to the v509 native-score "
        "output-dgrad route"
    );
    static_assert(
        !PUBLISH_REPRESENTED_BACKWARD_FP8 ||
            (PUBLISH_FP4 && PUBLISH_QK_FP8 && PUBLISH_V_FP8),
        "represented backward publication requires NVFP4 Q/K and the "
        "three backward E4M3 outputs"
    );
    static_assert(
        !PER_BLOCK_QK_SCALES ||
            (PUBLISH_FP4 && PUBLISH_FORWARD_QK),
        "per-block Q/K scales require forward NVFP4 publication"
    );
    static_assert(
        !EXPERIMENTAL_SPLIT_V_BACKWARD ||
            (PUBLISH_REPRESENTED_BACKWARD_FP8 && PER_BLOCK_QK_SCALES &&
             PUBLISH_V_MXFP4 && PUBLISH_V_FP8),
        "experimental split-V backward requires the represented per-block "
        "NVFP4-QK / MXFP4-V publication path"
    );
    static_assert(
        !EXPERIMENTAL_E4M3_DERIVED_MXFP4_V ||
            (PUBLISH_REPRESENTED_BACKWARD_FP8 && PER_BLOCK_QK_SCALES &&
             PUBLISH_V_MXFP4 && PUBLISH_V_FP8 && INTERLEAVE_CAUSAL_KV &&
             !PUBLISH_FORWARD_FP8 && !EXPERIMENTAL_SPLIT_V_BACKWARD),
        "E4M3-derived MXFP4 V requires the represented per-block causal "
        "NVFP4-QK path with a direct backward E4M3 V publication"
    );
    static_assert(
        !EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V ||
            (
                G::D_tile::rows == 128 && G::D_tile::cols == 32 &&
                !C::DENSE_FP8 && C::QK_DEPTH == 128 && !SINGLE_OUTPUT &&
                PUBLISH_V_MXFP4 &&
                !PUBLISH_FORWARD_FP8 &&
                !EXPERIMENTAL_E4M3_DERIVED_MXFP4_V &&
                !OUTPUT_IS_DOUT &&
                (
                    // Paired-D64 retains represented Q/K backward and a
                    // causal-interleaved forward V payload.
                    (PUBLISH_V_FP8 && !PUBLISH_V_BACKWARD_MXFP4 &&
                     PUBLISH_REPRESENTED_BACKWARD_FP8 &&
                     PER_BLOCK_QK_SCALES && INTERLEAVE_CAUSAL_KV &&
                     EXPERIMENTAL_SPLIT_V_BACKWARD) ||
                    // D128 retains ordinary order and publishes direct
                    // projection-accumulator E4M3 Q/K plus exactly one V
                    // representation: retained E4M3 or compact MXFP4.
                    (!PUBLISH_REPRESENTED_BACKWARD_FP8 &&
                     PER_BLOCK_QK_SCALES && !INTERLEAVE_CAUSAL_KV &&
                     !EXPERIMENTAL_SPLIT_V_BACKWARD && PUBLISH_QK_FP8 &&
                     (PUBLISH_V_FP8 != PUBLISH_V_BACKWARD_MXFP4))
                )
            ),
        "output-shared V publication requires the non-dense 128x32 output tile, "
        "direct-accumulator E4M3 backward, per-row-K16 Q/K, and direct "
        "BF16-to-MXFP4 forward publication contract"
    );
    static_assert(
        !EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V ||
            (
                EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V &&
                PUBLISH_V_MXFP4 && PUBLISH_V_BACKWARD_MXFP4 &&
                !PUBLISH_V_FP8 && !PUBLISH_REPRESENTED_BACKWARD_FP8 &&
                !INTERLEAVE_CAUSAL_KV && !OUTPUT_IS_DOUT &&
                C::QK_DEPTH == 128 && G::D_tile::rows == 128 &&
                G::D_tile::cols == 32
            ),
        "common-row MXFP4 V requires the experimental native-D128 "
        "output-shared MX-only backward route"
    );
    static_assert(
        !EXPERIMENTAL_SHARED_TILE_MXFP4_V ||
            (
                EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V &&
                PUBLISH_V_MXFP4 && PUBLISH_V_BACKWARD_MXFP4 &&
                !PUBLISH_V_FP8 && !PUBLISH_REPRESENTED_BACKWARD_FP8 &&
                V_SEQUENCE_MAJOR_SCALES && !INTERLEAVE_CAUSAL_KV &&
                !OUTPUT_IS_DOUT && !EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V &&
                C::QK_DEPTH == 128 && G::D_tile::rows == 128 &&
                G::D_tile::cols == 32
            ),
        "shared-tile MXFP4 V requires the native-D128 output-shared "
        "MX-only route with sequence-major forward scales"
    );
    static_assert(
        !C::DENSE_FP8 || !STORE_BF16 ||
            (
                SINGLE_OUTPUT && !OUTPUT_IS_DOUT && !PUBLISH_FP4 &&
                !PUBLISH_FORWARD_QK && !PUBLISH_V_MXFP4 &&
                !PUBLISH_V_FP8 && !PUBLISH_QK_FP8
            ),
        "dense E4M3 BF16 storage requires a publication-free single output"
    );
    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_tile>();
        g.B.template prefetch_tma<typename G::B_tile>();
        if constexpr (!C::DENSE_FP8) {
            g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
            g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        }
        if constexpr (STORE_BF16) {
            if constexpr (SINGLE_OUTPUT) {
                g.D.template prefetch_tma<typename G::D_tile>();
            } else {
                g.Q.template prefetch_tma<typename G::D_tile>();
                g.K.template prefetch_tma<typename G::D_tile>();
                g.V.template prefetch_tma<typename G::D_tile>();
            }
        }
    }

    const int warpgroup_id = kittens::warpgroup::groupid();
    const int cta_id = kittens::cluster_ctarank();
    const int cluster_id = kittens::clusterIdx().x;
    const int num_row_blocks = g.A.rows() / C::Mb;
    const int q_width = (OUTPUT_IS_DOUT || SINGLE_OUTPUT) ? 0 : g.Q.cols();
    const int k_width = (OUTPUT_IS_DOUT || SINGLE_OUTPUT) ? 0 : g.K.cols();
    const int total_width = SINGLE_OUTPUT
        ? g.output_width
        : q_width + k_width + g.v_width;
    const int num_col_blocks = total_width / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int block_begin = max(0, min(g.block_begin, num_blocks));
    const int block_end = g.block_end > block_begin
        ? min(g.block_end, num_blocks)
        : num_blocks;
    const int active_blocks = block_end - block_begin;
    const int num_red_blocks = C::DENSE_FP8
        ? g.A.cols() / C::Kb
        : 2 * g.A.cols() / C::Kb;
    const int blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    kittens::tma_swizzle_allocator allocator((int *)&__shm[0]);
    // Dense E4M3 doubles the staged A/B footprint.  If the output ring is
    // allocated after those stages its base lands above 128 KiB, where the
    // register-to-shared BF16 stmatrix path cannot address it reliably on
    // SM100.  Keep the output ring in the low shared-memory window for dense
    // mode; the legacy FP4 allocation order remains bit-for-bit unchanged.
    typename G::outputs_t *outputs_ptr = nullptr;
    if constexpr (C::DENSE_FP8) {
        outputs_ptr = &allocator.template allocate<
            typename G::outputs_t
        >();
    }
    typename G::input_tiles_t (&inputs)[C::LOAD_PIPE_DEPTH] =
        allocator.template allocate<typename G::input_tiles_t,
                                    C::LOAD_PIPE_DEPTH>();
    typename G::staged_input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        allocator.template allocate<typename G::staged_input_scales_t,
                                    C::LOAD_PIPE_DEPTH>();
    if constexpr (!C::DENSE_FP8) {
        outputs_ptr = &allocator.template allocate<
            typename G::outputs_t
        >();
    }
    typename G::outputs_t &outputs = *outputs_ptr;
    union reused_epilogue_scratch_t {
        uint8_t codes[kittens::WARPGROUP_WARPS][16][33];
        uint32_t bf16_pairs[
            kittens::WARPGROUP_WARPS
        ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE];
    };
    struct dual_epilogue_scratch_t {
        uint8_t codes[kittens::WARPGROUP_WARPS][16][33];
        uint32_t bf16_pairs[
            kittens::WARPGROUP_WARPS
        ][BF16_PAIR_SHARED_ROWS][BF16_PAIR_SHARED_STRIDE];
    };
    using epilogue_scratch_t = std::conditional_t<
        PUBLISH_FP4 && PUBLISH_QK_FP8 &&
            !PUBLISH_REPRESENTED_BACKWARD_FP8,
        dual_epilogue_scratch_t,
        reused_epilogue_scratch_t
    >;
    struct packed_rope_scratch_t {
        uint32_t data[
            PACKED_ROPE_SHARED_ROWS
        ][PACKED_ROPE_SHARED_STRIDE];
    };
    struct empty_rope_scratch_t {
        uint32_t value;
    };
    using rope_scratch_t = std::conditional_t<
        SHARED_PACKED_ROPE,
        packed_rope_scratch_t,
        empty_rope_scratch_t
    >;
    union shared_epilogue_scratch_t {
        epilogue_scratch_t epilogue;
        rope_scratch_t rope;
    };
    __shared__ shared_epilogue_scratch_t shared_epilogue_scratch;
    epilogue_scratch_t &epilogue_scratch =
        shared_epilogue_scratch.epilogue;
    struct perblock_qk_scale_scratch_t {
        uint8_t data[
            kittens::WARPGROUP_WARPS
        ][BF16_PAIR_SHARED_ROWS][2];
    };
    struct empty_qk_scale_scratch_t {
        uint8_t value;
    };
    using qk_scale_scratch_t = std::conditional_t<
        PER_BLOCK_QK_SCALES,
        perblock_qk_scale_scratch_t,
        empty_qk_scale_scratch_t
    >;
    __shared__ qk_scale_scratch_t qk_scale_scratch;
    const uint32_t *packed_rope_tile = nullptr;
    if constexpr (SHARED_PACKED_ROPE) {
        packed_rope_tile = &shared_epilogue_scratch.rope.data[0][0];
    }

    kittens::tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;
    __shared__ uint32_t tmem_addr;
    __shared__ kittens::semaphore tmem_provisioned;
    __shared__ kittens::semaphore tmem_finished;
    __shared__ kittens::semaphore inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ kittens::semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ kittens::semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ kittens::semaphore outputs_arrived;
    __shared__ kittens::semaphore outputs_finished;
    if (threadIdx.x == 32) {
        kittens::init_semaphore(tmem_provisioned, 0, 1);
        kittens::init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            kittens::init_semaphore(inputs_arrived[i], 0, 1);
            kittens::init_semaphore(scales_arrived[i], 0, 1);
            kittens::init_semaphore(inputs_finished[i], 0, 1);
        }
        kittens::init_semaphore(outputs_arrived, 0, 1);
        kittens::init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    kittens::everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS) {
        const int producer_warp =
            kittens::group<kittens::WARPGROUP_WARPS>::warpid();
        if (producer_warp == 3 && kittens::warp::elect_leader()) {
            kittens::everyone::tma::cluster::wait();
            for (int block_offset = cluster_id;
                 block_offset < active_blocks;
                 block_offset += gridDim.x / C::CLUSTER_SIZE) {
                const int block_idx = block_begin + block_offset;
                const int supergroup = block_idx / blocks_per_supergroup;
                const int within = block_idx % blocks_per_supergroup;
                const int rows_here = min(
                    C::SUPERGROUP_SIZE,
                    num_row_blocks - supergroup * C::SUPERGROUP_SIZE
                );
                const int row_block =
                    supergroup * C::SUPERGROUP_SIZE + within % rows_here;
                const int col_block = within / rows_here;
                for (int reduction = 0; reduction < num_red_blocks;
                     ++reduction) {
                    if (g.A_ready != nullptr) {
                        const int q_tile = row_block * 2 + cta_id;
                        wait_for_a_operand(
                            g.A_ready +
                                static_cast<size_t>(q_tile) *
                                    g.A_ready_reduction_tiles +
                                reduction,
                            g.A_ready_expected
                        );
                    }
                    kittens::wait(
                        inputs_finished[stage],
                        kittens::get_phasebit<1>(phasebits, stage)
                    );
                    kittens::tma::cluster::load_async(
                        inputs[stage].A,
                        g.A,
                        {row_block * 2 + cta_id, reduction},
                        inputs_arrived[stage],
                        static_cast<uint16_t>(1u << cta_id),
                        0
                    );
                    kittens::tma::cluster::load_async(
                        inputs[stage].B,
                        g.B,
                        {col_block * 2 + cta_id, reduction},
                        inputs_arrived[stage],
                        static_cast<uint16_t>(1u << cta_id),
                        0
                    );
                    kittens::update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (producer_warp == 2 && kittens::warp::elect_leader()) {
            if constexpr (!C::DENSE_FP8) {
                kittens::everyone::tma::cluster::wait();
                for (int block_offset = cluster_id;
                     block_offset < active_blocks;
                     block_offset += gridDim.x / C::CLUSTER_SIZE) {
                    const int block_idx = block_begin + block_offset;
                    const int supergroup = block_idx / blocks_per_supergroup;
                    const int within = block_idx % blocks_per_supergroup;
                    const int rows_here = min(
                        C::SUPERGROUP_SIZE,
                        num_row_blocks - supergroup * C::SUPERGROUP_SIZE
                    );
                    const int row_block =
                        supergroup * C::SUPERGROUP_SIZE + within % rows_here;
                    const int col_block = within / rows_here;
                    for (int reduction = 0; reduction < num_red_blocks;
                         ++reduction) {
                        if (g.A_ready != nullptr) {
                            const int q_tile = row_block * 2 + cta_id;
                            wait_for_a_operand(
                                g.A_ready +
                                    static_cast<size_t>(q_tile) *
                                        g.A_ready_reduction_tiles +
                                    reduction,
                                g.A_ready_expected
                            );
                        }
                        kittens::wait(
                            inputs_finished[stage],
                            kittens::get_phasebit<1>(phasebits, stage)
                        );
                        kittens::tma::cluster::load_async(
                            input_scales[stage].A,
                            g.A_sc,
                            {row_block * 2 + cta_id, reduction, 0},
                            scales_arrived[stage],
                            static_cast<uint16_t>(1u << cta_id),
                            0
                        );
                        kittens::tma::cluster::load_async(
                            input_scales[stage].B[cta_id],
                            g.B_sc,
                            {col_block * 2 + cta_id, reduction, 0},
                            scales_arrived[stage],
                            static_cast<uint16_t>(0b11),
                            0
                        );
                        kittens::update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            }
        } else if (producer_warp == 1) {
            if constexpr (CLEAR_DQ) {
                if (g.dq_clear != nullptr) {
                    const int64_t first =
                        static_cast<int64_t>(blockIdx.x) *
                            kittens::WARP_THREADS +
                        kittens::warp::laneid();
                    const int64_t stride =
                        static_cast<int64_t>(gridDim.x) *
                            kittens::WARP_THREADS;
                    const uint4 zero = make_uint4(0, 0, 0, 0);
                    for (int64_t vector = first;
                         vector < g.dq_clear_vectors;
                         vector += stride) {
                        g.dq_clear[vector] = zero;
                    }
                }
            }
        } else if (
            cta_id == 0 && producer_warp == 0 &&
            kittens::warp::elect_leader()
        ) {
            kittens::everyone::tma::cluster::wait();
            kittens::wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto output_tm =
                tm_allocator.template allocate<kittens::full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<
                kittens::full_tt_fp8e4m3<
                    16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH
                >
            >(256);
            auto B_sc_tm = tm_allocator.template allocate<
                kittens::full_tt_fp8e4m3<
                    32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH
                >
            >(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
            for (int block_offset = cluster_id;
                 block_offset < active_blocks;
                 block_offset += gridDim.x / C::CLUSTER_SIZE) {
                const int block_idx = block_begin + block_offset;
                kittens::wait(
                    outputs_finished,
                    kittens::get_phasebit<1>(phasebits, 0)
                );
                kittens::tensor_after_thread_sync();
                for (int reduction = 0; reduction < num_red_blocks;
                     ++reduction) {
                    if constexpr (!C::DENSE_FP8) {
                    kittens::tma::expect_bytes(
                        scales_arrived[stage],
                        2 * sizeof(typename G::input_scales_t)
                    );
                    kittens::wait(
                        scales_arrived[stage],
                        kittens::get_phasebit<0>(phasebits, stage)
                    );
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_subtile = A_sc_tm.template subtile<
                            kittens::full_tt_fp8e4m3<16>
                        >(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_shared = *reinterpret_cast<
                            kittens::st_fp8e4m3<32, 16, false> *
                        >(
                            reinterpret_cast<uint64_t>(
                                &input_scales[stage].A.data[0]
                            ) + 16 * 32 * ii
                        );
                        load_mxnv_scale_async2(
                            A_sc_subtile,
                            A_sc_shared
                        );
                        auto B_sc_subtile0 = B_sc_tm.template subtile<
                            kittens::full_tt_fp8e4m3<16>
                        >(
                            stage * C::MMA_PER_TILE * 32 +
                            ii * C::B_SC_SIZE * 16
                        );
                        auto &B_sc_shared0 = *reinterpret_cast<
                            kittens::st_fp8e4m3<32, 16, false> *
                        >(
                            reinterpret_cast<uint64_t>(
                                &input_scales[stage].B[0].data[0]
                            ) + 16 * 32 * ii
                        );
                        load_mxnv_scale_async2(
                            B_sc_subtile0,
                            B_sc_shared0
                        );
                        auto B_sc_subtile1 = B_sc_tm.template subtile<
                            kittens::full_tt_fp8e4m3<16>
                        >(
                            stage * C::MMA_PER_TILE * 32 +
                            ii * C::B_SC_SIZE * 16 + 16
                        );
                        auto &B_sc_shared1 = *reinterpret_cast<
                            kittens::st_fp8e4m3<32, 16, false> *
                        >(
                            reinterpret_cast<uint64_t>(
                                &input_scales[stage].B[1].data[0]
                            ) + 16 * 32 * ii
                        );
                        load_mxnv_scale_async2(
                            B_sc_subtile1,
                            B_sc_shared1
                        );
                    }
                    }
                    kittens::tma::expect_bytes(
                        inputs_arrived[stage],
                        2 * sizeof(typename G::input_tiles_t)
                    );
                    kittens::wait(
                        inputs_arrived[stage],
                        kittens::get_phasebit<0>(phasebits, stage)
                    );
                    if (reduction == 0) {
                        if constexpr (C::DENSE_FP8) {
                        kittens::mm2_ABt(
                            output_tm,
                            inputs[stage].A,
                            inputs[stage].B,
                            inputs_finished[stage]
                        );
                        } else {
                        kittens::mm2_ABt(
                            output_tm,
                            inputs[stage].A,
                            inputs[stage].B,
                            A_sc_tm.template subtile<
                                kittens::full_tt_fp8e4m3<
                                    C::MMA_PER_TILE * 16
                                >
                            >(stage * C::MMA_PER_TILE * 16),
                            B_sc_tm.template subtile<
                                kittens::full_tt_fp8e4m3<
                                    C::MMA_PER_TILE * 32
                                >
                            >(stage * C::MMA_PER_TILE * 32),
                            inputs_finished[stage]
                        );
                        }
                    } else {
                        if constexpr (C::DENSE_FP8) {
                        kittens::mma2_ABt(
                            output_tm,
                            inputs[stage].A,
                            inputs[stage].B,
                            inputs_finished[stage]
                        );
                        } else {
                        kittens::mma2_ABt(
                            output_tm,
                            inputs[stage].A,
                            inputs[stage].B,
                            A_sc_tm.template subtile<
                                kittens::full_tt_fp8e4m3<
                                    C::MMA_PER_TILE * 16
                                >
                            >(stage * C::MMA_PER_TILE * 16),
                            B_sc_tm.template subtile<
                                kittens::full_tt_fp8e4m3<
                                    C::MMA_PER_TILE * 32
                                >
                            >(stage * C::MMA_PER_TILE * 32),
                            inputs_finished[stage]
                        );
                        }
                    }
                    kittens::update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                kittens::tensor_commit<2>(outputs_arrived);
                kittens::update_phasebit<1>(phasebits, 0);
            }
        }
    } else {
        kittens::everyone::tma::cluster::wait_aligned();
        if (kittens::warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            kittens::warp::arrive(tmem_provisioned);
        }
        kittens::wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto output_tm =
            tm_allocator.template allocate<kittens::full_tt_fl<C::Nb>>(0);
        constexpr int kSlice = C::Nb / C::EPI_PIPE_DEPTH;
        using accum_rt = kittens::rt_fl<C::Mb / 8, kSlice>;
        using output_rt = kittens::rt_bf<C::Mb / 8, kSlice>;
        using row_decode_rv = kittens::col_vec<accum_rt>;
        using column_decode_rv = kittens::row_vec<accum_rt>;
        uint32_t rope_head_pair_cache[
            output_rt::height * output_rt::width * 4
        ];
        const float global_scale = C::DENSE_FP8
            ? 1.0f
            : g.A_scale[{0}] * g.B_scale[{0}];
        uint32_t output_phasebits = 0;

        for (int block_offset = cluster_id;
             block_offset < active_blocks;
             block_offset += gridDim.x / C::CLUSTER_SIZE) {
            const int block_idx = block_begin + block_offset;
            const int supergroup = block_idx / blocks_per_supergroup;
            const int within = block_idx % blocks_per_supergroup;
            const int rows_here = min(
                C::SUPERGROUP_SIZE,
                num_row_blocks - supergroup * C::SUPERGROUP_SIZE
            );
            const int row_block =
                supergroup * C::SUPERGROUP_SIZE + within % rows_here;
            const int col_block = within / rows_here;
            const int global_row_base =
                (row_block * 2 + cta_id) * (C::Mb / 2);
            row_decode_rv row_decode;
            if constexpr (C::DENSE_FP8) {
                kittens::warpgroup::load(
                    row_decode,
                    g.A_sc,
                    {global_row_base / (C::Mb / 2)}
                );
            }
            if constexpr (SHARED_PACKED_ROPE) {
                if (col_block * C::Nb < q_width + k_width) {
                    constexpr int kVectorsPerRow =
                        PACKED_ROPE_SHARED_PAIRS / 4;
                    constexpr int kVectorCount =
                        PACKED_ROPE_SHARED_ROWS * kVectorsPerRow;
                    const uint4 *source = reinterpret_cast<const uint4 *>(
                        g.rope_packed +
                        static_cast<size_t>(global_row_base) *
                            PACKED_ROPE_SHARED_PAIRS
                    );
                    #pragma unroll
                    for (int item = threadIdx.x; item < kVectorCount;
                         item += kittens::WARPGROUP_WARPS *
                             kittens::WARP_THREADS) {
                        const int row = item / kVectorsPerRow;
                        const int column_vector = item % kVectorsPerRow;
                        *reinterpret_cast<uint4 *>(
                            &shared_epilogue_scratch.rope.data[
                                row
                            ][column_vector * 4]
                        ) = source[item];
                    }
                    kittens::warpgroup::sync(1);
                }
            }
            kittens::wait(
                outputs_arrived,
                kittens::get_phasebit<0>(output_phasebits, 0)
            );
            // The complete previous tile can remain in this shared page while
            // the next tensor-core tile is produced.  Wait only before the
            // page is recycled, then drain all TMEM slices into it.
            kittens::warpgroup::tma::store_async_read_wait<0>();
            const int block_batch_idx = global_row_base / g.seq_len;
            const int block_seq_base =
                global_row_base - block_batch_idx * g.seq_len;
            #pragma unroll
            for (int ordered_epi = 0;
                 ordered_epi < C::EPI_PIPE_DEPTH;
                 ++ordered_epi) {
                // D128 packs two heads into each N256 tile.  Visit matching
                // N32 slices consecutively so the second head reuses the
                // first head's packed RoPE values from registers.
                const int epi = APPLY_ROPE && C::QK_DEPTH == 128
                    ? (ordered_epi >> 1) +
                        ((ordered_epi & 1) * (C::EPI_PIPE_DEPTH / 2))
                    : ordered_epi;
                accum_rt accumulator;
                output_rt registers;
                kittens::warpgroup::load_async(
                    accumulator,
                    output_tm.template subtile<
                        kittens::full_tt_fl<kSlice>
                    >(0, epi * kSlice)
                );
                kittens::tensor_load_wait();
                kittens::tensor_before_thread_sync();
                kittens::warpgroup::sync(1);
                if constexpr (C::DENSE_FP8) {
                    column_decode_rv column_decode;
                    kittens::warp::load(
                        column_decode,
                        g.B_sc,
                        {(col_block * C::Nb + epi * kSlice) / kSlice}
                    );
                    kittens::warp::mul_col(
                        accumulator,
                        accumulator,
                        column_decode
                    );
                    kittens::warp::mul_row(
                        accumulator,
                        accumulator,
                        row_decode
                    );
                    kittens::warp::copy(registers, accumulator);
                } else {
                    kittens::warp::mul(
                        accumulator,
                        accumulator,
                        global_scale
                    );
                    kittens::warp::copy(registers, accumulator);
                }
                if constexpr (APPLY_ROPE) {
                    const int combined_col =
                        col_block * C::Nb + epi * kSlice;
                    const bool is_q = combined_col < q_width;
                    const bool is_k =
                        combined_col >= q_width &&
                        combined_col < q_width + k_width;
                    if (is_q || is_k) {
                        const int local_col = is_q
                            ? combined_col
                            : combined_col - q_width;
                        if constexpr (C::QK_DEPTH == 128) {
                            if ((ordered_epi & 1) == 0) {
                                apply_rope_tile_head_pair_cached<
                                    false,
                                    PACKED_ROPE,
                                    SHARED_PACKED_ROPE
                                >(
                                    g,
                                    registers,
                                    global_row_base,
                                    local_col,
                                    rope_head_pair_cache,
                                    packed_rope_tile
                                );
                            } else {
                                apply_rope_tile_head_pair_cached<
                                    true,
                                    PACKED_ROPE,
                                    SHARED_PACKED_ROPE
                                >(
                                    g,
                                    registers,
                                    global_row_base,
                                    local_col,
                                    rope_head_pair_cache,
                                    packed_rope_tile
                                );
                            }
                        } else {
                            apply_rope_tile(
                                g,
                                registers,
                                global_row_base,
                                local_col
                            );
                        }
                    }
                }
                kittens::warpgroup::store(outputs.D[epi], registers);
                kittens::warpgroup::sync(1);
                if (ordered_epi == C::EPI_PIPE_DEPTH - 1) {
                    // TMEM is now fully copied into shared memory; the MMA
                    // producer can start the next block while this block is
                    // packed and sent to HBM.
                    kittens::warpgroup::tma::cluster::arrive(
                        outputs_finished,
                        0,
                        1
                    );
                }
            }
            float dpsum_accumulator = 0.0f;
            float direct_dpsum_accumulators[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f
            };
            // A D128 head occupies four consecutive N32 epilogue slices.
            // Its adaptive Q/K scale is uniform across those slices, so keep
            // the first lookup live instead of rebuilding the metadata address
            // and reloading the same scalar four times.
            float cached_adaptive_qk_scale = 16.0f;
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                const int combined_col = col_block * C::Nb + epi * kSlice;
                const bool is_q =
                    !OUTPUT_IS_DOUT && !SINGLE_OUTPUT &&
                    combined_col < q_width;
                const bool is_k =
                    !OUTPUT_IS_DOUT && !SINGLE_OUTPUT &&
                    combined_col >= q_width &&
                    combined_col < q_width + k_width;
                const bool is_v =
                    !SINGLE_OUTPUT &&
                    (OUTPUT_IS_DOUT || combined_col >= q_width + k_width);
                const int local_col = is_q
                    ? combined_col
                    : (is_k ? combined_col - q_width
                            : combined_col - q_width - k_width);
                if constexpr (STORE_BF16) {
                    if constexpr (SINGLE_OUTPUT) {
                        kittens::warpgroup::tma::store_async<
                            kittens::dim::ROW,
                            kittens::cache_policy::EVICT_FIRST
                        >(
                            g.D,
                            outputs.D[epi],
                            {
                                row_block * 2 + cta_id,
                                combined_col / kSlice
                            }
                        );
                    } else if (is_q) {
                        kittens::warpgroup::tma::store_async<
                            kittens::dim::ROW,
                            kittens::cache_policy::EVICT_FIRST
                        >(
                            g.Q,
                            outputs.D[epi],
                            {row_block * 2 + cta_id, local_col / kSlice}
                        );
                    } else if (is_k) {
                        kittens::warpgroup::tma::store_async<
                            kittens::dim::ROW,
                            kittens::cache_policy::EVICT_FIRST
                        >(
                            g.K,
                            outputs.D[epi],
                            {row_block * 2 + cta_id, local_col / kSlice}
                        );
                    } else {
                        kittens::warpgroup::tma::store_async<
                            kittens::dim::ROW,
                            kittens::cache_policy::EVICT_FIRST
                        >(
                            g.V,
                            outputs.D[epi],
                            {row_block * 2 + cta_id, local_col / kSlice}
                        );
                    }
                }
                if constexpr (
                    PUBLISH_FP4 || PUBLISH_V_MXFP4 || PUBLISH_V_FP8 ||
                    PUBLISH_QK_FP8
                ) {
                    if constexpr (
                        PUBLISH_V_FP8 && !PUBLISH_FP4 &&
                        !PUBLISH_V_MXFP4
                    ) {
                        if (is_v) {
                            publish_v_fp8_from_output_shared<
                                C,
                                OUTPUT_IS_DOUT && PUBLISH_DOUT_STATS,
                                NEGATE_DOUT_STATS,
                                PUBLISH_DOUT_E5M2
                            >(
                                g,
                                outputs.D[epi],
                                global_row_base,
                                local_col,
                                direct_dpsum_accumulators
                            );
                            kittens::warpgroup::sync(1);
                        }
                    } else {
                    if constexpr (EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V) {
                        if (is_v) {
                            // The complete BF16 output slice is still live in
                            // the projection ring. All enabled publishers are
                            // read-only: paired D64 emits backward E4M3 plus
                            // forward MX, while native D128 emits forward MX
                            // plus exactly one backward representation (E4M3
                            // or MX). No register reload, second shared staging
                            // pass, or barrier is needed.
                            if constexpr (PUBLISH_V_FP8) {
                                publish_v_fp8_from_output_shared<C, false>(
                                    g,
                                    outputs.D[epi],
                                    global_row_base,
                                    local_col,
                                    direct_dpsum_accumulators
                                );
                            }
                            publish_v_mxfp4_from_output_shared<
                                C,
                                OUTPUT_IS_DOUT || V_SEQUENCE_MAJOR_SCALES,
                                INTERLEAVE_CAUSAL_KV,
                                PUBLISH_V_BACKWARD_MXFP4 &&
                                    !EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V,
                                EXPERIMENTAL_SHARED_TILE_MXFP4_V
                            >(
                                g,
                                outputs.D[epi],
                                block_batch_idx,
                                block_seq_base,
                                local_col
                            );
                            if constexpr (
                                EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V
                            ) {
                                // Publish backward V once, after visiting the
                                // final D32 slice of each D128 head.  Forward
                                // MX above remains the retained per-slice path
                                // and is therefore bitwise unchanged.
                                if ((local_col & 127) == 96) {
                                    publish_v_common_rowscale_mxfp4_from_output_ring<
                                        C
                                    >(
                                        g,
                                        outputs,
                                        epi - 3,
                                        block_batch_idx,
                                        block_seq_base,
                                        local_col - 96
                                    );
                                }
                            }
                            continue;
                        }
                    }
                    output_rt registers;
                    kittens::warpgroup::load(registers, outputs.D[epi]);
                    kittens::warpgroup::sync(1);
                    if constexpr (PUBLISH_FP4) {
                        if (!is_v) {
                            float scale;
                            if constexpr (PER_BLOCK_QK_SCALES) {
                                scale = 1.0f;
                            } else if constexpr (PURE_QK_SINGLE_QUANT) {
                                scale = 16.0f;
                            } else {
                                const int batch_idx =
                                    block_batch_idx;
                                const int metadata_head_stride =
                                    g.Q.cols() / C::QK_DEPTH;
                                if constexpr (
                                    C::QK_DEPTH == 128 &&
                                    CACHE_ADAPTIVE_QK_SCALE
                                ) {
                                    if ((epi & 3) == 0) {
                                        const int head_idx = local_col / 128;
                                        cached_adaptive_qk_scale =
                                            g.adaptive_scales[
                                                (static_cast<size_t>(
                                                     batch_idx
                                                 ) * metadata_head_stride +
                                                 head_idx) * 7 +
                                                (is_k ? 1 : 0)
                                            ];
                                    }
                                    scale = cached_adaptive_qk_scale;
                                } else {
                                    const int head_idx =
                                        local_col / C::QK_DEPTH;
                                    scale = g.adaptive_scales[
                                        (static_cast<size_t>(batch_idx) *
                                             metadata_head_stride +
                                         head_idx) * 7 +
                                        (is_k ? 1 : 0)
                                    ];
                                }
                            }
                            if constexpr (PER_BLOCK_QK_SCALES) {
                                stage_codes_perblock_qk<
                                    PUBLISH_QK_FP8 &&
                                        !PUBLISH_REPRESENTED_BACKWARD_FP8
                                >(
                                    registers,
                                    epilogue_scratch.codes,
                                    epilogue_scratch.bf16_pairs,
                                    qk_scale_scratch.data
                                );
                            } else if constexpr (
                                PUBLISH_QK_FP8 &&
                                !PUBLISH_REPRESENTED_BACKWARD_FP8
                            ) {
                                stage_codes_and_bf16_pairs(
                                    registers,
                                    epilogue_scratch.codes,
                                    epilogue_scratch.bf16_pairs,
                                    scale
                                );
                            } else {
                                stage_codes(
                                    registers,
                                    epilogue_scratch.codes,
                                    scale
                                );
                            }
                            kittens::warpgroup::sync(1);
                            publish_codes<
                                C,
                                PUBLISH_ALIGNED_QK,
                                INTERLEAVE_CAUSAL_KV
                            >(
                                g,
                                epilogue_scratch.codes,
                                block_batch_idx,
                                block_seq_base,
                                local_col,
                                is_k
                            );
                            if constexpr (
                                PUBLISH_REPRESENTED_BACKWARD_FP8
                            ) {
                                if constexpr (PER_BLOCK_QK_SCALES) {
                                    publish_qk_fp8_from_perblock_codes(
                                        g,
                                        epilogue_scratch.codes,
                                        qk_scale_scratch.data,
                                        block_batch_idx,
                                        block_seq_base,
                                        local_col,
                                        is_k
                                    );
                                } else {
                                    publish_qk_fp8_from_codes(
                                        g,
                                        epilogue_scratch.codes,
                                        block_batch_idx,
                                        block_seq_base,
                                        local_col,
                                        is_k,
                                        scale
                                    );
                                }
                            }
                            if constexpr (
                                !PUBLISH_QK_FP8 ||
                                PUBLISH_REPRESENTED_BACKWARD_FP8
                            ) {
                                kittens::warpgroup::sync(1);
                            }
                            if constexpr (PUBLISH_PURE_QK) {
                                if constexpr (!PURE_QK_SINGLE_QUANT) {
                                    stage_codes(
                                        registers,
                                        epilogue_scratch.codes,
                                        16.0f
                                    );
                                    kittens::warpgroup::sync(1);
                                }
                                publish_sequence_compact(
                                    g,
                                    epilogue_scratch.codes,
                                    global_row_base,
                                    local_col,
                                    is_k
                                );
                                kittens::warpgroup::sync(1);
                            }
                        }
                    }
                    if constexpr (
                        PUBLISH_QK_FP8 &&
                        !PUBLISH_REPRESENTED_BACKWARD_FP8
                    ) {
                        if (!is_v) {
                            if constexpr (!PUBLISH_FP4) {
                                stage_bf16_pairs(
                                    registers,
                                    epilogue_scratch.bf16_pairs
                                );
                                kittens::warpgroup::sync(1);
                            }
                            publish_qk_fp8(
                                g,
                                epilogue_scratch.bf16_pairs,
                                block_batch_idx,
                                block_seq_base,
                                local_col,
                                is_k
                            );
                            kittens::warpgroup::sync(1);
                        }
                    }
                    if constexpr (PUBLISH_V_MXFP4 || PUBLISH_V_FP8) {
                        if (is_v) {
                            stage_bf16_pairs(
                                registers,
                                epilogue_scratch.bf16_pairs
                            );
                            kittens::warpgroup::sync(1);
                            if constexpr (
                                EXPERIMENTAL_E4M3_DERIVED_MXFP4_V
                            ) {
                                // Write backward's direct E4M3(x4) V first
                                // and retain the exact same bytes in scratch.
                                // Forward MXFP4 then consumes those bytes;
                                // no E4M3 tensor round-trip or second global
                                // publication is introduced.
                                publish_v_fp8<C, false, true>(
                                    g,
                                    epilogue_scratch.bf16_pairs,
                                    block_batch_idx,
                                    block_seq_base,
                                    local_col
                                );
                                // The causal gather reads rows staged by all
                                // four warps, so this synchronization is part
                                // of the bitwise derived-publication contract.
                                kittens::warpgroup::sync(1);
                                publish_v_mxfp4_from_backward_e4m3<
                                    C,
                                    OUTPUT_IS_DOUT || V_SEQUENCE_MAJOR_SCALES,
                                    INTERLEAVE_CAUSAL_KV
                                >(
                                    g,
                                    epilogue_scratch.bf16_pairs,
                                    block_batch_idx,
                                    block_seq_base,
                                    local_col
                                );
                            } else {
                                if constexpr (
                                    PUBLISH_V_FP8 &&
                                    EXPERIMENTAL_SPLIT_V_BACKWARD &&
                                    !PUBLISH_FORWARD_FP8
                                ) {
                                    // Split V keeps the direct backward
                                    // E4M3 publication independent of the
                                    // forward MXFP4 payload.  Issue that
                                    // read-only consumer first; unlike the
                                    // forward-FP8 specialization, it does
                                    // not replace the staged BF16 pairs.
                                    publish_v_fp8<C, false>(
                                        g,
                                        epilogue_scratch.bf16_pairs,
                                        block_batch_idx,
                                        block_seq_base,
                                        local_col
                                    );
                                }
                                if constexpr (PUBLISH_V_MXFP4) {
                                    publish_v_mxfp4<
                                        C,
                                        OUTPUT_IS_DOUT ||
                                            V_SEQUENCE_MAJOR_SCALES,
                                        PUBLISH_V_BACKWARD_MXFP4,
                                        INTERLEAVE_CAUSAL_KV,
                                        PUBLISH_REPRESENTED_BACKWARD_FP8 &&
                                            PUBLISH_V_MXFP4 &&
                                            !EXPERIMENTAL_SPLIT_V_BACKWARD
                                    >(
                                        g,
                                        epilogue_scratch.bf16_pairs,
                                        block_batch_idx,
                                        block_seq_base,
                                        local_col
                                    );
                                }
                                if constexpr (
                                    PUBLISH_V_FP8 &&
                                    (!PUBLISH_REPRESENTED_BACKWARD_FP8 ||
                                     !PUBLISH_V_MXFP4 ||
                                     EXPERIMENTAL_SPLIT_V_BACKWARD) &&
                                    (!EXPERIMENTAL_SPLIT_V_BACKWARD ||
                                     PUBLISH_FORWARD_FP8)
                                ) {
                                    publish_v_fp8<C, PUBLISH_FORWARD_FP8>(
                                        g,
                                        epilogue_scratch.bf16_pairs,
                                        block_batch_idx,
                                        block_seq_base,
                                        local_col
                                    );
                                }
                            }
                            if constexpr (
                                OUTPUT_IS_DOUT && PUBLISH_DOUT_STATS
                            ) {
                                const int warp =
                                    kittens::warpgroup::warpid();
                                const int lane = kittens::warp::laneid();
                                const int row =
                                    global_row_base + warp * 32 + lane;
                                const int batch_idx = row / g.seq_len;
                                const int seq_idx = row -
                                    batch_idx * g.seq_len;
                                const int head_idx =
                                    local_col / g.head_depth;
                                const int depth_base =
                                    local_col - head_idx * g.head_depth;
                                float partial = 0.0f;
                                #pragma unroll
                                for (int depth_pair = 0; depth_pair < 16;
                                     ++depth_pair) {
                                    const kittens::bf16_2 dout_pair =
                                        *reinterpret_cast<
                                            const kittens::bf16_2 *
                                        >(
                                            &epilogue_scratch.bf16_pairs[
                                                warp
                                            ][lane][depth_pair]
                                        );
                                    const size_t output_offset =
                                        ((static_cast<size_t>(batch_idx) *
                                              g.seq_len +
                                          seq_idx) * g.heads +
                                         head_idx) * g.head_depth +
                                        depth_base + 2 * depth_pair;
                                    const kittens::bf16_2 output_pair =
                                        *reinterpret_cast<
                                            const kittens::bf16_2 *
                                        >(g.attention_output + output_offset);
                                    const float2 dout_values =
                                        __bfloat1622float2(dout_pair);
                                    const float2 output_values =
                                        __bfloat1622float2(output_pair);
                                    partial += output_values.x *
                                                   dout_values.x +
                                        output_values.y * dout_values.y;
                                }
                                dpsum_accumulator += partial;
                                if (depth_base == g.head_depth - 32) {
                                    const size_t stats_offset =
                                        (static_cast<size_t>(batch_idx) *
                                             g.heads +
                                         head_idx) * g.seq_len + seq_idx;
                                    // E4M3 dP lifts both operands by four and
                                    // therefore centers an x16 accumulator.
                                    // Native MXFP4 dP instead applies the
                                    // standard 1/36 reconstruction correction
                                    // before centering, so its row statistic
                                    // must remain in the represented, unscaled
                                    // O dot dO domain.
                                    constexpr float dpsum_scale =
                                        PUBLISH_V_MXFP4 && !PUBLISH_V_FP8
                                        ? 1.0f
                                        : 16.0f;
                                    g.dpsum[stats_offset] =
                                        dpsum_accumulator * dpsum_scale;
                                    const size_t lse_offset =
                                        g.lse_head_major
                                        ? (static_cast<size_t>(batch_idx) *
                                               g.heads +
                                           head_idx) * g.seq_len + seq_idx
                                        : (static_cast<size_t>(batch_idx) *
                                               g.seq_len +
                                           seq_idx) * g.heads + head_idx;
                                    g.lse_log2[stats_offset] =
                                        g.lse[lse_offset] *
                                        1.4426950408889634f;
                                    dpsum_accumulator = 0.0f;
                                }
                            }
                            kittens::warpgroup::sync(1);
                        }
                    }
                    }
                }
                if constexpr (PUBLISH_FORWARD_QK) {
                    if (!is_v) {
                        if constexpr (PER_BLOCK_QK_SCALES) {
                            publish_forward_qk_perblock_scales<
                                C,
                                INTERLEAVE_CAUSAL_KV
                            >(
                                g,
                                qk_scale_scratch.data,
                                block_batch_idx,
                                block_seq_base,
                                local_col,
                                is_k
                            );
                        } else {
                            publish_forward_qk_scales<
                                C,
                                PURE_QK_SINGLE_QUANT
                            >(
                                g,
                                block_batch_idx,
                                block_seq_base,
                                local_col,
                                is_k
                            );
                        }
                        kittens::warpgroup::sync(1);
                    }
                }
            }
            kittens::update_phasebit<0>(output_phasebits, 0);
        }
        kittens::warpgroup::sync(1);
        kittens::warpgroup::tma::store_async_read_wait<0>();
        if (kittens::warpgroup::warpid() == 0) {
            if (kittens::warp::elect_leader()) {
                kittens::tma::cluster::arrive(tmem_finished, 1 - cta_id);
            }
            kittens::wait(tmem_finished, 0);
            tm_allocator.deprovision();
        }
    }
}

// One exact specialization for the fused v509 producer. Keeping this named
// wrapper beside host dispatch prevents constructing a partially compatible
// boolean combination at a call site.
template <typename C>
__device__ inline void kernel_v509_native_score_e5m2_dout(
    const globals<C> &g
) {
    kernel<
        C,
        false, false, false, false, true,
        false, false, false, false, true,
        false, true, false, false, true,
        false, false, false, true, true,
        false, false, false, false, false,
        false, false, false, false, true
    >(g);
}

template <
    typename C,
    bool PUBLISH_FP4,
    bool PUBLISH_FORWARD_QK = false,
    bool PUBLISH_V_MXFP4 = false,
    bool STORE_BF16 = true,
    bool OUTPUT_IS_DOUT = false,
    bool PUBLISH_PURE_QK = false,
    bool PURE_QK_SINGLE_QUANT = false,
    bool SINGLE_OUTPUT = false,
    bool APPLY_ROPE = false
>
inline void launch_on_stream(const globals<C> &g, cudaStream_t stream) {
    using G = globals<C>;
    CUDACHECK(cudaFuncSetAttribute(
        kittens::py::global_kernel<
            C,
            G,
            kernel<
                C,
                PUBLISH_FP4,
                PUBLISH_FORWARD_QK,
                PUBLISH_V_MXFP4,
                STORE_BF16,
                OUTPUT_IS_DOUT,
                PUBLISH_PURE_QK,
                PURE_QK_SINGLE_QUANT,
                SINGLE_OUTPUT,
                APPLY_ROPE
            >
        >,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()
    ));
    kittens::LaunchConfig<true, false> launch_config(
        g.grid(),
        g.block(),
        g.dynamic_shared_memory(),
        stream,
        dim3(C::CLUSTER_SIZE, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kittens::py::global_kernel<
            C,
            G,
            kernel<
                C,
                PUBLISH_FP4,
                PUBLISH_FORWARD_QK,
                PUBLISH_V_MXFP4,
                STORE_BF16,
                OUTPUT_IS_DOUT,
                PUBLISH_PURE_QK,
                PURE_QK_SINGLE_QUANT,
                SINGLE_OUTPUT,
                APPLY_ROPE
            >
        >,
        g
    ));
}

} // namespace tkfa4_projection

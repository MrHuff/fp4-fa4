#pragma once

// Experimental feature-major E4M3(x4) -> causal-interleaved MXFP4 V
// publication.  The exact projection route already owns a contiguous
// [B,H,D,S] E4M3 V tensor whose physical values are four times logical V.
// One warp consumes one depth row and one 128-token tile: lane j's four-byte
// load contains logical sequence positions 4*j + {0,1,2,3}, exactly the four
// 32-token causal quarters required by the MXFP4 forward kernel.  The
// permutation therefore remains register-only.

#include <cstdint>

#include "projection_fp4_epilogue.cuh"

namespace tkfa4_e4m3_to_mxfp4_v {

constexpr int kHeadDepth = 64;
constexpr int kSequenceTile = 128;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kScalePageBytes = 512;
constexpr uint32_t kE4m3MagnitudeMask = 0x7f7f7f7fu;
constexpr uint8_t kE4m3NanMagnitude = 0x7fu;
constexpr uint8_t kE8m0Nan = 0xffu;

static_assert(kHeadDepth % kWarpsPerBlock == 0);
static_assert(kScalePageBytes % sizeof(uint32_t) == 0);

__device__ __forceinline__ uint32_t warp_max_u8x4(uint32_t value) {
    // Each byte is an independent causal quarter.  One native SIMD byte-max
    // instruction per shuffle replaces four separate five-shuffle chains.
    // E4M3 finite magnitudes are monotonically ordered by their unsigned
    // seven-bit encoding, so this is byte-exact for every finite input.
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        value = __vmaxu4(
            value,
            __shfl_xor_sync(0xffffffffu, value, mask)
        );
    }
    return value;
}

__device__ __forceinline__ uint8_t logical_e8m0_from_physical_amax(
    uint8_t physical_amax_code
) {
    // E4M3FN has no infinity encoding.  Canonicalize any 0x7f magnitude to
    // the E8M0 NaN sentinel.  Its payload is zeroed below, so a NaN in one
    // 1x32 group cannot feed an undefined FP4 conversion or contaminate a
    // neighboring quarter.  This does not change any finite-byte result.
    if (physical_amax_code == kE4m3NanMagnitude) {
        return kE8m0Nan;
    }
    const uint32_t exponent = physical_amax_code >> 3;
    const uint32_t mantissa = physical_amax_code & 7u;
    if (exponent != 0) {
        // E4M3's normalized BF16 exponent is e + 120.  The 1x32 MSE
        // selector rounds up at BF16 mantissa 0x1a; an E4M3 mantissa is
        // m << 4, so exactly m >= 2 selects the upper exponent.  Subtract
        // log2(4) from that physical exponent for the logical MX scale.
        return static_cast<uint8_t>(
            exponent + 118u + static_cast<uint32_t>(mantissa >= 2u)
        );
    }
    // Exact logical exponents for E4M3 subnormals m * 2^-9 after undoing
    // the projection publisher's x4 lift and applying the same 0x1a MSE
    // cutoff.  Byte m of the packed table is selected without FP8->FP32,
    // FP32->BF16, or CLZ conversion work.
    constexpr uint64_t kSubnormalLogicalE8m0 = 0x7777777676757400ull;
    return static_cast<uint8_t>(
        kSubnormalLogicalE8m0 >> (mantissa * 8)
    );
}

__device__ __forceinline__ uint32_t
logical_e8m0_encode_multiplier_half2(uint8_t exponent) {
    if (exponent == 0 || exponent == kE8m0Nan) {
        return 0;
    }
    // Ordinary MX encoding multiplies logical values by 6 * 2^(127-e).
    // The source bytes encode 4 * V, so their multiplier is exactly
    // 1.5 * 2^(127-e).  The complete finite converter range is normal and
    // exact in FP16; duplicate it for one native half2 multiply.
    const uint32_t half_bits =
        ((142u - static_cast<uint32_t>(exponent)) << 10) | 0x200u;
    return half_bits | (half_bits << 16);
}

__device__ __forceinline__ uint32_t quantize_four_e4m3_pairs_to_mxfp4(
    uint16_t pair0,
    uint16_t pair1,
    uint16_t pair2,
    uint16_t pair3,
    uint32_t multiplier0,
    uint32_t multiplier1,
    uint32_t multiplier2,
    uint32_t multiplier3
) {
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b32 half0, half1, half2, half3;\n"
        "cvt.rn.f16x2.e4m3x2 half0, %1;\n"
        "cvt.rn.f16x2.e4m3x2 half1, %2;\n"
        "cvt.rn.f16x2.e4m3x2 half2, %3;\n"
        "cvt.rn.f16x2.e4m3x2 half3, %4;\n"
        "mul.rn.f16x2 half0, half0, %5;\n"
        "mul.rn.f16x2 half1, half1, %6;\n"
        "mul.rn.f16x2 half2, half2, %7;\n"
        "mul.rn.f16x2 half3, half3, %8;\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 byte0, half0;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 byte1, half1;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 byte2, half2;\n"
        "cvt.rn.satfinite.e2m1x2.f16x2 byte3, half3;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(packed)
        : "h"(pair0), "h"(pair1), "h"(pair2), "h"(pair3),
          "r"(multiplier0), "r"(multiplier1), "r"(multiplier2),
          "r"(multiplier3)
    );
    return packed;
}

__global__ __launch_bounds__(kThreadsPerBlock)
void convert_e4m3_x4_bhds_to_causal_mxfp4_kernel(
    const uint8_t *__restrict__ input,
    uint8_t *__restrict__ payload,
    uint8_t *__restrict__ scales,
    int heads,
    int sequence
) {
    const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    // One CTA owns a complete D64 tile for one (sequence tile, head, batch).
    // Its eight warps traverse eight depth rows apiece.  The 3-D launch needs
    // no integer divide or remainder and uses 16x fewer CTAs than assigning
    // one four-warp CTA to every four depth rows.
    const int sequence_tile = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int batch_index = static_cast<int>(blockIdx.z);
    const int sequence_tiles = sequence >> 7;

    #pragma unroll 1
    for (
        int depth = warp_in_block;
        depth < kHeadDepth;
        depth += kWarpsPerBlock
    ) {
        const size_t input_base =
            ((static_cast<size_t>(batch_index) * heads + head) *
                 kHeadDepth +
             depth) * sequence + sequence_tile * kSequenceTile;
        const uint32_t word = reinterpret_cast<const uint32_t *>(
            input + input_base
        )[lane];
        // All lanes participate before only even lanes retain one adjacent
        // E4M3x2 pair.  Shuffling the packed word once replaces four scalar
        // FP8 decodes plus four floating-point neighbor shuffles.
        const uint32_t next_word = __shfl_down_sync(
            0xffffffffu,
            word,
            1
        );

        const size_t payload_base =
            ((static_cast<size_t>(batch_index) * heads + head) *
                 kHeadDepth +
             depth) * (sequence >> 1) +
            sequence_tile * (kSequenceTile >> 1);
        const size_t scale_page =
            ((static_cast<size_t>(batch_index) * sequence_tiles +
              sequence_tile) * heads + head) * kScalePageBytes;
        const int scale_row = depth & 31;
        const int scale_group = depth >> 5;

        const uint32_t physical_amax_codes = warp_max_u8x4(
            word & kE4m3MagnitudeMask
        );
        uint32_t logical_scale_word = 0;
        if (lane == 0) {
            #pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                const uint8_t physical_amax_code = static_cast<uint8_t>(
                    physical_amax_codes >> (quarter * 8)
                );
                const uint8_t logical_e8m0 =
                    logical_e8m0_from_physical_amax(physical_amax_code);
                logical_scale_word |= static_cast<uint32_t>(logical_e8m0) <<
                    (quarter * 8);
            }
        }
        logical_scale_word = __shfl_sync(
            0xffffffffu,
            logical_scale_word,
            0
        );

        const uint8_t scale0 = static_cast<uint8_t>(logical_scale_word);
        const uint8_t scale1 = static_cast<uint8_t>(logical_scale_word >> 8);
        const uint8_t scale2 = static_cast<uint8_t>(logical_scale_word >> 16);
        const uint8_t scale3 = static_cast<uint8_t>(logical_scale_word >> 24);
        if ((lane & 1) == 0) {
            uint16_t pair0 = static_cast<uint16_t>(word & 0xffu) |
                static_cast<uint16_t>((next_word & 0xffu) << 8);
            uint16_t pair1 = static_cast<uint16_t>((word >> 8) & 0xffu) |
                static_cast<uint16_t>(next_word & 0xff00u);
            uint16_t pair2 = static_cast<uint16_t>((word >> 16) & 0xffu) |
                static_cast<uint16_t>((next_word >> 8) & 0xff00u);
            uint16_t pair3 = static_cast<uint16_t>((word >> 24) & 0xffu) |
                static_cast<uint16_t>((next_word >> 16) & 0xff00u);
            // Preserve the reference's explicit NaN-group policy: one NaN
            // amax publishes the sentinel scale and an all-zero payload,
            // rather than feeding NaN through a zero half2 multiplier.
            pair0 = scale0 == kE8m0Nan ? 0 : pair0;
            pair1 = scale1 == kE8m0Nan ? 0 : pair1;
            pair2 = scale2 == kE8m0Nan ? 0 : pair2;
            pair3 = scale3 == kE8m0Nan ? 0 : pair3;
            const uint32_t packed = quantize_four_e4m3_pairs_to_mxfp4(
                pair0,
                pair1,
                pair2,
                pair3,
                logical_e8m0_encode_multiplier_half2(scale0),
                logical_e8m0_encode_multiplier_half2(scale1),
                logical_e8m0_encode_multiplier_half2(scale2),
                logical_e8m0_encode_multiplier_half2(scale3)
            );
            const int output_byte = lane >> 1;
            payload[payload_base + output_byte] =
                static_cast<uint8_t>(packed);
            payload[payload_base + 16 + output_byte] =
                static_cast<uint8_t>(packed >> 8);
            payload[payload_base + 32 + output_byte] =
                static_cast<uint8_t>(packed >> 16);
            payload[payload_base + 48 + output_byte] =
                static_cast<uint8_t>(packed >> 24);
        }
        if (lane == 0) {
            // The four causal-quarter scale bytes are adjacent and the public
            // out contract guarantees a four-byte-aligned base.  Keep them in
            // quarter order with one aligned store.
            *reinterpret_cast<uint32_t *>(
                scales + scale_page + scale_row * 16 + scale_group * 4
            ) = logical_scale_word;
        }
    }
}

inline void launch(
    const uint8_t *input,
    uint8_t *payload,
    uint8_t *scales,
    int batch,
    int heads,
    int sequence,
    cudaStream_t stream
) {
    const dim3 grid(
        static_cast<unsigned int>(sequence / kSequenceTile),
        static_cast<unsigned int>(heads),
        static_cast<unsigned int>(batch)
    );
    convert_e4m3_x4_bhds_to_causal_mxfp4_kernel<<<
        grid,
        kThreadsPerBlock,
        0,
        stream
    >>>(
        input,
        payload,
        scales,
        heads,
        sequence
    );
}

}  // namespace tkfa4_e4m3_to_mxfp4_v

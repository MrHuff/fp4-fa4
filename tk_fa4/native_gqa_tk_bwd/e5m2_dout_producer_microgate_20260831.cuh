#pragma once

#include "../deprecated/fa4_common.cuh"

#include <cstdint>

// Standalone producer proof only.  It has no reference from the production
// projection epilogue, attention dispatch, or interface.
namespace tkfa4::native_gqa_tk_bwd::e5m2_dout_producer_microgate_20260831 {

constexpr int kDepth = 128;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * kittens::WARP_THREADS;
constexpr float kEncodeScale = 4.0f;
constexpr float kDecodeScale = 0.25f;
constexpr float kLogicalDstatScale = -16.0f;
constexpr float kPhysicalDstatScale =
    kLogicalDstatScale * kDecodeScale;

static_assert(kPhysicalDstatScale == -4.0f);

__device__ __forceinline__ uint16_t encode_e5m2_pair_x4(
    kittens::bf16_2 source
) {
    float2 values = __bfloat1622float2(source);
    values.x *= kEncodeScale;
    values.y *= kEncodeScale;
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

// One warp owns one complete D128 row.  Every lane publishes four genuine
// E5M2 bytes and immediately decodes those same bytes for the row statistic.
// The physical reduction
//
//   -4 * sum(O * raw_E5_decode)
//
// is exactly the production logical convention
//
//   -16 * sum(O * (raw_E5_decode * 0.25)).
__global__ __launch_bounds__(kThreads, 4)
void producer_kernel(
    const kittens::bf16 *__restrict__ dout_bf16,
    const kittens::bf16 *__restrict__ attention_output_bf16,
    kittens::fp8e5m2 *__restrict__ dout_e5m2,
    float *__restrict__ dstat,
    int rows
) {
    const int warp = kittens::warpid();
    const int lane = kittens::laneid();
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row >= rows) {
        return;
    }

    const int element = row * kDepth + 4 * lane;
    const auto *dout_pairs = reinterpret_cast<const kittens::bf16_2 *>(
        dout_bf16 + element
    );
    const auto *output_pairs = reinterpret_cast<const kittens::bf16_2 *>(
        attention_output_bf16 + element
    );

    const uint16_t encoded0 = encode_e5m2_pair_x4(dout_pairs[0]);
    const uint16_t encoded1 = encode_e5m2_pair_x4(dout_pairs[1]);
    const uint32_t encoded4 = static_cast<uint32_t>(encoded0) |
        (static_cast<uint32_t>(encoded1) << 16);
    *reinterpret_cast<uint32_t *>(dout_e5m2 + element) = encoded4;

    const float2 decoded0 = decode_e5m2_pair(encoded0);
    const float2 decoded1 = decode_e5m2_pair(encoded1);
    const float2 output0 = __bfloat1622float2(output_pairs[0]);
    const float2 output1 = __bfloat1622float2(output_pairs[1]);
    float partial = 0.0f;
    partial = fmaf(output0.x, decoded0.x, partial);
    partial = fmaf(output0.y, decoded0.y, partial);
    partial = fmaf(output1.x, decoded1.x, partial);
    partial = fmaf(output1.y, decoded1.y, partial);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) {
        dstat[row] = partial * kPhysicalDstatScale;
    }
}

inline void launch(
    at::Tensor &dout_bf16,
    at::Tensor &attention_output_bf16,
    at::Tensor &dout_e5m2,
    at::Tensor &dstat,
    cudaStream_t stream
) {
    const int rows = static_cast<int>(dout_bf16.size(0));
    const int blocks = (rows + kWarps - 1) / kWarps;
    producer_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const kittens::bf16 *>(dout_bf16.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(
            attention_output_bf16.data_ptr()
        ),
        reinterpret_cast<kittens::fp8e5m2 *>(dout_e5m2.data_ptr()),
        reinterpret_cast<float *>(dstat.data_ptr()),
        rows
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::e5m2_dout_producer_microgate_20260831

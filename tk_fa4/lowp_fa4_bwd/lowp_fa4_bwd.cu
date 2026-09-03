#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cuda/atomic>
#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>

#ifndef TK_FA4_BWD_PURE_MXFP4_DQ_N256
#define TK_FA4_BWD_PURE_MXFP4_DQ_N256 0
#endif

#include "b300_bwd_cute16_candidate.cuh"
#include "b300_common.cuh"
#include "../../ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_quantize.cuh"
#include "rmsnorm_nvfp4_quantize.cuh"
#include "inverse_rope_nvfp4_quantize.cuh"
#include "gqa_d128_hierarchical_qkv_nvfp4_quantize.cuh"
#include "projection_fp4_epilogue.cuh"
#include "e4m3_to_mxfp4_v.cuh"

namespace {

__global__ void wait_dq_owner_prefix_kernel(
    const uint32_t *owner_epochs,
    int head_begin,
    int head_end,
    int key_tiles,
    int owner_stride,
    uint32_t expected_epoch
) {
    const int selected_heads = head_end - head_begin;
    const int wait_count = selected_heads * key_tiles;
    for (int index = static_cast<int>(threadIdx.x); index < wait_count;
         index += static_cast<int>(blockDim.x)) {
        const int head = head_begin + index / key_tiles;
        const int key_tile = index - (index / key_tiles) * key_tiles;
        auto &word = *const_cast<uint32_t *>(
            owner_epochs + head * owner_stride + key_tile
        );
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(word);
        while (ready.load(cuda::memory_order_acquire) < expected_epoch) {
            __nanosleep(64);
        }
    }
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffffu, value, offset)
        );
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float e4m3_projection_clamp(float value) {
    // torch.clamp propagates NaNs.  fminf/fmaxf do not, so retain the
    // explicit branch for exact agreement with the functional PyTorch path.
    if (isnan(value)) {
        return value;
    }
    return fminf(448.0f, fmaxf(-448.0f, value));
}

__device__ __forceinline__ float e4m3_projection_divide(
    float numerator,
    float denominator
) {
    // The per-element normalization has a data-dependent denominator; pin it
    // to the IEEE division used by ATen's tensor/tensor path.
    float result;
    asm volatile(
        "div.rn.f32 %0, %1, %2;"
        : "=f"(result)
        : "f"(numerator), "f"(denominator)
    );
    return result;
}

__device__ __forceinline__ uint32_t e4m3_projection_pack4(
    const float4 &values,
    float decode
) {
    const float4 scaled = make_float4(
        e4m3_projection_clamp(e4m3_projection_divide(values.x, decode)),
        e4m3_projection_clamp(e4m3_projection_divide(values.y, decode)),
        e4m3_projection_clamp(e4m3_projection_divide(values.z, decode)),
        e4m3_projection_clamp(e4m3_projection_divide(values.w, decode))
    );
    return static_cast<uint32_t>(__nv_fp8x4_e4m3(scaled).__x);
}

// The Llama-1.2B projection shape has K=2048, so one 256-thread CTA can keep
// its complete row resident (eight BF16 values per thread) across the amax
// reduction.  That removes the temporary FP32 matrix and avoids rereading the
// input for quantization.  A generic two-pass kernel below retains the same
// API for other K values.
__global__ __launch_bounds__(256) void prepare_e4m3_rows_k2048_kernel(
    const kittens::bf16 *__restrict__ input,
    uint32_t *__restrict__ payload,
    float *__restrict__ decode,
    int rows
) {
    constexpr int kColumns = 2048;
    constexpr int kWarps = 8;
    __shared__ float warp_maxima[kWarps];
    __shared__ int warp_has_nan[kWarps];
    __shared__ float row_decode;

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int pair_offset = row * (kColumns / 2) + threadIdx.x * 4;
    const auto *input_pairs =
        reinterpret_cast<const kittens::bf16_2 *>(input) + pair_offset;
    const float2 pair0 = __bfloat1622float2(input_pairs[0]);
    const float2 pair1 = __bfloat1622float2(input_pairs[1]);
    const float2 pair2 = __bfloat1622float2(input_pairs[2]);
    const float2 pair3 = __bfloat1622float2(input_pairs[3]);
    const float4 values0 = make_float4(
        pair0.x, pair0.y, pair1.x, pair1.y
    );
    const float4 values1 = make_float4(
        pair2.x, pair2.y, pair3.x, pair3.y
    );
    const bool has_nan =
        isnan(values0.x) || isnan(values0.y) ||
        isnan(values0.z) || isnan(values0.w) ||
        isnan(values1.x) || isnan(values1.y) ||
        isnan(values1.z) || isnan(values1.w);
    float local_max = 0.0f;
    local_max = fmaxf(local_max, fabsf(values0.x));
    local_max = fmaxf(local_max, fabsf(values0.y));
    local_max = fmaxf(local_max, fabsf(values0.z));
    local_max = fmaxf(local_max, fabsf(values0.w));
    local_max = fmaxf(local_max, fabsf(values1.x));
    local_max = fmaxf(local_max, fabsf(values1.y));
    local_max = fmaxf(local_max, fabsf(values1.z));
    local_max = fmaxf(local_max, fabsf(values1.w));
    local_max = warp_reduce_max(local_max);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_has_any_nan = __any_sync(0xffffffffu, has_nan);
    if (lane == 0) {
        warp_maxima[warp] = local_max;
        warp_has_nan[warp] = warp_has_any_nan;
    }
    __syncthreads();
    if (warp == 0) {
        float block_max = lane < kWarps ? warp_maxima[lane] : 0.0f;
        block_max = warp_reduce_max(block_max);
        const bool block_has_nan = lane < kWarps && warp_has_nan[lane] != 0;
        const unsigned nan_mask = __ballot_sync(
            0xffffffffu,
            block_has_nan
        );
        if (lane == 0) {
            row_decode = nan_mask != 0
                ? nanf("")
                : fmaxf(block_max, 1.0e-12f) * (1.0f / 448.0f);
            decode[row] = row_decode;
        }
    }
    __syncthreads();

    const int output_word = row * (kColumns / 4) + threadIdx.x * 2;
    payload[output_word] = e4m3_projection_pack4(values0, row_decode);
    payload[output_word + 1] = e4m3_projection_pack4(values1, row_decode);
}

__global__ __launch_bounds__(256) void prepare_e4m3_rows_generic_kernel(
    const kittens::bf16 *__restrict__ input,
    uint32_t *__restrict__ payload,
    float *__restrict__ decode,
    int rows,
    int columns
) {
    constexpr int kWarps = 8;
    __shared__ float warp_maxima[kWarps];
    __shared__ int warp_has_nan[kWarps];
    __shared__ float row_decode;

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int groups = columns / 4;
    const auto *input_pairs = reinterpret_cast<const kittens::bf16_2 *>(
        input + static_cast<size_t>(row) * columns
    );
    float local_max = 0.0f;
    bool has_nan = false;
    for (int group = threadIdx.x; group < groups; group += blockDim.x) {
        const float2 first = __bfloat1622float2(input_pairs[group * 2]);
        const float2 second = __bfloat1622float2(input_pairs[group * 2 + 1]);
        has_nan = has_nan || isnan(first.x) || isnan(first.y) ||
            isnan(second.x) || isnan(second.y);
        local_max = fmaxf(local_max, fabsf(first.x));
        local_max = fmaxf(local_max, fabsf(first.y));
        local_max = fmaxf(local_max, fabsf(second.x));
        local_max = fmaxf(local_max, fabsf(second.y));
    }
    local_max = warp_reduce_max(local_max);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_has_any_nan = __any_sync(0xffffffffu, has_nan);
    if (lane == 0) {
        warp_maxima[warp] = local_max;
        warp_has_nan[warp] = warp_has_any_nan;
    }
    __syncthreads();
    if (warp == 0) {
        float block_max = lane < kWarps ? warp_maxima[lane] : 0.0f;
        block_max = warp_reduce_max(block_max);
        const bool block_has_nan = lane < kWarps && warp_has_nan[lane] != 0;
        const unsigned nan_mask = __ballot_sync(
            0xffffffffu,
            block_has_nan
        );
        if (lane == 0) {
            row_decode = nan_mask != 0
                ? nanf("")
                : fmaxf(block_max, 1.0e-12f) * (1.0f / 448.0f);
            decode[row] = row_decode;
        }
    }
    __syncthreads();

    uint32_t *row_payload = payload + static_cast<size_t>(row) * groups;
    for (int group = threadIdx.x; group < groups; group += blockDim.x) {
        const float2 first = __bfloat1622float2(input_pairs[group * 2]);
        const float2 second = __bfloat1622float2(input_pairs[group * 2 + 1]);
        row_payload[group] = e4m3_projection_pack4(
            make_float4(first.x, first.y, second.x, second.y),
            row_decode
        );
    }
}

__global__ void inverse_rope_interleaved_qkv_grad_kernel(
    kittens::bf16 *qkv_grad,
    const kittens::bf16 *rope_cos,
    const kittens::bf16 *rope_sin,
    int64_t row_heads,
    int heads
) {
    constexpr int kQkDepth = 192;
    constexpr int kVDepth = 128;
    constexpr int kRotaryPairs = kQkDepth / 2;
    constexpr int kHeadWidth = kQkDepth * 2 + kVDepth;
    const int64_t total_pairs = row_heads * kRotaryPairs;
    for (
        int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        linear < total_pairs;
        linear += static_cast<int64_t>(blockDim.x) * gridDim.x
    ) {
        const int pair = static_cast<int>(linear % kRotaryPairs);
        const int64_t row_head = linear / kRotaryPairs;
        const int64_t row = row_head / heads;
        const size_t rope_offset =
            static_cast<size_t>(row) * kRotaryPairs + pair;
        const float cosine = __bfloat162float(rope_cos[rope_offset]);
        const float sine = __bfloat162float(rope_sin[rope_offset]);
        kittens::bf16 *head_base =
            qkv_grad + row_head * kHeadWidth;
        kittens::bf16_2 *q_pair =
            reinterpret_cast<kittens::bf16_2 *>(head_base) + pair;
        kittens::bf16_2 *k_pair =
            reinterpret_cast<kittens::bf16_2 *>(head_base + kQkDepth) + pair;
        const float2 q_values = __bfloat1622float2(*q_pair);
        const float2 k_values = __bfloat1622float2(*k_pair);
        *q_pair = __floats2bfloat162_rn(
            fmaf(q_values.y, sine, q_values.x * cosine),
            fmaf(-q_values.x, sine, q_values.y * cosine)
        );
        *k_pair = __floats2bfloat162_rn(
            fmaf(k_values.y, sine, k_values.x * cosine),
            fmaf(-k_values.x, sine, k_values.y * cosine)
        );
    }
}

__global__ void stitch_gqa_d64_inverse_rope_grad_kernel(
    const kittens::bf16 *dq,
    const kittens::bf16 *dk,
    const kittens::bf16 *dv,
    const kittens::bf16 *rope_cos,
    const kittens::bf16 *rope_sin,
    kittens::bf16 *combined,
    int64_t rows,
    int q_heads,
    int kv_heads,
    float q_gradient_scale,
    float k_gradient_scale,
    float v_gradient_scale
) {
    constexpr int kDepth = 64;
    constexpr int kPairs = kDepth / 2;
    const int q_row_pairs = q_heads * kPairs;
    const int kv_row_pairs = kv_heads * kPairs;
    const int output_row_pairs = q_row_pairs + 2 * kv_row_pairs;
    const int64_t total_pairs = rows * output_row_pairs;
    for (
        int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        linear < total_pairs;
        linear += static_cast<int64_t>(blockDim.x) * gridDim.x
    ) {
        const int64_t row = linear / output_row_pairs;
        const int pair_column = static_cast<int>(
            linear - row * output_row_pairs
        );
        kittens::bf16_2 result;
        if (pair_column < q_row_pairs) {
            const int pair = pair_column % kPairs;
            const float cosine = __bfloat162float(
                rope_cos[row * kPairs + pair]
            );
            const float sine = __bfloat162float(
                rope_sin[row * kPairs + pair]
            );
            const float2 values = __bfloat1622float2(
                reinterpret_cast<const kittens::bf16_2 *>(dq)[
                    row * q_row_pairs + pair_column
                ]
            );
            result = __floats2bfloat162_rn(
                fmaf(values.y, sine, values.x * cosine) * q_gradient_scale,
                fmaf(-values.x, sine, values.y * cosine) * q_gradient_scale
            );
        } else if (pair_column < q_row_pairs + kv_row_pairs) {
            const int kv_pair_column = pair_column - q_row_pairs;
            const int pair = kv_pair_column % kPairs;
            const float cosine = __bfloat162float(
                rope_cos[row * kPairs + pair]
            );
            const float sine = __bfloat162float(
                rope_sin[row * kPairs + pair]
            );
            const float2 values = __bfloat1622float2(
                reinterpret_cast<const kittens::bf16_2 *>(dk)[
                    row * kv_row_pairs + kv_pair_column
                ]
            );
            result = __floats2bfloat162_rn(
                fmaf(values.y, sine, values.x * cosine) * k_gradient_scale,
                fmaf(-values.x, sine, values.y * cosine) * k_gradient_scale
            );
        } else {
            const int kv_pair_column =
                pair_column - q_row_pairs - kv_row_pairs;
            const float2 values = __bfloat1622float2(
                reinterpret_cast<const kittens::bf16_2 *>(dv)[
                    row * kv_row_pairs + kv_pair_column
                ]
            );
            result = __floats2bfloat162_rn(
                values.x * v_gradient_scale,
                values.y * v_gradient_scale
            );
        }
        reinterpret_cast<kittens::bf16_2 *>(combined)[linear] = result;
    }
}

__global__ void stitch_gqa_d128_inverse_rope_grad_kernel(
    const kittens::bf16 *dq,
    const kittens::bf16 *dk,
    const kittens::bf16 *dv,
    const kittens::bf16 *rope_cos,
    const kittens::bf16 *rope_sin,
    kittens::bf16 *combined,
    int64_t rows,
    int q_heads,
    int kv_heads,
    float q_gradient_scale,
    float k_gradient_scale,
    float v_gradient_scale
) {
    constexpr int kDepth = 128;
    constexpr int kPairs = kDepth / 2;
    const int q_row_pairs = q_heads * kPairs;
    const int kv_row_pairs = kv_heads * kPairs;
    const int output_row_pairs = q_row_pairs + 2 * kv_row_pairs;
    const int64_t total_pairs = rows * output_row_pairs;
    for (
        int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        linear < total_pairs;
        linear += static_cast<int64_t>(blockDim.x) * gridDim.x
    ) {
        const int64_t row = linear / output_row_pairs;
        const int pair_column = static_cast<int>(
            linear - row * output_row_pairs
        );
        kittens::bf16_2 result;
        if (pair_column < q_row_pairs) {
            const int pair = pair_column % kPairs;
            const float cosine = __bfloat162float(
                rope_cos[row * kPairs + pair]
            );
            const float sine = __bfloat162float(
                rope_sin[row * kPairs + pair]
            );
            const float2 values = __bfloat1622float2(
                reinterpret_cast<const kittens::bf16_2 *>(dq)[
                    row * q_row_pairs + pair_column
                ]
            );
            const kittens::bf16_2 inverse = __floats2bfloat162_rn(
                __fadd_rn(
                    __fmul_rn(values.x, cosine),
                    __fmul_rn(values.y, sine)
                ),
                __fadd_rn(
                    __fmul_rn(-values.x, sine),
                    __fmul_rn(values.y, cosine)
                )
            );
            const float2 rounded = __bfloat1622float2(inverse);
            result = __floats2bfloat162_rn(
                rounded.x * q_gradient_scale,
                rounded.y * q_gradient_scale
            );
        } else if (pair_column < q_row_pairs + kv_row_pairs) {
            const int kv_pair_column = pair_column - q_row_pairs;
            const int pair = kv_pair_column % kPairs;
            const float cosine = __bfloat162float(
                rope_cos[row * kPairs + pair]
            );
            const float sine = __bfloat162float(
                rope_sin[row * kPairs + pair]
            );
            const float2 values = __bfloat1622float2(
                reinterpret_cast<const kittens::bf16_2 *>(dk)[
                    row * kv_row_pairs + kv_pair_column
                ]
            );
            const kittens::bf16_2 inverse = __floats2bfloat162_rn(
                __fadd_rn(
                    __fmul_rn(values.x, cosine),
                    __fmul_rn(values.y, sine)
                ),
                __fadd_rn(
                    __fmul_rn(-values.x, sine),
                    __fmul_rn(values.y, cosine)
                )
            );
            const float2 rounded = __bfloat1622float2(inverse);
            result = __floats2bfloat162_rn(
                rounded.x * k_gradient_scale,
                rounded.y * k_gradient_scale
            );
        } else {
            const int kv_pair_column =
                pair_column - q_row_pairs - kv_row_pairs;
            const float2 values = __bfloat1622float2(
                reinterpret_cast<const kittens::bf16_2 *>(dv)[
                    row * kv_row_pairs + kv_pair_column
                ]
            );
            result = __floats2bfloat162_rn(
                values.x * v_gradient_scale,
                values.y * v_gradient_scale
            );
        }
        reinterpret_cast<kittens::bf16_2 *>(combined)[linear] = result;
    }
}

// Exact Llama-8B D128 layout: one warp owns one flattened [B,S] row.  This
// removes the generic kernel's runtime 64-bit row division and reuses each
// RoPE pair across all 32 Q and 8 K heads while retaining coalesced pair
// loads/stores.  Arithmetic intentionally matches the authenticated generic
// path, including both BF16 rounding boundaries.
__global__ __launch_bounds__(256) void
stitch_gqa_d128_h32_kv8_inverse_rope_grad_kernel(
    const kittens::bf16 *dq,
    const kittens::bf16 *dk,
    const kittens::bf16 *dv,
    const kittens::bf16 *rope_cos,
    const kittens::bf16 *rope_sin,
    kittens::bf16 *combined,
    int64_t rows,
    float q_gradient_scale,
    float k_gradient_scale,
    float v_gradient_scale
) {
    constexpr int kPairs = 64;
    constexpr int kQHeads = 32;
    constexpr int kKvHeads = 8;
    constexpr int kQRowPairs = kQHeads * kPairs;
    constexpr int kKvRowPairs = kKvHeads * kPairs;
    constexpr int kOutputRowPairs = kQRowPairs + 2 * kKvRowPairs;
    constexpr int kWarpsPerBlock = 8;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) {
        return;
    }

    const auto *dq_pairs = reinterpret_cast<const kittens::bf16_2 *>(dq) +
        row * kQRowPairs;
    const auto *dk_pairs = reinterpret_cast<const kittens::bf16_2 *>(dk) +
        row * kKvRowPairs;
    const auto *dv_pairs = reinterpret_cast<const kittens::bf16_2 *>(dv) +
        row * kKvRowPairs;
    auto *output_pairs = reinterpret_cast<kittens::bf16_2 *>(combined) +
        row * kOutputRowPairs;

    #pragma unroll
    for (int pair_half = 0; pair_half < 2; ++pair_half) {
        const int pair = lane + pair_half * 32;
        const float cosine = __bfloat162float(
            rope_cos[row * kPairs + pair]
        );
        const float sine = __bfloat162float(
            rope_sin[row * kPairs + pair]
        );

        #pragma unroll 1
        for (int head = 0; head < kQHeads; ++head) {
            const int pair_column = head * kPairs + pair;
            const float2 values = __bfloat1622float2(
                dq_pairs[pair_column]
            );
            const kittens::bf16_2 inverse = __floats2bfloat162_rn(
                __fadd_rn(
                    __fmul_rn(values.x, cosine),
                    __fmul_rn(values.y, sine)
                ),
                __fadd_rn(
                    __fmul_rn(-values.x, sine),
                    __fmul_rn(values.y, cosine)
                )
            );
            const float2 rounded = __bfloat1622float2(inverse);
            output_pairs[pair_column] = __floats2bfloat162_rn(
                rounded.x * q_gradient_scale,
                rounded.y * q_gradient_scale
            );
        }

        #pragma unroll 1
        for (int head = 0; head < kKvHeads; ++head) {
            const int pair_column = head * kPairs + pair;
            const float2 values = __bfloat1622float2(
                dk_pairs[pair_column]
            );
            const kittens::bf16_2 inverse = __floats2bfloat162_rn(
                __fadd_rn(
                    __fmul_rn(values.x, cosine),
                    __fmul_rn(values.y, sine)
                ),
                __fadd_rn(
                    __fmul_rn(-values.x, sine),
                    __fmul_rn(values.y, cosine)
                )
            );
            const float2 rounded = __bfloat1622float2(inverse);
            output_pairs[kQRowPairs + pair_column] =
                __floats2bfloat162_rn(
                    rounded.x * k_gradient_scale,
                    rounded.y * k_gradient_scale
                );
        }

        #pragma unroll 1
        for (int head = 0; head < kKvHeads; ++head) {
            const int pair_column = head * kPairs + pair;
            const float2 values = __bfloat1622float2(
                dv_pairs[pair_column]
            );
            output_pairs[kQRowPairs + kKvRowPairs + pair_column] =
                __floats2bfloat162_rn(
                    values.x * v_gradient_scale,
                    values.y * v_gradient_scale
                );
        }
    }
}

__global__ void rope_pair_qk_inplace_kernel(
    kittens::bf16 *q,
    kittens::bf16 *k,
    const kittens::bf16 *rope_cos,
    const kittens::bf16 *rope_sin,
    int64_t row_heads,
    int heads
) {
    constexpr int kQkDepth = 192;
    constexpr int kRotaryPairs = kQkDepth / 2;
    const int64_t total_pairs = row_heads * kRotaryPairs;
    for (
        int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        linear < total_pairs;
        linear += static_cast<int64_t>(blockDim.x) * gridDim.x
    ) {
        const int pair = static_cast<int>(linear % kRotaryPairs);
        const int64_t row_head = linear / kRotaryPairs;
        const int64_t row = row_head / heads;
        const size_t rope_offset =
            static_cast<size_t>(row) * kRotaryPairs + pair;
        const float cosine = __bfloat162float(rope_cos[rope_offset]);
        const float sine = __bfloat162float(rope_sin[rope_offset]);
        kittens::bf16_2 *q_pair =
            reinterpret_cast<kittens::bf16_2 *>(q) + linear;
        kittens::bf16_2 *k_pair =
            reinterpret_cast<kittens::bf16_2 *>(k) + linear;
        const float2 q_values = __bfloat1622float2(*q_pair);
        const float2 k_values = __bfloat1622float2(*k_pair);
        *q_pair = __floats2bfloat162_rn(
            fmaf(-q_values.y, sine, q_values.x * cosine),
            fmaf(q_values.x, sine, q_values.y * cosine)
        );
        *k_pair = __floats2bfloat162_rn(
            fmaf(-k_values.y, sine, k_values.x * cosine),
            fmaf(k_values.x, sine, k_values.y * cosine)
        );
    }
}

std::vector<at::Tensor> rope_pair_qk_inplace(
    at::Tensor q,
    at::Tensor k,
    at::Tensor rope_cos,
    at::Tensor rope_sin
) {
    constexpr int kQkDepth = 192;
    constexpr int kRotaryPairs = kQkDepth / 2;
    TORCH_CHECK(
        q.scalar_type() == at::ScalarType::BFloat16 &&
            k.scalar_type() == at::ScalarType::BFloat16 &&
            q.is_cuda() && k.is_cuda() &&
            q.is_contiguous() && k.is_contiguous() &&
            q.dim() == 4 && k.sizes() == q.sizes() &&
            q.size(3) == kQkDepth,
        "pair-native Q/K must be matching contiguous CUDA BF16 [B,S,H,192]"
    );
    TORCH_CHECK(
        rope_cos.scalar_type() == at::ScalarType::BFloat16 &&
            rope_sin.scalar_type() == at::ScalarType::BFloat16 &&
            rope_cos.is_cuda() && rope_sin.is_cuda() &&
            rope_cos.is_contiguous() && rope_sin.is_contiguous() &&
            rope_cos.dim() == 3 && rope_sin.dim() == 3 &&
            rope_cos.size(0) == q.size(0) &&
            rope_cos.size(1) == q.size(1) &&
            rope_cos.size(2) == kRotaryPairs &&
            rope_sin.sizes() == rope_cos.sizes(),
        "pair-native RoPE tables must be contiguous CUDA BF16 [B,S,96]"
    );
    kittens::py::device_check(q, k, rope_cos, rope_sin);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const int64_t row_heads = q.size(0) * q.size(1) * q.size(2);
    const int64_t total_pairs = row_heads * kRotaryPairs;
    constexpr int kThreads = 256;
    const int64_t requested_blocks =
        (total_pairs + kThreads - 1) / kThreads;
    const int blocks = static_cast<int>(
        std::min<int64_t>(requested_blocks, static_cast<int64_t>(65535))
    );
    rope_pair_qk_inplace_kernel<<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<kittens::bf16 *>(q.data_ptr()),
        reinterpret_cast<kittens::bf16 *>(k.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(rope_cos.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(rope_sin.data_ptr()),
        row_heads,
        static_cast<int>(q.size(2))
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return {q, k};
}

at::Tensor inverse_rope_interleaved_qkv_grad_inplace(
    at::Tensor qkv_grad,
    at::Tensor rope_cos,
    at::Tensor rope_sin
) {
    constexpr int kQkDepth = 192;
    constexpr int kVDepth = 128;
    constexpr int kRotaryPairs = kQkDepth / 2;
    constexpr int kHeadWidth = kQkDepth * 2 + kVDepth;
    TORCH_CHECK(
        qkv_grad.scalar_type() == at::ScalarType::BFloat16 &&
            qkv_grad.is_cuda() && qkv_grad.is_contiguous() &&
            qkv_grad.dim() == 4 && qkv_grad.size(3) == kHeadWidth,
        "pair-native QKV gradient must be contiguous CUDA BF16 [B,S,H,512]"
    );
    TORCH_CHECK(
        rope_cos.scalar_type() == at::ScalarType::BFloat16 &&
            rope_sin.scalar_type() == at::ScalarType::BFloat16 &&
            rope_cos.is_cuda() && rope_sin.is_cuda() &&
            rope_cos.is_contiguous() && rope_sin.is_contiguous() &&
            rope_cos.dim() == 3 && rope_sin.dim() == 3 &&
            rope_cos.size(0) == qkv_grad.size(0) &&
            rope_cos.size(1) == qkv_grad.size(1) &&
            rope_cos.size(2) == kRotaryPairs &&
            rope_sin.sizes() == rope_cos.sizes(),
        "pair-native RoPE tables must be contiguous CUDA BF16 [B,S,96]"
    );
    kittens::py::device_check(qkv_grad, rope_cos, rope_sin);
    const c10::cuda::CUDAGuard device_guard(qkv_grad.device());
    const int64_t row_heads =
        qkv_grad.size(0) * qkv_grad.size(1) * qkv_grad.size(2);
    const int64_t total_pairs = row_heads * kRotaryPairs;
    constexpr int kThreads = 256;
    const int64_t requested_blocks =
        (total_pairs + kThreads - 1) / kThreads;
    const int blocks = static_cast<int>(
        std::min<int64_t>(requested_blocks, static_cast<int64_t>(65535))
    );
    inverse_rope_interleaved_qkv_grad_kernel<<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<kittens::bf16 *>(qkv_grad.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(rope_cos.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(rope_sin.data_ptr()),
        row_heads,
        static_cast<int>(qkv_grad.size(2))
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return qkv_grad;
}

at::Tensor stitch_gqa_d64_inverse_rope_grad(
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    double q_gradient_scale,
    double k_gradient_scale,
    double v_gradient_scale
) {
    constexpr int kDepth = 64;
    constexpr int kPairs = kDepth / 2;
    TORCH_CHECK(
        dq.scalar_type() == at::ScalarType::BFloat16 &&
            dk.scalar_type() == at::ScalarType::BFloat16 &&
            dv.scalar_type() == at::ScalarType::BFloat16 &&
            dq.is_cuda() && dk.is_cuda() && dv.is_cuda() &&
            dq.is_contiguous() && dk.is_contiguous() && dv.is_contiguous() &&
            dq.dim() == 4 && dk.dim() == 4 && dv.dim() == 4 &&
            dq.size(0) > 0 && dq.size(1) > 0 &&
            dq.size(2) > 0 && dk.size(2) > 0 &&
            dq.size(3) == kDepth && dk.size(3) == kDepth &&
            dv.size(3) == kDepth &&
            dq.size(0) == dk.size(0) && dq.size(0) == dv.size(0) &&
            dq.size(1) == dk.size(1) && dq.size(1) == dv.size(1) &&
            dk.sizes() == dv.sizes() && dq.size(2) % dk.size(2) == 0,
        "D64 GQA gradients must be contiguous CUDA BF16 tensors with "
        "matching [B,S], matching dK/dV, and Hq divisible by Hkv"
    );
    TORCH_CHECK(
        rope_cos.scalar_type() == at::ScalarType::BFloat16 &&
            rope_sin.scalar_type() == at::ScalarType::BFloat16 &&
            rope_cos.is_cuda() && rope_sin.is_cuda() &&
            rope_cos.is_contiguous() && rope_sin.is_contiguous() &&
            rope_cos.dim() == 3 && rope_sin.dim() == 3 &&
            rope_cos.size(0) == dq.size(0) &&
            rope_cos.size(1) == dq.size(1) &&
            rope_cos.size(2) == kPairs &&
            rope_sin.sizes() == rope_cos.sizes(),
        "D64 pair-native RoPE tables must be contiguous CUDA BF16 [B,S,32]"
    );
    TORCH_CHECK(
        std::isfinite(q_gradient_scale) && q_gradient_scale > 0.0 &&
            std::isfinite(k_gradient_scale) && k_gradient_scale > 0.0 &&
            std::isfinite(v_gradient_scale) && v_gradient_scale > 0.0,
        "Q/K/V gradient decode scales must be finite and positive"
    );
    kittens::py::device_check(dq, dk, dv, rope_cos, rope_sin);
    const c10::cuda::CUDAGuard device_guard(dq.device());
    const int64_t rows = dq.size(0) * dq.size(1);
    const int q_heads = static_cast<int>(dq.size(2));
    const int kv_heads = static_cast<int>(dk.size(2));
    const int64_t output_width =
        static_cast<int64_t>(q_heads + 2 * kv_heads) * kDepth;
    at::Tensor combined = at::empty({rows, output_width}, dq.options());
    const int64_t total_pairs = rows * output_width / 2;
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total_pairs + kThreads - 1) / kThreads,
        static_cast<int64_t>(65535)
    ));
    stitch_gqa_d64_inverse_rope_grad_kernel<<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const kittens::bf16 *>(dq.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(dk.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(dv.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(rope_cos.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(rope_sin.data_ptr()),
        reinterpret_cast<kittens::bf16 *>(combined.data_ptr()),
        rows,
        q_heads,
        kv_heads,
        static_cast<float>(q_gradient_scale),
        static_cast<float>(k_gradient_scale),
        static_cast<float>(v_gradient_scale)
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return combined;
}

at::Tensor stitch_gqa_d128_inverse_rope_grad(
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    double q_gradient_scale,
    double k_gradient_scale,
    double v_gradient_scale
) {
    constexpr int kDepth = 128;
    constexpr int kPairs = kDepth / 2;
    TORCH_CHECK(
        dq.scalar_type() == at::ScalarType::BFloat16 &&
            dk.scalar_type() == at::ScalarType::BFloat16 &&
            dv.scalar_type() == at::ScalarType::BFloat16 &&
            dq.is_cuda() && dk.is_cuda() && dv.is_cuda() &&
            dq.is_contiguous() && dk.is_contiguous() && dv.is_contiguous() &&
            dq.dim() == 4 && dk.dim() == 4 && dv.dim() == 4 &&
            dq.size(0) > 0 && dq.size(1) > 0 &&
            dq.size(2) > 0 && dk.size(2) > 0 &&
            dq.size(3) == kDepth && dk.size(3) == kDepth &&
            dv.size(3) == kDepth &&
            dq.size(0) == dk.size(0) && dq.size(0) == dv.size(0) &&
            dq.size(1) == dk.size(1) && dq.size(1) == dv.size(1) &&
            dk.sizes() == dv.sizes() && dq.size(2) % dk.size(2) == 0,
        "D128 GQA gradients must be contiguous CUDA BF16 tensors with "
        "matching [B,S], matching dK/dV, and Hq divisible by Hkv"
    );
    TORCH_CHECK(
        rope_cos.scalar_type() == at::ScalarType::BFloat16 &&
            rope_sin.scalar_type() == at::ScalarType::BFloat16 &&
            rope_cos.is_cuda() && rope_sin.is_cuda() &&
            rope_cos.is_contiguous() && rope_sin.is_contiguous() &&
            rope_cos.dim() == 3 && rope_sin.dim() == 3 &&
            rope_cos.size(0) == dq.size(0) &&
            rope_cos.size(1) == dq.size(1) &&
            rope_cos.size(2) == kPairs &&
            rope_sin.sizes() == rope_cos.sizes(),
        "D128 pair-native RoPE tables must be contiguous CUDA BF16 [B,S,64]"
    );
    TORCH_CHECK(
        std::isfinite(q_gradient_scale) && q_gradient_scale > 0.0 &&
            std::isfinite(k_gradient_scale) && k_gradient_scale > 0.0 &&
            std::isfinite(v_gradient_scale) && v_gradient_scale > 0.0,
        "Q/K/V gradient decode scales must be finite and positive"
    );
    kittens::py::device_check(dq, dk, dv, rope_cos, rope_sin);
    const c10::cuda::CUDAGuard device_guard(dq.device());
    const int64_t rows = dq.size(0) * dq.size(1);
    const int q_heads = static_cast<int>(dq.size(2));
    const int kv_heads = static_cast<int>(dk.size(2));
    const int64_t output_width =
        static_cast<int64_t>(q_heads + 2 * kv_heads) * kDepth;
    at::Tensor combined = at::empty({rows, output_width}, dq.options());
    constexpr int kThreads = 256;
    const bool exact_h32_kv8 =
        dq.size(0) <= 2 && dq.size(1) == 4096 &&
        q_heads == 32 && kv_heads == 8;
    if (exact_h32_kv8) {
        constexpr int kWarpsPerBlock = kThreads / 32;
        const int blocks = static_cast<int>(
            (rows + kWarpsPerBlock - 1) / kWarpsPerBlock
        );
        stitch_gqa_d128_h32_kv8_inverse_rope_grad_kernel<<<
            blocks,
            kThreads,
            0,
            at::cuda::getCurrentCUDAStream()
        >>>(
            reinterpret_cast<const kittens::bf16 *>(dq.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(dk.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(dv.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(rope_cos.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(rope_sin.data_ptr()),
            reinterpret_cast<kittens::bf16 *>(combined.data_ptr()),
            rows,
            static_cast<float>(q_gradient_scale),
            static_cast<float>(k_gradient_scale),
            static_cast<float>(v_gradient_scale)
        );
    } else {
        const int64_t total_pairs = rows * output_width / 2;
        const int blocks = static_cast<int>(std::min<int64_t>(
            (total_pairs + kThreads - 1) / kThreads,
            static_cast<int64_t>(65535)
        ));
        stitch_gqa_d128_inverse_rope_grad_kernel<<<
            blocks,
            kThreads,
            0,
            at::cuda::getCurrentCUDAStream()
        >>>(
            reinterpret_cast<const kittens::bf16 *>(dq.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(dk.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(dv.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(rope_cos.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(rope_sin.data_ptr()),
            reinterpret_cast<kittens::bf16 *>(combined.data_ptr()),
            rows,
            q_heads,
            kv_heads,
            static_cast<float>(q_gradient_scale),
            static_cast<float>(k_gradient_scale),
            static_cast<float>(v_gradient_scale)
        );
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    return combined;
}

__global__ void reduce_adaptive_fp4_qk_scales_kernel(
    const kittens::bf16 *q,
    const kittens::bf16 *k,
    float *scale_state,
    int seq_len,
    int heads,
    float max_quant_scale,
    float min_quant_scale,
    float min_headroom,
    float rms_clip_multiple,
    float softmax_scale,
    float ds_quant_scale
) {
    constexpr int kDepth = 192;
    constexpr int kPackedDepth = kDepth / 2;
    constexpr int kScaleRecordWords = 7;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const size_t pairs_per_head =
        static_cast<size_t>(seq_len) * kPackedDepth;
    float q_amax = 0.0f;
    float k_amax = 0.0f;
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (
        size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        pair < pairs_per_head;
        pair += static_cast<size_t>(blockDim.x) * gridDim.x
    ) {
        const size_t seq_idx = pair / kPackedDepth;
        const size_t packed_depth = pair - seq_idx * kPackedDepth;
        const size_t input_pair =
            ((static_cast<size_t>(batch_idx) * seq_len + seq_idx) * heads +
             head_idx) * kPackedDepth + packed_depth;
        const kittens::bf16_2 q_pair =
            reinterpret_cast<const kittens::bf16_2 *>(q)[input_pair];
        const kittens::bf16_2 k_pair =
            reinterpret_cast<const kittens::bf16_2 *>(k)[input_pair];
        const float2 q_values = __bfloat1622float2(q_pair);
        const float2 k_values = __bfloat1622float2(k_pair);
        q_amax = fmaxf(
            q_amax,
            fmaxf(fabsf(q_values.x), fabsf(q_values.y))
        );
        k_amax = fmaxf(
            k_amax,
            fmaxf(fabsf(k_values.x), fabsf(k_values.y))
        );
        q_sum_sq = fmaf(q_values.x, q_values.x, q_sum_sq);
        q_sum_sq = fmaf(q_values.y, q_values.y, q_sum_sq);
        k_sum_sq = fmaf(k_values.x, k_values.x, k_sum_sq);
        k_sum_sq = fmaf(k_values.y, k_values.y, k_sum_sq);
    }

    q_amax = warp_reduce_max(q_amax);
    k_amax = warp_reduce_max(k_amax);
    q_sum_sq = warp_reduce_sum(q_sum_sq);
    k_sum_sq = warp_reduce_sum(k_sum_sq);
    __shared__ float warp_q_amax[8];
    __shared__ float warp_k_amax[8];
    __shared__ float warp_q_sum_sq[8];
    __shared__ float warp_k_sum_sq[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        warp_q_amax[warp] = q_amax;
        warp_k_amax[warp] = k_amax;
        warp_q_sum_sq[warp] = q_sum_sq;
        warp_k_sum_sq[warp] = k_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = static_cast<int>(blockDim.x) >> 5;
        q_amax = lane < warp_count ? warp_q_amax[lane] : 0.0f;
        k_amax = lane < warp_count ? warp_k_amax[lane] : 0.0f;
        q_sum_sq = lane < warp_count ? warp_q_sum_sq[lane] : 0.0f;
        k_sum_sq = lane < warp_count ? warp_k_sum_sq[lane] : 0.0f;
        q_amax = warp_reduce_max(q_amax);
        k_amax = warp_reduce_max(k_amax);
        q_sum_sq = warp_reduce_sum(q_sum_sq);
        k_sum_sq = warp_reduce_sum(k_sum_sq);
        if (lane == 0) {
            float *record = scale_state +
                (static_cast<size_t>(batch_idx) * heads + head_idx) *
                    kScaleRecordWords;
            // All values are non-negative, so IEEE float bit order matches
            // unsigned integer order and atomicMax is exact here.
            atomicMax(
                reinterpret_cast<unsigned int *>(&record[0]),
                __float_as_uint(q_amax)
            );
            atomicMax(
                reinterpret_cast<unsigned int *>(&record[1]),
                __float_as_uint(k_amax)
            );
            atomicAdd(&record[2], q_sum_sq);
            atomicAdd(&record[3], k_sum_sq);
            __threadfence();
            auto *completion = reinterpret_cast<unsigned int *>(
                &record[5]
            );
            const unsigned int ticket = atomicInc(
                completion,
                static_cast<unsigned int>(gridDim.x - 1)
            );
            if (ticket == static_cast<unsigned int>(gridDim.x - 1)) {
                const float reduced_q_amax = record[0];
                const float reduced_k_amax = record[1];
                const float combined_amax = fmaxf(
                    reduced_q_amax,
                    reduced_k_amax
                );
                const float inverse_element_count =
                    1.0f / static_cast<float>(seq_len * kDepth);
                const float combined_rms = fmaxf(
                    sqrtf(record[2] * inverse_element_count),
                    sqrtf(record[3] * inverse_element_count)
                );
                // Protect the bulk of a Gaussian-like distribution while
                // clipping a sparse heavy tail.  min_headroom prevents one
                // pathological value from being ignored completely; the RMS
                // term supplies the normal-distribution clipping floor.
                const float clipping_amax = fmaxf(
                    combined_amax * min_headroom,
                    combined_rms * rms_clip_multiple
                );
                const float target_multiplier = fmaxf(
                    min_quant_scale,
                    clipping_amax > 0.0f
                        ? fminf(
                              max_quant_scale,
                              6.0f / clipping_amax
                          )
                        : max_quant_scale
                );
                // Use one common Q/K scale so the score MMA can apply the
                // adaptive dequantization through its existing single NVFP4
                // scale page.  Snap the dequantizer to E4M3 first, then use
                // its exact reciprocal for packing; tensor-core scaling and
                // scalar dQ/dK correction therefore agree bit-for-bit.
                const kittens::fp8e4m3 encoded_dequant_scale =
                    kittens::base_types::convertor<
                        kittens::fp8e4m3,
                        float
                    >::convert(1.0f / target_multiplier);
                const float dequant_scale =
                    kittens::base_types::convertor<
                        float,
                        kittens::fp8e4m3
                    >::convert(encoded_dequant_scale);
                const float common_multiplier = 1.0f / dequant_scale;
                record[0] = common_multiplier;
                record[1] = common_multiplier;
                record[2] =
                    softmax_scale / (ds_quant_scale * common_multiplier);
                record[3] =
                    softmax_scale / (ds_quant_scale * common_multiplier);
                record[4] =
                    (softmax_scale * 0x1.715476p+0f) /
                    (common_multiplier * common_multiplier);
                const uint8_t scale_byte = std::bit_cast<uint8_t>(
                    encoded_dequant_scale
                );
                reinterpret_cast<uint32_t *>(record)[6] =
                    static_cast<uint32_t>(scale_byte) * 0x01010101u;
                __threadfence();
            }
        }
    }
}

__global__ void quantize_fp4_bshd_to_bhds_unpacked_kernel(
    const kittens::bf16 *input,
    uint8_t *output,
    int batch,
    int seq_len,
    int heads,
    int depth,
    float quant_scale
) {
    const int packed_seq_len = seq_len / 2;
    const size_t total = static_cast<size_t>(batch) * heads * depth *
        packed_seq_len;
    for (
        size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        linear < total;
        linear += static_cast<size_t>(blockDim.x) * gridDim.x
    ) {
        size_t remaining = linear;
        const int packed_seq = remaining % packed_seq_len;
        remaining /= packed_seq_len;
        const int depth_idx = remaining % depth;
        remaining /= depth;
        const int head_idx = remaining % heads;
        const int batch_idx = remaining / heads;
        const int seq0 = 2 * packed_seq;
        const size_t input0 =
            ((static_cast<size_t>(batch_idx) * seq_len + seq0) * heads +
             head_idx) * depth + depth_idx;
        const size_t input1 = input0 + static_cast<size_t>(heads) * depth;
        const float2 values = make_float2(
            __bfloat162float(input[input0]) * quant_scale,
            __bfloat162float(input[input1]) * quant_scale
        );
        const kittens::fp4e2m1_2 packed = kittens::base_types::convertor<
            kittens::fp4e2m1_2,
            float2
        >::convert(values);
        const uint8_t bits = std::bit_cast<uint8_t>(packed);
        const size_t output_base =
            ((static_cast<size_t>(batch_idx) * heads + head_idx) * depth +
             depth_idx) * seq_len;
        // F8F6F4 stores each 16-value group as eight packed E2M1 bytes
        // followed by an eight-byte alignment gap.
        const int physical_seq =
            (packed_seq / 8) * 16 + (packed_seq % 8);
        output[output_base + physical_seq] = bits;
    }
}

template <bool WriteCompact>
__global__ void quantize_fp4_bshd_to_bshd_unpacked_kernel(
    const kittens::bf16 *input,
    uint8_t *output,
    uint8_t *compact_output,
    int batch,
    int seq_len,
    int heads,
    int depth,
    float quant_scale
) {
    const int packed_depth = depth / 2;
    const size_t total = static_cast<size_t>(batch) * seq_len * heads *
        packed_depth;
    for (
        size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        linear < total;
        linear += static_cast<size_t>(blockDim.x) * gridDim.x
    ) {
        size_t remaining = linear;
        const int packed_col = remaining % packed_depth;
        remaining /= packed_depth;
        const int head_idx = remaining % heads;
        remaining /= heads;
        const int seq_idx = remaining % seq_len;
        const int batch_idx = remaining / seq_len;
        const int depth0 = 2 * packed_col;
        const size_t input0 =
            ((static_cast<size_t>(batch_idx) * seq_len + seq_idx) * heads +
             head_idx) * depth + depth0;
        const float2 values = make_float2(
            __bfloat162float(input[input0]) * quant_scale,
            __bfloat162float(input[input0 + 1]) * quant_scale
        );
        const kittens::fp4e2m1_2 packed = kittens::base_types::convertor<
            kittens::fp4e2m1_2,
            float2
        >::convert(values);
        const uint8_t bits = std::bit_cast<uint8_t>(packed);
        const size_t output_base =
            ((static_cast<size_t>(batch_idx) * seq_len + seq_idx) * heads +
             head_idx) * depth;
        const int physical_depth =
            (packed_col / 8) * 16 + (packed_col % 8);
        output[output_base + physical_depth] = bits;
        if constexpr (WriteCompact) {
            const size_t compact_output_base =
                ((static_cast<size_t>(batch_idx) * heads + head_idx) *
                     seq_len + seq_idx) * packed_depth;
            compact_output[compact_output_base + packed_col] = bits;
        }
    }
}

__global__ void quantize_fp4_bshd_to_dual_q_unpacked_kernel(
    const kittens::bf16 *input,
    uint8_t *sequence_packed_output,
    uint8_t *depth_packed_output,
    int batch,
    int seq_len,
    int heads,
    int depth,
    float quant_scale
) {
    const int packed_seq_len = seq_len / 2;
    const int packed_depth = depth / 2;
    const size_t total = static_cast<size_t>(batch) * seq_len * heads *
        packed_depth;
    for (
        size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        linear < total;
        linear += static_cast<size_t>(blockDim.x) * gridDim.x
    ) {
        // Sequence-packed BHDS view used by dK, where sequence is the
        // F8F6F4 reduction axis.
        size_t sequence_remaining = linear;
        const int packed_seq = sequence_remaining % packed_seq_len;
        sequence_remaining /= packed_seq_len;
        const int sequence_depth_idx = sequence_remaining % depth;
        sequence_remaining /= depth;
        const int sequence_head_idx = sequence_remaining % heads;
        const int sequence_batch_idx = sequence_remaining / heads;
        const int seq0 = 2 * packed_seq;
        const size_t sequence_input0 =
            ((static_cast<size_t>(sequence_batch_idx) * seq_len + seq0) *
                 heads +
             sequence_head_idx) * depth + sequence_depth_idx;
        const size_t sequence_input1 =
            sequence_input0 + static_cast<size_t>(heads) * depth;
        const float2 sequence_values = make_float2(
            __bfloat162float(input[sequence_input0]) * quant_scale,
            __bfloat162float(input[sequence_input1]) * quant_scale
        );
        const kittens::fp4e2m1_2 sequence_packed =
            kittens::base_types::convertor<
                kittens::fp4e2m1_2,
                float2
            >::convert(sequence_values);
        const size_t sequence_output_base =
            ((static_cast<size_t>(sequence_batch_idx) * heads +
              sequence_head_idx) * depth + sequence_depth_idx) * seq_len;
        const int physical_seq =
            (packed_seq / 8) * 16 + (packed_seq % 8);
        sequence_packed_output[sequence_output_base + physical_seq] =
            std::bit_cast<uint8_t>(sequence_packed);

        // Dense depth-packed BSHD view used by the scaled NVFP4 score MMA.
        size_t depth_remaining = linear;
        const int packed_col = depth_remaining % packed_depth;
        depth_remaining /= packed_depth;
        const int depth_head_idx = depth_remaining % heads;
        depth_remaining /= heads;
        const int depth_seq_idx = depth_remaining % seq_len;
        const int depth_batch_idx = depth_remaining / seq_len;
        const int depth0 = 2 * packed_col;
        const size_t depth_input0 =
            ((static_cast<size_t>(depth_batch_idx) * seq_len +
              depth_seq_idx) * heads + depth_head_idx) * depth + depth0;
        const float2 depth_values = make_float2(
            __bfloat162float(input[depth_input0]) * quant_scale,
            __bfloat162float(input[depth_input0 + 1]) * quant_scale
        );
        const kittens::fp4e2m1_2 depth_packed =
            kittens::base_types::convertor<
                kittens::fp4e2m1_2,
                float2
            >::convert(depth_values);
        const size_t depth_output_base =
            ((static_cast<size_t>(depth_batch_idx) * heads +
              depth_head_idx) * seq_len + depth_seq_idx) * packed_depth;
        depth_packed_output[depth_output_base + packed_col] =
            std::bit_cast<uint8_t>(depth_packed);
    }
}

__global__ void quantize_fp4_dual_qk_unpacked_kernel(
    const kittens::bf16 *q,
    const kittens::bf16 *k,
    uint8_t *q_sequence_packed,
    uint8_t *q_depth_packed,
    uint8_t *k_depth_aligned,
    uint8_t *k_depth_packed,
    uint8_t *q_sequence_compact,
    uint8_t *k_sequence_compact,
    int batch,
    int seq_len,
    int heads,
    int depth,
    float q_quant_scale,
    float k_quant_scale
) {
    // One thread owns a 2-sequence x 2-depth patch.  Quantizing Q by depth
    // first lets the same two E2M1 bytes feed both the compact score view and
    // the transposed sequence-packed dK view.  The previous dual-Q path
    // loaded and converted every Q value a second time for that transpose.
    const int packed_seq_len = seq_len / 2;
    const int packed_depth = depth / 2;
    const size_t total = static_cast<size_t>(batch) * packed_seq_len *
        heads * packed_depth;
    for (
        size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x +
            threadIdx.x;
        linear < total;
        linear += static_cast<size_t>(blockDim.x) * gridDim.x
    ) {
        size_t remaining = linear;
        const int packed_col = remaining % packed_depth;
        remaining /= packed_depth;
        const int head_idx = remaining % heads;
        remaining /= heads;
        const int packed_seq = remaining % packed_seq_len;
        const int batch_idx = remaining / packed_seq_len;
        const int seq0 = 2 * packed_seq;
        const int seq1 = seq0 + 1;
        const int depth0 = 2 * packed_col;

        const size_t q_input0 =
            ((static_cast<size_t>(batch_idx) * seq_len + seq0) * heads +
             head_idx) * depth + depth0;
        const size_t q_input1 =
            q_input0 + static_cast<size_t>(heads) * depth;
        const float2 q_values0 = make_float2(
            __bfloat162float(q[q_input0]) * q_quant_scale,
            __bfloat162float(q[q_input0 + 1]) * q_quant_scale
        );
        const float2 q_values1 = make_float2(
            __bfloat162float(q[q_input1]) * q_quant_scale,
            __bfloat162float(q[q_input1 + 1]) * q_quant_scale
        );
        const uint8_t q_bits0 = std::bit_cast<uint8_t>(
            kittens::base_types::convertor<
                kittens::fp4e2m1_2,
                float2
            >::convert(q_values0)
        );
        const uint8_t q_bits1 = std::bit_cast<uint8_t>(
            kittens::base_types::convertor<
                kittens::fp4e2m1_2,
                float2
            >::convert(q_values1)
        );

        const size_t q_compact_base =
            ((static_cast<size_t>(batch_idx) * heads + head_idx) * seq_len) *
            packed_depth;
        q_depth_packed[
            q_compact_base + static_cast<size_t>(seq0) * packed_depth +
            packed_col
        ] = q_bits0;
        q_depth_packed[
            q_compact_base + static_cast<size_t>(seq1) * packed_depth +
            packed_col
        ] = q_bits1;

        const int physical_seq =
            (packed_seq / 8) * 16 + (packed_seq % 8);
        const uint8_t q_depth0_bits = static_cast<uint8_t>(
            (q_bits0 & 0x0fu) | ((q_bits1 & 0x0fu) << 4)
        );
        const uint8_t q_depth1_bits = static_cast<uint8_t>(
            ((q_bits0 >> 4) & 0x0fu) | (q_bits1 & 0xf0u)
        );
        const size_t q_sequence_base =
            ((static_cast<size_t>(batch_idx) * heads + head_idx) * depth +
             depth0) * seq_len;
        q_sequence_packed[q_sequence_base + physical_seq] = q_depth0_bits;
        q_sequence_packed[q_sequence_base + seq_len + physical_seq] =
            q_depth1_bits;
        // The aligned F8F6F4 container reserves the upper eight bytes of
        // every 16-byte group.  Write those gaps in the same pass so the
        // outputs can use empty() instead of launching full-tensor memsets.
        q_sequence_packed[q_sequence_base + physical_seq + 8] = 0;
        q_sequence_packed[q_sequence_base + seq_len + physical_seq + 8] = 0;
        if (q_sequence_compact != nullptr) {
            const size_t q_sequence_compact_base =
                ((static_cast<size_t>(batch_idx) * heads + head_idx) *
                     depth +
                 depth0) * packed_seq_len;
            q_sequence_compact[
                q_sequence_compact_base + packed_seq
            ] = q_depth0_bits;
            q_sequence_compact[
                q_sequence_compact_base + packed_seq_len + packed_seq
            ] = q_depth1_bits;
        }

        const size_t k_input0 = q_input0;
        const size_t k_input1 = q_input1;
        const float2 k_values0 = make_float2(
            __bfloat162float(k[k_input0]) * k_quant_scale,
            __bfloat162float(k[k_input0 + 1]) * k_quant_scale
        );
        const float2 k_values1 = make_float2(
            __bfloat162float(k[k_input1]) * k_quant_scale,
            __bfloat162float(k[k_input1 + 1]) * k_quant_scale
        );
        const uint8_t k_bits0 = std::bit_cast<uint8_t>(
            kittens::base_types::convertor<
                kittens::fp4e2m1_2,
                float2
            >::convert(k_values0)
        );
        const uint8_t k_bits1 = std::bit_cast<uint8_t>(
            kittens::base_types::convertor<
                kittens::fp4e2m1_2,
                float2
            >::convert(k_values1)
        );

        const size_t k_compact_base = q_compact_base;
        k_depth_packed[
            k_compact_base + static_cast<size_t>(seq0) * packed_depth +
            packed_col
        ] = k_bits0;
        k_depth_packed[
            k_compact_base + static_cast<size_t>(seq1) * packed_depth +
            packed_col
        ] = k_bits1;

        const int physical_depth =
            (packed_col / 8) * 16 + (packed_col % 8);
        const size_t k_aligned_base0 =
            ((static_cast<size_t>(batch_idx) * seq_len + seq0) * heads +
             head_idx) * depth;
        const size_t k_aligned_base1 =
            k_aligned_base0 + static_cast<size_t>(heads) * depth;
        k_depth_aligned[k_aligned_base0 + physical_depth] = k_bits0;
        k_depth_aligned[k_aligned_base1 + physical_depth] = k_bits1;
        k_depth_aligned[k_aligned_base0 + physical_depth + 8] = 0;
        k_depth_aligned[k_aligned_base1 + physical_depth + 8] = 0;
        if (k_sequence_compact != nullptr) {
            const uint8_t k_depth0_bits = static_cast<uint8_t>(
                (k_bits0 & 0x0fu) | ((k_bits1 & 0x0fu) << 4)
            );
            const uint8_t k_depth1_bits = static_cast<uint8_t>(
                ((k_bits0 >> 4) & 0x0fu) | (k_bits1 & 0xf0u)
            );
            const size_t k_sequence_compact_base =
                ((static_cast<size_t>(batch_idx) * heads + head_idx) *
                     depth +
                 depth0) * packed_seq_len;
            k_sequence_compact[
                k_sequence_compact_base + packed_seq
            ] = k_depth0_bits;
            k_sequence_compact[
                k_sequence_compact_base + packed_seq_len + packed_seq
            ] = k_depth1_bits;
        }
    }
}

template <bool FoldScale16>
__device__ __forceinline__ uint8_t quantize_fp4_bf16_pair(
    const kittens::bf16 *input,
    float quant_scale
) {
    uint32_t packed_bf16 = *reinterpret_cast<const uint32_t *>(input);
    if constexpr (FoldScale16) {
        // Multiplication by 16 is an exact +4 BF16 exponent adjustment for
        // every normal value in the quantizer's representable input range.
        // Values near zero still round to E2M1 zero.  SIMD halfword addition
        // folds both scalar multiplies into one integer instruction.
        packed_bf16 = __vadd2(packed_bf16, 0x02000200u);
    }
    const kittens::bf16_2 pair =
        *reinterpret_cast<const kittens::bf16_2 *>(&packed_bf16);
    float2 values = __bfloat1622float2(pair);
    if constexpr (!FoldScale16) {
        values.x *= quant_scale;
        values.y *= quant_scale;
    }
    return std::bit_cast<uint8_t>(
        kittens::base_types::convertor<
            kittens::fp4e2m1_2,
            float2
        >::convert(values)
    );
}

template <
    int Depth,
    bool FoldScale16,
    bool EmitSequenceCompact = false,
    bool ReadDeviceScales = false
>
__global__ void quantize_fp4_dual_qk_tiled_kernel(
    const kittens::bf16 *q,
    const kittens::bf16 *k,
    uint8_t *q_sequence_packed,
    uint8_t *q_depth_packed,
    uint8_t *k_depth_aligned,
    uint8_t *k_depth_packed,
    uint8_t *q_sequence_compact,
    uint8_t *k_sequence_compact,
    int seq_len,
    int heads,
    float q_quant_scale,
    float k_quant_scale,
    const float *device_quant_scales
) {
    static_assert(Depth % 2 == 0);
    constexpr int kRows = 32;
    constexpr int kPackedDepth = Depth / 2;
#if TK_FA4_BWD_PURE_MXFP4_DQ_N256
    constexpr int kSequenceCompactDepth = 256;
#else
    constexpr int kSequenceCompactDepth = Depth;
#endif
    // Padding breaks the 96-byte row-stride bank alias when a warp gathers
    // one depth value from all 32 sequence rows for the BHDS transpose.
    __shared__ uint8_t q_codes[kRows][kPackedDepth + 1];
    __shared__ uint8_t k_codes[kRows][kPackedDepth + 1];

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int seq_base = static_cast<int>(blockIdx.x) * kRows;
    if constexpr (ReadDeviceScales) {
        constexpr int kScaleRecordWords = 7;
        const float *head_scales = device_quant_scales +
            (static_cast<size_t>(batch_idx) * heads + head_idx) *
                kScaleRecordWords;
        q_quant_scale = head_scales[0];
        k_quant_scale = head_scales[1];
    }
    constexpr int kTileValues = kRows * kPackedDepth;
    for (int item = threadIdx.x; item < kTileValues; item += blockDim.x) {
        const int row = item / kPackedDepth;
        const int packed_col = item - row * kPackedDepth;
        const int seq_idx = seq_base + row;
        const int depth0 = 2 * packed_col;
        const size_t input_base =
            ((static_cast<size_t>(batch_idx) * seq_len + seq_idx) * heads +
             head_idx) * Depth + depth0;

        const uint8_t q_bits = quantize_fp4_bf16_pair<FoldScale16>(
            q + input_base,
            q_quant_scale
        );
        q_codes[row][packed_col] = q_bits;
        const size_t compact_base =
            ((static_cast<size_t>(batch_idx) * heads + head_idx) * seq_len +
             seq_idx) * kPackedDepth + packed_col;
        q_depth_packed[compact_base] = q_bits;

        const uint8_t k_bits = quantize_fp4_bf16_pair<FoldScale16>(
            k + input_base,
            k_quant_scale
        );
        if constexpr (EmitSequenceCompact) {
            k_codes[row][packed_col] = k_bits;
        }
        k_depth_packed[compact_base] = k_bits;
        const int physical_depth =
            (packed_col / 8) * 16 + (packed_col % 8);
        const size_t k_aligned_base =
            ((static_cast<size_t>(batch_idx) * seq_len + seq_idx) * heads +
             head_idx) * Depth;
        k_depth_aligned[k_aligned_base + physical_depth] = k_bits;
        k_depth_aligned[k_aligned_base + physical_depth + 8] = 0;
    }
    __syncthreads();

    // Each warp writes one coalesced 32-byte slice of the aligned BHDS view.
    // Lanes 0..7 and 16..23 repack adjacent sequence rows; the other lanes
    // materialize the required F8F6F4 alignment gaps.
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int physical_seq_base = static_cast<int>(blockIdx.x) * 32;
    for (int depth_idx = warp; depth_idx < Depth; depth_idx += 8) {
        const int group = lane >> 4;
        const int within_group = lane & 15;
        uint8_t output = 0;
        if (within_group < 8) {
            const int packed_seq = group * 8 + within_group;
            const uint8_t row0 = q_codes[2 * packed_seq][depth_idx / 2];
            const uint8_t row1 = q_codes[2 * packed_seq + 1][depth_idx / 2];
            if ((depth_idx & 1) == 0) {
                output = static_cast<uint8_t>(
                    (row0 & 0x0fu) | ((row1 & 0x0fu) << 4)
                );
            } else {
                output = static_cast<uint8_t>(
                    ((row0 >> 4) & 0x0fu) | (row1 & 0xf0u)
                );
            }
        }
        const size_t q_aligned_base =
            ((static_cast<size_t>(batch_idx) * heads + head_idx) * Depth +
             depth_idx) * seq_len;
        q_sequence_packed[
            q_aligned_base + physical_seq_base + lane
        ] = output;
        if constexpr (EmitSequenceCompact) {
            if (lane < 16) {
                const int packed_seq = lane;
                const uint8_t q_row0 =
                    q_codes[2 * packed_seq][depth_idx / 2];
                const uint8_t q_row1 =
                    q_codes[2 * packed_seq + 1][depth_idx / 2];
                const uint8_t k_row0 =
                    k_codes[2 * packed_seq][depth_idx / 2];
                const uint8_t k_row1 =
                    k_codes[2 * packed_seq + 1][depth_idx / 2];
                const bool high_depth = (depth_idx & 1) != 0;
                const uint8_t q_output = high_depth
                    ? static_cast<uint8_t>(
                          ((q_row0 >> 4) & 0x0fu) | (q_row1 & 0xf0u)
                      )
                    : static_cast<uint8_t>(
                          (q_row0 & 0x0fu) | ((q_row1 & 0x0fu) << 4)
                      );
                const uint8_t k_output = high_depth
                    ? static_cast<uint8_t>(
                          ((k_row0 >> 4) & 0x0fu) | (k_row1 & 0xf0u)
                      )
                    : static_cast<uint8_t>(
                          (k_row0 & 0x0fu) | ((k_row1 & 0x0fu) << 4)
                      );
                const int packed_seq_len = seq_len / 2;
                // M256 dK uses a 4x1 TMEM map.  Its N192 operation is issued
                // as an N128 head plus an N64 tail, so stage compact Q as
                // each instruction's CTA-local N halves:
                // CTA0=[0:64,128:160], CTA1=[64:128,160:192].
                // M128 dQ instead uses a 2x2 TMEM map; its compact K remains
                // in contiguous 96-feature CTA halves below.
                const int compact_feature = depth_idx < 64
                    ? depth_idx
                    : depth_idx < 128
                        ? depth_idx + 32
                        : depth_idx < 160
                            ? depth_idx - 64
                            : depth_idx;
                const size_t q_compact_sequence_base =
                    ((static_cast<size_t>(batch_idx) * heads + head_idx) *
                         kSequenceCompactDepth +
                     compact_feature) * packed_seq_len +
                    static_cast<size_t>(blockIdx.x) * 16;
                const size_t k_compact_sequence_base =
                    ((static_cast<size_t>(batch_idx) * heads + head_idx) *
                         kSequenceCompactDepth +
                     depth_idx) * packed_seq_len +
                    static_cast<size_t>(blockIdx.x) * 16;
#if TK_FA4_BWD_PURE_MXFP4_B8_STMATRIX_TRANSPOSE
                // The B8 dS transpose emits each K32 block with packed-row
                // bit order [K2,K3,K4,K1].  Fold the identical permutation
                // into K's existing sequence pack so the dQ reduction remains
                // mathematically unchanged without another producer shuffle.
                const int k_packed_seq =
                    (packed_seq >> 1) | ((packed_seq & 1) << 3);
#else
                const int k_packed_seq = packed_seq;
#endif
                q_sequence_compact[
                    q_compact_sequence_base + packed_seq
                ] = q_output;
                k_sequence_compact[
                    k_compact_sequence_base + k_packed_seq
                ] = k_output;
            }
        }
    }
}

#if TK_FA4_BWD_PURE_NVFP4_QK
__global__ void quantize_fp4_qk_sequence_nvfp4_kernel(
    const kittens::bf16 *q,
    const kittens::bf16 *k,
    uint8_t *q_sequence_compact,
    uint8_t *k_sequence_compact,
    kittens::fp8e4m3 *q_scale_prepared,
    kittens::fp8e4m3 *k_scale_prepared,
    int seq_len,
    int heads
) {
    constexpr int kDepth = 192;
    constexpr int kFeaturesPerRank = 96;
    constexpr int kGroupsPerK64 = 4;
    constexpr int kScalePageBytes = 512;
    const int rank = static_cast<int>(blockIdx.z) & 1;
    const int batch_idx = static_cast<int>(blockIdx.z) >> 1;
    const int head_idx = static_cast<int>(blockIdx.y);
    const int k64_idx = static_cast<int>(blockIdx.x);
    const int seq_base = 64 * k64_idx;
    const int packed_seq_len = seq_len / 2;
    const int k64_blocks = seq_len / 64;

    for (
        int item = static_cast<int>(threadIdx.x);
        item < kFeaturesPerRank * kGroupsPerK64;
        item += static_cast<int>(blockDim.x)
    ) {
        const int local_feature = item / kGroupsPerK64;
        const int group = item - local_feature * kGroupsPerK64;
        const int physical_feature = rank * kFeaturesPerRank + local_feature;
        // dK's N192 map stores original feature ranges
        // [0:64,128:160] in CTA0 and [64:128,160:192] in CTA1.
        const int q_feature = physical_feature < 64
            ? physical_feature
            : physical_feature < 96
                ? physical_feature + 64
                : physical_feature < 160
                    ? physical_feature - 32
                    : physical_feature;
        const int k_feature = physical_feature;

        float q_values[16];
        float q_amax = 0.0f;
        #pragma unroll
        for (int element = 0; element < 16; ++element) {
            const int seq_idx = seq_base + group * 16 + element;
            const size_t input_idx =
                ((static_cast<size_t>(batch_idx) * seq_len + seq_idx) *
                     heads +
                 head_idx) * kDepth + q_feature;
            const float value = __bfloat162float(q[input_idx]);
            q_values[element] = value;
            q_amax = fmaxf(q_amax, fabsf(value));
        }
        const kittens::fp8e4m3 q_scale =
            kittens::base_types::convertor<
                kittens::fp8e4m3,
                float
            >::convert(q_amax * (1.0f / 6.0f));
        const float q_scale_dec =
            kittens::base_types::convertor<
                float,
                kittens::fp8e4m3
            >::convert(q_scale);
        const float q_multiplier = q_scale_dec > 0.0f
            ? 1.0f / q_scale_dec
            : 0.0f;
        const size_t q_output_base =
            ((static_cast<size_t>(batch_idx) * heads + head_idx) * kDepth +
             physical_feature) * packed_seq_len + seq_base / 2 + group * 8;
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const float2 normalized = make_float2(
                q_values[2 * pair + 0] * q_multiplier,
                q_values[2 * pair + 1] * q_multiplier
            );
            q_sequence_compact[q_output_base + pair] =
                std::bit_cast<uint8_t>(
                    kittens::base_types::convertor<
                        kittens::fp4e2m1_2,
                        float2
                    >::convert(normalized)
                );
        }

        float k_values[16];
        float k_amax = 0.0f;
        const int x32_half = group >> 1;
        const int packed_parity = group & 1;
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const int source_pair = 2 * pair + packed_parity;
            const int seq0 = seq_base + x32_half * 32 + 2 * source_pair;
            const size_t input_idx =
                ((static_cast<size_t>(batch_idx) * seq_len + seq0) * heads +
                 head_idx) * kDepth + k_feature;
            const float value0 = __bfloat162float(k[input_idx]);
            const float value1 = __bfloat162float(
                k[input_idx + static_cast<size_t>(heads) * kDepth]
            );
            k_values[2 * pair + 0] = value0;
            k_values[2 * pair + 1] = value1;
            k_amax = fmaxf(k_amax, fmaxf(fabsf(value0), fabsf(value1)));
        }
        const kittens::fp8e4m3 k_scale =
            kittens::base_types::convertor<
                kittens::fp8e4m3,
                float
            >::convert(k_amax * (1.0f / 6.0f));
        const float k_scale_dec =
            kittens::base_types::convertor<
                float,
                kittens::fp8e4m3
            >::convert(k_scale);
        const float k_multiplier = k_scale_dec > 0.0f
            ? 1.0f / k_scale_dec
            : 0.0f;
        const size_t k_output_base =
            ((static_cast<size_t>(batch_idx) * heads + head_idx) * kDepth +
             physical_feature) * packed_seq_len + seq_base / 2 + group * 8;
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const float2 normalized = make_float2(
                k_values[2 * pair + 0] * k_multiplier,
                k_values[2 * pair + 1] * k_multiplier
            );
            k_sequence_compact[k_output_base + pair] =
                std::bit_cast<uint8_t>(
                    kittens::base_types::convertor<
                        kittens::fp4e2m1_2,
                        float2
                    >::convert(normalized)
                );
        }

        // CTA-group-2 has one tensor issuer.  Its N192 result is an N128
        // head plus an N64 tail, so scale pages follow logical output N
        // rather than the two 96-row payload partitions resident in the
        // CTAs.  The existing two-page outer dimension is exactly the
        // required head/tail storage; only its interpretation changes.
        const int q_n_page = q_feature >> 7;
        const int q_scale_row = q_feature & 127;
        const int q_scale_page =
            (((batch_idx * heads + head_idx) * 2 + q_n_page) *
                 k64_blocks) +
            k64_idx;
        const int q_scale_offset =
            (q_scale_row & 31) * 16 +
            ((q_scale_row >> 5) & 3) * 4 + group;
        q_scale_prepared[
            q_scale_page * kScalePageBytes + q_scale_offset
        ] = q_scale;

        const int k_n_page = k_feature >> 7;
        const int k_scale_row = k_feature & 127;
        const int k_scale_page =
            (((batch_idx * heads + head_idx) * 2 + k_n_page) *
                 k64_blocks) +
            k64_idx;
        const int k_scale_offset =
            (k_scale_row & 31) * 16 +
            ((k_scale_row >> 5) & 3) * 4 + group;
        k_scale_prepared[
            k_scale_page * kScalePageBytes + k_scale_offset
        ] = k_scale;
    }
}
#endif

at::Tensor quantize_fp4_bhds_unpacked(at::Tensor input, float quant_scale) {
    tkfa4::check_bshd(input, "input", at::kBFloat16);
    TORCH_CHECK(
        input.is_contiguous() && input.is_cuda(),
        "input must be contiguous CUDA BF16"
    );
    TORCH_CHECK(
        input.size(1) % 2 == 0,
        "sequence length must be even for packed FP4"
    );
    TORCH_CHECK(
        std::isfinite(quant_scale) && quant_scale > 0.0f,
        "quant_scale must be finite and positive"
    );
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int batch = static_cast<int>(input.size(0));
    const int seq_len = static_cast<int>(input.size(1));
    const int heads = static_cast<int>(input.size(2));
    const int depth = static_cast<int>(input.size(3));
    auto options = input.options().dtype(at::ScalarType::Byte);
    at::Tensor output = at::zeros(
        {batch, heads, depth, seq_len},
        options
    );
    const size_t total = static_cast<size_t>(batch) * heads * depth *
        (seq_len / 2);
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(
        std::min<size_t>((total + kThreads - 1) / kThreads, 65535)
    );
    quantize_fp4_bshd_to_bhds_unpacked_kernel<<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const kittens::bf16 *>(input.data_ptr()),
        reinterpret_cast<uint8_t *>(output.data_ptr()),
        batch,
        seq_len,
        heads,
        depth,
        quant_scale
    );
    CUDACHECK(cudaGetLastError());
    return output;
}

at::Tensor quantize_fp4_bshd_unpacked(at::Tensor input, float quant_scale) {
    tkfa4::check_bshd(input, "input", at::kBFloat16);
    TORCH_CHECK(
        input.is_contiguous() && input.is_cuda(),
        "input must be contiguous CUDA BF16"
    );
    TORCH_CHECK(
        input.size(3) % 2 == 0,
        "depth must be even for packed FP4"
    );
    TORCH_CHECK(
        std::isfinite(quant_scale) && quant_scale > 0.0f,
        "quant_scale must be finite and positive"
    );
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int batch = static_cast<int>(input.size(0));
    const int seq_len = static_cast<int>(input.size(1));
    const int heads = static_cast<int>(input.size(2));
    const int depth = static_cast<int>(input.size(3));
    at::Tensor output = at::zeros(
        input.sizes(),
        input.options().dtype(at::ScalarType::Byte)
    );
    const size_t total = static_cast<size_t>(batch) * seq_len * heads *
        (depth / 2);
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(
        std::min<size_t>((total + kThreads - 1) / kThreads, 65535)
    );
    quantize_fp4_bshd_to_bshd_unpacked_kernel<false><<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const kittens::bf16 *>(input.data_ptr()),
        reinterpret_cast<uint8_t *>(output.data_ptr()),
        nullptr,
        batch,
        seq_len,
        heads,
        depth,
        quant_scale
    );
    CUDACHECK(cudaGetLastError());
    return output;
}

std::vector<at::Tensor> quantize_fp4_dual_k_unpacked(
    at::Tensor input,
    float quant_scale
) {
    tkfa4::check_bshd(input, "input", at::kBFloat16);
    TORCH_CHECK(
        input.is_contiguous() && input.is_cuda(),
        "input must be contiguous CUDA BF16"
    );
    TORCH_CHECK(
        input.size(3) % 2 == 0,
        "depth must be even for packed FP4"
    );
    TORCH_CHECK(
        std::isfinite(quant_scale) && quant_scale > 0.0f,
        "quant_scale must be finite and positive"
    );
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int batch = static_cast<int>(input.size(0));
    const int seq_len = static_cast<int>(input.size(1));
    const int heads = static_cast<int>(input.size(2));
    const int depth = static_cast<int>(input.size(3));
    auto byte_options = input.options().dtype(at::ScalarType::Byte);
    at::Tensor aligned = at::zeros(input.sizes(), byte_options);
    at::Tensor compact = at::empty(
        {batch, heads, seq_len, depth / 2},
        byte_options
    );
    const size_t total = static_cast<size_t>(batch) * seq_len * heads *
        (depth / 2);
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(
        std::min<size_t>((total + kThreads - 1) / kThreads, 65535)
    );
    quantize_fp4_bshd_to_bshd_unpacked_kernel<true><<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const kittens::bf16 *>(input.data_ptr()),
        reinterpret_cast<uint8_t *>(aligned.data_ptr()),
        reinterpret_cast<uint8_t *>(compact.data_ptr()),
        batch,
        seq_len,
        heads,
        depth,
        quant_scale
    );
    CUDACHECK(cudaGetLastError());
    return {aligned, compact};
}

std::vector<at::Tensor> quantize_fp4_dual_q_unpacked(
    at::Tensor input,
    float quant_scale
) {
    tkfa4::check_bshd(input, "input", at::kBFloat16);
    TORCH_CHECK(
        input.is_contiguous() && input.is_cuda(),
        "input must be contiguous CUDA BF16"
    );
    TORCH_CHECK(
        input.size(1) % 2 == 0 && input.size(3) % 2 == 0,
        "sequence length and depth must be even for dual packed FP4 Q"
    );
    TORCH_CHECK(
        std::isfinite(quant_scale) && quant_scale > 0.0f,
        "quant_scale must be finite and positive"
    );
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int batch = static_cast<int>(input.size(0));
    const int seq_len = static_cast<int>(input.size(1));
    const int heads = static_cast<int>(input.size(2));
    const int depth = static_cast<int>(input.size(3));
    auto byte_options = input.options().dtype(at::ScalarType::Byte);
    at::Tensor sequence_packed = at::zeros(
        {batch, heads, depth, seq_len},
        byte_options
    );
    at::Tensor depth_packed = at::empty(
        {batch, heads, seq_len, depth / 2},
        byte_options
    );
    const size_t total = static_cast<size_t>(batch) * seq_len * heads *
        (depth / 2);
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(
        std::min<size_t>((total + kThreads - 1) / kThreads, 65535)
    );
    quantize_fp4_bshd_to_dual_q_unpacked_kernel<<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const kittens::bf16 *>(input.data_ptr()),
        reinterpret_cast<uint8_t *>(sequence_packed.data_ptr()),
        reinterpret_cast<uint8_t *>(depth_packed.data_ptr()),
        batch,
        seq_len,
        heads,
        depth,
        quant_scale
    );
    CUDACHECK(cudaGetLastError());
    return {sequence_packed, depth_packed};
}

std::vector<at::Tensor> quantize_fp4_dual_qk_unpacked(
    at::Tensor q,
    at::Tensor k,
    float q_quant_scale,
    float k_quant_scale
) {
    tkfa4::check_bshd(q, "q", at::kBFloat16);
    tkfa4::check_bshd(k, "k", at::kBFloat16);
    TORCH_CHECK(
        q.sizes() == k.sizes(),
        "q and k must have identical BSHD shapes"
    );
    TORCH_CHECK(
        q.is_contiguous() && k.is_contiguous() && q.is_cuda() && k.is_cuda(),
        "q and k must be contiguous CUDA BF16 tensors"
    );
    TORCH_CHECK(
        q.size(1) % 2 == 0 && q.size(3) % 2 == 0,
        "sequence length and depth must be even for fused dual FP4 packing"
    );
    TORCH_CHECK(
        std::isfinite(q_quant_scale) && q_quant_scale > 0.0f &&
            std::isfinite(k_quant_scale) && k_quant_scale > 0.0f,
        "q_quant_scale and k_quant_scale must be finite and positive"
    );
    kittens::py::device_check(q, k);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int heads = static_cast<int>(q.size(2));
    const int depth = static_cast<int>(q.size(3));
    auto byte_options = q.options().dtype(at::ScalarType::Byte);
    at::Tensor q_sequence_packed = at::empty(
        {batch, heads, depth, seq_len},
        byte_options
    );
    at::Tensor q_depth_packed = at::empty(
        {batch, heads, seq_len, depth / 2},
        byte_options
    );
    at::Tensor k_depth_aligned = at::empty(q.sizes(), byte_options);
    at::Tensor k_depth_packed = at::empty(
        {batch, heads, seq_len, depth / 2},
        byte_options
    );
    constexpr int kThreads = 256;
    if (depth == 192 && seq_len % 32 == 0) {
        const dim3 grid(seq_len / 32, heads, batch);
        if (q_quant_scale == 16.0f && k_quant_scale == 16.0f) {
            quantize_fp4_dual_qk_tiled_kernel<192, true><<<
                grid,
                kThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
                reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
                reinterpret_cast<uint8_t *>(q_sequence_packed.data_ptr()),
                reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
                reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr()),
                reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
                nullptr,
                nullptr,
                seq_len,
                heads,
                q_quant_scale,
                k_quant_scale,
                nullptr
            );
        } else {
            quantize_fp4_dual_qk_tiled_kernel<192, false><<<
                grid,
                kThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
                reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
                reinterpret_cast<uint8_t *>(q_sequence_packed.data_ptr()),
                reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
                reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr()),
                reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
                nullptr,
                nullptr,
                seq_len,
                heads,
                q_quant_scale,
                k_quant_scale,
                nullptr
            );
        }
    } else {
        const size_t total = static_cast<size_t>(batch) * (seq_len / 2) *
            heads * (depth / 2);
        const int blocks = static_cast<int>(
            std::min<size_t>((total + kThreads - 1) / kThreads, 65535)
        );
        quantize_fp4_dual_qk_unpacked_kernel<<<
            blocks,
            kThreads,
            0,
            at::cuda::getCurrentCUDAStream()
        >>>(
            reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
            reinterpret_cast<uint8_t *>(q_sequence_packed.data_ptr()),
            reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
            reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr()),
            reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
            nullptr,
            nullptr,
            batch,
            seq_len,
            heads,
            depth,
            q_quant_scale,
            k_quant_scale
        );
    }
    CUDACHECK(cudaGetLastError());
    return {
        q_sequence_packed,
        q_depth_packed,
        k_depth_aligned,
        k_depth_packed
    };
}

std::vector<at::Tensor> quantize_fp4_dual_qk_blockscale(
    at::Tensor q,
    at::Tensor k,
    float q_quant_scale,
    float k_quant_scale
) {
    tkfa4::check_bshd(q, "q", at::kBFloat16);
    tkfa4::check_bshd(k, "k", at::kBFloat16);
    TORCH_CHECK(
        q.sizes() == k.sizes(),
        "q and k must have identical BSHD shapes"
    );
    TORCH_CHECK(
        q.is_contiguous() && k.is_contiguous() && q.is_cuda() && k.is_cuda(),
        "q and k must be contiguous CUDA BF16 tensors"
    );
    TORCH_CHECK(
        q.size(1) % 32 == 0 && q.size(3) == 192,
        "block-scaled FP4 Q/K packing requires D192 and sequence length "
        "divisible by 32"
    );
    TORCH_CHECK(
        std::isfinite(q_quant_scale) && q_quant_scale > 0.0f &&
            std::isfinite(k_quant_scale) && k_quant_scale > 0.0f,
        "q_quant_scale and k_quant_scale must be finite and positive"
    );
    kittens::py::device_check(q, k);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int heads = static_cast<int>(q.size(2));
    constexpr int kDepth = 192;
    auto byte_options = q.options().dtype(at::ScalarType::Byte);
    auto scale_options = q.options().dtype(at::ScalarType::Float8_e4m3fn);
    at::Tensor q_sequence_aligned = at::empty(
        {batch, heads, kDepth, seq_len},
        byte_options
    );
    at::Tensor q_depth_packed = at::empty(
        {batch, heads, seq_len, kDepth / 2},
        byte_options
    );
    at::Tensor k_depth_aligned = at::empty(q.sizes(), byte_options);
    at::Tensor k_depth_packed = at::empty(
        {batch, heads, seq_len, kDepth / 2},
        byte_options
    );
#if TK_FA4_BWD_PURE_MXFP4_DQ_N256
    constexpr int kSequenceCompactDepth = 256;
    at::Tensor q_sequence_compact = at::zeros(
        {batch, heads, kSequenceCompactDepth, seq_len / 2},
        byte_options
    );
    at::Tensor k_sequence_compact = at::zeros(
        {batch, heads, kSequenceCompactDepth, seq_len / 2},
        byte_options
    );
#else
    at::Tensor q_sequence_compact = at::empty(
        {batch, heads, kDepth, seq_len / 2},
        byte_options
    );
    at::Tensor k_sequence_compact = at::empty(
        {batch, heads, kDepth, seq_len / 2},
        byte_options
    );
#endif
#if TK_FA4_BWD_PURE_NVFP4_QK
    TORCH_CHECK(
        seq_len % 64 == 0,
        "NVFP4 Q/K block scaling requires sequence length divisible by 64"
    );
    const int scale_pages = batch * heads * 2 * (seq_len / 64);
    at::Tensor q_sequence_nvfp4_scale = at::zeros(
        {scale_pages, 32, 16},
        scale_options
    );
    at::Tensor k_sequence_nvfp4_scale = at::zeros(
        {scale_pages, 32, 16},
        scale_options
    );
#else
    // Keep the experimental block-scaled Q/K operands in the public harness
    // without charging the retained fixed-scale route for their preparation.
    at::Tensor q_sequence_nvfp4_scale = at::empty({0}, scale_options);
    at::Tensor k_sequence_nvfp4_scale = at::empty({0}, scale_options);
#endif

    constexpr int kThreads = 256;
    const dim3 grid(seq_len / 32, heads, batch);
    if (q_quant_scale == 16.0f && k_quant_scale == 16.0f) {
        quantize_fp4_dual_qk_tiled_kernel<192, true, true><<<
            grid,
            kThreads,
            0,
            at::cuda::getCurrentCUDAStream()
        >>>(
            reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
            reinterpret_cast<uint8_t *>(q_sequence_aligned.data_ptr()),
            reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
            reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr()),
            reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
            reinterpret_cast<uint8_t *>(q_sequence_compact.data_ptr()),
            reinterpret_cast<uint8_t *>(k_sequence_compact.data_ptr()),
            seq_len,
            heads,
            q_quant_scale,
            k_quant_scale,
            nullptr
        );
    } else {
        quantize_fp4_dual_qk_tiled_kernel<192, false, true><<<
            grid,
            kThreads,
            0,
            at::cuda::getCurrentCUDAStream()
        >>>(
            reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
            reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
            reinterpret_cast<uint8_t *>(q_sequence_aligned.data_ptr()),
            reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
            reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr()),
            reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
            reinterpret_cast<uint8_t *>(q_sequence_compact.data_ptr()),
            reinterpret_cast<uint8_t *>(k_sequence_compact.data_ptr()),
            seq_len,
            heads,
            q_quant_scale,
            k_quant_scale,
            nullptr
        );
    }
    CUDACHECK(cudaGetLastError());
#if TK_FA4_BWD_PURE_NVFP4_QK
    const dim3 nvfp4_grid(seq_len / 64, heads, batch * 2);
    quantize_fp4_qk_sequence_nvfp4_kernel<<<
        nvfp4_grid,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
        reinterpret_cast<uint8_t *>(q_sequence_compact.data_ptr()),
        reinterpret_cast<uint8_t *>(k_sequence_compact.data_ptr()),
        reinterpret_cast<kittens::fp8e4m3 *>(
            q_sequence_nvfp4_scale.data_ptr()
        ),
        reinterpret_cast<kittens::fp8e4m3 *>(
            k_sequence_nvfp4_scale.data_ptr()
        ),
        seq_len,
        heads
    );
    CUDACHECK(cudaGetLastError());
#endif
    return {
        q_sequence_aligned,
        q_depth_packed,
        k_depth_aligned,
        k_depth_packed,
        q_sequence_compact,
        k_sequence_compact,
        q_sequence_nvfp4_scale,
        k_sequence_nvfp4_scale
    };
}

std::vector<at::Tensor> quantize_fp4_dual_qk_adaptive(
    at::Tensor q,
    at::Tensor k,
    float max_quant_scale,
    float min_quant_scale,
    float min_headroom,
    float rms_clip_multiple,
    float softmax_scale,
    float ds_quant_scale
) {
    tkfa4::check_bshd(q, "q", at::kBFloat16);
    tkfa4::check_bshd(k, "k", at::kBFloat16);
    TORCH_CHECK(
        q.sizes() == k.sizes(),
        "q and k must have identical BSHD shapes"
    );
    TORCH_CHECK(
        q.is_contiguous() && k.is_contiguous() && q.is_cuda() && k.is_cuda(),
        "q and k must be contiguous CUDA BF16 tensors"
    );
    TORCH_CHECK(
        q.size(1) % 32 == 0 && q.size(3) == 192,
        "adaptive FP4 Q/K packing requires D192 and sequence length "
        "divisible by 32"
    );
    TORCH_CHECK(
        std::isfinite(max_quant_scale) && max_quant_scale > 0.0f &&
            std::isfinite(min_quant_scale) && min_quant_scale > 0.0f &&
            min_quant_scale <= max_quant_scale &&
            std::isfinite(min_headroom) && min_headroom > 0.0f &&
            min_headroom <= 1.0f &&
            std::isfinite(rms_clip_multiple) && rms_clip_multiple > 0.0f &&
            std::isfinite(softmax_scale) && softmax_scale > 0.0f &&
            std::isfinite(ds_quant_scale) && ds_quant_scale > 0.0f,
        "adaptive scale bounds, minimum headroom, RMS clipping multiple, "
        "and downstream scales must be finite and positive"
    );
    kittens::py::device_check(q, k);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int heads = static_cast<int>(q.size(2));
    constexpr int kDepth = 192;
    auto byte_options = q.options().dtype(at::ScalarType::Byte);
    auto float_options = q.options().dtype(at::ScalarType::Float);
    at::Tensor q_sequence_aligned = at::empty(
        {batch, heads, kDepth, seq_len},
        byte_options
    );
    at::Tensor q_depth_packed = at::empty(
        {batch, heads, seq_len, kDepth / 2},
        byte_options
    );
    at::Tensor k_depth_aligned = at::empty(q.sizes(), byte_options);
    at::Tensor k_depth_packed = at::empty(
        {batch, heads, seq_len, kDepth / 2},
        byte_options
    );
    // Per [batch, head]: [q multiplier, k multiplier, dQ factor, dK factor,
    // score factor, completion counter, bit-cast repeated E4M3 dequant-scale
    // word].  Words 2/3 temporarily accumulate Q/K sums of squares before
    // the last reduction block publishes the correction factors.
    at::Tensor adaptive_scales = at::empty(
        {batch, heads, 7},
        float_options
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUDACHECK(cudaMemsetAsync(
        adaptive_scales.data_ptr(),
        0,
        adaptive_scales.nbytes(),
        stream
    ));

    constexpr int kThreads = 256;
    const size_t pairs_per_head =
        static_cast<size_t>(seq_len) * (kDepth / 2);
    const int target_blocks_per_head = std::clamp(
        1024 / (batch * heads),
        8,
        256
    );
    const int reduction_blocks_per_head = static_cast<int>(std::min<size_t>(
        (pairs_per_head + kThreads - 1) / kThreads,
        static_cast<size_t>(target_blocks_per_head)
    ));
    const dim3 reduction_grid(reduction_blocks_per_head, heads, batch);
    reduce_adaptive_fp4_qk_scales_kernel<<<
        reduction_grid,
        kThreads,
        0,
        stream
    >>>(
        reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
        reinterpret_cast<float *>(adaptive_scales.data_ptr()),
        seq_len,
        heads,
        max_quant_scale,
        min_quant_scale,
        min_headroom,
        rms_clip_multiple,
        softmax_scale,
        ds_quant_scale
    );
    CUDACHECK(cudaGetLastError());

    const dim3 grid(seq_len / 32, heads, batch);
    quantize_fp4_dual_qk_tiled_kernel<192, false, false, true><<<
        grid,
        kThreads,
        0,
        stream
    >>>(
        reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
        reinterpret_cast<uint8_t *>(q_sequence_aligned.data_ptr()),
        reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
        reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr()),
        reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
        nullptr,
        nullptr,
        seq_len,
        heads,
        1.0f,
        1.0f,
        reinterpret_cast<const float *>(adaptive_scales.data_ptr())
    );
    CUDACHECK(cudaGetLastError());
    return {
        q_sequence_aligned,
        q_depth_packed,
        k_depth_aligned,
        k_depth_packed,
        adaptive_scales
    };
}

std::vector<at::Tensor> quantize_fp4_dual_qk_precomputed_scales(
    at::Tensor q,
    at::Tensor k,
    at::Tensor adaptive_scales
) {
    tkfa4::check_bshd(q, "q", at::kBFloat16);
    tkfa4::check_bshd(k, "k", at::kBFloat16);
    TORCH_CHECK(
        q.sizes() == k.sizes() && q.is_contiguous() && k.is_contiguous() &&
            q.is_cuda() && k.is_cuda(),
        "q and k must be identically shaped contiguous CUDA BF16 tensors"
    );
    TORCH_CHECK(
        q.size(1) % 32 == 0 && q.size(3) == 192,
        "precomputed-scale FP4 Q/K packing requires D192 and sequence "
        "length divisible by 32"
    );
    TORCH_CHECK(
            adaptive_scales.scalar_type() == at::ScalarType::Float &&
            adaptive_scales.is_cuda() && adaptive_scales.is_contiguous() &&
            adaptive_scales.dim() == 3 &&
            adaptive_scales.size(0) == q.size(0) &&
            adaptive_scales.size(1) == q.size(2) &&
            adaptive_scales.size(2) == 7,
        "adaptive_scales must be contiguous CUDA float32 [B, H, 7] with "
        "[q, k, dQ factor, dK factor, score factor, scratch, scale word]"
    );
    kittens::py::device_check(q, k, adaptive_scales);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const int batch = static_cast<int>(q.size(0));
    const int seq_len = static_cast<int>(q.size(1));
    const int heads = static_cast<int>(q.size(2));
    constexpr int kDepth = 192;
    auto byte_options = q.options().dtype(at::ScalarType::Byte);
    at::Tensor q_sequence_aligned = at::empty(
        {batch, heads, kDepth, seq_len},
        byte_options
    );
    at::Tensor q_depth_packed = at::empty(
        {batch, heads, seq_len, kDepth / 2},
        byte_options
    );
    at::Tensor k_depth_aligned = at::empty(q.sizes(), byte_options);
    at::Tensor k_depth_packed = at::empty(
        {batch, heads, seq_len, kDepth / 2},
        byte_options
    );
    constexpr int kThreads = 256;
    const dim3 grid(seq_len / 32, heads, batch);
    quantize_fp4_dual_qk_tiled_kernel<192, false, false, true><<<
        grid,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const kittens::bf16 *>(q.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(k.data_ptr()),
        reinterpret_cast<uint8_t *>(q_sequence_aligned.data_ptr()),
        reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
        reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr()),
        reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
        nullptr,
        nullptr,
        seq_len,
        heads,
        1.0f,
        1.0f,
        reinterpret_cast<const float *>(adaptive_scales.data_ptr())
    );
    CUDACHECK(cudaGetLastError());
    return {
        q_sequence_aligned,
        q_depth_packed,
        k_depth_aligned,
        k_depth_packed,
        adaptive_scales
    };
}

__global__ void multiply_nvfp4_global_scale_kernel(
    nvfp4_quantize::globals globals,
    float value_scale
) {
    globals.A_sc_global.raw_ptr[0] *= value_scale;
}

// Learned projection weights need the same true-2D 16x16 quantization in
// both GEMM orientations.  Preparing W and W^T independently is exact but
// needlessly rereads the BF16 parameter, materializes W^T, and launches the
// complete quantizer twice.  This producer computes each FP4 code and E4M3
// block scale once, then publishes both tcgen05-ready physical layouts.
//
// The Q/K/V variant additionally reads the canonical PyTorch parameters
// directly.  A D128 Q/K tile is converted from split-half rotary rows to the
// adjacent-pair physical order in shared memory; V and ordinary projection
// weights retain identity row order.  Since every source region and every
// physical boundary is 128-row aligned, one TMA tile never crosses a source.
struct nvfp4_dual_weight_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = 4;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
};

struct nvfp4_dual_weight_globals {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int K_BLOCK_SIZE = 16;

    using bf16_tile = kittens::st_bf<TILE_M, TILE_N, false>;
    using fp4_tile = kittens::st_fp4e2m1_2<TILE_M, TILE_N / 2, false>;
    using scale_vec = kittens::sv_hf<256>;
    using bf16_gl = kittens::gl<
        kittens::bf16,
        1,
        1,
        -1,
        -1,
        bf16_tile
    >;
    using fp4_gl = kittens::gl<
        kittens::fp4e2m1_2,
        1,
        1,
        -1,
        -1,
        fp4_tile
    >;
    using scale_gl = kittens::gl<
        kittens::half,
        1,
        -1,
        -1,
        256,
        scale_vec
    >;
    using global_scale_gl = kittens::gl<float, 1, 1, 1, 1>;

    bf16_gl q_bf16;
    bf16_gl k_bf16;
    bf16_gl v_bf16;
    fp4_gl forward_fp4;
    scale_gl forward_scales;
    fp4_gl backward_fp4;
    scale_gl backward_scales;
    global_scale_gl global_scale;
    int q_tiles;
    int k_tiles;
    int v_tiles;
    bool pair_interleave_qk;

    __host__ inline dim3 grid() const {
        return dim3(
            q_bf16.cols() / TILE_N,
            q_tiles + k_tiles + v_tiles
        );
    }

    __host__ inline int dynamic_shared_memory() const {
        // tma_swizzle_allocator aligns every distinct allocation to 1024
        // bytes.  The shared tile wrappers also carry type-level alignment,
        // so summing sizeof(...) undercounts the inter-object padding.  Keep
        // an explicit audited ceiling: memcheck observes the final scale page
        // ending below 53 KiB.  Keep 55 KiB to leave alignment headroom while
        // preserving four resident CTAs: the compiled kernel also uses 1040 B
        // static shared memory, so 4 * (55 KiB + 1040 B) = 229440 B, below
        // GB200's 233472 B shared-memory budget.
        return 55 * 1024;
    }
};

__device__ __forceinline__ int nvfp4_scale_swizzle_byte(
    int logical_row,
    int block_in_64
) {
    return (logical_row % 32) * 16 + (logical_row / 32) * 4 +
        block_in_64;
}

__device__ __forceinline__ uint32_t nvfp4_fix_e4m3_nan_word(
    uint32_t packed
) {
    // This is exactly the byte transform previously applied by the two
    // post-pack fp8_nan_fixup_kernel launches. Apply it only when scale bytes
    // are published: payload quantization still observes the original E4M3
    // conversion, preserving the established dual-pack representation even
    // for a saturated local scale.
    uint32_t fixed = 0;
    #pragma unroll
    for (int byte_index = 0; byte_index < 4; ++byte_index) {
        const int shift = byte_index * 8;
        uint32_t value = (packed >> shift) & 0xffu;
        if ((value & 0x7fu) == 0x7fu) {
            value = (value & 0x80u) | 0x7eu;
        }
        fixed |= value << shift;
    }
    return fixed;
}

__device__ inline void quantize_nvfp4_dual_weight_kernel(
    const nvfp4_dual_weight_globals &G
) {
    extern __shared__ int __shm[];
    kittens::tma_swizzle_allocator allocator(&__shm[0]);
    auto &input_smem =
        allocator.allocate<nvfp4_dual_weight_globals::bf16_tile>();
    auto &forward_smem =
        allocator.allocate<nvfp4_dual_weight_globals::fp4_tile>();
    auto &backward_smem =
        allocator.allocate<nvfp4_dual_weight_globals::fp4_tile>();
    auto (&forward_scale_smem)[2] = allocator.allocate<
        nvfp4_dual_weight_globals::scale_vec,
        2
    >();
    auto (&backward_scale_smem)[2] = allocator.allocate<
        nvfp4_dual_weight_globals::scale_vec,
        2
    >();

    const int tid = static_cast<int>(threadIdx.x);
    const int physical_row_tile = static_cast<int>(blockIdx.y);
    const int column_tile = static_cast<int>(blockIdx.x);
    const int k_region_begin = G.q_tiles;
    const int v_region_begin = G.q_tiles + G.k_tiles;
    int source_row_tile = physical_row_tile;
    bool interleave_rows = false;
    if (physical_row_tile < k_region_begin) {
        source_row_tile = physical_row_tile;
        interleave_rows = G.pair_interleave_qk;
    } else if (physical_row_tile < v_region_begin) {
        source_row_tile = physical_row_tile - k_region_begin;
        interleave_rows = G.pair_interleave_qk;
    } else {
        source_row_tile = physical_row_tile - v_region_begin;
    }

    __shared__ kittens::semaphore inputs_arrived;
    if (tid == 0) {
        kittens::init_semaphore(inputs_arrived, 0, 1);
        kittens::tma::expect(inputs_arrived, input_smem);
        if (physical_row_tile < k_region_begin) {
            kittens::tma::load_async(
                input_smem,
                G.q_bf16,
                {source_row_tile, column_tile},
                inputs_arrived
            );
        } else if (physical_row_tile < v_region_begin) {
            kittens::tma::load_async(
                input_smem,
                G.k_bf16,
                {source_row_tile, column_tile},
                inputs_arrived
            );
        } else {
            kittens::tma::load_async(
                input_smem,
                G.v_bf16,
                {source_row_tile, column_tile},
                inputs_arrived
            );
        }
    }

    const float global_decode = G.global_scale[{0}];
    const float global_encode =
        1.0f / fmaxf(global_decode, 0.000000000001f);
    constexpr int kBlocksPerHalf =
        nvfp4_dual_weight_globals::TILE_N /
        nvfp4_dual_weight_globals::K_BLOCK_SIZE / 2;
    constexpr int kPairsPerBlock =
        nvfp4_dual_weight_globals::K_BLOCK_SIZE / 2;
    kittens::bf16_2 values[2][kBlocksPerHalf][kPairsPerBlock];
    kittens::fp8e4m3 block_scales[2][kBlocksPerHalf];

    __syncthreads();
    kittens::wait(inputs_arrived, 0);

    // The physical row is the CUDA thread.  For D128 Q/K, adjacent physical
    // rows select corresponding elements from the canonical low/high halves.
    const int source_row = interleave_rows
        ? tid / 2 + (tid & 1) * 64
        : tid;
    #pragma unroll
    for (int column_half = 0; column_half < 2; ++column_half) {
        #pragma unroll
        for (int i = 0; i < kBlocksPerHalf; ++i) {
            const int block =
                (i + tid / 8) % kBlocksPerHalf +
                column_half * kBlocksPerHalf;
            #pragma unroll
            for (int j = 0; j < kPairsPerBlock; ++j) {
                const int column =
                    block * nvfp4_dual_weight_globals::K_BLOCK_SIZE +
                    ((tid + j) * 2) %
                        nvfp4_dual_weight_globals::K_BLOCK_SIZE;
                const int byte_offset =
                    (source_row * nvfp4_dual_weight_globals::TILE_N +
                     column) * sizeof(kittens::bf16);
                kittens::move<kittens::bf16_2>::lds(
                    values[column_half][i][j],
                    static_cast<uint32_t>(
                        __cvta_generic_to_shared(&input_smem)
                    ) + byte_offset
                );
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int column_half = 0; column_half < 2; ++column_half) {
        float maxima[kBlocksPerHalf];
        #pragma unroll
        for (int i = 0; i < kBlocksPerHalf; ++i) {
            const int block = (i + tid / 8) % kBlocksPerHalf;
            kittens::bf16_2 maximum =
                __habs2(values[column_half][i][0]);
            #pragma unroll
            for (int j = 1; j < kPairsPerBlock; ++j) {
                maximum = __hmax2(
                    maximum,
                    __habs2(values[column_half][i][j])
                );
            }
            maxima[block] = __bfloat162float(
                __hmax(maximum.x, maximum.y)
            );
        }
        // Match quantize_kernel<true>: physical 16-row groups share exactly
        // one E4M3 scale for every 16-column block.
        #pragma unroll
        for (int mask = 8; mask >= 1; mask >>= 1) {
            #pragma unroll
            for (int i = 0; i < kBlocksPerHalf; ++i) {
                maxima[i] = fmaxf(
                    maxima[i],
                    __shfl_xor_sync(0xffffffffu, maxima[i], mask)
                );
            }
        }
        #pragma unroll
        for (int i = 0; i < kBlocksPerHalf; ++i) {
            block_scales[column_half][i] = __nv_fp8_e4m3(
                maxima[i] / 6.0f * global_encode
            );
        }

        #pragma unroll
        for (int i = 0; i < kBlocksPerHalf; ++i) {
            const int block = (i + tid / 8) % kBlocksPerHalf;
            const float local_decode = static_cast<float>(
                block_scales[column_half][block]
            );
            const float encode = 1.0f / fmaxf(
                local_decode * global_decode,
                0.000000000001f
            );
            const int base =
                tid * nvfp4_dual_weight_globals::TILE_N / 2 +
                (block + column_half * kBlocksPerHalf) *
                    nvfp4_dual_weight_globals::K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int j = 0; j < kPairsPerBlock; ++j) {
                const int offset = base + ((tid + j) & 7);
                const float2 scaled{
                    __bfloat162float(values[column_half][i][j].x) * encode,
                    __bfloat162float(values[column_half][i][j].y) * encode,
                };
                const uint8_t packed = static_cast<uint8_t>(
                    __nv_cvt_float2_to_fp4x2(
                        scaled,
                        __NV_E2M1,
                        cudaRoundNearest
                    )
                );
                asm volatile(
                    "{st.shared.b8 [%0], %1;}"
                    :: "r"(
                        static_cast<uint32_t>(
                            __cvta_generic_to_shared(&forward_smem)
                        ) + offset
                    ),
                    "r"(static_cast<uint32_t>(packed))
                );
            }
        }
    }

    const int forward_scale_offset = nvfp4_scale_swizzle_byte(tid, 0);
    const uint32_t forward_scale_word0 = nvfp4_fix_e4m3_nan_word(
        *reinterpret_cast<uint32_t *>(&block_scales[0][0])
    );
    const uint32_t forward_scale_word1 = nvfp4_fix_e4m3_nan_word(
        *reinterpret_cast<uint32_t *>(&block_scales[1][0])
    );
    asm volatile(
        "{st.shared.b32 [%0], %1;}"
        :: "r"(
            static_cast<uint32_t>(
                __cvta_generic_to_shared(&forward_scale_smem[0])
            ) + forward_scale_offset
        ),
        "r"(forward_scale_word0)
    );
    asm volatile(
        "{st.shared.b32 [%0], %1;}"
        :: "r"(
            static_cast<uint32_t>(
                __cvta_generic_to_shared(&forward_scale_smem[1])
            ) + forward_scale_offset
        ),
        "r"(forward_scale_word1)
    );
    __syncthreads();

    // Transpose the already-rounded 4-bit codes.  Reusing nibbles, rather
    // than requantizing register values, makes the two publications exactly
    // transpose-consistent by construction.
    auto *forward_bytes = reinterpret_cast<const uint8_t *>(&forward_smem);
    auto *backward_bytes = reinterpret_cast<uint8_t *>(&backward_smem);
    #pragma unroll
    for (int index = tid;
         index < nvfp4_dual_weight_globals::TILE_M *
             nvfp4_dual_weight_globals::TILE_N / 2;
         index += nvfp4_dual_weight_config::NUM_THREADS) {
        const int transposed_row =
            index / (nvfp4_dual_weight_globals::TILE_M / 2);
        const int physical_row_pair =
            index % (nvfp4_dual_weight_globals::TILE_M / 2);
        const int source_byte_column = transposed_row / 2;
        const int source_shift = (transposed_row & 1) * 4;
        const uint8_t first = static_cast<uint8_t>(
            (forward_bytes[
                (physical_row_pair * 2) *
                    (nvfp4_dual_weight_globals::TILE_N / 2) +
                source_byte_column
            ] >> source_shift) & 0x0fu
        );
        const uint8_t second = static_cast<uint8_t>(
            (forward_bytes[
                (physical_row_pair * 2 + 1) *
                    (nvfp4_dual_weight_globals::TILE_N / 2) +
                source_byte_column
            ] >> source_shift) & 0x0fu
        );
        backward_bytes[index] = static_cast<uint8_t>(first | (second << 4));
    }

    // Transpose the 8x8 grid of shared 16x16 scales and reproduce the exact
    // NVIDIA 1x16 metadata swizzle in each orientation.
    const int source_block = tid / 16;
    const int source_half = source_block / 4;
    const int source_slot = source_block % 4;
    const auto *source_scale_bytes = reinterpret_cast<const uint8_t *>(
        &forward_scale_smem[source_half]
    );
    uint32_t low_groups = 0;
    uint32_t high_groups = 0;
    #pragma unroll
    for (int group = 0; group < 8; ++group) {
        const int representative_row = group * 16;
        const int source_offset = nvfp4_scale_swizzle_byte(
            representative_row,
            source_slot
        );
        const uint32_t scale = source_scale_bytes[source_offset];
        if (group < 4) {
            low_groups |= scale << (group * 8);
        } else {
            high_groups |= scale << ((group - 4) * 8);
        }
    }
    const int backward_scale_offset = nvfp4_scale_swizzle_byte(tid, 0);
    *reinterpret_cast<uint32_t *>(
        reinterpret_cast<uint8_t *>(&backward_scale_smem[0]) +
        backward_scale_offset
    ) = low_groups;
    *reinterpret_cast<uint32_t *>(
        reinterpret_cast<uint8_t *>(&backward_scale_smem[1]) +
        backward_scale_offset
    ) = high_groups;
    __syncthreads();

    if (tid == 0) {
        kittens::tma::store_async(
            G.forward_fp4,
            forward_smem,
            {physical_row_tile, column_tile}
        );
        kittens::tma::store_async(
            G.forward_scales,
            forward_scale_smem[0],
            {physical_row_tile, column_tile * 2, 0}
        );
        kittens::tma::store_async(
            G.forward_scales,
            forward_scale_smem[1],
            {physical_row_tile, column_tile * 2 + 1, 0}
        );
        kittens::tma::store_async(
            G.backward_fp4,
            backward_smem,
            {column_tile, physical_row_tile}
        );
        kittens::tma::store_async(
            G.backward_scales,
            backward_scale_smem[0],
            {column_tile, physical_row_tile * 2, 0}
        );
        kittens::tma::store_async(
            G.backward_scales,
            backward_scale_smem[1],
            {column_tile, physical_row_tile * 2 + 1, 0}
        );
        kittens::tma::store_async_wait();
    }
}

void check_nvfp4_dual_weight_outputs(
    const at::Tensor &input,
    const at::Tensor &forward_packed,
    const at::Tensor &forward_scales,
    const at::Tensor &backward_packed,
    const at::Tensor &backward_scales,
    const at::Tensor &global_scale,
    int64_t output_rows = -1
) {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16 &&
            input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
            input.size(0) > 0 && input.size(1) > 0 &&
            input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
        "dual NVFP4 projection-weight preparation requires contiguous CUDA "
        "BF16 [M,K] with both dimensions divisible by 128"
    );
    const int64_t rows = output_rows < 0 ? input.size(0) : output_rows;
    const int64_t columns = input.size(1);
    constexpr int64_t kMaxCudaGridY = 65535;
    TORCH_CHECK(
        rows > 0 && rows % 128 == 0 &&
            rows / 128 <= kMaxCudaGridY &&
            rows <= std::numeric_limits<int>::max() &&
            columns <= std::numeric_limits<int>::max(),
        "dual NVFP4 projection-weight dimensions exceed the CUDA grid or "
        "kernel index range"
    );
    TORCH_CHECK(
        forward_packed.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
            forward_packed.is_cuda() && forward_packed.is_contiguous() &&
            forward_packed.sizes() == at::IntArrayRef({rows, columns / 2}),
        "forward_packed must be caller-owned contiguous CUDA FP4x2 [M,K/2]"
    );
    TORCH_CHECK(
        forward_scales.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            forward_scales.is_cuda() && forward_scales.is_contiguous() &&
            forward_scales.sizes() == at::IntArrayRef(
                {rows / 128, columns / 64, 512}
            ),
        "forward_scales must be caller-owned contiguous CUDA E4M3 "
        "[M/128,K/64,512]"
    );
    TORCH_CHECK(
        backward_packed.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
            backward_packed.is_cuda() && backward_packed.is_contiguous() &&
            backward_packed.sizes() == at::IntArrayRef(
                {columns, rows / 2}
            ),
        "backward_packed must be caller-owned contiguous CUDA FP4x2 [K,M/2]"
    );
    TORCH_CHECK(
        backward_scales.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            backward_scales.is_cuda() && backward_scales.is_contiguous() &&
            backward_scales.sizes() == at::IntArrayRef(
                {columns / 128, rows / 64, 512}
            ),
        "backward_scales must be caller-owned contiguous CUDA E4M3 "
        "[K/128,M/64,512]"
    );
    TORCH_CHECK(
        global_scale.scalar_type() == at::ScalarType::Float &&
            global_scale.is_cuda() && global_scale.is_contiguous() &&
            global_scale.numel() == 1,
        "global_scale must be one caller-owned contiguous CUDA float32 value"
    );
    kittens::py::device_check(
        input,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale
    );

    using named_tensor = std::pair<const char *, const at::Tensor *>;
    const std::initializer_list<named_tensor> outputs{
        {"forward_packed", &forward_packed},
        {"forward_scales", &forward_scales},
        {"backward_packed", &backward_packed},
        {"backward_scales", &backward_scales},
        {"global_scale", &global_scale},
    };
    const auto byte_ranges_overlap = [](
        const at::Tensor &left,
        const at::Tensor &right
    ) {
        const auto left_begin = reinterpret_cast<uintptr_t>(left.data_ptr());
        const auto right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
        const auto left_bytes = static_cast<uintptr_t>(
            left.numel() * left.element_size()
        );
        const auto right_bytes = static_cast<uintptr_t>(
            right.numel() * right.element_size()
        );
        return left_begin <= right_begin
            ? right_begin - left_begin < left_bytes
            : left_begin - right_begin < right_bytes;
    };
    constexpr uintptr_t kTmaAlignment = alignof(uint4);
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % kTmaAlignment == 0,
        "dual NVFP4 projection-weight input must have a 16-byte-aligned base"
    );
    for (auto left = outputs.begin(); left != outputs.end(); ++left) {
        const uintptr_t required_alignment =
            left->second == &global_scale ? alignof(float) : kTmaAlignment;
        TORCH_CHECK(
            reinterpret_cast<uintptr_t>(left->second->data_ptr()) %
                    required_alignment ==
                0,
            left->first,
            " must have a suitably aligned base for dual NVFP4 publication"
        );
        for (auto right = left + 1; right != outputs.end(); ++right) {
            TORCH_CHECK(
                !byte_ranges_overlap(*left->second, *right->second),
                left->first, " and ", right->first,
                " must use disjoint storage"
            );
        }
        TORCH_CHECK(
            !byte_ranges_overlap(input, *left->second),
            "input and ", left->first, " must use disjoint storage"
        );
    }
}

void check_nvfp4_dual_outputs_disjoint_from_input(
    const char *input_name,
    const at::Tensor &input,
    const at::Tensor &forward_packed,
    const at::Tensor &forward_scales,
    const at::Tensor &backward_packed,
    const at::Tensor &backward_scales,
    const at::Tensor &global_scale
) {
    const auto input_begin = reinterpret_cast<uintptr_t>(input.data_ptr());
    const auto input_bytes = static_cast<uintptr_t>(
        input.numel() * input.element_size()
    );
    TORCH_CHECK(
        input_begin % alignof(uint4) == 0,
        input_name,
        " must have a 16-byte-aligned base for dual NVFP4 preparation"
    );
    for (const auto &output : {
             std::pair<const char *, const at::Tensor *>(
                 "forward_packed", &forward_packed
             ),
             std::pair<const char *, const at::Tensor *>(
                 "forward_scales", &forward_scales
             ),
             std::pair<const char *, const at::Tensor *>(
                 "backward_packed", &backward_packed
             ),
             std::pair<const char *, const at::Tensor *>(
                 "backward_scales", &backward_scales
             ),
             std::pair<const char *, const at::Tensor *>(
                 "global_scale", &global_scale
             )}) {
        const auto output_begin = reinterpret_cast<uintptr_t>(
            output.second->data_ptr()
        );
        const auto output_bytes = static_cast<uintptr_t>(
            output.second->numel() * output.second->element_size()
        );
        TORCH_CHECK(
            !(input_begin <= output_begin
                  ? output_begin - input_begin < input_bytes
                  : input_begin - output_begin < output_bytes),
            input_name, " and ", output.first,
            " must use disjoint storage"
        );
    }
}

nvfp4_quantize::globals make_nvfp4_absmax_globals(
    const at::Tensor &input,
    const at::Tensor &packed,
    const at::Tensor &scales,
    const at::Tensor &global_scale
) {
    using G = nvfp4_quantize::globals;
    return G{
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(input),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(packed),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl, false>(
            scales,
            1,
            scales.size(0),
            scales.size(1),
            256
        ),
        .A_sc_global =
            kittens::py::tensor_to_gl<G::A_sc_global_gl>(global_scale),
    };
}

void launch_nvfp4_dual_weight_quantization(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    at::Tensor forward_packed,
    at::Tensor forward_scales,
    at::Tensor backward_packed,
    at::Tensor backward_scales,
    at::Tensor global_scale,
    int q_tiles,
    int k_tiles,
    int v_tiles,
    bool pair_interleave_qk
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto q_absmax = make_nvfp4_absmax_globals(
        q,
        forward_packed,
        forward_scales,
        global_scale
    );
    nvfp4_quantize::zero_kernel<<<1, 1, 0, stream>>>(q_absmax);
    nvfp4_quantize::absmax_kernel<<<
        nvfp4_quantize::absmax_config::NUM_BLOCKS,
        nvfp4_quantize::absmax_config::NUM_THREADS,
        0,
        stream
    >>>(q_absmax);
    if (k_tiles != 0) {
        auto k_absmax = make_nvfp4_absmax_globals(
            k,
            forward_packed,
            forward_scales,
            global_scale
        );
        nvfp4_quantize::absmax_kernel<<<
            nvfp4_quantize::absmax_config::NUM_BLOCKS,
            nvfp4_quantize::absmax_config::NUM_THREADS,
            0,
            stream
        >>>(k_absmax);
    }
    if (v_tiles != 0) {
        auto v_absmax = make_nvfp4_absmax_globals(
            v,
            forward_packed,
            forward_scales,
            global_scale
        );
        nvfp4_quantize::absmax_kernel<<<
            nvfp4_quantize::absmax_config::NUM_BLOCKS,
            nvfp4_quantize::absmax_config::NUM_THREADS,
            0,
            stream
        >>>(v_absmax);
    }
    nvfp4_quantize::divide_kernel<<<1, 1, 0, stream>>>(q_absmax);

    using G = nvfp4_dual_weight_globals;
    G globals{
        .q_bf16 = kittens::py::tensor_to_gl<G::bf16_gl>(q),
        .k_bf16 = kittens::py::tensor_to_gl<G::bf16_gl>(k),
        .v_bf16 = kittens::py::tensor_to_gl<G::bf16_gl>(v),
        .forward_fp4 =
            kittens::py::tensor_to_gl<G::fp4_gl>(forward_packed),
        .forward_scales = kittens::py::tensor_to_gl<G::scale_gl, false>(
            forward_scales,
            1,
            forward_scales.size(0),
            forward_scales.size(1),
            256
        ),
        .backward_fp4 =
            kittens::py::tensor_to_gl<G::fp4_gl>(backward_packed),
        .backward_scales = kittens::py::tensor_to_gl<G::scale_gl, false>(
            backward_scales,
            1,
            backward_scales.size(0),
            backward_scales.size(1),
            256
        ),
        .global_scale =
            kittens::py::tensor_to_gl<G::global_scale_gl>(global_scale),
        .q_tiles = q_tiles,
        .k_tiles = k_tiles,
        .v_tiles = v_tiles,
        .pair_interleave_qk = pair_interleave_qk,
    };
    kittens::py::launch_kernel<
        nvfp4_dual_weight_config,
        G,
        quantize_nvfp4_dual_weight_kernel
    >(globals);

    CUDACHECK(cudaGetLastError());
}

void quantize_nvfp4_projection_weight_dual_out_impl(
    at::Tensor input,
    at::Tensor forward_packed,
    at::Tensor forward_scales,
    at::Tensor backward_packed,
    at::Tensor backward_scales,
    at::Tensor global_scale,
    bool checked
) {
    if (checked) {
        check_nvfp4_dual_weight_outputs(
            input,
            forward_packed,
            forward_scales,
            backward_packed,
            backward_scales,
            global_scale
        );
    }
    launch_nvfp4_dual_weight_quantization(
        input,
        input,
        input,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale,
        static_cast<int>(input.size(0) / 128),
        0,
        0,
        false
    );
}

void quantize_nvfp4_projection_weight_dual_out(
    at::Tensor input,
    at::Tensor forward_packed,
    at::Tensor forward_scales,
    at::Tensor backward_packed,
    at::Tensor backward_scales,
    at::Tensor global_scale
) {
    quantize_nvfp4_projection_weight_dual_out_impl(
        input,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale,
        true
    );
}

void quantize_nvfp4_projection_weight_dual_out_unchecked(
    at::Tensor input,
    at::Tensor forward_packed,
    at::Tensor forward_scales,
    at::Tensor backward_packed,
    at::Tensor backward_scales,
    at::Tensor global_scale
) {
    quantize_nvfp4_projection_weight_dual_out_impl(
        input,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale,
        false
    );
}

void quantize_gqa_d128_qkv_projection_weight_dual_out_impl(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor forward_packed,
    at::Tensor forward_scales,
    at::Tensor backward_packed,
    at::Tensor backward_scales,
    at::Tensor global_scale,
    bool checked
) {
    if (checked) {
        TORCH_CHECK(
            q.scalar_type() == at::ScalarType::BFloat16 && q.is_cuda() &&
                q.is_contiguous() && q.dim() == 2 &&
                q.size(0) > 0 && q.size(1) > 0 &&
                q.size(0) % 128 == 0 && q.size(1) % 128 == 0,
            "D128 Q weight must be contiguous CUDA BF16 [Hq*128,K]"
        );
        for (const auto &entry : {
                 std::pair<const char *, const at::Tensor *>("K", &k),
                 std::pair<const char *, const at::Tensor *>("V", &v)}) {
            TORCH_CHECK(
                entry.second->scalar_type() == at::ScalarType::BFloat16 &&
                    entry.second->is_cuda() &&
                    entry.second->is_contiguous() &&
                    entry.second->dim() == 2 &&
                    entry.second->size(0) > 0 &&
                    entry.second->size(1) > 0 &&
                    entry.second->size(0) % 128 == 0 &&
                    entry.second->size(1) == q.size(1),
                "D128 ", entry.first,
                " weight must be contiguous CUDA BF16 [Hkv*128,K]"
            );
        }
        TORCH_CHECK(
            k.size(0) == v.size(0) && q.size(0) % k.size(0) == 0,
            "D128 Q/K/V weights require equal K/V heads and integral GQA"
        );
        TORCH_CHECK(
            q.size(0) <= std::numeric_limits<int>::max() &&
                k.size(0) <= std::numeric_limits<int>::max() &&
                v.size(0) <= std::numeric_limits<int>::max() &&
                q.size(1) <= std::numeric_limits<int>::max(),
            "D128 Q/K/V weight dimensions exceed the kernel index range"
        );
        const int64_t total_rows = q.size(0) + k.size(0) + v.size(0);
        check_nvfp4_dual_weight_outputs(
            q,
            forward_packed,
            forward_scales,
            backward_packed,
            backward_scales,
            global_scale,
            total_rows
        );
        kittens::py::device_check(q, k, v);
        check_nvfp4_dual_outputs_disjoint_from_input(
            "K weight",
            k,
            forward_packed,
            forward_scales,
            backward_packed,
            backward_scales,
            global_scale
        );
        check_nvfp4_dual_outputs_disjoint_from_input(
            "V weight",
            v,
            forward_packed,
            forward_scales,
            backward_packed,
            backward_scales,
            global_scale
        );
    }
    launch_nvfp4_dual_weight_quantization(
        q,
        k,
        v,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale,
        static_cast<int>(q.size(0) / 128),
        static_cast<int>(k.size(0) / 128),
        static_cast<int>(v.size(0) / 128),
        true
    );
}

void quantize_gqa_d128_qkv_projection_weight_dual_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor forward_packed,
    at::Tensor forward_scales,
    at::Tensor backward_packed,
    at::Tensor backward_scales,
    at::Tensor global_scale
) {
    quantize_gqa_d128_qkv_projection_weight_dual_out_impl(
        q,
        k,
        v,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale,
        true
    );
}

void quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor forward_packed,
    at::Tensor forward_scales,
    at::Tensor backward_packed,
    at::Tensor backward_scales,
    at::Tensor global_scale
) {
    quantize_gqa_d128_qkv_projection_weight_dual_out_impl(
        q,
        k,
        v,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale,
        false
    );
}

std::vector<at::Tensor> quantize_e4m3_projection_rows_impl(
    at::Tensor input
) {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16 &&
            input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
            input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
        "E4M3 projection preparation requires contiguous CUDA BF16 [M, K] "
        "with both dimensions divisible by 128"
    );
    TORCH_CHECK(
        input.size(0) <= std::numeric_limits<int>::max() &&
            input.size(1) <= std::numeric_limits<int>::max(),
        "E4M3 projection preparation dimensions exceed the CUDA kernel "
        "index range"
    );
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int columns = static_cast<int>(input.size(1));
    at::Tensor payload = at::empty(
        input.sizes(),
        input.options().dtype(at::kFloat8_e4m3fn)
    );
    at::Tensor decode = at::empty(
        {rows},
        input.options().dtype(at::kFloat)
    );
    constexpr int kThreads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (columns == 2048) {
        prepare_e4m3_rows_k2048_kernel<<<rows, kThreads, 0, stream>>>(
            reinterpret_cast<const kittens::bf16 *>(input.data_ptr()),
            reinterpret_cast<uint32_t *>(payload.data_ptr()),
            reinterpret_cast<float *>(decode.data_ptr()),
            rows
        );
    } else {
        prepare_e4m3_rows_generic_kernel<<<rows, kThreads, 0, stream>>>(
            reinterpret_cast<const kittens::bf16 *>(input.data_ptr()),
            reinterpret_cast<uint32_t *>(payload.data_ptr()),
            reinterpret_cast<float *>(decode.data_ptr()),
            rows,
            columns
        );
    }
    CUDACHECK(cudaGetLastError());
    return {payload, decode};
}

std::vector<at::Tensor> quantize_e4m3_projection_operand(at::Tensor input) {
    return quantize_e4m3_projection_rows_impl(input);
}

std::vector<at::Tensor> quantize_e4m3_projection_weight(at::Tensor input) {
    // PyTorch linear weights are [N,K], so a channelwise weight scale is the
    // same row reduction and preserves the existing row-major payload layout.
    return quantize_e4m3_projection_rows_impl(input);
}

void check_e4m3_x4_v_to_causal_mxfp4_contract(
    const at::Tensor &input,
    const at::Tensor *payload,
    const at::Tensor *scales
) {
    constexpr int64_t kMaxCudaGridYz = 65535;
    constexpr std::uintptr_t kRequiredAlignment = alignof(uint32_t);
    const auto is_aligned = [](const at::Tensor &tensor) {
        return reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) %
            kRequiredAlignment == 0;
    };
    const auto ranges_overlap = [](
        const at::Tensor &first,
        const at::Tensor &second
    ) {
        const auto first_begin = reinterpret_cast<std::uintptr_t>(
            first.data_ptr()
        );
        const auto second_begin = reinterpret_cast<std::uintptr_t>(
            second.data_ptr()
        );
        const auto first_bytes = static_cast<std::uintptr_t>(
            first.numel() * first.element_size()
        );
        const auto second_bytes = static_cast<std::uintptr_t>(
            second.numel() * second.element_size()
        );
        return first_begin <= second_begin
            ? second_begin - first_begin < first_bytes
            : first_begin - second_begin < second_bytes;
    };
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() && input.dim() == 4 &&
            input.scalar_type() == at::kFloat8_e4m3fn,
        "E4M3(x4) V conversion requires contiguous CUDA E4M3 "
        "[B,H,D,S] input"
    );
    TORCH_CHECK(
        input.size(0) > 0 && input.size(1) > 0 &&
            input.size(2) == tkfa4_e4m3_to_mxfp4_v::kHeadDepth &&
            input.size(3) > 0 &&
            input.size(3) % tkfa4_e4m3_to_mxfp4_v::kSequenceTile == 0,
        "E4M3(x4) V conversion requires B,H > 0, D=64, and positive S "
        "divisible by 128"
    );
    TORCH_CHECK(
        input.size(0) <= kMaxCudaGridYz &&
            input.size(1) <= kMaxCudaGridYz &&
            input.size(3) <= std::numeric_limits<int>::max(),
        "E4M3(x4) V conversion dimensions exceed the CUDA 3-D grid or "
        "kernel index range"
    );
    TORCH_CHECK(
        is_aligned(input),
        "E4M3(x4) V input base must be four-byte aligned"
    );
    if (payload == nullptr || scales == nullptr) {
        return;
    }
    const int64_t batch = input.size(0);
    const int64_t heads = input.size(1);
    const int64_t sequence = input.size(3);
    TORCH_CHECK(
        payload->is_cuda() && payload->is_contiguous() &&
            payload->device() == input.device() && payload->dim() == 4 &&
            payload->scalar_type() == at::kFloat4_e2m1fn_x2 &&
            payload->size(0) == batch && payload->size(1) == heads &&
            payload->size(2) == tkfa4_e4m3_to_mxfp4_v::kHeadDepth &&
            payload->size(3) == sequence / 2,
        "MXFP4 V output must be contiguous CUDA packed E2M1 "
        "[B,H,64,S/2] on the input device"
    );
    TORCH_CHECK(
        scales->is_cuda() && scales->is_contiguous() &&
            scales->device() == input.device() && scales->dim() == 4 &&
            scales->scalar_type() == at::kFloat8_e4m3fn &&
            scales->size(0) == batch &&
            scales->size(1) == sequence / 128 &&
            scales->size(2) == heads &&
            scales->size(3) ==
                tkfa4_e4m3_to_mxfp4_v::kScalePageBytes,
        "MXFP4 V scales must be a contiguous byte-sized E8M0 container "
        "[B,S/128,H,512] on the input device"
    );
    TORCH_CHECK(
        is_aligned(*payload) && is_aligned(*scales),
        "MXFP4 V payload and scale bases must be four-byte aligned"
    );
    TORCH_CHECK(
        !ranges_overlap(input, *payload) &&
            !ranges_overlap(input, *scales) &&
            !ranges_overlap(*payload, *scales),
        "E4M3 input, MXFP4 payload, and MXFP4 scales must occupy "
        "disjoint byte ranges"
    );
}

void convert_e4m3_x4_v_bhds_to_causal_mxfp4_out(
    at::Tensor input,
    at::Tensor payload,
    at::Tensor scales
) {
    check_e4m3_x4_v_to_causal_mxfp4_contract(input, &payload, &scales);
    const c10::cuda::CUDAGuard device_guard(input.device());
    tkfa4_e4m3_to_mxfp4_v::launch(
        reinterpret_cast<const uint8_t *>(input.data_ptr()),
        reinterpret_cast<uint8_t *>(payload.data_ptr()),
        reinterpret_cast<uint8_t *>(scales.data_ptr()),
        static_cast<int>(input.size(0)),
        static_cast<int>(input.size(1)),
        static_cast<int>(input.size(3)),
        at::cuda::getCurrentCUDAStream()
    );
    CUDACHECK(cudaGetLastError());
}

std::vector<at::Tensor> convert_e4m3_x4_v_bhds_to_causal_mxfp4(
    at::Tensor input
) {
    check_e4m3_x4_v_to_causal_mxfp4_contract(input, nullptr, nullptr);
    const int64_t batch = input.size(0);
    const int64_t heads = input.size(1);
    const int64_t sequence = input.size(3);
    at::Tensor payload = at::empty(
        {batch, heads, tkfa4_e4m3_to_mxfp4_v::kHeadDepth, sequence / 2},
        input.options().dtype(at::kFloat4_e2m1fn_x2)
    );
    // Only the D64-addressable half of each 512-byte tcgen05 scale page is
    // part of this contract; unused reserved bytes remain unspecified, as in
    // the direct paired-D64 projection publisher.
    at::Tensor scales = at::empty(
        {batch, sequence / 128, heads,
         tkfa4_e4m3_to_mxfp4_v::kScalePageBytes},
        input.options().dtype(at::kFloat8_e4m3fn)
    );
    convert_e4m3_x4_v_bhds_to_causal_mxfp4_out(
        input,
        payload,
        scales
    );
    return {payload, scales};
}

std::vector<at::Tensor> quantize_nvfp4_projection_operand_impl(
    at::Tensor input,
    double value_scale,
    bool scale_2d
) {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16 &&
            input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
            input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
        "NVFP4 projection quantization requires contiguous CUDA BF16 [M, K] "
        "with both dimensions divisible by 128"
    );
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    at::Tensor packed = at::empty(
        {rows, cols / 2},
        input.options().dtype(at::kFloat4_e2m1fn_x2)
    );
    at::Tensor scales = at::empty(
        {rows / 128, cols / 64, 512},
        input.options().dtype(at::kFloat8_e4m3fn)
    );
    at::Tensor global_scale = at::empty(
        {1},
        input.options().dtype(at::kFloat)
    );
    using C = nvfp4_quantize::quantize_config;
    using G = nvfp4_quantize::globals;
    G globals{
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(input),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(packed),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl, false>(
            scales,
            1,
            scales.size(0),
            scales.size(1),
            256
        ),
        .A_sc_global =
            kittens::py::tensor_to_gl<G::A_sc_global_gl>(global_scale),
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvfp4_quantize::zero_kernel<<<1, 1, 0, stream>>>(globals);
    nvfp4_quantize::absmax_kernel<<<
        nvfp4_quantize::absmax_config::NUM_BLOCKS,
        nvfp4_quantize::absmax_config::NUM_THREADS,
        0,
        stream
    >>>(globals);
    nvfp4_quantize::divide_kernel<<<1, 1, 0, stream>>>(globals);
    if (scale_2d) {
        kittens::py::launch_kernel<
            C,
            G,
            nvfp4_quantize::quantize_kernel<true>
        >(globals);
    } else {
        kittens::py::launch_kernel<
            C,
            G,
            nvfp4_quantize::quantize_kernel<false>
        >(globals);
    }
    if (value_scale != 1.0) {
        multiply_nvfp4_global_scale_kernel<<<1, 1, 0, stream>>>(
            globals,
            static_cast<float>(value_scale)
        );
    }
    const int threads = 256;
    const int blocks = static_cast<int>(
        ((scales.numel() / 4) + threads - 1) / threads
    );
    fp8_nan_fixup_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint8_t *>(scales.data_ptr()),
        scales.numel()
    );
    CUDACHECK(cudaGetLastError());
    return {packed, scales, global_scale};
}

std::vector<at::Tensor> quantize_nvfp4_projection_operand(at::Tensor input) {
    return quantize_nvfp4_projection_operand_impl(input, 1.0, false);
}

std::vector<at::Tensor> quantize_nvfp4_projection_operand_rmsnorm(
    at::Tensor input,
    at::Tensor gamma,
    double epsilon
) {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16 &&
            input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
            input.size(0) > 0 && input.size(0) % 128 == 0 &&
            input.size(1) ==
                rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_COLUMNS,
        "fused RMSNorm NVFP4 preparation requires contiguous CUDA BF16 "
        "[M, 2048] with positive M divisible by 128"
    );
    TORCH_CHECK(
        gamma.scalar_type() == at::ScalarType::BFloat16 &&
            gamma.is_cuda() && gamma.is_contiguous() && gamma.dim() == 1 &&
            gamma.size(0) == input.size(1) && gamma.device() == input.device(),
        "fused RMSNorm NVFP4 preparation requires contiguous CUDA BF16 "
        "gamma [K] on the input device"
    );
    TORCH_CHECK(
        std::isfinite(epsilon) && epsilon > 0.0,
        "fused RMSNorm NVFP4 epsilon must be finite and positive"
    );

    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int columns = static_cast<int>(input.size(1));
    at::Tensor normalized = at::empty_like(input);
    at::Tensor inv_rms = at::empty(
        {rows},
        input.options().dtype(at::kFloat)
    );
    at::Tensor packed = at::empty(
        {rows, columns / 2},
        input.options().dtype(at::kFloat4_e2m1fn_x2)
    );
    at::Tensor scales = at::empty(
        {rows / 128, columns / 64, 512},
        input.options().dtype(at::kFloat8_e4m3fn)
    );
    at::Tensor global_scale = at::empty(
        {1},
        input.options().dtype(at::kFloat)
    );
    at::Tensor global_amax = at::empty(
        {1},
        input.options().dtype(at::kFloat)
    );

    using G = nvfp4_quantize::globals;
    G globals{
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(normalized),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(packed),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl, false>(
            scales,
            1,
            scales.size(0),
            scales.size(1),
            256
        ),
        .A_sc_global =
            kittens::py::tensor_to_gl<G::A_sc_global_gl>(global_scale),
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUDACHECK(cudaMemsetAsync(
        global_amax.data_ptr<float>(),
        0,
        sizeof(float),
        stream
    ));
    rmsnorm_nvfp4_quantize::rmsnorm_bf16_amax_kernel<<<
        rows,
        256,
        static_cast<size_t>(columns) * sizeof(kittens::bf16),
        stream
    >>>(
        reinterpret_cast<const kittens::bf16 *>(input.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(gamma.data_ptr()),
        reinterpret_cast<kittens::bf16 *>(normalized.data_ptr()),
        inv_rms.data_ptr<float>(),
        global_amax.data_ptr<float>(),
        static_cast<float>(epsilon),
        rows,
        columns
    );
    CUDACHECK(cudaGetLastError());
    rmsnorm_nvfp4_quantize::quantize_from_amax_kernel<<<
        globals.grid(),
        nvfp4_quantize::quantize_config::NUM_THREADS,
        globals.dynamic_shared_memory(),
        stream
    >>>(globals, global_amax.data_ptr<float>());
    CUDACHECK(cudaGetLastError());
    return {packed, scales, global_scale, inv_rms, normalized};
}

std::vector<at::Tensor> rmsnorm_backward_bf16(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor inv_rms,
    at::Tensor gradient
) {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16 &&
            input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
            input.size(0) %
                    rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_ROWS_PER_BLOCK ==
                0 &&
            input.size(1) ==
                rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_COLUMNS,
        "fused RMSNorm backward requires contiguous CUDA BF16 [M, 2048] "
        "with M divisible by 16"
    );
    TORCH_CHECK(
        gradient.scalar_type() == at::ScalarType::BFloat16 &&
            gradient.is_cuda() && gradient.is_contiguous() &&
            gradient.dim() == 2 && gradient.sizes() == input.sizes() &&
            gradient.device() == input.device(),
        "fused RMSNorm backward gradient must match the contiguous CUDA "
        "BF16 input"
    );
    TORCH_CHECK(
        gamma.scalar_type() == at::ScalarType::BFloat16 &&
            gamma.is_cuda() && gamma.is_contiguous() && gamma.dim() == 1 &&
            gamma.size(0) == input.size(1) && gamma.device() == input.device(),
        "fused RMSNorm backward gamma must be contiguous CUDA BF16 [2048] "
        "on the input device"
    );
    TORCH_CHECK(
        inv_rms.scalar_type() == at::ScalarType::Float &&
            inv_rms.is_cuda() && inv_rms.is_contiguous() &&
            inv_rms.dim() == 1 && inv_rms.size(0) == input.size(0) &&
            inv_rms.device() == input.device(),
        "fused RMSNorm backward inv_rms must be contiguous CUDA FP32 [M] "
        "on the input device"
    );

    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int partial_rows =
        rows /
        rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_ROWS_PER_BLOCK;
    at::Tensor input_gradient = at::empty_like(input);
    at::Tensor gamma_gradient = at::empty_like(gamma);
    at::Tensor gamma_gradient_partials = at::empty(
        {
            partial_rows,
            rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_COLUMNS,
        },
        input.options().dtype(at::kFloat)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_nvfp4_quantize::rmsnorm_backward_partial_kernel<<<
        partial_rows,
        rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_THREADS,
        0,
        stream
    >>>(
        reinterpret_cast<const kittens::bf16 *>(input.data_ptr()),
        reinterpret_cast<const kittens::bf16 *>(gamma.data_ptr()),
        inv_rms.data_ptr<float>(),
        reinterpret_cast<const kittens::bf16 *>(gradient.data_ptr()),
        reinterpret_cast<kittens::bf16 *>(input_gradient.data_ptr()),
        gamma_gradient_partials.data_ptr<float>(),
        rows
    );
    CUDACHECK(cudaGetLastError());
    rmsnorm_nvfp4_quantize::rmsnorm_backward_gamma_finalize_kernel<<<
        rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_COLUMNS /
            rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_THREADS,
        rmsnorm_nvfp4_quantize::RMSNORM_BACKWARD_THREADS,
        0,
        stream
    >>>(
        gamma_gradient_partials.data_ptr<float>(),
        reinterpret_cast<kittens::bf16 *>(gamma_gradient.data_ptr()),
        partial_rows
    );
    CUDACHECK(cudaGetLastError());
    return {input_gradient, gamma_gradient};
}

std::vector<at::Tensor> quantize_nvfp4_projection_weight(at::Tensor input) {
    // Learned weights use one shared E4M3 scale per 16x16 block.  The scale
    // is replicated into the 1x16 metadata rows consumed by tcgen05, making
    // the quantized representation invariant to the fprop/bprop transpose.
    return quantize_nvfp4_projection_operand_impl(input, 1.0, true);
}

__device__ __forceinline__ int64_t prepared_nvfp4_scale_index(
    int row,
    int column_group,
    int page_columns
) {
    const int page_row = row / 128;
    const int page_column = column_group / 4;
    const int scale_offset =
        (row % 32) * 16 + ((row % 128) / 32) * 4 + column_group % 4;
    return (
        (static_cast<int64_t>(page_row) * page_columns + page_column) * 512 +
        scale_offset
    );
}

__global__ __launch_bounds__(256) void transpose_prepared_nvfp4_weight_kernel(
    const uint8_t *__restrict__ source_payload,
    uint8_t *__restrict__ transpose_payload,
    const uint8_t *__restrict__ source_scales,
    uint8_t *__restrict__ transpose_scales,
    int rows,
    int columns
) {
    constexpr int kTile = 64;
    constexpr int kPackedTileColumns = kTile / 2;
    constexpr int kPayloadBytes = kTile * kPackedTileColumns;
    constexpr int kScaleTiles = (kTile / 16) * (kTile / 16);
    __shared__ uint8_t payload_tile[kTile][kTile + 1];
    __shared__ uint8_t scale_tile[kScaleTiles];

    const int source_row_base = static_cast<int>(blockIdx.y) * kTile;
    const int source_column_base = static_cast<int>(blockIdx.x) * kTile;
    const int source_payload_stride = columns / 2;
    const int transpose_payload_stride = rows / 2;
    for (
        int index = static_cast<int>(threadIdx.x);
        index < kPayloadBytes;
        index += static_cast<int>(blockDim.x)
    ) {
        const int local_row = index / kPackedTileColumns;
        const int local_column_byte = index % kPackedTileColumns;
        const uint8_t packed = source_payload[
            static_cast<int64_t>(source_row_base + local_row) *
                source_payload_stride +
            source_column_base / 2 + local_column_byte
        ];
        payload_tile[local_row][local_column_byte * 2] = packed & 0x0f;
        payload_tile[local_row][local_column_byte * 2 + 1] = packed >> 4;
    }

    // True-2D learned-weight scales are constant across each 16x16 tile.
    const int source_page_columns = columns / 64;
    const int transpose_page_columns = rows / 64;
    if (threadIdx.x < kScaleTiles) {
        const int source_tile_row = static_cast<int>(threadIdx.x) / 4;
        const int source_tile_column = static_cast<int>(threadIdx.x) % 4;
        scale_tile[threadIdx.x] = source_scales[
            prepared_nvfp4_scale_index(
                source_row_base + source_tile_row * 16,
                source_column_base / 16 + source_tile_column,
                source_page_columns
            )
        ];
    }
    __syncthreads();

    // A 64x64 nibble tile keeps both global-memory directions coalesced.  The
    // +1 shared-memory stride avoids the bank conflict of a square transpose.
    for (
        int index = static_cast<int>(threadIdx.x);
        index < kPayloadBytes;
        index += static_cast<int>(blockDim.x)
    ) {
        const int local_transpose_row = index / kPackedTileColumns;
        const int local_transpose_pair = index % kPackedTileColumns;
        const uint8_t value0 =
            payload_tile[local_transpose_pair * 2][local_transpose_row];
        const uint8_t value1 =
            payload_tile[local_transpose_pair * 2 + 1][local_transpose_row];
        transpose_payload[
            static_cast<int64_t>(source_column_base + local_transpose_row) *
                transpose_payload_stride +
            source_row_base / 2 + local_transpose_pair
        ] = static_cast<uint8_t>(value0 | (value1 << 4));
    }

    // Each block also transposes its 4x4 grid of scale tiles.  Load each tile
    // once above, then replicate its one E4M3 byte into the sixteen physical
    // metadata rows consumed by tcgen05.
    const int source_tile = static_cast<int>(threadIdx.x) / 16;
    const int scale_replica = static_cast<int>(threadIdx.x) % 16;
    const int source_tile_row = source_tile / 4;
    const int source_tile_column = source_tile % 4;
    const int transpose_row =
        source_column_base + source_tile_column * 16 + scale_replica;
    const int transpose_column_group =
        source_row_base / 16 + source_tile_row;
    const int64_t transpose_index = prepared_nvfp4_scale_index(
        transpose_row,
        transpose_column_group,
        transpose_page_columns
    );
    transpose_scales[transpose_index] = scale_tile[source_tile];
}

std::vector<at::Tensor> quantize_nvfp4_projection_weight_dual(
    at::Tensor input
) {
    // Quantize the current master weight once, then derive the physical
    // transpose from its exact FP4 codes and transpose-consistent 16x16 scale
    // tiles.  Both layouts therefore describe the same weight version used by
    // this forward/backward pair without rereading or requantizing BF16.
    auto forward = quantize_nvfp4_projection_operand_impl(input, 1.0, true);
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int columns = static_cast<int>(input.size(1));
    at::Tensor transpose_payload = at::empty(
        {columns, rows / 2},
        forward[0].options()
    );
    at::Tensor transpose_scales = at::empty(
        {columns / 128, rows / 64, 512},
        forward[1].options()
    );
    constexpr int kThreads = 256;
    constexpr int kTile = 64;
    const dim3 grid(columns / kTile, rows / kTile);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    transpose_prepared_nvfp4_weight_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(
        reinterpret_cast<const uint8_t *>(forward[0].data_ptr()),
        reinterpret_cast<uint8_t *>(transpose_payload.data_ptr()),
        reinterpret_cast<const uint8_t *>(forward[1].data_ptr()),
        reinterpret_cast<uint8_t *>(transpose_scales.data_ptr()),
        rows,
        columns
    );
    CUDACHECK(cudaGetLastError());
    return {
        forward[0],
        forward[1],
        forward[2],
        transpose_payload,
        transpose_scales,
        forward[2],
    };
}

std::vector<at::Tensor> quantize_nvfp4_projection_operand_scaled(
    at::Tensor input,
    double value_scale
) {
    TORCH_CHECK(
        std::isfinite(value_scale) && value_scale > 0.0,
        "NVFP4 projection operand value scale must be finite and positive"
    );
    return quantize_nvfp4_projection_operand_impl(input, value_scale, false);
}

std::vector<at::Tensor> quantize_nvfp4_projection_operand_precomputed_scale(
    at::Tensor input,
    at::Tensor global_scale
) {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16 &&
            input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
            input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
        "NVFP4 projection quantization requires contiguous CUDA BF16 [M, K] "
        "with both dimensions divisible by 128"
    );
    TORCH_CHECK(
        global_scale.scalar_type() == at::ScalarType::Float &&
            global_scale.is_cuda() && global_scale.is_contiguous() &&
            global_scale.numel() == 1,
        "precomputed NVFP4 global scale must be one contiguous CUDA float32 "
        "value"
    );
    kittens::py::device_check(input, global_scale);
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    at::Tensor packed = at::empty(
        {rows, cols / 2},
        input.options().dtype(at::kFloat4_e2m1fn_x2)
    );
    at::Tensor scales = at::empty(
        {rows / 128, cols / 64, 512},
        input.options().dtype(at::kFloat8_e4m3fn)
    );
    using C = nvfp4_quantize::quantize_config;
    using G = nvfp4_quantize::globals;
    G globals{
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(input),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(packed),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl, false>(
            scales,
            1,
            scales.size(0),
            scales.size(1),
            256
        ),
        .A_sc_global =
            kittens::py::tensor_to_gl<G::A_sc_global_gl>(global_scale),
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // The supplied decode scale comes from an upstream epilogue or delayed
    // scaling state. Skipping the matrix-wide amax reduction is the contract
    // that makes this representative of a fused dQ handoff.
    kittens::py::launch_kernel<
        C,
        G,
        nvfp4_quantize::quantize_kernel<false>
    >(globals);
    const int threads = 256;
    const int blocks = static_cast<int>(
        ((scales.numel() / 4) + threads - 1) / threads
    );
    fp8_nan_fixup_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint8_t *>(scales.data_ptr()),
        scales.numel()
    );
    CUDACHECK(cudaGetLastError());
    return {packed, scales, global_scale};
}

std::vector<at::Tensor>
quantize_nvfp4_projection_operand_precomputed_scale_inverse_rope(
    at::Tensor input,
    at::Tensor global_scale,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    bool publish_inverse_bf16
) {
    constexpr int kQkDepth = 192;
    constexpr int kVDepth = 128;
    constexpr int kHeadWidth = kQkDepth * 2 + kVDepth;
    constexpr int kRotaryPairs = kQkDepth / 2;
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16 &&
            input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
            input.size(0) % 128 == 0 && input.size(1) % 128 == 0 &&
            input.size(1) % kHeadWidth == 0,
        "inverse-RoPE NVFP4 projection quantization requires contiguous "
        "CUDA BF16 [B*S, H*512] with both dimensions divisible by 128"
    );
    TORCH_CHECK(
        global_scale.scalar_type() == at::ScalarType::Float &&
            global_scale.is_cuda() && global_scale.is_contiguous() &&
            global_scale.numel() == 1,
        "precomputed NVFP4 global scale must be one contiguous CUDA float32 "
        "value"
    );
    TORCH_CHECK(
        rope_cos.scalar_type() == at::ScalarType::BFloat16 &&
            rope_sin.scalar_type() == at::ScalarType::BFloat16 &&
            rope_cos.is_cuda() && rope_sin.is_cuda() &&
            rope_cos.is_contiguous() && rope_sin.is_contiguous() &&
            rope_cos.dim() == 3 && rope_sin.sizes() == rope_cos.sizes() &&
            rope_cos.size(0) * rope_cos.size(1) == input.size(0) &&
            rope_cos.size(2) == kRotaryPairs,
        "pair-native RoPE tables must be matching contiguous CUDA BF16 "
        "[B, S, 96] whose row count matches the operand"
    );
    kittens::py::device_check(input, global_scale, rope_cos, rope_sin);
    const c10::cuda::CUDAGuard device_guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    at::Tensor packed = at::empty(
        {rows, cols / 2},
        input.options().dtype(at::kFloat4_e2m1fn_x2)
    );
    at::Tensor scales = at::empty(
        {rows / 128, cols / 64, 512},
        input.options().dtype(at::kFloat8_e4m3fn)
    );
    using C = tkfa4_inverse_rope_nvfp4_quantize::config;
    using G = tkfa4_inverse_rope_nvfp4_quantize::globals;
    G globals{
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(input),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(packed),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl, false>(
            scales,
            1,
            scales.size(0),
            scales.size(1),
            256
        ),
        .A_sc_global =
            kittens::py::tensor_to_gl<G::A_sc_global_gl>(global_scale),
        .rope_cos = reinterpret_cast<const kittens::bf16 *>(
            rope_cos.data_ptr()
        ),
        .rope_sin = reinterpret_cast<const kittens::bf16 *>(
            rope_sin.data_ptr()
        ),
    };
    if (publish_inverse_bf16) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_inverse_rope_nvfp4_quantize::quantize_kernel<true>
        >(globals);
    } else {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_inverse_rope_nvfp4_quantize::quantize_kernel<false>
        >(globals);
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int blocks = static_cast<int>(
        ((scales.numel() / 4) + threads - 1) / threads
    );
    fp8_nan_fixup_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint8_t *>(scales.data_ptr()),
        scales.numel()
    );
    CUDACHECK(cudaGetLastError());
    return {packed, scales, global_scale};
}

std::vector<at::Tensor> project_qk_adaptive_fp4_nvfp4(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qk_weight_fp4,
    at::Tensor qk_weight_scales,
    at::Tensor qk_weight_global_scale,
    at::Tensor adaptive_scales,
    int batch,
    int seq_len,
    int heads,
    bool publish_fp4
) {
    constexpr int kDepth = 192;
    constexpr int kPackedDepth = kDepth / 2;
    const int rows = batch * seq_len;
    const int output_width = heads * kDepth;
    TORCH_CHECK(
        input_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            qk_weight_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            input_fp4.is_cuda() && qk_weight_fp4.is_cuda() &&
            input_fp4.is_contiguous() && qk_weight_fp4.is_contiguous() &&
            input_fp4.dim() == 2 && qk_weight_fp4.dim() == 2,
        "projection operands must be contiguous CUDA packed E2M1 matrices"
    );
    const int hidden = static_cast<int>(input_fp4.size(1) * 2);
    TORCH_CHECK(
        input_fp4.size(0) == rows &&
            qk_weight_fp4.size(0) == 2 * output_width &&
            qk_weight_fp4.size(1) == input_fp4.size(1) &&
            rows % 256 == 0 && seq_len % 256 == 0 &&
            output_width % 256 == 0 && hidden % 256 == 0,
        "NVFP4 projection geometry requires A=[B*S,K/2], B=[2*H*192,K/2], "
        "S divisible by 256, H*192 divisible by 256, and K divisible by 256"
    );
    TORCH_CHECK(
        input_scales.scalar_type() == at::kFloat8_e4m3fn &&
            qk_weight_scales.scalar_type() == at::kFloat8_e4m3fn &&
            input_scales.is_cuda() && qk_weight_scales.is_cuda() &&
            input_scales.is_contiguous() &&
            qk_weight_scales.is_contiguous() &&
            input_global_scale.scalar_type() == at::kFloat &&
            qk_weight_global_scale.scalar_type() == at::kFloat &&
            input_global_scale.is_cuda() && qk_weight_global_scale.is_cuda() &&
            input_global_scale.numel() == 1 &&
            qk_weight_global_scale.numel() == 1,
        "NVFP4 operands require contiguous E4M3 scale tensors and one float32 "
        "global scale each"
    );
    TORCH_CHECK(
        adaptive_scales.scalar_type() == at::ScalarType::Float &&
            adaptive_scales.is_cuda() && adaptive_scales.is_contiguous() &&
            adaptive_scales.dim() == 3 &&
            adaptive_scales.size(0) == batch &&
            adaptive_scales.size(1) == heads &&
            adaptive_scales.size(2) == 7,
        "adaptive_scales must be contiguous CUDA float32 [B, H, 7]"
    );
    kittens::py::device_check(
        input_fp4,
        input_scales,
        input_global_scale,
        qk_weight_fp4,
        qk_weight_scales,
        qk_weight_global_scale,
        adaptive_scales
    );
    const c10::cuda::CUDAGuard device_guard(input_fp4.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "persistent projection FP4 epilogue requires GB200 / SM100"
    );

    at::Tensor q = at::empty(
        {batch, seq_len, heads, kDepth},
        input_fp4.options().dtype(at::kBFloat16)
    );
    at::Tensor k = at::empty_like(q);
    auto byte_options = input_fp4.options().dtype(at::ScalarType::Byte);
    at::Tensor q_sequence_aligned = at::empty(
        {batch, heads, kDepth, seq_len},
        byte_options
    );
    at::Tensor q_depth_packed = at::empty(
        {batch, heads, seq_len, kPackedDepth},
        byte_options
    );
    at::Tensor k_depth_aligned = at::empty(q.sizes(), byte_options);
    at::Tensor k_depth_packed = at::empty(
        {batch, heads, seq_len, kPackedDepth},
        byte_options
    );

    // FP4 publication uses the static epilogue scratch; four dynamic load
    // stages leave insufficient shared-memory headroom on SM100.
    using C = tkfa4_projection::config<3, 4>;
    using G = tkfa4_projection::globals<C>;
    G globals{
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(
            input_fp4,
            1,
            1,
            rows,
            hidden / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            input_scales,
            1,
            input_scales.size(0),
            input_scales.size(1),
            256
        ),
        .A_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            input_global_scale
        ),
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(
            qk_weight_fp4,
            1,
            1,
            2 * output_width,
            hidden / 2
        ),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            qk_weight_scales,
            1,
            qk_weight_scales.size(0),
            qk_weight_scales.size(1),
            256
        ),
        .B_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            qk_weight_global_scale
        ),
        .Q = kittens::py::tensor_to_gl<typename G::D_gl>(
            q,
            1,
            1,
            rows,
            output_width
        ),
        .K = kittens::py::tensor_to_gl<typename G::D_gl>(
            k,
            1,
            1,
            rows,
            output_width
        ),
        .V = kittens::py::tensor_to_gl<typename G::D_gl>(
            k,
            1,
            1,
            rows,
            output_width
        ),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(
            k,
            1,
            1,
            rows,
            output_width
        ),
        .q_sequence_aligned =
            reinterpret_cast<uint8_t *>(q_sequence_aligned.data_ptr()),
        .q_depth_packed =
            reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
        .k_depth_aligned =
            reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr()),
        .k_depth_packed =
            reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
        .adaptive_scales =
            reinterpret_cast<const float *>(adaptive_scales.data_ptr()),
        .batch = batch,
        .seq_len = seq_len,
        .heads = heads,
        .v_width = 0,
        .v_scale_rows = 0,
    };
    if (publish_fp4) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<C, true>
        >(globals);
    } else {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<C, false>
        >(globals);
    }
    return {
        q,
        k,
        q_sequence_aligned,
        q_depth_packed,
        k_depth_aligned,
        k_depth_packed,
        adaptive_scales
    };
}

at::Tensor project_nvfp4_generic(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor weight_fp4,
    at::Tensor weight_scales,
    at::Tensor weight_global_scale
) {
    TORCH_CHECK(
        input_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            weight_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            input_fp4.is_cuda() && weight_fp4.is_cuda() &&
            input_fp4.is_contiguous() && weight_fp4.is_contiguous() &&
            input_fp4.dim() == 2 && weight_fp4.dim() == 2,
        "generic NVFP4 projection operands must be contiguous CUDA packed "
        "E2M1 matrices"
    );
    const int rows = static_cast<int>(input_fp4.size(0));
    const int reduction = static_cast<int>(input_fp4.size(1) * 2);
    const int output_width = static_cast<int>(weight_fp4.size(0));
    TORCH_CHECK(
        weight_fp4.size(1) == input_fp4.size(1) &&
            rows % 256 == 0 && reduction % 256 == 0 &&
            output_width % 256 == 0,
        "generic NVFP4 projection requires A=[M,K/2], B=[N,K/2], and "
        "M/K/N divisible by 256"
    );
    TORCH_CHECK(
        input_scales.scalar_type() == at::kFloat8_e4m3fn &&
            weight_scales.scalar_type() == at::kFloat8_e4m3fn &&
            input_scales.is_cuda() && weight_scales.is_cuda() &&
            input_scales.is_contiguous() &&
            weight_scales.is_contiguous() &&
            input_scales.dim() == 3 && weight_scales.dim() == 3 &&
            input_scales.size(0) == rows / 128 &&
            input_scales.size(1) == reduction / 64 &&
            input_scales.size(2) == 512 &&
            weight_scales.size(0) == output_width / 128 &&
            weight_scales.size(1) == reduction / 64 &&
            weight_scales.size(2) == 512,
        "generic NVFP4 projection scales must be E4M3 "
        "[rows/128,K/64,512]"
    );
    TORCH_CHECK(
        input_global_scale.scalar_type() == at::kFloat &&
            weight_global_scale.scalar_type() == at::kFloat &&
            input_global_scale.is_cuda() &&
            weight_global_scale.is_cuda() &&
            input_global_scale.numel() == 1 &&
            weight_global_scale.numel() == 1,
        "generic NVFP4 projection requires one float32 global scale per "
        "operand"
    );
    kittens::py::device_check(
        input_fp4,
        input_scales,
        input_global_scale,
        weight_fp4,
        weight_scales,
        weight_global_scale
    );
    const c10::cuda::CUDAGuard device_guard(input_fp4.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "generic NVFP4 projection requires GB200 / SM100"
    );

    at::Tensor output = at::empty(
        {rows, output_width},
        input_fp4.options().dtype(at::kBFloat16)
    );
    using C = tkfa4_projection::config<4, 4>;
    using G = tkfa4_projection::globals<C>;
    G globals{
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(
            input_fp4,
            1,
            1,
            rows,
            reduction / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            input_scales,
            1,
            input_scales.size(0),
            input_scales.size(1),
            256
        ),
        .A_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            input_global_scale
        ),
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(
            weight_fp4,
            1,
            1,
            output_width,
            reduction / 2
        ),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            weight_scales,
            1,
            weight_scales.size(0),
            weight_scales.size(1),
            256
        ),
        .B_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            weight_global_scale
        ),
        .Q = kittens::py::tensor_to_gl<typename G::D_gl>(
            output,
            1,
            1,
            rows,
            output_width
        ),
        .K = kittens::py::tensor_to_gl<typename G::D_gl>(
            output,
            1,
            1,
            rows,
            output_width
        ),
        .V = kittens::py::tensor_to_gl<typename G::D_gl>(
            output,
            1,
            1,
            rows,
            output_width
        ),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(
            output,
            1,
            1,
            rows,
            output_width
        ),
        .output_width = output_width,
    };
    kittens::py::launch_kernel<
        C,
        G,
        tkfa4_projection::kernel<
            C, false, false, false, true, false, false, false, true
        >
    >(globals);
    return output;
}

at::Tensor project_e4m3_generic(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor weight_fp8,
    at::Tensor weight_channel_decode
) {
    TORCH_CHECK(
        input_fp8.scalar_type() == at::kFloat8_e4m3fn &&
            weight_fp8.scalar_type() == at::kFloat8_e4m3fn &&
            input_fp8.is_cuda() && weight_fp8.is_cuda() &&
            input_fp8.is_contiguous() && weight_fp8.is_contiguous() &&
            input_fp8.dim() == 2 && weight_fp8.dim() == 2,
        "generic E4M3 projection operands must be contiguous CUDA E4M3 "
        "matrices"
    );
    const int64_t rows64 = input_fp8.size(0);
    const int64_t reduction64 = input_fp8.size(1);
    const int64_t output_width64 = weight_fp8.size(0);
    TORCH_CHECK(
        rows64 > 0 && reduction64 > 0 && output_width64 > 0 &&
            weight_fp8.size(1) == reduction64 && rows64 % 256 == 0 &&
            reduction64 % 128 == 0 && output_width64 % 256 == 0,
        "generic E4M3 projection requires A=[M,K], B=[N,K], M/N divisible "
        "by 256, and K divisible by 128"
    );
    TORCH_CHECK(
        rows64 <= std::numeric_limits<int>::max() &&
            reduction64 <= std::numeric_limits<int>::max() &&
            output_width64 <= std::numeric_limits<int>::max(),
        "generic E4M3 projection dimensions exceed the kernel index range"
    );
    TORCH_CHECK(
        input_row_decode.scalar_type() == at::kFloat &&
            weight_channel_decode.scalar_type() == at::kFloat &&
            input_row_decode.is_cuda() && weight_channel_decode.is_cuda() &&
            input_row_decode.is_contiguous() &&
            weight_channel_decode.is_contiguous() &&
            input_row_decode.dim() == 1 &&
            weight_channel_decode.dim() == 1 &&
            input_row_decode.numel() == rows64 &&
            weight_channel_decode.numel() == output_width64,
        "generic E4M3 projection decode scales must be contiguous CUDA "
        "float32 vectors with one value per input row and output channel"
    );
    kittens::py::device_check(
        input_fp8,
        input_row_decode,
        weight_fp8,
        weight_channel_decode
    );
    const c10::cuda::CUDAGuard device_guard(input_fp8.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "generic E4M3 projection requires GB200 / SM100"
    );

    const int rows = static_cast<int>(rows64);
    const int reduction = static_cast<int>(reduction64);
    const int output_width = static_cast<int>(output_width64);
    at::Tensor output = at::empty(
        {rows, output_width},
        input_fp8.options().dtype(at::kBFloat16)
    );
    // Reuse the exact dense-E4M3 geometry authenticated by the fused QKV
    // producer. K128 permits four load stages while retaining the 64 KiB BF16
    // output ring below SM100's shared-memory limit.
    using C = tkfa4_projection::config<4, 4, 128, 128, true>;
    using G = tkfa4_projection::globals<C>;
    typename G::A_sc_gl input_decode_descriptor{
        reinterpret_cast<float *>(input_row_decode.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        static_cast<size_t>(rows)
    };
    typename G::B_sc_gl weight_decode_descriptor{
        reinterpret_cast<float *>(weight_channel_decode.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        static_cast<size_t>(output_width)
    };
    // Dense E4M3 applies the row/channel decode vectors directly and never
    // reads the tensor-wide NVFP4 scale descriptors retained by the shared ABI.
    typename G::scale_gl unused_global_scale_descriptor{
        reinterpret_cast<float *>(input_row_decode.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    G globals{
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(
            input_fp8,
            1,
            1,
            rows,
            reduction
        ),
        .A_sc = input_decode_descriptor,
        .A_scale = unused_global_scale_descriptor,
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(
            weight_fp8,
            1,
            1,
            output_width,
            reduction
        ),
        .B_sc = weight_decode_descriptor,
        .B_scale = unused_global_scale_descriptor,
        .Q = kittens::py::tensor_to_gl<typename G::D_gl>(
            output,
            1,
            1,
            rows,
            output_width
        ),
        .K = kittens::py::tensor_to_gl<typename G::D_gl>(
            output,
            1,
            1,
            rows,
            output_width
        ),
        .V = kittens::py::tensor_to_gl<typename G::D_gl>(
            output,
            1,
            1,
            rows,
            output_width
        ),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(
            output,
            1,
            1,
            rows,
            output_width
        ),
        // Publication code is compile-time disabled, but retain non-zero row
        // metadata so future compiler changes cannot expose a latent divide by
        // zero in the shared epilogue's publication-only address arithmetic.
        .batch = 1,
        .seq_len = rows,
        .heads = 0,
        .v_width = 0,
        .v_scale_rows = 0,
        .output_width = output_width,
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tkfa4_projection::launch_on_stream<
        C,
        false,  // PUBLISH_FP4
        false,  // PUBLISH_FORWARD_QK
        false,  // PUBLISH_V_MXFP4
        true,   // STORE_BF16
        false,  // OUTPUT_IS_DOUT
        false,  // PUBLISH_PURE_QK
        false,  // PURE_QK_SINGLE_QUANT
        true,   // SINGLE_OUTPUT
        false   // APPLY_ROPE
    >(globals, stream);
    return output;
}

// Caller-owned output ABI shared by the D64 and D128 native NVFP4
// projections.  This is a route-neutral superset: exact FP8-PV writes
// feature-major FP8 V, while MXFP4-PV writes MXFP4 V and its scale pages.
// Both routes write normal-order E4M3 Q/K/V for backward.
struct paired_d64_nvfp4_forward_outputs {
    at::Tensor q_depth_packed;
    at::Tensor k_depth_packed;
    at::Tensor q_forward_scales;
    at::Tensor q_forward_global_scale;
    at::Tensor k_forward_scales;
    at::Tensor k_forward_global_scale;
    at::Tensor v_mxfp4;
    at::Tensor v_mxfp4_scales;
    at::Tensor v_forward_fp8;
    at::Tensor v_backward_fp8;
    at::Tensor q_backward_fp8;
    at::Tensor k_backward_fp8;
    // Optional compact D128-only publications.  Existing compact routes leave
    // these tensors undefined; the MX-backward-V specialization binds caller-
    // owned [B,S,Hkv,D/2] payload and [B,S/128,Hkv,512] scale storage.
    at::Tensor v_backward_mxfp4;
    at::Tensor v_backward_mxfp4_scales;
};

inline void check_paired_d64_nvfp4_forward_output(
    const at::Tensor &output,
    at::ScalarType dtype,
    at::IntArrayRef shape,
    const c10::Device &device,
    const char *name
) {
    TORCH_CHECK(
        output.scalar_type() == dtype && output.is_cuda() &&
            output.is_contiguous() && output.device() == device &&
            output.sizes() == shape && output.storage_offset() == 0 &&
            !output.requires_grad(),
        name,
        " must be a base contiguous non-grad CUDA tensor with dtype ",
        dtype,
        " and shape ",
        shape,
        " on the projection device"
    );
}

template <int kLogicalDepth>
inline void check_nvfp4_qkv_forward_outputs(
    const paired_d64_nvfp4_forward_outputs &outputs,
    const at::Tensor &input,
    const at::Tensor &input_scales,
    const at::Tensor &input_global_scale,
    const at::Tensor &qkv_weight_fp4,
    const at::Tensor &qkv_weight_scales,
    const at::Tensor &qkv_weight_global_scale,
    const at::Tensor &adaptive_scales,
    const at::Tensor &rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads
) {
    static_assert(kLogicalDepth == 64 || kLogicalDepth == 128);
    constexpr int kQkChunks = kLogicalDepth / 64;
    check_paired_d64_nvfp4_forward_output(
        outputs.q_depth_packed,
        at::kByte,
        {batch, q_heads, seq_len, kLogicalDepth / 2},
        input.device(),
        "q_depth_packed_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.k_depth_packed,
        at::kByte,
        {batch, kv_heads, seq_len, kLogicalDepth / 2},
        input.device(),
        "k_depth_packed_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.q_forward_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 128, q_heads * kQkChunks, 512},
        input.device(),
        "q_forward_scales_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.q_forward_global_scale,
        at::kFloat,
        {batch, q_heads},
        input.device(),
        "q_forward_global_scale_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.k_forward_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 64, kv_heads * kQkChunks, 512},
        input.device(),
        "k_forward_scales_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.k_forward_global_scale,
        at::kFloat,
        {batch, kv_heads},
        input.device(),
        "k_forward_global_scale_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.v_mxfp4,
        at::kFloat4_e2m1fn_x2,
        {batch, kv_heads, kLogicalDepth, seq_len / 2},
        input.device(),
        "v_mxfp4_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.v_mxfp4_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 128, kv_heads, 512},
        input.device(),
        "v_mxfp4_scales_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.v_forward_fp8,
        at::kFloat8_e4m3fn,
        {batch, kv_heads, kLogicalDepth, seq_len},
        input.device(),
        "v_forward_fp8_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.v_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, kv_heads, kLogicalDepth},
        input.device(),
        "v_backward_fp8_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.q_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, q_heads, kLogicalDepth},
        input.device(),
        "q_backward_fp8_out"
    );
    check_paired_d64_nvfp4_forward_output(
        outputs.k_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, kv_heads, kLogicalDepth},
        input.device(),
        "k_backward_fp8_out"
    );
    const at::Tensor *output_tensors[] = {
        &outputs.q_depth_packed,
        &outputs.k_depth_packed,
        &outputs.q_forward_scales,
        &outputs.q_forward_global_scale,
        &outputs.k_forward_scales,
        &outputs.k_forward_global_scale,
        &outputs.v_mxfp4,
        &outputs.v_mxfp4_scales,
        &outputs.v_forward_fp8,
        &outputs.v_backward_fp8,
        &outputs.q_backward_fp8,
        &outputs.k_backward_fp8,
    };
    const char *output_names[] = {
        "q_depth_packed_out",
        "k_depth_packed_out",
        "q_forward_scales_out",
        "q_forward_global_scale_out",
        "k_forward_scales_out",
        "k_forward_global_scale_out",
        "v_mxfp4_out",
        "v_mxfp4_scales_out",
        "v_forward_fp8_out",
        "v_backward_fp8_out",
        "q_backward_fp8_out",
        "k_backward_fp8_out",
    };
    const at::Tensor *read_tensors[] = {
        &input,
        &input_scales,
        &input_global_scale,
        &qkv_weight_fp4,
        &qkv_weight_scales,
        &qkv_weight_global_scale,
        &adaptive_scales,
        &rope_packed,
    };
    const char *read_names[] = {
        "input_fp4",
        "input_scales",
        "input_global_scale",
        "qkv_weight_fp4",
        "qkv_weight_scales",
        "qkv_weight_global_scale",
        "adaptive_scales",
        "rope_packed",
    };
    constexpr std::uintptr_t kOutputAlignment = alignof(uint4);
    static_assert(kOutputAlignment == 16);
    const auto ranges_overlap = [](
        const at::Tensor &first,
        const at::Tensor &second
    ) {
        const auto first_begin = reinterpret_cast<std::uintptr_t>(
            first.data_ptr()
        );
        const auto second_begin = reinterpret_cast<std::uintptr_t>(
            second.data_ptr()
        );
        const auto first_bytes = static_cast<std::uintptr_t>(
            first.numel() * first.element_size()
        );
        const auto second_bytes = static_cast<std::uintptr_t>(
            second.numel() * second.element_size()
        );
        return first_begin <= second_begin
            ? second_begin - first_begin < first_bytes
            : first_begin - second_begin < second_bytes;
    };
    for (int lhs = 0; lhs < 12; ++lhs) {
        const auto address = reinterpret_cast<std::uintptr_t>(
            output_tensors[lhs]->data_ptr()
        );
        TORCH_CHECK(
            address % kOutputAlignment == 0,
            output_names[lhs],
            " must have a 16-byte-aligned base for vector publication"
        );
        for (int rhs = lhs + 1; rhs < 12; ++rhs) {
            TORCH_CHECK(
                !ranges_overlap(
                    *output_tensors[lhs],
                    *output_tensors[rhs]
                ),
                output_names[lhs],
                " and ",
                output_names[rhs],
                " must occupy disjoint byte ranges"
            );
        }
        for (int read = 0; read < 8; ++read) {
            TORCH_CHECK(
                !ranges_overlap(
                    *output_tensors[lhs],
                    *read_tensors[read]
                ),
                output_names[lhs],
                " must not overlap read operand ",
                read_names[read]
            );
        }
    }
    TORCH_CHECK(
        outputs.v_backward_mxfp4.defined() ==
            outputs.v_backward_mxfp4_scales.defined(),
        "backward MXFP4 V payload and scales must be supplied together"
    );
    if (outputs.v_backward_mxfp4.defined()) {
        check_paired_d64_nvfp4_forward_output(
            outputs.v_backward_mxfp4,
            at::kByte,
            {batch, seq_len, kv_heads, kLogicalDepth / 2},
            input.device(),
            "v_backward_mxfp4_out"
        );
        check_paired_d64_nvfp4_forward_output(
            outputs.v_backward_mxfp4_scales,
            at::kByte,
            {batch, seq_len / 128, kv_heads, 512},
            input.device(),
            "v_backward_mxfp4_scales_out"
        );
        const at::Tensor *mx_outputs[] = {
            &outputs.v_backward_mxfp4,
            &outputs.v_backward_mxfp4_scales,
        };
        const char *mx_names[] = {
            "v_backward_mxfp4_out",
            "v_backward_mxfp4_scales_out",
        };
        for (int lhs = 0; lhs < 2; ++lhs) {
            const auto address = reinterpret_cast<std::uintptr_t>(
                mx_outputs[lhs]->data_ptr()
            );
            TORCH_CHECK(
                address % kOutputAlignment == 0,
                mx_names[lhs],
                " must have a 16-byte-aligned base for vector publication"
            );
            for (int rhs = lhs + 1; rhs < 2; ++rhs) {
                TORCH_CHECK(
                    !ranges_overlap(*mx_outputs[lhs], *mx_outputs[rhs]),
                    mx_names[lhs],
                    " and ",
                    mx_names[rhs],
                    " must occupy disjoint byte ranges"
                );
            }
            for (int rhs = 0; rhs < 12; ++rhs) {
                TORCH_CHECK(
                    !ranges_overlap(*mx_outputs[lhs], *output_tensors[rhs]),
                    mx_names[lhs],
                    " and ",
                    output_names[rhs],
                    " must occupy disjoint byte ranges"
                );
            }
            for (int read = 0; read < 8; ++read) {
                TORCH_CHECK(
                    !ranges_overlap(*mx_outputs[lhs], *read_tensors[read]),
                    mx_names[lhs],
                    " must not overlap read operand ",
                    read_names[read]
                );
            }
        }
    }
}

inline void check_paired_d64_nvfp4_forward_outputs(
    const paired_d64_nvfp4_forward_outputs &outputs,
    const at::Tensor &input,
    const at::Tensor &input_scales,
    const at::Tensor &input_global_scale,
    const at::Tensor &qkv_weight_fp4,
    const at::Tensor &qkv_weight_scales,
    const at::Tensor &qkv_weight_global_scale,
    const at::Tensor &adaptive_scales,
    const at::Tensor &rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads
) {
    check_nvfp4_qkv_forward_outputs<64>(
        outputs,
        input,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        rope_packed,
        batch,
        seq_len,
        q_heads,
        kv_heads
    );
}

template <
    int kQkDepth = 192,
    bool kPairedD64 = false,
    bool kInterleaveCausalKv = false,
    bool kCompactForwardOut = false,
    bool kValidateCompactContracts = false,
    bool kPublishRepresentedBackwardFp8 = false,
    bool kPerBlockQkScales = false,
    bool kExperimentalSplitVBackward = false,
    bool kExperimentalE4m3DerivedMxfp4V = false,
    bool kExperimentalOutputSharedSplitV = false,
    bool kCompactPublishesMxV = kInterleaveCausalKv,
    bool kCompactPublishesMxBackwardV = false,
    bool kExperimentalCommonRowscaleMxfp4V = false,
    bool kExperimentalSharedTileMxfp4V = false
>
std::vector<at::Tensor> project_qkv_unified_fp4_nvfp4_impl(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_pure_qk,
    bool pure_qk_single_quant,
    bool publish_fp8_backward,
    const at::Tensor *rope_cos,
    const at::Tensor *rope_sin,
    const at::Tensor *rope_packed,
    int cluster_cap = 0,
    bool cache_packed_rope = false,
    bool cache_adaptive_qk_scale = false,
    bool v_mxfp4_scale_2d = true,
    bool per_block_qk_scales = false,
    const paired_d64_nvfp4_forward_outputs *forward_outputs = nullptr
) {
    constexpr int kVDepth = 128;
    constexpr int kPackedQkDepth = kQkDepth / 2;
    constexpr int kQkChunks = kQkDepth / 64;
    constexpr int kVScaleRows = kQkDepth == 128 ? 1 : 2;
    constexpr int kLogicalDepth = kPairedD64 ? 64 : kQkDepth;
    constexpr int kLogicalHeadMultiplier = kPairedD64 ? 2 : 1;
    // The causal-interleaved route consumes the native MXFP4 V publication.
    // Publishing the redundant feature-major FP8 mirror would both waste an
    // output transaction and overwrite shared V pairs while the MX publisher
    // is still reading them across the warpgroup.
    constexpr bool kPublishForwardFp8 =
        !kInterleaveCausalKv &&
        (!kCompactForwardOut || !kCompactPublishesMxV);
    // Preserve the allocating legacy ABI, which always publishes MXFP4 V.
    // Represented-backward specializations instead match their compact ABI:
    // exact FP8 omits the inactive MX publication, while causal MX retains it.
    constexpr bool kPublishMxfp4V = kCompactForwardOut
        ? kCompactPublishesMxV
        : (kPublishRepresentedBackwardFp8 ? kInterleaveCausalKv : true);
    constexpr bool kPublishQkBackwardFp8 =
        kCompactPublishesMxBackwardV || kPublishRepresentedBackwardFp8;
    static_assert(!kValidateCompactContracts || kCompactForwardOut);
    static_assert(!kCompactForwardOut || kQkDepth == 128);
    static_assert(!kPairedD64 || kQkDepth == 128);
    static_assert(!kInterleaveCausalKv || kPairedD64);
    static_assert(
        !kCompactPublishesMxBackwardV ||
            (kCompactForwardOut && kCompactPublishesMxV &&
             kQkDepth == 128 && !kPairedD64 &&
             !kPublishRepresentedBackwardFp8),
        "MX-only backward V requires compact native-D128 MX publication"
    );
    static_assert(
        !kPublishRepresentedBackwardFp8 ||
            (
                kQkDepth == 128 && kPerBlockQkScales &&
                (
                    kPairedD64 ||
                    (
                        kCompactForwardOut && !kInterleaveCausalKv &&
                        !kCompactPublishesMxV &&
                        !kCompactPublishesMxBackwardV
                    )
                )
            ),
        "represented native NVFP4 backward requires per-block Q/K and either "
        "paired D64 or the caller-owned D128 FP8-PV route"
    );
    static_assert(
        !kPerBlockQkScales || kPublishRepresentedBackwardFp8 ||
            kCompactPublishesMxBackwardV ||
            (kExperimentalOutputSharedSplitV && kCompactForwardOut &&
             kQkDepth == 128 && !kPairedD64),
        "compile-time per-block Q/K requires represented backward publication "
        "or the compact D128 output-shared route"
    );
    static_assert(
        !kExperimentalSplitVBackward ||
            (kPublishRepresentedBackwardFp8 && kPerBlockQkScales &&
             kInterleaveCausalKv),
        "split-V backward requires represented per-block causal MX publication"
    );
    static_assert(
        !kExperimentalE4m3DerivedMxfp4V ||
            (kPublishRepresentedBackwardFp8 && kPerBlockQkScales &&
             kInterleaveCausalKv && !kExperimentalSplitVBackward),
        "E4M3-derived MXFP4 V requires represented per-block causal "
        "publication with direct backward E4M3 V"
    );
    static_assert(
        !kExperimentalOutputSharedSplitV ||
            (
                kCompactForwardOut && kQkDepth == 128 &&
                kCompactPublishesMxV &&
                (
                    (kPairedD64 && kPublishRepresentedBackwardFp8 &&
                     kPerBlockQkScales && kInterleaveCausalKv &&
                     kExperimentalSplitVBackward &&
                     !kExperimentalE4m3DerivedMxfp4V) ||
                    (!kPairedD64 && !kPublishRepresentedBackwardFp8 &&
                     kPerBlockQkScales && !kInterleaveCausalKv &&
                     !kExperimentalSplitVBackward &&
                     !kExperimentalE4m3DerivedMxfp4V)
                )
            ),
        "output-shared dual V requires compact D64 represented causal or "
        "ordinary D128 direct-accumulator E4M3 backward publication"
    );
    static_assert(
        !kExperimentalCommonRowscaleMxfp4V ||
            (kExperimentalOutputSharedSplitV &&
             kCompactPublishesMxBackwardV && kCompactPublishesMxV &&
             kCompactForwardOut && kQkDepth == 128 && !kPairedD64 &&
             !kPublishRepresentedBackwardFp8 && kPerBlockQkScales),
        "common-row MXFP4 V requires the shape-gated native-D128 compact "
        "MX-backward publication route"
    );
    const int rows = batch * seq_len;
    const int q_width = q_heads * kQkDepth;
    const int k_width = kv_heads * kQkDepth;
    const int v_width = kv_heads * kVDepth;
    const int total_width = q_width + k_width + v_width;
    const int logical_q_heads = q_heads * kLogicalHeadMultiplier;
    const int logical_kv_heads = kv_heads * kLogicalHeadMultiplier;
    const bool has_split_rope = rope_cos != nullptr || rope_sin != nullptr;
    const bool has_packed_rope = rope_packed != nullptr;
    const bool apply_rope = has_split_rope || has_packed_rope;
    const int hidden = static_cast<int>(input_fp4.size(1) * 2);
    const c10::cuda::CUDAGuard device_guard(input_fp4.device());
    if constexpr (!kCompactForwardOut || kValidateCompactContracts) {
    TORCH_CHECK(
        q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0,
        "unified QKV projection requires positive Hq/Hkv and Hq divisible "
        "by Hkv"
    );
    TORCH_CHECK(
        (rope_cos == nullptr) == (rope_sin == nullptr),
        "RoPE cosine and sine tables must be supplied together"
    );
    TORCH_CHECK(
        !has_split_rope || !has_packed_rope,
        "split and packed RoPE tables are mutually exclusive"
    );
    TORCH_CHECK(
        input_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            qkv_weight_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            input_fp4.is_cuda() && qkv_weight_fp4.is_cuda() &&
            input_fp4.is_contiguous() && qkv_weight_fp4.is_contiguous() &&
            input_fp4.dim() == 2 && qkv_weight_fp4.dim() == 2,
        "unified projection operands must be contiguous CUDA packed E2M1 "
        "matrices"
    );
    TORCH_CHECK(
        input_fp4.size(0) == rows &&
            qkv_weight_fp4.size(0) == total_width &&
            qkv_weight_fp4.size(1) == input_fp4.size(1) &&
            rows % 256 == 0 && seq_len % 256 == 0 &&
            total_width % 256 == 0 && q_width % 32 == 0 &&
            k_width % 32 == 0 && v_width % 32 == 0 &&
            hidden % 256 == 0,
        "unified NVFP4 projection requires A=[B*S,K/2], "
        "B=[Hq*D+Hkv*(D+128),K/2], S divisible by 256, supported head "
        "tiling, and K divisible by 256"
    );
    TORCH_CHECK(
        input_scales.scalar_type() == at::kFloat8_e4m3fn &&
            qkv_weight_scales.scalar_type() == at::kFloat8_e4m3fn &&
            input_scales.is_cuda() && qkv_weight_scales.is_cuda() &&
            input_scales.is_contiguous() &&
            qkv_weight_scales.is_contiguous() &&
            input_scales.dim() == 3 && qkv_weight_scales.dim() == 3 &&
            input_scales.size(0) == rows / 128 &&
            input_scales.size(1) == hidden / 64 &&
            input_scales.size(2) == 512 &&
            qkv_weight_scales.size(0) == total_width / 128 &&
            qkv_weight_scales.size(1) == hidden / 64 &&
            qkv_weight_scales.size(2) == 512 &&
            input_global_scale.scalar_type() == at::kFloat &&
            qkv_weight_global_scale.scalar_type() == at::kFloat &&
            input_global_scale.is_cuda() &&
            qkv_weight_global_scale.is_cuda() &&
            input_global_scale.numel() == 1 &&
            qkv_weight_global_scale.numel() == 1,
        "unified NVFP4 operands require contiguous E4M3 block scales "
        "A=[B*S/128,K/64,512], B=[N/128,K/64,512], and one float32 "
        "global scale each"
    );
    const bool has_adaptive_scales =
        adaptive_scales.scalar_type() == at::ScalarType::Float &&
        adaptive_scales.is_cuda() && adaptive_scales.is_contiguous() &&
        adaptive_scales.dim() == 3 &&
        adaptive_scales.size(0) == batch &&
        adaptive_scales.size(1) == q_heads &&
        adaptive_scales.size(2) == 7;
    const bool has_empty_pure_metadata =
        pure_qk_single_quant &&
        adaptive_scales.scalar_type() == at::ScalarType::Float &&
        adaptive_scales.is_cuda() && adaptive_scales.is_contiguous() &&
        adaptive_scales.numel() == 0;
    TORCH_CHECK(
        has_adaptive_scales || has_empty_pure_metadata,
        "adaptive_scales must be contiguous CUDA float32 [B, H, 7], or an "
        "empty CUDA float32 tensor for the fixed-scale pure-Q/K specialization"
    );
    kittens::py::device_check(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales
    );
    if (has_split_rope) {
        TORCH_CHECK(
            rope_cos->scalar_type() == at::ScalarType::BFloat16 &&
                rope_sin->scalar_type() == at::ScalarType::BFloat16 &&
                rope_cos->is_cuda() && rope_sin->is_cuda() &&
                rope_cos->is_contiguous() && rope_sin->is_contiguous() &&
                rope_cos->dim() == 3 && rope_sin->dim() == 3 &&
                rope_cos->size(0) == batch &&
                rope_cos->size(1) == seq_len &&
                rope_cos->size(2) == kQkDepth / 2 &&
                rope_sin->sizes() == rope_cos->sizes(),
            "pair-native RoPE tables must be contiguous CUDA BF16 "
            "[B, S, QK_DEPTH / 2]"
        );
        kittens::py::device_check(input_fp4, *rope_cos, *rope_sin);
    }
    if (has_packed_rope) {
        TORCH_CHECK(
            kQkDepth == 128 &&
                rope_packed->scalar_type() == at::ScalarType::Int &&
                rope_packed->is_cuda() && rope_packed->is_contiguous() &&
                rope_packed->dim() == 3 &&
                rope_packed->size(0) == batch &&
                rope_packed->size(1) == seq_len &&
                rope_packed->size(2) == kQkDepth / 2,
            "packed D128 RoPE must be contiguous CUDA int32 [B, S, 64]"
        );
        kittens::py::device_check(input_fp4, *rope_packed);
    }
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "unified projection FP4 epilogue requires GB200 / SM100"
    );
    TORCH_CHECK(
        !pure_qk_single_quant || publish_pure_qk,
        "pure_qk_single_quant requires publish_pure_qk=true"
    );
    TORCH_CHECK(
        !publish_fp8_backward ||
            (!publish_pure_qk && !pure_qk_single_quant),
        "FP8-backward publication requires the retained hybrid QKV "
        "epilogue without pure-Q/K publication"
    );
    TORCH_CHECK(
        !per_block_qk_scales ||
            ((kPerBlockQkScales && kPublishRepresentedBackwardFp8) ||
             (kQkDepth == 128 && !kPairedD64 &&
              (publish_fp8_backward || kCompactPublishesMxBackwardV) &&
              !publish_pure_qk && !pure_qk_single_quant)),
        "per-block Q/K scales require either represented paired-D64 or the "
        "native D128 NVFP4 backward publication"
    );
    if constexpr (kPerBlockQkScales) {
        TORCH_CHECK(
            per_block_qk_scales,
            "represented native NVFP4 specialization requires per-block "
            "Q/K scales"
        );
    }
    if constexpr (kCompactForwardOut) {
        TORCH_CHECK(
            forward_outputs != nullptr,
            "compact paired D64 native NVFP4 projection requires "
            "caller-owned forward outputs"
        );
        TORCH_CHECK(
            rope_packed != nullptr,
            "compact paired D64 native NVFP4 projection requires packed "
            "RoPE input"
        );
        check_nvfp4_qkv_forward_outputs<kLogicalDepth>(
            *forward_outputs,
            input_fp4,
            input_scales,
            input_global_scale,
            qkv_weight_fp4,
            qkv_weight_scales,
            qkv_weight_global_scale,
            adaptive_scales,
            *rope_packed,
            batch,
            seq_len,
            logical_q_heads,
            logical_kv_heads
        );
    }
    }

    const auto bf16_options = input_fp4.options().dtype(at::kBFloat16);
    const auto byte_options = input_fp4.options().dtype(at::kByte);
    const auto fp8_options =
        input_fp4.options().dtype(at::kFloat8_e4m3fn);
    const auto fp4_options =
        input_fp4.options().dtype(at::kFloat4_e2m1fn_x2);
    const bool publish_qk_backward_fp8 =
        publish_fp8_backward || kPublishQkBackwardFp8;
    const bool publish_aligned_qk = kPairedD64
        ? store_bf16
        : (kQkDepth != 128 || !(publish_qk_backward_fp8 && !store_bf16));
    at::Tensor q;
    at::Tensor k;
    at::Tensor v;
    at::Tensor q_sequence_aligned;
    at::Tensor q_depth_packed;
    at::Tensor k_depth_aligned;
    at::Tensor k_depth_packed;
    at::Tensor q_sequence_compact;
    at::Tensor k_sequence_compact;
    at::Tensor q_sequence_scales;
    at::Tensor k_sequence_scales;
    at::Tensor q_forward_scales;
    at::Tensor k_forward_scales;
    at::Tensor q_forward_global_scale;
    at::Tensor k_forward_global_scale;
    at::Tensor v_mxfp4;
    at::Tensor v_mxfp4_scales;
    at::Tensor v_backward_mxfp4;
    at::Tensor v_backward_mxfp4_scales;
    at::Tensor v_backward_fp8;
    at::Tensor v_forward_fp8;
    at::Tensor q_backward_fp8;
    at::Tensor k_backward_fp8;
    if constexpr (kCompactForwardOut) {
        q_depth_packed = forward_outputs->q_depth_packed;
        k_depth_packed = forward_outputs->k_depth_packed;
        q_forward_scales = forward_outputs->q_forward_scales;
        q_forward_global_scale = forward_outputs->q_forward_global_scale;
        k_forward_scales = forward_outputs->k_forward_scales;
        k_forward_global_scale = forward_outputs->k_forward_global_scale;
        v_mxfp4 = forward_outputs->v_mxfp4;
        v_mxfp4_scales = forward_outputs->v_mxfp4_scales;
        v_forward_fp8 = forward_outputs->v_forward_fp8;
        v_backward_fp8 = forward_outputs->v_backward_fp8;
        q_backward_fp8 = forward_outputs->q_backward_fp8;
        k_backward_fp8 = forward_outputs->k_backward_fp8;
        if constexpr (kCompactPublishesMxBackwardV) {
            v_backward_mxfp4 = forward_outputs->v_backward_mxfp4;
            v_backward_mxfp4_scales =
                forward_outputs->v_backward_mxfp4_scales;
        }
    } else {
        q = at::empty(
            {batch, seq_len, logical_q_heads, kLogicalDepth},
            bf16_options
        );
        k = at::empty(
            {batch, seq_len, logical_kv_heads, kLogicalDepth},
            bf16_options
        );
        v = at::empty(
            {batch, seq_len, logical_kv_heads,
             kPairedD64 ? kLogicalDepth : kVDepth},
            bf16_options
        );
        q_sequence_aligned = publish_aligned_qk
            ? at::empty(
                  {batch, logical_q_heads, kLogicalDepth, seq_len},
                  byte_options
              )
            : at::empty({0}, byte_options);
        q_depth_packed = at::empty(
            {batch, logical_q_heads, seq_len, kLogicalDepth / 2},
            byte_options
        );
        k_depth_aligned = publish_aligned_qk
            ? at::empty(k.sizes(), byte_options)
            : at::empty({0}, byte_options);
        k_depth_packed = at::empty(
            {batch, logical_kv_heads, seq_len, kLogicalDepth / 2},
            byte_options
        );
        q_sequence_compact = publish_pure_qk
            ? at::empty(
                  {batch, q_heads, kQkDepth, seq_len / 2},
                  byte_options
              )
            : at::empty({0}, byte_options);
        k_sequence_compact = publish_pure_qk
            ? at::empty(
                  {batch, kv_heads, kQkDepth, seq_len / 2},
                  byte_options
              )
            : at::empty({0}, byte_options);
        // The retained pure-Q/K route is fixed-scale E2M1. Empty scale
        // tensors make that contract explicit in the legacy tuple.
        q_sequence_scales = at::empty({0}, fp8_options);
        k_sequence_scales = at::empty({0}, fp8_options);
        q_forward_scales = at::empty(
            {batch, seq_len / 128, q_heads * kQkChunks, 512},
            fp8_options
        );
        k_forward_scales = at::empty(
            {batch, seq_len / 64, kv_heads * kQkChunks, 512},
            fp8_options
        );
        q_forward_global_scale = at::empty(
            {batch, logical_q_heads},
            input_fp4.options().dtype(at::kFloat)
        );
        k_forward_global_scale = at::empty(
            {batch, logical_kv_heads},
            input_fp4.options().dtype(at::kFloat)
        );
        v_mxfp4 = at::empty(
            {batch, logical_kv_heads,
             kPairedD64 ? kLogicalDepth : kVDepth, seq_len / 2},
            fp4_options
        );
        v_mxfp4_scales = kQkDepth == 128
            ? at::empty(
                  {batch, seq_len / 128, logical_kv_heads, 512},
                  fp8_options
              )
            : at::empty(
                  {batch, kv_heads, kVScaleRows, seq_len / 128, 32, 16},
                  byte_options
              );
        v_backward_mxfp4 = publish_fp8_backward
            ? at::empty({0}, byte_options)
            : at::empty(
                  {batch, seq_len, kv_heads, kVDepth / 2},
                  byte_options
              );
        v_backward_mxfp4_scales = publish_fp8_backward
            ? at::empty({0}, byte_options)
            : at::empty(
                  {batch, seq_len / 128, kv_heads, 512},
                  byte_options
              );
        v_backward_fp8 = publish_fp8_backward
            ? at::empty(
                  {batch, seq_len, logical_kv_heads,
                   kPairedD64 ? kLogicalDepth : kVDepth},
                  fp8_options
              )
            : at::empty({0}, fp8_options);
        v_forward_fp8 = publish_fp8_backward && kPublishForwardFp8
            ? at::empty(
                  {batch, logical_kv_heads,
                   kPairedD64 ? kLogicalDepth : kVDepth, seq_len},
                  fp8_options
              )
            : at::empty({0}, fp8_options);
        q_backward_fp8 = publish_qk_backward_fp8 && kQkDepth == 128
            ? at::empty(
                  {batch, seq_len, logical_q_heads, kLogicalDepth},
                  fp8_options
              )
            : at::empty({0}, fp8_options);
        k_backward_fp8 = publish_qk_backward_fp8 && kQkDepth == 128
            ? at::empty(
                  {batch, seq_len, logical_kv_heads, kLogicalDepth},
                  fp8_options
              )
            : at::empty({0}, fp8_options);
    }

    // The current V publisher keeps a 32x32 BF16 fragment in static shared
    // memory.  Four load stages plus the output ring leave less static shared
    // memory than that fragment needs on SM100, so cudaFuncSetAttribute fails
    // before a D192 launch.  Three stages retain the same epilogue contract
    // and leave enough space for both the 1-D diagnostic and retained 2-D V
    // policies.  Paired D64 already used three stages.
    using C = tkfa4_projection::config<3, 4, kQkDepth>;
    using G = tkfa4_projection::globals<C>;
    // STORE_BF16=false leaves these descriptors as shape-only metadata. Point
    // them at a live operand so the compact ABI does not allocate unreachable
    // Q/K/V tensors solely to construct global-layout descriptors.
    kittens::bf16 *descriptor_ptr = reinterpret_cast<kittens::bf16 *>(
        input_fp4.data_ptr()
    );
    typename G::D_gl q_descriptor{
        descriptor_ptr,
        nullptr,
        nullptr,
        static_cast<size_t>(rows),
        static_cast<size_t>(q_width)
    };
    typename G::D_gl k_descriptor{
        descriptor_ptr,
        nullptr,
        nullptr,
        static_cast<size_t>(rows),
        static_cast<size_t>(k_width)
    };
    typename G::D_gl v_descriptor{
        descriptor_ptr,
        nullptr,
        nullptr,
        static_cast<size_t>(rows),
        static_cast<size_t>(v_width)
    };
    G globals{
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(
            input_fp4,
            1,
            1,
            rows,
            hidden / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            input_scales,
            1,
            input_scales.size(0),
            input_scales.size(1),
            256
        ),
        .A_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            input_global_scale
        ),
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(
            qkv_weight_fp4,
            1,
            1,
            total_width,
            hidden / 2
        ),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            qkv_weight_scales,
            1,
            qkv_weight_scales.size(0),
            qkv_weight_scales.size(1),
            256
        ),
        .B_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            qkv_weight_global_scale
        ),
        .Q = store_bf16
            ? kittens::py::tensor_to_gl<typename G::D_gl>(
                  q, 1, 1, rows, q_width
              )
            : q_descriptor,
        .K = store_bf16
            ? kittens::py::tensor_to_gl<typename G::D_gl>(
                  k, 1, 1, rows, k_width
              )
            : k_descriptor,
        .V = store_bf16
            ? kittens::py::tensor_to_gl<typename G::D_gl>(
                  v, 1, 1, rows, v_width
              )
            : v_descriptor,
        .D = store_bf16
            ? kittens::py::tensor_to_gl<typename G::D_gl>(
                  v, 1, 1, rows, v_width
              )
            : v_descriptor,
        .q_sequence_aligned = q_sequence_aligned.numel()
            ? reinterpret_cast<uint8_t *>(q_sequence_aligned.data_ptr())
            : nullptr,
        .q_depth_packed =
            reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
        .k_depth_aligned = k_depth_aligned.numel()
            ? reinterpret_cast<uint8_t *>(k_depth_aligned.data_ptr())
            : nullptr,
        .k_depth_packed =
            reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
        .q_sequence_compact = publish_pure_qk
            ? reinterpret_cast<uint8_t *>(q_sequence_compact.data_ptr())
            : nullptr,
        .k_sequence_compact = publish_pure_qk
            ? reinterpret_cast<uint8_t *>(k_sequence_compact.data_ptr())
            : nullptr,
        .q_forward_scales =
            reinterpret_cast<uint8_t *>(q_forward_scales.data_ptr()),
        .k_forward_scales =
            reinterpret_cast<uint8_t *>(k_forward_scales.data_ptr()),
        .q_forward_global_scale = reinterpret_cast<float *>(
            q_forward_global_scale.data_ptr()
        ),
        .k_forward_global_scale = reinterpret_cast<float *>(
            k_forward_global_scale.data_ptr()
        ),
        .v_mxfp4 = kPublishMxfp4V
            ? reinterpret_cast<uint8_t *>(v_mxfp4.data_ptr())
            : nullptr,
        .v_mxfp4_scales = kPublishMxfp4V
            ? reinterpret_cast<uint8_t *>(v_mxfp4_scales.data_ptr())
            : nullptr,
        .v_backward_mxfp4 = publish_fp8_backward
            ? nullptr
            : reinterpret_cast<uint8_t *>(v_backward_mxfp4.data_ptr()),
        .v_backward_mxfp4_scales = publish_fp8_backward
            ? nullptr
            : reinterpret_cast<uint8_t *>(
                  v_backward_mxfp4_scales.data_ptr()
              ),
        .v_backward_fp8 = publish_fp8_backward
            ? reinterpret_cast<uint8_t *>(v_backward_fp8.data_ptr())
            : nullptr,
        .v_forward_fp8 = v_forward_fp8.numel() &&
                publish_fp8_backward && kPublishForwardFp8
            ? reinterpret_cast<uint8_t *>(v_forward_fp8.data_ptr())
            : nullptr,
        .q_backward_fp8 = publish_qk_backward_fp8 && kQkDepth == 128
            ? reinterpret_cast<uint8_t *>(q_backward_fp8.data_ptr())
            : nullptr,
        .k_backward_fp8 = publish_qk_backward_fp8 && kQkDepth == 128
            ? reinterpret_cast<uint8_t *>(k_backward_fp8.data_ptr())
            : nullptr,
        .adaptive_scales =
            reinterpret_cast<const float *>(adaptive_scales.data_ptr()),
        .rope_cos = has_split_rope
            ? reinterpret_cast<const kittens::bf16 *>(rope_cos->data_ptr())
            : nullptr,
        .rope_sin = has_split_rope
            ? reinterpret_cast<const kittens::bf16 *>(rope_sin->data_ptr())
            : nullptr,
        .rope_packed = has_packed_rope
            ? reinterpret_cast<const uint32_t *>(rope_packed->data_ptr())
            : nullptr,
        .batch = batch,
        .seq_len = seq_len,
        .heads = q_heads,
        .head_depth = kLogicalDepth,
        .paired_d64 = kPairedD64,
        .v_width = v_width,
        .v_scale_rows = kVScaleRows,
        .v_mxfp4_scale_2d = v_mxfp4_scale_2d,
        .cluster_cap = cluster_cap,
    };
    auto launch = [&]<
        bool ApplyRope,
        bool PackedRope,
        bool SharedPackedRope,
        bool CacheAdaptiveQkScale,
        bool PerBlockQkScales
    >() {
    if constexpr (kExperimentalSharedTileMxfp4V) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, false, false, false, false,
                false,
                ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV,
                false,
                true
            >
        >(globals);
    } else if constexpr (kExperimentalCommonRowscaleMxfp4V) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, false, false, false, false,
                false,
                ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV,
                true
            >
        >(globals);
    } else if (publish_fp8_backward) {
        if constexpr (kQkDepth == 128) {
        if (store_bf16) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, true, false, false, false,
                false, ApplyRope, true, false, true, true, true, true,
                PackedRope,
                SharedPackedRope, CacheAdaptiveQkScale, false, false,
                kInterleaveCausalKv, kPublishForwardFp8,
                kPublishRepresentedBackwardFp8, PerBlockQkScales,
                kExperimentalSplitVBackward,
                kExperimentalE4m3DerivedMxfp4V,
                kExperimentalOutputSharedSplitV
            >
        >(globals);
        } else {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, false, false, false, false,
                false, ApplyRope, true, false, true, true, true, false,
                PackedRope,
                SharedPackedRope, CacheAdaptiveQkScale, false, false,
                kInterleaveCausalKv, kPublishForwardFp8,
                kPublishRepresentedBackwardFp8, PerBlockQkScales,
                kExperimentalSplitVBackward,
                kExperimentalE4m3DerivedMxfp4V,
                kExperimentalOutputSharedSplitV
            >
        >(globals);
        }
        } else {
        if (store_bf16) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, true, false, false, false,
                false, ApplyRope, true, false
            >
        >(globals);
        } else {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, false, false, false, false,
                false, ApplyRope, true, false
            >
        >(globals);
        }
        }
    } else if (store_bf16 && pure_qk_single_quant) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, true, false, true, true,
                false,
                ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV
            >
        >(globals);
    } else if (pure_qk_single_quant) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, false, false, true, true,
                false,
                ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV
            >
        >(globals);
    } else if (store_bf16 && publish_pure_qk) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, true, false, true, false,
                false,
                ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV
            >
        >(globals);
    } else if (store_bf16) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, true, false, false, false,
                false,
                ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV
            >
        >(globals);
    } else if (publish_pure_qk) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, false, false, true, false,
                false,
                ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV
            >
        >(globals);
    } else {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C, true, true, kPublishMxfp4V, false, false, false, false,
                false,
                ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV
            >
        >(globals);
    }
    };
    auto launch_selected = [&]<
        bool ApplyRope,
        bool PackedRope,
        bool SharedPackedRope,
        bool CacheAdaptiveQkScale
    >() {
        if constexpr (kPerBlockQkScales) {
            launch.template operator()<
                ApplyRope,
                PackedRope,
                SharedPackedRope,
                CacheAdaptiveQkScale,
                true
            >();
        } else if constexpr (kQkDepth == 128 && !kPairedD64) {
            if (per_block_qk_scales) {
                launch.template operator()<
                    ApplyRope,
                    PackedRope,
                    SharedPackedRope,
                    CacheAdaptiveQkScale,
                    true
                >();
            } else {
                launch.template operator()<
                    ApplyRope,
                    PackedRope,
                    SharedPackedRope,
                    CacheAdaptiveQkScale,
                    false
                >();
            }
        } else {
            launch.template operator()<
                ApplyRope,
                PackedRope,
                SharedPackedRope,
                CacheAdaptiveQkScale,
                false
            >();
        }
    };
    if (has_packed_rope) {
        if (cache_packed_rope) {
            if (cache_adaptive_qk_scale) {
                launch_selected.template operator()<true, true, true, true>();
            } else {
                launch_selected.template operator()<true, true, true, false>();
            }
        } else {
            launch_selected.template operator()<true, true, false, false>();
        }
    } else if (apply_rope) {
        launch_selected.template operator()<true, false, false, false>();
    } else {
        launch_selected.template operator()<false, false, false, false>();
    }
    if constexpr (kCompactForwardOut) {
        if constexpr (kCompactPublishesMxBackwardV) {
            return {
                v_backward_mxfp4,
                v_backward_mxfp4_scales,
                q_backward_fp8,
                k_backward_fp8
            };
        }
        return {v_backward_fp8, q_backward_fp8, k_backward_fp8};
    }
    return {
        q,
        k,
        v,
        q_sequence_aligned,
        q_depth_packed,
        k_depth_aligned,
        k_depth_packed,
        adaptive_scales,
        q_forward_scales,
        q_forward_global_scale,
        k_forward_scales,
        k_forward_global_scale,
        v_mxfp4,
        v_mxfp4_scales,
        v_backward_mxfp4,
        v_backward_mxfp4_scales,
        q_sequence_compact,
        k_sequence_compact,
        q_sequence_scales,
        k_sequence_scales,
        v_backward_fp8,
        q_backward_fp8,
        k_backward_fp8,
        v_forward_fp8
    };
}

std::vector<at::Tensor> project_qkv_unified_fp4_nvfp4(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    int batch,
    int seq_len,
    int heads,
    bool store_bf16,
    bool publish_pure_qk,
    bool pure_qk_single_quant,
    bool publish_fp8_backward
) {
    return project_qkv_unified_fp4_nvfp4_impl(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        heads,
        heads,
        store_bf16,
        publish_pure_qk,
        pure_qk_single_quant,
        publish_fp8_backward,
        nullptr,
        nullptr,
        nullptr
    );
}

std::vector<at::Tensor> project_qkv_unified_fp4_nvfp4_rope(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    int batch,
    int seq_len,
    int heads,
    bool store_bf16,
    bool publish_pure_qk,
    bool pure_qk_single_quant,
    bool publish_fp8_backward
) {
    return project_qkv_unified_fp4_nvfp4_impl(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        heads,
        heads,
        store_bf16,
        publish_pure_qk,
        pure_qk_single_quant,
        publish_fp8_backward,
        &rope_cos,
        &rope_sin,
        nullptr
    );
}

std::vector<at::Tensor> project_qkv_gqa_d128_unified_fp4_nvfp4(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d,
    bool per_block_qk_scales
) {
    return project_qkv_unified_fp4_nvfp4_impl<128>(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        store_bf16,
        false,
        false,
        publish_fp8_backward,
        nullptr,
        nullptr,
        nullptr,
        0,
        false,
        false,
        v_mxfp4_scale_2d,
        per_block_qk_scales
    );
}

std::vector<at::Tensor> project_qkv_gqa_d128_unified_fp4_nvfp4_rope(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d,
    bool per_block_qk_scales
) {
    return project_qkv_unified_fp4_nvfp4_impl<128>(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        store_bf16,
        false,
        false,
        publish_fp8_backward,
        &rope_cos,
        &rope_sin,
        nullptr,
        0,
        false,
        false,
        v_mxfp4_scale_2d,
        per_block_qk_scales
    );
}

std::vector<at::Tensor> project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d,
    bool per_block_qk_scales
) {
    return project_qkv_unified_fp4_nvfp4_impl<128>(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        store_bf16,
        false,
        false,
        publish_fp8_backward,
        nullptr,
        nullptr,
        &rope_packed,
        0,
        false,
        false,
        v_mxfp4_scale_2d,
        per_block_qk_scales
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d
) {
    TORCH_CHECK(
        q_heads > 0 && kv_heads > 0 && q_heads % 2 == 0 &&
            kv_heads % 2 == 0 && q_heads % kv_heads == 0,
        "paired D64 projection requires positive even Hq/Hkv and Hq "
        "divisible by Hkv"
    );
    return project_qkv_unified_fp4_nvfp4_impl<128, true>(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        q_heads / 2,
        kv_heads / 2,
        store_bf16,
        false,
        false,
        publish_fp8_backward,
        nullptr,
        nullptr,
        &rope_packed,
        0,
        false,
        false,
        v_mxfp4_scale_2d
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d
) {
    TORCH_CHECK(
        q_heads > 0 && kv_heads > 0 && q_heads % 2 == 0 &&
            kv_heads % 2 == 0 && q_heads % kv_heads == 0,
        "paired D64 projection requires positive even Hq/Hkv and Hq "
        "divisible by Hkv"
    );
    TORCH_CHECK(
        publish_fp8_backward,
        "interleaved causal D64 publication currently requires the FP8 "
        "backward operands"
    );
    return project_qkv_unified_fp4_nvfp4_impl<128, true, true>(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        q_heads / 2,
        kv_heads / 2,
        store_bf16,
        false,
        false,
        publish_fp8_backward,
        nullptr,
        nullptr,
        &rope_packed,
        0,
        false,
        false,
        v_mxfp4_scale_2d
    );
}

template <bool InterleaveCausalKv, bool ExperimentalSplitVBackward>
std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp4_nvfp4_represented_perblock_impl(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d
) {
    TORCH_CHECK(
        q_heads > 0 && kv_heads > 0 && q_heads % 2 == 0 &&
            kv_heads % 2 == 0 && q_heads % kv_heads == 0,
        "represented paired D64 native NVFP4 projection requires positive "
        "even Hq/Hkv and Hq divisible by Hkv"
    );
    TORCH_CHECK(
        !store_bf16 && publish_fp8_backward,
        "represented paired D64 native NVFP4 projection requires "
        "store_bf16=false and publish_fp8_backward=true"
    );
    return project_qkv_unified_fp4_nvfp4_impl<
        128,
        true,
        InterleaveCausalKv,
        false,
        false,
        true,
        true,
        ExperimentalSplitVBackward
    >(
        std::move(input_fp4),
        std::move(input_scales),
        std::move(input_global_scale),
        std::move(qkv_weight_fp4),
        std::move(qkv_weight_scales),
        std::move(qkv_weight_global_scale),
        std::move(adaptive_scales),
        batch,
        seq_len,
        q_heads / 2,
        kv_heads / 2,
        false,
        false,
        false,
        true,
        nullptr,
        nullptr,
        &rope_packed,
        0,
        false,
        false,
        v_mxfp4_scale_2d,
        true
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_represented_backward_perblock_qk(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d
) {
    return project_qkv_gqa_d64_paired_unified_fp4_nvfp4_represented_perblock_impl<
        false,
        false
    >(
        std::move(input_fp4),
        std::move(input_scales),
        std::move(input_global_scale),
        std::move(qkv_weight_fp4),
        std::move(qkv_weight_scales),
        std::move(qkv_weight_global_scale),
        std::move(adaptive_scales),
        std::move(rope_packed),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        store_bf16,
        publish_fp8_backward,
        v_mxfp4_scale_2d
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d
) {
    return project_qkv_gqa_d64_paired_unified_fp4_nvfp4_represented_perblock_impl<
        true,
        true
    >(
        std::move(input_fp4),
        std::move(input_scales),
        std::move(input_global_scale),
        std::move(qkv_weight_fp4),
        std::move(qkv_weight_scales),
        std::move(qkv_weight_global_scale),
        std::move(adaptive_scales),
        std::move(rope_packed),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        store_bf16,
        publish_fp8_backward,
        v_mxfp4_scale_2d
    );
}

template <
    bool InterleaveCausalKv,
    bool ValidateContracts,
    bool PublishRepresentedBackwardFp8 = false,
    bool PerBlockQkScales = false,
    bool ExperimentalSplitVBackward = false,
    bool ExperimentalE4m3DerivedMxfp4V = false,
    bool ExperimentalOutputSharedSplitV = false
>
std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp4_nvfp4_forward_out_impl(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool v_mxfp4_scale_2d,
    at::Tensor q_depth_packed_out,
    at::Tensor k_depth_packed_out,
    at::Tensor q_forward_scales_out,
    at::Tensor q_forward_global_scale_out,
    at::Tensor k_forward_scales_out,
    at::Tensor k_forward_global_scale_out,
    at::Tensor v_mxfp4_out,
    at::Tensor v_mxfp4_scales_out,
    at::Tensor v_forward_fp8_out,
    at::Tensor v_backward_fp8_out,
    at::Tensor q_backward_fp8_out,
    at::Tensor k_backward_fp8_out
) {
    if constexpr (ValidateContracts) {
        TORCH_CHECK(
            q_heads > 0 && kv_heads > 0 && q_heads % 2 == 0 &&
                kv_heads % 2 == 0 && q_heads % kv_heads == 0,
            "paired D64 native NVFP4 projection requires positive even "
            "Hq/Hkv and Hq divisible by Hkv"
        );
        if constexpr (ExperimentalOutputSharedSplitV) {
            TORCH_CHECK(
                batch == 16 && seq_len == 4096 && q_heads == 32 &&
                    kv_heads == 8 && input_fp4.dim() == 2 &&
                    input_fp4.size(1) == 1024 && !v_mxfp4_scale_2d,
                "output-shared split-V is authenticated only for "
                "B16/S4096/H2048/Hq32/Hkv8/D64 with rowwise MXFP4 V scales"
            );
        }
    }
    paired_d64_nvfp4_forward_outputs outputs{
        .q_depth_packed = std::move(q_depth_packed_out),
        .k_depth_packed = std::move(k_depth_packed_out),
        .q_forward_scales = std::move(q_forward_scales_out),
        .q_forward_global_scale = std::move(q_forward_global_scale_out),
        .k_forward_scales = std::move(k_forward_scales_out),
        .k_forward_global_scale = std::move(k_forward_global_scale_out),
        .v_mxfp4 = std::move(v_mxfp4_out),
        .v_mxfp4_scales = std::move(v_mxfp4_scales_out),
        .v_forward_fp8 = std::move(v_forward_fp8_out),
        .v_backward_fp8 = std::move(v_backward_fp8_out),
        .q_backward_fp8 = std::move(q_backward_fp8_out),
        .k_backward_fp8 = std::move(k_backward_fp8_out),
    };
    return project_qkv_unified_fp4_nvfp4_impl<
        128,
        true,
        InterleaveCausalKv,
        true,
        ValidateContracts,
        PublishRepresentedBackwardFp8,
        PerBlockQkScales,
        ExperimentalSplitVBackward,
        ExperimentalE4m3DerivedMxfp4V,
        ExperimentalOutputSharedSplitV
    >(
        std::move(input_fp4),
        std::move(input_scales),
        std::move(input_global_scale),
        std::move(qkv_weight_fp4),
        std::move(qkv_weight_scales),
        std::move(qkv_weight_global_scale),
        std::move(adaptive_scales),
        batch,
        seq_len,
        q_heads / 2,
        kv_heads / 2,
        false,
        false,
        false,
        true,
        nullptr,
        nullptr,
        &rope_packed,
        0,
        false,
        false,
        v_mxfp4_scale_2d,
        PerBlockQkScales,
        &outputs
    );
}

#define TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(                                \
    NAME, INTERLEAVE, VALIDATE, REPRESENTED, PER_BLOCK, SPLIT_V, DERIVED_MX, \
    OUTPUT_SHARED_SPLIT_V                                                   \
)                                                                           \
std::vector<at::Tensor> NAME(                                               \
    at::Tensor input_fp4,                                                   \
    at::Tensor input_scales,                                                \
    at::Tensor input_global_scale,                                          \
    at::Tensor qkv_weight_fp4,                                              \
    at::Tensor qkv_weight_scales,                                           \
    at::Tensor qkv_weight_global_scale,                                     \
    at::Tensor adaptive_scales,                                             \
    at::Tensor rope_packed,                                                 \
    int batch,                                                              \
    int seq_len,                                                            \
    int q_heads,                                                            \
    int kv_heads,                                                           \
    bool v_mxfp4_scale_2d,                                                  \
    at::Tensor q_depth_packed_out,                                          \
    at::Tensor k_depth_packed_out,                                          \
    at::Tensor q_forward_scales_out,                                        \
    at::Tensor q_forward_global_scale_out,                                  \
    at::Tensor k_forward_scales_out,                                        \
    at::Tensor k_forward_global_scale_out,                                  \
    at::Tensor v_mxfp4_out,                                                 \
    at::Tensor v_mxfp4_scales_out,                                          \
    at::Tensor v_forward_fp8_out,                                           \
    at::Tensor v_backward_fp8_out,                                          \
    at::Tensor q_backward_fp8_out,                                          \
    at::Tensor k_backward_fp8_out                                           \
) {                                                                         \
    return project_qkv_gqa_d64_paired_unified_fp4_nvfp4_forward_out_impl<   \
        INTERLEAVE,                                                         \
        VALIDATE,                                                           \
        REPRESENTED,                                                        \
        PER_BLOCK,                                                          \
        SPLIT_V,                                                            \
        DERIVED_MX,                                                         \
        OUTPUT_SHARED_SPLIT_V                                               \
    >(                                                                      \
        std::move(input_fp4),                                               \
        std::move(input_scales),                                            \
        std::move(input_global_scale),                                      \
        std::move(qkv_weight_fp4),                                          \
        std::move(qkv_weight_scales),                                       \
        std::move(qkv_weight_global_scale),                                 \
        std::move(adaptive_scales),                                         \
        std::move(rope_packed),                                             \
        batch,                                                              \
        seq_len,                                                            \
        q_heads,                                                            \
        kv_heads,                                                           \
        v_mxfp4_scale_2d,                                                   \
        std::move(q_depth_packed_out),                                      \
        std::move(k_depth_packed_out),                                      \
        std::move(q_forward_scales_out),                                    \
        std::move(q_forward_global_scale_out),                              \
        std::move(k_forward_scales_out),                                    \
        std::move(k_forward_global_scale_out),                              \
        std::move(v_mxfp4_out),                                             \
        std::move(v_mxfp4_scales_out),                                      \
        std::move(v_forward_fp8_out),                                       \
        std::move(v_backward_fp8_out),                                      \
        std::move(q_backward_fp8_out),                                      \
        std::move(k_backward_fp8_out)                                       \
    );                                                                      \
}

TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_fp8_forward_out,
    false,
    true,
    false,
    false,
    false,
    false,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_fp8_forward_out_unchecked,
    false,
    false,
    false,
    false,
    false,
    false,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_mx_forward_out,
    true,
    true,
    false,
    false,
    false,
    false,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_mx_forward_out_unchecked,
    true,
    false,
    false,
    false,
    false,
    false,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_represented_backward_perblock_qk_fp8_forward_out,
    false,
    true,
    true,
    true,
    false,
    false,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_represented_backward_perblock_qk_fp8_forward_out_unchecked,
    false,
    false,
    true,
    true,
    false,
    false,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_mx_forward_out,
    true,
    true,
    true,
    true,
    true,
    false,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_mx_forward_out_unchecked,
    true,
    false,
    true,
    true,
    true,
    false,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_e4m3_derived_mx_forward_out,
    true,
    true,
    true,
    true,
    false,
    true,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_e4m3_derived_mx_forward_out_unchecked,
    true,
    false,
    true,
    true,
    false,
    true,
    false
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_output_shared_split_v_mx_forward_out,
    true,
    true,
    true,
    true,
    true,
    false,
    true
)
TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_output_shared_split_v_mx_forward_out_unchecked,
    true,
    false,
    true,
    true,
    true,
    false,
    true
)

#undef TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT

// Dense E4M3 QKV projection with rowwise activation decode and per-output-
// channel weight decode.  The tensor-core accumulator is dequantized in
// registers and flows directly through the retained paired-D64 RoPE and
// NVFP4-QK / E4M3-V publishers.  No BF16 QKV tensor is allocated or stored.
struct paired_d64_e4m3_forward_outputs {
    at::Tensor q_depth_packed;
    at::Tensor k_depth_packed;
    at::Tensor q_forward_scales;
    at::Tensor q_forward_global_scale;
    at::Tensor k_forward_scales;
    at::Tensor k_forward_global_scale;
    at::Tensor v_mxfp4;
    at::Tensor v_mxfp4_scales;
    at::Tensor v_forward_fp8;
    at::Tensor v_backward_fp8;
    at::Tensor q_backward_fp8;
    at::Tensor k_backward_fp8;
};

inline void check_paired_d64_e4m3_forward_output(
    const at::Tensor &output,
    at::ScalarType dtype,
    at::IntArrayRef shape,
    const c10::Device &device,
    const char *name
) {
    TORCH_CHECK(
        output.scalar_type() == dtype && output.is_cuda() &&
            output.is_contiguous() && output.device() == device &&
            output.sizes() == shape && output.storage_offset() == 0 &&
            !output.requires_grad(),
        name,
        " must be a base contiguous non-grad CUDA tensor with dtype ",
        dtype,
        " and shape ",
        shape,
        " on the projection device"
    );
}

inline void check_paired_d64_e4m3_forward_outputs(
    const paired_d64_e4m3_forward_outputs &outputs,
    const at::Tensor &input_fp8,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads
) {
    check_paired_d64_e4m3_forward_output(
        outputs.q_depth_packed,
        at::kByte,
        {batch, q_heads, seq_len, 32},
        input_fp8.device(),
        "q_depth_packed_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.k_depth_packed,
        at::kByte,
        {batch, kv_heads, seq_len, 32},
        input_fp8.device(),
        "k_depth_packed_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.q_forward_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 128, q_heads, 512},
        input_fp8.device(),
        "q_forward_scales_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.q_forward_global_scale,
        at::kFloat,
        {batch, q_heads},
        input_fp8.device(),
        "q_forward_global_scale_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.k_forward_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 64, kv_heads, 512},
        input_fp8.device(),
        "k_forward_scales_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.k_forward_global_scale,
        at::kFloat,
        {batch, kv_heads},
        input_fp8.device(),
        "k_forward_global_scale_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.v_mxfp4,
        at::kFloat4_e2m1fn_x2,
        {batch, kv_heads, 64, seq_len / 2},
        input_fp8.device(),
        "v_mxfp4_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.v_mxfp4_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 128, kv_heads, 512},
        input_fp8.device(),
        "v_mxfp4_scales_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.v_forward_fp8,
        at::kFloat8_e4m3fn,
        {batch, kv_heads, 64, seq_len},
        input_fp8.device(),
        "v_forward_fp8_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.v_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, kv_heads, 64},
        input_fp8.device(),
        "v_backward_fp8_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.q_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, q_heads, 64},
        input_fp8.device(),
        "q_backward_fp8_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.k_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, kv_heads, 64},
        input_fp8.device(),
        "k_backward_fp8_out"
    );
    const void *pointers[] = {
        outputs.q_depth_packed.data_ptr(),
        outputs.k_depth_packed.data_ptr(),
        outputs.q_forward_scales.data_ptr(),
        outputs.q_forward_global_scale.data_ptr(),
        outputs.k_forward_scales.data_ptr(),
        outputs.k_forward_global_scale.data_ptr(),
        outputs.v_mxfp4.data_ptr(),
        outputs.v_mxfp4_scales.data_ptr(),
        outputs.v_forward_fp8.data_ptr(),
        outputs.v_backward_fp8.data_ptr(),
        outputs.q_backward_fp8.data_ptr(),
        outputs.k_backward_fp8.data_ptr(),
    };
    for (int lhs = 0; lhs < 12; ++lhs) {
        for (int rhs = lhs + 1; rhs < 12; ++rhs) {
            TORCH_CHECK(
                pointers[lhs] != pointers[rhs],
                "paired D64 E4M3 caller-owned publications must use "
                "distinct base allocations"
            );
        }
    }
}

inline void check_d128_e4m3_forward_outputs(
    const paired_d64_e4m3_forward_outputs &outputs,
    const at::Tensor &input_fp8,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads
) {
    constexpr int kHeadDepth = 128;
    constexpr int kQkChunks = kHeadDepth / 64;
    check_paired_d64_e4m3_forward_output(
        outputs.q_depth_packed,
        at::kByte,
        {batch, q_heads, seq_len, kHeadDepth / 2},
        input_fp8.device(),
        "q_depth_packed_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.k_depth_packed,
        at::kByte,
        {batch, kv_heads, seq_len, kHeadDepth / 2},
        input_fp8.device(),
        "k_depth_packed_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.q_forward_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 128, q_heads * kQkChunks, 512},
        input_fp8.device(),
        "q_forward_scales_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.q_forward_global_scale,
        at::kFloat,
        {batch, q_heads},
        input_fp8.device(),
        "q_forward_global_scale_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.k_forward_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 64, kv_heads * kQkChunks, 512},
        input_fp8.device(),
        "k_forward_scales_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.k_forward_global_scale,
        at::kFloat,
        {batch, kv_heads},
        input_fp8.device(),
        "k_forward_global_scale_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.v_mxfp4,
        at::kFloat4_e2m1fn_x2,
        {batch, kv_heads, kHeadDepth, seq_len / 2},
        input_fp8.device(),
        "v_mxfp4_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.v_mxfp4_scales,
        at::kFloat8_e4m3fn,
        {batch, seq_len / 128, kv_heads, 512},
        input_fp8.device(),
        "v_mxfp4_scales_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.v_forward_fp8,
        at::kFloat8_e4m3fn,
        {batch, kv_heads, kHeadDepth, seq_len},
        input_fp8.device(),
        "v_forward_fp8_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.v_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, kv_heads, kHeadDepth},
        input_fp8.device(),
        "v_backward_fp8_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.q_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, q_heads, kHeadDepth},
        input_fp8.device(),
        "q_backward_fp8_out"
    );
    check_paired_d64_e4m3_forward_output(
        outputs.k_backward_fp8,
        at::kFloat8_e4m3fn,
        {batch, seq_len, kv_heads, kHeadDepth},
        input_fp8.device(),
        "k_backward_fp8_out"
    );
    const void *pointers[] = {
        outputs.q_depth_packed.data_ptr(),
        outputs.k_depth_packed.data_ptr(),
        outputs.q_forward_scales.data_ptr(),
        outputs.q_forward_global_scale.data_ptr(),
        outputs.k_forward_scales.data_ptr(),
        outputs.k_forward_global_scale.data_ptr(),
        outputs.v_mxfp4.data_ptr(),
        outputs.v_mxfp4_scales.data_ptr(),
        outputs.v_forward_fp8.data_ptr(),
        outputs.v_backward_fp8.data_ptr(),
        outputs.q_backward_fp8.data_ptr(),
        outputs.k_backward_fp8.data_ptr(),
    };
    for (int lhs = 0; lhs < 12; ++lhs) {
        for (int rhs = lhs + 1; rhs < 12; ++rhs) {
            TORCH_CHECK(
                pointers[lhs] != pointers[rhs],
                "native D128 E4M3 caller-owned publications must use "
                "distinct base allocations"
            );
        }
    }
}

template <
    bool PublishRepresentedBackwardFp8,
    bool PerBlockQkScales = false,
    bool ExperimentalSplitVBackward = false,
    bool CompactForwardOut = false,
    bool ValidateCompactContracts = false,
    bool NativeD128 = false
>
std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool publish_mxfp4_v,
    bool v_mxfp4_scale_2d,
    bool interleave_causal_kv,
    std::optional<at::Tensor> v_mxfp4_scales_out = std::nullopt,
    const paired_d64_e4m3_forward_outputs *forward_outputs = nullptr,
    int cluster_cap = 0
) {
    constexpr int kPhysicalDepth = 128;
    constexpr int kLogicalDepth = NativeD128 ? 128 : 64;
    constexpr int kQkChunks = kLogicalDepth / 64;
    static_assert(!ValidateCompactContracts || CompactForwardOut);
    static_assert(
        !NativeD128 ||
            (!PublishRepresentedBackwardFp8 && PerBlockQkScales &&
             !ExperimentalSplitVBackward),
        "native D128 E4M3 requires direct-accumulator backward Q/K/V and "
        "dynamic row-K16 forward Q/K scales"
    );
    if constexpr (ExperimentalSplitVBackward) {
        static_assert(
            PublishRepresentedBackwardFp8 && PerBlockQkScales,
            "split-V is defined only for represented per-block Q/K"
        );
    }
    const int rows = batch * seq_len;
    const int physical_q_heads = NativeD128 ? q_heads : q_heads / 2;
    const int physical_kv_heads = NativeD128 ? kv_heads : kv_heads / 2;
    const int q_width = physical_q_heads * kPhysicalDepth;
    const int k_width = physical_kv_heads * kPhysicalDepth;
    const int v_width = physical_kv_heads * kPhysicalDepth;
    const int total_width = q_width + k_width + v_width;
    int hidden;
    if constexpr (!CompactForwardOut || ValidateCompactContracts) {
        if constexpr (NativeD128) {
            TORCH_CHECK(
                q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0,
                "native D128 E4M3 projection requires positive Hq/Hkv and "
                "Hq divisible by Hkv"
            );
            TORCH_CHECK(
                !interleave_causal_kv,
                "native D128 E4M3 projection requires ordinary K/V order "
                "for both FP8-PV and MXFP4-PV"
            );
            TORCH_CHECK(
                cluster_cap >= 0,
                "native D128 E4M3 cluster_cap must be non-negative"
            );
        } else {
            TORCH_CHECK(
                q_heads > 0 && kv_heads > 0 && q_heads % 2 == 0 &&
                    kv_heads % 2 == 0 && q_heads % kv_heads == 0,
                "paired D64 E4M3 projection requires positive even Hq/Hkv "
                "and Hq divisible by Hkv"
            );
            TORCH_CHECK(
                interleave_causal_kv == publish_mxfp4_v,
                "paired D64 E4M3 projection requires normal K/V order for "
                "the exact-FP8 route and causal-interleaved K/MXFP4-V "
                "order for the MXFP4 route"
            );
        }
        if constexpr (ExperimentalSplitVBackward) {
            TORCH_CHECK(
                publish_mxfp4_v && interleave_causal_kv,
                "experimental split-V backward is available only for the "
                "causal-interleaved MXFP4-PV route"
            );
        }
        TORCH_CHECK(
            input_fp8.scalar_type() == at::kFloat8_e4m3fn &&
                qkv_weight_fp8.scalar_type() == at::kFloat8_e4m3fn &&
                input_fp8.is_cuda() && qkv_weight_fp8.is_cuda() &&
                input_fp8.is_contiguous() &&
                qkv_weight_fp8.is_contiguous() && input_fp8.dim() == 2 &&
                qkv_weight_fp8.dim() == 2,
            "dense E4M3 projection operands must be contiguous CUDA "
            "E4M3 matrices"
        );
        hidden = static_cast<int>(input_fp8.size(1));
        TORCH_CHECK(
            input_fp8.size(0) == rows &&
                qkv_weight_fp8.size(0) == total_width &&
                qkv_weight_fp8.size(1) == hidden && rows % 256 == 0 &&
                seq_len % 256 == 0 && hidden % 128 == 0 &&
                total_width % 256 == 0,
            "dense E4M3 projection requires A=[B*S,K], a matching packed "
            "QKV B matrix, S divisible by 256, K divisible by 128, and "
            "total output width divisible by 256"
        );
        TORCH_CHECK(
            input_row_decode.scalar_type() == at::kFloat &&
                qkv_weight_channel_decode.scalar_type() == at::kFloat &&
                input_row_decode.is_cuda() &&
                qkv_weight_channel_decode.is_cuda() &&
                input_row_decode.is_contiguous() &&
                qkv_weight_channel_decode.is_contiguous() &&
                input_row_decode.numel() == rows &&
                qkv_weight_channel_decode.numel() == total_width,
            "E4M3 projection decode scales must be contiguous CUDA float32 "
            "vectors with one value per input row and output channel"
        );
        TORCH_CHECK(
            adaptive_scales.scalar_type() == at::kFloat &&
                adaptive_scales.is_cuda() &&
                adaptive_scales.is_contiguous() &&
                adaptive_scales.dim() == 3 &&
                adaptive_scales.size(0) == batch &&
                adaptive_scales.size(1) == physical_q_heads &&
                adaptive_scales.size(2) == 7,
            "dense E4M3 adaptive scales must be contiguous CUDA float32 "
            "[B,physical_Hq,7]"
        );
        TORCH_CHECK(
            rope_packed.scalar_type() == at::kInt &&
                rope_packed.is_cuda() && rope_packed.is_contiguous() &&
                rope_packed.dim() == 3 && rope_packed.size(0) == batch &&
                rope_packed.size(1) == seq_len && rope_packed.size(2) == 64,
            "dense E4M3 RoPE must be contiguous CUDA int32 [B,S,64]"
        );
        kittens::py::device_check(
            input_fp8,
            input_row_decode,
            qkv_weight_fp8,
            qkv_weight_channel_decode,
            adaptive_scales,
            rope_packed
        );
    } else {
        hidden = static_cast<int>(input_fp8.size(1));
    }
    const c10::cuda::CUDAGuard device_guard(input_fp8.device());
    if constexpr (!CompactForwardOut || ValidateCompactContracts) {
        TORCH_CHECK(
            tkfa4::is_sm100_device(),
            "dense E4M3 projection requires GB200 / SM100"
        );
        if (v_mxfp4_scales_out.has_value()) {
            const at::Tensor &output = v_mxfp4_scales_out.value();
            TORCH_CHECK(
                publish_mxfp4_v,
                "a preallocated MXFP4 V-scale output requires "
                "publish_mxfp4_v=True"
            );
            TORCH_CHECK(
                output.scalar_type() == at::kFloat8_e4m3fn &&
                    output.is_cuda() && output.is_contiguous() &&
                    output.device() == input_fp8.device() &&
                    output.sizes() == at::IntArrayRef({
                        batch, seq_len / 128, kv_heads, 512
                    }),
                "preallocated MXFP4 V scales must be contiguous CUDA E4M3 "
                "[B,S/128,Hkv,512] on the projection device"
            );
        }
        if constexpr (CompactForwardOut) {
            TORCH_CHECK(
                forward_outputs != nullptr,
                "compact dense E4M3 projection requires caller-owned "
                "forward outputs"
            );
            if constexpr (NativeD128) {
                check_d128_e4m3_forward_outputs(
                    *forward_outputs,
                    input_fp8,
                    batch,
                    seq_len,
                    q_heads,
                    kv_heads
                );
            } else {
                check_paired_d64_e4m3_forward_outputs(
                    *forward_outputs,
                    input_fp8,
                    batch,
                    seq_len,
                    q_heads,
                    kv_heads
                );
            }
        }
    }

    const auto byte_options = input_fp8.options().dtype(at::kByte);
    const auto fp8_options = input_fp8.options().dtype(at::kFloat8_e4m3fn);
    const auto bf16_options = input_fp8.options().dtype(at::kBFloat16);
    const auto fp4_options =
        input_fp8.options().dtype(at::kFloat4_e2m1fn_x2);
    at::Tensor empty_bf16;
    at::Tensor empty_byte;
    at::Tensor empty_fp8;
    at::Tensor empty_fp4;
    at::Tensor q_depth_packed;
    at::Tensor k_depth_packed;
    at::Tensor q_forward_scales;
    at::Tensor q_forward_global_scale;
    at::Tensor k_forward_scales;
    at::Tensor k_forward_global_scale;
    at::Tensor v_mxfp4;
    at::Tensor v_mxfp4_scales;
    at::Tensor v_forward_fp8;
    at::Tensor v_backward_fp8;
    at::Tensor q_backward_fp8;
    at::Tensor k_backward_fp8;
    if constexpr (CompactForwardOut) {
        q_depth_packed = forward_outputs->q_depth_packed;
        k_depth_packed = forward_outputs->k_depth_packed;
        q_forward_scales = forward_outputs->q_forward_scales;
        q_forward_global_scale = forward_outputs->q_forward_global_scale;
        k_forward_scales = forward_outputs->k_forward_scales;
        k_forward_global_scale = forward_outputs->k_forward_global_scale;
        v_mxfp4 = forward_outputs->v_mxfp4;
        v_mxfp4_scales = forward_outputs->v_mxfp4_scales;
        v_forward_fp8 = forward_outputs->v_forward_fp8;
        v_backward_fp8 = forward_outputs->v_backward_fp8;
        q_backward_fp8 = forward_outputs->q_backward_fp8;
        k_backward_fp8 = forward_outputs->k_backward_fp8;
    } else {
        empty_bf16 = at::empty({0}, bf16_options);
        empty_byte = at::empty({0}, byte_options);
        empty_fp8 = at::empty({0}, fp8_options);
        empty_fp4 = at::empty({0}, fp4_options);
        q_depth_packed = at::empty(
            {batch, q_heads, seq_len, kLogicalDepth / 2},
            byte_options
        );
        k_depth_packed = at::empty(
            {batch, kv_heads, seq_len, kLogicalDepth / 2},
            byte_options
        );
        q_forward_scales = at::empty(
            {batch, seq_len / 128, q_heads * kQkChunks, 512},
            fp8_options
        );
        k_forward_scales = at::empty(
            {batch, seq_len / 64, kv_heads * kQkChunks, 512},
            fp8_options
        );
        q_forward_global_scale = at::empty(
            {batch, q_heads},
            input_fp8.options().dtype(at::kFloat)
        );
        k_forward_global_scale = at::empty(
            {batch, kv_heads},
            input_fp8.options().dtype(at::kFloat)
        );
        v_mxfp4 = publish_mxfp4_v
            ? at::empty(
                  {batch, kv_heads, kLogicalDepth, seq_len / 2},
                  fp4_options
              )
            : empty_fp4;
        v_mxfp4_scales = publish_mxfp4_v
            ? (
                  v_mxfp4_scales_out.has_value()
                      ? v_mxfp4_scales_out.value()
                      : at::empty(
                            {batch, seq_len / 128, kv_heads, 512},
                            fp8_options
                        )
              )
            : empty_fp8;
        v_forward_fp8 = publish_mxfp4_v
            ? empty_fp8
            : at::empty(
                  {batch, kv_heads, kLogicalDepth, seq_len},
                  fp8_options
              );
        v_backward_fp8 = at::empty(
            {batch, seq_len, kv_heads, kLogicalDepth},
            fp8_options
        );
        q_backward_fp8 = at::empty(
            {batch, seq_len, q_heads, kLogicalDepth},
            fp8_options
        );
        k_backward_fp8 = at::empty(
            {batch, seq_len, kv_heads, kLogicalDepth},
            fp8_options
        );
    }

    // K128 keeps dense FP8 A/B stages at 32 KiB apiece per cluster stage;
    // four stages plus the 64 KiB output ring remain within SM100 shared
    // memory while preserving enough lookahead to hide projection loads.
    using C = tkfa4_projection::config<4, 4, 128, 128, true>;
    using G = tkfa4_projection::globals<C>;
    at::Tensor descriptor_storage;
    kittens::bf16 *descriptor_ptr;
    if constexpr (CompactForwardOut) {
        // Dense E4M3 has STORE_BF16=false. The descriptor dimensions remain
        // live kernel metadata, but its data address is never dereferenced.
        // Reuse a live operand address instead of allocating a dummy tensor.
        descriptor_ptr = reinterpret_cast<kittens::bf16 *>(
            input_fp8.data_ptr()
        );
    } else {
        descriptor_storage = at::empty({1}, bf16_options);
        descriptor_ptr = reinterpret_cast<kittens::bf16 *>(
            descriptor_storage.data_ptr()
        );
    }
    typename G::D_gl q_descriptor{
        descriptor_ptr,
        nullptr,
        nullptr,
        static_cast<size_t>(rows),
        static_cast<size_t>(q_width)
    };
    typename G::D_gl k_descriptor{
        descriptor_ptr,
        nullptr,
        nullptr,
        static_cast<size_t>(rows),
        static_cast<size_t>(k_width)
    };
    typename G::D_gl v_descriptor{
        descriptor_ptr,
        nullptr,
        nullptr,
        static_cast<size_t>(rows),
        static_cast<size_t>(v_width)
    };
    typename G::A_sc_gl input_decode_descriptor{
        reinterpret_cast<float *>(input_row_decode.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        static_cast<size_t>(rows)
    };
    typename G::B_sc_gl weight_decode_descriptor{
        reinterpret_cast<float *>(qkv_weight_channel_decode.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        static_cast<size_t>(total_width)
    };
    // Dense E4M3 mode does not read the legacy tensor-wide FP4 scale
    // descriptors, but globals<C> retains them so the projection epilogue
    // has one ABI for both operand formats.  Point both at a live float32
    // tensor to satisfy the non-default-constructible GL type without
    // allocating or loading a standalone scale.
    typename G::scale_gl unused_global_scale_descriptor{
        reinterpret_cast<float *>(input_row_decode.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
    G globals{
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(
            input_fp8,
            1,
            1,
            rows,
            hidden
        ),
        .A_sc = input_decode_descriptor,
        .A_scale = unused_global_scale_descriptor,
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(
            qkv_weight_fp8,
            1,
            1,
            total_width,
            hidden
        ),
        .B_sc = weight_decode_descriptor,
        .B_scale = unused_global_scale_descriptor,
        .Q = q_descriptor,
        .K = k_descriptor,
        .V = v_descriptor,
        .D = v_descriptor,
        .q_sequence_aligned = nullptr,
        .q_depth_packed =
            reinterpret_cast<uint8_t *>(q_depth_packed.data_ptr()),
        .k_depth_aligned = nullptr,
        .k_depth_packed =
            reinterpret_cast<uint8_t *>(k_depth_packed.data_ptr()),
        .q_sequence_compact = nullptr,
        .k_sequence_compact = nullptr,
        .q_forward_scales =
            reinterpret_cast<uint8_t *>(q_forward_scales.data_ptr()),
        .k_forward_scales =
            reinterpret_cast<uint8_t *>(k_forward_scales.data_ptr()),
        .q_forward_global_scale = reinterpret_cast<float *>(
            q_forward_global_scale.data_ptr()
        ),
        .k_forward_global_scale = reinterpret_cast<float *>(
            k_forward_global_scale.data_ptr()
        ),
        .v_mxfp4 = publish_mxfp4_v
            ? reinterpret_cast<uint8_t *>(v_mxfp4.data_ptr())
            : nullptr,
        .v_mxfp4_scales = publish_mxfp4_v
            ? reinterpret_cast<uint8_t *>(v_mxfp4_scales.data_ptr())
            : nullptr,
        .v_backward_mxfp4 = nullptr,
        .v_backward_mxfp4_scales = nullptr,
        .v_backward_fp8 =
            reinterpret_cast<uint8_t *>(v_backward_fp8.data_ptr()),
        .v_forward_fp8 = !publish_mxfp4_v
            ? reinterpret_cast<uint8_t *>(v_forward_fp8.data_ptr())
            : nullptr,
        .q_backward_fp8 =
            reinterpret_cast<uint8_t *>(q_backward_fp8.data_ptr()),
        .k_backward_fp8 =
            reinterpret_cast<uint8_t *>(k_backward_fp8.data_ptr()),
        .adaptive_scales =
            reinterpret_cast<const float *>(adaptive_scales.data_ptr()),
        .rope_packed = reinterpret_cast<const uint32_t *>(
            rope_packed.data_ptr()
        ),
        .batch = batch,
        .seq_len = seq_len,
        .heads = physical_q_heads,
        .head_depth = kLogicalDepth,
        .paired_d64 = !NativeD128,
        .v_width = v_width,
        .v_scale_rows = 1,
        .v_mxfp4_scale_2d = v_mxfp4_scale_2d,
        .cluster_cap = cluster_cap,
    };
    auto launch = [&]<
        bool PublishMxfp4V,
        bool InterleaveCausalKv,
        bool PublishForwardFp8
    >() {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<
                C,
                true,           // PUBLISH_FP4 Q/K
                true,           // PUBLISH_FORWARD_QK
                PublishMxfp4V,  // PUBLISH_V_MXFP4
                false,          // STORE_BF16
                false,          // OUTPUT_IS_DOUT
                false,          // PUBLISH_PURE_QK
                false,          // PURE_QK_SINGLE_QUANT
                false,          // SINGLE_OUTPUT
                true,           // APPLY_ROPE
                true,           // PUBLISH_V_FP8
                false,          // PUBLISH_V_BACKWARD_MXFP4
                true,           // PUBLISH_DOUT_STATS (inactive here)
                true,           // PUBLISH_QK_FP8
                true,           // V_SEQUENCE_MAJOR_SCALES
                false,          // PUBLISH_ALIGNED_QK
                true,           // PACKED_ROPE
                false,          // SHARED_PACKED_ROPE
                false,          // CACHE_ADAPTIVE_QK_SCALE
                false,          // NEGATE_DOUT_STATS
                false,              // CLEAR_DQ
                InterleaveCausalKv, // INTERLEAVE_CAUSAL_KV
                PublishForwardFp8,  // PUBLISH_FORWARD_FP8
                PublishRepresentedBackwardFp8,
                PerBlockQkScales,
                ExperimentalSplitVBackward
            >
        >(globals);
    };
    if constexpr (ExperimentalSplitVBackward) {
        // Compile only the MX specialization: the kernel contract rejects
        // split-V for exact FP8, and the host check above makes that route
        // choice explicit before launch.
        launch.template operator()<true, true, false>();
    } else if (publish_mxfp4_v) {
        if constexpr (NativeD128) {
            launch.template operator()<true, false, false>();
        } else {
            launch.template operator()<true, true, false>();
        }
    } else {
        launch.template operator()<false, false, true>();
    }

    if constexpr (CompactForwardOut) {
        return {v_backward_fp8, q_backward_fp8, k_backward_fp8};
    } else {
        return {
            empty_bf16,                 // q
            empty_bf16,                 // k
            empty_bf16,                 // v
            empty_byte,                 // q_sequence_aligned
            q_depth_packed,
            empty_byte,                 // k_depth_aligned
            k_depth_packed,
            adaptive_scales,
            q_forward_scales,
            q_forward_global_scale,
            k_forward_scales,
            k_forward_global_scale,
            v_mxfp4,
            v_mxfp4_scales,
            empty_byte,                 // v_backward_mxfp4
            empty_byte,                 // v_backward_mxfp4_scales
            empty_byte,                 // q_sequence_compact
            empty_byte,                 // k_sequence_compact
            empty_fp8,                  // q_sequence_scales
            empty_fp8,                  // k_sequence_scales
            v_backward_fp8,
            q_backward_fp8,
            k_backward_fp8,
            v_forward_fp8
        };
    }
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool publish_mxfp4_v,
    bool v_mxfp4_scale_2d,
    bool interleave_causal_kv
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
        false,
        false
    >(
        input_fp8,
        input_row_decode,
        qkv_weight_fp8,
        qkv_weight_channel_decode,
        adaptive_scales,
        rope_packed,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        publish_mxfp4_v,
        v_mxfp4_scale_2d,
        interleave_causal_kv
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool publish_mxfp4_v,
    bool v_mxfp4_scale_2d,
    bool interleave_causal_kv
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
        true,
        false
    >(
        input_fp8,
        input_row_decode,
        qkv_weight_fp8,
        qkv_weight_channel_decode,
        adaptive_scales,
        rope_packed,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        publish_mxfp4_v,
        v_mxfp4_scale_2d,
        interleave_causal_kv
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool publish_mxfp4_v,
    bool v_mxfp4_scale_2d,
    bool interleave_causal_kv
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
        true,
        true
    >(
        input_fp8,
        input_row_decode,
        qkv_weight_fp8,
        qkv_weight_channel_decode,
        adaptive_scales,
        rope_packed,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        publish_mxfp4_v,
        v_mxfp4_scale_2d,
        interleave_causal_kv
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool publish_mxfp4_v,
    bool v_mxfp4_scale_2d,
    bool interleave_causal_kv
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
        true,
        true,
        true
    >(
        input_fp8,
        input_row_decode,
        qkv_weight_fp8,
        qkv_weight_channel_decode,
        adaptive_scales,
        rope_packed,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        publish_mxfp4_v,
        v_mxfp4_scale_2d,
        interleave_causal_kv
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_vscale_out(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool publish_mxfp4_v,
    bool v_mxfp4_scale_2d,
    bool interleave_causal_kv,
    at::Tensor v_mxfp4_scales_out
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
        true,
        true,
        true
    >(
        input_fp8,
        input_row_decode,
        qkv_weight_fp8,
        qkv_weight_channel_decode,
        adaptive_scales,
        rope_packed,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        publish_mxfp4_v,
        v_mxfp4_scale_2d,
        interleave_causal_kv,
        std::move(v_mxfp4_scales_out)
    );
}

template <bool Mxfp4Route, bool ValidateContracts>
inline std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_forward_out_impl(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool v_mxfp4_scale_2d,
    at::Tensor q_depth_packed_out,
    at::Tensor k_depth_packed_out,
    at::Tensor q_forward_scales_out,
    at::Tensor q_forward_global_scale_out,
    at::Tensor k_forward_scales_out,
    at::Tensor k_forward_global_scale_out,
    at::Tensor v_mxfp4_out,
    at::Tensor v_mxfp4_scales_out,
    at::Tensor v_forward_fp8_out,
    at::Tensor v_backward_fp8_out,
    at::Tensor q_backward_fp8_out,
    at::Tensor k_backward_fp8_out
) {
    paired_d64_e4m3_forward_outputs outputs{
        .q_depth_packed = std::move(q_depth_packed_out),
        .k_depth_packed = std::move(k_depth_packed_out),
        .q_forward_scales = std::move(q_forward_scales_out),
        .q_forward_global_scale = std::move(q_forward_global_scale_out),
        .k_forward_scales = std::move(k_forward_scales_out),
        .k_forward_global_scale = std::move(k_forward_global_scale_out),
        .v_mxfp4 = std::move(v_mxfp4_out),
        .v_mxfp4_scales = std::move(v_mxfp4_scales_out),
        .v_forward_fp8 = std::move(v_forward_fp8_out),
        .v_backward_fp8 = std::move(v_backward_fp8_out),
        .q_backward_fp8 = std::move(q_backward_fp8_out),
        .k_backward_fp8 = std::move(k_backward_fp8_out),
    };
    if constexpr (Mxfp4Route) {
        return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
            true,
            true,
            true,
            true,
            ValidateContracts
        >(
            input_fp8,
            input_row_decode,
            qkv_weight_fp8,
            qkv_weight_channel_decode,
            adaptive_scales,
            rope_packed,
            batch,
            seq_len,
            q_heads,
            kv_heads,
            true,
            v_mxfp4_scale_2d,
            true,
            std::nullopt,
            &outputs
        );
    } else {
        return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
            true,
            true,
            false,
            true,
            ValidateContracts
        >(
            input_fp8,
            input_row_decode,
            qkv_weight_fp8,
            qkv_weight_channel_decode,
            adaptive_scales,
            rope_packed,
            batch,
            seq_len,
            q_heads,
            kv_heads,
            false,
            v_mxfp4_scale_2d,
            false,
            std::nullopt,
            &outputs
        );
    }
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_fp8_forward_out(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool v_mxfp4_scale_2d,
    at::Tensor q_depth_packed_out,
    at::Tensor k_depth_packed_out,
    at::Tensor q_forward_scales_out,
    at::Tensor q_forward_global_scale_out,
    at::Tensor k_forward_scales_out,
    at::Tensor k_forward_global_scale_out,
    at::Tensor v_mxfp4_out,
    at::Tensor v_mxfp4_scales_out,
    at::Tensor v_forward_fp8_out,
    at::Tensor v_backward_fp8_out,
    at::Tensor q_backward_fp8_out,
    at::Tensor k_backward_fp8_out
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_forward_out_impl<
        false,
        true
    >(
        std::move(input_fp8),
        std::move(input_row_decode),
        std::move(qkv_weight_fp8),
        std::move(qkv_weight_channel_decode),
        std::move(adaptive_scales),
        std::move(rope_packed),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        v_mxfp4_scale_2d,
        std::move(q_depth_packed_out),
        std::move(k_depth_packed_out),
        std::move(q_forward_scales_out),
        std::move(q_forward_global_scale_out),
        std::move(k_forward_scales_out),
        std::move(k_forward_global_scale_out),
        std::move(v_mxfp4_out),
        std::move(v_mxfp4_scales_out),
        std::move(v_forward_fp8_out),
        std::move(v_backward_fp8_out),
        std::move(q_backward_fp8_out),
        std::move(k_backward_fp8_out)
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_fp8_forward_out_unchecked(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool v_mxfp4_scale_2d,
    at::Tensor q_depth_packed_out,
    at::Tensor k_depth_packed_out,
    at::Tensor q_forward_scales_out,
    at::Tensor q_forward_global_scale_out,
    at::Tensor k_forward_scales_out,
    at::Tensor k_forward_global_scale_out,
    at::Tensor v_mxfp4_out,
    at::Tensor v_mxfp4_scales_out,
    at::Tensor v_forward_fp8_out,
    at::Tensor v_backward_fp8_out,
    at::Tensor q_backward_fp8_out,
    at::Tensor k_backward_fp8_out
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_forward_out_impl<
        false,
        false
    >(
        std::move(input_fp8),
        std::move(input_row_decode),
        std::move(qkv_weight_fp8),
        std::move(qkv_weight_channel_decode),
        std::move(adaptive_scales),
        std::move(rope_packed),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        v_mxfp4_scale_2d,
        std::move(q_depth_packed_out),
        std::move(k_depth_packed_out),
        std::move(q_forward_scales_out),
        std::move(q_forward_global_scale_out),
        std::move(k_forward_scales_out),
        std::move(k_forward_global_scale_out),
        std::move(v_mxfp4_out),
        std::move(v_mxfp4_scales_out),
        std::move(v_forward_fp8_out),
        std::move(v_backward_fp8_out),
        std::move(q_backward_fp8_out),
        std::move(k_backward_fp8_out)
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_mx_forward_out(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool v_mxfp4_scale_2d,
    at::Tensor q_depth_packed_out,
    at::Tensor k_depth_packed_out,
    at::Tensor q_forward_scales_out,
    at::Tensor q_forward_global_scale_out,
    at::Tensor k_forward_scales_out,
    at::Tensor k_forward_global_scale_out,
    at::Tensor v_mxfp4_out,
    at::Tensor v_mxfp4_scales_out,
    at::Tensor v_forward_fp8_out,
    at::Tensor v_backward_fp8_out,
    at::Tensor q_backward_fp8_out,
    at::Tensor k_backward_fp8_out
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_forward_out_impl<
        true,
        true
    >(
        std::move(input_fp8),
        std::move(input_row_decode),
        std::move(qkv_weight_fp8),
        std::move(qkv_weight_channel_decode),
        std::move(adaptive_scales),
        std::move(rope_packed),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        v_mxfp4_scale_2d,
        std::move(q_depth_packed_out),
        std::move(k_depth_packed_out),
        std::move(q_forward_scales_out),
        std::move(q_forward_global_scale_out),
        std::move(k_forward_scales_out),
        std::move(k_forward_global_scale_out),
        std::move(v_mxfp4_out),
        std::move(v_mxfp4_scales_out),
        std::move(v_forward_fp8_out),
        std::move(v_backward_fp8_out),
        std::move(q_backward_fp8_out),
        std::move(k_backward_fp8_out)
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_mx_forward_out_unchecked(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool v_mxfp4_scale_2d,
    at::Tensor q_depth_packed_out,
    at::Tensor k_depth_packed_out,
    at::Tensor q_forward_scales_out,
    at::Tensor q_forward_global_scale_out,
    at::Tensor k_forward_scales_out,
    at::Tensor k_forward_global_scale_out,
    at::Tensor v_mxfp4_out,
    at::Tensor v_mxfp4_scales_out,
    at::Tensor v_forward_fp8_out,
    at::Tensor v_backward_fp8_out,
    at::Tensor q_backward_fp8_out,
    at::Tensor k_backward_fp8_out
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_forward_out_impl<
        true,
        false
    >(
        std::move(input_fp8),
        std::move(input_row_decode),
        std::move(qkv_weight_fp8),
        std::move(qkv_weight_channel_decode),
        std::move(adaptive_scales),
        std::move(rope_packed),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        v_mxfp4_scale_2d,
        std::move(q_depth_packed_out),
        std::move(k_depth_packed_out),
        std::move(q_forward_scales_out),
        std::move(q_forward_global_scale_out),
        std::move(k_forward_scales_out),
        std::move(k_forward_global_scale_out),
        std::move(v_mxfp4_out),
        std::move(v_mxfp4_scales_out),
        std::move(v_forward_fp8_out),
        std::move(v_backward_fp8_out),
        std::move(q_backward_fp8_out),
        std::move(k_backward_fp8_out)
    );
}

std::vector<at::Tensor>
project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool publish_mxfp4_v,
    bool v_mxfp4_scale_2d,
    int cluster_cap
) {
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
        false,  // direct projection-accumulator E4M3 backward Q/K/V
        true,   // dynamic row-K16 forward Q/K scales
        false,
        false,
        false,
        true    // native D128 geometry and ordinary K/V order
    >(
        std::move(input_fp8),
        std::move(input_row_decode),
        std::move(qkv_weight_fp8),
        std::move(qkv_weight_channel_decode),
        std::move(adaptive_scales),
        std::move(rope_packed),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        publish_mxfp4_v,
        v_mxfp4_scale_2d,
        false,
        std::nullopt,
        nullptr,
        cluster_cap
    );
}

template <bool PublishMxV, bool ValidateContracts>
std::vector<at::Tensor>
project_qkv_gqa_d128_e4m3_forward_out_impl(
    at::Tensor input_fp8,
    at::Tensor input_row_decode,
    at::Tensor qkv_weight_fp8,
    at::Tensor qkv_weight_channel_decode,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool v_mxfp4_scale_2d,
    int cluster_cap,
    at::Tensor q_depth_packed_out,
    at::Tensor k_depth_packed_out,
    at::Tensor q_forward_scales_out,
    at::Tensor q_forward_global_scale_out,
    at::Tensor k_forward_scales_out,
    at::Tensor k_forward_global_scale_out,
    at::Tensor v_mxfp4_out,
    at::Tensor v_mxfp4_scales_out,
    at::Tensor v_forward_fp8_out,
    at::Tensor v_backward_fp8_out,
    at::Tensor q_backward_fp8_out,
    at::Tensor k_backward_fp8_out
) {
    paired_d64_e4m3_forward_outputs outputs{
        .q_depth_packed = std::move(q_depth_packed_out),
        .k_depth_packed = std::move(k_depth_packed_out),
        .q_forward_scales = std::move(q_forward_scales_out),
        .q_forward_global_scale = std::move(q_forward_global_scale_out),
        .k_forward_scales = std::move(k_forward_scales_out),
        .k_forward_global_scale = std::move(k_forward_global_scale_out),
        .v_mxfp4 = std::move(v_mxfp4_out),
        .v_mxfp4_scales = std::move(v_mxfp4_scales_out),
        .v_forward_fp8 = std::move(v_forward_fp8_out),
        .v_backward_fp8 = std::move(v_backward_fp8_out),
        .q_backward_fp8 = std::move(q_backward_fp8_out),
        .k_backward_fp8 = std::move(k_backward_fp8_out),
    };
    return project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_impl<
        false,  // never derive backward Q/K from represented NVFP4 codes
        true,   // one dynamic scale per logical row x K16 block
        false,  // no represented/split-V publication
        true,
        ValidateContracts,
        true    // native D128, not paired D64
    >(
        std::move(input_fp8),
        std::move(input_row_decode),
        std::move(qkv_weight_fp8),
        std::move(qkv_weight_channel_decode),
        std::move(adaptive_scales),
        std::move(rope_packed),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        PublishMxV,
        v_mxfp4_scale_2d,
        false,  // D128 forward and backward both use ordinary K/V order
        std::nullopt,
        &outputs,
        cluster_cap
    );
}

#define TKFA4_DEFINE_D128_E4M3_FORWARD_OUT(NAME, PUBLISH_MX, VALIDATE)     \
std::vector<at::Tensor> NAME(                                              \
    at::Tensor input_fp8,                                                  \
    at::Tensor input_row_decode,                                           \
    at::Tensor qkv_weight_fp8,                                             \
    at::Tensor qkv_weight_channel_decode,                                  \
    at::Tensor adaptive_scales,                                            \
    at::Tensor rope_packed,                                                \
    int batch,                                                             \
    int seq_len,                                                           \
    int q_heads,                                                           \
    int kv_heads,                                                          \
    bool v_mxfp4_scale_2d,                                                 \
    int cluster_cap,                                                       \
    at::Tensor q_depth_packed_out,                                         \
    at::Tensor k_depth_packed_out,                                         \
    at::Tensor q_forward_scales_out,                                       \
    at::Tensor q_forward_global_scale_out,                                 \
    at::Tensor k_forward_scales_out,                                       \
    at::Tensor k_forward_global_scale_out,                                 \
    at::Tensor v_mxfp4_out,                                                \
    at::Tensor v_mxfp4_scales_out,                                         \
    at::Tensor v_forward_fp8_out,                                          \
    at::Tensor v_backward_fp8_out,                                         \
    at::Tensor q_backward_fp8_out,                                         \
    at::Tensor k_backward_fp8_out                                          \
) {                                                                        \
    return project_qkv_gqa_d128_e4m3_forward_out_impl<                    \
        PUBLISH_MX, VALIDATE                                               \
    >(                                                                     \
        std::move(input_fp8),                                              \
        std::move(input_row_decode),                                       \
        std::move(qkv_weight_fp8),                                         \
        std::move(qkv_weight_channel_decode),                              \
        std::move(adaptive_scales),                                        \
        std::move(rope_packed),                                            \
        batch,                                                             \
        seq_len,                                                           \
        q_heads,                                                           \
        kv_heads,                                                          \
        v_mxfp4_scale_2d,                                                  \
        cluster_cap,                                                       \
        std::move(q_depth_packed_out),                                     \
        std::move(k_depth_packed_out),                                     \
        std::move(q_forward_scales_out),                                   \
        std::move(q_forward_global_scale_out),                             \
        std::move(k_forward_scales_out),                                   \
        std::move(k_forward_global_scale_out),                             \
        std::move(v_mxfp4_out),                                            \
        std::move(v_mxfp4_scales_out),                                     \
        std::move(v_forward_fp8_out),                                      \
        std::move(v_backward_fp8_out),                                     \
        std::move(q_backward_fp8_out),                                     \
        std::move(k_backward_fp8_out)                                      \
    );                                                                     \
}

TKFA4_DEFINE_D128_E4M3_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_fp8_forward_out,
    false,
    true
)
TKFA4_DEFINE_D128_E4M3_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_fp8_forward_out_unchecked,
    false,
    false
)
TKFA4_DEFINE_D128_E4M3_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_mx_forward_out,
    true,
    true
)
TKFA4_DEFINE_D128_E4M3_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_mx_forward_out_unchecked,
    true,
    false
)

#undef TKFA4_DEFINE_D128_E4M3_FORWARD_OUT

std::vector<at::Tensor>
project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool v_mxfp4_scale_2d,
    bool per_block_qk_scales,
    int cluster_cap,
    bool cache_packed_rope,
    bool cache_adaptive_qk_scale
) {
    TORCH_CHECK(cluster_cap >= 0, "cluster_cap must be non-negative");
    return project_qkv_unified_fp4_nvfp4_impl<128>(
        input_fp4,
        input_scales,
        input_global_scale,
        qkv_weight_fp4,
        qkv_weight_scales,
        qkv_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        q_heads,
        kv_heads,
        store_bf16,
        false,
        false,
        publish_fp8_backward,
        nullptr,
        nullptr,
        &rope_packed,
        cluster_cap,
        cache_packed_rope,
        cache_adaptive_qk_scale,
        v_mxfp4_scale_2d,
        per_block_qk_scales
    );
}

template <
    bool PublishMxV,
    bool ValidateContracts,
    bool ExperimentalOutputSharedDualV = false,
    bool PublishMxBackwardV = false,
    bool ExperimentalCommonRowscaleMxBackwardV = false,
    bool ExperimentalSharedTileMxBackwardV = false,
    bool PublishRepresentedBackwardFp8 = false,
    bool PerBlockQkScales = ExperimentalOutputSharedDualV
>
std::vector<at::Tensor>
project_qkv_gqa_d128_nvfp4_forward_out_impl(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qkv_weight_fp4,
    at::Tensor qkv_weight_scales,
    at::Tensor qkv_weight_global_scale,
    at::Tensor adaptive_scales,
    at::Tensor rope_packed,
    int batch,
    int seq_len,
    int q_heads,
    int kv_heads,
    bool v_mxfp4_scale_2d,
    bool per_block_qk_scales,
    int cluster_cap,
    bool cache_packed_rope,
    bool cache_adaptive_qk_scale,
    at::Tensor q_depth_packed_out,
    at::Tensor k_depth_packed_out,
    at::Tensor q_forward_scales_out,
    at::Tensor q_forward_global_scale_out,
    at::Tensor k_forward_scales_out,
    at::Tensor k_forward_global_scale_out,
    at::Tensor v_mxfp4_out,
    at::Tensor v_mxfp4_scales_out,
    at::Tensor v_forward_fp8_out,
    at::Tensor v_backward_fp8_out,
    at::Tensor q_backward_fp8_out,
    at::Tensor k_backward_fp8_out,
    at::Tensor v_backward_mxfp4_out = at::Tensor(),
    at::Tensor v_backward_mxfp4_scales_out = at::Tensor()
) {
    static_assert(
        !ExperimentalOutputSharedDualV || PublishMxV,
        "D128 output-shared dual V requires the MXFP4 forward route"
    );
    static_assert(
        !PublishMxBackwardV || PublishMxV,
        "MX-only backward V requires the MX forward publisher"
    );
    static_assert(
        !ExperimentalCommonRowscaleMxBackwardV ||
            (ExperimentalOutputSharedDualV && PublishMxBackwardV),
        "common-row MX backward V requires the output-shared MX-only route"
    );
    static_assert(
        !ExperimentalSharedTileMxBackwardV ||
            (ExperimentalOutputSharedDualV && PublishMxBackwardV &&
             !ExperimentalCommonRowscaleMxBackwardV),
        "shared-tile MX backward V requires the output-shared MX-only route"
    );
    static_assert(
        !PublishRepresentedBackwardFp8 ||
            (!PublishMxV && !ExperimentalOutputSharedDualV &&
             !PublishMxBackwardV &&
             !ExperimentalCommonRowscaleMxBackwardV &&
             !ExperimentalSharedTileMxBackwardV && PerBlockQkScales),
        "represented-Q/K D128 backward is opt-in only for the per-block "
        "FP8-PV route"
    );
    if constexpr (ValidateContracts) {
        TORCH_CHECK(cluster_cap >= 0, "cluster_cap must be non-negative");
        if constexpr (PublishRepresentedBackwardFp8) {
            TORCH_CHECK(
                (batch == 1 || batch == 2) && seq_len == 4096 &&
                    q_heads == 32 && kv_heads == 8 &&
                    input_fp4.dim() == 2 && input_fp4.size(1) == 2048,
                "represented-Q/K D128 backward is authenticated only for "
                "B1/B2/S4096/H4096/Hq32/Hkv8/D128"
            );
        }
        if constexpr (PublishMxBackwardV) {
            check_paired_d64_nvfp4_forward_output(
                v_backward_mxfp4_out,
                at::kByte,
                {batch, seq_len, kv_heads, 64},
                input_fp4.device(),
                "v_backward_mxfp4_out"
            );
            check_paired_d64_nvfp4_forward_output(
                v_backward_mxfp4_scales_out,
                at::kByte,
                {batch, seq_len / 128, kv_heads, 512},
                input_fp4.device(),
                "v_backward_mxfp4_scales_out"
            );
            if constexpr (ExperimentalSharedTileMxBackwardV) {
                TORCH_CHECK(
                    v_mxfp4_scale_2d,
                    "shared-tile D128 MX backward V requires D32xS32 scales"
                );
            } else {
                TORCH_CHECK(
                    !v_mxfp4_scale_2d,
                    "D128 MX-only backward V requires rowwise 1x32 scales"
                );
            }
        }
        if constexpr (ExperimentalOutputSharedDualV) {
            TORCH_CHECK(
                (batch == 1 || batch == 2) && seq_len == 4096 &&
                    q_heads == 32 &&
                    kv_heads == 8 && input_fp4.dim() == 2 &&
                    input_fp4.size(1) == 2048 &&
                    (ExperimentalSharedTileMxBackwardV
                         ? v_mxfp4_scale_2d
                         : !v_mxfp4_scale_2d) &&
                    per_block_qk_scales,
                "D128 output-shared dual V is authenticated only for "
                "B1/B2/S4096/H4096/Hq32/Hkv8, the route-specific MXFP4 V "
                "scale policy, and per-row-K16 Q/K scales"
            );
        }
    }
    paired_d64_nvfp4_forward_outputs outputs{
        .q_depth_packed = std::move(q_depth_packed_out),
        .k_depth_packed = std::move(k_depth_packed_out),
        .q_forward_scales = std::move(q_forward_scales_out),
        .q_forward_global_scale = std::move(q_forward_global_scale_out),
        .k_forward_scales = std::move(k_forward_scales_out),
        .k_forward_global_scale = std::move(k_forward_global_scale_out),
        .v_mxfp4 = std::move(v_mxfp4_out),
        .v_mxfp4_scales = std::move(v_mxfp4_scales_out),
        .v_forward_fp8 = std::move(v_forward_fp8_out),
        .v_backward_fp8 = std::move(v_backward_fp8_out),
        .q_backward_fp8 = std::move(q_backward_fp8_out),
        .k_backward_fp8 = std::move(k_backward_fp8_out),
        .v_backward_mxfp4 = std::move(v_backward_mxfp4_out),
        .v_backward_mxfp4_scales =
            std::move(v_backward_mxfp4_scales_out),
    };
    return project_qkv_unified_fp4_nvfp4_impl<
        128,
        false,
        false,
        true,
        ValidateContracts,
        PublishRepresentedBackwardFp8,
        PerBlockQkScales,
        false,
        false,
        ExperimentalOutputSharedDualV,
        PublishMxV,
        PublishMxBackwardV,
        ExperimentalCommonRowscaleMxBackwardV,
        ExperimentalSharedTileMxBackwardV
    >(
        std::move(input_fp4),
        std::move(input_scales),
        std::move(input_global_scale),
        std::move(qkv_weight_fp4),
        std::move(qkv_weight_scales),
        std::move(qkv_weight_global_scale),
        std::move(adaptive_scales),
        batch,
        seq_len,
        q_heads,
        kv_heads,
        false,
        false,
        false,
        !PublishMxBackwardV,
        nullptr,
        nullptr,
        &rope_packed,
        cluster_cap,
        cache_packed_rope,
        cache_adaptive_qk_scale,
        v_mxfp4_scale_2d,
        per_block_qk_scales,
        &outputs
    );
}

#define TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(                               \
    NAME, PUBLISH_MX, VALIDATE, OUTPUT_SHARED_DUAL_V,                      \
    REPRESENTED_BACKWARD, PER_BLOCK_QK                                     \
)                                                                          \
std::vector<at::Tensor> NAME(                                              \
    at::Tensor input_fp4,                                                  \
    at::Tensor input_scales,                                               \
    at::Tensor input_global_scale,                                         \
    at::Tensor qkv_weight_fp4,                                             \
    at::Tensor qkv_weight_scales,                                          \
    at::Tensor qkv_weight_global_scale,                                    \
    at::Tensor adaptive_scales,                                            \
    at::Tensor rope_packed,                                                \
    int batch,                                                             \
    int seq_len,                                                           \
    int q_heads,                                                           \
    int kv_heads,                                                          \
    bool v_mxfp4_scale_2d,                                                 \
    bool per_block_qk_scales,                                              \
    int cluster_cap,                                                       \
    bool cache_packed_rope,                                                \
    bool cache_adaptive_qk_scale,                                          \
    at::Tensor q_depth_packed_out,                                         \
    at::Tensor k_depth_packed_out,                                         \
    at::Tensor q_forward_scales_out,                                       \
    at::Tensor q_forward_global_scale_out,                                 \
    at::Tensor k_forward_scales_out,                                       \
    at::Tensor k_forward_global_scale_out,                                 \
    at::Tensor v_mxfp4_out,                                                \
    at::Tensor v_mxfp4_scales_out,                                         \
    at::Tensor v_forward_fp8_out,                                          \
    at::Tensor v_backward_fp8_out,                                         \
    at::Tensor q_backward_fp8_out,                                         \
    at::Tensor k_backward_fp8_out                                          \
) {                                                                        \
    return project_qkv_gqa_d128_nvfp4_forward_out_impl<                    \
        PUBLISH_MX, VALIDATE, OUTPUT_SHARED_DUAL_V, false, false, false,   \
        REPRESENTED_BACKWARD, PER_BLOCK_QK                                 \
    >(                                                                     \
        std::move(input_fp4),                                              \
        std::move(input_scales),                                           \
        std::move(input_global_scale),                                     \
        std::move(qkv_weight_fp4),                                         \
        std::move(qkv_weight_scales),                                      \
        std::move(qkv_weight_global_scale),                                \
        std::move(adaptive_scales),                                        \
        std::move(rope_packed),                                            \
        batch,                                                             \
        seq_len,                                                           \
        q_heads,                                                           \
        kv_heads,                                                          \
        v_mxfp4_scale_2d,                                                  \
        per_block_qk_scales,                                               \
        cluster_cap,                                                       \
        cache_packed_rope,                                                 \
        cache_adaptive_qk_scale,                                           \
        std::move(q_depth_packed_out),                                     \
        std::move(k_depth_packed_out),                                     \
        std::move(q_forward_scales_out),                                   \
        std::move(q_forward_global_scale_out),                             \
        std::move(k_forward_scales_out),                                   \
        std::move(k_forward_global_scale_out),                             \
        std::move(v_mxfp4_out),                                            \
        std::move(v_mxfp4_scales_out),                                     \
        std::move(v_forward_fp8_out),                                      \
        std::move(v_backward_fp8_out),                                     \
        std::move(q_backward_fp8_out),                                     \
        std::move(k_backward_fp8_out)                                      \
    );                                                                     \
}

TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_fp8_forward_out,
    false,
    true,
    false,
    false,
    false
)
TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_fp8_forward_out_unchecked,
    false,
    false,
    false,
    false,
    false
)
TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_represented_backward_perblock_qk_fp8_forward_out,
    false,
    true,
    false,
    true,
    true
)
TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_represented_backward_perblock_qk_fp8_forward_out_unchecked,
    false,
    false,
    false,
    true,
    true
)
TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_mx_forward_out,
    true,
    true,
    false,
    false,
    false
)
TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_mx_forward_out_unchecked,
    true,
    false,
    false,
    false,
    false
)
TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_output_shared_dual_v_mx_forward_out,
    true,
    true,
    true,
    false,
    true
)
TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_output_shared_dual_v_mx_forward_out_unchecked,
    true,
    false,
    true,
    false,
    true
)

#define TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT(                \
    NAME, VALIDATE, COMMON_ROWSCALE, SHARED_TILE                          \
)                                                                          \
std::vector<at::Tensor> NAME(                                              \
    at::Tensor input_fp4,                                                  \
    at::Tensor input_scales,                                               \
    at::Tensor input_global_scale,                                         \
    at::Tensor qkv_weight_fp4,                                             \
    at::Tensor qkv_weight_scales,                                          \
    at::Tensor qkv_weight_global_scale,                                    \
    at::Tensor adaptive_scales,                                            \
    at::Tensor rope_packed,                                                \
    int batch,                                                             \
    int seq_len,                                                           \
    int q_heads,                                                           \
    int kv_heads,                                                          \
    bool v_mxfp4_scale_2d,                                                 \
    bool per_block_qk_scales,                                              \
    int cluster_cap,                                                       \
    bool cache_packed_rope,                                                \
    bool cache_adaptive_qk_scale,                                          \
    at::Tensor q_depth_packed_out,                                         \
    at::Tensor k_depth_packed_out,                                         \
    at::Tensor q_forward_scales_out,                                       \
    at::Tensor q_forward_global_scale_out,                                 \
    at::Tensor k_forward_scales_out,                                       \
    at::Tensor k_forward_global_scale_out,                                 \
    at::Tensor v_mxfp4_out,                                                \
    at::Tensor v_mxfp4_scales_out,                                         \
    at::Tensor v_forward_fp8_out,                                          \
    at::Tensor v_backward_fp8_out,                                         \
    at::Tensor q_backward_fp8_out,                                         \
    at::Tensor k_backward_fp8_out,                                         \
    at::Tensor v_backward_mxfp4_out,                                       \
    at::Tensor v_backward_mxfp4_scales_out                                 \
) {                                                                        \
    return project_qkv_gqa_d128_nvfp4_forward_out_impl<                    \
        true, VALIDATE, true, true, COMMON_ROWSCALE, SHARED_TILE           \
    >(                                                                     \
        std::move(input_fp4),                                              \
        std::move(input_scales),                                           \
        std::move(input_global_scale),                                     \
        std::move(qkv_weight_fp4),                                         \
        std::move(qkv_weight_scales),                                      \
        std::move(qkv_weight_global_scale),                                \
        std::move(adaptive_scales),                                        \
        std::move(rope_packed),                                            \
        batch,                                                             \
        seq_len,                                                           \
        q_heads,                                                           \
        kv_heads,                                                          \
        v_mxfp4_scale_2d,                                                  \
        per_block_qk_scales,                                               \
        cluster_cap,                                                       \
        cache_packed_rope,                                                 \
        cache_adaptive_qk_scale,                                           \
        std::move(q_depth_packed_out),                                     \
        std::move(k_depth_packed_out),                                     \
        std::move(q_forward_scales_out),                                   \
        std::move(q_forward_global_scale_out),                             \
        std::move(k_forward_scales_out),                                   \
        std::move(k_forward_global_scale_out),                             \
        std::move(v_mxfp4_out),                                            \
        std::move(v_mxfp4_scales_out),                                     \
        std::move(v_forward_fp8_out),                                      \
        std::move(v_backward_fp8_out),                                     \
        std::move(q_backward_fp8_out),                                     \
        std::move(k_backward_fp8_out),                                     \
        std::move(v_backward_mxfp4_out),                                   \
        std::move(v_backward_mxfp4_scales_out)                             \
    );                                                                     \
}

TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_mx_backward_v_mx_forward_out,
    true,
    false,
    false
)
TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_mx_backward_v_mx_forward_out_unchecked,
    false,
    false,
    false
)
TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_direct_common_rowscale_mx_backward_v_mx_forward_out,
    true,
    true,
    false
)
TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_direct_common_rowscale_mx_backward_v_mx_forward_out_unchecked,
    false,
    true,
    false
)
TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_shared_tile_mx_backward_v_mx_forward_out,
    true,
    false,
    true
)
TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT(
    project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_shared_tile_mx_backward_v_mx_forward_out_unchecked,
    false,
    false,
    true
)

#undef TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT
#undef TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT

template <bool PublishE5M2Dout>
std::vector<at::Tensor> project_dout_unified_fp4_nvfp4_impl(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor weight_fp4,
    at::Tensor weight_scales,
    at::Tensor weight_global_scale,
    at::Tensor attention_output,
    at::Tensor lse,
    int batch,
    int seq_len,
    int heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool publish_stats,
    std::optional<at::Tensor> stats_workspace,
    std::optional<at::Tensor> dq_clear,
    double probability_log2_lift
) {
    const int kDepth = static_cast<int>(attention_output.size(3));
    TORCH_CHECK(
        kDepth == 64 || kDepth == 128,
        "unified NVFP4 dO projection requires head dimension 64 or 128"
    );
    const int rows = batch * seq_len;
    const int output_width = heads * kDepth;
    TORCH_CHECK(
        input_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            weight_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            input_fp4.is_cuda() && weight_fp4.is_cuda() &&
            input_fp4.is_contiguous() && weight_fp4.is_contiguous() &&
            input_fp4.dim() == 2 && weight_fp4.dim() == 2,
        "dO projection operands must be contiguous CUDA packed E2M1 matrices"
    );
    const int hidden = static_cast<int>(input_fp4.size(1) * 2);
    TORCH_CHECK(
        input_fp4.size(0) == rows &&
            weight_fp4.size(0) == output_width &&
            weight_fp4.size(1) == input_fp4.size(1) &&
            rows % 256 == 0 && seq_len % 256 == 0 &&
            output_width % 256 == 0 && hidden % 256 == 0,
        "unified NVFP4 dO projection requires A=[B*S,K/2], "
        "B=[H*D,K/2], D in {64,128}, S and K divisible by 256, "
        "and H*D divisible by 256"
    );
    TORCH_CHECK(
        input_scales.scalar_type() == at::kFloat8_e4m3fn &&
            weight_scales.scalar_type() == at::kFloat8_e4m3fn &&
            input_scales.is_cuda() && weight_scales.is_cuda() &&
            input_scales.is_contiguous() && weight_scales.is_contiguous() &&
            input_global_scale.scalar_type() == at::kFloat &&
            weight_global_scale.scalar_type() == at::kFloat &&
            input_global_scale.is_cuda() && weight_global_scale.is_cuda() &&
            input_global_scale.numel() == 1 &&
            weight_global_scale.numel() == 1,
        "unified dO projection requires E4M3 block scales and one float32 "
        "global scale per operand"
    );
    TORCH_CHECK(
        attention_output.scalar_type() == at::kBFloat16 &&
            attention_output.is_cuda() && attention_output.is_contiguous() &&
            attention_output.sizes() == at::IntArrayRef({
                batch, seq_len, heads, kDepth
            }),
        "attention output must be contiguous BF16 [B,S,H,D]"
    );
    const bool lse_sequence_major =
        lse.sizes() == at::IntArrayRef({batch, seq_len, heads});
    const bool lse_head_major =
        lse.sizes() == at::IntArrayRef({batch, heads, 1, seq_len});
    TORCH_CHECK(
        lse.scalar_type() == at::kFloat && lse.is_cuda() &&
            lse.is_contiguous() &&
            (lse_sequence_major || lse_head_major),
        "LSE must be contiguous float32 [B,S,H] or [B,H,1,S]"
    );
    kittens::py::device_check(
        input_fp4,
        input_scales,
        input_global_scale,
        weight_fp4,
        weight_scales,
        weight_global_scale,
        attention_output,
        lse
    );
    const c10::cuda::CUDAGuard device_guard(input_fp4.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "unified dO projection requires GB200 / SM100"
    );
    TORCH_CHECK(
        publish_stats || publish_fp8_backward,
        "statistics-free dO publication is supported only by the retained "
        "FP8 backward route"
    );
    TORCH_CHECK(
        publish_stats || !stats_workspace.has_value(),
        "a dO statistics workspace requires publish_stats=True"
    );
    TORCH_CHECK(
        !stats_workspace.has_value() || publish_fp8_backward,
        "direct dO statistics workspace publication currently requires the "
        "retained FP8 backward operand"
    );
    TORCH_CHECK(
        !dq_clear.has_value() || publish_fp8_backward,
        "fused dQ clear requires the retained FP8 backward publication"
    );
    TORCH_CHECK(
        probability_log2_lift == 0.0 || probability_log2_lift == 8.0,
        "dO probability log2 lift must be exactly zero or eight"
    );
    TORCH_CHECK(
        kDepth == 128 || probability_log2_lift == 0.0,
        "an explicit dO probability log2 lift is supported only for D128"
    );
    TORCH_CHECK(
        probability_log2_lift == 0.0 ||
            (publish_fp8_backward && publish_stats &&
             stats_workspace.has_value()),
        "a dO probability log2 lift requires direct FP8 backward statistics "
        "workspace publication"
    );
    if constexpr (PublishE5M2Dout) {
        TORCH_CHECK(
            (batch == 1 || batch == 2 || batch == 4) &&
                seq_len == 4096 && heads == 32 && kDepth == 128 &&
                !store_bf16 && publish_fp8_backward && publish_stats &&
                stats_workspace.has_value() && dq_clear.has_value() &&
                probability_log2_lift == 8.0,
            "v509 E5M2 dO publication is restricted to B1/B2/B4 at "
            "S4096/H32/D128 "
            "native-score with no BF16 dO store, probability lift 8, direct "
            "statistics workspace publication, and fused dQ clear"
        );
    }

    const int64_t stats_numel =
        static_cast<int64_t>(batch) * heads * seq_len;
    const int64_t stats_workspace_bytes =
        2 * stats_numel * static_cast<int64_t>(sizeof(float));
    if (stats_workspace.has_value()) {
        const at::Tensor &workspace = stats_workspace.value();
        TORCH_CHECK(
            workspace.scalar_type() == at::ScalarType::Byte &&
                workspace.is_cuda() && workspace.is_contiguous() &&
                workspace.dim() == 1,
            "dO statistics workspace must be a contiguous 1D CUDA uint8 "
            "tensor"
        );
        TORCH_CHECK(
            workspace.device() == input_fp4.device(),
            "dO statistics workspace must be on the projection device"
        );
        TORCH_CHECK(
            workspace.numel() >= stats_workspace_bytes,
            "dO statistics workspace is too small: expected at least ",
            stats_workspace_bytes,
            " bytes, got ",
            workspace.numel()
        );
        if constexpr (PublishE5M2Dout) {
            TORCH_CHECK(
                workspace.numel() == stats_workspace_bytes,
                "v509 E5M2 dO publication requires the exact statistics "
                "workspace byte count"
            );
        }
        TORCH_CHECK(
            reinterpret_cast<std::uintptr_t>(workspace.data_ptr()) % 16 == 0,
            "dO statistics workspace must be 16-byte aligned"
        );
    }
    if (dq_clear.has_value()) {
        const at::Tensor &dq = dq_clear.value();
        const bool direct_shape =
            dq.sizes() == at::IntArrayRef({
                batch, seq_len, heads, kDepth
            });
        const bool hierarchical_shape =
            dq.sizes() == at::IntArrayRef({
                2, batch, heads, seq_len, kDepth
            });
        TORCH_CHECK(
            dq.scalar_type() == at::kBFloat16 && dq.is_cuda() &&
                dq.is_contiguous() &&
                (direct_shape || hierarchical_shape),
            "fused dQ clear target must be contiguous BF16 [B,S,H,D] "
            "or [2,B,H,S,D]"
        );
        TORCH_CHECK(
            dq.device() == input_fp4.device(),
            "fused dQ clear target must be on the projection device"
        );
        TORCH_CHECK(
            reinterpret_cast<std::uintptr_t>(dq.data_ptr()) % 16 == 0,
            "fused dQ clear target must be 16-byte aligned"
        );
        if constexpr (PublishE5M2Dout) {
            TORCH_CHECK(
                direct_shape,
                "v509 E5M2 dO publication requires direct BSHD dQ clear "
                "storage"
            );
        }
    }

    auto bf16_options = input_fp4.options().dtype(at::kBFloat16);
    // The no-BF16 specialization never touches D. Reuse the already valid
    // attention-output descriptor instead of allocating a dead full-size
    // publication buffer on the production path.
    at::Tensor dout = store_bf16
        ? at::empty({batch, seq_len, heads, kDepth}, bf16_options)
        : attention_output;
    auto byte_options = input_fp4.options().dtype(at::ScalarType::Byte);
    at::Tensor dout_dp = publish_fp8_backward
        ? at::empty({0}, byte_options)
        : at::empty(
              {batch, seq_len, heads, kDepth / 2},
              byte_options
          );
    at::Tensor dout_dp_scales = publish_fp8_backward
        ? at::empty({0}, byte_options)
        : at::empty(
              {batch, seq_len / 128, heads, 512},
              byte_options
          );
    at::Tensor dout_dv = publish_fp8_backward
        ? at::empty({0}, byte_options)
        : at::empty(
              {batch, heads, kDepth, seq_len / 2},
              byte_options
          );
    at::Tensor dout_dv_scales = publish_fp8_backward
        ? at::empty({0}, byte_options)
        : at::empty_like(dout_dp_scales);
    const at::ScalarType backward_fp8_dtype = PublishE5M2Dout
        ? at::ScalarType::Float8_e5m2
        : at::ScalarType::Float8_e4m3fn;
    at::Tensor dout_backward_fp8 = publish_fp8_backward
        ? at::empty(
              {batch, seq_len, heads, kDepth},
              input_fp4.options().dtype(backward_fp8_dtype)
          )
        : at::empty(
              {0},
              input_fp4.options().dtype(backward_fp8_dtype)
          );
    at::Tensor dpsum;
    at::Tensor lse_log2;
    if (publish_stats && stats_workspace.has_value()) {
        at::Tensor stats = stats_workspace.value()
            .narrow(0, 0, stats_workspace_bytes)
            .view(at::kFloat);
        dpsum = stats.narrow(0, 0, stats_numel).view(
            {batch, heads, 1, seq_len}
        );
        lse_log2 = stats.narrow(0, stats_numel, stats_numel).view(
            {batch, heads, 1, seq_len}
        );
    } else if (publish_stats) {
        dpsum = at::empty(
            {batch, heads, 1, seq_len},
            input_fp4.options().dtype(at::kFloat)
        );
        lse_log2 = at::empty_like(dpsum);
    } else {
        dpsum = at::empty({0}, input_fp4.options().dtype(at::kFloat));
        lse_log2 = at::empty_like(dpsum);
    }

    if constexpr (PublishE5M2Dout) {
        using named_tensor = std::pair<const char *, const at::Tensor *>;
        const std::initializer_list<named_tensor> write_destinations{
            {"dout_backward_e5m2", &dout_backward_fp8},
            {"stats_workspace", &stats_workspace.value()},
            {"dq_clear", &dq_clear.value()},
        };
        const std::initializer_list<named_tensor> read_operands{
            {"input_fp4", &input_fp4},
            {"input_scales", &input_scales},
            {"input_global_scale", &input_global_scale},
            {"weight_fp4", &weight_fp4},
            {"weight_scales", &weight_scales},
            {"weight_global_scale", &weight_global_scale},
            {"attention_output", &attention_output},
            {"lse", &lse},
        };
        const auto byte_ranges_overlap = [](
            const at::Tensor &left,
            const at::Tensor &right
        ) {
            const auto left_begin =
                reinterpret_cast<std::uintptr_t>(left.data_ptr());
            const auto right_begin =
                reinterpret_cast<std::uintptr_t>(right.data_ptr());
            const auto left_bytes = static_cast<std::uintptr_t>(
                left.numel() * left.element_size()
            );
            const auto right_bytes = static_cast<std::uintptr_t>(
                right.numel() * right.element_size()
            );
            return left_begin <= right_begin
                ? right_begin - left_begin < left_bytes
                : left_begin - right_begin < right_bytes;
        };
        for (auto left = write_destinations.begin();
             left != write_destinations.end(); ++left) {
            for (auto right = left + 1;
                 right != write_destinations.end(); ++right) {
                TORCH_CHECK(
                    !byte_ranges_overlap(*left->second, *right->second),
                    "v509 E5M2 write destinations must use disjoint storage: ",
                    left->first,
                    " overlaps ",
                    right->first
                );
            }
            for (const auto &read : read_operands) {
                TORCH_CHECK(
                    !byte_ranges_overlap(*left->second, *read.second),
                    read.first,
                    " must not overlap v509 E5M2 write destination ",
                    left->first
                );
            }
        }
    }

    // The retained FP8-only dO branch publishes directly from the dynamic
    // output tile and does not allocate the static MXFP4 staging fragment, so
    // it can preserve the four-stage load pipeline.
    using C = tkfa4_projection::config<4, 4>;
    using G = tkfa4_projection::globals<C>;
    G globals{
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(
            input_fp4, 1, 1, rows, hidden / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            input_scales,
            1,
            input_scales.size(0),
            input_scales.size(1),
            256
        ),
        .A_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            input_global_scale
        ),
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(
            weight_fp4, 1, 1, output_width, hidden / 2
        ),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            weight_scales,
            1,
            weight_scales.size(0),
            weight_scales.size(1),
            256
        ),
        .B_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            weight_global_scale
        ),
        // OUTPUT_IS_DOUT ignores Q/K width, but valid descriptors keep the
        // shared projection plumbing and optional BF16 store simple.
        .Q = kittens::py::tensor_to_gl<typename G::D_gl>(
            dout, 1, 1, rows, output_width
        ),
        .K = kittens::py::tensor_to_gl<typename G::D_gl>(
            dout, 1, 1, rows, output_width
        ),
        .V = kittens::py::tensor_to_gl<typename G::D_gl>(
            dout, 1, 1, rows, output_width
        ),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(
            dout, 1, 1, rows, output_width
        ),
        .v_mxfp4 = publish_fp8_backward
            ? nullptr
            : reinterpret_cast<uint8_t *>(dout_dv.data_ptr()),
        .v_mxfp4_scales = publish_fp8_backward
            ? nullptr
            : reinterpret_cast<uint8_t *>(dout_dv_scales.data_ptr()),
        .v_backward_mxfp4 = publish_fp8_backward
            ? nullptr
            : reinterpret_cast<uint8_t *>(dout_dp.data_ptr()),
        .v_backward_mxfp4_scales = publish_fp8_backward
            ? nullptr
            : reinterpret_cast<uint8_t *>(dout_dp_scales.data_ptr()),
        .v_backward_fp8 = publish_fp8_backward
            ? reinterpret_cast<uint8_t *>(dout_backward_fp8.data_ptr())
            : nullptr,
        .attention_output = reinterpret_cast<const bf16 *>(
            attention_output.data_ptr()
        ),
        .lse = reinterpret_cast<const float *>(lse.data_ptr()),
        .lse_head_major = lse_head_major,
        .dpsum = reinterpret_cast<float *>(dpsum.data_ptr()),
        .lse_log2 = reinterpret_cast<float *>(lse_log2.data_ptr()),
        .dout_probability_log2_lift =
            static_cast<float>(probability_log2_lift),
        .dq_clear = dq_clear.has_value()
            ? reinterpret_cast<uint4 *>(dq_clear.value().data_ptr())
            : nullptr,
        .dq_clear_vectors = dq_clear.has_value()
            ? dq_clear.value().numel() / 8
            : 0,
        .batch = batch,
        .seq_len = seq_len,
        .heads = heads,
        .head_depth = kDepth,
        .v_width = output_width,
        .v_scale_rows = 1,
        // dP's native MX path uses the standard width-six reconstruction
        // contract.  Make the dO publisher's 32x32/RTE policy explicit rather
        // than relying on the globals default.
        .v_mxfp4_scale_2d = true,
    };
    if constexpr (PublishE5M2Dout) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel_v509_native_score_e5m2_dout<C>
        >(globals);
    } else if (publish_fp8_backward && store_bf16 && publish_stats) {
        if (stats_workspace.has_value()) {
            if (dq_clear.has_value()) {
                kittens::py::launch_kernel<
                    C,
                    G,
                    tkfa4_projection::kernel<
                        C, false, false, false, true, true, false, false,
                        false, false, true, false, true, false, false, true,
                        false, false, false, true, true
                    >
                >(globals);
            } else {
                kittens::py::launch_kernel<
                    C,
                    G,
                    tkfa4_projection::kernel<
                        C, false, false, false, true, true, false, false,
                        false, false, true, false, true, false, false, true,
                        false, false, false, true
                    >
                >(globals);
            }
        } else {
            kittens::py::launch_kernel<
                C,
                G,
                tkfa4_projection::kernel<
                    C, false, false, false, true, true, false, false, false,
                    false, true, false
                >
            >(globals);
        }
    } else if (publish_fp8_backward && store_bf16) {
        if (dq_clear.has_value()) {
            kittens::py::launch_kernel<
                C,
                G,
                tkfa4_projection::kernel<
                    C, false, false, false, true, true, false, false,
                    false, false, true, false, false, false, false, true,
                    false, false, false, false, true
                >
            >(globals);
        } else {
            kittens::py::launch_kernel<
                C,
                G,
                tkfa4_projection::kernel<
                    C, false, false, false, true, true, false, false,
                    false, false, true, false, false
                >
            >(globals);
        }
    } else if (publish_fp8_backward && publish_stats) {
        if (stats_workspace.has_value()) {
            if (dq_clear.has_value()) {
                kittens::py::launch_kernel<
                    C,
                    G,
                    tkfa4_projection::kernel<
                        C, false, false, false, false, true, false, false,
                        false, false, true, false, true, false, false, true,
                        false, false, false, true, true
                    >
                >(globals);
            } else {
                kittens::py::launch_kernel<
                    C,
                    G,
                    tkfa4_projection::kernel<
                        C, false, false, false, false, true, false, false,
                        false, false, true, false, true, false, false, true,
                        false, false, false, true
                    >
                >(globals);
            }
        } else {
            kittens::py::launch_kernel<
                C,
                G,
                tkfa4_projection::kernel<
                    C, false, false, false, false, true, false, false, false,
                    false, true, false
                >
            >(globals);
        }
    } else if (publish_fp8_backward) {
        if (dq_clear.has_value()) {
            kittens::py::launch_kernel<
                C,
                G,
                tkfa4_projection::kernel<
                    C, false, false, false, false, true, false, false,
                    false, false, true, false, false, false, false, true,
                    false, false, false, false, true
                >
            >(globals);
        } else {
            kittens::py::launch_kernel<
                C,
                G,
                tkfa4_projection::kernel<
                    C, false, false, false, false, true, false, false,
                    false, false, true, false, false
                >
            >(globals);
        }
    } else if (store_bf16) {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<C, false, false, true, true, true>
        >(globals);
    } else {
        kittens::py::launch_kernel<
            C,
            G,
            tkfa4_projection::kernel<C, false, false, true, false, true>
        >(globals);
    }
    return {
        dout,
        dout_dp,
        dout_dp_scales,
        dout_dv,
        dout_dv_scales,
        dpsum,
        lse_log2,
        dout_backward_fp8
    };
}

std::vector<at::Tensor> project_dout_unified_fp4_nvfp4(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor weight_fp4,
    at::Tensor weight_scales,
    at::Tensor weight_global_scale,
    at::Tensor attention_output,
    at::Tensor lse,
    int batch,
    int seq_len,
    int heads,
    bool store_bf16,
    bool publish_fp8_backward,
    bool publish_stats,
    std::optional<at::Tensor> stats_workspace,
    std::optional<at::Tensor> dq_clear,
    double probability_log2_lift
) {
    return project_dout_unified_fp4_nvfp4_impl<false>(
        std::move(input_fp4),
        std::move(input_scales),
        std::move(input_global_scale),
        std::move(weight_fp4),
        std::move(weight_scales),
        std::move(weight_global_scale),
        std::move(attention_output),
        std::move(lse),
        batch,
        seq_len,
        heads,
        store_bf16,
        publish_fp8_backward,
        publish_stats,
        std::move(stats_workspace),
        std::move(dq_clear),
        probability_log2_lift
    );
}

std::vector<at::Tensor> project_dout_unified_fp4_nvfp4_v509_e5m2(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor weight_fp4,
    at::Tensor weight_scales,
    at::Tensor weight_global_scale,
    at::Tensor attention_output,
    at::Tensor lse,
    at::Tensor stats_workspace,
    at::Tensor dq_clear
) {
    constexpr int kSequence = 4096;
    constexpr int kHeads = 32;
    const int64_t batch = attention_output.size(0);
    TORCH_CHECK(
        batch == 1 || batch == 2 || batch == 4,
        "v509 E5M2 dO publisher batch must be 1, 2, or 4"
    );
    return project_dout_unified_fp4_nvfp4_impl<true>(
        std::move(input_fp4),
        std::move(input_scales),
        std::move(input_global_scale),
        std::move(weight_fp4),
        std::move(weight_scales),
        std::move(weight_global_scale),
        std::move(attention_output),
        std::move(lse),
        static_cast<int>(batch),
        kSequence,
        kHeads,
        false,
        true,
        true,
        std::move(stats_workspace),
        std::move(dq_clear),
        8.0
    );
}

pybind11::dict project_dout_unified_fp4_nvfp4_v509_e5m2_metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.v509_e5m2_dout_publisher.v1";
    result["source_identity"] =
        "v509_fused_nvfp4_output_projection_e5m2_dout_b1_b2_b4_s4096_v1";
    result["source_file"] = __FILE__;
    result["experimental"] = true;
    result["production_dispatch_connected"] = false;
    result["dispatch"] =
        "B1_B2_B4_S4096_H32_D128_native_score_only";
    result["selected_epilogue"] =
        "kernel_v509_native_score_e5m2_dout";
    result["payload_dtype"] = "float8_e5m2";
    result["payload_layout"] = "BSHD_contiguous";
    result["encode"] = "(BF16.float()*4).to(float8_e5m2)";
    result["encode_scale"] = 4.0;
    result["logical_decode_scale"] = 0.25;
    result["dstat_physical_abi"] = "-4*sum(O*raw_E5M2_dO)";
    result["lstat_abi"] = "8-LSE*log2(e)";
    result["probability_log2_lift"] = 8.0;
    result["batch_values"] = pybind11::make_tuple(1, 2, 4);
    result["sequence"] = 4096;
    result["query_heads"] = 32;
    result["head_dim"] = 128;
    result["store_bf16_dout"] = false;
    result["publish_e4m3_dout"] = false;
    result["publish_stats"] = true;
    result["clear_dq"] = true;
    result["raw_output_slots"] = 8;
    result["e5m2_payload_slot"] = 7;
    return result;
}

at::Tensor project_bf16_dq_persistent(
    at::Tensor dq,
    at::Tensor weight_t
) {
    constexpr int kDepth = 192;
    TORCH_CHECK(
        dq.scalar_type() == at::kBFloat16 && dq.is_cuda() &&
            dq.is_contiguous() && dq.dim() == 4 && dq.size(0) == 1 &&
            dq.size(3) == kDepth && dq.size(1) % 256 == 0,
        "persistent dQ projection requires contiguous BF16 [1,S,H,192] "
        "with S divisible by 256"
    );
    const int rows = static_cast<int>(dq.size(1));
    const int heads = static_cast<int>(dq.size(2));
    const int reduction = heads * kDepth;
    TORCH_CHECK(
        weight_t.scalar_type() == at::kBFloat16 && weight_t.is_cuda() &&
            weight_t.is_contiguous() && weight_t.dim() == 2 &&
            weight_t.size(1) == reduction && weight_t.size(0) % 256 == 0,
        "persistent dQ projection weight must be contiguous BF16 "
        "[hidden,H*192] with hidden divisible by 256"
    );
    kittens::py::device_check(dq, weight_t);
    const c10::cuda::CUDAGuard device_guard(dq.device());
    auto output = at::empty(
        {1, rows, weight_t.size(0)},
        dq.options()
    );
    const int q_tiles = rows / tkfa4_dq_projection::kTileRows;
    auto arrivals = at::full(
        {heads, q_tiles},
        std::numeric_limits<int32_t>::max(),
        dq.options().dtype(at::ScalarType::Int)
    );
    using C = tkfa4_dq_projection::config<>;
    using G = tkfa4_dq_projection::globals<C>;
    G globals{
        kittens::py::tensor_to_gl<typename G::A_gl, false>(
            dq, 1, 1, rows, reduction
        ),
        kittens::py::tensor_to_gl<typename G::B_gl, false>(
            weight_t, 1, 1, static_cast<int>(weight_t.size(0)), reduction
        ),
        kittens::py::tensor_to_gl<typename G::D_gl, false>(
            output, 1, 1, rows, static_cast<int>(weight_t.size(0))
        ),
        reinterpret_cast<const uint32_t *>(arrivals.data_ptr()),
        heads,
        0,
        (rows / C::Mb) *
            (static_cast<int>(weight_t.size(0)) / C::Nb),
        0,
    };
    tkfa4_dq_projection::launch(
        globals,
        at::cuda::getCurrentCUDAStream().stream()
    );
    return output;
}

// Publication is scalar/register work attached to a persistent tensor-core
// projection.  With a shallow reduction it extends the critical path; a
// standalone packer has enough independent CTAs to finish sooner.  Once the
// reduction reaches twelve K256 steps, the projection pipeline hides that
// work and avoiding the BF16 reread wins.  Keep the policy in the native
// producer so every backward consumer receives the same optimal layouts.
constexpr int kProjectionFusedPublicationMinHidden = 3072;

enum class ProjectionFp4PublicationPolicy : int {
    Auto = 0,
    Fused = 1,
    Separate = 2,
};

std::vector<at::Tensor> project_qk_adaptive_fp4_nvfp4_dispatch(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor qk_weight_fp4,
    at::Tensor qk_weight_scales,
    at::Tensor qk_weight_global_scale,
    at::Tensor adaptive_scales,
    int batch,
    int seq_len,
    int heads,
    int publication_policy
) {
    TORCH_CHECK(
        publication_policy >=
                static_cast<int>(ProjectionFp4PublicationPolicy::Auto) &&
            publication_policy <=
                static_cast<int>(ProjectionFp4PublicationPolicy::Separate),
        "publication_policy must be 0 (auto), 1 (fused), or 2 (separate)"
    );
    const int hidden = input_fp4.dim() == 2
        ? static_cast<int>(input_fp4.size(1) * 2)
        : 0;
    const auto policy =
        static_cast<ProjectionFp4PublicationPolicy>(publication_policy);
    const bool publish_fused =
        policy == ProjectionFp4PublicationPolicy::Fused ||
        (policy == ProjectionFp4PublicationPolicy::Auto &&
         hidden >= kProjectionFusedPublicationMinHidden);

    auto projected = project_qk_adaptive_fp4_nvfp4(
        input_fp4,
        input_scales,
        input_global_scale,
        qk_weight_fp4,
        qk_weight_scales,
        qk_weight_global_scale,
        adaptive_scales,
        batch,
        seq_len,
        heads,
        publish_fused
    );
    if (publish_fused) {
        return projected;
    }

    auto packed = quantize_fp4_dual_qk_precomputed_scales(
        projected[0],
        projected[1],
        adaptive_scales
    );
    return {
        projected[0],
        projected[1],
        packed[0],
        packed[1],
        packed[2],
        packed[3],
        packed[4]
    };
}

void check_bf16_control_inputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &lse,
    const at::Tensor &dout,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    tkfa4::check_bshd(q, "q", at::kBFloat16);
    tkfa4::check_bshd(k, "k", at::kBFloat16);
    tkfa4::check_bshd(v, "v", at::kBFloat16);
    tkfa4::check_bshd(out, "out", at::kBFloat16);
    tkfa4::check_bshd(dout, "dout", at::kBFloat16);
    kittens::py::device_check(q, k, v, out, lse, dout);
    tkfa4::check_exact_b300_qkv_bshd(q, k, v);
    tkfa4::check_exact_b300_lse_bsh(lse, q);
    TORCH_CHECK(tkfa4::is_sm100_device(), "lowp_fa4_bwd requires GB200 / SM100");
    TORCH_CHECK(out.sizes() == v.sizes(), "out must match v");
    TORCH_CHECK(dout.sizes() == v.sizes(), "dout must match v");
    TORCH_CHECK(q.size(0) == 1, "the copied V382 control currently supports batch size 1");
    TORCH_CHECK(q.size(1) % 256 == 0, "the copied V382 control requires seqlen divisible by 256");
    TORCH_CHECK(causal, "the copied V382 control currently supports causal=True only");
    TORCH_CHECK(!deterministic, "the copied V382 control currently supports deterministic=False only");
    TORCH_CHECK(
        softmax_scale == 0x1.279a74p-4f,
        "the copied V382 control requires the exact D192 default softmax scale"
    );
}

at::Tensor prepack_mixed_v(at::Tensor v) {
    tkfa4::check_bshd(v, "v", at::kBFloat16);
    TORCH_CHECK(
        v.is_cuda() && v.is_contiguous(),
        "mixed V prepack requires contiguous CUDA BF16 input"
    );
    TORCH_CHECK(
        v.size(3) == tkfa4::kB300VDim,
        "mixed V prepack requires head dimension 128"
    );
    TORCH_CHECK(
        v.size(1) % 8 == 0,
        "mixed V prepack requires sequence length divisible by eight"
    );
    const c10::cuda::CUDAGuard device_guard(v.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "mixed V prepack requires GB200 / SM100"
    );
    at::Tensor packed_v = at::empty(
        v.sizes(),
        v.options().dtype(at::ScalarType::Float8_e4m3fn)
    );
    tkfa4::bwd_cute16_candidate::launch_prepack_mixed_v<
        tkfa4::bwd_cute16_candidate::preprocess_config<tkfa4::kB300VDim>
    >(v, packed_v);
    return packed_v;
}

std::vector<at::Tensor> prepare_mxfp4_backward_operands(
    at::Tensor out,
    at::Tensor dout,
    at::Tensor v,
    at::Tensor lse
) {
    TORCH_CHECK(
        out.scalar_type() == at::ScalarType::BFloat16 &&
            dout.scalar_type() == at::ScalarType::BFloat16 &&
            v.scalar_type() == at::ScalarType::BFloat16 &&
            out.is_cuda() && dout.is_cuda() && v.is_cuda() &&
            out.is_contiguous() && dout.is_contiguous() &&
            v.is_contiguous() && out.dim() == 4 &&
            out.sizes() == dout.sizes() && out.sizes() == v.sizes() &&
            out.size(3) == tkfa4::kB300VDim,
        "MXFP4 backward operand production requires matching contiguous "
        "BF16 O/dO/V tensors [B,S,H,128]"
    );
    TORCH_CHECK(
        lse.scalar_type() == at::ScalarType::Float && lse.is_cuda() &&
            lse.is_contiguous() && lse.dim() == 3 &&
            lse.size(0) == out.size(0) &&
            lse.size(1) == out.size(1) &&
            lse.size(2) == out.size(2),
        "MXFP4 backward operand production requires contiguous float32 "
        "LSE [B,S,H]"
    );
    kittens::py::device_check(out, dout, v, lse);
    const c10::cuda::CUDAGuard device_guard(out.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "MXFP4 backward operand production requires GB200 / SM100"
    );
    auto byte_options = out.options().dtype(at::ScalarType::Byte);
    at::Tensor dout_dp = at::empty(
        {out.size(0), out.size(1), out.size(2), out.size(3) / 2},
        byte_options
    );
    at::Tensor v_dp = at::empty_like(dout_dp);
    at::Tensor dout_dp_scale = at::empty(
        {out.size(0), out.size(1) / 128, out.size(2), 512},
        byte_options
    );
    at::Tensor v_dp_scale = at::empty_like(dout_dp_scale);
    at::Tensor dout_dv = at::empty(
        {out.size(0), out.size(2), out.size(3), out.size(1) / 2},
        byte_options
    );
    at::Tensor dout_dv_scale = at::empty_like(dout_dp_scale);
    at::Tensor dpsum = at::empty(
        {out.size(0), out.size(2), 1, out.size(1)},
        lse.options()
    );
    at::Tensor lse_log2 = at::empty_like(dpsum);
    tkfa4::bwd_cute16_candidate::launch_preprocess_mxfp4_dp_dv<
        tkfa4::bwd_cute16_candidate::preprocess_config<tkfa4::kB300VDim>
    >(
        out,
        dout,
        v,
        lse,
        dout_dp,
        v_dp,
        dout_dp_scale,
        v_dp_scale,
        dout_dv,
        dout_dv_scale,
        dpsum,
        lse_log2
    );
    return {
        dout_dp,
        v_dp,
        dout_dp_scale,
        v_dp_scale,
        dout_dv,
        dout_dv_scale,
        dpsum,
        lse_log2
    };
}

template <
    int ExactQTileCount,
    int ExpectedSeqLen,
    int LowpMode = tkfa4::bwd_cute16_kernel_candidate::detail::
        kCta2DenseLowpNone,
    bool UseX32Fp8Pv = false,
    bool ReuseDqDsForDk = false,
    bool UseAdaptiveQkScales = false,
    bool ReturnBf16Dq = false,
    bool ReturnInterleavedQkv = false,
    bool ReturnDirectDqProjection = false,
    bool UseRank128Score = false,
    bool ReturnTileReadyNvfp4DqProjection = false
>
std::vector<at::Tensor> launch_v382_mode(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    bool deterministic,
    at::Tensor *q_lowp = nullptr,
    at::Tensor *k_lowp = nullptr,
    float dq_output_scale = 1.0f,
    float dk_output_scale = 1.0f,
    float ds_quant_scale =
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseFp8DefaultDsQuantScale,
    at::Tensor *score_q_lowp = nullptr,
    at::Tensor *score_k_lowp = nullptr,
    at::Tensor *q_dk_mxfp4 = nullptr,
    at::Tensor *k_dq_mxfp4 = nullptr,
    at::Tensor *q_dk_nvfp4_scale = nullptr,
    at::Tensor *k_dq_nvfp4_scale = nullptr,
    at::Tensor *mixed_v_prepacked = nullptr,
    at::Tensor *adaptive_qk_scales = nullptr,
    at::Tensor *dq_projection_weight = nullptr,
    tkfa4::bwd_cute16_candidate::producer_native_mxfp4_operands
        *producer_mxfp4 = nullptr,
    at::Tensor *dq_projection_weight_fp4 = nullptr,
    at::Tensor *dq_projection_weight_scales = nullptr,
    at::Tensor *dq_projection_weight_global_scale = nullptr,
    at::Tensor *dq_global_scale = nullptr,
    tkfa4::bwd_cute16_candidate::producer_native_fp8_operands
        *producer_fp8 = nullptr,
    at::Tensor *projection_rope_cos = nullptr,
    at::Tensor *projection_rope_sin = nullptr,
    bool hierarchical_qkv_projection = false
) {
    static_assert(
        (ExpectedSeqLen == 4096 && ExactQTileCount == 32) ||
        (ExpectedSeqLen == 8192 && ExactQTileCount == 64) ||
        (ExpectedSeqLen == 16384 && ExactQTileCount == 128) ||
        (ExpectedSeqLen == 32768 && ExactQTileCount == 256) ||
        (ExpectedSeqLen == 65536 && ExactQTileCount == 512)
    );
    TORCH_CHECK(q.size(1) == ExpectedSeqLen, "unexpected sequence length for copied V382 specialization");
    const bool supported_heads =
        (ExpectedSeqLen == 4096 &&
         (q.size(2) == 4 || q.size(2) == 8 || q.size(2) == 24 ||
          q.size(2) == 64)) ||
        (ExpectedSeqLen == 8192 &&
         (q.size(2) == 2 || q.size(2) == 4 || q.size(2) == 8 ||
          q.size(2) == 16 || q.size(2) == 24 || q.size(2) == 64)) ||
        (ExpectedSeqLen == 16384 &&
         (q.size(2) == 4 || q.size(2) == 8 || q.size(2) == 16 ||
          q.size(2) == 24 || q.size(2) == 32 || q.size(2) == 64 ||
          q.size(2) == 128)) ||
        ((ExpectedSeqLen == 32768 || ExpectedSeqLen == 65536) &&
         (q.size(2) == 16 || q.size(2) == 32 || q.size(2) == 64 ||
          q.size(2) == 128));
    TORCH_CHECK(supported_heads, "unsupported head count for copied V382 specialization");

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    static_assert(
        static_cast<int>(ReturnBf16Dq) +
                static_cast<int>(ReturnInterleavedQkv) +
                static_cast<int>(ReturnDirectDqProjection) +
                static_cast<int>(ReturnTileReadyNvfp4DqProjection) <=
            1,
        "direct dQ output modes are mutually exclusive"
    );
    constexpr bool ReturnDirectBf16 =
        ReturnBf16Dq || ReturnInterleavedQkv || ReturnDirectDqProjection ||
        ReturnTileReadyNvfp4DqProjection;
    // Direct BF16 consumers do not touch the legacy FP32 dQ descriptor. Keep
    // one typed element only so the shared launch plumbing remains unchanged
    // without allocating a second full-size gradient tensor.
    at::Tensor dq = ReturnDirectBf16
        ? at::empty({1}, lse.options())
        : at::empty(q.sizes(), lse.options());
    at::Tensor dk = ReturnInterleavedQkv
        ? at::empty({1}, k.options())
        : at::empty(k.sizes(), k.options());
    at::Tensor dv = ReturnInterleavedQkv
        ? at::empty({1}, v.options())
        : at::empty(v.sizes(), v.options());
    at::Tensor dq_bf16;
    at::Tensor projection_qkv;
    at::Tensor dq_projection_output;
    at::Tensor dq_tile_arrivals;
    at::Tensor dq_nvfp4;
    at::Tensor dq_nvfp4_scales;
    at::Tensor dq_nvfp4_ready;
    tkfa4::bwd_cute16_candidate::tile_ready_nvfp4_projection_operands
        nvfp4_projection{};
    if constexpr (ReturnDirectBf16) {
        static_assert(
            UseAdaptiveQkScales &&
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4Fp8DpPvReuseP,
            "direct BF16 outputs are specialized for the adaptive FP4+FP8 "
            "route"
        );
        if constexpr (ReturnBf16Dq) {
            dq_bf16 = at::empty(q.sizes(), q.options());
        } else {
            if constexpr (ReturnInterleavedQkv) {
                projection_qkv = at::empty(
                    {q.size(0), q.size(1), q.size(2),
                     tkfa4::kB300QKDim * 2 + tkfa4::kB300VDim},
                    q.options()
                );
            } else if constexpr (ReturnDirectDqProjection) {
                TORCH_CHECK(
                    dq_projection_weight != nullptr,
                    "direct dQ projection requires a projection weight"
                );
                dq_projection_output = at::empty(
                    {q.size(0), q.size(1), dq_projection_weight->size(0)},
                    q.options()
                );
                dq_tile_arrivals = at::empty(
                    {q.size(0), q.size(2), ExactQTileCount},
                    q.options().dtype(at::ScalarType::Int)
                );
            } else {
                TORCH_CHECK(
                    dq_projection_weight_fp4 != nullptr &&
                        dq_projection_weight_scales != nullptr &&
                        dq_projection_weight_global_scale != nullptr &&
                        dq_global_scale != nullptr,
                    "tile-ready NVFP4 dQ projection requires packed weight, "
                    "scale pages, and delayed global scales"
                );
                const int64_t rows = q.size(0) * q.size(1);
                const int64_t reduction = hierarchical_qkv_projection
                    ? q.size(2) *
                        (tkfa4::kB300QKDim * 2 + tkfa4::kB300VDim)
                    : q.size(2) * tkfa4::kB300QKDim;
                const int64_t hidden = dq_projection_weight_fp4->size(0);
                TORCH_CHECK(
                    rows % 256 == 0 && reduction % 256 == 0 &&
                        hidden % 256 == 0,
                    "tile-ready NVFP4 dQ projection requires M/K/N "
                    "divisible by 256"
                );
                dq_projection_output = at::empty(
                    {q.size(0), q.size(1), hidden},
                    q.options()
                );
                dq_tile_arrivals = at::empty(
                    {q.size(0), q.size(2), ExactQTileCount},
                    q.options().dtype(at::ScalarType::Int)
                );
                dq_nvfp4 = at::empty(
                    {rows, reduction / 2},
                    q.options().dtype(at::kFloat4_e2m1fn_x2)
                );
                dq_nvfp4_scales = at::empty(
                    {rows / 128, reduction / 64, 512},
                    q.options().dtype(at::kFloat8_e4m3fn)
                );
                dq_nvfp4_ready = at::empty(
                    {rows / 128, reduction / 256},
                    q.options().dtype(at::ScalarType::Int)
                );
                nvfp4_projection = {
                    .input_fp4 = &dq_nvfp4,
                    .input_scales = &dq_nvfp4_scales,
                    .input_global_scale = dq_global_scale,
                    .weight_fp4 = dq_projection_weight_fp4,
                    .weight_scales = dq_projection_weight_scales,
                    .weight_global_scale =
                        dq_projection_weight_global_scale,
                    .output = &dq_projection_output,
                    .operand_ready = &dq_nvfp4_ready,
                    .rope_cos = projection_rope_cos,
                    .rope_sin = projection_rope_sin,
                    .hierarchical_qkv = hierarchical_qkv_projection,
                };
            }
        }
    }

    if constexpr (UseX32Fp8Pv) {
        static_assert(
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp8 ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4Fp8Pv ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4Fp8PvReuseP ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4Fp8DpPvReuseP ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4MixedFp8MxFp4DpFp8DvReuseP ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4Fp8DpMxFp4DvReuseP ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogReuseP ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogSplitQReuseP ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4MxFp4DpDvForwardLogSplitQReuseP ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP
        );
        // Keep the native 32-row score fragment through P and dS packing.
        // This is the previously proved x32 mapping, but unlike the BF16
        // experiment it writes dense E4M3 P directly to TMEM and E4M3 dS
        // directly to its two shared consumers.
        tkfa4::bwd_cute16_candidate::
            launch_backward_cta2_fused_dense_bf16_integrated_frontier<
                HotConfig,
                ExactQTileCount,
                true,  // FenceDsBeforeDkdvReady
                true,  // UseNamedDkdvLocalFanIn
                true,  // LeaderOnlyQdoPublishFence
                true,  // CacheQdoReadyClusterAddress
                true,  // GroupQdoTmaLoads
                true,  // ElectedWideDkQTmaLoad
                true,  // ElectedPeerDoTmaLoad
                true,  // CacheRoleClusterAddresses
                true,  // CacheTensorCommitAddresses
                true,  // EnsureReducerOutputDrain
                true,  // ElectedScoreKTmaLoad
                true,  // UseExactClusterCoordinates
                true,  // EnforceDpTmemConsumerRelease
                true,  // SplitDpTmemConsumerRelease
                true,  // UseIterationCausalMask
                true,  // UseFusedTmemPAndDs
                true,  // OverlapFusedDqAPublication
                true,  // PrefetchNextQdoAfterDkdv
                true,  // PrefetchNextOwnerQdo
                true,  // UseFusedTmemRuntimeAccumulationPredicate
                true,  // UseBitwisePExpansion
                true,  // UseFusedExp2Pack
                true,  // PeelCausalPrefix
                true,  // BranchlessDoSourceLoad
                true,  // BranchlessDoSourceBaseSelect
                false, // PublishVOncePerOwner
                true,  // BulkDoDvStage
                true,  // LoaderOwnedDkQ
                true,  // FuseScoreScaleLse
                true,  // RetainPackedP
                true,  // SplitDirectDpsumAcrossDpDoneWait
                true,  // FusedExp2Fragment4First
                true,  // CarryDirectStatsOffset
                true,  // UseReducerDqLeaderArrive
                true,  // CarryAllRolePhases
                true,  // UseExactDefaultScaleLog2e
                true,  // ReverseDkTailTmemLoadIssue
                true,  // PrearmNextQdoBeforeDkDone
                false, // UseX32TmemComputeLayout
                false, // UseLongSeqStatsCache
                true,  // UseCompactScoreMma
                false, // UseCompactDpMma
                true,  // UsePackedBf16DsProduct
                true,  // SplitDqTmemAndSharedHandoff
                true,  // DistributedDqSharedReadWait
                true,  // BalancedSingleOwnerSchedule
                true,  // UseSingleOwnerWarpStatsCache
                true,  // CacheDqStageLanePointers
                true,  // UseSlicedFp32PForDs
                true,  // UseTmaVWithScoreK
                true,  // UseStatsWarpScoreFanout
                true,  // UseBatchedDqTmemLoads
                true,  // UseDynamicDpReleaseBarrierId
                true,  // PreissueFirstDpHalfBeforeQdoWait
                true,  // OverlapSecondDpLoadWithReleaseBarrier
                true,  // RelayDoDvCompletionViaExchangeWarp
                true,  // OverlapDqPeerCopyWithDoDvCompletion
                true,  // OverlapLocalDqStoreWithPeerCopy
                true,  // UseNonblockingDqPublicationFollowers
                true,  // SplitDqAliasLifetimeWithCuteTmemMap
                true,  // DeferFirstDsTmemStoreWait
                true,  // OverlapFinalDsTmemStoreWithPeerSharedStores
                false, // DelayScoreAliasReleaseUntilFirstDqTailLoad
                false, // InterleaveSteadyScoreExpPairs
                true,  // ShiftOverlappingScoreHalfBeforeDpRelease
                false, // BuildCompactDpDescriptorsAfterWait
                0,     // LateTensorCommitAddressSharedMask
                false, // CacheCompactDpDescriptorsInShared
                true,  // OverlapFirstDpsumQuarterWithSecondPStore
                true,  // HoistReducerDpReadyBeforeScoreWait
                true,  // PipelineFirstDpQuarterLoads
                true,  // PublishNextQdoAtDqAliasRelease
                true,  // JoinNextQdoWithDqAliasRelease
                true,  // PrecomputePostScoreFanoutAddresses
                true,  // PrecomputeScoreIterationDeltaUnderFanout
                true,  // UseNativeX32Lowp
                LowpMode,
                ReuseDqDsForDk,
                UseAdaptiveQkScales,
                ReturnDirectDqProjection,
                UseRank128Score,
                ReturnTileReadyNvfp4DqProjection
            >(
                q,
                k,
                v,
                out,
                lse,
                dout,
                dq,
                dk,
                dv,
                causal,
                softmax_scale,
                deterministic,
                q_lowp,
                k_lowp,
                dq_output_scale,
                dk_output_scale,
                ds_quant_scale,
                score_q_lowp,
                score_k_lowp,
                q_dk_mxfp4,
                k_dq_mxfp4,
                q_dk_nvfp4_scale,
                k_dq_nvfp4_scale,
                mixed_v_prepacked,
                adaptive_qk_scales,
                ReturnBf16Dq ? &dq_bf16 : nullptr,
                ReturnInterleavedQkv ? &projection_qkv : nullptr,
                ReturnDirectDqProjection ? dq_projection_weight : nullptr,
                ReturnDirectDqProjection ? &dq_projection_output : nullptr,
                (ReturnDirectDqProjection ||
                 ReturnTileReadyNvfp4DqProjection)
                    ? &dq_tile_arrivals
                    : nullptr,
                producer_mxfp4,
                ReturnTileReadyNvfp4DqProjection
                    ? &nvfp4_projection
                    : nullptr,
                producer_fp8
        );
        if constexpr (
            ReturnDirectDqProjection ||
            ReturnTileReadyNvfp4DqProjection
        ) {
            return {dq_projection_output, dk, dv};
        } else if constexpr (ReturnInterleavedQkv) {
            return {projection_qkv};
        } else {
            return {ReturnBf16Dq ? dq_bf16 : dq, dk, dv};
        }
    } else {
        // This is the exact retained V382 template selection from tk_fa4.cu.
        // Keeping it explicit makes the copied BF16 control a hard performance
        // invariant while FP8/FP4 operand changes are introduced behind new modes.
        tkfa4::bwd_cute16_candidate::
            launch_backward_cta2_fused_dense_bf16_integrated_frontier<
                HotConfig,
                ExactQTileCount,
                true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, false, true, true,
                true, true, true, true, true, true, true, true, true, true,
                false, false, true, false, true, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, false, false, true, false, 0, false,
                true, true, true, true, true, true, true,
                false,
                LowpMode,
                false,
                UseAdaptiveQkScales,
                ReturnDirectDqProjection,
                UseRank128Score,
                ReturnTileReadyNvfp4DqProjection
            >(
                q,
                k,
                v,
                out,
                lse,
                dout,
                dq,
                dk,
                dv,
                causal,
                softmax_scale,
                deterministic,
                q_lowp,
                k_lowp,
                dq_output_scale,
                dk_output_scale,
                ds_quant_scale,
                score_q_lowp,
                score_k_lowp,
                q_dk_mxfp4,
                k_dq_mxfp4,
                q_dk_nvfp4_scale,
                k_dq_nvfp4_scale,
                mixed_v_prepacked,
                adaptive_qk_scales,
                ReturnBf16Dq ? &dq_bf16 : nullptr,
                ReturnInterleavedQkv ? &projection_qkv : nullptr,
                ReturnDirectDqProjection ? dq_projection_weight : nullptr,
                ReturnDirectDqProjection ? &dq_projection_output : nullptr,
                (ReturnDirectDqProjection ||
                 ReturnTileReadyNvfp4DqProjection)
                    ? &dq_tile_arrivals
                    : nullptr,
                producer_mxfp4,
                ReturnTileReadyNvfp4DqProjection
                    ? &nvfp4_projection
                    : nullptr,
                producer_fp8
            );
        if constexpr (
            ReturnDirectDqProjection ||
            ReturnTileReadyNvfp4DqProjection
        ) {
            return {dq_projection_output, dk, dv};
        } else if constexpr (ReturnInterleavedQkv) {
            return {projection_qkv};
        } else {
            return {ReturnBf16Dq ? dq_bf16 : dq, dk, dv};
        }
    }
}

std::vector<at::Tensor> backward_bf16_control(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_bf16_control_inputs(q, k, v, out, lse, dout, causal, softmax_scale, deterministic);
    switch (q.size(1)) {
        case 4096:
            return launch_v382_mode<32, 4096>(
                q, k, v, out, lse, dout, causal, softmax_scale, deterministic
            );
        case 8192:
            return launch_v382_mode<64, 8192>(
                q, k, v, out, lse, dout, causal, softmax_scale, deterministic
            );
        case 16384:
            return launch_v382_mode<128, 16384>(
                q, k, v, out, lse, dout, causal, softmax_scale, deterministic
            );
        case 32768:
            return launch_v382_mode<256, 32768>(
                q, k, v, out, lse, dout, causal, softmax_scale, deterministic
            );
        case 65536:
            return launch_v382_mode<512, 65536>(
                q, k, v, out, lse, dout, causal, softmax_scale, deterministic
            );
        default:
            TORCH_CHECK(
                false,
                "copied V382 control supports S4096/S8192/S16384/"
                "S32768/S65536"
            );
    }
}

std::vector<at::Tensor> backward_fp8_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp8,
    at::Tensor k_fp8,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_bf16_control_inputs(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
    TORCH_CHECK(
        q.size(1) == 8192 && (q.size(2) == 8 || q.size(2) == 16),
        "the first native FP8 specialization supports B1 S8192 H8/H16"
    );
    TORCH_CHECK(
        q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            k_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "q_fp8 and k_fp8 must use torch.float8_e4m3fn"
    );
    TORCH_CHECK(
        q_fp8.dim() == 4 &&
            q_fp8.size(0) == q.size(0) &&
            q_fp8.size(1) == q.size(2) &&
            q_fp8.size(2) == q.size(3) &&
            q_fp8.size(3) == q.size(1),
        "q_fp8 must be prepacked as contiguous [B, H, D, S]"
    );
    TORCH_CHECK(
        k_fp8.dim() == 4 &&
            k_fp8.size(0) == k.size(0) &&
            k_fp8.size(1) == k.size(1) &&
            k_fp8.size(2) == k.size(2) &&
            k_fp8.size(3) == k.size(3),
        "k_fp8 must be prepacked as contiguous [B, S, H, D]"
    );
    TORCH_CHECK(
        q_fp8.is_cuda() && k_fp8.is_cuda() &&
            q_fp8.is_contiguous() && k_fp8.is_contiguous(),
        "q_fp8 and k_fp8 must be contiguous CUDA tensors"
    );
    kittens::py::device_check(q, q_fp8, k_fp8);
    TORCH_CHECK(
        std::isfinite(q_quant_scale) && q_quant_scale > 0.0f &&
            std::isfinite(k_quant_scale) && k_quant_scale > 0.0f &&
            std::isfinite(ds_quant_scale) && ds_quant_scale > 0.0f,
        "q_quant_scale, k_quant_scale, and ds_quant_scale must be finite and positive"
    );
    TORCH_CHECK(
        ds_quant_scale == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseFp8DefaultDsQuantScale,
        "the optimized native FP8 path requires ds_quant_scale=4096"
    );
    const float dq_output_scale =
        softmax_scale / (ds_quant_scale * k_quant_scale);
    const float dk_output_scale =
        softmax_scale / (ds_quant_scale * q_quant_scale);
    return launch_v382_mode<
        64,
        8192,
        tkfa4::bwd_cute16_kernel_candidate::detail::kCta2DenseLowpFp8,
        true,
        false
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic,
        &q_fp8,
        &k_fp8,
        dq_output_scale,
        dk_output_scale,
        ds_quant_scale
    );
}

template <
    int LowpMode,
    bool UseX32Fp8Pv = false,
    bool ReuseDqDsForDk = false,
    bool UseAdaptiveQkScales = false,
    bool ReturnBf16Dq = false,
    bool ReturnInterleavedQkv = false,
    bool ReturnDirectDqProjection = false,
    bool UseRank128Score = false,
    bool ReturnTileReadyNvfp4DqProjection = false
>
std::vector<at::Tensor> backward_fp4_native_mode(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic,
    at::Tensor *q_dk_mxfp4 = nullptr,
    at::Tensor *k_dq_mxfp4 = nullptr,
    at::Tensor *q_dk_nvfp4_scale = nullptr,
    at::Tensor *k_dq_nvfp4_scale = nullptr,
    at::Tensor *mixed_v_prepacked = nullptr,
    at::Tensor *adaptive_qk_scales = nullptr,
    at::Tensor *dq_projection_weight = nullptr,
    tkfa4::bwd_cute16_candidate::producer_native_mxfp4_operands
        *producer_mxfp4 = nullptr,
    at::Tensor *dq_projection_weight_fp4 = nullptr,
    at::Tensor *dq_projection_weight_scales = nullptr,
    at::Tensor *dq_projection_weight_global_scale = nullptr,
    at::Tensor *dq_global_scale = nullptr,
    tkfa4::bwd_cute16_candidate::producer_native_fp8_operands
        *producer_fp8 = nullptr,
    at::Tensor *projection_rope_cos = nullptr,
    at::Tensor *projection_rope_sin = nullptr,
    bool hierarchical_qkv_projection = false
) {
    static_assert(
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4 ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8Pv ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8PvReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MixedFp8MxFp4DpFp8DvReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogSplitQReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvForwardLogSplitQReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP
    );
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_bf16_control_inputs(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
    constexpr bool kSweepDpFormat =
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MixedFp8MxFp4DpFp8DvReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogSplitQReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvForwardLogSplitQReuseP ||
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP;
    TORCH_CHECK(
        (q.size(1) == 8192 && (q.size(2) == 8 || q.size(2) == 16)) ||
            (kSweepDpFormat &&
             (q.size(1) == 4096 || q.size(1) == 8192 ||
              q.size(1) == 16384) &&
             (q.size(2) == 24 || q.size(2) == 64)),
        "native FP4 route does not support this sequence/head shape"
    );
    TORCH_CHECK(
        q_fp4.scalar_type() == at::ScalarType::Byte &&
            score_q_fp4.scalar_type() == at::ScalarType::Byte &&
            k_fp4.scalar_type() == at::ScalarType::Byte &&
            score_k_fp4.scalar_type() == at::ScalarType::Byte,
        "aligned and compact q_fp4/k_fp4 layouts must use uint8 E2M1 containers"
    );
    TORCH_CHECK(
        q_fp4.dim() == 4 &&
            q_fp4.size(0) == q.size(0) &&
            q_fp4.size(1) == q.size(2) &&
            q_fp4.size(2) == q.size(3) &&
            q_fp4.size(3) == q.size(1),
        "q_fp4 must be aligned-U4 packed as contiguous uint8 [B, H, D, S]"
    );
    TORCH_CHECK(
        score_q_fp4.dim() == 4 &&
            score_q_fp4.size(0) == q.size(0) &&
            score_q_fp4.size(1) == q.size(2) &&
            score_q_fp4.size(2) == q.size(1) &&
            score_q_fp4.size(3) == q.size(3) / 2,
        "score_q_fp4 must be compact E2M1 as contiguous uint8 [B, H, S, D/2]"
    );
    TORCH_CHECK(
        k_fp4.dim() == 4 &&
            k_fp4.size(0) == k.size(0) &&
            k_fp4.size(1) == k.size(1) &&
            k_fp4.size(2) == k.size(2) &&
            k_fp4.size(3) == k.size(3),
        "k_fp4 must be unpacked as contiguous uint8 [B, S, H, D]"
    );
    TORCH_CHECK(
        score_k_fp4.dim() == 4 &&
            score_k_fp4.size(0) == k.size(0) &&
            score_k_fp4.size(1) == k.size(2) &&
            score_k_fp4.size(2) == k.size(1) &&
            score_k_fp4.size(3) == k.size(3) / 2,
        "score_k_fp4 must be compact E2M1 as contiguous uint8 [B, H, S, D/2]"
    );
    TORCH_CHECK(
        q_fp4.is_cuda() && score_q_fp4.is_cuda() && k_fp4.is_cuda() &&
            score_k_fp4.is_cuda() &&
            q_fp4.is_contiguous() && score_q_fp4.is_contiguous() &&
            k_fp4.is_contiguous() && score_k_fp4.is_contiguous(),
        "aligned and compact q_fp4/k_fp4 layouts must be contiguous CUDA tensors"
    );
    kittens::py::device_check(
        q,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4
    );
    if constexpr (UseAdaptiveQkScales) {
        constexpr bool kSupportsAdaptiveQk =
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4Fp8DpPvReuseP ||
            LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
                kCta2DenseLowpFp4MixedFp8MxFp4DpFp8DvReuseP;
        TORCH_CHECK(
            kSupportsAdaptiveQk,
            "adaptive Q/K scales require a retained hybrid specialization"
        );
        TORCH_CHECK(
            adaptive_qk_scales != nullptr,
            "adaptive Q/K specialization requires producer scale metadata"
        );
        TORCH_CHECK(
                adaptive_qk_scales->scalar_type() == at::ScalarType::Float &&
                adaptive_qk_scales->is_cuda() &&
                adaptive_qk_scales->is_contiguous() &&
                adaptive_qk_scales->dim() == 3 &&
                adaptive_qk_scales->size(0) == q.size(0) &&
                adaptive_qk_scales->size(1) == q.size(2) &&
                adaptive_qk_scales->size(2) == 7,
            "adaptive_qk_scales must be contiguous CUDA float32 [B, H, 7] "
            "with [q, k, dQ factor, dK factor, score factor, scratch, "
            "scale word]"
        );
        kittens::py::device_check(q, *adaptive_qk_scales);
    } else {
        TORCH_CHECK(
            adaptive_qk_scales == nullptr,
            "adaptive Q/K metadata requires an adaptive kernel specialization"
        );
    }
    if constexpr (
        LowpMode == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP
    ) {
        TORCH_CHECK(
            q_dk_mxfp4 != nullptr && k_dq_mxfp4 != nullptr,
            "pure MXFP4 dQ/dK requires compact sequence-packed Q/K"
        );
        TORCH_CHECK(
            q_dk_mxfp4->scalar_type() == at::ScalarType::Byte &&
                k_dq_mxfp4->scalar_type() == at::ScalarType::Byte &&
                q_dk_mxfp4->sizes() == k_dq_mxfp4->sizes() &&
                q_dk_mxfp4->dim() == 4 &&
                q_dk_mxfp4->size(0) == q.size(0) &&
                q_dk_mxfp4->size(1) == q.size(2) &&
                q_dk_mxfp4->size(2) ==
#if TK_FA4_BWD_PURE_MXFP4_DQ_N256
                    256 &&
#else
                    q.size(3) &&
#endif
                q_dk_mxfp4->size(3) == q.size(1) / 2,
            "compact MXFP4 Q/K must be uint8 [B, H, D, S/2]"
        );
        TORCH_CHECK(
            q_dk_mxfp4->is_cuda() && k_dq_mxfp4->is_cuda() &&
                q_dk_mxfp4->is_contiguous() &&
                k_dq_mxfp4->is_contiguous(),
            "compact MXFP4 Q/K must be contiguous CUDA tensors"
        );
        kittens::py::device_check(q, *q_dk_mxfp4, *k_dq_mxfp4);
#if TK_FA4_BWD_PURE_NVFP4_QK
        TORCH_CHECK(
            q_quant_scale == 16.0f && k_quant_scale == 16.0f,
            "NVFP4 Q/K currently requires the fused score path's exact x16 "
            "Q/K scale"
        );
        TORCH_CHECK(
            q_dk_nvfp4_scale != nullptr && k_dq_nvfp4_scale != nullptr,
            "NVFP4 Q/K requires prepared per-16 E4M3 scale pages"
        );
        const int64_t expected_scale_pages =
            q.size(0) * q.size(2) * 2 * (q.size(1) / 64);
        TORCH_CHECK(
            q_dk_nvfp4_scale->scalar_type() ==
                    at::ScalarType::Float8_e4m3fn &&
                k_dq_nvfp4_scale->scalar_type() ==
                    at::ScalarType::Float8_e4m3fn &&
                q_dk_nvfp4_scale->sizes() == k_dq_nvfp4_scale->sizes() &&
                q_dk_nvfp4_scale->dim() == 3 &&
                q_dk_nvfp4_scale->size(0) == expected_scale_pages &&
                q_dk_nvfp4_scale->size(1) == 32 &&
                q_dk_nvfp4_scale->size(2) == 16,
            "NVFP4 Q/K scales must be E4M3 [B*H*2*S/64, 32, 16]"
        );
        TORCH_CHECK(
            q_dk_nvfp4_scale->is_cuda() &&
                k_dq_nvfp4_scale->is_cuda() &&
                q_dk_nvfp4_scale->is_contiguous() &&
                k_dq_nvfp4_scale->is_contiguous(),
            "NVFP4 Q/K scales must be contiguous CUDA tensors"
        );
        kittens::py::device_check(
            q,
            *q_dk_nvfp4_scale,
            *k_dq_nvfp4_scale
        );
#endif
    }
    TORCH_CHECK(
        std::isfinite(q_quant_scale) && q_quant_scale > 0.0f &&
            std::isfinite(k_quant_scale) && k_quant_scale > 0.0f &&
            std::isfinite(ds_quant_scale) && ds_quant_scale > 0.0f,
        "q_quant_scale, k_quant_scale, and ds_quant_scale must be finite and positive"
    );
    TORCH_CHECK(
        ds_quant_scale == tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseFp4DefaultDsQuantScale,
        "the optimized native FP4 path requires FP8 ds_quant_scale=4096"
    );

    const float dq_output_scale =
        softmax_scale / (ds_quant_scale * k_quant_scale);
    const float dk_output_scale =
        softmax_scale / (ds_quant_scale * q_quant_scale);
    if constexpr (kSweepDpFormat) {
        if (q.size(1) == 4096) {
            return launch_v382_mode<
                32, 4096, LowpMode, UseX32Fp8Pv, ReuseDqDsForDk,
                UseAdaptiveQkScales, ReturnBf16Dq, ReturnInterleavedQkv,
                ReturnDirectDqProjection, UseRank128Score,
                ReturnTileReadyNvfp4DqProjection
            >(
                q, k, v, out, lse, dout, causal, softmax_scale,
                deterministic, &q_fp4, &k_fp4, dq_output_scale,
                dk_output_scale, ds_quant_scale, &score_q_fp4,
                &score_k_fp4, q_dk_mxfp4, k_dq_mxfp4,
                q_dk_nvfp4_scale, k_dq_nvfp4_scale,
                mixed_v_prepacked, adaptive_qk_scales,
                dq_projection_weight, producer_mxfp4,
                dq_projection_weight_fp4, dq_projection_weight_scales,
                dq_projection_weight_global_scale, dq_global_scale,
                producer_fp8, projection_rope_cos, projection_rope_sin,
                hierarchical_qkv_projection
            );
        }
        if (q.size(1) == 16384) {
            return launch_v382_mode<
                128, 16384, LowpMode, UseX32Fp8Pv, ReuseDqDsForDk,
                UseAdaptiveQkScales, ReturnBf16Dq, ReturnInterleavedQkv,
                ReturnDirectDqProjection, UseRank128Score,
                ReturnTileReadyNvfp4DqProjection
            >(
                q, k, v, out, lse, dout, causal, softmax_scale,
                deterministic, &q_fp4, &k_fp4, dq_output_scale,
                dk_output_scale, ds_quant_scale, &score_q_fp4,
                &score_k_fp4, q_dk_mxfp4, k_dq_mxfp4,
                q_dk_nvfp4_scale, k_dq_nvfp4_scale,
                mixed_v_prepacked, adaptive_qk_scales,
                dq_projection_weight, producer_mxfp4,
                dq_projection_weight_fp4, dq_projection_weight_scales,
                dq_projection_weight_global_scale, dq_global_scale,
                producer_fp8, projection_rope_cos, projection_rope_sin,
                hierarchical_qkv_projection
            );
        }
    }
    return launch_v382_mode<
        64, 8192, LowpMode, UseX32Fp8Pv, ReuseDqDsForDk,
        UseAdaptiveQkScales, ReturnBf16Dq, ReturnInterleavedQkv,
        ReturnDirectDqProjection, UseRank128Score,
        ReturnTileReadyNvfp4DqProjection
    >(
        q, k, v, out, lse, dout, causal, softmax_scale, deterministic,
        &q_fp4, &k_fp4, dq_output_scale, dk_output_scale, ds_quant_scale,
        &score_q_fp4, &score_k_fp4, q_dk_mxfp4, k_dq_mxfp4,
        q_dk_nvfp4_scale, k_dq_nvfp4_scale, mixed_v_prepacked,
        adaptive_qk_scales, dq_projection_weight, producer_mxfp4,
        dq_projection_weight_fp4, dq_projection_weight_scales,
        dq_projection_weight_global_scale, dq_global_scale, producer_fp8,
        projection_rope_cos, projection_rope_sin,
        hierarchical_qkv_projection
    );
}

std::vector<at::Tensor> backward_fp4_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::kCta2DenseLowpFp4
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> backward_fp4_fp8pv_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8Pv
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> backward_fp4_fp8pv_x32_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8Pv,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> backward_fp4_fp8pv_x32_reuse_p_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8PvReuseP,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> backward_fp4_fp8pv_x32_reuse_p_split_dk_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8PvReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> backward_fp4_fp8dpdv_x32_split_dk_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> backward_fp4_fp8dpdv_x32_split_dk_adaptive_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &adaptive_qk_scales
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &adaptive_qk_scales
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,
        true,
        true,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &adaptive_qk_scales
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    at::Tensor dout_fp8,
    at::Tensor v_fp8,
    at::Tensor dpsum,
    at::Tensor lse_log2,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    tkfa4::bwd_cute16_candidate::producer_native_fp8_operands producer{
        .dout_dp = &dout_fp8,
        .v_dp = &v_fp8,
        .dpsum = &dpsum,
        .lse_log2 = &lse_log2,
    };
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,
        true,
        true,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &adaptive_qk_scales,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &producer
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_dout_v_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    at::Tensor dout_fp8,
    at::Tensor v_fp8,
    bool stats_from_packed_dout,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    tkfa4::bwd_cute16_candidate::producer_native_fp8_operands producer{
        .dout_dp = &dout_fp8,
        .v_dp = &v_fp8,
        .dpsum = nullptr,
        .lse_log2 = nullptr,
        .stats_from_packed_dout = stats_from_packed_dout,
    };
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,
        true,
        true,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &adaptive_qk_scales,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &producer
    );
}

std::vector<at::Tensor>
project_stacked_qkv_gradient_nvfp4(
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor projection_weight_fp4,
    at::Tensor projection_weight_scales,
    at::Tensor projection_weight_global_scale,
    at::Tensor gradient_global_scale,
    at::Tensor rope_cos,
    at::Tensor rope_sin
) {
    using PackG = tkfa4_hierarchical_qkv_nvfp4::globals;
    using ProjectionC = tkfa4_projection::config<4, 4>;
    using ProjectionG = tkfa4_projection::globals<ProjectionC>;

    TORCH_CHECK(
        dq.scalar_type() == at::ScalarType::BFloat16 &&
            dk.scalar_type() == at::ScalarType::BFloat16 &&
            dv.scalar_type() == at::ScalarType::BFloat16 &&
            dq.is_cuda() && dk.is_cuda() && dv.is_cuda() &&
            dq.is_contiguous() && dk.is_contiguous() && dv.is_contiguous() &&
            dq.sizes() == dk.sizes() && dq.dim() == 4 && dv.dim() == 4 &&
            dq.size(0) == dv.size(0) && dq.size(1) == dv.size(1) &&
            dq.size(2) == dv.size(2) &&
            dq.size(3) == tkfa4::kB300QKDim &&
            dv.size(3) == tkfa4::kB300VDim,
        "stacked QKV projection requires contiguous CUDA BF16 dQ/dK/dV"
    );
    const int rows = static_cast<int>(dq.size(0) * dq.size(1));
    const int heads = static_cast<int>(dq.size(2));
    const int q_width = heads * tkfa4::kB300QKDim;
    const int v_width = heads * tkfa4::kB300VDim;
    const int reduction = 2 * q_width + v_width;
    const int hidden = static_cast<int>(projection_weight_fp4.size(0));
    const int q_tiles = rows / 128;
    TORCH_CHECK(
        rows % 256 == 0 && reduction % 256 == 0 && hidden % 256 == 0,
        "stacked QKV projection requires M/K/N divisible by 256"
    );
    TORCH_CHECK(
        projection_weight_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            projection_weight_fp4.is_cuda() &&
            projection_weight_fp4.is_contiguous() &&
            projection_weight_fp4.sizes() ==
                at::IntArrayRef({hidden, reduction / 2}) &&
            projection_weight_scales.scalar_type() ==
                at::kFloat8_e4m3fn &&
            projection_weight_scales.is_cuda() &&
            projection_weight_scales.is_contiguous() &&
            projection_weight_scales.sizes() == at::IntArrayRef(
                {hidden / 128, reduction / 64, 512}
            ) &&
            projection_weight_global_scale.scalar_type() ==
                at::ScalarType::Float &&
            projection_weight_global_scale.is_cuda() &&
            projection_weight_global_scale.is_contiguous() &&
            projection_weight_global_scale.numel() == 1 &&
            gradient_global_scale.scalar_type() == at::ScalarType::Float &&
            gradient_global_scale.is_cuda() &&
            gradient_global_scale.is_contiguous() &&
            gradient_global_scale.numel() == 1,
        "stacked QKV projection requires valid cached NVFP4 weight and "
        "gradient scale operands"
    );
    TORCH_CHECK(
        rope_cos.scalar_type() == at::ScalarType::BFloat16 &&
            rope_sin.scalar_type() == at::ScalarType::BFloat16 &&
            rope_cos.is_cuda() && rope_sin.is_cuda() &&
            rope_cos.is_contiguous() && rope_sin.is_contiguous() &&
            rope_cos.sizes() == rope_sin.sizes() &&
            rope_cos.numel() ==
                dq.size(0) * dq.size(1) * (tkfa4::kB300QKDim / 2),
        "stacked QKV projection requires contiguous BF16 RoPE tables"
    );
    kittens::py::device_check(
        dq,
        dk,
        dv,
        projection_weight_fp4,
        projection_weight_scales,
        projection_weight_global_scale,
        gradient_global_scale,
        rope_cos,
        rope_sin
    );

    at::Tensor packed = at::empty(
        {rows, reduction / 2},
        dq.options().dtype(at::kFloat4_e2m1fn_x2)
    );
    at::Tensor scales = at::empty(
        {q_tiles, reduction / 64, 512},
        dq.options().dtype(at::kFloat8_e4m3fn)
    );
    at::Tensor projected = at::empty(
        {dq.size(0), dq.size(1), hidden},
        dq.options()
    );
    auto stream = at::cuda::getCurrentCUDAStream();
    PackG pack_g{
        .dQ = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dq, 1, 1, rows, q_width
        ),
        .dK = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dk, 1, 1, rows, q_width
        ),
        .dV = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dv, 1, 1, rows, v_width
        ),
        .A = kittens::py::tensor_to_gl<typename PackG::fp4_gl>(
            packed, 1, 1, rows, reduction / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename PackG::scale_gl, false>(
            scales, 1, q_tiles, reduction / 64, 256
        ),
        .A_scale = kittens::py::tensor_to_gl<
            typename PackG::global_scale_gl
        >(gradient_global_scale),
        .rope_cos = reinterpret_cast<const kittens::bf16 *>(
            rope_cos.data_ptr()
        ),
        .rope_sin = reinterpret_cast<const kittens::bf16 *>(
            rope_sin.data_ptr()
        ),
        .rows = rows,
        .q_width = q_width,
        .v_width = v_width,
        .dq_reduction_lanes = 1,
    };
    tkfa4_hierarchical_qkv_nvfp4::launch(pack_g, stream.stream());

    ProjectionG projection_g{
        .A = kittens::py::tensor_to_gl<typename ProjectionG::A_gl>(
            packed, 1, 1, rows, reduction / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<
            typename ProjectionG::A_sc_gl,
            false
        >(scales, 1, q_tiles, reduction / 64, 256),
        .A_scale = kittens::py::tensor_to_gl<typename ProjectionG::scale_gl>(
            gradient_global_scale
        ),
        .B = kittens::py::tensor_to_gl<typename ProjectionG::B_gl>(
            projection_weight_fp4, 1, 1, hidden, reduction / 2
        ),
        .B_sc = kittens::py::tensor_to_gl<
            typename ProjectionG::B_sc_gl,
            false
        >(
            projection_weight_scales,
            1,
            hidden / 128,
            reduction / 64,
            256
        ),
        .B_scale = kittens::py::tensor_to_gl<typename ProjectionG::scale_gl>(
            projection_weight_global_scale
        ),
        .Q = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
            projected, 1, 1, rows, hidden
        ),
        .K = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
            projected, 1, 1, rows, hidden
        ),
        .V = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
            projected, 1, 1, rows, hidden
        ),
        .D = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
            projected, 1, 1, rows, hidden
        ),
        .output_width = hidden,
        .A_ready = nullptr,
        .A_ready_reduction_tiles = 0,
        .cluster_cap = 0,
        .A_ready_expected = 1u,
        .block_begin = 0,
        .block_end = 0,
    };
    tkfa4_projection::launch_on_stream<
        ProjectionC,
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        true,
        false
    >(projection_g, stream.stream());
    return {projected};
}

std::vector<at::Tensor>
project_gqa_d128_hierarchical_qkv_gradient_nvfp4(
    at::Tensor dq_or_lanes,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor projection_weight_fp4,
    at::Tensor projection_weight_scales,
    at::Tensor projection_weight_global_scale,
    at::Tensor gradient_global_scale,
    at::Tensor rope_packed,
    double dq_decode_scale,
    double dk_decode_scale,
    double dv_decode_scale
) {
    using PackG =
        tkfa4_gqa_d128_hierarchical_qkv_nvfp4::globals;
    using ProjectionC = tkfa4_projection::config<4, 4, 128>;
    using ProjectionG = tkfa4_projection::globals<ProjectionC>;
    constexpr int kDepth = 128;

    TORCH_CHECK(
        dk.scalar_type() == at::ScalarType::BFloat16 &&
            dv.scalar_type() == at::ScalarType::BFloat16 &&
            dk.is_cuda() && dv.is_cuda() &&
            dk.is_contiguous() && dv.is_contiguous() &&
            dk.sizes() == dv.sizes() && dk.dim() == 4 &&
            dk.size(0) > 0 && dk.size(1) > 0 && dk.size(2) > 0 &&
            dk.size(3) == kDepth,
        "D128 GQA projection requires matching contiguous CUDA BF16 "
        "dK/dV [B,S,Hkv,128]"
    );
    const bool hierarchical = dq_or_lanes.dim() == 5;
    const int64_t batch = dk.size(0);
    const int64_t sequence = dk.size(1);
    const int64_t kv_heads = dk.size(2);
    int64_t q_heads = 0;
    int dq_reduction_lanes = 1;
    if (hierarchical) {
        TORCH_CHECK(
            batch == 1,
            "hierarchical D128 GQA projection currently requires B=1"
        );
        TORCH_CHECK(
            dq_or_lanes.scalar_type() == at::ScalarType::BFloat16 &&
                dq_or_lanes.is_cuda() && dq_or_lanes.is_contiguous() &&
                dq_or_lanes.size(0) == 2 &&
                dq_or_lanes.size(1) == batch &&
                dq_or_lanes.size(3) == sequence &&
                dq_or_lanes.size(4) == kDepth,
            "hierarchical dQ must be contiguous CUDA BF16 "
            "[2,B,Hq,S,128]"
        );
        dq_reduction_lanes = 2;
        q_heads = dq_or_lanes.size(2);
    } else {
        TORCH_CHECK(
            batch == 1 || batch == 2 || batch == 4,
            "materialized D128 GQA projection is authenticated only for "
            "batch 1, 2, or 4"
        );
        TORCH_CHECK(
            dq_or_lanes.scalar_type() == at::ScalarType::BFloat16 &&
                dq_or_lanes.is_cuda() && dq_or_lanes.is_contiguous() &&
                dq_or_lanes.dim() == 4 &&
                dq_or_lanes.size(0) == batch &&
                dq_or_lanes.size(1) == sequence &&
                dq_or_lanes.size(3) == kDepth,
            "materialized dQ must be contiguous CUDA BF16 [B,S,Hq,128]"
        );
        q_heads = dq_or_lanes.size(2);
    }
    TORCH_CHECK(
        q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0,
        "D128 GQA projection requires Hq divisible by Hkv"
    );

    const int64_t rows = batch * sequence;
    const int64_t q_width = q_heads * kDepth;
    const int64_t kv_width = kv_heads * kDepth;
    const int64_t reduction = q_width + 2 * kv_width;
    const int64_t hidden = projection_weight_fp4.size(0);
    TORCH_CHECK(
        (batch != 2 && batch != 4) || (
            sequence == 4096 && q_heads == 32 && kv_heads == 8 &&
            hidden == 4096
        ),
        "materialized D128 B2/B4 projection is authenticated only for "
        "S4096/H4096/Hq32/Hkv8"
    );
    TORCH_CHECK(
        rows % 256 == 0 && reduction % 256 == 0 && hidden % 256 == 0,
        "D128 GQA projection requires M/K/N divisible by 256"
    );
    TORCH_CHECK(
        projection_weight_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            projection_weight_fp4.is_cuda() &&
            projection_weight_fp4.is_contiguous() &&
            projection_weight_fp4.sizes() ==
                at::IntArrayRef({hidden, reduction / 2}),
        "D128 GQA projection weight payload must be packed E2M1 "
        "[hidden,(Hq+2*Hkv)*64]"
    );
    TORCH_CHECK(
        projection_weight_scales.scalar_type() ==
                at::ScalarType::Float8_e4m3fn &&
            projection_weight_scales.is_cuda() &&
            projection_weight_scales.is_contiguous() &&
            projection_weight_scales.sizes() == at::IntArrayRef({
                hidden / 128, reduction / 64, 512
            }),
        "D128 GQA projection weight scales must be E4M3 "
        "[hidden/128,K/64,512]"
    );
    for (const auto &[name, scale] : {
             std::pair<const char *, at::Tensor>{
                 "projection weight global scale",
                 projection_weight_global_scale
             },
             std::pair<const char *, at::Tensor>{
                 "gradient global scale",
                 gradient_global_scale
             },
         }) {
        TORCH_CHECK(
            scale.scalar_type() == at::ScalarType::Float &&
                scale.is_cuda() && scale.is_contiguous() &&
                scale.numel() == 1,
            name, " must be one contiguous CUDA float32 value"
        );
    }
    TORCH_CHECK(
        rope_packed.scalar_type() == at::ScalarType::Int &&
            rope_packed.is_cuda() && rope_packed.is_contiguous() &&
            rope_packed.sizes() ==
                at::IntArrayRef({batch, sequence, kDepth / 2}),
        "D128 packed RoPE must be contiguous CUDA int32 [B,S,64]"
    );
    TORCH_CHECK(
        std::isfinite(dq_decode_scale) && dq_decode_scale > 0.0 &&
            std::isfinite(dk_decode_scale) && dk_decode_scale > 0.0 &&
            std::isfinite(dv_decode_scale) && dv_decode_scale > 0.0,
        "D128 Q/K/V decode scales must be finite and positive"
    );
    kittens::py::device_check(
        dq_or_lanes,
        dk,
        dv,
        projection_weight_fp4,
        projection_weight_scales,
        projection_weight_global_scale,
        gradient_global_scale,
        rope_packed
    );
    const c10::cuda::CUDAGuard device_guard(dq_or_lanes.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "D128 GQA hierarchical projection requires GB200 / SM100"
    );

    at::Tensor packed = at::empty(
        {rows, reduction / 2},
        dq_or_lanes.options().dtype(at::kFloat4_e2m1fn_x2)
    );
    at::Tensor scales = at::empty(
        {rows / 128, reduction / 64, 512},
        dq_or_lanes.options().dtype(at::kFloat8_e4m3fn)
    );
    at::Tensor projected = at::empty(
        {batch, sequence, hidden},
        dq_or_lanes.options()
    );
    at::Tensor dq_storage = hierarchical
        ? dq_or_lanes.view({
              dq_reduction_lanes * batch * q_heads * sequence,
              kDepth
          })
        : dq_or_lanes;
    auto stream = at::cuda::getCurrentCUDAStream();
    PackG pack_g{
        .dQ = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dq_storage,
            1,
            1,
            hierarchical
                ? dq_reduction_lanes * q_heads * rows
                : rows,
            hierarchical ? kDepth : q_width
        ),
        .dK = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dk, 1, 1, rows, kv_width
        ),
        .dV = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dv, 1, 1, rows, kv_width
        ),
        .A = kittens::py::tensor_to_gl<typename PackG::fp4_gl>(
            packed, 1, 1, rows, reduction / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename PackG::scale_gl, false>(
            scales, 1, rows / 128, reduction / 64, 256
        ),
        .A_scale = kittens::py::tensor_to_gl<
            typename PackG::global_scale_gl
        >(gradient_global_scale),
        .rope_packed = reinterpret_cast<const uint32_t *>(
            rope_packed.data_ptr()
        ),
        .rows = static_cast<int>(rows),
        .q_heads = static_cast<int>(q_heads),
        .kv_heads = static_cast<int>(kv_heads),
        .dq_reduction_lanes = dq_reduction_lanes,
        .dq_head_major = hierarchical,
        .dq_decode_scale = static_cast<float>(dq_decode_scale),
        .dk_decode_scale = static_cast<float>(dk_decode_scale),
        .dv_decode_scale = static_cast<float>(dv_decode_scale),
    };
    tkfa4_gqa_d128_hierarchical_qkv_nvfp4::launch(
        pack_g,
        stream.stream()
    );

    ProjectionG projection_g{
        .A = kittens::py::tensor_to_gl<typename ProjectionG::A_gl>(
            packed, 1, 1, rows, reduction / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<
            typename ProjectionG::A_sc_gl,
            false
        >(scales, 1, rows / 128, reduction / 64, 256),
        .A_scale = kittens::py::tensor_to_gl<typename ProjectionG::scale_gl>(
            gradient_global_scale
        ),
        .B = kittens::py::tensor_to_gl<typename ProjectionG::B_gl>(
            projection_weight_fp4,
            1,
            1,
            hidden,
            reduction / 2
        ),
        .B_sc = kittens::py::tensor_to_gl<
            typename ProjectionG::B_sc_gl,
            false
        >(
            projection_weight_scales,
            1,
            hidden / 128,
            reduction / 64,
            256
        ),
        .B_scale = kittens::py::tensor_to_gl<typename ProjectionG::scale_gl>(
            projection_weight_global_scale
        ),
        .Q = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
            projected, 1, 1, rows, hidden
        ),
        .K = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
            projected, 1, 1, rows, hidden
        ),
        .V = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
            projected, 1, 1, rows, hidden
        ),
        .D = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
            projected, 1, 1, rows, hidden
        ),
        .output_width = static_cast<int>(hidden),
    };
    tkfa4_projection::launch_on_stream<
        ProjectionC,
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        true
    >(projection_g, stream.stream());
    return {projected, packed, scales};
}

void pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
    at::Tensor dq_or_lanes,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor gradient_global_scale,
    at::Tensor rope_packed,
    at::Tensor packed,
    at::Tensor scales,
    at::Tensor dq_tile_arrivals,
    int64_t row_tile_begin,
    int64_t row_tile_end,
    int64_t col_tile_begin,
    int64_t col_tile_end,
    int64_t arrival_epoch
) {
    using PackG =
        tkfa4_gqa_d128_hierarchical_qkv_nvfp4::globals;
    constexpr int kDepth = 128;
    TORCH_CHECK(
        dk.scalar_type() == at::ScalarType::BFloat16 &&
            dv.scalar_type() == at::ScalarType::BFloat16 &&
            dk.is_cuda() && dv.is_cuda() &&
            dk.is_contiguous() && dv.is_contiguous() &&
            dk.sizes() == dv.sizes() && dk.dim() == 4 &&
            dk.size(3) == kDepth,
        "tile-ready D128 dK/dV must be matching contiguous CUDA BF16 "
        "[1,S,Hkv,128]"
    );
    const int64_t batch = dk.size(0);
    const int64_t sequence = dk.size(1);
    TORCH_CHECK(
        batch == 1,
        "tile-ready D128 GQA pack is authenticated only for batch 1"
    );
    const bool hierarchical = dq_or_lanes.dim() == 5;
    int64_t q_heads = 0;
    int dq_reduction_lanes = 1;
    if (hierarchical) {
        TORCH_CHECK(
            batch == 1 &&
                dq_or_lanes.scalar_type() == at::ScalarType::BFloat16 &&
                dq_or_lanes.is_cuda() && dq_or_lanes.is_contiguous() &&
                dq_or_lanes.size(0) == 2 &&
                dq_or_lanes.size(1) == batch &&
                dq_or_lanes.size(3) == sequence &&
                dq_or_lanes.size(4) == kDepth,
            "tile-ready hierarchical dQ must be contiguous CUDA BF16 "
            "[2,1,Hq,S,128]"
        );
        dq_reduction_lanes = 2;
        q_heads = dq_or_lanes.size(2);
    } else {
        TORCH_CHECK(
            dq_or_lanes.scalar_type() == at::ScalarType::BFloat16 &&
                dq_or_lanes.is_cuda() && dq_or_lanes.is_contiguous() &&
                dq_or_lanes.dim() == 4 &&
                dq_or_lanes.size(0) == batch &&
                dq_or_lanes.size(1) == sequence &&
                dq_or_lanes.size(3) == kDepth,
            "tile-ready materialized dQ must be contiguous CUDA BF16 "
            "[B,S,Hq,128]"
        );
        q_heads = dq_or_lanes.size(2);
    }
    const int64_t kv_heads = dk.size(2);
    TORCH_CHECK(
        q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 &&
            sequence % PackG::TILE_M == 0,
        "tile-ready D128 GQA requires Hq divisible by Hkv and S divisible "
        "by 128"
    );
    const int64_t q_tiles = sequence / PackG::TILE_M;
    const int64_t total_cols = q_heads + 2 * kv_heads;
    const int64_t reduction = total_cols * kDepth;
    TORCH_CHECK(
        row_tile_begin >= 0 && row_tile_begin < row_tile_end &&
            row_tile_end <= q_tiles && col_tile_begin >= 0 &&
            col_tile_begin < col_tile_end && col_tile_end <= total_cols,
        "tile-ready D128 pack range is outside the QKV operand"
    );
    TORCH_CHECK(
        gradient_global_scale.scalar_type() == at::ScalarType::Float &&
            gradient_global_scale.is_cuda() &&
            gradient_global_scale.is_contiguous() &&
            gradient_global_scale.numel() == 1,
        "tile-ready D128 pack requires one CUDA float32 gradient scale"
    );
    TORCH_CHECK(
        rope_packed.scalar_type() == at::ScalarType::Int &&
            rope_packed.is_cuda() && rope_packed.is_contiguous() &&
            rope_packed.sizes() ==
                at::IntArrayRef({batch, sequence, kDepth / 2}),
        "tile-ready D128 pack requires int32 RoPE [1,S,64]"
    );
    TORCH_CHECK(
        packed.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            packed.is_cuda() && packed.is_contiguous() &&
            packed.sizes() == at::IntArrayRef({sequence, reduction / 2}),
        "tile-ready D128 packed output must be E2M1 [S,K/2]"
    );
    TORCH_CHECK(
        scales.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            scales.is_cuda() && scales.is_contiguous() &&
            scales.sizes() ==
                at::IntArrayRef({q_tiles, reduction / 64, 512}),
        "tile-ready D128 scales must be E4M3 [S/128,K/64,512]"
    );
    TORCH_CHECK(
        dq_tile_arrivals.scalar_type() == at::ScalarType::Int &&
            dq_tile_arrivals.is_cuda() &&
            dq_tile_arrivals.is_contiguous() &&
            dq_tile_arrivals.sizes() ==
                at::IntArrayRef({batch, q_heads, q_tiles}),
        "tile-ready dQ arrivals must be int32 [1,Hq,S/128]"
    );
    TORCH_CHECK(
        arrival_epoch >= 0 &&
            arrival_epoch <=
                static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "tile-ready dQ arrival epoch exceeds uint32 range"
    );
    kittens::py::device_check(
        dq_or_lanes,
        dk,
        dv,
        gradient_global_scale,
        rope_packed,
        packed,
        scales,
        dq_tile_arrivals
    );
    const c10::cuda::CUDAGuard device_guard(dq_or_lanes.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "tile-ready D128 GQA pack requires GB200 / SM100"
    );
    auto stream = at::cuda::getCurrentCUDAStream();
    if (arrival_epoch > 0) {
        TORCH_CHECK(
            col_tile_begin == 0 && col_tile_end <= q_heads,
            "arrival waits are valid only for the dQ operand prefix"
        );
        const uint32_t expected = static_cast<uint32_t>(arrival_epoch);
        auto *arrival_base = reinterpret_cast<const uint32_t *>(
            dq_tile_arrivals.data_ptr()
        );
        wait_dq_owner_prefix_kernel<<<1, 256, 0, stream.stream()>>>(
            arrival_base,
            static_cast<int>(col_tile_begin),
            static_cast<int>(col_tile_end),
            static_cast<int>(row_tile_end),
            static_cast<int>(q_tiles),
            expected
        );
        CUDACHECK(cudaGetLastError());
    }

    at::Tensor dq_storage = hierarchical
        ? dq_or_lanes.view({
              dq_reduction_lanes * batch * q_heads * sequence,
              kDepth
          })
        : dq_or_lanes;
    PackG pack_g{
        .dQ = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dq_storage,
            1,
            1,
            hierarchical
                ? dq_reduction_lanes * q_heads * sequence
                : sequence,
            hierarchical ? kDepth : q_heads * kDepth
        ),
        .dK = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dk, 1, 1, sequence, kv_heads * kDepth
        ),
        .dV = kittens::py::tensor_to_gl<typename PackG::bf16_gl, false>(
            dv, 1, 1, sequence, kv_heads * kDepth
        ),
        .A = kittens::py::tensor_to_gl<typename PackG::fp4_gl>(
            packed, 1, 1, sequence, reduction / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename PackG::scale_gl, false>(
            scales, 1, q_tiles, reduction / 64, 256
        ),
        .A_scale = kittens::py::tensor_to_gl<
            typename PackG::global_scale_gl
        >(gradient_global_scale),
        .rope_packed = reinterpret_cast<const uint32_t *>(
            rope_packed.data_ptr()
        ),
        .rows = static_cast<int>(sequence),
        .q_heads = static_cast<int>(q_heads),
        .kv_heads = static_cast<int>(kv_heads),
        .dq_reduction_lanes = dq_reduction_lanes,
        .dq_head_major = hierarchical,
        .row_tile_begin = static_cast<int>(row_tile_begin),
        .row_tile_end = static_cast<int>(row_tile_end),
        .col_tile_begin = static_cast<int>(col_tile_begin),
        .col_tile_end = static_cast<int>(col_tile_end),
    };
    tkfa4_gqa_d128_hierarchical_qkv_nvfp4::launch(
        pack_g,
        stream.stream()
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_hierarchical_qkv_projection_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    at::Tensor dout_fp8,
    at::Tensor v_fp8,
    bool stats_from_packed_dout,
    at::Tensor projection_weight_fp4,
    at::Tensor projection_weight_scales,
    at::Tensor projection_weight_global_scale,
    at::Tensor gradient_global_scale,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    tkfa4::bwd_cute16_candidate::producer_native_fp8_operands producer{
        .dout_dp = &dout_fp8,
        .v_dp = &v_fp8,
        .dpsum = nullptr,
        .lse_log2 = nullptr,
        .stats_from_packed_dout = stats_from_packed_dout,
    };
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,   // UseX32Fp8Pv
        true,   // ReuseDqDsForDk
        true,   // UseAdaptiveQkScales
        false,  // ReturnBf16Dq
        false,  // ReturnInterleavedQkv
        false,  // ReturnDirectDqProjection
        false,  // UseRank128Score
        true    // ReturnTileReadyNvfp4DqProjection
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &adaptive_qk_scales,
        nullptr,
        nullptr,
        &projection_weight_fp4,
        &projection_weight_scales,
        &projection_weight_global_scale,
        &gradient_global_scale,
        &producer,
        &rope_cos,
        &rope_sin,
        true
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_stacked_qkv_projection_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    at::Tensor dout_fp8,
    at::Tensor v_fp8,
    bool stats_from_packed_dout,
    at::Tensor projection_weight_fp4,
    at::Tensor projection_weight_scales,
    at::Tensor projection_weight_global_scale,
    at::Tensor gradient_global_scale,
    at::Tensor rope_cos,
    at::Tensor rope_sin,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    tkfa4::bwd_cute16_candidate::producer_native_fp8_operands producer{
        .dout_dp = &dout_fp8,
        .v_dp = &v_fp8,
        .dpsum = nullptr,
        .lse_log2 = nullptr,
        .stats_from_packed_dout = stats_from_packed_dout,
    };
    auto gradients = backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,   // UseX32Fp8Pv
        true,   // ReuseDqDsForDk
        true,   // UseAdaptiveQkScales
        true    // ReturnBf16Dq
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &adaptive_qk_scales,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &producer
    );
    at::Tensor projected = project_stacked_qkv_gradient_nvfp4(
        gradients[0],
        gradients[1],
        gradients[2],
        projection_weight_fp4,
        projection_weight_scales,
        projection_weight_global_scale,
        gradient_global_scale,
        rope_cos,
        rope_sin
    )[0];
    return {projected, gradients[1], gradients[2]};
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_v_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    at::Tensor v_fp8,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,
        true,
        true,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &v_fp8,
        &adaptive_qk_scales
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_direct_dq_projection_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    at::Tensor dq_projection_weight,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    const bool force_pipelined =
        std::getenv("TK_FA4_DQ_PROJECTION_FORCE_PIPELINED") != nullptr;
    if (q.size(2) > 8 && !force_pipelined) {
        // The bounded in-flight consumer wins at H8.  Wider projections leave
        // too much work in its fixed-size tail, so keep the exact BF16 dQ
        // reduction and let the vendor GEMM consume it after attention.  This
        // prevents the experimental topology from regressing H16/H24/H64
        // while retaining an environment-controlled profiling rung.
        const int64_t reduction = q.size(2) * tkfa4::kB300QKDim;
        TORCH_CHECK(
            dq_projection_weight.scalar_type() == at::kBFloat16 &&
                dq_projection_weight.is_cuda() &&
                dq_projection_weight.is_contiguous() &&
                dq_projection_weight.dim() == 2 &&
                dq_projection_weight.size(1) == reduction,
            "dQ projection weight must be contiguous BF16 "
            "[hidden, heads * 192]"
        );
        auto gradients =
            backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native(
                q,
                k,
                v,
                out,
                lse,
                dout,
                q_fp4,
                score_q_fp4,
                k_fp4,
                score_k_fp4,
                adaptive_qk_scales,
                ds_quant_scale,
                causal,
                softmax_scale,
                deterministic
            );
        auto projected = at::mm(
            gradients[0].reshape({q.size(1), reduction}),
            dq_projection_weight.transpose(0, 1)
        ).view({q.size(0), q.size(1), dq_projection_weight.size(0)});
        return {projected, gradients[1], gradients[2]};
    }
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP,
        true,
        true,
        true,
        false,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &adaptive_qk_scales,
        &dq_projection_weight
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dpdv_x32_split_dk_adaptive_nvfp4_dq_projection_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    at::Tensor dq_projection_weight_fp4,
    at::Tensor dq_projection_weight_scales,
    at::Tensor dq_projection_weight_global_scale,
    at::Tensor dq_global_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    // Keep the retained spill-free attention specialization intact, then use
    // the established wide delayed-scale quantizer.  Attempts to quantize in
    // the final persistent attention contributor serialize work over one
    // cluster per head; bounded polling grids likewise lose more through
    // interference than they hide.  This composition is the measured winner
    // until dQ reduction itself is absorbed by projection backward.
    auto gradients =
        backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        adaptive_qk_scales,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
    const int64_t rows = q.size(0) * q.size(1);
    const int64_t reduction = q.size(2) * tkfa4::kB300QKDim;
    auto dq_matrix = gradients[0].view({rows, reduction});
    auto operand = quantize_nvfp4_projection_operand_precomputed_scale(
        dq_matrix,
        dq_global_scale
    );
    at::Tensor projected = project_nvfp4_generic(
        operand[0],
        operand[1],
        operand[2],
        dq_projection_weight_fp4,
        dq_projection_weight_scales,
        dq_projection_weight_global_scale
    );
    projected = projected.view(
        {q.size(0), q.size(1), dq_projection_weight_fp4.size(0)}
    );
    return {projected, gradients[1], gradients[2]};
}

std::vector<at::Tensor>
backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MixedFp8MxFp4DpFp8DvReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor>
backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_prepacked_v_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic,
    at::Tensor mixed_v_prepacked
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MixedFp8MxFp4DpFp8DvReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &mixed_v_prepacked
    );
}

std::vector<at::Tensor>
backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_adaptive_prepacked_v_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor adaptive_qk_scales,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic,
    at::Tensor mixed_v_prepacked
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MixedFp8MxFp4DpFp8DvReuseP,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        1.0f,
        1.0f,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &mixed_v_prepacked,
        &adaptive_qk_scales
    );
}

std::vector<at::Tensor> backward_fp4_fp8dp_mxfp4dv_x32_split_dk_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor>
backward_fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogSplitQReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor>
backward_fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvForwardLogSplitQReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor>
backward_fp4_mxfp4dpdvdsdqdk_forward_log_split_q_x32_native(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor q_dk_mxfp4,
    at::Tensor k_dq_mxfp4,
    at::Tensor q_dk_nvfp4_scale,
    at::Tensor k_dq_nvfp4_scale,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        &q_dk_mxfp4,
        &k_dq_mxfp4,
        &q_dk_nvfp4_scale,
        &k_dq_nvfp4_scale
    );
}

std::vector<at::Tensor>
backward_fp4_mxfp4dpdvdsdqdk_producer_native_x32(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor q_dk_mxfp4,
    at::Tensor k_dq_mxfp4,
    at::Tensor q_dk_nvfp4_scale,
    at::Tensor k_dq_nvfp4_scale,
    at::Tensor dout_dp,
    at::Tensor v_dp,
    at::Tensor dout_dp_scale,
    at::Tensor v_dp_scale,
    at::Tensor dout_dv,
    at::Tensor dout_dv_scale,
    at::Tensor dpsum,
    at::Tensor lse_log2,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    tkfa4::bwd_cute16_candidate::producer_native_mxfp4_operands operands{
        &dout_dp,
        &v_dp,
        &dout_dp_scale,
        &v_dp_scale,
        &dout_dv,
        &dout_dv_scale,
        &dpsum,
        &lse_log2,
    };
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        &q_dk_mxfp4,
        &k_dq_mxfp4,
        &q_dk_nvfp4_scale,
        &k_dq_nvfp4_scale,
        nullptr,
        nullptr,
        nullptr,
        &operands
    );
}

// Model-co-designed rank-128 Q/K route.  The public tensor contract stays
// D192 so gradients retain their standard shape, but the model must constrain
// the final 64 Q/K coordinates to zero (or otherwise outside the trained
// subspace).  Under that contract the third score K64 is algebraically empty
// and this specialization omits it without approximating a D192 model.
std::vector<at::Tensor>
backward_fp4_rank128_mxfp4dpdvdsdqdk_producer_native_x32(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor q_fp4,
    at::Tensor score_q_fp4,
    at::Tensor k_fp4,
    at::Tensor score_k_fp4,
    at::Tensor q_dk_mxfp4,
    at::Tensor k_dq_mxfp4,
    at::Tensor q_dk_nvfp4_scale,
    at::Tensor k_dq_nvfp4_scale,
    at::Tensor dout_dp,
    at::Tensor v_dp,
    at::Tensor dout_dp_scale,
    at::Tensor v_dp_scale,
    at::Tensor dout_dv,
    at::Tensor dout_dv_scale,
    at::Tensor dpsum,
    at::Tensor lse_log2,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    tkfa4::bwd_cute16_candidate::producer_native_mxfp4_operands operands{
        &dout_dp,
        &v_dp,
        &dout_dp_scale,
        &v_dp_scale,
        &dout_dv,
        &dout_dv_scale,
        &dpsum,
        &lse_log2,
    };
    return backward_fp4_native_mode<
        tkfa4::bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP,
        true,
        true,
        false,
        false,
        false,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic,
        &q_dk_mxfp4,
        &k_dq_mxfp4,
        &q_dk_nvfp4_scale,
        &k_dq_nvfp4_scale,
        nullptr,
        nullptr,
        nullptr,
        &operands
    );
}

std::vector<at::Tensor> backward_fp4_fused_quant(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    float q_quant_scale,
    float k_quant_scale,
    float ds_quant_scale,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    // Keep packing and backward submission in one extension call.  Besides
    // reducing host launch gaps, this makes the optimized tiled quantizer the
    // default end-to-end route instead of relying on Python to preserve four
    // temporary layout tensors.
    auto packed = quantize_fp4_dual_qk_unpacked(
        q,
        k,
        q_quant_scale,
        k_quant_scale
    );
    return backward_fp4_native(
        q,
        k,
        v,
        out,
        lse,
        dout,
        packed[0],
        packed[1],
        packed[2],
        packed[3],
        q_quant_scale,
        k_quant_scale,
        ds_quant_scale,
        causal,
        softmax_scale,
        deterministic
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "backward_bf16_control",
        &backward_bf16_control,
        "Fresh low-precision branch: exact copied TK V382 BF16 control"
    );
    m.def(
        "backward_fp8_native",
        &backward_fp8_native,
        "V382 schedule with native FP8 dSxQ and dS^TxK tensor-core MMAs"
    );
    m.def(
        "quantize_fp4_bhds_unpacked",
        &quantize_fp4_bhds_unpacked,
        "Quantize BSHD BF16 to BHDS aligned E2M1 byte containers"
    );
    m.def(
        "quantize_fp4_bshd_unpacked",
        &quantize_fp4_bshd_unpacked,
        "Quantize BSHD BF16 to row-major BSHD aligned E2M1 containers"
    );
    m.def(
        "quantize_fp4_dual_q_unpacked",
        &quantize_fp4_dual_q_unpacked,
        "Fused Q prepack producing aligned sequence and compact score views"
    );
    m.def(
        "quantize_fp4_dual_k_unpacked",
        &quantize_fp4_dual_k_unpacked,
        "Fused K prepack producing aligned dQ and compact score views"
    );
    m.def(
        "quantize_fp4_dual_qk_unpacked",
        &quantize_fp4_dual_qk_unpacked,
        "Single-pass Q/K E2M1 prepack reusing Q codes across both layouts"
    );
    m.def(
        "quantize_fp4_dual_qk_blockscale",
        &quantize_fp4_dual_qk_blockscale,
        "Single-pass Q/K E2M1 prepack with compact sequence operands for "
        "block-scaled dQ/dK"
    );
    m.def(
        "quantize_fp4_dual_qk_adaptive",
        &quantize_fp4_dual_qk_adaptive,
        "Robust per-head adaptive Q/K E2M1 producer with device-resident "
        "dequantization metadata"
    );
    m.def(
        "quantize_fp4_dual_qk_precomputed_scales",
        &quantize_fp4_dual_qk_precomputed_scales,
        "Q/K E2M1 producer consuming scale metadata emitted by an upstream "
        "projection"
    );
    m.def(
        "quantize_e4m3_projection_operand",
        &quantize_e4m3_projection_operand,
        "Prepare row-scaled dense-E4M3 projection activations in one fused "
        "CUDA kernel"
    );
    m.def(
        "quantize_e4m3_projection_weight",
        &quantize_e4m3_projection_weight,
        "Prepare channel-scaled dense-E4M3 projection weights in one fused "
        "CUDA kernel"
    );
    m.def(
        "convert_e4m3_x4_v_bhds_to_causal_mxfp4",
        &convert_e4m3_x4_v_bhds_to_causal_mxfp4,
        "Experimental register-only E4M3(x4) [B,H,64,S] to "
        "causal-interleaved 1x32 MXFP4 V publisher; finite bytes are exact "
        "and NaN groups emit the E8M0 NaN sentinel"
    );
    m.def(
        "convert_e4m3_x4_v_bhds_to_causal_mxfp4_out",
        &convert_e4m3_x4_v_bhds_to_causal_mxfp4_out,
        "Experimental caller-owned E4M3(x4) to causal-interleaved MXFP4 "
        "V publisher with aligned, disjoint output buffers"
    );
    m.def(
        "quantize_nvfp4_projection_operand",
        &quantize_nvfp4_projection_operand,
        "Prepare one BF16 projection operand for the persistent NVFP4 GEMM"
    );
    m.def(
        "quantize_nvfp4_projection_operand_rmsnorm",
        &quantize_nvfp4_projection_operand_rmsnorm,
        "Fuse RMSNorm with exact-dynamic native NVFP4 activation preparation"
    );
    m.def(
        "rmsnorm_backward_bf16",
        &rmsnorm_backward_bf16,
        "Fuse the exact RMSNorm input and weight gradients for the native "
        "NVFP4 attention path"
    );
    m.def(
        "quantize_nvfp4_projection_weight",
        &quantize_nvfp4_projection_weight,
        "Prepare one BF16 projection weight with transpose-consistent "
        "16x16 NVFP4 scaling"
    );
    m.def(
        "quantize_nvfp4_projection_weight_dual",
        &quantize_nvfp4_projection_weight_dual,
        "Prepare one 16x16-scaled NVFP4 learned weight and its exact physical "
        "transpose from a single BF16 quantization"
    );
    m.def(
        "quantize_nvfp4_projection_weight_dual_out",
        &quantize_nvfp4_projection_weight_dual_out,
        "Checked caller-owned true-2D NVFP4 weight publication for both "
        "GEMM orientations in one tiled quantization pass"
    );
    m.def(
        "quantize_nvfp4_projection_weight_dual_out_unchecked",
        &quantize_nvfp4_projection_weight_dual_out_unchecked,
        "Shape-bound caller-owned true-2D NVFP4 dual-weight publication"
    );
    m.def(
        "quantize_gqa_d128_qkv_projection_weight_dual_out",
        &quantize_gqa_d128_qkv_projection_weight_dual_out,
        "Checked direct-canonical D128 GQA Q/K/V true-2D NVFP4 weight "
        "preparation publishing both GEMM orientations"
    );
    m.def(
        "quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked",
        &quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked,
        "Shape-bound direct-canonical D128 GQA Q/K/V dual-orientation "
        "NVFP4 weight preparation"
    );
    m.def(
        "quantize_nvfp4_projection_operand_scaled",
        &quantize_nvfp4_projection_operand_scaled,
        "Prepare one BF16 projection operand while folding a positive value "
        "scale into its NVFP4 decode scalar"
    );
    m.def(
        "quantize_nvfp4_projection_operand_precomputed_scale",
        &quantize_nvfp4_projection_operand_precomputed_scale,
        "Prepare one BF16 projection operand using a supplied NVFP4 global "
        "scale"
    );
    m.def(
        "quantize_nvfp4_projection_operand_precomputed_scale_inverse_rope",
        &quantize_nvfp4_projection_operand_precomputed_scale_inverse_rope,
        "Apply pair-native inverse RoPE while preparing a delayed-scale "
        "NVFP4 projection operand, optionally republishing inverse BF16"
    );
    m.def(
        "project_qk_adaptive_fp4_nvfp4",
        &project_qk_adaptive_fp4_nvfp4,
        "Persistent SM100 NVFP4 Q/K projection whose register epilogue "
        "publishes all four adaptive E2M1 backward operand layouts"
    );
    m.def(
        "project_nvfp4_generic",
        &project_nvfp4_generic,
        "Persistent SM100 NVFP4 GEMM for compact dQ projection operands"
    );
    m.def(
        "project_e4m3_generic",
        &project_e4m3_generic,
        "Persistent SM100 rowwise/channelwise E4M3 GEMM with BF16 output"
    );
    m.def(
        "project_gqa_d128_hierarchical_qkv_gradient_nvfp4",
        &project_gqa_d128_hierarchical_qkv_gradient_nvfp4,
        "Fold one/two D128 GQA dQ reduction lanes into a stacked NVFP4 "
        "QKV projection-backward operand without publishing standalone dQ"
    );
    m.def(
        "pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles",
        &pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles,
        "Pack a tile range of hierarchical D128 GQA gradients after its "
        "release-counted dQ reduction frontier"
    );
    m.def(
        "project_qk_adaptive_fp4_nvfp4_dispatch",
        &project_qk_adaptive_fp4_nvfp4_dispatch,
        "Shape-dispatched NVFP4 Q/K projection selecting fused or parallel "
        "adaptive E2M1 publication"
    );
    m.def(
        "project_qkv_unified_fp4_nvfp4",
        &project_qkv_unified_fp4_nvfp4,
        "Persistent SM100 NVFP4 QKV projection publishing shared forward / "
        "backward Q/K payloads and transposed MXFP4 V"
    );
    m.def(
        "project_qkv_unified_fp4_nvfp4_rope",
        &project_qkv_unified_fp4_nvfp4_rope,
        "Persistent SM100 NVFP4 QKV projection applying pair-native RoPE "
        "before publishing shared forward / backward low-precision operands"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4",
        &project_qkv_gqa_d128_unified_fp4_nvfp4,
        "Persistent SM100 D128 GQA NVFP4 QKV projection publishing native "
        "NVFP4-QK / MXFP4-V forward operands"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope,
        "Persistent SM100 D128 GQA NVFP4 QKV projection fusing pair-native "
        "RoPE and native forward/backward operand publication"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed,
        "Persistent SM100 D128 GQA NVFP4 QKV projection consuming one packed "
        "BF16 cosine/sine word per rotary pair"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed,
        "Persistent SM100 D64 GQA NVFP4 QKV projection pairing adjacent "
        "heads in D128 MMA tiles while publishing native D64 forward and "
        "backward operands with fused RoPE"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal,
        "Persistent SM100 D64 GQA NVFP4 QKV projection publishing K/V in "
        "stride-four causal groups for the interleaved forward kernel"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "fp8_forward_out",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_fp8_forward_out,
        "Checked experimental normal-order native NVFP4 QKV projection "
        "writing NVFP4-QK, FP8-V, and independent E4M3 backward operands "
        "to caller-owned storage"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "fp8_forward_out_unchecked",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_fp8_forward_out_unchecked,
        "Shape-bound experimental normal-order native NVFP4 "
        "caller-publication route without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_mx_forward_out",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_mx_forward_out,
        "Checked experimental causal-interleaved native NVFP4 QKV "
        "projection writing NVFP4-QK, MXFP4-V, and independent E4M3 "
        "backward operands to caller-owned storage"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_mx_forward_out_unchecked",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_mx_forward_out_unchecked,
        "Shape-bound experimental causal-interleaved native NVFP4 "
        "caller-publication route without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "represented_backward_perblock_qk",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_represented_backward_perblock_qk,
        "Experimental paired-D64 native NVFP4 projection publishing exact "
        "FP8-PV plus represented per-row-K16 E4M3 Q/K backward operands"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward,
        "Experimental paired-D64 native NVFP4 projection publishing "
        "interleaved MXFP4-PV, represented per-row-K16 E4M3 Q/K, and "
        "direct projection-accumulator E4M3 V backward operands"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "represented_backward_perblock_qk_fp8_forward_out",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_represented_backward_perblock_qk_fp8_forward_out,
        "Checked experimental native NVFP4 exact-FP8 route writing "
        "represented per-row-K16 Q/K backward operands to caller-owned "
        "storage"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "represented_backward_perblock_qk_fp8_forward_out_unchecked",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_represented_backward_perblock_qk_fp8_forward_out_unchecked,
        "Shape-bound native NVFP4 exact-FP8 represented per-block caller "
        "publication route without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward_mx_forward_out",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_mx_forward_out,
        "Checked experimental native NVFP4 MX route writing represented "
        "per-row-K16 Q/K and direct projection-accumulator E4M3 V backward "
        "operands to caller-owned storage"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward_mx_forward_out_unchecked",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_mx_forward_out_unchecked,
        "Shape-bound native NVFP4 MX split-V represented per-block caller "
        "publication route without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "output_shared_split_v_mx_forward_out",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_output_shared_split_v_mx_forward_out,
        "Checked opt-in native NVFP4 MX split-V route consuming the resident "
        "BF16 output ring directly for backward E4M3 and forward MXFP4"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "output_shared_split_v_mx_forward_out_unchecked",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_output_shared_split_v_mx_forward_out_unchecked,
        "Shape-bound opt-in output-shared native NVFP4 MX split-V caller "
        "publication route without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "e4m3_derived_mx_forward_out",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_e4m3_derived_mx_forward_out,
        "Checked opt-in native NVFP4 projection deriving causal MXFP4 V "
        "inline from the exact backward E4M3(x4) publication"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "e4m3_derived_mx_forward_out_unchecked",
        &project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_interleaved_causal_represented_backward_perblock_qk_e4m3_derived_mx_forward_out_unchecked,
        "Shape-bound opt-in native NVFP4 inline E4M3-to-MXFP4 caller "
        "publication route without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal,
        "Persistent SM100 D64 GQA dense-E4M3 QKV projection with rowwise "
        "activation and per-output-channel weight decode, fused RoPE, and "
        "direct NVFP4-QK / E4M3-V publication"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward,
        "Opt-in paired-D64 dense-E4M3 QKV projection publishing backward "
        "E4M3 Q/K/V from the exact NVFP4-QK and MXFP4-V codes represented "
        "by the causal forward route"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk,
        "Opt-in paired-D64 dense-E4M3 QKV projection using one E4M3 scale "
        "per Q/K row x K16 block and lifting the exact represented codes "
        "into backward E4M3"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward,
        "Experimental MX-only split publication: retain represented "
        "per-block NVFP4 Q/K while publishing backward E4M3 V directly "
        "from the projection accumulator"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward_vscale_out",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_vscale_out,
        "Production MX split-V projection publishing forward E8M0 scales "
        "into a caller-owned layer workspace"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "fp8_forward_out",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_fp8_forward_out,
        "Checked exact-FP8 route writing all projection publications into "
        "caller-owned storage and returning backward E4M3 Q/K/V aliases"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "fp8_forward_out_unchecked",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_fp8_forward_out_unchecked,
        "Shape-bound exact-FP8 caller-publication route without repeated "
        "tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward_mx_forward_out",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_mx_forward_out,
        "Checked MX split-V route writing all projection publications into "
        "caller-owned storage and returning backward E4M3 Q/K/V aliases"
    );
    m.def(
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward_mx_forward_out_unchecked",
        &project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_interleaved_causal_represented_backward_perblock_qk_split_v_backward_mx_forward_out_unchecked,
        "Shape-bound MX split-V caller-publication route without repeated "
        "tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered",
        &project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered,
        "Persistent SM100 D128 GQA dense-E4M3 QKV projection with fused "
        "RoPE, ordinary-order FP8/MX V, dynamic row-K16 NVFP4 Q/K, and "
        "direct projection-accumulator E4M3 backward Q/K/V"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_"
        "fp8_forward_out",
        &project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_fp8_forward_out,
        "Checked D128 dense-E4M3 FP8-PV projection writing caller-owned "
        "row-K16 NVFP4 Q/K and direct-accumulator backward E4M3 Q/K/V"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_"
        "fp8_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_fp8_forward_out_unchecked,
        "Shape-bound D128 dense-E4M3 FP8-PV caller-publication route "
        "without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_"
        "mx_forward_out",
        &project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_mx_forward_out,
        "Checked D128 dense-E4M3 MXFP4-PV projection writing ordinary-order "
        "caller-owned V and direct-accumulator backward E4M3 Q/K/V"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_"
        "mx_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered_mx_forward_out_unchecked,
        "Shape-bound D128 dense-E4M3 MXFP4-PV caller-publication route "
        "without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered,
        "Persistent SM100 D128 GQA NVFP4 QKV projection with an explicit "
        "resident-cluster cap for workload-balance experiments"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "fp8_forward_out",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_fp8_forward_out,
        "Checked D128 native NVFP4 projection writing only active FP8-V "
        "forward data plus shared E4M3 backward operands into caller storage"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "fp8_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_fp8_forward_out_unchecked,
        "Shape-bound D128 exact-FP8 caller-publication route without repeated "
        "tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "represented_backward_perblock_qk_fp8_forward_out",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_represented_backward_perblock_qk_fp8_forward_out,
        "Checked opt-in D128 native NVFP4 FP8-PV projection publishing "
        "backward E4M3 Q/K from the exact represented per-row-K16 forward "
        "codes"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "represented_backward_perblock_qk_fp8_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_represented_backward_perblock_qk_fp8_forward_out_unchecked,
        "Shape-bound opt-in D128 represented-Q/K FP8-PV caller-publication "
        "route without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "mx_forward_out",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_mx_forward_out,
        "Checked D128 native NVFP4 projection writing only active MXFP4-V "
        "forward data plus shared E4M3 backward operands into caller storage"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "mx_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_mx_forward_out_unchecked,
        "Shape-bound D128 MX caller-publication route without repeated tensor "
        "contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "output_shared_dual_v_mx_forward_out",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_output_shared_dual_v_mx_forward_out,
        "Checked D128 native NVFP4 MX route publishing ordinary-order "
        "MXFP4 forward V and exact E4M3 backward V directly from the "
        "resident BF16 output ring"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "output_shared_dual_v_mx_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_output_shared_dual_v_mx_forward_out_unchecked,
        "Shape-bound D128 output-shared dual-V MX caller-publication route "
        "without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "mx_backward_v_mx_forward_out",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_mx_backward_v_mx_forward_out,
        "Checked D128 native NVFP4 route publishing rowwise width-six "
        "MXFP4 V for forward and dP while retaining E4M3 Q/K backward"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "mx_backward_v_mx_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_mx_backward_v_mx_forward_out_unchecked,
        "Shape-bound D128 MX-only-backward-V caller-publication route "
        "without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "direct_common_rowscale_mx_backward_v_mx_forward_out",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_direct_common_rowscale_mx_backward_v_mx_forward_out,
        "Checked experimental D128 native NVFP4 route publishing backward "
        "MXFP4 V directly under one repeated E8M0 code per D128 row"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "direct_common_rowscale_mx_backward_v_mx_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_direct_common_rowscale_mx_backward_v_mx_forward_out_unchecked,
        "Shape-bound experimental direct-common-row MX backward-V route "
        "without repeated tensor contracts"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "shared_tile_mx_backward_v_mx_forward_out",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_shared_tile_mx_backward_v_mx_forward_out,
        "Checked experimental D128 native NVFP4 route quantizing each "
        "D32xS32 V tile once and publishing both physical orientations"
    );
    m.def(
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "shared_tile_mx_backward_v_mx_forward_out_unchecked",
        &project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_shared_tile_mx_backward_v_mx_forward_out_unchecked,
        "Shape-bound experimental shared-tile MX backward-V route without "
        "repeated tensor contracts"
    );
    m.def(
        "inverse_rope_interleaved_qkv_grad_inplace",
        &inverse_rope_interleaved_qkv_grad_inplace,
        "Apply inverse pair-native RoPE in-place to interleaved BF16 QKV "
        "projection gradients"
    );
    m.def(
        "stitch_gqa_d64_inverse_rope_grad",
        &stitch_gqa_d64_inverse_rope_grad,
        "Fuse inverse RoPE, dV decode, loss-scale conversion, and D64 GQA "
        "Q/K/V projection-gradient layout publication"
    );
    m.def(
        "stitch_gqa_d128_inverse_rope_grad",
        &stitch_gqa_d128_inverse_rope_grad,
        "Fuse inverse RoPE, Q/K/V decode, loss-scale conversion, and D128 "
        "GQA projection-gradient layout publication"
    );
    m.def(
        "rope_pair_qk_inplace",
        &rope_pair_qk_inplace,
        "Apply pair-native RoPE in-place to contiguous BF16 Q/K tensors"
    );
    m.def(
        "project_dout_unified_fp4_nvfp4",
        &project_dout_unified_fp4_nvfp4,
        "Persistent SM100 NVFP4 output-gradient projection publishing both "
        "MXFP4 dO layouts, exact scales, dPsum, and log2 LSE"
    );
    m.def(
        "project_dout_unified_fp4_nvfp4_v509_e5m2",
        &project_dout_unified_fp4_nvfp4_v509_e5m2,
        "Shape-checked B1/B2/B4 S4096/H32/D128 v509 NVFP4 output-gradient "
        "projection publishing genuine E5M2 dO and matched statistics"
    );
    m.def(
        "project_dout_unified_fp4_nvfp4_v509_e5m2_metadata",
        &project_dout_unified_fp4_nvfp4_v509_e5m2_metadata,
        "Exact fused v509 E5M2 dO publisher metadata"
    );
    m.def(
        "project_bf16_dq_persistent",
        &project_bf16_dq_persistent,
        "Persistent SM100 BF16 dQ projection reduction diagnostic"
    );
    m.def(
        "prepack_mixed_v",
        &prepack_mixed_v,
        "Prepack V as reusable mixed E4M3/fixed-scale-E2M1 rows"
    );
    m.def(
        "prepare_mxfp4_backward_operands",
        &prepare_mxfp4_backward_operands,
        "Publish the exact producer-native MXFP4 dO/V/statistics bundle"
    );
    m.def(
        "backward_fp4_native",
        &backward_fp4_native,
        "V382 schedule with FP8 dS and native E2M1 Q/K F8F6F4 MMAs"
    );
    m.def(
        "backward_fp4_fp8pv_native",
        &backward_fp4_fp8pv_native,
        "Experimental native FP4 Q/K plus dense-E4M3 P/dO dV"
    );
    m.def(
        "backward_fp4_fp8pv_x32_native",
        &backward_fp4_fp8pv_x32_native,
        "Native-x32 FP4 Q/K plus dense-E4M3 P/dO dV"
    );
    m.def(
        "backward_fp4_fp8pv_x32_reuse_p_native",
        &backward_fp4_fp8pv_x32_reuse_p_native,
        "Native-x32 FP8 P reuse for dV and approximate packed-half2 dS"
    );
    m.def(
        "backward_fp4_fp8pv_x32_reuse_p_split_dk_native",
        &backward_fp4_fp8pv_x32_reuse_p_split_dk_native,
        "Native-x32 FP8 P reuse with canonical split dS shared by dK and dQ publication"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_native",
        &backward_fp4_fp8dpdv_x32_split_dk_native,
        "Forward-style fused E4M3 dP/dV operands with native-x32 split dK"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_native,
        "Retained FP4+FP8 route consuming device-resident adaptive Q/K "
        "scales"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native,
        "Adaptive FP4+FP8 route returning the completed BF16 dQ reduction "
        "directly for an adjacent projection-backward consumer"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_native,
        "Adaptive FP4+FP8 route publishing completed BF16 dQ/dK/dV "
        "directly in projection-ready per-head QKV order"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_"
        "prepacked_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_native,
        "Adaptive FP4+FP8 route consuming projection-native E4M3 dO/V "
        "and precomputed softmax statistics"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_"
        "prepacked_v_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_v_native,
        "Adaptive FP4+FP8 route consuming projection-native E4M3 V while "
        "producing dO and statistics in the retained preprocessing pass"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_"
        "prepacked_dout_v_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_dout_v_native,
        "Adaptive FP4+FP8 route consuming projection-native E4M3 dO/V "
        "while producing only softmax statistics in preprocessing"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_hierarchical_"
        "qkv_projection_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_hierarchical_qkv_projection_native,
        "Adaptive FP4+FP8 route folding two private dQ reduction lanes, "
        "inverse RoPE, and stacked QKV NVFP4 publication into projection "
        "backward"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_stacked_"
        "qkv_projection_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_stacked_qkv_projection_native,
        "Adaptive FP4+FP8 route publishing normal one-lane BF16 dQ and "
        "forming a stacked inverse-RoPE NVFP4 QKV projection operand"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_direct_dq_"
        "projection_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_direct_dq_projection_native,
        "Adaptive FP4+FP8 route feeding head-complete BF16 dQ slices into "
        "the projection reduction loop"
    );
    m.def(
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_nvfp4_dq_"
        "projection_native",
        &backward_fp4_fp8dpdv_x32_split_dk_adaptive_nvfp4_dq_projection_native,
        "Adaptive FP4+FP8 route handing completed dQ to an NVFP4 projection "
        "without publishing standalone dQ"
    );
    m.def(
        "backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native",
        &backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native,
        "Three-command mixed E4M3/MXFP4 dP with retained E4M3 dV"
    );
    m.def(
        "backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_"
        "prepacked_v_native",
        &backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_prepacked_v_native,
        "Mixed dP route consuming a forward-reusable prepacked V operand"
    );
    m.def(
        "backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_adaptive_"
        "prepacked_v_native",
        &backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_adaptive_prepacked_v_native,
        "Lifetime-safe adaptive Q/K mixed dP route with a dedicated fixed "
        "MXFP4 scale page"
    );
    m.def(
        "backward_fp4_fp8dp_mxfp4dv_x32_split_dk_native",
        &backward_fp4_fp8dp_mxfp4dv_x32_split_dk_native,
        "Forward-style E4M3 dP plus dynamic MXFP4 P/dO dV"
    );
    m.def(
        "backward_fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native",
        &backward_fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native,
        "Dynamic MXFP4 dV with log-domain scale selection and fused packing"
    );
    m.def(
        "backward_fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native",
        &backward_fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native,
        "Forward-log MXFP4 dV with score-Q readiness split from dO"
    );
    m.def(
        "backward_fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk_native",
        &backward_fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk_native,
        "Forward-log MXFP4 dP/dV with score-Q readiness split from dO"
    );
    m.def(
        "backward_fp4_mxfp4dpdvdsdqdk_forward_log_split_q_x32_native",
        &backward_fp4_mxfp4dpdvdsdqdk_forward_log_split_q_x32_native,
        "Pure block-scaled MXFP4 dP/dV/dS/dQ/dK with compact Q/K"
    );
    m.def(
        "backward_fp4_mxfp4dpdvdsdqdk_producer_native_x32",
        &backward_fp4_mxfp4dpdvdsdqdk_producer_native_x32,
        "Pure MXFP4 backward consuming producer-published dO/V operands "
        "and statistics without local requantization"
    );
    m.def(
        "backward_fp4_rank128_mxfp4dpdvdsdqdk_producer_native_x32",
        &backward_fp4_rank128_mxfp4dpdvdsdqdk_producer_native_x32,
        "Pure MXFP4 backward for a model-native rank-128 Q/K subspace"
    );
    m.def(
        "backward_fp4_fused_quant",
        &backward_fp4_fused_quant,
        "Single-call fused Q/K E2M1 packing plus native FP4 backward"
    );
}

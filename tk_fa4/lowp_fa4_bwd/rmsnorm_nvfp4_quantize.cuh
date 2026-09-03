#pragma once

#include "../../ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_quantize.cuh"

namespace rmsnorm_nvfp4_quantize {

using namespace kittens;

__device__ __forceinline__ float block_sum_256(float value) {
    constexpr int kWarps = 8;
    __shared__ float warp_values[kWarps];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    value = (warp == 0 && lane < kWarps) ? warp_values[lane] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;
}

__device__ __forceinline__ float block_max_256(float value) {
    constexpr int kWarps = 8;
    __shared__ float warp_values[kWarps];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffffu, value, offset)
        );
    }
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    value = (warp == 0 && lane < kWarps) ? warp_values[lane] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(
                value,
                __shfl_down_sync(0xffffffffu, value, offset)
            );
        }
    }
    return value;
}

// One CTA owns one complete row. The raw BF16 row remains in shared memory
// across the reduction so the normalized BF16 publication and its dynamic
// amax do not reread the residual activation from global memory.
__global__ __launch_bounds__(256) void rmsnorm_bf16_amax_kernel(
    const bf16 *__restrict__ input,
    const bf16 *__restrict__ gamma,
    bf16 *__restrict__ normalized,
    float *__restrict__ inv_rms,
    float *__restrict__ global_amax,
    float epsilon,
    int rows,
    int columns
) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    extern __shared__ bf16 input_cache[];
    const int64_t row_offset = static_cast<int64_t>(row) * columns;
    float sum_squares = 0.0f;
    for (int column = static_cast<int>(threadIdx.x); column < columns;
         column += static_cast<int>(blockDim.x)) {
        const bf16 value_bf16 = input[row_offset + column];
        input_cache[column] = value_bf16;
        const float value = __bfloat162float(value_bf16);
        sum_squares += __fmul_rn(value, value);
    }

    const float row_sum_squares = block_sum_256(sum_squares);
    __shared__ float row_inv_rms;
    if (threadIdx.x == 0) {
        row_inv_rms = rsqrtf(
            row_sum_squares / static_cast<float>(columns) + epsilon
        );
        inv_rms[row] = row_inv_rms;
    }
    __syncthreads();

    float local_amax = 0.0f;
    for (int column = static_cast<int>(threadIdx.x); column < columns;
         column += static_cast<int>(blockDim.x)) {
        const float value = __bfloat162float(input_cache[column]);
        const float weight = __bfloat162float(gamma[column]);
        // Preserve the two ordered FP32 multiplies in the eager RMSNorm.
        const float unit = __fmul_rn(value, row_inv_rms);
        const float scaled = __fmul_rn(unit, weight);
        const bf16 output = __float2bfloat16_rn(scaled);
        normalized[row_offset + column] = output;
        local_amax = fmaxf(
            local_amax,
            fabsf(__bfloat162float(output))
        );
    }

    const float row_amax = block_max_256(local_amax);
    if (threadIdx.x == 0) {
        atomicMax(
            reinterpret_cast<unsigned int *>(global_amax),
            __float_as_uint(row_amax)
        );
    }
}

__device__ __forceinline__ void sanitize_e4m3_scale(fp8e4m3 &value) {
    if ((value.__x & 0x7fu) == 0x7fu) {
        value.__x = static_cast<uint8_t>((value.__x & 0x80u) | 0x7eu);
    }
}

// This is the activation-scaling specialization of TK's native NVFP4 packer.
// Its only semantic change is that the matrix-wide amax is supplied separately
// and converted to the public decode scalar inside this second launch.
__global__ __launch_bounds__(128) void quantize_from_amax_kernel(
    const __grid_constant__ nvfp4_quantize::globals g,
    const float *__restrict__ global_amax
) {
    extern __shared__ int shared_storage[];
    tma_swizzle_allocator allocator(&shared_storage[0]);
    nvfp4_quantize::globals::A_bf16_tile &input_smem =
        allocator.allocate<nvfp4_quantize::globals::A_bf16_tile>();
    auto &packed_smem = *reinterpret_cast<
        nvfp4_quantize::globals::A_fp4x2_tile *>(&input_smem);
    auto (&scale_smem)[2] = *reinterpret_cast<
        nvfp4_quantize::globals::A_sc_vec(*)[2]>(
            reinterpret_cast<uint64_t>(&packed_smem) + sizeof(packed_smem)
        );

    const int tid = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int column_tile = static_cast<int>(blockIdx.x);

    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, input_smem);
        tma::load_async(
            input_smem,
            g.A_bf16,
            {row, column_tile},
            inputs_arrived
        );
    }

    const float global_decode = global_amax[0] / 2688.0f;
    const float global_encode = 1.0f / fmaxf(global_decode, 1.0e-12f);
    if (tid == 0 && row == 0 && column_tile == 0) {
        g.A_sc_global.raw_ptr[0] = global_decode;
    }

    constexpr int kHalfKBlocks =
        nvfp4_quantize::globals::TILE_N /
        nvfp4_quantize::globals::K_BLOCK_SIZE / 2;
    constexpr int kPairsPerKBlock =
        nvfp4_quantize::globals::K_BLOCK_SIZE / 2;
    bf16_2 input_registers[2][kHalfKBlocks][kPairsPerKBlock];
    fp8e4m3 scale_registers[2][kHalfKBlocks];

    __syncthreads();
    wait(inputs_arrived, 0);

    #pragma unroll
    for (int column_half = 0; column_half < 2; ++column_half) {
        #pragma unroll
        for (int index = 0; index < kHalfKBlocks; ++index) {
            const int k_block =
                (index + tid / 8) % kHalfKBlocks +
                column_half * kHalfKBlocks;
            #pragma unroll
            for (int pair = 0; pair < kPairsPerKBlock; ++pair) {
                const int tile_column =
                    k_block * nvfp4_quantize::globals::K_BLOCK_SIZE +
                    ((tid + pair) * 2) %
                        nvfp4_quantize::globals::K_BLOCK_SIZE;
                const int offset =
                    (tid * nvfp4_quantize::globals::TILE_N + tile_column) *
                    sizeof(bf16);
                move<bf16_2>::lds(
                    input_registers[column_half][index][pair],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_smem)) +
                        offset
                );
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int column_half = 0; column_half < 2; ++column_half) {
        float local_amax[kHalfKBlocks];
        #pragma unroll
        for (int index = 0; index < kHalfKBlocks; ++index) {
            const int k_block = (index + tid / 8) % kHalfKBlocks;
            bf16_2 pair_amax = __habs2(
                input_registers[column_half][index][0]
            );
            #pragma unroll
            for (int pair = 1; pair < kPairsPerKBlock; ++pair) {
                pair_amax = __hmax2(
                    pair_amax,
                    __habs2(input_registers[column_half][index][pair])
                );
            }
            local_amax[k_block] = __bfloat162float(
                __hmax(pair_amax.x, pair_amax.y)
            );
        }

        #pragma unroll
        for (int index = 0; index < kHalfKBlocks; ++index) {
            scale_registers[column_half][index] = __nv_fp8_e4m3(
                local_amax[index] / 6.0f * global_encode
            );
        }

        #pragma unroll
        for (int index = 0; index < kHalfKBlocks; ++index) {
            const int k_block = (index + tid / 8) % kHalfKBlocks;
            const float local_decode = static_cast<float>(
                scale_registers[column_half][k_block]
            );
            const float encode = 1.0f / fmaxf(
                local_decode * global_decode,
                1.0e-12f
            );
            const int output_base =
                tid * nvfp4_quantize::globals::TILE_N / 2 +
                (k_block + column_half * kHalfKBlocks) *
                    nvfp4_quantize::globals::K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int pair = 0; pair < kPairsPerKBlock; ++pair) {
                const int output_offset = output_base + ((tid + pair) & 7);
                const float2 scaled = {
                    __bfloat162float(
                        input_registers[column_half][index][pair].x
                    ) * encode,
                    __bfloat162float(
                        input_registers[column_half][index][pair].y
                    ) * encode,
                };
                asm volatile(
                    "{st.shared.b8 [%0], %1;}"
                    :: "r"(
                        static_cast<uint32_t>(
                            __cvta_generic_to_shared(&packed_smem)
                        ) + output_offset
                    ),
                    "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(
                        scaled,
                        __NV_E2M1,
                        cudaRoundNearest
                    )))
                );
            }
        }

        // Match the existing post-pack FP8 NaN fixup without a third kernel.
        #pragma unroll
        for (int index = 0; index < kHalfKBlocks; ++index) {
            sanitize_e4m3_scale(scale_registers[column_half][index]);
        }
    }

    const int scale_offset = (tid % 32) * 16 + (tid / 32) * 4;
    asm volatile(
        "{st.shared.b32 [%0], %1;}"
        :: "r"(
            static_cast<uint32_t>(
                __cvta_generic_to_shared(&scale_smem[0])
            ) + scale_offset
        ),
        "r"(*reinterpret_cast<uint32_t *>(&scale_registers[0][0]))
    );
    asm volatile(
        "{st.shared.b32 [%0], %1;}"
        :: "r"(
            static_cast<uint32_t>(
                __cvta_generic_to_shared(&scale_smem[1])
            ) + scale_offset
        ),
        "r"(*reinterpret_cast<uint32_t *>(&scale_registers[1][0]))
    );

    __syncthreads();
    if (tid == 0) {
        tma::store_async(g.A_fp4x2, packed_smem, {row, column_tile});
        tma::store_async(g.A_sc, scale_smem[0], {row, column_tile * 2, 0});
        tma::store_async(g.A_sc, scale_smem[1], {row, column_tile * 2 + 1, 0});
    }
}

// The saturated D64 path has exactly 2,048 hidden columns.  Keeping eight
// columns per thread in registers lets one CTA compute the row reduction,
// publish BF16 dx, and accumulate sixteen rows of dgamma without materializing
// any full-sized FP32 intermediates.  The per-CTA dgamma slab is reduced by a
// second, small kernel rather than issuing millions of contended atomics.
constexpr int RMSNORM_BACKWARD_COLUMNS = 2048;
constexpr int RMSNORM_BACKWARD_THREADS = 256;
constexpr int RMSNORM_BACKWARD_VALUES_PER_THREAD =
    RMSNORM_BACKWARD_COLUMNS / RMSNORM_BACKWARD_THREADS;
constexpr int RMSNORM_BACKWARD_ROWS_PER_BLOCK = 16;

__global__ __launch_bounds__(RMSNORM_BACKWARD_THREADS)
void rmsnorm_backward_partial_kernel(
    const bf16 *__restrict__ input,
    const bf16 *__restrict__ gamma,
    const float *__restrict__ inv_rms,
    const bf16 *__restrict__ gradient,
    bf16 *__restrict__ input_gradient,
    float *__restrict__ gamma_gradient_partials,
    int rows
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int row_begin =
        static_cast<int>(blockIdx.x) * RMSNORM_BACKWARD_ROWS_PER_BLOCK;
    float gamma_values[RMSNORM_BACKWARD_VALUES_PER_THREAD];
    float gamma_gradient[RMSNORM_BACKWARD_VALUES_PER_THREAD] = {};

    #pragma unroll
    for (int index = 0; index < RMSNORM_BACKWARD_VALUES_PER_THREAD; ++index) {
        const int column = tid + index * RMSNORM_BACKWARD_THREADS;
        gamma_values[index] = __bfloat162float(gamma[column]);
    }

    #pragma unroll
    for (int row_index = 0; row_index < RMSNORM_BACKWARD_ROWS_PER_BLOCK;
         ++row_index) {
        const int row = row_begin + row_index;
        if (row >= rows) {
            break;
        }
        const int64_t row_offset =
            static_cast<int64_t>(row) * RMSNORM_BACKWARD_COLUMNS;
        float input_values[RMSNORM_BACKWARD_VALUES_PER_THREAD];
        float gradient_values[RMSNORM_BACKWARD_VALUES_PER_THREAD];
        float local_projection = 0.0f;

        #pragma unroll
        for (int index = 0; index < RMSNORM_BACKWARD_VALUES_PER_THREAD;
             ++index) {
            const int column = tid + index * RMSNORM_BACKWARD_THREADS;
            const float input_value = __bfloat162float(
                input[row_offset + column]
            );
            const float gradient_value = __bfloat162float(
                gradient[row_offset + column]
            );
            input_values[index] = input_value;
            gradient_values[index] = gradient_value;
            local_projection +=
                gradient_value * gamma_values[index] * input_value;
        }

        const float projection_sum = block_sum_256(local_projection);
        __shared__ float row_projection;
        if (tid == 0) {
            row_projection =
                projection_sum /
                static_cast<float>(RMSNORM_BACKWARD_COLUMNS);
        }
        __syncthreads();

        const float inverse = inv_rms[row];
        const float correction = inverse * inverse * row_projection;
        #pragma unroll
        for (int index = 0; index < RMSNORM_BACKWARD_VALUES_PER_THREAD;
             ++index) {
            const int column = tid + index * RMSNORM_BACKWARD_THREADS;
            const float input_value = input_values[index];
            const float gradient_value = gradient_values[index];
            const float weighted_gradient =
                gradient_value * gamma_values[index];
            input_gradient[row_offset + column] = __float2bfloat16_rn(
                inverse * (
                    weighted_gradient - input_value * correction
                )
            );
            gamma_gradient[index] +=
                gradient_value * (input_value * inverse);
        }
    }

    const int64_t partial_offset =
        static_cast<int64_t>(blockIdx.x) * RMSNORM_BACKWARD_COLUMNS;
    #pragma unroll
    for (int index = 0; index < RMSNORM_BACKWARD_VALUES_PER_THREAD; ++index) {
        const int column = tid + index * RMSNORM_BACKWARD_THREADS;
        gamma_gradient_partials[partial_offset + column] =
            gamma_gradient[index];
    }
}

__global__ __launch_bounds__(RMSNORM_BACKWARD_THREADS)
void rmsnorm_backward_gamma_finalize_kernel(
    const float *__restrict__ gamma_gradient_partials,
    bf16 *__restrict__ gamma_gradient,
    int partial_rows
) {
    const int column =
        static_cast<int>(blockIdx.x) * RMSNORM_BACKWARD_THREADS +
        static_cast<int>(threadIdx.x);
    if (column >= RMSNORM_BACKWARD_COLUMNS) {
        return;
    }

    float accumulators[4] = {};
    int partial_row = 0;
    for (; partial_row + 3 < partial_rows; partial_row += 4) {
        accumulators[0] += gamma_gradient_partials[
            static_cast<int64_t>(partial_row) * RMSNORM_BACKWARD_COLUMNS +
            column
        ];
        accumulators[1] += gamma_gradient_partials[
            static_cast<int64_t>(partial_row + 1) *
                RMSNORM_BACKWARD_COLUMNS + column
        ];
        accumulators[2] += gamma_gradient_partials[
            static_cast<int64_t>(partial_row + 2) *
                RMSNORM_BACKWARD_COLUMNS + column
        ];
        accumulators[3] += gamma_gradient_partials[
            static_cast<int64_t>(partial_row + 3) *
                RMSNORM_BACKWARD_COLUMNS + column
        ];
    }
    for (; partial_row < partial_rows; ++partial_row) {
        accumulators[0] += gamma_gradient_partials[
            static_cast<int64_t>(partial_row) * RMSNORM_BACKWARD_COLUMNS +
            column
        ];
    }
    const float total =
        (accumulators[0] + accumulators[1]) +
        (accumulators[2] + accumulators[3]);
    gamma_gradient[column] = __float2bfloat16_rn(total);
}

}  // namespace rmsnorm_nvfp4_quantize

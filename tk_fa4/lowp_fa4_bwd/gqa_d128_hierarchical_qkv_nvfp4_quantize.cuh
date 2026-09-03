#pragma once

// D128 GQA projection-backward operand producer.
//
// The CuTe attention kernel leaves dQ as one or two head-major BF16 reduction
// lanes.  This producer folds the lanes in registers, applies pair-native
// inverse RoPE to dQ/dK, and emits one delayed-scale NVFP4
// [all dQ | all dK | all dV] operand.  The one-lane row-major specialization
// is retained as a materialized-dQ control with identical packing arithmetic.

#include "kittens.cuh"

namespace tkfa4_gqa_d128_hierarchical_qkv_nvfp4 {

using namespace kittens;

struct config {
    static constexpr int NUM_WARPS = 8;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int K_BLOCK_SIZE = 16;
    static constexpr int QK_DEPTH = 128;
    static constexpr int V_DEPTH = 128;
    static constexpr int ROTARY_PAIRS = QK_DEPTH / 2;

    using bf16_tile = st_bf<TILE_M, TILE_N, false>;
    using fp4_tile = st_fp4e2m1_2<TILE_M, TILE_N / 2, false>;
    using scale_vec = sv_hf<256>;
    using bf16_gl = gl<bf16, 1, 1, -1, -1, bf16_tile>;
    using fp4_gl = gl<fp4e2m1_2, 1, 1, -1, -1, fp4_tile>;
    using scale_gl = gl<half, 1, -1, -1, 256, scale_vec>;
    using global_scale_gl = gl<float, 1, 1, 1, 1>;

    // Row-major dQ is [rows, q_heads * 128].  Head-major hierarchical dQ is
    // viewed as [lanes * q_heads * rows, 128], ordered lane/head/row/depth.
    bf16_gl dQ;
    bf16_gl dK;
    bf16_gl dV;
    fp4_gl A;
    scale_gl A_sc;
    global_scale_gl A_scale;
    const uint32_t *rope_packed;
    int rows;
    int q_heads;
    int kv_heads;
    int dq_reduction_lanes;
    bool dq_head_major;
    // Decode gains are folded into the published E4M3 block scales after
    // payload quantization.  Power-of-two loss-scale corrections therefore
    // add no elementwise conversion or packing work.
    float dq_decode_scale = 1.0f;
    float dk_decode_scale = 1.0f;
    float dv_decode_scale = 1.0f;
    int row_tile_begin = 0;
    int row_tile_end = 0;
    int col_tile_begin = 0;
    int col_tile_end = 0;

    __host__ inline dim3 grid() const {
        const int total_rows = rows / TILE_M;
        const int total_cols = q_heads + 2 * kv_heads;
        const int row_begin = max(0, min(row_tile_begin, total_rows));
        const int row_end = row_tile_end > row_begin
            ? min(row_tile_end, total_rows)
            : total_rows;
        const int col_begin = max(0, min(col_tile_begin, total_cols));
        const int col_end = col_tile_end > col_begin
            ? min(col_tile_end, total_cols)
            : total_cols;
        return dim3(col_end - col_begin, row_end - row_begin);
    }

    __host__ inline dim3 block() const {
        return dim3(config::NUM_THREADS);
    }

    __host__ inline int dynamic_shared_memory() const {
        const int lane_tiles = dq_reduction_lanes == 2 ? 2 : 1;
        return lane_tiles * TILE_M * TILE_N * sizeof(bf16) + 1024;
    }

    __host__ inline int materialized_dynamic_shared_memory() const {
        return TILE_M * TILE_N * sizeof(bf16) +
            TILE_M * TILE_N / 2 + 2048;
    }
};

__device__ __forceinline__ bf16_2 inverse_rope_pair(
    bf16_2 pair,
    uint32_t packed_rope
) {
    const bf16 *rope = reinterpret_cast<const bf16 *>(&packed_rope);
    const float2 values = __bfloat1622float2(pair);
    const float cosine = __bfloat162float(rope[0]);
    const float sine = __bfloat162float(rope[1]);
    return __floats2bfloat162_rn(
        fmaf(values.y, sine, values.x * cosine),
        fmaf(-values.x, sine, values.y * cosine)
    );
}

template <bool HIERARCHICAL>
__global__ __launch_bounds__(config::NUM_THREADS, 1) void kernel(
    const __grid_constant__ globals g
) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(reinterpret_cast<int *>(&__shm[0]));
    globals::bf16_tile &input = allocator.allocate<globals::bf16_tile>();
    globals::bf16_tile *lane_one_input = nullptr;
    if constexpr (HIERARCHICAL) {
        lane_one_input = &allocator.allocate<globals::bf16_tile>();
    }
    globals::fp4_tile &packed =
        *reinterpret_cast<globals::fp4_tile *>(&input);
    globals::scale_vec (&scales)[2] =
        *reinterpret_cast<globals::scale_vec(*)[2]>(
            reinterpret_cast<uint64_t>(&packed) + sizeof(packed)
        );

    const int tid = static_cast<int>(threadIdx.x);
    const int tile_row = tid % globals::TILE_M;
    const int col_half = tid / globals::TILE_M;
    const int row_tile =
        g.row_tile_begin + static_cast<int>(blockIdx.y);
    const int col_tile =
        g.col_tile_begin + static_cast<int>(blockIdx.x);
    const bool is_dq = col_tile < g.q_heads;
    const bool is_dk =
        col_tile >= g.q_heads && col_tile < g.q_heads + g.kv_heads;
    const float field_decode_scale = is_dq
        ? g.dq_decode_scale
        : is_dk ? g.dk_decode_scale : g.dv_decode_scale;
    const int source_head = is_dq
        ? col_tile
        : is_dk
            ? col_tile - g.q_heads
            : col_tile - g.q_heads - g.kv_heads;
    const int q_tiles = g.rows / globals::TILE_M;

    if (tid == 0) {
        g.dQ.template prefetch_tma<globals::bf16_tile>();
        g.dK.template prefetch_tma<globals::bf16_tile>();
        g.dV.template prefetch_tma<globals::bf16_tile>();
    }

    __shared__ semaphore input_arrived[HIERARCHICAL ? 2 : 1];
    if (tid == 0) {
        init_semaphore(input_arrived[0], 0, 1);
        if constexpr (HIERARCHICAL) {
            init_semaphore(input_arrived[1], 0, 1);
        }
        tma::expect(input_arrived[0], input);
        if (is_dq) {
            if (g.dq_head_major) {
                tma::load_async(
                    input,
                    g.dQ,
                    {source_head * q_tiles + row_tile, 0},
                    input_arrived[0]
                );
            } else {
                tma::load_async(
                    input,
                    g.dQ,
                    {row_tile, source_head},
                    input_arrived[0]
                );
            }
        } else if (is_dk) {
            tma::load_async(
                input,
                g.dK,
                {row_tile, source_head},
                input_arrived[0]
            );
        } else {
            tma::load_async(
                input,
                g.dV,
                {row_tile, source_head},
                input_arrived[0]
            );
        }
        if constexpr (HIERARCHICAL) {
            if (is_dq) {
                tma::expect(input_arrived[1], *lane_one_input);
                tma::load_async(
                    *lane_one_input,
                    g.dQ,
                    {
                        g.q_heads * q_tiles +
                            source_head * q_tiles + row_tile,
                        0
                    },
                    input_arrived[1]
                );
            }
        }
    }

    const float global_decode = g.A_scale[{0}];
    const float global_encode =
        1.0f / fmaxf(global_decode, 0.000000000001f);
    constexpr int K_BLOCKS =
        globals::TILE_N / globals::K_BLOCK_SIZE / 2;
    constexpr int VALUES_PER_BLOCK = globals::K_BLOCK_SIZE / 2;
    bf16_2 values[K_BLOCKS][VALUES_PER_BLOCK];
    fp8e4m3 local_scales[K_BLOCKS];

    __syncthreads();
    wait(input_arrived[0], 0);
    if constexpr (HIERARCHICAL) {
        if (is_dq) {
            wait(input_arrived[1], 0);
        }
    }
    #pragma unroll
    for (int i = 0; i < K_BLOCKS; ++i) {
        const int block = (i + tile_row / 8) % K_BLOCKS +
            col_half * K_BLOCKS;
        #pragma unroll
        for (int j = 0; j < VALUES_PER_BLOCK; ++j) {
            const int col = block * globals::K_BLOCK_SIZE +
                ((tid + j) * 2) % globals::K_BLOCK_SIZE;
            const int offset =
                (tile_row * globals::TILE_N + col) * sizeof(bf16);
            move<bf16_2>::lds(
                values[i][j],
                static_cast<uint32_t>(__cvta_generic_to_shared(&input)) +
                    offset
            );
        }
    }

    if constexpr (HIERARCHICAL) {
        if (is_dq) {
            #pragma unroll
            for (int i = 0; i < K_BLOCKS; ++i) {
                const int block = (i + tile_row / 8) % K_BLOCKS +
                    col_half * K_BLOCKS;
                #pragma unroll
                for (int j = 0; j < VALUES_PER_BLOCK; ++j) {
                    const int col = block * globals::K_BLOCK_SIZE +
                        ((tid + j) * 2) % globals::K_BLOCK_SIZE;
                    const int offset =
                        (tile_row * globals::TILE_N + col) * sizeof(bf16);
                    bf16_2 lane_one;
                    move<bf16_2>::lds(
                        lane_one,
                        static_cast<uint32_t>(
                            __cvta_generic_to_shared(lane_one_input)
                        ) + offset
                    );
                    values[i][j] = __hadd2(values[i][j], lane_one);
                }
            }
        }
    }

    const int global_row = row_tile * globals::TILE_M + tile_row;
    if (is_dq || is_dk) {
        #pragma unroll
        for (int i = 0; i < K_BLOCKS; ++i) {
            const int block = (i + tile_row / 8) % K_BLOCKS +
                col_half * K_BLOCKS;
            #pragma unroll
            for (int j = 0; j < VALUES_PER_BLOCK; ++j) {
                const int tile_col = block * globals::K_BLOCK_SIZE +
                    ((tid + j) * 2) % globals::K_BLOCK_SIZE;
                const size_t rope_offset =
                    static_cast<size_t>(global_row) *
                        globals::ROTARY_PAIRS + tile_col / 2;
                values[i][j] = inverse_rope_pair(
                    values[i][j],
                    g.rope_packed[rope_offset]
                );
            }
        }
    }

    __syncthreads();
    float amax[K_BLOCKS];
    #pragma unroll
    for (int i = 0; i < K_BLOCKS; ++i) {
        const int block = (i + tile_row / 8) % K_BLOCKS;
        bf16_2 block_max = __habs2(values[i][0]);
        #pragma unroll
        for (int j = 1; j < VALUES_PER_BLOCK; ++j) {
            block_max = __hmax2(block_max, __habs2(values[i][j]));
        }
        amax[block] = __bfloat162float(__hmax(block_max.x, block_max.y));
    }
    #pragma unroll
    for (int i = 0; i < K_BLOCKS; ++i) {
        local_scales[i] = __nv_fp8_e4m3(
            fminf(amax[i] / 6.0f * global_encode, 448.0f)
        );
    }
    #pragma unroll
    for (int i = 0; i < K_BLOCKS; ++i) {
        const int block = (i + tile_row / 8) % K_BLOCKS;
        const float local_decode = static_cast<float>(local_scales[block]);
        const float encode = 1.0f / fmaxf(
            local_decode * global_decode,
            0.000000000001f
        );
        const int offset_base = tile_row * globals::TILE_N / 2 +
            (block + col_half * K_BLOCKS) *
                globals::K_BLOCK_SIZE / 2;
        #pragma unroll
        for (int j = 0; j < VALUES_PER_BLOCK; ++j) {
            const int offset = offset_base + ((tid + j) & 7);
            const float2 scaled = {
                __bfloat162float(values[i][j].x) * encode,
                __bfloat162float(values[i][j].y) * encode,
            };
            asm volatile(
                "{st.shared.b8 [%0], %1;}"
                :: "r"(
                    static_cast<uint32_t>(
                        __cvta_generic_to_shared(&packed)
                    ) + offset
                ),
                "r"(static_cast<uint32_t>(
                    __nv_cvt_float2_to_fp4x2(
                        scaled,
                        __NV_E2M1,
                        cudaRoundNearest
                    )
                ))
            );
        }
    }

    // The FP4 payload was encoded against the unscaled local decode value.
    // Scaling metadata here applies the field gain exactly once on decode
    // while preserving identical payload codes and avoiding scalar packing.
    #pragma unroll
    for (int i = 0; i < K_BLOCKS; ++i) {
        local_scales[i] = __nv_fp8_e4m3(
            fminf(
                static_cast<float>(local_scales[i]) * field_decode_scale,
                448.0f
            )
        );
    }

    const int scale_offset =
        (tile_row % 32) * 16 + (tile_row / 32) * 4;
    asm volatile(
        "{st.shared.b32 [%0], %1;}"
        :: "r"(
            static_cast<uint32_t>(
                __cvta_generic_to_shared(&scales[col_half])
            ) + scale_offset
        ),
        "r"(*reinterpret_cast<uint32_t *>(&local_scales[0]))
    );

    __syncthreads();
    if (tid == 0) {
        tma::store_async(g.A, packed, {row_tile, col_tile});
        tma::store_async(g.A_sc, scales[0], {row_tile, col_tile * 2, 0});
        tma::store_async(g.A_sc, scales[1], {row_tile, col_tile * 2 + 1, 0});
        tma::store_commit_group();
        tma::store_async_wait<0>();
    }
}

// The one-lane/materialized route needs no second thread per row.  Match the
// generic delayed-scale producer's four-warp map so each lane owns one full
// 128-value row.  This is particularly important for paired D64 heads: the
// eight-warp hierarchical map otherwise doubles CTA issue overhead despite
// loading only one BF16 reduction lane.
__global__ __launch_bounds__(128, 1) void materialized_kernel(
    const __grid_constant__ globals g
) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(reinterpret_cast<int *>(&__shm[0]));
    globals::bf16_tile &input = allocator.allocate<globals::bf16_tile>();
    // Keep the packed destination separate from the TMA input.  The extra
    // 8 KiB lets one thread consume and quantize a K64 half at a time instead
    // of retaining the full K128 row in registers.
    globals::fp4_tile &packed =
        *reinterpret_cast<globals::fp4_tile *>(
            reinterpret_cast<uint64_t>(&input) + sizeof(input)
        );
    globals::scale_vec (&scales)[2] =
        *reinterpret_cast<globals::scale_vec(*)[2]>(
            reinterpret_cast<uint64_t>(&packed) + sizeof(packed)
        );

    const int tid = static_cast<int>(threadIdx.x);
    const int tile_row = tid;
    const int row_tile =
        g.row_tile_begin + static_cast<int>(blockIdx.y);
    const int col_tile =
        g.col_tile_begin + static_cast<int>(blockIdx.x);
    const bool is_dq = col_tile < g.q_heads;
    const bool is_dk =
        col_tile >= g.q_heads && col_tile < g.q_heads + g.kv_heads;
    const float field_decode_scale = is_dq
        ? g.dq_decode_scale
        : is_dk ? g.dk_decode_scale : g.dv_decode_scale;
    const int source_head = is_dq
        ? col_tile
        : is_dk
            ? col_tile - g.q_heads
            : col_tile - g.q_heads - g.kv_heads;

    if (tid == 0) {
        g.dQ.template prefetch_tma<globals::bf16_tile>();
        g.dK.template prefetch_tma<globals::bf16_tile>();
        g.dV.template prefetch_tma<globals::bf16_tile>();
    }

    __shared__ semaphore input_arrived;
    if (tid == 0) {
        init_semaphore(input_arrived, 0, 1);
        tma::expect(input_arrived, input);
        if (is_dq) {
            tma::load_async(
                input,
                g.dQ,
                {row_tile, source_head},
                input_arrived
            );
        } else if (is_dk) {
            tma::load_async(
                input,
                g.dK,
                {row_tile, source_head},
                input_arrived
            );
        } else {
            tma::load_async(
                input,
                g.dV,
                {row_tile, source_head},
                input_arrived
            );
        }
    }

    const float global_decode = g.A_scale[{0}];
    const float global_encode =
        1.0f / fmaxf(global_decode, 0.000000000001f);
    constexpr int K_BLOCKS_HALF =
        globals::TILE_N / globals::K_BLOCK_SIZE / 2;
    constexpr int VALUES_PER_BLOCK = globals::K_BLOCK_SIZE / 2;
    bf16_2 values[K_BLOCKS_HALF][VALUES_PER_BLOCK];
    fp8e4m3 local_scales[K_BLOCKS_HALF];

    __syncthreads();
    wait(input_arrived, 0);
    const int global_row = row_tile * globals::TILE_M + tile_row;
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < K_BLOCKS_HALF; ++i) {
            const int block =
                (i + tid / 8) % K_BLOCKS_HALF +
                col_half * K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < VALUES_PER_BLOCK; ++j) {
                const int col = block * globals::K_BLOCK_SIZE +
                    ((tid + j) * 2) % globals::K_BLOCK_SIZE;
                const int offset =
                    (tile_row * globals::TILE_N + col) * sizeof(bf16);
                move<bf16_2>::lds(
                    values[i][j],
                    static_cast<uint32_t>(
                        __cvta_generic_to_shared(&input)
                    ) + offset
                );
            }
        }
        if (is_dq || is_dk) {
            #pragma unroll
            for (int i = 0; i < K_BLOCKS_HALF; ++i) {
                const int block =
                    (i + tid / 8) % K_BLOCKS_HALF +
                    col_half * K_BLOCKS_HALF;
                #pragma unroll
                for (int j = 0; j < VALUES_PER_BLOCK; ++j) {
                    const int tile_col = block * globals::K_BLOCK_SIZE +
                        ((tid + j) * 2) % globals::K_BLOCK_SIZE;
                    const size_t rope_offset =
                        static_cast<size_t>(global_row) *
                            globals::ROTARY_PAIRS + tile_col / 2;
                    values[i][j] = inverse_rope_pair(
                        values[i][j],
                        g.rope_packed[rope_offset]
                    );
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < K_BLOCKS_HALF; ++i) {
            const int block = (i + tid / 8) % K_BLOCKS_HALF;
            bf16_2 block_max = __habs2(values[i][0]);
            #pragma unroll
            for (int j = 1; j < VALUES_PER_BLOCK; ++j) {
                block_max = __hmax2(
                    block_max,
                    __habs2(values[i][j])
                );
            }
            local_scales[block] = __nv_fp8_e4m3(
                fminf(
                    __bfloat162float(
                        __hmax(block_max.x, block_max.y)
                    ) / 6.0f * global_encode,
                    448.0f
                )
            );
        }
        #pragma unroll
        for (int i = 0; i < K_BLOCKS_HALF; ++i) {
            const int block = (i + tid / 8) % K_BLOCKS_HALF;
            const float local_decode =
                static_cast<float>(local_scales[block]);
            const float encode = 1.0f / fmaxf(
                local_decode * global_decode,
                0.000000000001f
            );
            const int offset_base = tile_row * globals::TILE_N / 2 +
                (block + col_half * K_BLOCKS_HALF) *
                    globals::K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int j = 0; j < VALUES_PER_BLOCK; ++j) {
                const int offset = offset_base + ((tid + j) & 7);
                const float2 scaled = {
                    __bfloat162float(values[i][j].x) * encode,
                    __bfloat162float(values[i][j].y) * encode,
                };
                asm volatile(
                    "{st.shared.b8 [%0], %1;}"
                    :: "r"(
                        static_cast<uint32_t>(
                            __cvta_generic_to_shared(&packed)
                        ) + offset
                    ),
                    "r"(static_cast<uint32_t>(
                        __nv_cvt_float2_to_fp4x2(
                            scaled,
                            __NV_E2M1,
                            cudaRoundNearest
                        )
                    ))
                );
            }
        }
        #pragma unroll
        for (int i = 0; i < K_BLOCKS_HALF; ++i) {
            local_scales[i] = __nv_fp8_e4m3(
                fminf(
                    static_cast<float>(local_scales[i]) *
                        field_decode_scale,
                    448.0f
                )
            );
        }
        const int scale_offset =
            (tile_row % 32) * 16 + (tile_row / 32) * 4;
        asm volatile(
            "{st.shared.b32 [%0], %1;}"
            :: "r"(
                static_cast<uint32_t>(
                    __cvta_generic_to_shared(&scales[col_half])
                ) + scale_offset
            ),
            "r"(*reinterpret_cast<uint32_t *>(&local_scales[0]))
        );
    }

    __syncthreads();
    if (tid == 0) {
        tma::store_async(g.A, packed, {row_tile, col_tile});
        tma::store_async(g.A_sc, scales[0], {row_tile, col_tile * 2, 0});
        tma::store_async(
            g.A_sc,
            scales[1],
            {row_tile, col_tile * 2 + 1, 0}
        );
        tma::store_commit_group();
        tma::store_async_wait<0>();
    }
}

inline void launch(const globals &g, cudaStream_t stream) {
    auto launch_kernel = [&]<bool HIERARCHICAL>() {
        CUDACHECK(cudaFuncSetAttribute(
            kernel<HIERARCHICAL>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.dynamic_shared_memory()
        ));
        kernel<HIERARCHICAL><<<
            g.grid(),
            g.block(),
            g.dynamic_shared_memory(),
            stream
        >>>(g);
    };
    if (g.dq_reduction_lanes == 2) {
        launch_kernel.template operator()<true>();
    } else {
        CUDACHECK(cudaFuncSetAttribute(
            materialized_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.materialized_dynamic_shared_memory()
        ));
        materialized_kernel<<<
            g.grid(),
            128,
            g.materialized_dynamic_shared_memory(),
            stream
        >>>(g);
        CUDACHECK(cudaGetLastError());
    }
    CUDACHECK(cudaGetLastError());
}

} // namespace tkfa4_gqa_d128_hierarchical_qkv_nvfp4

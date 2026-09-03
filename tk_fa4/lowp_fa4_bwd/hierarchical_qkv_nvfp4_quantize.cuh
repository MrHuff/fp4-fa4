#pragma once

// Projection-native gradient operand producer for the hierarchical dQ route.
//
// Attention leaves even and odd K/V-owner contributions in two private BF16
// dQ reduction lanes.  Each K128 CTA uses two 128-thread row groups, one per
// K64 half, folds the lanes in registers, applies inverse RoPE to dQ/dK, and
// forms the delayed-scale NVFP4 operand.  dK and dV are read from their normal
// completed BF16 outputs.  The output reduction order is stacked
// [all dQ | all dK | all dV], matching the cached stacked weight layout.

#include "kittens.cuh"

namespace tkfa4_hierarchical_qkv_nvfp4 {

using namespace kittens;

struct config {
    static constexpr int NUM_WARPS = 8;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int K_BLOCK_SIZE = 16;
    static constexpr int QK_DEPTH = 192;
    static constexpr int V_DEPTH = 128;
    static constexpr int ROTARY_PAIRS = QK_DEPTH / 2;

    using bf16_tile = st_bf<TILE_M, TILE_N, false>;
    using fp4_tile = st_fp4e2m1_2<TILE_M, TILE_N / 2, false>;
    using scale_vec = sv_hf<256>;
    using bf16_gl = gl<bf16, 1, 1, -1, -1, bf16_tile>;
    using fp4_gl = gl<fp4e2m1_2, 1, 1, -1, -1, fp4_tile>;
    using scale_gl = gl<half, 1, -1, -1, 256, scale_vec>;
    using global_scale_gl = gl<float, 1, 1, 1, 1>;

    // dQ has [2 * rows, q_width]: lane one follows lane zero in the row
    // dimension. dK/dV each have [rows, width].
    bf16_gl dQ;
    bf16_gl dK;
    bf16_gl dV;
    fp4_gl A;
    scale_gl A_sc;
    global_scale_gl A_scale;
    const bf16 *rope_cos;
    const bf16 *rope_sin;
    int rows;
    int q_width;
    int v_width;
    int dq_reduction_lanes;

    __host__ inline dim3 grid() const {
        const int reduction = 2 * q_width + v_width;
        return dim3(reduction / TILE_N, rows / TILE_M);
    }

    __host__ inline dim3 block() const {
        return dim3(config::NUM_THREADS);
    }

    __host__ inline int dynamic_shared_memory() const {
        return TILE_M * TILE_N * sizeof(bf16) + 1024;
    }
};

__device__ __forceinline__ bf16_2 inverse_rope_pair(
    bf16_2 pair,
    bf16 cosine,
    bf16 sine
) {
    const float2 values = __bfloat1622float2(pair);
    const float c = __bfloat162float(cosine);
    const float s = __bfloat162float(sine);
    return __floats2bfloat162_rn(
        fmaf(values.y, s, values.x * c),
        fmaf(-values.x, s, values.y * c)
    );
}

__global__ __launch_bounds__(config::NUM_THREADS, 1) void kernel(
    const __grid_constant__ globals g
) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(reinterpret_cast<int *>(&__shm[0]));
    globals::bf16_tile &input = allocator.allocate<globals::bf16_tile>();
    // Packed output and scale storage overlay input only after both K64 row
    // groups have captured every source value needed by this K128 CTA.
    globals::fp4_tile &packed =
        *reinterpret_cast<globals::fp4_tile *>(&input);
    globals::scale_vec (&scales)[2] =
        *reinterpret_cast<globals::scale_vec(*)[2]>(
            reinterpret_cast<uint64_t>(&packed) + sizeof(packed)
        );

    const int tid = static_cast<int>(threadIdx.x);
    const int tile_row = tid % globals::TILE_M;
    const int col_half = tid / globals::TILE_M;
    const int row_tile = static_cast<int>(blockIdx.y);
    const int col_tile = static_cast<int>(blockIdx.x);
    const int global_col_base = col_tile * globals::TILE_N;
    const bool is_dq = global_col_base < g.q_width;
    const bool is_dk =
        global_col_base >= g.q_width && global_col_base < 2 * g.q_width;
    const int source_col_tile = is_dq
        ? col_tile
        : is_dk
            ? (global_col_base - g.q_width) / globals::TILE_N
            : (global_col_base - 2 * g.q_width) / globals::TILE_N;
    const int q_tiles = g.rows / globals::TILE_M;

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
                {row_tile, source_col_tile},
                input_arrived
            );
        } else if (is_dk) {
            tma::load_async(
                input,
                g.dK,
                {row_tile, source_col_tile},
                input_arrived
            );
        } else {
            tma::load_async(
                input,
                g.dV,
                {row_tile, source_col_tile},
                input_arrived
            );
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
    wait(input_arrived, 0);
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

    // The second dQ lane is the only missing part of the completed reduction.
    if (is_dq && g.dq_reduction_lanes == 2) {
        __syncthreads();
        if (tid == 0) {
            tma::expect(input_arrived, input);
            tma::load_async(
                input,
                g.dQ,
                {q_tiles + row_tile, source_col_tile},
                input_arrived
            );
        }
        __syncthreads();
        wait(input_arrived, 1);
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
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input)) +
                        offset
                );
                values[i][j] = __hadd2(values[i][j], lane_one);
            }
        }
    }

    const int global_row = row_tile * globals::TILE_M + tile_row;
    if (is_dq || is_dk) {
        const int qk_base = is_dq
            ? global_col_base
            : global_col_base - g.q_width;
        #pragma unroll
        for (int i = 0; i < K_BLOCKS; ++i) {
            const int block = (i + tile_row / 8) % K_BLOCKS +
                col_half * K_BLOCKS;
            #pragma unroll
            for (int j = 0; j < VALUES_PER_BLOCK; ++j) {
                const int tile_col = block * globals::K_BLOCK_SIZE +
                    ((tid + j) * 2) % globals::K_BLOCK_SIZE;
                const int qk_col =
                    (qk_base + tile_col) % globals::QK_DEPTH;
                const size_t rope_offset =
                    static_cast<size_t>(global_row) *
                        globals::ROTARY_PAIRS + qk_col / 2;
                values[i][j] = inverse_rope_pair(
                    values[i][j],
                    g.rope_cos[rope_offset],
                    g.rope_sin[rope_offset]
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

inline void launch(const globals &g, cudaStream_t stream) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()
    ));
    kernel<<<g.grid(), g.block(), g.dynamic_shared_memory(), stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

} // namespace tkfa4_hierarchical_qkv_nvfp4

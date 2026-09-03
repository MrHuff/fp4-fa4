#pragma once

// Delayed-scale NVFP4 quantization with pair-native inverse RoPE fused into
// the register path.  This is local to the FA4 integration rather than a
// modification of ThunderKittens' generic NVFP4 quantizer.

#include "kittens.cuh"

namespace tkfa4_inverse_rope_nvfp4_quantize {

using namespace kittens;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int K_BLOCK_SIZE = 16;
    static constexpr int QK_DEPTH = 192;
    static constexpr int V_DEPTH = 128;
    static constexpr int HEAD_WIDTH = QK_DEPTH * 2 + V_DEPTH;
    static constexpr int ROTARY_PAIRS = QK_DEPTH / 2;

    using A_bf16_tile  = st_bf<TILE_M, TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<TILE_M, TILE_N / 2, false>;
    using A_sc_vec     = sv_hf<256>;

    using A_bf16_gl      = gl<bf16,      1,  1, -1, -1, A_bf16_tile>;
    using A_fp4x2_gl     = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,      1, -1, -1, 256, A_sc_vec>;
    using A_sc_global_gl = gl<float,     1,  1,  1,  1>;

    A_bf16_gl A_bf16;
    A_fp4x2_gl A_fp4x2;
    A_sc_gl A_sc;
    A_sc_global_gl A_sc_global;
    const bf16 *rope_cos;
    const bf16 *rope_sin;

    __host__ inline dim3 grid() const {
        return dim3(A_bf16.cols() / TILE_N, A_bf16.rows() / TILE_M);
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
    // Match the standalone reference boundary: evaluate in FP32 and round
    // once to BF16 before local-scale selection and E2M1 conversion.
    const float2 values = __bfloat1622float2(pair);
    const float c = __bfloat162float(cosine);
    const float s = __bfloat162float(sine);
    return __floats2bfloat162_rn(
        fmaf(values.y, s, values.x * c),
        fmaf(-values.x, s, values.y * c)
    );
}

template <bool PUBLISH_INVERSE_BF16>
__device__ inline void quantize_kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator(reinterpret_cast<int *>(&__shm[0]));
    globals::A_bf16_tile &A_bf16_smem =
        sm_allocator.allocate<globals::A_bf16_tile>();
    globals::A_fp4x2_tile &A_fp4x2_smem =
        *reinterpret_cast<globals::A_fp4x2_tile *>(&A_bf16_smem);
    globals::A_sc_vec (&A_sc_smem)[2] =
        *reinterpret_cast<globals::A_sc_vec(*)[2]>(
            reinterpret_cast<uint64_t>(&A_fp4x2_smem) +
            sizeof(A_fp4x2_smem)
        );

    const int tid = threadIdx.x;
    const int row_tile = blockIdx.y;
    const int col_tile = blockIdx.x;
    const int tile_row = tid;

    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, A_bf16_smem);
        tma::load_async(
            A_bf16_smem,
            G.A_bf16,
            {row_tile, col_tile},
            inputs_arrived
        );
    }

    const float s_global_dec = G.A_sc_global[{0}];
    const float s_global_enc =
        1.0f / fmaxf(s_global_dec, 0.000000000001f);

    constexpr int NUM_K_BLOCKS_HALF =
        globals::TILE_N / globals::K_BLOCK_SIZE / 2;
    constexpr int N_PER_K_BLOCK = globals::K_BLOCK_SIZE / 2;
    bf16_2 A_bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 A_sc_reg[2][NUM_K_BLOCKS_HALF];

    __syncthreads();
    wait(inputs_arrived, 0);

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx =
                (i + tid / 8) % NUM_K_BLOCKS_HALF +
                col_half * NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int tile_col =
                    k_block_idx * globals::K_BLOCK_SIZE +
                    ((tid + j) * 2) % globals::K_BLOCK_SIZE;
                const int offset =
                    (tile_row * globals::TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(
                    A_bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(
                        __cvta_generic_to_shared(&A_bf16_smem)
                    ) + offset
                );
            }
        }
    }

    const int global_row = row_tile * globals::TILE_M + tile_row;
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx =
                (i + tid / 8) % NUM_K_BLOCKS_HALF +
                col_half * NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int tile_col =
                    k_block_idx * globals::K_BLOCK_SIZE +
                    ((tid + j) * 2) % globals::K_BLOCK_SIZE;
                const int global_col =
                    col_tile * globals::TILE_N + tile_col;
                const int head_col = global_col % globals::HEAD_WIDTH;
                if (head_col < 2 * globals::QK_DEPTH) {
                    const int qk_col = head_col % globals::QK_DEPTH;
                    const size_t rope_offset =
                        static_cast<size_t>(global_row) *
                            globals::ROTARY_PAIRS +
                        qk_col / 2;
                    A_bf16_reg[col_half][i][j] = inverse_rope_pair(
                        A_bf16_reg[col_half][i][j],
                        G.rope_cos[rope_offset],
                        G.rope_sin[rope_offset]
                    );
                }
            }
        }
    }

    // A_fp4x2_smem aliases A_bf16_smem.  Every thread must finish loading
    // and rotating its complete row before any thread can begin packed E2M1
    // publication, including the no-BF16-republication specialization.
    __syncthreads();

    if constexpr (PUBLISH_INVERSE_BF16) {
        #pragma unroll
        for (int col_half = 0; col_half < 2; ++col_half) {
            #pragma unroll
            for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
                const int k_block_idx =
                    (i + tid / 8) % NUM_K_BLOCKS_HALF +
                    col_half * NUM_K_BLOCKS_HALF;
                #pragma unroll
                for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                    const int tile_col =
                        k_block_idx * globals::K_BLOCK_SIZE +
                        ((tid + j) * 2) % globals::K_BLOCK_SIZE;
                    const int offset =
                        (tile_row * globals::TILE_N + tile_col) * sizeof(bf16);
                    move<bf16_2>::sts(
                        static_cast<uint32_t>(
                            __cvta_generic_to_shared(&A_bf16_smem)
                        ) + offset,
                        A_bf16_reg[col_half][i][j]
                    );
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            tma::store_async(
                G.A_bf16,
                A_bf16_smem,
                {row_tile, col_tile}
            );
        }
    }

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        float amax[NUM_K_BLOCKS_HALF];
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx =
                (i + tid / 8) % NUM_K_BLOCKS_HALF;
            bf16_2 block_max = __habs2(A_bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; ++j) {
                block_max = __hmax2(
                    block_max,
                    __habs2(A_bf16_reg[col_half][i][j])
                );
            }
            amax[k_block_idx] = __bfloat162float(
                __hmax(block_max.x, block_max.y)
            );
        }

        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            A_sc_reg[col_half][i] =
                __nv_fp8_e4m3(amax[i] / 6.0f * s_global_enc);
        }
    }

    if constexpr (PUBLISH_INVERSE_BF16) {
        if (tid == 0) {
            // Only the shared-memory read must complete before the BF16 tile
            // is overlaid with packed E2M1 output.
            tma::store_async_read_wait<0>();
        }
        __syncthreads();
    }

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx =
                (i + tid / 8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec =
                static_cast<float>(A_sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f /
                fmaxf(s_local_dec * s_global_dec, 0.000000000001f);
            const int offset_base =
                tile_row * globals::TILE_N / 2 +
                (k_block_idx + col_half * NUM_K_BLOCKS_HALF) *
                globals::K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int offset = offset_base + ((tid + j) & 7);
                const float2 scaled = {
                    __bfloat162float(A_bf16_reg[col_half][i][j].x) * s_enc,
                    __bfloat162float(A_bf16_reg[col_half][i][j].y) * s_enc
                };
                asm volatile(
                    "{st.shared.b8 [%0], %1;}"
                    :: "r"(
                        static_cast<uint32_t>(
                            __cvta_generic_to_shared(&A_fp4x2_smem)
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
    }

    const int scale_offset =
        (tile_row % 32) * 16 + (tile_row / 32) * 4;
    asm volatile(
        "{st.shared.b32 [%0], %1;}"
        :: "r"(
            static_cast<uint32_t>(
                __cvta_generic_to_shared(&A_sc_smem[0])
            ) + scale_offset
        ),
        "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[0][0]))
    );
    asm volatile(
        "{st.shared.b32 [%0], %1;}"
        :: "r"(
            static_cast<uint32_t>(
                __cvta_generic_to_shared(&A_sc_smem[1])
            ) + scale_offset
        ),
        "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[1][0]))
    );

    __syncthreads();
    if (tid == 0) {
        tma::store_async(G.A_fp4x2, A_fp4x2_smem, {row_tile, col_tile});
        tma::store_async(G.A_sc, A_sc_smem[0], {row_tile, col_tile * 2, 0});
        tma::store_async(
            G.A_sc,
            A_sc_smem[1],
            {row_tile, col_tile * 2 + 1, 0}
        );
    }
}

} // namespace tkfa4_inverse_rope_nvfp4_quantize

#pragma once

// Bounded, tile-ready delayed-scale NVFP4 producer for completed BF16 dQ.
//
// Attention reduces dQ into 128-row BF16 tiles and publishes a release
// counter only after every async-proxy write for a contributing owner has
// retired.  This kernel waits on those counters, converts two adjacent
// 128x128 BF16 tiles into the single K256 operand consumed by NVFP4 MMA, and
// then publishes a second release counter after the packed payload and both
// scale pages are globally visible.  A small persistent grid lets this work
// overlap later attention owners without occupying the machine.

#include <cuda/atomic>
#include <cstdlib>

#include "kittens.cuh"

namespace tkfa4_tile_ready_nvfp4 {

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
    static constexpr int REDUCTION_TILE = 256;
    static constexpr int K_BLOCK_SIZE = 16;
    static constexpr int HEAD_WIDTH = 192;

    using A_bf16_tile = st_bf<TILE_M, TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<TILE_M, TILE_N / 2, false>;
    using A_sc_vec = sv_hf<256>;

    using A_bf16_gl = gl<bf16, 1, 1, -1, -1, A_bf16_tile>;
    using A_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl = gl<half, 1, -1, -1, 256, A_sc_vec>;
    using A_sc_global_gl = gl<float, 1, 1, 1, 1>;

    A_bf16_gl A_bf16;
    A_fp4x2_gl A_fp4x2;
    A_sc_gl A_sc;
    A_sc_global_gl A_sc_global;
    const uint32_t *dq_tile_arrivals;
    uint32_t *operand_ready;
    int heads;
    int q_tiles;
    int reduction_tiles;
    int cta_cap;
    int task_begin = 0;
    int task_end = 0;

    __host__ inline dim3 grid() const {
        const int total_tasks = q_tiles * reduction_tiles;
        const int begin = max(0, min(task_begin, total_tasks));
        const int end = task_end > begin
            ? min(task_end, total_tasks)
            : total_tasks;
        const int tasks = end - begin;
        int cap = cta_cap < 0
            ? tasks
            : (cta_cap > 0 ? cta_cap : (heads <= 8 ? 6 : 12));
        if (cta_cap >= 0) {
            if (const char *value = std::getenv(
                    "TK_FA4_DQ_NVFP4_PACK_CTAS"
                )) {
                const int requested = std::atoi(value);
                if (requested > 0) {
                    cap = requested;
                }
            }
        }
        return dim3(max(1, min(tasks, cap)));
    }

    __host__ inline dim3 block() const {
        return dim3(config::NUM_THREADS);
    }

    __host__ inline int dynamic_shared_memory() const {
        return TILE_M * TILE_N * sizeof(bf16) + 1024;
    }
};

__device__ __forceinline__ void wait_for_dq_tile(
    const uint32_t *counter,
    uint32_t expected
) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> arrival(
        *const_cast<uint32_t *>(counter)
    );
    while (arrival.load(cuda::memory_order_acquire) < expected) {
        __nanosleep(256);
    }
}

__device__ __forceinline__ void publish_operand_tile(uint32_t *counter) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(*counter);
    ready.store(1u, cuda::memory_order_release);
}

__global__ __launch_bounds__(config::NUM_THREADS, 1) void kernel(
    const __grid_constant__ globals g
) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator(reinterpret_cast<int *>(&__shm[0]));
    globals::A_bf16_tile &A_bf16_smem =
        allocator.allocate<globals::A_bf16_tile>();
    globals::A_fp4x2_tile &A_fp4x2_smem =
        *reinterpret_cast<globals::A_fp4x2_tile *>(&A_bf16_smem);
    globals::A_sc_vec (&A_sc_smem)[2] =
        *reinterpret_cast<globals::A_sc_vec(*)[2]>(
            reinterpret_cast<uint64_t>(&A_fp4x2_smem) +
            sizeof(A_fp4x2_smem)
        );

    const int tid = static_cast<int>(threadIdx.x);
    const int tile_row = tid;
    const int total_tasks = g.q_tiles * g.reduction_tiles;
    const int task_begin = max(0, min(g.task_begin, total_tasks));
    const int task_end = g.task_end > task_begin
        ? min(g.task_end, total_tasks)
        : total_tasks;
    const int task_count = task_end - task_begin;
    const float s_global_dec = g.A_sc_global[{0}];
    const float s_global_enc =
        1.0f / fmaxf(s_global_dec, 0.000000000001f);

    if (tid == 0) {
        g.A_bf16.template prefetch_tma<globals::A_bf16_tile>();
    }

    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
    }
    __syncthreads();
    int input_phase = 0;

    constexpr int NUM_K_BLOCKS_HALF =
        globals::TILE_N / globals::K_BLOCK_SIZE / 2;
    constexpr int N_PER_K_BLOCK = globals::K_BLOCK_SIZE / 2;

    for (int task_offset = static_cast<int>(blockIdx.x);
         task_offset < task_count;
         task_offset += static_cast<int>(gridDim.x)) {
        const int task = task_begin + task_offset;
        const int row_tile = task / g.reduction_tiles;
        const int reduction_tile = task - row_tile * g.reduction_tiles;

        if (tid == 0) {
            const int first_col = reduction_tile * globals::REDUCTION_TILE;
            const int last_col = first_col + globals::REDUCTION_TILE - 1;
            const int first_head = first_col / globals::HEAD_WIDTH;
            const int last_head = last_col / globals::HEAD_WIDTH;
            const uint32_t expected = static_cast<uint32_t>(
                2 * ((row_tile >> 1) + 1)
            );
            for (int head = first_head; head <= last_head; ++head) {
                wait_for_dq_tile(
                    g.dq_tile_arrivals +
                        static_cast<size_t>(head) * g.q_tiles + row_tile,
                    expected
                );
            }
        }
        __syncthreads();

        #pragma unroll
        for (int column_half = 0; column_half < 2; ++column_half) {
            const int col_tile = reduction_tile * 2 + column_half;
            if (tid == 0) {
                tma::expect(inputs_arrived, A_bf16_smem);
                tma::load_async(
                    A_bf16_smem,
                    g.A_bf16,
                    {row_tile, col_tile},
                    inputs_arrived
                );
            }
            __syncthreads();
            wait(inputs_arrived, input_phase);

            bf16_2 A_bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
            fp8e4m3 A_sc_reg[2][NUM_K_BLOCKS_HALF];

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
                            (tile_row * globals::TILE_N + tile_col) *
                            sizeof(bf16);
                        move<bf16_2>::lds(
                            A_bf16_reg[col_half][i][j],
                            static_cast<uint32_t>(
                                __cvta_generic_to_shared(&A_bf16_smem)
                            ) + offset
                        );
                    }
                }
            }
            // The packed output aliases the BF16 input page.
            __syncthreads();

            #pragma unroll
            for (int col_half = 0; col_half < 2; ++col_half) {
                float amax[NUM_K_BLOCKS_HALF];
                #pragma unroll
                for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
                    const int k_block_idx =
                        (i + tid / 8) % NUM_K_BLOCKS_HALF;
                    bf16_2 block_max =
                        __habs2(A_bf16_reg[col_half][i][0]);
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
                    // E4M3's 0x7f encoding is NaN.  Saturating here is
                    // equivalent to the standalone delayed-scale fixup but
                    // lets the ready flag cover a genuinely complete page.
                    A_sc_reg[col_half][i] = __nv_fp8_e4m3(fminf(
                        amax[i] / 6.0f * s_global_enc,
                        448.0f
                    ));
                }
                #pragma unroll
                for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
                    const int k_block_idx =
                        (i + tid / 8) % NUM_K_BLOCKS_HALF;
                    const float s_local_dec =
                        static_cast<float>(A_sc_reg[col_half][k_block_idx]);
                    const float s_enc = 1.0f / fmaxf(
                        s_local_dec * s_global_dec,
                        0.000000000001f
                    );
                    const int offset_base =
                        tile_row * globals::TILE_N / 2 +
                        (k_block_idx +
                         col_half * NUM_K_BLOCKS_HALF) *
                            globals::K_BLOCK_SIZE / 2;
                    #pragma unroll
                    for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                        const int offset = offset_base + ((tid + j) & 7);
                        const float2 scaled = {
                            __bfloat162float(
                                A_bf16_reg[col_half][i][j].x
                            ) * s_enc,
                            __bfloat162float(
                                A_bf16_reg[col_half][i][j].y
                            ) * s_enc,
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
                tma::store_async(
                    g.A_fp4x2,
                    A_fp4x2_smem,
                    {row_tile, col_tile}
                );
                tma::store_async(
                    g.A_sc,
                    A_sc_smem[0],
                    {row_tile, col_tile * 2, 0}
                );
                tma::store_async(
                    g.A_sc,
                    A_sc_smem[1],
                    {row_tile, col_tile * 2 + 1, 0}
                );
                tma::store_commit_group();
                tma::store_async_wait<0>();
            }
            __syncthreads();
            input_phase ^= 1;
        }

        if (tid == 0) {
            // The preceding TMA wait makes payload and scale publication
            // complete before the release store becomes visible.
            publish_operand_tile(g.operand_ready + task);
        }
        __syncthreads();
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

} // namespace tkfa4_tile_ready_nvfp4

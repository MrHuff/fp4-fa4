/*
 * Pipelined AMAX Kernels — Tiled + Double-Buffered
 *
 * Replaces the naive scalar/vec8 amax reduction kernels with
 * pipelined versions that use:
 *   1. Shared memory tiling with cp.async for async HBM→SMEM copies
 *   2. Double-buffering (2 SMEM tiles) to overlap load N+1 with process N
 *   3. Vectorized processing (8 bf16 per operation via bfloat162)
 *   4. Warp shuffle + shared memory block reduction
 *
 * Three kernel variants:
 *   - fused_amax_pipelined_kernel:      single global amax + sg
 *   - grouped_amax_pipelined_kernel:    per-split amax (dim=0 splits)
 *   - grouped_amax_dim1_pipelined_kernel: per-column-group amax (dim=1 splits)
 *
 * Target: SM100 (GB200), compiled with -arch sm_100a
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace pipelined_amax {

// ─────────────────────── Configuration ───────────────────────
constexpr int THREADS    = 256;       // 8 warps per block
constexpr int WARPS      = THREADS / 32;
constexpr int VEC_BF16   = 8;        // 8 bf16 per vector (= 16 bytes = sizeof(int4))
constexpr int VEC_BYTES  = VEC_BF16 * 2;  // 16 bytes

// Tile = TILE_VECS vectors × VEC_BF16 bf16 per vec = TILE_ELEMS bf16
// Each tile is 16 KB (good for L1/shared memory locality)
constexpr int TILE_VECS  = 1024;     // 1024 vectors per tile
constexpr int TILE_ELEMS = TILE_VECS * VEC_BF16;  // 8192 bf16
constexpr int TILE_BYTES = TILE_ELEMS * 2;         // 16384 bytes = 16 KB
constexpr int NUM_BUFS   = 2;        // double buffer

// Each thread handles TILE_VECS/THREADS = 4 vectors (32 bf16) per tile
constexpr int VECS_PER_THREAD = TILE_VECS / THREADS;  // 4

constexpr int MAX_SPLITS  = 8;  // grouped kernels
constexpr int MAX_GROUPS  = 8;  // dim=1 grouped kernel

// Total shared memory per block for double-buffered tiles
constexpr int SMEM_BYTES = NUM_BUFS * TILE_BYTES;  // 32 KB


// ─────────────────────── Inline helpers ───────────────────────

// Reduce 8 bf16 (loaded as int4) to a single float max(|x|)
__device__ __forceinline__
float reduce_vec8_absmax(const void* ptr) {
    const __nv_bfloat162* v = reinterpret_cast<const __nv_bfloat162*>(ptr);
    __nv_bfloat162 m = __hmax2(__habs2(v[0]), __habs2(v[1]));
    m = __hmax2(m, __hmax2(__habs2(v[2]), __habs2(v[3])));
    return __bfloat162float(__hmax(__high2bfloat16(m), __low2bfloat16(m)));
}

// Warp-level fmax reduction
__device__ __forceinline__
float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

// Block-level fmax reduction (requires shared memory for inter-warp)
__device__ __forceinline__
float block_reduce_max(float val, float* warp_smem) {
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x % 32;

    val = warp_reduce_max(val);
    if (lane == 0) warp_smem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < WARPS) ? warp_smem[lane] : 0.0f;
        #pragma unroll
        for (int mask = WARPS / 2; mask > 0; mask >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// Global atomicMax for positive float (uses uint reinterpretation)
__device__ __forceinline__
void atomic_max_float(float* addr, float val) {
    unsigned int* p = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *p;
    unsigned int want = __float_as_uint(val);
    while (want > old) {
        old = atomicCAS(p, old, want);
    }
}

// Issue cp.async for 16 bytes (1 vector = 8 bf16) from global to shared
__device__ __forceinline__
void cp_async_vec(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr)
    );
}

__device__ __forceinline__
void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__
void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__
void cp_async_wait_prior() {
    // Wait for all but the most recent commit group
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}


// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Fused amax + sg — tiled + double-buffered
// ═══════════════════════════════════════════════════════════════════
//
// Flow per block:
//   1. Each block "owns" a chunk of the input (contiguous region)
//   2. The chunk is processed in TILE_ELEMS-sized tiles
//   3. Two shared memory buffers alternate (double-buffer)
//   4. cp.async loads tile N+1 while block processes tile N from SMEM
//   5. Block reduces to single float, atomicMax into global amax
//   6. Last block computes sg = amax / 2688.0

__launch_bounds__(THREADS)
__global__ void fused_amax_pipelined_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ amax_out,
    float* __restrict__ sg_out,
    int64_t n
) {
    // ── Shared memory layout ──
    // [0 .. TILE_BYTES-1]                  : tile buffer 0
    // [TILE_BYTES .. 2*TILE_BYTES-1]       : tile buffer 1
    // [2*TILE_BYTES .. 2*TILE_BYTES+31]    : warp reduction scratch
    extern __shared__ char smem_raw[];
    char* tile_buf[NUM_BUFS];
    tile_buf[0] = smem_raw;
    tile_buf[1] = smem_raw + TILE_BYTES;
    float* warp_smem = reinterpret_cast<float*>(smem_raw + 2 * TILE_BYTES);

    // ── Determine this block's chunk ──
    const int64_t num_vecs = n / VEC_BF16;  // total vectors
    const int64_t vecs_per_block = (num_vecs + gridDim.x - 1) / gridDim.x;
    // Align to TILE_VECS for clean tiling
    const int64_t aligned_vpb = ((vecs_per_block + TILE_VECS - 1) / TILE_VECS) * TILE_VECS;

    const int64_t my_vec_start = (int64_t)blockIdx.x * aligned_vpb;
    const int64_t my_vec_end   = min(my_vec_start + aligned_vpb, num_vecs);
    const int     num_tiles    = (int)((my_vec_end - my_vec_start + TILE_VECS - 1) / TILE_VECS);

    if (num_tiles <= 0) {
        // This block has no work. Still need to participate in last-block logic.
        __syncthreads();
        goto epilogue;
    }

    {
        float local_max = 0.0f;
        const __nv_bfloat16* base = input + my_vec_start * VEC_BF16;
        int stage = 0;

        // ── Prefetch tile 0 into buffer 0 ──
        {
            const int64_t tile_vec_start = 0;
            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD; v++) {
                int vec_in_tile = threadIdx.x + v * THREADS;
                int64_t global_vec = my_vec_start + tile_vec_start + vec_in_tile;
                void* dst = tile_buf[0] + vec_in_tile * VEC_BYTES;
                if (global_vec < num_vecs) {
                    cp_async_vec(dst, base + (tile_vec_start + vec_in_tile) * VEC_BF16);
                }
            }
            cp_async_commit();
        }

        for (int t = 0; t < num_tiles; t++) {
            int next_stage = 1 - stage;

            // ── Prefetch tile t+1 into the other buffer (if exists) ──
            if (t + 1 < num_tiles) {
                int64_t next_tile_vec_start = (int64_t)(t + 1) * TILE_VECS;
                #pragma unroll
                for (int v = 0; v < VECS_PER_THREAD; v++) {
                    int vec_in_tile = threadIdx.x + v * THREADS;
                    int64_t global_vec = my_vec_start + next_tile_vec_start + vec_in_tile;
                    void* dst = tile_buf[next_stage] + vec_in_tile * VEC_BYTES;
                    if (global_vec < num_vecs) {
                        cp_async_vec(dst, base + (next_tile_vec_start + vec_in_tile) * VEC_BF16);
                    }
                }
                cp_async_commit();
            }

            // ── Wait for current tile to be ready ──
            if (t + 1 < num_tiles)
                cp_async_wait_prior();  // wait all but most recent
            else
                cp_async_wait_all();    // last tile, wait for everything
            __syncthreads();

            // ── Process current tile from shared memory ──
            int64_t tile_vec_start = (int64_t)t * TILE_VECS;
            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD; v++) {
                int vec_in_tile = threadIdx.x + v * THREADS;
                int64_t global_vec = my_vec_start + tile_vec_start + vec_in_tile;
                if (global_vec < num_vecs) {
                    float val = reduce_vec8_absmax(tile_buf[stage] + vec_in_tile * VEC_BYTES);
                    local_max = fmaxf(local_max, val);
                }
            }

            __syncthreads();  // Ensure all threads done reading before next tile overwrites
            stage = next_stage;
        }

        // ── Block reduction ──
        float block_max = block_reduce_max(local_max, warp_smem);

        // ── Global atomicMax ──
        if (threadIdx.x == 0) {
            atomic_max_float(amax_out, block_max);
        }
    }

epilogue:
    // ── Last-block barrier to compute sg ──
    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        static __device__ unsigned int fused_block_count = 0;
        unsigned int prev = atomicAdd(&fused_block_count, 1);
        is_last = (prev == gridDim.x - 1);
        if (is_last) fused_block_count = 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        float amax = *amax_out;
        *sg_out = amax / 2688.0f;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Grouped amax (per-split, dim=0) — tiled + double-buffered
// ═══════════════════════════════════════════════════════════════════

__launch_bounds__(THREADS)
__global__ void grouped_amax_pipelined_kernel(
    const __nv_bfloat16* __restrict__ input,
    float*  __restrict__ amaxes,
    const int64_t* __restrict__ split_elem_offsets,
    int num_splits,
    int64_t total_elems
) {
    // ── Shared memory layout ──
    extern __shared__ char smem_raw[];
    char* tile_buf[NUM_BUFS];
    tile_buf[0] = smem_raw;
    tile_buf[1] = smem_raw + TILE_BYTES;
    float* warp_smem = reinterpret_cast<float*>(smem_raw + 2 * TILE_BYTES);

    // Load split offsets into shared memory
    __shared__ int64_t s_offsets[MAX_SPLITS + 1];
    if (threadIdx.x < num_splits + 1 && threadIdx.x < MAX_SPLITS + 1) {
        s_offsets[threadIdx.x] = split_elem_offsets[threadIdx.x];
    }
    __syncthreads();

    float local_max[MAX_SPLITS];
    #pragma unroll
    for (int s = 0; s < MAX_SPLITS; ++s) local_max[s] = 0.0f;

    // ── Determine this block's chunk ──
    const int64_t num_vecs = total_elems / VEC_BF16;
    const int64_t vecs_per_block = (num_vecs + gridDim.x - 1) / gridDim.x;
    const int64_t aligned_vpb = ((vecs_per_block + TILE_VECS - 1) / TILE_VECS) * TILE_VECS;

    const int64_t my_vec_start = (int64_t)blockIdx.x * aligned_vpb;
    const int64_t my_vec_end   = min(my_vec_start + aligned_vpb, num_vecs);
    const int     num_tiles    = (int)((my_vec_end - my_vec_start + TILE_VECS - 1) / TILE_VECS);

    if (num_tiles > 0) {
        const __nv_bfloat16* base = input + my_vec_start * VEC_BF16;
        int stage = 0;

        // Prefetch tile 0
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_in_tile = threadIdx.x + v * THREADS;
            int64_t global_vec = my_vec_start + vec_in_tile;
            if (global_vec < num_vecs)
                cp_async_vec(tile_buf[0] + vec_in_tile * VEC_BYTES,
                             base + vec_in_tile * VEC_BF16);
        }
        cp_async_commit();

        for (int t = 0; t < num_tiles; t++) {
            int next_stage = 1 - stage;

            // Prefetch next tile
            if (t + 1 < num_tiles) {
                int64_t next_tile_vec = (int64_t)(t + 1) * TILE_VECS;
                #pragma unroll
                for (int v = 0; v < VECS_PER_THREAD; v++) {
                    int vec_in_tile = threadIdx.x + v * THREADS;
                    int64_t global_vec = my_vec_start + next_tile_vec + vec_in_tile;
                    if (global_vec < num_vecs)
                        cp_async_vec(tile_buf[next_stage] + vec_in_tile * VEC_BYTES,
                                     base + (next_tile_vec + vec_in_tile) * VEC_BF16);
                }
                cp_async_commit();
            }

            // Wait for current tile
            if (t + 1 < num_tiles) cp_async_wait_prior();
            else                   cp_async_wait_all();
            __syncthreads();

            // Process current tile
            int64_t tile_vec_start = (int64_t)t * TILE_VECS;
            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD; v++) {
                int vec_in_tile = threadIdx.x + v * THREADS;
                int64_t global_vec = my_vec_start + tile_vec_start + vec_in_tile;
                if (global_vec < num_vecs) {
                    // Determine split for this vector
                    int64_t elem_idx = global_vec * VEC_BF16;
                    int split_id = 0;
                    #pragma unroll
                    for (int s = 1; s < MAX_SPLITS; ++s) {
                        if (s < num_splits && elem_idx >= s_offsets[s]) split_id = s;
                    }

                    float val = reduce_vec8_absmax(tile_buf[stage] + vec_in_tile * VEC_BYTES);
                    local_max[split_id] = fmaxf(local_max[split_id], val);
                }
            }

            __syncthreads();
            stage = next_stage;
        }
    }

    // ── Block reduction per split ──
    for (int s = 0; s < num_splits && s < MAX_SPLITS; ++s) {
        float val = warp_reduce_max(local_max[s]);
        int warp_id = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        if (lane == 0) warp_smem[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            val = (lane < WARPS) ? warp_smem[lane] : 0.0f;
            #pragma unroll
            for (int mask = WARPS / 2; mask > 0; mask >>= 1)
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            if (lane == 0)
                atomic_max_float(&amaxes[s], val);
        }
        __syncthreads();
    }
}


// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Grouped amax dim=1 (per-column-group) — tiled + double-buffered
// ═══════════════════════════════════════════════════════════════════

__launch_bounds__(THREADS)
__global__ void grouped_amax_dim1_pipelined_kernel(
    const __nv_bfloat16* __restrict__ input,
    float*  __restrict__ amaxes,
    const int* __restrict__ col_split_range,  // [num_groups+1]
    int num_groups,
    int64_t total_elems,
    int cols  // N_total
) {
    extern __shared__ char smem_raw[];
    char* tile_buf[NUM_BUFS];
    tile_buf[0] = smem_raw;
    tile_buf[1] = smem_raw + TILE_BYTES;
    float* warp_smem = reinterpret_cast<float*>(smem_raw + 2 * TILE_BYTES);

    __shared__ int s_range[MAX_GROUPS + 1];
    if (threadIdx.x < num_groups + 1 && threadIdx.x < MAX_GROUPS + 1)
        s_range[threadIdx.x] = col_split_range[threadIdx.x];
    __syncthreads();

    float local_max[MAX_GROUPS];
    #pragma unroll
    for (int g = 0; g < MAX_GROUPS; ++g) local_max[g] = 0.0f;

    // Determine this block's chunk
    const int64_t num_vecs = total_elems / VEC_BF16;
    const int64_t vecs_per_block = (num_vecs + gridDim.x - 1) / gridDim.x;
    const int64_t aligned_vpb = ((vecs_per_block + TILE_VECS - 1) / TILE_VECS) * TILE_VECS;

    const int64_t my_vec_start = (int64_t)blockIdx.x * aligned_vpb;
    const int64_t my_vec_end   = min(my_vec_start + aligned_vpb, num_vecs);
    const int     num_tiles    = (int)((my_vec_end - my_vec_start + TILE_VECS - 1) / TILE_VECS);

    if (num_tiles > 0) {
        const __nv_bfloat16* base = input + my_vec_start * VEC_BF16;
        int stage = 0;

        // Prefetch tile 0
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_in_tile = threadIdx.x + v * THREADS;
            int64_t global_vec = my_vec_start + vec_in_tile;
            if (global_vec < num_vecs)
                cp_async_vec(tile_buf[0] + vec_in_tile * VEC_BYTES,
                             base + vec_in_tile * VEC_BF16);
        }
        cp_async_commit();

        for (int t = 0; t < num_tiles; t++) {
            int next_stage = 1 - stage;

            if (t + 1 < num_tiles) {
                int64_t next_tile_vec = (int64_t)(t + 1) * TILE_VECS;
                #pragma unroll
                for (int v = 0; v < VECS_PER_THREAD; v++) {
                    int vec_in_tile = threadIdx.x + v * THREADS;
                    int64_t global_vec = my_vec_start + next_tile_vec + vec_in_tile;
                    if (global_vec < num_vecs)
                        cp_async_vec(tile_buf[next_stage] + vec_in_tile * VEC_BYTES,
                                     base + (next_tile_vec + vec_in_tile) * VEC_BF16);
                }
                cp_async_commit();
            }

            if (t + 1 < num_tiles) cp_async_wait_prior();
            else                   cp_async_wait_all();
            __syncthreads();

            int64_t tile_vec_start = (int64_t)t * TILE_VECS;
            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD; v++) {
                int vec_in_tile = threadIdx.x + v * THREADS;
                int64_t global_vec = my_vec_start + tile_vec_start + vec_in_tile;
                if (global_vec < num_vecs) {
                    // Column group determination
                    int64_t elem_idx = global_vec * VEC_BF16;
                    int col = (int)(elem_idx % cols);
                    int group_id = 0;
                    #pragma unroll
                    for (int g = 1; g < MAX_GROUPS; ++g) {
                        if (g < num_groups && col >= s_range[g]) group_id = g;
                    }

                    float val = reduce_vec8_absmax(tile_buf[stage] + vec_in_tile * VEC_BYTES);
                    local_max[group_id] = fmaxf(local_max[group_id], val);
                }
            }

            __syncthreads();
            stage = next_stage;
        }
    }

    // Block reduction per group
    for (int g = 0; g < num_groups && g < MAX_GROUPS; ++g) {
        float val = warp_reduce_max(local_max[g]);
        int warp_id = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        if (lane == 0) warp_smem[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            val = (lane < WARPS) ? warp_smem[lane] : 0.0f;
            for (int mask = WARPS / 2; mask > 0; mask >>= 1)
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            if (lane == 0)
                atomic_max_float(&amaxes[g], val);
        }
        __syncthreads();
    }
}


// ─────────────────────── Launch helpers ───────────────────────

// Required dynamic shared memory: 2 tile buffers + warp reduction scratch
inline int smem_size() {
    return SMEM_BYTES + WARPS * sizeof(float) + 64;  // +64 for alignment
}

// Optimal grid size for a given element count
inline int grid_size(int64_t n) {
    int num_vecs = (int)(n / VEC_BF16);
    // Enough blocks that each has ≥1 tile, but not more than hardware can run
    int blocks_by_data = (num_vecs + TILE_VECS - 1) / TILE_VECS;
    // Cap at reasonable limit — GB200 has 160 SMs, ~4 blocks/SM is good
    return min(blocks_by_data, 640);
}

} // namespace pipelined_amax

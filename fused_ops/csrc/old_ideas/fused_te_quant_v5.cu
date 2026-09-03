// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// V5b: Fused RMSNorm + Activation + NVFP4 Quantization
//      2D Tiled TMA (TILE_M rows × TILE_K cols) + Double-Buffered Pipeline
//
// Key insight: Use proper 2D tiles (multiple rows) to amortize TMA overhead.
// Each block processes a CHUNK_M × K region:
//   - CHUNK_M rows
//   - Tiles across K with TILE_K columns per tile
//   - Double-buffered: compute on tile[n] while loading tile[n+1]
//
// Tile dimensions:
//   TILE_M = 32, TILE_K = 128 → 32×128×2 = 8KB per buffer, 16KB for double-buf
//   CHUNK_M = TILE_M = 32 (rows per block)
//   Each block processes 32 rows × full K columns
//
// Within each block:
//   - 256 threads spread across 32 rows and K columns
//   - For RMS reduction: warp-level reduction within each row's thread set
//
// Architecture:
//   Pass 1: [2D TMA tiles → shmem] → compute RMS stats per row + block amax
//   Pass 2: [2D TMA tiles → shmem] → normalize + activate + quantize

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>
#include <cub/cub.cuh>

#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;

__device__ __forceinline__ float bf16_to_f32(nv_bfloat16 v) {
    return __bfloat162float(v);
}

constexpr int BLOCK_GROUP_SIZE = 16;      // NVFP4 quantization block
constexpr int TILE_M = 32;               // Rows per TMA tile
constexpr int TILE_K = 128;              // Cols per TMA tile (128 bf16 × 2B = 256B)
constexpr int CHUNK_M = TILE_M;          // Rows processed per block
constexpr int NUM_BUFFS = 2;             // Double buffering
constexpr size_t TMA_ALIGN = 128;        // TMA shared memory alignment
constexpr int BLOCK_SIZE_V5 = 256;       // Threads per block

// =========================================================================
// PTX Helpers
// =========================================================================

__device__ __forceinline__ void ptx_mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t p = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(p), "r"(count) : "memory");
}

__device__ __forceinline__ void ptx_mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx) {
    uint32_t p = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(p), "r"(tx) : "memory");
}

__device__ __forceinline__ void ptx_mbarrier_arrive(uint64_t* mbar) {
    uint32_t p = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(p) : "memory");
}

__device__ __forceinline__ void ptx_mbarrier_invalid(uint64_t* mbar) {
    uint32_t p = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(p) : "memory");
}

__device__ __forceinline__ bool ptx_mbarrier_try_wait_parity(uint64_t* mbar, uint32_t parity) {
    uint32_t p = __cvta_generic_to_shared(mbar);
    uint32_t done;
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P;\n"
        "}" : "=r"(done) : "r"(p), "r"(parity) : "memory");
    return done != 0;
}

__device__ __forceinline__ void ptx_mbarrier_wait_parity(uint64_t* mbar, uint32_t parity) {
    while (!ptx_mbarrier_try_wait_parity(mbar, parity)) {}
}

__device__ __forceinline__ void ptx_cp_async_bulk_tensor_2d(
    void* dst, const void* tmap, uint32_t x, uint32_t y, uint64_t* mbar
) {
    uint32_t d = __cvta_generic_to_shared(dst);
    uint32_t m = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(d), "l"(tmap), "r"(x), "r"(y), "r"(m) : "memory");
}

__device__ __forceinline__ void ptx_cp_async_wait_group_read_1() {
    asm volatile("cp.async.bulk.wait_group.read 1;");
}

__device__ __forceinline__ void ptx_fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// =========================================================================
// Activations
// =========================================================================

__device__ __forceinline__ float act_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float act_gelu(float x) {
    constexpr float k = 0.7978845608f, c = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
}

template<int ACT_MODE>
__device__ __forceinline__ float apply_activation(float x) {
    if constexpr (ACT_MODE == 0) return act_silu(x);
    else if constexpr (ACT_MODE == 1) return act_gelu(x);
    else return x;
}

// =========================================================================
// PTX Fused mul+cvt
// =========================================================================

struct fp4x4_packed { uint16_t bits; };

__device__ __forceinline__ fp4x4_packed mul_cvt_fp32_to_fp4_4x(
    const float2 in01, const float2 in23, const float2 scale
) {
    uint32_t out_4x = 0;
    asm volatile(
        "{\n"
        ".reg.b64 v01; .reg.b64 v23;\n\t"
        ".reg.b32 v0; .reg.b32 v1; .reg.b32 v2; .reg.b32 v3;\n\t"
        ".reg.b8 f0; .reg.b8 f1;\n\t"
        "mov.b64 {v0, v1}, %1;\n\t"
        "mov.b64 {v2, v3}, %2;\n\t"
        "mov.b64 v01, {v0, v1};\n\t"
        "mov.b64 v23, {v2, v3};\n\t"
        "mul.f32x2 v01, v01, %3;\n\t"
        "mul.f32x2 v23, v23, %3;\n\t"
        "mov.b64 {v1, v0}, v01;\n\t"
        "mov.b64 {v3, v2}, v23;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
        "mov.b32 %0, {f0, f1, f0, f1};\n\t"
        "}" : "=r"(out_4x)
        : "l"(reinterpret_cast<const uint64_t&>(in01)),
          "l"(reinterpret_cast<const uint64_t&>(in23)),
          "l"(reinterpret_cast<const uint64_t&>(scale)));
    return {static_cast<uint16_t>(out_4x & 0xFFFF)};
}

// =========================================================================
// Scale Computation
// =========================================================================

__device__ __forceinline__ float compute_global_scale_decode(float ga) {
    return (ga == 0.0f) ? 1.0f : ga / (448.0f * 6.0f);
}
__device__ __forceinline__ float compute_global_scale_encode(float ga) {
    if (ga == 0.0f) return 1.0f;
    float s = 448.0f * 6.0f / ga;
    return (s == 0.0f) ? 1.0f : fminf(s, 3.4e38f);
}
__device__ __forceinline__ __nv_fp8_e4m3 compute_block_scale_decode(float ba, float gs) {
    return static_cast<__nv_fp8_e4m3>(fminf(ba / (6.0f * gs), 448.0f));
}
__device__ __forceinline__ __nv_fp8_e4m3 compute_block_mult_encode(float ba, float se) {
    if (ba <= 1e-9f) return static_cast<__nv_fp8_e4m3>(448.0f);
    return static_cast<__nv_fp8_e4m3>(fminf(6.0f / (ba * se), 448.0f));
}

// =========================================================================
// Warp-level reduction for per-row stats
// =========================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, mask));
    return val;
}

// =========================================================================
// Pass 1: 2D TMA tiled + double-buffered → stats + block amax
//
// Thread layout within block:
//   256 threads / 32 rows = 8 threads per row
//   Each thread processes TILE_K/8 = 16 elements per tile per row
//   But TILE_K=128, so 128/8 = 16 elements per thread per tile
//   = 1 quantization block (16 elements) per thread per tile
// =========================================================================

template<int ACT_MODE = 0>
__global__ void __launch_bounds__(BLOCK_SIZE_V5)
fused_v5b_pass1(
    const __grid_constant__ CUtensorMap tensor_map_x,
    const nv_bfloat16* __restrict__ w_ptr,
    float epsilon,
    int rows, int cols,
    float* __restrict__ block_amax_scratch,
    float* __restrict__ inv_rms_cache,
    unsigned int* __restrict__ global_amax_bits
) {
    extern __shared__ char dyn_smem[];
    
    int tid = threadIdx.x;
    bool is_leader = (tid == 0);
    
    // Block handles rows [row_base, row_base + CHUNK_M)
    int row_base = blockIdx.x * CHUNK_M;
    int num_tiles_k = cols / TILE_K;
    int quant_blocks_per_tile = TILE_K / BLOCK_GROUP_SIZE;  // 128/16 = 8
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    uint32_t tile_bytes = TILE_M * TILE_K * sizeof(nv_bfloat16);  // 32×128×2 = 8KB
    
    // Thread assignment: each thread handles specific (row_offset, qb) pairs within a tile
    // 256 threads, 32 rows × 8 qb = 256 → perfect mapping
    int row_in_tile = tid / quant_blocks_per_tile;    // 0..31
    int qb_in_tile  = tid % quant_blocks_per_tile;    // 0..7
    int global_row  = row_base + row_in_tile;
    
    // Shared memory layout
    uintptr_t base = (reinterpret_cast<uintptr_t>(dyn_smem) + TMA_ALIGN - 1) & ~(TMA_ALIGN - 1);
    size_t buf_bytes = TILE_M * TILE_K * sizeof(nv_bfloat16);  // 8KB
    size_t buf_aligned = (buf_bytes + TMA_ALIGN - 1) & ~(TMA_ALIGN - 1);
    
    nv_bfloat16* buf0 = reinterpret_cast<nv_bfloat16*>(base);
    nv_bfloat16* buf1 = reinterpret_cast<nv_bfloat16*>(base + buf_aligned);
    nv_bfloat16* buffers[2] = {buf0, buf1};
    uint64_t* mbar = reinterpret_cast<uint64_t*>(base + 2 * buf_aligned);
    
    // Per-row partial reduction storage in shmem
    // 32 rows × sizeof(float) for sum_sq and amax
    float* row_stats = reinterpret_cast<float*>(
        reinterpret_cast<uintptr_t>(mbar) + NUM_BUFFS * sizeof(uint64_t) + 8
    );
    float* row_amaxf = row_stats + CHUNK_M;
    
    // Init barriers with full thread count (TE pattern)
    if (is_leader) {
        for (int i = 0; i < NUM_BUFFS; ++i)
            ptx_mbarrier_init(&mbar[i], BLOCK_SIZE_V5);
        ptx_fence_proxy_async_shared();
    }
    if (tid < CHUNK_M) {
        row_stats[tid] = 0.0f;
        row_amaxf[tid] = 0.0f;
    }
    __syncthreads();
    
    // Prefetch first tile: leader issues TMA + arrive_expect_tx, others arrive
    if (is_leader) {
        ptx_cp_async_bulk_tensor_2d(buf0, &tensor_map_x, 0, row_base, &mbar[0]);
        ptx_mbarrier_arrive_expect_tx(&mbar[0], tile_bytes);
    } else {
        ptx_mbarrier_arrive(&mbar[0]);
    }
    
    float my_sumsq = 0.0f;
    float my_amax = 0.0f;
    
    for (int tile = 0; tile < num_tiles_k; ++tile) {
        int b = tile % NUM_BUFFS;
        int next_tile = tile + 1;
        
        // Prefetch next tile: all threads must participate in barrier
        if (next_tile < num_tiles_k) {
            int nb = next_tile % NUM_BUFFS;
            if (next_tile >= NUM_BUFFS)
                ptx_cp_async_wait_group_read_1();
            if (is_leader) {
                ptx_cp_async_bulk_tensor_2d(buffers[nb], &tensor_map_x,
                    next_tile * TILE_K, row_base, &mbar[nb]);
                ptx_mbarrier_arrive_expect_tx(&mbar[nb], tile_bytes);
            } else {
                ptx_mbarrier_arrive(&mbar[nb]);
            }
        }
        
        ptx_fence_proxy_async_shared();
        ptx_mbarrier_wait_parity(&mbar[b], tile / NUM_BUFFS);
        
        // Each thread reads its 16 elements: buf[row_in_tile * TILE_K + qb * 16 .. +15]
        if (global_row < rows) {
            int local_off = row_in_tile * TILE_K + qb_in_tile * BLOCK_GROUP_SIZE;
            int global_col = tile * TILE_K + qb_in_tile * BLOCK_GROUP_SIZE;
            
            bf16x8 d0 = bf16x8::load(buffers[b] + local_off);
            bf16x8 d1 = bf16x8::load(buffers[b] + local_off + 8);
            bf16x8 w0 = bf16x8::load(w_ptr + global_col);
            bf16x8 w1 = bf16x8::load(w_ptr + global_col + 8);
            
            float block_max = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v = bf16_to_f32(d0[k]);
                float wv = bf16_to_f32(w0[k]);
                my_sumsq += v * v;
                block_max = fmaxf(block_max, fabsf(apply_activation<ACT_MODE>(v) * wv));
            }
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v = bf16_to_f32(d1[k]);
                float wv = bf16_to_f32(w1[k]);
                my_sumsq += v * v;
                block_max = fmaxf(block_max, fabsf(apply_activation<ACT_MODE>(v) * wv));
            }
            
            // Store block amax
            int global_qb = global_col / BLOCK_GROUP_SIZE;
            block_amax_scratch[global_row * num_blocks_per_row + global_qb] = block_max;
            my_amax = fmaxf(my_amax, block_max);
        }
    }
    
    // Destroy barriers
    if (is_leader) {
        for (int i = 0; i < NUM_BUFFS; ++i)
            ptx_mbarrier_invalid(&mbar[i]);
    }
    
    // --- Per-row reduction ---
    // Threads for the same row need to reduce their sumsq and amax
    // row_in_tile identifies which row; threads with same row_in_tile share a row
    // But with 256 threads and 32 rows, 8 threads per row
    // These 8 threads are tid=row*8+0 .. row*8+7, which is NOT a warp
    // Use atomicAdd to shared memory
    
    if (global_row < rows) {
        atomicAdd(&row_stats[row_in_tile], my_sumsq);
        // atomicMax for float via uint32_t
        unsigned int* amax_ptr_u = reinterpret_cast<unsigned int*>(&row_amaxf[row_in_tile]);
        atomicMax(amax_ptr_u, __float_as_uint(my_amax));
    }
    __syncthreads();
    
    // Compute inv_rms for each row (one thread per row)
    if (tid < CHUNK_M && (row_base + tid) < rows) {
        float sqs = row_stats[tid];
        float inv_rms = rsqrtf(sqs / cols + epsilon);
        inv_rms_cache[row_base + tid] = inv_rms;
        
        // Scale block amaxes and find row amax
        float ra = 0.0f;
        for (int qb = 0; qb < num_blocks_per_row; ++qb) {
            float s = block_amax_scratch[(row_base + tid) * num_blocks_per_row + qb] * inv_rms;
            block_amax_scratch[(row_base + tid) * num_blocks_per_row + qb] = s;
            ra = fmaxf(ra, s);
        }
        if (ra > 0.0f)
            atomicMax(global_amax_bits, __float_as_uint(ra));
    }
}

// =========================================================================
// Global scale
// =========================================================================

__global__ void compute_global_scale_v5b(
    const unsigned int* amax_bits, float* gs_ptr, int enc
) {
    float a = __uint_as_float(*amax_bits);
    if (a == 0.0f) a = 1.0f;
    *gs_ptr = enc ? compute_global_scale_encode(a) : compute_global_scale_decode(a);
}

// =========================================================================
// Pass 2: 2D TMA tiled + double-buffered → normalize + quantize
// =========================================================================

template<int ACT_MODE = 0, int SCALE_MODE = 0>
__global__ void __launch_bounds__(BLOCK_SIZE_V5)
fused_v5b_pass2(
    const __grid_constant__ CUtensorMap tensor_map_x,
    const nv_bfloat16* __restrict__ w_ptr,
    int rows, int cols,
    const float* __restrict__ block_amax_scratch,
    const float* __restrict__ inv_rms_cache,
    const float* __restrict__ global_scale_ptr,
    unsigned char* __restrict__ y_ptr,
    __nv_fp8_e4m3* __restrict__ scale_ptr
) {
    extern __shared__ char dyn_smem[];
    
    int tid = threadIdx.x;
    bool is_leader = (tid == 0);
    int row_base = blockIdx.x * CHUNK_M;
    int num_tiles_k = cols / TILE_K;
    int quant_blocks_per_tile = TILE_K / BLOCK_GROUP_SIZE;
    int num_blocks_per_row = cols / BLOCK_GROUP_SIZE;
    uint32_t tile_bytes = TILE_M * TILE_K * sizeof(nv_bfloat16);
    
    int row_in_tile = tid / quant_blocks_per_tile;
    int qb_in_tile  = tid % quant_blocks_per_tile;
    int global_row  = row_base + row_in_tile;
    
    uintptr_t base = (reinterpret_cast<uintptr_t>(dyn_smem) + TMA_ALIGN - 1) & ~(TMA_ALIGN - 1);
    size_t buf_aligned = (TILE_M * TILE_K * sizeof(nv_bfloat16) + TMA_ALIGN - 1) & ~(TMA_ALIGN - 1);
    nv_bfloat16* buffers[2];
    buffers[0] = reinterpret_cast<nv_bfloat16*>(base);
    buffers[1] = reinterpret_cast<nv_bfloat16*>(base + buf_aligned);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(base + 2 * buf_aligned);
    
    if (is_leader) {
        for (int i = 0; i < NUM_BUFFS; ++i) ptx_mbarrier_init(&mbar[i], BLOCK_SIZE_V5);
        ptx_fence_proxy_async_shared();
    }
    __syncthreads();
    
    float inv_rms = (global_row < rows) ? inv_rms_cache[global_row] : 1.0f;
    float gs = *global_scale_ptr;
    
    // Prefetch first tile: leader issues TMA, all threads arrive
    if (is_leader) {
        ptx_cp_async_bulk_tensor_2d(buffers[0], &tensor_map_x, 0, row_base, &mbar[0]);
        ptx_mbarrier_arrive_expect_tx(&mbar[0], tile_bytes);
    } else {
        ptx_mbarrier_arrive(&mbar[0]);
    }
    
    for (int tile = 0; tile < num_tiles_k; ++tile) {
        int b = tile % NUM_BUFFS;
        int next_tile = tile + 1;
        
        if (next_tile < num_tiles_k) {
            int nb = next_tile % NUM_BUFFS;
            if (next_tile >= NUM_BUFFS) ptx_cp_async_wait_group_read_1();
            if (is_leader) {
                ptx_cp_async_bulk_tensor_2d(buffers[nb], &tensor_map_x,
                    next_tile * TILE_K, row_base, &mbar[nb]);
                ptx_mbarrier_arrive_expect_tx(&mbar[nb], tile_bytes);
            } else {
                ptx_mbarrier_arrive(&mbar[nb]);
            }
        }
        
        ptx_fence_proxy_async_shared();
        ptx_mbarrier_wait_parity(&mbar[b], tile / NUM_BUFFS);
        
        if (global_row < rows) {
            int local_off = row_in_tile * TILE_K + qb_in_tile * BLOCK_GROUP_SIZE;
            int global_col = tile * TILE_K + qb_in_tile * BLOCK_GROUP_SIZE;
            
            bf16x8 d0 = bf16x8::load(buffers[b] + local_off);
            bf16x8 d1 = bf16x8::load(buffers[b] + local_off + 8);
            bf16x8 w0 = bf16x8::load(w_ptr + global_col);
            bf16x8 w1 = bf16x8::load(w_ptr + global_col + 8);
            
            float vals[16];
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                vals[k] = apply_activation<ACT_MODE>(bf16_to_f32(d0[k])) * bf16_to_f32(w0[k]) * inv_rms;
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                vals[8+k] = apply_activation<ACT_MODE>(bf16_to_f32(d1[k])) * bf16_to_f32(w1[k]) * inv_rms;
            
            int global_qb = global_col / BLOCK_GROUP_SIZE;
            float ba = block_amax_scratch[global_row * num_blocks_per_row + global_qb];
            float bsi; __nv_fp8_e4m3 ss;
            
            if constexpr (SCALE_MODE == 0) {
                ss = compute_block_scale_decode(ba, gs);
                float sf = float(ss);
                bsi = 1.0f / (fmaxf(sf, 1e-12f) * gs);
            } else {
                auto m = compute_block_mult_encode(ba, gs);
                float mf = float(m);
                bsi = fmaxf(mf, 1e-12f) * gs;
                ss = static_cast<__nv_fp8_e4m3>(1.0f / fmaxf(mf, 1e-12f));
            }
            
            float2 sc2 = {bsi, bsi};
            fp4x4_packed q0 = mul_cvt_fp32_to_fp4_4x({vals[0],vals[1]}, {vals[2],vals[3]}, sc2);
            fp4x4_packed q1 = mul_cvt_fp32_to_fp4_4x({vals[4],vals[5]}, {vals[6],vals[7]}, sc2);
            fp4x4_packed q2 = mul_cvt_fp32_to_fp4_4x({vals[8],vals[9]}, {vals[10],vals[11]}, sc2);
            fp4x4_packed q3 = mul_cvt_fp32_to_fp4_4x({vals[12],vals[13]}, {vals[14],vals[15]}, sc2);
            
            int ge = global_row * cols + global_col;
            uint16_t* o = reinterpret_cast<uint16_t*>(y_ptr + ge / 2);
            o[0] = q0.bits; o[1] = q1.bits; o[2] = q2.bits; o[3] = q3.bits;
            
            scale_ptr[global_row * num_blocks_per_row + global_qb] = ss;
        }
    }
    
    if (is_leader) {
        for (int i = 0; i < NUM_BUFFS; ++i) ptx_mbarrier_invalid(&mbar[i]);
    }
}

// =========================================================================
// Host: Create 2D tensor map
// =========================================================================

static void create_tensor_map_2d_bf16(
    CUtensorMap& tmap, const void* ptr, uint64_t rows, uint64_t cols
) {
    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank] = {cols, rows};
    uint64_t globalStride[rank - 1] = {cols * 2};
    uint32_t boxDim[rank] = {(uint32_t)TILE_K, (uint32_t)TILE_M};
    uint32_t elemStride[rank] = {1, 1};
    
    CUresult res = cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
        const_cast<void*>(ptr), globalDim, globalStride,
        boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    );
    if (res != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(res, &err_str);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") + err_str);
    }
}

// =========================================================================
// Host Launcher
// =========================================================================

void launch_fused_te_quant_v5(
    const nv_bfloat16* x, const nv_bfloat16* w,
    float epsilon, int rows, int cols,
    int act_mode, int scale_mode,
    unsigned char* y, __nv_fp8_e4m3* scales,
    float* global_scale, float* inv_rms_cache,
    float* block_amax_scratch
) {
    alignas(64) CUtensorMap tmap;
    create_tensor_map_2d_bf16(tmap, x, rows, cols);
    
    int num_blocks = (rows + CHUNK_M - 1) / CHUNK_M;
    
    // shmem: 2 × (TILE_M × TILE_K × 2) + mbarriers + row stats
    size_t buf_aligned = (TILE_M * TILE_K * sizeof(nv_bfloat16) + TMA_ALIGN - 1) & ~(TMA_ALIGN - 1);
    size_t shmem_size = 2 * buf_aligned + NUM_BUFFS * 8 + 8 + CHUNK_M * 2 * sizeof(float) + TMA_ALIGN;
    
    unsigned int* ga_bits;
    cudaMallocAsync(&ga_bits, sizeof(unsigned int), 0);
    cudaMemsetAsync(ga_bits, 0, sizeof(unsigned int), 0);
    
    #define DISPATCH_V5B_P1(AM) do { \
        auto k = fused_v5b_pass1<AM>; \
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size); \
        k<<<num_blocks, BLOCK_SIZE_V5, shmem_size>>>(tmap, w, epsilon, rows, cols, \
            block_amax_scratch, inv_rms_cache, ga_bits); \
    } while(0)
    
    switch (act_mode) {
        case 0: DISPATCH_V5B_P1(0); break;
        case 1: DISPATCH_V5B_P1(1); break;
        case 2: DISPATCH_V5B_P1(2); break;
    }
    #undef DISPATCH_V5B_P1
    
    compute_global_scale_v5b<<<1, 1>>>(ga_bits, global_scale, scale_mode);
    
    #define DISPATCH_V5B_P2(AM, SM) do { \
        auto k = fused_v5b_pass2<AM, SM>; \
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size); \
        k<<<num_blocks, BLOCK_SIZE_V5, shmem_size>>>(tmap, w, rows, cols, \
            block_amax_scratch, inv_rms_cache, global_scale, y, scales); \
    } while(0)
    
    switch (act_mode * 2 + scale_mode) {
        case 0: DISPATCH_V5B_P2(0, 0); break;
        case 1: DISPATCH_V5B_P2(0, 1); break;
        case 2: DISPATCH_V5B_P2(1, 0); break;
        case 3: DISPATCH_V5B_P2(1, 1); break;
        case 4: DISPATCH_V5B_P2(2, 0); break;
        case 5: DISPATCH_V5B_P2(2, 1); break;
    }
    #undef DISPATCH_V5B_P2
    
    CUDA_CHECK(cudaGetLastError());
    cudaFreeAsync(ga_bits, 0);
}

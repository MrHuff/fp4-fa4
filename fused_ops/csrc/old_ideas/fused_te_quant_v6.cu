// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// V6: Fused RMSNorm + Activation + NVFP4 Quantization — Single-Pass
//
// Uses a grid barrier (atomic sense-reverse) to avoid the 2nd kernel launch.
// Everything is done in 1 kernel: stats + amax → barrier → scale → quantize.
//
// Grid size = min(M, max_occupancy). Each block handles ceil(M/grid) rows.
//
// Memory traffic:
//   - If M ≤ max_occupancy: each block has 1 row, data stays in shmem
//     across barrier → 1× HBM read + 1× write (optimal!)
//   - If M > max_occupancy: persistent kernel, Phase B reloads from HBM
//     → 2× HBM read + 1× write (same as V1, but saves kernel launch overhead)

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cub/cub.cuh>

#include "vec.cuh"
#include "utils.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;

__device__ __forceinline__ float bf16_to_f32(nv_bfloat16 v) {
    return __bfloat162float(v);
}

constexpr int BG = 16;   // quantization block
constexpr int BS = 256;  // threads

// =========================================================================
// Grid barrier
// =========================================================================
__device__ void grid_barrier(unsigned int* ctr, unsigned int* sense, int n) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int s = *sense;
        if (atomicAdd(ctr, 1) == (unsigned int)(n - 1)) {
            *ctr = 0; __threadfence();
            atomicExch(sense, 1 - s);
        } else {
            while (atomicAdd(sense, 0) == s) {}
        }
    }
    __syncthreads();
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
template<int A> __device__ __forceinline__ float act(float x) {
    if constexpr (A==0) return act_silu(x);
    else if constexpr (A==1) return act_gelu(x);
    else return x;
}

// =========================================================================
// Block reductions
// =========================================================================
__device__ __forceinline__ float bsum(float v) {
    typedef cub::BlockReduce<float, BS> B; __shared__ typename B::TempStorage t;
    return B(t).Sum(v);
}
__device__ __forceinline__ float bmax(float v) {
    typedef cub::BlockReduce<float, BS> B; __shared__ typename B::TempStorage t;
    struct M { __device__ float operator()(float a, float b) const { return fmaxf(a,b); } };
    return B(t).Reduce(v, M());
}

// =========================================================================
// PTX fused mul+cvt
// =========================================================================
struct fp4x4_packed { uint16_t bits; };
__device__ __forceinline__ fp4x4_packed fp4_cvt(float2 a, float2 b, float2 s) {
    uint32_t o=0;
    asm volatile("{\n" ".reg.b64 v01;.reg.b64 v23;\n\t"
        ".reg.b32 v0;.reg.b32 v1;.reg.b32 v2;.reg.b32 v3;\n\t"
        ".reg.b8 f0;.reg.b8 f1;\n\t"
        "mov.b64 {v0,v1},%1;\n\t" "mov.b64 {v2,v3},%2;\n\t"
        "mov.b64 v01,{v0,v1};\n\t" "mov.b64 v23,{v2,v3};\n\t"
        "mul.f32x2 v01,v01,%3;\n\t" "mul.f32x2 v23,v23,%3;\n\t"
        "mov.b64 {v1,v0},v01;\n\t" "mov.b64 {v3,v2},v23;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0,v0,v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1,v2,v3;\n\t"
        "mov.b32 %0,{f0,f1,f0,f1};\n\t" "}" :"=r"(o)
        :"l"(reinterpret_cast<const uint64_t&>(a)),
         "l"(reinterpret_cast<const uint64_t&>(b)),
         "l"(reinterpret_cast<const uint64_t&>(s)));
    return {(uint16_t)(o&0xFFFF)};
}

// =========================================================================
// Scale helpers
// =========================================================================
__device__ __forceinline__ float gs_dec(float a) { return a==0?1:a/(448*6.0f); }
__device__ __forceinline__ float gs_enc(float a) { if(a==0)return 1; float s=448*6.0f/a; return s==0?1:fminf(s,3.4e38f); }
__device__ __forceinline__ __nv_fp8_e4m3 bs_dec(float b, float g) { return (__nv_fp8_e4m3)fminf(b/(6*g),448.f); }
__device__ __forceinline__ __nv_fp8_e4m3 bs_enc(float b, float g) { return b<=1e-9f?(__nv_fp8_e4m3)448.f:(__nv_fp8_e4m3)fminf(6/(b*g),448.f); }

// =========================================================================
// Process one row from shmem: Phase A (stats+amax)
// =========================================================================
template<int AM>
__device__ void process_row_phase_a(
    nv_bfloat16* shmem, const nv_bfloat16* w, int cols,
    int row, int nbpr,
    float* amax_scratch, float* irms_cache,
    unsigned int* ga_bits, float epsilon
) {
    int tid = threadIdx.x;
    
    // sum_sq
    float ss = 0;
    for (int i = tid*8; i < cols; i += BS*8) {
        bf16x8 d = bf16x8::load(shmem + i);
        #pragma unroll
        for (int k=0;k<8;++k) { float v=bf16_to_f32(d[k]); ss+=v*v; }
    }
    float rs = bsum(ss);
    __shared__ float s_ir;
    if (tid==0) { s_ir = rsqrtf(rs/cols+epsilon); irms_cache[row]=s_ir; }
    __syncthreads();
    float ir = s_ir;

    // block amax
    float ra = 0;
    for (int qb = tid; qb < nbpr; qb += BS) {
        int b = qb*BG;
        bf16x8 d0=bf16x8::load(shmem+b), d1=bf16x8::load(shmem+b+8);
        bf16x8 w0=bf16x8::load(w+b), w1=bf16x8::load(w+b+8);
        float bm = 0;
        #pragma unroll
        for (int k=0;k<8;++k) { float v=act<AM>(bf16_to_f32(d0[k])*ir)*bf16_to_f32(w0[k]); bm=fmaxf(bm,fabsf(v)); }
        #pragma unroll
        for (int k=0;k<8;++k) { float v=act<AM>(bf16_to_f32(d1[k])*ir)*bf16_to_f32(w1[k]); bm=fmaxf(bm,fabsf(v)); }
        amax_scratch[row*nbpr+qb] = bm;
        ra = fmaxf(ra, bm);
    }
    float rmax = bmax(ra);
    if (tid==0 && rmax>0) atomicMax(ga_bits, __float_as_uint(rmax));
    __syncthreads();
}

// =========================================================================
// Process one row from shmem: Phase B (quantize)
// =========================================================================
template<int AM, int SM>
__device__ void process_row_phase_b(
    nv_bfloat16* shmem, const nv_bfloat16* w, int cols,
    int row, int nbpr, float ir, float gs,
    float* amax_scratch,
    unsigned char* y_ptr, __nv_fp8_e4m3* sc_ptr
) {
    int tid = threadIdx.x;
    for (int qb = tid; qb < nbpr; qb += BS) {
        int b = qb*BG;
        bf16x8 d0=bf16x8::load(shmem+b), d1=bf16x8::load(shmem+b+8);
        bf16x8 w0=bf16x8::load(w+b), w1=bf16x8::load(w+b+8);
        float v[16];
        #pragma unroll
        for (int k=0;k<8;++k) v[k]=act<AM>(bf16_to_f32(d0[k])*ir)*bf16_to_f32(w0[k]);
        #pragma unroll
        for (int k=0;k<8;++k) v[8+k]=act<AM>(bf16_to_f32(d1[k])*ir)*bf16_to_f32(w1[k]);
        
        float ba = amax_scratch[row*nbpr+qb];
        float bsi; __nv_fp8_e4m3 ss;
        if constexpr (SM==0) {
            ss=bs_dec(ba,gs); bsi=1.f/(fmaxf((float)ss,1e-12f)*gs);
        } else {
            auto m=bs_enc(ba,gs); float mf=(float)m; bsi=fmaxf(mf,1e-12f)*gs;
            ss=(__nv_fp8_e4m3)(1.f/fmaxf(mf,1e-12f));
        }
        float2 s2={bsi,bsi};
        auto q0=fp4_cvt({v[0],v[1]},{v[2],v[3]},s2);
        auto q1=fp4_cvt({v[4],v[5]},{v[6],v[7]},s2);
        auto q2=fp4_cvt({v[8],v[9]},{v[10],v[11]},s2);
        auto q3=fp4_cvt({v[12],v[13]},{v[14],v[15]},s2);
        int ge=row*cols+b;
        uint16_t* o=reinterpret_cast<uint16_t*>(y_ptr+ge/2);
        o[0]=q0.bits; o[1]=q1.bits; o[2]=q2.bits; o[3]=q3.bits;
        sc_ptr[row*nbpr+qb]=ss;
    }
}

// =========================================================================
// V6 kernel
// =========================================================================
template<int AM=0, int SM=0>
__global__ void __launch_bounds__(BS)
fused_v6(
    const nv_bfloat16* __restrict__ x, const nv_bfloat16* __restrict__ w,
    float eps, int rows, int cols,
    unsigned char* __restrict__ y, __nv_fp8_e4m3* __restrict__ sc,
    float* __restrict__ gs_ptr, float* __restrict__ ir_cache,
    float* __restrict__ am_scratch, unsigned int* __restrict__ ga_bits,
    unsigned int* __restrict__ bc, unsigned int* __restrict__ bs_ptr,
    int grid_sz
) {
    extern __shared__ char smem[];
    nv_bfloat16* srow = reinterpret_cast<nv_bfloat16*>(smem);
    int tid = threadIdx.x;
    int nbpr = cols / BG;
    bool single_row_per_block = (rows <= grid_sz);

    // ============ PHASE A: Load + stats + amax for each assigned row ============
    for (int row = blockIdx.x; row < rows; row += grid_sz) {
        // Load HBM → shmem
        const nv_bfloat16* rp = x + (int64_t)row * cols;
        for (int i = tid*8; i < cols; i += BS*8) {
            bf16x8 c = bf16x8::load(rp + i);
            c.store(srow + i);
        }
        __syncthreads();
        
        process_row_phase_a<AM>(srow, w, cols, row, nbpr, am_scratch, ir_cache, ga_bits, eps);
    }

    // ============ Grid barrier ============
    grid_barrier(bc, bs_ptr, grid_sz);

    // ============ Compute global scale ============
    __shared__ float s_gs;
    if (tid == 0) {
        float a = __uint_as_float(*ga_bits);
        if (a == 0) a = 1;
        s_gs = (SM == 0) ? gs_dec(a) : gs_enc(a);
        if (blockIdx.x == 0) *gs_ptr = s_gs;
    }
    __syncthreads();
    float gscale = s_gs;

    // ============ PHASE B: Quantize ============
    if (single_row_per_block) {
        // Data is STILL in shmem! No HBM re-read needed!
        int row = blockIdx.x;
        if (row < rows) {
            float ir = ir_cache[row];
            process_row_phase_b<AM, SM>(srow, w, cols, row, nbpr, ir, gscale, am_scratch, y, sc);
        }
    } else {
        // Persistent case: must reload from HBM
        for (int row = blockIdx.x; row < rows; row += grid_sz) {
            const nv_bfloat16* rp = x + (int64_t)row * cols;
            for (int i = tid*8; i < cols; i += BS*8) {
                bf16x8 c = bf16x8::load(rp + i);
                c.store(srow + i);
            }
            __syncthreads();
            float ir = ir_cache[row];
            process_row_phase_b<AM, SM>(srow, w, cols, row, nbpr, ir, gscale, am_scratch, y, sc);
            __syncthreads();
        }
    }
}

// =========================================================================
// Host
// =========================================================================
void launch_fused_te_quant_v6(
    const nv_bfloat16* x, const nv_bfloat16* w,
    float eps, int rows, int cols,
    int am, int sm,
    unsigned char* y, __nv_fp8_e4m3* sc,
    float* gs, float* ir, float* ams
) {
    size_t shmem = cols * sizeof(nv_bfloat16);
    
    int mbpsm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mbpsm, fused_v6<0,0>, BS, shmem);
    int dev; cudaGetDevice(&dev);
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    int max_blocks = mbpsm * nsm;
    int grid = min(rows, max_blocks);
    if (grid <= 0) grid = 1;

    unsigned int *bc, *bsn, *gab;
    cudaMallocAsync(&bc, 4, 0); cudaMallocAsync(&bsn, 4, 0); cudaMallocAsync(&gab, 4, 0);
    cudaMemsetAsync(bc, 0, 4, 0); cudaMemsetAsync(bsn, 0, 4, 0); cudaMemsetAsync(gab, 0, 4, 0);

    #define L(A,S) do { auto k=fused_v6<A,S>; \
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem); \
        k<<<grid, BS, shmem>>>(x,w,eps,rows,cols,y,sc,gs,ir,ams,gab,bc,bsn,grid); } while(0)
    switch(am*2+sm) {
        case 0: L(0,0); break; case 1: L(0,1); break;
        case 2: L(1,0); break; case 3: L(1,1); break;
        case 4: L(2,0); break; case 5: L(2,1); break;
    }
    #undef L
    CUDA_CHECK(cudaGetLastError());
    cudaFreeAsync(bc, 0); cudaFreeAsync(bsn, 0); cudaFreeAsync(gab, 0);
}

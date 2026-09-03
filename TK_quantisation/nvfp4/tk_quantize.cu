/*
 * TK-Optimized NVFP4 Quantisation — Standalone PyBind11 Entrypoint
 *
 * Ports TE's custom NVFP4 quantise+transpose kernels to a standalone
 * module that outputs TK-native format (swizzled scales, fp4x2 data).
 *
 * Current template instantiations:
 *   - <false, false, true>  : no stochastic rounding, no fast_math, with transpose
 *   - <false, false, false> : no stochastic rounding, no fast_math, no transpose
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <dlfcn.h>

// Exclude TE host-side launcher functions from kernel headers
#define TK_STANDALONE

// TK Quantisation kernels (standalone headers)
#include "core.cuh"
#include "quantize_transpose_tuned.cuh"
#include "group_quantize_transpose.cuh"
#include "group_quantize_transpose_dim_1.cuh"

using namespace transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel;
namespace grp_kernel = transformer_engine::dispatch::nvfp4::group_quantize_transpose_kernel;
namespace grp_dim1_kernel = transformer_engine::dispatch::nvfp4::group_quantize_transpose_dim1_kernel;
using namespace transformer_engine;

// ─────────────────────── TMA tensor map creation (standalone) ────────────────
static void create_tma_2d(
    CUtensorMap &map,
    void *ptr,
    uint64_t globalY,
    uint64_t globalX,
    uint32_t shmemY,
    uint32_t shmemX,
    uint64_t strideX,
    size_t type_num_bits
) {
    // Use the CUDA driver API to resolve cuTensorMapEncodeTiled at runtime
    // This avoids link-time dependency on libcuda.so
    typedef CUresult (*cuTensorMapEncodeTiled_t)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
        const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
        const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

    static cuTensorMapEncodeTiled_t fn = nullptr;
    if (!fn) {
        void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
        TORCH_CHECK(handle != nullptr, "Failed to open libcuda.so.1");
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(dlsym(handle, "cuTensorMapEncodeTiled"));
        TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled not found in libcuda.so.1");
    }

    CUtensorMapDataType dataType;
    // Dimensions are always in elements (matching TE convention).
    // For 4-bit types, the TMA hardware handles packing natively.
    uint64_t globalDims[2] = {globalX, globalY};
    uint32_t boxDims[2] = {shmemX, shmemY};
    // Stride is in bytes: stride_elements * bits_per_element / 8
    uint64_t globalStrides[1] = {(strideX * type_num_bits) / 8};
    uint32_t elementStrides[2] = {1, 1};

    if (type_num_bits == 16) {
        dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if (type_num_bits == 4) {
        // Native 4-bit TMA data type — 16 x 4-bit elements in 8-byte aligned groups
        // Dimensions remain in elements (NOT halved), TMA hardware packs natively
        dataType = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    } else {
        TORCH_CHECK(false, "Unsupported type_num_bits: ", type_num_bits);
    }

    auto result = fn(
        &map, dataType, 2, ptr,
        globalDims, globalStrides, boxDims, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled failed: ", result);
}

// ─── Launch helper for a specific template config ───
template <bool SR, bool FM, bool RT>
static void launch_kernel(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    nvfp4_scale_t *scales_ptr,
    nvfp4_scale_t *scales_t_ptr,
    const float *amax_row_ptr,
    const float *amax_col_ptr,
    int64_t M, int64_t K,
    int64_t scale_stride,
    int64_t scale_stride_t,
    cudaStream_t stream
) {
    const int blocks_Y = (M + TunableConfig::CHUNK_DIM_Y - 1) / TunableConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + TunableConfig::CHUNK_DIM_X - 1) / TunableConfig::CHUNK_DIM_X;
    const dim3 grid(blocks_X, blocks_Y);

    constexpr int buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
    constexpr int buff_elems_total_in = BUFFS_NUM_IN * buff_elems;
    constexpr int bsz_in  = ((buff_elems_total_in * (int)sizeof(bf16) + 127) / 128) * 128;
    constexpr int bsz_out = ((BUFFS_NUM_OUT * BUFF_OUT_SIZE + 127) / 128) * 128;
    constexpr int bsz_out_t = RT ? (((BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE + 127) / 128) * 128) : 0;
    constexpr int bsz_sc  = ((TunableConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t) + 127) / 128) * 128;
    constexpr int bsz_sc_t = RT ? (((TunableConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t) + 127) / 128) * 128) : 0;
    constexpr int dshmem = bsz_in + bsz_out + bsz_out_t + bsz_sc + bsz_sc_t + 128;

    auto kernel = quantize_transpose_nvfp4_tuned_1D_kernel<SR, FM, RT>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    kernel<<<grid, THREADS_NUM, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        scales_ptr, scales_t_ptr,
        nullptr,        // noop
        amax_row_ptr, amax_col_ptr,
        M, K,
        scale_stride, scale_stride_t,
         nullptr,        // rng_state
        true            // always swizzle_scales
    );
}

// ─── Launch helper for grouped quantize+transpose kernel ───
// Launches group_quantize_transpose_nvfp4_kernel with pre-built
// MultiAmaxCastTransposeFusionArgs. This replaces 3× separate kernel
// launches + copy with a single grouped kernel call.
template <bool SR, bool RT>
static void launch_group_kernel(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    nvfp4_scale_t *scales_ptr,
    int64_t rows, int64_t cols,
    int64_t scale_stride,
    grp_kernel::MultiAmaxCastTransposeFusionArgs &kernel_args,
    cudaStream_t stream
) {
    const int blocks_Y = (rows + grp_kernel::CHUNK_DIM_Y - 1) / grp_kernel::CHUNK_DIM_Y;
    const int blocks_X = (cols + grp_kernel::CHUNK_DIM_X - 1) / grp_kernel::CHUNK_DIM_X;
    const dim3 grid(blocks_X, blocks_Y);

    constexpr size_t buff_elems = grp_kernel::BUFF_DIM_Y * grp_kernel::BUFF_DIM_X;
    constexpr size_t buff_elems_total = grp_kernel::BUFFS_NUM * buff_elems;
    constexpr size_t bsz_in  = ((buff_elems_total * sizeof(bf16) + 127) / 128) * 128;
    constexpr size_t bsz_out = (((buff_elems_total * 4) / 8 + 127) / 128) * 128;
    constexpr size_t bsz_out_t = RT ? bsz_out : 0;
    constexpr size_t bsz_sc_t = RT ? (((grp_kernel::CHUNK_DIM_Y * grp_kernel::CHUNK_DIM_X) / 16 * sizeof(nvfp4_scale_t) + 127) / 128) * 128 : 0;
    constexpr size_t dshmem = bsz_in + bsz_out + bsz_out_t + bsz_sc_t + 128;

    using Empty = transformer_engine::Empty;
    auto kernel = grp_kernel::group_quantize_transpose_nvfp4_kernel<false, Empty, nullptr, bf16, SR, RT>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<grid, grp_kernel::THREADS_NUM, dshmem, stream>>>(
        tmap_in, tmap_out,
        scales_ptr,
        nullptr,  // noop
        rows, cols,
        scale_stride,
        nullptr,  // rng_state
        kernel_args
    );
}

// ────────────────── Fused amax reduction kernel ──────────────────
// Computes max(|x|) over all bf16 elements, stores amax + sg = amax/2688
__global__ void fused_amax_bf16_kernel(const __nv_bfloat16 *__restrict__ input,
                                        float *__restrict__ amax_out,
                                        float *__restrict__ sg_out,
                                        int64_t n) {
    float local_max = 0.0f;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float val = fabsf(__bfloat162float(input[i]));
        local_max = fmaxf(local_max, val);
    }
    // Block reduction
    __shared__ float sdata[256];
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    // Global atomicMax via int cast (float is positive so int comparison works)
    if (threadIdx.x == 0) {
        unsigned int *amax_uint = reinterpret_cast<unsigned int*>(amax_out);
        unsigned int old = *amax_uint;
        unsigned int val = __float_as_uint(sdata[0]);
        while (val > old) {
            old = atomicCAS(amax_uint, old, val);
        }
    }
    // Barrier — last block writes sg
    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        static __device__ unsigned int block_count = 0;
        unsigned int prev = atomicAdd(&block_count, 1);
        is_last = (prev == gridDim.x - 1);
        if (is_last) block_count = 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        float amax = *amax_out;
        *sg_out = amax / 2688.0f;  // sg = amax / (fp8_max * fp4_max)
    }
}


// ────────────────────── Main entrypoint (raw) ──────────────────────
// amax_row and amax_col are SCALAR (1-element) float tensors on GPU.
// The kernel reads *amax_ptr as a single global amax to compute
// S_enc = fp8_max * fp4_max / amax.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_quantize_transpose(
    torch::Tensor input,
    torch::Tensor amax_row,
    torch::Tensor amax_col,
    bool return_transpose
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 2);
    TORCH_CHECK(amax_row.is_cuda() && amax_row.scalar_type() == torch::kFloat32,
                "amax_row must be a CUDA float32 tensor");
    TORCH_CHECK(amax_row.numel() == 1, "amax_row must be a scalar (1-element) tensor, got ",
                amax_row.numel());

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 32 == 0 && K % 32 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    // Scale layout: TK expects swizzled 128×4-tile format
    // For TK, we use: ntm = M/128, ntk = K/64, each tile = 512 bytes
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;

    // Allocate FP4 data outputs
    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    // Scale stride: aligned to 4 (TE convention for flat layout used by kernel internally)
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
    // Allocate scales in flat layout — kernel writes swizzled directly
    auto row_sc = torch::empty({M, scale_stride}, opts_u8);

    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_u8) : torch::empty({0}, opts_u8);
    auto col_sc  = return_transpose ? torch::empty({K, scale_stride_t}, opts_u8) : torch::empty({0}, opts_u8);

    // Create TMA tensor maps
    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
    if (return_transpose) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);
    }

    nvfp4_scale_t *sc_ptr = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    // amax_row and amax_col are SCALAR pointers — kernel reads *ptr
    const float *amax_r = reinterpret_cast<const float*>(amax_row.data_ptr());
    const float *amax_c = (amax_col.numel() > 0)
        ? reinterpret_cast<const float*>(amax_col.data_ptr())
        : amax_r;  // If no col amax, reuse row amax (same as TE convention)

    // Dispatch (only 2 variants for now — add more as needed)
    if (return_transpose) {
        launch_kernel<false, false, true>(
            tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr,
            amax_r, amax_c, M, K, scale_stride, scale_stride_t, stream);
    } else {
        launch_kernel<false, false, false>(
            tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr,
            amax_r, amax_c, M, K, scale_stride, scale_stride_t, stream);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_transpose failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


// ────────────────────── GEMM-ready entrypoint ──────────────────────
// Returns tensors in the exact format TK nvfp4_gemm expects:
//   fp4:  (M, K/2) fp4x2 dtype
//   sc:   (ntm, ntk, 512) fp8_e4m3 dtype (3D swizzled tile format)
//   sg:   scalar float32 = amax / 2688.0 (global scale)
// Computes amax internally (no need for caller to provide).
std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_quantize_for_gemm(torch::Tensor input, bool return_transpose) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 2);

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "For GEMM-ready output, M and K must be multiples of 128");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    // Tile dimensions for TK GEMM
    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;

    // Allocate amax + sg buffer (2 floats: [amax, sg])
    auto amax_buf = torch::zeros({2}, torch::dtype(torch::kFloat32).device(device));
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;

    // Step 1: Compute amax + sg via fused kernel
    int64_t n = M * K;
    int threads = 256;
    int blocks = std::min((int)((n + threads - 1) / threads), 1024);
    fused_amax_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        amax_ptr, sg_ptr, n);

    // Step 2: Allocate outputs — FP4 data in fp4x2 dtype, scales in 3D swizzled fp8
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);

    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4)   : torch::empty({0}, opts_fp4);
    auto col_sc  = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8) : torch::empty({0}, opts_fp8);

    // Scale stride for the kernel (flat view of the 3D swizzled buffer)
    // ntm * ntk * 512 = M/128 * K/64 * 512 = M * K/16384 * 512 = M * K/32
    // But kernel expects scale_stride as number of scale elements per row = K/16 (aligned to 4)
    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    // Create TMA tensor maps
    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
    if (return_transpose) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);
    }

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    // Step 3: Quantize
    if (return_transpose) {
        launch_kernel<false, false, true>(
            tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr,
            amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
    } else {
        launch_kernel<false, false, false>(
            tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr,
            amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_for_gemm failed: ", cudaGetErrorString(err));

    // sg as a 1-element tensor
    auto sg = amax_buf.narrow(0, 1, 1);

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, sg, sg);
}

// ────────────────── Grouped amax kernel (per-split) ──────────────────
// Same algorithm as TE's grouped_amax_bf16_kernel but self-contained.
// Computes amaxes[split_id] = max(|input[split_start..split_end]|) for all splits.
constexpr int GROUP_AMAX_THREADS = 256;
constexpr int GROUP_AMAX_MAX_SPLITS = 8;

__launch_bounds__(GROUP_AMAX_THREADS)
__global__ void grouped_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    float*  __restrict__ amaxes,
    const int64_t* __restrict__ split_elem_offsets,
    int num_splits,
    int64_t total_elems
) {
    // Load offsets into shared memory
    __shared__ int64_t s_offsets[GROUP_AMAX_MAX_SPLITS + 1];
    if (threadIdx.x < num_splits + 1 && threadIdx.x < GROUP_AMAX_MAX_SPLITS + 1) {
        s_offsets[threadIdx.x] = split_elem_offsets[threadIdx.x];
    }
    __syncthreads();

    float local_max[GROUP_AMAX_MAX_SPLITS];
    #pragma unroll
    for (int s = 0; s < GROUP_AMAX_MAX_SPLITS; ++s) local_max[s] = 0.0f;

    // Process 8 bf16 per iteration (16 bytes = 2 × int4)
    const int64_t nvec = 8;
    const int64_t num_vec = total_elems / nvec;
    const int64_t vec_stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_vec; i += vec_stride) {
        int64_t elem_idx = i * nvec;
        int split_id = 0;
        #pragma unroll
        for (int s = 1; s < GROUP_AMAX_MAX_SPLITS; ++s) {
            if (s < num_splits && elem_idx >= s_offsets[s]) split_id = s;
        }

        // Load 8 bf16
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(input + elem_idx);
        __nv_bfloat162 v0 = __habs2(p[0]);
        __nv_bfloat162 v1 = __habs2(p[1]);
        __nv_bfloat162 v2 = __habs2(p[2]);
        __nv_bfloat162 v3 = __habs2(p[3]);
        __nv_bfloat162 m = __hmax2(__hmax2(v0, v1), __hmax2(v2, v3));
        float val = __bfloat162float(__hmax(__high2bfloat16(m), __low2bfloat16(m)));
        local_max[split_id] = fmaxf(local_max[split_id], val);
    }

    // Block reduction via shared memory
    __shared__ float warp_maxes[GROUP_AMAX_MAX_SPLITS][GROUP_AMAX_THREADS / 32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    for (int s = 0; s < num_splits && s < GROUP_AMAX_MAX_SPLITS; ++s) {
        float val = local_max[s];
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
        if (lane == 0) warp_maxes[s][warp_id] = val;
    }
    __syncthreads();

    // Final reduction in warp 0
    if (warp_id == 0) {
        for (int s = 0; s < num_splits && s < GROUP_AMAX_MAX_SPLITS; ++s) {
            float val = (lane < (GROUP_AMAX_THREADS / 32)) ? warp_maxes[s][lane] : 0.0f;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
            }
            if (lane == 0) {
                // atomicMax for positive floats
                unsigned int* p = reinterpret_cast<unsigned int*>(&amaxes[s]);
                unsigned int old = *p;
                unsigned int want = __float_as_uint(val);
                while (want > old) { old = atomicCAS(p, old, want); }
            }
        }
    }
}

// Simple kernel: compute sg[i] = amax[i] / 2688.0 for each split
__global__ void compute_sg_kernel(const float* __restrict__ amaxes,
                                   float* __restrict__ sgs,
                                   int num_splits) {
    int i = threadIdx.x;
    if (i < num_splits) {
        sgs[i] = amaxes[i] / 2688.0f;
    }
}

// Broadcast sg to per-tile b_sg array
// For forward: tiles_per_split[i] = M_i / Nb
// For dgrad:   tiles are K / Nb, repeated N times
__global__ void broadcast_sg_to_b_sg(
    const float* __restrict__ sgs,
    float* __restrict__ b_sg,
    const int* __restrict__ tiles_per_split,
    const int* __restrict__ tile_offsets,
    int num_splits
) {
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    // Find which split this tile belongs to
    int split_id = 0;
    for (int s = 1; s < num_splits; ++s) {
        if (tile >= tile_offsets[s]) split_id = s;
    }
    if (tile < tile_offsets[num_splits]) {
        b_sg[tile] = sgs[split_id];
    }
}


// ────────────────── Grouped quantize for GEMM ──────────────────
// Per-split amax → per-split quantize → concatenated output for grouped GEMM.
// Returns: (fp4_row, sc_row, fwd_b_sg, [fp4_col_i...], [sc_col_i...], dgrad_b_sg, sg_cat, mega_buf)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0);

    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    TORCH_CHECK(N <= GROUP_AMAX_MAX_SPLITS, "Max ", GROUP_AMAX_MAX_SPLITS, " splits");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    constexpr int64_t Nb = 256;  // tile size for b_sg
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    // Validate splits
    int64_t sum_splits = 0;
    int64_t total_ntm_r = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        TORCH_CHECK(split_sections[i] % 128 == 0);
        sum_splits += split_sections[i];
        total_ntm_r += split_sections[i] / 128;
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad_tiles = (int64_t)N * dgrad_tiles_per;

    // ── Allocations ──
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    // Row FP4 output (contiguous for TMA): (total_rows, K/2) uint8
    auto wc_fp4_row_u8 = torch::empty({total_rows, K / 2}, opts_u8);

    // Per-split amax (must be zeroed for atomicMax)
    auto amaxes = torch::zeros({N}, opts_f32);

    // sg, fwd_b_sg, dgrad_b_sg — computed inside grouped kernel
    auto sg_cat    = torch::empty({N}, opts_f32);
    auto fwd_b_sg  = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad_tiles}, opts_f32);

    // Per-split allocations for row scales, col FP4, col scales
    std::vector<torch::Tensor> sc_row_allocs(N);
    std::vector<torch::Tensor> fp4_col_list(N), sc_col_allocs(N);

    // ── Step 1: Grouped amax ──
    {
        std::vector<int64_t> h_offsets(N + 1);
        h_offsets[0] = 0;
        for (int i = 0; i < N; ++i)
            h_offsets[i + 1] = h_offsets[i] + split_sections[i] * K;

        auto d_offsets = torch::empty({N + 1}, torch::dtype(torch::kInt64).device(device));
        cudaMemcpyAsync(d_offsets.data_ptr<int64_t>(), h_offsets.data(),
                        (N + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        int64_t n = total_rows * K;
        int blocks = std::min((int)((n / 8 + GROUP_AMAX_THREADS - 1) / GROUP_AMAX_THREADS), 65535);
        grouped_amax_kernel<<<blocks, GROUP_AMAX_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            amaxes.data_ptr<float>(),
            d_offsets.data_ptr<int64_t>(),
            N, n);
    }

    // ── Step 2: Build kernel args + allocate per-split outputs ──
    grp_kernel::MultiAmaxCastTransposeFusionArgs kernel_args;
    memset(&kernel_args, 0, sizeof(kernel_args));
    kernel_args.num_tensors = N;
    kernel_args.swizzle_scales = true;
    kernel_args.sg_output = sg_cat.data_ptr<float>();
    kernel_args.fwd_b_sg = fwd_b_sg.data_ptr<float>();
    kernel_args.dgrad_b_sg = dgrad_b_sg.data_ptr<float>();
    kernel_args.b_tile_size = (int)Nb;
    kernel_args.total_cols = (int)K;
    kernel_args.split_sections_range[0] = 0;

    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        kernel_args.split_sections_range[i + 1] = kernel_args.split_sections_range[i] + (int)M_i;

        // Amax ptr (same for row + col per split)
        kernel_args.rowwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
        kernel_args.colwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);

        // Row scales: per-split, kernel writes with swizzle
        sc_row_allocs[i] = torch::empty({M_i, scale_stride}, opts_u8);
        kernel_args.output_rowwise_scale_inv_list[i] = sc_row_allocs[i].data_ptr();

        // Col FP4: (K, M_i/2) uint8
        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        kernel_args.output_colwise_data_list[i] = fp4_col_list[i].data_ptr();

        // Col scales
        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        sc_col_allocs[i] = torch::zeros({K, c_sc_stride}, opts_u8);
        kernel_args.output_colwise_scale_inv_list[i] = sc_col_allocs[i].data_ptr();
        kernel_args.output_colwise_scale_stride[i] = (int)c_sc_stride;
    }

    // ── Step 3: Create TMA maps & launch grouped kernel ──
    alignas(64) CUtensorMap tmap_in{};
    alignas(64) CUtensorMap tmap_out{};

    create_tma_2d(tmap_in, input.data_ptr(),
                  total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X,
                  K, 16);  // bf16 = 16 bits
    create_tma_2d(tmap_out, wc_fp4_row_u8.data_ptr(),
                  total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X,
                  K, 4);   // fp4 = 4 bits

    // scales_ptr is the first split's scale ptr (kernel uses per-split ptrs from kernel_args)
    nvfp4_scale_t* scales_ptr = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[0].data_ptr());

    launch_group_kernel<false, true>(
        tmap_in, tmap_out,
        scales_ptr,
        total_rows, K, scale_stride,
        kernel_args, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_for_gemm grouped kernel failed: ",
                cudaGetErrorString(err));

    // ── Step 4: Reshape outputs to TK-native format ──
    auto wc_fp4_row = wc_fp4_row_u8.view(torch::kFloat4_e2m1fn_x2);

    // Concatenate per-split row scales → (total_ntm_r, ntk_r, 512) fp8
    std::vector<torch::Tensor> sc_row_parts(N);
    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        const int64_t ntm_i = M_i / 128;
        sc_row_parts[i] = sc_row_allocs[i].reshape({ntm_i, ntk_r, 512});
    }
    auto wc_sc_row = torch::cat(sc_row_parts, 0).view(torch::kFloat8_e4m3fn);

    // Reshape col outputs
    std::vector<torch::Tensor> sc_col_list(N);
    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        const int64_t ntk_c_i = M_i / 64;
        fp4_col_list[i] = fp4_col_list[i].view(torch::kFloat4_e2m1fn_x2);
        sc_col_list[i] = sc_col_allocs[i].reshape({ntm_c, ntk_c_i, 512}).view(torch::kFloat8_e4m3fn);
    }

    auto mega_buf = torch::empty({0}, opts_u8);

    return std::make_tuple(wc_fp4_row, wc_sc_row, fwd_b_sg,
                           fp4_col_list, sc_col_list, dgrad_b_sg, sg_cat, mega_buf);
}

// ────────────────── Dim=1 grouped amax kernel ──────────────────
// Computes per-column-group amax for a (M, N_total) bf16 tensor.
// Each column in [col_split_range[g], col_split_range[g+1]) belongs to group g.
// Linearised index i maps to row = i/N_total, col = i%N_total.
constexpr int DIM1_AMAX_THREADS = 256;
constexpr int DIM1_AMAX_MAX_GROUPS = 8;

__launch_bounds__(DIM1_AMAX_THREADS)
__global__ void grouped_amax_dim1_kernel(
    const __nv_bfloat16* __restrict__ input,
    float*  __restrict__ amaxes,
    const int* __restrict__ col_split_range,  // [num_groups+1]
    int num_groups,
    int64_t total_elems,
    int cols  // N_total
) {
    __shared__ int s_range[DIM1_AMAX_MAX_GROUPS + 1];
    if (threadIdx.x < num_groups + 1 && threadIdx.x < DIM1_AMAX_MAX_GROUPS + 1) {
        s_range[threadIdx.x] = col_split_range[threadIdx.x];
    }
    __syncthreads();

    float local_max[DIM1_AMAX_MAX_GROUPS];
    #pragma unroll
    for (int g = 0; g < DIM1_AMAX_MAX_GROUPS; ++g) local_max[g] = 0.0f;

    const int64_t nvec = 8;
    const int64_t num_vec = total_elems / nvec;
    const int64_t vec_stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_vec; i += vec_stride) {
        int64_t elem_idx = i * nvec;
        // Determine column group from the first element's column
        int col = (int)(elem_idx % cols);
        int group_id = 0;
        #pragma unroll
        for (int g = 1; g < DIM1_AMAX_MAX_GROUPS; ++g) {
            if (g < num_groups && col >= s_range[g]) group_id = g;
        }

        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(input + elem_idx);
        __nv_bfloat162 v0 = __habs2(p[0]); __nv_bfloat162 v1 = __habs2(p[1]);
        __nv_bfloat162 v2 = __habs2(p[2]); __nv_bfloat162 v3 = __habs2(p[3]);
        __nv_bfloat162 m = __hmax2(__hmax2(v0, v1), __hmax2(v2, v3));
        float val = __bfloat162float(__hmax(__high2bfloat16(m), __low2bfloat16(m)));
        local_max[group_id] = fmaxf(local_max[group_id], val);
    }

    // Block reduction
    __shared__ float warp_maxes[DIM1_AMAX_MAX_GROUPS][DIM1_AMAX_THREADS / 32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    for (int g = 0; g < num_groups && g < DIM1_AMAX_MAX_GROUPS; ++g) {
        float val = local_max[g];
        for (int mask = 16; mask > 0; mask >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
        if (lane == 0) warp_maxes[g][warp_id] = val;
    }
    __syncthreads();
    for (int g = 0; g < num_groups && g < DIM1_AMAX_MAX_GROUPS; ++g) {
        float val = (lane < DIM1_AMAX_THREADS / 32) ? warp_maxes[g][lane] : 0.0f;
        for (int mask = (DIM1_AMAX_THREADS / 32) / 2; mask > 0; mask >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
        if (lane == 0) {
            unsigned int* p = reinterpret_cast<unsigned int*>(&amaxes[g]);
            unsigned int old = *p;
            unsigned int want = __float_as_uint(val);
            while (want > old) { old = atomicCAS(p, old, want); }
        }
    }
}

// ────────────────── Dim=1 grouped quantize launch helper ──────────────────
template <bool USE_STOCHASTIC_ROUNDING, bool RETURN_TRANSPOSE>
static void launch_dim1_group_kernel(
    CUtensorMap &tmap_in, CUtensorMap &tmap_out,
    nvfp4_scale_t *scales_ptr,
    int64_t rows, int64_t cols, int64_t scale_stride,
    grp_dim1_kernel::Dim1GroupArgs &kernel_args,
    cudaStream_t stream) {
    dim3 grid(cols / grp_dim1_kernel::CHUNK_DIM_X,
             rows / grp_dim1_kernel::CHUNK_DIM_Y);

    constexpr size_t in_bytes = grp_dim1_kernel::BUFFS_NUM *
        grp_dim1_kernel::BUFF_DIM_Y * grp_dim1_kernel::BUFF_DIM_X * sizeof(__nv_bfloat16);
    constexpr size_t out_bytes = grp_dim1_kernel::BUFFS_NUM *
        grp_dim1_kernel::BUFF_DIM_Y * ((grp_dim1_kernel::BUFF_DIM_X * 4) / 8);
    constexpr size_t out_t_bytes = RETURN_TRANSPOSE ?
        grp_dim1_kernel::BUFFS_NUM * grp_dim1_kernel::BUFF_OUT_T_SIZE : 0;
    constexpr size_t col_scales_bytes = RETURN_TRANSPOSE ?
        grp_dim1_kernel::CHUNK_DIM_X * grp_dim1_kernel::SCALES_PER_CHUNK_Y : 0;
    constexpr size_t shmem =
        DIVUP_TO_MULTIPLE(in_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(out_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(out_t_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(col_scales_bytes, TMA_SHMEM_ALIGNMENT) +
        TMA_SHMEM_ALIGNMENT;

    auto func = grp_dim1_kernel::group_quantize_transpose_dim1_nvfp4_kernel<
        USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE>;
    cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    func<<<grid, grp_dim1_kernel::THREADS_NUM, shmem, stream>>>(
        tmap_in, tmap_out, scales_ptr, nullptr, rows, cols, scale_stride,
        nullptr, kernel_args);
}


// ────────────────── Dim=1 grouped quantize for GEMM ──────────────────
// Per-column-group amax → per-column-group quantize.
// Returns: ([fp4_row_i], [sc_row_i], sg_per_group, [fp4_col_i], [sc_col_i])
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_group_quantize_dim1_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), N_total = input.size(1);
    TORCH_CHECK(M % 128 == 0 && N_total % 128 == 0);
    const int G = (int)col_split_sections.size();
    TORCH_CHECK(G <= DIM1_AMAX_MAX_GROUPS, "Max ", DIM1_AMAX_MAX_GROUPS, " groups");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    // Validate splits
    int64_t sum_cols = 0;
    for (int i = 0; i < G; ++i) {
        TORCH_CHECK(col_split_sections[i] % 128 == 0,
                    "Split ", i, " = ", col_split_sections[i], " not 128-aligned");
        sum_cols += col_split_sections[i];
    }
    TORCH_CHECK(sum_cols == N_total);

    // ── Step 1: Per-column-group amax ──
    auto amaxes = torch::zeros({G}, opts_f32);
    {
        // Build col_split_range: [0, N_0, N_0+N_1, ...]
        std::vector<int> h_range(G + 1);
        h_range[0] = 0;
        for (int i = 0; i < G; ++i) h_range[i + 1] = h_range[i] + (int)col_split_sections[i];

        auto d_range = torch::empty({G + 1}, torch::dtype(torch::kInt32).device(device));
        cudaMemcpyAsync(d_range.data_ptr<int>(), h_range.data(),
                        (G + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

        int64_t n = M * N_total;
        int blocks = std::min((int)((n / 8 + DIM1_AMAX_THREADS - 1) / DIM1_AMAX_THREADS), 65535);
        grouped_amax_dim1_kernel<<<blocks, DIM1_AMAX_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            amaxes.data_ptr<float>(),
            d_range.data_ptr<int>(),
            G, n, (int)N_total);
    }

    // sg = amax / 2688
    auto sg_per_group = torch::empty({G}, opts_f32);
    compute_sg_kernel<<<1, G, 0, stream>>>(
        amaxes.data_ptr<float>(), sg_per_group.data_ptr<float>(), G);

    // ── Step 2: Build dim=1 kernel args + allocate per-group outputs ──
    grp_dim1_kernel::Dim1GroupArgs kernel_args;
    memset(&kernel_args, 0, sizeof(kernel_args));
    kernel_args.num_groups = G;
    kernel_args.swizzle_scales = true;
    kernel_args.sg_output = sg_per_group.data_ptr<float>();
    kernel_args.col_split_range[0] = 0;

    const int64_t ntm_r = M / 128;     // row tiles for row-quantized output
    const int64_t ntk_c = M / 64;      // col tiles for col-quantized output (rows / 64)

    // Row FP4 output — one contiguous buffer (we'll split it per-group after)
    auto fp4_row_full_u8 = torch::empty({M, N_total / 2}, opts_u8);

    std::vector<torch::Tensor> sc_row_allocs(G);
    std::vector<torch::Tensor> fp4_col_allocs(G), sc_col_allocs(G);

    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        kernel_args.col_split_range[g + 1] = kernel_args.col_split_range[g] + (int)N_g;

        // Amax (same ptr for row + col per group)
        kernel_args.rowwise_amax_list[g] = (void*)(amaxes.data_ptr<float>() + g);
        kernel_args.colwise_amax_list[g] = (void*)(amaxes.data_ptr<float>() + g);

        // Rowwise scales: (M, group_scale_stride) — per-group
        const int64_t group_scale_stride = ((N_g / 16) + 3) / 4 * 4;
        sc_row_allocs[g] = torch::empty({M, group_scale_stride}, opts_u8);
        kernel_args.output_rowwise_scale_inv_list[g] = sc_row_allocs[g].data_ptr();
        kernel_args.output_rowwise_scale_stride[g] = (int)group_scale_stride;

        // Colwise FP4: (N_g, M/2)
        fp4_col_allocs[g] = torch::empty({N_g, M / 2}, opts_u8);
        kernel_args.output_colwise_data_list[g] = fp4_col_allocs[g].data_ptr();

        // Colwise scales: (N_g, col_scale_stride)
        const int64_t col_scale_stride = ((M / 16) + 3) / 4 * 4;
        sc_col_allocs[g] = torch::zeros({N_g, col_scale_stride}, opts_u8);
        kernel_args.output_colwise_scale_inv_list[g] = sc_col_allocs[g].data_ptr();
        kernel_args.output_colwise_scale_stride[g] = (int)col_scale_stride;
    }

    // ── Step 3: TMA maps + launch ──
    alignas(64) CUtensorMap tmap_in{}, tmap_out{};
    create_tma_2d(tmap_in, input.data_ptr(), M, N_total,
                  grp_dim1_kernel::BUFF_DIM_Y, grp_dim1_kernel::BUFF_DIM_X,
                  N_total, 16);  // bf16
    create_tma_2d(tmap_out, fp4_row_full_u8.data_ptr(), M, N_total,
                  grp_dim1_kernel::BUFF_DIM_Y, grp_dim1_kernel::BUFF_DIM_X,
                  N_total, 4);   // fp4

    nvfp4_scale_t* sc_ptr = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[0].data_ptr());
    const int64_t global_scale_stride = ((N_total / 16) + 3) / 4 * 4;

    launch_dim1_group_kernel<false, true>(
        tmap_in, tmap_out, sc_ptr, M, N_total, global_scale_stride,
        kernel_args, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_dim1_for_gemm failed: ",
                cudaGetErrorString(err));

    // ── Step 4: Reshape outputs to TK-native format ──
    // Row FP4: split the contiguous buffer per-group
    std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
    std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        const int64_t ntk_r_g = N_g / 64;
        const int64_t ntm_c_g = N_g / 128;

        // Row FP4: slice from contiguous buffer
        fp4_row_list[g] = fp4_row_full_u8.narrow(1, col_offset / 2, N_g / 2)
                              .contiguous()
                              .view(torch::kFloat4_e2m1fn_x2);

        // Row scales: reshape to (ntm_r, ntk_r_g, 512) fp8
        sc_row_list[g] = sc_row_allocs[g]
                             .reshape({ntm_r, ntk_r_g, 512})
                             .view(torch::kFloat8_e4m3fn);

        // Col FP4: view as fp4x2
        fp4_col_list[g] = fp4_col_allocs[g].view(torch::kFloat4_e2m1fn_x2);

        // Col scales: reshape to (ntm_c_g, ntk_c, 512) fp8
        sc_col_list[g] = sc_col_allocs[g]
                             .reshape({ntm_c_g, ntk_c, 512})
                             .view(torch::kFloat8_e4m3fn);

        col_offset += N_g;
    }

    return std::make_tuple(fp4_row_list, sc_row_list, sg_per_group,
                           fp4_col_list, sc_col_list);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tk_quantize_transpose", &tk_quantize_transpose,
          "TK-optimized NVFP4 quantise+transpose (swizzled scales)\n"
          "amax_row/amax_col must be 1-element float32 CUDA tensors (global amax)",
          py::arg("input"), py::arg("amax_row"), py::arg("amax_col"),
          py::arg("return_transpose") = true);
    m.def("tk_quantize_for_gemm", &tk_quantize_for_gemm,
          "TK-optimized NVFP4 quantise+transpose — GEMM-ready output\n"
          "Returns (row_fp4, row_sc_3d, col_fp4, col_sc_3d, sg, sg)\n"
          "Computes amax internally. fp4 is fp4x2 dtype, sc is (ntm,ntk,512) fp8.",
          py::arg("input"), py::arg("return_transpose") = true);
    m.def("tk_group_quantize_for_gemm", &tk_group_quantize_for_gemm,
          "TK-optimized grouped NVFP4 quantise — per-split amax, GEMM-ready output\n"
          "Returns (fp4_row, sc_row, fwd_b_sg, [fp4_col_i], [sc_col_i], dgrad_b_sg, sg_cat, mega_buf)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_dim1_for_gemm", &tk_group_quantize_dim1_for_gemm,
          "Dim=1 grouped NVFP4 quantise — per-column-group amax\n"
          "Returns ([fp4_row_i], [sc_row_i], sg_per_group, [fp4_col_i], [sc_col_i])",
          py::arg("input"), py::arg("col_split_sections"));
}


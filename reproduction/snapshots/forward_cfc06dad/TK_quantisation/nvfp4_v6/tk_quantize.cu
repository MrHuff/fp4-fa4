/*
 * TK v5 — Hybrid dispatch layer + TMA scale output experiment
 *
 * Three-tier strategy for tk_quantize_for_gemm:
 *   1. grid ≤ max_concurrent → v3 fused single-pass (data stays in SMEM)
 *   2. grid > max_concurrent AND grid < persistent_threshold → v2 two-pass
 *   3. grid ≥ persistent_threshold → v5 persistent kernel (TMA scales)
 *
 * v5 change: persistent kernel uses TMA bulk stores for scales instead of
 * byte-level scattered GMEM writes.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <dlfcn.h>

#define TK_STANDALONE
#include "core.cuh"
#include "quantize_transpose_tuned.cuh"
#include "fused_amax_quantize.cuh"
#include "fused_group_amax_quantize.cuh"
#include "persistent_quantize.cuh"
#include "persistent_group_quantize.cuh"
#include "amax_pipelined.cuh"
#include "fused_norm_quantize.cuh"
#include "persistent_norm_quantize.cuh"
#include "fused_silu_quantize.cuh"
#include "persistent_silu_quantize.cuh"
#include "fused_silu_deriv_quantize.cuh"
#include "persistent_silu_deriv_quantize.cuh"
#include "v6_reconstruct.cuh"
#include "group_quantize_transpose.cuh"
#include "group_quantize_transpose_dim_1.cuh"

namespace grp_kernel = transformer_engine::dispatch::nvfp4::group_quantize_transpose_kernel;
namespace grp_dim1_kernel = transformer_engine::dispatch::nvfp4::group_quantize_transpose_dim1_kernel;

using namespace transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel;
using namespace transformer_engine;


// ─────────────────────── TMA tensor map creation ────────────────
static void create_tma_2d(
    CUtensorMap &map, void *ptr,
    uint64_t globalY, uint64_t globalX,
    uint32_t shmemY, uint32_t shmemX,
    uint64_t strideX, size_t type_num_bits,
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE
) {
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
        TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled not found");
    }

    CUtensorMapDataType dataType;
    uint64_t globalDims[2] = {globalX, globalY};
    uint32_t boxDims[2] = {shmemX, shmemY};
    uint64_t globalStrides[1] = {(strideX * type_num_bits) / 8};
    uint32_t elementStrides[2] = {1, 1};

    if (type_num_bits == 16) dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (type_num_bits == 8) dataType = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else if (type_num_bits == 4) dataType = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    else TORCH_CHECK(false, "Unsupported type_num_bits: ", type_num_bits);

    auto result = fn(&map, dataType, 2, ptr,
        globalDims, globalStrides, boxDims, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        l2promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}


// ═══════════════════════════════════════════════════════════════════
// Vectorized silu(h1)*h3 kernel for strided h13 layout
// Matches TE's fused_silu_mul_strided_amax_kernel: bf162 loads,
// grid-striding loop, capped grid size.
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void silu_strided_kernel(
    const __nv_bfloat16* __restrict__ h13,  // (M, 2H) contiguous
    __nv_bfloat16* __restrict__ out,        // (M, H) contiguous output
    int64_t M, int64_t H) {

    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    // Each thread processes 2 bf16 per iteration via bf162
    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        const __nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const __nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;

        __nv_bfloat162 h1_val = *reinterpret_cast<const __nv_bfloat162*>(h1_ptr);
        __nv_bfloat162 h3_val = *reinterpret_cast<const __nv_bfloat162*>(h3_ptr);

        float2 h1_f = __bfloat1622float2(h1_val);
        h1_f.x = silu_f(h1_f.x);
        h1_f.y = silu_f(h1_f.y);

        float2 h3_f = __bfloat1622float2(h3_val);

        float2 r_f;
        r_f.x = h1_f.x * h3_f.x;
        r_f.y = h1_f.y * h3_f.y;

        *reinterpret_cast<__nv_bfloat162*>(out + elem) = __float22bfloat162_rn(r_f);
    }

    // Handle odd remainder
    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float h1_f = silu_f(__bfloat162float(h13[row * row_stride + col]));
        float h3_f = __bfloat162float(h13[row * row_stride + H + col]);
        out[i] = __float2bfloat16(h1_f * h3_f);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Vectorized silu-deriv dual multiply kernel for backward pass
// Computes: out1 = dh * h3 * silu'(h1)   (gradient w.r.t. h1)
//           out2 = dh * silu(h1)          (gradient w.r.t. h3)
// silu'(x) = sigmoid(x) * (1 + x - silu(x))
// ═══════════════════════════════════════════════════════════════════
__global__ void silu_deriv_dual_strided_kernel(
    const __nv_bfloat16* __restrict__ dh,    // (M, H) contiguous
    const __nv_bfloat16* __restrict__ h13,   // (M, 2H) contiguous
    __nv_bfloat16* __restrict__ out1,        // (M, H) dh * h3 * silu'(h1)
    __nv_bfloat16* __restrict__ out2,        // (M, H) dh * silu(h1)
    int64_t M, int64_t H) {

    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        __nv_bfloat162 dh_val = *reinterpret_cast<const __nv_bfloat162*>(dh + elem);
        float2 dh_f = __bfloat1622float2(dh_val);

        const __nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const __nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;
        float2 h1_f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h1_ptr));
        float2 h3_f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h3_ptr));

        float sigx = 1.0f / (1.0f + __expf(-h1_f.x));
        float sigy = 1.0f / (1.0f + __expf(-h1_f.y));
        float silux = h1_f.x * sigx;
        float siluy = h1_f.y * sigy;
        float silupx = sigx * (1.0f + h1_f.x - silux);
        float silupy = sigy * (1.0f + h1_f.y - siluy);

        float2 c_f = { dh_f.x * h3_f.x * silupx, dh_f.y * h3_f.y * silupy };
        float2 e_f = { dh_f.x * silux, dh_f.y * siluy };

        *reinterpret_cast<__nv_bfloat162*>(out1 + elem) = __float22bfloat162_rn(c_f);
        *reinterpret_cast<__nv_bfloat162*>(out2 + elem) = __float22bfloat162_rn(e_f);
    }

    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float vd = __bfloat162float(dh[i]);
        float v1 = __bfloat162float(h13[row * row_stride + col]);
        float v3 = __bfloat162float(h13[row * row_stride + H + col]);
        float sig = 1.0f / (1.0f + __expf(-v1));
        float silu_v1 = v1 * sig;
        float silup_v1 = sig * (1.0f + v1 - silu_v1);
        out1[i] = __float2bfloat16(vd * v3 * silup_v1);
        out2[i] = __float2bfloat16(vd * silu_v1);
    }
}

// ═══════════════════════════════════════════════════════════════════
// v3 fused launch helper
// ═══════════════════════════════════════════════════════════════════
template <bool RT, bool ENCODE_CENTRIC = true>
static void launch_v3(
    const CUtensorMap &tmap_in, const CUtensorMap &tmap_out, const CUtensorMap &tmap_out_t,
    nvfp4_scale_t *sc_ptr, nvfp4_scale_t *sc_t_ptr,
    float *global_amax, float *sg_out,
    unsigned int *done_counter, unsigned int *ready_flag,
    int64_t M, int64_t K, int64_t scale_stride, int64_t scale_stride_t,
    cudaStream_t stream
) {
    using namespace tk_v3;
    const int blocks_Y = (M + V3Config::CHUNK_DIM_Y - 1) / V3Config::CHUNK_DIM_Y;
    const int blocks_X = (K + V3Config::CHUNK_DIM_X - 1) / V3Config::CHUNK_DIM_X;
    const int total_blocks = blocks_X * blocks_Y;
    const dim3 grid(blocks_X, blocks_Y);
    const int dshmem = v3_shmem_size<RT>();

    auto kernel = fused_amax_quantize_kernel<RT, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<grid, V3_THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        sc_ptr, sc_t_ptr, global_amax, sg_out,
        done_counter, ready_flag,
        M, K, scale_stride, scale_stride_t, total_blocks);
}

// ═══════════════════════════════════════════════════════════════════
// v2 launch helper (pipelined amax + separate quantize)
// ═══════════════════════════════════════════════════════════════════
template <bool SR, bool FM, bool RT, bool ENCODE_CENTRIC = true>
static void launch_v2_kernel(
    const CUtensorMap &tmap_in, const CUtensorMap &tmap_out, const CUtensorMap &tmap_out_t,
    nvfp4_scale_t *sc_ptr, nvfp4_scale_t *sc_t_ptr,
    const float *amax_row, const float *amax_col,
    int64_t M, int64_t K, int64_t scale_stride, int64_t scale_stride_t,
    cudaStream_t stream
) {
    const int blocks_Y = (M + TunableConfig::CHUNK_DIM_Y - 1) / TunableConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + TunableConfig::CHUNK_DIM_X - 1) / TunableConfig::CHUNK_DIM_X;
    const dim3 grid(blocks_X, blocks_Y);

    constexpr int buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
    constexpr int bsz_in  = ((BUFFS_NUM_IN * buff_elems * (int)sizeof(bf16) + 127) / 128) * 128;
    constexpr int bsz_out = ((BUFFS_NUM_OUT * BUFF_OUT_SIZE + 127) / 128) * 128;
    constexpr int bsz_out_t = RT ? (((BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE + 127) / 128) * 128) : 0;
    constexpr int bsz_sc  = ((TunableConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t) + 127) / 128) * 128;
    constexpr int bsz_sc_t = RT ? (((TunableConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t) + 127) / 128) * 128) : 0;
    constexpr int dshmem = bsz_in + bsz_out + bsz_out_t + bsz_sc + bsz_sc_t + 128;

    auto kernel = quantize_transpose_nvfp4_tuned_1D_kernel<SR, FM, RT, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<grid, THREADS_NUM, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr,
        nullptr, amax_row, amax_col, M, K,
        scale_stride, scale_stride_t, nullptr, true);
}


// ═══════════════════════════════════════════════════════════════════
// v2 grouped launch helpers
// ═══════════════════════════════════════════════════════════════════
template <bool SR, bool RT>
static void launch_group_kernel_v2(
    const CUtensorMap &tmap_in, const CUtensorMap &tmap_out,
    nvfp4_scale_t *sc_ptr,
    int64_t rows, int64_t cols, int64_t scale_stride,
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
    // Set shmem attribute only once (first call) — skip during CUDA graph capture
    static bool s_attr_set = false;
    if (!s_attr_set) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        s_attr_set = true;
    }
    kernel<<<grid, grp_kernel::THREADS_NUM, dshmem, stream>>>(
        tmap_in, tmap_out, sc_ptr, nullptr,
        rows, cols, scale_stride, nullptr, kernel_args);
}

template <bool SR, bool RT>
static void launch_dim1_group_kernel_v2(
    CUtensorMap &tmap_in, CUtensorMap &tmap_out,
    nvfp4_scale_t *sc_ptr,
    int64_t rows, int64_t cols, int64_t scale_stride,
    grp_dim1_kernel::Dim1GroupArgs &kernel_args,
    cudaStream_t stream
) {
    dim3 grid(cols / grp_dim1_kernel::CHUNK_DIM_X,
             rows / grp_dim1_kernel::CHUNK_DIM_Y);

    constexpr size_t in_bytes = grp_dim1_kernel::BUFFS_NUM *
        grp_dim1_kernel::BUFF_DIM_Y * grp_dim1_kernel::BUFF_DIM_X * sizeof(__nv_bfloat16);
    constexpr size_t out_bytes = grp_dim1_kernel::BUFFS_NUM *
        grp_dim1_kernel::BUFF_DIM_Y * ((grp_dim1_kernel::BUFF_DIM_X * 4) / 8);
    constexpr size_t out_t_bytes = RT ?
        grp_dim1_kernel::BUFFS_NUM * grp_dim1_kernel::BUFF_OUT_T_SIZE : 0;
    constexpr size_t col_scales_bytes = RT ?
        grp_dim1_kernel::CHUNK_DIM_X * grp_dim1_kernel::SCALES_PER_CHUNK_Y : 0;
    constexpr size_t shmem =
        DIVUP_TO_MULTIPLE(in_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(out_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(out_t_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(col_scales_bytes, TMA_SHMEM_ALIGNMENT) +
        TMA_SHMEM_ALIGNMENT;

    auto func = grp_dim1_kernel::group_quantize_transpose_dim1_nvfp4_kernel<SR, RT>;
    // Set shmem attribute only once (first call) — skip during CUDA graph capture
    static bool s_d1_attr_set = false;
    if (!s_d1_attr_set) {
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        s_d1_attr_set = true;
    }
    func<<<grid, grp_dim1_kernel::THREADS_NUM, shmem, stream>>>(
        tmap_in, tmap_out, sc_ptr, nullptr, rows, cols, scale_stride,
        nullptr, kernel_args);
}

// sg compute helper
__global__ void compute_sg_kernel(const float* __restrict__ amaxes,
                                   float* __restrict__ sgs,
                                   int num) {
    int i = threadIdx.x;
    if (i < num) sgs[i] = amaxes[i] / 2688.0f;
}


// ═══════════════════════════════════════════════════════════════════
// Cached device/occupancy info (initialized once, reused every call)
// ═══════════════════════════════════════════════════════════════════
struct CachedDeviceInfo {
    int num_sms         = 0;
    int v3_max_bps      = 0;  // v3 fused max blocks/SM
    int v3_max_bps_t    = 0;  // v3 fused max blocks/SM (transpose variant)
    int v4_max_bps      = 0;  // v5 persistent max blocks/SM
    int v4_max_bps_t    = 0;  // v5 persistent max blocks/SM (transpose variant)
    int grp_d0_max_bps  = 0;  // grouped dim=0 fused
    int grp_d1_max_bps  = 0;  // grouped dim=1 fused
    int pg_d0_max_bps   = 0;  // persistent grouped dim=0
    int pg_d1_max_bps   = 0;  // persistent grouped dim=1
    int v3_dshmem       = 0;
    int v3_dshmem_t     = 0;
    bool initialized    = false;
};

static CachedDeviceInfo& get_cached_info() {
    static CachedDeviceInfo info;
    if (!info.initialized) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&info.num_sms, cudaDevAttrMultiProcessorCount, dev);

        using namespace tk_v3;
        info.v3_dshmem   = v3_shmem_size<false>();
        info.v3_dshmem_t = v3_shmem_size<true>();

        // v3 fused kernel occupancy
        cudaFuncSetAttribute(fused_amax_quantize_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.v3_max_bps, fused_amax_quantize_kernel<false>, V3_THREADS, info.v3_dshmem);

        cudaFuncSetAttribute(fused_amax_quantize_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.v3_max_bps_t, fused_amax_quantize_kernel<true>, V3_THREADS, info.v3_dshmem_t);

        // v5 persistent kernel occupancy
        cudaFuncSetAttribute(tk_v5::persistent_quantize_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.v4_max_bps, tk_v5::persistent_quantize_kernel<false>, V3_THREADS, info.v3_dshmem);

        cudaFuncSetAttribute(tk_v5::persistent_quantize_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.v4_max_bps_t, tk_v5::persistent_quantize_kernel<true>, V3_THREADS, info.v3_dshmem_t);

        // v2 pipelined amax
        int amax_smem = pipelined_amax::smem_size();
        cudaFuncSetAttribute(pipelined_amax::fused_amax_pipelined_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, amax_smem);

        // Grouped dim=0 fused
        cudaFuncSetAttribute(fused_group_quantize_kernel_dim0<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.grp_d0_max_bps, fused_group_quantize_kernel_dim0<true>, V3_THREADS, info.v3_dshmem_t);

        // Grouped dim=1 fused
        cudaFuncSetAttribute(fused_group_quantize_kernel_dim1<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.grp_d1_max_bps, fused_group_quantize_kernel_dim1<true>, V3_THREADS, info.v3_dshmem_t);

        // Persistent grouped dim=0
        cudaFuncSetAttribute(tk_v5::persistent_group_quantize_kernel_dim0<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.pg_d0_max_bps, tk_v5::persistent_group_quantize_kernel_dim0<true>, V3_THREADS, info.v3_dshmem_t);

        // Persistent grouped dim=1
        cudaFuncSetAttribute(tk_v5::persistent_group_quantize_kernel_dim1<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.pg_d1_max_bps, tk_v5::persistent_group_quantize_kernel_dim1<true>, V3_THREADS, info.v3_dshmem_t);

        // Grouped pipelined amax
        cudaFuncSetAttribute(pipelined_amax::grouped_amax_pipelined_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, amax_smem);
        cudaFuncSetAttribute(pipelined_amax::grouped_amax_dim1_pipelined_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, amax_smem);

        info.initialized = true;
    }
    return info;
}

// Threshold: use persistent kernel when grid >= this many tiles
static constexpr int PERSISTENT_THRESHOLD = 4096;

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm(torch::Tensor input, bool return_transpose, bool encode_centric) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4) : torch::empty({0}, opts_fp4);
    auto col_sc  = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8) : torch::empty({0}, opts_fp8);

    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto amax_buf = torch::empty({2}, opts_f32);
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;

    auto sync_buf = torch::empty({4}, opts_u32);
    unsigned int *sync_data = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    // Grid dimensions
    using namespace tk_v3;
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached_info();
    const int v3_dshmem = return_transpose ? ci.v3_dshmem_t : ci.v3_dshmem;

    if (total_tiles < PERSISTENT_THRESHOLD) {
        // ─── v3 path (fused single-pass if grid fits, v2 fallback otherwise) ───
        const int max_bps = return_transpose ? ci.v3_max_bps_t : ci.v3_max_bps;
        const int max_concurrent = max_bps * ci.num_sms;
        const bool can_fuse = (total_tiles <= max_concurrent && max_bps > 0);

        if (can_fuse) {
            // v3 fused single-pass
            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
            create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
            if (return_transpose)
                create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

            unsigned int *done = sync_data + 2, *ready = sync_data + 3;
            if (encode_centric) {
                if (return_transpose)
                    launch_v3<true, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v3<false, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
            } else {
                if (return_transpose)
                    launch_v3<true, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v3<false, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
            }
        } else {
            // v2 two-pass fallback
            int64_t n = M * K;
            int amax_blocks = pipelined_amax::grid_size(n);
            int amax_smem = pipelined_amax::smem_size();
            pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), amax_ptr, sg_ptr, n);

            alignas(64) CUtensorMap ti{}, to{}, tot{};
            create_tma_2d(ti, input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
            create_tma_2d(to, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
            if (return_transpose)
                create_tma_2d(tot, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

            if (encode_centric) {
                if (return_transpose)
                    launch_v2_kernel<false, false, true, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v2_kernel<false, false, false, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
            } else {
                if (return_transpose)
                    launch_v2_kernel<false, false, true, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v2_kernel<false, false, false, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
            }
        }
    }
    else {
        // ─── v5 persistent (work-stealing, L2 promotion, TMA scale output) ───
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        if (return_transpose)
            create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        // TMA maps for scale tensors: view [ntm, ntk, 512] as 2D using BF16 type
        // Max TMA box = 512 bytes, so use shmemX=256 BF16 (512 bytes) per tile_k block
        // Each chunk does 2 TMA stores (one per tile_k × 512-byte block)
        alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;  // ntk_r*512 bytes / 2 = BF16 elements
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

        if (return_transpose && sc_t_ptr) {
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }

        const int p_max_bps = return_transpose ? ci.v4_max_bps_t : ci.v4_max_bps;
        int num_persistent = p_max_bps * ci.num_sms;
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = sync_data;
        pargs.work_counter_phase2 = sync_data + 1;
        pargs.global_amax  = amax_ptr;
        pargs.done_counter = sync_data + 2;
        pargs.ready_flag   = sync_data + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;
        pargs.col_scales_ptr = sc_t_ptr;
        pargs.col_scale_stride = scale_stride_t;
        pargs.swizzle_scales = true;

        const dim3 grid(num_persistent);
        if (encode_centric) {
            if (return_transpose)
                tk_v5::persistent_quantize_kernel<true, true><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
            else
                tk_v5::persistent_quantize_kernel<false, true><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
        } else {
            if (return_transpose)
                tk_v5::persistent_quantize_kernel<true, false><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
            else
                tk_v5::persistent_quantize_kernel<false, false><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_v5_quantize_for_gemm failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           amax_buf.narrow(0, 1, 1), amax_buf.narrow(0, 1, 1));
}


// ═══════════════════════════════════════════════════════════════════
// 1b. CUDA Graph-safe alloc/launch split for tk_v4_quantize_for_gemm
// ═══════════════════════════════════════════════════════════════════

// _alloc: pre-create all output tensors + sync buffers. Call OUTSIDE graph capture.
// Returns: (row_fp4, row_sc, col_fp4, col_sc, amax_buf, sync_buf)
std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);
    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_u8) : torch::empty({0}, opts_u8);
    auto col_sc  = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_u8) : torch::empty({0}, opts_u8);
    auto amax_buf = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_u32);

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, amax_buf, sync_buf);
}

// _launch: kernel-only dispatch — NO allocations, safe inside CUDA graph capture.
// Returns: (row_fp4_view, row_sc_view, col_fp4_view, col_sc_view, sg, amax)
std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm_launch(
    torch::Tensor input, bool return_transpose, bool encode_centric,
    // Pre-allocated buffers from _alloc:
    torch::Tensor row_fp4, torch::Tensor row_sc,
    torch::Tensor col_fp4, torch::Tensor col_sc,
    torch::Tensor amax_buf, torch::Tensor sync_buf
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;
    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;
    unsigned int *sync_data = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());

    // Graph-safe: cudaMemsetAsync
    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    using namespace tk_v3;
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached_info();
    const int v3_dshmem = return_transpose ? ci.v3_dshmem_t : ci.v3_dshmem;

    if (total_tiles < PERSISTENT_THRESHOLD) {
        const int max_bps = return_transpose ? ci.v3_max_bps_t : ci.v3_max_bps;
        const int max_concurrent = max_bps * ci.num_sms;
        const bool can_fuse = (total_tiles <= max_concurrent && max_bps > 0);

        if (can_fuse) {
            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
            create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
            if (return_transpose)
                create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
            unsigned int *done = sync_data + 2, *ready = sync_data + 3;
            if (encode_centric) {
                if (return_transpose)
                    launch_v3<true, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v3<false, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
            } else {
                if (return_transpose)
                    launch_v3<true, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v3<false, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
            }
        } else {
            int64_t n = M * K;
            int amax_blocks = pipelined_amax::grid_size(n);
            int amax_smem = pipelined_amax::smem_size();
            pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), amax_ptr, sg_ptr, n);
            alignas(64) CUtensorMap ti{}, to{}, tot{};
            create_tma_2d(ti, input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
            create_tma_2d(to, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
            if (return_transpose)
                create_tma_2d(tot, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);
            if (encode_centric) {
                if (return_transpose)
                    launch_v2_kernel<false, false, true, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v2_kernel<false, false, false, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
            } else {
                if (return_transpose)
                    launch_v2_kernel<false, false, true, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v2_kernel<false, false, false, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
            }
        }
    } else {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        if (return_transpose)
            create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        if (return_transpose && sc_t_ptr) {
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }

        const int p_max_bps = return_transpose ? ci.v4_max_bps_t : ci.v4_max_bps;
        int num_persistent = p_max_bps * ci.num_sms;
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = sync_data;
        pargs.work_counter_phase2 = sync_data + 1;
        pargs.global_amax  = amax_ptr;
        pargs.done_counter = sync_data + 2;
        pargs.ready_flag   = sync_data + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;
        pargs.col_scales_ptr = sc_t_ptr;
        pargs.col_scale_stride = scale_stride_t;
        pargs.swizzle_scales = true;

        const dim3 grid(num_persistent);
        if (encode_centric) {
            if (return_transpose)
                tk_v5::persistent_quantize_kernel<true, true><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
            else
                tk_v5::persistent_quantize_kernel<false, true><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
        } else {
            if (return_transpose)
                tk_v5::persistent_quantize_kernel<true, false><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
            else
                tk_v5::persistent_quantize_kernel<false, false><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_v4_quantize_for_gemm_launch failed: ", cudaGetErrorString(err));

    auto row_fp4_v = row_fp4.view(torch::kFloat4_e2m1fn_x2);
    auto row_sc_v  = row_sc.view(torch::kFloat8_e4m3fn);
    auto col_fp4_v = return_transpose ? col_fp4.view(torch::kFloat4_e2m1fn_x2) : col_fp4;
    auto col_sc_v  = return_transpose ? col_sc.view(torch::kFloat8_e4m3fn) : col_sc;
    return std::make_tuple(row_fp4_v, row_sc_v, col_fp4_v, col_sc_v,
                           amax_buf.narrow(0, 1, 1), amax_buf.narrow(0, 1, 1));
}


// ═══════════════════════════════════════════════════════════════════
// 2. tk_quantize_transpose — pre-computed amax (v2 kernel)
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_quantize_transpose(
    torch::Tensor input, torch::Tensor amax_row, torch::Tensor amax_col, bool return_transpose
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(amax_row.is_cuda() && amax_row.scalar_type() == torch::kFloat32 && amax_row.numel() == 1);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 32 == 0 && K % 32 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc  = torch::empty({M, scale_stride}, opts_u8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_u8) : torch::empty({0}, opts_u8);
    auto col_sc  = return_transpose ? torch::empty({K, scale_stride_t}, opts_u8) : torch::empty({0}, opts_u8);

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
    if (return_transpose)
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

    auto *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    auto *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    const float *amax_r = reinterpret_cast<const float*>(amax_row.data_ptr());
    const float *amax_c = (amax_col.numel() > 0) ? reinterpret_cast<const float*>(amax_col.data_ptr()) : amax_r;

    if (return_transpose)
        launch_v2_kernel<false, false, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K, scale_stride, scale_stride_t, stream);
    else
        launch_v2_kernel<false, false, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K, scale_stride, scale_stride_t, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_transpose failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


// ═══════════════════════════════════════════════════════════════════
// 3. Grouped dim=0 — fused single-pass with v2 fallback
// ═══════════════════════════════════════════════════════════════════

// ── Cached pinned host memory for TMA descriptor staging (CUDA graph-safe) ──
static CUtensorMap* s_pinned_tma_maps = nullptr;
static size_t s_pinned_tma_capacity = 0;

static CUtensorMap* get_pinned_tma_buffer(size_t n_maps) {
    if (s_pinned_tma_maps == nullptr || n_maps > s_pinned_tma_capacity) {
        if (s_pinned_tma_maps) cudaFreeHost(s_pinned_tma_maps);
        cudaHostAlloc(reinterpret_cast<void**>(&s_pinned_tma_maps),
                      n_maps * sizeof(CUtensorMap), cudaHostAllocDefault);
        s_pinned_tma_capacity = n_maps;
    }
    return s_pinned_tma_maps;
}

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
    TORCH_CHECK(N <= tk_v3::MAX_GROUPS, "Max ", tk_v3::MAX_GROUPS, " splits");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    int64_t sum_splits = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        TORCH_CHECK(split_sections[i] % 128 == 0);
        sum_splits += split_sections[i];
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad_tiles = (int64_t)N * dgrad_tiles_per;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    auto wc_fp4_row = torch::empty({total_rows, K / 2}, opts_fp4);
    auto wc_fp4_col = torch::empty({K, total_rows / 2}, opts_u8);

    auto sg_cat     = torch::empty({N}, opts_f32);
    auto fwd_b_sg   = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad_tiles}, opts_f32);

    auto amax_tensor = torch::empty({N}, opts_f32);
    auto sync_tensor = torch::empty({2 * N}, opts_u32);
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, N * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * N * sizeof(unsigned int), stream);

    std::vector<torch::Tensor> sc_row_allocs(N);
    std::vector<torch::Tensor> fp4_col_list(N), sc_col_allocs(N);

    tk_v3::FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + N;
    args.num_groups   = N;
    args.sg_output    = sg_cat.data_ptr<float>();
    args.fwd_b_sg     = fwd_b_sg.data_ptr<float>();
    args.dgrad_b_sg   = dgrad_b_sg.data_ptr<float>();
    args.b_tile_size  = (int)Nb;
    args.total_cols   = (int)K;
    args.swizzle_scales = true;
    args.split_range[0] = 0;

    using namespace tk_v3;
    const int blocks_X = K / V3Config::CHUNK_DIM_X;

    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        args.split_range[i + 1] = args.split_range[i] + (int)M_i;
        args.blocks_per_group[i] = (int)(M_i / V3Config::CHUNK_DIM_Y) * blocks_X;

        sc_row_allocs[i] = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        args.row_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[i].data_ptr());

        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        args.col_data_ptrs[i] = reinterpret_cast<fp4e2m1x2*>(fp4_col_list[i].data_ptr());

        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        sc_col_allocs[i] = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
        args.col_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[i].data_ptr());
        args.col_scale_stride[i] = (int)c_sc_stride;
    }

    const int blocks_Y = total_rows / V3Config::CHUNK_DIM_Y;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int max_concurrent = ci.grp_d0_max_bps * ci.num_sms;
    const bool use_fused = (total_grid <= max_concurrent && ci.grp_d0_max_bps > 0);

    auto tma_dev_buf = torch::empty({0}, opts_u8);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(), K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;

        // Create TMA maps in pinned host memory (CUDA graph-safe)
        CUtensorMap* pinned_maps = get_pinned_tma_buffer(2 * N);
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            const int64_t ntm_i = M_i / 128;
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            create_tma_2d(pinned_maps[i], sc_row_allocs[i].data_ptr(), ntm_i, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

            const int64_t ntm_c_g = K / 128;
            const int64_t sc_col_x_bf16 = (M_i / 64) * 256;
            create_tma_2d(pinned_maps[N + i], sc_col_allocs[i].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        tma_dev_buf = torch::empty({(int64_t)(2 * N * sizeof(CUtensorMap))}, opts_u8);
        cudaMemcpyAsync(tma_dev_buf.data_ptr(), pinned_maps, 2 * N * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
        args.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
        args.use_tma_scales = true;

        fused_group_quantize_kernel_dim0<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr, nullptr,
            total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
            args);
    } else {
        // ─── v4 persistent grouped (work-stealing, L2 re-reads) ───
        // Need additional sync buffers for persistent barrier
        auto psync_tensor = torch::empty({4}, opts_u32);
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());
        cudaMemsetAsync(psync, 0, 4 * sizeof(unsigned int), stream);

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(), K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        int num_persistent = ci.pg_d0_max_bps * ci.num_sms;
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = N;
        pargs.sg_output    = sg_cat.data_ptr<float>();
        pargs.fwd_b_sg     = fwd_b_sg.data_ptr<float>();
        pargs.dgrad_b_sg   = dgrad_b_sg.data_ptr<float>();
        pargs.b_tile_size  = (int)Nb;
        pargs.total_cols   = (int)K;
        pargs.swizzle_scales = true;

        for (int i = 0; i <= N; ++i) pargs.split_range[i] = args.split_range[i];
        for (int i = 0; i < N; ++i) {
            pargs.blocks_per_group[i] = args.blocks_per_group[i];
            pargs.row_scale_ptrs[i]   = args.row_scale_ptrs[i];
            pargs.col_data_ptrs[i]    = args.col_data_ptrs[i];
            pargs.col_scale_ptrs[i]   = args.col_scale_ptrs[i];
            pargs.col_scale_stride[i] = args.col_scale_stride[i];
        }

        // Create TMA maps in pinned host memory (CUDA graph-safe)
        CUtensorMap* pinned_maps_p = get_pinned_tma_buffer(2 * N);
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            const int64_t ntm_i = M_i / 128;
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            create_tma_2d(pinned_maps_p[i], sc_row_allocs[i].data_ptr(), ntm_i, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

            const int64_t ntm_c_g = K / 128;
            const int64_t sc_col_x_bf16 = (M_i / 64) * 256;
            create_tma_2d(pinned_maps_p[N + i], sc_col_allocs[i].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        tma_dev_buf = torch::empty({(int64_t)(2 * N * sizeof(CUtensorMap))}, opts_u8);
        cudaMemcpyAsync(tma_dev_buf.data_ptr(), pinned_maps_p, 2 * N * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
        pargs.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());

        const int dshmem = ci.v3_dshmem_t;
        const dim3 grid(num_persistent);
        tk_v5::persistent_group_quantize_kernel_dim0<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
            pargs);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_for_gemm failed: ", cudaGetErrorString(err));

    auto wc_sc_row_parts = std::vector<torch::Tensor>(N);
    for (int i = 0; i < N; ++i)
        wc_sc_row_parts[i] = sc_row_allocs[i].view(torch::kFloat8_e4m3fn);
    auto wc_sc_row = torch::cat(wc_sc_row_parts, 0);

    std::vector<torch::Tensor> sc_col_list(N);
    {
        int64_t row_offset = 0;
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            fp4_col_list[i] = wc_fp4_col.narrow(1, row_offset / 2, M_i / 2)
                                         .contiguous().view(torch::kFloat4_e2m1fn_x2);
            sc_col_list[i] = sc_col_allocs[i].view(torch::kFloat8_e4m3fn);
            row_offset += M_i;
        }
    }

    auto mega_buf = torch::empty({0}, opts_u8);
    return std::make_tuple(wc_fp4_row, wc_sc_row, fwd_b_sg,
                           fp4_col_list, sc_col_list, dgrad_b_sg, sg_cat, mega_buf);
}

// ═══════════════════════════════════════════════════════════════════
// 3b. Grouped dim=0 — NON-PERSISTENT two-pass (multi-stream safe)
//     Pass 1: pipelined grouped amax
//     Pass 2: v2 quantize kernel (pre-computed amax, no barriers)
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_for_gemm_v2(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0);

    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    TORCH_CHECK(N <= tk_v3::MAX_GROUPS, "Max ", tk_v3::MAX_GROUPS, " splits");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    int64_t sum_splits = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        TORCH_CHECK(split_sections[i] % 128 == 0);
        sum_splits += split_sections[i];
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad_tiles = (int64_t)N * dgrad_tiles_per;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    // Row FP4 (contiguous)
    auto wc_fp4_row = torch::empty({total_rows, K / 2}, opts_u8).view(torch::kFloat4_e2m1fn_x2);
    // Col FP4 (contiguous — we'll split per-group after)
    auto wc_fp4_col = torch::empty({K, total_rows / 2}, opts_u8);

    // Per-split amax (zeroed for atomicMax)
    auto amaxes = torch::zeros({N}, opts_f32);
    auto sg_cat = torch::empty({N}, opts_f32);
    auto fwd_b_sg  = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad_tiles}, opts_f32);

    // Per-split scale + col allocations
    std::vector<torch::Tensor> sc_row_allocs(N), fp4_col_list(N), sc_col_allocs(N);

    // ── Pass 1: Pipelined grouped amax ──
    {
        // Use pinned host memory for offsets (CUDA graph-safe)
        static int64_t* s_pinned_offsets = nullptr;
        static size_t s_pinned_offsets_cap = 0;
        size_t need = (size_t)(N + 1);
        if (!s_pinned_offsets || need > s_pinned_offsets_cap) {
            if (s_pinned_offsets) cudaFreeHost(s_pinned_offsets);
            cudaHostAlloc(reinterpret_cast<void**>(&s_pinned_offsets),
                          need * sizeof(int64_t), cudaHostAllocDefault);
            s_pinned_offsets_cap = need;
        }
        s_pinned_offsets[0] = 0;
        for (int i = 0; i < N; ++i)
            s_pinned_offsets[i + 1] = s_pinned_offsets[i] + split_sections[i] * K;

        auto d_offsets = torch::empty({N + 1}, torch::dtype(torch::kInt64).device(device));
        cudaMemcpyAsync(d_offsets.data_ptr<int64_t>(), s_pinned_offsets,
                        (N + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        int64_t n = total_rows * K;
        int grp_blocks = pipelined_amax::grid_size(n);
        int grp_smem = pipelined_amax::smem_size();
        pipelined_amax::grouped_amax_pipelined_kernel<<<grp_blocks, pipelined_amax::THREADS, grp_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            amaxes.data_ptr<float>(),
            d_offsets.data_ptr<int64_t>(),
            N, n);
    }

    // sg = amax / 2688
    compute_sg_kernel<<<1, N, 0, stream>>>(
        amaxes.data_ptr<float>(), sg_cat.data_ptr<float>(), N);

    // ── Build v2 kernel args ──
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
        kernel_args.rowwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
        kernel_args.colwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);

        sc_row_allocs[i] = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        kernel_args.output_rowwise_scale_inv_list[i] = sc_row_allocs[i].data_ptr();

        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        kernel_args.output_colwise_data_list[i] = fp4_col_list[i].data_ptr();

        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        sc_col_allocs[i] = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
        kernel_args.output_colwise_scale_inv_list[i] = sc_col_allocs[i].data_ptr();
        kernel_args.output_colwise_scale_stride[i] = (int)c_sc_stride;
    }

    // ── Pass 2: v2 grouped quantize (non-persistent) ──
    alignas(64) CUtensorMap tmap_in{}, tmap_out{};
    create_tma_2d(tmap_in, input.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 4);

    nvfp4_scale_t* scales_ptr = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[0].data_ptr());
    launch_group_kernel_v2<false, true>(
        tmap_in, tmap_out, scales_ptr,
        total_rows, K, scale_stride,
        kernel_args, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_for_gemm_v2 failed: ", cudaGetErrorString(err));

    // ── Reshape outputs (same as v5) ──
    auto wc_sc_row_parts = std::vector<torch::Tensor>(N);
    for (int i = 0; i < N; ++i)
        wc_sc_row_parts[i] = sc_row_allocs[i].view(torch::kFloat8_e4m3fn);
    auto wc_sc_row = torch::cat(wc_sc_row_parts, 0);

    std::vector<torch::Tensor> sc_col_list(N);
    {
        int64_t row_offset = 0;
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            fp4_col_list[i] = fp4_col_list[i].view(torch::kFloat4_e2m1fn_x2);
            sc_col_list[i] = sc_col_allocs[i].view(torch::kFloat8_e4m3fn);
            row_offset += M_i;
        }
    }

    auto mega_buf = torch::empty({0}, opts_u8);
    return std::make_tuple(wc_fp4_row, wc_sc_row, fwd_b_sg,
                           fp4_col_list, sc_col_list, dgrad_b_sg, sg_cat, mega_buf);
}


// ═══════════════════════════════════════════════════════════════════
// 3c. Multi-stream graph-safe split API:
//     - v2_alloc: pre-allocate all output tensors (call on capture/main stream)
//     - v2_launch: kernel-only dispatch (safe on ANY stream during graph capture)
// ═══════════════════════════════════════════════════════════════════

// Returns: (wc_fp4_row, amaxes, sg_cat, fwd_b_sg, dgrad_b_sg, d_offsets,
//           sc_row_list, fp4_col_list, sc_col_list)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
tk_group_quantize_v2_alloc(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    int64_t sum_splits = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        sum_splits += split_sections[i];
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad = (int64_t)N * dgrad_tiles_per;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    // Pre-allocate everything on the CURRENT stream (capture stream)
    auto wc_fp4_row = torch::empty({total_rows, K / 2}, opts_u8);
    auto amaxes     = torch::zeros({N}, opts_f32);
    auto sg_cat     = torch::empty({N}, opts_f32);
    auto fwd_b_sg   = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad}, opts_f32);
    auto d_offsets  = torch::empty({N + 1}, torch::dtype(torch::kInt64).device(device));

    std::vector<torch::Tensor> sc_row_list(N), fp4_col_list(N), sc_col_list(N);
    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        sc_row_list[i]  = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        sc_col_list[i]  = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
    }

    return std::make_tuple(wc_fp4_row, amaxes, sg_cat, fwd_b_sg, dgrad_b_sg,
                           d_offsets, sc_row_list, fp4_col_list, sc_col_list);
}

// Kernel-only dispatch — NO allocations, safe on any stream during graph capture.
// Takes pre-allocated buffers from v2_alloc.
// Returns: (fp4_row_viewed, fwd_b_sg, dgrad_b_sg, sg_cat)
// Caller reconstructs wc_sc_row = torch.cat([sc_row[i].view(torch.kFloat8_e4m3fn) for i in range(N)])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_v2_launch(
    torch::Tensor input,
    std::vector<int64_t> split_sections,
    // Pre-allocated buffers from v2_alloc:
    torch::Tensor wc_fp4_row,
    torch::Tensor amaxes,
    torch::Tensor sg_cat,
    torch::Tensor fwd_b_sg,
    torch::Tensor dgrad_b_sg,
    torch::Tensor d_offsets,
    std::vector<torch::Tensor> sc_row_list,
    std::vector<torch::Tensor> fp4_col_list,
    std::vector<torch::Tensor> sc_col_list
) {
    auto& ci = get_cached_info();
    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    // ── Pass 1: Pipelined grouped amax (cudaMemcpyAsync from pinned is graph-safe) ──
    {
        static int64_t* s_pinned_offsets = nullptr;
        static size_t s_pinned_offsets_cap = 0;
        size_t need = (size_t)(N + 1);
        if (!s_pinned_offsets || need > s_pinned_offsets_cap) {
            if (s_pinned_offsets) cudaFreeHost(s_pinned_offsets);
            cudaHostAlloc(reinterpret_cast<void**>(&s_pinned_offsets),
                          need * sizeof(int64_t), cudaHostAllocDefault);
            s_pinned_offsets_cap = need;
        }
        s_pinned_offsets[0] = 0;
        for (int i = 0; i < N; ++i)
            s_pinned_offsets[i + 1] = s_pinned_offsets[i] + split_sections[i] * K;

        cudaMemcpyAsync(d_offsets.data_ptr<int64_t>(), s_pinned_offsets,
                        (N + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        int64_t n = total_rows * K;
        int grp_blocks = pipelined_amax::grid_size(n);
        int grp_smem = pipelined_amax::smem_size();
        pipelined_amax::grouped_amax_pipelined_kernel<<<grp_blocks, pipelined_amax::THREADS, grp_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            amaxes.data_ptr<float>(),
            d_offsets.data_ptr<int64_t>(),
            N, n);
    }

    // sg = amax / 2688
    compute_sg_kernel<<<1, N, 0, stream>>>(
        amaxes.data_ptr<float>(), sg_cat.data_ptr<float>(), N);

    // ── Build kernel args ──
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
        kernel_args.rowwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
        kernel_args.colwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
        kernel_args.output_rowwise_scale_inv_list[i] = sc_row_list[i].data_ptr();
        kernel_args.output_colwise_data_list[i] = fp4_col_list[i].data_ptr();
        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        kernel_args.output_colwise_scale_inv_list[i] = sc_col_list[i].data_ptr();
        kernel_args.output_colwise_scale_stride[i] = (int)c_sc_stride;
    }

    // ── Pass 2: v2 grouped quantize (non-persistent, no allocations) ──
    alignas(64) CUtensorMap tmap_in{}, tmap_out{};
    create_tma_2d(tmap_in, input.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 4);

    nvfp4_scale_t* scales_ptr = reinterpret_cast<nvfp4_scale_t*>(sc_row_list[0].data_ptr());
    launch_group_kernel_v2<false, true>(
        tmap_in, tmap_out, scales_ptr,
        total_rows, K, scale_stride,
        kernel_args, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_v2_launch failed: ", cudaGetErrorString(err));

    // ── Only views (zero-allocation) — caller does torch::cat on capture stream ──
    auto wc_fp4_view = wc_fp4_row.view(torch::kFloat4_e2m1fn_x2);

    // Return: (fp4_row_viewed, fwd_b_sg, dgrad_b_sg, sg_cat)
    // The caller reconstructs wc_sc_row via torch::cat on the capture stream
    return std::make_tuple(wc_fp4_view, fwd_b_sg, dgrad_b_sg, sg_cat);
}


// ═══════════════════════════════════════════════════════════════════
// 3d. V5 Persistent/Fused split API for multi-stream graph capture:
//     - v5_alloc: pre-allocate all output tensors (call on capture/main stream)
//     - v5_launch: kernel-only dispatch (safe on ANY stream during graph capture)
// ═══════════════════════════════════════════════════════════════════

// Returns: (wc_fp4_row, wc_fp4_col, sg_cat, fwd_b_sg, dgrad_b_sg,
//           amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
//           sc_row_list, fp4_col_list, sc_col_list)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
tk_group_quantize_v5_alloc(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    int64_t sum_splits = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        sum_splits += split_sections[i];
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad = (int64_t)N * dgrad_tiles_per;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);

    // Pre-allocate everything on the CURRENT stream (capture stream)
    auto wc_fp4_row   = torch::empty({total_rows, K / 2}, opts_fp4);
    auto wc_fp4_col   = torch::empty({K, total_rows / 2}, opts_u8);
    auto sg_cat       = torch::empty({N}, opts_f32);
    auto fwd_b_sg     = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg   = torch::empty({total_dgrad}, opts_f32);
    auto amax_tensor  = torch::empty({N}, opts_f32);
    auto sync_tensor  = torch::empty({2 * N}, opts_u32);
    auto psync_tensor = torch::empty({4}, opts_u32);
    auto tma_dev_buf  = torch::empty({(int64_t)(2 * N * (int64_t)sizeof(CUtensorMap))}, opts_u8);

    std::vector<torch::Tensor> sc_row_list(N), fp4_col_list(N), sc_col_list(N);
    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        sc_row_list[i]  = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        sc_col_list[i]  = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
    }

    return std::make_tuple(wc_fp4_row, wc_fp4_col, sg_cat, fwd_b_sg, dgrad_b_sg,
                           amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
                           sc_row_list, fp4_col_list, sc_col_list);
}

// Kernel-only dispatch — NO allocations, safe on any stream during graph capture.
// Takes pre-allocated buffers from v5_alloc.
// Returns: (fp4_row_viewed, fwd_b_sg, dgrad_b_sg, sg_cat) — same as v2_launch
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_v5_launch(
    torch::Tensor input,
    std::vector<int64_t> split_sections,
    // Pre-allocated buffers from v5_alloc:
    torch::Tensor wc_fp4_row,
    torch::Tensor wc_fp4_col,
    torch::Tensor sg_cat,
    torch::Tensor fwd_b_sg,
    torch::Tensor dgrad_b_sg,
    torch::Tensor amax_tensor,
    torch::Tensor sync_tensor,
    torch::Tensor psync_tensor,
    torch::Tensor tma_dev_buf,
    std::vector<torch::Tensor> sc_row_list,
    std::vector<torch::Tensor> fp4_col_list,
    std::vector<torch::Tensor> sc_col_list
) {
    const auto& ci = get_cached_info();
    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    // Zero sync buffers (cudaMemsetAsync is graph-safe)
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, N * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * N * sizeof(unsigned int), stream);
    cudaMemsetAsync(psync_tensor.data_ptr(), 0, 4 * sizeof(unsigned int), stream);

    int64_t total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i)
        total_fwd_tiles += split_sections[i] / Nb;

    // ── Build FusedGroupArgs ──
    tk_v3::FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + N;
    args.num_groups   = N;
    args.sg_output    = sg_cat.data_ptr<float>();
    args.fwd_b_sg     = fwd_b_sg.data_ptr<float>();
    args.dgrad_b_sg   = dgrad_b_sg.data_ptr<float>();
    args.b_tile_size  = (int)Nb;
    args.total_cols   = (int)K;
    args.swizzle_scales = true;
    args.split_range[0] = 0;

    using namespace tk_v3;
    const int blocks_X = K / V3Config::CHUNK_DIM_X;

    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        args.split_range[i + 1] = args.split_range[i] + (int)M_i;
        args.blocks_per_group[i] = (int)(M_i / V3Config::CHUNK_DIM_Y) * blocks_X;
        args.row_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_row_list[i].data_ptr());
        args.col_data_ptrs[i] = reinterpret_cast<fp4e2m1x2*>(fp4_col_list[i].data_ptr());
        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        args.col_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_col_list[i].data_ptr());
        args.col_scale_stride[i] = (int)c_sc_stride;
    }

    const int blocks_Y = total_rows / V3Config::CHUNK_DIM_Y;
    const int total_grid = blocks_X * blocks_Y;
    const int max_concurrent = ci.grp_d0_max_bps * ci.num_sms;
    const bool use_fused = (total_grid <= max_concurrent && ci.grp_d0_max_bps > 0);

    // Create TMA maps in pinned host memory (CUDA graph-safe)
    CUtensorMap* pinned_maps = get_pinned_tma_buffer(2 * N);
    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        const int64_t ntm_i = M_i / 128;
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(pinned_maps[i], sc_row_list[i].data_ptr(), ntm_i, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

        const int64_t ntm_c_g = K / 128;
        const int64_t sc_col_x_bf16 = (M_i / 64) * 256;
        create_tma_2d(pinned_maps[N + i], sc_col_list[i].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }
    cudaMemcpyAsync(tma_dev_buf.data_ptr(), pinned_maps, 2 * N * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(), K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        args.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
        args.use_tma_scales = true;

        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;
        fused_group_quantize_kernel_dim0<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr, nullptr,
            total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
            args);
    } else {
        // ─── Persistent grouped path ───
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(), K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        int num_persistent = ci.pg_d0_max_bps * ci.num_sms;
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = N;
        pargs.sg_output    = sg_cat.data_ptr<float>();
        pargs.fwd_b_sg     = fwd_b_sg.data_ptr<float>();
        pargs.dgrad_b_sg   = dgrad_b_sg.data_ptr<float>();
        pargs.b_tile_size  = (int)Nb;
        pargs.total_cols   = (int)K;
        pargs.swizzle_scales = true;

        for (int i = 0; i <= N; ++i) pargs.split_range[i] = args.split_range[i];
        for (int i = 0; i < N; ++i) {
            pargs.blocks_per_group[i] = args.blocks_per_group[i];
            pargs.row_scale_ptrs[i]   = args.row_scale_ptrs[i];
            pargs.col_data_ptrs[i]    = args.col_data_ptrs[i];
            pargs.col_scale_ptrs[i]   = args.col_scale_ptrs[i];
            pargs.col_scale_stride[i] = args.col_scale_stride[i];
        }
        pargs.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());

        const int dshmem = ci.v3_dshmem_t;
        const dim3 grid(num_persistent);
        tk_v5::persistent_group_quantize_kernel_dim0<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
            pargs);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_v5_launch failed: ", cudaGetErrorString(err));

    // ── Only view (zero-allocation) — caller does torch::cat on capture stream ──
    auto wc_fp4_view = wc_fp4_row.view(torch::kFloat4_e2m1fn_x2);
    return std::make_tuple(wc_fp4_view, fwd_b_sg, dgrad_b_sg, sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_group_quantize_dim1_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), N_total = input.size(1);
    TORCH_CHECK(M % 128 == 0 && N_total % 128 == 0);
    const int G = (int)col_split_sections.size();
    TORCH_CHECK(G <= tk_v3::MAX_GROUPS, "Max ", tk_v3::MAX_GROUPS, " groups");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);

    int64_t sum_cols = 0;
    for (int i = 0; i < G; ++i) {
        TORCH_CHECK(col_split_sections[i] % 128 == 0);
        sum_cols += col_split_sections[i];
    }
    TORCH_CHECK(sum_cols == N_total);

    auto amax_tensor = torch::empty({G}, opts_f32);
    auto sync_tensor = torch::empty({2 * G}, opts_u32);
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, G * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * G * sizeof(unsigned int), stream);

    auto sg_per_group = torch::empty({G}, opts_f32);
    auto fp4_row_full = torch::empty({M, N_total / 2}, opts_u8);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;

    using namespace tk_v3;
    FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + G;
    args.num_groups   = G;
    args.sg_output    = sg_per_group.data_ptr<float>();
    args.fwd_b_sg     = nullptr;
    args.dgrad_b_sg   = nullptr;
    args.swizzle_scales = true;
    args.split_range[0] = 0;

    const int blocks_Y = M / V3Config::CHUNK_DIM_Y;

    std::vector<torch::Tensor> sc_row_allocs(G), fp4_col_allocs(G), sc_col_allocs(G);

    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        args.split_range[g + 1] = args.split_range[g] + (int)N_g;
        args.blocks_per_group[g] = (int)(N_g / V3Config::CHUNK_DIM_X) * blocks_Y;

        const int64_t ntk_r_g = N_g / 64;
        sc_row_allocs[g] = torch::empty({ntm_r, ntk_r_g, 512}, opts_u8);
        args.row_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[g].data_ptr());

        fp4_col_allocs[g] = torch::empty({N_g, M / 2}, opts_u8);
        args.col_data_ptrs[g] = reinterpret_cast<fp4e2m1x2*>(fp4_col_allocs[g].data_ptr());

        const int64_t ntm_c_g = N_g / 128;
        const int64_t col_sc_stride = ((M / 16) + 3) / 4 * 4;
        sc_col_allocs[g] = torch::zeros({ntm_c_g, ntk_c, 512}, opts_u8);
        args.col_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[g].data_ptr());
        args.col_scale_stride[g] = (int)col_sc_stride;
    }

    auto fp4_col_full = torch::empty({N_total, M / 2}, opts_u8);

    const int blocks_X = N_total / V3Config::CHUNK_DIM_X;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int max_concurrent = ci.grp_d1_max_bps * ci.num_sms;
    const bool use_fused = (total_grid <= max_concurrent && ci.grp_d1_max_bps > 0);

    const int64_t global_scale_stride = ((N_total / 16) + 3) / 4 * 4;
    auto tma_dev_buf = torch::empty({0}, opts_u8);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;

        // Create TMA maps on host, copy to device tensor
        alignas(64) CUtensorMap host_tma_maps[2 * tk_v3::MAX_GROUPS];
        for (int g = 0; g < G; ++g) {
            const int64_t N_g = col_split_sections[g];
            const int64_t ntk_r_g = N_g / 64;
            const int64_t sc_row_x_bf16 = ntk_r_g * 256;
            create_tma_2d(host_tma_maps[g], sc_row_allocs[g].data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

            const int64_t ntm_c_g = N_g / 128;
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(host_tma_maps[G + g], sc_col_allocs[g].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        tma_dev_buf = torch::empty({(int64_t)(2 * G * sizeof(CUtensorMap))}, opts_u8);
        cudaMemcpyAsync(tma_dev_buf.data_ptr(), host_tma_maps, 2 * G * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
        args.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
        args.use_tma_scales = true;

        fused_group_quantize_kernel_dim1<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
            args);
    } else {
        // ─── v4 persistent grouped (work-stealing, L2 re-reads) ───
        auto psync_tensor = torch::empty({4}, opts_u32);
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());
        cudaMemsetAsync(psync, 0, 4 * sizeof(unsigned int), stream);

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        int num_persistent = ci.pg_d1_max_bps * ci.num_sms;
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = G;
        pargs.sg_output    = sg_per_group.data_ptr<float>();
        pargs.fwd_b_sg     = nullptr;
        pargs.dgrad_b_sg   = nullptr;
        pargs.swizzle_scales = true;

        for (int g = 0; g <= G; ++g) pargs.split_range[g] = args.split_range[g];
        for (int g = 0; g < G; ++g) {
            pargs.blocks_per_group[g] = args.blocks_per_group[g];
            pargs.row_scale_ptrs[g]   = args.row_scale_ptrs[g];
            pargs.col_data_ptrs[g]    = args.col_data_ptrs[g];
            pargs.col_scale_ptrs[g]   = args.col_scale_ptrs[g];
            pargs.col_scale_stride[g] = args.col_scale_stride[g];
        }

        // Create TMA maps on host, copy to device tensor
        alignas(64) CUtensorMap host_tma_maps[2 * tk_v3::MAX_GROUPS];
        for (int g = 0; g < G; ++g) {
            const int64_t N_g = col_split_sections[g];
            const int64_t ntk_r_g = N_g / 64;
            const int64_t sc_row_x_bf16 = ntk_r_g * 256;
            create_tma_2d(host_tma_maps[g], sc_row_allocs[g].data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

            const int64_t ntm_c_g = N_g / 128;
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(host_tma_maps[G + g], sc_col_allocs[g].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        tma_dev_buf = torch::empty({(int64_t)(2 * G * sizeof(CUtensorMap))}, opts_u8);
        cudaMemcpyAsync(tma_dev_buf.data_ptr(), host_tma_maps, 2 * G * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
        pargs.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());

        const int dshmem = ci.v3_dshmem_t;
        const dim3 grid(num_persistent);
        tk_v5::persistent_group_quantize_kernel_dim1<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
            pargs);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_dim1_for_gemm failed: ", cudaGetErrorString(err));

    std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
    std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
    std::vector<torch::Tensor> sc_row_u8_list(G), sc_col_u8_list(G);
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];

        // narrow along dim=1 is non-contiguous — return as uint8 view (caller does .contiguous().view(fp4))
        fp4_row_list[g] = fp4_row_full.narrow(1, col_offset / 2, N_g / 2);
        sc_row_list[g] = sc_row_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_row_u8_list[g] = sc_row_allocs[g];

        fp4_col_list[g] = fp4_col_full.narrow(0, col_offset, N_g).view(torch::kFloat4_e2m1fn_x2);
        sc_col_list[g] = sc_col_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_col_u8_list[g] = sc_col_allocs[g];
        col_offset += N_g;
    }

    // Concatenate row/col scales in C++ — avoids Python torch.cat round-trip
    auto sc_row_cat = torch::cat(sc_row_u8_list, 1).view(torch::kFloat8_e4m3fn);
    auto sc_col_cat = torch::cat(sc_col_u8_list, 0).view(torch::kFloat8_e4m3fn);
    auto fp4_row_view = fp4_row_full.view(torch::kFloat4_e2m1fn_x2);
    auto fp4_col_view = fp4_col_full.view(torch::kFloat4_e2m1fn_x2);

    return std::make_tuple(fp4_row_list, sc_row_list, sg_per_group,
                           fp4_col_list, sc_col_list,
                           fp4_row_view, sc_row_cat,
                           fp4_col_view, sc_col_cat);
}


// ═══════════════════════════════════════════════════════════════════
// 3e. Dim-1 split API for CUDA graph capture:
//     - dim1_alloc: pre-allocate all output tensors (call BEFORE graph capture)
//     - dim1_launch: kernel-only dispatch (safe INSIDE graph capture)
// ═══════════════════════════════════════════════════════════════════

// Returns: (fp4_row_full, fp4_col_full, sg_per_group,
//           amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
//           sc_row_allocs, fp4_col_allocs, sc_col_allocs, tma_host_buf)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, torch::Tensor>
tk_group_quantize_dim1_alloc(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    const int64_t M = input.size(0), N_total = input.size(1);
    const int G = (int)col_split_sections.size();
    auto device = input.device();
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;

    auto fp4_row_full = torch::empty({M, N_total / 2}, opts_u8);
    auto fp4_col_full = torch::empty({N_total, M / 2}, opts_u8);
    auto sg_per_group = torch::empty({G}, opts_f32);
    auto amax_tensor  = torch::empty({G}, opts_f32);
    auto sync_tensor  = torch::empty({2 * G}, opts_u32);
    auto psync_tensor = torch::empty({4}, opts_u32);
    auto tma_dev_buf  = torch::empty({(int64_t)(2 * G * (int64_t)sizeof(CUtensorMap))}, opts_u8);

    // Dedicated pinned host buffer for TMA descriptors — NOT shared with
    // other kernels, so it won't be overwritten between CUDA graph replays.
    auto tma_host_buf = torch::empty({(int64_t)(2 * G * (int64_t)sizeof(CUtensorMap))},
                                     torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));

    std::vector<torch::Tensor> sc_row_allocs(G), fp4_col_allocs(G), sc_col_allocs(G);
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        const int64_t ntk_r_g = N_g / 64;
        sc_row_allocs[g]  = torch::empty({ntm_r, ntk_r_g, 512}, opts_u8);
        fp4_col_allocs[g] = torch::empty({N_g, M / 2}, opts_u8);
        const int64_t ntm_c_g = N_g / 128;
        sc_col_allocs[g]  = torch::zeros({ntm_c_g, ntk_c, 512}, opts_u8);
    }

    return std::make_tuple(fp4_row_full, fp4_col_full, sg_per_group,
                           amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
                           sc_row_allocs, fp4_col_allocs, sc_col_allocs,
                           tma_host_buf);
}

// Kernel-only dispatch — NO allocations, safe inside CUDA graph capture.
// Takes pre-allocated buffers from dim1_alloc.
// Returns: (fp4_row_list, sc_row_list, sg_per_group,
//           fp4_col_list, sc_col_list,
//           fp4_row_view, sc_row_cat, fp4_col_view, sc_col_cat)
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_group_quantize_dim1_launch(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections,
    // Pre-allocated buffers from dim1_alloc:
    torch::Tensor fp4_row_full,
    torch::Tensor fp4_col_full,
    torch::Tensor sg_per_group,
    torch::Tensor amax_tensor,
    torch::Tensor sync_tensor,
    torch::Tensor psync_tensor,
    torch::Tensor tma_host_buf,  // dedicated pinned host buffer for TMA descs
    torch::Tensor tma_dev_buf,
    std::vector<torch::Tensor> sc_row_allocs,
    std::vector<torch::Tensor> fp4_col_allocs,
    std::vector<torch::Tensor> sc_col_allocs,
    bool skip_cat = false
) {
    const int64_t M = input.size(0), N_total = input.size(1);
    const int G = (int)col_split_sections.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;

    // Zero sync buffers (cudaMemsetAsync is graph-safe)
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, G * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * G * sizeof(unsigned int), stream);
    cudaMemsetAsync(psync_tensor.data_ptr(), 0, 4 * sizeof(unsigned int), stream);
    // Zero sc_col_allocs (they use torch::zeros in alloc, but need re-zeroing for graph replay)
    for (int g = 0; g < G; ++g) {
        cudaMemsetAsync(sc_col_allocs[g].data_ptr(), 0,
                        sc_col_allocs[g].numel() * sc_col_allocs[g].element_size(), stream);
    }

    using namespace tk_v3;
    FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + G;
    args.num_groups   = G;
    args.sg_output    = sg_per_group.data_ptr<float>();
    args.fwd_b_sg     = nullptr;
    args.dgrad_b_sg   = nullptr;
    args.swizzle_scales = true;
    args.split_range[0] = 0;

    const int blocks_Y = M / V3Config::CHUNK_DIM_Y;

    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        args.split_range[g + 1] = args.split_range[g] + (int)N_g;
        args.blocks_per_group[g] = (int)(N_g / V3Config::CHUNK_DIM_X) * blocks_Y;

        args.row_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[g].data_ptr());
        args.col_data_ptrs[g] = reinterpret_cast<fp4e2m1x2*>(fp4_col_allocs[g].data_ptr());
        const int64_t col_sc_stride = ((M / 16) + 3) / 4 * 4;
        args.col_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[g].data_ptr());
        args.col_scale_stride[g] = (int)col_sc_stride;
    }

    const int blocks_X = N_total / V3Config::CHUNK_DIM_X;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int max_concurrent = ci.grp_d1_max_bps * ci.num_sms;
    const bool use_fused = (total_grid <= max_concurrent && ci.grp_d1_max_bps > 0);

    const int64_t global_scale_stride = ((N_total / 16) + 3) / 4 * 4;

    // Create TMA maps in DEDICATED pinned host buffer (not the shared one),
    // then copy to device. The dedicated buffer isn't overwritten by other
    // kernels between CUDA graph replays, so cudaMemcpyAsync reads correct data.
    CUtensorMap* pinned_maps = reinterpret_cast<CUtensorMap*>(tma_host_buf.data_ptr());
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        const int64_t ntk_r_g = N_g / 64;
        const int64_t sc_row_x_bf16 = ntk_r_g * 256;
        create_tma_2d(pinned_maps[g], sc_row_allocs[g].data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

        const int64_t ntm_c_g = N_g / 128;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(pinned_maps[G + g], sc_col_allocs[g].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }
    cudaMemcpyAsync(tma_dev_buf.data_ptr(), pinned_maps, 2 * G * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        args.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
        args.use_tma_scales = true;

        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;
        fused_group_quantize_kernel_dim1<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
            args);
    } else {
        // Persistent grouped path
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        int num_persistent = ci.pg_d1_max_bps * ci.num_sms;
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = G;
        pargs.sg_output    = sg_per_group.data_ptr<float>();
        pargs.fwd_b_sg     = nullptr;
        pargs.dgrad_b_sg   = nullptr;
        pargs.swizzle_scales = true;

        for (int g = 0; g <= G; ++g) pargs.split_range[g] = args.split_range[g];
        for (int g = 0; g < G; ++g) {
            pargs.blocks_per_group[g] = args.blocks_per_group[g];
            pargs.row_scale_ptrs[g]   = args.row_scale_ptrs[g];
            pargs.col_data_ptrs[g]    = args.col_data_ptrs[g];
            pargs.col_scale_ptrs[g]   = args.col_scale_ptrs[g];
            pargs.col_scale_stride[g] = args.col_scale_stride[g];
        }
        pargs.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());

        const int dshmem = ci.v3_dshmem_t;
        const dim3 grid(num_persistent);
        tk_v5::persistent_group_quantize_kernel_dim1<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
            pargs);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_dim1_launch failed: ", cudaGetErrorString(err));

    // Build output views (zero-allocation, just views/narrows of pre-allocated tensors)
    std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
    std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
    std::vector<torch::Tensor> sc_row_u8_list(G), sc_col_u8_list(G);
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        fp4_row_list[g] = fp4_row_full.narrow(1, col_offset / 2, N_g / 2);
        sc_row_list[g] = sc_row_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_row_u8_list[g] = sc_row_allocs[g];
        fp4_col_list[g] = fp4_col_full.narrow(0, col_offset, N_g).view(torch::kFloat4_e2m1fn_x2);
        sc_col_list[g] = sc_col_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_col_u8_list[g] = sc_col_allocs[g];
        col_offset += N_g;
    }

    torch::Tensor sc_row_cat, sc_col_cat;
    if (skip_cat) {
        // Graph-safe: skip torch::cat to avoid graph-pool allocations
        sc_row_cat = torch::empty({0}, torch::dtype(torch::kUInt8).device(input.device()));
        sc_col_cat = torch::empty({0}, torch::dtype(torch::kUInt8).device(input.device()));
    } else {
        sc_row_cat = torch::cat(sc_row_u8_list, 1).view(torch::kFloat8_e4m3fn);
        sc_col_cat = torch::cat(sc_col_u8_list, 0).view(torch::kFloat8_e4m3fn);
    }
    auto fp4_row_view = fp4_row_full.view(torch::kFloat4_e2m1fn_x2);
    auto fp4_col_view = fp4_col_full.view(torch::kFloat4_e2m1fn_x2);

    return std::make_tuple(fp4_row_list, sc_row_list, sg_per_group,
                           fp4_col_list, sc_col_list,
                           fp4_row_view, sc_row_cat,
                           fp4_col_view, sc_col_cat);
}
//
// Two paths:
//   FUSED: single-pass grid-barrier kernel (when grid fits on GPU)
//   FALLBACK: inv_rms → bf16 transform → pipelined amax → quantize with pre-computed amax
// ═══════════════════════════════════════════════════════════════════

// Helper: apply rmsnorm + optional silu, output bf16 + compute amax
// This is used in the fallback 2-pass path
template <bool WITH_SILU, int BLOCK_SIZE = 256>
__global__ void transform_and_amax_kernel(
    const __nv_bfloat16* __restrict__ x,       // (M, K) input
    const __nv_bfloat16* __restrict__ gamma,    // (K,) rmsnorm weight
    const float* __restrict__ inv_rms,          // (M,) pre-computed
    __nv_bfloat16* __restrict__ out,            // (M, K) transformed output
    float* __restrict__ global_amax,            // scalar (atomic max target)
    float* __restrict__ sg_out,                 // scalar
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* row_x = x + (int64_t)row * cols;
    __nv_bfloat16* row_out = out + (int64_t)row * cols;
    float row_inv_rms = inv_rms[row];
    float thread_max = 0.0f;

    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = __bfloat162float(row_x[i]);
        float g = __bfloat162float(gamma[i]);
        float transformed = val * row_inv_rms * g;
        if constexpr (WITH_SILU) {
            transformed = transformed / (1.0f + expf(-transformed));
        }
        row_out[i] = __float2bfloat16_rn(transformed);
        thread_max = fmaxf(thread_max, fabsf(transformed));
    }

    // Warp reduce
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));

    // Block reduce
    __shared__ float warp_max[BLOCK_SIZE / 32];
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) warp_max[wid] = thread_max;
    __syncthreads();

    if (wid == 0) {
        thread_max = (lane < BLOCK_SIZE / 32) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int mask = (BLOCK_SIZE / 32) / 2; mask > 0; mask >>= 1)
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    // Atomic max to global (skip if no amax output needed)
    if (threadIdx.x == 0 && thread_max > 0.0f && global_amax != nullptr) {
        unsigned int* p = reinterpret_cast<unsigned int*>(global_amax);
        unsigned int old = *p;
        unsigned int want = __float_as_uint(thread_max);
        while (want > old) {
            old = atomicCAS(p, old, want);
        }
    }
}

// Helper: compute sg from amax (launched as 1 thread)
__global__ void compute_sg_from_amax(const float* __restrict__ amax, float* __restrict__ sg) {
    *sg = *amax / 2688.0f;
}

template <bool WITH_SILU, bool RETURN_TRANSPOSE, bool USE_ILP_AMAX>
static int get_persistent_norm_num_persistent_cap() {
    static int cached = 0;
    if (cached == 0) {
        using namespace tk_v3;
        const int dshmem = tk_v5::persistent_norm_quant_smem_size(RETURN_TRANSPOSE);
        auto kernel = tk_v5::persistent_norm_quantize_kernel<WITH_SILU, RETURN_TRANSPOSE, USE_ILP_AMAX>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        int max_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, V3_THREADS, dshmem);
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        cached = max_blocks * num_sms;
    }
    return cached;
}


template <bool USE_ILP_AMAX>
std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>  // inv_rms (for backward)
tk_fused_norm_quantize_impl(
    torch::Tensor input,        // (M, K) bf16 raw pre-norm
    torch::Tensor gamma,        // (K,) bf16 rmsnorm weight
    float epsilon,
    bool with_silu,
    bool return_transpose
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16, "input must be CUDA bf16");
    TORCH_CHECK(gamma.is_cuda() && gamma.dtype() == torch::kBFloat16, "gamma must be CUDA bf16");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(gamma.size(0) == K, "gamma must match K dimension");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    using namespace tk_v3;

    auto device = input.device();
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    // Step 1: Compute inv_rms (lightweight, 1 block per row — always needed)
    auto inv_rms_tensor = torch::empty({M}, opts_f32);
    float* inv_rms_ptr = inv_rms_tensor.data_ptr<float>();
    {
        constexpr int BS = 256;
        inv_rms_kernel_ns::compute_inv_rms_kernel<BS><<<M, BS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            inv_rms_ptr, epsilon, M, K);
    }

    // Step 2: Persistent fused norm+quantize
    // Uses v5 persistent pattern: work-stealing loops for Phase 1 (amax) and
    // Phase 2 (quantize). RMSNorm is applied inline (2 muls/element, cheap).
    // No intermediate bf16 buffer needed.
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;
    const int64_t scale_stride_r = ntk_r * 4;

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_u8) : torch::empty({0}, opts_u8);
    auto row_sc = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto col_sc = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_u8) : torch::empty({0}, opts_u8);

    // Static sync buffers (same pattern as silu+quantize)
    static float*        s_amax_buf = nullptr;
    static unsigned int* s_sync_buf = nullptr;
    if (!s_amax_buf) {
        cudaMalloc(&s_amax_buf, 2 * sizeof(float));   // [amax, sg]
        cudaMalloc(&s_sync_buf, 4 * sizeof(unsigned int)); // [wc1, wc2, done, ready]
    }
    cudaMemsetAsync(s_amax_buf, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(s_sync_buf, 0, 4 * sizeof(unsigned int), stream);

    // Determine persistent block count
    int dshmem = tk_v5::persistent_norm_quant_smem_size(return_transpose);
    int num_persistent = 1;
    if (with_silu && return_transpose) {
        num_persistent = std::min(
            get_persistent_norm_num_persistent_cap<true, true, USE_ILP_AMAX>(),
            total_tiles);
    } else if (with_silu) {
        num_persistent = std::min(
            get_persistent_norm_num_persistent_cap<true, false, USE_ILP_AMAX>(),
            total_tiles);
    } else if (return_transpose) {
        num_persistent = std::min(
            get_persistent_norm_num_persistent_cap<false, true, USE_ILP_AMAX>(),
            total_tiles);
    } else {
        num_persistent = std::min(
            get_persistent_norm_num_persistent_cap<false, false, USE_ILP_AMAX>(),
            total_tiles);
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    // TMA maps
    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    if (return_transpose)
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    // Scale TMA maps: view [ntm, ntk, 512] as 2D using BF16 type
    // Max TMA box = 512 bytes, so use shmemX=256 BF16 (512 bytes) per tile_k block
    // Each chunk does 2 TMA stores (one per tile_k × 512-byte block)
    {
        const int64_t sc_row_x_bf16 = ntk_r * 256;  // ntk_r*512 bytes / 2 = BF16 elements
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    }
    if (return_transpose) {
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    tk_v5::PersistentArgs p_args;
    p_args.work_counter_phase1 = s_sync_buf;
    p_args.work_counter_phase2 = s_sync_buf + 1;
    p_args.global_amax = s_amax_buf;
    p_args.done_counter = s_sync_buf + 2;
    p_args.ready_flag = s_sync_buf + 3;
    p_args.tiles_X = tiles_X;
    p_args.tiles_Y = tiles_Y;
    p_args.total_tiles = total_tiles;
    p_args.num_persistent = num_persistent;
    p_args.sg_output = s_amax_buf + 1;
    p_args.col_scales_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;
    p_args.col_scale_stride = 0;
    p_args.swizzle_scales = true;

    #define LAUNCH_PERSISTENT_NORM_QUANT(SILU_, TR_) \
        tk_v5::persistent_norm_quantize_kernel<SILU_, TR_, USE_ILP_AMAX><<<num_persistent, V3_THREADS, dshmem, stream>>>( \
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col, \
            reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()), \
            inv_rms_ptr, \
            reinterpret_cast<const tk_v3::IType*>(gamma.data_ptr()), \
            M, K, scale_stride_r, p_args)

    if (with_silu && return_transpose)  { LAUNCH_PERSISTENT_NORM_QUANT(true,  true);  }
    else if (with_silu)                 { LAUNCH_PERSISTENT_NORM_QUANT(true,  false); }
    else if (return_transpose)          { LAUNCH_PERSISTENT_NORM_QUANT(false, true);  }
    else                                { LAUNCH_PERSISTENT_NORM_QUANT(false, false); }
    #undef LAUNCH_PERSISTENT_NORM_QUANT

    // Read back sg from device buffer
    auto sg_tensor = torch::empty({1}, opts_f32);
    cudaMemcpyAsync(sg_tensor.data_ptr<float>(), s_amax_buf + 1,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);
    auto amax_tensor = torch::empty({1}, opts_f32);
    cudaMemcpyAsync(amax_tensor.data_ptr<float>(), s_amax_buf,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto fp4_row = row_fp4.view(torch::kFloat4_e2m1fn_x2);
    auto sc_row  = row_sc.view(torch::kFloat8_e4m3fn);
    auto fp4_col = return_transpose ? col_fp4.view(torch::kFloat4_e2m1fn_x2) : col_fp4;
    auto sc_col  = return_transpose ? col_sc.view(torch::kFloat8_e4m3fn) : col_sc;
    return std::make_tuple(fp4_row, sc_row, fp4_col, sc_col, sg_tensor, inv_rms_tensor, amax_tensor);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_fused_norm_quantize(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon,
    bool with_silu,
    bool return_transpose
) {
    return tk_fused_norm_quantize_impl<true>(
        input, gamma, epsilon, with_silu, return_transpose);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_fused_norm_quantize_naive(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon,
    bool with_silu,
    bool return_transpose
) {
    return tk_fused_norm_quantize_impl<false>(
        input, gamma, epsilon, with_silu, return_transpose);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_fused_norm_quantize_ilp(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon,
    bool with_silu,
    bool return_transpose
) {
    return tk_fused_norm_quantize_impl<true>(
        input, gamma, epsilon, with_silu, return_transpose);
}

torch::Tensor reconstruct_v6_impl(
    torch::Tensor fp4,
    torch::Tensor sc,
    torch::Tensor sg,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(fp4.is_cuda() && sc.is_cuda() && sg.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(fp4.scalar_type() == torch::kFloat4_e2m1fn_x2, "fp4 tensor dtype mismatch");
    TORCH_CHECK(sc.scalar_type() == torch::kFloat8_e4m3fn, "scale tensor dtype mismatch");
    TORCH_CHECK(sg.scalar_type() == torch::kFloat32 && sg.numel() == 1, "sg must be float32 scalar tensor");

    auto out = torch::empty({rows, cols}, torch::dtype(torch::kBFloat16).device(fp4.device()));
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t numel = rows * cols;
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    tk_v6_reconstruct::reconstruct_rowwise_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_fp4x2_e2m1*>(fp4.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(sc.data_ptr()),
        sg.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(rows),
        static_cast<int>(cols));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_v6_reconstruct failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor tk_reconstruct_row(
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor sg
) {
    return reconstruct_v6_impl(row_fp4, row_sc, sg, row_fp4.size(0), row_fp4.size(1) * 2);
}

torch::Tensor tk_reconstruct_col(
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor sg
) {
    return reconstruct_v6_impl(col_fp4, col_sc, sg, col_fp4.size(0), col_fp4.size(1) * 2);
}


// ═══════════════════════════════════════════════════════════════════
// Fused strided-SiLU + quantize
// Input: h13 (M, 2H) bf16. Applies silu(h1)*h3 → quantize to NVFP4.
// Output: same as tk_quantize_for_gemm but for (M, H).
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_quantize_for_gemm(torch::Tensor h13, int64_t H) {
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(h13.scalar_type() == torch::kBFloat16 && h13.dim() == 2);
    const int64_t M = h13.size(0);
    TORCH_CHECK(h13.size(1) == 2 * H, "h13 must have shape (M, 2*H)");
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = h13.device();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    // Common output setup
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride   = ((H / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;


    // Static pre-allocated sync/amax buffers (avoids cudaMallocAsync per call)
    static float *s_amax_buf = nullptr;
    static unsigned int *s_sync_buf = nullptr;
    if (!s_amax_buf) {
        cudaMalloc(&s_amax_buf, 2 * sizeof(float));
        cudaMalloc(&s_sync_buf, 4 * sizeof(unsigned int));
    }

    // Cached shmem size and occupancy (computed once)
    static int s_dshmem = -1;
    static int s_np_max_bps = -1;
    if (s_dshmem < 0) {
        s_dshmem = fused_silu_quant::fused_silu_quant_smem_size<true>();
        cudaFuncSetAttribute(
            fused_silu_quant::fused_silu_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_np_max_bps,
            fused_silu_quant::fused_silu_quantize_kernel<true>,
            V3_THREADS, s_dshmem);
    }

    const auto& ci = get_cached_info();
    const int max_concurrent = s_np_max_bps * ci.num_sms;
    const bool can_fuse = (total_tiles <= max_concurrent && s_np_max_bps > 0);

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    // Declare output tensors here, allocate in each branch for better pipelining
    torch::Tensor row_fp4, row_sc, col_fp4, col_sc;

    if (can_fuse) {
        // ─── NON-PERSISTENT PATH: single kernel, data stays in SMEM ───
        row_fp4 = torch::empty({M, H / 2}, opts_fp4);
        row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        col_fp4 = torch::empty({H, M / 2}, opts_fp4);
        col_sc  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

        cudaMemsetAsync(s_amax_buf, 0, 2 * sizeof(float), stream);
        cudaMemsetAsync(s_sync_buf, 0, 4 * sizeof(unsigned int), stream);

        nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
        nvfp4_scale_t *sc_t_ptr = reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr());

        alignas(64) CUtensorMap tmap_h1{}, tmap_h3{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_h1, h13.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);
        void* h3_ptr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h13.data_ptr()) + H * sizeof(__nv_bfloat16));
        create_tma_2d(tmap_h3, h3_ptr, M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        const dim3 grid(tiles_X, tiles_Y);
        fused_silu_quant::fused_silu_quantize_kernel<true><<<grid, V3_THREADS, s_dshmem, stream>>>(
            tmap_h1, tmap_h3, tmap_out, tmap_out_t,
            sc_ptr, sc_t_ptr,
            s_amax_buf, s_amax_buf + 1,
            s_sync_buf, s_sync_buf + 1,
            M, H, scale_stride, scale_stride_t,
            total_tiles);
    } else {
        // ─── LARGE GRID PATH: silu + v5 persistent quantize ───
        // CRITICAL: launch silu kernel FIRST so GPU starts working while
        // we do output tensor allocation + TMA map creation on the CPU.
        // This mimics the natural pipelining of the eager path.

        // Step 1: allocate silu output + launch silu kernel → GPU starts immediately
        auto h_silu = torch::empty({M, H}, h13.options());  // bf16
        {
            const __nv_bfloat16* h13_ptr = reinterpret_cast<const __nv_bfloat16*>(h13.data_ptr());
            __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(h_silu.data_ptr());
            const int64_t total_pairs = (int64_t)M * H / 2;
            const int threads = 256;
            int blocks = (int)((total_pairs + threads - 1) / threads);
            if (blocks > 65535) blocks = 65535;
            silu_strided_kernel<<<blocks, threads, 0, stream>>>(
                h13_ptr, out_ptr, M, H);
        }
        // GPU is now running silu kernel! Do remaining setup on CPU:

        // Step 2: allocate output tensors (CPU work, overlapped with GPU silu)
        row_fp4 = torch::empty({M, H / 2}, opts_fp4);
        row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        col_fp4 = torch::empty({H, M / 2}, opts_fp4);
        col_sc  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

        cudaMemsetAsync(s_amax_buf, 0, 2 * sizeof(float), stream);
        cudaMemsetAsync(s_sync_buf, 0, 4 * sizeof(unsigned int), stream);

        nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
        nvfp4_scale_t *sc_t_ptr = reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr());

        // Step 3: create TMA maps (CPU work, overlapped with GPU silu)
        alignas(64) CUtensorMap tmap_in_q{}, tmap_out_q{}, tmap_out_t_q{};
        create_tma_2d(tmap_in_q, h_silu.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out_q, row_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
        create_tma_2d(tmap_out_t_q, col_fp4.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

        // Step 4: launch v5 persistent quantize (queued behind silu on same stream)
        int v3_dshmem_q = ci.v3_dshmem_t;
        const int p_max_bps = ci.v4_max_bps_t;
        int num_persistent = p_max_bps * ci.num_sms;
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = s_sync_buf;
        pargs.work_counter_phase2 = s_sync_buf + 1;
        pargs.global_amax  = s_amax_buf;
        pargs.done_counter = s_sync_buf + 2;
        pargs.ready_flag   = s_sync_buf + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = s_amax_buf + 1;
        pargs.col_scales_ptr = sc_t_ptr;
        pargs.col_scale_stride = scale_stride_t;
        pargs.swizzle_scales = true;

        const dim3 grid(num_persistent);
        tk_v5::persistent_quantize_kernel<true><<<grid, V3_THREADS, v3_dshmem_q, stream>>>(
            tmap_in_q, tmap_out_q, tmap_out_t_q, tmap_sc_row, tmap_sc_col,
            sc_ptr, M, H, scale_stride, pargs);
    }

    // Copy sg to output tensor
    auto sg_buf = torch::empty({1}, opts_f32);
    cudaMemcpyAsync(sg_buf.data_ptr<float>(), s_amax_buf + 1, sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_quantize_for_gemm failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           sg_buf, sg_buf);
}


// ═══════════════════════════════════════════════════════════════════
// 6. Fused SiLU Derivative + Dual Quantize (BACKWARD)
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_for_gemm(
    torch::Tensor dh,
    torch::Tensor h13,
    int64_t H
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h13.scalar_type() == torch::kBFloat16 && h13.dim() == 2);

    const int64_t M = dh.size(0);
    TORCH_CHECK(dh.size(1) == H);
    TORCH_CHECK(h13.size(0) == M && h13.size(1) == 2 * H);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = dh.device();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    // Common output setup
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    // Static pre-allocated sync/amax buffers for backward
    static float *s_bwd_amax_buf = nullptr;
    static unsigned int *s_bwd_sync_buf = nullptr;
    if (!s_bwd_amax_buf) {
        cudaMalloc(&s_bwd_amax_buf, 2 * sizeof(float));
        cudaMalloc(&s_bwd_sync_buf, 4 * sizeof(unsigned int));
    }

    // Cached shmem size and occupancy for backward persistent kernel
    static int s_bwd_dshmem = -1;
    static int s_bwd_max_bps = -1;
    if (s_bwd_dshmem < 0) {
        s_bwd_dshmem = persistent_silu_deriv_quant::fused_silu_deriv_quant_smem_size<true>();
        cudaFuncSetAttribute(
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_bwd_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_bwd_max_bps,
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            V3_THREADS, s_bwd_dshmem);
    }

    const auto& ci = get_cached_info();
    const int max_concurrent = s_bwd_max_bps * ci.num_sms;
    const bool can_fuse = (total_tiles <= max_concurrent && s_bwd_max_bps > 0);

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    torch::Tensor out1_fp4, out1_sc, out2_fp4, out2_sc;
    torch::Tensor out1_fp4_t, out1_sc_t, out2_fp4_t, out2_sc_t;
    auto sg_buf = torch::empty({2}, opts_f32);
    float *sg_ptr = sg_buf.data_ptr<float>();

    if (can_fuse) {
        // ─── PERSISTENT PATH: small M, all fits in concurrent CTAs ───
        out1_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out1_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out2_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out2_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out1_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out1_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
        out2_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out2_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

        cudaMemsetAsync(s_bwd_amax_buf, 0, 2 * sizeof(float), stream);
        cudaMemsetAsync(s_bwd_sync_buf, 0, 4 * sizeof(unsigned int), stream);

        // TMA maps: inputs
        alignas(64) CUtensorMap tmap_dh{}, tmap_h1{}, tmap_h3{};
        create_tma_2d(tmap_dh, dh.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16);
        create_tma_2d(tmap_h1, h13.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);
        void* h3_ptr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h13.data_ptr()) + H * sizeof(__nv_bfloat16));
        create_tma_2d(tmap_h3, h3_ptr, M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);

        // TMA maps: row outputs
        alignas(64) CUtensorMap tmap_out1{}, tmap_out2{};
        create_tma_2d(tmap_out1, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
        create_tma_2d(tmap_out2, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);

        // TMA maps: col outputs
        alignas(64) CUtensorMap tmap_out1_t{}, tmap_out2_t{};
        create_tma_2d(tmap_out1_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
        create_tma_2d(tmap_out2_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        // TMA maps: scales
        alignas(64) CUtensorMap tmap_sc_row1{}, tmap_sc_row2{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row1, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        create_tma_2d(tmap_sc_row2, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        alignas(64) CUtensorMap tmap_sc_col1{}, tmap_sc_col2{};
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col1, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        create_tma_2d(tmap_sc_col2, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

        int num_persistent = s_bwd_max_bps * ci.num_sms;
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = s_bwd_sync_buf;
        pargs.work_counter_phase2 = s_bwd_sync_buf + 1;
        pargs.global_amax  = s_bwd_amax_buf;
        pargs.done_counter = s_bwd_sync_buf + 2;
        pargs.ready_flag   = s_bwd_sync_buf + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;

        const dim3 grid(num_persistent);
        persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true><<<grid, V3_THREADS, s_bwd_dshmem, stream>>>(
            tmap_dh, tmap_h1, tmap_h3,
            tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
            tmap_sc_row1, tmap_sc_row2, tmap_sc_col1, tmap_sc_col2,
            M, H, scale_stride, pargs, s_bwd_amax_buf + 1);
    } else {
        // ─── LARGE GRID PATH: silu-deriv elementwise + 2× v5 persistent quantize ───
        // Launch silu-deriv FIRST so GPU starts while we set up quantize.

        // Step 1: silu-deriv kernel → dh1_bf16, dh3_bf16 (GPU starts immediately)
        auto dh1_bf16 = torch::empty({M, H}, dh.options());
        auto dh3_bf16 = torch::empty({M, H}, dh.options());
        {
            const __nv_bfloat16* dh_ptr = reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr());
            const __nv_bfloat16* h13_ptr = reinterpret_cast<const __nv_bfloat16*>(h13.data_ptr());
            __nv_bfloat16* out1_ptr = reinterpret_cast<__nv_bfloat16*>(dh1_bf16.data_ptr());
            __nv_bfloat16* out2_ptr = reinterpret_cast<__nv_bfloat16*>(dh3_bf16.data_ptr());
            const int64_t total_pairs = (int64_t)M * H / 2;
            const int threads = 256;
            int blocks = (int)((total_pairs + threads - 1) / threads);
            if (blocks > 65535) blocks = 65535;
            silu_deriv_dual_strided_kernel<<<blocks, threads, 0, stream>>>(
                dh_ptr, h13_ptr, out1_ptr, out2_ptr, M, H);
        }
        // GPU is now running silu-deriv! Do quantize setup on CPU:

        // Step 2: allocate outputs (CPU work, overlapped with GPU)
        out1_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out1_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out2_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out2_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out1_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out1_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
        out2_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out2_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

        nvfp4_scale_t *sc1_ptr   = reinterpret_cast<nvfp4_scale_t*>(out1_sc.data_ptr());
        nvfp4_scale_t *sc1_t_ptr = reinterpret_cast<nvfp4_scale_t*>(out1_sc_t.data_ptr());
        nvfp4_scale_t *sc2_ptr   = reinterpret_cast<nvfp4_scale_t*>(out2_sc.data_ptr());
        nvfp4_scale_t *sc2_t_ptr = reinterpret_cast<nvfp4_scale_t*>(out2_sc_t.data_ptr());

        // Quantize dh1 (queued on stream behind silu-deriv)
        {
            cudaMemsetAsync(s_bwd_amax_buf, 0, sizeof(float), stream);
            cudaMemsetAsync(s_bwd_sync_buf, 0, 4 * sizeof(unsigned int), stream);

            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(tmap_in, dh1_bf16.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
            create_tma_2d(tmap_out, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
            create_tma_2d(tmap_out_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

            alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            create_tma_2d(tmap_sc_row, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

            int v3_dshmem_q = ci.v3_dshmem_t;
            const int p_max_bps = ci.v4_max_bps_t;
            int num_persistent = p_max_bps * ci.num_sms;
            if (num_persistent > total_tiles) num_persistent = total_tiles;

            tk_v5::PersistentArgs pargs;
            memset(&pargs, 0, sizeof(pargs));
            pargs.work_counter_phase1 = s_bwd_sync_buf;
            pargs.work_counter_phase2 = s_bwd_sync_buf + 1;
            pargs.global_amax  = s_bwd_amax_buf;
            pargs.done_counter = s_bwd_sync_buf + 2;
            pargs.ready_flag   = s_bwd_sync_buf + 3;
            pargs.tiles_X = tiles_X;
            pargs.tiles_Y = tiles_Y;
            pargs.total_tiles = total_tiles;
            pargs.num_persistent = num_persistent;
            pargs.sg_output = sg_ptr;  // sg for dh1
            pargs.col_scales_ptr = sc1_t_ptr;
            pargs.col_scale_stride = scale_stride_t;
            pargs.swizzle_scales = true;

            const dim3 grid(num_persistent);
            tk_v5::persistent_quantize_kernel<true><<<grid, V3_THREADS, v3_dshmem_q, stream>>>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                sc1_ptr, M, H, scale_stride, pargs);
        }

        // Quantize dh3 (queued after dh1 quantize on same stream)
        {
            cudaMemsetAsync(s_bwd_amax_buf + 1, 0, sizeof(float), stream);
            cudaMemsetAsync(s_bwd_sync_buf, 0, 4 * sizeof(unsigned int), stream);

            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(tmap_in, dh3_bf16.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
            create_tma_2d(tmap_out, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
            create_tma_2d(tmap_out_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

            alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            create_tma_2d(tmap_sc_row, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

            int v3_dshmem_q = ci.v3_dshmem_t;
            const int p_max_bps = ci.v4_max_bps_t;
            int num_persistent = p_max_bps * ci.num_sms;
            if (num_persistent > total_tiles) num_persistent = total_tiles;

            tk_v5::PersistentArgs pargs;
            memset(&pargs, 0, sizeof(pargs));
            pargs.work_counter_phase1 = s_bwd_sync_buf;
            pargs.work_counter_phase2 = s_bwd_sync_buf + 1;
            pargs.global_amax  = s_bwd_amax_buf + 1;
            pargs.done_counter = s_bwd_sync_buf + 2;
            pargs.ready_flag   = s_bwd_sync_buf + 3;
            pargs.tiles_X = tiles_X;
            pargs.tiles_Y = tiles_Y;
            pargs.total_tiles = total_tiles;
            pargs.num_persistent = num_persistent;
            pargs.sg_output = sg_ptr + 1;  // sg for dh3
            pargs.col_scales_ptr = sc2_t_ptr;
            pargs.col_scale_stride = scale_stride_t;
            pargs.swizzle_scales = true;

            const dim3 grid(num_persistent);
            tk_v5::persistent_quantize_kernel<true><<<grid, V3_THREADS, v3_dshmem_q, stream>>>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                sc2_ptr, M, H, scale_stride, pargs);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_quantize_for_gemm failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
        sg_buf.narrow(0, 0, 1), torch::zeros({1}, opts_f32),
        out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
        sg_buf.narrow(0, 1, 1), torch::zeros({1}, opts_f32)
    );
}

// ═══════════════════════════════════════════════════════════════════
// 6b. CUDA Graph-safe alloc/launch split for silu_deriv+quantize
// ═══════════════════════════════════════════════════════════════════

// _alloc: pre-create all output tensors + sync buffers. Call OUTSIDE graph capture.
// Returns: (out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
//           out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
//           sg_buf, amax_buf, sync_buf, dh1_bf16, dh3_bf16)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_for_gemm_alloc(int64_t M, int64_t H, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(device);

    auto out1_fp4   = torch::empty({M, H / 2}, opts_u8);
    auto out1_sc    = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto out1_fp4_t = torch::empty({H, M / 2}, opts_u8);
    auto out1_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_u8);
    auto out2_fp4   = torch::empty({M, H / 2}, opts_u8);
    auto out2_sc    = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto out2_fp4_t = torch::empty({H, M / 2}, opts_u8);
    auto out2_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_u8);
    auto sg_buf     = torch::empty({2}, opts_f32);
    auto amax_buf   = torch::empty({2}, opts_f32);
    auto sync_buf   = torch::empty({4}, opts_u32);
    // Intermediate bf16 buffers for large-grid (silu-deriv output)
    auto dh1_bf16   = torch::empty({M, H}, opts_bf16);
    auto dh3_bf16   = torch::empty({M, H}, opts_bf16);

    return std::make_tuple(out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
                           out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
                           sg_buf, amax_buf, sync_buf, dh1_bf16, dh3_bf16);
}

// _launch: kernel-only dispatch — NO allocations, safe inside CUDA graph capture.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_for_gemm_launch(
    torch::Tensor dh, torch::Tensor h13, int64_t H,
    // Pre-allocated buffers from _alloc:
    torch::Tensor out1_fp4, torch::Tensor out1_sc,
    torch::Tensor out1_fp4_t, torch::Tensor out1_sc_t,
    torch::Tensor out2_fp4, torch::Tensor out2_sc,
    torch::Tensor out2_fp4_t, torch::Tensor out2_sc_t,
    torch::Tensor sg_buf, torch::Tensor amax_buf,
    torch::Tensor sync_buf,
    torch::Tensor dh1_bf16, torch::Tensor dh3_bf16
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    const int64_t M = dh.size(0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = sg_buf.data_ptr<float>();
    unsigned int *sync_data = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());

    // Cached shmem size and occupancy for backward persistent kernel
    static int s_bwd_dshmem_g = -1;
    static int s_bwd_max_bps_g = -1;
    if (s_bwd_dshmem_g < 0) {
        s_bwd_dshmem_g = persistent_silu_deriv_quant::fused_silu_deriv_quant_smem_size<true>();
        cudaFuncSetAttribute(
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_bwd_dshmem_g);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_bwd_max_bps_g,
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            V3_THREADS, s_bwd_dshmem_g);
    }

    const auto& ci = get_cached_info();
    const int max_concurrent = s_bwd_max_bps_g * ci.num_sms;
    const bool can_fuse = (total_tiles <= max_concurrent && s_bwd_max_bps_g > 0);

    if (can_fuse) {
        // ─── PERSISTENT PATH ───
        cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);
        cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

        alignas(64) CUtensorMap tmap_dh{}, tmap_h1{}, tmap_h3{};
        create_tma_2d(tmap_dh, dh.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16);
        create_tma_2d(tmap_h1, h13.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);
        void* h3_ptr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h13.data_ptr()) + H * sizeof(__nv_bfloat16));
        create_tma_2d(tmap_h3, h3_ptr, M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);

        alignas(64) CUtensorMap tmap_out1{}, tmap_out2{};
        create_tma_2d(tmap_out1, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
        create_tma_2d(tmap_out2, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);

        alignas(64) CUtensorMap tmap_out1_t{}, tmap_out2_t{};
        create_tma_2d(tmap_out1_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
        create_tma_2d(tmap_out2_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        alignas(64) CUtensorMap tmap_sc_row1{}, tmap_sc_row2{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row1, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        create_tma_2d(tmap_sc_row2, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        alignas(64) CUtensorMap tmap_sc_col1{}, tmap_sc_col2{};
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col1, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        create_tma_2d(tmap_sc_col2, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

        int num_persistent = s_bwd_max_bps_g * ci.num_sms;
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = sync_data;
        pargs.work_counter_phase2 = sync_data + 1;
        pargs.global_amax  = amax_ptr;
        pargs.done_counter = sync_data + 2;
        pargs.ready_flag   = sync_data + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;

        const dim3 grid(num_persistent);
        persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true><<<grid, V3_THREADS, s_bwd_dshmem_g, stream>>>(
            tmap_dh, tmap_h1, tmap_h3,
            tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
            tmap_sc_row1, tmap_sc_row2, tmap_sc_col1, tmap_sc_col2,
            M, H, scale_stride, pargs, amax_ptr + 1);
    } else {
        // ─── LARGE GRID: silu-deriv + 2× v5 persistent quantize ───
        {
            const __nv_bfloat16* dh_ptr = reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr());
            const __nv_bfloat16* h13_ptr = reinterpret_cast<const __nv_bfloat16*>(h13.data_ptr());
            __nv_bfloat16* out1_ptr = reinterpret_cast<__nv_bfloat16*>(dh1_bf16.data_ptr());
            __nv_bfloat16* out2_ptr = reinterpret_cast<__nv_bfloat16*>(dh3_bf16.data_ptr());
            const int64_t total_pairs = (int64_t)M * H / 2;
            const int threads = 256;
            int blocks = (int)((total_pairs + threads - 1) / threads);
            if (blocks > 65535) blocks = 65535;
            silu_deriv_dual_strided_kernel<<<blocks, threads, 0, stream>>>(
                dh_ptr, h13_ptr, out1_ptr, out2_ptr, M, H);
        }

        nvfp4_scale_t *sc1_ptr   = reinterpret_cast<nvfp4_scale_t*>(out1_sc.data_ptr());
        nvfp4_scale_t *sc1_t_ptr = reinterpret_cast<nvfp4_scale_t*>(out1_sc_t.data_ptr());
        nvfp4_scale_t *sc2_ptr   = reinterpret_cast<nvfp4_scale_t*>(out2_sc.data_ptr());
        nvfp4_scale_t *sc2_t_ptr = reinterpret_cast<nvfp4_scale_t*>(out2_sc_t.data_ptr());

        // Quantize dh1
        {
            cudaMemsetAsync(amax_ptr, 0, sizeof(float), stream);
            cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(tmap_in, dh1_bf16.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
            create_tma_2d(tmap_out, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
            create_tma_2d(tmap_out_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

            alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            create_tma_2d(tmap_sc_row, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

            int v3_dshmem_q = ci.v3_dshmem_t;
            const int p_max_bps = ci.v4_max_bps_t;
            int num_persistent = p_max_bps * ci.num_sms;
            if (num_persistent > total_tiles) num_persistent = total_tiles;

            tk_v5::PersistentArgs pargs;
            memset(&pargs, 0, sizeof(pargs));
            pargs.work_counter_phase1 = sync_data;
            pargs.work_counter_phase2 = sync_data + 1;
            pargs.global_amax  = amax_ptr;
            pargs.done_counter = sync_data + 2;
            pargs.ready_flag   = sync_data + 3;
            pargs.tiles_X = tiles_X;
            pargs.tiles_Y = tiles_Y;
            pargs.total_tiles = total_tiles;
            pargs.num_persistent = num_persistent;
            pargs.sg_output = sg_ptr;
            pargs.col_scales_ptr = sc1_t_ptr;
            pargs.col_scale_stride = scale_stride_t;
            pargs.swizzle_scales = true;

            const dim3 grid(num_persistent);
            tk_v5::persistent_quantize_kernel<true><<<grid, V3_THREADS, v3_dshmem_q, stream>>>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                sc1_ptr, M, H, scale_stride, pargs);
        }

        // Quantize dh3
        {
            cudaMemsetAsync(amax_ptr + 1, 0, sizeof(float), stream);
            cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(tmap_in, dh3_bf16.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
            create_tma_2d(tmap_out, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
            create_tma_2d(tmap_out_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

            alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            create_tma_2d(tmap_sc_row, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

            int v3_dshmem_q = ci.v3_dshmem_t;
            const int p_max_bps = ci.v4_max_bps_t;
            int num_persistent = p_max_bps * ci.num_sms;
            if (num_persistent > total_tiles) num_persistent = total_tiles;

            tk_v5::PersistentArgs pargs;
            memset(&pargs, 0, sizeof(pargs));
            pargs.work_counter_phase1 = sync_data;
            pargs.work_counter_phase2 = sync_data + 1;
            pargs.global_amax  = amax_ptr + 1;
            pargs.done_counter = sync_data + 2;
            pargs.ready_flag   = sync_data + 3;
            pargs.tiles_X = tiles_X;
            pargs.tiles_Y = tiles_Y;
            pargs.total_tiles = total_tiles;
            pargs.num_persistent = num_persistent;
            pargs.sg_output = sg_ptr + 1;
            pargs.col_scales_ptr = sc2_t_ptr;
            pargs.col_scale_stride = scale_stride_t;
            pargs.swizzle_scales = true;

            const dim3 grid(num_persistent);
            tk_v5::persistent_quantize_kernel<true><<<grid, V3_THREADS, v3_dshmem_q, stream>>>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                sc2_ptr, M, H, scale_stride, pargs);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_quantize_for_gemm_launch failed: ", cudaGetErrorString(err));

    auto opts_f32 = torch::dtype(torch::kFloat32).device(dh.device());
    return std::make_tuple(
        out1_fp4.view(torch::kFloat4_e2m1fn_x2), out1_sc.view(torch::kFloat8_e4m3fn),
        out1_fp4_t.view(torch::kFloat4_e2m1fn_x2), out1_sc_t.view(torch::kFloat8_e4m3fn),
        sg_buf.narrow(0, 0, 1), torch::zeros({1}, opts_f32),
        out2_fp4.view(torch::kFloat4_e2m1fn_x2), out2_sc.view(torch::kFloat8_e4m3fn),
        out2_fp4_t.view(torch::kFloat4_e2m1fn_x2), out2_sc_t.view(torch::kFloat8_e4m3fn),
        sg_buf.narrow(0, 1, 1), torch::zeros({1}, opts_f32)
    );
}


// ═══════════════════════════════════════════════════════════════════
// Pybind11 bindings
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tk_quantize_for_gemm", &tk_v4_quantize_for_gemm,
          "v5 hybrid: quantize for GEMM (TMA scale output)",
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_quantize_transpose", &tk_quantize_transpose,
          "quantize with pre-computed amax",
          py::arg("input"), py::arg("amax_row"), py::arg("amax_col"), py::arg("return_transpose"));
    m.def("tk_group_quantize_for_gemm", &tk_group_quantize_for_gemm,
          "Grouped NVFP4 quantise — per-split amax (dim=0)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_for_gemm_v2", &tk_group_quantize_for_gemm_v2,
          "Grouped NVFP4 quantise — non-persistent two-pass (multi-stream safe)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_v2_alloc", &tk_group_quantize_v2_alloc,
          "Pre-allocate buffers for v2 grouped quant (call on capture stream)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_v2_launch", &tk_group_quantize_v2_launch,
          "Kernel-only v2 grouped quant using pre-allocated buffers (graph-safe on any stream)",
          py::arg("input"), py::arg("split_sections"),
          py::arg("wc_fp4_row"), py::arg("amaxes"), py::arg("sg_cat"),
          py::arg("fwd_b_sg"), py::arg("dgrad_b_sg"), py::arg("d_offsets"),
          py::arg("sc_row_list"), py::arg("fp4_col_list"), py::arg("sc_col_list"));
    m.def("tk_group_quantize_v5_alloc", &tk_group_quantize_v5_alloc,
          "Pre-allocate buffers for v5 persistent/fused grouped quant (call on capture stream)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_v5_launch", &tk_group_quantize_v5_launch,
          "Kernel-only v5 persistent/fused grouped quant using pre-allocated buffers (graph-safe on any stream)",
          py::arg("input"), py::arg("split_sections"),
          py::arg("wc_fp4_row"), py::arg("wc_fp4_col"), py::arg("sg_cat"),
          py::arg("fwd_b_sg"), py::arg("dgrad_b_sg"),
          py::arg("amax_tensor"), py::arg("sync_tensor"), py::arg("psync_tensor"),
          py::arg("tma_dev_buf"),
          py::arg("sc_row_list"), py::arg("fp4_col_list"), py::arg("sc_col_list"));
    m.def("tk_group_quantize_dim1_for_gemm", &tk_group_quantize_dim1_for_gemm,
          "Grouped NVFP4 quantise — per-column-group amax (dim=1). Returns: "
          "(fp4_row_list, sc_row_list, sg, fp4_col_list, sc_col_list, fp4_row_full, sc_row_cat, fp4_col_full, sc_col_cat)",
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_group_quantize_dim1_alloc", &tk_group_quantize_dim1_alloc,
          "Pre-allocate buffers for dim1 grouped quant (call BEFORE graph capture)",
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_group_quantize_dim1_launch", &tk_group_quantize_dim1_launch,
          "Kernel-only dim1 grouped quant using pre-allocated buffers (graph-safe inside capture)",
          py::arg("input"), py::arg("col_split_sections"),
          py::arg("fp4_row_full"), py::arg("fp4_col_full"), py::arg("sg_per_group"),
          py::arg("amax_tensor"), py::arg("sync_tensor"), py::arg("psync_tensor"),
          py::arg("tma_host_buf"), py::arg("tma_dev_buf"),
          py::arg("sc_row_allocs"), py::arg("fp4_col_allocs"), py::arg("sc_col_allocs"),
          py::arg("skip_cat") = false);
    m.def("tk_fused_norm_quantize", &tk_fused_norm_quantize,
          "Fused RMSNorm + optional SiLU + FP4 quantize (single GMEM pass)",
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("with_silu"), py::arg("return_transpose"));
    m.def("tk_fused_norm_quantize_naive", &tk_fused_norm_quantize_naive,
          "Fused RMSNorm + optional SiLU + FP4 quantize using scalar absmax tracking",
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("with_silu"), py::arg("return_transpose"));
    m.def("tk_fused_norm_quantize_ilp", &tk_fused_norm_quantize_ilp,
          "Fused RMSNorm + optional SiLU + FP4 quantize using packed absmax tracking",
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("with_silu"), py::arg("return_transpose"));
    m.def("tk_reconstruct_row", &tk_reconstruct_row,
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("sg"));
    m.def("tk_reconstruct_col", &tk_reconstruct_col,
          py::arg("col_fp4"), py::arg("col_sc"), py::arg("sg"));
    m.def("tk_silu_quantize_for_gemm", &tk_silu_quantize_for_gemm,
          "Fused silu(h1)*h3 + FP4 quantize from (M,2H) buffer",
          py::arg("h13"), py::arg("H"));
    m.def("tk_silu_deriv_quantize_for_gemm", &tk_silu_deriv_quantize_for_gemm,
          "Fused SiLU deriv + dual FP4 quantize for backward pass",
          py::arg("dh"), py::arg("h13"), py::arg("H"));

    // ── CUDA graph-safe alloc/launch split APIs ──
    m.def("tk_v4_quantize_for_gemm_alloc", &tk_v4_quantize_for_gemm_alloc,
          "Pre-allocate buffers for v4 quantize (call BEFORE graph capture)",
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_v4_quantize_for_gemm_launch", &tk_v4_quantize_for_gemm_launch,
          "Kernel-only v4 quantize using pre-allocated buffers (graph-safe)",
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric") = true,
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("amax_buf"), py::arg("sync_buf"));
    m.def("tk_silu_deriv_quantize_for_gemm_alloc", &tk_silu_deriv_quantize_for_gemm_alloc,
          "Pre-allocate buffers for silu_deriv+quantize (call BEFORE graph capture)",
          py::arg("M"), py::arg("H"), py::arg("device"));
    m.def("tk_silu_deriv_quantize_for_gemm_launch", &tk_silu_deriv_quantize_for_gemm_launch,
          "Kernel-only silu_deriv+quantize using pre-allocated buffers (graph-safe)",
          py::arg("dh"), py::arg("h13"), py::arg("H"),
          py::arg("out1_fp4"), py::arg("out1_sc"),
          py::arg("out1_fp4_t"), py::arg("out1_sc_t"),
          py::arg("out2_fp4"), py::arg("out2_sc"),
          py::arg("out2_fp4_t"), py::arg("out2_sc_t"),
          py::arg("sg_buf"), py::arg("amax_buf"),
          py::arg("sync_buf"),
          py::arg("dh1_bf16"), py::arg("dh3_bf16"));
}

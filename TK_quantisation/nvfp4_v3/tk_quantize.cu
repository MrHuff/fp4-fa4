/*
 * TK v3 — Full dispatch layer
 *
 * ALL quantize kernels use fused single-pass amax:
 *   1. tk_v3_quantize_for_gemm    — fused single-pass (spin-wait barrier)
 *   2. tk_quantize_transpose      — pre-computed amax (v2 kernel, no fusion needed)
 *   3. tk_group_quantize_for_gemm — fused per-split amax (dim=0)
 *   4. tk_group_quantize_dim1_for_gemm — fused per-group amax (dim=1)
 *
 * Also re-exports as tk_quantize_for_gemm for backward compat.
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
#include "amax_pipelined.cuh"
#include "group_quantize_transpose.cuh"
#include "group_quantize_transpose_dim_1.cuh"

namespace grp_kernel = transformer_engine::dispatch::nvfp4::group_quantize_transpose_kernel;
namespace grp_dim1_kernel = transformer_engine::dispatch::nvfp4::group_quantize_transpose_dim1_kernel;

using namespace transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel;
using namespace transformer_engine;

// ═══════════════════════════════════════════════════════════════════
// v2 fallback launch helpers for grouped kernels
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
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
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
    cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    func<<<grid, grp_dim1_kernel::THREADS_NUM, shmem, stream>>>(
        tmap_in, tmap_out, sc_ptr, nullptr, rows, cols, scale_stride,
        nullptr, kernel_args);
}

// Simple kernel: compute sg[i] = amax[i] / 2688.0 for each group
__global__ void compute_sg_kernel(const float* __restrict__ amaxes,
                                   float* __restrict__ sgs,
                                   int num) {
    int i = threadIdx.x;
    if (i < num) sgs[i] = amaxes[i] / 2688.0f;
}

// ─────────────────────── TMA tensor map creation ────────────────
static void create_tma_2d(
    CUtensorMap &map, void *ptr,
    uint64_t globalY, uint64_t globalX,
    uint32_t shmemY, uint32_t shmemX,
    uint64_t strideX, size_t type_num_bits
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
    else if (type_num_bits == 4) dataType = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    else TORCH_CHECK(false, "Unsupported type_num_bits: ", type_num_bits);

    auto result = fn(&map, dataType, 2, ptr,
        globalDims, globalStrides, boxDims, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}


// ═══════════════════════════════════════════════════════════════════
// v2 launch helper (for tk_quantize_transpose only)
// ═══════════════════════════════════════════════════════════════════

template <bool SR, bool FM, bool RT>
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

    auto kernel = quantize_transpose_nvfp4_tuned_1D_kernel<SR, FM, RT>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<grid, THREADS_NUM, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr,
        nullptr, amax_row, amax_col, M, K,
        scale_stride, scale_stride_t, nullptr, true);
}


// ═══════════════════════════════════════════════════════════════════
// v3 launch helpers
// ═══════════════════════════════════════════════════════════════════

template <bool RT>
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

    auto kernel = fused_amax_quantize_kernel<RT>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<grid, V3_THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        sc_ptr, sc_t_ptr, global_amax, sg_out,
        done_counter, ready_flag,
        M, K, scale_stride, scale_stride_t, total_blocks);
}


// ═══════════════════════════════════════════════════════════════════
// 1. tk_v3_quantize_for_gemm — Fused single-pass
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v3_quantize_for_gemm(torch::Tensor input, bool return_transpose) {
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

    auto sync_buf = torch::empty({2}, opts_u32);
    unsigned int *done_counter = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    unsigned int *ready_flag = done_counter + 1;

    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(done_counter, 0, 2 * sizeof(unsigned int), stream);

    using namespace tk_v3;
    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    if (return_transpose)
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    // Check occupancy for spin-wait
    const int blocks_Y = (M + V3Config::CHUNK_DIM_Y - 1) / V3Config::CHUNK_DIM_Y;
    const int blocks_X = (K + V3Config::CHUNK_DIM_X - 1) / V3Config::CHUNK_DIM_X;
    const int total_blocks = blocks_X * blocks_Y;

    int max_blocks_per_sm = 0;
    {
        const int dshmem = return_transpose ? v3_shmem_size<true>() : v3_shmem_size<false>();
        auto kernel = return_transpose ?
            (void*)fused_amax_quantize_kernel<true> :
            (void*)fused_amax_quantize_kernel<false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            return_transpose ? fused_amax_quantize_kernel<true> : fused_amax_quantize_kernel<false>,
            V3_THREADS, dshmem);
    }

    int dev; cudaGetDevice(&dev);
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    const int max_concurrent = max_blocks_per_sm * num_sms;
    const bool use_v3 = (total_blocks <= max_concurrent && max_blocks_per_sm > 0);

    if (use_v3) {
        if (return_transpose)
            launch_v3<true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done_counter, ready_flag, M, K, scale_stride, scale_stride_t, stream);
        else
            launch_v3<false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done_counter, ready_flag, M, K, scale_stride, scale_stride_t, stream);
    } else {
        // v2 fallback
        int64_t n = M * K;
        int amax_blocks = pipelined_amax::grid_size(n);
        int amax_smem = pipelined_amax::smem_size();
        cudaFuncSetAttribute(pipelined_amax::fused_amax_pipelined_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, amax_smem);
        pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), amax_ptr, sg_ptr, n);

        alignas(64) CUtensorMap ti{}, to{}, tot{};
        create_tma_2d(ti, input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
        create_tma_2d(to, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
        if (return_transpose)
            create_tma_2d(tot, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

        if (return_transpose)
            launch_v2_kernel<false, false, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
        else
            launch_v2_kernel<false, false, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_v3_quantize_for_gemm failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
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

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

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
// 3. tk_group_quantize_for_gemm — FUSED per-split amax (dim=0)
// ═══════════════════════════════════════════════════════════════════

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

    // FP4 row output (contiguous)
    auto wc_fp4_row = torch::empty({total_rows, K / 2}, opts_fp4);

    // FP4 col output — need TMA map, so we use a single contiguous buffer
    // then TMA writes transposed data directly
    // For transpose, we need per-split col buffers
    auto wc_fp4_col = torch::empty({K, total_rows / 2}, opts_u8);

    auto sg_cat     = torch::empty({N}, opts_f32);
    auto fwd_b_sg   = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad_tiles}, opts_f32);

    // Sync buffers: per-group amax + done_counter + ready_flag
    auto amax_tensor = torch::empty({N}, opts_f32);
    auto sync_tensor = torch::empty({2 * N}, opts_u32);
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, N * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * N * sizeof(unsigned int), stream);

    // Per-split scale allocations
    std::vector<torch::Tensor> sc_row_allocs(N);
    std::vector<torch::Tensor> fp4_col_list(N), sc_col_allocs(N);

    // Build FusedGroupArgs
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

    const int blocks_X = K / tk_v3::V3Config::CHUNK_DIM_X;

    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        args.split_range[i + 1] = args.split_range[i] + (int)M_i;
        args.blocks_per_group[i] = (int)(M_i / tk_v3::V3Config::CHUNK_DIM_Y) * blocks_X;

        // Row scales
        sc_row_allocs[i] = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        args.row_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[i].data_ptr());

        // Col FP4 + scales
        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        args.col_data_ptrs[i] = reinterpret_cast<fp4e2m1x2*>(fp4_col_list[i].data_ptr());

        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        sc_col_allocs[i] = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
        args.col_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[i].data_ptr());
        args.col_scale_stride[i] = (int)c_sc_stride;
    }

    // Occupancy check: total grid must fit concurrently
    using namespace tk_v3;
    const int blocks_Y = total_rows / V3Config::CHUNK_DIM_Y;
    const int total_grid = blocks_X * blocks_Y;
    const int dshmem = v3_shmem_size<true>();

    int max_blocks_per_sm = 0;
    cudaFuncSetAttribute(fused_group_quantize_kernel_dim0<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fused_group_quantize_kernel_dim0<true>, V3_THREADS, dshmem);
    int dev; cudaGetDevice(&dev);
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    const int max_concurrent = max_blocks_per_sm * num_sms;
    const bool use_fused = (total_grid <= max_concurrent && max_blocks_per_sm > 0);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(),
                      total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(),
                      total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(),
                      K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        const dim3 grid(blocks_X, blocks_Y);
        auto kernel = fused_group_quantize_kernel_dim0<true>;
        kernel<<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr, nullptr,
            total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
            args);
    } else {
        // v2 fallback: pipelined amax + grouped kernel
        auto amaxes = torch::zeros({N}, opts_f32);
        {
            std::vector<int64_t> h_offsets(N + 1);
            h_offsets[0] = 0;
            for (int i = 0; i < N; ++i)
                h_offsets[i + 1] = h_offsets[i] + split_sections[i] * K;
            auto d_offsets = torch::empty({N + 1}, torch::dtype(torch::kInt64).device(device));
            cudaMemcpyAsync(d_offsets.data_ptr<int64_t>(), h_offsets.data(),
                            (N + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
            int64_t n = total_rows * K;
            int grp_blocks = pipelined_amax::grid_size(n);
            int grp_smem = pipelined_amax::smem_size();
            cudaFuncSetAttribute(pipelined_amax::grouped_amax_pipelined_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, grp_smem);
            pipelined_amax::grouped_amax_pipelined_kernel<<<grp_blocks, pipelined_amax::THREADS, grp_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                amaxes.data_ptr<float>(), d_offsets.data_ptr<int64_t>(), N, n);
        }

        // Build v2 kernel args
        grp_kernel::MultiAmaxCastTransposeFusionArgs v2_args;
        memset(&v2_args, 0, sizeof(v2_args));
        v2_args.num_tensors = N;
        v2_args.swizzle_scales = true;
        v2_args.sg_output = sg_cat.data_ptr<float>();
        v2_args.fwd_b_sg = fwd_b_sg.data_ptr<float>();
        v2_args.dgrad_b_sg = dgrad_b_sg.data_ptr<float>();
        v2_args.b_tile_size = (int)Nb;
        v2_args.total_cols = (int)K;
        v2_args.split_sections_range[0] = 0;

        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            v2_args.split_sections_range[i + 1] = v2_args.split_sections_range[i] + (int)M_i;
            v2_args.rowwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
            v2_args.colwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
            v2_args.output_rowwise_scale_inv_list[i] = sc_row_allocs[i].data_ptr();
            v2_args.output_colwise_data_list[i] = fp4_col_list[i].data_ptr();
            const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
            v2_args.output_colwise_scale_inv_list[i] = sc_col_allocs[i].data_ptr();
            v2_args.output_colwise_scale_stride[i] = (int)c_sc_stride;
        }

        alignas(64) CUtensorMap tmap_in{}, tmap_out{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K,
                      grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 16);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K,
                      grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 4);

        nvfp4_scale_t* v2_sc = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[0].data_ptr());
        launch_group_kernel_v2<false, true>(tmap_in, tmap_out, v2_sc,
                                           total_rows, K, scale_stride, v2_args, stream);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_for_gemm failed: ", cudaGetErrorString(err));

    // Reshape outputs
    auto wc_sc_row_parts = std::vector<torch::Tensor>(N);
    for (int i = 0; i < N; ++i) {
        wc_sc_row_parts[i] = sc_row_allocs[i].view(torch::kFloat8_e4m3fn);
    }
    auto wc_sc_row = torch::cat(wc_sc_row_parts, 0);

    std::vector<torch::Tensor> sc_col_list(N);
    if (use_fused) {
        // Fused path: TMA wrote transpose to contiguous wc_fp4_col [K, total_rows/2]
        // Split along dim=1 (rows become columns in transpose)
        int64_t row_offset = 0;
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            // In the transposed buffer, split i occupies columns [row_offset/2, row_offset/2 + M_i/2)
            fp4_col_list[i] = wc_fp4_col.narrow(1, row_offset / 2, M_i / 2)
                                         .contiguous()
                                         .view(torch::kFloat4_e2m1fn_x2);
            sc_col_list[i] = sc_col_allocs[i].view(torch::kFloat8_e4m3fn);
            row_offset += M_i;
        }
    } else {
        // Fallback: v2 kernel wrote directly to per-split fp4_col_list[i]
        for (int i = 0; i < N; ++i) {
            fp4_col_list[i] = fp4_col_list[i].view(torch::kFloat4_e2m1fn_x2);
            sc_col_list[i] = sc_col_allocs[i].view(torch::kFloat8_e4m3fn);
        }
    }

    auto mega_buf = torch::empty({0}, opts_u8);
    return std::make_tuple(wc_fp4_row, wc_sc_row, fwd_b_sg,
                           fp4_col_list, sc_col_list, dgrad_b_sg, sg_cat, mega_buf);
}


// ═══════════════════════════════════════════════════════════════════
// 4. tk_group_quantize_dim1_for_gemm — FUSED per-column-group amax
// ═══════════════════════════════════════════════════════════════════

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

    // Sync buffers
    auto amax_tensor = torch::empty({G}, opts_f32);
    auto sync_tensor = torch::empty({2 * G}, opts_u32);
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, G * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * G * sizeof(unsigned int), stream);

    auto sg_per_group = torch::empty({G}, opts_f32);

    // Contiguous row FP4 output (for TMA) — allocate as u8 since fp4 copy_ not implemented
    auto fp4_row_full = torch::empty({M, N_total / 2}, opts_u8);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;

    // Build args
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

        // Row scales per group
        const int64_t ntk_r_g = N_g / 64;
        sc_row_allocs[g] = torch::empty({ntm_r, ntk_r_g, 512}, opts_u8);
        args.row_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[g].data_ptr());

        // Col FP4 per group
        fp4_col_allocs[g] = torch::empty({N_g, M / 2}, opts_u8);
        args.col_data_ptrs[g] = reinterpret_cast<fp4e2m1x2*>(fp4_col_allocs[g].data_ptr());

        // Col scales per group
        const int64_t ntm_c_g = N_g / 128;
        const int64_t col_sc_stride = ((M / 16) + 3) / 4 * 4;
        sc_col_allocs[g] = torch::zeros({ntm_c_g, ntk_c, 512}, opts_u8);
        args.col_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[g].data_ptr());
        args.col_scale_stride[g] = (int)col_sc_stride;
    }

    // Contiguous transposed FP4 output (for TMA) — shape [N_total, M/2]
    auto fp4_col_full = torch::empty({N_total, M / 2}, opts_u8);

    // Occupancy check
    const int blocks_X = N_total / V3Config::CHUNK_DIM_X;
    const int total_grid = blocks_X * blocks_Y;
    const int dshmem = v3_shmem_size<true>();

    int max_blocks_per_sm = 0;
    cudaFuncSetAttribute(fused_group_quantize_kernel_dim1<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fused_group_quantize_kernel_dim1<true>, V3_THREADS, dshmem);
    int dev; cudaGetDevice(&dev);
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    const int max_concurrent = max_blocks_per_sm * num_sms;
    const bool use_fused = (total_grid <= max_concurrent && max_blocks_per_sm > 0);

    const int64_t global_scale_stride = ((N_total / 16) + 3) / 4 * 4;

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total,
                      V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total,
                      V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M,
                      V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        const dim3 grid(blocks_X, blocks_Y);
        auto kernel = fused_group_quantize_kernel_dim1<true>;
        kernel<<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
            args);
    } else {
        // v2 fallback: pipelined dim1 amax + v2 dim1 kernel
        {
            std::vector<int> h_range(G + 1);
            h_range[0] = 0;
            for (int i = 0; i < G; ++i) h_range[i + 1] = h_range[i] + (int)col_split_sections[i];
            auto d_range = torch::empty({G + 1}, torch::dtype(torch::kInt32).device(device));
            cudaMemcpyAsync(d_range.data_ptr<int>(), h_range.data(),
                            (G + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
            int64_t n = M * N_total;
            int dim1_blocks = pipelined_amax::grid_size(n);
            int dim1_smem = pipelined_amax::smem_size();
            cudaFuncSetAttribute(pipelined_amax::grouped_amax_dim1_pipelined_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, dim1_smem);
            pipelined_amax::grouped_amax_dim1_pipelined_kernel<<<dim1_blocks, pipelined_amax::THREADS, dim1_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                amax_tensor.data_ptr<float>(), d_range.data_ptr<int>(), G, n, (int)N_total);
        }
        compute_sg_kernel<<<1, G, 0, stream>>>(
            amax_tensor.data_ptr<float>(), sg_per_group.data_ptr<float>(), G);

        grp_dim1_kernel::Dim1GroupArgs v2_args;
        memset(&v2_args, 0, sizeof(v2_args));
        v2_args.num_groups = G;
        v2_args.swizzle_scales = true;
        v2_args.sg_output = sg_per_group.data_ptr<float>();
        v2_args.col_split_range[0] = 0;

        for (int g = 0; g < G; ++g) {
            const int64_t N_g = col_split_sections[g];
            v2_args.col_split_range[g + 1] = v2_args.col_split_range[g] + (int)N_g;
            v2_args.rowwise_amax_list[g] = (void*)(amax_tensor.data_ptr<float>() + g);
            v2_args.colwise_amax_list[g] = (void*)(amax_tensor.data_ptr<float>() + g);
            v2_args.output_rowwise_scale_inv_list[g] = sc_row_allocs[g].data_ptr();
            v2_args.output_rowwise_scale_stride[g] = (int)(((N_g / 16) + 3) / 4 * 4);
            v2_args.output_colwise_data_list[g] = fp4_col_allocs[g].data_ptr();
            const int64_t col_sc_stride = ((M / 16) + 3) / 4 * 4;
            v2_args.output_colwise_scale_inv_list[g] = sc_col_allocs[g].data_ptr();
            v2_args.output_colwise_scale_stride[g] = (int)col_sc_stride;
        }

        alignas(64) CUtensorMap tmap_in{}, tmap_out{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total,
                      grp_dim1_kernel::BUFF_DIM_Y, grp_dim1_kernel::BUFF_DIM_X, N_total, 16);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total,
                      grp_dim1_kernel::BUFF_DIM_Y, grp_dim1_kernel::BUFF_DIM_X, N_total, 4);

        nvfp4_scale_t* v2_sc = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[0].data_ptr());
        launch_dim1_group_kernel_v2<false, true>(tmap_in, tmap_out, v2_sc,
                                                 M, N_total, global_scale_stride,
                                                 v2_args, stream);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_dim1_for_gemm failed: ", cudaGetErrorString(err));

    // Reshape outputs
    std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
    std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];

        fp4_row_list[g] = fp4_row_full.narrow(1, col_offset / 2, N_g / 2).contiguous().view(torch::kFloat4_e2m1fn_x2);
        sc_row_list[g] = sc_row_allocs[g].view(torch::kFloat8_e4m3fn);

        if (use_fused) {
            // Fused: TMA wrote transpose to contiguous fp4_col_full [N_total, M/2]
            // Split along dim=0 (rows correspond to column groups)
            fp4_col_list[g] = fp4_col_full.narrow(0, col_offset, N_g).contiguous().view(torch::kFloat4_e2m1fn_x2);
        } else {
            // Fallback: v2 kernel wrote directly to per-group fp4_col_allocs
            fp4_col_list[g] = fp4_col_allocs[g].view(torch::kFloat4_e2m1fn_x2);
        }
        sc_col_list[g] = sc_col_allocs[g].view(torch::kFloat8_e4m3fn);

        col_offset += N_g;
    }

    return std::make_tuple(fp4_row_list, sc_row_list, sg_per_group,
                           fp4_col_list, sc_col_list);
}


// ═══════════════════════════════════════════════════════════════════
// PYBIND11 module
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tk_v3_quantize_for_gemm", &tk_v3_quantize_for_gemm,
          "v3 Fused Single-Pass Amax+Quantize (one HBM read).",
          py::arg("input"), py::arg("return_transpose") = true);

    m.def("tk_quantize_for_gemm", &tk_v3_quantize_for_gemm,
          "Alias for tk_v3_quantize_for_gemm (backward compat)",
          py::arg("input"), py::arg("return_transpose") = true);

    m.def("tk_quantize_transpose", &tk_quantize_transpose,
          "NVFP4 quantise+transpose with pre-computed amax.",
          py::arg("input"), py::arg("amax_row"), py::arg("amax_col"),
          py::arg("return_transpose") = true);

    m.def("tk_group_quantize_for_gemm", &tk_group_quantize_for_gemm,
          "Fused grouped NVFP4 quantise — per-split amax (single HBM read per element).",
          py::arg("input"), py::arg("split_sections"));

    m.def("tk_group_quantize_dim1_for_gemm", &tk_group_quantize_dim1_for_gemm,
          "Fused dim=1 grouped NVFP4 quantise — per-column-group amax (single HBM read).",
          py::arg("input"), py::arg("col_split_sections"));
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <tuple>
#include <vector>

#define TK_STANDALONE
#include "fused_localcta_quantize.cuh"
#include "localcta_reconstruct.cuh"
#include "../nvfp4_v5/silu_split_bf16.cuh"

using transformer_engine::dispatch::nvfp4::nvfp4_scale_t;
namespace py = pybind11;

namespace {

struct LocalCTACachedInfo {
    int num_sms = 0;
    int max_bps = 0;
    int max_bps_t = 0;
    bool initialized = false;
};

static LocalCTACachedInfo& get_localcta_cached_info() {
    static LocalCTACachedInfo info;
    if (!info.initialized) {
        using namespace tk_localcta;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&info.num_sms, cudaDevAttrMultiProcessorCount, dev);

        const int dshmem = shmem_size<false>();
        auto kernel = fused_localcta_quantize_kernel<false, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.max_bps, kernel, THREADS, dshmem);

        const int dshmem_t = shmem_size<true>();
        auto kernel_t = fused_localcta_quantize_kernel<true, true>;
        cudaFuncSetAttribute(kernel_t, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.max_bps_t, kernel_t, THREADS, dshmem_t);

        info.initialized = true;
    }
    return info;
}

struct LocalCTA2PreparedTuning {
    int threads = 160;
    int pipe_depth = 1;
    bool shared_amax = false;
};

struct LocalCTA1PreparedTuning {
    int threads = 160;
    int pipe_depth = 1;
};

static LocalCTA2PreparedTuning& get_localcta2_prepared_tuning() {
    static LocalCTA2PreparedTuning tuning;
    return tuning;
}

static LocalCTA1PreparedTuning& get_localcta1_prepared_tuning() {
    static LocalCTA1PreparedTuning tuning;
    return tuning;
}

static float& get_localcta_global_scale_num_host() {
    static float value = tk_localcta::LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM;
    return value;
}

void tk_localcta_set_global_scale_num(float value) {
    TORCH_CHECK(std::isfinite(value) && value > 0.0f, "global scale number must be finite and > 0");
    auto err = cudaMemcpyToSymbol(tk_localcta::kLocalCTAGlobalScaleNum, &value, sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for localCTA global scale num: ",
                cudaGetErrorString(err));
    get_localcta_global_scale_num_host() = value;
}

float tk_localcta_get_global_scale_num() {
    return get_localcta_global_scale_num_host();
}

void tk_localcta_reset_global_scale_num() {
    tk_localcta_set_global_scale_num(tk_localcta::LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM);
}

static bool should_use_localcta2_prepared_auto(int64_t M, int64_t K) {
    using namespace tk_localcta;
    const int blocks_Y = static_cast<int>((M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y);
    const int blocks_X = static_cast<int>((K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X);
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    if (total_macro_tiles <= 0) {
        return false;
    }
    if (total_macro_tiles <= 1024) {
        return true;
    }
    // Narrow-K shapes keep the X dimension small enough that 2CTA remains
    // competitive well past the original crossover.
    if (blocks_X <= 16) {
        return true;
    }
    // The transition band just above the old cutoff still favors 2CTA in
    // several measured forward and dgrad cases, so widen it slightly.
    return total_macro_tiles <= 1152;
}

static bool should_use_localcta1_prepared_auto(int64_t M, int64_t K, bool return_transpose) {
    if (return_transpose) {
        return false;
    }
    using namespace tk_localcta;
    const int blocks_Y = static_cast<int>((M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y);
    const int blocks_X = static_cast<int>((K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X);
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    return total_macro_tiles > 1024;
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH, bool SHARED_AMAX>
static void launch_localcta_quant_2cta_prepared_tuned(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    if (total_macro_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_2cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_kernel_2cta_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, SHARED_AMAX, RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int num_sms = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_clusters = max_bps * num_sms / 2;
    if (num_clusters > total_macro_tiles) {
        num_clusters = total_macro_tiles;
    }
    if (num_clusters <= 0) {
        num_clusters = 1;
    }

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.preferredClusterDim.x = 2;
    attrs[0].val.preferredClusterDim.y = 1;
    attrs[0].val.preferredClusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 2;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_clusters * 2, 1, 1);
    config.blockDim = dim3(TOTAL_THREADS, 1, 1);
    config.dynamicSmemBytes = dshmem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;

    auto err = cudaLaunchKernelEx(
        &config,
        kernel,
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, K, blocks_X, blocks_Y, total_macro_tiles);
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchKernelEx failed for tuned localCTA 2-CTA prepared quant: ",
                cudaGetErrorString(err));
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_quant_2cta_prepared_tuned_dispatch(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    cudaStream_t stream
) {
    const auto cfg = get_localcta2_prepared_tuning();
    if (cfg.threads == 160 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 1 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 1, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 2 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 2, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 3 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 3, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 4 && !cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 4, false>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 384 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 384, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 1 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 1, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 2 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 2, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 3 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 3, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 512 && cfg.pipe_depth == 4 && cfg.shared_amax) {
        launch_localcta_quant_2cta_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 512, 4, true>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else {
        TORCH_CHECK(false, "Unsupported localCTA2 prepared tuning config: threads=", cfg.threads,
                    " pipe_depth=", cfg.pipe_depth, " shared_amax=", cfg.shared_amax);
    }
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC, int TOTAL_THREADS, int PIPE_DEPTH>
static void launch_localcta_quant_prepared_tuned(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    if (total_tiles <= 0) {
        return;
    }

    const int dshmem = prepared_1cta_tuned_shmem_size<PIPE_DEPTH, RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_kernel_prepared_tuned<
        TOTAL_THREADS, PIPE_DEPTH, RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    int num_sms = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps, kernel, TOTAL_THREADS, dshmem);
    int num_persistent = max_bps * num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    kernel<<<num_persistent, TOTAL_THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, K, blocks_X, total_tiles);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_quant_prepared_tuned_dispatch(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    int64_t M,
    int64_t K,
    cudaStream_t stream
) {
    const auto cfg = get_localcta1_prepared_tuning();
    if (cfg.threads == 160 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 1>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 160 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 160, 2>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 1>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 192 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 192, 2>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 1) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 1>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else if (cfg.threads == 256 && cfg.pipe_depth == 2) {
        launch_localcta_quant_prepared_tuned<RETURN_TRANSPOSE, ENCODE_CENTRIC, 256, 2>(
            tmap_in, tmap_out, tmap_out_t, tmap_sc_row_prepared, tmap_sc_col_prepared,
            row_sg_ptr, col_sg_ptr, M, K, stream);
    } else {
        TORCH_CHECK(false, "Unsupported localCTA1 prepared tuning config: threads=", cfg.threads,
                    " pipe_depth=", cfg.pipe_depth);
    }
}

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
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(
            dlsym(handle, "cuTensorMapEncodeTiled"));
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

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_quant(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    unsigned int *work_counter,
    int64_t M,
    int64_t K,
    bool write_raw_scales,
    bool write_prepared,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int total_tiles = blocks_X * blocks_Y;
    const int dshmem = shmem_size<RETURN_TRANSPOSE>();
    auto &ci = get_localcta_cached_info();
    int num_persistent = (RETURN_TRANSPOSE ? ci.max_bps_t : ci.max_bps) * ci.num_sms;
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    if (num_persistent <= 0) {
        num_persistent = 1;
    }

    LocalCTAPersistentArgs args {
        .work_counter = work_counter,
        .tiles_X = blocks_X,
        .total_tiles = total_tiles
    };

    auto kernel = fused_localcta_quantize_kernel<RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<num_persistent, THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr,
        M, K, args, write_raw_scales, write_prepared);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static void launch_localcta_quant_2cta(
    const CUtensorMap &tmap_in,
    const CUtensorMap &tmap_out,
    const CUtensorMap &tmap_out_t,
    const CUtensorMap &tmap_sc_row,
    const CUtensorMap &tmap_sc_col,
    const CUtensorMap &tmap_sc_row_prepared,
    const CUtensorMap &tmap_sc_col_prepared,
    float *row_sg_ptr,
    float *col_sg_ptr,
    unsigned int *work_counter,
    float *cluster_amax_scratch,
    int64_t M,
    int64_t K,
    bool write_raw_scales,
    bool write_prepared,
    cudaStream_t stream
) {
    using namespace tk_localcta;
    const int blocks_Y = (M + LocalCTAConfig::CHUNK_DIM_Y - 1) / LocalCTAConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + LocalCTAConfig::CHUNK_DIM_X - 1) / LocalCTAConfig::CHUNK_DIM_X;
    const int macro_tiles_Y = (blocks_Y + 1) / 2;
    const int total_macro_tiles = blocks_X * macro_tiles_Y;
    if (total_macro_tiles <= 0) {
        return;
    }
    auto &ci = get_localcta_cached_info();
    int num_clusters = ci.num_sms;
    if (num_clusters > total_macro_tiles) {
        num_clusters = total_macro_tiles;
    }
    if (num_clusters <= 0) {
        num_clusters = 1;
    }

    const int dshmem = shmem_size<RETURN_TRANSPOSE>();
    auto kernel = fused_localcta_quantize_kernel_2cta<RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    LocalCTA2ClusterArgs args {
        .work_counter = work_counter,
        .tiles_X = blocks_X,
        .tiles_Y = blocks_Y,
        .total_macro_tiles = total_macro_tiles
    };

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[0].val.preferredClusterDim.x = 2;
    attrs[0].val.preferredClusterDim.y = 1;
    attrs[0].val.preferredClusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 2;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_clusters * 2, 1, 1);
    config.blockDim = dim3(THREADS, 1, 1);
    config.dynamicSmemBytes = dshmem;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;

    auto err = cudaLaunchKernelEx(
        &config,
        kernel,
        tmap_in, tmap_out, tmap_out_t,
        tmap_sc_row, tmap_sc_col,
        tmap_sc_row_prepared, tmap_sc_col_prepared,
        row_sg_ptr, col_sg_ptr, cluster_amax_scratch,
        M, K, args, write_raw_scales, write_prepared);
    TORCH_CHECK(err == cudaSuccess, "cudaLaunchKernelEx failed for localCTA 2-CTA quant: ",
                cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
allocate_quant_outputs(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4)
                                    : torch::empty({0}, opts_fp4);
    auto col_sc = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8)
                                   : torch::empty({0}, opts_fp8);
    auto row_sg = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg = return_transpose ? torch::empty({K / 128, M / 128}, opts_f32) : row_sg;

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
allocate_quant_outputs_fast(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs(M, K, return_transpose, device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto row_sc_prepared = torch::empty_like(row_sc, opts_fp8);
    auto col_sc_prepared = return_transpose ? torch::empty_like(col_sc, opts_fp8)
                                            : torch::empty({0}, opts_fp8);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           row_sg, col_sg, row_sc_prepared, col_sc_prepared);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
allocate_quant_outputs_prepared(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc_prepared = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4)
                                    : torch::empty({0}, opts_fp4);
    auto col_sc_prepared = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8)
                                            : torch::empty({0}, opts_fp8);
    auto row_sg = torch::empty({M / 128, K / 128}, opts_f32);
    auto col_sg = return_transpose ? torch::empty({K / 128, M / 128}, opts_f32) : row_sg;

    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

void quantize_into_outputs(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    torch::Tensor row_sc_prepared = torch::Tensor(),
    torch::Tensor col_sc_prepared = torch::Tensor(),
    bool use_2cta = false
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    const int blocks_Y = static_cast<int>(M / 128);
    const int blocks_X = static_cast<int>(K / 128);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto *row_sg_ptr = row_sg.data_ptr<float>();
    auto *col_sg_ptr = return_transpose ? col_sg.data_ptr<float>() : row_sg_ptr;
    const bool write_raw_scales = row_sc.defined() && row_sc.numel() > 0;
    const bool write_prepared = row_sc_prepared.defined() && row_sc_prepared.numel() > 0;
    const bool prepared_only = write_prepared && !write_raw_scales;
    const bool use_1cta_tuned = prepared_only && !use_2cta &&
        should_use_localcta1_prepared_auto(M, K, return_transpose);
    TORCH_CHECK(write_raw_scales || write_prepared,
                "quantize_into_outputs requires raw scales, prepared scales, or both");

    torch::Tensor cluster_amax_scratch;
    float *cluster_amax_scratch_ptr = nullptr;
    torch::Tensor work_counter;
    unsigned int *work_counter_ptr = nullptr;
    if (use_2cta && !prepared_only) {
        const int total_macro_tiles = blocks_X * ((blocks_Y + 1) / 2);
        auto &ci = get_localcta_cached_info();
        int num_clusters = ci.num_sms;
        if (num_clusters > total_macro_tiles) {
            num_clusters = total_macro_tiles;
        }
        if (num_clusters <= 0) {
            num_clusters = 1;
        }
        cluster_amax_scratch = torch::zeros(
            {num_clusters * 2},
            torch::dtype(torch::kFloat32).device(input.device()));
        cluster_amax_scratch_ptr = cluster_amax_scratch.data_ptr<float>();
        work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
        work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
    }

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    alignas(64) CUtensorMap tmap_sc_row_prepared{}, tmap_sc_col_prepared{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, tk_localcta::BUFF_DIM_Y,
                  tk_localcta::BUFF_DIM_X, K, 4);
    if (return_transpose) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, tk_localcta::BUFF_DIM_X,
                      tk_localcta::BUFF_DIM_Y, M, 4);
    }

    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    if (write_raw_scales) {
        TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous(),
                    "row_sc must be a contiguous CUDA tensor");
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    }
    if (write_prepared) {
        TORCH_CHECK(row_sc_prepared.is_cuda() && row_sc_prepared.is_contiguous(),
                    "row_sc_prepared must be a contiguous CUDA tensor");
        if (write_raw_scales) {
            TORCH_CHECK(row_sc_prepared.sizes() == row_sc.sizes(),
                        "row_sc_prepared must match row_sc shape");
        }
        create_tma_2d(tmap_sc_row_prepared, row_sc_prepared.data_ptr(),
                      ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    }
    if (return_transpose) {
        const int64_t ntm_c = K / 128;
        const int64_t ntk_c = M / 64;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        if (write_raw_scales) {
            TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous(),
                        "col_sc must be a contiguous CUDA tensor");
            create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        if (write_prepared) {
            TORCH_CHECK(col_sc_prepared.is_cuda() && col_sc_prepared.is_contiguous(),
                        "col_sc_prepared must be a contiguous CUDA tensor");
            if (write_raw_scales) {
                TORCH_CHECK(col_sc_prepared.sizes() == col_sc.sizes(),
                            "col_sc_prepared must match col_sc shape");
            }
            create_tma_2d(tmap_sc_col_prepared, col_sc_prepared.data_ptr(),
                          ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
    }

    if (encode_centric) {
        if (return_transpose) {
            if (use_2cta) {
                if (prepared_only) {
                    launch_localcta_quant_2cta_prepared_tuned_dispatch<true, true>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    launch_localcta_quant_2cta<true, true>(tmap_in, tmap_out, tmap_out_t,
                                                           tmap_sc_row, tmap_sc_col,
                                                           tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                           row_sg_ptr, col_sg_ptr, work_counter_ptr, cluster_amax_scratch_ptr,
                                                           M, K, write_raw_scales, write_prepared, stream);
                }
            } else {
                if (use_1cta_tuned) {
                    launch_localcta_quant_prepared_tuned_dispatch<true, true>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
                    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
                    launch_localcta_quant<true, true>(tmap_in, tmap_out, tmap_out_t,
                                                      tmap_sc_row, tmap_sc_col,
                                                      tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                      row_sg_ptr, col_sg_ptr,
                                                      work_counter_ptr,
                                                      M, K, write_raw_scales, write_prepared, stream);
                }
            }
        } else {
            if (use_2cta) {
                if (prepared_only) {
                    launch_localcta_quant_2cta_prepared_tuned_dispatch<false, true>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    launch_localcta_quant_2cta<false, true>(tmap_in, tmap_out, tmap_out_t,
                                                            tmap_sc_row, tmap_sc_col,
                                                            tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                            row_sg_ptr, col_sg_ptr, work_counter_ptr, cluster_amax_scratch_ptr,
                                                            M, K, write_raw_scales, write_prepared, stream);
                }
            } else {
                if (use_1cta_tuned) {
                    launch_localcta_quant_prepared_tuned_dispatch<false, true>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
                    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
                    launch_localcta_quant<false, true>(tmap_in, tmap_out, tmap_out_t,
                                                       tmap_sc_row, tmap_sc_col,
                                                       tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                       row_sg_ptr, col_sg_ptr,
                                                       work_counter_ptr,
                                                       M, K, write_raw_scales, write_prepared, stream);
                }
            }
        }
    } else {
        if (return_transpose) {
            if (use_2cta) {
                if (prepared_only) {
                    launch_localcta_quant_2cta_prepared_tuned_dispatch<true, false>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    launch_localcta_quant_2cta<true, false>(tmap_in, tmap_out, tmap_out_t,
                                                            tmap_sc_row, tmap_sc_col,
                                                            tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                            row_sg_ptr, col_sg_ptr, work_counter_ptr, cluster_amax_scratch_ptr,
                                                            M, K, write_raw_scales, write_prepared, stream);
                }
            } else {
                if (use_1cta_tuned) {
                    launch_localcta_quant_prepared_tuned_dispatch<true, false>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
                    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
                    launch_localcta_quant<true, false>(tmap_in, tmap_out, tmap_out_t,
                                                       tmap_sc_row, tmap_sc_col,
                                                       tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                       row_sg_ptr, col_sg_ptr,
                                                       work_counter_ptr,
                                                       M, K, write_raw_scales, write_prepared, stream);
                }
            }
        } else {
            if (use_2cta) {
                if (prepared_only) {
                    launch_localcta_quant_2cta_prepared_tuned_dispatch<false, false>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    launch_localcta_quant_2cta<false, false>(tmap_in, tmap_out, tmap_out_t,
                                                             tmap_sc_row, tmap_sc_col,
                                                             tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                             row_sg_ptr, col_sg_ptr, work_counter_ptr, cluster_amax_scratch_ptr,
                                                             M, K, write_raw_scales, write_prepared, stream);
                }
            } else {
                if (use_1cta_tuned) {
                    launch_localcta_quant_prepared_tuned_dispatch<false, false>(
                        tmap_in, tmap_out, tmap_out_t,
                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                        row_sg_ptr, col_sg_ptr, M, K, stream);
                } else {
                    auto work_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
                    auto *work_counter_ptr = reinterpret_cast<unsigned int*>(work_counter.data_ptr<int>());
                    launch_localcta_quant<false, false>(tmap_in, tmap_out, tmap_out_t,
                                                        tmap_sc_row, tmap_sc_col,
                                                        tmap_sc_row_prepared, tmap_sc_col_prepared,
                                                        row_sg_ptr, col_sg_ptr,
                                                        work_counter_ptr,
                                                        M, K, write_raw_scales, write_prepared, stream);
                }
            }
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_quantize_for_gemm failed: ",
                cudaGetErrorString(err));
}

torch::Tensor reconstruct_rowwise_impl(
    torch::Tensor fp4,
    torch::Tensor sc,
    torch::Tensor sg,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(fp4.is_cuda() && sc.is_cuda() && sg.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(fp4.scalar_type() == torch::kFloat4_e2m1fn_x2, "fp4 tensor dtype mismatch");
    TORCH_CHECK(sc.scalar_type() == torch::kFloat8_e4m3fn, "scale tensor dtype mismatch");
    TORCH_CHECK(sg.scalar_type() == torch::kFloat32, "sg tensor dtype mismatch");

    auto out = torch::empty({rows, cols}, torch::dtype(torch::kBFloat16).device(fp4.device()));
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t numel = rows * cols;
    const int threads = 256;
    const int blocks = (int)((numel + threads - 1) / threads);
    tk_localcta_reconstruct::reconstruct_rowwise_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_fp4x2_e2m1*>(fp4.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(sc.data_ptr()),
        sg.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        (int)rows, (int)cols, (int)sg.size(0), (int)sg.size(1));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_localcta_reconstruct failed: ",
                cudaGetErrorString(err));
    return out;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm(torch::Tensor input,
                              bool return_transpose,
                              bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta2_quantize_for_gemm(torch::Tensor input,
                               bool return_transpose,
                               bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg] =
        allocate_quant_outputs(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                          torch::Tensor(), torch::Tensor(), true);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_fast(torch::Tensor input,
                                   bool return_transpose,
                                   bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, row_sc_prepared, col_sc_prepared] =
        allocate_quant_outputs_fast(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           row_sg, col_sg, row_sc_prepared, col_sc_prepared);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta2_quantize_for_gemm_fast(torch::Tensor input,
                                    bool return_transpose,
                                    bool encode_centric) {
    auto [row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, row_sc_prepared, col_sc_prepared] =
        allocate_quant_outputs_fast(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, true);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           row_sg, col_sg, row_sc_prepared, col_sc_prepared);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_prepared(torch::Tensor input,
                                       bool return_transpose,
                                       bool encode_centric) {
    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        allocate_quant_outputs_prepared(input.size(0), input.size(1), return_transpose, input.device());
    const bool use_2cta_prepared = should_use_localcta2_prepared_auto(input.size(0), input.size(1));
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, use_2cta_prepared);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta2_quantize_for_gemm_prepared(torch::Tensor input,
                                        bool return_transpose,
                                        bool encode_centric) {
    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        allocate_quant_outputs_prepared(input.size(0), input.size(1), return_transpose, input.device());
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, true);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    return allocate_quant_outputs(M, K, return_transpose, device);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_fast_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    return allocate_quant_outputs_fast(M, K, return_transpose, device);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta2_quantize_for_gemm_prepared_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    return allocate_quant_outputs_prepared(M, K, return_transpose, device);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_prepared_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    return allocate_quant_outputs_prepared(M, K, return_transpose, device);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_launch(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_fast_launch(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor row_sg,
    torch::Tensor col_sg,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_sc_prepared
) {
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           row_sg, col_sg, row_sc_prepared, col_sc_prepared);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta2_quantize_for_gemm_prepared_launch(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, true);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_quantize_for_gemm_prepared_launch(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    torch::Tensor row_fp4,
    torch::Tensor row_sc_prepared,
    torch::Tensor col_fp4,
    torch::Tensor col_sc_prepared,
    torch::Tensor row_sg,
    torch::Tensor col_sg
) {
    const bool use_2cta_prepared = should_use_localcta2_prepared_auto(input.size(0), input.size(1));
    quantize_into_outputs(input, return_transpose, encode_centric,
                          row_fp4, torch::Tensor(), col_fp4, torch::Tensor(), row_sg, col_sg,
                          row_sc_prepared, col_sc_prepared, use_2cta_prepared);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta_group_quantize_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_parts;
    std::vector<torch::Tensor> row_sc_parts;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t row_offset = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 128 == 0, "split rows must be multiples of 128");
        auto chunk = input.narrow(0, row_offset, rows_i);
        auto [rf, rs, cf, cs, rsg, csg] = tk_localcta_quantize_for_gemm(chunk, true, true);
        row_fp4_parts.push_back(rf);
        row_sc_parts.push_back(rs);
        row_sg_parts.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_list.push_back(cs);
        col_sg_list.push_back(csg);
        row_offset += rows_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_parts, 0);
    auto row_sc_cat = torch::cat(row_sc_parts, 0);
    auto row_sg_cat = torch::cat(row_sg_parts, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_list, col_sc_list, col_sg_cat,
                           row_sg_parts, col_sg_list);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>>
tk_localcta_group_quantize_for_gemm_fast(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_parts;
    std::vector<torch::Tensor> row_sc_parts;
    std::vector<torch::Tensor> row_sc_prepared_parts;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t row_offset = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 128 == 0, "split rows must be multiples of 128");
        auto chunk = input.narrow(0, row_offset, rows_i);
        auto [rf, rs, cf, cs, rsg, csg, rsp, csp] =
            tk_localcta_quantize_for_gemm_fast(chunk, true, true);
        row_fp4_parts.push_back(rf);
        row_sc_parts.push_back(rs);
        row_sc_prepared_parts.push_back(rsp);
        row_sg_parts.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_list.push_back(cs);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        row_offset += rows_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_parts, 0);
    auto row_sc_cat = torch::cat(row_sc_parts, 0);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_parts, 0);
    auto row_sg_cat = torch::cat(row_sg_parts, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_list, col_sc_list, col_sg_cat,
                           row_sg_parts, col_sg_list,
                           row_sc_prepared_cat, col_sc_prepared_list);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta_group_quantize_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_parts;
    std::vector<torch::Tensor> row_sc_prepared_parts;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t row_offset = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 128 == 0, "split rows must be multiples of 128");
        auto chunk = input.narrow(0, row_offset, rows_i);
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta_quantize_for_gemm_prepared(chunk, true, true);
        row_fp4_parts.push_back(rf);
        row_sc_prepared_parts.push_back(rsp);
        row_sg_parts.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        row_offset += rows_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_parts, 0);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_parts, 0);
    auto row_sg_cat = torch::cat(row_sg_parts, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_list, col_sc_prepared_list, col_sg_cat,
                           row_sg_parts, col_sg_list);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta2_group_quantize_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [N_total, K]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_parts;
    std::vector<torch::Tensor> row_sc_prepared_parts;
    std::vector<torch::Tensor> row_sg_parts;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t row_offset = 0;
    for (int64_t rows_i : split_sections) {
        TORCH_CHECK(rows_i % 128 == 0, "split rows must be multiples of 128");
        auto chunk = input.narrow(0, row_offset, rows_i);
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta2_quantize_for_gemm_prepared(chunk, true, true);
        row_fp4_parts.push_back(rf);
        row_sc_prepared_parts.push_back(rsp);
        row_sg_parts.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        row_offset += rows_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_parts, 0);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_parts, 0);
    auto row_sg_cat = torch::cat(row_sg_parts, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_list, col_sc_prepared_list, col_sg_cat,
                           row_sg_parts, col_sg_list);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_list;
    std::vector<torch::Tensor> row_sc_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t col_offset = 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        auto chunk = input.narrow(1, col_offset, cols_i).contiguous();
        auto [rf, rs, cf, cs, rsg, csg] = tk_localcta_quantize_for_gemm(chunk, true, true);
        row_fp4_list.push_back(rf);
        row_sc_list.push_back(rs);
        row_sg_list.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_list.push_back(cs);
        col_sg_list.push_back(csg);
        col_offset += cols_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_list, 1);
    auto row_sc_cat = torch::cat(row_sc_list, 1);
    auto row_sg_cat = torch::cat(row_sg_list, 1);
    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_cat = torch::cat(col_sc_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                           col_fp4_list, col_sc_list, col_sg_list,
                           row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_cat, col_sc_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_for_gemm_fast(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_list;
    std::vector<torch::Tensor> row_sc_list;
    std::vector<torch::Tensor> row_sc_prepared_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t col_offset = 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        auto chunk = input.narrow(1, col_offset, cols_i).contiguous();
        auto [rf, rs, cf, cs, rsg, csg, rsp, csp] =
            tk_localcta_quantize_for_gemm_fast(chunk, true, true);
        row_fp4_list.push_back(rf);
        row_sc_list.push_back(rs);
        row_sc_prepared_list.push_back(rsp);
        row_sg_list.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_list.push_back(cs);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        col_offset += cols_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_list, 1);
    auto row_sc_cat = torch::cat(row_sc_list, 1);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_list, 1);
    auto row_sg_cat = torch::cat(row_sg_list, 1);
    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_cat = torch::cat(col_sc_list, 0);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                           col_fp4_list, col_sc_list, col_sg_list,
                           row_fp4_cat, row_sc_cat, row_sg_cat,
                           col_fp4_cat, col_sc_cat, col_sg_cat,
                           row_sc_prepared_list, col_sc_prepared_list,
                           row_sc_prepared_cat, col_sc_prepared_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_list;
    std::vector<torch::Tensor> row_sc_prepared_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t col_offset = 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        auto chunk = input.narrow(1, col_offset, cols_i).contiguous();
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta_quantize_for_gemm_prepared(chunk, true, true);
        row_fp4_list.push_back(rf);
        row_sc_prepared_list.push_back(rsp);
        row_sg_list.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        col_offset += cols_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_list, 1);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_list, 1);
    auto row_sg_cat = torch::cat(row_sg_list, 1);
    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_prepared_list, row_sg_list,
                           col_fp4_list, col_sc_prepared_list, col_sg_list,
                           row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta2_group_quantize_dim1_for_gemm_prepared(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                "input must be contiguous [M, N_total]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");

    std::vector<torch::Tensor> row_fp4_list;
    std::vector<torch::Tensor> row_sc_prepared_list;
    std::vector<torch::Tensor> row_sg_list;
    std::vector<torch::Tensor> col_fp4_list;
    std::vector<torch::Tensor> col_sc_prepared_list;
    std::vector<torch::Tensor> col_sg_list;

    int64_t col_offset = 0;
    for (int64_t cols_i : col_split_sections) {
        TORCH_CHECK(cols_i % 128 == 0, "column splits must be multiples of 128");
        auto chunk = input.narrow(1, col_offset, cols_i).contiguous();
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta2_quantize_for_gemm_prepared(chunk, true, true);
        row_fp4_list.push_back(rf);
        row_sc_prepared_list.push_back(rsp);
        row_sg_list.push_back(rsg);
        col_fp4_list.push_back(cf);
        col_sc_prepared_list.push_back(csp);
        col_sg_list.push_back(csg);
        col_offset += cols_i;
    }

    auto row_fp4_cat = torch::cat(row_fp4_list, 1);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_list, 1);
    auto row_sg_cat = torch::cat(row_sg_list, 1);
    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_prepared_list, row_sg_list,
                           col_fp4_list, col_sc_prepared_list, col_sg_list,
                           row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta_batched_quantize_for_gemm(
    const std::vector<torch::Tensor> &inputs,
    bool return_transpose,
    bool encode_centric
) {
    std::vector<torch::Tensor> row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs;
    row_fp4s.reserve(inputs.size());
    row_scs.reserve(inputs.size());
    col_fp4s.reserve(inputs.size());
    col_scs.reserve(inputs.size());
    row_sgs.reserve(inputs.size());
    col_sgs.reserve(inputs.size());

    for (const auto &input : inputs) {
        auto [rf, rs, cf, cs, rsg, csg] =
            tk_localcta_quantize_for_gemm(input, return_transpose, encode_centric);
        row_fp4s.push_back(rf);
        row_scs.push_back(rs);
        col_fp4s.push_back(cf);
        col_scs.push_back(cs);
        row_sgs.push_back(rsg);
        col_sgs.push_back(csg);
    }
    return std::make_tuple(row_fp4s, row_scs, col_fp4s, col_scs, row_sgs, col_sgs);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta_batched_quantize_for_gemm_fast(
    const std::vector<torch::Tensor> &inputs,
    bool return_transpose,
    bool encode_centric
) {
    std::vector<torch::Tensor> row_fp4s, row_scs, col_fp4s, col_scs;
    std::vector<torch::Tensor> row_sgs, col_sgs, row_sc_prepareds, col_sc_prepareds;
    row_fp4s.reserve(inputs.size());
    row_scs.reserve(inputs.size());
    col_fp4s.reserve(inputs.size());
    col_scs.reserve(inputs.size());
    row_sgs.reserve(inputs.size());
    col_sgs.reserve(inputs.size());
    row_sc_prepareds.reserve(inputs.size());
    col_sc_prepareds.reserve(inputs.size());

    for (const auto &input : inputs) {
        auto [rf, rs, cf, cs, rsg, csg, rsp, csp] =
            tk_localcta_quantize_for_gemm_fast(input, return_transpose, encode_centric);
        row_fp4s.push_back(rf);
        row_scs.push_back(rs);
        col_fp4s.push_back(cf);
        col_scs.push_back(cs);
        row_sgs.push_back(rsg);
        col_sgs.push_back(csg);
        row_sc_prepareds.push_back(rsp);
        col_sc_prepareds.push_back(csp);
    }
    return std::make_tuple(row_fp4s, row_scs, col_fp4s, col_scs,
                           row_sgs, col_sgs, row_sc_prepareds, col_sc_prepareds);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta_batched_quantize_for_gemm_prepared(
    const std::vector<torch::Tensor> &inputs,
    bool return_transpose,
    bool encode_centric
) {
    std::vector<torch::Tensor> row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs;
    row_fp4s.reserve(inputs.size());
    row_sc_prepareds.reserve(inputs.size());
    col_fp4s.reserve(inputs.size());
    col_sc_prepareds.reserve(inputs.size());
    row_sgs.reserve(inputs.size());
    col_sgs.reserve(inputs.size());

    for (const auto &input : inputs) {
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta_quantize_for_gemm_prepared(input, return_transpose, encode_centric);
        row_fp4s.push_back(rf);
        row_sc_prepareds.push_back(rsp);
        col_fp4s.push_back(cf);
        col_sc_prepareds.push_back(csp);
        row_sgs.push_back(rsg);
        col_sgs.push_back(csg);
    }
    return std::make_tuple(row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta2_batched_quantize_for_gemm_prepared(
    const std::vector<torch::Tensor> &inputs,
    bool return_transpose,
    bool encode_centric
) {
    std::vector<torch::Tensor> row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs;
    row_fp4s.reserve(inputs.size());
    row_sc_prepareds.reserve(inputs.size());
    col_fp4s.reserve(inputs.size());
    col_sc_prepareds.reserve(inputs.size());
    row_sgs.reserve(inputs.size());
    col_sgs.reserve(inputs.size());

    for (const auto &input : inputs) {
        auto [rf, rsp, cf, csp, rsg, csg] =
            tk_localcta2_quantize_for_gemm_prepared(input, return_transpose, encode_centric);
        row_fp4s.push_back(rf);
        row_sc_prepareds.push_back(rsp);
        col_fp4s.push_back(cf);
        col_sc_prepareds.push_back(csp);
        row_sgs.push_back(rsg);
        col_sgs.push_back(csg);
    }
    return std::make_tuple(row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_silu_quantize_split_for_gemm(
    torch::Tensor h1_raw,
    torch::Tensor h3
) {
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(), "h1_raw and h3 must have identical shape");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(h1_raw.device());

    auto out = torch::empty({M, H}, opts_bf16);
    tk_silu_split::launch_forward(
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, H, stream);

    auto [row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg] =
        tk_localcta_quantize_for_gemm_prepared(out, true, true);
    return std::make_tuple(row_fp4, row_sc_prepared, col_fp4, col_sc_prepared, row_sg, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_localcta_silu_deriv_quantize_split_for_gemm(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    TORCH_CHECK(dh.dim() == 2 && dh.is_cuda() && dh.is_contiguous(),
                "dh must be contiguous CUDA [M, H]");
    TORCH_CHECK(h3.dim() == 2 && h3.is_cuda() && h3.is_contiguous(),
                "h3 must be contiguous CUDA [M, H]");
    TORCH_CHECK(h1_raw.dim() == 2 && h1_raw.is_cuda() && h1_raw.is_contiguous(),
                "h1_raw must be contiguous CUDA [M, H]");
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16, "dh must be bf16");
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16, "h3 must be bf16");
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16, "h1_raw must be bf16");
    TORCH_CHECK(dh.sizes() == h3.sizes(), "dh and h3 must have identical shape");
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have identical shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(dh.device());

    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);

    tk_silu_split::launch_backward(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
        M, H, stream);

    auto [row_fp4s, row_sc_prepareds, col_fp4s, col_sc_prepareds, row_sgs, col_sgs] =
        tk_localcta_batched_quantize_for_gemm_prepared({dh1, dh3_out}, true, true);

    return std::make_tuple(
        row_fp4s[0], row_sc_prepareds[0], col_fp4s[0], col_sc_prepareds[0],
        row_sgs[0], col_sgs[0],
        row_fp4s[1], row_sc_prepareds[1], col_fp4s[1], col_sc_prepareds[1],
        row_sgs[1], col_sgs[1]
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
tk_localcta_group_quantize_split_for_gemm_prepared(
    torch::Tensor input0,
    torch::Tensor input1
) {
    for (const auto &input : {input0, input1}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split inputs must be contiguous [N_i, K]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(0) % 128 == 0, "split rows must be multiples of 128");
    }
    TORCH_CHECK(input0.size(1) == input1.size(1),
                "split inputs must have the same K dimension");

    auto [row_fp4_parts, row_sc_prepared_parts, col_fp4_list, col_sc_prepared_list, row_sg_parts, col_sg_list] =
        tk_localcta_batched_quantize_for_gemm_prepared({input0, input1}, true, true);

    auto row_fp4_cat = torch::cat(row_fp4_parts, 0);
    auto row_sc_prepared_cat = torch::cat(row_sc_prepared_parts, 0);
    auto row_sg_cat = torch::cat(row_sg_parts, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 1);

    return std::make_tuple(row_fp4_cat, row_sc_prepared_cat, row_sg_cat,
                           col_fp4_list, col_sc_prepared_list, col_sg_cat,
                           row_sg_parts, col_sg_list);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split3_for_gemm(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2
) {
    for (const auto &input : {input0, input1, input2}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split inputs must be contiguous [M, N_i]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");

    auto [row_fp4_list, row_sc_list, col_fp4_list, col_sc_list, row_sg_list, col_sg_list] =
        tk_localcta_batched_quantize_for_gemm({input0, input1, input2}, true, true);

    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_cat = torch::cat(col_sc_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_list, row_sg_list,
                           col_fp4_list, col_sc_list, col_sg_list,
                           col_fp4_cat, col_sc_cat, col_sg_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_localcta_group_quantize_dim1_split3_for_gemm_prepared(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2
) {
    for (const auto &input : {input0, input1, input2}) {
        TORCH_CHECK(input.dim() == 2 && input.is_cuda() && input.is_contiguous(),
                    "split inputs must be contiguous [M, N_i]");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "split inputs must be bf16");
        TORCH_CHECK(input.size(1) % 128 == 0, "split widths must be multiples of 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");

    auto [row_fp4_list, row_sc_prepared_list, col_fp4_list, col_sc_prepared_list, row_sg_list, col_sg_list] =
        tk_localcta_batched_quantize_for_gemm_prepared({input0, input1, input2}, true, true);

    auto col_fp4_cat = torch::cat(col_fp4_list, 0);
    auto col_sc_prepared_cat = torch::cat(col_sc_prepared_list, 0);
    auto col_sg_cat = torch::cat(col_sg_list, 0);

    return std::make_tuple(row_fp4_list, row_sc_prepared_list, row_sg_list,
                           col_fp4_list, col_sc_prepared_list, col_sg_list,
                           col_fp4_cat, col_sc_prepared_cat, col_sg_cat);
}

torch::Tensor tk_localcta_reconstruct_row(
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor row_sg_chunks
) {
    return reconstruct_rowwise_impl(row_fp4, row_sc, row_sg_chunks,
                                    row_fp4.size(0), row_fp4.size(1) * 2);
}

torch::Tensor tk_localcta_reconstruct_col(
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor col_sg_chunks
) {
    return reconstruct_rowwise_impl(col_fp4, col_sc, col_sg_chunks,
                                    col_fp4.size(0), col_fp4.size(1) * 2);
}

void tk_localcta_set_2cta_prepared_tuning(
    int threads,
    int pipe_depth,
    bool shared_amax
) {
    TORCH_CHECK(threads == 160 || threads == 192 || threads == 256 || threads == 384 || threads == 512,
                "threads must be one of {160, 192, 256, 384, 512}");
    TORCH_CHECK(pipe_depth >= 1 && pipe_depth <= 4,
                "pipe_depth must be in [1, 4]");
    auto &cfg = get_localcta2_prepared_tuning();
    cfg.threads = threads;
    cfg.pipe_depth = pipe_depth;
    cfg.shared_amax = shared_amax;
}

std::tuple<int, int, bool> tk_localcta_get_2cta_prepared_tuning() {
    const auto &cfg = get_localcta2_prepared_tuning();
    return std::make_tuple(cfg.threads, cfg.pipe_depth, cfg.shared_amax);
}

void tk_localcta_set_1cta_prepared_tuning(
    int threads,
    int pipe_depth
) {
    TORCH_CHECK(threads == 160 || threads == 192 || threads == 256,
                "threads must be one of {160, 192, 256}");
    TORCH_CHECK(pipe_depth == 1 || pipe_depth == 2,
                "pipe_depth must be one of {1, 2}");
    auto &cfg = get_localcta1_prepared_tuning();
    cfg.threads = threads;
    cfg.pipe_depth = pipe_depth;
}

std::tuple<int, int> tk_localcta_get_1cta_prepared_tuning() {
    const auto &cfg = get_localcta1_prepared_tuning();
    return std::make_tuple(cfg.threads, cfg.pipe_depth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tk_localcta_quantize_for_gemm", &tk_localcta_quantize_for_gemm,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_fast", &tk_localcta_quantize_for_gemm_fast,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_quantize_for_gemm_prepared", &tk_localcta_quantize_for_gemm_prepared,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta2_quantize_for_gemm", &tk_localcta2_quantize_for_gemm,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta2_quantize_for_gemm_fast", &tk_localcta2_quantize_for_gemm_fast,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta2_quantize_for_gemm_prepared", &tk_localcta2_quantize_for_gemm_prepared,
          py::arg("input"), py::arg("return_transpose"),
          py::arg("encode_centric") = true);
    m.def("tk_localcta_set_2cta_prepared_tuning", &tk_localcta_set_2cta_prepared_tuning,
          py::arg("threads"), py::arg("pipe_depth"), py::arg("shared_amax"));
    m.def("tk_localcta_get_2cta_prepared_tuning", &tk_localcta_get_2cta_prepared_tuning);
    m.def("tk_localcta_set_1cta_prepared_tuning", &tk_localcta_set_1cta_prepared_tuning,
          py::arg("threads"), py::arg("pipe_depth"));
    m.def("tk_localcta_get_1cta_prepared_tuning", &tk_localcta_get_1cta_prepared_tuning);
    m.def("tk_localcta_set_global_scale_num", &tk_localcta_set_global_scale_num,
          py::arg("value"));
    m.def("tk_localcta_get_global_scale_num", &tk_localcta_get_global_scale_num);
    m.def("tk_localcta_reset_global_scale_num", &tk_localcta_reset_global_scale_num);
    m.def("tk_localcta_quantize_for_gemm_alloc", &tk_localcta_quantize_for_gemm_alloc,
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_localcta_quantize_for_gemm_fast_alloc", &tk_localcta_quantize_for_gemm_fast_alloc,
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_localcta2_quantize_for_gemm_prepared_alloc", &tk_localcta2_quantize_for_gemm_prepared_alloc,
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_localcta_quantize_for_gemm_prepared_alloc", &tk_localcta_quantize_for_gemm_prepared_alloc,
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_localcta_quantize_for_gemm_launch", &tk_localcta_quantize_for_gemm_launch,
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("row_sg_chunks"), py::arg("col_sg_chunks"));
    m.def("tk_localcta_quantize_for_gemm_fast_launch", &tk_localcta_quantize_for_gemm_fast_launch,
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("row_sg_chunks"), py::arg("col_sg_chunks"),
          py::arg("row_sc_prepared"), py::arg("col_sc_prepared"));
    m.def("tk_localcta2_quantize_for_gemm_prepared_launch", &tk_localcta2_quantize_for_gemm_prepared_launch,
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric"),
          py::arg("row_fp4"), py::arg("row_sc_prepared"), py::arg("col_fp4"), py::arg("col_sc_prepared"),
          py::arg("row_sg_chunks"), py::arg("col_sg_chunks"));
    m.def("tk_localcta_quantize_for_gemm_prepared_launch", &tk_localcta_quantize_for_gemm_prepared_launch,
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric"),
          py::arg("row_fp4"), py::arg("row_sc_prepared"), py::arg("col_fp4"), py::arg("col_sc_prepared"),
          py::arg("row_sg_chunks"), py::arg("col_sg_chunks"));
    m.def("tk_localcta_batched_quantize_for_gemm", &tk_localcta_batched_quantize_for_gemm,
          py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta_batched_quantize_for_gemm_fast", &tk_localcta_batched_quantize_for_gemm_fast,
          py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta_batched_quantize_for_gemm_prepared", &tk_localcta_batched_quantize_for_gemm_prepared,
          py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta2_batched_quantize_for_gemm_prepared", &tk_localcta2_batched_quantize_for_gemm_prepared,
          py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_localcta_silu_quantize_split_for_gemm", &tk_localcta_silu_quantize_split_for_gemm,
          py::arg("h1_raw"), py::arg("h3"));
    m.def("tk_localcta_silu_deriv_quantize_split_for_gemm", &tk_localcta_silu_deriv_quantize_split_for_gemm,
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_localcta_group_quantize_for_gemm", &tk_localcta_group_quantize_for_gemm,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_for_gemm_fast", &tk_localcta_group_quantize_for_gemm_fast,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_for_gemm_prepared", &tk_localcta_group_quantize_for_gemm_prepared,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_split_for_gemm_prepared", &tk_localcta_group_quantize_split_for_gemm_prepared,
          py::arg("input0"), py::arg("input1"));
    m.def("tk_localcta2_group_quantize_for_gemm_prepared", &tk_localcta2_group_quantize_for_gemm_prepared,
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_localcta_group_quantize_dim1_for_gemm", &tk_localcta_group_quantize_dim1_for_gemm,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_group_quantize_dim1_for_gemm_fast", &tk_localcta_group_quantize_dim1_for_gemm_fast,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_group_quantize_dim1_for_gemm_prepared", &tk_localcta_group_quantize_dim1_for_gemm_prepared,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_group_quantize_dim1_split3_for_gemm", &tk_localcta_group_quantize_dim1_split3_for_gemm,
          py::arg("input0"), py::arg("input1"), py::arg("input2"));
    m.def("tk_localcta_group_quantize_dim1_split3_for_gemm_prepared", &tk_localcta_group_quantize_dim1_split3_for_gemm_prepared,
          py::arg("input0"), py::arg("input1"), py::arg("input2"));
    m.def("tk_localcta2_group_quantize_dim1_for_gemm_prepared", &tk_localcta2_group_quantize_dim1_for_gemm_prepared,
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_localcta_reconstruct_row", &tk_localcta_reconstruct_row,
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("row_sg_chunks"));
    m.def("tk_localcta_reconstruct_col", &tk_localcta_reconstruct_col,
          py::arg("col_fp4"), py::arg("col_sc"), py::arg("col_sg_chunks"));
}

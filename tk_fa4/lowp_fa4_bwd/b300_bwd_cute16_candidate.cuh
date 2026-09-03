#pragma once

#include <cstdio>
#include <cstdlib>

#include "b300_bwd_cute.cuh"
#include "b300_bwd_cute16.cuh"
#include "b300_bwd_cute16_candidate_cute.cuh"
#include "b300_bwd_cute16_kernel_candidate.cuh"
#include "dq_projection_consumer.cuh"
#include "hierarchical_qkv_nvfp4_quantize.cuh"
#include "projection_fp4_epilogue.cuh"
#include "tile_ready_nvfp4_quantize.cuh"
#include "b300_bwd_fa4.cuh"
#include "b300_bwd_fa4_postprocess.cuh"
#include "b300_bwd_fa4_preprocess.cuh"
#include "b300_bwd_hot.cuh"

#ifndef TK_FA4_BWD_MIXED_PREPROCESS_OVERLAP_DQ_ZERO
// The mixed dO/statistics pass and the mandatory FP32 dQ clear touch disjoint
// allocations.  Run the clear on the existing auxiliary stream and join it
// only immediately before the dense launch, hiding its bandwidth/launch cost
// under preprocessing without changing any kernel ownership or arithmetic.
#define TK_FA4_BWD_MIXED_PREPROCESS_OVERLAP_DQ_ZERO 1
#endif

namespace tkfa4::bwd_cute16_candidate {

namespace detail {

__global__ void zero_interleaved_projection_dq_kernel(
    uint4 *output,
    int64_t row_count
) {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    constexpr int kProjectionVectors =
        (kB300QKDim * 2 + kB300VDim) * sizeof(bf16) / sizeof(uint4);
    constexpr int kDqVectors =
        kB300QKDim * sizeof(bf16) / sizeof(uint4);
    static_assert(kProjectionVectors == 64 && kDqVectors == 24);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int64_t row_stride = static_cast<int64_t>(gridDim.x) * kWarps;
    for (
        int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
        row < row_count;
        row += row_stride
    ) {
        if (lane < kDqVectors) {
            output[row * kProjectionVectors + lane] = make_uint4(0, 0, 0, 0);
        }
    }
}

template <typename T>
concept hot_candidate_has_compute_warp_base = requires { T::ComputeWarpBase; };

template <typename T>
concept hot_candidate_has_load_warp_id = requires { T::LoadWarpId; };

struct split_hybrid_resources {
    int device = -1;
    uintptr_t owner_stream = 0;
    int batch = -1;
    int heads = -1;
    int seq_len = -1;
    int dense_split_capacity = 0;
    cudaStream_t dq_stream = nullptr;
    cudaStream_t dq_pack_stream = nullptr;
    cudaStream_t dq_projection_stream = nullptr;
    cudaStream_t dkdv_stream = nullptr;
    cudaEvent_t call_entry_ready = nullptr;
    cudaEvent_t preprocess_done = nullptr;
    cudaEvent_t dq_done = nullptr;
    cudaEvent_t dq_pack_done = nullptr;
    cudaEvent_t dq_projection_done = nullptr;
    cudaEvent_t dkdv_done = nullptr;
    cudaEvent_t dense_main_done = nullptr;
    cudaEvent_t frontier_dv_done = nullptr;
    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_end = nullptr;
    cudaEvent_t preprocess_start = nullptr;
    cudaEvent_t preprocess_end = nullptr;
    cudaEvent_t dkdv_start = nullptr;
    cudaEvent_t dkdv_end = nullptr;
    cudaEvent_t dq_start = nullptr;
    cudaEvent_t dq_zero_end = nullptr;
    cudaEvent_t dq_ready = nullptr;
    cudaEvent_t dq_end = nullptr;
    at::Tensor dpsum;
    at::Tensor lse_log2;
    at::Tensor dout_fp8;
    at::Tensor dout_dp_fp8;
    at::Tensor v_fp8;
    at::Tensor dout_dp_mxfp4;
    at::Tensor v_dp_mxfp4;
    at::Tensor dout_dp_mxfp4_scale;
    at::Tensor v_dp_mxfp4_scale;
    at::Tensor dout_mxfp4;
    at::Tensor dout_mxfp4_scale;
    at::Tensor dqacc;
    at::Tensor ds_scratch;
    at::Tensor frontier_dk;
    at::Tensor frontier_dv;
    at::Tensor dense_split_dk;
    at::Tensor dense_split_dv;
    at::Tensor cached_base_dk;
    at::Tensor cached_base_dv;
};

inline bool matches_fp32_base(
    const at::Tensor &tensor,
    const at::Tensor &q,
    int feature_dim
) {
    return tensor.defined() &&
        tensor.device() == q.device() &&
        tensor.scalar_type() == at::kFloat &&
        tensor.is_contiguous() &&
        tensor.dim() == 4 &&
        tensor.size(0) == q.size(0) &&
        tensor.size(1) == q.size(1) &&
        tensor.size(2) == q.size(2) &&
        tensor.size(3) == feature_dim;
}

inline bool split_timing_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("TK_FA4_SPLIT_TIMING");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

inline bool hot_clustered_dq_probe_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("TK_FA4_USE_HOT_CLUSTERED_DQ");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

inline split_hybrid_resources &get_split_hybrid_resources(
    int device,
    cudaStream_t current_stream,
    const at::Tensor &q,
    const at::Tensor &lse,
    int scratch_rows,
    int dqk_dim,
    bool need_ds_scratch = false,
    int frontier_dvo_dim = 0,
    int dense_split_count = 1,
    bool cache_dkdv_fp32_base = false,
    bool need_dout_fp8 = false,
    bool need_fp8_dp = false,
    bool need_mxfp4_dv = false,
    bool need_mxfp4_dp = false
) {
    TORCH_CHECK(
        dense_split_count == 1 || dense_split_count == 2 || dense_split_count == 3 ||
            dense_split_count == 4 || dense_split_count == 8,
        "dense split count must be 1, 2, 3, 4, or 8"
    );
    TORCH_CHECK(
        dense_split_count == 1 || frontier_dvo_dim > 0,
        "dense split scratch requires a positive dV dimension"
    );
    TORCH_CHECK(
        !cache_dkdv_fp32_base || frontier_dvo_dim > 0,
        "cached FP32 dK/dV bases require a positive dV dimension"
    );
    static thread_local split_hybrid_resources cache;
    const bool device_changed = cache.device != device;
    if (device_changed) {
        if (cache.device >= 0 && cache.preprocess_done != nullptr) {
            const c10::cuda::CUDAGuard old_device_guard(
                static_cast<c10::DeviceIndex>(cache.device)
            );
            cudaEventDestroy(cache.call_entry_ready);
            cudaEventDestroy(cache.preprocess_done);
            cudaEventDestroy(cache.dq_done);
            cudaEventDestroy(cache.dq_pack_done);
            cudaEventDestroy(cache.dq_projection_done);
            cudaEventDestroy(cache.dkdv_done);
            cudaEventDestroy(cache.dense_main_done);
            cudaEventDestroy(cache.frontier_dv_done);
            cudaEventDestroy(cache.total_start);
            cudaEventDestroy(cache.total_end);
            cudaEventDestroy(cache.preprocess_start);
            cudaEventDestroy(cache.preprocess_end);
            cudaEventDestroy(cache.dkdv_start);
            cudaEventDestroy(cache.dkdv_end);
            cudaEventDestroy(cache.dq_start);
            cudaEventDestroy(cache.dq_zero_end);
            cudaEventDestroy(cache.dq_ready);
            cudaEventDestroy(cache.dq_end);
        }
        cache.device = device;
        cache.dq_stream = at::cuda::getStreamFromPool(false, device).stream();
        cache.dq_pack_stream =
            at::cuda::getStreamFromPool(true, device).stream();
        cache.dq_projection_stream =
            at::cuda::getStreamFromPool(true, device).stream();
        cache.dkdv_stream = at::cuda::getStreamFromPool(false, device).stream();
        CUDACHECK(cudaEventCreateWithFlags(&cache.call_entry_ready, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.preprocess_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.dq_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(
            &cache.dq_pack_done,
            cudaEventDisableTiming
        ));
        CUDACHECK(cudaEventCreateWithFlags(
            &cache.dq_projection_done,
            cudaEventDisableTiming
        ));
        CUDACHECK(cudaEventCreateWithFlags(&cache.dkdv_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.dense_main_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&cache.frontier_dv_done, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreate(&cache.total_start));
        CUDACHECK(cudaEventCreate(&cache.total_end));
        CUDACHECK(cudaEventCreate(&cache.preprocess_start));
        CUDACHECK(cudaEventCreate(&cache.preprocess_end));
        CUDACHECK(cudaEventCreate(&cache.dkdv_start));
        CUDACHECK(cudaEventCreate(&cache.dkdv_end));
        CUDACHECK(cudaEventCreate(&cache.dq_start));
        CUDACHECK(cudaEventCreate(&cache.dq_zero_end));
        CUDACHECK(cudaEventCreate(&cache.dq_ready));
        CUDACHECK(cudaEventCreate(&cache.dq_end));
    }
    const uintptr_t stream_key = reinterpret_cast<uintptr_t>(current_stream);
    const bool needs_dqacc = scratch_rows > 0;
    const int dense_split_partials = dense_split_count - 1;
    const int fp8_dp_workspace_dim =
        need_fp8_dp && need_mxfp4_dp &&
                TK_FA4_BWD_MIXED_DP_COMPACT_96B_OPERANDS
            ? 96
            : frontier_dvo_dim;
    const bool needs_scratch_refresh =
        device_changed ||
        cache.owner_stream != stream_key ||
        cache.batch != q.size(0) ||
        cache.heads != q.size(2) ||
        cache.seq_len != q.size(1) ||
        !cache.dpsum.defined() ||
        !cache.lse_log2.defined() ||
        (need_dout_fp8 && !cache.dout_fp8.defined()) ||
        (need_fp8_dp &&
         (!cache.dout_dp_fp8.defined() || !cache.v_fp8.defined() ||
          cache.dout_dp_fp8.size(3) != fp8_dp_workspace_dim ||
          cache.v_fp8.size(3) != fp8_dp_workspace_dim)) ||
        (need_mxfp4_dp &&
         (!cache.dout_dp_mxfp4.defined() ||
          !cache.v_dp_mxfp4.defined() ||
          !cache.dout_dp_mxfp4_scale.defined() ||
          !cache.v_dp_mxfp4_scale.defined())) ||
        (need_mxfp4_dv &&
         (!cache.dout_mxfp4.defined() ||
          !cache.dout_mxfp4_scale.defined())) ||
        (need_ds_scratch && (
            !cache.ds_scratch.defined() ||
            cache.ds_scratch.size(1) != q.size(1) ||
            cache.ds_scratch.size(3) != q.size(1)
        )) ||
        (frontier_dvo_dim > 0 && (
            !cache.frontier_dk.defined() ||
            !cache.frontier_dv.defined() ||
            cache.frontier_dk.size(3) != dqk_dim ||
            cache.frontier_dv.size(3) != frontier_dvo_dim
        )) ||
        (dense_split_count > 1 && (
            !cache.dense_split_dk.defined() ||
            !cache.dense_split_dv.defined() ||
            cache.dense_split_capacity < dense_split_partials ||
            cache.dense_split_dk.size(3) != dqk_dim ||
            cache.dense_split_dv.size(3) != frontier_dvo_dim
        )) ||
        (needs_dqacc && (
            !cache.dqacc.defined() ||
            cache.dqacc.size(2) != scratch_rows ||
            cache.dqacc.size(3) != dqk_dim
        )) ||
        (cache_dkdv_fp32_base && (
            !matches_fp32_base(cache.cached_base_dk, q, dqk_dim) ||
            !matches_fp32_base(cache.cached_base_dv, q, frontier_dvo_dim)
        ));
    if (needs_scratch_refresh) {
        cache.owner_stream = stream_key;
        cache.batch = static_cast<int>(q.size(0));
        cache.heads = static_cast<int>(q.size(2));
        cache.seq_len = static_cast<int>(q.size(1));
        cache.dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
        cache.lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
        if (need_dout_fp8) {
            cache.dout_fp8 = at::empty(
                {
                    q.size(0),
                    q.size(1),
                    q.size(2),
                    frontier_dvo_dim
                },
                q.options().dtype(at::ScalarType::Float8_e4m3fn)
            );
        } else {
            cache.dout_fp8 = at::Tensor();
        }
        if (need_fp8_dp) {
            cache.dout_dp_fp8 = at::empty(
                {
                    q.size(0),
                    q.size(1),
                    q.size(2),
                    fp8_dp_workspace_dim
                },
                q.options().dtype(at::ScalarType::Float8_e4m3fn)
            );
            cache.v_fp8 = at::empty(
                {
                    q.size(0),
                    q.size(1),
                    q.size(2),
                    fp8_dp_workspace_dim
                },
                q.options().dtype(at::ScalarType::Float8_e4m3fn)
            );
        } else {
            cache.dout_dp_fp8 = at::Tensor();
            cache.v_fp8 = at::Tensor();
        }
        if (need_mxfp4_dp) {
            cache.dout_dp_mxfp4 = at::empty(
                {
                    q.size(0),
                    q.size(1),
                    q.size(2),
                    frontier_dvo_dim / 2
                },
                q.options().dtype(at::ScalarType::Byte)
            );
            cache.v_dp_mxfp4 = at::empty_like(cache.dout_dp_mxfp4);
            cache.dout_dp_mxfp4_scale = at::empty(
                {
                    q.size(0),
                    q.size(1) / 128,
                    q.size(2),
                    512
                },
                q.options().dtype(at::ScalarType::Byte)
            );
            cache.v_dp_mxfp4_scale =
                at::empty_like(cache.dout_dp_mxfp4_scale);
        } else {
            cache.dout_dp_mxfp4 = at::Tensor();
            cache.v_dp_mxfp4 = at::Tensor();
            cache.dout_dp_mxfp4_scale = at::Tensor();
            cache.v_dp_mxfp4_scale = at::Tensor();
        }
        if (need_mxfp4_dv) {
            cache.dout_mxfp4 = at::empty(
                {
                    q.size(0),
                    q.size(2),
                    frontier_dvo_dim,
                    q.size(1) / 2
                },
                q.options().dtype(at::ScalarType::Byte)
            );
            cache.dout_mxfp4_scale = at::empty(
                {
                    q.size(0),
                    q.size(1) / 128,
                    q.size(2),
                    512
                },
                q.options().dtype(at::ScalarType::Byte)
            );
        } else {
            cache.dout_mxfp4 = at::Tensor();
            cache.dout_mxfp4_scale = at::Tensor();
        }
        if (need_ds_scratch) {
            cache.ds_scratch = at::empty(
                {q.size(0), q.size(1), q.size(2), q.size(1)},
                q.options()
            );
        } else {
            cache.ds_scratch = at::Tensor();
        }
        if (needs_dqacc) {
            cache.dqacc = at::empty({q.size(0), q.size(2), scratch_rows, dqk_dim}, lse.options());
        } else {
            cache.dqacc = at::Tensor();
        }
        if (frontier_dvo_dim > 0) {
            cache.frontier_dk = at::empty(
                {q.size(0), q.size(1), q.size(2), dqk_dim},
                lse.options()
            );
            cache.frontier_dv = at::empty(
                {q.size(0), q.size(1), q.size(2), frontier_dvo_dim},
                lse.options()
            );
        } else {
            cache.frontier_dk = at::Tensor();
            cache.frontier_dv = at::Tensor();
        }
        if (dense_split_count > 1) {
            cache.dense_split_capacity = dense_split_partials;
            cache.dense_split_dk = at::empty(
                {q.size(0) * dense_split_partials, q.size(1), q.size(2), dqk_dim},
                lse.options()
            );
            cache.dense_split_dv = at::empty(
                {q.size(0) * dense_split_partials, q.size(1), q.size(2), frontier_dvo_dim},
                lse.options()
            );
        } else {
            cache.dense_split_capacity = 0;
            cache.dense_split_dk = at::Tensor();
            cache.dense_split_dv = at::Tensor();
        }
        if (cache_dkdv_fp32_base) {
            cache.cached_base_dk = at::empty(
                {q.size(0), q.size(1), q.size(2), dqk_dim},
                lse.options().dtype(at::kFloat)
            );
            cache.cached_base_dv = at::empty(
                {q.size(0), q.size(1), q.size(2), frontier_dvo_dim},
                lse.options().dtype(at::kFloat)
            );
        } else {
            cache.cached_base_dk = at::Tensor();
            cache.cached_base_dv = at::Tensor();
        }
    }
    return cache;
}

}  // namespace detail

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
using config = bwd_cute16_kernel_candidate::config<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;

template <int Dvo>
struct preprocess_config {
    static_assert(Dvo == kB300VDim, "Exact B300 candidate preprocess only supports Dvo=128");
    static constexpr int DvoDim = Dvo;
};

// Operand contract for a producer (normally the QKV projection and the
// output-gradient projection) that publishes the exact MXFP4 layouts consumed
// by the dense backward kernel.  Keeping this as an optional host-side bundle
// lets the retained entry points continue to build the operands locally while
// a fused producer can bypass both O(S) quantization kernels entirely.
struct producer_native_mxfp4_operands {
    at::Tensor *dout_dp;
    at::Tensor *v_dp;
    at::Tensor *dout_dp_scale;
    at::Tensor *v_dp_scale;
    at::Tensor *dout_dv;
    at::Tensor *dout_dv_scale;
    at::Tensor *dpsum;
    at::Tensor *lse_log2;
};

// Retained FP4+FP8 backward producer contract.  QKV projection publishes V
// and the output-gradient projection publishes dO, all in the exact
// layouts/scales consumed by the hot kernel.  Statistics may be supplied by
// that producer as well, or left null so backward computes only O*dO and log2
// LSE without repeating the dO conversion.
struct producer_native_fp8_operands {
    at::Tensor *dout_dp;
    at::Tensor *v_dp;
    at::Tensor *dpsum;
    at::Tensor *lse_log2;
    bool stats_from_packed_dout;
};

// Storage contract for the tile-ready NVFP4 dQ projection.  The BF16 dQ
// reduction remains private to the attention launch; completed tiles are
// packed into input_fp4/input_scales and consumed by the persistent
// projection as soon as operand_ready is released.
struct tile_ready_nvfp4_projection_operands {
    at::Tensor *input_fp4;
    at::Tensor *input_scales;
    at::Tensor *input_global_scale;
    at::Tensor *weight_fp4;
    at::Tensor *weight_scales;
    at::Tensor *weight_global_scale;
    at::Tensor *output;
    at::Tensor *operand_ready;
    at::Tensor *rope_cos = nullptr;
    at::Tensor *rope_sin = nullptr;
    bool hierarchical_qkv = false;
};

template <typename C>
struct preprocess_globals {
    using stats_tile = col_vec<st_fl<kRefTileM, C::DvoDim>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    const bf16 *o_ptr;
    const bf16 *dout_ptr;
    const float *lse_ptr;
    stats_gl dpsum;
    stats_gl lse_log2;
    int seq_len;
    int heads;
};

template <typename C>
__global__ __launch_bounds__(4 * kWarpThreads, 8)
void preprocess_kernel(const __grid_constant__ preprocess_globals<C> g) {
    constexpr int RowThreads = 8;
    constexpr int RowsPerWarp = kWarpThreads / RowThreads;
    const int q_tile_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = laneid();
    const int row_in_warp = lane / RowThreads;
    const int lane_in_row = lane % RowThreads;
    const int row = warp * RowsPerWarp + row_in_warp;
    if (row < kRefTileM) {
        const int seq_idx = q_tile_idx * kRefTileM + row;
        float dpsum = 0.0f;
        const size_t base = (((size_t)batch_idx * g.seq_len + seq_idx) * g.heads + head_idx) * C::DvoDim;
        #pragma unroll
        for (int d = lane_in_row; d < C::DvoDim; d += RowThreads) {
            dpsum += __bfloat162float(g.o_ptr[base + d]) * __bfloat162float(g.dout_ptr[base + d]);
        }
        #pragma unroll
        for (int offset = RowThreads / 2; offset > 0; offset >>= 1) {
            dpsum += __shfl_down_sync(0xffffffff, dpsum, offset, RowThreads);
        }
        if (lane_in_row == 0) {
            const size_t lse_offset = ((size_t)batch_idx * g.seq_len + seq_idx) * g.heads + head_idx;
            g.dpsum[{batch_idx, head_idx, 0, seq_idx}] = dpsum;
            g.lse_log2[{batch_idx, head_idx, 0, seq_idx}] = g.lse_ptr[lse_offset] * kLog2E;
        }
    }
}

template <typename C>
inline void launch_preprocess(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &lse,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    using G = preprocess_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, out.size(0), out.size(2), 1, out.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, out.size(0), out.size(2), 1, out.size(1)),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / kRefTileM, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_kernel<C><<<grid, 4 * kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// FP8-PV reuses the mandatory dPsum pass to quantize dO once.  The fixed
// 2^8 scale is folded into the BF16 exponent before the E4M3 conversion, so
// the preprocessing work replaces the repeated in-kernel conversion without
// adding a separate quantization launch or floating-point multiply.
template <typename C>
struct preprocess_fp8_dout_globals {
    using stats_tile = col_vec<st_fl<kRefTileM, C::DvoDim>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    const bf16 *o_ptr;
    const bf16 *dout_ptr;
    const bf16 *v_ptr;
    const float *lse_ptr;
    fp8e4m3 *dout_fp8_ptr;
    fp8e4m3 *dout_dp_fp8_ptr;
    fp8e4m3 *v_fp8_ptr;
    stats_gl dpsum;
    stats_gl lse_log2;
    int seq_len;
    int heads;
};

template <
    typename C,
    bool QuantizeDpOperands = false,
    bool QuantizeV = true,
    bool QuantizeDout = true,
    bool ScaleDpsumX16 = QuantizeDpOperands,
    bool StatsFromFp8 = false
>
__global__ __launch_bounds__(4 * kWarpThreads, 8)
void preprocess_fp8_dout_kernel(
    const __grid_constant__ preprocess_fp8_dout_globals<C> g
) {
    static_assert(
        !StatsFromFp8 || (!QuantizeDpOperands && !QuantizeDout),
        "packed-dO statistics must not requantize their FP8 source"
    );
    constexpr int RowThreads = 8;
    constexpr int RowsPerWarp = kWarpThreads / RowThreads;
    const int q_tile_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = laneid();
    const int row_in_warp = lane / RowThreads;
    const int lane_in_row = lane % RowThreads;
    const int row = warp * RowsPerWarp + row_in_warp;
    if (row < kRefTileM) {
        const int seq_idx = q_tile_idx * kRefTileM + row;
        float dpsum = 0.0f;
        const size_t base =
            (((size_t)batch_idx * g.seq_len + seq_idx) * g.heads +
             head_idx) * C::DvoDim;
        constexpr int ElementsPerLane = StatsFromFp8 ? 8 : 2;
        #pragma unroll
        for (int d = ElementsPerLane * lane_in_row; d < C::DvoDim;
            d += ElementsPerLane * RowThreads) {
            if constexpr (StatsFromFp8) {
                const uint4 out_packed =
                    *reinterpret_cast<const uint4 *>(g.o_ptr + base + d);
                const bf16_2 out_pair0 =
                    *reinterpret_cast<const bf16_2 *>(&out_packed.x);
                const bf16_2 out_pair1 =
                    *reinterpret_cast<const bf16_2 *>(&out_packed.y);
                const bf16_2 out_pair2 =
                    *reinterpret_cast<const bf16_2 *>(&out_packed.z);
                const bf16_2 out_pair3 =
                    *reinterpret_cast<const bf16_2 *>(&out_packed.w);
                const float2 out_values0 =
                    __bfloat1622float2(out_pair0);
                const float2 out_values1 =
                    __bfloat1622float2(out_pair1);
                const float2 out_values2 =
                    __bfloat1622float2(out_pair2);
                const float2 out_values3 =
                    __bfloat1622float2(out_pair3);
                const uint2 encoded_bits =
                    *reinterpret_cast<const uint2 *>(
                        g.dout_dp_fp8_ptr + base + d
                    );
                const float4 dout_values0 =
                    base_types::convertor<float4, fp8e4m3_4>::convert(
                        std::bit_cast<fp8e4m3_4>(encoded_bits.x)
                    );
                const float4 dout_values1 =
                    base_types::convertor<float4, fp8e4m3_4>::convert(
                        std::bit_cast<fp8e4m3_4>(encoded_bits.y)
                    );
                dpsum +=
                    out_values0.x * dout_values0.x +
                    out_values0.y * dout_values0.y +
                    out_values1.x * dout_values0.z +
                    out_values1.y * dout_values0.w +
                    out_values2.x * dout_values1.x +
                    out_values2.y * dout_values1.y +
                    out_values3.x * dout_values1.z +
                    out_values3.y * dout_values1.w;
            } else {
                const bf16_2 out_pair =
                    *reinterpret_cast<const bf16_2 *>(g.o_ptr + base + d);
                const auto *out_values =
                    reinterpret_cast<const bf16 *>(&out_pair);
                const bf16_2 dout_pair =
                    *reinterpret_cast<const bf16_2 *>(
                        g.dout_ptr + base + d
                    );
                const auto *dout_values =
                    reinterpret_cast<const bf16 *>(&dout_pair);
                dpsum +=
                    __bfloat162float(out_values[0]) *
                        __bfloat162float(dout_values[0]) +
                    __bfloat162float(out_values[1]) *
                        __bfloat162float(dout_values[1]);
                if constexpr (QuantizeDpOperands) {
                    // One 2^2-scaled dO representation serves both dP and dV.
                    // This mirrors the forward pipeline's representation
                    // reuse: quantize once in the mandatory statistics pass,
                    // then fold the consumer-specific power of two into the
                    // output scale.
                    const uint16_t dout_dp_packed =
                        bwd_cute16_kernel_candidate::detail::
                            cta2_role_split_convert_scaled_bf16_pair_to_fp8<
                                bwd_cute16_kernel_candidate::detail::
                                    kCta2DenseFp8DpOperandScaleBf16PairDelta
                            >(dout_pair);
                    *reinterpret_cast<uint16_t *>(
                        g.dout_dp_fp8_ptr + base + d
                    ) = dout_dp_packed;
                    if constexpr (QuantizeV) {
                        const bf16_2 v_pair =
                            *reinterpret_cast<const bf16_2 *>(
                                g.v_ptr + base + d
                            );
                        const uint16_t v_packed =
                            bwd_cute16_kernel_candidate::detail::
                                cta2_role_split_convert_scaled_bf16_pair_to_fp8<
                                    bwd_cute16_kernel_candidate::detail::
                                        kCta2DenseFp8DpOperandScaleBf16PairDelta
                                >(v_pair);
                        *reinterpret_cast<uint16_t *>(
                            g.v_fp8_ptr + base + d
                        ) = v_packed;
                    }
                } else if constexpr (QuantizeDout) {
                    const uint16_t packed =
                        bwd_cute16_kernel_candidate::detail::
                            cta2_role_split_convert_scaled_bf16_pair_to_fp8<
                                bwd_cute16_kernel_candidate::detail::
                                    kCta2DenseFp8PvDoScaleBf16PairDelta
                            >(dout_pair);
                    *reinterpret_cast<uint16_t *>(
                        g.dout_fp8_ptr + base + d
                    ) = packed;
                }
            }
        }
        #pragma unroll
        for (int offset = RowThreads / 2; offset > 0; offset >>= 1) {
            dpsum += __shfl_down_sync(0xffffffff, dpsum, offset, RowThreads);
        }
        if (lane_in_row == 0) {
            const size_t lse_offset =
                ((size_t)batch_idx * g.seq_len + seq_idx) * g.heads +
                head_idx;
            g.dpsum[{batch_idx, head_idx, 0, seq_idx}] =
                StatsFromFp8
                    ? dpsum * 4.0f
                    : (ScaleDpsumX16 ? dpsum * 16.0f : dpsum);
            g.lse_log2[{batch_idx, head_idx, 0, seq_idx}] =
                g.lse_ptr[lse_offset] * kLog2E;
        }
    }
}

// The output-projection epilogue can publish its register-resident dO in the
// exact fixed-scale E4M3 representation without also extending the projection
// critical path with the O*dO dot product.  In that producer mode this small
// pass computes only the mandatory statistics: it deliberately performs no
// duplicate dO conversion or V publication.
template <typename C>
inline void launch_preprocess_fp8_stats_only(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &lse,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    using G = preprocess_fp8_dout_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        nullptr,
        reinterpret_cast<const float *>(lse.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / kRefTileM, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_fp8_dout_kernel<C, false, false, false, true><<<
        grid,
        4 * kWarpThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_preprocess_fp8_stats_from_packed_dout(
    at::Tensor &out,
    at::Tensor &dout_fp8,
    at::Tensor &lse,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    using G = preprocess_fp8_dout_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        nullptr,
        nullptr,
        reinterpret_cast<const float *>(lse.data_ptr()),
        nullptr,
        reinterpret_cast<fp8e4m3 *>(dout_fp8.data_ptr()),
        nullptr,
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / kRefTileM, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_fp8_dout_kernel<C, false, false, false, true, true><<<
        grid,
        4 * kWarpThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_preprocess_fp8_dout(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &lse,
    at::Tensor &dout_fp8,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    TORCH_CHECK(
        dout_fp8.sizes() == dout.sizes() &&
            dout_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "FP8-PV preprocess requires a shape-matched E4M3 dO workspace"
    );
    using G = preprocess_fp8_dout_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        nullptr,
        reinterpret_cast<const float *>(lse.data_ptr()),
        reinterpret_cast<fp8e4m3 *>(dout_fp8.data_ptr()),
        nullptr,
        nullptr,
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / kRefTileM, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_fp8_dout_kernel<C, false><<<
        grid,
        4 * kWarpThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_preprocess_fp8_dout_v(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &v,
    at::Tensor &lse,
    at::Tensor &dout_dp_fp8,
    at::Tensor &v_fp8,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    TORCH_CHECK(
        dout_dp_fp8.sizes() == dout.sizes() &&
            v_fp8.sizes() == v.sizes() &&
            dout_dp_fp8.scalar_type() ==
                at::ScalarType::Float8_e4m3fn &&
            v_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "FP8-dP preprocess requires shape-matched E4M3 workspaces"
    );
    using G = preprocess_fp8_dout_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        nullptr,
        reinterpret_cast<fp8e4m3 *>(dout_dp_fp8.data_ptr()),
        reinterpret_cast<fp8e4m3 *>(v_fp8.data_ptr()),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / kRefTileM, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_fp8_dout_kernel<C, true><<<
        grid,
        4 * kWarpThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_preprocess_fp8_dout_prepacked_v(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &lse,
    at::Tensor &dout_dp_fp8,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    TORCH_CHECK(
        dout_dp_fp8.sizes() == dout.sizes() &&
            dout_dp_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "FP8-dP preprocess requires a shape-matched E4M3 dO workspace"
    );
    using G = preprocess_fp8_dout_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        nullptr,
        reinterpret_cast<const float *>(lse.data_ptr()),
        nullptr,
        reinterpret_cast<fp8e4m3 *>(dout_dp_fp8.data_ptr()),
        nullptr,
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / kRefTileM, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_fp8_dout_kernel<C, true, false><<<
        grid,
        4 * kWarpThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
struct preprocess_mixed_dp_half_globals {
    using stats_tile = col_vec<st_fl<kRefTileM, C::DvoDim>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    const bf16 *o_ptr;
    const bf16 *dout_ptr;
    const bf16 *v_ptr;
    const float *lse_ptr;
    fp8e4m3 *dout_dv_fp8_ptr;
    uint8_t *dout_dp_mixed_ptr;
    uint8_t *v_dp_mixed_ptr;
    uint8_t *dout_dp_scale_ptr;
    uint8_t *v_dp_scale_ptr;
    stats_gl dpsum;
    stats_gl lse_log2;
    int seq_len;
    int heads;
};

template <int FixedScaleE8M0>
__device__ __forceinline__ uint32_t
preprocess_pack_fixed_bf16x8_e2m1(
    uint32_t b01,
    uint32_t b23,
    uint32_t b45,
    uint32_t b67
) {
    static_assert(FixedScaleE8M0 >= 117 && FixedScaleE8M0 <= 129);
    // Scaling to the E2M1 endpoint is exactly 2^(129-scale).  Increment both
    // BF16 exponents in one packed integer operation, then convert directly
    // to four E2M1 pairs.  This is the preprocessing analogue of the forward
    // path's exponent-delta pack and removes eight FP32 multiplies.
    constexpr uint32_t kBf16ExponentDelta =
        static_cast<uint32_t>(129 - FixedScaleE8M0) << 7;
    constexpr uint32_t kPackedExponentDelta =
        kBf16ExponentDelta * 0x00010001u;
    const uint32_t scaled01 = __vadd2(b01, kPackedExponentDelta);
    const uint32_t scaled23 = __vadd2(b23, kPackedExponentDelta);
    const uint32_t scaled45 = __vadd2(b45, kPackedExponentDelta);
    const uint32_t scaled67 = __vadd2(b67, kPackedExponentDelta);
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;\n"
        "mov.b32 {b0, b1}, %1;\n"
        "mov.b32 {b2, b3}, %2;\n"
        "mov.b32 {b4, b5}, %3;\n"
        "mov.b32 {b6, b7}, %4;\n"
        ".reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;\n"
        "cvt.f32.bf16 f0, b0;\n"
        "cvt.f32.bf16 f1, b1;\n"
        "cvt.f32.bf16 f2, b2;\n"
        "cvt.f32.bf16 f3, b3;\n"
        "cvt.f32.bf16 f4, b4;\n"
        "cvt.f32.bf16 f5, b5;\n"
        "cvt.f32.bf16 f6, b6;\n"
        "cvt.f32.bf16 f7, b7;\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, f1, f0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, f3, f2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, f5, f4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, f7, f6;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(packed)
        : "r"(scaled01), "r"(scaled23), "r"(scaled45), "r"(scaled67)
    );
    return packed;
}

#ifndef TK_FA4_BWD_MIXED_DP_PREPROCESS_ROW_THREADS
#define TK_FA4_BWD_MIXED_DP_PREPROCESS_ROW_THREADS 16
#endif

template <typename C, bool QuantizeV = true>
__global__ __launch_bounds__(4 * kWarpThreads, 16)
void preprocess_mixed_dp_half_kernel(
    const __grid_constant__ preprocess_mixed_dp_half_globals<C> g
) {
    static_assert(C::DvoDim == 128);
    constexpr int kMixedOperandBytes =
        TK_FA4_BWD_MIXED_DP_COMPACT_96B_OPERANDS ? 96 : C::DvoDim;
    constexpr int kFixedScaleE8M0 =
        TK_FA4_BWD_MIXED_DP_FIXED_SCALE_E8M0;
    constexpr int kTailPackScaleE8M0 =
        TK_FA4_BWD_MIXED_DP_STATIC_FP4_TAIL
            ? 127
            : kFixedScaleE8M0;
    constexpr int kRowThreads =
        TK_FA4_BWD_MIXED_DP_PREPROCESS_ROW_THREADS;
    static_assert(kRowThreads == 8 || kRowThreads == 16);
    static_assert(
        !TK_FA4_BWD_MIXED_DV_PADDED_DO_TAIL ||
            (kRowThreads == 16 && kFixedScaleE8M0 != 0 &&
             kMixedOperandBytes == 128),
        "padded mixed-dV packing requires fixed-scale 128-byte rows"
    );
    constexpr int kValuesPerLane = C::DvoDim / kRowThreads;
    constexpr int kPairsPerLane = kValuesPerLane / 2;
    constexpr int kLanesPerKBlock = 32 / kValuesPerLane;
    constexpr int kRowsPerWarp = kWarpThreads / kRowThreads;
    constexpr int kRowsPerBlock = 4 * kRowsPerWarp;
    constexpr int kMxScalePageRows = 128;
    const int row_tile_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int warp = static_cast<int>(threadIdx.x) / kWarpThreads;
    const int lane = laneid();
    const int row_in_warp = lane / kRowThreads;
    const int lane_in_row = lane % kRowThreads;
    const int row = warp * kRowsPerWarp + row_in_warp;
    const int k_block = lane_in_row / kLanesPerKBlock;
    const int lane_in_block = lane_in_row % kLanesPerKBlock;
    const int depth_base =
        k_block * 32 + lane_in_block * kValuesPerLane;
    const int seq_idx = row_tile_idx * kRowsPerBlock + row;
    const size_t input_base =
        ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
             g.heads +
         head_idx) * C::DvoDim;
    const size_t mixed_output_base =
        ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
             g.heads +
         head_idx) * kMixedOperandBytes;
    const bool write_padded_dv_tail =
        TK_FA4_BWD_MIXED_DV_PADDED_DO_TAIL &&
        (seq_idx & 127) >= 64;

    // Fixed-scale packing consumes each quartet immediately.  Keeping only
    // four BF16 pairs per operand shortens the tail's live range while the
    // dynamic-scale fallback still retains the complete K32 for its amax.
    uint32_t dout_bf16[
        kFixedScaleE8M0 != 0 ? 4 : kPairsPerLane
    ];
    uint32_t v_bf16[kFixedScaleE8M0 != 0 ? 4 : kPairsPerLane];
    uint32_t dout_words[kPairsPerLane / 4];
    uint32_t v_words[kPairsPerLane / 4];
    uint32_t dout_fp8_words[kPairsPerLane / 2];
    uint32_t v_fp8_words[kPairsPerLane / 2];
    float dout_amax = 0.0f;
    float v_amax = 0.0f;
    float dpsum = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < kPairsPerLane; ++pair) {
        const int depth = depth_base + 2 * pair;
        const bf16_2 out_pair = *reinterpret_cast<const bf16_2 *>(
            g.o_ptr + input_base + depth
        );
        const bf16_2 dout_pair = *reinterpret_cast<const bf16_2 *>(
            g.dout_ptr + input_base + depth
        );
        bf16_2 v_pair;
        if constexpr (QuantizeV) {
            v_pair = *reinterpret_cast<const bf16_2 *>(
                g.v_ptr + input_base + depth
            );
        }
        const uint16_t dout_fp8 =
            bwd_cute16_kernel_candidate::detail::
                cta2_role_split_convert_scaled_bf16_pair_to_fp8<
                    bwd_cute16_kernel_candidate::detail::
                        kCta2DenseFp8DpOperandScaleBf16PairDelta
                >(dout_pair);
        if ((pair & 1) == 0) {
            dout_fp8_words[pair / 2] = dout_fp8;
        } else {
            dout_fp8_words[pair / 2] |=
                static_cast<uint32_t>(dout_fp8) << 16;
        }
        if constexpr (QuantizeV) {
            if (k_block < 2) {
                // Only the first K64 remains E4M3 in the mixed dP
                // descriptor. Do not materialize the FP8 bytes that the
                // packed tail replaces.
                const uint16_t v_fp8 =
                    bwd_cute16_kernel_candidate::detail::
                        cta2_role_split_convert_scaled_bf16_pair_to_fp8<
                            bwd_cute16_kernel_candidate::detail::
                                kCta2DenseFp8DpOperandScaleBf16PairDelta
                        >(v_pair);
                if ((pair & 1) == 0) {
                    v_fp8_words[pair / 2] = v_fp8;
                } else {
                    v_fp8_words[pair / 2] |=
                        static_cast<uint32_t>(v_fp8) << 16;
                }
            }
        }
        const float2 out_values = __bfloat1622float2(out_pair);
        const float2 dout_values = __bfloat1622float2(dout_pair);
        dpsum = fmaf(
            out_values.x,
            dout_values.x,
            fmaf(out_values.y, dout_values.y, dpsum)
        );
        if constexpr (kFixedScaleE8M0 != 0) {
            if (k_block >= 2 || write_padded_dv_tail) {
                const int quartet_pair = pair & 3;
                dout_bf16[quartet_pair] =
                    *reinterpret_cast<const uint32_t *>(&dout_pair);
                if (quartet_pair == 3) {
                    const int word = pair >> 2;
                    dout_words[word] =
                        preprocess_pack_fixed_bf16x8_e2m1<
                            kTailPackScaleE8M0
                        >(
                            dout_bf16[0],
                            dout_bf16[1],
                            dout_bf16[2],
                            dout_bf16[3]
                        );
                }
            }
            if constexpr (QuantizeV) {
                if (k_block >= 2) {
                    const int quartet_pair = pair & 3;
                    v_bf16[quartet_pair] =
                        *reinterpret_cast<const uint32_t *>(&v_pair);
                    if (quartet_pair == 3) {
                        const int word = pair >> 2;
                        v_words[word] =
                            preprocess_pack_fixed_bf16x8_e2m1<
                                kTailPackScaleE8M0
                            >(
                                v_bf16[0],
                                v_bf16[1],
                                v_bf16[2],
                                v_bf16[3]
                            );
                    }
                }
            }
        } else if (k_block >= 2) {
            {
                dout_bf16[pair] =
                    *reinterpret_cast<const uint32_t *>(&dout_pair);
                v_bf16[pair] =
                    *reinterpret_cast<const uint32_t *>(&v_pair);
            }
            const float2 v_values = QuantizeV
                ? __bfloat1622float2(v_pair)
                : make_float2(0.0f, 0.0f);
            dout_amax = fmaxf(
                dout_amax,
                fmaxf(fabsf(dout_values.x), fabsf(dout_values.y))
            );
            if constexpr (QuantizeV) {
                v_amax = fmaxf(
                    v_amax,
                    fmaxf(fabsf(v_values.x), fabsf(v_values.y))
                );
            }
        }
    }
    if constexpr (kPairsPerLane == 4) {
        const uint2 dout_dv_fp8_vector = make_uint2(
            dout_fp8_words[0],
            dout_fp8_words[1]
        );
        *reinterpret_cast<uint2 *>(
            g.dout_dv_fp8_ptr + input_base + depth_base
        ) = dout_dv_fp8_vector;
        if (k_block < 2) {
            *reinterpret_cast<uint2 *>(
                g.dout_dp_mixed_ptr + mixed_output_base + depth_base
            ) = dout_dv_fp8_vector;
            if constexpr (QuantizeV) {
                const uint2 v_fp8_vector = make_uint2(
                    v_fp8_words[0],
                    v_fp8_words[1]
                );
                *reinterpret_cast<uint2 *>(
                    g.v_dp_mixed_ptr + mixed_output_base + depth_base
                ) = v_fp8_vector;
            }
        }
    } else {
        const uint4 dout_fp8_vector = make_uint4(
            dout_fp8_words[0],
            dout_fp8_words[1],
            dout_fp8_words[2],
            dout_fp8_words[3]
        );
        *reinterpret_cast<uint4 *>(
            g.dout_dv_fp8_ptr + input_base + depth_base
        ) = dout_fp8_vector;
        if (k_block < 2) {
            *reinterpret_cast<uint4 *>(
                g.dout_dp_mixed_ptr + mixed_output_base + depth_base
            ) = dout_fp8_vector;
            if constexpr (QuantizeV) {
                *reinterpret_cast<uint4 *>(
                    g.v_dp_mixed_ptr + mixed_output_base + depth_base
                ) = make_uint4(
                    v_fp8_words[0],
                    v_fp8_words[1],
                    v_fp8_words[2],
                    v_fp8_words[3]
                );
            }
        }
    }
    #pragma unroll
    for (int offset = kRowThreads / 2; offset > 0; offset >>= 1) {
        dpsum += __shfl_down_sync(0xffffffff, dpsum, offset, kRowThreads);
    }
    if (lane_in_row == 0) {
        const size_t stats_offset =
            (static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
                g.heads +
            head_idx;
        g.dpsum[{batch_idx, head_idx, 0, seq_idx}] = dpsum * 16.0f;
        g.lse_log2[{batch_idx, head_idx, 0, seq_idx}] =
            g.lse_ptr[stats_offset] * kLog2E;
    }
    if constexpr (kFixedScaleE8M0 == 0) {
        #pragma unroll
        for (int offset = kLanesPerKBlock / 2; offset > 0; offset >>= 1) {
            dout_amax = fmaxf(
                dout_amax,
                __shfl_xor_sync(
                    0xffffffff,
                    dout_amax,
                    offset,
                    kLanesPerKBlock
                )
            );
            v_amax = fmaxf(
                v_amax,
                __shfl_xor_sync(
                    0xffffffff,
                    v_amax,
                    offset,
                    kLanesPerKBlock
                )
            );
        }
    }
    if constexpr (TK_FA4_BWD_MIXED_DV_PADDED_DO_TAIL) {
        if (TK_FA4_BWD_MIXED_DV_PADDED_DO_WRITE &&
            write_padded_dv_tail) {
            // CTA rank 0 loads the first 64 query rows, so those rows carry
            // the FP4 values from rows +64 for feature columns 0..63.  Rank
            // 1 loads the second 64 rows, whose padding carries its own
            // feature columns 64..127.  Both arrive as query-major K64xN64.
            const int target_seq_idx = k_block < 2
                ? seq_idx - 64
                : seq_idx;
            const int target_feature_block = k_block < 2
                ? k_block
                : k_block - 2;
            const size_t padded_row_base =
                ((static_cast<size_t>(batch_idx) * g.seq_len +
                  target_seq_idx) * g.heads + head_idx) *
                kMixedOperandBytes;
            const int padded_byte_base =
                96 + target_feature_block * 16 +
                lane_in_block * kPairsPerLane;
            *reinterpret_cast<uint32_t *>(
                g.dout_dp_mixed_ptr + padded_row_base + padded_byte_base
            ) = dout_words[0];
        }
    }
    if (k_block < 2) {
        return;
    }
    const uint8_t dout_e8m0 = kFixedScaleE8M0 != 0
        ? static_cast<uint8_t>(kFixedScaleE8M0)
        : bwd_cute16_kernel_candidate::detail::
              cta2_role_split_float_to_e8m0_rte(dout_amax);
    const uint8_t v_e8m0 = kFixedScaleE8M0 != 0
        ? static_cast<uint8_t>(kFixedScaleE8M0)
        : bwd_cute16_kernel_candidate::detail::
              cta2_role_split_float_to_e8m0_rte(v_amax);
    if constexpr (kFixedScaleE8M0 == 0) {
        // quant_multiplier_pow2 reconstructs each operand at 4*x, matching
        // the retained FP8 dP representation without a per-element fixup.
        const float dout_multiplier =
            bwd_cute16_kernel_candidate::detail::
                cta2_role_split_e8m0_quant_multiplier_pow2(dout_e8m0);
        const float v_multiplier =
            bwd_cute16_kernel_candidate::detail::
                cta2_role_split_e8m0_quant_multiplier_pow2(v_e8m0);
        #pragma unroll
        for (int word = 0; word < kPairsPerLane / 4; ++word) {
            float dout_values[8];
            float v_values[8];
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int source_pair = 4 * word + pair;
                const bf16_2 dout_pair = *reinterpret_cast<const bf16_2 *>(
                    &dout_bf16[source_pair]
                );
                const bf16_2 v_pair = *reinterpret_cast<const bf16_2 *>(
                    &v_bf16[source_pair]
                );
                const float2 dout_f32 = __bfloat1622float2(dout_pair);
                const float2 v_f32 = __bfloat1622float2(v_pair);
                dout_values[2 * pair + 0] = dout_f32.x * dout_multiplier;
                dout_values[2 * pair + 1] = dout_f32.y * dout_multiplier;
                v_values[2 * pair + 0] = v_f32.x * v_multiplier;
                v_values[2 * pair + 1] = v_f32.y * v_multiplier;
            }
            dout_words[word] =
                bwd_cute16_kernel_candidate::detail::
                    cta2_role_split_pack_e2m1_8(
                        dout_values[0], dout_values[1],
                        dout_values[2], dout_values[3],
                        dout_values[4], dout_values[5],
                        dout_values[6], dout_values[7]
                    );
            v_words[word] =
                bwd_cute16_kernel_candidate::detail::
                    cta2_role_split_pack_e2m1_8(
                        v_values[0], v_values[1],
                        v_values[2], v_values[3],
                        v_values[4], v_values[5],
                        v_values[6], v_values[7]
                    );
        }
    }

    // The shared operand is still loaded as one 128-byte E4M3 row.  The
    // FP4 descriptor reads chunk 2, so place the two packed K32 blocks at
    // byte offsets 64 and 80 and leave the first K64 untouched.
    const size_t mixed_row_base = mixed_output_base;
    const int packed_byte_base =
        64 + (k_block - 2) * 16 + lane_in_block * kPairsPerLane;
    if constexpr (kPairsPerLane == 4) {
        *reinterpret_cast<uint32_t *>(
            g.dout_dp_mixed_ptr + mixed_row_base + packed_byte_base
        ) = dout_words[0];
        if constexpr (QuantizeV) {
            *reinterpret_cast<uint32_t *>(
                g.v_dp_mixed_ptr + mixed_row_base + packed_byte_base
            ) = v_words[0];
        }
    } else {
        *reinterpret_cast<uint2 *>(
            g.dout_dp_mixed_ptr + mixed_row_base + packed_byte_base
        ) = make_uint2(dout_words[0], dout_words[1]);
        if constexpr (QuantizeV) {
            *reinterpret_cast<uint2 *>(
                g.v_dp_mixed_ptr + mixed_row_base + packed_byte_base
            ) = make_uint2(v_words[0], v_words[1]);
        }
    }

    if constexpr (kFixedScaleE8M0 == 0) {
        if (lane_in_block == 0) {
            const size_t scale_record_base =
                ((static_cast<size_t>(batch_idx) *
                      (g.seq_len / kMxScalePageRows) +
                  seq_idx / kMxScalePageRows) *
                     g.heads +
                 head_idx) * 512;
            const int scale_index =
                bwd_cute16_kernel_candidate::detail::
                    cta2_role_split_mxfp4_scale_swizzle_idx(
                        seq_idx % kMxScalePageRows,
                        k_block
                    );
            const int mirror_scale_index =
                bwd_cute16_kernel_candidate::detail::
                    cta2_role_split_mxfp4_scale_swizzle_idx(
                        seq_idx % kMxScalePageRows,
                        k_block - 2
                    );
            g.dout_dp_scale_ptr[scale_record_base + scale_index] =
                dout_e8m0;
            if constexpr (QuantizeV) {
                g.v_dp_scale_ptr[scale_record_base + scale_index] = v_e8m0;
            }
            // The mixed command uses SFID 2, but seed the inactive pair too.
            // This prevents stale at::empty bytes from becoming exponent
            // inputs if the fetch spans the complete K128 page.
            g.dout_dp_scale_ptr[scale_record_base + mirror_scale_index] =
                dout_e8m0;
            if constexpr (QuantizeV) {
                g.v_dp_scale_ptr[scale_record_base + mirror_scale_index] =
                    v_e8m0;
            }
        }
    }
}

template <typename C, bool QuantizeV = true>
inline void launch_preprocess_mixed_fp8_mxfp4_dp(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &v,
    at::Tensor &lse,
    at::Tensor &dout_dv_fp8,
    at::Tensor &dout_dp_mixed,
    at::Tensor &v_dp_mixed,
    at::Tensor &dout_dp_scale,
    at::Tensor &v_dp_scale,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    TORCH_CHECK(
        dout_dv_fp8.sizes() == dout.sizes() &&
            dout_dv_fp8.scalar_type() ==
                at::ScalarType::Float8_e4m3fn,
        "mixed dP preprocess requires a full E4M3 dO workspace for dV"
    );
    TORCH_CHECK(
        dout_dp_mixed.sizes() == at::IntArrayRef({
            dout.size(0),
            dout.size(1),
            dout.size(2),
            TK_FA4_BWD_MIXED_DP_COMPACT_96B_OPERANDS ? 96 : dout.size(3)
        }) &&
            v_dp_mixed.sizes() == dout_dp_mixed.sizes() &&
            dout_dp_mixed.scalar_type() ==
                at::ScalarType::Float8_e4m3fn &&
            v_dp_mixed.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "mixed dP preprocess requires 96-byte compact E4M3/E2M1 workspaces"
    );
    TORCH_CHECK(
        dout_dp_scale.scalar_type() == at::ScalarType::Byte &&
            v_dp_scale.scalar_type() == at::ScalarType::Byte &&
            dout_dp_scale.sizes() == at::IntArrayRef({
                dout.size(0), dout.size(1) / 128, dout.size(2), 512
            }) &&
            v_dp_scale.sizes() == dout_dp_scale.sizes(),
        "mixed dP preprocess requires E8M0 pages [B, S/128, H, 512]"
    );
    using G = preprocess_mixed_dp_half_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        reinterpret_cast<fp8e4m3 *>(dout_dv_fp8.data_ptr()),
        reinterpret_cast<uint8_t *>(dout_dp_mixed.data_ptr()),
        reinterpret_cast<uint8_t *>(v_dp_mixed.data_ptr()),
        reinterpret_cast<uint8_t *>(dout_dp_scale.data_ptr()),
        reinterpret_cast<uint8_t *>(v_dp_scale.data_ptr()),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    constexpr int kRowsPerBlock =
        4 * kWarpThreads / TK_FA4_BWD_MIXED_DP_PREPROCESS_ROW_THREADS;
    dim3 grid(
        out.size(1) / kRowsPerBlock,
        out.size(2),
        out.size(0)
    );
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_mixed_dp_half_kernel<C, QuantizeV><<<
        grid,
        4 * kWarpThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
struct prepack_mixed_v_globals {
    const bf16 *v_ptr;
    uint8_t *packed_ptr;
    int seq_len;
    int heads;
};

template <typename C>
__global__ __launch_bounds__(4 * kWarpThreads, 8)
void prepack_mixed_v_kernel(
    const __grid_constant__ prepack_mixed_v_globals<C> g
) {
    static_assert(C::DvoDim == 128);
    static_assert(TK_FA4_BWD_MIXED_DP_FIXED_SCALE_E8M0 != 0);
    constexpr int kRowThreads = 16;
    constexpr int kRowsPerBlock = 4 * kWarpThreads / kRowThreads;
    constexpr int kPairsPerLane = 4;
    const int thread = static_cast<int>(threadIdx.x);
    const int row_in_block = thread / kRowThreads;
    const int lane_in_row = thread % kRowThreads;
    const int k_block = lane_in_row / 4;
    const int lane_in_block = lane_in_row & 3;
    const int seq_idx =
        static_cast<int>(blockIdx.x) * kRowsPerBlock + row_in_block;
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int depth_base = k_block * 32 + lane_in_block * 8;
    const size_t row_base =
        ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
             g.heads +
         head_idx) * C::DvoDim;
    uint32_t pairs[kPairsPerLane];
    #pragma unroll
    for (int pair = 0; pair < kPairsPerLane; ++pair) {
        pairs[pair] = *reinterpret_cast<const uint32_t *>(
            g.v_ptr + row_base + depth_base + 2 * pair
        );
    }
    if (k_block < 2) {
        uint16_t fp8_pairs[kPairsPerLane];
        #pragma unroll
        for (int pair = 0; pair < kPairsPerLane; ++pair) {
            const bf16_2 value =
                *reinterpret_cast<const bf16_2 *>(&pairs[pair]);
            fp8_pairs[pair] =
                bwd_cute16_kernel_candidate::detail::
                    cta2_role_split_convert_scaled_bf16_pair_to_fp8<
                        bwd_cute16_kernel_candidate::detail::
                            kCta2DenseFp8DpOperandScaleBf16PairDelta
                    >(value);
        }
        *reinterpret_cast<uint2 *>(
            g.packed_ptr + row_base + depth_base
        ) = make_uint2(
            static_cast<uint32_t>(fp8_pairs[0]) |
                (static_cast<uint32_t>(fp8_pairs[1]) << 16),
            static_cast<uint32_t>(fp8_pairs[2]) |
                (static_cast<uint32_t>(fp8_pairs[3]) << 16)
        );
    } else {
        const uint32_t packed =
            preprocess_pack_fixed_bf16x8_e2m1<
                TK_FA4_BWD_MIXED_DP_FIXED_SCALE_E8M0
            >(pairs[0], pairs[1], pairs[2], pairs[3]);
        const int packed_byte =
            64 + (k_block - 2) * 16 + lane_in_block * 4;
        *reinterpret_cast<uint32_t *>(
            g.packed_ptr + row_base + packed_byte
        ) = packed;
    }
    if (lane_in_row < 8) {
        *reinterpret_cast<uint32_t *>(
            g.packed_ptr + row_base + 96 + 4 * lane_in_row
        ) = 0;
    }
}

template <typename C>
inline void launch_prepack_mixed_v(
    at::Tensor &v,
    at::Tensor &packed_v
) {
    TORCH_CHECK(
        v.scalar_type() == at::ScalarType::BFloat16 &&
            v.is_contiguous() && v.dim() == 4 &&
            v.size(3) == C::DvoDim,
        "mixed V prepack requires contiguous BF16 [B,S,H,128] input"
    );
    TORCH_CHECK(
        packed_v.sizes() == v.sizes() &&
            packed_v.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            packed_v.is_contiguous(),
        "mixed V prepack requires shape-matched contiguous byte workspace"
    );
    using G = prepack_mixed_v_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(v.data_ptr()),
        reinterpret_cast<uint8_t *>(packed_v.data_ptr()),
        static_cast<int>(v.size(1)),
        static_cast<int>(v.size(2)),
    };
    constexpr int kRowsPerBlock = 4 * kWarpThreads / 16;
    TORCH_CHECK(
        v.size(1) % kRowsPerBlock == 0,
        "mixed V prepack requires sequence divisible by eight"
    );
    const dim3 grid(v.size(1) / kRowsPerBlock, v.size(2), v.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    prepack_mixed_v_kernel<C><<<grid, 4 * kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
struct preprocess_mxfp4_dp_globals {
    using stats_tile = col_vec<st_fl<kRefTileM, C::DvoDim>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    const bf16 *o_ptr;
    const bf16 *dout_ptr;
    const bf16 *v_ptr;
    const float *lse_ptr;
    uint8_t *dout_mxfp4_ptr;
    uint8_t *v_mxfp4_ptr;
    uint8_t *dout_mxfp4_scale_ptr;
    uint8_t *v_mxfp4_scale_ptr;
    stats_gl dpsum;
    stats_gl lse_log2;
    int seq_len;
    int heads;
};

template <typename C>
__global__ __launch_bounds__(4 * kWarpThreads, 8)
void preprocess_mxfp4_dp_kernel(
    const __grid_constant__ preprocess_mxfp4_dp_globals<C> g
) {
    static_assert(C::DvoDim == 128);
    constexpr int kRowThreads = 8;
    constexpr int kRowsPerWarp = kWarpThreads / kRowThreads;
    constexpr int kRowsPerBlock = 4 * kRowsPerWarp;
    constexpr int kMxScalePageRows = 128;
    constexpr int kPackedDepth = C::DvoDim / 2;
    const int row_tile_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int warp = static_cast<int>(threadIdx.x) / kWarpThreads;
    const int lane = laneid();
    const int row_in_warp = lane / kRowThreads;
    const int lane_in_row = lane % kRowThreads;
    const int row = warp * kRowsPerWarp + row_in_warp;

    // Two adjacent lanes own one contiguous K32 block.  Each lane packs its
    // sixteen values into two naturally aligned 32-bit stores, while the
    // pair shares the exact E8M0 maximum used by the tensor-core operand.
    const int k_block = lane_in_row >> 1;
    const int lane_in_block = lane_in_row & 1;
    const int depth_base = k_block * 32 + lane_in_block * 16;
    const int seq_idx = row_tile_idx * kRowsPerBlock + row;
    const size_t input_base =
        ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
             g.heads +
         head_idx) * C::DvoDim;

    uint32_t dout_bf16[8];
    uint32_t v_bf16[8];
    float dout_amax = 0.0f;
    float v_amax = 0.0f;
    float dpsum = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const int depth = depth_base + 2 * pair;
        const bf16_2 o_pair = *reinterpret_cast<const bf16_2 *>(
            g.o_ptr + input_base + depth
        );
        const bf16_2 dout_pair = *reinterpret_cast<const bf16_2 *>(
            g.dout_ptr + input_base + depth
        );
        const bf16_2 v_pair = *reinterpret_cast<const bf16_2 *>(
            g.v_ptr + input_base + depth
        );
        dout_bf16[pair] = *reinterpret_cast<const uint32_t *>(&dout_pair);
        v_bf16[pair] = *reinterpret_cast<const uint32_t *>(&v_pair);
        const float2 o_values = __bfloat1622float2(o_pair);
        const float2 dout_values = __bfloat1622float2(dout_pair);
        const float2 v_values = __bfloat1622float2(v_pair);
        dpsum += o_values.x * dout_values.x +
            o_values.y * dout_values.y;
        dout_amax = fmaxf(
            dout_amax,
            fmaxf(fabsf(dout_values.x), fabsf(dout_values.y))
        );
        v_amax = fmaxf(
            v_amax,
            fmaxf(fabsf(v_values.x), fabsf(v_values.y))
        );
    }
    dout_amax = fmaxf(
        dout_amax,
        __shfl_xor_sync(0xffffffff, dout_amax, 1, 2)
    );
    v_amax = fmaxf(
        v_amax,
        __shfl_xor_sync(0xffffffff, v_amax, 1, 2)
    );
    const uint8_t dout_e8m0 =
        bwd_cute16_kernel_candidate::detail::
            cta2_role_split_float_to_e8m0_rte(dout_amax);
    const uint8_t v_e8m0 =
        bwd_cute16_kernel_candidate::detail::
            cta2_role_split_float_to_e8m0_rte(v_amax);
    const float dout_multiplier =
        bwd_cute16_kernel_candidate::detail::
            cta2_role_split_e8m0_quant_multiplier_pow2(dout_e8m0);
    const float v_multiplier =
        bwd_cute16_kernel_candidate::detail::
            cta2_role_split_e8m0_quant_multiplier_pow2(v_e8m0);

    uint32_t dout_words[2];
    uint32_t v_words[2];
    #pragma unroll
    for (int word = 0; word < 2; ++word) {
        float dout_values[8];
        float v_values[8];
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int source_pair = 4 * word + pair;
            const bf16_2 dout_pair = *reinterpret_cast<const bf16_2 *>(
                &dout_bf16[source_pair]
            );
            const bf16_2 v_pair = *reinterpret_cast<const bf16_2 *>(
                &v_bf16[source_pair]
            );
            const float2 dout_f32 = __bfloat1622float2(dout_pair);
            const float2 v_f32 = __bfloat1622float2(v_pair);
            dout_values[2 * pair + 0] = dout_f32.x * dout_multiplier;
            dout_values[2 * pair + 1] = dout_f32.y * dout_multiplier;
            v_values[2 * pair + 0] = v_f32.x * v_multiplier;
            v_values[2 * pair + 1] = v_f32.y * v_multiplier;
        }
        dout_words[word] =
            bwd_cute16_kernel_candidate::detail::
                cta2_role_split_pack_e2m1_8(
                    dout_values[0],
                    dout_values[1],
                    dout_values[2],
                    dout_values[3],
                    dout_values[4],
                    dout_values[5],
                    dout_values[6],
                    dout_values[7]
                );
        v_words[word] =
            bwd_cute16_kernel_candidate::detail::
                cta2_role_split_pack_e2m1_8(
                    v_values[0],
                    v_values[1],
                    v_values[2],
                    v_values[3],
                    v_values[4],
                    v_values[5],
                    v_values[6],
                    v_values[7]
                );
    }
    const size_t packed_row_base =
        ((static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
             g.heads +
         head_idx) * kPackedDepth;
    const int packed_depth_base = k_block * 16 + lane_in_block * 8;
    *reinterpret_cast<uint2 *>(
        g.dout_mxfp4_ptr + packed_row_base + packed_depth_base
    ) = make_uint2(dout_words[0], dout_words[1]);
    *reinterpret_cast<uint2 *>(
        g.v_mxfp4_ptr + packed_row_base + packed_depth_base
    ) = make_uint2(v_words[0], v_words[1]);

    if (lane_in_block == 0) {
        const size_t scale_record_base =
            ((static_cast<size_t>(batch_idx) *
                  (g.seq_len / kMxScalePageRows) +
              seq_idx / kMxScalePageRows) *
                 g.heads +
             head_idx) * 512;
        const int scale_index =
            bwd_cute16_kernel_candidate::detail::
                cta2_role_split_mxfp4_scale_swizzle_idx(
                    seq_idx % kMxScalePageRows,
                    k_block
                );
        g.dout_mxfp4_scale_ptr[scale_record_base + scale_index] =
            dout_e8m0;
        g.v_mxfp4_scale_ptr[scale_record_base + scale_index] = v_e8m0;
    }

    #pragma unroll
    for (int offset = kRowThreads / 2; offset > 0; offset >>= 1) {
        dpsum += __shfl_down_sync(
            0xffffffff,
            dpsum,
            offset,
            kRowThreads
        );
    }
    if (lane_in_row == 0) {
        const size_t lse_offset =
            (static_cast<size_t>(batch_idx) * g.seq_len + seq_idx) *
                g.heads +
            head_idx;
        g.dpsum[{batch_idx, head_idx, 0, seq_idx}] = dpsum * 16.0f;
        g.lse_log2[{batch_idx, head_idx, 0, seq_idx}] =
            g.lse_ptr[lse_offset] * kLog2E;
    }
}

template <typename C>
struct preprocess_mxfp4_dout_globals {
    using stats_tile = col_vec<st_fl<kRefTileM, C::DvoDim>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    const bf16 *o_ptr;
    const bf16 *dout_ptr;
    const bf16 *v_ptr;
    const float *lse_ptr;
    fp8e4m3 *dout_dp_fp8_ptr;
    fp8e4m3 *v_fp8_ptr;
    uint8_t *dout_mxfp4_ptr;
    uint8_t *dout_mxfp4_scale_ptr;
    stats_gl dpsum;
    stats_gl lse_log2;
    int seq_len;
    int heads;
};

template <typename C, bool MatchFp8DvScale = false>
__global__ __launch_bounds__(512, 2)
void preprocess_mxfp4_dout_pack_kernel(
    const __grid_constant__ preprocess_mxfp4_dout_globals<C> g
) {
    static_assert(C::DvoDim == 128);
    constexpr int kFixedScaleE8M0 = MatchFp8DvScale
        ? TK_FA4_BWD_MIXED_DV_FIXED_DO_SCALE_E8M0
        : 0;
    constexpr int kRows = 128;
    constexpr int kPackedRows = kRows / 2;
    const int q_tile_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int thread = static_cast<int>(threadIdx.x);
    const int q_row_base = q_tile_idx * kRows;
    // Four threads per feature independently produce its four K32 blocks.
    // This turns the serial K128 pack into sixteen fully occupied warps; the
    // mandatory statistics/FP8 work runs in its already efficient kernel.
    const int feature = thread & (C::DvoDim - 1);
    const int k_block = thread / C::DvoDim;
#if TK_FA4_BWD_MXFP4_PV_INTERLEAVED_KBLOCKS
    // Match the P producer's physical reduction order.  Swapping the middle
    // K32 blocks groups the two blocks owned by each producer warp without
    // changing the dV dot product.
#if TK_FA4_BWD_MXFP4_PV_INTERLEAVED_KBLOCKS == 2
    const int physical_k_block = k_block == 0 ? 0 : k_block - 1 +
        (k_block == 1 ? 3 : 0);
#else
    const int physical_k_block =
        ((k_block & 1) << 1) | ((k_block & 2) >> 1);
#endif
#else
    const int physical_k_block = k_block;
#endif
    const size_t packed_feature_base =
        (((static_cast<size_t>(batch_idx) * g.heads + head_idx) *
              C::DvoDim +
          feature) *
         (g.seq_len / 2)) +
        q_tile_idx * kPackedRows;
    const size_t scale_record_base =
        ((static_cast<size_t>(batch_idx) * (g.seq_len / kRows) +
          q_tile_idx) *
             g.heads +
         head_idx) * 512;

    uint32_t values_bf16[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < 16; ++pair) {
        const int row0 = q_row_base + k_block * 32 + 2 * pair;
        const int row1 = row0 + 1;
        const size_t offset0 =
            ((static_cast<size_t>(batch_idx) * g.seq_len + row0) *
                 g.heads +
             head_idx) *
                C::DvoDim +
            feature;
        const size_t offset1 =
            ((static_cast<size_t>(batch_idx) * g.seq_len + row1) *
                 g.heads +
             head_idx) *
                C::DvoDim +
            feature;
        const uint16_t bits0 = *reinterpret_cast<const uint16_t *>(
            g.dout_ptr + offset0
        );
        const uint16_t bits1 = *reinterpret_cast<const uint16_t *>(
            g.dout_ptr + offset1
        );
        values_bf16[pair] =
            static_cast<uint32_t>(bits0) |
            (static_cast<uint32_t>(bits1) << 16);
        const bf16_2 values =
            *reinterpret_cast<const bf16_2 *>(&values_bf16[pair]);
        const float2 values_f32 = __bfloat1622float2(values);
        if constexpr (kFixedScaleE8M0 == 0) {
            block_amax = fmaxf(
                block_amax,
                fmaxf(fabsf(values_f32.x), fabsf(values_f32.y))
            );
        }
    }

    const uint8_t e8m0 = kFixedScaleE8M0 != 0
        ? static_cast<uint8_t>(kFixedScaleE8M0)
        : bwd_cute16_kernel_candidate::detail::
              cta2_role_split_float_to_e8m0_rte(block_amax);
    const float multiplier = MatchFp8DvScale
        ? bwd_cute16_kernel_candidate::detail::
              cta2_role_split_e8m0_quant_multiplier_pow2(e8m0)
        : bwd_cute16_kernel_candidate::detail::
              cta2_role_split_e8m0_quant_multiplier(e8m0);
    uint32_t packed_words[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        float values[8];
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const bf16_2 packed = *reinterpret_cast<const bf16_2 *>(
                &values_bf16[4 * word + pair]
            );
            const float2 converted = __bfloat1622float2(packed);
            values[2 * pair + 0] = converted.x * multiplier;
            values[2 * pair + 1] = converted.y * multiplier;
        }
        packed_words[word] =
            bwd_cute16_kernel_candidate::detail::
                cta2_role_split_pack_e2m1_8(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7]
                );
    }
    *reinterpret_cast<uint4 *>(
        g.dout_mxfp4_ptr + packed_feature_base + physical_k_block * 16
    ) = make_uint4(
        packed_words[0],
        packed_words[1],
        packed_words[2],
        packed_words[3]
    );
    g.dout_mxfp4_scale_ptr[
        scale_record_base +
        bwd_cute16_kernel_candidate::detail::
            cta2_role_split_mxfp4_scale_swizzle_idx(
                feature,
                physical_k_block
            )
    ] = MatchFp8DvScale && e8m0 != 0
        ? static_cast<uint8_t>(min(static_cast<int>(e8m0) + 3, 254))
        : e8m0;
}

template <typename C>
inline void launch_preprocess_mxfp4_dout_v(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &v,
    at::Tensor &lse,
    at::Tensor &dout_dp_fp8,
    at::Tensor &v_fp8,
    at::Tensor &dout_mxfp4,
    at::Tensor &dout_mxfp4_scale,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    TORCH_CHECK(
        dout_dp_fp8.sizes() == dout.sizes() &&
            v_fp8.sizes() == v.sizes() &&
            dout_dp_fp8.scalar_type() ==
                at::ScalarType::Float8_e4m3fn &&
            v_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "MXFP4-dV preprocess requires shape-matched E4M3 dP operands"
    );
    TORCH_CHECK(
        dout_mxfp4.scalar_type() == at::ScalarType::Byte &&
            dout_mxfp4.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(2),
                dout.size(3),
                dout.size(1) / 2
            }),
        "MXFP4-dV preprocess requires packed dO [B, H, D, S/2]"
    );
    TORCH_CHECK(
        dout_mxfp4_scale.scalar_type() == at::ScalarType::Byte &&
            dout_mxfp4_scale.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(1) / 128,
                dout.size(2),
                512
        }),
        "MXFP4-dV preprocess requires E8M0 scales [B, S/128, H, 512]"
    );
    // Keep the mandatory dPsum/LSE and E4M3 dP/V pass on its compact
    // four-warp implementation.  The MXFP4 pack below can then devote all
    // sixteen warps to independent (feature, K32) blocks.
    launch_preprocess_fp8_dout_v<C>(
        out,
        dout,
        v,
        lse,
        dout_dp_fp8,
        v_fp8,
        dpsum,
        lse_log2
    );
    using G = preprocess_mxfp4_dout_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        reinterpret_cast<fp8e4m3 *>(dout_dp_fp8.data_ptr()),
        reinterpret_cast<fp8e4m3 *>(v_fp8.data_ptr()),
        reinterpret_cast<uint8_t *>(dout_mxfp4.data_ptr()),
        reinterpret_cast<uint8_t *>(dout_mxfp4_scale.data_ptr()),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / 128, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_mxfp4_dout_pack_kernel<C><<<grid, 512, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_preprocess_mxfp4_dout_mixed_dv_tail(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &v,
    at::Tensor &lse,
    at::Tensor &dout_mxfp4,
    at::Tensor &dout_mxfp4_scale,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    TORCH_CHECK(
        dout_mxfp4.scalar_type() == at::ScalarType::Byte &&
            dout_mxfp4.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(2),
                dout.size(3),
                dout.size(1) / 2
            }),
        "mixed dV requires packed dO [B, H, D, S/2]"
    );
    TORCH_CHECK(
        dout_mxfp4_scale.scalar_type() == at::ScalarType::Byte &&
            dout_mxfp4_scale.sizes() == at::IntArrayRef({
                dout.size(0), dout.size(1) / 128, dout.size(2), 512
            }),
        "mixed dV requires E8M0 scales [B, S/128, H, 512]"
    );
    using G = preprocess_mxfp4_dout_globals<C>;
    G g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        nullptr,
        nullptr,
        reinterpret_cast<uint8_t *>(dout_mxfp4.data_ptr()),
        reinterpret_cast<uint8_t *>(dout_mxfp4_scale.data_ptr()),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename G::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    dim3 grid(out.size(1) / 128, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_mxfp4_dout_pack_kernel<C, true><<<
        grid,
        512,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_preprocess_mxfp4_dp_dv(
    at::Tensor &out,
    at::Tensor &dout,
    at::Tensor &v,
    at::Tensor &lse,
    at::Tensor &dout_dp_mxfp4,
    at::Tensor &v_dp_mxfp4,
    at::Tensor &dout_dp_mxfp4_scale,
    at::Tensor &v_dp_mxfp4_scale,
    at::Tensor &dout_dv_mxfp4,
    at::Tensor &dout_dv_mxfp4_scale,
    at::Tensor &dpsum,
    at::Tensor &lse_log2
) {
    TORCH_CHECK(
        dout_dp_mxfp4.scalar_type() == at::ScalarType::Byte &&
            v_dp_mxfp4.scalar_type() == at::ScalarType::Byte &&
            dout_dp_mxfp4.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(1),
                dout.size(2),
                dout.size(3) / 2
            }) &&
            v_dp_mxfp4.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(1),
                dout.size(2),
                dout.size(3) / 2
            }),
        "MXFP4-dP preprocess requires packed dO/V [B, S, H, D/2]"
    );
    TORCH_CHECK(
        dout_dp_mxfp4_scale.scalar_type() == at::ScalarType::Byte &&
            v_dp_mxfp4_scale.scalar_type() == at::ScalarType::Byte &&
            dout_dp_mxfp4_scale.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(1) / 128,
                dout.size(2),
                512
            }) &&
            v_dp_mxfp4_scale.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(1) / 128,
                dout.size(2),
                512
            }),
        "MXFP4-dP preprocess requires E8M0 dO/V scales [B, S/128, H, 512]"
    );
    TORCH_CHECK(
        dout_dv_mxfp4.scalar_type() == at::ScalarType::Byte &&
            dout_dv_mxfp4.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(2),
                dout.size(3),
                dout.size(1) / 2
            }),
        "MXFP4-dV preprocess requires packed dO [B, H, D, S/2]"
    );
    TORCH_CHECK(
        dout_dv_mxfp4_scale.scalar_type() == at::ScalarType::Byte &&
            dout_dv_mxfp4_scale.sizes() == at::IntArrayRef({
                dout.size(0),
                dout.size(1) / 128,
                dout.size(2),
                512
            }),
        "MXFP4-dV preprocess requires E8M0 scales [B, S/128, H, 512]"
    );

    using DpG = preprocess_mxfp4_dp_globals<C>;
    DpG dp_g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        reinterpret_cast<uint8_t *>(dout_dp_mxfp4.data_ptr()),
        reinterpret_cast<uint8_t *>(v_dp_mxfp4.data_ptr()),
        reinterpret_cast<uint8_t *>(dout_dp_mxfp4_scale.data_ptr()),
        reinterpret_cast<uint8_t *>(v_dp_mxfp4_scale.data_ptr()),
        kittens::py::tensor_to_gl<typename DpG::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename DpG::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    const dim3 grid(out.size(1) / 16, out.size(2), out.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    preprocess_mxfp4_dp_kernel<C><<<
        grid,
        4 * kWarpThreads,
        0,
        stream
    >>>(dp_g);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // dV consumes dO with sequence as K, so retain its independent
    // feature-major payload and scale page.  This second O(S) pack replaces
    // the O(S^2) in-loop transpose/quantize work without producing FP8 data.
    using DvG = preprocess_mxfp4_dout_globals<C>;
    DvG dv_g{
        reinterpret_cast<const bf16 *>(out.data_ptr()),
        reinterpret_cast<const bf16 *>(dout.data_ptr()),
        reinterpret_cast<const bf16 *>(v.data_ptr()),
        reinterpret_cast<const float *>(lse.data_ptr()),
        nullptr,
        nullptr,
        reinterpret_cast<uint8_t *>(dout_dv_mxfp4.data_ptr()),
        reinterpret_cast<uint8_t *>(dout_dv_mxfp4_scale.data_ptr()),
        kittens::py::tensor_to_gl<typename DvG::stats_gl>(
            dpsum,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        kittens::py::tensor_to_gl<typename DvG::stats_gl>(
            lse_log2,
            out.size(0),
            out.size(2),
            1,
            out.size(1)
        ),
        static_cast<int>(out.size(1)),
        static_cast<int>(out.size(2)),
    };
    const dim3 dv_grid(
        out.size(1) / 128,
        out.size(2),
        out.size(0)
    );
    preprocess_mxfp4_dout_pack_kernel<C><<<
        dv_grid,
        512,
        0,
        stream
    >>>(dv_g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

namespace detail {

template <typename C>
struct fa4_bshd_direct_dq_globals {
    using q_tma_tile = st_bf<kRefTileM, C::Dqk, true, 64>;
    using do_tma_tile = st_bf<kRefTileM, C::Dvo, true, 64>;
    using dq_chunk_tile = st_fl<kRefTileM, 64>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tma_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, C::Dqk>;
    using v_gl = gl<bf16, -1, -1, -1, C::Dvo>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tma_tile, dim::DEPTH>>;
    using dq_out_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dq_chunk_tile, dim::DEPTH>>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dq_out_gl dq0;
    dq_out_gl dq1;
    dq_out_gl dq2;
    dk_gl dk;
    dv_gl dv;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
    int actual_seq_len;
};

template <typename C>
struct hot_bshd_dkdv_globals {
    using q_tile = st_bf<C::TileRows, C::Dqk>;
    using k_tile = st_bf<C::TileRows, C::Dqk>;
    using v_tile = st_bf<C::TileRows, C::Dvo>;
    using do_tile = st_bf<C::TileRows, C::Dvo>;
    using dk_full_tile = st_fl<C::TileRows, 64>;
    using dv_full_tile = st_fl<C::TileRows, C::Dvo>;
    using dv_tile = st_fl<kRefTileM, C::Dvo>;
    using stats_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;

    using q_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<q_tile, dim::DEPTH>>;
    using k_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<k_tile, dim::DEPTH>>;
    using v_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>;
    using do_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<do_tile, dim::DEPTH>>;
    using dk_gl = gl<float, -1, -1, -1, C::Dqk>;
    using dk_full_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dk_full_tile, dim::DEPTH>>;
    using dv_gl = gl<float, -1, -1, -1, C::Dvo>;
    using dv_full_gl = gl<float, -1, -1, -1, -1, tma::descriptor<dv_full_tile, dim::DEPTH>>;
    using stats_gl = gl<float, -1, -1, -1, -1, stats_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dk_gl dk;
    dk_full_gl dk0_full;
    dk_full_gl dk1_full;
    dk_full_gl dk2_full;
    dv_gl dv;
    dv_full_gl dv_full;
    stats_gl lse_log2;
    stats_gl dpsum;
    float scale;
    float scale_log2e;
    int seq_len;
};

template <typename C, bool CAUSAL>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void fa4_bshd_main_kernel(const __grid_constant__ bwd_fa4::main_globals<C> g) {
    constexpr int q_tiles_buffered = 4;
    using qk_bf_tile = st_bf<kRefTileM, C::Dqk>;
    using v_bf_tile = st_bf<kRefTileM, C::Dvo>;
    using qk_fl_tile = st_fl<kRefTileM, C::Dqk>;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) qk_fl_tile dq_smem[C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = cluster_ctarank();
    const int kv_block_idx = static_cast<int>(clusterIdx().x) * C::ClusterSize + cluster_rank;
    const int num_k_blocks = g.seq_len / (kRefTileN * C::WarpTiles);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }
    const int cluster_group_idx = kv_block_idx / C::ClusterSize;

    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int num_q_blocks = g.seq_len / (kRefTileM * q_tiles_buffered);
    const bool dense_unmasked = !CAUSAL && g.actual_seq_len == g.seq_len;

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, C::Dqk> dk_accum, dq_total, dq_reg, dq_existing;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;

        if (warp < q_tiles_buffered) {
            rt_bf<kRefTileM, C::Dqk> q_stage_reg;
            rt_bf<kRefTileM, C::Dvo> do_stage_reg;
            stats_vec lse_stage_vec, dpsum_stage_vec;
            warp::load<dim::DEPTH>(q_stage_reg, g.q, {batch_idx, q_tile_base + warp, head_idx, 0});
            warp::store(q_smem[warp], q_stage_reg);
            warp::load<dim::DEPTH>(do_stage_reg, g.dout, {batch_idx, q_tile_base + warp, head_idx, 0});
            warp::store(do_smem[warp], do_stage_reg);
            warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_stage_vec);
            warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_stage_vec);
        }
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, kRefTileN> p, dp, ds;
            rt_fl<kRefTileM, C::Dqk> dq_partial;
            stats_vec lse_log2_vec, dpsum_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            bwd_fa4::detail::backward_tile_step<C, CAUSAL>(
                p,
                dp,
                ds,
                dq_partial,
                dk_accum,
                dv_accum,
                q_reg,
                k_reg,
                v_reg,
                do_reg,
                lse_log2_vec,
                dpsum_vec,
                g.scale,
                g.scale_log2e,
                q_tile_idx,
                kv_subtile_idx,
                g.actual_seq_len,
                dense_unmasked
            );

            bwd_fa4::detail::accumulate_dq_partial<C>(
                g,
                dq_partial,
                dq_total,
                dq_reg,
                dq_existing,
                dq_smem,
                batch_idx,
                head_idx,
                q_tile_idx,
                cluster_rank,
                cluster_group_idx,
                warp
            );
        }
        __syncthreads();
    }

    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C, bool CAUSAL>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void fa4_bshd_direct_dq_main_kernel(const __grid_constant__ fa4_bshd_direct_dq_globals<C> g) {
    constexpr int q_tiles_buffered = 4;
    using qk_bf_tile = typename fa4_bshd_direct_dq_globals<C>::q_tma_tile;
    using v_bf_tile = typename fa4_bshd_direct_dq_globals<C>::do_tma_tile;
    using qk_fl_tile = st_fl<kRefTileM, C::Dqk>;
    using dq_chunk_tile = typename fa4_bshd_direct_dq_globals<C>::dq_chunk_tile;
    using stats_smem_tile = col_vec<st_fl<kRefTileM, C::Dvo, true, 64>>;
    using stats_vec = typename rt_fl<kRefTileM, kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) dq_chunk_tile dq_smem[3][C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = cluster_ctarank();
    const int kv_block_idx = static_cast<int>(clusterIdx().x) * C::ClusterSize + cluster_rank;
    const int num_k_blocks = g.seq_len / (kRefTileN * C::WarpTiles);
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int num_q_blocks = g.seq_len / (kRefTileM * q_tiles_buffered);
    const bool dense_unmasked = !CAUSAL && g.actual_seq_len == g.seq_len;

    rt_bf<kRefTileM, C::Dqk> k_reg;
    rt_bf<kRefTileM, C::Dvo> v_reg;
    rt_fl<kRefTileM, C::Dqk> dk_accum;
    rt_fl<kRefTileM, C::Dvo> dv_accum;

    warp::load<dim::DEPTH>(k_reg, g.k, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::load<dim::DEPTH>(v_reg, g.v, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::zero(dk_accum);
    warp::zero(dv_accum);

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<qk_bf_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<v_bf_tile, dim::DEPTH>();
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
    }
    __syncthreads();

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        const int phase = q_block_idx & 1;

        if (warp == C::WarpTiles - 2) {
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<qk_bf_tile> q_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[subtile], g.q, q_tile_idx, q_b[0]);
            }
        }
        if (warp == C::WarpTiles - 1) {
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]) * q_tiles_buffered);
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                coord<v_bf_tile> do_tile_idx = {batch_idx, q_tile_base + subtile, head_idx, 0};
                warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[subtile], g.dout, do_tile_idx, o_b[0]);
            }
        }
        if (warp < q_tiles_buffered) {
            stats_vec lse_stage_vec, dpsum_stage_vec;
            warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(lse_log2_smem[warp], lse_stage_vec);
            warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + warp});
            warp::store(dpsum_smem[warp], dpsum_stage_vec);
        }
        wait(q_b[0], phase);
        wait(o_b[0], phase);
        __syncthreads();

        #pragma unroll 1
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<kRefTileM, C::Dqk> q_reg;
            rt_bf<kRefTileM, C::Dvo> do_reg;
            rt_fl<kRefTileM, kRefTileN> p, dp, ds;
            rt_fl<kRefTileM, C::Dqk> dq_partial;
            rt_fl<kRefTileM, 64> dq0, dq1, dq2;
            stats_vec lse_log2_vec, dpsum_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            bwd_fa4::detail::backward_tile_step<C, CAUSAL>(
                p,
                dp,
                ds,
                dq_partial,
                dk_accum,
                dv_accum,
                q_reg,
                k_reg,
                v_reg,
                do_reg,
                lse_log2_vec,
                dpsum_vec,
                g.scale,
                g.scale_log2e,
                q_tile_idx,
                kv_subtile_idx,
                g.actual_seq_len,
                dense_unmasked
            );

            if (q_block_idx != 0 || subtile != 0) {
                warp::tma::store_async_read_wait();
            }
            ::tkfa4::bwd_cute16_kernel_candidate::detail::extract_chunk<0>(dq_partial, dq0);
            ::tkfa4::bwd_cute16_kernel_candidate::detail::extract_chunk<1>(dq_partial, dq1);
            ::tkfa4::bwd_cute16_kernel_candidate::detail::extract_chunk<2>(dq_partial, dq2);
            warp::store(dq_smem[0][warp], dq0);
            warp::store(dq_smem[1][warp], dq1);
            warp::store(dq_smem[2][warp], dq2);
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq0, dq_smem[0][warp], {batch_idx, q_tile_idx, head_idx, 0});
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq1, dq_smem[1][warp], {batch_idx, q_tile_idx, head_idx, 1});
            warp::tma::store_add_async<dim::DEPTH, cache_policy::NORMAL>(g.dq2, dq_smem[2][warp], {batch_idx, q_tile_idx, head_idx, 2});
        }
        warp::tma::store_commit_group();
        __syncthreads();
    }

    warp::tma::store_async_read_wait();
    warp::store<dim::DEPTH>(g.dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(g.dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

template <typename C, bool CAUSAL>
__global__ __launch_bounds__(C::DkdvBlockThreads, C::MinBlocksPerSm)
void hot_bshd_dkdv_main_kernel(const __grid_constant__ hot_bshd_dkdv_globals<C> g) {
    constexpr bool kUseExactRegFrontier = false;
    constexpr bool kUseTmemDense = true;
    constexpr bool kUseTmemDenseDk = true;
    constexpr bool kUseTmemDenseDv = true;
    constexpr bool kUseDirectDenseReconstruct = false;
    constexpr bool kUseCuteDenseHelper = false;
    using q_tile = typename hot_bshd_dkdv_globals<C>::q_tile;
    using k_tile = typename hot_bshd_dkdv_globals<C>::k_tile;
    using v_tile = typename hot_bshd_dkdv_globals<C>::v_tile;
    using do_tile = typename hot_bshd_dkdv_globals<C>::do_tile;
    using stats_smem_tile = typename hot_bshd_dkdv_globals<C>::stats_tile;
    using ds_warp_tile = st_bf<kRefTileM, C::TileRows>;
    using attn_tt = half_tt_fl<C::TileRows>;
    using dk_tt = half_tt_fl<64>;
    using dv_tt = half_tt_fl<C::Dvo>;

    struct main_shared_storage {
        k_tile k_smem[C::ConsumerWarpgroups];
        v_tile v_smem[C::ConsumerWarpgroups];
        q_tile q_smem[1];
        do_tile do_smem[1];
        ds_warp_tile ds_warp_smem[C::ConsumerWarpgroups][WARPGROUP_WARPS];
        stats_smem_tile lse_log2_smem[C::QSubtiles];
        stats_smem_tile dpsum_smem[C::QSubtiles];
    };
    struct epilogue_shared_storage {
        typename hot_bshd_dkdv_globals<C>::dk_full_tile dk0_smem[C::ConsumerWarpgroups];
        typename hot_bshd_dkdv_globals<C>::dk_full_tile dk1_smem[C::ConsumerWarpgroups];
        typename hot_bshd_dkdv_globals<C>::dk_full_tile dk2_smem[C::ConsumerWarpgroups];
        typename hot_bshd_dkdv_globals<C>::dv_full_tile dv_smem[C::ConsumerWarpgroups];
    };
    union shared_storage {
        main_shared_storage main;
        epilogue_shared_storage epilogue;
    };

    __shared__ alignas(1024) shared_storage smem;
    auto &k_smem = smem.main.k_smem;
    auto &v_smem = smem.main.v_smem;
    auto &q_smem = smem.main.q_smem;
    auto &do_smem = smem.main.do_smem;
    auto &ds_warp_smem = smem.main.ds_warp_smem;
    auto &lse_log2_smem = smem.main.lse_log2_smem;
    auto &dpsum_smem = smem.main.dpsum_smem;

    __shared__ __align__(16) kittens::semaphore kv_b;
    __shared__ __align__(16) kittens::semaphore q_b[1];
    __shared__ __align__(16) kittens::semaphore o_b[1];
    __shared__ __align__(16) kittens::semaphore score_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore dp_ready[C::ConsumerWarpgroups][1];
    __shared__ __align__(16) kittens::semaphore kv_tmem_ready[C::ConsumerWarpgroups];

    const int warp = kittens::warpid();
    const bool is_compute = warp < C::ComputeWarps;
    const bool is_load = warp == C::ComputeWarps;
    const int consumer_idx = is_compute ? (warp / kittens::WARPGROUP_WARPS) : -1;

    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int cluster_rank = cluster_ctarank();
    const int cluster_idx = static_cast<int>(blockIdx.x) / C::ClusterSize;
    const int num_k_blocks = g.seq_len / (C::TileRows * C::ConsumerWarpgroups);
    const int kv_block_idx = cluster_idx * C::ClusterSize + cluster_rank;
    if (kv_block_idx >= num_k_blocks) {
        return;
    }

    const int kv_tile_base = kv_block_idx * C::ConsumerWarpgroups;
    const int q_blocks = g.seq_len / C::TileRows;
    const int q_start_block = CAUSAL ? kv_tile_base : 0;
    const int first_dense_q_block =
        (CAUSAL && kUseExactRegFrontier) ? (q_start_block + C::ConsumerWarpgroups) : q_start_block;
    rt_fl<kRefTileM, 64> dk0_reg_accum, dk1_reg_accum, dk2_reg_accum;
    rt_fl<kRefTileM, C::Dvo> dv_reg_accum;
    tensor_allocator<1, 1> tm_alloc{};
    attn_tt score_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    attn_tt dp_tt[C::ConsumerWarpgroups] = {attn_tt{0}, attn_tt{0}};
    dk_tt dk0_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk1_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dk_tt dk2_tt[C::ConsumerWarpgroups] = {dk_tt{0}, dk_tt{0}};
    dv_tt dv_accum_tt[C::ConsumerWarpgroups] = {dv_tt{0}, dv_tt{0}};

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<q_tile, dim::DEPTH>();
        g.k.template prefetch_tma<k_tile, dim::DEPTH>();
        g.v.template prefetch_tma<v_tile, dim::DEPTH>();
        g.dout.template prefetch_tma<do_tile, dim::DEPTH>();

        init_semaphore(kv_b, 0, 1);
        init_semaphore(q_b[0], 0, 1);
        init_semaphore(o_b[0], 0, 1);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            init_semaphore(score_ready[w][0], 0, 1);
            init_semaphore(dp_ready[w][0], 0, 1);
            init_semaphore(kv_tmem_ready[w], 0, 1);
        }
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * C::ConsumerWarpgroups);
        for (int w = 0; w < C::ConsumerWarpgroups; ++w) {
            coord<k_tile> tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            coord<v_tile> v_tile_idx = {batch_idx, kv_tile_base + w, head_idx, 0};
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[w], g.v, v_tile_idx, kv_b);
        }
    }
    __syncthreads();

    if (is_compute) {
        warp::zero(dk0_reg_accum);
        warp::zero(dk1_reg_accum);
        warp::zero(dk2_reg_accum);
        warp::zero(dv_reg_accum);
        score_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, 0);
        dp_tt[consumer_idx] = tm_alloc.template allocate<attn_tt>(consumer_idx, C::TileRows);
        dk0_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 2 * C::TileRows);
        dk1_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 3 * C::TileRows);
        dk2_tt[consumer_idx] = tm_alloc.template allocate<dk_tt>(consumer_idx, 4 * C::TileRows);
        dv_accum_tt[consumer_idx] = tm_alloc.template allocate<dv_tt>(consumer_idx, 5 * C::TileRows);
        rt_fl<kRefTileM, 64> zero_dk;
        rt_fl<kRefTileM, C::Dvo> zero_dv;
        warp::zero(zero_dk);
        warp::zero(zero_dv);
        warpgroup::store_async(dk0_tt[consumer_idx], zero_dk);
        warpgroup::store_async(dk1_tt[consumer_idx], zero_dk);
        warpgroup::store_async(dk2_tt[consumer_idx], zero_dk);
        warpgroup::store_async(dv_accum_tt[consumer_idx], zero_dv);
        tensor_store_wait();
    }

    if (is_compute) {
        if constexpr (kUseExactRegFrontier && CAUSAL) {
            wait(kv_b, 0);
            rt_bf<kRefTileM, C::Dqk> q_reg, k_reg;
            rt_bf<kRefTileM, C::Dvo> v_reg, do_reg;
            using stats_vec = typename rt_fl<kRefTileM, C::TileRows>::col_vec;
            stats_vec lse_log2_vec, dpsum_vec;
            const int consumer_warp = warpgroup::warpid();
            warp::load(k_reg, k_smem[consumer_idx].template subtile<kRefTileM, C::Dqk>({consumer_warp, 0}));
            warp::load(v_reg, v_smem[consumer_idx].template subtile<kRefTileM, C::Dvo>({consumer_warp, 0}));

            for (int tile_offset = consumer_idx; tile_offset < C::ConsumerWarpgroups; ++tile_offset) {
                const int q_block_tile_idx = q_start_block + tile_offset;
                if (q_block_tile_idx >= q_blocks) {
                    break;
                }

                int q_subtile_begin = 0;
                if (tile_offset == consumer_idx) {
                    const int q_subtile = consumer_warp;
                    const int q_tile_idx = q_block_tile_idx * C::QSubtiles + q_subtile;
                    warp::load<dim::DEPTH>(q_reg, g.q, {batch_idx, q_tile_idx, head_idx, 0});
                    warp::load<dim::DEPTH>(do_reg, g.dout, {batch_idx, q_tile_idx, head_idx, 0});
                    warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
                    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
                    bwd_cute16_kernel_candidate::detail::repair_dkdv_step_chunked<C, true, false>(
                        dk0_reg_accum,
                        dk1_reg_accum,
                        dk2_reg_accum,
                        dv_reg_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        dpsum_vec,
                        g.scale,
                        g.scale_log2e,
                        q_tile_idx,
                        (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + consumer_warp,
                        g.seq_len
                    );
                    q_subtile_begin = consumer_warp + 1;
                }

                for (int q_subtile = q_subtile_begin; q_subtile < C::QSubtiles; ++q_subtile) {
                    const int q_tile_idx = q_block_tile_idx * C::QSubtiles + q_subtile;
                    warp::load<dim::DEPTH>(q_reg, g.q, {batch_idx, q_tile_idx, head_idx, 0});
                    warp::load<dim::DEPTH>(do_reg, g.dout, {batch_idx, q_tile_idx, head_idx, 0});
                    warp::load(lse_log2_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_idx});
                    warp::load(dpsum_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_idx});
                    bwd_cute16_kernel_candidate::detail::repair_dkdv_step_chunked<C, true, true>(
                        dk0_reg_accum,
                        dk1_reg_accum,
                        dk2_reg_accum,
                        dv_reg_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        dpsum_vec,
                        g.scale,
                        g.scale_log2e,
                        q_tile_idx,
                        (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + consumer_warp,
                        g.seq_len
                    );
                }
            }
        }
    }
    __syncthreads();

    for (int q_block_idx = first_dense_q_block; q_block_idx < q_blocks; ++q_block_idx) {
        const int phase = (q_block_idx - first_dense_q_block) & 1;
        if (is_load) {
            coord<q_tile> q_tile_idx = {batch_idx, q_block_idx, head_idx, 0};
            warp::tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(q_smem[0], g.q, q_tile_idx, q_b[0]);
            warp::tma::expect_bytes(o_b[0], sizeof(do_smem[0]));
            warp::tma::load_async<dim::DEPTH, cache_policy::NORMAL>(do_smem[0], g.dout, q_tile_idx, o_b[0]);
            const int q_tile_base = q_block_idx * C::QSubtiles;
            for (int subtile = 0; subtile < C::QSubtiles; ++subtile) {
                typename rt_fl<kRefTileM, C::TileRows>::col_vec lse_stage_vec, dpsum_stage_vec;
                warp::load(lse_stage_vec, g.lse_log2, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(lse_log2_smem[subtile], lse_stage_vec);
                warp::load(dpsum_stage_vec, g.dpsum, {batch_idx, head_idx, 0, q_tile_base + subtile});
                warp::store(dpsum_smem[subtile], dpsum_stage_vec);
            }
        }
        if (is_compute) {
            if constexpr (!kUseTmemDense || kUseDirectDenseReconstruct) {
                wait(q_b[0], phase);
                wait(o_b[0], phase);
            }
        }
        __syncthreads();

        if (is_compute) {
            if constexpr (kUseTmemDense) {
                rt_fl<kRefTileM, C::TileRows> p_block_t, dp_block_t, ds_block_t;
                rt_bf<kRefTileM, C::TileRows> p_block_t_mma, ds_block_t_mma;
                wait(kv_b, 0);
                if constexpr (kUseCuteDenseHelper) {
                    bwd_cute::detail::cute_compute_dkdv_loop<CAUSAL, C>(
                        q_b[0],
                        o_b[0],
                        score_ready[consumer_idx][0],
                        dp_ready[consumer_idx][0],
                        p_block_t,
                        dp_block_t,
                        ds_block_t,
                        p_block_t_mma,
                        ds_block_t_mma,
                        score_tt[consumer_idx],
                        dp_tt[consumer_idx],
                        dk0_tt[consumer_idx],
                        dk1_tt[consumer_idx],
                        dk2_tt[consumer_idx],
                        dv_accum_tt[consumer_idx],
                        q_smem,
                        k_smem,
                        v_smem,
                        do_smem,
                        ds_warp_smem,
                        lse_log2_smem,
                        dpsum_smem,
                        consumer_idx,
                        g.scale,
                        g.scale_log2e,
                        phase,
                        q_block_idx > (CAUSAL ? (kv_tile_base + consumer_idx) : 0),
                        q_block_idx,
                        kv_tile_base + consumer_idx,
                        g.seq_len
                    );
                } else if constexpr (kUseDirectDenseReconstruct) {
                    bwd_hot::detail::hot_compute_dkdv_tmem_exact_loop_direct_qdo<
                        CAUSAL,
                        kUseTmemDenseDk,
                        kUseTmemDenseDv,
                        C
                    >(
                        p_block_t,
                        dp_block_t,
                        ds_block_t,
                        p_block_t_mma,
                        ds_block_t_mma,
                        score_tt[consumer_idx],
                        dp_tt[consumer_idx],
                        dk0_tt[consumer_idx],
                        dk1_tt[consumer_idx],
                        dk2_tt[consumer_idx],
                        dv_accum_tt[consumer_idx],
                        dk0_reg_accum,
                        dk1_reg_accum,
                        dk2_reg_accum,
                        dv_reg_accum,
                        q_smem,
                        k_smem,
                        v_smem,
                        do_smem,
                        g.q,
                        g.dout,
                        g.lse_log2,
                        g.dpsum,
                        batch_idx,
                        head_idx,
                        g.scale,
                        g.scale_log2e,
                        q_block_idx > (CAUSAL ? (kv_tile_base + consumer_idx) : 0),
                        q_block_idx,
                        kv_tile_base + consumer_idx,
                        g.seq_len
                    );
                } else {
                    bwd_hot::detail::hot_compute_dkdv_tmem_exact_loop<
                        CAUSAL,
                        kUseTmemDenseDk,
                        kUseTmemDenseDv,
                        C
                    >(
                        q_b[0],
                        o_b[0],
                        score_ready[consumer_idx][0],
                        dp_ready[consumer_idx][0],
                        p_block_t,
                        dp_block_t,
                        ds_block_t,
                        p_block_t_mma,
                        ds_block_t_mma,
                        score_tt[consumer_idx],
                        dp_tt[consumer_idx],
                        dk0_tt[consumer_idx],
                        dk1_tt[consumer_idx],
                        dk2_tt[consumer_idx],
                        dv_accum_tt[consumer_idx],
                        dk0_reg_accum,
                        dk1_reg_accum,
                        dk2_reg_accum,
                        dv_reg_accum,
                        q_smem,
                        k_smem,
                        v_smem,
                        do_smem,
                        lse_log2_smem,
                        dpsum_smem,
                        g.scale,
                        g.scale_log2e,
                        phase,
                        q_block_idx > (CAUSAL ? (kv_tile_base + consumer_idx) : 0),
                        q_block_idx,
                        kv_tile_base + consumer_idx,
                        g.seq_len
                    );
                }
            }
        }
        __syncthreads();
    }

    if (is_compute) {
        const int kv_subtile_idx = (kv_tile_base + consumer_idx) * kittens::WARPGROUP_WARPS + warpgroup::warpid();
        if constexpr (kUseTmemDense) {
            rt_fl<kRefTileM, 64> dk0_reg, dk1_reg, dk2_reg;
            rt_fl<kRefTileM, C::Dvo> dv_reg;
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(kv_tmem_ready[consumer_idx]);
            }
            wait(kv_tmem_ready[consumer_idx], 0);
            warpgroup::load_async(dk0_reg, dk0_tt[consumer_idx]);
            warpgroup::load_async(dk1_reg, dk1_tt[consumer_idx]);
            warpgroup::load_async(dk2_reg, dk2_tt[consumer_idx]);
            warpgroup::load_async(dv_reg, dv_accum_tt[consumer_idx]);
            tensor_load_wait();
            warp::add(dk0_reg, dk0_reg, dk0_reg_accum);
            warp::add(dk1_reg, dk1_reg, dk1_reg_accum);
            warp::add(dk2_reg, dk2_reg, dk2_reg_accum);
            warp::add(dv_reg, dv_reg, dv_reg_accum);
            warpgroup::store(smem.epilogue.dk0_smem[consumer_idx], dk0_reg);
            warpgroup::store(smem.epilogue.dk1_smem[consumer_idx], dk1_reg);
            warpgroup::store(smem.epilogue.dk2_smem[consumer_idx], dk2_reg);
            warpgroup::store(smem.epilogue.dv_smem[consumer_idx], dv_reg);
            group<4>::sync(warpgroup::groupid() + 4);
            if (warpgroup::warpid() == 0) {
                coord<typename hot_bshd_dkdv_globals<C>::dk_full_tile> dk0_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 0};
                coord<typename hot_bshd_dkdv_globals<C>::dk_full_tile> dk1_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 1};
                coord<typename hot_bshd_dkdv_globals<C>::dk_full_tile> dk2_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 2};
                coord<typename hot_bshd_dkdv_globals<C>::dv_full_tile> dv_tile_idx = {batch_idx, kv_tile_base + consumer_idx, head_idx, 0};
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.dk0_full, smem.epilogue.dk0_smem[consumer_idx], dk0_tile_idx);
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.dk1_full, smem.epilogue.dk1_smem[consumer_idx], dk1_tile_idx);
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.dk2_full, smem.epilogue.dk2_smem[consumer_idx], dk2_tile_idx);
                warp::tma::store_async<dim::DEPTH, cache_policy::NORMAL>(g.dv_full, smem.epilogue.dv_smem[consumer_idx], dv_tile_idx);
                warp::tma::store_commit_group();
                warp::tma::store_async_wait();
            }
        } else {
            warp::store<dim::DEPTH>(g.dk, dk0_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
            warp::store<dim::DEPTH>(g.dk, dk1_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 1});
            warp::store<dim::DEPTH>(g.dk, dk2_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 2});
            warp::store<dim::DEPTH>(g.dv, dv_reg_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
        }
    }
}

template <typename C>
inline void launch_fa4_bshd_main_kernel(
    const bwd_fa4::main_globals<C> &g,
    bool causal,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(num_k_blocks, heads, batch_size),
        dim3(C::BlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if (causal) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, fa4_bshd_main_kernel<C, true>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, fa4_bshd_main_kernel<C, false>, g));
    }
}

template <typename C>
inline void launch_fa4_bshd_direct_dq_main_kernel(
    const fa4_bshd_direct_dq_globals<C> &g,
    bool causal,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(num_k_blocks, heads, batch_size),
        dim3(C::BlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if (causal) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, fa4_bshd_direct_dq_main_kernel<C, true>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, fa4_bshd_direct_dq_main_kernel<C, false>, g));
    }
}

template <typename C>
inline void launch_fa4_bshd_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &dq_accum,
    at::Tensor &dq_semaphore,
    bool causal,
    float scale,
    int actual_seq_len,
    bool deterministic
) {
    using G = bwd_fa4::main_globals<C>;
    using dqacc_gl = typename G::dqacc_gl;
    const int q_tile_groups = static_cast<int>(dq_accum.size(2));
    const int dq_tiles = static_cast<int>(q.size(1) / kRefTileM);
    const int dqacc_rows = q_tile_groups * C::ClusterSize * C::Mb;

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        ::kittens::make_gl<dqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(2)),
            dqacc_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        actual_seq_len,
        dq_semaphore.defined() ? reinterpret_cast<int *>(dq_semaphore.data_ptr<int>()) : nullptr,
        static_cast<int>(q.size(2)),
        dq_tiles,
        deterministic ? 1 : 0,
    };

    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_fa4_bshd_main_kernel<C>(
        g,
        causal,
        num_k_blocks,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_hot_bshd_dkdv_main_kernel(
    const hot_bshd_dkdv_globals<C> &g,
    bool causal,
    int num_k_blocks,
    int heads,
    int batch_size,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(num_k_blocks, heads, batch_size),
        dim3(C::DkdvBlockThreads, 1, 1),
        0,
        stream,
        dim3(C::ClusterSize, 1, 1)
    );
    if (causal) {
        CUDACHECK(cudaLaunchKernelEx(launch_config, hot_bshd_dkdv_main_kernel<C, true>, g));
    } else {
        CUDACHECK(cudaLaunchKernelEx(launch_config, hot_bshd_dkdv_main_kernel<C, false>, g));
    }
}

template <typename C>
inline void launch_fa4_bshd_backward_direct_dq(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    int actual_seq_len
) {
    using G = fa4_bshd_direct_dq_globals<C>;

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dq_out_gl>(dq),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
        actual_seq_len,
    };

    const int num_k_blocks = static_cast<int>(q.size(1) / (kRefTileN * C::WarpTiles));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    CUDACHECK(cudaMemsetAsync(
        dq.data_ptr<float>(),
        0,
        static_cast<size_t>(dq.numel()) * sizeof(float),
        stream
    ));
    launch_fa4_bshd_direct_dq_main_kernel<C>(
        g,
        causal,
        num_k_blocks,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
inline void launch_hot_bshd_dkdv_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale
) {
    using G = hot_bshd_dkdv_globals<C>;

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk_full_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk_full_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dk_full_gl>(dk),
        kittens::py::tensor_to_gl<typename G::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename G::dv_full_gl>(dv),
        kittens::py::tensor_to_gl<typename G::stats_gl>(lse_log2, q.size(0), q.size(2), 1, q.size(1)),
        kittens::py::tensor_to_gl<typename G::stats_gl>(dpsum, q.size(0), q.size(2), 1, q.size(1)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(1)),
    };

    const int num_k_blocks = static_cast<int>(q.size(1) / (C::TileRows * C::ConsumerWarpgroups));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_hot_bshd_dkdv_main_kernel<C>(
        g,
        causal,
        num_k_blocks,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(0)),
        stream
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename C>
__global__ __launch_bounds__(kWarpThreads, 8)
void fa4_bshd_postprocess_kernel(const __grid_constant__ bwd_fa4::postprocess_globals<C> g) {
    constexpr int kQSubtilesPerTile = C::Mb / kRefTileM;
    const int q_tile_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int q_tile_group_idx = q_tile_idx / kQSubtilesPerTile;
    const int q_subtile_in_group = q_tile_idx % kQSubtilesPerTile;
    const int scratch_tile_base =
        (q_tile_group_idx * C::ClusterSize) * kQSubtilesPerTile + q_subtile_in_group;

    if (g.deterministic && laneid() == 0) {
        for (int cluster_rank = 0; cluster_rank < C::ClusterSize; ++cluster_rank) {
            int *sem = g.dq_semaphore + bwd_fa4::detail::dq_semaphore_index(g, batch_idx, head_idx, q_tile_idx, cluster_rank);
            while (atomicAdd(sem, 0) < g.cluster_groups) {
            }
        }
    }
    __syncwarp();

    rt_fl<kRefTileM, C::Dqk> dq_reg, dq_partial;
    warp::load(dq_reg, g.dq_accum, {batch_idx, head_idx, scratch_tile_base, 0});
    if constexpr (C::ClusterSize == 2) {
        warp::load(dq_partial, g.dq_accum, {batch_idx, head_idx, scratch_tile_base + kQSubtilesPerTile, 0});
        warp::add(dq_reg, dq_reg, dq_partial);
    }
    warp::store<dim::DEPTH>(g.dq, dq_reg, {batch_idx, q_tile_idx, head_idx, 0});
}

template <typename C>
inline void launch_fa4_bshd_postprocess(
    at::Tensor &dq_accum,
    at::Tensor &dq,
    at::Tensor &dq_semaphore,
    bool deterministic
) {
    using G = bwd_fa4::postprocess_globals<C>;
    using dqacc_gl = typename G::dqacc_gl;
    const int q_tile_groups = static_cast<int>(dq_accum.size(2));
    const int q_tiles = static_cast<int>(dq.size(1) / kRefTileM);
    const int dqacc_rows = q_tile_groups * C::ClusterSize * C::Mb;
    const int cluster_groups = static_cast<int>(dq.size(1) / (kForwardTileN * C::ClusterSize));

    G g{
        ::kittens::make_gl<dqacc_gl>(
            reinterpret_cast<uint64_t>(dq_accum.data_ptr<float>()),
            static_cast<int>(dq.size(0)),
            static_cast<int>(dq.size(2)),
            dqacc_rows,
            C::Dqk
        ),
        kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
        dq_semaphore.defined() ? reinterpret_cast<int *>(dq_semaphore.data_ptr<int>()) : nullptr,
        static_cast<int>(dq.size(2)),
        q_tiles,
        cluster_groups,
        deterministic ? 1 : 0,
    };

    dim3 grid(q_tiles, dq.size(2), dq.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    fa4_bshd_postprocess_kernel<C><<<grid, kWarpThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace detail

template <typename C>
inline void launch_fa4_exact_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    using ExactConfig = bwd_fa4::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;

    at::Tensor q_bhsd = q.permute({0, 2, 1, 3}).contiguous();
    at::Tensor k_bhsd = k.permute({0, 2, 1, 3}).contiguous();
    at::Tensor v_bhsd = v.permute({0, 2, 1, 3}).contiguous();
    at::Tensor out_bhsd = out.permute({0, 2, 1, 3}).contiguous();
    at::Tensor dout_bhsd = dout.permute({0, 2, 1, 3}).contiguous();
    at::Tensor lse_bh1s = lse.permute({0, 2, 1}).contiguous().unsqueeze(2);

    at::Tensor dq_bhsd = at::empty_like(q_bhsd, lse.options());
    at::Tensor dk_bhsd = at::empty_like(k_bhsd, lse.options());
    at::Tensor dv_bhsd = at::empty_like(v_bhsd, lse.options());
    at::Tensor dpsum = at::empty({q_bhsd.size(0), q_bhsd.size(1), 1, q_bhsd.size(2)}, lse.options());
    at::Tensor lse_log2 = at::empty({q_bhsd.size(0), q_bhsd.size(1), 1, q_bhsd.size(2)}, lse.options());
    at::Tensor dq_accum = at::zeros(
        {q_bhsd.size(0), q_bhsd.size(1), q_bhsd.size(2) / kForwardTileM, C::ClusterSize, kForwardTileM, kB300QKDim},
        lse.options()
    );
    at::Tensor dq_semaphore = at::empty(
        {q_bhsd.size(0), q_bhsd.size(1), q_bhsd.size(2) / kRefTileM, C::ClusterSize},
        q.options().dtype(at::kInt)
    );

    bwd_fa4::launch_preprocess<bwd_fa4::preprocess_config<kB300VDim>>(
        out_bhsd,
        dout_bhsd,
        lse_bh1s,
        dpsum,
        lse_log2,
        dq_accum,
        dq_semaphore
    );
    bwd_fa4::launch_backward<ExactConfig>(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        dout_bhsd,
        lse_log2,
        dpsum,
        dk_bhsd,
        dv_bhsd,
        dq_accum,
        dq_semaphore,
        causal,
        scale,
        static_cast<int>(q.size(1)),
        deterministic
    );
    bwd_fa4::launch_postprocess<ExactConfig>(dq_accum, dq_bhsd, dq_semaphore, deterministic);

    dq.copy_(dq_bhsd.permute({0, 2, 1, 3}).contiguous());
    dk.copy_(dk_bhsd.permute({0, 2, 1, 3}).contiguous());
    dv.copy_(dv_bhsd.permute({0, 2, 1, 3}).contiguous());
}

template <typename C>
inline void launch_fa4_exact_backward_bshd(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    using ExactConfig = bwd_fa4::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;

    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor dq_accum = at::zeros(
        {q.size(0), q.size(2), q.size(1) / kForwardTileM, ExactConfig::ClusterSize, kForwardTileM, kB300QKDim},
        lse.options()
    );
    at::Tensor dq_semaphore = at::zeros(
        {q.size(0), q.size(2), q.size(1) / kRefTileM, ExactConfig::ClusterSize},
        q.options().dtype(at::kInt)
    );

    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
    detail::launch_fa4_bshd_backward<ExactConfig>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dk,
        dv,
        dq_accum,
        dq_semaphore,
        causal,
        scale,
        static_cast<int>(q.size(1)),
        deterministic
    );
    detail::launch_fa4_bshd_postprocess<ExactConfig>(dq_accum, dq, dq_semaphore, deterministic);
}

template <typename C>
inline void launch_fa4_exact_backward_bshd_direct_dq(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    if (deterministic) {
        launch_fa4_exact_backward_bshd<C>(q, k, v, out, lse, dout, dq, dk, dv, causal, scale, deterministic);
        return;
    }

    using ExactConfig = bwd_fa4::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());

    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
    detail::launch_fa4_bshd_backward_direct_dq<ExactConfig>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk,
        dv,
        causal,
        scale,
        static_cast<int>(q.size(1))
    );
}

template <typename C, typename DkdvOutT = float, typename DqOutT = float>
inline void launch_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    if (deterministic) {
        launch_fa4_exact_backward<C>(q, k, v, out, lse, dout, dq, dk, dv, causal, scale, deterministic);
        return;
    }

    if (causal) {
        using Cluster1Config = config<C::Mb, C::Nb, C::Dqk, C::Dvo, 1>;
        using DedicatedLoadDqConfig = bwd_cute16_kernel_candidate::dq_only_dedicated_load_config<C::Mb, C::Nb, C::Dqk, C::Dvo, 1>;
        using Cluster2DqConfig = bwd_cute16_kernel_candidate::dq_only_compact_cluster2_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;
        using HotClusteredDqConfig = bwd_cute16_kernel_candidate::dq_only_clustered_cluster1_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;
        using Seq2048ExactConfig = bwd_cute16_kernel_candidate::seq2048_exact_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;
        using HotBshdConfig = bwd_hot::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
        using FullStockCuTeConfig = bwd_hot::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
        using FullStockCuTe16Config = tkfa4::bwd_cute16::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
        using CuTe16NativeExactConfig = tkfa4::bwd_cute16::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
        using SharedDsMonolithicConfig = config<C::Mb, C::Nb, C::Dqk, C::Dvo, 1>;
        constexpr bool kUseCuTe16NativeExact2048 = false;
        constexpr bool kUseFullStockCuTe2048 = false;
        constexpr bool kUseFullStockCuTe16Exact2048 = false;
        constexpr bool kUseCute16HotExact2048 = false;
        constexpr bool kUseSharedDsMonolithic2048 = false;
        constexpr bool kUseDsScratchDqOnly2048 = false;
        constexpr bool kUseClusteredDqOnly = false;
        constexpr bool kUseReducedDqOnly2048 = false;
        constexpr bool kUseHotClusteredDqOnly2048 = true;
        constexpr bool kUseDedicatedLoadDqOnly2048 = false;
        constexpr bool kUseSeq2048ExactOwnership = false;
        constexpr bool kUseDkdvPoolStream2048 = false;
        constexpr bool kUseScratchBackedDqOnly = kUseReducedDqOnly2048 || kUseHotClusteredDqOnly2048;
        constexpr bool kUseDirectDqOnly = !kUseClusteredDqOnly && !kUseScratchBackedDqOnly;
        const bool can_use_exact_split_concurrent =
            q.size(1) >= 2048 &&
            q.size(1) <= 32768 &&
            q.size(1) == k.size(1) &&
            q.size(1) == v.size(1) &&
            q.size(1) == out.size(1) &&
            q.size(1) == dout.size(1) &&
            (q.size(1) % (kForwardTileM * 2) == 0);
        if constexpr (kUseCuTe16NativeExact2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                bwd_cute16_candidate_cute::launch_causal_backward_cute16_native_exact<CuTe16NativeExactConfig>(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    dout,
                    dq,
                    dk,
                    dv,
                    scale
                );
                return;
            }
        }
        if constexpr (kUseFullStockCuTe2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
                bwd_cute16_candidate_cute::launch_causal_backward<FullStockCuTeConfig>(
                    q,
                    k,
                    v,
                    dout,
                    lse_log2,
                    dpsum,
                    dq,
                    dk,
                    dv,
                    scale
                );
                return;
            }
        }
        if constexpr (kUseFullStockCuTe16Exact2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                tkfa4::bwd_cute16::launch_backward<FullStockCuTe16Config>(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    dout,
                    dq,
                    dk,
                    dv,
                    causal,
                    scale,
                    false
                );
                return;
            }
        }
        if constexpr (kUseCute16HotExact2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                auto current_stream = at::cuda::getCurrentCUDAStream();
                at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
                bwd_cute16_kernel_candidate::launch_backward_dkdv_only<Cluster1Config, DkdvOutT>(
                    q,
                    k,
                    v,
                    dout,
                    lse_log2,
                    dpsum,
                    dk,
                    dv,
                    causal,
                    scale
                );
                CUDACHECK(cudaMemsetAsync(
                    dq.data_ptr(),
                    0,
                    dq.nbytes(),
                    current_stream.stream()
                ));
                bwd_cute16_kernel_candidate::launch_backward_dq_only<Cluster1Config>(
                    q,
                    k,
                    v,
                    dout,
                    lse_log2,
                    dpsum,
                    dq,
                    scale,
                    current_stream.stream()
                );
                return;
            }
        }
        if constexpr (kUseSharedDsMonolithic2048) {
            if (can_use_exact_split_concurrent && !deterministic) {
                at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
                launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
                bwd_cute16_kernel_candidate::launch_backward_shared_ds_exact<SharedDsMonolithicConfig>(
                    q,
                    k,
                    v,
                    dout,
                    lse_log2,
                    dpsum,
                    dq,
                    dk,
                    dv,
                    scale,
                    at::cuda::getCurrentCUDAStream().stream()
                );
                return;
            }
        }
        if (can_use_exact_split_concurrent) {
            auto current_stream = at::cuda::getCurrentCUDAStream();
            const bool use_hot_clustered_dq_probe = detail::hot_clustered_dq_probe_enabled();
            const int split_dqacc_rows = use_hot_clustered_dq_probe
                ? static_cast<int>(q.size(1) * HotClusteredDqConfig::ClusterSize)
                : (kUseDirectDqOnly
                    ? 0
                    : static_cast<int>(q.size(1) * (kUseSeq2048ExactOwnership ? Seq2048ExactConfig::ClusterSize : (kUseHotClusteredDqOnly2048 ? HotClusteredDqConfig::ClusterSize : (kUseClusteredDqOnly ? Cluster2DqConfig::ClusterSize : Cluster1Config::ClusterSize)))));
            auto &split = detail::get_split_hybrid_resources(
                current_stream.device_index(),
                current_stream.stream(),
                q,
                lse,
                split_dqacc_rows,
                C::Dqk,
                kUseDsScratchDqOnly2048
            );
            CUDACHECK(cudaEventRecord(split.call_entry_ready, current_stream.stream()));
            CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.call_entry_ready));
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
                CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
            }
            launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, split.dpsum, split.lse_log2);
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
                CUDACHECK(cudaEventRecord(split.dq_start, split.dq_stream));
            }
            if (use_hot_clustered_dq_probe) {
                if constexpr (
                    bwd_cute16_kernel_candidate::kUseDirectFinalClusteredDq &&
                    kUseHotClusteredDqOnly2048
                ) {
                    CUDACHECK(cudaMemsetAsync(
                        dq.data_ptr(),
                        0,
                        dq.nbytes(),
                        split.dq_stream
                    ));
                } else {
                    CUDACHECK(cudaMemsetAsync(
                        split.dqacc.data_ptr<float>(),
                        0,
                        static_cast<size_t>(split.dqacc.numel()) * sizeof(float),
                        split.dq_stream
                    ));
                }
            } else if constexpr (
                bwd_cute16_kernel_candidate::kUseDirectFinalClusteredDq &&
                kUseHotClusteredDqOnly2048
            ) {
                CUDACHECK(cudaMemsetAsync(
                    dq.data_ptr(),
                    0,
                    dq.nbytes(),
                    split.dq_stream
                ));
            } else if constexpr (kUseDirectDqOnly) {
                CUDACHECK(cudaMemsetAsync(
                    dq.data_ptr(),
                    0,
                    dq.nbytes(),
                    split.dq_stream
                ));
            } else {
                CUDACHECK(cudaMemsetAsync(
                    split.dqacc.data_ptr<float>(),
                    0,
                    static_cast<size_t>(split.dqacc.numel()) * sizeof(float),
                    split.dq_stream
                ));
            }
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.dq_zero_end, split.dq_stream));
            }
            CUDACHECK(cudaEventRecord(split.preprocess_done, current_stream.stream()));
            CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.preprocess_done));
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.dq_ready, split.dq_stream));
            }
            if constexpr (kUseDkdvPoolStream2048) {
                CUDACHECK(cudaStreamWaitEvent(split.dkdv_stream, split.preprocess_done));
            } else if (use_hot_clustered_dq_probe) {
                CUDACHECK(cudaStreamWaitEvent(split.dkdv_stream, split.preprocess_done));
            }
            cudaStream_t dkdv_exec_stream = (kUseDkdvPoolStream2048 || use_hot_clustered_dq_probe)
                ? split.dkdv_stream
                : current_stream.stream();
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.dkdv_start, dkdv_exec_stream));
            }
            if constexpr (kUseSeq2048ExactOwnership) {
                bwd_cute16_kernel_candidate::launch_backward_seq2048_exact_dkdv_only<Seq2048ExactConfig>(
                    q,
                    k,
                    v,
                    dout,
                    split.lse_log2,
                    split.dpsum,
                    dk,
                    dv,
                    scale,
                    dkdv_exec_stream
                );
                bwd_cute16_kernel_candidate::launch_backward_dq_only<Cluster1Config>(
                    q,
                    k,
                    v,
                    dout,
                    split.lse_log2,
                    split.dpsum,
                    dq,
                    scale,
                    split.dq_stream
                );
            } else if constexpr (kUseDsScratchDqOnly2048) {
                bwd_cute16_kernel_candidate::launch_backward_dkdv_only_store_ds<Cluster1Config>(
                    q,
                    k,
                    v,
                    dout,
                    split.lse_log2,
                    split.dpsum,
                    dk,
                    dv,
                    split.ds_scratch,
                    scale,
                    dkdv_exec_stream
                );
                bwd_cute16_kernel_candidate::launch_backward_dq_from_ds<Cluster1Config>(
                    k,
                    split.ds_scratch,
                    dq,
                    split.dq_stream
                );
            } else {
                bwd_cute16_kernel_candidate::launch_backward_dkdv_only<Cluster1Config, DkdvOutT>(
                    q,
                    k,
                    v,
                    dout,
                    split.lse_log2,
                    split.dpsum,
                    dk,
                    dv,
                    scale,
                    dkdv_exec_stream
                );
                if (use_hot_clustered_dq_probe) {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<HotClusteredDqConfig, DqOutT>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream,
                        &split.dqacc
                    );
                } else if constexpr (kUseClusteredDqOnly) {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only<Cluster2DqConfig>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream
                    );
                } else if constexpr (kUseHotClusteredDqOnly2048) {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<HotClusteredDqConfig, DqOutT>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream,
                        &split.dqacc
                    );
                } else if constexpr (kUseDedicatedLoadDqOnly2048) {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only<DedicatedLoadDqConfig, kUseDirectDqOnly>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream,
                        kUseDirectDqOnly ? nullptr : &split.dqacc
                    );
                } else {
                    bwd_cute16_kernel_candidate::launch_backward_dq_only<Cluster1Config, kUseDirectDqOnly>(
                        q,
                        k,
                        v,
                        dout,
                        split.lse_log2,
                        split.dpsum,
                        dq,
                        scale,
                        split.dq_stream,
                        kUseDirectDqOnly ? nullptr : &split.dqacc
                    );
                }
            }
            if (detail::split_timing_enabled()) {
                CUDACHECK(cudaEventRecord(split.dkdv_end, dkdv_exec_stream));
                CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
            }
            if constexpr (kUseDkdvPoolStream2048) {
                CUDACHECK(cudaEventRecord(split.dkdv_done, dkdv_exec_stream));
                CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dkdv_done));
            } else if (use_hot_clustered_dq_probe) {
                CUDACHECK(cudaEventRecord(split.dkdv_done, dkdv_exec_stream));
                CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dkdv_done));
            }
            CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
            CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
            if (detail::split_timing_enabled()) {
                float total_ms = 0.0f;
                float preprocess_ms = 0.0f;
                float dkdv_ms = 0.0f;
                float dq_ms = 0.0f;
                float dq_zero_ms = 0.0f;
                float dq_wait_ms = 0.0f;
                float dq_kernel_ms = 0.0f;
                CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
                CUDACHECK(cudaEventSynchronize(split.total_end));
                CUDACHECK(cudaEventElapsedTime(&total_ms, split.total_start, split.total_end));
                CUDACHECK(cudaEventElapsedTime(&preprocess_ms, split.preprocess_start, split.preprocess_end));
                CUDACHECK(cudaEventElapsedTime(&dkdv_ms, split.dkdv_start, split.dkdv_end));
                CUDACHECK(cudaEventElapsedTime(&dq_ms, split.dq_start, split.dq_end));
                CUDACHECK(cudaEventElapsedTime(&dq_zero_ms, split.dq_start, split.dq_zero_end));
                CUDACHECK(cudaEventElapsedTime(&dq_wait_ms, split.dq_zero_end, split.dq_ready));
                CUDACHECK(cudaEventElapsedTime(&dq_kernel_ms, split.dq_ready, split.dq_end));
                std::fprintf(
                    stderr,
                    "split_timing_us preprocess=%.2f dkdv=%.2f dq=%.2f dq_zero=%.2f dq_wait=%.2f dq_kernel=%.2f total=%.2f\n",
                    preprocess_ms * 1000.0f,
                    dkdv_ms * 1000.0f,
                    dq_ms * 1000.0f,
                    dq_zero_ms * 1000.0f,
                    dq_wait_ms * 1000.0f,
                    dq_kernel_ms * 1000.0f,
                    total_ms * 1000.0f
                );
            }
            return;
        }

        at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
        at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
        launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
        at::Tensor dqacc = at::zeros({q.size(0), q.size(2), q.size(1) * Cluster1Config::ClusterSize, C::Dqk}, lse.options());
        bwd_cute16_kernel_candidate::launch_backward<Cluster1Config>(
            q,
            k,
            v,
            dout,
            q,
            k,
            v,
            dout,
            lse_log2,
            dpsum,
            dq,
            dk,
            dv,
            dqacc,
            nullptr,
            causal,
            scale,
            deterministic
        );
        return;
    }

    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor dqacc = at::zeros({q.size(0), q.size(2), q.size(1) * C::ClusterSize, C::Dqk}, lse.options());

    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2);
    bwd_cute16_kernel_candidate::launch_backward<C>(
        q,
        k,
        v,
        dout,
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk,
        dv,
        dqacc,
        nullptr,
        causal,
        scale,
        deterministic
    );
}

template <typename C, typename DqOutT = float>
inline void launch_backward_cute_parity_2cta_dkdv(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    TORCH_CHECK(causal, "CuTe parity 2CTA dK/dV candidate only supports causal=True");
    TORCH_CHECK(!deterministic, "CuTe parity 2CTA dK/dV candidate only supports deterministic=False");
    TORCH_CHECK(q.size(1) == 2048, "CuTe parity 2CTA dK/dV candidate is fixed to seqlen 2048");

    using Seq2048DkdvConfig = bwd_cute16_kernel_candidate::seq2048_exact_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;
    using HotClusteredDqConfig = bwd_cute16_kernel_candidate::dq_only_clustered_cluster1_config<C::Mb, C::Nb, C::Dqk, C::Dvo>;

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        static_cast<int>(q.size(1) * HotClusteredDqConfig::ClusterSize),
        C::Dqk,
        false
    );
    CUDACHECK(cudaEventRecord(split.call_entry_ready, current_stream.stream()));
    CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.call_entry_ready));

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
    }
    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, split.dpsum, split.lse_log2);
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.dq_start, split.dq_stream));
    }

    CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), split.dq_stream));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_zero_end, split.dq_stream));
    }

    CUDACHECK(cudaEventRecord(split.preprocess_done, current_stream.stream()));
    CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.preprocess_done));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_ready, split.dq_stream));
        CUDACHECK(cudaEventRecord(split.dkdv_start, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::launch_backward_seq2048_exact_dkdv_only<Seq2048DkdvConfig>(
        q,
        k,
        v,
        dout,
        split.lse_log2,
        split.dpsum,
        dk,
        dv,
        scale,
        current_stream.stream()
    );
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dkdv_end, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<HotClusteredDqConfig, DqOutT>(
        q,
        k,
        v,
        dout,
        split.lse_log2,
        split.dpsum,
        dq,
        scale,
        split.dq_stream,
        &split.dqacc
    );
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
    }

    CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
    CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
    if (detail::split_timing_enabled()) {
        float total_ms = 0.0f;
        float preprocess_ms = 0.0f;
        float dkdv_ms = 0.0f;
        float dq_ms = 0.0f;
        float dq_zero_ms = 0.0f;
        float dq_wait_ms = 0.0f;
        float dq_kernel_ms = 0.0f;
        CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
        CUDACHECK(cudaEventSynchronize(split.total_end));
        CUDACHECK(cudaEventElapsedTime(&total_ms, split.total_start, split.total_end));
        CUDACHECK(cudaEventElapsedTime(&preprocess_ms, split.preprocess_start, split.preprocess_end));
        CUDACHECK(cudaEventElapsedTime(&dkdv_ms, split.dkdv_start, split.dkdv_end));
        CUDACHECK(cudaEventElapsedTime(&dq_ms, split.dq_start, split.dq_end));
        CUDACHECK(cudaEventElapsedTime(&dq_zero_ms, split.dq_start, split.dq_zero_end));
        CUDACHECK(cudaEventElapsedTime(&dq_wait_ms, split.dq_zero_end, split.dq_ready));
        CUDACHECK(cudaEventElapsedTime(&dq_kernel_ms, split.dq_ready, split.dq_end));
        std::fprintf(
            stderr,
            "parity_2cta_split_timing_us preprocess=%.2f dkdv=%.2f dq=%.2f dq_zero=%.2f dq_wait=%.2f dq_kernel=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            dkdv_ms * 1000.0f,
            dq_ms * 1000.0f,
            dq_zero_ms * 1000.0f,
            dq_wait_ms * 1000.0f,
            dq_kernel_ms * 1000.0f,
            total_ms * 1000.0f
        );
    }
}

template <
    typename C,
    typename DqOutT = float,
    bool UseChunkedTmemDq = false,
    bool UsePipelinedTmemDq = false,
    bool DoubleBufferPipelinedInputs = false,
    bool EnqueuePipelinedDqEarly = true,
    int DenseSplitCount = 1,
    int DqReplaySplitCount = 1,
    bool UseTmemScoreDp = false,
    bool UseTmemFrontier = false,
    bool FuseDenseDq = false,
    bool AdaptiveLastQuarter = false,
    bool OverlapLoadAndDqReduce = false,
    bool SkipAdaptiveTailScratch = false,
    bool UseLdsmTransposeDs = false,
    bool DoubleBufferFusedDqTma = false,
    bool MaterializeDkdvBf16 = false,
    bool CacheDkdvFp32Base = false,
    bool ReleaseTmemOperandsEachIteration = false,
    bool SerializePipelinedDqBeforeDkdv = false,
    bool SerializeDenseFrontier = false
>
inline void launch_backward_dense_tmem_frontier_dkdv(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic,
    at::Tensor *materialized_dk = nullptr,
    at::Tensor *materialized_dv = nullptr
) {
    static_assert(!(UseChunkedTmemDq && UsePipelinedTmemDq), "Select only one opt-in TMEM dQ route");
    static_assert(
        !DoubleBufferPipelinedInputs || UsePipelinedTmemDq,
        "Double-buffered inputs require the pipelined TMEM dQ route"
    );
    static_assert(
        DenseSplitCount == 1 || DenseSplitCount == 2 || DenseSplitCount == 3 ||
            DenseSplitCount == 4 || DenseSplitCount == 8,
        "Dense split count must be 1, 2, 3, 4, or 8"
    );
    static_assert(
        DqReplaySplitCount == 1 || DqReplaySplitCount == 2,
        "dQ replay split count must be 1 or 2"
    );
    static_assert(
        DqReplaySplitCount == 1 || UsePipelinedTmemDq,
        "dQ replay splitting requires the pipelined TMEM dQ route"
    );
    static_assert(!FuseDenseDq || UseTmemScoreDp, "fused dQ requires TMEM score/dP");
    static_assert(!FuseDenseDq || UseTmemFrontier, "fused dQ requires the TMEM frontier owner");
    static_assert(!FuseDenseDq || DqReplaySplitCount == 1, "fused dQ does not use replay splits");
    static_assert(
        !AdaptiveLastQuarter || (FuseDenseDq && DenseSplitCount == 2),
        "Adaptive last-quarter ownership requires fused split-2"
    );
    static_assert(
        !OverlapLoadAndDqReduce || FuseDenseDq,
        "Load/reducer overlap requires fused dQ"
    );
    static_assert(
        !SkipAdaptiveTailScratch || AdaptiveLastQuarter,
        "Skipping tail scratch requires adaptive last-quarter ownership"
    );
    static_assert(
        !MaterializeDkdvBf16 || (DenseSplitCount == 2 && !SkipAdaptiveTailScratch),
        "BF16 materialization currently supports the complete split-2 merge"
    );
    static_assert(
        !CacheDkdvFp32Base || MaterializeDkdvBf16,
        "cached FP32 dK/dV bases require BF16 materialization"
    );
    TORCH_CHECK(causal, "Dense TMEM frontier dK/dV candidate only supports causal=True");
    TORCH_CHECK(!deterministic, "Dense TMEM frontier dK/dV candidate only supports deterministic=False");
    if constexpr (DenseSplitCount > 1) {
        TORCH_CHECK(
            q.size(1) >= 512 && q.size(1) % 512 == 0,
            "Split dense TMEM dK/dV candidate requires seqlen >= 512 and divisible by 512"
        );
    } else {
        TORCH_CHECK(q.size(1) == 2048, "Dense TMEM frontier dK/dV candidate is fixed to seqlen 2048");
    }

    using DenseDkdvConfig = bwd_cute16_kernel_candidate::dense_tmem_frontier_config<
        C::Mb,
        C::Nb,
        C::Dqk,
        C::Dvo
    >;
    using HotClusteredDqConfig = bwd_cute16_kernel_candidate::dq_only_clustered_cluster1_config<
        C::Mb,
        C::Nb,
        C::Dqk,
        C::Dvo
    >;
    using PipelinedDqConfig = bwd_cute16_kernel_candidate::dq_only_clustered_pipelined_config<
        C::Mb,
        C::Nb,
        C::Dqk,
        C::Dvo
    >;

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        static_cast<int>(q.size(1) * HotClusteredDqConfig::ClusterSize),
        C::Dqk,
        false,
        C::Dvo,
        DenseSplitCount,
        CacheDkdvFp32Base
    );
    CUDACHECK(cudaEventRecord(split.call_entry_ready, current_stream.stream()));
    CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.call_entry_ready));

    at::Tensor &dk_base = CacheDkdvFp32Base ? split.cached_base_dk : dk;
    at::Tensor &dv_base = CacheDkdvFp32Base ? split.cached_base_dv : dv;

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
    }
    launch_preprocess<preprocess_config<kB300VDim>>(out, dout, lse, split.dpsum, split.lse_log2);
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.dq_start, split.dq_stream));
    }

    CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), split.dq_stream));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_zero_end, split.dq_stream));
    }

    CUDACHECK(cudaEventRecord(split.preprocess_done, current_stream.stream()));
    CUDACHECK(cudaStreamWaitEvent(split.dq_stream, split.preprocess_done));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dq_ready, split.dq_stream));
    }
    if constexpr (FuseDenseDq) {
        CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
        CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
    }

    if constexpr (!FuseDenseDq && UsePipelinedTmemDq && EnqueuePipelinedDqEarly) {
        bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered_pipelined<
            PipelinedDqConfig,
            DqOutT,
            DoubleBufferPipelinedInputs,
            DqReplaySplitCount
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            scale,
            split.dq_stream,
            &split.dqacc
        );
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
        }
    } else if constexpr (!FuseDenseDq && UseChunkedTmemDq) {
        bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<
            HotClusteredDqConfig,
            DqOutT,
            true
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            scale,
            split.dq_stream,
            &split.dqacc
        );
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
        }
    }

    if constexpr (
        !FuseDenseDq &&
        UsePipelinedTmemDq &&
        EnqueuePipelinedDqEarly &&
        SerializePipelinedDqBeforeDkdv
    ) {
        const bool serialize_clustered_dq =
            ((q.size(1) == 2048 || q.size(1) == 4096) && q.size(2) == 8) ||
            (q.size(1) >= 8192 && q.size(2) >= 2);
        if (serialize_clustered_dq) {
            CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
            CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
        }
    }

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dkdv_start, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::launch_backward_dense_tmem_frontier_dkdv<
        DenseDkdvConfig,
        DenseSplitCount,
        UseTmemScoreDp,
        UseTmemFrontier,
        FuseDenseDq,
        AdaptiveLastQuarter,
        OverlapLoadAndDqReduce,
        SkipAdaptiveTailScratch,
        UseLdsmTransposeDs,
        DoubleBufferFusedDqTma,
        MaterializeDkdvBf16,
        ReleaseTmemOperandsEachIteration,
        SerializeDenseFrontier
    >(
        q,
        k,
        v,
        dout,
        split.lse_log2,
        split.dpsum,
        dk_base,
        dv_base,
        scale,
        current_stream.stream(),
        split.dkdv_stream,
        split.dense_main_done,
        split.frontier_dv_done,
        &split.frontier_dk,
        &split.frontier_dv,
        DenseSplitCount > 1 ? &split.dense_split_dk : nullptr,
        DenseSplitCount > 1 ? &split.dense_split_dv : nullptr,
        FuseDenseDq ? &dq : nullptr,
        materialized_dk,
        materialized_dv
    );
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dkdv_end, current_stream.stream()));
        if constexpr (FuseDenseDq) {
            CUDACHECK(cudaEventRecord(split.dq_end, current_stream.stream()));
        }
    }

    if constexpr (!FuseDenseDq && UsePipelinedTmemDq && !EnqueuePipelinedDqEarly) {
        bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered_pipelined<
            PipelinedDqConfig,
            DqOutT,
            DoubleBufferPipelinedInputs,
            DqReplaySplitCount
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            scale,
            split.dq_stream,
            &split.dqacc
        );
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
        }
    } else if constexpr (!FuseDenseDq && !UseChunkedTmemDq && !UsePipelinedTmemDq) {
        bwd_cute16_kernel_candidate::launch_backward_dq_only_clustered<
            HotClusteredDqConfig,
            DqOutT,
            false
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            scale,
            split.dq_stream,
            &split.dqacc
        );
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_end, split.dq_stream));
        }
    }

    if constexpr (!FuseDenseDq) {
        CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
        CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
    }
    if (detail::split_timing_enabled()) {
        float total_ms = 0.0f;
        float preprocess_ms = 0.0f;
        float dkdv_ms = 0.0f;
        float dq_ms = 0.0f;
        float dq_zero_ms = 0.0f;
        float dq_wait_ms = 0.0f;
        float dq_kernel_ms = 0.0f;
        CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
        CUDACHECK(cudaEventSynchronize(split.total_end));
        CUDACHECK(cudaEventElapsedTime(&total_ms, split.total_start, split.total_end));
        CUDACHECK(cudaEventElapsedTime(&preprocess_ms, split.preprocess_start, split.preprocess_end));
        CUDACHECK(cudaEventElapsedTime(&dkdv_ms, split.dkdv_start, split.dkdv_end));
        CUDACHECK(cudaEventElapsedTime(&dq_ms, split.dq_start, split.dq_end));
        CUDACHECK(cudaEventElapsedTime(&dq_zero_ms, split.dq_start, split.dq_zero_end));
        CUDACHECK(cudaEventElapsedTime(&dq_wait_ms, split.dq_zero_end, split.dq_ready));
        CUDACHECK(cudaEventElapsedTime(&dq_kernel_ms, split.dq_ready, split.dq_end));
        std::fprintf(
            stderr,
            "dense_tmem_frontier_split_timing_us preprocess=%.2f dkdv=%.2f dq=%.2f dq_zero=%.2f dq_wait=%.2f dq_kernel=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            dkdv_ms * 1000.0f,
            dq_ms * 1000.0f,
            dq_zero_ms * 1000.0f,
            dq_wait_ms * 1000.0f,
            dq_kernel_ms * 1000.0f,
            total_ms * 1000.0f
        );
    }
}

template <
    typename C,
    bool UseTmemDs = true,
    bool OverlapDsExchange = true,
    bool OverlapQWithDp = true,
    bool UseTmemP = false,
    bool OverlapDoWithDp = false,
    bool UseDpOperandReadyMbar = false,
    bool UseDqOperandReadyMbar = false,
    bool OverlapDvWithDs = false,
    bool PipelineNextScore = false,
    bool PreloadDqA = false,
    bool UseScoreOperandReadyMbar = false,
    bool UseDsWarpMulticastMbar = false,
    bool UseRoleSplit = false,
    bool RetainDsExchange = false,
    bool RetainDsLocal = false,
    bool UseNormalDoDv = false,
    bool UseTmaScoreK = false,
    bool DirectNextQdoDuringDqDrain = false,
    bool SingleOwnerCluster = false,
    bool UseFastExp2 = false,
    bool UseWarpStatsCache = false,
    bool PipelineLsePrefetch = false,
    bool UseDirectStatsLoads = false,
    bool SplitDvDkReady = false,
    bool StageDqAfterDv = false,
    bool StageDqPeerBeforeDv = false,
    bool UseWideDkN192 = false,
    bool DirectDsHalfStore = false,
    bool AsymmetricDvPublish = false
>
inline void launch_backward_cta2_fused_dense(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    TORCH_CHECK(causal, "2-CTA fused dense route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA fused dense route supports deterministic=False only"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA fused dense route supports batch size 1 only");
    TORCH_CHECK(
        (q.size(1) == 4096 && q.size(2) == 8) ||
            (q.size(1) == 8192 &&
             (q.size(2) == 2 || q.size(2) == 4 || q.size(2) == 8 ||
              q.size(2) == 16)) ||
            (q.size(1) == 16384 &&
             (q.size(2) == 4 || q.size(2) == 8 || q.size(2) == 16)),
        "2-CTA fused dense route supports S4096 H8 and "
        "S8192 H2/H4/H8/H16 or S16384 H4/H8/H16"
    );

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        0,
        C::Dqk,
        false,
        C::Dvo,
        1
    );

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
    }
    launch_preprocess<preprocess_config<kB300VDim>>(
        out,
        dout,
        lse,
        split.dpsum,
        split.lse_log2
    );
    CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), current_stream.stream()));
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.preprocess_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.dkdv_start, current_stream.stream()));
    }

    bwd_cute16_kernel_candidate::launch_backward_cta2_fused_dense_with_frontier(
        q,
        k,
        v,
        dout,
        split.lse_log2,
        split.dpsum,
        dq,
        dk,
        dv,
        scale,
        current_stream.stream(),
        split.dkdv_stream,
        split.dense_main_done,
        split.frontier_dv_done,
        split.frontier_dk,
        split.frontier_dv,
        UseTmemDs,
        OverlapDsExchange,
        OverlapQWithDp,
        UseTmemP,
        OverlapDoWithDp,
        UseDpOperandReadyMbar,
        UseDqOperandReadyMbar,
        OverlapDvWithDs,
        PipelineNextScore,
        PreloadDqA,
        UseScoreOperandReadyMbar,
        UseDsWarpMulticastMbar,
        UseRoleSplit,
        RetainDsExchange,
        RetainDsLocal,
        UseNormalDoDv,
        UseTmaScoreK,
        DirectNextQdoDuringDqDrain,
        SingleOwnerCluster,
        UseFastExp2,
        UseWarpStatsCache,
        PipelineLsePrefetch,
        UseDirectStatsLoads,
        SplitDvDkReady,
        StageDqAfterDv,
        StageDqPeerBeforeDv,
        UseWideDkN192,
        DirectDsHalfStore,
        AsymmetricDvPublish
    );

    if (detail::split_timing_enabled()) {
        float preprocess_ms = 0.0f;
        float dense_frontier_ms = 0.0f;
        float total_ms = 0.0f;
        CUDACHECK(cudaEventRecord(split.dkdv_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
        CUDACHECK(cudaEventSynchronize(split.total_end));
        CUDACHECK(cudaEventElapsedTime(
            &preprocess_ms,
            split.preprocess_start,
            split.preprocess_end
        ));
        CUDACHECK(cudaEventElapsedTime(
            &dense_frontier_ms,
            split.dkdv_start,
            split.dkdv_end
        ));
        CUDACHECK(cudaEventElapsedTime(
            &total_ms,
            split.total_start,
            split.total_end
        ));
        std::fprintf(
            stderr,
            "cta2_fused_dense_timing_us preprocess=%.2f dense_frontier=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            dense_frontier_ms * 1000.0f,
            total_ms * 1000.0f
        );
    }
}

template <
    typename C,
    bool CoalescedBf16Store = false,
    bool DirectAsyncPeerDs = false,
    bool ProducerBulkPeerDs = false,
    bool ProducerBulkPeerDsCtaFenceOnly = false,
    bool DqReadHandoffBeforeCompletion = false,
    bool AggregateScoreConsumed = false,
    bool DirectTmaDkQ = false,
    bool TimeoutDqWait = false,
    bool UseWideDkN192 = false,
    bool TimeoutAllRoleWaits = false,
    bool UseNamedDoSourceBarrier = false,
    bool UseComputeScoreFanout = false,
    bool UseRuntimeAccumulationPredicate = false,
    bool UseReducerDqFanout = false,
    bool UseReducerDqLeaderArrive = false,
    bool MergeScoreDpReady = false,
    bool WideCoalescedBf16Store = false,
    bool BulkPeerDsFromFullTile = false,
    bool CoalescedPeerDsBulk = false,
    bool WideDqKGlobalToShared = false,
    bool IntegrateCausalFrontier = false,
    int ExactQTileCount = 0,
    bool FenceDsBeforeDkdvReady = false,
    bool UseNamedDkdvLocalFanIn = false,
    bool LeaderOnlyQdoPublishFence = false,
    bool CacheQdoReadyClusterAddress = false,
    bool GroupQdoTmaLoads = false,
    bool ElectedWideDkQTmaLoad = false,
    bool ElectedPeerDoTmaLoad = false,
    bool CacheRoleClusterAddresses = false,
    bool CacheTensorCommitAddresses = false,
    bool EnsureReducerOutputDrain = false,
    bool ElectedScoreKTmaLoad = false,
    bool UseExactClusterCoordinates = false,
    bool EnforceDpTmemConsumerRelease = false,
    bool SplitDpTmemConsumerRelease = false,
    bool UseIterationCausalMask = false,
    bool UseFusedTmemPAndDs = false,
    bool OverlapFusedDqAPublication = false,
    bool PrefetchNextQdoAfterDkdv = false,
    bool PrefetchNextOwnerQdo = false,
    bool UseFusedTmemRuntimeAccumulationPredicate = false,
    bool UseBitwisePExpansion = false,
    bool UseFusedExp2Pack = false,
    bool PeelCausalPrefix = false,
    bool BranchlessDoSourceLoad = false,
    bool BranchlessDoSourceBaseSelect = false,
    bool PublishVOncePerOwner = false,
    bool BulkDoDvStage = false,
    bool LoaderOwnedDkQ = false,
    bool FuseScoreScaleLse = false,
    bool RetainPackedP = false,
    bool SplitDirectDpsumAcrossDpDoneWait = false,
    bool FusedExp2Fragment4First = false,
    bool CarryDirectStatsOffset = false,
    bool CarryAllRolePhases = false,
    bool UseExactDefaultScaleLog2e = false,
    bool ReverseDkTailTmemLoadIssue = false,
    bool PrearmNextQdoBeforeDkDone = false,
    bool UseX32TmemComputeLayout = false,
    bool UseLongSeqStatsCache = false,
    bool UseCompactScoreMma = false,
    bool UseCompactDpMma = false,
    bool UsePackedBf16DsProduct = false,
    bool SplitDqTmemAndSharedHandoff = false,
    bool DistributedDqSharedReadWait = false,
    bool BalancedSingleOwnerSchedule = false,
    bool UseSingleOwnerWarpStatsCache = false,
    bool CacheDqStageLanePointers = false,
    bool UseSlicedFp32PForDs = false,
    bool UseTmaVWithScoreK = false,
    bool UseStatsWarpScoreFanout = false,
    bool UseBatchedDqTmemLoads = false,
    bool UseDynamicDpReleaseBarrierId = false,
    bool PreissueFirstDpHalfBeforeQdoWait = false,
    bool OverlapSecondDpLoadWithReleaseBarrier = false,
    bool RelayDoDvCompletionViaExchangeWarp = false,
    bool OverlapDqPeerCopyWithDoDvCompletion = false,
    bool OverlapLocalDqStoreWithPeerCopy = false,
    bool UseNonblockingDqPublicationFollowers = false,
    bool SplitDqAliasLifetimeWithCuteTmemMap = false,
    bool DeferFirstDsTmemStoreWait = false,
    bool OverlapFinalDsTmemStoreWithPeerSharedStores = false,
    bool DelayScoreAliasReleaseUntilFirstDqTailLoad = false,
    bool InterleaveSteadyScoreExpPairs = false,
    bool ShiftOverlappingScoreHalfBeforeDpRelease = false,
    bool BuildCompactDpDescriptorsAfterWait = false,
    int LateTensorCommitAddressSharedMask = 0,
    bool CacheCompactDpDescriptorsInShared = false,
    bool OverlapFirstDpsumQuarterWithSecondPStore = false,
    bool HoistReducerDpReadyBeforeScoreWait = false,
    bool PipelineFirstDpQuarterLoads = false,
    bool PublishNextQdoAtDqAliasRelease = false,
    bool JoinNextQdoWithDqAliasRelease = false,
    bool PrecomputePostScoreFanoutAddresses = false,
    bool PrecomputeScoreIterationDeltaUnderFanout = false,
    bool UseNativeX32Lowp = false,
    int LowpMode =
        bwd_cute16_kernel_candidate::detail::kCta2DenseLowpNone,
    bool ReuseDqDsForDk = false,
    bool UseAdaptiveQkScales = false,
    bool DirectDqProjection = false,
    bool UseRank128Score = false,
    bool TileReadyNvfp4DqProjection = false
>
inline void launch_backward_cta2_fused_dense_bf16_dkdv(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic,
    at::Tensor *q_lowp = nullptr,
    at::Tensor *k_lowp = nullptr,
    float dq_output_scale = 1.0f,
    float dk_output_scale = 1.0f,
    float ds_quant_scale =
        bwd_cute16_kernel_candidate::detail::
            kCta2DenseFp8DefaultDsQuantScale,
    at::Tensor *score_q_lowp = nullptr,
    at::Tensor *score_k_lowp = nullptr,
    at::Tensor *q_dk_mxfp4 = nullptr,
    at::Tensor *k_dq_mxfp4 = nullptr,
    at::Tensor *q_dk_nvfp4_scale = nullptr,
    at::Tensor *k_dq_nvfp4_scale = nullptr,
    at::Tensor *mixed_v_prepacked = nullptr,
    at::Tensor *adaptive_qk_scales = nullptr,
    at::Tensor *dq_bf16_output = nullptr,
    at::Tensor *projection_qkv_output = nullptr,
    at::Tensor *dq_projection_weight = nullptr,
    at::Tensor *dq_projection_output = nullptr,
    at::Tensor *dq_tile_arrivals = nullptr,
    producer_native_mxfp4_operands *producer_mxfp4 = nullptr,
    tile_ready_nvfp4_projection_operands *nvfp4_projection = nullptr,
    producer_native_fp8_operands *producer_fp8 = nullptr
) {
    constexpr bool UseMxFp4Dv =
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvReuseP ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogReuseP ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogSplitQReuseP ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvForwardLogSplitQReuseP ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP;
    constexpr bool UseMxFp4Dp =
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvForwardLogSplitQReuseP ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MxFp4DpDvDsDqDkForwardLogSplitQReuseP;
    constexpr bool UseMixedFp8MxFp4Dp =
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4MixedFp8MxFp4DpFp8DvReuseP;
    constexpr bool UseMixedFp8MxFp4Dv =
        UseMixedFp8MxFp4Dp && TK_FA4_BWD_MIXED_DV_THREE_COMMAND;
    constexpr bool UseFp8Dp =
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp8 ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP ||
        UseMixedFp8MxFp4Dp ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvReuseP ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogReuseP ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpMxFp4DvForwardLogSplitQReuseP;
    TORCH_CHECK(
        mixed_v_prepacked == nullptr || UseMixedFp8MxFp4Dp || UseFp8Dp,
        "prepacked V is valid only for an FP8-consuming dP route"
    );
    TORCH_CHECK(
        producer_mxfp4 == nullptr || UseMxFp4Dp,
        "producer-native MXFP4 operands are valid only for an all-MX dP/dV route"
    );
    TORCH_CHECK(
        producer_fp8 == nullptr || UseFp8Dp,
        "producer-native FP8 operands are valid only for an FP8 dP/dV route"
    );
    TORCH_CHECK(
        producer_mxfp4 == nullptr || producer_fp8 == nullptr,
        "producer-native MXFP4 and FP8 operands are mutually exclusive"
    );
    TORCH_CHECK(
        producer_fp8 == nullptr || mixed_v_prepacked == nullptr,
        "full producer-native FP8 operands and a standalone prepacked V "
        "operand are mutually exclusive"
    );
    if (mixed_v_prepacked != nullptr) {
        TORCH_CHECK(
            mixed_v_prepacked->sizes() == v.sizes() &&
                mixed_v_prepacked->scalar_type() ==
                    at::ScalarType::Float8_e4m3fn &&
                mixed_v_prepacked->is_cuda() &&
                mixed_v_prepacked->is_contiguous(),
            "prepacked mixed V must be contiguous E4M3 storage matching V"
        );
        kittens::py::device_check(v, *mixed_v_prepacked);
    }
    if (producer_mxfp4 != nullptr) {
        TORCH_CHECK(
            producer_mxfp4->dout_dp != nullptr &&
                producer_mxfp4->v_dp != nullptr &&
                producer_mxfp4->dout_dp_scale != nullptr &&
                producer_mxfp4->v_dp_scale != nullptr &&
                producer_mxfp4->dout_dv != nullptr &&
                producer_mxfp4->dout_dv_scale != nullptr &&
                producer_mxfp4->dpsum != nullptr &&
                producer_mxfp4->lse_log2 != nullptr,
            "producer-native MXFP4 bundle must provide every payload, scale, and statistic"
        );
        const std::vector<int64_t> row_shape{
            dout.size(0), dout.size(1), dout.size(2), dout.size(3) / 2
        };
        const std::vector<int64_t> scale_shape{
            dout.size(0), dout.size(1) / 128, dout.size(2), 512
        };
        const std::vector<int64_t> column_shape{
            dout.size(0), dout.size(2), dout.size(3), dout.size(1) / 2
        };
        const std::vector<int64_t> stats_shape{
            dout.size(0), dout.size(2), 1, dout.size(1)
        };
        TORCH_CHECK(
            producer_mxfp4->dout_dp->scalar_type() == at::ScalarType::Byte &&
                producer_mxfp4->v_dp->scalar_type() == at::ScalarType::Byte &&
                producer_mxfp4->dout_dp->sizes() == row_shape &&
                producer_mxfp4->v_dp->sizes() == row_shape &&
                producer_mxfp4->dout_dp->is_contiguous() &&
                producer_mxfp4->v_dp->is_contiguous(),
            "producer-native dP payloads must be contiguous uint8 [B,S,H,64]"
        );
        TORCH_CHECK(
            producer_mxfp4->dout_dp_scale->scalar_type() ==
                    at::ScalarType::Byte &&
                producer_mxfp4->v_dp_scale->scalar_type() ==
                    at::ScalarType::Byte &&
                producer_mxfp4->dout_dp_scale->sizes() == scale_shape &&
                producer_mxfp4->v_dp_scale->sizes() == scale_shape &&
                producer_mxfp4->dout_dp_scale->is_contiguous() &&
                producer_mxfp4->v_dp_scale->is_contiguous(),
            "producer-native dP scales must be contiguous uint8 [B,S/128,H,512]"
        );
        TORCH_CHECK(
            producer_mxfp4->dout_dv->scalar_type() == at::ScalarType::Byte &&
                producer_mxfp4->dout_dv->sizes() == column_shape &&
                producer_mxfp4->dout_dv->is_contiguous(),
            "producer-native dV dO must be contiguous uint8 [B,H,128,S/2]"
        );
        TORCH_CHECK(
            producer_mxfp4->dout_dv_scale->scalar_type() ==
                    at::ScalarType::Byte &&
                producer_mxfp4->dout_dv_scale->sizes() == scale_shape &&
                producer_mxfp4->dout_dv_scale->is_contiguous(),
            "producer-native dV scales must be contiguous uint8 [B,S/128,H,512]"
        );
        TORCH_CHECK(
            producer_mxfp4->dpsum->scalar_type() == at::ScalarType::Float &&
                producer_mxfp4->lse_log2->scalar_type() ==
                    at::ScalarType::Float &&
                producer_mxfp4->dpsum->sizes() == stats_shape &&
                producer_mxfp4->lse_log2->sizes() == stats_shape &&
                producer_mxfp4->dpsum->is_contiguous() &&
                producer_mxfp4->lse_log2->is_contiguous(),
            "producer-native statistics must be contiguous float32 [B,H,1,S]"
        );
        kittens::py::device_check(
            q,
            *producer_mxfp4->dout_dp,
            *producer_mxfp4->v_dp,
            *producer_mxfp4->dout_dp_scale,
            *producer_mxfp4->v_dp_scale,
            *producer_mxfp4->dout_dv,
            *producer_mxfp4->dout_dv_scale,
            *producer_mxfp4->dpsum,
            *producer_mxfp4->lse_log2
        );
    }
    if (producer_fp8 != nullptr) {
        TORCH_CHECK(
            producer_fp8->dout_dp != nullptr &&
                producer_fp8->v_dp != nullptr,
            "producer-native FP8 bundle must provide dO and V"
        );
        TORCH_CHECK(
            (producer_fp8->dpsum == nullptr) ==
                (producer_fp8->lse_log2 == nullptr),
            "producer-native FP8 statistics must provide both dPsum and "
            "log2 LSE or neither"
        );
        const std::vector<int64_t> operand_shape{
            dout.size(0), dout.size(1), dout.size(2), dout.size(3)
        };
        const std::vector<int64_t> stats_shape{
            dout.size(0), dout.size(2), 1, dout.size(1)
        };
        TORCH_CHECK(
            producer_fp8->dout_dp->scalar_type() ==
                    at::ScalarType::Float8_e4m3fn &&
                producer_fp8->v_dp->scalar_type() ==
                    at::ScalarType::Float8_e4m3fn &&
                producer_fp8->dout_dp->sizes() == operand_shape &&
                producer_fp8->v_dp->sizes() == operand_shape &&
                producer_fp8->dout_dp->is_contiguous() &&
                producer_fp8->v_dp->is_contiguous(),
            "producer-native FP8 dO/V must be contiguous E4M3 [B,S,H,128]"
        );
        if (producer_fp8->dpsum != nullptr) {
            TORCH_CHECK(
                producer_fp8->dpsum->scalar_type() ==
                        at::ScalarType::Float &&
                    producer_fp8->lse_log2->scalar_type() ==
                        at::ScalarType::Float &&
                    producer_fp8->dpsum->sizes() == stats_shape &&
                    producer_fp8->lse_log2->sizes() == stats_shape &&
                    producer_fp8->dpsum->is_contiguous() &&
                    producer_fp8->lse_log2->is_contiguous(),
                "producer-native FP8 statistics must be contiguous float32 "
                "[B,H,1,S]"
            );
            kittens::py::device_check(
                q,
                *producer_fp8->dout_dp,
                *producer_fp8->v_dp,
                *producer_fp8->dpsum,
                *producer_fp8->lse_log2
            );
        } else {
            kittens::py::device_check(
                q,
                *producer_fp8->dout_dp,
                *producer_fp8->v_dp
            );
        }
    }
    constexpr bool UseLowpDp = UseFp8Dp || UseMxFp4Dp;
    constexpr bool UseFp8Pv =
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8Pv ||
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8PvReuseP ||
        UseLowpDp;
    constexpr bool UseBf16DqReduction =
        LowpMode == bwd_cute16_kernel_candidate::detail::
            kCta2DenseLowpFp4Fp8DpPvReuseP &&
        TK_FA4_BWD_FP8DPDV_BF16_DQ_REDUCTION;
    constexpr bool SignalDqTileReady =
        DirectDqProjection || TileReadyNvfp4DqProjection;
    static_assert(
        !SignalDqTileReady ||
            (UseBf16DqReduction && UseAdaptiveQkScales &&
             BalancedSingleOwnerSchedule && IntegrateCausalFrontier),
        "tile-ready dQ projection requires the balanced adaptive BF16 route"
    );
    static_assert(
        !(DirectDqProjection && TileReadyNvfp4DqProjection),
        "BF16 and NVFP4 tile-ready dQ consumers are mutually exclusive"
    );
    TORCH_CHECK(causal, "BF16 dK/dV 2-CTA route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "BF16 dK/dV 2-CTA route supports deterministic=False only"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        ((q.size(1) == 2048 && q.size(2) == 8) ||
         (q.size(1) == 4096 &&
          (q.size(2) == 4 || q.size(2) == 8 || q.size(2) == 24 ||
           q.size(2) == 64)) ||
         (q.size(1) == 8192 &&
          (q.size(2) == 2 || q.size(2) == 4 || q.size(2) == 8 ||
           q.size(2) == 16 || q.size(2) == 24 || q.size(2) == 64)) ||
         (q.size(1) == 16384 &&
          (q.size(2) == 4 || q.size(2) == 8 || q.size(2) == 16 ||
           q.size(2) == 24 || q.size(2) == 32 || q.size(2) == 64 ||
           q.size(2) == 128)) ||
         (q.size(1) == 32768 &&
          (q.size(2) == 16 || q.size(2) == 32 || q.size(2) == 64 ||
           q.size(2) == 128)) ||
         (q.size(1) == 65536 &&
          (q.size(2) == 16 || q.size(2) == 32 || q.size(2) == 64 ||
           q.size(2) == 128)));
    TORCH_CHECK(
        supported_shape,
        "BF16 dK/dV 2-CTA route does not support this sequence/head shape; "
        "S32768 H16/H32/H64/H128, or S65536 H16/H32/H64/H128"
    );

    auto current_stream = at::cuda::getCurrentCUDAStream();
    int dq_projection_reduction_lanes = 1;
    if constexpr (DirectDqProjection) {
        if (const char *value = std::getenv(
                "TK_FA4_DQ_PROJECTION_HIERARCHICAL"
            )) {
            if (std::atoi(value) != 0) {
                dq_projection_reduction_lanes = 2;
            }
        }
    }
    if constexpr (TileReadyNvfp4DqProjection) {
        if (nvfp4_projection != nullptr &&
            nvfp4_projection->hierarchical_qkv) {
            dq_projection_reduction_lanes = 2;
            if (const char *value = std::getenv(
                    "TK_FA4_QKV_PROJECTION_DQ_LANES"
                )) {
                const int requested = std::atoi(value);
                if (requested == 1 || requested == 2) {
                    dq_projection_reduction_lanes = requested;
                }
            }
        }
    }
    at::Tensor dq_bf16_reduce;
    if constexpr (UseBf16DqReduction) {
        TORCH_CHECK(
            dq_bf16_output == nullptr || projection_qkv_output == nullptr,
            "standalone dQ and interleaved QKV outputs are mutually exclusive"
        );
        if (projection_qkv_output != nullptr) {
            TORCH_CHECK(
                projection_qkv_output->scalar_type() ==
                        at::ScalarType::BFloat16 &&
                    projection_qkv_output->is_cuda() &&
                    projection_qkv_output->is_contiguous() &&
                    projection_qkv_output->dim() == 4 &&
                    projection_qkv_output->size(0) == q.size(0) &&
                    projection_qkv_output->size(1) == q.size(1) &&
                    projection_qkv_output->size(2) == q.size(2) &&
                    projection_qkv_output->size(3) ==
                        kB300QKDim * 2 + kB300VDim,
                "projection QKV output must be contiguous BF16 "
                "[B, S, H, 512]"
            );
            kittens::py::device_check(q, *projection_qkv_output);
            dq_bf16_reduce = *projection_qkv_output;
        } else if (dq_bf16_output != nullptr) {
            TORCH_CHECK(
                dq_bf16_output->sizes() == q.sizes() &&
                    dq_bf16_output->scalar_type() ==
                        at::ScalarType::BFloat16 &&
                    dq_bf16_output->is_cuda() &&
                    dq_bf16_output->is_contiguous(),
                "direct BF16 dQ output must be contiguous BF16 matching Q"
            );
            kittens::py::device_check(q, *dq_bf16_output);
            dq_bf16_reduce = *dq_bf16_output;
        } else {
            if (dq_projection_reduction_lanes == 1) {
                dq_bf16_reduce = at::empty(q.sizes(), q.options());
            } else {
                dq_bf16_reduce = at::empty(
                    {
                        q.size(0) * dq_projection_reduction_lanes,
                        q.size(1),
                        q.size(2),
                        q.size(3)
                    },
                    q.options()
                );
            }
        }
    } else {
        TORCH_CHECK(
            dq_bf16_output == nullptr && projection_qkv_output == nullptr,
            "direct BF16 outputs require BF16 dQ reduction"
        );
    }
    if constexpr (DirectDqProjection) {
        TORCH_CHECK(
            dq_bf16_output == nullptr && projection_qkv_output == nullptr,
            "the tile-ready dQ consumer uses private BF16 reduction scratch"
        );
        TORCH_CHECK(
            nvfp4_projection == nullptr,
            "the BF16 dQ consumer cannot also launch the NVFP4 consumer"
        );
        TORCH_CHECK(
            dq_projection_weight != nullptr &&
                dq_projection_output != nullptr &&
                dq_tile_arrivals != nullptr,
            "direct dQ projection requires weight, output, and arrival storage"
        );
        const int64_t reduction = q.size(2) * kB300QKDim;
        TORCH_CHECK(
            dq_projection_weight->scalar_type() ==
                    at::ScalarType::BFloat16 &&
                dq_projection_weight->is_cuda() &&
                dq_projection_weight->is_contiguous() &&
                dq_projection_weight->dim() == 2 &&
                dq_projection_weight->size(1) == reduction &&
                dq_projection_weight->size(0) %
                        tkfa4_dq_projection::config<>::Nb ==
                    0,
            "dQ projection weight must be contiguous BF16 "
            "[hidden, heads * 192] with hidden divisible by 256"
        );
        TORCH_CHECK(
            dq_projection_output->scalar_type() ==
                    at::ScalarType::BFloat16 &&
                dq_projection_output->is_cuda() &&
                dq_projection_output->is_contiguous() &&
                dq_projection_output->dim() == 3 &&
                dq_projection_output->size(0) == q.size(0) &&
                dq_projection_output->size(1) == q.size(1) &&
                dq_projection_output->size(2) ==
                    dq_projection_weight->size(0),
            "direct dQ projection output must be contiguous BF16 "
            "[B, S, hidden]"
        );
        TORCH_CHECK(
            dq_tile_arrivals->scalar_type() == at::ScalarType::Int &&
                dq_tile_arrivals->is_cuda() &&
                dq_tile_arrivals->is_contiguous() &&
                dq_tile_arrivals->dim() == 3 &&
                dq_tile_arrivals->size(0) == q.size(0) &&
                dq_tile_arrivals->size(1) == q.size(2) &&
                dq_tile_arrivals->size(2) == q.size(1) /
                    tkfa4_dq_projection::kTileRows,
            "dQ tile arrivals must be contiguous int32 [B, H, S / 128]"
        );
        kittens::py::device_check(
            q,
            *dq_projection_weight,
            *dq_projection_output,
            *dq_tile_arrivals
        );
    } else if constexpr (TileReadyNvfp4DqProjection) {
        TORCH_CHECK(
            dq_bf16_output == nullptr && projection_qkv_output == nullptr &&
                dq_projection_weight == nullptr &&
                dq_projection_output == nullptr,
            "the tile-ready NVFP4 consumer uses private BF16 reduction "
            "scratch and compact projection tensors"
        );
        TORCH_CHECK(
            dq_tile_arrivals != nullptr && nvfp4_projection != nullptr &&
                nvfp4_projection->input_fp4 != nullptr &&
                nvfp4_projection->input_scales != nullptr &&
                nvfp4_projection->input_global_scale != nullptr &&
                nvfp4_projection->weight_fp4 != nullptr &&
                nvfp4_projection->weight_scales != nullptr &&
                nvfp4_projection->weight_global_scale != nullptr &&
                nvfp4_projection->output != nullptr &&
                nvfp4_projection->operand_ready != nullptr,
            "tile-ready NVFP4 projection requires payload, scale, output, "
            "and readiness storage"
        );
        const int64_t rows = q.size(0) * q.size(1);
        const int64_t dq_reduction = q.size(2) * kB300QKDim;
        const int64_t reduction = nvfp4_projection->hierarchical_qkv
            ? q.size(2) * (kB300QKDim * 2 + kB300VDim)
            : dq_reduction;
        const int64_t hidden = nvfp4_projection->weight_fp4->size(0);
        const int64_t q_tiles = rows / 128;
        const int64_t reduction_tiles = reduction / 256;
        TORCH_CHECK(
            rows % 256 == 0 && reduction % 256 == 0 &&
                hidden % 256 == 0,
            "tile-ready NVFP4 projection requires M/K/N divisible by 256"
        );
        TORCH_CHECK(
            !nvfp4_projection->hierarchical_qkv ||
                ((dq_projection_reduction_lanes == 1 ||
                  dq_projection_reduction_lanes == 2) &&
                 nvfp4_projection->rope_cos != nullptr &&
                 nvfp4_projection->rope_sin != nullptr),
            "hierarchical QKV projection requires one/two dQ lanes and RoPE tables"
        );
        if (nvfp4_projection->hierarchical_qkv) {
            TORCH_CHECK(
                nvfp4_projection->rope_cos->scalar_type() ==
                        at::ScalarType::BFloat16 &&
                    nvfp4_projection->rope_sin->scalar_type() ==
                        at::ScalarType::BFloat16 &&
                    nvfp4_projection->rope_cos->is_cuda() &&
                    nvfp4_projection->rope_sin->is_cuda() &&
                    nvfp4_projection->rope_cos->is_contiguous() &&
                    nvfp4_projection->rope_sin->is_contiguous() &&
                    nvfp4_projection->rope_cos->sizes() ==
                        nvfp4_projection->rope_sin->sizes() &&
                    nvfp4_projection->rope_cos->numel() ==
                        q.size(0) * q.size(1) * (kB300QKDim / 2),
                "hierarchical QKV projection requires contiguous BF16 RoPE "
                "tables with B*S*96 elements"
            );
        }
        TORCH_CHECK(
            dq_tile_arrivals->scalar_type() == at::ScalarType::Int &&
                dq_tile_arrivals->is_cuda() &&
                dq_tile_arrivals->is_contiguous() &&
                dq_tile_arrivals->sizes() ==
                    at::IntArrayRef({q.size(0), q.size(2), q_tiles}),
            "dQ tile arrivals must be contiguous int32 [B,H,S/128]"
        );
        TORCH_CHECK(
            nvfp4_projection->input_fp4->scalar_type() ==
                    at::kFloat4_e2m1fn_x2 &&
                nvfp4_projection->input_fp4->is_cuda() &&
                nvfp4_projection->input_fp4->is_contiguous() &&
                nvfp4_projection->input_fp4->sizes() ==
                    at::IntArrayRef({rows, reduction / 2}),
            "tile-ready dQ payload must be packed E2M1 [B*S,K/2]"
        );
        TORCH_CHECK(
            nvfp4_projection->input_scales->scalar_type() ==
                    at::kFloat8_e4m3fn &&
                nvfp4_projection->input_scales->is_cuda() &&
                nvfp4_projection->input_scales->is_contiguous() &&
                nvfp4_projection->input_scales->sizes() == at::IntArrayRef(
                    {q_tiles, reduction / 64, 512}
                ),
            "tile-ready dQ scales must be E4M3 [B*S/128,K/64,512]"
        );
        TORCH_CHECK(
            nvfp4_projection->input_global_scale->scalar_type() ==
                    at::ScalarType::Float &&
                nvfp4_projection->input_global_scale->is_cuda() &&
                nvfp4_projection->input_global_scale->is_contiguous() &&
                nvfp4_projection->input_global_scale->numel() == 1,
            "tile-ready dQ requires one delayed float32 global scale"
        );
        TORCH_CHECK(
            nvfp4_projection->weight_fp4->scalar_type() ==
                    at::kFloat4_e2m1fn_x2 &&
                nvfp4_projection->weight_fp4->is_cuda() &&
                nvfp4_projection->weight_fp4->is_contiguous() &&
                nvfp4_projection->weight_fp4->dim() == 2 &&
                nvfp4_projection->weight_fp4->size(1) == reduction / 2,
            "tile-ready projection weight must be packed E2M1 [N,K/2]"
        );
        TORCH_CHECK(
            nvfp4_projection->weight_scales->scalar_type() ==
                    at::kFloat8_e4m3fn &&
                nvfp4_projection->weight_scales->is_cuda() &&
                nvfp4_projection->weight_scales->is_contiguous() &&
                nvfp4_projection->weight_scales->sizes() == at::IntArrayRef(
                    {hidden / 128, reduction / 64, 512}
                ),
            "tile-ready projection weight scales must be E4M3 "
            "[N/128,K/64,512]"
        );
        TORCH_CHECK(
            nvfp4_projection->weight_global_scale->scalar_type() ==
                    at::ScalarType::Float &&
                nvfp4_projection->weight_global_scale->is_cuda() &&
                nvfp4_projection->weight_global_scale->is_contiguous() &&
                nvfp4_projection->weight_global_scale->numel() == 1,
            "tile-ready projection weight requires one float32 global scale"
        );
        TORCH_CHECK(
            nvfp4_projection->output->scalar_type() ==
                    at::ScalarType::BFloat16 &&
                nvfp4_projection->output->is_cuda() &&
                nvfp4_projection->output->is_contiguous() &&
                nvfp4_projection->output->sizes() == at::IntArrayRef(
                    {q.size(0), q.size(1), hidden}
                ),
            "tile-ready projection output must be BF16 [B,S,N]"
        );
        TORCH_CHECK(
            nvfp4_projection->operand_ready->scalar_type() ==
                    at::ScalarType::Int &&
                nvfp4_projection->operand_ready->is_cuda() &&
                nvfp4_projection->operand_ready->is_contiguous() &&
                nvfp4_projection->operand_ready->sizes() == at::IntArrayRef(
                    {q_tiles, reduction_tiles}
                ),
            "tile-ready operand counters must be int32 [B*S/128,K/256]"
        );
        kittens::py::device_check(
            q,
            *dq_tile_arrivals,
            *nvfp4_projection->input_fp4,
            *nvfp4_projection->input_scales,
            *nvfp4_projection->input_global_scale,
            *nvfp4_projection->weight_fp4,
            *nvfp4_projection->weight_scales,
            *nvfp4_projection->weight_global_scale,
            *nvfp4_projection->output,
            *nvfp4_projection->operand_ready
        );
        if (nvfp4_projection->hierarchical_qkv) {
            kittens::py::device_check(
                q,
                *nvfp4_projection->rope_cos,
                *nvfp4_projection->rope_sin
            );
        }
    } else {
        TORCH_CHECK(
            dq_projection_weight == nullptr &&
                dq_projection_output == nullptr &&
                dq_tile_arrivals == nullptr && nvfp4_projection == nullptr,
            "dQ projection tensors require the direct-consumer specialization"
        );
    }
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        0,
        C::Dqk,
        false,
        C::Dvo,
        1,
        false,
        (UseFp8Pv && !UseFp8Dp) || UseMixedFp8MxFp4Dp,
        UseFp8Dp,
        (UseMxFp4Dv || UseMixedFp8MxFp4Dv) && producer_mxfp4 == nullptr,
        (UseMxFp4Dp || UseMixedFp8MxFp4Dp) && producer_mxfp4 == nullptr
    );
    constexpr bool OverlapDqZero =
        (UseMixedFp8MxFp4Dp &&
         TK_FA4_BWD_MIXED_PREPROCESS_OVERLAP_DQ_ZERO) ||
        UseBf16DqReduction;

    if constexpr (OverlapDqZero) {
        CUDACHECK(cudaEventRecord(
            split.call_entry_ready,
            current_stream.stream()
        ));
        CUDACHECK(cudaStreamWaitEvent(
            split.dq_stream,
            split.call_entry_ready
        ));
    }

    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.total_start, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.preprocess_start, current_stream.stream()));
        if constexpr (OverlapDqZero) {
            CUDACHECK(cudaEventRecord(split.dq_start, split.dq_stream));
        }
    }
    if constexpr (UseMixedFp8MxFp4Dp && OverlapDqZero) {
        CUDACHECK(cudaMemsetAsync(
            dq.data_ptr(),
            0,
            dq.nbytes(),
            split.dq_stream
        ));
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_zero_end, split.dq_stream));
        }
    } else if constexpr (UseBf16DqReduction) {
        if (projection_qkv_output != nullptr) {
            constexpr int kThreads = 256;
            constexpr int kWarps = kThreads / 32;
            const int64_t row_count =
                projection_qkv_output->numel() /
                (kB300QKDim * 2 + kB300VDim);
            int blocks = static_cast<int>((row_count + kWarps - 1) / kWarps);
            blocks = blocks < 1024 ? blocks : 1024;
            detail::zero_interleaved_projection_dq_kernel<<<
                blocks,
                kThreads,
                0,
                split.dq_stream
            >>>(
                reinterpret_cast<uint4 *>(
                    projection_qkv_output->data_ptr()
                ),
                row_count
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        } else {
            CUDACHECK(cudaMemsetAsync(
                dq_bf16_reduce.data_ptr(),
                0,
                dq_bf16_reduce.nbytes(),
                split.dq_stream
            ));
        }
        if constexpr (SignalDqTileReady) {
            CUDACHECK(cudaMemsetAsync(
                dq_tile_arrivals->data_ptr(),
                0,
                dq_tile_arrivals->nbytes(),
                split.dq_stream
            ));
        }
        if constexpr (TileReadyNvfp4DqProjection) {
            CUDACHECK(cudaMemsetAsync(
                nvfp4_projection->operand_ready->data_ptr(),
                0,
                nvfp4_projection->operand_ready->nbytes(),
                split.dq_stream
            ));
        }
        if (detail::split_timing_enabled()) {
            CUDACHECK(cudaEventRecord(split.dq_zero_end, split.dq_stream));
        }
    }
    if constexpr (UseMixedFp8MxFp4Dp) {
        if (mixed_v_prepacked != nullptr) {
            launch_preprocess_mixed_fp8_mxfp4_dp<
                preprocess_config<kB300VDim>,
                false
            >(
                out,
                dout,
                v,
                lse,
                split.dout_fp8,
                split.dout_dp_fp8,
                split.v_fp8,
                split.dout_dp_mxfp4_scale,
                split.v_dp_mxfp4_scale,
                split.dpsum,
                split.lse_log2
            );
        } else {
            launch_preprocess_mixed_fp8_mxfp4_dp<
                preprocess_config<kB300VDim>,
                true
            >(
                out,
                dout,
                v,
                lse,
                split.dout_fp8,
                split.dout_dp_fp8,
                split.v_fp8,
                split.dout_dp_mxfp4_scale,
                split.v_dp_mxfp4_scale,
                split.dpsum,
                split.lse_log2
            );
        }
        if constexpr (
            UseMixedFp8MxFp4Dv &&
            !TK_FA4_BWD_MIXED_DV_PADDED_DO_TAIL
        ) {
            launch_preprocess_mxfp4_dout_mixed_dv_tail<
                preprocess_config<kB300VDim>
            >(
                out,
                dout,
                v,
                lse,
                split.dout_mxfp4,
                split.dout_mxfp4_scale,
                split.dpsum,
                split.lse_log2
            );
        }
    } else if constexpr (UseMxFp4Dp) {
        if (producer_mxfp4 == nullptr) {
            // Emit both dP operands and their exact E8M0 pages directly in
            // the mandatory statistics pass.  A separate feature-major dO
            // pack is retained for dV, whose MMA sees sequence as K.
            launch_preprocess_mxfp4_dp_dv<preprocess_config<kB300VDim>>(
                out,
                dout,
                v,
                lse,
                split.dout_dp_mxfp4,
                split.v_dp_mxfp4,
                split.dout_dp_mxfp4_scale,
                split.v_dp_mxfp4_scale,
                split.dout_mxfp4,
                split.dout_mxfp4_scale,
                split.dpsum,
                split.lse_log2
            );
        }
    } else if constexpr (UseMxFp4Dv) {
        // Produce dPsum, E4M3 dP operands, and the forward-compatible MXFP4
        // dV operand in one O(S) pass.  The dense kernel then loads the packed
        // payload directly instead of transposing and requantizing it for
        // every O(S^2) attention tile.
        launch_preprocess_mxfp4_dout_v<preprocess_config<kB300VDim>>(
            out,
            dout,
            v,
            lse,
            split.dout_dp_fp8,
            split.v_fp8,
            split.dout_mxfp4,
            split.dout_mxfp4_scale,
            split.dpsum,
            split.lse_log2
        );
    } else if constexpr (UseFp8Dp) {
        if (producer_fp8 == nullptr) {
            if (mixed_v_prepacked != nullptr) {
                launch_preprocess_fp8_dout_prepacked_v<
                    preprocess_config<kB300VDim>
                >(
                    out,
                    dout,
                    lse,
                    split.dout_dp_fp8,
                    split.dpsum,
                    split.lse_log2
                );
            } else {
                launch_preprocess_fp8_dout_v<
                    preprocess_config<kB300VDim>
                >(
                    out,
                    dout,
                    v,
                    lse,
                    split.dout_dp_fp8,
                    split.v_fp8,
                    split.dpsum,
                    split.lse_log2
                );
            }
        } else if (producer_fp8->dpsum == nullptr) {
            if (producer_fp8->stats_from_packed_dout) {
                launch_preprocess_fp8_stats_from_packed_dout<
                    preprocess_config<kB300VDim>
                >(
                    out,
                    *producer_fp8->dout_dp,
                    lse,
                    split.dpsum,
                    split.lse_log2
                );
            } else {
                launch_preprocess_fp8_stats_only<
                    preprocess_config<kB300VDim>
                >(
                    out,
                    dout,
                    lse,
                    split.dpsum,
                    split.lse_log2
                );
            }
        }
    } else if constexpr (UseFp8Pv) {
        launch_preprocess_fp8_dout<preprocess_config<kB300VDim>>(
            out,
            dout,
            lse,
            split.dout_fp8,
            split.dpsum,
            split.lse_log2
        );
    } else {
        launch_preprocess<preprocess_config<kB300VDim>>(
            out,
            dout,
            lse,
            split.dpsum,
            split.lse_log2
        );
    }
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(
            split.preprocess_end,
            current_stream.stream()
        ));
    }
    if constexpr (OverlapDqZero) {
        CUDACHECK(cudaEventRecord(split.dq_done, split.dq_stream));
        CUDACHECK(cudaStreamWaitEvent(current_stream.stream(), split.dq_done));
    } else {
        CUDACHECK(cudaMemsetAsync(
            dq.data_ptr(),
            0,
            dq.nbytes(),
            current_stream.stream()
        ));
    }
#if TK_FA4_BWD_FP8DPDV_STREAM_DV_TAIL_BF16
    if constexpr (UseMixedFp8MxFp4Dp) {
        CUDACHECK(cudaMemsetAsync(
            split.frontier_dv.data_ptr(),
            0,
            split.frontier_dv.nbytes(),
            current_stream.stream()
        ));
    }
#endif
    if (detail::split_timing_enabled()) {
        CUDACHECK(cudaEventRecord(split.dkdv_start, current_stream.stream()));
    }

    at::Tensor &active_lse_log2 = producer_mxfp4 != nullptr
        ? *producer_mxfp4->lse_log2
        : producer_fp8 != nullptr && producer_fp8->lse_log2 != nullptr
            ? *producer_fp8->lse_log2
            : split.lse_log2;
    at::Tensor &active_dpsum = producer_mxfp4 != nullptr
        ? *producer_mxfp4->dpsum
        : producer_fp8 != nullptr && producer_fp8->dpsum != nullptr
            ? *producer_fp8->dpsum
            : split.dpsum;
    at::Tensor *active_dout_dp_fp8 = producer_fp8 != nullptr
        ? producer_fp8->dout_dp
        : (UseFp8Dp ? &split.dout_dp_fp8 : nullptr);
    at::Tensor *active_v_fp8 = producer_fp8 != nullptr
        ? producer_fp8->v_dp
        : mixed_v_prepacked != nullptr
            ? mixed_v_prepacked
            : (UseFp8Dp ? &split.v_fp8 : nullptr);
    at::Tensor *active_dout_dp_mxfp4 = producer_mxfp4 != nullptr
        ? producer_mxfp4->dout_dp
        : (UseMxFp4Dp ? &split.dout_dp_mxfp4 : nullptr);
    at::Tensor *active_v_dp_mxfp4 = producer_mxfp4 != nullptr
        ? producer_mxfp4->v_dp
        : (UseMxFp4Dp ? &split.v_dp_mxfp4 : nullptr);
    at::Tensor *active_dout_dp_mxfp4_scale = producer_mxfp4 != nullptr
        ? producer_mxfp4->dout_dp_scale
        : ((UseMxFp4Dp || UseMixedFp8MxFp4Dp)
            ? &split.dout_dp_mxfp4_scale
            : nullptr);
    at::Tensor *active_v_dp_mxfp4_scale = producer_mxfp4 != nullptr
        ? producer_mxfp4->v_dp_scale
        : ((UseMxFp4Dp || UseMixedFp8MxFp4Dp)
            ? &split.v_dp_mxfp4_scale
            : nullptr);
    at::Tensor *active_dout_mxfp4 = producer_mxfp4 != nullptr
        ? producer_mxfp4->dout_dv
        : ((UseMxFp4Dv || UseMixedFp8MxFp4Dv)
            ? &split.dout_mxfp4
            : nullptr);
    at::Tensor *active_dout_mxfp4_scale = producer_mxfp4 != nullptr
        ? producer_mxfp4->dout_dv_scale
        : ((UseMxFp4Dv || UseMixedFp8MxFp4Dv)
            ? &split.dout_mxfp4_scale
            : nullptr);

    bwd_cute16_kernel_candidate::
        launch_backward_cta2_fused_dense_bf16_with_frontier<
            CoalescedBf16Store,
            DirectAsyncPeerDs,
            ProducerBulkPeerDs,
            ProducerBulkPeerDsCtaFenceOnly,
            DqReadHandoffBeforeCompletion,
            AggregateScoreConsumed,
            DirectTmaDkQ,
            TimeoutDqWait,
            UseWideDkN192,
            TimeoutAllRoleWaits,
            UseNamedDoSourceBarrier,
            UseComputeScoreFanout,
            UseRuntimeAccumulationPredicate,
            UseReducerDqFanout,
            UseReducerDqLeaderArrive,
            MergeScoreDpReady,
            WideCoalescedBf16Store,
            BulkPeerDsFromFullTile,
            CoalescedPeerDsBulk,
            WideDqKGlobalToShared,
            IntegrateCausalFrontier,
            ExactQTileCount,
            FenceDsBeforeDkdvReady,
            UseNamedDkdvLocalFanIn,
            LeaderOnlyQdoPublishFence,
            CacheQdoReadyClusterAddress,
            GroupQdoTmaLoads,
            ElectedWideDkQTmaLoad,
            ElectedPeerDoTmaLoad,
            CacheRoleClusterAddresses,
            CacheTensorCommitAddresses,
            EnsureReducerOutputDrain,
            ElectedScoreKTmaLoad,
            UseExactClusterCoordinates,
            EnforceDpTmemConsumerRelease,
            SplitDpTmemConsumerRelease,
            UseIterationCausalMask,
            UseFusedTmemPAndDs,
            OverlapFusedDqAPublication,
            PrefetchNextQdoAfterDkdv,
            PrefetchNextOwnerQdo,
            UseFusedTmemRuntimeAccumulationPredicate,
            UseBitwisePExpansion,
            UseFusedExp2Pack,
            PeelCausalPrefix,
            BranchlessDoSourceLoad,
            BranchlessDoSourceBaseSelect,
            PublishVOncePerOwner,
            BulkDoDvStage,
            LoaderOwnedDkQ,
            FuseScoreScaleLse,
            RetainPackedP,
            SplitDirectDpsumAcrossDpDoneWait,
            FusedExp2Fragment4First,
            CarryDirectStatsOffset,
            CarryAllRolePhases,
            UseExactDefaultScaleLog2e,
            ReverseDkTailTmemLoadIssue,
            PrearmNextQdoBeforeDkDone,
            UseX32TmemComputeLayout,
            UseLongSeqStatsCache,
            UseCompactScoreMma,
            UseCompactDpMma,
            UsePackedBf16DsProduct,
            SplitDqTmemAndSharedHandoff,
            DistributedDqSharedReadWait,
            BalancedSingleOwnerSchedule,
            UseSingleOwnerWarpStatsCache,
            CacheDqStageLanePointers,
            UseSlicedFp32PForDs,
            UseTmaVWithScoreK,
            UseStatsWarpScoreFanout,
            UseBatchedDqTmemLoads,
            UseDynamicDpReleaseBarrierId,
            PreissueFirstDpHalfBeforeQdoWait,
            OverlapSecondDpLoadWithReleaseBarrier,
            RelayDoDvCompletionViaExchangeWarp,
            OverlapDqPeerCopyWithDoDvCompletion,
            OverlapLocalDqStoreWithPeerCopy,
            UseNonblockingDqPublicationFollowers,
            SplitDqAliasLifetimeWithCuteTmemMap,
            DeferFirstDsTmemStoreWait,
            OverlapFinalDsTmemStoreWithPeerSharedStores,
            DelayScoreAliasReleaseUntilFirstDqTailLoad,
            InterleaveSteadyScoreExpPairs,
            ShiftOverlappingScoreHalfBeforeDpRelease,
            BuildCompactDpDescriptorsAfterWait,
            LateTensorCommitAddressSharedMask,
            CacheCompactDpDescriptorsInShared,
            OverlapFirstDpsumQuarterWithSecondPStore,
            HoistReducerDpReadyBeforeScoreWait,
            PipelineFirstDpQuarterLoads,
            PublishNextQdoAtDqAliasRelease,
            JoinNextQdoWithDqAliasRelease,
            PrecomputePostScoreFanoutAddresses,
            PrecomputeScoreIterationDeltaUnderFanout,
            UseNativeX32Lowp,
            LowpMode,
            ReuseDqDsForDk,
            UseAdaptiveQkScales,
            SignalDqTileReady,
            UseRank128Score
        >(
            q,
            k,
            v,
            dout,
            active_lse_log2,
            active_dpsum,
            dq,
            dk,
            dv,
            scale,
            current_stream.stream(),
            split.dkdv_stream,
            split.dense_main_done,
            split.frontier_dv_done,
            split.frontier_dk,
            split.frontier_dv,
            q_lowp,
            k_lowp,
            dq_output_scale,
            dk_output_scale,
            ds_quant_scale,
            score_q_lowp,
            score_k_lowp,
            UseFp8Pv
                ? (UseMixedFp8MxFp4Dp
                    ? &split.dout_fp8
                    : (UseFp8Dp && !UseMxFp4Dv
                    ? active_dout_dp_fp8
                    : (UseMxFp4Dv ? nullptr : &split.dout_fp8)))
                : nullptr,
            active_dout_dp_fp8,
            UseFp8Dp
                ? (mixed_v_prepacked != nullptr
                    ? mixed_v_prepacked
                    : active_v_fp8)
                : nullptr,
            active_dout_mxfp4,
            active_dout_mxfp4_scale,
            active_dout_dp_mxfp4,
            active_v_dp_mxfp4,
            active_dout_dp_mxfp4_scale,
            active_v_dp_mxfp4_scale,
            q_dk_mxfp4,
            k_dq_mxfp4,
            q_dk_nvfp4_scale,
            k_dq_nvfp4_scale,
            adaptive_qk_scales,
            UseBf16DqReduction ? &dq_bf16_reduce : nullptr,
            projection_qkv_output,
            dq_bf16_output == nullptr && projection_qkv_output == nullptr &&
                !SignalDqTileReady,
            SignalDqTileReady
                ? reinterpret_cast<uint32_t *>(dq_tile_arrivals->data_ptr())
                : nullptr,
            dq_projection_reduction_lanes,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        );

    if constexpr (DirectDqProjection) {
        auto launch_projection = [&]<typename ProjectionC>() {
        using ProjectionG = tkfa4_dq_projection::globals<ProjectionC>;
        const int heads = static_cast<int>(q.size(2));
        const int reduction = heads * kB300QKDim;
        const int hidden = static_cast<int>(dq_projection_weight->size(0));
        const int projection_row_blocks = static_cast<int>(q.size(1)) /
            ProjectionC::Mb;
        const int projection_column_blocks = hidden / ProjectionC::Nb;
        const int projection_block_count =
            projection_row_blocks * projection_column_blocks;
        const bool wait_for_attention =
            std::getenv("TK_FA4_DQ_PROJECTION_WAIT_FOR_ATTENTION") != nullptr;
        const char *assist_value = std::getenv(
            "TK_FA4_DQ_PROJECTION_ASSIST"
        );
        const char *cublas_tail_value = std::getenv(
            "TK_FA4_DQ_PROJECTION_CUBLAS_TAIL"
        );
        const bool use_cublas_tail =
            dq_projection_reduction_lanes == 1 &&
            !wait_for_attention &&
            (cublas_tail_value == nullptr
                ? hidden > 1024
                : std::atoi(cublas_tail_value) != 0);
        const bool use_tail_assist =
            !wait_for_attention && !use_cublas_tail &&
            assist_value != nullptr && std::atoi(assist_value) != 0;
        int early_row_blocks = hidden <= 1024
            ? (projection_row_blocks * 3) / 4
            : projection_row_blocks / 4;
        if (const char *value = std::getenv(
                "TK_FA4_DQ_PROJECTION_EARLY_ROW_BLOCKS"
            )) {
            const int requested = std::atoi(value);
            if (requested > 0) {
                early_row_blocks = min(requested, projection_row_blocks);
            }
        }
        early_row_blocks = max(1, min(
            early_row_blocks,
            projection_row_blocks
        ));
        const bool split_projection_tail =
            use_cublas_tail || use_tail_assist;
        const int early_block_end = split_projection_tail
            ? early_row_blocks * projection_column_blocks
            : projection_block_count;
        ProjectionG projection_g{
            kittens::py::tensor_to_gl<typename ProjectionG::A_gl, false>(
                dq_bf16_reduce,
                1,
                1,
                static_cast<int>(q.size(1)) *
                    dq_projection_reduction_lanes,
                reduction
            ),
            kittens::py::tensor_to_gl<typename ProjectionG::B_gl, false>(
                *dq_projection_weight,
                1,
                1,
                hidden,
                reduction
            ),
            kittens::py::tensor_to_gl<typename ProjectionG::D_gl, false>(
                *dq_projection_output,
                1,
                1,
                static_cast<int>(q.size(1)),
                hidden
            ),
            reinterpret_cast<const uint32_t *>(
                dq_tile_arrivals->data_ptr()
            ),
            heads,
            0,
            early_block_end,
            0,
        };
        // A pre-launched polling cluster prevents the producer cluster grid
        // from being admitted on SM100.  Enqueue after the producer launch on
        // a high-priority stream instead: completed producer waves then admit
        // the bounded consumer ahead of pending low-priority work.
        CUDACHECK(cudaStreamWaitEvent(
            split.dq_projection_stream,
            split.dq_done
        ));
        // A device-side stream wait delays cluster admission without
        // consuming an SM.  Admit after a useful head prefix is complete;
        // the persistent kernel can drain that prefix immediately, then its
        // per-head acquire waits overlap the remaining attention heads.
        if (wait_for_attention) {
            // Diagnostic/full-completion rung: if this is stable while the
            // per-head route is not, the remaining defect is producer
            // publication rather than the projection GEMM itself.
            CUDACHECK(cudaEventRecord(split.dq_end, current_stream.stream()));
            CUDACHECK(cudaStreamWaitEvent(
                split.dq_projection_stream,
                split.dq_end
            ));
        } else {
            int admission_heads = min(2, heads);
            if (const char *value = std::getenv(
                    "TK_FA4_DQ_PROJECTION_ADMISSION_HEADS"
                )) {
                const int requested = std::atoi(value);
                if (requested > 0) {
                    admission_heads = min(requested, heads);
                }
            }
            const int q_tiles = static_cast<int>(q.size(1)) /
                tkfa4_dq_projection::kTileRows;
            auto *admission_counter =
                reinterpret_cast<uint32_t *>(dq_tile_arrivals->data_ptr()) +
                static_cast<size_t>(admission_heads - 1) * q_tiles;
            CUCHECK(cuStreamWaitValue32(
                reinterpret_cast<CUstream>(split.dq_projection_stream),
                static_cast<CUdeviceptr>(
                    reinterpret_cast<uintptr_t>(admission_counter)
                ),
                2u,
                CU_STREAM_WAIT_VALUE_GEQ
            ));
        }
        tkfa4_dq_projection::launch(
            projection_g,
            split.dq_projection_stream
        );
        CUDACHECK(cudaEventRecord(
            split.dq_projection_done,
            split.dq_projection_stream
        ));
        if (use_cublas_tail && early_block_end < projection_block_count) {
            const int tail_row = early_row_blocks * ProjectionC::Mb;
            auto dq_matrix = dq_bf16_reduce.view({
                static_cast<int64_t>(q.size(1)),
                static_cast<int64_t>(reduction)
            });
            auto output_matrix = dq_projection_output->view({
                static_cast<int64_t>(q.size(1)),
                static_cast<int64_t>(hidden)
            });
            auto dq_tail = dq_matrix.narrow(
                0,
                tail_row,
                static_cast<int64_t>(q.size(1)) - tail_row
            );
            auto output_tail = output_matrix.narrow(
                0,
                tail_row,
                static_cast<int64_t>(q.size(1)) - tail_row
            );
            auto projection_weight = dq_projection_weight->transpose(0, 1);
            // This GEMM is enqueued after attention by current-stream order,
            // while the high-priority persistent kernel finishes a disjoint
            // row prefix.  cuBLAS supplies the wide cleanup grid that a
            // bounded persistent launch cannot grow after attention retires.
            at::mm_out(
                output_tail,
                dq_tail,
                projection_weight
            );
        } else if (
            use_tail_assist && early_block_end < projection_block_count
        ) {
            ProjectionG assist_g = projection_g;
            assist_g.block_begin = early_block_end;
            assist_g.block_end = projection_block_count;
            assist_g.cluster_cap = hidden <= 1024
                ? 48
                : 56;
            if (const char *value = std::getenv(
                    "TK_FA4_DQ_PROJECTION_ASSIST_CLUSTERS"
                )) {
                const int requested = std::atoi(value);
                if (requested > 0) {
                    assist_g.cluster_cap = min(
                        requested,
                        ProjectionC::MAX_CLUSTERS
                    );
                }
            }
            // Stream order places this helper after attention.  It fills the
            // unclaimed output suffix with a much wider grid while the small
            // early grid finishes its disjoint prefix on the projection
            // stream.  No block atomics or duplicate output accumulation are
            // required.
            tkfa4_dq_projection::launch(
                assist_g,
                current_stream.stream()
            );
        }
        CUDACHECK(cudaStreamWaitEvent(
            current_stream.stream(),
            split.dq_projection_done
        ));
        };
        if (dq_projection_reduction_lanes == 2) {
            launch_projection.template operator()<
                tkfa4_dq_projection::config<3, 64, 2>
            >();
        } else {
            launch_projection.template operator()<
                tkfa4_dq_projection::config<>
            >();
        }
    }

    if constexpr (TileReadyNvfp4DqProjection) {
        if (nvfp4_projection->hierarchical_qkv) {
            using PackG = tkfa4_hierarchical_qkv_nvfp4::globals;
            using ProjectionC = tkfa4_projection::config<4, 4>;
            using ProjectionG = tkfa4_projection::globals<ProjectionC>;
            const int rows = static_cast<int>(q.size(0) * q.size(1));
            const int q_width = static_cast<int>(q.size(2)) * kB300QKDim;
            const int v_width = static_cast<int>(q.size(2)) * kB300VDim;
            const int reduction = 2 * q_width + v_width;
            const int hidden = static_cast<int>(
                nvfp4_projection->weight_fp4->size(0)
            );
            const int q_tiles = rows / 128;

            PackG pack_g{
                .dQ = kittens::py::tensor_to_gl<
                    typename PackG::bf16_gl,
                    false
                >(
                    dq_bf16_reduce,
                    1,
                    1,
                    rows * dq_projection_reduction_lanes,
                    q_width
                ),
                .dK = kittens::py::tensor_to_gl<
                    typename PackG::bf16_gl,
                    false
                >(dk, 1, 1, rows, q_width),
                .dV = kittens::py::tensor_to_gl<
                    typename PackG::bf16_gl,
                    false
                >(dv, 1, 1, rows, v_width),
                .A = kittens::py::tensor_to_gl<typename PackG::fp4_gl>(
                    *nvfp4_projection->input_fp4,
                    1,
                    1,
                    rows,
                    reduction / 2
                ),
                .A_sc = kittens::py::tensor_to_gl<
                    typename PackG::scale_gl,
                    false
                >(
                    *nvfp4_projection->input_scales,
                    1,
                    q_tiles,
                    reduction / 64,
                    256
                ),
                .A_scale = kittens::py::tensor_to_gl<
                    typename PackG::global_scale_gl
                >(*nvfp4_projection->input_global_scale),
                .rope_cos = reinterpret_cast<const bf16 *>(
                    nvfp4_projection->rope_cos->data_ptr()
                ),
                .rope_sin = reinterpret_cast<const bf16 *>(
                    nvfp4_projection->rope_sin->data_ptr()
                ),
                .rows = rows,
                .q_width = q_width,
                .v_width = v_width,
                .dq_reduction_lanes = dq_projection_reduction_lanes,
            };
            tkfa4_hierarchical_qkv_nvfp4::launch(
                pack_g,
                current_stream.stream()
            );

            ProjectionG projection_g{
                .A = kittens::py::tensor_to_gl<typename ProjectionG::A_gl>(
                    *nvfp4_projection->input_fp4,
                    1,
                    1,
                    rows,
                    reduction / 2
                ),
                .A_sc = kittens::py::tensor_to_gl<
                    typename ProjectionG::A_sc_gl,
                    false
                >(
                    *nvfp4_projection->input_scales,
                    1,
                    q_tiles,
                    reduction / 64,
                    256
                ),
                .A_scale = kittens::py::tensor_to_gl<
                    typename ProjectionG::scale_gl
                >(*nvfp4_projection->input_global_scale),
                .B = kittens::py::tensor_to_gl<typename ProjectionG::B_gl>(
                    *nvfp4_projection->weight_fp4,
                    1,
                    1,
                    hidden,
                    reduction / 2
                ),
                .B_sc = kittens::py::tensor_to_gl<
                    typename ProjectionG::B_sc_gl,
                    false
                >(
                    *nvfp4_projection->weight_scales,
                    1,
                    hidden / 128,
                    reduction / 64,
                    256
                ),
                .B_scale = kittens::py::tensor_to_gl<
                    typename ProjectionG::scale_gl
                >(*nvfp4_projection->weight_global_scale),
                .Q = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
                    *nvfp4_projection->output,
                    1,
                    1,
                    rows,
                    hidden
                ),
                .K = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
                    *nvfp4_projection->output,
                    1,
                    1,
                    rows,
                    hidden
                ),
                .V = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
                    *nvfp4_projection->output,
                    1,
                    1,
                    rows,
                    hidden
                ),
                .D = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
                    *nvfp4_projection->output,
                    1,
                    1,
                    rows,
                    hidden
                ),
                .output_width = hidden,
                .A_ready = nullptr,
                .A_ready_reduction_tiles = 0,
                .cluster_cap = 0,
                .A_ready_expected = 1u,
                .block_begin = 0,
                .block_end = 0,
            };
            tkfa4_projection::launch_on_stream<
                ProjectionC,
                false,
                false,
                false,
                true,
                false,
                false,
                false,
                true,
                false
            >(projection_g, current_stream.stream());
        } else {
        using PackG = tkfa4_tile_ready_nvfp4::globals;
        using ProjectionC = tkfa4_projection::config<4, 4>;
        using ProjectionG = tkfa4_projection::globals<ProjectionC>;
        const int rows = static_cast<int>(q.size(0) * q.size(1));
        const int heads = static_cast<int>(q.size(2));
        const int reduction = heads * kB300QKDim;
        const int hidden = static_cast<int>(
            nvfp4_projection->weight_fp4->size(0)
        );
        const int q_tiles = rows / 128;
        const int reduction_tiles = reduction / 256;
        const int total_pack_tasks = q_tiles * reduction_tiles;
        const int projection_row_blocks = rows / ProjectionC::Mb;
        const int projection_column_blocks = hidden / ProjectionC::Nb;
        const int projection_block_count =
            projection_row_blocks * projection_column_blocks;
        const int projection_blocks_per_supergroup =
            ProjectionC::SUPERGROUP_SIZE * projection_column_blocks;

        // A small prefix overlaps attention without leaving either pack or
        // projection artificially narrow after the producer retires.  One
        // projection supergroup owns four M256 rows, hence eight 128-row
        // readiness tiles.
        int early_q_tiles = heads <= 8 ? 16 : 8;
        if (const char *value = std::getenv(
                "TK_FA4_DQ_NVFP4_EARLY_Q_TILES"
            )) {
            const int requested = std::atoi(value);
            if (requested >= 0) {
                early_q_tiles = requested;
            }
        }
        if (early_q_tiles > 0) {
            early_q_tiles = min(q_tiles, max(8, early_q_tiles));
            early_q_tiles = max(8, (early_q_tiles / 8) * 8);
        }
        const int early_pack_tasks = early_q_tiles * reduction_tiles;
        const int early_projection_blocks = min(
            projection_block_count,
            (early_q_tiles / 8) * projection_blocks_per_supergroup
        );

        int pack_ctas = heads <= 8 ? 4 : 8;
        if (const char *value = std::getenv(
                "TK_FA4_DQ_NVFP4_PACK_CTAS"
            )) {
            const int requested = std::atoi(value);
            if (requested > 0) {
                pack_ctas = requested;
            }
        }

        int projection_clusters = hidden <= 1024 ? 4 : 8;
        if (const char *value = std::getenv(
                "TK_FA4_DQ_NVFP4_PROJECTION_CLUSTERS"
            )) {
            const int requested = std::atoi(value);
            if (requested > 0) {
                projection_clusters = requested;
            }
        }

        PackG pack_g{
            .A_bf16 = kittens::py::tensor_to_gl<
                typename PackG::A_bf16_gl,
                false
            >(dq_bf16_reduce, 1, 1, rows, reduction),
            .A_fp4x2 = kittens::py::tensor_to_gl<
                typename PackG::A_fp4x2_gl,
                false
            >(*nvfp4_projection->input_fp4, 1, 1, rows, reduction / 2),
            .A_sc = kittens::py::tensor_to_gl<
                typename PackG::A_sc_gl,
                false
            >(
                *nvfp4_projection->input_scales,
                1,
                q_tiles,
                reduction / 64,
                256
            ),
            .A_sc_global = kittens::py::tensor_to_gl<
                typename PackG::A_sc_global_gl
            >(*nvfp4_projection->input_global_scale),
            .dq_tile_arrivals = reinterpret_cast<const uint32_t *>(
                dq_tile_arrivals->data_ptr()
            ),
            .operand_ready = reinterpret_cast<uint32_t *>(
                nvfp4_projection->operand_ready->data_ptr()
            ),
            .heads = heads,
            .q_tiles = q_tiles,
            .reduction_tiles = reduction_tiles,
            .cta_cap = pack_ctas,
            .task_begin = 0,
            .task_end = early_pack_tasks,
        };
        ProjectionG projection_g{
            .A = kittens::py::tensor_to_gl<typename ProjectionG::A_gl>(
                *nvfp4_projection->input_fp4,
                1,
                1,
                rows,
                reduction / 2
            ),
            .A_sc = kittens::py::tensor_to_gl<
                typename ProjectionG::A_sc_gl,
                false
            >(
                *nvfp4_projection->input_scales,
                1,
                q_tiles,
                reduction / 64,
                256
            ),
            .A_scale = kittens::py::tensor_to_gl<
                typename ProjectionG::scale_gl
            >(*nvfp4_projection->input_global_scale),
            .B = kittens::py::tensor_to_gl<typename ProjectionG::B_gl>(
                *nvfp4_projection->weight_fp4,
                1,
                1,
                hidden,
                reduction / 2
            ),
            .B_sc = kittens::py::tensor_to_gl<
                typename ProjectionG::B_sc_gl,
                false
            >(
                *nvfp4_projection->weight_scales,
                1,
                hidden / 128,
                reduction / 64,
                256
            ),
            .B_scale = kittens::py::tensor_to_gl<
                typename ProjectionG::scale_gl
            >(*nvfp4_projection->weight_global_scale),
            .Q = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
                *nvfp4_projection->output,
                1,
                1,
                rows,
                hidden
            ),
            .K = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
                *nvfp4_projection->output,
                1,
                1,
                rows,
                hidden
            ),
            .V = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
                *nvfp4_projection->output,
                1,
                1,
                rows,
                hidden
            ),
            .D = kittens::py::tensor_to_gl<typename ProjectionG::D_gl>(
                *nvfp4_projection->output,
                1,
                1,
                rows,
                hidden
            ),
            .output_width = hidden,
            .A_ready = reinterpret_cast<const uint32_t *>(
                nvfp4_projection->operand_ready->data_ptr()
            ),
            .A_ready_reduction_tiles = reduction_tiles,
            .cluster_cap = projection_clusters,
            .A_ready_expected = 1u,
            .block_begin = 0,
            .block_end = early_projection_blocks,
        };

        if (early_pack_tasks > 0) {
            // Delay admission until the first K256 tile's second head is
            // ready; polling CTAs must not occupy SMs while attention starts.
            CUDACHECK(cudaStreamWaitEvent(
                split.dq_pack_stream,
                split.dq_done
            ));
            CUCHECK(cuStreamWaitValue32(
                reinterpret_cast<CUstream>(split.dq_pack_stream),
                static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(
                    reinterpret_cast<uint32_t *>(
                        dq_tile_arrivals->data_ptr()
                    ) + q_tiles
                )),
                2u,
                CU_STREAM_WAIT_VALUE_GEQ
            ));
            tkfa4_tile_ready_nvfp4::launch(pack_g, split.dq_pack_stream);
            CUDACHECK(cudaEventRecord(
                split.dq_pack_done,
                split.dq_pack_stream
            ));

            CUDACHECK(cudaStreamWaitEvent(
                split.dq_projection_stream,
                split.dq_done
            ));
            CUCHECK(cuStreamWaitValue32(
                reinterpret_cast<CUstream>(split.dq_projection_stream),
                static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(
                    nvfp4_projection->operand_ready->data_ptr()
                )),
                1u,
                CU_STREAM_WAIT_VALUE_GEQ
            ));
            tkfa4_projection::launch_on_stream<
                ProjectionC,
                false,
                false,
                false,
                true,
                false,
                false,
                false,
                true,
                false
            >(projection_g, split.dq_projection_stream);
            CUDACHECK(cudaEventRecord(
                split.dq_projection_done,
                split.dq_projection_stream
            ));
        }

        // Current-stream order places the cleanup after attention.  Give
        // every remaining tile its own CTA, then use the normal wide
        // projection grid for the disjoint output suffix.
        if (early_pack_tasks < total_pack_tasks) {
            PackG tail_pack_g = pack_g;
            tail_pack_g.cta_cap = -1;
            tail_pack_g.task_begin = early_pack_tasks;
            tail_pack_g.task_end = total_pack_tasks;
            tkfa4_tile_ready_nvfp4::launch(
                tail_pack_g,
                current_stream.stream()
            );
        }
        if (early_pack_tasks > 0) {
            CUDACHECK(cudaStreamWaitEvent(
                current_stream.stream(),
                split.dq_pack_done
            ));
        }
        if (early_projection_blocks < projection_block_count) {
            ProjectionG tail_projection_g = projection_g;
            tail_projection_g.cluster_cap = 0;
            tail_projection_g.block_begin = early_projection_blocks;
            tail_projection_g.block_end = projection_block_count;
            tkfa4_projection::launch_on_stream<
                ProjectionC,
                false,
                false,
                false,
                true,
                false,
                false,
                false,
                true,
                false
            >(tail_projection_g, current_stream.stream());
        }
        if (early_projection_blocks > 0) {
            CUDACHECK(cudaStreamWaitEvent(
                current_stream.stream(),
                split.dq_projection_done
            ));
        }
        }
    }

    if (detail::split_timing_enabled()) {
        float preprocess_ms = 0.0f;
        float dq_zero_ms = 0.0f;
        float dense_frontier_ms = 0.0f;
        float total_ms = 0.0f;
        CUDACHECK(cudaEventRecord(split.dkdv_end, current_stream.stream()));
        CUDACHECK(cudaEventRecord(split.total_end, current_stream.stream()));
        CUDACHECK(cudaEventSynchronize(split.total_end));
        CUDACHECK(cudaEventElapsedTime(
            &preprocess_ms,
            split.preprocess_start,
            split.preprocess_end
        ));
        if constexpr (OverlapDqZero) {
            CUDACHECK(cudaEventElapsedTime(
                &dq_zero_ms,
                split.dq_start,
                split.dq_zero_end
            ));
        }
        CUDACHECK(cudaEventElapsedTime(
            &dense_frontier_ms,
            split.dkdv_start,
            split.dkdv_end
        ));
        CUDACHECK(cudaEventElapsedTime(
            &total_ms,
            split.total_start,
            split.total_end
        ));
        std::fprintf(
            stderr,
            "cta2_fused_dense_bf16_timing_us preprocess=%.2f "
            "dq_zero=%.2f "
            "dense_frontier=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            dq_zero_ms * 1000.0f,
            dense_frontier_ms * 1000.0f,
            total_ms * 1000.0f
        );
    }
}

template <typename C, int ExactQTileCount, int ExpectedHeads>
inline void launch_backward_cta2_fused_dense_owner_q_split(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic
) {
    static_assert(
        ExactQTileCount == 16 || ExactQTileCount == 32 ||
        ExactQTileCount == 64
    );
    static_assert(ExpectedHeads > 0);
    TORCH_CHECK(causal, "owner-Q split route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "owner-Q split route supports deterministic=False only"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            q.size(1) == ExactQTileCount * 128 &&
            q.size(2) == ExpectedHeads,
        "owner-Q split route received an unsupported exact shape"
    );

    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto &split = detail::get_split_hybrid_resources(
        current_stream.device_index(),
        current_stream.stream(),
        q,
        lse,
        0,
        C::Dqk,
        false,
        C::Dvo,
        1,
        true
    );
    launch_preprocess<preprocess_config<kB300VDim>>(
        out,
        dout,
        lse,
        split.dpsum,
        split.lse_log2
    );
    CUDACHECK(cudaMemsetAsync(
        dq.data_ptr(),
        0,
        dq.nbytes(),
        current_stream.stream()
    ));

    bwd_cute16_kernel_candidate::
        launch_backward_cta2_fused_dense_owner_q_split<
            ExactQTileCount,
            ExpectedHeads
        >(
            q,
            k,
            v,
            dout,
            split.lse_log2,
            split.dpsum,
            dq,
            dk,
            dv,
            split.cached_base_dk,
            split.cached_base_dv,
            split.frontier_dk,
            split.frontier_dv,
            scale,
            current_stream.stream(),
            split.dkdv_stream,
            split.dense_main_done,
            split.frontier_dv_done
        );
}

template <
    typename C,
    int ExactQTileCount = 0,
    bool FenceDsBeforeDkdvReady = false,
    bool UseNamedDkdvLocalFanIn = false,
    bool LeaderOnlyQdoPublishFence = false,
    bool CacheQdoReadyClusterAddress = false,
    bool GroupQdoTmaLoads = false,
    bool ElectedWideDkQTmaLoad = false,
    bool ElectedPeerDoTmaLoad = false,
    bool CacheRoleClusterAddresses = false,
    bool CacheTensorCommitAddresses = false,
    bool EnsureReducerOutputDrain = false,
    bool ElectedScoreKTmaLoad = false,
    bool UseExactClusterCoordinates = false,
    bool EnforceDpTmemConsumerRelease = false,
    bool SplitDpTmemConsumerRelease = false,
    bool UseIterationCausalMask = false,
    bool UseFusedTmemPAndDs = false,
    bool OverlapFusedDqAPublication = false,
    bool PrefetchNextQdoAfterDkdv = false,
    bool PrefetchNextOwnerQdo = false,
    bool UseFusedTmemRuntimeAccumulationPredicate = false,
    bool UseBitwisePExpansion = false,
    bool UseFusedExp2Pack = false,
    bool PeelCausalPrefix = false,
    bool BranchlessDoSourceLoad = false,
    bool BranchlessDoSourceBaseSelect = false,
    bool PublishVOncePerOwner = false,
    bool BulkDoDvStage = false,
    bool LoaderOwnedDkQ = false,
    bool FuseScoreScaleLse = false,
    bool RetainPackedP = false,
    bool SplitDirectDpsumAcrossDpDoneWait = false,
    bool FusedExp2Fragment4First = false,
    bool CarryDirectStatsOffset = false,
    bool UseReducerDqLeaderArrive = true,
    bool CarryAllRolePhases = false,
    bool UseExactDefaultScaleLog2e = false,
    bool ReverseDkTailTmemLoadIssue = false,
    bool PrearmNextQdoBeforeDkDone = false,
    bool UseX32TmemComputeLayout = false,
    bool UseLongSeqStatsCache = false,
    bool UseCompactScoreMma = false,
    bool UseCompactDpMma = false,
    bool UsePackedBf16DsProduct = false,
    bool SplitDqTmemAndSharedHandoff = false,
    bool DistributedDqSharedReadWait = false,
    bool BalancedSingleOwnerSchedule = false,
    bool UseSingleOwnerWarpStatsCache = false,
    bool CacheDqStageLanePointers = false,
    bool UseSlicedFp32PForDs = false,
    bool UseTmaVWithScoreK = false,
    bool UseStatsWarpScoreFanout = false,
    bool UseBatchedDqTmemLoads = false,
    bool UseDynamicDpReleaseBarrierId = false,
    bool PreissueFirstDpHalfBeforeQdoWait = false,
    bool OverlapSecondDpLoadWithReleaseBarrier = false,
    bool RelayDoDvCompletionViaExchangeWarp = false,
    bool OverlapDqPeerCopyWithDoDvCompletion = false,
    bool OverlapLocalDqStoreWithPeerCopy = false,
    bool UseNonblockingDqPublicationFollowers = false,
    bool SplitDqAliasLifetimeWithCuteTmemMap = false,
    bool DeferFirstDsTmemStoreWait = false,
    bool OverlapFinalDsTmemStoreWithPeerSharedStores = false,
    bool DelayScoreAliasReleaseUntilFirstDqTailLoad = false,
    bool InterleaveSteadyScoreExpPairs = false,
    bool ShiftOverlappingScoreHalfBeforeDpRelease = false,
    bool BuildCompactDpDescriptorsAfterWait = false,
    int LateTensorCommitAddressSharedMask = 0,
    bool CacheCompactDpDescriptorsInShared = false,
    bool OverlapFirstDpsumQuarterWithSecondPStore = false,
    bool HoistReducerDpReadyBeforeScoreWait = false,
    bool PipelineFirstDpQuarterLoads = false,
    bool PublishNextQdoAtDqAliasRelease = false,
    bool JoinNextQdoWithDqAliasRelease = false,
    bool PrecomputePostScoreFanoutAddresses = false,
    bool PrecomputeScoreIterationDeltaUnderFanout = false,
    bool UseNativeX32Lowp = false,
    int LowpMode =
        bwd_cute16_kernel_candidate::detail::kCta2DenseLowpNone,
    bool ReuseDqDsForDk = false,
    bool UseAdaptiveQkScales = false,
    bool DirectDqProjection = false,
    bool UseRank128Score = false,
    bool TileReadyNvfp4DqProjection = false
>
inline void launch_backward_cta2_fused_dense_bf16_integrated_frontier(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    bool deterministic,
    at::Tensor *q_lowp = nullptr,
    at::Tensor *k_lowp = nullptr,
    float dq_output_scale = 1.0f,
    float dk_output_scale = 1.0f,
    float ds_quant_scale =
        bwd_cute16_kernel_candidate::detail::
            kCta2DenseFp8DefaultDsQuantScale,
    at::Tensor *score_q_lowp = nullptr,
    at::Tensor *score_k_lowp = nullptr,
    at::Tensor *q_dk_mxfp4 = nullptr,
    at::Tensor *k_dq_mxfp4 = nullptr,
    at::Tensor *q_dk_nvfp4_scale = nullptr,
    at::Tensor *k_dq_nvfp4_scale = nullptr,
    at::Tensor *mixed_v_prepacked = nullptr,
    at::Tensor *adaptive_qk_scales = nullptr,
    at::Tensor *dq_bf16_output = nullptr,
    at::Tensor *projection_qkv_output = nullptr,
    at::Tensor *dq_projection_weight = nullptr,
    at::Tensor *dq_projection_output = nullptr,
    at::Tensor *dq_tile_arrivals = nullptr,
    producer_native_mxfp4_operands *producer_mxfp4 = nullptr,
    tile_ready_nvfp4_projection_operands *nvfp4_projection = nullptr,
    producer_native_fp8_operands *producer_fp8 = nullptr
) {
    launch_backward_cta2_fused_dense_bf16_dkdv<
        C,
        true,   // CoalescedBf16Store
        false,  // DirectAsyncPeerDs
        true,   // ProducerBulkPeerDs
        true,   // ProducerBulkPeerDsCtaFenceOnly
        true,   // DqReadHandoffBeforeCompletion
        true,   // AggregateScoreConsumed
        true,   // DirectTmaDkQ
        true,   // TimeoutDqWait
        true,   // UseWideDkN192
        true,   // TimeoutAllRoleWaits
        true,   // UseNamedDoSourceBarrier
        true,   // UseComputeScoreFanout
        true,   // UseRuntimeAccumulationPredicate
        true,   // UseReducerDqFanout
        UseReducerDqLeaderArrive,
        true,   // MergeScoreDpReady
        true,   // WideCoalescedBf16Store
        true,   // BulkPeerDsFromFullTile
        true,   // CoalescedPeerDsBulk
        true,   // WideDqKGlobalToShared
        true,   // IntegrateCausalFrontier
        ExactQTileCount,
        FenceDsBeforeDkdvReady,
        UseNamedDkdvLocalFanIn,
        LeaderOnlyQdoPublishFence,
        CacheQdoReadyClusterAddress,
        GroupQdoTmaLoads,
        ElectedWideDkQTmaLoad,
        ElectedPeerDoTmaLoad,
        CacheRoleClusterAddresses,
        CacheTensorCommitAddresses,
        EnsureReducerOutputDrain,
        ElectedScoreKTmaLoad,
        UseExactClusterCoordinates,
        EnforceDpTmemConsumerRelease,
        SplitDpTmemConsumerRelease,
        UseIterationCausalMask,
        UseFusedTmemPAndDs,
        OverlapFusedDqAPublication,
        PrefetchNextQdoAfterDkdv,
        PrefetchNextOwnerQdo,
        UseFusedTmemRuntimeAccumulationPredicate,
        UseBitwisePExpansion,
        UseFusedExp2Pack,
        PeelCausalPrefix,
        BranchlessDoSourceLoad,
        BranchlessDoSourceBaseSelect,
        PublishVOncePerOwner,
        BulkDoDvStage,
        LoaderOwnedDkQ,
        FuseScoreScaleLse,
        RetainPackedP,
        SplitDirectDpsumAcrossDpDoneWait,
        FusedExp2Fragment4First,
        CarryDirectStatsOffset,
        CarryAllRolePhases,
        UseExactDefaultScaleLog2e,
        ReverseDkTailTmemLoadIssue,
        PrearmNextQdoBeforeDkDone,
        UseX32TmemComputeLayout,
        UseLongSeqStatsCache,
        UseCompactScoreMma,
        UseCompactDpMma,
        UsePackedBf16DsProduct,
        SplitDqTmemAndSharedHandoff,
        DistributedDqSharedReadWait,
        BalancedSingleOwnerSchedule,
        UseSingleOwnerWarpStatsCache,
        CacheDqStageLanePointers,
        UseSlicedFp32PForDs,
        UseTmaVWithScoreK,
        UseStatsWarpScoreFanout,
        UseBatchedDqTmemLoads,
        UseDynamicDpReleaseBarrierId,
        PreissueFirstDpHalfBeforeQdoWait,
        OverlapSecondDpLoadWithReleaseBarrier,
        RelayDoDvCompletionViaExchangeWarp,
        OverlapDqPeerCopyWithDoDvCompletion,
        OverlapLocalDqStoreWithPeerCopy,
        UseNonblockingDqPublicationFollowers,
        SplitDqAliasLifetimeWithCuteTmemMap,
        DeferFirstDsTmemStoreWait,
        OverlapFinalDsTmemStoreWithPeerSharedStores,
        DelayScoreAliasReleaseUntilFirstDqTailLoad,
        InterleaveSteadyScoreExpPairs,
        ShiftOverlappingScoreHalfBeforeDpRelease,
        BuildCompactDpDescriptorsAfterWait,
        LateTensorCommitAddressSharedMask,
        CacheCompactDpDescriptorsInShared,
        OverlapFirstDpsumQuarterWithSecondPStore,
        HoistReducerDpReadyBeforeScoreWait,
        PipelineFirstDpQuarterLoads,
        PublishNextQdoAtDqAliasRelease,
        JoinNextQdoWithDqAliasRelease,
        PrecomputePostScoreFanoutAddresses,
        PrecomputeScoreIterationDeltaUnderFanout,
        UseNativeX32Lowp,
        LowpMode,
        ReuseDqDsForDk,
        UseAdaptiveQkScales,
        DirectDqProjection,
        UseRank128Score,
        TileReadyNvfp4DqProjection
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        scale,
        deterministic,
        q_lowp,
        k_lowp,
        dq_output_scale,
        dk_output_scale,
        ds_quant_scale,
        score_q_lowp,
        score_k_lowp,
        q_dk_mxfp4,
        k_dq_mxfp4,
        q_dk_nvfp4_scale,
        k_dq_nvfp4_scale,
        mixed_v_prepacked,
        adaptive_qk_scales,
        dq_bf16_output,
        projection_qkv_output,
        dq_projection_weight,
        dq_projection_output,
        dq_tile_arrivals,
        producer_mxfp4,
        nvfp4_projection,
        producer_fp8
    );
}

}  // namespace tkfa4::bwd_cute16_candidate

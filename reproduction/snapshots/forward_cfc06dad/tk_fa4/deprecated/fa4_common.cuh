#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>

namespace tkfa4 {

using namespace kittens;

constexpr int kRefTileM = 16;
constexpr int kRefTileN = 16;
constexpr int kForwardTileM = 128;
constexpr int kForwardSubtileM = 64;
constexpr int kForwardTileN = 128;
constexpr int kWarpThreads = kittens::WARP_THREADS;
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

inline bool is_sm100_device() {
    const auto *props = at::cuda::getCurrentDeviceProperties();
    return props != nullptr && props->major >= 10;
}

template <typename T>
inline T* data_ptr(at::Tensor &t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline const T* data_ptr(const at::Tensor &t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template <typename RT>
__device__ inline void apply_reference_mask(
    RT &scores,
    int q_tile_idx,
    int k_tile_idx,
    int actual_seq_len,
    bool causal
) {
    constexpr float neg_inf = kittens::base_types::constants<float>::neg_infty();
    const int q_base = q_tile_idx * kRefTileM;
    const int k_base = k_tile_idx * kRefTileN;
    warp::apply(scores, scores, [=](int row, int col, float value) {
        const int q_idx = q_base + row;
        const int k_idx = k_base + col;
        if (q_idx >= actual_seq_len || k_idx >= actual_seq_len) {
            return neg_inf;
        }
        if (causal && k_idx > q_idx) {
            return neg_inf;
        }
        return value;
    });
}

inline void check_bhsd(const at::Tensor &t, const char *name, at::ScalarType dtype) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dim() == 4, name, " must have shape (batch, heads, seqlen, head_dim)");
    TORCH_CHECK(t.dtype() == dtype, name, " has incorrect dtype");
}

inline void check_shapes(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v) {
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch sizes must match");
    TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "sequence lengths must match");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "head_dim must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v heads must match");
    TORCH_CHECK(q.size(1) >= k.size(1) && q.size(1) % k.size(1) == 0, "q heads must be a multiple of kv heads");
}

}  // namespace tkfa4

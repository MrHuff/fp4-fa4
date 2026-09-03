#pragma once

#include "fa4_common.cuh"
#include "fa4_scheduler.cuh"

#include <cstring>
#include <cstdlib>

namespace tkfa4::bwd {

template <int D>
struct dq_globals {
    using q_gl = gl<bf16, -1, -1, -1, D>;
    using k_gl = gl<bf16, -1, -1, -1, D>;
    using v_gl = gl<bf16, -1, -1, -1, D>;
    using do_gl = gl<bf16, -1, -1, -1, D>;
    using dq_gl = gl<float, -1, -1, -1, D>;
    using l_tile = col_vec<st_fl<kRefTileM, D>>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl = gl<float, -1, -1, -1, -1, l_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    do_gl dout;
    dq_gl dq;
    l_gl l_aux;
    d_gl delta;
    float scale;
    float scale_log2e;
    int seq_len;
    int actual_seq_len;
    int head_ratio;
};

namespace detail {

enum class backward_mode {
    Auto,
    Ref,
    Hot,
};

inline backward_mode resolve_backward_mode() {
    const char *mode = std::getenv("TK_FA4_BWD_MODE");
    if (mode == nullptr) {
        return backward_mode::Auto;
    }
    if (std::strcmp(mode, "ref") == 0) {
        return backward_mode::Ref;
    }
    if (std::strcmp(mode, "hot") == 0) {
        return backward_mode::Hot;
    }
    return backward_mode::Auto;
}

inline bool hot_backward_supported(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len
) {
    return q.size(3) == 128 &&
           !causal &&
           q.size(1) == k.size(1) &&
           actual_seq_len == q.size(2) &&
           q.size(2) % 128 == 0;
}

inline bool use_hot_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    bool causal,
    int actual_seq_len
) {
    const bool supported = hot_backward_supported(q, k, causal, actual_seq_len);
    switch (resolve_backward_mode()) {
        case backward_mode::Ref:
            return false;
        case backward_mode::Hot:
            TORCH_CHECK(supported, "TK_FA4_BWD_MODE=hot requires dense non-causal MHA with head_dim=128 and unpadded seqlen");
            return true;
        case backward_mode::Auto:
            // Keep auto on the tiled reference path until the explicit hot kernels are competitive.
            return false;
    }
    return false;
}

template <typename T>
__device__ inline T *cluster_map_shared_ptr(T *ptr, int dst_cta) {
    const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint32_t mapped_addr = 0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(mapped_addr)
        : "r"(shared_addr), "r"(dst_cta)
    );
    const unsigned long long mapped_addr64 = static_cast<unsigned long long>(mapped_addr);
    unsigned long long generic_addr = 0;
    asm volatile(
        "cvta.shared.u64 %0, %1;\n"
        : "=l"(generic_addr)
        : "l"(mapped_addr64)
    );
    return reinterpret_cast<T *>(generic_addr);
}

__host__ __device__ inline int64_t q_offset_bhsd(
    int batch_idx,
    int head_idx,
    int seq_idx,
    int d,
    int num_heads,
    int seqlen,
    int head_dim
) {
    return (((static_cast<int64_t>(batch_idx) * num_heads + head_idx) * seqlen + seq_idx) * head_dim + d);
}

__host__ __device__ inline int64_t row_offset_bhs(
    int batch_idx,
    int head_idx,
    int seq_idx,
    int num_heads,
    int seqlen
) {
    return ((static_cast<int64_t>(batch_idx) * num_heads + head_idx) * seqlen + seq_idx);
}

__device__ inline float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ inline float block_reduce_sum_128(float value, float *warp_sums) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = lane < 4 ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = block_sum;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

__device__ inline float warp_broadcast(float value, int src_lane = 0) {
    return __shfl_sync(0xffffffffu, value, src_lane);
}

__device__ inline void load_q_rows_to_shared(
    const __nv_bfloat16 *src,
    float *dst,
    int num_heads,
    int seqlen,
    int head_dim,
    int batch_idx,
    int seq_base,
    int head_idx,
    int rows,
    int tid,
    int threads
) {
    const int total = rows * head_dim;
    for (int idx = tid; idx < total; idx += threads) {
        const int row = idx / head_dim;
        const int d = idx % head_dim;
        dst[idx] = __bfloat162float(src[q_offset_bhsd(batch_idx, head_idx, seq_base + row, d, num_heads, seqlen, head_dim)]);
    }
}

__device__ inline void load_row_scalars_to_shared(
    const float *src,
    float *dst,
    int num_heads,
    int seqlen,
    int batch_idx,
    int seq_base,
    int head_idx,
    int rows,
    int tid,
    int threads
) {
    for (int idx = tid; idx < rows; idx += threads) {
        dst[idx] = src[row_offset_bhs(batch_idx, head_idx, seq_base + idx, num_heads, seqlen)];
    }
}

__device__ inline void load_lse_rows_to_shared(
    const float *src,
    float *dst,
    int num_heads,
    int seqlen,
    int batch_idx,
    int seq_base,
    int head_idx,
    int rows,
    float scale,
    int tid,
    int threads
) {
    for (int idx = tid; idx < rows; idx += threads) {
        dst[idx] = -src[row_offset_bhs(batch_idx, head_idx, seq_base + idx, num_heads, seqlen)] * scale;
    }
}

inline int persistent_waves(int seqlen) {
    if (seqlen <= 512) {
        return 1;
    }
    if (seqlen <= 4096) {
        return 2;
    }
    return 3;
}

template <int D>
__global__ __launch_bounds__(256, 1)
void dq_2cta_hot_kernel(const __grid_constant__ dq_globals<D> g) {
    static_assert(D == 128, "Hot dQ kernel is specialized for head_dim 128.");
    constexpr int kTilesPerBlock = 8;

    using bf_tile = st_bf<kRefTileM, D>;

    __shared__ alignas(1024) bf_tile k_smem[kTilesPerBlock];
    __shared__ alignas(1024) bf_tile v_smem[kTilesPerBlock];

    const int seqlen = g.seq_len;
    const int warp = threadIdx.x >> 5;
    const int num_k_blocks = seqlen / (kRefTileN * kTilesPerBlock);

    const int q_block_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int q_tile_base = q_block_idx * kTilesPerBlock;
    const int q_subtile_idx = q_tile_base + warp;

    rt_bf<kRefTileM, D> q_reg, do_reg, k_reg, v_reg;
    rt_bf<kRefTileM, kRefTileN> ds_bf;
    rt_bf<kRefTileM, D, ducks::rt_layout::col> k_reg_col;
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;
    rt_fl<kRefTileM, D> dq_accum;
    using vec_t = typename rt_fl<kRefTileM, kRefTileN>::col_vec;
    vec_t l_aux_vec, delta_vec;

    warp::load(q_reg, g.q, {batch_idx, head_idx, q_subtile_idx, 0});
    warp::load(do_reg, g.dout, {batch_idx, head_idx, q_subtile_idx, 0});
    warp::load(l_aux_vec, g.l_aux, {batch_idx, head_idx, 0, q_subtile_idx});
    warp::load(delta_vec, g.delta, {batch_idx, head_idx, 0, q_subtile_idx});
    warp::zero(dq_accum);

    for (int k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
        const int k_tile_base = k_block_idx * kTilesPerBlock;
        warp::load(k_reg, g.k, {batch_idx, head_idx, k_tile_base + warp, 0});
        warp::load(v_reg, g.v, {batch_idx, head_idx, k_tile_base + warp, 0});
        warp::store(k_smem[warp], k_reg);
        warp::store(v_smem[warp], v_reg);
        __syncthreads();

        for (int subtile = 0; subtile < kTilesPerBlock; ++subtile) {
            warp::load(k_reg, k_smem[subtile]);
            warp::load(v_reg, v_smem[subtile]);
            warp::swap_layout(k_reg_col, k_reg);

            warp::broadcast_row(p, l_aux_vec);
            warp::mma_ABt(p, q_reg, k_reg, p);
            warp::mul(p, p, g.scale_log2e);
            warp::exp2(p, p);

            warp::zero(dp);
            warp::mma_ABt(dp, do_reg, v_reg, dp);
            warp::sub_row(dp, dp, delta_vec);
            warp::mul(ds, p, dp);
            warp::mul(ds, ds, g.scale);
            warp::copy(ds_bf, ds);
            warp::mma_AB(dq_accum, ds_bf, k_reg_col, dq_accum);
        }
        __syncthreads();
    }

    warp::store(g.dq, dq_accum, {batch_idx, head_idx, q_subtile_idx, 0});
}

template <int D>
__global__ __launch_bounds__(128, 1)
void dq_hot_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const __nv_bfloat16 *__restrict__ dout,
    const float *__restrict__ l_aux,
    const float *__restrict__ delta,
    float *__restrict__ dq,
    int num_heads,
    int seqlen,
    float scale
) {
    static_assert(D == 128, "Hot dQ kernel is specialized for head_dim 128.");
    const int seq_q = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int d = threadIdx.x;

    __shared__ float q_row[128];
    __shared__ float dO_row[128];
    __shared__ float warp_score[4];
    __shared__ float warp_dp[4];
    __shared__ float prob_shared;
    __shared__ float dS_shared;

    q_row[d] = __bfloat162float(q[q_offset_bhsd(batch_idx, head_idx, seq_q, d, num_heads, seqlen, D)]);
    dO_row[d] = __bfloat162float(dout[q_offset_bhsd(batch_idx, head_idx, seq_q, d, num_heads, seqlen, D)]);
    __syncthreads();

    float dq_acc = 0.0f;
    const float row_lse = -l_aux[row_offset_bhs(batch_idx, head_idx, seq_q, num_heads, seqlen)] * scale;
    const float row_delta = delta[row_offset_bhs(batch_idx, head_idx, seq_q, num_heads, seqlen)];

    for (int seq_k = 0; seq_k < seqlen; ++seq_k) {
        const float kv = __bfloat162float(k[q_offset_bhsd(batch_idx, head_idx, seq_k, d, num_heads, seqlen, D)]);
        const float vv = __bfloat162float(v[q_offset_bhsd(batch_idx, head_idx, seq_k, d, num_heads, seqlen, D)]);

        const float score = block_reduce_sum_128(q_row[d] * kv, warp_score);
        const float dP = block_reduce_sum_128(dO_row[d] * vv, warp_dp);

        if (d == 0) {
            prob_shared = __expf(score * scale - row_lse);
            dS_shared = prob_shared * (dP - row_delta);
        }
        __syncthreads();

        dq_acc += scale * dS_shared * kv;
        __syncthreads();
    }

    dq[q_offset_bhsd(batch_idx, head_idx, seq_q, d, num_heads, seqlen, D)] = dq_acc;
}

template <int D>
inline void launch_dq_hot(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    float scale
) {
    static_assert(D == 128, "Hot dQ launch is specialized for head_dim 128.");
    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int seqlen = static_cast<int>(q.size(2));

    const bool use_2cta = []() {
        const char *value = std::getenv("TK_FA4_HOT_BACKWARD_DQ_2CTA");
        return value == nullptr || std::atoi(value) != 0;
    }();

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (use_2cta) {
        using G = dq_globals<D>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(q),
            kittens::py::tensor_to_gl<typename G::k_gl>(k),
            kittens::py::tensor_to_gl<typename G::v_gl>(v),
            kittens::py::tensor_to_gl<typename G::do_gl>(dout),
            kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
            kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
            kittens::py::tensor_to_gl<typename G::d_gl>(delta, q.size(0), q.size(1), 1, q.size(2)),
            scale,
            scale * kLog2E,
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(1) / k.size(1)),
        };
        dim3 grid(seqlen / 128, num_heads, batch_size);
        dq_2cta_hot_kernel<D><<<grid, 256, 0, stream>>>(g);
    } else {
        dim3 grid(seqlen, num_heads, batch_size);
        dq_hot_kernel<D><<<grid, 128, 0, stream>>>(
            data_ptr<__nv_bfloat16>(q),
            data_ptr<__nv_bfloat16>(k),
            data_ptr<__nv_bfloat16>(v),
            data_ptr<__nv_bfloat16>(dout),
            data_ptr<float>(l_aux),
            data_ptr<float>(delta),
            data_ptr<float>(dq),
            num_heads,
            seqlen,
            scale
        );
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace detail

template <int D>
__device__ inline void reconstruct_probability_tile(
    rt_fl<kRefTileM, kRefTileN> &p,
    const rt_bf<kRefTileM, D> &q_reg,
    const rt_bf<kRefTileM, D> &k_reg,
    const typename rt_fl<kRefTileM, kRefTileN>::col_vec &l_aux,
    float scale_log2e,
    int q_tile_idx,
    int k_tile_idx,
    int actual_seq_len,
    bool causal
) {
    warp::broadcast_row(p, l_aux);
    warp::mma_ABt(p, q_reg, k_reg, p);
    warp::mul(p, p, scale_log2e);
    apply_reference_mask(p, q_tile_idx, k_tile_idx, actual_seq_len, causal);
    warp::exp2(p, p);
}

template <int D, bool CAUSAL>
__global__ __launch_bounds__(kWarpThreads, 8)
void dq_kernel(const __grid_constant__ dq_globals<D> g) {
    const int q_tile_idx = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int kv_head_idx = scheduler::q_head_to_kv_head(q_head_idx, g.head_ratio);

    rt_bf<kRefTileM, D> q_reg, k_reg, v_reg, do_reg;
    rt_bf<kRefTileM, kRefTileN> ds_bf;
    rt_bf<kRefTileM, D, ducks::rt_layout::col> k_reg_col;
    rt_fl<kRefTileM, kRefTileN> p, dp, ds;
    rt_fl<kRefTileM, D> dq_accum;
    using vec_t = typename rt_fl<kRefTileM, kRefTileN>::col_vec;
    vec_t l_aux, delta_vec;

    warp::load(q_reg, g.q, {batch_idx, q_head_idx, q_tile_idx, 0});
    warp::load(do_reg, g.dout, {batch_idx, q_head_idx, q_tile_idx, 0});
    warp::load(l_aux, g.l_aux, {batch_idx, q_head_idx, 0, q_tile_idx});
    warp::load(delta_vec, g.delta, {batch_idx, q_head_idx, 0, q_tile_idx});
    warp::zero(dq_accum);

    const int num_k_tiles = g.seq_len / kRefTileN;
    for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        warp::load(k_reg, g.k, {batch_idx, kv_head_idx, k_tile_idx, 0});
        warp::load(v_reg, g.v, {batch_idx, kv_head_idx, k_tile_idx, 0});
        warp::swap_layout(k_reg_col, k_reg);

        reconstruct_probability_tile(
            p, q_reg, k_reg, l_aux, g.scale_log2e, q_tile_idx, k_tile_idx, g.actual_seq_len, CAUSAL
        );

        warp::zero(dp);
        warp::mma_ABt(dp, do_reg, v_reg, dp);
        warp::sub_row(dp, dp, delta_vec);
        warp::mul(ds, p, dp);
        warp::mul(ds, ds, g.scale);
        warp::copy(ds_bf, ds);
        warp::mma_AB(dq_accum, ds_bf, k_reg_col, dq_accum);
    }

    warp::store(g.dq, dq_accum, {batch_idx, q_head_idx, q_tile_idx, 0});
}

template <int D>
inline void launch_dq(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    bool causal,
    float scale,
    int actual_seq_len
) {
    if constexpr (D == 128) {
        if (detail::use_hot_backward(q, k, causal, actual_seq_len)) {
            detail::launch_dq_hot<D>(q, k, v, dout, l_aux, delta, dq, scale);
            return;
        }
    }

    using G = dq_globals<D>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        kittens::py::tensor_to_gl<typename G::do_gl>(dout),
        kittens::py::tensor_to_gl<typename G::dq_gl>(dq),
        kittens::py::tensor_to_gl<typename G::l_gl>(l_aux, q.size(0), q.size(1), 1, q.size(2)),
        kittens::py::tensor_to_gl<typename G::d_gl>(delta, q.size(0), q.size(1), 1, q.size(2)),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(2)),
        actual_seq_len,
        static_cast<int>(q.size(1) / k.size(1)),
    };

    dim3 grid(q.size(2) / kRefTileM, q.size(1), q.size(0));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (causal) {
        dq_kernel<D, true><<<grid, kWarpThreads, 0, stream>>>(g);
    } else {
        dq_kernel<D, false><<<grid, kWarpThreads, 0, stream>>>(g);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::bwd

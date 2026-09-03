// Copyright (c) 2026, Google DeepMind
// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + SiLU Backward Kernel with dgamma (supports combined w1+w3 gradient)
//
// Single kernel computing:
//   1. x_normed = x * inv_rms * gamma   (recomputed)
//   2. silu_grad = sigmoid(x_normed) * (1 + x_normed * (1 - sigmoid(x_normed)))
//   3. d_normed = dx_proj * silu_grad + d_add  (d_add is optional; from w3 path)
//   4. dgamma  = sum_over_batch(d_normed * x_hat)  where x_hat = x * inv_rms
//   5. dx = inv_rms * (d_normed * gamma - x_hat * mean(x_hat * d_normed * gamma))

#include <cuda_bf16.h>
#include <cub/cub.cuh>

// SiLU backward: silu'(u) = sigmoid(u) * (1 + u * (1 - sigmoid(u)))
// Uses full-precision expf (not __expf) to match PyTorch's torch.sigmoid
__device__ __forceinline__ float silu_backward_full(float dy, float u) {
    float s = 1.0f / (1.0f + expf(-u));
    return dy * s * (1.0f + u * (1.0f - s));
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum_t(float val) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    return BlockReduce(temp).Sum(val);
}

// ─────────────────────────────────────────────────────────────────
// Main kernel: fused SiLU' + RMSNorm backward + dgamma
//
// One block per row. Computes grad_input and accumulates dgamma
// via atomicAdd (f32) across rows. All in one kernel.
// ─────────────────────────────────────────────────────────────────

template<int BLOCK_SIZE = 256>
__global__ void fused_silu_rmsnorm_backward_kernel(
    const nv_bfloat16* __restrict__ grad_output,   // dx_proj from w1 (M, K)
    const nv_bfloat16* __restrict__ d_add,         // d_normed from w3 (M, K), may be nullptr
    const nv_bfloat16* __restrict__ input,         // x_raw   (M, K)
    const nv_bfloat16* __restrict__ weight,        // gamma   (K,)
    const float* __restrict__ cached_inv_rms,      // (M,) float
    int rows, int cols,
    nv_bfloat16* __restrict__ grad_input,          // dx (M, K)
    float* __restrict__ dgamma                     // dgamma (K,) float32 — atomicAdd accumulation
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    const nv_bfloat16* row_input = input + row * cols;
    const nv_bfloat16* row_grad  = grad_output + row * cols;
    const nv_bfloat16* row_dadd  = d_add ? (d_add + row * cols) : nullptr;
    nv_bfloat16* row_dx = grad_input + row * cols;

    float inv_rms = cached_inv_rms[row];

    // ═══ PASS 1: Reduce sum(dz * x) + accumulate dgamma ═══
    // Uses Kahan compensated summation to match PyTorch's reduction precision
    float local_sum_dz_x = 0.0f;
    float kahan_comp = 0.0f;  // Kahan compensation term

    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float x_val = __bfloat162float(row_input[i]);
        float w_val = __bfloat162float(weight[i]);
        float g_out = __bfloat162float(row_grad[i]);

        float x_hat = x_val * inv_rms;

        // norm_in: round to bf16 to match what SiLU saw in forward
        float norm_in_f32 = x_val * inv_rms * w_val;
        float norm_in = __bfloat162float(__float2bfloat16_rn(norm_in_f32));

        float d_normed_f32 = silu_backward_full(g_out, norm_in);

        // Match autograd's BF16 roundtrip:
        // In unfused baseline, backward through normed.float() casts SiLU grad to bf16,
        // then w3's grad (also bf16) is accumulated in bf16 space,
        // then backward through .to(bf16) casts combined grad back to f32.
        nv_bfloat16 d_normed_bf16 = __float2bfloat16_rn(d_normed_f32);

        if (row_dadd) {
            // Accumulate in bf16 space (matching autograd grad accumulation for bf16 tensor)
            float d_add_val = __bfloat162float(row_dadd[i]);
            d_normed_bf16 = __float2bfloat16_rn(__bfloat162float(d_normed_bf16) + d_add_val);
        }

        // Cast back to f32 (backward through .to(bf16))
        float d_normed = __bfloat162float(d_normed_bf16);

        // Accumulate dgamma[i] += d_normed * x_hat (atomicAdd across rows)
        atomicAdd(&dgamma[i], d_normed * x_hat);

        float dz = d_normed * w_val;
        // Kahan compensated sum: sum(dz * x)
        float y = dz * x_val - kahan_comp;
        float t = local_sum_dz_x + y;
        kahan_comp = (t - local_sum_dz_x) - y;
        local_sum_dz_x = t;
    }

    // Block reduce → c = sum(dz * x) / cols * inv_rms
    // Autograd computes: d_inv_rms = sum(dz * x), then d_x_corr = x * (-inv_rms^3) * d_inv_rms / cols
    // Equivalent to: c = sum(dz * x) / cols, dx = inv_rms * dz - x * inv_rms^3 * c
    float row_sum = block_reduce_sum_t<BLOCK_SIZE>(local_sum_dz_x);
    __shared__ float s_c;
    if (tid == 0) s_c = row_sum / cols;
    __syncthreads();
    float c = s_c;

    float inv_rms3 = inv_rms * inv_rms * inv_rms;

    // ═══ PASS 2: Compute dx = inv_rms * dz - x * inv_rms^3 * c ═══
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float x_val = __bfloat162float(row_input[i]);
        float w_val = __bfloat162float(weight[i]);
        float g_out = __bfloat162float(row_grad[i]);

        float norm_in_f32 = x_val * inv_rms * w_val;
        float norm_in = __bfloat162float(__float2bfloat16_rn(norm_in_f32));

        float d_normed_f32 = silu_backward_full(g_out, norm_in);

        // Same BF16 roundtrip as pass 1 (matching autograd)
        nv_bfloat16 d_normed_bf16 = __float2bfloat16_rn(d_normed_f32);
        if (row_dadd) {
            float d_add_val = __bfloat162float(row_dadd[i]);
            d_normed_bf16 = __float2bfloat16_rn(__bfloat162float(d_normed_bf16) + d_add_val);
        }
        float d_normed = __bfloat162float(d_normed_bf16);

        float dz = d_normed * w_val;

        // Two gradient paths for x, matching autograd's separate accumulation:
        // Path 1: backward through (x.float() * inv_rms_u)  → dz * inv_rms
        // Path 2: backward through rsqrt(mean(x^2)+eps) chain → -x * inv_rms^3 * c
        // Autograd accumulates each path's grad on x (bf16) separately:
        //   x.grad += path1.to(bf16)
        //   x.grad += path2.to(bf16)
        // So we must round each term to bf16 before adding.
        float d_x_from_norm  = inv_rms * dz;
        float d_x_from_rsqrt = -x_val * inv_rms3 * c;

        nv_bfloat16 dx1 = __float2bfloat16_rn(d_x_from_norm);
        nv_bfloat16 dx2 = __float2bfloat16_rn(d_x_from_rsqrt);
        row_dx[i] = __float2bfloat16_rn(__bfloat162float(dx1) + __bfloat162float(dx2));
    }
}

// ─────────────────────────────────────────────────────────────────
// Host launcher (C-linkage for use from .cpp)
// ─────────────────────────────────────────────────────────────────

extern "C" void launch_fused_silu_rmsnorm_backward(
    const void* grad_output,
    const void* d_add,          // nullable: gradient from w3 path to add after SiLU bwd
    const void* input,
    const void* weight,
    const float* cached_inv_rms,
    int rows, int cols,
    void* grad_input,
    float* dgamma,              // (K,) float32: pre-zeroed by caller
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    fused_silu_rmsnorm_backward_kernel<BLOCK_SIZE><<<rows, BLOCK_SIZE, 0, stream>>>(
        static_cast<const nv_bfloat16*>(grad_output),
        static_cast<const nv_bfloat16*>(d_add),
        static_cast<const nv_bfloat16*>(input),
        static_cast<const nv_bfloat16*>(weight),
        cached_inv_rms,
        rows, cols,
        static_cast<nv_bfloat16*>(grad_input),
        dgamma
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("fused_silu_rmsnorm_backward: ") + cudaGetErrorString(err));
    }
}


// ─────────────────────────────────────────────────────────────────
// Pure RMSNorm backward (NO SiLU) — used for attention QKV paths
// ─────────────────────────────────────────────────────────────────
//
// Given d_normed (gradient w.r.t. RMSNorm output), computes:
//   1. dgamma = sum_over_batch(d_normed * x_hat)
//   2. dx = inv_rms * (d_normed * gamma - x_hat * mean(x_hat * d_normed * gamma))

template<int BLOCK_SIZE = 256>
__global__ void fused_rmsnorm_backward_kernel(
    const nv_bfloat16* __restrict__ grad_normed,    // d_normed (M, K)
    const nv_bfloat16* __restrict__ input,          // x_raw   (M, K)
    const nv_bfloat16* __restrict__ weight,         // gamma   (K,)
    const float* __restrict__ cached_inv_rms,       // (M,) float
    int rows, int cols,
    nv_bfloat16* __restrict__ grad_input,           // dx (M, K)
    float* __restrict__ grad_weight_accum           // dgamma (K,) float32
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    const nv_bfloat16* row_input = input + row * cols;
    const nv_bfloat16* row_grad  = grad_normed + row * cols;
    nv_bfloat16* row_dx = grad_input + row * cols;

    float inv_rms = cached_inv_rms[row];

    // ═══ PASS 1: Reduce sum(d_normed * norm_in) + accumulate dgamma ═══
    float local_sum = 0.0f;

    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float x_val = __bfloat162float(row_input[i]);
        float w_val = __bfloat162float(weight[i]);
        float d_y   = __bfloat162float(row_grad[i]);   // No SiLU backward

        float norm_in = x_val * inv_rms * w_val;       // x_normed
        local_sum += d_y * norm_in;

        // dgamma += d_normed * x_hat
        float x_hat = x_val * inv_rms;
        atomicAdd(&grad_weight_accum[i], d_y * x_hat);
    }

    // Block reduce → mean
    float row_sum = block_reduce_sum_t<BLOCK_SIZE>(local_sum);
    __shared__ float s_mean;
    if (tid == 0) s_mean = row_sum / cols;
    __syncthreads();
    float mean_val = s_mean;

    // ═══ PASS 2: Compute dx ═══
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float x_val = __bfloat162float(row_input[i]);
        float w_val = __bfloat162float(weight[i]);
        float d_y   = __bfloat162float(row_grad[i]);

        float d_z = d_y * w_val;
        float d_x = inv_rms * (d_z - x_val * inv_rms * mean_val);
        row_dx[i] = __float2bfloat16(d_x);
    }
}


extern "C" void launch_fused_rmsnorm_backward(
    const void* grad_normed,
    const void* input,
    const void* weight,
    const float* cached_inv_rms,
    int rows, int cols,
    void* grad_input,
    float* grad_weight_accum,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    fused_rmsnorm_backward_kernel<BLOCK_SIZE><<<rows, BLOCK_SIZE, 0, stream>>>(
        static_cast<const nv_bfloat16*>(grad_normed),
        static_cast<const nv_bfloat16*>(input),
        static_cast<const nv_bfloat16*>(weight),
        cached_inv_rms,
        rows, cols,
        static_cast<nv_bfloat16*>(grad_input),
        grad_weight_accum
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("fused_rmsnorm_backward: ") + cudaGetErrorString(err));
    }
}

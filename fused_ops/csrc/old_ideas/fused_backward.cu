#include <cuda_bf16.h>
#include "utils.cuh"
#include "vec.cuh"
#include <cub/cub.cuh>

using bf16x8 = GenericVector<nv_bfloat16, 8>;

// SiLU Derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__device__ __forceinline__ float silu_backward(float dy, float x) {
    float s = 1.0f / (1.0f + expf(-x));
    float d_act = dy * s * (1.0f + x * (1.0f - s));
    return d_act;
}

// Block Reductions
struct SumOp {
    __device__ __forceinline__ float operator()(const float &a, const float &b) const {
        return a + b;
    }
};

__device__ __forceinline__ float block_reduce_sum_custom(float val) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    return BlockReduce(temp_storage).Reduce(val, SumOp());
}

// -------------------------------------------------------------------------
// Backward Kernel - Optimized to use cached inv_rms when available
// -------------------------------------------------------------------------

__global__ void fused_backward_kernel_v2(
    const nv_bfloat16* __restrict__ grad_output,
    const nv_bfloat16* __restrict__ input,
    const nv_bfloat16* __restrict__ weight,
    const float* __restrict__ cached_inv_rms,  // Can be nullptr for fallback
    float epsilon,
    int rows, int cols,
    nv_bfloat16* __restrict__ grad_input,
    float* __restrict__ grad_weight_accum
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;

    __shared__ float inv_rms;
    
    if (cached_inv_rms != nullptr) {
        // Use cached inv_rms from forward pass (NO RECOMPUTATION!)
        if (tid == 0) {
            inv_rms = cached_inv_rms[row];
        }
        __syncthreads();
    } else {
        // Fallback: Recompute RMS stats for standalone testing
        float sum_sq = 0.0f;
        for (int i = tid * 8; i < cols; i += blockDim.x * 8) {
            bf16x8 data = bf16x8::load(input + row * cols + i);
            #pragma unroll
            for(int k=0; k<8; ++k) {
                 float val = static_cast<float>(data[k]);
                 sum_sq += val * val;
            }
        }
        float row_sum_sq = block_reduce_sum_custom(sum_sq);
        if (tid == 0) {
            inv_rms = rsqrtf(row_sum_sq / cols + epsilon);
        }
        __syncthreads();
    }

    // Pass 1: Compute sum(dy * y) for RMSNorm backward
    float sum_dy_y = 0.0f;
    for (int i = tid * 8; i < cols; i += blockDim.x * 8) {
        bf16x8 x_vec = bf16x8::load(input + row * cols + i);
        bf16x8 w_vec = bf16x8::load(weight + i);
        bf16x8 dy_vec = bf16x8::load(grad_output + row * cols + i);
        
        #pragma unroll
        for(int k=0; k<8; ++k) {
             float x_val = (float)x_vec[k];
             float w_val = (float)w_vec[k];
             float g_out = (float)dy_vec[k];
             float norm_in = x_val * inv_rms * w_val;
             
             // Backprop SiLU
             float d_y = silu_backward(g_out, norm_in);
             
             // sum(dy * y) = sum(d_y * norm_in)
             sum_dy_y += d_y * norm_in;
        }
    }
    float block_sum_dy_y = block_reduce_sum_custom(sum_dy_y);
    __shared__ float mean_dy_y;
    if (tid == 0) mean_dy_y = block_sum_dy_y / cols;
    __syncthreads();
    
    // Pass 2: Compute dX
    for (int i = tid * 8; i < cols; i += blockDim.x * 8) {
        bf16x8 x_vec = bf16x8::load(input + row * cols + i);
        bf16x8 w_vec = bf16x8::load(weight + i);
        bf16x8 dy_vec = bf16x8::load(grad_output + row * cols + i);
        bf16x8 dx_out_vec;
        
        #pragma unroll
        for(int k=0; k<8; ++k) {
             float x_val = (float)x_vec[k];
             float w_val = (float)w_vec[k];
             float g_out = (float)dy_vec[k];
             float norm_in = x_val * inv_rms * w_val;
             
             float d_y = silu_backward(g_out, norm_in);
             
             // d(z) = d_y * w
             float d_z = d_y * w_val;
             
             // RMS Backward: dx = inv_rms * (dz - x*inv_rms * mean_dy_y)
             float d_x = inv_rms * (d_z - x_val * inv_rms * mean_dy_y);
             dx_out_vec[k] = (nv_bfloat16)d_x;
        }
        dx_out_vec.store(grad_input + row * cols + i);
    }
}

// -------------------------------------------------------------------------
// Host Wrappers
// -------------------------------------------------------------------------

// Optimized version with cached inv_rms
void launch_fused_backward(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    const float* cached_inv_rms,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* grad_input
) {
    int block_size = 256;
    int grid_size = rows;
    fused_backward_kernel_v2<<<grid_size, block_size>>>(
        grad_output, input, weight, cached_inv_rms, epsilon, rows, cols, grad_input, nullptr
    );
    CUDA_CHECK(cudaGetLastError());
}

// Legacy version (backward compatible, recomputes inv_rms)
void launch_fused_backward(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* grad_input
) {
    launch_fused_backward(grad_output, input, weight, nullptr, epsilon, rows, cols, grad_input);
}

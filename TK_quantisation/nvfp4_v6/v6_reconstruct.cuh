#ifndef TK_V6_RECONSTRUCT_CUH_
#define TK_V6_RECONSTRUCT_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace tk_v6_reconstruct {

__device__ __forceinline__ int scale_swizzle_idx(int row, int k_block, int k_blocks) {
    const int m_block = row / 128;
    const int k_block_groups = k_blocks / 4;
    const int k_block_group = k_block / 4;
    const int row_in_32 = row % 32;
    const int tile_in_block = (row / 32) % 4;
    const int kb_in_block = k_block % 4;

    const int block_base = (m_block * k_block_groups + k_block_group) * 512;
    const int local_idx = row_in_32 * 16 + tile_in_block * 4 + kb_in_block;
    return block_base + local_idx;
}

__device__ __forceinline__ float unpack_fp4_elem(
    const __nv_fp4x2_e2m1 pair,
    int lane
) {
    const float2 vals = static_cast<float2>(pair);
    return lane == 0 ? vals.x : vals.y;
}

__global__ void reconstruct_rowwise_kernel(
    const __nv_fp4x2_e2m1* __restrict__ fp4,
    const __nv_fp8_e4m3* __restrict__ sc,
    const float* __restrict__ sg,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int cols
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t numel = (int64_t)rows * cols;
    if (idx >= numel) {
        return;
    }

    const int row = idx / cols;
    const int col = idx % cols;
    const int packed_idx = row * (cols / 2) + col / 2;
    const int k_block = col / 16;
    const int scale_idx = scale_swizzle_idx(row, k_block, cols / 16);
    const float local_scale = static_cast<float>(sc[scale_idx]);
    const float global_scale = sg[0];
    const float x = unpack_fp4_elem(fp4[packed_idx], col & 1);
    out[idx] = __float2bfloat16(x * local_scale * global_scale);
}

}  // namespace tk_v6_reconstruct

#endif  // TK_V6_RECONSTRUCT_CUH_

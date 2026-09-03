/*
 * TK Scale Swizzle — converts flat [M, K/16] FP8 scales to TK's
 * expected format [n_tile_m, n_tile_k, 512].
 *
 * TK's swizzle mapping for a 128×4 tile:
 *   dst = (row % 32) * 16 + (row / 32) * 4 + k
 * where row ∈ [0,128), k ∈ [0,4)
 */

#include <cuda_runtime.h>
#include <cstdint>

__global__ void tk_swizzle_scales_kernel(
    const uint8_t* __restrict__ flat_scales,  // [M, K_div16]
    uint8_t* __restrict__ tk_scales,          // [n_tile_m * n_tile_k * 512]
    int M, int K_div16, int n_tile_m, int n_tile_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tile_m * n_tile_k * 512;
    if (idx >= total) return;

    // Decompose output index into (tile_m, tile_k, dst_within_tile)
    int dst_within_tile = idx % 512;
    int tile_idx = idx / 512;
    int tile_k = tile_idx % n_tile_k;
    int tile_m = tile_idx / n_tile_k;

    // Inverse swizzle: recover (row, k) from dst
    // dst = (row % 32) * 16 + (row / 32) * 4 + k
    int group = dst_within_tile / 16;   // row % 32
    int offset = dst_within_tile % 16;  // (row/32)*4 + k
    int row_hi = offset / 4;            // row / 32
    int k = offset % 4;                 // k
    int row = row_hi * 32 + group;      // full row [0,128)

    // Source position in flat_scales[M, K_div16]
    int src_row = tile_m * 128 + row;
    int src_col = tile_k * 4 + k;

    uint8_t val = 0;
    if (src_row < M && src_col < K_div16) {
        val = flat_scales[src_row * K_div16 + src_col];
    }

    tk_scales[idx] = val;
}

extern "C" void launch_tk_swizzle_scales(
    const uint8_t* flat_scales,
    uint8_t* tk_scales,
    int M, int K_div16, int n_tile_m, int n_tile_k,
    cudaStream_t stream
) {
    int total = n_tile_m * n_tile_k * 512;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    tk_swizzle_scales_kernel<<<grid_size, block_size, 0, stream>>>(
        flat_scales, tk_scales, M, K_div16, n_tile_m, n_tile_k
    );
}

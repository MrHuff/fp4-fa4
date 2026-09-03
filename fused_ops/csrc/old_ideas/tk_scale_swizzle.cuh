/*
 * TK Scale Swizzle — header with extern declaration.
 * Actual kernel is in tk_scale_swizzle.cu (compiled by nvcc).
 */
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void launch_tk_swizzle_scales(
    const uint8_t* flat_scales,
    uint8_t* tk_scales,
    int M, int K_div16, int n_tile_m, int n_tile_k,
    cudaStream_t stream
);

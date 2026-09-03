/*
 * Forward declarations for custom quantization wrapper functions.
 * These are implemented in custom_quantize.cu which compiles the
 * custom_quantisation/ kernels in a separate TU.
 *
 * Uses NVTETensor (opaque C API pointers) so this header can be included
 * from .cpp files that only have access to TE's public C/C++ API.
 *
 * Controlled by NVTE_CUSTOM_QUANT=1 env var + -DCUSTOM_QUANT_ENABLED.
 */
#pragma once

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

// Single-tensor quantize+transpose using custom_quantisation/ path.
// input/output/noop are NVTETensor opaque pointers (from TensorWrapper::data()).
void custom_nvfp4_quantize_transpose(
    NVTETensor input, NVTETensor output, cudaStream_t stream);

// Grouped quantize+transpose using custom_quantisation/ path.
void custom_nvfp4_group_quantize_transpose(
    NVTETensor input, NVTETensor* output_list,
    const size_t* split_sections, size_t num_tensors,
    cudaStream_t stream);

// Grouped quantize+transpose with TK sg/b_sg output computed inside the kernel.
void custom_nvfp4_group_quantize_transpose_tk(
    NVTETensor input, NVTETensor* output_list,
    const size_t* split_sections, size_t num_tensors,
    float* sg_output, float* fwd_b_sg, float* dgrad_b_sg,
    int b_tile_size,
    cudaStream_t stream);

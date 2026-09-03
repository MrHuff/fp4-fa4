// ═══════════════════════════════════════════════════════════════════
// MXFP4 v2 Quantize — PyBind11 dispatch
//
// Exports:
//   mxfp4_quantize_for_gemm(input)      → (fp4, scales)  [persistent]
//   mxfp4_group_quantize_dim0(input, Ms) → list[(fp4, scales)]
//   mxfp4_multi_quantize(inputs)         → list[(fp4, scales)]
// ═══════════════════════════════════════════════════════════════════

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <vector>
#include <algorithm>

#include "mxfp4_quantize.cuh"
#include "mxfp4_group_quantize.cuh"

using namespace mxfp4_v2;

// ═══════════════════════════════════════════════════════════════════
// TMA helper (local to this CU — avoids symbol collision)
// ═══════════════════════════════════════════════════════════════════
static void create_tma_2d(
    CUtensorMap& tma,
    void* ptr,
    int64_t dimY, int64_t dimX,
    int boxY, int boxX,
    int64_t strideX,
    int elemBits
) {
    CUtensorMapDataType dtype;
    switch (elemBits) {
        case 8:  dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8; break;
        case 16: dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; break;
        case 32: dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32; break;
        default: TORCH_CHECK(false, "Unsupported elem bits: ", elemBits);
    }
    uint64_t size[2]   = {(uint64_t)dimX, (uint64_t)dimY};
    uint64_t stride[1] = {(uint64_t)(strideX * elemBits / 8)};
    uint32_t box[2]    = {(uint32_t)boxX, (uint32_t)boxY};
    uint32_t elStride[2] = {1, 1};
    auto result = cuTensorMapEncodeTiled(
        &tma, dtype, 2, ptr,
        size, stride, box, elStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}

// ═══════════════════════════════════════════════════════════════════
// Cached SM count and work counter
// ═══════════════════════════════════════════════════════════════════
static int get_num_sms() {
    static int num_sms = -1;
    if (num_sms < 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return num_sms;
}

static unsigned int* g_work_counter = nullptr;
static void ensure_work_counter() {
    if (!g_work_counter) {
        cudaMalloc(&g_work_counter, sizeof(unsigned int));
    }
}


// ═══════════════════════════════════════════════════════════════════
// Single tensor quantize — persistent kernel
// ═══════════════════════════════════════════════════════════════════
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    ensure_work_counter();

    // Allocate outputs
    auto fp4_out = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    // Create TMA descriptors
    CUtensorMap tma_in, tma_out;
    create_tma_2d(tma_in, input.data_ptr(),
                  M, K, MX_CHUNK_DIM, MX_CHUNK_DIM, K, 16);
    create_tma_2d(tma_out, fp4_out.data_ptr(),
                  M, K / 2, MX_CHUNK_DIM, MX_CHUNK_DIM / 2, K / 2, 8);

    // Zero work counter
    cudaMemsetAsync(g_work_counter, 0, sizeof(unsigned int), stream);

    // Launch persistent kernel
    const int chunks_X = K / MX_CHUNK_DIM;
    const int chunks_Y = M / MX_CHUNK_DIM;
    const int total_tiles = chunks_X * chunks_Y;
    const int num_sms = get_num_sms();
    const int num_persistent = std::min(num_sms, total_tiles);

    PersistentArgs args;
    args.work_counter = g_work_counter;
    args.tiles_X = chunks_X;
    args.tiles_Y = chunks_Y;
    args.total_tiles = total_tiles;

    const int smem = v2_shmem_size();
    cudaFuncSetAttribute(mxfp4_v2_persistent_quantize_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    mxfp4_v2_persistent_quantize_kernel<<<num_persistent, MX_THREADS, smem, stream>>>(
        tma_in, tma_out,
        reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
        M, K, args);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v2_persistent_quantize: ", cudaGetErrorString(err));

    return std::make_tuple(fp4_out, sc_out);
}


// ═══════════════════════════════════════════════════════════════════
// Group quantize dim0 — contiguous input, split at given boundaries
// ═══════════════════════════════════════════════════════════════════
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_group_quantize_dim0(torch::Tensor input, std::vector<int64_t> group_sizes) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % 128 == 0, "K must be multiple of 128");

    int ng = (int)group_sizes.size();
    TORCH_CHECK(ng >= 1 && ng <= MAX_GROUPS, "1-16 groups supported, got ", ng);

    int64_t total = 0;
    for (auto s : group_sizes) {
        TORCH_CHECK(s % 128 == 0, "Group size must be multiple of 128, got ", s);
        total += s;
    }
    TORCH_CHECK(total == M, "Group sizes sum to ", total, " but input has M=", M);

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Allocate per-group outputs + contiguous FP4 output
    auto fp4_out = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));

    std::vector<torch::Tensor> sc_outs;
    GroupArgs gargs;
    gargs.num_groups = ng;
    gargs.boundaries[0] = 0;
    for (int i = 0; i < ng; ++i) {
        int gi = (int)group_sizes[i];
        gargs.boundaries[i + 1] = gargs.boundaries[i] + gi;
        auto sc = torch::empty({gi / 128, (int)(K / 128), 32, 16},
            torch::dtype(torch::kUInt8).device(device));
        sc_outs.push_back(sc);
        gargs.scale_ptrs[i] = reinterpret_cast<uint8_t*>(sc.data_ptr());
    }

    // TMA maps for contiguous input/output
    CUtensorMap tma_in, tma_out;
    create_tma_2d(tma_in, input.data_ptr(),
                  M, K, MX_CHUNK_DIM, MX_CHUNK_DIM, K, 16);
    create_tma_2d(tma_out, fp4_out.data_ptr(),
                  M, K / 2, MX_CHUNK_DIM, MX_CHUNK_DIM / 2, K / 2, 8);

    const int chunks_X = K / MX_CHUNK_DIM;
    const int chunks_Y = M / MX_CHUNK_DIM;
    const int smem = v2_shmem_size();

    cudaFuncSetAttribute(mxfp4_v2_group_quantize_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    mxfp4_v2_group_quantize_kernel<<<dim3(chunks_X, chunks_Y), MX_THREADS, smem, stream>>>(
        tma_in, tma_out, M, K, gargs);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v2_group_quantize: ", cudaGetErrorString(err));

    // Split FP4 output and return per-group results
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    int64_t offset = 0;
    for (int i = 0; i < ng; ++i) {
        auto fp4_slice = fp4_out.narrow(0, offset, group_sizes[i]);
        results.push_back(std::make_tuple(fp4_slice.contiguous(), sc_outs[i]));
        offset += group_sizes[i];
    }
    return results;
}


// ═══════════════════════════════════════════════════════════════════
// Multi quantize — non-contiguous tensors
// ═══════════════════════════════════════════════════════════════════
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_multi_quantize(std::vector<torch::Tensor> inputs) {
    int ng = (int)inputs.size();
    TORCH_CHECK(ng >= 1 && ng <= MAX_GROUPS, "1-16 inputs supported");

    int64_t K = inputs[0].size(1);
    TORCH_CHECK(K % 128 == 0);
    for (auto& t : inputs) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous());
        TORCH_CHECK(t.scalar_type() == torch::kBFloat16 && t.dim() == 2);
        TORCH_CHECK(t.size(1) == K, "All inputs must have same K");
        TORCH_CHECK(t.size(0) % 128 == 0);
    }

    auto device = inputs[0].device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Compute boundaries and total M
    GroupArgs gargs;
    gargs.num_groups = ng;
    gargs.boundaries[0] = 0;
    int64_t total_M = 0;
    for (int i = 0; i < ng; ++i) {
        total_M += inputs[i].size(0);
        gargs.boundaries[i + 1] = (int)total_M;
    }

    // Allocate per-group outputs
    std::vector<torch::Tensor> fp4_outs, sc_outs;
    for (int i = 0; i < ng; ++i) {
        int64_t Mi = inputs[i].size(0);
        fp4_outs.push_back(torch::empty({Mi, K / 2},
            torch::dtype(torch::kFloat4_e2m1fn_x2).device(device)));
        auto sc = torch::empty({Mi / 128, (int)(K / 128), 32, 16},
            torch::dtype(torch::kUInt8).device(device));
        sc_outs.push_back(sc);
        gargs.scale_ptrs[i] = reinterpret_cast<uint8_t*>(sc.data_ptr());
    }

    // Per-group TMA maps (on device)
    std::vector<CUtensorMap> in_maps(ng), out_maps(ng);
    for (int i = 0; i < ng; ++i) {
        int64_t Mi = inputs[i].size(0);
        create_tma_2d(in_maps[i], inputs[i].data_ptr(),
                      Mi, K, MX_CHUNK_DIM, MX_CHUNK_DIM, K, 16);
        create_tma_2d(out_maps[i], fp4_outs[i].data_ptr(),
                      Mi, K / 2, MX_CHUNK_DIM, MX_CHUNK_DIM / 2, K / 2, 8);
    }

    // Copy TMA maps to device
    CUtensorMap *d_in_maps, *d_out_maps;
    cudaMalloc(&d_in_maps, ng * sizeof(CUtensorMap));
    cudaMalloc(&d_out_maps, ng * sizeof(CUtensorMap));
    cudaMemcpyAsync(d_in_maps, in_maps.data(), ng * sizeof(CUtensorMap),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_maps, out_maps.data(), ng * sizeof(CUtensorMap),
                    cudaMemcpyHostToDevice, stream);

    const int chunks_X = K / MX_CHUNK_DIM;
    const int chunks_Y = total_M / MX_CHUNK_DIM;
    const int smem = v2_shmem_size();

    cudaFuncSetAttribute(mxfp4_v2_multi_quantize_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    mxfp4_v2_multi_quantize_kernel<<<dim3(chunks_X, chunks_Y), MX_THREADS, smem, stream>>>(
        d_in_maps, d_out_maps, K, gargs);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v2_multi_quantize: ", cudaGetErrorString(err));

    // Cleanup (after stream completes)
    cudaStreamSynchronize(stream);
    cudaFree(d_in_maps);
    cudaFree(d_out_maps);

    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    for (int i = 0; i < ng; ++i) {
        results.push_back(std::make_tuple(fp4_outs[i], sc_outs[i]));
    }
    return results;
}


// ═══════════════════════════════════════════════════════════════════
// PyBind11
// ═══════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_quantize_for_gemm", &mxfp4_quantize_for_gemm,
          "MXFP4 quantize (persistent) → (fp4, scales)");
    m.def("mxfp4_group_quantize_dim0", &mxfp4_group_quantize_dim0,
          "MXFP4 group quantize contiguous input → list[(fp4, scales)]");
    m.def("mxfp4_multi_quantize", &mxfp4_multi_quantize,
          "MXFP4 multi-tensor quantize → list[(fp4, scales)]");
}

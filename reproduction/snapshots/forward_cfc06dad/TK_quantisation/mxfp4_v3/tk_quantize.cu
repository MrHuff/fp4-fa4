// MXFP4 v3 Quantize — PyBind11 dispatch
//
// Exports:
//   mxfp4_quantize_for_gemm(input) → (fp4, scales)
//   mxfp4_group_quantize_dim0(input, Ms) → list[(fp4, scales)]

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <math_constants.h>
#include <vector>
#include <algorithm>
#include <dlfcn.h>

#include "mxfp4_v3_quantize.cuh"

using namespace mxfp4_v3;

// ═══════════════════════════════════════════════════════════════════
// TMA helper
// ═══════════════════════════════════════════════════════════════════
static void create_tma_2d(
    CUtensorMap& tma,
    void* ptr,
    uint64_t dimY, uint64_t dimX,
    uint32_t boxY, uint32_t boxX,
    uint64_t strideX, size_t elemBits,
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE
) {
    typedef CUresult (*cuTensorMapEncodeTiled_t)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
        const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
        const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

    static cuTensorMapEncodeTiled_t fn = nullptr;
    if (!fn) {
        void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
        TORCH_CHECK(handle, "Failed to open libcuda.so.1");
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(dlsym(handle, "cuTensorMapEncodeTiled"));
        TORCH_CHECK(fn, "cuTensorMapEncodeTiled not found");
    }

    CUtensorMapDataType dtype;
    if (elemBits == 16) dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (elemBits == 8) dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else if (elemBits == 4) dtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    else TORCH_CHECK(false, "Unsupported elem bits: ", elemBits);

    uint64_t size[2]   = {dimX, dimY};
    uint64_t stride[1] = {strideX * elemBits / 8};
    uint32_t box[2]    = {boxX, boxY};
    uint32_t elStride[2] = {1, 1};
    auto result = fn(&tma, dtype, 2, ptr,
        size, stride, box, elStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        l2promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}

// ═══════════════════════════════════════════════════════════════════
// Cached SM count + persistent threshold
// ═══════════════════════════════════════════════════════════════════
struct CachedInfo {
    int num_sms = 0;
    int max_bps = 0;   // max blocks per SM (fused kernel)
    int p_max_bps = 0; // max blocks per SM (persistent kernel)
    int dshmem = 0;
    bool initialized = false;
};

static CachedInfo& get_cached() {
    static CachedInfo ci;
    if (!ci.initialized) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ci.num_sms, cudaDevAttrMultiProcessorCount, dev);
        ci.dshmem = v3_shmem_size();

        // Fused kernel occupancy (default RTE template)
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ci.max_bps, mxfp4_v3_fused_kernel<QuantMode::RTE>, THREADS, ci.dshmem);

        // Set attributes for other mode instantiations too
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);

        // Persistent kernel occupancy
        cudaFuncSetAttribute(mxfp4_v3_persistent_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ci.p_max_bps, mxfp4_v3_persistent_kernel<QuantMode::RTE>, THREADS, ci.dshmem);

        ci.initialized = true;
    }
    return ci;
}

// NOTE: Persistent kernel disabled pending non-determinism fix.
// Fused kernel has same pipelining and is bit-exact with v2.
static constexpr int PERSISTENT_THRESHOLD = 999999999;  // effectively disabled

// ═══════════════════════════════════════════════════════════════════
// Global work counter for persistent kernel
// ═══════════════════════════════════════════════════════════════════
static unsigned int* g_work_counter = nullptr;
static void ensure_work_counter() {
    if (!g_work_counter) {
        cudaMalloc(&g_work_counter, sizeof(unsigned int));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Single tensor quantize
// ═══════════════════════════════════════════════════════════════════
// Templated quantize function
template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm_impl(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto fp4_out = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    // TMA maps: use TILE_DIM (64) as box size for sub-tile pipelining
    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 16,
                  total_tiles >= PERSISTENT_THRESHOLD ?
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B :
                      CU_TENSOR_MAP_L2_PROMOTION_NONE);
    create_tma_2d(tma_out, fp4_out.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 4);

    if (total_tiles >= PERSISTENT_THRESHOLD) {
        // Persistent path
        ensure_work_counter();
        cudaMemsetAsync(g_work_counter, 0, sizeof(unsigned int), stream);

        int num_persistent = ci.p_max_bps * ci.num_sms;
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        PersistentArgs args;
        args.work_counter = g_work_counter;
        args.tiles_X = tiles_X;
        args.tiles_Y = tiles_Y;
        args.total_tiles = total_tiles;

        mxfp4_v3_persistent_kernel<MODE><<<num_persistent, THREADS, dshmem, stream>>>(
            tma_in, tma_out,
            reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
            M, K, args);
    } else {
        // Fused path — one CTA per chunk
        dim3 grid(tiles_X, tiles_Y);
        mxfp4_v3_fused_kernel<MODE><<<grid, THREADS, dshmem, stream>>>(
            tma_in, tma_out,
            reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
            M, K);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 quantize: ", cudaGetErrorString(err));

    return std::make_tuple(fp4_out, sc_out);
}

// Python-facing dispatch by mode int
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm(torch::Tensor input, int mode) {
    switch (mode) {
        case 1: return mxfp4_quantize_for_gemm_impl<QuantMode::ENCODE>(input);
        case 2: return mxfp4_quantize_for_gemm_impl<QuantMode::DECODE>(input);
        default: return mxfp4_quantize_for_gemm_impl<QuantMode::RTE>(input);
    }
}


// ═══════════════════════════════════════════════════════════════════
// Group quantize dim0 — single kernel launch for all groups
//
// Uses mxfp4_v3_fused_group_kernel: one kernel launch over the
// entire contiguous input, with per-group scale pointers in GroupArgs.
// FP4 output is contiguous, scales are per-group.
// ═══════════════════════════════════════════════════════════════════
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_group_quantize_dim0(torch::Tensor input, std::vector<int64_t> group_sizes) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % 128 == 0);

    int ng = (int)group_sizes.size();
    TORCH_CHECK(ng >= 1 && ng <= MAX_GROUPS);

    int64_t total = 0;
    for (auto s : group_sizes) {
        TORCH_CHECK(s % 128 == 0);
        total += s;
    }
    TORCH_CHECK(total == M);

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Allocate contiguous FP4 output
    auto fp4_all = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));

    // Allocate per-group scale tensors
    std::vector<torch::Tensor> sc_allocs(ng);
    GroupArgs args;
    memset(&args, 0, sizeof(args));
    args.num_groups = ng;
    args.boundaries[0] = 0;

    for (int i = 0; i < ng; ++i) {
        int64_t Mi = group_sizes[i];
        args.boundaries[i + 1] = args.boundaries[i] + (int)Mi;
        sc_allocs[i] = torch::empty({Mi / 128, K / 128, 32, 16},
            torch::dtype(torch::kUInt8).device(device));
        args.scale_ptrs[i] = reinterpret_cast<uint8_t*>(sc_allocs[i].data_ptr());
    }

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    // Set up grouped kernel occupancy (first time only)
    static bool grp_init = false;
    if (!grp_init) {
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        grp_init = true;
    }

    // TMA maps
    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_out, fp4_all.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 4);

    // Single kernel launch for all groups
    dim3 grid(tiles_X, tiles_Y);
    mxfp4_v3_fused_group_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
        tma_in, tma_out, M, K, args);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 group quantize: ", cudaGetErrorString(err));

    // Split FP4 output and pair with per-group scales
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    int64_t row_off = 0;
    for (int i = 0; i < ng; ++i) {
        int64_t Mi = group_sizes[i];
        auto fp4_slice = fp4_all.narrow(0, row_off, Mi).contiguous();
        results.push_back(std::make_tuple(fp4_slice, sc_allocs[i]));
        row_off += Mi;
    }
    return results;
}

// ═══════════════════════════════════════════════════════════════════
// Multi quantize — multiple non-contiguous tensors
// ═══════════════════════════════════════════════════════════════════
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_multi_quantize(std::vector<torch::Tensor> inputs) {
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    for (auto& t : inputs) {
        results.push_back(mxfp4_quantize_for_gemm(t, 0));
    }
    return results;
}


// ═══════════════════════════════════════════════════════════════════
// Fast BF16 matrix transpose kernel (tiled, smem-based)
// ═══════════════════════════════════════════════════════════════════
static constexpr int TR_TILE = 32;
__global__ void bf16_transpose_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int M, int K
) {
    __shared__ __nv_bfloat16 tile[TR_TILE][TR_TILE + 1];  // +1 to avoid bank conflicts

    int bx = blockIdx.x * TR_TILE;
    int by = blockIdx.y * TR_TILE;

    // Load tile from src (row-major: M×K)
    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    for (int j = 0; j < TR_TILE; j += blockDim.y) {
        if ((y + j) < M && x < K) {
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * K + x];
        }
    }
    __syncthreads();

    // Write transposed tile to dst (row-major: K×M)
    x = by + threadIdx.x;  // swapped
    y = bx + threadIdx.y;
    for (int j = 0; j < TR_TILE; j += blockDim.y) {
        if ((y + j) < K && x < M) {
            dst[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


// ═══════════════════════════════════════════════════════════════════
// Row + Col quantize: quantize input AND its transpose
//
// Returns: (row_fp4, row_sc, col_fp4, col_sc)
// where col is the MXFP4 quantization of input^T
// ═══════════════════════════════════════════════════════════════════
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col(torch::Tensor input, int mode) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // 1) Row-wise quantize
    auto [row_fp4, row_sc] = mxfp4_quantize_for_gemm(input, mode);

    // 2) Fast BF16 transpose: (M, K) → (K, M)
    auto transposed = torch::empty({K, M}, torch::dtype(torch::kBFloat16).device(device));
    {
        dim3 block(TR_TILE, 8);
        dim3 grid((K + TR_TILE - 1) / TR_TILE, (M + TR_TILE - 1) / TR_TILE);
        bf16_transpose_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(transposed.data_ptr()),
            M, K
        );
    }

    // 3) Quantize the transpose: (K, M) → fp4, scales
    auto [col_fp4, col_sc] = mxfp4_quantize_for_gemm(transposed, mode);

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

// ═══════════════════════════════════════════════════════════════════
// Direct BSHD dOut row + transposed-col quantize for FA backward.
//
// Narrow contract: input is contiguous BF16 [B, S, H, 128], S % 128 == 0.
// One CTA loads a [128 sequence rows x 128 head-dim cols] tile into smem,
// then emits both:
//   row_fp4/sc: quantize rows [S, 128]
//   col_fp4/sc: quantize the logical transpose [128, S]
// This removes the per-head Python loop and the intermediate BF16 transpose.
// ═══════════════════════════════════════════════════════════════════
__device__ __forceinline__ uint16_t mxfp4_bf16_abs_bits(__nv_bfloat16 v) {
    return static_cast<uint16_t>(*reinterpret_cast<const uint16_t*>(&v) & 0x7fffu);
}

template<QuantMode MODE>
__device__ __forceinline__ uint8_t mxfp4_bf16_abs_bits_to_e8m0(uint16_t abs_bits) {
    const uint8_t exp = static_cast<uint8_t>((abs_bits >> 7) & 0xffu);
    if (exp == 0) {
        return 0x00;
    }
    const uint16_t mant = abs_bits & 0x7fu;
    if constexpr (MODE == QuantMode::ENCODE) {
        return static_cast<uint8_t>((mant > 0 && exp < 0xfeu) ? exp + 1 : exp);
    } else if constexpr (MODE == QuantMode::DECODE) {
        return exp;
    } else {
        const bool round_up = (mant > 0x40u) || (mant == 0x40u && (exp & 1u));
        return static_cast<uint8_t>((round_up && exp < 0xfeu) ? exp + 1 : exp);
    }
}

template<QuantMode MODE, bool WITH_DELTA>
__global__ __launch_bounds__(THREADS)
void mxfp4_bshd128_row_col_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ out,
    fp4e2m1x2* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    fp4e2m1x2* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    float* __restrict__ delta,
    const int B,
    const int S,
    const int H
) {
    constexpr int D = 128;
    constexpr int PACKED_D = D / 2;
    constexpr int THREAD_COUNT = THREADS;

    const int s_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh - b * H;
    if (b >= B) {
        return;
    }

    __shared__ __nv_bfloat16 tile[CHUNK_DIM][CHUNK_DIM + 1];
    __shared__ float delta_part[CHUNK_DIM][SCALES_PER_CHUNK];

    const int s_base = s_tile * CHUNK_DIM;
    for (int idx = threadIdx.x; idx < CHUNK_DIM * D; idx += THREAD_COUNT) {
        const int r = idx / D;
        const int d = idx - r * D;
        const size_t in_idx =
            (((static_cast<size_t>(b) * S + (s_base + r)) * H + h) * D + d);
        tile[r][d] = input[in_idx];
    }
    __syncthreads();

    const int tiles_s = S / CHUNK_DIM;
    const size_t bh_row_base = static_cast<size_t>(bh) * S;
    const size_t bh_col_base = static_cast<size_t>(bh) * D;

    for (int combo = threadIdx.x; combo < CHUNK_DIM * SCALES_PER_CHUNK; combo += THREAD_COUNT) {
        const int row = combo / SCALES_PER_CHUNK;
        const int mx_block = combo - row * SCALES_PER_CHUNK;
        const int d0 = mx_block * MX_BLOCK;

        __nv_bfloat16 vals[MX_BLOCK];
        uint16_t block_amax_bits = 0;
        float delta_acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const __nv_bfloat16 v = tile[row][d0 + i];
            vals[i] = v;
            const uint16_t abs_bits = mxfp4_bf16_abs_bits(v);
            block_amax_bits = abs_bits > block_amax_bits ? abs_bits : block_amax_bits;
            if constexpr (WITH_DELTA) {
                const size_t in_idx =
                    (((static_cast<size_t>(b) * S + (s_base + row)) * H + h) * D + (d0 + i));
                delta_acc += __bfloat162float(out[in_idx]) * __bfloat162float(v);
            }
        }
        if constexpr (WITH_DELTA) {
            delta_part[row][mx_block] = delta_acc;
        }

        const uint8_t e8m0_val = mxfp4_bf16_abs_bits_to_e8m0<MODE>(block_amax_bits);
        const int j = row & 31;
        const int grp = row >> 5;
        const size_t sc_base =
            (static_cast<size_t>(bh) * tiles_s + s_tile) * 512
            + static_cast<size_t>(j) * 16 + grp * 4 + mx_block;
        row_sc[sc_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* row_out = reinterpret_cast<uint8_t*>(row_fp4)
            + (bh_row_base + (s_base + row)) * PACKED_D + d0 / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = vals[pack + 0];
            packed_bf[0].y = vals[pack + 1];
            packed_bf[1].x = vals[pack + 2];
            packed_bf[1].y = vals[pack + 3];
            packed_bf[2].x = vals[pack + 4];
            packed_bf[2].y = vals[pack + 5];
            packed_bf[3].x = vals[pack + 6];
            packed_bf[3].y = vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(row_out + pack / 2) = packed_fp4;
        }
    }

    if constexpr (WITH_DELTA) {
        __syncthreads();
        for (int row = threadIdx.x; row < CHUNK_DIM; row += THREAD_COUNT) {
            const float acc =
                delta_part[row][0] +
                delta_part[row][1] +
                delta_part[row][2] +
                delta_part[row][3];
            delta[(static_cast<size_t>(b) * H + h) * S + (s_base + row)] = acc;
        }
    }

    for (int combo = threadIdx.x; combo < CHUNK_DIM * SCALES_PER_CHUNK; combo += THREAD_COUNT) {
        const int d = combo / SCALES_PER_CHUNK;
        const int mx_block = combo - d * SCALES_PER_CHUNK;
        const int s0 = mx_block * MX_BLOCK;

        __nv_bfloat16 vals[MX_BLOCK];
        uint16_t block_amax_bits = 0;
        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const __nv_bfloat16 v = tile[s0 + i][d];
            vals[i] = v;
            const uint16_t abs_bits = mxfp4_bf16_abs_bits(v);
            block_amax_bits = abs_bits > block_amax_bits ? abs_bits : block_amax_bits;
        }

        const uint8_t e8m0_val = mxfp4_bf16_abs_bits_to_e8m0<MODE>(block_amax_bits);
        const int j = d & 31;
        const int grp = d >> 5;
        const size_t sc_base =
            (static_cast<size_t>(bh) * tiles_s + s_tile) * 512
            + static_cast<size_t>(j) * 16 + grp * 4 + mx_block;
        col_sc[sc_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* col_out = reinterpret_cast<uint8_t*>(col_fp4)
            + (bh_col_base + d) * (S / 2) + (s_base + s0) / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = vals[pack + 0];
            packed_bf[0].y = vals[pack + 1];
            packed_bf[1].x = vals[pack + 2];
            packed_bf[1].y = vals[pack + 3];
            packed_bf[2].x = vals[pack + 4];
            packed_bf[2].y = vals[pack + 5];
            packed_bf[3].x = vals[pack + 6];
            packed_bf[3].y = vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(col_out + pack / 2) = packed_fp4;
        }
    }
}

template<QuantMode MODE>
__global__ __launch_bounds__(THREADS)
void mxfp4_bshd128_row_delta_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ out,
    fp4e2m1x2* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    float* __restrict__ delta,
    const int B,
    const int S,
    const int H
) {
    constexpr int D = 128;
    constexpr int PACKED_D = D / 2;
    constexpr int THREAD_COUNT = THREADS;

    const int s_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh - b * H;
    if (b >= B) {
        return;
    }

    __shared__ float delta_part[CHUNK_DIM][SCALES_PER_CHUNK];

    const int s_base = s_tile * CHUNK_DIM;
    const int tiles_s = S / CHUNK_DIM;
    const size_t bh_row_base = static_cast<size_t>(bh) * S;

    for (int combo = threadIdx.x; combo < CHUNK_DIM * SCALES_PER_CHUNK; combo += THREAD_COUNT) {
        const int row = combo / SCALES_PER_CHUNK;
        const int mx_block = combo - row * SCALES_PER_CHUNK;
        const int d0 = mx_block * MX_BLOCK;
        const size_t base =
            (((static_cast<size_t>(b) * S + (s_base + row)) * H + h) * D + d0);

        __nv_bfloat16 vals[MX_BLOCK];
        uint16_t block_amax_bits = 0;
        float delta_acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const __nv_bfloat16 v = input[base + i];
            vals[i] = v;
            const uint16_t abs_bits = mxfp4_bf16_abs_bits(v);
            block_amax_bits = abs_bits > block_amax_bits ? abs_bits : block_amax_bits;
            delta_acc += __bfloat162float(out[base + i]) * __bfloat162float(v);
        }
        delta_part[row][mx_block] = delta_acc;

        const uint8_t e8m0_val = mxfp4_bf16_abs_bits_to_e8m0<MODE>(block_amax_bits);
        const int j = row & 31;
        const int grp = row >> 5;
        const size_t sc_base =
            (static_cast<size_t>(bh) * tiles_s + s_tile) * 512
            + static_cast<size_t>(j) * 16 + grp * 4 + mx_block;
        row_sc[sc_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* row_out = reinterpret_cast<uint8_t*>(row_fp4)
            + (bh_row_base + (s_base + row)) * PACKED_D + d0 / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = vals[pack + 0];
            packed_bf[0].y = vals[pack + 1];
            packed_bf[1].x = vals[pack + 2];
            packed_bf[1].y = vals[pack + 3];
            packed_bf[2].x = vals[pack + 4];
            packed_bf[2].y = vals[pack + 5];
            packed_bf[3].x = vals[pack + 6];
            packed_bf[3].y = vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(row_out + pack / 2) = packed_fp4;
        }
    }

    __syncthreads();
    for (int row = threadIdx.x; row < CHUNK_DIM; row += THREAD_COUNT) {
        const float acc =
            delta_part[row][0] +
            delta_part[row][1] +
            delta_part[row][2] +
            delta_part[row][3];
        delta[(static_cast<size_t>(b) * H + h) * S + (s_base + row)] = acc;
    }
}

template<QuantMode MODE>
__global__ __launch_bounds__(THREADS)
void mxfp4_bshd128_col_kernel(
    const __nv_bfloat16* __restrict__ input,
    fp4e2m1x2* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    const int B,
    const int S,
    const int H
) {
    constexpr int D = 128;
    constexpr int THREAD_COUNT = THREADS;

    const int s_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh - b * H;
    if (b >= B) {
        return;
    }

    __shared__ __nv_bfloat16 tile[CHUNK_DIM][CHUNK_DIM + 1];

    const int s_base = s_tile * CHUNK_DIM;
    for (int idx = threadIdx.x; idx < CHUNK_DIM * D; idx += THREAD_COUNT) {
        const int r = idx / D;
        const int d = idx - r * D;
        const size_t in_idx =
            (((static_cast<size_t>(b) * S + (s_base + r)) * H + h) * D + d);
        tile[r][d] = input[in_idx];
    }
    __syncthreads();

    const int tiles_s = S / CHUNK_DIM;
    const size_t bh_col_base = static_cast<size_t>(bh) * D;

    for (int combo = threadIdx.x; combo < CHUNK_DIM * SCALES_PER_CHUNK; combo += THREAD_COUNT) {
        const int d = combo / SCALES_PER_CHUNK;
        const int mx_block = combo - d * SCALES_PER_CHUNK;
        const int s0 = mx_block * MX_BLOCK;

        __nv_bfloat16 vals[MX_BLOCK];
        uint16_t block_amax_bits = 0;
        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const __nv_bfloat16 v = tile[s0 + i][d];
            vals[i] = v;
            const uint16_t abs_bits = mxfp4_bf16_abs_bits(v);
            block_amax_bits = abs_bits > block_amax_bits ? abs_bits : block_amax_bits;
        }

        const uint8_t e8m0_val = mxfp4_bf16_abs_bits_to_e8m0<MODE>(block_amax_bits);
        const int j = d & 31;
        const int grp = d >> 5;
        const size_t sc_base =
            (static_cast<size_t>(bh) * tiles_s + s_tile) * 512
            + static_cast<size_t>(j) * 16 + grp * 4 + mx_block;
        col_sc[sc_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* col_out = reinterpret_cast<uint8_t*>(col_fp4)
            + (bh_col_base + d) * (S / 2) + (s_base + s0) / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = vals[pack + 0];
            packed_bf[0].y = vals[pack + 1];
            packed_bf[1].x = vals[pack + 2];
            packed_bf[1].y = vals[pack + 3];
            packed_bf[2].x = vals[pack + 4];
            packed_bf[2].y = vals[pack + 5];
            packed_bf[3].x = vals[pack + 6];
            packed_bf[3].y = vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(col_out + pack / 2) = packed_fp4;
        }
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_row_and_col_impl(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 4);
    const int64_t B = input.size(0);
    const int64_t S = input.size(1);
    const int64_t H = input.size(2);
    const int64_t D = input.size(3);
    TORCH_CHECK(D == 128, "mxfp4_quantize_bshd128_row_and_col requires D=128");
    TORCH_CHECK(S % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(B > 0 && H > 0 && S > 0);

    auto device = input.device();
    auto row_fp4 = torch::empty({B, H, S, D / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({B, H, S / 128, D / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({B, H, D, S / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({B, H, D / 128, S / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(static_cast<unsigned int>(S / 128), static_cast<unsigned int>(B * H));
    mxfp4_bshd128_row_col_kernel<MODE, false><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        nullptr,
        reinterpret_cast<fp4e2m1x2*>(row_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        nullptr,
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(H)
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 bshd128 row+col quantize: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_row_and_col(torch::Tensor input, int mode) {
    switch (mode) {
        case 1: return mxfp4_quantize_bshd128_row_and_col_impl<QuantMode::ENCODE>(input);
        case 2: return mxfp4_quantize_bshd128_row_and_col_impl<QuantMode::DECODE>(input);
    default: return mxfp4_quantize_bshd128_row_and_col_impl<QuantMode::RTE>(input);
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_row_and_col_with_delta_impl(torch::Tensor input, torch::Tensor out) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(out.is_cuda() && out.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 4);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 4);
    TORCH_CHECK(out.sizes() == input.sizes(), "out must match input shape");
    const int64_t B = input.size(0);
    const int64_t S = input.size(1);
    const int64_t H = input.size(2);
    const int64_t D = input.size(3);
    TORCH_CHECK(D == 128, "mxfp4_quantize_bshd128_row_and_col_with_delta requires D=128");
    TORCH_CHECK(S % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(B > 0 && H > 0 && S > 0);

    auto device = input.device();
    auto row_fp4 = torch::empty({B, H, S, D / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({B, H, S / 128, D / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({B, H, D, S / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({B, H, D / 128, S / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto delta = torch::empty({B, H, S}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(static_cast<unsigned int>(S / 128), static_cast<unsigned int>(B * H));
    mxfp4_bshd128_row_col_kernel<MODE, true><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(row_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        reinterpret_cast<float*>(delta.data_ptr()),
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(H)
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 bshd128 row+col quantize with delta: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, delta);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_row_and_col_with_delta(torch::Tensor input, torch::Tensor out, int mode) {
    switch (mode) {
        case 1: return mxfp4_quantize_bshd128_row_and_col_with_delta_impl<QuantMode::ENCODE>(input, out);
        case 2: return mxfp4_quantize_bshd128_row_and_col_with_delta_impl<QuantMode::DECODE>(input, out);
        default: return mxfp4_quantize_bshd128_row_and_col_with_delta_impl<QuantMode::RTE>(input, out);
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_row_and_col_with_delta_out_impl(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor delta
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(out.is_cuda() && out.is_contiguous());
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(delta.is_cuda() && delta.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 4);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 4);
    TORCH_CHECK(out.sizes() == input.sizes(), "out must match input shape");
    const int64_t B = input.size(0);
    const int64_t S = input.size(1);
    const int64_t H = input.size(2);
    const int64_t D = input.size(3);
    TORCH_CHECK(D == 128, "mxfp4_quantize_bshd128_row_and_col_with_delta_out requires D=128");
    TORCH_CHECK(S % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(B > 0 && H > 0 && S > 0);
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(delta.scalar_type() == torch::kFloat32);
    TORCH_CHECK(row_fp4.sizes() == c10::IntArrayRef({B, H, S, D / 2}), "row_fp4 has wrong shape");
    TORCH_CHECK(row_sc.sizes() == c10::IntArrayRef({B, H, S / 128, D / 128, 32, 16}), "row_sc has wrong shape");
    TORCH_CHECK(col_fp4.sizes() == c10::IntArrayRef({B, H, D, S / 2}), "col_fp4 has wrong shape");
    TORCH_CHECK(col_sc.sizes() == c10::IntArrayRef({B, H, D / 128, S / 128, 32, 16}), "col_sc has wrong shape");
    TORCH_CHECK(delta.sizes() == c10::IntArrayRef({B, H, S}), "delta has wrong shape");
    TORCH_CHECK(row_fp4.device() == input.device(), "row_fp4 must be on input device");
    TORCH_CHECK(row_sc.device() == input.device(), "row_sc must be on input device");
    TORCH_CHECK(col_fp4.device() == input.device(), "col_fp4 must be on input device");
    TORCH_CHECK(col_sc.device() == input.device(), "col_sc must be on input device");
    TORCH_CHECK(delta.device() == input.device(), "delta must be on input device");

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(static_cast<unsigned int>(S / 128), static_cast<unsigned int>(B * H));
    mxfp4_bshd128_row_col_kernel<MODE, true><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(row_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        reinterpret_cast<float*>(delta.data_ptr()),
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(H)
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 bshd128 row+col quantize with delta out: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, delta);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_row_and_col_with_delta_out(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor delta,
    int mode
) {
    switch (mode) {
        case 1:
            return mxfp4_quantize_bshd128_row_and_col_with_delta_out_impl<QuantMode::ENCODE>(
                input, out, row_fp4, row_sc, col_fp4, col_sc, delta);
        case 2:
            return mxfp4_quantize_bshd128_row_and_col_with_delta_out_impl<QuantMode::DECODE>(
                input, out, row_fp4, row_sc, col_fp4, col_sc, delta);
        default:
            return mxfp4_quantize_bshd128_row_and_col_with_delta_out_impl<QuantMode::RTE>(
                input, out, row_fp4, row_sc, col_fp4, col_sc, delta);
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_row_with_delta_out_impl(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor delta
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(out.is_cuda() && out.is_contiguous());
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(delta.is_cuda() && delta.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 4);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 4);
    TORCH_CHECK(out.sizes() == input.sizes(), "out must match input shape");
    const int64_t B = input.size(0);
    const int64_t S = input.size(1);
    const int64_t H = input.size(2);
    const int64_t D = input.size(3);
    TORCH_CHECK(D == 128, "mxfp4_quantize_bshd128_row_with_delta_out requires D=128");
    TORCH_CHECK(S % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(B > 0 && H > 0 && S > 0);
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(delta.scalar_type() == torch::kFloat32);
    TORCH_CHECK(row_fp4.sizes() == c10::IntArrayRef({B, H, S, D / 2}), "row_fp4 has wrong shape");
    TORCH_CHECK(row_sc.sizes() == c10::IntArrayRef({B, H, S / 128, D / 128, 32, 16}), "row_sc has wrong shape");
    TORCH_CHECK(delta.sizes() == c10::IntArrayRef({B, H, S}), "delta has wrong shape");
    TORCH_CHECK(row_fp4.device() == input.device(), "row_fp4 must be on input device");
    TORCH_CHECK(row_sc.device() == input.device(), "row_sc must be on input device");
    TORCH_CHECK(delta.device() == input.device(), "delta must be on input device");

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(static_cast<unsigned int>(S / 128), static_cast<unsigned int>(B * H));
    mxfp4_bshd128_row_delta_kernel<MODE><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(row_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<float*>(delta.data_ptr()),
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(H)
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 bshd128 row quantize with delta out: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, delta);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_row_with_delta_out(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor delta,
    int mode
) {
    switch (mode) {
        case 1:
            return mxfp4_quantize_bshd128_row_with_delta_out_impl<QuantMode::ENCODE>(
                input, out, row_fp4, row_sc, delta);
        case 2:
            return mxfp4_quantize_bshd128_row_with_delta_out_impl<QuantMode::DECODE>(
                input, out, row_fp4, row_sc, delta);
        default:
            return mxfp4_quantize_bshd128_row_with_delta_out_impl<QuantMode::RTE>(
                input, out, row_fp4, row_sc, delta);
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_col_out_impl(
    torch::Tensor input,
    torch::Tensor col_fp4,
    torch::Tensor col_sc
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 4);
    const int64_t B = input.size(0);
    const int64_t S = input.size(1);
    const int64_t H = input.size(2);
    const int64_t D = input.size(3);
    TORCH_CHECK(D == 128, "mxfp4_quantize_bshd128_col_out requires D=128");
    TORCH_CHECK(S % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(B > 0 && H > 0 && S > 0);
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_fp4.sizes() == c10::IntArrayRef({B, H, D, S / 2}), "col_fp4 has wrong shape");
    TORCH_CHECK(col_sc.sizes() == c10::IntArrayRef({B, H, D / 128, S / 128, 32, 16}), "col_sc has wrong shape");
    TORCH_CHECK(col_fp4.device() == input.device(), "col_fp4 must be on input device");
    TORCH_CHECK(col_sc.device() == input.device(), "col_sc must be on input device");

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(static_cast<unsigned int>(S / 128), static_cast<unsigned int>(B * H));
    mxfp4_bshd128_col_kernel<MODE><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        static_cast<int>(B),
        static_cast<int>(S),
        static_cast<int>(H)
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 bshd128 col quantize out: ", cudaGetErrorString(err));
    return std::make_tuple(col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_bshd128_col_out(
    torch::Tensor input,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    switch (mode) {
        case 1:
            return mxfp4_quantize_bshd128_col_out_impl<QuantMode::ENCODE>(
                input, col_fp4, col_sc);
        case 2:
            return mxfp4_quantize_bshd128_col_out_impl<QuantMode::DECODE>(
                input, col_fp4, col_sc);
        default:
            return mxfp4_quantize_bshd128_col_out_impl<QuantMode::RTE>(
                input, col_fp4, col_sc);
    }
}

template<QuantMode MODE>
__global__ void softmax_tile_mxfp4_quant_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ row_max,
    fp4e2m1x2* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    float* __restrict__ row_sum,
    int rows,
    int cols,
    int padded_cols,
    float payload_scale
) {
    const int tile_k = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int tid = threadIdx.x;
    const int local_row_base = tid / SCALES_PER_CHUNK;
    const int mx_block = tid % SCALES_PER_CHUNK;
    const int row_stride = blockDim.x / SCALES_PER_CHUNK;
    const int ntk = padded_cols / CHUNK_DIM;
    const int packed_cols = padded_cols / 2;

    for (int local_row = local_row_base; local_row < CHUNK_DIM; local_row += row_stride) {
        const int row = tile_m * CHUNK_DIM + local_row;
        if (row >= rows) {
            continue;
        }

        const int col0 = tile_k * CHUNK_DIM + mx_block * MX_BLOCK;
        __nv_bfloat16 bf_vals[MX_BLOCK];
        float block_amax = 0.0f;
        float partial_sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const int col = col0 + i;
            float p = 0.0f;
            if (col < cols) {
                const float x = logits[row * cols + col];
                if (isfinite(x)) {
                    p = __expf(x - row_max[row]);
                }
            }
            partial_sum += p;
            const __nv_bfloat16 bf = __float2bfloat16(p * payload_scale);
            bf_vals[i] = bf;
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(bf)));
        }

        const int lane = tid & 31;
        float reduced_sum = partial_sum;
        reduced_sum += __shfl_down_sync(0xffffffff, reduced_sum, 2, 4);
        reduced_sum += __shfl_down_sync(0xffffffff, reduced_sum, 1, 4);
        if ((lane & 3) == 0) {
            atomicAdd(row_sum + row, reduced_sum);
        }

        const uint8_t e8m0_val = float_to_e8m0<MODE>(block_amax);
        const int j = local_row % 32;
        const int grp = local_row / 32;
        const int scale_base = (tile_m * ntk + tile_k) * 512 + j * 16 + grp * 4 + mx_block;
        scales_out[scale_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* out_u8 = reinterpret_cast<uint8_t*>(fp4_out);
        uint8_t* row_out = out_u8 + static_cast<size_t>(row) * packed_cols + col0 / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = bf_vals[pack + 0];
            packed_bf[0].y = bf_vals[pack + 1];
            packed_bf[1].x = bf_vals[pack + 2];
            packed_bf[1].y = bf_vals[pack + 3];
            packed_bf[2].x = bf_vals[pack + 4];
            packed_bf[2].y = bf_vals[pack + 5];
            packed_bf[3].x = bf_vals[pack + 6];
            packed_bf[3].y = bf_vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(row_out + pack / 2) = packed_fp4;
        }
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_impl(
    torch::Tensor logits,
    torch::Tensor row_max,
    int64_t padded_cols,
    double payload_scale
) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous());
    TORCH_CHECK(row_max.is_cuda() && row_max.is_contiguous());
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.dim() == 2);
    TORCH_CHECK(row_max.scalar_type() == torch::kFloat32 && row_max.dim() == 1);
    const int64_t rows = logits.size(0);
    const int64_t cols = logits.size(1);
    TORCH_CHECK(row_max.size(0) == rows);
    TORCH_CHECK(rows % CHUNK_DIM == 0, "rows must be a multiple of 128");
    TORCH_CHECK(padded_cols >= cols, "padded_cols must be >= cols");
    TORCH_CHECK(padded_cols % CHUNK_DIM == 0, "padded_cols must be a multiple of 128");
    TORCH_CHECK(payload_scale > 0.0, "payload_scale must be positive");

    auto device = logits.device();
    auto fp4_out = torch::empty({rows, padded_cols / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({rows / CHUNK_DIM, padded_cols / CHUNK_DIM, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto sum_out = torch::zeros({rows}, torch::dtype(torch::kFloat32).device(device));

    const dim3 grid(padded_cols / CHUNK_DIM, rows / CHUNK_DIM);
    const auto stream = at::cuda::getCurrentCUDAStream();
    softmax_tile_mxfp4_quant_kernel<MODE><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const float*>(logits.data_ptr()),
        reinterpret_cast<const float*>(row_max.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(fp4_out.data_ptr()),
        reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
        reinterpret_cast<float*>(sum_out.data_ptr()),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(padded_cols),
        static_cast<float>(payload_scale)
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 softmax tile quantize: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out, sum_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize(torch::Tensor logits, torch::Tensor row_max, int64_t padded_cols, double payload_scale, int mode) {
    switch (mode) {
        case 1: return mxfp4_softmax_tile_quantize_impl<QuantMode::ENCODE>(logits, row_max, padded_cols, payload_scale);
        case 2: return mxfp4_softmax_tile_quantize_impl<QuantMode::DECODE>(logits, row_max, padded_cols, payload_scale);
        default: return mxfp4_softmax_tile_quantize_impl<QuantMode::RTE>(logits, row_max, padded_cols, payload_scale);
    }
}

template<QuantMode MODE>
__global__ void softmax_tile_mxfp4_quant_transpose_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ lse,
    fp4e2m1x2* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int rows,
    int cols,
    int padded_cols,
    int tile_start,
    float payload_scale
) {
    const int tile_k = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int tid = threadIdx.x;
    const int local_out_row_base = tid / SCALES_PER_CHUNK;
    const int mx_block = tid % SCALES_PER_CHUNK;
    const int row_stride = blockDim.x / SCALES_PER_CHUNK;
    const int ntk = rows / CHUNK_DIM;
    const int packed_k = rows / 2;

    for (int local_out_row = local_out_row_base; local_out_row < CHUNK_DIM; local_out_row += row_stride) {
        const int out_row = tile_m * CHUNK_DIM + local_out_row;
        const int src_col = out_row;
        const int global_col = tile_start + src_col;
        const int src_row0 = tile_k * CHUNK_DIM + mx_block * MX_BLOCK;

        __nv_bfloat16 bf_vals[MX_BLOCK];
        float block_amax = 0.0f;

        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const int src_row = src_row0 + i;
            float p = 0.0f;
            if (src_col < cols && src_row < rows && global_col <= src_row) {
                const float x = logits[src_row * cols + src_col];
                const float row_lse = lse[src_row];
                if (isfinite(x) && isfinite(row_lse)) {
                    p = __expf(x - row_lse);
                }
            }
            const __nv_bfloat16 bf = __float2bfloat16(p * payload_scale);
            bf_vals[i] = bf;
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(bf)));
        }

        const uint8_t e8m0_val = float_to_e8m0<MODE>(block_amax);
        const int j = local_out_row % 32;
        const int grp = local_out_row / 32;
        const int scale_base = (tile_m * ntk + tile_k) * 512 + j * 16 + grp * 4 + mx_block;
        scales_out[scale_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* out_u8 = reinterpret_cast<uint8_t*>(fp4_out);
        uint8_t* row_out = out_u8 + static_cast<size_t>(out_row) * packed_k + src_row0 / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = bf_vals[pack + 0];
            packed_bf[0].y = bf_vals[pack + 1];
            packed_bf[1].x = bf_vals[pack + 2];
            packed_bf[1].y = bf_vals[pack + 3];
            packed_bf[2].x = bf_vals[pack + 4];
            packed_bf[2].y = bf_vals[pack + 5];
            packed_bf[3].x = bf_vals[pack + 6];
            packed_bf[3].y = bf_vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(row_out + pack / 2) = packed_fp4;
        }
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose_impl(
    torch::Tensor logits,
    torch::Tensor lse,
    int64_t tile_start,
    int64_t padded_cols,
    double payload_scale
) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous());
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous());
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.dim() == 2);
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.dim() == 1);
    const int64_t rows = logits.size(0);
    const int64_t cols = logits.size(1);
    TORCH_CHECK(lse.size(0) == rows);
    TORCH_CHECK(rows % CHUNK_DIM == 0, "rows must be a multiple of 128");
    TORCH_CHECK(padded_cols >= cols, "padded_cols must be >= cols");
    TORCH_CHECK(padded_cols % CHUNK_DIM == 0, "padded_cols must be a multiple of 128");
    TORCH_CHECK(payload_scale > 0.0, "payload_scale must be positive");
    TORCH_CHECK(tile_start >= 0, "tile_start must be non-negative");

    auto device = logits.device();
    auto fp4_out = torch::empty({padded_cols, rows / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({padded_cols / CHUNK_DIM, rows / CHUNK_DIM, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const dim3 grid(rows / CHUNK_DIM, padded_cols / CHUNK_DIM);
    const auto stream = at::cuda::getCurrentCUDAStream();
    softmax_tile_mxfp4_quant_transpose_kernel<MODE><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const float*>(logits.data_ptr()),
        reinterpret_cast<const float*>(lse.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(fp4_out.data_ptr()),
        reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(padded_cols),
        static_cast<int>(tile_start),
        static_cast<float>(payload_scale)
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 softmax tile transpose quantize: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose(torch::Tensor logits, torch::Tensor lse, int64_t tile_start, int64_t padded_cols, double payload_scale, int mode) {
    switch (mode) {
        case 1: return mxfp4_softmax_tile_quantize_transpose_impl<QuantMode::ENCODE>(logits, lse, tile_start, padded_cols, payload_scale);
        case 2: return mxfp4_softmax_tile_quantize_transpose_impl<QuantMode::DECODE>(logits, lse, tile_start, padded_cols, payload_scale);
        default: return mxfp4_softmax_tile_quantize_transpose_impl<QuantMode::RTE>(logits, lse, tile_start, padded_cols, payload_scale);
    }
}

template<QuantMode MODE>
__global__ void softmax_tile_mxfp4_quant_transpose_batched_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ lse,
    fp4e2m1x2* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int groups,
    int rows,
    int cols,
    int padded_cols,
    int tile_start,
    float payload_scale
) {
    const int tile_k = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int group = blockIdx.z;
    const int tid = threadIdx.x;
    const int local_out_row_base = tid / SCALES_PER_CHUNK;
    const int mx_block = tid % SCALES_PER_CHUNK;
    const int row_stride = blockDim.x / SCALES_PER_CHUNK;
    const int ntk = rows / CHUNK_DIM;
    const int packed_k = rows / 2;
    const float* group_logits = logits + static_cast<size_t>(group) * rows * cols;
    const float* group_lse = lse + static_cast<size_t>(group) * rows;
    fp4e2m1x2* group_fp4 = fp4_out + static_cast<size_t>(group) * padded_cols * packed_k;
    uint8_t* group_scales = scales_out + static_cast<size_t>(group) * (padded_cols / CHUNK_DIM) * ntk * 512;

    for (int local_out_row = local_out_row_base; local_out_row < CHUNK_DIM; local_out_row += row_stride) {
        const int out_row = tile_m * CHUNK_DIM + local_out_row;
        const int src_col = out_row;
        const int global_col = tile_start + src_col;
        const int src_row0 = tile_k * CHUNK_DIM + mx_block * MX_BLOCK;

        __nv_bfloat16 bf_vals[MX_BLOCK];
        float block_amax = 0.0f;

        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const int src_row = src_row0 + i;
            float p = 0.0f;
            if (src_col < cols && src_row < rows && global_col <= src_row) {
                const float x = group_logits[src_row * cols + src_col];
                const float row_lse = group_lse[src_row];
                if (isfinite(x) && isfinite(row_lse)) {
                    p = __expf(x - row_lse);
                }
            }
            const __nv_bfloat16 bf = __float2bfloat16(p * payload_scale);
            bf_vals[i] = bf;
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(bf)));
        }

        const uint8_t e8m0_val = float_to_e8m0<MODE>(block_amax);
        const int j = local_out_row % 32;
        const int grp = local_out_row / 32;
        const int scale_base = (tile_m * ntk + tile_k) * 512 + j * 16 + grp * 4 + mx_block;
        group_scales[scale_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* out_u8 = reinterpret_cast<uint8_t*>(group_fp4);
        uint8_t* row_out = out_u8 + static_cast<size_t>(out_row) * packed_k + src_row0 / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = bf_vals[pack + 0];
            packed_bf[0].y = bf_vals[pack + 1];
            packed_bf[1].x = bf_vals[pack + 2];
            packed_bf[1].y = bf_vals[pack + 3];
            packed_bf[2].x = bf_vals[pack + 4];
            packed_bf[2].y = bf_vals[pack + 5];
            packed_bf[3].x = bf_vals[pack + 6];
            packed_bf[3].y = bf_vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(row_out + pack / 2) = packed_fp4;
        }
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose_batched_impl(
    torch::Tensor logits,
    torch::Tensor lse,
    int64_t tile_start,
    int64_t padded_cols,
    double payload_scale
) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous());
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous());
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.dim() == 3);
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.dim() == 2);
    const int64_t groups = logits.size(0);
    const int64_t rows = logits.size(1);
    const int64_t cols = logits.size(2);
    TORCH_CHECK(lse.size(0) == groups && lse.size(1) == rows);
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(rows % CHUNK_DIM == 0, "rows must be a multiple of 128");
    TORCH_CHECK(padded_cols >= cols, "padded_cols must be >= cols");
    TORCH_CHECK(padded_cols % CHUNK_DIM == 0, "padded_cols must be a multiple of 128");
    TORCH_CHECK(payload_scale > 0.0, "payload_scale must be positive");
    TORCH_CHECK(tile_start >= 0, "tile_start must be non-negative");

    auto device = logits.device();
    auto fp4_out = torch::empty({groups, padded_cols, rows / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({groups, padded_cols / CHUNK_DIM, rows / CHUNK_DIM, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const dim3 grid(rows / CHUNK_DIM, padded_cols / CHUNK_DIM, groups);
    const auto stream = at::cuda::getCurrentCUDAStream();
    softmax_tile_mxfp4_quant_transpose_batched_kernel<MODE><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const float*>(logits.data_ptr()),
        reinterpret_cast<const float*>(lse.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(fp4_out.data_ptr()),
        reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
        static_cast<int>(groups),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(padded_cols),
        static_cast<int>(tile_start),
        static_cast<float>(payload_scale)
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 batched softmax tile transpose quantize: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose_batched(torch::Tensor logits, torch::Tensor lse, int64_t tile_start, int64_t padded_cols, double payload_scale, int mode) {
    switch (mode) {
        case 1: return mxfp4_softmax_tile_quantize_transpose_batched_impl<QuantMode::ENCODE>(logits, lse, tile_start, padded_cols, payload_scale);
        case 2: return mxfp4_softmax_tile_quantize_transpose_batched_impl<QuantMode::DECODE>(logits, lse, tile_start, padded_cols, payload_scale);
        default: return mxfp4_softmax_tile_quantize_transpose_batched_impl<QuantMode::RTE>(logits, lse, tile_start, padded_cols, payload_scale);
    }
}

template<QuantMode MODE, bool HAS_ROW_OFFSET>
__global__ void softmax_tile_mxfp4_quant_transpose_batched_bf16_scaled_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    fp4e2m1x2* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int groups,
    int rows,
    int cols,
    int padded_cols,
    int row_start,
    int tile_start,
    float logits_scale,
    float payload_scale
) {
    const int tile_k = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int group = blockIdx.z;
    const int tid = threadIdx.x;
    const int local_out_row_base = tid / SCALES_PER_CHUNK;
    const int mx_block = tid % SCALES_PER_CHUNK;
    const int row_stride = blockDim.x / SCALES_PER_CHUNK;
    const int ntk = rows / CHUNK_DIM;
    const int packed_k = rows / 2;
    const __nv_bfloat16* group_logits = logits + static_cast<size_t>(group) * rows * cols;
    const float* group_lse = lse + static_cast<size_t>(group) * rows;
    fp4e2m1x2* group_fp4 = fp4_out + static_cast<size_t>(group) * padded_cols * packed_k;
    uint8_t* group_scales = scales_out + static_cast<size_t>(group) * (padded_cols / CHUNK_DIM) * ntk * 512;
    const int tile_col0 = tile_m * CHUNK_DIM;
    const int global_col0 = tile_start + tile_col0;
    const int row_base = HAS_ROW_OFFSET ? row_start : 0;
    const int row_tile_first = row_base + tile_k * CHUNK_DIM;
    const int row_tile_last = row_base + (tile_k + 1) * CHUNK_DIM - 1;

    if (tile_col0 >= cols || global_col0 > row_tile_last) {
        uint8_t* out_u8 = reinterpret_cast<uint8_t*>(group_fp4);
        constexpr int words_per_tile_row = CHUNK_DIM / 8;
        for (int word_idx = tid; word_idx < CHUNK_DIM * words_per_tile_row; word_idx += blockDim.x) {
            const int out_row = tile_col0 + word_idx / words_per_tile_row;
            const int word_in_row = word_idx % words_per_tile_row;
            uint8_t* row_out = out_u8 + static_cast<size_t>(out_row) * packed_k + tile_k * (CHUNK_DIM / 2);
            *reinterpret_cast<uint32_t*>(row_out + word_in_row * sizeof(uint32_t)) = 0x77777777u;
        }
        const int scale_base = (tile_m * ntk + tile_k) * 512;
        for (int idx = tid; idx < 512; idx += blockDim.x) {
            group_scales[scale_base + idx] = 0;
        }
        return;
    }

    const bool full_valid_tile =
        tile_col0 + CHUNK_DIM <= cols &&
        (tile_k + 1) * CHUNK_DIM <= rows &&
        global_col0 + CHUNK_DIM - 1 <= row_tile_first;
    if (full_valid_tile) {
        for (int local_out_row = local_out_row_base; local_out_row < CHUNK_DIM; local_out_row += row_stride) {
            const int out_row = tile_m * CHUNK_DIM + local_out_row;
            const int src_col = out_row;
            const int src_row0 = tile_k * CHUNK_DIM + mx_block * MX_BLOCK;

            __nv_bfloat16 bf_vals[MX_BLOCK];
            float block_amax = 0.0f;

            #pragma unroll
            for (int i = 0; i < MX_BLOCK; ++i) {
                const int src_row = src_row0 + i;
                float p = 0.0f;
                const float x = __bfloat162float(group_logits[src_row * cols + src_col]) * logits_scale;
                const float row_lse = group_lse[src_row];
                if (isfinite(x) && isfinite(row_lse)) {
                    p = __expf(x - row_lse);
                }
                const __nv_bfloat16 bf = __float2bfloat16(p * payload_scale);
                bf_vals[i] = bf;
                block_amax = fmaxf(block_amax, fabsf(__bfloat162float(bf)));
            }

            const uint8_t e8m0_val = float_to_e8m0<MODE>(block_amax);
            const int j = local_out_row % 32;
            const int grp = local_out_row / 32;
            const int scale_base = (tile_m * ntk + tile_k) * 512 + j * 16 + grp * 4 + mx_block;
            group_scales[scale_base] = e8m0_val;

            const float coeff = 6.0f * exp2f_rcp(e8m0_val);
            uint8_t* out_u8 = reinterpret_cast<uint8_t*>(group_fp4);
            uint8_t* row_out = out_u8 + static_cast<size_t>(out_row) * packed_k + src_row0 / 2;

            #pragma unroll
            for (int pack = 0; pack < MX_BLOCK; pack += 8) {
                IType2 packed_bf[4];
                packed_bf[0].x = bf_vals[pack + 0];
                packed_bf[0].y = bf_vals[pack + 1];
                packed_bf[1].x = bf_vals[pack + 2];
                packed_bf[1].y = bf_vals[pack + 3];
                packed_bf[2].x = bf_vals[pack + 4];
                packed_bf[2].y = bf_vals[pack + 5];
                packed_bf[3].x = bf_vals[pack + 6];
                packed_bf[3].y = bf_vals[pack + 7];
                const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
                const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
                const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
                *reinterpret_cast<uint32_t*>(row_out + pack / 2) = packed_fp4;
            }
        }
        return;
    }

    for (int local_out_row = local_out_row_base; local_out_row < CHUNK_DIM; local_out_row += row_stride) {
        const int out_row = tile_m * CHUNK_DIM + local_out_row;
        const int src_col = out_row;
        const int global_col = tile_start + src_col;
        const int src_row0 = tile_k * CHUNK_DIM + mx_block * MX_BLOCK;

        __nv_bfloat16 bf_vals[MX_BLOCK];
        float block_amax = 0.0f;

        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const int src_row = src_row0 + i;
            float p = 0.0f;
            const int global_row = row_base + src_row;
            if (src_col < cols && src_row < rows && global_col <= global_row) {
                const float x = __bfloat162float(group_logits[src_row * cols + src_col]) * logits_scale;
                const float row_lse = group_lse[src_row];
                if (isfinite(x) && isfinite(row_lse)) {
                    p = __expf(x - row_lse);
                }
            }
            const __nv_bfloat16 bf = __float2bfloat16(p * payload_scale);
            bf_vals[i] = bf;
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(bf)));
        }

        const uint8_t e8m0_val = float_to_e8m0<MODE>(block_amax);
        const int j = local_out_row % 32;
        const int grp = local_out_row / 32;
        const int scale_base = (tile_m * ntk + tile_k) * 512 + j * 16 + grp * 4 + mx_block;
        group_scales[scale_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* out_u8 = reinterpret_cast<uint8_t*>(group_fp4);
        uint8_t* row_out = out_u8 + static_cast<size_t>(out_row) * packed_k + src_row0 / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = bf_vals[pack + 0];
            packed_bf[0].y = bf_vals[pack + 1];
            packed_bf[1].x = bf_vals[pack + 2];
            packed_bf[1].y = bf_vals[pack + 3];
            packed_bf[2].x = bf_vals[pack + 4];
            packed_bf[2].y = bf_vals[pack + 5];
            packed_bf[3].x = bf_vals[pack + 6];
            packed_bf[3].y = bf_vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(row_out + pack / 2) = packed_fp4;
        }
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_impl(
    torch::Tensor logits,
    torch::Tensor lse,
    int64_t tile_start,
    int64_t padded_cols,
    int64_t row_start,
    double logits_scale,
    double payload_scale
) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous());
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous());
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 3);
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.dim() == 2);
    const int64_t groups = logits.size(0);
    const int64_t rows = logits.size(1);
    const int64_t cols = logits.size(2);
    TORCH_CHECK(lse.size(0) == groups && lse.size(1) == rows);
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(rows % CHUNK_DIM == 0, "rows must be a multiple of 128");
    TORCH_CHECK(padded_cols >= cols, "padded_cols must be >= cols");
    TORCH_CHECK(padded_cols % CHUNK_DIM == 0, "padded_cols must be a multiple of 128");
    TORCH_CHECK(row_start >= 0, "row_start must be non-negative");
    TORCH_CHECK(row_start % CHUNK_DIM == 0, "row_start must be a multiple of 128");
    TORCH_CHECK(logits_scale > 0.0, "logits_scale must be positive");
    TORCH_CHECK(payload_scale > 0.0, "payload_scale must be positive");
    TORCH_CHECK(tile_start >= 0, "tile_start must be non-negative");

    auto device = logits.device();
    auto fp4_out = torch::empty({groups, padded_cols, rows / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({groups, padded_cols / CHUNK_DIM, rows / CHUNK_DIM, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const dim3 grid(rows / CHUNK_DIM, padded_cols / CHUNK_DIM, groups);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (row_start == 0) {
        softmax_tile_mxfp4_quant_transpose_batched_bf16_scaled_kernel<MODE, false><<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            reinterpret_cast<const float*>(lse.data_ptr()),
            reinterpret_cast<fp4e2m1x2*>(fp4_out.data_ptr()),
            reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
            static_cast<int>(groups),
            static_cast<int>(rows),
            static_cast<int>(cols),
            static_cast<int>(padded_cols),
            0,
            static_cast<int>(tile_start),
            static_cast<float>(logits_scale),
            static_cast<float>(payload_scale)
        );
    } else {
        softmax_tile_mxfp4_quant_transpose_batched_bf16_scaled_kernel<MODE, true><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        reinterpret_cast<const float*>(lse.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(fp4_out.data_ptr()),
        reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
        static_cast<int>(groups),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(padded_cols),
        static_cast<int>(row_start),
        static_cast<int>(tile_start),
        static_cast<float>(logits_scale),
        static_cast<float>(payload_scale)
        );
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 bf16 scaled batched softmax tile transpose quantize: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled(
    torch::Tensor logits,
    torch::Tensor lse,
    int64_t tile_start,
    int64_t padded_cols,
    double logits_scale,
    double payload_scale,
    int mode
) {
    switch (mode) {
        case 1: return mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_impl<QuantMode::ENCODE>(logits, lse, tile_start, padded_cols, 0, logits_scale, payload_scale);
        case 2: return mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_impl<QuantMode::DECODE>(logits, lse, tile_start, padded_cols, 0, logits_scale, payload_scale);
        default: return mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_impl<QuantMode::RTE>(logits, lse, tile_start, padded_cols, 0, logits_scale, payload_scale);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_offset(
    torch::Tensor logits,
    torch::Tensor lse,
    int64_t row_start,
    int64_t tile_start,
    int64_t padded_cols,
    double logits_scale,
    double payload_scale,
    int mode
) {
    switch (mode) {
        case 1: return mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_impl<QuantMode::ENCODE>(logits, lse, tile_start, padded_cols, row_start, logits_scale, payload_scale);
        case 2: return mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_impl<QuantMode::DECODE>(logits, lse, tile_start, padded_cols, row_start, logits_scale, payload_scale);
        default: return mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_impl<QuantMode::RTE>(logits, lse, tile_start, padded_cols, row_start, logits_scale, payload_scale);
    }
}

static constexpr int SOFTMAX_STATS_THREADS = 256;

__global__ void causal_softmax_tile_stats_kernel(
    const float* __restrict__ logits,
    float* __restrict__ tile_max_out,
    float* __restrict__ tile_sum_out,
    int rows,
    int cols,
    int tile_start
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float scratch[SOFTMAX_STATS_THREADS];

    float local_max = -CUDART_INF_F;
    for (int col = tid; col < cols; col += blockDim.x) {
        const int global_col = tile_start + col;
        if (global_col <= row) {
            const float x = logits[row * cols + col];
            if (isfinite(x)) {
                local_max = fmaxf(local_max, x);
            }
        }
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + offset]);
        }
        __syncthreads();
    }
    const float row_max = scratch[0];

    float local_sum = 0.0f;
    if (isfinite(row_max)) {
        for (int col = tid; col < cols; col += blockDim.x) {
            const int global_col = tile_start + col;
            if (global_col <= row) {
                const float x = logits[row * cols + col];
                if (isfinite(x)) {
                    local_sum += __expf(x - row_max);
                }
            }
        }
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        tile_max_out[row] = row_max;
        tile_sum_out[row] = scratch[0];
    }
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_stats(torch::Tensor logits, int64_t tile_start) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous());
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.dim() == 2);
    const int64_t rows = logits.size(0);
    const int64_t cols = logits.size(1);
    TORCH_CHECK(rows > 0 && cols > 0, "logits must be non-empty");
    TORCH_CHECK(rows % CHUNK_DIM == 0, "rows must be a multiple of 128");
    TORCH_CHECK(tile_start >= 0, "tile_start must be non-negative");

    auto device = logits.device();
    auto tile_max = torch::empty({rows}, torch::dtype(torch::kFloat32).device(device));
    auto tile_sum = torch::empty({rows}, torch::dtype(torch::kFloat32).device(device));
    const auto stream = at::cuda::getCurrentCUDAStream();
    causal_softmax_tile_stats_kernel<<<rows, SOFTMAX_STATS_THREADS, 0, stream>>>(
        reinterpret_cast<const float*>(logits.data_ptr()),
        reinterpret_cast<float*>(tile_max.data_ptr()),
        reinterpret_cast<float*>(tile_sum.data_ptr()),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(tile_start)
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 softmax tile stats: ", cudaGetErrorString(err));
    return std::make_tuple(tile_max, tile_sum);
}

template<QuantMode MODE>
__global__ void softmax_tile_mxfp4_quant_transpose_stats_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ row_max,
    const float* __restrict__ row_sum,
    fp4e2m1x2* __restrict__ fp4_out,
    uint8_t* __restrict__ scales_out,
    int rows,
    int cols,
    int padded_cols,
    int tile_start,
    float payload_scale
) {
    const int tile_k = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int tid = threadIdx.x;
    const int local_out_row_base = tid / SCALES_PER_CHUNK;
    const int mx_block = tid % SCALES_PER_CHUNK;
    const int row_stride = blockDim.x / SCALES_PER_CHUNK;
    const int ntk = rows / CHUNK_DIM;
    const int packed_k = rows / 2;

    for (int local_out_row = local_out_row_base; local_out_row < CHUNK_DIM; local_out_row += row_stride) {
        const int out_row = tile_m * CHUNK_DIM + local_out_row;
        const int src_col = out_row;
        const int global_col = tile_start + src_col;
        const int src_row0 = tile_k * CHUNK_DIM + mx_block * MX_BLOCK;

        __nv_bfloat16 bf_vals[MX_BLOCK];
        float block_amax = 0.0f;

        #pragma unroll
        for (int i = 0; i < MX_BLOCK; ++i) {
            const int src_row = src_row0 + i;
            float p = 0.0f;
            if (src_col < cols && src_row < rows && global_col <= src_row) {
                const float x = logits[src_row * cols + src_col];
                const float m = row_max[src_row];
                const float s = row_sum[src_row];
                if (isfinite(x) && isfinite(m) && s > 0.0f) {
                    p = __expf(x - m) / s;
                }
            }
            const __nv_bfloat16 bf = __float2bfloat16(p * payload_scale);
            bf_vals[i] = bf;
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(bf)));
        }

        const uint8_t e8m0_val = float_to_e8m0<MODE>(block_amax);
        const int j = local_out_row % 32;
        const int grp = local_out_row / 32;
        const int scale_base = (tile_m * ntk + tile_k) * 512 + j * 16 + grp * 4 + mx_block;
        scales_out[scale_base] = e8m0_val;

        const float coeff = 6.0f * exp2f_rcp(e8m0_val);
        uint8_t* out_u8 = reinterpret_cast<uint8_t*>(fp4_out);
        uint8_t* row_out = out_u8 + static_cast<size_t>(out_row) * packed_k + src_row0 / 2;

        #pragma unroll
        for (int pack = 0; pack < MX_BLOCK; pack += 8) {
            IType2 packed_bf[4];
            packed_bf[0].x = bf_vals[pack + 0];
            packed_bf[0].y = bf_vals[pack + 1];
            packed_bf[1].x = bf_vals[pack + 2];
            packed_bf[1].y = bf_vals[pack + 3];
            packed_bf[2].x = bf_vals[pack + 4];
            packed_bf[2].y = bf_vals[pack + 5];
            packed_bf[3].x = bf_vals[pack + 6];
            packed_bf[3].y = bf_vals[pack + 7];
            const uint64_t e03 = *reinterpret_cast<uint64_t*>(&packed_bf[0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t*>(&packed_bf[2]);
            const uint32_t packed_fp4 = mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            *reinterpret_cast<uint32_t*>(row_out + pack / 2) = packed_fp4;
        }
    }
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose_from_stats_impl(
    torch::Tensor logits,
    torch::Tensor row_max,
    torch::Tensor row_sum,
    int64_t tile_start,
    int64_t padded_cols,
    double payload_scale
) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous());
    TORCH_CHECK(row_max.is_cuda() && row_max.is_contiguous());
    TORCH_CHECK(row_sum.is_cuda() && row_sum.is_contiguous());
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.dim() == 2);
    TORCH_CHECK(row_max.scalar_type() == torch::kFloat32 && row_max.dim() == 1);
    TORCH_CHECK(row_sum.scalar_type() == torch::kFloat32 && row_sum.dim() == 1);
    const int64_t rows = logits.size(0);
    const int64_t cols = logits.size(1);
    TORCH_CHECK(row_max.size(0) == rows && row_sum.size(0) == rows);
    TORCH_CHECK(rows % CHUNK_DIM == 0, "rows must be a multiple of 128");
    TORCH_CHECK(padded_cols >= cols, "padded_cols must be >= cols");
    TORCH_CHECK(padded_cols % CHUNK_DIM == 0, "padded_cols must be a multiple of 128");
    TORCH_CHECK(payload_scale > 0.0, "payload_scale must be positive");
    TORCH_CHECK(tile_start >= 0, "tile_start must be non-negative");

    auto device = logits.device();
    auto fp4_out = torch::empty({padded_cols, rows / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({padded_cols / CHUNK_DIM, rows / CHUNK_DIM, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const dim3 grid(rows / CHUNK_DIM, padded_cols / CHUNK_DIM);
    const auto stream = at::cuda::getCurrentCUDAStream();
    softmax_tile_mxfp4_quant_transpose_stats_kernel<MODE><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const float*>(logits.data_ptr()),
        reinterpret_cast<const float*>(row_max.data_ptr()),
        reinterpret_cast<const float*>(row_sum.data_ptr()),
        reinterpret_cast<fp4e2m1x2*>(fp4_out.data_ptr()),
        reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(padded_cols),
        static_cast<int>(tile_start),
        static_cast<float>(payload_scale)
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 softmax tile transpose stats quantize: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_softmax_tile_quantize_transpose_from_stats(torch::Tensor logits, torch::Tensor row_max, torch::Tensor row_sum, int64_t tile_start, int64_t padded_cols, double payload_scale, int mode) {
    switch (mode) {
        case 1: return mxfp4_softmax_tile_quantize_transpose_from_stats_impl<QuantMode::ENCODE>(logits, row_max, row_sum, tile_start, padded_cols, payload_scale);
        case 2: return mxfp4_softmax_tile_quantize_transpose_from_stats_impl<QuantMode::DECODE>(logits, row_max, row_sum, tile_start, padded_cols, payload_scale);
        default: return mxfp4_softmax_tile_quantize_transpose_from_stats_impl<QuantMode::RTE>(logits, row_max, row_sum, tile_start, padded_cols, payload_scale);
    }
}


// ═══════════════════════════════════════════════════════════════════
// PyBind11
// ═══════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_quantize_for_gemm", &mxfp4_quantize_for_gemm,
          "MXFP4 v3 quantize (pipelined) → (fp4, scales). mode: 0=RTE, 1=ENCODE, 2=DECODE",
          py::arg("input"), py::arg("mode") = 0);
    m.def("mxfp4_group_quantize_dim0", &mxfp4_group_quantize_dim0,
          "MXFP4 v3 group quantize contiguous input → list[(fp4, scales)]");
    m.def("mxfp4_multi_quantize", &mxfp4_multi_quantize,
          "MXFP4 v3 multi-tensor quantize → list[(fp4, scales)]");
    m.def("mxfp4_quantize_row_and_col", &mxfp4_quantize_row_and_col,
          "MXFP4 v3 quantize row + transpose + col quantize → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("input"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_bshd128_row_and_col", &mxfp4_quantize_bshd128_row_and_col,
          "MXFP4 v3 direct quantize contiguous BF16 [B,S,H,128] row + transposed col → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("input"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_bshd128_row_and_col_with_delta", &mxfp4_quantize_bshd128_row_and_col_with_delta,
          "MXFP4 v3 direct quantize BF16 dOut [B,S,H,128] row + transposed col and compute delta=sum(out*dOut) → (row_fp4, row_sc, col_fp4, col_sc, delta)",
          py::arg("input"), py::arg("out"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_bshd128_row_and_col_with_delta_out", &mxfp4_quantize_bshd128_row_and_col_with_delta_out,
          "MXFP4 v3 direct quantize BF16 dOut [B,S,H,128] row + transposed col and delta into preallocated outputs",
          py::arg("input"), py::arg("out"), py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("col_fp4"), py::arg("col_sc"), py::arg("delta"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_bshd128_row_with_delta_out", &mxfp4_quantize_bshd128_row_with_delta_out,
          "MXFP4 v3 direct quantize BF16 dOut [B,S,H,128] row and delta into preallocated outputs",
          py::arg("input"), py::arg("out"), py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("delta"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_bshd128_col_out", &mxfp4_quantize_bshd128_col_out,
          "MXFP4 v3 direct quantize BF16 dOut [B,S,H,128] transposed col into preallocated outputs",
          py::arg("input"), py::arg("col_fp4"), py::arg("col_sc"), py::arg("mode") = 0);
    m.def("mxfp4_softmax_tile_quantize", &mxfp4_softmax_tile_quantize,
          "Fused exp(row logits - row_max), row-sum, scale and MXFP4 quantize → (fp4, scales, row_sum)",
          py::arg("logits"), py::arg("row_max"), py::arg("padded_cols"),
          py::arg("payload_scale") = 448.0, py::arg("mode") = 0);
    m.def("mxfp4_softmax_tile_quantize_transpose", &mxfp4_softmax_tile_quantize_transpose,
          "Fused causal softmax(logits - lse), transpose, scale and MXFP4 quantize → (fp4, scales)",
          py::arg("logits"), py::arg("lse"), py::arg("tile_start"), py::arg("padded_cols"),
          py::arg("payload_scale") = 448.0, py::arg("mode") = 0);
    m.def("mxfp4_softmax_tile_quantize_transpose_batched", &mxfp4_softmax_tile_quantize_transpose_batched,
          "Batched fused causal softmax(logits - lse), transpose, scale and MXFP4 quantize → (fp4, scales)",
          py::arg("logits"), py::arg("lse"), py::arg("tile_start"), py::arg("padded_cols"),
          py::arg("payload_scale") = 448.0, py::arg("mode") = 0);
    m.def("mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled",
          &mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled,
          "Batched fused causal softmax(float(bf16 logits) * logits_scale - lse), transpose, scale and MXFP4 quantize → (fp4, scales)",
          py::arg("logits"), py::arg("lse"), py::arg("tile_start"), py::arg("padded_cols"),
          py::arg("logits_scale"), py::arg("payload_scale") = 448.0, py::arg("mode") = 0);
    m.def("mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_offset",
          &mxfp4_softmax_tile_quantize_transpose_batched_bf16_scaled_offset,
          "Batched fused causal softmax for a query-row suffix: float(bf16 logits) * logits_scale - lse, transpose, scale and MXFP4 quantize → (fp4, scales)",
          py::arg("logits"), py::arg("lse"), py::arg("row_start"), py::arg("tile_start"), py::arg("padded_cols"),
          py::arg("logits_scale"), py::arg("payload_scale") = 448.0, py::arg("mode") = 0);
    m.def("mxfp4_softmax_tile_stats", &mxfp4_softmax_tile_stats,
          "Fused causal tile softmax row stats → (tile_max, tile_sum)",
          py::arg("logits"), py::arg("tile_start"));
    m.def("mxfp4_softmax_tile_quantize_transpose_from_stats", &mxfp4_softmax_tile_quantize_transpose_from_stats,
          "Fused causal softmax from row_max/row_sum, transpose, scale and MXFP4 quantize → (fp4, scales)",
          py::arg("logits"), py::arg("row_max"), py::arg("row_sum"), py::arg("tile_start"), py::arg("padded_cols"),
          py::arg("payload_scale") = 448.0, py::arg("mode") = 0);
}

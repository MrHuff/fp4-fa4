/*
 * MXFP4 Single-Pass Quantisation Dispatch Layer
 *
 * MXFP4 uses E8M0 scales (power-of-2, block-32), which means:
 *   - No global amax needed (each block's scale is independent)
 *   - True single-pass: TMA load → compute E8M0 → quantise → store
 *   - No grid barrier, no multi-kernel pipeline
 *
 * Supports:
 *   1. mxfp4_quantize_for_gemm: single-pass quantise + optional transpose
 *   2. mxfp4_group_quantize_dim0: grouped quantise along rows (dim=0)
 *   3. mxfp4_group_quantize_dim1: grouped quantise along columns (dim=1)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <dlfcn.h>

// ═══════════════════════════════════════════════════════════════════
// Configuration constants
// ═══════════════════════════════════════════════════════════════════

static constexpr int CHUNK_DIM = 128;           // Process 128×128 chunks
static constexpr int SCALE_BLOCK = 32;          // E8M0 scale per 32 elements
static constexpr int MX_THREADS  = 128;         // Threads per CTA
static constexpr int FP4_MAX_VAL = 6;           // Max representable FP4 value

// ═══════════════════════════════════════════════════════════════════
// TMA tensor map creation helper
// ═══════════════════════════════════════════════════════════════════
static void create_tma_2d(
    CUtensorMap &map, void *ptr,
    uint64_t globalY, uint64_t globalX,
    uint32_t shmemY, uint32_t shmemX,
    uint64_t strideX, size_t type_num_bits
) {
    typedef CUresult (*cuTensorMapEncodeTiled_t)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
        const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
        const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

    static cuTensorMapEncodeTiled_t fn = nullptr;
    if (!fn) {
        void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
        TORCH_CHECK(handle != nullptr, "Failed to open libcuda.so.1");
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(dlsym(handle, "cuTensorMapEncodeTiled"));
        TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled not found");
    }

    CUtensorMapDataType dataType;
    uint64_t globalDims[2] = {globalX, globalY};
    uint32_t boxDims[2] = {shmemX, shmemY};
    uint64_t globalStrides[1] = {(strideX * type_num_bits) / 8};
    uint32_t elementStrides[2] = {1, 1};

    if (type_num_bits == 16) dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (type_num_bits == 8) dataType = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else if (type_num_bits == 4) dataType = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    else TORCH_CHECK(false, "Unsupported type_num_bits: ", type_num_bits);

    auto result = fn(&map, dataType, 2, ptr,
        globalDims, globalStrides, boxDims, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}


// ═══════════════════════════════════════════════════════════════════
// MXFP4 single-pass quantise kernel
//
// Each CTA processes one 128×128 chunk:
//   - TMA load 128×128 bf16 into SMEM
//   - For each block of 32 elements:
//       * Compute amax
//       * E8M0 = round(log2(amax)) + 127
//       * scale_inv = 6.0 / 2^(e8m0 - 127)
//       * Quantise each element: clamp(x * scale_inv, -6, 6) → FP4
//   - Store quantised FP4 and E8M0 scales via TMA or global stores
//
// SMEM layout: 128×128 bf16 = 32KB input
//              128×64 fp4x2 = 4KB output
//              128×4 uint8  = 512B scales (4 scale blocks per row)
// ═══════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(MX_THREADS)
mxfp4_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    __nv_fp4x2_e2m1* __restrict__ output_fp4,     // [M, K/2]
    uint8_t* __restrict__ output_scales,            // [M/128, K/128, 32, 16] swizzled
    const int64_t M, const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // This CTA's chunk coordinates
    const int chunk_col = blockIdx.x;  // K dimension
    const int chunk_row = blockIdx.y;  // M dimension
    const int gY = chunk_row * CHUNK_DIM;
    const int gX = chunk_col * CHUNK_DIM;

    // Shared memory for input tile
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // TMA load the 128×128 bf16 tile
    __shared__ uint64_t mbar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(1));
        asm volatile("fence.proxy.async.shared::cta;");
        // Expect 128×128×2B = 32768 bytes
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(CHUNK_DIM * CHUNK_DIM * 2));
        uint64_t tmap_addr = reinterpret_cast<uint64_t>(&tensor_map_input);
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_input);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(tmap_addr), "r"(gX), "r"(gY), "r"((uint32_t)__cvta_generic_to_shared(&mbar))
            : "memory");
    }
    __syncthreads();

    // Wait for TMA load
    {
        uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "WAIT_LOOP:\n"
            "    mbarrier.try_wait.parity.acquire.cta.shared.b64 p, [%0], %1;\n"
            "    @!p bra WAIT_LOOP;\n"
            "}\n"
            :: "r"(mbar_addr), "r"(0));
    }

    // Each thread processes 2 rows × full 128 columns = 256 elements
    // 128 threads × 2 rows = 256 rows, but chunk is 128 rows → each thread does 1 row
    // Actually: 128 threads process 128 rows (1 row per thread), each row has 128 elements = 4 blocks of 32
    const int tid = threadIdx.x;

    // Process 1 row per thread (128 threads for 128 rows)
    if (tid < CHUNK_DIM) {
        const int row = tid;
        constexpr int NUM_BLOCKS = CHUNK_DIM / SCALE_BLOCK; // 4 blocks per row

        uint8_t e8m0_vals[NUM_BLOCKS];

        #pragma unroll
        for (int b = 0; b < NUM_BLOCKS; b++) {
            const int col_start = b * SCALE_BLOCK;

            // Compute amax over 32 elements
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < SCALE_BLOCK; j++) {
                float val = __bfloat162float(smem_input[row * CHUNK_DIM + col_start + j]);
                block_amax = fmaxf(block_amax, fabsf(val));
            }

            // Compute E8M0 scale: round(log2(amax)) + 127
            int e8m0_val;
            if (block_amax <= 1e-9f) {
                e8m0_val = 0;
            } else {
                int exp = (int)roundf(log2f(block_amax));
                e8m0_val = min(max(exp + 127, 0), 255);
            }
            e8m0_vals[b] = (uint8_t)e8m0_val;

            // Quantise: x * (6.0 / 2^exponent)
            float scale_pow2 = exp2f((float)(e8m0_val - 127));
            float scale_inv = 6.0f / scale_pow2;

            #pragma unroll
            for (int j = 0; j < SCALE_BLOCK; j += 2) {
                float2 scaled = {
                    __bfloat162float(smem_input[row * CHUNK_DIM + col_start + j]) * scale_inv,
                    __bfloat162float(smem_input[row * CHUNK_DIM + col_start + j + 1]) * scale_inv
                };
                // Pack to FP4x2
                uint8_t packed = (uint8_t)__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);

                // Write to global output directly
                int64_t out_idx = (int64_t)(gY + row) * (K / 2) + (gX + col_start + j) / 2;
                reinterpret_cast<uint8_t*>(output_fp4)[out_idx] = packed;
            }
        }

        // Write E8M0 scales in swizzled layout: [M/128, K/128, 32, 16]
        // Swizzle: row_in_128 → (row % 32) gives j, (row / 32) gives grp
        int row_in_128 = row;
        int j = row_in_128 % 32;
        int grp = row_in_128 / 32;
        int scale_base = ((int64_t)chunk_row * (K / 128) + chunk_col) * 512;
        int scale_offset = j * 16 + grp * 4;

        // Pack 4 E8M0 values
        output_scales[scale_base + scale_offset + 0] = e8m0_vals[0];
        output_scales[scale_base + scale_offset + 1] = e8m0_vals[1];
        output_scales[scale_base + scale_offset + 2] = e8m0_vals[2];
        output_scales[scale_base + scale_offset + 3] = e8m0_vals[3];
    }

    // Clean up mbarrier
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Grouped quantise dim=0 kernel
//
// Input is [M, K] bf16, split along M into groups.
// Each group gets its own output FP4 tensor and scale tensor.
// The kernel is identical to single-pass quantise but each CTA
// determines which group its rows belong to and writes to the
// corresponding output.
// ═══════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(MX_THREADS)
mxfp4_group_quantize_dim0_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    __nv_fp4x2_e2m1** __restrict__ output_fp4_ptrs,  // [num_groups] pointers
    uint8_t** __restrict__ output_scale_ptrs,         // [num_groups] pointers
    const int* __restrict__ group_boundaries,          // [num_groups + 1] row boundaries
    const int num_groups,
    const int64_t M, const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int chunk_col = blockIdx.x;
    const int chunk_row = blockIdx.y;
    const int gY = chunk_row * CHUNK_DIM;
    const int gX = chunk_col * CHUNK_DIM;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // TMA load
    __shared__ uint64_t mbar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(1));
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(CHUNK_DIM * CHUNK_DIM * 2));
        uint64_t tmap_addr = reinterpret_cast<uint64_t>(&tensor_map_input);
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_input);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(tmap_addr), "r"(gX), "r"(gY), "r"((uint32_t)__cvta_generic_to_shared(&mbar))
            : "memory");
    }
    __syncthreads();

    // Wait for TMA
    {
        uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "WAIT_LOOP_D0:\n"
            "    mbarrier.try_wait.parity.acquire.cta.shared.b64 p, [%0], %1;\n"
            "    @!p bra WAIT_LOOP_D0;\n"
            "}\n"
            :: "r"(mbar_addr), "r"(0));
    }

    const int tid = threadIdx.x;

    if (tid < CHUNK_DIM) {
        const int global_row = gY + tid;

        // Find which group this row belongs to
        int grp_idx = 0;
        for (int g = 0; g < num_groups; g++) {
            if (global_row >= group_boundaries[g] && global_row < group_boundaries[g + 1]) {
                grp_idx = g;
                break;
            }
        }

        int grp_row_start = group_boundaries[grp_idx];
        int grp_M = group_boundaries[grp_idx + 1] - grp_row_start;
        int local_row = global_row - grp_row_start;

        __nv_fp4x2_e2m1* out_fp4 = output_fp4_ptrs[grp_idx];
        uint8_t* out_scales = output_scale_ptrs[grp_idx];

        constexpr int NUM_BLOCKS = CHUNK_DIM / SCALE_BLOCK;
        uint8_t e8m0_vals[NUM_BLOCKS];

        #pragma unroll
        for (int b = 0; b < NUM_BLOCKS; b++) {
            const int col_start = b * SCALE_BLOCK;
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < SCALE_BLOCK; j++) {
                float val = __bfloat162float(smem_input[tid * CHUNK_DIM + col_start + j]);
                block_amax = fmaxf(block_amax, fabsf(val));
            }

            int e8m0_val;
            if (block_amax <= 1e-9f) {
                e8m0_val = 0;
            } else {
                int exp = (int)roundf(log2f(block_amax));
                e8m0_val = min(max(exp + 127, 0), 255);
            }
            e8m0_vals[b] = (uint8_t)e8m0_val;

            float scale_pow2 = exp2f((float)(e8m0_val - 127));
            float scale_inv = 6.0f / scale_pow2;

            #pragma unroll
            for (int j = 0; j < SCALE_BLOCK; j += 2) {
                float2 scaled = {
                    __bfloat162float(smem_input[tid * CHUNK_DIM + col_start + j]) * scale_inv,
                    __bfloat162float(smem_input[tid * CHUNK_DIM + col_start + j + 1]) * scale_inv
                };
                uint8_t packed = (uint8_t)__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
                int64_t out_idx = (int64_t)local_row * (K / 2) + (gX + col_start + j) / 2;
                reinterpret_cast<uint8_t*>(out_fp4)[out_idx] = packed;
            }
        }

        // Write scales in swizzled layout relative to the group
        int local_chunk_row = local_row / 128;
        int row_in_128 = local_row % 128;
        int j = row_in_128 % 32;
        int grp = row_in_128 / 32;
        int scale_base = ((int64_t)local_chunk_row * (K / 128) + chunk_col) * 512;
        int scale_offset = j * 16 + grp * 4;

        out_scales[scale_base + scale_offset + 0] = e8m0_vals[0];
        out_scales[scale_base + scale_offset + 1] = e8m0_vals[1];
        out_scales[scale_base + scale_offset + 2] = e8m0_vals[2];
        out_scales[scale_base + scale_offset + 3] = e8m0_vals[3];
    }

    if (threadIdx.x == 0)
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Grouped quantise dim=1 kernel
//
// Input is [M, K] bf16, split along K into groups.
// Each group gets its own output FP4 tensor and scale tensor.
// Each 128×128 chunk may span group boundaries along K.
// ═══════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(MX_THREADS)
mxfp4_group_quantize_dim1_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    __nv_fp4x2_e2m1** __restrict__ output_fp4_ptrs,
    uint8_t** __restrict__ output_scale_ptrs,
    const int* __restrict__ group_boundaries,   // [num_groups + 1] column boundaries
    const int num_groups,
    const int64_t M, const int64_t K
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const int chunk_col = blockIdx.x;
    const int chunk_row = blockIdx.y;
    const int gY = chunk_row * CHUNK_DIM;
    const int gX = chunk_col * CHUNK_DIM;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    __shared__ uint64_t mbar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(1));
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(CHUNK_DIM * CHUNK_DIM * 2));
        uint64_t tmap_addr = reinterpret_cast<uint64_t>(&tensor_map_input);
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_input);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(tmap_addr), "r"(gX), "r"(gY), "r"((uint32_t)__cvta_generic_to_shared(&mbar))
            : "memory");
    }
    __syncthreads();

    {
        uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "WAIT_LOOP_D1:\n"
            "    mbarrier.try_wait.parity.acquire.cta.shared.b64 p, [%0], %1;\n"
            "    @!p bra WAIT_LOOP_D1;\n"
            "}\n"
            :: "r"(mbar_addr), "r"(0));
    }

    const int tid = threadIdx.x;

    if (tid < CHUNK_DIM) {
        const int row = tid;
        const int global_row = gY + row;
        constexpr int NUM_BLOCKS = CHUNK_DIM / SCALE_BLOCK;

        #pragma unroll
        for (int b = 0; b < NUM_BLOCKS; b++) {
            const int col_start = b * SCALE_BLOCK;
            const int global_col = gX + col_start;

            // Find which group this column block belongs to
            int grp_idx = 0;
            for (int g = 0; g < num_groups; g++) {
                if (global_col >= group_boundaries[g] && global_col < group_boundaries[g + 1]) {
                    grp_idx = g;
                    break;
                }
            }

            int grp_col_start = group_boundaries[grp_idx];
            int grp_K = group_boundaries[grp_idx + 1] - grp_col_start;
            int local_col = global_col - grp_col_start;

            __nv_fp4x2_e2m1* out_fp4 = output_fp4_ptrs[grp_idx];
            uint8_t* out_scales = output_scale_ptrs[grp_idx];

            // Compute amax
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < SCALE_BLOCK; j++) {
                float val = __bfloat162float(smem_input[row * CHUNK_DIM + col_start + j]);
                block_amax = fmaxf(block_amax, fabsf(val));
            }

            int e8m0_val;
            if (block_amax <= 1e-9f) {
                e8m0_val = 0;
            } else {
                int exp = (int)roundf(log2f(block_amax));
                e8m0_val = min(max(exp + 127, 0), 255);
            }

            float scale_pow2 = exp2f((float)(e8m0_val - 127));
            float scale_inv = 6.0f / scale_pow2;

            // Quantise and write to group's output
            #pragma unroll
            for (int j = 0; j < SCALE_BLOCK; j += 2) {
                float2 scaled = {
                    __bfloat162float(smem_input[row * CHUNK_DIM + col_start + j]) * scale_inv,
                    __bfloat162float(smem_input[row * CHUNK_DIM + col_start + j + 1]) * scale_inv
                };
                uint8_t packed = (uint8_t)__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
                int64_t out_idx = (int64_t)global_row * (grp_K / 2) + (local_col + j) / 2;
                reinterpret_cast<uint8_t*>(out_fp4)[out_idx] = packed;
            }

            // Write scale in swizzled layout relative to the group
            int local_chunk_col = local_col / 128;
            int local_scale_block = (local_col % 128) / SCALE_BLOCK;
            int row_in_128 = global_row % 128;
            int local_chunk_row = global_row / 128;

            int j_idx = row_in_128 % 32;
            int grp_in_128 = row_in_128 / 32;
            int scale_base = ((int64_t)local_chunk_row * (grp_K / 128) + local_chunk_col) * 512;
            int scale_offset = j_idx * 16 + grp_in_128 * 4 + local_scale_block;

            out_scales[scale_base + scale_offset] = (uint8_t)e8m0_val;
        }
    }

    if (threadIdx.x == 0)
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
#endif
}


// ═══════════════════════════════════════════════════════════════════
// PyTorch dispatch functions
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    auto row_fp4 = torch::empty({M, K / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc  = torch::empty({M / 128, K / 128, 32, 16}, torch::dtype(torch::kUInt8).device(device));

    // Create TMA map for input (bf16)
    alignas(64) CUtensorMap tmap_in{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, CHUNK_DIM, CHUNK_DIM, K, 16);

    // Launch kernel
    const int chunks_X = K / CHUNK_DIM;
    const int chunks_Y = M / CHUNK_DIM;
    const dim3 grid(chunks_X, chunks_Y);
    const int smem_bytes = CHUNK_DIM * CHUNK_DIM * sizeof(__nv_bfloat16) + 128;

    cudaFuncSetAttribute(mxfp4_quantize_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    void* args[] = {
        const_cast<CUtensorMap*>(&tmap_in),
        row_fp4.data_ptr(),
        row_sc.data_ptr(),
        const_cast<int64_t*>(&M),
        const_cast<int64_t*>(&K)
    };

    mxfp4_quantize_kernel<<<grid, MX_THREADS, smem_bytes, stream>>>(
        tmap_in,
        reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        M, K
    );

    return {row_fp4, row_sc};
}


std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
mxfp4_group_quantize_dim0(torch::Tensor input, std::vector<int64_t> split_sizes) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    int num_groups = split_sizes.size();
    TORCH_CHECK(num_groups >= 1 && num_groups <= 8);

    // Compute group boundaries
    std::vector<int> boundaries(num_groups + 1);
    boundaries[0] = 0;
    int64_t total = 0;
    for (int i = 0; i < num_groups; i++) {
        total += split_sizes[i];
        boundaries[i + 1] = total;
        TORCH_CHECK(split_sizes[i] % 128 == 0, "Each group size along dim=0 must be multiple of 128");
    }
    TORCH_CHECK(total == M, "Split sizes must sum to M");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    // Allocate per-group outputs
    std::vector<torch::Tensor> fp4_list, sc_list;
    for (int i = 0; i < num_groups; i++) {
        int grp_M = split_sizes[i];
        fp4_list.push_back(torch::empty({grp_M, K / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device)));
        sc_list.push_back(torch::empty({grp_M / 128, K / 128, 32, 16}, torch::dtype(torch::kUInt8).device(device)));
    }

    // Allocate device-side pointer arrays and boundaries
    auto d_fp4_ptrs = torch::empty({num_groups}, torch::dtype(torch::kInt64).device(device));
    auto d_sc_ptrs = torch::empty({num_groups}, torch::dtype(torch::kInt64).device(device));
    auto d_boundaries = torch::empty({num_groups + 1}, torch::dtype(torch::kInt32).device(device));

    std::vector<int64_t> h_fp4_ptrs(num_groups), h_sc_ptrs(num_groups);
    for (int i = 0; i < num_groups; i++) {
        h_fp4_ptrs[i] = reinterpret_cast<int64_t>(fp4_list[i].data_ptr());
        h_sc_ptrs[i] = reinterpret_cast<int64_t>(sc_list[i].data_ptr());
    }

    cudaMemcpyAsync(d_fp4_ptrs.data_ptr(), h_fp4_ptrs.data(), num_groups * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sc_ptrs.data_ptr(), h_sc_ptrs.data(), num_groups * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boundaries.data_ptr(), boundaries.data(), (num_groups + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Create TMA for input
    alignas(64) CUtensorMap tmap_in{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, CHUNK_DIM, CHUNK_DIM, K, 16);

    const int chunks_X = K / CHUNK_DIM;
    const int chunks_Y = M / CHUNK_DIM;
    const dim3 grid(chunks_X, chunks_Y);
    const int smem_bytes = CHUNK_DIM * CHUNK_DIM * sizeof(__nv_bfloat16) + 128;

    cudaFuncSetAttribute(mxfp4_group_quantize_dim0_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    mxfp4_group_quantize_dim0_kernel<<<grid, MX_THREADS, smem_bytes, stream>>>(
        tmap_in,
        reinterpret_cast<__nv_fp4x2_e2m1**>(d_fp4_ptrs.data_ptr()),
        reinterpret_cast<uint8_t**>(d_sc_ptrs.data_ptr()),
        d_boundaries.data_ptr<int>(),
        num_groups, M, K
    );

    return {fp4_list, sc_list};
}


std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
mxfp4_group_quantize_dim1(torch::Tensor input, std::vector<int64_t> split_sizes) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    int num_groups = split_sizes.size();
    TORCH_CHECK(num_groups >= 1 && num_groups <= 8);

    std::vector<int> boundaries(num_groups + 1);
    boundaries[0] = 0;
    int64_t total = 0;
    for (int i = 0; i < num_groups; i++) {
        total += split_sizes[i];
        boundaries[i + 1] = total;
        TORCH_CHECK(split_sizes[i] % 128 == 0, "Each group size along dim=1 must be multiple of 128");
    }
    TORCH_CHECK(total == K, "Split sizes must sum to K");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    std::vector<torch::Tensor> fp4_list, sc_list;
    for (int i = 0; i < num_groups; i++) {
        int grp_K = split_sizes[i];
        fp4_list.push_back(torch::empty({M, grp_K / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device)));
        sc_list.push_back(torch::empty({M / 128, grp_K / 128, 32, 16}, torch::dtype(torch::kUInt8).device(device)));
    }

    auto d_fp4_ptrs = torch::empty({num_groups}, torch::dtype(torch::kInt64).device(device));
    auto d_sc_ptrs = torch::empty({num_groups}, torch::dtype(torch::kInt64).device(device));
    auto d_boundaries = torch::empty({num_groups + 1}, torch::dtype(torch::kInt32).device(device));

    std::vector<int64_t> h_fp4_ptrs(num_groups), h_sc_ptrs(num_groups);
    for (int i = 0; i < num_groups; i++) {
        h_fp4_ptrs[i] = reinterpret_cast<int64_t>(fp4_list[i].data_ptr());
        h_sc_ptrs[i] = reinterpret_cast<int64_t>(sc_list[i].data_ptr());
    }

    cudaMemcpyAsync(d_fp4_ptrs.data_ptr(), h_fp4_ptrs.data(), num_groups * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sc_ptrs.data_ptr(), h_sc_ptrs.data(), num_groups * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boundaries.data_ptr(), boundaries.data(), (num_groups + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    alignas(64) CUtensorMap tmap_in{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, CHUNK_DIM, CHUNK_DIM, K, 16);

    const int chunks_X = K / CHUNK_DIM;
    const int chunks_Y = M / CHUNK_DIM;
    const dim3 grid(chunks_X, chunks_Y);
    const int smem_bytes = CHUNK_DIM * CHUNK_DIM * sizeof(__nv_bfloat16) + 128;

    cudaFuncSetAttribute(mxfp4_group_quantize_dim1_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    mxfp4_group_quantize_dim1_kernel<<<grid, MX_THREADS, smem_bytes, stream>>>(
        tmap_in,
        reinterpret_cast<__nv_fp4x2_e2m1**>(d_fp4_ptrs.data_ptr()),
        reinterpret_cast<uint8_t**>(d_sc_ptrs.data_ptr()),
        d_boundaries.data_ptr<int>(),
        num_groups, M, K
    );

    return {fp4_list, sc_list};
}


// ═══════════════════════════════════════════════════════════════════
// pybind11 module
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(mxfp4_quant, m) {
    m.def("mxfp4_quantize_for_gemm", &mxfp4_quantize_for_gemm,
        "MXFP4 single-pass quantise (E8M0 scales, block-32)",
        py::arg("input"));
    m.def("mxfp4_group_quantize_dim0", &mxfp4_group_quantize_dim0,
        "MXFP4 grouped quantise along dim=0 (rows)",
        py::arg("input"), py::arg("split_sizes"));
    m.def("mxfp4_group_quantize_dim1", &mxfp4_group_quantize_dim1,
        "MXFP4 grouped quantise along dim=1 (columns)",
        py::arg("input"), py::arg("split_sizes"));
}

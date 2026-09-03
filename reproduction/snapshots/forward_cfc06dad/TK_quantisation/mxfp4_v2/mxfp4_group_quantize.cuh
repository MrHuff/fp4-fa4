/*************************************************************************
 * MXFP4 Group Quantize — v2
 *
 * Two kernels:
 *   1. mxfp4_v2_group_quantize_kernel: Quantizes a contiguous input with
 *      per-group scale output, using a single TMA map for the full input.
 *   2. mxfp4_v2_multi_quantize_kernel: Quantizes multiple non-contiguous
 *      tensors, using per-group TMA maps stored in device memory.
 *************************************************************************/

#pragma once
#include "mxfp4_quantize.cuh"

namespace mxfp4_v2 {

static constexpr int MAX_GROUPS = 16;

struct GroupArgs {
    int boundaries[MAX_GROUPS + 1];
    uint8_t* scale_ptrs[MAX_GROUPS];
    int num_groups;
};

// ═══════════════════════════════════════════════════════════════════
// Group quantize kernel — contiguous input, per-group scale output
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(MX_THREADS)
mxfp4_v2_group_quantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const int64_t M, const int64_t K,
    GroupArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int tid = threadIdx.x;

    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int gY = ctaid_Y * MX_CHUNK_DIM;
    const int gX = ctaid_X * MX_CHUNK_DIM;

    // Find which group this chunk's rows belong to
    int grp_idx = 0;
    for (int g = 1; g < args.num_groups; ++g) {
        if (gY >= args.boundaries[g]) grp_idx = g;
    }
    const int local_row_base = gY - args.boundaries[grp_idx];
    const int grp_M = args.boundaries[grp_idx + 1] - args.boundaries[grp_idx];

    // SMEM
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(MX_CHUNK_DIM * MX_CHUNK_DIM * (int)sizeof(__nv_bfloat16), TMA_SHMEM_ALIGNMENT);
    __nv_bfloat16* smem_in = reinterpret_cast<__nv_bfloat16*>(dshmem);
    __nv_fp4x2_e2m1* smem_fp4 = reinterpret_cast<__nv_fp4x2_e2m1*>(dshmem + in_bytes);

    // TMA load (uses global coordinates into the contiguous input)
    __shared__ uint64_t in_mbar;
    if (leading) {
        mbarrier_init(&in_mbar, 1);
        fence_proxy_async_shared_cta();
        mbarrier_arrive_expect_tx(&in_mbar, MX_CHUNK_DIM * MX_CHUNK_DIM * sizeof(__nv_bfloat16));
        cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(smem_in),
            reinterpret_cast<const uint64_t*>(&tensor_map_input),
            gX, gY, &in_mbar);
    }
    __syncthreads();
    mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar, 0);

    // Quantize
    uint8_t e8m0_vals[MX_NUM_BLOCKS];
    if (tid < MX_CHUNK_DIM) {
        quantize_row(smem_in, smem_fp4, e8m0_vals, tid);
    }
    __syncthreads();

    // TMA store FP4 to contiguous output (same as single-pass)
    if (leading) {
        fence_proxy_async_shared_cta();
        cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t*>(&tensor_map_output),
            gX / 2, gY,
            reinterpret_cast<uint64_t*>(smem_fp4));
        cp_async_bulk_commit_group();
    }

    // Write scales to per-group output buffer using LOCAL coordinates
    if (tid < MX_CHUNK_DIM) {
        const int local_row = local_row_base + tid;
        if (local_row < grp_M) {
            uint8_t* out_sc = args.scale_ptrs[grp_idx];
            const int tm = local_row / 128;
            const int ntk = K / 128;
            const int j = (local_row % 128) % 32;
            const int grp = (local_row % 128) / 32;
            const int base = (tm * ntk + ctaid_X) * 512 + j * 16 + grp * 4;
            uint32_t pk;
            uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
            p[0] = e8m0_vals[0]; p[1] = e8m0_vals[1];
            p[2] = e8m0_vals[2]; p[3] = e8m0_vals[3];
            *reinterpret_cast<uint32_t*>(out_sc + base) = pk;
        }
    }

    if (leading) {
        cp_async_bulk_wait_group_read<0>();
        mbarrier_invalid(&in_mbar);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════
// Multi quantize kernel — non-contiguous tensors, per-group TMA maps
// ═══════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(MX_THREADS)
mxfp4_v2_multi_quantize_kernel(
    const CUtensorMap* __restrict__ input_tma_maps,
    const CUtensorMap* __restrict__ output_tma_maps,
    const int64_t K,
    GroupArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = (threadIdx.x == 0);
    const int tid = threadIdx.x;

    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int gY = ctaid_Y * MX_CHUNK_DIM;
    const int gX = ctaid_X * MX_CHUNK_DIM;

    // Find which group
    int grp_idx = 0;
    for (int g = 1; g < args.num_groups; ++g) {
        if (gY >= args.boundaries[g]) grp_idx = g;
    }
    const int local_gY = gY - args.boundaries[grp_idx];
    const int grp_M = args.boundaries[grp_idx + 1] - args.boundaries[grp_idx];

    // SMEM
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    constexpr int in_bytes = DIVUP_TO_MULTIPLE(MX_CHUNK_DIM * MX_CHUNK_DIM * (int)sizeof(__nv_bfloat16), TMA_SHMEM_ALIGNMENT);
    __nv_bfloat16* smem_in = reinterpret_cast<__nv_bfloat16*>(dshmem);
    __nv_fp4x2_e2m1* smem_fp4 = reinterpret_cast<__nv_fp4x2_e2m1*>(dshmem + in_bytes);

    // TMA load using per-group input descriptor with LOCAL coordinates
    __shared__ uint64_t in_mbar;
    if (leading) {
        mbarrier_init(&in_mbar, 1);
        fence_proxy_async_shared_cta();
        mbarrier_arrive_expect_tx(&in_mbar, MX_CHUNK_DIM * MX_CHUNK_DIM * sizeof(__nv_bfloat16));
        // Load from per-group TMA map using local_gY
        uint64_t tmap_addr = reinterpret_cast<uint64_t>(&input_tma_maps[grp_idx]);
        uint32_t smem_addr = __cvta_generic_to_shared(smem_in);
        uint32_t mbar_addr = __cvta_generic_to_shared(&in_mbar);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
            ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(tmap_addr), "r"(gX), "r"(local_gY), "r"(mbar_addr)
            : "memory");
    }
    __syncthreads();
    mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar, 0);

    // Quantize
    uint8_t e8m0_vals[MX_NUM_BLOCKS];
    if (tid < MX_CHUNK_DIM) {
        quantize_row(smem_in, smem_fp4, e8m0_vals, tid);
    }
    __syncthreads();

    // TMA store FP4 using per-group output descriptor with LOCAL coordinates
    if (leading) {
        fence_proxy_async_shared_cta();
        uint64_t out_tmap_addr = reinterpret_cast<uint64_t>(&output_tma_maps[grp_idx]);
        uint32_t fp4_smem_addr = __cvta_generic_to_shared(smem_fp4);
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
            :: "l"(out_tmap_addr), "r"(gX / 2), "r"(local_gY), "r"(fp4_smem_addr)
            : "memory");
        cp_async_bulk_commit_group();
    }

    // Write scales
    if (tid < MX_CHUNK_DIM) {
        const int local_row = local_gY + tid;
        if (local_row < grp_M) {
            uint8_t* out_sc = args.scale_ptrs[grp_idx];
            const int tm = local_row / 128;
            const int ntk = K / 128;
            const int j = (local_row % 128) % 32;
            const int grp = (local_row % 128) / 32;
            const int base = (tm * ntk + ctaid_X) * 512 + j * 16 + grp * 4;
            uint32_t pk;
            uint8_t* p = reinterpret_cast<uint8_t*>(&pk);
            p[0] = e8m0_vals[0]; p[1] = e8m0_vals[1];
            p[2] = e8m0_vals[2]; p[3] = e8m0_vals[3];
            *reinterpret_cast<uint32_t*>(out_sc + base) = pk;
        }
    }

    if (leading) {
        cp_async_bulk_wait_group_read<0>();
        mbarrier_invalid(&in_mbar);
    }
#endif
}

} // namespace mxfp4_v2

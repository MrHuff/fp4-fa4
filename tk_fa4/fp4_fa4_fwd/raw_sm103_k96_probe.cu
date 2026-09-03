#include <cstdint>
#include <type_traits>

#include "kittens.cuh"

using namespace kittens;

#include "fwd_raw_fp4_throughput_helpers.inc"

struct alignas(8) raw_sm103_k96_globals {
    const uint8_t *a_fp4;
    const uint8_t *b_fp4;
    const bf16 *a_bf16;
    const bf16 *b_bf16;
    uint64_t *cycles;
    uint64_t *start_globaltimer;
    uint64_t *end_globaltimer;
    int32_t *smids;
    int32_t iterations;
    int32_t blocks;
};

static_assert(sizeof(raw_sm103_k96_globals) == 72);

__device__ __forceinline__ uint64_t raw_sm103_globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ void raw_sm103_cluster_arrive_release(
    semaphore &bar,
    int dst_cta
) {
    const uint32_t local_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t remote_addr;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(remote_addr)
        : "r"(local_addr), "r"(dst_cta));
    asm volatile(
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n"
        :
        : "r"(remote_addr)
        : "memory");
}

template <typename Tile>
__device__ __forceinline__ void raw_sm103_load_fp4_tile(
    Tile &tile,
    const uint8_t *source
) {
    constexpr int bytes_per_store = 16;
    constexpr int stores_per_row = Tile::cols / bytes_per_store;
    const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * stores_per_row;
         linear += blockDim.x) {
        const int row = linear / stores_per_row;
        const int col = (linear % stores_per_row) * bytes_per_store;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            source + row * Tile::cols + col);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(Tile::idx(base, {row, col})),
            "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename Tile>
__device__ __forceinline__ void raw_sm103_load_fp4_tile_repeated(
    Tile &tile,
    const uint8_t *source
) {
    constexpr int source_bytes_per_row = 64;
    constexpr int bytes_per_store = 16;
    constexpr int stores_per_row = Tile::cols / bytes_per_store;
    const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * stores_per_row;
         linear += blockDim.x) {
        const int row = linear / stores_per_row;
        const int col = (linear % stores_per_row) * bytes_per_store;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            source + row * source_bytes_per_row +
            col % source_bytes_per_row);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(Tile::idx(base, {row, col})),
            "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <int Accumulate, bool Ultra, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void raw_sm103_issue_k192(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const auto make_k96_desc = [&](uint64_t current, uint64_t next) {
        return (current & ~lbo_field_mask) |
               ((next & 0x3fffull) << 16) |
               absolute_lbo_mode;
    };
    const uint64_t a_k96 = make_k96_desc(
        a_desc.chunk_descriptor(0), a_desc.chunk_descriptor(1));
    const uint64_t b_k96 = make_k96_desc(
        b_desc.chunk_descriptor(0), b_desc.chunk_descriptor(1));
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 128, false, 0>() |
        (1u << 31);
    if constexpr (Ultra) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.eq.u32 p, 1, %6;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
            "}\n" ::
            "r"(output.addr), "l"(a_k96), "l"(b_k96), "n"(idesc),
            "r"(a_scale.addr), "r"(b_scale.addr), "n"(Accumulate));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.eq.u32 p, 1, 1;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
            "}\n" ::
            "r"(output.addr), "l"(a_k96), "l"(b_k96), "n"(idesc),
            "r"(a_scale.addr), "r"(b_scale.addr));
    } else {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.eq.u32 p, 1, %6;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
            "}\n" ::
            "r"(output.addr), "l"(a_k96), "l"(b_k96), "n"(idesc),
            "r"(a_scale.addr), "r"(b_scale.addr), "n"(Accumulate));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.eq.u32 p, 1, 1;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
            "}\n" ::
            "r"(output.addr), "l"(a_k96), "l"(b_k96), "n"(idesc),
            "r"(a_scale.addr), "r"(b_scale.addr));
    }
}

template <int Accumulate, typename Output, typename ADesc, typename BDesc,
          typename AScale, typename BScale>
__device__ __forceinline__ void raw_sm103_issue_k192_nvfp4_ultra(
    Output &output,
    ADesc &a_desc,
    BDesc &b_desc,
    const AScale &a_scale,
    const BScale &b_scale
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const auto make_k96_desc = [&](uint64_t current, uint64_t next) {
        return (current & ~lbo_field_mask) |
               ((next & 0x3fffull) << 16) |
               absolute_lbo_mode;
    };
    const uint64_t a_k96 = make_k96_desc(
        a_desc.chunk_descriptor(0), a_desc.chunk_descriptor(1));
    const uint64_t b_k96 = make_k96_desc(
        b_desc.chunk_descriptor(0), b_desc.chunk_descriptor(1));
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 128, false, 0>() |
        (1u << 31);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "l"(a_k96), "l"(b_k96), "n"(idesc),
        "r"(a_scale.addr), "r"(b_scale.addr), "n"(Accumulate));
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], 1;\n" ::
        "r"(output.addr), "l"(a_k96), "l"(b_k96), "n"(idesc),
        "r"(a_scale.addr), "r"(b_scale.addr));
}

template <int Accumulate, typename Output>
__device__ __forceinline__ void raw_sm103_issue_nvfp4_ultra_k96(
    Output &output,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t a_scale_addr,
    uint32_t b_scale_addr
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 128, false, 0>() |
        (1u << 31);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "l"(a_desc), "l"(b_desc), "n"(idesc),
        "r"(a_scale_addr), "r"(b_scale_addr), "n"(Accumulate));
}

__device__ __forceinline__ void raw_sm103_fill_tmem_128x16(
    uint32_t tmem_addr,
    uint32_t value
) {
    #pragma unroll
    for (int row = 0; row < 128; row += 32) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32 "
            "[%0], {%1, %1, %1, %1};" ::
            "r"(tmem_addr + (static_cast<uint32_t>(row) << 16)),
            "r"(value));
    }
}

__device__ __forceinline__ void raw_sm103_fill_nvfp4_scale(
    uint32_t tmem_addr
) {
    raw_sm103_fill_tmem_128x16(tmem_addr, 0x38383838u);
}

using raw_sm103_output_t = tt<float, 128, 128>;
using raw_sm103_scaled_tile = st_fp4e2m1_2<128, 64>;
using raw_sm103_ultra_tile = st_fp4e2m1_2<128, 384>;
using raw_sm103_scale_t = full_tt_fp8e8m0<16>;
using raw_sm103_nv_scale_t = full_tt_fp8e4m3<16>;
using raw_sm103_nv_scale_wide_t = full_tt_fp8e4m3<48>;
using raw_sm103_group2_b_tile = st_fp4e2m1_2<64, 64>;
using raw_sm103_group2_tail_a_tile = st_fp4e2m1_2<128, 128>;
using raw_sm103_group2_tail_b_tile = st_fp4e2m1_2<64, 128>;
using raw_sm103_nv_scale_smem_t = st_fp8e4m3<32, 16, false>;
using raw_sm103_mx_scale_smem_t = st_fp8e8m0<32, 16, false>;
using raw_sm103_group2_n192_k_tile = st_fp4e2m1_2<96, 64>;
using raw_sm103_group2_n192_v_tile = st_fp4e2m1_2<64, 64>;
using raw_sm103_group2_n192_score_t = tt<float, 128, 192>;

template <int Accumulate, int ScaleFactorId, typename Output>
__device__ __forceinline__ void raw_sm103_issue_nvfp4_k64_group1(
    Output &output,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t a_scale_addr,
    uint32_t b_scale_addr
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 128, 128,
            false, ScaleFactorId>();
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "l"(a_desc), "l"(b_desc), "n"(idesc),
        "r"(a_scale_addr), "r"(b_scale_addr), "n"(Accumulate));
}

template <int Accumulate, int ScaleFactorId, typename Output>
__device__ __forceinline__ void raw_sm103_issue_nvfp4_k64_group2(
    Output &output,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t a_scale_addr,
    uint32_t b_scale_addr
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 256, 128,
            false, ScaleFactorId>();
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "l"(a_desc), "l"(b_desc), "n"(idesc),
        "r"(a_scale_addr), "r"(b_scale_addr), "n"(Accumulate));
}

template <int Accumulate, typename Output>
__device__ __forceinline__ void raw_sm103_issue_nvfp4_ultra_k96_group2(
    Output &output,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t a_scale_addr,
    uint32_t b_scale_addr
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 256, 128, false, 0>() |
        (1u << 31);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "l"(a_desc), "l"(b_desc), "n"(idesc),
        "r"(a_scale_addr), "r"(b_scale_addr), "n"(Accumulate));
}

template <int Accumulate, typename Output>
__device__ __forceinline__ void raw_sm103_issue_nvfp4_k64_group2_n192(
    Output &output,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t a_scale_addr,
    uint32_t b_scale_addr
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e4m3;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 256, 192, false, 0>();
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "l"(a_desc), "l"(b_desc), "n"(idesc),
        "r"(a_scale_addr), "r"(b_scale_addr), "n"(Accumulate));
}

template <int Accumulate, typename Output>
__device__ __forceinline__ void raw_sm103_issue_mxfp4_k96_group2(
    Output &output,
    uint32_t a_addr,
    uint64_t b_desc,
    uint32_t a_scale_addr,
    uint32_t b_scale_addr
) {
    using fp4_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4_t, scale_t, 256, 128, false, 0>() |
        (1u << 31);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, 1, %6;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block32 "
        "[%0], [%1], %2, %3, [%4], [%5], p;\n\t"
        "}\n" ::
        "r"(output.addr), "r"(a_addr), "l"(b_desc), "n"(idesc),
        "r"(a_scale_addr), "r"(b_scale_addr), "n"(Accumulate)
        : "memory");
}

template <typename Scale>
__device__ __forceinline__ void raw_sm103_fill_scale_smem(
    Scale &scale,
    uint32_t value
) {
    constexpr int words = sizeof(scale) / sizeof(uint4);
    const uint32_t base = __cvta_generic_to_shared(&scale.data[0]);
    for (int word = threadIdx.x; word < words; word += blockDim.x) {
        asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" ::
            "r"(base + static_cast<uint32_t>(word * sizeof(uint4))),
            "r"(value));
    }
}

__device__ __forceinline__ void raw_sm103_fill_nvfp4_scale_smem(
    raw_sm103_nv_scale_smem_t &scale
) {
    raw_sm103_fill_scale_smem(scale, 0x38383838u);
}

__device__ __forceinline__ void raw_sm103_group2_n192_joint_probe_body(
    const raw_sm103_k96_globals &g,
    raw_sm103_group2_n192_score_t score,
    raw_sm103_output_t output,
    raw_sm103_nv_scale_t q_scale,
    raw_sm103_nv_scale_t k_scale,
    raw_sm103_scale_t p_scale,
    raw_sm103_scale_t v_scale,
    raw_sm103_scaled_tile &q,
    raw_sm103_group2_n192_k_tile &k,
    raw_sm103_group2_n192_v_tile &v0,
    raw_sm103_group2_n192_v_tile &v1,
    raw_sm103_nv_scale_smem_t &qk_scale_smem,
    raw_sm103_mx_scale_smem_t &pv_scale_smem,
    semaphore &qk_done,
    semaphore &pv_done,
    semaphore &qk_remote_ready,
    semaphore &p_remote_ready
) {
    const int rank = cluster_ctarank();
    if (threadIdx.x == 0) {
        init_semaphore(qk_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
        init_semaphore(qk_remote_ready, 0, 1);
        init_semaphore(p_remote_ready, 0, 1);
    }
    raw_sm103_load_fp4_tile(q, g.a_fp4);
    raw_sm103_load_fp4_tile(k, g.b_fp4);
    raw_sm103_load_fp4_tile(v0, g.a_fp4);
    raw_sm103_load_fp4_tile(v1, g.b_fp4);
    raw_sm103_fill_scale_smem(qk_scale_smem, 0x38383838u);
    raw_sm103_fill_scale_smem(pv_scale_smem, 0x7f7f7f7fu);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");

    if (rank == 0 && threadIdx.x < 32) {
        load_mxnv_scale_async<2>(q_scale, qk_scale_smem);
        load_mxnv_scale_async<2>(k_scale, qk_scale_smem);
        load_mxnv_scale_async<2>(p_scale, pv_scale_smem);
        load_mxnv_scale_async<2>(v_scale, pv_scale_smem);
        tensor_load_wait();
        tensor_before_thread_sync();
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");

    st_descriptor<raw_sm103_scaled_tile, transpose::N> q_desc(q);
    st_descriptor<raw_sm103_group2_n192_k_tile, transpose::N> k_desc(k);
    st_descriptor<raw_sm103_group2_n192_v_tile, transpose::N> v0_desc(v0);
    st_descriptor<raw_sm103_group2_n192_v_tile, transpose::N> v1_desc(v1);
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const auto make_k96_desc = [&](uint64_t current, uint64_t next) {
        return (current & ~lbo_field_mask) |
               ((next & 0x3fffull) << 16) |
               absolute_lbo_mode;
    };
    const uint64_t v0_k96 = make_k96_desc(
        v0_desc.chunk_descriptor(0), v0_desc.chunk_descriptor(1));
    const uint64_t v1_k96 = make_k96_desc(
        v1_desc.chunk_descriptor(0), v1_desc.chunk_descriptor(1));

    uint64_t start_cycle = 0;
    if (threadIdx.x == 0) {
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
    }
    int qk_phase = 0;
    int pv_phase = 0;
    #pragma unroll 1
    for (int iter = 0; iter < g.iterations; ++iter) {
        if (rank == 0 && threadIdx.x == 0) {
            raw_sm103_issue_nvfp4_k64_group2_n192<0>(
                score,
                q_desc.chunk_descriptor(0),
                k_desc.chunk_descriptor(0),
                q_scale.addr,
                k_scale.addr);
            raw_sm103_issue_nvfp4_k64_group2_n192<1>(
                score,
                q_desc.chunk_descriptor(1),
                k_desc.chunk_descriptor(1),
                q_scale.addr + 4,
                k_scale.addr + 4);
            detail::tcgen05::commit<2>(qk_done);
            wait(qk_done, qk_phase);
        }
        if (rank == 0 && threadIdx.x < 32) {
            __syncwarp();
            tensor_after_thread_sync();
            __syncwarp();
            if (threadIdx.x == 0) {
                raw_sm103_cluster_arrive_release(qk_remote_ready, 1);
            }
        } else if (rank == 1 && threadIdx.x == 0) {
            tma::cluster::wait(qk_remote_ready, qk_phase);
        }

        if (threadIdx.x < 32) {
            __syncwarp();
            raw_sm103_fill_tmem_128x16(score.addr, 0x22222222u);
            raw_sm103_fill_tmem_128x16(score.addr + 16, 0x22222222u);
            fp4pv_tmem_store_wait();
            tensor_before_thread_sync();
            __syncwarp();
            if (rank == 1 && threadIdx.x == 0) {
                raw_sm103_cluster_arrive_release(p_remote_ready, 0);
            } else if (rank == 0 && threadIdx.x == 0) {
                tma::cluster::wait(p_remote_ready, pv_phase);
            }
            __syncwarp();
        }

        if (rank == 0 && threadIdx.x == 0) {
            if (iter == 0) {
                raw_sm103_issue_mxfp4_k96_group2<0>(
                    output, score.addr, v0_k96,
                    p_scale.addr, v_scale.addr);
            } else {
                raw_sm103_issue_mxfp4_k96_group2<1>(
                    output, score.addr, v0_k96,
                    p_scale.addr, v_scale.addr);
            }
            raw_sm103_issue_mxfp4_k96_group2<1>(
                output, score.addr + 12, v1_k96,
                p_scale.addr + 4, v_scale.addr + 4);
            detail::tcgen05::commit<2>(pv_done);
            wait(pv_done, pv_phase);
        }
        if (rank == 0 && threadIdx.x < 32) {
            __syncwarp();
            tensor_after_thread_sync();
        }
        qk_phase ^= 1;
        pv_phase ^= 1;
    }
    if (threadIdx.x == 0) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
}

__device__ __forceinline__ void raw_sm103_k96_nvfp4_group2_probe_body(
    const raw_sm103_k96_globals &g,
    raw_sm103_output_t output0,
    raw_sm103_output_t output1,
    raw_sm103_nv_scale_t a_scale,
    raw_sm103_nv_scale_t b_scale,
    raw_sm103_scaled_tile &a,
    raw_sm103_group2_b_tile &b,
    raw_sm103_nv_scale_smem_t &scale_smem,
    semaphore &done
) {
    const int rank = cluster_ctarank();
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    raw_sm103_load_fp4_tile(a, g.a_fp4);
    raw_sm103_load_fp4_tile(b, g.b_fp4);
    raw_sm103_fill_nvfp4_scale_smem(scale_smem);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");

    if (rank == 0 && threadIdx.x < 32) {
        load_mxnv_scale_async<2>(a_scale, scale_smem);
        load_mxnv_scale_async<2>(b_scale, scale_smem);
        tensor_load_wait();
        tensor_before_thread_sync();
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");

    st_descriptor<raw_sm103_scaled_tile, transpose::N> a_desc(a);
    st_descriptor<raw_sm103_group2_b_tile, transpose::N> b_desc(b);
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const auto make_k96_desc = [&](uint64_t current, uint64_t next) {
        return (current & ~lbo_field_mask) |
               ((next & 0x3fffull) << 16) |
               absolute_lbo_mode;
    };
    const uint64_t ad = make_k96_desc(
        a_desc.chunk_descriptor(0), a_desc.chunk_descriptor(1));
    const uint64_t bd = make_k96_desc(
        b_desc.chunk_descriptor(0), b_desc.chunk_descriptor(1));

    uint64_t start_cycle = 0;
    if (threadIdx.x == 0) {
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
    }
    if (rank == 0 && threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            raw_sm103_issue_nvfp4_ultra_k96_group2<0>(
                output0, ad, bd, a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_ultra_k96_group2<1>(
                output0, ad, bd, a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_ultra_k96_group2<0>(
                output1, ad, bd, a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_ultra_k96_group2<1>(
                output1, ad, bd, a_scale.addr, b_scale.addr);
            #pragma unroll 1
            for (int iter = 2; iter < g.iterations; iter += 2) {
                raw_sm103_issue_nvfp4_ultra_k96_group2<1>(
                    output0, ad, bd, a_scale.addr, b_scale.addr);
                raw_sm103_issue_nvfp4_ultra_k96_group2<1>(
                    output0, ad, bd, a_scale.addr, b_scale.addr);
                raw_sm103_issue_nvfp4_ultra_k96_group2<1>(
                    output1, ad, bd, a_scale.addr, b_scale.addr);
                raw_sm103_issue_nvfp4_ultra_k96_group2<1>(
                    output1, ad, bd, a_scale.addr, b_scale.addr);
            }
            detail::tcgen05::commit<2>(done);
            wait(done, 0);
        }
        __syncwarp();
        tensor_after_thread_sync();
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");
    if (threadIdx.x == 0) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
}

__device__ __forceinline__ void raw_sm103_nvfp4_k128_probe_body(
    const raw_sm103_k96_globals &g,
    raw_sm103_output_t output0,
    raw_sm103_output_t output1,
    raw_sm103_nv_scale_t a_scale,
    raw_sm103_nv_scale_t b_scale,
    raw_sm103_scaled_tile &a,
    raw_sm103_scaled_tile &b,
    semaphore &done
) {
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    raw_sm103_load_fp4_tile(a, g.a_fp4);
    raw_sm103_load_fp4_tile(b, g.b_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    if (threadIdx.x < 32) {
        raw_sm103_fill_nvfp4_scale(a_scale.addr);
        raw_sm103_fill_nvfp4_scale(b_scale.addr);
        fp4pv_tmem_store_wait();
    }
    __syncthreads();

    st_descriptor<raw_sm103_scaled_tile, transpose::N> a_desc(a);
    st_descriptor<raw_sm103_scaled_tile, transpose::N> b_desc(b);
    uint64_t start_cycle = 0;
    if (threadIdx.x == 0) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        raw_sm103_issue_nvfp4_k64_group1<0, 0>(
            output0, a_desc.chunk_descriptor(0),
            b_desc.chunk_descriptor(0), a_scale.addr, b_scale.addr);
        raw_sm103_issue_nvfp4_k64_group1<1, 0>(
            output0, a_desc.chunk_descriptor(1),
            b_desc.chunk_descriptor(1), a_scale.addr + 4, b_scale.addr + 4);
        raw_sm103_issue_nvfp4_k64_group1<0, 0>(
            output1, a_desc.chunk_descriptor(0),
            b_desc.chunk_descriptor(0), a_scale.addr, b_scale.addr);
        raw_sm103_issue_nvfp4_k64_group1<1, 0>(
            output1, a_desc.chunk_descriptor(1),
            b_desc.chunk_descriptor(1), a_scale.addr + 4, b_scale.addr + 4);
        #pragma unroll 1
        for (int iter = 2; iter < g.iterations; iter += 2) {
            raw_sm103_issue_nvfp4_k64_group1<1, 0>(
                output0, a_desc.chunk_descriptor(0),
                b_desc.chunk_descriptor(0), a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_k64_group1<1, 0>(
                output0, a_desc.chunk_descriptor(1),
                b_desc.chunk_descriptor(1),
                a_scale.addr + 4, b_scale.addr + 4);
            raw_sm103_issue_nvfp4_k64_group1<1, 0>(
                output1, a_desc.chunk_descriptor(0),
                b_desc.chunk_descriptor(0), a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_k64_group1<1, 0>(
                output1, a_desc.chunk_descriptor(1),
                b_desc.chunk_descriptor(1),
                a_scale.addr + 4, b_scale.addr + 4);
        }
        detail::tcgen05::commit<1>(done);
    }
    if (threadIdx.x < 32) {
        wait(done, 0);
        tensor_after_thread_sync();
    }
    if (threadIdx.x == 0) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
}

__device__ __forceinline__ void raw_sm103_nvfp4_group2_k128_mixed_body(
    const raw_sm103_k96_globals &g,
    raw_sm103_output_t output0,
    raw_sm103_output_t output1,
    raw_sm103_nv_scale_t a_scale,
    raw_sm103_nv_scale_t b_scale,
    raw_sm103_group2_tail_a_tile &a,
    raw_sm103_group2_tail_b_tile &b,
    raw_sm103_nv_scale_smem_t &scale_smem,
    semaphore &done
) {
    const int rank = cluster_ctarank();
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    raw_sm103_load_fp4_tile_repeated(a, g.a_fp4);
    raw_sm103_load_fp4_tile_repeated(b, g.b_fp4);
    raw_sm103_fill_nvfp4_scale_smem(scale_smem);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");

    if (rank == 0 && threadIdx.x < 32) {
        load_mxnv_scale_async<2>(a_scale, scale_smem);
        load_mxnv_scale_async<2>(b_scale, scale_smem);
        tensor_load_wait();
        tensor_before_thread_sync();
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");

    st_descriptor<raw_sm103_group2_tail_a_tile, transpose::N> a_desc(a);
    st_descriptor<raw_sm103_group2_tail_b_tile, transpose::N> b_desc(b);
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const auto make_k96_desc = [&](uint64_t current, uint64_t next) {
        return (current & ~lbo_field_mask) |
               ((next & 0x3fffull) << 16) |
               absolute_lbo_mode;
    };
    const auto advance = [](uint64_t desc, int bytes) {
        return desc + detail::matrix_descriptor_encode(bytes);
    };
    const uint64_t ad_k96 = make_k96_desc(
        a_desc.chunk_descriptor(0), a_desc.chunk_descriptor(1));
    const uint64_t bd_k96 = make_k96_desc(
        b_desc.chunk_descriptor(0), b_desc.chunk_descriptor(1));
    const uint64_t ad_tail = advance(a_desc.chunk_descriptor(0), 48);
    const uint64_t bd_tail = advance(b_desc.chunk_descriptor(0), 48);

    uint64_t start_cycle = 0;
    if (threadIdx.x == 0) {
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
    }
    if (rank == 0 && threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            raw_sm103_issue_nvfp4_ultra_k96_group2<0>(
                output0, ad_k96, bd_k96, a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_k64_group2<1, 0>(
                output0, ad_tail, bd_tail, a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_ultra_k96_group2<0>(
                output1, ad_k96, bd_k96, a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_k64_group2<1, 0>(
                output1, ad_tail, bd_tail, a_scale.addr, b_scale.addr);
            #pragma unroll 1
            for (int iter = 2; iter < g.iterations; iter += 2) {
                raw_sm103_issue_nvfp4_ultra_k96_group2<1>(
                    output0, ad_k96, bd_k96,
                    a_scale.addr, b_scale.addr);
                raw_sm103_issue_nvfp4_k64_group2<1, 0>(
                    output0, ad_tail, bd_tail,
                    a_scale.addr, b_scale.addr);
                raw_sm103_issue_nvfp4_ultra_k96_group2<1>(
                    output1, ad_k96, bd_k96,
                    a_scale.addr, b_scale.addr);
                raw_sm103_issue_nvfp4_k64_group2<1, 0>(
                    output1, ad_tail, bd_tail,
                    a_scale.addr, b_scale.addr);
            }
            detail::tcgen05::commit<2>(done);
            wait(done, 0);
        }
        __syncwarp();
        tensor_after_thread_sync();
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n");
    if (threadIdx.x == 0) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
}

template <bool Ultra>
__device__ __forceinline__ void raw_sm103_k96_probe_body(
    const raw_sm103_k96_globals &g,
    raw_sm103_output_t output0,
    raw_sm103_output_t output1,
    raw_sm103_scale_t a_scale,
    raw_sm103_scale_t b_scale,
    raw_sm103_scaled_tile &a,
    raw_sm103_scaled_tile &b,
    semaphore &done
) {
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    raw_sm103_load_fp4_tile(a, g.a_fp4);
    raw_sm103_load_fp4_tile(b, g.b_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    if (threadIdx.x < 32) {
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            a_scale, 0x7f7f7f7fu);
        fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
            b_scale, 0x7f7f7f7fu);
        fp4pv_tmem_store_wait();
    }
    __syncthreads();

    st_descriptor<raw_sm103_scaled_tile, transpose::N> a_desc(a);
    st_descriptor<raw_sm103_scaled_tile, transpose::N> b_desc(b);
    uint64_t start_cycle = 0;
    if (threadIdx.x == 0) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        raw_sm103_issue_k192<0, Ultra>(
            output0, a_desc, b_desc, a_scale, b_scale);
        raw_sm103_issue_k192<0, Ultra>(
            output1, a_desc, b_desc, a_scale, b_scale);
        #pragma unroll 1
        for (int iter = 2; iter < g.iterations; iter += 2) {
            raw_sm103_issue_k192<1, Ultra>(
                output0, a_desc, b_desc, a_scale, b_scale);
            raw_sm103_issue_k192<1, Ultra>(
                output1, a_desc, b_desc, a_scale, b_scale);
        }
        detail::tcgen05::commit<1>(done);
    }
    if (threadIdx.x < 32) {
        wait(done, 0);
        tensor_after_thread_sync();
    }
    if (threadIdx.x == 0) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
}

__device__ __forceinline__ void raw_sm103_k96_nvfp4_ultra_probe_body(
    const raw_sm103_k96_globals &g,
    raw_sm103_output_t output0,
    raw_sm103_output_t output1,
    raw_sm103_nv_scale_t a_scale,
    raw_sm103_nv_scale_t b_scale,
    raw_sm103_scaled_tile &a,
    raw_sm103_scaled_tile &b,
    semaphore &done
) {
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    raw_sm103_load_fp4_tile(a, g.a_fp4);
    raw_sm103_load_fp4_tile(b, g.b_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    if (threadIdx.x < 32) {
        raw_sm103_fill_nvfp4_scale(a_scale.addr);
        raw_sm103_fill_nvfp4_scale(b_scale.addr);
        fp4pv_tmem_store_wait();
    }
    __syncthreads();

    st_descriptor<raw_sm103_scaled_tile, transpose::N> a_desc(a);
    st_descriptor<raw_sm103_scaled_tile, transpose::N> b_desc(b);
    uint64_t start_cycle = 0;
    if (threadIdx.x == 0) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        raw_sm103_issue_k192_nvfp4_ultra<0>(
            output0, a_desc, b_desc, a_scale, b_scale);
        raw_sm103_issue_k192_nvfp4_ultra<0>(
            output1, a_desc, b_desc, a_scale, b_scale);
        #pragma unroll 1
        for (int iter = 2; iter < g.iterations; iter += 2) {
            raw_sm103_issue_k192_nvfp4_ultra<1>(
                output0, a_desc, b_desc, a_scale, b_scale);
            raw_sm103_issue_k192_nvfp4_ultra<1>(
                output1, a_desc, b_desc, a_scale, b_scale);
        }
        detail::tcgen05::commit<1>(done);
    }
    if (threadIdx.x < 32) {
        wait(done, 0);
        tensor_after_thread_sync();
    }
    if (threadIdx.x == 0) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
}

__device__ __forceinline__ void raw_sm103_k96_nvfp4_rotating_probe_body(
    const raw_sm103_k96_globals &g,
    raw_sm103_output_t output0,
    raw_sm103_output_t output1,
    raw_sm103_nv_scale_wide_t a_scale,
    raw_sm103_nv_scale_wide_t b_scale,
    raw_sm103_ultra_tile &a,
    raw_sm103_ultra_tile &b,
    semaphore &done
) {
    if (threadIdx.x == 0) {
        init_semaphore(done, 0, 1);
    }
    raw_sm103_load_fp4_tile_repeated(a, g.a_fp4);
    raw_sm103_load_fp4_tile_repeated(b, g.b_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int col = 0; col < 48; col += 16) {
            raw_sm103_fill_nvfp4_scale(a_scale.addr + col);
            raw_sm103_fill_nvfp4_scale(b_scale.addr + col);
        }
        fp4pv_tmem_store_wait();
    }
    __syncthreads();

    st_descriptor<raw_sm103_ultra_tile, transpose::N> a_desc(a);
    st_descriptor<raw_sm103_ultra_tile, transpose::N> b_desc(b);
    constexpr uint64_t lbo_field_mask = 0x3fffull << 16;
    constexpr uint64_t absolute_lbo_mode = 1ull << 52;
    const auto advance = [](uint64_t desc, int bytes) {
        return desc + detail::matrix_descriptor_encode(bytes);
    };
    const auto make_k96_desc = [&](uint64_t current, uint64_t next) {
        return (current & ~lbo_field_mask) |
               ((next & 0x3fffull) << 16) |
               absolute_lbo_mode;
    };
    const uint64_t a_page0 = a_desc.chunk_descriptor(0);
    const uint64_t a_page1 = a_desc.chunk_descriptor(4);
    const uint64_t a_page2 = a_desc.chunk_descriptor(8);
    const uint64_t b_page0 = b_desc.chunk_descriptor(0);
    const uint64_t b_page1 = b_desc.chunk_descriptor(4);
    const uint64_t b_page2 = b_desc.chunk_descriptor(8);
    const uint64_t ad0 = make_k96_desc(advance(a_page0, 0), a_page0);
    const uint64_t ad1 = make_k96_desc(advance(a_page0, 48), a_page0);
    const uint64_t ad2 = make_k96_desc(advance(a_page0, 96), a_page1);
    const uint64_t ad3 = make_k96_desc(advance(a_page1, 16), a_page1);
    const uint64_t ad4 = make_k96_desc(advance(a_page1, 64), a_page1);
    const uint64_t ad5 = make_k96_desc(advance(a_page1, 112), a_page2);
    const uint64_t ad6 = make_k96_desc(advance(a_page2, 32), a_page2);
    const uint64_t ad7 = make_k96_desc(advance(a_page2, 80), a_page2);
    const uint64_t bd0 = make_k96_desc(advance(b_page0, 0), b_page0);
    const uint64_t bd1 = make_k96_desc(advance(b_page0, 48), b_page0);
    const uint64_t bd2 = make_k96_desc(advance(b_page0, 96), b_page1);
    const uint64_t bd3 = make_k96_desc(advance(b_page1, 16), b_page1);
    const uint64_t bd4 = make_k96_desc(advance(b_page1, 64), b_page1);
    const uint64_t bd5 = make_k96_desc(advance(b_page1, 112), b_page2);
    const uint64_t bd6 = make_k96_desc(advance(b_page2, 32), b_page2);
    const uint64_t bd7 = make_k96_desc(advance(b_page2, 80), b_page2);

    uint64_t start_cycle = 0;
    if (threadIdx.x == 0) {
        tensor_before_thread_sync();
        start_cycle = clock64();
        g.start_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        raw_sm103_issue_nvfp4_ultra_k96<0>(
            output0, ad0, bd0, a_scale.addr, b_scale.addr);
        raw_sm103_issue_nvfp4_ultra_k96<0>(
            output1, ad1, bd1, a_scale.addr + 6, b_scale.addr + 6);
        raw_sm103_issue_nvfp4_ultra_k96<1>(
            output0, ad2, bd2, a_scale.addr + 12, b_scale.addr + 12);
        raw_sm103_issue_nvfp4_ultra_k96<1>(
            output1, ad3, bd3, a_scale.addr + 18, b_scale.addr + 18);
        raw_sm103_issue_nvfp4_ultra_k96<1>(
            output0, ad4, bd4, a_scale.addr + 24, b_scale.addr + 24);
        raw_sm103_issue_nvfp4_ultra_k96<1>(
            output1, ad5, bd5, a_scale.addr + 30, b_scale.addr + 30);
        raw_sm103_issue_nvfp4_ultra_k96<1>(
            output0, ad6, bd6, a_scale.addr + 36, b_scale.addr + 36);
        raw_sm103_issue_nvfp4_ultra_k96<1>(
            output1, ad7, bd7, a_scale.addr + 42, b_scale.addr + 42);
        #pragma unroll 1
        for (int iter = 4; iter < g.iterations; iter += 4) {
            raw_sm103_issue_nvfp4_ultra_k96<1>(
                output0, ad0, bd0, a_scale.addr, b_scale.addr);
            raw_sm103_issue_nvfp4_ultra_k96<1>(
                output1, ad1, bd1, a_scale.addr + 6, b_scale.addr + 6);
            raw_sm103_issue_nvfp4_ultra_k96<1>(
                output0, ad2, bd2, a_scale.addr + 12, b_scale.addr + 12);
            raw_sm103_issue_nvfp4_ultra_k96<1>(
                output1, ad3, bd3, a_scale.addr + 18, b_scale.addr + 18);
            raw_sm103_issue_nvfp4_ultra_k96<1>(
                output0, ad4, bd4, a_scale.addr + 24, b_scale.addr + 24);
            raw_sm103_issue_nvfp4_ultra_k96<1>(
                output1, ad5, bd5, a_scale.addr + 30, b_scale.addr + 30);
            raw_sm103_issue_nvfp4_ultra_k96<1>(
                output0, ad6, bd6, a_scale.addr + 36, b_scale.addr + 36);
            raw_sm103_issue_nvfp4_ultra_k96<1>(
                output1, ad7, bd7, a_scale.addr + 42, b_scale.addr + 42);
        }
        detail::tcgen05::commit<1>(done);
    }
    if (threadIdx.x < 32) {
        wait(done, 0);
        tensor_after_thread_sync();
    }
    if (threadIdx.x == 0) {
        const uint64_t end_cycle = clock64();
        g.end_globaltimer[blockIdx.x] = raw_sm103_globaltimer();
        g.cycles[blockIdx.x] = end_cycle - start_cycle;
        g.smids[blockIdx.x] = smid();
    }
}

extern "C" __launch_bounds__(128, 1)
__global__ void raw_sm103_k96_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<1, 1> tm_alloc{};
    raw_sm103_output_t output0 = tm_alloc.allocate<raw_sm103_output_t>(0);
    raw_sm103_output_t output1 = tm_alloc.allocate<raw_sm103_output_t>(128);
    raw_sm103_scale_t a_scale = tm_alloc.allocate<raw_sm103_scale_t>(256);
    raw_sm103_scale_t b_scale = tm_alloc.allocate<raw_sm103_scale_t>(272);
    __shared__ raw_sm103_scaled_tile a;
    __shared__ raw_sm103_scaled_tile b;
    __shared__ semaphore done;
    raw_sm103_k96_probe_body<false>(
        g, output0, output1, a_scale, b_scale, a, b, done);
}

extern "C" __launch_bounds__(128, 1)
__global__ void raw_sm103_k96_ultra_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<1, 1> tm_alloc{};
    raw_sm103_output_t output0 = tm_alloc.allocate<raw_sm103_output_t>(0);
    raw_sm103_output_t output1 = tm_alloc.allocate<raw_sm103_output_t>(128);
    raw_sm103_scale_t a_scale = tm_alloc.allocate<raw_sm103_scale_t>(256);
    raw_sm103_scale_t b_scale = tm_alloc.allocate<raw_sm103_scale_t>(272);
    __shared__ raw_sm103_scaled_tile a;
    __shared__ raw_sm103_scaled_tile b;
    __shared__ semaphore done;
    raw_sm103_k96_probe_body<true>(
        g, output0, output1, a_scale, b_scale, a, b, done);
}

extern "C" __launch_bounds__(128, 1)
__global__ void raw_sm103_k96_nvfp4_ultra_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<1, 1> tm_alloc{};
    raw_sm103_output_t output0 = tm_alloc.allocate<raw_sm103_output_t>(0);
    raw_sm103_output_t output1 = tm_alloc.allocate<raw_sm103_output_t>(128);
    raw_sm103_nv_scale_t a_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(256);
    raw_sm103_nv_scale_t b_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(272);
    __shared__ raw_sm103_scaled_tile a;
    __shared__ raw_sm103_scaled_tile b;
    __shared__ semaphore done;
    raw_sm103_k96_nvfp4_ultra_probe_body(
        g, output0, output1, a_scale, b_scale, a, b, done);
}

extern "C" __launch_bounds__(128, 1)
__global__ void raw_sm103_k96_nvfp4_rotating_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<1, 1> tm_alloc{};
    raw_sm103_output_t output0 = tm_alloc.allocate<raw_sm103_output_t>(0);
    raw_sm103_output_t output1 = tm_alloc.allocate<raw_sm103_output_t>(128);
    raw_sm103_nv_scale_wide_t a_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_wide_t>(256);
    raw_sm103_nv_scale_wide_t b_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_wide_t>(304);
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    raw_sm103_ultra_tile &a = allocator.allocate<raw_sm103_ultra_tile>();
    raw_sm103_ultra_tile &b = allocator.allocate<raw_sm103_ultra_tile>();
    __shared__ semaphore done;
    raw_sm103_k96_nvfp4_rotating_probe_body(
        g, output0, output1, a_scale, b_scale, a, b, done);
}

extern "C" __launch_bounds__(128, 2)
__global__ void raw_sm103_k96_nvfp4_rotating_2cta_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<2, 1> tm_alloc{};
    raw_sm103_output_t output = tm_alloc.allocate<raw_sm103_output_t>(0);
    raw_sm103_nv_scale_wide_t a_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_wide_t>(128);
    raw_sm103_nv_scale_wide_t b_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_wide_t>(176);
    extern __shared__ int __shm[];
    // CUDA places this dynamic region behind the kernel's 1 KiB alignment
    // reservation. Using it directly avoids charging another alignment page
    // per CTA, which would reduce the B300 shared-memory residency to one.
    raw_sm103_ultra_tile &a =
        *reinterpret_cast<raw_sm103_ultra_tile *>(&__shm[0]);
    // The raw gate only measures issue rate, so both operands may share one
    // immutable page. Keeping the duplicate page would force one CTA/SM via
    // shared memory and defeat the residency experiment.
    raw_sm103_ultra_tile &b = a;
    __shared__ semaphore done;
    raw_sm103_k96_nvfp4_rotating_probe_body(
        g, output, output, a_scale, b_scale, a, b, done);
}

extern "C" __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
__global__ void raw_sm103_k96_nvfp4_group2_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<1, 2> tm_alloc{};
    raw_sm103_output_t output0 = tm_alloc.allocate<raw_sm103_output_t>(0);
    raw_sm103_output_t output1 = tm_alloc.allocate<raw_sm103_output_t>(128);
    raw_sm103_nv_scale_t a_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(256);
    raw_sm103_nv_scale_t b_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(272);
    __shared__ raw_sm103_scaled_tile a;
    __shared__ raw_sm103_group2_b_tile b;
    __shared__ raw_sm103_nv_scale_smem_t scale_smem;
    __shared__ semaphore done;
    raw_sm103_k96_nvfp4_group2_probe_body(
        g, output0, output1, a_scale, b_scale,
        a, b, scale_smem, done);
}

extern "C" __launch_bounds__(128, 1)
__global__ void raw_sm103_nvfp4_k128_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<1, 1> tm_alloc{};
    raw_sm103_output_t output0 = tm_alloc.allocate<raw_sm103_output_t>(0);
    raw_sm103_output_t output1 = tm_alloc.allocate<raw_sm103_output_t>(128);
    raw_sm103_nv_scale_t a_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(256);
    raw_sm103_nv_scale_t b_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(272);
    __shared__ raw_sm103_scaled_tile a;
    __shared__ raw_sm103_scaled_tile b;
    __shared__ semaphore done;
    raw_sm103_nvfp4_k128_probe_body(
        g, output0, output1, a_scale, b_scale, a, b, done);
}

extern "C" __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
__global__ void raw_sm103_nvfp4_group2_k128_mixed_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<1, 2> tm_alloc{};
    raw_sm103_output_t output0 = tm_alloc.allocate<raw_sm103_output_t>(0);
    raw_sm103_output_t output1 = tm_alloc.allocate<raw_sm103_output_t>(128);
    raw_sm103_nv_scale_t a_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(256);
    raw_sm103_nv_scale_t b_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(272);
    __shared__ raw_sm103_group2_tail_a_tile a;
    __shared__ raw_sm103_group2_tail_b_tile b;
    __shared__ raw_sm103_nv_scale_smem_t scale_smem;
    __shared__ semaphore done;
    raw_sm103_nvfp4_group2_k128_mixed_body(
        g, output0, output1, a_scale, b_scale,
        a, b, scale_smem, done);
}

extern "C" __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
__global__ void raw_sm103_group2_n192_joint_probe(
    const __grid_constant__ raw_sm103_k96_globals g
) {
    tensor_allocator<1, 2> tm_alloc{};
    raw_sm103_group2_n192_score_t score =
        tm_alloc.allocate<raw_sm103_group2_n192_score_t>(0);
    raw_sm103_output_t output =
        tm_alloc.allocate<raw_sm103_output_t>(192);
    raw_sm103_nv_scale_t q_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(320);
    raw_sm103_nv_scale_t k_scale =
        tm_alloc.allocate<raw_sm103_nv_scale_t>(336);
    raw_sm103_scale_t p_scale =
        tm_alloc.allocate<raw_sm103_scale_t>(352);
    raw_sm103_scale_t v_scale =
        tm_alloc.allocate<raw_sm103_scale_t>(368);
    static_assert(384 <= tensor_allocator<1, 2>::cols);

    __shared__ raw_sm103_scaled_tile q;
    __shared__ raw_sm103_group2_n192_k_tile k;
    __shared__ raw_sm103_group2_n192_v_tile v0;
    __shared__ raw_sm103_group2_n192_v_tile v1;
    __shared__ raw_sm103_nv_scale_smem_t qk_scale_smem;
    __shared__ raw_sm103_mx_scale_smem_t pv_scale_smem;
    __shared__ semaphore qk_done;
    __shared__ semaphore pv_done;
    __shared__ semaphore qk_remote_ready;
    __shared__ semaphore p_remote_ready;
    raw_sm103_group2_n192_joint_probe_body(
        g, score, output, q_scale, k_scale, p_scale, v_scale,
        q, k, v0, v1, qk_scale_smem, pv_scale_smem,
        qk_done, pv_done, qk_remote_ready, p_remote_ready);
}

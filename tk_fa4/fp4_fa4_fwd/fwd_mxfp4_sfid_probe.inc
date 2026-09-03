#pragma once

#ifndef TK_FA4_MXFP4_SFID_PROBE_BUILD
#define TK_FA4_MXFP4_SFID_PROBE_BUILD 0
#endif

#if TK_FA4_MXFP4_SFID_PROBE_BUILD

template <int A_SFID, int B_SFID>
__device__ static inline constexpr uint32_t
mxfp4_sfid_independent_descriptor() {
    static_assert(A_SFID == 0 || A_SFID == 2,
                  "NVFP4 A SFID must be 0 or 2");
    static_assert(B_SFID == 0 || B_SFID == 2,
                  "NVFP4 B SFID must be 0 or 2");
    constexpr uint32_t base =
        detail::tcgen05::instruction_descriptor<
            float, fp4e2m1_2, fp8e8m0, 128, 128, false, 0>();
    constexpr uint32_t sfid_mask = (3u << 4) | (3u << 29);
    return (base & ~sfid_mask)
        | (static_cast<uint32_t>(B_SFID) << 4)
        | (static_cast<uint32_t>(A_SFID) << 29);
}

struct config_mxfp4_sfid_probe {
    static constexpr int Mb = 128;
    static constexpr int Nb = 64;
    static constexpr int Dvo = 128;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 128;
};

struct globals_mxfp4_sfid_probe {
    using payload_tile = st_fp4e2m1_2<128, 32>;
    using scale_tile = st_fp8e8m0<32, 16, false>;

    const uint8_t *a_fp4;
    const uint8_t *b_fp4;
    float *o;
    int64_t *trace;
    uint32_t a_word;
    uint32_t b_word;
    uint32_t canary_word;
    uint32_t replacement_a_word;
    uint32_t replacement_b_word;
    int canary_col;
    int canary_target;
    int overwrite_stage;

    __host__ __inline__ dim3 grid() const { return dim3(1); }
    __host__ __inline__ dim3 block() const {
        return dim3(config_mxfp4_sfid_probe::NUM_THREADS);
    }
    __host__ __inline__ int dynamic_shared_memory() const {
        return 48 * 1024;
    }
};

template <typename Tile>
__device__ inline void load_mxfp4_sfid_payload(
    Tile &tile,
    const uint8_t *source
) {
    static_assert(Tile::rows == 128 && Tile::cols == 32,
                  "SFID probe payload is dense logical K64");
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * STORES_PER_ROW;
         linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            source + row * Tile::cols + col);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(Tile::idx(base, {row, col})),
            "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename ScaleTT>
__device__ inline void mxfp4_sfid_store_column_warp(
    ScaleTT &scale,
    int col,
    uint32_t word
) {
    const uint32_t address = scale.addr + col;
    asm volatile(
        "{tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};}" ::
        "r"(address), "r"(word));
}

template <typename ScaleTT>
__device__ inline void mxfp4_sfid_fill_scale_warp(
    ScaleTT &scale,
    uint32_t word
) {
    fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
        scale, 0, word, word, word, word);
}

template <int A_SFID, int B_SFID>
__launch_bounds__(config_mxfp4_sfid_probe::NUM_THREADS, 1)
__global__ void kernel_mxfp4_sfid_probe(
    const __grid_constant__ globals_mxfp4_sfid_probe g
) {
    using C = config_mxfp4_sfid_probe;
    using G = globals_mxfp4_sfid_probe;
    using output_t = tt<float, 128, 128>;
    using scale_t = full_tt_fp8e8m0<16>;

    tensor_allocator<1, 1> tm_alloc{};
    output_t output = tm_alloc.template allocate<output_t>(0);
    scale_t a_scale = tm_alloc.template allocate<scale_t>(128);
    scale_t b_scale = tm_alloc.template allocate<scale_t>(144);
    static_assert(output_t::cols + 2 * 16 <= 512,
                  "SFID probe TMEM map must fit one SM allocation");

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::payload_tile &a_smem =
        allocator.allocate<typename G::payload_tile>();
    typename G::payload_tile &b_smem =
        allocator.allocate<typename G::payload_tile>();
    typename G::scale_tile &a_scale_smem =
        allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &b_scale_smem =
        allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &replacement_a_scale_smem =
        allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &replacement_b_scale_smem =
        allocator.allocate<typename G::scale_tile>();
    __shared__ semaphore scales_ready;
    __shared__ semaphore replacement_ready;
    __shared__ semaphore issued;
    __shared__ semaphore done;
    if (threadIdx.x == 0) {
        init_semaphore(scales_ready, 0, 1);
        init_semaphore(replacement_ready, 0, 1);
        init_semaphore(issued, 0, 1);
        init_semaphore(done, 0, 1);
    }
    load_mxfp4_sfid_payload(a_smem, g.a_fp4);
    load_mxfp4_sfid_payload(b_smem, g.b_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();

    const int local_warp = warpgroup::warpid();
    const bool issue_warp = local_warp == 0;
    const bool overwrite_warp = local_warp == 1;
    const bool leader = issue_warp && warp::elect_leader();
    if (issue_warp) {
        fp4pv_store_mxfp4_scale_word4_swizzled(
            a_scale_smem, warp::laneid(),
            g.a_word, g.a_word, g.a_word, g.a_word);
        fp4pv_store_mxfp4_scale_word4_swizzled(
            b_scale_smem, warp::laneid(),
            g.b_word, g.b_word, g.b_word, g.b_word);
    } else if (overwrite_warp) {
        fp4pv_store_mxfp4_scale_word4_swizzled(
            replacement_a_scale_smem, warp::laneid(),
            g.replacement_a_word, g.replacement_a_word,
            g.replacement_a_word, g.replacement_a_word);
        fp4pv_store_mxfp4_scale_word4_swizzled(
            replacement_b_scale_smem, warp::laneid(),
            g.replacement_b_word, g.replacement_b_word,
            g.replacement_b_word, g.replacement_b_word);
    }
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();
    if (leader) {
        load_mxnv_scale_async<1>(a_scale, a_scale_smem);
        load_mxnv_scale_async<1>(b_scale, b_scale_smem);
        detail::tcgen05::commit<1>(scales_ready);
    }
    if (issue_warp) {
        wait(scales_ready, 0);
        tensor_after_thread_sync();
        if (g.canary_col >= 0) {
            if (g.canary_target & 1) {
                mxfp4_sfid_store_column_warp(
                    a_scale, g.canary_col, g.canary_word);
            }
            if (g.canary_target & 2) {
                mxfp4_sfid_store_column_warp(
                    b_scale, g.canary_col, g.canary_word);
            }
            fp4pv_tmem_store_wait();
        }
    }
    __syncthreads();
    if (overwrite_warp && g.overwrite_stage == 1) {
        if (warp::elect_leader()) {
            load_mxnv_scale_async<1>(a_scale, replacement_a_scale_smem);
            load_mxnv_scale_async<1>(b_scale, replacement_b_scale_smem);
            detail::tcgen05::commit<1>(replacement_ready);
        }
        wait(replacement_ready, 0);
        tensor_after_thread_sync();
    }
    __syncthreads();

    st_descriptor<typename G::payload_tile, transpose::N> a_desc(a_smem);
    st_descriptor<typename G::payload_tile, transpose::N> b_desc(b_smem);
    constexpr uint32_t idesc =
        mxfp4_sfid_independent_descriptor<A_SFID, B_SFID>();
    if (issue_warp) {
        tensor_before_thread_sync();
    }
    if (leader) {
        detail::tcgen05::st_st<fp4e2m1_2, fp8e8m0, 0, 1, 32>(
            output.addr,
            a_desc.chunk_descriptor(0),
            b_desc.chunk_descriptor(0),
            a_scale.addr,
            b_scale.addr,
            idesc);
        arrive(issued);
        detail::tcgen05::commit<1>(done);
        g.trace[0] = static_cast<int64_t>(output.addr);
        g.trace[1] = static_cast<int64_t>(a_scale.addr);
        g.trace[2] = static_cast<int64_t>(b_scale.addr);
        g.trace[3] = static_cast<int64_t>(idesc);
        g.trace[4] = A_SFID;
        g.trace[5] = B_SFID;
        g.trace[6] = g.canary_col;
        g.trace[7] = g.overwrite_stage;
    }

    if (overwrite_warp && g.overwrite_stage == 2) {
        wait(issued, 0);
        if (warp::elect_leader()) {
            load_mxnv_scale_async<1>(a_scale, replacement_a_scale_smem);
            load_mxnv_scale_async<1>(b_scale, replacement_b_scale_smem);
            detail::tcgen05::commit<1>(replacement_ready);
        }
        wait(replacement_ready, 0);
        tensor_after_thread_sync();
    } else if (overwrite_warp && g.overwrite_stage == 3) {
        wait(done, 0);
        tensor_after_thread_sync();
        if (warp::elect_leader()) {
            load_mxnv_scale_async<1>(a_scale, replacement_a_scale_smem);
            load_mxnv_scale_async<1>(b_scale, replacement_b_scale_smem);
            detail::tcgen05::commit<1>(replacement_ready);
        }
        wait(replacement_ready, 0);
        tensor_after_thread_sync();
    }

    if (issue_warp) {
        wait(done, 0);
    }
    __syncthreads();
    tensor_after_thread_sync();
    const int row = warpgroup::laneid();
    #pragma unroll
    for (int col = 0; col < 128; col += 16) {
        float2 values[8];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(values[0].x), "=f"(values[0].y),
              "=f"(values[1].x), "=f"(values[1].y),
              "=f"(values[2].x), "=f"(values[2].y),
              "=f"(values[3].x), "=f"(values[3].y),
              "=f"(values[4].x), "=f"(values[4].y),
              "=f"(values[5].x), "=f"(values[5].y),
              "=f"(values[6].x), "=f"(values[6].y),
              "=f"(values[7].x), "=f"(values[7].y)
            : "r"(output.addr + ((local_warp * 32) << 16) + col));
        tensor_load_wait();
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            reinterpret_cast<float2 *>(&g.o[row * 128 + col])[i] = values[i];
        }
    }
    __syncthreads();
}

template <int A_SFID, int B_SFID>
static void launch_mxfp4_sfid_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor o,
    at::Tensor trace,
    uint32_t a_word,
    uint32_t b_word,
    uint32_t canary_word,
    uint32_t replacement_a_word,
    uint32_t replacement_b_word,
    int canary_col,
    int canary_target,
    int overwrite_stage
) {
    using C = config_mxfp4_sfid_probe;
    globals_mxfp4_sfid_probe g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        o.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        a_word,
        b_word,
        canary_word,
        replacement_a_word,
        replacement_b_word,
        canary_col,
        canary_target,
        overwrite_stage,
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_sfid_probe<A_SFID, B_SFID>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_mxfp4_sfid_probe<A_SFID, B_SFID>, g));
}

void dispatch_mxfp4_sfid_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor o,
    at::Tensor trace,
    int64_t a_word,
    int64_t b_word,
    int64_t canary_word,
    int64_t replacement_a_word,
    int64_t replacement_b_word,
    int a_sfid,
    int b_sfid,
    int canary_col,
    int canary_target,
    int overwrite_stage
) {
    CHECK_INPUT(a_fp4);
    CHECK_INPUT(b_fp4);
    CHECK_INPUT(o);
    CHECK_INPUT(trace);
    TORCH_CHECK(a_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "a_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(b_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "b_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(o.dtype() == at::ScalarType::Float,
                "o must be float32");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long,
                "trace must be int64");
    TORCH_CHECK(a_fp4.dim() == 2 && a_fp4.size(0) == 128
                    && a_fp4.size(1) == 32,
                "a_fp4 must have shape (128,32 packed K64 bytes)");
    TORCH_CHECK(b_fp4.dim() == 2 && b_fp4.size(0) == 128
                    && b_fp4.size(1) == 32,
                "b_fp4 must have shape (128,32 packed K64 bytes)");
    TORCH_CHECK(o.dim() == 2 && o.size(0) == 128 && o.size(1) == 128,
                "o must have shape (128,128)");
    TORCH_CHECK(trace.numel() >= 32, "trace must contain at least 32 int64 values");
    TORCH_CHECK((a_sfid == 0 || a_sfid == 2)
                    && (b_sfid == 0 || b_sfid == 2),
                "A and B SFIDs must independently be 0 or 2");
    TORCH_CHECK(canary_col >= -1 && canary_col < 16,
                "canary_col must be -1 or a fragment offset in [0,16)");
    TORCH_CHECK(canary_target >= 0 && canary_target <= 3,
                "canary_target is a bitmask: 1=A, 2=B");
    TORCH_CHECK(overwrite_stage >= 0 && overwrite_stage <= 3,
                "overwrite_stage must be none/before/after-issue/after-completion");

    const uint32_t aw = static_cast<uint32_t>(a_word);
    const uint32_t bw = static_cast<uint32_t>(b_word);
    const uint32_t cw = static_cast<uint32_t>(canary_word);
    const uint32_t raw = static_cast<uint32_t>(replacement_a_word);
    const uint32_t rbw = static_cast<uint32_t>(replacement_b_word);
    if (a_sfid == 0 && b_sfid == 0) {
        launch_mxfp4_sfid_probe<0, 0>(
            a_fp4, b_fp4, o, trace, aw, bw, cw, raw, rbw,
            canary_col, canary_target, overwrite_stage);
    } else if (a_sfid == 0 && b_sfid == 2) {
        launch_mxfp4_sfid_probe<0, 2>(
            a_fp4, b_fp4, o, trace, aw, bw, cw, raw, rbw,
            canary_col, canary_target, overwrite_stage);
    } else if (a_sfid == 2 && b_sfid == 0) {
        launch_mxfp4_sfid_probe<2, 0>(
            a_fp4, b_fp4, o, trace, aw, bw, cw, raw, rbw,
            canary_col, canary_target, overwrite_stage);
    } else {
        launch_mxfp4_sfid_probe<2, 2>(
            a_fp4, b_fp4, o, trace, aw, bw, cw, raw, rbw,
            canary_col, canary_target, overwrite_stage);
    }
}

struct globals_mxfp4_fine_sfid_probe {
    using payload_tile = st_fp4e2m1_2<128, 32>;
    using scale_tile = st_fp8e8m0<32, 16, false>;

    const uint8_t *q_fp4;
    const uint8_t *k_fp4;
    const uint8_t *p_fp4;
    const uint8_t *v_fp4;
    const int32_t *q_scale;
    const int32_t *k_scale;
    const int32_t *p_scale;
    const int32_t *v_scale;
    float *o;
    int64_t *trace;
    int epochs;

    __host__ __inline__ dim3 grid() const { return dim3(1); }
    __host__ __inline__ dim3 block() const { return dim3(128); }
    __host__ __inline__ int dynamic_shared_memory() const { return 32 * 1024; }
};

template <typename ScaleTile>
__device__ __forceinline__ void mxfp4_fine_sfid_write_scale_tile(
    ScaleTile &tile,
    const int32_t *low_source,
    const int32_t *high_source,
    int low_epoch,
    int high_epoch
) {
    const int lane = warp::laneid();
    const int low_base = (low_epoch * 32 + lane) * 4;
    const int high_base = (high_epoch * 32 + lane) * 4;
    uint32_t words[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        const uint32_t low = static_cast<uint32_t>(low_source[low_base + word]) & 0xffffu;
        const uint32_t high = static_cast<uint32_t>(high_source[high_base + word]) & 0xffffu;
        words[word] = low | (high << 16);
    }
    fp4pv_store_mxfp4_scale_word4_swizzled(
        tile, lane, words[0], words[1], words[2], words[3]);
}

template <bool PACKED>
__launch_bounds__(128, 1)
__global__ void kernel_mxfp4_fine_sfid_probe(
    const __grid_constant__ globals_mxfp4_fine_sfid_probe g
) {
    using G = globals_mxfp4_fine_sfid_probe;
    using output_t = tt<float, 128, 128>;
    using scale_t = full_tt_fp8e8m0<16>;

    tensor_allocator<1, 1> tm_alloc{};
    output_t output = tm_alloc.template allocate<output_t>(0);
    scale_t scale0 = tm_alloc.template allocate<scale_t>(128);
    scale_t scale1 = tm_alloc.template allocate<scale_t>(144);
    scale_t scale2 = tm_alloc.template allocate<scale_t>(160);
    scale_t scale3 = tm_alloc.template allocate<scale_t>(176);
    static_assert(!PACKED || 128 + 2 * 16 <= 512,
                  "packed fine-scale probe uses two aligned scale fragments");
    static_assert(PACKED || 128 + 4 * 16 <= 512,
                  "separate fine-scale probe uses four aligned scale fragments");

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::payload_tile &q_smem = allocator.allocate<typename G::payload_tile>();
    typename G::payload_tile &k_smem = allocator.allocate<typename G::payload_tile>();
    typename G::payload_tile &p_smem = allocator.allocate<typename G::payload_tile>();
    typename G::payload_tile &v_smem = allocator.allocate<typename G::payload_tile>();
    typename G::scale_tile &scale_smem0 = allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &scale_smem1 = allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &scale_smem2 = allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &scale_smem3 = allocator.allocate<typename G::scale_tile>();

    __shared__ semaphore scale_copy_done;
    __shared__ semaphore mma_done;
    if (threadIdx.x == 0) {
        init_semaphore(scale_copy_done, 0, 1);
        init_semaphore(mma_done, 0, 1);
    }
    load_mxfp4_sfid_payload(q_smem, g.q_fp4);
    load_mxfp4_sfid_payload(k_smem, g.k_fp4);
    load_mxfp4_sfid_payload(p_smem, g.p_fp4);
    load_mxfp4_sfid_payload(v_smem, g.v_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();

    const int local_warp = warpgroup::warpid();
    const bool issue_warp = local_warp == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    int scale_phase = 0;
    int command_phase = 0;
    int scale_copies = 0;
    int qk_commands = 0;
    int pv_commands = 0;

    auto copy_scales = [&](int count) {
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();
        if (issue_lane) {
            load_mxnv_scale_async<1>(scale0, scale_smem0);
            load_mxnv_scale_async<1>(scale1, scale_smem1);
            if constexpr (!PACKED) {
                load_mxnv_scale_async<1>(scale2, scale_smem2);
                load_mxnv_scale_async<1>(scale3, scale_smem3);
            }
            detail::tcgen05::commit<1>(scale_copy_done);
        }
        if (issue_warp) {
            wait(scale_copy_done, scale_phase);
            tensor_after_thread_sync();
        }
        scale_phase ^= 1;
        scale_copies += count;
        __syncthreads();
    };

    auto issue_and_store = [&]<int A_SFID, int B_SFID>(
        typename G::payload_tile &a_smem,
        typename G::payload_tile &b_smem,
        scale_t &a_scale,
        scale_t &b_scale,
        int output_index
    ) {
        st_descriptor<typename G::payload_tile, transpose::N> a_desc(a_smem);
        st_descriptor<typename G::payload_tile, transpose::N> b_desc(b_smem);
        constexpr uint32_t idesc =
            mxfp4_sfid_independent_descriptor<A_SFID, B_SFID>();
        if (issue_warp) {
            tensor_before_thread_sync();
        }
        if (issue_lane) {
            detail::tcgen05::st_st<fp4e2m1_2, fp8e8m0, 0, 1, 32>(
                output.addr,
                a_desc.chunk_descriptor(0),
                b_desc.chunk_descriptor(0),
                a_scale.addr,
                b_scale.addr,
                idesc);
            detail::tcgen05::commit<1>(mma_done);
        }
        if (issue_warp) {
            wait(mma_done, command_phase);
        }
        command_phase ^= 1;
        __syncthreads();
        tensor_after_thread_sync();
        const int row = warpgroup::laneid();
        float *dst = g.o + output_index * 128 * 128 + row * 128;
        #pragma unroll
        for (int col = 0; col < 128; col += 16) {
            float2 values[8];
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(values[0].x), "=f"(values[0].y),
                  "=f"(values[1].x), "=f"(values[1].y),
                  "=f"(values[2].x), "=f"(values[2].y),
                  "=f"(values[3].x), "=f"(values[3].y),
                  "=f"(values[4].x), "=f"(values[4].y),
                  "=f"(values[5].x), "=f"(values[5].y),
                  "=f"(values[6].x), "=f"(values[6].y),
                  "=f"(values[7].x), "=f"(values[7].y)
                : "r"(output.addr + ((local_warp * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                reinterpret_cast<float2 *>(dst + col)[pair] = values[pair];
            }
        }
        __syncthreads();
    };

    for (int epoch = 0; epoch < g.epochs; ++epoch) {
        if constexpr (PACKED) {
            const int companion_epoch = epoch == 0 ? 0 : epoch - 1;
            if (local_warp == 0) {
                mxfp4_fine_sfid_write_scale_tile(
                    scale_smem0, g.q_scale, g.p_scale,
                    epoch, companion_epoch);
            } else if (local_warp == 1) {
                mxfp4_fine_sfid_write_scale_tile(
                    scale_smem1, g.k_scale, g.v_scale,
                    epoch, companion_epoch);
            }
            copy_scales(2);
            issue_and_store.template operator()<0, 0>(
                q_smem, k_smem, scale0, scale1, epoch * 2);
            ++qk_commands;

            if (local_warp == 0) {
                mxfp4_fine_sfid_write_scale_tile(
                    scale_smem0, g.q_scale, g.p_scale, epoch, epoch);
            } else if (local_warp == 1) {
                mxfp4_fine_sfid_write_scale_tile(
                    scale_smem1, g.k_scale, g.v_scale, epoch, epoch);
            }
            copy_scales(2);
            issue_and_store.template operator()<2, 2>(
                p_smem, v_smem, scale0, scale1, epoch * 2 + 1);
            ++pv_commands;
        } else {
            if (local_warp == 0) {
                mxfp4_fine_sfid_write_scale_tile(
                    scale_smem0, g.q_scale, g.q_scale, epoch, epoch);
            } else if (local_warp == 1) {
                mxfp4_fine_sfid_write_scale_tile(
                    scale_smem1, g.k_scale, g.k_scale, epoch, epoch);
            } else if (local_warp == 2) {
                mxfp4_fine_sfid_write_scale_tile(
                    scale_smem2, g.p_scale, g.p_scale, epoch, epoch);
            } else {
                mxfp4_fine_sfid_write_scale_tile(
                    scale_smem3, g.v_scale, g.v_scale, epoch, epoch);
            }
            copy_scales(4);
            issue_and_store.template operator()<0, 0>(
                q_smem, k_smem, scale0, scale1, epoch * 2);
            ++qk_commands;
            issue_and_store.template operator()<0, 0>(
                p_smem, v_smem, scale2, scale3, epoch * 2 + 1);
            ++pv_commands;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        g.trace[0] = PACKED ? 1 : 0;
        g.trace[1] = g.epochs;
        g.trace[2] = scale_copies;
        g.trace[3] = qk_commands;
        g.trace[4] = pv_commands;
        g.trace[5] = scale0.addr;
        g.trace[6] = scale1.addr;
        g.trace[7] = PACKED ? 160 : 192;
    }
}

template <bool PACKED>
static void launch_mxfp4_fine_sfid_probe(
    const globals_mxfp4_fine_sfid_probe &g
) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_fine_sfid_probe<PACKED>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_mxfp4_fine_sfid_probe<PACKED>, g));
}

void dispatch_mxfp4_fine_sfid_probe(
    at::Tensor q_fp4,
    at::Tensor k_fp4,
    at::Tensor p_fp4,
    at::Tensor v_fp4,
    at::Tensor q_scale,
    at::Tensor k_scale,
    at::Tensor p_scale,
    at::Tensor v_scale,
    at::Tensor o,
    at::Tensor trace,
    bool packed
) {
    for (const at::Tensor *tensor : {
             &q_fp4, &k_fp4, &p_fp4, &v_fp4,
             &q_scale, &k_scale, &p_scale, &v_scale, &o, &trace}) {
        TORCH_CHECK(tensor->device().is_cuda(), "probe inputs must be CUDA tensors");
        TORCH_CHECK(tensor->is_contiguous(), "probe inputs must be contiguous");
    }
    for (const at::Tensor *tensor : {&q_fp4, &k_fp4, &p_fp4, &v_fp4}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Float4_e2m1fn_x2,
                    "payloads must be float4_e2m1fn_x2");
        TORCH_CHECK(tensor->dim() == 2 && tensor->size(0) == 128
                        && tensor->size(1) == 32,
                    "payloads must have shape (128,32 packed K64 bytes)");
    }
    const int epochs = q_scale.size(0);
    for (const at::Tensor *tensor : {&q_scale, &k_scale, &p_scale, &v_scale}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Int,
                    "scale words must be int32");
        TORCH_CHECK(tensor->dim() == 3 && tensor->size(0) == epochs
                        && tensor->size(1) == 32 && tensor->size(2) == 4,
                    "scale words must have shape (epochs,32,4)");
    }
    TORCH_CHECK(epochs >= 1 && epochs <= 16,
                "fine SFID probe supports 1..16 reuse epochs");
    TORCH_CHECK(o.dtype() == at::ScalarType::Float && o.dim() == 4
                    && o.size(0) == epochs && o.size(1) == 2
                    && o.size(2) == 128 && o.size(3) == 128,
                "o must be float32 (epochs,2,128,128)");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long && trace.numel() >= 32,
                "trace must contain at least 32 int64 values");

    globals_mxfp4_fine_sfid_probe g{
        reinterpret_cast<const uint8_t *>(q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(k_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v_fp4.data_ptr()),
        q_scale.data_ptr<int32_t>(),
        k_scale.data_ptr<int32_t>(),
        p_scale.data_ptr<int32_t>(),
        v_scale.data_ptr<int32_t>(),
        o.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        epochs,
    };
    if (packed) {
        launch_mxfp4_fine_sfid_probe<true>(g);
    } else {
        launch_mxfp4_fine_sfid_probe<false>(g);
    }
}

// Production-faithful scale-occupancy probe. QK uses the deployed Dqk192
// contract: three K64 commands, each with a complete E4M3 scale-fragment
// reload. PV uses the deployed K128 contract: two K64 commands selecting
// SFID 0 and 2 from complete E8M0 scale fragments.
struct config_mxfp4_production_sfid_probe {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dqk = 192;
    static constexpr int Dvo = 128;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int QK_SCALE_CHUNKS = 3;
};

struct globals_mxfp4_production_sfid_probe {
    using q_tile = st_fp4e2m1_2<128, 96>;
    using k_tile = st_fp4e2m1_2<128, 96>;
    using p_tile = st_fp4e2m1_2<128, 64>;
    using v_tile = st_fp4e2m1_2<128, 64>;
    using qk_scale_tile = st_fp8e4m3<32, 16, false>;
    using pv_scale_tile = st_fp8e8m0<32, 16, false>;

    const uint8_t *q_fp4;
    const uint8_t *k_fp4;
    const uint8_t *p_fp4;
    const uint8_t *v_fp4;
    const int32_t *q_scale;
    const int32_t *k_scale;
    const int32_t *p_scale;
    const int32_t *v_scale;
    float *o;
    int64_t *trace;
    int epochs;

    __host__ __inline__ dim3 grid() const { return dim3(1); }
    __host__ __inline__ dim3 block() const { return dim3(128); }
    __host__ __inline__ int dynamic_shared_memory() const { return 64 * 1024; }
};

template <typename Tile>
__device__ __forceinline__ void load_mxfp4_production_sfid_payload(
    Tile &tile,
    const uint8_t *source
) {
    static_assert(Tile::rows == 128 && Tile::cols % 16 == 0,
                  "production SFID payload must be M128 and vector-loadable");
    constexpr int BYTES_PER_STORE = 16;
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * STORES_PER_ROW;
         linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            source + row * Tile::cols + col);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(Tile::idx(base, {row, col})),
            "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename ScaleTile>
__device__ __forceinline__ void mxfp4_production_sfid_write_scale_tile(
    ScaleTile &tile,
    const int32_t *low_source,
    int low_chunks,
    int low_chunk,
    const int32_t *high_source,
    int high_chunks,
    int high_chunk,
    int epoch
) {
    const int lane = warp::laneid();
    const int low_base = ((epoch * low_chunks + low_chunk) * 32 + lane) * 4;
    const int high_base = ((epoch * high_chunks + high_chunk) * 32 + lane) * 4;
    uint32_t words[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        const uint32_t low = static_cast<uint32_t>(low_source[low_base + word]);
        const uint32_t high = static_cast<uint32_t>(high_source[high_base + word]);
        words[word] = (low & 0xffffu) | (high & 0xffff0000u);
    }
    fp4pv_store_mxfp4_scale_word4_swizzled(
        tile, lane, words[0], words[1], words[2], words[3]);
}

// MODE 0: four separate fragments.
// MODE 1: Q/K low bytes plus P/V high bytes, no temporal overwrite.
// MODE 2: P/V low bytes plus Q/K high bytes, no temporal overwrite.
// MODE 3: two fragments, Q/K first and P/V only after QK completion.
template <int MODE>
__launch_bounds__(128, 1)
__global__ void kernel_mxfp4_production_sfid_probe(
    const __grid_constant__ globals_mxfp4_production_sfid_probe g
) {
    using C = config_mxfp4_production_sfid_probe;
    using G = globals_mxfp4_production_sfid_probe;
    using output_t = tt<float, 128, 128>;
    using qk_scale_t = full_tt_fp8e4m3<16>;
    using pv_scale_t = full_tt_fp8e8m0<16>;
    static_assert(MODE >= 0 && MODE <= 3, "invalid production SFID probe mode");

    tensor_allocator<1, 1> tm_alloc{};
    output_t qk_output = tm_alloc.template allocate<output_t>(0);
    output_t pv_lower_output = tm_alloc.template allocate<output_t>(128);
    output_t pv_upper_output = tm_alloc.template allocate<output_t>(256);
    qk_scale_t q_scale_tm = tm_alloc.template allocate<qk_scale_t>(384);
    qk_scale_t k_scale_tm = tm_alloc.template allocate<qk_scale_t>(400);
    pv_scale_t p_scale_alias = tm_alloc.template allocate<pv_scale_t>(384);
    pv_scale_t v_scale_alias = tm_alloc.template allocate<pv_scale_t>(400);
    pv_scale_t p_scale_separate = tm_alloc.template allocate<pv_scale_t>(416);
    pv_scale_t v_scale_separate = tm_alloc.template allocate<pv_scale_t>(432);
    static_assert(384 + 4 * 16 <= 512,
                  "production SFID probe map must fit one TMEM allocation");

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::q_tile &q_smem = allocator.allocate<typename G::q_tile>();
    typename G::k_tile &k_smem = allocator.allocate<typename G::k_tile>();
    typename G::p_tile &p_smem = allocator.allocate<typename G::p_tile>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::qk_scale_tile &a_scale_smem =
        allocator.allocate<typename G::qk_scale_tile>();
    typename G::qk_scale_tile &b_scale_smem =
        allocator.allocate<typename G::qk_scale_tile>();
    typename G::pv_scale_tile &a_pv_scale_smem =
        *reinterpret_cast<typename G::pv_scale_tile *>(&a_scale_smem);
    typename G::pv_scale_tile &b_pv_scale_smem =
        *reinterpret_cast<typename G::pv_scale_tile *>(&b_scale_smem);

    __shared__ semaphore scale_copy_done;
    __shared__ semaphore command_done;
    if (threadIdx.x == 0) {
        init_semaphore(scale_copy_done, 0, 1);
        init_semaphore(command_done, 0, 1);
    }
    load_mxfp4_production_sfid_payload(q_smem, g.q_fp4);
    load_mxfp4_production_sfid_payload(k_smem, g.k_fp4);
    load_mxfp4_production_sfid_payload(p_smem, g.p_fp4);
    load_mxfp4_production_sfid_payload(v_smem, g.v_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();

    const int local_warp = warpgroup::warpid();
    const bool issue_warp = local_warp == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    int scale_phase = 0;
    int command_phase = 0;
    int qk_commands = 0;
    int pv_commands = 0;
    int scale_copies = 0;

    auto publish_scale_smem = [&]() {
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();
    };

    auto copy_scale_pair = [&](auto &a_tm, auto &b_tm,
                               const auto &a_smem, const auto &b_smem) {
        if (issue_lane) {
            load_mxnv_scale_async<1>(a_tm, a_smem);
            load_mxnv_scale_async<1>(b_tm, b_smem);
            detail::tcgen05::commit<1>(scale_copy_done);
        }
        if (issue_warp) {
            wait(scale_copy_done, scale_phase);
            tensor_after_thread_sync();
        }
        scale_phase ^= 1;
        scale_copies += 2;
        __syncthreads();
    };

    auto wait_command = [&]() {
        if (issue_warp) {
            wait(command_done, command_phase);
            tensor_after_thread_sync();
        }
        command_phase ^= 1;
        __syncthreads();
    };

    st_descriptor<typename G::q_tile, transpose::N> q_desc(q_smem);
    st_descriptor<typename G::k_tile, transpose::N> k_desc(k_smem);
    constexpr uint32_t qk_idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4e2m1_2, fp8e4m3, 128, 128, false, 0>();

    auto store_output = [&](output_t &output, int output_index) {
        const int row = warpgroup::laneid();
        float *dst = g.o + output_index * 128 * 128 + row * 128;
        #pragma unroll
        for (int col = 0; col < 128; col += 16) {
            float2 values[8];
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(values[0].x), "=f"(values[0].y),
                  "=f"(values[1].x), "=f"(values[1].y),
                  "=f"(values[2].x), "=f"(values[2].y),
                  "=f"(values[3].x), "=f"(values[3].y),
                  "=f"(values[4].x), "=f"(values[4].y),
                  "=f"(values[5].x), "=f"(values[5].y),
                  "=f"(values[6].x), "=f"(values[6].y),
                  "=f"(values[7].x), "=f"(values[7].y)
                : "r"(output.addr + ((local_warp * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                reinterpret_cast<float2 *>(dst + col)[pair] = values[pair];
            }
        }
        __syncthreads();
    };

    for (int epoch = 0; epoch < g.epochs; ++epoch) {
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            if (local_warp == 0) {
                if constexpr (MODE == 1) {
                    mxfp4_production_sfid_write_scale_tile(
                        a_scale_smem, g.q_scale, 3, chunk,
                        g.p_scale, 1, 0, epoch);
                } else if constexpr (MODE == 2) {
                    mxfp4_production_sfid_write_scale_tile(
                        a_scale_smem, g.p_scale, 1, 0,
                        g.q_scale, 3, chunk, epoch);
                } else {
                    mxfp4_production_sfid_write_scale_tile(
                        a_scale_smem, g.q_scale, 3, chunk,
                        g.q_scale, 3, chunk, epoch);
                }
            } else if (local_warp == 1) {
                if constexpr (MODE == 1) {
                    mxfp4_production_sfid_write_scale_tile(
                        b_scale_smem, g.k_scale, 3, chunk,
                        g.v_scale, 1, 0, epoch);
                } else if constexpr (MODE == 2) {
                    mxfp4_production_sfid_write_scale_tile(
                        b_scale_smem, g.v_scale, 1, 0,
                        g.k_scale, 3, chunk, epoch);
                } else {
                    mxfp4_production_sfid_write_scale_tile(
                        b_scale_smem, g.k_scale, 3, chunk,
                        g.k_scale, 3, chunk, epoch);
                }
            }
            publish_scale_smem();
            copy_scale_pair(
                q_scale_tm, k_scale_tm, a_scale_smem, b_scale_smem);
            if (issue_warp) {
                tensor_before_thread_sync();
            }
            if (issue_lane) {
                if (chunk == 0) {
                    detail::tcgen05::st_st<fp4e2m1_2, fp8e4m3, 0, 1, 16>(
                        qk_output.addr,
                        q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk),
                        q_scale_tm.addr,
                        k_scale_tm.addr,
                        qk_idesc);
                } else {
                    detail::tcgen05::st_st<fp4e2m1_2, fp8e4m3, 1, 1, 16>(
                        qk_output.addr,
                        q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk),
                        q_scale_tm.addr,
                        k_scale_tm.addr,
                        qk_idesc);
                }
                ++qk_commands;
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(command_done);
        }
        wait_command();
        store_output(qk_output, epoch * 3);

        if constexpr (MODE == 0 || MODE == 3) {
            if (local_warp == 0) {
                mxfp4_production_sfid_write_scale_tile(
                    a_scale_smem, g.p_scale, 1, 0,
                    g.p_scale, 1, 0, epoch);
            } else if (local_warp == 1) {
                mxfp4_production_sfid_write_scale_tile(
                    b_scale_smem, g.v_scale, 1, 0,
                    g.v_scale, 1, 0, epoch);
            }
            publish_scale_smem();
            if constexpr (MODE == 0) {
                copy_scale_pair(
                    p_scale_separate, v_scale_separate,
                    a_pv_scale_smem, b_pv_scale_smem);
            } else {
                copy_scale_pair(
                    p_scale_alias, v_scale_alias,
                    a_pv_scale_smem, b_pv_scale_smem);
            }
        }
        auto issue_pv_halves = [&](pv_scale_t &p_scale_tm,
                                   pv_scale_t &v_scale_tm) {
            if (issue_warp) {
                tensor_before_thread_sync();
            }
            if (issue_lane) {
                fp4pv_mma_p_stage_ABt_k64_chunk_view<0, 0, 0, true, C>(
                    pv_lower_output, p_smem, v_smem,
                    p_scale_tm, v_scale_tm, command_done);
                ++pv_commands;
            }
            wait_command();
            store_output(pv_lower_output, epoch * 3 + 1);

            if (issue_warp) {
                tensor_before_thread_sync();
            }
            if (issue_lane) {
                fp4pv_mma_p_stage_ABt_k64_chunk_view<1, 0, 2, true, C>(
                    pv_upper_output, p_smem, v_smem,
                    p_scale_tm, v_scale_tm, command_done);
                ++pv_commands;
            }
            wait_command();
            store_output(pv_upper_output, epoch * 3 + 2);
        };
        if constexpr (MODE == 0) {
            issue_pv_halves(p_scale_separate, v_scale_separate);
        } else {
            issue_pv_halves(p_scale_alias, v_scale_alias);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        g.trace[0] = MODE;
        g.trace[1] = g.epochs;
        g.trace[2] = qk_commands;
        g.trace[3] = pv_commands;
        g.trace[4] = scale_copies;
        g.trace[5] = 384;
        g.trace[6] = 400;
        g.trace[7] = MODE == 0 ? 448 : 416;
    }
}

template <int MODE>
static void launch_mxfp4_production_sfid_probe(
    const globals_mxfp4_production_sfid_probe &g
) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_production_sfid_probe<MODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_mxfp4_production_sfid_probe<MODE>, g));
}

void dispatch_mxfp4_production_sfid_probe(
    at::Tensor q_fp4,
    at::Tensor k_fp4,
    at::Tensor p_fp4,
    at::Tensor v_fp4,
    at::Tensor q_scale,
    at::Tensor k_scale,
    at::Tensor p_scale,
    at::Tensor v_scale,
    at::Tensor o,
    at::Tensor trace,
    int mode
) {
    for (const at::Tensor *tensor : {
             &q_fp4, &k_fp4, &p_fp4, &v_fp4,
             &q_scale, &k_scale, &p_scale, &v_scale, &o, &trace}) {
        TORCH_CHECK(tensor->device().is_cuda(), "probe inputs must be CUDA tensors");
        TORCH_CHECK(tensor->is_contiguous(), "probe inputs must be contiguous");
    }
    for (const at::Tensor *tensor : {&q_fp4, &k_fp4, &p_fp4, &v_fp4}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Float4_e2m1fn_x2,
                    "payloads must be float4_e2m1fn_x2");
    }
    TORCH_CHECK(q_fp4.sizes() == at::IntArrayRef({128, 96})
                    && k_fp4.sizes() == at::IntArrayRef({128, 96}),
                "Q/K payloads must be (128,96 packed D192 bytes)");
    TORCH_CHECK(p_fp4.sizes() == at::IntArrayRef({128, 64})
                    && v_fp4.sizes() == at::IntArrayRef({128, 64}),
                "P/V payloads must be (128,64 packed K128 bytes)");
    const int epochs = q_scale.size(0);
    for (const at::Tensor *tensor : {&q_scale, &k_scale}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Int
                        && tensor->dim() == 4
                        && tensor->size(0) == epochs
                        && tensor->size(1) == 3
                        && tensor->size(2) == 32
                        && tensor->size(3) == 4,
                    "Q/K scale words must be int32 (epochs,3,32,4)");
    }
    for (const at::Tensor *tensor : {&p_scale, &v_scale}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Int
                        && tensor->dim() == 3
                        && tensor->size(0) == epochs
                        && tensor->size(1) == 32
                        && tensor->size(2) == 4,
                    "P/V scale words must be int32 (epochs,32,4)");
    }
    TORCH_CHECK(epochs >= 1 && epochs <= 8,
                "production SFID probe supports 1..8 epochs");
    TORCH_CHECK(o.dtype() == at::ScalarType::Float
                    && o.dim() == 4 && o.size(0) == epochs
                    && o.size(1) == 3 && o.size(2) == 128 && o.size(3) == 128,
                "o must be float32 (epochs,3,128,128)");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long && trace.numel() >= 32,
                "trace must contain at least 32 int64 values");
    TORCH_CHECK(mode >= 0 && mode <= 3, "mode must be in [0,3]");

    globals_mxfp4_production_sfid_probe g{
        reinterpret_cast<const uint8_t *>(q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(k_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v_fp4.data_ptr()),
        q_scale.data_ptr<int32_t>(),
        k_scale.data_ptr<int32_t>(),
        p_scale.data_ptr<int32_t>(),
        v_scale.data_ptr<int32_t>(),
        o.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        epochs,
    };
    if (mode == 0) {
        launch_mxfp4_production_sfid_probe<0>(g);
    } else if (mode == 1) {
        launch_mxfp4_production_sfid_probe<1>(g);
    } else if (mode == 2) {
        launch_mxfp4_production_sfid_probe<2>(g);
    } else {
        launch_mxfp4_production_sfid_probe<3>(g);
    }
}

// Alternating exact-scale page lifetime probe. Each 32-column page is either
// Q16+K16 for one Dqk192 QK command group or P16+V16 for one K128 PV command
// group. MODE 0 drains PV before issuing the next QK. MODE 1 issues the next
// QK behind the current PV without a completion wait, using the other page.
template <bool CONCURRENT>
__launch_bounds__(128, 1)
__global__ void kernel_mxfp4_scale_page_swap_probe(
    const __grid_constant__ globals_mxfp4_production_sfid_probe g
) {
    using C = config_mxfp4_production_sfid_probe;
    using G = globals_mxfp4_production_sfid_probe;
    using output_t = tt<float, 128, 128>;
    using qk_scale_t = full_tt_fp8e4m3<16>;
    using pv_scale_t = full_tt_fp8e8m0<16>;

    tensor_allocator<1, 1> tm_alloc{};
    output_t qk_output = tm_alloc.template allocate<output_t>(0);
    output_t pv_output = tm_alloc.template allocate<output_t>(128);
    qk_scale_t q_scale_page0 =
        tm_alloc.template allocate<qk_scale_t>(384);
    qk_scale_t k_scale_page0 =
        tm_alloc.template allocate<qk_scale_t>(400);
    qk_scale_t q_scale_page1 =
        tm_alloc.template allocate<qk_scale_t>(416);
    qk_scale_t k_scale_page1 =
        tm_alloc.template allocate<qk_scale_t>(432);
    pv_scale_t p_scale_page0 =
        tm_alloc.template allocate<pv_scale_t>(384);
    pv_scale_t v_scale_page0 =
        tm_alloc.template allocate<pv_scale_t>(400);
    pv_scale_t p_scale_page1 =
        tm_alloc.template allocate<pv_scale_t>(416);
    pv_scale_t v_scale_page1 =
        tm_alloc.template allocate<pv_scale_t>(432);
    static_assert(384 + 2 * 32 <= 512,
                  "two exact scale pages must fit one TMEM allocation");

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::q_tile &q_smem = allocator.allocate<typename G::q_tile>();
    typename G::k_tile &k_smem = allocator.allocate<typename G::k_tile>();
    typename G::p_tile &p_smem = allocator.allocate<typename G::p_tile>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::qk_scale_tile &q_scale_smem =
        allocator.allocate<typename G::qk_scale_tile>();
    typename G::qk_scale_tile &k_scale_smem =
        allocator.allocate<typename G::qk_scale_tile>();
    typename G::pv_scale_tile &p_scale_smem =
        allocator.allocate<typename G::pv_scale_tile>();
    typename G::pv_scale_tile &v_scale_smem =
        allocator.allocate<typename G::pv_scale_tile>();

    __shared__ semaphore pv_scale_copy_done[2];
    __shared__ semaphore qk_done[2];
    __shared__ semaphore pv_done[2];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int page = 0; page < 2; ++page) {
            init_semaphore(pv_scale_copy_done[page], 0, 1);
            init_semaphore(qk_done[page], 0, 1);
            init_semaphore(pv_done[page], 0, 1);
        }
    }

    load_mxfp4_production_sfid_payload(q_smem, g.q_fp4);
    load_mxfp4_production_sfid_payload(k_smem, g.k_fp4);
    load_mxfp4_production_sfid_payload(p_smem, g.p_fp4);
    load_mxfp4_production_sfid_payload(v_smem, g.v_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();

    const int local_warp = warpgroup::warpid();
    const bool issue_warp = local_warp == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    st_descriptor<typename G::q_tile, transpose::N> q_desc(q_smem);
    st_descriptor<typename G::k_tile, transpose::N> k_desc(k_smem);
    constexpr uint32_t qk_idesc =
        detail::tcgen05::instruction_descriptor<
            float, fp4e2m1_2, fp8e4m3, 128, 128, false, 0>();

    auto publish_scale_source = [&]() {
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();
    };
    auto q_page = [&](int page) -> qk_scale_t {
        return page == 0 ? q_scale_page0 : q_scale_page1;
    };
    auto k_page = [&](int page) -> qk_scale_t {
        return page == 0 ? k_scale_page0 : k_scale_page1;
    };
    auto p_page = [&](int page) -> pv_scale_t {
        return page == 0 ? p_scale_page0 : p_scale_page1;
    };
    auto v_page = [&](int page) -> pv_scale_t {
        return page == 0 ? v_scale_page0 : v_scale_page1;
    };

    auto issue_qk = [&](int epoch) {
        const int page = epoch & 1;
        auto q_scale_tm = q_page(page);
        auto k_scale_tm = k_page(page);
        if (issue_warp) {
            tensor_before_thread_sync();
        }
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            if (local_warp == 0) {
                mxfp4_production_sfid_write_scale_tile(
                    q_scale_smem, g.q_scale, 3, chunk,
                    g.q_scale, 3, chunk, epoch);
            } else if (local_warp == 1) {
                mxfp4_production_sfid_write_scale_tile(
                    k_scale_smem, g.k_scale, 3, chunk,
                    g.k_scale, 3, chunk, epoch);
            }
            publish_scale_source();
            if (issue_lane) {
                load_mxnv_scale_async<1>(q_scale_tm, q_scale_smem);
                load_mxnv_scale_async<1>(k_scale_tm, k_scale_smem);
                if (chunk == 0) {
                    detail::tcgen05::st_st<
                        fp4e2m1_2, fp8e4m3, 0, 1, 16>(
                        qk_output.addr,
                        q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk),
                        q_scale_tm.addr,
                        k_scale_tm.addr,
                        qk_idesc);
                } else {
                    detail::tcgen05::st_st<
                        fp4e2m1_2, fp8e4m3, 1, 1, 16>(
                        qk_output.addr,
                        q_desc.chunk_descriptor(chunk),
                        k_desc.chunk_descriptor(chunk),
                        q_scale_tm.addr,
                        k_scale_tm.addr,
                        qk_idesc);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done[page]);
        }
    };

    auto issue_pv = [&](int epoch) {
        const int page = epoch & 1;
        auto p_scale_tm = p_page(page);
        auto v_scale_tm = v_page(page);
        if (local_warp == 0) {
            mxfp4_production_sfid_write_scale_tile(
                p_scale_smem, g.p_scale, 1, 0,
                g.p_scale, 1, 0, epoch);
        } else if (local_warp == 1) {
            mxfp4_production_sfid_write_scale_tile(
                v_scale_smem, g.v_scale, 1, 0,
                g.v_scale, 1, 0, epoch);
        }
        publish_scale_source();
        if (issue_lane) {
            load_mxnv_scale_async<1>(p_scale_tm, p_scale_smem);
            load_mxnv_scale_async<1>(v_scale_tm, v_scale_smem);
            detail::tcgen05::commit<1>(pv_scale_copy_done[page]);
        }
        if (issue_warp) {
            wait(pv_scale_copy_done[page], (epoch >> 1) & 1);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
        }
        if (issue_lane) {
            fp4pv_mma_p_stage_ABt_k64_chunk_view<
                0, 0, 0, false, C>(
                pv_output, p_smem, v_smem,
                p_scale_tm, v_scale_tm, pv_done[page]);
            fp4pv_mma_p_stage_ABt_k64_chunk_view<
                1, 1, 2, true, C>(
                pv_output, p_smem, v_smem,
                p_scale_tm, v_scale_tm, pv_done[page]);
        }
    };

    auto wait_command = [&](semaphore &done, int phase) {
        if (issue_warp) {
            wait(done, phase);
            tensor_after_thread_sync();
        }
        __syncthreads();
    };

    auto store_output = [&](output_t &output, int output_index) {
        tensor_after_thread_sync();
        const int row = warpgroup::laneid();
        float *dst = g.o + output_index * 128 * 128 + row * 128;
        #pragma unroll
        for (int col = 0; col < 128; col += 16) {
            float2 values[8];
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(values[0].x), "=f"(values[0].y),
                  "=f"(values[1].x), "=f"(values[1].y),
                  "=f"(values[2].x), "=f"(values[2].y),
                  "=f"(values[3].x), "=f"(values[3].y),
                  "=f"(values[4].x), "=f"(values[4].y),
                  "=f"(values[5].x), "=f"(values[5].y),
                  "=f"(values[6].x), "=f"(values[6].y),
                  "=f"(values[7].x), "=f"(values[7].y)
                : "r"(output.addr + ((local_warp * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                reinterpret_cast<float2 *>(dst + col)[pair] = values[pair];
            }
        }
        __syncthreads();
    };

    auto stamp = [&](int epoch, int field) {
        if (issue_lane) {
            g.trace[8 + epoch * 8 + field] =
                static_cast<int64_t>(clock64());
        }
    };

    stamp(0, 0);
    issue_qk(0);
    wait_command(qk_done[0], 0);
    stamp(0, 1);
    store_output(qk_output, 0);

    for (int epoch = 0; epoch < g.epochs; ++epoch) {
        const int page = epoch & 1;
        const int page_phase = (epoch >> 1) & 1;
        stamp(epoch, 2);
        issue_pv(epoch);
        if constexpr (CONCURRENT) {
            if (epoch + 1 < g.epochs) {
                stamp(epoch, 3);
                stamp(epoch + 1, 0);
                issue_qk(epoch + 1);
            }
            wait_command(pv_done[page], page_phase);
            stamp(epoch, 4);
            store_output(pv_output, epoch * 2 + 1);
            if (epoch + 1 < g.epochs) {
                const int next_page = (epoch + 1) & 1;
                const int next_phase = ((epoch + 1) >> 1) & 1;
                wait_command(qk_done[next_page], next_phase);
                stamp(epoch, 5);
                stamp(epoch + 1, 1);
                store_output(qk_output, (epoch + 1) * 2);
            }
        } else {
            wait_command(pv_done[page], page_phase);
            stamp(epoch, 4);
            store_output(pv_output, epoch * 2 + 1);
            if (epoch + 1 < g.epochs) {
                stamp(epoch, 3);
                stamp(epoch + 1, 0);
                issue_qk(epoch + 1);
                const int next_page = (epoch + 1) & 1;
                const int next_phase = ((epoch + 1) >> 1) & 1;
                wait_command(qk_done[next_page], next_phase);
                stamp(epoch, 5);
                stamp(epoch + 1, 1);
                store_output(qk_output, (epoch + 1) * 2);
            }
        }
        if (issue_lane) {
            g.trace[8 + epoch * 8 + 6] = page;
            g.trace[8 + epoch * 8 + 7] =
                (epoch + 1 < g.epochs &&
                 g.trace[8 + epoch * 8 + 3] <
                     g.trace[8 + epoch * 8 + 4]) ? 1 : 0;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        g.trace[0] = CONCURRENT ? 1 : 0;
        g.trace[1] = g.epochs;
        g.trace[2] = 3 * g.epochs;
        g.trace[3] = 2 * g.epochs;
        g.trace[4] = 384;
        g.trace[5] = 416;
        g.trace[6] = 32;
        g.trace[7] = CONCURRENT ? g.epochs - 1 : 0;
    }
}

template <bool CONCURRENT>
static void launch_mxfp4_scale_page_swap_probe(
    const globals_mxfp4_production_sfid_probe &g
) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_scale_page_swap_probe<CONCURRENT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_mxfp4_scale_page_swap_probe<CONCURRENT>, g));
}

void dispatch_mxfp4_scale_page_swap_probe(
    at::Tensor q_fp4,
    at::Tensor k_fp4,
    at::Tensor p_fp4,
    at::Tensor v_fp4,
    at::Tensor q_scale,
    at::Tensor k_scale,
    at::Tensor p_scale,
    at::Tensor v_scale,
    at::Tensor o,
    at::Tensor trace,
    bool concurrent
) {
    for (const at::Tensor *tensor : {
             &q_fp4, &k_fp4, &p_fp4, &v_fp4,
             &q_scale, &k_scale, &p_scale, &v_scale, &o, &trace}) {
        TORCH_CHECK(tensor->device().is_cuda(),
                    "scale-page probe inputs must be CUDA tensors");
        TORCH_CHECK(tensor->is_contiguous(),
                    "scale-page probe inputs must be contiguous");
    }
    for (const at::Tensor *tensor : {&q_fp4, &k_fp4, &p_fp4, &v_fp4}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Float4_e2m1fn_x2,
                    "scale-page payloads must be float4_e2m1fn_x2");
    }
    TORCH_CHECK(q_fp4.sizes() == at::IntArrayRef({128, 96})
                    && k_fp4.sizes() == at::IntArrayRef({128, 96}),
                "Q/K payloads must be (128,96 packed D192 bytes)");
    TORCH_CHECK(p_fp4.sizes() == at::IntArrayRef({128, 64})
                    && v_fp4.sizes() == at::IntArrayRef({128, 64}),
                "P/V payloads must be (128,64 packed K128 bytes)");
    const int epochs = q_scale.size(0);
    for (const at::Tensor *tensor : {&q_scale, &k_scale}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Int
                        && tensor->dim() == 4
                        && tensor->size(0) == epochs
                        && tensor->size(1) == 3
                        && tensor->size(2) == 32
                        && tensor->size(3) == 4,
                    "Q/K scales must be int32 (epochs,3,32,4)");
    }
    for (const at::Tensor *tensor : {&p_scale, &v_scale}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Int
                        && tensor->dim() == 3
                        && tensor->size(0) == epochs
                        && tensor->size(1) == 32
                        && tensor->size(2) == 4,
                    "P/V scales must be int32 (epochs,32,4)");
    }
    TORCH_CHECK(epochs >= 2 && epochs <= 8,
                "scale-page probe supports 2..8 epochs");
    TORCH_CHECK(o.dtype() == at::ScalarType::Float
                    && o.dim() == 4 && o.size(0) == epochs
                    && o.size(1) == 2 && o.size(2) == 128
                    && o.size(3) == 128,
                "o must be float32 (epochs,2,128,128)");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long
                    && trace.numel() >= 8 + epochs * 8,
                "trace must contain at least 8 + epochs*8 int64 values");

    globals_mxfp4_production_sfid_probe g{
        reinterpret_cast<const uint8_t *>(q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(k_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v_fp4.data_ptr()),
        q_scale.data_ptr<int32_t>(),
        k_scale.data_ptr<int32_t>(),
        p_scale.data_ptr<int32_t>(),
        v_scale.data_ptr<int32_t>(),
        o.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        epochs,
    };
    if (concurrent) {
        launch_mxfp4_scale_page_swap_probe<true>(g);
    } else {
        launch_mxfp4_scale_page_swap_probe<false>(g);
    }
}

#endif

#pragma once

#ifndef TK_FA4_TMEM_OUTPUT_CHECKPOINT_PROBE_BUILD
#define TK_FA4_TMEM_OUTPUT_CHECKPOINT_PROBE_BUILD 0
#endif

#if TK_FA4_TMEM_OUTPUT_CHECKPOINT_PROBE_BUILD

struct config_mxfp4_output_checkpoint_probe {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dvo = 128;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 128;
};

struct globals_mxfp4_output_checkpoint_probe {
    using payload_tile = st_fp4e2m1_2<128, 64>;
    using scale_tile = st_fp8e8m0<32, 16, false>;

    const uint8_t *p0_fp4;
    const uint8_t *v0_fp4;
    const uint8_t *p1_fp4;
    const uint8_t *v1_fp4;
    const int32_t *scale_words;
    float *restored_o;
    float *final_o;
    int64_t *trace;
    int roundtrips;

    __host__ __inline__ dim3 grid() const { return dim3(1); }
    __host__ __inline__ dim3 block() const {
        return dim3(config_mxfp4_output_checkpoint_probe::NUM_THREADS);
    }
    __host__ __inline__ int dynamic_shared_memory() const {
        return 48 * 1024;
    }
};

template <typename Tile>
__device__ __forceinline__ void mxfp4_checkpoint_load_payload(
    Tile &tile,
    const uint8_t *source
) {
    static_assert((Tile::rows == 64 || Tile::rows == 128) && Tile::cols == 64,
                  "checkpoint payload must be a dense M64/M128xK128 FP4 tile");
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
__device__ __forceinline__ void mxfp4_checkpoint_write_scale_tile(
    ScaleTile &tile,
    const int32_t *source
) {
    const int lane = warp::laneid();
    const int base = lane * 4;
    fp4pv_store_mxfp4_scale_word4_swizzled(
        tile, lane,
        static_cast<uint32_t>(source[base + 0]),
        static_cast<uint32_t>(source[base + 1]),
        static_cast<uint32_t>(source[base + 2]),
        static_cast<uint32_t>(source[base + 3]));
}

template <typename OutputTT>
__device__ __forceinline__ void mxfp4_checkpoint_store_output(
    const OutputTT &output,
    float *destination
) {
    const int row = warpgroup::laneid();
    const int local_warp = warpgroup::warpid();
    #pragma unroll
    for (int col = 0; col < OutputTT::cols; col += 16) {
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
            reinterpret_cast<float2 *>(destination + row * OutputTT::cols + col)[pair] =
                values[pair];
        }
    }
}

// The standard typed 16-bit TMEM loader uses tcgen05.ld.pack::16b, which is
// for an expanded one-16-bit-value-per-TMEM-cell source. The cold checkpoint
// stores BF16x2/FP16x2 words densely with the bare tcgen05.st form, so restore
// must use the matching bare load and leave numeric widening to warp::copy.
template <typename ColdRT, typename ColdChunk>
__device__ __forceinline__ void mxfp4_checkpoint_load_packed_cold(
    ColdRT &destination,
    const ColdChunk &source
) {
    static_assert(ColdRT::rows == 32 && ColdRT::cols == 32,
                  "packed cold load expects one warp's 32x32 register tile");
    static_assert(ColdRT::height == 2 && ColdRT::width == 2,
                  "packed cold load is specialized for 16x16 base tiles");
    #pragma unroll
    for (int tile_row = 0; tile_row < 2; ++tile_row) {
        asm volatile(
            "{tcgen05.ld.sync.aligned.16x128b.x4.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];}"
            : "=r"(*reinterpret_cast<uint32_t *>(
                      &destination.tiles[tile_row][0].data[0])),
              "=r"(*reinterpret_cast<uint32_t *>(
                      &destination.tiles[tile_row][0].data[1])),
              "=r"(*reinterpret_cast<uint32_t *>(
                      &destination.tiles[tile_row][0].data[2])),
              "=r"(*reinterpret_cast<uint32_t *>(
                      &destination.tiles[tile_row][0].data[3])),
              "=r"(*reinterpret_cast<uint32_t *>(
                      &destination.tiles[tile_row][1].data[0])),
              "=r"(*reinterpret_cast<uint32_t *>(
                      &destination.tiles[tile_row][1].data[1])),
              "=r"(*reinterpret_cast<uint32_t *>(
                      &destination.tiles[tile_row][1].data[2])),
              "=r"(*reinterpret_cast<uint32_t *>(
                      &destination.tiles[tile_row][1].data[3]))
            : "r"(source.addr
                    + (((warpgroup::warpid() * 32 + tile_row * 16)) << 16)));
    }
}

template <typename ColdTT, typename ColdChunk, typename ColdRT, typename OutputTT>
__device__ __forceinline__ void mxfp4_checkpoint_save(
    OutputTT &active,
    ColdTT &cold
) {
    using active_chunk_t = full_tt_fl<32>;
    using active_rt_t = rt_fl<32, 32>;

    static_assert(OutputTT::cols == ColdTT::cols && OutputTT::cols % 32 == 0,
                  "active and cold checkpoint widths must match in 32-column chunks");
    #pragma unroll
    for (int chunk = 0; chunk < OutputTT::cols / 32; ++chunk) {
        active_rt_t active_regs;
        ColdRT cold_regs;
        auto active_chunk =
            active.template subtile<active_chunk_t>(chunk * 32);
        auto cold_chunk = cold.template subtile<ColdChunk>(chunk * 32);
        warpgroup::load_async(active_regs, active_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        warp::copy(cold_regs, active_regs);
        warpgroup::store_async(cold_chunk, cold_regs);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
    warpgroup::sync(1);
}

template <typename OutputTT>
__device__ __forceinline__ void mxfp4_checkpoint_clear(
    OutputTT &active
) {
    using active_chunk_t = full_tt_fl<32>;
    using active_rt_t = rt_fl<32, 32>;

    static_assert(OutputTT::cols % 32 == 0,
                  "active checkpoint width must use 32-column chunks");
    #pragma unroll
    for (int chunk = 0; chunk < OutputTT::cols / 32; ++chunk) {
        active_rt_t zeros;
        warp::zero(zeros);
        auto active_chunk =
            active.template subtile<active_chunk_t>(chunk * 32);
        warpgroup::store_async(active_chunk, zeros);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
    warpgroup::sync(1);
}

template <typename ColdTT, typename ColdChunk, typename ColdRT, typename OutputTT>
__device__ __forceinline__ void mxfp4_checkpoint_restore(
    OutputTT &active,
    ColdTT &cold
) {
    using active_chunk_t = full_tt_fl<32>;
    using active_rt_t = rt_fl<32, 32>;

    static_assert(OutputTT::cols == ColdTT::cols && OutputTT::cols % 32 == 0,
                  "active and cold restore widths must match in 32-column chunks");
    #pragma unroll
    for (int chunk = 0; chunk < OutputTT::cols / 32; ++chunk) {
        ColdRT cold_regs;
        active_rt_t active_regs;
        auto cold_chunk = cold.template subtile<ColdChunk>(chunk * 32);
        auto active_chunk =
            active.template subtile<active_chunk_t>(chunk * 32);
        mxfp4_checkpoint_load_packed_cold(cold_regs, cold_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        warp::copy(active_regs, cold_regs);
        warpgroup::store_async(active_chunk, active_regs);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
    warpgroup::sync(1);
}

template <typename ColdTT, typename ColdChunk, typename ColdRT, typename OutputTT>
__device__ __forceinline__ void mxfp4_checkpoint_roundtrip(
    OutputTT &active,
    ColdTT &cold
) {
    // The clear is a proof device: it establishes that continuation cannot use
    // stale FP32 state. A live active/cold ownership swap does not require it.
    mxfp4_checkpoint_save<ColdTT, ColdChunk, ColdRT>(active, cold);
    mxfp4_checkpoint_clear(active);
    mxfp4_checkpoint_restore<ColdTT, ColdChunk, ColdRT>(active, cold);
}

template <typename ColdTT, typename ColdChunk, typename ColdRT, typename OutputTT>
__device__ __forceinline__ void mxfp4_checkpoint_swap(
    OutputTT &active,
    ColdTT &cold
) {
    using active_chunk_t = full_tt_fl<32>;
    using active_rt_t = rt_fl<32, 32>;
    static_assert(OutputTT::cols == ColdTT::cols && OutputTT::cols % 32 == 0,
                  "active/cold swap widths must match in 32-column chunks");
    #pragma unroll
    for (int chunk = 0; chunk < OutputTT::cols / 32; ++chunk) {
        active_rt_t outgoing_active;
        active_rt_t incoming_active;
        ColdRT outgoing_cold;
        ColdRT incoming_cold;
        auto active_chunk =
            active.template subtile<active_chunk_t>(chunk * 32);
        auto cold_chunk = cold.template subtile<ColdChunk>(chunk * 32);
        warpgroup::load_async(outgoing_active, active_chunk);
        mxfp4_checkpoint_load_packed_cold(incoming_cold, cold_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        warp::copy(outgoing_cold, outgoing_active);
        warp::copy(incoming_active, incoming_cold);
        warpgroup::store_async(cold_chunk, outgoing_cold);
        warpgroup::store_async(active_chunk, incoming_active);
        tensor_store_wait();
        tensor_after_thread_sync();
    }
    warpgroup::sync(1);
}

template <int FORMAT>
__launch_bounds__(config_mxfp4_output_checkpoint_probe::NUM_THREADS, 1)
__global__ void kernel_mxfp4_output_checkpoint_probe(
    const __grid_constant__ globals_mxfp4_output_checkpoint_probe g
) {
    using C = config_mxfp4_output_checkpoint_probe;
    using G = globals_mxfp4_output_checkpoint_probe;
    using output_t = full_tt_fl<128>;
    using cold_bf_t = full_tt_bf<128>;
    using cold_hf_t = full_tt_hf<128>;
    using scale_t = full_tt_fp8e8m0<16>;
    static_assert(FORMAT >= 0 && FORMAT <= 2, "invalid checkpoint format");
    static_assert(128 + 64 + 16 + 16 <= 512,
                  "checkpoint primitive TMEM map exceeds one allocation");

    tensor_allocator<1, 1> tm_alloc{};
    output_t active = tm_alloc.template allocate<output_t>(0);
    cold_bf_t cold_bf = tm_alloc.template allocate<cold_bf_t>(128);
    cold_hf_t cold_hf = tm_alloc.template allocate<cold_hf_t>(128);
    scale_t p_scale_tm = tm_alloc.template allocate<scale_t>(192);
    scale_t v_scale_tm = tm_alloc.template allocate<scale_t>(208);

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::payload_tile &p0_smem =
        allocator.allocate<typename G::payload_tile>();
    typename G::payload_tile &v0_smem =
        allocator.allocate<typename G::payload_tile>();
    typename G::payload_tile &p1_smem =
        allocator.allocate<typename G::payload_tile>();
    typename G::payload_tile &v1_smem =
        allocator.allocate<typename G::payload_tile>();
    typename G::scale_tile &p_scale_smem =
        allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &v_scale_smem =
        allocator.allocate<typename G::scale_tile>();

    __shared__ semaphore scale_copy_done;
    __shared__ semaphore pv_done;
    if (threadIdx.x == 0) {
        init_semaphore(scale_copy_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
    }
    mxfp4_checkpoint_load_payload(p0_smem, g.p0_fp4);
    mxfp4_checkpoint_load_payload(v0_smem, g.v0_fp4);
    mxfp4_checkpoint_load_payload(p1_smem, g.p1_fp4);
    mxfp4_checkpoint_load_payload(v1_smem, g.v1_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();

    const bool issue_warp = warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    int scale_phase = 0;
    int pv_phase = 0;

    auto issue_pv = [&](auto &p_smem, auto &v_smem, int pair, bool clear) {
        if (warpgroup::warpid() == 0) {
            mxfp4_checkpoint_write_scale_tile(
                p_scale_smem, g.scale_words + (pair * 2 + 0) * 32 * 4);
        } else if (warpgroup::warpid() == 1) {
            mxfp4_checkpoint_write_scale_tile(
                v_scale_smem, g.scale_words + (pair * 2 + 1) * 32 * 4);
        }
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();
        if (issue_lane) {
            load_mxnv_scale_async<1>(p_scale_tm, p_scale_smem);
            load_mxnv_scale_async<1>(v_scale_tm, v_scale_smem);
            detail::tcgen05::commit<1>(scale_copy_done);
        }
        if (issue_warp) {
            wait(scale_copy_done, scale_phase);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
        }
        scale_phase ^= 1;
        if (issue_lane) {
            if (clear) {
                fp4pv_p_stage_ABt_split_k64<C, true>(
                    active, p_smem, v_smem,
                    p_scale_tm, v_scale_tm, pv_done);
            } else {
                fp4pv_p_stage_ABt_split_k64<C, false>(
                    active, p_smem, v_smem,
                    p_scale_tm, v_scale_tm, pv_done);
            }
        }
        if (issue_warp) {
            wait(pv_done, pv_phase);
            tensor_after_thread_sync();
        }
        pv_phase ^= 1;
        __syncthreads();
    };

    const uint64_t first_start = threadIdx.x == 0 ? clock64() : 0;
    issue_pv(p0_smem, v0_smem, 0, true);
    const uint64_t first_end = threadIdx.x == 0 ? clock64() : 0;

    uint64_t checkpoint_start = 0;
    uint64_t checkpoint_end = 0;
    uint64_t save_cycles = 0;
    uint64_t clear_cycles = 0;
    uint64_t restore_cycles = 0;
    if constexpr (FORMAT != 0) {
        warpgroup::sync(1);
        if (threadIdx.x == 0) {
            checkpoint_start = clock64();
        }
        for (int roundtrip = 0; roundtrip < g.roundtrips; ++roundtrip) {
            uint64_t phase_start = threadIdx.x == 0 ? clock64() : 0;
            if constexpr (FORMAT == 1) {
                mxfp4_checkpoint_save<
                    cold_bf_t, full_tt_bf<32>, rt_bf<32, 32>>(
                        active, cold_bf);
            } else {
                mxfp4_checkpoint_save<
                    cold_hf_t, full_tt_hf<32>, rt_hf<32, 32>>(
                        active, cold_hf);
            }
            if (threadIdx.x == 0) {
                const uint64_t phase_end = clock64();
                save_cycles += phase_end - phase_start;
                phase_start = phase_end;
            }
            mxfp4_checkpoint_clear(active);
            if (threadIdx.x == 0) {
                const uint64_t phase_end = clock64();
                clear_cycles += phase_end - phase_start;
                phase_start = phase_end;
            }
            if constexpr (FORMAT == 1) {
                mxfp4_checkpoint_restore<
                    cold_bf_t, full_tt_bf<32>, rt_bf<32, 32>>(
                        active, cold_bf);
            } else {
                mxfp4_checkpoint_restore<
                    cold_hf_t, full_tt_hf<32>, rt_hf<32, 32>>(
                        active, cold_hf);
            }
            if (threadIdx.x == 0) {
                restore_cycles += clock64() - phase_start;
            }
        }
        warpgroup::sync(1);
        if (threadIdx.x == 0) {
            checkpoint_end = clock64();
        }
    }

    mxfp4_checkpoint_store_output(active, g.restored_o);
    __syncthreads();
    const uint64_t second_start = threadIdx.x == 0 ? clock64() : 0;
    issue_pv(p1_smem, v1_smem, 1, false);
    const uint64_t second_end = threadIdx.x == 0 ? clock64() : 0;
    mxfp4_checkpoint_store_output(active, g.final_o);
    __syncthreads();

    if (threadIdx.x == 0) {
        g.trace[0] = FORMAT;
        g.trace[1] = g.roundtrips;
        g.trace[2] = static_cast<int64_t>(first_end - first_start);
        g.trace[3] = static_cast<int64_t>(checkpoint_end - checkpoint_start);
        g.trace[4] = static_cast<int64_t>(second_end - second_start);
        g.trace[5] = active.addr;
        g.trace[6] = FORMAT == 1 ? cold_bf.addr : cold_hf.addr;
        g.trace[7] = p_scale_tm.addr;
        g.trace[8] = v_scale_tm.addr;
        g.trace[9] = 224;
        g.trace[10] = static_cast<int64_t>(save_cycles);
        g.trace[11] = static_cast<int64_t>(clear_cycles);
        g.trace[12] = static_cast<int64_t>(restore_cycles);
    }
}

template <int FORMAT>
static void launch_mxfp4_output_checkpoint_probe(
    const globals_mxfp4_output_checkpoint_probe &g
) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_output_checkpoint_probe<FORMAT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_mxfp4_output_checkpoint_probe<FORMAT>, g));
}

void dispatch_mxfp4_output_checkpoint_probe(
    at::Tensor p0_fp4,
    at::Tensor v0_fp4,
    at::Tensor p1_fp4,
    at::Tensor v1_fp4,
    at::Tensor scale_words,
    at::Tensor restored_o,
    at::Tensor final_o,
    at::Tensor trace,
    int format,
    int roundtrips
) {
    for (const at::Tensor *tensor : {
             &p0_fp4, &v0_fp4, &p1_fp4, &v1_fp4,
             &scale_words, &restored_o, &final_o, &trace}) {
        TORCH_CHECK(tensor->device().is_cuda(),
                    "checkpoint probe inputs must be CUDA tensors");
        TORCH_CHECK(tensor->is_contiguous(),
                    "checkpoint probe inputs must be contiguous");
    }
    for (const at::Tensor *tensor : {&p0_fp4, &v0_fp4, &p1_fp4, &v1_fp4}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Float4_e2m1fn_x2
                        && tensor->sizes() == at::IntArrayRef({128, 64}),
                    "checkpoint payloads must be float4_e2m1fn_x2 (128,64)");
    }
    TORCH_CHECK(scale_words.dtype() == at::ScalarType::Int
                    && scale_words.sizes() == at::IntArrayRef({4, 32, 4}),
                "scale_words must be int32 (4,32,4): P0,V0,P1,V1");
    for (const at::Tensor *tensor : {&restored_o, &final_o}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Float
                        && tensor->sizes() == at::IntArrayRef({128, 128}),
                    "checkpoint outputs must be float32 (128,128)");
    }
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long && trace.numel() >= 32,
                "trace must contain at least 32 int64 values");
    TORCH_CHECK(format >= 0 && format <= 2,
                "format must be 0 (reference), 1 (BF16), or 2 (FP16)");
    TORCH_CHECK(roundtrips == 0 || roundtrips == 1 || roundtrips == 2
                    || roundtrips == 4 || roundtrips == 8,
                "roundtrips must be one of 0,1,2,4,8");
    TORCH_CHECK(format == 0 || roundtrips > 0,
                "BF16/FP16 checkpoint modes require at least one roundtrip");

    globals_mxfp4_output_checkpoint_probe g{
        reinterpret_cast<const uint8_t *>(p0_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v0_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(p1_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v1_fp4.data_ptr()),
        scale_words.data_ptr<int32_t>(),
        restored_o.data_ptr<float>(),
        final_o.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        roundtrips,
    };
    if (format == 0) {
        launch_mxfp4_output_checkpoint_probe<0>(g);
    } else if (format == 1) {
        launch_mxfp4_output_checkpoint_probe<1>(g);
    } else {
        launch_mxfp4_output_checkpoint_probe<2>(g);
    }
}

struct config_mxfp4_output_checkpoint_dvo64_swap_probe {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dvo = 64;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 128;
};

struct globals_mxfp4_output_checkpoint_dvo64_swap_probe {
    using p_tile = st_fp4e2m1_2<128, 64>;
    using v_tile = st_fp4e2m1_2<64, 64>;
    using scale_tile = st_fp8e8m0<32, 16, false>;

    const uint8_t *p0_fp4;
    const uint8_t *v0_fp4;
    const uint8_t *p1_fp4;
    const uint8_t *v1_fp4;
    const int32_t *scale_words;
    float *output0;
    float *output1;
    int64_t *trace;
    int transitions;

    __host__ __inline__ dim3 grid() const { return dim3(1); }
    __host__ __inline__ dim3 block() const {
        return dim3(config_mxfp4_output_checkpoint_dvo64_swap_probe::NUM_THREADS);
    }
    __host__ __inline__ int dynamic_shared_memory() const { return 40 * 1024; }
};

template <int FORMAT>
__launch_bounds__(config_mxfp4_output_checkpoint_dvo64_swap_probe::NUM_THREADS, 1)
__global__ void kernel_mxfp4_output_checkpoint_dvo64_swap_probe(
    const __grid_constant__ globals_mxfp4_output_checkpoint_dvo64_swap_probe g
) {
    using C = config_mxfp4_output_checkpoint_dvo64_swap_probe;
    using G = globals_mxfp4_output_checkpoint_dvo64_swap_probe;
    using output_t = full_tt_fl<64>;
    using cold_bf_t = full_tt_bf<64>;
    using cold_hf_t = full_tt_hf<64>;
    using scale_t = full_tt_fp8e8m0<16>;
    static_assert(FORMAT >= 0 && FORMAT <= 2, "invalid DVO64 swap format");
    constexpr int P_SCALE_BASE = FORMAT == 0 ? 128 : 96;
    constexpr int V_SCALE_BASE = P_SCALE_BASE + 16;
    constexpr int TMEM_END = V_SCALE_BASE + 16;
    static_assert(TMEM_END <= 512, "DVO64 swap probe exceeds one TMEM allocation");

    tensor_allocator<1, 1> tm_alloc{};
    output_t active = tm_alloc.template allocate<output_t>(0);
    output_t reference1 = tm_alloc.template allocate<output_t>(64);
    cold_bf_t cold_bf = tm_alloc.template allocate<cold_bf_t>(64);
    cold_hf_t cold_hf = tm_alloc.template allocate<cold_hf_t>(64);
    scale_t p_scale_tm = tm_alloc.template allocate<scale_t>(P_SCALE_BASE);
    scale_t v_scale_tm = tm_alloc.template allocate<scale_t>(V_SCALE_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::p_tile &p0_smem = allocator.allocate<typename G::p_tile>();
    typename G::v_tile &v0_smem = allocator.allocate<typename G::v_tile>();
    typename G::p_tile &p1_smem = allocator.allocate<typename G::p_tile>();
    typename G::v_tile &v1_smem = allocator.allocate<typename G::v_tile>();
    typename G::scale_tile &p_scale_smem =
        allocator.allocate<typename G::scale_tile>();
    typename G::scale_tile &v_scale_smem =
        allocator.allocate<typename G::scale_tile>();

    __shared__ semaphore scale_copy_done;
    __shared__ semaphore pv_done;
    if (threadIdx.x == 0) {
        init_semaphore(scale_copy_done, 0, 1);
        init_semaphore(pv_done, 0, 1);
    }
    mxfp4_checkpoint_load_payload(p0_smem, g.p0_fp4);
    mxfp4_checkpoint_load_payload(v0_smem, g.v0_fp4);
    mxfp4_checkpoint_load_payload(p1_smem, g.p1_fp4);
    mxfp4_checkpoint_load_payload(v1_smem, g.v1_fp4);
    __syncthreads();
    fp4pv_shared_async_proxy_fence_cta();
    __syncthreads();

    const bool issue_warp = warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    int scale_phase = 0;
    int pv_phase = 0;
    auto issue_pv = [&](output_t &output, auto &p_smem, auto &v_smem,
                        int p_record, int v_record, bool clear) {
        if (warpgroup::warpid() == 0) {
            mxfp4_checkpoint_write_scale_tile(
                p_scale_smem, g.scale_words + p_record * 32 * 4);
        } else if (warpgroup::warpid() == 1) {
            mxfp4_checkpoint_write_scale_tile(
                v_scale_smem, g.scale_words + v_record * 32 * 4);
        }
        __syncthreads();
        fp4pv_shared_async_proxy_fence_cta();
        __syncthreads();
        if (issue_lane) {
            load_mxnv_scale_async<1>(p_scale_tm, p_scale_smem);
            load_mxnv_scale_async<1>(v_scale_tm, v_scale_smem);
            detail::tcgen05::commit<1>(scale_copy_done);
        }
        if (issue_warp) {
            wait(scale_copy_done, scale_phase);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
        }
        scale_phase ^= 1;
        if (issue_lane) {
            if (clear) {
                fp4pv_p_stage_ABt_split_k64<C, true>(
                    output, p_smem, v_smem, p_scale_tm, v_scale_tm, pv_done);
            } else {
                fp4pv_p_stage_ABt_split_k64<C, false>(
                    output, p_smem, v_smem, p_scale_tm, v_scale_tm, pv_done);
            }
        }
        if (issue_warp) {
            wait(pv_done, pv_phase);
            tensor_after_thread_sync();
        }
        pv_phase ^= 1;
        __syncthreads();
    };

    uint64_t save_cycles = 0;
    uint64_t swap_cycles = 0;
    if constexpr (FORMAT == 0) {
        issue_pv(active, p0_smem, v0_smem, 0, 1, true);
        issue_pv(reference1, p0_smem, v1_smem, 0, 3, true);
        for (int transition = 0; transition < g.transitions; ++transition) {
            issue_pv(active, p1_smem, v0_smem, 2, 1, false);
            issue_pv(reference1, p1_smem, v1_smem, 2, 3, false);
        }
        mxfp4_checkpoint_store_output(active, g.output0);
        mxfp4_checkpoint_store_output(reference1, g.output1);
    } else {
        issue_pv(active, p0_smem, v0_smem, 0, 1, true);
        uint64_t phase_start = threadIdx.x == 0 ? clock64() : 0;
        if constexpr (FORMAT == 1) {
            mxfp4_checkpoint_save<cold_bf_t, full_tt_bf<32>, rt_bf<32, 32>>(
                active, cold_bf);
        } else {
            mxfp4_checkpoint_save<cold_hf_t, full_tt_hf<32>, rt_hf<32, 32>>(
                active, cold_hf);
        }
        if (threadIdx.x == 0) {
            save_cycles = clock64() - phase_start;
        }
        issue_pv(active, p0_smem, v1_smem, 0, 3, true);
        for (int transition = 0; transition < g.transitions; ++transition) {
            phase_start = threadIdx.x == 0 ? clock64() : 0;
            if constexpr (FORMAT == 1) {
                mxfp4_checkpoint_swap<
                    cold_bf_t, full_tt_bf<32>, rt_bf<32, 32>>(active, cold_bf);
            } else {
                mxfp4_checkpoint_swap<
                    cold_hf_t, full_tt_hf<32>, rt_hf<32, 32>>(active, cold_hf);
            }
            if (threadIdx.x == 0) {
                swap_cycles += clock64() - phase_start;
            }
            issue_pv(active, p1_smem, v0_smem, 2, 1, false);
            phase_start = threadIdx.x == 0 ? clock64() : 0;
            if constexpr (FORMAT == 1) {
                mxfp4_checkpoint_swap<
                    cold_bf_t, full_tt_bf<32>, rt_bf<32, 32>>(active, cold_bf);
            } else {
                mxfp4_checkpoint_swap<
                    cold_hf_t, full_tt_hf<32>, rt_hf<32, 32>>(active, cold_hf);
            }
            if (threadIdx.x == 0) {
                swap_cycles += clock64() - phase_start;
            }
            issue_pv(active, p1_smem, v1_smem, 2, 3, false);
        }
        phase_start = threadIdx.x == 0 ? clock64() : 0;
        if constexpr (FORMAT == 1) {
            mxfp4_checkpoint_swap<
                cold_bf_t, full_tt_bf<32>, rt_bf<32, 32>>(active, cold_bf);
        } else {
            mxfp4_checkpoint_swap<
                cold_hf_t, full_tt_hf<32>, rt_hf<32, 32>>(active, cold_hf);
        }
        if (threadIdx.x == 0) {
            swap_cycles += clock64() - phase_start;
        }
        mxfp4_checkpoint_store_output(active, g.output0);
        phase_start = threadIdx.x == 0 ? clock64() : 0;
        if constexpr (FORMAT == 1) {
            mxfp4_checkpoint_swap<
                cold_bf_t, full_tt_bf<32>, rt_bf<32, 32>>(active, cold_bf);
        } else {
            mxfp4_checkpoint_swap<
                cold_hf_t, full_tt_hf<32>, rt_hf<32, 32>>(active, cold_hf);
        }
        if (threadIdx.x == 0) {
            swap_cycles += clock64() - phase_start;
        }
        mxfp4_checkpoint_store_output(active, g.output1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        g.trace[0] = FORMAT;
        g.trace[1] = g.transitions;
        g.trace[2] = static_cast<int64_t>(save_cycles);
        g.trace[3] = static_cast<int64_t>(swap_cycles);
        g.trace[4] = active.addr;
        g.trace[5] = FORMAT == 1 ? cold_bf.addr : cold_hf.addr;
        g.trace[6] = p_scale_tm.addr;
        g.trace[7] = v_scale_tm.addr;
        g.trace[8] = TMEM_END;
    }
}

template <int FORMAT>
static void launch_mxfp4_output_checkpoint_dvo64_swap_probe(
    const globals_mxfp4_output_checkpoint_dvo64_swap_probe &g
) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_output_checkpoint_dvo64_swap_probe<FORMAT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_mxfp4_output_checkpoint_dvo64_swap_probe<FORMAT>, g));
}

void dispatch_mxfp4_output_checkpoint_dvo64_swap_probe(
    at::Tensor p0_fp4,
    at::Tensor v0_fp4,
    at::Tensor p1_fp4,
    at::Tensor v1_fp4,
    at::Tensor scale_words,
    at::Tensor output0,
    at::Tensor output1,
    at::Tensor trace,
    int format,
    int transitions
) {
    for (const at::Tensor *tensor : {
             &p0_fp4, &v0_fp4, &p1_fp4, &v1_fp4,
             &scale_words, &output0, &output1, &trace}) {
        TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "DVO64 swap probe tensors must be contiguous CUDA tensors");
    }
    for (const at::Tensor *tensor : {&p0_fp4, &p1_fp4}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Float4_e2m1fn_x2
                        && tensor->sizes() == at::IntArrayRef({128, 64}),
                    "DVO64 swap P payloads must be float4_e2m1fn_x2 (128,64)");
    }
    for (const at::Tensor *tensor : {&v0_fp4, &v1_fp4}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Float4_e2m1fn_x2
                        && tensor->sizes() == at::IntArrayRef({64, 64}),
                    "DVO64 swap V payloads must be float4_e2m1fn_x2 (64,64)");
    }
    TORCH_CHECK(scale_words.dtype() == at::ScalarType::Int
                    && scale_words.sizes() == at::IntArrayRef({4, 32, 4}),
                "DVO64 scale_words must be int32 (4,32,4)");
    for (const at::Tensor *tensor : {&output0, &output1}) {
        TORCH_CHECK(tensor->dtype() == at::ScalarType::Float
                        && tensor->sizes() == at::IntArrayRef({128, 64}),
                    "DVO64 swap outputs must be float32 (128,64)");
    }
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long && trace.numel() >= 32,
                "DVO64 swap trace must contain at least 32 int64 values");
    TORCH_CHECK(format >= 0 && format <= 2,
                "DVO64 swap format must be 0, 1, or 2");
    TORCH_CHECK(transitions == 1 || transitions == 2 || transitions == 4
                    || transitions == 8,
                "DVO64 swap transitions must be one of 1,2,4,8");

    globals_mxfp4_output_checkpoint_dvo64_swap_probe g{
        reinterpret_cast<const uint8_t *>(p0_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v0_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(p1_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v1_fp4.data_ptr()),
        scale_words.data_ptr<int32_t>(),
        output0.data_ptr<float>(),
        output1.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        transitions,
    };
    if (format == 0) {
        launch_mxfp4_output_checkpoint_dvo64_swap_probe<0>(g);
    } else if (format == 1) {
        launch_mxfp4_output_checkpoint_dvo64_swap_probe<1>(g);
    } else {
        launch_mxfp4_output_checkpoint_dvo64_swap_probe<2>(g);
    }
}

#endif

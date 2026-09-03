#pragma once

#ifndef TK_FA4_RAW_FP4_FACTOR_PROBE_BUILD
#define TK_FA4_RAW_FP4_FACTOR_PROBE_BUILD 0
#endif

#if TK_FA4_RAW_FP4_FACTOR_PROBE_BUILD

template <int _K, bool _AUDIT_PADDING>
struct config_raw_fp4_factor_probe {
    static_assert(_K == 32 || _K == 64 || _K == 128,
                  "raw FP4 probe instantiates logical K32/K64/K128 candidates");
    static constexpr int Mb = 128;
    static constexpr int Nb = _K;
    static constexpr int Dvo = 128;
    static constexpr bool AUDIT_PADDING = _AUDIT_PADDING;
    static constexpr int ACC_DVO = Dvo;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_THREADS = NUM_WARPGROUPS * WARPGROUP_WARPS * WARP_THREADS;
};

template <typename C>
struct globals_raw_fp4_factor_probe {
    // Raw f8f6f4 shared operands hold 16 E2M1 values in 8 bytes followed by
    // 8 bytes of padding. The physical shared row is therefore one byte per
    // logical K element, twice the ordinary dense e2m1x2 storage.
    using a_tile = st_fp4e2m1_2<C::Mb, C::Nb>;
    using b_tile = st_fp4e2m1_2<C::Dvo, C::Nb>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_tile));
        add((int)sizeof(b_tile));
        return align_up_constexpr(bytes, 1024) + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();
    const uint8_t *a_fp4;
    const uint8_t *b_fp4;
    float *o;
    int32_t *padding_errors;
    int rows;
    int a_packed_cols;
    int b_packed_cols;

    __host__ __inline__ dim3 grid() const { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() const { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C, typename Tile>
__device__ inline void load_raw_fp4_padded_tile(
    Tile &tile,
    const uint8_t *src,
    int src_row_stride,
    int src_row_base
) {
    fp4pv_shape_zero_tile(tile);
    __syncthreads();
    constexpr int GROUP_LOGICAL_K = 16;
    constexpr int GROUP_DATA_BYTES = GROUP_LOGICAL_K / 2;
    constexpr int GROUP_SHARED_BYTES = 16;
    constexpr int GROUPS_PER_ROW = C::Nb / GROUP_LOGICAL_K;
    static_assert(C::Nb % GROUP_LOGICAL_K == 0);
    static_assert(Tile::cols == GROUPS_PER_ROW * GROUP_SHARED_BYTES);
    uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * GROUPS_PER_ROW;
         linear += blockDim.x) {
        const int row = linear / GROUPS_PER_ROW;
        const int group = linear - row * GROUPS_PER_ROW;
        const uint2 value = *reinterpret_cast<const uint2 *>(
            src + (src_row_base + row) * src_row_stride
                + group * GROUP_DATA_BYTES);
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                     :: "r"(Tile::idx(smem_base, {row, group * GROUP_SHARED_BYTES})),
                        "r"(value.x), "r"(value.y));
    }
}

template <typename C, typename Tile>
__device__ inline int count_raw_fp4_padding_errors(const Tile &tile) {
    constexpr int GROUP_LOGICAL_K = 16;
    constexpr int GROUP_SHARED_BYTES = 16;
    constexpr int GROUP_PADDING_BYTES = 8;
    constexpr int GROUPS_PER_ROW = C::Nb / GROUP_LOGICAL_K;
    uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    int errors = 0;
    for (int linear = threadIdx.x;
         linear < Tile::rows * GROUPS_PER_ROW * GROUP_PADDING_BYTES;
         linear += blockDim.x) {
        const int byte = linear % GROUP_PADDING_BYTES;
        const int group_linear = linear / GROUP_PADDING_BYTES;
        const int group = group_linear % GROUPS_PER_ROW;
        const int row = group_linear / GROUPS_PER_ROW;
        uint32_t value;
        asm volatile("ld.shared.u8 %0, [%1];"
                     : "=r"(value)
                     : "r"(Tile::idx(
                           smem_base,
                           {row, group * GROUP_SHARED_BYTES + 8 + byte})));
        errors += value != 0;
    }
    return errors;
}

template <typename C, typename Output, typename ATile, typename BTile>
__device__ inline void raw_fp4_mm_abt_k32_chunks(
    Output &output,
    const ATile &a,
    const BTile &b,
    semaphore &done
) {
    using fp4_t = fp4e2m1_2;
    constexpr int K32_CHUNKS = C::Nb / 32;
    static_assert(C::Nb % 32 == 0, "raw kind::f8f6f4 reduces logical K32");
    static_assert(K32_CHUNKS <= 4,
                  "probe descriptor stepping stays within one shared swizzle unit");
    static_assert(ATile::swizzle_bytes >= K32_CHUNKS * 32);
    static_assert(BTile::swizzle_bytes >= K32_CHUNKS * 32);

    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<float, fp4_t,
                                                 C::Mb, C::Dvo,
                                                 false, false, false>();
    st_descriptor<ATile, transpose::N> a_desc(a);
    st_descriptor<BTile, transpose::N> b_desc(b);

    // PTX specifies dense kind::f8f6f4 E2M1 as K=32. Its shared-memory
    // operand consumes two 8-byte data + 8-byte padding groups per call.
    #pragma unroll
    for (int chunk = 0; chunk < K32_CHUNKS; ++chunk) {
        if (chunk == 0) {
            detail::tcgen05::st_st<fp4_t, 0, 1>(
                output.addr,
                a_desc.chunk_descriptor(chunk),
                b_desc.chunk_descriptor(chunk),
                idesc);
        } else {
            detail::tcgen05::st_st<fp4_t, 1, 1>(
                output.addr,
                a_desc.chunk_descriptor(chunk),
                b_desc.chunk_descriptor(chunk),
                idesc);
        }
    }
    detail::tcgen05::commit<1>(done);
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_raw_fp4_factor_probe(
    const __grid_constant__ globals_raw_fp4_factor_probe<C> g
) {
    using G = globals_raw_fp4_factor_probe<C>;
    using output_t = tt<float, C::Mb, C::Dvo>;
    static_assert(C::Nb % 32 == 0, "raw FP4 TCGEN reduction is issued in logical K32 chunks");
    static_assert(output_t::cols == 128, "probe consumes exactly 128 TMEM columns");

    tensor_allocator<1, 1> tm_alloc{};
    output_t tt_output = tm_alloc.template allocate<output_t>(0);
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_tile &a_smem = al.allocate<typename G::a_tile>();
    typename G::b_tile &b_smem = al.allocate<typename G::b_tile>();
    __shared__ semaphore mma_done;
    if (threadIdx.x == 0) {
        init_semaphore(mma_done, 0, 1);
    }
    __syncthreads();

    const int row_tile = (int)blockIdx.x;
    load_raw_fp4_padded_tile<C>(
        a_smem, g.a_fp4, g.a_packed_cols, row_tile * C::Mb);
    load_raw_fp4_padded_tile<C>(
        b_smem, g.b_fp4, g.b_packed_cols, 0);
    __syncthreads();
    if constexpr (C::AUDIT_PADDING) {
        const int padding_errors = count_raw_fp4_padding_errors<C>(a_smem)
            + count_raw_fp4_padding_errors<C>(b_smem);
        if (padding_errors != 0) {
            atomicAdd(g.padding_errors, padding_errors);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    const bool issuer = warpgroup::groupid() == 1
        && warpgroup::warpid() == 0
        && warp::elect_leader();
    if (issuer) {
        tensor_before_thread_sync();
        raw_fp4_mm_abt_k32_chunks<C>(tt_output, a_smem, b_smem, mma_done);
    }

    if (warpgroup::groupid() == 0) {
        wait(mma_done, 0);
        tensor_after_thread_sync();
        warpgroup::sync(1);
        const int row = row_tile * C::Mb + warpgroup::laneid();
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
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
                : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                reinterpret_cast<float2 *>(&g.o[row * C::Dvo + col])[i] = values[i];
            }
        }
    }
    __syncthreads();
}

template <int K, bool AUDIT_PADDING>
static void launch_raw_fp4_factor_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor o,
    at::Tensor padding_errors
) {
    using C = config_raw_fp4_factor_probe<K, AUDIT_PADDING>;
    using G = globals_raw_fp4_factor_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        o.data_ptr<float>(),
        padding_errors.data_ptr<int32_t>(),
        (int)a_fp4.size(0),
        (int)a_fp4.size(1),
        (int)b_fp4.size(1),
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_factor_probe<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_raw_fp4_factor_probe<C>, g));
}

void dispatch_raw_fp4_factor_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor o,
    at::Tensor padding_errors,
    int k,
    bool audit_padding
) {
    CHECK_INPUT(a_fp4);
    CHECK_INPUT(b_fp4);
    CHECK_INPUT(o);
    CHECK_INPUT(padding_errors);
    TORCH_CHECK(a_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "a_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(b_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "b_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(o.dtype() == at::ScalarType::Float, "o must be float32");
    TORCH_CHECK(padding_errors.dtype() == at::ScalarType::Int,
                "padding_errors must be int32");
    TORCH_CHECK(padding_errors.numel() == 1,
                "padding_errors must contain one scalar");
    TORCH_CHECK(k == 32 || k == 64 || k == 128,
                "raw probe supports logical K32, K64, or K128");
    TORCH_CHECK(a_fp4.dim() == 2 && b_fp4.dim() == 2 && o.dim() == 2,
                "raw probe tensors must be rank two");
    TORCH_CHECK(a_fp4.size(0) > 0 && a_fp4.size(0) % 128 == 0,
                "A rows must be a positive multiple of 128");
    TORCH_CHECK(a_fp4.size(1) == k / 2, "A must have exactly K/2 packed columns");
    TORCH_CHECK(b_fp4.size(0) == 128 && b_fp4.size(1) == k / 2,
                "B must have shape (128,K/2)");
    TORCH_CHECK(o.size(0) == a_fp4.size(0) && o.size(1) == 128,
                "O must have shape (A.rows,128)");
    if (k == 32 && audit_padding) {
        launch_raw_fp4_factor_probe<32, true>(a_fp4, b_fp4, o, padding_errors);
    } else if (k == 32) {
        launch_raw_fp4_factor_probe<32, false>(a_fp4, b_fp4, o, padding_errors);
    } else if (k == 64 && audit_padding) {
        launch_raw_fp4_factor_probe<64, true>(a_fp4, b_fp4, o, padding_errors);
    } else if (k == 64) {
        launch_raw_fp4_factor_probe<64, false>(a_fp4, b_fp4, o, padding_errors);
    } else if (audit_padding) {
        launch_raw_fp4_factor_probe<128, true>(a_fp4, b_fp4, o, padding_errors);
    } else {
        launch_raw_fp4_factor_probe<128, false>(a_fp4, b_fp4, o, padding_errors);
    }
}

template <int _DESCRIPTOR_MODE>
struct config_raw_fp4_isa_extent_probe {
    static_assert(_DESCRIPTOR_MODE >= 0 && _DESCRIPTOR_MODE <= 2);
    static constexpr int Mb = 128;
    static constexpr int Nb = 64;
    static constexpr int Dvo = 128;
    static constexpr int DESCRIPTOR_MODE = _DESCRIPTOR_MODE;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_THREADS =
        NUM_WARPGROUPS * WARPGROUP_WARPS * WARP_THREADS;
};

template <typename C>
struct globals_raw_fp4_isa_extent_probe {
    using a_tile = st_fp4e2m1_2<C::Mb, C::Nb>;
    using b_tile = st_fp4e2m1_2<C::Dvo, C::Nb>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(a_tile));
        add((int)sizeof(b_tile));
        return align_up_constexpr(bytes, 1024) + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();
    const uint8_t *a_fp4;
    const uint8_t *b_fp4;
    float *o;
    int rows;
    int a_packed_cols;
    int b_packed_cols;

    __host__ __inline__ dim3 grid() const { return dim3(rows / C::Mb); }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() const {
        return DYNAMIC_SHARED_MEMORY;
    }
};

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_raw_fp4_isa_extent_probe(
    const __grid_constant__ globals_raw_fp4_isa_extent_probe<C> g
) {
    using G = globals_raw_fp4_isa_extent_probe<C>;
    using output_t = tt<float, C::Mb, C::Dvo>;
    using fp4_t = fp4e2m1_2;

    tensor_allocator<1, 1> tm_alloc{};
    output_t tt_output = tm_alloc.template allocate<output_t>(0);
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::a_tile &a_smem = al.allocate<typename G::a_tile>();
    typename G::b_tile &b_smem = al.allocate<typename G::b_tile>();
    __shared__ semaphore mma_done;
    if (threadIdx.x == 0) {
        init_semaphore(mma_done, 0, 1);
    }
    __syncthreads();

    const int row_tile = (int)blockIdx.x;
    load_raw_fp4_padded_tile<C>(
        a_smem, g.a_fp4, g.a_packed_cols, row_tile * C::Mb);
    load_raw_fp4_padded_tile<C>(
        b_smem, g.b_fp4, g.b_packed_cols, 0);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    const bool issuer = warpgroup::groupid() == 1
        && warpgroup::warpid() == 0
        && warp::elect_leader();
    if (issuer) {
        constexpr uint32_t official_idesc =
            detail::tcgen05::instruction_descriptor<float, fp4_t,
                                                     C::Mb, C::Dvo,
                                                     false, false, false>();
        constexpr uint32_t idesc = C::DESCRIPTOR_MODE == 1
            ? official_idesc | (1u << 31)
            : official_idesc;
        st_descriptor<typename G::a_tile, transpose::N> a_desc(a_smem);
        st_descriptor<typename G::b_tile, transpose::N> b_desc(b_smem);
        tensor_before_thread_sync();
        detail::tcgen05::st_st<fp4_t, 0, 1>(
            tt_output.addr,
            a_desc.chunk_descriptor(0),
            b_desc.chunk_descriptor(0),
            idesc);
        if constexpr (C::DESCRIPTOR_MODE == 2) {
            detail::tcgen05::st_st<fp4_t, 1, 1>(
                tt_output.addr,
                a_desc.chunk_descriptor(1),
                b_desc.chunk_descriptor(1),
                official_idesc);
        }
        detail::tcgen05::commit<1>(mma_done);
    }

    if (warpgroup::groupid() == 0) {
        wait(mma_done, 0);
        tensor_after_thread_sync();
        warpgroup::sync(1);
        const int row = row_tile * C::Mb + warpgroup::laneid();
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
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
                : "r"(tt_output.addr
                      + ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                reinterpret_cast<float2 *>(
                    &g.o[row * C::Dvo + col])[i] = values[i];
            }
        }
    }
    __syncthreads();
}

template <int DESCRIPTOR_MODE>
static void launch_raw_fp4_isa_extent_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor o
) {
    using C = config_raw_fp4_isa_extent_probe<DESCRIPTOR_MODE>;
    using G = globals_raw_fp4_isa_extent_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(a_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(b_fp4.data_ptr()),
        o.data_ptr<float>(),
        (int)a_fp4.size(0),
        (int)a_fp4.size(1),
        (int)b_fp4.size(1),
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_isa_extent_probe<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_raw_fp4_isa_extent_probe<C>, g));
}

void dispatch_raw_fp4_isa_extent_probe(
    at::Tensor a_fp4,
    at::Tensor b_fp4,
    at::Tensor o,
    int descriptor_mode
) {
    CHECK_INPUT(a_fp4);
    CHECK_INPUT(b_fp4);
    CHECK_INPUT(o);
    TORCH_CHECK(a_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "a_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(b_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "b_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(o.dtype() == at::ScalarType::Float, "o must be float32");
    TORCH_CHECK(a_fp4.dim() == 2 && b_fp4.dim() == 2 && o.dim() == 2,
                "raw ISA extent tensors must be rank two");
    TORCH_CHECK(a_fp4.size(0) > 0 && a_fp4.size(0) % 128 == 0,
                "A rows must be a positive multiple of 128");
    TORCH_CHECK(a_fp4.size(1) == 32, "A must have 64 logical K values");
    TORCH_CHECK(b_fp4.size(0) == 128 && b_fp4.size(1) == 32,
                "B must have shape (128,32 packed bytes)");
    TORCH_CHECK(o.size(0) == a_fp4.size(0) && o.size(1) == 128,
                "O must have shape (A.rows,128)");
    TORCH_CHECK(descriptor_mode >= 0 && descriptor_mode <= 2,
                "descriptor_mode must be 0 (official), 1 (bit31), or 2 (two K32 control)");
    if (descriptor_mode == 0) {
        launch_raw_fp4_isa_extent_probe<0>(a_fp4, b_fp4, o);
    } else if (descriptor_mode == 1) {
        launch_raw_fp4_isa_extent_probe<1>(a_fp4, b_fp4, o);
    } else {
        launch_raw_fp4_isa_extent_probe<2>(a_fp4, b_fp4, o);
    }
}

#endif

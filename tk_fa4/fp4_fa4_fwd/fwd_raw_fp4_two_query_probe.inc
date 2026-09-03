#pragma once

#if TK_FA4_RAW_FP4_FACTOR_PROBE_BUILD

template <bool _AUDIT>
struct config_raw_fp4_two_query_probe {
    static constexpr bool AUDIT = _AUDIT;
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dqk = 192;
    static constexpr int Dvo = 128;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_THREADS = NUM_WARPGROUPS * WARPGROUP_WARPS * WARP_THREADS;
};

template <typename C>
struct globals_raw_fp4_two_query_probe {
    using q_tile = st_fp4e2m1_2<C::Mb, C::Dqk>;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk>;
    using p_tile = st_fp4e2m1_2<C::Mb, C::Nb>;
    using v_tile = st_fp4e2m1_2<C::Dvo, C::Nb>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int allocation) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + allocation;
        };
        add(2 * (int)sizeof(q_tile));
        add((int)sizeof(k_tile));
        add(2 * (int)sizeof(p_tile));
        add((int)sizeof(v_tile));
        return align_up_constexpr(bytes, 1024) + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();
    const uint8_t *q_fp4;
    const uint8_t *k_fp4;
    const uint8_t *v_fp4;
    float *o;
    float *lse;
    int64_t *trace;
    int heads;
    int seqlen;

    __host__ __inline__ dim3 grid() const {
        const int query_tiles = seqlen / C::Mb;
        return dim3(heads * ((query_tiles + 1) / 2));
    }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() const {
        return DYNAMIC_SHARED_MEMORY;
    }
};

template <int LOGICAL_K, typename Tile>
__device__ inline void load_raw_fp4_padded_payload(
    Tile &tile,
    const uint8_t *src,
    int src_row_stride,
    int src_row_base
) {
    constexpr int GROUP_LOGICAL_K = 16;
    constexpr int GROUP_DATA_BYTES = 8;
    constexpr int GROUP_SHARED_BYTES = 16;
    constexpr int GROUPS_PER_ROW = LOGICAL_K / GROUP_LOGICAL_K;
    static_assert(LOGICAL_K % GROUP_LOGICAL_K == 0);
    static_assert(Tile::cols == GROUPS_PER_ROW * GROUP_SHARED_BYTES);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = threadIdx.x;
         linear < Tile::rows * GROUPS_PER_ROW;
         linear += blockDim.x) {
        const int row = linear / GROUPS_PER_ROW;
        const int group = linear - row * GROUPS_PER_ROW;
        const uint2 value = *reinterpret_cast<const uint2 *>(
            src + (src_row_base + row) * src_row_stride
                + group * GROUP_DATA_BYTES);
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                     :: "r"(Tile::idx(
                            smem_base,
                            {row, group * GROUP_SHARED_BYTES})),
                        "r"(value.x), "r"(value.y));
    }
}

template <int LOGICAL_K, typename Tile>
__device__ inline int count_raw_fp4_two_query_padding(const Tile &tile) {
    constexpr int GROUP_LOGICAL_K = 16;
    constexpr int GROUP_SHARED_BYTES = 16;
    constexpr int GROUP_PADDING_BYTES = 8;
    constexpr int GROUPS_PER_ROW = LOGICAL_K / GROUP_LOGICAL_K;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
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

template <int K, int M, int N, typename Output, typename ATile, typename BTile>
__device__ inline void raw_fp4_mma_abt(
    Output &output,
    const ATile &a,
    const BTile &b,
    bool accumulate,
    semaphore &done
) {
    using fp4_t = fp4e2m1_2;
    constexpr int K32_CHUNKS = K / 32;
    static_assert(K % 32 == 0);
    static_assert(ATile::cols == K && BTile::cols == K);
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<float, fp4_t,
                                                 M, N,
                                                 false, false, false>();
    st_descriptor<ATile, transpose::N> a_desc(a);
    st_descriptor<BTile, transpose::N> b_desc(b);
    #pragma unroll
    for (int chunk = 0; chunk < K32_CHUNKS; ++chunk) {
        if (chunk == 0 && !accumulate) {
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

template <typename Tile>
__device__ inline int store_raw_binary_p_quarter(
    Tile &tile,
    const float2 (&scores)[16],
    int row,
    int quarter,
    int key_tile,
    int query_row
) {
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    int count = 0;
    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        uint32_t words[2] = {0u, 0u};
        #pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            const int pair = group * 8 + byte;
            const int key_col = key_tile * 128 + quarter * 32 + pair * 2;
            const bool valid_x = key_col <= query_row;
            const bool valid_y = key_col + 1 <= query_row;
            const bool px = valid_x && scores[pair].x > 0.0f;
            const bool py = valid_y && scores[pair].y > 0.0f;
            count += (int)px + (int)py;
            const uint32_t packed = (px ? 2u : 0u) | (py ? 0x20u : 0u);
            words[byte / 4] |= packed << ((byte % 4) * 8);
        }
        asm volatile("st.shared.v2.b32 [%0], {%1, %2};"
                     :: "r"(Tile::idx(
                            smem_base,
                            {row, quarter * 32 + group * 16})),
                        "r"(words[0]), "r"(words[1]));
    }
    return count;
}

template <typename C, typename ScoreTile, typename PTile>
__device__ inline int consume_raw_score_to_binary_p(
    ScoreTile &score,
    PTile &p,
    int key_tile,
    int query_tile,
    semaphore &done
) {
    wait(done, key_tile & 1);
    tensor_after_thread_sync();
    const int row = warpgroup::laneid();
    const int query_row = query_tile * C::Mb + row;
    const uint32_t score_row_base =
        score.addr + ((warpgroup::warpid() * 32) << 16);
    int count = 0;
    #pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
        float2 scores[16];
        fp4pv_load_score_quarter_tmem(scores, score_row_base, quarter);
        tensor_load_wait();
        count += store_raw_binary_p_quarter(
            p, scores, row, quarter, key_tile, query_row);
    }
    tensor_before_thread_sync();
    return count;
}

template <typename C, typename OutputTile>
__device__ inline void store_raw_two_query_output(
    const globals_raw_fp4_two_query_probe<C> &g,
    OutputTile &output,
    int query_tile,
    int h
) {
    const int row = warpgroup::laneid();
    const int query_row = query_tile * C::Mb + row;
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
            : "r"(output.addr + ((warpgroup::warpid() * 32) << 16) + col));
        tensor_load_wait();
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            reinterpret_cast<float2 *>(
                &g.o[(query_row * g.heads + h) * C::Dvo + col])[i] = values[i];
        }
    }
}

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_raw_fp4_two_query_probe(
    const __grid_constant__ globals_raw_fp4_two_query_probe<C> g
) {
    using G = globals_raw_fp4_two_query_probe<C>;
    using tile_t = tt<float, C::Mb, 128>;
    static_assert(C::NUM_THREADS == 512);
    static_assert(sizeof(typename G::q_tile) == 128 * 192);
    static_assert(sizeof(typename G::k_tile) == 128 * 192);
    static_assert(sizeof(typename G::p_tile) == 128 * 128);
    static_assert(sizeof(typename G::v_tile) == 128 * 128);
    static_assert(G::DYNAMIC_SHARED_MEMORY == 123904);

    constexpr int SCORE0_BASE = 0;
    constexpr int SCORE1_BASE = 128;
    constexpr int OUTPUT0_BASE = 256;
    constexpr int OUTPUT1_BASE = 384;
    static_assert(OUTPUT1_BASE + 128 == MAX_TENSOR_COLS);

    const int query_tiles = g.seqlen / C::Mb;
    const int pairs_per_head = (query_tiles + 1) / 2;
    const int h = (int)blockIdx.x / pairs_per_head;
    const int pair = (int)blockIdx.x - h * pairs_per_head;
    const int query_tile[2] = {2 * pair, 2 * pair + 1};
    const bool query_active[2] = {true, query_tile[1] < query_tiles};
    const int key_tiles = query_active[1] ? query_tile[1] + 1 : query_tile[0] + 1;

    tensor_allocator<1, 1> tm_alloc{};
    tile_t tt_score0 = tm_alloc.template allocate<tile_t>(SCORE0_BASE);
    tile_t tt_score1 = tm_alloc.template allocate<tile_t>(SCORE1_BASE);
    tile_t tt_output0 = tm_alloc.template allocate<tile_t>(OUTPUT0_BASE);
    tile_t tt_output1 = tm_alloc.template allocate<tile_t>(OUTPUT1_BASE);

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile (&q_smem)[2] = al.allocate<typename G::q_tile, 2>();
    typename G::k_tile &k_smem = al.allocate<typename G::k_tile>();
    typename G::p_tile (&p_smem)[2] = al.allocate<typename G::p_tile, 2>();
    typename G::v_tile &v_smem = al.allocate<typename G::v_tile>();

    __shared__ semaphore qk_done[2];
    __shared__ semaphore pv_done[2];
    if (threadIdx.x == 0) {
        init_semaphore(qk_done[0], 0, 1);
        init_semaphore(qk_done[1], 0, 1);
        init_semaphore(pv_done[0], 0, 1);
        init_semaphore(pv_done[1], 0, 1);
    }
    fp4pv_shape_zero_tile(q_smem[0]);
    fp4pv_shape_zero_tile(q_smem[1]);
    fp4pv_shape_zero_tile(k_smem);
    fp4pv_shape_zero_tile(p_smem[0]);
    fp4pv_shape_zero_tile(p_smem[1]);
    fp4pv_shape_zero_tile(v_smem);
    __syncthreads();

    const uint8_t *q_head =
        g.q_fp4 + (int64_t)h * g.seqlen * (C::Dqk / 2);
    load_raw_fp4_padded_payload<C::Dqk>(
        q_smem[0], q_head, C::Dqk / 2, query_tile[0] * C::Mb);
    if (query_active[1]) {
        load_raw_fp4_padded_payload<C::Dqk>(
            q_smem[1], q_head, C::Dqk / 2, query_tile[1] * C::Mb);
    }
    __syncthreads();
    if constexpr (C::AUDIT) {
        const int errors = count_raw_fp4_two_query_padding<C::Dqk>(q_smem[0])
            + count_raw_fp4_two_query_padding<C::Dqk>(q_smem[1]);
        if (errors != 0) {
            atomicAdd(reinterpret_cast<unsigned long long *>(
                          &g.trace[(int)blockIdx.x * 8 + 6]),
                      (unsigned long long)errors);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    float row_count = 0.0f;
    const int role = warpgroup::groupid();
    const bool issuer = role == 2 && warpgroup::warpid() == 0
        && warp::elect_leader();

    #pragma unroll 1
    for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
        const uint8_t *k_head =
            g.k_fp4 + (int64_t)h * g.seqlen * (C::Dqk / 2);
        load_raw_fp4_padded_payload<C::Dqk>(
            k_smem, k_head, C::Dqk / 2, key_tile * C::Nb);
        const uint8_t *v_tile_src =
            g.v_fp4 + (int64_t)h * C::Dvo * (g.seqlen / 2)
                + key_tile * (C::Nb / 2);
        load_raw_fp4_padded_payload<C::Nb>(
            v_smem, v_tile_src, g.seqlen / 2, 0);
        __syncthreads();
        if constexpr (C::AUDIT) {
            const int errors = count_raw_fp4_two_query_padding<C::Dqk>(k_smem)
                + count_raw_fp4_two_query_padding<C::Nb>(v_smem);
            if (errors != 0) {
                atomicAdd(reinterpret_cast<unsigned long long *>(
                              &g.trace[(int)blockIdx.x * 8 + 6]),
                          (unsigned long long)errors);
            }
            if (threadIdx.x == 0) {
                ++g.trace[(int)blockIdx.x * 8 + 0];
                ++g.trace[(int)blockIdx.x * 8 + 1];
            }
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();

        const bool q0_uses_key = key_tile <= query_tile[0];
        const bool q1_uses_key = query_active[1] && key_tile <= query_tile[1];
        if (issuer) {
            tensor_before_thread_sync();
            if (q0_uses_key) {
                raw_fp4_mma_abt<C::Dqk, C::Mb, C::Nb>(
                    tt_score0, q_smem[0], k_smem, false, qk_done[0]);
                if constexpr (C::AUDIT) {
                    ++g.trace[(int)blockIdx.x * 8 + 2];
                }
            }
            if (q1_uses_key) {
                raw_fp4_mma_abt<C::Dqk, C::Mb, C::Nb>(
                    tt_score1, q_smem[1], k_smem, false, qk_done[1]);
                if constexpr (C::AUDIT) {
                    ++g.trace[(int)blockIdx.x * 8 + 3];
                }
            }
        }

        if (role == 0 && q0_uses_key) {
            row_count += (float)consume_raw_score_to_binary_p<C>(
                tt_score0, p_smem[0], key_tile, query_tile[0], qk_done[0]);
        } else if (role == 1 && q1_uses_key) {
            row_count += (float)consume_raw_score_to_binary_p<C>(
                tt_score1, p_smem[1], key_tile, query_tile[1], qk_done[1]);
        }
        __syncthreads();
        if constexpr (C::AUDIT) {
            const int errors = count_raw_fp4_two_query_padding<C::Nb>(p_smem[0])
                + count_raw_fp4_two_query_padding<C::Nb>(p_smem[1]);
            if (errors != 0) {
                atomicAdd(reinterpret_cast<unsigned long long *>(
                              &g.trace[(int)blockIdx.x * 8 + 6]),
                          (unsigned long long)errors);
            }
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();

        if (issuer) {
            tensor_before_thread_sync();
            if (q0_uses_key) {
                raw_fp4_mma_abt<C::Nb, C::Mb, C::Dvo>(
                    tt_output0, p_smem[0], v_smem,
                    key_tile != 0, pv_done[0]);
                if constexpr (C::AUDIT) {
                    ++g.trace[(int)blockIdx.x * 8 + 4];
                }
            }
            if (q1_uses_key) {
                raw_fp4_mma_abt<C::Nb, C::Mb, C::Dvo>(
                    tt_output1, p_smem[1], v_smem,
                    key_tile != 0, pv_done[1]);
                if constexpr (C::AUDIT) {
                    ++g.trace[(int)blockIdx.x * 8 + 5];
                }
            }
        }
        if (role == 2) {
            if (q0_uses_key) {
                wait(pv_done[0], key_tile & 1);
            }
            if (q1_uses_key) {
                wait(pv_done[1], key_tile & 1);
            }
            tensor_after_thread_sync();
        }
        __syncthreads();
    }

    if (role == 0) {
        tensor_after_thread_sync();
        store_raw_two_query_output<C>(g, tt_output0, query_tile[0], h);
        g.lse[h * g.seqlen + query_tile[0] * C::Mb + warpgroup::laneid()] =
            row_count;
    } else if (role == 1 && query_active[1]) {
        tensor_after_thread_sync();
        store_raw_two_query_output<C>(g, tt_output1, query_tile[1], h);
        g.lse[h * g.seqlen + query_tile[1] * C::Mb + warpgroup::laneid()] =
            row_count;
    }
    __syncthreads();
    if constexpr (C::AUDIT) {
        if (threadIdx.x == 0) {
            g.trace[(int)blockIdx.x * 8 + 7] = query_active[1] ? 1 : 0;
        }
    }
    __syncthreads();
}

template <bool AUDIT>
static void launch_raw_fp4_two_query_probe(
    at::Tensor q_fp4,
    at::Tensor k_fp4,
    at::Tensor v_fp4,
    at::Tensor o,
    at::Tensor lse,
    at::Tensor trace
) {
    using C = config_raw_fp4_two_query_probe<AUDIT>;
    using G = globals_raw_fp4_two_query_probe<C>;
    G g{
        reinterpret_cast<const uint8_t *>(q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(k_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(v_fp4.data_ptr()),
        o.data_ptr<float>(),
        lse.data_ptr<float>(),
        trace.data_ptr<int64_t>(),
        (int)q_fp4.size(0),
        (int)q_fp4.size(1),
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_raw_fp4_two_query_probe<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_raw_fp4_two_query_probe<C>, g));
}

void dispatch_raw_fp4_two_query_probe(
    at::Tensor q_fp4,
    at::Tensor k_fp4,
    at::Tensor v_fp4,
    at::Tensor o,
    at::Tensor lse,
    at::Tensor trace,
    bool audit
) {
    CHECK_INPUT(q_fp4);
    CHECK_INPUT(k_fp4);
    CHECK_INPUT(v_fp4);
    CHECK_INPUT(o);
    CHECK_INPUT(lse);
    CHECK_INPUT(trace);
    TORCH_CHECK(q_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                k_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                v_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Q/K/V must be float4_e2m1fn_x2");
    TORCH_CHECK(o.dtype() == at::ScalarType::Float &&
                lse.dtype() == at::ScalarType::Float,
                "O and LSE must be float32");
    TORCH_CHECK(trace.dtype() == at::ScalarType::Long,
                "trace must be int64");
    TORCH_CHECK(q_fp4.dim() == 3 && k_fp4.dim() == 3 && v_fp4.dim() == 3,
                "Q/K/V must be rank three");
    const int heads = (int)q_fp4.size(0);
    const int seqlen = (int)q_fp4.size(1);
    TORCH_CHECK(heads > 0 && seqlen > 0 && seqlen % 128 == 0,
                "heads and seqlen must be positive; seqlen must be divisible by 128");
    TORCH_CHECK(q_fp4.size(2) == 96 &&
                k_fp4.size(0) == heads && k_fp4.size(1) == seqlen &&
                k_fp4.size(2) == 96,
                "Q/K must have shape (heads,seqlen,96 packed bytes)");
    TORCH_CHECK(v_fp4.size(0) == heads && v_fp4.size(1) == 128 &&
                v_fp4.size(2) == seqlen / 2,
                "V must have shape (heads,128,seqlen/2 packed bytes)");
    TORCH_CHECK(o.dim() == 3 && o.size(0) == seqlen &&
                o.size(1) == heads && o.size(2) == 128,
                "O must have shape (seqlen,heads,128)");
    TORCH_CHECK(lse.dim() == 2 && lse.size(0) == heads &&
                lse.size(1) == seqlen,
                "LSE must have shape (heads,seqlen)");
    const int blocks = heads * ((seqlen / 128 + 1) / 2);
    TORCH_CHECK(trace.dim() == 2 && trace.size(0) == blocks &&
                trace.size(1) >= 8,
                "trace must have shape (grid_blocks,>=8)");
    if (audit) {
        launch_raw_fp4_two_query_probe<true>(
            q_fp4, k_fp4, v_fp4, o, lse, trace);
    } else {
        launch_raw_fp4_two_query_probe<false>(
            q_fp4, k_fp4, v_fp4, o, lse, trace);
    }
}

#endif

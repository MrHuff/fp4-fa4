#pragma once

#ifndef TK_FA4_SCALED_K64_TWO_QUERY_CEILING_BUILD
#define TK_FA4_SCALED_K64_TWO_QUERY_CEILING_BUILD 0
#endif

#if TK_FA4_SCALED_K64_TWO_QUERY_CEILING_BUILD

template <bool _TWO_QUERY, bool _AUDIT, bool _DEFER_PV0_WAIT>
struct config_scaled_k64_two_query_ceiling {
    static constexpr bool TWO_QUERY = _TWO_QUERY;
    static constexpr bool AUDIT = _AUDIT;
    static constexpr bool DEFER_PV0_WAIT = _DEFER_PV0_WAIT;
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dqk = 192;
    static constexpr int Dvo = 128;
    static constexpr int QK_SCALE_CHUNKS = Dqk / 64;
    static constexpr int LOAD_STAGES = 2;
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_THREADS = 512;
};

template <typename C>
struct globals_scaled_k64_two_query_ceiling {
    using q_tile = st_fp4e2m1_2<C::Mb, C::Dqk / 2>;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using p_tile = st_fp4e2m1_2<C::Mb, C::Nb / 2>;
    using v_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using q_gl = gl<fp4e2m1_2, -1, -1, -1, -1,
                    tma::descriptor<q_tile, dim::ROW>>;
    using k_gl = gl<fp4e2m1_2, -1, -1, -1, -1,
                    tma::descriptor<k_tile, dim::ROW>>;
    using p_gl = gl<fp4e2m1_2, -1, -1, -1, -1,
                    tma::descriptor<p_tile, dim::ROW>>;
    using v_gl = gl<fp4e2m1_2, -1, -1, -1, -1,
                    tma::descriptor<v_tile, dim::ROW>>;
    using qk_scale_tile = st_fp8e4m3<32, 16, false>;
    using pv_scale_tile = st_fp8e8m0<32, 16, false>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int allocation) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + allocation;
        };
        add(2 * (int)sizeof(q_tile));
        add(4 * (int)sizeof(p_tile));
        add(C::LOAD_STAGES * (int)sizeof(k_tile));
        add(C::LOAD_STAGES * (int)sizeof(v_tile));
        add((int)sizeof(qk_scale_tile));
        add((int)sizeof(pv_scale_tile));
        return align_up_constexpr(bytes, 1024) + 1024;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY =
        compute_dynamic_shared_memory();
    q_gl q;
    k_gl k;
    p_gl p;
    v_gl v;
    float *o;
    float *score_audit;
    int64_t *counts;
    int batch;
    int heads;
    int seqlen;

    __host__ __inline__ dim3 grid() const {
        const int tiles = seqlen / C::Mb;
        const int units = C::TWO_QUERY ? (tiles + 1) / 2 : tiles;
        return dim3(batch * heads * units);
    }
    __host__ __inline__ dim3 block() const {
        return dim3(C::NUM_THREADS);
    }
    __host__ __inline__ int dynamic_shared_memory() const {
        return DYNAMIC_SHARED_MEMORY;
    }
};

template <typename G, typename Tile>
__device__ __forceinline__ void scaled_k64_ceiling_store_tile(
    const G &g,
    Tile &tile,
    float *dst,
    int b,
    int h,
    int query_tile
) {
    const int row = warpgroup::laneid();
    const int query_row = query_tile * 128 + row;
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
            : "r"(tile.addr +
                  ((warpgroup::warpid() * 32) << 16) + col));
        tensor_load_wait();
        const int64_t offset =
            (((int64_t)b * g.seqlen + query_row) * g.heads + h) * 128 + col;
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            reinterpret_cast<float2 *>(dst + offset)[pair] = values[pair];
        }
    }
}

template <typename C, int ACC, typename Output, typename PTile,
          typename VTile, typename PScale, typename VScale>
__device__ __forceinline__ void scaled_k64_ceiling_issue_pv(
    Output &output,
    const PTile &p,
    const VTile &v,
    const PScale &p_scale,
    const VScale &v_scale,
    semaphore &done
) {
    using payload_t = fp4e2m1_2;
    using scale_t = fp8e8m0;
    constexpr uint32_t IDESC0 =
        detail::tcgen05::instruction_descriptor<
            float, payload_t, scale_t, 128, 128, false, 0>();
    constexpr uint32_t IDESC2 =
        detail::tcgen05::instruction_descriptor<
            float, payload_t, scale_t, 128, 128, false, 2>();
    st_descriptor<PTile, transpose::N> p_desc(p);
    st_descriptor<VTile, transpose::N> v_desc(v);
    detail::tcgen05::template st_st<payload_t, scale_t, ACC, 1, 32>(
        output.addr,
        p_desc.chunk_descriptor(0),
        v_desc.chunk_descriptor(0),
        p_scale.addr,
        v_scale.addr,
        IDESC0);
    detail::tcgen05::template st_st<payload_t, scale_t, 1, 1, 32>(
        output.addr,
        p_desc.chunk_descriptor(1),
        v_desc.chunk_descriptor(1),
        p_scale.addr,
        v_scale.addr,
        IDESC2);
    detail::tcgen05::commit<1>(done);
}

template <typename C>
__cluster_dims__(1, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_scaled_k64_two_query_ceiling(
    const __grid_constant__ globals_scaled_k64_two_query_ceiling<C> g
) {
    using G = globals_scaled_k64_two_query_ceiling<C>;
    using score_t = tt<float, 128, 128>;
    using output_t = tt<float, 128, 128>;
    using qk_scale_t = full_tt_fp8e4m3<16>;
    using pv_scale_t = full_tt_fp8e8m0<16>;
    static_assert(C::NUM_THREADS == 512 && C::CLUSTER_SIZE == 1);
    static_assert(C::QK_SCALE_CHUNKS == 3);
    static_assert(G::DYNAMIC_SHARED_MEMORY <= MAX_SHARED_MEMORY - 4096);

    constexpr int SCORE0_BASE = 0;
    constexpr int SCORE1_BASE = 128;
    constexpr int OUTPUT0_BASE = 256;
    constexpr int OUTPUT1_BASE = 384;
    constexpr int INIT_BARRIER_ID = 14;
    static_assert(OUTPUT1_BASE + 128 == MAX_TENSOR_COLS);

    const int tiles = g.seqlen / 128;
    const int units = C::TWO_QUERY ? (tiles + 1) / 2 : tiles;
    const int bh = (int)blockIdx.x / units;
    const int unit = (int)blockIdx.x - bh * units;
    const int b = bh / g.heads;
    const int h = bh - b * g.heads;
    const int q_tile[2] = {
        C::TWO_QUERY ? 2 * unit : unit,
        C::TWO_QUERY ? 2 * unit + 1 : -1};
    const bool q_active[2] = {
        true, C::TWO_QUERY && q_tile[1] < tiles};
    const int key_tiles = q_active[1] ? q_tile[1] + 1 : q_tile[0] + 1;
    const int role = warpgroup::groupid();
    const bool issue_warp = role == 2 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const bool loader_lane =
        role == 3 && warpgroup::warpid() == 3 && warp::elect_leader();

    tensor_allocator<1, 1> tm_alloc{};
    score_t score[2] = {
        tm_alloc.template allocate<score_t>(SCORE0_BASE),
        tm_alloc.template allocate<score_t>(SCORE1_BASE)};
    output_t output[2] = {
        tm_alloc.template allocate<output_t>(OUTPUT0_BASE),
        tm_alloc.template allocate<output_t>(OUTPUT1_BASE)};
    qk_scale_t q_scale[2] = {
        tm_alloc.template allocate<qk_scale_t>(SCORE0_BASE + 0),
        tm_alloc.template allocate<qk_scale_t>(SCORE1_BASE + 0)};
    qk_scale_t k_scale[2] = {
        tm_alloc.template allocate<qk_scale_t>(SCORE0_BASE + 16),
        tm_alloc.template allocate<qk_scale_t>(SCORE1_BASE + 16)};
    pv_scale_t p_scale[2] = {
        tm_alloc.template allocate<pv_scale_t>(SCORE0_BASE + 32),
        tm_alloc.template allocate<pv_scale_t>(SCORE1_BASE + 32)};
    pv_scale_t v_scale[2] = {
        tm_alloc.template allocate<pv_scale_t>(SCORE0_BASE + 48),
        tm_alloc.template allocate<pv_scale_t>(SCORE1_BASE + 48)};

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile (&q_smem)[2] =
        al.allocate<typename G::q_tile, 2>();
    typename G::p_tile (&p_smem)[4] =
        al.allocate<typename G::p_tile, 4>();
    typename G::k_tile (&k_smem)[C::LOAD_STAGES] =
        al.allocate<typename G::k_tile, C::LOAD_STAGES>();
    typename G::v_tile (&v_smem)[C::LOAD_STAGES] =
        al.allocate<typename G::v_tile, C::LOAD_STAGES>();
    typename G::qk_scale_tile &qk_scale_smem =
        al.allocate<typename G::qk_scale_tile>();
    typename G::pv_scale_tile &pv_scale_smem =
        al.allocate<typename G::pv_scale_tile>();

    __shared__ semaphore q_arrived[2];
    __shared__ semaphore p_arrived[4];
    __shared__ semaphore k_arrived[C::LOAD_STAGES];
    __shared__ semaphore v_arrived[C::LOAD_STAGES];
    __shared__ semaphore k_reusable[C::LOAD_STAGES];
    __shared__ semaphore v_reusable[C::LOAD_STAGES];
    __shared__ semaphore scale_smem_ready;
    __shared__ semaphore scale_done;
    __shared__ semaphore qk_done[2];
    __shared__ semaphore pv_done[2];
    __shared__ semaphore output_ready[2];
    __shared__ semaphore score_audit_ready[2];
    __shared__ semaphore score_audit_released[2];

    if (issue_lane) {
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            init_semaphore(q_arrived[q], 0, 1);
            init_semaphore(qk_done[q], 0, 1);
            init_semaphore(pv_done[q], 0, 1);
            init_semaphore(output_ready[q], 0, 1);
            init_semaphore(score_audit_ready[q], 0, 1);
            init_semaphore(score_audit_released[q], 0, 1);
        }
        #pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            init_semaphore(p_arrived[slot], 0, 1);
        }
        #pragma unroll
        for (int slot = 0; slot < C::LOAD_STAGES; ++slot) {
            init_semaphore(k_arrived[slot], 0, 1);
            init_semaphore(v_arrived[slot], 0, 1);
            init_semaphore(k_reusable[slot], 0, 1);
            init_semaphore(v_reusable[slot], 0, 1);
        }
        init_semaphore(scale_smem_ready, 0, 1);
        init_semaphore(scale_done, 0, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    // One warp from each role participates; this is a 128-thread startup
    // rendezvous, not a full-CTA barrier. Each role then fans readiness to its
    // own warp group using the retained overlay primitive.
    if (warpgroup::warpid() == 0) {
        asm volatile("bar.sync %0, 128;" :: "n"(INIT_BARRIER_ID) : "memory");
    }
    warpgroup::sync(role + 1);

    if (role == 3) {
        const int lane = warpgroup::laneid();
        const uint32_t qk_base =
            __cvta_generic_to_shared(&qk_scale_smem.data[0]);
        const uint32_t pv_base =
            __cvta_generic_to_shared(&pv_scale_smem.data[0]);
        asm volatile("st.shared.b32 [%0], %1;" ::
                     "r"(qk_base + lane * 4), "r"(0x38383838u));
        asm volatile("st.shared.b32 [%0], %1;" ::
                     "r"(pv_base + lane * 4), "r"(0x7f7f7f7fu));
        warpgroup::sync(4);
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(4);
        if (!loader_lane) {
            return;
        }
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.k.template prefetch_tma<typename G::k_tile, dim::ROW>();
        g.p.template prefetch_tma<typename G::p_tile, dim::ROW>();
        g.v.template prefetch_tma<typename G::v_tile, dim::ROW>();
        arrive(scale_smem_ready);

        int q_loads = 0;
        int p_loads = 0;
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            if (q_active[q]) {
                tma::expect_bytes(q_arrived[q], sizeof(typename G::q_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    q_smem[q], g.q, {b, h, q_tile[q], 0}, q_arrived[q]);
                ++q_loads;
                #pragma unroll
                for (int kind = 0; kind < 2; ++kind) {
                    const int p_slot = q * 2 + kind;
                    tma::expect_bytes(
                        p_arrived[p_slot], sizeof(typename G::p_tile));
                    tma::load_async<dim::ROW, cache_policy::NORMAL>(
                        p_smem[p_slot], g.p,
                        {q_tile[q] & 1, kind, 0, 0},
                        p_arrived[p_slot]);
                    ++p_loads;
                }
            }
        }

        int k_loads = 0;
        int v_loads = 0;
        #pragma unroll 1
        for (int idx = 0; idx < key_tiles; ++idx) {
            const int slot = idx & 1;
            const int epoch = idx >> 1;
            if (idx >= C::LOAD_STAGES) {
                wait(k_reusable[slot], (epoch - 1) & 1);
                wait(v_reusable[slot], (epoch - 1) & 1);
            }
            tma::expect_bytes(k_arrived[slot], sizeof(typename G::k_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                k_smem[slot], g.k, {b, h, idx, 0}, k_arrived[slot]);
            ++k_loads;
            tma::expect_bytes(v_arrived[slot], sizeof(typename G::v_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                v_smem[slot], g.v, {b, h, 0, idx}, v_arrived[slot]);
            ++v_loads;
        }
        if constexpr (C::AUDIT) {
            int64_t *row = g.counts + (int64_t)blockIdx.x * 12;
            row[4] = k_loads;
            row[5] = v_loads;
            row[6] = q_loads;
            row[7] = p_loads;
        }
        return;
    }

    if (role == 0 || role == 1) {
        const int q = role;
        if (!q_active[q]) {
            return;
        }
        if constexpr (C::AUDIT) {
            wait(score_audit_ready[q], 0);
            tensor_after_thread_sync();
            const int final_score_bank = C::TWO_QUERY ? q : (q_tile[0] & 1);
            scaled_k64_ceiling_store_tile(
                g, score[final_score_bank], g.score_audit,
                b, h, q_tile[q]);
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                arrive(score_audit_released[q]);
            }
        }
        wait(output_ready[q], 0);
        tensor_after_thread_sync();
        scaled_k64_ceiling_store_tile(
            g, output[q], g.o, b, h, q_tile[q]);
        return;
    }

    if (!issue_warp) {
        return;
    }

    wait(scale_smem_ready, 0);
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        if (q_active[q]) {
            wait(q_arrived[q], 0);
            wait(p_arrived[q * 2 + 0], 0);
            wait(p_arrived[q * 2 + 1], 0);
        }
    }

    int scale_phase = 0;
    int qk_commands = 0;
    int pv_commands = 0;
    int qk_ops = 0;
    int pv_ops = 0;
    int scale_loads = 0;

    auto load_scale_bank = [&](int bank, bool load_qk, bool load_pv) {
        if (issue_lane) {
            if (load_qk) {
                fp4pv_load_mxnv_scale_async_ranklocal(
                    q_scale[bank], qk_scale_smem);
                fp4pv_load_mxnv_scale_async_ranklocal(
                    k_scale[bank], qk_scale_smem);
            }
            if (load_pv) {
                fp4pv_load_mxnv_scale_async_ranklocal(
                    p_scale[bank], pv_scale_smem);
                fp4pv_load_mxnv_scale_async_ranklocal(
                    v_scale[bank], pv_scale_smem);
            }
            detail::tcgen05::commit<1>(scale_done);
        }
        wait(scale_done, scale_phase);
        tensor_after_thread_sync();
        scale_phase ^= 1;
        scale_loads += 2 * (int)load_qk + 2 * (int)load_pv;
    };

    auto issue_qk = [&](int q, int key_idx, int score_bank, int scale_bank) {
        const int slot = key_idx & 1;
        using payload_t = fp4e2m1_2;
        using scale_value_t = fp8e4m3;
        constexpr uint32_t IDESC =
            detail::tcgen05::instruction_descriptor<
                float, payload_t, scale_value_t, 128, 128, false, 0>();
        st_descriptor<typename G::q_tile, 0> q_desc(q_smem[q]);
        st_descriptor<typename G::k_tile, 0> k_desc(k_smem[slot]);
        tensor_before_thread_sync();
        if (issue_lane) {
            #pragma unroll
            for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                if (chunk == 0) {
                    detail::tcgen05::template st_st<
                        payload_t, scale_value_t, 0, 1, 16>(
                            score[score_bank].addr,
                            q_desc.chunk_descriptor(chunk),
                            k_desc.chunk_descriptor(chunk),
                            q_scale[scale_bank].addr,
                            k_scale[scale_bank].addr,
                            IDESC);
                } else {
                    detail::tcgen05::template st_st<
                        payload_t, scale_value_t, 1, 1, 16>(
                            score[score_bank].addr,
                            q_desc.chunk_descriptor(chunk),
                            k_desc.chunk_descriptor(chunk),
                            q_scale[scale_bank].addr,
                            k_scale[scale_bank].addr,
                            IDESC);
                }
            }
            detail::tcgen05::commit<1>(qk_done[q]);
        }
        qk_commands += 3;
        ++qk_ops;
    };

    auto issue_pv = [&](int q, int key_idx, int scale_bank) {
        const int p_slot = q * 2 + (key_idx == q_tile[q] ? 1 : 0);
        const int v_slot = key_idx & 1;
        tensor_before_thread_sync();
        if (issue_lane) {
            if (key_idx == 0) {
                scaled_k64_ceiling_issue_pv<C, 0>(
                    output[q], p_smem[p_slot], v_smem[v_slot],
                    p_scale[scale_bank], v_scale[scale_bank], pv_done[q]);
            } else {
                scaled_k64_ceiling_issue_pv<C, 1>(
                    output[q], p_smem[p_slot], v_smem[v_slot],
                    p_scale[scale_bank], v_scale[scale_bank], pv_done[q]);
            }
        }
        pv_commands += 2;
        ++pv_ops;
    };

    auto wait_qk = [&](int q, int key_idx) {
        wait(qk_done[q], key_idx & 1);
        tensor_after_thread_sync();
        if constexpr (C::AUDIT) {
            if (key_idx == q_tile[q]) {
                if (issue_lane) {
                    arrive(score_audit_ready[q]);
                }
                wait(score_audit_released[q], 0);
                tensor_after_thread_sync();
            }
        }
    };

    auto wait_pv = [&](int q, int key_idx) {
        wait(pv_done[q], key_idx & 1);
        tensor_after_thread_sync();
    };

    wait(k_arrived[0], 0);
    load_scale_bank(1, true, false);
    issue_qk(0, 0, 0, 1);

    if constexpr (C::TWO_QUERY) {
        #pragma unroll 1
        for (int idx = 0; idx < key_tiles; ++idx) {
            const int slot = idx & 1;
            const int phase = (idx >> 1) & 1;
            const bool q0_uses = idx <= q_tile[0];
            const bool q1_uses = q_active[1] && idx <= q_tile[1];
            if (q0_uses) {
                wait_qk(0, idx);
            }
            if (!q0_uses) {
                wait(k_arrived[slot], phase);
            }
            wait(v_arrived[slot], phase);

            load_scale_bank(0, q1_uses, q0_uses);
            if (q0_uses) {
                issue_pv(0, idx, 0);
            }
            if (q1_uses) {
                issue_qk(1, idx, 1, 0);
                wait_qk(1, idx);
            }
            if constexpr (!C::DEFER_PV0_WAIT) {
                if (q0_uses) {
                    wait_pv(0, idx);
                }
            }
            if (issue_lane) {
                arrive(k_reusable[slot]);
            }

            const bool q0_next = idx + 1 <= q_tile[0];
            if (q1_uses || q0_next) {
                if (q0_next) {
                    const int next = idx + 1;
                    wait(k_arrived[next & 1], (next >> 1) & 1);
                }
                load_scale_bank(1, q0_next, q1_uses);
                if (q1_uses) {
                    issue_pv(1, idx, 1);
                }
                if (q0_next) {
                    issue_qk(0, idx + 1, 0, 1);
                }
                if (q1_uses) {
                    wait_pv(1, idx);
                }
            }
            if constexpr (C::DEFER_PV0_WAIT) {
                if (q0_uses) {
                    wait_pv(0, idx);
                }
            }
            if (issue_lane) {
                arrive(v_reusable[slot]);
            }
        }
    } else {
        #pragma unroll 1
        for (int idx = 0; idx < key_tiles; ++idx) {
            const int slot = idx & 1;
            const int phase = (idx >> 1) & 1;
            const int score_bank = idx & 1;
            wait_qk(0, idx);
            wait(v_arrived[slot], phase);
            if (issue_lane) {
                arrive(k_reusable[slot]);
            }
            const bool next = idx + 1 < key_tiles;
            if (next) {
                const int next_idx = idx + 1;
                wait(k_arrived[next_idx & 1], (next_idx >> 1) & 1);
            }
            load_scale_bank(score_bank, next, true);
            issue_pv(0, idx, score_bank);
            if (next) {
                issue_qk(0, idx + 1, score_bank ^ 1, score_bank);
            }
            wait_pv(0, idx);
            if (issue_lane) {
                arrive(v_reusable[slot]);
            }
        }
    }

    if (issue_lane) {
        arrive(output_ready[0]);
        if (q_active[1]) {
            arrive(output_ready[1]);
        }
        if constexpr (C::AUDIT) {
            int64_t *row = g.counts + (int64_t)blockIdx.x * 12;
            row[0] = qk_commands;
            row[1] = pv_commands;
            row[2] = qk_ops;
            row[3] = pv_ops;
            row[8] = scale_loads;
            row[9] = (int)q_active[0] + (int)q_active[1];
            row[10] = (int)q_active[0] + (int)q_active[1];
            row[11] = (int)q_active[0] + (int)q_active[1];
        }
    }
}

template <bool TWO_QUERY, bool AUDIT, bool DEFER_PV0_WAIT>
static void launch_scaled_k64_two_query_ceiling(
    at::Tensor q,
    at::Tensor k,
    at::Tensor p,
    at::Tensor v,
    at::Tensor o,
    at::Tensor score_audit,
    at::Tensor counts
) {
    using C = config_scaled_k64_two_query_ceiling<
        TWO_QUERY, AUDIT, DEFER_PV0_WAIT>;
    using G = globals_scaled_k64_two_query_ceiling<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(q),
        kittens::py::tensor_to_gl<typename G::k_gl>(k),
        kittens::py::tensor_to_gl<typename G::p_gl>(p),
        kittens::py::tensor_to_gl<typename G::v_gl>(v),
        o.data_ptr<float>(),
        AUDIT ? score_audit.data_ptr<float>() : nullptr,
        AUDIT ? counts.data_ptr<int64_t>() : nullptr,
        (int)q.size(0),
        (int)q.size(1),
        (int)q.size(2)};
    CUDACHECK(cudaFuncSetAttribute(
        kernel_scaled_k64_two_query_ceiling<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_scaled_k64_two_query_ceiling<C>, g));
}

static void check_scaled_k64_two_query_ceiling_inputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &p,
    const at::Tensor &v,
    const at::Tensor &o
) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(p); CHECK_INPUT(v);
    CHECK_INPUT(o);
    TORCH_CHECK(q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                k.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                p.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                v.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "scaled-K64 ceiling Q/K/P/V must be packed E2M1");
    TORCH_CHECK(o.dtype() == at::ScalarType::Float,
                "scaled-K64 ceiling output must be FP32");
    TORCH_CHECK(q.dim() == 4 && q.size(3) == 96 &&
                k.sizes() == q.sizes(),
                "Q/K must have shape [B,H,S,96 packed]");
    const int batch = q.size(0);
    const int heads = q.size(1);
    const int seqlen = q.size(2);
    TORCH_CHECK(batch > 0 && heads > 0 && seqlen > 0 &&
                seqlen % 128 == 0,
                "B/H must be positive and S must be divisible by 128");
    TORCH_CHECK(p.dim() == 4 && p.size(0) == 2 && p.size(1) == 2 &&
                p.size(2) == 128 && p.size(3) == 64,
                "P must have shape [2 parity,2 full/diagonal,128,64 packed]");
    TORCH_CHECK(v.dim() == 4 && v.size(0) == batch &&
                v.size(1) == heads && v.size(2) == 128 &&
                v.size(3) == seqlen / 2,
                "V must have shape [B,H,128,S/2 packed]");
    TORCH_CHECK(o.dim() == 4 && o.size(0) == batch &&
                o.size(1) == seqlen && o.size(2) == heads &&
                o.size(3) == 128,
                "O must have shape [B,S,H,128]");
}

template <bool TWO_QUERY, bool DEFER_PV0_WAIT>
static void dispatch_scaled_k64_ceiling_authoritative(
    at::Tensor q,
    at::Tensor k,
    at::Tensor p,
    at::Tensor v,
    at::Tensor o
) {
    check_scaled_k64_two_query_ceiling_inputs(q, k, p, v, o);
    launch_scaled_k64_two_query_ceiling<
        TWO_QUERY, false, DEFER_PV0_WAIT>(
        q, k, p, v, o, o, o);
}

template <bool TWO_QUERY, bool DEFER_PV0_WAIT>
static void dispatch_scaled_k64_ceiling_audit(
    at::Tensor q,
    at::Tensor k,
    at::Tensor p,
    at::Tensor v,
    at::Tensor o,
    at::Tensor score_audit,
    at::Tensor counts
) {
    check_scaled_k64_two_query_ceiling_inputs(q, k, p, v, o);
    CHECK_INPUT(score_audit); CHECK_INPUT(counts);
    TORCH_CHECK(score_audit.dtype() == at::ScalarType::Float &&
                score_audit.sizes() == o.sizes(),
                "score_audit must be FP32 with O shape");
    const int tiles = q.size(2) / 128;
    const int units = TWO_QUERY ? (tiles + 1) / 2 : tiles;
    const int blocks = q.size(0) * q.size(1) * units;
    TORCH_CHECK(counts.dtype() == at::ScalarType::Long &&
                counts.dim() == 2 && counts.size(0) == blocks &&
                counts.size(1) >= 12,
                "counts must have shape [grid_blocks,>=12] int64");
    launch_scaled_k64_two_query_ceiling<
        TWO_QUERY, true, DEFER_PV0_WAIT>(
        q, k, p, v, o, score_audit, counts);
}

void dispatch_scaled_k64_two_query_ceiling(
    at::Tensor q, at::Tensor k, at::Tensor p, at::Tensor v, at::Tensor o) {
    dispatch_scaled_k64_ceiling_authoritative<true, false>(q, k, p, v, o);
}

void dispatch_scaled_k64_one_query_ceiling(
    at::Tensor q, at::Tensor k, at::Tensor p, at::Tensor v, at::Tensor o) {
    dispatch_scaled_k64_ceiling_authoritative<false, false>(q, k, p, v, o);
}

void dispatch_scaled_k64_two_query_ceiling_audit(
    at::Tensor q, at::Tensor k, at::Tensor p, at::Tensor v, at::Tensor o,
    at::Tensor score_audit, at::Tensor counts) {
    dispatch_scaled_k64_ceiling_audit<true, false>(
        q, k, p, v, o, score_audit, counts);
}

void dispatch_scaled_k64_one_query_ceiling_audit(
    at::Tensor q, at::Tensor k, at::Tensor p, at::Tensor v, at::Tensor o,
    at::Tensor score_audit, at::Tensor counts) {
    dispatch_scaled_k64_ceiling_audit<false, false>(
        q, k, p, v, o, score_audit, counts);
}

void dispatch_scaled_k64_two_query_ceiling_deferpv0(
    at::Tensor q, at::Tensor k, at::Tensor p, at::Tensor v, at::Tensor o) {
    dispatch_scaled_k64_ceiling_authoritative<true, true>(q, k, p, v, o);
}

void dispatch_scaled_k64_two_query_ceiling_deferpv0_audit(
    at::Tensor q, at::Tensor k, at::Tensor p, at::Tensor v, at::Tensor o,
    at::Tensor score_audit, at::Tensor counts) {
    dispatch_scaled_k64_ceiling_audit<true, true>(
        q, k, p, v, o, score_audit, counts);
}

#endif

#pragma once

#ifndef TK_FA4_QUARTER_LIFETIME_TWO_QUERY_BUILD
#define TK_FA4_QUARTER_LIFETIME_TWO_QUERY_BUILD 0
#endif

#if TK_FA4_QUARTER_LIFETIME_TWO_QUERY_BUILD

// Isolated full-grid paired-query route.  Arithmetic traits are inherited from
// the corrected real-softmax parent; only score/scale ownership is quartered.
struct config_fp4pv_quarter_lifetime_two_query_overlay
    : public config_fp4pv_stage2_ex2_alu_dual_query_two_score_overlay {
    static constexpr bool QUARTER_LIFETIME_OVERLAY = true;
    static constexpr bool QUARTER_LIFETIME_PRUNED = false;
    static constexpr bool QUARTER_LIFETIME_FUSE_CORRECTION_R0 = false;
    static constexpr int ONLINE_STAGE2_EX2_EMU_PERIOD = 0;
    static constexpr uint32_t ONLINE_STAGE2_EX2_EMU_MASK = 0;
};

// v31a keeps the exact v30i ownership graph and arithmetic, but removes only
// synchronization that is dominated by the later P-half/P-full publication.
// It also retains three scalar quarter maxima across the quarter-0 pack so the
// three mandated TMEM rereads do not repeat the max reduction.
struct config_fp4pv_quarter_lifetime_two_query_overlay_pruned
    : public config_fp4pv_quarter_lifetime_two_query_overlay {
    static constexpr bool QUARTER_LIFETIME_PRUNED = true;
};

// v31b additionally publishes correction at the already-required quarter-0
// warpgroup rendezvous.  Keeping this as a distinct specialization makes the
// latency cost of that fusion independently attributable.
struct config_fp4pv_quarter_lifetime_two_query_overlay_pruned_fused
    : public config_fp4pv_quarter_lifetime_two_query_overlay_pruned {
    static constexpr bool QUARTER_LIFETIME_FUSE_CORRECTION_R0 = true;
};

template <typename C>
struct fp4pv_quarter_lifetime_two_query_overlay_smem_layout {
    using G = globals_fp4pv_mxfp4_dv<C, true>;

    static constexpr int bytes() {
        int total = 0;
        auto add = [&](int allocation) constexpr {
            total = align_up_constexpr(total, 1024) + allocation;
        };
        add((int)sizeof(typename G::q_tile) * 2);
        add((int)sizeof(typename G::q_sc_smem_storage));
        add((int)sizeof(typename G::k_tile) * C::LOAD_STAGES);
        add((int)sizeof(typename G::k_sc_smem_storage));
        add((int)sizeof(typename G::k_sc_guard_slot));
        add((int)sizeof(typename G::v_fp4_tile) * C::LOAD_STAGES);
        add((int)sizeof(typename G::v_sc_tile) * C::LOAD_STAGES);
        add((int)sizeof(typename G::p_fp4_tile) * 2);
        // One immutable scale snapshot per query/K64 half prevents an async
        // low-half copy from racing owner writes to the upper-half bytes.
        add((int)sizeof(typename G::p_sc_tile) * 4);
        add((int)sizeof(typename G::o_tile));
        // Static shared objects can place the extern segment at a non-1024B
        // address.  Reserve one page for the allocator's first alignment.
        return total + 1024;
    }
};

template <typename C, typename OutputTT>
__device__ __forceinline__ void fp4pv_quarter_lifetime_two_query_overlay_rescale_output(
    OutputTT &tt_output,
    float correction
) {
    const float2 correction2 = {correction, correction};
    #pragma unroll
    for (int col = 0; col < C::Dvo; col += 16) {
        float2 out[8];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(out[0].x), "=f"(out[0].y), "=f"(out[1].x), "=f"(out[1].y),
              "=f"(out[2].x), "=f"(out[2].y), "=f"(out[3].x), "=f"(out[3].y),
              "=f"(out[4].x), "=f"(out[4].y), "=f"(out[5].x), "=f"(out[5].y),
              "=f"(out[6].x), "=f"(out[6].y), "=f"(out[7].x), "=f"(out[7].y)
            : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
        tensor_load_wait();
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = __fmul2_rn(out[i], correction2);
        }
        asm volatile(
            "{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
            :: "f"(out[0].x), "f"(out[0].y), "f"(out[1].x), "f"(out[1].y),
               "f"(out[2].x), "f"(out[2].y), "f"(out[3].x), "f"(out[3].y),
               "f"(out[4].x), "f"(out[4].y), "f"(out[5].x), "f"(out[5].y),
               "f"(out[6].x), "f"(out[6].y), "f"(out[7].x), "f"(out[7].y),
               "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
    }
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C>
__cluster_dims__(1, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_fp4pv_quarter_lifetime_two_query_overlay(
    const __grid_constant__ globals_fp4pv_mxfp4_dv<C, true> g
) {
    static_assert(C::CLUSTER_SIZE == 1 && C::NUM_SOFTMAXXERS == 2 &&
                  C::TOTAL_WGS == 4 && C::Mb == 128 && C::Nb == 128 &&
                  C::Dqk == 192 && C::Dvo == 128,
                  "dual-query probe is specialized to the production head shape");
    static_assert(C::QUARTER_LIFETIME_OVERLAY,
                  "the isolated route must instantiate quarter ownership");
    static_assert(!C::QUARTER_LIFETIME_FUSE_CORRECTION_R0 ||
                      C::QUARTER_LIFETIME_PRUNED,
                  "correction/R0 fusion requires the pruned lifecycle");
    static_assert(C::ONLINE_STAGE2_EX2_EMU_PERIOD == 0,
                  "the candidate must retain the parent's native EX2 path");
    using G = globals_fp4pv_mxfp4_dv<C, true>;
    using Layout = fp4pv_quarter_lifetime_two_query_overlay_smem_layout<C>;
    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_output = tt<float, C::Mb, C::Dvo>;
    using d_tt_qk_scale = full_tt_fp8e4m3<16>;
    using d_tt_pv_scale = full_tt_fp8e8m0<16>;
    constexpr int SCORE0_BASE = 0;
    constexpr int SCORE1_BASE = C::Nb;
    constexpr int OUTPUT0_BASE = 2 * C::Nb;
    constexpr int OUTPUT1_BASE = OUTPUT0_BASE + C::Dvo;
    constexpr int SCALE_BANK_WIDTH = 4 * 16;
    static_assert(SCORE0_BASE == 0 && SCORE1_BASE == 128 &&
                  OUTPUT0_BASE == 256 && OUTPUT1_BASE == 384,
                  "quarter-lifetime TMEM map must be exactly 0/128/256/384");
    static_assert(d_tt_qk_scale::cols == 16 && d_tt_pv_scale::cols == 16,
                  "every JIT scale operand must occupy exactly 16 columns");
    static_assert(SCORE0_BASE + 48 + 16 == SCORE0_BASE + 64 &&
                  SCORE0_BASE + 80 + 16 == SCORE0_BASE + 96 &&
                  SCORE1_BASE + 48 + 16 == SCORE1_BASE + 64 &&
                  SCORE1_BASE + 80 + 16 == SCORE1_BASE + 96,
                  "P/V scale pages must remain inside disjoint score regions");
    static_assert(SCORE0_BASE + 96 <= SCORE1_BASE &&
                  SCORE1_BASE + 96 <= OUTPUT0_BASE,
                  "JIT scale pages must remain inside their score banks");
    static_assert(OUTPUT1_BASE + C::Dvo == MAX_TENSOR_COLS,
                  "two-score overlay must consume exactly 512 TMEM columns");
    static_assert(SCALE_BANK_WIDTH <= C::Nb,
                  "scale operands must fit in one inactive score bank");
    static_assert(Layout::bytes() <= MAX_SHARED_MEMORY - 4096,
                  "dual-query overlay exceeds the shared-memory budget");

    const int tiles_m = g.q.rows() / C::Mb;
    const int pairs_per_head = (tiles_m + 1) / 2;
    const int bh = blockIdx.x / pairs_per_head;
    const int pair_linear = blockIdx.x - bh * pairs_per_head;
    const int pair = pairs_per_head - 1 - pair_linear;
    const int b = bh / g.q.depth();
    const int h = bh - b * g.q.depth();
    // Keep query coordinates scalar.  The predecessor's two-element local
    // arrays were materialized in the CUDA stack and generated LDL/STL even
    // without compiler-reported spills.
    const int q0_tile = pair * 2;
    const bool q1_active = q0_tile + 1 < tiles_m;
    const int q1_tile = q1_active ? q0_tile + 1 : q0_tile;
    const int iters = q1_tile + 1;
    const float scale_log2 = qk_scale_log2(g, b, h);
    const bool zero_scale = scale_log2 == 0.0f;

    tensor_allocator<1, 1> tm_alloc{};
    // Keep score descriptors scalar for the same reason.  Dynamic selection
    // is a register predicate; there is no descriptor array to reload locally.
    d_tt_scores tt_score0 =
        tm_alloc.template allocate<d_tt_scores>(SCORE0_BASE);
    d_tt_scores tt_score1 =
        tm_alloc.template allocate<d_tt_scores>(SCORE1_BASE);
    d_tt_output tt_output[2] = {
        tm_alloc.template allocate<d_tt_output>(OUTPUT0_BASE),
        tm_alloc.template allocate<d_tt_output>(OUTPUT1_BASE)};
    // A query writes score[q] and sources all four scale operands from the
    // other query's inactive score bank.  Q/K occupy offsets 0/16 while the
    // previous query's P/V occupy 32/48, so QK and PV can coexist.
    constexpr int SCALE_BANK_BASE[2] = {SCORE1_BASE, SCORE0_BASE};
    d_tt_qk_scale q_sc_tm[2] = {
        tm_alloc.template allocate<d_tt_qk_scale>(SCALE_BANK_BASE[0] + 0 * 16),
        tm_alloc.template allocate<d_tt_qk_scale>(SCALE_BANK_BASE[1] + 0 * 16)};
    d_tt_qk_scale k_sc_tm[2] = {
        tm_alloc.template allocate<d_tt_qk_scale>(SCALE_BANK_BASE[0] + 1 * 16),
        tm_alloc.template allocate<d_tt_qk_scale>(SCALE_BANK_BASE[1] + 1 * 16)};
    // Each K64 command owns a distinct 32-column P/V page.  Page 0 aliases
    // score quarter1 and page1 aliases score quarter2 of the same query.
    d_tt_pv_scale p_sc_tm[2][2] = {
        {tm_alloc.template allocate<d_tt_pv_scale>(SCORE0_BASE + 32),
         tm_alloc.template allocate<d_tt_pv_scale>(SCORE0_BASE + 64)},
        {tm_alloc.template allocate<d_tt_pv_scale>(SCORE1_BASE + 32),
         tm_alloc.template allocate<d_tt_pv_scale>(SCORE1_BASE + 64)}};
    d_tt_pv_scale v_sc_tm[2][2] = {
        {tm_alloc.template allocate<d_tt_pv_scale>(SCORE0_BASE + 48),
         tm_alloc.template allocate<d_tt_pv_scale>(SCORE0_BASE + 80)},
        {tm_alloc.template allocate<d_tt_pv_scale>(SCORE1_BASE + 48),
         tm_alloc.template allocate<d_tt_pv_scale>(SCORE1_BASE + 80)}};

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile (&q_smem)[2] = al.allocate<typename G::q_tile, 2>();
    typename G::q_sc_smem_storage &q_sc_smem =
        al.allocate<typename G::q_sc_smem_storage>();
    typename G::k_tile (&k_smem)[C::LOAD_STAGES] =
        al.allocate<typename G::k_tile, C::LOAD_STAGES>();
    typename G::k_sc_smem_storage &k_sc_smem =
        al.allocate<typename G::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] = al.allocate<typename G::k_sc_guard_slot, 1>();
    (void)k_sc_guard;
    typename G::v_fp4_tile (&v_smem)[C::LOAD_STAGES] =
        al.allocate<typename G::v_fp4_tile, C::LOAD_STAGES>();
    typename G::v_sc_tile (&v_sc_smem)[C::LOAD_STAGES] =
        al.allocate<typename G::v_sc_tile, C::LOAD_STAGES>();
    typename G::p_fp4_tile (&p_smem)[2] =
        al.allocate<typename G::p_fp4_tile, 2>();
    typename G::p_sc_tile (&p_sc_smem)[4] =
        al.allocate<typename G::p_sc_tile, 4>();
    typename G::o_tile &o_smem = al.allocate<typename G::o_tile>();

    __shared__ sv_fl<C::Mb> correction[2];
    __shared__ sv_fl<C::Mb> row_max_shared[2];
    __shared__ sv_fl<C::Mb> row_sum_shared[2];
    __shared__ semaphore q_arrived[2], q_sc_arrived[2];
    __shared__ semaphore k_arrived[C::LOAD_STAGES];
    __shared__ semaphore k_sc_arrived[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore v_arrived[C::LOAD_STAGES];
    __shared__ semaphore v_sc_arrived[C::LOAD_STAGES];
    __shared__ semaphore k_reusable[C::LOAD_STAGES];
    __shared__ semaphore k_sc_reusable[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore v_reusable[C::LOAD_STAGES];
    __shared__ semaphore scores_arrived[2];
    __shared__ semaphore score_region_dead[2][4];
    __shared__ semaphore p_half_ready[2];
    __shared__ semaphore p_full_ready[2];
    __shared__ semaphore correction_ready[2];
    __shared__ semaphore p_reusable[2];
    __shared__ semaphore output_ready[2];
    __shared__ semaphore pv_chunk_done[2][2];
    __shared__ semaphore stats_ready[2];
    __shared__ semaphore qk_scale_done[2][C::QK_SCALE_CHUNKS];
    __shared__ semaphore pv_scale_done[2][2];

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<typename G::q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<typename G::k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<typename G::k_sc_tile, dim::ROW>();
        g.v_fp4_tma.template prefetch_tma<typename G::v_fp4_tile, dim::ROW>();
        g.o.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            init_semaphore(q_arrived[q], 0, 1);
            init_semaphore(q_sc_arrived[q], 0, 1);
            init_semaphore(scores_arrived[q], 0, 1);
            #pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                init_semaphore(score_region_dead[q][quarter], 0, 1);
            }
            init_semaphore(p_half_ready[q], 0, 1);
            init_semaphore(p_full_ready[q], 0, 1);
            init_semaphore(correction_ready[q], 0, 1);
            init_semaphore(p_reusable[q], 0, 1);
            init_semaphore(output_ready[q], 0, 1);
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                init_semaphore(pv_chunk_done[q][half], 0, 1);
                init_semaphore(pv_scale_done[q][half], 0, 1);
            }
            init_semaphore(stats_ready[q], 0, 1);
            #pragma unroll
            for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                init_semaphore(qk_scale_done[q][chunk], 0, 1);
            }
        }
        #pragma unroll
        for (int slot = 0; slot < C::LOAD_STAGES; ++slot) {
            init_semaphore(k_arrived[slot], 0, 1);
            init_semaphore(k_reusable[slot], 0, 1);
            init_semaphore(v_arrived[slot], 0, 1);
            init_semaphore(v_sc_arrived[slot], 0, 1);
            init_semaphore(v_reusable[slot], 0, 1);
        }
        #pragma unroll
        for (int slot = 0; slot < C::K_SCALE_SMEM_SLOTS; ++slot) {
            init_semaphore(k_sc_arrived[slot], 0, 1);
            init_semaphore(k_sc_reusable[slot], 0, 1);
        }
    }

    __syncthreads();
#if defined(TK_FA4_DUAL_QUERY_STOP_AFTER_INIT) && TK_FA4_DUAL_QUERY_STOP_AFTER_INIT
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        fp4pv_pairpsc_desc_diag[0] = SCORE0_BASE;
        fp4pv_pairpsc_desc_diag[1] = SCORE1_BASE;
        fp4pv_pairpsc_desc_diag[2] = OUTPUT0_BASE;
        fp4pv_pairpsc_desc_diag[3] = OUTPUT1_BASE;
    }
    return;
#endif

    if (warpgroup::groupid() == 3) {
        warpgroup::decrease_registers<96>();
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                if (q == 1 && !q1_active) {
                    continue;
                }
                const int query_tile = q == 0 ? q0_tile : q1_tile;
#if defined(TK_FA4_DUAL_QUERY_Q_ONE_LOAD) && TK_FA4_DUAL_QUERY_Q_ONE_LOAD
                if (q != 0) {
                    continue;
                }
#endif
                fp4pv_expect_bytes<C>(
                    q_arrived[q], sizeof(typename G::q_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    q_smem[q], g.q, {b, h, query_tile, 0}, q_arrived[q]);
#if !defined(TK_FA4_DUAL_QUERY_SKIP_Q_SCALE_LOAD) || !TK_FA4_DUAL_QUERY_SKIP_Q_SCALE_LOAD
                fp4pv_expect_bytes<C>(
                    q_sc_arrived[q], sizeof(typename G::q_sc_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    slot_tile_at(q_sc_smem, q), g.q_sc,
                    {b, query_tile, h, 0},
                    q_sc_arrived[q]);
#endif
            }

            #pragma unroll 1
            for (int idx = 0; idx < iters; ++idx) {
                const int slot = idx & 1;
                const int epoch = idx >> 1;
                if (idx >= 2) {
                    wait(k_reusable[slot], (epoch - 1) & 1);
                    wait(k_sc_reusable[slot], (epoch - 1) & 1);
                    wait(v_reusable[slot], (epoch - 1) & 1);
                }
#if !defined(TK_FA4_DUAL_QUERY_SKIP_K_LOAD) || !TK_FA4_DUAL_QUERY_SKIP_K_LOAD
                fp4pv_expect_bytes<C>(
                    k_arrived[slot], sizeof(typename G::k_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    k_smem[slot], g.k, {b, h, idx, 0}, k_arrived[slot]);
                fp4pv_expect_bytes<C>(
                    k_sc_arrived[slot], sizeof(typename G::k_sc_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    slot_tile_at(k_sc_smem, slot), g.k_sc,
                    {b, 2 * idx, h, 0}, k_sc_arrived[slot]);
#endif

#if !defined(TK_FA4_DUAL_QUERY_SKIP_V_LOAD) || !TK_FA4_DUAL_QUERY_SKIP_V_LOAD
                fp4pv_expect_bytes<C>(v_arrived[slot], sizeof(typename G::v_fp4_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    v_smem[slot], g.v_fp4_tma, {b, h, 0, idx}, v_arrived[slot]);
                const int v_sc_offset =
                    (((((b * g.q.depth() + h) * 2) *
                       (g.k.rows() / 128)) + idx) * 32 * 16);
#if !defined(TK_FA4_DUAL_QUERY_SKIP_V_SCALE_LOAD) || !TK_FA4_DUAL_QUERY_SKIP_V_SCALE_LOAD
                fp4pv_expect_bytes<C>(v_sc_arrived[slot], sizeof(typename G::v_sc_tile));
                tma::load_async(
                    reinterpret_cast<void *>(&v_sc_smem[slot].data[0]),
                    const_cast<uint8_t *>(&g.v_sc_prepared[v_sc_offset]),
                    sizeof(typename G::v_sc_tile), v_sc_arrived[slot]);
#endif
#endif
            }
        } else if (warpgroup::warpid() == 0) {
            const bool local_leader = warp::elect_leader();
            auto issue_qk = [&](int idx, int q) {
                const int slot = idx & 1;
                const int phase = (idx >> 1) & 1;
                wait(k_arrived[slot], phase);
                wait(k_sc_arrived[slot], phase);
                using T_AB = typename G::q_tile::T;
                using T_SAB = typename d_tt_qk_scale::T;
                using T_D = typename d_tt_scores::T;
                constexpr uint32_t IDESC =
                    detail::tcgen05::instruction_descriptor<
                        T_D, T_AB, T_SAB, C::Mb, C::Nb, false, 0>();
                st_descriptor<typename G::q_tile, 0> q_desc(q_smem[q]);
                st_descriptor<typename G::k_tile, 0> k_desc(k_smem[slot]);
                #pragma unroll
                for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                    if (local_leader) {
                        fp4pv_load_q_scale_chunk_ranklocal(
                            q_sc_tm[q], slot_tile_at(q_sc_smem, q), chunk);
                        fp4pv_load_k_scale_chunk_ranklocal(
                            k_sc_tm[q], slot_tile_at(k_sc_smem, slot), chunk);
                        detail::tcgen05::commit<1>(qk_scale_done[q][chunk]);
                    }
                    wait(qk_scale_done[q][chunk], idx & 1);
                    tensor_after_thread_sync();
                    tensor_before_thread_sync();
                    if (local_leader) {
                        if (chunk == 0) {
                            detail::tcgen05::template st_st<
                                T_AB, T_SAB, 0, 1, 16>(
                                q == 0 ? tt_score0.addr : tt_score1.addr,
                                q_desc.chunk_descriptor(chunk),
                                k_desc.chunk_descriptor(chunk),
                                q_sc_tm[q].addr, k_sc_tm[q].addr, IDESC);
                        } else {
                            detail::tcgen05::template st_st<
                                T_AB, T_SAB, 1, 1, 16>(
                                q == 0 ? tt_score0.addr : tt_score1.addr,
                                q_desc.chunk_descriptor(chunk),
                                k_desc.chunk_descriptor(chunk),
                                q_sc_tm[q].addr, k_sc_tm[q].addr, IDESC);
                        }
                    }
                }
                if (local_leader) {
                    detail::tcgen05::commit<1>(scores_arrived[q]);
                }
            };

            auto issue_pv_half = [&](int idx, int q, int half) {
                const int v_slot = idx & 1;
                const int v_phase = (idx >> 1) & 1;
                const int phase = idx & 1;
                // The two K64 commands use the same immutable V/V-scale slot.
                // One acquire before the first half dominates the second.
                if constexpr (!C::QUARTER_LIFETIME_PRUNED) {
                    wait(v_arrived[v_slot], v_phase);
                    wait(v_sc_arrived[v_slot], v_phase);
                } else if (half == 0) {
                    wait(v_arrived[v_slot], v_phase);
                    wait(v_sc_arrived[v_slot], v_phase);
                }
                if (half == 0) {
                    if constexpr (!C::QUARTER_LIFETIME_PRUNED) {
                        wait(score_region_dead[q][1], phase);
                    }
                    wait(p_half_ready[q], phase);
                    wait(output_ready[q], phase);
                } else {
                    if constexpr (!C::QUARTER_LIFETIME_PRUNED) {
                        wait(score_region_dead[q][2], phase);
                    }
                    wait(p_full_ready[q], phase);
                }
                tensor_after_thread_sync();
                __threadfence_block();
                fp4pv_publish_shared_backing_proxy_only();
                if (local_leader) {
                    fp4pv_load_mxnv_scale_async_ranklocal(
                        p_sc_tm[q][half], p_sc_smem[q * 2 + half]);
                    fp4pv_load_mxnv_scale_async_ranklocal(
                        v_sc_tm[q][half], v_sc_smem[v_slot]);
                    detail::tcgen05::commit<1>(pv_scale_done[q][half]);
                }
                wait(pv_scale_done[q][half], phase);
                tensor_after_thread_sync();
                tensor_before_thread_sync();
                if (local_leader) {
                    if (half == 0 && idx == 0) {
                        fp4pv_p_stage_ABt_k64_chunk<0, 0, 0, true, C>(
                            tt_output[q], p_smem[q], v_smem[v_slot],
                            p_sc_tm[q][0], v_sc_tm[q][0],
                            pv_chunk_done[q][0]);
                    } else if (half == 0) {
                        fp4pv_p_stage_ABt_k64_chunk<0, 1, 0, true, C>(
                            tt_output[q], p_smem[q], v_smem[v_slot],
                            p_sc_tm[q][0], v_sc_tm[q][0],
                            pv_chunk_done[q][0]);
                    } else {
                        fp4pv_p_stage_ABt_k64_chunk<1, 1, 2, true, C>(
                            tt_output[q], p_smem[q], v_smem[v_slot],
                            p_sc_tm[q][1], v_sc_tm[q][1],
                            pv_chunk_done[q][1]);
                    }
                }
            };

            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                if (q == 1 && !q1_active) {
                    continue;
                }
                if (local_leader) {
                    wait(q_arrived[q], 0);
                    wait(q_sc_arrived[q], 0);
                }
            }
            issue_qk(0, 0);
            #pragma unroll 1
            for (int idx = 0; idx < iters; ++idx) {
                const int phase = idx & 1;
                const bool q0_uses = idx <= q0_tile;
                const bool q1_uses = q1_active && idx <= q1_tile;
                const bool q0_next_uses = idx + 1 <= q0_tile;

                if (q1_uses) {
                    // When q0 has a score in this epoch, its packed quarter0
                    // releases score0/R0 for q1 Q/K.  The adjacent pair's last
                    // q1-only key tile reuses the already-released prior page.
                    if (q0_uses) {
                        wait(score_region_dead[0][0], phase);
                        tensor_after_thread_sync();
                    }
                    issue_qk(idx, 1);
                }

                // Each legal K64 command owns a distinct score0 scale page and
                // publishes its own TCGEN completion before that page is reused.
                if (q0_uses) {
                    issue_pv_half(idx, 0, 0);
                    issue_pv_half(idx, 0, 1);
                    wait(pv_chunk_done[0][0], phase);
                    wait(pv_chunk_done[0][1], phase);
                    tensor_after_thread_sync();
                }

                if (q0_next_uses) {
                    if (q1_uses) {
                        // q1 quarter0 frees score1/R0 for QK0(next) scales and
                        // also proves QK1 no longer reads score0/R0.
                        wait(score_region_dead[1][0], phase);
                    } else {
                        // Odd-tail q0-only pair: QK0 completion is carried by
                        // q0's own quarter0 acquire/release chain.
                        wait(score_region_dead[0][0], phase);
                    }
                    tensor_after_thread_sync();
                    issue_qk(idx + 1, 0);
                }

                if (q1_uses) {
                    issue_pv_half(idx, 1, 0);
                    issue_pv_half(idx, 1, 1);
                    wait(pv_chunk_done[1][0], phase);
                    wait(pv_chunk_done[1][1], phase);
                    tensor_after_thread_sync();
                }
                if (local_leader) {
                    arrive(k_reusable[idx & 1]);
                    arrive(k_sc_reusable[idx & 1]);
                }
            }
        }
    } else if (warpgroup::groupid() == 2) {
        warpgroup::decrease_registers<48>();
        const bool issue_lane =
            warpgroup::warpid() == 0 && warp::elect_leader();

        #pragma unroll 1
        for (int idx = 0; idx < iters; ++idx) {
            const int phase = idx & 1;
            const int v_slot = idx & 1;
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int query_tile = q == 0 ? q0_tile : q1_tile;
                const bool query_active = q == 0 || q1_active;
                const bool q_uses = query_active && idx <= query_tile;
                if (!q_uses) {
                    continue;
                }
                if (idx == 0) {
                    zero_output_scratch<C>(tt_output[q]);
                } else {
                    wait(correction_ready[q], phase);
                    fp4pv_quarter_lifetime_two_query_overlay_rescale_output<C>(
                        tt_output[q],
                        stats_load(correction[q], warpgroup::laneid()));
                }
                warpgroup::arrive(output_ready[q]);
            }
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int query_tile = q == 0 ? q0_tile : q1_tile;
                const bool query_active = q == 0 || q1_active;
                const bool q_uses = query_active && idx <= query_tile;
                if (!q_uses) {
                    continue;
                }
                wait(pv_chunk_done[q][0], phase);
                wait(pv_chunk_done[q][1], phase);
                tensor_after_thread_sync();
                if (issue_lane) {
                    arrive(p_reusable[q]);
                }
            }
            if (issue_lane) {
                arrive(v_reusable[v_slot]);
            }
        }

#if defined(TK_FA4_DUAL_QUERY_STOP_BEFORE_OUTPUT) && TK_FA4_DUAL_QUERY_STOP_BEFORE_OUTPUT
        return;
#endif
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const bool query_active = q == 0 || q1_active;
            const int query_tile = q == 0 ? q0_tile : q1_tile;
            if (!query_active) {
                continue;
            }
            if (q != 0) {
                warpgroup::tma::store_async_read_wait<0>();
                warpgroup::sync(warpgroup::groupid() + 1);
            }
            wait(stats_ready[q], 0);
            const int row = warpgroup::laneid();
            const float row_sum = stats_load(row_sum_shared[q], row);
            const bool invalid = row_sum == 0.0f || row_sum != row_sum;
            float inv_norm;
            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                         : "=f"(inv_norm) : "f"(invalid ? 1.0f : row_sum));
            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
            const float output_scale =
                (MXFP4_ALPHA / FP4PV_MXFP4_ONLINE_P_SCALE) * inv_norm;
            if (query_active) {
                constexpr int SUBTILE_COLS =
                    G::o_tile::swizzle_bytes / sizeof(bf16);
                constexpr uint32_t SWIZZLE_MASK =
                    G::o_tile::swizzle_bytes * 8 - 1;
                const uint32_t base_addr =
                    __cvta_generic_to_shared(&o_smem.data[0]);
                const uint32_t row_offset =
                    row * SUBTILE_COLS * sizeof(bf16);
                const float2 output_scale2 = {
                    invalid ? 0.0f : output_scale,
                    invalid ? 0.0f : output_scale};
                #pragma unroll
                for (int col = 0; col < C::Dvo; col += 16) {
                    float2 out[8];
                    asm volatile(
                        "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                        : "=f"(out[0].x), "=f"(out[0].y),
                          "=f"(out[1].x), "=f"(out[1].y),
                          "=f"(out[2].x), "=f"(out[2].y),
                          "=f"(out[3].x), "=f"(out[3].y),
                          "=f"(out[4].x), "=f"(out[4].y),
                          "=f"(out[5].x), "=f"(out[5].y),
                          "=f"(out[6].x), "=f"(out[6].y),
                          "=f"(out[7].x), "=f"(out[7].y)
                        : "r"(tt_output[q].addr +
                              ((warpgroup::warpid() * 32) << 16) + col));
                    tensor_load_wait();
                    const uint32_t row_base =
                        base_addr +
                        (col / SUBTILE_COLS) *
                            (C::Mb * SUBTILE_COLS * sizeof(bf16)) +
                        row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
                    #pragma unroll
                    for (int i = 0; i < 8; i += 4) {
                        const bf16_2 packed0 = __float22bfloat162_rn(
                            __fmul2_rn(out[i], output_scale2));
                        const bf16_2 packed1 = __float22bfloat162_rn(
                            __fmul2_rn(out[i + 1], output_scale2));
                        const bf16_2 packed2 = __float22bfloat162_rn(
                            __fmul2_rn(out[i + 2], output_scale2));
                        const bf16_2 packed3 = __float22bfloat162_rn(
                            __fmul2_rn(out[i + 3], output_scale2));
                        const uint32_t address = row_base + i * sizeof(bf16_2);
                        const uint32_t swizzled =
                            address ^ (((address & SWIZZLE_MASK) >> 7) << 4);
                        asm volatile(
                            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                            :: "r"(swizzled),
                               "r"(*reinterpret_cast<const uint32_t *>(&packed0)),
                               "r"(*reinterpret_cast<const uint32_t *>(&packed1)),
                               "r"(*reinterpret_cast<const uint32_t *>(&packed2)),
                               "r"(*reinterpret_cast<const uint32_t *>(&packed3)));
                    }
                }
                warpgroup::sync(warpgroup::groupid() + 1);
                warpgroup::tma::store_async<
                    dim::DEPTH, cache_policy::EVICT_FIRST>(
                    g.o, o_smem, {b, query_tile, h, 0});
                if (g.disable_lse_store == 0) {
                    // row_max_shared is immutable after stats_ready.  Loading
                    // it only at its scalar LSE use avoids carrying it through
                    // all eight output TMEM loads (the sole v30f spill) while
                    // retaining the parent's post-output-store LSE order.
                    const float row_max =
                        stats_load(row_max_shared[q], row);
                    constexpr float LN2 = 0.693147180559945f;
                    float lse = base_types::constants<float>::neg_infty();
                    if (!invalid) {
                        float log2_sum;
                        asm("lg2.approx.ftz.f32 %0, %1;"
                            : "=f"(log2_sum) : "f"(row_sum));
                        lse =
                            (row_max * scale_log2 + log2_sum) * LN2;
                    }
                    const int s = query_tile * C::Mb + row;
                    g.lse[{b, h, 0, s}] = lse;
                }
            }
        }
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(warpgroup::groupid() + 1);
    } else {
        const int qid = warpgroup::groupid();
        warpgroup::increase_registers<184>();
#if defined(TK_FA4_DUAL_QUERY_QK_ONE_QUERY) && TK_FA4_DUAL_QUERY_QK_ONE_QUERY
        if (qid != 0) {
#if (defined(TK_FA4_DUAL_QUERY_STOP_AFTER_QK) && TK_FA4_DUAL_QUERY_STOP_AFTER_QK) || \
    (defined(TK_FA4_DUAL_QUERY_STOP_AFTER_P) && TK_FA4_DUAL_QUERY_STOP_AFTER_P)
            goto fp4pv_quarter_lifetime_two_query_overlay_role_done;
#else
            return;
#endif
        }
#endif
#if defined(TK_FA4_DUAL_QUERY_QK_NO_SCORE_LOAD) && TK_FA4_DUAL_QUERY_QK_NO_SCORE_LOAD
        goto fp4pv_quarter_lifetime_two_query_overlay_role_done;
#endif
#if defined(TK_FA4_DUAL_QUERY_STOP_AFTER_LOADS) && TK_FA4_DUAL_QUERY_STOP_AFTER_LOADS
        goto fp4pv_quarter_lifetime_two_query_overlay_role_done;
#endif
        float row_max = base_types::constants<float>::neg_infty();
        float row_sum = 0.0f;
        const bool owner_active = qid == 0 || q1_active;
        const int owner_query_tile = qid == 0 ? q0_tile : q1_tile;
        const int owner_iters = owner_active ? owner_query_tile + 1 : 0;

        #pragma unroll 1
        for (int idx = 0; idx < owner_iters; ++idx) {
            if (idx > 0) {
                wait(p_reusable[qid], (idx - 1) & 1);
            }
            wait(scores_arrived[qid], idx & 1);
            tensor_after_thread_sync();
            const uint32_t score_row_base =
                (qid == 0 ? tt_score0.addr : tt_score1.addr) +
                ((warpgroup::warpid() * 32) << 16);

            // Max pass: scan quarters 1..3 with a single scratch quarter,
            // then load quarter0 last and retain only that one quarter through
            // the max reduction.  No four-quarter score array exists.
            float tile_max = base_types::constants<float>::neg_infty();
            float quarter1_max = base_types::constants<float>::neg_infty();
            float quarter2_max = base_types::constants<float>::neg_infty();
            float quarter3_max = base_types::constants<float>::neg_infty();
            fp4pv_stage2_static_for<1, 4>([&](auto quarter_tag) {
                constexpr int QUARTER = decltype(quarter_tag)::value;
                float2 scores_quarter[16];
                fp4pv_load_score_quarter_tmem(
                    scores_quarter, score_row_base, QUARTER);
                tensor_load_wait();
                fp4pv_apply_online_causal_mask_score_quarter<C>(
                    scores_quarter, QUARTER, idx, owner_query_tile);
                float value = base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    asm volatile("{max.f32 %0, %1, %2, %3;}"
                                 : "=f"(value)
                                 : "f"(value),
                                   "f"(scores_quarter[j].x),
                                   "f"(scores_quarter[j].y));
                }
                if constexpr (C::QUARTER_LIFETIME_PRUNED) {
                    if constexpr (QUARTER == 1) {
                        quarter1_max = value;
                    } else if constexpr (QUARTER == 2) {
                        quarter2_max = value;
                    } else {
                        quarter3_max = value;
                    }
                }
                tile_max = fmaxf(tile_max, value);
            });

            float2 retained_quarter0[16];
            fp4pv_load_score_quarter_tmem(
                retained_quarter0, score_row_base, 0);
            tensor_load_wait();
            fp4pv_apply_online_causal_mask_score_quarter<C>(
                retained_quarter0, 0, idx, owner_query_tile);
            float quarter0_max = base_types::constants<float>::neg_infty();
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                quarter0_max = fmaxf(
                    quarter0_max,
                    fmaxf(retained_quarter0[j].x, retained_quarter0[j].y));
            }
            tile_max = fmaxf(tile_max, quarter0_max);

            const float row_max_old = row_max;
            row_max = fmaxf(row_max, tile_max);
            float acc_scale = 1.0f;
            if (idx > 0) {
                acc_scale = softmax_rescale_factor_strict(
                    row_max, row_max_old, scale_log2);
            }

            // Correction depends only on the complete N128 max, so publish it
            // before P packing.  This lets the output WG rescale while QK1 and
            // the remaining q0 quarter rereads are in flight.
            stats_store(correction[qid], warpgroup::laneid(), acc_scale);
            if constexpr (!C::QUARTER_LIFETIME_FUSE_CORRECTION_R0) {
                warpgroup::sync(warpgroup::groupid() + 1);
                warpgroup::arrive(correction_ready[qid]);
            }

            float tile_sum = 0.0f;
            const float2 score_scale = {scale_log2, scale_log2};
            auto pack_quarter = [&](auto quarter_tag,
                                    float2 (&scores_quarter)[16],
                                    float block_max) {
                constexpr int QUARTER = decltype(quarter_tag)::value;
                uint8_t p_e8m0 = 0;
                if (isfinite(block_max) && isfinite(row_max)) {
                    const float block_scale_log2 =
                        FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                        (block_max - row_max) * scale_log2;
                    p_e8m0 =
                        fp4pv_log2_to_e8m0_floor_finite(block_scale_log2);
                }
                const float p_log2_coeff = p_e8m0 == 0
                    ? 0.0f
                    : FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE -
                          static_cast<float>(static_cast<int>(p_e8m0) - 127);
                const float p_rcp_coeff =
                    fp4pv_exp2f_e8m0(p_e8m0) *
                    FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;
                fp4pv_store_mxfp4_scale_byte_swizzled(
                    p_sc_smem[qid * 2 + (QUARTER >> 1)],
                    warpgroup::laneid(), QUARTER, p_e8m0);
                uint32_t words[4];
                float2 sum0 = {0.0f, 0.0f};
                float2 sum1 = {0.0f, 0.0f};
                const float bias =
                    row_max * (-scale_log2) + p_log2_coeff;
                const float2 bias2 = {bias, bias};
                fp4pv_stage2_static_for<0, 4>([&](auto pack_tag) {
                    constexpr int PACK = decltype(pack_tag)::value;
                    fp4pv_stage2_static_for<0, 4>([&](auto element_tag) {
                        constexpr int ELEMENT = decltype(element_tag)::value;
                        constexpr int J = PACK * 4 + ELEMENT;
                        if (!zero_scale) {
                            scores_quarter[J] = __ffma2_rn(
                                scores_quarter[J], score_scale, bias2);
                        } else {
                            scores_quarter[J].x =
                                isfinite(scores_quarter[J].x)
                                    ? p_log2_coeff
                                    : scores_quarter[J].x;
                            scores_quarter[J].y =
                                isfinite(scores_quarter[J].y)
                                    ? p_log2_coeff
                                    : scores_quarter[J].y;
                        }
                        // Production native EX2 path.  The paired parent uses
                        // the scalar native helper when its emulation period is
                        // zero; do not instantiate the cadence emulator with 0.
                        scores_quarter[J].x =
                            fp4pv_exp2_approx(scores_quarter[J].x);
                        scores_quarter[J].y =
                            fp4pv_exp2_approx(scores_quarter[J].y);
                        if constexpr ((J & 1) == 0) {
                            sum0 = __fadd2_rn(sum0, scores_quarter[J]);
                        } else {
                            sum1 = __fadd2_rn(sum1, scores_quarter[J]);
                        }
                    });
                    words[PACK] = p_e8m0 == 0
                        ? 0u
                        : fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(
                              &scores_quarter[PACK * 4]);
                });
                sum0 = __fadd2_rn(sum0, sum1);
                tile_sum +=
                    (sum0.x + sum0.y) * p_rcp_coeff;
                fp4pv_store_quantized_scores_group_mxfp4(
                    p_smem[qid], words, warpgroup::laneid(), QUARTER);
            };

            // R0: quarter0 is the sole retained quarter.  Its release permits
            // the issuer to stage q(1-qid) Q/K scales into these 32 columns.
            pack_quarter(
                std::integral_constant<int, 0>{},
                retained_quarter0, quarter0_max);
            if constexpr (!C::QUARTER_LIFETIME_PRUNED) {
                publish_shared_backing();
            }
            tensor_before_thread_sync();
            warpgroup::sync(warpgroup::groupid() + 1);
            warpgroup::arrive(score_region_dead[qid][0]);
            if constexpr (C::QUARTER_LIFETIME_FUSE_CORRECTION_R0) {
                warpgroup::arrive(correction_ready[qid]);
            }

            // R1..R3 are reread only when packed, so each can be handed off
            // immediately after its final TMEM read and WG-local publication.
            fp4pv_stage2_static_for<1, 4>([&](auto quarter_tag) {
                constexpr int QUARTER = decltype(quarter_tag)::value;
                float2 scores_quarter[16];
                fp4pv_load_score_quarter_tmem(
                    scores_quarter, score_row_base, QUARTER);
                tensor_load_wait();
                fp4pv_apply_online_causal_mask_score_quarter<C>(
                    scores_quarter, QUARTER, idx, owner_query_tile);
                float block_max;
                if constexpr (C::QUARTER_LIFETIME_PRUNED) {
                    if constexpr (QUARTER == 1) {
                        block_max = quarter1_max;
                    } else if constexpr (QUARTER == 2) {
                        block_max = quarter2_max;
                    } else {
                        block_max = quarter3_max;
                    }
                } else {
                    block_max = base_types::constants<float>::neg_infty();
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        block_max = fmaxf(
                            block_max,
                            fmaxf(scores_quarter[j].x, scores_quarter[j].y));
                    }
                }
                pack_quarter(quarter_tag, scores_quarter, block_max);
                if constexpr (!C::QUARTER_LIFETIME_PRUNED ||
                              QUARTER == 1 || QUARTER == 3) {
                    publish_shared_backing();
                    tensor_before_thread_sync();
                    warpgroup::sync(warpgroup::groupid() + 1);
                }
                if constexpr (!C::QUARTER_LIFETIME_PRUNED) {
                    warpgroup::arrive(score_region_dead[qid][QUARTER]);
                }
                if constexpr (QUARTER == 1) {
                    // P quarters0+1 and their scale bytes are now visible.
                    warpgroup::arrive(p_half_ready[qid]);
                } else if constexpr (QUARTER == 3) {
                    // P quarters2+3 and the complete scale tile are visible.
                    warpgroup::arrive(p_full_ready[qid]);
                }
            });

            row_sum = row_sum * acc_scale + tile_sum;
        }
        if (owner_active) {
            stats_store(row_max_shared[qid], warpgroup::laneid(), row_max);
            stats_store(row_sum_shared[qid], warpgroup::laneid(), row_sum);
            warpgroup::sync(warpgroup::groupid() + 1);
            warpgroup::arrive(stats_ready[qid]);
        }
    }
fp4pv_quarter_lifetime_two_query_overlay_role_done:
    // tensor_allocator is deprovisioned by block warp 0.  Join every role so
    // q0 cannot release TMEM while q1 or the shared QK/PV issuer still owns it.
    __syncthreads();
}

inline void check_quarter_lifetime_two_query_inputs(
    const at::Tensor &Q,
    const at::Tensor &Q_sc,
    const at::Tensor &Q_sg,
    const at::Tensor &K,
    const at::Tensor &K_sc,
    const at::Tensor &K_sg,
    const at::Tensor &V_fp4,
    const at::Tensor &V_sc,
    const at::Tensor &O,
    const at::Tensor &LSE
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V_fp4); CHECK_INPUT(V_sc); CHECK_INPUT(O); CHECK_INPUT(LSE);
    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "quarter-lifetime Q/K/V must be packed E2M1");
    TORCH_CHECK(Q.dim() == 4 && Q.size(3) == 96 && K.sizes() == Q.sizes(),
                "quarter-lifetime Q/K must be [B,H,S,96 packed]");
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(batch > 0 && heads > 0 && seqlen > 0 && seqlen % 128 == 0,
                "quarter-lifetime route requires S divisible by 128");
    TORCH_CHECK(V_fp4.dim() == 4 && V_fp4.size(0) == batch &&
                V_fp4.size(1) == heads && V_fp4.size(2) == 128 &&
                V_fp4.size(3) == seqlen / 2,
                "quarter-lifetime V must be [B,H,128,S/2 packed]");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte && V_sc.dim() == 6 &&
                V_sc.size(0) == batch && V_sc.size(1) == heads &&
                V_sc.size(2) == 2 && V_sc.size(3) == seqlen / 128 &&
                V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "quarter-lifetime V_sc must be [B,H,2,S/128,32,16]");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 && O.dim() == 4 &&
                O.size(0) == batch && O.size(1) == seqlen &&
                O.size(2) == heads && O.size(3) == 128,
                "quarter-lifetime O must be BF16 [B,S,H,128]");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float && LSE.dim() == 4 &&
                LSE.size(0) == batch && LSE.size(1) == heads &&
                LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "quarter-lifetime LSE must be FP32 [B,H,1,S]");
}

template <typename C>
void dispatch_quarter_lifetime_two_query_overlay_impl(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    bool store_lse
) {
    using G = globals_fp4pv_mxfp4_dv<C, true>;
    check_quarter_lifetime_two_query_inputs(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE);
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    const int k_sc_depth = seqlen / 64;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(
            Q_sc, batch, seqlen / C::Mb,
            heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(
            Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(
            K_sc, batch, k_sc_depth,
            heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(
            K_sg, batch, heads, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        1.0f,
        2};
    g.disable_lse_store = store_lse ? 0 : 1;
    const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
    const dim3 grid(batch * heads * query_pairs_per_head);
    constexpr int dynamic_shared_memory =
        fp4pv_quarter_lifetime_two_query_overlay_smem_layout<C>::bytes();
    CUDACHECK(cudaFuncSetAttribute(
        kernel_fp4pv_quarter_lifetime_two_query_overlay<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    kernel_fp4pv_quarter_lifetime_two_query_overlay<C>
        <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
           at::cuda::getCurrentCUDAStream()>>>(g);
    CUDACHECK(cudaGetLastError());
}

void dispatch_quarter_lifetime_two_query_overlay(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_quarter_lifetime_two_query_overlay_impl<
        config_fp4pv_quarter_lifetime_two_query_overlay>(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_quarter_lifetime_two_query_overlay_pruned(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_quarter_lifetime_two_query_overlay_impl<
        config_fp4pv_quarter_lifetime_two_query_overlay_pruned>(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_quarter_lifetime_two_query_overlay_pruned_fused(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_quarter_lifetime_two_query_overlay_impl<
        config_fp4pv_quarter_lifetime_two_query_overlay_pruned_fused>(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, store_lse);
}

#endif  // TK_FA4_QUARTER_LIFETIME_TWO_QUERY_BUILD

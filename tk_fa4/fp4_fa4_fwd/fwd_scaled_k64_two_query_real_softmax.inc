#pragma once

#ifndef TK_FA4_SCALED_K64_TWO_QUERY_REAL_BUILD
#define TK_FA4_SCALED_K64_TWO_QUERY_REAL_BUILD 0
#endif

#if TK_FA4_SCALED_K64_TWO_QUERY_REAL_BUILD

// Four-TMEM adjacent-query real-softmax route.  This forks the retained
// scaled-K64 ceiling's ownership and issue topology, but owns independent
// production-equivalent online-softmax state and never consumes fake P.
struct config_scaled_k64_two_query_real_serial
    : public config_fp4pv_stage2_ex2_alu_dual_query_two_score_overlay {
    static constexpr bool SERIAL_PHASE_CORRECT = true;
    static constexpr bool REVERSE_PAIR_ORDER = false;
    static constexpr bool SHARED_PRODUCER_WG = false;
    static constexpr bool FUSED_SCORE_REG_REUSE = false;
    static constexpr bool EARLY_SCORE_RELEASE = false;
    static constexpr bool Q1_QK_BEFORE_Q0_PV = false;
    static constexpr bool P_K64_HALF_READY = false;
    static constexpr bool P_K64_SCALE_PREP_AHEAD = false;
    static constexpr bool PARENT_EXACT_OUTPUT_RESCALE = false;
    static constexpr bool FIXED_PERSISTENT_JOB_LOOP = false;
    static constexpr bool CYCLE_DIAG = false;
    static constexpr int P_PAYLOAD_STAGES = 2;
};

struct config_scaled_k64_two_query_real_overlap
    : public config_fp4pv_stage2_ex2_alu_dual_query_two_score_overlay {
    static constexpr bool SERIAL_PHASE_CORRECT = false;
    static constexpr bool REVERSE_PAIR_ORDER = true;
    static constexpr bool SHARED_PRODUCER_WG = false;
    static constexpr bool FUSED_SCORE_REG_REUSE = false;
    static constexpr bool EARLY_SCORE_RELEASE = false;
    static constexpr bool Q1_QK_BEFORE_Q0_PV = false;
    static constexpr bool P_K64_HALF_READY = false;
    static constexpr bool P_K64_SCALE_PREP_AHEAD = false;
    static constexpr bool PARENT_EXACT_OUTPUT_RESCALE = false;
    static constexpr bool FIXED_PERSISTENT_JOB_LOOP = false;
    static constexpr bool CYCLE_DIAG = false;
    static constexpr int P_PAYLOAD_STAGES = 2;
    static constexpr int LOAD_STAGES = 3;
    static constexpr int V_LOAD_STAGES = 3;
    static constexpr int K_SCALE_SMEM_SLOTS = 3;
};

// v4 keeps the exact v3 ownership/cadence, but folds the dedicated loader WG
// into producer WG2: warp0 remains the sole TCGEN issuer and warp3 becomes the
// sole TMA loader.  Warps1/2 are intentionally idle in this first isolation.
struct config_scaled_k64_two_query_real_overlap_3wg
    : public config_scaled_k64_two_query_real_overlap {
    static constexpr bool SHARED_PRODUCER_WG = true;
    static constexpr int TOTAL_WGS = 3;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

// v5 isolates production-style score retention: each owner loads the four
// score quarters once, then reuses those registers for max, exp2, and P pack.
// Score release deliberately stays at the v4 position for this attribution.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse
    : public config_scaled_k64_two_query_real_overlap_3wg {
    static constexpr bool FUSED_SCORE_REG_REUSE = true;
    static constexpr bool EARLY_SCORE_RELEASE = false;
};

// v7 overlaps q1's QK service with q0's owner-side P construction instead of
// first blocking the sole issue warp on q0 p_ready. All score-bank ownership
// and PV-completion handoffs remain identical to v5.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early
    : public config_scaled_k64_two_query_real_overlap_3wg_score_reuse {
    static constexpr bool Q1_QK_BEFORE_Q0_PV = true;
    static constexpr bool EARLY_SCORE_RELEASE = true;
};

// v9 isolates query-local P queue backpressure. Payload and scale stay paired
// in four SMEM slots, while the exact two TMEM scale pages and two PV
// completion slots retain their existing ownership and phases.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_p4
    : public config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early {
    static constexpr int P_PAYLOAD_STAGES = 4;
};

// v10 publishes quarters 0/1 as soon as native scaled-K64 PV chunk 0 can
// consume them. The owner produces quarters 2/3 concurrently with chunk 0;
// payload depth and QK/PV command count remain identical to v7e.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_phalf
    : public config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early {
    static constexpr bool P_K64_HALF_READY = true;
};

// v11 precomputes all four P block scales before payload packing, copies that
// complete scale tile once, then separates the two native K64 commands only by
// the query-owner payload publications. Chunk 0 has no completion commit;
// chunk 1 publishes the sole final PV completion.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_scaleahead
    : public config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early {
    static constexpr bool P_K64_SCALE_PREP_AHEAD = true;
};

// Equal-effective-wave arithmetic candidate.  It keeps the proven adjacent
// query mapping, four-TMEM ownership, shared K/V traversal, and q1-early issue
// schedule, but restores the production online-softmax recurrence: after
// qk_done(i) proves PV(i-1) complete, the query owner rescales its resident
// output before publishing P(i).  No lazy correction is folded into P scales
// or deferred to the final store.  Native EX2 matches the production parent.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact
    : public config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early {
    static constexpr bool PARENT_EXACT_OUTPUT_RESCALE = true;
    static constexpr int ONLINE_STAGE2_EX2_EMU_PERIOD = 0;
    static constexpr uint32_t ONLINE_STAGE2_EX2_EMU_MASK = 0;
};

// Fixed-worker form of the accepted v28c arithmetic.  It owns the same four
// TMEM tiles for the lifetime of one physical CTA and executes independent
// paired-query jobs in a CTA-coherent grid-stride loop.  Every job resets the
// original semaphore surface; no QK/PV work is replayed or duplicated.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact_persistent
    : public config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact {
    static constexpr bool FIXED_PERSISTENT_JOB_LOOP = true;
};

// Supporting-only cycle clone of the exact q1-early schedule.  As with the
// retained-score cycle control, its event time is never performance evidence.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_cycles
    : public config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early {
    static constexpr bool CYCLE_DIAG = true;
};

// Supporting-only cycle clone. It records register-accumulated stage cycles
// once at the selected physical CTA's role tails; its event time is never an
// authoritative performance measurement.
struct config_scaled_k64_two_query_real_overlap_3wg_score_reuse_cycles
    : public config_scaled_k64_two_query_real_overlap_3wg_score_reuse {
    static constexpr bool CYCLE_DIAG = true;
};

template <typename C>
struct scaled_k64_two_query_real_smem_layout {
    using G = globals_fp4pv_mxfp4_dv<C, true>;

    static constexpr int bytes() {
        int total = 0;
        auto add = [&](int allocation) constexpr {
            total = align_up_constexpr(total, 1024) + allocation;
        };
        add(2 * (int)sizeof(typename G::q_tile));
        add((int)sizeof(typename G::q_sc_smem_storage));
        add(C::LOAD_STAGES * (int)sizeof(typename G::k_tile));
        add((int)sizeof(typename G::k_sc_smem_storage));
        add((int)sizeof(typename G::k_sc_guard_slot));
        add(C::LOAD_STAGES * (int)sizeof(typename G::v_fp4_tile));
        add(C::LOAD_STAGES * (int)sizeof(typename G::v_sc_tile));
        add(2 * C::P_PAYLOAD_STAGES *
            (int)sizeof(typename G::p_fp4_tile));
        add(2 * C::P_PAYLOAD_STAGES *
            (int)sizeof(typename G::p_sc_tile));
        return total + 1024;
    }
};

template <typename C, int BARRIER_ID, typename OutputTT>
__device__ inline void scaled_k64_two_query_real_zero_output_3wg(
    OutputTT &tt_output
) {
    using out_rt_fl = rt_fl<C::Mb / 4, C::OUTPUT_CHUNK_COLS>;
    using out_tt_chunk = full_tt_fl<C::OUTPUT_CHUNK_COLS>;
    out_rt_fl zeros;
    warp::zero(zeros);
    #pragma unroll
    for (int chunk = 0; chunk < C::OUTPUT_CHUNKS; ++chunk) {
        auto tt_chunk = tt_output.template subtile<out_tt_chunk>(
            chunk * C::OUTPUT_CHUNK_COLS);
        warpgroup::store_async(tt_chunk, zeros);
    }
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(BARRIER_ID);
}

template <typename C>
__cluster_dims__(1, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_scaled_k64_two_query_real_serial(
    const __grid_constant__ globals_fp4pv_mxfp4_dv<C, true> g
) {
    using G = globals_fp4pv_mxfp4_dv<C, true>;
    using Layout = scaled_k64_two_query_real_smem_layout<C>;
    using score_t = tt<float, C::Mb, C::Nb>;
    using output_t = tt<float, C::Mb, C::Dvo>;
    using qk_scale_t = full_tt_fp8e4m3<16>;
    using pv_scale_t = full_tt_fp8e8m0<16>;

    static_assert(C::CLUSTER_SIZE == 1 && C::NUM_SOFTMAXXERS == 2 &&
                  (C::SHARED_PRODUCER_WG
                       ? (C::TOTAL_WGS == 3 && C::NUM_THREADS == 384)
                       : (C::TOTAL_WGS == 4 && C::NUM_THREADS == 512)) &&
                  C::Mb == 128 && C::Nb == 128 &&
                  C::Dqk == 192 && C::Dvo == 128 &&
                  (C::LOAD_STAGES == 2 || C::LOAD_STAGES == 3) &&
                  C::V_LOAD_STAGES == C::LOAD_STAGES &&
                  C::K_SCALE_SMEM_SLOTS == C::LOAD_STAGES &&
                  (C::P_PAYLOAD_STAGES == 2 || C::P_PAYLOAD_STAGES == 4),
                  "real two-query route requires the frozen four-TMEM surface");
    static_assert(C::SERIAL_PHASE_CORRECT
                      ? C::LOAD_STAGES == 2
                      : C::LOAD_STAGES == 3,
                  "serial control uses two K/V stages; overlap uses three");
    static_assert(!C::EARLY_SCORE_RELEASE || C::FUSED_SCORE_REG_REUSE,
                  "early score release requires register-retained scores");
    static_assert(!C::P_K64_HALF_READY ||
                      (C::P_PAYLOAD_STAGES == 2 &&
                       C::Q1_QK_BEFORE_Q0_PV &&
                       !C::SERIAL_PHASE_CORRECT),
                  "K64-half P publication is isolated on the v7e schedule");
    static_assert(!C::P_K64_SCALE_PREP_AHEAD ||
                      (C::P_PAYLOAD_STAGES == 2 &&
                       C::Q1_QK_BEFORE_Q0_PV &&
                       C::FUSED_SCORE_REG_REUSE &&
                       !C::SERIAL_PHASE_CORRECT),
                  "early full P-scale preparation requires retained scores");
    static_assert(!(C::P_K64_HALF_READY &&
                    C::P_K64_SCALE_PREP_AHEAD),
                  "v10 and v11 P-publication experiments are isolated");
    static_assert(!C::PARENT_EXACT_OUTPUT_RESCALE ||
                      (C::Q1_QK_BEFORE_Q0_PV &&
                       C::FUSED_SCORE_REG_REUSE &&
                       !C::P_K64_HALF_READY &&
                       !C::P_K64_SCALE_PREP_AHEAD &&
                       fp4pv_online_stage2_ex2_emu_period<C>::value == 0),
                  "parent-exact paired recurrence requires the native-EX2 "
                  "q1-early full-P publication path");
    static_assert(!C::FIXED_PERSISTENT_JOB_LOOP ||
                      (C::PARENT_EXACT_OUTPUT_RESCALE &&
                       C::SHARED_PRODUCER_WG &&
                       C::TOTAL_WGS == 3 && C::NUM_THREADS == 384 &&
                       C::Q1_QK_BEFORE_Q0_PV &&
                       C::FUSED_SCORE_REG_REUSE &&
                       !C::CYCLE_DIAG && C::CLUSTER_SIZE == 1 &&
                       C::NUM_SM == 152),
                  "fixed persistent paired jobs require the accepted v28c "
                  "three-warpgroup arithmetic and one 152-worker wave");
    static_assert(!C::CYCLE_DIAG || FP4PV_PAIRPSC_DESC_DIAG_WORDS >= 44,
                  "q1-early cycle diagnostics require the 44-word record");
    static_assert(Layout::bytes() <= MAX_SHARED_MEMORY - 4096,
                  "real two-query route exceeds shared-memory budget");

    constexpr int SCORE0_BASE = 0;
    constexpr int SCORE1_BASE = 128;
    constexpr int OUTPUT0_BASE = 256;
    constexpr int OUTPUT1_BASE = 384;
    // WG-local sync IDs are role+1: 1..4 for v3 and 1..3 for v4.  ID4 is the
    // lowest nonconflicting CTA subset barrier in the 3WG route.
    constexpr int INIT_BARRIER_ID = C::SHARED_PRODUCER_WG ? 4 : 14;
    constexpr int INIT_BARRIER_THREADS = C::TOTAL_WGS * WARP_THREADS;
    static_assert(OUTPUT1_BASE + C::Dvo == MAX_TENSOR_COLS,
                  "real two-query route must own exactly four TMEM tiles");

    const int tiles = g.q.rows() / C::Mb;
    const int units = (tiles + 1) / 2;
    const int total_jobs = g.q.batch() * g.q.depth() * units;
    const int role = warpgroup::groupid();
    const bool issue_warp = role == 2 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const int loader_role = C::SHARED_PRODUCER_WG ? 2 : 3;
    const bool loader_warp =
        role == loader_role && warpgroup::warpid() == 3;
    const bool loader_lane = loader_warp && warp::elect_leader();

    tensor_allocator<1, 1> tm_alloc{};
    score_t score[2] = {
        tm_alloc.template allocate<score_t>(SCORE0_BASE),
        tm_alloc.template allocate<score_t>(SCORE1_BASE)};
    output_t output[2] = {
        tm_alloc.template allocate<output_t>(OUTPUT0_BASE),
        tm_alloc.template allocate<output_t>(OUTPUT1_BASE)};

    // Q/K scales live in the opposite score bank.  P/V scales live in the
    // same query's score bank, after that owner has released every score word.
    qk_scale_t q_sc_tm[2] = {
        tm_alloc.template allocate<qk_scale_t>(SCORE1_BASE + 0),
        tm_alloc.template allocate<qk_scale_t>(SCORE0_BASE + 0)};
    qk_scale_t k_sc_tm[2] = {
        tm_alloc.template allocate<qk_scale_t>(SCORE1_BASE + 16),
        tm_alloc.template allocate<qk_scale_t>(SCORE0_BASE + 16)};
    // Two disjoint P/V scale pages per query let the overlap route issue the
    // next query's QK without overwriting scales still owned by an older PV.
    pv_scale_t p_sc_tm[2][2] = {
        {tm_alloc.template allocate<pv_scale_t>(SCORE0_BASE + 32),
         tm_alloc.template allocate<pv_scale_t>(SCORE0_BASE + 64)},
        {tm_alloc.template allocate<pv_scale_t>(SCORE1_BASE + 32),
         tm_alloc.template allocate<pv_scale_t>(SCORE1_BASE + 64)}};
    pv_scale_t v_sc_tm[2][2] = {
        {tm_alloc.template allocate<pv_scale_t>(SCORE0_BASE + 48),
         tm_alloc.template allocate<pv_scale_t>(SCORE0_BASE + 80)},
        {tm_alloc.template allocate<pv_scale_t>(SCORE1_BASE + 48),
         tm_alloc.template allocate<pv_scale_t>(SCORE1_BASE + 80)}};

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile (&q_smem)[2] =
        al.allocate<typename G::q_tile, 2>();
    typename G::q_sc_smem_storage &q_sc_smem =
        al.allocate<typename G::q_sc_smem_storage>();
    typename G::k_tile (&k_smem)[C::LOAD_STAGES] =
        al.allocate<typename G::k_tile, C::LOAD_STAGES>();
    typename G::k_sc_smem_storage &k_sc_smem =
        al.allocate<typename G::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] =
        al.allocate<typename G::k_sc_guard_slot, 1>();
    (void)k_sc_guard;
    typename G::v_fp4_tile (&v_smem)[C::LOAD_STAGES] =
        al.allocate<typename G::v_fp4_tile, C::LOAD_STAGES>();
    typename G::v_sc_tile (&v_sc_smem)[C::LOAD_STAGES] =
        al.allocate<typename G::v_sc_tile, C::LOAD_STAGES>();
    typename G::p_fp4_tile (&p_smem)[2 * C::P_PAYLOAD_STAGES] =
        al.allocate<typename G::p_fp4_tile, 2 * C::P_PAYLOAD_STAGES>();
    typename G::p_sc_tile (&p_sc_smem)[2 * C::P_PAYLOAD_STAGES] =
        al.allocate<typename G::p_sc_tile, 2 * C::P_PAYLOAD_STAGES>();

    __shared__ semaphore q_arrived[2];
    __shared__ semaphore q_sc_arrived[2];
    __shared__ semaphore k_arrived[C::LOAD_STAGES];
    __shared__ semaphore k_sc_arrived[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore v_arrived[C::LOAD_STAGES];
    __shared__ semaphore v_sc_arrived[C::LOAD_STAGES];
    __shared__ semaphore k_reusable[C::LOAD_STAGES];
    __shared__ semaphore k_sc_reusable[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore v_reusable[C::LOAD_STAGES];
    __shared__ semaphore qk_done[2];
    __shared__ semaphore score_released[2];
    __shared__ semaphore p_scale_ready[2][C::P_PAYLOAD_STAGES];
    __shared__ semaphore p_ready[2][C::P_PAYLOAD_STAGES];
    __shared__ semaphore p_full_ready[2][C::P_PAYLOAD_STAGES];
    __shared__ semaphore p_first_scale_copied[2][C::P_PAYLOAD_STAGES];
    __shared__ semaphore p_reusable[2][C::P_PAYLOAD_STAGES];
    __shared__ semaphore output_rescale_ready[2];
    __shared__ semaphore pv_half_done[2][2];
    __shared__ semaphore pv_done[2][2];
    __shared__ semaphore qk_scale_done[2][C::QK_SCALE_CHUNKS];
    __shared__ semaphore pv_scale_done[2];
    __shared__ semaphore pv_scale_second_done[2];

    const int job_limit = C::FIXED_PERSISTENT_JOB_LOOP
        ? total_jobs
        : static_cast<int>(gridDim.x);
    #pragma unroll 1
    for (int linear_job = static_cast<int>(blockIdx.x);
         linear_job < job_limit;
         linear_job += static_cast<int>(gridDim.x)) {
        // The single-iteration scope converts role-local job completion into
        // a branch to the common CTA tail without terminating the physical
        // worker.  All task-specific state remains inside this scope.
        do {
            const int bh = linear_job / units;
            const int physical_unit = linear_job - bh * units;
            const int unit = C::REVERSE_PAIR_ORDER
                ? units - 1 - physical_unit
                : physical_unit;
            const int b = bh / g.q.depth();
            const int h = bh - b * g.q.depth();
            const int q_tile[2] = {2 * unit, 2 * unit + 1};
            const bool q_active[2] = {true, q_tile[1] < tiles};
            const int key_tiles =
                q_active[1] ? q_tile[1] + 1 : q_tile[0] + 1;
            const float scale_log2 = qk_scale_log2(g, b, h);
            const bool zero_scale = scale_log2 == 0.0f;

    if (issue_lane) {
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            init_semaphore(q_arrived[q], 0, 1);
            init_semaphore(q_sc_arrived[q], 0, 1);
            init_semaphore(qk_done[q], 0, 1);
            init_semaphore(score_released[q], 0, 1);
            if constexpr (C::SERIAL_PHASE_CORRECT) {
                init_semaphore(output_rescale_ready[q], 0, 1);
            }
            init_semaphore(pv_scale_done[q], 0, 1);
            if constexpr (C::P_K64_HALF_READY) {
                init_semaphore(pv_scale_second_done[q], 0, 1);
            }
            #pragma unroll
            for (int slot = 0; slot < C::P_PAYLOAD_STAGES; ++slot) {
                if constexpr (C::P_K64_SCALE_PREP_AHEAD) {
                    init_semaphore(p_scale_ready[q][slot], 0, 1);
                }
                init_semaphore(p_ready[q][slot], 0, 1);
                if constexpr (C::P_K64_HALF_READY ||
                              C::P_K64_SCALE_PREP_AHEAD) {
                    init_semaphore(p_full_ready[q][slot], 0, 1);
                }
                if constexpr (C::P_K64_HALF_READY) {
                    init_semaphore(p_first_scale_copied[q][slot], 0, 1);
                }
                init_semaphore(p_reusable[q][slot], 0, 1);
            }
            #pragma unroll
            for (int slot = 0; slot < 2; ++slot) {
                if constexpr (C::P_K64_HALF_READY) {
                    init_semaphore(pv_half_done[q][slot], 0, 1);
                }
                init_semaphore(pv_done[q][slot], 0, 1);
            }
            #pragma unroll
            for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                init_semaphore(qk_scale_done[q][chunk], 0, 1);
            }
        }
        #pragma unroll
        for (int slot = 0; slot < C::LOAD_STAGES; ++slot) {
            init_semaphore(k_arrived[slot], 0, 1);
            init_semaphore(k_sc_arrived[slot], 0, 1);
            init_semaphore(v_arrived[slot], 0, 1);
            init_semaphore(v_sc_arrived[slot], 0, 1);
            init_semaphore(k_reusable[slot], 0, 1);
            init_semaphore(k_sc_reusable[slot], 0, 1);
            init_semaphore(v_reusable[slot], 0, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warpgroup::warpid() == 0) {
        asm volatile("bar.sync %0, %1;" ::
                     "n"(INIT_BARRIER_ID), "n"(INIT_BARRIER_THREADS) :
                     "memory");
    }
    if constexpr (C::SHARED_PRODUCER_WG) {
        if (role == 0) {
            warpgroup::sync(1);
        } else if (role == 1) {
            warpgroup::sync(2);
        } else {
            warpgroup::sync(3);
        }
    } else {
        warpgroup::sync(role + 1);
    }

    if (loader_warp) {
        if (!loader_lane) {
            break;
        }
        const bool cycle_loader =
            C::CYCLE_DIAG && static_cast<int>(blockIdx.x) == 0;
        unsigned long long loader_reuse_cycles = 0;
        unsigned long long loader_service_cycles = 0;
        unsigned long long loader_total_begin =
            cycle_loader ? clock64() : 0;
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<typename G::q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<typename G::k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<typename G::k_sc_tile, dim::ROW>();
        g.v_fp4_tma.template prefetch_tma<typename G::v_fp4_tile, dim::ROW>();

        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            if (q_active[q]) {
                fp4pv_expect_bytes<C>(
                    q_arrived[q], sizeof(typename G::q_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    q_smem[q], g.q, {b, h, q_tile[q], 0}, q_arrived[q]);
                fp4pv_expect_bytes<C>(
                    q_sc_arrived[q], sizeof(typename G::q_sc_tile));
                tma::load_async<dim::ROW, cache_policy::NORMAL>(
                    slot_tile_at(q_sc_smem, q), g.q_sc,
                    {b, q_tile[q], h, 0}, q_sc_arrived[q]);
            }
        }

        #pragma unroll 1
        for (int idx = 0; idx < key_tiles; ++idx) {
            const int slot = idx % C::LOAD_STAGES;
            const int epoch = idx / C::LOAD_STAGES;
            if (idx >= C::LOAD_STAGES) {
                const unsigned long long reuse_begin =
                    cycle_loader ? clock64() : 0;
                wait(k_reusable[slot], (epoch - 1) & 1);
                wait(k_sc_reusable[slot], (epoch - 1) & 1);
                wait(v_reusable[slot], (epoch - 1) & 1);
                if (cycle_loader) {
                    loader_reuse_cycles += clock64() - reuse_begin;
                }
            }
            const unsigned long long service_begin =
                cycle_loader ? clock64() : 0;
            fp4pv_expect_bytes<C>(
                k_arrived[slot], sizeof(typename G::k_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                k_smem[slot], g.k, {b, h, idx, 0}, k_arrived[slot]);
            fp4pv_expect_bytes<C>(
                k_sc_arrived[slot], sizeof(typename G::k_sc_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                slot_tile_at(k_sc_smem, slot), g.k_sc,
                {b, 2 * idx, h, 0}, k_sc_arrived[slot]);
            fp4pv_expect_bytes<C>(
                v_arrived[slot], sizeof(typename G::v_fp4_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                v_smem[slot], g.v_fp4_tma,
                {b, h, 0, idx}, v_arrived[slot]);
            const int v_sc_offset =
                (((((b * g.q.depth() + h) * 2) *
                    (g.k.rows() / 128)) + idx) * 32 * 16);
            fp4pv_expect_bytes<C>(
                v_sc_arrived[slot], sizeof(typename G::v_sc_tile));
            tma::load_async(
                reinterpret_cast<void *>(&v_sc_smem[slot].data[0]),
                const_cast<uint8_t *>(&g.v_sc_prepared[v_sc_offset]),
                sizeof(typename G::v_sc_tile), v_sc_arrived[slot]);
            if (cycle_loader) {
                loader_service_cycles += clock64() - service_begin;
            }
        }
        if (cycle_loader) {
            fp4pv_pairpsc_desc_diag[18] =
                static_cast<unsigned long long>(key_tiles);
            fp4pv_pairpsc_desc_diag[19] = loader_reuse_cycles;
            fp4pv_pairpsc_desc_diag[20] = loader_service_cycles;
            fp4pv_pairpsc_desc_diag[21] =
                clock64() - loader_total_begin;
        }
        break;
    }
    if constexpr (!C::SHARED_PRODUCER_WG) {
        if (role == 3) {
            break;
        }
    }

    if (role == 0 || role == 1) {
        const int q = role;
        if (!q_active[q]) {
            break;
        }
        const bool cycle_owner =
            C::CYCLE_DIAG && static_cast<int>(blockIdx.x) == 0 &&
            warpgroup::warpid() == 0 && warp::elect_leader();
        unsigned long long owner_p_reuse_cycles = 0;
        unsigned long long owner_qk_wait_cycles = 0;
        unsigned long long owner_score_load_cycles = 0;
        unsigned long long owner_alu_pack_cycles = 0;
        unsigned long long owner_publish_cycles = 0;
        unsigned long long owner_final_pv_cycles = 0;
        unsigned long long owner_output_cycles = 0;
        unsigned long long owner_alu_begin = 0;
        int owner_work_count = 0;
        if constexpr (C::SHARED_PRODUCER_WG) {
            if (q == 0) {
                scaled_k64_two_query_real_zero_output_3wg<C, 1>(output[q]);
            } else {
                scaled_k64_two_query_real_zero_output_3wg<C, 2>(output[q]);
            }
        } else {
            zero_output_scratch<C>(output[q]);
        }

        float row_max = base_types::constants<float>::neg_infty();
        float row_sum = 0.0f;
        float output_lazy_log2 = 0.0f;
        float output_lazy_scale = 1.0f;
        const uint32_t score_row_base =
            score[q].addr + ((warpgroup::warpid() * 32) << 16);

        #pragma unroll 1
        for (int idx = 0; idx <= q_tile[q]; ++idx) {
            const int slot = idx % C::P_PAYLOAD_STAGES;
            const int epoch = idx / C::P_PAYLOAD_STAGES;
            if (idx >= C::P_PAYLOAD_STAGES) {
                const unsigned long long reuse_begin =
                    cycle_owner ? clock64() : 0;
                wait(p_reusable[q][slot], (epoch - 1) & 1);
                if (cycle_owner) {
                    owner_p_reuse_cycles += clock64() - reuse_begin;
                }
            }
            const unsigned long long qk_wait_begin =
                cycle_owner ? clock64() : 0;
            wait(qk_done[q], idx & 1);
            tensor_after_thread_sync();
            if (cycle_owner) {
                owner_qk_wait_cycles += clock64() - qk_wait_begin;
            }

            float2 retained_scores[
                C::FUSED_SCORE_REG_REUSE ? 4 : 1][16];
            float retained_block_max[
                C::FUSED_SCORE_REG_REUSE ? 4 : 1];
            float tile_max = base_types::constants<float>::neg_infty();
            if constexpr (C::FUSED_SCORE_REG_REUSE) {
                const unsigned long long score_load_begin =
                    cycle_owner ? clock64() : 0;
                #pragma unroll
                for (int quarter = 0; quarter < 4; ++quarter) {
                    fp4pv_load_score_quarter_tmem(
                        retained_scores[quarter], score_row_base, quarter);
                }
                tensor_load_wait();
                if (cycle_owner) {
                    owner_score_load_cycles +=
                        clock64() - score_load_begin;
                    owner_alu_begin = clock64();
                }
                if constexpr (C::EARLY_SCORE_RELEASE) {
                    // All four warps have captured every score word in
                    // registers; only now may the opposite query overwrite
                    // this score bank's aliased Q/K scale columns.
                    tensor_before_thread_sync();
                    if constexpr (C::SHARED_PRODUCER_WG) {
                        if (q == 0) {
                            warpgroup::sync(1);
                        } else {
                            warpgroup::sync(2);
                        }
                    } else {
                        warpgroup::sync(q + 1);
                    }
                    warpgroup::arrive(score_released[q]);
                }
                fp4pv_stage2_static_for<0, 4>([&](auto quarter_tag) {
                    constexpr int QUARTER = decltype(quarter_tag)::value;
                    fp4pv_apply_online_causal_mask_score_quarter<C>(
                        retained_scores[QUARTER], QUARTER, idx, q_tile[q]);
                    float value = base_types::constants<float>::neg_infty();
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        asm volatile("{max.f32 %0, %1, %2, %3;}"
                                     : "=f"(value)
                                     : "f"(value),
                                       "f"(retained_scores[QUARTER][j].x),
                                       "f"(retained_scores[QUARTER][j].y));
                    }
                    retained_block_max[QUARTER] = value;
                    tile_max = fmaxf(tile_max, value);
                });
            } else {
                fp4pv_stage2_static_for<0, 4>([&](auto quarter_tag) {
                    constexpr int QUARTER = decltype(quarter_tag)::value;
                    float2 scores_quarter[16];
                    fp4pv_load_score_quarter_tmem(
                        scores_quarter, score_row_base, QUARTER);
                    tensor_load_wait();
                    fp4pv_apply_online_causal_mask_score_quarter<C>(
                        scores_quarter, QUARTER, idx, q_tile[q]);
                    float value = base_types::constants<float>::neg_infty();
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        asm volatile("{max.f32 %0, %1, %2, %3;}"
                                     : "=f"(value)
                                     : "f"(value),
                                       "f"(scores_quarter[j].x),
                                       "f"(scores_quarter[j].y));
                    }
                    tile_max = fmaxf(tile_max, value);
                });
            }

            const float row_max_old = row_max;
            row_max = fmaxf(row_max, tile_max);
            float acc_scale = 1.0f;
            if (idx > 0) {
                acc_scale = softmax_rescale_factor_strict(
                    row_max, row_max_old, scale_log2);
                if constexpr (!C::PARENT_EXACT_OUTPUT_RESCALE) {
                    if (!zero_scale && isfinite(row_max_old) &&
                        isfinite(row_max) && isfinite(acc_scale)) {
                        output_lazy_log2 +=
                            (row_max_old - row_max) * scale_log2;
                        output_lazy_scale *= acc_scale;
                    }
                }
            }
            const float output_lazy_log2_rcp =
                C::PARENT_EXACT_OUTPUT_RESCALE ? 0.0f : -output_lazy_log2;

            uint32_t packed_scales = 0;
            uint8_t prepared_p_e8m0[
                C::P_K64_SCALE_PREP_AHEAD ? 4 : 1];
            if constexpr (C::P_K64_SCALE_PREP_AHEAD) {
                fp4pv_stage2_static_for<0, 4>([&](auto quarter_tag) {
                    constexpr int QUARTER = decltype(quarter_tag)::value;
                    const float block_max = retained_block_max[QUARTER];
                    uint8_t p_e8m0 = 0;
                    if (isfinite(block_max) && isfinite(row_max)) {
                        const float block_scale_log2 =
                            FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                            (block_max - row_max) * scale_log2 +
                            output_lazy_log2_rcp;
                        p_e8m0 = fp4pv_log2_to_e8m0_floor_finite(
                            block_scale_log2);
                    }
                    prepared_p_e8m0[QUARTER] = p_e8m0;
                    fp4pv_store_mxfp4_scale_byte_swizzled(
                        p_sc_smem[q * C::P_PAYLOAD_STAGES + slot],
                        warpgroup::laneid(), QUARTER, p_e8m0);
                });
                publish_shared_backing();
                if constexpr (C::SHARED_PRODUCER_WG) {
                    if (q == 0) {
                        warpgroup::sync(1);
                    } else {
                        warpgroup::sync(2);
                    }
                } else {
                    warpgroup::sync(q + 1);
                }
                warpgroup::arrive(p_scale_ready[q][slot]);
            }
            float tile_sum = 0.0f;
            const float2 score_scale = {scale_log2, scale_log2};
            auto pack_score_quarter = [&] (
                auto quarter_tag,
                auto &scores_quarter,
                float block_max
            ) {
                constexpr int QUARTER = decltype(quarter_tag)::value;
                uint8_t p_e8m0 = 0;
                if constexpr (C::P_K64_SCALE_PREP_AHEAD) {
                    p_e8m0 = prepared_p_e8m0[QUARTER];
                } else {
                    if (isfinite(block_max) && isfinite(row_max)) {
                        const float block_scale_log2 =
                            FP4PV_MXFP4_ONLINE_LOG2_P_SCALE +
                            (block_max - row_max) * scale_log2 +
                            output_lazy_log2_rcp;
                        p_e8m0 = fp4pv_log2_to_e8m0_floor_finite(
                            block_scale_log2);
                    }
                }
                const float p_log2_coeff = p_e8m0 == 0
                    ? 0.0f
                    : FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE -
                          static_cast<float>(
                              static_cast<int>(p_e8m0) - 127);
                const float p_rcp_coeff =
                    fp4pv_exp2f_e8m0(p_e8m0) *
                    FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;
                if constexpr (C::P_K64_HALF_READY) {
                    fp4pv_store_mxfp4_scale_byte_swizzled(
                        p_sc_smem[q * C::P_PAYLOAD_STAGES + slot],
                        warpgroup::laneid(), QUARTER, p_e8m0);
                } else if constexpr (!C::P_K64_SCALE_PREP_AHEAD) {
                    packed_scales |=
                        static_cast<uint32_t>(p_e8m0) << (QUARTER * 8);
                }
                uint32_t words[4];
                float2 sum0 = {0.0f, 0.0f};
                float2 sum1 = {0.0f, 0.0f};
                const float bias =
                    row_max * (-scale_log2) +
                    output_lazy_log2_rcp + p_log2_coeff;
                const float2 bias2 = {bias, bias};
                fp4pv_stage2_static_for<0, 4>([&](auto pack_tag) {
                    constexpr int PACK = decltype(pack_tag)::value;
                    fp4pv_stage2_static_for<0, 4>([&](auto element_tag) {
                        constexpr int ELEMENT = decltype(element_tag)::value;
                        constexpr int J = PACK * 4 + ELEMENT;
                        constexpr int GLOBAL_SI = QUARTER * 16 + J;
                        if (!zero_scale) {
                            scores_quarter[J] = __ffma2_rn(
                                scores_quarter[J], score_scale, bias2);
                        } else {
                            scores_quarter[J].x =
                                isfinite(scores_quarter[J].x)
                                    ? output_lazy_log2_rcp + p_log2_coeff
                                    : scores_quarter[J].x;
                            scores_quarter[J].y =
                                isfinite(scores_quarter[J].y)
                                    ? output_lazy_log2_rcp + p_log2_coeff
                                    : scores_quarter[J].y;
                        }
                        if constexpr (C::PARENT_EXACT_OUTPUT_RESCALE) {
                            scores_quarter[J].x =
                                fp4pv_exp2_approx(scores_quarter[J].x);
                            scores_quarter[J].y =
                                fp4pv_exp2_approx(scores_quarter[J].y);
                        } else {
                            scores_quarter[J] = fp4pv_stage2_exp2_pair<
                                fp4pv_online_stage2_ex2_emu_period<C>::value,
                                fp4pv_online_stage2_ex2_emu_mask<C>::value,
                                GLOBAL_SI>(scores_quarter[J]);
                        }
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
                    p_smem[q * C::P_PAYLOAD_STAGES + slot], words,
                    warpgroup::laneid(), QUARTER);
                if constexpr ((C::P_K64_HALF_READY ||
                               C::P_K64_SCALE_PREP_AHEAD) &&
                              QUARTER == 1) {
                    // Both 32-column quarters required by scaled K64 chunk 0
                    // are now complete. This is query-owner-local only.
                    publish_shared_backing();
                    if constexpr (C::SHARED_PRODUCER_WG) {
                        if (q == 0) {
                            warpgroup::sync(1);
                        } else {
                            warpgroup::sync(2);
                        }
                    } else {
                        warpgroup::sync(q + 1);
                    }
                    warpgroup::arrive(p_ready[q][slot]);
                    if constexpr (C::P_K64_HALF_READY) {
                        wait(p_first_scale_copied[q][slot], epoch & 1);
                        tensor_after_thread_sync();
                    }
                }
            };
            fp4pv_stage2_static_for<0, 4>([&](auto quarter_tag) {
                constexpr int QUARTER = decltype(quarter_tag)::value;
                if constexpr (C::FUSED_SCORE_REG_REUSE) {
                    pack_score_quarter(
                        quarter_tag,
                        retained_scores[QUARTER],
                        retained_block_max[QUARTER]);
                } else {
                    float2 scores_quarter[16];
                    fp4pv_load_score_quarter_tmem(
                        scores_quarter, score_row_base, QUARTER);
                    tensor_load_wait();
                    fp4pv_apply_online_causal_mask_score_quarter<C>(
                        scores_quarter, QUARTER, idx, q_tile[q]);
                    float block_max =
                        base_types::constants<float>::neg_infty();
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        block_max = fmaxf(
                            block_max,
                            fmaxf(scores_quarter[j].x, scores_quarter[j].y));
                    }
                    pack_score_quarter(
                        quarter_tag, scores_quarter, block_max);
                }
            });

            if (cycle_owner) {
                owner_alu_pack_cycles += clock64() - owner_alu_begin;
            }
            const unsigned long long publish_begin =
                cycle_owner ? clock64() : 0;
            if constexpr (!C::EARLY_SCORE_RELEASE) {
                tensor_before_thread_sync();
                if constexpr (C::SHARED_PRODUCER_WG) {
                    if (q == 0) {
                        warpgroup::sync(1);
                    } else {
                        warpgroup::sync(2);
                    }
                } else {
                    warpgroup::sync(q + 1);
                }
                warpgroup::arrive(score_released[q]);
            }
            if constexpr (!C::P_K64_HALF_READY &&
                          !C::P_K64_SCALE_PREP_AHEAD) {
                fp4pv_store_mxfp4_scale_word_swizzled(
                    p_sc_smem[q * C::P_PAYLOAD_STAGES + slot],
                    warpgroup::laneid(), packed_scales);
            }
            if constexpr (C::PARENT_EXACT_OUTPUT_RESCALE) {
                // qk_done(i) is published only after the issue owner has
                // observed this query's PV(i-1) completion.  Rescale here,
                // after retained-score packing has ended, so no score float
                // gains the output-load/store live range.  P(i) is published
                // only after the warpgroup has reconverged below.
                if (idx > 0) {
                    const bool lane_needs_rescale = acc_scale < 0.999f;
                    const bool needs_rescale =
                        __any_sync(0xffffffffu, lane_needs_rescale);
                    if (needs_rescale) {
                        // Reconstruct the fixed query-local TMEM descriptor at
                        // the use site.  Indexing output[q] here extends the
                        // descriptor array through retained-score packing and
                        // makes ptxas reload its address from the local stack.
                        // The allocation columns are compile-time invariants,
                        // so this descriptor carries no additional state.
                        output_t output_rescale(tm_alloc.get_addr(
                            q == 0 ? OUTPUT0_BASE : OUTPUT1_BASE));
                        rescale_tt_output_inplace_direct_streaming<C>(
                            output_rescale, acc_scale, lane_needs_rescale);
                    }
                    if (q == 0) {
                        warpgroup::sync(1);
                    } else {
                        warpgroup::sync(2);
                    }
                }
                row_sum = row_sum * acc_scale + tile_sum;
            } else {
                row_sum =
                    row_sum * acc_scale + tile_sum * output_lazy_scale;
            }
            publish_shared_backing();
            if constexpr (C::SHARED_PRODUCER_WG) {
                if (q == 0) {
                    warpgroup::sync(1);
                } else {
                    warpgroup::sync(2);
                }
            } else {
                warpgroup::sync(q + 1);
            }
            if constexpr (C::P_K64_HALF_READY ||
                          C::P_K64_SCALE_PREP_AHEAD) {
                warpgroup::arrive(p_full_ready[q][slot]);
            } else {
                warpgroup::arrive(p_ready[q][slot]);
            }
            if (cycle_owner) {
                owner_publish_cycles += clock64() - publish_begin;
                ++owner_work_count;
            }
            if constexpr (C::SERIAL_PHASE_CORRECT) {
                // Bring-up-only redundant control: p_ready already implies
                // that lazy correction is folded into payload and scale.
                warpgroup::arrive(output_rescale_ready[q]);
            }
        }

        const int final_p_slot = q_tile[q] & 1;
        const int final_p_phase = (q_tile[q] >> 1) & 1;
        const unsigned long long final_pv_begin =
            cycle_owner ? clock64() : 0;
        wait(pv_done[q][final_p_slot], final_p_phase);
        tensor_after_thread_sync();
        if (cycle_owner) {
            owner_final_pv_cycles += clock64() - final_pv_begin;
        }
        const unsigned long long output_begin =
            cycle_owner ? clock64() : 0;
        const int row = warpgroup::laneid();
        const bool invalid = row_sum == 0.0f || row_sum != row_sum;
        float inv_norm;
        asm volatile("rcp.approx.ftz.f32 %0, %1;"
                     : "=f"(inv_norm)
                     : "f"(invalid ? 1.0f : row_sum));
        constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
        const float output_scale =
            (MXFP4_ALPHA / FP4PV_MXFP4_ONLINE_P_SCALE) * inv_norm *
            (C::PARENT_EXACT_OUTPUT_RESCALE ? 1.0f : output_lazy_scale);
        const int s = q_tile[q] * C::Mb + row;
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
                : "r"(output[q].addr +
                      ((warpgroup::warpid() * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float x = invalid ? 0.0f : out[i].x;
                const float y = invalid ? 0.0f : out[i].y;
                g.o[{b, s, h, col + i * 2 + 0}] =
                    __float2bfloat16(x * output_scale);
                g.o[{b, s, h, col + i * 2 + 1}] =
                    __float2bfloat16(y * output_scale);
            }
        }
        if (g.disable_lse_store == 0) {
            constexpr float LN2 = 0.693147180559945f;
            float lse = base_types::constants<float>::neg_infty();
            if (!invalid) {
                float log2_sum;
                asm("lg2.approx.ftz.f32 %0, %1;"
                    : "=f"(log2_sum) : "f"(row_sum));
                lse = (row_max * scale_log2 + log2_sum) * LN2;
            }
            g.lse[{b, h, 0, s}] = lse;
        }
        if (cycle_owner) {
            owner_output_cycles += clock64() - output_begin;
            const int diag_base = q == 0 ? 2 : 10;
            fp4pv_pairpsc_desc_diag[diag_base + 0] =
                static_cast<unsigned long long>(owner_work_count);
            fp4pv_pairpsc_desc_diag[diag_base + 1] =
                owner_p_reuse_cycles;
            fp4pv_pairpsc_desc_diag[diag_base + 2] =
                owner_qk_wait_cycles;
            fp4pv_pairpsc_desc_diag[diag_base + 3] =
                owner_score_load_cycles;
            fp4pv_pairpsc_desc_diag[diag_base + 4] =
                owner_alu_pack_cycles;
            fp4pv_pairpsc_desc_diag[diag_base + 5] =
                owner_publish_cycles;
            fp4pv_pairpsc_desc_diag[diag_base + 6] =
                owner_final_pv_cycles;
            fp4pv_pairpsc_desc_diag[diag_base + 7] =
                owner_output_cycles;
        }
        break;
    }

    if (!issue_warp) {
        break;
    }

    const bool cycle_issue =
        C::CYCLE_DIAG && static_cast<int>(blockIdx.x) == 0 && issue_lane;
    unsigned long long issue_k_wait_cycles = 0;
    unsigned long long issue_qk_service_cycles = 0;
    unsigned long long issue_v_wait_cycles = 0;
    unsigned long long issue_p_ready_wait_cycles = 0;
    unsigned long long issue_pv_service_cycles = 0;
    unsigned long long issue_retire_cycles[2] = {0, 0};
    unsigned long long issue_bank_cycles[2] = {0, 0};
    unsigned long long issue_q1early_q0_release_wait_cycles = 0;
    unsigned long long issue_q1early_prior_q1_pv_wait_cycles = 0;
    int issue_qk_count = 0;
    int issue_pv_count = 0;
    int issue_retire_count[2] = {0, 0};
    int issue_bank_count[2] = {0, 0};
    int issue_q1early_q0_release_wait_count = 0;
    int issue_q1early_prior_q1_pv_wait_count = 0;
    const unsigned long long issue_total_begin =
        cycle_issue ? clock64() : 0;

    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        if (q_active[q]) {
            wait(q_arrived[q], 0);
            wait(q_sc_arrived[q], 0);
        }
    }

    auto issue_qk = [&](int q, int idx) {
        const int slot = idx % C::LOAD_STAGES;
        const int phase = (idx / C::LOAD_STAGES) & 1;
        const unsigned long long k_wait_begin =
            cycle_issue ? clock64() : 0;
        wait(k_arrived[slot], phase);
        wait(k_sc_arrived[slot], phase);
        if (cycle_issue) {
            issue_k_wait_cycles += clock64() - k_wait_begin;
        }
        const unsigned long long qk_service_begin =
            cycle_issue ? clock64() : 0;
        using payload_t = typename G::q_tile::T;
        using scale_t = typename qk_scale_t::T;
        using result_t = typename score_t::T;
        constexpr uint32_t IDESC =
            detail::tcgen05::instruction_descriptor<
                result_t, payload_t, scale_t,
                C::Mb, C::Nb, false, 0>();
        st_descriptor<typename G::q_tile, 0> q_desc(q_smem[q]);
        st_descriptor<typename G::k_tile, 0> k_desc(k_smem[slot]);
        #pragma unroll
        for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
            if (issue_lane) {
                fp4pv_load_q_scale_chunk_ranklocal(
                    q_sc_tm[q], slot_tile_at(q_sc_smem, q), chunk);
                fp4pv_load_k_scale_chunk_ranklocal(
                    k_sc_tm[q], slot_tile_at(k_sc_smem, slot), chunk);
                detail::tcgen05::commit<1>(qk_scale_done[q][chunk]);
            }
            wait(qk_scale_done[q][chunk], idx & 1);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
            if (issue_lane) {
                if (chunk == 0) {
                    detail::tcgen05::template st_st<
                        payload_t, scale_t, 0, 1, 16>(
                            score[q].addr,
                            q_desc.chunk_descriptor(chunk),
                            k_desc.chunk_descriptor(chunk),
                            q_sc_tm[q].addr, k_sc_tm[q].addr, IDESC);
                } else {
                    detail::tcgen05::template st_st<
                        payload_t, scale_t, 1, 1, 16>(
                            score[q].addr,
                            q_desc.chunk_descriptor(chunk),
                            k_desc.chunk_descriptor(chunk),
                            q_sc_tm[q].addr, k_sc_tm[q].addr, IDESC);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done[q]);
        }
        if (cycle_issue) {
            issue_qk_service_cycles += clock64() - qk_service_begin;
            ++issue_qk_count;
        }
    };

    auto issue_pv = [&](int q, int idx) {
        const int p_payload_slot = idx % C::P_PAYLOAD_STAGES;
        const int p_scale_slot = idx & 1;
        const int v_slot = idx % C::LOAD_STAGES;
        const int p_payload_phase =
            (idx / C::P_PAYLOAD_STAGES) & 1;
        const int p_scale_phase = (idx >> 1) & 1;
        const int v_phase = (idx / C::LOAD_STAGES) & 1;
        const unsigned long long v_wait_begin =
            cycle_issue ? clock64() : 0;
        wait(v_arrived[v_slot], v_phase);
        wait(v_sc_arrived[v_slot], v_phase);
        if (cycle_issue) {
            issue_v_wait_cycles += clock64() - v_wait_begin;
        }
        const unsigned long long p_ready_wait_begin =
            cycle_issue ? clock64() : 0;
        if constexpr (C::P_K64_SCALE_PREP_AHEAD) {
            wait(p_scale_ready[q][p_payload_slot], p_payload_phase);
        } else {
            wait(p_ready[q][p_payload_slot], p_payload_phase);
            if (cycle_issue) {
                issue_p_ready_wait_cycles +=
                    clock64() - p_ready_wait_begin;
            }
        }
        if constexpr (C::SERIAL_PHASE_CORRECT) {
            wait(output_rescale_ready[q], idx & 1);
        }
        const unsigned long long pv_service_begin =
            cycle_issue ? clock64() : 0;
        __threadfence_block();
        fp4pv_publish_shared_backing_proxy_only();
        if (issue_lane) {
            fp4pv_load_mxnv_scale_async_ranklocal(
                p_sc_tm[q][p_scale_slot],
                p_sc_smem[q * C::P_PAYLOAD_STAGES + p_payload_slot]);
            fp4pv_load_mxnv_scale_async_ranklocal(
                v_sc_tm[q][p_scale_slot], v_sc_smem[v_slot]);
            detail::tcgen05::commit<1>(pv_scale_done[q]);
        }
        wait(pv_scale_done[q], idx & 1);
        tensor_after_thread_sync();
        if constexpr (C::P_K64_SCALE_PREP_AHEAD) {
            const unsigned long long payload_ready_begin =
                cycle_issue ? clock64() : 0;
            wait(p_ready[q][p_payload_slot], p_payload_phase);
            if (cycle_issue) {
                issue_p_ready_wait_cycles +=
                    clock64() - payload_ready_begin;
            }
            // The scale proxy fence precedes low-half payload publication.
            // Publish that disjoint payload half before its TCGEN command.
            __threadfence_block();
            fp4pv_publish_shared_backing_proxy_only();
        }
        if constexpr (C::P_K64_HALF_READY) {
            if (issue_lane) {
                arrive(p_first_scale_copied[q][p_payload_slot]);
            }
        }
        tensor_before_thread_sync();
        if constexpr (C::P_K64_HALF_READY) {
            if (issue_lane) {
                if (idx == 0) {
                    fp4pv_p_stage_ABt_k64_chunk<
                        0, 0, 0, true, C>(
                            output[q],
                            p_smem[
                                q * C::P_PAYLOAD_STAGES + p_payload_slot],
                            v_smem[v_slot], p_sc_tm[q][p_scale_slot],
                            v_sc_tm[q][p_scale_slot],
                            pv_half_done[q][p_scale_slot]);
                } else {
                    fp4pv_p_stage_ABt_k64_chunk<
                        0, 1, 0, true, C>(
                            output[q],
                            p_smem[
                                q * C::P_PAYLOAD_STAGES + p_payload_slot],
                            v_smem[v_slot], p_sc_tm[q][p_scale_slot],
                            v_sc_tm[q][p_scale_slot],
                            pv_half_done[q][p_scale_slot]);
                }
            }

            // Chunk 0 runs while the owner finishes quarters 2/3. Completion
            // owns the P-scale TMEM page before the second scale copy.
            wait(p_full_ready[q][p_payload_slot], p_payload_phase);
            wait(pv_half_done[q][p_scale_slot], p_scale_phase);
            tensor_after_thread_sync();
            __threadfence_block();
            fp4pv_publish_shared_backing_proxy_only();
            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(
                    p_sc_tm[q][p_scale_slot],
                    p_sc_smem[
                        q * C::P_PAYLOAD_STAGES + p_payload_slot]);
                detail::tcgen05::commit<1>(pv_scale_second_done[q]);
            }
            wait(pv_scale_second_done[q], idx & 1);
            tensor_after_thread_sync();
            tensor_before_thread_sync();
            if (issue_lane) {
                fp4pv_p_stage_ABt_k64_chunk<
                    1, 1, 2, true, C>(
                        output[q],
                        p_smem[q * C::P_PAYLOAD_STAGES + p_payload_slot],
                        v_smem[v_slot], p_sc_tm[q][p_scale_slot],
                        v_sc_tm[q][p_scale_slot],
                        pv_done[q][p_scale_slot]);
            }
        } else if constexpr (C::P_K64_SCALE_PREP_AHEAD) {
            if (issue_lane) {
                if (idx == 0) {
                    fp4pv_p_stage_ABt_k64_chunk<
                        0, 0, 0, false, C>(
                            output[q],
                            p_smem[
                                q * C::P_PAYLOAD_STAGES + p_payload_slot],
                            v_smem[v_slot], p_sc_tm[q][p_scale_slot],
                            v_sc_tm[q][p_scale_slot],
                            pv_done[q][p_scale_slot]);
                } else {
                    fp4pv_p_stage_ABt_k64_chunk<
                        0, 1, 0, false, C>(
                            output[q],
                            p_smem[
                                q * C::P_PAYLOAD_STAGES + p_payload_slot],
                            v_smem[v_slot], p_sc_tm[q][p_scale_slot],
                            v_sc_tm[q][p_scale_slot],
                            pv_done[q][p_scale_slot]);
                }
            }
            // Chunk 0 has no completion commit and may execute while the
            // owner packs quarters 2/3.  The following mbarrier wait is the
            // execution-ordering operation that makes the high payload live;
            // fence subsequent TCGEN work after that wait, not before it.
            wait(p_full_ready[q][p_payload_slot], p_payload_phase);
            tensor_after_thread_sync();
            __threadfence_block();
            fp4pv_publish_shared_backing_proxy_only();
            tensor_before_thread_sync();
            if (issue_lane) {
                fp4pv_p_stage_ABt_k64_chunk<
                    1, 1, 2, true, C>(
                        output[q],
                        p_smem[q * C::P_PAYLOAD_STAGES + p_payload_slot],
                        v_smem[v_slot], p_sc_tm[q][p_scale_slot],
                        v_sc_tm[q][p_scale_slot],
                        pv_done[q][p_scale_slot]);
            }
        } else if (issue_lane) {
            if (idx == 0) {
                fp4pv_mm_p_stage_ranklocal_ABt<C>(
                    output[q],
                    p_smem[q * C::P_PAYLOAD_STAGES + p_payload_slot],
                    v_smem[v_slot], p_sc_tm[q][p_scale_slot],
                    v_sc_tm[q][p_scale_slot],
                    pv_done[q][p_scale_slot]);
            } else {
                fp4pv_mma_p_stage_ranklocal_ABt<C>(
                    output[q],
                    p_smem[q * C::P_PAYLOAD_STAGES + p_payload_slot],
                    v_smem[v_slot], p_sc_tm[q][p_scale_slot],
                    v_sc_tm[q][p_scale_slot],
                    pv_done[q][p_scale_slot]);
            }
        }
        if constexpr (C::SERIAL_PHASE_CORRECT) {
            wait(pv_done[q][p_scale_slot], p_scale_phase);
            tensor_after_thread_sync();
            if (issue_lane) {
                arrive(p_reusable[q][p_payload_slot]);
            }
        }
        if (cycle_issue) {
            issue_pv_service_cycles += clock64() - pv_service_begin;
            ++issue_pv_count;
        }
    };

    if constexpr (C::SERIAL_PHASE_CORRECT) {
        #pragma unroll 1
        for (int idx = 0; idx < key_tiles; ++idx) {
            const bool q0_uses = idx <= q_tile[0];
            const bool q1_uses = q_active[1] && idx <= q_tile[1];
            if (q0_uses) {
                issue_qk(0, idx);
                wait(score_released[0], idx & 1);
                tensor_after_thread_sync();
                issue_pv(0, idx);
            }
            if (q1_uses) {
                issue_qk(1, idx);
                wait(score_released[1], idx & 1);
                tensor_after_thread_sync();
                issue_pv(1, idx);
            }
            if (issue_lane) {
                const int slot = idx % C::LOAD_STAGES;
                arrive(k_reusable[slot]);
                arrive(k_sc_reusable[slot]);
                arrive(v_reusable[slot]);
            }
        }
    } else {
        auto retire_pv = [&](int idx) {
            const bool q0_uses = idx <= q_tile[0];
            const bool q1_uses = q_active[1] && idx <= q_tile[1];
            if (q0_uses) {
                const unsigned long long retire_begin =
                    cycle_issue ? clock64() : 0;
                wait(pv_done[0][idx & 1], (idx >> 1) & 1);
                tensor_after_thread_sync();
                if (cycle_issue) {
                    issue_retire_cycles[0] += clock64() - retire_begin;
                    ++issue_retire_count[0];
                }
                if (issue_lane) {
                    if constexpr (C::P_PAYLOAD_STAGES == 2) {
                        // Preserve the frozen v7e machine expression.
                        arrive(p_reusable[0][idx & 1]);
                    } else {
                        arrive(p_reusable[0][idx & 3]);
                    }
                }
            }
            if (q1_uses) {
                const unsigned long long retire_begin =
                    cycle_issue ? clock64() : 0;
                wait(pv_done[1][idx & 1], (idx >> 1) & 1);
                tensor_after_thread_sync();
                if (cycle_issue) {
                    issue_retire_cycles[1] += clock64() - retire_begin;
                    ++issue_retire_count[1];
                }
                if (issue_lane) {
                    if constexpr (C::P_PAYLOAD_STAGES == 2) {
                        // Preserve the frozen v7e machine expression.
                        arrive(p_reusable[1][idx & 1]);
                    } else {
                        arrive(p_reusable[1][idx & 3]);
                    }
                }
            }
            if (issue_lane) {
                const int slot = idx % C::LOAD_STAGES;
                arrive(k_reusable[slot]);
                arrive(k_sc_reusable[slot]);
                arrive(v_reusable[slot]);
            }
        };

        // Seed q0. v5 preserves PV0,QK1,PV1,QK0-next; v7 moves only QK1
        // ahead of the blocking PV0 P-ready wait.
        issue_qk(0, 0);
        #pragma unroll 1
        for (int idx = 0; idx < key_tiles; ++idx) {
            if (idx >= 2) {
                retire_pv(idx - 2);
            }
            const bool q0_uses = idx <= q_tile[0];
            const bool q1_uses = q_active[1] && idx <= q_tile[1];
            if constexpr (C::Q1_QK_BEFORE_Q0_PV) {
                if (q1_uses) {
                    // q_sc_tm[1]/k_sc_tm[1] alias score0 columns 0..31.
                    // Wait only while q0 has a score at this key tile. Every
                    // adjacent causal pair has one unmatched q1-only final
                    // tile, for which q0 emits no score-release phase.
                    if (q0_uses) {
                        const unsigned long long release_begin =
                            cycle_issue ? clock64() : 0;
                        wait(score_released[0], idx & 1);
                        tensor_after_thread_sync();
                        if (cycle_issue) {
                            issue_q1early_q0_release_wait_cycles +=
                                clock64() - release_begin;
                            ++issue_q1early_q0_release_wait_count;
                        }
                    }
                    if (idx > 0) {
                        // Preserve the exact v5 score1 scale-bank handoff.
                        const int previous = idx - 1;
                        const unsigned long long prior_q1_pv_begin =
                            cycle_issue ? clock64() : 0;
                        wait(
                            pv_done[1][previous & 1],
                            (previous >> 1) & 1);
                        tensor_after_thread_sync();
                        if (cycle_issue) {
                            issue_q1early_prior_q1_pv_wait_cycles +=
                                clock64() - prior_q1_pv_begin;
                            ++issue_q1early_prior_q1_pv_wait_count;
                        }
                    }
                    issue_qk(1, idx);
                }
            }
            if (q0_uses) {
                issue_pv(0, idx);
            }
            if (q1_uses) {
                if constexpr (!C::Q1_QK_BEFORE_Q0_PV) {
                    if (idx > 0) {
                        // QK1 overwrites score1, including q1's P/V scale
                        // pages. PV1(i-1) completion owns this handoff.
                        const int previous = idx - 1;
                        const unsigned long long bank_begin =
                            cycle_issue ? clock64() : 0;
                        wait(
                            pv_done[1][previous & 1],
                            (previous >> 1) & 1);
                        tensor_after_thread_sync();
                        if (cycle_issue) {
                            issue_bank_cycles[1] += clock64() - bank_begin;
                            ++issue_bank_count[1];
                        }
                    }
                    issue_qk(1, idx);
                }
                issue_pv(1, idx);
            }
            if (idx + 1 <= q_tile[0]) {
                // Symmetric ownership handoff before QK0 overwrites score0.
                const unsigned long long bank_begin =
                    cycle_issue ? clock64() : 0;
                wait(pv_done[0][idx & 1], (idx >> 1) & 1);
                tensor_after_thread_sync();
                if (cycle_issue) {
                    issue_bank_cycles[0] += clock64() - bank_begin;
                    ++issue_bank_count[0];
                }
                issue_qk(0, idx + 1);
            }
        }
        if (key_tiles >= 2) {
            retire_pv(key_tiles - 2);
        }
        retire_pv(key_tiles - 1);
    }
    if (cycle_issue) {
        fp4pv_pairpsc_desc_diag[22] = 0x56354359434c4553ULL;
        fp4pv_pairpsc_desc_diag[23] =
            static_cast<unsigned long long>(q_tile[0] & 0xffff) |
            (static_cast<unsigned long long>(q_tile[1] & 0xffff) << 16) |
            (static_cast<unsigned long long>(key_tiles & 0xffff) << 32);
        fp4pv_pairpsc_desc_diag[24] =
            static_cast<unsigned long long>(issue_qk_count);
        fp4pv_pairpsc_desc_diag[25] =
            static_cast<unsigned long long>(issue_pv_count);
        fp4pv_pairpsc_desc_diag[26] = issue_k_wait_cycles;
        fp4pv_pairpsc_desc_diag[27] = issue_qk_service_cycles;
        fp4pv_pairpsc_desc_diag[28] = issue_v_wait_cycles;
        fp4pv_pairpsc_desc_diag[29] = issue_p_ready_wait_cycles;
        fp4pv_pairpsc_desc_diag[30] = issue_pv_service_cycles;
        fp4pv_pairpsc_desc_diag[31] = issue_retire_cycles[0];
        fp4pv_pairpsc_desc_diag[32] = issue_retire_cycles[1];
        fp4pv_pairpsc_desc_diag[33] = issue_bank_cycles[0];
        fp4pv_pairpsc_desc_diag[34] = issue_bank_cycles[1];
        fp4pv_pairpsc_desc_diag[35] = clock64() - issue_total_begin;
        fp4pv_pairpsc_desc_diag[36] =
            static_cast<unsigned long long>(issue_retire_count[0]);
        fp4pv_pairpsc_desc_diag[37] =
            static_cast<unsigned long long>(issue_retire_count[1]);
        fp4pv_pairpsc_desc_diag[38] =
            static_cast<unsigned long long>(issue_bank_count[0]);
        fp4pv_pairpsc_desc_diag[39] =
            static_cast<unsigned long long>(issue_bank_count[1]);
        fp4pv_pairpsc_desc_diag[40] =
            issue_q1early_q0_release_wait_cycles;
        fp4pv_pairpsc_desc_diag[41] =
            static_cast<unsigned long long>(
                issue_q1early_q0_release_wait_count);
        fp4pv_pairpsc_desc_diag[42] =
            issue_q1early_prior_q1_pv_wait_cycles;
        fp4pv_pairpsc_desc_diag[43] =
            static_cast<unsigned long long>(
                issue_q1early_prior_q1_pv_wait_count);
    }
        } while (false);

        if constexpr (C::FIXED_PERSISTENT_JOB_LOOP) {
            // Every role has completed the current paired job.  Fence TCGEN
            // visibility before the full-CTA rendezvous; only then may the
            // issue lane reinitialize the single-CTA semaphore surface for
            // the next logical job.
            tensor_before_thread_sync();
            __syncthreads();
            tensor_after_thread_sync();
        }
    }
}

static void check_scaled_k64_two_query_real_inputs(
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
                "real two-query Q/K/V must be packed E2M1");
    TORCH_CHECK(Q.dim() == 4 && Q.size(3) == 96 && K.sizes() == Q.sizes(),
                "real two-query Q/K must have shape [B,H,S,96 packed]");
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(batch > 0 && heads > 0 && seqlen > 0 &&
                seqlen % 128 == 0,
                "real two-query requires positive B/H and S divisible by 128");
    TORCH_CHECK(V_fp4.dim() == 4 && V_fp4.size(0) == batch &&
                V_fp4.size(1) == heads && V_fp4.size(2) == 128 &&
                V_fp4.size(3) == seqlen / 2,
                "real two-query V must have shape [B,H,128,S/2 packed]");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte && V_sc.dim() == 6 &&
                V_sc.size(0) == batch && V_sc.size(1) == heads &&
                V_sc.size(2) == 2 && V_sc.size(3) == seqlen / 128 &&
                V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "real two-query V_sc must have canonical [B,H,2,S/128,32,16]");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 && O.dim() == 4 &&
                O.size(0) == batch && O.size(1) == seqlen &&
                O.size(2) == heads && O.size(3) == 128,
                "real two-query O must be BF16 [B,S,H,128]");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float && LSE.dim() == 4 &&
                LSE.size(0) == batch && LSE.size(1) == heads &&
                LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "real two-query LSE must be FP32 [B,H,1,S]");
}

template <typename C>
static void dispatch_scaled_k64_two_query_real(
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
    check_scaled_k64_two_query_real_inputs(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE);
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    // K scale rows are K64-addressed even though the payload traversal is
    // N128: each logical key tile owns two consecutive scale-depth rows.
    constexpr int K_SCALE_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / K_SCALE_DEPTH_DIVISOR;
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
    const int units = ((seqlen / C::Mb) + 1) / 2;
    const int total_jobs = batch * heads * units;
    const int grid_x = C::FIXED_PERSISTENT_JOB_LOOP
        ? (total_jobs < C::NUM_SM ? total_jobs : C::NUM_SM)
        : total_jobs;
    TORCH_CHECK(!C::FIXED_PERSISTENT_JOB_LOOP ||
                    (grid_x > 0 && grid_x <= C::NUM_SM),
                "fixed persistent paired route requires 1..152 workers");
    const dim3 grid(grid_x);
    constexpr int dynamic_shared_memory =
        scaled_k64_two_query_real_smem_layout<C>::bytes();
    CUDACHECK(cudaFuncSetAttribute(
        kernel_scaled_k64_two_query_real_serial<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    kernel_scaled_k64_two_query_real_serial<C>
        <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
           at::cuda::getCurrentCUDAStream()>>>(g);
    CUDACHECK(cudaGetLastError());
}

void dispatch_scaled_k64_two_query_real_serial(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_serial>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_cycles(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse_cycles>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_p4(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_p4>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_phalf(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_phalf>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_scaleahead(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_scaleahead>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact_persistent(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact_persistent>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

void dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_cycles(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, bool store_lse
) {
    dispatch_scaled_k64_two_query_real<
        config_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_cycles>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg,
            V_fp4, V_sc, O, LSE, store_lse);
}

#endif

// Fused single-warpgroup attention stream (venue A, v0: one stream per CTA).
//
// One 128-thread warpgroup does everything for its own (b, h, m) tile: TMA
// issue, V loads, softmax, quant, payload store, its own QK/PV MMA issue, and
// its own accumulator rescale. There is no cross-warpgroup protocol at all -
// every ordering that the production kernel buys with semaphore hops is
// program order here. The only mbarriers are data arrivals (TMA) and the
// per-tile PV drain commit, which alone covers payload-slot reuse, V-slot
// reuse, rescale-vs-PV ordering, and the epilogue (the waits are sequential).
//
// v0 scope: CLUSTER_SIZE == 1, Nb = 64 (single score slot; QK(i+1) issues
// right after the LDTM of tile i in program order, so no double buffering is
// needed), fullgrid only, extra warpgroups of the inherited 3WG config park
// at the end barrier. Validation bar: det == 0, refs at production floor.

template <typename C, bool DYNAMIC_P_SCALE = false, bool DIRECT_P_SCALE_TMEM = false>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_fp4pv_fused_stream(
    const __grid_constant__ globals_fp4pv_mxfp4_dv<C, true> g
) {
    using G = globals_fp4pv_mxfp4_dv<C, true>;
    static_assert(C::CLUSTER_SIZE == 1, "fused stream v0 is single-CTA");
    static_assert(C::Nb == 128,
        "fused stream v0 is currently restored for the production-proven Nb=128 tile layout");
    constexpr int SCORE_CHUNKS = C::Nb / 32;          // LDTM x32 chunks
    constexpr int OUT_CHUNKS = C::Dvo / 32;

    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_q_sc = full_tt_fp8e4m3<16 * C::QK_SCALE_CHUNKS>;
    using d_tt_k_sc = full_tt_fp8e4m3<16>;
    using d_tt_p_sc = full_tt_fp8e8m0<16>;
    using d_tt_v_sc = full_tt_fp8e8m0<32>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    typename G::q_tile (&q_smem)[1] = al.allocate<typename G::q_tile, 1>();
    typename G::q_sc_smem_storage &q_sc_smem = al.allocate<typename G::q_sc_smem_storage>();
    using k_ld_t = typename G::k_tile;
    k_ld_t (&k_smem)[2] = al.allocate<k_ld_t, 2>();
    typename G::k_sc_smem_storage &k_sc_smem = al.allocate<typename G::k_sc_smem_storage>();
    typename G::v_fp4_tile (&v_fp4_smem)[2] = al.allocate<typename G::v_fp4_tile, 2>();
    typename G::v_sc_tile (&v_sc_smem)[2] = al.allocate<typename G::v_sc_tile, 2>();
    typename G::p_fp4_tile (&p_fp4_stage)[2] = al.allocate<typename G::p_fp4_tile, 2>();
    typename G::p_sc_tile (&p_sc_stage)[1] = al.allocate<typename G::p_sc_tile, 1>();
    typename G::p_sc_tile (&qk_sc_zero)[3] = al.allocate<typename G::p_sc_tile, 3>();

    __shared__ semaphore q_arrived[1], q_sc_arrived[1];
    __shared__ semaphore k_arrived[2], k_sc_arrived[2];
    __shared__ semaphore scores_arrived[2];
    __shared__ semaphore pv_drained[1];
    __shared__ semaphore sc_staged[1];
    __shared__ semaphore k_scratch[1];
    __shared__ uint32_t p_dyn_scale_words[C::Mb];

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};

    const int tiles_m = g.q.rows() / C::Mb;
    const int num_head = g.q.depth();
    const int bid = blockIdx.x;
    // Largest m first: long causal tasks start in the first wave so the grid
    // tail is short tasks (mirrors the production reverse-m wave packing).
    const int m_tile = tiles_m - 1 - (bid % tiles_m);
    const int h = (bid / tiles_m) % num_head;
    const int b = bid / (tiles_m * num_head);
    const int iters = (m_tile + 1) * (C::Mb / C::Nb);

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<typename G::q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<k_ld_t, dim::ROW>();
        g.k_sc.template prefetch_tma<typename G::k_sc_tile, dim::ROW>();
        init_semaphore(q_arrived[0], 0, 1);
        init_semaphore(q_sc_arrived[0], 0, 1);
        for (int i = 0; i < 2; ++i) {
            init_semaphore(k_arrived[i], 0, 1);
            init_semaphore(k_sc_arrived[i], 0, 1);
        }
        init_semaphore(scores_arrived[0], 0, 1);
        init_semaphore(scores_arrived[1], 0, 1);
        init_semaphore(pv_drained[0], 0, 1);
        init_semaphore(sc_staged[0], 0, 1);
        init_semaphore(k_scratch[0], 0, 1);
    }
    if constexpr (!DYNAMIC_P_SCALE) {
        // Constant P scale (fixed-pscale family semantics).
        fp4pv_fill_mxfp4_scale_stage_word_swizzled(
            p_sc_stage[0], C::Mb, FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_WORD);
    }
    // The v5 backend folds QK scales upstream: zero e4m3 QK-scale tiles are
    // the semantic no-op (verified: heads whose TMA staging lands zeros are
    // bit-correct at Nb=128 AND Nb=64; nonzero content is mis-consumed by the
    // N=64 descriptor). Stage zeros for q_sc/k_sc always.
    for (int w = threadIdx.x; w < 3 * (int)sizeof(typename G::p_sc_tile) / 4; w += blockDim.x) {
        reinterpret_cast<uint32_t *>(&qk_sc_zero[0].data[0])[w] = 0u;
    }
    publish_shared_backing();
    __syncthreads();

    constexpr int WG_BAR = 14;  // end barrier for all warpgroups
    if (warpgroup::groupid() == 0) {
        const bool leader = (warp::laneid() == 0) && (warpgroup::warpid() == 0);
        const int lane = warpgroup::laneid();
        const bool row_active = lane < C::Mb;
        const int query_col = m_tile * C::Mb + lane;

        // TMEM map.
        auto tt_score0 = tm_alloc.template allocate<d_tt_scores>(0);
        auto tt_score1 = tm_alloc.template allocate<d_tt_scores>(C::Nb);
        auto tt_output = tm_alloc.template allocate<d_tt_outputs>(2 * C::Nb);
        constexpr int SC_BASE = 2 * C::Nb + C::Dvo;
        static_assert(SC_BASE + 16 * C::QK_SCALE_CHUNKS + 64 <= 512, "fused stream TMEM budget");
        auto q_sc_tm = tm_alloc.template allocate<d_tt_q_sc>(SC_BASE);
        // k_sc TMEM is double-buffered: two QKs are in flight (the lookahead
        // issue at loop top), and a single staging slot lets QK(i+1)'s scale
        // cp overwrite the slot QK(i)'s running MMA still reads. This race is
        // timing-masked at Nb=128 and bites at Nb=64 (nondeterministic LSE).
        // At Nb=64 the k_sc tensors hold all 3 chunks (preloaded issue path:
        // staging is completion-waited, the issue is pure MMAs). At Nb=128
        // they stay 16-col per-issue staging slots (frozen reference path).
        constexpr int KSC_W = (C::Nb == 64) ? 16 * C::QK_SCALE_CHUNKS : 16;
        using d_tt_k_sc_active = full_tt_fp8e4m3<KSC_W>;
        auto k_sc_tm0 = tm_alloc.template allocate<d_tt_k_sc_active>(SC_BASE + 16 * C::QK_SCALE_CHUNKS);
        auto k_sc_tm1 = tm_alloc.template allocate<d_tt_k_sc_active>(SC_BASE + 16 * C::QK_SCALE_CHUNKS + KSC_W);
        auto p_sc_tm = tm_alloc.template allocate<d_tt_p_sc>(SC_BASE + 16 * C::QK_SCALE_CHUNKS + 2 * KSC_W);
        auto v_sc_tm = tm_alloc.template allocate<d_tt_v_sc>(SC_BASE + 16 * C::QK_SCALE_CHUNKS + 2 * KSC_W + 16);
        static_assert(SC_BASE + 16 * C::QK_SCALE_CHUNKS + 2 * KSC_W + 16 + 32 <= 512,
            "fused stream TMEM budget");
        // Completion-waited scale staging: cps for all chunks, one commit on
        // the last, wait the parity. Per the law: timing slack is not a
        // protocol - the helper's no-sem cp -> MMA pattern races at Nb=64.
        int sc_staged_phase = 0;
        auto stage_scales_waited = [&](d_tt_k_sc_active &dst, auto &src_tile) {
            #pragma unroll
            for (int c = 0; c < C::QK_SCALE_CHUNKS; ++c) {
                auto sub = dst.template subtile<full_tt_fp8e4m3<16>>(c * 16);
                auto &sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
                    reinterpret_cast<uint64_t>(&src_tile.data[0]) + 16 * 32 * c);
                if (c + 1 < C::QK_SCALE_CHUNKS) {
                    load_mxnv_scale_async<1>(sub, sm_sub);
                } else {
                    load_mxnv_scale_async<1>(sub, sm_sub, sc_staged[0]);
                }
            }
            wait(sc_staged[0], sc_staged_phase);
            sc_staged_phase ^= 1;
        };

#ifdef FSK_SF_MARKER
        // SF marker experiment: replace staged scale content with synthetic
        // patterns IDENTICALLY in both kernels; output diffs isolate
        // consumption differences. Mode 1: K markers (Q constant 1.0).
        // Mode 2: Q markers (K constant 1.0). Pattern: byte at page-linear
        // position p gets e4m3 2^((p/4) % 3 - 1) = {0.5, 1, 2} by position.
        auto fsk_overwrite = [&](auto &tile, bool markers) {
            auto *bytes = reinterpret_cast<uint8_t *>(&tile.data[0]);
            for (int i = (int)threadIdx.x % 128; i < 1536; i += 128) {
                bytes[i] = markers ? (uint8_t)(0x28 + 8 * ((uint32_t)(i * 2654435761u) >> 28) % 5 * 8 / 8) : (uint8_t)0x38;
                if (markers) bytes[i] = (uint8_t)(0x70 + ((uint32_t)(i * 2654435761u) >> 28) % 15);
            }
            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        };
#endif
        const float SCALE_LOG2 = qk_scale_log2(g, b, h);
        const bool ZERO_SCALE = (SCALE_LOG2 == 0.0f);

        // Prologue loads.
        if (leader) {
            const coord<typename G::q_tile> q_coord = {b, h, m_tile, 0};
            // PRODUCTION-COMPATIBLE coordinate (known bug, kept for bit-parity
            // with the frozen reference): the scale gl d2 coord is in 3-row
            // tile units, so h*QK_SCALE_CHUNKS over-multiplies - heads 1-3
            // consume heads 3/6/9's scale constants and heads >=4 consume OOB
            // zeros. The correct coord is plain h (validated: deterministic,
            // ~0.13 LSE shift = the heads' true scales). Ship decision pending.
            const coord<typename G::q_sc_tile> q_sc_coord = {b, m_tile, h * C::QK_SCALE_CHUNKS, 0};
            fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(q_smem[0], g.q, q_coord, q_arrived[0], (uint16_t)1, 0);
            fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(q_sc_smem, 0), g.q_sc, q_sc_coord, q_sc_arrived[0], (uint16_t)1, 0);
            constexpr int TPB0 = (C::Nb == 64) ? 2 : 1;
            const int nblk0 = (iters + TPB0 - 1) / TPB0;
            for (int sld = 0; sld < 2 && sld < nblk0; ++sld) {
                const coord<k_ld_t> k_coord = {b, h, sld, 0};
                const coord<typename G::k_sc_tile> k_sc_coord = {b, sld, h * C::QK_SCALE_CHUNKS, 0};
                fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(k_smem[sld], g.k, k_coord, k_arrived[sld], (uint16_t)1, 0);
                fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(k_sc_smem, sld), g.k_sc, k_sc_coord, k_sc_arrived[sld], (uint16_t)1, 0);
            }
        }
        // V loads are split LDG-prefetch (loop top, latency hidden under the
        // tile compute) + STS-commit (after the drain wait frees the slot).
        const int v_packed_seqlen = g.k.rows() / 2;
        constexpr int V_STORES_PER_ROW = (C::Nb / 2) / 16;
        constexpr int V_LDG_PER_LANE = (C::Dvo * V_STORES_PER_ROW + 127) / 128;
        uint4 v_pf_vals[V_LDG_PER_LANE];
        uint32_t v_pf_addrs[V_LDG_PER_LANE];
        uint32_t v_sc_pf_val;
        uint32_t v_sc_pf_addr;
        auto v_prefetch_ldg = [&](int k_iter, int slot) {
            const int col_base = k_iter * (C::Nb / 2);
            const uint32_t smem_base = __cvta_generic_to_shared(&v_fp4_smem[slot].data[0]);
            #pragma unroll
            for (int u = 0; u < V_LDG_PER_LANE; ++u) {
                const int linear = lane + u * 128;
                const int row = linear / V_STORES_PER_ROW;
                const int col = (linear % V_STORES_PER_ROW) * 16;
                v_pf_vals[u] = *reinterpret_cast<const uint4 *>(
                    &g.v_fp4[((((b * num_head + h) * C::Dvo) + row) * v_packed_seqlen) + col_base + col]);
                v_pf_addrs[u] = G::v_fp4_tile::idx(smem_base, {row, col});
            }
            // v_sc: one word per lane covers the whole tile. The prepared global
            // is 128-row granular: at Nb=64 select the matching half-word.
            constexpr int VSC_WORDS = (int)sizeof(typename G::v_sc_tile) / 4;
            constexpr int VSC_ITERS_PER_TILE = 128 / C::Nb;
            const int scale_tile = k_iter / VSC_ITERS_PER_TILE;
            const int scale_half = k_iter - scale_tile * VSC_ITERS_PER_TILE;
            const int vsc_base = ((((b * num_head + h) * (C::Dvo / 128) + 0) * (g.k.rows() / 128)) + scale_tile) * 32 * 16;
            if (lane < VSC_WORDS) {
                uint32_t packed = *reinterpret_cast<const uint32_t *>(&g.v_sc_prepared[vsc_base + lane * 4]);
                if constexpr (C::Nb == 64) {
                    packed = scale_half == 0 ? (packed & 0x0000ffffu) : ((packed >> 16) & 0x0000ffffu);
                }
                v_sc_pf_val = packed;
                v_sc_pf_addr = __cvta_generic_to_shared(
                    reinterpret_cast<uint32_t *>(&v_sc_smem[slot].data[0]) + lane);
            }
        };
        auto v_commit_sts = [&]() {
            #pragma unroll
            for (int u = 0; u < V_LDG_PER_LANE; ++u) {
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                             :: "r"(v_pf_addrs[u]), "r"(v_pf_vals[u].x), "r"(v_pf_vals[u].y),
                                "r"(v_pf_vals[u].z), "r"(v_pf_vals[u].w));
            }
            constexpr int VSC_WORDS = (int)sizeof(typename G::v_sc_tile) / 4;
            if (lane < VSC_WORDS) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(v_sc_pf_addr), "r"(v_sc_pf_val));
            }
        };
        v_prefetch_ldg(0, 0);
        v_commit_sts();
        __threadfence_block();
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(1);

        // Stage constant P scale, when used, and Q scales into TMEM.
        if (leader) {
            if constexpr (!DYNAMIC_P_SCALE) {
                auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
                fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[0]);
            }
            fp4pv_expect_bytes<C>(q_sc_arrived[0], sizeof(typename G::q_sc_tile));
            fp4pv_wait<C>(q_sc_arrived[0], 0);
        }
#ifdef FSK_SF_MARKER
        warpgroup::sync(1);
        fsk_overwrite(slot_tile_at(q_sc_smem, 0), FSK_SF_MARKER == 2);
        warpgroup::sync(1);
#endif
        if constexpr (C::Nb == 64) {
            // The N=64 MMA reads A-scales at (lane m, word m/32); the cp
            // staging only populates that layout for rows < 64 (sfa_atlas).
            // Stage directly: warp g lane t supplies row (32g+t)'s 4 quad
            // bytes, which live at page offset 16t+4g (consumption layout).
            warpgroup::sync(1);
#ifndef FSK_STTM_QSC
            // Completion-waited q_sc staging (race fix: the no-sem cp -> MMA
            // pattern in the production helper relies on timing slack).
            if (leader) {
                #pragma unroll
                for (int c = 0; c < C::QK_SCALE_CHUNKS; ++c) {
                    auto sub = q_sc_tm.template subtile<full_tt_fp8e4m3<16>>(c * 16);
                    auto &sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
                        reinterpret_cast<uint64_t>(&slot_tile_at(q_sc_smem, 0).data[0]) + 16 * 32 * c);
                    if (c + 1 < C::QK_SCALE_CHUNKS) {
                        load_mxnv_scale_async<1>(sub, sm_sub);
                    } else {
                        load_mxnv_scale_async<1>(sub, sm_sub, sc_staged[0]);
                    }
                }
                wait(sc_staged[0], sc_staged_phase);
                sc_staged_phase ^= 1;
            }
#else
            {
                const int g = warpgroup::warpid();
                const int t = warp::laneid();
                const uint8_t *page = reinterpret_cast<const uint8_t *>(
                    &slot_tile_at(q_sc_smem, 0).data[0]);
                #pragma unroll
                for (int c = 0; c < C::QK_SCALE_CHUNKS; ++c) {
#ifdef FSK_ZERO_QKSC
                    const uint32_t w = 0u;
#else
                    const uint32_t w = *reinterpret_cast<const uint32_t *>(
                        page + 512 * c + 16 * t + 4 * g);
#endif
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};}"
                                 :: "r"(q_sc_tm.addr + 16 * c + g), "r"(w));
                }
                kittens::tensor_store_wait();
                tensor_before_thread_sync();
            }
#endif
            warpgroup::sync(1);
        } else if (leader) {
#ifdef FSK_ZERO_QKSC
            fp4pv_stage_q_scale_tmem_cluster<C>(q_sc_tm, qk_sc_zero[0]);
#else
            fp4pv_stage_q_scale_tmem_cluster<C>(q_sc_tm, slot_tile_at(q_sc_smem, 0));
#endif
        }
        if (leader) {
            if constexpr (C::Nb == 64) {
                tensor_after_thread_sync();
            }
            // QK(0).
            fp4pv_expect_bytes<C>(q_arrived[0], sizeof(typename G::q_tile));
            fp4pv_wait<C>(q_arrived[0], 0);
            fp4pv_expect_bytes<C>(k_arrived[0], sizeof(k_ld_t));
            fp4pv_wait<C>(k_arrived[0], 0);
            fp4pv_expect_bytes<C>(k_sc_arrived[0], sizeof(typename G::k_sc_tile));
            fp4pv_wait<C>(k_sc_arrived[0], 0);
#ifdef FSK_SF_MARKER
            fsk_overwrite(slot_tile_at(k_sc_smem, 0), FSK_SF_MARKER == 1);
#endif
            if constexpr (C::Nb == 64) {
                stage_scales_waited(k_sc_tm0, slot_tile_at(k_sc_smem, 0));
                issue_qk_chunked_qsc_tmem_preloaded_ksc<C>(
                    tt_score0, q_smem[0], k_smem[0], q_sc_tm, k_sc_tm0, k_arrived[0]);
            } else {
#ifdef FSK_ZERO_QKSC
            fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
                tt_score0, q_smem[0], k_smem[0], q_sc_tm,
                qk_sc_zero[0], k_sc_tm0, k_arrived[0]);
#else
            fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
                tt_score0, q_smem[0], k_smem[0], q_sc_tm,
                slot_tile_at(k_sc_smem, 0), k_sc_tm0, k_arrived[0]);
#endif
            }
            detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
            if constexpr (C::Nb == 64) {
                // Pair-granular QK: the N=64-idesc issue against the TMA
                // <64, Dqk/2> tile (which carries 128 K rows) writes 128
                // score columns = BOTH adjacent 64-col slots; arm slot 1 too.
                detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[1]);
            }
        }

        // Shift an odd 64-tile's scale bytes into the consumed positions:
        // within every 16-byte group of the k_sc slot, byte o <- byte o+8.
        // (In-place is safe: reads [8,16) writes [0,8) per group.) Warp 0 only.
        auto ksc_repack_odd = [&](int slot) {
            if constexpr (C::Nb == 64) {
                if (warpgroup::warpid() == 0) {
                    auto *bytes = reinterpret_cast<uint64_t *>(&slot_tile_at(k_sc_smem, slot).data[0]);
                    constexpr int GROUPS = (int)sizeof(typename G::k_sc_tile) / 16;
                    for (int gidx = warp::laneid(); gidx < GROUPS; gidx += WARP_THREADS) {
                        bytes[gidx * 2] = bytes[gidx * 2 + 1];
                    }
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                }
            }
        };
        float row_max = base_types::constants<float>::neg_infty();
        float row_sum = 0.0f;
        // k_arrived alternates [TMA arrive -> parity 1, QK drain commit -> 0]
        // strictly, so QK waits parity 0 and the refill waits parity 1, both
        // constant. k_sc_arrived only ever gets TMA arrives: per-slot mask.
        int k_sc_phase_mask = 0;
        float2 scores_reg[C::Nb / 2];

        for (int i = 0; i < iters; ++i) {
            const int buf = i & 1;
            // Refill K slot for tile i+2 (the slot is free: QK(i) consumed it and
            // the qk helper committed k_arrived[slot] reuse via its finished sem -
            // we reuse k_arrived as the finished sem, so the TMA refill must wait
            // its drain parity before reissuing into the same slot).
            {
                constexpr int TPB = (C::Nb == 64) ? 2 : 1;  // tiles per K block
                const int nblk = (iters + TPB - 1) / TPB;
                // nb64: refill on ODD iterations (after the even-i issue read
                // the slot) - same-slot TMA-vs-issue clobber is impossible.
                const int p = (C::Nb == 64) ? ((i + 3) / 2) : (i + 2);
                if (leader && (C::Nb == 64 ? ((i & 1) == 1) : true) && p < nblk) {
                    const int slot = p & 1;
                    fp4pv_wait<C>(k_arrived[slot], 1);
                    const coord<k_ld_t> k_coord = {b, h, p, 0};
                    const coord<typename G::k_sc_tile> k_sc_coord = {b, p, h * C::QK_SCALE_CHUNKS, 0};
                    fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(k_smem[slot], g.k, k_coord, k_arrived[slot], (uint16_t)1, 0);
                    fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(k_sc_smem, slot), g.k_sc, k_sc_coord, k_sc_arrived[slot], (uint16_t)1, 0);
                    k_sc_phase_mask ^= (1 << slot);
                }
            }
            // V(i+1) LDG prefetch: latency hides under this whole tile.
            if (i + 1 < iters) {
                v_prefetch_ldg(i + 1, (i + 1) & 1);
            }
            // QK(i+1) into the other score slot (its LDTM finished last iter).
            // nb64 pair flow: issue pair i/2 at even i>0 - both tiles of the
            // previous pair are consumed by then (LDTM at i-2, i-1), so the
            // score0||score1 overwrite is safe. No lookahead (QK latency
            // exposed per pair - correctness first, perf pass later).
            if ((C::Nb != 64 && i + 1 < iters) ||
                (C::Nb == 64 && (i & 1) == 0 && i > 0)) {
                const int slot = (C::Nb == 64) ? ((i / 2) & 1) : ((i + 1) & 1);
                if (leader) {
                    fp4pv_expect_bytes<C>(k_arrived[slot], sizeof(k_ld_t));
                    fp4pv_wait<C>(k_arrived[slot], 0);
                    fp4pv_expect_bytes<C>(k_sc_arrived[slot], sizeof(typename G::k_sc_tile));
                    fp4pv_wait<C>(k_sc_arrived[slot], (k_sc_phase_mask >> slot) & 1);
                }
                // (The odd-tile repack built on the retracted session-21
                // truncation theory was the residual bug: N=64 consumes all
                // rows' positions naturally - sfa_k192 - so shifting them
                // corrupted odd tiles. Removed; depth=i/2 stages the block
                // page both halves read directly.)
                if (leader) {
#ifdef FSK_SF_MARKER
                    {
                        auto &tile = slot_tile_at(k_sc_smem, slot);
                        auto *bytes = reinterpret_cast<uint8_t *>(&tile.data[0]);
                        for (int ii = 0; ii < 1536; ++ii) {
                            bytes[ii] = (FSK_SF_MARKER == 1) ? (uint8_t)(0x70 + ((uint32_t)(ii * 2654435761u) >> 28) % 15) : (uint8_t)0x38;
                        }
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    }
#endif
                    if constexpr (C::Nb == 64) {
                        // one issue per pair: fills score0||score1 (128 cols)
                        if ((i / 2) & 1) {
                            stage_scales_waited(k_sc_tm1, slot_tile_at(k_sc_smem, slot));
                            issue_qk_chunked_qsc_tmem_preloaded_ksc<C>(
                                tt_score0, q_smem[0], k_smem[slot], q_sc_tm, k_sc_tm1, k_arrived[slot]);
                        } else {
                            stage_scales_waited(k_sc_tm0, slot_tile_at(k_sc_smem, slot));
                            issue_qk_chunked_qsc_tmem_preloaded_ksc<C>(
                                tt_score0, q_smem[0], k_smem[slot], q_sc_tm, k_sc_tm0, k_arrived[slot]);
                        }
                    } else {
#ifdef FSK_ZERO_QKSC
                    auto &ksc_src = qk_sc_zero[0];
#else
                    auto &ksc_src = slot_tile_at(k_sc_smem, slot);
#endif
                    if ((i + 1) & 1) {
                        fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
                            tt_score1, q_smem[0], k_smem[slot], q_sc_tm,
                            ksc_src, k_sc_tm1, k_arrived[slot]);
                    } else {
                        fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
                            tt_score0, q_smem[0], k_smem[slot], q_sc_tm,
                            ksc_src, k_sc_tm0, k_arrived[slot]);
                    }
                    }
                    if constexpr (C::Nb == 64) {
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[1]);
                    } else {
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[(i + 1) & 1]);
                    }
                }
            }
            // Scores for tile i.
            wait(scores_arrived[i & 1], (i >> 1) & 1);
            const uint32_t score_addr = ((i & 1) ? tt_score1 : tt_score0).addr;
            #pragma unroll
            for (int chunk = 0; chunk < SCORE_CHUNKS; ++chunk) {
                float2 *c = &scores_reg[chunk * 16];
                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                    : "=f"(c[0].x), "=f"(c[0].y), "=f"(c[1].x), "=f"(c[1].y), "=f"(c[2].x), "=f"(c[2].y), "=f"(c[3].x), "=f"(c[3].y),
                      "=f"(c[4].x), "=f"(c[4].y), "=f"(c[5].x), "=f"(c[5].y), "=f"(c[6].x), "=f"(c[6].y), "=f"(c[7].x), "=f"(c[7].y),
                      "=f"(c[8].x), "=f"(c[8].y), "=f"(c[9].x), "=f"(c[9].y), "=f"(c[10].x), "=f"(c[10].y), "=f"(c[11].x), "=f"(c[11].y),
                      "=f"(c[12].x), "=f"(c[12].y), "=f"(c[13].x), "=f"(c[13].y), "=f"(c[14].x), "=f"(c[14].y), "=f"(c[15].x), "=f"(c[15].y)
                    : "r"(score_addr + chunk * 32));
            }
            kittens::tensor_load_wait();
            warpgroup::sync(1);
#ifdef FSK_SCORE_STG
            // Trusted readout: rows 45 and 100, tile 0, through a device
            // global (plain STG at probe time; printed from reloaded memory
            // at task end - no vararg marshalling of live registers).
            if (bid == tiles_m - 1 && i == 0 && (lane == 45 || lane == 100)) {
                const int base = (lane == 45) ? 0 : 128;
                #pragma unroll
                for (int j = 0; j < C::Nb / 2 && 2 * j + 1 < 128; ++j) {
                    fp4pv_score_dump[base + 2 * j] = scores_reg[j].x;
                    fp4pv_score_dump[base + 2 * j + 1] = scores_reg[j].y;
                }
                __threadfence();
            }
#endif
            // Causal mask.
            const int tile_col_base = i * C::Nb;
            if (tile_col_base + C::Nb - 1 > query_col) {
                #pragma unroll
                for (int j = 0; j < C::Nb / 2; ++j) {
                    const int col0 = tile_col_base + j * 2;
                    if (col0 > query_col) scores_reg[j].x = base_types::constants<float>::neg_infty();
                    if (col0 + 1 > query_col) scores_reg[j].y = base_types::constants<float>::neg_infty();
                }
            }
            // Row max and correction.
            float tile_max = base_types::constants<float>::neg_infty();
            #pragma unroll
            for (int j = 0; j < C::Nb / 2; ++j) {
                tile_max = fmaxf(tile_max, fmaxf(scores_reg[j].x, scores_reg[j].y));
            }
            const float row_max_old = row_max;
            row_max = fmaxf(row_max, tile_max);
            float corr = 1.0f;
            if (i > 0 && !ZERO_SCALE) {
                corr = softmax_rescale_factor_strict(row_max, row_max_old, SCALE_LOG2);
            }
            // PV(i-1) drain: one wait covers rescale safety, payload slot i&1
            // reuse (PV(i-2) <= this drain), V slot reuse, and epilogue order.
            if (i > 0) {
                wait(pv_drained[0], (i - 1) & 1);
            }
            const bool lane_needs_rescale = row_active && (corr < 0.999f);
            if (i > 0 && __any_sync(0xFFFFFFFF, lane_needs_rescale)) {
                rescale_tt_output_inplace_direct_streaming<C>(tt_output, corr, lane_needs_rescale);
            }
            warpgroup::sync(1);
            // Quant + payload store + sums.
            float2 tile_sum0 = {0.0f, 0.0f}, tile_sum1 = {0.0f, 0.0f};
            {
                const float neg_max_scaled = ZERO_SCALE
                    ? (DYNAMIC_P_SCALE ? 0.0f : FP4PV_MXFP4_ONLINE_FIXED_P_LOG2_COEFF)
                    : (row_max * (-SCALE_LOG2) +
                       (DYNAMIC_P_SCALE ? 0.0f : FP4PV_MXFP4_ONLINE_FIXED_P_LOG2_COEFF));
                const float2 neg2 = {neg_max_scaled, neg_max_scaled};
                const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                uint32_t packed_p_scales = 0u;
                #pragma unroll
                for (int qid = 0; qid < C::Nb / 32; ++qid) {
                    uint32_t words[4];
                    float p_block_amax = 0.0f;
                    #pragma unroll
                    for (int pack = 0; pack < 4; ++pack) {
                        #pragma unroll
                        for (int kk = 0; kk < 4; ++kk) {
                            const int jj = pack * 4 + kk;
                            const int si = qid * 16 + jj;
                            if (!ZERO_SCALE) {
                                scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg2);
                            } else {
                                scores_reg[si].x = isfinite(scores_reg[si].x) ? neg_max_scaled : scores_reg[si].x;
                                scores_reg[si].y = isfinite(scores_reg[si].y) ? neg_max_scaled : scores_reg[si].y;
                            }
                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            if constexpr (DYNAMIC_P_SCALE) {
                                p_block_amax = fmaxf(
                                    p_block_amax,
                                    fmaxf(fabsf(scores_reg[si].x), fabsf(scores_reg[si].y)));
                            }
                            if ((jj & 1) == 0) tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                            else tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                        }
                        if constexpr (!DYNAMIC_P_SCALE) {
                            words[pack] = fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(
                                &scores_reg[qid * 16 + pack * 4]);
                        }
#ifdef FSK_PAYLOAD_DUMP
                        if (bid == tiles_m - 1 && i <= 1 && (lane == 45 || lane == 100)) {
                            fp4pv_score_dump[(lane == 100 ? 32 : 0) + i * 16 + qid * 4 + pack] =
                                __uint_as_float(words[pack]);
                        }
#endif
                    }
                    if constexpr (DYNAMIC_P_SCALE) {
                        uint8_t e8m0 = 0;
                        fp4pv_quantize_mxfp4_scores_group_with_amax(
                            &scores_reg[qid * 16], p_block_amax, g.p_quant_mode, words, e8m0);
                        packed_p_scales |= static_cast<uint32_t>(e8m0) << (qid * 8);
                        if constexpr (!DIRECT_P_SCALE_TMEM) {
                            if (row_active) {
                                fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_stage[0], lane, qid, e8m0);
                            }
                        }
                    }
                    if (row_active) {
                        fp4pv_store_quantized_scores_group_mxfp4_selected<
                            false, false, false, false, false, false>(
                            p_fp4_stage[buf], words, lane, qid);
                    }
                }
                if constexpr (DYNAMIC_P_SCALE && DIRECT_P_SCALE_TMEM) {
                    if (row_active) {
                        p_dyn_scale_words[lane] = packed_p_scales;
                    }
                }
            }
            // V(i+1) STS commit (LDGs issued at loop top; slot freed by the
            // drain wait above).
            if (i + 1 < iters) {
                v_commit_sts();
            }
            __threadfence_block();
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
            // PV(i) into our own accumulator.
            if constexpr (DYNAMIC_P_SCALE && DIRECT_P_SCALE_TMEM) {
                const int source_row = warp::laneid();
                const uint32_t row_tile0 = p_dyn_scale_words[source_row + 0 * 32];
                const uint32_t row_tile1 = p_dyn_scale_words[source_row + 1 * 32];
                const uint32_t row_tile2 = p_dyn_scale_words[source_row + 2 * 32];
                const uint32_t row_tile3 = p_dyn_scale_words[source_row + 3 * 32];
                fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
                    p_sc_tm,
                    warpgroup::warpid() * 32,
                    row_tile0,
                    row_tile1,
                    row_tile2,
                    row_tile3);
                fp4pv_tmem_store_wait();
                warpgroup::sync(1);
            }
            if (leader) {
                if constexpr (DYNAMIC_P_SCALE && !DIRECT_P_SCALE_TMEM) {
                    auto p_sc_tm_sub = p_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
                    load_mxnv_scale_async<1>(p_sc_tm_sub, p_sc_stage[0], sc_staged[0]);
                    wait(sc_staged[0], sc_staged_phase);
                    sc_staged_phase ^= 1;
                }
                auto v_sc_tm_sub = v_sc_tm.template subtile<full_tt_fp8e8m0<16>>(0);
                fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm_sub, v_sc_smem[buf]);
                if (i == 0) {
                    fp4pv_mm_p_stage_ABt<C>(tt_output, p_fp4_stage[buf], v_fp4_smem[buf], p_sc_tm, v_sc_tm, pv_drained[0]);
                } else {
                    fp4pv_mma_p_stage_ABt<C>(tt_output, p_fp4_stage[buf], v_fp4_smem[buf], p_sc_tm, v_sc_tm, pv_drained[0]);
                }
            }
            tile_sum0 = __fadd2_rn(tile_sum0, tile_sum1);
            row_sum = row_sum * corr + (tile_sum0.x + tile_sum0.y);
        }

        // Epilogue.
        wait(pv_drained[0], (iters - 1) & 1);
        const bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
        float inv_norm;
        asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_norm) : "f"(row_invalid ? 1.0f : row_sum));
        float o_reg[C::Dvo];
        #pragma unroll
        for (int chunk = 0; chunk < OUT_CHUNKS; ++chunk) {
            float2 *c = reinterpret_cast<float2 *>(&o_reg[chunk * 32]);
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                : "=f"(c[0].x), "=f"(c[0].y), "=f"(c[1].x), "=f"(c[1].y), "=f"(c[2].x), "=f"(c[2].y), "=f"(c[3].x), "=f"(c[3].y),
                  "=f"(c[4].x), "=f"(c[4].y), "=f"(c[5].x), "=f"(c[5].y), "=f"(c[6].x), "=f"(c[6].y), "=f"(c[7].x), "=f"(c[7].y),
                  "=f"(c[8].x), "=f"(c[8].y), "=f"(c[9].x), "=f"(c[9].y), "=f"(c[10].x), "=f"(c[10].y), "=f"(c[11].x), "=f"(c[11].y),
                  "=f"(c[12].x), "=f"(c[12].y), "=f"(c[13].x), "=f"(c[13].y), "=f"(c[14].x), "=f"(c[14].y), "=f"(c[15].x), "=f"(c[15].y)
                : "r"(tt_output.addr + chunk * 32));
        }
        kittens::tensor_load_wait();
#ifdef FSK_PAYLOAD_DUMP
        if (bid == tiles_m - 1 && lane == 0 && warpgroup::warpid() == 0) {
            for (int rr = 0; rr < 2; ++rr) {
                for (int grp = 0; grp < 3; ++grp) {
                    printf("PAYW row%d w%d: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                           rr ? 100 : 45, grp * 8,
                           __float_as_uint(fp4pv_score_dump[rr*32+grp*8+0]), __float_as_uint(fp4pv_score_dump[rr*32+grp*8+1]),
                           __float_as_uint(fp4pv_score_dump[rr*32+grp*8+2]), __float_as_uint(fp4pv_score_dump[rr*32+grp*8+3]),
                           __float_as_uint(fp4pv_score_dump[rr*32+grp*8+4]), __float_as_uint(fp4pv_score_dump[rr*32+grp*8+5]),
                           __float_as_uint(fp4pv_score_dump[rr*32+grp*8+6]), __float_as_uint(fp4pv_score_dump[rr*32+grp*8+7]));
                }
            }
        }
#endif
#ifdef FSK_SCORE_STG
        if (bid == tiles_m - 1 && lane == 0 && warpgroup::warpid() == 0) {
            for (int blk45 = 0; blk45 < 2; ++blk45) {
                const int base = blk45 * 128;
                printf("TRUSTED row%d c28-35: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f | c60-67: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f | c96-99: %.1f %.1f %.1f %.1f\n",
                       blk45 ? 100 : 45,
                       fp4pv_score_dump[base+28], fp4pv_score_dump[base+29], fp4pv_score_dump[base+30], fp4pv_score_dump[base+31],
                       fp4pv_score_dump[base+32], fp4pv_score_dump[base+33], fp4pv_score_dump[base+34], fp4pv_score_dump[base+35],
                       fp4pv_score_dump[base+60], fp4pv_score_dump[base+61], fp4pv_score_dump[base+62], fp4pv_score_dump[base+63],
                       fp4pv_score_dump[base+64], fp4pv_score_dump[base+65], fp4pv_score_dump[base+66], fp4pv_score_dump[base+67],
                       fp4pv_score_dump[base+96], fp4pv_score_dump[base+97], fp4pv_score_dump[base+98], fp4pv_score_dump[base+99]);
            }
        }
#endif
        // Production MXFP4 epilogue descale. The fixed-scale probe bakes the
        // fixed P coefficient into P before PV; the dynamic probe carries true
        // probabilities plus per-group E8M0 scales.
        constexpr float MXFP4_ALPHA = DYNAMIC_P_SCALE
            ? (1.0f / 36.0f)
            : (FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF / 36.0f);
        const float output_scale = (MXFP4_ALPHA / FP4PV_MXFP4_ONLINE_P_SCALE) * inv_norm;
        if (row_active) {
            const int s = m_tile * C::Mb + lane;
#if defined(FSK_SCORE_STG) && !defined(FSK_KEEP_EPI)
            if (bid == tiles_m - 1) goto fsk_skip_o_store;
#endif
            #pragma unroll
            for (int d = 0; d < C::Dvo; ++d) {
                g.o[{b, s, h, d}] = __float2bfloat16(o_reg[d] * output_scale);
            }
#if defined(FSK_SCORE_STG) && !defined(FSK_KEEP_EPI)
            fsk_skip_o_store:;
#endif
            if (g.disable_lse_store == 0) {
                constexpr float LN2 = 0.693147180559945f;
                float lse_val;
                if (!row_invalid) {
                    float log2_row_sum;
                    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                    if constexpr (!DYNAMIC_P_SCALE) {
                        log2_row_sum -= FP4PV_MXFP4_ONLINE_FIXED_P_LOG2_COEFF;
                    }
                    lse_val = (row_max * SCALE_LOG2 + log2_row_sum) * LN2;
                } else {
                    lse_val = base_types::constants<float>::neg_infty();
                }
                g.lse[{b, h, 0, s}] = lse_val;
            }
        }
    }
    asm volatile("bar.sync %0, %1;" :: "n"(WG_BAR), "n"(C::NUM_THREADS) : "memory");
}

enum debug_stage_cut_mode : int {
    DEBUG_STAGE_CUT_FULL = 0,
    DEBUG_STAGE_CUT_ISSUE3_ONLY = 1,
    DEBUG_STAGE_CUT_CONSUME3_COPY = 2,
    DEBUG_STAGE_CUT_CONSUME3_ROWMAX = 3,
    DEBUG_STAGE_CUT_CONSUME3_EXP = 4,
    DEBUG_STAGE_CUT_CONSUME3_SCAN = 5,
    DEBUG_STAGE_CUT_PRODUCER3_POST_WAIT = 6,
    DEBUG_STAGE_CUT_PRODUCER3_POST_ISSUE = 7,
    DEBUG_STAGE_CUT_PRODUCER3_POST_COMMIT = 8,
    DEBUG_STAGE_CUT_PRODUCER3_POST_KSC_FINISH = 9,
    DEBUG_STAGE_CUT_CONSUME3_WAIT_ONLY = 10,
    DEBUG_STAGE_CUT_PRODUCER3_SKIP_COMMIT_TAIL = 11,
    DEBUG_STAGE_CUT_PRODUCER3_FRESH_COMMIT_TAIL = 12,
    DEBUG_STAGE_CUT_FRESH3_WAIT_ONLY = 13,
    DEBUG_STAGE_CUT_FRESH3_COPY = 14,
    DEBUG_STAGE_CUT_FRESH3_FULL = 15,
    DEBUG_STAGE_CUT_DUAL_SLOT_FULL = 16,
    DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL = 17,
    DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY = 100,
    DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY = 101,
    DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY = 102,
    DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE = 103,
    DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE = 104,
    DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY = 105,
    DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY = 106,
    DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE = 107,
    DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE = 108,
    DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM = 109,
    DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM = 110,
    DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL = 111,
};

enum debug_q_scale_mode : int {
    DEBUG_Q_SCALE_MODE_TMEM = 0,
    DEBUG_Q_SCALE_MODE_SMEM_CHUNKED = 1,
};

enum debug_tmem_layout_mode : int {
    DEBUG_TMEM_LAYOUT_COMPACT = 0,
    DEBUG_TMEM_LAYOUT_PRODUCTION_LIKE = 1,
};

enum debug_k_scale_depth_mode : int {
    DEBUG_K_SCALE_DEPTH_MODE_TILE128 = 0,
    DEBUG_K_SCALE_DEPTH_MODE_SLAB64 = 1,
};

enum debug_scale_probe_stage_mode : int {
    DEBUG_SCALE_PROBE_STAGE_SOURCE_ONLY = 0,
    DEBUG_SCALE_PROBE_STAGE_Q_LOAD_SYNC = 1,
    DEBUG_SCALE_PROBE_STAGE_Q_RT_LOAD_ONLY = 2,
    DEBUG_SCALE_PROBE_STAGE_Q_COPY_SHARED_ONLY = 3,
    DEBUG_SCALE_PROBE_STAGE_Q_LOAD_COPY = 4,
    DEBUG_SCALE_PROBE_STAGE_K_LOAD_SYNC = 5,
    DEBUG_SCALE_PROBE_STAGE_K_LOAD_COPY = 6,
    DEBUG_SCALE_PROBE_STAGE_FULL = 7,
    DEBUG_SCALE_PROBE_STAGE_Q_DIRECT_DUMP = 8,
    DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP = 9,
};

enum live_dp_debug_mode : int {
    LIVE_DP_DEBUG_FULL = 0,
    LIVE_DP_DEBUG_PROFILE_TC_HANDSHAKE_ZERO_DP = 21,
    LIVE_DP_DEBUG_X8_WAIT = 22,
    LIVE_DP_DEBUG_X8_ALIGNED_PROBE = 23,
    LIVE_DP_DEBUG_X16_WAIT = 24,
};

template <typename G>
__device__ __forceinline__ bool live_dp_mode_consumes_tmem(const G &g) {
    return g.live_dp_debug_mode == LIVE_DP_DEBUG_FULL ||
           g.live_dp_debug_mode == 13 ||
           g.live_dp_debug_mode == 18 ||
           g.live_dp_debug_mode == 19 ||
           g.live_dp_debug_mode == 20 ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_WAIT ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_ALIGNED_PROBE ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X16_WAIT ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_PROFILE_TC_HANDSHAKE_ZERO_DP;
}

template <typename G>
__device__ __forceinline__ bool live_dp_mode_applies_values(const G &g) {
    return g.live_dp_debug_mode == LIVE_DP_DEBUG_FULL ||
           g.live_dp_debug_mode == 13 ||
           g.live_dp_debug_mode == 18 ||
           g.live_dp_debug_mode == 19 ||
           g.live_dp_debug_mode == 20 ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_WAIT ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_ALIGNED_PROBE ||
           g.live_dp_debug_mode == LIVE_DP_DEBUG_X16_WAIT;
}

template <typename G>
__device__ __forceinline__ bool live_dp_mode_uses_scaled_mma_values(const G &g) {
    return g.live_dp_debug_mode != 13;
}

template <typename C>
struct globals_quant_p_from_scores_debug {
    using q_tile = typename globals_fp4pv<C>::q_tile;
    using q_sc_tile = typename globals_fp4pv<C>::q_sc_tile;
    using k_tile = typename globals_fp4pv<C>::k_tile;
    using k_sc_tile = typename globals_fp4pv<C>::k_sc_tile;
    using p_fp4_tile = typename globals_fp4pv<C>::p_fp4_tile;
    using p_sc_tile = typename globals_fp4pv<C>::p_sc_tile;
    using bwd_dq_k_fp4_tile = std::conditional_t<
        C::CLUSTER_SIZE == 1,
        st_fp4e2m1_2<C::Dvo, C::Nb / 2>,
        typename globals_fp4pv<C>::v_fp4_tile>;
    using bwd_dq_sc_tile = st_fp8e8m0<32, 16, false>;
    using q_sc_smem_storage = typename globals_fp4pv<C>::q_sc_smem_storage;
    using k_sc_smem_storage = typename globals_fp4pv<C>::k_sc_smem_storage;
    using k_sc_guard_slot = typename globals_fp4pv<C>::k_sc_guard_slot;
    using p_sc_guard_slot = typename globals_fp4pv<C>::p_sc_guard_slot;
    using live_dp_dout_fp4_tile = st_fp4e2m1_2<C::Mb, C::Dvo / 2>;
    using live_dp_v_row_fp4_tile = st_fp4e2m1_2<C::Nb / 2, C::Dvo / 2>;
    using live_dp_sc_tile = st_fp8e8m0<32, 16, false>;

    using q_gl = typename globals_fp4pv<C>::q_gl;
    using q_sc_gl = typename globals_fp4pv<C>::q_sc_gl;
    using q_sg_gl = typename globals_fp4pv<C>::q_sg_gl;
    using k_gl = typename globals_fp4pv<C>::k_gl;
    using k_sc_gl = typename globals_fp4pv<C>::k_sc_gl;
    using k_sg_gl = typename globals_fp4pv<C>::k_sg_gl;
    using lse_gl = typename globals_fp4pv<C>::lse_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_tile) * C::NUM_SOFTMAXXERS);
        add((int)sizeof(q_sc_smem_storage));
        add((int)sizeof(k_tile) * C::LOAD_STAGES);
        add((int)sizeof(k_sc_smem_storage));
        add((int)sizeof(k_sc_guard_slot));
        add((int)sizeof(fp4pv_p_scan3d));
        add((int)sizeof(st_fp8e4m3<128, 32, false>));
        add((int)sizeof(p_fp4_tile) * 2);
        add((int)sizeof(p_sc_tile) * 2);
        add((int)sizeof(bwd_dq_k_fp4_tile) * 2);
        add((int)sizeof(bwd_dq_sc_tile) * 2);
        add((int)sizeof(bwd_dq_k_fp4_tile) * 2);
        add((int)sizeof(bwd_dq_sc_tile) * 2);
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(bwd_dq_k_fp4_tile));
        add((int)sizeof(p_sc_guard_slot));
        return bytes;
    }

    static constexpr int compute_dynamic_shared_memory_live_dp() {
        int bytes = compute_dynamic_shared_memory();
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(live_dp_dout_fp4_tile));
        add((int)sizeof(live_dp_sc_tile));
        add((int)sizeof(live_dp_v_row_fp4_tile) * 2);
        add((int)sizeof(live_dp_sc_tile) * 2);
        bytes += 4096;
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();
    static constexpr int DYNAMIC_SHARED_MEMORY_LIVE_DP = compute_dynamic_shared_memory_live_dp();

    q_gl q;
    q_sc_gl q_sc;
    q_sg_gl q_sg;
    k_gl k;
    k_sc_gl k_sc;
    k_sg_gl k_sg;
    uint8_t *p_fp4;
    uint8_t *p_sc_prepared;
    uint8_t *p_sc_rowmajor_live;
    uint8_t *p_sc_rowmajor_scan;
    bf16 *p_bf16_live;
    lse_gl lse;
    float *debug_row_max;
    float *debug_row_sum;
    float *debug_tile_amax;
    int selected_row_tile;
    int tile_begin;
    int tiles_to_run;
    int max_k_iters = -1;
    int selected_row_tile_span = 1;
    int compute_k_iter_base = 0;
    int output_full_layout = 0;
    int disable_lse_store = 0;
    int use_external_lse = 0;
    int debug_stage_cut = DEBUG_STAGE_CUT_FULL;
    int debug_q_scale_mode = DEBUG_Q_SCALE_MODE_TMEM;
    int debug_tmem_layout_mode = DEBUG_TMEM_LAYOUT_COMPACT;
    int debug_k_scale_depth_mode = DEBUG_K_SCALE_DEPTH_MODE_TILE128;
    int debug_qk_chunk_mask = (1 << C::QK_SCALE_CHUNKS) - 1;
    float *debug_tile_score_max = nullptr;
    float *debug_tile_exp_sum = nullptr;
    bf16 *debug_scores_bf16_live = nullptr;
    bf16 *debug_scores_bf16_scan_repacked = nullptr;
    bf16 *debug_scores_bf16_tmem_x16 = nullptr;
    uint8_t *debug_q_sc_loaded = nullptr;
    uint8_t *debug_k_sc_loaded = nullptr;
    uint8_t *debug_q_sc_tmem_loaded = nullptr;
    uint8_t *debug_k_sc_tmem_loaded = nullptr;
    bf16 *backward_dp_bf16 = nullptr;
    const uint8_t *backward_dout_row_fp4 = nullptr;
    const uint8_t *backward_dout_row_sc = nullptr;
    const uint8_t *backward_v_row_fp4 = nullptr;
    const uint8_t *backward_v_row_sc = nullptr;
    int live_dp_debug_mode = 0;
    int live_dp_selected_head = -1;
    float *backward_delta = nullptr;
    float backward_softmax_scale = 1.0f;
    int emit_backward_ds = 0;
    uint8_t *backward_ds_fp4_row = nullptr;
    uint8_t *backward_ds_sc_row = nullptr;
    uint8_t *backward_ds_fp4_col = nullptr;
    uint8_t *backward_ds_sc_col = nullptr;
    int emit_backward_ds_row_mxfp4 = 0;
    int emit_backward_ds_col_mxfp4 = 0;
    int backward_ds_row_skip_future_zero = 0;
    int backward_ds_col_skip_future_zero = 0;
    int backward_skip_global_ds_row_store = 0;
    int backward_ds_quant_mode = 0;
    uint32_t *backward_fused_stage_marker = nullptr;
    int backward_fused_stage_transpose = 0;
    const uint8_t *backward_dq_k_col_fp4 = nullptr;
    const uint8_t *backward_dq_k_col_sc = nullptr;
    const uint8_t *backward_pair_k_col_fp4 = nullptr;
    const uint8_t *backward_pair_k_col_sc = nullptr;
    bf16 *backward_dq_pad = nullptr;
    bf16 *backward_dq_bshd = nullptr;
    bf16 *backward_pair_dq_pad = nullptr;
    int backward_dq_half = 0;

    __host__ __inline__ int total_bids() {
        const int head_work = live_dp_selected_head >= 0 ? 1 : q.depth();
        if (selected_row_tile >= 0) {
            return q.batch() * head_work * C::CLUSTER_SIZE;
        }
        const int tiles_m = q.rows() / C::Mb;
        const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
        const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
        return q.batch() * head_work * num_block * C::CLUSTER_SIZE;
    }
    __host__ __inline__ dim3 grid() {
        const int total_bids = this->total_bids();
        constexpr int PERSISTENT_GRID_CAP = fp4pv_persistent_grid_cap<C>();
        int grid_x = total_bids < PERSISTENT_GRID_CAP ? total_bids : PERSISTENT_GRID_CAP;
        if (grid_x < C::CLUSTER_SIZE) {
            grid_x = C::CLUSTER_SIZE;
        } else {
            grid_x = ((grid_x + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        }
        return dim3(grid_x);
    }
    __host__ __inline__ dim3 grid_exact() {
        int grid_x = this->total_bids();
        if (grid_x < C::CLUSTER_SIZE) {
            grid_x = C::CLUSTER_SIZE;
        }
        return dim3(grid_x);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
    __host__ __inline__ int dynamic_shared_memory_live_dp() { return DYNAMIC_SHARED_MEMORY_LIVE_DP; }
};

template <typename C, typename BTile>
__device__ inline void load_bwd_dq_k_col_fp4_tile_from_ptr(
    BTile &dst,
    const globals_quant_p_from_scores_debug<C> &g,
    const uint8_t *src_fp4,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int dq_half_override = -1
) {
    if (src_fp4 == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int BYTES_PER_STORE = 16;
    static_assert(BTile::cols >= C::Nb / 2, "dQ K-col tile must hold one packed K128 slice");
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "packed K dimension must be vector-loadable");
    static_assert(BTile::swizzle_bytes >= BYTES_PER_STORE, "dQ K-col tile swizzle must support vector stores");
    const int packed_seqlen = g.q.rows() / 2;
    const int rows_per_cta = C::Dvo / C::CLUSTER_SIZE;
    const int dq_half = dq_half_override >= 0 ? dq_half_override : g.backward_dq_half;
    const int row_base = dq_half * C::Dvo + cta_rank * rows_per_cta;
    const int col_base = k_iter * (C::Nb / 2);
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    const int total_stores = rows_per_cta * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < total_stores; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear - row * STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &src_fp4[((((b * g.q.depth() + h) * DQ_PAD) + row_base + row) * packed_seqlen) + col_base + col]);
        const uint32_t smem_addr = __cvta_generic_to_shared(dst.idx(&dst.data[0], {row, col}));
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(smem_addr),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BTile>
__device__ inline void load_bwd_dq_k_col_fp4_tile(
    BTile &dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int dq_half_override = -1
) {
    load_bwd_dq_k_col_fp4_tile_from_ptr<C>(
        dst, g, g.backward_dq_k_col_fp4, b, h, k_iter, cta_rank, dq_half_override);
}

template <typename C, typename BTile>
__device__ inline void load_bwd_dq_k_col_fp4_tail64_tile(
    BTile &dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int k_iter,
    int cta_rank
) {
    if (g.backward_dq_k_col_fp4 == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int BYTES_PER_STORE = 16;
    constexpr int TAIL_HALF = 1;
    constexpr int TAIL_COLS = C::Dvo / 2;
    constexpr int ROWS_PER_CTA = TAIL_COLS / C::CLUSTER_SIZE;
    static_assert(C::CLUSTER_SIZE == 2, "dQ tail64 loader is only for the cluster-size-2 production path");
    static_assert(C::Dvo == 128 && C::Dqk == 192, "dQ tail64 loader assumes a 64-column upper tail");
    static_assert(BTile::rows >= ROWS_PER_CTA, "dQ tail64 tile must hold the local output rows");
    static_assert(BTile::cols >= C::Nb / 2, "dQ tail64 K-col tile must hold one packed K128 slice");
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "packed K dimension must be vector-loadable");
    static_assert(BTile::swizzle_bytes >= BYTES_PER_STORE, "dQ tail64 K-col tile swizzle must support vector stores");
    const int packed_seqlen = g.q.rows() / 2;
    const int row_base = TAIL_HALF * C::Dvo + cta_rank * ROWS_PER_CTA;
    const int col_base = k_iter * (C::Nb / 2);
    const uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = ROWS_PER_CTA * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear - row * STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.backward_dq_k_col_fp4[((((b * g.q.depth() + h) * DQ_PAD) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(BTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BScTile>
__device__ inline void load_bwd_dq_k_col_sc_tile_from_ptr(
    BScTile &dst,
    const globals_quant_p_from_scores_debug<C> &g,
    const uint8_t *src_sc,
    int b,
    int h,
    int k_iter,
    int dq_half_override = -1
) {
    if (src_sc == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int BYTES_PER_STORE = 16;
    static_assert(sizeof(BScTile) == 32 * 16, "dQ K-col MXFP4 scale tile must be one 32x16 E8M0 tile");
    static_assert(sizeof(BScTile) % BYTES_PER_STORE == 0, "scale tile must be vector-loadable");
    const int scale_tiles_k = g.q.rows() / C::Nb;
    const int scale_n_tile = dq_half_override >= 0 ? dq_half_override : g.backward_dq_half;
    const int base_offset =
        ((((b * g.q.depth() + h) * (DQ_PAD / C::Dvo) + scale_n_tile) * scale_tiles_k + k_iter) * 32 * 16);
    const uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int TOTAL_STORES = (32 * 16) / BYTES_PER_STORE;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int byte_offset = linear * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(&src_sc[base_offset + byte_offset]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(smem_base + byte_offset),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename BScTile>
__device__ inline void load_bwd_dq_k_col_sc_tile(
    BScTile &dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int k_iter,
    int dq_half_override = -1
) {
    load_bwd_dq_k_col_sc_tile_from_ptr<C>(
        dst, g, g.backward_dq_k_col_sc, b, h, k_iter, dq_half_override);
}

template <typename C, int DQ_HALF_OVERRIDE = -1, typename OutputTT>
__device__ inline void store_bwd_dq_pad_tmem_tile(
    OutputTT &tt_output,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile
) {
    if (g.backward_dq_pad == nullptr && g.backward_dq_bshd == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int DQ_OUT = 192;
    const int row = warpgroup::laneid();
    const int global_row = m_tile * C::Mb + row;
    const int dq_half = DQ_HALF_OVERRIDE >= 0 ? DQ_HALF_OVERRIDE : g.backward_dq_half;
    if (global_row >= g.q.rows()) {
        return;
    }
    constexpr int static_tail_cols = DQ_OUT - DQ_HALF_OVERRIDE * C::Dvo;
    constexpr int static_valid_cols =
        static_tail_cols <= 0 ? 0 : (static_tail_cols < C::Dvo ? static_tail_cols : C::Dvo);
    const int valid_cols =
        DQ_HALF_OVERRIDE >= 0 ? static_valid_cols : max(0, min(C::Dvo, DQ_OUT - dq_half * C::Dvo));
    #pragma unroll
    for (int col = 0; col < valid_cols; col += 16) {
        float2 out_reg[8];
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(out_reg[0].x), "=f"(out_reg[0].y), "=f"(out_reg[1].x), "=f"(out_reg[1].y),
              "=f"(out_reg[2].x), "=f"(out_reg[2].y), "=f"(out_reg[3].x), "=f"(out_reg[3].y),
              "=f"(out_reg[4].x), "=f"(out_reg[4].y), "=f"(out_reg[5].x), "=f"(out_reg[5].y),
              "=f"(out_reg[6].x), "=f"(out_reg[6].y), "=f"(out_reg[7].x), "=f"(out_reg[7].y)
            : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
        tensor_load_wait();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            const int out_col = dq_half * C::Dvo + col + ii * 2;
            const bf16_2 packed = __float22bfloat162_rn(out_reg[ii]);
            if (g.backward_dq_bshd != nullptr) {
                const int64_t offset =
                    (((static_cast<int64_t>(b) * g.q.rows() + global_row) * g.q.depth() + h) * DQ_OUT) + out_col;
                *reinterpret_cast<bf16_2 *>(&g.backward_dq_bshd[offset]) = packed;
            } else {
                const int64_t offset =
                    (((static_cast<int64_t>(b) * g.q.depth() + h) * g.q.rows() + global_row) * DQ_PAD) + out_col;
                *reinterpret_cast<bf16_2 *>(&g.backward_dq_pad[offset]) = packed;
            }
        }
    }
}

template <typename C, int DQ_HALF_OVERRIDE = -1, typename OutputTT>
__device__ inline void store_bwd_pair_dq_pad_tmem_tile(
    OutputTT &tt_output,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile
) {
    if (g.backward_pair_dq_pad == nullptr) {
        return;
    }
    constexpr int DQ_PAD = 256;
    constexpr int DQ_OUT = 192;
    const int row = warpgroup::laneid();
    const int global_row = m_tile * C::Mb + row;
    const int dq_half = DQ_HALF_OVERRIDE >= 0 ? DQ_HALF_OVERRIDE : g.backward_dq_half;
    if (global_row >= g.q.rows()) {
        return;
    }
    constexpr int static_tail_cols = DQ_OUT - DQ_HALF_OVERRIDE * C::Dvo;
    constexpr int static_valid_cols =
        static_tail_cols <= 0 ? 0 : (static_tail_cols < C::Dvo ? static_tail_cols : C::Dvo);
    const int valid_cols =
        DQ_HALF_OVERRIDE >= 0 ? static_valid_cols : max(0, min(C::Dvo, DQ_OUT - dq_half * C::Dvo));
    #pragma unroll
    for (int col = 0; col < valid_cols; col += 16) {
        float2 out_reg[8];
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(out_reg[0].x), "=f"(out_reg[0].y), "=f"(out_reg[1].x), "=f"(out_reg[1].y),
              "=f"(out_reg[2].x), "=f"(out_reg[2].y), "=f"(out_reg[3].x), "=f"(out_reg[3].y),
              "=f"(out_reg[4].x), "=f"(out_reg[4].y), "=f"(out_reg[5].x), "=f"(out_reg[5].y),
              "=f"(out_reg[6].x), "=f"(out_reg[6].y), "=f"(out_reg[7].x), "=f"(out_reg[7].y)
            : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
        tensor_load_wait();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            const int out_col = dq_half * C::Dvo + col + ii * 2;
            const bf16_2 packed = __float22bfloat162_rn(out_reg[ii]);
            const int64_t offset =
                (((static_cast<int64_t>(b) * g.q.depth() + h) * g.q.rows() + global_row) * DQ_PAD) + out_col;
            *reinterpret_cast<bf16_2 *>(&g.backward_pair_dq_pad[offset]) = packed;
        }
    }
}

template <typename C>
struct globals_loaded_qk_scale_chunk_debug {
    using q_sc_tile = typename globals_fp4pv<C>::q_sc_tile;
    using k_sc_tile = typename globals_fp4pv<C>::k_sc_tile;
    using q_sc_smem_storage = typename globals_fp4pv<C>::q_sc_smem_storage;
    using k_sc_smem_storage = typename globals_fp4pv<C>::k_sc_smem_storage;
    using k_sc_guard_slot = typename globals_fp4pv<C>::k_sc_guard_slot;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_sc_smem_storage));
        add((int)sizeof(k_sc_smem_storage));
        add((int)sizeof(k_sc_guard_slot));
        add(4 * (int)sizeof(st_fp8e4m3<32, 32, false>));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *q_sc_tile_bytes;
    const uint8_t *k_sc_tile_bytes;
    uint8_t *q_sc_source_chunk;
    uint8_t *k_sc_source_chunk;
    bf16 *q_sc_tmem_chunk;
    bf16 *k_sc_tmem_chunk;
    uint32_t *q_sc_tmem_raw_words;
    uint32_t *k_sc_tmem_raw_words;
    int chunk_idx;
    int stage_mode = DEBUG_SCALE_PROBE_STAGE_FULL;

    __host__ __inline__ dim3 grid() { return dim3(C::CLUSTER_SIZE); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
__device__ inline bool debug_tile_selected(const globals_quant_p_from_scores_debug<C> &g, int k_iter) {
    return k_iter >= g.tile_begin && k_iter < (g.tile_begin + g.tiles_to_run);
}

template <typename C>
__device__ inline int debug_window_k_iter(const globals_quant_p_from_scores_debug<C> &g, int k_iter) {
    return k_iter - g.tile_begin;
}

template <typename C>
__device__ inline int debug_selected_row_tile_span(const globals_quant_p_from_scores_debug<C> &g) {
    if (g.selected_row_tile < 0) {
        return g.q.rows() / C::Mb;
    }
    const int tiles_m = g.q.rows() / C::Mb;
    return min(max(g.selected_row_tile_span, 1), tiles_m - g.selected_row_tile);
}

template <typename C>
__device__ inline bool debug_output_row_selected(const globals_quant_p_from_scores_debug<C> &g, int global_row) {
    if (g.selected_row_tile < 0) {
        return true;
    }
    const int row_base = g.selected_row_tile * C::Mb;
    const int row_limit = row_base + debug_selected_row_tile_span<C>(g) * C::Mb;
    return global_row >= row_base && global_row < row_limit;
}

template <typename C>
__device__ inline int debug_output_rows(const globals_quant_p_from_scores_debug<C> &g) {
    if (g.output_full_layout) {
        return g.q.rows();
    }
    return g.selected_row_tile < 0 ? g.q.rows() : debug_selected_row_tile_span<C>(g) * C::Mb;
}

template <typename C>
__device__ inline int debug_output_row_index(const globals_quant_p_from_scores_debug<C> &g, int global_row) {
    if (g.output_full_layout) {
        return global_row;
    }
    return g.selected_row_tile < 0 ? global_row : (global_row - g.selected_row_tile * C::Mb);
}

template <typename C>
__device__ inline bool debug_output_m_tile_selected(const globals_quant_p_from_scores_debug<C> &g, int m_tile) {
    return g.selected_row_tile < 0 ||
           (m_tile >= g.selected_row_tile && m_tile < g.selected_row_tile + debug_selected_row_tile_span<C>(g));
}

template <typename C>
__device__ inline bool debug_use_q_scale_tmem(const globals_quant_p_from_scores_debug<C> &g) {
    return g.debug_q_scale_mode == DEBUG_Q_SCALE_MODE_TMEM;
}

template <typename C>
__device__ inline bool debug_qk_chunk_enabled(const globals_quant_p_from_scores_debug<C> &g, int chunk) {
    return ((g.debug_qk_chunk_mask >> chunk) & 1) != 0;
}

template <typename C>
__device__ inline int debug_first_enabled_qk_chunk(const globals_quant_p_from_scores_debug<C> &g) {
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        if (debug_qk_chunk_enabled<C>(g, chunk)) {
            return chunk;
        }
    }
    return 0;
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTile, typename KScTile, typename QScTT, typename KScTT>
__device__ inline void issue_qk_chunked_debug_masked(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    const QScTile &q_sc_smem,
    const KScTile &k_sc_smem,
    QScTT &q_sc_tm,
    KScTT &k_sc_tm,
    semaphore &finished,
    const globals_quant_p_from_scores_debug<C> &g
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    bool first_issue = true;
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        if (!debug_qk_chunk_enabled<C>(g, chunk)) {
            continue;
        }
        fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, chunk);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (first_issue) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
            first_issue = false;
        } else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTTFull, typename KScTile, typename KScTT>
__device__ inline void issue_qk_chunked_qsc_tmem_debug_masked(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    QScTTFull &q_sc_tm_full,
    const KScTile &k_sc_smem,
    KScTT &k_sc_tm,
    semaphore &finished,
    const globals_quant_p_from_scores_debug<C> &g
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTTFull::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    bool first_issue = true;
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        if (!debug_qk_chunk_enabled<C>(g, chunk)) {
            continue;
        }
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (first_issue) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
            first_issue = false;
        } else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTile, typename KScTile, typename QScTT, typename KScTT>
__device__ inline void fp4pv_issue_qk_chunked_cluster(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    const QScTile &q_sc_smem,
    const KScTile &k_sc_smem,
    QScTT &q_sc_tm,
    KScTT &k_sc_tm,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; chunk++) {
        fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, chunk);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (chunk == 0) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
        else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTTFull, typename KScTile, typename KScTT>
__device__ inline void fp4pv_issue_qk_chunked_qsc_tmem_cluster(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    QScTTFull &q_sc_tm_full,
    const KScTile &k_sc_smem,
    KScTT &k_sc_tm,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTTFull::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; chunk++) {
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
        if (chunk == 0) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
        else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C, typename KScTTFull, typename KScTile>
__device__ inline void stage_k_scale_tmem_full(KScTTFull &k_sc_tm_full, const KScTile &k_sc_smem) {
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        auto k_sc_tm = k_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        fp4pv_load_k_scale_chunk_cluster<C>(k_sc_tm, k_sc_smem, chunk);
    }
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTTFull, typename KScTTFull>
__device__ inline void issue_qk_chunked_qsc_tmem_preloaded_ksc(
    ScoresTT &scores,
    const QTile &q,
    const KTile &k,
    QScTTFull &q_sc_tm_full,
    KScTTFull &k_sc_tm_full,
    semaphore &finished
) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTTFull::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        auto k_sc_tm = k_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        if (chunk == 0) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 0, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        } else {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, C::CLUSTER_SIZE, 16>(
                scores.addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                q_sc_tm.addr, k_sc_tm.addr, IDESC);
        }
    }
    detail::tcgen05::commit<C::CLUSTER_SIZE>(finished);
}

template <typename C>
__device__ inline int debug_physical_k_iter(
    const globals_quant_p_from_scores_debug<C> &g,
    int local_k_iter
) {
    return g.compute_k_iter_base + local_k_iter;
}

template <typename C>
__device__ inline int debug_k_sc_depth_index(
    const globals_quant_p_from_scores_debug<C> &g,
    int idx,
    int cta_rank,
    int tiles_m
) {
    const int physical_idx = debug_physical_k_iter<C>(g, idx);
    if (g.debug_k_scale_depth_mode == DEBUG_K_SCALE_DEPTH_MODE_SLAB64) {
        const int k_sc_depth = g.k_sc.depth();
        return min(physical_idx * C::CLUSTER_SIZE + cta_rank, k_sc_depth - 1);
    }
    return min(physical_idx * C::CLUSTER_SIZE + cta_rank, tiles_m - 1);
}

template <typename C>
__device__ inline int debug_output_m_tiles(const globals_quant_p_from_scores_debug<C> &g) {
    if (g.output_full_layout) {
        return g.q.rows() / C::Mb;
    }
    return g.selected_row_tile < 0 ? (g.q.rows() / C::Mb) : debug_selected_row_tile_span<C>(g);
}

template <typename C>
__device__ inline int debug_output_m_tile_index(const globals_quant_p_from_scores_debug<C> &g, int m_tile) {
    if (g.output_full_layout) {
        return m_tile;
    }
    return g.selected_row_tile < 0 ? m_tile : (m_tile - g.selected_row_tile);
}

template <typename C>
__device__ inline int debug_executed_k_iters(
    const globals_quant_p_from_scores_debug<C> &g,
    int full_iters_per_task
) {
    int executed;
    if (g.max_k_iters < 0) {
        executed = full_iters_per_task;
    } else {
        const int remaining_iters = full_iters_per_task - g.compute_k_iter_base;
        if (remaining_iters <= 0) {
            return 0;
        }
        executed = min(remaining_iters, g.max_k_iters);
    }
    if (g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM) {
        executed = min(executed, 2);
    }
    return executed;
}

template <typename C>
__device__ inline int debug_valid_causal_k_iters_for_row(
    const globals_quant_p_from_scores_debug<C> &g,
    int executed_k_iters,
    int m_tile
) {
    const int remaining_valid = m_tile + 1 - g.compute_k_iter_base;
    if (remaining_valid <= 0) {
        return 0;
    }
    return min(executed_k_iters, remaining_valid);
}

template <typename ScaleTile>
__device__ inline void copy_scale_tile_bytes_to_shared(ScaleTile &dst, const uint8_t *src) {
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int TILE_BYTES = sizeof(ScaleTile);
    static_assert(TILE_BYTES % WORD_BYTES == 0, "scale tile byte size must be word aligned");
    uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(&dst.data[0]);
    for (int linear = threadIdx.x * WORD_BYTES; linear < TILE_BYTES; linear += blockDim.x * WORD_BYTES) {
        uint32_t value = *reinterpret_cast<const uint32_t *>(src + linear);
        *reinterpret_cast<uint32_t *>(dst_bytes + linear) = value;
    }
}

template <typename ScaleTile>
__device__ inline void store_scale_chunk_32x16_to_global(const ScaleTile &tile, uint8_t *dst, int chunk_idx) {
    if (dst == nullptr) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int CHUNK_BYTES = 32 * 16;
    constexpr int WORD_COUNT = CHUNK_BYTES / WORD_BYTES;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]) + static_cast<uint32_t>(chunk_idx * CHUNK_BYTES);
    for (int word = warp::laneid(); word < WORD_COUNT; word += WARP_THREADS) {
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(word * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(dst + word * WORD_BYTES) = value;
    }
}

template <typename ScaleTile>
__device__ inline void store_scale_tmem_chunk_128x16_to_global(const ScaleTile &tile, uint8_t *dst) {
    if (dst == nullptr) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_ROW = 16 / WORD_BYTES;
    constexpr int TOTAL_WORDS = 128 * WORDS_PER_ROW;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = warp::laneid(); linear < TOTAL_WORDS; linear += WARP_THREADS) {
        const int row = linear / WORDS_PER_ROW;
        const int word = linear - row * WORDS_PER_ROW;
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(row * ScaleTile::cols + word * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(dst + linear * WORD_BYTES) = value;
    }
}

template <typename ScaleTile>
__device__ inline void store_scale_bf16_chunk_32x16_to_global(const ScaleTile &tile, bf16 *dst) {
    if (dst == nullptr) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int BF16_BYTES_PER_ROW = 16 * (int)sizeof(bf16);
    constexpr int WORDS_PER_ROW = BF16_BYTES_PER_ROW / WORD_BYTES;
    constexpr int TOTAL_WORDS = 32 * WORDS_PER_ROW;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(dst);
    for (int linear = warp::laneid(); linear < TOTAL_WORDS; linear += WARP_THREADS) {
        const int row = linear / WORDS_PER_ROW;
        const int word = linear - row * WORDS_PER_ROW;
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(row * tile.cols * sizeof(bf16) + word * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(dst_bytes + row * BF16_BYTES_PER_ROW + word * WORD_BYTES) = value;
    }
}

template <typename ScaleTile>
__device__ inline void store_scale_fp8_chunk_32x16_to_global_bf16(const ScaleTile &tile, bf16 *dst) {
    if (dst == nullptr) {
        return;
    }
    static_assert(std::is_same_v<typename ScaleTile::dtype, fp8e4m3>, "shared tile must use fp8e4m3 elements");
    constexpr int HALF_COLS = 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = warp::laneid(); linear < 32 * (HALF_COLS / 4); linear += WARP_THREADS) {
        const int row = linear / (HALF_COLS / 4);
        const int pack_col = linear - row * (HALF_COLS / 4);
        uint32_t packed_u32;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(packed_u32)
                     : "r"(smem_base + static_cast<uint32_t>(row * tile.cols + pack_col * 4)));
        const fp8e4m3_4 packed = *reinterpret_cast<const fp8e4m3_4 *>(&packed_u32);
        const float4 vals = kittens::base_types::convertor<float4, fp8e4m3_4>::convert(packed);
        dst[row * HALF_COLS + pack_col * 4 + 0] = __float2bfloat16(vals.x);
        dst[row * HALF_COLS + pack_col * 4 + 1] = __float2bfloat16(vals.y);
        dst[row * HALF_COLS + pack_col * 4 + 2] = __float2bfloat16(vals.z);
        dst[row * HALF_COLS + pack_col * 4 + 3] = __float2bfloat16(vals.w);
    }
}

template <typename RT>
__device__ inline void store_fp8_rt_left_half_32x16_to_global_bf16(const RT &src, bf16 *dst) {
    static_assert(std::is_same_v<typename RT::dtype, fp8e4m3_4>, "register tile must be fp8-packed");
    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "register tile must be row-major");
    static_assert(RT::rows == 32 && RT::cols == 32, "register tile must be 32x32");
    constexpr int HALF_COLS = 16;
    const int lane = warp::laneid();
    if (lane >= 16) {
        return;
    }
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int k = 0; k < RT::packed_per_tile; ++k) {
            const int row = i * TILE_ROW_DIM<typename RT::T> + lane;
            const int col = k * 4;
            const float4 vals = kittens::base_types::convertor<float4, fp8e4m3_4>::convert(src.tiles[i][0].data[k]);
            dst[row * HALF_COLS + col + 0] = __float2bfloat16(vals.x);
            dst[row * HALF_COLS + col + 1] = __float2bfloat16(vals.y);
            dst[row * HALF_COLS + col + 2] = __float2bfloat16(vals.z);
            dst[row * HALF_COLS + col + 3] = __float2bfloat16(vals.w);
        }
    }
}

template <typename RT>
__device__ inline void store_fp8_rt_raw_32x32_to_global_u32(const RT &src, uint32_t *dst) {
    if (dst == nullptr) {
        return;
    }
    static_assert(std::is_same_v<typename RT::dtype, fp8e4m3_4>, "register tile must be fp8-packed");
    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "register tile must be row-major");
    static_assert(RT::rows == 32 && RT::cols == 32, "register tile must be 32x32");
    const int lane = warp::laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int k = 0; k < RT::packed_per_tile; ++k) {
            dst[((lane * RT::height) + i) * RT::packed_per_tile + k] =
                *reinterpret_cast<const uint32_t *>(&src.tiles[i][0].data[k]);
        }
    }
}

template <typename C>
__device__ inline int debug_consumer_k_iters(
    const globals_quant_p_from_scores_debug<C> &g,
    int producer_iters_per_task
) {
    const bool producer_only_cut =
        g.debug_stage_cut == DEBUG_STAGE_CUT_ISSUE3_ONLY ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_WAIT ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_ISSUE ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_COMMIT ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_KSC_FINISH ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_SKIP_COMMIT_TAIL ||
        g.debug_stage_cut == DEBUG_STAGE_CUT_PRODUCER3_FRESH_COMMIT_TAIL;
    if (producer_only_cut && producer_iters_per_task == 3) {
        return 2;
    }
    return producer_iters_per_task;
}

template <typename C>
__device__ inline bool debug_dual_scores_slot_mode(
    const globals_quant_p_from_scores_debug<C> &g
) {
    return g.debug_stage_cut == DEBUG_STAGE_CUT_FULL ||
           g.debug_stage_cut == DEBUG_STAGE_CUT_DUAL_SLOT_FULL;
}

template <typename C>
__device__ inline bool debug_full_like_mode(
    const globals_quant_p_from_scores_debug<C> &g
) {
    return g.debug_stage_cut == DEBUG_STAGE_CUT_FULL ||
           g.debug_stage_cut == DEBUG_STAGE_CUT_DUAL_SLOT_FULL ||
           g.debug_stage_cut == DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL;
}

template <typename C>
__device__ inline semaphore &debug_scores_arrived_sem_for_iter(
    semaphore (&scores_arrived)[1],
    semaphore (&scores_arrived_fresh)[1],
    const globals_quant_p_from_scores_debug<C> &g,
    int k_iter
) {
    if (debug_dual_scores_slot_mode(g) && (k_iter & 1)) {
        return scores_arrived_fresh[0];
    }
    return scores_arrived[0];
}

template <typename C>
__device__ inline int debug_scores_arrived_phase_for_iter(
    const globals_quant_p_from_scores_debug<C> &g,
    int k_iter,
    int legacy_phase
) {
    if (debug_dual_scores_slot_mode(g)) {
        return (k_iter >> 1) & 1;
    }
    return legacy_phase;
}

template <typename C>
__device__ inline void store_lse_row_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    float lse_val
) {
    if (g.disable_lse_store) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    g.lse[{b, h, 0, debug_output_row_index(g, global_row)}] = lse_val;
}

template <typename C>
__device__ inline float qk_scale_log2(const globals_quant_p_from_scores_debug<C> &g, int b, int h) {
    constexpr float LOG2E_OVER_SQRT_D = 0.10408340855860843f;  // log2(e) / sqrt(192)
    float q_scale = 0.0f;
    float k_scale = 0.0f;
    if (warp::laneid() == 0) {
        q_scale = g.q_sg[{b, h, 0, 0}];
        k_scale = g.k_sg[{b, h, 0, 0}];
    }
    q_scale = __shfl_sync(0xFFFFFFFFu, q_scale, 0);
    k_scale = __shfl_sync(0xFFFFFFFFu, k_scale, 0);
    return LOG2E_OVER_SQRT_D * q_scale * k_scale;
}

template <typename C, typename PTile>
__device__ inline void store_p_fp4_tile_to_global_warpgroup(
    const PTile &p_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int packed_seqlen = g.tiles_to_run * (C::Nb / 2);
    const int row_base = m_tile * C::Mb;
    const int col_base = debug_window_k_iter(g, k_iter) * (C::Nb / 2);
    const uint32_t smem_base = __cvta_generic_to_shared(&p_smem.data[0]);
    constexpr int BYTES_PER_STORE = 4;
    static_assert(PTile::cols % BYTES_PER_STORE == 0, "Expected fp4 P tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = PTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = PTile::rows * STORES_PER_ROW;
    for (int linear = lane; linear < TOTAL_STORES; linear += 128) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(PTile::idx(smem_base, {row, col})));
        *reinterpret_cast<uint32_t *>(
            &g.p_fp4[((((b * heads + h) * seqlen) + row_base + row) * packed_seqlen) + col_base + col]
        ) = value;
    }
}

template <typename C, typename PScTile>
__device__ inline void store_p_sc_prepared_to_global_warpgroup(
    const PScTile &p_sc_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const uint32_t smem_base = __cvta_generic_to_shared(&p_sc_smem.data[0]);
    const int scale_k_base = debug_window_k_iter(g, k_iter) * 2;
    const int prepared_cols = g.tiles_to_run * 2;
    constexpr int ROW_BYTES = 8;
    constexpr int ROWS_PER_SLAB = 512 / ROW_BYTES;
    constexpr int TOTAL_ROWS = sizeof(PScTile) / ROW_BYTES;
    static_assert(TOTAL_ROWS == 128, "Expected 128 raw scale rows per P scale tile");
    for (int row = lane; row < TOTAL_ROWS; row += 128) {
        const int slab = row / ROWS_PER_SLAB;
        const int row_in_slab = row % ROWS_PER_SLAB;
        uint64_t value;
        asm volatile("ld.shared.b64 %0, [%1];"
                     : "=l"(value)
                     : "r"(smem_base + static_cast<uint32_t>(row * ROW_BYTES)));
        *reinterpret_cast<uint64_t *>(
            &g.p_sc_prepared[((((b * heads + h) * (seqlen / 128) + m_tile) * prepared_cols) + scale_k_base + slab) * 512 + row_in_slab * ROW_BYTES]
        ) = value;
    }
}

template <typename C, typename PTile>
__device__ inline void store_backward_ds_fp4_row_tile_to_global_warpgroup(
    const PTile &p_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    if (g.backward_ds_fp4_row == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int packed_seqlen = g.output_full_layout ? (seqlen / 2) : (g.tiles_to_run * (C::Nb / 2));
    const int row_base = m_tile * C::Mb;
    const int col_base = (g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter)) * (C::Nb / 2);
    const uint32_t smem_base = __cvta_generic_to_shared(&p_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert(PTile::cols % BYTES_PER_STORE == 0, "Expected fp4 dS tile cols divisible by vector width");
    static_assert(PTile::swizzle_bytes >= BYTES_PER_STORE, "Expected fp4 dS tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = PTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = PTile::rows * STORES_PER_ROW;
    for (int linear = lane; linear < TOTAL_STORES; linear += 128) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        uint4 value;
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                     : "r"(PTile::idx(smem_base, {row, col})));
        *reinterpret_cast<uint4 *>(
            &g.backward_ds_fp4_row[((((b * heads + h) * seqlen) + row_base + row) * packed_seqlen) + col_base + col]
        ) = value;
    }
}

template <typename C, typename PScTile>
__device__ inline void store_backward_ds_sc_row_tile_to_global_warpgroup(
    const PScTile &p_sc_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    static_assert(C::Nb == 128, "Direct row-MXFP4 dS scale emission expects K128 score tiles");
    if (g.backward_ds_sc_row == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int row_tiles = seqlen / C::Mb;
    const int k_tiles = g.output_full_layout ? (seqlen / C::Nb) : g.tiles_to_run;
    const int scale_k = g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter);
    const uint32_t smem_base = __cvta_generic_to_shared(&p_sc_smem.data[0]);
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_MXFP4_SCALE_TILE = 512 / WORD_BYTES;
    for (int linear = lane; linear < WORDS_PER_MXFP4_SCALE_TILE; linear += 128) {
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(linear * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(
            &g.backward_ds_sc_row[((((b * heads + h) * row_tiles + m_tile) * k_tiles + scale_k) * 512) + linear * WORD_BYTES]
        ) = value;
    }
}

template <typename C, typename PScTile>
__device__ inline void store_backward_ds_sc_row_word_to_global_warpgroup(
    PScTile &p_sc_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int global_row
) {
    static_assert(C::Nb == 128, "Direct row-MXFP4 dS scale emission expects K128 score tiles");
    if (g.backward_ds_sc_row == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int row_tiles = seqlen / C::Mb;
    const int k_tiles = g.output_full_layout ? (seqlen / C::Nb) : g.tiles_to_run;
    const int scale_k = g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter);
    uint8_t *scale_tile = &g.backward_ds_sc_row[((((b * heads + h) * row_tiles + m_tile) * k_tiles + scale_k) * 512)];
    const uint32_t packed = fp4pv_load_mxfp4_scale_word_swizzled(p_sc_smem, global_row);
    *reinterpret_cast<uint32_t *>(&scale_tile[fp4pv_scale_swizzle_idx(global_row & 127, 0)]) = packed;
}

template <typename C>
__device__ inline void store_backward_ds_row_mxfp4_direct_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int k_iter,
    const float2 (&scores_reg)[64]
) {
    static_assert(C::Nb == 128, "Direct row-MXFP4 dS emission expects K128 score tiles");
    if (g.backward_ds_fp4_row == nullptr || g.backward_ds_sc_row == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int output_row = debug_output_row_index(g, global_row);
    const int packed_seqlen = g.output_full_layout ? (seqlen / 2) : (g.tiles_to_run * (C::Nb / 2));
    const int col_base = (g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter)) * (C::Nb / 2);
    const int row_tiles = seqlen / C::Mb;
    const int k_tiles = g.output_full_layout ? (seqlen / C::Nb) : g.tiles_to_run;
    const int m_tile = global_row / C::Mb;
    const int scale_k = g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter);
    uint8_t *payload_row = &g.backward_ds_fp4_row[((((b * heads + h) * debug_output_rows(g)) + output_row) * packed_seqlen) + col_base];
    uint8_t *scale_tile = &g.backward_ds_sc_row[((((b * heads + h) * row_tiles + m_tile) * k_tiles + scale_k) * 512)];
    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group_bf16_amax(
            &scores_reg[mx_block * 16],
            g.backward_ds_quant_mode,
            words,
            e8m0);
        uint8_t *payload_dst = payload_row + mx_block * 16;
        *reinterpret_cast<uint32_t *>(payload_dst + 0) = words[0];
        *reinterpret_cast<uint32_t *>(payload_dst + 4) = words[1];
        *reinterpret_cast<uint32_t *>(payload_dst + 8) = words[2];
        *reinterpret_cast<uint32_t *>(payload_dst + 12) = words[3];
        scale_tile[fp4pv_scale_swizzle_idx(global_row & 127, mx_block)] = e8m0;
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename PTile, typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_transposed_stage_mxfp4_marker_bf16_amax(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int m_tile,
    int k_iter,
    const float2 (&scores_reg)[64]
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Local transposed dS stage expects 128x128 score tiles");
    #pragma unroll
    for (int qid = 0; qid < 4; ++qid) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[qid * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, qid);
    }
    warpgroup::sync(warpgroup::groupid() + 1);

    const int local_col = warpgroup::laneid();
    const int src_tile_x = local_col >> 6;
    const int src_col_in_tile = local_col & 63;
    const bool diagonal_tile = (k_iter == m_tile);
    uint32_t packed_scales = 0u;

    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        if (diagonal_tile && (mx_block * 32 + 31) < local_col) {
            constexpr uint32_t ZERO_FP4_WORD = 0x77777777u;
            uint32_t zero_words[4] = {
                ZERO_FP4_WORD,
                ZERO_FP4_WORD,
                ZERO_FP4_WORD,
                ZERO_FP4_WORD,
            };
            fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, zero_words, local_col, mx_block);
            fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, local_col, mx_block, 0u);
            continue;
        }

        __align__(8) bf16_2 values_bf[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int src_row0 = mx_block * 32 + jj * 2 + 0;
            const int src_row1 = mx_block * 32 + jj * 2 + 1;
            const int tile_idx0 = ((src_row0 >> 6) * 2) + src_tile_x;
            const int tile_idx1 = ((src_row1 >> 6) * 2) + src_tile_x;
            values_bf[jj].x = scan[tile_idx0][src_row0 & 63][src_col_in_tile];
            values_bf[jj].y = scan[tile_idx1][src_row1 & 63][src_col_in_tile];
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(values_bf, g.backward_ds_quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, local_col, mx_block);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, local_col, mx_block, e8m0);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    warpgroup::sync(warpgroup::groupid() + 1);
    return packed_scales;
}

template <typename C>
__device__ inline void store_backward_ds_col_mxfp4_from_populated_scan_full(
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Direct dS^T MXFP4 emission expects 128x128 score tiles");
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int tiles = seqlen / 128;
    const int local_col = warpgroup::laneid();
    const int out_row = k_iter * C::Nb + local_col;
    const int m_tile = global_row / C::Mb;
    const int packed_seqlen = seqlen / 2;
    uint8_t *payload_row =
        &g.backward_ds_fp4_col[((((b * heads + h) * seqlen) + out_row) * packed_seqlen) + m_tile * (C::Mb / 2)];
    uint8_t *scale_tile =
        &g.backward_ds_sc_col[((((b * heads + h) * tiles + k_iter) * tiles + m_tile) * 512)];
    const int src_tile_x = local_col >> 6;
    const int src_col_in_tile = local_col & 63;
    const bool diagonal_tile = (k_iter == m_tile);

    uint32_t packed_scales = 0u;
    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        uint8_t e8m0 = 0;
        if (diagonal_tile && (mx_block * 32 + 31) < local_col) {
            constexpr uint32_t ZERO_FP4_WORD = 0x77777777u;
            uint8_t *payload_dst = payload_row + mx_block * 16;
            fp4pv_store_global_u32x4_value(payload_dst, ZERO_FP4_WORD);
            packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
            continue;
        }
        __align__(8) bf16_2 values_bf[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int src_row0 = mx_block * 32 + jj * 2 + 0;
            const int src_row1 = mx_block * 32 + jj * 2 + 1;
            const int tile_idx0 = ((src_row0 >> 6) * 2) + src_tile_x;
            const int tile_idx1 = ((src_row1 >> 6) * 2) + src_tile_x;
            values_bf[jj].x = scan[tile_idx0][src_row0 & 63][src_col_in_tile];
            values_bf[jj].y = scan[tile_idx1][src_row1 & 63][src_col_in_tile];
        }
        uint32_t words[4];
        fp4pv_quantize_mxfp4_bf16_pairs_amax(values_bf, g.backward_ds_quant_mode, words, e8m0);
        uint8_t *payload_dst = payload_row + mx_block * 16;
        fp4pv_store_global_u32x4(payload_dst, words);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    fp4pv_store_global_scale_word_swizzled(scale_tile, local_col, packed_scales);
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C>
__device__ __noinline__ void zero_backward_ds_row_mxfp4_future_scale_tiles(
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Future row dS scale zeroing expects 128x128 tiles");
    if (g.backward_ds_sc_row == nullptr || !g.emit_backward_ds_row_mxfp4 || g.backward_ds_row_skip_future_zero) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int tiles = seqlen / 128;
    const int local_row = warpgroup::laneid();
    for (int k_iter = m_tile + 1; k_iter < tiles; ++k_iter) {
        uint8_t *scale_tile =
            &g.backward_ds_sc_row[((((b * heads + h) * tiles + m_tile) * tiles + k_iter) * 512)];
        fp4pv_store_global_scale_word_swizzled(scale_tile, local_row, 0u);
    }
}

template <typename C>
__device__ inline void zero_backward_ds_col_mxfp4_future_scale_tiles(
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Future dS^T scale zeroing expects 128x128 tiles");
    if (g.backward_ds_sc_col == nullptr || !g.emit_backward_ds_col_mxfp4 || g.backward_ds_col_skip_future_zero) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int tiles = seqlen / 128;
    const int local_col = warpgroup::laneid();
    for (int k_iter = m_tile + 1; k_iter < tiles; ++k_iter) {
        uint8_t *scale_tile =
            &g.backward_ds_sc_col[((((b * heads + h) * tiles + k_iter) * tiles + m_tile) * 512)];
        fp4pv_store_global_scale_word_swizzled(scale_tile, local_col, 0u);
    }
}

template <typename C>
__device__ inline void store_backward_ds_col_mxfp4_from_populated_scan_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Direct dS^T MXFP4 emission expects 128x128 score tiles");
    if (g.backward_ds_fp4_col == nullptr || g.backward_ds_sc_col == nullptr || !g.emit_backward_ds_col_mxfp4) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    store_backward_ds_col_mxfp4_from_populated_scan_full<C>(
        g, scan, b, h, global_row, k_iter);
}

template <typename C>
__device__ inline void store_backward_ds_col_mxfp4_direct_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter,
    const float2 (&scores_reg)[64]
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Direct dS^T MXFP4 emission expects 128x128 score tiles");
    if (g.backward_ds_fp4_col == nullptr || g.backward_ds_sc_col == nullptr || !g.emit_backward_ds_col_mxfp4) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }

    #pragma unroll
    for (int qid = 0; qid < 4; ++qid) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[qid * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, qid);
    }
    store_backward_ds_col_mxfp4_from_populated_scan_if_needed<C>(
        g, scan, b, h, global_row, k_iter);
}

template <typename C>
__device__ inline void store_backward_ds_row_col_mxfp4_direct_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter,
    const float2 (&scores_reg)[64]
) {
    static_assert(C::Mb == 128 && C::Nb == 128, "Direct dS row+col MXFP4 emission expects 128x128 score tiles");
    if (g.backward_ds_fp4_row == nullptr || g.backward_ds_sc_row == nullptr ||
        g.backward_ds_fp4_col == nullptr || g.backward_ds_sc_col == nullptr ||
        !g.emit_backward_ds_col_mxfp4) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }

    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int output_row = debug_output_row_index(g, global_row);
    const int packed_seqlen = g.output_full_layout ? (seqlen / 2) : (g.tiles_to_run * (C::Nb / 2));
    const int col_base = (g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter)) * (C::Nb / 2);
    const int row_tiles = seqlen / C::Mb;
    const int k_tiles = g.output_full_layout ? (seqlen / C::Nb) : g.tiles_to_run;
    const int m_tile = global_row / C::Mb;
    const int scale_k = g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter);
    uint8_t *payload_row = &g.backward_ds_fp4_row[((((b * heads + h) * debug_output_rows(g)) + output_row) * packed_seqlen) + col_base];
    uint8_t *scale_row_tile = &g.backward_ds_sc_row[((((b * heads + h) * row_tiles + m_tile) * k_tiles + scale_k) * 512)];

    uint32_t packed_row_scales = 0u;
    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, mx_block);

        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, g.backward_ds_quant_mode, words, e8m0);
        uint8_t *payload_dst = payload_row + mx_block * 16;
        fp4pv_store_global_u32x4(payload_dst, words);
        packed_row_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    fp4pv_store_global_scale_word_swizzled(scale_row_tile, global_row & 127, packed_row_scales);
    warpgroup::sync(warpgroup::groupid() + 1);

    const int tiles = seqlen / 128;
    const int local_col = warpgroup::laneid();
    const int out_row = k_iter * C::Nb + local_col;
    uint8_t *payload_col_row =
        &g.backward_ds_fp4_col[((((b * heads + h) * seqlen) + out_row) * (seqlen / 2)) + m_tile * (C::Mb / 2)];
    uint8_t *scale_col_tile =
        &g.backward_ds_sc_col[((((b * heads + h) * tiles + k_iter) * tiles + m_tile) * 512)];
    const int src_tile_x = local_col >> 6;
    const int src_col_in_tile = local_col & 63;

    uint32_t packed_col_scales = 0u;
    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        __align__(8) bf16_2 values_bf[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int src_row0 = mx_block * 32 + jj * 2 + 0;
            const int src_row1 = mx_block * 32 + jj * 2 + 1;
            const int tile_idx0 = ((src_row0 >> 6) * 2) + src_tile_x;
            const int tile_idx1 = ((src_row1 >> 6) * 2) + src_tile_x;
            values_bf[jj].x = scan[tile_idx0][src_row0 & 63][src_col_in_tile];
            values_bf[jj].y = scan[tile_idx1][src_row1 & 63][src_col_in_tile];
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(values_bf, g.backward_ds_quant_mode, words, e8m0);
        uint8_t *payload_dst = payload_col_row + mx_block * 16;
        fp4pv_store_global_u32x4(payload_dst, words);
        packed_col_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    fp4pv_store_global_scale_word_swizzled(scale_col_tile, local_col, packed_col_scales);
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C>
__device__ inline void store_p_bf16_live_tile_if_needed(
    bf16 *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int k_iter,
    const float2 *scores_reg
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_rows = debug_output_rows(g);
    const int output_row = debug_output_row_index(g, global_row);
    const int col_base = (g.output_full_layout ? k_iter : debug_window_k_iter(g, k_iter)) * C::Nb;
    const int window_cols = g.output_full_layout ? g.q.rows() : g.tiles_to_run * C::Nb;
    uint32_t *row_dst = reinterpret_cast<uint32_t *>(
        &dst[((((b * heads + h) * output_rows) + output_row) * window_cols) + col_base]
    );
    #pragma unroll
    for (int q = 0; q < C::Nb / 2; ++q) {
        bf16_2 packed = __float22bfloat162_rn(scores_reg[q]);
        row_dst[q] = *reinterpret_cast<uint32_t *>(&packed);
    }
}

__device__ __forceinline__ void fp4pv_load_score_quarter_tmem(
    float2 (&scores)[16],
    uint32_t score_tt_base,
    int qid
) {
    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
        : "=f"(scores[0].x), "=f"(scores[0].y), "=f"(scores[1].x), "=f"(scores[1].y),
          "=f"(scores[2].x), "=f"(scores[2].y), "=f"(scores[3].x), "=f"(scores[3].y),
          "=f"(scores[4].x), "=f"(scores[4].y), "=f"(scores[5].x), "=f"(scores[5].y),
          "=f"(scores[6].x), "=f"(scores[6].y), "=f"(scores[7].x), "=f"(scores[7].y),
          "=f"(scores[8].x), "=f"(scores[8].y), "=f"(scores[9].x), "=f"(scores[9].y),
          "=f"(scores[10].x), "=f"(scores[10].y), "=f"(scores[11].x), "=f"(scores[11].y),
          "=f"(scores[12].x), "=f"(scores[12].y), "=f"(scores[13].x), "=f"(scores[13].y),
          "=f"(scores[14].x), "=f"(scores[14].y), "=f"(scores[15].x), "=f"(scores[15].y)
        : "r"(score_tt_base + qid * 32));
}

__device__ __forceinline__ void fp4pv_load_score_half_tmem(
    float2 (&scores)[32],
    uint32_t score_tt_base,
    int half
) {
    const int qid0 = half * 2;
    const int qid1 = qid0 + 1;
    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
        : "=f"(scores[0].x), "=f"(scores[0].y), "=f"(scores[1].x), "=f"(scores[1].y),
          "=f"(scores[2].x), "=f"(scores[2].y), "=f"(scores[3].x), "=f"(scores[3].y),
          "=f"(scores[4].x), "=f"(scores[4].y), "=f"(scores[5].x), "=f"(scores[5].y),
          "=f"(scores[6].x), "=f"(scores[6].y), "=f"(scores[7].x), "=f"(scores[7].y),
          "=f"(scores[8].x), "=f"(scores[8].y), "=f"(scores[9].x), "=f"(scores[9].y),
          "=f"(scores[10].x), "=f"(scores[10].y), "=f"(scores[11].x), "=f"(scores[11].y),
          "=f"(scores[12].x), "=f"(scores[12].y), "=f"(scores[13].x), "=f"(scores[13].y),
          "=f"(scores[14].x), "=f"(scores[14].y), "=f"(scores[15].x), "=f"(scores[15].y)
        : "r"(score_tt_base + qid0 * 32));
    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
        : "=f"(scores[16].x), "=f"(scores[16].y), "=f"(scores[17].x), "=f"(scores[17].y),
          "=f"(scores[18].x), "=f"(scores[18].y), "=f"(scores[19].x), "=f"(scores[19].y),
          "=f"(scores[20].x), "=f"(scores[20].y), "=f"(scores[21].x), "=f"(scores[21].y),
          "=f"(scores[22].x), "=f"(scores[22].y), "=f"(scores[23].x), "=f"(scores[23].y),
          "=f"(scores[24].x), "=f"(scores[24].y), "=f"(scores[25].x), "=f"(scores[25].y),
          "=f"(scores[26].x), "=f"(scores[26].y), "=f"(scores[27].x), "=f"(scores[27].y),
          "=f"(scores[28].x), "=f"(scores[28].y), "=f"(scores[29].x), "=f"(scores[29].y),
          "=f"(scores[30].x), "=f"(scores[30].y), "=f"(scores[31].x), "=f"(scores[31].y)
        : "r"(score_tt_base + qid1 * 32));
}

__device__ __forceinline__ void fp4pv_apply_live_dp_quarter_tmem_x16(
    float2 *scores_reg,
    uint32_t score_tt_base,
    int qid,
    float2 live_dp_scale_2,
    float2 neg_delta_scaled_2
) {
    #pragma unroll
    for (int seg = 0; seg < 2; ++seg) {
        float dp_half[16];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];}"
            : "=f"(dp_half[0]), "=f"(dp_half[1]), "=f"(dp_half[2]), "=f"(dp_half[3]),
              "=f"(dp_half[4]), "=f"(dp_half[5]), "=f"(dp_half[6]), "=f"(dp_half[7]),
              "=f"(dp_half[8]), "=f"(dp_half[9]), "=f"(dp_half[10]), "=f"(dp_half[11]),
              "=f"(dp_half[12]), "=f"(dp_half[13]), "=f"(dp_half[14]), "=f"(dp_half[15])
            : "r"(score_tt_base + qid * 32 + seg * 16)
        );
        #pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const int q = qid * 16 + seg * 8 + jj;
            const float2 dp_pair = {dp_half[jj * 2], dp_half[jj * 2 + 1]};
            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                scores_reg[q], dp_pair, live_dp_scale_2, neg_delta_scaled_2);
        }
    }
}

__device__ __forceinline__ void fp4pv_apply_live_dp_quarter_tmem_x8(
    float2 *scores_reg,
    uint32_t score_tt_base,
    int qid,
    float2 live_dp_scale_2,
    float2 neg_delta_scaled_2
) {
    #pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
        float dp_word[8];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];}"
            : "=f"(dp_word[0]), "=f"(dp_word[1]), "=f"(dp_word[2]), "=f"(dp_word[3]),
              "=f"(dp_word[4]), "=f"(dp_word[5]), "=f"(dp_word[6]), "=f"(dp_word[7])
            : "r"(score_tt_base + qid * 32 + seg * 8)
        );
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int q = qid * 16 + seg * 4 + jj;
            const float2 dp_pair = {dp_word[jj * 2], dp_word[jj * 2 + 1]};
            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                scores_reg[q], dp_pair, live_dp_scale_2, neg_delta_scaled_2);
        }
    }
}

__device__ __forceinline__ void fp4pv_apply_live_dp_quarter_tmem_x8_aligned_probe(
    float2 *scores_reg,
    uint32_t score_tt_base,
    int qid,
    float2 live_dp_scale_2,
    float2 neg_delta_scaled_2
) {
    #pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
        float dp_word[8];
        const bool aligned_base = (seg == 0 || seg == 2);
        if (aligned_base) {
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];}"
                : "=f"(dp_word[0]), "=f"(dp_word[1]), "=f"(dp_word[2]), "=f"(dp_word[3]),
                  "=f"(dp_word[4]), "=f"(dp_word[5]), "=f"(dp_word[6]), "=f"(dp_word[7])
                : "r"(score_tt_base + qid * 32 + seg * 8)
            );
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dp_word[i] = 0.0f;
            }
        }
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int q = qid * 16 + seg * 4 + jj;
            const float2 dp_pair = {dp_word[jj * 2], dp_word[jj * 2 + 1]};
            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                scores_reg[q], dp_pair, live_dp_scale_2, neg_delta_scaled_2);
        }
    }
}

__device__ __forceinline__ void fp4pv_apply_online_causal_mask_score_quarter(
    float2 (&scores)[16],
    int qid,
    int idx,
    int m_tile
) {
    if (idx >= m_tile) {
        const int causal_col = (idx > m_tile) ? -1 : static_cast<int>(warpgroup::laneid());
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int col0 = qid * 32 + jj * 2;
            if (col0 > causal_col) {
                scores[jj].x = base_types::constants<float>::neg_infty();
            }
            if (col0 + 1 > causal_col) {
                scores[jj].y = base_types::constants<float>::neg_infty();
            }
        }
    }
}

__device__ __forceinline__ void fp4pv_apply_online_causal_mask_score_half(
    float2 (&scores)[32],
    int half,
    int idx,
    int m_tile
) {
    if (idx >= m_tile) {
        const int causal_col = (idx > m_tile) ? -1 : static_cast<int>(warpgroup::laneid());
        #pragma unroll
        for (int jj = 0; jj < 32; ++jj) {
            const int col0 = half * 64 + jj * 2;
            if (col0 > causal_col) {
                scores[jj].x = base_types::constants<float>::neg_infty();
            }
            if (col0 + 1 > causal_col) {
                scores[jj].y = base_types::constants<float>::neg_infty();
            }
        }
    }
}

template <typename C>
__device__ inline void store_scores_bf16_scan_repacked_tile_if_needed(
    bf16 *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    const fp4pv_p_scan3d &scan,
    int b,
    int h,
    int global_row,
    int k_iter
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_rows = debug_output_rows(g);
    const int output_row = debug_output_row_index(g, global_row);
    const int col_base = debug_window_k_iter(g, k_iter) * C::Nb;
    const int window_cols = g.tiles_to_run * C::Nb;
    uint64_t *row_dst = reinterpret_cast<uint64_t *>(
        &dst[((((b * heads + h) * output_rows) + output_row) * window_cols) + col_base]
    );
    const int top_bottom = (global_row >> 6) & 1;
    const int local_row = global_row & 63;
    for (int quarter = 0; quarter < 4; ++quarter) {
        const int tile_x = quarter >> 1;
        const int local_col_base = (quarter & 1) * 32;
        const int tile_idx = top_bottom * 2 + tile_x;
        const uint32_t smem_base = __cvta_generic_to_shared(&scan[tile_idx][local_row][local_col_base]);
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            uint64_t packed;
            asm volatile("ld.shared.b64 %0, [%1];"
                         : "=l"(packed)
                         : "r"(smem_base + static_cast<uint32_t>(i * sizeof(bf16_2))));
            row_dst[quarter * 8 + i / 2] = packed;
        }
    }
}

template <typename C>
__device__ inline void store_scores_bf16_tmem_x16_tile_if_needed(
    bf16 *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int m_tile,
    int k_iter,
    uint32_t score_tt_base
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_rows = debug_output_rows(g);
    const int output_row = debug_output_row_index(g, global_row);
    const int col_base = debug_window_k_iter(g, k_iter) * C::Nb;
    const int window_cols = g.tiles_to_run * C::Nb;
    uint32_t *row_dst = reinterpret_cast<uint32_t *>(
        &dst[((((b * heads + h) * output_rows) + output_row) * window_cols) + col_base]
    );
    const int causal_col = (k_iter > m_tile) ? -1 : (int)warpgroup::laneid();
    const bool apply_causal = k_iter >= m_tile;
    #pragma unroll
    for (int seg = 0; seg < C::Nb / 16; ++seg) {
        float reg[16];
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];}"
            : "=f"(reg[0]), "=f"(reg[1]), "=f"(reg[2]), "=f"(reg[3]),
              "=f"(reg[4]), "=f"(reg[5]), "=f"(reg[6]), "=f"(reg[7]),
              "=f"(reg[8]), "=f"(reg[9]), "=f"(reg[10]), "=f"(reg[11]),
              "=f"(reg[12]), "=f"(reg[13]), "=f"(reg[14]), "=f"(reg[15])
            : "r"(score_tt_base + seg * 16)
        );
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 pair = {reg[i * 2], reg[i * 2 + 1]};
            if (apply_causal) {
                const int col0 = seg * 16 + i * 2;
                const int col1 = col0 + 1;
                if (col0 > causal_col) {
                    pair.x = base_types::constants<float>::neg_infty();
                }
                if (col1 > causal_col) {
                    pair.y = base_types::constants<float>::neg_infty();
                }
            }
            const bf16_2 packed = __float22bfloat162_rn(pair);
            row_dst[seg * 8 + i] = *reinterpret_cast<const uint32_t *>(&packed);
        }
    }
}

template <typename C, typename PScTile>
__device__ inline void store_p_sc_rowmajor_live_tile_if_needed(
    const PScTile &p_sc_smem,
    uint8_t *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int k_iter
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    constexpr int GROUPS_PER_TILE = tk_localcta::SCALES_PER_CHUNK_X;
    static_assert(GROUPS_PER_TILE == 8, "Expected 8 rowwise scale groups per 128-column tile");
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int rowmajor_cols = g.tiles_to_run * GROUPS_PER_TILE;
    const int col_base = debug_window_k_iter(g, k_iter) * GROUPS_PER_TILE;
    auto &raw_scales = *reinterpret_cast<const fp4pv_p_raw_scales *>(&p_sc_smem.data[0]);
    const int row = global_row & 127;
    *reinterpret_cast<uint64_t *>(
        &dst[((((b * heads + h) * seqlen) + global_row) * rowmajor_cols) + col_base]
    ) = *reinterpret_cast<const uint64_t *>(&raw_scales[row][0]);
}

template <typename C, typename PScTile>
__device__ inline void store_p_sc_rowmajor_scan_tile_if_needed(
    const PScTile &p_sc_smem,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int lane
) {
    if (g.p_sc_rowmajor_scan == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    const int seqlen = g.q.rows();
    const int heads = g.q.depth();
    const int rowmajor_cols = g.tiles_to_run * tk_localcta::SCALES_PER_CHUNK_X;
    const int col_base = debug_window_k_iter(g, k_iter) * tk_localcta::SCALES_PER_CHUNK_X;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_sc_smem.data[0]);
    for (int row = lane; row < 128; row += 128) {
        uint64_t value;
        asm volatile("ld.shared.b64 %0, [%1];"
                     : "=l"(value)
                     : "r"(smem_base + static_cast<uint32_t>(row * sizeof(uint64_t))));
        *reinterpret_cast<uint64_t *>(
            &g.p_sc_rowmajor_scan[((((b * heads + h) * seqlen) + m_tile * C::Mb + row) * rowmajor_cols) + col_base]
        ) = value;
    }
}

template <typename C, typename PScTile>
__device__ inline void quantize_localcta_p_scales_from_scan_with_amax_and_store_raw(
    PScTile &p_sc_prepared_tile,
    fp4pv_p_scan3d &scan,
    float amax_val,
    int consumer_tid,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter
) {
    float s_enc = FP4PV_LOCALCTA_GLOBAL_SCALE_NUM / amax_val;
    s_enc = fminf(s_enc, transformer_engine::detail::TypeExtrema<float>::max);
    if (amax_val == 0.0f || s_enc == 0.0f) {
        s_enc = 1.0f;
    }

    #pragma unroll
    for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
        const int stage_y = t / tk_localcta::TILES_X;
        const int stage_x = t % tk_localcta::TILES_X;
        rowwise_scaling_scales_only_group_padded<128>(
            scan,
            reinterpret_cast<tk_localcta::nvfp4_scale_t *>(&p_sc_prepared_tile.data[0]),
            s_enc,
            stage_y,
            stage_x,
            t,
            consumer_tid);
    }
    tk_localcta::subgroup_barrier_sync<128>();
    store_p_sc_rowmajor_scan_tile_if_needed<C>(p_sc_prepared_tile, g, b, h, m_tile, k_iter, consumer_tid);
    tk_localcta::subgroup_barrier_sync<128>();
    finalize_localcta_p_scales_with_amax<C>(p_sc_prepared_tile, amax_val, consumer_tid);
}

template <typename C>
__device__ inline void store_debug_row_stat_if_needed(
    float *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int global_row,
    int k_iter,
    float value
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_row_selected(g, global_row)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_rows = debug_output_rows(g);
    dst[((((b * heads + h) * output_rows) + debug_output_row_index(g, global_row)) * g.tiles_to_run) + debug_window_k_iter(g, k_iter)] = value;
}

template <typename C>
__device__ inline void store_debug_tile_amax_if_needed(
    float *dst,
    const globals_quant_p_from_scores_debug<C> &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    float value
) {
    if (dst == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    const int heads = g.q.depth();
    const int output_m_tiles = debug_output_m_tiles(g);
    dst[((((b * heads + h) * output_m_tiles) + debug_output_m_tile_index(g, m_tile)) * g.tiles_to_run) + debug_window_k_iter(g, k_iter)] = value;
}

template <typename C, typename ScaleTile>
__device__ inline void store_debug_q_sc_tile_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    const ScaleTile &tile,
    int b,
    int h,
    int m_tile
) {
    if (g.debug_q_sc_loaded == nullptr) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_CHUNK = 512 / WORD_BYTES;
    constexpr int TOTAL_WORDS = C::QK_SCALE_CHUNKS * WORDS_PER_CHUNK;
    const int heads = g.q.depth();
    const int output_m_tile = debug_output_m_tile_index(g, m_tile);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = 0; linear < TOTAL_WORDS; ++linear) {
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(linear * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(
            &g.debug_q_sc_loaded[((((b * debug_output_m_tiles(g)) + output_m_tile) * (heads * C::QK_SCALE_CHUNKS) + h * C::QK_SCALE_CHUNKS) * 512) + linear * WORD_BYTES]
        ) = value;
    }
}

template <typename C, typename ScaleTile>
__device__ inline void store_debug_k_sc_tile_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    const ScaleTile &tile,
    int b,
    int h,
    int k_iter
) {
    if (g.debug_k_sc_loaded == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_CHUNK = 512 / WORD_BYTES;
    constexpr int TOTAL_WORDS = C::QK_SCALE_CHUNKS * WORDS_PER_CHUNK;
    const int heads = g.q.depth();
    const int out_k = debug_window_k_iter(g, k_iter);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int linear = 0; linear < TOTAL_WORDS; ++linear) {
        uint32_t value;
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(value)
                     : "r"(smem_base + static_cast<uint32_t>(linear * WORD_BYTES)));
        *reinterpret_cast<uint32_t *>(
            &g.debug_k_sc_loaded[((((b * g.tiles_to_run) + out_k) * (heads * C::QK_SCALE_CHUNKS) + h * C::QK_SCALE_CHUNKS) * 512) + linear * WORD_BYTES]
        ) = value;
    }
}

template <typename C, typename ScaleTile>
__device__ inline void store_debug_q_sc_tmem_chunk_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    const ScaleTile &tile,
    int b,
    int h,
    int m_tile,
    int chunk
) {
    if (g.debug_q_sc_tmem_loaded == nullptr) {
        return;
    }
    if (!debug_output_m_tile_selected(g, m_tile)) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_ROW = 16 / WORD_BYTES;
    const int heads = g.q.depth();
    const int output_m_tile = debug_output_m_tile_index(g, m_tile);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int row = 0; row < 128; ++row) {
        for (int word = 0; word < WORDS_PER_ROW; ++word) {
            uint32_t value;
            asm volatile("ld.shared.b32 %0, [%1];"
                         : "=r"(value)
                         : "r"(smem_base + static_cast<uint32_t>(row * ScaleTile::cols + word * WORD_BYTES)));
            *reinterpret_cast<uint32_t *>(
                &g.debug_q_sc_tmem_loaded[((((b * debug_output_m_tiles(g)) + output_m_tile) * (heads * C::QK_SCALE_CHUNKS) + h * C::QK_SCALE_CHUNKS + chunk) * 512) + (row * 16 + word * WORD_BYTES)]
            ) = value;
        }
    }
}

template <typename C, typename ScaleTile>
__device__ inline void store_debug_k_sc_tmem_chunk_if_needed(
    const globals_quant_p_from_scores_debug<C> &g,
    const ScaleTile &tile,
    int b,
    int h,
    int k_iter,
    int chunk
) {
    if (g.debug_k_sc_tmem_loaded == nullptr) {
        return;
    }
    if (!debug_tile_selected(g, k_iter)) {
        return;
    }
    constexpr int WORD_BYTES = sizeof(uint32_t);
    constexpr int WORDS_PER_ROW = 16 / WORD_BYTES;
    const int heads = g.q.depth();
    const int out_k = debug_window_k_iter(g, k_iter);
    const uint32_t smem_base = __cvta_generic_to_shared(&tile.data[0]);
    for (int row = 0; row < 128; ++row) {
        for (int word = 0; word < WORDS_PER_ROW; ++word) {
            uint32_t value;
            asm volatile("ld.shared.b32 %0, [%1];"
                         : "=r"(value)
                         : "r"(smem_base + static_cast<uint32_t>(row * ScaleTile::cols + word * WORD_BYTES)));
            *reinterpret_cast<uint32_t *>(
                &g.debug_k_sc_tmem_loaded[((((b * g.tiles_to_run) + out_k) * (heads * C::QK_SCALE_CHUNKS) + h * C::QK_SCALE_CHUNKS + chunk) * 512) + (row * 16 + word * WORD_BYTES)]
            ) = value;
        }
    }
}

template <typename TT, typename ST>
__device__ inline void copy_scale_tt_to_shared(ST &dst, const TT &src) {
    rt_fp8e4m3<128, 32> scale_rt;
    kittens::group<1>::load_async(scale_rt, src);
    tensor_load_wait();
    tensor_before_thread_sync();
    kittens::group<1>::store(dst, scale_rt);
}

template <typename TT>
__device__ inline void load_scale_tt_subtile_to_rt(rt_fp8e4m3<32, 32> &dst, const TT &src, int row_offset) {
    auto src_subtile = src.template subtile<tt<typename TT::dtype, 32, 32>>(row_offset, 0);
    kittens::group<1>::load_async(dst, src_subtile);
    tensor_load_wait();
    tensor_before_thread_sync();
}

template <typename TT, typename ST>
__device__ inline void copy_scale_tt_subtile_to_shared(ST &dst, const TT &src, int row_offset) {
    rt_fp8e4m3<32, 32> scale_rt;
    load_scale_tt_subtile_to_rt(scale_rt, src, row_offset);
    kittens::group<1>::store(dst, scale_rt);
}

template <typename TT>
__device__ inline void dump_scale_tt_subtile_to_global_bf16(bf16 *dst, const TT &src, int row_offset) {
    rt_fp8e4m3<32, 32> scale_rt_fp8;
    load_scale_tt_subtile_to_rt(scale_rt_fp8, src, row_offset);
    store_fp8_rt_left_half_32x16_to_global_bf16(scale_rt_fp8, dst);
}

template <typename TT>
__device__ inline void dump_scale_tt_subtile_to_global_raw_u32(uint32_t *dst, const TT &src, int row_offset) {
    rt_fp8e4m3<32, 32> scale_rt_fp8;
    load_scale_tt_subtile_to_rt(scale_rt_fp8, src, row_offset);
    store_fp8_rt_raw_32x32_to_global_u32(scale_rt_fp8, dst);
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_probe_loaded_qk_scale_chunks_debug(const __grid_constant__ globals_loaded_qk_scale_chunk_debug<C> g) {
    const int cta_rank = cluster_ctarank();

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    typename globals_loaded_qk_scale_chunk_debug<C>::q_sc_smem_storage &q_sc_smem =
        al.allocate<typename globals_loaded_qk_scale_chunk_debug<C>::q_sc_smem_storage>();
    typename globals_loaded_qk_scale_chunk_debug<C>::k_sc_smem_storage &k_sc_smem =
        al.allocate<typename globals_loaded_qk_scale_chunk_debug<C>::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] = al.allocate<typename globals_loaded_qk_scale_chunk_debug<C>::k_sc_guard_slot, 1>();
    (void)k_sc_guard;
    auto (&scale_dbg_smem)[4] = al.allocate<st_fp8e4m3<32, 32, false>, 4>();

    if (cta_rank == 0) {
        copy_scale_tile_bytes_to_shared(slot_tile_at(q_sc_smem, 0), g.q_sc_tile_bytes);
        copy_scale_tile_bytes_to_shared(slot_tile_at(k_sc_smem, 0), g.k_sc_tile_bytes);
    }
    __syncthreads();
    if (cta_rank == 0 && threadIdx.x == 0) {
        publish_shared_backing();
        publish_cluster_shared_backing_if_needed<C>();
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    if (warpgroup::groupid() != 3 || warpgroup::warpid() != 0 || cta_rank != 0) {
        return;
    }

    const bool issue_lane = warp::elect_leader();
    static_assert(64 <= MAX_TENSOR_COLS, "scale chunk probe exceeds TMEM budget");
    full_tt_fp8e4m3<32> q_sc_tm_dbg = tm_alloc.template allocate<full_tt_fp8e4m3<32>>(0);
    full_tt_fp8e4m3<32> k_sc_tm_dbg = tm_alloc.template allocate<full_tt_fp8e4m3<32>>(32);

    store_scale_chunk_32x16_to_global(slot_tile_at(q_sc_smem, 0), g.q_sc_source_chunk, g.chunk_idx);
    store_scale_chunk_32x16_to_global(slot_tile_at(k_sc_smem, 0), g.k_sc_source_chunk, g.chunk_idx);
    __syncwarp();
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_SOURCE_ONLY) {
        return;
    }

    auto q_sc_tm_dbg_lo = q_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(0);
    auto q_sc_tm_dbg_hi = q_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(16);
    if (issue_lane) {
        load_q_scale_chunk<C>(q_sc_tm_dbg_lo, slot_tile_at(q_sc_smem, 0), g.chunk_idx);
        load_q_scale_chunk<C>(q_sc_tm_dbg_hi, slot_tile_at(q_sc_smem, 0), g.chunk_idx);
    }
    __syncwarp();
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_LOAD_SYNC) {
        tensor_load_wait();
        tensor_before_thread_sync();
        return;
    }
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_RT_LOAD_ONLY) {
        #pragma unroll
        for (int row_block = 0; row_block < 4; ++row_block) {
            rt_fp8e4m3<32, 32> scale_rt;
            load_scale_tt_subtile_to_rt(scale_rt, q_sc_tm_dbg, row_block * 32);
        }
        return;
    }
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_DIRECT_DUMP || g.stage_mode == DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP) {
        #pragma unroll
        for (int row_block = 0; row_block < 4; ++row_block) {
            dump_scale_tt_subtile_to_global_bf16(g.q_sc_tmem_chunk + row_block * 32 * 16, q_sc_tm_dbg, row_block * 32);
            dump_scale_tt_subtile_to_global_raw_u32(g.q_sc_tmem_raw_words + row_block * 32 * 2 * 4, q_sc_tm_dbg, row_block * 32);
        }
        if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_DIRECT_DUMP) {
            return;
        }
    }
    if (g.stage_mode != DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP) {
        #pragma unroll
        for (int row_block = 0; row_block < 4; ++row_block) {
            copy_scale_tt_subtile_to_shared(scale_dbg_smem[row_block], q_sc_tm_dbg, row_block * 32);
        }
        __syncwarp();
        if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_COPY_SHARED_ONLY) {
            return;
        }
        for (int row_block = 0; row_block < 4; ++row_block) {
            store_scale_fp8_chunk_32x16_to_global_bf16(scale_dbg_smem[row_block], g.q_sc_tmem_chunk + row_block * 32 * 16);
        }
        __syncwarp();
        if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_Q_LOAD_COPY) {
            return;
        }
    }

    auto k_sc_tm_dbg_lo = k_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(0);
    auto k_sc_tm_dbg_hi = k_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(16);
    if (issue_lane) {
        load_k_scale_chunk<C>(k_sc_tm_dbg_lo, slot_tile_at(k_sc_smem, 0), g.chunk_idx);
        load_k_scale_chunk<C>(k_sc_tm_dbg_hi, slot_tile_at(k_sc_smem, 0), g.chunk_idx);
    }
    __syncwarp();
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_K_LOAD_SYNC) {
        tensor_load_wait();
        tensor_before_thread_sync();
        return;
    }
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP) {
        #pragma unroll
        for (int row_block = 0; row_block < 4; ++row_block) {
            dump_scale_tt_subtile_to_global_bf16(g.k_sc_tmem_chunk + row_block * 32 * 16, k_sc_tm_dbg, row_block * 32);
            dump_scale_tt_subtile_to_global_raw_u32(g.k_sc_tmem_raw_words + row_block * 32 * 2 * 4, k_sc_tm_dbg, row_block * 32);
        }
        return;
    }
    #pragma unroll
    for (int row_block = 0; row_block < 4; ++row_block) {
        copy_scale_tt_subtile_to_shared(scale_dbg_smem[row_block], k_sc_tm_dbg, row_block * 32);
    }
    __syncwarp();
    for (int row_block = 0; row_block < 4; ++row_block) {
        store_scale_fp8_chunk_32x16_to_global_bf16(scale_dbg_smem[row_block], g.k_sc_tmem_chunk + row_block * 32 * 16);
    }
    __syncwarp();
    if (g.stage_mode == DEBUG_SCALE_PROBE_STAGE_K_LOAD_COPY) {
        return;
    }
}

template <
    typename C,
    bool NORMALIZE_TILES = true,
    bool QK_ONLY = false,
    bool SAVED_LSE_BWD_ONLY = false,
    bool SAVED_BWD_EMIT_ROW_MXFP4 = false,
    bool SAVED_BWD_EMIT_COL_MXFP4 = false,
    bool SAVED_BWD_STORE_BF16 = true,
    bool SAVED_BWD_LOCAL_STAGE_ONLY = false,
    bool SAVED_BWD_DQ_DUAL_HALVES = false,
    int FIXED_DEBUG_STAGE_CUT = -1,
    bool FIXED_FULL_PRODUCTION_PATH = false,
    bool LIVE_DP = false,
    bool LIVE_DP_MODE0_ONLY = false,
    bool LIVE_DP_MODE0_X8 = false,
    bool BWD_PAIR_SINGLE_KERNEL = false>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_quantize_p_from_scores_debug(const __grid_constant__ globals_quant_p_from_scores_debug<C> g) {
    using G = globals_quant_p_from_scores_debug<C>;
    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_tail_outputs = tt<float, C::Mb, C::Dvo / 2>;
    using d_tt_q_sc = full_tt_fp8e4m3<C::Q_SC_TMEM_WIDTH>;
    using d_tt_sc16 = full_tt_fp8e4m3<16>;
    using d_tt_bwd_p_sc = full_tt_fp8e8m0<16>;
    using d_tt_bwd_k_sc = full_tt_fp8e8m0<16>;
    constexpr int COMPACT_Q_SC_BASE = C::Nb;
    constexpr int COMPACT_K_SC_BASE = COMPACT_Q_SC_BASE + C::Q_SC_TMEM_WIDTH;
    constexpr int QK_COMPACT_ACTIVE_TMEM_END = COMPACT_K_SC_BASE + 16;
    constexpr int BWD_DQ_P_SC_BASE = QK_COMPACT_ACTIVE_TMEM_END;
    constexpr int BWD_DQ_K_SC_BASE = BWD_DQ_P_SC_BASE + 16;
    constexpr int BWD_DQ_K1_SC_BASE = BWD_DQ_K_SC_BASE + 16;
    constexpr int BWD_DQ_OUTPUT_BASE = C::Nb + C::Dvo;
    constexpr int BWD_DQ_OUTPUT1_BASE = BWD_DQ_OUTPUT_BASE + C::Dvo;
    constexpr bool BWD_DQ_TAIL64_CLUSTER =
        FIXED_FULL_PRODUCTION_PATH && SAVED_BWD_DQ_DUAL_HALVES &&
        C::CLUSTER_SIZE == 2 && C::Dvo == 128 && C::Dqk == 192;
    constexpr int LIVE_DP_DOUT_SC_BASE =
        BWD_DQ_TAIL64_CLUSTER ? (BWD_DQ_OUTPUT1_BASE + C::Dvo / 2) : (BWD_DQ_K1_SC_BASE + 16);
    constexpr int LIVE_DP_V_SC_BASE =
        BWD_DQ_TAIL64_CLUSTER ? (LIVE_DP_DOUT_SC_BASE + 16) : COMPACT_K_SC_BASE;
    static_assert(BWD_DQ_K1_SC_BASE + 16 <= BWD_DQ_OUTPUT_BASE, "fused dQ scale TMEM layout overlaps dQ accumulators");
    static_assert(
        (!BWD_DQ_TAIL64_CLUSTER && LIVE_DP_V_SC_BASE + 16 <= BWD_DQ_OUTPUT_BASE) ||
            (BWD_DQ_TAIL64_CLUSTER &&
             BWD_DQ_OUTPUT1_BASE + C::Dvo / 2 <= LIVE_DP_DOUT_SC_BASE &&
             LIVE_DP_V_SC_BASE + 16 <= BWD_DQ_OUTPUT1_BASE + C::Dvo),
        "live dP scale TMEM layout overlaps active accumulators");
    static_assert(BWD_DQ_OUTPUT1_BASE + C::Dvo <= MAX_TENSOR_COLS, "dual-half fused dQ TMEM layout exceeds budget");
    static_assert(BWD_DQ_OUTPUT1_BASE + C::Dvo / 2 <= MAX_TENSOR_COLS, "tail64 fused dQ TMEM layout exceeds budget");

    const int cta_rank = cluster_ctarank();
    const int stage_cut = (FIXED_DEBUG_STAGE_CUT >= 0) ? FIXED_DEBUG_STAGE_CUT : g.debug_stage_cut;
    const int tiles_m = g.q.rows() / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int num_head = g.q.depth();
    const int num_batch = g.q.batch();
    const bool restrict_to_selected_head = g.live_dp_selected_head >= 0;
    const int head_work = restrict_to_selected_head ? 1 : num_head;
    const int num_hb = head_work * num_batch;
    const int size_one_head = g.k.rows() * (C::Dqk + C::Dvo) * 2;
    constexpr int size_l2 = 50 * 1024 * 1024;
    int swizzle = 1;
    if (size_l2 >= size_one_head) {
        swizzle = 1 << (31 - __clz(size_l2 / size_one_head));
    }
    // Keep the L2 swizzle inside the active batch/head domain; otherwise long
    // contexts with small H can map blocks to out-of-range heads.
    swizzle = min(swizzle, max(num_hb, 1));
    const int num_hb_quotient = num_hb / swizzle;
    const int num_hb_remainder = num_hb - num_hb_quotient * swizzle;
    const int l2_major = swizzle * num_block;
    const bool restrict_to_selected_row_tile = g.selected_row_tile >= 0;
    const int total_bids = restrict_to_selected_row_tile
        ? num_batch * head_work * C::CLUSTER_SIZE
        : num_batch * head_work * num_block * C::CLUSTER_SIZE;

    auto get_tile_idx = [&](int block_idx) -> int3 {
        if (restrict_to_selected_row_tile) {
            const int bidhb_actual = block_idx / C::CLUSTER_SIZE;
            const int b = restrict_to_selected_head ? bidhb_actual : (bidhb_actual / num_head);
            const int h = restrict_to_selected_head ? g.live_dp_selected_head : (bidhb_actual - b * num_head);
            const int selected_row_cluster = g.selected_row_tile / C::CLUSTER_SIZE;
            const int m_tile_base_cluster = selected_row_cluster * C::CLUSTER_SIZE;
            return {b, m_tile_base_cluster, h};
        }
        int cluster_linear = block_idx / C::CLUSTER_SIZE;
        int bidhb = cluster_linear / l2_major;
        int l2_mod = cluster_linear - bidhb * l2_major;
        int m_cluster, bidhb_residual;
        if (bidhb < num_hb_quotient) {
            m_cluster = l2_mod / swizzle;
            bidhb_residual = l2_mod - m_cluster * swizzle;
        } else {
            int divisor = (num_hb_remainder > 0) ? num_hb_remainder : 1;
            m_cluster = l2_mod / divisor;
            bidhb_residual = l2_mod - m_cluster * divisor;
        }
        int bidhb_actual = bidhb * swizzle + bidhb_residual;
        int b = restrict_to_selected_head ? bidhb_actual : (bidhb_actual / num_head);
        int h = restrict_to_selected_head ? g.live_dp_selected_head : (bidhb_actual - b * num_head);
        m_cluster = num_block - 1 - m_cluster;
        int m_tile_base_cluster = m_cluster * (C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE);
        return {b, m_tile_base_cluster, h};
    };
    constexpr bool fixed_fresh3_full =
        FIXED_DEBUG_STAGE_CUT == DEBUG_STAGE_CUT_FRESH3_FULL;
    constexpr bool fixed_full_production_path = FIXED_FULL_PRODUCTION_PATH;
    static_assert(
        !fixed_full_production_path || fixed_fresh3_full,
        "fixed full production path must use DEBUG_STAGE_CUT_FRESH3_FULL");
    static_assert(
        !fixed_full_production_path ||
            (QK_ONLY &&
             SAVED_LSE_BWD_ONLY &&
             SAVED_BWD_LOCAL_STAGE_ONLY &&
             (SAVED_BWD_DQ_DUAL_HALVES || BWD_PAIR_SINGLE_KERNEL) &&
	             ((SAVED_BWD_EMIT_COL_MXFP4 && !SAVED_BWD_EMIT_ROW_MXFP4) ||
	              (SAVED_BWD_EMIT_ROW_MXFP4 && !SAVED_BWD_EMIT_COL_MXFP4) ||
                  (!SAVED_BWD_EMIT_ROW_MXFP4 && !SAVED_BWD_EMIT_COL_MXFP4))),
        "fixed full production path is only valid for fused dQ + at most one dS MXFP4 layout");
    static_assert(
        !LIVE_DP ||
            (SAVED_BWD_LOCAL_STAGE_ONLY &&
             (SAVED_BWD_DQ_DUAL_HALVES || BWD_PAIR_SINGLE_KERNEL) &&
             fixed_full_production_path &&
             C::Dvo == 128 &&
             C::Nb == 128 &&
             C::CLUSTER_SIZE == 2),
        "live dP is currently specialized for single-layout local-stage replay paths");
    static_assert(!LIVE_DP_MODE0_ONLY || LIVE_DP, "mode0-only live dP instantiation requires LIVE_DP");
    static_assert(!LIVE_DP_MODE0_X8 || LIVE_DP_MODE0_ONLY, "x8 live dP TMEM loads require mode0-only live dP");
    auto executed_k_iters = [&](int full_iters_per_task) -> int {
        if constexpr (fixed_fresh3_full) {
            return full_iters_per_task;
        } else {
            return debug_executed_k_iters<C>(g, full_iters_per_task);
        }
    };
    auto physical_k_iter_for = [&](int local_k_iter) -> int {
        if constexpr (fixed_fresh3_full) {
            return local_k_iter;
        } else {
            return debug_physical_k_iter<C>(g, local_k_iter);
        }
    };

    tensor_allocator<1, C::CLUSTER_SIZE> tm_alloc{};
    auto allocate_tmem_tile = [&]<typename TT>(int col_offset) {
        if constexpr (TT::rows == 64) {
            return tm_alloc.template allocate<TT>(0, col_offset);
        } else {
            return tm_alloc.template allocate<TT>(col_offset);
        }
    };
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    typename G::q_tile (&q_smem)[C::NUM_SOFTMAXXERS] = al.allocate<typename G::q_tile, C::NUM_SOFTMAXXERS>();
    typename G::q_sc_smem_storage &q_sc_smem = al.allocate<typename G::q_sc_smem_storage>();
    typename G::k_tile (&k_smem)[C::LOAD_STAGES] = al.allocate<typename G::k_tile, C::LOAD_STAGES>();
    typename G::k_sc_smem_storage &k_sc_smem = al.allocate<typename G::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] = al.allocate<typename G::k_sc_guard_slot, 1>();
    (void)k_sc_guard;
    auto (&p_bf16_scan)[1] = al.allocate<fp4pv_p_scan3d, 1>();
    auto (&scale_dbg_smem)[1] = al.allocate<st_fp8e4m3<128, 32, false>, 1>();
    auto (&p_fp4_stage)[2] = al.allocate<typename G::p_fp4_tile, 2>();
    auto (&p_sc_stage)[2] = al.allocate<typename G::p_sc_tile, 2>();
    typename G::bwd_dq_k_fp4_tile (&bwd_dq_k_smem)[2] = al.allocate<typename G::bwd_dq_k_fp4_tile, 2>();
    typename G::bwd_dq_sc_tile (&bwd_dq_k_sc_smem)[2] = al.allocate<typename G::bwd_dq_sc_tile, 2>();
    typename G::bwd_dq_k_fp4_tile (&bwd_dq_k1_smem)[2] = al.allocate<typename G::bwd_dq_k_fp4_tile, 2>();
    typename G::bwd_dq_sc_tile (&bwd_dq_k1_sc_smem)[2] = al.allocate<typename G::bwd_dq_sc_tile, 2>();
    auto (&bwd_dq_p_dummy)[1] = al.allocate<typename G::p_fp4_tile, 1>();
    typename G::bwd_dq_k_fp4_tile (&bwd_dq_k_dummy)[1] = al.allocate<typename G::bwd_dq_k_fp4_tile, 1>();
    auto (&p_sc_guard)[1] = al.allocate<typename G::p_sc_guard_slot, 1>();
    (void)p_sc_guard;
    typename G::live_dp_dout_fp4_tile *live_dp_dout_smem = nullptr;
    typename G::live_dp_sc_tile *live_dp_dout_sc_smem = nullptr;
    typename G::live_dp_v_row_fp4_tile *live_dp_v_smem = nullptr;
    typename G::live_dp_sc_tile *live_dp_v_sc_smem = nullptr;
    if constexpr (LIVE_DP) {
        auto (&live_dp_dout_alloc)[1] = al.allocate<typename G::live_dp_dout_fp4_tile, 1>();
        auto (&live_dp_dout_sc_alloc)[1] = al.allocate<typename G::live_dp_sc_tile, 1>();
        auto (&live_dp_v_alloc)[2] = al.allocate<typename G::live_dp_v_row_fp4_tile, 2>();
        auto (&live_dp_v_sc_alloc)[2] = al.allocate<typename G::live_dp_sc_tile, 2>();
        live_dp_dout_smem = &live_dp_dout_alloc[0];
        live_dp_dout_sc_smem = &live_dp_dout_sc_alloc[0];
        live_dp_v_smem = &live_dp_v_alloc[0];
        live_dp_v_sc_smem = &live_dp_v_sc_alloc[0];
    }

    __shared__ semaphore q_arrived[1], q_sc_arrived[1], q_finished[1];
    __shared__ semaphore k_arrived[C::LOAD_STAGES], k_sc_arrived[C::K_SCALE_SMEM_SLOTS], k_finished[C::LOAD_STAGES], k_sc_finished[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore scores_arrived[1], scores_arrived_fresh[1], corr_arrived[1], rescale_finished[1], p_copy_done[2], p_pack_ready[2], p_store_done[2];
    __shared__ semaphore bwd_dq_stage_ready[2], bwd_dq_stage_remote_ready[2], bwd_dq_k_remote_ready[2], bwd_dq_issue_ready[2], bwd_dq_src_finished[2], bwd_dq_src1_finished[2], bwd_dq_tmem_ready[1], bwd_dq_accum_done[1];
    __shared__ semaphore live_dp_dout_ready[1], live_dp_dout_remote_ready[1], live_dp_dout_finished[1];
    __shared__ semaphore live_dp_v_ready[2], live_dp_v_remote_ready[2], live_dp_v_finished[2];
    __shared__ semaphore live_dp_arrived[1], live_dp_copy_done[1];
    __shared__ float p_softmax_warp_amax[2][4];
    __shared__ float p_cta_amax[2];

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<typename G::q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<typename G::k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<typename G::k_sc_tile, dim::ROW>();
        init_semaphore(q_arrived[0], 0, 1);
        init_semaphore(q_sc_arrived[0], 0, 1);
        init_semaphore(q_finished[0], 0, 1);
        init_semaphore(scores_arrived[0], 0, 1);
        init_semaphore(scores_arrived_fresh[0], 0, 1);
        init_semaphore(corr_arrived[0], 0, 4);
        init_semaphore(rescale_finished[0], 0, 1);
        init_semaphore(p_copy_done[0], 0, 1);
        init_semaphore(p_copy_done[1], 0, 1);
        init_semaphore(p_pack_ready[0], 0, 1);
        init_semaphore(p_pack_ready[1], 0, 1);
        init_semaphore(p_store_done[0], 0, 1);
        init_semaphore(p_store_done[1], 0, 1);
        init_semaphore(bwd_dq_stage_ready[0], 0, 1);
        init_semaphore(bwd_dq_stage_ready[1], 0, 1);
        init_semaphore(bwd_dq_stage_remote_ready[0], 0, 1);
        init_semaphore(bwd_dq_stage_remote_ready[1], 0, 1);
        init_semaphore(bwd_dq_k_remote_ready[0], 0, 1);
        init_semaphore(bwd_dq_k_remote_ready[1], 0, 1);
        init_semaphore(bwd_dq_issue_ready[0], 0, 1);
        init_semaphore(bwd_dq_issue_ready[1], 0, 1);
        init_semaphore(bwd_dq_src_finished[0], 0, 1);
        init_semaphore(bwd_dq_src_finished[1], 0, 1);
        init_semaphore(bwd_dq_src1_finished[0], 0, 1);
        init_semaphore(bwd_dq_src1_finished[1], 0, 1);
        init_semaphore(bwd_dq_tmem_ready[0], 0, 1);
        init_semaphore(bwd_dq_accum_done[0], 0, 1);
        init_semaphore(live_dp_dout_ready[0], 0, 1);
        init_semaphore(live_dp_dout_remote_ready[0], 0, 1);
        init_semaphore(live_dp_dout_finished[0], 0, 1);
        init_semaphore(live_dp_v_ready[0], 0, 1);
        init_semaphore(live_dp_v_ready[1], 0, 1);
        init_semaphore(live_dp_v_remote_ready[0], 0, 1);
        init_semaphore(live_dp_v_remote_ready[1], 0, 1);
        init_semaphore(live_dp_v_finished[0], 0, 1);
        init_semaphore(live_dp_v_finished[1], 0, 1);
        init_semaphore(live_dp_arrived[0], 0, 1);
        init_semaphore(live_dp_copy_done[0], 0, C::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < C::LOAD_STAGES; ++i) {
            init_semaphore(k_arrived[i], 0, 1);
            init_semaphore(k_finished[i], 0, 1);
            init_semaphore(k_sc_arrived[i], 0, 1);
            init_semaphore(k_sc_finished[i], 0, 1);
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();
    if (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM ||
        stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM) {
        fp4pv_zero_shared_tile_block(bwd_dq_p_dummy[0]);
        fp4pv_zero_shared_tile_block(bwd_dq_k_dummy[0]);
        __syncthreads();
        publish_shared_backing();
        publish_cluster_shared_backing_if_needed<C>();
        fp4pv_cluster_sync<C>();
        __syncthreads();
    }

    auto run_bwd_dq_issue_path = [&]() {
        if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
            if ((g.backward_dq_pad == nullptr && g.backward_dq_bshd == nullptr) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY)) {
                return;
            }
            const bool issue_warp = (cta_rank == 0) && (warpgroup::warpid() == 0);
            const bool issue_lane = issue_warp && warp::elect_leader();
            d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT_BASE);
            d_tt_bwd_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_bwd_p_sc>(BWD_DQ_P_SC_BASE);
            d_tt_bwd_k_sc k_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K_SC_BASE);
            int stage_phase_mask = 0;
            int src_phase_mask = 0;
            int stage_remote_phase_mask = 0;
            int k_remote_phase_mask = 0;
            int tmem_ready_count = 0;
            const bool bwd_dq_issue_tmem_active =
                (stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM);
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                const int m_tile = t_coord.y + cta_rank;
                if (m_tile >= tiles_m) {
                    continue;
                }
                const bool peer_cta_valid = (t_coord.y + 1) < tiles_m;
                const int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                if (issue_lane && bwd_dq_issue_tmem_active) {
                    zero_output_scratch_issue_lane<C>(tt_output);
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const int buf = idx & 1;
                    const int physical_k_iter = physical_k_iter_for(idx);
                    if (idx >= 2) {
                        const int src_phase = (src_phase_mask >> buf) & 1;
                        wait(bwd_dq_src_finished[buf], src_phase);
                        src_phase_mask ^= (1 << buf);
                    }
                    const int stage_phase = (stage_phase_mask >> buf) & 1;
                    wait(bwd_dq_stage_ready[buf], stage_phase);
                    stage_phase_mask ^= (1 << buf);
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            if (peer_cta_valid && (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
                                const int stage_remote_phase = (stage_remote_phase_mask >> buf) & 1;
                                tma::cluster::wait(bwd_dq_stage_remote_ready[buf], stage_remote_phase);
                                stage_remote_phase_mask ^= (1 << buf);
                            }
                        }
                    }
	                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
	                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
	                            warpgroup::arrive(bwd_dq_src_finished[buf]);
	                        }
	                        continue;
                    }
                    if (warpgroup::warpid() == 0) {
                        load_bwd_dq_k_col_fp4_tile<C>(bwd_dq_k_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                        load_bwd_dq_k_col_sc_tile<C>(bwd_dq_k_sc_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter);
                    }
                    warpgroup::sync(warpgroup::groupid() + 1);
                    fp4pv_publish_shared_backing_proxy_only();
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        publish_cluster_shared_backing_if_needed<C>();
                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(bwd_dq_k_remote_ready[buf], 0);
                        }
                    }
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            if (peer_cta_valid) {
                                const int k_remote_phase = (k_remote_phase_mask >> buf) & 1;
                                tma::cluster::wait(bwd_dq_k_remote_ready[buf], k_remote_phase);
                                k_remote_phase_mask ^= (1 << buf);
                            }
                        }
                    }
                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY)) {
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            warpgroup::arrive(bwd_dq_src_finished[buf]);
                        }
                        continue;
                    }
                    if (issue_warp) {
                        auto &p_sc_e8m0 = *reinterpret_cast<typename G::bwd_dq_sc_tile *>(&p_sc_stage[buf]);
                        fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm, p_sc_e8m0);
                        fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm, bwd_dq_k_sc_smem[buf]);
                        tensor_load_wait();
                        if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY)) {
                            tensor_before_thread_sync();
                            if (issue_lane) {
                                if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY)) {
                                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM)) {
                                        auto &dq_p_operand =
                                            ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                             (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM))
                                                ? bwd_dq_p_dummy[0]
                                                : p_fp4_stage[buf];
                                        auto &dq_k_operand =
                                            ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM))
                                                ? bwd_dq_k_dummy[0]
                                                : bwd_dq_k_smem[buf];
                                        if (idx == 0) {
                                            fp4pv_mm_unscaled_nosem_cluster_ABt<C>(tt_output, dq_p_operand, dq_k_operand);
                                        } else {
                                            fp4pv_mma_unscaled_nosem_cluster_ABt<C>(tt_output, dq_p_operand, dq_k_operand);
                                        }
                                    } else if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE)) {
                                        if (idx == 0) {
                                            fp4pv_mm_unscaled_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], bwd_dq_src_finished[buf]);
                                        } else {
                                            fp4pv_mma_unscaled_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], bwd_dq_src_finished[buf]);
                                        }
                                    } else {
                                        if (idx == 0) {
                                            fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
                                        } else {
                                            fp4pv_mma_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
                                        }
                                    }
                                    if ((stage_cut != DEBUG_STAGE_CUT_FRESH3_FULL)) {
                                        tensor_commit<C::CLUSTER_SIZE>(bwd_dq_tmem_ready[0]);
                                        ++tmem_ready_count;
                                    }
                                    if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) &&
                                        (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) &&
                                        idx == iters_per_task - 1) {
                                        detail::tcgen05::commit<C::CLUSTER_SIZE>(bwd_dq_accum_done[0]);
                                    }
                                }
                            }
                        }
                    }
                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY)) {
                        continue;
                    }
                    if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM)) {
                        continue;
                    }
                }
                (void)tmem_ready_count;
            }
        }
    };

    if (warpgroup::groupid() == 3) {
        if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<128>();
        }
        if (warpgroup::warpid() == 3) {
            const bool local_leader = warp::elect_leader();
            int k_idx = 0;
            int k_phase = 1;
            int k_sc_idx = 0;
            int k_sc_phase = 0;
            int q_phase = 1;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                const int q_tile_idx = min(t_coord.y + cta_rank, tiles_m - 1);
                if (local_leader) {
                    if (cta_rank < q_tiles_this_cluster) {
                        const coord<typename G::q_tile> q_coord = {t_coord.x, t_coord.z, q_tile_idx, 0};
                        const coord<typename G::q_sc_tile> q_sc_coord = {t_coord.x, q_tile_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0};
                        fp4pv_wait<C>(q_finished[0], q_phase);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(q_smem[0], g.q, q_coord, q_arrived[0], (uint16_t)(1 << cta_rank), 0);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(q_sc_smem, 0), g.q_sc, q_sc_coord, q_sc_arrived[0], (uint16_t)(1 << cta_rank), 0);
                    }
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const int physical_k_iter = physical_k_iter_for(idx);
                    const int k_tile_idx = min(physical_k_iter * C::CLUSTER_SIZE + cta_rank, tiles_m - 1);
                    if (local_leader) {
                        const coord<typename G::k_tile> k_coord = {t_coord.x, t_coord.z, k_tile_idx, 0};
                        const int k_sc_depth_idx = debug_k_sc_depth_index<C>(g, idx, cta_rank, tiles_m);
                        const coord<typename G::k_sc_tile> k_sc_coord = {t_coord.x, k_sc_depth_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0};
                        fp4pv_wait<C>(k_finished[k_idx], k_phase);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(k_smem[k_idx], g.k, k_coord, k_arrived[k_idx], (uint16_t)(1 << cta_rank), 0);
                        wait(k_sc_finished[k_sc_idx], k_sc_phase ^ 1);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(k_sc_smem, k_sc_idx), g.k_sc, k_sc_coord, k_sc_arrived[k_sc_idx], (uint16_t)(1 << cta_rank), 0);
                    }
                    k_idx++;
                    if (k_idx == C::LOAD_STAGES) {
                        k_idx = 0;
                        k_phase ^= 1;
                    }
                    k_sc_idx++;
                    if (k_sc_idx == C::K_SCALE_SMEM_SLOTS) {
                        k_sc_idx = 0;
                        k_sc_phase ^= 1;
                    }
                }
                q_phase ^= 1;
            }
        }
        if constexpr (LIVE_DP) {
            if constexpr (LIVE_DP_MODE0_ONLY) {
                if (warpgroup::warpid() == 1 && cta_rank != 0) {
                    const bool local_leader = warp::elect_leader();
                    int v_slot = 0;
                    int v_finished_phase_mask = 0;
                    int dout_finished_phase = 0;
                    for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, ++task_num) {
                        int3 t_coord = get_tile_idx(cur_bid);
                        int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                        if (iters_per_task <= 0) {
                            continue;
                        }
                        const int m_tile = t_coord.y + cta_rank;
                        const bool peer_row_valid = m_tile < tiles_m;
                        if (task_num > 0 && local_leader) {
                            wait(live_dp_dout_finished[0], dout_finished_phase);
                            dout_finished_phase ^= 1;
                        }
                        warp::sync();
                        if (peer_row_valid) {
                            load_live_dp_dout_row_fp4_tile_warp<C>(*live_dp_dout_smem, g, t_coord.x, t_coord.z, m_tile);
                            load_live_dp_dout_row_sc_tile_warp<C>(*live_dp_dout_sc_smem, g, t_coord.x, t_coord.z, m_tile);
                        } else {
                            fp4pv_zero_shared_tile_warp(*live_dp_dout_smem);
                            fp4pv_zero_shared_tile_warp(*live_dp_dout_sc_smem);
                        }
                        warp::sync();
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        publish_cluster_shared_backing_if_needed<C>();
                        warp::sync();
                        if (local_leader) {
                            arrive(live_dp_dout_ready[0]);
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if (cta_rank == 1) {
                                    tma::cluster::arrive(live_dp_dout_remote_ready[0], 0);
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int idx = 0; idx < iters_per_task; ++idx) {
                            const int physical_k_iter = physical_k_iter_for(idx);
                            if ((task_num > 0 || idx >= 2) && local_leader) {
                                const int v_finished_phase = (v_finished_phase_mask >> v_slot) & 1;
                                wait(live_dp_v_finished[v_slot], v_finished_phase);
                                v_finished_phase_mask ^= (1 << v_slot);
                            }
                            warp::sync();
                            load_live_dp_v_row_fp4_tile_warp<C>(live_dp_v_smem[v_slot], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                            load_live_dp_v_row_sc_tile_warp<C>(live_dp_v_sc_smem[v_slot], g, t_coord.x, t_coord.z, physical_k_iter);
                            warp::sync();
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            fp4pv_publish_shared_backing_proxy_only();
                            publish_cluster_shared_backing_if_needed<C>();
                            warp::sync();
                            if (local_leader) {
                                arrive(live_dp_v_ready[v_slot]);
                                if constexpr (C::CLUSTER_SIZE > 1) {
                                    if (cta_rank == 1) {
                                        tma::cluster::arrive(live_dp_v_remote_ready[v_slot], 0);
                                    }
                                }
                            }
                            v_slot ^= 1;
                        }
                    }
                }
            } else {
                if (warpgroup::warpid() == 1 && g.live_dp_debug_mode != 2 && cta_rank != 0) {
                    const bool local_leader = warp::elect_leader();
                    const bool local_only_debug =
                        (g.live_dp_debug_mode == 7 || g.live_dp_debug_mode == 8 ||
                         g.live_dp_debug_mode == 9 || g.live_dp_debug_mode == 10 ||
                         g.live_dp_debug_mode == 14 || g.live_dp_debug_mode == 17);
                    int v_slot = 0;
                    int v_finished_phase_mask = 0;
                    int dout_finished_phase = 0;
                    for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, ++task_num) {
                        int3 t_coord = get_tile_idx(cur_bid);
                        int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                        if (iters_per_task <= 0) {
                            continue;
                        }
                        const int m_tile = t_coord.y + cta_rank;
                        const bool peer_row_valid = m_tile < tiles_m;
                        const bool prepare_dout_source =
                            g.live_dp_debug_mode != 3 && g.live_dp_debug_mode != 9 &&
                            g.live_dp_debug_mode != 10;
                        if (task_num > 0 && prepare_dout_source && local_leader) {
                            wait(live_dp_dout_finished[0], dout_finished_phase);
                            dout_finished_phase ^= 1;
                        }
                        warp::sync();
                        if (peer_row_valid && prepare_dout_source) {
                            load_live_dp_dout_row_fp4_tile_warp<C>(*live_dp_dout_smem, g, t_coord.x, t_coord.z, m_tile);
                            load_live_dp_dout_row_sc_tile_warp<C>(*live_dp_dout_sc_smem, g, t_coord.x, t_coord.z, m_tile);
                        } else if (prepare_dout_source) {
                            fp4pv_zero_shared_tile_warp(*live_dp_dout_smem);
                            fp4pv_zero_shared_tile_warp(*live_dp_dout_sc_smem);
                        }
                        if (prepare_dout_source) {
                            warp::sync();
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            if (!local_only_debug) {
                                publish_cluster_shared_backing_if_needed<C>();
                            }
                            warp::sync();
                        }
                        if (prepare_dout_source && local_leader) {
                            arrive(live_dp_dout_ready[0]);
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if (!local_only_debug && cta_rank == 1) {
                                    tma::cluster::arrive(live_dp_dout_remote_ready[0], 0);
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int idx = 0; idx < iters_per_task; ++idx) {
                            const int physical_k_iter = physical_k_iter_for(idx);
                            if ((task_num > 0 || idx >= 2) && local_leader) {
                                const int v_finished_phase = (v_finished_phase_mask >> v_slot) & 1;
                                wait(live_dp_v_finished[v_slot], v_finished_phase);
                                v_finished_phase_mask ^= (1 << v_slot);
                            }
                            warp::sync();
                            if (g.live_dp_debug_mode != 4 && g.live_dp_debug_mode != 9) {
                                if (g.live_dp_debug_mode != 5 && g.live_dp_debug_mode != 7 &&
                                    g.live_dp_debug_mode != 10) {
                                    load_live_dp_v_row_fp4_tile_warp<C>(live_dp_v_smem[v_slot], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                                }
                                if (g.live_dp_debug_mode != 6 && g.live_dp_debug_mode != 8) {
                                    load_live_dp_v_row_sc_tile_warp<C>(live_dp_v_sc_smem[v_slot], g, t_coord.x, t_coord.z, physical_k_iter);
                                }
                            }
                            if (g.live_dp_debug_mode == 14) {
                                store_live_dp_source_debug_warp<C>(
                                    g, t_coord.x, t_coord.z, m_tile, physical_k_iter, cta_rank);
                            }
                            warp::sync();
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            fp4pv_publish_shared_backing_proxy_only();
                            if (!local_only_debug) {
                                publish_cluster_shared_backing_if_needed<C>();
                            }
                            warp::sync();
                            if (local_leader) {
                                arrive(live_dp_v_ready[v_slot]);
                                if constexpr (C::CLUSTER_SIZE > 1) {
                                    if (!local_only_debug && cta_rank == 1) {
                                        tma::cluster::arrive(live_dp_v_remote_ready[v_slot], 0);
                                    }
                                }
                            }
                            v_slot ^= 1;
                        }
                    }
                }
            }
        }
        if (warpgroup::warpid() == 0 && cta_rank == 0) {
            const bool issue_lane = warp::elect_leader();
            d_tt_scores tt_score = tm_alloc.template allocate<d_tt_scores>(0);
            constexpr int COMPACT_Q_SC_BASE = C::Nb;
            constexpr int COMPACT_K_SC_BASE = C::Nb + C::Q_SC_TMEM_WIDTH;
            constexpr int COMPACT_Q_SC_DBG_BASE = COMPACT_K_SC_BASE + 16;
            constexpr int COMPACT_K_SC_DBG_BASE = COMPACT_Q_SC_DBG_BASE + 32;
            constexpr int PRODUCTION_LIKE_Q_SC_BASE = C::Nb + C::Dvo;
            constexpr int PRODUCTION_LIKE_K_SC_BASE = C::Nb + C::Dvo + C::Q_SC_TMEM_WIDTH;
            constexpr int PRODUCTION_LIKE_Q_SC_DBG_BASE = PRODUCTION_LIKE_K_SC_BASE + 16;
            constexpr int PRODUCTION_LIKE_K_SC_DBG_BASE = PRODUCTION_LIKE_Q_SC_DBG_BASE + 32;
            static_assert(COMPACT_K_SC_DBG_BASE + 32 <= MAX_TENSOR_COLS, "debug compact TMEM layout exceeds TMEM budget");
            static_assert(PRODUCTION_LIKE_K_SC_DBG_BASE + 32 <= MAX_TENSOR_COLS, "debug production-like TMEM layout exceeds TMEM budget");
            const bool production_like_tmem_layout = g.debug_tmem_layout_mode == DEBUG_TMEM_LAYOUT_PRODUCTION_LIKE;
            d_tt_q_sc q_sc_tm = production_like_tmem_layout
                ? tm_alloc.template allocate<d_tt_q_sc>(PRODUCTION_LIKE_Q_SC_BASE)
                : tm_alloc.template allocate<d_tt_q_sc>(COMPACT_Q_SC_BASE);
            d_tt_sc16 k_sc_tm = production_like_tmem_layout
                ? tm_alloc.template allocate<d_tt_sc16>(PRODUCTION_LIKE_K_SC_BASE)
                : tm_alloc.template allocate<d_tt_sc16>(COMPACT_K_SC_BASE);
            d_tt_bwd_p_sc live_dp_dout_sc_tm = tm_alloc.template allocate<d_tt_bwd_p_sc>(LIVE_DP_DOUT_SC_BASE);
            d_tt_bwd_k_sc live_dp_v_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(LIVE_DP_V_SC_BASE);
            full_tt_fp8e4m3<32> q_sc_tm_dbg = production_like_tmem_layout
                ? tm_alloc.template allocate<full_tt_fp8e4m3<32>>(PRODUCTION_LIKE_Q_SC_DBG_BASE)
                : tm_alloc.template allocate<full_tt_fp8e4m3<32>>(COMPACT_Q_SC_DBG_BASE);
            full_tt_fp8e4m3<32> k_sc_tm_dbg = production_like_tmem_layout
                ? tm_alloc.template allocate<full_tt_fp8e4m3<32>>(PRODUCTION_LIKE_K_SC_DBG_BASE)
                : tm_alloc.template allocate<full_tt_fp8e4m3<32>>(COMPACT_K_SC_DBG_BASE);
            auto q_sc_tm_chunk = q_sc_tm.template subtile<full_tt_fp8e4m3<16>>(0);
            int k_idx = 0, k_phase = 0, k_sc_slot = 0;
            int k_sc_arrived_phase_mask = 0;
	            int current_b = 0;
	            int current_h = 0;
	            int current_m_tile = 0;
			            bool current_peer_cta_valid = false;
			            bool current_peer_live_dout_source_ready = false;
			            bool current_peer_live_v_valid = false;
            int live_dp_dout_ready_phase = 0;
            int live_dp_v_ready_phase_mask = 0;
            int live_dp_v_debug_done_phase_mask = 0;
            int live_dp_copy_phase = 0;
            // The debug softmax consumer serializes score-tile extraction through one live
            // reuse slot, so the producer must also treat p_copy_done as a single-buffer
            // handshake and wait for the first score copy before reissuing QK.
            int p_copy_phase_mask = 1;
            auto dump_q_sc_tmem_chunk = [&](int b, int h, int m_tile, int chunk) {
                if (g.debug_q_sc_tmem_loaded == nullptr) {
                    return;
                }
                auto q_sc_tm_dbg_lo = q_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(0);
                auto q_sc_tm_dbg_hi = q_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(16);
                if (issue_lane) {
                    load_q_scale_chunk<C>(q_sc_tm_dbg_lo, slot_tile_at(q_sc_smem, 0), chunk);
                    load_q_scale_chunk<C>(q_sc_tm_dbg_hi, slot_tile_at(q_sc_smem, 0), chunk);
                }
                __syncwarp();
                copy_scale_tt_to_shared(scale_dbg_smem[0], q_sc_tm_dbg);
                __syncwarp();
                if (issue_lane) {
                    store_debug_q_sc_tmem_chunk_if_needed<C>(g, scale_dbg_smem[0], b, h, m_tile, chunk);
                }
                __syncwarp();
            };
            auto dump_k_sc_tmem_chunk = [&](int b, int h, int k_iter, int chunk) {
                if (g.debug_k_sc_tmem_loaded == nullptr) {
                    return;
                }
                auto k_sc_tm_dbg_lo = k_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(0);
                auto k_sc_tm_dbg_hi = k_sc_tm_dbg.template subtile<full_tt_fp8e4m3<16>>(16);
                if (issue_lane) {
                    load_k_scale_chunk<C>(k_sc_tm_dbg_lo, slot_tile_at(k_sc_smem, k_sc_slot), chunk);
                    load_k_scale_chunk<C>(k_sc_tm_dbg_hi, slot_tile_at(k_sc_smem, k_sc_slot), chunk);
                }
                __syncwarp();
                copy_scale_tt_to_shared(scale_dbg_smem[0], k_sc_tm_dbg);
                __syncwarp();
                if (issue_lane) {
                    store_debug_k_sc_tmem_chunk_if_needed<C>(g, scale_dbg_smem[0], b, h, k_iter, chunk);
                }
                __syncwarp();
            };
            auto issue_live_dp_after_score_copy = [&](int prev_idx, bool final_live_dp_for_task) {
                if constexpr (LIVE_DP) {
                    if constexpr (LIVE_DP_MODE0_ONLY) {
                        const int v_slot = prev_idx & 1;
                        const int prev_physical_k_iter = physical_k_iter_for(prev_idx);
                        if (issue_lane) {
                            const int qk_done_slot = prev_idx % C::LOAD_STAGES;
                            const int qk_done_phase = (prev_idx / C::LOAD_STAGES) & 1;
                            fp4pv_wait<C>(k_finished[qk_done_slot], qk_done_phase);
                        }
                        if (prev_idx == 0) {
                            load_live_dp_dout_row_fp4_tile_warp<C>(
                                *live_dp_dout_smem, g, current_b, current_h, current_m_tile);
                            load_live_dp_dout_row_sc_tile_warp<C>(
                                *live_dp_dout_sc_smem, g, current_b, current_h, current_m_tile);
                        }
                        load_live_dp_v_row_fp4_tile_warp<C>(
                            live_dp_v_smem[v_slot], g, current_b, current_h,
                            prev_physical_k_iter, cta_rank);
                        load_live_dp_v_row_sc_tile_warp<C>(
                            live_dp_v_sc_smem[v_slot], g, current_b, current_h,
                            prev_physical_k_iter);
                        __syncwarp(__activemask());
                        __threadfence_block();
                        fp4pv_publish_shared_backing_proxy_only();
                        publish_cluster_shared_backing_if_needed<C>();
                        if (prev_idx == 0) {
                            if (issue_lane) {
                                if constexpr (C::CLUSTER_SIZE > 1) {
                                    if (current_peer_live_dout_source_ready) {
                                        tma::cluster::wait(live_dp_dout_remote_ready[0], live_dp_dout_ready_phase);
                                    }
                                }
                                live_dp_dout_ready_phase ^= 1;
                            }
                        }
                        if (issue_lane) {
                            const int v_ready_phase = (live_dp_v_ready_phase_mask >> v_slot) & 1;
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if (current_peer_live_v_valid) {
                                    tma::cluster::wait(live_dp_v_remote_ready[v_slot], v_ready_phase);
                                }
                            }
                            live_dp_v_ready_phase_mask ^= (1 << v_slot);
                        }
                        __syncwarp(__activemask());
                        if (issue_lane) {
                            if (prev_idx == 0) {
                                fp4pv_load_mxnv_scale_async_cluster<C>(live_dp_dout_sc_tm, *live_dp_dout_sc_smem);
                                tensor_load_wait();
                            }
                            fp4pv_load_mxnv_scale_async_cluster<C>(live_dp_v_sc_tm, live_dp_v_sc_smem[v_slot]);
                            tensor_load_wait();
                            tensor_before_thread_sync();
                            fp4pv_mm_cluster_ABt<C>(
                                tt_score,
                                *live_dp_dout_smem,
                                live_dp_v_smem[v_slot],
                                live_dp_dout_sc_tm,
                                live_dp_v_sc_tm,
                                live_dp_v_finished[v_slot]);
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(live_dp_arrived[0]);
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if (!current_peer_cta_valid) {
                                    arrive(live_dp_copy_done[0]);
                                }
                            }
                            wait(live_dp_copy_done[0], live_dp_copy_phase);
                            live_dp_copy_phase ^= 1;
                            if (final_live_dp_for_task) {
                                arrive(live_dp_dout_finished[0]);
                                fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
                            }
                        }
                        return;
                    }
                    if (g.live_dp_debug_mode == 2) {
                        if (issue_lane && final_live_dp_for_task) {
                            arrive(live_dp_dout_finished[0]);
                            fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
                        }
                        return;
                    }
                    const bool local_only_debug =
                        (g.live_dp_debug_mode == 7 || g.live_dp_debug_mode == 8 ||
                         g.live_dp_debug_mode == 9 || g.live_dp_debug_mode == 10 ||
                         g.live_dp_debug_mode == 14 || g.live_dp_debug_mode == 17);
                    const bool skip_live_ready_waits = (g.live_dp_debug_mode == 12);
                    const int v_slot = prev_idx & 1;
                    const bool skip_live_self_stage =
                        skip_live_ready_waits || g.live_dp_debug_mode == 9;
                    const int prev_physical_k_iter = physical_k_iter_for(prev_idx);
                    if (issue_lane) {
                        const int qk_done_slot = prev_idx % C::LOAD_STAGES;
                        const int qk_done_phase = (prev_idx / C::LOAD_STAGES) & 1;
                        fp4pv_wait<C>(k_finished[qk_done_slot], qk_done_phase);
                    }
                    if (!skip_live_self_stage) {
                        if (prev_idx == 0 &&
                            g.live_dp_debug_mode != 3 && g.live_dp_debug_mode != 9 &&
                            g.live_dp_debug_mode != 10) {
                            load_live_dp_dout_row_fp4_tile_warp<C>(
                                *live_dp_dout_smem, g, current_b, current_h, current_m_tile);
                            load_live_dp_dout_row_sc_tile_warp<C>(
                                *live_dp_dout_sc_smem, g, current_b, current_h, current_m_tile);
                        }
                        if (g.live_dp_debug_mode != 4 && g.live_dp_debug_mode != 9) {
                            if (g.live_dp_debug_mode != 5 && g.live_dp_debug_mode != 7 &&
                                g.live_dp_debug_mode != 10) {
                                load_live_dp_v_row_fp4_tile_warp<C>(
                                    live_dp_v_smem[v_slot], g, current_b, current_h,
                                    prev_physical_k_iter, cta_rank);
                            }
                            if (g.live_dp_debug_mode != 6 && g.live_dp_debug_mode != 8) {
                                load_live_dp_v_row_sc_tile_warp<C>(
                                    live_dp_v_sc_smem[v_slot], g, current_b, current_h,
                                    prev_physical_k_iter);
                            }
                        }
                        if (g.live_dp_debug_mode == 14) {
                            store_live_dp_source_debug_warp<C>(
                                g, current_b, current_h, current_m_tile, prev_physical_k_iter, cta_rank);
                        }
	                        __syncwarp(__activemask());
	                        __threadfence_block();
	                        fp4pv_publish_shared_backing_proxy_only();
                            if (!local_only_debug) {
                                publish_cluster_shared_backing_if_needed<C>();
                            }
	                        if (g.live_dp_debug_mode == 17) {
	                            store_live_dp_shared_debug_warp<C>(
                                g, *live_dp_dout_smem, *live_dp_dout_sc_smem,
                                live_dp_v_smem[v_slot], live_dp_v_sc_smem[v_slot],
                                current_b, current_h, current_m_tile);
                        }
                    }
			                    if (prev_idx == 0) {
		                        if (issue_lane && !skip_live_ready_waits) {
		                            if constexpr (C::CLUSTER_SIZE > 1) {
				                                if (!local_only_debug && current_peer_live_dout_source_ready) {
			                                    tma::cluster::wait(live_dp_dout_remote_ready[0], live_dp_dout_ready_phase);
			                                }
	                            }
	                            live_dp_dout_ready_phase ^= 1;
		                        }
		                    }
			                    if (issue_lane) {
			                        const int v_ready_phase = (live_dp_v_ready_phase_mask >> v_slot) & 1;
			                        if (!skip_live_ready_waits) {
			                            if constexpr (C::CLUSTER_SIZE > 1) {
				                                if (!local_only_debug && current_peer_live_v_valid) {
				                                    tma::cluster::wait(live_dp_v_remote_ready[v_slot], v_ready_phase);
				                                }
			                            }
			                            live_dp_v_ready_phase_mask ^= (1 << v_slot);
			                        }
	                                }
	                                __syncwarp(__activemask());
                                __syncwarp(__activemask());
				                    if (issue_lane) {
			                        if (g.live_dp_debug_mode == 1 || g.live_dp_debug_mode == 3 ||
			                            g.live_dp_debug_mode == 4 || g.live_dp_debug_mode == 5 ||
			                            g.live_dp_debug_mode == 6 || g.live_dp_debug_mode == 7 ||
			                            g.live_dp_debug_mode == 8 || g.live_dp_debug_mode == 9 ||
			                            g.live_dp_debug_mode == 10 || g.live_dp_debug_mode == 12 ||
	                                    g.live_dp_debug_mode == 14 || g.live_dp_debug_mode == 17) {
		                                arrive(live_dp_v_finished[v_slot]);
		                                fp4pv_remote_arrive_if_needed<C>(live_dp_v_finished[v_slot], 1);
		                                if (final_live_dp_for_task) {
		                                    arrive(live_dp_dout_finished[0]);
		                                    fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
			                            }
			                            return;
			                        }
			                        if (g.live_dp_debug_mode == 13) {
			                            tensor_before_thread_sync();
				                        fp4pv_mm_unscaled_cluster_ABt<C>(
			                                tt_score,
			                                *live_dp_dout_smem,
			                                live_dp_v_smem[v_slot],
			                                live_dp_v_finished[v_slot]);
			                        } else {
				                            constexpr uint32_t packed_unit_scale = 0x7f7f7f7fu;
                                            const bool unit_dout_scale =
                                                g.live_dp_debug_mode == 18 || g.live_dp_debug_mode == 19;
                                            const bool unit_v_scale =
                                                g.live_dp_debug_mode == 18 || g.live_dp_debug_mode == 20;
                                            if (unit_dout_scale) {
                                                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(live_dp_dout_sc_tm, packed_unit_scale);
                                            } else {
                                                fp4pv_load_mxnv_scale_async_cluster<C>(live_dp_dout_sc_tm, *live_dp_dout_sc_smem);
                                                tensor_load_wait();
                                            }
                                            if (unit_v_scale) {
                                                fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(live_dp_v_sc_tm, packed_unit_scale);
                                            } else {
                                                fp4pv_load_mxnv_scale_async_cluster<C>(live_dp_v_sc_tm, live_dp_v_sc_smem[v_slot]);
                                                tensor_load_wait();
                                            }
                                            if (unit_dout_scale || unit_v_scale) {
                                                fp4pv_tmem_store_wait();
                                            }
				                            tensor_before_thread_sync();
			                            fp4pv_mm_cluster_ABt<C>(
			                                tt_score,
			                                *live_dp_dout_smem,
			                                live_dp_v_smem[v_slot],
			                                live_dp_dout_sc_tm,
			                                live_dp_v_sc_tm,
			                                live_dp_v_finished[v_slot]);
			                        }
                                if (g.live_dp_debug_mode == 15) {
                                    const int done_phase = (live_dp_v_debug_done_phase_mask >> v_slot) & 1;
                                    wait(live_dp_v_finished[v_slot], done_phase);
                                    live_dp_v_debug_done_phase_mask ^= (1 << v_slot);
                                    if (g.backward_dp_bf16 != nullptr) {
                                        const int heads = g.q.depth();
                                        const int seqlen = g.q.rows();
                                        const int row = current_m_tile * C::Mb + warp::laneid();
                                        const int col_base = prev_physical_k_iter * C::Nb;
                                        const int64_t dp_row_base =
                                            (((static_cast<int64_t>(current_b) * heads + current_h) * seqlen) + row) * seqlen;
                                        #pragma unroll
                                        for (int qid = 0; qid < 4; ++qid) {
                                            float2 dp_chunk[16];
                                            fp4pv_load_score_quarter_tmem(dp_chunk, tt_score.addr, qid);
                                            if (g.live_dp_debug_mode != 13) {
                                                constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                                const float2 alpha_2 = {MXFP4_ALPHA, MXFP4_ALPHA};
                                                #pragma unroll
                                                for (int jj = 0; jj < 16; ++jj) {
                                                    dp_chunk[jj] = __fmul2_rn(dp_chunk[jj], alpha_2);
                                                }
                                            }
                                            #pragma unroll
                                            for (int jj = 0; jj < 16; ++jj) {
                                                const int col = col_base + (qid * 16 + jj) * 2;
                                                const bf16_2 dp_pair_bf = __float22bfloat162_rn(dp_chunk[jj]);
                                                *reinterpret_cast<bf16_2 *>(&g.backward_dp_bf16[dp_row_base + col]) = dp_pair_bf;
                                            }
                                        }
                                    }
                                    if (final_live_dp_for_task) {
                                        arrive(live_dp_dout_finished[0]);
                                        fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
                                    }
                                    return;
                                }
		                        detail::tcgen05::commit<C::CLUSTER_SIZE>(live_dp_arrived[0]);
		                        if constexpr (C::CLUSTER_SIZE > 1) {
		                            if (!current_peer_cta_valid) {
		                                arrive(live_dp_copy_done[0]);
		                            }
		                        }
		                        wait(live_dp_copy_done[0], live_dp_copy_phase);
		                        live_dp_copy_phase ^= 1;
		                        if (final_live_dp_for_task) {
		                            arrive(live_dp_dout_finished[0]);
		                            fp4pv_remote_arrive_if_needed<C>(live_dp_dout_finished[0], 1);
		                        }
		                    }
	                    (void)prev_physical_k_iter;
	                }
            };
            auto issue_next_qk = [&](bool final_q_for_task, int copy_buf, int k_iter, int iters_per_task) -> bool {
                (void)copy_buf;
                constexpr bool ENABLE_DEBUG_STAGE_CUTS = QK_ONLY && !SAVED_LSE_BWD_ONLY;
                const bool producer_third_issue_cut =
                    ENABLE_DEBUG_STAGE_CUTS &&
                    g.max_k_iters == 3 &&
                    final_q_for_task &&
                    stage_cut >= DEBUG_STAGE_CUT_PRODUCER3_POST_WAIT &&
                    stage_cut <= DEBUG_STAGE_CUT_FRESH3_FULL;
                constexpr int reuse_buf = 0;
                const int copy_phase = (p_copy_phase_mask >> reuse_buf) & 1;
                if (issue_lane) {
                    wait(p_copy_done[reuse_buf], copy_phase);
                }
                p_copy_phase_mask ^= (1 << reuse_buf);
                issue_live_dp_after_score_copy(k_iter - 1, false);
                if (producer_third_issue_cut && stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_WAIT) {
                    return true;
                }
                const int next_k_slot = k_idx;
                if (issue_lane) {
                    fp4pv_expect_bytes<C>(k_arrived[next_k_slot], C::CLUSTER_SIZE * sizeof(typename G::k_tile));
                    fp4pv_wait<C>(k_arrived[next_k_slot], k_phase);
                    fp4pv_expect_bytes<C>(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(typename G::k_sc_tile));
                }
                {
                    const int k_sc_phase = (k_sc_arrived_phase_mask >> k_sc_slot) & 1;
                    if (issue_lane) {
                        fp4pv_wait<C>(k_sc_arrived[k_sc_slot], k_sc_phase);
                    }
                    k_sc_arrived_phase_mask ^= (1 << k_sc_slot);
                }
                if (issue_lane) {
                    const int physical_k_iter = physical_k_iter_for(k_iter);
                    if constexpr (!fixed_fresh3_full) {
                        store_debug_k_sc_tile_if_needed<C>(g, slot_tile_at(k_sc_smem, k_sc_slot), current_b, current_h, physical_k_iter);
                    }
                }
                if constexpr (!fixed_fresh3_full) {
                    #pragma unroll
                    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                        if (debug_qk_chunk_enabled<C>(g, chunk)) {
                            dump_k_sc_tmem_chunk(current_b, current_h, physical_k_iter_for(k_iter), chunk);
                        }
                    }
                }
                if (issue_lane) {
                    if constexpr (fixed_fresh3_full) {
                        fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
                            tt_score, q_smem[0], k_smem[next_k_slot],
                            q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm,
                            k_finished[next_k_slot]);
                    } else {
                        if (debug_use_q_scale_tmem(g)) {
                            issue_qk_chunked_qsc_tmem_debug_masked<C>(tt_score, q_smem[0], k_smem[next_k_slot], q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[next_k_slot], g);
                        } else {
                            issue_qk_chunked_debug_masked<C>(tt_score, q_smem[0], k_smem[next_k_slot], slot_tile_at(q_sc_smem, 0), slot_tile_at(k_sc_smem, k_sc_slot), q_sc_tm_chunk, k_sc_tm, k_finished[next_k_slot], g);
                        }
                    }
                }
                if (producer_third_issue_cut && stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_ISSUE) {
                    return true;
                }
                const bool skip_scores_commit =
                    producer_third_issue_cut &&
                    stage_cut == DEBUG_STAGE_CUT_PRODUCER3_SKIP_COMMIT_TAIL;
                const bool fresh_scores_commit =
                    producer_third_issue_cut &&
                    (stage_cut == DEBUG_STAGE_CUT_PRODUCER3_FRESH_COMMIT_TAIL ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_WAIT_ONLY ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_COPY ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL);
                if (issue_lane) {
                    if (fresh_scores_commit) {
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived_fresh[0]);
                    } else if (!skip_scores_commit) {
                        if constexpr (SAVED_LSE_BWD_ONLY) {
                            if (k_iter & 1) {
                                detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived_fresh[0]);
                            } else {
                                detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                            }
                        } else {
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(
                                debug_scores_arrived_sem_for_iter<C>(scores_arrived, scores_arrived_fresh, g, k_iter)
                            );
                        }
                    }
                }
                if (producer_third_issue_cut && stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_COMMIT) {
                    return true;
                }
                if (issue_lane) {
                    arrive(k_sc_finished[k_sc_slot]);
                    fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1);
                }
                if (producer_third_issue_cut && stage_cut == DEBUG_STAGE_CUT_PRODUCER3_POST_KSC_FINISH) {
                    return true;
                }
                if (issue_lane && final_q_for_task) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
                }
                const bool producer_fresh_tail_cut =
                    ENABLE_DEBUG_STAGE_CUTS &&
                    stage_cut == DEBUG_STAGE_CUT_PRODUCER3_FRESH_COMMIT_TAIL;
                if (skip_scores_commit || producer_fresh_tail_cut) {
                    return true;
                }
                k_idx ^= 1;
                if (k_idx == 0) {
                    k_phase ^= 1;
                }
                k_sc_slot ^= 1;
                return false;
            };

            for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, ++task_num) {
                int3 t_coord = get_tile_idx(cur_bid);
                current_b = t_coord.x;
	                current_h = t_coord.z;
	                current_m_tile = t_coord.y;
			                current_peer_cta_valid = (t_coord.y + 1) < tiles_m;
			                current_peer_live_dout_source_ready =
			                    g.live_dp_debug_mode != 3 && g.live_dp_debug_mode != 9 &&
			                    g.live_dp_debug_mode != 10;
			                current_peer_live_v_valid = true;
	                int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                int q_phase = task_num & 1;
                if (issue_lane) {
                    fp4pv_expect_bytes<C>(q_arrived[0], q_tiles_this_cluster * sizeof(typename G::q_tile));
                    fp4pv_wait<C>(q_arrived[0], q_phase);
                    fp4pv_expect_bytes<C>(q_sc_arrived[0], q_tiles_this_cluster * sizeof(typename G::q_sc_tile));
                    fp4pv_wait<C>(q_sc_arrived[0], q_phase);
                    if constexpr (!fixed_fresh3_full) {
                        store_debug_q_sc_tile_if_needed<C>(g, slot_tile_at(q_sc_smem, 0), current_b, current_h, t_coord.y);
                    }
                    if constexpr (fixed_fresh3_full) {
                        fp4pv_stage_q_scale_tmem_cluster<C>(q_sc_tm, slot_tile_at(q_sc_smem, 0));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                    } else if (debug_use_q_scale_tmem(g)) {
                        fp4pv_stage_q_scale_tmem_cluster<C>(q_sc_tm, slot_tile_at(q_sc_smem, 0));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                    }
                }
                if constexpr (!fixed_fresh3_full) {
                    #pragma unroll
                    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                        if (debug_qk_chunk_enabled<C>(g, chunk)) {
                            dump_q_sc_tmem_chunk(current_b, current_h, t_coord.y, chunk);
                        }
                    }
                }

                int k_slot = k_idx;
                if (issue_lane) {
                    fp4pv_expect_bytes<C>(k_arrived[k_slot], C::CLUSTER_SIZE * sizeof(typename G::k_tile));
                    fp4pv_wait<C>(k_arrived[k_slot], k_phase);
                    fp4pv_expect_bytes<C>(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(typename G::k_sc_tile));
                }
                {
                    const int k_sc_phase = (k_sc_arrived_phase_mask >> k_sc_slot) & 1;
                    if (issue_lane) {
                        fp4pv_wait<C>(k_sc_arrived[k_sc_slot], k_sc_phase);
                    }
                    k_sc_arrived_phase_mask ^= (1 << k_sc_slot);
                }
                if (issue_lane) {
                    if constexpr (!fixed_fresh3_full) {
                        store_debug_k_sc_tile_if_needed<C>(g, slot_tile_at(k_sc_smem, k_sc_slot), t_coord.x, t_coord.z, physical_k_iter_for(0));
                    }
                }
                if constexpr (!fixed_fresh3_full) {
                    #pragma unroll
                    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
                        if (debug_qk_chunk_enabled<C>(g, chunk)) {
                            dump_k_sc_tmem_chunk(t_coord.x, t_coord.z, physical_k_iter_for(0), chunk);
                        }
                    }
                }
	                if (issue_lane) {
	                    if constexpr (fixed_fresh3_full) {
	                        fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
	                            tt_score, q_smem[0], k_smem[k_slot],
	                            q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm,
	                            k_finished[k_slot]);
	                    } else {
	                        if (debug_use_q_scale_tmem(g)) {
	                            issue_qk_chunked_qsc_tmem_debug_masked<C>(tt_score, q_smem[0], k_smem[k_slot], q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[k_slot], g);
	                        } else {
                                issue_qk_chunked_debug_masked<C>(tt_score, q_smem[0], k_smem[k_slot], slot_tile_at(q_sc_smem, 0), slot_tile_at(k_sc_smem, k_sc_slot), q_sc_tm_chunk, k_sc_tm, k_finished[k_slot], g);
                            }
                    }
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                    arrive(k_sc_finished[k_sc_slot]);
                    fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1);
	                    if (iters_per_task == 1) {
	                        detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
	                    }
	                }
	                k_idx ^= 1;
                if (k_idx == 0) {
                    k_phase ^= 1;
                }
                k_sc_slot ^= 1;

                bool stop_after_producer_stage_cut = false;
                if (iters_per_task > 1) {
                    stop_after_producer_stage_cut = issue_next_qk(iters_per_task == 2, 0, 1, iters_per_task);
                    for (int idx = 1; idx < iters_per_task - 1; ++idx) {
                        if (stop_after_producer_stage_cut) {
                            break;
                        }
                        stop_after_producer_stage_cut = issue_next_qk(idx == iters_per_task - 2, idx & 1, idx + 1, iters_per_task);
                    }
                }
                if (stop_after_producer_stage_cut) {
                    break;
                }
                if constexpr (LIVE_DP) {
                    const int reuse_buf = 0;
                    const int copy_phase = (p_copy_phase_mask >> reuse_buf) & 1;
                    if (issue_lane) {
                        wait(p_copy_done[reuse_buf], copy_phase);
                    }
                    p_copy_phase_mask ^= (1 << reuse_buf);
                    issue_live_dp_after_score_copy(iters_per_task - 1, true);
                }
            }
        }
    } else if (warpgroup::groupid() == 1) {
        warpgroup::increase_registers<168>();
        d_tt_scores tt_scores = tm_alloc.template allocate<d_tt_scores>(0);
        int scores_phase = 0;
        int rescale_phase = 1;
        int bwd_dq_src_phase_mask = 0;
        int bwd_dq_src1_phase_mask = 0;
        int live_dp_phase = 0;
        // The dump-only path is debug-focused, so we serialize P-stage reuse through a single
        // buffer to avoid multi-iteration phase hazards that do not exist in the fused kernel.
            int p_store_phase = 1;
            uint32_t score_tt_base = tt_scores.addr + ((warpgroup::warpid() * 32) << 16);
            auto arrive_live_dp_copy_done = [&]() {
                // This releases the producer to reuse tt_scores for live dP. If this warp
                // loaded live dP from TMEM, call tensor_load_wait() before this handoff.
                warpgroup::sync(warpgroup::groupid() + 1);
                if constexpr (C::CLUSTER_SIZE > 1) {
                    if (cta_rank == 0) {
                        warpgroup::arrive(live_dp_copy_done[0]);
                    } else if (cta_rank == 1) {
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(live_dp_copy_done[0], 0);
                        }
                    }
                } else {
                    warpgroup::arrive(live_dp_copy_done[0]);
                }
            };
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                int producer_iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (producer_iters_per_task <= 0) {
                    continue;
                }
                int m_tile = t_coord.y + cta_rank;
                if (m_tile >= tiles_m) {
                    continue;
                }
                if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY && SAVED_BWD_EMIT_COL_MXFP4) {
                    zero_backward_ds_col_mxfp4_future_scale_tiles<C>(
                        g, t_coord.x, t_coord.z, m_tile);
                }
                if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY && SAVED_BWD_EMIT_ROW_MXFP4) {
                    if (!g.backward_skip_global_ds_row_store) {
                        zero_backward_ds_row_mxfp4_future_scale_tiles<C>(
                            g, t_coord.x, t_coord.z, m_tile);
                    }
                }
                int consumer_iters_per_task = SAVED_LSE_BWD_ONLY
                    ? producer_iters_per_task
                    : debug_consumer_k_iters<C>(g, producer_iters_per_task);
            const float SCALE_LOG2 = qk_scale_log2(g, t_coord.x, t_coord.z);
            const bool ZERO_SCALE = (SCALE_LOG2 == 0.0f);
            float row_sum = 0.0f;
            float row_max = base_types::constants<float>::neg_infty();
            bool stop_after_stage_cut = false;
            const bool bwd_dq_stage_consumer_active =
                fixed_full_production_path
                    ? true
                    : ((g.backward_dq_pad != nullptr || g.backward_dq_bshd != nullptr) &&
                       (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY));
            const bool bwd_dq_nosem_full =
                stage_cut == DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL;
            constexpr bool bwd_dq_dual_halves = SAVED_BWD_DQ_DUAL_HALVES;
            constexpr bool bwd_pair_single = BWD_PAIR_SINGLE_KERNEL;
            const bool bwd_dq_src_completion_active =
                fixed_full_production_path
                    ? true
                    : (bwd_dq_stage_consumer_active &&
                       !bwd_dq_nosem_full &&
                       ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE)));
            const bool bwd_dq_src1_completion_active =
                fixed_full_production_path
                    ? (bwd_dq_dual_halves || bwd_pair_single)
                    : (bwd_dq_dual_halves &&
                       bwd_dq_stage_consumer_active &&
                       !bwd_dq_nosem_full &&
                       ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                        (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE)));
            auto wait_bwd_dq_src_if_needed = [&](int bwd_dq_buf, int idx) {
                if constexpr (fixed_full_production_path) {
                    if (idx >= 2) {
                        const int src_phase = (bwd_dq_src_phase_mask >> bwd_dq_buf) & 1;
                        wait(bwd_dq_src_finished[bwd_dq_buf], src_phase);
                        bwd_dq_src_phase_mask ^= (1 << bwd_dq_buf);
                        if constexpr (SAVED_BWD_DQ_DUAL_HALVES || BWD_PAIR_SINGLE_KERNEL) {
                            const int src1_phase = (bwd_dq_src1_phase_mask >> bwd_dq_buf) & 1;
                            wait(bwd_dq_src1_finished[bwd_dq_buf], src1_phase);
                            bwd_dq_src1_phase_mask ^= (1 << bwd_dq_buf);
                        }
                    }
                } else {
                    if (bwd_dq_src_completion_active && idx >= 2) {
                        const int src_phase = (bwd_dq_src_phase_mask >> bwd_dq_buf) & 1;
                        wait(bwd_dq_src_finished[bwd_dq_buf], src_phase);
                        bwd_dq_src_phase_mask ^= (1 << bwd_dq_buf);
                        if (bwd_dq_src1_completion_active) {
                            const int src1_phase = (bwd_dq_src1_phase_mask >> bwd_dq_buf) & 1;
                            wait(bwd_dq_src1_finished[bwd_dq_buf], src1_phase);
                            bwd_dq_src1_phase_mask ^= (1 << bwd_dq_buf);
                        }
                    }
                }
            };
            auto publish_bwd_dq_stage_ready = [&](int bwd_dq_buf) {
                if constexpr (fixed_full_production_path) {
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        publish_cluster_shared_backing_if_needed<C>();
                        if (cta_rank == 1 && warpgroup::laneid() == 0) {
                            tma::cluster::arrive(bwd_dq_stage_remote_ready[bwd_dq_buf], 0);
                        }
                    }
                    warpgroup::arrive(bwd_dq_stage_ready[bwd_dq_buf]);
                } else {
                    if (bwd_dq_stage_consumer_active) {
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
                                publish_cluster_shared_backing_if_needed<C>();
                                if (cta_rank == 1 && warpgroup::laneid() == 0) {
                                    tma::cluster::arrive(bwd_dq_stage_remote_ready[bwd_dq_buf], 0);
                                }
                            }
                        }
                        warpgroup::arrive(bwd_dq_stage_ready[bwd_dq_buf]);
                    }
                }
            };
            if constexpr (!QK_ONLY) {
                wait(rescale_finished[0], rescale_phase);
                rescale_phase ^= 1;
            }

            for (int idx = 0; idx < consumer_iters_per_task; ++idx) {
                constexpr int buf = 0;
                const int bwd_dq_buf = idx & 1;
                if constexpr (!QK_ONLY) {
                    wait(p_store_done[buf], p_store_phase);
                    p_store_phase ^= 1;
                }
                float2 scores_reg[C::Nb / 2];
                float tile_max = 0.0f;
                const int global_row = warpgroup::laneid();
                const int softmax_warp = warpgroup::laneid() / 32;
                const int physical_k_iter = physical_k_iter_for(idx);
                const bool third_iter_stage_cut =
                    QK_ONLY &&
                    !SAVED_LSE_BWD_ONLY &&
                    producer_iters_per_task == 3 &&
                    idx == 2 &&
                    !debug_full_like_mode(g);
                const bool fresh_consume_cut =
                    third_iter_stage_cut &&
                    (stage_cut == DEBUG_STAGE_CUT_FRESH3_WAIT_ONLY ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_COPY ||
                     stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL);
                const int wait_phase = SAVED_LSE_BWD_ONLY
                    ? ((idx >> 1) & 1)
                    : debug_scores_arrived_phase_for_iter<C>(g, idx, scores_phase);
                if constexpr (SAVED_LSE_BWD_ONLY) {
                    if (idx & 1) {
                        wait(scores_arrived_fresh[0], wait_phase);
                    } else {
                        wait(scores_arrived[0], wait_phase);
                    }
                } else {
                    if (fresh_consume_cut) {
                        wait(scores_arrived_fresh[0], wait_phase);
                    } else {
                        wait(debug_scores_arrived_sem_for_iter<C>(scores_arrived, scores_arrived_fresh, g, idx), wait_phase);
                    }
                }
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_WAIT_ONLY) {
                    stop_after_stage_cut = true;
                    break;
                }
                if (fresh_consume_cut && stage_cut == DEBUG_STAGE_CUT_FRESH3_WAIT_ONLY) {
                    stop_after_stage_cut = true;
                    break;
                }
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 32; ++ii) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(scores_reg[ii * 16 + 0].x), "=f"(scores_reg[ii * 16 + 0].y), "=f"(scores_reg[ii * 16 + 1].x), "=f"(scores_reg[ii * 16 + 1].y), "=f"(scores_reg[ii * 16 + 2].x), "=f"(scores_reg[ii * 16 + 2].y), "=f"(scores_reg[ii * 16 + 3].x), "=f"(scores_reg[ii * 16 + 3].y),
                              "=f"(scores_reg[ii * 16 + 4].x), "=f"(scores_reg[ii * 16 + 4].y), "=f"(scores_reg[ii * 16 + 5].x), "=f"(scores_reg[ii * 16 + 5].y), "=f"(scores_reg[ii * 16 + 6].x), "=f"(scores_reg[ii * 16 + 6].y), "=f"(scores_reg[ii * 16 + 7].x), "=f"(scores_reg[ii * 16 + 7].y),
                              "=f"(scores_reg[ii * 16 + 8].x), "=f"(scores_reg[ii * 16 + 8].y), "=f"(scores_reg[ii * 16 + 9].x), "=f"(scores_reg[ii * 16 + 9].y), "=f"(scores_reg[ii * 16 + 10].x), "=f"(scores_reg[ii * 16 + 10].y), "=f"(scores_reg[ii * 16 + 11].x), "=f"(scores_reg[ii * 16 + 11].y),
                              "=f"(scores_reg[ii * 16 + 12].x), "=f"(scores_reg[ii * 16 + 12].y), "=f"(scores_reg[ii * 16 + 13].x), "=f"(scores_reg[ii * 16 + 13].y), "=f"(scores_reg[ii * 16 + 14].x), "=f"(scores_reg[ii * 16 + 14].y), "=f"(scores_reg[ii * 16 + 15].x), "=f"(scores_reg[ii * 16 + 15].y)
                            : "r"(score_tt_base + ii * 32));
                }
                warpgroup::arrive(p_copy_done[buf]);
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_COPY) {
                    stop_after_stage_cut = true;
                    break;
                }
                if (fresh_consume_cut && stage_cut == DEBUG_STAGE_CUT_FRESH3_COPY) {
                    stop_after_stage_cut = true;
                    break;
                }
                if constexpr (!fixed_fresh3_full) {
                    store_scores_bf16_tmem_x16_tile_if_needed<C>(
                        g.debug_scores_bf16_tmem_x16, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, m_tile, physical_k_iter, score_tt_base
                    );
                }
                if constexpr (fixed_full_production_path) {
                    if (physical_k_iter > m_tile) {
                        if constexpr (LIVE_DP) {
                            if constexpr (LIVE_DP_MODE0_ONLY) {
                                wait(live_dp_arrived[0], live_dp_phase);
                                arrive_live_dp_copy_done();
                                live_dp_phase ^= 1;
                            } else if (live_dp_mode_consumes_tmem(g)) {
                                wait(live_dp_arrived[0], live_dp_phase);
                                arrive_live_dp_copy_done();
                                live_dp_phase ^= 1;
                            }
                        }
                        wait_bwd_dq_src_if_needed(bwd_dq_buf, idx);
                        fp4pv_zero_invalid_causal_groups_mxfp4(
                            p_fp4_stage[bwd_dq_buf], p_sc_stage[bwd_dq_buf], global_row, 0);
                        tk_localcta::subgroup_barrier_sync<128>();
                        publish_shared_backing();
                        tk_localcta::subgroup_barrier_sync<128>();
                        publish_bwd_dq_stage_ready(bwd_dq_buf);
                        scores_phase ^= 1;
                        continue;
                    }
                }
	                if (physical_k_iter >= m_tile) {
	                    int causal_col = (physical_k_iter > m_tile) ? -1 : (int)warpgroup::laneid();
	                    #pragma unroll
	                    for (int q = 0; q < C::Nb / 2; ++q) {
	                        if (q * 2 > causal_col) {
	                            scores_reg[q].x = base_types::constants<float>::neg_infty();
	                        }
	                        if (q * 2 + 1 > causal_col) {
	                            scores_reg[q].y = base_types::constants<float>::neg_infty();
	                        }
	                    }
	                }
                if constexpr (!fixed_fresh3_full) {
                    store_p_bf16_live_tile_if_needed<C>(
                        g.debug_scores_bf16_live, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg
                    );
                    if (g.debug_scores_bf16_scan_repacked != nullptr) {
                        #pragma unroll
                        for (int qid = 0; qid < 4; ++qid) {
                            __align__(8) bf16_2 scores_bf_reg[16];
                            #pragma unroll
                            for (int jj = 0; jj < 16; ++jj) {
                                scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[qid * 16 + jj]);
                            }
                            store_scores_quarter_to_localcta_scan(p_bf16_scan[0], scores_bf_reg, qid);
                        }
                        store_scores_bf16_scan_repacked_tile_if_needed<C>(
                            g.debug_scores_bf16_scan_repacked, g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter
                        );
                    }
                }
                if constexpr (SAVED_LSE_BWD_ONLY) {
                    if constexpr (!fixed_full_production_path) {
                        if (physical_k_iter > m_tile) {
                            if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
                                if (g.backward_dq_pad != nullptr || g.backward_dq_bshd != nullptr) {
                                    wait_bwd_dq_src_if_needed(bwd_dq_buf, idx);
                                    fp4pv_zero_invalid_causal_groups_mxfp4(
                                        p_fp4_stage[bwd_dq_buf], p_sc_stage[bwd_dq_buf], global_row, 0);
                                    tk_localcta::subgroup_barrier_sync<128>();
                                    publish_shared_backing();
                                    tk_localcta::subgroup_barrier_sync<128>();
                                    publish_bwd_dq_stage_ready(bwd_dq_buf);
                                }
                            }
                            scores_phase ^= 1;
                            continue;
                        }
                    }
                    constexpr float LOG2E = 1.4426950408889634f;
                    const float saved_lse_log2 = g.lse[{t_coord.x, t_coord.z, 0, m_tile * C::Mb + global_row}] * LOG2E;
                    const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
	                    const float2 neg_lse_2 = {-saved_lse_log2, -saved_lse_log2};
	                    #pragma unroll
	                    for (int qid = 0; qid < 4; ++qid) {
	                        #pragma unroll
	                        for (int jj = 0; jj < 16; ++jj) {
	                            const int si = qid * 16 + jj;
	                            scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_lse_2);
	                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
	                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
	                        }
	                    }
                    const int heads = g.q.depth();
                    const int seqlen = g.q.rows();
                    const int row = m_tile * C::Mb + global_row;
                    const float delta_val = g.backward_delta[((t_coord.x * heads + t_coord.z) * seqlen) + row];
                    const float dp_scale = g.backward_softmax_scale;
                    const float neg_delta_scaled = -delta_val * dp_scale;
                    const float2 dp_scale_2 = {dp_scale, dp_scale};
                    const float2 neg_delta_scaled_2 = {neg_delta_scaled, neg_delta_scaled};
                    if constexpr (LIVE_DP) {
                        if constexpr (LIVE_DP_MODE0_ONLY) {
                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                            const float live_dp_scale = dp_scale * MXFP4_ALPHA;
                            const float2 live_dp_scale_2 = {live_dp_scale, live_dp_scale};
                            wait(live_dp_arrived[0], live_dp_phase);
                            #pragma unroll
                            for (int qid = 0; qid < 4; ++qid) {
                                if constexpr (LIVE_DP_MODE0_X8) {
                                    fp4pv_apply_live_dp_quarter_tmem_x8(
                                        scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                } else {
                                    fp4pv_apply_live_dp_quarter_tmem_x16(
                                        scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                }
                            }
                            tensor_load_wait();
                            arrive_live_dp_copy_done();
                            live_dp_phase ^= 1;
                        } else {
                            if (g.live_dp_debug_mode == LIVE_DP_DEBUG_FULL) {
                                wait(live_dp_arrived[0], live_dp_phase);
	                            #pragma unroll
	                            for (int qid = 0; qid < 4; ++qid) {
	                                float2 dp_chunk[16];
	                                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                                        : "=f"(dp_chunk[0].x), "=f"(dp_chunk[0].y), "=f"(dp_chunk[1].x), "=f"(dp_chunk[1].y), "=f"(dp_chunk[2].x), "=f"(dp_chunk[2].y), "=f"(dp_chunk[3].x), "=f"(dp_chunk[3].y),
                                          "=f"(dp_chunk[4].x), "=f"(dp_chunk[4].y), "=f"(dp_chunk[5].x), "=f"(dp_chunk[5].y), "=f"(dp_chunk[6].x), "=f"(dp_chunk[6].y), "=f"(dp_chunk[7].x), "=f"(dp_chunk[7].y),
                                          "=f"(dp_chunk[8].x), "=f"(dp_chunk[8].y), "=f"(dp_chunk[9].x), "=f"(dp_chunk[9].y), "=f"(dp_chunk[10].x), "=f"(dp_chunk[10].y), "=f"(dp_chunk[11].x), "=f"(dp_chunk[11].y),
	                                          "=f"(dp_chunk[12].x), "=f"(dp_chunk[12].y), "=f"(dp_chunk[13].x), "=f"(dp_chunk[13].y), "=f"(dp_chunk[14].x), "=f"(dp_chunk[14].y), "=f"(dp_chunk[15].x), "=f"(dp_chunk[15].y)
	                                        : "r"(score_tt_base + qid * 32));
	                                constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
	                                const float2 alpha_2 = {MXFP4_ALPHA, MXFP4_ALPHA};
	                                    #pragma unroll
	                                    for (int jj = 0; jj < 16; ++jj) {
	                                        const int q = qid * 16 + jj;
	                                        const float2 dp_val = __fmul2_rn(dp_chunk[jj], alpha_2);
	                                        scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
	                                        scores_reg[q], dp_val, dp_scale_2, neg_delta_scaled_2);
	                                }
	                            }
                                tensor_load_wait();
                                arrive_live_dp_copy_done();
                                live_dp_phase ^= 1;
                            } else {
                                if (!live_dp_mode_consumes_tmem(g)) {
                                    const float2 zero_dp = {0.0f, 0.0f};
                                    #pragma unroll
                                    for (int q = 0; q < C::Nb / 2; ++q) {
                                        scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                                            scores_reg[q], zero_dp, dp_scale_2, neg_delta_scaled_2);
                                    }
                                } else {
                                    wait(live_dp_arrived[0], live_dp_phase);
                                    if (live_dp_mode_applies_values(g)) {
                                        if (g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_WAIT &&
                                            g.backward_dp_bf16 == nullptr) {
                                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                            const float live_dp_scale = dp_scale * MXFP4_ALPHA;
                                            const float2 live_dp_scale_2 = {live_dp_scale, live_dp_scale};
                                            #pragma unroll
                                            for (int qid = 0; qid < 4; ++qid) {
                                                fp4pv_apply_live_dp_quarter_tmem_x8(
                                                    scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                            }
                                        } else if (g.live_dp_debug_mode == LIVE_DP_DEBUG_X8_ALIGNED_PROBE &&
                                                   g.backward_dp_bf16 == nullptr) {
                                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                            const float live_dp_scale = dp_scale * MXFP4_ALPHA;
                                            const float2 live_dp_scale_2 = {live_dp_scale, live_dp_scale};
                                            #pragma unroll
                                            for (int qid = 0; qid < 4; ++qid) {
                                                fp4pv_apply_live_dp_quarter_tmem_x8_aligned_probe(
                                                    scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                            }
                                        } else if (g.live_dp_debug_mode == LIVE_DP_DEBUG_X16_WAIT &&
                                                   g.backward_dp_bf16 == nullptr) {
                                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                            const float live_dp_scale = dp_scale * MXFP4_ALPHA;
                                            const float2 live_dp_scale_2 = {live_dp_scale, live_dp_scale};
                                            #pragma unroll
                                            for (int qid = 0; qid < 4; ++qid) {
                                                fp4pv_apply_live_dp_quarter_tmem_x16(
                                                    scores_reg, score_tt_base, qid, live_dp_scale_2, neg_delta_scaled_2);
                                            }
                                        } else {
	                                        #pragma unroll
	                                        for (int qid = 0; qid < 4; ++qid) {
	                                            float2 dp_chunk[16];
	                                            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                                                : "=f"(dp_chunk[0].x), "=f"(dp_chunk[0].y), "=f"(dp_chunk[1].x), "=f"(dp_chunk[1].y), "=f"(dp_chunk[2].x), "=f"(dp_chunk[2].y), "=f"(dp_chunk[3].x), "=f"(dp_chunk[3].y),
                                                  "=f"(dp_chunk[4].x), "=f"(dp_chunk[4].y), "=f"(dp_chunk[5].x), "=f"(dp_chunk[5].y), "=f"(dp_chunk[6].x), "=f"(dp_chunk[6].y), "=f"(dp_chunk[7].x), "=f"(dp_chunk[7].y),
                                                  "=f"(dp_chunk[8].x), "=f"(dp_chunk[8].y), "=f"(dp_chunk[9].x), "=f"(dp_chunk[9].y), "=f"(dp_chunk[10].x), "=f"(dp_chunk[10].y), "=f"(dp_chunk[11].x), "=f"(dp_chunk[11].y),
	                                                  "=f"(dp_chunk[12].x), "=f"(dp_chunk[12].y), "=f"(dp_chunk[13].x), "=f"(dp_chunk[13].y), "=f"(dp_chunk[14].x), "=f"(dp_chunk[14].y), "=f"(dp_chunk[15].x), "=f"(dp_chunk[15].y)
	                                                : "r"(score_tt_base + qid * 32));
		                                        if (live_dp_mode_uses_scaled_mma_values(g)) {
		                                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
		                                            const float2 alpha_2 = {MXFP4_ALPHA, MXFP4_ALPHA};
	                                            #pragma unroll
	                                            for (int jj = 0; jj < 16; ++jj) {
	                                                dp_chunk[jj] = __fmul2_rn(dp_chunk[jj], alpha_2);
	                                            }
	                                        }
	                                        if (g.backward_dp_bf16 != nullptr) {
	                                            const int col_base = physical_k_iter * C::Nb;
	                                            const int64_t dp_row_base =
                                                (((static_cast<int64_t>(t_coord.x) * heads + t_coord.z) * seqlen) + row) * seqlen;
                                            #pragma unroll
                                            for (int jj = 0; jj < 16; ++jj) {
                                                const int col = col_base + (qid * 16 + jj) * 2;
                                                const bf16_2 dp_pair_bf = __float22bfloat162_rn(dp_chunk[jj]);
                                                *reinterpret_cast<bf16_2 *>(&g.backward_dp_bf16[dp_row_base + col]) = dp_pair_bf;
                                            }
                                        }
                                        #pragma unroll
                                        for (int jj = 0; jj < 16; ++jj) {
                                            const int q = qid * 16 + jj;
	                                            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
	                                                scores_reg[q], dp_chunk[jj], dp_scale_2, neg_delta_scaled_2);
	                                        }
	                                    }
                                        }
                                        tensor_load_wait();
                                    } else {
                                        const float2 zero_dp = {0.0f, 0.0f};
                                        #pragma unroll
                                        for (int q = 0; q < C::Nb / 2; ++q) {
                                            scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                                                scores_reg[q], zero_dp, dp_scale_2, neg_delta_scaled_2);
                                        }
                                    }
	                                    arrive_live_dp_copy_done();
	                                    live_dp_phase ^= 1;
                                }
                            }
                        }
                    } else {
                        const int col_base = physical_k_iter * C::Nb;
                        const int64_t dp_row_base = (((static_cast<int64_t>(t_coord.x) * heads + t_coord.z) * seqlen) + row) * seqlen;
                        #pragma unroll
                        for (int qid = 0; qid < 4; ++qid) {
                            #pragma unroll
                            for (int jj = 0; jj < 16; ++jj) {
                                const int q = qid * 16 + jj;
                                const int col = col_base + q * 2;
                                const bf16_2 dp_pair_bf =
                                    *reinterpret_cast<const bf16_2 *>(&g.backward_dp_bf16[dp_row_base + col]);
                                scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                                    scores_reg[q], dp_pair_bf, dp_scale_2, neg_delta_scaled_2);
                            }
                        }
                    }
                    if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
                        // Keep this after live_dp_copy_done. Holding live-dP TMEM reuse while
                        // waiting for the dQ source stage can form a producer/consumer cycle.
                        wait_bwd_dq_src_if_needed(bwd_dq_buf, idx);
                        uint32_t packed_scale_marker = 0u;
	                        if constexpr (fixed_full_production_path) {
                                if constexpr (!SAVED_BWD_EMIT_ROW_MXFP4 && !SAVED_BWD_EMIT_COL_MXFP4) {
                                    if (g.backward_fused_stage_transpose) {
                                        packed_scale_marker = fp4pv_pack_scores_to_transposed_stage_mxfp4_marker_bf16_amax<C>(
                                            p_fp4_stage[bwd_dq_buf],
                                            p_sc_stage[bwd_dq_buf],
                                            g,
                                            p_bf16_scan[0],
                                            m_tile,
                                            physical_k_iter,
                                            scores_reg);
                                    } else {
                                        fp4pv_pack_scores_to_stage_mxfp4_bf16_amax(
                                            p_fp4_stage[bwd_dq_buf],
                                            p_sc_stage[bwd_dq_buf],
                                            scores_reg,
                                            global_row,
                                            g.backward_ds_quant_mode);
                                    }
	                            } else if constexpr (SAVED_BWD_EMIT_ROW_MXFP4 && !SAVED_BWD_EMIT_COL_MXFP4) {
	                                fp4pv_pack_scores_to_stage_mxfp4_bf16_amax(
	                                    p_fp4_stage[bwd_dq_buf],
	                                    p_sc_stage[bwd_dq_buf],
                                    scores_reg,
                                    global_row,
                                    g.backward_ds_quant_mode);
                            } else {
                                fp4pv_pack_scores_to_stage_mxfp4_bf16_amax_and_scan(
                                    p_fp4_stage[bwd_dq_buf],
                                    p_sc_stage[bwd_dq_buf],
                                    p_bf16_scan[0],
                                    scores_reg,
                                    global_row,
                                    g.backward_ds_quant_mode);
                            }
                        } else {
                            if (g.backward_fused_stage_transpose) {
                                packed_scale_marker = fp4pv_pack_scores_to_transposed_stage_mxfp4_marker_bf16_amax<C>(
                                    p_fp4_stage[bwd_dq_buf],
                                    p_sc_stage[bwd_dq_buf],
                                    g,
                                    p_bf16_scan[0],
                                    m_tile,
                                    physical_k_iter,
                                    scores_reg);
                            } else {
                                if constexpr (SAVED_BWD_EMIT_COL_MXFP4 && !SAVED_BWD_EMIT_ROW_MXFP4) {
                                    if constexpr (fixed_fresh3_full) {
                                        fp4pv_pack_scores_to_stage_mxfp4_bf16_amax_and_scan(
                                            p_fp4_stage[bwd_dq_buf],
                                            p_sc_stage[bwd_dq_buf],
                                            p_bf16_scan[0],
                                            scores_reg,
                                            global_row,
                                            g.backward_ds_quant_mode);
                                    } else {
                                        packed_scale_marker = fp4pv_pack_scores_to_stage_mxfp4_marker_bf16_amax_and_scan(
                                            p_fp4_stage[bwd_dq_buf],
                                            p_sc_stage[bwd_dq_buf],
                                            p_bf16_scan[0],
                                            scores_reg,
                                            global_row,
                                            g.backward_ds_quant_mode);
                                    }
                                } else {
                                    packed_scale_marker = fp4pv_pack_scores_to_stage_mxfp4_marker_bf16_amax(
                                        p_fp4_stage[bwd_dq_buf],
                                        p_sc_stage[bwd_dq_buf],
                                        scores_reg,
                                        global_row,
                                        g.backward_ds_quant_mode);
                                }
                            }
                        }
                        tk_localcta::subgroup_barrier_sync<128>();
                        publish_shared_backing();
                        tk_localcta::subgroup_barrier_sync<128>();
                        publish_bwd_dq_stage_ready(bwd_dq_buf);
                        if constexpr (!fixed_fresh3_full) {
                            if (g.backward_fused_stage_marker != nullptr) {
                                const int64_t marker_idx =
                                    ((((static_cast<int64_t>(t_coord.x) * heads + t_coord.z) * seqlen) + row) * (seqlen / C::Nb)) +
                                    physical_k_iter;
                                g.backward_fused_stage_marker[marker_idx] = packed_scale_marker;
                            }
                        }
	                            if constexpr (SAVED_BWD_EMIT_ROW_MXFP4) {
	                            if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
	                                store_backward_ds_row_col_mxfp4_direct_if_needed<C>(
	                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
	                            } else {
			                                if constexpr (fixed_full_production_path) {
			                                    if (!g.backward_skip_global_ds_row_store) {
			                                        store_backward_ds_fp4_row_tile_to_global_warpgroup<C>(
			                                            p_fp4_stage[bwd_dq_buf], g, t_coord.x, t_coord.z, m_tile, physical_k_iter, warpgroup::laneid());
			                                        store_backward_ds_sc_row_word_to_global_warpgroup<C>(
			                                            p_sc_stage[bwd_dq_buf], g, t_coord.x, t_coord.z, m_tile, physical_k_iter, m_tile * C::Mb + global_row);
			                                        warpgroup::sync(warpgroup::groupid() + 1);
			                                    }
			                                } else {
	                                    store_backward_ds_row_mxfp4_direct_if_needed<C>(
	                                        g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
	                                }
	                            }
	                        } else if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                            if constexpr (fixed_full_production_path) {
                                store_backward_ds_col_mxfp4_from_populated_scan_full<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter);
                            } else if (g.backward_fused_stage_transpose) {
                                store_backward_ds_col_mxfp4_direct_if_needed<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
                            } else if constexpr (!SAVED_BWD_EMIT_ROW_MXFP4) {
                                store_backward_ds_col_mxfp4_from_populated_scan_full<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter);
                            } else {
                                store_backward_ds_col_mxfp4_from_populated_scan_if_needed<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter);
                            }
                        }
                        scores_phase ^= 1;
                        continue;
                    }
                    if constexpr (SAVED_BWD_EMIT_ROW_MXFP4) {
                        if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                            store_backward_ds_row_col_mxfp4_direct_if_needed<C>(
                                g, p_bf16_scan[0], t_coord.x, t_coord.z, row, physical_k_iter, scores_reg);
                        } else {
                            store_backward_ds_row_mxfp4_direct_if_needed<C>(
                                g, t_coord.x, t_coord.z, row, physical_k_iter, scores_reg);
                        }
                    } else if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                        store_backward_ds_col_mxfp4_direct_if_needed<C>(
                            g, p_bf16_scan[0], t_coord.x, t_coord.z, row, physical_k_iter, scores_reg);
                    }
                    if constexpr (SAVED_BWD_STORE_BF16) {
                        store_p_bf16_live_tile_if_needed<C>(
                            g.p_bf16_live, g, t_coord.x, t_coord.z, row, physical_k_iter, scores_reg
                        );
                    }
                    scores_phase ^= 1;
                    continue;
                }
                if constexpr (QK_ONLY) {
                    if (g.use_external_lse) {
                        if (physical_k_iter > m_tile) {
                            // Full-layout dS outputs are pre-zeroed before launching this
                            // saved-LSE backward path, so future causal tiles require no
                            // explicit BF16 or MXFP4 zero stores.
                            scores_phase ^= 1;
                            continue;
                        }
                        constexpr float LOG2E = 1.4426950408889634f;
                        const float saved_lse_log2 = g.lse[{t_coord.x, t_coord.z, 0, m_tile * C::Mb + global_row}] * LOG2E;
                        const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                        const float2 neg_lse_2 = {-saved_lse_log2, -saved_lse_log2};
                        #pragma unroll
                        for (int qid = 0; qid < 4; ++qid) {
                            #pragma unroll
                            for (int jj = 0; jj < 16; ++jj) {
                                const int si = qid * 16 + jj;
                                scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_lse_2);
                                scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                                scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            }
                        }
                        if (g.emit_backward_ds) {
                            const int heads = g.q.depth();
                            const int seqlen = g.q.rows();
                            const int row = m_tile * C::Mb + global_row;
                            const float delta_val = g.backward_delta[((t_coord.x * heads + t_coord.z) * seqlen) + row];
                            const float dp_scale = g.backward_softmax_scale;
                            const float neg_delta_scaled = -delta_val * dp_scale;
                            const float2 dp_scale_2 = {dp_scale, dp_scale};
                            const float2 neg_delta_scaled_2 = {neg_delta_scaled, neg_delta_scaled};
                            const int col_base = physical_k_iter * C::Nb;
                            const int64_t dp_row_base = (((static_cast<int64_t>(t_coord.x) * heads + t_coord.z) * seqlen) + row) * seqlen;
                            #pragma unroll
                            for (int qid = 0; qid < 4; ++qid) {
                                #pragma unroll
                                for (int jj = 0; jj < 16; ++jj) {
                                    const int q = qid * 16 + jj;
                                    const int col = col_base + q * 2;
                                    const bf16_2 dp_pair_bf =
                                        *reinterpret_cast<const bf16_2 *>(&g.backward_dp_bf16[dp_row_base + col]);
                                    scores_reg[q] = fp4pv_apply_backward_dp_delta_scale(
                                        scores_reg[q], dp_pair_bf, dp_scale_2, neg_delta_scaled_2);
                                }
                            }
                        }
                        if (g.emit_backward_ds_row_mxfp4) {
                            if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                                store_backward_ds_row_col_mxfp4_direct_if_needed<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
                            } else {
                                store_backward_ds_row_mxfp4_direct_if_needed<C>(
                                    g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
                            }
                        } else if (g.emit_backward_ds_col_mxfp4) {
                            if constexpr (SAVED_BWD_EMIT_COL_MXFP4) {
                                store_backward_ds_col_mxfp4_direct_if_needed<C>(
                                    g, p_bf16_scan[0], t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg);
                            }
                        }
                        store_p_bf16_live_tile_if_needed<C>(
                            g.p_bf16_live, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg
                        );
                        scores_phase ^= 1;
                        continue;
                    }
                }

                float row_max_old = row_max;
                float tile_score_max = base_types::constants<float>::neg_infty();
                float lm0 = base_types::constants<float>::neg_infty(), lm1 = base_types::constants<float>::neg_infty();
                float lm2 = base_types::constants<float>::neg_infty(), lm3 = base_types::constants<float>::neg_infty();
                float lm4 = base_types::constants<float>::neg_infty(), lm5 = base_types::constants<float>::neg_infty();
                float lm6 = base_types::constants<float>::neg_infty(), lm7 = base_types::constants<float>::neg_infty();
                #pragma unroll
                for (int j = 0; j < C::Nb / 2; j += 8) {
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(scores_reg[j + 0].x), "f"(scores_reg[j + 0].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm1) : "f"(lm1), "f"(scores_reg[j + 1].x), "f"(scores_reg[j + 1].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm2) : "f"(lm2), "f"(scores_reg[j + 2].x), "f"(scores_reg[j + 2].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm3) : "f"(lm3), "f"(scores_reg[j + 3].x), "f"(scores_reg[j + 3].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(scores_reg[j + 4].x), "f"(scores_reg[j + 4].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm5) : "f"(lm5), "f"(scores_reg[j + 5].x), "f"(scores_reg[j + 5].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm6) : "f"(lm6), "f"(scores_reg[j + 6].x), "f"(scores_reg[j + 6].y));
                    asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm7) : "f"(lm7), "f"(scores_reg[j + 7].x), "f"(scores_reg[j + 7].y));
                }
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm1), "f"(lm2));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm4) : "f"(lm4), "f"(lm5), "f"(lm6));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(lm0) : "f"(lm0), "f"(lm3), "f"(lm4));
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(tile_score_max) : "f"(tile_score_max), "f"(lm0), "f"(lm7));
                row_max = fmaxf(row_max, tile_score_max);
                float acc_scale = 1.0f;
                if (physical_k_iter >= m_tile || idx > 0) {
                    if (!ZERO_SCALE) {
                        constexpr float RESCALE_THRESHOLD = 8.f;
                        float acc_scale_log2 = (row_max_old - row_max) * SCALE_LOG2;
                        if (acc_scale_log2 >= -RESCALE_THRESHOLD) {
                            row_max = row_max_old;
                            acc_scale = 1.0f;
                        } else {
                            acc_scale = fp4pv_exp2_approx(acc_scale_log2);
                        }
                    }
                }
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_ROWMAX) {
                    stop_after_stage_cut = true;
                    break;
                }
                if constexpr (!QK_ONLY) {
                    warp::sync();
                    warp::arrive(corr_arrived[0]);
                }

                float2 tile_sum0 = {0.0f, 0.0f}, tile_sum1 = {0.0f, 0.0f};
                if (!ZERO_SCALE) {
                    float neg_max_scaled = row_max * (-SCALE_LOG2);
                    float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                    const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                    #pragma unroll
                    for (int qid = 0; qid < C::Nb / 32; ++qid) {
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj) {
                            int si = qid * 16 + jj;
                            scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_max_scaled_2);
                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            if ((jj & 1) == 0) {
                                tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                            } else {
                                tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int qid = 0; qid < C::Nb / 32; ++qid) {
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj) {
                            int si = qid * 16 + jj;
                            scores_reg[si].x = isfinite(scores_reg[si].x) ? 0.0f : scores_reg[si].x;
                            scores_reg[si].y = isfinite(scores_reg[si].y) ? 0.0f : scores_reg[si].y;
                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            if ((jj & 1) == 0) {
                                tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                            } else {
                                tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                            }
                        }
                    }
                }

                if constexpr (!QK_ONLY) {
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                }
                tile_sum0 = __fadd2_rn(tile_sum0, tile_sum1);
                const float tile_exp_sum = tile_sum0.x + tile_sum0.y;
                row_sum = row_sum * acc_scale + tile_exp_sum;
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_EXP) {
                    stop_after_stage_cut = true;
                    break;
                }
                if constexpr (NORMALIZE_TILES) {
                    const float inv_row_sum =
                        (row_sum > 0.0f && isfinite(row_sum)) ? __frcp_rn(row_sum) : 0.0f;
                    tile_max = 0.0f;
                    #pragma unroll
                    for (int qid = 0; qid < C::Nb / 32; ++qid) {
                        __align__(8) bf16_2 scores_bf_reg[16];
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj) {
                            const int si = qid * 16 + jj;
                            scores_reg[si].x *= inv_row_sum;
                            scores_reg[si].y *= inv_row_sum;
                            tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[si]);
                        }
                        store_scores_quarter_to_localcta_scan(p_bf16_scan[0], scores_bf_reg, qid);
                    }
                }
                if (third_iter_stage_cut && stage_cut == DEBUG_STAGE_CUT_CONSUME3_SCAN) {
                    stop_after_stage_cut = true;
                    break;
                }
                if constexpr (!QK_ONLY) {
                    warp::sync();
                    #pragma unroll
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, mask));
                    }
                    if (warp::laneid() == 0) {
                        p_softmax_warp_amax[buf][softmax_warp] = tile_max;
                    }
                    tk_localcta::subgroup_barrier_sync<128>();
                    if (warpgroup::laneid() == 0) {
                        float cta_amax = p_softmax_warp_amax[buf][0];
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][1]);
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][2]);
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][3]);
                        p_cta_amax[buf] = cta_amax;
                    }
                    tk_localcta::subgroup_barrier_sync<128>();
                    fp4pv_zero_raw_scale_stage(p_sc_stage[buf], warpgroup::laneid());
                    if constexpr (NORMALIZE_TILES) {
                        fp4pv_pack_scores_to_stage_payload_from_localcta_scan_row(
                            p_fp4_stage[buf], p_bf16_scan[0], p_cta_amax[buf], global_row
                        );
                        fp4pv_store_scales_from_localcta_scan_row(
                            p_sc_stage[buf], p_bf16_scan[0], p_cta_amax[buf], global_row
                        );
                    } else {
                        fp4pv_pack_scores_to_stage_and_scales(
                            p_fp4_stage[buf], p_sc_stage[buf], scores_reg, p_cta_amax[buf], global_row
                        );
                    }
                    int valid_groups = tk_localcta::SCALES_PER_CHUNK_X;
                    if (physical_k_iter >= m_tile) {
                        valid_groups = (physical_k_iter > m_tile) ? 0 : ((warpgroup::laneid() / 16) + 1);
                    }
                    fp4pv_zero_invalid_causal_groups(
                        p_fp4_stage[buf], p_sc_stage[buf], global_row, valid_groups
                    );
                    store_p_sc_rowmajor_live_tile_if_needed<C>(
                        p_sc_stage[buf], g.p_sc_rowmajor_live, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter
                    );
                    if (blockIdx.x == 0 && cta_rank == 1 && idx == 1 && warpgroup::laneid() == 32) {
                        auto &raw_scales_dbg = *reinterpret_cast<fp4pv_p_raw_scales *>(&p_sc_stage[buf].data[0]);
                        printf("raw_sc row=%d amax=%f :", global_row, p_cta_amax[buf]);
                        #pragma unroll
                        for (int gidx = 0; gidx < tk_localcta::SCALES_PER_CHUNK_X; ++gidx) {
                            printf(" %f", static_cast<float>(raw_scales_dbg[global_row][gidx]));
                        }
                        printf("\n");
                    }
                }
                store_p_bf16_live_tile_if_needed<C>(
                    g.p_bf16_live, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, scores_reg
                );
                if constexpr (!QK_ONLY) {
                    tk_localcta::subgroup_barrier_sync<128>();
                    publish_shared_backing();
                    tk_localcta::subgroup_barrier_sync<128>();
                    warpgroup::arrive(p_pack_ready[buf]);
                }
                store_debug_row_stat_if_needed<C>(g.debug_row_max, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, row_max);
                store_debug_row_stat_if_needed<C>(g.debug_row_sum, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, row_sum);
                store_debug_row_stat_if_needed<C>(g.debug_tile_score_max, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, tile_score_max);
                store_debug_row_stat_if_needed<C>(g.debug_tile_exp_sum, g, t_coord.x, t_coord.z, m_tile * C::Mb + global_row, physical_k_iter, tile_exp_sum);
                if (warpgroup::laneid() == 0) {
                    const float debug_tile_amax = QK_ONLY ? 0.0f : p_cta_amax[buf];
                    store_debug_tile_amax_if_needed<C>(g.debug_tile_amax, g, t_coord.x, t_coord.z, m_tile, physical_k_iter, debug_tile_amax);
	                }
	                scores_phase ^= 1;
	            }
	            if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
                if constexpr (fixed_full_production_path) {
                    const int tail_begin = consumer_iters_per_task > 2 ? consumer_iters_per_task - 2 : 0;
                    for (int tail_idx = tail_begin; tail_idx < consumer_iters_per_task; ++tail_idx) {
                        const int tail_buf = tail_idx & 1;
                        const int src_phase = (bwd_dq_src_phase_mask >> tail_buf) & 1;
                        wait(bwd_dq_src_finished[tail_buf], src_phase);
                        bwd_dq_src_phase_mask ^= (1 << tail_buf);
                        if constexpr (SAVED_BWD_DQ_DUAL_HALVES || BWD_PAIR_SINGLE_KERNEL) {
                            const int src1_phase = (bwd_dq_src1_phase_mask >> tail_buf) & 1;
                            wait(bwd_dq_src1_finished[tail_buf], src1_phase);
                            bwd_dq_src1_phase_mask ^= (1 << tail_buf);
                        }
                    }
                } else if (bwd_dq_src_completion_active && !stop_after_stage_cut) {
	                    const int tail_begin = consumer_iters_per_task > 2 ? consumer_iters_per_task - 2 : 0;
	                    for (int tail_idx = tail_begin; tail_idx < consumer_iters_per_task; ++tail_idx) {
	                        const int tail_buf = tail_idx & 1;
	                        const int src_phase = (bwd_dq_src_phase_mask >> tail_buf) & 1;
	                        wait(bwd_dq_src_finished[tail_buf], src_phase);
	                        bwd_dq_src_phase_mask ^= (1 << tail_buf);
		                        if (bwd_dq_src1_completion_active) {
		                            const int src1_phase = (bwd_dq_src1_phase_mask >> tail_buf) & 1;
		                            wait(bwd_dq_src1_finished[tail_buf], src1_phase);
		                            bwd_dq_src1_phase_mask ^= (1 << tail_buf);
		                        }
	                    }
	                }
	            }
	            if (QK_ONLY && stage_cut == DEBUG_STAGE_CUT_ISSUE3_ONLY && producer_iters_per_task == 3) {
                stop_after_stage_cut = true;
            }
            if (stop_after_stage_cut) {
                break;
            }

            bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
            float lse_val;
            if (!row_invalid) {
                constexpr float LN2 = 0.693147180559945f;
                float log2_row_sum;
                asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                lse_val = (row_max * SCALE_LOG2 + log2_row_sum) * LN2;
            } else {
                lse_val = base_types::constants<float>::neg_infty();
            }
            store_lse_row_if_needed<C>(g, t_coord.x, t_coord.z, m_tile * C::Mb + warpgroup::laneid(), lse_val);
        }
    } else if (warpgroup::groupid() == 2) {
        if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<48>();
        }
        if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
            if constexpr (fixed_full_production_path) {
                d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT_BASE);
                int accum_phase = 0;
                for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    const int m_tile = t_coord.y + cta_rank;
                    if (m_tile >= tiles_m) {
                        continue;
                    }
                    const int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                    if (iters_per_task <= 0) {
                        continue;
                    }
                    wait(bwd_dq_accum_done[0], accum_phase);
                    accum_phase ^= 1;
                    warpgroup::tma::store_async_read_wait<0>();
                    warpgroup::sync(warpgroup::groupid() + 1);
                    if constexpr (BWD_PAIR_SINGLE_KERNEL) {
                        d_tt_outputs tt_pair_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                        store_bwd_dq_pad_tmem_tile<C>(tt_output, g, t_coord.x, t_coord.z, m_tile);
                        store_bwd_pair_dq_pad_tmem_tile<C>(tt_pair_output, g, t_coord.x, t_coord.z, m_tile);
                    } else {
                        store_bwd_dq_pad_tmem_tile<C, 0>(tt_output, g, t_coord.x, t_coord.z, m_tile);
                        if constexpr (BWD_DQ_TAIL64_CLUSTER) {
                            d_tt_tail_outputs tt_output1 = tm_alloc.template allocate<d_tt_tail_outputs>(BWD_DQ_OUTPUT1_BASE);
                            store_bwd_dq_pad_tmem_tile<C, 1>(tt_output1, g, t_coord.x, t_coord.z, m_tile);
                        } else {
                            d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                            store_bwd_dq_pad_tmem_tile<C, 1>(tt_output1, g, t_coord.x, t_coord.z, m_tile);
                        }
                    }
                }
            } else {
                const bool bwd_dq_wait_tmem_active =
                    (g.backward_dq_pad != nullptr || g.backward_dq_bshd != nullptr) &&
                    ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM));
                if (bwd_dq_wait_tmem_active) {
                    d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT_BASE);
                    constexpr bool bwd_dq_dual_halves = SAVED_BWD_DQ_DUAL_HALVES;
                    int accum_phase = 0;
                    int tmem_phase = 0;
                    for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                        int3 t_coord = get_tile_idx(cur_bid);
                        const int m_tile = t_coord.y + cta_rank;
                        if (m_tile >= tiles_m) {
                            continue;
                        }
                        const int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                        if (iters_per_task <= 0) {
                            continue;
                        }
                        if ((stage_cut != DEBUG_STAGE_CUT_FRESH3_FULL)) {
                            for (int idx = 0; idx < iters_per_task; ++idx) {
                                wait(bwd_dq_tmem_ready[0], tmem_phase);
                                tmem_phase ^= 1;
                            }
                        }
                        wait(bwd_dq_accum_done[0], accum_phase);
                        accum_phase ^= 1;
                        if ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                            (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL)) {
                            warpgroup::tma::store_async_read_wait<0>();
                            warpgroup::sync(warpgroup::groupid() + 1);
                            if constexpr (bwd_dq_dual_halves) {
                                d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                store_bwd_dq_pad_tmem_tile<C, 0>(tt_output, g, t_coord.x, t_coord.z, m_tile);
                                store_bwd_dq_pad_tmem_tile<C, 1>(tt_output1, g, t_coord.x, t_coord.z, m_tile);
                            } else {
                                store_bwd_dq_pad_tmem_tile<C>(tt_output, g, t_coord.x, t_coord.z, m_tile);
                            }
                        }
                    }
                }
            }
        } else if constexpr (!QK_ONLY) {
            int corr_phase = 0;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                const bool row_tile_valid = (t_coord.y + cta_rank) < tiles_m;
                if (!row_tile_valid) {
                    continue;
                }
                int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    wait(corr_arrived[0], corr_phase);
                    warpgroup::sync(warpgroup::groupid() + 1);
                    warpgroup::arrive(rescale_finished[0]);
                    corr_phase ^= 1;
                }
            }
        }
    } else if (warpgroup::groupid() == 0) {
        if constexpr (SAVED_BWD_LOCAL_STAGE_ONLY) {
            if ((g.backward_dq_pad != nullptr || g.backward_dq_bshd != nullptr) &&
                (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY)) {
                const bool issue_warp = (cta_rank == 0) && (warpgroup::warpid() == 0);
                const bool issue_lane = issue_warp && warp::elect_leader();
                d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT_BASE);
                d_tt_bwd_p_sc p_sc_tm = tm_alloc.template allocate<d_tt_bwd_p_sc>(BWD_DQ_P_SC_BASE);
                d_tt_bwd_k_sc k_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K_SC_BASE);
                int stage_phase_mask = 0;
                int src_phase_mask = 0;
                int src1_phase_mask = 0;
                int stage_remote_phase_mask = 0;
                int k_remote_phase_mask = 0;
                int tmem_ready_count = 0;
                constexpr bool bwd_dq_dual_halves = SAVED_BWD_DQ_DUAL_HALVES;
                constexpr bool bwd_pair_single = BWD_PAIR_SINGLE_KERNEL;
                const bool bwd_dq_nosem_full =
                    stage_cut == DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL;
                const bool bwd_dq_src1_wait_active =
                    (bwd_dq_dual_halves || bwd_pair_single) &&
                    !bwd_dq_nosem_full &&
                    ((stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_WAIT_TMEM_NO_STORE));
                constexpr bool bwd_dq_prepare_issue_active = false;
                for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    const int m_tile = t_coord.y + cta_rank;
                    if (m_tile >= tiles_m) {
                        continue;
                    }
                    const bool peer_cta_valid = (t_coord.y + 1) < tiles_m;
                    const int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                    if (iters_per_task <= 0) {
                        continue;
                    }
                    for (int idx = 0; idx < iters_per_task; ++idx) {
                        const int buf = idx & 1;
                        const int physical_k_iter = physical_k_iter_for(idx);
                        if (idx >= 2 && !bwd_dq_nosem_full) {
                            const int src_phase = (src_phase_mask >> buf) & 1;
                            wait(bwd_dq_src_finished[buf], src_phase);
                            src_phase_mask ^= (1 << buf);
	                            if (bwd_dq_src1_wait_active) {
	                                const int src1_phase = (src1_phase_mask >> buf) & 1;
	                                wait(bwd_dq_src1_finished[buf], src1_phase);
	                                src1_phase_mask ^= (1 << buf);
	                            }
                        }
                        const int stage_phase = (stage_phase_mask >> buf) & 1;
                        wait(bwd_dq_stage_ready[buf], stage_phase);
                        stage_phase_mask ^= (1 << buf);
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                if constexpr (fixed_fresh3_full) {
                                    if (peer_cta_valid) {
                                        const int stage_remote_phase = (stage_remote_phase_mask >> buf) & 1;
                                        tma::cluster::wait(bwd_dq_stage_remote_ready[buf], stage_remote_phase);
                                        stage_remote_phase_mask ^= (1 << buf);
                                    }
                                } else if (peer_cta_valid && (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
                                    const int stage_remote_phase = (stage_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(bwd_dq_stage_remote_ready[buf], stage_remote_phase);
                                    stage_remote_phase_mask ^= (1 << buf);
                                }
                            }
                        }
	                        if constexpr (!fixed_fresh3_full) {
	                        if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_STAGE_READY_ONLY)) {
	                            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
	                                warpgroup::arrive(bwd_dq_src_finished[buf]);
	                            }
	                            continue;
	                        }
	                        }
                        if (warpgroup::warpid() == 0) {
                            if constexpr (bwd_dq_dual_halves) {
                                load_bwd_dq_k_col_fp4_tile<C>(bwd_dq_k_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank, 0);
                                load_bwd_dq_k_col_sc_tile<C>(bwd_dq_k_sc_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, 0);
                                if constexpr (BWD_DQ_TAIL64_CLUSTER) {
                                    load_bwd_dq_k_col_fp4_tail64_tile<C>(bwd_dq_k1_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                                } else {
                                    load_bwd_dq_k_col_fp4_tile<C>(bwd_dq_k1_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank, 1);
                                }
                                load_bwd_dq_k_col_sc_tile<C>(bwd_dq_k1_sc_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, 1);
                            } else {
                                load_bwd_dq_k_col_fp4_tile<C>(bwd_dq_k_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter, cta_rank);
                                load_bwd_dq_k_col_sc_tile<C>(bwd_dq_k_sc_smem[buf], g, t_coord.x, t_coord.z, physical_k_iter);
                                if constexpr (bwd_pair_single) {
                                    load_bwd_dq_k_col_fp4_tile_from_ptr<C>(
                                        bwd_dq_k1_smem[buf],
                                        g,
                                        g.backward_pair_k_col_fp4,
                                        t_coord.x,
                                        t_coord.z,
                                        physical_k_iter,
                                        cta_rank);
                                    load_bwd_dq_k_col_sc_tile_from_ptr<C>(
                                        bwd_dq_k1_sc_smem[buf],
                                        g,
                                        g.backward_pair_k_col_sc,
                                        t_coord.x,
                                        t_coord.z,
                                        physical_k_iter);
                                }
                            }
                        }
                        warpgroup::sync(warpgroup::groupid() + 1);
                        publish_shared_backing();
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            publish_cluster_shared_backing_if_needed<C>();
                            if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                tma::cluster::arrive(bwd_dq_k_remote_ready[buf], 0);
                            }
                        }
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                if (peer_cta_valid) {
                                    const int k_remote_phase = (k_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(bwd_dq_k_remote_ready[buf], k_remote_phase);
                                    k_remote_phase_mask ^= (1 << buf);
                                }
                            }
                        }
                        if constexpr (!fixed_fresh3_full) {
                        if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_LOAD_K_ONLY)) {
                            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                warpgroup::arrive(bwd_dq_src_finished[buf]);
                                if constexpr (bwd_dq_dual_halves) {
                                    warpgroup::arrive(bwd_dq_src1_finished[buf]);
                                }
                            }
                            continue;
                        }
                        }
                        if (bwd_dq_prepare_issue_active) {
                            if (issue_lane) {
                                warpgroup::arrive(bwd_dq_issue_ready[buf]);
                            }
                            continue;
                        }
                        if (issue_warp) {
                            auto &p_sc_e8m0 = *reinterpret_cast<typename G::bwd_dq_sc_tile *>(&p_sc_stage[buf]);
                            fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm, p_sc_e8m0);
                            fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm, bwd_dq_k_sc_smem[buf]);
                            if constexpr (bwd_dq_dual_halves) {
                                d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                fp4pv_load_mxnv_scale_async_cluster<C>(k1_sc_tm, bwd_dq_k1_sc_smem[buf]);
                            }
                            if constexpr (bwd_pair_single) {
                                d_tt_bwd_k_sc pair_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                fp4pv_load_mxnv_scale_async_cluster<C>(pair_sc_tm, bwd_dq_k1_sc_smem[buf]);
                            }
                            tensor_load_wait();
                            if constexpr (fixed_fresh3_full) {
                                tensor_before_thread_sync();
                                if (issue_lane) {
	                                    if (idx == 0) {
	                                        fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
	                                        if constexpr (bwd_dq_dual_halves) {
	                                            d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                                if constexpr (BWD_DQ_TAIL64_CLUSTER) {
                                                    d_tt_tail_outputs tt_output1 = tm_alloc.template allocate<d_tt_tail_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                    auto bwd_dq_k1_tail = bwd_dq_k1_smem[buf].template subtile<C::Dvo / 4, C::Nb / 2>(int2{0, 0});
                                                    fp4pv_mm_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_tail, p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                } else {
                                                    d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                    fp4pv_mm_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                }
	                                        }
	                                        if constexpr (bwd_pair_single) {
	                                            d_tt_outputs tt_pair_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
	                                            d_tt_bwd_k_sc pair_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
	                                            fp4pv_mm_cluster_ABt<C>(
	                                                tt_pair_output,
	                                                p_fp4_stage[buf],
	                                                bwd_dq_k1_smem[buf],
	                                                p_sc_tm,
	                                                pair_sc_tm,
	                                                bwd_dq_src1_finished[buf]);
	                                        }
	                                    } else {
	                                        fp4pv_mma_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
	                                        if constexpr (bwd_dq_dual_halves) {
	                                            d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                                if constexpr (BWD_DQ_TAIL64_CLUSTER) {
                                                    d_tt_tail_outputs tt_output1 = tm_alloc.template allocate<d_tt_tail_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                    auto bwd_dq_k1_tail = bwd_dq_k1_smem[buf].template subtile<C::Dvo / 4, C::Nb / 2>(int2{0, 0});
                                                    fp4pv_mma_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_tail, p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                } else {
                                                    d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                    fp4pv_mma_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                }
	                                        }
	                                        if constexpr (bwd_pair_single) {
	                                            d_tt_outputs tt_pair_output = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
	                                            d_tt_bwd_k_sc pair_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
	                                            fp4pv_mma_cluster_ABt<C>(
	                                                tt_pair_output,
	                                                p_fp4_stage[buf],
	                                                bwd_dq_k1_smem[buf],
	                                                p_sc_tm,
	                                                pair_sc_tm,
	                                                bwd_dq_src1_finished[buf]);
	                                        }
	                                    }
                                    if (idx == iters_per_task - 1) {
                                        detail::tcgen05::commit<C::CLUSTER_SIZE>(bwd_dq_accum_done[0]);
                                    }
                                }
                            } else {
                                if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY)) {
                                    tensor_before_thread_sync();
                                    if (issue_lane) {
                                        if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY)) {
                                            if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM)) {
                                                auto &dq_p_operand =
                                                    ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                                     (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM))
                                                        ? bwd_dq_p_dummy[0]
                                                        : p_fp4_stage[buf];
                                                auto &dq_k_operand =
                                                    ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM))
                                                        ? bwd_dq_k_dummy[0]
                                                        : bwd_dq_k_smem[buf];
                                                if (idx == 0) {
                                                    fp4pv_mm_unscaled_nosem_cluster_ABt<C>(tt_output, dq_p_operand, dq_k_operand);
                                                } else {
                                                    fp4pv_mma_unscaled_nosem_cluster_ABt<C>(tt_output, dq_p_operand, dq_k_operand);
                                                }
                                            } else if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE)) {
                                                if (idx == 0) {
                                                    fp4pv_mm_unscaled_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], bwd_dq_src_finished[buf]);
                                                } else {
                                                    fp4pv_mma_unscaled_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], bwd_dq_src_finished[buf]);
                                                }
                                            } else {
                                                if (idx == 0) {
                                                    if (bwd_dq_nosem_full) {
                                                        fp4pv_mm_nosem_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm);
                                                    } else {
                                                        fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
                                                    }
                                                    if constexpr (bwd_dq_dual_halves) {
                                                        d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                        d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                                        if (bwd_dq_nosem_full) {
                                                            fp4pv_mm_nosem_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm);
                                                        } else {
                                                            fp4pv_mm_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                        }
                                                    }
                                                } else {
                                                    if (bwd_dq_nosem_full) {
                                                        fp4pv_mma_nosem_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm);
                                                    } else {
                                                        fp4pv_mma_cluster_ABt<C>(tt_output, p_fp4_stage[buf], bwd_dq_k_smem[buf], p_sc_tm, k_sc_tm, bwd_dq_src_finished[buf]);
                                                    }
                                                    if constexpr (bwd_dq_dual_halves) {
                                                        d_tt_outputs tt_output1 = tm_alloc.template allocate<d_tt_outputs>(BWD_DQ_OUTPUT1_BASE);
                                                        d_tt_bwd_k_sc k1_sc_tm = tm_alloc.template allocate<d_tt_bwd_k_sc>(BWD_DQ_K1_SC_BASE);
                                                        if (bwd_dq_nosem_full) {
                                                            fp4pv_mma_nosem_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm);
                                                        } else {
                                                            fp4pv_mma_cluster_ABt<C>(tt_output1, p_fp4_stage[buf], bwd_dq_k1_smem[buf], p_sc_tm, k1_sc_tm, bwd_dq_src1_finished[buf]);
                                                        }
                                                    }
                                                }
                                            }
                                            if ((stage_cut != DEBUG_STAGE_CUT_FRESH3_FULL) &&
                                                (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL)) {
                                                tensor_commit<C::CLUSTER_SIZE>(bwd_dq_tmem_ready[0]);
                                                ++tmem_ready_count;
                                            }
                                            if ((stage_cut != DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) &&
                                                (stage_cut != DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) &&
                                                idx == iters_per_task - 1) {
                                                detail::tcgen05::commit<C::CLUSTER_SIZE>(bwd_dq_accum_done[0]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if constexpr (!fixed_fresh3_full) {
                            if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_SCALE_LOAD_ONLY) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_PRE_SYNC_ONLY)) {
                                continue;
                            }
                            if ((stage_cut == DEBUG_STAGE_CUT_BWD_DQ_ISSUE_NO_STORE) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_ISSUE_NO_STORE) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_UNSCALED_NOSEM_ISSUE_NO_STORE) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_P_NOSEM_WAIT_TMEM) ||
                                (stage_cut == DEBUG_STAGE_CUT_BWD_DQ_DUMMY_PK_NOSEM_WAIT_TMEM)) {
                                continue;
                            }
                        }
                    }
                    (void)tmem_ready_count;
                }
            }
        } else if constexpr (!QK_ONLY) {
            int p_pack_phase = 0;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                int m_tile = t_coord.y + cta_rank;
                if (m_tile >= tiles_m) {
                    continue;
                }
                int iters_per_task = executed_k_iters(min(t_coord.y + C::CLUSTER_SIZE, tiles_m));
                if (iters_per_task <= 0) {
                    continue;
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    constexpr int buf = 0;
                    const int consumer_tid = threadIdx.x;
                    wait(p_pack_ready[buf], p_pack_phase);
                    p_pack_phase ^= 1;
                    if constexpr (NORMALIZE_TILES) {
                        quantize_localcta_p_scales_from_scan_with_amax_and_store_raw<C>(
                            p_sc_stage[buf], p_bf16_scan[0], p_cta_amax[buf], consumer_tid,
                            g, t_coord.x, t_coord.z, m_tile, idx);
                        } else {
                            finalize_localcta_p_scales_with_amax<C>(
                                p_sc_stage[buf], p_cta_amax[buf], consumer_tid);
                        }
                    store_p_fp4_tile_to_global_warpgroup<C>(p_fp4_stage[buf], g, t_coord.x, t_coord.z, m_tile, idx, consumer_tid);
                    store_p_sc_prepared_to_global_warpgroup<C>(p_sc_stage[buf], g, t_coord.x, t_coord.z, m_tile, idx, consumer_tid);
                    tk_localcta::subgroup_barrier_sync<128>();
                    warpgroup::arrive(p_store_done[buf]);
                }
            }
        }
    }
}

__device__ inline float softmax_rescale_factor_strict(float row_max, float row_max_old, float scale_log2) {
    if (!isfinite(row_max_old) || !isfinite(row_max) || scale_log2 == 0.0f) {
        return 1.0f;
    }
    return fp4pv_exp2_approx((row_max_old - row_max) * scale_log2);
}

__device__ inline float softmax_rescale_factor_fast(float &row_max, float row_max_old, float scale_log2, float rescale_threshold = 8.f) {
    if (!isfinite(row_max_old) || !isfinite(row_max) || scale_log2 == 0.0f) {
        return 1.0f;
    }
    float acc_scale_log2 = (row_max_old - row_max) * scale_log2;
    if (!isfinite(acc_scale_log2) || acc_scale_log2 >= -rescale_threshold) {
        row_max = row_max_old;
        return 1.0f;
    }
    return fp4pv_exp2_approx(acc_scale_log2);
}

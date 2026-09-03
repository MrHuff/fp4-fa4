{
    constexpr int Stage = HAO_DIRECT_SOFTMAX_STAGE;
    static_assert(Stage == 0 || Stage == 1);
    using PairScaleC = hao_direct_pair_scale_stage_config<C, Stage>;
    constexpr bool InterleaveQ1Max =
        C::HAO_DIRECT_MX_Q1_MAX_INTERLEAVE && Stage == 1;
    constexpr bool SelfInterleaveQ1Max =
        C::HAO_DIRECT_MX_Q1_SELF_MAX &&
        (PairScaleC::HAO_DIRECT_MX_PAIR_SCALE_REUSE & 1) != 0;
    constexpr int OppositeScaleStage = 1 - Stage;
    constexpr uint32_t StageAffineMask =
        Stage == 0
            ? C::HAO_DIRECT_STAGE0_AFFINE_MASK
            : C::HAO_DIRECT_STAGE1_AFFINE_MASK;
    constexpr bool Q3CorrectionOffload =
        C::HAO_DIRECT_MX_Q3_CORR_WG && Stage == 0;
    constexpr bool DualQ3CorrectionOffload =
        C::HAO_DIRECT_MX_DUAL_Q3_CORR_WG ||
        C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG ||
        C::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG;
    constexpr bool DualQ3SmemOffload =
        C::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG;
    constexpr bool DualQ3TmemOffload =
        C::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG;
    constexpr bool Q01CorrectionOffload =
        C::HAO_DIRECT_NV_Q01_CORR_WG && Stage == 0;
    constexpr bool GlobalAnchorPrepackQ1 =
        C::HAO_DIRECT_NV_GLOBAL_ANCHOR_PREPACK_Q1 &&
        Stage == 0;
    constexpr bool GlobalAnchorCorrWG =
        C::HAO_DIRECT_NV_GLOBAL_ANCHOR_CORR_WG &&
        Stage == 0;
    constexpr bool GlobalAnchorDelayed =
        C::HAO_DIRECT_NV_GLOBAL_ANCHOR64 &&
        !GlobalAnchorCorrWG &&
        !GlobalAnchorPrepackQ1;
    const int reader_warp = warpgroup::warpid();
    const Score score_owned =
        tm.template allocate<Score>(Stage == 0 ? S0 : S1);
    const P p_owned = Stage == 0 ? p0 : p1;
    auto &p_scale_owned = p_sc_smem[Stage];

    int score_phase = 0;
    int corr_producer_phase = 1;
    int q3_sum_consumer_phase = 0;
    int q3_checkpoint_empty_phase = 0;
    int global_anchor_consumer_phase = 0;
    int nv_denom_done_phase = 0;
    int softmax_v_item = 1;

    // Stage-0 QK scales live in S1.  Before S1 exists, its reader supplies
    // the initial lease exactly as the HAO softmax-1 role does.
    if constexpr (Stage == 1) {
        if (warp::elect_leader()) {
            arrive(sfqk_load[0]);
        }
    }

    for (int logical_job = int(blockIdx.x);
         logical_job < C::LOGICAL_JOBS;
         logical_job += int(gridDim.x)) {
        const hao_direct_task task =
            hao_direct_decode_task<C>(logical_job);
        const int query_tile =
            Stage == 0 ? task.qtile0 : task.qtile1;
        const float scale_log2 =
            qk_scale_log2(g, task.batch, task.head);
        auto &row_max_owned = row_max_smem[Stage];
        auto &row_sum_owned = row_sum_smem[Stage];
        auto &correction_owned = correction_smem[Stage];

        if constexpr (!C::HAO_DIRECT_MX_DEFER_JOB_STATS_WAIT) {
            hao_direct_fp4pv_wait(
                softmax_corr_empty[Stage], corr_producer_phase);
            corr_producer_phase ^= 1;
        }

        float row_max =
            C::HAO_DIRECT_FIXED_P_CEILING ||
                    C::HAO_DIRECT_SCORE_PACK_CEILING ||
                    C::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX ||
                    C::HAO_DIRECT_MX_LOCAL_NATIVE_DEFERRED_SCALE
                ? 0.0f
                : base_types::constants<float>::neg_infty();
        float row_sum = 0.0f;
        int row_anchor_log2 = 0;
        float mx_anchor_quant_bias_base = 0.0f;
        float mx_anchor_affine_scale = 0.0f;
        uint8_t lagged_q1_e8m0 = 0;

        #pragma unroll 1
        for (int n = 0; n < C::N_TILES; ++n) {
            hao_direct_fp4pv_wait(
                score_full[Stage], score_phase);
            tensor_after_thread_sync();
            const uint32_t score_row_base =
                score_owned.addr + (reader_warp * 32 << 16);

            if constexpr (C::HAO_DIRECT_FIXED_P_CEILING) {
                if constexpr (
                    !C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                    stats_store(
                        correction_owned,
                        warpgroup::laneid(),
                        1.0f);
                    __threadfence_block();
                    if (warp::elect_leader()) {
                        arrive(softmax_corr_full[Stage]);
                    }
                }

                float tile_sum =
                    hao_direct_store_fixed_p_quarter<2, C>(
                        p_owned, p_scale_owned, reader_warp);
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<0, C>(
                        p_owned, p_scale_owned, reader_warp);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<1, C>(
                        p_owned, p_scale_owned, reader_warp);
                if constexpr (C::HAO_DIRECT_EARLY_P) {
                    if constexpr (
                        C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                        hao_direct_store_mxfp4_p_scale_half_direct<0>(
                            score_owned.addr + 32,
                            p_scale_owned,
                            reader_warp);
                    } else if constexpr (
                        C::HAO_DIRECT_SPARSE_SPLIT_K128) {
                        hao_direct_store_sparse_nvfp4_p_scale_half_direct<0>(
                            score_owned.addr + 32,
                            p_scale_owned,
                            reader_warp);
                    } else if constexpr (
                        C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE &&
                        !C::HAO_DIRECT_SPARSE_PV) {
                        hao_direct_store_nvfp4_p_scale_half_direct<0>(
                            score_owned.addr + 32,
                            p_scale_owned,
                            reader_warp);
                    }
                    fp4pv_publish_cta_shared_only();
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(p_first_o_rescaled[Stage]);
                    }
                }
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<3, C>(
                        p_owned, p_scale_owned, reader_warp);
                if constexpr (
                    C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                    hao_direct_store_mxfp4_p_scale_half_direct<1>(
                        score_owned.addr + 48,
                        p_scale_owned,
                        reader_warp);
                } else if constexpr (
                    C::HAO_DIRECT_SPARSE_SPLIT_K128) {
                    hao_direct_store_sparse_nvfp4_p_scale_half_direct<1>(
                        score_owned.addr + 36,
                        p_scale_owned,
                        reader_warp);
                } else if constexpr (
                    C::HAO_DIRECT_SPARSE_PV) {
                    hao_direct_store_sparse_nvfp4_p_scale_direct(
                        score_owned.addr + 32,
                        p_scale_owned,
                        reader_warp);
                } else if constexpr (
                    C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
                    hao_direct_store_nvfp4_p_scale_half_direct<1>(
                        score_owned.addr + 36,
                        p_scale_owned,
                        reader_warp);
                }

                fp4pv_publish_cta_shared_only();
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    if constexpr (!C::HAO_DIRECT_EARLY_P) {
                        arrive(p_first_o_rescaled[Stage]);
                    }
                    arrive(p_tail[Stage]);
                }

                if constexpr (
                    !C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                    hao_direct_fp4pv_wait(
                        softmax_corr_empty[Stage],
                        corr_producer_phase);
                    corr_producer_phase ^= 1;
                }
                row_sum += tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_SCORE_PACK_CEILING) {
                if constexpr (
                    !C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                    stats_store(
                        correction_owned,
                        warpgroup::laneid(),
                        1.0f);
                    __threadfence_block();
                    if (warp::elect_leader()) {
                        arrive(softmax_corr_full[Stage]);
                    }
                }

                score_quarter_regs scratch;
                upstream_load_score_quarter_x32<2, C>(
                    scratch, score_row_base, n, query_tile);
                float tile_sum =
                    hao_direct_pack_raw_score_quarter<2, C>(
                        p_owned,
                        p_scale_owned,
                        reader_warp,
                        scratch);
                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
                tile_sum +=
                    hao_direct_pack_raw_score_quarter<0, C>(
                        p_owned,
                        p_scale_owned,
                        reader_warp,
                        scratch);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }
                upstream_load_score_quarter_x32<1, C>(
                    scratch, score_row_base, n, query_tile);
                tile_sum +=
                    hao_direct_pack_raw_score_quarter<1, C>(
                        p_owned,
                        p_scale_owned,
                        reader_warp,
                        scratch);
                if constexpr (C::HAO_DIRECT_EARLY_P) {
                    if constexpr (
                        C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                        hao_direct_store_mxfp4_p_scale_half_direct<0>(
                            score_owned.addr + 32,
                            p_scale_owned,
                            reader_warp);
                    }
                    fp4pv_publish_cta_shared_only();
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(p_first_o_rescaled[Stage]);
                    }
                }
                upstream_load_score_quarter_x32<3, C>(
                    scratch, score_row_base, n, query_tile);
                tile_sum +=
                    hao_direct_pack_raw_score_quarter<3, C>(
                        p_owned,
                        p_scale_owned,
                        reader_warp,
                        scratch);
                if constexpr (
                    C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                    hao_direct_store_mxfp4_p_scale_half_direct<1>(
                        score_owned.addr + 48,
                        p_scale_owned,
                        reader_warp);
                }

                fp4pv_publish_cta_shared_only();
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    if constexpr (!C::HAO_DIRECT_EARLY_P) {
                        arrive(p_first_o_rescaled[Stage]);
                    }
                    arrive(p_tail[Stage]);
                }

                if constexpr (
                    !C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                    hao_direct_fp4pv_wait(
                        softmax_corr_empty[Stage],
                        corr_producer_phase);
                    corr_producer_phase ^= 1;
                }
                row_sum += tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_NV_FULL_ROW_STABILIZER) {
                static_assert(C::HAO_DIRECT_NVFP4_PV);
                static_assert(C::HAO_DIRECT_LOG2_QUANT);
                static_assert(C::HAO_DIRECT_NV_QUARTER_SCALE);
                static_assert(C::HAO_DIRECT_NV_DENOM_SOURCE == 1);
                static_assert(C::HAO_DIRECT_EARLY_P);

                score_quarter_regs q0_scores;
                score_quarter_regs q1_scores;
                score_quarter_regs q2_scores;
                score_quarter_regs q3_scores;
                float q0_max =
                    base_types::constants<float>::neg_infty();
                float q1_max =
                    base_types::constants<float>::neg_infty();
                float q2_max =
                    base_types::constants<float>::neg_infty();
                float q3_max =
                    base_types::constants<float>::neg_infty();

                upstream_load_score_quarter_x32<0, C, false>(
                    q0_scores, score_row_base, n, query_tile);
                upstream_load_score_quarter_x32<1, C, false>(
                    q1_scores, score_row_base, n, query_tile);
                tensor_load_wait();
                upstream_reduce_score_quarter<C>(
                    q0_scores, q0_max);
                upstream_reduce_score_quarter<C>(
                    q1_scores, q1_max);

                upstream_load_score_quarter_x32<2, C, false>(
                    q2_scores, score_row_base, n, query_tile);
                upstream_load_score_quarter_x32<3, C, false>(
                    q3_scores, score_row_base, n, query_tile);
                tensor_load_wait();
                upstream_reduce_score_quarter<C>(
                    q2_scores, q2_max);
                upstream_reduce_score_quarter<C>(
                    q3_scores, q3_max);

                const float tile_max = fmaxf(
                    fmaxf(q0_max, q1_max),
                    fmaxf(q2_max, q3_max));
                const float old_max = row_max;
                row_max = fmaxf(row_max, tile_max);
                const float correction =
                    n == 0
                        ? 1.0f
                        : softmax_rescale_factor_strict(
                              row_max, old_max, scale_log2);
                stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    correction);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                // All score values are now register-resident. Both the Q/K
                // scale writer and packed P may safely reclaim score TMEM.
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }

                float tile_sum =
                    hao_direct_transform_pack_fp4_quarter<
                        2,
                        C,
                        (StageAffineMask & 0x4u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            q2_scores,
                            row_max,
                            scale_log2,
                            q2_max);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        0,
                        C,
                        (StageAffineMask & 0x1u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            q0_scores,
                            row_max,
                            scale_log2,
                            q0_max);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        1,
                        C,
                        (StageAffineMask & 0x2u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            q1_scores,
                            row_max,
                            scale_log2,
                            q1_max);

                if constexpr (
                    C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
                    hao_direct_store_nvfp4_p_scale_half_direct<0>(
                        score_owned.addr + 32,
                        p_scale_owned,
                        reader_warp);
                } else {
                    fp4pv_publish_cta_shared_only();
                }
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                }

                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        3,
                        C,
                        (StageAffineMask & 0x8u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            q3_scores,
                            row_max,
                            scale_log2,
                            q3_max);

                if constexpr (
                    C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
                    hao_direct_store_nvfp4_p_scale_half_direct<1>(
                        score_owned.addr + 36,
                        p_scale_owned,
                        reader_warp);
                } else {
                    fp4pv_publish_cta_shared_only();
                }
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_tail[Stage]);
                }

                hao_direct_fp4pv_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum = row_sum * correction + tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_NV_Q0_STABILIZER) {
                static_assert(C::HAO_DIRECT_NVFP4_PV);
                static_assert(C::HAO_DIRECT_LOG2_QUANT);
                static_assert(C::HAO_DIRECT_NV_QUARTER_SCALE);
                static_assert(C::HAO_DIRECT_NV_DENOM_SOURCE == 1);
                static_assert(C::HAO_DIRECT_EARLY_P);

                score_quarter_regs retained_q0;
                score_quarter_regs scratch;
                float q0_max =
                    base_types::constants<float>::neg_infty();
                float quarter_max =
                    base_types::constants<float>::neg_infty();

                // Q0 is a causal-safe N32 anchor sample. Retain it while Q2
                // is consumed so every score quarter is loaded exactly once.
                upstream_load_score_quarter_x32<0, C, false>(
                    retained_q0,
                    score_row_base,
                    n,
                    query_tile);
                upstream_load_score_quarter_x32<2, C, false>(
                    scratch,
                    score_row_base,
                    n,
                    query_tile);
                tensor_load_wait();
                upstream_reduce_score_quarter<C>(
                    retained_q0, q0_max);

                const float old_max = row_max;
                row_max = fmaxf(row_max, q0_max);
                const float correction =
                    n == 0
                        ? 1.0f
                        : softmax_rescale_factor_strict(
                              row_max, old_max, scale_log2);
                stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    correction);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                float tile_sum =
                    hao_direct_transform_pack_fp4_quarter<
                        2,
                        C,
                        (StageAffineMask & 0x4u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max);

                // Q0 is already register-resident and Q2 has retired, so the
                // opposite score-scale writer can reclaim its TMEM columns.
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        0,
                        C,
                        (StageAffineMask & 0x1u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            retained_q0,
                            row_max,
                            scale_log2,
                            q0_max);

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<1, C>(
                    scratch,
                    score_row_base,
                    n,
                    query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        1,
                        C,
                        (StageAffineMask & 0x2u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max);

                if constexpr (
                    C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
                    hao_direct_store_nvfp4_p_scale_half_direct<0>(
                        score_owned.addr + 32,
                        p_scale_owned,
                        reader_warp);
                } else {
                    fp4pv_publish_cta_shared_only();
                }
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                }

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<3, C>(
                    scratch,
                    score_row_base,
                    n,
                    query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        3,
                        C,
                        (StageAffineMask & 0x8u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max);

                if constexpr (
                    C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
                    hao_direct_store_nvfp4_p_scale_half_direct<1>(
                        score_owned.addr + 36,
                        p_scale_owned,
                        reader_warp);
                } else {
                    fp4pv_publish_cta_shared_only();
                }
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_tail[Stage]);
                }

                hao_direct_fp4pv_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum = row_sum * correction + tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_NV_PREVIEW_STABILIZER) {
                static_assert(C::HAO_DIRECT_NVFP4_PV);
                static_assert(C::HAO_DIRECT_LOG2_QUANT);
                static_assert(C::HAO_DIRECT_NV_QUARTER_SCALE);
                static_assert(C::HAO_DIRECT_NV_DENOM_SOURCE == 1);
                static_assert(C::HAO_DIRECT_EARLY_P);

                score_quarter_regs scratch;
                float preview_max =
                    base_types::constants<float>::neg_infty();
                float quarter_max =
                    base_types::constants<float>::neg_infty();

                // Sample every N32 quarter before the one-pass consume. The
                // staggered offsets avoid concentrating the estimate in one
                // contiguous part of the score row.
                if constexpr (C::HAO_DIRECT_NV_PREVIEW_WIDTH == 8) {
                    float2 preview0[4];
                    float2 preview1[4];
                    upstream_load_score_sixteenth_tmem<0, 0, C, false>(
                        preview0, score_row_base, n, query_tile);
                    upstream_load_score_sixteenth_tmem<1, 1, C, false>(
                        preview1, score_row_base, n, query_tile);
                    tensor_load_wait();
                    upstream_reduce_score_pairs<4>(
                        preview0, preview_max);
                    upstream_reduce_score_pairs<4>(
                        preview1, preview_max);
                    upstream_load_score_sixteenth_tmem<2, 2, C, false>(
                        preview0, score_row_base, n, query_tile);
                    upstream_load_score_sixteenth_tmem<3, 3, C, false>(
                        preview1, score_row_base, n, query_tile);
                    tensor_load_wait();
                    upstream_reduce_score_pairs<4>(
                        preview0, preview_max);
                    upstream_reduce_score_pairs<4>(
                        preview1, preview_max);
                } else {
                    upstream_load_score_quarter_half_x16<0, 0, C, false>(
                        scratch, score_row_base, n, query_tile);
                    upstream_load_score_quarter_half_x16<1, 1, C, false>(
                        scratch, score_row_base, n, query_tile);
                    tensor_load_wait();
                    upstream_reduce_score_pairs<8>(
                        &scratch.v0, preview_max);
                    upstream_reduce_score_pairs<8>(
                        &scratch.v8, preview_max);
                    upstream_load_score_quarter_half_x16<2, 0, C, false>(
                        scratch, score_row_base, n, query_tile);
                    upstream_load_score_quarter_half_x16<3, 1, C, false>(
                        scratch, score_row_base, n, query_tile);
                    tensor_load_wait();
                    upstream_reduce_score_pairs<8>(
                        &scratch.v0, preview_max);
                    upstream_reduce_score_pairs<8>(
                        &scratch.v8, preview_max);
                }

                preview_max +=
                    0.125f * static_cast<float>(
                        C::HAO_DIRECT_NV_PREVIEW_BIAS_X8);
                const float old_max = row_max;
                row_max = fmaxf(row_max, preview_max);
                const float correction =
                    n == 0
                        ? 1.0f
                        : softmax_rescale_factor_strict(
                              row_max, old_max, scale_log2);
                stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    correction);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                upstream_load_score_quarter_x32<2, C>(
                    scratch,
                    score_row_base,
                    n,
                    query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                float tile_sum =
                    hao_direct_transform_pack_fp4_quarter<
                        2,
                        C,
                        (StageAffineMask & 0x4u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max);

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<0, C>(
                    scratch,
                    score_row_base,
                    n,
                    query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        0,
                        C,
                        (StageAffineMask & 0x1u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max);

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<1, C>(
                    scratch,
                    score_row_base,
                    n,
                    query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        1,
                        C,
                        (StageAffineMask & 0x2u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max);

                if constexpr (
                    C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
                    hao_direct_store_nvfp4_p_scale_half_direct<0>(
                        score_owned.addr + 32,
                        p_scale_owned,
                        reader_warp);
                } else {
                    fp4pv_publish_cta_shared_only();
                }
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                }

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<3, C>(
                    scratch,
                    score_row_base,
                    n,
                    query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<
                        3,
                        C,
                        (StageAffineMask & 0x8u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max);

                if constexpr (
                    C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
                    hao_direct_store_nvfp4_p_scale_half_direct<1>(
                        score_owned.addr + 36,
                        p_scale_owned,
                        reader_warp);
                } else {
                    fp4pv_publish_cta_shared_only();
                }
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_tail[Stage]);
                }

                hao_direct_fp4pv_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum = row_sum * correction + tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_MX_SAMPLED_STABILIZER) {
                static_assert(C::HAO_DIRECT_MXFP4_PV);
                static_assert(C::HAO_DIRECT_MX_LOG2_QUANT);
                static_assert(!C::HAO_DIRECT_EARLY_P);

                score_quarter_regs scratch;
                float quarter_max =
                    base_types::constants<float>::neg_infty();

                // Quarter 2 supplies a stable common shift for this score
                // tile and is immediately consumed. Each remaining quarter
                // supplies only its local block maximum, so no score quarter
                // needs to be loaded twice.
                upstream_load_score_quarter_x32<2, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);

                const float old_max = row_max;
                row_max = fmaxf(row_max, quarter_max);
                const float correction =
                    n == 0
                        ? 1.0f
                        : softmax_rescale_factor_strict(
                              row_max, old_max, scale_log2);
                stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    correction);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                float tile_sum =
                    hao_direct_transform_pack_fp4_quarter<2, C>(
                        p_owned,
                        p_scale_owned,
                        reader_warp,
                        scratch,
                        row_max,
                        scale_log2,
                        quarter_max);

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<0, C>(
                        p_owned,
                        p_scale_owned,
                        reader_warp,
                        scratch,
                        row_max,
                        scale_log2,
                        quarter_max);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }

                score_quarter_regs paired_tail;
                if constexpr (C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                    upstream_load_score_quarter_x32<3, C, false>(
                        scratch, score_row_base, n, query_tile);
                    upstream_load_score_quarter_x32<1, C, false>(
                        paired_tail, score_row_base, n, query_tile);
                    tensor_load_wait();
                } else {
                    upstream_load_score_quarter_x32<3, C>(
                        scratch, score_row_base, n, query_tile);
                }
                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<3, C>(
                        p_owned,
                        p_scale_owned,
                        reader_warp,
                        scratch,
                        row_max,
                        scale_log2,
                        quarter_max);

                quarter_max =
                    base_types::constants<float>::neg_infty();
                if constexpr (C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                    upstream_reduce_score_quarter<C>(
                        paired_tail, quarter_max);
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<1, C>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            paired_tail,
                            row_max,
                            scale_log2,
                            quarter_max);
                } else {
                    upstream_load_score_quarter_x32<1, C>(
                        scratch, score_row_base, n, query_tile);
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max);
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<1, C>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max);
                }

                fp4pv_publish_cta_shared_only();
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                    arrive(p_tail[Stage]);
                }

                hao_direct_fp4pv_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum = row_sum * correction + tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX) {
                static_assert(C::HAO_DIRECT_FP4_PV);
                static_assert(
                    C::HAO_DIRECT_MX_LOG2_QUANT ||
                    C::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX);
                static_assert(
                    !C::HAO_DIRECT_EARLY_P ||
                    C::HAO_DIRECT_MX_EARLY_P ||
                    C::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX);
                static_assert(
                    !C::HAO_DIRECT_MX_EARLY_P ||
                    C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE);
                static_assert(
                    !C::HAO_DIRECT_EARLY_P ||
                    !C::HAO_DIRECT_MX_HALF_PREFETCH_Q1);
                if constexpr (
                    !C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                    stats_store(
                        correction_owned,
                        warpgroup::laneid(),
                        1.0f);
                    __threadfence_block();
                    if (warp::elect_leader()) {
                        arrive(softmax_corr_full[Stage]);
                    }
                }

                if constexpr (Q01CorrectionOffload) {
                    static_assert(C::HAO_DIRECT_NVFP4_PV);
                    static_assert(C::HAO_DIRECT_P_QUARTER_MAX);

                    score_quarter_regs q2_scores;
                    score_quarter_regs q3_scores;
                    upstream_load_score_quarter_x32<2, C, false>(
                        q2_scores,
                        score_row_base,
                        n,
                        query_tile);
                    upstream_load_score_quarter_x32<3, C, false>(
                        q3_scores,
                        score_row_base,
                        n,
                        query_tile);
                    tensor_load_wait();

                    // The first P half overlays score Q2. Release its writer
                    // only after both tail score quarters are resident.
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(dual_q2_ready[Stage]);
                    }

                    float q2_max =
                        base_types::constants<float>::neg_infty();
                    float q3_max =
                        base_types::constants<float>::neg_infty();
                    upstream_reduce_score_quarter<C>(
                        q2_scores, q2_max);
                    upstream_reduce_score_quarter<C>(
                        q3_scores, q3_max);

                    // Different warp groups cannot safely issue concurrent
                    // stores into one P TMEM page, even to disjoint columns.
                    // Keep the max work overlapped, then transfer ownership.
                    hao_direct_fp4pv_wait(
                        p_first_o_rescaled[Stage],
                        score_phase);
                    float tile_sum =
                        hao_direct_transform_pack_fp4_quarter<
                            2,
                            C,
                            (StageAffineMask & 0x4u) != 0>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                q2_scores,
                                0.0f,
                                scale_log2,
                                q2_max);
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            3,
                            C,
                            (StageAffineMask & 0x8u) != 0>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                q3_scores,
                                0.0f,
                                scale_log2,
                                q3_max);
                    hao_direct_store_nvfp4_p_scale_half_direct<1>(
                        score_owned.addr + 36,
                        p_scale_owned,
                        reader_warp);
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(p_tail[Stage]);
                    }

                    row_sum += tile_sum;
                    score_phase ^= 1;
                    continue;
                }

                float tile_sum;
                if constexpr (
                    C::HAO_DIRECT_MX_DELAYED_HALF_Q2) {
                    score_quarter_regs scratch;
                    score_quarter_regs delayed_q2;
                    score_quarter_regs prefetched_q1;
                    score_quarter_regs retained_anchor_q3;
                    uint32_t retained_first_words[2][4];
                    uint32_t mx_denom_payload_words[4];
                    float quarter_max =
                        base_types::constants<float>::neg_infty();
                    float q0_anchor_max =
                        base_types::constants<float>::neg_infty();
                    float prefetched_q1_max =
                        base_types::constants<float>::neg_infty();

                    // Q0 packing overwrites the low half of score Q2. Retain
                    // that half first, then pair each remaining score load.
                    upstream_load_score_quarter_half_x16<
                        2, 0, C, false>(
                            delayed_q2,
                            score_row_base,
                            n,
                            query_tile);
                    upstream_load_score_quarter_x32<0, C, false>(
                        scratch,
                        score_row_base,
                        n,
                        query_tile);
                    tensor_load_wait();
                    if constexpr (
                        C::HAO_DIRECT_MX_GLOBAL_ANCHOR128) {
                        if (n == 0) {
                            upstream_load_score_quarter_half_x16<
                                2, 1, C, false>(
                                    delayed_q2,
                                    score_row_base,
                                    n,
                                    query_tile);
                            upstream_load_score_quarter_x32<
                                1, C, false>(
                                    prefetched_q1,
                                    score_row_base,
                                    n,
                                    query_tile);
                            upstream_load_score_quarter_x32<
                                3, C, false>(
                                    retained_anchor_q3,
                                    score_row_base,
                                    n,
                                    query_tile);
                            tensor_load_wait();
                        }
                    }
                    if constexpr (
                        C::HAO_DIRECT_MX_PWL_EXP2_MODE == 23 ||
                        C::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX) {
                        if constexpr (
                            GlobalAnchorCorrWG) {
                            if (n != 0) {
                                tensor_before_thread_sync();
                                if (warp::elect_leader()) {
                                    arrive(
                                        sfqk_load[
                                            OppositeScaleStage]);
                                }
                            }
                        } else {
                            tensor_before_thread_sync();
                            if (warp::elect_leader()) {
                                arrive(
                                    sfqk_load[
                                        OppositeScaleStage]);
                            }
                        }
                    }
                    if constexpr (
                        Stage == 0 ||
                        (Stage == 1 &&
                         C::HAO_DIRECT_MX_STAGE1_EARLY_Q1)) {
                        if constexpr (
                            C::HAO_DIRECT_MX_GLOBAL_ANCHOR128) {
                            if (n != 0) {
                                upstream_load_score_quarter_x32<
                                    1, C, false>(
                                        prefetched_q1,
                                        score_row_base,
                                        n,
                                        query_tile);
                                upstream_load_score_quarter_half_x16<
                                    2, 1, C, false>(
                                        delayed_q2,
                                        score_row_base,
                                        n,
                                        query_tile);
                            }
                        } else {
                            upstream_load_score_quarter_x32<
                                1, C, false>(
                                    prefetched_q1,
                                    score_row_base,
                                    n,
                                    query_tile);
                            upstream_load_score_quarter_half_x16<
                                2, 1, C, false>(
                                    delayed_q2,
                                    score_row_base,
                                    n,
                                    query_tile);
                        }
                    } else if constexpr (
                        GlobalAnchorPrepackQ1) {
                        if (n == 0) {
                            upstream_load_score_quarter_half_x16<
                                2, 1, C, false>(
                                    delayed_q2,
                                    score_row_base,
                                    n,
                                    query_tile);
                            upstream_load_score_quarter_x32<1, C, false>(
                                prefetched_q1,
                                score_row_base,
                                n,
                                query_tile);
                        }
                    }
                    if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                        upstream_reduce_score_quarter<C>(
                            scratch, quarter_max);
                    }
                    if constexpr (
                        C::HAO_DIRECT_MX_GLOBAL_ANCHOR128) {
                        if (n == 0) {
                            q0_anchor_max = quarter_max;
                            tensor_load_wait();
                            float q1_anchor_max =
                                base_types::constants<float>::
                                    neg_infty();
                            float q2_anchor_max =
                                base_types::constants<float>::
                                    neg_infty();
                            float q3_anchor_max =
                                base_types::constants<float>::
                                    neg_infty();
                            upstream_reduce_score_quarter<C>(
                                prefetched_q1,
                                q1_anchor_max);
                            upstream_reduce_score_quarter<C>(
                                delayed_q2,
                                q2_anchor_max);
                            tensor_load_wait();
                            upstream_reduce_score_quarter<C>(
                                retained_anchor_q3,
                                q3_anchor_max);
                            row_max = fmaxf(
                                fmaxf(
                                    q0_anchor_max,
                                    q1_anchor_max),
                                fmaxf(
                                    q2_anchor_max,
                                    q3_anchor_max));
                            row_max +=
                                C::HAO_DIRECT_MX_GLOBAL_ANCHOR_BIAS;
                        }
                    } else if constexpr (
                        GlobalAnchorCorrWG) {
                        static_assert(
                            C::HAO_DIRECT_P_QUARTER_MAX);
                        if (n == 0) {
                            hao_direct_fp4pv_wait(
                                global_anchor_full[Stage],
                                global_anchor_consumer_phase);
                            q0_anchor_max = quarter_max;
                            row_max =
                                fmaxf(
                                    q0_anchor_max,
                                    stats_load(
                                        row_max_owned,
                                        warpgroup::laneid())) *
                                    C::HAO_DIRECT_NV_GLOBAL_ANCHOR_SCALE +
                                C::HAO_DIRECT_NV_GLOBAL_ANCHOR_BIAS;
                            global_anchor_consumer_phase ^= 1;
                            tensor_before_thread_sync();
                            if (warp::elect_leader()) {
                                arrive(
                                    sfqk_load[
                                        OppositeScaleStage]);
                            }
                        }
                    } else if constexpr (
                        C::HAO_DIRECT_GLOBAL_ANCHOR) {
                        if (n == 0) {
                            q0_anchor_max = quarter_max;
                            if constexpr (C::HAO_DIRECT_MXFP4_PV) {
                                row_max =
                                    quarter_max +
                                    C::HAO_DIRECT_MX_GLOBAL_ANCHOR_BIAS;
                            } else if constexpr (
                                !GlobalAnchorPrepackQ1) {
                                row_max = quarter_max;
                                row_max *=
                                    C::HAO_DIRECT_NV_GLOBAL_ANCHOR_SCALE;
                                row_max +=
                                    C::HAO_DIRECT_NV_GLOBAL_ANCHOR_BIAS;
                            }
                        }
                    }
                    if constexpr (
                        GlobalAnchorPrepackQ1) {
                        if (n == 0) {
                            tensor_load_wait();
                            upstream_reduce_score_quarter<C>(
                                prefetched_q1,
                                prefetched_q1_max);
                            row_max =
                                fmaxf(
                                    q0_anchor_max,
                                    prefetched_q1_max) *
                                    C::HAO_DIRECT_NV_GLOBAL_ANCHOR_SCALE +
                                C::HAO_DIRECT_NV_GLOBAL_ANCHOR_BIAS;
                        }
                    }
                    float shiftless_row_max =
                        C::HAO_DIRECT_GLOBAL_ANCHOR
                            ? row_max
                            : 0.0f;
                    if constexpr (
                        C::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST) {
                        if (n == 0) {
                            constexpr float LOG2_FP4_MAX =
                                2.584962500721156f;
                            mx_anchor_quant_bias_base =
                                -shiftless_row_max * scale_log2 +
                                LOG2_FP4_MAX;
                            mx_anchor_affine_scale =
                                C::HAO_DIRECT_MX_AFFINE_A * scale_log2;
                        }
                    }
                    uint16_t q0_scale_bits = 0;
                    uint8_t q0_mx_scale = 0;
                    uint8_t q01_pair_e8m0 = 0;
                    uint8_t q23_pair_e8m0 = 0;
                    const float q0_pair_max = quarter_max;
                    if constexpr (
                        InterleaveQ1Max) {
                        tensor_load_wait();
                        quarter_max =
                            base_types::constants<float>::neg_infty();
                    }
                    float q0_unscaled_sum = 0.0f;
                    float q0_sum;
                    if constexpr (
                        C::HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF ||
                        GlobalAnchorDelayed) {
                        q0_sum =
                            hao_direct_transform_pack_nvfp4_quarter_register_scale<
                                0,
                                C,
                                (StageAffineMask & 0x1u) != 0,
                                C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                shiftless_row_max,
                                scale_log2,
                                quarter_max,
                                q0_scale_bits,
                                GlobalAnchorDelayed
                                    ? &q0_unscaled_sum
                                    : nullptr,
                                retained_first_words[0]);
                    } else {
                        q0_sum =
                            hao_direct_transform_pack_fp4_quarter<
                                0,
                                PairScaleC,
                                (StageAffineMask & 0x1u) != 0,
                                C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM,
                                C::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST,
                                (C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF ||
                                 C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH),
                                InterleaveQ1Max>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                shiftless_row_max,
                                scale_log2,
                                q0_pair_max,
                                mx_denom_payload_words,
                                mx_anchor_quant_bias_base,
                                mx_anchor_affine_scale,
                                &q0_mx_scale,
                                &q01_pair_e8m0,
                                true,
                                &prefetched_q1,
                                &quarter_max);
                        if constexpr (
                            C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH) {
                            hao_direct_store_mxfp4_scale_byte_linear(
                                p_scale_owned,
                                reader_warp * 32 + warp::laneid(),
                                0,
                                q0_mx_scale);
                        }
                        if constexpr (
                            C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM) {
                            hao_direct_store_mx_denom_payload_quarter_smem<0>(
                                mx_denom_payload_smem[Stage],
                                reader_warp,
                                mx_denom_payload_words);
                        }
                    }
                    if constexpr (
                        C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF) {
                        hao_direct_store_nv_first_payload_quarter_smem<0>(
                            nv_first_payload_smem[Stage],
                            reader_warp,
                            retained_first_words[0]);
                    }
                    if constexpr (
                        (C::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK &
                         (1 << Stage)) != 0) {
                        const int v_slot =
                            softmax_v_item % C::KV_STAGES;
                        const int v_phase =
                            (softmax_v_item / C::KV_STAGES) & 1;
                        if (reader_warp == 0 &&
                            warp::elect_leader()) {
                            hao_direct_fp4pv_wait(
                                kv_full[v_slot], v_phase);
                            if constexpr (Stage == 1) {
                                if (n + 1 < C::N_TILES) {
                                    hao_direct_fp4pv_wait(
                                        score_full[0],
                                        score_phase ^ 1);
                                    tensor_after_thread_sync();
                                }
                            }
                            hao_direct_load_nvfp4_scale_ranklocal<0>(
                                score_owned.addr + 8,
                                kv_scale[v_slot].v);
                            hao_direct_load_nvfp4_scale_ranklocal<1>(
                                score_owned.addr + 12,
                                kv_scale[v_slot].v);
                        }
                    }

                    if constexpr (
                        DualQ3SmemOffload ||
                        DualQ3TmemOffload) {
                        upstream_load_score_quarter_x32<3, C>(
                            scratch,
                            score_row_base,
                            n,
                            query_tile);
                        if constexpr (DualQ3SmemOffload) {
                            hao_direct_store_q3_fp8_smem<C>(
                                q3_stage_smem[Stage],
                                reader_warp,
                                scratch);
                            __threadfence_block();
                        } else {
                            hao_direct_store_q3_fp8_tmem(
                                p_owned,
                                reader_warp,
                                scratch);
                            tensor_before_thread_sync();
                        }
                        if (warp::elect_leader()) {
                            arrive(dual_q3_staged[Stage]);
                        }
                    }
                    if constexpr (
                        C::HAO_DIRECT_MX_PWL_EXP2_MODE != 23 &&
                        !C::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX &&
                        !GlobalAnchorCorrWG) {
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(sfqk_load[OppositeScaleStage]);
                        }
                    }

                    if constexpr (
                        !InterleaveQ1Max) {
                        quarter_max =
                            base_types::constants<float>::neg_infty();
                    }
                    if constexpr (
                        Stage == 1 &&
                        !C::HAO_DIRECT_MX_STAGE1_EARLY_Q1) {
                        if constexpr (
                            C::HAO_DIRECT_MX_GLOBAL_ANCHOR128) {
                            if (n != 0) {
                                upstream_load_score_quarter_half_x16<
                                    2, 1, C, false>(
                                        delayed_q2,
                                        score_row_base,
                                        n,
                                        query_tile);
                                upstream_load_score_quarter_x32<
                                    1, C, false>(
                                        prefetched_q1,
                                        score_row_base,
                                        n,
                                        query_tile);
                            }
                        } else if constexpr (
                            GlobalAnchorPrepackQ1) {
                            if (n != 0) {
                                upstream_load_score_quarter_half_x16<
                                    2, 1, C, false>(
                                        delayed_q2,
                                        score_row_base,
                                        n,
                                        query_tile);
                                upstream_load_score_quarter_x32<
                                    1, C, false>(
                                        prefetched_q1,
                                        score_row_base,
                                        n,
                                        query_tile);
                            }
                        } else {
                            upstream_load_score_quarter_half_x16<
                                2, 1, C, false>(
                                    delayed_q2,
                                    score_row_base,
                                    n,
                                    query_tile);
                            upstream_load_score_quarter_x32<1, C, false>(
                                prefetched_q1,
                                score_row_base,
                                n,
                                query_tile);
                        }
                    }
                    bool reuse_q1_pair_scale = false;
                    if constexpr (
                        (PairScaleC::HAO_DIRECT_MX_PAIR_SCALE_REUSE & 1) !=
                        0) {
                        if constexpr (
                            PairScaleC::HAO_DIRECT_MX_PAIR_SCALE_PERIOD ==
                            1) {
                            reuse_q1_pair_scale = true;
                        } else {
                            reuse_q1_pair_scale =
                                (n &
                                 (PairScaleC::
                                      HAO_DIRECT_MX_PAIR_SCALE_PERIOD -
                                  1)) == 0;
                        }
                    }
                    if constexpr (
                        !InterleaveQ1Max) {
                        tensor_load_wait();
                    }
                    if constexpr (
                        C::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD) {
                        hao_direct_fp4pv_wait(
                            q1_scale_full[Stage], score_phase);
                        const int q1_scale_row =
                            reader_warp * 32 + warp::laneid();
                        const float transferred_q1_max =
                            q1_scale_smem[Stage].raw_max[q1_scale_row];
                        q01_pair_e8m0 =
                            hao_direct_select_mxfp4_e8m0<PairScaleC>(
                                transferred_q1_max * scale_log2 +
                                0.5f * static_cast<float>(
                                    C::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2));
                    }
                    if constexpr (
                        PairScaleC::HAO_DIRECT_MX_Q1_PAIR_PROBE != 0) {
                        if (reuse_q1_pair_scale) {
                            const float q1_probe =
                                hao_direct_q1_pair_scale_probe<
                                    PairScaleC::HAO_DIRECT_MX_Q1_PAIR_PROBE>(
                                        prefetched_q1);
                            const float delta_log2 = __fmaf_rn(
                                q1_probe - q0_pair_max,
                                scale_log2,
                                PairScaleC::
                                    HAO_DIRECT_MX_Q1_PAIR_PROBE_BIAS);
                            int adjustment = delta_log2 > 0.5f ? 1 : 0;
                            adjustment =
                                delta_log2 < -0.5f ? -1 : adjustment;
                            int adjusted_scale =
                                static_cast<int>(q01_pair_e8m0) +
                                adjustment;
                            asm(
                                "max.s32 %0, %0, 1;\n\t"
                                "min.s32 %0, %0, 254;"
                                : "+r"(adjusted_scale));
                            q01_pair_e8m0 =
                                static_cast<uint8_t>(adjusted_scale);
                        }
                    }
                    if constexpr (
                        C::HAO_DIRECT_P_QUARTER_MAX &&
                        !InterleaveQ1Max) {
                        if (!reuse_q1_pair_scale) {
                            if constexpr (
                                GlobalAnchorPrepackQ1) {
                                if (n == 0) {
                                    quarter_max = prefetched_q1_max;
                                } else {
                                    hao_direct_reduce_q1_score_quarter<C>(
                                        prefetched_q1, quarter_max);
                                }
                            } else {
                                hao_direct_reduce_q1_score_quarter<C>(
                                    prefetched_q1, quarter_max);
                            }
                        }
                    }
                    if constexpr (
                        C::HAO_DIRECT_NV_GLOBAL_ANCHOR64 &&
                        !GlobalAnchorCorrWG &&
                        !GlobalAnchorPrepackQ1) {
                        if (n == 0) {
                            row_max =
                                fmaxf(q0_anchor_max, quarter_max) *
                                    C::HAO_DIRECT_NV_GLOBAL_ANCHOR_SCALE +
                                C::HAO_DIRECT_NV_GLOBAL_ANCHOR_BIAS;
                            shiftless_row_max = row_max;

                            constexpr float LOG2_FP4_MAX =
                                2.584962500721156f;
                            const float q0_stored_log2_scale =
                                (q0_anchor_max - row_max) * scale_log2 -
                                LOG2_FP4_MAX +
                                C::HAO_DIRECT_NV_P_GLOBAL_LOG2;
                            int q0_scale_code =
                                hao_direct_float_to_s32_rn_sat(
                                    __fmaf_rn(
                                        q0_stored_log2_scale,
                                        8.0f,
                                        56.0f));
                            asm(
                                "max.s32 %0, %0, 8;\n\t"
                                "min.s32 %0, %0, 126;"
                                : "+r"(q0_scale_code));
                            q0_scale_bits =
                                static_cast<uint16_t>(q0_scale_code) |
                                (static_cast<uint16_t>(
                                     q0_scale_code) << 8);
                            const uint32_t q0_fp32_scale_bits =
                                (static_cast<uint32_t>(
                                     q0_scale_code) << 20) +
                                (120u << 23);
                            q0_sum =
                                __uint_as_float(q0_fp32_scale_bits) *
                                q0_unscaled_sum;
                        }
                    }
                    uint16_t q1_scale_bits = 0;
                    uint8_t q1_mx_scale = 0;
                    float lagged_q1_raw_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (
                        C::HAO_DIRECT_MX_LAGGED_Q1_SCALE) {
                        q01_pair_e8m0 =
                            n == 0 ? q01_pair_e8m0 : lagged_q1_e8m0;
                    }
                    float q1_sum;
                    if constexpr (
                        C::HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF ||
                        GlobalAnchorDelayed) {
                        q1_sum =
                            hao_direct_transform_pack_nvfp4_quarter_register_scale<
                                1,
                                C,
                                (StageAffineMask & 0x2u) != 0,
                                C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                prefetched_q1,
                                shiftless_row_max,
                                scale_log2,
                                quarter_max,
                                q1_scale_bits,
                                nullptr,
                                retained_first_words[1]);
                    } else {
                        q1_sum =
                            hao_direct_transform_pack_fp4_quarter<
                                1,
                                PairScaleC,
                                (StageAffineMask & 0x2u) != 0,
                                C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM,
                                C::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST,
                                (C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF ||
                                 C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH ||
                                 C::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD),
                                (C::HAO_DIRECT_MX_LAGGED_Q1_SCALE ||
                                 SelfInterleaveQ1Max),
                                (SelfInterleaveQ1Max &&
                                 C::HAO_DIRECT_MX_Q1_SELF_MAX >= 2)>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                prefetched_q1,
                                shiftless_row_max,
                                scale_log2,
                                quarter_max,
                                mx_denom_payload_words,
                                mx_anchor_quant_bias_base,
                                mx_anchor_affine_scale,
                                &q1_mx_scale,
                                &q01_pair_e8m0,
                                reuse_q1_pair_scale,
                                &prefetched_q1,
                                &lagged_q1_raw_max);
                        if constexpr (
                            C::HAO_DIRECT_MX_LAGGED_Q1_SCALE) {
                            const float lagged_q1_group_max =
                                (lagged_q1_raw_max - shiftless_row_max) *
                                    scale_log2 +
                                0.5f * static_cast<float>(
                                    C::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2);
                            lagged_q1_e8m0 =
                                hao_direct_select_mxfp4_e8m0<PairScaleC>(
                                    lagged_q1_group_max);
                        }
                        if constexpr (
                            C::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD) {
                            fp4pv_store_mxfp4_scale_byte_swizzled(
                                p_scale_owned,
                                reader_warp * 32 + warp::laneid(),
                                1,
                                q1_mx_scale);
                        }
                        if constexpr (
                            C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM) {
                            hao_direct_store_mx_denom_payload_quarter_smem<1>(
                                mx_denom_payload_smem[Stage],
                                reader_warp,
                                mx_denom_payload_words);
                        }
                    }

                    if constexpr (
                        C::HAO_DIRECT_MX_DELAYED_EARLY_Q3 &&
                        !DualQ3CorrectionOffload) {
                        if constexpr (
                            C::HAO_DIRECT_MX_GLOBAL_ANCHOR128) {
                            if (n == 0) {
                                scratch = retained_anchor_q3;
                            } else {
                                upstream_load_score_quarter_x32<
                                    3, C, false>(
                                        scratch,
                                        score_row_base,
                                        n,
                                        query_tile);
                            }
                        } else {
                            upstream_load_score_quarter_x32<3, C, false>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                        }
                    }
                    if constexpr (C::HAO_DIRECT_MXFP4_PV) {
                        if constexpr (
                            C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF) {
                            hao_direct_store_mxfp4_p_scale_half_register<
                                0,
                                !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE>(
                                    score_owned.addr + 32,
                                    reader_warp,
                                    q0_mx_scale,
                                    q1_mx_scale);
                        } else if constexpr (
                            C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH) {
                            const uint8_t q0_scale =
                                hao_direct_load_mxfp4_scale_byte_linear(
                                    p_scale_owned,
                                    reader_warp * 32 + warp::laneid(),
                                    0);
                            hao_direct_store_mxfp4_p_scale_half_register<
                                0,
                                !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE>(
                                    score_owned.addr + 32,
                                    reader_warp,
                                    q0_scale,
                                    q1_mx_scale);
                        } else {
                            hao_direct_store_mxfp4_p_scale_half_direct<
                                0,
                                !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                                C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                                    score_owned.addr + 32,
                                    p_scale_owned,
                                    reader_warp);
                        }
                    } else if constexpr (
                        C::HAO_DIRECT_SPARSE_SPLIT_K128) {
                        hao_direct_store_sparse_nvfp4_p_scale_half_direct<0>(
                            score_owned.addr + 32,
                            p_scale_owned,
                            reader_warp);
                    } else if constexpr (
                        C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE &&
                        !C::HAO_DIRECT_SPARSE_PV) {
                        if constexpr (
                            C::HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF ||
                            GlobalAnchorDelayed) {
                            hao_direct_store_nvfp4_p_scale_half_register<0>(
                                score_owned.addr + 32,
                                reader_warp,
                                q0_scale_bits,
                                q1_scale_bits);
                        } else {
                            hao_direct_store_nvfp4_p_scale_half_direct<0>(
                                score_owned.addr + 32,
                                p_scale_owned,
                                reader_warp);
                        }
                    }
                    if constexpr (
                        C::HAO_DIRECT_MX_EARLY_Q2_REDUCE) {
                        quarter_max =
                            base_types::constants<float>::neg_infty();
                        if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                            upstream_reduce_score_quarter<C>(
                                delayed_q2, quarter_max);
                        }
                    }
                    if constexpr (
                        !C::HAO_DIRECT_EARLY_DIRECT_SCALE) {
                        fp4pv_publish_cta_shared_only();
                    } else if constexpr (
                        C::HAO_DIRECT_SPARSE_SPLIT_K128) {
                        fp4pv_publish_cta_shared_only();
                    }
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(p_first_o_rescaled[Stage]);
                    }
                    if constexpr (
                        C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF) {
                        hao_direct_store_nv_first_payload_quarter_smem<1>(
                            nv_first_payload_smem[Stage],
                            reader_warp,
                            retained_first_words[1]);
                    }
                    if constexpr (DualQ3TmemOffload) {
                        // Q2 reuses the first half of the temporary FP8
                        // checkpoint. Do not start its monolithic transform
                        // and store until correction has completed the x8 load.
                        hao_direct_fp4pv_wait(
                            dual_q3_checkpoint_empty[Stage],
                            q3_checkpoint_empty_phase);
                        q3_checkpoint_empty_phase ^= 1;
                    }

                    // The first PV K64 consumes Q0/Q1 while Q2 is built from
                    // retained registers and the independent Q3 load runs.
                    if constexpr (
                        !C::HAO_DIRECT_MX_DELAYED_EARLY_Q3 &&
                        !DualQ3CorrectionOffload) {
                        upstream_load_score_quarter_x32<3, C, false>(
                            scratch,
                            score_row_base,
                            n,
                            query_tile);
                    }
                    if constexpr (
                        !C::HAO_DIRECT_MX_EARLY_Q2_REDUCE) {
                        quarter_max =
                            base_types::constants<float>::neg_infty();
                        if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                            upstream_reduce_score_quarter<C>(
                                delayed_q2, quarter_max);
                        }
                    }
                    uint16_t q2_scale_bits = 0;
                    uint8_t q2_mx_scale = 0;
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD != 0
                    float hooked_q3_sum = 0.0f;
                    static_assert(C::HAO_DIRECT_NVFP4_PV);
                    auto q3_hook = [&]() {
                        tensor_load_wait();
                        float q3_max =
                            base_types::constants<float>::neg_infty();
                        upstream_reduce_score_quarter<C>(
                            scratch, q3_max);
                        hooked_q3_sum =
                            hao_direct_transform_pack_fp4_quarter<
                                3,
                                C,
                                (StageAffineMask & 0x8u) != 0>(
                                    p_owned,
                                    p_scale_owned,
                                    reader_warp,
                                    scratch,
                                    shiftless_row_max,
                                    scale_log2,
                                    q3_max);
                    };
                    tile_sum =
                        hao_direct_transform_pack_nvfp4_quarter<
                            2,
                            C,
                            (StageAffineMask & 0x4u) != 0,
                            true>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                delayed_q2,
                                shiftless_row_max,
                                scale_log2,
                                quarter_max,
                                nullptr,
                                nullptr,
                                nullptr,
                                q3_hook);
                    tile_sum += q0_sum;
                    tile_sum += q1_sum;
                    tile_sum += hooked_q3_sum;
#else
                    if constexpr (
                        C::HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF) {
                        tile_sum =
                            hao_direct_transform_pack_nvfp4_quarter_register_scale<
                                2,
                                C,
                                (StageAffineMask & 0x4u) != 0>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                delayed_q2,
                                shiftless_row_max,
                                scale_log2,
                                quarter_max,
                                q2_scale_bits);
                    } else {
                        tile_sum =
                            hao_direct_transform_pack_fp4_quarter<
                                2,
                                PairScaleC,
                                (StageAffineMask & 0x4u) != 0,
                                C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM,
                                C::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST,
                                (C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF ||
                                 C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH)>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                delayed_q2,
                                shiftless_row_max,
                                scale_log2,
                                quarter_max,
                                mx_denom_payload_words,
                                mx_anchor_quant_bias_base,
                                mx_anchor_affine_scale,
                                &q2_mx_scale,
                                &q23_pair_e8m0);
                        if constexpr (
                            C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH) {
                            hao_direct_store_mxfp4_scale_byte_linear(
                                p_scale_owned,
                                reader_warp * 32 + warp::laneid(),
                                2,
                                q2_mx_scale);
                        }
                        if constexpr (
                            C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM) {
                            hao_direct_store_mx_denom_payload_quarter_smem<2>(
                                mx_denom_payload_smem[Stage],
                                reader_warp,
                                mx_denom_payload_words);
                        }
                    }
                    tile_sum += q0_sum;
                    tile_sum += q1_sum;
#endif

                    if constexpr (DualQ3CorrectionOffload) {
                        // Q3 is built by the correction warpgroup. Publish
                        // Q2's payload and scale-byte lease so that group can
                        // assemble and release the tail K64 page.
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(dual_q2_ready[Stage]);
                        }
                    } else {
                        uint8_t q3_mx_scale = 0;
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD == 0
                        uint16_t q3_scale_bits = 0;
                        tensor_load_wait();
                        quarter_max =
                            base_types::constants<float>::neg_infty();
                        if constexpr (
                            C::HAO_DIRECT_P_QUARTER_MAX &&
                            !(PairScaleC::HAO_DIRECT_MX_PAIR_SCALE_REUSE & 2)) {
                            upstream_reduce_score_quarter<C>(
                                scratch, quarter_max);
                        }
                        if constexpr (
                            C::HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF) {
                            tile_sum +=
                                hao_direct_transform_pack_nvfp4_quarter_register_scale<
                                    3,
                                    C,
                                    (StageAffineMask & 0x8u) != 0>(
                                    p_owned,
                                    p_scale_owned,
                                    reader_warp,
                                    scratch,
                                    shiftless_row_max,
                                    scale_log2,
                                    quarter_max,
                                    q3_scale_bits);
                        } else {
                            tile_sum +=
                                hao_direct_transform_pack_fp4_quarter<
                                3,
                                PairScaleC,
                                (StageAffineMask & 0x8u) != 0,
                                C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM,
                                C::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST,
                                (C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF ||
                                 C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH)>(
                                    p_owned,
                                    p_scale_owned,
                                    reader_warp,
                                    scratch,
                                    shiftless_row_max,
                                    scale_log2,
                                    quarter_max,
                                    mx_denom_payload_words,
                                    mx_anchor_quant_bias_base,
                                    mx_anchor_affine_scale,
                                    &q3_mx_scale,
                                    &q23_pair_e8m0);
                            if constexpr (
                                C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM) {
                                hao_direct_store_mx_denom_payload_quarter_smem<3>(
                                    mx_denom_payload_smem[Stage],
                                    reader_warp,
                                    mx_denom_payload_words);
                            }
                        }
#else
                        uint16_t q3_scale_bits = 0;
#endif

                        if constexpr (C::HAO_DIRECT_MXFP4_PV) {
                            if constexpr (
                                C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF) {
                                hao_direct_store_mxfp4_p_scale_half_register<
                                    1,
                                    !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE>(
                                        score_owned.addr + 48,
                                        reader_warp,
                                        q2_mx_scale,
                                        q3_mx_scale);
                            } else if constexpr (
                                C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH) {
                                const uint8_t q2_scale =
                                    hao_direct_load_mxfp4_scale_byte_linear(
                                        p_scale_owned,
                                        reader_warp * 32 + warp::laneid(),
                                        2);
                                hao_direct_store_mxfp4_p_scale_half_register<
                                    1,
                                    !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE>(
                                        score_owned.addr + 48,
                                        reader_warp,
                                        q2_scale,
                                        q3_mx_scale);
                            } else {
                                hao_direct_store_mxfp4_p_scale_half_direct<
                                    1,
                                    !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                                    C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                                        score_owned.addr + 48,
                                        p_scale_owned,
                                        reader_warp);
                            }
                        } else if constexpr (
                            C::HAO_DIRECT_SPARSE_PV) {
                            if constexpr (
                                C::HAO_DIRECT_SPARSE_SPLIT_K128) {
                                hao_direct_store_sparse_nvfp4_p_scale_half_direct<1>(
                                    score_owned.addr + 36,
                                    p_scale_owned,
                                    reader_warp);
                            } else {
                                hao_direct_store_sparse_nvfp4_p_scale_direct(
                                    score_owned.addr + 32,
                                    p_scale_owned,
                                    reader_warp);
                            }
                        } else if constexpr (
                            C::HAO_DIRECT_NV_EARLY_DIRECT_SCALE) {
                            if constexpr (
                                C::HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF) {
                                hao_direct_store_nvfp4_p_scale_half_register<1>(
                                    score_owned.addr + 36,
                                    reader_warp,
                                    q2_scale_bits,
                                    q3_scale_bits);
                            } else {
                                hao_direct_store_nvfp4_p_scale_half_direct<1>(
                                    score_owned.addr + 36,
                                    p_scale_owned,
                                    reader_warp);
                            }
                        }
                        if constexpr (
                            !C::HAO_DIRECT_EARLY_DIRECT_SCALE) {
                            fp4pv_publish_cta_shared_only();
                        }
                        if constexpr (C::HAO_DIRECT_SPARSE_PV) {
                            fp4pv_shared_async_proxy_fence_cta();
                        }
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(p_tail[Stage]);
                        }
                    }
                } else {
                    score_quarter_regs scratch;
                    score_quarter_regs paired_tail;
                    score_quarter_regs prefetched_q0;
                    float quarter_max =
                        base_types::constants<float>::neg_infty();
                    upstream_load_score_quarter_x32<2, C>(
                        scratch, score_row_base, n, query_tile);
                    if constexpr (
                        C::HAO_DIRECT_MX_HALF_PREFETCH_Q0) {
                        upstream_load_score_quarter_half_x16<
                            0, 0, C, false>(
                                prefetched_q0,
                                score_row_base,
                                n,
                                query_tile);
                    }
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max);
                    tile_sum =
                        hao_direct_transform_pack_fp4_quarter<2, C>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            0.0f,
                            scale_log2,
                            quarter_max);

                    quarter_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (
                        C::HAO_DIRECT_MX_HALF_PREFETCH_Q0) {
                        upstream_load_score_quarter_half_x16<
                            0, 1, C, false>(
                                prefetched_q0,
                                score_row_base,
                                n,
                                query_tile);
                        tensor_load_wait();
                        if constexpr (
                            C::HAO_DIRECT_MX_HALF_PREFETCH_Q3) {
                            upstream_load_score_quarter_half_x16<
                                3, 0, C, false>(
                                    scratch,
                                    score_row_base,
                                    n,
                                    query_tile);
                        }
                        upstream_reduce_score_quarter<C>(
                            prefetched_q0, quarter_max);
                        tile_sum +=
                            hao_direct_transform_pack_fp4_quarter<0, C>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                prefetched_q0,
                                0.0f,
                                scale_log2,
                                quarter_max);
                    } else {
                        upstream_load_score_quarter_x32<0, C>(
                            scratch, score_row_base, n, query_tile);
                        upstream_reduce_score_quarter<C>(
                            scratch, quarter_max);
                        tile_sum +=
                            hao_direct_transform_pack_fp4_quarter<0, C>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                0.0f,
                                scale_log2,
                                quarter_max);
                    }
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        arrive(sfqk_load[OppositeScaleStage]);
                    }

                    quarter_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (Q3CorrectionOffload) {
                        upstream_load_score_quarter_x32<1, C>(
                            paired_tail,
                            score_row_base,
                            n,
                            query_tile);
                    } else if constexpr (
                        C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                        if constexpr (
                            C::HAO_DIRECT_MX_HALF_PREFETCH_Q3) {
                            upstream_load_score_quarter_half_x16<
                                3, 1, C, false>(
                                    scratch,
                                    score_row_base,
                                    n,
                                    query_tile);
                        } else {
                            upstream_load_score_quarter_x32<3, C, false>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                        }
                        if constexpr (
                            C::HAO_DIRECT_MX_HALF_PREFETCH_Q1) {
                            upstream_load_score_quarter_half_x16<
                                1, 0, C, false>(
                                    paired_tail,
                                    score_row_base,
                                    n,
                                    query_tile);
                        } else {
                            upstream_load_score_quarter_x32<1, C, false>(
                                paired_tail,
                                score_row_base,
                                n,
                                query_tile);
                        }
                        tensor_load_wait();
                    } else if constexpr (C::HAO_DIRECT_EARLY_P) {
                        upstream_load_score_quarter_x32<1, C>(
                            paired_tail,
                            score_row_base,
                            n,
                            query_tile);
                    } else {
                        upstream_load_score_quarter_x32<3, C>(
                            scratch, score_row_base, n, query_tile);
                    }

                    if constexpr (C::HAO_DIRECT_EARLY_P) {
                        upstream_reduce_score_quarter<C>(
                            paired_tail, quarter_max);
                        tile_sum +=
                            hao_direct_transform_pack_fp4_quarter<1, C>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                paired_tail,
                                0.0f,
                                scale_log2,
                                quarter_max);
                        if constexpr (
                            C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                            hao_direct_store_mxfp4_p_scale_half_direct<
                                0,
                                !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                                C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                                    score_owned.addr + 32,
                                    p_scale_owned,
                                    reader_warp);
                        }
                        // Q0/Q1 form the first scaled K64. Publish them while
                        // Q3 remains resident so the issuer can start PV before
                        // this warp constructs the Q2/Q3 tail scale page.
                        if constexpr (
                            !C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                            fp4pv_publish_cta_shared_only();
                        }
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            arrive(p_first_o_rescaled[Stage]);
                        }

                        if constexpr (!Q3CorrectionOffload) {
                            quarter_max =
                                base_types::constants<float>::neg_infty();
                            if constexpr (
                                !C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                                upstream_load_score_quarter_x32<3, C>(
                                    scratch,
                                    score_row_base,
                                    n,
                                    query_tile);
                            }
                            upstream_reduce_score_quarter<C>(
                                scratch, quarter_max);
                            tile_sum +=
                                hao_direct_transform_pack_fp4_quarter<3, C>(
                                    p_owned,
                                    p_scale_owned,
                                    reader_warp,
                                    scratch,
                                    0.0f,
                                    scale_log2,
                                    quarter_max);
                        }
                    } else {
                        upstream_reduce_score_quarter<C>(
                            scratch, quarter_max);
                        tile_sum +=
                            hao_direct_transform_pack_fp4_quarter<3, C>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                0.0f,
                                scale_log2,
                                quarter_max);

                        quarter_max =
                            base_types::constants<float>::neg_infty();
                        if constexpr (
                            C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                            if constexpr (
                                C::HAO_DIRECT_MX_HALF_PREFETCH_Q1) {
                                upstream_load_score_quarter_half_x16<
                                    1, 1, C, false>(
                                        paired_tail,
                                        score_row_base,
                                        n,
                                        query_tile);
                                tensor_load_wait();
                            }
                            upstream_reduce_score_quarter<C>(
                                paired_tail, quarter_max);
                            tile_sum +=
                                hao_direct_transform_pack_fp4_quarter<1, C>(
                                    p_owned,
                                    p_scale_owned,
                                    reader_warp,
                                    paired_tail,
                                    0.0f,
                                    scale_log2,
                                    quarter_max);
                        } else {
                            upstream_load_score_quarter_x32<1, C>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                            upstream_reduce_score_quarter<C>(
                                scratch, quarter_max);
                            tile_sum +=
                                hao_direct_transform_pack_fp4_quarter<1, C>(
                                    p_owned,
                                    p_scale_owned,
                                    reader_warp,
                                    scratch,
                                    0.0f,
                                    scale_log2,
                                    quarter_max);
                        }
                    }
                    if constexpr (!Q3CorrectionOffload) {
                        if constexpr (
                            C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                            hao_direct_store_mxfp4_p_scale_half_direct<
                                1,
                                !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                                C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                                    score_owned.addr + 48,
                                    p_scale_owned,
                                    reader_warp);
                        }
                        if constexpr (
                            !C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                            fp4pv_publish_cta_shared_only();
                        }
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            if constexpr (!C::HAO_DIRECT_EARLY_P) {
                                arrive(p_first_o_rescaled[Stage]);
                            }
                            arrive(p_tail[Stage]);
                        }
                    }
                }

                if constexpr (
                    !C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS) {
                    hao_direct_fp4pv_wait(
                        softmax_corr_empty[Stage],
                        corr_producer_phase);
                    corr_producer_phase ^= 1;
                }
                if constexpr (
                    !C::HAO_DIRECT_DENOM_CORR_WG) {
                    row_sum += tile_sum;
                }
                if constexpr (
                    (C::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK &
                     (1 << Stage)) != 0) {
                    softmax_v_item += 2;
                }
                score_phase ^= 1;
                continue;
            }

            if constexpr (
                C::HAO_DIRECT_MX_LOCAL_NATIVE_DEFERRED_SCALE) {
                static_assert(C::HAO_DIRECT_MXFP4_PV);
                static_assert(C::HAO_DIRECT_MX_LOG2_QUANT);
                static_assert(!C::HAO_DIRECT_EARLY_P);

                score_quarter_regs scratch;
                score_quarter_regs paired_tail;
                float quarter_max0 =
                    base_types::constants<float>::neg_infty();
                float quarter_max1 =
                    base_types::constants<float>::neg_infty();
                float quarter_max2 =
                    base_types::constants<float>::neg_infty();
                float quarter_max3 =
                    base_types::constants<float>::neg_infty();
                int absolute_e8m0_0;
                int absolute_e8m0_1;
                int absolute_e8m0_2;
                int absolute_e8m0_3;

                // The packed-P overlay starts in score quarter 2. Consume
                // each quarter around its own maximum, write the payload
                // immediately, and retain only its maximum and quantized sum.
                upstream_load_score_quarter_x32<2, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max2);
                const float local_sum2 =
                    hao_direct_store_mxfp4_local_native_quarter<
                        2, C>(
                            p_owned,
                            reader_warp,
                            scratch,
                            scale_log2,
                            quarter_max2,
                            absolute_e8m0_2);

                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max0);
                const float local_sum0 =
                    hao_direct_store_mxfp4_local_native_quarter<
                        0, C>(
                            p_owned,
                            reader_warp,
                            scratch,
                            scale_log2,
                            quarter_max0,
                            absolute_e8m0_0);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }

                if constexpr (C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                    upstream_load_score_quarter_x32<3, C, false>(
                        scratch, score_row_base, n, query_tile);
                    upstream_load_score_quarter_x32<1, C, false>(
                        paired_tail, score_row_base, n, query_tile);
                    tensor_load_wait();
                } else {
                    upstream_load_score_quarter_x32<3, C>(
                        scratch, score_row_base, n, query_tile);
                }
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max3);
                const float local_sum3 =
                    hao_direct_store_mxfp4_local_native_quarter<
                        3, C>(
                            p_owned,
                            reader_warp,
                            scratch,
                            scale_log2,
                            quarter_max3,
                            absolute_e8m0_3);

                float local_sum1;
                if constexpr (C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                    upstream_reduce_score_quarter<C>(
                        paired_tail, quarter_max1);
                    local_sum1 =
                        hao_direct_store_mxfp4_local_native_quarter<
                            1, C>(
                                p_owned,
                                reader_warp,
                                paired_tail,
                                scale_log2,
                                quarter_max1,
                                absolute_e8m0_1);
                } else {
                    upstream_load_score_quarter_x32<1, C>(
                        scratch, score_row_base, n, query_tile);
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max1);
                    local_sum1 =
                        hao_direct_store_mxfp4_local_native_quarter<
                            1, C>(
                                p_owned,
                                reader_warp,
                                scratch,
                                scale_log2,
                                quarter_max1,
                                absolute_e8m0_1);
                }

                const float tile_max = fmaxf(
                    fmaxf(quarter_max0, quarter_max1),
                    fmaxf(quarter_max2, quarter_max3));
                const int tile_anchor_log2 =
                    __float2int_ru(tile_max * scale_log2);
                const int old_anchor_log2 = row_anchor_log2;
                if (n == 0) {
                    row_anchor_log2 = tile_anchor_log2;
                } else {
                    asm(
                        "max.s32 %0, %1, %2;"
                        : "=r"(row_anchor_log2)
                        : "r"(old_anchor_log2),
                          "r"(tile_anchor_log2));
                }
                const float correction =
                    n == 0
                        ? 1.0f
                        : hao_direct_pow2_integer_scalar(
                              old_anchor_log2 -
                              row_anchor_log2);
                stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    correction);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                float tile_sum =
                    hao_direct_store_mxfp4_scale_and_local_sum<
                        2, C>(
                            p_scale_owned,
                            reader_warp,
                            absolute_e8m0_2,
                            row_anchor_log2,
                            local_sum2);
                tile_sum +=
                    hao_direct_store_mxfp4_scale_and_local_sum<
                        0, C>(
                            p_scale_owned,
                            reader_warp,
                            absolute_e8m0_0,
                            row_anchor_log2,
                            local_sum0);
                tile_sum +=
                    hao_direct_store_mxfp4_scale_and_local_sum<
                        3, C>(
                            p_scale_owned,
                            reader_warp,
                            absolute_e8m0_3,
                            row_anchor_log2,
                            local_sum3);
                tile_sum +=
                    hao_direct_store_mxfp4_scale_and_local_sum<
                        1, C>(
                            p_scale_owned,
                            reader_warp,
                            absolute_e8m0_1,
                            row_anchor_log2,
                            local_sum1);

                fp4pv_publish_cta_shared_only();
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                    arrive(p_tail[Stage]);
                }

                hao_direct_fp4pv_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum = row_sum * correction + tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_MX_SCAN_FUSED_P) {
                static_assert(C::HAO_DIRECT_MXFP4_PV);
                static_assert(C::HAO_DIRECT_MX_DIRECT_LOG_CODE);
                static_assert(!C::HAO_DIRECT_EARLY_P);

                score_quarter_regs scratch;
                float quarter_max0 =
                    base_types::constants<float>::neg_infty();
                float quarter_max1 =
                    base_types::constants<float>::neg_infty();
                float quarter_max2 =
                    base_types::constants<float>::neg_infty();
                float quarter_max3 =
                    base_types::constants<float>::neg_infty();

                // P occupies the first 16 columns of score quarter 2. Consume
                // and overwrite that quarter first; the other score quarters
                // remain readable while their local FP4 payloads are emitted.
                upstream_load_score_quarter_x32<2, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max2);
                const int qsum2 =
                    hao_direct_store_mxfp4_local_log_quarter<2, C>(
                        p_owned,
                        reader_warp,
                        scratch,
                        scale_log2,
                        quarter_max2);

                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max0);
                const int qsum0 =
                    hao_direct_store_mxfp4_local_log_quarter<0, C>(
                        p_owned,
                        reader_warp,
                        scratch,
                        scale_log2,
                        quarter_max0);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }

                upstream_load_score_quarter_x32<3, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max3);
                const int qsum3 =
                    hao_direct_store_mxfp4_local_log_quarter<3, C>(
                        p_owned,
                        reader_warp,
                        scratch,
                        scale_log2,
                        quarter_max3);

                upstream_load_score_quarter_x32<1, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max1);
                const int qsum1 =
                    hao_direct_store_mxfp4_local_log_quarter<1, C>(
                        p_owned,
                        reader_warp,
                        scratch,
                        scale_log2,
                        quarter_max1);

                const float tile_max = fmaxf(
                    fmaxf(quarter_max0, quarter_max1),
                    fmaxf(quarter_max2, quarter_max3));
                const float old_max = row_max;
                row_max = fmaxf(row_max, tile_max);
                const float correction =
                    n == 0
                        ? 1.0f
                        : softmax_rescale_factor_strict(
                              row_max, old_max, scale_log2);
                stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    correction);
                __threadfence_block();
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                float tile_sum =
                    hao_direct_store_mxfp4_scale_and_quantized_sum<
                        2, C>(
                            p_scale_owned,
                            reader_warp,
                            quarter_max2,
                            row_max,
                            scale_log2,
                            qsum2);
                tile_sum +=
                    hao_direct_store_mxfp4_scale_and_quantized_sum<
                        0, C>(
                            p_scale_owned,
                            reader_warp,
                            quarter_max0,
                            row_max,
                            scale_log2,
                            qsum0);
                tile_sum +=
                    hao_direct_store_mxfp4_scale_and_quantized_sum<
                        3, C>(
                            p_scale_owned,
                            reader_warp,
                            quarter_max3,
                            row_max,
                            scale_log2,
                            qsum3);
                tile_sum +=
                    hao_direct_store_mxfp4_scale_and_quantized_sum<
                        1, C>(
                            p_scale_owned,
                            reader_warp,
                            quarter_max1,
                            row_max,
                            scale_log2,
                            qsum1);

                fp4pv_publish_cta_shared_only();
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                    arrive(p_tail[Stage]);
                }

                hao_direct_fp4pv_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum = row_sum * correction + tile_sum;
                score_phase ^= 1;
                continue;
            }

            // Role-local allocation supports HAO's R192 score-reader budget,
            // but retaining the full row serializes this nvcc quantizer. Use
            // one fragment and reacquire quarters instead. Quarter 2 is
            // consumed first because P starts 64 columns into the score bank.
            score_quarter_regs scratch;
            score_quarter_regs retained_q0;
            score_quarter_regs paired_scan;
            float tile_max =
                base_types::constants<float>::neg_infty();
            float quarter_max0 =
                base_types::constants<float>::neg_infty();
            float quarter_max1 =
                base_types::constants<float>::neg_infty();
            float quarter_max2 =
                base_types::constants<float>::neg_infty();
            float quarter_max3 =
                base_types::constants<float>::neg_infty();
            if constexpr (C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                upstream_load_score_quarter_x32<0, C, false>(
                    scratch, score_row_base, n, query_tile);
                upstream_load_score_quarter_x32<1, C, false>(
                    paired_scan, score_row_base, n, query_tile);
                tensor_load_wait();
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max0);
                tile_max = fmaxf(tile_max, quarter_max0);
                upstream_reduce_score_quarter<C>(
                    paired_scan, quarter_max1);
                tile_max = fmaxf(tile_max, quarter_max1);
                upstream_load_score_quarter_x32<3, C, false>(
                    paired_scan, score_row_base, n, query_tile);
                upstream_load_score_quarter_x32<2, C, false>(
                    scratch, score_row_base, n, query_tile);
                tensor_load_wait();
                upstream_reduce_score_quarter<C>(
                    paired_scan, quarter_max3);
                tile_max = fmaxf(tile_max, quarter_max3);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max2);
                tile_max = fmaxf(tile_max, quarter_max2);
            } else if constexpr (!C::HAO_DIRECT_MX_RETAIN_Q0) {
                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
                if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max0);
                    tile_max = fmaxf(tile_max, quarter_max0);
                } else {
                    upstream_reduce_score_quarter<C>(
                        scratch, tile_max);
                }
            }
            if constexpr (!C::HAO_DIRECT_MX_PAIR_LOAD_SCAN) {
                upstream_load_score_quarter_x32<1, C>(
                    scratch, score_row_base, n, query_tile);
                if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max1);
                    tile_max = fmaxf(tile_max, quarter_max1);
                } else {
                    upstream_reduce_score_quarter<C>(
                        scratch, tile_max);
                }
                upstream_load_score_quarter_x32<3, C>(
                    scratch, score_row_base, n, query_tile);
                if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max3);
                    tile_max = fmaxf(tile_max, quarter_max3);
                } else {
                    upstream_reduce_score_quarter<C>(
                        scratch, tile_max);
                }
                if constexpr (C::HAO_DIRECT_MX_RETAIN_Q0) {
                    upstream_load_score_quarter_x32<0, C>(
                        retained_q0, score_row_base, n, query_tile);
                    upstream_reduce_score_quarter<C>(
                        retained_q0, quarter_max0);
                    tile_max = fmaxf(tile_max, quarter_max0);
                }
                // Scan Q2 last so the quarter that packed P overwrites first
                // is still resident when the row maximum becomes known.
                upstream_load_score_quarter_x32<2, C>(
                    scratch, score_row_base, n, query_tile);
                if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max2);
                    tile_max = fmaxf(tile_max, quarter_max2);
                } else {
                    upstream_reduce_score_quarter<C>(
                        scratch, tile_max);
                }
            }

            const float old_max = row_max;
            const float candidate_max = fmaxf(row_max, tile_max);
            row_max = candidate_max;
            const float correction =
                n == 0
                    ? 1.0f
                    : softmax_rescale_factor_strict(
                          row_max, old_max, scale_log2);
            stats_store(
                correction_owned,
                warpgroup::laneid(),
                correction);
            __threadfence_block();
            if (warp::elect_leader()) {
                arrive(softmax_corr_full[Stage]);
            }

            float tile_sum =
                hao_direct_transform_pack_fp4_quarter<2, C>(
                p_owned,
                p_scale_owned,
                reader_warp,
                scratch,
                row_max,
                    scale_log2,
                    quarter_max2);
            uint32_t early_denom_words[4];
            if constexpr (!C::HAO_DIRECT_MX_RETAIN_Q0) {
                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
            }
            // The opposite stage's Q/K scales overwrite the low 32 columns
            // of this retired score bank. Publish that lease only after q0
            // is resident in registers; q1/q3 and the packed-P destinations
            // are disjoint from the scale pages.
            tensor_before_thread_sync();
            if (warp::elect_leader()) {
                arrive(sfqk_load[OppositeScaleStage]);
            }
            if constexpr (C::HAO_DIRECT_MX_RETAIN_Q0) {
                if constexpr (C::HAO_DIRECT_MX_DENOM_EARLY_SMEM) {
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            0, C, false, true>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                retained_q0,
                                row_max,
                                scale_log2,
                                quarter_max0,
                                early_denom_words);
                    hao_direct_store_nv_first_payload_quarter_smem<0>(
                        nv_first_payload_smem[Stage],
                        reader_warp,
                        early_denom_words);
                } else {
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<0, C>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            retained_q0,
                            row_max,
                            scale_log2,
                            quarter_max0);
                }
            } else {
                if constexpr (C::HAO_DIRECT_MX_DENOM_EARLY_SMEM) {
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            0, C, false, true>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                row_max,
                                scale_log2,
                                quarter_max0,
                                early_denom_words);
                    hao_direct_store_nv_first_payload_quarter_smem<0>(
                        nv_first_payload_smem[Stage],
                        reader_warp,
                        early_denom_words);
                } else {
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<0, C>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max0);
                }
            }
            if constexpr (C::HAO_DIRECT_EARLY_P) {
                upstream_load_score_quarter_x32<1, C>(
                    scratch, score_row_base, n, query_tile);
                if constexpr (C::HAO_DIRECT_MX_DENOM_EARLY_SMEM) {
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            1, C, false, true>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                row_max,
                                scale_log2,
                                quarter_max1,
                                early_denom_words);
                    hao_direct_store_nv_first_payload_quarter_smem<1>(
                        nv_first_payload_smem[Stage],
                        reader_warp,
                        early_denom_words);
                } else {
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<1, C>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            row_max,
                            scale_log2,
                            quarter_max1);
                }
                if constexpr (
                    C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                    hao_direct_store_mxfp4_p_scale_half_direct<0>(
                        score_owned.addr + 32,
                        p_scale_owned,
                        reader_warp);
                }
                // Q0/Q1 and their K64 scale page are complete. The tensor
                // issuer can copy that page and consume the first P half
                // while this warp constructs Q3; Q2 was retired first.
                fp4pv_publish_cta_shared_only();
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                }
            }

            upstream_load_score_quarter_x32<3, C>(
                scratch, score_row_base, n, query_tile);
            tile_sum += hao_direct_transform_pack_fp4_quarter<3, C>(
                p_owned,
                p_scale_owned,
                reader_warp,
                scratch,
                row_max,
                scale_log2,
                quarter_max3);
            if constexpr (
                C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                hao_direct_store_mxfp4_p_scale_half_direct<1>(
                    score_owned.addr + 48,
                    p_scale_owned,
                    reader_warp);
            }
            if constexpr (!C::HAO_DIRECT_EARLY_P) {
                upstream_load_score_quarter_x32<1, C>(
                    scratch, score_row_base, n, query_tile);
                tile_sum +=
                    hao_direct_transform_pack_fp4_quarter<1, C>(
                        p_owned,
                        p_scale_owned,
                        reader_warp,
                        scratch,
                        row_max,
                        scale_log2,
                        quarter_max1);
            }

            fp4pv_publish_cta_shared_only();
            tensor_before_thread_sync();
            if (warp::elect_leader()) {
                if constexpr (!C::HAO_DIRECT_EARLY_P) {
                    arrive(p_first_o_rescaled[Stage]);
                }
                arrive(p_tail[Stage]);
            }

            hao_direct_fp4pv_wait(
                softmax_corr_empty[Stage],
                corr_producer_phase);
            corr_producer_phase ^= 1;
            row_sum = row_sum * correction + tile_sum;
            score_phase ^= 1;
        }

        if constexpr (
            C::HAO_DIRECT_MX_LOCAL_NATIVE_DEFERRED_SCALE) {
            row_max =
                static_cast<float>(row_anchor_log2) / scale_log2;
        }
        if constexpr (C::HAO_DIRECT_DENOM_CORR_WG) {
            hao_direct_fp4pv_wait(
                nv_denom_done[Stage],
                nv_denom_done_phase);
            nv_denom_done_phase ^= 1;
        }
        if constexpr (Q01CorrectionOffload) {
            hao_direct_fp4pv_wait(
                q3_sum_full,
                q3_sum_consumer_phase);
            row_sum += stats_load(
                correction_owned,
                warpgroup::laneid());
            if (warp::elect_leader()) {
                arrive(q3_sum_empty);
            }
            q3_sum_consumer_phase ^= 1;
        } else if constexpr (Q3CorrectionOffload) {
            hao_direct_fp4pv_wait(
                q3_sum_full,
                q3_sum_consumer_phase);
            row_sum += stats_load(
                correction_owned,
                warpgroup::laneid());
            if (warp::elect_leader()) {
                arrive(q3_sum_empty);
            }
            q3_sum_consumer_phase ^= 1;
        } else if constexpr (DualQ3CorrectionOffload) {
            hao_direct_fp4pv_wait(
                dual_q3_sum_full[Stage],
                q3_sum_consumer_phase);
            row_sum += stats_load(
                row_sum_owned,
                warpgroup::laneid());
            if (warp::elect_leader()) {
                arrive(dual_q3_sum_empty[Stage]);
            }
            q3_sum_consumer_phase ^= 1;
        }
        if constexpr (C::HAO_DIRECT_MX_DEFER_JOB_STATS_WAIT) {
            hao_direct_fp4pv_wait(
                softmax_corr_empty[Stage], corr_producer_phase);
            corr_producer_phase ^= 1;
        }
        stats_store(
            row_max_owned, warpgroup::laneid(), row_max);
        if constexpr (!C::HAO_DIRECT_DENOM_CORR_WG) {
            stats_store(
                row_sum_owned, warpgroup::laneid(), row_sum);
        }
        __threadfence_block();
        if (warp::elect_leader()) {
            arrive(softmax_corr_full[Stage]);
        }
    }

    if (warp::elect_leader()) {
        arrive(tmem_dealloc);
    }
}

{
    constexpr int Stage = HAO_DIRECT_SOFTMAX_STAGE;
    static_assert(Stage == 0 || Stage == 1);
    constexpr bool DualLse = C::HAO_DIRECT_MX_DUAL_LSE_DENOM;
    constexpr bool StableLseLogsum =
        C::HAO_DIRECT_MX_STABLE_LSE_LOGSUM;
    constexpr bool AlternateLseStat =
        C::HAO_DIRECT_MX_ALTERNATE_LSE_STAT;
    using PairScaleC = hao_direct_pair_scale_stage_config<C, Stage>;
    constexpr bool InterleaveQ1Max =
        (C::HAO_DIRECT_MX_Q1_MAX_INTERLEAVE && Stage == 1) ||
        (C::HAO_DIRECT_MX_STAGE0_Q1_MAX_INTERLEAVE && Stage == 0);
    constexpr bool SelfInterleaveQ1Max =
        !InterleaveQ1Max &&
        C::HAO_DIRECT_MX_Q1_SELF_MAX &&
        (PairScaleC::HAO_DIRECT_MX_PAIR_SCALE_REUSE & 1) != 0;
    constexpr bool InterleaveQ2Max =
        C::HAO_DIRECT_MX_Q2_MAX_INTERLEAVE;
    constexpr bool InterleaveQ3Max =
        C::HAO_DIRECT_MX_Q3_MAX_INTERLEAVE;
    constexpr int OppositeScaleStage =
        C::QUERY_STAGES == 1 ? Stage : 1 - Stage;
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
    constexpr int DeferDenomFinalizeMode =
        !DualQ3CorrectionOffload
            ? C::HAO_DIRECT_MX_DEFER_DENOM_FINALIZE
            : 0;
    constexpr bool DeferDenomFinalize =
        DeferDenomFinalizeMode != 0;
    static_assert(
        !StableLseLogsum || DeferDenomFinalizeMode == 2,
        "stable MX saved-LSE logsum requires all four denominator carriers");
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
    constexpr bool Sm103LdRedEligible =
        C::HAO_DIRECT_SM103_LDRED &&
        C::HAO_DIRECT_NONCAUSAL &&
        !C::HAO_DIRECT_GLOBAL_ANCHOR &&
        !C::HAO_DIRECT_MX_GLOBAL_ANCHOR128 &&
        !DualQ3CorrectionOffload &&
        !InterleaveQ1Max;
    // Self-owned Q1 consumes the ordinary Q1 load, but the other three
    // quarters remain independent fused-reduction candidates.
    constexpr uint32_t Sm103LdRedOwnershipMask =
        SelfInterleaveQ1Max ? 0xdu : 0xfu;
    constexpr uint32_t Sm103LdRedStageMask =
        Stage == 0
            ? C::HAO_DIRECT_SM103_LDRED_STAGE0_MASK
            : C::HAO_DIRECT_SM103_LDRED_STAGE1_MASK;
    constexpr uint32_t Sm103LdRedMask =
        Sm103LdRedEligible
            ? Sm103LdRedStageMask & Sm103LdRedOwnershipMask
            : 0u;
    constexpr bool Sm103LdRedQ0 = (Sm103LdRedMask & 0x1u) != 0;
    constexpr bool Sm103LdRedQ1 = (Sm103LdRedMask & 0x2u) != 0;
    constexpr bool Sm103LdRedQ2 = (Sm103LdRedMask & 0x4u) != 0;
    constexpr bool Sm103LdRedQ3 =
        C::Nb >= 128 && (Sm103LdRedMask & 0x8u) != 0;
    constexpr bool Sm103LdRedQ12LoadPath =
        Sm103LdRedQ1 || Sm103LdRedQ2;
    const int reader_warp = warpgroup::warpid();
    const Score score_owned =
        tm.template allocate<Score>(Stage == 0 ? S0 : S1);
    const P p_stage_owned = Stage == 0 ? p0 : p1;
    auto &p_scale_owned = p_sc_smem[Stage];
    const uint32_t scale_base_owned =
        C::HAO_DIRECT_D64_COMPACT_P
            ? static_cast<uint32_t>(
                  Stage == 0
                      ? D64_P_SCALE0 - 32
                      : D64_P_SCALE1 - 32)
            : (C::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY ||
               C::HAO_DIRECT_D64_DETACHED_P)
            ? static_cast<uint32_t>(
                  Stage == 0
                      ? DOUBLE_SCORE_SCALE0
                      : DOUBLE_SCORE_SCALE1)
            : score_owned.addr;
    const uint32_t p_scale_store_base_owned =
        C::HAO_DIRECT_D64_DETACHED_P
            ? scale_base_owned
            : score_owned.addr;

    int score_phase = 0;
    int corr_producer_phase = 1;
    int q3_sum_consumer_phase = 0;
    int q3_checkpoint_empty_phase = 0;
    int global_anchor_consumer_phase = 0;
    int nv_denom_done_phase = 0;
    int softmax_v_item = 1;
    int n256_prefix_phase = 0;
    bool d64_p_first_reuse_started = false;
    bool d64_p_tail_reuse_started = false;
    int d64_p_first_reuse_phase = 0;
    int d64_p_tail_reuse_phase = 0;
    int d64_kv_job_base = 0;
    int d64_previous_p_v_item = -1;
    float2 n256_output_prefix_regs[8];

    // Stage-0 QK scales live in S1.  Before S1 exists, its reader supplies
    // the initial lease exactly as the HAO softmax-1 role does.
    if constexpr (
        Stage == 1 &&
        !C::HAO_DIRECT_N96_DUAL_QUERY &&
        !C::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY) {
        if (hao_direct_publication_leader<C>()) {
            hao_direct_arrive_rank0<C>(sfqk_load[0]);
        }
    } else if constexpr (
        C::HAO_DIRECT_SM103_GROUP2 &&
        C::QUERY_STAGES == 1) {
        // Both rank-local softmax warpgroups contribute the initial score
        // lease to the rank-0 collective issuer.
        if (hao_direct_publication_leader<C>()) {
            hao_direct_arrive_rank0<C>(sfqk_load[0]);
        }
    }

    for (int logical_job = hao_direct_logical_job_start<C>();
         logical_job < C::LOGICAL_JOBS;
         logical_job += hao_direct_logical_job_stride<C>()) {
        const hao_direct_task task =
            hao_direct_decode_task<C>(logical_job);
        const int query_tile =
            hao_direct_query_tile<C>(task, Stage);
        const int key_tiles =
            hao_direct_key_tiles<C>(task);
        const bool direct_stage0_tail_handoff =
            C::N_TILES > 2 &&
            (C::Dqk == 64 || key_tiles > 2);
        const float scale_log2 =
            qk_scale_log2(
                g, task.batch, task.head, task.kv_head);
        auto &row_max_owned = row_max_smem[Stage];
        auto &row_sum_owned = row_sum_smem[Stage];
        auto &correction_owned = correction_smem[Stage];
        // Rows 0..31 have their complete causal prefix in Q0.  The deployed
        // path stores P scales HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2 exponent
        // steps lower and restores that fixed shift in saved LSE.  If the raw
        // E8M0 code cannot represent the full stored shift, the stored code
        // saturates at one and the fixed LSE correction is no longer valid.
        // Rebase only those initial causal rows to their exact maximum so the
        // published scale and LSE retain the same shift contract.
        const bool stabilize_initial_causal_quarter =
            C::HAO_DIRECT_MXFP4_PV &&
            C::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
            C::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2 == 0 &&
            !C::HAO_DIRECT_GLOBAL_ANCHOR &&
            C::HAO_DIRECT_CAUSAL &&
            query_tile == 0 && reader_warp == 0;
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
        float lse_row_sum = 0.0f;
        int lse_max_e8m0 = 0;
        float lse_scaled_quantized_sum_x2 = 0.0f;
        int row_anchor_log2 = 0;
        float mx_anchor_quant_bias_base = 0.0f;
        float mx_anchor_affine_scale = 0.0f;
        uint8_t lagged_q1_e8m0 = 0;

        #pragma unroll 1
        for (int n =
                 C::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY ? Stage : 0;
             n < key_tiles;
             n += C::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY ? 2 : 1) {
            if constexpr (
                C::HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP &&
                Stage == 0) {
                // No stage-0 score generation exists for this fully masked
                // tile, so leave score_phase unchanged for the next job.
                if (n + 1 == key_tiles &&
                    direct_stage0_tail_handoff) {
                    continue;
                }
            }
            P p_owned = p_stage_owned;
            if constexpr (
                C::HAO_DIRECT_D64_TRIPLE_P &&
                C::HAO_DIRECT_D64_TRIPLE_P_STAGE == Stage) {
                if ((n & 1) != 0) {
                    p_owned = d64_p2;
                }
            }
            if constexpr (
                C::HAO_DIRECT_SM103_N256_DUAL_QUERY &&
                C::Nb == 256 &&
                (Stage == 1 ||
                 C::HAO_DIRECT_SM103_N256_SPLIT_OUTPUT_SCALES)) {
                if (n != 0) {
                    hao_direct_fp4pv_wait(
                        n256_prefix_save_ready[Stage],
                        C::HAO_DIRECT_SM103_N256_DIRECT_PV_HANDOFF
                            ? ((n - 1) & 1)
                            : n256_prefix_phase);
                    tensor_after_thread_sync();
                    if constexpr (
                        C::HAO_DIRECT_SM103_N256_PREFIX_REGS) {
                        hao_direct_load_tmem_x16_to_regs(
                            Stage == 0 ? output0.addr : output1.addr,
                            n256_output_prefix_regs);
                        tensor_after_thread_sync();
                        warpgroup::sync(1);
                    } else {
                        hao_direct_save_tmem_x16_to_smem(
                            Stage == 0 ? output0.addr : output1.addr,
                            *n256_output_prefix_smem);
                    }
                    if (hao_direct_publication_leader<C>()) {
                        arrive(n256_prefix_saved[Stage]);
                    }
                    if constexpr (
                        !C::HAO_DIRECT_SM103_N256_DIRECT_PV_HANDOFF) {
                        n256_prefix_phase ^= 1;
                    }
                }
            }
            hao_direct_n96_lifecycle_stamp<C>(
                g, 16 + reader_warp, 4 * n + 1);
            hao_direct_fp4pv_wait(
                score_full[Stage], score_phase);
            hao_direct_n96_lifecycle_stamp<C>(
                g, 16 + reader_warp, 4 * n + 2);
            tensor_after_thread_sync();
            if constexpr (
                C::HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP &&
                Stage == 0) {
                if (n + 1 == key_tiles) {
                    // The two-tile job is faster when the reader carries the
                    // synthetic handoff instead of the issuer bulk-arriving
                    // three barriers.
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        hao_direct_arrive_rank0<C>(
                            sfqk_load[OppositeScaleStage]);
                        hao_direct_arrive_rank0<C>(
                            p_first_o_rescaled[Stage]);
                        hao_direct_arrive_rank0<C>(p_tail[Stage]);
                    }
                    score_phase ^= 1;
                    continue;
                }
            }
            const uint32_t score_row_base =
                score_owned.addr + (reader_warp * 32 << 16);

            if constexpr (
                C::HAO_DIRECT_SM103_N256_ASYNC_PREFIX_RESTORE) {
                if (n != 0) {
                    hao_direct_restore_regs_x16_to_tmem_async(
                        Stage == 0 ? output0.addr : output1.addr,
                        n256_output_prefix_regs);
                }
            }

            if constexpr (C::HAO_DIRECT_N96) {
                static_assert(C::Nb == 96);
                static_assert(C::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX);
                static_assert(C::HAO_DIRECT_MXFP4_PV);

                if constexpr (C::HAO_DIRECT_FIXED_P_CEILING) {
                    float tile_sum =
                        hao_direct_store_fixed_p_quarter<2, C>(
                            p_owned, p_scale_owned, reader_warp);
                    tile_sum +=
                        hao_direct_store_fixed_p_quarter<0, C>(
                            p_owned, p_scale_owned, reader_warp);
                    tile_sum +=
                        hao_direct_store_fixed_p_quarter<1, C>(
                            p_owned, p_scale_owned, reader_warp);

                    if constexpr (C::HAO_DIRECT_N96_DUAL_QUERY) {
                        hao_direct_store_n192_p_scale_direct<0>(
                            (Stage == 0 ? S0 : S1) + 96,
                            p_scale_owned,
                            reader_warp);
                        fp4pv_tmem_store_wait();
                    }

                    fp4pv_publish_cta_shared_only();
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        if constexpr (C::HAO_DIRECT_N96_DUAL_QUERY) {
                            hao_direct_arrive_rank0<C>(
                                p_first_o_rescaled[Stage]);
                        }
                        hao_direct_arrive_rank0<C>(p_tail[Stage]);
                    }
                    row_sum += tile_sum;
                    score_phase ^= 1;
                    continue;
                }

                if constexpr (C::HAO_DIRECT_SCORE_PACK_CEILING) {
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
                    upstream_load_score_quarter_x32<1, C>(
                        scratch, score_row_base, n, query_tile);
                    tile_sum +=
                        hao_direct_pack_raw_score_quarter<1, C>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch);

                    if constexpr (C::HAO_DIRECT_N96_DUAL_QUERY) {
                        hao_direct_store_n192_p_scale_direct<0>(
                            (Stage == 0 ? S0 : S1) + 96,
                            p_scale_owned,
                            reader_warp);
                        fp4pv_tmem_store_wait();
                    }

                    fp4pv_publish_cta_shared_only();
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        if constexpr (C::HAO_DIRECT_N96_DUAL_QUERY) {
                            hao_direct_arrive_rank0<C>(
                                p_first_o_rescaled[Stage]);
                        }
                        hao_direct_arrive_rank0<C>(p_tail[Stage]);
                    }
                    row_sum += tile_sum;
                    score_phase ^= 1;
                    continue;
                }

                score_quarter_regs scratch;
                uint8_t pair_e8m0 = 0;
                const bool complete_n96_tile =
                    n + 1 < key_tiles;
                float tile_sum = 0.0f;
                float quarter_max =
                    base_types::constants<float>::neg_infty();

                // P starts in retired score quarter 2 so the first packed P
                // store aliases that region. Consume quarter 2 before any P
                // store; quarters 0 and 1 remain live in disjoint columns.
                if constexpr (
                    C::N96_VALID_TAIL_QUARTERS == 3) {
                    quarter_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (Sm103LdRedQ2) {
                        quarter_max =
                            upstream_load_score_quarter_x32_red<2, C>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                    } else {
                        upstream_load_score_quarter_x32<2, C>(
                            scratch, score_row_base, n, query_tile);
                        upstream_reduce_score_quarter<C>(
                            scratch, quarter_max);
                    }
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            2,
                            C,
                            (StageAffineMask & 0x4u) != 0>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                0.0f,
                                scale_log2,
                                quarter_max);
                } else if (complete_n96_tile) {
                    quarter_max =
                        base_types::constants<float>::neg_infty();
                    upstream_load_score_quarter_x32<2, C>(
                        scratch, score_row_base, n, query_tile);
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max);
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            2,
                            C,
                            (StageAffineMask & 0x4u) != 0>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                0.0f,
                                scale_log2,
                                quarter_max);
                } else {
                    hao_direct_store_zero_mxfp4_p_quarter<2, C>(
                        p_owned, p_scale_owned, reader_warp);
                }

                quarter_max =
                    base_types::constants<float>::neg_infty();
                if constexpr (Sm103LdRedQ0) {
                    quarter_max =
                        upstream_load_score_quarter_x32_red<0, C>(
                            scratch,
                            score_row_base,
                            n,
                            query_tile);
                } else {
                    upstream_load_score_quarter_x32<0, C>(
                        scratch, score_row_base, n, query_tile);
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max);
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
                            0.0f,
                            scale_log2,
                            quarter_max,
                            nullptr,
                            0.0f,
                            0.0f,
                            nullptr,
                            &pair_e8m0);

                if constexpr (
                    C::N96_VALID_TAIL_QUARTERS >= 2) {
                    quarter_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (Sm103LdRedQ1) {
                        quarter_max =
                            upstream_load_score_quarter_x32_red<1, C>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                    } else {
                        upstream_load_score_quarter_x32<1, C>(
                            scratch, score_row_base, n, query_tile);
                        upstream_reduce_score_quarter<C>(
                            scratch, quarter_max);
                    }
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            1,
                            C,
                            (StageAffineMask & 0x2u) != 0>(
                                p_owned,
                                p_scale_owned,
                                reader_warp,
                                scratch,
                                0.0f,
                                scale_log2,
                                quarter_max,
                                nullptr,
                                0.0f,
                                0.0f,
                                nullptr,
                                &pair_e8m0);
                } else if (complete_n96_tile) {
                    quarter_max =
                        base_types::constants<float>::neg_infty();
                    upstream_load_score_quarter_x32<1, C>(
                        scratch, score_row_base, n, query_tile);
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
                                0.0f,
                                scale_log2,
                                quarter_max,
                                nullptr,
                                0.0f,
                                0.0f,
                                nullptr,
                                &pair_e8m0);
                } else {
                    hao_direct_store_zero_mxfp4_p_quarter<1, C>(
                        p_owned, p_scale_owned, reader_warp);
                }

                if constexpr (C::HAO_DIRECT_N96_DUAL_QUERY) {
                    // K96 block-scale MMA consumes three consecutive E8M0
                    // bytes. The ordinary N128 scale copy rearranges four
                    // quarters and is not a valid K96 scale page.
                    hao_direct_store_n192_p_scale_direct<0>(
                        (Stage == 0 ? S0 : S1) + 96,
                        p_scale_owned,
                        reader_warp);
                    fp4pv_tmem_store_wait();
                }

                fp4pv_publish_cta_shared_only();
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    if constexpr (C::HAO_DIRECT_N96_DUAL_QUERY) {
                        hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                    }
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
                }
                hao_direct_n96_lifecycle_stamp<C>(
                    g, 16 + reader_warp, 4 * n + 3);
                row_sum += tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_SM103_N256) {
                static_assert(C::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS);

                if constexpr (!C::HAO_DIRECT_FIXED_P_CEILING) {
                    static_assert(C::Nb == 192);
                    static_assert(
                        C::HAO_DIRECT_SM103_N256_DUAL_QUERY);
                    static_assert(C::HAO_DIRECT_MXFP4_PV);
                    static_assert(
                        C::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX);

                    score_quarter_regs scratch;
                    score_quarter_regs paired;
                    uint8_t pair_e8m0_01 = 0;
                    uint8_t pair_e8m0_23 = 0;
                    float tile_sum = 0.0f;
                    float quarter_max =
                        base_types::constants<float>::neg_infty();

                    // Packed P aliases retired score quarter 2. Consume it
                    // before the first P store, then pair independent TMEM
                    // loads to amortize the six-quarter read latency.
                    upstream_load_score_quarter_x32<2, C>(
                        scratch, score_row_base, n, query_tile);
                    upstream_reduce_score_quarter<C>(
                        scratch, quarter_max);
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            2,
                            PairScaleC,
                            (StageAffineMask & 0x4u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            0.0f,
                            scale_log2,
                            quarter_max,
                            nullptr,
                            0.0f,
                            0.0f,
                            nullptr,
                            &pair_e8m0_23);

                    float q0_max =
                        base_types::constants<float>::neg_infty();
                    float q1_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (Sm103LdRedQ0 && Sm103LdRedQ1) {
                        q0_max =
                            upstream_load_score_quarter_x32_red<0, C>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                        q1_max =
                            upstream_load_score_quarter_x32_red<1, C>(
                                paired,
                                score_row_base,
                                n,
                                query_tile);
                    } else {
                        upstream_load_score_quarter_x32<0, C, false>(
                            scratch, score_row_base, n, query_tile);
                        upstream_load_score_quarter_x32<1, C, false>(
                            paired, score_row_base, n, query_tile);
                        tensor_load_wait();
                        upstream_reduce_score_quarter<C>(
                            scratch, q0_max);
                        upstream_reduce_score_quarter<C>(
                            paired, q1_max);
                    }
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            0,
                            PairScaleC,
                            (StageAffineMask & 0x1u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            0.0f,
                            scale_log2,
                            q0_max,
                            nullptr,
                            0.0f,
                            0.0f,
                            nullptr,
                            &pair_e8m0_01);
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            1,
                            PairScaleC,
                            (StageAffineMask & 0x2u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            paired,
                            0.0f,
                            scale_log2,
                            q1_max,
                            nullptr,
                            0.0f,
                            0.0f,
                            nullptr,
                            &pair_e8m0_01);

                    float q3_max =
                        base_types::constants<float>::neg_infty();
                    float q4_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (Sm103LdRedQ3 && Sm103LdRedQ0) {
                        q3_max =
                            upstream_load_score_quarter_x32_red<3, C>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                        q4_max =
                            upstream_load_score_quarter_x32_red<4, C>(
                                paired,
                                score_row_base,
                                n,
                                query_tile);
                    } else {
                        upstream_load_score_quarter_x32<3, C, false>(
                            scratch, score_row_base, n, query_tile);
                        upstream_load_score_quarter_x32<4, C, false>(
                            paired, score_row_base, n, query_tile);
                        tensor_load_wait();
                        upstream_reduce_score_quarter<C>(
                            scratch, q3_max);
                        upstream_reduce_score_quarter<C>(
                            paired, q4_max);
                    }

                    // Scale pages reuse score-quarter-3 columns, so only
                    // publish the first K96 scale triplet after q3 is held
                    // in registers.
                    const uint32_t scale_base =
                        C::HAO_DIRECT_SM103_N192_DEDICATED_P0 && Stage == 0
                            ? p_owned.addr + 24
                            : score_owned.addr + 96;
                    hao_direct_store_n192_p_scale_direct<0>(
                        scale_base, p_scale_owned, reader_warp);
                    if constexpr (
                        C::HAO_DIRECT_SM103_N192_EARLY_K96) {
                        fp4pv_tmem_store_wait();
                        fp4pv_publish_cta_shared_only();
                        tensor_before_thread_sync();
                        if (warp::elect_leader()) {
                            hao_direct_arrive_rank0<C>(
                                p_first_o_rescaled[Stage]);
                        }
                    }

                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            3,
                            PairScaleC,
                            (StageAffineMask & 0x8u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            0.0f,
                            scale_log2,
                            q3_max,
                            nullptr,
                            0.0f,
                            0.0f,
                            nullptr,
                            &pair_e8m0_23);
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            4,
                            PairScaleC,
                            (StageAffineMask & 0x1u) != 0 ||
                                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 19 ||
                                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 20 ||
                                C::HAO_DIRECT_MX_PWL_EXP2_MODE == 23>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            paired,
                            0.0f,
                            scale_log2,
                            q4_max);

                    quarter_max =
                        base_types::constants<float>::neg_infty();
                    if constexpr (Sm103LdRedQ1) {
                        quarter_max =
                            upstream_load_score_quarter_x32_red<5, C>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                    } else {
                        upstream_load_score_quarter_x32<5, C>(
                            scratch, score_row_base, n, query_tile);
                        upstream_reduce_score_quarter<C>(
                            scratch, quarter_max);
                    }
                    tile_sum +=
                        hao_direct_transform_pack_fp4_quarter<
                            5,
                            PairScaleC,
                            (StageAffineMask & 0x2u) != 0>(
                            p_owned,
                            p_scale_owned,
                            reader_warp,
                            scratch,
                            0.0f,
                            scale_log2,
                            quarter_max);

                    hao_direct_store_n192_p_scale_direct<1>(
                        scale_base, p_scale_owned, reader_warp);
                    fp4pv_tmem_store_wait();
                    fp4pv_publish_cta_shared_only();
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        hao_direct_arrive_rank0<C>(p_tail[Stage]);
                    }

                    row_sum += tile_sum;
                    score_phase ^= 1;
                    continue;
                }

                // P occupies the retired score-quarter-2 columns. Keep the
                // q2-first order so the same lifetime remains valid when the
                // real eight-quarter softmax replaces this fixed-P gate.
                float tile_sum =
                    hao_direct_store_fixed_p_quarter<2, C>(
                        p_owned, p_scale_owned, reader_warp);
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<0, C>(
                        p_owned, p_scale_owned, reader_warp);
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<1, C>(
                        p_owned, p_scale_owned, reader_warp);
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<3, C>(
                        p_owned, p_scale_owned, reader_warp);
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<4, C>(
                        p_owned, p_scale_owned, reader_warp);
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<5, C>(
                        p_owned, p_scale_owned, reader_warp);
                if constexpr (C::Nb == 256) {
                    tile_sum +=
                        hao_direct_store_fixed_p_quarter<6, C>(
                            p_owned, p_scale_owned, reader_warp);
                    tile_sum +=
                        hao_direct_store_fixed_p_quarter<7, C>(
                            p_owned, p_scale_owned, reader_warp);
                }

                const uint32_t scale_base = score_owned.addr +
                    (C::HAO_DIRECT_SM103_N256_DUAL_QUERY ? 96 : 384);
                hao_direct_store_n256_fixed_p_scale_direct<0>(
                    scale_base, reader_warp);
                hao_direct_store_n256_fixed_p_scale_direct<1>(
                    scale_base, reader_warp);
                if constexpr (C::Nb == 256) {
                    hao_direct_store_n256_fixed_p_scale_direct<2>(
                        scale_base, reader_warp);
                }
                fp4pv_tmem_store_wait();
                if constexpr (
                    C::HAO_DIRECT_SM103_N256_ASYNC_PREFIX_RESTORE) {
                    tensor_after_thread_sync();
                    warpgroup::sync(1);
                }
                if constexpr (
                    C::HAO_DIRECT_SM103_N256_DUAL_QUERY &&
                    C::Nb == 256 &&
                    (Stage == 1 ||
                     C::HAO_DIRECT_SM103_N256_SPLIT_OUTPUT_SCALES)) {
                    if (n != 0 &&
                        !C::HAO_DIRECT_SM103_N256_ASYNC_PREFIX_RESTORE) {
                        if constexpr (
                            C::HAO_DIRECT_SM103_N256_PREFIX_REGS) {
                            hao_direct_restore_regs_x16_to_tmem(
                                Stage == 0 ? output0.addr : output1.addr,
                                n256_output_prefix_regs);
                        } else {
                            hao_direct_restore_smem_x16_to_tmem(
                                Stage == 0 ? output0.addr : output1.addr,
                                *n256_output_prefix_smem);
                        }
                    }
                }
                fp4pv_publish_cta_shared_only();
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
                }

                row_sum += tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (
                C::HAO_DIRECT_FIXED_P_CEILING &&
                !C::HAO_DIRECT_SM103_N256) {
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
                if constexpr (
                    !C::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY) {
                    tensor_before_thread_sync();
                    if (warp::elect_leader()) {
                        hao_direct_arrive_rank0<C>(
                            sfqk_load[OppositeScaleStage]);
                    }
                }
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<1, C>(
                        p_owned, p_scale_owned, reader_warp);
                if constexpr (C::HAO_DIRECT_EARLY_P) {
                    if constexpr (
                        C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                        hao_direct_store_mxfp4_p_scale_half_direct<0>(
                            scale_base_owned + 32,
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
                        hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                    }
                }
                tile_sum +=
                    hao_direct_store_fixed_p_quarter<3, C>(
                        p_owned, p_scale_owned, reader_warp);
                if constexpr (
                    C::HAO_DIRECT_MX_EARLY_DIRECT_SCALE) {
                    hao_direct_store_mxfp4_p_scale_half_direct<1>(
                        scale_base_owned + 48,
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
                        hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                    }
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
                    hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                        hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
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
                        hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                    }
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
                    hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                if (hao_direct_publication_leader<C>()) {
                    hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
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
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
                upstream_apply_causal_score_quarter_x32<0, C>(
                    retained_q0, n, query_tile);
                upstream_apply_causal_score_quarter_x32<2, C>(
                    scratch, n, query_tile);
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
                    hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                    hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
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
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
                    hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                    hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
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
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
                    hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                    hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
                        hao_direct_arrive_rank0<C>(p_tail[Stage]);
                    }

                    row_sum += tile_sum;
                    score_phase ^= 1;
                    continue;
                }

                float tile_lse_sum = 0.0f;
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
                    float delayed_q2_max =
                        base_types::constants<float>::neg_infty();
                    float delayed_q3_max =
                        base_types::constants<float>::neg_infty();

                    // Q0 packing overwrites the low half of score Q2. Retain
                    // that half first, then pair each remaining score load.
                    if constexpr (Sm103LdRedQ2) {
                        delayed_q2_max =
                            upstream_load_score_quarter_x32_red<2, C>(
                                delayed_q2,
                                score_row_base,
                                n,
                                query_tile);
                    } else {
                        upstream_load_score_quarter_half_x16<
                            2, 0, C, false>(
                                delayed_q2,
                                score_row_base,
                                n,
                                query_tile);
                    }
                    if constexpr (Sm103LdRedQ0) {
                        quarter_max =
                            upstream_load_score_quarter_x32_red<0, C>(
                                scratch,
                                score_row_base,
                                n,
                                query_tile);
                    } else {
                        upstream_load_score_quarter_x32<0, C, false>(
                            scratch,
                            score_row_base,
                            n,
                            query_tile);
                    }
                    tensor_load_wait();
                    upstream_apply_causal_score_quarter_half_x16<
                        2, 0, C>(delayed_q2, n, query_tile);
                    upstream_apply_causal_score_quarter_x32<0, C>(
                        scratch, n, query_tile);
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
                                    hao_direct_arrive_rank0<C>(
                                        sfqk_load[
                                            OppositeScaleStage]);
                                }
                            }
                        } else {
                            tensor_before_thread_sync();
                            if (warp::elect_leader()) {
                                hao_direct_arrive_rank0<C>(
                                    sfqk_load[
                                        OppositeScaleStage]);
                            }
                        }
                    }
                    if constexpr (
                        Stage == 0 ||
                        (Stage == 1 &&
                         C::HAO_DIRECT_MX_STAGE1_EARLY_Q1)) {
                        if constexpr (Sm103LdRedQ12LoadPath) {
                            if constexpr (Sm103LdRedQ1) {
                                prefetched_q1_max =
                                    upstream_load_score_quarter_x32_red<1, C>(
                                        prefetched_q1,
                                        score_row_base,
                                        n,
                                        query_tile);
                            } else {
                                upstream_load_score_quarter_x32<1, C, false>(
                                    prefetched_q1,
                                    score_row_base,
                                    n,
                                    query_tile);
                            }
                            if constexpr (!Sm103LdRedQ2) {
                                upstream_load_score_quarter_half_x16<
                                    2, 1, C, false>(
                                        delayed_q2,
                                        score_row_base,
                                        n,
                                        query_tile);
                            }
                        } else if constexpr (
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
                        if constexpr (!Sm103LdRedQ0) {
                            upstream_reduce_score_quarter<C>(
                                scratch, quarter_max);
                        }
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
                                hao_direct_arrive_rank0<C>(
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
                    bool stabilize_initial_causal_row = false;
                    if (stabilize_initial_causal_quarter && n == 0) {
                        const float q0_group_max =
                            quarter_max * scale_log2 +
                            0.5f * static_cast<float>(
                                C::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2);
                        const uint8_t q0_shiftless_e8m0 =
                            hao_direct_select_mxfp4_e8m0<
                                PairScaleC,
                                C::HAO_DIRECT_MX_P_EFFECTIVE_MAX>(
                                    q0_group_max);
                        stabilize_initial_causal_row =
                            q0_shiftless_e8m0 <=
                            C::HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2;
                        if (stabilize_initial_causal_row) {
                            row_max = quarter_max;
                            shiftless_row_max = row_max;
                        }
                    }
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
                        upstream_apply_causal_score_quarter_x32<1, C>(
                            prefetched_q1, n, query_tile);
                        upstream_apply_causal_score_quarter_half_x16<
                            2, 1, C>(delayed_q2, n, query_tile);
                        quarter_max =
                            base_types::constants<float>::neg_infty();
                    }
                    float q0_unscaled_sum = 0.0f;
                    uint32_t q0_denom_carrier = 0;
                    uint32_t q1_denom_carrier = 0;
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
                                retained_first_words[0],
                                C::HAO_DIRECT_CAUSAL &&
                                    C::HAO_DIRECT_NV_DENOM_WORDS < 4 &&
                                    n == query_tile
                                        ? static_cast<int>(
                                              warpgroup::laneid())
                                        : -1);
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
                                InterleaveQ1Max,
                                false,
                                C::HAO_DIRECT_MX_CAUSAL_Q3_PROGRESSIVE_REUSE,
                                DeferDenomFinalize>(
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
                                &quarter_max,
                                score_row_base,
                                n,
                                query_tile,
                                &q0_denom_carrier,
                                &tile_lse_sum,
                                &lse_max_e8m0,
                                &lse_scaled_quantized_sum_x2);
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
                    if (stabilize_initial_causal_row) {
                        shiftless_row_max = 0.0f;
                    }
                    if constexpr (
                        C::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF) {
                        hao_direct_store_nv_first_payload_quarter_smem<0>(
                            nv_first_payload_smem[Stage],
                            reader_warp,
                                retained_first_words[0]);
                    }
                    if constexpr (
                        C::HAO_DIRECT_MX_CAUSAL_Q3_LOOKAHEAD == 1 &&
                        !C::HAO_DIRECT_MX_CAUSAL_Q3_PROGRESSIVE_REUSE &&
                        !DualQ3CorrectionOffload) {
                        upstream_load_score_quarter_x32<3, C, false>(
                            scratch,
                            score_row_base,
                            n,
                            query_tile);
                    }
                    if constexpr (
                        ((C::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK |
                          C::HAO_DIRECT_MX_SOFTMAX_V_SCALE_PREFETCH_MASK) &
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
                                if (n + 1 < key_tiles) {
                                    hao_direct_fp4pv_wait(
                                        score_full[0],
                                        score_phase ^ 1);
                                    tensor_after_thread_sync();
                                }
                            }
                            if constexpr (C::HAO_DIRECT_NVFP4_PV) {
                                hao_direct_load_nvfp4_scale_ranklocal<0>(
                                    score_owned.addr + 8,
                                    kv_scale[v_slot].v);
                                hao_direct_load_nvfp4_scale_ranklocal<1>(
                                    score_owned.addr + 12,
                                    kv_scale[v_slot].v);
                            } else {
                                full_tt_fp8e8m0<16> v_scale(
                                    score_owned.addr + 16);
                                hao_direct_load_mxfp4_scale_ranklocal(
                                    v_scale,
                                    kv_scale[v_slot].v);
                            }
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
                            hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                        if constexpr (Sm103LdRedQ12LoadPath) {
                            if constexpr (!Sm103LdRedQ2) {
                                upstream_load_score_quarter_half_x16<
                                    2, 1, C, false>(
                                        delayed_q2,
                                        score_row_base,
                                        n,
                                        query_tile);
                            }
                            if constexpr (Sm103LdRedQ1) {
                                prefetched_q1_max =
                                    upstream_load_score_quarter_x32_red<1, C>(
                                        prefetched_q1,
                                        score_row_base,
                                        n,
                                        query_tile);
                            } else {
                                upstream_load_score_quarter_x32<1, C, false>(
                                    prefetched_q1,
                                    score_row_base,
                                    n,
                                    query_tile);
                            }
                        } else if constexpr (
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
                        upstream_apply_causal_score_quarter_x32<1, C>(
                            prefetched_q1, n, query_tile);
                        upstream_apply_causal_score_quarter_half_x16<
                            2, 1, C>(delayed_q2, n, query_tile);
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
                            if constexpr (Sm103LdRedQ1) {
                                quarter_max = prefetched_q1_max;
                            } else if constexpr (
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
                                retained_first_words[1],
                                C::HAO_DIRECT_CAUSAL &&
                                    C::HAO_DIRECT_NV_DENOM_WORDS < 4 &&
                                    n == query_tile
                                        ? static_cast<int>(
                                              warpgroup::laneid())
                                        : -1);
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
                                (InterleaveQ2Max ||
                                 C::HAO_DIRECT_MX_LAGGED_Q1_SCALE ||
                                 SelfInterleaveQ1Max),
                                (SelfInterleaveQ1Max &&
                                 C::HAO_DIRECT_MX_Q1_SELF_MAX >= 2),
                                false,
                                DeferDenomFinalize>(
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
                                reuse_q1_pair_scale && !InterleaveQ1Max,
                                InterleaveQ2Max
                                    ? &delayed_q2
                                    : &prefetched_q1,
                                InterleaveQ2Max
                                    ? &delayed_q2_max
                                    : &lagged_q1_raw_max,
                                0,
                                0,
                                0,
                                &q1_denom_carrier,
                                &tile_lse_sum,
                                &lse_max_e8m0,
                                &lse_scaled_quantized_sum_x2);
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
                        C::HAO_DIRECT_MX_CAUSAL_Q3_LOOKAHEAD == 0 &&
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
                            if constexpr (Sm103LdRedQ3) {
                                delayed_q3_max =
                                    upstream_load_score_quarter_x32_red<3, C>(
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
                        }
                    }
                    if constexpr (
                        C::HAO_DIRECT_D64_TRIPLE_P &&
                        C::HAO_DIRECT_D64_TRIPLE_P_STAGE == Stage) {
                        if constexpr (
                            C::HAO_DIRECT_D64_TRIPLE_P_NO_RELEASE) {
                            // Diagnostic: rely on TCGEN operand capture.
                        } else if constexpr (
                            C::HAO_DIRECT_D64_TRIPLE_P_KV_RELEASE) {
                            if (d64_previous_p_v_item >= 0) {
                                const int reusable_slot =
                                    d64_previous_p_v_item % C::KV_STAGES;
                                const int reusable_phase =
                                    (d64_previous_p_v_item /
                                     C::KV_STAGES) & 1;
                                hao_direct_fp4pv_wait(
                                    kv_empty_at(reusable_slot),
                                    reusable_phase);
                                tensor_after_thread_sync();
                            }
                            d64_previous_p_v_item =
                                d64_kv_job_base + 1 + 2 * n;
                        } else if (d64_p_first_reuse_started) {
                            if constexpr (
                                C::HAO_DIRECT_D64_TRIPLE_P_SPLIT_RELEASE) {
                                hao_direct_fp4pv_wait(
                                    d64_p_first_reusable[Stage],
                                    d64_p_first_reuse_phase);
                                d64_p_first_reuse_phase ^= 1;
                            } else if constexpr (
                                C::HAO_DIRECT_D64_TRIPLE_P_LEADER_WAIT) {
                                if (reader_warp == 0 &&
                                    warp::elect_leader()) {
                                    hao_direct_fp4pv_wait(
                                        d64_p_tail_reusable[Stage],
                                        d64_p_tail_reuse_phase);
                                }
                                warpgroup::sync(2 + Stage);
                                d64_p_tail_reuse_phase ^= 1;
                            } else {
                                hao_direct_fp4pv_wait(
                                    d64_p_tail_reusable[Stage],
                                    d64_p_tail_reuse_phase);
                                d64_p_tail_reuse_phase ^= 1;
                            }
                            tensor_after_thread_sync();
                        } else if constexpr (
                            !C::HAO_DIRECT_D64_TRIPLE_P_KV_RELEASE) {
                            d64_p_first_reuse_started = true;
                        }
                    }
                    if constexpr (C::HAO_DIRECT_MXFP4_PV) {
                        if constexpr (
                            C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF) {
                            hao_direct_store_mxfp4_p_scale_half_register<
                                0,
                                !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE>(
                                    p_scale_store_base_owned + 32,
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
                                    p_scale_store_base_owned + 32,
                                    reader_warp,
                                    q0_scale,
                                    q1_mx_scale);
                        } else {
                            hao_direct_store_mxfp4_p_scale_half_direct<
                                0,
                                !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                                C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                                    p_scale_store_base_owned + 32,
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
                        if constexpr (InterleaveQ2Max) {
                            quarter_max = delayed_q2_max;
                        } else {
                            quarter_max = Sm103LdRedQ2
                                ? delayed_q2_max
                                : base_types::constants<float>::neg_infty();
                            if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                                if constexpr (!Sm103LdRedQ2) {
                                    upstream_reduce_score_quarter<C>(
                                        delayed_q2, quarter_max);
                                }
                            }
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
                        hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
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
                        if constexpr (Sm103LdRedQ3) {
                            delayed_q3_max =
                                upstream_load_score_quarter_x32_red<3, C>(
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
                    }
                    if constexpr (DeferDenomFinalizeMode == 1) {
                        q0_sum =
                            hao_direct_finalize_mxfp4_denom_carrier(
                                q0_denom_carrier);
                        q1_sum =
                            hao_direct_finalize_mxfp4_denom_carrier(
                                q1_denom_carrier);
                    }
                    if constexpr (
                        !C::HAO_DIRECT_MX_EARLY_Q2_REDUCE) {
                        quarter_max = Sm103LdRedQ2
                            ? delayed_q2_max
                            : base_types::constants<float>::neg_infty();
                        if constexpr (C::HAO_DIRECT_P_QUARTER_MAX) {
                            if constexpr (!Sm103LdRedQ2) {
                                upstream_reduce_score_quarter<C>(
                                    delayed_q2, quarter_max);
                            }
                        }
                    }
                    uint16_t q2_scale_bits = 0;
                    uint8_t q2_mx_scale = 0;
                    uint32_t q2_denom_carrier = 0;
                    if constexpr (InterleaveQ3Max) {
                        tensor_load_wait();
                        upstream_apply_causal_score_quarter_x32<3, C>(
                            scratch, n, query_tile);
                        delayed_q3_max =
                            base_types::constants<float>::neg_infty();
                    }
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
                                -1,
                                q3_hook);
                    if constexpr (!DeferDenomFinalize) {
                        tile_sum += q0_sum;
                        tile_sum += q1_sum;
                    }
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
                                q2_scale_bits,
                                nullptr,
                                nullptr,
                                C::HAO_DIRECT_CAUSAL &&
                                    C::HAO_DIRECT_NV_DENOM_WORDS < 4 &&
                                    n == query_tile
                                        ? static_cast<int>(
                                              warpgroup::laneid())
                                        : -1);
                    } else {
                        tile_sum =
                            hao_direct_transform_pack_fp4_quarter<
                                2,
                                PairScaleC,
                                (StageAffineMask & 0x4u) != 0,
                                C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM,
                                C::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST,
                                (C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF ||
                                 C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH),
                                InterleaveQ3Max,
                                false,
                                false,
                                DeferDenomFinalize>(
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
                                &q23_pair_e8m0,
                                true,
                                &scratch,
                                &delayed_q3_max,
                                0,
                                0,
                                0,
                                &q2_denom_carrier,
                                &tile_lse_sum,
                                &lse_max_e8m0,
                                &lse_scaled_quantized_sum_x2);
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
                    if constexpr (!DeferDenomFinalize) {
                        tile_sum += q0_sum;
                        tile_sum += q1_sum;
                    }
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
                        uint32_t q3_denom_carrier = 0;
#if TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD == 0
                        uint16_t q3_scale_bits = 0;
                        if constexpr (
                            C::HAO_DIRECT_MX_CAUSAL_Q3_LOOKAHEAD == 0) {
                            tensor_load_wait();
                        }
                        if constexpr (!InterleaveQ3Max) {
                            upstream_apply_causal_score_quarter_x32<3, C>(
                                scratch, n, query_tile);
                        }
                        quarter_max =
                            InterleaveQ3Max || Sm103LdRedQ3
                                ? delayed_q3_max
                                : base_types::constants<float>::neg_infty();
                        if constexpr (
                            C::HAO_DIRECT_P_QUARTER_MAX &&
                            !(PairScaleC::HAO_DIRECT_MX_PAIR_SCALE_REUSE & 2)) {
                            if constexpr (
                                !InterleaveQ3Max && !Sm103LdRedQ3) {
                                upstream_reduce_score_quarter<C>(
                                    scratch, quarter_max);
                            }
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
                                    q3_scale_bits,
                                    nullptr,
                                    nullptr,
                                    C::HAO_DIRECT_CAUSAL &&
                                        C::HAO_DIRECT_NV_DENOM_WORDS < 4 &&
                                        n == query_tile
                                            ? static_cast<int>(
                                                  warpgroup::laneid())
                                            : -1);
                        } else {
                            tile_sum +=
                                hao_direct_transform_pack_fp4_quarter<
                                3,
                                PairScaleC,
                                (StageAffineMask & 0x8u) != 0,
                                C::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM,
                                C::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST,
                                (C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF ||
                                 C::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH),
                                false,
                                false,
                                false,
                                DeferDenomFinalize>(
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
                                    &q23_pair_e8m0,
                                    true,
                                    nullptr,
                                    nullptr,
                                    0,
                                    0,
                                    0,
                                    &q3_denom_carrier,
                                    &tile_lse_sum,
                                    &lse_max_e8m0,
                                    &lse_scaled_quantized_sum_x2);
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

                        if constexpr (
                            C::HAO_DIRECT_MX_D4ALL_P_SCALE_PUBLICATION) {
                            // Q3 is the last d4all quarter produced.  Read the
                            // completed Q0..Q3 byte word from its canonical
                            // swizzled shared-memory owner before releasing P.
                            hao_direct_publish_d4all_p_scale_word<C>(
                                g,
                                p_scale_owned,
                                task.batch,
                                task.head,
                                query_tile,
                                n,
                                reader_warp);
                        }

                        if constexpr (
                            C::HAO_DIRECT_D64_TRIPLE_P &&
                            C::HAO_DIRECT_D64_TRIPLE_P_STAGE == Stage &&
                            C::HAO_DIRECT_D64_TRIPLE_P_SPLIT_RELEASE) {
                            if (d64_p_tail_reuse_started) {
                                hao_direct_fp4pv_wait(
                                    d64_p_tail_reusable[Stage],
                                    d64_p_tail_reuse_phase);
                                d64_p_tail_reuse_phase ^= 1;
                                tensor_after_thread_sync();
                            } else {
                                d64_p_tail_reuse_started = true;
                            }
                        }
                        if constexpr (C::HAO_DIRECT_MXFP4_PV) {
                            if constexpr (
                                C::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF) {
                                hao_direct_store_mxfp4_p_scale_half_register<
                                    1,
                                    !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE>(
                                        p_scale_store_base_owned + 48,
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
                                        p_scale_store_base_owned + 48,
                                        reader_warp,
                                        q2_scale,
                                        q3_mx_scale);
                            } else {
                                hao_direct_store_mxfp4_p_scale_half_direct<
                                    1,
                                    !C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,
                                    C::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH>(
                                        p_scale_store_base_owned + 48,
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
                            hao_direct_arrive_rank0<C>(p_tail[Stage]);
                        }
                        if constexpr (DeferDenomFinalize) {
                            if constexpr (StableLseLogsum && !DualLse) {
                                hao_direct_accumulate_mxfp4_denom_carrier_log_domain(
                                    q0_denom_carrier,
                                    lse_max_e8m0,
                                    lse_scaled_quantized_sum_x2);
                                hao_direct_accumulate_mxfp4_denom_carrier_log_domain(
                                    q1_denom_carrier,
                                    lse_max_e8m0,
                                    lse_scaled_quantized_sum_x2);
                                hao_direct_accumulate_mxfp4_denom_carrier_log_domain(
                                    q2_denom_carrier,
                                    lse_max_e8m0,
                                    lse_scaled_quantized_sum_x2);
                                hao_direct_accumulate_mxfp4_denom_carrier_log_domain(
                                    q3_denom_carrier,
                                    lse_max_e8m0,
                                    lse_scaled_quantized_sum_x2);
                            }
                            if constexpr (DeferDenomFinalizeMode >= 2) {
                                q0_sum =
                                    hao_direct_finalize_mxfp4_denom_carrier(
                                        q0_denom_carrier);
                                q1_sum =
                                    hao_direct_finalize_mxfp4_denom_carrier(
                                        q1_denom_carrier);
                            }
                            tile_sum =
                                hao_direct_finalize_mxfp4_denom_carrier(
                                    q2_denom_carrier);
                            tile_sum += q0_sum;
                            tile_sum += q1_sum;
                            tile_sum +=
                                hao_direct_finalize_mxfp4_denom_carrier(
                                    q3_denom_carrier);
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
                        hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                            hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
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
                                hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                            }
                            hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
                if constexpr (DualLse) {
                    lse_row_sum += tile_lse_sum;
                }
                if constexpr (
                    ((C::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK |
                      C::HAO_DIRECT_MX_SOFTMAX_V_SCALE_PREFETCH_MASK) &
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
                    hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                    hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
                    hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                    hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                    hao_direct_arrive_rank0<C>(p_tail[Stage]);
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
            tensor_before_thread_sync();
            if (hao_direct_publication_leader<C>()) {
                hao_direct_arrive_rank0<C>(sfqk_load[OppositeScaleStage]);
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
                    hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
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
            if (hao_direct_publication_leader<C>()) {
                if constexpr (!C::HAO_DIRECT_EARLY_P) {
                    hao_direct_arrive_rank0<C>(p_first_o_rescaled[Stage]);
                }
                hao_direct_arrive_rank0<C>(p_tail[Stage]);
            }

            hao_direct_fp4pv_wait(
                softmax_corr_empty[Stage],
                corr_producer_phase);
            corr_producer_phase ^= 1;
            row_sum = row_sum * correction + tile_sum;
            score_phase ^= 1;
        }

        if constexpr (
            C::HAO_DIRECT_D64_TRIPLE_P_KV_RELEASE &&
            C::HAO_DIRECT_D64_TRIPLE_P_STAGE == Stage) {
            d64_kv_job_base += 2 * key_tiles;
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
        if constexpr (AlternateLseStat) {
            const float alternate_lse_stat = StableLseLogsum
                ? hao_direct_finalize_mxfp4_denom_carrier_log2(
                      lse_max_e8m0,
                      lse_scaled_quantized_sum_x2)
                : lse_row_sum;
            stats_store(
                lse_sum_smem[Stage],
                warpgroup::laneid(),
                alternate_lse_stat);
        }
        __threadfence_block();
        if (warp::elect_leader()) {
            arrive(softmax_corr_full[Stage]);
        }
        hao_direct_n96_lifecycle_stamp<C>(
            g, 16 + reader_warp, 99);
    }

    if (warp::elect_leader()) {
        arrive(tmem_dealloc);
    }
}

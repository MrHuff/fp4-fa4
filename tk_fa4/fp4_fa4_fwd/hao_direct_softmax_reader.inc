{
    constexpr int Stage = HAO_DIRECT_SOFTMAX_STAGE;
    static_assert(Stage == 0 || Stage == 1);
    constexpr int OppositeScaleStage = 1 - Stage;
    const int reader_warp = warpgroup::warpid();
    const Score score_owned =
        tm.template allocate<Score>(Stage == 0 ? S0 : S1);
    const P p_owned = tm.template allocate<P>(Stage == 0 ? P0 : P1);

    int score_phase = 0;
    int corr_producer_phase = 1;

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

        hao_direct_wait(
            softmax_corr_empty[Stage], corr_producer_phase);
        corr_producer_phase ^= 1;

        float row_max =
            C::HAO_DIRECT_FIXED_P_CEILING ||
                    C::HAO_DIRECT_SCORE_PACK_CEILING ||
                    C::HAO_DIRECT_SHIFTLESS
                ? 0.0f
                : base_types::constants<float>::neg_infty();
        float row_sum = 0.0f;

        #pragma unroll 1
        for (int n = 0; n < C::N_TILES; ++n) {
            hao_direct_wait(score_full[Stage], score_phase);
            tensor_after_thread_sync();
            const uint32_t score_row_base =
                score_owned.addr + (reader_warp * 32 << 16);

            if constexpr (C::HAO_DIRECT_FIXED_P_CEILING) {
                hao_direct_shared_stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    1.0f);
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                float tile_sum =
                    hao_direct_store_fixed_p_fp8_quarter<2>(
                        p_owned, reader_warp);
                tile_sum +=
                    hao_direct_store_fixed_p_fp8_quarter<0>(
                        p_owned, reader_warp);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }
                tile_sum +=
                    hao_direct_store_fixed_p_fp8_quarter<1>(
                        p_owned, reader_warp);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                }
                tile_sum +=
                    hao_direct_store_fixed_p_fp8_quarter<3>(
                        p_owned, reader_warp);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_tail[Stage]);
                }

                hao_direct_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum += tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_SCORE_PACK_CEILING) {
                hao_direct_shared_stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    1.0f);
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                score_quarter_regs scratch;
                upstream_load_score_quarter_x32<2, C>(
                    scratch, score_row_base, n, query_tile);
                float tile_sum =
                    hao_direct_pack_raw_score_fp8_quarter<2>(
                        p_owned, reader_warp, scratch);
                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
                tile_sum +=
                    hao_direct_pack_raw_score_fp8_quarter<0>(
                        p_owned, reader_warp, scratch);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }
                upstream_load_score_quarter_x32<1, C>(
                    scratch, score_row_base, n, query_tile);
                tile_sum +=
                    hao_direct_pack_raw_score_fp8_quarter<1>(
                        p_owned, reader_warp, scratch);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                }
                upstream_load_score_quarter_x32<3, C>(
                    scratch, score_row_base, n, query_tile);
                tile_sum +=
                    hao_direct_pack_raw_score_fp8_quarter<3>(
                        p_owned, reader_warp, scratch);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_tail[Stage]);
                }

                hao_direct_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum += tile_sum;
                score_phase ^= 1;
                continue;
            }

            if constexpr (C::HAO_DIRECT_SHIFTLESS) {
                hao_direct_shared_stats_store(
                    correction_owned,
                    warpgroup::laneid(),
                    1.0f);
                if (warp::elect_leader()) {
                    arrive(softmax_corr_full[Stage]);
                }

                score_quarter_regs scratch;
                float quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<2, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                float tile_sum =
                    hao_direct_transform_pack_shiftless_fp8_quarter<
                        2, C>(
                        p_owned,
                        reader_warp,
                        scratch,
                        quarter_max,
                        scale_log2);

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }
                tile_sum +=
                    hao_direct_transform_pack_shiftless_fp8_quarter<
                        0, C>(
                        p_owned,
                        reader_warp,
                        scratch,
                        quarter_max,
                        scale_log2);

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<1, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tile_sum +=
                    hao_direct_transform_pack_shiftless_fp8_quarter<
                        1, C>(
                        p_owned,
                        reader_warp,
                        scratch,
                        quarter_max,
                        scale_log2);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                }

                quarter_max =
                    base_types::constants<float>::neg_infty();
                upstream_load_score_quarter_x32<3, C>(
                    scratch, score_row_base, n, query_tile);
                upstream_reduce_score_quarter<C>(
                    scratch, quarter_max);
                tile_sum +=
                    hao_direct_transform_pack_shiftless_fp8_quarter<
                        3, C>(
                        p_owned,
                        reader_warp,
                        scratch,
                        quarter_max,
                        scale_log2);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_tail[Stage]);
                }

                hao_direct_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum += tile_sum;
                score_phase ^= 1;
                continue;
            }

            // nvcc cannot reproduce CUTE DSL's role-local R192 assignment in
            // a 512-thread monolithic kernel. Use one zero-spill score
            // fragment for the max scan, then reacquire each quarter before
            // its overlapping P store. Quarter 2 must be consumed first
            // because P starts 64 columns into the score bank.
#if TK_HAO_DIRECT_RETAIN_Q2 == 1
            score_quarter_regs retained2, scratch;
#elif TK_HAO_DIRECT_RETAIN_Q2 == 2
            hao_direct_score_quarter_bf16_regs retained2;
            score_quarter_regs scratch;
#else
            score_quarter_regs scratch;
#endif
#if TK_HAO_DIRECT_RETAIN_Q0
            score_quarter_regs retained0;
#endif
#if TK_HAO_DIRECT_RETAIN_Q1
            score_quarter_regs retained1;
#endif
#if TK_HAO_DIRECT_RETAIN_Q3
            score_quarter_regs retained3;
#endif
            float tile_max =
                base_types::constants<float>::neg_infty();
            upstream_load_score_quarter_x32<0, C>(
#if TK_HAO_DIRECT_RETAIN_Q0
                retained0, score_row_base, n, query_tile);
            upstream_reduce_score_quarter<C>(
                retained0, tile_max);
#else
                scratch, score_row_base, n, query_tile);
            upstream_reduce_score_quarter<C>(
                scratch, tile_max);
#endif
            upstream_load_score_quarter_x32<1, C>(
#if TK_HAO_DIRECT_RETAIN_Q1
                retained1, score_row_base, n, query_tile);
            upstream_reduce_score_quarter<C>(
                retained1, tile_max);
#else
                scratch, score_row_base, n, query_tile);
            upstream_reduce_score_quarter<C>(
                scratch, tile_max);
#endif
            upstream_load_score_quarter_x32<2, C>(
#if TK_HAO_DIRECT_RETAIN_Q2 == 1
                retained2, score_row_base, n, query_tile);
            upstream_reduce_score_quarter<C>(
                retained2, tile_max);
#else
                scratch, score_row_base, n, query_tile);
            upstream_reduce_score_quarter<C>(
                scratch, tile_max);
#if TK_HAO_DIRECT_RETAIN_Q2 == 2
            hao_direct_compress_score_quarter_bf16(
                scratch, retained2);
#endif
#endif
            upstream_load_score_quarter_x32<3, C>(
#if TK_HAO_DIRECT_RETAIN_Q3
                retained3, score_row_base, n, query_tile);
            upstream_reduce_score_quarter<C>(
                retained3, tile_max);
#else
                scratch, score_row_base, n, query_tile);
            upstream_reduce_score_quarter<C>(
                scratch, tile_max);
#endif

            const float old_max = row_max;
            const float candidate_max = fmaxf(row_max, tile_max);
            float correction = 1.0f;
            if (n == 0) {
                row_max = candidate_max;
            } else {
                const float correction_log2 =
                    (old_max - candidate_max) * scale_log2;
                const float candidate_correction =
                    fp4pv_exp2_approx(correction_log2);
                const bool keep_old_max = correction_log2 >= -8.0f;
                row_max = keep_old_max ? old_max : candidate_max;
                correction = keep_old_max
                    ? 1.0f
                    : candidate_correction;
            }
            hao_direct_shared_stats_store(
                correction_owned,
                warpgroup::laneid(),
                correction);
            if (warp::elect_leader()) {
                arrive(softmax_corr_full[Stage]);
            }

#if TK_HAO_DIRECT_RETAIN_Q2 == 0
            upstream_load_score_quarter_x32<2, C>(
                scratch, score_row_base, n, query_tile);
#endif
            if constexpr (C::HAO_DIRECT_TWO_CHUNK_PIPELINE) {
                float2 q2_slice0[4];
                float2 q2_slice1[4];
                float2 q2_slice2[4];
                float2 q2_slice3[4];
                upstream_load_score_sixteenth_tmem<2, 0, C>(
                    q2_slice0, score_row_base, n, query_tile);

#if !TK_HAO_DIRECT_RETAIN_Q0
                upstream_load_score_quarter_x32<0, C>(
                    scratch, score_row_base, n, query_tile);
#endif
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(sfqk_load[OppositeScaleStage]);
                }
                float tile_sum =
                    hao_direct_transform_pack_retained_quarter<0, C>(
                        p_owned,
                        reader_warp,
#if TK_HAO_DIRECT_RETAIN_Q0
                        retained0,
#else
                        scratch,
#endif
                        row_max,
                        scale_log2);

                upstream_load_score_sixteenth_tmem<2, 1, C>(
                    q2_slice1, score_row_base, n, query_tile);
#if !TK_HAO_DIRECT_RETAIN_Q1
                upstream_load_score_quarter_x32<1, C>(
                    scratch, score_row_base, n, query_tile);
#endif
                tile_sum +=
                    hao_direct_transform_pack_retained_quarter<1, C>(
                        p_owned,
                        reader_warp,
#if TK_HAO_DIRECT_RETAIN_Q1
                        retained1,
#else
                        scratch,
#endif
                        row_max,
                        scale_log2);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_first_o_rescaled[Stage]);
                }

                upstream_load_score_sixteenth_tmem<2, 2, C>(
                    q2_slice2, score_row_base, n, query_tile);
                uint32_t q2_word0;
                uint32_t q2_word1;
                uint32_t q2_word2;
                uint32_t q2_word3;
                tile_sum +=
                    hao_direct_transform_pack_fp8_slice<2, 0, C>(
                        q2_slice0,
                        row_max,
                        scale_log2,
                        q2_word0,
                        q2_word1);
                tile_sum +=
                    hao_direct_transform_pack_fp8_slice<2, 1, C>(
                        q2_slice1,
                        row_max,
                        scale_log2,
                        q2_word2,
                        q2_word3);
                hao_direct_store_p_fp8_half_quarter_tmem<2, 0>(
                    p_owned,
                    reader_warp,
                    q2_word0,
                    q2_word1,
                    q2_word2,
                    q2_word3);

                upstream_load_score_sixteenth_tmem<2, 3, C>(
                    q2_slice3, score_row_base, n, query_tile);
#if !TK_HAO_DIRECT_RETAIN_Q3
                if constexpr (C::HAO_DIRECT_EARLY_Q3) {
                    upstream_load_score_quarter_half_x16<
                        3, 0, C, false>(
                            scratch,
                            score_row_base,
                            n,
                            query_tile);
                }
#endif
                tile_sum +=
                    hao_direct_transform_pack_fp8_slice<2, 2, C>(
                        q2_slice2,
                        row_max,
                        scale_log2,
                        q2_word0,
                        q2_word1);
                tile_sum +=
                    hao_direct_transform_pack_fp8_slice<2, 3, C>(
                        q2_slice3,
                        row_max,
                        scale_log2,
                        q2_word2,
                        q2_word3);
                hao_direct_store_p_fp8_half_quarter_tmem<2, 1>(
                    p_owned,
                    reader_warp,
                    q2_word0,
                    q2_word1,
                    q2_word2,
                    q2_word3);
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_middle[Stage]);
                }

#if TK_HAO_DIRECT_RETAIN_Q3
                tile_sum +=
                    hao_direct_transform_pack_retained_quarter<3, C>(
                        p_owned,
                        reader_warp,
                        retained3,
                        row_max,
                        scale_log2);
#else
                if constexpr (C::HAO_DIRECT_EARLY_Q3) {
                    tensor_load_wait();
                    tile_sum +=
                        hao_direct_transform_pack_fp8_slice<3, 0, C>(
                            &scratch.v0,
                            row_max,
                            scale_log2,
                            q2_word0,
                            q2_word1);
                    tile_sum +=
                        hao_direct_transform_pack_fp8_slice<3, 1, C>(
                            &scratch.v4,
                            row_max,
                            scale_log2,
                            q2_word2,
                            q2_word3);
                    upstream_load_score_quarter_half_x16<
                        3, 1, C, false>(
                            scratch,
                            score_row_base,
                            n,
                            query_tile);
                    hao_direct_store_p_fp8_half_quarter_tmem<3, 0>(
                        p_owned,
                        reader_warp,
                        q2_word0,
                        q2_word1,
                        q2_word2,
                        q2_word3);
                    tensor_load_wait();
                    tile_sum +=
                        hao_direct_transform_pack_fp8_slice<3, 2, C>(
                            &scratch.v8,
                            row_max,
                            scale_log2,
                            q2_word0,
                            q2_word1);
                    tile_sum +=
                        hao_direct_transform_pack_fp8_slice<3, 3, C>(
                            &scratch.v12,
                            row_max,
                            scale_log2,
                            q2_word2,
                            q2_word3);
                    hao_direct_store_p_fp8_half_quarter_tmem<3, 1>(
                        p_owned,
                        reader_warp,
                        q2_word0,
                        q2_word1,
                        q2_word2,
                        q2_word3);
                } else {
                    upstream_load_score_quarter_x32<3, C>(
                        scratch, score_row_base, n, query_tile);
                    tile_sum +=
                        hao_direct_transform_pack_retained_quarter<3, C>(
                            p_owned,
                            reader_warp,
                            scratch,
                            row_max,
                        scale_log2);
                }
#endif
                tensor_before_thread_sync();
                if (warp::elect_leader()) {
                    arrive(p_tail[Stage]);
                }

                hao_direct_wait(
                    softmax_corr_empty[Stage],
                    corr_producer_phase);
                corr_producer_phase ^= 1;
                row_sum = row_sum * correction + tile_sum;
                score_phase ^= 1;
                continue;
            }

            float tile_sum = 0.0f;
            if constexpr (C::P_FIRST_CHUNKS == 3) {
#if TK_HAO_DIRECT_RETAIN_Q2 == 2
                tile_sum +=
                    hao_direct_transform_pack_compressed_quarter<2, C>(
                        p_owned,
                        reader_warp,
                        retained2,
                        row_max,
                        scale_log2);
#else
                tile_sum +=
                    hao_direct_transform_pack_retained_quarter<2, C>(
                        p_owned,
                        reader_warp,
#if TK_HAO_DIRECT_RETAIN_Q2 == 1
                        retained2,
#else
                        scratch,
#endif
                        row_max,
                        scale_log2);
#endif
            }
#if !TK_HAO_DIRECT_RETAIN_Q0
            upstream_load_score_quarter_x32<0, C>(
                scratch, score_row_base, n, query_tile);
#endif
            // The opposite stage's Q/K scales overwrite the low 32 columns
            // of this retired score bank.  Publish that lease only after q0
            // is resident in registers; q1/q3 and the packed-P destinations
            // are disjoint from the scale pages.
            tensor_before_thread_sync();
            if (warp::elect_leader()) {
                arrive(sfqk_load[OppositeScaleStage]);
            }
            tile_sum += hao_direct_transform_pack_retained_quarter<0, C>(
                p_owned,
                reader_warp,
#if TK_HAO_DIRECT_RETAIN_Q0
                retained0,
#else
                scratch,
#endif
                row_max,
                scale_log2);
#if !TK_HAO_DIRECT_RETAIN_Q1
            upstream_load_score_quarter_x32<1, C>(
                scratch, score_row_base, n, query_tile);
#endif
            tile_sum += hao_direct_transform_pack_retained_quarter<1, C>(
                p_owned,
                reader_warp,
#if TK_HAO_DIRECT_RETAIN_Q1
                retained1,
#else
                scratch,
#endif
                row_max,
                scale_log2);

#if !TK_HAO_DIRECT_RETAIN_Q3
            if constexpr (C::HAO_DIRECT_EARLY_Q3) {
                upstream_load_score_quarter_x32<3, C, false>(
                    scratch, score_row_base, n, query_tile);
            }
#endif
            tensor_before_thread_sync();
            if (warp::elect_leader()) {
                arrive(p_first_o_rescaled[Stage]);
            }

            if constexpr (C::P_FIRST_CHUNKS == 2) {
#if TK_HAO_DIRECT_RETAIN_Q2 == 2
                tile_sum +=
                    hao_direct_transform_pack_compressed_quarter<2, C>(
                        p_owned,
                        reader_warp,
                        retained2,
                        row_max,
                        scale_log2);
#else
                tile_sum +=
                    hao_direct_transform_pack_retained_quarter<2, C>(
                        p_owned,
                        reader_warp,
#if TK_HAO_DIRECT_RETAIN_Q2 == 1
                        retained2,
#else
                        scratch,
#endif
                        row_max,
                        scale_log2);
#endif
            }

#if TK_HAO_DIRECT_RETAIN_Q3
            tile_sum += hao_direct_transform_pack_retained_quarter<3, C>(
                p_owned,
                reader_warp,
                retained3,
                row_max,
                scale_log2);
#else
            if constexpr (C::HAO_DIRECT_EARLY_Q3) {
                tensor_load_wait();
            } else {
                upstream_load_score_quarter_x32<3, C>(
                    scratch, score_row_base, n, query_tile);
            }
            tile_sum += hao_direct_transform_pack_retained_quarter<3, C>(
                p_owned,
                reader_warp,
                scratch,
                row_max,
                scale_log2);
#endif
            tensor_before_thread_sync();
            if (warp::elect_leader()) {
                arrive(p_tail[Stage]);
            }

            hao_direct_wait(
                softmax_corr_empty[Stage],
                corr_producer_phase);
            corr_producer_phase ^= 1;
            row_sum = row_sum * correction + tile_sum;
            score_phase ^= 1;
        }

        hao_direct_shared_stats_store(
            row_max_owned, warpgroup::laneid(), row_max);
        hao_direct_shared_stats_store(
            row_sum_owned, warpgroup::laneid(), row_sum);
        if (warp::elect_leader()) {
            arrive(softmax_corr_full[Stage]);
        }
    }

    if (warp::elect_leader()) {
        arrive(tmem_dealloc);
    }
}

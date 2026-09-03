// Compile-time-owned dense-E4M3 P reader.  It establishes one N128 maximum,
// then rereads/exp2-packs each score exactly in that common domain.
{
    constexpr int Stage = UPSTREAM_FP8PV_READER_STAGE;
    static_assert(Stage == 0 || Stage == 1);
    const int query_tile = Stage == 0 ? qtile0 : qtile1;
    static_assert(C::UPSTREAM_NATIVE_NONCAUSAL);
    const int reader_iters = C::N_TILES;
    const int reader_warp = warpgroup::warpid();
    const Score score_owned =
        tm.template allocate<Score>(Stage == 0 ? S0 : S1);
    const P p_owned = tm.template allocate<P>(Stage == 0 ? P0 : P1);
    auto &row_max_smem_owned = row_max_smem[Stage];
    auto &row_sum_smem_owned = row_sum_smem[Stage];
    auto &correction_smem_owned = correction_smem[Stage];

    float row_max = base_types::constants<float>::neg_infty();
    float row_sum = 0.0f;

    #pragma unroll 1
    for (int n = 0; n < reader_iters; ++n) {
        wait(score_full[Stage], n & 1);
        tensor_after_thread_sync();
        const uint32_t score_row_base =
            score_owned.addr + (reader_warp * 32 << 16);

        score_quarter_regs keep2, scratch;
        float tile_max = base_types::constants<float>::neg_infty();
        upstream_load_score_quarter_x32<0,C>(scratch,score_row_base,n,query_tile); upstream_reduce_score_quarter<C>(scratch,tile_max);
        upstream_load_score_quarter_x32<1,C>(scratch,score_row_base,n,query_tile); upstream_reduce_score_quarter<C>(scratch,tile_max);
        upstream_load_score_quarter_x32<2,C>(keep2,score_row_base,n,query_tile); upstream_reduce_score_quarter<C>(keep2,tile_max);
        upstream_load_score_quarter_x32<3,C>(scratch,score_row_base,n,query_tile); upstream_reduce_score_quarter<C>(scratch,tile_max);
        const float row_max_old = row_max;
        row_max = fmaxf(row_max, tile_max);
        const float correction = n == 0 ? 1.0f :
            softmax_rescale_factor_strict(
                row_max, row_max_old, scale_log2);
        stats_store(
            correction_smem_owned, warpgroup::laneid(), correction);
        __threadfence_block();
        if (warp::elect_leader()) arrive(corr_ready[Stage]);

        // Dense P occupies score+64..95.  Quarter 2 is therefore acquired in
        // full before any P word is stored.  The remaining source quarters do
        // not alias the P overlay.
        float tile_sum = upstream_transform_pack_retained_quarter<2,C>(p_owned,reader_warp,keep2,row_max,scale_log2);
        upstream_load_score_quarter_x32<0,C>(scratch,score_row_base,n,query_tile); tile_sum += upstream_transform_pack_retained_quarter<0,C>(p_owned,reader_warp,scratch,row_max,scale_log2);
        upstream_load_score_quarter_x32<1,C>(scratch,score_row_base,n,query_tile); tile_sum += upstream_transform_pack_retained_quarter<1,C>(p_owned,reader_warp,scratch,row_max,scale_log2);
        upstream_load_score_quarter_x32<3,C>(scratch,score_row_base,n,query_tile); tile_sum += upstream_transform_pack_retained_quarter<3,C>(p_owned,reader_warp,scratch,row_max,scale_log2);

        // Release both score-read completion and dense-P visibility through
        // the existing four-reader generations.  There is no P-scale proxy.
        tensor_before_thread_sync();
        if (warp::elect_leader()) {
            arrive(score_retired[Stage]);
            arrive(p_ready[Stage]);
        }

        row_sum = row_sum * correction + tile_sum;
        if (n + 1 == reader_iters) {
            stats_store(
                row_max_smem_owned, warpgroup::laneid(), row_max);
            stats_store(
                row_sum_smem_owned, warpgroup::laneid(), row_sum);
            __threadfence_block();
            if (warp::elect_leader()) arrive(stats_ready[Stage]);
        }
    }
}

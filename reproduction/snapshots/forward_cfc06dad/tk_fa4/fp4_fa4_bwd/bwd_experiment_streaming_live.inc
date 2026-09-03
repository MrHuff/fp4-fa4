template <typename C, typename OTile>
__device__ inline void zero_output_tile_streaming(OTile &o_tile) {
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            #pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
                o_tile[{row, col + ii}] = __float2bfloat16(0.0f);
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OTile>
__device__ inline void rescale_output_tile_streaming(OTile &o_tile, float correction) {
    const int row = warpgroup::laneid();
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            #pragma unroll
            for (int ii = 0; ii < 16; ++ii) {
                const float scaled = __bfloat162float(o_tile[{row, col + ii}]) * correction;
                o_tile[{row, col + ii}] = __float2bfloat16(scaled);
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OTile, typename OutputTT>
__device__ inline void accumulate_tt_output_into_tile_streaming(OTile &o_tile, OutputTT &tt_output) {
    const int row = warpgroup::laneid();
    const int subtile = row / 32;
    if (row < C::Mb) {
        #pragma unroll
        for (int col = 0; col < C::Dvo; col += 16) {
            float2 o_reg[8];
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((subtile * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                const float cur0 = __bfloat162float(o_tile[{row, col + ii * 2 + 0}]);
                const float cur1 = __bfloat162float(o_tile[{row, col + ii * 2 + 1}]);
                o_tile[{row, col + ii * 2 + 0}] = __float2bfloat16(cur0 + o_reg[ii].x);
                o_tile[{row, col + ii * 2 + 1}] = __float2bfloat16(cur1 + o_reg[ii].y);
            }
        }
    }
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, typename OutputTT>
__device__ __forceinline__ void rescale_tt_output_inplace_direct_streaming(
    OutputTT &tt_output,
    float correction,
    bool lane_needs_rescale
) {
    if (warpgroup::laneid() >= C::Mb) {
        return;
    }
    const float2 corr_2 = {lane_needs_rescale ? correction : 1.0f,
                           lane_needs_rescale ? correction : 1.0f};
    #pragma unroll
    for (int col = 0; col < C::Dvo; col += 16) {
        float2 o_reg[8];
        const uint32_t addr = tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
              "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
              "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
              "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
            : "r"(addr));
        tensor_load_wait();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
        }
        asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
            :: "f"(o_reg[0].x), "f"(o_reg[0].y), "f"(o_reg[1].x), "f"(o_reg[1].y),
               "f"(o_reg[2].x), "f"(o_reg[2].y), "f"(o_reg[3].x), "f"(o_reg[3].y),
               "f"(o_reg[4].x), "f"(o_reg[4].y), "f"(o_reg[5].x), "f"(o_reg[5].y),
               "f"(o_reg[6].x), "f"(o_reg[6].y), "f"(o_reg[7].x), "f"(o_reg[7].y),
               "r"(addr));
    }
    tensor_store_wait();
}

template <typename C, typename OutputTT>
__device__ inline void merge_dual_tt_output_accum_streaming(
    OutputTT &tt_output_main,
    OutputTT &tt_output_spare,
    float correction,
    bool lane_needs_rescale
) {
    if (warpgroup::laneid() >= C::Mb) {
        warpgroup::sync(warpgroup::groupid() + 1);
        return;
    }
    const float2 corr_2 = {lane_needs_rescale ? correction : 1.0f,
                           lane_needs_rescale ? correction : 1.0f};
    #pragma unroll
    for (int col = 0; col < C::Dvo; col += 16) {
        float2 main_reg[8];
        float2 spare_reg[8];
        const uint32_t main_addr = tt_output_main.addr + ((warpgroup::warpid() * 32) << 16) + col;
        const uint32_t spare_addr = tt_output_spare.addr + ((warpgroup::warpid() * 32) << 16) + col;
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(main_reg[0].x), "=f"(main_reg[0].y), "=f"(main_reg[1].x), "=f"(main_reg[1].y),
              "=f"(main_reg[2].x), "=f"(main_reg[2].y), "=f"(main_reg[3].x), "=f"(main_reg[3].y),
              "=f"(main_reg[4].x), "=f"(main_reg[4].y), "=f"(main_reg[5].x), "=f"(main_reg[5].y),
              "=f"(main_reg[6].x), "=f"(main_reg[6].y), "=f"(main_reg[7].x), "=f"(main_reg[7].y)
            : "r"(main_addr));
        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
            : "=f"(spare_reg[0].x), "=f"(spare_reg[0].y), "=f"(spare_reg[1].x), "=f"(spare_reg[1].y),
              "=f"(spare_reg[2].x), "=f"(spare_reg[2].y), "=f"(spare_reg[3].x), "=f"(spare_reg[3].y),
              "=f"(spare_reg[4].x), "=f"(spare_reg[4].y), "=f"(spare_reg[5].x), "=f"(spare_reg[5].y),
              "=f"(spare_reg[6].x), "=f"(spare_reg[6].y), "=f"(spare_reg[7].x), "=f"(spare_reg[7].y)
            : "r"(spare_addr));
        tensor_load_wait();
        #pragma unroll
        for (int ii = 0; ii < 8; ++ii) {
            main_reg[ii] = __fadd2_rn(__fmul2_rn(main_reg[ii], corr_2), spare_reg[ii]);
        }
        asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
            :: "f"(main_reg[0].x), "f"(main_reg[0].y), "f"(main_reg[1].x), "f"(main_reg[1].y),
               "f"(main_reg[2].x), "f"(main_reg[2].y), "f"(main_reg[3].x), "f"(main_reg[3].y),
               "f"(main_reg[4].x), "f"(main_reg[4].y), "f"(main_reg[5].x), "f"(main_reg[5].y),
               "f"(main_reg[6].x), "f"(main_reg[6].y), "f"(main_reg[7].x), "f"(main_reg[7].y),
               "r"(main_addr));
    }
    tensor_store_wait();
    warpgroup::sync(warpgroup::groupid() + 1);
}

template <typename C, bool ONLINE, int STATIC_CONSUMER_MODE = -1, bool EXTERNAL_COL_LSE = false, bool MXFP4_PV = false>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_streaming_live_fp4pv(
    const __grid_constant__ std::conditional_t<
        MXFP4_PV,
        globals_fp4pv_mxfp4_dv<C, ONLINE>,
        globals_fp4pv<C, ONLINE || !FP4PV_DIAG_USE_DIRECT_ROW_UPDATE>
    > g,
    int consumer_mode,
    int k_iter_begin,
    int k_iter_end
) {
    using G = std::conditional_t<
        MXFP4_PV,
        globals_fp4pv_mxfp4_dv<C, ONLINE>,
        globals_fp4pv<C, ONLINE || !FP4PV_DIAG_USE_DIRECT_ROW_UPDATE>
    >;
    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_q_sc = full_tt_fp8e4m3<C::Q_SC_TMEM_WIDTH>;
    using d_tt_sc16 = full_tt_fp8e4m3<16>;
    using d_tt_k_sc_pre = full_tt_fp8e4m3<16 * C::QK_SCALE_CHUNKS>;
    constexpr bool STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 380 || STATIC_CONSUMER_MODE == 381 ||
         STATIC_CONSUMER_MODE == 382);
    constexpr bool STATIC_DIRECT_SOFTMAX_PV_STAGE_SINGLE_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 381 || STATIC_CONSUMER_MODE == 382);
    constexpr bool STATIC_DIRECT_SOFTMAX_PV_STAGE_SINGLE_WARP_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 382);
    constexpr bool STATIC_DECOUPLED_PV_EARLY_K64_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 395 || STATIC_CONSUMER_MODE == 396);
    constexpr bool STATIC_DECOUPLED_PV_EARLY_K64_SKIP_REMOTE_DEBUG =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 396);
    constexpr bool STATIC_DECOUPLED_PV_INLINE_P_SCALE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 397);
    constexpr bool STATIC_DECOUPLED_PV_DIRECT_PUBLISH_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 398 || STATIC_CONSUMER_MODE == 399 ||
         STATIC_CONSUMER_MODE == 400 || STATIC_CONSUMER_MODE == 401 ||
         STATIC_CONSUMER_MODE == 402 || STATIC_CONSUMER_MODE == 403);
    constexpr bool STATIC_DECOUPLED_PV_DIRECT_PUBLISH_EARLY_READY_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 399 || STATIC_CONSUMER_MODE == 401 ||
         STATIC_CONSUMER_MODE == 403);
    constexpr bool STATIC_DECOUPLED_PV_DIRECT_PUBLISH_DELAY_WAIT_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 400 || STATIC_CONSUMER_MODE == 401);
    constexpr bool STATIC_DECOUPLED_PV_FOLD_REMOTE_READY_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 402 || STATIC_CONSUMER_MODE == 403);
    constexpr bool STATIC_ONLINE_DECOUPLED_PV_ISSUE_REWRITE =
        MXFP4_PV && ONLINE && (!EXTERNAL_COL_LSE) &&
        fp4pv_online_decoupled_pv_issue<C>::value;
    constexpr bool STATIC_DECOUPLED_PV_ISSUE_REWRITE =
        STATIC_ONLINE_DECOUPLED_PV_ISSUE_REWRITE ||
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 390 || STATIC_CONSUMER_MODE == 391 ||
         STATIC_CONSUMER_MODE == 392 || STATIC_CONSUMER_MODE == 393 ||
         STATIC_CONSUMER_MODE == 394 ||
         STATIC_DECOUPLED_PV_INLINE_P_SCALE_REWRITE ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_REWRITE ||
         STATIC_DECOUPLED_PV_EARLY_K64_REWRITE);
    constexpr bool STATIC_DECOUPLED_PV_DIRECT_SCALE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 391 || STATIC_CONSUMER_MODE == 392 ||
         STATIC_CONSUMER_MODE == 394 ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_REWRITE);
    constexpr bool STATIC_ONLINE_MXFP4_REG_TUNE =
        MXFP4_PV && ONLINE && (!EXTERNAL_COL_LSE);
    constexpr bool STATIC_ONLINE_MXFP4_DIRECT_P_READY =
        MXFP4_PV && ONLINE && (!EXTERNAL_COL_LSE) &&
        (C::TOTAL_WGS == 3 || C::TOTAL_WGS == 4);
    constexpr bool STATIC_ONLINE_MXFP4_3WG =
        STATIC_ONLINE_MXFP4_DIRECT_P_READY && C::TOTAL_WGS == 3;
    constexpr bool STATIC_ONLINE_MXFP4_PRODUCER_ASYNC_P_SCALE_COPY =
        STATIC_ONLINE_MXFP4_DIRECT_P_READY && fp4pv_online_async_p_scale_copy<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_DUAL_SCORE_TMEM =
        STATIC_ONLINE_MXFP4_DIRECT_P_READY && fp4pv_online_dual_score_tmem<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM =
        STATIC_ONLINE_MXFP4_DIRECT_P_READY && fp4pv_online_direct_p_scale_tmem<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM =
        STATIC_ONLINE_MXFP4_DIRECT_P_READY && fp4pv_online_fixed_p_scale_tmem<C>::value;
    static_assert(!(STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM && STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM),
        "online MXFP4 direct and fixed P-scale TMEM paths are mutually exclusive");
    constexpr bool STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM =
        STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
        fp4pv_online_split_output_accum<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH =
        STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
        STATIC_ONLINE_MXFP4_DIRECT_P_READY &&
        !STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM;
    constexpr bool STATIC_ONLINE_MXFP4_EARLY_P_READY =
        STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH &&
        fp4pv_online_early_p_ready<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_EARLY_CORR_PUBLISH =
        STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH &&
        !STATIC_ONLINE_MXFP4_EARLY_P_READY &&
        fp4pv_online_early_corr_publish<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_FP32_P_PACK =
        STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
        fp4pv_online_fp32_p_pack<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64 =
        STATIC_ONLINE_MXFP4_EARLY_P_READY &&
        STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
        STATIC_ONLINE_MXFP4_FP32_P_PACK &&
        fp4pv_online_split_p_ready_k64<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG =
        STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64 &&
        fp4pv_online_split_p_quant_2wg<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_STREAM_SCORE_RELOAD =
        (STATIC_ONLINE_MXFP4_EARLY_P_READY ||
         fp4pv_online_stream_score_reload_no_early<C>::value) &&
        STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
        STATIC_ONLINE_MXFP4_FP32_P_PACK &&
        fp4pv_online_stream_score_reload<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_STREAM_SCORE_RELOAD_HALF =
        (STATIC_ONLINE_MXFP4_EARLY_P_READY ||
         fp4pv_online_stream_score_reload_no_early<C>::value) &&
        STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
        STATIC_ONLINE_MXFP4_FP32_P_PACK &&
        fp4pv_online_stream_score_reload_half<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_PRE_P_QUANT_CORR_PUBLISH =
        STATIC_ONLINE_MXFP4_EARLY_P_READY &&
        STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
        STATIC_ONLINE_MXFP4_FP32_P_PACK &&
        fp4pv_online_early_corr_publish<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_P_READY_BEFORE_RESCALE_WAIT =
        STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH &&
        STATIC_ONLINE_MXFP4_DIRECT_P_READY &&
        !STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM &&
        !STATIC_ONLINE_MXFP4_EARLY_P_READY &&
        fp4pv_online_p_ready_before_rescale_wait<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_DEFER_P_REMOTE_UNTIL_PV =
        STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH &&
        STATIC_ONLINE_MXFP4_DIRECT_P_READY &&
        STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
        !STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64 &&
        fp4pv_online_defer_p_remote_until_pv<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_WARP_RESCALE_ARRIVE =
        STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH &&
        STATIC_ONLINE_MXFP4_DIRECT_P_READY &&
        !STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM &&
        !STATIC_ONLINE_MXFP4_EARLY_P_READY &&
        !STATIC_ONLINE_MXFP4_P_READY_BEFORE_RESCALE_WAIT &&
        fp4pv_online_warp_rescale_arrive<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM =
        STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH &&
        STATIC_ONLINE_MXFP4_DIRECT_P_READY &&
        !STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM &&
        (!STATIC_ONLINE_MXFP4_EARLY_P_READY ||
         fp4pv_online_dual_output_accum_early_p_ready<C>::value) &&
        fp4pv_online_dual_output_accum_tmem<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC =
        STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
        fp4pv_online_dual_output_accum_skip_flag_sync<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_FORCE_SPARE =
        STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
        fp4pv_online_dual_output_accum_force_spare<C>::value;
    constexpr bool STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE =
        STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
        fp4pv_online_dual_output_accum_direct_after_rescale<C>::value;
    constexpr int STATIC_ONLINE_MXFP4_CORR_SLOTS =
        STATIC_ONLINE_MXFP4_EARLY_P_READY ? 2 : 1;
    constexpr int STATIC_ONLINE_MXFP4_V_LOAD_WARPS =
        STATIC_ONLINE_MXFP4_3WG ? fp4pv_online_v_load_warps<C>::value : 1;
    static_assert(STATIC_ONLINE_MXFP4_V_LOAD_WARPS >= 1 && STATIC_ONLINE_MXFP4_V_LOAD_WARPS <= 3,
        "online MXFP4 V staging supports one, two, or three producer warps");
    constexpr int STATIC_ONLINE_MXFP4_PRODUCER_REGS = fp4pv_online_producer_regs<C>::value;
    constexpr int STATIC_ONLINE_MXFP4_QUANT_REGS = fp4pv_online_quant_regs<C>::value;
    constexpr int STATIC_ONLINE_MXFP4_OUTPUT_REGS = fp4pv_online_output_regs<C>::value;
    constexpr int STATIC_ONLINE_MXFP4_RELAY_REGS = fp4pv_online_relay_regs<C>::value;
    constexpr bool STATIC_DECOUPLED_PV_STAGE_V_FIRST_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 393 || STATIC_CONSUMER_MODE == 394 ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_EARLY_READY_REWRITE);
    constexpr bool STATIC_DECOUPLED_PV_EARLY_SCALE_READY_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 394 ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_EARLY_READY_REWRITE);
    constexpr bool STATIC_P128_DIRECT_SCALE_REDESIGN =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 271 || STATIC_CONSUMER_MODE == 272);
    constexpr bool STATIC_P32_LIGHT_PUBLISH_READY_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 301 || STATIC_CONSUMER_MODE == 302);
    constexpr bool STATIC_P32_PAYLOAD_FEED_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 291 || STATIC_CONSUMER_MODE == 292 ||
         STATIC_P32_LIGHT_PUBLISH_READY_REWRITE);
    constexpr bool STATIC_P32_PUBLISH_READY_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 295 || STATIC_CONSUMER_MODE == 296 ||
         STATIC_P32_LIGHT_PUBLISH_READY_REWRITE);
    constexpr bool STATIC_P32_SPLIT_SCALE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 297 || STATIC_CONSUMER_MODE == 298 ||
         STATIC_CONSUMER_MODE == 299 || STATIC_CONSUMER_MODE == 300);
    constexpr bool STATIC_P32_SPLIT_SCALE_LEADER_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 299 || STATIC_CONSUMER_MODE == 300);
    constexpr bool STATIC_P32_SPLIT_SCALE_REMOTE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 300);
    constexpr bool STATIC_P128_PRODUCER_FOLD_REMOTE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 320 || STATIC_CONSUMER_MODE == 321);
    constexpr bool STATIC_P128_STAGE_V_FAST_PUBLISH_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 326 || STATIC_CONSUMER_MODE == 327 ||
         STATIC_CONSUMER_MODE == 328 || STATIC_CONSUMER_MODE == 329 ||
         STATIC_CONSUMER_MODE == 330 || STATIC_CONSUMER_MODE == 331 ||
         STATIC_CONSUMER_MODE == 332 || STATIC_CONSUMER_MODE == 333);
    constexpr bool STATIC_P128_FAST_PUBLISH_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 309 || STATIC_CONSUMER_MODE == 310 ||
         STATIC_CONSUMER_MODE == 311 ||
         STATIC_P128_PRODUCER_FOLD_REMOTE_REWRITE ||
         STATIC_P128_STAGE_V_FAST_PUBLISH_REWRITE);
    constexpr bool STATIC_P128_FAST_SKIP_LOCAL_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 310 || STATIC_CONSUMER_MODE == 311 ||
         STATIC_CONSUMER_MODE == 321 || STATIC_CONSUMER_MODE == 327 ||
         STATIC_CONSUMER_MODE == 329 || STATIC_CONSUMER_MODE == 331 ||
         STATIC_CONSUMER_MODE == 333);
    constexpr bool STATIC_P128_FAST_KEEP_LOCAL_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 309 || STATIC_CONSUMER_MODE == 320 ||
         STATIC_CONSUMER_MODE == 326 || STATIC_CONSUMER_MODE == 328 ||
         STATIC_CONSUMER_MODE == 330 || STATIC_CONSUMER_MODE == 332);
    constexpr bool STATIC_P128_FAST_WAIT_ALIAS_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 310);
    constexpr bool STATIC_P128_SMALL_NO_ALIAS_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 312 || STATIC_CONSUMER_MODE == 313);
    constexpr bool STATIC_P128_SMALL_SKIP_LOCAL_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 313);
    constexpr bool STATIC_P128_SPLIT_SCALE_ASYNC_COPY_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 303 || STATIC_CONSUMER_MODE == 304 ||
         STATIC_CONSUMER_MODE == 305 || STATIC_CONSUMER_MODE == 306);
    constexpr bool STATIC_P128_DIRECT_SCALE_EARLY_P_WAIT_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 324 || STATIC_CONSUMER_MODE == 325);
    constexpr bool STATIC_P128_DIRECT_SCALE_FOLD_REMOTE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 318 || STATIC_CONSUMER_MODE == 319 ||
         STATIC_CONSUMER_MODE == 322 || STATIC_CONSUMER_MODE == 323 ||
         STATIC_P128_DIRECT_SCALE_EARLY_P_WAIT_REWRITE);
    constexpr bool STATIC_P128_DIRECT_SCALE_FOLD_REMOTE_SKIP_LOCAL_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 319 || STATIC_CONSUMER_MODE == 323 ||
         STATIC_CONSUMER_MODE == 325);
    constexpr bool STATIC_P128_DIRECT_SCALE_TRIPLE_SLOT_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 322 || STATIC_CONSUMER_MODE == 323 ||
         STATIC_CONSUMER_MODE == 324 || STATIC_CONSUMER_MODE == 325);
    constexpr bool STATIC_DIRECT_P_SCALE_FROM_CROSSBAR =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 56 ||
         STATIC_P128_DIRECT_SCALE_FOLD_REMOTE_REWRITE);
    constexpr bool STATIC_DIRECT_P_SCALE_FROM_STAGE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 62 ||
         STATIC_CONSUMER_MODE == 346 || STATIC_CONSUMER_MODE == 347);
    constexpr bool STATIC_EARLY_DIRECT_P_SCALE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 70 || STATIC_CONSUMER_MODE == 76 ||
         STATIC_CONSUMER_MODE == 77 || STATIC_CONSUMER_MODE == 78 ||
         STATIC_CONSUMER_MODE == 80 || STATIC_CONSUMER_MODE == 81 ||
         STATIC_CONSUMER_MODE == 82 || STATIC_CONSUMER_MODE == 85 ||
         STATIC_CONSUMER_MODE == 86 || STATIC_CONSUMER_MODE == 91 ||
         STATIC_CONSUMER_MODE == 92 || STATIC_CONSUMER_MODE == 93 ||
         STATIC_CONSUMER_MODE == 126 || STATIC_CONSUMER_MODE == 127 ||
         STATIC_CONSUMER_MODE == 174 || STATIC_CONSUMER_MODE == 184 ||
         STATIC_CONSUMER_MODE == 185 || STATIC_CONSUMER_MODE == 186 ||
         STATIC_CONSUMER_MODE == 187 || STATIC_CONSUMER_MODE == 188 ||
         STATIC_CONSUMER_MODE == 189 || STATIC_CONSUMER_MODE == 190 ||
         STATIC_CONSUMER_MODE == 191 || STATIC_CONSUMER_MODE == 192 ||
         STATIC_CONSUMER_MODE == 193 || STATIC_CONSUMER_MODE == 194 ||
         STATIC_CONSUMER_MODE == 195 || STATIC_CONSUMER_MODE == 196 ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE ||
         STATIC_DECOUPLED_PV_DIRECT_SCALE_REWRITE);
    constexpr bool STATIC_EARLY_ASYNC_P_SCALE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 71);
    constexpr bool STATIC_EARLY_LOCAL_P_SCALE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 72 || STATIC_CONSUMER_MODE == 75);
    constexpr bool STATIC_EARLY_P_SCALE_TMEM =
        STATIC_EARLY_DIRECT_P_SCALE_TMEM || STATIC_EARLY_ASYNC_P_SCALE_TMEM ||
        STATIC_EARLY_LOCAL_P_SCALE_TMEM;
    constexpr bool STATIC_DIRECT_P_SCALE_TMEM_STORE =
        STATIC_DIRECT_P_SCALE_FROM_CROSSBAR || STATIC_DIRECT_P_SCALE_FROM_STAGE;
    constexpr bool STATIC_ANY_DIRECT_P_SCALE_TMEM_STORE =
        STATIC_DIRECT_P_SCALE_TMEM_STORE || STATIC_EARLY_DIRECT_P_SCALE_TMEM ||
        STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM;
    constexpr bool STATIC_DELAY_DIRECT_P_SCALE_TMEM_WAIT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 126 || STATIC_CONSUMER_MODE == 127 ||
         STATIC_CONSUMER_MODE == 272 ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_DELAY_WAIT_REWRITE);
    constexpr bool STATIC_EARLY_PV_K64_LAYOUT_SWEEP =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 251 || STATIC_CONSUMER_MODE == 252);
    constexpr bool STATIC_EARLY_PV_K64_MERGED_SCALE_WAIT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 273 || STATIC_CONSUMER_MODE == 274 ||
         STATIC_CONSUMER_MODE == 275);
    constexpr bool STATIC_EARLY_PV_K64_EAGER_PAYLOAD_REMOTE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 253 || STATIC_CONSUMER_MODE == 274 ||
         STATIC_CONSUMER_MODE == 275 || STATIC_CONSUMER_MODE == 290);
    constexpr bool STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT =
        STATIC_EARLY_PV_K64_LAYOUT_SWEEP || STATIC_EARLY_PV_K64_EAGER_PAYLOAD_REMOTE ||
        STATIC_EARLY_PV_K64_MERGED_SCALE_WAIT;
    constexpr bool STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 316 || STATIC_CONSUMER_MODE == 317 ||
         STATIC_CONSUMER_MODE == 370 || STATIC_CONSUMER_MODE == 371);
    constexpr bool STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE_SKIP_LOCAL =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 317 || STATIC_CONSUMER_MODE == 371);
    constexpr bool STATIC_EARLY_PV_K64_FULL_PAYLOAD_REMOTE_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 370 || STATIC_CONSUMER_MODE == 371);
    constexpr bool STATIC_EARLY_PV_K64_PRODUCER_FOLDS_PAYLOAD_REMOTE =
        STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE &&
        !STATIC_EARLY_PV_K64_FULL_PAYLOAD_REMOTE_REWRITE;
    constexpr bool STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 276);
    constexpr bool STATIC_P128_PUBLISH_REDUCE_VARIANT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        ((STATIC_CONSUMER_MODE >= 277 && STATIC_CONSUMER_MODE <= 280) ||
         STATIC_CONSUMER_MODE == 282 ||
         STATIC_CONSUMER_MODE == 328 || STATIC_CONSUMER_MODE == 329 ||
         STATIC_CONSUMER_MODE == 330 || STATIC_CONSUMER_MODE == 331 ||
         STATIC_CONSUMER_MODE == 332 || STATIC_CONSUMER_MODE == 333);
    constexpr bool STATIC_P128_SKIP_LOCAL_PUBLISH_VARIANT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 277 || STATIC_CONSUMER_MODE == 278 ||
         STATIC_CONSUMER_MODE == 279);
    constexpr bool STATIC_P128_SKIP_PRE_PACK_VARIANT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 277 || STATIC_CONSUMER_MODE == 279 ||
         STATIC_CONSUMER_MODE == 280);
    constexpr bool STATIC_P128_SCALAR_PAYLOAD_VARIANT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 278 || STATIC_CONSUMER_MODE == 279);
    constexpr bool STATIC_P128_SINGLE_LOCAL_PRE_PACK_VARIANT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 280);
    constexpr bool STATIC_P128_ASYNC_PAYLOAD_FAST_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 282 ||
         STATIC_CONSUMER_MODE == 328 || STATIC_CONSUMER_MODE == 329 ||
         STATIC_CONSUMER_MODE == 330 || STATIC_CONSUMER_MODE == 331 ||
         STATIC_CONSUMER_MODE == 332 || STATIC_CONSUMER_MODE == 333);
    constexpr bool STATIC_FUSED_QUANT_ASYNC_SCALE_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 348 || STATIC_CONSUMER_MODE == 349);
    constexpr bool STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 350 || STATIC_CONSUMER_MODE == 351);
    constexpr bool STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_NOSYNC_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 352 || STATIC_CONSUMER_MODE == 353);
    constexpr bool STATIC_FUSED_QUANT_ANY_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE =
        STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE ||
        STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_NOSYNC_SKIP_PREPACK_REWRITE;
    constexpr bool STATIC_FUSED_PRODUCER_EARLY_SCALE_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 354 || STATIC_CONSUMER_MODE == 355 ||
         STATIC_CONSUMER_MODE == 358 || STATIC_CONSUMER_MODE == 359 ||
         STATIC_CONSUMER_MODE == 362 || STATIC_CONSUMER_MODE == 363 ||
         STATIC_CONSUMER_MODE == 366 || STATIC_CONSUMER_MODE == 367);
    constexpr bool STATIC_FUSED_PRODUCER_EARLY_SCALE_PROXY_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 356 || STATIC_CONSUMER_MODE == 357 ||
         STATIC_CONSUMER_MODE == 360 || STATIC_CONSUMER_MODE == 361 ||
         STATIC_CONSUMER_MODE == 364 || STATIC_CONSUMER_MODE == 365 ||
         STATIC_CONSUMER_MODE == 368 || STATIC_CONSUMER_MODE == 369);
    constexpr bool STATIC_FUSED_PRODUCER_ANY_EARLY_SCALE_SKIP_PREPACK_REWRITE =
        STATIC_FUSED_PRODUCER_EARLY_SCALE_SKIP_PREPACK_REWRITE ||
        STATIC_FUSED_PRODUCER_EARLY_SCALE_PROXY_SKIP_PREPACK_REWRITE;
    constexpr bool STATIC_FUSED_PRODUCER_FOLD_REMOTE_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 362 || STATIC_CONSUMER_MODE == 363 ||
         STATIC_CONSUMER_MODE == 364 || STATIC_CONSUMER_MODE == 365);
    constexpr bool STATIC_FUSED_PRODUCER_PREFETCH_REMOTE_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 366 || STATIC_CONSUMER_MODE == 367 ||
         STATIC_CONSUMER_MODE == 368 || STATIC_CONSUMER_MODE == 369);
    constexpr bool STATIC_FUSED_PAYLOAD_READY_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 344 || STATIC_CONSUMER_MODE == 345 ||
         STATIC_CONSUMER_MODE == 346 || STATIC_CONSUMER_MODE == 347 ||
         STATIC_FUSED_QUANT_ASYNC_SCALE_SKIP_PREPACK_REWRITE);
    constexpr bool STATIC_FUSED_DIRECT_SCALE_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 346 || STATIC_CONSUMER_MODE == 347);
    constexpr bool STATIC_FUSED_ASYNC_LIGHT_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 336 || STATIC_CONSUMER_MODE == 344);
    constexpr bool STATIC_FUSED_ASYNC_SKIP_LOCAL_SKIP_PREPACK_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 337 || STATIC_CONSUMER_MODE == 345);
    constexpr bool STATIC_FUSED_ASYNC_SKIP_PREPACK_REWRITE =
        STATIC_FUSED_ASYNC_LIGHT_SKIP_PREPACK_REWRITE ||
        STATIC_FUSED_ASYNC_SKIP_LOCAL_SKIP_PREPACK_REWRITE;
    constexpr bool STATIC_FUSED_LIGHT_SKIP_PREPACK_REWRITE =
        STATIC_FUSED_ASYNC_LIGHT_SKIP_PREPACK_REWRITE ||
        (STATIC_FUSED_DIRECT_SCALE_SKIP_PREPACK_REWRITE && STATIC_CONSUMER_MODE == 346) ||
        (STATIC_FUSED_QUANT_ASYNC_SCALE_SKIP_PREPACK_REWRITE && STATIC_CONSUMER_MODE == 348) ||
        (STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE && STATIC_CONSUMER_MODE == 350) ||
        (STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_NOSYNC_SKIP_PREPACK_REWRITE && STATIC_CONSUMER_MODE == 352) ||
        (STATIC_FUSED_PRODUCER_EARLY_SCALE_SKIP_PREPACK_REWRITE &&
         (STATIC_CONSUMER_MODE == 354 || STATIC_CONSUMER_MODE == 358 ||
          STATIC_CONSUMER_MODE == 362 || STATIC_CONSUMER_MODE == 366)) ||
        (STATIC_FUSED_PRODUCER_EARLY_SCALE_PROXY_SKIP_PREPACK_REWRITE &&
         (STATIC_CONSUMER_MODE == 356 || STATIC_CONSUMER_MODE == 360 ||
          STATIC_CONSUMER_MODE == 364 || STATIC_CONSUMER_MODE == 368));
    constexpr bool STATIC_FUSED_SKIP_LOCAL_SKIP_PREPACK_REWRITE =
        STATIC_FUSED_ASYNC_SKIP_LOCAL_SKIP_PREPACK_REWRITE ||
        (STATIC_FUSED_DIRECT_SCALE_SKIP_PREPACK_REWRITE && STATIC_CONSUMER_MODE == 347) ||
        (STATIC_FUSED_QUANT_ASYNC_SCALE_SKIP_PREPACK_REWRITE && STATIC_CONSUMER_MODE == 349) ||
        (STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE && STATIC_CONSUMER_MODE == 351) ||
        (STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_NOSYNC_SKIP_PREPACK_REWRITE && STATIC_CONSUMER_MODE == 353) ||
        (STATIC_FUSED_PRODUCER_EARLY_SCALE_SKIP_PREPACK_REWRITE &&
         (STATIC_CONSUMER_MODE == 355 || STATIC_CONSUMER_MODE == 359 ||
          STATIC_CONSUMER_MODE == 363 || STATIC_CONSUMER_MODE == 367)) ||
        (STATIC_FUSED_PRODUCER_EARLY_SCALE_PROXY_SKIP_PREPACK_REWRITE &&
         (STATIC_CONSUMER_MODE == 357 || STATIC_CONSUMER_MODE == 361 ||
          STATIC_CONSUMER_MODE == 365 || STATIC_CONSUMER_MODE == 369));
    constexpr bool STATIC_FUSED_SKIP_PREPACK_REWRITE =
        STATIC_FUSED_LIGHT_SKIP_PREPACK_REWRITE || STATIC_FUSED_SKIP_LOCAL_SKIP_PREPACK_REWRITE;
    constexpr bool STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 283 || STATIC_CONSUMER_MODE == 284 ||
         STATIC_CONSUMER_MODE == 285);
    constexpr bool STATIC_P128_WARP_ASYNC_PAYLOAD_SCALAR =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 283);
    constexpr bool STATIC_P128_WARP_ASYNC_PAYLOAD_HALF_STRIPED =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 284);
    constexpr bool STATIC_P128_WARP_ASYNC_PAYLOAD_V2 =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 285);
    constexpr bool STATIC_P128_SPLIT_SCALE_READY_VARIANT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 286 || STATIC_CONSUMER_MODE == 287 ||
         STATIC_CONSUMER_MODE == 288 || STATIC_CONSUMER_MODE == 289 ||
         STATIC_P128_SPLIT_SCALE_ASYNC_COPY_REWRITE ||
         STATIC_FUSED_PRODUCER_ANY_EARLY_SCALE_SKIP_PREPACK_REWRITE ||
         STATIC_P32_SPLIT_SCALE_REWRITE);
    constexpr bool STATIC_P128_SPLIT_SCALE_LEADER_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 288 || STATIC_CONSUMER_MODE == 289 ||
         STATIC_CONSUMER_MODE == 305 || STATIC_CONSUMER_MODE == 306 ||
         STATIC_P32_SPLIT_SCALE_LEADER_REWRITE);
    constexpr bool STATIC_P128_SPLIT_SCALE_REMOTE_READY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 287 || STATIC_CONSUMER_MODE == 289 ||
         STATIC_P128_SPLIT_SCALE_ASYNC_COPY_REWRITE ||
         STATIC_FUSED_PRODUCER_ANY_EARLY_SCALE_SKIP_PREPACK_REWRITE ||
         STATIC_P32_SPLIT_SCALE_REMOTE_REWRITE);
    constexpr bool STATIC_FULL_K128_P64_SINGLE_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 254 || STATIC_CONSUMER_MODE == 255);
    constexpr bool STATIC_FULL_K128_P64_LIGHT_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 255);
    constexpr bool STATIC_FULL_K128_P32_SINGLE_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 256 || STATIC_CONSUMER_MODE == 257 ||
         STATIC_CONSUMER_MODE == 258 || STATIC_CONSUMER_MODE == 259);
    constexpr bool STATIC_FULL_K128_P32_LIGHT_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 257);
    constexpr bool STATIC_FULL_K128_P32_WARP_PACK_READY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 258 || STATIC_CONSUMER_MODE == 259);
    constexpr bool STATIC_FULL_K128_P32_ASYNC_PAYLOAD_CLUSTER_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 259);
    constexpr bool STATIC_FULL_K128_P32_BASE_WARP_PACK_READY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 260 || STATIC_CONSUMER_MODE == 261 ||
         STATIC_CONSUMER_MODE == 307 ||
         STATIC_P32_PAYLOAD_FEED_REWRITE || STATIC_P32_PUBLISH_READY_REWRITE);
    constexpr bool STATIC_FULL_K128_P32_BASE_ASYNC_PAYLOAD_CLUSTER_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 261 || STATIC_CONSUMER_MODE == 281 ||
         STATIC_CONSUMER_MODE == 307 ||
         STATIC_P32_PAYLOAD_FEED_REWRITE || STATIC_P32_PUBLISH_READY_REWRITE);
    constexpr bool STATIC_FULL_K128_P32_BASE_LIGHT_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 268 || STATIC_CONSUMER_MODE == 281 ||
         STATIC_CONSUMER_MODE == 307 ||
         STATIC_P32_PAYLOAD_FEED_REWRITE);
    constexpr bool STATIC_FULL_K128_P32_BASE_SINGLE_WG_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 268 || STATIC_CONSUMER_MODE == 281 ||
         STATIC_CONSUMER_MODE == 307 ||
         STATIC_P32_PAYLOAD_FEED_REWRITE);
    constexpr bool STATIC_FULL_K128_P32_BASE_VARIANT =
        STATIC_FULL_K128_P32_BASE_WARP_PACK_READY ||
        STATIC_FULL_K128_P32_BASE_SINGLE_WG_LOCAL_PUBLISH;
    constexpr bool STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT =
        STATIC_FULL_K128_P64_SINGLE_LOCAL_PUBLISH ||
        STATIC_FULL_K128_P32_SINGLE_LOCAL_PUBLISH;
    constexpr bool STATIC_FULL_K128_LIGHT_LOCAL_PUBLISH_VARIANT =
        STATIC_FULL_K128_P64_LIGHT_LOCAL_PUBLISH ||
        STATIC_FULL_K128_P32_LIGHT_LOCAL_PUBLISH;
    constexpr bool STATIC_EARLY_PV_K64_HALF_READY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 231 || STATIC_CONSUMER_MODE == 232 ||
         STATIC_CONSUMER_MODE == 233 || STATIC_CONSUMER_MODE == 234 ||
         STATIC_CONSUMER_MODE == 235 || STATIC_CONSUMER_MODE == 236 ||
         STATIC_CONSUMER_MODE == 237 || STATIC_CONSUMER_MODE == 238 ||
         STATIC_CONSUMER_MODE == 239 || STATIC_CONSUMER_MODE == 240 ||
         STATIC_CONSUMER_MODE == 241 || STATIC_CONSUMER_MODE == 242 ||
         STATIC_CONSUMER_MODE == 243 || STATIC_CONSUMER_MODE == 244 ||
         STATIC_CONSUMER_MODE == 245 || STATIC_CONSUMER_MODE == 246 ||
         STATIC_CONSUMER_MODE == 247 || STATIC_CONSUMER_MODE == 248 ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DECOUPLED_PV_EARLY_K64_REWRITE);
    constexpr bool STATIC_EARLY_PV_K64_CROSSBAR_SCALE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 232);
    constexpr bool STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        ((STATIC_CONSUMER_MODE >= 238 && STATIC_CONSUMER_MODE <= 243) ||
         STATIC_CONSUMER_MODE == 246 || STATIC_CONSUMER_MODE == 248);
    constexpr bool STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE_SWIZZLED =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 242 || STATIC_CONSUMER_MODE == 243 ||
         STATIC_CONSUMER_MODE == 248);
    constexpr bool STATIC_EARLY_PV_K64_HALF_SCALE_CROSSBAR =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 244);
    constexpr bool STATIC_EARLY_PV_K64_BYTE_SCALE_CROSSBAR =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 245 || STATIC_CONSUMER_MODE == 247 ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT);
    constexpr bool STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 233 || STATIC_CONSUMER_MODE == 235 ||
         STATIC_CONSUMER_MODE == 236 || STATIC_CONSUMER_MODE == 237 ||
         STATIC_CONSUMER_MODE == 244 || STATIC_CONSUMER_MODE == 245 ||
         STATIC_CONSUMER_MODE == 247 ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE);
    constexpr bool STATIC_EARLY_PV_K64_DIRECT_SCALE_INLINE_REUSE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 234);
    constexpr bool STATIC_EARLY_PV_K64_DIRECT_SCALE_FOLDED_REMOTE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 235 || STATIC_CONSUMER_MODE == 237 ||
         STATIC_CONSUMER_MODE == 239 || STATIC_CONSUMER_MODE == 240 ||
         STATIC_CONSUMER_MODE == 241 || STATIC_CONSUMER_MODE == 242 ||
         STATIC_CONSUMER_MODE == 243 || STATIC_CONSUMER_MODE == 244 ||
         STATIC_CONSUMER_MODE == 245 || STATIC_CONSUMER_MODE == 246 ||
         STATIC_CONSUMER_MODE == 247 || STATIC_CONSUMER_MODE == 248 ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT);
    constexpr bool STATIC_EARLY_PV_K64_CROSSBAR_DIRECT_SCALE =
        (STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE &&
         !STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE) ||
        STATIC_EARLY_PV_K64_DIRECT_SCALE_INLINE_REUSE;
    constexpr bool STATIC_EARLY_PV_K64_DIRECT_SCALE =
        STATIC_EARLY_PV_K64_CROSSBAR_DIRECT_SCALE ||
        STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE;
    constexpr bool STATIC_EARLY_PV_K64_DIRECT_SCALE_REMOTE_SYNC =
        STATIC_EARLY_PV_K64_DIRECT_SCALE &&
        (!STATIC_EARLY_PV_K64_DIRECT_SCALE_FOLDED_REMOTE);
    constexpr bool STATIC_EARLY_PV_K64_ANY_CROSSBAR_SCALE =
        STATIC_EARLY_PV_K64_CROSSBAR_SCALE || STATIC_EARLY_PV_K64_CROSSBAR_DIRECT_SCALE;
    constexpr bool STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 373);
    constexpr bool STATIC_TRIPLE_SCORE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 90 || STATIC_CONSUMER_MODE == 91 ||
          STATIC_CONSUMER_MODE == 92 || STATIC_CONSUMER_MODE == 93 ||
          STATIC_CONSUMER_MODE == 94 || STATIC_CONSUMER_MODE == 95 ||
          STATIC_CONSUMER_MODE == 96 || STATIC_CONSUMER_MODE == 97 ||
          STATIC_CONSUMER_MODE == 98 || STATIC_CONSUMER_MODE == 99 ||
          STATIC_CONSUMER_MODE == 100 || STATIC_CONSUMER_MODE == 101 ||
          STATIC_CONSUMER_MODE == 102 || STATIC_CONSUMER_MODE == 105 ||
          STATIC_CONSUMER_MODE == 107 || STATIC_CONSUMER_MODE == 109 ||
          STATIC_CONSUMER_MODE == 111 || STATIC_CONSUMER_MODE == 127 ||
          STATIC_CONSUMER_MODE == 152 || STATIC_CONSUMER_MODE == 153 ||
          STATIC_CONSUMER_MODE == 154 || STATIC_CONSUMER_MODE == 155 ||
          STATIC_CONSUMER_MODE == 157 || STATIC_CONSUMER_MODE == 158 ||
          STATIC_CONSUMER_MODE == 159 || STATIC_CONSUMER_MODE == 160 ||
          STATIC_CONSUMER_MODE == 161 || STATIC_CONSUMER_MODE == 162 ||
          STATIC_CONSUMER_MODE == 164 || STATIC_CONSUMER_MODE == 165 ||
          STATIC_CONSUMER_MODE == 169 || STATIC_CONSUMER_MODE == 171 ||
          STATIC_CONSUMER_MODE == 172 || STATIC_CONSUMER_MODE == 173 ||
          STATIC_CONSUMER_MODE == 198 || STATIC_CONSUMER_MODE == 199 ||
          STATIC_CONSUMER_MODE == 200 || STATIC_CONSUMER_MODE == 201 ||
          STATIC_CONSUMER_MODE == 202 || STATIC_CONSUMER_MODE == 203 ||
          STATIC_CONSUMER_MODE == 204 || STATIC_CONSUMER_MODE == 205 ||
          STATIC_CONSUMER_MODE == 206 || STATIC_CONSUMER_MODE == 207 ||
          STATIC_CONSUMER_MODE == 208 || STATIC_CONSUMER_MODE == 209 ||
          STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
          STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
          STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
          STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
          STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
          STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
          STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
          STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
          STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
          STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
          STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
          STATIC_CONSUMER_MODE == 225 ||
          STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
          STATIC_CONSUMER_MODE == 228 ||
          STATIC_P128_ASYNC_PAYLOAD_FAST_PUBLISH ||
          STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
          STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
          STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
          STATIC_P128_PUBLISH_REDUCE_VARIANT ||
          STATIC_FULL_K128_P32_BASE_VARIANT ||
          (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
          STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
          STATIC_DECOUPLED_PV_EARLY_K64_REWRITE ||
          STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE);
    constexpr bool STATIC_ALIAS_SCALE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 91 || STATIC_CONSUMER_MODE == 92 ||
         STATIC_CONSUMER_MODE == 93 || STATIC_CONSUMER_MODE == 94 ||
         STATIC_CONSUMER_MODE == 95 || STATIC_CONSUMER_MODE == 96 ||
         STATIC_CONSUMER_MODE == 97 || STATIC_CONSUMER_MODE == 98 ||
         STATIC_CONSUMER_MODE == 99 || STATIC_CONSUMER_MODE == 100 ||
         STATIC_CONSUMER_MODE == 101 || STATIC_CONSUMER_MODE == 102 ||
         STATIC_CONSUMER_MODE == 105 || STATIC_CONSUMER_MODE == 107 ||
         STATIC_CONSUMER_MODE == 109 || STATIC_CONSUMER_MODE == 111 ||
         STATIC_CONSUMER_MODE == 112 || STATIC_CONSUMER_MODE == 123 ||
         STATIC_CONSUMER_MODE == 127 || STATIC_CONSUMER_MODE == 152 ||
         STATIC_CONSUMER_MODE == 153 || STATIC_CONSUMER_MODE == 154 ||
         STATIC_CONSUMER_MODE == 155 || STATIC_CONSUMER_MODE == 157 ||
         STATIC_CONSUMER_MODE == 158 || STATIC_CONSUMER_MODE == 159 ||
         STATIC_CONSUMER_MODE == 160 || STATIC_CONSUMER_MODE == 161 ||
         STATIC_CONSUMER_MODE == 162 || STATIC_CONSUMER_MODE == 164 ||
         STATIC_CONSUMER_MODE == 165 || STATIC_CONSUMER_MODE == 169 ||
         STATIC_CONSUMER_MODE == 171 || STATIC_CONSUMER_MODE == 172 ||
         STATIC_CONSUMER_MODE == 173 || STATIC_CONSUMER_MODE == 198 ||
         STATIC_CONSUMER_MODE == 199 || STATIC_CONSUMER_MODE == 200 ||
         STATIC_CONSUMER_MODE == 201 || STATIC_CONSUMER_MODE == 202 ||
         STATIC_CONSUMER_MODE == 203 || STATIC_CONSUMER_MODE == 204 ||
         STATIC_CONSUMER_MODE == 205 || STATIC_CONSUMER_MODE == 206 ||
         STATIC_CONSUMER_MODE == 207 || STATIC_CONSUMER_MODE == 208 ||
         STATIC_CONSUMER_MODE == 209 || STATIC_CONSUMER_MODE == 210 ||
         STATIC_CONSUMER_MODE == 211 || STATIC_CONSUMER_MODE == 212 ||
         STATIC_CONSUMER_MODE == 213 || STATIC_CONSUMER_MODE == 214 ||
         STATIC_CONSUMER_MODE == 215 || STATIC_CONSUMER_MODE == 216 ||
         STATIC_CONSUMER_MODE == 217 || STATIC_CONSUMER_MODE == 218 ||
         STATIC_CONSUMER_MODE == 219 || STATIC_CONSUMER_MODE == 220 ||
         STATIC_CONSUMER_MODE == 221 || STATIC_CONSUMER_MODE == 222 ||
         STATIC_CONSUMER_MODE == 223 || STATIC_CONSUMER_MODE == 224 ||
         STATIC_CONSUMER_MODE == 269 || STATIC_CONSUMER_MODE == 372 ||
         STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 ||
         STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 225 || STATIC_CONSUMER_MODE == 226 ||
         STATIC_CONSUMER_MODE == 227 || STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_P128_ASYNC_PAYLOAD_FAST_PUBLISH ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_DECOUPLED_PV_EARLY_K64_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE);
    constexpr bool STATIC_DUAL_SCORE_TMEM =
        STATIC_ONLINE_MXFP4_DUAL_SCORE_TMEM ||
        (MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 52 || STATIC_CONSUMER_MODE == 55 ||
         STATIC_CONSUMER_MODE == 66 || STATIC_CONSUMER_MODE == 68 ||
         STATIC_CONSUMER_MODE == 103 || STATIC_CONSUMER_MODE == 104 ||
         STATIC_CONSUMER_MODE == 106 || STATIC_CONSUMER_MODE == 108 ||
         STATIC_CONSUMER_MODE == 110 || STATIC_CONSUMER_MODE == 112 ||
         STATIC_CONSUMER_MODE == 115 || STATIC_CONSUMER_MODE == 116 ||
         STATIC_CONSUMER_MODE == 117 || STATIC_CONSUMER_MODE == 118 ||
         STATIC_CONSUMER_MODE == 123 || STATIC_CONSUMER_MODE == 128 ||
         STATIC_CONSUMER_MODE == 129 || STATIC_CONSUMER_MODE == 130 ||
         STATIC_CONSUMER_MODE == 132 || STATIC_CONSUMER_MODE == 133 ||
         STATIC_CONSUMER_MODE == 134 || STATIC_CONSUMER_MODE == 135 ||
         STATIC_CONSUMER_MODE == 136 || STATIC_CONSUMER_MODE == 137 ||
         STATIC_CONSUMER_MODE == 138 || STATIC_CONSUMER_MODE == 139 ||
         STATIC_CONSUMER_MODE == 140 || STATIC_CONSUMER_MODE == 141 ||
         STATIC_CONSUMER_MODE == 142 || STATIC_CONSUMER_MODE == 143 ||
         STATIC_CONSUMER_MODE == 144 || STATIC_CONSUMER_MODE == 145 ||
         STATIC_CONSUMER_MODE == 146 || STATIC_CONSUMER_MODE == 147 ||
         STATIC_CONSUMER_MODE == 148 || STATIC_CONSUMER_MODE == 149 ||
         STATIC_CONSUMER_MODE == 150 || STATIC_CONSUMER_MODE == 151 ||
         STATIC_CONSUMER_MODE == 156 || STATIC_CONSUMER_MODE == 163 ||
         STATIC_CONSUMER_MODE == 168 || STATIC_CONSUMER_MODE == 170 ||
         STATIC_CONSUMER_MODE == 174 || STATIC_CONSUMER_MODE == 176 ||
         STATIC_CONSUMER_MODE == 180 || STATIC_CONSUMER_MODE == 181 ||
         STATIC_CONSUMER_MODE == 182 || STATIC_CONSUMER_MODE == 183 ||
         STATIC_CONSUMER_MODE == 184 || STATIC_CONSUMER_MODE == 185 ||
         STATIC_CONSUMER_MODE == 186 || STATIC_CONSUMER_MODE == 187 ||
         STATIC_CONSUMER_MODE == 188 || STATIC_CONSUMER_MODE == 189 ||
         STATIC_CONSUMER_MODE == 190 || STATIC_CONSUMER_MODE == 191 ||
         STATIC_CONSUMER_MODE == 192 || STATIC_CONSUMER_MODE == 193 ||
         STATIC_CONSUMER_MODE == 194 || STATIC_CONSUMER_MODE == 195 ||
         STATIC_CONSUMER_MODE == 196 || STATIC_CONSUMER_MODE == 197 ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_CONSUMER_MODE == 63 ||
         STATIC_CONSUMER_MODE == 64 ||
         STATIC_CONSUMER_MODE == 65 ||
         STATIC_EARLY_ASYNC_P_SCALE_TMEM ||
         STATIC_EARLY_LOCAL_P_SCALE_TMEM ||
         STATIC_ANY_DIRECT_P_SCALE_TMEM_STORE ||
         STATIC_DECOUPLED_PV_ISSUE_REWRITE));
    constexpr bool STATIC_COMPACT_Q_SCALE_TMEM =
        STATIC_DUAL_SCORE_TMEM || STATIC_TRIPLE_SCORE_TMEM ||
        STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM;
    using d_tt_q_sc_active = std::conditional_t<STATIC_COMPACT_Q_SCALE_TMEM, d_tt_sc16, d_tt_q_sc>;
    constexpr bool STATIC_MXFP4_K256 =
        MXFP4_PV && (STATIC_CONSUMER_MODE == 6 || STATIC_CONSUMER_MODE == 178);
    constexpr int MXFP4_PV_MMA_PER_TILE = STATIC_MXFP4_K256 ? 2 : 1;
    using d_tt_p_sc = std::conditional_t<MXFP4_PV, full_tt_fp8e8m0<16 * MXFP4_PV_MMA_PER_TILE>, full_tt_fp8e4m3<C::P_SC_TMEM_WIDTH>>;
    using d_tt_v_sc = std::conditional_t<MXFP4_PV, full_tt_fp8e8m0<32 * MXFP4_PV_MMA_PER_TILE>, full_tt_fp8e4m3<C::V_SC_TMEM_WIDTH>>;
    constexpr bool STATIC_PRESTAGE_K_SCALE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 50 || STATIC_CONSUMER_MODE == 51);
    constexpr int SCORE_TMEM_SLOTS = STATIC_TRIPLE_SCORE_TMEM ? 3 : (STATIC_DUAL_SCORE_TMEM ? 2 : 1);
    constexpr int P_STAGE_SLOTS = fp4pv_p_stage_slots<C>::value;
    static_assert(P_STAGE_SLOTS == 2 || P_STAGE_SLOTS == 3 || P_STAGE_SLOTS == 4,
        "FP4PV P staging supports two, three, or four slots");
    static_assert(!STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64 ||
        (C::Nb == 128 && C::CLUSTER_SIZE == 1 && !STATIC_MXFP4_K256),
        "online split P-ready K64 currently supports Nb=128, cluster1, non-K256 only");
    static_assert(!STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG ||
        (C::TOTAL_WGS == 4 && STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64),
        "online split P-quant requires the 4-WG split-ready config");
    static_assert(!STATIC_ONLINE_MXFP4_PRE_P_QUANT_CORR_PUBLISH ||
        (MXFP4_PV && ONLINE && !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM),
        "pre-P-quant correction publish is only implemented for non-dual online MXFP4 PV");
    static_assert(!STATIC_ONLINE_MXFP4_STREAM_SCORE_RELOAD ||
        (MXFP4_PV && ONLINE && !EXTERNAL_COL_LSE && C::Nb == 128 &&
         C::CLUSTER_SIZE == 1 && !STATIC_MXFP4_K256 &&
         !STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64 &&
         !STATIC_ONLINE_MXFP4_PRE_P_QUANT_CORR_PUBLISH),
        "online stream-score reload currently supports the Nb128 fixed-scale fp32-pack path only");
    static_assert(!STATIC_ONLINE_MXFP4_STREAM_SCORE_RELOAD_HALF ||
        (MXFP4_PV && ONLINE && !EXTERNAL_COL_LSE && C::Nb == 128 &&
         C::CLUSTER_SIZE == 1 && !STATIC_MXFP4_K256 &&
         !STATIC_ONLINE_MXFP4_STREAM_SCORE_RELOAD &&
         !STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64 &&
         !STATIC_ONLINE_MXFP4_PRE_P_QUANT_CORR_PUBLISH),
        "online half-stream score reload currently supports the Nb128 fixed-scale fp32-pack path only");
    constexpr int P_COPY_DONE_SLOTS = SCORE_TMEM_SLOTS > 1 ? SCORE_TMEM_SLOTS : P_STAGE_SLOTS;
    constexpr int SCORE_TMEM_WIDTH = SCORE_TMEM_SLOTS * C::Nb;
    constexpr bool STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM =
        STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM && STATIC_DUAL_SCORE_TMEM;
    constexpr bool STATIC_DIRECT_AFTER_RESCALE_SHAPE_OK =
        (C::Nb == 128 && C::CLUSTER_SIZE == 1) ||
        (C::Mb == 128 && C::Nb == 128 && C::CLUSTER_SIZE == 2) ||
        (C::Mb == 64 && C::Nb == 64 && C::CLUSTER_SIZE == 2);
    static_assert(!STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE ||
        (MXFP4_PV && ONLINE && !EXTERNAL_COL_LSE &&
         STATIC_DIRECT_AFTER_RESCALE_SHAPE_OK && !STATIC_MXFP4_K256 &&
         STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM &&
         !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_FORCE_SPARE),
        "direct-after-rescale dual output accumulation currently supports online Nb128 MXFP4 PV with score-TMEM accumulators only");
    constexpr int OUTPUT_TMEM_SLOTS =
        (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
         !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM) ? 2 : 1;
    constexpr int SCALE_TMEM_BASE = SCORE_TMEM_WIDTH + OUTPUT_TMEM_SLOTS * C::Dvo;
    constexpr int ALIAS_Q_SC_OFFSET = 0;
    constexpr int ALIAS_K_SC_OFFSET = 16;
    constexpr int ALIAS_P_SC_OFFSET = 32;
    constexpr int ALIAS_P_SC_K64_HALF1_OFFSET = 48;
    constexpr int ALIAS_V_SC_OFFSET = 64;
    constexpr int Q_SCALE_TMEM_WIDTH = STATIC_COMPACT_Q_SCALE_TMEM ? 16 : C::Q_SC_TMEM_WIDTH;
    constexpr int K_SCALE_TMEM_WIDTH = STATIC_PRESTAGE_K_SCALE_TMEM ? 16 * C::QK_SCALE_CHUNKS : 16;
    constexpr int P_SCALE_TMEM_WIDTH = MXFP4_PV ? 16 * MXFP4_PV_MMA_PER_TILE : C::P_SC_TMEM_WIDTH;
    constexpr int V_SCALE_TMEM_WIDTH = MXFP4_PV ? 32 * MXFP4_PV_MMA_PER_TILE : C::V_SC_TMEM_WIDTH;
    using p_fp4_k256_tile = st_fp4e2m1_2<C::Mb, C::Nb>;
    using p_sc_k256_tile = st_fp8e8m0<32, 16, false>;
    using v_fp4_k256_tile = st_fp4e2m1_2<C::Dvo / 2, C::Nb>;
    using v_sc_k256_tile = st_fp8e8m0<32, 16, false>;
    static_assert(!STATIC_ALIAS_SCALE_TMEM || C::Nb >= ALIAS_V_SC_OFFSET + V_SCALE_TMEM_WIDTH,
        "aliased scale TMEM layout must fit inside one score slot");
    static_assert(!STATIC_EARLY_PV_K64_HALF_READY || STATIC_ALIAS_SCALE_TMEM,
        "early K64 PV staging currently uses the aliased score-scale TMEM layout");
    static_assert(!STATIC_EARLY_PV_K64_HALF_READY ||
        (ALIAS_P_SC_K64_HALF1_OFFSET + P_SCALE_TMEM_WIDTH <= ALIAS_V_SC_OFFSET),
        "early K64 PV staging needs a second P-scale slot before V scales");

    const int cta_rank = cluster_ctarank();
    const int tiles_m = g.q.rows() / C::Mb;
    static_assert(C::Mb % C::Nb == 0, "streaming live FP4PV expects score-N to tile the row block");
    constexpr int SCORE_ITERS_PER_M_TILE = C::Mb / C::Nb;
    const int score_tiles_m = tiles_m * SCORE_ITERS_PER_M_TILE;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int num_head = g.q.depth();
    const int num_batch = g.q.batch();
    const int seqlen_k = g.k.rows();
    const int size_one_head = seqlen_k * (C::Dqk + C::Dvo) * 2;
    constexpr int size_l2 = 50 * 1024 * 1024;
    int swizzle = 1;
    if (size_l2 >= size_one_head) {
        swizzle = 1 << (31 - __clz(size_l2 / size_one_head));
    }
    const int num_hb = num_head * num_batch;
    const int num_hb_quotient = num_hb / swizzle;
    const int num_hb_remainder = num_hb - num_hb_quotient * swizzle;
    const int l2_major = swizzle * num_block;
    const int total_bids = num_batch * num_head * num_block * C::CLUSTER_SIZE;
    constexpr bool HAS_STATIC_CONSUMER_MODE = STATIC_CONSUMER_MODE >= 0;
    constexpr bool STATIC_SPLIT_ACCUM = (!ONLINE) && (STATIC_CONSUMER_MODE == 1);
    constexpr bool STATIC_TCGEN_ACCUM = (!ONLINE) && (STATIC_CONSUMER_MODE == 0);
    constexpr bool STATIC_PROD_TCGEN = (!ONLINE) && (STATIC_CONSUMER_MODE == 2);
    constexpr bool STATIC_PROD_TCGEN_MM2 = (!ONLINE) && (STATIC_CONSUMER_MODE == 3);
    constexpr bool STATIC_PROD_TCGEN_AUTO = (!ONLINE) && (STATIC_CONSUMER_MODE == 4);
    constexpr bool STATIC_PROD_TCGEN_SYNC = (!ONLINE) && (STATIC_CONSUMER_MODE == 5);
    constexpr bool STATIC_EXTERNAL_SPLIT_P_QUANT_2WG = MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 9 || STATIC_CONSUMER_MODE == 90 ||
         STATIC_CONSUMER_MODE == 178 ||
         STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE);
    constexpr bool STATIC_SPLIT_P_QUANT_2WG =
        STATIC_EXTERNAL_SPLIT_P_QUANT_2WG || STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG;
    constexpr bool STATIC_SPLIT_P_QUANT_4WG = MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 10 || STATIC_CONSUMER_MODE == 17 ||
         STATIC_CONSUMER_MODE == 19 || STATIC_CONSUMER_MODE == 26 ||
         STATIC_CONSUMER_MODE == 31 || STATIC_CONSUMER_MODE == 32 ||
         STATIC_CONSUMER_MODE == 38 || STATIC_CONSUMER_MODE == 39 ||
         STATIC_CONSUMER_MODE == 40 || STATIC_CONSUMER_MODE == 41 ||
         STATIC_CONSUMER_MODE == 42 || STATIC_CONSUMER_MODE == 43 ||
         STATIC_CONSUMER_MODE == 44 || STATIC_CONSUMER_MODE == 45 ||
         STATIC_CONSUMER_MODE == 46 || STATIC_CONSUMER_MODE == 47 ||
         STATIC_CONSUMER_MODE == 48 || STATIC_CONSUMER_MODE == 49 ||
         STATIC_CONSUMER_MODE == 50 || STATIC_CONSUMER_MODE == 51 ||
         STATIC_CONSUMER_MODE == 52 || STATIC_CONSUMER_MODE == 55 ||
         STATIC_CONSUMER_MODE == 66 || STATIC_CONSUMER_MODE == 68 ||
         STATIC_CONSUMER_MODE == 63 || STATIC_CONSUMER_MODE == 64 ||
         STATIC_CONSUMER_MODE == 65 ||
         STATIC_CONSUMER_MODE == 91 ||
         STATIC_CONSUMER_MODE == 92 ||
         STATIC_CONSUMER_MODE == 93 || STATIC_CONSUMER_MODE == 94 ||
         STATIC_CONSUMER_MODE == 95 || STATIC_CONSUMER_MODE == 96 ||
         STATIC_CONSUMER_MODE == 97 || STATIC_CONSUMER_MODE == 98 ||
         STATIC_CONSUMER_MODE == 99 || STATIC_CONSUMER_MODE == 100 ||
         STATIC_CONSUMER_MODE == 101 || STATIC_CONSUMER_MODE == 102 ||
         STATIC_CONSUMER_MODE == 103 ||
         STATIC_CONSUMER_MODE == 104 || STATIC_CONSUMER_MODE == 105 ||
         STATIC_CONSUMER_MODE == 106 || STATIC_CONSUMER_MODE == 107 ||
         STATIC_CONSUMER_MODE == 108 || STATIC_CONSUMER_MODE == 109 ||
         STATIC_CONSUMER_MODE == 110 || STATIC_CONSUMER_MODE == 111 ||
         STATIC_CONSUMER_MODE == 112 ||
         STATIC_CONSUMER_MODE == 113 || STATIC_CONSUMER_MODE == 114 ||
         STATIC_CONSUMER_MODE == 115 || STATIC_CONSUMER_MODE == 116 ||
         STATIC_CONSUMER_MODE == 131 || STATIC_CONSUMER_MODE == 132 ||
         STATIC_CONSUMER_MODE == 117 || STATIC_CONSUMER_MODE == 118 ||
         STATIC_CONSUMER_MODE == 119 || STATIC_CONSUMER_MODE == 120 ||
         STATIC_CONSUMER_MODE == 123 || STATIC_CONSUMER_MODE == 128 ||
         STATIC_CONSUMER_MODE == 129 || STATIC_CONSUMER_MODE == 130 ||
         STATIC_CONSUMER_MODE == 133 || STATIC_CONSUMER_MODE == 134 ||
         STATIC_CONSUMER_MODE == 135 || STATIC_CONSUMER_MODE == 136 ||
         STATIC_CONSUMER_MODE == 137 ||
         STATIC_CONSUMER_MODE == 138 || STATIC_CONSUMER_MODE == 139 ||
         STATIC_CONSUMER_MODE == 140 || STATIC_CONSUMER_MODE == 141 ||
         STATIC_CONSUMER_MODE == 142 || STATIC_CONSUMER_MODE == 143 ||
         STATIC_CONSUMER_MODE == 144 || STATIC_CONSUMER_MODE == 145 ||
         STATIC_CONSUMER_MODE == 146 || STATIC_CONSUMER_MODE == 147 ||
         STATIC_CONSUMER_MODE == 148 || STATIC_CONSUMER_MODE == 149 ||
         STATIC_CONSUMER_MODE == 150 || STATIC_CONSUMER_MODE == 151 ||
         STATIC_CONSUMER_MODE == 152 || STATIC_CONSUMER_MODE == 153 ||
         STATIC_CONSUMER_MODE == 154 || STATIC_CONSUMER_MODE == 155 ||
         STATIC_CONSUMER_MODE == 157 || STATIC_CONSUMER_MODE == 158 ||
         STATIC_CONSUMER_MODE == 159 || STATIC_CONSUMER_MODE == 160 ||
         STATIC_CONSUMER_MODE == 161 || STATIC_CONSUMER_MODE == 162 ||
         STATIC_CONSUMER_MODE == 163 || STATIC_CONSUMER_MODE == 164 ||
         STATIC_CONSUMER_MODE == 165 || STATIC_CONSUMER_MODE == 166 ||
         STATIC_CONSUMER_MODE == 167 || STATIC_CONSUMER_MODE == 168 ||
         STATIC_CONSUMER_MODE == 169 || STATIC_CONSUMER_MODE == 170 ||
         STATIC_CONSUMER_MODE == 171 || STATIC_CONSUMER_MODE == 172 ||
         STATIC_CONSUMER_MODE == 173 || STATIC_CONSUMER_MODE == 174 ||
         STATIC_CONSUMER_MODE == 176 || STATIC_CONSUMER_MODE == 180 ||
         STATIC_CONSUMER_MODE == 181 || STATIC_CONSUMER_MODE == 182 ||
         STATIC_CONSUMER_MODE == 183 ||
         STATIC_CONSUMER_MODE == 156 ||
         STATIC_CONSUMER_MODE == 197 || STATIC_CONSUMER_MODE == 198 ||
         STATIC_CONSUMER_MODE == 199 || STATIC_CONSUMER_MODE == 200 ||
         STATIC_CONSUMER_MODE == 201 || STATIC_CONSUMER_MODE == 202 ||
         STATIC_CONSUMER_MODE == 203 || STATIC_CONSUMER_MODE == 204 ||
         STATIC_CONSUMER_MODE == 205 || STATIC_CONSUMER_MODE == 206 ||
         STATIC_CONSUMER_MODE == 207 || STATIC_CONSUMER_MODE == 208 ||
         STATIC_CONSUMER_MODE == 209 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
         STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
         STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
         STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
         STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
         STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
         STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
         STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 225 ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
         STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_P128_SPLIT_SCALE_ASYNC_COPY_REWRITE ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 237) ||
         STATIC_CONSUMER_MODE == 244 || STATIC_CONSUMER_MODE == 245 ||
         STATIC_CONSUMER_MODE == 247 ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_EARLY_ASYNC_P_SCALE_TMEM ||
         STATIC_EARLY_LOCAL_P_SCALE_TMEM ||
         STATIC_ANY_DIRECT_P_SCALE_TMEM_STORE ||
         STATIC_DECOUPLED_PV_ISSUE_REWRITE ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE);
    constexpr bool STATIC_QUANT_ASYNC_P_SCALE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 17 || STATIC_CONSUMER_MODE == 40 ||
         STATIC_CONSUMER_MODE == 41 || STATIC_CONSUMER_MODE == 46 ||
         STATIC_CONSUMER_MODE == 49 || STATIC_CONSUMER_MODE == 50 ||
         STATIC_CONSUMER_MODE == 51 || STATIC_CONSUMER_MODE == 52 ||
         STATIC_CONSUMER_MODE == 55 ||
         STATIC_CONSUMER_MODE == 94 ||
         STATIC_CONSUMER_MODE == 95 || STATIC_CONSUMER_MODE == 96 ||
          STATIC_CONSUMER_MODE == 97 || STATIC_CONSUMER_MODE == 98 ||
          STATIC_CONSUMER_MODE == 99 || STATIC_CONSUMER_MODE == 100 ||
          STATIC_CONSUMER_MODE == 101 || STATIC_CONSUMER_MODE == 102 ||
          STATIC_CONSUMER_MODE == 103 ||
          STATIC_CONSUMER_MODE == 104 || STATIC_CONSUMER_MODE == 105 ||
          STATIC_CONSUMER_MODE == 106 || STATIC_CONSUMER_MODE == 107 ||
          STATIC_CONSUMER_MODE == 108 || STATIC_CONSUMER_MODE == 109 ||
          STATIC_CONSUMER_MODE == 110 || STATIC_CONSUMER_MODE == 111 ||
          STATIC_CONSUMER_MODE == 112 ||
          STATIC_CONSUMER_MODE == 117 || STATIC_CONSUMER_MODE == 118 ||
          STATIC_CONSUMER_MODE == 119 || STATIC_CONSUMER_MODE == 120 ||
          STATIC_CONSUMER_MODE == 123 || STATIC_CONSUMER_MODE == 128 ||
          STATIC_CONSUMER_MODE == 133 || STATIC_CONSUMER_MODE == 134 ||
          STATIC_CONSUMER_MODE == 66 || STATIC_CONSUMER_MODE == 68 ||
          STATIC_CONSUMER_MODE == 63 ||
          STATIC_CONSUMER_MODE == 64 ||
          STATIC_CONSUMER_MODE == 65 ||
          STATIC_FUSED_QUANT_ASYNC_SCALE_SKIP_PREPACK_REWRITE ||
          STATIC_FUSED_QUANT_ANY_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE ||
          STATIC_DIRECT_P_SCALE_TMEM_STORE);
    constexpr bool STATIC_WARP_LSE_BROADCAST =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 64 || STATIC_CONSUMER_MODE == 80 ||
         STATIC_CONSUMER_MODE == 82 || STATIC_CONSUMER_MODE == 86 ||
         STATIC_CONSUMER_MODE == 91 || STATIC_CONSUMER_MODE == 92 ||
         STATIC_CONSUMER_MODE == 93 || STATIC_CONSUMER_MODE == 106 ||
         STATIC_CONSUMER_MODE == 107 || STATIC_CONSUMER_MODE == 155 ||
         STATIC_CONSUMER_MODE == 156 || STATIC_CONSUMER_MODE == 157 ||
         STATIC_CONSUMER_MODE == 180 || STATIC_CONSUMER_MODE == 181 ||
         STATIC_CONSUMER_MODE == 182 || STATIC_CONSUMER_MODE == 183 ||
         STATIC_CONSUMER_MODE == 184 || STATIC_CONSUMER_MODE == 185 ||
         STATIC_CONSUMER_MODE == 186 || STATIC_CONSUMER_MODE == 187 ||
         STATIC_CONSUMER_MODE == 188 || STATIC_CONSUMER_MODE == 189 ||
         STATIC_CONSUMER_MODE == 190 || STATIC_CONSUMER_MODE == 191 ||
         STATIC_CONSUMER_MODE == 192 || STATIC_CONSUMER_MODE == 193 ||
         STATIC_CONSUMER_MODE == 194 || STATIC_CONSUMER_MODE == 195 ||
         STATIC_CONSUMER_MODE == 196 || STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_P32_PAYLOAD_FEED_REWRITE || STATIC_P32_PUBLISH_READY_REWRITE ||
         STATIC_P32_SPLIT_SCALE_REWRITE ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE);
    constexpr bool STATIC_SHARED_WARP_LSE_BROADCAST =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 158);
    constexpr bool STATIC_PRELOAD_WARP_LSE_BROADCAST =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 157 || STATIC_CONSUMER_MODE == 181);
    constexpr bool STATIC_FINAL_PV_READY_ONLY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 108 || STATIC_CONSUMER_MODE == 109 ||
         STATIC_CONSUMER_MODE == 110 || STATIC_CONSUMER_MODE == 111 ||
         STATIC_CONSUMER_MODE == 112 ||
         STATIC_CONSUMER_MODE == 113 || STATIC_CONSUMER_MODE == 114 ||
         STATIC_CONSUMER_MODE == 115 || STATIC_CONSUMER_MODE == 116 ||
         STATIC_CONSUMER_MODE == 131 || STATIC_CONSUMER_MODE == 132 ||
         STATIC_CONSUMER_MODE == 117 || STATIC_CONSUMER_MODE == 118 ||
         STATIC_CONSUMER_MODE == 119 || STATIC_CONSUMER_MODE == 120 ||
         STATIC_CONSUMER_MODE == 123 || STATIC_CONSUMER_MODE == 134 ||
         STATIC_CONSUMER_MODE == 136);
    constexpr bool STATIC_DECOUPLE_P_SCALE_REMOTE_WAIT =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 133 || STATIC_CONSUMER_MODE == 134 ||
         STATIC_CONSUMER_MODE == 135 || STATIC_CONSUMER_MODE == 136 ||
         STATIC_CONSUMER_MODE == 137 ||
         STATIC_CONSUMER_MODE == 138 || STATIC_CONSUMER_MODE == 139 ||
         STATIC_CONSUMER_MODE == 140 || STATIC_CONSUMER_MODE == 141 ||
         STATIC_CONSUMER_MODE == 142 || STATIC_CONSUMER_MODE == 143 ||
         STATIC_CONSUMER_MODE == 144 || STATIC_CONSUMER_MODE == 145 ||
         STATIC_CONSUMER_MODE == 146 || STATIC_CONSUMER_MODE == 147 ||
         STATIC_CONSUMER_MODE == 148 || STATIC_CONSUMER_MODE == 149 ||
         STATIC_CONSUMER_MODE == 150 || STATIC_CONSUMER_MODE == 151 ||
         STATIC_CONSUMER_MODE == 152 || STATIC_CONSUMER_MODE == 153 ||
         STATIC_CONSUMER_MODE == 154 || STATIC_CONSUMER_MODE == 155 ||
         STATIC_CONSUMER_MODE == 157 || STATIC_CONSUMER_MODE == 158 ||
         STATIC_CONSUMER_MODE == 159 || STATIC_CONSUMER_MODE == 160 ||
         STATIC_CONSUMER_MODE == 161 || STATIC_CONSUMER_MODE == 162 ||
         STATIC_CONSUMER_MODE == 163 || STATIC_CONSUMER_MODE == 164 ||
         STATIC_CONSUMER_MODE == 165 || STATIC_CONSUMER_MODE == 166 ||
         STATIC_CONSUMER_MODE == 167 || STATIC_CONSUMER_MODE == 168 ||
         STATIC_CONSUMER_MODE == 169 || STATIC_CONSUMER_MODE == 170 ||
         STATIC_CONSUMER_MODE == 171 || STATIC_CONSUMER_MODE == 172 ||
         STATIC_CONSUMER_MODE == 173 || STATIC_CONSUMER_MODE == 174 ||
         STATIC_CONSUMER_MODE == 176 || STATIC_CONSUMER_MODE == 180 ||
         STATIC_CONSUMER_MODE == 181 || STATIC_CONSUMER_MODE == 182 ||
         STATIC_CONSUMER_MODE == 183 || STATIC_CONSUMER_MODE == 184 ||
         STATIC_CONSUMER_MODE == 185 || STATIC_CONSUMER_MODE == 186 ||
         STATIC_CONSUMER_MODE == 187 || STATIC_CONSUMER_MODE == 188 ||
         STATIC_CONSUMER_MODE == 189 || STATIC_CONSUMER_MODE == 190 ||
         STATIC_CONSUMER_MODE == 191 || STATIC_CONSUMER_MODE == 192 ||
         STATIC_CONSUMER_MODE == 193 || STATIC_CONSUMER_MODE == 194 ||
         STATIC_CONSUMER_MODE == 195 || STATIC_CONSUMER_MODE == 196 ||
         STATIC_CONSUMER_MODE == 197 || STATIC_CONSUMER_MODE == 198 ||
         STATIC_CONSUMER_MODE == 199 || STATIC_CONSUMER_MODE == 200 ||
         STATIC_CONSUMER_MODE == 201 || STATIC_CONSUMER_MODE == 202 ||
         STATIC_CONSUMER_MODE == 203 || STATIC_CONSUMER_MODE == 204 ||
         STATIC_CONSUMER_MODE == 205 || STATIC_CONSUMER_MODE == 206 ||
         STATIC_CONSUMER_MODE == 207 || STATIC_CONSUMER_MODE == 208 ||
         STATIC_CONSUMER_MODE == 209 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
         STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
         STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
         STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
         STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
         STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
         STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
         STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         (STATIC_P128_FAST_PUBLISH_REWRITE && !STATIC_P128_PRODUCER_FOLD_REMOTE_REWRITE) ||
         STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_CONSUMER_MODE == 225 ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
         STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_CONSUMER_MODE == 156 ||
         STATIC_DECOUPLED_PV_ISSUE_REWRITE ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE);
    constexpr bool STATIC_TCGEN_P_STAGE_REUSE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 159 || STATIC_DECOUPLED_PV_ISSUE_REWRITE);
    constexpr bool STATIC_ANY_TCGEN_P_STAGE_REUSE =
        STATIC_TCGEN_P_STAGE_REUSE ||
        STATIC_ONLINE_MXFP4_EARLY_P_READY ||
        STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM;
    constexpr bool STATIC_REG_TUNE_Q96_P64_O32 =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 65 || STATIC_CONSUMER_MODE == 148 ||
         STATIC_CONSUMER_MODE == 149 || STATIC_CONSUMER_MODE == 213);
    constexpr bool STATIC_PRODUCER_ASYNC_P_SCALE_AGGREGATE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 42 || STATIC_CONSUMER_MODE == 43 ||
         STATIC_CONSUMER_MODE == 113 || STATIC_CONSUMER_MODE == 114 ||
         STATIC_CONSUMER_MODE == 115 || STATIC_CONSUMER_MODE == 116 ||
         STATIC_CONSUMER_MODE == 131 || STATIC_CONSUMER_MODE == 132);
    constexpr bool STATIC_PRODUCER_ASYNC_P_SCALE_COPY =
        STATIC_ONLINE_MXFP4_PRODUCER_ASYNC_P_SCALE_COPY ||
        (MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
         (STATIC_CONSUMER_MODE == 44 || STATIC_CONSUMER_MODE == 45 ||
          STATIC_CONSUMER_MODE == 48 || STATIC_CONSUMER_MODE == 135 ||
          STATIC_CONSUMER_MODE == 136 || STATIC_CONSUMER_MODE == 137 ||
          STATIC_CONSUMER_MODE == 138 || STATIC_CONSUMER_MODE == 139 ||
          STATIC_CONSUMER_MODE == 140 || STATIC_CONSUMER_MODE == 141 ||
          STATIC_CONSUMER_MODE == 142 || STATIC_CONSUMER_MODE == 143 ||
          STATIC_CONSUMER_MODE == 144 || STATIC_CONSUMER_MODE == 145 ||
          STATIC_CONSUMER_MODE == 146 || STATIC_CONSUMER_MODE == 147 ||
          STATIC_CONSUMER_MODE == 148 || STATIC_CONSUMER_MODE == 149 ||
          STATIC_CONSUMER_MODE == 150 || STATIC_CONSUMER_MODE == 151 ||
          STATIC_CONSUMER_MODE == 152 || STATIC_CONSUMER_MODE == 153 ||
          STATIC_CONSUMER_MODE == 154 || STATIC_CONSUMER_MODE == 155 ||
          STATIC_CONSUMER_MODE == 157 || STATIC_CONSUMER_MODE == 158 ||
         STATIC_CONSUMER_MODE == 159 || STATIC_CONSUMER_MODE == 160 ||
         STATIC_CONSUMER_MODE == 161 || STATIC_CONSUMER_MODE == 162 ||
         STATIC_CONSUMER_MODE == 163 || STATIC_CONSUMER_MODE == 164 ||
         STATIC_CONSUMER_MODE == 165 || STATIC_CONSUMER_MODE == 166 ||
         STATIC_CONSUMER_MODE == 167 || STATIC_CONSUMER_MODE == 168 ||
         STATIC_CONSUMER_MODE == 169 || STATIC_CONSUMER_MODE == 170 ||
         STATIC_CONSUMER_MODE == 171 || STATIC_CONSUMER_MODE == 172 ||
         STATIC_CONSUMER_MODE == 173 || STATIC_CONSUMER_MODE == 176 ||
         STATIC_CONSUMER_MODE == 180 || STATIC_CONSUMER_MODE == 181 ||
         STATIC_CONSUMER_MODE == 182 || STATIC_CONSUMER_MODE == 183 ||
         STATIC_CONSUMER_MODE == 197 || STATIC_CONSUMER_MODE == 198 ||
         STATIC_CONSUMER_MODE == 199 || STATIC_CONSUMER_MODE == 200 ||
         STATIC_CONSUMER_MODE == 201 || STATIC_CONSUMER_MODE == 202 ||
         STATIC_CONSUMER_MODE == 203 || STATIC_CONSUMER_MODE == 204 ||
         STATIC_CONSUMER_MODE == 205 || STATIC_CONSUMER_MODE == 206 ||
         STATIC_CONSUMER_MODE == 207 || STATIC_CONSUMER_MODE == 208 ||
         STATIC_CONSUMER_MODE == 209 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
         STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
         STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
         STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
         STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
         STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
         STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
         STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_ASYNC_SKIP_PREPACK_REWRITE ||
         STATIC_FUSED_PRODUCER_ANY_EARLY_SCALE_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_CONSUMER_MODE == 225 ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
         STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_DECOUPLED_PV_EARLY_K64_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_CONSUMER_MODE == 156));
    constexpr bool STATIC_WAIT_ALIAS_SCALE_SLOT_REUSE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 152 || STATIC_CONSUMER_MODE == 153 ||
         STATIC_CONSUMER_MODE == 154 || STATIC_CONSUMER_MODE == 155 ||
         STATIC_CONSUMER_MODE == 157 || STATIC_CONSUMER_MODE == 158 ||
         STATIC_CONSUMER_MODE == 159 || STATIC_CONSUMER_MODE == 160 ||
         STATIC_CONSUMER_MODE == 161 || STATIC_CONSUMER_MODE == 162 ||
         STATIC_CONSUMER_MODE == 164 || STATIC_CONSUMER_MODE == 165 ||
         STATIC_CONSUMER_MODE == 169 || STATIC_CONSUMER_MODE == 171 ||
         STATIC_CONSUMER_MODE == 172 || STATIC_CONSUMER_MODE == 173 ||
         STATIC_CONSUMER_MODE == 198 || STATIC_CONSUMER_MODE == 199 ||
         STATIC_CONSUMER_MODE == 200 || STATIC_CONSUMER_MODE == 201 ||
         STATIC_CONSUMER_MODE == 202 || STATIC_CONSUMER_MODE == 203 ||
         STATIC_CONSUMER_MODE == 204 || STATIC_CONSUMER_MODE == 205 ||
         STATIC_CONSUMER_MODE == 206 || STATIC_CONSUMER_MODE == 207 ||
         STATIC_CONSUMER_MODE == 208 || STATIC_CONSUMER_MODE == 209 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
         STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
         STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
         STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
         STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
         STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
         STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
         STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_WAIT_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 225 ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
         STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_DECOUPLED_PV_EARLY_K64_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE);
    constexpr bool STATIC_WAIT_P_STAGE_SLOT_REUSE =
        STATIC_ONLINE_MXFP4_EARLY_P_READY ||
        STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM ||
        (MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
         (STATIC_CONSUMER_MODE == 154 || STATIC_TCGEN_P_STAGE_REUSE));
    constexpr bool STATIC_EARLY_LOCAL_P_SCALE_READY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 135 || STATIC_CONSUMER_MODE == 136 ||
         STATIC_CONSUMER_MODE == 137 ||
         STATIC_CONSUMER_MODE == 138 || STATIC_CONSUMER_MODE == 139 ||
         STATIC_CONSUMER_MODE == 140 || STATIC_CONSUMER_MODE == 141 ||
         STATIC_CONSUMER_MODE == 142 || STATIC_CONSUMER_MODE == 143 ||
         STATIC_CONSUMER_MODE == 144 || STATIC_CONSUMER_MODE == 145 ||
         STATIC_CONSUMER_MODE == 146 || STATIC_CONSUMER_MODE == 147 ||
         STATIC_CONSUMER_MODE == 148 || STATIC_CONSUMER_MODE == 149 ||
         STATIC_CONSUMER_MODE == 150 || STATIC_CONSUMER_MODE == 151 ||
         STATIC_CONSUMER_MODE == 152 || STATIC_CONSUMER_MODE == 153 ||
         STATIC_CONSUMER_MODE == 154 || STATIC_CONSUMER_MODE == 155 ||
         STATIC_CONSUMER_MODE == 157 || STATIC_CONSUMER_MODE == 158 ||
         STATIC_CONSUMER_MODE == 159 || STATIC_CONSUMER_MODE == 160 ||
         STATIC_CONSUMER_MODE == 161 || STATIC_CONSUMER_MODE == 162 ||
         STATIC_CONSUMER_MODE == 163 || STATIC_CONSUMER_MODE == 164 ||
         STATIC_CONSUMER_MODE == 165 || STATIC_CONSUMER_MODE == 166 ||
         STATIC_CONSUMER_MODE == 167 || STATIC_CONSUMER_MODE == 168 ||
         STATIC_CONSUMER_MODE == 169 || STATIC_CONSUMER_MODE == 170 ||
         STATIC_CONSUMER_MODE == 171 || STATIC_CONSUMER_MODE == 172 ||
         STATIC_CONSUMER_MODE == 173 || STATIC_CONSUMER_MODE == 174 ||
         STATIC_CONSUMER_MODE == 176 || STATIC_CONSUMER_MODE == 180 ||
         STATIC_CONSUMER_MODE == 181 || STATIC_CONSUMER_MODE == 182 ||
         STATIC_CONSUMER_MODE == 183 || STATIC_CONSUMER_MODE == 184 ||
         STATIC_CONSUMER_MODE == 185 || STATIC_CONSUMER_MODE == 186 ||
         STATIC_CONSUMER_MODE == 187 || STATIC_CONSUMER_MODE == 188 ||
         STATIC_CONSUMER_MODE == 189 || STATIC_CONSUMER_MODE == 190 ||
         STATIC_CONSUMER_MODE == 191 || STATIC_CONSUMER_MODE == 192 ||
         STATIC_CONSUMER_MODE == 193 || STATIC_CONSUMER_MODE == 194 ||
         STATIC_CONSUMER_MODE == 195 || STATIC_CONSUMER_MODE == 196 ||
         STATIC_CONSUMER_MODE == 197 || STATIC_CONSUMER_MODE == 198 ||
         STATIC_CONSUMER_MODE == 199 || STATIC_CONSUMER_MODE == 200 ||
         STATIC_CONSUMER_MODE == 201 || STATIC_CONSUMER_MODE == 202 ||
         STATIC_CONSUMER_MODE == 203 || STATIC_CONSUMER_MODE == 204 ||
         STATIC_CONSUMER_MODE == 205 || STATIC_CONSUMER_MODE == 206 ||
         STATIC_CONSUMER_MODE == 207 || STATIC_CONSUMER_MODE == 208 ||
         STATIC_CONSUMER_MODE == 209 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
         STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
         STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
         STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
         STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
         STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
         STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
         STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 225 ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
         STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_CONSUMER_MODE == 156);
    constexpr bool STATIC_SCALAR_P_PAYLOAD_STORE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 141 || STATIC_CONSUMER_MODE == 142 ||
         STATIC_CONSUMER_MODE == 151 || STATIC_CONSUMER_MODE == 208 ||
         STATIC_P128_SCALAR_PAYLOAD_VARIANT ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_SCALAR);
    constexpr bool STATIC_STRIPED_P_PAYLOAD_STORE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 144 || STATIC_CONSUMER_MODE == 145 ||
         STATIC_CONSUMER_MODE == 207);
    constexpr bool STATIC_HALF_STRIPED_P_PAYLOAD_STORE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 146 || STATIC_CONSUMER_MODE == 147 ||
         STATIC_CONSUMER_MODE == 161 || STATIC_CONSUMER_MODE == 165 ||
         STATIC_CONSUMER_MODE == 173 || STATIC_CONSUMER_MODE == 187 ||
         STATIC_CONSUMER_MODE == 188 || STATIC_CONSUMER_MODE == 205 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_HALF_STRIPED);
    constexpr bool STATIC_V2_P_PAYLOAD_STORE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 160 || STATIC_CONSUMER_MODE == 162 ||
         STATIC_CONSUMER_MODE == 199 || STATIC_CONSUMER_MODE == 200 ||
         STATIC_CONSUMER_MODE == 209 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_V2);
    constexpr bool STATIC_STAGGERED_P_PAYLOAD_STORE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 217 || STATIC_CONSUMER_MODE == 220);
    constexpr bool STATIC_HALF_STAGGERED_P_PAYLOAD_STORE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 218);
    constexpr bool STATIC_PRECOMPUTED_P_AMAX =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 163 || STATIC_CONSUMER_MODE == 164 ||
         STATIC_CONSUMER_MODE == 165 || STATIC_CONSUMER_MODE == 169 ||
         STATIC_CONSUMER_MODE == 170 || STATIC_CONSUMER_MODE == 171 ||
         STATIC_CONSUMER_MODE == 172 || STATIC_CONSUMER_MODE == 173 ||
         STATIC_CONSUMER_MODE == 174 || STATIC_CONSUMER_MODE == 180 ||
         STATIC_CONSUMER_MODE == 181 || STATIC_CONSUMER_MODE == 182 ||
         STATIC_CONSUMER_MODE == 183 || STATIC_CONSUMER_MODE == 184 ||
         STATIC_CONSUMER_MODE == 185 || STATIC_CONSUMER_MODE == 186 ||
         STATIC_CONSUMER_MODE == 187 || STATIC_CONSUMER_MODE == 188 ||
         STATIC_CONSUMER_MODE == 189 || STATIC_CONSUMER_MODE == 190 ||
         STATIC_CONSUMER_MODE == 191 || STATIC_CONSUMER_MODE == 192 ||
         STATIC_CONSUMER_MODE == 193 || STATIC_CONSUMER_MODE == 194 ||
         STATIC_CONSUMER_MODE == 195 || STATIC_CONSUMER_MODE == 196 ||
         STATIC_CONSUMER_MODE == 197 || STATIC_CONSUMER_MODE == 198 ||
         STATIC_CONSUMER_MODE == 199 || STATIC_CONSUMER_MODE == 200 ||
         STATIC_CONSUMER_MODE == 201 || STATIC_CONSUMER_MODE == 202 ||
         STATIC_CONSUMER_MODE == 203 || STATIC_CONSUMER_MODE == 204 ||
         STATIC_CONSUMER_MODE == 205 || STATIC_CONSUMER_MODE == 206 ||
         STATIC_CONSUMER_MODE == 207 || STATIC_CONSUMER_MODE == 208 ||
         STATIC_CONSUMER_MODE == 209 || STATIC_CONSUMER_MODE == 210 ||
         STATIC_CONSUMER_MODE == 211 || STATIC_CONSUMER_MODE == 212 ||
         STATIC_CONSUMER_MODE == 213 || STATIC_CONSUMER_MODE == 214 ||
         STATIC_CONSUMER_MODE == 215 || STATIC_CONSUMER_MODE == 216 ||
         STATIC_CONSUMER_MODE == 217 || STATIC_CONSUMER_MODE == 218 ||
         STATIC_CONSUMER_MODE == 219 || STATIC_CONSUMER_MODE == 220 ||
         STATIC_CONSUMER_MODE == 221 || STATIC_CONSUMER_MODE == 222 ||
         STATIC_CONSUMER_MODE == 223 || STATIC_CONSUMER_MODE == 224 ||
         STATIC_CONSUMER_MODE == 269 || STATIC_CONSUMER_MODE == 372 ||
         STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 ||
         STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 225 || STATIC_CONSUMER_MODE == 226 ||
         STATIC_CONSUMER_MODE == 227 || STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DECOUPLED_PV_ISSUE_REWRITE ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_REWRITE);
    constexpr bool STATIC_PRODUCER_ASYNC_V_SCALE_TMEM =
        (MXFP4_PV && ONLINE && (!EXTERNAL_COL_LSE) &&
         fp4pv_online_producer_async_v_scale_tmem<C>::value) ||
        (MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
         (STATIC_CONSUMER_MODE == 46 || STATIC_CONSUMER_MODE == 47));
    constexpr bool STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK =
        (MXFP4_PV && ONLINE && (!EXTERNAL_COL_LSE) &&
         fp4pv_online_stage_v_scale_before_next_qk<C>::value) ||
        (MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 49 || STATIC_CONSUMER_MODE == 51 ||
         STATIC_CONSUMER_MODE == 63 || STATIC_CONSUMER_MODE == 68 ||
         STATIC_CONSUMER_MODE == 92 || STATIC_CONSUMER_MODE == 95 ||
         STATIC_CONSUMER_MODE == 97 || STATIC_CONSUMER_MODE == 169 ||
         STATIC_CONSUMER_MODE == 170 || STATIC_CONSUMER_MODE == 176 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 372 ||
         STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_P128_STAGE_V_FAST_PUBLISH_REWRITE ||
         STATIC_DECOUPLED_PV_STAGE_V_FIRST_REWRITE));
    constexpr bool STATIC_ONLINE_MXFP4_STAGE_P_BEFORE_V =
        MXFP4_PV && ONLINE && (!EXTERNAL_COL_LSE) &&
        fp4pv_online_stage_p_before_v<C>::value;
    constexpr bool STATIC_DEFER_SHARED_LSE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 19 || STATIC_CONSUMER_MODE == 26 ||
         STATIC_CONSUMER_MODE == 31 || STATIC_CONSUMER_MODE == 32 ||
         STATIC_CONSUMER_MODE == 38 || STATIC_CONSUMER_MODE == 39 ||
         STATIC_CONSUMER_MODE == 40 || STATIC_CONSUMER_MODE == 41 ||
         STATIC_CONSUMER_MODE == 42 || STATIC_CONSUMER_MODE == 43 ||
         STATIC_CONSUMER_MODE == 44 || STATIC_CONSUMER_MODE == 45 ||
         STATIC_CONSUMER_MODE == 47 ||
         STATIC_CONSUMER_MODE == 113 || STATIC_CONSUMER_MODE == 114 ||
         STATIC_CONSUMER_MODE == 115 || STATIC_CONSUMER_MODE == 116 ||
         STATIC_CONSUMER_MODE == 131 || STATIC_CONSUMER_MODE == 132 ||
         (STATIC_DIRECT_P_SCALE_TMEM_STORE &&
          !STATIC_P128_DIRECT_SCALE_EARLY_P_WAIT_REWRITE));
    constexpr bool STATIC_PACKED_P_SCALE_CROSSBAR =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 26 || STATIC_CONSUMER_MODE == 31 ||
         STATIC_CONSUMER_MODE == 32 || STATIC_EARLY_ASYNC_P_SCALE_TMEM);
    constexpr bool STATIC_WIDE_P_SCALE_CROSSBAR =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 38 || STATIC_CONSUMER_MODE == 39 ||
         STATIC_CONSUMER_MODE == 40 || STATIC_CONSUMER_MODE == 41 ||
         STATIC_CONSUMER_MODE == 42 || STATIC_CONSUMER_MODE == 43 ||
         STATIC_CONSUMER_MODE == 44 || STATIC_CONSUMER_MODE == 45 ||
         STATIC_CONSUMER_MODE == 47 || STATIC_CONSUMER_MODE == 76 ||
         STATIC_CONSUMER_MODE == 80 || STATIC_CONSUMER_MODE == 81 ||
         STATIC_CONSUMER_MODE == 82 || STATIC_CONSUMER_MODE == 85 ||
         STATIC_CONSUMER_MODE == 86 || STATIC_CONSUMER_MODE == 91 ||
         STATIC_CONSUMER_MODE == 92 || STATIC_CONSUMER_MODE == 93 ||
         STATIC_CONSUMER_MODE == 104 || STATIC_CONSUMER_MODE == 105 ||
         STATIC_CONSUMER_MODE == 113 || STATIC_CONSUMER_MODE == 114 ||
         STATIC_CONSUMER_MODE == 115 || STATIC_CONSUMER_MODE == 116 ||
         STATIC_CONSUMER_MODE == 126 || STATIC_CONSUMER_MODE == 127 ||
         STATIC_CONSUMER_MODE == 128 || STATIC_CONSUMER_MODE == 129 ||
         STATIC_CONSUMER_MODE == 130 || STATIC_CONSUMER_MODE == 131 ||
         STATIC_CONSUMER_MODE == 132 || STATIC_CONSUMER_MODE == 174 ||
         STATIC_CONSUMER_MODE == 184 || STATIC_CONSUMER_MODE == 185 ||
         STATIC_CONSUMER_MODE == 186 || STATIC_CONSUMER_MODE == 187 ||
         STATIC_CONSUMER_MODE == 188 || STATIC_CONSUMER_MODE == 189 ||
         STATIC_CONSUMER_MODE == 190 || STATIC_CONSUMER_MODE == 191 ||
         STATIC_CONSUMER_MODE == 192 || STATIC_CONSUMER_MODE == 193 ||
         STATIC_CONSUMER_MODE == 194 || STATIC_CONSUMER_MODE == 195 ||
         STATIC_CONSUMER_MODE == 196 || STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 392 ||
         (STATIC_EARLY_PV_K64_ANY_CROSSBAR_SCALE &&
          !STATIC_EARLY_PV_K64_HALF_SCALE_CROSSBAR &&
          !STATIC_EARLY_PV_K64_BYTE_SCALE_CROSSBAR));
    constexpr bool STATIC_BANK_FREE_P_SCALE_STAGE_STORE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 128);
    constexpr bool STATIC_BANK_FREE_P_SCALE_STAGE_STORE_2W =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 129);
    constexpr bool STATIC_BANK_FREE_P_SCALE_STAGE_STORE_4W =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 130);
    constexpr bool STATIC_BANK_FREE_P_SCALE_AGGREGATE_STAGE_STORE =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 131 || STATIC_CONSUMER_MODE == 132);
    constexpr bool STATIC_EARLY_CLUSTER_P_PAYLOAD_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 81 || STATIC_CONSUMER_MODE == 82 ||
         STATIC_CONSUMER_MODE == 86 || STATIC_CONSUMER_MODE == 91 ||
         STATIC_CONSUMER_MODE == 92 || STATIC_CONSUMER_MODE == 93 ||
         STATIC_CONSUMER_MODE == 110 || STATIC_CONSUMER_MODE == 111);
    constexpr bool STATIC_SINGLE_WG_P_PAYLOAD_CLUSTER_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 117 || STATIC_CONSUMER_MODE == 118 ||
         STATIC_CONSUMER_MODE == 120 || STATIC_CONSUMER_MODE == 123 ||
         STATIC_CONSUMER_MODE == 138 || STATIC_CONSUMER_MODE == 139 ||
         STATIC_CONSUMER_MODE == 185 || STATIC_CONSUMER_MODE == 186 ||
         STATIC_CONSUMER_MODE == 188 || STATIC_CONSUMER_MODE == 190 ||
         STATIC_CONSUMER_MODE == 192 || STATIC_CONSUMER_MODE == 193 ||
         STATIC_CONSUMER_MODE == 194 || STATIC_CONSUMER_MODE == 195 ||
         STATIC_CONSUMER_MODE == 196 || STATIC_CONSUMER_MODE == 197 ||
         STATIC_CONSUMER_MODE == 198 || STATIC_CONSUMER_MODE == 199 ||
         STATIC_CONSUMER_MODE == 200 || STATIC_CONSUMER_MODE == 201 ||
         STATIC_CONSUMER_MODE == 202 || STATIC_CONSUMER_MODE == 203 ||
         STATIC_CONSUMER_MODE == 204 || STATIC_CONSUMER_MODE == 205 ||
         STATIC_CONSUMER_MODE == 206 || STATIC_CONSUMER_MODE == 207 ||
         STATIC_CONSUMER_MODE == 208 || STATIC_CONSUMER_MODE == 209 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
         STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
         STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
         STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
         STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
         STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
         STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
         STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 225 ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
         STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_SINGLE_PUBLISH ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_REWRITE);
    constexpr bool STATIC_SINGLE_WARP_P_PAYLOAD_CLUSTER_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 118 || STATIC_CONSUMER_MODE == 120 ||
         STATIC_CONSUMER_MODE == 123 || STATIC_CONSUMER_MODE == 139 ||
         STATIC_CONSUMER_MODE == 186 || STATIC_CONSUMER_MODE == 202 ||
         STATIC_CONSUMER_MODE == 204 || STATIC_CONSUMER_MODE == 205 ||
         STATIC_CONSUMER_MODE == 206 || STATIC_CONSUMER_MODE == 207 ||
         STATIC_CONSUMER_MODE == 208 || STATIC_CONSUMER_MODE == 209 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
         STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
         STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
         STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
         STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
         STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
         STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
         STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 225 ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
         STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_SINGLE_WARP_PUBLISH ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_REWRITE);
    constexpr bool STATIC_SKIP_PRE_PACK_READY_SYNC =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
         (STATIC_CONSUMER_MODE == 203 || STATIC_CONSUMER_MODE == 219 ||
        STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 269 ||
        STATIC_CONSUMER_MODE == 372 ||
        STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_SKIP_PRE_PACK_VARIANT ||
         STATIC_P128_ASYNC_PAYLOAD_FAST_PUBLISH);
    constexpr bool STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 204 || STATIC_CONSUMER_MODE == 205 ||
         STATIC_CONSUMER_MODE == 206 || STATIC_CONSUMER_MODE == 207 ||
         STATIC_CONSUMER_MODE == 208 || STATIC_CONSUMER_MODE == 209 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 211 ||
         STATIC_CONSUMER_MODE == 212 || STATIC_CONSUMER_MODE == 213 ||
         STATIC_CONSUMER_MODE == 214 || STATIC_CONSUMER_MODE == 215 ||
         STATIC_CONSUMER_MODE == 216 || STATIC_CONSUMER_MODE == 217 ||
         STATIC_CONSUMER_MODE == 218 || STATIC_CONSUMER_MODE == 219 ||
         STATIC_CONSUMER_MODE == 220 || STATIC_CONSUMER_MODE == 221 ||
         STATIC_CONSUMER_MODE == 222 || STATIC_CONSUMER_MODE == 223 ||
         STATIC_CONSUMER_MODE == 224 || STATIC_CONSUMER_MODE == 269 ||
         STATIC_CONSUMER_MODE == 372 || STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 308 || STATIC_FUSED_SKIP_PREPACK_REWRITE ||
         STATIC_P128_FAST_PUBLISH_REWRITE || STATIC_P128_SMALL_NO_ALIAS_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_CONSUMER_MODE == 225 ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 227 ||
         STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_PUBLISH_REDUCE_VARIANT ||
         STATIC_FULL_K128_P32_BASE_VARIANT ||
         (STATIC_CONSUMER_MODE >= 229 && STATIC_CONSUMER_MODE <= 248) ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_SINGLE_PUBLISH ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_REWRITE);
    constexpr bool STATIC_LIGHT_P_PAYLOAD_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 206 || STATIC_CONSUMER_MODE == 224 ||
         STATIC_FUSED_LIGHT_SKIP_PREPACK_REWRITE ||
         STATIC_CONSUMER_MODE == 269 || STATIC_CONSUMER_MODE == 372 ||
         STATIC_CONSUMER_MODE == 308 ||
         STATIC_P128_FAST_KEEP_LOCAL_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_ASYNC_PAYLOAD_FAST_PUBLISH ||
         STATIC_P128_SINGLE_LOCAL_PRE_PACK_VARIANT ||
         STATIC_CONSUMER_MODE == 236 || STATIC_CONSUMER_MODE == 237 ||
         STATIC_CONSUMER_MODE == 239 || STATIC_CONSUMER_MODE == 240 ||
         STATIC_CONSUMER_MODE == 241 || STATIC_CONSUMER_MODE == 242 ||
         STATIC_CONSUMER_MODE == 243 || STATIC_CONSUMER_MODE == 244 ||
         STATIC_CONSUMER_MODE == 245 || STATIC_CONSUMER_MODE == 246 ||
         STATIC_CONSUMER_MODE == 247 || STATIC_CONSUMER_MODE == 248 ||
         STATIC_FULL_K128_P32_BASE_LIGHT_LOCAL_PUBLISH ||
         STATIC_FULL_K128_LIGHT_LOCAL_PUBLISH_VARIANT ||
         STATIC_EARLY_PV_K64_BYTE_DIRECT_VARIANT ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_SINGLE_PUBLISH ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_REWRITE);
    constexpr bool STATIC_SINGLE_WG_P_PAYLOAD_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 189 || STATIC_CONSUMER_MODE == 190 ||
         STATIC_CONSUMER_MODE == 223 || STATIC_CONSUMER_MODE == 224 ||
         STATIC_FUSED_LIGHT_SKIP_PREPACK_REWRITE ||
         STATIC_CONSUMER_MODE == 269 || STATIC_CONSUMER_MODE == 372 ||
         STATIC_CONSUMER_MODE == 308 ||
         STATIC_P128_FAST_KEEP_LOCAL_REWRITE ||
         STATIC_P128_DIRECT_SCALE_REDESIGN ||
         STATIC_ASYNC_P_SCALE_COPY_P32_FAST_PUBLISH ||
         STATIC_P128_ASYNC_PAYLOAD_FAST_PUBLISH ||
         STATIC_P128_SINGLE_LOCAL_PRE_PACK_VARIANT ||
         STATIC_FULL_K128_P32_BASE_SINGLE_WG_LOCAL_PUBLISH ||
         STATIC_FULL_K128_SINGLE_LOCAL_PUBLISH_VARIANT ||
         STATIC_DIRECT_SOFTMAX_PV_STAGE_SINGLE_PUBLISH ||
         STATIC_DECOUPLED_PV_DIRECT_PUBLISH_REWRITE);
    constexpr bool STATIC_SKIP_P_PAYLOAD_LOCAL_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 191 || STATIC_CONSUMER_MODE == 192 ||
         STATIC_CONSUMER_MODE == 193 || STATIC_CONSUMER_MODE == 194 ||
         STATIC_CONSUMER_MODE == 195 || STATIC_CONSUMER_MODE == 196 ||
         STATIC_CONSUMER_MODE == 210 || STATIC_CONSUMER_MODE == 212 ||
         STATIC_FUSED_SKIP_LOCAL_SKIP_PREPACK_REWRITE ||
         STATIC_CONSUMER_MODE == 226 || STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_CONSUMER_MODE == 230 ||
         STATIC_P128_SMALL_SKIP_LOCAL_REWRITE ||
         STATIC_P128_FAST_SKIP_LOCAL_REWRITE ||
         STATIC_EARLY_PV_K64_DIRECT_SCALE_REWRITE_SKIP_LOCAL ||
         STATIC_P128_DIRECT_SCALE_FOLD_REMOTE_SKIP_LOCAL_REWRITE ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SKIP_LOCAL_PUBLISH_VARIANT);
    constexpr bool STATIC_ASYNC_P_PAYLOAD_CLUSTER_PUBLISH =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 227 || STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_P128_ASYNC_PAYLOAD_FAST_PUBLISH ||
         STATIC_FUSED_PAYLOAD_READY_SKIP_PREPACK_REWRITE ||
         STATIC_FULL_K128_P32_ASYNC_PAYLOAD_CLUSTER_PUBLISH ||
         STATIC_FULL_K128_P32_BASE_ASYNC_PAYLOAD_CLUSTER_PUBLISH);
    constexpr bool STATIC_P_PAYLOAD_PUBLISHED_READY =
        STATIC_P32_PUBLISH_READY_REWRITE ||
        STATIC_FUSED_PAYLOAD_READY_SKIP_PREPACK_REWRITE ||
        STATIC_DECOUPLED_PV_EARLY_SCALE_READY_REWRITE ||
        (MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
         (STATIC_CONSUMER_MODE == 332 || STATIC_CONSUMER_MODE == 333));
    constexpr bool STATIC_SPLIT_PV_MMA_K64 =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 229 || STATIC_CONSUMER_MODE == 230 ||
         STATIC_EARLY_PV_K64_HALF_READY);
    constexpr bool STATIC_QK_TWO_AHEAD =
        (MXFP4_PV && ONLINE && (!EXTERNAL_COL_LSE) &&
         fp4pv_online_qk_two_ahead<C>::value) ||
        (MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
         (STATIC_CONSUMER_MODE == 85 || STATIC_CONSUMER_MODE == 86 ||
          STATIC_CONSUMER_MODE == 96 || STATIC_CONSUMER_MODE == 97 ||
          STATIC_CONSUMER_MODE == 216));
    constexpr bool STATIC_HALF_P_SCALE_CROSSBAR =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 78 || STATIC_EARLY_PV_K64_HALF_SCALE_CROSSBAR);
    constexpr bool STATIC_BYTE_P_SCALE_CROSSBAR =
        STATIC_PACKED_P_SCALE_CROSSBAR || STATIC_DIRECT_P_SCALE_FROM_CROSSBAR ||
        STATIC_EARLY_PV_K64_BYTE_SCALE_CROSSBAR ||
        (STATIC_EARLY_DIRECT_P_SCALE_TMEM && !STATIC_WIDE_P_SCALE_CROSSBAR &&
         !STATIC_HALF_P_SCALE_CROSSBAR);
    constexpr bool STATIC_ANY_PACKED_P_SCALE_CROSSBAR =
        STATIC_BYTE_P_SCALE_CROSSBAR || STATIC_WIDE_P_SCALE_CROSSBAR ||
        STATIC_HALF_P_SCALE_CROSSBAR;
    constexpr bool STATIC_WARP_P_SCALE_CROSSBAR_READY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 225 || STATIC_CONSUMER_MODE == 226 ||
         STATIC_CONSUMER_MODE == 227 || STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT);
    constexpr bool STATIC_WARP_P_PACK_READY =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 225 || STATIC_CONSUMER_MODE == 226 ||
         STATIC_CONSUMER_MODE == 227 || STATIC_CONSUMER_MODE == 228 ||
         STATIC_P128_WARP_ASYNC_STAGE_V_REWRITE ||
         STATIC_P128_WARP_ASYNC_PAYLOAD_VARIANT ||
         STATIC_P128_SPLIT_SCALE_READY_VARIANT ||
         STATIC_FULL_K128_P32_WARP_PACK_READY ||
         STATIC_FULL_K128_P32_BASE_WARP_PACK_READY);
    constexpr bool STATIC_PACKED_P_SCALE_LOCAL_BYPASS =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) && (STATIC_CONSUMER_MODE == 32);
    constexpr bool STATIC_DEFER_P_SCALE_UNTIL_AFTER_V =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 31 || STATIC_CONSUMER_MODE == 39 ||
         STATIC_CONSUMER_MODE == 41 || STATIC_CONSUMER_MODE == 43 ||
         STATIC_CONSUMER_MODE == 113 || STATIC_CONSUMER_MODE == 115 ||
         STATIC_CONSUMER_MODE == 45 || STATIC_CONSUMER_MODE == 47 ||
         STATIC_CONSUMER_MODE == 215);
    constexpr bool STATIC_ASYNC_P_SCALE_TMEM =
        STATIC_QUANT_ASYNC_P_SCALE_TMEM || STATIC_PRODUCER_ASYNC_P_SCALE_AGGREGATE ||
        STATIC_PRODUCER_ASYNC_P_SCALE_COPY || STATIC_EARLY_P_SCALE_TMEM ||
        STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM;
    constexpr bool STATIC_TRIPLE_P_SCALE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 166 || STATIC_CONSUMER_MODE == 167 ||
         STATIC_CONSUMER_MODE == 358 || STATIC_CONSUMER_MODE == 359 ||
         STATIC_CONSUMER_MODE == 360 || STATIC_CONSUMER_MODE == 361 ||
         STATIC_P128_DIRECT_SCALE_TRIPLE_SLOT_REWRITE);
    constexpr bool STATIC_SINGLE_V_SCALE_TMEM =
        MXFP4_PV && EXTERNAL_COL_LSE && (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 166 || STATIC_CONSUMER_MODE == 167 ||
         STATIC_P128_DIRECT_SCALE_TRIPLE_SLOT_REWRITE);
    constexpr int P_SCALE_TMEM_SLOTS =
        STATIC_ASYNC_P_SCALE_TMEM
            ? (STATIC_ALIAS_SCALE_TMEM ? SCORE_TMEM_SLOTS : (STATIC_TRIPLE_P_SCALE_TMEM ? 3 : 2))
            : 1;
    constexpr int V_SCALE_TMEM_SLOTS = STATIC_SINGLE_V_SCALE_TMEM ? 1 : 2;
    constexpr bool STATIC_SPLIT_P_QUANT = STATIC_SPLIT_P_QUANT_2WG || STATIC_SPLIT_P_QUANT_4WG;
    constexpr bool STATIC_PROD_TCGEN_EARLY_P_SCALE = (!ONLINE) &&
        (STATIC_CONSUMER_MODE == 7 || (STATIC_SPLIT_P_QUANT && !STATIC_DEFER_P_SCALE_UNTIL_AFTER_V));
    constexpr bool STATIC_STAGE_P_BEFORE_V =
        STATIC_PROD_TCGEN_EARLY_P_SCALE || STATIC_ONLINE_MXFP4_STAGE_P_BEFORE_V;
    constexpr bool STATIC_PROD_TCGEN_FAMILY =
        STATIC_TCGEN_ACCUM || STATIC_PROD_TCGEN || STATIC_PROD_TCGEN_MM2 ||
        STATIC_PROD_TCGEN_AUTO || STATIC_PROD_TCGEN_SYNC || STATIC_PROD_TCGEN_EARLY_P_SCALE ||
        STATIC_SPLIT_P_QUANT;
    constexpr bool NEED_NONFIRST_MM2_GUARD = STATIC_PROD_TCGEN_AUTO;
    constexpr int P_QUANT_WGS = STATIC_SPLIT_P_QUANT_4WG ? 4 : (STATIC_SPLIT_P_QUANT_2WG ? 2 : 1);
    constexpr int P_PACK_READY_WGS = STATIC_WARP_P_PACK_READY ? (P_QUANT_WGS * 4) : P_QUANT_WGS;
    constexpr int P_SC_CROSSBAR_READY_WGS =
        STATIC_WARP_P_SCALE_CROSSBAR_READY ? (P_QUANT_WGS * 4) : P_QUANT_WGS;
    constexpr int P_QUANT_READY_WGS = STATIC_ANY_PACKED_P_SCALE_CROSSBAR ? 1 : P_QUANT_WGS;
    constexpr int P_REMOTE_READY_WGS =
        (STATIC_ANY_DIRECT_P_SCALE_TMEM_STORE || STATIC_SINGLE_WG_P_PAYLOAD_CLUSTER_PUBLISH)
            ? 1
            : P_QUANT_READY_WGS;
    constexpr int EXTERNAL_LSE_SLOTS = STATIC_SPLIT_P_QUANT_4WG ? 4 : (STATIC_SPLIT_P_QUANT_2WG ? 2 : 1);
    constexpr int PRODUCER_WG =
        STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG ? 3 :
        (STATIC_SPLIT_P_QUANT_4WG ? 5 : (STATIC_SPLIT_P_QUANT_2WG ? 4 : (STATIC_ONLINE_MXFP4_3WG ? 2 : 3)));
    constexpr int OUTPUT_WG =
        STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG ? 0 :
        (STATIC_SPLIT_P_QUANT_4WG ? 4 : (STATIC_SPLIT_P_QUANT_2WG ? 3 : (STATIC_ONLINE_MXFP4_3WG ? 0 : 2)));
    constexpr int QUANT_WG0 = STATIC_SPLIT_P_QUANT_4WG ? 0 : 1;
    constexpr int QUANT_WG1 = STATIC_SPLIT_P_QUANT_4WG ? 1 : (STATIC_SPLIT_P_QUANT_2WG ? 2 : -1);
    constexpr int QUANT_WG2 = STATIC_SPLIT_P_QUANT_4WG ? 2 : -1;
    constexpr int QUANT_WG3 = STATIC_SPLIT_P_QUANT_4WG ? 3 : -1;
    constexpr int RELAY_WG =
        (STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG || STATIC_SPLIT_P_QUANT_4WG || STATIC_ONLINE_MXFP4_3WG) ? -1 : 0;
    constexpr int STATIC_V_PRODUCER_WARPS =
        STATIC_SPLIT_P_QUANT ? 2 : STATIC_ONLINE_MXFP4_V_LOAD_WARPS;
    constexpr bool STATIC_MULTI_WARP_V_PRODUCER = STATIC_V_PRODUCER_WARPS > 1;
    constexpr int STATIC_V_PRODUCER_THREADS = STATIC_V_PRODUCER_WARPS * WARP_THREADS;

    auto get_tile_idx = [&](int block_idx) -> int3 {
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
        int b = bidhb_actual / num_head;
        int h = bidhb_actual - b * num_head;
        m_cluster = num_block - 1 - m_cluster;
        int m_tile_base_cluster = m_cluster * (C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE);
        return {b, m_tile_base_cluster, h};
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
    typename G::v_fp4_tile (&v_fp4_smem)[2] = al.allocate<typename G::v_fp4_tile, 2>();
    typename G::v_sc_tile (&v_sc_smem)[2] = al.allocate<typename G::v_sc_tile, 2>();
    auto (&o_smem)[1] = al.allocate<typename G::o_tile, 1>();
    auto p_bf16_scan = fp4pv_allocate_p_scan<G::NEED_P_SCAN_STORAGE, 2>(al);
    auto (&p_fp4_stage)[P_STAGE_SLOTS] = al.allocate<typename G::p_fp4_tile, P_STAGE_SLOTS>();
    auto (&p_sc_stage)[P_STAGE_SLOTS] = al.allocate<typename G::p_sc_tile, P_STAGE_SLOTS>();
    auto (&p_sc_guard)[1] = al.allocate<typename G::p_sc_guard_slot, 1>();
    (void)p_sc_guard;
    p_fp4_k256_tile *p_fp4_stage_k256 = nullptr;
    p_sc_k256_tile *p_sc_stage_k256 = nullptr;
    v_fp4_k256_tile *v_fp4_smem_k256 = nullptr;
    v_sc_k256_tile *v_sc_smem_k256 = nullptr;
    if constexpr (STATIC_MXFP4_K256) {
        auto (&p_fp4_stage_k256_alloc)[2] = al.allocate<p_fp4_k256_tile, 2>();
        auto (&p_sc_stage_k256_alloc)[4] = al.allocate<p_sc_k256_tile, 4>();
        auto (&v_fp4_smem_k256_alloc)[2] = al.allocate<v_fp4_k256_tile, 2>();
        auto (&v_sc_smem_k256_alloc)[4] = al.allocate<v_sc_k256_tile, 4>();
        p_fp4_stage_k256 = &p_fp4_stage_k256_alloc[0];
        p_sc_stage_k256 = &p_sc_stage_k256_alloc[0];
        v_fp4_smem_k256 = &v_fp4_smem_k256_alloc[0];
        v_sc_smem_k256 = &v_sc_smem_k256_alloc[0];
    }

    __shared__ sv_fl<C::Mb> corr_vec_smem[STATIC_ONLINE_MXFP4_CORR_SLOTS];
    __shared__ sv_fl<C::Mb> max_vec_smem[1];
    __shared__ sv_fl<C::Mb> lse_smem[1];
    __shared__ float external_lse_log2_smem[EXTERNAL_LSE_SLOTS][C::Nb];

    __shared__ semaphore q_arrived[1], q_sc_arrived[1], q_finished[1];
    __shared__ semaphore k_arrived[C::LOAD_STAGES], k_sc_arrived[C::K_SCALE_SMEM_SLOTS], k_finished[C::LOAD_STAGES], k_sc_finished[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore scores_arrived[SCORE_TMEM_SLOTS], corr_arrived[STATIC_ONLINE_MXFP4_CORR_SLOTS], stats_arrived[1], tile_arrived[1], rescale_finished[STATIC_ONLINE_MXFP4_CORR_SLOTS], pv_tmem_ready[1], pv_final_ready[1], output_reusable[1], dual_output_spare_reusable[SCORE_TMEM_SLOTS];
    __shared__ semaphore tt_output_reusable[1], tt_output_remote_reusable[1];
    __shared__ semaphore v_arrived[2], v_finished[2], v_remote_ready[2];
    __shared__ semaphore p_copy_done[P_COPY_DONE_SLOTS], p_pack_ready[P_STAGE_SLOTS], p_scale_ready[P_STAGE_SLOTS], p_scale_published_ready[P_STAGE_SLOTS], p_payload_published_ready[P_STAGE_SLOTS], p_quant_ready[P_STAGE_SLOTS], p_remote_ready[P_STAGE_SLOTS], p_scale_remote_ready[P_STAGE_SLOTS], p_payload_remote_ready[P_STAGE_SLOTS], p_stage_reusable[P_STAGE_SLOTS];
    __shared__ semaphore p_sc_tmem_ready[P_SCALE_TMEM_SLOTS], p_sc_tmem_reusable[P_SCALE_TMEM_SLOTS];
    __shared__ semaphore p_k64_ready[P_STAGE_SLOTS][STATIC_EARLY_PV_K64_HALF_READY ? 2 : 1];
    __shared__ semaphore p_online_k64_ready[P_STAGE_SLOTS][STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64 ? 2 : 1];
    __shared__ semaphore p_online_split_max_ready[STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG ? P_STAGE_SLOTS : 1];
    __shared__ semaphore p_online_split_sum_ready[STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG ? P_STAGE_SLOTS : 1];
    __shared__ semaphore p_k64_stage_ready[P_STAGE_SLOTS][STATIC_EARLY_PV_K64_CROSSBAR_SCALE ? 2 : 1];
    __shared__ semaphore p_k64_scale_slot_ready[STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE ? SCORE_TMEM_SLOTS : 1];
    __shared__ semaphore p_k64_scale_remote_ready[P_STAGE_SLOTS][STATIC_EARLY_PV_K64_DIRECT_SCALE_REMOTE_SYNC ? 2 : 1];
    __shared__ semaphore p_k64_remote_ready[P_STAGE_SLOTS][STATIC_EARLY_PV_K64_HALF_READY ? 2 : 1];
    __shared__ semaphore p_k64_sc_tmem_ready[SCORE_TMEM_SLOTS][STATIC_EARLY_PV_K64_HALF_READY ? 2 : 1];
    __shared__ semaphore v_sc_tmem_ready[V_SCALE_TMEM_SLOTS];
    __shared__ semaphore v_sc_tmem_reusable[V_SCALE_TMEM_SLOTS];
    __shared__ semaphore p_sc_crossbar_ready[STATIC_ANY_PACKED_P_SCALE_CROSSBAR ? P_STAGE_SLOTS : 1];
    __shared__ semaphore producer_v_sync_sem[1];
    __shared__ uint8_t p_sc_crossbar[STATIC_BYTE_P_SCALE_CROSSBAR ? P_STAGE_SLOTS : 1][4][C::Mb];
    __shared__ uint16_t p_sc_crossbar_half[STATIC_HALF_P_SCALE_CROSSBAR ? P_STAGE_SLOTS : 1][4][C::Mb];
    __shared__ uint32_t p_sc_crossbar_wide[STATIC_WIDE_P_SCALE_CROSSBAR ? P_STAGE_SLOTS : 1][4][C::Mb];
    __shared__ uint32_t p_online_mxfp4_p_scale_words[STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM ? P_STAGE_SLOTS : 1][C::Mb];
    __shared__ uint16_t p_sc_k64_half_local[STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE ? P_STAGE_SLOTS : 1][2][C::Mb];
    __shared__ uint16_t p_sc_k64_half_swizzled[STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE_SWIZZLED ? P_STAGE_SLOTS : 1][2][32][4];
    __shared__ float p_softmax_warp_amax[P_STAGE_SLOTS][4];
    __shared__ float p_cta_amax[P_STAGE_SLOTS];
    __shared__ int p_softmax_warp_needs_rescale[P_STAGE_SLOTS][4];
    __shared__ int p_nonfirst_mm2_ok[P_STAGE_SLOTS];
    __shared__ float p_online_split_tile_max[STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG ? P_STAGE_SLOTS : 1][2][C::Mb];
    __shared__ float p_online_split_tile_sum[STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG ? P_STAGE_SLOTS : 1][2][C::Mb];

    if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM) {
        fp4pv_fill_mxfp4_scale_stage_word_swizzled(
            p_sc_stage[0], C::Mb, FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_WORD);
        publish_shared_backing();
        publish_cluster_shared_backing_if_needed<C>();
    }

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<typename G::q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<typename G::k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<typename G::k_sc_tile, dim::ROW>();
        g.o.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        init_semaphore(q_arrived[0], 0, 1);
        init_semaphore(q_sc_arrived[0], 0, 1);
        init_semaphore(q_finished[0], 0, 1);
        #pragma unroll
        for (int i = 0; i < SCORE_TMEM_SLOTS; ++i) {
            init_semaphore(scores_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < STATIC_ONLINE_MXFP4_CORR_SLOTS; ++i) {
            init_semaphore(corr_arrived[i], 0, 4);
            init_semaphore(rescale_finished[i], 0, STATIC_ONLINE_MXFP4_WARP_RESCALE_ARRIVE ? 4 : 1);
        }
        if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
            init_semaphore(stats_arrived[0], 0, 4);
        }
        init_semaphore(tile_arrived[0], 0, 1);
        init_semaphore(pv_tmem_ready[0], 0, 1);
        init_semaphore(pv_final_ready[0], 0, 1);
        init_semaphore(output_reusable[0], 0, 1);
        #pragma unroll
        for (int i = 0; i < SCORE_TMEM_SLOTS; ++i) {
            init_semaphore(dual_output_spare_reusable[i], 0, 1);
        }
        init_semaphore(tt_output_reusable[0], 0, 1);
        init_semaphore(tt_output_remote_reusable[0], 0, 1);
        init_semaphore(v_arrived[0], 0, 1);
        init_semaphore(v_arrived[1], 0, 1);
        init_semaphore(v_finished[0], 0, 1);
        init_semaphore(v_finished[1], 0, 1);
        init_semaphore(v_remote_ready[0], 0, 1);
        init_semaphore(v_remote_ready[1], 0, 1);
        #pragma unroll
        for (int i = 0; i < P_COPY_DONE_SLOTS; ++i) {
            init_semaphore(p_copy_done[i], 0, P_QUANT_WGS);
        }
        #pragma unroll
        for (int i = 0; i < P_STAGE_SLOTS; ++i) {
            init_semaphore(p_pack_ready[i], 0, P_PACK_READY_WGS);
            if constexpr (STATIC_P_PAYLOAD_PUBLISHED_READY) {
                init_semaphore(p_payload_published_ready[i], 0, 1);
            }
            if constexpr (STATIC_P128_SPLIT_SCALE_READY_VARIANT) {
                init_semaphore(p_scale_ready[i], 0, P_PACK_READY_WGS);
                init_semaphore(p_scale_remote_ready[i], 0, 1);
                if constexpr (STATIC_P128_SPLIT_SCALE_LEADER_PUBLISH) {
                    init_semaphore(p_scale_published_ready[i], 0, 1);
                }
            }
            init_semaphore(p_quant_ready[i], 0, P_QUANT_READY_WGS);
            init_semaphore(p_remote_ready[i], 0, P_REMOTE_READY_WGS);
            if constexpr (STATIC_FUSED_PRODUCER_PREFETCH_REMOTE_SKIP_PREPACK_REWRITE) {
                init_semaphore(p_payload_remote_ready[i], 0, 1);
            }
            if constexpr (STATIC_WAIT_P_STAGE_SLOT_REUSE) {
                init_semaphore(p_stage_reusable[i], 1, 0);
            }
        }
        if constexpr (STATIC_EARLY_PV_K64_HALF_READY) {
            constexpr int P_K64_READY_WGS = STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE ? 1 : 2;
            #pragma unroll
            for (int i = 0; i < P_STAGE_SLOTS; ++i) {
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    init_semaphore(p_k64_ready[i][half], 0, P_K64_READY_WGS);
                    if constexpr (STATIC_EARLY_PV_K64_CROSSBAR_SCALE) {
                        init_semaphore(p_k64_stage_ready[i][half], 0, 1);
                    }
                    if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_REMOTE_SYNC) {
                        init_semaphore(p_k64_scale_remote_ready[i][half], 0, 1);
                    }
                    init_semaphore(p_k64_remote_ready[i][half], 0, 1);
                }
            }
            #pragma unroll
            for (int i = 0; i < SCORE_TMEM_SLOTS; ++i) {
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    init_semaphore(p_k64_sc_tmem_ready[i][half], 0, 1);
                }
                if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE) {
                    init_semaphore(p_k64_scale_slot_ready[i], 0, 1);
                }
            }
        }
        if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64) {
            #pragma unroll
            for (int i = 0; i < P_STAGE_SLOTS; ++i) {
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    init_semaphore(p_online_k64_ready[i][half], 0, 1);
                }
            }
        }
        if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG) {
            #pragma unroll
            for (int i = 0; i < P_STAGE_SLOTS; ++i) {
                init_semaphore(p_online_split_max_ready[i], 0, 2);
                init_semaphore(p_online_split_sum_ready[i], 0, 2);
            }
        }
        if constexpr (STATIC_ASYNC_P_SCALE_TMEM) {
            #pragma unroll
            for (int i = 0; i < P_SCALE_TMEM_SLOTS; ++i) {
                init_semaphore(p_sc_tmem_ready[i], 0, 1);
                init_semaphore(p_sc_tmem_reusable[i], 0, 1);
            }
        }
        if constexpr (STATIC_PRODUCER_ASYNC_V_SCALE_TMEM || STATIC_SINGLE_V_SCALE_TMEM) {
            #pragma unroll
            for (int i = 0; i < V_SCALE_TMEM_SLOTS; ++i) {
                init_semaphore(v_sc_tmem_ready[i], 0, 1);
                init_semaphore(v_sc_tmem_reusable[i], 0, 1);
            }
        }
        if constexpr (STATIC_ANY_PACKED_P_SCALE_CROSSBAR) {
            #pragma unroll
            for (int i = 0; i < P_STAGE_SLOTS; ++i) {
                init_semaphore(p_sc_crossbar_ready[i], 0, P_SC_CROSSBAR_READY_WGS);
            }
        }
        if constexpr (NEED_NONFIRST_MM2_GUARD ||
                      STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM) {
            #pragma unroll
            for (int i = 0; i < P_STAGE_SLOTS; ++i) {
                p_nonfirst_mm2_ok[i] = 0;
            }
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_STAGES; ++i) {
            init_semaphore(k_arrived[i], 0, 1);
            init_semaphore(k_finished[i], 0, 1);
            init_semaphore(k_sc_arrived[i], 0, 1);
            init_semaphore(k_sc_finished[i], 0, 1);
        }
        init_semaphore(producer_v_sync_sem[0], 0, STATIC_V_PRODUCER_WARPS);
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    if (warpgroup::groupid() == PRODUCER_WG) {
        if constexpr (STATIC_ONLINE_MXFP4_REG_TUNE && (C::TOTAL_WGS == 3 || C::TOTAL_WGS == 4)) {
            warpgroup::decrease_registers<STATIC_ONLINE_MXFP4_PRODUCER_REGS>();
        } else if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<128>();
        } else if constexpr (STATIC_SPLIT_P_QUANT_4WG) {
            if constexpr (STATIC_REG_TUNE_Q96_P64_O32) {
                warpgroup::decrease_registers<64>();
            } else {
                warpgroup::decrease_registers<80>();
            }
        } else if constexpr (STATIC_SPLIT_P_QUANT_2WG) {
            warpgroup::decrease_registers<96>();
        }
        const bool v_loader_warp =
            (warpgroup::warpid() == 3) ||
            (STATIC_V_PRODUCER_WARPS >= 2 && warpgroup::warpid() == 2) ||
            (STATIC_V_PRODUCER_WARPS >= 3 && warpgroup::warpid() == 0);
        if (v_loader_warp) {
            const bool local_leader = warp::elect_leader();
            const bool qk_loader = (warpgroup::warpid() == 3);
            const int v_producer_lane =
                (STATIC_V_PRODUCER_WARPS >= 3 && warpgroup::warpid() == 0)
                    ? warp::laneid()
                    : ((warpgroup::warpid() - 2) * WARP_THREADS + warp::laneid() +
                       (STATIC_V_PRODUCER_WARPS >= 3 ? WARP_THREADS : 0));
            int producer_v_sync_phase = 0;
            auto producer_v_sync = [&]() {
                if constexpr (STATIC_MULTI_WARP_V_PRODUCER) {
                    if (local_leader) {
                        arrive(producer_v_sync_sem[0]);
                    }
                    wait(producer_v_sync_sem[0], producer_v_sync_phase);
                    producer_v_sync_phase ^= 1;
                    warp::sync();
                } else {
                    warp::sync();
                }
            };
            int k_idx = 0;
            int k_phase = 1;
            int k_sc_idx = 0;
            int k_sc_phase = 0;
            int v_idx = 0;
            int v_phase = 1;
            int q_phase = 1;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                    if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                        continue;
                    }
                }
                int q_iter_begin = 0;
                int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                if constexpr (EXTERNAL_COL_LSE) {
                    q_iter_begin = t_coord.y * SCORE_ITERS_PER_M_TILE;
                    iters_per_task = score_tiles_m - q_iter_begin;
                }
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                const int q_tile_idx = min(t_coord.y + cta_rank, tiles_m - 1);
                if (qk_loader && local_leader && cta_rank < q_tiles_this_cluster) {
                    const coord<typename G::q_tile> q_coord = {t_coord.x, t_coord.z, q_tile_idx, 0};
                    const coord<typename G::q_sc_tile> q_sc_coord = {t_coord.x, q_tile_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0};
                    fp4pv_wait<C>(q_finished[0], q_phase);
                    fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(q_smem[0], g.q, q_coord, q_arrived[0], (uint16_t)(1 << cta_rank), 0);
                    fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(q_sc_smem, 0), g.q_sc, q_sc_coord, q_sc_arrived[0], (uint16_t)(1 << cta_rank), 0);
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const int global_idx = q_iter_begin + idx;
                    const int k_tile_limit = (STATIC_TRIPLE_SCORE_TMEM && C::Nb == 64)
                        ? (g.k.rows() / (C::Nb / C::CLUSTER_SIZE)) - 1
                        : tiles_m - 1;
                    const int k_tile_idx = min(global_idx * C::CLUSTER_SIZE + cta_rank, k_tile_limit);
                    if (qk_loader && local_leader) {
                        const int k_sc_depth_idx = (STATIC_TRIPLE_SCORE_TMEM && C::Nb == 64) ? global_idx : k_tile_idx;
                        const coord<typename G::k_tile> k_coord = {t_coord.x, t_coord.z, k_tile_idx, 0};
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

                    if constexpr (STATIC_MXFP4_K256) {
                        if (qk_loader) {
                            const int pair_buf = (idx >> 1) & 1;
                            const int pair_half = idx & 1;
                            if (local_leader && pair_half == 0 && (cur_bid != blockIdx.x || idx >= 2 * C::LOAD_STAGES)) {
                                wait(v_finished[pair_buf], v_phase);
                            }
                            warp::sync();
                            load_v_fp4_tile_from_global<C>(
                                v_fp4_smem_k256[pair_buf], g, t_coord.x, t_coord.z, global_idx, cta_rank, pair_half * (C::Nb / 2));
                            load_v_sc_prepared_from_global<C>(v_sc_smem_k256[pair_buf * 2 + pair_half], g, t_coord.x, t_coord.z, global_idx);
                            warp::sync();
                            if (pair_half == 1) {
                                __threadfence_block();
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                publish_cluster_shared_backing_if_needed<C>();
                                warp::sync();
                                if (local_leader) {
                                    arrive(v_arrived[pair_buf]);
                                    if constexpr (C::CLUSTER_SIZE == 1) {
                                        arrive(v_remote_ready[pair_buf]);
                                    } else if (cta_rank == 1) {
                                        tma::cluster::arrive(v_remote_ready[pair_buf], 0);
                                    }
                                }
                                if (pair_buf == C::LOAD_STAGES - 1) {
                                    v_phase ^= 1;
                                }
                            }
                        }
                    } else {
                        if (qk_loader && local_leader && (cur_bid != blockIdx.x || idx >= C::LOAD_STAGES)) {
                            wait(v_finished[v_idx], v_phase);
                        }
                        producer_v_sync();
                        if constexpr (STATIC_MULTI_WARP_V_PRODUCER) {
                            load_v_fp4_tile_from_global_producer_group<STATIC_V_PRODUCER_THREADS, C>(
                                v_fp4_smem[v_idx], g, t_coord.x, t_coord.z, global_idx, cta_rank, v_producer_lane);
                            load_v_sc_prepared_from_global_producer_group<STATIC_V_PRODUCER_THREADS, C>(
                                v_sc_smem[v_idx], g, t_coord.x, t_coord.z, global_idx, v_producer_lane);
                        } else {
                            load_v_fp4_tile_from_global<C>(v_fp4_smem[v_idx], g, t_coord.x, t_coord.z, global_idx, cta_rank);
                            load_v_sc_prepared_from_global<C>(v_sc_smem[v_idx], g, t_coord.x, t_coord.z, global_idx);
                        }
                        producer_v_sync();
                        if constexpr (STATIC_ONLINE_MXFP4_3WG) {
                            fp4pv_publish_shared_backing_proxy_only();
                        } else {
                            publish_shared_backing();
                        }
                        publish_cluster_shared_backing_if_needed<C>();
                        producer_v_sync();
                        if (qk_loader && local_leader) {
                            arrive(v_arrived[v_idx]);
                            if constexpr (C::CLUSTER_SIZE == 1) {
                                arrive(v_remote_ready[v_idx]);
                            } else if (cta_rank == 1) {
                                tma::cluster::arrive(v_remote_ready[v_idx], 0);
                            }
                        }
                        v_idx++;
                        if (v_idx == C::LOAD_STAGES) {
                            v_idx = 0;
                            v_phase ^= 1;
                        }
                    }
                }
                q_phase ^= 1;
            }
        } else if (warpgroup::warpid() == 1 && !STATIC_DECOUPLED_PV_ISSUE_REWRITE) {
            if constexpr (STATIC_PRODUCER_ASYNC_V_SCALE_TMEM) {
                constexpr int P_SC_BASE = STATIC_ALIAS_SCALE_TMEM
                    ? 0
                    : SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                constexpr int V_SC_BASE = STATIC_ALIAS_SCALE_TMEM
                    ? 0
                    : P_SC_BASE + P_SCALE_TMEM_SLOTS * P_SCALE_TMEM_WIDTH;
                static_assert(V_SC_BASE + 2 * V_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS, "producer V-scale TMEM ping-pong exceeds TMEM budget");
                d_tt_v_sc v_sc_tm0 = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE + 0 * V_SCALE_TMEM_WIDTH);
                d_tt_v_sc v_sc_tm1 = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE + 1 * V_SCALE_TMEM_WIDTH);
                int v_idx = 0;
                int v_phase = 0;
                int v_remote_phase_mask = 0;
                int v_sc_reuse_phase_mask = 0;
                int v_sc_ready_count = 0;
                if (cta_rank == 0 && warp::elect_leader()) {
                    for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                        int3 t_coord = get_tile_idx(cur_bid);
                        if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                            if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                                continue;
                            }
                        }
                        int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                        if constexpr (EXTERNAL_COL_LSE) {
                            iters_per_task = score_tiles_m - t_coord.y * SCORE_ITERS_PER_M_TILE;
                        }
                        for (int idx = 0; idx < iters_per_task; ++idx) {
                            const int v_buf = v_idx;
                            if (v_sc_ready_count >= 2) {
                                const int reuse_phase = (v_sc_reuse_phase_mask >> v_buf) & 1;
                                wait(v_sc_tmem_reusable[v_buf], reuse_phase);
                                v_sc_reuse_phase_mask ^= (1 << v_buf);
                            }
                            wait(v_arrived[v_buf], v_phase);
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                const int v_remote_phase = (v_remote_phase_mask >> v_buf) & 1;
                                tma::cluster::wait(v_remote_ready[v_buf], v_remote_phase);
                                v_remote_phase_mask ^= (1 << v_buf);
                            }
                            auto &v_sc_tm_cur = v_buf == 0 ? v_sc_tm0 : v_sc_tm1;
                            auto v_sc_tm_sub = v_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                            fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm_sub, v_sc_smem[v_buf]);
                            arrive(v_sc_tmem_ready[v_buf]);
                            ++v_sc_ready_count;
                            v_idx ^= 1;
                            if (v_idx == 0) {
                                v_phase ^= 1;
                            }
                        }
                    }
                }
            } else if constexpr (STATIC_EARLY_PV_K64_HALF_READY) {
                static_assert(STATIC_ALIAS_SCALE_TMEM && SCORE_TMEM_SLOTS == 3,
                    "early K64 P-scale staging expects aliased triple-score TMEM");
                int p_k64_ready_phase_mask[2] = {0, 0};
                int p_k64_stage_ready_phase_mask[2] = {0, 0};
                int p_sc_reuse_phase_mask = 0;
                int p_sc_ready_count = 0;
                for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                        if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                            continue;
                        }
                    }
                    int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                    if constexpr (EXTERNAL_COL_LSE) {
                        iters_per_task = score_tiles_m - t_coord.y * SCORE_ITERS_PER_M_TILE;
                    }
                    if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE) {
                        const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                        if (cta_rank >= q_tiles_this_cluster) {
                            continue;
                        }
                        if (warp::elect_leader()) {
                            for (int idx = 0; idx < iters_per_task; ++idx) {
                                const int p_sc_slot = idx % SCORE_TMEM_SLOTS;
                                if (p_sc_ready_count >= SCORE_TMEM_SLOTS) {
                                    const int reuse_phase = (p_sc_reuse_phase_mask >> p_sc_slot) & 1;
                                    wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                                    p_sc_reuse_phase_mask ^= (1 << p_sc_slot);
                                }
                                arrive(p_k64_scale_slot_ready[p_sc_slot]);
                                ++p_sc_ready_count;
                            }
                        }
                    } else if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_INLINE_REUSE) {
                        // The publishing quant WG claims the aliased scale slot directly.
                    } else if (cta_rank == 0 && warp::elect_leader()) {
                        for (int idx = 0; idx < iters_per_task; ++idx) {
                            const int buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx);
                            const int p_sc_slot = idx % SCORE_TMEM_SLOTS;
                            if (p_sc_ready_count >= SCORE_TMEM_SLOTS) {
                                const int reuse_phase = (p_sc_reuse_phase_mask >> p_sc_slot) & 1;
                                wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                                p_sc_reuse_phase_mask ^= (1 << p_sc_slot);
                            }
                            #pragma unroll
                            for (int half = 0; half < 2; ++half) {
                                if constexpr (STATIC_EARLY_PV_K64_CROSSBAR_SCALE) {
                                    const int p_stage_phase = (p_k64_stage_ready_phase_mask[half] >> buf) & 1;
                                    wait(p_k64_stage_ready[buf][half], p_stage_phase);
                                    p_k64_stage_ready_phase_mask[half] ^= (1 << buf);
                                } else {
                                    const int p_half_phase = (p_k64_ready_phase_mask[half] >> buf) & 1;
                                    wait(p_k64_ready[buf][half], p_half_phase);
                                    p_k64_ready_phase_mask[half] ^= (1 << buf);
                                }
                                const int p_sc_offset = half == 0 ? ALIAS_P_SC_OFFSET : ALIAS_P_SC_K64_HALF1_OFFSET;
                                d_tt_p_sc p_sc_tm_cur = d_tt_p_sc(tm_alloc.get_addr(p_sc_slot * C::Nb + p_sc_offset));
                                auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                                fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[buf]);
                                arrive(p_k64_sc_tmem_ready[p_sc_slot][half]);
                            }
                            ++p_sc_ready_count;
                        }
                    }
                }
            } else if constexpr (STATIC_PRODUCER_ASYNC_P_SCALE_COPY) {
                constexpr int P_SC_BASE = STATIC_ALIAS_SCALE_TMEM
                    ? 0
                    : SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                static_assert(STATIC_ALIAS_SCALE_TMEM || P_SCALE_TMEM_SLOTS == 2 || P_SCALE_TMEM_SLOTS == 3,
                    "non-aliased producer P-scale copy expects two or three TMEM scale slots");
                static_assert(STATIC_ALIAS_SCALE_TMEM ||
                    P_SC_BASE + P_SCALE_TMEM_SLOTS * P_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS,
                    "producer P-scale TMEM ping-pong exceeds TMEM budget");
                d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
                d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
                d_tt_p_sc p_sc_tm2 = tm_alloc.template allocate<d_tt_p_sc>(
                    P_SC_BASE + (P_SCALE_TMEM_SLOTS == 3 ? 2 : 1) * P_SCALE_TMEM_WIDTH);
                auto p_sc_tensor_for_producer_idx = [&](int score_idx) -> d_tt_p_sc {
                    if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                        const int scale_slot = score_idx % SCORE_TMEM_SLOTS;
                        return d_tt_p_sc(tm_alloc.get_addr(scale_slot * C::Nb + ALIAS_P_SC_OFFSET));
                    } else if constexpr (P_SCALE_TMEM_SLOTS == 3) {
                        const int scale_slot = score_idx % P_SCALE_TMEM_SLOTS;
                        return scale_slot == 0 ? p_sc_tm0 : (scale_slot == 1 ? p_sc_tm1 : p_sc_tm2);
                    } else {
                        const int scale_slot = score_idx & 1;
                        return scale_slot == 0 ? p_sc_tm0 : p_sc_tm1;
                    }
                };
                int p_quant_phase_mask = 0;
                int p_remote_phase_mask = 0;
                int p_scale_remote_phase_mask = 0;
                int p_sc_reuse_phase_mask = 0;
                int p_sc_ready_count = 0;
                for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                        if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                            continue;
                        }
                    }
                    int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                    if constexpr (EXTERNAL_COL_LSE) {
                        iters_per_task = score_tiles_m - t_coord.y * SCORE_ITERS_PER_M_TILE;
                    }
                    const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                    const bool wait_remote_p_quant = q_tiles_this_cluster > 1;
                    for (int idx = 0; idx < iters_per_task; ++idx) {
                        const int buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx);
                        const int p_sc_slot = idx % P_SCALE_TMEM_SLOTS;
                        if (cta_rank == 0 && warp::elect_leader() && p_sc_ready_count >= P_SCALE_TMEM_SLOTS) {
                            const int reuse_phase = (p_sc_reuse_phase_mask >> p_sc_slot) & 1;
                            wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                            p_sc_reuse_phase_mask ^= (1 << p_sc_slot);
                        }
                        const int p_quant_phase = (p_quant_phase_mask >> buf) & 1;
                        if constexpr (STATIC_P128_SPLIT_SCALE_LEADER_PUBLISH) {
                            wait(p_scale_published_ready[buf], p_quant_phase);
                        } else if constexpr (STATIC_P128_SPLIT_SCALE_READY_VARIANT) {
                            wait(p_scale_ready[buf], p_quant_phase);
                        } else if constexpr (STATIC_P_PAYLOAD_PUBLISHED_READY) {
                            wait(p_payload_published_ready[buf], p_quant_phase);
                        } else if constexpr (STATIC_EARLY_LOCAL_P_SCALE_READY) {
                            wait(p_pack_ready[buf], p_quant_phase);
                        } else {
                            wait(p_quant_ready[buf], p_quant_phase);
                        }
                        p_quant_phase_mask ^= (1 << buf);
                        if (cta_rank == 0 && warp::elect_leader()) {
                            if (wait_remote_p_quant) {
                                if constexpr (STATIC_P128_SPLIT_SCALE_REMOTE_READY) {
                                    const int p_scale_remote_phase = (p_scale_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(p_scale_remote_ready[buf], p_scale_remote_phase);
                                    p_scale_remote_phase_mask ^= (1 << buf);
                                } else if constexpr (!STATIC_DECOUPLE_P_SCALE_REMOTE_WAIT) {
                                    const int p_remote_phase = (p_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                    p_remote_phase_mask ^= (1 << buf);
                                }
                            }
                            auto p_sc_tm_cur = p_sc_tensor_for_producer_idx(idx);
                            auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                            fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[buf]);
                            if constexpr (STATIC_FUSED_PRODUCER_FOLD_REMOTE_SKIP_PREPACK_REWRITE) {
                                if (wait_remote_p_quant) {
                                    const int p_remote_phase = (p_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                    p_remote_phase_mask ^= (1 << buf);
                                }
                            }
                            arrive(p_sc_tmem_ready[p_sc_slot]);
                            if constexpr (STATIC_FUSED_PRODUCER_PREFETCH_REMOTE_SKIP_PREPACK_REWRITE) {
                                if (wait_remote_p_quant) {
                                    const int p_remote_phase = (p_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                    p_remote_phase_mask ^= (1 << buf);
                                    arrive(p_payload_remote_ready[buf]);
                                }
                            }
                            ++p_sc_ready_count;
                        }
                    }
                }
            } else if constexpr (STATIC_PRODUCER_ASYNC_P_SCALE_AGGREGATE) {
                constexpr int P_SC_BASE = SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                static_assert(P_SC_BASE + 2 * P_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS, "producer P-scale TMEM ping-pong exceeds TMEM budget");
                d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
                d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
                int p_sc_crossbar_phase_mask = 0;
                int p_remote_phase_mask = 0;
                int p_sc_reuse_phase_mask = 0;
                int p_sc_ready_count = 0;
                for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                        if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                            continue;
                        }
                    }
                    int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                    if constexpr (EXTERNAL_COL_LSE) {
                        iters_per_task = score_tiles_m - t_coord.y * SCORE_ITERS_PER_M_TILE;
                    }
                    const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                    const bool wait_remote_p_quant = q_tiles_this_cluster > 1;
                    for (int idx = 0; idx < iters_per_task; ++idx) {
                        const int buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx);
                        if (cta_rank == 0 && warp::elect_leader() && p_sc_ready_count >= 2) {
                            const int reuse_phase = (p_sc_reuse_phase_mask >> buf) & 1;
                            wait(p_sc_tmem_reusable[buf], reuse_phase);
                            p_sc_reuse_phase_mask ^= (1 << buf);
                        }
                        const int p_sc_crossbar_phase = (p_sc_crossbar_phase_mask >> buf) & 1;
                        wait(p_sc_crossbar_ready[buf], p_sc_crossbar_phase);
                        p_sc_crossbar_phase_mask ^= (1 << buf);
                        const int row_in_32 = warp::laneid();
                        if constexpr (STATIC_BANK_FREE_P_SCALE_AGGREGATE_STAGE_STORE) {
                            auto pack_aggregate_scale_row = [&](int scale_row) {
                                return
                                    (p_sc_crossbar_wide[buf][0][scale_row] & 0xffu) |
                                    ((p_sc_crossbar_wide[buf][1][scale_row] & 0xffu) << 8) |
                                    ((p_sc_crossbar_wide[buf][2][scale_row] & 0xffu) << 16) |
                                    ((p_sc_crossbar_wide[buf][3][scale_row] & 0xffu) << 24);
                            };
                            fp4pv_store_mxfp4_scale_word4_swizzled(
                                p_sc_stage[buf],
                                row_in_32,
                                pack_aggregate_scale_row(row_in_32 + 0 * 32),
                                pack_aggregate_scale_row(row_in_32 + 1 * 32),
                                pack_aggregate_scale_row(row_in_32 + 2 * 32),
                                pack_aggregate_scale_row(row_in_32 + 3 * 32));
                        } else {
                            #pragma unroll
                            for (int row_tile = 0; row_tile < 4; ++row_tile) {
                                const int row = row_tile * 32 + row_in_32;
                                const uint32_t packed =
                                    (p_sc_crossbar_wide[buf][0][row] & 0xffu) |
                                    ((p_sc_crossbar_wide[buf][1][row] & 0xffu) << 8) |
                                    ((p_sc_crossbar_wide[buf][2][row] & 0xffu) << 16) |
                                    ((p_sc_crossbar_wide[buf][3][row] & 0xffu) << 24);
                                fp4pv_store_mxfp4_scale_word_swizzled(p_sc_stage[buf], row, packed);
                            }
                        }
                        publish_shared_backing();
                        if constexpr (!STATIC_EARLY_CLUSTER_P_PAYLOAD_PUBLISH) {
                            publish_cluster_shared_backing_if_needed<C>();
                        }
                        warp::sync();
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if (cta_rank == 1 && warp::elect_leader()) {
                                tma::cluster::arrive(p_remote_ready[buf], 0);
                            }
                        }
                        if (cta_rank == 0 && warp::elect_leader()) {
                            if (wait_remote_p_quant) {
                                const int p_remote_phase = (p_remote_phase_mask >> buf) & 1;
                                tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                p_remote_phase_mask ^= (1 << buf);
                            }
                            auto &p_sc_tm_cur = buf == 0 ? p_sc_tm0 : p_sc_tm1;
                            auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                            fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[buf]);
                            arrive(p_sc_tmem_ready[buf]);
                            ++p_sc_ready_count;
                        }
                    }
                }
            }
        } else if (STATIC_DECOUPLED_PV_EARLY_K64_REWRITE &&
                   STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE &&
                   warpgroup::warpid() == 1 && cta_rank != 0 && warp::elect_leader()) {
            int p_sc_reuse_phase_mask = 0;
            int p_sc_ready_count = 0;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                    if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                        continue;
                    }
                }
                int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                if constexpr (EXTERNAL_COL_LSE) {
                    iters_per_task = score_tiles_m - t_coord.y * SCORE_ITERS_PER_M_TILE;
                }
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                if (cta_rank >= q_tiles_this_cluster) {
                    continue;
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const int p_sc_slot = idx % SCORE_TMEM_SLOTS;
                    if (p_sc_ready_count >= SCORE_TMEM_SLOTS) {
                        const int reuse_phase = (p_sc_reuse_phase_mask >> p_sc_slot) & 1;
                        wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                        p_sc_reuse_phase_mask ^= (1 << p_sc_slot);
                    }
                    arrive(p_k64_scale_slot_ready[p_sc_slot]);
                    ++p_sc_ready_count;
                }
            }
        } else if ((warpgroup::warpid() == 0 ||
                    (STATIC_DECOUPLED_PV_ISSUE_REWRITE && warpgroup::warpid() == 1)) &&
                   cta_rank == 0 && warp::elect_leader()) {
            d_tt_scores tt_score_base = allocate_tmem_tile.template operator()<d_tt_scores>(0);
            d_tt_outputs tt_output = allocate_tmem_tile.template operator()<d_tt_outputs>(SCORE_TMEM_WIDTH);
            d_tt_outputs tt_output_spare = allocate_tmem_tile.template operator()<d_tt_outputs>(SCORE_TMEM_WIDTH + C::Dvo);
            d_tt_q_sc_active q_sc_tm = tm_alloc.template allocate<d_tt_q_sc_active>(SCALE_TMEM_BASE);
            d_tt_sc16 k_sc_tm = tm_alloc.template allocate<d_tt_sc16>(SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH);
            d_tt_k_sc_pre k_sc_tm_pre = tm_alloc.template allocate<d_tt_k_sc_pre>(SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH);
            constexpr int P_SC_BASE = STATIC_ALIAS_SCALE_TMEM
                ? 0
                : SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
            constexpr int V_SC_BASE = STATIC_ALIAS_SCALE_TMEM
                ? 0
                : P_SC_BASE + P_SCALE_TMEM_SLOTS * P_SCALE_TMEM_WIDTH;
            static_assert(V_SC_BASE + V_SCALE_TMEM_SLOTS * V_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS, "streaming live scale ping-pong exceeds TMEM budget");
            d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
            d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
            d_tt_p_sc p_sc_tm2 = tm_alloc.template allocate<d_tt_p_sc>(
                P_SC_BASE + (P_SCALE_TMEM_SLOTS == 3 ? 2 : 1) * P_SCALE_TMEM_WIDTH);
            d_tt_v_sc v_sc_tm0 = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE + 0 * V_SCALE_TMEM_WIDTH);
            d_tt_v_sc v_sc_tm1 = tm_alloc.template allocate<d_tt_v_sc>(
                V_SC_BASE + (V_SCALE_TMEM_SLOTS == 2 ? 1 : 0) * V_SCALE_TMEM_WIDTH);
            int k_idx = 0, k_phase = 0, k_sc_slot = 0;
            int k_sc_arrived_phase_mask = 0;
            int p_copy_phase_mask = 0;
            int p_quant_phase_mask = 0;
            int p_payload_ready_phase_mask = 0;
            int p_payload_published_phase_mask = 0;
            int p_remote_phase_mask = 0;
            int p_sc_tmem_ready_phase_mask = 0;
            int p_k64_sc_tmem_ready_phase_mask[2] = {0, 0};
            int p_k64_remote_phase_mask[2] = {0, 0};
            int p_online_k64_ready_phase_mask[2] = {0, 0};
            int p_k64_scale_slot_claim_phase_mask = 0;
            int p_k64_scale_slot_claim_count = 0;
            int p_inline_sc_reuse_phase_mask = 0;
            int p_inline_sc_ready_count = 0;
            int alias_scale_reuse_phase_mask = 0;
            int v_sc_tmem_ready_phase_mask = 0;
            int v_sc_tmem_reuse_phase = 0;
            int v_idx = 0, v_phase = 0;
            int v_remote_phase_mask = 0;
            int output_reuse_phase = 0;
            int tt_output_reuse_phase = 0;
            int tt_output_remote_reuse_phase = 0;
            int online_rescale_ready_phase_mask = 0;
            int dual_output_spare_reuse_phase = 0;
            int dual_output_spare_reuse_ready_count = 0;
            int dual_score_spare_reuse_phase_mask = 0;
            int dual_score_spare_pending_mask = 0;
            int direct_main_rescale_phase = 0;
            const bool qk_issue_lane =
                !STATIC_DECOUPLED_PV_ISSUE_REWRITE || warpgroup::warpid() == 0;
            const bool pv_issue_lane =
                !STATIC_DECOUPLED_PV_ISSUE_REWRITE || warpgroup::warpid() == 1;

            if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM) {
                auto p_sc_tm_sub = p_sc_tm0.template subtile<full_tt_fp8e8m0<16>>(0);
                fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[0]);
            }

            auto wait_score_copy_done = [&](int copy_buf) {
                const int copy_phase = (p_copy_phase_mask >> copy_buf) & 1;
                wait(p_copy_done[copy_buf], copy_phase);
                p_copy_phase_mask ^= (1 << copy_buf);
            };
            auto score_tensor_for_slot = [&](int score_slot) -> d_tt_scores {
                if constexpr (SCORE_TMEM_SLOTS > 1) {
                    return d_tt_scores(tt_score_base.addr + score_slot * C::Nb);
                } else {
                    return tt_score_base;
                }
            };
            auto output_spare_tensor_for_score_idx = [&](int score_idx) -> d_tt_outputs {
                if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM) {
                    const int score_slot = score_idx % SCORE_TMEM_SLOTS;
                    return d_tt_outputs(tt_score_base.addr + score_slot * C::Nb);
                } else {
                    return tt_output_spare;
                }
            };
            auto qk_scale_slot_for_score_idx = [&](int score_idx) -> int {
                if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                    return (score_idx + 1) % SCORE_TMEM_SLOTS;
                } else {
                    return 0;
                }
            };
            auto q_sc_tensor_for_score_idx = [&](int score_idx) -> d_tt_q_sc_active {
                if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                    const int scale_slot = qk_scale_slot_for_score_idx(score_idx);
                    return d_tt_q_sc_active(tt_score_base.addr + scale_slot * C::Nb + ALIAS_Q_SC_OFFSET);
                } else {
                    return q_sc_tm;
                }
            };
            auto k_sc_tensor_for_score_idx = [&](int score_idx) -> d_tt_sc16 {
                if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                    const int scale_slot = qk_scale_slot_for_score_idx(score_idx);
                    return d_tt_sc16(tt_score_base.addr + scale_slot * C::Nb + ALIAS_K_SC_OFFSET);
                } else {
                    return k_sc_tm;
                }
            };
            auto p_sc_tensor_for_score_idx = [&](int score_idx) -> d_tt_p_sc {
                if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM) {
                    return p_sc_tm0;
                } else if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                    const int scale_slot = score_idx % SCORE_TMEM_SLOTS;
                    return d_tt_p_sc(tt_score_base.addr + scale_slot * C::Nb + ALIAS_P_SC_OFFSET);
                } else if constexpr (P_SCALE_TMEM_SLOTS == 3) {
                    const int scale_slot = score_idx % P_SCALE_TMEM_SLOTS;
                    return scale_slot == 0 ? p_sc_tm0 : (scale_slot == 1 ? p_sc_tm1 : p_sc_tm2);
                } else {
                    const int scale_slot = score_idx % P_SCALE_TMEM_SLOTS;
                    return scale_slot == 0 ? p_sc_tm0 : p_sc_tm1;
                }
            };
            auto p_sc_tensor_for_score_idx_k64_half = [&](int score_idx, int half) -> d_tt_p_sc {
                if constexpr (STATIC_EARLY_PV_K64_HALF_READY) {
                    const int scale_slot = score_idx % SCORE_TMEM_SLOTS;
                    const int p_sc_offset = half == 0 ? ALIAS_P_SC_OFFSET : ALIAS_P_SC_K64_HALF1_OFFSET;
                    return d_tt_p_sc(tt_score_base.addr + scale_slot * C::Nb + p_sc_offset);
                } else {
                    return p_sc_tensor_for_score_idx(score_idx);
                }
            };
            auto v_sc_tensor_for_score_idx = [&](int score_idx) -> d_tt_v_sc {
                if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                    const int scale_slot = score_idx % SCORE_TMEM_SLOTS;
                    return d_tt_v_sc(tt_score_base.addr + scale_slot * C::Nb + ALIAS_V_SC_OFFSET);
                } else if constexpr (STATIC_SINGLE_V_SCALE_TMEM) {
                    return v_sc_tm0;
                } else {
                    return (score_idx & 1) == 0 ? v_sc_tm0 : v_sc_tm1;
                }
            };
            auto issue_next_qk = [&](bool final_q_for_task, int copy_buf, int next_idx) {
                if (cta_rank != 0) {
                    return;
                }
                const int score_slot = SCORE_TMEM_SLOTS > 1 ? (next_idx % SCORE_TMEM_SLOTS) : 0;
                auto tt_score = score_tensor_for_slot(score_slot);
                const int next_k_slot = k_idx;
                fp4pv_expect_bytes<C>(k_arrived[next_k_slot], C::CLUSTER_SIZE * sizeof(typename G::k_tile));
                fp4pv_wait<C>(k_arrived[next_k_slot], k_phase);
                fp4pv_expect_bytes<C>(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(typename G::k_sc_tile));
                const int k_sc_phase = (k_sc_arrived_phase_mask >> k_sc_slot) & 1;
                fp4pv_wait<C>(k_sc_arrived[k_sc_slot], k_sc_phase);
                k_sc_arrived_phase_mask ^= (1 << k_sc_slot);
                if constexpr (STATIC_PRESTAGE_K_SCALE_TMEM) {
                    stage_k_scale_tmem_full<C>(k_sc_tm_pre, slot_tile_at(k_sc_smem, k_sc_slot));
                }
                if constexpr (SCORE_TMEM_SLOTS > 1) {
                    if (next_idx >= SCORE_TMEM_SLOTS) {
                        if constexpr (STATIC_WAIT_ALIAS_SCALE_SLOT_REUSE) {
                            const int scale_reuse_phase = (alias_scale_reuse_phase_mask >> score_slot) & 1;
                            wait(p_sc_tmem_reusable[score_slot], scale_reuse_phase);
                            alias_scale_reuse_phase_mask ^= (1 << score_slot);
                        }
                        wait_score_copy_done(score_slot);
                        if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM &&
                                      !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
                            if (next_idx >= SCORE_TMEM_SLOTS + 1) {
                                const int prev_idx = next_idx - SCORE_TMEM_SLOTS;
                                const int prev_buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(prev_idx);
                                if (p_nonfirst_mm2_ok[prev_buf] == 0) {
                                    const int spare_reuse_phase =
                                        (dual_score_spare_reuse_phase_mask >> score_slot) & 1;
                                    wait(dual_output_spare_reusable[score_slot], spare_reuse_phase);
                                    dual_score_spare_reuse_phase_mask ^= (1 << score_slot);
                                }
                            }
                        }
                    }
                } else {
                    wait_score_copy_done(copy_buf);
                }
                if constexpr (STATIC_COMPACT_Q_SCALE_TMEM) {
                    auto q_sc_tm_cur = q_sc_tensor_for_score_idx(next_idx);
                    auto k_sc_tm_cur = k_sc_tensor_for_score_idx(next_idx);
                    fp4pv_issue_qk_chunked_cluster<C>(
                        tt_score, q_smem[0], k_smem[next_k_slot],
                        slot_tile_at(q_sc_smem, 0), slot_tile_at(k_sc_smem, k_sc_slot),
                        q_sc_tm_cur, k_sc_tm_cur, k_finished[next_k_slot]);
                } else if constexpr (STATIC_PRESTAGE_K_SCALE_TMEM) {
                    issue_qk_chunked_qsc_tmem_preloaded_ksc<C>(
                        tt_score, q_smem[0], k_smem[next_k_slot], q_sc_tm, k_sc_tm_pre, k_finished[next_k_slot]);
                } else {
                    fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
                        tt_score, q_smem[0], k_smem[next_k_slot], q_sc_tm,
                        slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[next_k_slot]);
                }
                detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[score_slot]);
                arrive(k_sc_finished[k_sc_slot]);
                fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1);
                if (final_q_for_task) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
                }
                k_idx ^= 1;
                if (k_idx == 0) {
                    k_phase ^= 1;
                }
                k_sc_slot ^= 1;
            };

            auto wait_and_stage_p_sc = [&](int score_idx, int p_buf, bool wait_remote_p_quant) {
                if (cta_rank != 0) {
                    return;
                }
                auto p_sc_tm_cur = p_sc_tensor_for_score_idx(score_idx);
                if constexpr (STATIC_ASYNC_P_SCALE_TMEM) {
                    const int p_sc_slot = STATIC_ALIAS_SCALE_TMEM ? (score_idx % SCORE_TMEM_SLOTS) : (score_idx % P_SCALE_TMEM_SLOTS);
                    if constexpr (STATIC_DECOUPLED_PV_INLINE_P_SCALE_REWRITE) {
                        if (p_inline_sc_ready_count >= P_SCALE_TMEM_SLOTS) {
                            const int reuse_phase = (p_inline_sc_reuse_phase_mask >> p_sc_slot) & 1;
                            wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                            p_inline_sc_reuse_phase_mask ^= (1 << p_sc_slot);
                        }
                        const int p_quant_phase = (p_quant_phase_mask >> p_buf) & 1;
                        wait(p_quant_ready[p_buf], p_quant_phase);
                        p_quant_phase_mask ^= (1 << p_buf);
                        if (wait_remote_p_quant) {
                            const int p_remote_phase = (p_remote_phase_mask >> p_buf) & 1;
                            tma::cluster::wait(p_remote_ready[p_buf], p_remote_phase);
                            p_remote_phase_mask ^= (1 << p_buf);
                        }
                        auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                        fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[p_buf]);
                        ++p_inline_sc_ready_count;
                    } else {
                        const int p_sc_tmem_phase = (p_sc_tmem_ready_phase_mask >> p_sc_slot) & 1;
                        wait(p_sc_tmem_ready[p_sc_slot], p_sc_tmem_phase);
                        p_sc_tmem_ready_phase_mask ^= (1 << p_sc_slot);
                        if constexpr (STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM) {
                            if (wait_remote_p_quant) {
                                const int p_remote_phase = (p_remote_phase_mask >> p_buf) & 1;
                                tma::cluster::wait(p_remote_ready[p_buf], p_remote_phase);
                                p_remote_phase_mask ^= (1 << p_buf);
                            }
                        }
                        if constexpr (STATIC_P128_SPLIT_SCALE_ASYNC_COPY_REWRITE) {
                            const int p_payload_phase = (p_payload_ready_phase_mask >> p_buf) & 1;
                            wait(p_pack_ready[p_buf], p_payload_phase);
                            p_payload_ready_phase_mask ^= (1 << p_buf);
                        }
                        if constexpr (STATIC_DECOUPLED_PV_EARLY_SCALE_READY_REWRITE) {
                            const int p_payload_phase = (p_payload_published_phase_mask >> p_buf) & 1;
                            wait(p_payload_published_ready[p_buf], p_payload_phase);
                            p_payload_published_phase_mask ^= (1 << p_buf);
                        }
                        if constexpr (STATIC_DECOUPLE_P_SCALE_REMOTE_WAIT &&
                                      !STATIC_DECOUPLED_PV_FOLD_REMOTE_READY_REWRITE &&
                                      !STATIC_FUSED_PRODUCER_FOLD_REMOTE_SKIP_PREPACK_REWRITE) {
                            if (wait_remote_p_quant) {
                                const int p_remote_phase = (p_remote_phase_mask >> p_buf) & 1;
                                if constexpr (STATIC_FUSED_PRODUCER_PREFETCH_REMOTE_SKIP_PREPACK_REWRITE) {
                                    wait(p_payload_remote_ready[p_buf], p_remote_phase);
                                } else {
                                    tma::cluster::wait(p_remote_ready[p_buf], p_remote_phase);
                                }
                                p_remote_phase_mask ^= (1 << p_buf);
                            }
                        }
                    }
                } else if constexpr (MXFP4_PV) {
                    if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64) {
                        (void)wait_remote_p_quant;
                        static_assert(STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM,
                            "online split P-ready K64 relies on the fixed P-scale TMEM path");
                    } else {
                        if constexpr (STATIC_P_PAYLOAD_PUBLISHED_READY) {
                            const int p_payload_phase = (p_payload_published_phase_mask >> p_buf) & 1;
                            wait(p_payload_published_ready[p_buf], p_payload_phase);
                            p_payload_published_phase_mask ^= (1 << p_buf);
                        } else {
                            const int p_quant_phase = (p_quant_phase_mask >> p_buf) & 1;
                            wait(p_quant_ready[p_buf], p_quant_phase);
                            p_quant_phase_mask ^= (1 << p_buf);
                        }
                        if (wait_remote_p_quant) {
                            if constexpr (!STATIC_ONLINE_MXFP4_DEFER_P_REMOTE_UNTIL_PV) {
                                const int p_remote_phase = (p_remote_phase_mask >> p_buf) & 1;
                                tma::cluster::wait(p_remote_ready[p_buf], p_remote_phase);
                                p_remote_phase_mask ^= (1 << p_buf);
                            }
                        }
                        if constexpr (!STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM) {
                            auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                            fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[p_buf]);
                        }
                    }
                } else {
                    const int p_quant_phase = (p_quant_phase_mask >> p_buf) & 1;
                    wait(p_quant_ready[p_buf], p_quant_phase);
                    p_quant_phase_mask ^= (1 << p_buf);
                    if (wait_remote_p_quant) {
                        const int p_remote_phase = (p_remote_phase_mask >> p_buf) & 1;
                        tma::cluster::wait(p_remote_ready[p_buf], p_remote_phase);
                        p_remote_phase_mask ^= (1 << p_buf);
                    }
                    #pragma unroll
                    for (int ii = 0; ii < C::PV_SCALE_CHUNKS; ++ii) {
                        auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                        load_q_scale_chunk<C>(p_sc_tm_sub, p_sc_stage[p_buf], ii);
                    }
                }
            };

            auto wait_and_stage_v_sc = [&](int score_idx, int v_buf) {
                auto v_sc_tm_cur = v_sc_tensor_for_score_idx(score_idx);
                if constexpr (STATIC_PRODUCER_ASYNC_V_SCALE_TMEM) {
                    const int v_sc_tmem_phase = (v_sc_tmem_ready_phase_mask >> v_buf) & 1;
                    wait(v_sc_tmem_ready[v_buf], v_sc_tmem_phase);
                    v_sc_tmem_ready_phase_mask ^= (1 << v_buf);
                } else if constexpr (STATIC_SINGLE_V_SCALE_TMEM) {
                    if (score_idx > 0) {
                        wait(v_sc_tmem_reusable[0], v_sc_tmem_reuse_phase);
                        v_sc_tmem_reuse_phase ^= 1;
                    }
                    wait(v_arrived[v_buf], v_phase);
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        const int v_remote_phase = (v_remote_phase_mask >> v_buf) & 1;
                        tma::cluster::wait(v_remote_ready[v_buf], v_remote_phase);
                        v_remote_phase_mask ^= (1 << v_buf);
                    }
                } else {
                    wait(v_arrived[v_buf], v_phase);
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        const int v_remote_phase = (v_remote_phase_mask >> v_buf) & 1;
                        tma::cluster::wait(v_remote_ready[v_buf], v_remote_phase);
                        v_remote_phase_mask ^= (1 << v_buf);
                    }
                }
                if constexpr (MXFP4_PV && !STATIC_PRODUCER_ASYNC_V_SCALE_TMEM) {
                    auto v_sc_tm_sub = v_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                    fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm_sub, v_sc_smem[v_buf]);
                } else if constexpr (!MXFP4_PV) {
                    #pragma unroll
                    for (int ii = 0; ii < C::PV_SCALE_CHUNKS; ++ii) {
                        auto v_sc_tm_sub = v_sc_tm_cur.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                        load_k_scale_chunk<C>(v_sc_tm_sub, v_sc_smem[v_buf], ii);
                    }
                }
            };

            auto wait_and_stage_p_sc_k256 = [&](int tt_buf, int pair_buf, bool wait_remote_p_quant) {
                if constexpr (STATIC_MXFP4_K256) {
                    if (cta_rank != 0) {
                        return;
                    }
                    auto &p_sc_tm_cur = tt_buf == 0 ? p_sc_tm0 : p_sc_tm1;
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int p_quant_phase = (p_quant_phase_mask >> half) & 1;
                        wait(p_quant_ready[half], p_quant_phase);
                        p_quant_phase_mask ^= (1 << half);
                        if (wait_remote_p_quant) {
                            const int p_remote_phase = (p_remote_phase_mask >> half) & 1;
                            tma::cluster::wait(p_remote_ready[half], p_remote_phase);
                            p_remote_phase_mask ^= (1 << half);
                        }
                        auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(half * 16);
                        fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage_k256[pair_buf * 2 + half]);
                    }
                }
            };

            auto wait_and_stage_v_sc_k256 = [&](int tt_buf, int v_buf) {
                if constexpr (STATIC_MXFP4_K256) {
                    auto &v_sc_tm_cur = tt_buf == 0 ? v_sc_tm0 : v_sc_tm1;
                    wait(v_arrived[v_buf], v_phase);
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        const int v_remote_phase = (v_remote_phase_mask >> v_buf) & 1;
                        tma::cluster::wait(v_remote_ready[v_buf], v_remote_phase);
                        v_remote_phase_mask ^= (1 << v_buf);
                    }
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        auto v_sc_tm_sub = v_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(half * 16);
                        fp4pv_load_mxnv_scale_async_cluster<C>(v_sc_tm_sub, v_sc_smem_k256[v_buf * 2 + half]);
                    }
                }
            };

            auto issue_pv = [&](int score_idx, int p_buf, int v_buf, bool is_first, bool is_final, bool wait_remote_tt_output) {
                if (cta_rank != 0) {
                    return;
                }
                auto p_sc_tm_cur = p_sc_tensor_for_score_idx(score_idx);
                auto v_sc_tm_cur = v_sc_tensor_for_score_idx(score_idx);
                const bool dual_accum_direct_main =
                    STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
                    !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_FORCE_SPARE &&
                    !is_first &&
                    (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE ||
                     (p_nonfirst_mm2_ok[p_buf] != 0));
                auto tt_output_cur = [&]() -> d_tt_outputs {
                    if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM) {
                        return (is_first || dual_accum_direct_main)
                            ? tt_output
                            : output_spare_tensor_for_score_idx(score_idx);
                    } else {
                        return tt_output;
                    }
                }();
                if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY &&
                              !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM) {
                    const int corr_slot = score_idx & 1;
                    const int rescale_phase = (online_rescale_ready_phase_mask >> corr_slot) & 1;
                    wait(rescale_finished[corr_slot], rescale_phase);
                    online_rescale_ready_phase_mask ^= (1 << corr_slot);
                }
                if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
                              !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM) {
                    if (!is_first && !dual_accum_direct_main) {
                        if (dual_output_spare_reuse_ready_count > 0) {
                            wait(dual_output_spare_reusable[0], dual_output_spare_reuse_phase);
                            dual_output_spare_reuse_phase ^= 1;
                        }
                        ++dual_output_spare_reuse_ready_count;
                    }
                }
                if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
                    if (dual_accum_direct_main) {
                        wait(rescale_finished[0], direct_main_rescale_phase);
                        direct_main_rescale_phase ^= 1;
                    }
                }
                auto wait_online_p_k64_half = [&](int half) {
                    if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64) {
                        const int p_half_phase = (p_online_k64_ready_phase_mask[half] >> p_buf) & 1;
                        wait(p_online_k64_ready[p_buf][half], p_half_phase);
                        p_online_k64_ready_phase_mask[half] ^= (1 << p_buf);
                    }
                };
                auto issue_pv_stage = [&](bool clear_acc) {
                    if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64) {
                        wait_online_p_k64_half(0);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (clear_acc) {
                            fp4pv_p_stage_ABt_k64_chunk<0, 0, 0, false, C>(
                                tt_output_cur, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                                p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                        } else {
                            fp4pv_p_stage_ABt_k64_chunk<0, 1, 0, false, C>(
                                tt_output_cur, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                                p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                        }
                        wait_online_p_k64_half(1);
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        fp4pv_p_stage_ABt_k64_chunk<1, 1, 2, true, C>(
                            tt_output_cur, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                            p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                    } else if constexpr (STATIC_SPLIT_PV_MMA_K64) {
                        if (clear_acc) {
                            fp4pv_p_stage_ABt_split_k64<C, true>(
                                tt_output_cur, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                                p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                        } else {
                            fp4pv_p_stage_ABt_split_k64<C, false>(
                                tt_output_cur, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                                p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                        }
                    } else if (clear_acc) {
                        fp4pv_mm_p_stage_ABt<C>(
                            tt_output_cur, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                            p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                    } else {
                        fp4pv_mma_p_stage_ABt<C>(
                            tt_output_cur, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                            p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                    }
                };
                if constexpr (STATIC_PROD_TCGEN_FAMILY) {
                    if constexpr (STATIC_PROD_TCGEN_SYNC) {
                        if (!is_first) {
                            wait(tt_output_reusable[0], tt_output_reuse_phase);
                            tt_output_reuse_phase ^= 1;
                            if (wait_remote_tt_output) {
                                tma::cluster::wait(tt_output_remote_reusable[0], tt_output_remote_reuse_phase);
                                tt_output_remote_reuse_phase ^= 1;
                            }
                        }
                    }
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    if (is_first) {
                        issue_pv_stage(true);
                    } else {
                        if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
                            issue_pv_stage(false);
                        } else if constexpr (STATIC_PROD_TCGEN_MM2 || STATIC_PROD_TCGEN_SYNC) {
                            issue_pv_stage(true);
                        } else if constexpr (STATIC_PROD_TCGEN_AUTO) {
                            if (p_nonfirst_mm2_ok[p_buf]) {
                                issue_pv_stage(true);
                            } else {
                                issue_pv_stage(false);
                            }
                        } else {
                            issue_pv_stage(false);
                        }
                    }
                } else {
                    const bool split_accum =
                        STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM ||
                        STATIC_SPLIT_ACCUM ||
                        ((!HAS_STATIC_CONSUMER_MODE) && (!ONLINE) && (consumer_mode == 1));
                    if (split_accum && !is_first) {
                        wait(tt_output_reusable[0], tt_output_reuse_phase);
                        tt_output_reuse_phase ^= 1;
                        if (wait_remote_tt_output) {
                            tma::cluster::wait(tt_output_remote_reusable[0], tt_output_remote_reuse_phase);
                            tt_output_remote_reuse_phase ^= 1;
                        }
                    }
	                    tensor_load_wait();
	                    tensor_before_thread_sync();
                        if constexpr (STATIC_ONLINE_MXFP4_DEFER_P_REMOTE_UNTIL_PV) {
                            if (wait_remote_tt_output) {
                                const int p_remote_phase = (p_remote_phase_mask >> p_buf) & 1;
                                tma::cluster::wait(p_remote_ready[p_buf], p_remote_phase);
                                p_remote_phase_mask ^= (1 << p_buf);
                            }
                        }
	                    if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM) {
	                        if (is_first || !dual_accum_direct_main) {
	                            issue_pv_stage(true);
	                        } else {
	                            issue_pv_stage(false);
	                        }
	                    } else if (split_accum || is_first) {
                        issue_pv_stage(true);
                    } else {
                        issue_pv_stage(false);
                    }
                }
                if constexpr (STATIC_FINAL_PV_READY_ONLY) {
                    if (is_final) {
                        tensor_commit<C::CLUSTER_SIZE>(pv_final_ready[0]);
                    } else {
                        tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[0]);
                    }
                } else {
                    tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[0]);
                }
                if constexpr (STATIC_ANY_TCGEN_P_STAGE_REUSE) {
                    tensor_commit<C::CLUSTER_SIZE>(p_stage_reusable[p_buf]);
                }
                if constexpr (STATIC_ASYNC_P_SCALE_TMEM) {
                    const int p_sc_slot = STATIC_ALIAS_SCALE_TMEM ? (score_idx % SCORE_TMEM_SLOTS) : (score_idx % P_SCALE_TMEM_SLOTS);
                    if constexpr (!STATIC_WAIT_ALIAS_SCALE_SLOT_REUSE) {
                        arrive(p_sc_tmem_reusable[p_sc_slot]);
                        if constexpr (STATIC_ANY_DIRECT_P_SCALE_TMEM_STORE && C::CLUSTER_SIZE > 1) {
                            if (wait_remote_tt_output) {
                                tma::cluster::arrive(p_sc_tmem_reusable[p_sc_slot], 1);
                            }
                        }
                    }
                }
                if constexpr (STATIC_PRODUCER_ASYNC_V_SCALE_TMEM) {
                    arrive(v_sc_tmem_reusable[v_buf]);
                }
                if (is_final) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                }
                v_idx ^= 1;
                if (v_idx == 0) {
                    v_phase ^= 1;
                }
            };

            auto wait_k64_p_half = [&](int score_idx, int p_buf, int half, bool wait_remote_p_quant) -> d_tt_p_sc {
                const int p_sc_slot = score_idx % SCORE_TMEM_SLOTS;
                auto p_sc_tm_cur = p_sc_tensor_for_score_idx_k64_half(score_idx, half);
                if constexpr (STATIC_DECOUPLED_PV_EARLY_K64_REWRITE &&
                              !STATIC_EARLY_PV_K64_DIRECT_SCALE) {
                    const int p_sc_half_phase =
                        (p_k64_sc_tmem_ready_phase_mask[half] >> p_sc_slot) & 1;
                    wait(p_k64_sc_tmem_ready[p_sc_slot][half], p_sc_half_phase);
                    p_k64_sc_tmem_ready_phase_mask[half] ^= (1 << p_sc_slot);
                    auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                    fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[p_buf]);
                } else {
                    const int p_sc_half_phase =
                        (p_k64_sc_tmem_ready_phase_mask[half] >> p_sc_slot) & 1;
                    wait(p_k64_sc_tmem_ready[p_sc_slot][half], p_sc_half_phase);
                    p_k64_sc_tmem_ready_phase_mask[half] ^= (1 << p_sc_slot);
                }
                if (wait_remote_p_quant &&
                    !STATIC_EARLY_PV_K64_PRODUCER_FOLDS_PAYLOAD_REMOTE) {
                    if constexpr (STATIC_EARLY_PV_K64_FULL_PAYLOAD_REMOTE_REWRITE) {
                        if (half == 0) {
                            const int p_remote_phase = (p_remote_phase_mask >> p_buf) & 1;
                            tma::cluster::wait(p_remote_ready[p_buf], p_remote_phase);
                            p_remote_phase_mask ^= (1 << p_buf);
                        }
                    } else {
                        const int p_remote_half_phase = (p_k64_remote_phase_mask[half] >> p_buf) & 1;
                        tma::cluster::wait(p_k64_remote_ready[p_buf][half], p_remote_half_phase);
                        p_k64_remote_phase_mask[half] ^= (1 << p_buf);
                    }
                }
                return p_sc_tm_cur;
            };

            auto claim_k64_scale_slot_for_quant = [&](int score_idx) {
                if constexpr (STATIC_DECOUPLED_PV_EARLY_K64_REWRITE &&
                              STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE) {
                    const int p_sc_slot = score_idx % SCORE_TMEM_SLOTS;
                    if (p_k64_scale_slot_claim_count >= SCORE_TMEM_SLOTS) {
                        const int reuse_phase =
                            (p_k64_scale_slot_claim_phase_mask >> p_sc_slot) & 1;
                        wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                        p_k64_scale_slot_claim_phase_mask ^= (1 << p_sc_slot);
                    }
                    arrive(p_k64_scale_slot_ready[p_sc_slot]);
                    ++p_k64_scale_slot_claim_count;
                }
            };

            auto issue_pv_k64_early = [&](int score_idx, int p_buf, int v_buf, bool is_first, bool is_final, bool wait_remote_p_quant) {
                if constexpr (STATIC_EARLY_PV_K64_HALF_READY) {
                    if (cta_rank != 0) {
                        return;
                    }
                    auto v_sc_tm_cur = v_sc_tensor_for_score_idx(score_idx);

                    auto p_sc_tm_half0 = wait_k64_p_half(score_idx, p_buf, 0, wait_remote_p_quant);
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    if (is_first) {
                        fp4pv_p_stage_ABt_k64_chunk<0, 0, 0, false, C>(
                            tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                            p_sc_tm_half0, v_sc_tm_cur, v_finished[v_buf]);
                    } else {
                        fp4pv_p_stage_ABt_k64_chunk<0, 1, 0, false, C>(
                            tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                            p_sc_tm_half0, v_sc_tm_cur, v_finished[v_buf]);
                    }

                    auto p_sc_tm_half1 = wait_k64_p_half(score_idx, p_buf, 1, wait_remote_p_quant);
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    fp4pv_p_stage_ABt_k64_chunk<1, 1, 2, true, C>(
                        tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_buf],
                        p_sc_tm_half1, v_sc_tm_cur, v_finished[v_buf]);

                    if constexpr (STATIC_FINAL_PV_READY_ONLY) {
                        if (is_final) {
                            tensor_commit<C::CLUSTER_SIZE>(pv_final_ready[0]);
                        } else {
                            tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[0]);
                        }
                    } else {
                        tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[0]);
                    }
                    if constexpr (STATIC_ASYNC_P_SCALE_TMEM) {
                        const int p_sc_slot = STATIC_ALIAS_SCALE_TMEM ? (score_idx % SCORE_TMEM_SLOTS) : (score_idx % P_SCALE_TMEM_SLOTS);
                        if constexpr (!STATIC_WAIT_ALIAS_SCALE_SLOT_REUSE) {
                            arrive(p_sc_tmem_reusable[p_sc_slot]);
                        }
                    }
                    if (is_final) {
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                    }
                    v_idx ^= 1;
                    if (v_idx == 0) {
                        v_phase ^= 1;
                    }
                }
            };

            auto issue_pv_k256 = [&](int tt_buf, int pair_buf, int v_buf, bool is_first, bool is_final, bool wait_remote_tt_output) {
                if constexpr (STATIC_MXFP4_K256) {
                    if (cta_rank != 0) {
                        return;
                    }
                    auto &p_sc_tm_cur = tt_buf == 0 ? p_sc_tm0 : p_sc_tm1;
                    auto &v_sc_tm_cur = tt_buf == 0 ? v_sc_tm0 : v_sc_tm1;
                if constexpr (STATIC_PROD_TCGEN_FAMILY) {
                    if constexpr (STATIC_PROD_TCGEN_SYNC) {
                        if (!is_first) {
                            wait(tt_output_reusable[0], tt_output_reuse_phase);
                                tt_output_reuse_phase ^= 1;
                                if (wait_remote_tt_output) {
                                    tma::cluster::wait(tt_output_remote_reusable[0], tt_output_remote_reuse_phase);
                                    tt_output_remote_reuse_phase ^= 1;
                                }
                            }
                        }
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (is_first) {
                            fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage_k256[pair_buf], v_fp4_smem_k256[v_buf], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                        } else {
                            if constexpr (STATIC_PROD_TCGEN_MM2 || STATIC_PROD_TCGEN_SYNC || STATIC_PROD_TCGEN_AUTO) {
                                fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage_k256[pair_buf], v_fp4_smem_k256[v_buf], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                            } else {
                                fp4pv_mma_cluster_ABt<C>(tt_output, p_fp4_stage_k256[pair_buf], v_fp4_smem_k256[v_buf], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                            }
                        }
                    } else {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if (is_first) {
                            fp4pv_mm_cluster_ABt<C>(tt_output, p_fp4_stage_k256[pair_buf], v_fp4_smem_k256[v_buf], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                        } else {
                            fp4pv_mma_cluster_ABt<C>(tt_output, p_fp4_stage_k256[pair_buf], v_fp4_smem_k256[v_buf], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_buf]);
                        }
                    }
                    tensor_commit<C::CLUSTER_SIZE>(pv_tmem_ready[0]);
                    if (is_final) {
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                    }
                    v_idx ^= 1;
                    if (v_idx == 0) {
                        v_phase ^= 1;
                    }
                }
            };

            for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, ++task_num) {
                int3 t_coord = get_tile_idx(cur_bid);
                if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                    if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                        continue;
                    }
                }
                int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                if constexpr (EXTERNAL_COL_LSE) {
                    iters_per_task = score_tiles_m - t_coord.y * SCORE_ITERS_PER_M_TILE;
                }
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                if (qk_issue_lane) {
                    int q_phase = task_num & 1;
                    if (task_num > 0) {
                        wait(output_reusable[0], output_reuse_phase);
                        output_reuse_phase ^= 1;
                        if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM &&
                                      !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
                            #pragma unroll
                            for (int slot = 0; slot < SCORE_TMEM_SLOTS; ++slot) {
                                if (dual_score_spare_pending_mask & (1 << slot)) {
                                    const int spare_reuse_phase =
                                        (dual_score_spare_reuse_phase_mask >> slot) & 1;
                                    wait(dual_output_spare_reusable[slot], spare_reuse_phase);
                                    dual_score_spare_reuse_phase_mask ^= (1 << slot);
                                }
                            }
                            dual_score_spare_pending_mask = 0;
                        }
                    }
                    fp4pv_expect_bytes<C>(q_arrived[0], q_tiles_this_cluster * sizeof(typename G::q_tile));
                    fp4pv_wait<C>(q_arrived[0], q_phase);
                    fp4pv_expect_bytes<C>(q_sc_arrived[0], q_tiles_this_cluster * sizeof(typename G::q_sc_tile));
                    fp4pv_wait<C>(q_sc_arrived[0], q_phase);
                    if constexpr (!STATIC_COMPACT_Q_SCALE_TMEM) {
                        fp4pv_stage_q_scale_tmem_cluster<C>(q_sc_tm, slot_tile_at(q_sc_smem, 0));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                    }

                    int k_slot = k_idx;
                    fp4pv_expect_bytes<C>(k_arrived[k_slot], C::CLUSTER_SIZE * sizeof(typename G::k_tile));
                    fp4pv_wait<C>(k_arrived[k_slot], k_phase);
                    fp4pv_expect_bytes<C>(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(typename G::k_sc_tile));
                    {
                        const int k_sc_phase = (k_sc_arrived_phase_mask >> k_sc_slot) & 1;
                        fp4pv_wait<C>(k_sc_arrived[k_sc_slot], k_sc_phase);
                        k_sc_arrived_phase_mask ^= (1 << k_sc_slot);
                    }
                    if constexpr (STATIC_COMPACT_Q_SCALE_TMEM) {
                        auto tt_score = score_tensor_for_slot(0);
                        auto q_sc_tm_cur = q_sc_tensor_for_score_idx(0);
                        auto k_sc_tm_cur = k_sc_tensor_for_score_idx(0);
                        fp4pv_issue_qk_chunked_cluster<C>(
                            tt_score, q_smem[0], k_smem[k_slot],
                            slot_tile_at(q_sc_smem, 0), slot_tile_at(k_sc_smem, k_sc_slot),
                            q_sc_tm_cur, k_sc_tm_cur, k_finished[k_slot]);
                    } else {
                        auto tt_score = score_tensor_for_slot(0);
                        fp4pv_issue_qk_chunked_qsc_tmem_cluster<C>(
                            tt_score, q_smem[0], k_smem[k_slot], q_sc_tm,
                            slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[k_slot]);
                    }
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                    arrive(k_sc_finished[k_sc_slot]);
                    fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1);
                    if (iters_per_task == 1) {
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
                    }
                    k_idx ^= 1;
                    if (k_idx == 0) {
                        k_phase ^= 1;
                    }
                    k_sc_slot ^= 1;
                }
                if constexpr (STATIC_QK_TWO_AHEAD) {
                    if (qk_issue_lane && iters_per_task > 1) {
                        issue_next_qk(iters_per_task == 2, 0, 1);
                    }
                }
                if (pv_issue_lane) {
                    zero_output_scratch_issue_lane<C>(tt_output);
                }
                auto run_iteration = [&](int idx, bool is_first, bool is_final) {
                    const int buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx);
                    const bool wait_remote_p =
                        (q_tiles_this_cluster > 1) &&
                        !STATIC_DECOUPLED_PV_EARLY_K64_SKIP_REMOTE_DEBUG;
                    const bool wait_remote_tt_output = (q_tiles_this_cluster > 1);
                    const int v_slot = v_idx;
                    if constexpr (STATIC_DECOUPLED_PV_ISSUE_REWRITE) {
                        if (qk_issue_lane) {
                            if constexpr (STATIC_QK_TWO_AHEAD) {
                                if (idx + 2 < iters_per_task) {
                                    issue_next_qk(idx + 2 == iters_per_task - 1, buf, idx + 2);
                                }
                            } else if (!is_final) {
                                issue_next_qk(idx == iters_per_task - 2, buf, idx + 1);
                            }
                        } else {
                            if constexpr (STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK) {
                                wait_and_stage_v_sc(idx, v_slot);
                            }
                            if constexpr (STATIC_EARLY_PV_K64_HALF_READY) {
                                if constexpr (!STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK) {
                                    wait_and_stage_v_sc(idx, v_slot);
                                }
                                issue_pv_k64_early(idx, buf, v_slot, is_first, is_final, wait_remote_p);
                                if constexpr (STATIC_DECOUPLED_PV_EARLY_K64_REWRITE &&
                                              STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE) {
                                    const int next_claim_idx = idx + SCORE_TMEM_SLOTS;
                                    if (next_claim_idx < iters_per_task) {
                                        claim_k64_scale_slot_for_quant(next_claim_idx);
                                    }
                                }
                            } else {
                                if constexpr (STATIC_STAGE_P_BEFORE_V) {
                                    wait_and_stage_p_sc(idx, buf, wait_remote_p);
                                    if constexpr (!STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK) {
                                        wait_and_stage_v_sc(idx, v_slot);
                                    }
                                } else {
                                    if constexpr (!STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK) {
                                        wait_and_stage_v_sc(idx, v_slot);
                                    }
                                    wait_and_stage_p_sc(idx, buf, wait_remote_p);
                                }
                                issue_pv(idx, buf, v_slot, is_first, is_final, wait_remote_tt_output);
                            }
                        }
                    } else {
                        if constexpr (STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK) {
                            wait_and_stage_v_sc(idx, v_slot);
                        }
                        if constexpr (STATIC_QK_TWO_AHEAD) {
                            if (idx + 2 < iters_per_task) {
                                issue_next_qk(idx + 2 == iters_per_task - 1, buf, idx + 2);
                            }
                        } else if (!is_final) {
                            issue_next_qk(idx == iters_per_task - 2, buf, idx + 1);
                        }
                        if constexpr (STATIC_EARLY_PV_K64_HALF_READY) {
                            if constexpr (!STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK) {
                                wait_and_stage_v_sc(idx, v_slot);
                            }
                            issue_pv_k64_early(idx, buf, v_slot, is_first, is_final, wait_remote_p);
                        } else {
                            if constexpr (STATIC_STAGE_P_BEFORE_V) {
                                wait_and_stage_p_sc(idx, buf, wait_remote_p);
                                if constexpr (!STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK) {
                                    wait_and_stage_v_sc(idx, v_slot);
                                }
                            } else {
                                if constexpr (!STATIC_STAGE_V_SCALE_BEFORE_NEXT_QK) {
                                    wait_and_stage_v_sc(idx, v_slot);
                                }
                                wait_and_stage_p_sc(idx, buf, wait_remote_p);
                            }
                            issue_pv(idx, buf, v_slot, is_first, is_final, wait_remote_tt_output);
                        }
                    }
                };

                if constexpr (STATIC_MXFP4_K256) {
                    for (int pair_start = 0; pair_start < iters_per_task; pair_start += 2) {
                        const int pair_buf = (pair_start >> 1) & 1;
                        const bool wait_remote_p = (q_tiles_this_cluster > 1);
                        const bool wait_remote_tt_output = (q_tiles_this_cluster > 1);
                        const bool is_first = (pair_start == 0);
                        const bool is_final = (pair_start + 2 >= iters_per_task);
                        if (pair_start == 0 && pair_start + 1 < iters_per_task) {
                            issue_next_qk((pair_start + 1) == iters_per_task - 1, 0, pair_start + 1);
                        } else if (pair_start > 0) {
                            issue_next_qk(pair_start == iters_per_task - 1, 1, pair_start);
                            if (pair_start + 1 < iters_per_task) {
                                issue_next_qk((pair_start + 1) == iters_per_task - 1, 0, pair_start + 1);
                            }
                        }
                        const int v_slot = v_idx;
                        wait_and_stage_v_sc_k256(pair_buf, v_slot);
                        wait_and_stage_p_sc_k256(pair_buf, pair_buf, wait_remote_p);
                        issue_pv_k256(pair_buf, pair_buf, v_slot, is_first, is_final, wait_remote_tt_output);
                    }
                } else if (iters_per_task == 1) {
                    if constexpr (STATIC_DECOUPLED_PV_EARLY_K64_REWRITE &&
                                  STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE) {
                        if (pv_issue_lane) {
                            claim_k64_scale_slot_for_quant(0);
                        }
                    }
                    run_iteration(0, true, true);
                } else {
                    if constexpr (STATIC_DECOUPLED_PV_EARLY_K64_REWRITE &&
                                  STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE) {
                        if (pv_issue_lane) {
                            const int initial_claims =
                                iters_per_task < SCORE_TMEM_SLOTS ? iters_per_task : SCORE_TMEM_SLOTS;
                            for (int idx = 0; idx < initial_claims; ++idx) {
                                claim_k64_scale_slot_for_quant(idx);
                            }
                        }
                    }
                    run_iteration(0, true, false);
                    for (int idx = 1; idx < iters_per_task - 1; ++idx) {
                        run_iteration(idx, false, false);
                    }
                    run_iteration(iters_per_task - 1, false, true);
                }
                if constexpr (SCORE_TMEM_SLOTS > 1) {
                    const int tail_begin = iters_per_task > SCORE_TMEM_SLOTS
                        ? iters_per_task - SCORE_TMEM_SLOTS
                        : 0;
                    for (int idx = tail_begin; idx < iters_per_task; ++idx) {
                        if constexpr (STATIC_DECOUPLED_PV_ISSUE_REWRITE) {
                            if (qk_issue_lane) {
                                wait_score_copy_done(idx % SCORE_TMEM_SLOTS);
                            }
                        } else {
                            // The PV issue path has already waited on p_sc_tmem_ready for these
                            // tail slots; that ready signal is sequenced after p_quant_ready, which
                            // is sequenced after p_copy_done.
                            p_copy_phase_mask ^= (1 << (idx % SCORE_TMEM_SLOTS));
                        }
                    }
                }
                if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM &&
                              !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
                    if (qk_issue_lane) {
                        int pending_begin = iters_per_task > SCORE_TMEM_SLOTS
                            ? iters_per_task - SCORE_TMEM_SLOTS
                            : 1;
                        if (pending_begin < 1) {
                            pending_begin = 1;
                        }
                        for (int idx = pending_begin; idx < iters_per_task; ++idx) {
                            const int buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx);
                            if (p_nonfirst_mm2_ok[buf] == 0) {
                                dual_score_spare_pending_mask |= (1 << (idx % SCORE_TMEM_SLOTS));
                            }
                        }
                    }
                }
            }
        }
    } else if (warpgroup::groupid() == OUTPUT_WG) {
        if constexpr (STATIC_ONLINE_MXFP4_REG_TUNE && (C::TOTAL_WGS == 3 || C::TOTAL_WGS == 4)) {
            warpgroup::decrease_registers<STATIC_ONLINE_MXFP4_OUTPUT_REGS>();
        } else if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<48>();
        } else if constexpr (STATIC_SPLIT_P_QUANT_4WG) {
            if constexpr (STATIC_REG_TUNE_Q96_P64_O32) {
                warpgroup::decrease_registers<32>();
            } else {
                warpgroup::decrease_registers<40>();
            }
        } else if constexpr (STATIC_SPLIT_P_QUANT_2WG) {
            warpgroup::decrease_registers<48>();
        }
        d_tt_scores tt_score_base = allocate_tmem_tile.template operator()<d_tt_scores>(0);
        d_tt_outputs tt_output = allocate_tmem_tile.template operator()<d_tt_outputs>(SCORE_TMEM_WIDTH);
        d_tt_outputs tt_output_spare = allocate_tmem_tile.template operator()<d_tt_outputs>(SCORE_TMEM_WIDTH + C::Dvo);
        auto output_spare_tensor_for_score_idx = [&](int score_idx) -> d_tt_outputs {
            if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM) {
                const int score_slot = score_idx % SCORE_TMEM_SLOTS;
                return d_tt_outputs(tt_score_base.addr + score_slot * C::Nb);
            } else {
                return tt_output_spare;
            }
        };
        int end_phase = 0;
        int corr_phase = 0;
        int corr_phase_mask = 0;
        int stats_phase = 0;
        int pv_phase = 0;
        int pv_final_phase = 0;
        auto wait_corr_for_score = [&](int score_idx) -> int {
            if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                const int corr_slot = score_idx & 1;
                const int phase = (corr_phase_mask >> corr_slot) & 1;
                wait(corr_arrived[corr_slot], phase);
                corr_phase_mask ^= (1 << corr_slot);
                return corr_slot;
            } else {
                wait(corr_arrived[0], corr_phase);
                corr_phase ^= 1;
                return 0;
            }
        };
        auto arrive_rescale_finished_for_slot = [&](int corr_slot) {
            if constexpr (STATIC_ONLINE_MXFP4_WARP_RESCALE_ARRIVE) {
                warp::arrive(rescale_finished[corr_slot]);
            } else {
                warpgroup::arrive(rescale_finished[corr_slot]);
            }
        };
        auto arrive_alias_scale_slot_reusable = [&](int score_idx) {
            if constexpr (STATIC_WAIT_ALIAS_SCALE_SLOT_REUSE) {
                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                    if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE) {
                        arrive(p_sc_tmem_reusable[score_idx % SCORE_TMEM_SLOTS]);
                    } else if (cta_rank == 0) {
                        const int p_sc_slot = score_idx % SCORE_TMEM_SLOTS;
                        arrive(p_sc_tmem_reusable[p_sc_slot]);
                        if constexpr (STATIC_ANY_DIRECT_P_SCALE_TMEM_STORE && C::CLUSTER_SIZE > 1) {
                            tma::cluster::arrive(p_sc_tmem_reusable[p_sc_slot], 1);
                        }
                    }
                }
            }
        };
        auto arrive_v_scale_slot_reusable = [&]() {
            if constexpr (STATIC_SINGLE_V_SCALE_TMEM) {
                if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                    arrive(v_sc_tmem_reusable[0]);
                }
            }
        };
        auto arrive_p_stage_slot_reusable = [&](int score_idx) {
            if constexpr (STATIC_WAIT_P_STAGE_SLOT_REUSE) {
                if constexpr (!STATIC_ANY_TCGEN_P_STAGE_REUSE) {
                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                        arrive(p_stage_reusable[score_idx % P_STAGE_SLOTS]);
                    }
                }
            }
        };
        for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
            int3 t_coord = get_tile_idx(cur_bid);
            if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                    continue;
                }
            }
            const bool row_tile_valid = (t_coord.y + cta_rank) < tiles_m;
            if (!row_tile_valid) {
                continue;
            }
            const float scale_log2 = qk_scale_log2(g, t_coord.x, t_coord.z);
            int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
            if constexpr (EXTERNAL_COL_LSE) {
                iters_per_task = score_tiles_m - t_coord.y * SCORE_ITERS_PER_M_TILE;
            }
            if constexpr (STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM || (!ONLINE && !STATIC_PROD_TCGEN_FAMILY)) {
                if (STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM ||
                    STATIC_SPLIT_ACCUM ||
                    ((!HAS_STATIC_CONSUMER_MODE) && (!ONLINE) && (consumer_mode == 1))) {
                    int clipped_begin = k_iter_begin < 0 ? 0 : k_iter_begin;
                    int clipped_end = (k_iter_end < 0 || k_iter_end > iters_per_task) ? iters_per_task : k_iter_end;
                    if (clipped_begin > clipped_end) {
                        clipped_begin = clipped_end;
                    }

                    wait(corr_arrived[0], corr_phase);
                    arrive_rescale_finished_for_slot(0);
                    corr_phase ^= 1;
                    zero_output_tile_streaming<C>(o_smem[0]);

                    for (int idx = 0; idx + 1 < iters_per_task; ++idx) {
                        wait(pv_tmem_ready[0], pv_phase);
                        pv_phase ^= 1;
                        arrive_alias_scale_slot_reusable(idx);
                        arrive_v_scale_slot_reusable();
                        arrive_p_stage_slot_reusable(idx);
                        if (idx >= clipped_begin && idx < clipped_end) {
                            accumulate_tt_output_into_tile_streaming<C>(o_smem[0], tt_output);
                        }
                        warpgroup::sync(warpgroup::groupid() + 1);
                        if (idx + 1 < iters_per_task) {
                            zero_output_scratch_warp<C>(tt_output);
                            if constexpr (C::CLUSTER_SIZE == 1) {
                                warpgroup::arrive(tt_output_reusable[0]);
                            } else if (cta_rank == 0) {
                                warpgroup::arrive(tt_output_reusable[0]);
                            } else {
                                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    tma::cluster::arrive(tt_output_remote_reusable[0], 0);
                                }
                            }
                            wait(corr_arrived[0], corr_phase);
                            float correction;
                            if constexpr (ONLINE) {
                                correction = stats_load(corr_vec_smem[0], warpgroup::laneid());
                            } else {
                                correction = stats_load(max_vec_smem[0], warpgroup::laneid());
                            }
                            float correction_threshold = 1.0f;
                            if constexpr (MXFP4_PV && ONLINE) {
                                correction_threshold = 0.999f;
                            }
                            bool needs_rescale = __any_sync(0xFFFFFFFF, correction < correction_threshold);
                            if (needs_rescale) {
                                rescale_output_tile_streaming<C>(o_smem[0], correction);
                            }
                            arrive_rescale_finished_for_slot(0);
                            corr_phase ^= 1;
                        }
                    }

                    wait(corr_arrived[0], corr_phase);
                    wait(pv_tmem_ready[0], pv_phase);
                    pv_phase ^= 1;
                    const int final_idx = iters_per_task - 1;
                    arrive_alias_scale_slot_reusable(final_idx);
                    arrive_v_scale_slot_reusable();
                    arrive_p_stage_slot_reusable(final_idx);
                    if (final_idx >= clipped_begin && final_idx < clipped_end) {
                        accumulate_tt_output_into_tile_streaming<C>(o_smem[0], tt_output);
                    }
                    warpgroup::sync(warpgroup::groupid() + 1);
                    float row_sum = stats_load(max_vec_smem[0], warpgroup::laneid());
                    float row_max = stats_load(lse_smem[0], warpgroup::laneid());
                    arrive_rescale_finished_for_slot(0);
                    bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
                    if constexpr (EXTERNAL_COL_LSE) {
                        row_invalid = false;
                        row_sum = 1.0f;
                        row_max = 0.0f;
                    }
                    float inv_norm_s;
                    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_norm_s) : "f"(row_invalid ? 1.0f : row_sum));
                    float2 inv_norm = {inv_norm_s, inv_norm_s};
                    if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                        inv_norm = {g.p_output_scale, g.p_output_scale};
                    }
                    constexpr float LN2 = 0.693147180559945f;
                    float lse_val;
                    if (!row_invalid) {
                        float log2_row_sum;
                        asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                        lse_val = (row_max * scale_log2 + log2_row_sum) * LN2;
                    } else {
                        lse_val = base_types::constants<float>::neg_infty();
                    }
                    wait(tile_arrived[0], end_phase);
                    warpgroup::tma::store_async_read_wait<0>();
                    warpgroup::sync(warpgroup::groupid() + 1);
                    const int row = warpgroup::laneid();

                    for (int col = 0; col < C::Dvo; col += 16) {
                        #pragma unroll
                        for (int i = 0; i < 8; i += 2) {
                            float2 o_reg0 = {
                                __bfloat162float(o_smem[0][{row, col + i * 2 + 0}]),
                                __bfloat162float(o_smem[0][{row, col + i * 2 + 1}])
                            };
                            float2 o_reg1 = {
                                __bfloat162float(o_smem[0][{row, col + i * 2 + 2}]),
                                __bfloat162float(o_smem[0][{row, col + i * 2 + 3}])
                            };
                            if (row_invalid) {
                                o_reg0 = float2{0.0f, 0.0f};
                                o_reg1 = float2{0.0f, 0.0f};
                            }
                            bf16_2 tmp0;
                            bf16_2 tmp1;
                            if constexpr (MXFP4_PV && ONLINE) {
                                constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                                const float output_scale = (MXFP4_ALPHA / FP4PV_MXFP4_ONLINE_P_SCALE) * inv_norm.x;
                                const float2 p_scale = {output_scale, output_scale};
                                tmp0 = __float22bfloat162_rn(__fmul2_rn(o_reg0, p_scale));
                                tmp1 = __float22bfloat162_rn(__fmul2_rn(o_reg1, p_scale));
                            } else {
                                tmp0 = __float22bfloat162_rn(__fmul2_rn(o_reg0, inv_norm));
                                tmp1 = __float22bfloat162_rn(__fmul2_rn(o_reg1, inv_norm));
                            }
                            o_smem[0][{row, col + i * 2 + 0}] = tmp0.x;
                            o_smem[0][{row, col + i * 2 + 1}] = tmp0.y;
                            o_smem[0][{row, col + i * 2 + 2}] = tmp1.x;
                            o_smem[0][{row, col + i * 2 + 3}] = tmp1.y;
                        }
                    }
                    warpgroup::sync(warpgroup::groupid() + 1);
                    warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, o_smem[0], {t_coord.x, t_coord.y + cta_rank, t_coord.z, 0});
                    warpgroup::arrive(output_reusable[0]);
                    if constexpr (!EXTERNAL_COL_LSE) {
                        g.lse[{t_coord.x, t_coord.z, 0, (t_coord.y + cta_rank) * C::Mb + warpgroup::laneid()}] = lse_val;
                    }
                    corr_phase ^= 1;
                    end_phase ^= 1;
                    continue;
                }
            }
            if constexpr (!EXTERNAL_COL_LSE) {
                if constexpr (!(STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE &&
                                !STATIC_FINAL_PV_READY_ONLY &&
                                !STATIC_MXFP4_K256)) {
                    const int corr_slot = wait_corr_for_score(0);
                    arrive_rescale_finished_for_slot(corr_slot);
                }
            }
            if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
                          !STATIC_FINAL_PV_READY_ONLY &&
                          !STATIC_MXFP4_K256 &&
                          !EXTERNAL_COL_LSE) {
                auto wait_pv_and_release = [&](int score_idx) {
                    wait(pv_tmem_ready[0], pv_phase);
                    pv_phase ^= 1;
                    arrive_alias_scale_slot_reusable(score_idx);
                    arrive_v_scale_slot_reusable();
                    arrive_p_stage_slot_reusable(score_idx);
                };
                if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
                    auto rescale_main_output_for_corr = [&](int corr_slot) {
                        const bool output_row_active = warpgroup::laneid() < C::Mb;
                        const float correction = output_row_active
                            ? stats_load(corr_vec_smem[corr_slot], warpgroup::laneid())
                            : 1.0f;
                        const bool lane_needs_rescale = correction < 0.999f;
                        const bool needs_rescale = __any_sync(0xFFFFFFFF, lane_needs_rescale);
                        if (needs_rescale) {
                            rescale_tt_output_inplace_direct_streaming<C>(tt_output, correction, lane_needs_rescale);
                        }
                        if constexpr (!STATIC_ONLINE_MXFP4_WARP_RESCALE_ARRIVE) {
                            warpgroup::sync(warpgroup::groupid() + 1);
                        }
                    };
                    (void)wait_corr_for_score(0);
                    wait_pv_and_release(0);
                    for (int idx = 1; idx < iters_per_task; ++idx) {
                        const int corr_slot = wait_corr_for_score(idx);
                        rescale_main_output_for_corr(corr_slot);
                        arrive_rescale_finished_for_slot(corr_slot);
                        wait_pv_and_release(idx);
                    }
                } else {
                    wait_pv_and_release(0);
                    for (int idx = 1; idx < iters_per_task; ++idx) {
                        const int corr_slot = wait_corr_for_score(idx);
                        wait_pv_and_release(idx);
                        const bool output_row_active = warpgroup::laneid() < C::Mb;
                        const float correction = output_row_active
                            ? stats_load(corr_vec_smem[corr_slot], warpgroup::laneid())
                            : 1.0f;
                        const int buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx);
                        if (p_nonfirst_mm2_ok[buf] == 0) {
                            const bool lane_needs_rescale = correction < 0.999f;
                            auto tt_output_spare_cur = output_spare_tensor_for_score_idx(idx);
                            merge_dual_tt_output_accum_streaming<C>(
                                tt_output, tt_output_spare_cur, correction, lane_needs_rescale);
                            const int spare_reuse_slot = STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SCORE_TMEM
                                ? (idx % SCORE_TMEM_SLOTS)
                                : 0;
                            warpgroup::arrive(dual_output_spare_reusable[spare_reuse_slot]);
                        }
                        if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                            arrive_rescale_finished_for_slot(corr_slot);
                        }
                        if constexpr (STATIC_ONLINE_MXFP4_P_READY_BEFORE_RESCALE_WAIT) {
                            arrive_rescale_finished_for_slot(corr_slot);
                        }
                    }
                }
            } else if constexpr (!STATIC_FINAL_PV_READY_ONLY) {
                for (int idx = 1; idx < iters_per_task; ++idx) {
                    int corr_slot = 0;
                    if constexpr (!EXTERNAL_COL_LSE) {
                        corr_slot = wait_corr_for_score(idx);
                    }
                    if constexpr (STATIC_MXFP4_K256) {
                        if constexpr (!EXTERNAL_COL_LSE) {
                            arrive_rescale_finished_for_slot(0);
                            corr_phase ^= 1;
                        }
                        if ((idx & 1) == 1) {
                            wait(pv_tmem_ready[0], pv_phase);
                            pv_phase ^= 1;
                        }
                        continue;
                    } else {
                        wait(pv_tmem_ready[0], pv_phase);
                        pv_phase ^= 1;
                        arrive_alias_scale_slot_reusable(idx - 1);
                        arrive_v_scale_slot_reusable();
                        arrive_p_stage_slot_reusable(idx - 1);
                    }
                    if constexpr (!EXTERNAL_COL_LSE) {
                        float correction = 1.0f;
                        if (warpgroup::laneid() < C::Mb) {
                            correction = ONLINE
                                ? stats_load(corr_vec_smem[corr_slot], warpgroup::laneid())
                                : stats_load(max_vec_smem[0], warpgroup::laneid());
                        }
                        float correction_threshold = 1.0f;
                        if constexpr (!ONLINE && (STATIC_CONSUMER_MODE == 3 || STATIC_CONSUMER_MODE == 4 || STATIC_CONSUMER_MODE == 5)) {
                            correction_threshold = 0.999f;
                        } else if constexpr (MXFP4_PV && ONLINE) {
                            correction_threshold = 0.999f;
                        }
                        bool lane_needs_rescale = correction < correction_threshold;
                        bool needs_rescale = __any_sync(0xFFFFFFFF, lane_needs_rescale);
                        if (needs_rescale) {
                            #pragma unroll
                            for (int col = 0; col < C::Dvo; col += 16) {
                                float2 o_reg[8];
                                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                                      "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                                      "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                                      "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                                    : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                                tensor_load_wait();
                                if (lane_needs_rescale) {
                                    float2 corr_2 = {correction, correction};
                                    #pragma unroll
                                    for (int ii = 0; ii < 8; ++ii) {
                                        o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
                                    }
                                }
                                asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                                    :: "f"(o_reg[0].x), "f"(o_reg[0].y), "f"(o_reg[1].x), "f"(o_reg[1].y),
                                       "f"(o_reg[2].x), "f"(o_reg[2].y), "f"(o_reg[3].x), "f"(o_reg[3].y),
                                       "f"(o_reg[4].x), "f"(o_reg[4].y), "f"(o_reg[5].x), "f"(o_reg[5].y),
                                       "f"(o_reg[6].x), "f"(o_reg[6].y), "f"(o_reg[7].x), "f"(o_reg[7].y),
                                       "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                            }
                            tensor_store_wait();
                        }
                        if constexpr (!STATIC_ONLINE_MXFP4_WARP_RESCALE_ARRIVE) {
                            warpgroup::sync(warpgroup::groupid() + 1);
                        }
                        if constexpr (STATIC_PROD_TCGEN_SYNC) {
                            if constexpr (C::CLUSTER_SIZE == 1) {
                                warpgroup::arrive(tt_output_reusable[0]);
                            } else if (cta_rank == 0) {
                                warpgroup::arrive(tt_output_reusable[0]);
                            } else if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                tma::cluster::arrive(tt_output_remote_reusable[0], 0);
                            }
                        }
                        arrive_rescale_finished_for_slot(corr_slot);
                    }
                }
            }

            if constexpr (!EXTERNAL_COL_LSE) {
                if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                    wait(stats_arrived[0], stats_phase);
                    stats_phase ^= 1;
                } else {
                    wait(corr_arrived[0], corr_phase);
                    corr_phase ^= 1;
                }
            }
            if constexpr (STATIC_FINAL_PV_READY_ONLY) {
                wait(pv_final_ready[0], pv_final_phase);
                pv_final_phase ^= 1;
            } else if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
                                 !STATIC_MXFP4_K256 &&
                                 !EXTERNAL_COL_LSE) {
                // The dual-accumulator path drains every PV tile before final stats.
            } else if constexpr (!STATIC_MXFP4_K256) {
                wait(pv_tmem_ready[0], pv_phase);
                pv_phase ^= 1;
                arrive_alias_scale_slot_reusable(iters_per_task - 1);
                arrive_v_scale_slot_reusable();
                arrive_p_stage_slot_reusable(iters_per_task - 1);
            }
            float row_sum = 1.0f;
            float row_max = 0.0f;
            bool row_invalid = false;
            const int output_row = warpgroup::laneid();
            const bool output_row_active = output_row < C::Mb;
            if constexpr (!EXTERNAL_COL_LSE) {
                if (output_row_active) {
                    row_sum = stats_load(max_vec_smem[0], output_row);
                    row_max = stats_load(lse_smem[0], output_row);
                }
                if constexpr (!STATIC_ONLINE_MXFP4_EARLY_P_READY &&
                              !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM) {
                    arrive_rescale_finished_for_slot(0);
                }
                row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
            }
            float2 inv_norm = {1.0f, 1.0f};
            float lse_val = 0.0f;
            if constexpr (!EXTERNAL_COL_LSE) {
                float inv_norm_s;
                asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_norm_s) : "f"(row_invalid ? 1.0f : row_sum));
                inv_norm = {inv_norm_s, inv_norm_s};
                const bool should_store_lse = [&]() {
                    if constexpr (MXFP4_PV) {
                        return g.disable_lse_store == 0;
                    } else {
                        return true;
                    }
                }();
                if (should_store_lse) {
                    constexpr float LN2 = 0.693147180559945f;
                    if (!row_invalid) {
                        float log2_row_sum;
                        asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                        lse_val = (row_max * scale_log2 + log2_row_sum) * LN2;
                    } else {
                        lse_val = base_types::constants<float>::neg_infty();
                    }
                }
            }
            wait(tile_arrived[0], end_phase);
            warpgroup::tma::store_async_read_wait<0>();
            warpgroup::sync(warpgroup::groupid() + 1);
            constexpr int SUBTILE_COLS = G::o_tile::swizzle_bytes / sizeof(bf16);
            constexpr uint32_t SWIZZLE_MASK = G::o_tile::swizzle_bytes * 8 - 1;
            uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
            uint32_t row_offset = output_row * SUBTILE_COLS * sizeof(bf16);

            if (output_row_active) {
                for (int col = 0; col < C::Dvo; col += 16) {
                    float2 o_reg[8];
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                        : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                          "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                          "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                          "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                        : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                    tensor_load_wait();
                    uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16))
                                      + row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
                    #pragma unroll
                    for (int i = 0; i < 8; i += 4) {
                        if (row_invalid) {
                            o_reg[i] = float2{0.0f, 0.0f};
                            o_reg[i + 1] = float2{0.0f, 0.0f};
                            o_reg[i + 2] = float2{0.0f, 0.0f};
                            o_reg[i + 3] = float2{0.0f, 0.0f};
                        }
                        bf16_2 tmp0;
                        bf16_2 tmp1;
                        bf16_2 tmp2;
                        bf16_2 tmp3;
                        if constexpr (MXFP4_PV && ONLINE) {
                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                            const float output_scale = (MXFP4_ALPHA / FP4PV_MXFP4_ONLINE_P_SCALE) * inv_norm.x;
                            const float2 p_scale = {output_scale, output_scale};
                            tmp0 = __float22bfloat162_rn(__fmul2_rn(o_reg[i], p_scale));
                            tmp1 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 1], p_scale));
                            tmp2 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 2], p_scale));
                            tmp3 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 3], p_scale));
                        } else if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
                            const float output_scale = g.p_output_scale;
                            const float2 p_scale = {output_scale * MXFP4_ALPHA, output_scale * MXFP4_ALPHA};
                            tmp0 = __float22bfloat162_rn(__fmul2_rn(o_reg[i], p_scale));
                            tmp1 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 1], p_scale));
                            tmp2 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 2], p_scale));
                            tmp3 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 3], p_scale));
                        } else if constexpr (ONLINE || EXTERNAL_COL_LSE) {
                            tmp0 = __float22bfloat162_rn(o_reg[i]);
                            tmp1 = __float22bfloat162_rn(o_reg[i + 1]);
                            tmp2 = __float22bfloat162_rn(o_reg[i + 2]);
                            tmp3 = __float22bfloat162_rn(o_reg[i + 3]);
                        } else {
                            tmp0 = __float22bfloat162_rn(__fmul2_rn(o_reg[i], inv_norm));
                            tmp1 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 1], inv_norm));
                            tmp2 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 2], inv_norm));
                            tmp3 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 3], inv_norm));
                        }
                        uint32_t addr = row_base + i * 4;
                        uint32_t swizzled_addr = addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4);
                        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                            "r"(swizzled_addr),
                            "r"(*(uint32_t *)&tmp0),
                            "r"(*(uint32_t *)&tmp1),
                            "r"(*(uint32_t *)&tmp2),
                            "r"(*(uint32_t *)&tmp3));
                    }
                }
            }
            warpgroup::sync(warpgroup::groupid() + 1);
            warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, o_smem[0], {t_coord.x, t_coord.y + cta_rank, t_coord.z, 0});
            warpgroup::arrive(output_reusable[0]);
            if constexpr (!EXTERNAL_COL_LSE) {
                const bool should_store_lse = [&]() {
                    if constexpr (MXFP4_PV) {
                        return g.disable_lse_store == 0;
                    } else {
                        return true;
                    }
                }();
                if (should_store_lse && output_row_active) {
                    g.lse[{t_coord.x, t_coord.z, 0, (t_coord.y + cta_rank) * C::Mb + output_row}] = lse_val;
                }
            }
            end_phase ^= 1;
        }
        warpgroup::tma::store_async_wait<0>();
    } else if (warpgroup::groupid() == QUANT_WG0 || warpgroup::groupid() == QUANT_WG1 ||
               warpgroup::groupid() == QUANT_WG2 || warpgroup::groupid() == QUANT_WG3) {
        if constexpr (STATIC_ONLINE_MXFP4_REG_TUNE && (C::TOTAL_WGS == 3 || C::TOTAL_WGS == 4)) {
            warpgroup::increase_registers<STATIC_ONLINE_MXFP4_QUANT_REGS>();
        } else if constexpr (C::TOTAL_WGS == 4) {
            warpgroup::increase_registers<168>();
        } else if constexpr (STATIC_SPLIT_P_QUANT_2WG) {
            if constexpr (STATIC_CONSUMER_MODE == 240 || STATIC_CONSUMER_MODE == 242 ||
                          STATIC_CONSUMER_MODE == 248) {
                warpgroup::increase_registers<144>();
            } else if constexpr (STATIC_CONSUMER_MODE == 241 || STATIC_CONSUMER_MODE == 243) {
                warpgroup::increase_registers<136>();
            } else {
                warpgroup::increase_registers<152>();
            }
        } else if constexpr (STATIC_SPLIT_P_QUANT_4WG) {
            if constexpr (STATIC_REG_TUNE_Q96_P64_O32) {
                warpgroup::increase_registers<96>();
            } else {
                warpgroup::increase_registers<88>();
            }
        }
        auto quant_wg_sync = []() {
            if constexpr (STATIC_SPLIT_P_QUANT) {
                warpgroup::sync(warpgroup::groupid() + 1);
            } else {
                tk_localcta::subgroup_barrier_sync<128>();
            }
        };
        d_tt_scores tt_scores = allocate_tmem_tile.template operator()<d_tt_scores>(0);
        int scores_phase_mask = 0;
        int rescale_phase = 1;
        constexpr int SPLIT_2WG_BLOCKS_PER_WG = C::Nb == 64 ? 1 : 2;
        constexpr int SPLIT_2WG_SECOND_BEGIN = C::Nb == 64 ? 1 : 2;
        const int split_lse_slot = STATIC_SPLIT_P_QUANT_4WG
            ? (warpgroup::groupid() - QUANT_WG0)
            : ((STATIC_SPLIT_P_QUANT_2WG && warpgroup::groupid() == QUANT_WG1) ? 1 : 0);
        const int split_mx_block_begin = STATIC_SPLIT_P_QUANT_4WG
            ? (warpgroup::groupid() - QUANT_WG0)
            : ((STATIC_SPLIT_P_QUANT_2WG && warpgroup::groupid() == QUANT_WG1) ? SPLIT_2WG_SECOND_BEGIN : 0);
        const int split_mx_block_end = STATIC_SPLIT_P_QUANT_4WG
            ? (split_mx_block_begin + 1)
            : (STATIC_SPLIT_P_QUANT_2WG ? (split_mx_block_begin + SPLIT_2WG_BLOCKS_PER_WG) : 4);
        const int split_lse_col_begin = split_mx_block_begin * 32;
        const int split_lse_col_end = split_mx_block_end * 32;
        const uint32_t score_tt_base0 = tt_scores.addr + ((warpgroup::warpid() * 32) << 16);
        int quant_p_scale_ready_count = 0;
        int quant_p_scale_tmem_ready_phase_mask = 0;
        int quant_p_scale_remote_phase_mask = 0;
        int quant_p_scale_reuse_phase_mask = 0;
        int p_stage_reuse_phase_mask = 0;
        int p_stage_reuse_ready_count = 0;
        int corr_reuse_phase_mask = 0;
        int corr_reuse_ready_count = 0;
        int p_sc_crossbar_phase_mask = 0;
        int p_payload_publish_phase_mask = 0;
        int p_scale_publish_phase_mask = 0;
        int p_k64_publish_phase_mask[2] = {0, 0};
        int p_k64_scale_slot_phase_mask = 0;
        int p_k64_scale_ready_count = 0;
        int p_k64_scale_remote_phase_mask[2] = {0, 0};
        int p_k64_payload_remote_fold_phase_mask[2] = {0, 0};
        int p_online_split_max_phase_mask = 0;
        int p_online_split_sum_phase_mask = 0;
        auto wait_for_p_stage_reusable = [&](int idx, int buf) {
            if constexpr (STATIC_WAIT_P_STAGE_SLOT_REUSE) {
                const bool needs_wait = STATIC_ONLINE_MXFP4_EARLY_P_READY
                    ? (p_stage_reuse_ready_count >= P_STAGE_SLOTS)
                    : (idx >= P_STAGE_SLOTS);
                if (needs_wait) {
                    const int reuse_phase = (p_stage_reuse_phase_mask >> buf) & 1;
                    wait(p_stage_reusable[buf], reuse_phase);
                    p_stage_reuse_phase_mask ^= (1 << buf);
                }
                if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                    ++p_stage_reuse_ready_count;
                }
            }
        };
        auto quant_stage_p_scale_tmem = [&](int idx, int buf, bool wait_remote_p_quant) {
            if constexpr (STATIC_QUANT_ASYNC_P_SCALE_TMEM &&
                          !STATIC_DECOUPLED_PV_INLINE_P_SCALE_REWRITE) {
                if constexpr (STATIC_DIRECT_P_SCALE_TMEM_STORE) {
                    if (warpgroup::groupid() == QUANT_WG0) {
                        constexpr int P_SC_BASE = SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                        if constexpr (!STATIC_ALIAS_SCALE_TMEM) {
                            static_assert(P_SCALE_TMEM_SLOTS == 2 || P_SCALE_TMEM_SLOTS == 3,
                                "direct P-scale TMEM store expects two or three slots");
                            static_assert(P_SC_BASE + P_SCALE_TMEM_SLOTS * P_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS,
                                "direct P-scale TMEM ping-pong exceeds TMEM budget");
                        }
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            const int p_sc_slot = STATIC_ALIAS_SCALE_TMEM ? (idx % SCORE_TMEM_SLOTS) : (idx % P_SCALE_TMEM_SLOTS);
                            if (quant_p_scale_ready_count >= P_SCALE_TMEM_SLOTS) {
                                const int reuse_phase = (quant_p_scale_reuse_phase_mask >> p_sc_slot) & 1;
                                wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                                quant_p_scale_reuse_phase_mask ^= (1 << p_sc_slot);
                            }
                            const int p_quant_phase = (quant_p_scale_tmem_ready_phase_mask >> buf) & 1;
                            if constexpr (STATIC_FUSED_DIRECT_SCALE_SKIP_PREPACK_REWRITE) {
                                wait(p_payload_published_ready[buf], p_quant_phase);
                            } else {
                                wait(p_quant_ready[buf], p_quant_phase);
                            }
                            quant_p_scale_tmem_ready_phase_mask ^= (1 << buf);
                        }
                        quant_wg_sync();
                        const int p_sc_slot_for_tmem = idx % P_SCALE_TMEM_SLOTS;
                        d_tt_p_sc p_sc_tm_cur;
                        if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                            const int scale_slot = idx % SCORE_TMEM_SLOTS;
                            p_sc_tm_cur = d_tt_p_sc(tt_scores.addr + scale_slot * C::Nb + ALIAS_P_SC_OFFSET);
                        } else {
                            d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
                            d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
                            d_tt_p_sc p_sc_tm2 = tm_alloc.template allocate<d_tt_p_sc>(
                                P_SC_BASE + (P_SCALE_TMEM_SLOTS == 3 ? 2 : 1) * P_SCALE_TMEM_WIDTH);
                            p_sc_tm_cur = p_sc_slot_for_tmem == 0
                                ? p_sc_tm0
                                : (p_sc_slot_for_tmem == 1 ? p_sc_tm1 : p_sc_tm2);
                        }
                        const int source_row = warp::laneid();
                        auto pack_scale_row = [&](int row) {
                            if constexpr (STATIC_DIRECT_P_SCALE_FROM_STAGE) {
                                return fp4pv_load_mxfp4_scale_word_swizzled(p_sc_stage[buf], row);
                            } else if constexpr (fp4pv_p_scale_tmem_direct_layout<C>::value == 1) {
                                return
                                    (p_sc_crossbar[buf][3][row] & 0xffu) |
                                    ((p_sc_crossbar[buf][2][row] & 0xffu) << 8) |
                                    ((p_sc_crossbar[buf][1][row] & 0xffu) << 16) |
                                    ((p_sc_crossbar[buf][0][row] & 0xffu) << 24);
                            } else if constexpr (fp4pv_p_scale_tmem_direct_layout<C>::value == 2) {
                                return
                                    (p_sc_crossbar[buf][0][row] & 0xffu) |
                                    ((p_sc_crossbar[buf][0][row] & 0xffu) << 8) |
                                    ((p_sc_crossbar[buf][0][row] & 0xffu) << 16) |
                                    ((p_sc_crossbar[buf][0][row] & 0xffu) << 24);
                            } else if constexpr (fp4pv_p_scale_tmem_direct_layout<C>::value == 3) {
                                return
                                    (p_sc_crossbar[buf][1][row] & 0xffu) |
                                    ((p_sc_crossbar[buf][1][row] & 0xffu) << 8) |
                                    ((p_sc_crossbar[buf][1][row] & 0xffu) << 16) |
                                    ((p_sc_crossbar[buf][1][row] & 0xffu) << 24);
                            } else if constexpr (fp4pv_p_scale_tmem_direct_layout<C>::value == 4) {
                                return
                                    (p_sc_crossbar[buf][2][row] & 0xffu) |
                                    ((p_sc_crossbar[buf][2][row] & 0xffu) << 8) |
                                    ((p_sc_crossbar[buf][2][row] & 0xffu) << 16) |
                                    ((p_sc_crossbar[buf][2][row] & 0xffu) << 24);
                            } else if constexpr (fp4pv_p_scale_tmem_direct_layout<C>::value == 5) {
                                return
                                    (p_sc_crossbar[buf][3][row] & 0xffu) |
                                    ((p_sc_crossbar[buf][3][row] & 0xffu) << 8) |
                                    ((p_sc_crossbar[buf][3][row] & 0xffu) << 16) |
                                    ((p_sc_crossbar[buf][3][row] & 0xffu) << 24);
                            } else {
                                return
                                    (p_sc_crossbar[buf][0][row] & 0xffu) |
                                    ((p_sc_crossbar[buf][1][row] & 0xffu) << 8) |
                                    ((p_sc_crossbar[buf][2][row] & 0xffu) << 16) |
                                    ((p_sc_crossbar[buf][3][row] & 0xffu) << 24);
                            }
                        };
                        const uint32_t row_tile0 = pack_scale_row(source_row + 0 * 32);
                        const uint32_t row_tile1 = pack_scale_row(source_row + 1 * 32);
                        const uint32_t row_tile2 = pack_scale_row(source_row + 2 * 32);
                        const uint32_t row_tile3 = pack_scale_row(source_row + 3 * 32);
                        fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
                            p_sc_tm_cur,
                            warpgroup::warpid() * 32,
                            row_tile0,
                            row_tile1,
                            row_tile2,
                            row_tile3);
                        fp4pv_tmem_store_wait();
                        quant_wg_sync();
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            if constexpr (C::CLUSTER_SIZE > 1) {
                                if constexpr (STATIC_FUSED_DIRECT_SCALE_SKIP_PREPACK_REWRITE) {
                                    if (cta_rank == 1) {
                                        tma::cluster::arrive(p_remote_ready[buf], 0);
                                    }
                                } else if constexpr (!STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY) {
                                    if (cta_rank == 1) {
                                        tma::cluster::arrive(p_remote_ready[buf], 0);
                                    }
                                }
                                if constexpr (!STATIC_DECOUPLE_P_SCALE_REMOTE_WAIT) {
                                    if (cta_rank == 0 && wait_remote_p_quant) {
                                        const int p_remote_phase = (quant_p_scale_remote_phase_mask >> buf) & 1;
                                        tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                        quant_p_scale_remote_phase_mask ^= (1 << buf);
                                    }
                                }
                            }
                            if (cta_rank == 0) {
                                const int p_sc_slot = STATIC_ALIAS_SCALE_TMEM ? (idx % SCORE_TMEM_SLOTS) : (idx % P_SCALE_TMEM_SLOTS);
                                arrive(p_sc_tmem_ready[p_sc_slot]);
                            }
                            ++quant_p_scale_ready_count;
                        }
                    }
                } else if (cta_rank == 0 && warpgroup::groupid() == QUANT_WG0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                    const int p_sc_slot = STATIC_ALIAS_SCALE_TMEM ? (idx % SCORE_TMEM_SLOTS) : (idx % P_SCALE_TMEM_SLOTS);
                    if (quant_p_scale_ready_count >= P_SCALE_TMEM_SLOTS) {
                        const int reuse_phase = (quant_p_scale_reuse_phase_mask >> p_sc_slot) & 1;
                        wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                        quant_p_scale_reuse_phase_mask ^= (1 << p_sc_slot);
                    }
                    const int p_quant_phase = (quant_p_scale_tmem_ready_phase_mask >> buf) & 1;
                    if constexpr (STATIC_FUSED_QUANT_ASYNC_SCALE_SKIP_PREPACK_REWRITE) {
                        wait(p_payload_published_ready[buf], p_quant_phase);
                    } else if constexpr (STATIC_FUSED_QUANT_ANY_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE) {
                        wait(p_pack_ready[buf], p_quant_phase);
                    } else {
                        wait(p_quant_ready[buf], p_quant_phase);
                    }
                    quant_p_scale_tmem_ready_phase_mask ^= (1 << buf);
                    if (wait_remote_p_quant && !STATIC_DECOUPLE_P_SCALE_REMOTE_WAIT) {
                        const int p_remote_phase = (quant_p_scale_remote_phase_mask >> buf) & 1;
                        tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                        quant_p_scale_remote_phase_mask ^= (1 << buf);
                    }
                    d_tt_p_sc p_sc_tm_cur;
                    if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                        const int scale_slot = idx % SCORE_TMEM_SLOTS;
                        p_sc_tm_cur = d_tt_p_sc(tt_scores.addr + scale_slot * C::Nb + ALIAS_P_SC_OFFSET);
                    } else {
                        constexpr int P_SC_BASE = SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                        static_assert(P_SC_BASE + 2 * P_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS, "quant P-scale TMEM ping-pong exceeds TMEM budget");
                        d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
                        d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
                        p_sc_tm_cur = p_sc_slot == 0 ? p_sc_tm0 : p_sc_tm1;
                    }
                    auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                    fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[buf]);
                    arrive(p_sc_tmem_ready[p_sc_slot]);
                    ++quant_p_scale_ready_count;
                }
            }
        };
        auto wait_for_scores = [&](int idx) -> uint32_t {
            const int score_slot = SCORE_TMEM_SLOTS > 1 ? (idx % SCORE_TMEM_SLOTS) : 0;
            const int scores_phase = (scores_phase_mask >> score_slot) & 1;
            wait(scores_arrived[score_slot], scores_phase);
            scores_phase_mask ^= (1 << score_slot);
            return score_tt_base0 + score_slot * C::Nb;
        };
        auto arrive_score_copy_done = [&](int idx, int buf) {
            const int copy_done_slot = SCORE_TMEM_SLOTS > 1 ? (idx % SCORE_TMEM_SLOTS) : buf;
            warpgroup::arrive(p_copy_done[copy_done_slot]);
        };
        for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
            int3 t_coord = get_tile_idx(cur_bid);
            if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                    continue;
                }
            }
            int m_tile = t_coord.y + cta_rank;
            if (m_tile >= tiles_m) {
                continue;
            }
            int q_iter_begin = 0;
            int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
            if constexpr (EXTERNAL_COL_LSE) {
                q_iter_begin = t_coord.y * SCORE_ITERS_PER_M_TILE;
                iters_per_task = score_tiles_m - q_iter_begin;
            }
            const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
            const bool wait_remote_p_quant = q_tiles_this_cluster > 1;
            const float SCALE_LOG2 = qk_scale_log2(g, t_coord.x, t_coord.z);
            const bool ZERO_SCALE = (SCALE_LOG2 == 0.0f);
            float row_sum = 0.0f;
            float row_max = base_types::constants<float>::neg_infty();
            if constexpr (!EXTERNAL_COL_LSE) {
                if constexpr (!STATIC_ONLINE_MXFP4_EARLY_P_READY &&
                              !STATIC_ONLINE_MXFP4_P_READY_BEFORE_RESCALE_WAIT) {
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                }
            }
            for (int idx = 0; idx < iters_per_task; ++idx) {
                const int buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx);
                const int global_idx = q_iter_begin + idx;
                const int global_row = warpgroup::laneid();
                const bool row_active = global_row < C::Mb;
                const int corr_slot = STATIC_ONLINE_MXFP4_EARLY_P_READY ? (idx & 1) : 0;
                float lane_lse_log2_pre = 0.0f;
                if constexpr (EXTERNAL_COL_LSE) {
                    constexpr float LOG2E = 1.4426950408889634f;
                    if constexpr (STATIC_PRELOAD_WARP_LSE_BROADCAST) {
                        const int lse_col = split_lse_col_begin + warp::laneid();
                        lane_lse_log2_pre =
                            g.lse[{t_coord.x, t_coord.z, 0, global_idx * C::Nb + lse_col}] * LOG2E;
                    } else if constexpr (!STATIC_DEFER_SHARED_LSE && !STATIC_WARP_LSE_BROADCAST) {
                        if (!STATIC_SPLIT_P_QUANT ||
                            (global_row >= split_lse_col_begin && global_row < split_lse_col_end)) {
                            external_lse_log2_smem[split_lse_slot][global_row] =
                                g.lse[{t_coord.x, t_coord.z, 0, global_idx * C::Nb + global_row}] * LOG2E;
                        }
                        quant_wg_sync();
                    }
                }
                if constexpr (STATIC_SPLIT_P_QUANT_4WG) {
                    float2 scores_one[16];
                    const uint32_t score_tt_base = wait_for_scores(idx);
                    const int mx_block = split_mx_block_begin;
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                        : "=f"(scores_one[0].x), "=f"(scores_one[0].y), "=f"(scores_one[1].x), "=f"(scores_one[1].y), "=f"(scores_one[2].x), "=f"(scores_one[2].y), "=f"(scores_one[3].x), "=f"(scores_one[3].y),
                          "=f"(scores_one[4].x), "=f"(scores_one[4].y), "=f"(scores_one[5].x), "=f"(scores_one[5].y), "=f"(scores_one[6].x), "=f"(scores_one[6].y), "=f"(scores_one[7].x), "=f"(scores_one[7].y),
                          "=f"(scores_one[8].x), "=f"(scores_one[8].y), "=f"(scores_one[9].x), "=f"(scores_one[9].y), "=f"(scores_one[10].x), "=f"(scores_one[10].y), "=f"(scores_one[11].x), "=f"(scores_one[11].y),
                          "=f"(scores_one[12].x), "=f"(scores_one[12].y), "=f"(scores_one[13].x), "=f"(scores_one[13].y), "=f"(scores_one[14].x), "=f"(scores_one[14].y), "=f"(scores_one[15].x), "=f"(scores_one[15].y)
                        : "r"(score_tt_base + mx_block * 32));
                    if constexpr (STATIC_DEFER_SHARED_LSE) {
                        constexpr float LOG2E = 1.4426950408889634f;
                        if (global_row >= split_lse_col_begin && global_row < split_lse_col_end) {
                            external_lse_log2_smem[split_lse_slot][global_row] =
                                g.lse[{t_coord.x, t_coord.z, 0, global_idx * C::Nb + global_row}] * LOG2E;
                        }
                        quant_wg_sync();
                    }
                    tensor_load_wait();
                    arrive_score_copy_done(idx, buf);
                    wait_for_p_stage_reusable(idx, buf);

                    const int global_k_row = m_tile * C::Mb + global_row;
                    const float2 payload_scale = {1.0f / g.p_output_scale, 1.0f / g.p_output_scale};
                    const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                    float lane_lse_log2 = 0.0f;
                    if constexpr (STATIC_WARP_LSE_BROADCAST || STATIC_SHARED_WARP_LSE_BROADCAST) {
                        if constexpr (STATIC_PRELOAD_WARP_LSE_BROADCAST) {
                            lane_lse_log2 = lane_lse_log2_pre;
                        } else if constexpr (STATIC_SHARED_WARP_LSE_BROADCAST) {
                            const int lse_col = split_lse_col_begin + warp::laneid();
                            lane_lse_log2 = external_lse_log2_smem[split_lse_slot][lse_col];
                        } else {
                            constexpr float LOG2E = 1.4426950408889634f;
                            const int lse_col = split_lse_col_begin + warp::laneid();
                            lane_lse_log2 =
                                g.lse[{t_coord.x, t_coord.z, 0, global_idx * C::Nb + lse_col}] * LOG2E;
                        }
                    }
                    float p_block_amax = 0.0f;
                    #pragma unroll
                    for (int jj = 0; jj < 16; ++jj) {
                        const int global_si = mx_block * 16 + jj;
                        const int global_q_col0 = global_idx * C::Nb + global_si * 2;
                        const int global_q_col1 = global_q_col0 + 1;
                        float lse0;
                        float lse1;
                        if constexpr (STATIC_WARP_LSE_BROADCAST || STATIC_SHARED_WARP_LSE_BROADCAST) {
                            lse0 = __shfl_sync(0xffffffffu, lane_lse_log2, jj * 2);
                            lse1 = __shfl_sync(0xffffffffu, lane_lse_log2, jj * 2 + 1);
                        } else {
                            lse0 = external_lse_log2_smem[split_lse_slot][global_si * 2];
                            lse1 = external_lse_log2_smem[split_lse_slot][global_si * 2 + 1];
                        }
                        scores_one[jj] = __fmul2_rn(scores_one[jj], scale_2);
                        scores_one[jj].x = (global_q_col0 >= global_k_row)
                            ? fp4pv_exp2_approx(scores_one[jj].x - lse0)
                            : 0.0f;
                        scores_one[jj].y = (global_q_col1 >= global_k_row)
                            ? fp4pv_exp2_approx(scores_one[jj].y - lse1)
                            : 0.0f;
                        scores_one[jj] = __fmul2_rn(scores_one[jj], payload_scale);
                        if constexpr (STATIC_PRECOMPUTED_P_AMAX) {
                            p_block_amax = fmaxf(p_block_amax, fmaxf(scores_one[jj].x, scores_one[jj].y));
                        }
                    }
                    uint8_t local_p_scale_e8m0 = 0;
                    if constexpr (STATIC_ANY_PACKED_P_SCALE_CROSSBAR) {
                        if constexpr (STATIC_PRECOMPUTED_P_AMAX) {
                            local_p_scale_e8m0 = fp4pv_pack_scores_one_to_stage_mxfp4_no_scale_with_amax<
                                STATIC_SCALAR_P_PAYLOAD_STORE,
                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], scores_one, p_block_amax, global_row, mx_block, g.p_quant_mode);
                        } else {
                            local_p_scale_e8m0 = fp4pv_pack_scores_one_to_stage_mxfp4_no_scale<
                                STATIC_SCALAR_P_PAYLOAD_STORE,
                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], scores_one, global_row, mx_block, g.p_quant_mode);
                        }
                        if constexpr (STATIC_WIDE_P_SCALE_CROSSBAR) {
                            p_sc_crossbar_wide[buf][mx_block][global_row & 127] =
                                static_cast<uint32_t>(local_p_scale_e8m0);
                        } else if constexpr (STATIC_HALF_P_SCALE_CROSSBAR) {
                            p_sc_crossbar_half[buf][mx_block][global_row & 127] =
                                static_cast<uint16_t>(local_p_scale_e8m0);
                        } else if constexpr (STATIC_PACKED_P_SCALE_LOCAL_BYPASS) {
                            if (warpgroup::groupid() != QUANT_WG0) {
                                p_sc_crossbar[buf][mx_block][global_row & 127] = local_p_scale_e8m0;
                            }
                        } else {
                            p_sc_crossbar[buf][mx_block][global_row & 127] = local_p_scale_e8m0;
                        }
                    } else {
                        if constexpr (STATIC_P128_SPLIT_SCALE_READY_VARIANT) {
                            uint32_t p_words[4];
                            uint8_t p_e8m0 = 0;
                            if constexpr (STATIC_PRECOMPUTED_P_AMAX) {
                                fp4pv_quantize_mxfp4_scores_group_with_amax(
                                    scores_one, p_block_amax, g.p_quant_mode, p_words, p_e8m0);
                            } else {
                                fp4pv_quantize_mxfp4_scores_group(
                                    scores_one, g.p_quant_mode, p_words, p_e8m0);
                            }
                            fp4pv_store_mxfp4_scale_byte_swizzled(
                                p_sc_stage[buf], global_row, mx_block, p_e8m0);
                            if (warp::elect_leader()) {
                                arrive(p_scale_ready[buf]);
                            }
                            if constexpr (STATIC_P128_SPLIT_SCALE_LEADER_PUBLISH) {
                                if (warpgroup::groupid() == QUANT_WG0 &&
                                    warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    const int p_scale_phase = (p_scale_publish_phase_mask >> buf) & 1;
                                    wait(p_scale_ready[buf], p_scale_phase);
                                    p_scale_publish_phase_mask ^= (1 << buf);
                                    publish_shared_backing();
                                    arrive(p_scale_published_ready[buf]);
                                    if constexpr (C::CLUSTER_SIZE > 1) {
                                        if (cta_rank == 1) {
                                            tma::cluster::arrive(p_scale_remote_ready[buf], 0);
                                        }
                                    }
                                }
                            } else {
                                if constexpr (STATIC_FUSED_PRODUCER_EARLY_SCALE_PROXY_SKIP_PREPACK_REWRITE) {
                                    fp4pv_publish_shared_backing_proxy_only();
                                } else {
                                    publish_shared_backing();
                                }
                                if constexpr (C::CLUSTER_SIZE > 1) {
                                    if (cta_rank == 1 && warpgroup::groupid() == QUANT_WG0 &&
                                        warpgroup::warpid() == 0 && warp::elect_leader()) {
                                        const int p_scale_phase = (p_scale_publish_phase_mask >> buf) & 1;
                                        wait(p_scale_ready[buf], p_scale_phase);
                                        p_scale_publish_phase_mask ^= (1 << buf);
                                        tma::cluster::arrive(p_scale_remote_ready[buf], 0);
                                    }
                                }
                            }
                            fp4pv_store_quantized_scores_group_mxfp4_selected<
                                STATIC_SCALAR_P_PAYLOAD_STORE,
                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], p_words, global_row, mx_block, 0);
                        } else if constexpr (STATIC_PRECOMPUTED_P_AMAX) {
	                            fp4pv_pack_scores_one_to_stage_mxfp4_with_amax<
	                                STATIC_SCALAR_P_PAYLOAD_STORE,
	                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], p_sc_stage[buf], scores_one, p_block_amax, global_row, mx_block, g.p_quant_mode);
                        } else {
                            fp4pv_pack_scores_one_to_stage_mxfp4<
                                STATIC_SCALAR_P_PAYLOAD_STORE,
                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], p_sc_stage[buf], scores_one, global_row, mx_block, g.p_quant_mode);
                        }
                    }
                    if constexpr (!STATIC_SINGLE_WG_P_PAYLOAD_LOCAL_PUBLISH &&
                                  !STATIC_SKIP_P_PAYLOAD_LOCAL_PUBLISH) {
                        if constexpr (STATIC_LIGHT_P_PAYLOAD_LOCAL_PUBLISH) {
                            fp4pv_publish_shared_backing_proxy_only();
                        } else {
                            publish_shared_backing();
                        }
                    }
                    if constexpr (STATIC_EARLY_PV_K64_HALF_READY) {
                        const int p_half = mx_block >> 1;
                        warpgroup::arrive(p_k64_ready[buf][p_half]);
                        const bool publish_half =
                            (mx_block == 0 && warpgroup::groupid() == QUANT_WG0) ||
                            (mx_block == 2 && warpgroup::groupid() == QUANT_WG2);
                        if (publish_half) {
                            if constexpr (STATIC_EARLY_PV_K64_CROSSBAR_SCALE) {
                                const int p_half_phase = (p_k64_publish_phase_mask[p_half] >> buf) & 1;
                                wait(p_k64_ready[buf][p_half], p_half_phase);
                                p_k64_publish_phase_mask[p_half] ^= (1 << buf);
                            } else if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                const int p_half_phase = (p_k64_publish_phase_mask[p_half] >> buf) & 1;
                                wait(p_k64_ready[buf][p_half], p_half_phase);
                                p_k64_publish_phase_mask[p_half] ^= (1 << buf);
                            }
                            if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE &&
                                          STATIC_EARLY_PV_K64_MERGED_SCALE_WAIT) {
                                const int p_sc_slot = idx % SCORE_TMEM_SLOTS;
                                if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE) {
                                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                        const int p_slot_phase = (p_k64_scale_slot_phase_mask >> p_sc_slot) & 1;
                                        wait(p_k64_scale_slot_ready[p_sc_slot], p_slot_phase);
                                        p_k64_scale_slot_phase_mask ^= (1 << p_sc_slot);
                                    }
                                } else if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_INLINE_REUSE) {
                                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                        if (p_k64_scale_ready_count >= SCORE_TMEM_SLOTS) {
                                            const int reuse_phase = (p_k64_scale_slot_phase_mask >> p_sc_slot) & 1;
                                            wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                                            p_k64_scale_slot_phase_mask ^= (1 << p_sc_slot);
                                        }
                                        ++p_k64_scale_ready_count;
                                    }
                                }
                            }
                            quant_wg_sync();
                            if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE) {
                                const int p_sc_slot = idx % SCORE_TMEM_SLOTS;
                                if constexpr (!STATIC_EARLY_PV_K64_MERGED_SCALE_WAIT) {
                                    if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_PRODUCER_REUSE) {
                                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                            const int p_slot_phase = (p_k64_scale_slot_phase_mask >> p_sc_slot) & 1;
                                            wait(p_k64_scale_slot_ready[p_sc_slot], p_slot_phase);
                                            p_k64_scale_slot_phase_mask ^= (1 << p_sc_slot);
                                        }
                                    } else if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_INLINE_REUSE) {
                                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                            if (p_k64_scale_ready_count >= SCORE_TMEM_SLOTS) {
                                                const int reuse_phase = (p_k64_scale_slot_phase_mask >> p_sc_slot) & 1;
                                                wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                                                p_k64_scale_slot_phase_mask ^= (1 << p_sc_slot);
                                            }
                                            ++p_k64_scale_ready_count;
                                        }
                                    }
                                    quant_wg_sync();
                                }
                                const int p_sc_offset = p_half == 0 ? ALIAS_P_SC_OFFSET : ALIAS_P_SC_K64_HALF1_OFFSET;
                                d_tt_p_sc p_sc_tm_cur = d_tt_p_sc(tt_scores.addr + p_sc_slot * C::Nb + p_sc_offset);
                                const int row_in_32 = warp::laneid();
                                auto pack_direct_half_row = [&](int scale_row) {
                                    const int block0 = p_half * 2;
                                    const int block1 = block0 + 1;
                                    uint32_t half_word;
                                    if constexpr (STATIC_WIDE_P_SCALE_CROSSBAR) {
                                        half_word =
                                            (p_sc_crossbar_wide[buf][block0][scale_row] & 0xffu) |
                                            ((p_sc_crossbar_wide[buf][block1][scale_row] & 0xffu) << 8);
                                    } else if constexpr (STATIC_HALF_P_SCALE_CROSSBAR) {
                                        half_word =
                                            (static_cast<uint32_t>(p_sc_crossbar_half[buf][block0][scale_row]) & 0xffu) |
                                            ((static_cast<uint32_t>(p_sc_crossbar_half[buf][block1][scale_row]) & 0xffu) << 8);
                                    } else {
                                        half_word =
                                            static_cast<uint32_t>(p_sc_crossbar[buf][block0][scale_row]) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][block1][scale_row]) << 8);
                                    }
                                    return p_half == 0 ? half_word : (half_word << 16);
                                };
                                fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
                                    p_sc_tm_cur,
                                    warpgroup::warpid() * 32,
                                    pack_direct_half_row(row_in_32 + 0 * 32),
                                    pack_direct_half_row(row_in_32 + 1 * 32),
                                    pack_direct_half_row(row_in_32 + 2 * 32),
                                    pack_direct_half_row(row_in_32 + 3 * 32));
                                if constexpr (STATIC_EARLY_PV_K64_EAGER_PAYLOAD_REMOTE) {
                                    if (warpgroup::warpid() == 0) {
                                        publish_cluster_shared_backing_if_needed<C>();
                                        if constexpr (C::CLUSTER_SIZE > 1) {
                                            if (cta_rank == 1 && warp::elect_leader()) {
                                                tma::cluster::arrive(p_k64_remote_ready[buf][p_half], 0);
                                            }
                                        }
                                    }
                                }
                                fp4pv_tmem_store_wait();
                                quant_wg_sync();
                                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_REMOTE_SYNC) {
                                        if constexpr (C::CLUSTER_SIZE > 1) {
                                            if (q_tiles_this_cluster > 1) {
                                                if (cta_rank == 1) {
                                                    tma::cluster::arrive(p_k64_scale_remote_ready[buf][p_half], 0);
                                                } else {
                                                    const int scale_remote_phase =
                                                        (p_k64_scale_remote_phase_mask[p_half] >> buf) & 1;
                                                    tma::cluster::wait(p_k64_scale_remote_ready[buf][p_half], scale_remote_phase);
                                                    p_k64_scale_remote_phase_mask[p_half] ^= (1 << buf);
                                                }
                                            }
                                        }
                                    }
                                    if (cta_rank == 0 &&
                                        !STATIC_EARLY_PV_K64_PRODUCER_FOLDS_PAYLOAD_REMOTE) {
                                        arrive(p_k64_sc_tmem_ready[p_sc_slot][p_half]);
                                    }
                                }
                            } else if constexpr (STATIC_DECOUPLED_PV_EARLY_K64_REWRITE) {
                                if (warpgroup::warpid() == 0 && warp::elect_leader() && cta_rank == 0) {
                                    const int p_sc_slot = idx % SCORE_TMEM_SLOTS;
                                    arrive(p_k64_sc_tmem_ready[p_sc_slot][p_half]);
                                }
                            } else if constexpr (STATIC_EARLY_PV_K64_CROSSBAR_SCALE) {
                                if (warpgroup::warpid() == 0) {
                                    const int row_in_32 = warp::laneid();
                                    auto pack_scale_half_row = [&](int scale_row) {
                                        const int block0 = p_half * 2;
                                        const int block1 = block0 + 1;
                                        return
                                            (p_sc_crossbar_wide[buf][block0][scale_row] & 0xffu) |
                                            ((p_sc_crossbar_wide[buf][block1][scale_row] & 0xffu) << 8);
                                    };
                                    if (p_half == 0) {
                                        fp4pv_store_mxfp4_scale_halfword4_swizzled<0>(
                                            p_sc_stage[buf],
                                            row_in_32,
                                            pack_scale_half_row(row_in_32 + 0 * 32),
                                            pack_scale_half_row(row_in_32 + 1 * 32),
                                            pack_scale_half_row(row_in_32 + 2 * 32),
                                            pack_scale_half_row(row_in_32 + 3 * 32));
                                    } else {
                                        fp4pv_store_mxfp4_scale_halfword4_swizzled<1>(
                                            p_sc_stage[buf],
                                            row_in_32,
                                            pack_scale_half_row(row_in_32 + 0 * 32),
                                            pack_scale_half_row(row_in_32 + 1 * 32),
                                            pack_scale_half_row(row_in_32 + 2 * 32),
                                            pack_scale_half_row(row_in_32 + 3 * 32));
                                    }
                                }
                                quant_wg_sync();
                                if (warpgroup::warpid() == 0) {
                                    publish_shared_backing();
                                }
                                quant_wg_sync();
                                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    arrive(p_k64_stage_ready[buf][p_half]);
                                }
                            }
                            if constexpr (!STATIC_EARLY_PV_K64_EAGER_PAYLOAD_REMOTE) {
                                if constexpr (STATIC_EARLY_PV_K64_FULL_PAYLOAD_REMOTE_REWRITE) {
                                    if (p_half == 1 && warpgroup::warpid() == 0) {
                                        publish_cluster_shared_backing_if_needed<C>();
                                        if constexpr (C::CLUSTER_SIZE > 1) {
                                            if (cta_rank == 1 && warp::elect_leader()) {
                                                tma::cluster::arrive(p_remote_ready[buf], 0);
                                            }
                                        }
                                    }
                                } else {
                                    if (warpgroup::warpid() == 0) {
                                        publish_cluster_shared_backing_if_needed<C>();
                                        if constexpr (C::CLUSTER_SIZE > 1) {
                                            if (cta_rank == 1 && warp::elect_leader()) {
                                                tma::cluster::arrive(p_k64_remote_ready[buf][p_half], 0);
                                            }
                                        }
                                    }
                                }
                            }
                            if constexpr (STATIC_EARLY_PV_K64_PRODUCER_FOLDS_PAYLOAD_REMOTE) {
                                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    if constexpr (C::CLUSTER_SIZE > 1) {
                                        if (q_tiles_this_cluster > 1 && cta_rank == 0) {
                                            const int payload_remote_phase =
                                                (p_k64_payload_remote_fold_phase_mask[p_half] >> buf) & 1;
                                            tma::cluster::wait(p_k64_remote_ready[buf][p_half], payload_remote_phase);
                                            p_k64_payload_remote_fold_phase_mask[p_half] ^= (1 << buf);
                                        }
                                    }
                                    if (cta_rank == 0) {
                                        const int p_sc_slot = idx % SCORE_TMEM_SLOTS;
                                        arrive(p_k64_sc_tmem_ready[p_sc_slot][p_half]);
                                    }
                                }
                            }
                            quant_wg_sync();
                        }
                        continue;
                    }
                    if constexpr (STATIC_EARLY_CLUSTER_P_PAYLOAD_PUBLISH) {
                        publish_cluster_shared_backing_if_needed<C>();
                    }
                    if constexpr (STATIC_ANY_PACKED_P_SCALE_CROSSBAR) {
                        if constexpr (STATIC_WARP_P_SCALE_CROSSBAR_READY) {
                            if (warp::elect_leader()) {
                                arrive(p_sc_crossbar_ready[buf]);
                            }
                        } else {
                            quant_wg_sync();
                        }
                        if constexpr (STATIC_SINGLE_WG_P_PAYLOAD_LOCAL_PUBLISH) {
                            if (warpgroup::groupid() == QUANT_WG0) {
                                if constexpr (STATIC_LIGHT_P_PAYLOAD_LOCAL_PUBLISH) {
                                    fp4pv_publish_shared_backing_proxy_only();
                                } else {
                                    publish_shared_backing();
                                }
                            }
                            quant_wg_sync();
                        }
                        if constexpr (!STATIC_WARP_P_SCALE_CROSSBAR_READY) {
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            arrive(p_sc_crossbar_ready[buf]);
                        }
                        }
                        if constexpr (!STATIC_PRODUCER_ASYNC_P_SCALE_AGGREGATE) {
                        if (warpgroup::groupid() == QUANT_WG0) {
                            const int p_sc_crossbar_phase = (p_sc_crossbar_phase_mask >> buf) & 1;
                            wait(p_sc_crossbar_ready[buf], p_sc_crossbar_phase);
                            p_sc_crossbar_phase_mask ^= (1 << buf);
                            const int row = warpgroup::laneid();
                            if constexpr (STATIC_EARLY_DIRECT_P_SCALE_TMEM) {
                                const int p_sc_slot = STATIC_ALIAS_SCALE_TMEM ? (idx % SCORE_TMEM_SLOTS) : (idx % P_SCALE_TMEM_SLOTS);
                                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    if (quant_p_scale_ready_count >= P_SCALE_TMEM_SLOTS) {
                                        const int reuse_phase = (quant_p_scale_reuse_phase_mask >> p_sc_slot) & 1;
                                        wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                                        quant_p_scale_reuse_phase_mask ^= (1 << p_sc_slot);
                                    }
                                }
                                quant_wg_sync();
                                d_tt_p_sc p_sc_tm_cur;
                                if constexpr (STATIC_ALIAS_SCALE_TMEM) {
                                    p_sc_tm_cur = d_tt_p_sc(tt_scores.addr + p_sc_slot * C::Nb + ALIAS_P_SC_OFFSET);
                                } else {
                                    constexpr int P_SC_BASE = SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                                    static_assert(P_SC_BASE + 2 * P_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS, "early direct P-scale TMEM ping-pong exceeds TMEM budget");
                                    d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
                                    d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
                                    p_sc_tm_cur = p_sc_slot == 0 ? p_sc_tm0 : p_sc_tm1;
                                }
                                const int source_row = warp::laneid();
                                auto pack_scale_row = [&](int scale_row) {
                                    if constexpr (STATIC_WIDE_P_SCALE_CROSSBAR) {
                                        return
                                            (p_sc_crossbar_wide[buf][0][scale_row] & 0xffu) |
                                            ((p_sc_crossbar_wide[buf][1][scale_row] & 0xffu) << 8) |
                                            ((p_sc_crossbar_wide[buf][2][scale_row] & 0xffu) << 16) |
                                            ((p_sc_crossbar_wide[buf][3][scale_row] & 0xffu) << 24);
                                    } else if constexpr (STATIC_HALF_P_SCALE_CROSSBAR) {
                                        return
                                            (static_cast<uint32_t>(p_sc_crossbar_half[buf][0][scale_row]) & 0xffu) |
                                            ((static_cast<uint32_t>(p_sc_crossbar_half[buf][1][scale_row]) & 0xffu) << 8) |
                                            ((static_cast<uint32_t>(p_sc_crossbar_half[buf][2][scale_row]) & 0xffu) << 16) |
                                            ((static_cast<uint32_t>(p_sc_crossbar_half[buf][3][scale_row]) & 0xffu) << 24);
                                    } else if constexpr (STATIC_PACKED_P_SCALE_LOCAL_BYPASS) {
                                        return
                                            static_cast<uint32_t>(local_p_scale_e8m0) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][1][scale_row]) << 8) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][2][scale_row]) << 16) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][3][scale_row]) << 24);
                                    } else {
                                        return
                                            static_cast<uint32_t>(p_sc_crossbar[buf][0][scale_row]) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][1][scale_row]) << 8) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][2][scale_row]) << 16) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][3][scale_row]) << 24);
                                    }
                                };
                                fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
                                    p_sc_tm_cur,
                                    warpgroup::warpid() * 32,
                                    pack_scale_row(source_row + 0 * 32),
                                    pack_scale_row(source_row + 1 * 32),
                                    pack_scale_row(source_row + 2 * 32),
                                    pack_scale_row(source_row + 3 * 32));
                                if constexpr (!STATIC_DELAY_DIRECT_P_SCALE_TMEM_WAIT) {
                                    fp4pv_tmem_store_wait();
                                }
                                quant_wg_sync();
                                if constexpr (STATIC_DECOUPLED_PV_EARLY_SCALE_READY_REWRITE) {
                                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                        if (cta_rank == 0) {
                                            arrive(p_sc_tmem_ready[p_sc_slot]);
                                        }
                                        ++quant_p_scale_ready_count;
                                    }
                                }
                            } else if constexpr (!STATIC_DIRECT_P_SCALE_TMEM_STORE) {
                                auto pack_scale_row_to_word = [&](int scale_row) {
                                    uint32_t packed;
                                if constexpr (STATIC_WIDE_P_SCALE_CROSSBAR) {
                                    packed =
                                            (p_sc_crossbar_wide[buf][0][scale_row] & 0xffu) |
                                            ((p_sc_crossbar_wide[buf][1][scale_row] & 0xffu) << 8) |
                                            ((p_sc_crossbar_wide[buf][2][scale_row] & 0xffu) << 16) |
                                            ((p_sc_crossbar_wide[buf][3][scale_row] & 0xffu) << 24);
                                } else if constexpr (STATIC_HALF_P_SCALE_CROSSBAR) {
                                    packed =
                                            (static_cast<uint32_t>(p_sc_crossbar_half[buf][0][scale_row]) & 0xffu) |
                                            ((static_cast<uint32_t>(p_sc_crossbar_half[buf][1][scale_row]) & 0xffu) << 8) |
                                            ((static_cast<uint32_t>(p_sc_crossbar_half[buf][2][scale_row]) & 0xffu) << 16) |
                                            ((static_cast<uint32_t>(p_sc_crossbar_half[buf][3][scale_row]) & 0xffu) << 24);
                                } else if constexpr (STATIC_PACKED_P_SCALE_LOCAL_BYPASS) {
                                    packed =
                                        static_cast<uint32_t>(local_p_scale_e8m0) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][1][scale_row]) << 8) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][2][scale_row]) << 16) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][3][scale_row]) << 24);
                                } else {
                                    packed =
                                            static_cast<uint32_t>(p_sc_crossbar[buf][0][scale_row]) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][1][scale_row]) << 8) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][2][scale_row]) << 16) |
                                            (static_cast<uint32_t>(p_sc_crossbar[buf][3][scale_row]) << 24);
                                }
                                    return packed;
                                };
                                if constexpr (STATIC_BANK_FREE_P_SCALE_STAGE_STORE) {
                                    if (warpgroup::warpid() == 0) {
                                        const int row_in_32 = warp::laneid();
                                        fp4pv_store_mxfp4_scale_word4_swizzled(
                                            p_sc_stage[buf],
                                            row_in_32,
                                            pack_scale_row_to_word(row_in_32 + 0 * 32),
                                            pack_scale_row_to_word(row_in_32 + 1 * 32),
                                            pack_scale_row_to_word(row_in_32 + 2 * 32),
                                            pack_scale_row_to_word(row_in_32 + 3 * 32));
                                    }
                                } else if constexpr (STATIC_BANK_FREE_P_SCALE_STAGE_STORE_2W) {
                                    if (warpgroup::warpid() < 2 && warp::laneid() < 16) {
                                        const int row_in_32 = warpgroup::warpid() * 16 + warp::laneid();
                                        fp4pv_store_mxfp4_scale_word4_swizzled(
                                            p_sc_stage[buf],
                                            row_in_32,
                                            pack_scale_row_to_word(row_in_32 + 0 * 32),
                                            pack_scale_row_to_word(row_in_32 + 1 * 32),
                                            pack_scale_row_to_word(row_in_32 + 2 * 32),
                                            pack_scale_row_to_word(row_in_32 + 3 * 32));
                                    }
                                } else if constexpr (STATIC_BANK_FREE_P_SCALE_STAGE_STORE_4W) {
                                    if (warp::laneid() < 8) {
                                        const int row_in_32 = warpgroup::warpid() * 8 + warp::laneid();
                                        fp4pv_store_mxfp4_scale_word4_swizzled(
                                            p_sc_stage[buf],
                                            row_in_32,
                                            pack_scale_row_to_word(row_in_32 + 0 * 32),
                                            pack_scale_row_to_word(row_in_32 + 1 * 32),
                                            pack_scale_row_to_word(row_in_32 + 2 * 32),
                                            pack_scale_row_to_word(row_in_32 + 3 * 32));
                                    }
                                } else {
                                    fp4pv_store_mxfp4_scale_word_swizzled(
                                        p_sc_stage[buf], row, pack_scale_row_to_word(row));
                                }
                                if constexpr (STATIC_LIGHT_P_PAYLOAD_LOCAL_PUBLISH) {
                                    fp4pv_publish_shared_backing_proxy_only();
                                } else {
                                    publish_shared_backing();
                                }
                                if constexpr (STATIC_EARLY_ASYNC_P_SCALE_TMEM) {
                                    quant_wg_sync();
                                    if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                        constexpr int P_SC_BASE = SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                                        static_assert(P_SC_BASE + 2 * P_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS, "early async P-scale TMEM ping-pong exceeds TMEM budget");
                                        d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
                                        d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
                                        if (quant_p_scale_ready_count >= 2) {
                                            const int reuse_phase = (quant_p_scale_reuse_phase_mask >> buf) & 1;
                                            wait(p_sc_tmem_reusable[buf], reuse_phase);
                                            quant_p_scale_reuse_phase_mask ^= (1 << buf);
                                        }
                                        auto &p_sc_tm_cur = buf == 0 ? p_sc_tm0 : p_sc_tm1;
                                        auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                                        fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[buf]);
                                    }
                                }
                            }
                            if constexpr (STATIC_SINGLE_WG_P_PAYLOAD_CLUSTER_PUBLISH) {
                                quant_wg_sync();
                                if (warpgroup::groupid() == QUANT_WG0 &&
                                    (!STATIC_SINGLE_WARP_P_PAYLOAD_CLUSTER_PUBLISH || warpgroup::warpid() == 0)) {
                                    publish_cluster_shared_backing_if_needed<C>();
                                    if constexpr (STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY &&
                                                  C::CLUSTER_SIZE > 1 && !STATIC_DIRECT_P_SCALE_TMEM_STORE) {
                                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                            tma::cluster::arrive(p_remote_ready[buf], 0);
                                        }
                                    }
                                    if constexpr (STATIC_P_PAYLOAD_PUBLISHED_READY) {
                                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                            if constexpr (STATIC_DECOUPLED_PV_FOLD_REMOTE_READY_REWRITE &&
                                                          C::CLUSTER_SIZE > 1) {
                                                if (cta_rank == 0 && wait_remote_p_quant) {
                                                    const int p_remote_phase =
                                                        (quant_p_scale_remote_phase_mask >> buf) & 1;
                                                    tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                                    quant_p_scale_remote_phase_mask ^= (1 << buf);
                                                }
                                            }
                                            arrive(p_payload_published_ready[buf]);
                                        }
                                    }
                                }
                                if constexpr (!STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY) {
                                    quant_wg_sync();
                                }
                            } else {
                                if constexpr (!STATIC_EARLY_CLUSTER_P_PAYLOAD_PUBLISH) {
                                    publish_cluster_shared_backing_if_needed<C>();
                                }
                                quant_wg_sync();
                                if constexpr (STATIC_DECOUPLED_PV_EARLY_SCALE_READY_REWRITE) {
                                    if (warpgroup::groupid() == QUANT_WG0 &&
                                        warpgroup::warpid() == 0 && warp::elect_leader()) {
                                        arrive(p_payload_published_ready[buf]);
                                    }
                                }
                            }
                            if constexpr (C::CLUSTER_SIZE > 1 && !STATIC_DIRECT_P_SCALE_TMEM_STORE &&
                                          !STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY) {
                                if (cta_rank == 1 &&
                                    (!STATIC_SINGLE_WG_P_PAYLOAD_CLUSTER_PUBLISH ||
                                     warpgroup::groupid() == QUANT_WG0) &&
                                    warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    tma::cluster::arrive(p_remote_ready[buf], 0);
                                }
                            }
                            if constexpr (STATIC_DELAY_DIRECT_P_SCALE_TMEM_WAIT) {
                                fp4pv_tmem_store_wait();
                                quant_wg_sync();
                            }
                            if constexpr (STATIC_EARLY_P_SCALE_TMEM &&
                                          !STATIC_DECOUPLED_PV_EARLY_SCALE_READY_REWRITE) {
                                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    if (cta_rank == 0) {
                                        if constexpr (!STATIC_DECOUPLE_P_SCALE_REMOTE_WAIT ||
                                                      STATIC_DECOUPLED_PV_FOLD_REMOTE_READY_REWRITE) {
                                            if (wait_remote_p_quant) {
                                                const int p_remote_phase = (quant_p_scale_remote_phase_mask >> buf) & 1;
                                                tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                                quant_p_scale_remote_phase_mask ^= (1 << buf);
                                            }
                                        }
                                        const int p_sc_slot = STATIC_ALIAS_SCALE_TMEM ? (idx % SCORE_TMEM_SLOTS) : (idx % P_SCALE_TMEM_SLOTS);
                                        arrive(p_sc_tmem_ready[p_sc_slot]);
                                    }
                                    ++quant_p_scale_ready_count;
                                }
                            } else {
                                if constexpr (!STATIC_EARLY_LOCAL_P_SCALE_READY) {
                                    if constexpr (!STATIC_P_PAYLOAD_PUBLISHED_READY) {
                                        warpgroup::arrive(p_quant_ready[buf]);
                                    }
                                }
                            }
                        }
                    }
                } else {
                        if constexpr (!STATIC_SKIP_PRE_PACK_READY_SYNC && !STATIC_WARP_P_PACK_READY) {
                            quant_wg_sync();
                        }
                        if constexpr (STATIC_EARLY_LOCAL_P_SCALE_READY) {
                            if constexpr (STATIC_WARP_P_PACK_READY) {
                                if (warp::elect_leader()) {
                                    arrive(p_pack_ready[buf]);
                                }
                            } else {
                                warpgroup::arrive(p_pack_ready[buf]);
                            }
                        }
                        if constexpr (STATIC_FUSED_QUANT_ANY_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE) {
                            quant_wg_sync();
                            if (warpgroup::groupid() == QUANT_WG0) {
                                fp4pv_publish_shared_backing_proxy_only();
                                if constexpr (STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_NOSYNC_SKIP_PREPACK_REWRITE) {
                                    quant_stage_p_scale_tmem(idx, buf, wait_remote_p_quant);
                                }
                            }
                            if constexpr (!STATIC_FUSED_QUANT_EARLY_ASYNC_SCALE_NOSYNC_SKIP_PREPACK_REWRITE) {
                                quant_wg_sync();
                                quant_stage_p_scale_tmem(idx, buf, wait_remote_p_quant);
                            }
                        }
                        if constexpr (STATIC_EARLY_LOCAL_P_SCALE_TMEM) {
                            if (cta_rank == 0 && warpgroup::groupid() == QUANT_WG0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                constexpr int P_SC_BASE = SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                                static_assert(P_SC_BASE + 2 * P_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS, "early local P-scale TMEM ping-pong exceeds TMEM budget");
                                d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
                                d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
                                if (quant_p_scale_ready_count >= 2) {
                                    const int reuse_phase = (quant_p_scale_reuse_phase_mask >> buf) & 1;
                                    wait(p_sc_tmem_reusable[buf], reuse_phase);
                                    quant_p_scale_reuse_phase_mask ^= (1 << buf);
                                }
                                auto &p_sc_tm_cur = buf == 0 ? p_sc_tm0 : p_sc_tm1;
                                auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e8m0<16>>(0);
                                fp4pv_load_mxnv_scale_async_cluster<C>(p_sc_tm_sub, p_sc_stage[buf]);
                            }
                        }
                        if constexpr (STATIC_SINGLE_WG_P_PAYLOAD_CLUSTER_PUBLISH) {
                            if constexpr (STATIC_ASYNC_P_PAYLOAD_CLUSTER_PUBLISH) {
                                if (warpgroup::groupid() == QUANT_WG0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    const int p_payload_phase = (p_payload_publish_phase_mask >> buf) & 1;
                                    wait(p_pack_ready[buf], p_payload_phase);
                                    p_payload_publish_phase_mask ^= (1 << buf);
                                    publish_cluster_shared_backing_if_needed<C>();
                                    if constexpr (STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY &&
                                                  C::CLUSTER_SIZE > 1 && !STATIC_DIRECT_P_SCALE_TMEM_STORE) {
                                        if (cta_rank == 1) {
                                            tma::cluster::arrive(p_remote_ready[buf], 0);
                                        }
                                    }
                                    if constexpr (STATIC_P_PAYLOAD_PUBLISHED_READY) {
                                        if constexpr (STATIC_DECOUPLED_PV_FOLD_REMOTE_READY_REWRITE &&
                                                      C::CLUSTER_SIZE > 1) {
                                            if (cta_rank == 0 && wait_remote_p_quant) {
                                                const int p_remote_phase =
                                                    (quant_p_scale_remote_phase_mask >> buf) & 1;
                                                tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                                quant_p_scale_remote_phase_mask ^= (1 << buf);
                                            }
                                        }
                                        arrive(p_payload_published_ready[buf]);
                                    }
                                }
                            } else {
                                quant_wg_sync();
                                if (warpgroup::groupid() == QUANT_WG0 &&
                                    (!STATIC_SINGLE_WARP_P_PAYLOAD_CLUSTER_PUBLISH || warpgroup::warpid() == 0)) {
                                    publish_cluster_shared_backing_if_needed<C>();
                                    if constexpr (STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY &&
                                                  C::CLUSTER_SIZE > 1 && !STATIC_DIRECT_P_SCALE_TMEM_STORE) {
                                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                            tma::cluster::arrive(p_remote_ready[buf], 0);
                                        }
                                    }
                                    if constexpr (STATIC_P_PAYLOAD_PUBLISHED_READY) {
                                        arrive(p_payload_published_ready[buf]);
                                    }
                                }
                                if constexpr (!STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY) {
                                    quant_wg_sync();
                                }
                            }
                        } else {
                            if constexpr (!STATIC_EARLY_CLUSTER_P_PAYLOAD_PUBLISH) {
                                publish_cluster_shared_backing_if_needed<C>();
                            }
                            quant_wg_sync();
                            if constexpr (STATIC_DECOUPLED_PV_EARLY_SCALE_READY_REWRITE) {
                                if (warpgroup::groupid() == QUANT_WG0 &&
                                    warpgroup::warpid() == 0 && warp::elect_leader()) {
                                    arrive(p_payload_published_ready[buf]);
                                }
                            }
                        }
                        if constexpr (C::CLUSTER_SIZE > 1 && !STATIC_DIRECT_P_SCALE_TMEM_STORE &&
                                      !STATIC_FUSE_P_PAYLOAD_CLUSTER_PUBLISH_REMOTE_READY) {
                            if (cta_rank == 1 &&
                                (!STATIC_SINGLE_WG_P_PAYLOAD_CLUSTER_PUBLISH ||
                                 warpgroup::groupid() == QUANT_WG0) &&
                                warpgroup::warpid() == 0 && warp::elect_leader()) {
                                tma::cluster::arrive(p_remote_ready[buf], 0);
                            }
                        }
                        if constexpr (STATIC_EARLY_LOCAL_P_SCALE_TMEM) {
                            if (cta_rank == 0 && warpgroup::groupid() == QUANT_WG0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                if (wait_remote_p_quant) {
                                    const int p_remote_phase = (quant_p_scale_remote_phase_mask >> buf) & 1;
                                    tma::cluster::wait(p_remote_ready[buf], p_remote_phase);
                                    quant_p_scale_remote_phase_mask ^= (1 << buf);
                                }
                                arrive(p_sc_tmem_ready[buf]);
                                ++quant_p_scale_ready_count;
                            }
                        } else {
                            if constexpr (!STATIC_EARLY_LOCAL_P_SCALE_READY) {
                                if constexpr (!STATIC_P_PAYLOAD_PUBLISHED_READY) {
                                    warpgroup::arrive(p_quant_ready[buf]);
                                }
                            }
                        }
                    }
                    if constexpr (!STATIC_FUSED_QUANT_ANY_EARLY_ASYNC_SCALE_SKIP_PREPACK_REWRITE) {
                        quant_stage_p_scale_tmem(idx, buf, wait_remote_p_quant);
                    }
                    continue;
                }
                if constexpr (STATIC_EXTERNAL_SPLIT_P_QUANT_2WG) {
                    float2 scores_half[32];
                    const uint32_t score_tt_base = wait_for_scores(idx);
                    #pragma unroll
                    for (int local_ii = 0; local_ii < SPLIT_2WG_BLOCKS_PER_WG; ++local_ii) {
                        const int ii = split_mx_block_begin + local_ii;
                        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(scores_half[local_ii * 16 + 0].x), "=f"(scores_half[local_ii * 16 + 0].y), "=f"(scores_half[local_ii * 16 + 1].x), "=f"(scores_half[local_ii * 16 + 1].y), "=f"(scores_half[local_ii * 16 + 2].x), "=f"(scores_half[local_ii * 16 + 2].y), "=f"(scores_half[local_ii * 16 + 3].x), "=f"(scores_half[local_ii * 16 + 3].y),
                              "=f"(scores_half[local_ii * 16 + 4].x), "=f"(scores_half[local_ii * 16 + 4].y), "=f"(scores_half[local_ii * 16 + 5].x), "=f"(scores_half[local_ii * 16 + 5].y), "=f"(scores_half[local_ii * 16 + 6].x), "=f"(scores_half[local_ii * 16 + 6].y), "=f"(scores_half[local_ii * 16 + 7].x), "=f"(scores_half[local_ii * 16 + 7].y),
                              "=f"(scores_half[local_ii * 16 + 8].x), "=f"(scores_half[local_ii * 16 + 8].y), "=f"(scores_half[local_ii * 16 + 9].x), "=f"(scores_half[local_ii * 16 + 9].y), "=f"(scores_half[local_ii * 16 + 10].x), "=f"(scores_half[local_ii * 16 + 10].y), "=f"(scores_half[local_ii * 16 + 11].x), "=f"(scores_half[local_ii * 16 + 11].y),
                              "=f"(scores_half[local_ii * 16 + 12].x), "=f"(scores_half[local_ii * 16 + 12].y), "=f"(scores_half[local_ii * 16 + 13].x), "=f"(scores_half[local_ii * 16 + 13].y), "=f"(scores_half[local_ii * 16 + 14].x), "=f"(scores_half[local_ii * 16 + 14].y), "=f"(scores_half[local_ii * 16 + 15].x), "=f"(scores_half[local_ii * 16 + 15].y)
                            : "r"(score_tt_base + ii * 32));
                    }
                    tensor_load_wait();
                    arrive_score_copy_done(idx, buf);

                    const int global_k_row = m_tile * C::Mb + global_row;
                    const float2 payload_scale = {1.0f / g.p_output_scale, 1.0f / g.p_output_scale};
                    const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                    #pragma unroll
                    for (int local_block = 0; local_block < SPLIT_2WG_BLOCKS_PER_WG; ++local_block) {
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj) {
                            const int local_si = local_block * 16 + jj;
                            const int global_si = (split_mx_block_begin + local_block) * 16 + jj;
                            const int global_q_col0 = global_idx * C::Nb + global_si * 2;
                            const int global_q_col1 = global_q_col0 + 1;
                            const float lse0 = external_lse_log2_smem[split_lse_slot][global_si * 2];
                            const float lse1 = external_lse_log2_smem[split_lse_slot][global_si * 2 + 1];
                            scores_half[local_si] = __fmul2_rn(scores_half[local_si], scale_2);
                            scores_half[local_si].x = (global_q_col0 >= global_k_row)
                                ? fp4pv_exp2_approx(scores_half[local_si].x - lse0)
                                : 0.0f;
                            scores_half[local_si].y = (global_q_col1 >= global_k_row)
                                ? fp4pv_exp2_approx(scores_half[local_si].y - lse1)
                                : 0.0f;
                            scores_half[local_si] = __fmul2_rn(scores_half[local_si], payload_scale);
                        }
                    }
                    if constexpr (STATIC_MXFP4_K256) {
                        const int pair_buf = (idx >> 1) & 1;
                        const int pair_half = idx & 1;
                        if (warpgroup::groupid() == QUANT_WG0) {
                            fp4pv_pack_scores_half_to_stage_mxfp4_range<0, 2>(
                                p_fp4_stage_k256[pair_buf],
                                p_sc_stage_k256[pair_buf * 2 + pair_half],
                                scores_half,
                                global_row,
                                g.p_quant_mode,
                                pair_half * (C::Nb / 2));
                        } else {
                            fp4pv_pack_scores_half_to_stage_mxfp4_range<2, 4>(
                                p_fp4_stage_k256[pair_buf],
                                p_sc_stage_k256[pair_buf * 2 + pair_half],
                                scores_half,
                                global_row,
                                g.p_quant_mode,
                                pair_half * (C::Nb / 2));
                        }
                    } else {
                        if constexpr (STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE) {
                            static_assert(C::Nb == 128, "split-2WG K64 direct scale staging expects Nb=128");
                            const int p_half = (warpgroup::groupid() == QUANT_WG0) ? 0 : 1;
                            uint32_t local_p_half_scale = 0;
                            #pragma unroll
                            for (int local_block = 0; local_block < SPLIT_2WG_BLOCKS_PER_WG; ++local_block) {
                                const int mx_block = split_mx_block_begin + local_block;
                                uint32_t words[4];
                                uint8_t e8m0 = 0;
                                fp4pv_quantize_mxfp4_scores_group(
                                    &scores_half[local_block * 16], g.p_quant_mode, words, e8m0);
                                fp4pv_store_quantized_scores_group_mxfp4(
                                    p_fp4_stage[buf], words, global_row, mx_block);
                                local_p_half_scale |= static_cast<uint32_t>(e8m0) << (local_block * 8);
                            }
                            const uint16_t local_p_half_word =
                                static_cast<uint16_t>(local_p_half_scale & 0xffffu);
                            if constexpr (STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE_SWIZZLED) {
                                p_sc_k64_half_swizzled[buf][p_half][global_row & 31][(global_row >> 5) & 3] =
                                    local_p_half_word;
                            } else {
                                p_sc_k64_half_local[buf][p_half][global_row & 127] = local_p_half_word;
                            }
                            if constexpr (STATIC_LIGHT_P_PAYLOAD_LOCAL_PUBLISH) {
                                fp4pv_publish_shared_backing_proxy_only();
                            } else {
                                publish_shared_backing();
                            }
                            quant_wg_sync();
                            const int p_sc_slot = idx % SCORE_TMEM_SLOTS;
                            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                const int p_slot_phase = (p_k64_scale_slot_phase_mask >> p_sc_slot) & 1;
                                wait(p_k64_scale_slot_ready[p_sc_slot], p_slot_phase);
                                p_k64_scale_slot_phase_mask ^= (1 << p_sc_slot);
                            }
                            quant_wg_sync();
                            const int p_sc_offset = p_half == 0 ? ALIAS_P_SC_OFFSET : ALIAS_P_SC_K64_HALF1_OFFSET;
                            d_tt_p_sc p_sc_tm_cur = d_tt_p_sc(tt_scores.addr + p_sc_slot * C::Nb + p_sc_offset);
                            const int row_in_32 = warp::laneid();
                            if constexpr (STATIC_EARLY_PV_K64_SPLIT2_DIRECT_SCALE_SWIZZLED) {
                                const uint16_t *row_scales = &p_sc_k64_half_swizzled[buf][p_half][row_in_32][0];
                                const uint32_t row_tile0 = static_cast<uint32_t>(row_scales[0]);
                                const uint32_t row_tile1 = static_cast<uint32_t>(row_scales[1]);
                                const uint32_t row_tile2 = static_cast<uint32_t>(row_scales[2]);
                                const uint32_t row_tile3 = static_cast<uint32_t>(row_scales[3]);
                                fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
                                    p_sc_tm_cur,
                                    warpgroup::warpid() * 32,
                                    p_half == 0 ? row_tile0 : (row_tile0 << 16),
                                    p_half == 0 ? row_tile1 : (row_tile1 << 16),
                                    p_half == 0 ? row_tile2 : (row_tile2 << 16),
                                    p_half == 0 ? row_tile3 : (row_tile3 << 16));
                            } else {
                                auto pack_direct_half_row = [&](int scale_row) {
                                    const uint32_t half_word =
                                        static_cast<uint32_t>(p_sc_k64_half_local[buf][p_half][scale_row]) & 0xffffu;
                                    return p_half == 0 ? half_word : (half_word << 16);
                                };
                                fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
                                    p_sc_tm_cur,
                                    warpgroup::warpid() * 32,
                                    pack_direct_half_row(row_in_32 + 0 * 32),
                                    pack_direct_half_row(row_in_32 + 1 * 32),
                                    pack_direct_half_row(row_in_32 + 2 * 32),
                                    pack_direct_half_row(row_in_32 + 3 * 32));
                            }
                            fp4pv_tmem_store_wait();
                            quant_wg_sync();
                            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                                if constexpr (STATIC_EARLY_PV_K64_DIRECT_SCALE_REMOTE_SYNC) {
                                    if constexpr (C::CLUSTER_SIZE > 1) {
                                        if (q_tiles_this_cluster > 1) {
                                            if (cta_rank == 1) {
                                                tma::cluster::arrive(p_k64_scale_remote_ready[buf][p_half], 0);
                                            } else {
                                                const int scale_remote_phase =
                                                    (p_k64_scale_remote_phase_mask[p_half] >> buf) & 1;
                                                tma::cluster::wait(
                                                    p_k64_scale_remote_ready[buf][p_half], scale_remote_phase);
                                                p_k64_scale_remote_phase_mask[p_half] ^= (1 << buf);
                                            }
                                        }
                                    }
                                }
                                if (cta_rank == 0) {
                                    arrive(p_k64_sc_tmem_ready[p_sc_slot][p_half]);
                                }
                            }
                            if (warpgroup::warpid() == 0) {
                                publish_cluster_shared_backing_if_needed<C>();
                                if constexpr (C::CLUSTER_SIZE > 1) {
                                    if (cta_rank == 1 && warp::elect_leader()) {
                                        tma::cluster::arrive(p_k64_remote_ready[buf][p_half], 0);
                                    }
                                }
                            }
                            quant_wg_sync();
                            continue;
                        } else if constexpr (C::Nb == 64) {
                            if (warpgroup::groupid() == QUANT_WG0) {
                                fp4pv_pack_scores_half_to_stage_mxfp4_range<0, 1>(
                                    p_fp4_stage[buf], p_sc_stage[buf], scores_half, global_row, g.p_quant_mode);
                            } else {
                                fp4pv_pack_scores_half_to_stage_mxfp4_range<1, 2>(
                                    p_fp4_stage[buf], p_sc_stage[buf], scores_half, global_row, g.p_quant_mode);
                            }
                        } else {
                            if (warpgroup::groupid() == QUANT_WG0) {
                                fp4pv_pack_scores_half_to_stage_mxfp4_range<0, 2>(
                                    p_fp4_stage[buf], p_sc_stage[buf], scores_half, global_row, g.p_quant_mode);
                            } else {
                                fp4pv_pack_scores_half_to_stage_mxfp4_range<2, 4>(
                                    p_fp4_stage[buf], p_sc_stage[buf], scores_half, global_row, g.p_quant_mode);
                            }
                        }
	                    }
	                    publish_shared_backing();
	                    publish_cluster_shared_backing_if_needed<C>();
	                    quant_wg_sync();
	                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(p_remote_ready[buf], 0);
                        }
                    }
	                    warpgroup::arrive(p_quant_ready[buf]);
	                    continue;
	                } else {
                if constexpr (STATIC_ONLINE_MXFP4_STREAM_SCORE_RELOAD_HALF) {
                    if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                        wait_for_p_stage_reusable(idx, buf);
                    }
                    const uint32_t score_tt_base = wait_for_scores(idx);
                    float row_max_old = row_max;
                    float tile_row_max = base_types::constants<float>::neg_infty();
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        float2 scores_half[32];
                        fp4pv_load_score_half_tmem(scores_half, score_tt_base, half);
                        tensor_load_wait();
                        fp4pv_apply_online_causal_mask_score_half(scores_half, half, idx, m_tile);
                        #pragma unroll
                        for (int jj = 0; jj < 32; ++jj) {
                            asm volatile("{max.f32 %0, %1, %2, %3;}"
                                : "=f"(tile_row_max)
                                : "f"(tile_row_max), "f"(scores_half[jj].x), "f"(scores_half[jj].y));
                        }
                    }
                    asm volatile("{max.f32 %0, %1, %2;}"
                        : "=f"(row_max)
                        : "f"(row_max), "f"(tile_row_max));

                    float acc_scale = 1.0f;
                    if (idx >= m_tile || idx > 0) {
                        if (!ZERO_SCALE) {
                            acc_scale = softmax_rescale_factor_strict(row_max, row_max_old, SCALE_LOG2);
                        }
                    }

                    float2 tile_sum0 = {0.0f, 0.0f}, tile_sum1 = {0.0f, 0.0f};
                    if (!ZERO_SCALE) {
                        const float neg_max_scaled = row_max * (-SCALE_LOG2);
                        const float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                        const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                        #pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            float2 scores_half[32];
                            fp4pv_load_score_half_tmem(scores_half, score_tt_base, half);
                            tensor_load_wait();
                            fp4pv_apply_online_causal_mask_score_half(scores_half, half, idx, m_tile);
                            #pragma unroll
                            for (int qlocal = 0; qlocal < 2; ++qlocal) {
                                const int qid = half * 2 + qlocal;
                                uint32_t online_fixed_p_words[4];
                                #pragma unroll
                                for (int pack = 0; pack < 4; ++pack) {
                                    #pragma unroll
                                    for (int kk = 0; kk < 4; ++kk) {
                                        const int jj = qlocal * 16 + pack * 4 + kk;
                                        scores_half[jj] = __ffma2_rn(scores_half[jj], scale_2, neg_max_scaled_2);
                                        scores_half[jj].x = fp4pv_exp2_approx(scores_half[jj].x);
                                        scores_half[jj].y = fp4pv_exp2_approx(scores_half[jj].y);
                                        if ((jj & 1) == 0) {
                                            tile_sum0 = __fadd2_rn(tile_sum0, scores_half[jj]);
                                        } else {
                                            tile_sum1 = __fadd2_rn(tile_sum1, scores_half[jj]);
                                        }
                                    }
                                    online_fixed_p_words[pack] =
                                        fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(
                                            &scores_half[qlocal * 16 + pack * 4]);
                                }
                                fp4pv_store_quantized_scores_group_mxfp4_selected<
                                    STATIC_SCALAR_P_PAYLOAD_STORE,
                                    STATIC_STRIPED_P_PAYLOAD_STORE,
                                    STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                    STATIC_V2_P_PAYLOAD_STORE,
                                    STATIC_STAGGERED_P_PAYLOAD_STORE,
                                    STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                    p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            float2 scores_half[32];
                            fp4pv_load_score_half_tmem(scores_half, score_tt_base, half);
                            tensor_load_wait();
                            fp4pv_apply_online_causal_mask_score_half(scores_half, half, idx, m_tile);
                            #pragma unroll
                            for (int qlocal = 0; qlocal < 2; ++qlocal) {
                                const int qid = half * 2 + qlocal;
                                uint32_t online_fixed_p_words[4];
                                #pragma unroll
                                for (int pack = 0; pack < 4; ++pack) {
                                    #pragma unroll
                                    for (int kk = 0; kk < 4; ++kk) {
                                        const int jj = qlocal * 16 + pack * 4 + kk;
                                        scores_half[jj].x = isfinite(scores_half[jj].x) ? 0.0f : scores_half[jj].x;
                                        scores_half[jj].y = isfinite(scores_half[jj].y) ? 0.0f : scores_half[jj].y;
                                        scores_half[jj].x = fp4pv_exp2_approx(scores_half[jj].x);
                                        scores_half[jj].y = fp4pv_exp2_approx(scores_half[jj].y);
                                        if ((jj & 1) == 0) {
                                            tile_sum0 = __fadd2_rn(tile_sum0, scores_half[jj]);
                                        } else {
                                            tile_sum1 = __fadd2_rn(tile_sum1, scores_half[jj]);
                                        }
                                    }
                                    online_fixed_p_words[pack] =
                                        fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(
                                            &scores_half[qlocal * 16 + pack * 4]);
                                }
                                fp4pv_store_quantized_scores_group_mxfp4_selected<
                                    STATIC_SCALAR_P_PAYLOAD_STORE,
                                    STATIC_STRIPED_P_PAYLOAD_STORE,
                                    STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                    STATIC_V2_P_PAYLOAD_STORE,
                                    STATIC_STAGGERED_P_PAYLOAD_STORE,
                                    STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                    p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
                            }
                        }
                    }
                    arrive_score_copy_done(idx, buf);

                    tile_sum0 = __fadd2_rn(tile_sum0, tile_sum1);
                    const float tile_sum_scalar = tile_sum0.x + tile_sum0.y;
                    row_sum = row_sum * acc_scale + tile_sum_scalar;
                    const float correction = acc_scale;
                    if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                        if (corr_reuse_ready_count >= STATIC_ONLINE_MXFP4_CORR_SLOTS) {
                            const int reuse_phase = (corr_reuse_phase_mask >> corr_slot) & 1;
                            wait(rescale_finished[corr_slot], reuse_phase);
                            corr_reuse_phase_mask ^= (1 << corr_slot);
                        }
                    }
                    stats_store(corr_vec_smem[corr_slot], warpgroup::laneid(), correction);
                    warp::sync();
                    warp::arrive(corr_arrived[corr_slot]);
                    ++corr_reuse_ready_count;
	                    if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
	                                  !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_FORCE_SPARE &&
	                                  !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
                        const int softmax_warp = warpgroup::laneid() / 32;
                        const bool active_rescale = (idx > 0) && (correction < 0.999f);
                        const bool warp_needs_rescale = __any_sync(0xffffffffu, active_rescale);
                        if (warp::laneid() == 0) {
                            p_softmax_warp_needs_rescale[buf][softmax_warp] =
                                warp_needs_rescale ? 1 : 0;
                        }
                        quant_wg_sync();
                        if (warpgroup::laneid() == 0) {
                            p_nonfirst_mm2_ok[buf] =
                                (p_softmax_warp_needs_rescale[buf][0] |
                                 p_softmax_warp_needs_rescale[buf][1] |
                                 p_softmax_warp_needs_rescale[buf][2] |
                                 p_softmax_warp_needs_rescale[buf][3]) ? 0 : 1;
                        }
                        if constexpr (!STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC) {
                            quant_wg_sync();
                        }
                    }

                    quant_wg_sync();
                    if constexpr (STATIC_ONLINE_MXFP4_3WG) {
                        fp4pv_publish_shared_backing_proxy_only();
                    } else {
                        publish_shared_backing();
                    }
                    publish_cluster_shared_backing_if_needed<C>();
                    if constexpr (!STATIC_ONLINE_MXFP4_3WG) {
                        quant_wg_sync();
                    }
                    if constexpr (STATIC_ONLINE_MXFP4_DIRECT_P_READY) {
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                tma::cluster::arrive(p_remote_ready[buf], 0);
                            }
                        }
                        warpgroup::arrive(p_quant_ready[buf]);
                    } else {
                        warpgroup::arrive(p_pack_ready[buf]);
                    }
                    continue;
                }
	                if constexpr (STATIC_ONLINE_MXFP4_STREAM_SCORE_RELOAD) {
	                    if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
	                        wait_for_p_stage_reusable(idx, buf);
	                    }
                    const uint32_t score_tt_base = wait_for_scores(idx);
                    float row_max_old = row_max;
                    float tile_row_max = base_types::constants<float>::neg_infty();
                    #pragma unroll
                                for (int qid = 0; qid < C::Nb / 32; ++qid) {
                        float2 scores_chunk[16];
                        fp4pv_load_score_quarter_tmem(scores_chunk, score_tt_base, qid);
                        tensor_load_wait();
                        fp4pv_apply_online_causal_mask_score_quarter(scores_chunk, qid, idx, m_tile);
                        #pragma unroll
                        for (int jj = 0; jj < 16; ++jj) {
                            asm volatile("{max.f32 %0, %1, %2, %3;}"
                                : "=f"(tile_row_max)
                                : "f"(tile_row_max), "f"(scores_chunk[jj].x), "f"(scores_chunk[jj].y));
                        }
                    }
                    asm volatile("{max.f32 %0, %1, %2;}"
                        : "=f"(row_max)
                        : "f"(row_max), "f"(tile_row_max));

                    float acc_scale = 1.0f;
                    if (idx >= m_tile || idx > 0) {
                        if (!ZERO_SCALE) {
                            acc_scale = softmax_rescale_factor_strict(row_max, row_max_old, SCALE_LOG2);
                        }
                    }

                    float2 tile_sum0 = {0.0f, 0.0f}, tile_sum1 = {0.0f, 0.0f};
                    if (!ZERO_SCALE) {
                        const float neg_max_scaled = row_max * (-SCALE_LOG2);
                        const float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                        const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                        #pragma unroll
                        for (int qid = 0; qid < 4; ++qid) {
                            float2 scores_chunk[16];
                            uint32_t online_fixed_p_words[4];
                            fp4pv_load_score_quarter_tmem(scores_chunk, score_tt_base, qid);
                            tensor_load_wait();
                            fp4pv_apply_online_causal_mask_score_quarter(scores_chunk, qid, idx, m_tile);
                            #pragma unroll
                            for (int pack = 0; pack < 4; ++pack) {
                                #pragma unroll
                                for (int kk = 0; kk < 4; ++kk) {
                                    const int jj = pack * 4 + kk;
                                    scores_chunk[jj] = __ffma2_rn(scores_chunk[jj], scale_2, neg_max_scaled_2);
                                    scores_chunk[jj].x = fp4pv_exp2_approx(scores_chunk[jj].x);
                                    scores_chunk[jj].y = fp4pv_exp2_approx(scores_chunk[jj].y);
                                    if ((jj & 1) == 0) {
                                        tile_sum0 = __fadd2_rn(tile_sum0, scores_chunk[jj]);
                                    } else {
                                        tile_sum1 = __fadd2_rn(tile_sum1, scores_chunk[jj]);
                                    }
                                }
                                online_fixed_p_words[pack] =
                                    fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(&scores_chunk[pack * 4]);
                            }
                            fp4pv_store_quantized_scores_group_mxfp4_selected<
                                STATIC_SCALAR_P_PAYLOAD_STORE,
                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
                        }
                    } else {
                        #pragma unroll
                        for (int qid = 0; qid < 4; ++qid) {
                            float2 scores_chunk[16];
                            uint32_t online_fixed_p_words[4];
                            fp4pv_load_score_quarter_tmem(scores_chunk, score_tt_base, qid);
                            tensor_load_wait();
                            fp4pv_apply_online_causal_mask_score_quarter(scores_chunk, qid, idx, m_tile);
                            #pragma unroll
                            for (int pack = 0; pack < 4; ++pack) {
                                #pragma unroll
                                for (int kk = 0; kk < 4; ++kk) {
                                    const int jj = pack * 4 + kk;
                                    scores_chunk[jj].x = isfinite(scores_chunk[jj].x) ? 0.0f : scores_chunk[jj].x;
                                    scores_chunk[jj].y = isfinite(scores_chunk[jj].y) ? 0.0f : scores_chunk[jj].y;
                                    scores_chunk[jj].x = fp4pv_exp2_approx(scores_chunk[jj].x);
                                    scores_chunk[jj].y = fp4pv_exp2_approx(scores_chunk[jj].y);
                                    if ((jj & 1) == 0) {
                                        tile_sum0 = __fadd2_rn(tile_sum0, scores_chunk[jj]);
                                    } else {
                                        tile_sum1 = __fadd2_rn(tile_sum1, scores_chunk[jj]);
                                    }
                                }
                                online_fixed_p_words[pack] =
                                    fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(&scores_chunk[pack * 4]);
                            }
                            fp4pv_store_quantized_scores_group_mxfp4_selected<
                                STATIC_SCALAR_P_PAYLOAD_STORE,
                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
                        }
                    }
                    arrive_score_copy_done(idx, buf);

                    tile_sum0 = __fadd2_rn(tile_sum0, tile_sum1);
                    const float tile_sum_scalar = tile_sum0.x + tile_sum0.y;
                    row_sum = row_sum * acc_scale + tile_sum_scalar;
                    const float correction = acc_scale;
                    if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                        if (corr_reuse_ready_count >= STATIC_ONLINE_MXFP4_CORR_SLOTS) {
                            const int reuse_phase = (corr_reuse_phase_mask >> corr_slot) & 1;
                            wait(rescale_finished[corr_slot], reuse_phase);
                            corr_reuse_phase_mask ^= (1 << corr_slot);
                        }
                    }
                    stats_store(corr_vec_smem[corr_slot], warpgroup::laneid(), correction);
                    warp::sync();
                    warp::arrive(corr_arrived[corr_slot]);
                    ++corr_reuse_ready_count;
	                    if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
	                                  !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_FORCE_SPARE &&
	                                  !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
                        const int softmax_warp = warpgroup::laneid() / 32;
                        const bool active_rescale = (idx > 0) && (correction < 0.999f);
                        const bool warp_needs_rescale = __any_sync(0xffffffffu, active_rescale);
                        if (warp::laneid() == 0) {
                            p_softmax_warp_needs_rescale[buf][softmax_warp] =
                                warp_needs_rescale ? 1 : 0;
                        }
                        quant_wg_sync();
                        if (warpgroup::laneid() == 0) {
                            p_nonfirst_mm2_ok[buf] =
                                (p_softmax_warp_needs_rescale[buf][0] |
                                 p_softmax_warp_needs_rescale[buf][1] |
                                 p_softmax_warp_needs_rescale[buf][2] |
                                 p_softmax_warp_needs_rescale[buf][3]) ? 0 : 1;
                        }
                        if constexpr (!STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC) {
                            quant_wg_sync();
                        }
                    }

                    quant_wg_sync();
                    if constexpr (STATIC_ONLINE_MXFP4_3WG) {
                        fp4pv_publish_shared_backing_proxy_only();
                    } else {
                        publish_shared_backing();
                    }
                    publish_cluster_shared_backing_if_needed<C>();
                    if constexpr (!STATIC_ONLINE_MXFP4_3WG) {
                        quant_wg_sync();
                    }
                    if constexpr (STATIC_ONLINE_MXFP4_DIRECT_P_READY) {
                        if constexpr (C::CLUSTER_SIZE > 1) {
                            if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                                tma::cluster::arrive(p_remote_ready[buf], 0);
                            }
                        }
                        warpgroup::arrive(p_quant_ready[buf]);
                    } else {
                        warpgroup::arrive(p_pack_ready[buf]);
                    }
                    continue;
                }
                if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG) {
                    static_assert(C::Nb == 128 && C::CLUSTER_SIZE == 1,
                        "online split P quant is specialized for Nb128 cluster1");
                    const int p_half = (warpgroup::groupid() == QUANT_WG0) ? 0 : 1;
                    float2 scores_half[32];
                    const uint32_t score_tt_base = wait_for_scores(idx);
                    fp4pv_load_score_half_tmem(scores_half, score_tt_base, p_half);
                    tensor_load_wait();
                    arrive_score_copy_done(idx, buf);
                    if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                        wait_for_p_stage_reusable(idx, buf);
                    }
                    fp4pv_apply_online_causal_mask_score_half(scores_half, p_half, idx, m_tile);

                    float local_tile_max = base_types::constants<float>::neg_infty();
                    #pragma unroll
                    for (int jj = 0; jj < 32; ++jj) {
                        asm volatile("{max.f32 %0, %1, %2, %3;}"
                            : "=f"(local_tile_max)
                            : "f"(local_tile_max), "f"(scores_half[jj].x), "f"(scores_half[jj].y));
                    }
                    p_online_split_tile_max[buf][p_half][global_row] = local_tile_max;
                    publish_shared_backing();
                    warpgroup::arrive(p_online_split_max_ready[buf]);
                    const int split_max_phase = (p_online_split_max_phase_mask >> buf) & 1;
                    wait(p_online_split_max_ready[buf], split_max_phase);
                    p_online_split_max_phase_mask ^= (1 << buf);

                    const float row_max_old = row_max;
                    float tile_row_max = p_online_split_tile_max[buf][0][global_row];
                    tile_row_max = fmaxf(tile_row_max, p_online_split_tile_max[buf][1][global_row]);
                    asm volatile("{max.f32 %0, %1, %2;}"
                        : "=f"(row_max)
                        : "f"(row_max), "f"(tile_row_max));

                    float acc_scale = 1.0f;
                    if (idx >= m_tile || idx > 0) {
                        if (!ZERO_SCALE) {
                            acc_scale = softmax_rescale_factor_strict(row_max, row_max_old, SCALE_LOG2);
                        }
                    }

                    float2 tile_sum0 = {0.0f, 0.0f};
                    float2 tile_sum1 = {0.0f, 0.0f};
                    if (!ZERO_SCALE) {
                        const float neg_max_scaled = row_max * (-SCALE_LOG2);
                        const float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                        const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                        #pragma unroll
                        for (int qlocal = 0; qlocal < 2; ++qlocal) {
                            const int qid = p_half * 2 + qlocal;
                            uint32_t online_fixed_p_words[4];
                            #pragma unroll
                            for (int pack = 0; pack < 4; ++pack) {
                                #pragma unroll
                                for (int kk = 0; kk < 4; ++kk) {
                                    const int jj = qlocal * 16 + pack * 4 + kk;
                                    scores_half[jj] = __ffma2_rn(scores_half[jj], scale_2, neg_max_scaled_2);
                                    scores_half[jj].x = fp4pv_exp2_approx(scores_half[jj].x);
                                    scores_half[jj].y = fp4pv_exp2_approx(scores_half[jj].y);
                                    if ((jj & 1) == 0) {
                                        tile_sum0 = __fadd2_rn(tile_sum0, scores_half[jj]);
                                    } else {
                                        tile_sum1 = __fadd2_rn(tile_sum1, scores_half[jj]);
                                    }
                                }
                                online_fixed_p_words[pack] =
                                    fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(
                                        &scores_half[qlocal * 16 + pack * 4]);
                            }
                            fp4pv_store_quantized_scores_group_mxfp4_selected<
                                STATIC_SCALAR_P_PAYLOAD_STORE,
                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
                        }
                    } else {
                        #pragma unroll
                        for (int qlocal = 0; qlocal < 2; ++qlocal) {
                            const int qid = p_half * 2 + qlocal;
                            uint32_t online_fixed_p_words[4];
                            #pragma unroll
                            for (int pack = 0; pack < 4; ++pack) {
                                #pragma unroll
                                for (int kk = 0; kk < 4; ++kk) {
                                    const int jj = qlocal * 16 + pack * 4 + kk;
                                    scores_half[jj].x = isfinite(scores_half[jj].x) ? 0.0f : scores_half[jj].x;
                                    scores_half[jj].y = isfinite(scores_half[jj].y) ? 0.0f : scores_half[jj].y;
                                    scores_half[jj].x = fp4pv_exp2_approx(scores_half[jj].x);
                                    scores_half[jj].y = fp4pv_exp2_approx(scores_half[jj].y);
                                    if ((jj & 1) == 0) {
                                        tile_sum0 = __fadd2_rn(tile_sum0, scores_half[jj]);
                                    } else {
                                        tile_sum1 = __fadd2_rn(tile_sum1, scores_half[jj]);
                                    }
                                }
                                online_fixed_p_words[pack] =
                                    fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(
                                        &scores_half[qlocal * 16 + pack * 4]);
                            }
                            fp4pv_store_quantized_scores_group_mxfp4_selected<
                                STATIC_SCALAR_P_PAYLOAD_STORE,
                                STATIC_STRIPED_P_PAYLOAD_STORE,
                                STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                STATIC_V2_P_PAYLOAD_STORE,
                                STATIC_STAGGERED_P_PAYLOAD_STORE,
                                STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
                        }
                    }

                    tile_sum0 = __fadd2_rn(tile_sum0, tile_sum1);
                    const float tile_sum_scalar = tile_sum0.x + tile_sum0.y;
                    row_sum = row_sum * acc_scale + tile_sum_scalar;
                    publish_shared_backing();
                    warpgroup::arrive(p_online_k64_ready[buf][p_half]);
                    if (warpgroup::groupid() == QUANT_WG0) {
                        if (corr_reuse_ready_count >= STATIC_ONLINE_MXFP4_CORR_SLOTS) {
                            const int reuse_phase = (corr_reuse_phase_mask >> corr_slot) & 1;
                            wait(rescale_finished[corr_slot], reuse_phase);
                            corr_reuse_phase_mask ^= (1 << corr_slot);
                        }
                        stats_store(corr_vec_smem[corr_slot], warpgroup::laneid(), acc_scale);
                        warp::sync();
                        warp::arrive(corr_arrived[corr_slot]);
                        ++corr_reuse_ready_count;
                    }
                    continue;
                }
	                float2 scores_reg[C::Nb / 2];
                if constexpr (C::Mb < 128) {
                    if (!row_active) {
                        #pragma unroll
                        for (int qid = 0; qid < C::Nb / 2; ++qid) {
                            scores_reg[qid] = float2{
                                base_types::constants<float>::neg_infty(),
                                base_types::constants<float>::neg_infty()
                            };
                        }
                    }
                }
	                float tile_max = 0.0f;
                const int softmax_warp = warpgroup::laneid() / 32;
                const uint32_t score_tt_base = wait_for_scores(idx);
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 32; ++ii) {
                    if (row_active &&
                        (!STATIC_SPLIT_P_QUANT_2WG || (ii >= split_mx_block_begin && ii < split_mx_block_end))) {
                        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(scores_reg[ii * 16 + 0].x), "=f"(scores_reg[ii * 16 + 0].y), "=f"(scores_reg[ii * 16 + 1].x), "=f"(scores_reg[ii * 16 + 1].y), "=f"(scores_reg[ii * 16 + 2].x), "=f"(scores_reg[ii * 16 + 2].y), "=f"(scores_reg[ii * 16 + 3].x), "=f"(scores_reg[ii * 16 + 3].y),
                              "=f"(scores_reg[ii * 16 + 4].x), "=f"(scores_reg[ii * 16 + 4].y), "=f"(scores_reg[ii * 16 + 5].x), "=f"(scores_reg[ii * 16 + 5].y), "=f"(scores_reg[ii * 16 + 6].x), "=f"(scores_reg[ii * 16 + 6].y), "=f"(scores_reg[ii * 16 + 7].x), "=f"(scores_reg[ii * 16 + 7].y),
                              "=f"(scores_reg[ii * 16 + 8].x), "=f"(scores_reg[ii * 16 + 8].y), "=f"(scores_reg[ii * 16 + 9].x), "=f"(scores_reg[ii * 16 + 9].y), "=f"(scores_reg[ii * 16 + 10].x), "=f"(scores_reg[ii * 16 + 10].y), "=f"(scores_reg[ii * 16 + 11].x), "=f"(scores_reg[ii * 16 + 11].y),
                              "=f"(scores_reg[ii * 16 + 12].x), "=f"(scores_reg[ii * 16 + 12].y), "=f"(scores_reg[ii * 16 + 13].x), "=f"(scores_reg[ii * 16 + 13].y), "=f"(scores_reg[ii * 16 + 14].x), "=f"(scores_reg[ii * 16 + 14].y), "=f"(scores_reg[ii * 16 + 15].x), "=f"(scores_reg[ii * 16 + 15].y)
                            : "r"(score_tt_base + ii * 32));
                    }
                }
                tensor_load_wait();
                if constexpr (STATIC_SPLIT_P_QUANT_2WG) {
                    quant_wg_sync();
                }
                arrive_score_copy_done(idx, buf);
                if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                    wait_for_p_stage_reusable(idx, buf);
                }
                if constexpr (!EXTERNAL_COL_LSE) {
                    if (idx >= m_tile) {
                        int causal_col = (idx > m_tile) ? -1 : (int)warpgroup::laneid();
                        #pragma unroll
                        for (int qid = 0; qid < C::Nb / 2; ++qid) {
                            if (qid * 2 > causal_col) scores_reg[qid].x = base_types::constants<float>::neg_infty();
                            if (qid * 2 + 1 > causal_col) scores_reg[qid].y = base_types::constants<float>::neg_infty();
                        }
                    }
                }

                if constexpr (EXTERNAL_COL_LSE) {
                    const int global_k_row = m_tile * C::Mb + global_row;
                    float2 payload_scale = {1.0f, 1.0f};
                    if constexpr (MXFP4_PV) {
                        payload_scale = {1.0f / g.p_output_scale, 1.0f / g.p_output_scale};
                    }
                    const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                    #pragma unroll
                    for (int qid = 0; qid < 4; ++qid) {
                        if (!STATIC_SPLIT_P_QUANT_2WG || (qid >= split_mx_block_begin && qid < split_mx_block_end)) {
                            #pragma unroll
                            for (int jj = 0; jj < 16; ++jj) {
                                const int si = qid * 16 + jj;
                                const int global_q_col0 = global_idx * C::Nb + si * 2;
                                const int global_q_col1 = global_q_col0 + 1;
                                float lse0 = external_lse_log2_smem[split_lse_slot][si * 2];
                                float lse1 = external_lse_log2_smem[split_lse_slot][si * 2 + 1];
                                scores_reg[si] = __fmul2_rn(scores_reg[si], scale_2);
                                scores_reg[si].x = (global_q_col0 >= global_k_row)
                                    ? fp4pv_exp2_approx(scores_reg[si].x - lse0)
                                    : 0.0f;
                                scores_reg[si].y = (global_q_col1 >= global_k_row)
                                    ? fp4pv_exp2_approx(scores_reg[si].y - lse1)
                                    : 0.0f;
                                if constexpr (MXFP4_PV) {
                                    scores_reg[si] = __fmul2_rn(scores_reg[si], payload_scale);
                                }
                                if constexpr (!MXFP4_PV) {
                                    tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                                }
                            }
                        }
                    }
                    if constexpr (!MXFP4_PV) {
                        fp4pv_zero_raw_scale_stage(p_sc_stage[buf], warpgroup::laneid());
                    }
                    if constexpr (MXFP4_PV) {
                        if constexpr (STATIC_MXFP4_K256) {
                            const int pair_buf = (idx >> 1) & 1;
                            const int pair_half = idx & 1;
                            if constexpr (STATIC_SPLIT_P_QUANT_2WG) {
                                if (warpgroup::groupid() == QUANT_WG0) {
                                    fp4pv_pack_scores_to_stage_mxfp4_range<0, 2>(
                                        p_fp4_stage_k256[pair_buf],
                                        p_sc_stage_k256[pair_buf * 2 + pair_half],
                                        scores_reg,
                                        global_row,
                                        g.p_quant_mode,
                                        pair_half * (C::Nb / 2));
                                } else {
                                    fp4pv_pack_scores_to_stage_mxfp4_range<2, 4>(
                                        p_fp4_stage_k256[pair_buf],
                                        p_sc_stage_k256[pair_buf * 2 + pair_half],
                                        scores_reg,
                                        global_row,
                                        g.p_quant_mode,
                                        pair_half * (C::Nb / 2));
                                }
                            } else {
                                fp4pv_pack_scores_to_stage_mxfp4(
                                    p_fp4_stage_k256[pair_buf],
                                    p_sc_stage_k256[pair_buf * 2 + pair_half],
                                    scores_reg,
                                    global_row,
                                    g.p_quant_mode,
                                    pair_half * (C::Nb / 2));
                            }
                        } else {
                            if constexpr (STATIC_SPLIT_P_QUANT_2WG) {
                                if (warpgroup::groupid() == QUANT_WG0) {
                                    fp4pv_pack_scores_to_stage_mxfp4_range<0, 2>(
                                        p_fp4_stage[buf], p_sc_stage[buf], scores_reg, global_row, g.p_quant_mode);
                                } else {
                                    fp4pv_pack_scores_to_stage_mxfp4_range<2, 4>(
                                        p_fp4_stage[buf], p_sc_stage[buf], scores_reg, global_row, g.p_quant_mode);
                                }
                            } else {
                                fp4pv_pack_scores_to_stage_mxfp4(
                                    p_fp4_stage[buf], p_sc_stage[buf], scores_reg, global_row, g.p_quant_mode);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int mask = 16; mask > 0; mask >>= 1) {
                            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, mask));
                        }
                        if (warp::laneid() == 0) {
                            p_softmax_warp_amax[buf][softmax_warp] = tile_max;
                        }
                        quant_wg_sync();
                        if (warpgroup::laneid() == 0) {
                            float cta_amax = p_softmax_warp_amax[buf][0];
                            cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][1]);
                            cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][2]);
                            cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][3]);
                            p_cta_amax[buf] = cta_amax;
                        }
                        quant_wg_sync();
                        fp4pv_pack_scores_to_stage_and_scales(
                            p_fp4_stage[buf], p_sc_stage[buf], scores_reg, p_cta_amax[buf], global_row);
                    }
                    quant_wg_sync();
                    publish_shared_backing();
                    publish_cluster_shared_backing_if_needed<C>();
                    quant_wg_sync();
                    warpgroup::arrive(p_pack_ready[buf]);
                    continue;
                }

                float row_max_old = row_max;
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
                asm volatile("{max.f32 %0, %1, %2, %3;}" : "=f"(row_max) : "f"(row_max), "f"(lm0), "f"(lm7));
                float acc_scale = 1.0f;
                if (idx >= m_tile || idx > 0) {
                    if (!ZERO_SCALE) {
                        if constexpr (ONLINE) {
                            acc_scale = softmax_rescale_factor_strict(row_max, row_max_old, SCALE_LOG2);
                        } else {
                            if constexpr (STATIC_CONSUMER_MODE == 3 || STATIC_CONSUMER_MODE == 4) {
                                acc_scale = softmax_rescale_factor_fast(row_max, row_max_old, SCALE_LOG2, 10.f);
                            } else if constexpr (STATIC_CONSUMER_MODE == 0 || STATIC_CONSUMER_MODE == 2 || STATIC_CONSUMER_MODE == 5 || STATIC_CONSUMER_MODE == 7) {
                                acc_scale = softmax_rescale_factor_fast(row_max, row_max_old, SCALE_LOG2);
                            } else {
                                acc_scale = softmax_rescale_factor(row_max, row_max_old, SCALE_LOG2);
                            }
                        }
                    }
	                    if constexpr (!ONLINE) {
	                        stats_store(max_vec_smem[0], warpgroup::laneid(), acc_scale);
	                    }
	                }
	                    if constexpr (STATIC_ONLINE_MXFP4_PRE_P_QUANT_CORR_PUBLISH) {
		                    if (corr_reuse_ready_count >= STATIC_ONLINE_MXFP4_CORR_SLOTS) {
		                        const int reuse_phase = (corr_reuse_phase_mask >> corr_slot) & 1;
		                        wait(rescale_finished[corr_slot], reuse_phase);
		                        corr_reuse_phase_mask ^= (1 << corr_slot);
		                    }
		                    if (row_active) {
		                        stats_store(corr_vec_smem[corr_slot], warpgroup::laneid(), acc_scale);
		                    }
		                    warp::sync();
		                    warp::arrive(corr_arrived[corr_slot]);
		                    ++corr_reuse_ready_count;
	                }
	                if constexpr (!ONLINE) {
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
                        uint32_t online_fixed_p_words[4];
                        #pragma unroll
                        for (int pack = 0; pack < 4; ++pack) {
                            __align__(8) bf16_2 scores_bf_pack[4];
                            #pragma unroll
                            for (int kk = 0; kk < 4; ++kk) {
                                const int jj = pack * 4 + kk;
                                int si = qid * 16 + jj;
                                scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_max_scaled_2);
                                scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                                scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                                if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
                                              !STATIC_ONLINE_MXFP4_EARLY_CORR_PUBLISH &&
                                              !STATIC_ONLINE_MXFP4_FP32_P_PACK) {
                                    scores_bf_pack[kk] = __float22bfloat162_rn(scores_reg[si]);
                                }
                                if constexpr (!(MXFP4_PV && ONLINE)) {
                                    tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                                }
                                if ((jj & 1) == 0) tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                                else tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                            }
                            if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
                                          !STATIC_ONLINE_MXFP4_EARLY_CORR_PUBLISH) {
                                if constexpr (STATIC_ONLINE_MXFP4_FP32_P_PACK) {
                                    online_fixed_p_words[pack] =
                                        fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(
                                            &scores_reg[qid * 16 + pack * 4]);
                                } else {
                                    const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf_pack[0]);
                                    const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf_pack[2]);
                                    online_fixed_p_words[pack] =
                                        transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(
                                            e03, e47, FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF);
                                }
                            }
                        }
	                        if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
	                                      !STATIC_ONLINE_MXFP4_EARLY_CORR_PUBLISH) {
                            if (row_active) {
                                fp4pv_store_quantized_scores_group_mxfp4_selected<
                                    STATIC_SCALAR_P_PAYLOAD_STORE,
                                    STATIC_STRIPED_P_PAYLOAD_STORE,
                                    STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                    STATIC_V2_P_PAYLOAD_STORE,
                                    STATIC_STAGGERED_P_PAYLOAD_STORE,
                                    STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                    p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
                            }
                            if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64) {
                                if (qid == 1) {
                                    quant_wg_sync();
                                    fp4pv_publish_shared_backing_proxy_only();
                                    warpgroup::arrive(p_online_k64_ready[buf][0]);
                                }
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int qid = 0; qid < C::Nb / 32; ++qid) {
                        uint32_t online_fixed_p_words[4];
                        #pragma unroll
                        for (int pack = 0; pack < 4; ++pack) {
                            __align__(8) bf16_2 scores_bf_pack[4];
                            #pragma unroll
                            for (int kk = 0; kk < 4; ++kk) {
                                const int jj = pack * 4 + kk;
                                int si = qid * 16 + jj;
                                scores_reg[si].x = isfinite(scores_reg[si].x) ? 0.0f : scores_reg[si].x;
                                scores_reg[si].y = isfinite(scores_reg[si].y) ? 0.0f : scores_reg[si].y;
                                scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                                scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                                if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
                                              !STATIC_ONLINE_MXFP4_EARLY_CORR_PUBLISH &&
                                              !STATIC_ONLINE_MXFP4_FP32_P_PACK) {
                                    scores_bf_pack[kk] = __float22bfloat162_rn(scores_reg[si]);
                                }
                                if constexpr (!(MXFP4_PV && ONLINE)) {
                                    tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                                }
                                if ((jj & 1) == 0) tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                                else tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                            }
                            if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
                                          !STATIC_ONLINE_MXFP4_EARLY_CORR_PUBLISH) {
                                if constexpr (STATIC_ONLINE_MXFP4_FP32_P_PACK) {
                                    online_fixed_p_words[pack] =
                                        fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(
                                            &scores_reg[qid * 16 + pack * 4]);
                                } else {
                                    const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf_pack[0]);
                                    const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf_pack[2]);
                                    online_fixed_p_words[pack] =
                                        transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(
                                            e03, e47, FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF);
                                }
                            }
                        }
	                        if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM &&
	                                      !STATIC_ONLINE_MXFP4_EARLY_CORR_PUBLISH) {
                            if (row_active) {
                                fp4pv_store_quantized_scores_group_mxfp4_selected<
                                    STATIC_SCALAR_P_PAYLOAD_STORE,
                                    STATIC_STRIPED_P_PAYLOAD_STORE,
                                    STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
                                    STATIC_V2_P_PAYLOAD_STORE,
                                    STATIC_STAGGERED_P_PAYLOAD_STORE,
                                    STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
                                    p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
                            }
                            if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64) {
                                if (qid == 1) {
                                    quant_wg_sync();
                                    fp4pv_publish_shared_backing_proxy_only();
                                    warpgroup::arrive(p_online_k64_ready[buf][0]);
                                }
                            }
                        }
                    }
                }

                tile_sum0 = __fadd2_rn(tile_sum0, tile_sum1);
                const float tile_sum_scalar = tile_sum0.x + tile_sum0.y;
                if constexpr (!ONLINE) {
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                    if (acc_scale == 1.0f) {
                        row_sum += tile_sum_scalar;
                    } else {
                        row_sum = row_sum * acc_scale + tile_sum_scalar;
                    }
                } else {
                    const float row_sum_old = row_sum;
                    row_sum = row_sum * acc_scale + tile_sum_scalar;
                    float correction = 1.0f;
                    if constexpr (MXFP4_PV) {
                        // SageAttention3-style MXFP4 PV accumulates unnormalized exp tiles.
                        // The running output therefore rescales by the row-max alpha and
                        // applies 1 / row_sum only once in the final store.
                        correction = acc_scale;
                    } else {
                        const float prev_contrib = row_sum_old * acc_scale;
                        if (prev_contrib > 0.0f && row_sum > 0.0f && isfinite(prev_contrib) && isfinite(row_sum)) {
                            correction = prev_contrib / row_sum;
                        }
                    }
	                    if constexpr (!MXFP4_PV) {
	                        if (!(correction >= 0.0f) || !isfinite(correction)) {
	                            correction = 0.0f;
	                        }
	                    }
		                    if constexpr (!STATIC_ONLINE_MXFP4_PRE_P_QUANT_CORR_PUBLISH) {
		                        if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
		                            if (corr_reuse_ready_count >= STATIC_ONLINE_MXFP4_CORR_SLOTS) {
		                                const int reuse_phase = (corr_reuse_phase_mask >> corr_slot) & 1;
		                                wait(rescale_finished[corr_slot], reuse_phase);
		                                corr_reuse_phase_mask ^= (1 << corr_slot);
		                            }
		                        }
			                        if (row_active) {
			                            stats_store(corr_vec_smem[corr_slot], warpgroup::laneid(), correction);
			                        }
			                        warp::sync();
			                        warp::arrive(corr_arrived[corr_slot]);
			                        if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
			                            ++corr_reuse_ready_count;
			                        }
		                    }
		                            if constexpr (STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
	                                              !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_FORCE_SPARE &&
	                                              !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE) {
	                                const bool active_rescale =
	                                    (idx > 0) && (correction < 0.999f);
                                const bool warp_needs_rescale =
                                    __any_sync(0xffffffffu, active_rescale);
                                if (warp::laneid() == 0) {
                                    p_softmax_warp_needs_rescale[buf][softmax_warp] =
                                        warp_needs_rescale ? 1 : 0;
                                }
                                quant_wg_sync();
                                if (warpgroup::laneid() == 0) {
                                    p_nonfirst_mm2_ok[buf] =
                                        (p_softmax_warp_needs_rescale[buf][0] |
                                         p_softmax_warp_needs_rescale[buf][1] |
                                         p_softmax_warp_needs_rescale[buf][2] |
                                         p_softmax_warp_needs_rescale[buf][3]) ? 0 : 1;
                                }
                                if constexpr (!STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC) {
                                    quant_wg_sync();
                                }
                            }
			                    if constexpr (MXFP4_PV) {
	                        if constexpr (STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM) {
	                            if constexpr (STATIC_ONLINE_MXFP4_EARLY_CORR_PUBLISH) {
		                                #pragma unroll
		                                for (int qid = 0; qid < C::Nb / 32; ++qid) {
		                                    uint32_t online_fixed_p_words[4];
	                                    fp4pv_quantize_mxfp4_scores_group_fixed_scaled_rte(
	                                        &scores_reg[qid * 16], online_fixed_p_words);
	                                    if (row_active) {
	                                        fp4pv_store_quantized_scores_group_mxfp4_selected<
	                                            STATIC_SCALAR_P_PAYLOAD_STORE,
	                                            STATIC_STRIPED_P_PAYLOAD_STORE,
	                                            STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
	                                            STATIC_V2_P_PAYLOAD_STORE,
	                                            STATIC_STAGGERED_P_PAYLOAD_STORE,
	                                            STATIC_HALF_STAGGERED_P_PAYLOAD_STORE>(
	                                            p_fp4_stage[buf], online_fixed_p_words, global_row, qid);
	                                    }
	                                }
	                            } else {
                                // Payload words are stored while converting each MX block above.
                            }
                        } else {
                            const float2 mxfp4_online_p_scale = {
                                FP4PV_MXFP4_ONLINE_P_SCALE,
                                FP4PV_MXFP4_ONLINE_P_SCALE
                            };
	                            if (row_active) {
	                                uint32_t packed_online_p_scales = fp4pv_pack_scores_to_stage_mxfp4_scaled_rte<
	                                    STATIC_SCALAR_P_PAYLOAD_STORE,
	                                    STATIC_STRIPED_P_PAYLOAD_STORE,
	                                    STATIC_HALF_STRIPED_P_PAYLOAD_STORE,
	                                    STATIC_V2_P_PAYLOAD_STORE,
	                                    STATIC_STAGGERED_P_PAYLOAD_STORE,
	                                    STATIC_HALF_STAGGERED_P_PAYLOAD_STORE,
	                                    !STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM>(
	                                    p_fp4_stage[buf], p_sc_stage[buf], scores_reg,
	                                    mxfp4_online_p_scale, global_row);
	                                if constexpr (STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM) {
	                                    p_online_mxfp4_p_scale_words[buf][global_row & 127] = packed_online_p_scales;
	                                }
	                            }
                        }
                    }
                    if constexpr (!STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH) {
                        wait(rescale_finished[0], rescale_phase);
                        rescale_phase ^= 1;
                    }
                }
                if constexpr (ONLINE) {
                    if constexpr (!MXFP4_PV) {
                        const float inv_row_sum =
                            (row_sum > 0.0f && isfinite(row_sum)) ? __frcp_rn(row_sum) : 0.0f;
                        tile_max = 0.0f;
                        #pragma unroll
                        for (int q = 0; q < C::Nb / 32; ++q) {
                            __align__(8) bf16_2 scores_bf_reg[16];
                            #pragma unroll
                            for (int jj = 0; jj < 16; ++jj) {
                                const int si = q * 16 + jj;
                                scores_reg[si].x *= inv_row_sum;
                                scores_reg[si].y *= inv_row_sum;
                                tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                                scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[si]);
                            }
                            store_scores_quarter_to_localcta_scan(p_bf16_scan[buf], scores_bf_reg, q);
                        }
                    }
                }
                if constexpr (!(MXFP4_PV && ONLINE)) {
                    warp::sync();
                    #pragma unroll
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, mask));
                    }
                    if (warp::laneid() == 0) {
                        p_softmax_warp_amax[buf][softmax_warp] = tile_max;
                        if constexpr (NEED_NONFIRST_MM2_GUARD) {
                            const bool active_rescale =
                                (tile_sum_scalar > 0.0f) && isfinite(tile_sum_scalar) && (acc_scale < 1.0f);
                            p_softmax_warp_needs_rescale[buf][softmax_warp] = active_rescale ? 1 : 0;
                        }
                    }
                    quant_wg_sync();
                    if (warpgroup::laneid() == 0) {
                        float cta_amax = p_softmax_warp_amax[buf][0];
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][1]);
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][2]);
                        cta_amax = fmaxf(cta_amax, p_softmax_warp_amax[buf][3]);
                        p_cta_amax[buf] = cta_amax;
                        if constexpr (NEED_NONFIRST_MM2_GUARD) {
                            p_nonfirst_mm2_ok[buf] =
                                (p_softmax_warp_needs_rescale[buf][0] |
                                 p_softmax_warp_needs_rescale[buf][1] |
                                 p_softmax_warp_needs_rescale[buf][2] |
                                 p_softmax_warp_needs_rescale[buf][3]) ? 0 : 1;
                        }
                    }
                    quant_wg_sync();
                }
                if constexpr (!MXFP4_PV) {
                    fp4pv_zero_raw_scale_stage(p_sc_stage[buf], warpgroup::laneid());
                }
                if constexpr (ONLINE) {
                    if constexpr (MXFP4_PV) {
                        // Packed above while the correction WG rescales the previous output.
                    } else {
                        fp4pv_pack_scores_to_stage_payload_from_localcta_scan_row(
                            p_fp4_stage[buf], p_bf16_scan[buf], p_cta_amax[buf], global_row);
                        fp4pv_store_scales_from_localcta_scan_row(
                            p_sc_stage[buf], p_bf16_scan[buf], p_cta_amax[buf], global_row);
                    }
                } else {
                    fp4pv_pack_scores_to_stage_and_scales(
                        p_fp4_stage[buf], p_sc_stage[buf], scores_reg, p_cta_amax[buf], global_row);
                }
                if constexpr (MXFP4_PV) {
                    if constexpr (!(ONLINE && STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM && C::CLUSTER_SIZE == 1)) {
	                        if (row_active && idx >= m_tile) {
	                            const int valid_groups = (idx > m_tile) ? 0 : min((warpgroup::laneid() / 32) + 1, C::Nb / 32);
	                            if constexpr (STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM ||
	                                          STATIC_ONLINE_MXFP4_FIXED_P_SCALE_TMEM) {
                                fp4pv_zero_invalid_causal_payload_groups_mxfp4(
                                    p_fp4_stage[buf], global_row, valid_groups);
                            } else {
                                fp4pv_zero_invalid_causal_groups_mxfp4(
                                    p_fp4_stage[buf], p_sc_stage[buf], global_row, valid_groups);
                            }
                        }
                    }
                } else {
                    int valid_groups = tk_localcta::SCALES_PER_CHUNK_X;
                    if (idx >= m_tile) {
                        valid_groups = (idx > m_tile) ? 0 : ((warpgroup::laneid() / 16) + 1);
                    }
                    fp4pv_zero_invalid_causal_groups(
                        p_fp4_stage[buf], p_sc_stage[buf], global_row, valid_groups);
                }
                quant_wg_sync();
                if constexpr (STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM) {
                    const int p_sc_slot = idx % P_SCALE_TMEM_SLOTS;
                    if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                        if (quant_p_scale_ready_count >= P_SCALE_TMEM_SLOTS) {
                            const int reuse_phase = (quant_p_scale_reuse_phase_mask >> p_sc_slot) & 1;
                            wait(p_sc_tmem_reusable[p_sc_slot], reuse_phase);
                            quant_p_scale_reuse_phase_mask ^= (1 << p_sc_slot);
                        }
                    }
                    quant_wg_sync();
                    constexpr int P_SC_BASE = SCALE_TMEM_BASE + Q_SCALE_TMEM_WIDTH + K_SCALE_TMEM_WIDTH;
                    static_assert(P_SCALE_TMEM_SLOTS == 2, "online direct P-scale TMEM expects two scale slots");
                    static_assert(P_SC_BASE + 2 * P_SCALE_TMEM_WIDTH <= MAX_TENSOR_COLS,
                        "online direct P-scale TMEM ping-pong exceeds TMEM budget");
                    d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * P_SCALE_TMEM_WIDTH);
                    d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * P_SCALE_TMEM_WIDTH);
                    d_tt_p_sc p_sc_tm_cur = p_sc_slot == 0 ? p_sc_tm0 : p_sc_tm1;
                    const int source_row = warp::laneid();
                    const uint32_t row_tile0 = p_online_mxfp4_p_scale_words[buf][source_row + 0 * 32];
                    const uint32_t row_tile1 = (C::Mb >= 64)
                        ? p_online_mxfp4_p_scale_words[buf][source_row + 1 * 32]
                        : 0u;
                    const uint32_t row_tile2 = (C::Mb >= 96)
                        ? p_online_mxfp4_p_scale_words[buf][source_row + 2 * 32]
                        : 0u;
                    const uint32_t row_tile3 = (C::Mb >= 128)
                        ? p_online_mxfp4_p_scale_words[buf][source_row + 3 * 32]
                        : 0u;
                    fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
                        p_sc_tm_cur,
                        warpgroup::warpid() * 32,
                        row_tile0,
                        row_tile1,
                        row_tile2,
                        row_tile3);
                    fp4pv_tmem_store_wait();
                    quant_wg_sync();
                }
                if constexpr (STATIC_ONLINE_MXFP4_3WG) {
                    fp4pv_publish_shared_backing_proxy_only();
                } else {
                    publish_shared_backing();
                }
                publish_cluster_shared_backing_if_needed<C>();
                if constexpr (!STATIC_ONLINE_MXFP4_3WG) {
                    quant_wg_sync();
                }
                if constexpr (STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64) {
                    warpgroup::arrive(p_online_k64_ready[buf][1]);
                }
                if constexpr (STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH &&
                              !STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM &&
                              !STATIC_ONLINE_MXFP4_EARLY_P_READY &&
                              !STATIC_ONLINE_MXFP4_DUAL_OUTPUT_ACCUM_TMEM &&
                              !STATIC_ONLINE_MXFP4_P_READY_BEFORE_RESCALE_WAIT) {
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                }
                if constexpr (STATIC_ONLINE_MXFP4_DIRECT_P_READY) {
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(p_remote_ready[buf], 0);
                        }
                    }
                    if constexpr (STATIC_ONLINE_MXFP4_DIRECT_P_SCALE_TMEM) {
                        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                            if (cta_rank == 0) {
                                const int p_sc_slot = idx % P_SCALE_TMEM_SLOTS;
                                arrive(p_sc_tmem_ready[p_sc_slot]);
                            }
                            ++quant_p_scale_ready_count;
                        }
	                        } else {
                        if constexpr (!STATIC_ONLINE_MXFP4_SPLIT_P_READY_K64) {
                            warpgroup::arrive(p_quant_ready[buf]);
                        }
                    }
                } else {
                    warpgroup::arrive(p_pack_ready[buf]);
                }
                if constexpr (STATIC_ONLINE_MXFP4_P_READY_BEFORE_RESCALE_WAIT) {
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                }
                if constexpr (STATIC_ONLINE_MXFP4_EARLY_FIXED_P_PAYLOAD_PUBLISH &&
                              STATIC_ONLINE_MXFP4_SPLIT_OUTPUT_ACCUM) {
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                }
                }
            }
            if constexpr (!EXTERNAL_COL_LSE) {
                const bool final_row_active = warpgroup::laneid() < C::Mb;
                if constexpr (!STATIC_ONLINE_MXFP4_SPLIT_P_QUANT_2WG) {
                    if (final_row_active) {
                        stats_store(lse_smem[0], warpgroup::laneid(), row_max);
                        stats_store(max_vec_smem[0], warpgroup::laneid(), row_sum);
                    }
                    warp::sync();
                    if constexpr (STATIC_ONLINE_MXFP4_EARLY_P_READY) {
                        warp::arrive(stats_arrived[0]);
                    } else {
                        warp::arrive(corr_arrived[0]);
                    }
                } else {
                    const int p_half = (warpgroup::groupid() == QUANT_WG0) ? 0 : 1;
                    const int final_row = warpgroup::laneid();
                    if (final_row_active) {
                        p_online_split_tile_sum[0][p_half][final_row] = row_sum;
                    }
                    publish_shared_backing();
                    warpgroup::arrive(p_online_split_sum_ready[0]);
                    if (warpgroup::groupid() == QUANT_WG0) {
                        const int split_sum_phase = p_online_split_sum_phase_mask & 1;
                        wait(p_online_split_sum_ready[0], split_sum_phase);
                        p_online_split_sum_phase_mask ^= 1;
                        const float row_sum_full =
                            final_row_active
                                ? (p_online_split_tile_sum[0][0][final_row] +
                                   p_online_split_tile_sum[0][1][final_row])
                                : 0.0f;
                        if (final_row_active) {
                            stats_store(lse_smem[0], warpgroup::laneid(), row_max);
                            stats_store(max_vec_smem[0], warpgroup::laneid(), row_sum_full);
                        }
                        warp::sync();
                        warp::arrive(stats_arrived[0]);
                    }
                }
            }
        }
    } else if (warpgroup::groupid() == RELAY_WG) {
        if constexpr (STATIC_ONLINE_MXFP4_REG_TUNE && C::TOTAL_WGS == 4) {
            warpgroup::decrease_registers<STATIC_ONLINE_MXFP4_RELAY_REGS>();
        } else if constexpr (STATIC_SPLIT_P_QUANT) {
            warpgroup::decrease_registers<24>();
        }
        if constexpr (!STATIC_SPLIT_P_QUANT && !STATIC_ONLINE_MXFP4_DIRECT_P_READY) {
            int p_pack_phase_mask = 0;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                if constexpr (MXFP4_PV && EXTERNAL_COL_LSE) {
                    if (k_iter_end >= 0 && (t_coord.y < k_iter_begin || t_coord.y >= k_iter_end)) {
                        continue;
                    }
                }
                if (t_coord.y + cta_rank >= tiles_m) {
                    continue;
                }
                int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                if constexpr (EXTERNAL_COL_LSE) {
                    iters_per_task = score_tiles_m - t_coord.y * SCORE_ITERS_PER_M_TILE;
                }
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const int buf = fp4pv_p_stage_buf_for_idx<P_STAGE_SLOTS>(idx);
                    const int p_pack_phase = (p_pack_phase_mask >> buf) & 1;
                    wait(p_pack_ready[buf], p_pack_phase);
                    p_pack_phase_mask ^= (1 << buf);
                    if constexpr (MXFP4_PV) {
                        publish_shared_backing();
                        tk_localcta::subgroup_barrier_sync<128>();
                    } else {
                        finalize_localcta_p_scales_with_amax<C>(p_sc_stage[buf], p_cta_amax[buf], warpgroup::laneid());
                    }
                    publish_cluster_shared_backing_if_needed<C>();
                    tk_localcta::subgroup_barrier_sync<128>();
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(p_remote_ready[buf], 0);
                        }
                    }
                    warpgroup::arrive(p_quant_ready[buf]);
                }
            }
        }
    }
}

inline void check_fp4pv_p_tensor(const at::Tensor &t, int batch, int heads, int seqlen) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Float4_e2m1fn_x2, "P_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(t.dim() == 4, "P_fp4 must have shape (batch, heads, seqlen, seqlen/2)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads && t.size(2) == seqlen && t.size(3) == seqlen / 2,
                "P_fp4 must have shape (batch, heads, seqlen, seqlen/2)");
}

__device__ __forceinline__ float fp4pv_warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

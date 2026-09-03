template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg : public config_fp4pv<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    using base = config_fp4pv<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;
    static constexpr int TOTAL_WGS = 3;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_4wg : public config_fp4pv<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    using base = config_fp4pv<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_async_pscale : public config_fp4pv_3wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_ASYNC_P_SCALE_COPY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score : public config_fp4pv_3wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_SCORE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_force_persistent : public config_fp4pv_3wg_dual_score<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
    static constexpr int ONLINE_V_LOAD_WARPS = 1;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_async_pscale_force_persistent
    : public config_fp4pv_3wg_dual_score_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_ASYNC_P_SCALE_COPY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_force_persistent
    : public config_fp4pv_3wg_dual_score_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_DIRECT_P_SCALE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_force_persistent
    : public config_fp4pv_3wg_dual_score_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_FIXED_P_SCALE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_splitaccum_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_OUTPUT_ACCUM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_early_pready_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_P_READY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_earlycorr_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_CORR_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_FP32_P_PACK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_P_READY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_stagev_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_QUANT_REGS = 160;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_stagev_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_o48_stagev_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_stagev_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_OUTPUT_REGS = 48;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_QUANT_REGS = 176;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_stagev_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_QUANT_REGS = 168;
    static constexpr int ONLINE_OUTPUT_REGS = 48;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_asyncv_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_PRODUCER_ASYNC_V_SCALE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_asyncv_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_PRODUCER_ASYNC_V_SCALE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_pstage3_q168_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_earlycorr_q168_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_CORR_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_q168_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q168_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_q168_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_CORR_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q168_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q168_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o40_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_OUTPUT_REGS = 40;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o40_stagev_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o40_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_OUTPUT_REGS = 48;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_dualaccum_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_v1_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_V_LOAD_WARPS = 1;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_v3_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_V_LOAD_WARPS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p112_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_PRODUCER_REGS = 112;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p104_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_PRODUCER_REGS = 104;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p96_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_PRODUCER_REGS = 96;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p120_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_PRODUCER_REGS = 120;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p136_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_PRODUCER_REGS = 136;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_asyncv_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_PRODUCER_ASYNC_V_SCALE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_asyncv_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_PRODUCER_ASYNC_V_SCALE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o56_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_OUTPUT_REGS = 56;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q184_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_QUANT_REGS = 184;
    static constexpr int ONLINE_OUTPUT_REGS = 48;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_pstage3_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_streamscore_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_STREAM_SCORE_RELOAD = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_streamscore_half_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_STREAM_SCORE_RELOAD_HALF = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_earlycorr_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_CORR_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_CORR_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 4;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_PRODUCER_REGS = _ProducerRegs;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_PRODUCER_REGS = _ProducerRegs;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _VLoadWarps, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_vload_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_V_LOAD_WARPS = _VLoadWarps;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q184_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q184_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
    static constexpr bool ONLINE_EARLY_CORR_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q184_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q184_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q184_o48_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q184_o48_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 4;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q176_o40_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o40_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
    static constexpr bool ONLINE_EARLY_CORR_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o40_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q176_o40_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o40_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q176_o40_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 4;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_early_pready_split2wg_q176_o48_force_persistent
    : public config_fp4pv_4wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
    static constexpr bool ONLINE_DUAL_SCORE_TMEM = true;
    static constexpr bool ONLINE_FIXED_P_SCALE_TMEM = true;
    static constexpr bool ONLINE_FP32_P_PACK = true;
    static constexpr bool ONLINE_EARLY_P_READY = true;
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
    static constexpr bool ONLINE_SPLIT_P_QUANT_2WG = true;
    static constexpr int ONLINE_QUANT_REGS = 152;
    static constexpr int ONLINE_OUTPUT_REGS = 48;
    static constexpr int ONLINE_PRODUCER_REGS = 120;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent
    : public config_fp4pv_4wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
    static constexpr bool ONLINE_DUAL_SCORE_TMEM = true;
    static constexpr bool ONLINE_FIXED_P_SCALE_TMEM = true;
    static constexpr bool ONLINE_FP32_P_PACK = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = true;
    static constexpr int ONLINE_QUANT_REGS = _QuantRegs;
    static constexpr int ONLINE_OUTPUT_REGS = _OutputRegs;
    static constexpr int ONLINE_PRODUCER_REGS = _ProducerRegs;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_P_READY = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_EARLY_P_READY = true;
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
    static constexpr bool ONLINE_SPLIT_P_QUANT_2WG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_vload3_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int ONLINE_V_LOAD_WARPS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_rowsplit2wg_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_QUANT_ROW_2WG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent
    : public config_fp4pv_4wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
    static constexpr bool ONLINE_DUAL_SCORE_TMEM = true;
    static constexpr bool ONLINE_DIRECT_P_SCALE_TMEM = true;
    static constexpr bool ONLINE_FP32_P_PACK = true;
    static constexpr bool ONLINE_EARLY_P_READY = true;
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
    static constexpr bool ONLINE_SPLIT_P_QUANT_2WG = true;
    static constexpr bool ONLINE_SPLIT_P_QUANT_LOCALMAX_2WG = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = true;
    static constexpr int ONLINE_QUANT_REGS = _QuantRegs;
    static constexpr int ONLINE_OUTPUT_REGS = _OutputRegs;
    static constexpr int ONLINE_PRODUCER_REGS = _ProducerRegs;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_directpscale_localmax_leader_split2wg_dualaccum_directrescale_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_QUANT_LEADER_SUM_2WG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_directpscale_localmax_rowsplit2wg_dualaccum_directrescale_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_QUANT_ROW_2WG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_directpscale_localmax_rowsplit2wg_dualaccum_directrescale_pstage3_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_directpscale_localmax_rowsplit2wg_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pstage3_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_QUANT_REGS = _QuantRegs;
    static constexpr int ONLINE_OUTPUT_REGS = _OutputRegs;
    static constexpr int ONLINE_PRODUCER_REGS = _ProducerRegs;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_FP32_P_PACK = true;
    static constexpr bool ONLINE_SCORE_DERIVED_P_SCALE_PACK = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = true;
    static constexpr bool ONLINE_DECOUPLED_PV_ISSUE = true;
    static constexpr int P_STAGE_SLOTS = 4;
    static constexpr int ONLINE_QUANT_REGS = _QuantRegs;
    static constexpr int ONLINE_OUTPUT_REGS = _OutputRegs;
    static constexpr int ONLINE_PRODUCER_REGS = _ProducerRegs;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SCORE_DERIVED_P_PRESCALED_PACK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_rte_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int ONLINE_SCORE_DERIVED_P_SCALE_MODE = 0;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int ONLINE_SCORE_DERIVED_P_SCALE_MODE = 2;
    static constexpr bool ONLINE_SCORE_DERIVED_P_X1_SCALE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SCORE_DERIVED_P_FUSED_BLOCK_MAX = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool QK_SCALE_COORD_HEAD_INDEX = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_latepack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SCORE_DERIVED_P_PACK_SCALES_LATE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 2;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool QK_SCALE_COORD_HEAD_INDEX = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_qkscfix_nodirectrescale_dualaccum_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = false;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_dualaccum_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = false;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_qklogn_dualaccum_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_dualaccum_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_pairqk_dualaccum_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_dualaccum_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DEDICATED_SPARE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_pairqk_qklogn_dualaccum_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_pairqk_dualaccum_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_fusedmax_qkscfix_forcespare_pairqk_dualaccum_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int ONLINE_SCORE_DERIVED_P_SCALE_MODE = 2;
    static constexpr bool ONLINE_SCORE_DERIVED_P_FUSED_BLOCK_MAX = true;
    static constexpr bool QK_SCALE_COORD_HEAD_INDEX = true;
    static constexpr int P_STAGE_SLOTS = 2;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = false;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DEDICATED_SPARE = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_pairqk_singleout_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_FP32_P_PACK = true;
    static constexpr bool ONLINE_SCORE_DERIVED_P_SCALE_PACK = true;
    static constexpr bool ONLINE_SCORE_DERIVED_P_PRESCALED_PACK = true;
    static constexpr int ONLINE_SCORE_DERIVED_P_SCALE_MODE = 2;
    static constexpr bool ONLINE_SCORE_DERIVED_P_X1_SCALE_TMEM = true;
    static constexpr bool ONLINE_SCORE_DERIVED_P_FUSED_BLOCK_MAX = true;
    static constexpr bool QK_SCALE_COORD_HEAD_INDEX = true;
    static constexpr bool ONLINE_DECOUPLED_PV_ISSUE = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
    static constexpr int P_STAGE_SLOTS = 2;
    static constexpr int ONLINE_QUANT_REGS = _QuantRegs;
    static constexpr int ONLINE_OUTPUT_REGS = _OutputRegs;
    static constexpr int ONLINE_PRODUCER_REGS = _ProducerRegs;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DIAGONAL_CAUSAL_PAYLOAD_ZERO_ONLY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 2;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_TMA_V_PAYLOAD = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_TMA_V_SCALE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_pchain_c
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE = true;
    static constexpr bool ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static_assert(_Ex2EmuPeriod == -1 ||
                  (_Ex2EmuPeriod >= 4 && _Ex2EmuPeriod <= 16));
    static constexpr int ONLINE_STAGE2_EX2_EMU_PERIOD = _Ex2EmuPeriod;
    static constexpr uint32_t ONLINE_STAGE2_EX2_EMU_MASK =
        _Ex2EmuPeriod == -1
            ? 0xffffu
            : (0x0fu << (_Ex2EmuPeriod - 4));
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c
    : public config_fp4pv_stage2_ex2_alu<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE = true;
    static constexpr bool ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved
    : public config_fp4pv_stage2_ex2_alu_pchain_c<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_kvstage3
    : public config_fp4pv_stage2_ex2_alu_pchain_c<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int LOAD_STAGES = 3;
    static constexpr int K_SCALE_SMEM_SLOTS = LOAD_STAGES;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_kvstage3
    : public config_fp4pv_stage2_ex2_alu_pchain_c_kvstage3<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_qk_two_ahead
    : public config_fp4pv_stage2_ex2_alu_pchain_c<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_QK_TWO_AHEAD = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_qk_two_ahead
    : public config_fp4pv_stage2_ex2_alu_pchain_c_qk_two_ahead<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_lazy_output_rescale
    : public config_fp4pv_stage2_ex2_alu_pchain_c<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_LAZY_OUTPUT_RESCALE = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale
    : public config_fp4pv_stage2_ex2_alu_pchain_c_lazy_output_rescale<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
};

// Isolated correctness probe for the retained large-H mapping.  The deployed
// selector remains on the historical half-K contract until this route passes
// the full-K lifecycle and performance gates.
template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

#if TK_FA4_REGISTER_OUTPUT_PROBE_BUILD
template <bool _TripleScore>
struct config_fp4pv_fullk_register_output_4wg
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<
          16, 128, 128, 192, 128, 192, 104, 112, 1> {
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_REGISTER_OUTPUT_4WG = true;
    static constexpr bool ONLINE_REGISTER_OUTPUT_PROGRESSIVE_QK = false;
    static constexpr bool ONLINE_TRIPLE_SCORE_TMEM = _TripleScore;
    static constexpr bool ONLINE_LAZY_OUTPUT_RESCALE = false;
    static constexpr bool ONLINE_P_READY_BEFORE_RESCALE_WAIT = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = false;
    static constexpr bool ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT = false;
    static constexpr bool ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT = false;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
    static constexpr bool ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE = false;
    static constexpr bool ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE = false;
    static constexpr bool ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE = false;
    static constexpr bool ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY = false;
    static constexpr int ONLINE_QUANT_REGS = 192;
    static constexpr int ONLINE_OUTPUT_REGS = 104;
    static constexpr int ONLINE_PRODUCER_REGS = 112;
};

using config_fp4pv_fullk_register_output_4wg_2score =
    config_fp4pv_fullk_register_output_4wg<false>;
using config_fp4pv_fullk_register_output_4wg_3score =
    config_fp4pv_fullk_register_output_4wg<true>;

// Isolated hybrid ownership probe. WG2 retains DVO[0:64) in registers while
// WG3 retains DVO[64:128) in a 32 KiB FP32 shared-memory tile.
struct config_fp4pv_fullk_register_smem_hybrid
    : public config_fp4pv_fullk_register_output_4wg<false> {
    static constexpr bool ONLINE_REGISTER_SMEM_HYBRID = true;
};

struct config_fp4pv_fullk_register_output_progressive_n64
    : public config_fp4pv_fullk_register_output_4wg<false> {
    static constexpr bool ONLINE_REGISTER_OUTPUT_PROGRESSIVE_QK = true;
};

struct config_fp4pv_fullk_register_output_progressive_n32
    : public config_fp4pv_fullk_register_output_progressive_n64 {
    static constexpr bool ONLINE_REGISTER_OUTPUT_PROGRESSIVE_QK_N32 = true;
};

struct config_fp4pv_fullk_register_output_progressive_n64_diag
    : public config_fp4pv_fullk_register_output_progressive_n64 {
    static constexpr bool ONLINE_REGISTER_OUTPUT_PROGRESSIVE_DIAG = false;
    static constexpr bool ONLINE_REGISTER_OUTPUT_PROGRESSIVE_BOUNDARY_DIAG = true;
};

struct config_fp4pv_fullk_register_output_progressive_n64_producer_diag
    : public config_fp4pv_fullk_register_output_progressive_n64 {
    static constexpr bool ONLINE_REGISTER_OUTPUT_PROGRESSIVE_DIAG = false;
    static constexpr bool
        ONLINE_REGISTER_OUTPUT_PROGRESSIVE_PRODUCER_BOUNDARY_DIAG = true;
};

// Arithmetic control for the register recurrence. It keeps the candidate's
// non-lazy P construction and lifecycle traits, but uses the established
// three-role TMEM output accumulator.
struct config_fp4pv_fullk_register_output_tmem_control
    : public config_fp4pv_fullk_register_output_4wg<false> {
    static constexpr int TOTAL_WGS = 3;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_REGISTER_OUTPUT_4WG = false;
    static constexpr bool ONLINE_TRIPLE_SCORE_TMEM = false;
    static constexpr int ONLINE_QUANT_REGS = 200;
    static constexpr int ONLINE_OUTPUT_REGS = 56;
    static constexpr int ONLINE_PRODUCER_REGS = 112;
};

// Exact fallback ownership probe: preserve the non-lazy full-K arithmetic,
// but retain the online output in FP32 shared memory instead of TMEM or
// long-lived registers. The existing BF16 output tile remains the final TMA
// store staging area.
struct config_fp4pv_fullk_fp32_smem_output
    : public config_fp4pv_fullk_register_output_tmem_control {
    static constexpr bool ONLINE_FP32_SMEM_OUTPUT_ACCUM = true;
    static constexpr bool ONLINE_SPLIT_OUTPUT_ACCUM = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = false;
};
#endif
#if TK_FA4_OPTION_AB_PROBE_BUILD
struct config_fp4pv_option_b_nb64_dvo128
    : public config_fp4pv<128, 64, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
};
#endif

#if TK_FA4_TMEM224_PROBE_BUILD
struct config_fp4pv_tmem224_nb64_dvo128
    : public config_fp4pv<128, 64, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
};

struct config_fp4pv_tmem224_nb64_dvo64
    : public config_fp4pv<128, 64, 192, 64, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
};
#endif

// Isolated full-width checkpoint family.  The scale control keeps the exact
// fine-grained production scale format but serializes P/V scale reuse into one
// physical slot each.  The checkpoint variant additionally packs each
// consumed score-bank output into a dedicated FP16 cold tile before releasing
// that bank back to QK.
template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo,
          int _QuantRegs, int _OutputRegs, int _ProducerRegs,
          int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_single_scale_spare
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs,
          _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_NARROW_V_SCALE_TMEM = true;
    static constexpr bool ONLINE_SINGLE_PV_SCALE_TMEM = true;
    static constexpr int ONLINE_P_SCALE_TMEM_SLOTS = 1;
    static constexpr int ONLINE_P_SCALE_TMEM_CREDIT_LIMIT = 1;
    static constexpr bool ONLINE_P_SCALE_TMEM_TCGEN_REUSE = true;
    static constexpr bool ONLINE_V_SCALE_TMEM_TCGEN_REUSE = true;
    static constexpr bool ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE = false;
    static constexpr bool ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE = false;
    static constexpr bool ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE = false;
    static constexpr bool ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY = false;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
    static constexpr bool ONLINE_TCGEN_P_STAGE_REUSE = false;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = false;
    static constexpr bool ONLINE_LAZY_OUTPUT_RESCALE = false;
    static constexpr bool ONLINE_P_READY_BEFORE_RESCALE_WAIT = true;
};

// Two exact 32-column scale pages alternate ownership between QK and PV.
// Mode 1 drains the prior PV before the next QK; mode 2 only waits when QK
// reclaims the same physical page and therefore preserves one-epoch overlap.
template <int _Mode, int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk,
          int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs,
          int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_scale_pages
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_single_scale_spare<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs,
          _ProducerRegs, _ClusterSize> {
    static_assert(_Mode == 1 || _Mode == 2,
                  "exact scale-page mode must be serialized or overlapped");
    static constexpr int ONLINE_EXACT_SCALE_PAGE_MODE = _Mode;
    static constexpr bool ONLINE_SINGLE_PV_SCALE_TMEM = false;
    static constexpr int ONLINE_P_SCALE_TMEM_SLOTS = 2;
    static constexpr int ONLINE_P_SCALE_TMEM_CREDIT_LIMIT = 2;
};

using config_fp4pv_fullk_scale_pages_serial =
    config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_scale_pages<
        1, 16, 128, 128, 192, 128, 200, 56, 112, 1>;
using config_fp4pv_fullk_scale_pages_overlap =
    config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_scale_pages<
        2, 16, 128, 128, 192, 128, 200, 56, 112, 1>;

// Exact two-page scale lifecycle applied to the retained lazy-rescale parent.
// Output ownership and arithmetic remain inherited from headlazy; only the
// P/V scale credits and their TCGEN completion-backed reuse are separated.
struct config_fp4pv_fullk_scale_pages_lazy_overlap
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<
          16, 128, 128, 192, 128, 200, 56, 112, 1> {
    static constexpr int ONLINE_EXACT_SCALE_PAGE_MODE = 2;
    static constexpr bool ONLINE_NARROW_V_SCALE_TMEM = true;
    static constexpr bool ONLINE_SINGLE_PV_SCALE_TMEM = false;
    static constexpr int ONLINE_P_SCALE_TMEM_SLOTS = 2;
    static constexpr int ONLINE_P_SCALE_TMEM_CREDIT_LIMIT = 2;
    static constexpr bool ONLINE_P_SCALE_TMEM_TCGEN_REUSE = true;
    static constexpr bool ONLINE_V_SCALE_TMEM_TCGEN_REUSE = true;
    static constexpr bool ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE = false;
    static constexpr bool ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE = false;
    static constexpr bool ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE = false;
    static constexpr bool ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY = false;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
    static constexpr bool ONLINE_TCGEN_P_STAGE_REUSE = false;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo,
          int _QuantRegs, int _OutputRegs, int _ProducerRegs,
          int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_fp16_cold_output
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_single_scale_spare<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs,
          _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_FP16_COLD_OUTPUT_CHECKPOINT = true;
};

template <int _HeadGroupSize, int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk,
          int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs,
          int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_grouped_lazy_output_rescale_fullk
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs,
          _ProducerRegs, _ClusterSize> {
    static_assert(_HeadGroupSize > 0, "head-group mapping requires a positive group size");
    static constexpr int ONLINE_Q4_HEAD_GROUP_SIZE = _HeadGroupSize;
};

template <int _PersistentGridMultiplier, int _Ex2EmuPeriod, int _Mb, int _Nb,
          int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs,
          int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs,
          _ProducerRegs, _ClusterSize> {
    static_assert(_PersistentGridMultiplier == 1 || _PersistentGridMultiplier == 2,
                  "saturated full-K persistent probe supports one- or two-wave caps");
    static constexpr bool ONLINE_PERSISTENT_OUTER_TASK_LOOP = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_NORMALIZE_RESCALE_PHASE = true;
    static constexpr int PERSISTENT_GRID_MULTIPLIER = _PersistentGridMultiplier;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo,
          int _QuantRegs, int _OutputRegs, int _ProducerRegs,
          int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent_late_rescale_wait
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent<
          2, _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs,
          _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DEFER_DIRECT_RESCALE_WAIT_UNTIL_PV = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo,
          int _QuantRegs, int _OutputRegs, int _ProducerRegs,
          int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent_vstage3
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent<
          2, _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs,
          _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int V_LOAD_STAGES = 3;
};


#if (defined(TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) && \
     TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) || \
    (defined(TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD) && \
     TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD)
// Track-B candidate-only reproduction of the retained Stage-2 saturated-H
// topology.  The inner pipeline is inherited byte-for-byte from the proven
// full-K/head-interleaved/lazy-rescale specialization; only the grid-stride
// outer task loop and its two-wave physical-grid cap are selected here.
struct config_fp4pv_track_b_stage2_fullk_cap2_saturated
    : public config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent<
          2, 16, 128, 128, 192, 128, 200, 56, 112, 1> {
    using arithmetic_base =
        config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent<
            2, 16, 128, 128, 192, 128, 200, 56, 112, 1>;

    static constexpr bool TRACK_B_STAGE2_CAP2_TARGET = true;
    static constexpr bool TRACK_B_STAGE2_FULLGRID_CONTROL = false;
    static constexpr bool REQUIRE_TRACK_B_STAGE2_CAP2_PERSISTENT = true;

    static_assert(CLUSTER_SIZE == 1);
    static_assert(Mb == 128 && Nb == 128 && Dqk == 192 && Dvo == 128);
    static_assert(NUM_SOFTMAXXERS == 1);
    static_assert(TOTAL_WGS == 3 && NUM_THREADS == 384);
    static_assert(ONLINE_STAGE2_EX2_EMU_PERIOD == 16);
    static_assert(ONLINE_FULL_K_TILE);
    static_assert(ONLINE_Q4_HEAD_INTERLEAVED_MAPPING);
    static_assert(ONLINE_LAZY_OUTPUT_RESCALE);
    static_assert(ONLINE_PERSISTENT_OUTER_TASK_LOOP);
    static_assert(ONLINE_PERSISTENT_OUTER_NORMALIZE_RESCALE_PHASE);
    static_assert(FORCE_PERSISTENT_GRID);
    static_assert(PERSISTENT_GRID_MULTIPLIER == 2);
};

// A separately named exact-type control gives the static auditor an
// independent symbol while retaining every inherited arithmetic and task-loop
// trait.  Host dispatch launches it full-grid, so every CTA executes one task.
struct config_fp4pv_track_b_stage2_fullk_fullgrid_control final
    : public config_fp4pv_track_b_stage2_fullk_cap2_saturated {
    static constexpr bool TRACK_B_STAGE2_CAP2_TARGET = false;
    static constexpr bool TRACK_B_STAGE2_FULLGRID_CONTROL = true;
    static constexpr bool REQUIRE_TRACK_B_STAGE2_CAP2_PERSISTENT = false;

    static_assert(ONLINE_STAGE2_EX2_EMU_PERIOD == 16);
    static_assert(ONLINE_FULL_K_TILE);
    static_assert(ONLINE_Q4_HEAD_INTERLEAVED_MAPPING);
    static_assert(ONLINE_LAZY_OUTPUT_RESCALE);
    static_assert(ONLINE_PERSISTENT_OUTER_TASK_LOOP);
    static_assert(ONLINE_PERSISTENT_OUTER_NORMALIZE_RESCALE_PHASE);
    static_assert(PERSISTENT_GRID_MULTIPLIER == 2);
};
#endif

#if TK_FA4_DVO64_TWO_PASS_BUILD
struct config_fp4pv_dvo128_parent_fullk_ex2e16pc
    : public config_fp4pv_stage2_ex2_alu<
          16, 128, 128, 192, 128, 200, 56, 112, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk,
          int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_dvo64_two_pass
    : public config_fp4pv_stage2_ex2_alu<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk, 64,
          _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DVO64_TWO_PASS = true;
    static constexpr bool ONLINE_DUAL_SCORE_TMEM = false;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = false;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC = false;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = false;
    static constexpr bool ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE = false;
    static constexpr bool ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE = false;
    static constexpr bool ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE = false;
    static constexpr bool ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY = false;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
    static constexpr bool ONLINE_NARROW_V_SCALE_TMEM = true;
    static constexpr int ONLINE_P_SCALE_TMEM_SLOTS = 1;
    static constexpr int ONLINE_P_SCALE_TMEM_CREDIT_LIMIT = 1;
    static constexpr bool ONLINE_P_SCALE_TMEM_TCGEN_REUSE = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk,
          int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_dvo64_two_pass_static_regs
    : public config_fp4pv_stage2_ex2_alu_dvo64_two_pass<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk,
          _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DISABLE_REG_TUNE = true;
};

struct config_fp4pv_dvo64_lean2wg_ex2e16pc
    : public config_fp4pv<128, 128, 192, 64, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_DVO64_TWO_PASS = true;
    static constexpr int ONLINE_STAGE2_EX2_EMU_PERIOD = 16;
    static constexpr uint32_t ONLINE_STAGE2_EX2_EMU_MASK = 0xf000u;
};

struct config_fp4pv_dvo64_pair2wg_ex2e16pc
    : public config_fp4pv<128, 128, 192, 64, 2> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_DVO64_TWO_PASS = true;
    static constexpr int ONLINE_STAGE2_EX2_EMU_PERIOD = 16;
    static constexpr uint32_t ONLINE_STAGE2_EX2_EMU_MASK = 0xf000u;
};

struct config_fp4pv_dvo64_onewalk_reg2wg_ex2e16pc
    : public config_fp4pv<128, 128, 192, 64, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_DVO64_TWO_PASS = true;
    static constexpr bool DVO64_ONE_WALK_REGISTER_ACCUM = true;
    static constexpr int ONLINE_STAGE2_EX2_EMU_PERIOD = 16;
    static constexpr uint32_t ONLINE_STAGE2_EX2_EMU_MASK = 0xf000u;
};

struct config_fp4pv_dvo64_onewalk_fp16_coldswap_2wg_ex2e16pc
    : public config_fp4pv_dvo64_onewalk_reg2wg_ex2e16pc {
    // Half 1 remains in the active FP32 D64 accumulator. Half 0 is retained
    // in a densely packed FP16 D64 checkpoint and exchanged into the active
    // accumulator only while its PV update is in flight.
    static constexpr bool DVO64_ONE_WALK_FP16_COLD_SWAP = true;
};

struct config_fp4pv_dvo64_onewalk_bf16_coldswap_2wg_ex2e16pc
    : public config_fp4pv_dvo64_onewalk_reg2wg_ex2e16pc {
    // Exponent-range control for the packed cold-state lifecycle. BF16 keeps
    // the raw MXFP4 accumulator domain while using the same 32-column map.
    static constexpr bool DVO64_ONE_WALK_BF16_COLD_SWAP = true;
};

struct config_fp4pv_dvo64_onewalk_overlap32_2wg_ex2e16pc
    : public config_fp4pv_dvo64_onewalk_reg2wg_ex2e16pc {
    static constexpr bool DVO64_ONE_WALK_PARTIAL_OVERLAP = true;
};

struct config_fp4pv_dvo64_onewalk_overlap32_l2lpt_2wg_ex2e16pc
    : public config_fp4pv_dvo64_onewalk_overlap32_2wg_ex2e16pc {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
};


struct config_fp4pv_dvo64_onewalk_overlap32_queuepv_2wg_ex2e16pc
    : public config_fp4pv_dvo64_onewalk_overlap32_2wg_ex2e16pc {
    static constexpr bool DVO64_ONE_WALK_EAGER_SECOND_PV = true;
};

struct config_fp4pv_dvo64_onewalk_deferpv_2wg_ex2e16pc
    : public config_fp4pv_dvo64_onewalk_overlap32_queuepv_2wg_ex2e16pc {
    static constexpr bool DVO64_ONE_WALK_DEFER_SECOND_PV_WAIT = true;
};
#endif

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
struct config_fp4pv_dvo64_pv_residency_shell
    : public config_fp4pv<128, 128, 192, 64, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct config_fp4pv_nb64_dvo128_residency_shell
    : public config_fp4pv<128, 64, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
};

struct config_fp4pv_nb64_dual_dvo64_residency_shell
    : public config_fp4pv<128, 64, 192, 64, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
};

struct config_fp4pv_nb128_dual_dvo64_attention_shell
    : public config_fp4pv<128, 128, 192, 64, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int OUTPUT_CHUNK_COLS = 32;
    static constexpr int OUTPUT_CHUNKS = Dvo / OUTPUT_CHUNK_COLS;
    static constexpr bool REGISTER_ACCUM = false;
};

struct config_fp4pv_nb128_dvo128_multistep_shell
    : public config_fp4pv<128, 128, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int OUTPUT_CHUNK_COLS = 32;
    static constexpr int OUTPUT_CHUNKS = Dvo / OUTPUT_CHUNK_COLS;
    static constexpr bool REGISTER_ACCUM = false;
};

struct config_fp4pv_nb128_dvo128_register_multistep_shell
    : public config_fp4pv_nb128_dvo128_multistep_shell {
    static constexpr bool REGISTER_ACCUM = true;
};

// Compile-gated large-grid ownership checkpoint. This is intentionally a
// separate route from the one-query multistep and DVO64 shells.
struct config_fp4pv_nb128_dvo128_qstage2
    : public config_fp4pv<128, 128, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool QSTAGE2 = true;
};

// Separate two-stage KV pipeline experiment; production remains qstage2.
struct config_fp4pv_nb128_dvo128_qstage2_pipeline
    : public config_fp4pv_nb128_dvo128_qstage2 {
    static constexpr int PIPELINE_STAGES = 2;
};

struct config_fp4pv_nb128_dvo128_qstage2_two_n64
    : public config_fp4pv<128, 64, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = 16;
    static constexpr int NUM_THREADS = 512;
    static constexpr int PIPELINE_STAGES = 2;
    static constexpr bool ONLINE_FULL_K_TILE = true;
};
#endif

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_true_half_k256
    : public config_fp4pv_stage2_ex2_alu<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SCORE_DERIVED_K256 = true;
    static constexpr bool ONLINE_TRUE_HALF_QK_N256 = true;
    static constexpr bool ONLINE_NARROW_V_SCALE_TMEM = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = false;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC = false;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = false;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
    static constexpr bool ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE = false;
    static constexpr bool ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE = false;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_true_half_k256_pipe2
    : public config_fp4pv_stage2_ex2_alu_true_half_k256<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_TRUE_HALF_QK_PIPE2 = true;
    static constexpr int K_SCALE_SMEM_SLOTS = 4;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_dual_query_seq_halfgrid
    : public config_fp4pv_stage2_ex2_alu_pchain_c<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_PERSISTENT_OUTER_TASK_LOOP = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SCHEDWARP = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SCHEDWG = true;
    static constexpr int ONLINE_V_LOAD_WARPS = 1;
    static constexpr bool ONLINE_V_TMA_SINGLE_WARP_PUBLISH_FIX = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ADJACENT_QUERY_PAIR = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_EPOCH_BROADCAST = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ROLE_DONE_WG_SYNC = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_CARRY_PHASES = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_NO_RESET = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_LONG_SPIN = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_ACTIVE_ONLY = true;
    static constexpr int ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS = 5;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ROLE_START_SYNC = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_PRODUCER_DONE_SYNC = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_DRAIN_Q_FINISHED_BEFORE_DONE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_DRAIN_V_FINISHED_BEFORE_DONE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_Q_PHASES = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_CORR_RESCALE = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_SCORE_COPY = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE = false;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_dual_query_seq_3wg_halfgrid
    : public config_fp4pv_stage2_ex2_alu_dual_query_seq_halfgrid<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo,
          _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int TOTAL_WGS = 3;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SCHEDWG = false;
    static constexpr bool ONLINE_NARROW_V_SCALE_TMEM = true;
};

// Full-grid adjacent-query probe.  Unlike the persistent task-loop probes
// above, this config is consumed by a dedicated kernel with two independent
// score/P owners and one time-multiplexed output accumulator.
template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo,
          int _QuantRegs, int _OutputRegs, int _ProducerRegs,
          int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid
    : public config_fp4pv_stage2_ex2_alu_pchain_c<
          _Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo,
          _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static_assert(_ClusterSize == 1,
                  "the concurrent dual-query scaffold is rank-local");
    static constexpr int NUM_SOFTMAXXERS = 2;
    static constexpr int Q_SCALE_SMEM_SLOTS = 2;
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int OUTPUT_CHUNK_COLS = 32;
    static constexpr int OUTPUT_CHUNKS = _Dvo / OUTPUT_CHUNK_COLS;
    static constexpr bool ONLINE_DUAL_QUERY_CONCURRENT_HALFGRID = true;
    static constexpr bool ONLINE_NARROW_V_SCALE_TMEM = true;
};

struct config_fp4pv_stage2_ex2_alu_dual_query_one_n128
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

#if TK_FA4_OPTION_AB_PROBE_BUILD
// Option A owns adjacent M128 queries, reuses one N128 score hotplate, and
// deliberately replays QK so no score array survives outside TMEM.
struct config_fp4pv_option_a_replay
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};
#endif

// Two adjacent M128 queries retain independent N128 score and D128 output
// owners.  Q/K/P/V scales are overlaid into the inactive score bank.
struct config_fp4pv_stage2_ex2_alu_dual_query_two_score_overlay
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_stage2_ex2_alu_dual_query_one_n128_bf16ret
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_stage2_ex2_alu_dual_query_one_n128_fp16ret
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_stage2_ex2_alu_dual_query_one_n128_fp16scaledret
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_stage2_ex2_alu_dual_query_one_n128_evac
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_stage2_ex2_alu_dual_query_one_n128_3wg
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 152, 48, 80, 1> {
    static constexpr int TOTAL_WGS = 3;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_stage2_ex2_alu_dual_query_one_n128_smem_evac
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_stage2_ex2_alu_dual_query_one_n128_no_retention
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

// One CTA owns two adjacent M128 queries.  Physical K/V remain N128 while
// each query has an independent N64 score lane; the two halves form one
// online-softmax recurrence.
struct config_fp4pv_stage2_ex2_alu_dual_query_two_n64
    : public config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<
          16, 128, 128, 192, 128, 184, 48, 96, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_dual_query_parent_fullk_ex2e16pc
    : public config_fp4pv_stage2_ex2_alu<
          16, 128, 128, 192, 128, 200, 56, 112, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

#ifndef TK_FA4_DEEP_JOINT_STAGE
#define TK_FA4_DEEP_JOINT_STAGE 0
#endif

#ifndef TK_FA4_DEEP_JOINT_V_STAGE
#define TK_FA4_DEEP_JOINT_V_STAGE 0
#endif

template <int _Stages>
struct config_fp4pv_parent_fullk_deep_joint_ex2e16pc
    : public config_fp4pv_dual_query_parent_fullk_ex2e16pc {
    static_assert(_Stages >= 2 && _Stages <= 7,
                  "deep joint K/V ring stage count must be in [2, 7]");
    static constexpr int DEEP_JOINT_V_STAGES =
        TK_FA4_DEEP_JOINT_V_STAGE >= 2
            ? TK_FA4_DEEP_JOINT_V_STAGE
            : _Stages;
    static_assert(DEEP_JOINT_V_STAGES >= 2 && DEEP_JOINT_V_STAGES <= 7,
                  "deep joint V-stage diagnostic override must be in [2, 7]");
    static constexpr int LOAD_STAGES = _Stages;
    static constexpr int K_SCALE_SMEM_SLOTS = _Stages;
    static constexpr int V_LOAD_STAGES = DEEP_JOINT_V_STAGES;
    static constexpr int ONLINE_V_LOAD_WARPS = 1;
    static constexpr bool ONLINE_DEEP_JOINT_RING = true;
    static constexpr bool ONLINE_PRODUCER_ASYNC_V_SCALE_TMEM = false;
};

#if TK_FA4_DEEP_JOINT_STAGE >= 2
using config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc =
    config_fp4pv_parent_fullk_deep_joint_ex2e16pc<
        TK_FA4_DEEP_JOINT_STAGE>;

struct config_fp4pv_parent_fullk_deep_joint_vgrant_shared_ready_ex2e16pc
    : public config_fp4pv_parent_fullk_deep_joint_ex2e16pc<2> {
    static constexpr bool ONLINE_DEEP_V_GRANT_SHARED_READY = true;
};

struct config_fp4pv_parent_fullk_deep_joint_vgrant_cp_issued_ex2e16pc
    : public config_fp4pv_parent_fullk_deep_joint_ex2e16pc<2> {
    static constexpr bool ONLINE_DEEP_V_GRANT_CP_ISSUED = true;
};
#endif

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_stage2_ex2_alu_dual_query_static_halfgrid
    : public config_fp4pv_stage2_ex2_alu_pchain_c<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_TASK_LOOP = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SCHEDWARP = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_STATIC_STRIDE = true;
    static constexpr int ONLINE_V_LOAD_WARPS = 1;
    static constexpr bool ONLINE_V_TMA_SINGLE_WARP_PUBLISH_FIX = true;
    static constexpr bool ONLINE_NARROW_V_SCALE_TMEM = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = false;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split
    : public config_fp4pv_stage2_ex2_alu_pchain_c<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_SCORE_DERIVED_SPLIT_P_QUANT_2WG = true;
};

template <int _Ex2EmuPeriod, int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split_k64
    : public config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split<_Ex2EmuPeriod, _Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SCORE_DERIVED_K64_HALF_READY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pready_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_P_READY_BEFORE_RESCALE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_NARROW_V_SCALE_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_lazyorescale_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_LAZY_OUTPUT_RESCALE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_TASK_LOOP = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SCHEDWARP = true;
    static constexpr int ONLINE_V_LOAD_WARPS = 1;
    static constexpr bool ONLINE_V_TMA_SINGLE_WARP_PUBLISH_FIX = true;
};

#if TK_FA4_REAL_FWD_TIMELINE_PROBE_BUILD
// Exact trace-free and trace-only clones of the installed Dqk192/Dvo128
// production route.  Both are forced through a full-grid launch by the host
// probe dispatch; only the trace type opts into timestamp stores.
using config_fp4pv_real_fwd_timeline_parent =
    config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<
        128, 128, 192, 128, 200, 56, 112, 1>;

struct config_fp4pv_real_fwd_timeline_control
    : public config_fp4pv_real_fwd_timeline_parent {};

struct config_fp4pv_real_fwd_timeline_trace
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool REAL_FWD_TIMELINE_TRACE = true;
};

// Tail-only readiness probe for the exact same S4096/H64 production route.
// It performs nonblocking semaphore tests at the natural wait sites, retains
// miss masks in the PV issue leader's registers, and emits one record at task
// tail.  It deliberately does not opt into the detailed timestamp trace.
struct config_fp4pv_real_fwd_readiness_probe
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool REAL_FWD_READINESS_PROBE = true;
};

// Trace-free work-attribution and scheduling candidates.  S4096/H64 uses the
// installed persistouter/CLC parent above, while S8192/H16 uses the installed
// non-persistouter VTMA/VSTMA route.  Keeping both parents explicit prevents a
// long-S result from being attributed to the wrong production control.
using config_fp4pv_real_fwd_long_parent =
    config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<
        128, 128, 192, 128, 200, 56, 112, 1>;

struct config_fp4pv_real_fwd_long_control
    : public config_fp4pv_real_fwd_long_parent {};

// Sparse cycle-only attribution clones for the correction/PV dependency
// boundary.  The two controls are distinct kernel instantiations with no
// diagnostic trait; their event time remains the same-build trace-free
// reference.  Cycle records are supporting-only and never used for wall-time
// deltas.
struct config_fp4pv_real_fwd_high_corr_pv_control
    : public config_fp4pv_real_fwd_timeline_parent {};

// Production-derived native cta_group::2 full-FP4 route.  This deliberately
// starts from the same real-softmax/scorepack/direct-scale arithmetic as the
// authoritative parent, not from any paired-query overlay kernel.  The
// cluster2-specific handoff folds P payload and P-scale ownership into the
// group2 TCGEN completion and combines the rank-1 P/V readiness domains.
struct config_fp4pv_native_group2_fp4qk_fp4pv_lean
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<
          128, 128, 192, 128, 200, 56, 112, 2> {
    static constexpr bool ONLINE_NATIVE_GROUP2_LEAN_HANDOFF = true;
    static constexpr bool ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT = true;
    static constexpr bool ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT = true;
    static constexpr bool ONLINE_TCGEN_P_STAGE_REUSE = true;
    static constexpr bool ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE = true;
    static constexpr bool ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE = true;
    static constexpr bool ONLINE_TMA_V_PAYLOAD = true;
    static constexpr bool ONLINE_TMA_V_SCALE = true;

    static_assert(CLUSTER_SIZE == 2 && TOTAL_WGS == 3);
    static_assert(Mb == 128 && Nb == 128 && Dqk == 192 && Dvo == 128);
    static_assert(P_STAGE_SLOTS == 2);
    static_assert(ONLINE_FP32_P_PACK && ONLINE_SCORE_DERIVED_P_SCALE_PACK &&
                  ONLINE_SCORE_DERIVED_P_PRESCALED_PACK &&
                  ONLINE_SCORE_DERIVED_P_X1_SCALE_TMEM &&
                  ONLINE_SCORE_DERIVED_P_FUSED_BLOCK_MAX);
};

// Native-group2 score-destination ownership fix.  The lean repro used only
// each CTA's local p_copy_done credit before rank 0 collectively overwrote the
// two-rank score destination.  That is sufficient for idx 0/1, but not for the
// first slot wrap at idx 2.  These isolated routes add one pair-wide reuse
// generation per score slot; the diagnostic specialization additionally
// bounds the idx>=2 acquire and records the last per-rank lifecycle edge.
struct config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit
    : public config_fp4pv_native_group2_fp4qk_fp4pv_lean {
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT = true;
};

struct config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit_diag
    : public config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit {
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT_DIAG = true;
};

// Downstream-only continuation of the bounded score-credit diagnostic.  Fix3
// proved the complete QK/score-copy/V-feed lifecycle through idx3; this route
// preserves that evidence contract and records the subsequent group2 PV,
// output, release, and CTA-exit edges in otherwise-unused diagnostic words.
struct config_fp4pv_native_group2_fp4qk_fp4pv_downstream_diag
    : public config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit_diag {
    static constexpr bool ONLINE_NATIVE_GROUP2_DOWNSTREAM_DIAG = true;
};

// Production-attributable score-credit route with final row statistics on a
// distinct one-shot semaphore.  It retains the lean full-FP4 arithmetic and
// pair-wide score ownership without the downstream diagnostic marker path.
struct config_fp4pv_native_group2_fp4qk_fp4pv_split_final_stats
    : public config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit {
    static constexpr bool ONLINE_NATIVE_GROUP2_SPLIT_FINAL_STATS = true;
};

// LSE-only attribution/correction for the production-derived full-FP4 route.
// The score-derived FP4 P packing, reconstructed output-normalization sum,
// correction, and PV arithmetic remain unchanged.  A separate true-domain
// prequantization sum is used only for the reported LSE.
struct config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse
    : public config_fp4pv_native_group2_fp4qk_fp4pv_split_final_stats {
    static constexpr bool ONLINE_NATIVE_GROUP2_EXACT_LSE_SUM = true;
};

// Match the proven native-group2 BF16-PV LSE recurrence without changing the
// full-FP4 P/PV recurrence: retain an independent LSE-only row maximum until
// its base-2 rescale would be smaller than 2^-8.
struct config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse_threshold
    : public config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse {
    static constexpr bool ONLINE_NATIVE_GROUP2_EXACT_LSE_THRESHOLD = true;
};

// Isolated score-load ownership correction.  The arithmetic, scale layout,
// score-pair credit and exact-LSE recurrence are inherited unchanged; only the
// per-thread TCGEN score-load completion is fenced before the elected lane can
// publish the pair-wide overwrite credit.
struct config_fp4pv_native_group2_fp4qk_fp4pv_score_load_fence
    : public config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse_threshold {
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_LOAD_THREAD_FENCE = true;
};

// Ownership-correct score reuse. Every reader executes the TCGEN load fence,
// and each of the four reader warps on both CTA ranks publishes one completion
// credit. Rank 0 therefore observes count 8 before reusing a score bank.
struct config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_count8
    : public config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse_threshold {
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_LOAD_THREAD_FENCE = true;
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_READER_WARP_CREDIT = true;
};

// Spill-free ownership control.  Every reader still completes and fences its
// TCGEN load, then the complete 128-thread quant warpgroup rendezvous before
// one leader per CTA rank publishes the existing count-2 pair credit.  This
// preserves the legacy local p_copy_done generation and moves the cluster
// address operation out of the full score-register live range.
struct config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_wg_rendezvous
    : public config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse_threshold {
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_LOAD_THREAD_FENCE = true;
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_WG_RENDEZVOUS = true;
};

// Attribution-only immutable Q-scale layout.  The compact A path normally
// rewrites one 16-column Q-scale page for each of the three K64 MMAs before
// their single completion commit.  Keep the v7 arithmetic and score ownership
// unchanged, but retain all three Q-scale chunks at distinct TMEM addresses.
// Two narrow V-scale slots retain the inherited ping-pong phase/reuse
// lifecycle without collapsing the two production P-scale generations:
//   score 0..255, output 256..383, Q 384..431, K 432..447,
//   P0/P1 448..479, V0/V1 480..511.
struct config_fp4pv_native_group2_fp4qk_fp4pv_q_scale_three_page
    : public config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_wg_rendezvous {
    static constexpr bool ONLINE_NATIVE_GROUP2_Q_SCALE_THREE_PAGE = true;
};

// Diagnostic-only child of the ownership-correct v9 route.  It preserves the
// v9 arithmetic and scale map while enabling bounded selected-row attribution;
// the uninstrumented v9 specialization remains available in the same binary.
struct config_fp4pv_native_group2_fp4qk_fp4pv_q_scale_three_page_selected_trace
    : public config_fp4pv_native_group2_fp4qk_fp4pv_q_scale_three_page {
    static constexpr bool ONLINE_NATIVE_GROUP2_SELECTED_ROW_TRACE = true;
};

// Attribution-only immutable K-scale layout.  The selected-row trace first
// diverges at the score-bank wrap (idx 16) while A and B1 still agree on raw
// scores through idx 15.  Keep the compact single Q-scale page and all v9
// arithmetic/ownership unchanged, but retain the three K64 scale chunks at
// distinct TMEM addresses:
//   score 0..255, output 256..383, Q 384..399, K 400..447,
//   P0/P1 448..479, V0/V1 480..511.
struct config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page
    : public config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_wg_rendezvous {
    static constexpr bool ONLINE_NATIVE_GROUP2_K_SCALE_THREE_PAGE = true;
};

// Correct the native-group2 half-N K descriptor's physical tile bound without
// changing the frozen K3, Q3, or same-build parent specializations.  The base
// producer historically derived the clamp from C::Nb (32 logical N128 tiles
// at S4096), while this route's rank-owned descriptor has 64 physical N64
// tiles.  At the first wrap, idx16 must therefore address ranks 0/1 as tiles
// 32/33 rather than clamping both to tile31.
struct config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles
    : public config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page {
    static constexpr bool ONLINE_NATIVE_GROUP2_PHYSICAL_K_TILE_LIMIT = true;
};

// Production-derived hybrid: retain the validated collective cta_group::2 QK
// and score-ownership lifecycle, but let each CTA rank consume its own real P
// and full-D V tile with an independent cta_group::1 FP4 PV.  This is a PV-only
// lowering; it must never opt into ROWPAR2_RANKLOCAL because that would also
// lower QK and collective TMEM allocation/deallocation.
struct config_fp4pv_native_group2_fp4qk_local_group1_fp4pv
    : public config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles {
    static constexpr bool ONLINE_NATIVE_GROUP2_LOCAL_GROUP1_PV = true;

    static_assert(ONLINE_NATIVE_GROUP2_LEAN_HANDOFF);
    static_assert(ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT);
    static_assert(ONLINE_NATIVE_GROUP2_SCORE_PAIR_WG_RENDEZVOUS);
    static_assert(ONLINE_NATIVE_GROUP2_K_SCALE_THREE_PAGE);
    static_assert(ONLINE_NATIVE_GROUP2_PHYSICAL_K_TILE_LIMIT);
    static_assert(CLUSTER_SIZE == 2 && TOTAL_WGS == 3);
    static_assert(Mb == 128 && Nb == 128 && Dqk == 192 && Dvo == 128);
};

// Candidate-only score-release proxy.  The frozen hybrid publishes pair-wide
// score credit from the quant WG after a named 128-thread rendezvous.  This
// child keeps each reader's TCGEN acquire/fence and uses a count-4 local
// p_copy_done generation (one release per reader warp), but moves the cluster
// address operation into the PV issue leader.  One issue leader on each CTA
// rank acquires its complete local score generation and release-arrives rank
// 0's count-2 pair generation; rank 0's QK issue lane is the only pair-credit
// consumer.
struct config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_issue_proxy
    : public config_fp4pv_native_group2_fp4qk_local_group1_fp4pv {
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_WG_RENDEZVOUS =
        false;
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_READER_WARP_CREDIT =
        false;
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_ISSUE_PROXY = true;

    static_assert(ONLINE_NATIVE_GROUP2_SCORE_LOAD_THREAD_FENCE);
    static_assert(ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT);
    static_assert(!ONLINE_NATIVE_GROUP2_SCORE_PAIR_READER_WARP_CREDIT);
    static_assert(!ONLINE_NATIVE_GROUP2_SCORE_PAIR_WG_RENDEZVOUS);
    static_assert(ONLINE_NATIVE_GROUP2_LOCAL_GROUP1_PV);
    static_assert(CLUSTER_SIZE == 2 && TOTAL_WGS == 3);
};

// Bounded quant-reader score-release proxy.  Keep the validated hybrid-v2
// tensor-role mapping unchanged: rank 0 WG0 remains the sole group2 QK owner,
// and WG1 on each rank remains its legal rank-local group1 PV owner.  Each of
// the four quant reader warps releases one local p_copy_done credit after its
// per-thread TCGEN acquire/fence.  Quant warp 0 then consumes that complete
// local generation and publishes exactly one rank credit into rank 0's
// count-2 pair generation.  No issue warp consumes the local barrier.
struct config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_quant_reader_proxy
    : public config_fp4pv_native_group2_fp4qk_local_group1_fp4pv {
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_WG_RENDEZVOUS =
        false;
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_READER_WARP_CREDIT =
        false;
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_ISSUE_PROXY = false;
    static constexpr bool ONLINE_NATIVE_GROUP2_SCORE_PAIR_QUANT_READER_PROXY =
        true;

    static_assert(ONLINE_NATIVE_GROUP2_SCORE_LOAD_THREAD_FENCE);
    static_assert(ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT);
    static_assert(!ONLINE_NATIVE_GROUP2_SCORE_PAIR_READER_WARP_CREDIT);
    static_assert(!ONLINE_NATIVE_GROUP2_SCORE_PAIR_WG_RENDEZVOUS);
    static_assert(!ONLINE_NATIVE_GROUP2_SCORE_PAIR_ISSUE_PROXY);
    static_assert(ONLINE_NATIVE_GROUP2_LOCAL_GROUP1_PV);
    static_assert(CLUSTER_SIZE == 2 && TOTAL_WGS == 3);
};

// Compile-gated selected-row child of the physical-K-tile correction.  The
// uninstrumented child is the correctness candidate; this child exists only
// to prove all 32 raw-score hashes against B1 before the numerical ladder.
struct config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles_selected_trace
    : public config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles {
    static constexpr bool ONLINE_NATIVE_GROUP2_SELECTED_ROW_TRACE = true;
};

// Compile-gated selected-row child of K3.  The uninstrumented K3 and frozen
// Q3 specializations remain separate symbols in the same diagnostic binary.
struct config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_selected_trace
    : public config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page {
    static constexpr bool ONLINE_NATIVE_GROUP2_SELECTED_ROW_TRACE = true;
};

// The downstream diagnostic proved that rank 1 can observe the final-stats
// publication before it consumes correction(T-1) when both generations share
// corr_arrived[0].  Keep correction publication on corr_arrived and place only
// the final row-statistics epoch on the already-existing stats_arrived barrier.
// This is an isolated synchronization fix; QK/P/PV arithmetic and ownership are
// inherited byte-for-byte from the downstream score-credit route.
struct config_fp4pv_native_group2_fp4qk_fp4pv_split_final_stats_diag
    : public config_fp4pv_native_group2_fp4qk_fp4pv_downstream_diag {
    static constexpr bool ONLINE_NATIVE_GROUP2_SPLIT_FINAL_STATS = true;
};

// Dedicated one-wave persistent routes.  These retain the 3WG production
// arithmetic but opt into the scheduler-owned multitask/reuse lifecycle;
// static-stride is intentionally not used (it is known to deadlock at S4096).
struct config_fp4pv_real_fwd_long_corr_pv_control
    : public config_fp4pv_real_fwd_long_parent {};

// Parent-schedule controls for isolating the register-lifetime-only mapping
// rewrite used by the spill-free correction-lookahead kernels.  These retain
// the exact production arithmetic and output-consumer lifecycle.
struct config_fp4pv_real_fwd_high_corr_mapping_control
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool ONLINE_CORR_MAPPING_LOCALIZED = true;
};

struct config_fp4pv_real_fwd_long_corr_mapping_control
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool ONLINE_CORR_MAPPING_LOCALIZED = true;
};

// Original-mapping correction wait-only pipeline.  Unlike the cached-float
// lookahead below, these routes overlap only the corr semaphore wait with the
// preceding PV completion.  The correction LDS and all rescale arithmetic
// remain after PV completes, so no correction float crosses the PV wait.  The
// existing P-stage/folded-scale reuse arrival moves from the issue owner to the
// output owner after that post-PV rescale; no ownership object is added.
struct config_fp4pv_real_fwd_high_corr_wait_only_lookahead
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool ONLINE_CORR_PV_WAIT_ONLY_LOOKAHEAD = true;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
};

struct config_fp4pv_real_fwd_long_corr_wait_only_lookahead
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool ONLINE_CORR_PV_WAIT_ONLY_LOOKAHEAD = true;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
};

// Measured synchronization experiment: consume the current PV-ready edge
// before blocking on the next correction edge.  This removes the unconditional
// next-correction wait from the PV critical path; arithmetic and ownership are
// otherwise identical to the wait-only parent.
struct config_fp4pv_real_fwd_high_corr_wait_after_pv
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool ONLINE_CORR_PV_WAIT_ONLY_LOOKAHEAD = true;
    static constexpr bool ONLINE_CORR_PV_WAIT_AFTER_PV = true;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
};

struct config_fp4pv_real_fwd_long_corr_wait_after_pv
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool ONLINE_CORR_PV_WAIT_ONLY_LOOKAHEAD = true;
    static constexpr bool ONLINE_CORR_PV_WAIT_AFTER_PV = true;
    static constexpr bool ONLINE_ARRIVE_P_STAGE_REUSE = false;
};

struct config_fp4pv_real_fwd_high_corr_precheck
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool ONLINE_CORR_PRECHECK = true;
};
struct config_fp4pv_real_fwd_long_corr_precheck
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool ONLINE_CORR_PRECHECK = true;
};
struct config_fp4pv_real_fwd_high_corr_precheck_v15 : public config_fp4pv_real_fwd_high_corr_precheck {};
struct config_fp4pv_real_fwd_long_corr_precheck_v15 : public config_fp4pv_real_fwd_long_corr_precheck {};
struct config_fp4pv_real_fwd_high_corr_precheck_v16 : public config_fp4pv_real_fwd_high_corr_precheck {};
struct config_fp4pv_real_fwd_long_corr_precheck_v16 : public config_fp4pv_real_fwd_long_corr_precheck {};
struct config_fp4pv_real_fwd_high_corr_precheck_v18 : public config_fp4pv_real_fwd_high_corr_precheck {};
struct config_fp4pv_real_fwd_long_corr_precheck_v18 : public config_fp4pv_real_fwd_long_corr_precheck {};
struct config_fp4pv_real_fwd_high_corr_precheck_v19 : public config_fp4pv_real_fwd_high_corr_precheck {};
struct config_fp4pv_real_fwd_long_corr_precheck_v19 : public config_fp4pv_real_fwd_long_corr_precheck {};
struct config_fp4pv_real_fwd_high_corr_precheck_v20 : public config_fp4pv_real_fwd_high_corr_precheck {};
struct config_fp4pv_real_fwd_long_corr_precheck_v20 : public config_fp4pv_real_fwd_long_corr_precheck {};
// v22 A/B diagnostic routes. A retains the early observation but forces the
// established blocking consume edge. B uses an explicit warpgroup decision.
struct config_fp4pv_real_fwd_high_corr_precheck_v22_a : public config_fp4pv_real_fwd_high_corr_precheck { static constexpr bool ONLINE_CORR_PRECHECK_CONTROL = true; };
struct config_fp4pv_real_fwd_long_corr_precheck_v22_a : public config_fp4pv_real_fwd_long_corr_precheck { static constexpr bool ONLINE_CORR_PRECHECK_CONTROL = true; };
struct config_fp4pv_real_fwd_high_corr_precheck_v22_b : public config_fp4pv_real_fwd_high_corr_precheck { static constexpr bool ONLINE_CORR_PRECHECK_WG_UNIFORM = true; };
struct config_fp4pv_real_fwd_long_corr_precheck_v22_b : public config_fp4pv_real_fwd_long_corr_precheck { static constexpr bool ONLINE_CORR_PRECHECK_WG_UNIFORM = true; };
struct config_fp4pv_real_fwd_high_corr_precheck_v23_c : public config_fp4pv_real_fwd_high_corr_precheck_v22_b { static constexpr bool ONLINE_CORR_PRECHECK_WG_CONSUME_TRYWAIT = true; };
struct config_fp4pv_real_fwd_long_corr_precheck_v23_c : public config_fp4pv_real_fwd_long_corr_precheck_v22_b { static constexpr bool ONLINE_CORR_PRECHECK_WG_CONSUME_TRYWAIT = true; };
struct config_fp4pv_real_fwd_high_corr_precheck_v23_d : public config_fp4pv_real_fwd_high_corr_precheck_v22_b { static constexpr bool ONLINE_CORR_PRECHECK_CONTROL = true; };
struct config_fp4pv_real_fwd_long_corr_precheck_v23_d : public config_fp4pv_real_fwd_long_corr_precheck_v22_b { static constexpr bool ONLINE_CORR_PRECHECK_CONTROL = true; };
struct config_fp4pv_real_fwd_high_corr_precheck_v24_e : public config_fp4pv_real_fwd_high_corr_precheck_v22_b { static constexpr bool ONLINE_CORR_PRECHECK_WG_ALL_LANES = true; };
struct config_fp4pv_real_fwd_long_corr_precheck_v24_e : public config_fp4pv_real_fwd_long_corr_precheck_v22_b { static constexpr bool ONLINE_CORR_PRECHECK_WG_ALL_LANES = true; };
struct config_fp4pv_real_fwd_high_corr_precheck_v24_f : public config_fp4pv_real_fwd_high_corr_precheck_v22_b { static constexpr bool ONLINE_CORR_PRECHECK_NATIVE_CONSUME = true; };
struct config_fp4pv_real_fwd_long_corr_precheck_v24_f : public config_fp4pv_real_fwd_long_corr_precheck_v22_b { static constexpr bool ONLINE_CORR_PRECHECK_NATIVE_CONSUME = true; };
struct config_fp4pv_real_fwd_high_corr_precheck_v25_g : public config_fp4pv_real_fwd_high_corr_precheck_v22_a {};

struct config_fp4pv_real_fwd_long_corr_precheck_v25_g : public config_fp4pv_real_fwd_long_corr_precheck_v22_a {};

// Trace-free output-consumer software pipeline.  It changes only the order in
// which WG0 consumes the existing corr_arrived and pv_tmem_ready edges; all
// arithmetic, resource ownership, storage, and issue-side semantics inherit
// the exact production parents.
struct config_fp4pv_real_fwd_high_corr_lookahead
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool ONLINE_CORR_PV_LOOKAHEAD = true;
};

struct config_fp4pv_real_fwd_long_corr_lookahead
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool ONLINE_CORR_PV_LOOKAHEAD = true;
};

struct config_fp4pv_real_fwd_high_corr_pv_cycle
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool REAL_FWD_CORR_PV_CYCLE_DIAG = true;
    static constexpr int REAL_FWD_CORR_PV_CYCLE_MASK = 7;
};

struct config_fp4pv_real_fwd_long_corr_pv_cycle
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool REAL_FWD_CORR_PV_CYCLE_DIAG = true;
    static constexpr int REAL_FWD_CORR_PV_CYCLE_MASK = 7;
};

// The exact production lifecycle is sensitive to an output-WG-only trace
// delay.  These cycle-only clones add bounded clock-read padding to the
// sampled quant/issue owners so the three roles retain comparable progress.
// The padding performs no stores, waits, semaphore operations, or syncs.
struct config_fp4pv_real_fwd_high_corr_pv_cycle_balanced
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool REAL_FWD_CORR_PV_CYCLE_DIAG = true;
    static constexpr bool REAL_FWD_CORR_PV_CYCLE_BALANCE_PAD = true;
    static constexpr int REAL_FWD_CORR_PV_CYCLE_MASK = 7;
};

struct config_fp4pv_real_fwd_long_corr_pv_cycle_balanced
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool REAL_FWD_CORR_PV_CYCLE_DIAG = true;
    static constexpr bool REAL_FWD_CORR_PV_CYCLE_BALANCE_PAD = true;
    static constexpr int REAL_FWD_CORR_PV_CYCLE_MASK = 7;
};

template <int Mask>
struct config_fp4pv_real_fwd_high_corr_pv_cycle_masked
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool REAL_FWD_CORR_PV_CYCLE_DIAG = true;
    static constexpr int REAL_FWD_CORR_PV_CYCLE_MASK = Mask;
};

struct config_fp4pv_real_fwd_high_store_ceiling
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool REAL_FWD_SKIP_FINAL_TMA_STORE = true;
};

struct config_fp4pv_real_fwd_long_store_ceiling
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool REAL_FWD_SKIP_FINAL_TMA_STORE = true;
};

// Machine-real P-scale TMEM-service ceilings. Keep the exact direct-scale
// parent score, softmax, P-payload, lifecycle, QK, and PV paths. The kernel
// preloads both P-scale TMEM slots once and skips only the per-tile scale TMEM
// stores/completion wait. These are attribution ceilings, not accuracy
// candidates.
struct config_fp4pv_real_fwd_high_pscale_ceiling
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool REAL_FWD_P_SCALE_TMEM_SERVICE_CEILING = true;
};

struct config_fp4pv_real_fwd_long_pscale_ceiling
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool REAL_FWD_P_SCALE_TMEM_SERVICE_CEILING = true;
};

struct config_fp4pv_real_fwd_high_pchain
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE = true;
    static constexpr bool ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY = true;
};

struct config_fp4pv_real_fwd_long_pchain
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE = true;
    static constexpr bool ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY = true;
};

struct config_fp4pv_real_fwd_high_headlazy
    : public config_fp4pv_real_fwd_timeline_parent {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
    static constexpr bool ONLINE_LAZY_OUTPUT_RESCALE = true;
};

struct config_fp4pv_real_fwd_long_headlazy
    : public config_fp4pv_real_fwd_long_parent {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
    static constexpr bool ONLINE_LAZY_OUTPUT_RESCALE = true;
};

struct config_fp4pv_real_fwd_high_pchain_headlazy
    : public config_fp4pv_real_fwd_high_pchain {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
    static constexpr bool ONLINE_LAZY_OUTPUT_RESCALE = true;
};

struct config_fp4pv_real_fwd_long_pchain_headlazy
    : public config_fp4pv_real_fwd_long_pchain {
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = true;
    static constexpr bool ONLINE_LAZY_OUTPUT_RESCALE = true;
};
#endif

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int ONLINE_V_LOAD_WARPS = 2;
    static constexpr bool ONLINE_V_TMA_SINGLE_WARP_PUBLISH_FIX = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SCHEDWG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_k64halfready_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SCORE_DERIVED_K64_HALF_READY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_k64halfready_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SCORE_DERIVED_K64_HALF_READY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_diag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_CLUSTER_SLOT_LIFETIME_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_schedown_taskdiag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_CLUSTER_SLOT_LIFETIME_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_ACTIVE_ONLY = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_PRODUCER_DONE_SYNC = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_CORR_RESCALE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_SCORE_COPY = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_livemirror_taskdiag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_schedown_taskdiag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_CLUSTER_SLOT_LIFETIME_LIVE_MIRROR = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_TASK_LOOP = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SCHEDWARP = true;
    static constexpr int ONLINE_V_LOAD_WARPS = 2;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int ONLINE_V_LOAD_WARPS = 1;
    static constexpr bool ONLINE_V_TMA_SINGLE_WARP_PUBLISH_FIX = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_staticstride_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_STATIC_STRIDE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int ONLINE_V_LOAD_WARPS = 2;
    static constexpr bool ONLINE_V_TMA_SINGLE_WARP_PUBLISH_FIX = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SCHEDWG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_CARRY_PHASES = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_REUSED_TASK_BODIES = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_NO_RESET = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_ACTIVE_ONLY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_PRODUCER_DONE_SYNC = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_NO_RESET = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_startsync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ROLE_START_SYNC = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_CORR_RESCALE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_AFTER_STORE_WAIT = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_PV_WAIT_OUTPUT_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS = 6;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_FENCE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_alltasks_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_FENCE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_SCORE_COPY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_ISSUE_TAIL_SCORE_COPY_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_alltasks_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_vpipereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_longspin_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_LONG_SPIN = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_vpipereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_FENCE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_pvoutreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_vpipereset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_LEADER = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_AFTER_STORE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_PV_WAIT_OUTPUT_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_ACTIVE_ONLY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_startsync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ROLE_START_SYNC = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2fullv_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_TMA_V_PAYLOAD = true;
    static constexpr bool ONLINE_TMA_V_SCALE = true;
    static constexpr bool ONLINE_ROWPAR2_RANKLOCAL = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2fullv_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_V_MULTICAST = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcpt_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2fullv_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_SHARED_K_LIFETIME = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcpt_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_K_MULTICAST_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptshortdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_K_MULTICAST_SHORT_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptringdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_K_MULTICAST_RING_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptqkdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_K_MULTICAST_QK_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptskiptail_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcpt_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_ISSUE_TAIL_SCORE_COPY_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_V_MULTICAST_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2fullv_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST = true;
    static constexpr bool ONLINE_ROWPAR2_V_MULTICAST_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcshortdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_SHORT_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcscalediag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_SCALE_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymc2slotdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcscalediag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_2SLOT_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcclpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_ROWPAR2_V_MULTICAST_CLUSTER_PUBLISH = true;
};

// Corrected-full-K controls for the existing cluster2 row-parallel lifecycle.
// The first keeps rank-local K/V loads; the second uses the completion-backed
// shared-K and V-multicast protocols.  Both remain explicit probe routes.
struct config_fp4pv_cluster2_ranklocal_fullk
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2fullv_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<
          128, 128, 192, 128, 200, 56, 112, 2> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_cluster2_vmc_fullk
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<
          128, 128, 192, 128, 200, 56, 112, 2> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

struct config_fp4pv_cluster2_kmc_fullk
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcpt_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<
          128, 128, 192, 128, 200, 56, 112, 2> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

#if TK_FA4_KMC_TAIL_DRAIN_PROBE_BUILD
// K-only lifecycle candidates selected only by explicit probe routes.
// Mode 1 holds rank 0 alive until producer-only shared-K tail slots receive
// rank 1's completion. Mode 2 drains every final in-flight shared-K and
// K-scale slot, distinguishing the measured asymmetric edge from a broader
// task-tail lifetime requirement.
struct config_fp4pv_cluster2_kmc_fullk_unmatched_tail_drain
    : public config_fp4pv_cluster2_kmc_fullk {
    static constexpr int ONLINE_ROWPAR2_K_TAIL_DRAIN_MODE = 1;
};

struct config_fp4pv_cluster2_kmc_fullk_all_tail_drain
    : public config_fp4pv_cluster2_kmc_fullk {
    static constexpr int ONLINE_ROWPAR2_K_TAIL_DRAIN_MODE = 2;
};

// K-only completion-ownership controls. Both ranks receive TCGEN completion
// on a CTA-local barrier, wait locally, and then contribute an explicit local
// or remote arrival to rank 0's shared reuse barrier. The paired variant also
// retains every final K/K-scale epoch until both contributions complete.
struct config_fp4pv_cluster2_kmc_fullk_symmetric_completion
    : public config_fp4pv_cluster2_kmc_fullk {
    static constexpr bool ONLINE_ROWPAR2_SYMMETRIC_LOCAL_K_COMPLETION = true;
};

struct config_fp4pv_cluster2_kmc_fullk_symmetric_completion_all_tail_drain
    : public config_fp4pv_cluster2_kmc_fullk_all_tail_drain {
    static constexpr bool ONLINE_ROWPAR2_SYMMETRIC_LOCAL_K_COMPLETION = true;
};

// Candidate A multicasts only the K prefix consumed by both causal query
// ranks. The unmatched rank-1 tail uses CTA-local payload/scale transfers and
// CTA-local readiness/reuse barriers.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail
    : public config_fp4pv_cluster2_kmc_fullk {
    static constexpr bool ONLINE_ROWPAR2_COMMON_PREFIX_LOCAL_K_TAIL = true;
};

// Candidate B keeps candidate A's common-prefix multicast and genuinely
// rank-local unmatched tail unchanged. For the shared prefix only, both CTA
// ranks commit QK to their own k_finished, observe that completion locally,
// and then contribute one arrival to rank 0's shared payload-reuse barrier.
// K-scale publication remains at QK issue; deferring it is candidate C.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_symmetric_completion
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail {
    static constexpr bool ONLINE_ROWPAR2_SYMMETRIC_LOCAL_K_COMPLETION = true;
};

// Candidate C is intentionally separate from B. It retains candidate A's
// asymmetric K-payload completion, but delays shared K-scale reuse until each
// consuming CTA has observed a CTA-local completion covering QK. The local
// unmatched-tail payload/scale movement remains candidate A verbatim.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_deferred_kscale_completion
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail {
    static constexpr bool
        ONLINE_ROWPAR2_DEFER_K_SC_SHARED_UNTIL_LOCAL_QK_COMPLETION = true;
};

// Combined K-only control after B and C failed separately. It derives B, so
// both ranks already commit QK to local k_finished. The deferred-scale trait
// only moves shared K-scale publication below that existing wait; it must not
// add a second completion wait or change movement.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_combined_completion
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_symmetric_completion {
    static constexpr bool
        ONLINE_ROWPAR2_DEFER_K_SC_SHARED_UNTIL_LOCAL_QK_COMPLETION = true;
    static constexpr bool
        ONLINE_ROWPAR2_COMBINED_SYMMETRIC_DEFERRED_K_COMPLETION = true;
};

// Candidate D closes the pre-issue destination-arm race for the shared K
// payload. The sole TMA owner directly executes expect_tx on both its local
// completion barrier and rank 1's mapped barrier before issuing multicast.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_owner_arm
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_combined_completion {
    static constexpr bool
        ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_PAYLOAD_MBAR = true;
};

// Candidate E is exactly D plus the analogous K-scale destination-arm
// operation. Rank 0 directly arms rank 1's mapped k_scale_remote_ready before
// issuing the scale multicast. The rank-1 local tail remains unchanged.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_scale_owner_arm
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_owner_arm {
    static constexpr bool
        ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_SCALE_MBAR = true;
};

// Candidate F fixes the remaining unmatched-tail lifetime leak in E. The
// full-grid probe has no later task that can reuse this CTA's shared prefix
// barriers, so the rank-1-only tail must not publish K/K-scale completion to
// rank 0 and rank 0 must not advance a bookkeeping-only shared tail epoch.
// Payload, scale, QK completion, and scale reuse are entirely rank local.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_strict_local_tail
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_scale_owner_arm {
    static constexpr bool ONLINE_ROWPAR2_STRICT_LOCAL_K_TAIL = true;
};

// Candidate G mirrors the explicit async-proxy acquire used by the retained
// diagnostic and V-multicast consumers, but only after shared K-payload
// arrival. It changes no movement, barrier phase, completion, or tail path.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_payload_proxy_acquire
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_scale_owner_arm {
    static constexpr bool ONLINE_ROWPAR2_POST_K_PAYLOAD_PROXY_ACQUIRE = true;
};

// Candidate H is G plus the same acquire after shared K-scale arrival. The
// pair isolates payload visibility from scale visibility in one build.
struct config_fp4pv_cluster2_kmc_fullk_common_prefix_payload_scale_proxy_acquire
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_payload_proxy_acquire {
    static constexpr bool ONLINE_ROWPAR2_POST_K_SCALE_PROXY_ACQUIRE = true;
};

// Candidate I starts from E and corrects only the rank-0-to-rank-1
// K-scale-load-issued semaphore pair to explicit cluster release/acquire.
struct config_fp4pv_cluster2_kmc_scope_handoff_i
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_scale_owner_arm {
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_SCALE_HANDOFF = true;
};

// Candidate J starts from the same E control and corrects only rank 1's
// remote K/K-scale reuse publications and rank 0's matching waits. The
// K-scale-load-issued handoff intentionally retains its legacy CTA scope.
struct config_fp4pv_cluster2_kmc_scope_reuse_j
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_scale_owner_arm {
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_REUSE = true;
};

// Candidate K combines I and J, giving every active cross-CTA mbarrier pair
// in the K-only shared prefix explicit release.cluster/acquire.cluster scope.
struct config_fp4pv_cluster2_kmc_scope_full_k
    : public config_fp4pv_cluster2_kmc_scope_reuse_j {
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_SCALE_HANDOFF = true;
};

// Candidate L retains only common-prefix K-payload multicast. K scales are
// loaded, consumed, and released independently by each CTA. Candidate M is
// the paired attribution control: K payload is rank local while K scale alone
// is multicast. Both inherit the strict rank-1-local unmatched tail and use
// explicit cluster scope for whichever shared domain remains active.
struct config_fp4pv_cluster2_kmc_payload_only_l
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_strict_local_tail {
    static constexpr bool ONLINE_ROWPAR2_LOCAL_K_SCALE_SPLIT = true;
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_SCALE_HANDOFF = true;
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_REUSE = true;
};

struct config_fp4pv_cluster2_kmc_scale_only_m
    : public config_fp4pv_cluster2_kmc_fullk_common_prefix_strict_local_tail {
    static constexpr bool ONLINE_ROWPAR2_LOCAL_K_PAYLOAD_SPLIT = true;
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_SCALE_HANDOFF = true;
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_REUSE = true;
};

// Candidate N closes only the payload destination-arm/request-publication
// race exposed by L.  Both QK issue paths observe a per-slot publication made
// after rank 0 arms both payload destinations and issues the multicast.
struct config_fp4pv_cluster2_kmc_payload_issue_gate_n
    : public config_fp4pv_cluster2_kmc_payload_only_l {
    static constexpr bool ONLINE_ROWPAR2_K_PAYLOAD_ISSUE_GATE = true;
};

// Candidate O is the paired scale-only control.  It retains M's rank-local K
// payload and makes both QK issue paths observe the existing per-slot
// k_scale_load_issued publication before testing scale arrival.
struct config_fp4pv_cluster2_kmc_scale_issue_gate_o
    : public config_fp4pv_cluster2_kmc_scale_only_m {
    static constexpr bool ONLINE_ROWPAR2_K_SCALE_ISSUE_GATE = true;
};

// Candidate P isolates the payload destination-arm ordering exposed by L/N.
// Rank 0 must finish the prior shared-payload reuse wait before it rearms
// either destination k_arrived barrier for the next multicast transaction.
struct config_fp4pv_cluster2_kmc_payload_postreuse_arm_p
    : public config_fp4pv_cluster2_kmc_payload_only_l {
    static constexpr bool ONLINE_ROWPAR2_K_PAYLOAD_ARM_AFTER_REUSE = true;
};

// Candidate Q is the independent scale-domain control derived from M. Rank 0
// finishes the prior shared-scale reuse wait before rearming either
// destination k_scale_remote_ready barrier. Payload remains rank local.
struct config_fp4pv_cluster2_kmc_scale_postreuse_arm_q
    : public config_fp4pv_cluster2_kmc_scale_only_m {
    static constexpr bool ONLINE_ROWPAR2_K_SCALE_ARM_AFTER_REUSE = true;
};

// Candidate R composes only the two measured payload-domain corrections:
// P's reuse-before-rearm ordering and N's post-request issue publication.
// K scale remains rank local, so this is not a payload+scale multicast route.
struct config_fp4pv_cluster2_kmc_payload_postreuse_issue_gate_r
    : public config_fp4pv_cluster2_kmc_payload_postreuse_arm_p {
    static constexpr bool ONLINE_ROWPAR2_K_PAYLOAD_ISSUE_GATE = true;
};

// Candidate S is the independent scale-domain composition. It retains Q's
// reuse-before-rearm ordering and adds O's post-request scale publication;
// K payload remains rank local.
struct config_fp4pv_cluster2_kmc_scale_postreuse_issue_gate_s
    : public config_fp4pv_cluster2_kmc_scale_postreuse_arm_q {
    static constexpr bool ONLINE_ROWPAR2_K_SCALE_ISSUE_GATE = true;
};

// Candidate T mirrors stable V slot reuse for the payload-only route. Rank 0
// waits its CTA-local QK completion and a separate remote-only completion in
// k_shared_finished before rearming. No issue-publication gate is retained.
struct config_fp4pv_cluster2_kmc_payload_split_local_remote_reuse_t
    : public config_fp4pv_cluster2_kmc_payload_postreuse_arm_p {
    static constexpr bool
        ONLINE_ROWPAR2_SPLIT_LOCAL_REMOTE_K_PAYLOAD_REUSE = true;
};

// Candidate U is the independent scale-domain control. The existing
// k_sc_shared_finished storage carries only rank 1's completion; rank 0's
// local QK completion is observed separately before scale rearm.
struct config_fp4pv_cluster2_kmc_scale_split_local_remote_reuse_u
    : public config_fp4pv_cluster2_kmc_scale_postreuse_arm_q {
    static constexpr bool
        ONLINE_ROWPAR2_SPLIT_LOCAL_REMOTE_K_SCALE_REUSE = true;
};

// Candidate V closes the payload request/readiness loop by composing T's
// split local/remote completion with the already-isolated post-request issue
// gate. K scale remains rank local and the strict tail skips the gate.
struct config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_v
    : public config_fp4pv_cluster2_kmc_payload_split_local_remote_reuse_t {
    static constexpr bool ONLINE_ROWPAR2_K_PAYLOAD_ISSUE_GATE = true;
    static constexpr bool
        ONLINE_ROWPAR2_SPLIT_REUSE_ISSUE_GATE_COMPOSITION = true;
};

// Candidate W is the independent scale-domain composition. It retains U's
// rank-local payload and remote-only scale completion, then gates QK only
// after the owner has armed and issued the scale multicast request.
struct config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_w
    : public config_fp4pv_cluster2_kmc_scale_split_local_remote_reuse_u {
    static constexpr bool ONLINE_ROWPAR2_K_SCALE_ISSUE_GATE = true;
    static constexpr bool
        ONLINE_ROWPAR2_SPLIT_REUSE_ISSUE_GATE_COMPOSITION = true;
};

// Candidate X is V plus the one remaining consumer-side operation present in
// the stable V-multicast lifecycle: after the shared K-payload transaction is
// observed complete, establish async/shared proxy visibility before TCGEN
// reads the tile.  K scale remains rank local.
struct config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_proxy_x
    : public config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_v {
    static constexpr bool ONLINE_ROWPAR2_POST_K_PAYLOAD_PROXY_ACQUIRE = true;
    static constexpr bool
        ONLINE_ROWPAR2_SPLIT_REUSE_ISSUE_GATE_PROXY_COMPOSITION = true;
};

// Candidate Y is the independent scale-domain pair.  K payload remains rank
// local; only the shared K-scale arrival is followed by the same proxy
// visibility operation used by the stable V-scale consumer.
struct config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_proxy_y
    : public config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_w {
    static constexpr bool ONLINE_ROWPAR2_POST_K_SCALE_PROXY_ACQUIRE = true;
    static constexpr bool
        ONLINE_ROWPAR2_SPLIT_REUSE_ISSUE_GATE_PROXY_COMPOSITION = true;
};

// Candidate Z is X plus the matching producer-side ordering edge from the
// stable V-multicast path.  The elected rank-0 payload TMA issuer executes an
// async/shared proxy fence after issuing the multicast and before publishing
// the per-slot request gate.  No warp rendezvous or opposite-domain change is
// included.
struct config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_producer_proxy_z
    : public config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_proxy_x {
    static constexpr bool
        ONLINE_ROWPAR2_K_PAYLOAD_POST_TMA_PROXY_PUBLISH = true;
    static constexpr bool
        ONLINE_ROWPAR2_SPLIT_REUSE_PRODUCER_CONSUMER_PROXY_COMPOSITION = true;
};

// Candidate AA is the independent scale-domain pair.  It derives Y, keeps K
// payload rank local, and inserts the producer proxy edge between scale TMA
// multicast issue and the existing scale-load-issued publication.
struct config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_producer_proxy_aa
    : public config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_proxy_y {
    static constexpr bool
        ONLINE_ROWPAR2_K_SCALE_POST_TMA_PROXY_PUBLISH = true;
    static constexpr bool
        ONLINE_ROWPAR2_SPLIT_REUSE_PRODUCER_CONSUMER_PROXY_COMPOSITION = true;
};

// Candidate AB isolates the first remaining stable-V synchronization edge on
// the only v26 branch that passed the semantic gate.  After rank 0 issues the
// shared K-scale TMA request, the existing single producer warp rendezvous
// completes before the same elected issuer executes the proxy fence and
// publishes scale-load-issued.  Payload remains rank local.
struct config_fp4pv_cluster2_kmc_scale_producer_rendezvous_ab
    : public config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_producer_proxy_aa {
    static constexpr int ONLINE_ROWPAR2_K_SCALE_PRODUCER_RENDEZVOUS_MODE = 1;
};

// Candidate AC derives AB and adds only the second stable-V rendezvous after
// the issuer proxy fence and before gate publication.  It does not broaden
// the fence to non-issuer lanes and adds no pre-issue synchronization.
struct config_fp4pv_cluster2_kmc_scale_producer_rendezvous_ac
    : public config_fp4pv_cluster2_kmc_scale_producer_rendezvous_ab {
    static constexpr int ONLINE_ROWPAR2_K_SCALE_PRODUCER_RENDEZVOUS_MODE = 2;
};

// Candidate AD applies AB's successful producer ordering to the independent
// payload-only Z branch.  After the elected rank-0 lane issues the shared K
// payload TMA, the exact one producer warp rendezvous before that same issuer
// executes Z's proxy fence and publishes payload-load-issued.  K scale and the
// unmatched rank-1 tail remain rank local.
struct config_fp4pv_cluster2_kmc_payload_producer_rendezvous_ad
    : public config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_producer_proxy_z {
    static constexpr int ONLINE_ROWPAR2_K_PAYLOAD_PRODUCER_RENDEZVOUS_MODE = 1;
};

// Candidate AE derives AD and adds only the paired post-proxy/pre-publication
// producer-warp rendezvous.  It does not change bytes, barriers, the opposite
// scale domain, or the strict CTA-group-1 tail.
struct config_fp4pv_cluster2_kmc_payload_producer_rendezvous_ae
    : public config_fp4pv_cluster2_kmc_payload_producer_rendezvous_ad {
    static constexpr int ONLINE_ROWPAR2_K_PAYLOAD_PRODUCER_RENDEZVOUS_MODE = 2;
};

// Candidate AF closes the final per-request producer-rendezvous edge present
// in the retained V-multicast sequence. It derives AE and adds only one
// uniform producer-warp rendezvous before the elected rank-0 lane arms and
// issues the shared K payload. The existing post-issue/proxy rendezvous pair,
// rank-local K scale, and strict rank-1 tail remain unchanged.
struct config_fp4pv_cluster2_kmc_payload_preissue_rendezvous_af
    : public config_fp4pv_cluster2_kmc_payload_producer_rendezvous_ae {
    static constexpr bool
        ONLINE_ROWPAR2_K_PAYLOAD_PRE_ISSUE_RENDEZVOUS = true;
};

// Candidate AG removes only AF's auxiliary payload-request publication
// handshake. Both consumers instead wait the already-owner-armed natural
// k_arrived TMA barrier directly, then execute the unchanged consumer proxy
// acquire. AF's three producer rendezvous, producer proxy, split local/remote
// reuse, rank-local K scale, and strict CTA-group-1 tail remain unchanged.
struct config_fp4pv_cluster2_kmc_payload_direct_tma_ready_ag
    : public config_fp4pv_cluster2_kmc_payload_preissue_rendezvous_af {
    static constexpr bool ONLINE_ROWPAR2_K_PAYLOAD_DIRECT_TMA_READY = true;
};

// Candidate AH repairs the producer half of the documented cross-thread
// TCGEN handoff for the direct-rescale output tile.  The output warp drains
// its TMEM stores, executes fence::before_thread_sync, and only then publishes
// rescale_finished to the PV issue warp.  K payload/scale multicast and every
// other lifecycle edge remain the original full-K K-only control.
struct config_fp4pv_cluster2_kmc_output_rescale_producer_fence_ah
    : public config_fp4pv_cluster2_kmc_fullk {
    static constexpr bool
        ONLINE_ROWPAR2_OUTPUT_RESCALE_PRODUCER_FENCE = true;
};

// Candidate AI completes AH's TCGEN handoff.  After observing
// rescale_finished, the PV issue warp executes fence::after_thread_sync before
// staging scales and accumulating the next PV into the same TMEM output tile.
struct config_fp4pv_cluster2_kmc_output_rescale_paired_fence_ai
    : public config_fp4pv_cluster2_kmc_output_rescale_producer_fence_ah {
    static constexpr bool
        ONLINE_ROWPAR2_OUTPUT_RESCALE_CONSUMER_FENCE = true;
};

#endif

// Exact K-only full-K diagnostic clone.  This retains the K-only protocol and
// enables only the existing bounded wait diagnostics; it is never selected by
// production auto dispatch.
struct config_fp4pv_cluster2_kmc_fullk_diag
    : public config_fp4pv_cluster2_kmc_fullk {
    static constexpr bool ONLINE_ROWPAR2_K_MULTICAST_DIAG = true;
};

#if TK_FA4_KMC_PASSIVE_PROBE_BUILD
// Exact K-only full-K clone with sparse write-only lifecycle snapshots.  The
// trait adds observations only; it does not replace any production wait.
struct config_fp4pv_cluster2_kmc_fullk_passive
    : public config_fp4pv_cluster2_kmc_fullk {
    static constexpr bool ONLINE_ROWPAR2_K_MULTICAST_PASSIVE = true;
};
#endif

struct config_fp4pv_cluster2_kvmc_fullk
    : public config_fp4pv_cluster2_kmc_fullk {
    static constexpr bool ONLINE_ROWPAR2_V_MULTICAST = true;
};

// Corrected K+V multicast candidate.  Unlike the known-invalid legacy K+V
// probe, both CTA ranks first receive and wait on a local QK source-completion
// epoch, then publish that completion into rank 0's shared K reuse barrier.
// This keeps K and V local completion ownership symmetric while leaving P
// strictly rank-local.
struct config_fp4pv_cluster2_kvmc_symmetric_fullk
    : public config_fp4pv_cluster2_kvmc_fullk {
    static constexpr bool ONLINE_ROWPAR2_SYMMETRIC_LOCAL_K_COMPLETION = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
    static constexpr bool ONLINE_TMA_V_PAYLOAD = true;
    static constexpr bool ONLINE_TMA_V_SCALE = true;
    static constexpr bool ONLINE_PAIR_P_STAGE_P_SCALE_LIFETIME = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_idxwait_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_ISSUED_IDX_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_psrcdone_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_TCGEN_P_STAGE_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pdelay1_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_DELAY_PAYLOAD_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_alldelay1_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_DELAY_ALL_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_forcespare_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = false;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasknorm_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_TASK_PHASE_NORMALIZE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasktrans_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_TASK_TRANSITION_NORMALIZE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasktail_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_TASK_TAIL_NORMALIZE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_SPLIT_FINAL_STATS = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvscown_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_PV_SCALE_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvruse_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvscown_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_PV_PAYLOAD_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvarruse_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvruse_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_PV_PAYLOAD_ARRIVE_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_corr3_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_CORR_SLOTS3 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pubsync_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_POST_PUBLISH_SYNC = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pvscown_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_PV_SCALE_REUSE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_descwait_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_DESC_IDX_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_descdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_DESC_IDX_DIAG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
    static constexpr bool ONLINE_TMA_V_PAYLOAD = true;
    static constexpr bool ONLINE_TMA_V_SCALE = true;
    static constexpr bool ONLINE_PAIR_P_STAGE_P_SCALE_LIFETIME = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 1>
struct config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_pairpsc_throttle2_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PAIR_P_STAGE_THROTTLE2_DEBUG = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_earlycorr_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_CORR_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_warprescale_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_WARP_RESCALE_ARRIVE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_fixed_pscale_fp32pack_dualaccum
    : public config_fp4pv_3wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_V_LOAD_WARPS = 2;
    static constexpr bool ONLINE_FIXED_P_SCALE_TMEM = true;
    static constexpr bool ONLINE_FP32_P_PACK = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_fixed_pscale_fp32pack_dualaccum_force_persistent
    : public config_fp4pv_3wg_fixed_pscale_fp32pack_dualaccum<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_legacy_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_TMEM = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int ONLINE_QUANT_REGS = _QuantRegs;
    static constexpr int ONLINE_OUTPUT_REGS = _OutputRegs;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_streamscore_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_STREAM_SCORE_RELOAD = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_streamscore_half_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_STREAM_SCORE_RELOAD_HALF = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr int ONLINE_PRODUCER_REGS = _ProducerRegs;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _PayloadStoreMode, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int ONLINE_P_PAYLOAD_STORE_MODE = _PayloadStoreMode;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _PayloadStoreMode, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_lightpublish_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _PayloadStoreMode, _ClusterSize> {
    static constexpr bool ONLINE_LIGHT_P_PAYLOAD_LOCAL_PUBLISH = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pready_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_P_READY_BEFORE_RESCALE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pfirst_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_P_BEFORE_V = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_qk2_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_QK_TWO_AHEAD = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_deferremote_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DEFER_P_REMOTE_UNTIL_PV = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 4;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage4_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage5_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 5;
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage6_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 6;
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_DECOUPLED_PV_ISSUE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DECOUPLED_PV_ISSUE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DECOUPLED_PV_ISSUE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DECOUPLED_PV_ISSUE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pready_stagev_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_P_READY_BEFORE_RESCALE_WAIT = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage4_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PRESCALED_P_PACK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage4_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_splitk64_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PRESCALED_P_PACK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage5_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 5;
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage6_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 6;
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage6_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage6_splitk64_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_PRESCALED_P_PACK = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_pstage3_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_STREAM_SCORE_RELOAD = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_STREAM_SCORE_RELOAD_HALF = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_pstage3_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_pstage3_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_dualaccum_forcespare_streamscore_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_P_READY = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_EARLY_P_READY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_dualaccum_forcespare_streamscore_half_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_EARLY_P_READY = true;
    static constexpr bool ONLINE_DUAL_OUTPUT_ACCUM_EARLY_P_READY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_noearly_dualaccum_forcespare_streamscore_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_STREAM_SCORE_RELOAD_NO_EARLY = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_noearly_dualaccum_forcespare_streamscore_half_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_STREAM_SCORE_RELOAD_NO_EARLY = true;
};

template <typename C, typename = void>
struct fp4pv_online_async_p_scale_copy {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_async_p_scale_copy<C, std::void_t<decltype(C::ONLINE_ASYNC_P_SCALE_COPY)>> {
    static constexpr bool value = C::ONLINE_ASYNC_P_SCALE_COPY;
};

template <typename C, typename = void>
struct fp4pv_online_dual_score_tmem {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dual_score_tmem<C, std::void_t<decltype(C::ONLINE_DUAL_SCORE_TMEM)>> {
    static constexpr bool value = C::ONLINE_DUAL_SCORE_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_direct_p_scale_tmem {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_direct_p_scale_tmem<C, std::void_t<decltype(C::ONLINE_DIRECT_P_SCALE_TMEM)>> {
    static constexpr bool value = C::ONLINE_DIRECT_P_SCALE_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_fixed_p_scale_tmem {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_fixed_p_scale_tmem<C, std::void_t<decltype(C::ONLINE_FIXED_P_SCALE_TMEM)>> {
    static constexpr bool value = C::ONLINE_FIXED_P_SCALE_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_p_payload_store_mode {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_online_p_payload_store_mode<C, std::void_t<decltype(C::ONLINE_P_PAYLOAD_STORE_MODE)>> {
    static constexpr int value = C::ONLINE_P_PAYLOAD_STORE_MODE;
};

template <typename C, typename = void>
struct fp4pv_online_light_p_payload_local_publish {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_light_p_payload_local_publish<C, std::void_t<decltype(C::ONLINE_LIGHT_P_PAYLOAD_LOCAL_PUBLISH)>> {
    static constexpr bool value = C::ONLINE_LIGHT_P_PAYLOAD_LOCAL_PUBLISH;
};

template <typename C, typename = void>
struct fp4pv_online_split_output_accum {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_split_output_accum<C, std::void_t<decltype(C::ONLINE_SPLIT_OUTPUT_ACCUM)>> {
    static constexpr bool value = C::ONLINE_SPLIT_OUTPUT_ACCUM;
};

template <typename C, typename = void>
struct fp4pv_online_fp32_smem_output_accum {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_fp32_smem_output_accum<
    C, std::void_t<decltype(C::ONLINE_FP32_SMEM_OUTPUT_ACCUM)>> {
    static constexpr bool value = C::ONLINE_FP32_SMEM_OUTPUT_ACCUM;
};

template <typename C, typename = void>
struct fp4pv_online_register_smem_hybrid {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_register_smem_hybrid<
    C, std::void_t<decltype(C::ONLINE_REGISTER_SMEM_HYBRID)>> {
    static constexpr bool value = C::ONLINE_REGISTER_SMEM_HYBRID;
};

template <typename C, typename = void>
struct fp4pv_online_early_p_ready {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_early_p_ready<C, std::void_t<decltype(C::ONLINE_EARLY_P_READY)>> {
    static constexpr bool value = C::ONLINE_EARLY_P_READY;
};

template <typename C, typename = void>
struct fp4pv_online_stage_v_scale_before_next_qk {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_stage_v_scale_before_next_qk<C, std::void_t<decltype(C::ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK)>> {
    static constexpr bool value = C::ONLINE_STAGE_V_SCALE_BEFORE_NEXT_QK;
};

template <typename C, typename = void>
struct fp4pv_online_stage_p_before_v {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_stage_p_before_v<C, std::void_t<decltype(C::ONLINE_STAGE_P_BEFORE_V)>> {
    static constexpr bool value = C::ONLINE_STAGE_P_BEFORE_V;
};

template <typename C, typename = void>
struct fp4pv_online_qk_two_ahead {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_qk_two_ahead<C, std::void_t<decltype(C::ONLINE_QK_TWO_AHEAD)>> {
    static constexpr bool value = C::ONLINE_QK_TWO_AHEAD;
};

template <typename C, typename = void>
struct fp4pv_online_prepublish_p_payload_before_p_scale_wait {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_prepublish_p_payload_before_p_scale_wait<C, std::void_t<decltype(C::ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT)>> {
    static constexpr bool value = C::ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_WAIT;
};

template <typename C, typename = void>
struct fp4pv_online_prepublish_p_payload_before_p_scale_reuse_wait {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_prepublish_p_payload_before_p_scale_reuse_wait<C, std::void_t<decltype(C::ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT)>> {
    static constexpr bool value = C::ONLINE_PREPUBLISH_P_PAYLOAD_BEFORE_P_SCALE_REUSE_WAIT;
};

template <typename C, typename = void>
struct fp4pv_online_tcgen_p_stage_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_tcgen_p_stage_reuse<C, std::void_t<decltype(C::ONLINE_TCGEN_P_STAGE_REUSE)>> {
    static constexpr bool value = C::ONLINE_TCGEN_P_STAGE_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_arrive_p_stage_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_arrive_p_stage_reuse<C, std::void_t<decltype(C::ONLINE_ARRIVE_P_STAGE_REUSE)>> {
    static constexpr bool value = C::ONLINE_ARRIVE_P_STAGE_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_ranklocal {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_ranklocal<C, std::void_t<decltype(C::ONLINE_ROWPAR2_RANKLOCAL)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_RANKLOCAL;
};

// Exact opt-in for the production-derived native cta_group::2 FP4 route.
// Keep this separate from ROWPAR2_RANKLOCAL: the latter deliberately lowers
// QK/PV to cta_group::1, whereas this route retains collective group2
// allocation, issue, completion, and deallocation.
template <typename C, typename = void>
struct fp4pv_online_native_group2_lean_handoff {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_lean_handoff<
    C, std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_LEAN_HANDOFF)>> {
    static constexpr bool value = C::ONLINE_NATIVE_GROUP2_LEAN_HANDOFF;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_local_group1_pv {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_local_group1_pv<
    C, std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_LOCAL_GROUP1_PV)>> {
    static constexpr bool value = C::ONLINE_NATIVE_GROUP2_LOCAL_GROUP1_PV;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_score_pair_credit {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_score_pair_credit<
    C, std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT)>> {
    static constexpr bool value = C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_score_pair_credit_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_score_pair_credit_diag<
    C, std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT_DIAG)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_CREDIT_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_downstream_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_downstream_diag<
    C, std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_DOWNSTREAM_DIAG)>> {
    static constexpr bool value = C::ONLINE_NATIVE_GROUP2_DOWNSTREAM_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_split_final_stats {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_split_final_stats<
    C, std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_SPLIT_FINAL_STATS)>> {
    static constexpr bool value = C::ONLINE_NATIVE_GROUP2_SPLIT_FINAL_STATS;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_exact_lse_sum {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_exact_lse_sum<
    C, std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_EXACT_LSE_SUM)>> {
    static constexpr bool value = C::ONLINE_NATIVE_GROUP2_EXACT_LSE_SUM;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_exact_lse_threshold {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_exact_lse_threshold<
    C, std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_EXACT_LSE_THRESHOLD)>> {
    static constexpr bool value = C::ONLINE_NATIVE_GROUP2_EXACT_LSE_THRESHOLD;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_score_load_thread_fence {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_score_load_thread_fence<
    C,
    std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_SCORE_LOAD_THREAD_FENCE)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_SCORE_LOAD_THREAD_FENCE;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_score_pair_reader_warp_credit {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_score_pair_reader_warp_credit<
    C,
    std::void_t<
        decltype(C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_READER_WARP_CREDIT)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_READER_WARP_CREDIT;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_score_pair_wg_rendezvous {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_score_pair_wg_rendezvous<
    C,
    std::void_t<
        decltype(C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_WG_RENDEZVOUS)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_WG_RENDEZVOUS;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_score_pair_issue_proxy {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_score_pair_issue_proxy<
    C,
    std::void_t<
        decltype(C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_ISSUE_PROXY)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_ISSUE_PROXY;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_score_pair_quant_reader_proxy {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_score_pair_quant_reader_proxy<
    C,
    std::void_t<
        decltype(C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_QUANT_READER_PROXY)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_SCORE_PAIR_QUANT_READER_PROXY;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_q_scale_three_page {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_q_scale_three_page<
    C,
    std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_Q_SCALE_THREE_PAGE)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_Q_SCALE_THREE_PAGE;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_k_scale_three_page {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_k_scale_three_page<
    C,
    std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_K_SCALE_THREE_PAGE)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_K_SCALE_THREE_PAGE;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_physical_k_tile_limit {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_physical_k_tile_limit<
    C,
    std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_PHYSICAL_K_TILE_LIMIT)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_PHYSICAL_K_TILE_LIMIT;
};

template <typename C, typename = void>
struct fp4pv_online_native_group2_selected_row_trace {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_native_group2_selected_row_trace<
    C,
    std::void_t<decltype(C::ONLINE_NATIVE_GROUP2_SELECTED_ROW_TRACE)>> {
    static constexpr bool value =
        C::ONLINE_NATIVE_GROUP2_SELECTED_ROW_TRACE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_v_multicast {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_v_multicast<C, std::void_t<decltype(C::ONLINE_ROWPAR2_V_MULTICAST)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_V_MULTICAST;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_v_payload_multicast {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_v_payload_multicast<C, std::void_t<decltype(C::ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_v_payload_multicast_short_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_v_payload_multicast_short_diag<C, std::void_t<decltype(C::ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_SHORT_DIAG)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_SHORT_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_v_payload_multicast_scale_diag {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_rowpar2_v_payload_multicast_scale_diag<C, std::void_t<decltype(C::ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_SCALE_DIAG)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_SCALE_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_v_payload_multicast_2slot_diag {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_rowpar2_v_payload_multicast_2slot_diag<C, std::void_t<decltype(C::ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_2SLOT_DIAG)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_V_PAYLOAD_MULTICAST_2SLOT_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_v_multicast_cluster_publish {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_rowpar2_v_multicast_cluster_publish<C, std::void_t<decltype(C::ONLINE_ROWPAR2_V_MULTICAST_CLUSTER_PUBLISH)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_V_MULTICAST_CLUSTER_PUBLISH;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_v_multicast_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_v_multicast_diag<C, std::void_t<decltype(C::ONLINE_ROWPAR2_V_MULTICAST_DIAG)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_V_MULTICAST_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_shared_k_lifetime {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_shared_k_lifetime<C, std::void_t<decltype(C::ONLINE_ROWPAR2_SHARED_K_LIFETIME)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_SHARED_K_LIFETIME;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_local_k_payload_split {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_local_k_payload_split<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_LOCAL_K_PAYLOAD_SPLIT)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_LOCAL_K_PAYLOAD_SPLIT;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_local_k_scale_split {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_local_k_scale_split<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_LOCAL_K_SCALE_SPLIT)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_LOCAL_K_SCALE_SPLIT;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_payload_issue_gate {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_payload_issue_gate<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_K_PAYLOAD_ISSUE_GATE)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_PAYLOAD_ISSUE_GATE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_scale_issue_gate {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_scale_issue_gate<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_K_SCALE_ISSUE_GATE)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_SCALE_ISSUE_GATE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_payload_arm_after_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_payload_arm_after_reuse<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_K_PAYLOAD_ARM_AFTER_REUSE)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_PAYLOAD_ARM_AFTER_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_scale_arm_after_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_scale_arm_after_reuse<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_K_SCALE_ARM_AFTER_REUSE)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_SCALE_ARM_AFTER_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_split_local_remote_k_payload_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_split_local_remote_k_payload_reuse<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_SPLIT_LOCAL_REMOTE_K_PAYLOAD_REUSE)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_SPLIT_LOCAL_REMOTE_K_PAYLOAD_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_split_local_remote_k_scale_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_split_local_remote_k_scale_reuse<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_SPLIT_LOCAL_REMOTE_K_SCALE_REUSE)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_SPLIT_LOCAL_REMOTE_K_SCALE_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_split_reuse_issue_gate_composition {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_split_reuse_issue_gate_composition<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_SPLIT_REUSE_ISSUE_GATE_COMPOSITION)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_SPLIT_REUSE_ISSUE_GATE_COMPOSITION;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_split_reuse_issue_gate_proxy_composition {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_split_reuse_issue_gate_proxy_composition<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_SPLIT_REUSE_ISSUE_GATE_PROXY_COMPOSITION)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_SPLIT_REUSE_ISSUE_GATE_PROXY_COMPOSITION;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_split_reuse_producer_consumer_proxy_composition {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_split_reuse_producer_consumer_proxy_composition<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_SPLIT_REUSE_PRODUCER_CONSUMER_PROXY_COMPOSITION)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_SPLIT_REUSE_PRODUCER_CONSUMER_PROXY_COMPOSITION;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_symmetric_local_k_completion {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_symmetric_local_k_completion<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_SYMMETRIC_LOCAL_K_COMPLETION)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_SYMMETRIC_LOCAL_K_COMPLETION;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_tail_drain_mode {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_online_rowpar2_k_tail_drain_mode<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_K_TAIL_DRAIN_MODE)>> {
    static constexpr int value = C::ONLINE_ROWPAR2_K_TAIL_DRAIN_MODE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_common_prefix_local_k_tail {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_common_prefix_local_k_tail<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_COMMON_PREFIX_LOCAL_K_TAIL)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_COMMON_PREFIX_LOCAL_K_TAIL;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_strict_local_k_tail {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_strict_local_k_tail<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_STRICT_LOCAL_K_TAIL)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_STRICT_LOCAL_K_TAIL;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_defer_k_sc_shared_until_local_qk_completion {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_defer_k_sc_shared_until_local_qk_completion<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_DEFER_K_SC_SHARED_UNTIL_LOCAL_QK_COMPLETION)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_DEFER_K_SC_SHARED_UNTIL_LOCAL_QK_COMPLETION;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_combined_symmetric_deferred_k_completion {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_combined_symmetric_deferred_k_completion<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_COMBINED_SYMMETRIC_DEFERRED_K_COMPLETION)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_COMBINED_SYMMETRIC_DEFERRED_K_COMPLETION;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_owner_arms_remote_k_payload_mbar {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_owner_arms_remote_k_payload_mbar<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_PAYLOAD_MBAR)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_PAYLOAD_MBAR;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_owner_arms_remote_k_scale_mbar {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_owner_arms_remote_k_scale_mbar<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_SCALE_MBAR)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_SCALE_MBAR;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_post_k_payload_proxy_acquire {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_post_k_payload_proxy_acquire<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_POST_K_PAYLOAD_PROXY_ACQUIRE)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_POST_K_PAYLOAD_PROXY_ACQUIRE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_post_k_scale_proxy_acquire {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_post_k_scale_proxy_acquire<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_POST_K_SCALE_PROXY_ACQUIRE)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_POST_K_SCALE_PROXY_ACQUIRE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_payload_post_tma_proxy_publish {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_payload_post_tma_proxy_publish<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_K_PAYLOAD_POST_TMA_PROXY_PUBLISH)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_K_PAYLOAD_POST_TMA_PROXY_PUBLISH;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_scale_post_tma_proxy_publish {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_scale_post_tma_proxy_publish<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_K_SCALE_POST_TMA_PROXY_PUBLISH)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_K_SCALE_POST_TMA_PROXY_PUBLISH;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_scale_producer_rendezvous_mode {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_online_rowpar2_k_scale_producer_rendezvous_mode<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_K_SCALE_PRODUCER_RENDEZVOUS_MODE)>> {
    static constexpr int value =
        C::ONLINE_ROWPAR2_K_SCALE_PRODUCER_RENDEZVOUS_MODE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_payload_producer_rendezvous_mode {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_online_rowpar2_k_payload_producer_rendezvous_mode<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_K_PAYLOAD_PRODUCER_RENDEZVOUS_MODE)>> {
    static constexpr int value =
        C::ONLINE_ROWPAR2_K_PAYLOAD_PRODUCER_RENDEZVOUS_MODE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_payload_pre_issue_rendezvous {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_payload_pre_issue_rendezvous<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_K_PAYLOAD_PRE_ISSUE_RENDEZVOUS)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_K_PAYLOAD_PRE_ISSUE_RENDEZVOUS;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_payload_direct_tma_ready {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_payload_direct_tma_ready<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_K_PAYLOAD_DIRECT_TMA_READY)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_K_PAYLOAD_DIRECT_TMA_READY;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_output_rescale_producer_fence {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_output_rescale_producer_fence<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_OUTPUT_RESCALE_PRODUCER_FENCE)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_OUTPUT_RESCALE_PRODUCER_FENCE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_output_rescale_consumer_fence {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_output_rescale_consumer_fence<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_OUTPUT_RESCALE_CONSUMER_FENCE)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_OUTPUT_RESCALE_CONSUMER_FENCE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_cluster_scope_k_scale_handoff {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_cluster_scope_k_scale_handoff<
    C,
    std::void_t<decltype(
        C::ONLINE_ROWPAR2_CLUSTER_SCOPE_K_SCALE_HANDOFF)>> {
    static constexpr bool value =
        C::ONLINE_ROWPAR2_CLUSTER_SCOPE_K_SCALE_HANDOFF;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_cluster_scope_k_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_cluster_scope_k_reuse<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_CLUSTER_SCOPE_K_REUSE)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_CLUSTER_SCOPE_K_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_multicast_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_multicast_diag<C, std::void_t<decltype(C::ONLINE_ROWPAR2_K_MULTICAST_DIAG)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_MULTICAST_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_multicast_passive {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_multicast_passive<
    C,
    std::void_t<decltype(C::ONLINE_ROWPAR2_K_MULTICAST_PASSIVE)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_MULTICAST_PASSIVE;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_multicast_short_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_multicast_short_diag<C, std::void_t<decltype(C::ONLINE_ROWPAR2_K_MULTICAST_SHORT_DIAG)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_MULTICAST_SHORT_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_multicast_ring_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_multicast_ring_diag<C, std::void_t<decltype(C::ONLINE_ROWPAR2_K_MULTICAST_RING_DIAG)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_MULTICAST_RING_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_rowpar2_k_multicast_qk_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_rowpar2_k_multicast_qk_diag<C, std::void_t<decltype(C::ONLINE_ROWPAR2_K_MULTICAST_QK_DIAG)>> {
    static constexpr bool value = C::ONLINE_ROWPAR2_K_MULTICAST_QK_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_fold_p_scale_reuse_with_p_stage {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_fold_p_scale_reuse_with_p_stage<C, std::void_t<decltype(C::ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE)>> {
    static constexpr bool value = C::ONLINE_FOLD_P_SCALE_REUSE_WITH_P_STAGE;
};

template <typename C, typename = void>
struct fp4pv_online_skip_folded_p_scale_reuse_arrive {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_skip_folded_p_scale_reuse_arrive<C, std::void_t<decltype(C::ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE)>> {
    static constexpr bool value = C::ONLINE_SKIP_FOLDED_P_SCALE_REUSE_ARRIVE;
};

template <typename C, typename = void>
struct fp4pv_online_p_scale_tmem_slots {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_online_p_scale_tmem_slots<C, std::void_t<decltype(C::ONLINE_P_SCALE_TMEM_SLOTS)>> {
    static constexpr int value = C::ONLINE_P_SCALE_TMEM_SLOTS;
};

template <typename C, typename = void>
struct fp4pv_online_p_scale_tmem_credit_limit {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_online_p_scale_tmem_credit_limit<C, std::void_t<decltype(C::ONLINE_P_SCALE_TMEM_CREDIT_LIMIT)>> {
    static constexpr int value = C::ONLINE_P_SCALE_TMEM_CREDIT_LIMIT;
};

template <typename C, typename = void>
struct fp4pv_online_p_scale_tmem_tcgen_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_p_scale_tmem_tcgen_reuse<C, std::void_t<decltype(C::ONLINE_P_SCALE_TMEM_TCGEN_REUSE)>> {
    static constexpr bool value = C::ONLINE_P_SCALE_TMEM_TCGEN_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_v_scale_tmem_tcgen_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_v_scale_tmem_tcgen_reuse<C, std::void_t<decltype(C::ONLINE_V_SCALE_TMEM_TCGEN_REUSE)>> {
    static constexpr bool value = C::ONLINE_V_SCALE_TMEM_TCGEN_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_exact_scale_page_mode {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_online_exact_scale_page_mode<C, std::void_t<decltype(C::ONLINE_EXACT_SCALE_PAGE_MODE)>> {
    static constexpr int value = C::ONLINE_EXACT_SCALE_PAGE_MODE;
};

template <typename C, typename = void>
struct fp4pv_online_single_pv_scale_tmem {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_single_pv_scale_tmem<C, std::void_t<decltype(C::ONLINE_SINGLE_PV_SCALE_TMEM)>> {
    static constexpr bool value = C::ONLINE_SINGLE_PV_SCALE_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_fp16_cold_output_checkpoint {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_fp16_cold_output_checkpoint<C, std::void_t<decltype(C::ONLINE_FP16_COLD_OUTPUT_CHECKPOINT)>> {
    static constexpr bool value = C::ONLINE_FP16_COLD_OUTPUT_CHECKPOINT;
};

template <typename C, typename = void>
struct fp4pv_online_dvo64_two_pass {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dvo64_two_pass<C, std::void_t<decltype(C::ONLINE_DVO64_TWO_PASS)>> {
    static constexpr bool value = C::ONLINE_DVO64_TWO_PASS;
};

template <typename C, typename = void>
struct fp4pv_online_disable_reg_tune {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_disable_reg_tune<C, std::void_t<decltype(C::ONLINE_DISABLE_REG_TUNE)>> {
    static constexpr bool value = C::ONLINE_DISABLE_REG_TUNE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_p_scale_lifetime {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_p_scale_lifetime<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_P_SCALE_LIFETIME)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_P_SCALE_LIFETIME;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_throttle2_debug {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_throttle2_debug<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_THROTTLE2_DEBUG)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_THROTTLE2_DEBUG;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_issued_idx_wait {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_issued_idx_wait<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_ISSUED_IDX_WAIT)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_ISSUED_IDX_WAIT;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_delay_payload_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_delay_payload_reuse<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_DELAY_PAYLOAD_REUSE)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_DELAY_PAYLOAD_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_delay_all_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_delay_all_reuse<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_DELAY_ALL_REUSE)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_DELAY_ALL_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_task_phase_normalize {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_task_phase_normalize<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_TASK_PHASE_NORMALIZE)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_TASK_PHASE_NORMALIZE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_task_transition_normalize {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_task_transition_normalize<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_TASK_TRANSITION_NORMALIZE)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_TASK_TRANSITION_NORMALIZE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_task_tail_normalize {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_task_tail_normalize<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_TASK_TAIL_NORMALIZE)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_TASK_TAIL_NORMALIZE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_split_final_stats {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_split_final_stats<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_SPLIT_FINAL_STATS)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_SPLIT_FINAL_STATS;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_corr_slots3 {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_corr_slots3<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_CORR_SLOTS3)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_CORR_SLOTS3;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_post_publish_sync {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_post_publish_sync<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_POST_PUBLISH_SYNC)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_POST_PUBLISH_SYNC;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_pv_scale_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_pv_scale_reuse<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_PV_SCALE_REUSE)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_PV_SCALE_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_pv_payload_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_pv_payload_reuse<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_PV_PAYLOAD_REUSE)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_PV_PAYLOAD_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_pv_payload_arrive_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_pv_payload_arrive_reuse<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_PV_PAYLOAD_ARRIVE_REUSE)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_PV_PAYLOAD_ARRIVE_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_desc_idx_wait {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_desc_idx_wait<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_DESC_IDX_WAIT)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_DESC_IDX_WAIT;
};

template <typename C, typename = void>
struct fp4pv_online_pair_p_stage_desc_idx_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_pair_p_stage_desc_idx_diag<C, std::void_t<decltype(C::ONLINE_PAIR_P_STAGE_DESC_IDX_DIAG)>> {
    static constexpr bool value = C::ONLINE_PAIR_P_STAGE_DESC_IDX_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_defer_p_remote_until_pv {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_defer_p_remote_until_pv<C, std::void_t<decltype(C::ONLINE_DEFER_P_REMOTE_UNTIL_PV)>> {
    static constexpr bool value = C::ONLINE_DEFER_P_REMOTE_UNTIL_PV;
};

template <typename C, typename = void>
struct fp4pv_online_producer_async_v_scale_tmem {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_producer_async_v_scale_tmem<C, std::void_t<decltype(C::ONLINE_PRODUCER_ASYNC_V_SCALE_TMEM)>> {
    static constexpr bool value = C::ONLINE_PRODUCER_ASYNC_V_SCALE_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_early_corr_publish {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_early_corr_publish<C, std::void_t<decltype(C::ONLINE_EARLY_CORR_PUBLISH)>> {
    static constexpr bool value = C::ONLINE_EARLY_CORR_PUBLISH;
};

template <typename C, typename = void>
struct fp4pv_online_fp32_p_pack {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_fp32_p_pack<C, std::void_t<decltype(C::ONLINE_FP32_P_PACK)>> {
    static constexpr bool value = C::ONLINE_FP32_P_PACK;
};

template <typename C, typename = void>
struct fp4pv_online_prescaled_p_pack {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_prescaled_p_pack<C, std::void_t<decltype(C::ONLINE_PRESCALED_P_PACK)>> {
    static constexpr bool value = C::ONLINE_PRESCALED_P_PACK;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_p_scale_pack {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_score_derived_p_scale_pack<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_P_SCALE_PACK)>> {
    static constexpr bool value = C::ONLINE_SCORE_DERIVED_P_SCALE_PACK;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_p_prescaled_pack {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_score_derived_p_prescaled_pack<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_P_PRESCALED_PACK)>> {
    static constexpr bool value = C::ONLINE_SCORE_DERIVED_P_PRESCALED_PACK;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_p_pack_scales_late {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_score_derived_p_pack_scales_late<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_P_PACK_SCALES_LATE)>> {
    static constexpr bool value = C::ONLINE_SCORE_DERIVED_P_PACK_SCALES_LATE;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_p_scale_mode {
    static constexpr int value = -1;
};
template <typename C>
struct fp4pv_online_score_derived_p_scale_mode<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_P_SCALE_MODE)>> {
    static constexpr int value = C::ONLINE_SCORE_DERIVED_P_SCALE_MODE;
};

template <typename C, typename = void>
struct fp4pv_qk_scale_coord_head_index {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_qk_scale_coord_head_index<C, std::void_t<decltype(C::QK_SCALE_COORD_HEAD_INDEX)>> {
    static constexpr bool value = C::QK_SCALE_COORD_HEAD_INDEX;
};

template <typename C, typename = void>
struct fp4pv_online_qk_pair_nb64 {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_qk_pair_nb64<C, std::void_t<decltype(C::ONLINE_QK_PAIR_NB64)>> {
    static constexpr bool value = C::ONLINE_QK_PAIR_NB64;
};

template <typename C, typename = void>
struct fp4pv_online_full_k_tile {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_full_k_tile<C, std::void_t<decltype(C::ONLINE_FULL_K_TILE)>> {
    static constexpr bool value = C::ONLINE_FULL_K_TILE;
};

template <typename C, typename = void>
struct fp4pv_online_deep_joint_ring {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_deep_joint_ring<
    C, std::void_t<decltype(C::ONLINE_DEEP_JOINT_RING)>> {
    static constexpr bool value = C::ONLINE_DEEP_JOINT_RING;
};

template <typename C, typename = void>
struct fp4pv_online_diagonal_causal_payload_zero_only {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_diagonal_causal_payload_zero_only<C, std::void_t<decltype(C::ONLINE_DIAGONAL_CAUSAL_PAYLOAD_ZERO_ONLY)>> {
    static constexpr bool value = C::ONLINE_DIAGONAL_CAUSAL_PAYLOAD_ZERO_ONLY;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_p_x1_scale_tmem {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_score_derived_p_x1_scale_tmem<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_P_X1_SCALE_TMEM)>> {
    static constexpr bool value = C::ONLINE_SCORE_DERIVED_P_X1_SCALE_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_p_fused_block_max {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_score_derived_p_fused_block_max<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_P_FUSED_BLOCK_MAX)>> {
    static constexpr bool value = C::ONLINE_SCORE_DERIVED_P_FUSED_BLOCK_MAX;
};

template <typename C, typename = void>
struct fp4pv_online_stage2_early_async_p_scale_store {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_stage2_early_async_p_scale_store<C, std::void_t<decltype(C::ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE)>> {
    static constexpr bool value = C::ONLINE_STAGE2_EARLY_ASYNC_P_SCALE_STORE;
};

template <typename C, typename = void>
struct fp4pv_online_stage2_defer_row_sum_after_p_ready {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_stage2_defer_row_sum_after_p_ready<C, std::void_t<decltype(C::ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY)>> {
    static constexpr bool value = C::ONLINE_STAGE2_DEFER_ROW_SUM_AFTER_P_READY;
};

template <typename C, typename = void>
struct fp4pv_online_stage2_ex2_emu_period {
    static constexpr int value = 0;
};
template <typename C>
struct fp4pv_online_stage2_ex2_emu_period<C, std::void_t<decltype(C::ONLINE_STAGE2_EX2_EMU_PERIOD)>> {
    static constexpr int value = C::ONLINE_STAGE2_EX2_EMU_PERIOD;
};

template <typename C, typename = void>
struct fp4pv_online_stage2_ex2_emu_mask {
    static constexpr uint32_t value = 0;
};
template <typename C>
struct fp4pv_online_stage2_ex2_emu_mask<C, std::void_t<decltype(C::ONLINE_STAGE2_EX2_EMU_MASK)>> {
    static constexpr uint32_t value = C::ONLINE_STAGE2_EX2_EMU_MASK;
};

template <typename C, typename = void>
struct fp4pv_online_q4_head_interleaved_mapping {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_q4_head_interleaved_mapping<C, std::void_t<decltype(C::ONLINE_Q4_HEAD_INTERLEAVED_MAPPING)>> {
    static constexpr bool value = C::ONLINE_Q4_HEAD_INTERLEAVED_MAPPING;
};

template <typename C, typename = void>
struct fp4pv_online_q4_head_group_size {
    static constexpr int value = 0;
};
template <typename C>
struct fp4pv_online_q4_head_group_size<C, std::void_t<decltype(C::ONLINE_Q4_HEAD_GROUP_SIZE)>> {
    static constexpr int value = C::ONLINE_Q4_HEAD_GROUP_SIZE;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_k256 {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_score_derived_k256<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_K256)>> {
    static constexpr bool value = C::ONLINE_SCORE_DERIVED_K256;
};

template <typename C, typename = void>
struct fp4pv_online_split_p_ready_k64 {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_split_p_ready_k64<C, std::void_t<decltype(C::ONLINE_SPLIT_P_READY_K64)>> {
    static constexpr bool value = C::ONLINE_SPLIT_P_READY_K64;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_k64_half_ready {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_score_derived_k64_half_ready<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_K64_HALF_READY)>> {
    static constexpr bool value = C::ONLINE_SCORE_DERIVED_K64_HALF_READY;
};

template <typename C, typename = void>
struct fp4pv_online_split_p_quant_2wg {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_split_p_quant_2wg<C, std::void_t<decltype(C::ONLINE_SPLIT_P_QUANT_2WG)>> {
    static constexpr bool value = C::ONLINE_SPLIT_P_QUANT_2WG;
};

template <typename C, typename = void>
struct fp4pv_online_score_derived_split_p_quant_2wg {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_score_derived_split_p_quant_2wg<C, std::void_t<decltype(C::ONLINE_SCORE_DERIVED_SPLIT_P_QUANT_2WG)>> {
    static constexpr bool value = C::ONLINE_SCORE_DERIVED_SPLIT_P_QUANT_2WG;
};

template <typename C, typename = void>
struct fp4pv_online_split_p_quant_row_2wg {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_split_p_quant_row_2wg<C, std::void_t<decltype(C::ONLINE_SPLIT_P_QUANT_ROW_2WG)>> {
    static constexpr bool value = C::ONLINE_SPLIT_P_QUANT_ROW_2WG;
};

template <typename C, typename = void>
struct fp4pv_online_split_p_quant_localmax_2wg {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_split_p_quant_localmax_2wg<C, std::void_t<decltype(C::ONLINE_SPLIT_P_QUANT_LOCALMAX_2WG)>> {
    static constexpr bool value = C::ONLINE_SPLIT_P_QUANT_LOCALMAX_2WG;
};

template <typename C, typename = void>
struct fp4pv_online_split_p_quant_leader_sum_2wg {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_split_p_quant_leader_sum_2wg<C, std::void_t<decltype(C::ONLINE_SPLIT_P_QUANT_LEADER_SUM_2WG)>> {
    static constexpr bool value = C::ONLINE_SPLIT_P_QUANT_LEADER_SUM_2WG;
};

template <typename C, typename = void>
struct fp4pv_online_stream_score_reload {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_stream_score_reload<C, std::void_t<decltype(C::ONLINE_STREAM_SCORE_RELOAD)>> {
    static constexpr bool value = C::ONLINE_STREAM_SCORE_RELOAD;
};

template <typename C, typename = void>
struct fp4pv_online_stream_score_reload_half {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_stream_score_reload_half<C, std::void_t<decltype(C::ONLINE_STREAM_SCORE_RELOAD_HALF)>> {
    static constexpr bool value = C::ONLINE_STREAM_SCORE_RELOAD_HALF;
};

template <typename C, typename = void>
struct fp4pv_online_stream_score_reload_no_early {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_stream_score_reload_no_early<C, std::void_t<decltype(C::ONLINE_STREAM_SCORE_RELOAD_NO_EARLY)>> {
    static constexpr bool value = C::ONLINE_STREAM_SCORE_RELOAD_NO_EARLY;
};

template <typename C, typename = void>
struct fp4pv_online_p_ready_before_rescale_wait {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_p_ready_before_rescale_wait<C, std::void_t<decltype(C::ONLINE_P_READY_BEFORE_RESCALE_WAIT)>> {
    static constexpr bool value = C::ONLINE_P_READY_BEFORE_RESCALE_WAIT;
};

template <typename C, typename = void>
struct fp4pv_online_warp_rescale_arrive {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_warp_rescale_arrive<C, std::void_t<decltype(C::ONLINE_WARP_RESCALE_ARRIVE)>> {
    static constexpr bool value = C::ONLINE_WARP_RESCALE_ARRIVE;
};

template <typename C, typename = void>
struct fp4pv_online_dual_output_accum_tmem {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dual_output_accum_tmem<C, std::void_t<decltype(C::ONLINE_DUAL_OUTPUT_ACCUM_TMEM)>> {
    static constexpr bool value = C::ONLINE_DUAL_OUTPUT_ACCUM_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_dual_output_accum_early_p_ready {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dual_output_accum_early_p_ready<C, std::void_t<decltype(C::ONLINE_DUAL_OUTPUT_ACCUM_EARLY_P_READY)>> {
    static constexpr bool value = C::ONLINE_DUAL_OUTPUT_ACCUM_EARLY_P_READY;
};

template <typename C, typename = void>
struct fp4pv_online_dual_output_accum_skip_flag_sync {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dual_output_accum_skip_flag_sync<C, std::void_t<decltype(C::ONLINE_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC)>> {
    static constexpr bool value = C::ONLINE_DUAL_OUTPUT_ACCUM_SKIP_FLAG_SYNC;
};

template <typename C, typename = void>
struct fp4pv_online_dual_output_accum_force_spare {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dual_output_accum_force_spare<C, std::void_t<decltype(C::ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE)>> {
    static constexpr bool value = C::ONLINE_DUAL_OUTPUT_ACCUM_FORCE_SPARE;
};

template <typename C, typename = void>
struct fp4pv_online_dual_output_accum_dedicated_spare {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dual_output_accum_dedicated_spare<C, std::void_t<decltype(C::ONLINE_DUAL_OUTPUT_ACCUM_DEDICATED_SPARE)>> {
    static constexpr bool value = C::ONLINE_DUAL_OUTPUT_ACCUM_DEDICATED_SPARE;
};

template <typename C, typename = void>
struct fp4pv_online_dual_output_accum_direct_after_rescale {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dual_output_accum_direct_after_rescale<C, std::void_t<decltype(C::ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE)>> {
    static constexpr bool value = C::ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE;
};

template <typename C, typename = void>
struct fp4pv_online_defer_direct_rescale_wait_until_pv {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_defer_direct_rescale_wait_until_pv<C, std::void_t<decltype(C::ONLINE_DEFER_DIRECT_RESCALE_WAIT_UNTIL_PV)>> {
    static constexpr bool value = C::ONLINE_DEFER_DIRECT_RESCALE_WAIT_UNTIL_PV;
};

template <typename C, typename = void>
struct fp4pv_online_corr_pv_lookahead {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_corr_pv_lookahead<
    C, std::void_t<decltype(C::ONLINE_CORR_PV_LOOKAHEAD)>> {
    static constexpr bool value = C::ONLINE_CORR_PV_LOOKAHEAD;
};

template <typename C, typename = void>
struct fp4pv_online_corr_pv_wait_only_lookahead {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_corr_pv_wait_only_lookahead<
    C, std::void_t<decltype(C::ONLINE_CORR_PV_WAIT_ONLY_LOOKAHEAD)>> {
    static constexpr bool value = C::ONLINE_CORR_PV_WAIT_ONLY_LOOKAHEAD;
};

template <typename C, typename = void>
struct fp4pv_online_corr_pv_wait_after_pv {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_corr_pv_wait_after_pv<
    C, std::void_t<decltype(C::ONLINE_CORR_PV_WAIT_AFTER_PV)>> {
    static constexpr bool value = C::ONLINE_CORR_PV_WAIT_AFTER_PV;
};

template <typename C, typename = void>
struct fp4pv_online_corr_precheck { static constexpr bool value = false; };
template <typename C>
struct fp4pv_online_corr_precheck<C, std::void_t<decltype(C::ONLINE_CORR_PRECHECK)>> {
    static constexpr bool value = C::ONLINE_CORR_PRECHECK;
};
template <typename C, typename = void>
struct fp4pv_online_corr_precheck_control { static constexpr bool value = false; };
template <typename C>
struct fp4pv_online_corr_precheck_control<C, std::void_t<decltype(C::ONLINE_CORR_PRECHECK_CONTROL)>> {
    static constexpr bool value = C::ONLINE_CORR_PRECHECK_CONTROL;
};
template <typename C, typename = void>
struct fp4pv_online_corr_precheck_wg_uniform { static constexpr bool value = false; };
template <typename C>
struct fp4pv_online_corr_precheck_wg_uniform<C, std::void_t<decltype(C::ONLINE_CORR_PRECHECK_WG_UNIFORM)>> {
    static constexpr bool value = C::ONLINE_CORR_PRECHECK_WG_UNIFORM;
};
template <typename C, typename = void>
struct fp4pv_online_corr_precheck_wg_consume_trywait { static constexpr bool value = false; };
template <typename C>
struct fp4pv_online_corr_precheck_wg_consume_trywait<C, std::void_t<decltype(C::ONLINE_CORR_PRECHECK_WG_CONSUME_TRYWAIT)>> { static constexpr bool value = C::ONLINE_CORR_PRECHECK_WG_CONSUME_TRYWAIT; };
template <typename C, typename = void> struct fp4pv_online_corr_precheck_wg_all_lanes { static constexpr bool value = false; };
template <typename C> struct fp4pv_online_corr_precheck_wg_all_lanes<C, std::void_t<decltype(C::ONLINE_CORR_PRECHECK_WG_ALL_LANES)>> { static constexpr bool value = C::ONLINE_CORR_PRECHECK_WG_ALL_LANES; };
template <typename C, typename = void> struct fp4pv_online_corr_precheck_native_consume { static constexpr bool value = false; };
template <typename C> struct fp4pv_online_corr_precheck_native_consume<C, std::void_t<decltype(C::ONLINE_CORR_PRECHECK_NATIVE_CONSUME)>> { static constexpr bool value = C::ONLINE_CORR_PRECHECK_NATIVE_CONSUME; };

template <typename C, typename = void>
struct fp4pv_online_corr_mapping_localized {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_corr_mapping_localized<
    C, std::void_t<decltype(C::ONLINE_CORR_MAPPING_LOCALIZED)>> {
    static constexpr bool value = C::ONLINE_CORR_MAPPING_LOCALIZED;
};

template <typename C, typename = void>
struct fp4pv_online_lazy_output_rescale {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_lazy_output_rescale<C, std::void_t<decltype(C::ONLINE_LAZY_OUTPUT_RESCALE)>> {
    static constexpr bool value = C::ONLINE_LAZY_OUTPUT_RESCALE;
};

template <typename C, typename = void>
struct fp4pv_online_decoupled_pv_issue {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_decoupled_pv_issue<C, std::void_t<decltype(C::ONLINE_DECOUPLED_PV_ISSUE)>> {
    static constexpr bool value = C::ONLINE_DECOUPLED_PV_ISSUE;
};

template <typename C, typename = void>
struct fp4pv_online_v_load_warps {
    static constexpr int value = 1;
};

template <typename C>
struct fp4pv_online_v_load_warps<C, std::void_t<decltype(C::ONLINE_V_LOAD_WARPS)>> {
    static constexpr int value = C::ONLINE_V_LOAD_WARPS;
};

template <typename C, typename = void>
struct fp4pv_online_tma_v_payload {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_tma_v_payload<C, std::void_t<decltype(C::ONLINE_TMA_V_PAYLOAD)>> {
    static constexpr bool value = C::ONLINE_TMA_V_PAYLOAD;
};

template <typename C, typename = void>
struct fp4pv_online_tma_v_scale {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_tma_v_scale<C, std::void_t<decltype(C::ONLINE_TMA_V_SCALE)>> {
    static constexpr bool value = C::ONLINE_TMA_V_SCALE;
};

template <typename C, typename = void>
struct fp4pv_online_narrow_v_scale_tmem {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_narrow_v_scale_tmem<C, std::void_t<decltype(C::ONLINE_NARROW_V_SCALE_TMEM)>> {
    static constexpr bool value = C::ONLINE_NARROW_V_SCALE_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_qk_logical_n {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_qk_logical_n<C, std::void_t<decltype(C::ONLINE_QK_LOGICAL_N)>> {
    static constexpr bool value = C::ONLINE_QK_LOGICAL_N;
};

template <typename C, typename = void>
struct fp4pv_online_true_half_qk_n256 {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_true_half_qk_n256<C, std::void_t<decltype(C::ONLINE_TRUE_HALF_QK_N256)>> {
    static constexpr bool value = C::ONLINE_TRUE_HALF_QK_N256;
};

template <typename C, typename = void>
struct fp4pv_online_true_half_qk_pipe2 {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_true_half_qk_pipe2<C, std::void_t<decltype(C::ONLINE_TRUE_HALF_QK_PIPE2)>> {
    static constexpr bool value = C::ONLINE_TRUE_HALF_QK_PIPE2;
};

template <typename C, typename = void>
struct fp4pv_online_producer_regs {
    static constexpr int value = 128;
};

template <typename C>
struct fp4pv_online_producer_regs<C, std::void_t<decltype(C::ONLINE_PRODUCER_REGS)>> {
    static constexpr int value = C::ONLINE_PRODUCER_REGS;
};

template <typename C, typename = void>
struct fp4pv_online_quant_regs {
    static constexpr int value = 168;
};

template <typename C>
struct fp4pv_online_quant_regs<C, std::void_t<decltype(C::ONLINE_QUANT_REGS)>> {
    static constexpr int value = C::ONLINE_QUANT_REGS;
};

template <typename C, typename = void>
struct fp4pv_online_output_regs {
    static constexpr int value = 64;
};

template <typename C>
struct fp4pv_online_output_regs<C, std::void_t<decltype(C::ONLINE_OUTPUT_REGS)>> {
    static constexpr int value = C::ONLINE_OUTPUT_REGS;
};

template <typename C, typename = void>
struct fp4pv_online_register_output_4wg {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_register_output_4wg<
    C, std::void_t<decltype(C::ONLINE_REGISTER_OUTPUT_4WG)>> {
    static constexpr bool value = C::ONLINE_REGISTER_OUTPUT_4WG;
};

template <typename C, typename = void>
struct fp4pv_online_register_output_progressive_qk {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_register_output_progressive_qk<
    C, std::void_t<decltype(C::ONLINE_REGISTER_OUTPUT_PROGRESSIVE_QK)>> {
    static constexpr bool value = C::ONLINE_REGISTER_OUTPUT_PROGRESSIVE_QK;
};

template <typename C, typename = void>
struct fp4pv_online_register_output_progressive_qk_n32 {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_register_output_progressive_qk_n32<
    C, std::void_t<decltype(C::ONLINE_REGISTER_OUTPUT_PROGRESSIVE_QK_N32)>> {
    static constexpr bool value = C::ONLINE_REGISTER_OUTPUT_PROGRESSIVE_QK_N32;
};

template <typename C, typename = void>
struct fp4pv_online_triple_score_tmem {
    static constexpr bool value = false;
};
template <typename C>
struct fp4pv_online_triple_score_tmem<
    C, std::void_t<decltype(C::ONLINE_TRIPLE_SCORE_TMEM)>> {
    static constexpr bool value = C::ONLINE_TRIPLE_SCORE_TMEM;
};

template <typename C, typename = void>
struct fp4pv_online_relay_regs {
    static constexpr int value = 24;
};

template <typename C>
struct fp4pv_online_relay_regs<C, std::void_t<decltype(C::ONLINE_RELAY_REGS)>> {
    static constexpr int value = C::ONLINE_RELAY_REGS;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_5wg : public config_fp4pv<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    using base = config_fp4pv<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;
    static constexpr int TOTAL_WGS = 5;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_5wg_p128_force : public config_fp4pv_5wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_FP4_SWIZZLE_BYTES = 128;
    static constexpr bool FORCE_PERSISTENT_GRID = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_5wg_p128_pstage3_force : public config_fp4pv_5wg_p128_force<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg : public config_fp4pv<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    using base = config_fp4pv<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;
    static constexpr int TOTAL_WGS = 6;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p32 : public config_fp4pv_6wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_FP4_SWIZZLE_BYTES = 32;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p32_force : public config_fp4pv_6wg_p32<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p32_pstage3_force : public config_fp4pv_6wg_p32_force<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p64 : public config_fp4pv_6wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_FP4_SWIZZLE_BYTES = 64;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p64_force : public config_fp4pv_6wg_p64<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p128 : public config_fp4pv_6wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_FP4_SWIZZLE_BYTES = 128;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p128_force : public config_fp4pv_6wg_p128<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p128_pstage3_force : public config_fp4pv_6wg_p128_force<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p64_persist1 : public config_fp4pv_6wg_p64<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int PERSISTENT_GRID_MULTIPLIER = 1;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p64_pstage3_force : public config_fp4pv_6wg_p64_force<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p64_persist1_force : public config_fp4pv_6wg_p64_persist1<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p64_persist1_pstage3 : public config_fp4pv_6wg_p64_persist1<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_STAGE_SLOTS = 3;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_6wg_p64_persist1_pstage3_force : public config_fp4pv_6wg_p64_persist1_pstage3<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr bool FORCE_PERSISTENT_GRID = true;
};

template <int _Mb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv_5wg_n64_p32_persist1 : public config_fp4pv_5wg<_Mb, 64, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_FP4_SWIZZLE_BYTES = 32;
    static constexpr int PERSISTENT_GRID_MULTIPLIER = 1;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _DirectLayout, int _ClusterSize = 2>
struct config_fp4pv_6wg_direct_sc : public config_fp4pv_6wg<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize> {
    static constexpr int P_SCALE_TMEM_DIRECT_LAYOUT = _DirectLayout;
};

template <typename C, typename = void>
struct fp4pv_p_swizzle_bytes {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_p_swizzle_bytes<C, std::void_t<decltype(C::P_FP4_SWIZZLE_BYTES)>> {
    static constexpr int value = C::P_FP4_SWIZZLE_BYTES;
};

template <typename C, typename = void>
struct fp4pv_p_stage_slots {
    static constexpr int value = 2;
};

template <typename C>
struct fp4pv_p_stage_slots<C, std::void_t<decltype(C::P_STAGE_SLOTS)>> {
    static constexpr int value = C::P_STAGE_SLOTS;
};

template <int P_STAGE_SLOTS>
__device__ __forceinline__ int fp4pv_p_stage_buf_for_idx(int idx) {
    if constexpr (P_STAGE_SLOTS == 4) {
        return idx & 3;
    } else if constexpr (P_STAGE_SLOTS == 3) {
        return idx % 3;
    } else {
        return idx % P_STAGE_SLOTS;
    }
}

template <typename C, typename = void>
struct fp4pv_p_scale_tmem_direct_layout {
    static constexpr int value = -1;
};

template <typename C>
struct fp4pv_p_scale_tmem_direct_layout<C, std::void_t<decltype(C::P_SCALE_TMEM_DIRECT_LAYOUT)>> {
    static constexpr int value = C::P_SCALE_TMEM_DIRECT_LAYOUT;
};

template <typename C, typename = void>
struct fp4pv_persistent_grid_multiplier {
    static constexpr int value = 2;
};

template <typename C>
struct fp4pv_persistent_grid_multiplier<C, std::void_t<decltype(C::PERSISTENT_GRID_MULTIPLIER)>> {
    static constexpr int value = C::PERSISTENT_GRID_MULTIPLIER;
};

template <typename C, typename = void>
struct fp4pv_force_persistent_grid {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_force_persistent_grid<C, std::void_t<decltype(C::FORCE_PERSISTENT_GRID)>> {
    static constexpr bool value = C::FORCE_PERSISTENT_GRID;
};


template <typename C, typename = void>
struct fp4pv_track_b_stage2_cap2_target {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_track_b_stage2_cap2_target<
    C, std::void_t<decltype(C::TRACK_B_STAGE2_CAP2_TARGET)>> {
    static constexpr bool value = C::TRACK_B_STAGE2_CAP2_TARGET;
};

template <typename C, typename = void>
struct fp4pv_track_b_stage2_fullgrid_control {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_track_b_stage2_fullgrid_control<
    C, std::void_t<decltype(C::TRACK_B_STAGE2_FULLGRID_CONTROL)>> {
    static constexpr bool value = C::TRACK_B_STAGE2_FULLGRID_CONTROL;
};

template <typename C, typename = void>
struct fp4pv_require_track_b_stage2_cap2_persistent {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_require_track_b_stage2_cap2_persistent<
    C, std::void_t<decltype(C::REQUIRE_TRACK_B_STAGE2_CAP2_PERSISTENT)>> {
    static constexpr bool value = C::REQUIRE_TRACK_B_STAGE2_CAP2_PERSISTENT;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_task_loop {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_task_loop<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_TASK_LOOP)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_TASK_LOOP;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_normalize_rescale_phase {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_normalize_rescale_phase<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_NORMALIZE_RESCALE_PHASE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_NORMALIZE_RESCALE_PHASE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_schedwarp {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_schedwarp<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_SCHEDWARP)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_SCHEDWARP;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_schedwg {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_schedwg<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_SCHEDWG)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_SCHEDWG;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_static_stride {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_static_stride<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_STATIC_STRIDE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_STATIC_STRIDE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_multitask_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_multitask_reuse<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE;
};

template <typename C, typename = void>
struct fp4pv_require_one_wave_persistent { static constexpr bool value = false; };
template <typename C>
struct fp4pv_require_one_wave_persistent<C, std::void_t<decltype(C::REQUIRE_ONE_WAVE_PERSISTENT)>> {
    static constexpr bool value = C::REQUIRE_ONE_WAVE_PERSISTENT;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_adjacent_query_pair {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_adjacent_query_pair<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_ADJACENT_QUERY_PAIR)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_ADJACENT_QUERY_PAIR;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_task_epoch_broadcast {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_task_epoch_broadcast<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_TASK_EPOCH_BROADCAST)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_TASK_EPOCH_BROADCAST;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_role_done_wg_sync {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_role_done_wg_sync<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_ROLE_DONE_WG_SYNC)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_ROLE_DONE_WG_SYNC;
};

template <typename C, typename = void>
struct fp4pv_online_cluster_slot_lifetime_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_cluster_slot_lifetime_diag<C, std::void_t<decltype(C::ONLINE_CLUSTER_SLOT_LIFETIME_DIAG)>> {
    static constexpr bool value = C::ONLINE_CLUSTER_SLOT_LIFETIME_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_cluster_slot_lifetime_live_mirror {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_cluster_slot_lifetime_live_mirror<C, std::void_t<decltype(C::ONLINE_CLUSTER_SLOT_LIFETIME_LIVE_MIRROR)>> {
    static constexpr bool value = C::ONLINE_CLUSTER_SLOT_LIFETIME_LIVE_MIRROR;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_carry_phases {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_carry_phases<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_CARRY_PHASES)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_CARRY_PHASES;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_issue_done_leader {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_issue_done_leader<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_LEADER)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_LEADER;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_output_reuse_after_store_wait {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_output_reuse_after_store_wait<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_AFTER_STORE_WAIT)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_AFTER_STORE_WAIT;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_output_reuse_fence {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_output_reuse_fence<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_FENCE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_OUTPUT_REUSE_FENCE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_task_done_arrivals {
    static constexpr int value = 5;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_task_done_arrivals<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS)>> {
    static constexpr int value = C::ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_pv_wait_output_reuse {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_pv_wait_output_reuse<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_PV_WAIT_OUTPUT_REUSE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_PV_WAIT_OUTPUT_REUSE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_reset_pv_output {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_reset_pv_output<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_reset_v_pipe {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_reset_v_pipe<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_reset_score_copy {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_reset_score_copy<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_RESET_SCORE_COPY)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_RESET_SCORE_COPY;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_skip_issue_tail_score_copy_wait {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_skip_issue_tail_score_copy_wait<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_SKIP_ISSUE_TAIL_SCORE_COPY_WAIT)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_SKIP_ISSUE_TAIL_SCORE_COPY_WAIT;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_issue_done_active_only {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_issue_done_active_only<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_ACTIVE_ONLY)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_ACTIVE_ONLY;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_role_start_sync {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_role_start_sync<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_ROLE_START_SYNC)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_ROLE_START_SYNC;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_producer_done_sync {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_producer_done_sync<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_PRODUCER_DONE_SYNC)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_PRODUCER_DONE_SYNC;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_drain_q_finished_before_done {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_drain_q_finished_before_done<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_DRAIN_Q_FINISHED_BEFORE_DONE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_DRAIN_Q_FINISHED_BEFORE_DONE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_drain_v_finished_before_done {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_drain_v_finished_before_done<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_DRAIN_V_FINISHED_BEFORE_DONE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_DRAIN_V_FINISHED_BEFORE_DONE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_reset_q_phases {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_reset_q_phases<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_RESET_Q_PHASES)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_RESET_Q_PHASES;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_task_done_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_task_done_diag<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_task_done_timeout_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_task_done_timeout_diag<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_taskdiag_no_reset {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_taskdiag_no_reset<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_NO_RESET)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_NO_RESET;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_taskdiag_all_tasks {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_taskdiag_all_tasks<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_taskdiag_long_spin {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_taskdiag_long_spin<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_LONG_SPIN)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_LONG_SPIN;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_skip_reused_task_bodies {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_skip_reused_task_bodies<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_SKIP_REUSED_TASK_BODIES)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_SKIP_REUSED_TASK_BODIES;
};
template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_second_task_diag { static constexpr bool value = false; };
template <typename C>
struct fp4pv_online_persistent_outer_clc_second_task_diag<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_SECOND_TASK_DIAG)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_SECOND_TASK_DIAG;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_reinit_task_done {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_reinit_task_done<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_reset_corr_rescale {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_reset_corr_rescale<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_RESET_CORR_RESCALE)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_RESET_CORR_RESCALE;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_skip_task_output_reuse_wait {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_skip_task_output_reuse_wait<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT;
};

template <typename C, typename = void>
struct fp4pv_online_persistent_outer_clc_blocking_post_task_reinit {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_persistent_outer_clc_blocking_post_task_reinit<C, std::void_t<decltype(C::ONLINE_PERSISTENT_OUTER_CLC_BLOCKING_POST_TASK_REINIT)>> {
    static constexpr bool value = C::ONLINE_PERSISTENT_OUTER_CLC_BLOCKING_POST_TASK_REINIT;
};

template <typename C, typename = void>
struct fp4pv_online_v_tma_single_warp_publish_fix {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_v_tma_single_warp_publish_fix<C, std::void_t<decltype(C::ONLINE_V_TMA_SINGLE_WARP_PUBLISH_FIX)>> {
    static constexpr bool value = C::ONLINE_V_TMA_SINGLE_WARP_PUBLISH_FIX;
};

template <typename C, typename = void>
struct fp4pv_real_fwd_timeline_trace {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_real_fwd_timeline_trace<
    C, std::void_t<decltype(C::REAL_FWD_TIMELINE_TRACE)>> {
    static constexpr bool value = C::REAL_FWD_TIMELINE_TRACE;
};

template <typename C, typename = void>
struct fp4pv_real_fwd_readiness_probe {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_real_fwd_readiness_probe<
    C, std::void_t<decltype(C::REAL_FWD_READINESS_PROBE)>> {
    static constexpr bool value = C::REAL_FWD_READINESS_PROBE;
};

template <typename C, typename = void>
struct fp4pv_real_fwd_corr_pv_cycle_diag {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_real_fwd_corr_pv_cycle_diag<
    C, std::void_t<decltype(C::REAL_FWD_CORR_PV_CYCLE_DIAG)>> {
    static constexpr bool value = C::REAL_FWD_CORR_PV_CYCLE_DIAG;
};

template <typename C, typename = void>
struct fp4pv_real_fwd_corr_pv_cycle_mask {
    static constexpr int value = 0;
};

template <typename C>
struct fp4pv_real_fwd_corr_pv_cycle_mask<
    C, std::void_t<decltype(C::REAL_FWD_CORR_PV_CYCLE_MASK)>> {
    static constexpr int value = C::REAL_FWD_CORR_PV_CYCLE_MASK;
};

template <typename C, typename = void>
struct fp4pv_real_fwd_corr_pv_cycle_balance_pad {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_real_fwd_corr_pv_cycle_balance_pad<
    C, std::void_t<decltype(C::REAL_FWD_CORR_PV_CYCLE_BALANCE_PAD)>> {
    static constexpr bool value = C::REAL_FWD_CORR_PV_CYCLE_BALANCE_PAD;
};

template <typename C>
static constexpr int fp4pv_persistent_grid_cap() {
    return fp4pv_persistent_grid_multiplier<C>::value * C::NUM_SM;
}

static constexpr float FP4PV_MXFP4_ONLINE_P_SCALE = 2688.0f;
static constexpr float FP4PV_MXFP4_ONLINE_LOG2_P_SCALE = 11.39231742277876f;
static constexpr float FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE = 13.977279923499916f;
static constexpr float FP4PV_MXFP4_ONLINE_RCP_6P_SCALE = 6.200396825396825e-05f;
static constexpr uint8_t FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_E8M0 = 0x8b;
static constexpr uint32_t FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_WORD = 0x8b8b8b8bu;
static constexpr float FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_VALUE = 4096.0f;
static constexpr float FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF =
    (6.0f * FP4PV_MXFP4_ONLINE_P_SCALE) / FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_VALUE;
// log2(FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF) = log2(3.9375); folding the
// payload quantization coefficient into log-space avoids an extra multiply in
// the fused softmax/P-pack path.
static constexpr float FP4PV_MXFP4_ONLINE_FIXED_P_LOG2_COEFF = 1.9772799234999165f;

template <typename C, bool NEED_P_SCAN = false>
struct globals_fp4pv_mxfp4_dv {
    using base = globals_fp4pv<C, NEED_P_SCAN>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = std::conditional_t<
        fp4pv_online_qk_pair_nb64<C>::value ||
            fp4pv_online_full_k_tile<C>::value,
        st_fp4e2m1_2<C::Nb, C::Dqk/2>,
        typename base::k_tile>;
    using k_true_half_tile = st_fp4e2m1_2<2 * C::Nb, C::Dqk / 2>;
    using k_progressive_half_tile = st_fp4e2m1_2<C::Nb / 2, C::Dqk / 2>;
    using k_progressive_quarter_tile =
        st_fp4e2m1_2<C::Nb / 4, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = std::conditional_t<
        C::Nb == 96,
        st_fp4e2m1_2<C::Mb, C::Nb / 2, false>,
        std::conditional_t<
        fp4pv_p_swizzle_bytes<C>::value == 0,
        typename base::p_fp4_tile,
        std::conditional_t<
            fp4pv_p_swizzle_bytes<C>::value == 128,
            st_fp4e2m1_2<C::Mb, C::Nb, true, 128>,
            st_fp4e2m1_2<C::Mb, C::Nb / 2, true, fp4pv_p_swizzle_bytes<C>::value>>>>;
    using p_sc_tile = st_fp8e8m0<32, 16, false>;
    using p_sc_guard_slot = byte_guard<1024>;
    using v_fp4_tile = std::conditional_t<
        (C::CLUSTER_SIZE == 1 || fp4pv_online_rowpar2_ranklocal<C>::value ||
         fp4pv_online_native_group2_local_group1_pv<C>::value),
        st_fp4e2m1_2<
            C::Dvo, C::Nb == 96 ? 64 : C::Nb / 2>,
        typename base::v_fp4_tile>;
    using v_sc_tile = st_fp8e8m0<32, 16, false>;
    using p_fp4_k256_tile = st_fp4e2m1_2<C::Mb, C::Nb>;
    using p_sc_k256_tile = st_fp8e8m0<32, 16, false>;
    using v_fp4_k256_tile = st_fp4e2m1_2<C::Dvo / C::CLUSTER_SIZE, C::Nb>;
    using v_sc_k256_tile = st_fp8e8m0<32, 16, false>;
    using o_tile = typename base::o_tile;
    using fp32_output_accum_tile = st_fl<C::Mb, C::Dvo, false>;
    using fp32_output_half_accum_tile = st_fl<C::Mb, C::Dvo / 2, false>;
    using q_sc_smem_storage = typename base::q_sc_smem_storage;
    using k_sc_smem_storage = typename base::k_sc_smem_storage;
    using k_sc_guard_slot = typename base::k_sc_guard_slot;

    using q_gl = typename base::q_gl;
    using q_sc_gl = typename base::q_sc_gl;
    using q_sg_gl = typename base::q_sg_gl;
    using k_gl = std::conditional_t<
        fp4pv_online_register_output_progressive_qk<C>::value,
        gl<fp4e2m1_2, -1, -1, -1, -1,
           tma::descriptor<k_tile, dim::ROW>,
           tma::descriptor<k_progressive_half_tile, dim::ROW>,
           tma::descriptor<k_progressive_quarter_tile, dim::ROW>>,
        std::conditional_t<
        fp4pv_online_true_half_qk_n256<C>::value,
        gl<fp4e2m1_2, -1, -1, -1, -1,
           tma::descriptor<k_tile, dim::ROW>,
           tma::descriptor<k_true_half_tile, dim::ROW>>,
        std::conditional_t<fp4pv_online_qk_pair_nb64<C>::value ||
            fp4pv_online_full_k_tile<C>::value,
        gl<fp4e2m1_2, -1, -1, -1, -1, tma::descriptor<k_tile, dim::ROW>>,
        typename base::k_gl>>>;
    using k_sc_gl = typename base::k_sc_gl;
    using k_sg_gl = typename base::k_sg_gl;
    using v_fp4_gl = gl<fp4e2m1_2, -1, -1, -1, -1, tma::descriptor<v_fp4_tile, dim::ROW>>;
    using o_gl = typename base::o_gl;
    using lse_gl = typename base::lse_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_tile) * C::NUM_SOFTMAXXERS);
        add((int)sizeof(q_sc_smem_storage));
        add((int)sizeof(k_tile) * C::LOAD_STAGES);
        if constexpr (fp4pv_online_true_half_qk_n256<C>::value) {
            constexpr int true_half_k_slots = fp4pv_online_true_half_qk_pipe2<C>::value ? 2 : 1;
            add((int)sizeof(k_true_half_tile) * true_half_k_slots);
        }
        add((int)sizeof(k_sc_smem_storage));
        if constexpr (fp4pv_online_register_output_progressive_qk<C>::value) {
            constexpr int repack_pages =
                fp4pv_online_register_output_progressive_qk_n32<C>::value
                    ? 3
                    : 1;
            add((int)sizeof(k_sc_tile) * repack_pages);
        }
        add((int)sizeof(k_sc_guard_slot));
        if constexpr (fp4pv_online_rowpar2_k_multicast_short_diag<C>::value) {
            add((int)sizeof(k_sc_tile));
        }
        add((int)sizeof(v_fp4_tile) * C::V_LOAD_STAGES);
        add((int)sizeof(v_sc_tile) * C::V_LOAD_STAGES);
        add((int)sizeof(o_tile));
        if constexpr (fp4pv_online_fp32_smem_output_accum<C>::value) {
            add((int)sizeof(fp32_output_accum_tile));
        } else if constexpr (fp4pv_online_register_smem_hybrid<C>::value) {
            add((int)sizeof(fp32_output_half_accum_tile));
        }
        add((int)(sizeof(p_fp4_tile) * fp4pv_p_stage_slots<C>::value));
        add((int)(sizeof(p_sc_tile) * fp4pv_p_stage_slots<C>::value));
        add((int)sizeof(p_sc_guard_slot));
        return bytes;
    }

    static constexpr int compute_k256_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_tile) * C::NUM_SOFTMAXXERS);
        add((int)sizeof(q_sc_smem_storage));
        add((int)sizeof(k_tile) * C::LOAD_STAGES);
        if constexpr (fp4pv_online_true_half_qk_n256<C>::value) {
            constexpr int true_half_k_slots = fp4pv_online_true_half_qk_pipe2<C>::value ? 2 : 1;
            add((int)sizeof(k_true_half_tile) * true_half_k_slots);
        }
        add((int)sizeof(k_sc_smem_storage));
        add((int)sizeof(k_sc_guard_slot));
        if constexpr (fp4pv_online_rowpar2_k_multicast_short_diag<C>::value) {
            add((int)sizeof(k_sc_tile));
        }
        add((int)(sizeof(v_fp4_k256_tile) * 2));
        add((int)(sizeof(v_sc_k256_tile) * 4));
        add((int)sizeof(o_tile));
        add((int)(sizeof(p_fp4_k256_tile) * 2));
        add((int)(sizeof(p_sc_k256_tile) * 4));
        add((int)sizeof(p_sc_guard_slot));
        return bytes;
    }

public:
    static constexpr bool NEED_P_SCAN_STORAGE = false;
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();
    static constexpr int K256_DYNAMIC_SHARED_MEMORY = compute_k256_dynamic_shared_memory();

    q_gl q;
    q_sc_gl q_sc;
    q_sg_gl q_sg;
    k_gl k;
    k_sc_gl k_sc;
    k_sg_gl k_sg;
    v_fp4_gl v_fp4_tma;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    o_gl o;
    lse_gl lse;
    float p_output_scale;
    int p_quant_mode;
    int disable_lse_store = 0;
    int dvo_half = 0;
    int v_scale_records_per_head = 1;

    __host__ __inline__ dim3 grid() {
        const int tiles_m = q.rows() / C::Mb;
        const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
        const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
        const int total_bids = q.batch() * q.depth() * num_block * C::CLUSTER_SIZE;
        constexpr int PERSISTENT_GRID_CAP = fp4pv_persistent_grid_cap<C>();
        int grid_x = min(total_bids, PERSISTENT_GRID_CAP);
        if (grid_x < C::CLUSTER_SIZE) {
            grid_x = C::CLUSTER_SIZE;
        } else {
            grid_x = ((grid_x + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        }
        return dim3(grid_x);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
    __host__ __inline__ int dynamic_shared_memory_k256() { return K256_DYNAMIC_SHARED_MEMORY; }
};

#if TK_FA4_REAL_FWD_TIMELINE_PROBE_BUILD
// Persistent S4096/H64 scaffold: scheduler-owned 4WG full-V lifecycle with
// one resident worker wave (152 CTAs) and explicit per-task resets.
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent
    : public config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_schedown_taskdiag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128,128,192,128,200,56,112,1> {
    static constexpr int PERSISTENT_GRID_MULTIPLIER = 1;
    static constexpr bool REQUIRE_ONE_WAVE_PERSISTENT = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ISSUE_DONE_ACTIVE_ONLY = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_CORR_RESCALE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_SCORE_COPY = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE = true;
    static_assert(ONLINE_PERSISTENT_OUTER_TASK_LOOP && ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE &&
                  ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG &&
                  ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS &&
                  ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE &&
                  PERSISTENT_GRID_MULTIPLIER == 1);
};
struct config_fp4pv_real_fwd_high_corr_precheck_v25_g_persistent
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent {
    static constexpr bool ONLINE_CORR_PRECHECK = true;
    static constexpr bool ONLINE_CORR_PRECHECK_CONTROL = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = true;
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent {
    static constexpr int ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS = 6;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS == 6);
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_reused
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_REUSED_TASK_BODIES = true;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_SKIP_REUSED_TASK_BODIES);
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_skip_reused
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6 {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_REUSED_TASK_BODIES = true;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS == 6 &&
                  ONLINE_PERSISTENT_OUTER_CLC_SKIP_REUSED_TASK_BODIES);
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_skip_reused_epoch
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_skip_reused {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_EPOCH_BROADCAST = true;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS == 6 &&
                  ONLINE_PERSISTENT_OUTER_CLC_TASK_EPOCH_BROADCAST);
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_fullskip
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_skip_reused {
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS == 6 &&
                  ONLINE_PERSISTENT_OUTER_CLC_SKIP_REUSED_TASK_BODIES);
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_fullskip_timeout
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_fullskip {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG = true;
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_second_task_diag
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SECOND_TASK_DIAG = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_Q_PHASES = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG = true;
    // Scheduler task_done already includes output; skip the redundant task-output reuse wait.
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT = true;
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_diag
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT = true;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS);
    static_assert(ONLINE_CLUSTER_SLOT_LIFETIME_DIAG);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_CORR_RESCALE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_SCORE_COPY);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE);
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG = false;
    static constexpr bool ONLINE_CLUSTER_SLOT_LIFETIME_DIAG = false;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT = true;
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG);
    static_assert(!ONLINE_CLUSTER_SLOT_LIFETIME_DIAG);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_CORR_RESCALE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_PV_OUTPUT);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_SCORE_COPY);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_V_PIPE);
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_reinit
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag {
    // The diagnostic scaffold honored REINIT_TASK_DONE in its post-task branch.
    // Keep the production/blocking path semantically identical without enabling
    // diagnostic polling or atomics.
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_BLOCKING_POST_TASK_REINIT = true;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_BLOCKING_POST_TASK_REINIT);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_REINIT_TASK_DONE);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG);
    static_assert(!ONLINE_CLUSTER_SLOT_LIFETIME_DIAG);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_SKIP_TASK_OUTPUT_REUSE_WAIT);
};
struct config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_safe
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_reinit {
    // Every task-local Q/V semaphore is freshly initialized at the scheduler
    // boundary.  Consume those barriers with their fresh phase and drain the
    // final asynchronous QK/PV completions before task_done permits the next
    // reinitialization.
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_RESET_Q_PHASES = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_DRAIN_Q_FINISHED_BEFORE_DONE = true;
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_DRAIN_V_FINISHED_BEFORE_DONE = true;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_Q_PHASES);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_DRAIN_Q_FINISHED_BEFORE_DONE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_DRAIN_V_FINISHED_BEFORE_DONE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_BLOCKING_POST_TASK_REINIT);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG);
    static_assert(!ONLINE_CLUSTER_SLOT_LIFETIME_DIAG);
};
// Equal-wave structural control: preserve the reinit-only parent arithmetic
// and task lifecycle, but bind each CTA to one adjacent pair of M128 queries.
// The dedicated host route launches H*(S/256) CTAs instead of the one-wave
// scheduler grid.  At S4096/H64 this is 1024 CTAs, or seven residency waves,
// matching the 1024 two-query jobs served by the 148-CTA BF16 persistent grid.
struct config_fp4pv_real_fwd_high_corr_pv_control_adjacent_pair_halfgrid_reinit
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_reinit {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ADJACENT_QUERY_PAIR = true;
    // Make the default five-role contract explicit for this topology instead
    // of relying on fp4pv_online_persistent_outer_clc_task_done_arrivals's
    // fallback.  The live roles are producer, QK issue, PV issue, output, and
    // quant; decoupled PV compiles out the separate V-scale role.
    static constexpr int ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS = 5;
    static constexpr bool REQUIRE_ONE_WAVE_PERSISTENT = false;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_ADJACENT_QUERY_PAIR);
    static_assert(ONLINE_PERSISTENT_OUTER_TASK_LOOP);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_BLOCKING_POST_TASK_REINIT);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS == 5);
    static_assert(CLUSTER_SIZE == 1);
    static_assert(NUM_SOFTMAXXERS == 1);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG);
    static_assert(!ONLINE_CLUSTER_SLOT_LIFETIME_DIAG);
    static_assert(!REQUIRE_ONE_WAVE_PERSISTENT);
};
// Same adjacent-pair topology with the validated safe task boundary: reset
// the freshly initialized Q phases and drain final Q/V completion before the
// scheduler reinitializes task-local barriers.  This isolates the task1-only
// nondeterminism seen in the reinit-only v26 route without changing mapping
// or attention arithmetic.
struct config_fp4pv_real_fwd_high_corr_pv_control_adjacent_pair_halfgrid_safe
    : public config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_safe {
    static constexpr bool ONLINE_PERSISTENT_OUTER_CLC_ADJACENT_QUERY_PAIR = true;
    static constexpr int ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS = 5;
    static constexpr bool REQUIRE_ONE_WAVE_PERSISTENT = false;
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_ADJACENT_QUERY_PAIR);
    static_assert(ONLINE_PERSISTENT_OUTER_TASK_LOOP);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_MULTITASK_REUSE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_BLOCKING_POST_TASK_REINIT);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_RESET_Q_PHASES);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_DRAIN_Q_FINISHED_BEFORE_DONE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_DRAIN_V_FINISHED_BEFORE_DONE);
    static_assert(ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_ARRIVALS == 5);
    static_assert(CLUSTER_SIZE == 1);
    static_assert(NUM_SOFTMAXXERS == 1);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_DIAG);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASKDIAG_ALL_TASKS);
    static_assert(!ONLINE_PERSISTENT_OUTER_CLC_TASK_DONE_TIMEOUT_DIAG);
    static_assert(!ONLINE_CLUSTER_SLOT_LIFETIME_DIAG);
    static_assert(!REQUIRE_ONE_WAVE_PERSISTENT);
};
#endif

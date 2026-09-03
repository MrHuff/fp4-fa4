#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) static void __tk_fa4_fp4pv_hidden_module(pybind11::module_ &variable)
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1
#include "../b300_causal/bf16_b300_mha_causal_fp4.cu"
#undef TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH
#pragma pop_macro("PYBIND11_MODULE")
#include "../b300_bwd_fa4.cuh"

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
    static constexpr int ONLINE_V_LOAD_WARPS = 2;
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

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_regs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ClusterSize> {
    static constexpr bool ONLINE_DECOUPLED_PV_ISSUE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_DECOUPLED_PV_ISSUE = true;
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _QuantRegs, int _OutputRegs, int _ProducerRegs, int _ClusterSize = 2>
struct config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_splitk64_pregs_force_persistent
    : public config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<_Mb, _Nb, _Dqk, _Dvo, _QuantRegs, _OutputRegs, _ProducerRegs, _ClusterSize> {
    static constexpr bool ONLINE_SPLIT_P_READY_K64 = true;
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
struct fp4pv_online_split_output_accum {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_split_output_accum<C, std::void_t<decltype(C::ONLINE_SPLIT_OUTPUT_ACCUM)>> {
    static constexpr bool value = C::ONLINE_SPLIT_OUTPUT_ACCUM;
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
struct fp4pv_online_split_p_ready_k64 {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_split_p_ready_k64<C, std::void_t<decltype(C::ONLINE_SPLIT_P_READY_K64)>> {
    static constexpr bool value = C::ONLINE_SPLIT_P_READY_K64;
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
struct fp4pv_online_dual_output_accum_direct_after_rescale {
    static constexpr bool value = false;
};

template <typename C>
struct fp4pv_online_dual_output_accum_direct_after_rescale<C, std::void_t<decltype(C::ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE)>> {
    static constexpr bool value = C::ONLINE_DUAL_OUTPUT_ACCUM_DIRECT_AFTER_RESCALE;
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
        return idx & 1;
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

template <typename C>
static constexpr int fp4pv_persistent_grid_cap() {
    return fp4pv_persistent_grid_multiplier<C>::value * C::NUM_SM;
}

static constexpr float FP4PV_MXFP4_ONLINE_P_SCALE = 2688.0f;
static constexpr uint8_t FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_E8M0 = 0x8b;
static constexpr uint32_t FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_WORD = 0x8b8b8b8bu;
static constexpr float FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_VALUE = 4096.0f;
static constexpr float FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF =
    (6.0f * FP4PV_MXFP4_ONLINE_P_SCALE) / FP4PV_MXFP4_ONLINE_FIXED_P_SCALE_VALUE;

template <typename C, bool NEED_P_SCAN = false>
struct globals_fp4pv_mxfp4_dv {
    using base = globals_fp4pv<C, NEED_P_SCAN>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = typename base::k_tile;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = std::conditional_t<
        fp4pv_p_swizzle_bytes<C>::value == 0,
        typename base::p_fp4_tile,
        std::conditional_t<
            fp4pv_p_swizzle_bytes<C>::value == 128,
            st_fp4e2m1_2<C::Mb, C::Nb, true, 128>,
            st_fp4e2m1_2<C::Mb, C::Nb / 2, true, fp4pv_p_swizzle_bytes<C>::value>>>;
    using p_sc_tile = st_fp8e8m0<32, 16, false>;
    using p_sc_guard_slot = byte_guard<1024>;
    using v_fp4_tile = std::conditional_t<
        C::CLUSTER_SIZE == 1,
        st_fp4e2m1_2<C::Dvo, C::Nb / 2>,
        typename base::v_fp4_tile>;
    using v_sc_tile = st_fp8e8m0<32, 16, false>;
    using p_fp4_k256_tile = st_fp4e2m1_2<C::Mb, C::Nb>;
    using p_sc_k256_tile = st_fp8e8m0<32, 16, false>;
    using v_fp4_k256_tile = st_fp4e2m1_2<C::Dvo / 2, C::Nb>;
    using v_sc_k256_tile = st_fp8e8m0<32, 16, false>;
    using o_tile = typename base::o_tile;
    using q_sc_smem_storage = typename base::q_sc_smem_storage;
    using k_sc_smem_storage = typename base::k_sc_smem_storage;
    using k_sc_guard_slot = typename base::k_sc_guard_slot;

    using q_gl = typename base::q_gl;
    using q_sc_gl = typename base::q_sc_gl;
    using q_sg_gl = typename base::q_sg_gl;
    using k_gl = typename base::k_gl;
    using k_sc_gl = typename base::k_sc_gl;
    using k_sg_gl = typename base::k_sg_gl;
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
        add((int)sizeof(k_sc_smem_storage));
        add((int)sizeof(k_sc_guard_slot));
        add((int)sizeof(v_fp4_tile) * C::LOAD_STAGES);
        add((int)sizeof(v_sc_tile) * C::LOAD_STAGES);
        add((int)sizeof(o_tile));
        add((int)(sizeof(p_fp4_tile) * fp4pv_p_stage_slots<C>::value));
        add((int)(sizeof(p_sc_tile) * fp4pv_p_stage_slots<C>::value));
        add((int)sizeof(p_sc_guard_slot));
        return bytes;
    }

    static constexpr int compute_k256_dynamic_shared_memory() {
        int bytes = compute_dynamic_shared_memory();
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)(sizeof(p_fp4_k256_tile) * 2));
        add((int)(sizeof(p_sc_k256_tile) * 4));
        add((int)(sizeof(v_fp4_k256_tile) * 2));
        add((int)(sizeof(v_sc_k256_tile) * 4));
        return bytes;
    }

public:
    static constexpr bool NEED_P_SCAN_STORAGE = false;
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();
    static constexpr int K256_DYNAMIC_SHARED_MEMORY = compute_k256_dynamic_shared_memory();
    static constexpr int K256_DYNAMIC_SHARED_MEMORY_PADDED = K256_DYNAMIC_SHARED_MEMORY + 16 * 1024;

    q_gl q;
    q_sc_gl q_sc;
    q_sg_gl q_sg;
    k_gl k;
    k_sc_gl k_sc;
    k_sg_gl k_sg;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    o_gl o;
    lse_gl lse;
    float p_output_scale;
    int p_quant_mode;
    int disable_lse_store = 0;

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
};

template <typename C, bool NEED_P_SCAN>
__device__ inline float qk_scale_log2(const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g, int b, int h) {
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

__device__ __forceinline__ uint8_t fp4pv_float_to_e8m0_rte(float val) {
    uint32_t u = __float_as_uint(val);
    uint32_t exp32 = (u >> 23) & 0xFFu;
    if (!(val > 1e-38f) || exp32 == 0xFFu) {
        return 0x00;
    }
    uint8_t exp = static_cast<uint8_t>(exp32);
    uint32_t mant = u & 0x7FFFFF;
    constexpr uint32_t half = 1u << 22;
    const bool round_up = (mant > half) || (mant == half && (exp & 1));
    if (round_up && exp < 0xFE) {
        ++exp;
    }
    return exp;
}

__device__ __forceinline__ uint8_t fp4pv_float_to_e8m0_ceil(float val) {
    uint32_t u = __float_as_uint(val);
    uint32_t exp32 = (u >> 23) & 0xFFu;
    if (!(val > 1e-38f) || exp32 == 0xFFu) {
        return 0x00;
    }
    uint8_t exp = static_cast<uint8_t>(exp32);
    const uint32_t mant = u & 0x7FFFFF;
    if (mant > 0 && exp < 0xFE) {
        ++exp;
    }
    return exp;
}

__device__ __forceinline__ uint8_t fp4pv_float_to_e8m0_floor(float val) {
    uint32_t u = __float_as_uint(val);
    uint32_t exp32 = (u >> 23) & 0xFFu;
    if (!(val > 1e-38f) || exp32 == 0xFFu) {
        return 0x00;
    }
    return static_cast<uint8_t>(exp32);
}

__device__ __forceinline__ uint8_t fp4pv_float_to_e8m0(float val, int mode) {
    if (mode == 1) {
        return fp4pv_float_to_e8m0_ceil(val);
    }
    if (mode == 2) {
        return fp4pv_float_to_e8m0_floor(val);
    }
    return fp4pv_float_to_e8m0_rte(val);
}

__device__ __forceinline__ float fp4pv_exp2f_rcp_e8m0(uint8_t e8m0) {
    if (e8m0 == 0) {
        return 0.0f;
    }
    return __uint_as_float((uint32_t)(254 - e8m0) << 23);
}

__device__ __forceinline__ uint16_t fp4pv_bf16_pair_max_abs_bits(const bf16_2 &v) {
    const uint32_t bits = *reinterpret_cast<const uint32_t *>(&v);
    const uint16_t x = static_cast<uint16_t>(bits & 0x7fffu);
    const uint16_t y = static_cast<uint16_t>((bits >> 16) & 0x7fffu);
    return x > y ? x : y;
}

__device__ __forceinline__ uint8_t fp4pv_bf16_abs_bits_to_e8m0(uint16_t abs_bits, int mode) {
    const uint8_t exp = static_cast<uint8_t>((abs_bits >> 7) & 0xffu);
    if (exp == 0 || exp == 0xff) {
        return 0x00;
    }
    const uint16_t mant = abs_bits & 0x7fu;
    if (mode == 1) {
        return static_cast<uint8_t>((mant > 0 && exp < 0xfeu) ? exp + 1 : exp);
    }
    if (mode == 2) {
        return exp;
    }
    const bool round_up = (mant > 0x40u) || (mant == 0x40u && (exp & 1u));
    return static_cast<uint8_t>((round_up && exp < 0xfeu) ? exp + 1 : exp);
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
        "r"(smem_addr),
        "r"(words[0]),
        "r"(words[1]),
        "r"(words[2]),
        "r"(words[3]));
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_v2(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::
        "r"(smem_addr),
        "r"(words[0]),
        "r"(words[1]));
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::
        "r"(smem_addr + 8),
        "r"(words[2]),
        "r"(words[3]));
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_scalar(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 0), "r"(words[0]));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 4), "r"(words[1]));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 8), "r"(words[2]));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 12), "r"(words[3]));
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_striped(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    const int row_phase = warp::laneid() >> 3;
    #pragma unroll
    for (int phase = 0; phase < 4; ++phase) {
        if (row_phase == phase) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                "r"(smem_addr),
                "r"(words[0]),
                "r"(words[1]),
                "r"(words[2]),
                "r"(words[3]));
        }
    }
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_half_striped(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    const int row_phase = warp::laneid() >> 4;
    #pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        if (row_phase == phase) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                "r"(smem_addr),
                "r"(words[0]),
                "r"(words[1]),
                "r"(words[2]),
                "r"(words[3]));
        }
    }
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_staggered(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    const int row_phase = warp::laneid() >> 3;
    #pragma unroll
    for (int phase = 0; phase < 4; ++phase) {
        if (((row_phase + mx_block) & 3) == phase) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                "r"(smem_addr),
                "r"(words[0]),
                "r"(words[1]),
                "r"(words[2]),
                "r"(words[3]));
        }
    }
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_half_staggered(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    const int row = global_row & 127;
    const int col = packed_col_offset + mx_block * 16;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    const int row_phase = warp::laneid() >> 4;
    #pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        if (((row_phase + (mx_block & 1)) & 1) == phase) {
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                "r"(smem_addr),
                "r"(words[0]),
                "r"(words[1]),
                "r"(words[2]),
                "r"(words[3]));
        }
    }
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group_mxfp4_selected(
    PTile &p_fp4_tile,
    const uint32_t (&words)[4],
    int global_row,
    int mx_block,
    int packed_col_offset = 0
) {
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
}

template <typename PScTile>
__device__ inline void fp4pv_store_mxfp4_scale_byte_swizzled(
    PScTile &p_sc_tile,
    int global_row,
    int mx_block,
    uint8_t e8m0
) {
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, mx_block);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    asm volatile("st.shared.b8 [%0], %1;" :: "r"(smem_addr), "r"((uint32_t)e8m0));
}

template <typename PScTile>
__device__ inline void fp4pv_store_mxfp4_scale_word_swizzled(
    PScTile &p_sc_tile,
    int global_row,
    uint32_t packed
) {
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, 0);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr), "r"(packed));
}

template <typename PScTile>
__device__ inline void fp4pv_store_mxfp4_scale_word4_swizzled(
    PScTile &p_sc_tile,
    int row_in_32,
    uint32_t packed0,
    uint32_t packed1,
    uint32_t packed2,
    uint32_t packed3
) {
    const int dest = (row_in_32 & 31) * 16;
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
        "r"(smem_addr),
        "r"(packed0),
        "r"(packed1),
        "r"(packed2),
        "r"(packed3));
}

__device__ __forceinline__ void fp4pv_store_global_u32x4(
    uint8_t *__restrict__ dst,
    const uint32_t (&words)[4]
) {
    const uint4 packed = {words[0], words[1], words[2], words[3]};
    *reinterpret_cast<uint4 *>(dst) = packed;
}

__device__ __forceinline__ void fp4pv_store_global_u32x4_value(
    uint8_t *__restrict__ dst,
    uint32_t word
) {
    const uint4 packed = {word, word, word, word};
    *reinterpret_cast<uint4 *>(dst) = packed;
}

__device__ __forceinline__ void fp4pv_store_global_scale_word_swizzled(
    uint8_t *__restrict__ scale_tile,
    int row,
    uint32_t packed
) {
    uint8_t *dst = scale_tile + fp4pv_scale_swizzle_idx(row, 0);
    *reinterpret_cast<uint32_t *>(dst) = packed;
}

template <int HALF, typename PScTile>
__device__ inline void fp4pv_store_mxfp4_scale_halfword4_swizzled(
    PScTile &p_sc_tile,
    int row_in_32,
    uint32_t packed0,
    uint32_t packed1,
    uint32_t packed2,
    uint32_t packed3
) {
    static_assert(HALF == 0 || HALF == 1, "K64 half-scale shared store expects half 0 or 1");
    const int dest = (row_in_32 & 31) * 16 + HALF * 2;
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    asm volatile(
        "st.shared.b16 [%0], %1;\n"
        "st.shared.b16 [%2], %3;\n"
        "st.shared.b16 [%4], %5;\n"
        "st.shared.b16 [%6], %7;\n" ::
        "r"(smem_addr + 0u),
        "r"(packed0 & 0xffffu),
        "r"(smem_addr + 4u),
        "r"(packed1 & 0xffffu),
        "r"(smem_addr + 8u),
        "r"(packed2 & 0xffffu),
        "r"(smem_addr + 12u),
        "r"(packed3 & 0xffffu));
}

template <typename PScTile>
__device__ inline uint32_t fp4pv_load_mxfp4_scale_word_swizzled(
    PScTile &p_sc_tile,
    int global_row
) {
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, 0);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_tile.data[0])) + dest;
    uint32_t packed;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(packed) : "r"(smem_addr));
    return packed;
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_zero_invalid_causal_groups_mxfp4(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    int global_row,
    int valid_mx_blocks
) {
    uint32_t zero_words[4] = {0u, 0u, 0u, 0u};
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    for (int mx_block = valid_mx_blocks; mx_block < MX_BLOCKS; ++mx_block) {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, zero_words, global_row, mx_block);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, 0u);
    }
}

template <typename PTile>
__device__ inline void fp4pv_zero_invalid_causal_payload_groups_mxfp4(
    PTile &p_fp4_tile,
    int global_row,
    int valid_mx_blocks
) {
    uint32_t zero_words[4] = {0u, 0u, 0u, 0u};
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    for (int mx_block = valid_mx_blocks; mx_block < MX_BLOCKS; ++mx_block) {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, zero_words, global_row, mx_block);
    }
}

template <typename PScTT>
__device__ inline void fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
    PScTT &p_sc_tm,
    int row_offset,
    uint32_t row_tile0,
    uint32_t row_tile1,
    uint32_t row_tile2,
    uint32_t row_tile3
) {
    static_assert(std::is_same_v<typename PScTT::T, fp8e8m0>, "direct P-scale TMEM store expects e8m0 scales");
    static_assert(PScTT::rows == 128 && PScTT::cols == 16, "direct P-scale TMEM store expects a 128x16 scale tile");
    const uint32_t tmem_addr = p_sc_tm.addr + (static_cast<uint32_t>(row_offset) << 16);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::
        "r"(tmem_addr),
        "r"(row_tile0),
        "r"(row_tile1),
        "r"(row_tile2),
        "r"(row_tile3));
}

__device__ __forceinline__ void fp4pv_tmem_store_wait() {
    asm volatile("{tcgen05.wait::st.sync.aligned;}" ::);
}

template <typename ScaleTT>
__device__ __forceinline__ void fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(ScaleTT &scale_tm, uint32_t packed_scale) {
    #pragma unroll
    for (int row = 0; row < 128; row += 32) {
        fp4pv_store_mxfp4_scale_tmem_32x32b_x4(
            scale_tm, row, packed_scale, packed_scale, packed_scale, packed_scale);
    }
}

template <typename C, typename TT, typename ST>
__device__ inline void fp4pv_load_mxnv_scale_async_cluster(TT &dst, const ST &src) {
    static_assert(C::CLUSTER_SIZE == 1 || C::CLUSTER_SIZE == 2, "FP4PV scale staging supports cluster size 1 or 2");
    load_mxnv_scale_async<C::CLUSTER_SIZE>(dst, src);
}

__device__ inline void fp4pv_publish_shared_backing_proxy_only() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <typename C, typename TT, typename ST>
__device__ inline void fp4pv_load_q_scale_chunk_cluster(TT &q_sc_tm, const ST &q_sc_smem, int chunk_idx) {
    auto &q_sc_sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
        reinterpret_cast<uint64_t>(&q_sc_smem.data[0]) + 16 * 32 * chunk_idx);
    fp4pv_load_mxnv_scale_async_cluster<C>(q_sc_tm, q_sc_sm_sub);
}

template <typename C, typename TT, typename ST>
__device__ inline void fp4pv_load_k_scale_chunk_cluster(TT &k_sc_tm, const ST &k_sc_smem, int chunk_idx) {
    auto &k_sc_sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
        reinterpret_cast<uint64_t>(&k_sc_smem.data[0]) + 16 * 32 * chunk_idx);
    fp4pv_load_mxnv_scale_async_cluster<C>(k_sc_tm, k_sc_sm_sub);
}

template <typename C, typename QScTTFull, typename QScTile>
__device__ inline void fp4pv_stage_q_scale_tmem_cluster(QScTTFull &q_sc_tm_full, const QScTile &q_sc_smem) {
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        fp4pv_load_q_scale_chunk_cluster<C>(q_sc_tm, q_sc_smem, chunk);
    }
}

__device__ __forceinline__ float2 fp4pv_apply_backward_dp_delta_scale(
    float2 p_pair,
    bf16_2 dp_pair_bf,
    float2 scale_2,
    float2 bias_2
) {
    const float2 dp_pair = __bfloat1622float2(dp_pair_bf);
    const float2 ds_factor = __ffma2_rn(dp_pair, scale_2, bias_2);
    return {p_pair.x * ds_factor.x, p_pair.y * ds_factor.y};
}

__device__ __forceinline__ float2 fp4pv_apply_backward_dp_delta_scale(
    float2 p_pair,
    float2 dp_pair,
    float2 scale_2,
    float2 bias_2
) {
    const float2 ds_factor = __ffma2_rn(dp_pair, scale_2, bias_2);
    return {p_pair.x * ds_factor.x, p_pair.y * ds_factor.y};
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_cluster_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_ABt(d, a, b, sa, sb, sem);
    } else {
        mm2_ABt(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_cluster_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_ABt(d, a, b, sa, sb, sem);
    } else {
        mma2_ABt(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_cluster_AtB(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_AtB(d, a, b, sa, sb, sem);
    } else {
        mm2_AtB(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_cluster_AtB(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_AtB(d, a, b, sa, sb, sem);
    } else {
        mma2_AtB(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_nosem_cluster_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_ABt(d, a, b, sa, sb);
    } else {
        mm2_ABt(d, a, b, sa, sb);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_nosem_cluster_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_ABt(d, a, b, sa, sb);
    } else {
        mma2_ABt(d, a, b, sa, sb);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mm_unscaled_cluster_ABt(
    D &d, A &a, B &b, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_ABt(d, a, b, sem);
    } else {
        mm2_ABt(d, a, b, sem);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mma_unscaled_cluster_ABt(
    D &d, A &a, B &b, semaphore &sem
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_ABt(d, a, b, sem);
    } else {
        mma2_ABt(d, a, b, sem);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mm_unscaled_nosem_cluster_ABt(
    D &d, A &a, B &b
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_ABt(d, a, b);
    } else {
        mm2_ABt(d, a, b);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ __forceinline__ void fp4pv_mma_unscaled_nosem_cluster_ABt(
    D &d, A &a, B &b
) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_ABt(d, a, b);
    } else {
        mma2_ABt(d, a, b);
    }
}

template <typename Tile>
__device__ __forceinline__ void fp4pv_zero_shared_tile_block(Tile &tile) {
    auto *raw = reinterpret_cast<unsigned char *>(&tile);
    for (int i = threadIdx.x; i < (int)sizeof(Tile); i += blockDim.x) {
        raw[i] = 0;
    }
}

template <typename Tile>
__device__ __forceinline__ void fp4pv_zero_shared_tile_warp(Tile &tile) {
    auto *raw = reinterpret_cast<unsigned char *>(&tile);
    for (int i = warp::laneid(); i < (int)sizeof(Tile); i += WARP_THREADS) {
        raw[i] = 0;
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mm_p_stage_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mm_cluster_ABt<C>(d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mm_cluster_ABt<C>(d, a, b, sa, sb, sem);
    }
}

template <typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_p_stage_ABt(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mma_cluster_ABt<C>(d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mma_cluster_ABt<C>(d, a, b, sa, sb, sem);
    }
}

template <int CHUNK, int ACC, int SCALE_FACTOR_ID, bool DO_COMMIT, typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_p_stage_ABt_k64_chunk_view(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    using T_D = typename D::T;
    using T_AB = typename A::T;
    using T_SAB = typename SA::T;
    static_assert(CHUNK == 0 || CHUNK == 1, "split-K64 FP4 PV only supports the two K64 chunks in a K128 tile");
    static_assert(ACC == 0 || ACC == 1, "tcgen05 accumulator predicate must be 0 or 1");
    static_assert(std::is_same_v<T_AB, fp4e2m1_2>, "split-K64 PV is only for FP4 payloads");
    static_assert(std::is_same_v<T_SAB, fp8e8m0>, "split-K64 PV expects MXFP4 e8m0 scales");
    constexpr int ncta = C::CLUSTER_SIZE;
    constexpr int trans_a = transpose::N;
    constexpr int trans_b = 1 - transpose::T;
    constexpr int M = A::rows * ncta;
    constexpr int N = B::rows * ncta;
    constexpr int K = A::cols * 2;
    constexpr int block_size = 32;
    static_assert(K == 128, "split-K64 FP4 PV expects a logical K128 source tile");
    static_assert(M == D::rows * ncta && ((ncta == 1 && M == 128) || (ncta == 2 && (M == 128 || M == 256))));
    static_assert(N == D::cols && ((ncta == 1 && N % 8 == 0) || (ncta == 2 && N % 16 == 0)));
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);
    constexpr uint32_t idesc =
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, SCALE_FACTOR_ID>();
    detail::tcgen05::template st_st<T_AB, T_SAB, ACC, ncta, block_size>(
        d.addr,
        a_desc.chunk_descriptor(CHUNK),
        b_desc.chunk_descriptor(CHUNK),
        sa.addr,
        sb.addr,
        idesc);
    if constexpr (DO_COMMIT) {
        detail::tcgen05::commit<ncta>(sem);
    }
}

template <typename C, bool CLEAR_ACC, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_mma_p_stage_ABt_split_k64_view(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    fp4pv_mma_p_stage_ABt_k64_chunk_view<0, CLEAR_ACC ? 0 : 1, 0, false, C>(d, a, b, sa, sb, sem);
    fp4pv_mma_p_stage_ABt_k64_chunk_view<1, 1, 2, true, C>(d, a, b, sa, sb, sem);
}

template <int CHUNK, int ACC, int SCALE_FACTOR_ID, bool DO_COMMIT, typename C, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_p_stage_ABt_k64_chunk(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mma_p_stage_ABt_k64_chunk_view<CHUNK, ACC, SCALE_FACTOR_ID, DO_COMMIT, C>(
            d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mma_p_stage_ABt_k64_chunk_view<CHUNK, ACC, SCALE_FACTOR_ID, DO_COMMIT, C>(
            d, a, b, sa, sb, sem);
    }
}

template <typename C, bool CLEAR_ACC, typename D, typename A, typename B, typename SA, typename SB>
__device__ __forceinline__ void fp4pv_p_stage_ABt_split_k64(
    D &d, A &a, B &b, SA &sa, SB &sb, semaphore &sem
) {
    if constexpr (fp4pv_p_swizzle_bytes<C>::value == 128) {
        auto a_view = a.template subtile<C::Mb, C::Nb / 2>(int2{0, 0});
        fp4pv_mma_p_stage_ABt_split_k64_view<C, CLEAR_ACC>(d, a_view, b, sa, sb, sem);
    } else {
        fp4pv_mma_p_stage_ABt_split_k64_view<C, CLEAR_ACC>(d, a, b, sa, sb, sem);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group(
    const float2 *scores_group,
    int quant_mode,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const float2 pair_f = scores_group[jj];
        scores_bf[jj] = __float22bfloat162_rn(pair_f);
        block_amax = fmaxf(block_amax, fmaxf(pair_f.x, pair_f.y));
    }
    e8m0 = fp4pv_float_to_e8m0(block_amax, quant_mode);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0;
        }
        e8m0 = 0;
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_rte(
    const float2 *scores_group,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const float2 pair_f = scores_group[jj];
        scores_bf[jj] = __float22bfloat162_rn(pair_f);
        block_amax = fmaxf(block_amax, fmaxf(pair_f.x, pair_f.y));
    }
    e8m0 = fp4pv_float_to_e8m0_rte(block_amax);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0x77777777u;
        }
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_bf16_amax(
    const float2 *scores_group,
    int quant_mode,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        scores_bf[jj] = __float22bfloat162_rn(scores_group[jj]);
        block_amax = fmaxf(
            block_amax,
            fmaxf(fabsf(__bfloat162float(scores_bf[jj].x)), fabsf(__bfloat162float(scores_bf[jj].y))));
    }
    e8m0 = fp4pv_float_to_e8m0(block_amax, quant_mode);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0x77777777u;
        }
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_bf16_pairs_amax(
    const bf16_2 (&scores_bf)[16],
    int quant_mode,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    uint16_t block_amax_bits = 0;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const uint16_t pair_amax_bits = fp4pv_bf16_pair_max_abs_bits(scores_bf[jj]);
        block_amax_bits = pair_amax_bits > block_amax_bits ? pair_amax_bits : block_amax_bits;
    }
    e8m0 = fp4pv_bf16_abs_bits_to_e8m0(block_amax_bits, quant_mode);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0x77777777u;
        }
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_scaled_rte(
    const float2 *scores_group,
    const float2 p_scale,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const float2 pair_f = scores_group[jj];
        scores_bf[jj] = __float22bfloat162_rn(pair_f);
        block_amax = fmaxf(block_amax, fmaxf(pair_f.x, pair_f.y));
    }
    const float p_scale_scalar = p_scale.x;
    e8m0 = fp4pv_float_to_e8m0_rte(block_amax * p_scale_scalar);
    const float coeff = (6.0f * p_scale_scalar) * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0;
        }
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_fixed_scaled_rte(
    const float2 *scores_group,
    uint32_t (&words)[4]
) {
    __align__(8) bf16_2 scores_bf[16];
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        scores_bf[jj] = __float22bfloat162_rn(scores_group[jj]);
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(
            e03, e47, FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF);
    }
}

__device__ __forceinline__ uint32_t fp4pv_mul_cvt_fp32_to_fp4_8x_fixed_scaled_rte(
    const float2 *scores_group
) {
    const float2 coeff = {
        FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF,
        FP4PV_MXFP4_ONLINE_FIXED_P_QUANT_COEFF
    };
    const auto lo = transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x_with_rn(
        scores_group[0], scores_group[1], coeff, 0);
    const auto hi = transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x_with_rn(
        scores_group[2], scores_group[3], coeff, 0);
    const uint16_t lo_bits = *reinterpret_cast<const uint16_t *>(&lo);
    const uint16_t hi_bits = *reinterpret_cast<const uint16_t *>(&hi);
    return static_cast<uint32_t>(lo_bits) | (static_cast<uint32_t>(hi_bits) << 16);
}

__device__ __forceinline__ void fp4pv_quantize_mxfp4_scores_group_with_amax(
    const float2 *scores_group,
    float block_amax,
    int quant_mode,
    uint32_t (&words)[4],
    uint8_t &e8m0
) {
    __align__(8) bf16_2 scores_bf[16];
    #pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        scores_bf[jj] = __float22bfloat162_rn(scores_group[jj]);
    }
    e8m0 = fp4pv_float_to_e8m0(block_amax, quant_mode);
    const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
    if (e8m0 == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            words[i] = 0;
        }
        e8m0 = 0;
        return;
    }
    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        const int off = pack * 4;
        const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 0]);
        const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&scores_bf[off + 2]);
        words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
    }
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile>
__device__ inline uint8_t fp4pv_pack_scores_one_to_stage_mxfp4_no_scale(
    PTile &p_fp4_tile,
    const float2 (&scores_one)[16],
    int global_row,
    int mx_block,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t words[4];
    uint8_t e8m0 = 0;
    fp4pv_quantize_mxfp4_scores_group(scores_one, quant_mode, words, e8m0);
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
    return e8m0;
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile>
__device__ inline uint8_t fp4pv_pack_scores_one_to_stage_mxfp4_no_scale_with_amax(
    PTile &p_fp4_tile,
    const float2 (&scores_one)[16],
    float block_amax,
    int global_row,
    int mx_block,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t words[4];
    uint8_t e8m0 = 0;
    fp4pv_quantize_mxfp4_scores_group_with_amax(scores_one, block_amax, quant_mode, words, e8m0);
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
    return e8m0;
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group(&scores_reg[mx_block * 16], quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <typename PTile, typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_stage_mxfp4_marker(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t packed_scales = 0u;
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group(&scores_reg[mx_block * 16], quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    return packed_scales;
}

template <typename PTile, typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_stage_mxfp4_marker_bf16_amax(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t packed_scales = 0u;
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    return packed_scales;
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_bf16_amax(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <typename PTile, typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_stage_mxfp4_marker_bf16_amax_and_scan(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    fp4pv_p_scan3d &scan,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t packed_scales = 0u;
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, mx_block);
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    return packed_scales;
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_bf16_amax_and_scan(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    fp4pv_p_scan3d &scan,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        __align__(8) bf16_2 scores_bf_reg[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[mx_block * 16 + jj]);
        }
        store_scores_quarter_to_localcta_scan(scan, scores_bf_reg, mx_block);
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_bf16_pairs_amax(scores_bf_reg, quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_rte(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group_rte(&scores_reg[mx_block * 16], words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    bool STORE_SCALE_TO_SMEM = true,
    typename PTile,
    typename PScTile>
__device__ inline uint32_t fp4pv_pack_scores_to_stage_mxfp4_scaled_rte(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    const float2 p_scale,
    int global_row,
    int packed_col_offset = 0
) {
    uint32_t packed_scales = 0u;
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group_scaled_rte(&scores_reg[mx_block * 16], p_scale, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4_selected<
            SCALAR_P_PAYLOAD_STORE,
            STRIPED_P_PAYLOAD_STORE,
            HALF_STRIPED_P_PAYLOAD_STORE,
            V2_P_PAYLOAD_STORE,
            STAGGERED_P_PAYLOAD_STORE,
            HALF_STAGGERED_P_PAYLOAD_STORE>(
            p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        packed_scales |= static_cast<uint32_t>(e8m0) << (mx_block * 8);
    }
    if constexpr (STORE_SCALE_TO_SMEM) {
        fp4pv_store_mxfp4_scale_word_swizzled(p_sc_tile, global_row, packed_scales);
    }
    return packed_scales;
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_fixed_scaled_rte(
    PTile &p_fp4_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int packed_col_offset = 0
) {
    constexpr int MX_BLOCKS = (PTile::cols / 16 < 4) ? (PTile::cols / 16) : 4;
    #pragma unroll
    for (int mx_block = 0; mx_block < MX_BLOCKS; ++mx_block) {
        uint32_t words[4];
        fp4pv_quantize_mxfp4_scores_group_fixed_scaled_rte(&scores_reg[mx_block * 16], words);
        fp4pv_store_quantized_scores_group_mxfp4_selected<
            SCALAR_P_PAYLOAD_STORE,
            STRIPED_P_PAYLOAD_STORE,
            HALF_STRIPED_P_PAYLOAD_STORE,
            V2_P_PAYLOAD_STORE,
            STAGGERED_P_PAYLOAD_STORE,
            HALF_STAGGERED_P_PAYLOAD_STORE>(
            p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
}

template <typename PScTile>
__device__ inline void fp4pv_fill_mxfp4_scale_stage_word_swizzled(
    PScTile &p_sc_tile,
    int rows,
    uint32_t packed_scales
) {
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        fp4pv_store_mxfp4_scale_word_swizzled(p_sc_tile, row, packed_scales);
    }
}

template <int MX_BLOCK_BEGIN, int MX_BLOCK_END, typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_mxfp4_range(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_reg)[64],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    #pragma unroll
    for (int mx_block = MX_BLOCK_BEGIN; mx_block < MX_BLOCK_END; ++mx_block) {
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group(&scores_reg[mx_block * 16], quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <int MX_BLOCK_BEGIN, int MX_BLOCK_END, typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_half_to_stage_mxfp4_range(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_half)[32],
    int global_row,
    int quant_mode,
    int packed_col_offset = 0
) {
    #pragma unroll
    for (int mx_block = MX_BLOCK_BEGIN; mx_block < MX_BLOCK_END; ++mx_block) {
        const int local_block = mx_block - MX_BLOCK_BEGIN;
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group(&scores_half[local_block * 16], quant_mode, words, e8m0);
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
        fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
    }
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile,
    typename PScTile>
__device__ inline void fp4pv_pack_scores_one_to_stage_mxfp4(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_one)[16],
    int global_row,
    int mx_block,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t words[4];
    uint8_t e8m0 = 0;
    fp4pv_quantize_mxfp4_scores_group(scores_one, quant_mode, words, e8m0);
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
    fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
}

template <
    bool SCALAR_P_PAYLOAD_STORE = false,
    bool STRIPED_P_PAYLOAD_STORE = false,
    bool HALF_STRIPED_P_PAYLOAD_STORE = false,
    bool V2_P_PAYLOAD_STORE = false,
    bool STAGGERED_P_PAYLOAD_STORE = false,
    bool HALF_STAGGERED_P_PAYLOAD_STORE = false,
    typename PTile,
    typename PScTile>
__device__ inline void fp4pv_pack_scores_one_to_stage_mxfp4_with_amax(
    PTile &p_fp4_tile,
    PScTile &p_sc_tile,
    const float2 (&scores_one)[16],
    float block_amax,
    int global_row,
    int mx_block,
    int quant_mode,
    int packed_col_offset = 0
) {
    uint32_t words[4];
    uint8_t e8m0 = 0;
    fp4pv_quantize_mxfp4_scores_group_with_amax(scores_one, block_amax, quant_mode, words, e8m0);
    if constexpr (HALF_STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STAGGERED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_staggered(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (HALF_STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_half_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (STRIPED_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_striped(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (SCALAR_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_scalar(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else if constexpr (V2_P_PAYLOAD_STORE) {
        fp4pv_store_quantized_scores_group_mxfp4_v2(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    } else {
        fp4pv_store_quantized_scores_group_mxfp4(p_fp4_tile, words, global_row, mx_block, packed_col_offset);
    }
    fp4pv_store_mxfp4_scale_byte_swizzled(p_sc_tile, global_row, mx_block, e8m0);
}

template <typename C, bool NEED_P_SCAN, typename VTile>
__device__ inline void load_v_fp4_tile_from_global(
    VTile &v_smem,
    const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int packed_col_offset = 0
) {
    const int packed_seqlen = g.k.rows() / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "Expected fp4 V load width divisible by vector width");
    static_assert(VTile::cols >= C::Nb / 2, "Expected fp4 V tile to hold at least one 128-column packed slice");
    static_assert(VTile::swizzle_bytes >= BYTES_PER_STORE, "Expected V tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.v_fp4[((((b * g.q.depth() + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(VTile::idx(smem_base, {row, packed_col_offset + col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <int PRODUCER_THREADS, typename C, bool NEED_P_SCAN, typename VTile>
__device__ inline void load_v_fp4_tile_from_global_producer_group(
    VTile &v_smem,
    const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int producer_lane,
    int packed_col_offset = 0
) {
    const int packed_seqlen = g.k.rows() / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert((C::Nb / 2) % BYTES_PER_STORE == 0, "Expected fp4 V load width divisible by vector width");
    static_assert(VTile::cols >= C::Nb / 2, "Expected fp4 V tile to hold at least one 128-column packed slice");
    static_assert(VTile::swizzle_bytes >= BYTES_PER_STORE, "Expected V tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = (C::Nb / 2) / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    if constexpr (PRODUCER_THREADS == 64 && C::Dvo == 128 && C::Nb == 128 && C::CLUSTER_SIZE == 2) {
        static_assert(TOTAL_STORES == 4 * PRODUCER_THREADS,
            "Expected four V FP4 vector chunks per producer lane");
        uint4 values[4];
        uint32_t smem_addrs[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int linear = producer_lane + u * PRODUCER_THREADS;
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            values[u] = *reinterpret_cast<const uint4 *>(
                &g.v_fp4[((((b * g.q.depth() + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
            smem_addrs[u] = VTile::idx(smem_base, {row, packed_col_offset + col});
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint4 value = values[u];
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(smem_addrs[u]),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    } else {
        for (int linear = producer_lane; linear < TOTAL_STORES; linear += PRODUCER_THREADS) {
            const int row = linear / STORES_PER_ROW;
            const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.v_fp4[((((b * g.q.depth() + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(VTile::idx(smem_base, {row, packed_col_offset + col})),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    }
}

template <typename C, bool NEED_P_SCAN, typename VScTile>
__device__ inline void load_v_sc_prepared_from_global(
    VScTile &v_sc_smem,
    const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g,
    int b,
    int h,
    int k_iter
) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&v_sc_smem.data[0]);
    static_assert(sizeof(VScTile) % sizeof(uint32_t) == 0, "Expected aligned MXFP4 V scale tile size");
    constexpr int TOTAL_WORDS = sizeof(VScTile) / sizeof(uint32_t);
    constexpr int SCALE_ITERS_PER_TILE = 128 / C::Nb;
    const int words_per_tile = 512 / sizeof(uint32_t);
    const int scale_tile = k_iter / SCALE_ITERS_PER_TILE;
    const int scale_half = k_iter - scale_tile * SCALE_ITERS_PER_TILE;
    const int base_offset = (((((b * g.q.depth() + h) * (C::Dvo / 128) + 0) * (g.k.rows() / 128)) + scale_tile) * 32 * 16);
    if constexpr (C::Nb == 128) {
        constexpr int BYTES_PER_STORE = 16;
        static_assert(sizeof(VScTile) % BYTES_PER_STORE == 0, "Expected aligned MXFP4 V scale tile size");
        constexpr int TOTAL_STORES = sizeof(VScTile) / BYTES_PER_STORE;
        uint32_t smem_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);
        for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
            const int byte_offset = linear * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.v_sc_prepared[base_offset + byte_offset]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(smem_base + byte_offset),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    } else {
        for (int linear = warp::laneid(); linear < TOTAL_WORDS; linear += WARP_THREADS) {
            uint32_t packed = *reinterpret_cast<const uint32_t *>(
                &g.v_sc_prepared[base_offset + linear * sizeof(uint32_t)]);
            if constexpr (C::Nb == 64) {
                packed = scale_half == 0 ? (packed & 0x0000ffffu) : ((packed >> 16) & 0x0000ffffu);
            }
            dst[linear] = packed;
        }
    }
    (void)words_per_tile;
}

template <int PRODUCER_THREADS, typename C, bool NEED_P_SCAN, typename VScTile>
__device__ inline void load_v_sc_prepared_from_global_producer_group(
    VScTile &v_sc_smem,
    const globals_fp4pv_mxfp4_dv<C, NEED_P_SCAN> &g,
    int b,
    int h,
    int k_iter,
    int producer_lane
) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&v_sc_smem.data[0]);
    static_assert(sizeof(VScTile) % sizeof(uint32_t) == 0, "Expected aligned MXFP4 V scale tile size");
    constexpr int TOTAL_WORDS = sizeof(VScTile) / sizeof(uint32_t);
    constexpr int SCALE_ITERS_PER_TILE = 128 / C::Nb;
    const int scale_tile = k_iter / SCALE_ITERS_PER_TILE;
    const int scale_half = k_iter - scale_tile * SCALE_ITERS_PER_TILE;
    const int base_offset = (((((b * g.q.depth() + h) * (C::Dvo / 128) + 0) * (g.k.rows() / 128)) + scale_tile) * 32 * 16);
    if constexpr (C::Nb == 128) {
        constexpr int BYTES_PER_STORE = 16;
        static_assert(sizeof(VScTile) % BYTES_PER_STORE == 0, "Expected aligned MXFP4 V scale tile size");
        constexpr int TOTAL_STORES = sizeof(VScTile) / BYTES_PER_STORE;
        uint32_t smem_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);
        for (int linear = producer_lane; linear < TOTAL_STORES; linear += PRODUCER_THREADS) {
            const int byte_offset = linear * BYTES_PER_STORE;
            const uint4 value = *reinterpret_cast<const uint4 *>(
                &g.v_sc_prepared[base_offset + byte_offset]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :
                         : "r"(smem_base + byte_offset),
                           "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
        }
    } else {
        for (int linear = producer_lane; linear < TOTAL_WORDS; linear += PRODUCER_THREADS) {
            uint32_t packed = *reinterpret_cast<const uint32_t *>(
                &g.v_sc_prepared[base_offset + linear * sizeof(uint32_t)]);
            if constexpr (C::Nb == 64) {
                packed = scale_half == 0 ? (packed & 0x0000ffffu) : ((packed >> 16) & 0x0000ffffu);
            }
            dst[linear] = packed;
        }
    }
}

template <typename C>
struct globals_pv_from_p {
    using p_fp4_tile = typename globals_fp4pv<C>::p_fp4_tile;
    using p_sc_tile = typename globals_fp4pv<C>::p_sc_tile;
    using v_fp4_tile = typename globals_fp4pv<C>::v_fp4_tile;
    using v_sc_tile = typename globals_fp4pv<C>::v_sc_tile;
    using o_tile = typename globals_fp4pv<C>::o_tile;
    using o_gl = typename globals_fp4pv<C>::o_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        add((int)sizeof(o_tile));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc_prepared;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    o_gl o;
    int tile_begin;
    int tiles_to_run;
    int second_tile_mm2;
    int pv_wait_count;
    int early_tile_arrived;
    int batch;
    int heads;
    int seqlen;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / (C::Mb * C::CLUSTER_SIZE)) * C::CLUSTER_SIZE);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_two_tile_pv_tmem_debug {
    using p_fp4_tile = typename globals_fp4pv<C>::p_fp4_tile;
    using p_sc_tile = typename globals_fp4pv<C>::p_sc_tile;
    using v_fp4_tile = typename globals_fp4pv<C>::v_fp4_tile;
    using v_sc_tile = typename globals_fp4pv<C>::v_sc_tile;
    using o_tile = typename globals_fp4pv<C>::o_tile;
    using o_gl = typename globals_fp4pv<C>::o_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(p_fp4_tile));
        add((int)sizeof(p_sc_tile));
        add((int)sizeof(v_fp4_tile));
        add((int)sizeof(v_sc_tile));
        add((int)sizeof(o_tile));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc_prepared;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    o_gl o_snapshot;
    int tiles_to_run;
    int batch;
    int heads;
    int seqlen;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / (C::Mb * C::CLUSTER_SIZE)) * C::CLUSTER_SIZE);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_two_tile_pv_mainpath_debug {
    using p_fp4_tile = typename globals_fp4pv<C>::p_fp4_tile;
    using p_sc_tile = typename globals_fp4pv<C>::p_sc_tile;
    using p_sc_guard_slot = typename globals_fp4pv<C>::p_sc_guard_slot;
    using v_fp4_tile = typename globals_fp4pv<C>::v_fp4_tile;
    using v_sc_tile = typename globals_fp4pv<C>::v_sc_tile;
    using o_tile = typename globals_fp4pv<C>::o_tile;
    using o_gl = typename globals_fp4pv<C>::o_gl;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(v_fp4_tile) * C::LOAD_STAGES);
        add((int)sizeof(v_sc_tile) * C::LOAD_STAGES);
        add((int)sizeof(o_tile));
        add((int)sizeof(p_fp4_tile) * 2);
        add((int)sizeof(p_sc_tile) * 2);
        add((int)sizeof(p_sc_guard_slot));
        return bytes;
    }

public:
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    const uint8_t *p_fp4;
    const uint8_t *p_sc_prepared;
    const uint8_t *v_fp4;
    const uint8_t *v_sc_prepared;
    o_gl o;
    int tile_begin;
    int tiles_to_run;
    int second_tile_mm2;
    int pv_wait_count;
    int early_tile_arrived;
    int batch;
    int heads;
    int seqlen;

    __host__ __inline__ dim3 grid() {
        return dim3(batch * heads * (seqlen / (C::Mb * C::CLUSTER_SIZE)) * C::CLUSTER_SIZE);
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <typename C>
struct globals_two_tile_correction_replay_debug {
    const float *pre_correction;
    const float *correction;
    const float *row_sum;
    bf16 *out;
    int batch;
    int heads;
    int seqlen;
};

template <typename C, typename PTile, typename G>
__device__ inline void load_p_fp4_tile_from_global_allthreads(
    PTile &p_smem,
    const G &g,
    int b,
    int h,
    int m_tile,
    int k_iter
) {
    const int packed_seqlen = g.seqlen / 2;
    const int row_base = m_tile * C::Mb;
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&p_smem.data[0]);
    constexpr int BYTES_PER_STORE = 4;
    static_assert(PTile::cols % BYTES_PER_STORE == 0, "Expected fp4 P tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = PTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = PTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint32_t value = *reinterpret_cast<const uint32_t *>(
            &g.p_fp4[((((b * g.heads + h) * g.seqlen) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.b32 [%0], %1;"
                     :
                     : "r"(PTile::idx(smem_base, {row, col})), "r"(value));
    }
}

template <typename C, typename PScTile, typename G>
__device__ inline void load_p_sc_prepared_from_global_allthreads(
    PScTile &p_sc_smem,
    const G &g,
    int b,
    int h,
    int m_tile,
    int k_iter
) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&p_sc_smem.data[0]);
    const int scale_k_base = k_iter * 2;
    static_assert(sizeof(PScTile) % sizeof(uint32_t) == 0, "Expected aligned P scale tile size");
    constexpr int WORDS_PER_SLAB = 512 / sizeof(uint32_t);
    constexpr int TOTAL_WORDS = sizeof(PScTile) / sizeof(uint32_t);
    for (int linear = threadIdx.x; linear < TOTAL_WORDS; linear += blockDim.x) {
        const int slab = linear / WORDS_PER_SLAB;
        const int word = linear % WORDS_PER_SLAB;
        dst[linear] = *reinterpret_cast<const uint32_t *>(
            &g.p_sc_prepared[((((b * g.heads + h) * (g.seqlen / 128) + m_tile) * (g.seqlen / 64)) + scale_k_base + slab) * 512 + word * sizeof(uint32_t)]);
    }
}

template <typename C, typename VTile, typename G>
__device__ inline void load_v_fp4_tile_from_global_allthreads(
    VTile &v_smem,
    const G &g,
    int b,
    int h,
    int k_iter,
    int cta_rank,
    int dq_half_override = -1
) {
    const int packed_seqlen = g.seqlen / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert(VTile::cols % BYTES_PER_STORE == 0, "Expected fp4 V tile cols divisible by vector width");
    static_assert(VTile::swizzle_bytes >= BYTES_PER_STORE, "Expected V tile swizzle to support vector stores");
    constexpr int STORES_PER_ROW = VTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    for (int linear = threadIdx.x; linear < TOTAL_STORES; linear += blockDim.x) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.v_fp4[((((b * g.heads + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(VTile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename VScTile, typename G>
__device__ inline void load_v_sc_prepared_from_global_allthreads(
    VScTile &v_sc_smem,
    const G &g,
    int b,
    int h,
    int k_iter
) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&v_sc_smem.data[0]);
    const int scale_k_base = k_iter * 2;
    static_assert(sizeof(VScTile) % sizeof(uint32_t) == 0, "Expected aligned V scale tile size");
    constexpr int WORDS_PER_SLAB = 512 / sizeof(uint32_t);
    constexpr int TOTAL_WORDS = sizeof(VScTile) / sizeof(uint32_t);
    for (int linear = threadIdx.x; linear < TOTAL_WORDS; linear += blockDim.x) {
        const int slab = linear / WORDS_PER_SLAB;
        const int word = linear % WORDS_PER_SLAB;
        dst[linear] = *reinterpret_cast<const uint32_t *>(
            &g.v_sc_prepared[((((b * g.heads + h) * (C::Dvo / 128) + 0) * (g.seqlen / 64)) + scale_k_base + slab) * 512 + word * sizeof(uint32_t)]);
    }
}

template <typename C, typename Tile, typename G>
__device__ inline void load_live_dp_dout_row_fp4_tile_warp(
    Tile &dst,
    const G &g,
    int b,
    int h,
    int m_tile
) {
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int packed_dvo = C::Dvo / 2;
    const int row_base = m_tile * C::Mb;
    uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert(Tile::cols % BYTES_PER_STORE == 0, "Expected dO row FP4 tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.backward_dout_row_fp4[
                ((((static_cast<int64_t>(b) * heads + h) * seqlen) + row_base + row) * packed_dvo) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(Tile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename Tile, typename G>
__device__ inline void load_live_dp_v_row_fp4_tile_warp(
    Tile &dst,
    const G &g,
    int b,
    int h,
    int k_iter,
    int cta_rank
) {
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int packed_dvo = C::Dvo / 2;
    const int row_base = k_iter * C::Nb + cta_rank * (C::Nb / C::CLUSTER_SIZE);
    uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int BYTES_PER_STORE = 16;
    static_assert(Tile::cols % BYTES_PER_STORE == 0, "Expected V row FP4 tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = Tile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = Tile::rows * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint4 value = *reinterpret_cast<const uint4 *>(
            &g.backward_v_row_fp4[
                ((((static_cast<int64_t>(b) * heads + h) * seqlen) + row_base + row) * packed_dvo) + col]);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(Tile::idx(smem_base, {row, col})),
                       "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
}

template <typename C, typename ScTile, typename G>
__device__ inline void load_live_dp_dout_row_sc_tile_warp(
    ScTile &dst,
    const G &g,
    int b,
    int h,
    int m_tile
) {
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int row_tiles = seqlen / C::Mb;
    constexpr int TOTAL_BYTES = sizeof(ScTile);
    static_assert(TOTAL_BYTES == 32 * 16, "Expected dO row scale tile to be 32x16 bytes");
    uint32_t smem_base = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(dst.data[0]));
    const int64_t base =
        (((((static_cast<int64_t>(b) * heads + h) * row_tiles + m_tile) * (C::Dvo / 128)) + 0) * 32 * 16);
    // Byte staging is intentional here: wider stores have produced stale scale
    // bytes in clustered live-dP scale loads on multi-head launches.
    for (int byte_offset = warp::laneid(); byte_offset < TOTAL_BYTES; byte_offset += WARP_THREADS) {
        const uint32_t value = static_cast<uint32_t>(g.backward_dout_row_sc[base + byte_offset]);
        asm volatile("st.shared.b8 [%0], %1;" :: "r"(smem_base + byte_offset), "r"(value));
        if (g.live_dp_debug_mode == 17 && g.backward_dp_bf16 != nullptr) {
            uint32_t check;
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(check) : "r"(smem_base + byte_offset));
            const int out_row = 15 + byte_offset / seqlen;
            const int out_col = byte_offset - (out_row - 15) * seqlen;
            const int64_t debug_row_base =
                (((static_cast<int64_t>(b) * heads + h) * seqlen + m_tile * C::Mb) * seqlen);
            g.backward_dp_bf16[debug_row_base + out_row * seqlen + out_col] =
                __float2bfloat16(static_cast<float>(check));
        }
    }
}

template <typename C, typename ScTile, typename G>
__device__ inline void load_live_dp_v_row_sc_tile_warp(
    ScTile &dst,
    const G &g,
    int b,
    int h,
    int k_iter
) {
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int row_tiles = seqlen / C::Mb;
    constexpr int TOTAL_BYTES = sizeof(ScTile);
    static_assert(TOTAL_BYTES == 32 * 16, "Expected V row scale tile to be 32x16 bytes");
    uint32_t smem_base = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(dst.data[0]));
    const int64_t base =
        (((((static_cast<int64_t>(b) * heads + h) * row_tiles + k_iter) * (C::Dvo / 128)) + 0) * 32 * 16);
    // See the dO scale loader above; this path must preserve byte-granular
    // shared staging for the live-dP scale TMEM copy.
    for (int byte_offset = warp::laneid(); byte_offset < TOTAL_BYTES; byte_offset += WARP_THREADS) {
        const uint32_t value = static_cast<uint32_t>(g.backward_v_row_sc[base + byte_offset]);
        asm volatile("st.shared.b8 [%0], %1;" :: "r"(smem_base + byte_offset), "r"(value));
        if (g.live_dp_debug_mode == 17 && g.backward_dp_bf16 != nullptr) {
            uint32_t check;
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(check) : "r"(smem_base + byte_offset));
            const int out_row = 11 + byte_offset / seqlen;
            const int out_col = byte_offset - (out_row - 11) * seqlen;
            const int64_t debug_row_base = (((static_cast<int64_t>(b) * heads + h) * seqlen) * seqlen);
            g.backward_dp_bf16[debug_row_base + out_row * seqlen + out_col] =
                __float2bfloat16(static_cast<float>(check));
        }
    }
}

template <typename C, typename G>
__device__ inline void store_live_dp_source_debug_warp(
    const G &g,
    int b,
    int h,
    int m_tile,
    int k_iter,
    int cta_rank
) {
    if (g.backward_dp_bf16 == nullptr) {
        return;
    }
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int packed_dvo = C::Dvo / 2;
    const int row_tiles = seqlen / C::Mb;
    const int dout_row_base = m_tile * C::Mb;
    const int v_row_base = k_iter * C::Nb + cta_rank * (C::Nb / C::CLUSTER_SIZE);
    if (dout_row_base + 7 >= seqlen) {
        return;
    }
    const int64_t debug_row_base =
        (((static_cast<int64_t>(b) * heads + h) * seqlen) + dout_row_base) * seqlen;
    const int64_t dout_payload_base =
        ((((static_cast<int64_t>(b) * heads + h) * seqlen) + dout_row_base) * packed_dvo);
    const int64_t v_payload_base =
        ((((static_cast<int64_t>(b) * heads + h) * seqlen) + v_row_base) * packed_dvo);
    const int64_t dout_scale_base =
        (((((static_cast<int64_t>(b) * heads + h) * row_tiles + m_tile) * (C::Dvo / 128)) + 0) * 32 * 16);
    const int64_t v_scale_base =
        (((((static_cast<int64_t>(b) * heads + h) * row_tiles + k_iter) * (C::Dvo / 128)) + 0) * 32 * 16);
    for (int col = warp::laneid(); col < 64; col += WARP_THREADS) {
        g.backward_dp_bf16[debug_row_base + col] =
            __float2bfloat16(static_cast<float>(g.backward_dout_row_fp4[dout_payload_base + col]));
        g.backward_dp_bf16[debug_row_base + seqlen + col] =
            __float2bfloat16(static_cast<float>(g.backward_v_row_fp4[v_payload_base + col]));
        g.backward_dp_bf16[debug_row_base + 2 * seqlen + col] =
            __float2bfloat16(static_cast<float>(g.backward_dout_row_sc[dout_scale_base + col]));
        g.backward_dp_bf16[debug_row_base + 3 * seqlen + col] =
            __float2bfloat16(static_cast<float>(g.backward_v_row_sc[v_scale_base + col]));
    }
}

template <typename C, typename DTile, typename VTile, typename ScTile, typename G>
__device__ inline void store_live_dp_shared_debug_warp(
    const G &g,
    const DTile &dout_smem,
    const ScTile &dout_sc_smem,
    const VTile &v_smem,
    const ScTile &v_sc_smem,
    int b,
    int h,
    int m_tile
) {
    if (g.backward_dp_bf16 == nullptr) {
        return;
    }
    const int heads = g.q.depth();
    const int seqlen = g.q.rows();
    const int dout_row_base = m_tile * C::Mb;
    if (dout_row_base + 3 >= seqlen) {
        return;
    }
    const int64_t debug_row_base =
        (((static_cast<int64_t>(b) * heads + h) * seqlen) + dout_row_base) * seqlen;
    const uint32_t dout_base = __cvta_generic_to_shared(&dout_smem.data[0]);
    const uint32_t v_base = __cvta_generic_to_shared(&v_smem.data[0]);
    const uint32_t dout_sc_base = __cvta_generic_to_shared(&dout_sc_smem.data[0]);
    const uint32_t v_sc_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);
    for (int col = warp::laneid(); col < 64; col += WARP_THREADS) {
        uint32_t dout_byte;
        uint32_t v_byte;
        uint32_t dout_sc_byte;
        uint32_t v_sc_byte;
        asm volatile("ld.shared.u8 %0, [%1];"
                     : "=r"(dout_byte)
                     : "r"(DTile::idx(dout_base, {0, col})));
        asm volatile("ld.shared.u8 %0, [%1];"
                     : "=r"(v_byte)
                     : "r"(VTile::idx(v_base, {0, col})));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(dout_sc_byte) : "r"(dout_sc_base + col));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v_sc_byte) : "r"(v_sc_base + col));
        g.backward_dp_bf16[debug_row_base + col] =
            __float2bfloat16(static_cast<float>(dout_byte));
        g.backward_dp_bf16[debug_row_base + seqlen + col] =
            __float2bfloat16(static_cast<float>(v_byte));
        g.backward_dp_bf16[debug_row_base + 2 * seqlen + col] =
            __float2bfloat16(static_cast<float>(dout_sc_byte));
        g.backward_dp_bf16[debug_row_base + 3 * seqlen + col] =
            __float2bfloat16(static_cast<float>(v_sc_byte));
        uint32_t dout_phys_byte;
        uint32_t v_phys_byte;
        uint32_t dout_sc_phys_byte;
        uint32_t v_sc_phys_byte;
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(dout_phys_byte) : "r"(dout_base + col));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v_phys_byte) : "r"(v_base + col));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(dout_sc_phys_byte) : "r"(dout_sc_base + col));
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v_sc_phys_byte) : "r"(v_sc_base + col));
        g.backward_dp_bf16[debug_row_base + 4 * seqlen + col] =
            __float2bfloat16(static_cast<float>(dout_phys_byte));
        g.backward_dp_bf16[debug_row_base + 5 * seqlen + col] =
            __float2bfloat16(static_cast<float>(v_phys_byte));
        g.backward_dp_bf16[debug_row_base + 6 * seqlen + col] =
            __float2bfloat16(static_cast<float>(dout_sc_phys_byte));
        g.backward_dp_bf16[debug_row_base + 7 * seqlen + col] =
            __float2bfloat16(static_cast<float>(v_sc_phys_byte));
    }
}

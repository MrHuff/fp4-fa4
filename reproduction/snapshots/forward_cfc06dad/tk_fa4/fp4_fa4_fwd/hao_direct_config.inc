#pragma once

// Direct TK translation of the HAO SM100 forward topology.  Keep arithmetic
// knobs separate from the ownership protocol so PV formats can share one
// scheduler and barrier lifecycle.
struct hao_direct_arithmetic_base
    : public config_fp4pv<128, 128, 128, 128, 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

#ifndef TK_HAO_DIRECT_SEQ_LEN
#define TK_HAO_DIRECT_SEQ_LEN 4096
#endif
#ifndef TK_HAO_DIRECT_HEADS
#define TK_HAO_DIRECT_HEADS 24
#endif
#ifndef TK_HAO_DIRECT_BATCH
#define TK_HAO_DIRECT_BATCH 1
#endif
#ifndef TK_HAO_DIRECT_QK_SCALE_MODE
// 0: NVFP4 E4M3 scale per 16 values
// 1: MXFP4 E8M0 scale per 32 values
#define TK_HAO_DIRECT_QK_SCALE_MODE 0
#endif
#ifndef TK_HAO_DIRECT_EX2_EMU_MASK
// HAO's SM100 FP8-PV default is 0x3198 for the first three x32
// fragments and native MUFU for the last fragment.
#define TK_HAO_DIRECT_EX2_EMU_MASK 0x3198u
#endif
#ifndef TK_HAO_DIRECT_RETAIN_Q2
#define TK_HAO_DIRECT_RETAIN_Q2 1
#endif
#ifndef TK_HAO_DIRECT_RETAIN_Q0
#define TK_HAO_DIRECT_RETAIN_Q0 0
#endif
#ifndef TK_HAO_DIRECT_SOFTMAX_REGS
#define TK_HAO_DIRECT_SOFTMAX_REGS 176
#endif
#ifndef TK_HAO_DIRECT_CORRECTION_REGS
#define TK_HAO_DIRECT_CORRECTION_REGS 80
#endif
#ifndef TK_HAO_DIRECT_PRODUCER_REGS
#define TK_HAO_DIRECT_PRODUCER_REGS 80
#endif
#ifndef TK_HAO_DIRECT_WAIT_HINT
// 0 keeps TK's generic spin wait. 10000000 matches the suspend-time hint
// emitted by HAO's CUTE DSL pipeline on SM100.
#define TK_HAO_DIRECT_WAIT_HINT 10000000
#endif
#ifndef TK_HAO_DIRECT_FP8PV_FIXED_P_CEILING
// Diagnostic only: preserve QK, score-bank ownership, P publication, and PV,
// but replace score processing with an E4M3 P=1 payload.
#define TK_HAO_DIRECT_FP8PV_FIXED_P_CEILING 0
#endif
#ifndef TK_HAO_DIRECT_FP8PV_SCORE_PACK_CEILING
// Diagnostic only: read each real score quarter once and convert it directly
// to E4M3 P, skipping max, exp2, denominator, and online normalization.
#define TK_HAO_DIRECT_FP8PV_SCORE_PACK_CEILING 0
#endif
#ifndef TK_HAO_DIRECT_FP8PV_P_ILP_PIPELINE
// Keep two P-construction groups in flight while preserving FP8 arithmetic.
#define TK_HAO_DIRECT_FP8PV_P_ILP_PIPELINE 0
#endif
#ifndef TK_HAO_DIRECT_FP8PV_EARLY_Q3
// Start the final score-quarter load before its normal serial position. With
// the two-chunk pipeline this uses two x16 loads around the Q2 publication.
#define TK_HAO_DIRECT_FP8PV_EARLY_Q3 0
#endif
#ifndef TK_HAO_DIRECT_FP8PV_TWO_CHUNK_PIPELINE
// Publish Q0/Q1 first, build Q2 while their K32 PV commands execute, then
// publish Q2 and Q3 independently.
#define TK_HAO_DIRECT_FP8PV_TWO_CHUNK_PIPELINE 0
#endif
#ifndef TK_HAO_DIRECT_FP8PV_SHIFTLESS_MODE
// Experimental matched-policy FP8 control. Consume each score quarter once
// and carry absolute probabilities in E4M3. Modes 1/2 use the full-FP4
// cubic/affine mix; modes 3/4 retain the cubic for FP8's unscaled range.
// Odd modes use the exact approximate-P sum; even modes sample the four
// native EX2 pairs as the selected full-FP4 policy does.
#define TK_HAO_DIRECT_FP8PV_SHIFTLESS_MODE 0
#endif
#ifndef TK_HAO_DIRECT_ROUTE_NAME
#if TK_HAO_DIRECT_QK_SCALE_MODE == 0
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_nvfp4_fp8pv"
#else
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_mxfp4_fp8pv"
#endif
#endif

struct config_fp4pv_hao_direct_fp8pv
    : public hao_direct_arithmetic_base {
    static constexpr bool HAO_DIRECT_PIPELINE = true;
    static constexpr bool HAO_DIRECT_NVFP4_QK =
        TK_HAO_DIRECT_QK_SCALE_MODE == 0;
    static constexpr bool HAO_DIRECT_MXFP4_QK =
        TK_HAO_DIRECT_QK_SCALE_MODE == 1;
    static constexpr int QK_SCALE_ROWS =
        HAO_DIRECT_NVFP4_QK ? 2 : 1;
    static constexpr bool HAO_DIRECT_FP8_PV = true;
    static constexpr bool HAO_DIRECT_FIXED_P_CEILING =
        TK_HAO_DIRECT_FP8PV_FIXED_P_CEILING;
    static constexpr bool HAO_DIRECT_SCORE_PACK_CEILING =
        TK_HAO_DIRECT_FP8PV_SCORE_PACK_CEILING;
    static constexpr bool HAO_DIRECT_P_ILP_PIPELINE =
        TK_HAO_DIRECT_FP8PV_P_ILP_PIPELINE;
    static constexpr bool HAO_DIRECT_EARLY_Q3 =
        TK_HAO_DIRECT_FP8PV_EARLY_Q3;
    static constexpr bool HAO_DIRECT_TWO_CHUNK_PIPELINE =
        TK_HAO_DIRECT_FP8PV_TWO_CHUNK_PIPELINE;
    static constexpr int HAO_DIRECT_SHIFTLESS_MODE =
        TK_HAO_DIRECT_FP8PV_SHIFTLESS_MODE;
    static constexpr bool HAO_DIRECT_SHIFTLESS =
        HAO_DIRECT_SHIFTLESS_MODE != 0;
    static constexpr bool HAO_DIRECT_NONCAUSAL = true;
    static constexpr bool UPSTREAM_NATIVE_NONCAUSAL =
        HAO_DIRECT_NONCAUSAL;
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = 16;
    static constexpr int NUM_THREADS = 512;
    static constexpr int KV_STAGES = 4;
    static constexpr int LOAD_STAGES = KV_STAGES;
    static constexpr int K_SCALE_SMEM_SLOTS = KV_STAGES;
    static constexpr int QUERY_STAGES = 2;
    static constexpr int TMEM_COLUMNS = 512;
    static constexpr int P_FIRST_CHUNKS =
        HAO_DIRECT_TWO_CHUNK_PIPELINE ? 2 : 3;
    // nvcc/ptxas allocates this monolithic kernel against the lowest
    // setmaxnreg branch.  The CUTE-DSL 192/80/48 split therefore spills the
    // score readers in TK.  176/80/80 consumes the same 65,536-register CTA
    // budget while retaining the proven spill-free score-reader floor.
    static constexpr int SOFTMAX_REGS = TK_HAO_DIRECT_SOFTMAX_REGS;
    static constexpr int CORRECTION_REGS =
        TK_HAO_DIRECT_CORRECTION_REGS;
    static constexpr int PRODUCER_REGS = TK_HAO_DIRECT_PRODUCER_REGS;
    // Match HAO's SM100 FP8-PV cadence: six of sixteen pairs in the first
    // three x32 fragments use packed ALU emulation; the final fragment stays
    // on native MUFU.
    static constexpr int ONLINE_STAGE2_EX2_EMU_PERIOD = 16;
    static constexpr uint32_t ONLINE_STAGE2_EX2_EMU_MASK =
        TK_HAO_DIRECT_EX2_EMU_MASK;
    static constexpr int N_TILES = TK_HAO_DIRECT_SEQ_LEN / 128;
    static constexpr int HEADS = TK_HAO_DIRECT_HEADS;
    static constexpr int BATCH = TK_HAO_DIRECT_BATCH;
    static constexpr int LOGICAL_JOBS =
        BATCH * HEADS * N_TILES / QUERY_STAGES;
    static constexpr int PHYSICAL_GRID_CAP = NUM_SM;
};

using hao_direct_config = config_fp4pv_hao_direct_fp8pv;

static_assert(hao_direct_config::HAO_DIRECT_PIPELINE);
static_assert(
    hao_direct_config::HAO_DIRECT_NVFP4_QK !=
    hao_direct_config::HAO_DIRECT_MXFP4_QK);
static_assert(hao_direct_config::ONLINE_FULL_K_TILE);
static_assert(hao_direct_config::CLUSTER_SIZE == 1);
static_assert(hao_direct_config::Dqk == 128);
static_assert(hao_direct_config::Dvo == 128);
static_assert(hao_direct_config::N_TILES > 0);
static_assert(
    hao_direct_config::N_TILES % hao_direct_config::QUERY_STAGES == 0);
static_assert(hao_direct_config::QK_SCALE_CHUNKS == 2);
static_assert(
    hao_direct_config::P_FIRST_CHUNKS == 2 ||
    hao_direct_config::P_FIRST_CHUNKS == 3);
static_assert(hao_direct_config::TMEM_COLUMNS == 512);
static_assert(
    TK_HAO_DIRECT_WAIT_HINT == 0 ||
    TK_HAO_DIRECT_WAIT_HINT == 10000000);
static_assert(
    TK_HAO_DIRECT_FP8PV_FIXED_P_CEILING == 0 ||
    TK_HAO_DIRECT_FP8PV_FIXED_P_CEILING == 1);
static_assert(
    TK_HAO_DIRECT_FP8PV_SCORE_PACK_CEILING == 0 ||
    TK_HAO_DIRECT_FP8PV_SCORE_PACK_CEILING == 1);
static_assert(
    TK_HAO_DIRECT_FP8PV_P_ILP_PIPELINE == 0 ||
    TK_HAO_DIRECT_FP8PV_P_ILP_PIPELINE == 1);
static_assert(
    TK_HAO_DIRECT_FP8PV_EARLY_Q3 == 0 ||
    TK_HAO_DIRECT_FP8PV_EARLY_Q3 == 1);
static_assert(
    TK_HAO_DIRECT_FP8PV_TWO_CHUNK_PIPELINE == 0 ||
    TK_HAO_DIRECT_FP8PV_TWO_CHUNK_PIPELINE == 1);
static_assert(
    TK_HAO_DIRECT_FP8PV_SHIFTLESS_MODE >= 0 &&
    TK_HAO_DIRECT_FP8PV_SHIFTLESS_MODE <= 4);
static_assert(
    !hao_direct_config::HAO_DIRECT_TWO_CHUNK_PIPELINE ||
    (!hao_direct_config::HAO_DIRECT_FIXED_P_CEILING &&
     !hao_direct_config::HAO_DIRECT_SCORE_PACK_CEILING));
static_assert(
    !hao_direct_config::HAO_DIRECT_SHIFTLESS ||
    (!hao_direct_config::HAO_DIRECT_FIXED_P_CEILING &&
     !hao_direct_config::HAO_DIRECT_SCORE_PACK_CEILING &&
     !hao_direct_config::HAO_DIRECT_TWO_CHUNK_PIPELINE));
static_assert(
    !hao_direct_config::HAO_DIRECT_FIXED_P_CEILING ||
    !hao_direct_config::HAO_DIRECT_SCORE_PACK_CEILING);
static_assert(
    hao_direct_config::SOFTMAX_REGS * 2 +
        hao_direct_config::CORRECTION_REGS +
        hao_direct_config::PRODUCER_REGS ==
    512);

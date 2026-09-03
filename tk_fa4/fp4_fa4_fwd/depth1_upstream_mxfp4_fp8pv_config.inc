#pragma once

// Isolated native-protocol TK topology port.  This first milestone keeps the
// HAO S0/S1/O0/O1 ownership and dense E4M3 P/V, but deliberately launches the
// full logical grid.  It is correctness-only until static persistence is
// implemented as a separate lifecycle delta.
using upstream_mxfp4_fp8pv_arithmetic_base =
    config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<
        16, 128, 128, 128, 128, 192, 80, 64, 1>;

#ifndef TK_HAO_SEQ_LEN
#define TK_HAO_SEQ_LEN 32768
#endif
#ifndef TK_HAO_HEADS
#define TK_HAO_HEADS 24
#endif
#ifndef TK_HAO_BATCH
#define TK_HAO_BATCH 1
#endif
#ifndef TK_HAO_ROUTE_NAME
#define TK_HAO_ROUTE_NAME "real_fwd_tk_hao_native_b1s32768h24d128_fullgrid"
#endif
#ifndef TK_HAO_DIRECT_CAUSAL_PAIR_QUARTER_MASK
#define TK_HAO_DIRECT_CAUSAL_PAIR_QUARTER_MASK 0
#endif
struct config_fp4pv_tk_hao_native_b1s32768h24d128_fullgrid
    : public upstream_mxfp4_fp8pv_arithmetic_base {
    static constexpr bool UPSTREAM_OWNED_TWO_STAGE = true;
    static constexpr bool UPSTREAM_DENSE_FP8_PV = true;
    static constexpr bool UPSTREAM_NATIVE_NONCAUSAL = true;
    static constexpr bool HAO_DIRECT_CAUSAL = false;
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = 16;
    static constexpr int NUM_THREADS = 512;
    static constexpr int KV_STAGES = 4;
    static constexpr int LOAD_STAGES = KV_STAGES;
    static constexpr int K_SCALE_SMEM_SLOTS = KV_STAGES;
    static constexpr int QUERY_STAGES = 2;
    static constexpr int TMEM_COLUMNS = 512;
    static constexpr int N_TILES = TK_HAO_SEQ_LEN / 128;
    static constexpr int HEADS = TK_HAO_HEADS;
    static constexpr int BATCH = TK_HAO_BATCH;
    static constexpr int LOGICAL_JOBS = BATCH * HEADS * N_TILES / QUERY_STAGES;
    static constexpr bool ONLINE_PERSISTENT_OUTER_TASK_LOOP = false;
};

using tk_hao_native_port_config =
    config_fp4pv_tk_hao_native_b1s32768h24d128_fullgrid;

static_assert(tk_hao_native_port_config::ONLINE_FULL_K_TILE);
static_assert(tk_hao_native_port_config::N_TILES % tk_hao_native_port_config::QUERY_STAGES == 0);
static_assert(tk_hao_native_port_config::CLUSTER_SIZE == 1);
static_assert(tk_hao_native_port_config::Dqk == 128);
static_assert(tk_hao_native_port_config::Dvo == 128);
static_assert(tk_hao_native_port_config::TOTAL_WGS == 4);
static_assert(tk_hao_native_port_config::NUM_THREADS == 512);
static_assert(tk_hao_native_port_config::LOAD_STAGES == 4);
static_assert(
    tk_hao_native_port_config::LOAD_STAGES ==
    tk_hao_native_port_config::K_SCALE_SMEM_SLOTS);
static_assert(tk_hao_native_port_config::UPSTREAM_DENSE_FP8_PV);
static_assert(tk_hao_native_port_config::UPSTREAM_NATIVE_NONCAUSAL);

enum class upstream_kv_kind : int { K = 0, V = 1 };

template <typename C>
constexpr int upstream_kv_item(int tile, upstream_kv_kind kind) {
    return 2 * tile + static_cast<int>(kind);
}

template <typename C>
constexpr int upstream_kv_slot(int tile, upstream_kv_kind kind) {
    return upstream_kv_item<C>(tile, kind) % C::KV_STAGES;
}

template <typename C>
constexpr int upstream_kv_phase(int tile, upstream_kv_kind kind) {
    return (upstream_kv_item<C>(tile, kind) / C::KV_STAGES) & 1;
}

template <typename C, int Tile, upstream_kv_kind Kind>
constexpr bool upstream_kv_generation_matches() {
    constexpr int item = 2 * Tile + static_cast<int>(Kind);
    return upstream_kv_slot<C>(Tile, Kind) == item % C::KV_STAGES &&
           upstream_kv_phase<C>(Tile, Kind) ==
               (item / C::KV_STAGES) % 2;
}

// Prologue, first wraps, and the exact S4096 causal tail.  The one helper is
// the sole payload/scale source of both slot and generation.
#define UPSTREAM_ASSERT_KV_GENERATION(Tile)                                  \
    static_assert(upstream_kv_generation_matches<                            \
        tk_hao_native_port_config, Tile, upstream_kv_kind::K>());             \
    static_assert(upstream_kv_generation_matches<                            \
        tk_hao_native_port_config, Tile, upstream_kv_kind::V>())
UPSTREAM_ASSERT_KV_GENERATION(0);
UPSTREAM_ASSERT_KV_GENERATION(1);
UPSTREAM_ASSERT_KV_GENERATION(2);
UPSTREAM_ASSERT_KV_GENERATION(3);
UPSTREAM_ASSERT_KV_GENERATION(254);
UPSTREAM_ASSERT_KV_GENERATION(255);
#undef UPSTREAM_ASSERT_KV_GENERATION

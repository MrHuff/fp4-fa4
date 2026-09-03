#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include "../../TK_quantisation/nvfp4_CTA_local_v1/fused_localcta_quantize.cuh"

using namespace kittens;

#ifndef TK_FA4_NATIVE_GROUP2_RAW_QK_DIAG_BUILD
#define TK_FA4_NATIVE_GROUP2_RAW_QK_DIAG_BUILD 0
#endif

#ifndef TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
#define TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD 0
#endif

#ifndef TK_FA4_SINGLE_CTA_FP4QK_BF16PV_ONLY_BUILD
#define TK_FA4_SINGLE_CTA_FP4QK_BF16PV_ONLY_BUILD 0
#endif

#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
// Bounded A-vs-B1 selected-row attribution.  Unlike the older raw-QK
// diagnostic below, this records compact summaries for all 32 causal blocks
// and permits a full 128-value dump for only one runtime-selected block.
// The instrumented A/B1 child specializations are route-local; the v9 control
// specialization in the same artifact does not instantiate these calls.
constexpr int FP4PV_NATIVE_GROUP2_SELECTED_ROW_MAX_ITERS = 32;
constexpr int FP4PV_NATIVE_GROUP2_SELECTED_ROW_RECORD_WORDS = 40;
constexpr int FP4PV_NATIVE_GROUP2_SELECTED_ROW_HEADER_WORDS = 32;
constexpr int FP4PV_NATIVE_GROUP2_SELECTED_ROW_RAW_WORDS = 128;
constexpr int FP4PV_NATIVE_GROUP2_SELECTED_ROW_TRACE_WORDS =
    FP4PV_NATIVE_GROUP2_SELECTED_ROW_HEADER_WORDS +
    FP4PV_NATIVE_GROUP2_SELECTED_ROW_MAX_ITERS *
        FP4PV_NATIVE_GROUP2_SELECTED_ROW_RECORD_WORDS +
    FP4PV_NATIVE_GROUP2_SELECTED_ROW_RAW_WORDS;

__device__ unsigned int fp4pv_native_group2_selected_row_trace[
    FP4PV_NATIVE_GROUP2_SELECTED_ROW_TRACE_WORDS];
// enabled, batch, head, absolute query row, target raw-dump idx, route kind
__device__ int fp4pv_native_group2_selected_row_trace_control[6];

__device__ __forceinline__ bool
fp4pv_native_group2_selected_row_trace_matches(
    int route_kind, int3 t_coord, int m_tile, int idx) {
    if (fp4pv_native_group2_selected_row_trace_control[0] == 0 ||
        route_kind != fp4pv_native_group2_selected_row_trace_control[5] ||
        idx < 0 || idx >= FP4PV_NATIVE_GROUP2_SELECTED_ROW_MAX_ITERS) {
        return false;
    }
    const int lane = warpgroup::laneid();
    const int query_row = m_tile * 128 + lane;
    return t_coord.x ==
               fp4pv_native_group2_selected_row_trace_control[1] &&
           t_coord.z ==
               fp4pv_native_group2_selected_row_trace_control[2] &&
           query_row ==
               fp4pv_native_group2_selected_row_trace_control[3];
}

__device__ __forceinline__ int
fp4pv_native_group2_selected_row_record_base(int idx) {
    return FP4PV_NATIVE_GROUP2_SELECTED_ROW_HEADER_WORDS +
        idx * FP4PV_NATIVE_GROUP2_SELECTED_ROW_RECORD_WORDS;
}

__device__ __forceinline__ void
fp4pv_native_group2_selected_row_record_raw(
    int route_kind,
    int3 t_coord,
    int m_tile,
    int idx,
    int iters_per_task,
    int cta_rank,
    uint32_t score_tt_base,
    int q_scale_tmem_base,
    int k_scale_tmem_base,
    const float2 *scores_reg) {
    if (!fp4pv_native_group2_selected_row_trace_matches(
            route_kind, t_coord, m_tile, idx)) {
        return;
    }
    constexpr unsigned int MAGIC = 0x53525431u;  // "SRT1"
    const int base = fp4pv_native_group2_selected_row_record_base(idx);
    unsigned long long hash = 1469598103934665603ull;
    float raw_max = base_types::constants<float>::neg_infty();
    float raw_sum = 0.0f;
    float quarter_max[4] = {
        base_types::constants<float>::neg_infty(),
        base_types::constants<float>::neg_infty(),
        base_types::constants<float>::neg_infty(),
        base_types::constants<float>::neg_infty()};
    float quarter_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int pair = 0; pair < 64; ++pair) {
        const int quarter = pair >> 4;
        const unsigned int x_word = __float_as_uint(scores_reg[pair].x);
        const unsigned int y_word = __float_as_uint(scores_reg[pair].y);
        hash = (hash ^ static_cast<unsigned long long>(x_word)) *
            1099511628211ull;
        hash = (hash ^ static_cast<unsigned long long>(y_word)) *
            1099511628211ull;
        raw_max = fmaxf(raw_max, fmaxf(scores_reg[pair].x,
                                      scores_reg[pair].y));
        raw_sum += scores_reg[pair].x + scores_reg[pair].y;
        quarter_max[quarter] = fmaxf(
            quarter_max[quarter],
            fmaxf(scores_reg[pair].x, scores_reg[pair].y));
        quarter_sum[quarter] +=
            scores_reg[pair].x + scores_reg[pair].y;
    }

    fp4pv_native_group2_selected_row_trace[base + 0] = MAGIC;
    fp4pv_native_group2_selected_row_trace[base + 1] = route_kind;
    fp4pv_native_group2_selected_row_trace[base + 2] = idx;
    fp4pv_native_group2_selected_row_trace[base + 3] = blockIdx.x;
    fp4pv_native_group2_selected_row_trace[base + 4] = cta_rank;
    fp4pv_native_group2_selected_row_trace[base + 5] = t_coord.x;
    fp4pv_native_group2_selected_row_trace[base + 6] = t_coord.z;
    fp4pv_native_group2_selected_row_trace[base + 7] = m_tile;
    fp4pv_native_group2_selected_row_trace[base + 8] =
        warpgroup::laneid();
    fp4pv_native_group2_selected_row_trace[base + 9] = score_tt_base;
    fp4pv_native_group2_selected_row_trace[base + 10] = q_scale_tmem_base;
    fp4pv_native_group2_selected_row_trace[base + 11] = k_scale_tmem_base;
    fp4pv_native_group2_selected_row_trace[base + 12] =
        static_cast<unsigned int>(hash);
    fp4pv_native_group2_selected_row_trace[base + 13] =
        static_cast<unsigned int>(hash >> 32);
    fp4pv_native_group2_selected_row_trace[base + 14] =
        __float_as_uint(raw_max);
    fp4pv_native_group2_selected_row_trace[base + 15] =
        __float_as_uint(raw_sum);
    #pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
        fp4pv_native_group2_selected_row_trace[base + 16 + quarter] =
            __float_as_uint(quarter_max[quarter]);
        fp4pv_native_group2_selected_row_trace[base + 20 + quarter] =
            __float_as_uint(quarter_sum[quarter]);
    }
    fp4pv_native_group2_selected_row_trace[base + 32] = m_tile;
    fp4pv_native_group2_selected_row_trace[base + 33] =
        idx * 2 + cta_rank;
    fp4pv_native_group2_selected_row_trace[base + 34] = iters_per_task;
    fp4pv_native_group2_selected_row_trace[base + 39] = 1u;

    if (idx == fp4pv_native_group2_selected_row_trace_control[4]) {
        const int raw_base =
            FP4PV_NATIVE_GROUP2_SELECTED_ROW_HEADER_WORDS +
            FP4PV_NATIVE_GROUP2_SELECTED_ROW_MAX_ITERS *
                FP4PV_NATIVE_GROUP2_SELECTED_ROW_RECORD_WORDS;
        #pragma unroll
        for (int pair = 0; pair < 64; ++pair) {
            fp4pv_native_group2_selected_row_trace[raw_base + pair * 2] =
                __float_as_uint(scores_reg[pair].x);
            fp4pv_native_group2_selected_row_trace[
                raw_base + pair * 2 + 1] =
                __float_as_uint(scores_reg[pair].y);
        }
        fp4pv_native_group2_selected_row_trace[7] = idx;
    }
    atomicOr(&fp4pv_native_group2_selected_row_trace[4], 1u << idx);
    fp4pv_native_group2_selected_row_trace[0] = MAGIC;
    fp4pv_native_group2_selected_row_trace[1] = route_kind;
    fp4pv_native_group2_selected_row_trace[2] =
        fp4pv_native_group2_selected_row_trace_control[2];
    fp4pv_native_group2_selected_row_trace[3] =
        fp4pv_native_group2_selected_row_trace_control[3];
}

__device__ __forceinline__ void
fp4pv_native_group2_selected_row_record_softmax(
    int route_kind,
    int3 t_coord,
    int m_tile,
    int idx,
    const float2 *scores_reg,
    float scale_log2,
    float row_max_old,
    float row_max,
    float acc_scale,
    float exact_row_max) {
    if (!fp4pv_native_group2_selected_row_trace_matches(
            route_kind, t_coord, m_tile, idx)) {
        return;
    }
    float post_mask_max = base_types::constants<float>::neg_infty();
    #pragma unroll
    for (int pair = 0; pair < 64; ++pair) {
        post_mask_max = fmaxf(
            post_mask_max,
            fmaxf(scores_reg[pair].x, scores_reg[pair].y));
    }
    const int base = fp4pv_native_group2_selected_row_record_base(idx);
    fp4pv_native_group2_selected_row_trace[base + 24] =
        __float_as_uint(post_mask_max);
    fp4pv_native_group2_selected_row_trace[base + 25] =
        __float_as_uint(row_max_old);
    fp4pv_native_group2_selected_row_trace[base + 26] =
        __float_as_uint(row_max);
    fp4pv_native_group2_selected_row_trace[base + 27] =
        __float_as_uint(acc_scale);
    fp4pv_native_group2_selected_row_trace[base + 35] =
        __float_as_uint((post_mask_max - row_max) * scale_log2);
    fp4pv_native_group2_selected_row_trace[base + 36] =
        __float_as_uint(exact_row_max);
    fp4pv_native_group2_selected_row_trace[base + 39] |= 2u;
    atomicOr(&fp4pv_native_group2_selected_row_trace[5], 1u << idx);
}

__device__ __forceinline__ void
fp4pv_native_group2_selected_row_record_sums(
    int route_kind,
    int3 t_coord,
    int m_tile,
    int idx,
    float true_tile_sum,
    float running_exact_sum,
    float payload_tile_sum,
    float running_payload_sum) {
    if (!fp4pv_native_group2_selected_row_trace_matches(
            route_kind, t_coord, m_tile, idx)) {
        return;
    }
    const int base = fp4pv_native_group2_selected_row_record_base(idx);
    fp4pv_native_group2_selected_row_trace[base + 28] =
        __float_as_uint(true_tile_sum);
    fp4pv_native_group2_selected_row_trace[base + 29] =
        __float_as_uint(running_exact_sum);
    fp4pv_native_group2_selected_row_trace[base + 30] =
        __float_as_uint(payload_tile_sum);
    fp4pv_native_group2_selected_row_trace[base + 31] =
        __float_as_uint(running_payload_sum);
    fp4pv_native_group2_selected_row_trace[base + 39] |= 4u;
    atomicOr(&fp4pv_native_group2_selected_row_trace[6], 1u << idx);
}
#endif

#if TK_FA4_NATIVE_GROUP2_RAW_QK_DIAG_BUILD
// Selected-row attribution only.  One launch records one logical query row for
// every N128 QK iteration immediately after the TMEM load acquire and before
// causal masking or softmax arithmetic.  The A and B routes are launched after
// separate resets, so they intentionally share this buffer.
constexpr int FP4PV_NATIVE_GROUP2_RAW_QK_MAX_ITERS = 32;
constexpr int FP4PV_NATIVE_GROUP2_RAW_QK_SCORE_WORDS = 128;
constexpr int FP4PV_NATIVE_GROUP2_RAW_QK_META_WORDS = 16;
constexpr int FP4PV_NATIVE_GROUP2_RAW_QK_RECORD_WORDS =
    FP4PV_NATIVE_GROUP2_RAW_QK_META_WORDS +
    FP4PV_NATIVE_GROUP2_RAW_QK_SCORE_WORDS;
constexpr int FP4PV_NATIVE_GROUP2_RAW_QK_HEADER_WORDS = 32;
constexpr int FP4PV_NATIVE_GROUP2_RAW_QK_DIAG_WORDS =
    FP4PV_NATIVE_GROUP2_RAW_QK_HEADER_WORDS +
    FP4PV_NATIVE_GROUP2_RAW_QK_MAX_ITERS *
        FP4PV_NATIVE_GROUP2_RAW_QK_RECORD_WORDS;

__device__ unsigned int fp4pv_native_group2_raw_qk_diag[
    FP4PV_NATIVE_GROUP2_RAW_QK_DIAG_WORDS];
// enabled, batch, head, absolute query row
__device__ int fp4pv_native_group2_raw_qk_diag_control[4];

__device__ __forceinline__ void fp4pv_native_group2_record_raw_qk_row(
    int route_kind,
    int3 t_coord,
    int m_tile,
    int idx,
    int iters_per_task,
    int cta_rank,
    uint32_t score_tt_base,
    int q_scale_tmem_base,
    int k_scale_tmem_base,
    const float2 *scores_reg) {
    if (fp4pv_native_group2_raw_qk_diag_control[0] == 0 ||
        idx < 0 || idx >= FP4PV_NATIVE_GROUP2_RAW_QK_MAX_ITERS) {
        return;
    }
    const int lane = warpgroup::laneid();
    const int query_row = m_tile * 128 + lane;
    if (t_coord.x != fp4pv_native_group2_raw_qk_diag_control[1] ||
        t_coord.z != fp4pv_native_group2_raw_qk_diag_control[2] ||
        query_row != fp4pv_native_group2_raw_qk_diag_control[3]) {
        return;
    }

    constexpr unsigned int MAGIC = 0x51524b31u;  // "QRK1"
    const int base = FP4PV_NATIVE_GROUP2_RAW_QK_HEADER_WORDS +
        idx * FP4PV_NATIVE_GROUP2_RAW_QK_RECORD_WORDS;
    fp4pv_native_group2_raw_qk_diag[base + 0] = MAGIC;
    fp4pv_native_group2_raw_qk_diag[base + 1] =
        static_cast<unsigned int>(route_kind);
    fp4pv_native_group2_raw_qk_diag[base + 2] =
        static_cast<unsigned int>(idx);
    fp4pv_native_group2_raw_qk_diag[base + 3] =
        static_cast<unsigned int>(blockIdx.x);
    fp4pv_native_group2_raw_qk_diag[base + 4] =
        static_cast<unsigned int>(cta_rank);
    fp4pv_native_group2_raw_qk_diag[base + 5] =
        static_cast<unsigned int>(t_coord.x);
    fp4pv_native_group2_raw_qk_diag[base + 6] =
        static_cast<unsigned int>(t_coord.z);
    fp4pv_native_group2_raw_qk_diag[base + 7] =
        static_cast<unsigned int>(t_coord.y);
    fp4pv_native_group2_raw_qk_diag[base + 8] =
        static_cast<unsigned int>(m_tile);
    fp4pv_native_group2_raw_qk_diag[base + 9] =
        static_cast<unsigned int>(lane);
    fp4pv_native_group2_raw_qk_diag[base + 10] = score_tt_base;
    fp4pv_native_group2_raw_qk_diag[base + 11] =
        static_cast<unsigned int>(q_scale_tmem_base);
    fp4pv_native_group2_raw_qk_diag[base + 12] =
        static_cast<unsigned int>(k_scale_tmem_base);
    // These are the exact TMA logical coordinates used by both native-group2
    // implementations: Q scale page for this rank's M128 tile and the rank-
    // local K64 half that contributes to this N128 group2 score tile.
    fp4pv_native_group2_raw_qk_diag[base + 13] =
        static_cast<unsigned int>(m_tile);
    fp4pv_native_group2_raw_qk_diag[base + 14] =
        static_cast<unsigned int>(idx * 2 + cta_rank);
    fp4pv_native_group2_raw_qk_diag[base + 15] =
        static_cast<unsigned int>(iters_per_task);
    #pragma unroll
    for (int pair = 0; pair < FP4PV_NATIVE_GROUP2_RAW_QK_SCORE_WORDS / 2;
         ++pair) {
        fp4pv_native_group2_raw_qk_diag[
            base + FP4PV_NATIVE_GROUP2_RAW_QK_META_WORDS + pair * 2 + 0] =
            __float_as_uint(scores_reg[pair].x);
        fp4pv_native_group2_raw_qk_diag[
            base + FP4PV_NATIVE_GROUP2_RAW_QK_META_WORDS + pair * 2 + 1] =
            __float_as_uint(scores_reg[pair].y);
    }
    if (idx < 32) {
        atomicOr(&fp4pv_native_group2_raw_qk_diag[4], 1u << idx);
    }
    fp4pv_native_group2_raw_qk_diag[0] = MAGIC;
    fp4pv_native_group2_raw_qk_diag[1] =
        static_cast<unsigned int>(route_kind);
    fp4pv_native_group2_raw_qk_diag[2] =
        static_cast<unsigned int>(fp4pv_native_group2_raw_qk_diag_control[2]);
    fp4pv_native_group2_raw_qk_diag[3] =
        static_cast<unsigned int>(fp4pv_native_group2_raw_qk_diag_control[3]);
    __threadfence_system();
}
#endif

constexpr int align_up_constexpr(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

template <typename Tile>
struct alignas(1024) tma_aligned_slot {
    Tile tile;
};

template <typename Slot, int N>
struct slot_array {
    Slot slots[N];
};

template <typename Slot, int LastPadBytes>
struct compact_last_padded_slots2 {
    Slot first;
    struct {
        Slot slot;
        char pad[LastPadBytes];
    } last;
};

template <int Bytes>
struct byte_guard {
    char bytes[Bytes];
};

static constexpr int FP4PV_DEBUG_TRAP_STAGE = 0;
static constexpr int FP4PV_DEBUG_PRINT_ROWSTATE = 0;
static constexpr bool FP4PV_DIAG_FULL_ZERO_NONFIRST = false;
// The tcgen accumulate path is more stable on this kernel when non-first PV issues
// start from a fresh mm2 issue instead of mma2 accumulation.
static constexpr bool FP4PV_USE_MM2_NONFIRST = true;
static constexpr bool FP4PV_DIAG_ISSUE_NEXT_QK_AFTER_PV = false;
// Sage-inspired online output rescaling with direct exp-domain FP4 P quantization.
static constexpr bool FP4PV_DIAG_USE_DIRECT_ROW_UPDATE = true;
static constexpr bool FP4PV_DIAG_CLUSTER_FENCE_P_STAGE = true;
static constexpr bool FP4PV_DIAG_SKIP_CORRECTION = false;
static constexpr bool FP4PV_USE_FIXED_P_TILE_SCALE = true;
static constexpr bool FP4PV_FUSE_P_GROUP_AMAX = true;
static constexpr int FP4PV_P_SCAN_PAD_BF16 = 8;
static constexpr int FP4PV_P_SCAN_STRIDE_BF16 = tk_localcta::BUFF_DIM_X + FP4PV_P_SCAN_PAD_BF16;
static constexpr float FP4PV_LOCALCTA_GLOBAL_SCALE_NUM = tk_localcta::LOCALCTA_DEFAULT_GLOBAL_SCALE_NUM;
static constexpr float FP4PV_LOCALCTA_GLOBAL_SCALE_RCP = 1.0f / FP4PV_LOCALCTA_GLOBAL_SCALE_NUM;
using fp4pv_p_scan3d = tk_localcta::IType[tk_localcta::BUFFS_NUM_IN][tk_localcta::BUFF_DIM_Y][FP4PV_P_SCAN_STRIDE_BF16];
using fp4pv_p_raw_scales = tk_localcta::ScalesType2D;

__device__ inline void fp4pv_packed_float_to_ue4m3(
    float f0, float f1, float f2, float f3,
    uint32_t &out
) {
    asm volatile(
        "{\n"
        ".reg .b32 val1;\n"
        ".reg .b32 val2;\n"
        ".reg .b32 val3;\n"
        ".reg .b32 val4;\n"
        "mov.b32 val1, %1;\n"
        "mov.b32 val2, %2;\n"
        "mov.b32 val3, %3;\n"
        "mov.b32 val4, %4;\n"
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %5;\n"
        "}\n"
        : "=r"(out)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3), "r"(0x80008000));
}

__device__ inline void fp4pv_packed_float_to_e2m1(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    uint32_t &out
) {
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(out)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3),
          "f"(f4), "f"(f5), "f"(f6), "f"(f7));
}

__device__ __forceinline__ float fp4pv_exp2_approx(float x) {
    float out;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(x));
    return out;
}

template <int N>
struct semaphore_array {
    semaphore sems[N];
    __device__ inline semaphore &operator[](int idx) { return sems[idx]; }
    __device__ inline const semaphore &operator[](int idx) const { return sems[idx]; }
};

template <>
struct semaphore_array<0> {};

template <typename Tile>
__host__ __device__ inline Tile &slot_tile(Tile &tile) {
    return tile;
}

template <typename Tile>
__host__ __device__ inline const Tile &slot_tile(const Tile &tile) {
    return tile;
}

template <typename Tile>
__host__ __device__ inline Tile &slot_tile(tma_aligned_slot<Tile> &slot) {
    return slot.tile;
}

template <typename Tile>
__host__ __device__ inline const Tile &slot_tile(const tma_aligned_slot<Tile> &slot) {
    return slot.tile;
}

template <typename Slot, int N>
__host__ __device__ inline auto &slot_tile_at(slot_array<Slot, N> &storage, int idx) {
    return slot_tile(storage.slots[idx]);
}

template <typename Slot, int N>
__host__ __device__ inline const auto &slot_tile_at(const slot_array<Slot, N> &storage, int idx) {
    return slot_tile(storage.slots[idx]);
}

template <typename Slot, int LastPadBytes>
__host__ __device__ inline auto &slot_tile_at(compact_last_padded_slots2<Slot, LastPadBytes> &storage, int idx) {
    return idx == 0 ? slot_tile(storage.first) : slot_tile(storage.last.slot);
}

template <typename Slot, int LastPadBytes>
__host__ __device__ inline const auto &slot_tile_at(const compact_last_padded_slots2<Slot, LastPadBytes> &storage, int idx) {
    return idx == 0 ? slot_tile(storage.first) : slot_tile(storage.last.slot);
}

template <int _Mb, int _Nb, int _Dqk, int _Dvo, bool _persistent,
          bool _fp4_v_dequant = false,
          bool _native_group2_hybrid = false,
          bool _selected_row_trace = false,
          bool _single_cta_fp4qk_bf16pv = false,
          int _cluster_size = 2>
struct config {
    static_assert(_Mb == 64 || _Mb == 128,  "Mb must be 64 or 128");
    static_assert(_Nb == 128,  "Nb must be 128");
    static_assert(_Dqk == 192, "Dqk must be 192, other shapes will be supported in the future");
    static_assert(_Dvo == 128, "Dvo must be 128");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;
    static constexpr bool persistent = _persistent;
    static constexpr bool FP4_V_DEQUANT = _fp4_v_dequant;
    static constexpr bool NATIVE_GROUP2_HYBRID = _native_group2_hybrid;
    static constexpr bool SELECTED_ROW_TRACE = _selected_row_trace;
    static constexpr bool SINGLE_CTA_FP4QK_BF16PV =
        _single_cta_fp4qk_bf16pv;
    static constexpr bool BF16_PV_HYBRID =
        NATIVE_GROUP2_HYBRID || SINGLE_CTA_FP4QK_BF16PV;
    static_assert(!FP4_V_DEQUANT || persistent,
                  "the FP4-V/BF16-PV control uses the persistent native-group2 schedule");
    static_assert(!NATIVE_GROUP2_HYBRID || persistent,
                  "the native-group2 hybrid controls use the one-softmaxer schedule");
    static_assert(!SELECTED_ROW_TRACE || NATIVE_GROUP2_HYBRID,
                  "selected-row B trace requires the native-group2 hybrid route");
    static_assert(_cluster_size == 1 || _cluster_size == 2,
                  "BF16-PV cluster size must be one or two CTAs");
    static_assert(!SINGLE_CTA_FP4QK_BF16PV ||
                      (persistent && _cluster_size == 1),
                  "single-CTA FP4-QK/BF16-PV requires the one-softmaxer group-1 schedule");
    static_assert(!NATIVE_GROUP2_HYBRID || _cluster_size == 2,
                  "native-group2 hybrid must retain its two-CTA topology");

    static constexpr int CLUSTER_SIZE = _cluster_size;
    static constexpr int NUM_SM = 152;

    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_CORRECTORS = 1;
    static constexpr int NUM_SOFTMAXXERS = persistent ? 1 : 2;
    static constexpr int TOTAL_WGS = NUM_PRODUCERS + NUM_CORRECTORS + NUM_SOFTMAXXERS;
    static constexpr int NUM_WARPS = (TOTAL_WGS) * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_STAGES = persistent ? 2 : 3;
    static constexpr int QK_SCALE_CHUNKS = Dqk / 64;
    static constexpr int Q_SCALE_SMEM_SLOTS = persistent ? NUM_SOFTMAXXERS : 1;
    static constexpr int K_SCALE_SMEM_SLOTS = persistent ? LOAD_STAGES : 2;
    static constexpr int Q_SC_TMEM_WIDTH = 16 * QK_SCALE_CHUNKS;
    static constexpr int K_SC_TMEM_WIDTH = 32 * QK_SCALE_CHUNKS;
    static constexpr int OUTPUT_CHUNK_COLS = 32;
    static constexpr int OUTPUT_CHUNKS = Dvo / OUTPUT_CHUNK_COLS;
    // Production-derived hybrid ownership: score [0,128), output [128,256),
    // copied BF16 P [256,384), then Q/K scale staging [384,448).
    static constexpr int BF16_P_TMEM_BASE = Nb + Dvo;
    static constexpr int HYBRID_SCALE_TMEM_BASE = BF16_P_TMEM_BASE + Nb;
    static_assert(!BF16_PV_HYBRID || BF16_P_TMEM_BASE + Nb <= MAX_TENSOR_COLS,
                  "BF16-PV hybrid score/P separation exceeds TMEM capacity");
    static_assert(!BF16_PV_HYBRID ||
                      HYBRID_SCALE_TMEM_BASE + Q_SC_TMEM_WIDTH + 16 <=
                          MAX_TENSOR_COLS,
                  "BF16-PV hybrid Q/K scale staging exceeds TMEM capacity");
};

#if TK_FA4_SINGLE_CTA_FP4QK_BF16PV_ONLY_BUILD
struct config_single_cta_fp4qk_fp4v_dequant_bf16pv
    : config<128, 128, 192, 128, true, true, false, false, true, 1> {
    static constexpr int SCORE_TMEM_BASE = 0;
    static constexpr int OUTPUT_TMEM_BASE = 128;
    static constexpr int BF16_P_TMEM_BASE = 256;
    static constexpr int Q_SCALE_TMEM_BASE = 384;
    static constexpr int K_SCALE_TMEM_BASE = 432;
    static constexpr int TMEM_COLS = 448;
    static_assert(Mb == 128 && Nb == 128 && Dqk == 192 && Dvo == 128);
    static_assert(CLUSTER_SIZE == 1 && SINGLE_CTA_FP4QK_BF16PV);
    static_assert(FP4_V_DEQUANT && BF16_PV_HYBRID);
    static_assert(HYBRID_SCALE_TMEM_BASE == Q_SCALE_TMEM_BASE);
    static_assert(Q_SCALE_TMEM_BASE + Q_SC_TMEM_WIDTH == K_SCALE_TMEM_BASE);
    static_assert(K_SCALE_TMEM_BASE + 16 == TMEM_COLS);
    static_assert(TMEM_COLS <= MAX_TENSOR_COLS);
};

struct config_single_cta_fp4qk_bf16v_bf16pv_ceiling
    : config<128, 128, 192, 128, true, false, false, false, true, 1> {
    static_assert(CLUSTER_SIZE == 1 && SINGLE_CTA_FP4QK_BF16PV);
    static_assert(!FP4_V_DEQUANT && BF16_PV_HYBRID);
};
#endif

template <typename C>
struct globals {
    using q_tile = st_fp4e2m1_2<C::Mb, C::Dqk/2>;
    using q_sc_tile = st_hf<C::QK_SCALE_CHUNKS, 256, false>;
    // Each rank owns Nb/CLUSTER_SIZE keys.  The historical Nb/2 spelling is
    // correct only for native cta_group::2; Track A is cta_group::1 and must
    // present the complete N128 K operand to tcgen05.
    using k_tile = st_fp4e2m1_2<C::Nb / C::CLUSTER_SIZE, C::Dqk/2>;
    using k_sc_tile = st_hf<C::QK_SCALE_CHUNKS, 256, false>;
    using v_full_tile = st_bf<C::Nb, C::Dvo / C::CLUSTER_SIZE>;
    using v_quarter_tile = st_bf<C::Nb / 4, C::Dvo / C::CLUSTER_SIZE>;
    // K32 BF16-PV operands must be genuine independently-swizzled tiles for
    // the single-CTA N128 route.  Raw byte subviews of st_bf<128,128> alias
    // across its two N64 swizzle panels.
    using v_tile = std::conditional_t<
        C::SINGLE_CTA_FP4QK_BF16PV, v_quarter_tile, v_full_tile>;
    static constexpr int V_SMEM_TILES_PER_STAGE =
        C::SINGLE_CTA_FP4QK_BF16PV ? 4 : 1;
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo / C::CLUSTER_SIZE, C::Nb / 2>;
    // Exact authoritative-production MXFP4 V-scale page for one K128 tile.
    // The uint8 container holds 32x16 E8M0 bytes, one scale per K32 group.
    using v_sc_tile = st_fp8e8m0<32, 16, false>;
    using o_tile = std::conditional_t<C::persistent, st_bf<C::Mb, C::Dvo, true, 64>, st_bf<C::Mb, C::Dvo>>;
    using q_sc_smem_storage = slot_array<std::conditional_t<C::persistent, tma_aligned_slot<q_sc_tile>, q_sc_tile>, C::Q_SCALE_SMEM_SLOTS>;
    using k_sc_smem_storage = slot_array<tma_aligned_slot<k_sc_tile>, C::K_SCALE_SMEM_SLOTS>;
    using k_sc_guard_slot = byte_guard<1024>;
    using o_backing_tile = std::conditional_t<C::persistent, st_bf<32, 32>, st_fl<C::Mb, C::Dvo, false>>;

    using q_gl = gl<fp4e2m1_2, -1, -1, -1, -1, tma::descriptor<q_tile, dim::ROW>>;
    using q_sc_gl = gl<half, -1, -1, -1, 256, tma::descriptor<q_sc_tile, dim::ROW>>;
    using q_sg_gl = gl<float, -1, -1, -1, -1>;
    using k_gl = gl<fp4e2m1_2, -1, -1, -1, -1, tma::descriptor<k_tile, dim::ROW>>;
    using k_sc_gl = gl<half, -1, -1, -1, 256, tma::descriptor<k_sc_tile, dim::ROW>>;
    using k_sg_gl = gl<float, -1, -1, -1, -1>;
    struct unused_v_gl {};
    struct unused_v_fp4_gl {};
    using v_gl = std::conditional_t<
        C::FP4_V_DEQUANT,
        unused_v_gl,
        gl<bf16, -1, -1, -1, -1, tma::descriptor<v_tile, dim::DEPTH>>>;
    using v_fp4_gl = std::conditional_t<
        C::FP4_V_DEQUANT,
        gl<fp4e2m1_2, -1, -1, -1, -1,
           tma::descriptor<v_fp4_tile, dim::ROW>>,
        unused_v_fp4_gl>;
    using o_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<o_tile, dim::DEPTH>>;
    using lse_gl = gl<float, -1, -1, -1, -1>;

private:
    static constexpr int compute_dynamic_shared_memory() {
        int bytes = 0;
        auto add = [&](int alloc_bytes) constexpr {
            bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
        };
        add((int)sizeof(q_tile) * C::NUM_SOFTMAXXERS);
        add((int)sizeof(q_sc_smem_storage));
        add((int)sizeof(k_tile) * C::LOAD_STAGES);
        add((int)sizeof(v_tile) * C::LOAD_STAGES * V_SMEM_TILES_PER_STAGE);
        if constexpr (C::FP4_V_DEQUANT) {
            add((int)sizeof(v_fp4_tile) * C::LOAD_STAGES);
            add((int)sizeof(v_sc_tile) * C::LOAD_STAGES);
        }
        add((int)sizeof(o_backing_tile) * C::NUM_SOFTMAXXERS);
        if constexpr (C::persistent) {
            add((int)sizeof(o_tile));
        }
        add((int)sizeof(k_sc_smem_storage));
        add((int)sizeof(k_sc_guard_slot));
        return bytes;
    }

public:
    // Keep this in sync with the allocator order in kernel().
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

    q_gl q;
    q_sc_gl q_sc;
    q_sg_gl q_sg;
    k_gl k;
    k_sc_gl k_sc;
    k_sg_gl k_sg;
    v_gl v;
    v_fp4_gl v_fp4_tma;
    const uint8_t *v_sc_prepared;
    int v_scale_records_per_head;
    o_gl o;
    lse_gl lse;

    __host__ __inline__ dim3 grid() {
        if constexpr (C::persistent) {
            const int total_bids = q.batch() * q.depth() * (q.rows() / (C::Mb * C::NUM_SOFTMAXXERS));
            constexpr int PERSISTENT_GRID_CAP = 2 * C::NUM_SM;
            return dim3(total_bids < PERSISTENT_GRID_CAP ? total_bids : PERSISTENT_GRID_CAP);
        }
        else {
            return dim3(q.batch() * q.depth() * (q.rows() / (C::Mb * C::NUM_SOFTMAXXERS)));
        }
    }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize = 2>
struct config_fp4pv {
    static_assert(_Mb == 64 || _Mb == 128,  "Mb must be 64 or 128");
    static_assert(_Nb == 128 || _Nb == 64, "Nb must be 128 or 64");
    static_assert(_Dqk == 192, "Dqk must be 192");
    static_assert(_Dvo == 64 || _Dvo == 128, "Dvo must be 64 or 128");
    static_assert(_ClusterSize == 1 || _ClusterSize == 2, "fp4pv cluster size must be 1 or 2");

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Dqk = _Dqk;
    static constexpr int Dvo = _Dvo;

    static constexpr int CLUSTER_SIZE = _ClusterSize;
    static constexpr int NUM_SM = 152;

    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_CORRECTORS = 1;
    static constexpr int NUM_SOFTMAXXERS = 1;
    static constexpr int NUM_QUANTIZERS = 1;
    static constexpr int TOTAL_WGS = NUM_PRODUCERS + NUM_CORRECTORS + NUM_SOFTMAXXERS + NUM_QUANTIZERS;
    static constexpr int NUM_WARPS = TOTAL_WGS * 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_STAGES = 2;
    static constexpr int V_LOAD_STAGES = 2;
    static constexpr int QK_SCALE_CHUNKS = Dqk / 64;
    static constexpr int Q_SCALE_SMEM_SLOTS = NUM_SOFTMAXXERS;
    static constexpr int K_SCALE_SMEM_SLOTS = LOAD_STAGES;
    static constexpr int PV_SCALE_CHUNKS = Nb / 64;
    static constexpr int Q_SC_TMEM_WIDTH = 16 * QK_SCALE_CHUNKS;
    static constexpr int K_SC_TMEM_WIDTH = 32 * QK_SCALE_CHUNKS;
    static constexpr int P_SC_TMEM_WIDTH = 16 * PV_SCALE_CHUNKS;
    static constexpr int V_SC_TMEM_WIDTH = 32 * PV_SCALE_CHUNKS;
};

template <typename C, bool NEED_P_SCAN = true>
struct globals_fp4pv {
    using q_tile = st_fp4e2m1_2<C::Mb, C::Dqk/2>;
    using q_sc_tile = st_hf<C::QK_SCALE_CHUNKS, 256, false>;
    using k_tile = st_fp4e2m1_2<C::Nb/2, C::Dqk/2>;
    using k_sc_tile = st_hf<C::QK_SCALE_CHUNKS, 256, false>;
    using p_fp4_tile = st_fp4e2m1_2<C::Mb, C::Nb/2>;
    using p_sc_tile = st_hf<C::PV_SCALE_CHUNKS, 256, false>;
    using p_sc_guard_slot = byte_guard<1024>;
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo/2, C::Nb/2>;
    using v_sc_tile = st_hf<C::PV_SCALE_CHUNKS, 256, false>;
    using o_tile = st_bf<C::Mb, C::Dvo, true, 64>;
    using q_sc_smem_storage = slot_array<tma_aligned_slot<q_sc_tile>, C::Q_SCALE_SMEM_SLOTS>;
    using k_sc_smem_storage = slot_array<tma_aligned_slot<k_sc_tile>, C::K_SCALE_SMEM_SLOTS>;
    using k_sc_guard_slot = byte_guard<1024>;

    using q_gl = gl<fp4e2m1_2, -1, -1, -1, -1, tma::descriptor<q_tile, dim::ROW>>;
    using q_sc_gl = gl<half, -1, -1, -1, 256, tma::descriptor<q_sc_tile, dim::ROW>>;
    using q_sg_gl = gl<float, -1, -1, -1, -1>;
    using k_gl = gl<fp4e2m1_2, -1, -1, -1, -1, tma::descriptor<k_tile, dim::ROW>>;
    using k_sc_gl = gl<half, -1, -1, -1, 256, tma::descriptor<k_sc_tile, dim::ROW>>;
    using k_sg_gl = gl<float, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1, tma::descriptor<o_tile, dim::DEPTH>>;
    using lse_gl = gl<float, -1, -1, -1, -1>;

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
        add((int)sizeof(v_fp4_tile) * C::V_LOAD_STAGES);
        add((int)sizeof(v_sc_tile) * C::V_LOAD_STAGES);
        add((int)sizeof(o_tile));
        if constexpr (NEED_P_SCAN) {
            add((int)(sizeof(fp4pv_p_scan3d) * 2));
        }
        add((int)(sizeof(p_fp4_tile) * 2));
        add((int)(sizeof(p_sc_tile) * 2));
        add((int)sizeof(p_sc_guard_slot));
        return bytes;
    }

public:
    static constexpr bool NEED_P_SCAN_STORAGE = NEED_P_SCAN;
    static constexpr int DYNAMIC_SHARED_MEMORY = compute_dynamic_shared_memory();

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

    __host__ __inline__ dim3 grid() {
        const int tiles_m = q.rows() / C::Mb;
        const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
        const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
        const int total_bids = q.batch() * q.depth() * num_block * C::CLUSTER_SIZE;
        constexpr int PERSISTENT_GRID_CAP = 2 * C::NUM_SM;
        constexpr int FULLGRID_SEQ_THRESHOLD = 4096;
        int grid_x = (q.rows() >= FULLGRID_SEQ_THRESHOLD)
            ? total_bids
            : (total_bids < PERSISTENT_GRID_CAP ? total_bids : PERSISTENT_GRID_CAP);
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

template <bool NEED_P_SCAN, int N>
struct fp4pv_p_scan_view;

template <int N>
struct fp4pv_p_scan_view<true, N> {
    fp4pv_p_scan3d *ptr;

    __device__ inline fp4pv_p_scan3d &operator[](int idx) const { return ptr[idx]; }
};

template <int N>
struct fp4pv_p_scan_view<false, N> {
    __device__ inline fp4pv_p_scan3d &operator[](int) const {
        asm volatile("trap;");
        return *reinterpret_cast<fp4pv_p_scan3d *>(0);
    }
};

template <bool NEED_P_SCAN, int N>
__device__ inline fp4pv_p_scan_view<NEED_P_SCAN, N> fp4pv_allocate_p_scan(tma_swizzle_allocator &al) {
    if constexpr (NEED_P_SCAN) {
        auto (&scan)[N] = al.allocate<fp4pv_p_scan3d, N>();
        return fp4pv_p_scan_view<true, N>{&scan[0]};
    } else {
        return fp4pv_p_scan_view<false, N>{};
    }
}

template <typename C>
__device__ inline float qk_scale_log2(const globals<C> &g, int b, int h) {
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

template <typename C, bool NEED_P_SCAN>
__device__ inline float qk_scale_log2(const globals_fp4pv<C, NEED_P_SCAN> &g, int b, int h) {
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

template <typename C>
__device__ inline void fp4pv_cluster_sync() {
    if constexpr (C::CLUSTER_SIZE > 1) {
        everyone::tma::cluster::sync();
    } else {
        __syncthreads();
    }
}

template <typename C>
__device__ inline void fp4pv_expect_bytes(semaphore &bar, uint32_t bytes) {
    if constexpr (C::CLUSTER_SIZE > 1) {
        tma::cluster::expect_bytes(bar, bytes);
    } else {
        tma::expect_bytes(bar, bytes);
    }
}

template <typename C>
__device__ inline void fp4pv_wait(semaphore &bar, int phase) {
    if constexpr (C::CLUSTER_SIZE > 1) {
        tma::cluster::wait(bar, phase);
    } else {
        wait(bar, phase);
    }
}

template <int axis, cache_policy policy, typename C, typename ST, typename GL, typename COORD>
__device__ inline void fp4pv_load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar, uint16_t cluster_mask, int dst_mbar_cta = -1) {
    if constexpr (C::CLUSTER_SIZE > 1) {
        tma::cluster::load_async<axis, policy>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
    } else {
        tma::load_async<axis, policy>(dst, src, idx, bar);
    }
}

template <typename C>
__device__ inline void fp4pv_remote_arrive_if_needed(semaphore &bar, int dst_cta) {
    if constexpr (C::CLUSTER_SIZE > 1) {
        tma::cluster::arrive(bar, dst_cta);
    }
}

template <typename SV>
__device__ inline float stats_load(const SV &vec, int idx) {
    if constexpr (std::is_same_v<typename SV::dtype, bf16>) {
        return __bfloat162float(vec[idx]);
    }
    else {
        return vec[idx];
    }
}

template <typename SV>
__device__ inline void stats_store(SV &vec, int idx, float value) {
    if constexpr (std::is_same_v<typename SV::dtype, bf16>) {
        vec[idx] = __float2bfloat16(value);
    }
    else {
        vec[idx] = value;
    }
}

__device__ inline void publish_shared_backing() {
    __threadfence_block();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <typename C>
__device__ inline void publish_cluster_shared_backing_if_needed() {
    if constexpr (C::CLUSTER_SIZE > 1) {
        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
    }
}

__device__ inline float softmax_rescale_factor(float &row_max, float row_max_old, float scale_log2) {
    constexpr float rescale_threshold = 8.f;
    if (!isfinite(row_max_old) || !isfinite(row_max) || scale_log2 == 0.0f) {
        return 1.0f;
    }
    float acc_scale_log2 = (row_max_old - row_max) * scale_log2;
    if (!isfinite(acc_scale_log2) || acc_scale_log2 >= -rescale_threshold) {
        row_max = row_max_old;
        return 1.0f;
    }
    return exp2f(acc_scale_log2);
}

__device__ inline float2 apply_softmax_scale(float2 scores, float scale_log2, float2 scale_2, float2 neg_max_scaled_2) {
    if (scale_log2 == 0.0f) {
        scores.x = isfinite(scores.x) ? 0.0f : scores.x;
        scores.y = isfinite(scores.y) ? 0.0f : scores.y;
        return scores;
    }
    return __ffma2_rn(scores, scale_2, neg_max_scaled_2);
}

template <typename C, typename TT, typename ST>
__device__ inline void load_q_scale_chunk(TT &q_sc_tm, const ST &q_sc_smem, int chunk_idx) {
    auto &q_sc_sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
        reinterpret_cast<uint64_t>(&q_sc_smem.data[0]) + 16 * 32 * chunk_idx);
    load_mxnv_scale_async<C::CLUSTER_SIZE>(q_sc_tm, q_sc_sm_sub);
}

template <typename C, typename TT, typename ST>
__device__ inline void load_k_scale_chunk(TT &k_sc_tm, const ST &k_sc_smem, int chunk_idx) {
    auto &k_sc_sm_sub = *reinterpret_cast<const st_fp8e4m3<32, 16, false> *>(
        reinterpret_cast<uint64_t>(&k_sc_smem.data[0]) + 16 * 32 * chunk_idx);
    load_mxnv_scale_async<C::CLUSTER_SIZE>(k_sc_tm, k_sc_sm_sub);
}

template <typename C, typename QScTTFull, typename QScTile>
__device__ inline void stage_q_scale_tmem(QScTTFull &q_sc_tm_full, const QScTile &q_sc_smem) {
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; ++chunk) {
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        load_q_scale_chunk<C>(q_sc_tm, q_sc_smem, chunk);
    }
}

template <typename C, bool NEED_P_SCAN, typename VTile>
__device__ inline void load_v_fp4_tile_from_global(VTile &v_smem, const globals_fp4pv<C, NEED_P_SCAN> &g,
                                                   int b, int h, int k_iter, int cta_rank) {
    const int packed_seqlen = g.k.rows() / 2;
    const int row_base = cta_rank * (C::Dvo / C::CLUSTER_SIZE);
    const int col_base = k_iter * (C::Nb / 2);
    uint32_t smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    constexpr int BYTES_PER_STORE = 4;
    static_assert(VTile::cols % BYTES_PER_STORE == 0, "Expected fp4 V tile cols divisible by vector width");
    constexpr int STORES_PER_ROW = VTile::cols / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = VTile::rows * STORES_PER_ROW;
    for (int linear = warp::laneid(); linear < TOTAL_STORES; linear += WARP_THREADS) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const uint32_t value = *reinterpret_cast<const uint32_t *>(
            &g.v_fp4[((((b * g.q.depth() + h) * C::Dvo) + row_base + row) * packed_seqlen) + col_base + col]);
        asm volatile("st.shared.b32 [%0], %1;"
                     :
                     : "r"(VTile::idx(smem_base, {row, col})), "r"(value));
    }
}

template <typename C, bool NEED_P_SCAN, typename VScTile>
__device__ inline void load_v_sc_prepared_from_global(VScTile &v_sc_smem, const globals_fp4pv<C, NEED_P_SCAN> &g,
                                                      int b, int h, int k_iter) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(&v_sc_smem.data[0]);
    const int scale_k_base = k_iter * 2;
    static_assert(sizeof(VScTile) % sizeof(uint32_t) == 0, "Expected aligned V scale tile size");
    constexpr int WORDS_PER_SLAB = 512 / sizeof(uint32_t);
    constexpr int TOTAL_WORDS = sizeof(VScTile) / sizeof(uint32_t);
    for (int linear = warp::laneid(); linear < TOTAL_WORDS; linear += WARP_THREADS) {
        const int slab = linear / WORDS_PER_SLAB;
        const int word = linear % WORDS_PER_SLAB;
        dst[linear] = *reinterpret_cast<const uint32_t *>(
            &g.v_sc_prepared[((((b * g.q.depth() + h) * (C::Dvo / 128) + 0) * (g.k.rows() / 64)) + scale_k_base + slab) * 512 + word * sizeof(uint32_t)]);
    }
}

template <typename ScoresBfTT>
__device__ inline void copy_scores_quarter_to_localcta_scan(fp4pv_p_scan3d &scan,
                                                            const ScoresBfTT &tt_scores_bf,
                                                            int quarter) {
    const int warp_row_block = warpgroup::warpid();
    const int lane = warp::laneid();
    const int global_row = warp_row_block * 32 + lane;
    const int top_bottom = global_row / 64;
    const int local_row = global_row % 64;
    const int tile_x = quarter >> 1;
    const int local_col_base = (quarter & 1) * 32;
    const int tile_idx = top_bottom * 2 + tile_x;

    bf16_2 reg[16];
    asm volatile(
        "{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
        : "=r"(*(uint32_t *)&reg[0]), "=r"(*(uint32_t *)&reg[1]), "=r"(*(uint32_t *)&reg[2]), "=r"(*(uint32_t *)&reg[3]),
          "=r"(*(uint32_t *)&reg[4]), "=r"(*(uint32_t *)&reg[5]), "=r"(*(uint32_t *)&reg[6]), "=r"(*(uint32_t *)&reg[7]),
          "=r"(*(uint32_t *)&reg[8]), "=r"(*(uint32_t *)&reg[9]), "=r"(*(uint32_t *)&reg[10]), "=r"(*(uint32_t *)&reg[11]),
          "=r"(*(uint32_t *)&reg[12]), "=r"(*(uint32_t *)&reg[13]), "=r"(*(uint32_t *)&reg[14]), "=r"(*(uint32_t *)&reg[15])
        : "r"(tt_scores_bf.addr + ((warp_row_block * 32) << 16) + quarter * 16));
    tensor_load_wait();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        *reinterpret_cast<bf16_2 *>(&scan[tile_idx][local_row][local_col_base + i * 2]) = reg[i];
    }
}

__device__ inline void store_scores_quarter_to_localcta_scan(fp4pv_p_scan3d &scan,
                                                             const bf16_2 (&scores_bf_reg)[16],
                                                             int quarter) {
    const int warp_row_block = warpgroup::warpid();
    const int lane = warp::laneid();
    const int global_row = warp_row_block * 32 + lane;
    const int top_bottom = global_row / 64;
    const int local_row = global_row % 64;
    const int tile_x = quarter >> 1;
    const int local_col_base = (quarter & 1) * 32;
    const int tile_idx = top_bottom * 2 + tile_x;
    uint32_t smem_addr = __cvta_generic_to_shared(&scan[tile_idx][local_row][local_col_base]);

    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        const uint64_t packed = *reinterpret_cast<const uint64_t *>(&scores_bf_reg[i]);
        asm volatile("st.shared.b64 [%0], %1;" :: "r"(smem_addr + static_cast<uint32_t>(i * sizeof(bf16_2))), "l"(packed));
    }
}

__device__ inline void fp4pv_store_packed_scale_word(fp4pv_p_raw_scales &raw_scales,
                                                     int global_row, int word_idx,
                                                     tk_localcta::nvfp4_scale_t s0,
                                                     tk_localcta::nvfp4_scale_t s1,
                                                     tk_localcta::nvfp4_scale_t s2,
                                                     tk_localcta::nvfp4_scale_t s3) {
    const int row = global_row & 127;
    const uint32_t packed =
        static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(s0)) |
        (static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(s1)) << 8) |
        (static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(s2)) << 16) |
        (static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(s3)) << 24);
    const uint32_t smem_addr = __cvta_generic_to_shared(&raw_scales[row][word_idx * 4]);
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr), "r"(packed));
}

__device__ __forceinline__ int fp4pv_scale_swizzle_idx(int row, int k_block) {
    const int row_in_32 = row & 31;
    const int tile_in_block = (row >> 5) & 3;
    const int k_block_group = k_block >> 2;
    const int kb_in_block = k_block & 3;
    return k_block_group * 512 + row_in_32 * 16 + tile_in_block * 4 + kb_in_block;
}

// B2 control: reconstruct one rank-local K128xN64 BF16 tile from TMA-staged
// production MXFP4 payload and E8M0 scales.  No lane performs a global
// scalar payload/scale load; the producer owns both TMA transactions and this
// warp only reads their shared-memory destinations after acquire waits.
template <typename C, typename VTile, typename VFp4Tile, typename VScTile>
__device__ inline void dequantize_prepared_v_fp4_quarter_to_bf16_smem(
    VTile &v_smem,
    const VFp4Tile &v_fp4_smem,
    const VScTile &v_sc_smem,
    int cta_rank,
    int quarter
) {
    static_assert(C::Dvo == 128 && C::Nb == 128,
                  "B2 V reconstruction is specialized for Dvo=Nb=128");
    static_assert(
        (VTile::rows == C::Nb || VTile::rows == C::Nb / 4) &&
            VTile::cols == C::Dvo / C::CLUSTER_SIZE,
        "FP4-V reconstruction must feed a full or descriptor-correct K32 BF16 PV tile");
    constexpr bool QUARTER_DEST = VTile::rows == C::Nb / 4;

    constexpr int N_PER_RANK = C::Dvo / C::CLUSTER_SIZE;
    constexpr int PACKED_K_PER_TILE = C::Nb / 2;
    constexpr int PACKED_K_PER_QUARTER = (C::Nb / 4) / 2;
    constexpr int PACKED_VALUES = N_PER_RANK * PACKED_K_PER_QUARTER;
    const uint32_t out_smem_base = __cvta_generic_to_shared(&v_smem.data[0]);
    const uint32_t fp4_smem_base = __cvta_generic_to_shared(&v_fp4_smem.data[0]);
    const uint32_t scale_smem_base = __cvta_generic_to_shared(&v_sc_smem.data[0]);

    for (int linear = warp::laneid(); linear < PACKED_VALUES; linear += WARP_THREADS) {
        const int n_local = linear / PACKED_K_PER_QUARTER;
        const int packed_k_local =
            linear - n_local * PACKED_K_PER_QUARTER;
        const int packed_k =
            quarter * PACKED_K_PER_QUARTER + packed_k_local;
        const int n_global = cta_rank * N_PER_RANK + n_local;
        uint32_t packed_bits;
        asm volatile("ld.shared.u8 %0, [%1];"
                     : "=r"(packed_bits)
                     : "r"(VFp4Tile::idx(fp4_smem_base, {n_local, packed_k})));
        const uint8_t packed_byte = static_cast<uint8_t>(packed_bits);
        const fp4e2m1_2 packed =
            *reinterpret_cast<const fp4e2m1_2 *>(&packed_byte);
        const float2 values = static_cast<float2>(packed);
        // MXFP4 stores one E8M0 scale per 32 logical K values (16 packed bytes).
        const int k_block = packed_k / 16;
        const uint32_t scale_offset = static_cast<uint32_t>(
            fp4pv_scale_swizzle_idx(n_global, k_block));
        uint32_t scale_bits;
        asm volatile("ld.shared.u8 %0, [%1];"
                     : "=r"(scale_bits)
                     : "r"(scale_smem_base + scale_offset));
        const uint8_t scale_byte = static_cast<uint8_t>(scale_bits);
        const fp8e8m0 scale_encoded =
            *reinterpret_cast<const fp8e8m0 *>(&scale_byte);
        // Match _dequantize_mxfp4_rows_2d exactly: E8M0 encodes the block
        // amax, while the E2M1 payload range is [-6, 6].  Reserved/zero scale
        // bytes contribute zero.
        const float scale =
            (scale_byte == 0 || scale_byte >= 0xfe)
                ? 0.0f
                : static_cast<float>(scale_encoded) * (1.0f / 6.0f);

        const bf16 v0 = __float2bfloat16(values.x * scale);
        const bf16 v1 = __float2bfloat16(values.y * scale);
        const int output_k =
            QUARTER_DEST ? 2 * packed_k_local : 2 * packed_k;
        asm volatile("st.shared.b16 [%0], %1;"
                     :: "r"(VTile::idx(out_smem_base, {output_k + 0, n_local})),
                        "h"(*reinterpret_cast<const uint16_t *>(&v0)));
        asm volatile("st.shared.b16 [%0], %1;"
                     :: "r"(VTile::idx(out_smem_base, {output_k + 1, n_local})),
                        "h"(*reinterpret_cast<const uint16_t *>(&v1)));
    }
}

template <typename C, typename VTile, typename VFp4Tile, typename VScTile>
__device__ inline void dequantize_prepared_v_fp4_to_bf16_smem(
    VTile &v_smem,
    const VFp4Tile &v_fp4_smem,
    const VScTile &v_sc_smem,
    int cta_rank
) {
    #pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
        dequantize_prepared_v_fp4_quarter_to_bf16_smem<C>(
            v_smem, v_fp4_smem, v_sc_smem, cta_rank, quarter);
    }
}

template <typename PScTile>
__device__ inline void fp4pv_store_packed_scale_word_swizzled(PScTile &p_sc_prepared_tile,
                                                              int global_row, int word_idx,
                                                              tk_localcta::nvfp4_scale_t s0,
                                                              tk_localcta::nvfp4_scale_t s1,
                                                              tk_localcta::nvfp4_scale_t s2,
                                                              tk_localcta::nvfp4_scale_t s3) {
    const uint32_t packed =
        static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(s0)) |
        (static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(s1)) << 8) |
        (static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(s2)) << 16) |
        (static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(s3)) << 24);
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, word_idx * 4);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_prepared_tile.data[0])) + dest;
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr), "r"(packed));
}

template <typename PScTile>
__device__ inline void fp4pv_store_packed_scale_word_swizzled_f32(PScTile &p_sc_prepared_tile,
                                                                  int global_row, int word_idx,
                                                                  float s0, float s1, float s2, float s3) {
    uint32_t packed;
    fp4pv_packed_float_to_ue4m3(s0, s1, s2, s3, packed);
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, word_idx * 4);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_prepared_tile.data[0])) + dest;
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr), "r"(packed));
}

template <typename PScTile>
__device__ inline void fp4pv_store_scale_byte_swizzled(PScTile &p_sc_prepared_tile,
                                                       int global_row,
                                                       int group_idx,
                                                       tk_localcta::nvfp4_scale_t scale) {
    const int row = global_row & 127;
    const int dest = fp4pv_scale_swizzle_idx(row, group_idx);
    const uint32_t smem_addr = __cvta_generic_to_shared(&reinterpret_cast<uint8_t &>(p_sc_prepared_tile.data[0])) + dest;
    const uint32_t value = static_cast<uint32_t>(reinterpret_cast<const uint8_t &>(scale));
    asm volatile("st.shared.b8 [%0], %1;" :: "r"(smem_addr), "r"(value));
}

__device__ __forceinline__ float fp4pv_compute_tile_s_enc(float tile_amax) {
    float s_enc = FP4PV_LOCALCTA_GLOBAL_SCALE_NUM / tile_amax;
    s_enc = fminf(s_enc, transformer_engine::detail::TypeExtrema<float>::max);
    if (!(tile_amax > 0.0f) || !isfinite(tile_amax) || !(s_enc > 0.0f) || !isfinite(s_enc)) {
        s_enc = 1.0f;
    }
    return s_enc;
}

__device__ __forceinline__ void fp4pv_quantize_scores_group(const bf16_2 *scores_group_bf,
                                                            float s_enc,
                                                            uint32_t &word0,
                                                            uint32_t &word1,
                                                            tk_localcta::nvfp4_scale_t &decode_scale_fp8) {
    using namespace transformer_engine::dispatch::nvfp4::quantization_and_transposition_SF;

    tk_localcta::IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    #pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
        const tk_localcta::IType2 pair =
            *reinterpret_cast<const tk_localcta::IType2 *>(&scores_group_bf[jj]);
        transformer_engine::ptx::abs_max_2x(amax_2x, amax_2x, pair);
    }

    const float block_amax = tk_localcta::get_amax_of_pair(amax_2x);
    if (!(block_amax > 0.0f) || !isfinite(block_amax)) {
        word0 = 0;
        word1 = 0;
        decode_scale_fp8 = tk_localcta::nvfp4_scale_t(0.0f);
        return;
    }

    constexpr float float_max = 3.4028235e+38f;
    const tk_localcta::nvfp4_scale_t s_b_fp8 = compute_decoding_scaling_factor(block_amax, s_enc);
    const float s_b = static_cast<float>(s_b_fp8);
    const float s_dec = 1.0f / s_enc;
    const float coeff = fminf(1.0f / (s_b * s_dec), float_max);
    if (!(s_b > 0.0f) || !isfinite(s_b) || !(s_dec > 0.0f) || !isfinite(s_dec) ||
        !(coeff > 0.0f) || !isfinite(coeff)) {
        word0 = 0;
        word1 = 0;
        decode_scale_fp8 = tk_localcta::nvfp4_scale_t(0.0f);
        return;
    }
    decode_scale_fp8 = s_b_fp8;

    const uint64_t e03_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[0]);
    const uint64_t e47_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[2]);
    const uint64_t e03_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[4]);
    const uint64_t e47_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[6]);
    word0 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_0, e47_0, coeff);
    word1 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff);
}

__device__ __forceinline__ void fp4pv_quantize_scores_group_from_float2(const float2 *scores_group,
                                                                        float s_enc,
                                                                        uint32_t &word0,
                                                                        uint32_t &word1,
                                                                        tk_localcta::nvfp4_scale_t &decode_scale_fp8) {
    __align__(8) bf16_2 scores_group_bf[8];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
        const float2 pair_f = scores_group[jj];
        scores_group_bf[jj] = __float22bfloat162_rn(pair_f);
        block_amax = fmaxf(block_amax, fmaxf(fabsf(pair_f.x), fabsf(pair_f.y)));
    }
    using namespace transformer_engine::dispatch::nvfp4::quantization_and_transposition_SF;
    if (!(block_amax > 0.0f) || !isfinite(block_amax)) {
        word0 = 0;
        word1 = 0;
        decode_scale_fp8 = tk_localcta::nvfp4_scale_t(0.0f);
        return;
    }
    constexpr float float_max = 3.4028235e+38f;
    const tk_localcta::nvfp4_scale_t s_b_fp8 = compute_decoding_scaling_factor(block_amax, s_enc);
    const float s_b = static_cast<float>(s_b_fp8);
    const float s_dec = 1.0f / s_enc;
    const float coeff = fminf(1.0f / (s_b * s_dec), float_max);
    if (!(s_b > 0.0f) || !isfinite(s_b) || !(s_dec > 0.0f) || !isfinite(s_dec) ||
        !(coeff > 0.0f) || !isfinite(coeff)) {
        word0 = 0;
        word1 = 0;
        decode_scale_fp8 = tk_localcta::nvfp4_scale_t(0.0f);
        return;
    }
    decode_scale_fp8 = s_b_fp8;
    const uint64_t e03_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[0]);
    const uint64_t e47_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[2]);
    const uint64_t e03_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[4]);
    const uint64_t e47_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[6]);
    word0 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_0, e47_0, coeff);
    word1 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff);
}

__device__ __forceinline__ void fp4pv_quantize_scores_group_from_float2_fixed_tile_scale(
    const float2 *scores_group,
    uint32_t &word0,
    uint32_t &word1,
    float &decode_scale) {
    __align__(8) bf16_2 scores_group_bf[8];
    float block_amax = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
        const float2 pair_f = scores_group[jj];
        scores_group_bf[jj] = __float22bfloat162_rn(pair_f);
        block_amax = fmaxf(block_amax, fmaxf(fabsf(pair_f.x), fabsf(pair_f.y)));
    }

    if (!(block_amax > 0.0f)) {
        word0 = 0;
        word1 = 0;
        decode_scale = 0.0f;
        return;
    }

    float inv_amax;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_amax) : "f"(block_amax));
    const float coeff = inv_amax * 6.0f;
    decode_scale = block_amax * (1.0f / 6.0f);

    const uint64_t e03_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[0]);
    const uint64_t e47_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[2]);
    const uint64_t e03_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[4]);
    const uint64_t e47_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[6]);
    word0 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_0, e47_0, coeff);
    word1 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff);
}

__device__ __forceinline__ void fp4pv_quantize_scores_group_from_float2_fixed_tile_scale_known_amax(
    const float2 *scores_group,
    float block_amax,
    uint32_t &word0,
    uint32_t &word1,
    float &decode_scale) {
    if (!(block_amax > 0.0f)) {
        word0 = 0;
        word1 = 0;
        decode_scale = 0.0f;
        return;
    }

    __align__(8) bf16_2 scores_group_bf[8];
    #pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
        scores_group_bf[jj] = __float22bfloat162_rn(scores_group[jj]);
    }

    float inv_amax;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_amax) : "f"(block_amax));
    const float coeff = inv_amax * 6.0f;
    decode_scale = block_amax * (1.0f / 6.0f);

    const uint64_t e03_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[0]);
    const uint64_t e47_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[2]);
    const uint64_t e03_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[4]);
    const uint64_t e47_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[6]);
    word0 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_0, e47_0, coeff);
    word1 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff);
}

__device__ __forceinline__ void fp4pv_quantize_scores_group_payload_only(const bf16_2 *scores_group_bf,
                                                                         float s_enc,
                                                                         uint32_t &word0,
                                                                         uint32_t &word1) {
    using namespace transformer_engine::dispatch::nvfp4::quantization_and_transposition_SF;

    tk_localcta::IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    #pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
        const tk_localcta::IType2 pair =
            *reinterpret_cast<const tk_localcta::IType2 *>(&scores_group_bf[jj]);
        transformer_engine::ptx::abs_max_2x(amax_2x, amax_2x, pair);
    }

    const float block_amax = tk_localcta::get_amax_of_pair(amax_2x);
    if (!(block_amax > 0.0f) || !isfinite(block_amax)) {
        word0 = 0;
        word1 = 0;
        return;
    }

    constexpr float float_max = 3.4028235e+38f;
    const tk_localcta::nvfp4_scale_t s_b_fp8 = compute_decoding_scaling_factor(block_amax, s_enc);
    const float s_b = static_cast<float>(s_b_fp8);
    const float s_dec = 1.0f / s_enc;
    const float coeff = fminf(1.0f / (s_b * s_dec), float_max);
    if (!(s_b > 0.0f) || !isfinite(s_b) || !(s_dec > 0.0f) || !isfinite(s_dec) ||
        !(coeff > 0.0f) || !isfinite(coeff)) {
        word0 = 0;
        word1 = 0;
        return;
    }

    const uint64_t e03_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[0]);
    const uint64_t e47_0 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[2]);
    const uint64_t e03_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[4]);
    const uint64_t e47_1 = *reinterpret_cast<const uint64_t *>(&scores_group_bf[6]);
    word0 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_0, e47_0, coeff);
    word1 = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03_1, e47_1, coeff);
}

__device__ __forceinline__ tk_localcta::nvfp4_scale_t fp4pv_quantize_scores_group_scale_only(
    const bf16_2 *scores_group_bf,
    float s_enc) {
    using namespace transformer_engine::dispatch::nvfp4::quantization_and_transposition_SF;

    tk_localcta::IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    #pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
        const tk_localcta::IType2 pair =
            *reinterpret_cast<const tk_localcta::IType2 *>(&scores_group_bf[jj]);
        transformer_engine::ptx::abs_max_2x(amax_2x, amax_2x, pair);
    }

    const float block_amax = tk_localcta::get_amax_of_pair(amax_2x);
    if (!(block_amax > 0.0f) || !isfinite(block_amax)) {
        return tk_localcta::nvfp4_scale_t(0.0f);
    }
    return compute_decoding_scaling_factor(block_amax, s_enc);
}

template <typename PTile>
__device__ inline void fp4pv_store_quantized_scores_group(PTile &p_fp4_tile,
                                                          uint32_t word0,
                                                          uint32_t word1,
                                                          int global_row,
                                                          int quarter,
                                                          int group_in_quarter) {
    const int row = global_row & 127;
    const int col = quarter * 16 + group_in_quarter * 8;
    const uint32_t smem_base = __cvta_generic_to_shared(&p_fp4_tile.data[0]);
    const uint32_t smem_addr = PTile::idx(smem_base, {row, col});
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 0), "r"(word0));
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_addr + 4), "r"(word1));
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_and_scales(PTile &p_fp4_tile,
                                                             PScTile &p_sc_prepared_tile,
                                                             const float2 (&scores_reg)[64],
                                                             const float (&group_amax)[tk_localcta::SCALES_PER_CHUNK_X],
                                                             float amax_val,
                                                             int global_row) {
    static_assert(sizeof(tk_localcta::nvfp4_scale_t) == sizeof(uint8_t), "nvfp4_scale_t must be byte-sized");
    float s_enc = 1.0f;
    float sg_val = 1.0f;
    if constexpr (!FP4PV_USE_FIXED_P_TILE_SCALE) {
        s_enc = fp4pv_compute_tile_s_enc(amax_val);
        sg_val = amax_val * FP4PV_LOCALCTA_GLOBAL_SCALE_RCP;
    }
    float pending_scale0 = 0.0f;
    float pending_scale1 = 0.0f;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float2 *scores_q = &scores_reg[q * 16];
        uint32_t word0_0, word0_1, word1_0, word1_1;
        float scale0_f;
        float scale1_f;
        if constexpr (FP4PV_USE_FIXED_P_TILE_SCALE) {
            float scale0, scale1;
            if constexpr (FP4PV_FUSE_P_GROUP_AMAX) {
                fp4pv_quantize_scores_group_from_float2_fixed_tile_scale_known_amax(
                    scores_q + 0, group_amax[q * 2 + 0], word0_0, word0_1, scale0);
                fp4pv_quantize_scores_group_from_float2_fixed_tile_scale_known_amax(
                    scores_q + 8, group_amax[q * 2 + 1], word1_0, word1_1, scale1);
            } else {
                fp4pv_quantize_scores_group_from_float2_fixed_tile_scale(scores_q + 0, word0_0, word0_1, scale0);
                fp4pv_quantize_scores_group_from_float2_fixed_tile_scale(scores_q + 8, word1_0, word1_1, scale1);
            }
            scale0_f = scale0;
            scale1_f = scale1;
        } else {
            tk_localcta::nvfp4_scale_t scale0, scale1;
            fp4pv_quantize_scores_group_from_float2(scores_q + 0, s_enc, word0_0, word0_1, scale0);
            fp4pv_quantize_scores_group_from_float2(scores_q + 8, s_enc, word1_0, word1_1, scale1);
            scale0_f = static_cast<float>(scale0) * sg_val;
            scale1_f = static_cast<float>(scale1) * sg_val;
        }

        fp4pv_store_quantized_scores_group(p_fp4_tile, word0_0, word0_1, global_row, q, 0);
        fp4pv_store_quantized_scores_group(p_fp4_tile, word1_0, word1_1, global_row, q, 1);

        if ((q & 1) == 0) {
            pending_scale0 = scale0_f;
            pending_scale1 = scale1_f;
        } else {
            fp4pv_store_packed_scale_word_swizzled_f32(
                p_sc_prepared_tile, global_row, q >> 1,
                pending_scale0, pending_scale1, scale0_f, scale1_f);
        }
    }
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_pack_scores_to_stage_and_scales(PTile &p_fp4_tile,
                                                             PScTile &p_sc_prepared_tile,
                                                             const float2 (&scores_reg)[64],
                                                             float amax_val,
                                                             int global_row) {
    float group_amax[tk_localcta::SCALES_PER_CHUNK_X];
    #pragma unroll
    for (int g = 0; g < tk_localcta::SCALES_PER_CHUNK_X; ++g) {
        group_amax[g] = 0.0f;
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int si = q * 16 + jj;
            const int group_idx = q * 2 + (jj >> 3);
            group_amax[group_idx] = fmaxf(
                group_amax[group_idx],
                fmaxf(fabsf(scores_reg[si].x), fabsf(scores_reg[si].y)));
        }
    }
    fp4pv_pack_scores_to_stage_and_scales(
        p_fp4_tile, p_sc_prepared_tile, scores_reg, group_amax, amax_val, global_row);
}

template <typename PScTile>
__device__ inline void fp4pv_store_scales_from_localcta_scan_row(PScTile &p_sc_prepared_tile,
                                                                 const fp4pv_p_scan3d &scan,
                                                                 float amax_val,
                                                                 int global_row) {
    static_assert(sizeof(tk_localcta::nvfp4_scale_t) == sizeof(uint8_t), "nvfp4_scale_t must be byte-sized");
    float s_enc = FP4PV_LOCALCTA_GLOBAL_SCALE_NUM / amax_val;
    const float sg_val = amax_val * FP4PV_LOCALCTA_GLOBAL_SCALE_RCP;
    s_enc = fminf(s_enc, transformer_engine::detail::TypeExtrema<float>::max);
    if (amax_val == 0.0f || s_enc == 0.0f) {
        s_enc = 1.0f;
    }
    auto &raw_scales = *reinterpret_cast<fp4pv_p_raw_scales *>(&p_sc_prepared_tile.data[0]);
    const int row = global_row & 127;
    const int top_bottom = row / 64;
    const int local_row = row % 64;
    tk_localcta::nvfp4_scale_t row_scales[8];

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int tile_x = q >> 1;
        const int local_col_base = (q & 1) * 32;
        const int tile_idx = top_bottom * 2 + tile_x;
        const uint32_t smem_addr = __cvta_generic_to_shared(&scan[tile_idx][local_row][local_col_base]);
        __align__(8) bf16_2 scores_bf[16];
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            uint32_t packed0, packed1;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(packed0)
                         : "r"(smem_addr + static_cast<uint32_t>((i + 0) * sizeof(bf16_2))));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(packed1)
                         : "r"(smem_addr + static_cast<uint32_t>((i + 1) * sizeof(bf16_2))));
            reinterpret_cast<uint32_t &>(scores_bf[i + 0]) = packed0;
            reinterpret_cast<uint32_t &>(scores_bf[i + 1]) = packed1;
        }

        row_scales[q * 2 + 0] = tk_localcta::nvfp4_scale_t(
            static_cast<float>(fp4pv_quantize_scores_group_scale_only(&scores_bf[0], s_enc)) * sg_val);
        row_scales[q * 2 + 1] = tk_localcta::nvfp4_scale_t(
            static_cast<float>(fp4pv_quantize_scores_group_scale_only(&scores_bf[8], s_enc)) * sg_val);
    }

    fp4pv_store_packed_scale_word(raw_scales, global_row, 0, row_scales[0], row_scales[1], row_scales[2], row_scales[3]);
    fp4pv_store_packed_scale_word(raw_scales, global_row, 1, row_scales[4], row_scales[5], row_scales[6], row_scales[7]);
}

template <typename PTile>
__device__ inline void fp4pv_pack_scores_to_stage_payload_only(PTile &p_fp4_tile,
                                                               const float2 (&scores_reg)[64],
                                                               float amax_val,
                                                               int global_row) {
    const float s_enc = fp4pv_compute_tile_s_enc(amax_val);

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        __align__(8) bf16_2 scores_bf[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            scores_bf[jj] = __float22bfloat162_rn(scores_reg[q * 16 + jj]);
        }

        uint32_t word0_0, word0_1, word1_0, word1_1;
        fp4pv_quantize_scores_group_payload_only(&scores_bf[0], s_enc, word0_0, word0_1);
        fp4pv_quantize_scores_group_payload_only(&scores_bf[8], s_enc, word1_0, word1_1);

        fp4pv_store_quantized_scores_group(p_fp4_tile, word0_0, word0_1, global_row, q, 0);
        fp4pv_store_quantized_scores_group(p_fp4_tile, word1_0, word1_1, global_row, q, 1);
    }
}

template <typename PTile>
__device__ inline void fp4pv_pack_scores_to_stage_payload_from_localcta_scan_row(PTile &p_fp4_tile,
                                                                                 const fp4pv_p_scan3d &scan,
                                                                                 float amax_val,
                                                                                 int global_row) {
    float s_enc = FP4PV_LOCALCTA_GLOBAL_SCALE_NUM / amax_val;
    s_enc = fminf(s_enc, transformer_engine::detail::TypeExtrema<float>::max);
    if (amax_val == 0.0f || s_enc == 0.0f) {
        s_enc = 1.0f;
    }

    const int row = global_row & 127;
    const int top_bottom = row / 64;
    const int local_row = row % 64;

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int tile_x = q >> 1;
        const int local_col_base = (q & 1) * 32;
        const int tile_idx = top_bottom * 2 + tile_x;
        const uint32_t smem_addr = __cvta_generic_to_shared(&scan[tile_idx][local_row][local_col_base]);
        __align__(8) bf16_2 scores_bf[16];
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            uint32_t packed0, packed1;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(packed0)
                         : "r"(smem_addr + static_cast<uint32_t>((i + 0) * sizeof(bf16_2))));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(packed1)
                         : "r"(smem_addr + static_cast<uint32_t>((i + 1) * sizeof(bf16_2))));
            reinterpret_cast<uint32_t &>(scores_bf[i + 0]) = packed0;
            reinterpret_cast<uint32_t &>(scores_bf[i + 1]) = packed1;
        }

        uint32_t word0_0, word0_1, word1_0, word1_1;
        fp4pv_quantize_scores_group_payload_only(&scores_bf[0], s_enc, word0_0, word0_1);
        fp4pv_quantize_scores_group_payload_only(&scores_bf[8], s_enc, word1_0, word1_1);

        fp4pv_store_quantized_scores_group(p_fp4_tile, word0_0, word0_1, global_row, q, 0);
        fp4pv_store_quantized_scores_group(p_fp4_tile, word1_0, word1_1, global_row, q, 1);
    }
}

template <typename PScTile>
__device__ inline void fp4pv_zero_raw_scale_stage(PScTile &p_sc_prepared_tile, int tid) {
    static_assert((sizeof(PScTile) % sizeof(uint64_t)) == 0, "scale stage must be 64-bit aligned");
    uint32_t smem_addr = __cvta_generic_to_shared(&p_sc_prepared_tile.data[0]);
    constexpr int NUM_STORES = sizeof(PScTile) / sizeof(uint64_t);
    for (int i = tid; i < NUM_STORES; i += 128) {
        asm volatile("st.shared.b64 [%0], %1;"
                     :
                     : "r"(smem_addr + static_cast<uint32_t>(i * sizeof(uint64_t))),
                       "l"(0ull));
    }
    tk_localcta::subgroup_barrier_sync<128>();
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_zero_invalid_causal_groups(PTile &p_fp4_tile,
                                                        PScTile &p_sc_prepared_tile,
                                                        int global_row,
                                                        int valid_groups) {
    auto &raw_scales = *reinterpret_cast<fp4pv_p_raw_scales *>(&p_sc_prepared_tile.data[0]);
    const int row = global_row & 127;
    for (int g = valid_groups; g < tk_localcta::SCALES_PER_CHUNK_X; ++g) {
        raw_scales[row][g] = tk_localcta::nvfp4_scale_t(0.0f);
        fp4pv_store_quantized_scores_group(p_fp4_tile, 0u, 0u, global_row, g / 2, g & 1);
    }
}

template <typename PTile, typename PScTile>
__device__ inline void fp4pv_zero_invalid_causal_groups_swizzled(PTile &p_fp4_tile,
                                                                 PScTile &p_sc_prepared_tile,
                                                                 int global_row,
                                                                 int valid_groups) {
    for (int g = valid_groups; g < tk_localcta::SCALES_PER_CHUNK_X; ++g) {
        fp4pv_store_scale_byte_swizzled(p_sc_prepared_tile, global_row, g, tk_localcta::nvfp4_scale_t(0.0f));
        fp4pv_store_quantized_scores_group(p_fp4_tile, 0u, 0u, global_row, g / 2, g & 1);
    }
}

template <typename PTile>
__device__ inline void fp4pv_zero_invalid_causal_payload_groups(PTile &p_fp4_tile,
                                                                int global_row,
                                                                int valid_groups) {
    for (int g = valid_groups; g < tk_localcta::SCALES_PER_CHUNK_X; ++g) {
        fp4pv_store_quantized_scores_group(p_fp4_tile, 0u, 0u, global_row, g / 2, g & 1);
    }
}

template <int GROUP_THREADS>
__device__ inline float scan_tile_amax_group_padded(
    const fp4pv_p_scan3d &s_in,
    int buff_in,
    int tid
) {
    static_assert(GROUP_THREADS == 128, "scan_tile_amax_group_padded currently expects 128 consumer threads");
    const int lane = tid % THREADS_PER_WARP;
    const int bank_group = lane / tk_localcta::THREADS_PER_BANK;
    const int tid_y = tid / tk_localcta::THREADS_X_ROWWISE;
    const int tid_x = tid % tk_localcta::THREADS_X_ROWWISE;
    const int off_x = tid_x * tk_localcta::ELTS_PER_THREAD;
    float tile_max = 0.0f;

    #pragma unroll
    for (int it = 0; it < tk_localcta::ITERATIONS_NORMAL; ++it) {
        const int row = tid_y + it * tk_localcta::THREADS_Y_ROWWISE;
        #pragma unroll
        for (int w = 0; w < tk_localcta::WAVES; ++w) {
            const int sw = ((w + bank_group) * tk_localcta::PACK_SIZE) % tk_localcta::ELTS_PER_THREAD;
            __uint128_t elts = transformer_engine::ptx::ld_shared_b128(&s_in[buff_in][row][off_x + sw]);
            const tk_localcta::IType2 *pairs = reinterpret_cast<const tk_localcta::IType2 *>(&elts);
            #pragma unroll
            for (int e = 0; e < tk_localcta::PACK_SIZE / 2; ++e) {
                float a = __bfloat162float(__habs(pairs[e].x));
                float b = __bfloat162float(__habs(pairs[e].y));
                tile_max = fmaxf(tile_max, fmaxf(a, b));
            }
        }
    }
    return tile_max;
}

template <int GROUP_THREADS, bool ENCODE_CENTRIC = true>
__device__ inline void rowwise_scaling_group_padded(
    const fp4pv_p_scan3d &s_in,
    tk_localcta::fp4e2m1x2 *s_out_ptr,
    tk_localcta::nvfp4_scale_t *s_sf_rowwise_ptr,
    float s_enc,
    int stage_y,
    int stage_x,
    int buff_in,
    int buff_out,
    int tid
) {
    static_assert(GROUP_THREADS == 128, "rowwise_scaling_group_padded currently expects 128 consumer threads");
    using namespace transformer_engine::dispatch::nvfp4::quantization_and_transposition_SF;

    auto &s_out = *reinterpret_cast<tk_localcta::OType2x3D *>(s_out_ptr);
    auto &s_sf_rowwise = *reinterpret_cast<tk_localcta::ScalesType2D *>(s_sf_rowwise_ptr);

    const int thread_lane = tid % THREADS_PER_WARP;
    const int bank_group = thread_lane / tk_localcta::THREADS_PER_BANK;
    const int tid_y = tid / tk_localcta::THREADS_X_ROWWISE;
    const int tid_x = tid % tk_localcta::THREADS_X_ROWWISE;
    const int thread_offset_x = tid_x * tk_localcta::ELTS_PER_THREAD;

    const int sf_tid_y = tid_y;
    const int sf_tid_x = tid_x / tk_localcta::THREADS_PER_SCALE_ROWWISE;
    const bool sf_storing = (tid_x % tk_localcta::THREADS_PER_SCALE_ROWWISE == 0);
    const int stage_sc_y = sf_tid_y + stage_y * tk_localcta::TILE_DIM_Y;
    const int stage_sc_x = sf_tid_x + stage_x * tk_localcta::SCALES_PER_TILE_X;

    #pragma unroll
    for (int it = 0; it < tk_localcta::ITERATIONS_NORMAL; ++it) {
        const int row = tid_y + it * tk_localcta::THREADS_Y_ROWWISE;

        __align__(16) tk_localcta::IType2 r_in[tk_localcta::WAVES][tk_localcta::PACK_SIZE / 2];
        tk_localcta::IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < tk_localcta::WAVES; ++w) {
            const int sw = ((w + bank_group) * tk_localcta::PACK_SIZE) % tk_localcta::ELTS_PER_THREAD;
            __uint128_t &elts = *reinterpret_cast<__uint128_t *>(&r_in[w]);
            elts = transformer_engine::ptx::ld_shared_b128(&s_in[buff_in][row][thread_offset_x + sw]);
            #pragma unroll
            for (int e = 0; e < tk_localcta::PACK_SIZE / 2; ++e) {
                transformer_engine::ptx::abs_max_2x(amax_2x, amax_2x, r_in[w][e]);
            }
        }

        const float block_amax = tk_localcta::get_amax_of_pair(amax_2x);

        float coeff;
        tk_localcta::nvfp4_scale_t s_b_fp8;
        if constexpr (ENCODE_CENTRIC) {
            const tk_localcta::nvfp4_scale_t s_mult_fp8 =
                transformer_engine::dispatch::nvfp4::quantization_SF::compute_encoding_scaling_factor_nv(block_amax, s_enc);
            coeff = static_cast<float>(s_mult_fp8) * s_enc;
            s_b_fp8 = static_cast<tk_localcta::nvfp4_scale_t>(1.0f / static_cast<float>(s_mult_fp8));
        } else {
            s_b_fp8 = compute_decoding_scaling_factor(block_amax, s_enc);
            constexpr float float_max = 3.4028235e+38f;
            const float s_dec = 1.0f / s_enc;
            coeff = fminf(1.0f / (static_cast<float>(s_b_fp8) * s_dec), float_max);
        }

        if (sf_storing) {
            s_sf_rowwise[stage_sc_y + it * tk_localcta::THREADS_Y_ROWWISE][stage_sc_x] = s_b_fp8;
        }

        #pragma unroll
        for (int w = 0; w < tk_localcta::WAVES; ++w) {
            const uint64_t e03 = *reinterpret_cast<uint64_t *>(&r_in[w][0]);
            const uint64_t e47 = *reinterpret_cast<uint64_t *>(&r_in[w][2]);
            uint32_t out = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            const int sw = ((w + bank_group) * tk_localcta::PACK_SIZE) % tk_localcta::ELTS_PER_THREAD;
            transformer_engine::ptx::st_shared_b32(&s_out[buff_out][row][(sw + thread_offset_x) / 2], out);
        }
    }
}

template <int GROUP_THREADS>
__device__ inline void rowwise_scaling_scales_only_group_padded(
    const fp4pv_p_scan3d &s_in,
    tk_localcta::nvfp4_scale_t *s_sf_rowwise_ptr,
    float s_enc,
    int stage_y,
    int stage_x,
    int buff_in,
    int tid
) {
    static_assert(GROUP_THREADS == 128, "rowwise_scaling_scales_only_group_padded currently expects 128 consumer threads");
    using namespace transformer_engine::dispatch::nvfp4::quantization_and_transposition_SF;

    auto &s_sf_rowwise = *reinterpret_cast<tk_localcta::ScalesType2D *>(s_sf_rowwise_ptr);

    const int thread_lane = tid % THREADS_PER_WARP;
    const int bank_group = thread_lane / tk_localcta::THREADS_PER_BANK;
    const int tid_y = tid / tk_localcta::THREADS_X_ROWWISE;
    const int tid_x = tid % tk_localcta::THREADS_X_ROWWISE;
    const int thread_offset_x = tid_x * tk_localcta::ELTS_PER_THREAD;

    const int sf_tid_y = tid_y;
    const int sf_tid_x = tid_x / tk_localcta::THREADS_PER_SCALE_ROWWISE;
    const bool sf_storing = (tid_x % tk_localcta::THREADS_PER_SCALE_ROWWISE == 0);
    const int stage_sc_y = sf_tid_y + stage_y * tk_localcta::TILE_DIM_Y;
    const int stage_sc_x = sf_tid_x + stage_x * tk_localcta::SCALES_PER_TILE_X;

    #pragma unroll
    for (int it = 0; it < tk_localcta::ITERATIONS_NORMAL; ++it) {
        const int row = tid_y + it * tk_localcta::THREADS_Y_ROWWISE;

        __align__(16) tk_localcta::IType2 r_in[tk_localcta::WAVES][tk_localcta::PACK_SIZE / 2];
        tk_localcta::IType2 amax_2x = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};

        #pragma unroll
        for (int w = 0; w < tk_localcta::WAVES; ++w) {
            const int sw = ((w + bank_group) * tk_localcta::PACK_SIZE) % tk_localcta::ELTS_PER_THREAD;
            __uint128_t &elts = *reinterpret_cast<__uint128_t *>(&r_in[w]);
            elts = transformer_engine::ptx::ld_shared_b128(&s_in[buff_in][row][thread_offset_x + sw]);
            #pragma unroll
            for (int e = 0; e < tk_localcta::PACK_SIZE / 2; ++e) {
                transformer_engine::ptx::abs_max_2x(amax_2x, amax_2x, r_in[w][e]);
            }
        }

        const float block_amax = tk_localcta::get_amax_of_pair(amax_2x);
        tk_localcta::nvfp4_scale_t s_b_fp8 = tk_localcta::nvfp4_scale_t(0.0f);
        if (block_amax > 0.0f && isfinite(block_amax)) {
            s_b_fp8 = compute_decoding_scaling_factor(block_amax, s_enc);
        }

        if (sf_storing) {
            s_sf_rowwise[stage_sc_y + it * tk_localcta::THREADS_Y_ROWWISE][stage_sc_x] = s_b_fp8;
        }
    }
}

template <typename PTile>
__device__ inline void copy_localcta_tmp_to_fp4_tile(PTile &dst,
                                                      const tk_localcta::OType2x3D &tmp,
                                                      int tmp_buf, int stage_y, int stage_x,
                                                      int consumer_tid) {
    const uint64_t *src = reinterpret_cast<const uint64_t *>(&tmp[tmp_buf][0][0]);
    uint32_t smem_base = __cvta_generic_to_shared(&dst.data[0]);
    constexpr int BYTES_PER_STORE = 8;
    constexpr int STORES_PER_ROW = tk_localcta::BUFF_OUT_DIM_X / BYTES_PER_STORE;
    constexpr int TOTAL_STORES = tk_localcta::BUFF_OUT_DIM_Y * STORES_PER_ROW;
    for (int linear = consumer_tid; linear < TOTAL_STORES; linear += 128) {
        const int row = linear / STORES_PER_ROW;
        const int col = (linear % STORES_PER_ROW) * BYTES_PER_STORE;
        const int dst_row = stage_y * tk_localcta::TILE_DIM_Y + row;
        const int dst_col = stage_x * tk_localcta::BUFF_OUT_DIM_X + col;
        const uint32_t dst_addr = PTile::idx(smem_base, {dst_row, dst_col});
        asm volatile("st.shared.b64 [%0], %1;"
                     :
                     : "r"(dst_addr),
                       "l"(src[linear]));
    }
}

template <typename C, typename PScTile>
__device__ inline void finalize_localcta_p_scales_with_amax(PScTile &p_sc_prepared_tile,
                                                            float amax_val,
                                                            int consumer_tid) {
    const float sg_val = amax_val * FP4PV_LOCALCTA_GLOBAL_SCALE_RCP;
    auto *scales_ptr = reinterpret_cast<tk_localcta::nvfp4_scale_t *>(&p_sc_prepared_tile.data[0]);
    constexpr int TOTAL_SCALES =
        tk_localcta::LocalCTAConfig::CHUNK_DIM_Y * tk_localcta::SCALES_PER_CHUNK_X;
    tk_localcta::swizzle_scales_row_inplace_group<128>(
        scales_ptr,
        tk_localcta::SCALES_PER_CHUNK_X,
        consumer_tid);
    tk_localcta::scale_swizzled_scales_inplace_group<128>(
        scales_ptr,
        TOTAL_SCALES,
        sg_val,
        consumer_tid);
    tk_localcta::subgroup_barrier_sync<128>();
    publish_shared_backing();
    tk_localcta::subgroup_barrier_sync<128>();
}

template <typename PScTile>
__device__ inline void finalize_localcta_p_scales_direct(PScTile &p_sc_prepared_tile,
                                                         int consumer_tid) {
    (void)p_sc_prepared_tile;
    (void)consumer_tid;
}

template <typename C, typename PTile, typename PScTile>
__device__ inline void quantize_localcta_p_tile_with_amax(PTile &p_fp4_tile, PScTile &p_sc_prepared_tile,
                                                          fp4pv_p_scan3d &scan,
                                                          tk_localcta::OType2x3D &tmp_out,
                                                          float amax_val,
                                                          int consumer_tid) {
    float s_enc = FP4PV_LOCALCTA_GLOBAL_SCALE_NUM / amax_val;
    s_enc = fminf(s_enc, transformer_engine::detail::TypeExtrema<float>::max);
    if (amax_val == 0.0f || s_enc == 0.0f) {
        s_enc = 1.0f;
    }
    const float sg_val = amax_val * FP4PV_LOCALCTA_GLOBAL_SCALE_RCP;
    int tmp_buf = 0;
    auto *scales_ptr = reinterpret_cast<tk_localcta::nvfp4_scale_t *>(&p_sc_prepared_tile.data[0]);

    #pragma unroll
    for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
        const int stage_y = t / tk_localcta::TILES_X;
        const int stage_x = t % tk_localcta::TILES_X;
        rowwise_scaling_group_padded<128, true>(
            scan,
            reinterpret_cast<tk_localcta::fp4e2m1x2 *>(&tmp_out[0][0][0]),
            scales_ptr,
            s_enc,
            stage_y, stage_x,
            t, tmp_buf,
            consumer_tid);
        tk_localcta::subgroup_barrier_sync<128>();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tk_localcta::subgroup_barrier_sync<128>();
        copy_localcta_tmp_to_fp4_tile(p_fp4_tile, tmp_out, tmp_buf, stage_y, stage_x, consumer_tid);
        tk_localcta::subgroup_barrier_sync<128>();
        tmp_buf = (tmp_buf + 1) % tk_localcta::BUFFS_NUM_OUT;
    }

    tk_localcta::swizzle_scales_row_inplace_group<128>(
        scales_ptr,
        tk_localcta::SCALES_PER_CHUNK_X,
        consumer_tid);
    tk_localcta::scale_swizzled_scales_inplace_group<128>(
        scales_ptr,
        tk_localcta::LocalCTAConfig::CHUNK_DIM_Y * tk_localcta::SCALES_PER_CHUNK_X,
        sg_val,
        consumer_tid);
    tk_localcta::subgroup_barrier_sync<128>();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    tk_localcta::subgroup_barrier_sync<128>();
}

template <typename C, typename PScTile>
__device__ inline void quantize_localcta_p_scales_from_scan_with_amax(PScTile &p_sc_prepared_tile,
                                                                      fp4pv_p_scan3d &scan,
                                                                      tk_localcta::OType2x3D &tmp_out,
                                                                      float amax_val,
                                                                      int consumer_tid) {
    float s_enc = FP4PV_LOCALCTA_GLOBAL_SCALE_NUM / amax_val;
    s_enc = fminf(s_enc, transformer_engine::detail::TypeExtrema<float>::max);
    if (amax_val == 0.0f || s_enc == 0.0f) {
        s_enc = 1.0f;
    }

    #pragma unroll
    for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
        const int stage_y = t / tk_localcta::TILES_X;
        const int stage_x = t % tk_localcta::TILES_X;
        rowwise_scaling_group_padded<128, false>(
            scan,
            reinterpret_cast<tk_localcta::fp4e2m1x2 *>(&tmp_out[0][0][0]),
            reinterpret_cast<tk_localcta::nvfp4_scale_t *>(&p_sc_prepared_tile.data[0]),
            s_enc,
            stage_y,
            stage_x,
            t,
            0,
            consumer_tid);
    }
    tk_localcta::subgroup_barrier_sync<128>();
    finalize_localcta_p_scales_with_amax<C>(p_sc_prepared_tile, amax_val, consumer_tid);
}

template <typename C, typename PTile, typename PScTile>
__device__ inline void quantize_localcta_p_tile(PTile &p_fp4_tile, PScTile &p_sc_prepared_tile,
                                                fp4pv_p_scan3d &scan,
                                                tk_localcta::OType2x3D &tmp_out,
                                                int consumer_tid) {
    float cta_max = 0.0f;
    #pragma unroll
    for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
        cta_max = fmaxf(cta_max, scan_tile_amax_group_padded<128>(scan, t, consumer_tid));
    }

    const int lane = consumer_tid % 32;
    const int wid = consumer_tid / 32;
    __shared__ float warp_max[4];
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
    }
    if (lane == 0) {
        warp_max[wid] = cta_max;
    }
    tk_localcta::subgroup_barrier_sync<128>();

    if (wid == 0) {
        cta_max = (lane < 4) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int mask = 2; mask > 0; mask >>= 1) {
            cta_max = fmaxf(cta_max, __shfl_xor_sync(0xffffffff, cta_max, mask));
        }
        if (lane == 0) {
            warp_max[0] = cta_max;
        }
    }
    tk_localcta::subgroup_barrier_sync<128>();

    quantize_localcta_p_tile_with_amax<C>(p_fp4_tile, p_sc_prepared_tile, scan, tmp_out, warp_max[0], consumer_tid);
}

template <typename C, typename ScoresTT, typename QTile, typename KTile, typename QScTile, typename KScTile, typename QScTT, typename KScTT>
__device__ inline void issue_qk_chunked(ScoresTT &scores, const QTile &q, const KTile &k,
                                        const QScTile &q_sc_smem, const KScTile &k_sc_smem,
                                        QScTT &q_sc_tm, KScTT &k_sc_tm, semaphore &finished) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTT::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    static_assert(N == C::Nb,
                  "QK K operand must cover the complete logical N tile");
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; chunk++) {
        load_q_scale_chunk<C>(q_sc_tm, q_sc_smem, chunk);
        load_k_scale_chunk<C>(k_sc_tm, k_sc_smem, chunk);
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
__device__ inline void issue_qk_chunked_qsc_tmem(ScoresTT &scores, const QTile &q, const KTile &k,
                                                 QScTTFull &q_sc_tm_full, const KScTile &k_sc_smem,
                                                 KScTT &k_sc_tm, semaphore &finished) {
    using T_AB = typename QTile::T;
    using T_SAB = typename QScTTFull::T;
    using T_D = typename ScoresTT::T;
    constexpr int M = QTile::rows * C::CLUSTER_SIZE;
    constexpr int N = KTile::rows * C::CLUSTER_SIZE;
    static_assert(N == C::Nb,
                  "QK K operand must cover the complete logical N tile");
    constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>();
    st_descriptor<QTile, 0> q_desc(q);
    st_descriptor<KTile, 0> k_desc(k);
    #pragma unroll
    for (int chunk = 0; chunk < C::QK_SCALE_CHUNKS; chunk++) {
        auto q_sc_tm = q_sc_tm_full.template subtile<full_tt_fp8e4m3<16>>(chunk * 16);
        load_k_scale_chunk<C>(k_sc_tm, k_sc_smem, chunk);
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

// The mixed FP4-QK/BF16-PV family has two deliberately separate tensor
// topologies.  Existing B1/B2 keep native cta_group::2.  The Track-A child is
// a true single-CTA kernel and must never instantiate a 2CTA PV opcode.
template <typename C, typename D, typename A, typename B>
__device__ inline void issue_bf16_pv_mm(D &d, const A &a, const B &b) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_AB(d, a, b);
    } else {
        mm2_AB(d, a, b);
    }
}

template <typename C, typename D, typename A, typename B, typename Sem>
__device__ inline void issue_bf16_pv_mm(
    D &d, const A &a, const B &b, Sem &finished) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mm_AB(d, a, b, finished);
    } else {
        mm2_AB(d, a, b, finished);
    }
}

template <typename C, typename D, typename A, typename B>
__device__ inline void issue_bf16_pv_mma(D &d, const A &a, const B &b) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_AB(d, a, b);
    } else {
        mma2_AB(d, a, b);
    }
}

template <typename C, typename D, typename A, typename B, typename Sem>
__device__ inline void issue_bf16_pv_mma(
    D &d, const A &a, const B &b, Sem &finished) {
    if constexpr (C::CLUSTER_SIZE == 1) {
        mma_AB(d, a, b, finished);
    } else {
        mma2_AB(d, a, b, finished);
    }
}

template <typename C, typename OutputTT>
__device__ inline void zero_output_scratch(OutputTT &tt_output) {
    using out_rt_fl = rt_fl<C::Mb / 4, C::OUTPUT_CHUNK_COLS>;
    using out_tt_chunk = full_tt_fl<C::OUTPUT_CHUNK_COLS>;
    out_rt_fl zeros;
    warp::zero(zeros);
    #pragma unroll
    for (int chunk = 0; chunk < C::OUTPUT_CHUNKS; chunk++) {
        auto tt_chunk = tt_output.template subtile<out_tt_chunk>(chunk * C::OUTPUT_CHUNK_COLS);
        warpgroup::store_async(tt_chunk, zeros);
    }
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(warpgroup::groupid()+1);
}

template <typename C, typename OutputTT, typename OTile>
__device__ inline void load_output_backing(OutputTT &tt_output, OTile &o_backing) {
    using out_rt_fl = rt_fl<C::Mb / 4, C::OUTPUT_CHUNK_COLS>;
    using out_tt_chunk = full_tt_fl<C::OUTPUT_CHUNK_COLS>;
    #pragma unroll
    for (int chunk = 0; chunk < C::OUTPUT_CHUNKS; chunk++) {
        out_rt_fl out_fl;
        auto backing_view = o_backing.template subtile<32, C::OUTPUT_CHUNK_COLS>({warpgroup::warpid(), chunk});
        auto tt_chunk = tt_output.template subtile<out_tt_chunk>(chunk * C::OUTPUT_CHUNK_COLS);
        warp::load(out_fl, backing_view);
        warpgroup::store_async(tt_chunk, out_fl);
    }
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(warpgroup::groupid()+1);
}

template <typename C, typename OTile, typename OutputTT>
__device__ inline void spill_output_to_backing(OTile &o_backing, OutputTT &tt_output) {
    using out_rt_fl = rt_fl<C::Mb / 4, C::OUTPUT_CHUNK_COLS>;
    using out_tt_chunk = full_tt_fl<C::OUTPUT_CHUNK_COLS>;
    #pragma unroll
    for (int chunk = 0; chunk < C::OUTPUT_CHUNKS; chunk++) {
        out_rt_fl out_fl;
        auto tt_chunk = tt_output.template subtile<out_tt_chunk>(chunk * C::OUTPUT_CHUNK_COLS);
        warpgroup::load_async(out_fl, tt_chunk);
        tensor_load_wait();
        tensor_before_thread_sync();
        auto backing_view = o_backing.template subtile<32, C::OUTPUT_CHUNK_COLS>({warpgroup::warpid(), chunk});
        warp::store(backing_view, out_fl);
    }
    warpgroup::sync(warpgroup::groupid()+1);
}


template <typename C, typename OutputTT, typename OTile>
__device__ inline void load_output_backing_warp(OutputTT &tt_output, OTile &o_backing) {
    #pragma unroll
    for (int subtile = 0; subtile < C::Mb / 32; subtile++) {
        const int row = subtile * 32 + warp::laneid();
        #pragma unroll
        for (int col = 0; col < OutputTT::cols; col += 16) {
            float2 o_reg[8];
            #pragma unroll
            for (int ii = 0; ii < 8; ii++) {
                o_reg[ii].x = o_backing[{row, col + ii * 2 + 0}];
                o_reg[ii].y = o_backing[{row, col + ii * 2 + 1}];
            }
            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};}"
                :: "f"(o_reg[0].x), "f"(o_reg[0].y), "f"(o_reg[1].x), "f"(o_reg[1].y),
                   "f"(o_reg[2].x), "f"(o_reg[2].y), "f"(o_reg[3].x), "f"(o_reg[3].y),
                   "f"(o_reg[4].x), "f"(o_reg[4].y), "f"(o_reg[5].x), "f"(o_reg[5].y),
                   "f"(o_reg[6].x), "f"(o_reg[6].y), "f"(o_reg[7].x), "f"(o_reg[7].y),
                   "r"(tt_output.addr + ((subtile * 32) << 16) + col));
        }
    }
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(warpgroup::groupid()+1);
}


template <typename C, typename OTile, typename OutputTT>
__device__ inline void spill_output_to_backing_warp(OTile &o_backing, OutputTT &tt_output) {
    #pragma unroll
    for (int subtile = 0; subtile < C::Mb / 32; subtile++) {
        const int row = subtile * 32 + warp::laneid();
        #pragma unroll
        for (int col = 0; col < OutputTT::cols; col += 16) {
            float2 o_reg[8];
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                : "r"(tt_output.addr + ((subtile * 32) << 16) + col));
            tensor_load_wait();
            #pragma unroll
            for (int ii = 0; ii < 8; ii++) {
                o_backing[{row, col + ii * 2 + 0}] = o_reg[ii].x;
                o_backing[{row, col + ii * 2 + 1}] = o_reg[ii].y;
            }
        }
        warp::sync();
    }
}


template <typename C, typename OutputTT>
__device__ inline void zero_output_scratch_warp(OutputTT &tt_output) {
    #pragma unroll
    for (int subtile = 0; subtile < C::Mb / 32; subtile++) {
        #pragma unroll
        for (int col = 0; col < OutputTT::cols; col += 16) {
            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};}"
                :: "r"(tt_output.addr + ((subtile * 32) << 16) + col),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
        }
    }
    tensor_store_wait();
    tensor_after_thread_sync();
    warpgroup::sync(warpgroup::groupid()+1);
}

template <typename C, typename OutputTT>
__device__ inline void zero_output_scratch_issue_lane(OutputTT &tt_output) {
    #pragma unroll
    for (int subtile = 0; subtile < C::Mb / 32; subtile++) {
        #pragma unroll
        for (int col = 0; col < OutputTT::cols; col += 16) {
            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};}"
                :: "r"(tt_output.addr + ((subtile * 32) << 16) + col),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
        }
    }
    tensor_store_wait();
    tensor_after_thread_sync();
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    using G = globals<C>;
    const int cta_rank = cluster_ctarank();
    const int tiles_m = g.q.rows() / C::Mb;
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

    auto get_tile_idx = [&](int block_idx) -> int3 {
        int cluster_linear = block_idx / C::CLUSTER_SIZE;
        int bidhb  = cluster_linear / l2_major;
        int l2_mod = cluster_linear - bidhb * l2_major;
        int m_cluster, bidhb_residual;
        if (bidhb < num_hb_quotient) {
            m_cluster      = l2_mod / swizzle;
            bidhb_residual = l2_mod - m_cluster * swizzle;
        }
        else {
            int divisor    = (num_hb_remainder > 0) ? num_hb_remainder : 1;
            m_cluster      = l2_mod / divisor;
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
    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_scores_bf = tt<bf16, C::Mb, C::Nb>;
    using d_tt_scores_bf_1q = tt<bf16, C::Mb, C::Nb/4>;
    using v_quarter_tile = typename G::v_quarter_tile;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using max_vec_sv = sv_fl<C::Mb>;
    using lse_vec_sv = std::conditional_t<C::persistent, sv_fl<C::Mb>, sv_bf<C::Mb>>;
    using o_store_tile = typename G::o_tile;
    using o_backing_tile = typename G::o_backing_tile;
    using d_tt_q_sc = full_tt_fp8e4m3<C::Q_SC_TMEM_WIDTH>;
    using d_tt_k_sc = full_tt_fp8e4m3<C::K_SC_TMEM_WIDTH>;
    static_assert(G::DYNAMIC_SHARED_MEMORY <= MAX_SHARED_MEMORY - 1024);
    typename G::q_tile (&q_smem)[C::NUM_SOFTMAXXERS] = al.allocate<G::q_tile, C::NUM_SOFTMAXXERS>();
    typename G::q_sc_smem_storage &q_sc_smem = al.allocate<typename G::q_sc_smem_storage>();
    typename G::k_tile (&k_smem)[C::LOAD_STAGES] = al.allocate<G::k_tile, C::LOAD_STAGES>();
    typename G::v_tile (&v_smem)[C::LOAD_STAGES * G::V_SMEM_TILES_PER_STAGE] =
        al.allocate<G::v_tile, C::LOAD_STAGES * G::V_SMEM_TILES_PER_STAGE>();
    typename G::v_fp4_tile *v_fp4_stage = nullptr;
    typename G::v_sc_tile *v_sc_stage = nullptr;
    if constexpr (C::FP4_V_DEQUANT) {
        auto (&v_fp4_stage_storage)[C::LOAD_STAGES] =
            al.allocate<typename G::v_fp4_tile, C::LOAD_STAGES>();
        auto (&v_sc_stage_storage)[C::LOAD_STAGES] =
            al.allocate<typename G::v_sc_tile, C::LOAD_STAGES>();
        v_fp4_stage = &v_fp4_stage_storage[0];
        v_sc_stage = &v_sc_stage_storage[0];
    }
    o_backing_tile (&o_backing)[C::NUM_SOFTMAXXERS] = al.allocate<o_backing_tile, C::NUM_SOFTMAXXERS>();
    o_store_tile *o_smem_ptr = nullptr;
    if constexpr (C::persistent) {
        auto (&o_smem)[1] = al.allocate<o_store_tile, 1>();
        o_smem_ptr = &o_smem[0];
    }
    else {
        o_smem_ptr = nullptr;
    }
    typename G::k_sc_smem_storage &k_sc_smem = al.allocate<typename G::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] = al.allocate<typename G::k_sc_guard_slot, 1>();
    (void)k_sc_guard;
    __shared__ max_vec_sv max_vec_smem[C::NUM_SOFTMAXXERS];
    __shared__ lse_vec_sv lse_smem[C::NUM_SOFTMAXXERS];

    __shared__ semaphore q_arrived[C::NUM_SOFTMAXXERS], q_sc_arrived[C::NUM_SOFTMAXXERS], q_finished[C::NUM_SOFTMAXXERS];
    __shared__ semaphore k_arrived[C::LOAD_STAGES], k_sc_arrived[C::K_SCALE_SMEM_SLOTS], k_finished[C::LOAD_STAGES], k_sc_finished[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore v_arrived[C::LOAD_STAGES], v_finished[C::LOAD_STAGES];
    __shared__ semaphore v_fp4_stage_arrived[C::LOAD_STAGES];
    __shared__ semaphore v_sc_stage_arrived[C::LOAD_STAGES];
    __shared__ semaphore v_stage_reusable[C::LOAD_STAGES];
    __shared__ semaphore_array<
        C::SINGLE_CTA_FP4QK_BF16PV ? C::LOAD_STAGES * 4 : 0>
        v_quarter_arrived;
    __shared__ semaphore scores_arrived[C::NUM_SOFTMAXXERS], norm_scores_arrived[C::NUM_SOFTMAXXERS], norm_scores_quarter_arrived[3][C::NUM_SOFTMAXXERS];
    __shared__ semaphore score_copy_done[C::NUM_SOFTMAXXERS];
    __shared__ semaphore_array<
        C::BF16_PV_HYBRID ? C::NUM_SOFTMAXXERS : 0>
        stats_arrived;
    __shared__ semaphore corr_arrived[C::NUM_SOFTMAXXERS], tile_arrived[C::NUM_SOFTMAXXERS];
    __shared__ semaphore rescale_finished[C::NUM_SOFTMAXXERS];
    __shared__ semaphore p_tmem_reusable[C::NUM_SOFTMAXXERS];
    __shared__ semaphore_array<C::persistent ? 0 : C::NUM_SOFTMAXXERS> backing_reusable;
    __shared__ semaphore_array<C::persistent ? 0 : C::NUM_SOFTMAXXERS> pv_tmem_ready;
    __shared__ semaphore_array<C::persistent ? 0 : C::NUM_SOFTMAXXERS> pv_slot_ready_remote;
    __shared__ semaphore_array<C::persistent ? 0 : C::NUM_SOFTMAXXERS> pv_issue_done_remote;

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<typename G::q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<typename G::k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<typename G::k_sc_tile, dim::ROW>();
        if constexpr (!C::FP4_V_DEQUANT) {
            g.v.template prefetch_tma<typename G::v_tile, dim::DEPTH>();
        } else {
            g.v_fp4_tma.template prefetch_tma<typename G::v_fp4_tile, dim::ROW>();
        }
        g.o.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        #pragma unroll
        for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
            init_semaphore(q_arrived[i], 0, 1);
            init_semaphore(q_sc_arrived[i], 0, 1);
            init_semaphore(scores_arrived[i], 0, 1);
            init_semaphore(score_copy_done[i], 0, 4 * C::CLUSTER_SIZE);
            if constexpr (C::BF16_PV_HYBRID) {
                init_semaphore(stats_arrived[i], 0, 4);
            }
            init_semaphore(norm_scores_arrived[i], 0, C::persistent ? (4 + C::NUM_CORRECTORS) * C::CLUSTER_SIZE : 4 * C::CLUSTER_SIZE);
            #pragma unroll
            for (int q = 0; q < 3; q++) {
                init_semaphore(norm_scores_quarter_arrived[q][i], 0, 4 * C::CLUSTER_SIZE);
            }
            init_semaphore(corr_arrived[i], 0, 4);
            init_semaphore(tile_arrived[i], 0, 1);
            init_semaphore(rescale_finished[i], 0, 1);
            init_semaphore(p_tmem_reusable[i], 0, 1);
            init_semaphore(q_finished[i], 0, 1);
            if constexpr (!C::persistent) {
                init_semaphore(backing_reusable[i], 0, 1);
                init_semaphore(pv_tmem_ready[i], 0, 1);
                init_semaphore(pv_slot_ready_remote[i], 0, 1);
                init_semaphore(pv_issue_done_remote[i], 0, 1);
            }
        }
        #pragma unroll
        for (int i = 0; i < C::LOAD_STAGES; i++) {
            init_semaphore(k_arrived[i], 0, 1);
            init_semaphore(k_finished[i], 0, C::NUM_SOFTMAXXERS);
            // The normal B1 path receives one TMA transaction completion from
            // the pair.  B2 has one dequantizing producer warp per rank, so
            // rank0 waits for two ordinary arrivals before issuing group2 PV.
            init_semaphore(v_arrived[i], 0, C::FP4_V_DEQUANT ? C::CLUSTER_SIZE : 1);
            init_semaphore(v_finished[i], 0, C::NUM_SOFTMAXXERS);
            init_semaphore(v_fp4_stage_arrived[i], 0, 1);
            init_semaphore(v_sc_stage_arrived[i], 0, 1);
            init_semaphore(
                v_stage_reusable[i], 0,
                C::SINGLE_CTA_FP4QK_BF16PV ? 2 : 1);
            if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                #pragma unroll
                for (int quarter = 0; quarter < 4; ++quarter) {
                    init_semaphore(
                        v_quarter_arrived[i * 4 + quarter], 0, 1);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < C::K_SCALE_SMEM_SLOTS; i++) {
            init_semaphore(k_sc_arrived[i], 0, 1);
            init_semaphore(k_sc_finished[i], 0, 1);
        }
    }

    if constexpr (!C::persistent) {
        using backing_dtype = typename o_backing_tile::dtype;
        backing_dtype *backing = reinterpret_cast<backing_dtype *>(&o_backing[0].data[0]);
        constexpr int backing_elems = C::NUM_SOFTMAXXERS * C::Mb * C::Dvo;
        for (int idx = threadIdx.x; idx < backing_elems; idx += blockDim.x) {
            if constexpr (std::is_same_v<backing_dtype, bf16>) {
                backing[idx] = __float2bfloat16(0.0f);
            }
            else {
                backing[idx] = 0.0f;
            }
        }
    }
    __syncthreads();

    fp4pv_cluster_sync<C>();

    if (warpgroup::groupid() == C::TOTAL_WGS - 1) {  // producer warpgroup
        warpgroup::decrease_registers<128>();
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {  // tma warp
            if constexpr (C::persistent) {
                int k_idx = 0;
                int k_phase = 1;
                int v_idx = 0;
                int v_phase = 1;
                int q_phase = 1;
                for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
                    // q_sc_tile/k_sc_tile already span every Dqk/64 scale row
                    // for one logical attention head.  The native-group2
                    // hybrid ABI stores those rows as one contiguous head
                    // tile, so its TMA coordinate is the logical head index;
                    // multiplying by QK_SCALE_CHUNKS would stride by nine
                    // physical rows for Dqk=192 and read the wrong head.
                    static_assert(!C::BF16_PV_HYBRID ||
                                      C::QK_SCALE_CHUNKS == 3,
                                  "native-group2 hybrid expects one Dqk=192 scale tile per head");
                    const int qk_scale_head_coord =
                        C::BF16_PV_HYBRID
                            ? t_coord.z
                            : t_coord.z * C::QK_SCALE_CHUNKS;
                    #pragma unroll
                    for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
                        const int q_tile_idx = (t_coord.y + cta_rank * C::NUM_SOFTMAXXERS) + i;
                        tma::cluster::wait(q_finished[i], q_phase);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(
                            q_smem[i], g.q,
                            coord<typename G::q_tile>{
                                t_coord.x, t_coord.z, q_tile_idx, 0},
                            q_arrived[i], (uint16_t)(1 << cta_rank), 0);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(
                            slot_tile_at(q_sc_smem, i), g.q_sc,
                            coord<typename G::q_sc_tile>{
                                t_coord.x, q_tile_idx, qk_scale_head_coord, 0},
                            q_sc_arrived[i], (uint16_t)(1 << cta_rank), 0);
                    }
                    for (int idx = 0; idx < iters_per_task; idx++) {
                        const int k_tile_idx = idx * C::CLUSTER_SIZE + cta_rank;
                        // Prepared K scales keep the production S/64 depth:
                        // the two records for one K128 tile are identical
                        // complete scale pages.  Group1 consumes the even
                        // record; group2 consumes one record per rank.
                        const int k_scale_tile_idx = idx * 2 + cta_rank;
                        tma::cluster::wait(k_finished[k_idx], k_phase);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(
                            k_smem[k_idx], g.k,
                            coord<typename G::k_tile>{
                                t_coord.x, t_coord.z, k_tile_idx, 0},
                            k_arrived[k_idx], (uint16_t)(1 << cta_rank), 0);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(
                            slot_tile_at(k_sc_smem, k_idx), g.k_sc,
                            coord<typename G::k_sc_tile>{
                                t_coord.x, k_scale_tile_idx,
                                qk_scale_head_coord, 0},
                            k_sc_arrived[k_idx],
                            (uint16_t)(1 << cta_rank), 0);
                        k_idx++; if (k_idx == C::LOAD_STAGES) { k_idx = 0; k_phase ^= 1; }

                        if constexpr (!C::FP4_V_DEQUANT) {
                            tma::cluster::wait(v_finished[v_idx], v_phase);
                            if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                                // The BF16-V ceiling stages four real K32xN128
                                // descriptors.  Four completions contribute to
                                // one stage generation; the consumer expects
                                // their combined byte count before acquiring.
                                #pragma unroll
                                for (int quarter = 0; quarter < 4; ++quarter) {
                                    tma::expect_bytes(
                                        v_quarter_arrived[
                                            v_idx * 4 + quarter],
                                        sizeof(typename G::v_tile));
                                    fp4pv_load_async<
                                        dim::DEPTH, cache_policy::NORMAL, C>(
                                        v_smem[v_idx * 4 + quarter], g.v,
                                        coord<typename G::v_tile>{
                                            t_coord.x, idx * 4 + quarter,
                                            t_coord.z, 0},
                                        v_quarter_arrived[
                                            v_idx * 4 + quarter], 1, 0);
                                }
                            }
                            else {
                                fp4pv_load_async<
                                    dim::DEPTH, cache_policy::NORMAL, C>(
                                    v_smem[v_idx], g.v,
                                    coord<typename G::v_tile>{
                                        t_coord.x, idx, t_coord.z, cta_rank},
                                    v_arrived[v_idx],
                                    (uint16_t)(1 << cta_rank), 0);
                            }
                        } else {
                            // B2 preserves the authoritative MXFP4 V ABI and
                            // moves both payload and E8M0 scale page with TMA. The
                            // dequant warp releases this staging slot only
                            // after every lane has consumed both transactions.
                            wait(v_stage_reusable[v_idx], v_phase);
                            tma::expect_bytes(v_fp4_stage_arrived[v_idx],
                                              sizeof(typename G::v_fp4_tile));
                            const coord<typename G::v_fp4_tile> v_coord = {
                                t_coord.x, t_coord.z, cta_rank, idx};
                            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                                v_fp4_stage[v_idx], g.v_fp4_tma, v_coord,
                                v_fp4_stage_arrived[v_idx]);

                            constexpr int V_SCALE_LOAD_BYTES =
                                sizeof(typename G::v_sc_tile);
                            static_assert(V_SCALE_LOAD_BYTES == 32 * 16,
                                          "B2 must stage one production MXFP4 E8M0 page");
                            const int v_sc_record =
                                t_coord.z * g.v_scale_records_per_head;
                            const int64_t v_sc_offset =
                                ((static_cast<int64_t>(t_coord.x) *
                                      g.q.depth() * g.v_scale_records_per_head +
                                  v_sc_record) * (g.k.rows() / 128) + idx) *
                                V_SCALE_LOAD_BYTES;
                            tma::expect_bytes(v_sc_stage_arrived[v_idx],
                                              V_SCALE_LOAD_BYTES);
                            tma::load_async(
                                reinterpret_cast<void *>(&v_sc_stage[v_idx].data[0]),
                                const_cast<uint8_t *>(&g.v_sc_prepared[v_sc_offset]),
                                V_SCALE_LOAD_BYTES,
                                v_sc_stage_arrived[v_idx]);
                        }
                        v_idx++; if (v_idx == C::LOAD_STAGES) { v_idx = 0; v_phase ^= 1; }
                    }
                    q_phase ^= 1;
                }
            }
            else {
                int3 t_coord = get_tile_idx(blockIdx.x);
                int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
                #pragma unroll
                for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
                    const int q_tile_idx = (t_coord.y + cta_rank * C::NUM_SOFTMAXXERS) + i;
                    tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(q_smem[i], g.q, {t_coord.x, t_coord.z, q_tile_idx, 0}, q_arrived[i], (uint16_t)(1<<cta_rank), 0);
                }
                int k_idx = 0;
                int k_phase = 1;
                int k_sc_idx = 0;
                int k_sc_phase = 0;
                int v_idx = 0;
                int v_phase = 1;
                for (int idx = 0; idx < iters_per_task; idx++) {
                    const int k_tile_idx = idx * C::CLUSTER_SIZE + cta_rank;
                    tma::cluster::wait(k_finished[k_idx], k_phase);
                    tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(k_smem[k_idx], g.k, {t_coord.x, t_coord.z, k_tile_idx, 0}, k_arrived[k_idx], (uint16_t)(1<<cta_rank), 0);
                    wait(k_sc_finished[k_sc_idx], k_sc_phase ^ 1);
                    tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(slot_tile_at(k_sc_smem, k_sc_idx), g.k_sc, {t_coord.x, k_tile_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0}, k_sc_arrived[k_sc_idx], (uint16_t)(1<<cta_rank), 0);
                    k_idx++; if (k_idx == C::LOAD_STAGES) { k_idx = 0; k_phase ^= 1; }
                    k_sc_idx++;
                    if (k_sc_idx == C::K_SCALE_SMEM_SLOTS) {
                        k_sc_idx = 0;
                        k_sc_phase ^= 1;
                    }

                    tma::cluster::wait(v_finished[v_idx], v_phase);
                    tma::cluster::load_async<dim::DEPTH, cache_policy::NORMAL>(v_smem[v_idx], g.v, {t_coord.x, idx, t_coord.z, cta_rank}, v_arrived[v_idx], (uint16_t)(1<<cta_rank), 0);
                    v_idx++; if (v_idx == C::LOAD_STAGES) { v_idx = 0; v_phase ^= 1; }
                }
            }
        }
        else if (
            C::FP4_V_DEQUANT &&
            (warpgroup::warpid() == 2 ||
             (C::SINGLE_CTA_FP4QK_BF16PV &&
              warpgroup::warpid() == 1))) {
            // B2 owns V reconstruction in an otherwise idle producer warp.
            // Both CTA ranks reconstruct their rank-local N64 half, then
            // arrive at rank0's pair-wide event.  The native group2 PV issue
            // remains byte-for-byte shared with B1 below.
            static_assert(
                !C::FP4_V_DEQUANT ||
                    (C::persistent &&
                     (C::CLUSTER_SIZE == 1 || C::CLUSTER_SIZE == 2)),
                "FP4-V reconstruction requires the persistent one-softmaxer schedule");
            int v_idx = 0;
            int v_phase = 1;
            int v_stage_phase = 0;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                const int3 t_coord = get_tile_idx(cur_bid);
                const int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    if (warp::elect_leader()) {
                        tma::cluster::wait(v_finished[v_idx], v_phase);
                    }
                    // Every dequant lane performs the acquire waits for the
                    // two TMA transactions it will read; a leader-only wait
                    // plus Boolean broadcast would not transfer visibility.
                    tma::cluster::wait(v_fp4_stage_arrived[v_idx],
                                       v_stage_phase);
                    tma::cluster::wait(v_sc_stage_arrived[v_idx],
                                       v_stage_phase);
                    warp::sync();
                    if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                        // Two otherwise-idle producer warps own alternating
                        // K32 quarters.  Each quarter is visible and released
                        // independently; no CTA-wide quarter rendezvous is
                        // introduced.
                        const int first_quarter = warpgroup::warpid() - 1;
                        #pragma unroll
                        for (int quarter_offset = 0;
                             quarter_offset < 2; ++quarter_offset) {
                            const int quarter =
                                first_quarter + quarter_offset * 2;
                            dequantize_prepared_v_fp4_quarter_to_bf16_smem<C>(
                                v_smem[v_idx * 4 + quarter],
                                v_fp4_stage[v_idx],
                                v_sc_stage[v_idx], cta_rank, quarter);
                            warp::sync();
                            publish_shared_backing();
                            warp::sync();
                            if (warp::elect_leader()) {
                                arrive(v_quarter_arrived[
                                    v_idx * 4 + quarter]);
                            }
                        }
                    } else {
                        dequantize_prepared_v_fp4_to_bf16_smem<C>(
                            v_smem[v_idx], v_fp4_stage[v_idx],
                            v_sc_stage[v_idx], cta_rank);
                    }
                    // All lanes have consumed the TMA staging bytes before
                    // the producer may recycle this slot.
                    warp::sync();
                    if (warp::elect_leader()) {
                        arrive(v_stage_reusable[v_idx]);
                    }
                    if constexpr (!C::SINGLE_CTA_FP4QK_BF16PV) {
                        publish_shared_backing();
                        publish_cluster_shared_backing_if_needed<C>();
                        warp::sync();
                        if (warp::elect_leader()) {
                            tma::cluster::arrive(v_arrived[v_idx], 0);
                        }
                    }
                    v_idx++;
                    if (v_idx == C::LOAD_STAGES) {
                        v_idx = 0;
                        v_phase ^= 1;
                        v_stage_phase ^= 1;
                    }
                }
            }
        }
        else if (warpgroup::warpid() == 0 && (C::persistent ? (cta_rank == 0 && warp::elect_leader()) : true)) {  // warp 0 issues QK/PV and manages nonpersistent scratch copies
            const bool local_leader = (warpgroup::laneid() == 0);
            const bool issue_lane = C::persistent ? true : (cta_rank == 0 && local_leader);
            int k_idx = 0;
            int k_phase = 0;
            int v_idx = 0;
            int v_phase = 0;
            int norm_scores_phase = 0;
            int score_copy_phase = 0;
            if constexpr (C::persistent) {
                static_assert(C::NUM_SOFTMAXXERS == 1);
                d_tt_scores tt_score;
                d_tt_scores_bf tt_p;
                d_tt_outputs tt_output;
                using d_tt_q_sc = full_tt_fp8e4m3<C::Q_SC_TMEM_WIDTH>;
                using d_tt_k_sc_chunk = full_tt_fp8e4m3<16>;
                d_tt_q_sc q_sc_tm;
                d_tt_k_sc_chunk k_sc_tm;
                constexpr int SCALE_BASE = C::BF16_PV_HYBRID
                    ? C::HYBRID_SCALE_TMEM_BASE
                    : C::NUM_SOFTMAXXERS * (C::Nb + C::Dvo);
                constexpr int K_SCALE_BASE = SCALE_BASE + C::Q_SC_TMEM_WIDTH;
                tt_score = tm_alloc.template allocate<d_tt_scores>(0);
                tt_output = tm_alloc.template allocate<d_tt_outputs>(C::Nb);
                if constexpr (C::BF16_PV_HYBRID) {
                    tt_p = tm_alloc.template allocate<d_tt_scores_bf>(C::BF16_P_TMEM_BASE);
                } else {
                    tt_p = d_tt_scores_bf(tt_score.addr);
                }
                q_sc_tm = tm_alloc.template allocate<d_tt_q_sc>(SCALE_BASE);
                k_sc_tm = tm_alloc.template allocate<d_tt_k_sc_chunk>(K_SCALE_BASE);

                for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, task_num++) {
                    int3 t_coord = get_tile_idx(cur_bid);
                    int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
                    int q_phase = task_num & 1;
                    if (issue_lane) {
                        tma::cluster::expect_bytes(q_arrived[0], C::CLUSTER_SIZE * sizeof(G::q_tile));
                        tma::cluster::wait(q_arrived[0], q_phase);
                        tma::cluster::expect_bytes(q_sc_arrived[0], C::CLUSTER_SIZE * sizeof(G::q_sc_tile));
                        tma::cluster::wait(q_sc_arrived[0], q_phase);
                        stage_q_scale_tmem<C>(q_sc_tm, slot_tile_at(q_sc_smem, 0));
                    }
                    int k_slot = k_idx;
                    if (issue_lane) {
                        tma::cluster::expect_bytes(k_arrived[k_slot], C::CLUSTER_SIZE * sizeof(G::k_tile));
                        tma::cluster::wait(k_arrived[k_slot], k_phase);
                        tma::cluster::expect_bytes(k_sc_arrived[k_slot], C::CLUSTER_SIZE * sizeof(G::k_sc_tile));
                        tma::cluster::wait(k_sc_arrived[k_slot], k_phase);
                        issue_qk_chunked_qsc_tmem<C>(tt_score, q_smem[0], k_smem[k_slot], q_sc_tm, slot_tile_at(k_sc_smem, k_slot), k_sc_tm, k_finished[k_slot]);
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                    }
                    if (iters_per_task == 1) {
                        if (issue_lane) {
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
                        }
                    }
                    k_idx++; if (k_idx == C::LOAD_STAGES) { k_idx = 0; k_phase ^= 1; }

                    for (int idx = 0; idx < iters_per_task - 1; idx++) {
                        int v_slot = v_idx;
                        int k_slot = k_idx;
                        if constexpr (C::BF16_PV_HYBRID) {
                            // Score ownership is independent of V/P readiness.
                            // Release it and enqueue QK(idx+1) before waiting
                            // for the current V tile or BF16 P construction.
                            tma::cluster::wait(
                                score_copy_done[0], score_copy_phase);
                            score_copy_phase ^= 1;
                            if (issue_lane) {
                                tma::cluster::expect_bytes(
                                    k_arrived[k_slot],
                                    C::CLUSTER_SIZE * sizeof(G::k_tile));
                                tma::cluster::wait(k_arrived[k_slot], k_phase);
                                tma::cluster::expect_bytes(
                                    k_sc_arrived[k_slot],
                                    C::CLUSTER_SIZE * sizeof(G::k_sc_tile));
                                tma::cluster::wait(k_sc_arrived[k_slot], k_phase);
                                issue_qk_chunked_qsc_tmem<C>(
                                    tt_score, q_smem[0], k_smem[k_slot], q_sc_tm,
                                    slot_tile_at(k_sc_smem, k_slot), k_sc_tm,
                                    k_finished[k_slot]);
                                detail::tcgen05::commit<C::CLUSTER_SIZE>(
                                    scores_arrived[0]);
                            }
                            if (idx == iters_per_task - 2 && issue_lane) {
                                detail::tcgen05::commit<C::CLUSTER_SIZE>(
                                    q_finished[0]);
                            }
                        }
                        const int v_wait_phase = v_phase;
                        if constexpr (!C::FP4_V_DEQUANT &&
                                      !C::SINGLE_CTA_FP4QK_BF16PV) {
                            tma::cluster::expect_bytes(
                                v_arrived[v_slot],
                                C::CLUSTER_SIZE *
                                    G::V_SMEM_TILES_PER_STAGE *
                                    sizeof(G::v_tile));
                        }
                        if constexpr (!C::SINGLE_CTA_FP4QK_BF16PV) {
                            tma::cluster::wait(v_arrived[v_slot], v_phase);
                        }
                        v_idx++; if (v_idx == C::LOAD_STAGES) { v_idx = 0; v_phase ^= 1; }
                        const v_quarter_tile* v_q[4];
                        if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                            #pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                v_q[q] = &v_smem[v_slot * 4 + q];
                            }
                        }
                        else {
                            auto* v_base = reinterpret_cast<const char*>(
                                &v_smem[v_slot]);
                            #pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                v_q[q] = reinterpret_cast<
                                    const v_quarter_tile*>(
                                    v_base + q * sizeof(v_quarter_tile));
                            }
                        }
                        if constexpr (!C::BF16_PV_HYBRID) {
                            tma::cluster::wait(
                                norm_scores_arrived[0], norm_scores_phase);
                        }
                        if constexpr (C::BF16_PV_HYBRID &&
                                      !C::SINGLE_CTA_FP4QK_BF16PV) {
                            // P construction has its own readiness domain.
                            tma::cluster::wait(
                                norm_scores_arrived[0], norm_scores_phase);
                            #pragma unroll
                            for (int q = 1; q < 4; ++q) {
                                tma::cluster::wait(
                                    norm_scores_quarter_arrived[q - 1][0],
                                    norm_scores_phase);
                            }
                        }
                        if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                            tma::cluster::wait(
                                norm_scores_arrived[0], norm_scores_phase);
                            wait(v_quarter_arrived[v_slot * 4],
                                 v_wait_phase);
                        }
                        if (issue_lane) {
                            if (idx == 0) {
                                issue_bf16_pv_mm<C>(
                                    tt_output, d_tt_scores_bf_1q(tt_p.addr),
                                    *v_q[0]);
                            }
                            else {
                                issue_bf16_pv_mma<C>(
                                    tt_output, d_tt_scores_bf_1q(tt_p.addr),
                                    *v_q[0]);
                            }
                        }
                        #pragma unroll
                        for (int q = 1; q < 4; q++) {
                            if constexpr (!C::BF16_PV_HYBRID ||
                                          C::SINGLE_CTA_FP4QK_BF16PV) {
                                tma::cluster::wait(
                                    norm_scores_quarter_arrived[q - 1][0],
                                    norm_scores_phase);
                            }
                            if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                                wait(v_quarter_arrived[v_slot * 4 + q],
                                     v_wait_phase);
                            }
                            if (issue_lane) {
                                if (q == 3)
                                    issue_bf16_pv_mma<C>(
                                        tt_output,
                                        d_tt_scores_bf_1q(
                                            tt_p.addr + q * C::Nb / 8),
                                        *v_q[q], v_finished[v_slot]);
                                else
                                    issue_bf16_pv_mma<C>(
                                        tt_output,
                                        d_tt_scores_bf_1q(
                                            tt_p.addr + q * C::Nb / 8),
                                        *v_q[q]);
                            }
                        }
                        if constexpr (!C::BF16_PV_HYBRID) {
                            if (issue_lane) {
                            tma::cluster::expect_bytes(k_arrived[k_slot], C::CLUSTER_SIZE * sizeof(G::k_tile));
                            tma::cluster::wait(k_arrived[k_slot], k_phase);
                            tma::cluster::expect_bytes(k_sc_arrived[k_slot], C::CLUSTER_SIZE * sizeof(G::k_sc_tile));
                            tma::cluster::wait(k_sc_arrived[k_slot], k_phase);
                            issue_qk_chunked_qsc_tmem<C>(tt_score, q_smem[0], k_smem[k_slot], q_sc_tm, slot_tile_at(k_sc_smem, k_slot), k_sc_tm, k_finished[k_slot]);
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                            }
                            if (idx == iters_per_task - 2 && issue_lane) {
                                detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
                            }
                        }
                        if constexpr (C::BF16_PV_HYBRID) {
                            if (issue_lane) {
                                // One completion event owns the single copied
                                // P tile.  The next softmax iteration cannot
                                // overwrite it until this PV has consumed all
                                // four quarters.
                                detail::tcgen05::commit<C::CLUSTER_SIZE>(
                                    p_tmem_reusable[0]);
                            }
                        }
                        k_idx++; if (k_idx == C::LOAD_STAGES) { k_idx = 0; k_phase ^= 1; }
                        norm_scores_phase ^= 1;
                    }

                    int v_slot = v_idx;
                    if constexpr (!C::FP4_V_DEQUANT &&
                                  !C::SINGLE_CTA_FP4QK_BF16PV) {
                        tma::cluster::expect_bytes(
                            v_arrived[v_slot],
                            C::CLUSTER_SIZE *
                                G::V_SMEM_TILES_PER_STAGE *
                                sizeof(G::v_tile));
                    }
                    if constexpr (!C::SINGLE_CTA_FP4QK_BF16PV) {
                        tma::cluster::wait(v_arrived[v_slot], v_phase);
                    }

                    const v_quarter_tile* v_q[4];
                    if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            v_q[q] = &v_smem[v_slot * 4 + q];
                        }
                    }
                    else {
                        auto* v_base = reinterpret_cast<const char*>(
                            &v_smem[v_slot]);
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            v_q[q] = reinterpret_cast<
                                const v_quarter_tile*>(
                                v_base + q * sizeof(v_quarter_tile));
                        }
                    }
                    tma::cluster::wait(
                        norm_scores_arrived[0], norm_scores_phase);
                    if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                        wait(v_quarter_arrived[v_slot * 4], v_phase);
                    }
                    if (issue_lane) {
                        if (iters_per_task == 1) {
                            issue_bf16_pv_mm<C>(
                                tt_output, d_tt_scores_bf_1q(tt_p.addr),
                                *v_q[0]);
                        }
                        else {
                            issue_bf16_pv_mma<C>(
                                tt_output, d_tt_scores_bf_1q(tt_p.addr),
                                *v_q[0]);
                        }
                    }
                    #pragma unroll
                    for (int q = 1; q < 4; q++) {
                        tma::cluster::wait(
                            norm_scores_quarter_arrived[q - 1][0],
                            norm_scores_phase);
                        if constexpr (C::SINGLE_CTA_FP4QK_BF16PV) {
                            wait(v_quarter_arrived[v_slot * 4 + q],
                                 v_phase);
                        }
                        if (issue_lane) {
                            if (q == 3)
                                issue_bf16_pv_mma<C>(
                                    tt_output,
                                    d_tt_scores_bf_1q(
                                        tt_p.addr + q * C::Nb / 8),
                                    *v_q[q], v_finished[v_slot]);
                            else
                                issue_bf16_pv_mma<C>(
                                    tt_output,
                                    d_tt_scores_bf_1q(
                                        tt_p.addr + q * C::Nb / 8),
                                    *v_q[q]);
                        }
                    }
                    if (issue_lane) {
                        if constexpr (C::BF16_PV_HYBRID) {
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(
                                p_tmem_reusable[0]);
                        }
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                    }

                    v_idx++; if (v_idx == C::LOAD_STAGES) { v_idx = 0; v_phase ^= 1; }
                    norm_scores_phase ^= 1;
                }
            }
            else {
                d_tt_scores tt_scores[C::NUM_SOFTMAXXERS];
                using d_tt_q_sc = full_tt_fp8e4m3<C::Q_SC_TMEM_WIDTH>;
                using d_tt_k_sc_chunk = full_tt_fp8e4m3<16>;
                constexpr int SCORE0_BASE = 0;
                constexpr int SCORE1_BASE = C::Nb + C::Dvo;
                constexpr int SCALE_BASE = SCORE1_BASE + C::Nb;
                int3 t_coord = get_tile_idx(blockIdx.x);
                int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
                d_tt_q_sc q_sc_tm[C::NUM_SOFTMAXXERS];
                if (cta_rank == 0) {
                    #pragma unroll
                    for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                        q_sc_tm[qid] = tm_alloc.template allocate<d_tt_q_sc>(SCALE_BASE + qid * C::Q_SC_TMEM_WIDTH);
                    }
                }
                auto stage_q_scale = [&](int qid) {
                    if (!local_leader) return;
                    const int q_tile_idx = t_coord.y + cta_rank * C::NUM_SOFTMAXXERS + qid;
                    tma::cluster::load_async<dim::ROW, cache_policy::NORMAL>(
                        slot_tile_at(q_sc_smem, 0),
                        g.q_sc,
                        {t_coord.x, q_tile_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0},
                        q_sc_arrived[qid],
                        (uint16_t)(1 << cta_rank),
                        0
                    );
                };
                auto load_q_scale = [&](int qid) {
                    if (cta_rank != 0 || !local_leader) return;
                    auto &q_sc_stage = slot_tile_at(q_sc_smem, 0);
                    #pragma unroll
                    for (int ii = 0; ii < C::QK_SCALE_CHUNKS; ii++) {
                        auto q_sc_tm_sub = q_sc_tm[qid].template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                        auto &q_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&q_sc_stage.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(q_sc_tm_sub, q_sc_sm_sub);
                    }
                };
                #pragma unroll
                for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                    if (cta_rank == 1 && qid > 0 && local_leader) {
                        // Reuse the existing remote semaphore as a startup-only "slot reusable"
                        // handoff so CTA1 cannot overwrite the single Q_sc stage slot before
                        // CTA0 has finished the previous async TMEM scale load.
                        tma::cluster::wait(pv_issue_done_remote[qid - 1], 0);
                    }
                    warp::sync();
                    if (cta_rank == 0 && local_leader) {
                        tma::cluster::expect_bytes(q_arrived[qid], C::CLUSTER_SIZE * sizeof(G::q_tile));
                        tma::cluster::wait(q_arrived[qid], 0);
                        tma::cluster::expect_bytes(q_sc_arrived[qid], C::CLUSTER_SIZE * sizeof(G::q_sc_tile));
                    }
                    warp::sync();
                    stage_q_scale(qid);
                    if (cta_rank == 0 && local_leader) {
                        tma::cluster::wait(q_sc_arrived[qid], 0);
                    }
                    warp::sync();
                    load_q_scale(qid);
                    if (cta_rank == 0) {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                    }
                    warp::sync();
                    if (cta_rank == 0 && local_leader) {
                        tma::cluster::arrive(pv_issue_done_remote[qid], 1);
                    }
                    warp::sync();
                }
                if (cta_rank == 0) {
                    d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(C::Nb);
                    d_tt_k_sc_chunk k_sc_tm = tm_alloc.template allocate<d_tt_k_sc_chunk>(SCALE_BASE + C::NUM_SOFTMAXXERS * C::Q_SC_TMEM_WIDTH);
                    int scratch_phase[C::NUM_SOFTMAXXERS];
                    int remote_ready_phase[C::NUM_SOFTMAXXERS];
                    int k_sc_slot = 0;
                    int k_sc_arrived_phase[C::K_SCALE_SMEM_SLOTS] = {0};
                    #pragma unroll
                    for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                        scratch_phase[qid] = 0;
                        remote_ready_phase[qid] = 0;
                    }
                    #pragma unroll
                    for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                        tt_scores[qid] = tm_alloc.template allocate<d_tt_scores>(qid == 0 ? SCORE0_BASE : SCORE1_BASE);
                    }
                    int k_slot = k_idx;
                    if (issue_lane) {
                        tma::cluster::expect_bytes(k_arrived[k_slot], C::CLUSTER_SIZE * sizeof(G::k_tile));
                        tma::cluster::wait(k_arrived[k_slot], k_phase);
                        tma::cluster::expect_bytes(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(G::k_sc_tile));
                        tma::cluster::wait(k_sc_arrived[k_sc_slot], k_sc_arrived_phase[k_sc_slot]);
                        k_sc_arrived_phase[k_sc_slot] ^= 1;
                    }
                    #pragma unroll
                    for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                        if (issue_lane) {
                            issue_qk_chunked_qsc_tmem<C>(tt_scores[qid], q_smem[qid], k_smem[k_slot], q_sc_tm[qid], slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[k_slot]);
                            detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[qid]);
                            if (qid == C::NUM_SOFTMAXXERS - 1) {
                                arrive(k_sc_finished[k_sc_slot]);
                                tma::cluster::arrive(k_sc_finished[k_sc_slot], 1);
                            }
                        }
                    }
                    k_sc_slot++;
                    if (k_sc_slot == C::K_SCALE_SMEM_SLOTS) {
                        k_sc_slot = 0;
                    }
                    k_idx++; if (k_idx == C::LOAD_STAGES) { k_idx = 0; k_phase ^= 1; }

                    for (int idx = 0; idx < iters_per_task; idx++) {
                        const bool is_last_iter = (idx == iters_per_task - 1);
                        int v_slot = v_idx;
                        if (local_leader) {
                            tma::cluster::expect_bytes(v_arrived[v_slot], C::CLUSTER_SIZE * sizeof(G::v_tile));
                        }
                        tma::cluster::wait(v_arrived[v_slot], v_phase);
                        if (!is_last_iter) {
                            v_idx++;
                            if (v_idx == C::LOAD_STAGES) { v_idx = 0; v_phase ^= 1; }
                        }
                        int k_slot = k_idx;
                        auto* v_base = reinterpret_cast<const char*>(&v_smem[v_slot]);
                        const v_quarter_tile* v_q[4];
                        #pragma unroll
                        for (int q = 0; q < 4; q++)
                            v_q[q] = reinterpret_cast<const v_quarter_tile*>(v_base + q * sizeof(v_quarter_tile));
                        #pragma unroll
                        for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                            wait(backing_reusable[qid], scratch_phase[qid]);
                            scratch_phase[qid] ^= 1;
                            tma::cluster::wait(pv_slot_ready_remote[qid], remote_ready_phase[qid]);
                            remote_ready_phase[qid] ^= 1;
                            if (local_leader) {
                                tma::cluster::wait(norm_scores_arrived[qid], norm_scores_phase);
                            }
                            warp::sync();
                            tensor_before_thread_sync();
                            warp::sync();
                            if (issue_lane) {
                                if (idx == 0) {
                                    issue_bf16_pv_mm<C>(
                                        tt_output,
                                        d_tt_scores_bf_1q(
                                            tt_scores[qid].addr),
                                        *v_q[0]);
                                }
                                else {
                                    issue_bf16_pv_mma<C>(
                                        tt_output,
                                        d_tt_scores_bf_1q(
                                            tt_scores[qid].addr),
                                        *v_q[0]);
                                }
                            }
                            #pragma unroll
                            for (int q = 1; q < 4; q++) {
                                if (local_leader) {
                                    tma::cluster::wait(norm_scores_quarter_arrived[q-1][qid], norm_scores_phase);
                                }
                                warp::sync();
                                tensor_before_thread_sync();
                                warp::sync();
                                if (issue_lane) {
                                    if (q == 3)
                                        issue_bf16_pv_mma<C>(
                                            tt_output,
                                            d_tt_scores_bf_1q(
                                                tt_scores[qid].addr +
                                                q * C::Nb / 8),
                                            *v_q[q], v_finished[v_slot]);
                                    else
                                        issue_bf16_pv_mma<C>(
                                            tt_output,
                                            d_tt_scores_bf_1q(
                                                tt_scores[qid].addr +
                                                q * C::Nb / 8),
                                            *v_q[q]);
                                }
                            }
                            if (issue_lane) {
                                tensor_commit<C::CLUSTER_SIZE>(
                                    pv_tmem_ready[qid]);
                            }
                            if (!is_last_iter) {
                                if (qid == 0) {
                                    if (issue_lane) {
                                        tma::cluster::expect_bytes(k_arrived[k_slot], C::CLUSTER_SIZE * sizeof(G::k_tile));
                                        tma::cluster::wait(k_arrived[k_slot], k_phase);
                                        tma::cluster::expect_bytes(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(G::k_sc_tile));
                                        tma::cluster::wait(k_sc_arrived[k_sc_slot], k_sc_arrived_phase[k_sc_slot]);
                                        k_sc_arrived_phase[k_sc_slot] ^= 1;
                                    }
                                }
                                if (issue_lane) {
                                    issue_qk_chunked_qsc_tmem<C>(tt_scores[qid], q_smem[qid], k_smem[k_slot], q_sc_tm[qid], slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[k_slot]);
                                    detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[qid]);
                                    if (qid == C::NUM_SOFTMAXXERS - 1) {
                                        arrive(k_sc_finished[k_sc_slot]);
                                        tma::cluster::arrive(k_sc_finished[k_sc_slot], 1);
                                    }
                                }
                            }
                            else {
                                if (issue_lane) {
                                    detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[qid]);
                                }
                            }
                        }
                        if (!is_last_iter && idx == iters_per_task - 2) {
                            if (issue_lane) {
                                for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
                                    detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[i]);
                                }
                            }
                        }
                        if (!is_last_iter) {
                            k_idx++; if (k_idx == C::LOAD_STAGES) { k_idx = 0; k_phase ^= 1; }
                            k_sc_slot++;
                            if (k_sc_slot == C::K_SCALE_SMEM_SLOTS) {
                                k_sc_slot = 0;
                            }
                            norm_scores_phase ^= 1;
                        }
                    }
                }
                else {
                }
                if (FP4PV_DEBUG_TRAP_STAGE == 9 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
                    asm volatile("trap;");
                }
            }
        }
    }
    else if (warpgroup::groupid() == C::TOTAL_WGS - 2) {  // correction warpgroup
        warpgroup::decrease_registers<48>();
        static constexpr int CORR_TILE = 16;
        if constexpr (C::persistent) {
            static_assert(C::NUM_SOFTMAXXERS == 1);
            d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(C::Nb);
            int end_phase = 0;
            int corr_phase = 0;
            int corr_pv_phase = 0;
            int stats_phase = 0;
            warpgroup::tma::cluster::arrive(norm_scores_arrived[0], 0);
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                const float scale_log2 = qk_scale_log2(g, t_coord.x, t_coord.z);
                int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
                wait(corr_arrived[0], corr_phase);
                warpgroup::arrive(rescale_finished[0]);
                corr_phase ^= 1;
                for (int idx = 1; idx < iters_per_task; idx++) {
                    wait(corr_arrived[0], corr_phase);
                    if constexpr (C::BF16_PV_HYBRID) {
                        // Early QK may complete before the previous PV.  The
                        // correction WG therefore acquires the same PV-backed
                        // completion used to reclaim P before touching the
                        // live output accumulator.
                        wait(p_tmem_reusable[0], corr_pv_phase);
                        corr_pv_phase ^= 1;
                    }
                    float correction = stats_load(max_vec_smem[0], warpgroup::laneid());
                    bool needs_rescale = __any_sync(0xFFFFFFFF, correction < 1.0f);
                    if (needs_rescale) {
                        float2 corr_2 = {correction, correction};
                        #pragma unroll
                        for (int col = 0; col < C::Dvo; col += CORR_TILE) {
                            float2 o_reg[CORR_TILE / 2];
                            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                                : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                            tensor_load_wait();
                            #pragma unroll
                            for (int ii = 0; ii < CORR_TILE / 2; ii++) {
                                o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
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
                    warpgroup::sync(warpgroup::groupid()+1);
                    warpgroup::tma::cluster::arrive(norm_scores_arrived[0], 0);
                    warpgroup::arrive(rescale_finished[0]);
                    corr_phase ^= 1;
                }
                // Final row statistics are a distinct publication generation
                // for the native-group2 hybrid.  Per-iteration correction
                // vectors continue to use corr_arrived unchanged.
                if constexpr (C::BF16_PV_HYBRID) {
                    wait(stats_arrived[0], stats_phase);
                    stats_phase ^= 1;
                } else {
                    wait(corr_arrived[0], corr_phase);
                }
                float row_sum = stats_load(max_vec_smem[0], warpgroup::laneid());
                float row_max = stats_load(lse_smem[0], warpgroup::laneid());
                warpgroup::arrive(rescale_finished[0]);
                warpgroup::tma::store_async_read_wait<0>();
                bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
                float inv_norm_s;
                asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_norm_s) : "f"(row_invalid ? 1.0f : row_sum));
                float2 inv_norm = {inv_norm_s, inv_norm_s};
                constexpr float LN2 = 0.693147180559945f;
                float lse_val;
                if (!row_invalid) {
                    float log2_row_sum;
                    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                    lse_val = (row_max * scale_log2 + log2_row_sum) * LN2;
                }
                else {
                    lse_val = base_types::constants<float>::neg_infty();
                }
                int m_tile = t_coord.y + cta_rank * C::NUM_SOFTMAXXERS;
                wait(tile_arrived[0], end_phase);
                warpgroup::sync(warpgroup::groupid()+1);
                constexpr int SUBTILE_COLS = o_store_tile::swizzle_bytes / sizeof(bf16);
                constexpr uint32_t SWIZZLE_MASK = o_store_tile::swizzle_bytes * 8 - 1;
                uint32_t base_addr = __cvta_generic_to_shared(&o_smem_ptr->data[0]);
                uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);

                for (int col = 0; col < C::Dvo; col += CORR_TILE) {
                    float2 o_reg[CORR_TILE / 2];
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                        : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                          "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                          "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                          "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                        : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                    uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16))
                                      + row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
                    #pragma unroll
                    for (int i = 0; i < CORR_TILE / 2; i += 2) {
                        bf16_2 tmp0 = __float22bfloat162_rn(__fmul2_rn(o_reg[i], inv_norm));
                        bf16_2 tmp1 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 1], inv_norm));
                        uint32_t addr = row_base + i * 4;
                        uint32_t swizzled_addr = addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4);
                        uint64_t packed = (uint64_t)(*(uint32_t*)&tmp0) | ((uint64_t)(*(uint32_t*)&tmp1) << 32);
                        asm volatile("st.shared.b64 [%0], %1;" :: "r"(swizzled_addr), "l"(packed));
                    }
                }
                warpgroup::sync(warpgroup::groupid()+1);
                warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, *o_smem_ptr, {t_coord.x, t_coord.y + cta_rank * C::NUM_SOFTMAXXERS, t_coord.z, 0});
                warpgroup::tma::cluster::arrive(norm_scores_arrived[0], 0);

                g.lse[{t_coord.x, t_coord.z, 0, m_tile * C::Mb + warpgroup::laneid()}] = lse_val;
                if constexpr (!C::BF16_PV_HYBRID) {
                    corr_phase ^= 1;
                }
                end_phase ^= 1;
            }
        }
        else {
            d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(C::Nb);
            int3 t_coord = get_tile_idx(blockIdx.x);
            int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
            int corr_phase = 0;
            int pv_phase[C::NUM_SOFTMAXXERS];
            #pragma unroll
            for (int i = 0; i < C::NUM_SOFTMAXXERS; i++) {
                pv_phase[i] = 0;
            }
            const bool cluster_leader = (warpgroup::warpid() == 0 && warp::elect_leader());
            for (int idx = 0; idx < iters_per_task; idx++) {
                const bool is_last_iter = (idx == iters_per_task - 1);
                #pragma unroll
                for (int qid = 0; qid < C::NUM_SOFTMAXXERS; qid++) {
                    if (idx == 0) {
                        zero_output_scratch<C>(tt_output);
                    }
                    else {
                        wait(corr_arrived[qid], corr_phase);
                        load_output_backing<C>(tt_output, o_backing[qid]);
                        float correction = stats_load(max_vec_smem[qid], warpgroup::laneid());
                        bool needs_rescale = __any_sync(0xFFFFFFFF, correction < 1.0f);
                        if (needs_rescale) {
                            float2 corr_2 = {correction, correction};
                            #pragma unroll
                            for (int col = 0; col < C::Dvo; col += CORR_TILE) {
                                float2 o_reg[CORR_TILE / 2];
                                asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                                    : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                                      "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                                      "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                                      "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                                    : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                                tensor_load_wait();
                                #pragma unroll
                                for (int ii = 0; ii < CORR_TILE / 2; ii++) {
                                    o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
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
                        warpgroup::arrive(rescale_finished[qid]);
                    }

                    if (idx == 0) {
                        wait(corr_arrived[qid], corr_phase);
                        warpgroup::arrive(rescale_finished[qid]);
                    }

                    if (cta_rank == 0) {
                        warpgroup::arrive(backing_reusable[qid]);
                    }
                    else if (cluster_leader) {
                        tma::cluster::arrive(pv_slot_ready_remote[qid], 0);
                    }

                    wait(pv_tmem_ready[qid], pv_phase[qid]);
                    pv_phase[qid] ^= 1;

                    if (!is_last_iter) {
                        spill_output_to_backing<C>(o_backing[qid], tt_output);
                        warpgroup::sync(warpgroup::groupid()+1);
                        publish_shared_backing();
                    }
                    else {
                        float row_sum = stats_load(max_vec_smem[qid], warpgroup::laneid());
                        float row_max = stats_load(lse_smem[qid], warpgroup::laneid());
                        bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
                        float inv_norm_s;
                        asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_norm_s) : "f"(row_invalid ? 1.0f : row_sum));
                        float2 inv_norm = {inv_norm_s, inv_norm_s};
                        wait(tile_arrived[qid], 0);
                        warpgroup::tma::store_async_read_wait<0>();
                        warpgroup::sync(warpgroup::groupid()+1);
                        constexpr int SUBTILE_COLS = o_store_tile::swizzle_bytes / sizeof(bf16);
                        constexpr uint32_t SWIZZLE_MASK = o_store_tile::swizzle_bytes * 8 - 1;
                        auto &o_store_alias = *reinterpret_cast<o_store_tile *>(&o_backing[qid]);
                        uint32_t base_addr = __cvta_generic_to_shared(&o_store_alias.data[0]);
                        uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);

                        for (int col = 0; col < C::Dvo; col += CORR_TILE) {
                            float2 o_reg[CORR_TILE / 2];
                            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                                : "=f"(o_reg[0].x), "=f"(o_reg[0].y), "=f"(o_reg[1].x), "=f"(o_reg[1].y),
                                  "=f"(o_reg[2].x), "=f"(o_reg[2].y), "=f"(o_reg[3].x), "=f"(o_reg[3].y),
                                  "=f"(o_reg[4].x), "=f"(o_reg[4].y), "=f"(o_reg[5].x), "=f"(o_reg[5].y),
                                  "=f"(o_reg[6].x), "=f"(o_reg[6].y), "=f"(o_reg[7].x), "=f"(o_reg[7].y)
                                : "r"(tt_output.addr + ((warpgroup::warpid() * 32) << 16) + col));
                            uint32_t row_base = base_addr + (col / SUBTILE_COLS) * (C::Mb * SUBTILE_COLS * sizeof(bf16))
                                              + row_offset + (col % SUBTILE_COLS) * sizeof(bf16);
                            #pragma unroll
                            for (int i = 0; i < CORR_TILE / 2; i++) {
                                bf16_2 tmp = __float22bfloat162_rn(__fmul2_rn(o_reg[i], inv_norm));
                                uint32_t addr = row_base + i * 4;
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4)), "r"(*(uint32_t*)&tmp));
                            }
                        }
                        warpgroup::sync(warpgroup::groupid()+1);
                        warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, o_store_alias, {t_coord.x, (t_coord.y+cta_rank*C::NUM_SOFTMAXXERS)+qid, t_coord.z, 0});

                        const float SCALE_LOG2 = qk_scale_log2(g, t_coord.x, t_coord.z);
                        constexpr float LN2 = 0.693147180559945f;
                        float lse_val;
                        if (!row_invalid) {
                            float log2_row_sum;
                            asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                            lse_val = (row_max * SCALE_LOG2 + log2_row_sum) * LN2;
                        }
                        else {
                            lse_val = base_types::constants<float>::neg_infty();
                        }
                        int m_tile = t_coord.y + cta_rank * C::NUM_SOFTMAXXERS + qid;
                        g.lse[{t_coord.x, t_coord.z, 0, m_tile * C::Mb + warpgroup::laneid()}] = lse_val;
                    }
                }
                corr_phase ^= 1;
            }
        }
    }
    else if (warpgroup::groupid() < C::NUM_SOFTMAXXERS) {  // softmax warpgroups
        warpgroup::increase_registers<
            C::SINGLE_CTA_FP4QK_BF16PV ? 192 : 168>();
        d_tt_scores tt_scores;
        d_tt_scores_bf tt_scores_bf;
        if constexpr (C::persistent) {
            tt_scores = tm_alloc.template allocate<d_tt_scores>(warpgroup::groupid() * (C::Nb + C::Dvo));
            if constexpr (C::BF16_PV_HYBRID) {
                tt_scores_bf = tm_alloc.template allocate<d_tt_scores_bf>(
                    C::BF16_P_TMEM_BASE);
            } else {
                tt_scores_bf = d_tt_scores_bf(tt_scores.addr);
            }
        }
        else {
            constexpr int SCORE1_BASE = C::Nb + C::Dvo;
            tt_scores = tm_alloc.template allocate<d_tt_scores>(warpgroup::groupid() == 0 ? 0 : SCORE1_BASE);
        }
        if constexpr (!C::persistent) {
            tt_scores_bf = d_tt_scores_bf(tt_scores.addr);
        }
        int scores_phase = 0;
        int rescale_phase = 1;
        int p_reuse_phase = 0;
        uint32_t score_tt_base = tt_scores.addr + ((warpgroup::warpid() * 32) << 16);
        uint32_t score_bf_base = tt_scores_bf.addr + ((warpgroup::warpid() * 32) << 16);
        if constexpr (C::persistent) {
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
                int m_tile = t_coord.y + cta_rank * C::NUM_SOFTMAXXERS + warpgroup::groupid();
                const float SCALE_LOG2 = qk_scale_log2(g, t_coord.x, t_coord.z);
                const bool ZERO_SCALE = (SCALE_LOG2 == 0.0f);
                float row_sum = 0.0f;
                float row_max = base_types::constants<float>::neg_infty();
                wait(rescale_finished[warpgroup::groupid()], rescale_phase);
                rescale_phase ^= 1;
                for (int idx = 0; idx < iters_per_task; idx++) {
                    float2 scores_reg[C::Nb / 2];  // each thread holds one row of tmem
                    wait(scores_arrived[warpgroup::groupid()], scores_phase);
                    #pragma unroll
                    for (int ii = 0; ii < C::Nb / 32; ii++) {
                        asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                                : "=f"(scores_reg[ii * 16 + 0].x), "=f"(scores_reg[ii * 16 + 0].y), "=f"(scores_reg[ii * 16 + 1].x), "=f"(scores_reg[ii * 16 + 1].y), "=f"(scores_reg[ii * 16 + 2].x), "=f"(scores_reg[ii * 16 + 2].y), "=f"(scores_reg[ii * 16 + 3].x), "=f"(scores_reg[ii * 16 + 3].y),
                                  "=f"(scores_reg[ii * 16 + 4].x), "=f"(scores_reg[ii * 16 + 4].y), "=f"(scores_reg[ii * 16 + 5].x), "=f"(scores_reg[ii * 16 + 5].y), "=f"(scores_reg[ii * 16 + 6].x), "=f"(scores_reg[ii * 16 + 6].y), "=f"(scores_reg[ii * 16 + 7].x), "=f"(scores_reg[ii * 16 + 7].y),
                                  "=f"(scores_reg[ii * 16 + 8].x), "=f"(scores_reg[ii * 16 + 8].y), "=f"(scores_reg[ii * 16 + 9].x), "=f"(scores_reg[ii * 16 + 9].y), "=f"(scores_reg[ii * 16 + 10].x), "=f"(scores_reg[ii * 16 + 10].y), "=f"(scores_reg[ii * 16 + 11].x), "=f"(scores_reg[ii * 16 + 11].y),
                                  "=f"(scores_reg[ii * 16 + 12].x), "=f"(scores_reg[ii * 16 + 12].y), "=f"(scores_reg[ii * 16 + 13].x), "=f"(scores_reg[ii * 16 + 13].y), "=f"(scores_reg[ii * 16 + 14].x), "=f"(scores_reg[ii * 16 + 14].y), "=f"(scores_reg[ii * 16 + 15].x), "=f"(scores_reg[ii * 16 + 15].y)
                                : "r"(score_tt_base + ii * 32));
                    }
                    if constexpr (C::BF16_PV_HYBRID) {
                        // All score values are now lane-owned registers.  The
                        // explicit load acquire precedes the pair-wide release
                        // that allows rank0 to enqueue QK for idx+1.  P
                        // construction and its distinct ready events follow.
                        tensor_load_wait();
                        tensor_after_thread_sync();
#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
                        if constexpr (C::SELECTED_ROW_TRACE) {
                            fp4pv_native_group2_selected_row_record_raw(
                                12,
                                t_coord,
                                m_tile,
                                idx,
                                iters_per_task,
                                cta_rank,
                                score_tt_base,
                                C::HYBRID_SCALE_TMEM_BASE,
                                C::HYBRID_SCALE_TMEM_BASE +
                                    C::Q_SC_TMEM_WIDTH,
                                scores_reg);
                            // The selected row is lane 95 (reader warp 2).
                            // Hold that warp's leader until capture completes;
                            // rank0 cannot observe count 8 and overwrite the
                            // score bank before this warp publishes credit.
                            warp::sync();
                        }
#endif
#if TK_FA4_NATIVE_GROUP2_RAW_QK_DIAG_BUILD
                        fp4pv_native_group2_record_raw_qk_row(
                            C::FP4_V_DEQUANT ? 3 : 2,
                            t_coord,
                            m_tile,
                            idx,
                            iters_per_task,
                            cta_rank,
                            score_tt_base,
                            C::HYBRID_SCALE_TMEM_BASE,
                            C::HYBRID_SCALE_TMEM_BASE + C::Q_SC_TMEM_WIDTH,
                            scores_reg);
#endif
                        warp::tma::cluster::arrive(
                            score_copy_done[warpgroup::groupid()], 0);
                    }

                    if (idx >= m_tile) {
                        int causal_col = (idx > m_tile) ? -1 : (int)warpgroup::laneid();
                        #pragma unroll
                        for (int k = 0; k < C::Nb / 2; k++) {
                            if (k * 2 > causal_col) scores_reg[k].x = base_types::constants<float>::neg_infty();
                            if (k * 2 + 1 > causal_col) scores_reg[k].y = base_types::constants<float>::neg_infty();
                        }
                    }

                    float row_max_old = row_max;
                    // calculate row max
                    float lm0 = base_types::constants<float>::neg_infty(), lm1 = base_types::constants<float>::neg_infty(), lm2 = base_types::constants<float>::neg_infty(), lm3 = base_types::constants<float>::neg_infty();
                    float lm4 = base_types::constants<float>::neg_infty(), lm5 = base_types::constants<float>::neg_infty(), lm6 = base_types::constants<float>::neg_infty(), lm7 = base_types::constants<float>::neg_infty();
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

                    // give rescale factor to correction warpgroup
                    if (idx >= m_tile || idx > 0) {
                        if (!ZERO_SCALE) {
                            constexpr float RESCALE_THRESHOLD = 8.f;
                            float acc_scale_log2 = (row_max_old - row_max) * SCALE_LOG2;
                            if (acc_scale_log2 >= -RESCALE_THRESHOLD) {
                                row_max = row_max_old;
                                acc_scale = 1.0f;
                            }
                            else {
                                acc_scale = exp2f(acc_scale_log2);
                            }
                        }
                        stats_store(max_vec_smem[warpgroup::groupid()], warpgroup::laneid(), acc_scale);
                    }
#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
                    if constexpr (C::SELECTED_ROW_TRACE) {
                        fp4pv_native_group2_selected_row_record_softmax(
                            12,
                            t_coord,
                            m_tile,
                            idx,
                            scores_reg,
                            SCALE_LOG2,
                            row_max_old,
                            row_max,
                            acc_scale,
                            row_max);
                    }
#endif
                    warp::sync();
                    warp::arrive(corr_arrived[warpgroup::groupid()]);

                    if constexpr (C::BF16_PV_HYBRID) {
                        if (idx > 0) {
                            // The copied BF16 P tile is single-buffered.  QK
                            // may already own/rewrite score TMEM, but P itself
                            // is not reclaimed until the prior group2 PV
                            // completion reaches this event.
                            wait(p_tmem_reusable[warpgroup::groupid()],
                                 p_reuse_phase);
                            p_reuse_phase ^= 1;
                        }
                    }

                    constexpr int CONVERT_SIZE = 32;
                    // scale, exp2, convert and store P in 4 quarters, signal after each
                    if (!ZERO_SCALE) {
                        float neg_max_scaled = row_max * (-SCALE_LOG2);
                        float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                        const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            int ii = q;
                            bf16_2 scores_bf_reg[CONVERT_SIZE / 2];
                            #pragma unroll
                            for (int jj = 0; jj < 16; jj++) {
                                int si = ii * 16 + jj;
                                scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_max_scaled_2);
                                scores_reg[si].x = exp2f(scores_reg[si].x);
                                scores_reg[si].y = exp2f(scores_reg[si].y);
                                scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[si]);
                            }
                            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15};}"
                                :: "r"(*(uint32_t*)&scores_bf_reg[0]), "r"(*(uint32_t*)&scores_bf_reg[1]), "r"(*(uint32_t*)&scores_bf_reg[2]), "r"(*(uint32_t*)&scores_bf_reg[3]),
                                   "r"(*(uint32_t*)&scores_bf_reg[4]), "r"(*(uint32_t*)&scores_bf_reg[5]), "r"(*(uint32_t*)&scores_bf_reg[6]), "r"(*(uint32_t*)&scores_bf_reg[7]),
                                   "r"(*(uint32_t*)&scores_bf_reg[8]), "r"(*(uint32_t*)&scores_bf_reg[9]), "r"(*(uint32_t*)&scores_bf_reg[10]), "r"(*(uint32_t*)&scores_bf_reg[11]),
                                   "r"(*(uint32_t*)&scores_bf_reg[12]), "r"(*(uint32_t*)&scores_bf_reg[13]), "r"(*(uint32_t*)&scores_bf_reg[14]), "r"(*(uint32_t*)&scores_bf_reg[15]),
                                   "r"(score_bf_base + ii * 16));
                            tensor_store_wait();
                            tensor_after_thread_sync();
                            if (q == 0)
                                warp::tma::cluster::arrive(norm_scores_arrived[warpgroup::groupid()], 0);
                            else
                                warp::tma::cluster::arrive(norm_scores_quarter_arrived[q-1][warpgroup::groupid()], 0);
                        }
                    }
                    else {
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            int ii = q;
                            bf16_2 scores_bf_reg[CONVERT_SIZE / 2];
                            #pragma unroll
                            for (int jj = 0; jj < 16; jj++) {
                                int si = ii * 16 + jj;
                                scores_reg[si].x = isfinite(scores_reg[si].x) ? 0.0f : scores_reg[si].x;
                                scores_reg[si].y = isfinite(scores_reg[si].y) ? 0.0f : scores_reg[si].y;
                                scores_reg[si].x = exp2f(scores_reg[si].x);
                                scores_reg[si].y = exp2f(scores_reg[si].y);
                                scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[si]);
                            }
                            asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15};}"
                                :: "r"(*(uint32_t*)&scores_bf_reg[0]), "r"(*(uint32_t*)&scores_bf_reg[1]), "r"(*(uint32_t*)&scores_bf_reg[2]), "r"(*(uint32_t*)&scores_bf_reg[3]),
                                   "r"(*(uint32_t*)&scores_bf_reg[4]), "r"(*(uint32_t*)&scores_bf_reg[5]), "r"(*(uint32_t*)&scores_bf_reg[6]), "r"(*(uint32_t*)&scores_bf_reg[7]),
                                   "r"(*(uint32_t*)&scores_bf_reg[8]), "r"(*(uint32_t*)&scores_bf_reg[9]), "r"(*(uint32_t*)&scores_bf_reg[10]), "r"(*(uint32_t*)&scores_bf_reg[11]),
                                   "r"(*(uint32_t*)&scores_bf_reg[12]), "r"(*(uint32_t*)&scores_bf_reg[13]), "r"(*(uint32_t*)&scores_bf_reg[14]), "r"(*(uint32_t*)&scores_bf_reg[15]),
                                   "r"(score_bf_base + ii * 16));
                            tensor_store_wait();
                            tensor_after_thread_sync();
                            if (q == 0)
                                warp::tma::cluster::arrive(norm_scores_arrived[warpgroup::groupid()], 0);
                            else
                                warp::tma::cluster::arrive(norm_scores_quarter_arrived[q-1][warpgroup::groupid()], 0);
                        }
                    }
                    wait(rescale_finished[warpgroup::groupid()], rescale_phase);
                    rescale_phase ^= 1;
                    // row sum
                    float2 ls0 = {0.0f, 0.0f}, ls1 = {0.0f, 0.0f};
                    #pragma unroll
                    for (int ii = 0; ii < C::Nb / 2; ii += 2) {
                        ls0 = __fadd2_rn(ls0, scores_reg[ii]);
                        ls1 = __fadd2_rn(ls1, scores_reg[ii + 1]);
                    }
                    ls0 = __fadd2_rn(ls0, ls1);
                    row_sum = row_sum * acc_scale + ls0.x + ls0.y;
#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
                    if constexpr (C::SELECTED_ROW_TRACE) {
                        const float tile_sum = ls0.x + ls0.y;
                        fp4pv_native_group2_selected_row_record_sums(
                            12,
                            t_coord,
                            m_tile,
                            idx,
                            tile_sum,
                            row_sum,
                            tile_sum,
                            row_sum);
                    }
#endif

                    scores_phase ^= 1;
                }
                stats_store(lse_smem[warpgroup::groupid()], warpgroup::laneid(), row_max);
                stats_store(max_vec_smem[warpgroup::groupid()], warpgroup::laneid(), row_sum);
                warp::sync();
                if constexpr (C::BF16_PV_HYBRID) {
                    warp::arrive(stats_arrived[warpgroup::groupid()]);
                } else {
                    warp::arrive(corr_arrived[warpgroup::groupid()]);
                }
            }
        }
        else {
            int3 t_coord = get_tile_idx(blockIdx.x);
            int iters_per_task = t_coord.y + C::CLUSTER_SIZE * C::NUM_SOFTMAXXERS;
            int m_tile = t_coord.y + cta_rank * C::NUM_SOFTMAXXERS + warpgroup::groupid();
            float row_sum = 0.0f;
            float row_max = base_types::constants<float>::neg_infty();
            wait(rescale_finished[warpgroup::groupid()], rescale_phase);
            rescale_phase ^= 1;
            const float SCALE_LOG2 = qk_scale_log2(g, t_coord.x, t_coord.z);
            for (int idx = 0; idx < m_tile; idx++) {
                float2 scores_reg[C::Nb / 2];  // each thread holds one row of tmem
                wait(scores_arrived[warpgroup::groupid()], scores_phase);
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 32; ii++) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(scores_reg[ii * 16 + 0].x), "=f"(scores_reg[ii * 16 + 0].y), "=f"(scores_reg[ii * 16 + 1].x), "=f"(scores_reg[ii * 16 + 1].y), "=f"(scores_reg[ii * 16 + 2].x), "=f"(scores_reg[ii * 16 + 2].y), "=f"(scores_reg[ii * 16 + 3].x), "=f"(scores_reg[ii * 16 + 3].y),
                              "=f"(scores_reg[ii * 16 + 4].x), "=f"(scores_reg[ii * 16 + 4].y), "=f"(scores_reg[ii * 16 + 5].x), "=f"(scores_reg[ii * 16 + 5].y), "=f"(scores_reg[ii * 16 + 6].x), "=f"(scores_reg[ii * 16 + 6].y), "=f"(scores_reg[ii * 16 + 7].x), "=f"(scores_reg[ii * 16 + 7].y),
                              "=f"(scores_reg[ii * 16 + 8].x), "=f"(scores_reg[ii * 16 + 8].y), "=f"(scores_reg[ii * 16 + 9].x), "=f"(scores_reg[ii * 16 + 9].y), "=f"(scores_reg[ii * 16 + 10].x), "=f"(scores_reg[ii * 16 + 10].y), "=f"(scores_reg[ii * 16 + 11].x), "=f"(scores_reg[ii * 16 + 11].y),
                              "=f"(scores_reg[ii * 16 + 12].x), "=f"(scores_reg[ii * 16 + 12].y), "=f"(scores_reg[ii * 16 + 13].x), "=f"(scores_reg[ii * 16 + 13].y), "=f"(scores_reg[ii * 16 + 14].x), "=f"(scores_reg[ii * 16 + 14].y), "=f"(scores_reg[ii * 16 + 15].x), "=f"(scores_reg[ii * 16 + 15].y)
                            : "r"(score_tt_base + ii * 32));
                }

                float row_max_old = row_max;
                // calculate row max
                float lm0 = base_types::constants<float>::neg_infty(), lm1 = base_types::constants<float>::neg_infty(), lm2 = base_types::constants<float>::neg_infty(), lm3 = base_types::constants<float>::neg_infty();
                float lm4 = base_types::constants<float>::neg_infty(), lm5 = base_types::constants<float>::neg_infty(), lm6 = base_types::constants<float>::neg_infty(), lm7 = base_types::constants<float>::neg_infty();
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

                // give rescale factor to correction warpgroup
                if (idx > 0) {
                    acc_scale = softmax_rescale_factor(row_max, row_max_old, SCALE_LOG2);
                    stats_store(max_vec_smem[warpgroup::groupid()], warpgroup::laneid(), acc_scale);
                }
                warp::sync();
                warp::arrive(corr_arrived[warpgroup::groupid()]);

                float neg_max_scaled = row_max * (-SCALE_LOG2);
                float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                constexpr int CONVERT_SIZE = 32;
                // scale, exp2, convert and store P in 4 quarters, signal after each
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    int ii = q;
                    bf16_2 scores_bf_reg[CONVERT_SIZE / 2];
                    #pragma unroll
                    for (int jj = 0; jj < 16; jj++) {
                        int si = ii * 16 + jj;
                        scores_reg[si] = apply_softmax_scale(scores_reg[si], SCALE_LOG2, scale_2, neg_max_scaled_2);
                        scores_reg[si].x = exp2f(scores_reg[si].x);
                        scores_reg[si].y = exp2f(scores_reg[si].y);
                        scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[si]);
                    }
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15};}"
                        :: "r"(*(uint32_t*)&scores_bf_reg[0]), "r"(*(uint32_t*)&scores_bf_reg[1]), "r"(*(uint32_t*)&scores_bf_reg[2]), "r"(*(uint32_t*)&scores_bf_reg[3]),
                           "r"(*(uint32_t*)&scores_bf_reg[4]), "r"(*(uint32_t*)&scores_bf_reg[5]), "r"(*(uint32_t*)&scores_bf_reg[6]), "r"(*(uint32_t*)&scores_bf_reg[7]),
                           "r"(*(uint32_t*)&scores_bf_reg[8]), "r"(*(uint32_t*)&scores_bf_reg[9]), "r"(*(uint32_t*)&scores_bf_reg[10]), "r"(*(uint32_t*)&scores_bf_reg[11]),
                           "r"(*(uint32_t*)&scores_bf_reg[12]), "r"(*(uint32_t*)&scores_bf_reg[13]), "r"(*(uint32_t*)&scores_bf_reg[14]), "r"(*(uint32_t*)&scores_bf_reg[15]),
                           "r"(score_bf_base + ii * 16));
                    tensor_store_wait();
                    tensor_after_thread_sync();
                    if (q == 0)
                        warp::tma::cluster::arrive(norm_scores_arrived[warpgroup::groupid()], 0);
                    else
                        warp::tma::cluster::arrive(norm_scores_quarter_arrived[q-1][warpgroup::groupid()], 0);
                }

                wait(rescale_finished[warpgroup::groupid()], rescale_phase);
                rescale_phase ^= 1;
                // row sum
                float2 ls0 = {0.0f, 0.0f}, ls1 = {0.0f, 0.0f};
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 2; ii += 2) {
                    ls0 = __fadd2_rn(ls0, scores_reg[ii]);
                    ls1 = __fadd2_rn(ls1, scores_reg[ii + 1]);
                }
                ls0 = __fadd2_rn(ls0, ls1);
                row_sum = row_sum * acc_scale + ls0.x + ls0.y;

                scores_phase ^= 1;
            }

            for (int idx = m_tile; idx < iters_per_task; idx++) {
                float2 scores_reg[C::Nb / 2];  // each thread holds one row of tmem
                wait(scores_arrived[warpgroup::groupid()], scores_phase);
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 32; ii++) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(scores_reg[ii * 16 + 0].x), "=f"(scores_reg[ii * 16 + 0].y), "=f"(scores_reg[ii * 16 + 1].x), "=f"(scores_reg[ii * 16 + 1].y), "=f"(scores_reg[ii * 16 + 2].x), "=f"(scores_reg[ii * 16 + 2].y), "=f"(scores_reg[ii * 16 + 3].x), "=f"(scores_reg[ii * 16 + 3].y),
                              "=f"(scores_reg[ii * 16 + 4].x), "=f"(scores_reg[ii * 16 + 4].y), "=f"(scores_reg[ii * 16 + 5].x), "=f"(scores_reg[ii * 16 + 5].y), "=f"(scores_reg[ii * 16 + 6].x), "=f"(scores_reg[ii * 16 + 6].y), "=f"(scores_reg[ii * 16 + 7].x), "=f"(scores_reg[ii * 16 + 7].y),
                              "=f"(scores_reg[ii * 16 + 8].x), "=f"(scores_reg[ii * 16 + 8].y), "=f"(scores_reg[ii * 16 + 9].x), "=f"(scores_reg[ii * 16 + 9].y), "=f"(scores_reg[ii * 16 + 10].x), "=f"(scores_reg[ii * 16 + 10].y), "=f"(scores_reg[ii * 16 + 11].x), "=f"(scores_reg[ii * 16 + 11].y),
                              "=f"(scores_reg[ii * 16 + 12].x), "=f"(scores_reg[ii * 16 + 12].y), "=f"(scores_reg[ii * 16 + 13].x), "=f"(scores_reg[ii * 16 + 13].y), "=f"(scores_reg[ii * 16 + 14].x), "=f"(scores_reg[ii * 16 + 14].y), "=f"(scores_reg[ii * 16 + 15].x), "=f"(scores_reg[ii * 16 + 15].y)
                            : "r"(score_tt_base + ii * 32));
                }

                int causal_col = (idx > m_tile) ? -1 : (int)warpgroup::laneid();
                #pragma unroll
                for (int k = 0; k < C::Nb / 2; k++) {
                    if (k * 2 > causal_col) scores_reg[k].x = base_types::constants<float>::neg_infty();
                    if (k * 2 + 1 > causal_col) scores_reg[k].y = base_types::constants<float>::neg_infty();
                }

                float row_max_old = row_max;
                // calculate row max
                float lm0 = base_types::constants<float>::neg_infty(), lm1 = base_types::constants<float>::neg_infty(), lm2 = base_types::constants<float>::neg_infty(), lm3 = base_types::constants<float>::neg_infty();
                float lm4 = base_types::constants<float>::neg_infty(), lm5 = base_types::constants<float>::neg_infty(), lm6 = base_types::constants<float>::neg_infty(), lm7 = base_types::constants<float>::neg_infty();
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

                // give rescale factor to correction warpgroup
                acc_scale = softmax_rescale_factor(row_max, row_max_old, SCALE_LOG2);
                stats_store(max_vec_smem[warpgroup::groupid()], warpgroup::laneid(), acc_scale);
                warp::sync();
                warp::arrive(corr_arrived[warpgroup::groupid()]);

                float neg_max_scaled = row_max * (-SCALE_LOG2);
                float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                constexpr int CONVERT_SIZE = 32;
                // scale, exp2, convert and store P in 4 quarters, signal after each
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    int ii = q;
                    bf16_2 scores_bf_reg[CONVERT_SIZE / 2];
                    #pragma unroll
                    for (int jj = 0; jj < 16; jj++) {
                        int si = ii * 16 + jj;
                        scores_reg[si] = apply_softmax_scale(scores_reg[si], SCALE_LOG2, scale_2, neg_max_scaled_2);
                        scores_reg[si].x = exp2f(scores_reg[si].x);
                        scores_reg[si].y = exp2f(scores_reg[si].y);
                        scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[si]);
                    }
                    asm volatile("{tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15};}"
                        :: "r"(*(uint32_t*)&scores_bf_reg[0]), "r"(*(uint32_t*)&scores_bf_reg[1]), "r"(*(uint32_t*)&scores_bf_reg[2]), "r"(*(uint32_t*)&scores_bf_reg[3]),
                           "r"(*(uint32_t*)&scores_bf_reg[4]), "r"(*(uint32_t*)&scores_bf_reg[5]), "r"(*(uint32_t*)&scores_bf_reg[6]), "r"(*(uint32_t*)&scores_bf_reg[7]),
                           "r"(*(uint32_t*)&scores_bf_reg[8]), "r"(*(uint32_t*)&scores_bf_reg[9]), "r"(*(uint32_t*)&scores_bf_reg[10]), "r"(*(uint32_t*)&scores_bf_reg[11]),
                           "r"(*(uint32_t*)&scores_bf_reg[12]), "r"(*(uint32_t*)&scores_bf_reg[13]), "r"(*(uint32_t*)&scores_bf_reg[14]), "r"(*(uint32_t*)&scores_bf_reg[15]),
                           "r"(score_bf_base + ii * 16));
                    tensor_store_wait();
                    tensor_after_thread_sync();
                    if (q == 0)
                        warp::tma::cluster::arrive(norm_scores_arrived[warpgroup::groupid()], 0);
                    else
                        warp::tma::cluster::arrive(norm_scores_quarter_arrived[q-1][warpgroup::groupid()], 0);
                }

                wait(rescale_finished[warpgroup::groupid()], rescale_phase);
                rescale_phase ^= 1;
                // row sum
                float2 ls0 = {0.0f, 0.0f}, ls1 = {0.0f, 0.0f};
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 2; ii += 2) {
                    ls0 = __fadd2_rn(ls0, scores_reg[ii]);
                    ls1 = __fadd2_rn(ls1, scores_reg[ii + 1]);
                }
                ls0 = __fadd2_rn(ls0, ls1);
                row_sum = row_sum * acc_scale + ls0.x + ls0.y;

                scores_phase ^= 1;
            }
            stats_store(lse_smem[warpgroup::groupid()], warpgroup::laneid(), row_max);
            stats_store(max_vec_smem[warpgroup::groupid()], warpgroup::laneid(), row_sum);
            warp::sync();
            warp::arrive(corr_arrived[warpgroup::groupid()]);
        }
    }
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE, 1, 1) __launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_fp4pv(const __grid_constant__ globals_fp4pv<C, !FP4PV_DIAG_USE_DIRECT_ROW_UPDATE> g) {
    using G = globals_fp4pv<C, !FP4PV_DIAG_USE_DIRECT_ROW_UPDATE>;
    using d_tt_scores = tt<float, C::Mb, C::Nb>;
    using d_tt_scores_bf = tt<bf16, C::Mb, C::Nb>;
    using d_tt_outputs = tt<float, C::Mb, C::Dvo>;
    using d_tt_scores_bf_1q = tt<bf16, C::Mb, C::Nb/4>;
    using d_tt_q_sc = full_tt_fp8e4m3<C::Q_SC_TMEM_WIDTH>;
    using d_tt_sc16 = full_tt_fp8e4m3<16>;
    using d_tt_p_sc = full_tt_fp8e4m3<C::P_SC_TMEM_WIDTH>;
    using d_tt_v_sc = full_tt_fp8e4m3<C::V_SC_TMEM_WIDTH>;
    const int cta_rank = cluster_ctarank();
    const int tiles_m = g.q.rows() / C::Mb;
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

    auto get_tile_idx = [&](int block_idx) -> int3 {
        int cluster_linear = block_idx / C::CLUSTER_SIZE;
        int bidhb  = cluster_linear / l2_major;
        int l2_mod = cluster_linear - bidhb * l2_major;
        int m_cluster, bidhb_residual;
        if (bidhb < num_hb_quotient) {
            m_cluster      = l2_mod / swizzle;
            bidhb_residual = l2_mod - m_cluster * swizzle;
        }
        else {
            int divisor    = (num_hb_remainder > 0) ? num_hb_remainder : 1;
            m_cluster      = l2_mod / divisor;
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
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::q_tile (&q_smem)[C::NUM_SOFTMAXXERS] = al.allocate<typename G::q_tile, C::NUM_SOFTMAXXERS>();
    typename G::q_sc_smem_storage &q_sc_smem = al.allocate<typename G::q_sc_smem_storage>();
    typename G::k_tile (&k_smem)[C::LOAD_STAGES] = al.allocate<typename G::k_tile, C::LOAD_STAGES>();
    typename G::k_sc_smem_storage &k_sc_smem = al.allocate<typename G::k_sc_smem_storage>();
    auto (&k_sc_guard)[1] = al.allocate<typename G::k_sc_guard_slot, 1>();
    (void)k_sc_guard;
    typename G::v_fp4_tile (&v_fp4_smem)[C::LOAD_STAGES] = al.allocate<typename G::v_fp4_tile, C::LOAD_STAGES>();
    typename G::v_sc_tile (&v_sc_smem)[C::LOAD_STAGES] = al.allocate<typename G::v_sc_tile, C::LOAD_STAGES>();
    auto (&o_smem)[1] = al.allocate<typename G::o_tile, 1>();
    auto p_bf16_scan = fp4pv_allocate_p_scan<G::NEED_P_SCAN_STORAGE, 2>(al);
    auto (&p_fp4_stage)[2] = al.allocate<typename G::p_fp4_tile, 2>();
    auto (&p_sc_stage)[2] = al.allocate<typename G::p_sc_tile, 2>();
    auto (&p_sc_guard)[1] = al.allocate<typename G::p_sc_guard_slot, 1>();
    (void)p_sc_guard;

    __shared__ sv_fl<C::Mb> corr_vec_smem[1];
    __shared__ sv_fl<C::Mb> max_vec_smem[1];
    __shared__ sv_fl<C::Mb> lse_smem[1];

    __shared__ semaphore q_arrived[1], q_sc_arrived[1], q_finished[1];
    __shared__ semaphore k_arrived[C::LOAD_STAGES], k_sc_arrived[C::K_SCALE_SMEM_SLOTS], k_finished[C::LOAD_STAGES], k_sc_finished[C::K_SCALE_SMEM_SLOTS];
    __shared__ semaphore v_arrived[C::LOAD_STAGES], v_finished[C::LOAD_STAGES], v_remote_ready[C::LOAD_STAGES];
    __shared__ semaphore scores_arrived[1];
    __shared__ semaphore corr_arrived[1], tile_arrived[1], rescale_finished[1], pv_tmem_ready[1], output_reusable[1];
    __shared__ semaphore p_copy_done[2], p_pack_ready[2], p_quant_ready[2], p_remote_ready[2];
    __shared__ float p_softmax_warp_amax[2][4];
    __shared__ float p_cta_amax[2];

    if (threadIdx.x == 0) {
        g.q.template prefetch_tma<typename G::q_tile, dim::ROW>();
        g.q_sc.template prefetch_tma<typename G::q_sc_tile, dim::ROW>();
        g.k.template prefetch_tma<typename G::k_tile, dim::ROW>();
        g.k_sc.template prefetch_tma<typename G::k_sc_tile, dim::ROW>();
        g.o.template prefetch_tma<typename G::o_tile, dim::DEPTH>();
        init_semaphore(q_arrived[0], 0, 1);
        init_semaphore(q_sc_arrived[0], 0, 1);
        init_semaphore(q_finished[0], 0, 1);
        init_semaphore(scores_arrived[0], 0, 1);
        init_semaphore(corr_arrived[0], 0, 4);
        init_semaphore(tile_arrived[0], 0, 1);
        init_semaphore(rescale_finished[0], 0, 1);
        init_semaphore(pv_tmem_ready[0], 0, 1);
        init_semaphore(output_reusable[0], 0, 1);
        init_semaphore(p_copy_done[0], 0, 1);
        init_semaphore(p_copy_done[1], 0, 1);
        init_semaphore(p_pack_ready[0], 0, 1);
        init_semaphore(p_pack_ready[1], 0, 1);
        init_semaphore(p_quant_ready[0], 0, 1);
        init_semaphore(p_quant_ready[1], 0, 1);
        init_semaphore(p_remote_ready[0], 0, 1);
        init_semaphore(p_remote_ready[1], 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_STAGES; i++) {
            init_semaphore(k_arrived[i], 0, 1);
            init_semaphore(k_finished[i], 0, 1);
            init_semaphore(v_arrived[i], 0, 1);
            init_semaphore(v_finished[i], 0, 1);
            init_semaphore(v_remote_ready[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < C::K_SCALE_SMEM_SLOTS; i++) {
            init_semaphore(k_sc_arrived[i], 0, 1);
            init_semaphore(k_sc_finished[i], 0, 1);
        }
    }
    __syncthreads();
    fp4pv_cluster_sync<C>();

    if (warpgroup::groupid() == 3) {
        warpgroup::decrease_registers<128>();
        if (warpgroup::warpid() == 3) {
            const bool local_leader = warp::elect_leader();
            int k_idx = 0;
            int k_phase = 1;
            int k_sc_idx = 0;
            int k_sc_phase = 0;
            int v_idx = 0;
            int v_phase = 1;
            int q_phase = 1;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                const int q_tile_idx = min(t_coord.y + cta_rank, tiles_m - 1);
                if (local_leader) {
                    if (cta_rank < q_tiles_this_cluster) {
                        const coord<typename G::q_tile> q_coord = {t_coord.x, t_coord.z, q_tile_idx, 0};
                        const coord<typename G::q_sc_tile> q_sc_coord = {t_coord.x, q_tile_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0};
                        fp4pv_wait<C>(q_finished[0], q_phase);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(q_smem[0], g.q, q_coord, q_arrived[0], (uint16_t)(1<<cta_rank), 0);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(q_sc_smem, 0), g.q_sc, q_sc_coord, q_sc_arrived[0], (uint16_t)(1<<cta_rank), 0);
                    }
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    const int k_tile_idx = min(idx * C::CLUSTER_SIZE + cta_rank, tiles_m - 1);
                    if (local_leader) {
                        const coord<typename G::k_tile> k_coord = {t_coord.x, t_coord.z, k_tile_idx, 0};
                        const coord<typename G::k_sc_tile> k_sc_coord = {t_coord.x, k_tile_idx, t_coord.z * C::QK_SCALE_CHUNKS, 0};
                        fp4pv_wait<C>(k_finished[k_idx], k_phase);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(k_smem[k_idx], g.k, k_coord, k_arrived[k_idx], (uint16_t)(1<<cta_rank), 0);
                        wait(k_sc_finished[k_sc_idx], k_sc_phase ^ 1);
                        fp4pv_load_async<dim::ROW, cache_policy::NORMAL, C>(slot_tile_at(k_sc_smem, k_sc_idx), g.k_sc, k_sc_coord, k_sc_arrived[k_sc_idx], (uint16_t)(1<<cta_rank), 0);
                    }
                    k_idx++; if (k_idx == C::LOAD_STAGES) { k_idx = 0; k_phase ^= 1; }
                    k_sc_idx++;
                    if (k_sc_idx == C::K_SCALE_SMEM_SLOTS) {
                        k_sc_idx = 0;
                        k_sc_phase ^= 1;
                    }

	                    if (local_leader && (cur_bid != blockIdx.x || idx >= C::LOAD_STAGES)) {
	                        wait(v_finished[v_idx], v_phase);
	                    }
                    warp::sync();
                    load_v_fp4_tile_from_global<C>(v_fp4_smem[v_idx], g, t_coord.x, t_coord.z, idx, cta_rank);
                    load_v_sc_prepared_from_global<C>(v_sc_smem[v_idx], g, t_coord.x, t_coord.z, idx);
                    warp::sync();
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    publish_cluster_shared_backing_if_needed<C>();
                    warp::sync();
                    if (local_leader) {
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
                q_phase ^= 1;
            }
            if (FP4PV_DEBUG_TRAP_STAGE == 16 && blockIdx.x == 1 && cta_rank == 1) {
                asm volatile("trap;");
            }
            if (FP4PV_DEBUG_TRAP_STAGE == 38 && blockIdx.x == 0 && cta_rank == 0) {
                asm volatile("trap;");
            }
        } else if (warpgroup::warpid() == 0 && cta_rank == 0 && warp::elect_leader()) {
            d_tt_scores tt_score = tm_alloc.template allocate<d_tt_scores>(0);
            d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(C::Nb);
            d_tt_q_sc q_sc_tm = tm_alloc.template allocate<d_tt_q_sc>(C::Nb + C::Dvo);
            d_tt_sc16 k_sc_tm = tm_alloc.template allocate<d_tt_sc16>(C::Nb + C::Dvo + C::Q_SC_TMEM_WIDTH);
            constexpr int P_SC_BASE = C::Nb + C::Dvo + C::Q_SC_TMEM_WIDTH + 16;
            constexpr int V_SC_BASE = P_SC_BASE + 2 * C::P_SC_TMEM_WIDTH;
            static_assert(V_SC_BASE + 2 * C::V_SC_TMEM_WIDTH <= MAX_TENSOR_COLS, "fp4pv producer scale ping-pong exceeds TMEM budget");
            d_tt_p_sc p_sc_tm0 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 0 * C::P_SC_TMEM_WIDTH);
            d_tt_p_sc p_sc_tm1 = tm_alloc.template allocate<d_tt_p_sc>(P_SC_BASE + 1 * C::P_SC_TMEM_WIDTH);
            d_tt_v_sc v_sc_tm0 = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE + 0 * C::V_SC_TMEM_WIDTH);
            d_tt_v_sc v_sc_tm1 = tm_alloc.template allocate<d_tt_v_sc>(V_SC_BASE + 1 * C::V_SC_TMEM_WIDTH);
            int k_idx = 0, k_phase = 0, k_sc_slot = 0;
            int v_idx = 0, v_phase = 0;
            int k_sc_arrived_phase_mask = 0;
            int p_copy_phase_mask = 0;
            int p_quant_phase_mask = 0;
            int p_remote_phase_mask = 0;
            int v_remote_phase_mask = 0;
            int output_reuse_phase = 0;
            auto issue_next_qk = [&](bool final_q_for_task, int copy_buf, bool wait_remote_p_copy) {
                const int copy_phase = (p_copy_phase_mask >> copy_buf) & 1;
                wait(p_copy_done[copy_buf], copy_phase);
                p_copy_phase_mask ^= (1 << copy_buf);
                const int next_k_slot = k_idx;
                fp4pv_expect_bytes<C>(k_arrived[next_k_slot], C::CLUSTER_SIZE * sizeof(typename G::k_tile));
                fp4pv_wait<C>(k_arrived[next_k_slot], k_phase);
                fp4pv_expect_bytes<C>(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(typename G::k_sc_tile));
                const int k_sc_phase = (k_sc_arrived_phase_mask >> k_sc_slot) & 1;
                fp4pv_wait<C>(k_sc_arrived[k_sc_slot], k_sc_phase);
                k_sc_arrived_phase_mask ^= (1 << k_sc_slot);
                issue_qk_chunked_qsc_tmem<C>(tt_score, q_smem[0], k_smem[next_k_slot], q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[next_k_slot]);
                detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
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

            auto wait_and_stage_v_sc = [&](int tt_buf, int v_slot) {
                auto &v_sc_tm_cur = tt_buf == 0 ? v_sc_tm0 : v_sc_tm1;
                wait(v_arrived[v_slot], v_phase);
                if (FP4PV_DEBUG_TRAP_STAGE == 10 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
                    asm volatile("trap;");
                }
                if constexpr (C::CLUSTER_SIZE > 1) {
                    const int v_remote_phase = (v_remote_phase_mask >> v_slot) & 1;
                    tma::cluster::wait(v_remote_ready[v_slot], v_remote_phase);
                    if (FP4PV_DEBUG_TRAP_STAGE == 11 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
                        asm volatile("trap;");
                    }
                    v_remote_phase_mask ^= (1 << v_slot);
                }
                #pragma unroll
                for (int ii = 0; ii < C::PV_SCALE_CHUNKS; ++ii) {
                    auto v_sc_tm_sub = v_sc_tm_cur.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                    load_k_scale_chunk<C>(v_sc_tm_sub, v_sc_smem[v_slot], ii);
                }
            };

            auto wait_and_stage_p_sc = [&](int tt_buf, int p_buf, bool wait_remote_p_quant) {
                auto &p_sc_tm_cur = tt_buf == 0 ? p_sc_tm0 : p_sc_tm1;
                const int p_quant_phase = (p_quant_phase_mask >> p_buf) & 1;
                wait(p_quant_ready[p_buf], p_quant_phase);
                p_quant_phase_mask ^= (1 << p_buf);
                if (wait_remote_p_quant) {
                    const int p_remote_phase = (p_remote_phase_mask >> p_buf) & 1;
                    tma::cluster::wait(p_remote_ready[p_buf], p_remote_phase);
                    p_remote_phase_mask ^= (1 << p_buf);
                }
                if (FP4PV_DEBUG_TRAP_STAGE == 3 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
                    asm volatile("trap;");
                }
                #pragma unroll
                for (int ii = 0; ii < C::PV_SCALE_CHUNKS; ++ii) {
                    auto p_sc_tm_sub = p_sc_tm_cur.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                    load_q_scale_chunk<C>(p_sc_tm_sub, p_sc_stage[p_buf], ii);
                }
            };

            for (int cur_bid = blockIdx.x, task_num = 0; cur_bid < total_bids; cur_bid += gridDim.x, task_num++) {
                int3 t_coord = get_tile_idx(cur_bid);
                int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                const int q_tiles_this_cluster = min(C::CLUSTER_SIZE, tiles_m - t_coord.y);
                const bool wait_remote_p = (q_tiles_this_cluster > 1);
                int q_phase = task_num & 1;
                if (task_num > 0) {
                    wait(output_reusable[0], output_reuse_phase);
                    output_reuse_phase ^= 1;
                }
                fp4pv_expect_bytes<C>(q_arrived[0], q_tiles_this_cluster * sizeof(typename G::q_tile));
                fp4pv_wait<C>(q_arrived[0], q_phase);
                if (FP4PV_DEBUG_TRAP_STAGE == 34 && blockIdx.x == 0 && cta_rank == 0) {
                    asm volatile("trap;");
                }
                fp4pv_expect_bytes<C>(q_sc_arrived[0], q_tiles_this_cluster * sizeof(typename G::q_sc_tile));
                fp4pv_wait<C>(q_sc_arrived[0], q_phase);
                stage_q_scale_tmem<C>(q_sc_tm, slot_tile_at(q_sc_smem, 0));
                tensor_load_wait();
                tensor_before_thread_sync();
                if (FP4PV_DEBUG_TRAP_STAGE == 35 && blockIdx.x == 0 && cta_rank == 0) {
                    asm volatile("trap;");
                }
                int k_slot = k_idx;
                fp4pv_expect_bytes<C>(k_arrived[k_slot], C::CLUSTER_SIZE * sizeof(typename G::k_tile));
                fp4pv_wait<C>(k_arrived[k_slot], k_phase);
                if (FP4PV_DEBUG_TRAP_STAGE == 36 && blockIdx.x == 0 && cta_rank == 0) {
                    asm volatile("trap;");
                }
                fp4pv_expect_bytes<C>(k_sc_arrived[k_sc_slot], C::CLUSTER_SIZE * sizeof(typename G::k_sc_tile));
                {
                    const int k_sc_phase = (k_sc_arrived_phase_mask >> k_sc_slot) & 1;
                    fp4pv_wait<C>(k_sc_arrived[k_sc_slot], k_sc_phase);
                    if (FP4PV_DEBUG_TRAP_STAGE == 37 && blockIdx.x == 0 && cta_rank == 0) {
                        asm volatile("trap;");
                    }
                    k_sc_arrived_phase_mask ^= (1 << k_sc_slot);
                }
                if (FP4PV_DEBUG_TRAP_STAGE == 33 && blockIdx.x == 0 && cta_rank == 0) {
                    asm volatile("trap;");
                }
                issue_qk_chunked_qsc_tmem<C>(tt_score, q_smem[0], k_smem[k_slot], q_sc_tm, slot_tile_at(k_sc_smem, k_sc_slot), k_sc_tm, k_finished[k_slot]);
                detail::tcgen05::commit<C::CLUSTER_SIZE>(scores_arrived[0]);
                if (FP4PV_DEBUG_TRAP_STAGE == 32 && blockIdx.x == 0 && cta_rank == 0) {
                    asm volatile("trap;");
                }
                arrive(k_sc_finished[k_sc_slot]);
                fp4pv_remote_arrive_if_needed<C>(k_sc_finished[k_sc_slot], 1);
                if (iters_per_task == 1) {
                    detail::tcgen05::commit<C::CLUSTER_SIZE>(q_finished[0]);
                }
                k_idx ^= 1; if (k_idx == 0) { k_phase ^= 1; }
                k_sc_slot ^= 1;

                zero_output_scratch_issue_lane<C>(tt_output);
                auto issue_pv = [&](int tt_buf, int p_buf, int v_slot, bool is_first, bool is_final) {
                    auto &p_sc_tm_cur = tt_buf == 0 ? p_sc_tm0 : p_sc_tm1;
                    auto &v_sc_tm_cur = tt_buf == 0 ? v_sc_tm0 : v_sc_tm1;
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    if (FP4PV_DEBUG_TRAP_STAGE == 4 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
                        asm volatile("trap;");
                    }
                    if (is_first) {
                        mm2_ABt(tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_slot], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_slot]);
                        if (FP4PV_DEBUG_TRAP_STAGE == 5 && blockIdx.x == 0 && cta_rank == 0) {
                            asm volatile("trap;");
                        }
                    } else {
                        if constexpr (FP4PV_USE_MM2_NONFIRST) {
                            mm2_ABt(tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_slot], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_slot]);
                        } else {
                            mma2_ABt(tt_output, p_fp4_stage[p_buf], v_fp4_smem[v_slot], p_sc_tm_cur, v_sc_tm_cur, v_finished[v_slot]);
                        }
                    }
                    tensor_commit<2>(pv_tmem_ready[0]);
                    if (is_final) {
                        detail::tcgen05::commit<C::CLUSTER_SIZE>(tile_arrived[0]);
                        if (FP4PV_DEBUG_TRAP_STAGE == 6 && blockIdx.x == 0 && cta_rank == 0) {
                            asm volatile("trap;");
                        }
                    }
                    v_idx ^= 1;
                    if (v_idx == 0) {
                        v_phase ^= 1;
                    }
                };
                auto run_iteration = [&](int idx, bool is_first, bool is_final) {
                    const int buf = idx & 1;
                    if (!is_final && !FP4PV_DIAG_ISSUE_NEXT_QK_AFTER_PV) {
                        issue_next_qk(idx == iters_per_task - 2, buf, wait_remote_p);
                    }
                    if constexpr (FP4PV_DIAG_FULL_ZERO_NONFIRST) {
                        if (!is_first) {
                            zero_output_scratch_issue_lane<C>(tt_output);
                        }
                    }
                    const int v_slot = v_idx;
                    wait_and_stage_v_sc(buf, v_slot);
                    wait_and_stage_p_sc(buf, buf, wait_remote_p);
                    issue_pv(buf, buf, v_slot, is_first, is_final);
                    if (!is_final && FP4PV_DIAG_ISSUE_NEXT_QK_AFTER_PV) {
                        issue_next_qk(idx == iters_per_task - 2, buf, wait_remote_p);
                    }
                };

                if (iters_per_task == 1) {
                    run_iteration(0, true, true);
                } else {
                    run_iteration(0, true, false);
                    for (int idx = 1; idx < iters_per_task - 1; ++idx) {
                        run_iteration(idx, false, false);
                    }
                    run_iteration(iters_per_task - 1, false, true);
                }
                const int final_copy_buf = (iters_per_task - 1) & 1;
                const int final_copy_phase = (p_copy_phase_mask >> final_copy_buf) & 1;
                wait(p_copy_done[final_copy_buf], final_copy_phase);
                p_copy_phase_mask ^= (1 << final_copy_buf);
            }
            if (FP4PV_DEBUG_TRAP_STAGE == 14 && blockIdx.x == 0 && cta_rank == 0) {
                asm volatile("trap;");
            }
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 24) {
            warpgroup::sync(warpgroup::groupid()+1);
            if (blockIdx.x == 1 && cta_rank == 1 && warpgroup::laneid() == 0) {
                asm volatile("trap;");
            }
        }
    }
    else if (warpgroup::groupid() == 2) {
        warpgroup::decrease_registers<48>();
        d_tt_outputs tt_output = tm_alloc.template allocate<d_tt_outputs>(C::Nb);
        int end_phase = 0;
        int corr_phase = 0;
        int pv_phase = 0;
        for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
            int3 t_coord = get_tile_idx(cur_bid);
            const bool row_tile_valid = (t_coord.y + cta_rank) < tiles_m;
            if (!row_tile_valid) {
                continue;
            }
            const float scale_log2 = qk_scale_log2(g, t_coord.x, t_coord.z);
            int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);

            wait(corr_arrived[0], corr_phase);
            warpgroup::arrive(rescale_finished[0]);
            corr_phase ^= 1;
            for (int idx = 1; idx < iters_per_task; idx++) {
                wait(corr_arrived[0], corr_phase);
                wait(pv_tmem_ready[0], pv_phase);
                pv_phase ^= 1;
                float correction = FP4PV_DIAG_USE_DIRECT_ROW_UPDATE
                    ? stats_load(max_vec_smem[0], warpgroup::laneid())
                    : stats_load(corr_vec_smem[0], warpgroup::laneid());
                bool lane_needs_rescale = correction < 1.0f;
                bool needs_rescale = __any_sync(0xFFFFFFFF, lane_needs_rescale);
            if (FP4PV_DEBUG_PRINT_ROWSTATE && blockIdx.x == 0 && cta_rank == 1 &&
                (warpgroup::laneid() == 64 || warpgroup::laneid() == 96) && idx < 2) {
                printf("corr idx=%d lane=%d correction=%f needs_rescale=%d\n", idx, (int)warpgroup::laneid(), correction, (int)needs_rescale);
            }
                if (needs_rescale && !FP4PV_DIAG_SKIP_CORRECTION) {
                    const float applied_correction = lane_needs_rescale ? correction : 1.0f;
                    float2 corr_2 = {applied_correction, applied_correction};
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
                        #pragma unroll
                        for (int ii = 0; ii < 8; ii++) {
                            o_reg[ii] = __fmul2_rn(o_reg[ii], corr_2);
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
                warpgroup::sync(warpgroup::groupid()+1);
                warpgroup::arrive(rescale_finished[0]);
                corr_phase ^= 1;
            }

            wait(corr_arrived[0], corr_phase);
            wait(pv_tmem_ready[0], pv_phase);
            pv_phase ^= 1;
            float row_sum = stats_load(max_vec_smem[0], warpgroup::laneid());
            float row_max = stats_load(lse_smem[0], warpgroup::laneid());
            warpgroup::arrive(rescale_finished[0]);
            bool row_invalid = (row_sum == 0.0f) | (row_sum != row_sum);
            float inv_norm_s = 1.0f;
            if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE) {
                asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_norm_s) : "f"(row_invalid ? 1.0f : row_sum));
            }
            float2 inv_norm = {inv_norm_s, inv_norm_s};
            constexpr float LN2 = 0.693147180559945f;
            float lse_val;
            if (!row_invalid) {
                float log2_row_sum;
                asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_row_sum) : "f"(row_sum));
                lse_val = (row_max * scale_log2 + log2_row_sum) * LN2;
            } else {
                lse_val = base_types::constants<float>::neg_infty();
            }
            warpgroup::tma::store_async_read_wait<0>();
            wait(tile_arrived[0], end_phase);
            if (FP4PV_DEBUG_TRAP_STAGE == 7 && blockIdx.x == 2 && cta_rank == 0 && warpgroup::laneid() == 0) {
                asm volatile("trap;");
            }
            warpgroup::sync(warpgroup::groupid()+1);
            constexpr int SUBTILE_COLS = G::o_tile::swizzle_bytes / sizeof(bf16);
            constexpr uint32_t SWIZZLE_MASK = G::o_tile::swizzle_bytes * 8 - 1;
            uint32_t base_addr = __cvta_generic_to_shared(&o_smem[0].data[0]);
            uint32_t row_offset = warpgroup::laneid() * SUBTILE_COLS * sizeof(bf16);

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
                for (int i = 0; i < 8; i += 2) {
                    if (row_invalid) {
                        o_reg[i] = float2{0.0f, 0.0f};
                        o_reg[i + 1] = float2{0.0f, 0.0f};
                    }
                    bf16_2 tmp0;
                    bf16_2 tmp1;
                    if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE) {
                        tmp0 = __float22bfloat162_rn(__fmul2_rn(o_reg[i], inv_norm));
                        tmp1 = __float22bfloat162_rn(__fmul2_rn(o_reg[i + 1], inv_norm));
                    } else {
                        tmp0 = __float22bfloat162_rn(o_reg[i]);
                        tmp1 = __float22bfloat162_rn(o_reg[i + 1]);
                    }
                    uint32_t addr = row_base + i * 4;
                    uint32_t swizzled_addr = addr ^ (((addr & SWIZZLE_MASK) >> 7) << 4);
                    uint64_t packed = (uint64_t)(*(uint32_t*)&tmp0) | ((uint64_t)(*(uint32_t*)&tmp1) << 32);
                    asm volatile("st.shared.b64 [%0], %1;" :: "r"(swizzled_addr), "l"(packed));
                }
            }
            warpgroup::sync(warpgroup::groupid()+1);
            warpgroup::tma::store_async<dim::DEPTH, cache_policy::EVICT_FIRST>(g.o, o_smem[0], {t_coord.x, t_coord.y + cta_rank, t_coord.z, 0});
            warpgroup::arrive(output_reusable[0]);
            g.lse[{t_coord.x, t_coord.z, 0, (t_coord.y + cta_rank) * C::Mb + warpgroup::laneid()}] = lse_val;
            if (FP4PV_DEBUG_TRAP_STAGE == 8 && blockIdx.x == 1 && cta_rank == 1 && warpgroup::laneid() == 0) {
                asm volatile("trap;");
            }
            corr_phase ^= 1;
            end_phase ^= 1;
        }
        warpgroup::tma::store_async_wait<0>();
        if (FP4PV_DEBUG_TRAP_STAGE == 22) {
            warpgroup::sync(warpgroup::groupid()+1);
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 12 && blockIdx.x == 1 && cta_rank == 1 && warpgroup::laneid() == 0) {
            asm volatile("trap;");
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 17 && blockIdx.x == 1 && cta_rank == 1 && warpgroup::laneid() == 127) {
            asm volatile("trap;");
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 22 && blockIdx.x == 1 && cta_rank == 1 && warpgroup::laneid() == 0) {
            asm volatile("trap;");
        }
    }
    else if (warpgroup::groupid() == 1) {
        warpgroup::increase_registers<168>();
        d_tt_scores tt_scores = tm_alloc.template allocate<d_tt_scores>(0);
        int scores_phase = 0;
        int rescale_phase = 1;
        uint32_t score_tt_base = tt_scores.addr + ((warpgroup::warpid() * 32) << 16);
        for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
            int3 t_coord = get_tile_idx(cur_bid);
            int m_tile = t_coord.y + cta_rank;
            if (m_tile >= tiles_m) {
                continue;
            }
            int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
            const float SCALE_LOG2 = qk_scale_log2(g, t_coord.x, t_coord.z);
            const bool ZERO_SCALE = (SCALE_LOG2 == 0.0f);
            float row_sum = 0.0f;
            float row_max = base_types::constants<float>::neg_infty();
            wait(rescale_finished[0], rescale_phase);
            rescale_phase ^= 1;
            for (int idx = 0; idx < iters_per_task; idx++) {
                const int buf = idx & 1;
                float2 scores_reg[C::Nb / 2];
                float tile_max = FP4PV_USE_FIXED_P_TILE_SCALE ? 1.0f : 0.0f;
                float p_group_amax[tk_localcta::SCALES_PER_CHUNK_X];
                #pragma unroll
                for (int g = 0; g < tk_localcta::SCALES_PER_CHUNK_X; ++g) {
                    p_group_amax[g] = 0.0f;
                }
                const int global_row = warpgroup::laneid();
                const int softmax_warp = warpgroup::laneid() / 32;
                wait(scores_arrived[0], scores_phase);
                if (FP4PV_DEBUG_TRAP_STAGE == 29 && blockIdx.x == 0 && cta_rank == 0 && idx == 0 && warpgroup::laneid() == 0) {
                    asm volatile("trap;");
                }
                #pragma unroll
                for (int ii = 0; ii < C::Nb / 32; ii++) {
                    asm volatile("{tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];}"
                            : "=f"(scores_reg[ii * 16 + 0].x), "=f"(scores_reg[ii * 16 + 0].y), "=f"(scores_reg[ii * 16 + 1].x), "=f"(scores_reg[ii * 16 + 1].y), "=f"(scores_reg[ii * 16 + 2].x), "=f"(scores_reg[ii * 16 + 2].y), "=f"(scores_reg[ii * 16 + 3].x), "=f"(scores_reg[ii * 16 + 3].y),
                              "=f"(scores_reg[ii * 16 + 4].x), "=f"(scores_reg[ii * 16 + 4].y), "=f"(scores_reg[ii * 16 + 5].x), "=f"(scores_reg[ii * 16 + 5].y), "=f"(scores_reg[ii * 16 + 6].x), "=f"(scores_reg[ii * 16 + 6].y), "=f"(scores_reg[ii * 16 + 7].x), "=f"(scores_reg[ii * 16 + 7].y),
                              "=f"(scores_reg[ii * 16 + 8].x), "=f"(scores_reg[ii * 16 + 8].y), "=f"(scores_reg[ii * 16 + 9].x), "=f"(scores_reg[ii * 16 + 9].y), "=f"(scores_reg[ii * 16 + 10].x), "=f"(scores_reg[ii * 16 + 10].y), "=f"(scores_reg[ii * 16 + 11].x), "=f"(scores_reg[ii * 16 + 11].y),
                              "=f"(scores_reg[ii * 16 + 12].x), "=f"(scores_reg[ii * 16 + 12].y), "=f"(scores_reg[ii * 16 + 13].x), "=f"(scores_reg[ii * 16 + 13].y), "=f"(scores_reg[ii * 16 + 14].x), "=f"(scores_reg[ii * 16 + 14].y), "=f"(scores_reg[ii * 16 + 15].x), "=f"(scores_reg[ii * 16 + 15].y)
                            : "r"(score_tt_base + ii * 32));
                }
                warpgroup::arrive(p_copy_done[buf]);
                if (idx >= m_tile) {
                    int causal_col = (idx > m_tile) ? -1 : (int)warpgroup::laneid();
                    #pragma unroll
                    for (int k = 0; k < C::Nb / 2; k++) {
                        if (k * 2 > causal_col) scores_reg[k].x = base_types::constants<float>::neg_infty();
                        if (k * 2 + 1 > causal_col) scores_reg[k].y = base_types::constants<float>::neg_infty();
                    }
                }
                float row_max_old = row_max;
                float lm0 = base_types::constants<float>::neg_infty(), lm1 = base_types::constants<float>::neg_infty(), lm2 = base_types::constants<float>::neg_infty(), lm3 = base_types::constants<float>::neg_infty();
                float lm4 = base_types::constants<float>::neg_infty(), lm5 = base_types::constants<float>::neg_infty(), lm6 = base_types::constants<float>::neg_infty(), lm7 = base_types::constants<float>::neg_infty();
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
                        constexpr float RESCALE_THRESHOLD = 8.f;
                        float acc_scale_log2 = (row_max_old - row_max) * SCALE_LOG2;
                        if (acc_scale_log2 >= -RESCALE_THRESHOLD) {
                            row_max = row_max_old;
                            acc_scale = 1.0f;
                        } else {
                            acc_scale = exp2f(acc_scale_log2);
                        }
                    }
                }
                if (FP4PV_DEBUG_PRINT_ROWSTATE && blockIdx.x == 0 && cta_rank == 1 &&
                    (warpgroup::laneid() == 64 || warpgroup::laneid() == 96) && idx < 2) {
                    const float q_sg_dbg = g.q_sg[{t_coord.x, t_coord.z, 0, 0}];
                    const float k_sg_dbg = g.k_sg[{t_coord.x, t_coord.z, 0, 0}];
                    printf("softmax idx=%d lane=%d scale_log2=%f zero_scale=%d q_sg=%f k_sg=%f row_max_old=%f row_max=%f acc_scale=%f row_sum_before=%f\n",
                           idx, (int)warpgroup::laneid(), SCALE_LOG2, int(ZERO_SCALE), q_sg_dbg, k_sg_dbg, row_max_old, row_max, acc_scale, row_sum);
                }

                float2 tile_sum0 = {0.0f, 0.0f}, tile_sum1 = {0.0f, 0.0f};
                if (!ZERO_SCALE) {
                    float neg_max_scaled = row_max * (-SCALE_LOG2);
                    float2 neg_max_scaled_2 = {neg_max_scaled, neg_max_scaled};
                    const float2 scale_2 = {SCALE_LOG2, SCALE_LOG2};
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        #pragma unroll
                        for (int jj = 0; jj < 16; jj++) {
                            int si = q * 16 + jj;
                            scores_reg[si] = __ffma2_rn(scores_reg[si], scale_2, neg_max_scaled_2);
                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE &&
                                          FP4PV_USE_FIXED_P_TILE_SCALE &&
                                          FP4PV_FUSE_P_GROUP_AMAX) {
                                const int group_idx = q * 2 + (jj >> 3);
                                p_group_amax[group_idx] = fmaxf(
                                    p_group_amax[group_idx],
                                    fmaxf(scores_reg[si].x, scores_reg[si].y));
                            }
                            if constexpr (!FP4PV_USE_FIXED_P_TILE_SCALE) {
                                tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            }
                            if ((jj & 1) == 0) tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                            else               tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        #pragma unroll
                        for (int jj = 0; jj < 16; jj++) {
                            int si = q * 16 + jj;
                            scores_reg[si].x = isfinite(scores_reg[si].x) ? 0.0f : scores_reg[si].x;
                            scores_reg[si].y = isfinite(scores_reg[si].y) ? 0.0f : scores_reg[si].y;
                            scores_reg[si].x = fp4pv_exp2_approx(scores_reg[si].x);
                            scores_reg[si].y = fp4pv_exp2_approx(scores_reg[si].y);
                            if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE &&
                                          FP4PV_USE_FIXED_P_TILE_SCALE &&
                                          FP4PV_FUSE_P_GROUP_AMAX) {
                                const int group_idx = q * 2 + (jj >> 3);
                                p_group_amax[group_idx] = fmaxf(
                                    p_group_amax[group_idx],
                                    fmaxf(scores_reg[si].x, scores_reg[si].y));
                            }
                            if constexpr (!FP4PV_USE_FIXED_P_TILE_SCALE) {
                                tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            }
                            if ((jj & 1) == 0) tile_sum0 = __fadd2_rn(tile_sum0, scores_reg[si]);
                            else               tile_sum1 = __fadd2_rn(tile_sum1, scores_reg[si]);
                        }
                    }
                }
                tile_sum0 = __fadd2_rn(tile_sum0, tile_sum1);
                if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE) {
                    if (idx >= m_tile || idx > 0) {
                        stats_store(max_vec_smem[0], warpgroup::laneid(), acc_scale);
                    }
                    warp::sync();
                    warp::arrive(corr_arrived[0]);
                    if constexpr (FP4PV_USE_FIXED_P_TILE_SCALE && FP4PV_FUSE_P_GROUP_AMAX) {
                        // P group amax is fused into softmax above; pack while the correction WG rescales the previous output.
                        fp4pv_pack_scores_to_stage_and_scales(
                            p_fp4_stage[buf], p_sc_stage[buf], scores_reg, p_group_amax, 1.0f, global_row);
                    }
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                    row_sum = row_sum * acc_scale + tile_sum0.x + tile_sum0.y;
                } else {
                    const float row_sum_old = row_sum;
                    row_sum = row_sum * acc_scale + tile_sum0.x + tile_sum0.y;
                    const float prev_contrib = row_sum_old * acc_scale;
                    float correction = 1.0f;
                    if (prev_contrib > 0.0f && row_sum > 0.0f && isfinite(prev_contrib) && isfinite(row_sum)) {
                        correction = prev_contrib / row_sum;
                    }
                    if (!(correction >= 0.0f) || !isfinite(correction)) {
                        correction = 0.0f;
                    }
                    stats_store(corr_vec_smem[0], warpgroup::laneid(), correction);
                    const float inv_row_sum = (row_sum > 0.0f && isfinite(row_sum)) ? __frcp_rn(row_sum) : 0.0f;
                    tile_max = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE) {
                            #pragma unroll
                            for (int jj = 0; jj < 16; jj++) {
                                const int si = q * 16 + jj;
                                scores_reg[si].x *= inv_row_sum;
                                scores_reg[si].y *= inv_row_sum;
                                tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                            }
                        } else {
                            __align__(8) bf16_2 scores_bf_reg[16];
                            #pragma unroll
                            for (int jj = 0; jj < 16; jj++) {
                                const int si = q * 16 + jj;
                                scores_reg[si].x *= inv_row_sum;
                                scores_reg[si].y *= inv_row_sum;
                                tile_max = fmaxf(tile_max, fmaxf(scores_reg[si].x, scores_reg[si].y));
                                scores_bf_reg[jj] = __float22bfloat162_rn(scores_reg[si]);
                            }
                            store_scores_quarter_to_localcta_scan(p_bf16_scan[buf], scores_bf_reg, q);
                        }
                    }
                    warp::sync();
                    warp::arrive(corr_arrived[0]);
                    wait(rescale_finished[0], rescale_phase);
                    rescale_phase ^= 1;
                }
                if constexpr (!(FP4PV_DIAG_USE_DIRECT_ROW_UPDATE && FP4PV_USE_FIXED_P_TILE_SCALE)) {
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
                }
                const float p_stage_amax =
                    (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE && FP4PV_USE_FIXED_P_TILE_SCALE) ? 1.0f : p_cta_amax[buf];
                if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE &&
                              FP4PV_USE_FIXED_P_TILE_SCALE &&
                              FP4PV_FUSE_P_GROUP_AMAX) {
                    (void)p_stage_amax;
                } else if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE) {
                    fp4pv_pack_scores_to_stage_and_scales(
                        p_fp4_stage[buf], p_sc_stage[buf], scores_reg, p_group_amax, p_stage_amax, global_row);
                } else {
                    fp4pv_pack_scores_to_stage_payload_from_localcta_scan_row(
                        p_fp4_stage[buf], p_bf16_scan[buf], p_stage_amax, global_row);
                    fp4pv_store_scales_from_localcta_scan_row(
                        p_sc_stage[buf], p_bf16_scan[buf], p_stage_amax, global_row);
                }
                if (idx >= m_tile) {
                    const int valid_groups = (idx > m_tile) ? 0 : ((warpgroup::laneid() / 16) + 1);
                    if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE) {
                        fp4pv_zero_invalid_causal_groups_swizzled(
                            p_fp4_stage[buf], p_sc_stage[buf], global_row, valid_groups);
                    } else {
                        fp4pv_zero_invalid_causal_groups(
                            p_fp4_stage[buf], p_sc_stage[buf], global_row, valid_groups);
                    }
                }
                tk_localcta::subgroup_barrier_sync<128>();
                publish_shared_backing();
                if constexpr (FP4PV_DIAG_CLUSTER_FENCE_P_STAGE) {
                    publish_cluster_shared_backing_if_needed<C>();
                }
                tk_localcta::subgroup_barrier_sync<128>();
                if constexpr (FP4PV_DIAG_USE_DIRECT_ROW_UPDATE) {
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(p_remote_ready[buf], 0);
                        }
                    }
                    warpgroup::arrive(p_quant_ready[buf]);
                } else {
                    warpgroup::arrive(p_pack_ready[buf]);
                }
                if (FP4PV_DEBUG_PRINT_ROWSTATE && blockIdx.x == 0 && cta_rank == 1 &&
                    (warpgroup::laneid() == 64 || warpgroup::laneid() == 96) && idx < 2) {
                    printf("softmax idx=%d lane=%d row_sum_after=%f tile_sum=%f acc_scale=%f\n",
                           idx, (int)warpgroup::laneid(), row_sum, tile_sum0.x + tile_sum0.y, acc_scale);
                }
                scores_phase ^= 1;
            }
            stats_store(lse_smem[0], warpgroup::laneid(), row_max);
            stats_store(max_vec_smem[0], warpgroup::laneid(), row_sum);
            warp::sync();
            warp::arrive(corr_arrived[0]);
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 20 || FP4PV_DEBUG_TRAP_STAGE == 25) {
            warpgroup::sync(warpgroup::groupid()+1);
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 13 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
            asm volatile("trap;");
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 18 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 127) {
            asm volatile("trap;");
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 20 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
            asm volatile("trap;");
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 25 && blockIdx.x == 1 && cta_rank == 1 && warpgroup::laneid() == 0) {
            asm volatile("trap;");
        }
    }
    else {
        if constexpr (!FP4PV_DIAG_USE_DIRECT_ROW_UPDATE) {
            int p_pack_phase_mask = 0;
            for (int cur_bid = blockIdx.x; cur_bid < total_bids; cur_bid += gridDim.x) {
                int3 t_coord = get_tile_idx(cur_bid);
                if (t_coord.y + cta_rank >= tiles_m) {
                    continue;
                }
                int iters_per_task = min(t_coord.y + C::CLUSTER_SIZE, tiles_m);
                (void)t_coord;
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const int buf = idx & 1;
                    const int p_pack_phase = (p_pack_phase_mask >> buf) & 1;
                    wait(p_pack_ready[buf], p_pack_phase);
                    p_pack_phase_mask ^= (1 << buf);
                    if (FP4PV_DEBUG_TRAP_STAGE == 30 && blockIdx.x == 0 && cta_rank == 0 && idx == 0 && warpgroup::laneid() == 0) {
                        asm volatile("trap;");
                    }
                    if (FP4PV_DEBUG_TRAP_STAGE == 28 && blockIdx.x == 1 && cta_rank == 1 && idx == 0 && warpgroup::laneid() == 0) {
                        asm volatile("trap;");
                    }
                    if (FP4PV_DEBUG_TRAP_STAGE == 1 && blockIdx.x == 0 && cta_rank == 0 && idx == 0 && warpgroup::laneid() == 0) {
                        asm volatile("trap;");
                    }
                    if (FP4PV_DEBUG_TRAP_STAGE == 27 && blockIdx.x == 1 && cta_rank == 1 && idx == 0 && warpgroup::laneid() == 0) {
                        asm volatile("trap;");
                    }
                    finalize_localcta_p_scales_direct(
                        p_sc_stage[buf],
                        warpgroup::laneid());
                    if constexpr (FP4PV_DIAG_CLUSTER_FENCE_P_STAGE) {
                        publish_cluster_shared_backing_if_needed<C>();
                        tk_localcta::subgroup_barrier_sync<128>();
                    }
                    if constexpr (C::CLUSTER_SIZE > 1) {
                        if (cta_rank == 1 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                            tma::cluster::arrive(p_remote_ready[buf], 0);
                        }
                    }
                    warpgroup::arrive(p_quant_ready[buf]);
                    if (FP4PV_DEBUG_TRAP_STAGE == 2 && blockIdx.x == 0 && cta_rank == 0 && idx == 0 && warpgroup::laneid() == 0) {
                        asm volatile("trap;");
                    }
                }
            }
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 21 || FP4PV_DEBUG_TRAP_STAGE == 26) {
            warpgroup::sync(warpgroup::groupid()+1);
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 15 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
            asm volatile("trap;");
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 19 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 127) {
            asm volatile("trap;");
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 21 && blockIdx.x == 0 && cta_rank == 0 && warpgroup::laneid() == 0) {
            asm volatile("trap;");
        }
        if (FP4PV_DEBUG_TRAP_STAGE == 26 && blockIdx.x == 1 && cta_rank == 1 && warpgroup::laneid() == 0) {
            asm volatile("trap;");
        }
    }

    if (FP4PV_DEBUG_TRAP_STAGE == 23) {
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 1 && cta_rank == 1) {
            asm volatile("trap;");
        }
    }
}


template <typename GL>
GL tensor_to_bh_scalar_gl(const at::Tensor &t, int batch, int heads, const char *name) {
    CHECK_CUDA(t);
    CHECK_CONTIGUOUS(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Float, name, " must be float32");
    if (t.dim() == 2) {
        TORCH_CHECK(t.size(0) == batch && t.size(1) == heads, name, " must have shape (batch, heads)");
        return kittens::py::tensor_to_gl<GL>(t, batch, heads, 1, 1);
    }
    TORCH_CHECK(t.dim() == 4, name, " must have shape (batch, heads) or (batch, heads, 1, 1)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads && t.size(2) == 1 && t.size(3) == 1,
                name, " must have shape (batch, heads, 1, 1)");
    return kittens::py::tensor_to_gl<GL>(t);
}

template <typename GL>
GL tensor_to_fp4_scale_gl(const at::Tensor &t, int batch, int depth, int rows, const char *name) {
    CHECK_CUDA(t);
    CHECK_CONTIGUOUS(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e4m3fn, name, " must be float8_e4m3fn");
    TORCH_CHECK(t.dim() == 4, name, " must have rank 4");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == depth && t.size(2) == rows && t.size(3) == 512,
                name, " must have shape (batch, depth_tiles, rows, 512)");
    return kittens::py::tensor_to_gl<GL, false>(t, batch, depth, rows, 256);
}

inline void check_fp4pv_v_tensor(const at::Tensor &t, int batch, int heads, int seqlen, int dvo) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(t.dim() == 4, "V_fp4 must have shape (batch, heads, dvo, seqlen/2)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads && t.size(2) == dvo && t.size(3) == seqlen / 2,
                "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
}

inline void check_fp4pv_v_scale_tensor(const at::Tensor &t, int batch, int heads, int seqlen, int dvo) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e4m3fn, "V_sc_prepared must be float8_e4m3fn");
    TORCH_CHECK(t.dim() == 5, "V_sc_prepared must have shape (batch, heads, dvo/128, seqlen/64, 512)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads && t.size(2) == dvo / 128 && t.size(3) == seqlen / 64 && t.size(4) == 512,
                "V_sc_prepared must have shape (batch, heads, 1, seqlen/64, 512)");
}

inline void check_native_group2_mxfp4_v_scale_tensor(
    const at::Tensor &t, int batch, int heads, int seqlen, int dvo) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Byte,
                "B2 MXFP4 V_sc must be uint8 E8M0");
    TORCH_CHECK(t.dim() == 6,
                "B2 MXFP4 V_sc must have shape "
                "(batch, heads, scale_rows, seqlen/128, 32, 16)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads &&
                    t.size(2) >= dvo / 128 &&
                    t.size(3) == seqlen / 128 &&
                    t.size(4) == 32 && t.size(5) == 16,
                "B2 MXFP4 V_sc must have shape "
                "(batch, heads, >=dvo/128, seqlen/128, 32, 16)");
}

#if !defined(TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH)
template <bool persistent,
          bool force_native_group2_fullgrid = false,
          bool selected_row_trace = false>
void dispatch(at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
              at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
              at::Tensor V, at::Tensor O, at::Tensor LSE) {
    if constexpr (!persistent) {
        // The persistent schedule is currently both correct and faster for the
        // supported causal FP4 kernel family, so route `forward` through it.
        dispatch<true>(Q, Q_sc, Q_sg, K, K_sc, K_sg, V, O, LSE);
        return;
    }

    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V); CHECK_INPUT(O); CHECK_INPUT(LSE);

    using C = config<128, 128, 192, 128, persistent, false,
                     force_native_group2_fullgrid, selected_row_trace>;
    using G = globals<C>;

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(V.dtype() == at::ScalarType::BFloat16, "V must be bfloat16");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(V.dim() == 4 && O.dim() == 4, "V and O must be rank-4 tensors in (batch, seqlen, heads, dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch && V.size(0) == batch && O.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads && V.size(2) == heads && O.size(2) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen && V.size(1) == seqlen && O.size(1) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == C::Dqk / 2 && K.size(3) == C::Dqk / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(V.size(3) == C::Dvo && O.size(3) == C::Dvo, "V and O head_dim must be 128");
    TORCH_CHECK(seqlen % C::Mb == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, seqlen / (C::Nb / C::CLUSTER_SIZE), heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_gl>(V),
        typename G::v_fp4_gl{},
        nullptr,
        0,
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE)
    };

    CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    dim3 launch_grid = g.grid();
    if constexpr (force_native_group2_fullgrid) {
        static_assert(persistent,
                      "native group2 full-grid launch requires the one-softmaxer paired config");
        const int tiles_m = seqlen / C::Mb;
        const int num_clusters = (tiles_m + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE;
        launch_grid = dim3(batch * heads * num_clusters * C::CLUSTER_SIZE);
        TORCH_CHECK(launch_grid.x % C::CLUSTER_SIZE == 0,
                    "native group2 full-grid launch must contain complete CTA pairs");
    }
    LaunchConfig<true, false> launch_config(
        launch_grid, g.block(), g.dynamic_shared_memory(), 0, C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
}

// B1 is an attributable speed-of-light control for the established hybrid:
// native cta_group::2 FP4 QK, real BF16 softmax/P, native cta_group::2 BF16
// PV, and the existing vectorized normalization/store epilogue.  Its retained
// BF16 V input is deliberately a changed contract and must not be reported as
// an apples-to-apples production-input result.
void dispatch_native_group2_fp4qk_bf16pv_group2_b1(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V, at::Tensor O, at::Tensor LSE
) {
    TORCH_CHECK(Q.dim() == 4, "Q must be rank 4");
    TORCH_CHECK(Q.size(2) % 256 == 0,
                "B1 requires complete M256 CTA pairs; odd M128 tails are rejected");
    dispatch<true, true>(Q, Q_sc, Q_sg, K, K_sc, K_sg, V, O, LSE);
}

#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
void dispatch_native_group2_fp4qk_bf16pv_group2_b1_selected_trace(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V, at::Tensor O, at::Tensor LSE
) {
    TORCH_CHECK(Q.dim() == 4, "Q must be rank 4");
    TORCH_CHECK(Q.size(2) % 256 == 0,
                "B1 selected-row trace requires complete M256 CTA pairs");
    dispatch<true, true, true>(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V, O, LSE);
}
#endif

// B2 keeps the production MXFP4 V payload and uint8 E8M0 scale ABI, reconstructs
// rank-local BF16 V in SMEM, then enters exactly the same native group2 BF16
// PV and vectorized epilogue used by B1.  This isolates FP4 V/scale staging
// from the full FP4 P/PV ownership protocol without changing P arithmetic.
void dispatch_native_group2_fp4qk_fp4v_dequant_bf16pv_group2_b2(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc_prepared,
    at::Tensor O, at::Tensor LSE
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(O); CHECK_INPUT(LSE);

    using C = config<128, 128, 192, 128, true, true, true>;
    using G = globals<C>;
    static_assert(C::persistent && C::FP4_V_DEQUANT && C::CLUSTER_SIZE == 2);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4,
                "Q and K must be rank-4 (batch, heads, seqlen, packed_dim)");
    TORCH_CHECK(O.dim() == 4, "O must be rank-4 (batch, seqlen, heads, dim)");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(K.size(0) == batch && O.size(0) == batch,
                "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads && O.size(2) == heads,
                "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen && O.size(1) == seqlen,
                "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == C::Dqk / 2 && K.size(3) == C::Dqk / 2,
                "Q and K packed head_dim must be 96");
    TORCH_CHECK(O.size(3) == C::Dvo, "O head_dim must be 128");
    TORCH_CHECK(seqlen % (C::Mb * C::CLUSTER_SIZE) == 0,
                "B2 requires complete M256 CTA pairs; odd M128 tails are rejected");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch &&
                    LSE.size(1) == heads && LSE.size(2) == 1 &&
                    LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, C::Dvo);
    check_native_group2_mxfp4_v_scale_tensor(
        V_sc_prepared, batch, heads, seqlen, C::Dvo);

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(
            Q_sc, batch, seqlen / C::Mb,
            heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(
            Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(
            K_sc, batch, seqlen / (C::Nb / C::CLUSTER_SIZE),
            heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(
            K_sg, batch, heads, "K_sg"),
        typename G::v_gl{},
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        static_cast<int>(V_sc_prepared.size(2)),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE)
    };

    CUDACHECK(cudaFuncSetAttribute(
        kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    const int tiles_m = seqlen / C::Mb;
    const int num_clusters = (tiles_m + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE;
    const dim3 launch_grid(batch * heads * num_clusters * C::CLUSTER_SIZE);
    TORCH_CHECK(launch_grid.x % C::CLUSTER_SIZE == 0,
                "B2 native group2 full-grid launch must contain complete CTA pairs");
    LaunchConfig<true, false> launch_config(
        launch_grid, g.block(), g.dynamic_shared_memory(), 0, C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
}

void dispatch_fp4pv(at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
                    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
                    at::Tensor V_fp4, at::Tensor V_sc_prepared,
                    at::Tensor O, at::Tensor LSE) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(O); CHECK_INPUT(LSE);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(O.dim() == 4, "O must be rank-4");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch && O.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads && O.size(2) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen && O.size(1) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(O.size(3) == 128, "O head_dim must be 128");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");

    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_v_scale_tensor(V_sc_prepared, batch, heads, seqlen, 128);

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto launch_fp4pv = [&](auto cfg_tag) {
        using C = decltype(cfg_tag);
        using G = globals_fp4pv<C, !FP4PV_DIAG_USE_DIRECT_ROW_UPDATE>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE)
        };

        CUDACHECK(cudaFuncSetAttribute(kernel_fp4pv<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
        LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), 0, C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_fp4pv<C>, g));
    };

    launch_fp4pv(config_fp4pv<128, 128, 192, 128, 2>{});
}

template <typename C>
static inline dim3 fullgrid_dim_fp4pv(int batch, int heads, int seqlen) {
    const int tiles_m = seqlen / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
    return dim3(total_bids);
}

void dispatch_fp4pv_fullgrid(at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
                             at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
                             at::Tensor V_fp4, at::Tensor V_sc_prepared,
                             at::Tensor O, at::Tensor LSE) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(O); CHECK_INPUT(LSE);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(O.dim() == 4, "O must be rank-4");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch && O.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads && O.size(2) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen && O.size(1) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(O.size(3) == 128, "O head_dim must be 128");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");

    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_v_scale_tensor(V_sc_prepared, batch, heads, seqlen, 128);

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_fp4pv<C, !FP4PV_DIAG_USE_DIRECT_ROW_UPDATE>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE)
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_fp4pv<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(fullgrid_dim_fp4pv<C>(batch, heads, seqlen), g.block(), g.dynamic_shared_memory(), 0, C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_fp4pv<C>, g));
}

#endif

#if TK_FA4_SINGLE_CTA_FP4QK_BF16PV_ONLY_BUILD
template <typename C>
static inline void check_single_cta_fp4qk_bf16pv_common(
    const at::Tensor &Q, const at::Tensor &Q_sc, const at::Tensor &Q_sg,
    const at::Tensor &K, const at::Tensor &K_sc, const at::Tensor &K_sg,
    const at::Tensor &O, const at::Tensor &LSE) {
    static_assert(C::CLUSTER_SIZE == 1);
    static_assert(C::SINGLE_CTA_FP4QK_BF16PV);
    static_assert(C::Mb == 128 && C::Nb == 128 && C::Dqk == 192 &&
                  C::Dvo == 128);
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(O); CHECK_INPUT(LSE);
    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16,
                "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float,
                "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4,
                "Q and K must be rank-4");
    TORCH_CHECK(O.dim() == 4, "O must be rank-4");
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(K.size(0) == batch && O.size(0) == batch,
                "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads && O.size(2) == heads,
                "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen && O.size(1) == seqlen,
                "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == C::Dqk / 2 &&
                    K.size(3) == C::Dqk / 2,
                "Q and K packed head_dim must be 96");
    TORCH_CHECK(O.size(3) == C::Dvo, "O head_dim must be 128");
    TORCH_CHECK(seqlen % C::Mb == 0,
                "single-CTA route requires complete M128 query tiles");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch &&
                    LSE.size(1) == heads && LSE.size(2) == 1 &&
                    LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
}

void dispatch_single_cta_fp4qk_bf16v_bf16pv_ceiling(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V, at::Tensor O, at::Tensor LSE) {
    using C = config_single_cta_fp4qk_bf16v_bf16pv_ceiling;
    using G = globals<C>;
    check_single_cta_fp4qk_bf16pv_common<C>(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, O, LSE);
    CHECK_INPUT(V);
    TORCH_CHECK(V.dtype() == at::ScalarType::BFloat16,
                "V must be bfloat16 for the ceiling route");
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(V.dim() == 4 && V.size(0) == batch &&
                    V.size(1) == seqlen && V.size(2) == heads &&
                    V.size(3) == C::Dvo,
                "V must have shape (batch, seqlen, heads, 128)");
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(
            Q_sc, batch, seqlen / C::Mb,
            heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(
            Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(
            K_sc, batch, seqlen / (C::Nb / 2),
            heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(
            K_sg, batch, heads, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_gl>(V),
        typename G::v_fp4_gl{}, nullptr, 0,
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE)};
    CUDACHECK(cudaFuncSetAttribute(
        kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    const dim3 grid(batch * heads * (seqlen / C::Mb));
    LaunchConfig<true, false> launch_config(
        grid, g.block(), g.dynamic_shared_memory(), 0, 1);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
}

void dispatch_single_cta_fp4qk_fp4v_dequant_bf16pv(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp4, at::Tensor V_sc_prepared,
    at::Tensor O, at::Tensor LSE) {
    using C = config_single_cta_fp4qk_fp4v_dequant_bf16pv;
    using G = globals<C>;
    check_single_cta_fp4qk_bf16pv_common<C>(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, O, LSE);
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, C::Dvo);
    check_native_group2_mxfp4_v_scale_tensor(
        V_sc_prepared, batch, heads, seqlen, C::Dvo);
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(
            Q_sc, batch, seqlen / C::Mb,
            heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(
            Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(
            K_sc, batch, seqlen / (C::Nb / 2),
            heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(
            K_sg, batch, heads, "K_sg"),
        typename G::v_gl{},
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        static_cast<int>(V_sc_prepared.size(2)),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE)};
    CUDACHECK(cudaFuncSetAttribute(
        kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    const dim3 grid(batch * heads * (seqlen / C::Mb));
    LaunchConfig<true, false> launch_config(
        grid, g.block(), g.dynamic_shared_memory(), 0, 1);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
}
#endif

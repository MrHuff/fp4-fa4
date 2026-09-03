#pragma once

#include <array>
#include <cstdint>
#include <optional>

#ifndef TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
#define TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH 0
#endif
static_assert(
    TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH == 0 ||
    TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH == 1);

constexpr std::size_t HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_CAPACITY = 256;
constexpr std::size_t HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_WAYS = 4;
constexpr std::size_t HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_SETS =
    HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_CAPACITY /
    HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_WAYS;
constexpr std::size_t HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_GL_SLOTS = 10;
constexpr std::uint64_t HAO_DIRECT_FP4PV_DESCRIPTOR_DEVICE_MIX =
    0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t HAO_DIRECT_FP4PV_DESCRIPTOR_HASH_MIX_1 =
    0xbf58476d1ce4e5b9ULL;
constexpr std::uint64_t HAO_DIRECT_FP4PV_DESCRIPTOR_HASH_MIX_2 =
    0x94d049bb133111ebULL;
static_assert(
    HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_CAPACITY %
            HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_WAYS ==
        0);
static_assert(HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_SETS == 64);

struct hao_direct_fp4pv_descriptor_cache_counters {
    unsigned long long hits = 0;
    unsigned long long misses = 0;
    unsigned long long evictions = 0;
    unsigned long long clears = 0;
};

// Counters deliberately have the same scope as the cache. Python launches and
// topology reads normally use the same host thread, while avoiding atomics on
// every attention layer keeps the fixed-route fast path lightweight.
static thread_local hao_direct_fp4pv_descriptor_cache_counters
    hao_direct_fp4pv_cache_counters;

template <std::size_t GlSlot, typename Descriptor, typename Builder>
inline Descriptor hao_direct_fp4pv_cached_gl(
    int device,
    const void *data_pointer,
    Builder build_descriptor
) {
#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
    static_assert(
        GlSlot < HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_GL_SLOTS);
    struct descriptor_cache_key {
        int device;
        const void *data_pointer;

        bool operator==(const descriptor_cache_key &other) const {
            return device == other.device &&
                data_pointer == other.data_pointer;
        }
    };
    struct descriptor_cache_entry {
        descriptor_cache_key key;
        Descriptor descriptor;
    };
    struct descriptor_cache_set {
        std::array<
            std::optional<descriptor_cache_entry>,
            HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_WAYS>
            entries;
        std::size_t next_victim = 0;
    };
    struct descriptor_cache_state {
        std::array<
            descriptor_cache_set,
            HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_SETS>
            sets;
    };
    const descriptor_cache_key key{device, data_pointer};
    // Avalanche the full data pointer so regularly spaced CUDA allocations do
    // not alias one four-way set. Keeping every address bit also distributes
    // contiguous tensor views whose data pointers are not allocation-aligned.
    const std::uint64_t device_key =
        static_cast<std::uint64_t>(static_cast<unsigned int>(device)) + 1ULL;
    std::uint64_t hash_key =
        static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(data_pointer)) ^
        (device_key * HAO_DIRECT_FP4PV_DESCRIPTOR_DEVICE_MIX);
    hash_key = (hash_key ^ (hash_key >> 30)) *
        HAO_DIRECT_FP4PV_DESCRIPTOR_HASH_MIX_1;
    hash_key = (hash_key ^ (hash_key >> 27)) *
        HAO_DIRECT_FP4PV_DESCRIPTOR_HASH_MIX_2;
    hash_key ^= hash_key >> 31;
    const std::size_t set_index = static_cast<std::size_t>(
        hash_key & (HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_SETS - 1));
    static thread_local descriptor_cache_state descriptor_cache;
    auto &cache_set = descriptor_cache.sets[set_index];
    for (const auto &slot : cache_set.entries) {
        if (slot.has_value() && slot->key == key) {
            ++hao_direct_fp4pv_cache_counters.hits;
            return slot->descriptor;
        }
    }
    ++hao_direct_fp4pv_cache_counters.misses;
    Descriptor descriptor = build_descriptor();
    for (auto &slot : cache_set.entries) {
        if (!slot.has_value()) {
            slot.emplace(descriptor_cache_entry{key, descriptor});
            return descriptor;
        }
    }
    // Set-local replacement bounds work without flushing reusable descriptors
    // in other GL slots or sets. Descriptors own no Tensor/device allocation.
    auto &victim = cache_set.entries[cache_set.next_victim];
    cache_set.next_victim =
        (cache_set.next_victim + 1) %
        HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_WAYS;
    victim.emplace(descriptor_cache_entry{key, descriptor});
    ++hao_direct_fp4pv_cache_counters.evictions;
    return descriptor;
#else
    (void)device;
    (void)data_pointer;
    return build_descriptor();
#endif
}

struct hao_direct_topology_state {
    long long valid = 0;
    long long physical_grid = 0;
    long long logical_jobs = 0;
    long long threads = 0;
    long long kv_stages = 0;
    long long tmem_columns = 0;
    long long dynamic_smem = 0;
};

static std::mutex hao_direct_topology_mutex;
static hao_direct_topology_state hao_direct_last_topology;
#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
// Candidate extensions are shape- and route-specialized. Cache host launch
// invariants per calling thread and CUDA device so multi-device callers stay
// correct without paying cudaFuncSetAttribute on every layer.
static thread_local int
    hao_direct_fp4pv_initialized_device = -1;
#endif

constexpr int HAO_DIRECT_N96_LIFECYCLE_SLOTS = 64;
static std::once_flag hao_direct_n96_lifecycle_once;
static volatile int *hao_direct_n96_lifecycle_host = nullptr;
static int *hao_direct_n96_lifecycle_device = nullptr;

void init_hao_direct_n96_lifecycle() {
#if TK_HAO_DIRECT_N96_LIFECYCLE_DEBUG
    std::call_once(hao_direct_n96_lifecycle_once, [] {
        void *host_pointer = nullptr;
        CUDACHECK(cudaHostAlloc(
            &host_pointer,
            HAO_DIRECT_N96_LIFECYCLE_SLOTS * sizeof(int),
            cudaHostAllocMapped | cudaHostAllocPortable));
        void *device_pointer = nullptr;
        CUDACHECK(cudaHostGetDevicePointer(
            &device_pointer,
            host_pointer,
            0));
        hao_direct_n96_lifecycle_host =
            static_cast<volatile int *>(host_pointer);
        hao_direct_n96_lifecycle_device =
            static_cast<int *>(device_pointer);
    });
#endif
}

void reset_hao_direct_n96_lifecycle() {
    init_hao_direct_n96_lifecycle();
#if TK_HAO_DIRECT_N96_LIFECYCLE_DEBUG
    for (int i = 0; i < HAO_DIRECT_N96_LIFECYCLE_SLOTS; ++i) {
        hao_direct_n96_lifecycle_host[i] = 0;
    }
#endif
}

pybind11::list read_hao_direct_n96_lifecycle() {
    init_hao_direct_n96_lifecycle();
    pybind11::list out;
    for (int i = 0; i < HAO_DIRECT_N96_LIFECYCLE_SLOTS; ++i) {
#if TK_HAO_DIRECT_N96_LIFECYCLE_DEBUG
        out.append(hao_direct_n96_lifecycle_host[i]);
#else
        out.append(0);
#endif
    }
    return out;
}

pybind11::dict read_hao_direct_topology() {
    std::lock_guard<std::mutex> lock(
        hao_direct_topology_mutex);
    const auto &s = hao_direct_last_topology;
    pybind11::dict out;
    out["schema"] = "tk_hao_direct_pipeline_v1";
    out["valid"] = s.valid;
    out["route"] = TK_HAO_DIRECT_ROUTE_NAME;
    out["fixed_route_fastpath"] =
        bool(TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH);
    out["route_env_guard_per_launch"] =
        !bool(TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH);
    out["kernel_attribute_init"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? "once_per_host_thread_and_cuda_device"
            : "per_launch";
    out["tma_descriptor_cache"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? "bounded_thread_local_gl_descriptors"
            : "disabled";
    out["tma_descriptor_cache_capacity"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_CAPACITY
            : 0;
    out["tma_descriptor_cache_lookup"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? "splitmix64_device_pointer_four_way_set_associative"
            : "disabled";
    out["tma_descriptor_cache_set_hash"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? "splitmix64_device_pointer_v1"
            : "disabled";
    out["tma_descriptor_cache_sets"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_SETS
            : 0;
    out["tma_descriptor_cache_ways"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_WAYS
            : 0;
    out["tma_descriptor_cache_capacity_scope"] =
        "per_compile_time_gl_slot";
    out["tma_descriptor_cache_gl_slots"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_GL_SLOTS
            : 0;
    out["tma_descriptor_cache_total_entry_ceiling"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_CAPACITY *
                HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_GL_SLOTS
            : 0;
    out["tma_descriptor_cache_key"] =
        "cuda_device_data_ptr_and_compile_time_gl_slot";
    out["tma_descriptor_cache_owns_tensors"] = false;
    out["tma_descriptor_cache_counter_scope"] =
        "calling_host_thread";
    out["tma_descriptor_cache_hits"] =
        hao_direct_fp4pv_cache_counters.hits;
    out["tma_descriptor_cache_misses"] =
        hao_direct_fp4pv_cache_counters.misses;
    out["tma_descriptor_cache_evictions"] =
        hao_direct_fp4pv_cache_counters.evictions;
    out["tma_descriptor_cache_clears"] =
        hao_direct_fp4pv_cache_counters.clears;
    out["batch"] = hao_direct_config::BATCH;
    out["heads"] = hao_direct_config::HEADS;
    out["kv_heads"] = hao_direct_config::KV_HEADS;
    out["seqlen"] = hao_direct_config::SEQ_LEN;
    out["query_tile"] = 128;
    out["key_tile"] = hao_direct_config::Nb;
    out["key_tiles"] = hao_direct_config::N_TILES;
    out["causal"] = hao_direct_config::HAO_DIRECT_CAUSAL;
    out["causal_stage0_tail_skip"] =
        hao_direct_config::HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP;
    out["causal_diagonal_mask"] =
        hao_direct_config::HAO_DIRECT_CAUSAL_DIAGONAL_MASK;
    out["causal_interleaved_kv"] =
        hao_direct_config::HAO_DIRECT_CAUSAL_INTERLEAVED_KV;
    out["sm103_n256"] =
        hao_direct_config::HAO_DIRECT_SM103_N256;
    out["sm103_n256_dual_query"] =
        hao_direct_config::HAO_DIRECT_SM103_N256_DUAL_QUERY;
    out["sm103_n192_early_k96"] =
        hao_direct_config::HAO_DIRECT_SM103_N192_EARLY_K96;
    out["sm103_n192_dedicated_p0"] =
        hao_direct_config::HAO_DIRECT_SM103_N192_DEDICATED_P0;
    out["sm103_n256_prefix_regs"] =
        hao_direct_config::HAO_DIRECT_SM103_N256_PREFIX_REGS;
    out["sm103_n256_direct_pv_handoff"] =
        hao_direct_config::HAO_DIRECT_SM103_N256_DIRECT_PV_HANDOFF;
    out["sm103_n256_chain_qk1"] =
        hao_direct_config::HAO_DIRECT_SM103_N256_CHAIN_QK1;
    out["sm103_n256_split_output_scales"] =
        hao_direct_config::HAO_DIRECT_SM103_N256_SPLIT_OUTPUT_SCALES;
    out["sm103_n256_compact_v"] =
        hao_direct_config::HAO_DIRECT_SM103_N256_COMPACT_V;
    out["sm103_n256_async_prefix_restore"] =
        hao_direct_config::HAO_DIRECT_SM103_N256_ASYNC_PREFIX_RESTORE;
    out["tail_keys"] = hao_direct_config::N96_TAIL_KEYS;
    out["dqk"] = hao_direct_config::Dqk;
    out["dvo"] = hao_direct_config::Dvo;
    out["logical_jobs"] = s.logical_jobs;
    out["physical_grid_ctas"] = s.physical_grid;
    out["compiled_num_sm"] = hao_direct_config::NUM_SM;
    out["threads_per_cta"] = s.threads;
    out["cluster_size"] = hao_direct_config::CLUSTER_SIZE;
    out["query_stages"] = hao_direct_config::QUERY_STAGES;
    out["kv_stages"] = s.kv_stages;
    out["tmem_columns"] = s.tmem_columns;
    out["dynamic_smem_bytes"] = s.dynamic_smem;
    out["persistent"] = true;
    out["p_first_chunks"] =
        hao_direct_config::P_FIRST_CHUNKS;
    out["kv_reuse_ceiling"] =
        hao_direct_config::HAO_DIRECT_KV_REUSE_CEILING;
    out["early_p_k64"] =
        hao_direct_config::HAO_DIRECT_EARLY_P &&
        !hao_direct_config::HAO_DIRECT_SPARSE_PV;
    out["sparse_pv"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV;
    out["sparse_pattern"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV
            ? "paired_4of8"
            : "dense";
    out["sparse_selector_mode"] =
        hao_direct_config::HAO_DIRECT_SPARSE_SELECTOR_MODE;
    out["sparse_fixed_selector"] =
        hao_direct_config::HAO_DIRECT_SPARSE_SELECTOR_MODE;
    out["sparse_split_k128"] =
        hao_direct_config::HAO_DIRECT_SPARSE_SPLIT_K128;
    out["sparse_issue_count"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV
            ? (hao_direct_config::HAO_DIRECT_SPARSE_SPLIT_K128 ? 2 : 1)
            : 0;
    out["sparse_logical_k_per_issue"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV ? 128 : 0;
    out["sparse_early_half"] =
        hao_direct_config::HAO_DIRECT_SPARSE_SPLIT_K128;
    out["mx_disjoint_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_MX_EARLY_P;
    out["mx_direct_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_MX_EARLY_DIRECT_SCALE;
    out["mx_register_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF;
    out["mx_pair_scale_scratch"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH;
    out["mx_async_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_MX_EARLY_ASYNC_SCALE;
    out["mx_shiftless_corr_bypass"] =
        hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS;
    out["mx_defer_job_stats_wait"] =
        hao_direct_config::HAO_DIRECT_MX_DEFER_JOB_STATS_WAIT;
    out["mx_direct_scale_scratch"] =
        hao_direct_config::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH;
    out["mx_split_stage_pv"] =
        hao_direct_config::HAO_DIRECT_MX_SPLIT_STAGE_PV;
    out["mx_qk_scale_preload"] =
        hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD;
    out["mx_qk_scale_preload_mask"] =
        hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_MASK;
    out["mx_qk_scale_preload_before_p"] =
        hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_BEFORE_P;
    out["mx_qk_scale_offload"] =
        hao_direct_config::HAO_DIRECT_MX_QK_SCALE_OFFLOAD;
    out["d64_detached_p"] =
        hao_direct_config::HAO_DIRECT_D64_DETACHED_P;
    out["d64_p_buffer_mode"] =
        hao_direct_config::HAO_DIRECT_D64_P_BUFFER_MODE;
    out["mx_dual_issuer"] =
        hao_direct_config::HAO_DIRECT_MX_DUAL_ISSUER;
    out["nv_qk_folded_k64_scales"] =
        hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES;
    out["nv_qk_folded_k64_scale_mask"] =
        hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK;
    out["nv_qk_compact_folded_scales"] =
        hao_direct_config::HAO_DIRECT_NV_QK_COMPACT_FOLDED_SCALES;
    out["nv_qk_preload_page_mask"] =
        hao_direct_config::HAO_DIRECT_NV_QK_PRELOAD_PAGE_MASK;
    out["mx_q3_corr_wg"] =
        hao_direct_config::HAO_DIRECT_MX_Q3_CORR_WG;
    out["mx_dual_q3_corr_wg"] =
        hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_CORR_WG;
    out["mx_dual_q3_smem_wg"] =
        hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG;
    out["mx_dual_q3_tmem_wg"] =
        hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG;
    out["fused_pv_issue"] =
        hao_direct_config::HAO_DIRECT_FUSED_PV_ISSUE;
    out["log2_p_quant"] =
        hao_direct_config::HAO_DIRECT_LOG2_QUANT;
    out["mx_log2_p_quant"] =
        hao_direct_config::HAO_DIRECT_MX_LOG2_QUANT;
    out["mx_scale_select"] =
        hao_direct_config::HAO_DIRECT_MX_SCALE_SELECT;
    out["mx_skip_zero_scale_mask"] =
        hao_direct_config::HAO_DIRECT_MX_SKIP_ZERO_SCALE_MASK;
    out["mx_direct_log_code"] =
        hao_direct_config::HAO_DIRECT_MX_DIRECT_LOG_CODE;
    out["mx_direct_log_code_mode"] =
        hao_direct_config::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE;
    out["mx_log_lut_bits"] =
        hao_direct_config::HAO_DIRECT_MX_LOG_LUT_BITS;
    out["mx_log_lut_bit_code_mode"] =
        hao_direct_config::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE;
    out["mx_quantized_denom"] =
        hao_direct_config::HAO_DIRECT_MX_QUANTIZED_DENOM;
    out["mx_denom_corr_wg"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_CORR_WG;
    out["mx_denom_load_width"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_LOAD_WIDTH;
    out["mx_denom_pipelined_sum"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_PIPELINED_SUM;
    out["mx_denom_pair_overlap"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_PAIR_OVERLAP;
    out["mx_local_denom_pipeline"] =
        hao_direct_config::HAO_DIRECT_MX_LOCAL_DENOM_PIPELINE;
    out["mx_defer_denom_finalize"] =
        hao_direct_config::HAO_DIRECT_MX_DEFER_DENOM_FINALIZE;
    out["mx_denom_factor_hoist"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_FACTOR_HOIST;
    out["mx_denom_int_scale_fusion"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_INT_SCALE_FUSION;
    out["mx_denom_sum_lanes"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SUM_LANES;
    out["mx_denom_words"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_WORDS;
    out["mx_denom_known_max"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_KNOWN_MAX;
    out["mx_denom_sample_layout"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SAMPLE_LAYOUT;
    out["mx_denom_decode_mode"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_DECODE_MODE;
    out["mx_denom_scale_smem"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SCALE_SMEM;
    out["mx_denom_payload_smem"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM;
    out["mx_denom_magic_i2f"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_MAGIC_I2F;
    out["mx_denom_early_smem"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_EARLY_SMEM;
    out["mx_denom_early_split"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_EARLY_SPLIT;
    out["mx_denom_split_prefetch"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SPLIT_PREFETCH;
    out["mx_global_anchor32"] =
        hao_direct_config::HAO_DIRECT_MX_GLOBAL_ANCHOR32;
    out["mx_global_anchor128"] =
        hao_direct_config::HAO_DIRECT_MX_GLOBAL_ANCHOR128;
    out["mx_global_anchor_bias"] =
        hao_direct_config::HAO_DIRECT_MX_GLOBAL_ANCHOR_BIAS;
    out["mx_global_anchor_margin_log2"] =
        TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2;
    out["mx_anchor_affine_hoist"] =
        hao_direct_config::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST;
    out["mx_fixed_anchor_log2"] =
        hao_direct_config::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2;
    out["mx_stored_scale_shift_log2"] =
        hao_direct_config::HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2;
    out["p_scale_publication_supported"] =
        hao_direct_config::HAO_DIRECT_MX_D4ALL_P_SCALE_PUBLICATION;
    out["mx_p_effective_max"] =
        hao_direct_config::HAO_DIRECT_MX_P_EFFECTIVE_MAX;
    out["debug_lse_stat"] =
        hao_direct_config::HAO_DIRECT_DEBUG_LSE_STAT;
    out["mx_denom_sample_stride"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SAMPLE_STRIDE;
    out["mx_max_sample_stride"] =
        hao_direct_config::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE;
    out["mx_max_sample_bias_x2"] =
        hao_direct_config::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2;
    out["mx_q1_max_sample_stride"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_MAX_SAMPLE_STRIDE;
    out["mx_pair_scale_reuse"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_REUSE;
    out["mx_pair_scale_stage_mask"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_STAGE_MASK;
    out["mx_q3_scale_reuse_stage_mask"] =
        hao_direct_config::HAO_DIRECT_MX_Q3_SCALE_REUSE_STAGE_MASK;
    out["mx_pair_scale_period"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_PERIOD;
    out["mx_pair_scale_delta"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_DELTA;
    out["mx_q1_pair_probe"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_PAIR_PROBE;
    out["mx_q1_pair_probe_bias"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_PAIR_PROBE_BIAS;
    out["mx_lagged_q1_scale"] =
        hao_direct_config::HAO_DIRECT_MX_LAGGED_Q1_SCALE;
    out["mx_q1_scale_offload"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD;
    out["mx_q1_self_max"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_SELF_MAX;
    out["mx_scan_fused_p"] =
        hao_direct_config::HAO_DIRECT_MX_SCAN_FUSED_P;
    out["mx_local_native_deferred_scale"] =
        hao_direct_config::HAO_DIRECT_MX_LOCAL_NATIVE_DEFERRED_SCALE;
    out["mx_bit_exp2_mask"] =
        hao_direct_config::HAO_DIRECT_MX_BIT_EXP2_MASK;
    out["mx_pwl_exp2"] =
        hao_direct_config::HAO_DIRECT_MX_PWL_EXP2;
    out["mx_pwl_exp2_mode"] =
        hao_direct_config::HAO_DIRECT_MX_PWL_EXP2_MODE;
    out["mx_mode23_native_density"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY;
    out["mx_mode23_native_quarter_mask"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_NATIVE_QUARTER_MASK;
    out["mx_mode23_native_density2_quarter_mask"] =
        hao_direct_config::
            HAO_DIRECT_MX_MODE23_NATIVE_DENSITY2_QUARTER_MASK;
    out["mx_mode23_native_density3_quarter_mask"] =
        hao_direct_config::
            HAO_DIRECT_MX_MODE23_NATIVE_DENSITY3_QUARTER_MASK;
    out["mx_mode23_native_density3_stage_mask"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY3_STAGE_MASK;
    out["mx_mode23_native_stage_mask"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_NATIVE_STAGE_MASK;
    out["mx_mode23_self_stage0_native"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_SELF_STAGE0_NATIVE;
    out["mx_mode23_early_native"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE;
    out["mx_mode23_early_native_stage_mask"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_STAGE_MASK;
    out["mx_mode23_early_native_quarter_mask"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_QUARTER_MASK;
    out["mx_mode23_early_native_lookahead"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_LOOKAHEAD;
    out["mx_mode23_early_native_order"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_ORDER;
    out["mx_pwl_four_sample_denom"] =
        hao_direct_config::HAO_DIRECT_MX_PWL_FOUR_SAMPLE_DENOM;
    out["mx_full_approx_denom"] =
        hao_direct_config::HAO_DIRECT_MX_FULL_APPROX_DENOM;
    out["mx_full_approx_denom_mode"] =
        hao_direct_config::HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE;
    out["mx_dual_lse_denom"] =
        hao_direct_config::HAO_DIRECT_MX_DUAL_LSE_DENOM;
    out["mx_stable_lse_logsum"] =
        hao_direct_config::HAO_DIRECT_MX_STABLE_LSE_LOGSUM;
    out["mx_alternate_lse_stat"] =
        hao_direct_config::HAO_DIRECT_MX_ALTERNATE_LSE_STAT;
    out["mx_affine_a"] =
        hao_direct_config::HAO_DIRECT_MX_AFFINE_A;
    out["mx_affine_b"] =
        hao_direct_config::HAO_DIRECT_MX_AFFINE_B;
    out["mx_cubic_a"] =
        hao_direct_config::HAO_DIRECT_MX_CUBIC_A;
    out["mx_cubic_b"] =
        hao_direct_config::HAO_DIRECT_MX_CUBIC_B;
    out["mx_cubic_c"] =
        hao_direct_config::HAO_DIRECT_MX_CUBIC_C;
    out["mx_cubic_d"] =
        hao_direct_config::HAO_DIRECT_MX_CUBIC_D;
    out["mx_stage0_affine_mask"] =
        hao_direct_config::HAO_DIRECT_MX_STAGE0_AFFINE_MASK;
    out["mx_stage1_affine_mask"] =
        hao_direct_config::HAO_DIRECT_MX_STAGE1_AFFINE_MASK;
    out["mx_shiftless_softmax"] =
        hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX;
    out["nv_shiftless_softmax"] =
        hao_direct_config::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX;
    out["nv_shiftless_scale_lift"] =
        hao_direct_config::HAO_DIRECT_NV_SHIFTLESS_SCALE_LIFT;
    out["nv_scale_satfinite"] =
        hao_direct_config::HAO_DIRECT_NV_SCALE_SATFINITE;
    out["nv_v_global_decode"] =
        hao_direct_config::HAO_DIRECT_NV_V_GLOBAL_DECODE;
    out["nv_p_global_log2"] =
        hao_direct_config::HAO_DIRECT_NV_P_GLOBAL_LOG2;
    out["nv_policy_id"] =
        hao_direct_config::HAO_DIRECT_NV_POLICY_ID;
    out["nv_pwl_exp2_mode"] =
        hao_direct_config::HAO_DIRECT_NV_PWL_EXP2_MODE;
    out["nv_stage0_affine_mask"] =
        hao_direct_config::HAO_DIRECT_NV_STAGE0_AFFINE_MASK;
    out["nv_stage1_affine_mask"] =
        hao_direct_config::HAO_DIRECT_NV_STAGE1_AFFINE_MASK;
    out["nv_fast_approx_mode"] =
        hao_direct_config::HAO_DIRECT_NV_FAST_APPROX_MODE;
    out["nv_affine_a"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_AFFINE_A);
    out["nv_affine_b"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_AFFINE_B);
    out["nv_cubic_a"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_CUBIC_A);
    out["nv_cubic_b"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_CUBIC_B);
    out["nv_cubic_c"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_CUBIC_C);
    out["nv_cubic_d"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_CUBIC_D);
    out["nv_refit_cubic_a"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_A);
    out["nv_refit_cubic_b"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_B);
    out["nv_refit_cubic_c"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_C);
    out["nv_refit_cubic_d"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_D);
    out["nv_hinge_m"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_HINGE_M);
    out["nv_hinge_b"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_HINGE_B);
    out["nv_hinge_d"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_HINGE_D);
    out["nv_hinge_t"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_HINGE_T);
    out["nv_native_pairs"] =
        hao_direct_config::HAO_DIRECT_NV_NATIVE_PAIRS;
    out["nv_native6_quarter_mask"] =
        hao_direct_config::HAO_DIRECT_NV_NATIVE6_QUARTER_MASK;
    out["nv_relu_pack"] =
        hao_direct_config::HAO_DIRECT_NV_RELU_PACK;
    out["nv_quantized_denom"] =
        hao_direct_config::HAO_DIRECT_NV_QUANTIZED_DENOM;
    out["nv_hybrid_denom"] =
        hao_direct_config::HAO_DIRECT_NV_HYBRID_DENOM;
    out["nv_sampled_denom"] =
        hao_direct_config::HAO_DIRECT_NV_SAMPLED_DENOM;
    out["mx_sampled_stabilizer"] =
        hao_direct_config::HAO_DIRECT_MX_SAMPLED_STABILIZER;
    out["mx_retain_q0"] =
        hao_direct_config::HAO_DIRECT_MX_RETAIN_Q0;
    out["mx_pair_load_scan"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_LOAD_SCAN;
    out["mx_half_prefetch_q0"] =
        hao_direct_config::HAO_DIRECT_MX_HALF_PREFETCH_Q0;
    out["mx_half_prefetch_q1"] =
        hao_direct_config::HAO_DIRECT_MX_HALF_PREFETCH_Q1;
    out["mx_half_prefetch_q3"] =
        hao_direct_config::HAO_DIRECT_MX_HALF_PREFETCH_Q3;
    out["mx_delayed_half_q2"] =
        hao_direct_config::HAO_DIRECT_MX_DELAYED_HALF_Q2;
    out["mx_delayed_early_q3"] =
        hao_direct_config::HAO_DIRECT_MX_DELAYED_EARLY_Q3;
    out["mx_causal_q3_lookahead"] =
        hao_direct_config::HAO_DIRECT_MX_CAUSAL_Q3_LOOKAHEAD;
    out["mx_causal_q3_progressive_reuse"] =
        hao_direct_config::HAO_DIRECT_MX_CAUSAL_Q3_PROGRESSIVE_REUSE;
    out["mx_early_q2_reduce"] =
        hao_direct_config::HAO_DIRECT_MX_EARLY_Q2_REDUCE;
    out["mx_stage1_early_q1"] =
        hao_direct_config::HAO_DIRECT_MX_STAGE1_EARLY_Q1;
    out["mx_q1_max_interleave"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_MAX_INTERLEAVE;
    out["mx_stage0_q1_max_interleave"] =
        hao_direct_config::HAO_DIRECT_MX_STAGE0_Q1_MAX_INTERLEAVE;
    out["mx_q2_max_interleave"] =
        hao_direct_config::HAO_DIRECT_MX_Q2_MAX_INTERLEAVE;
    out["mx_q3_max_interleave"] =
        hao_direct_config::HAO_DIRECT_MX_Q3_MAX_INTERLEAVE;
    out["mx_tree_max"] =
        hao_direct_config::HAO_DIRECT_MX_TREE_MAX;
    out["sm103_ldred"] =
        hao_direct_config::HAO_DIRECT_SM103_LDRED;
    out["sm103_ldred_mask"] =
        hao_direct_config::HAO_DIRECT_SM103_LDRED_MASK;
    out["sm103_ldred_stage0_mask"] =
        hao_direct_config::HAO_DIRECT_SM103_LDRED_STAGE0_MASK;
    out["sm103_ldred_stage1_mask"] =
        hao_direct_config::HAO_DIRECT_SM103_LDRED_STAGE1_MASK;
    out["mx_max3_reduce"] =
        hao_direct_config::HAO_DIRECT_MX_MAX3_REDUCE;
    out["mx_max3_wide_reduce"] =
        hao_direct_config::HAO_DIRECT_MX_MAX3_WIDE_REDUCE;
    out["nv_quarter_scale"] =
        hao_direct_config::HAO_DIRECT_NV_QUARTER_SCALE;
    out["nv_scale_encode_mode"] =
        hao_direct_config::HAO_DIRECT_NV_SCALE_ENCODE_MODE;
    out["nv_quarter_schedule_mode"] =
        hao_direct_config::HAO_DIRECT_NV_QUARTER_SCHEDULE_MODE;
    out["nv_denom_source"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_SOURCE;
    out["nv_denom_corr_wg"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_CORR_WG;
    out["nv_denom_smem_handoff"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF;
    out["nv_denom_words"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_WORDS;
    out["nv_denom_known_max"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_KNOWN_MAX;
    out["nv_denom_short_exact"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_SHORT_EXACT;
    out["nv_denom_exact_max_tiles"] =
        TK_HAO_DIRECT_FP4PV_NV_DENOM_EXACT_MAX_TILES;
    out["nv_global_anchor64"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR64;
    out["nv_global_anchor32"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR32;
    out["nv_global_anchor_samples"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR64
            ? 64
            : (hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR32
                   ? 32
                   : 0);
    out["nv_global_anchor_scale"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR_SCALE;
    out["nv_global_anchor_bias"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR_BIAS;
    out["nv_debug_lse_rowmax"] =
        hao_direct_config::HAO_DIRECT_NV_DEBUG_LSE_ROWMAX;
    out["nv_tail_q3_hook_word"] =
        hao_direct_config::HAO_DIRECT_NV_TAIL_Q3_HOOK_WORD;
    out["nv_q01_corr_wg"] =
        hao_direct_config::HAO_DIRECT_NV_Q01_CORR_WG;
    out["nv_tree_max"] =
        hao_direct_config::HAO_DIRECT_NV_TREE_MAX;
    out["nv_max3_reduce"] =
        hao_direct_config::HAO_DIRECT_NV_MAX3_REDUCE;
    out["nv_max_sample_stride"] =
        hao_direct_config::HAO_DIRECT_NVFP4_PV
            ? hao_direct_config::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE
            : 1;
    out["nv_max_sample_bias_x2"] =
        hao_direct_config::HAO_DIRECT_NVFP4_PV
            ? hao_direct_config::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2
            : 0;
    out["nv_q0_stabilizer"] =
        hao_direct_config::HAO_DIRECT_NV_Q0_STABILIZER;
    out["nv_preview_stabilizer"] =
        hao_direct_config::HAO_DIRECT_NV_PREVIEW_STABILIZER;
    out["nv_preview_width"] =
        hao_direct_config::HAO_DIRECT_NV_PREVIEW_WIDTH;
    out["nv_preview_bias_x8"] =
        hao_direct_config::HAO_DIRECT_NV_PREVIEW_BIAS_X8;
    out["nv_full_row_stabilizer"] =
        hao_direct_config::HAO_DIRECT_NV_FULL_ROW_STABILIZER;
    out["nv_direct_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_NV_EARLY_DIRECT_SCALE;
    out["nv_preload_v_tail_mode"] =
        hao_direct_config::HAO_DIRECT_NV_PRELOAD_V_TAIL_MODE;
    out["nv_issuer_v_scale_prefetch_mask"] =
        hao_direct_config::
            HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK;
    out["nv_register_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF;
    out["nv_reuse_v_scale"] =
        hao_direct_config::HAO_DIRECT_NV_REUSE_V_SCALE;
    out["nv_softmax_v_scale_prefetch"] =
        hao_direct_config::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH;
    out["mx_softmax_v_scale_prefetch_mask"] =
        hao_direct_config::
            HAO_DIRECT_MX_SOFTMAX_V_SCALE_PREFETCH_MASK;
    out["p_ilp_pipeline"] =
        hao_direct_config::HAO_DIRECT_P_ILP_PIPELINE;
    out["fixed_p_ceiling"] =
        hao_direct_config::HAO_DIRECT_FIXED_P_CEILING;
    out["scale_copy_ceiling_mask"] =
        hao_direct_config::HAO_DIRECT_SCALE_COPY_CEILING_MASK;
    out["score_pack_ceiling"] =
        hao_direct_config::HAO_DIRECT_SCORE_PACK_CEILING;
    out["rowmax_pack_ceiling"] =
        hao_direct_config::HAO_DIRECT_ROWMAX_PACK_CEILING;
    out["fake_denom_ceiling"] =
        hao_direct_config::HAO_DIRECT_FAKE_DENOM_CEILING;
    out["ex2_emu_mask"] =
        hao_direct_config::ONLINE_STAGE2_EX2_EMU_MASK;
    out["ex2_alu_degree"] =
        hao_direct_config::ONLINE_STAGE2_EX2_ALU_DEGREE;
    out["mx_ex2_q1_mask"] =
        hao_direct_config::HAO_DIRECT_MX_EX2_Q1_MASK;
    out["mx_ex2_q2_mask"] =
        hao_direct_config::HAO_DIRECT_MX_EX2_Q2_MASK;
    out["mx_ex2_q3_mask"] =
        hao_direct_config::HAO_DIRECT_MX_EX2_Q3_MASK;
    out["task_order"] = hao_direct_config::TASK_ORDER;
    out["wait_hint_cycles"] = TK_HAO_DIRECT_WAIT_HINT;
    out["qk_format"] =
        hao_direct_config::HAO_DIRECT_NVFP4_QK
            ? "nvfp4_e4m3_block16"
            : "mxfp4_e8m0_block32";
    out["pv_format"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV
            ? "sparse_nvfp4_e4m3_block32"
            : (hao_direct_config::HAO_DIRECT_NVFP4_PV
                   ? "nvfp4_e4m3_block16"
                   : "mxfp4_e8m0_block32");
    return out;
}

void dispatch_hao_direct_fp4pv_impl(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    float V_sg,
    at::Tensor O,
    at::Tensor LSE,
    int p_quant_mode,
    bool persistent_launch,
    bool store_lse,
    const at::Tensor *P_scales) {
    using C = hao_direct_config;
    using G = globals_fp4pv_hao_direct<C>;
    using Layout = hao_direct_pipeline_smem_layout<C>;
#if !TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
    const char *route =
        std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    TORCH_CHECK(
        route != nullptr &&
            std::strcmp(route, TK_HAO_DIRECT_ROUTE_NAME) == 0,
        "TK HAO direct candidate route mismatch");
#endif
    TORCH_CHECK(
        persistent_launch,
        "HAO direct port requires persistent_launch=True");
    TORCH_CHECK(
        p_quant_mode == 0,
        "HAO direct full-FP4 port has one fixed P quantizer");
    TORCH_CHECK(
        !C::HAO_DIRECT_NV_V_GLOBAL_DECODE ||
            (std::isfinite(V_sg) && V_sg > 0.0f),
        "V_sg must be finite and positive");
    CHECK_INPUT(Q);
    CHECK_INPUT(Q_sc);
    CHECK_INPUT(Q_sg);
    CHECK_INPUT(K);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(K_sg);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(O);
    CHECK_INPUT(LSE);
    if (P_scales != nullptr) {
        TORCH_CHECK(
            C::HAO_DIRECT_MX_D4ALL_P_SCALE_PUBLICATION,
            "probability-scale publication requires the causal D64 "
            "MXFP4 d4all forward topology");
        const at::Tensor &p_scales = *P_scales;
        CHECK_INPUT(p_scales);
        TORCH_CHECK(
            p_scales.dtype() == at::ScalarType::Int,
            "P_scales must be int32");
        TORCH_CHECK(
            p_scales.dim() == 4 &&
                p_scales.size(0) == C::BATCH &&
                p_scales.size(1) == C::HEADS &&
                p_scales.size(2) == C::SEQ_LEN / 128 &&
                p_scales.size(3) == C::SEQ_LEN,
            "P_scales must have shape [B,H,S/128,S]");
    }
    TORCH_CHECK(
        Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            V_sc.dtype() == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(
        O.dtype() == at::ScalarType::BFloat16 &&
            LSE.dtype() == at::ScalarType::Float);
    TORCH_CHECK(
        Q.dim() == 4 &&
            Q.size(0) == C::BATCH &&
            Q.size(1) == C::HEADS &&
            Q.size(2) == C::SEQ_LEN &&
            Q.size(3) == C::Dqk / 2);
    TORCH_CHECK(
        K.dim() == 4 &&
            K.size(0) == C::BATCH &&
            K.size(1) == C::KV_HEADS &&
            K.size(2) == C::SEQ_LEN &&
            K.size(3) == C::Dqk / 2);
    TORCH_CHECK(
        V_fp4.dim() == 4 &&
            V_fp4.size(0) == C::BATCH &&
            V_fp4.size(1) == C::KV_HEADS &&
            V_fp4.size(2) == C::Dvo &&
            V_fp4.size(3) ==
                (C::HAO_DIRECT_SM103_N256
                     ? (C::HAO_DIRECT_SM103_N256_COMPACT_V
                            ? C::SEQ_LEN / 2
                            : C::N_TILES *
                                  C::HAO_DIRECT_SM103_WIDE_SEGMENTS * 64)
                     : C::Nb == 96
                     ? C::N_TILES * 64
                     : C::SEQ_LEN / 2));
    TORCH_CHECK(
        V_sc.dim() == 4 &&
            V_sc.size(0) == C::BATCH &&
            V_sc.size(1) ==
                C::N_TILES *
                    C::HAO_DIRECT_SM103_WIDE_SEGMENTS &&
            V_sc.size(2) ==
                C::KV_HEADS * C::PV_SCALE_ROWS &&
            V_sc.size(3) == 512);
    TORCH_CHECK(
        O.dim() == 4 &&
            O.size(0) == C::BATCH &&
            O.size(1) == C::SEQ_LEN &&
            O.size(2) == C::HEADS &&
            O.size(3) == C::Dvo);
    TORCH_CHECK(
        LSE.dim() == 4 &&
            LSE.size(0) == C::BATCH &&
            LSE.size(1) == C::HEADS &&
            LSE.size(2) == 1 &&
            LSE.size(3) == C::SEQ_LEN);

#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
    // The public helper checks these scale/scalar contracts while constructing
    // their GLs. Repeat the checks before a cache lookup so a same-address view
    // with different metadata cannot bypass validation on a cache hit.
    const auto check_scale = [](
        const at::Tensor &tensor,
        int batch,
        int depth,
        int rows,
        const char *name) {
        CHECK_CUDA(tensor);
        CHECK_CONTIGUOUS(tensor);
        TORCH_CHECK(
            tensor.dtype() == at::ScalarType::Float8_e4m3fn,
            name,
            " must be float8_e4m3fn");
        TORCH_CHECK(
            tensor.dim() == 4 &&
                tensor.size(0) == batch &&
                tensor.size(1) == depth &&
                tensor.size(2) == rows &&
                tensor.size(3) == 512,
            name,
            " must have shape (batch, depth_tiles, rows, 512)");
    };
    const auto check_scalar = [](
        const at::Tensor &tensor,
        int batch,
        int heads,
        const char *name) {
        CHECK_CUDA(tensor);
        CHECK_CONTIGUOUS(tensor);
        TORCH_CHECK(
            tensor.dtype() == at::ScalarType::Float,
            name,
            " must be float32");
        TORCH_CHECK(
            (tensor.dim() == 2 &&
             tensor.size(0) == batch &&
             tensor.size(1) == heads) ||
                (tensor.dim() == 4 &&
                 tensor.size(0) == batch &&
                 tensor.size(1) == heads &&
                 tensor.size(2) == 1 &&
                 tensor.size(3) == 1),
            name,
            " must have shape (batch, heads) or (batch, heads, 1, 1)");
    };
    check_scale(
        Q_sc,
        C::BATCH,
        C::QUERY_TILES,
        C::HEADS * C::QK_SCALE_ROWS,
        "Q_sc");
    check_scalar(Q_sg, C::BATCH, C::HEADS, "Q_sg");
    check_scale(
        K_sc,
        C::BATCH,
        C::N_TILES * 2,
        C::KV_HEADS * C::QK_SCALE_ROWS,
        "K_sc");
    check_scalar(K_sg, C::BATCH, C::KV_HEADS, "K_sg");
#endif

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int device = static_cast<int>(stream.device_index());
    const auto build_globals = [&]() -> G {
        return G{
            hao_direct_fp4pv_cached_gl<0, typename G::q_gl>(
                device,
                Q.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::q_gl>(Q); }),
            hao_direct_fp4pv_cached_gl<1, typename G::q_sc_gl>(
                device,
                Q_sc.data_ptr(),
                [&] { return tensor_to_fp4_scale_gl<
                    typename G::q_sc_gl>(
                        Q_sc,
                        C::BATCH,
                        C::QUERY_TILES,
                        C::HEADS * C::QK_SCALE_ROWS,
                        "Q_sc"); }),
            hao_direct_fp4pv_cached_gl<2, typename G::q_sg_gl>(
                device,
                Q_sg.data_ptr(),
                [&] { return tensor_to_bh_scalar_gl<
                    typename G::q_sg_gl>(
                        Q_sg, C::BATCH, C::HEADS, "Q_sg"); }),
            hao_direct_fp4pv_cached_gl<3, typename G::k_gl>(
                device,
                K.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::k_gl>(K); }),
            hao_direct_fp4pv_cached_gl<4, typename G::k_sc_gl>(
                device,
                K_sc.data_ptr(),
                [&] { return tensor_to_fp4_scale_gl<
                    typename G::k_sc_gl>(
                        K_sc,
                        C::BATCH,
                        C::N_TILES * 2,
                        C::KV_HEADS * C::QK_SCALE_ROWS,
                        "K_sc"); }),
            hao_direct_fp4pv_cached_gl<5, typename G::k_sg_gl>(
                device,
                K_sg.data_ptr(),
                [&] { return tensor_to_bh_scalar_gl<
                    typename G::k_sg_gl>(
                        K_sg, C::BATCH, C::KV_HEADS, "K_sg"); }),
            hao_direct_fp4pv_cached_gl<6, typename G::v_fp4_gl>(
                device,
                V_fp4.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::v_fp4_gl>(V_fp4); }),
            hao_direct_fp4pv_cached_gl<7, typename G::v_sc_gl>(
                device,
                V_sc.data_ptr(),
                [&] { return tensor_to_fp4_scale_gl<
                    typename G::v_sc_gl>(
                        V_sc,
                        C::BATCH,
                        C::N_TILES *
                            C::HAO_DIRECT_SM103_WIDE_SEGMENTS,
                        C::KV_HEADS * C::PV_SCALE_ROWS,
                        "V_sc"); }),
            V_sg,
            hao_direct_fp4pv_cached_gl<8, typename G::o_gl>(
                device,
                O.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::o_gl>(O); }),
            hao_direct_fp4pv_cached_gl<9, typename G::lse_gl>(
                device,
                LSE.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::lse_gl>(LSE); }),
            0,
            nullptr};
    };

    // Each fixed-geometry GL descriptor is cached independently. Transient
    // O/LSE allocations therefore cannot invalidate reusable Q/K/V/scale GLs.
    G g = build_globals();
    // These fields can change without changing a TMA descriptor. Always reset
    // them after a cache hit so optional/debug launches cannot leak state.
    g.v_global_decode = V_sg;
    g.disable_lse_store = store_lse ? 0 : 1;
    g.lifecycle_debug = nullptr;
    g.group2_trap_stage = -1;
    g.p_scale_output = nullptr;
    if (P_scales != nullptr) {
        const at::Tensor &p_scales = *P_scales;
        g.p_scale_output = reinterpret_cast<uint32_t *>(
            p_scales.data_ptr<int32_t>());
    }
    if constexpr (C::HAO_DIRECT_SM103_GROUP2) {
        const char *trap_stage =
            std::getenv("TK_HAO_DIRECT_GROUP2_TRAP_STAGE");
        g.group2_trap_stage =
            trap_stage == nullptr ? -1 : std::atoi(trap_stage);
    }
#if TK_HAO_DIRECT_N96_LIFECYCLE_DEBUG
    init_hao_direct_n96_lifecycle();
    g.lifecycle_debug = hao_direct_n96_lifecycle_device;
    if constexpr (C::HAO_DIRECT_SM103_GROUP2) {
        hao_direct_n96_lifecycle_host[58] = 1234;
    }
#endif

    const int physical_grid = [&] {
        if constexpr (C::HAO_DIRECT_SM103_GROUP2) {
            const int cluster_cap =
                C::PHYSICAL_GRID_CAP / C::CLUSTER_SIZE;
            const int clusters =
                C::LOGICAL_JOBS < cluster_cap
                    ? C::LOGICAL_JOBS
                    : cluster_cap;
            return clusters * C::CLUSTER_SIZE;
        }
        return C::LOGICAL_JOBS < C::PHYSICAL_GRID_CAP
            ? C::LOGICAL_JOBS
            : C::PHYSICAL_GRID_CAP;
    }();
    constexpr int DYNAMIC_SMEM = Layout::bytes();
    const auto initialize_host_invariants = [&] {
        {
            std::lock_guard<std::mutex> lock(
                hao_direct_topology_mutex);
            hao_direct_last_topology = {
                1,
                physical_grid,
                C::LOGICAL_JOBS,
                C::NUM_THREADS,
                C::KV_STAGES,
                C::TMEM_COLUMNS,
                DYNAMIC_SMEM};
        }
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_hao_direct_fp4pv<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            DYNAMIC_SMEM));
    };
#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
    if (hao_direct_fp4pv_initialized_device != device) {
        initialize_host_invariants();
        hao_direct_fp4pv_initialized_device = device;
    }
#else
    initialize_host_invariants();
#endif
    if constexpr (C::HAO_DIRECT_SM103_GROUP2) {
        TORCH_CHECK(
            physical_grid % C::CLUSTER_SIZE == 0,
            "SM103 group2 launch requires complete CTA pairs");
        LaunchConfig<true, false> launch_config(
            dim3(physical_grid),
            dim3(C::NUM_THREADS),
            DYNAMIC_SMEM,
            stream,
            C::CLUSTER_SIZE);
#if TK_HAO_DIRECT_N96_LIFECYCLE_DEBUG
        int max_active_clusters = -1;
        CUDACHECK(cudaOccupancyMaxActiveClusters(
            &max_active_clusters,
            kernel_fp4pv_hao_direct_fp4pv<C>,
            &launch_config.config));
        std::cerr << "hao group2 max active clusters: "
                  << max_active_clusters << std::endl;
#endif
        CUDACHECK(cudaLaunchKernelEx(
            launch_config,
            kernel_fp4pv_hao_direct_fp4pv<C>,
            g));
    } else {
        kernel_fp4pv_hao_direct_fp4pv<C>
            <<<dim3(physical_grid),
               dim3(C::NUM_THREADS),
               DYNAMIC_SMEM,
               stream>>>(g);
    }
    CUDACHECK(cudaGetLastError());
}

void dispatch_hao_direct_fp4pv(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    int p_quant_mode,
    bool persistent_launch,
    bool store_lse) {
    dispatch_hao_direct_fp4pv_impl(
        Q, Q_sc, Q_sg,
        K, K_sc, K_sg,
        V_fp4, V_sc, 1.0f,
        O, LSE,
        p_quant_mode,
        persistent_launch,
        store_lse,
        nullptr);
}

void dispatch_hao_direct_fp4pv_with_p_scales(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    at::Tensor P_scales,
    int p_quant_mode,
    bool persistent_launch,
    bool store_lse) {
    dispatch_hao_direct_fp4pv_impl(
        Q, Q_sc, Q_sg,
        K, K_sc, K_sg,
        V_fp4, V_sc, 1.0f,
        O, LSE,
        p_quant_mode,
        persistent_launch,
        store_lse,
        &P_scales);
}

void dispatch_hao_direct_fp4pv_vscale(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    float V_sg,
    at::Tensor O,
    at::Tensor LSE,
    int p_quant_mode,
    bool persistent_launch,
    bool store_lse) {
    dispatch_hao_direct_fp4pv_impl(
        Q, Q_sc, Q_sg,
        K, K_sc, K_sg,
        V_fp4, V_sc, V_sg,
        O, LSE,
        p_quant_mode,
        persistent_launch,
        store_lse,
        nullptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_hao_direct_fp4pv",
        &dispatch_hao_direct_fp4pv);
    m.def(
        "forward_hao_direct_fp4pv_vscale",
        &dispatch_hao_direct_fp4pv_vscale);
    m.def(
        "forward_hao_direct_fp4pv_with_p_scales",
        &dispatch_hao_direct_fp4pv_with_p_scales);
    m.def(
        "read_hao_direct_topology",
        &read_hao_direct_topology);
    m.def(
        "reset_hao_direct_n96_lifecycle",
        &reset_hao_direct_n96_lifecycle);
    m.def(
        "read_hao_direct_n96_lifecycle",
        &read_hao_direct_n96_lifecycle);
}

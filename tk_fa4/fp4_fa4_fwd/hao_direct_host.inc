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

constexpr std::size_t HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_CAPACITY = 256;
constexpr std::size_t HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_WAYS = 4;
constexpr std::size_t HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_SETS =
    HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_CAPACITY /
    HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_WAYS;
constexpr std::size_t HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_GL_SLOTS = 9;
constexpr std::uint64_t HAO_DIRECT_FP8PV_DESCRIPTOR_DEVICE_MIX =
    0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t HAO_DIRECT_FP8PV_DESCRIPTOR_HASH_MIX_1 =
    0xbf58476d1ce4e5b9ULL;
constexpr std::uint64_t HAO_DIRECT_FP8PV_DESCRIPTOR_HASH_MIX_2 =
    0x94d049bb133111ebULL;
static_assert(
    HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_CAPACITY %
            HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_WAYS ==
        0);
static_assert(HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_SETS == 64);

struct hao_direct_fp8pv_descriptor_cache_counters {
    unsigned long long hits = 0;
    unsigned long long misses = 0;
    unsigned long long evictions = 0;
    unsigned long long clears = 0;
};

// Counters deliberately have the same scope as the cache. Python launches and
// topology reads normally use the same host thread, while avoiding atomics on
// every attention layer keeps the fixed-route fast path lightweight.
static thread_local hao_direct_fp8pv_descriptor_cache_counters
    hao_direct_fp8pv_cache_counters;

template <std::size_t GlSlot, typename Descriptor, typename Builder>
inline Descriptor hao_direct_fp8pv_cached_gl(
    int device,
    const void *data_pointer,
    Builder build_descriptor
) {
#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
    static_assert(
        GlSlot < HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_GL_SLOTS);
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
            HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_WAYS>
            entries;
        std::size_t next_victim = 0;
    };
    struct descriptor_cache_state {
        std::array<
            descriptor_cache_set,
            HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_SETS>
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
        (device_key * HAO_DIRECT_FP8PV_DESCRIPTOR_DEVICE_MIX);
    hash_key = (hash_key ^ (hash_key >> 30)) *
        HAO_DIRECT_FP8PV_DESCRIPTOR_HASH_MIX_1;
    hash_key = (hash_key ^ (hash_key >> 27)) *
        HAO_DIRECT_FP8PV_DESCRIPTOR_HASH_MIX_2;
    hash_key ^= hash_key >> 31;
    const std::size_t set_index = static_cast<std::size_t>(
        hash_key & (HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_SETS - 1));
    static thread_local descriptor_cache_state descriptor_cache;
    auto &cache_set = descriptor_cache.sets[set_index];
    for (const auto &slot : cache_set.entries) {
        if (slot.has_value() && slot->key == key) {
            ++hao_direct_fp8pv_cache_counters.hits;
            return slot->descriptor;
        }
    }
    ++hao_direct_fp8pv_cache_counters.misses;
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
        HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_WAYS;
    victim.emplace(descriptor_cache_entry{key, descriptor});
    ++hao_direct_fp8pv_cache_counters.evictions;
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
    hao_direct_fp8pv_initialized_device = -1;
#endif

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
            ? HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_CAPACITY
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
            ? HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_SETS
            : 0;
    out["tma_descriptor_cache_ways"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_WAYS
            : 0;
    out["tma_descriptor_cache_capacity_scope"] =
        "per_compile_time_gl_slot";
    out["tma_descriptor_cache_gl_slots"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_GL_SLOTS
            : 0;
    out["tma_descriptor_cache_total_entry_ceiling"] =
        TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
            ? HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_CAPACITY *
                HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_GL_SLOTS
            : 0;
    out["tma_descriptor_cache_key"] =
        "cuda_device_data_ptr_and_compile_time_gl_slot";
    out["tma_descriptor_cache_owns_tensors"] = false;
    out["tma_descriptor_cache_counter_scope"] =
        "calling_host_thread";
    out["tma_descriptor_cache_hits"] =
        hao_direct_fp8pv_cache_counters.hits;
    out["tma_descriptor_cache_misses"] =
        hao_direct_fp8pv_cache_counters.misses;
    out["tma_descriptor_cache_evictions"] =
        hao_direct_fp8pv_cache_counters.evictions;
    out["tma_descriptor_cache_clears"] =
        hao_direct_fp8pv_cache_counters.clears;
    out["batch"] = hao_direct_config::BATCH;
    out["heads"] = hao_direct_config::HEADS;
    // This FP8-PV kernel always uses the original contiguous causal score
    // mapping.  Export that negative capability explicitly so authenticated
    // callers can fail closed instead of treating a missing field as false.
    out["causal_interleaved_kv"] = false;
    out["seqlen"] = hao_direct_config::N_TILES * 128;
    out["dqk"] = hao_direct_config::Dqk;
    out["dvo"] = hao_direct_config::Dvo;
    out["logical_jobs"] = s.logical_jobs;
    out["physical_grid_ctas"] = s.physical_grid;
    out["threads_per_cta"] = s.threads;
    out["query_stages"] = hao_direct_config::QUERY_STAGES;
    out["kv_stages"] = s.kv_stages;
    out["tmem_columns"] = s.tmem_columns;
    out["dynamic_smem_bytes"] = s.dynamic_smem;
    out["persistent"] = true;
    out["p_first_chunks"] =
        hao_direct_config::P_FIRST_CHUNKS;
    out["ex2_emu_mask"] =
        hao_direct_config::ONLINE_STAGE2_EX2_EMU_MASK;
    out["retain_q2_mode"] = TK_HAO_DIRECT_RETAIN_Q2;
    out["retain_q0"] = TK_HAO_DIRECT_RETAIN_Q0;
    out["retain_q1"] = TK_HAO_DIRECT_RETAIN_Q1;
    out["retain_q3"] = TK_HAO_DIRECT_RETAIN_Q3;
    out["wait_hint_cycles"] = TK_HAO_DIRECT_WAIT_HINT;
    out["fixed_p_ceiling"] =
        hao_direct_config::HAO_DIRECT_FIXED_P_CEILING;
    out["score_pack_ceiling"] =
        hao_direct_config::HAO_DIRECT_SCORE_PACK_CEILING;
    out["p_ilp_pipeline"] =
        hao_direct_config::HAO_DIRECT_P_ILP_PIPELINE;
    out["early_q3"] =
        hao_direct_config::HAO_DIRECT_EARLY_Q3;
    out["two_chunk_pipeline"] =
        hao_direct_config::HAO_DIRECT_TWO_CHUNK_PIPELINE;
    out["shiftless_fp8_mode"] =
        hao_direct_config::HAO_DIRECT_SHIFTLESS_MODE;
    out["softmax_regs"] = hao_direct_config::SOFTMAX_REGS;
    out["correction_regs"] =
        hao_direct_config::CORRECTION_REGS;
    out["producer_regs"] = hao_direct_config::PRODUCER_REGS;
    out["qk_format"] =
        hao_direct_config::HAO_DIRECT_NVFP4_QK
            ? "nvfp4_e4m3_block16"
            : "mxfp4_e8m0_block32";
    out["pv_format"] = "e4m3_fp8";
    return out;
}

void dispatch_hao_direct_fp8pv(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp8,
    at::Tensor O,
    at::Tensor LSE,
    int p_quant_mode,
    bool persistent_launch,
    bool store_lse) {
    using C = hao_direct_config;
    using G = globals_fp8pv_hao_direct<C>;
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
        "FP8-PV direct port does not use FP4 P quant mode");
    CHECK_INPUT(Q);
    CHECK_INPUT(Q_sc);
    CHECK_INPUT(Q_sg);
    CHECK_INPUT(K);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(K_sg);
    CHECK_INPUT(V_fp8);
    CHECK_INPUT(O);
    CHECK_INPUT(LSE);
    TORCH_CHECK(
        Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            V_fp8.dtype() == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(
        O.dtype() == at::ScalarType::BFloat16 &&
            LSE.dtype() == at::ScalarType::Float);
    TORCH_CHECK(
        Q.dim() == 4 &&
            Q.size(0) == C::BATCH &&
            Q.size(1) == C::HEADS &&
            Q.size(2) == C::N_TILES * 128 &&
            Q.size(3) == C::Dqk / 2);
    TORCH_CHECK(K.sizes() == Q.sizes());
    TORCH_CHECK(
        V_fp8.dim() == 4 &&
            V_fp8.size(0) == C::BATCH &&
            V_fp8.size(1) == C::HEADS &&
            V_fp8.size(2) == C::Dvo &&
            V_fp8.size(3) == C::N_TILES * 128);
    TORCH_CHECK(
        O.dim() == 4 &&
            O.size(0) == C::BATCH &&
            O.size(1) == C::N_TILES * 128 &&
            O.size(2) == C::HEADS &&
            O.size(3) == C::Dvo);
    TORCH_CHECK(
        LSE.dim() == 4 &&
            LSE.size(0) == C::BATCH &&
            LSE.size(1) == C::HEADS &&
            LSE.size(2) == 1 &&
            LSE.size(3) == C::N_TILES * 128);

#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
    // The public helper validates scale/scalar metadata while constructing
    // their GLs. Repeat those checks before a cache lookup so a same-address
    // view with different metadata cannot bypass validation on a cache hit.
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
    const int q_heads = static_cast<int>(Q.size(1));
    const int kv_heads = static_cast<int>(K.size(1));
    check_scale(
        Q_sc,
        C::BATCH,
        C::N_TILES,
        q_heads * C::QK_SCALE_ROWS,
        "Q_sc");
    check_scalar(Q_sg, C::BATCH, q_heads, "Q_sg");
    check_scale(
        K_sc,
        C::BATCH,
        C::N_TILES * 2,
        kv_heads * C::QK_SCALE_ROWS,
        "K_sc");
    check_scalar(K_sg, C::BATCH, kv_heads, "K_sg");
#endif

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int device = static_cast<int>(stream.device_index());
    const auto build_globals = [&]() -> G {
        return G{
            hao_direct_fp8pv_cached_gl<0, typename G::q_gl>(
                device,
                Q.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::q_gl>(Q); }),
            hao_direct_fp8pv_cached_gl<1, typename G::q_sc_gl>(
                device,
                Q_sc.data_ptr(),
                [&] { return tensor_to_fp4_scale_gl<
                    typename G::q_sc_gl>(
                        Q_sc,
                        C::BATCH,
                        C::N_TILES,
                        C::HEADS * C::QK_SCALE_ROWS,
                        "Q_sc"); }),
            hao_direct_fp8pv_cached_gl<2, typename G::q_sg_gl>(
                device,
                Q_sg.data_ptr(),
                [&] { return tensor_to_bh_scalar_gl<
                    typename G::q_sg_gl>(
                        Q_sg, C::BATCH, C::HEADS, "Q_sg"); }),
            hao_direct_fp8pv_cached_gl<3, typename G::k_gl>(
                device,
                K.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::k_gl>(K); }),
            hao_direct_fp8pv_cached_gl<4, typename G::k_sc_gl>(
                device,
                K_sc.data_ptr(),
                [&] { return tensor_to_fp4_scale_gl<
                    typename G::k_sc_gl>(
                        K_sc,
                        C::BATCH,
                        C::N_TILES * 2,
                        C::HEADS * C::QK_SCALE_ROWS,
                        "K_sc"); }),
            hao_direct_fp8pv_cached_gl<5, typename G::k_sg_gl>(
                device,
                K_sg.data_ptr(),
                [&] { return tensor_to_bh_scalar_gl<
                    typename G::k_sg_gl>(
                        K_sg, C::BATCH, C::HEADS, "K_sg"); }),
            hao_direct_fp8pv_cached_gl<6, typename G::v_fp8_gl>(
                device,
                V_fp8.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::v_fp8_gl>(V_fp8); }),
            hao_direct_fp8pv_cached_gl<7, typename G::o_gl>(
                device,
                O.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::o_gl>(O); }),
            hao_direct_fp8pv_cached_gl<8, typename G::lse_gl>(
                device,
                LSE.data_ptr(),
                [&] { return kittens::py::tensor_to_gl<
                    typename G::lse_gl>(LSE); }),
            0};
    };

    // Each fixed-geometry GL descriptor is cached independently. Transient
    // O/LSE allocations therefore cannot invalidate reusable Q/K/V/scale GLs.
    G g = build_globals();
    // This launch scalar can change without changing a TMA descriptor.
    // Always overwrite it after a cache hit.
    g.disable_lse_store = store_lse ? 0 : 1;

    const int physical_grid =
        C::LOGICAL_JOBS < C::PHYSICAL_GRID_CAP
            ? C::LOGICAL_JOBS
            : C::PHYSICAL_GRID_CAP;
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
            kernel_fp4pv_hao_direct_fp8pv<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            DYNAMIC_SMEM));
    };
#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH
    if (hao_direct_fp8pv_initialized_device != device) {
        initialize_host_invariants();
        hao_direct_fp8pv_initialized_device = device;
    }
#else
    initialize_host_invariants();
#endif
    kernel_fp4pv_hao_direct_fp8pv<C>
        <<<dim3(physical_grid),
           dim3(C::NUM_THREADS),
           DYNAMIC_SMEM,
           stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_hao_direct_fp8pv",
        &dispatch_hao_direct_fp8pv);
    m.def(
        "forward_hao_direct_nvfp4_fp8pv",
        &dispatch_hao_direct_fp8pv);
    m.def(
        "read_hao_direct_topology",
        &read_hao_direct_topology);
}

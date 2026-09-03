#pragma once

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

pybind11::dict read_hao_direct_topology() {
    std::lock_guard<std::mutex> lock(
        hao_direct_topology_mutex);
    const auto &s = hao_direct_last_topology;
    pybind11::dict out;
    out["schema"] = "tk_hao_direct_pipeline_v1";
    out["valid"] = s.valid;
    out["route"] = TK_HAO_DIRECT_ROUTE_NAME;
    out["batch"] = hao_direct_config::BATCH;
    out["heads"] = hao_direct_config::HEADS;
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
    const char *route =
        std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    TORCH_CHECK(
        route != nullptr &&
            std::strcmp(route, TK_HAO_DIRECT_ROUTE_NAME) == 0,
        "TK HAO direct candidate route mismatch");
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
            Q.size(3) == 64);
    TORCH_CHECK(K.sizes() == Q.sizes());
    TORCH_CHECK(
        V_fp8.dim() == 4 &&
            V_fp8.size(0) == C::BATCH &&
            V_fp8.size(1) == C::HEADS &&
            V_fp8.size(2) == 128 &&
            V_fp8.size(3) == C::N_TILES * 128);
    TORCH_CHECK(
        O.dim() == 4 &&
            O.size(0) == C::BATCH &&
            O.size(1) == C::N_TILES * 128 &&
            O.size(2) == C::HEADS &&
            O.size(3) == 128);
    TORCH_CHECK(
        LSE.dim() == 4 &&
            LSE.size(0) == C::BATCH &&
            LSE.size(1) == C::HEADS &&
            LSE.size(2) == 1 &&
            LSE.size(3) == C::N_TILES * 128);

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(
            Q_sc,
            C::BATCH,
            C::N_TILES,
            C::HEADS * C::QK_SCALE_ROWS,
            "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(
            Q_sg, C::BATCH, C::HEADS, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(
            K_sc,
            C::BATCH,
            C::N_TILES * 2,
            C::HEADS * C::QK_SCALE_ROWS,
            "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(
            K_sg, C::BATCH, C::HEADS, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_fp8_gl>(
            V_fp8),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        0};
    g.disable_lse_store = store_lse ? 0 : 1;

    const int physical_grid =
        C::LOGICAL_JOBS < C::PHYSICAL_GRID_CAP
            ? C::LOGICAL_JOBS
            : C::PHYSICAL_GRID_CAP;
    constexpr int DYNAMIC_SMEM = Layout::bytes();
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
    kernel_fp4pv_hao_direct_fp8pv<C>
        <<<dim3(physical_grid),
           dim3(C::NUM_THREADS),
           DYNAMIC_SMEM,
           at::cuda::getCurrentCUDAStream()>>>(g);
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

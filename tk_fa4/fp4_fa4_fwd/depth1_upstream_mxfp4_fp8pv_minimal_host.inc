#pragma once

struct upstream_mxfp4_pipeline_topology_state {
    long long valid = 0;
    long long grid = 0;
    long long threads = 0;
    long long query_stages = 0;
    long long kv_stages = 0;
    long long tmem_columns = 0;
    long long dynamic_smem = 0;
    long long grid1 = 0;
    long long logical_jobs = 0;
    long long logical_q_tiles = 0;
};

static std::mutex upstream_mxfp4_pipeline_topology_mutex;
static upstream_mxfp4_pipeline_topology_state
    upstream_mxfp4_pipeline_last_topology;

pybind11::dict read_upstream_mxfp4_pipeline_topology() {
    std::lock_guard<std::mutex> lock(upstream_mxfp4_pipeline_topology_mutex);
    const auto &s = upstream_mxfp4_pipeline_last_topology;
    pybind11::dict out;
    out["schema"] = "tk_hao_native_b1s32768h24d128_topology_v1";
    out["valid"] = s.valid;
    out["route"] = TK_HAO_ROUTE_NAME;
    out["batch"] = tk_hao_native_port_config::BATCH;
    out["heads"] = tk_hao_native_port_config::HEADS;
    out["seqlen"] = tk_hao_native_port_config::N_TILES * 128;
    out["dqk"] = 128;
    out["dvo"] = 128;
    out["logical_jobs"] = s.logical_jobs;
    out["logical_q_tiles"] = s.logical_q_tiles;
    out["physical_grid_ctas"] = s.grid;
    out["threads_per_cta"] = s.threads;
    out["query_stages"] = s.query_stages;
    out["kv_stages"] = s.kv_stages;
    out["tmem_columns"] = s.tmem_columns;
    out["dynamic_smem_bytes"] = s.dynamic_smem;
    out["grid1"] = s.grid1;
    return out;
}

void dispatch_upstream_mxfp4_pipeline(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V_fp8,
    at::Tensor O, at::Tensor LSE,
    int p_quant_mode, bool persistent_launch
) {
    using C = tk_hao_native_port_config;
    using G = globals_fp4pv_upstream_fp8pv<C>;
    using Layout = upstream_mxfp4_pipeline_smem_layout<C>;
    const char *route = std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    const bool full = route != nullptr && std::strcmp(route, TK_HAO_ROUTE_NAME) == 0;
    TORCH_CHECK(full, "TK HAO native-port candidate-only route mismatch");
    TORCH_CHECK(!persistent_launch,
        "full-grid bring-up must not be mislabeled persistent");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2);
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V_fp8); CHECK_INPUT(O); CHECK_INPUT(LSE);
    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V_fp8.dtype() == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 &&
                LSE.dtype() == at::ScalarType::Float);
    TORCH_CHECK(Q.dim() == 4 && Q.size(0) == C::BATCH && Q.size(1) == C::HEADS &&
                Q.size(2) == C::N_TILES * 128 && Q.size(3) == 64);
    TORCH_CHECK(K.sizes() == Q.sizes());
    TORCH_CHECK(O.dim() == 4 && O.size(0) == C::BATCH && O.size(1) == C::N_TILES*128 &&
                O.size(2) == C::HEADS && O.size(3) == 128);
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == C::BATCH && LSE.size(1) == C::HEADS &&
                LSE.size(2) == 1 && LSE.size(3) == C::N_TILES*128);
    TORCH_CHECK(
        V_fp8.dim() == 4 && V_fp8.size(0) == C::BATCH &&
        V_fp8.size(1) == C::HEADS && V_fp8.size(2) == 128 &&
        V_fp8.size(3) == C::N_TILES*128,
        "V_fp8 must be contiguous [B,H,128,S]");

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(
            Q_sc, C::BATCH, C::N_TILES, C::HEADS * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, C::BATCH, C::HEADS, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(
            K_sc, C::BATCH, C::N_TILES*2, C::HEADS * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, C::BATCH, C::HEADS, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_fp8_gl>(V_fp8),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        0};
    g.disable_lse_store = 0;

    const int grid = C::LOGICAL_JOBS;
    constexpr int DYNAMIC_SMEM = Layout::bytes();
    {
        std::lock_guard<std::mutex> lock(upstream_mxfp4_pipeline_topology_mutex);
        upstream_mxfp4_pipeline_last_topology = {
            1, grid, C::NUM_THREADS, C::QUERY_STAGES, C::KV_STAGES,
            C::TMEM_COLUMNS, DYNAMIC_SMEM, 0, C::LOGICAL_JOBS,
            C::BATCH * C::HEADS * C::N_TILES};
    }
    CUDACHECK(cudaFuncSetAttribute(
        kernel_fp4pv_tk_hao_native_fullgrid<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM));
    kernel_fp4pv_tk_hao_native_fullgrid<C>
        <<<dim3(grid), dim3(C::NUM_THREADS), DYNAMIC_SMEM,
           at::cuda::getCurrentCUDAStream()>>>(g);
    CUDACHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_streaming_live_mxfp4",
          &dispatch_upstream_mxfp4_pipeline);
    m.def("read_upstream_mxfp4_pipeline_topology",
          &read_upstream_mxfp4_pipeline_topology);
}

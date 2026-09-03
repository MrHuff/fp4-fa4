// Minimal host boundary for the exact B1/S4096/H64 uniform-pair K/V-only
// cluster route.  No generic dispatch catalog is linked into this artifact.

struct large_i73_uniform_kv_topology_state {
    long long valid = 0;
    long long batch = 0;
    long long heads = 0;
    long long seqlen = 0;
    long long logical_query_ctas = 0;
    long long physical_grid_ctas = 0;
    long long physical_clusters = 0;
    long long queries_per_cta = 0;
    long long cluster_size = 0;
    long long threads_per_cta = 0;
    long long uniform_pair_max = 0;
    long long k_v_only_multicast = 0;
};

static std::mutex large_i73_uniform_kv_topology_mutex;
static large_i73_uniform_kv_topology_state
    large_i73_uniform_kv_last_topology;

pybind11::dict read_large_i73_uniform_kv_topology() {
    std::lock_guard<std::mutex> lock(large_i73_uniform_kv_topology_mutex);
    const auto &state = large_i73_uniform_kv_last_topology;
    pybind11::dict out;
    out["schema"] = "mxfp4_large_i73_uniform_kv_topology_v1";
    out["valid"] = state.valid;
    out["batch"] = state.batch;
    out["heads"] = state.heads;
    out["seqlen"] = state.seqlen;
    out["logical_query_ctas"] = state.logical_query_ctas;
    out["physical_grid_ctas"] = state.physical_grid_ctas;
    out["physical_clusters"] = state.physical_clusters;
    out["queries_per_cta"] = state.queries_per_cta;
    out["cluster_size"] = state.cluster_size;
    out["threads_per_cta"] = state.threads_per_cta;
    out["uniform_pair_max"] = state.uniform_pair_max;
    out["k_v_only_multicast"] = state.k_v_only_multicast;
    return out;
}

void dispatch_large_i73_uniform_kv(
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
    bool persistent_launch) {
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

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    K.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "uniform K/V route requires packed MXFP4 Q/K");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 &&
                    LSE.dtype() == at::ScalarType::Float,
                "uniform K/V route requires BF16 O and FP32 LSE");
    TORCH_CHECK(!persistent_launch,
                "uniform K/V route is exact nonpersistent full-grid");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2,
                "p_quant_mode must be in [0, 2]");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(batch == 1 && heads == 64 && seqlen == 4096,
                "uniform K/V route is fail-closed to B1/S4096/H64");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && O.dim() == 4,
                "Q/K/O must be rank-4");
    TORCH_CHECK(Q.size(3) == 96 && K.size(3) == 96 && O.size(3) == 128,
                "packed Q/K or output head dimension is invalid");
    TORCH_CHECK(K.size(0) == batch && K.size(1) == heads &&
                    K.size(2) == seqlen && O.size(0) == batch &&
                    O.size(1) == seqlen && O.size(2) == heads,
                "Q/K/O shape mismatch");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch &&
                    LSE.size(1) == heads && LSE.size(2) == 1 &&
                    LSE.size(3) == seqlen,
                "LSE must have shape [B,H,1,S]");
    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_mxfp4_v_scale_half_contract(V_sc, batch, heads, seqlen);

    using C = config_fp4pv_large_i73_uniform_kv_owned;
    static_assert(std::is_base_of_v<large_i73_uniform_kv_canonical_base, C>);
    static_assert(C::CLUSTER_SIZE == 2 && C::NUM_THREADS == 384);
    static_assert(C::Mb == 128 && C::Nb == 128);
    static_assert(C::Dqk == 192 && C::Dvo == 128);
    static_assert(C::ONLINE_STAGE2_EX2_EMU_PERIOD == 16);
    static_assert(C::ONLINE_LAZY_OUTPUT_RESCALE);
    static_assert(!C::ONLINE_Q4_HEAD_INTERLEAVED_MAPPING);
    static_assert(C::ONLINE_ROWPAR2_RANKLOCAL);
    static_assert(C::ONLINE_ROWPAR2_SHARED_K_LIFETIME);
    static_assert(C::ONLINE_ROWPAR2_V_MULTICAST);
    static_assert(C::ONLINE_ROWPAR2_SYMMETRIC_LOCAL_K_COMPLETION);
    static_assert(C::ONLINE_ROWPAR2_UNIFORM_PAIR_MAX_ITERS);
    static_assert(C::ONLINE_ROWPAR2_CANONICAL_LOCAL_FASTPATH);
    static_assert(C::ONLINE_ROWPAR2_DEFER_K_SC_SHARED_UNTIL_LOCAL_QK_COMPLETION);
    static_assert(C::ONLINE_ROWPAR2_COMBINED_SYMMETRIC_DEFERRED_K_COMPLETION);
    static_assert(C::ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_PAYLOAD_MBAR);
    static_assert(C::ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_SCALE_MBAR);
    static_assert(C::ONLINE_ROWPAR2_K_PAYLOAD_ARM_AFTER_REUSE);
    static_assert(C::ONLINE_ROWPAR2_K_SCALE_ARM_AFTER_REUSE);
    static_assert(C::ONLINE_ROWPAR2_CLUSTER_SCOPE_K_SCALE_HANDOFF);
    static_assert(C::ONLINE_ROWPAR2_CLUSTER_SCOPE_K_REUSE);

    const char *route = std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    TORCH_CHECK(route != nullptr &&
                    std::strcmp(route,
                        "real_fwd_large_i73_cluster2_uniform_kvmc_owned") == 0,
                "candidate-only extension requires exact uniform K/V route");

    using G = globals_fp4pv_mxfp4_dv<C, true>;
    constexpr int K_SCALE_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / K_SCALE_DEPTH_DIVISOR;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(
            Q_sc, batch, seqlen / C::Mb,
            heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(
            Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(
            K_sc, batch, k_sc_depth,
            heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(
            K_sg, batch, heads, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        1.0f,
        p_quant_mode};
    g.disable_lse_store = 0;
    g.v_scale_records_per_head = V_sc.size(2);

    constexpr int LOGICAL_QUERY_CTAS = 64 * (4096 / 128);
    constexpr int GRID_CTAS = 2048;
    constexpr int PHYSICAL_CLUSTERS = GRID_CTAS / C::CLUSTER_SIZE;
    static_assert(LOGICAL_QUERY_CTAS == GRID_CTAS);
    static_assert(PHYSICAL_CLUSTERS == 1024);
    {
        std::lock_guard<std::mutex> lock(large_i73_uniform_kv_topology_mutex);
        large_i73_uniform_kv_last_topology = {
            1, batch, heads, seqlen, LOGICAL_QUERY_CTAS, GRID_CTAS,
            PHYSICAL_CLUSTERS, 1, C::CLUSTER_SIZE, C::NUM_THREADS, 1, 1};
    }

    const int dynamic_shared_memory = g.dynamic_shared_memory();
    LaunchConfig<true, false> launch_config(
        dim3(GRID_CTAS), g.block(), dynamic_shared_memory,
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaFuncSetAttribute(
        kernel_streaming_live_fp4pv<C, true, -1, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_streaming_live_fp4pv<C, true, -1, false, true>,
        g,
        0,
        0,
        -1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_streaming_live_mxfp4",
          &dispatch_large_i73_uniform_kv,
          "Exact S4096/H64 canonical-local K/V-only cluster route");
    m.def("read_large_i73_uniform_kv_topology",
          &read_large_i73_uniform_kv_topology,
          "Read the last uniform K/V launch topology");
}

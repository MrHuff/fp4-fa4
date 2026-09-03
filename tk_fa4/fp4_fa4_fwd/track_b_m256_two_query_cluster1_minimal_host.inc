// Minimal fail-closed host binding for the isolated M256 candidate.

struct track_b_m256_topology_state {
    long long valid = 0;
    long long batch = 0;
    long long heads = 0;
    long long seqlen = 0;
    long long logical_query_tiles = 0;
    long long logical_paired_jobs = 0;
    long long physical_grid_ctas = 0;
    long long queries_per_cta = 0;
    long long cluster_size = 0;
    long long threads_per_cta = 0;
};

static std::mutex track_b_m256_topology_mutex;
static track_b_m256_topology_state track_b_m256_last_topology;

pybind11::dict read_track_b_m256_two_query_topology() {
    std::lock_guard<std::mutex> lock(track_b_m256_topology_mutex);
    const auto &state = track_b_m256_last_topology;
    pybind11::dict out;
    out["schema"] = "mxfp4_track_b_m256_two_query_cluster1_topology_v1";
    out["valid"] = state.valid;
    out["batch"] = state.batch;
    out["heads"] = state.heads;
    out["seqlen"] = state.seqlen;
    out["logical_query_tiles"] = state.logical_query_tiles;
    out["logical_paired_jobs"] = state.logical_paired_jobs;
    out["physical_grid_ctas"] = state.physical_grid_ctas;
    out["queries_per_cta"] = state.queries_per_cta;
    out["cluster_size"] = state.cluster_size;
    out["threads_per_cta"] = state.threads_per_cta;
    return out;
}

void dispatch_track_b_m256_two_query_cluster1(
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
    using C = config_fp4pv_track_b_m256_two_query_cluster1_scalarized;
    using G = globals_fp4pv_mxfp4_dv<C, true>;

    const char *fwd_config = std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    TORCH_CHECK(
        fwd_config != nullptr &&
            std::strcmp(
                fwd_config,
                "real_fwd_track_b_m256_two_query_cluster1_scalarized") == 0,
        "candidate-only M256 extension requires its exact route; "
        "auto/default and policy aliases are forbidden");

    static_assert(C::TRACK_B_M256_TWO_QUERY_CLUSTER1);
    static_assert(C::CLUSTER_SIZE == 1);
    static_assert(C::NUM_SOFTMAXXERS == 2);
    static_assert(C::TOTAL_WGS == 3);
    static_assert(C::Mb == 128 && C::Nb == 128);
    static_assert(C::Dqk == 192 && C::Dvo == 128);
    static_assert(C::NUM_THREADS == 384);

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
    TORCH_CHECK(!persistent_launch,
                "M256 cluster-1 candidate is a nonpersistent half-grid route");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2,
                "p_quant_mode must be in [0, 2]");
    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    O.dtype() == at::ScalarType::BFloat16 &&
                    LSE.dtype() == at::ScalarType::Float,
                "M256 candidate input/output dtypes do not match MXFP4 FA4");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(batch == 1, "M256 candidate currently requires batch=1");
    TORCH_CHECK(seqlen % C::Mb == 0,
                "M256 candidate requires an integral M128 tile count");
    const bool legality_shape =
        seqlen == 256 && (heads == 1 || heads == 2);
    const bool target_shape =
        (seqlen == 512 && heads == 64) ||
        (seqlen == 1024 && heads == 32);
    TORCH_CHECK(legality_shape || target_shape,
                "M256 candidate is fail-closed to S256 legality and "
                "S512H64/S1024H32 target cells");
    TORCH_CHECK((seqlen / C::Mb) % 2 == 0,
                "M256 performance route requires an even M128 tile count");
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
    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, C::Dvo);
    check_fp4pv_mxfp4_v_scale_half_contract(V_sc, batch, heads, seqlen);

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

    const int tiles_m = seqlen / C::Mb;
    const int pairs_per_head = (tiles_m + 1) / 2;
    const int logical_query_tiles = batch * heads * tiles_m;
    const int logical_paired_jobs = batch * heads * pairs_per_head;
    const dim3 grid(logical_paired_jobs);
    TORCH_CHECK(grid.x * 2 >= static_cast<unsigned>(logical_query_tiles) &&
                    grid.x * 2 <
                        static_cast<unsigned>(logical_query_tiles + 2 * heads),
                "M256 half-grid topology calculation drifted");

    {
        std::lock_guard<std::mutex> lock(track_b_m256_topology_mutex);
        track_b_m256_last_topology = {
            1, batch, heads, seqlen, logical_query_tiles,
            logical_paired_jobs, static_cast<long long>(grid.x),
            2, C::CLUSTER_SIZE, C::NUM_THREADS};
    }

    constexpr int dynamic_shared_memory =
        fp4pv_one_n128_3wg_smem_layout<C>::bytes();
    CUDACHECK(cudaFuncSetAttribute(
        kernel_fp4pv_one_n128_3wg_concurrent_halfgrid<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    kernel_fp4pv_one_n128_3wg_concurrent_halfgrid<C>
        <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
           at::cuda::getCurrentCUDAStream()>>>(g);
    CUDACHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_streaming_live_mxfp4",
          &dispatch_track_b_m256_two_query_cluster1,
          "Track-B M256 two-query cluster-1 MXFP4 forward");
    m.def("read_track_b_m256_two_query_topology",
          &read_track_b_m256_two_query_topology,
          "Read the last fail-closed M256 launch topology");
    // Compatibility name used only by the hardened Track-B correctness
    // worker.  Both bindings return the same M256-specific schema and state.
    m.def("read_track_b_stage2_cap2_topology",
          &read_track_b_m256_two_query_topology,
          "Read the last fail-closed M256 launch topology (gate alias)");
}

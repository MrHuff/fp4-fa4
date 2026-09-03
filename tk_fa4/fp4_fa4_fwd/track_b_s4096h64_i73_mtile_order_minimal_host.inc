// Minimal host/pybind boundary for the two exact S4096/H64 I73 CTA orders.

struct track_b_s4096h64_i73_topology_state {
    long long valid = 0;
    long long route_kind = 0;  // 1=ascending candidate, 2=descending control.
    long long batch = 0;
    long long heads = 0;
    long long seqlen = 0;
    long long logical_query_tiles = 0;
    long long physical_grid_ctas = 0;
    long long queries_per_cta = 0;
    long long cluster_size = 0;
    long long threads_per_cta = 0;
    long long m_tile_order = 0;
};

static std::mutex track_b_s4096h64_i73_topology_mutex;
static track_b_s4096h64_i73_topology_state
    track_b_s4096h64_i73_last_topology;

pybind11::dict read_track_b_s4096h64_i73_mtile_order_topology() {
    std::lock_guard<std::mutex> lock(
        track_b_s4096h64_i73_topology_mutex);
    const auto &state = track_b_s4096h64_i73_last_topology;
    pybind11::dict out;
    out["schema"] = "mxfp4_track_b_s4096h64_i73_mtile_order_topology_v1";
    out["valid"] = state.valid;
    out["route_kind"] = state.route_kind;
    out["batch"] = state.batch;
    out["heads"] = state.heads;
    out["seqlen"] = state.seqlen;
    out["logical_query_tiles"] = state.logical_query_tiles;
    out["physical_grid_ctas"] = state.physical_grid_ctas;
    out["queries_per_cta"] = state.queries_per_cta;
    out["cluster_size"] = state.cluster_size;
    out["threads_per_cta"] = state.threads_per_cta;
    out["m_tile_order"] = state.m_tile_order;
    return out;
}

void dispatch_track_b_s4096h64_i73_mtile_order(
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
                "I73 order routes require packed MXFP4 Q/K");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 &&
                    LSE.dtype() == at::ScalarType::Float,
                "I73 order routes require BF16 O and FP32 LSE");
    TORCH_CHECK(!persistent_launch,
                "I73 order routes are exact nonpersistent full-grid routes");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2,
                "p_quant_mode must be in [0, 2]");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(batch == 1 && heads == 64 && seqlen == 4096,
                "I73 order routes are fail-closed to batch1 S4096/H64");
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
    check_fp4pv_mxfp4_v_scale_half_contract(
        V_sc, batch, heads, seqlen);

    auto launch = [&]<typename C>(std::type_identity<C>) {
        static_assert(std::is_base_of_v<track_b_s4096h64_i73_base, C>);
        static_assert(C::TRACK_B_S4096H64_M_TILE_ORDER == 1 ||
                      C::TRACK_B_S4096H64_M_TILE_ORDER == -1);
        static_assert(C::CLUSTER_SIZE == 1);
        static_assert(C::Mb == 128 && C::Nb == 128);
        static_assert(C::Dqk == 192 && C::Dvo == 128);
        static_assert(C::NUM_THREADS == 384);
        static_assert(C::ONLINE_STAGE2_EX2_EMU_PERIOD == 16);
        static_assert(C::ONLINE_LAZY_OUTPUT_RESCALE);
        static_assert(C::ONLINE_Q4_HEAD_INTERLEAVED_MAPPING);

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

        constexpr int LOGICAL_QUERY_TILES = 64 * (4096 / 128);
        constexpr int GRID_CTAS = 2048;
        static_assert(LOGICAL_QUERY_TILES == GRID_CTAS);
        const dim3 grid(GRID_CTAS);
        {
            std::lock_guard<std::mutex> lock(
                track_b_s4096h64_i73_topology_mutex);
            track_b_s4096h64_i73_last_topology = {
                1,
                C::TRACK_B_S4096H64_M_TILE_ORDER > 0 ? 1 : 2,
                batch,
                heads,
                seqlen,
                LOGICAL_QUERY_TILES,
                GRID_CTAS,
                1,
                C::CLUSTER_SIZE,
                C::NUM_THREADS,
                C::TRACK_B_S4096H64_M_TILE_ORDER,
            };
        }

        const int dynamic_shared_memory = g.dynamic_shared_memory();
        LaunchConfig<true, false> launch_config(
            grid, g.block(), dynamic_shared_memory,
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
    };

    const char *route = std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    TORCH_CHECK(route != nullptr,
                "candidate-only I73 order extension requires an exact route");
    if (std::strcmp(
            route,
            "real_fwd_track_b_s4096h64_i73_mtile_ascending") == 0) {
        launch(std::type_identity<
               config_fp4pv_track_b_s4096h64_i73_mtile_ascending>{});
    } else if (std::strcmp(
                   route,
                   "real_fwd_track_b_s4096h64_i73_mtile_descending_control") ==
               0) {
        launch(std::type_identity<
               config_fp4pv_track_b_s4096h64_i73_mtile_descending_control>{});
    } else {
        TORCH_CHECK(false,
                    "candidate-only I73 order extension forbids auto/default "
                    "and unrelated routes");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_streaming_live_mxfp4",
          &dispatch_track_b_s4096h64_i73_mtile_order,
          "Track-B S4096/H64 I73 M-tile order discriminator");
    m.def("read_track_b_s4096h64_i73_mtile_order_topology",
          &read_track_b_s4096h64_i73_mtile_order_topology,
          "Read the last exact I73 order launch topology");
}

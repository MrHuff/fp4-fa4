// Minimal candidate-only host dispatch.  Keep every launch call site in this
// file attributable to one of the two Track-B config types below.

struct track_b_stage2_cap2_topology_state {
    long long valid = 0;
    long long route_kind = 0;  // 1=cap2 target, 2=full-grid control.
    long long batch = 0;
    long long heads = 0;
    long long seqlen = 0;
    long long logical_tasks = 0;
    long long physical_grid_ctas = 0;
    long long grid_cap = 0;
    long long num_sm = 0;
    long long cluster_size = 0;
    long long threads_per_cta = 0;
    long long allow_persistent = 0;
    long long persistent_requested = 0;
    long long persistent_grid_safe = 0;
    long long use_persistent = 0;
};

static std::mutex track_b_stage2_cap2_topology_mutex;
static track_b_stage2_cap2_topology_state
    track_b_stage2_cap2_last_topology;

static void record_track_b_stage2_cap2_topology(
    const track_b_stage2_cap2_topology_state &state) {
    std::lock_guard<std::mutex> lock(track_b_stage2_cap2_topology_mutex);
    track_b_stage2_cap2_last_topology = state;
}

pybind11::dict read_track_b_stage2_cap2_topology() {
    std::lock_guard<std::mutex> lock(track_b_stage2_cap2_topology_mutex);
    const auto &state = track_b_stage2_cap2_last_topology;
    pybind11::dict out;
    out["schema"] = "mxfp4_track_b_stage2_cap2_topology_v1";
    out["valid"] = state.valid;
    out["route_kind"] = state.route_kind;
    out["batch"] = state.batch;
    out["heads"] = state.heads;
    out["seqlen"] = state.seqlen;
    out["logical_tasks"] = state.logical_tasks;
    out["physical_grid_ctas"] = state.physical_grid_ctas;
    out["grid_cap"] = state.grid_cap;
    out["num_sm"] = state.num_sm;
    out["cluster_size"] = state.cluster_size;
    out["threads_per_cta"] = state.threads_per_cta;
    out["allow_persistent"] = state.allow_persistent;
    out["persistent_requested"] = state.persistent_requested;
    out["persistent_grid_safe"] = state.persistent_grid_safe;
    out["use_persistent"] = state.use_persistent;
    return out;
}

template <typename C>
static inline dim3 track_b_fullgrid_dim(
    int batch, int heads, int seqlen) {
    const int tiles_m = seqlen / C::Mb;
    const int tiles_per_cluster =
        C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block =
        (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    return dim3(batch * heads * num_block * C::CLUSTER_SIZE);
}

template <typename C>
static inline dim3 track_b_persistent_dim(
    int batch, int heads, int seqlen) {
    const int total_bids =
        static_cast<int>(track_b_fullgrid_dim<C>(batch, heads, seqlen).x);
    constexpr int grid_cap = fp4pv_persistent_grid_cap<C>();
    int grid_x = total_bids < grid_cap ? total_bids : grid_cap;
    if (grid_x < C::CLUSTER_SIZE) {
        grid_x = C::CLUSTER_SIZE;
    } else {
        grid_x =
            ((grid_x + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE) *
            C::CLUSTER_SIZE;
    }
    return dim3(grid_x);
}

void dispatch_forward_streaming_live_mxfp4(
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
    CHECK_INPUT(O);
    CHECK_INPUT(LSE);

    TORCH_CHECK(
        Q.dtype() == at::ScalarType::Float4_e2m1fn_x2,
        "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(
        K.dtype() == at::ScalarType::Float4_e2m1fn_x2,
        "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16,
                "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float,
                "LSE must be float32");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2,
                "p_quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4,
                "Q and K must be rank-4 packed tensors");
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
    TORCH_CHECK(Q.size(3) == 96 && K.size(3) == 96,
                "Q and K packed head_dim must be 96");
    TORCH_CHECK(O.size(3) == 128, "O head_dim must be 128");
    TORCH_CHECK(seqlen % 128 == 0,
                "sequence length must be divisible by 128");
    TORCH_CHECK(
        LSE.dim() == 4 && LSE.size(0) == batch && LSE.size(1) == heads &&
            LSE.size(2) == 1 && LSE.size(3) == seqlen,
        "LSE must have shape (batch, heads, 1, seqlen)");

    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_mxfp4_v_scale_half_contract(V_sc, batch, heads, seqlen);
    constexpr int K_SCALE_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / K_SCALE_DEPTH_DIVISOR;

    auto launch_track_b = [&]<typename C>(
                              std::type_identity<C>,
                              bool allow_persistent) {
        constexpr bool is_target =
            fp4pv_track_b_stage2_cap2_target<C>::value;
        constexpr bool is_control =
            fp4pv_track_b_stage2_fullgrid_control<C>::value;
        static_assert(is_target != is_control);
        static_assert(
            fp4pv_require_track_b_stage2_cap2_persistent<C>::value ==
            is_target);
        static_assert(C::NUM_SM == 152);
        static_assert(C::CLUSTER_SIZE == 1);
        static_assert(C::NUM_SOFTMAXXERS == 1);
        static_assert(C::Mb == 128 && C::Nb == 128);
        static_assert(C::Dqk == 192 && C::Dvo == 128);
        static_assert(fp4pv_persistent_grid_multiplier<C>::value == 2);

        using G = globals_fp4pv_mxfp4_dv<C, true>;
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

        const bool diagnostic_shape =
            heads == 2 && (seqlen == 256 || seqlen == 384);
        const bool matrix_shape =
            (seqlen == 512 || seqlen == 1024 || seqlen == 2048) &&
            (heads == 4 || heads == 16 || heads == 32 || heads == 64);
        TORCH_CHECK(
            batch == 1 && (diagnostic_shape || matrix_shape),
            "Track-B cap2 routes support only batch=1, S256/S384-H2 "
            "legality probes, or S512/S1024/S2048 with H4/H16/H32/H64; "
            "S4096 is fail-closed");

        const int tiles_m = seqlen / C::Mb;
        const int logical_tasks = batch * heads * tiles_m;
        const int tiles_per_cluster =
            C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
        const int num_block =
            (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
        const int total_bids =
            batch * heads * num_block * C::CLUSTER_SIZE;
        TORCH_CHECK(total_bids == logical_tasks,
                    "Track-B logical task calculation drifted");

        const int batch_heads = batch * heads;
        const bool persistent_grid_safe =
            fp4pv_online_persistent_outer_clc_multitask_reuse<C>::value ||
            fp4pv_online_persistent_outer_clc_static_stride<C>::value ||
            (!(batch_heads > 4 && seqlen >= 4096) &&
             !(batch_heads <= 4 && seqlen >= 9984));
        const bool use_persistent =
            persistent_grid_safe && allow_persistent && persistent_launch &&
            (fp4pv_force_persistent_grid<C>::value ||
             total_bids <= 2 * C::NUM_SM);
        const dim3 grid = use_persistent
            ? track_b_persistent_dim<C>(batch, heads, seqlen)
            : track_b_fullgrid_dim<C>(batch, heads, seqlen);
        constexpr int grid_cap = 2 * C::NUM_SM;

        if constexpr (is_target) {
            const int expected_grid =
                logical_tasks < grid_cap ? logical_tasks : grid_cap;
            TORCH_CHECK(
                allow_persistent && persistent_launch &&
                    persistent_grid_safe && use_persistent,
                "Track-B cap2 target requires a real persistent launch; "
                "full-grid fallback is forbidden");
            TORCH_CHECK(
                grid.x == static_cast<unsigned>(expected_grid),
                "Track-B cap2 target selected an unexpected physical grid");
        } else {
            TORCH_CHECK(
                !allow_persistent && !persistent_launch && !use_persistent,
                "Track-B full-grid control forbids persistent launch state");
            TORCH_CHECK(
                grid.x == static_cast<unsigned>(logical_tasks),
                "Track-B full-grid control must use one CTA per M128 task");
        }

        record_track_b_stage2_cap2_topology({
            1,
            is_target ? 1 : 2,
            batch,
            heads,
            seqlen,
            logical_tasks,
            static_cast<long long>(grid.x),
            grid_cap,
            C::NUM_SM,
            C::CLUSTER_SIZE,
            C::NUM_THREADS,
            allow_persistent ? 1 : 0,
            persistent_launch ? 1 : 0,
            persistent_grid_safe ? 1 : 0,
            use_persistent ? 1 : 0,
        });

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

    const char *fwd_config = std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    TORCH_CHECK(
        fwd_config != nullptr,
        "candidate-only Track-B extension requires an exact route; "
        "auto/default is forbidden");
    if (std::strcmp(
            fwd_config,
            "real_fwd_track_b_stage2_fullk_cap2_saturated") == 0) {
        launch_track_b(
            std::type_identity<
                config_fp4pv_track_b_stage2_fullk_cap2_saturated>{},
            true);
    } else if (std::strcmp(
                   fwd_config,
                   "real_fwd_track_b_stage2_fullk_fullgrid_control") == 0) {
        launch_track_b(
            std::type_identity<
                config_fp4pv_track_b_stage2_fullk_fullgrid_control>{},
            false);
    } else {
        TORCH_CHECK(
            false,
            "candidate-only Track-B extension requires the exact cap2 target "
            "or full-grid control route; auto/default is forbidden");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_streaming_live_mxfp4",
        &dispatch_forward_streaming_live_mxfp4,
        "Track-B candidate-only MXFP4 forward");
    m.def(
        "read_track_b_stage2_cap2_topology",
        &read_track_b_stage2_cap2_topology,
        "Read the last fail-closed Track-B launch topology");
}

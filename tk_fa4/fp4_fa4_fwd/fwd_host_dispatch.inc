#ifndef TK_FA4_HYBRID_ISSUE_PROXY_ONLY_BUILD
#define TK_FA4_HYBRID_ISSUE_PROXY_ONLY_BUILD 0
#endif

#ifndef TK_FA4_HYBRID_QUANT_READER_PROXY_ONLY_BUILD
#define TK_FA4_HYBRID_QUANT_READER_PROXY_ONLY_BUILD 0
#endif


#if (defined(TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) && \
     TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) || \
    (defined(TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD) && \
     TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD)
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
#endif

void reset_pairpsc_desc_diag() {
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_pairpsc_desc_diag));
    CUDACHECK(cudaMemset(
        ptr, 0,
        FP4PV_PAIRPSC_DESC_DIAG_WORDS * sizeof(unsigned long long)));
}

std::vector<unsigned long long> read_pairpsc_desc_diag() {
    std::vector<unsigned long long> out(
        FP4PV_PAIRPSC_DESC_DIAG_WORDS, 0);
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_pairpsc_desc_diag));
    CUDACHECK(cudaMemcpy(
        out.data(), ptr,
        FP4PV_PAIRPSC_DESC_DIAG_WORDS * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    return out;
}

std::vector<unsigned long long> read_pairpsc_desc_diag_nonblocking() {
    std::vector<unsigned long long> out(
        FP4PV_PAIRPSC_DESC_DIAG_WORDS, 0);
    void *ptr = nullptr;
    cudaStream_t stream = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_pairpsc_desc_diag));
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDACHECK(cudaMemcpyAsync(
        out.data(), ptr,
        FP4PV_PAIRPSC_DESC_DIAG_WORDS * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
    return out;
}

#if TK_FA4_NATIVE_GROUP2_RAW_QK_DIAG_BUILD
void reset_native_group2_raw_qk_diag() {
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &ptr, fp4pv_native_group2_raw_qk_diag));
    CUDACHECK(cudaMemset(
        ptr, 0,
        FP4PV_NATIVE_GROUP2_RAW_QK_DIAG_WORDS * sizeof(unsigned int)));
}

void set_native_group2_raw_qk_diag_target(
    int batch, int head, int query_row) {
    TORCH_CHECK(batch >= 0, "raw-QK diagnostic batch must be nonnegative");
    TORCH_CHECK(head >= 0, "raw-QK diagnostic head must be nonnegative");
    TORCH_CHECK(query_row >= 0,
                "raw-QK diagnostic query row must be nonnegative");
    const int control[4] = {1, batch, head, query_row};
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &ptr, fp4pv_native_group2_raw_qk_diag_control));
    CUDACHECK(cudaMemcpy(
        ptr, control, sizeof(control), cudaMemcpyHostToDevice));
}

std::vector<unsigned int> read_native_group2_raw_qk_diag() {
    std::vector<unsigned int> out(
        FP4PV_NATIVE_GROUP2_RAW_QK_DIAG_WORDS, 0);
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &ptr, fp4pv_native_group2_raw_qk_diag));
    CUDACHECK(cudaMemcpy(
        out.data(), ptr, out.size() * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));
    return out;
}
#endif

#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
void reset_native_group2_selected_row_trace() {
    void *trace_ptr = nullptr;
    void *control_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &trace_ptr, fp4pv_native_group2_selected_row_trace));
    CUDACHECK(cudaGetSymbolAddress(
        &control_ptr, fp4pv_native_group2_selected_row_trace_control));
    CUDACHECK(cudaMemset(
        trace_ptr, 0,
        FP4PV_NATIVE_GROUP2_SELECTED_ROW_TRACE_WORDS *
            sizeof(unsigned int)));
    CUDACHECK(cudaMemset(
        control_ptr, 0,
        sizeof(fp4pv_native_group2_selected_row_trace_control)));
}

void set_native_group2_selected_row_trace_target(
    int batch, int head, int query_row, int target_idx, int route_kind) {
    TORCH_CHECK(batch >= 0, "selected-row batch must be nonnegative");
    TORCH_CHECK(head >= 0, "selected-row head must be nonnegative");
    TORCH_CHECK(query_row >= 0,
                "selected-row query must be nonnegative");
    TORCH_CHECK(target_idx >= -1 &&
                    target_idx < FP4PV_NATIVE_GROUP2_SELECTED_ROW_MAX_ITERS,
                "selected-row target_idx must be -1 or in [0, 31]");
    TORCH_CHECK(route_kind == 11 || route_kind == 12,
                "selected-row route kind must be A=11 or B1=12");
    const int control[6] = {
        1, batch, head, query_row, target_idx, route_kind};
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &ptr, fp4pv_native_group2_selected_row_trace_control));
    CUDACHECK(cudaMemcpy(
        ptr, control, sizeof(control), cudaMemcpyHostToDevice));
}

std::vector<unsigned int> read_native_group2_selected_row_trace() {
    std::vector<unsigned int> out(
        FP4PV_NATIVE_GROUP2_SELECTED_ROW_TRACE_WORDS, 0);
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &ptr, fp4pv_native_group2_selected_row_trace));
    CUDACHECK(cudaMemcpy(
        out.data(), ptr, out.size() * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));
    return out;
}

std::vector<int> native_group2_selected_row_trace_contract() {
    return {
        1,
        FP4PV_NATIVE_GROUP2_SELECTED_ROW_MAX_ITERS,
        FP4PV_NATIVE_GROUP2_SELECTED_ROW_RECORD_WORDS,
        FP4PV_NATIVE_GROUP2_SELECTED_ROW_HEADER_WORDS,
        FP4PV_NATIVE_GROUP2_SELECTED_ROW_RAW_WORDS,
        FP4PV_NATIVE_GROUP2_SELECTED_ROW_TRACE_WORDS};
}
#endif

void set_pairpsc_desc_diag_slot(int slot, unsigned long long value) {
    TORCH_CHECK(
        slot >= 0 && slot < FP4PV_PAIRPSC_DESC_DIAG_WORDS,
        "diag slot out of range");
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_pairpsc_desc_diag));
    auto *typed = reinterpret_cast<unsigned long long *>(ptr);
    CUDACHECK(cudaMemcpy(typed + slot, &value, sizeof(unsigned long long), cudaMemcpyHostToDevice));
}

#if TK_FA4_KMC_PASSIVE_PROBE_BUILD
void reset_kmc_passive_records() {
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_kmc_passive_records));
    CUDACHECK(cudaMemset(
        ptr, 0,
        FP4PV_KMC_PASSIVE_MAX_TASKS * FP4PV_KMC_PASSIVE_FIELDS *
            sizeof(unsigned long long)));
}

std::vector<unsigned long long> read_kmc_passive_records() {
    std::vector<unsigned long long> out(
        FP4PV_KMC_PASSIVE_MAX_TASKS * FP4PV_KMC_PASSIVE_FIELDS, 0);
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_kmc_passive_records));
    CUDACHECK(cudaMemcpy(
        out.data(), ptr, out.size() * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    return out;
}
#endif

#if TK_FA4_REGISTER_OUTPUT_PROBE_BUILD
void reset_progressive_diag_records() {
    void *counters = nullptr;
    void *records = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &counters, fp4pv_progressive_diag_counters));
    CUDACHECK(cudaGetSymbolAddress(
        &records, fp4pv_progressive_diag_records));
    CUDACHECK(cudaMemset(
        counters, 0,
        FP4PV_PROGRESSIVE_DIAG_MAX_BLOCKS *
            FP4PV_PROGRESSIVE_DIAG_ROLES * sizeof(unsigned long long)));
    CUDACHECK(cudaMemset(
        records, 0,
        FP4PV_PROGRESSIVE_DIAG_MAX_BLOCKS *
            FP4PV_PROGRESSIVE_DIAG_ROLES *
            FP4PV_PROGRESSIVE_DIAG_MAX_EVENTS *
            FP4PV_PROGRESSIVE_DIAG_FIELDS * sizeof(unsigned long long)));
}

std::vector<unsigned long long> read_progressive_diag_records() {
    constexpr size_t COUNTERS =
        FP4PV_PROGRESSIVE_DIAG_MAX_BLOCKS *
        FP4PV_PROGRESSIVE_DIAG_ROLES;
    constexpr size_t RECORDS =
        COUNTERS * FP4PV_PROGRESSIVE_DIAG_MAX_EVENTS *
        FP4PV_PROGRESSIVE_DIAG_FIELDS;
    std::vector<unsigned long long> out(COUNTERS + RECORDS, 0);
    void *counters = nullptr;
    void *records = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &counters, fp4pv_progressive_diag_counters));
    CUDACHECK(cudaGetSymbolAddress(
        &records, fp4pv_progressive_diag_records));
    CUDACHECK(cudaMemcpy(
        out.data(), counters, COUNTERS * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(
        out.data() + COUNTERS, records,
        RECORDS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return out;
}

std::vector<unsigned long long> read_progressive_diag_records_nonblocking() {
    constexpr size_t COUNTERS =
        FP4PV_PROGRESSIVE_DIAG_MAX_BLOCKS *
        FP4PV_PROGRESSIVE_DIAG_ROLES;
    constexpr size_t RECORDS =
        COUNTERS * FP4PV_PROGRESSIVE_DIAG_MAX_EVENTS *
        FP4PV_PROGRESSIVE_DIAG_FIELDS;
    std::vector<unsigned long long> out(COUNTERS + RECORDS, 0);
    void *counters = nullptr;
    void *records = nullptr;
    cudaStream_t stream = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &counters, fp4pv_progressive_diag_counters));
    CUDACHECK(cudaGetSymbolAddress(
        &records, fp4pv_progressive_diag_records));
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDACHECK(cudaMemcpyAsync(
        out.data(), counters, COUNTERS * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaMemcpyAsync(
        out.data() + COUNTERS, records,
        RECORDS * sizeof(unsigned long long), cudaMemcpyDeviceToHost,
        stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
    return out;
}
#endif

#if defined(TK_FA4_Q4_LARGE_PROBE_BUILD) || \
    defined(TK_FA4_FULLK_WAVE_PROBE_BUILD) || \
    TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
void reset_q4_large_diag_records() {
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_q4_large_diag_records));
    CUDACHECK(cudaMemset(
        ptr,
        0,
        FP4PV_Q4_LARGE_DIAG_MAX_TASKS * FP4PV_Q4_LARGE_DIAG_FIELDS *
            sizeof(unsigned long long)));
}

std::vector<unsigned long long> read_q4_large_diag_records() {
    std::vector<unsigned long long> out(
        FP4PV_Q4_LARGE_DIAG_MAX_TASKS * FP4PV_Q4_LARGE_DIAG_FIELDS, 0);
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_q4_large_diag_records));
    CUDACHECK(cudaMemcpy(
        out.data(),
        ptr,
        out.size() * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    return out;
}
#endif

#if TK_FA4_OPTION_AB_PROBE_BUILD
void dispatch_mxfp4_option_b_nb64_dvo128(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    at::Tensor Trace,
    int min_blocks_per_sm) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V); CHECK_INPUT(V_sc); CHECK_INPUT(O); CHECK_INPUT(LSE);
    CHECK_INPUT(Trace);
    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Option B Q/K/V must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Float8_e4m3fn &&
                K_sc.dtype() == at::ScalarType::Float8_e4m3fn,
                "Option B Q_sc/K_sc must be float8_e4m3fn");
    TORCH_CHECK(Q_sg.dtype() == at::ScalarType::Float &&
                K_sg.dtype() == at::ScalarType::Float,
                "Option B Q_sg/K_sg must be float32");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte,
                "Option B V_sc must be uint8 E8M0");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 &&
                LSE.dtype() == at::ScalarType::Float &&
                Trace.dtype() == at::ScalarType::Long,
                "Option B requires BF16 O, FP32 LSE, and int64 Trace");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4 &&
                Q_sc.dim() == 4 && K_sc.dim() == 4 &&
                Q_sg.dim() == 2 && K_sg.dim() == 2 &&
                V_sc.dim() == 6 && O.dim() == 4 && LSE.dim() == 4,
                "Option B received an invalid tensor rank");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    constexpr int qk_chunks = 192 / 64;
    TORCH_CHECK(batch > 0 && heads > 0 && seqlen > 0 && seqlen % 128 == 0,
                "Option B requires positive B/H and S divisible by 128");
    TORCH_CHECK(Q.size(3) == 96 &&
                K.sizes() == Q.sizes(),
                "Option B Q/K must have shape (B,H,S,96 packed)");
    TORCH_CHECK(V.size(0) == batch && V.size(1) == heads &&
                V.size(2) == 128 && V.size(3) == seqlen / 2,
                "Option B V must have shape (B,H,128,S/2 packed)");
    TORCH_CHECK(Q_sc.size(0) == batch && Q_sc.size(1) == seqlen / 128 &&
                Q_sc.size(2) == heads * qk_chunks && Q_sc.size(3) == 512,
                "Option B Q_sc must have shape (B,S/128,H*3,512)");
    TORCH_CHECK(K_sc.size(0) == batch && K_sc.size(1) == seqlen / 64 &&
                K_sc.size(2) == heads * qk_chunks && K_sc.size(3) == 512,
                "Option B K_sc must have shape (B,S/64,H*3,512)");
    TORCH_CHECK(Q_sg.size(0) == batch && Q_sg.size(1) == heads &&
                K_sg.sizes() == Q_sg.sizes(),
                "Option B Q_sg/K_sg must have shape (B,H)");
    check_fp4pv_mxfp4_v_scale_half_contract(V_sc, batch, heads, seqlen);
    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen &&
                O.size(2) == heads && O.size(3) == 128,
                "Option B O must have shape (B,S,H,128)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads &&
                LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "Option B LSE must have shape (B,H,1,S)");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "Option B min_blocks_per_sm must be 1 or 2");

    using C = config_fp4pv_option_b_nb64_dvo128;
    using G = globals_mxfp4_option_b_nb64_dvo128<C>;
    constexpr int trace_stride = 16;
    const int grid_blocks = batch * heads * (seqlen / C::Mb);
    const bool enable_trace = Trace.numel() != 0;
    TORCH_CHECK(!enable_trace || Trace.numel() >= grid_blocks * trace_stride,
                "Option B Trace must be empty or contain grid_blocks*16 int64 values");
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        kittens::py::tensor_to_gl<typename G::v_gl>(V),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const float *>(Q_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const float *>(K_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<float *>(LSE.data_ptr()),
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        batch,
        heads,
        seqlen,
        enable_trace ? trace_stride : 0
    };

    if (min_blocks_per_sm == 1) {
        constexpr int ONE_RESIDENT_CONTROL_SMEM = 128 * 1024;
        const int launch_smem = std::max(
            g.dynamic_shared_memory(), ONE_RESIDENT_CONTROL_SMEM);
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), launch_smem,
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        if (enable_trace) {
            CUDACHECK(cudaFuncSetAttribute(
                kernel_mxfp4_option_b_nb64_dvo128<C, 1, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, launch_smem));
            CUDACHECK(cudaLaunchKernelEx(
                launch_config,
                kernel_mxfp4_option_b_nb64_dvo128<C, 1, true>, g));
        } else {
            CUDACHECK(cudaFuncSetAttribute(
                kernel_mxfp4_option_b_nb64_dvo128<C, 1, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, launch_smem));
            CUDACHECK(cudaLaunchKernelEx(
                launch_config,
                kernel_mxfp4_option_b_nb64_dvo128<C, 1, false>, g));
        }
    } else {
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), g.dynamic_shared_memory(),
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        if (enable_trace) {
            CUDACHECK(cudaFuncSetAttribute(
                kernel_mxfp4_option_b_nb64_dvo128<C, 2, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                g.dynamic_shared_memory()));
            CUDACHECK(cudaLaunchKernelEx(
                launch_config,
                kernel_mxfp4_option_b_nb64_dvo128<C, 2, true>, g));
        } else {
            CUDACHECK(cudaFuncSetAttribute(
                kernel_mxfp4_option_b_nb64_dvo128<C, 2, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                g.dynamic_shared_memory()));
            CUDACHECK(cudaLaunchKernelEx(
                launch_config,
                kernel_mxfp4_option_b_nb64_dvo128<C, 2, false>, g));
        }
    }
}

void dispatch_mxfp4_option_a_replay(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    int max_key_tiles) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V); CHECK_INPUT(V_sc); CHECK_INPUT(O); CHECK_INPUT(LSE);
    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Option A Q/K/V must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Float8_e4m3fn &&
                K_sc.dtype() == at::ScalarType::Float8_e4m3fn,
                "Option A Q_sc/K_sc must be float8_e4m3fn");
    TORCH_CHECK(Q_sg.dtype() == at::ScalarType::Float &&
                K_sg.dtype() == at::ScalarType::Float,
                "Option A Q_sg/K_sg must be float32");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte,
                "Option A V_sc must be uint8 E8M0");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 &&
                LSE.dtype() == at::ScalarType::Float,
                "Option A requires BF16 O and FP32 LSE");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4 &&
                Q_sc.dim() == 4 && K_sc.dim() == 4 &&
                Q_sg.dim() == 2 && K_sg.dim() == 2 &&
                V_sc.dim() == 6 && O.dim() == 4 && LSE.dim() == 4,
                "Option A received an invalid tensor rank");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    constexpr int qk_chunks = 192 / 64;
    TORCH_CHECK(batch > 0 && heads > 0 && seqlen > 0 && seqlen % 128 == 0,
                "Option A requires positive B/H and S divisible by 128");
    TORCH_CHECK(Q.size(3) == 96 && K.sizes() == Q.sizes(),
                "Option A Q/K must have shape (B,H,S,96 packed)");
    TORCH_CHECK(V.size(0) == batch && V.size(1) == heads &&
                V.size(2) == 128 && V.size(3) == seqlen / 2,
                "Option A V must have shape (B,H,128,S/2 packed)");
    TORCH_CHECK(Q_sc.size(0) == batch && Q_sc.size(1) == seqlen / 128 &&
                Q_sc.size(2) == heads * qk_chunks && Q_sc.size(3) == 512,
                "Option A Q_sc must have shape (B,S/128,H*3,512)");
    TORCH_CHECK(K_sc.size(0) == batch && K_sc.size(1) == seqlen / 64 &&
                K_sc.size(2) == heads * qk_chunks && K_sc.size(3) == 512,
                "Option A K_sc must have shape (B,S/64,H*3,512)");
    TORCH_CHECK(Q_sg.size(0) == batch && Q_sg.size(1) == heads &&
                K_sg.sizes() == Q_sg.sizes(),
                "Option A Q_sg/K_sg must have shape (B,H)");
    check_fp4pv_mxfp4_v_scale_half_contract(V_sc, batch, heads, seqlen);
    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen &&
                O.size(2) == heads && O.size(3) == 128,
                "Option A O must have shape (B,S,H,128)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads &&
                LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "Option A LSE must have shape (B,H,1,S)");
    TORCH_CHECK(max_key_tiles == 0 || max_key_tiles == 1,
                "Option A max_key_tiles must be 0 (full) or 1 (shell)");

    using C = config_fp4pv_option_a_replay;
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
            K_sc, batch, seqlen / 64,
            heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(
            K_sg, batch, heads, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V),
        reinterpret_cast<const uint8_t *>(V.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        1.0f,
        0
    };
    g.disable_lse_store = 0;
    const int pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
    const dim3 grid(batch * heads * pairs_per_head);
    constexpr int smem = fp4pv_option_a_replay_smem_layout<C>::bytes();
    TORCH_CHECK(smem <= MAX_SHARED_MEMORY - 4096,
                "Option A dynamic SMEM exceeds the guarded device limit");
    if (max_key_tiles == 1) {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_option_a_replay_concurrent_halfgrid<C, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kernel_fp4pv_option_a_replay_concurrent_halfgrid<C, 1>
            <<<grid, dim3(C::NUM_THREADS), smem,
               at::cuda::getCurrentCUDAStream()>>>(g);
    } else {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_option_a_replay_concurrent_halfgrid<C, 0>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kernel_fp4pv_option_a_replay_concurrent_halfgrid<C, 0>
            <<<grid, dim3(C::NUM_THREADS), smem,
               at::cuda::getCurrentCUDAStream()>>>(g);
    }
    CUDACHECK(cudaGetLastError());
}
#endif

#if TK_FA4_TMEM224_PROBE_BUILD
static void check_mxfp4_tmem224_common(
    const at::Tensor &Q, const at::Tensor &Q_sc, const at::Tensor &Q_sg,
    const at::Tensor &K, const at::Tensor &K_sc, const at::Tensor &K_sg,
    const at::Tensor &V, const at::Tensor &V_sc, const at::Tensor &O,
    const at::Tensor &LSE) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V); CHECK_INPUT(V_sc); CHECK_INPUT(O); CHECK_INPUT(LSE);
    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "TMEM224 Q/K/V must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Float8_e4m3fn &&
                K_sc.dtype() == at::ScalarType::Float8_e4m3fn,
                "TMEM224 Q_sc/K_sc must be float8_e4m3fn");
    TORCH_CHECK(Q_sg.dtype() == at::ScalarType::Float &&
                K_sg.dtype() == at::ScalarType::Float &&
                V_sc.dtype() == at::ScalarType::Byte,
                "TMEM224 scalar scales must be FP32 and V_sc uint8");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 &&
                LSE.dtype() == at::ScalarType::Float,
                "TMEM224 O/LSE must be BF16/FP32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4 &&
                Q_sc.dim() == 4 && K_sc.dim() == 4 &&
                Q_sg.dim() == 2 && K_sg.dim() == 2 &&
                V_sc.dim() == 6 && O.dim() == 4 && LSE.dim() == 4,
                "TMEM224 received an invalid tensor rank");
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    constexpr int chunks = 3;
    TORCH_CHECK(batch > 0 && heads > 0 && seqlen > 0 && seqlen % 128 == 0,
                "TMEM224 requires positive B/H and S divisible by 128");
    TORCH_CHECK(Q.size(3) == 96 && K.sizes() == Q.sizes(),
                "TMEM224 Q/K must have shape (B,H,S,96 packed)");
    TORCH_CHECK(V.size(0) == batch && V.size(1) == heads &&
                V.size(2) == 128 && V.size(3) == seqlen / 2,
                "TMEM224 V must have shape (B,H,128,S/2 packed)");
    TORCH_CHECK(Q_sc.size(0) == batch && Q_sc.size(1) == seqlen / 128 &&
                Q_sc.size(2) == heads * chunks && Q_sc.size(3) == 512,
                "TMEM224 Q_sc shape mismatch");
    TORCH_CHECK(K_sc.size(0) == batch && K_sc.size(1) == seqlen / 64 &&
                K_sc.size(2) == heads * chunks && K_sc.size(3) == 512,
                "TMEM224 K_sc shape mismatch");
    TORCH_CHECK(Q_sg.size(0) == batch && Q_sg.size(1) == heads &&
                K_sg.sizes() == Q_sg.sizes(),
                "TMEM224 Q_sg/K_sg shape mismatch");
    check_fp4pv_mxfp4_v_scale_half_contract(V_sc, batch, heads, seqlen);
    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen &&
                O.size(2) == heads && O.size(3) == 128,
                "TMEM224 O must have shape (B,S,H,128)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads &&
                LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "TMEM224 LSE must have shape (B,H,1,S)");
}

void dispatch_mxfp4_tmem224_d64_stripmine(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V, at::Tensor V_sc, at::Tensor O, at::Tensor LSE,
    at::Tensor LSE_aux, at::Tensor Trace, int min_blocks_per_sm,
    int max_key_tiles) {
    check_mxfp4_tmem224_common(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V, V_sc, O, LSE);
    CHECK_INPUT(LSE_aux); CHECK_INPUT(Trace);
    TORCH_CHECK(LSE_aux.dtype() == at::ScalarType::Float &&
                LSE_aux.sizes() == LSE.sizes(),
                "D64 stripmine LSE_aux must match LSE");
    TORCH_CHECK(Trace.dtype() == at::ScalarType::Long,
                "D64 stripmine Trace must be int64");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "D64 stripmine min_blocks_per_sm must be 1 or 2");
    TORCH_CHECK(max_key_tiles == 0 || max_key_tiles == 1,
                "D64 stripmine max_key_tiles must be 0 or 1");
    const int batch = Q.size(0), heads = Q.size(1), seqlen = Q.size(2);
    using C = config_fp4pv_tmem224_nb64_dvo64;
    using G = globals_mxfp4_tmem224_pair<C>;
    constexpr int trace_stride = 16;
    const int grid_blocks = batch * heads * ((seqlen / 128 + 1) / 2);
    const bool trace_enabled = Trace.numel() != 0;
    TORCH_CHECK(!trace_enabled || Trace.numel() >= grid_blocks * trace_stride,
                "D64 stripmine Trace is too small");
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        kittens::py::tensor_to_gl<typename G::v_gl>(
            V, batch, heads * 2, 64, seqlen / 2),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const float *>(Q_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const float *>(K_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<float *>(LSE.data_ptr()),
        reinterpret_cast<float *>(LSE_aux.data_ptr()),
        nullptr,
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        batch, heads, seqlen, trace_enabled ? trace_stride : 0};
    auto launch = [&]<int MIN_BLOCKS, int MAX_KEYS, bool TRACE>() {
        int smem = g.dynamic_shared_memory();
        if constexpr (MIN_BLOCKS == 1) {
            smem = std::max(smem, 128 * 1024);
        }
        LaunchConfig<true, false> config(
            g.grid(), g.block(), smem, at::cuda::getCurrentCUDAStream(), 1);
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_tmem224_d64_stripmine<C, MIN_BLOCKS, MAX_KEYS, TRACE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CUDACHECK(cudaLaunchKernelEx(
            config,
            kernel_mxfp4_tmem224_d64_stripmine<
                C, MIN_BLOCKS, MAX_KEYS, TRACE>, g));
    };
    if (min_blocks_per_sm == 1) {
        if (max_key_tiles == 1) {
            if (trace_enabled) launch.template operator()<1, 1, true>();
            else launch.template operator()<1, 1, false>();
        } else {
            if (trace_enabled) launch.template operator()<1, 0, true>();
            else launch.template operator()<1, 0, false>();
        }
    } else {
        if (max_key_tiles == 1) {
            if (trace_enabled) launch.template operator()<2, 1, true>();
            else launch.template operator()<2, 1, false>();
        } else {
            if (trace_enabled) launch.template operator()<2, 0, true>();
            else launch.template operator()<2, 0, false>();
        }
    }
}

void dispatch_mxfp4_tmem224_serial_qpair(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V, at::Tensor V_sc, at::Tensor O, at::Tensor LSE,
    at::Tensor Trace, int min_blocks_per_sm) {
    check_mxfp4_tmem224_common(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V, V_sc, O, LSE);
    CHECK_INPUT(Trace);
    TORCH_CHECK(Trace.dtype() == at::ScalarType::Long,
                "serialized qpair Trace must be int64");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "serialized qpair min_blocks_per_sm must be 1 or 2");
    const int batch = Q.size(0), heads = Q.size(1), seqlen = Q.size(2);
    using C = config_fp4pv_tmem224_nb64_dvo128;
    using G = globals_mxfp4_tmem224_pair<C>;
    constexpr int trace_stride = 16;
    const int grid_blocks = batch * heads * ((seqlen / 128 + 1) / 2);
    const bool trace_enabled = Trace.numel() != 0;
    TORCH_CHECK(!trace_enabled || Trace.numel() >= grid_blocks * trace_stride,
                "serialized qpair Trace is too small");
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        kittens::py::tensor_to_gl<typename G::v_gl>(V),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const float *>(Q_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const float *>(K_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<float *>(LSE.data_ptr()),
        nullptr, nullptr,
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        batch, heads, seqlen, trace_enabled ? trace_stride : 0};
    auto launch = [&]<int MIN_BLOCKS, bool TRACE>() {
        int smem = g.dynamic_shared_memory();
        if constexpr (MIN_BLOCKS == 1) {
            smem = std::max(smem, 128 * 1024);
        }
        LaunchConfig<true, false> config(
            g.grid(), g.block(), smem, at::cuda::getCurrentCUDAStream(), 1);
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_tmem224_serial_qpair<C, MIN_BLOCKS, TRACE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CUDACHECK(cudaLaunchKernelEx(
            config,
            kernel_mxfp4_tmem224_serial_qpair<C, MIN_BLOCKS, TRACE>, g));
    };
    if (min_blocks_per_sm == 1) {
        if (trace_enabled) launch.template operator()<1, true>();
        else launch.template operator()<1, false>();
    } else {
        if (trace_enabled) launch.template operator()<2, true>();
        else launch.template operator()<2, false>();
    }
}

void dispatch_mxfp4_tmem224_output_evacuation(
    at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
    at::Tensor K, at::Tensor K_sc, at::Tensor K_sg,
    at::Tensor V, at::Tensor V_sc, at::Tensor O, at::Tensor LSE,
    at::Tensor Workspace, at::Tensor Trace, int external_mode,
    int min_blocks_per_sm) {
    check_mxfp4_tmem224_common(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V, V_sc, O, LSE);
    CHECK_INPUT(Workspace); CHECK_INPUT(Trace);
    TORCH_CHECK(Workspace.dtype() == at::ScalarType::Float,
                "output evacuation Workspace must be float32");
    TORCH_CHECK(Trace.dtype() == at::ScalarType::Long,
                "output evacuation Trace must be int64");
    TORCH_CHECK(external_mode >= TMEM224_EXTERNAL_FP32_SMEM &&
                external_mode <= TMEM224_EXTERNAL_FP16_SMEM,
                "output evacuation external_mode must be 0..4");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "output evacuation min_blocks_per_sm must be 1 or 2");
    const int batch = Q.size(0), heads = Q.size(1), seqlen = Q.size(2);
    using C = config_fp4pv_tmem224_nb64_dvo128;
    using G = globals_mxfp4_tmem224_pair<C>;
    constexpr int trace_stride = 16;
    const int grid_blocks = batch * heads * ((seqlen / 128 + 1) / 2);
    const bool trace_enabled = Trace.numel() != 0;
    TORCH_CHECK(!trace_enabled || Trace.numel() >= grid_blocks * trace_stride,
                "output evacuation Trace is too small");
    if (external_mode == TMEM224_EXTERNAL_FP32_GLOBAL) {
        TORCH_CHECK(Workspace.numel() >=
                    (unsigned long long)grid_blocks * 128 * 128,
                    "global output evacuation Workspace is too small");
    }
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        kittens::py::tensor_to_gl<typename G::v_gl>(V),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const float *>(Q_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const float *>(K_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<float *>(LSE.data_ptr()),
        nullptr,
        external_mode == TMEM224_EXTERNAL_FP32_GLOBAL
            ? reinterpret_cast<float *>(Workspace.data_ptr()) : nullptr,
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        batch, heads, seqlen, trace_enabled ? trace_stride : 0};

    auto launch = [&]<int MODE, int MIN_BLOCKS, bool TRACE>() {
        constexpr int external_bytes =
            MODE == TMEM224_EXTERNAL_FP32_SMEM ? 128 * 128 * 4 :
            (MODE == TMEM224_EXTERNAL_BF16_SMEM ||
             MODE == TMEM224_EXTERNAL_FP16_SMEM) ? 128 * 128 * 2 : 0;
        int smem = G::DYNAMIC_SHARED_MEMORY + external_bytes +
            (external_bytes != 0 ? 2048 : 0);
        if constexpr (MIN_BLOCKS == 1) {
            smem = std::max(smem, 128 * 1024);
        }
        TORCH_CHECK(smem <= MAX_SHARED_MEMORY - 4096,
                    "output evacuation dynamic SMEM exceeds device guard");
        LaunchConfig<true, false> config(
            g.grid(), g.block(), smem, at::cuda::getCurrentCUDAStream(), 1);
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_tmem224_output_evacuation<
                C, MODE, MIN_BLOCKS, TRACE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CUDACHECK(cudaLaunchKernelEx(
            config,
            kernel_mxfp4_tmem224_output_evacuation<
                C, MODE, MIN_BLOCKS, TRACE>, g));
    };
    auto choose_trace = [&]<int MODE, int MIN_BLOCKS>() {
        if (trace_enabled) launch.template operator()<MODE, MIN_BLOCKS, true>();
        else launch.template operator()<MODE, MIN_BLOCKS, false>();
    };
    auto choose_blocks = [&]<int MODE>() {
        if (min_blocks_per_sm == 1) {
            choose_trace.template operator()<MODE, 1>();
        } else {
            choose_trace.template operator()<MODE, 2>();
        }
    };
    switch (external_mode) {
        case TMEM224_EXTERNAL_FP32_SMEM:
            choose_blocks.template operator()<TMEM224_EXTERNAL_FP32_SMEM>();
            break;
        case TMEM224_EXTERNAL_FP32_GLOBAL:
            choose_blocks.template operator()<TMEM224_EXTERNAL_FP32_GLOBAL>();
            break;
        case TMEM224_EXTERNAL_FP32_REGISTER:
            choose_blocks.template operator()<TMEM224_EXTERNAL_FP32_REGISTER>();
            break;
        case TMEM224_EXTERNAL_BF16_SMEM:
            choose_blocks.template operator()<TMEM224_EXTERNAL_BF16_SMEM>();
            break;
        default:
            choose_blocks.template operator()<TMEM224_EXTERNAL_FP16_SMEM>();
            break;
    }
}
#endif

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
void dispatch_mxfp4_nb128_dvo128_qstage2_pipeline(
    at::Tensor Q_fp4, at::Tensor Q_sc, at::Tensor K_fp4, at::Tensor K_sc,
    at::Tensor Q_sg, at::Tensor K_sg, at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, at::Tensor Trace, int query_tiles,
    int heads, int seqlen) {
    CHECK_INPUT(Q_fp4); CHECK_INPUT(Q_sc); CHECK_INPUT(K_fp4); CHECK_INPUT(K_sc);
    CHECK_INPUT(Q_sg); CHECK_INPUT(K_sg); CHECK_INPUT(V_fp4); CHECK_INPUT(V_sc);
    CHECK_INPUT(O); CHECK_INPUT(LSE); CHECK_INPUT(Trace);
    TORCH_CHECK(Q_fp4.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
                K_fp4.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
                V_fp4.scalar_type() == at::ScalarType::Float4_e2m1fn_x2);
    TORCH_CHECK(Q_sc.scalar_type() == at::ScalarType::Byte &&
                K_sc.scalar_type() == at::ScalarType::Byte &&
                V_sc.scalar_type() == at::ScalarType::Byte);
    TORCH_CHECK(Q_sg.scalar_type() == at::ScalarType::Float &&
                K_sg.scalar_type() == at::ScalarType::Float &&
                Q_sg.dim() == 2 && K_sg.dim() == 2 && Q_sg.size(0) == 1 &&
                K_sg.size(0) == 1 && Q_sg.size(1) == heads &&
                K_sg.size(1) == heads && Q_sg.is_contiguous() && K_sg.is_contiguous());
    TORCH_CHECK(Q_fp4.is_contiguous() && Q_sc.is_contiguous() &&
                K_fp4.is_contiguous() && K_sc.is_contiguous() &&
                V_fp4.is_contiguous() && V_sc.is_contiguous() &&
                O.is_contiguous() && LSE.is_contiguous() && Trace.is_contiguous());
    TORCH_CHECK(Q_fp4.dim() == 4 && Q_fp4.size(0) == query_tiles &&
                Q_fp4.size(1) == heads && Q_fp4.size(2) == 128 && Q_fp4.size(3) == 96);
    TORCH_CHECK(K_fp4.dim() == 4 && K_fp4.size(0) == heads &&
                K_fp4.size(1) == query_tiles && K_fp4.size(2) == 128 && K_fp4.size(3) == 96);
    TORCH_CHECK(O.scalar_type() == at::ScalarType::BFloat16 && O.dim() == 4 &&
                O.size(0) == 1 && O.size(1) == seqlen && O.size(2) == heads && O.size(3) == 128);
    TORCH_CHECK(LSE.scalar_type() == at::ScalarType::Float && LSE.dim() == 4 &&
                LSE.size(0) == 1 && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen);
    TORCH_CHECK(Trace.scalar_type() == at::ScalarType::Long && Trace.dim() == 1 &&
                Trace.numel() >= ((query_tiles + 1) / 2) * heads * 32);
    TORCH_CHECK(query_tiles > 0 && heads > 0 && seqlen == query_tiles * 128);
    using C = config_fp4pv_nb128_dvo128_qstage2_pipeline;
    using G = globals_mxfp4_nb128_dvo128_qstage2_pipeline<C>;
    G g{reinterpret_cast<const uint8_t *>(Q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        Q_sg.data_ptr<float>(), K_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()), LSE.data_ptr<float>(),
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        query_tiles, heads, seqlen, 0.10408340855860843f};
    const int smem = g.dynamic_shared_memory();
    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_nb128_dvo128_qstage2_pipeline<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    LaunchConfig<true, false> lc(g.grid(), g.block(), smem,
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(lc,
        kernel_mxfp4_nb128_dvo128_qstage2_pipeline<C>, g));
}

void dispatch_mxfp4_nb128_dvo128_qstage2_two_n64(
    at::Tensor Q_fp4, at::Tensor Q_sc, at::Tensor K_fp4, at::Tensor K_sc,
    at::Tensor Q_sg, at::Tensor K_sg, at::Tensor V_fp4, at::Tensor V_sc,
    at::Tensor O, at::Tensor LSE, at::Tensor Trace, int query_tiles,
    int heads, int seqlen) {
    CHECK_INPUT(Q_fp4); CHECK_INPUT(Q_sc); CHECK_INPUT(K_fp4); CHECK_INPUT(K_sc);
    CHECK_INPUT(Q_sg); CHECK_INPUT(K_sg); CHECK_INPUT(V_fp4); CHECK_INPUT(V_sc);
    CHECK_INPUT(O); CHECK_INPUT(LSE); CHECK_INPUT(Trace);
    TORCH_CHECK(Q_fp4.is_contiguous() && Q_sc.is_contiguous() && K_fp4.is_contiguous() &&
                K_sc.is_contiguous() && V_fp4.is_contiguous() && V_sc.is_contiguous() &&
                O.is_contiguous() && LSE.is_contiguous() && Trace.is_contiguous());
    TORCH_CHECK(Q_fp4.dim() == 4 && Q_fp4.size(0) == query_tiles && Q_fp4.size(1) == heads &&
                Q_fp4.size(2) == 128 && Q_fp4.size(3) == 96);
    TORCH_CHECK(K_fp4.dim() == 4 && K_fp4.size(0) == heads && K_fp4.size(1) == query_tiles &&
                K_fp4.size(2) == 128 && K_fp4.size(3) == 96);
    TORCH_CHECK(query_tiles > 0 && heads > 0 && seqlen == query_tiles * 128);
    using C = config_fp4pv_nb128_dvo128_qstage2_two_n64;
    using G = globals_mxfp4_nb128_dvo128_qstage2_two_n64<C>;
    G g{reinterpret_cast<const uint8_t *>(Q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()), Q_sg.data_ptr<float>(),
        K_sg.data_ptr<float>(), reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()), reinterpret_cast<bf16 *>(O.data_ptr()),
        LSE.data_ptr<float>(), reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        query_tiles, heads, seqlen, 0.10408340855860843f};
    const int smem = g.dynamic_shared_memory();
    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_nb128_dvo128_qstage2_two_n64<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    LaunchConfig<true, false> lc(g.grid(), g.block(), smem,
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(lc, kernel_mxfp4_nb128_dvo128_qstage2_two_n64<C>, g));
}
#endif

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
void dispatch_mxfp4_nb128_dvo128_qstage2(
    at::Tensor Q_fp4, at::Tensor Q_sc, at::Tensor K_fp4, at::Tensor K_sc,
    at::Tensor Q_sg, at::Tensor K_sg, at::Tensor V_fp4, at::Tensor V_sc, at::Tensor O, at::Tensor LSE,
    at::Tensor Trace, int query_tiles, int heads, int seqlen) {
    CHECK_INPUT(Q_fp4); CHECK_INPUT(Q_sc); CHECK_INPUT(K_fp4); CHECK_INPUT(K_sc);
    CHECK_INPUT(Q_sg); CHECK_INPUT(K_sg);
    CHECK_INPUT(V_fp4); CHECK_INPUT(V_sc); CHECK_INPUT(O); CHECK_INPUT(LSE);
    CHECK_INPUT(Trace);
    TORCH_CHECK(Q_fp4.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
                K_fp4.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
                V_fp4.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
                "Q/K/V must be packed FP4");
    TORCH_CHECK(Q_sc.scalar_type() == at::ScalarType::Byte &&
                K_sc.scalar_type() == at::ScalarType::Byte &&
                V_sc.scalar_type() == at::ScalarType::Byte,
                "Q/K/V scales must be byte tensors");
    TORCH_CHECK(Q_sg.scalar_type() == at::ScalarType::Float && K_sg.scalar_type() == at::ScalarType::Float &&
                Q_sg.dim() == 2 && K_sg.dim() == 2 && Q_sg.size(0) == 1 && K_sg.size(0) == 1 &&
                Q_sg.size(1) == heads && K_sg.size(1) == heads && Q_sg.is_contiguous() && K_sg.is_contiguous(),
                "Q_sg/K_sg must be contiguous float32 [1, heads]");
    TORCH_CHECK(Q_fp4.is_contiguous() && Q_sc.is_contiguous() &&
                K_fp4.is_contiguous() && K_sc.is_contiguous() &&
                V_fp4.is_contiguous() && V_sc.is_contiguous(),
                "qstage2 inputs must be contiguous");
    TORCH_CHECK(query_tiles > 0 && heads > 0 && seqlen == query_tiles * 128,
                "qstage2 dimensions must be positive and seqlen=query_tiles*128");
    TORCH_CHECK(Q_fp4.numel() >= (int64_t)query_tiles * heads * 128 * 96,
                "Q_fp4 is too small for two-query ownership records");
    TORCH_CHECK(Q_sc.numel() >= (int64_t)query_tiles * heads * 3 * 32 * 16,
                "Q_sc is too small for two-query ownership records");
    TORCH_CHECK(K_fp4.dim() == 4 && K_fp4.size(0) == heads &&
                K_fp4.size(1) == query_tiles && K_fp4.size(2) == 128 && K_fp4.size(3) == 96,
                "K_fp4 must be [heads, key_tiles, 128, 96] with key_tiles=query_tiles");
    TORCH_CHECK(K_sc.numel() >= (int64_t)heads * query_tiles * 3 * 32 * 16,
                "K_sc must contain one record per head/key tile");
    TORCH_CHECK(Q_fp4.dim() == 4 && Q_fp4.size(0) == query_tiles &&
                Q_fp4.size(1) == heads && Q_fp4.size(2) == 128 && Q_fp4.size(3) == 96,
                "Q_fp4 must be [query_tiles, heads, 128, 96]");
    TORCH_CHECK(Q_sc.dim() == 5 && Q_sc.size(0) == query_tiles &&
                Q_sc.size(1) == heads && Q_sc.size(2) == 3 &&
                Q_sc.size(3) == 32 && Q_sc.size(4) == 16,
                "Q_sc must be [query_tiles, heads, 3, 32, 16]");
    TORCH_CHECK(K_sc.dim() == 5 && K_sc.size(0) == heads &&
                K_sc.size(1) == query_tiles && K_sc.size(2) == 3 &&
                K_sc.size(3) == 32 && K_sc.size(4) == 16,
                "K_sc must be [heads, key_tiles, 3, 32, 16]");
    TORCH_CHECK(V_fp4.dim() == 4 && V_fp4.size(0) == 1 &&
                V_fp4.size(1) == heads && V_fp4.size(2) == 128 &&
                V_fp4.size(3) == seqlen / 2,
                "V_fp4 must be [1, heads, 128, seqlen/2 packed]");
    TORCH_CHECK(V_sc.dim() == 5 && V_sc.size(0) == heads &&
                V_sc.size(1) == 1 && V_sc.size(2) == query_tiles &&
                V_sc.size(3) == 32 && V_sc.size(4) == 16,
                "V_sc must be [heads, 1, key_tiles, 32, 16] for full DVO128");
    TORCH_CHECK(O.scalar_type() == at::ScalarType::BFloat16 && O.dim() == 4 &&
                O.size(0) == 1 && O.size(1) == seqlen && O.size(2) == heads &&
                O.size(3) == 128,
                "O must be contiguous [1, seqlen, heads, 128] BF16");
    TORCH_CHECK(LSE.scalar_type() == at::ScalarType::Float && LSE.dim() == 4 &&
                LSE.size(0) == 1 && LSE.size(1) == heads && LSE.size(2) == 1 &&
                LSE.size(3) == seqlen,
                "LSE must be contiguous [1, heads, 1, seqlen] FP32");
    TORCH_CHECK(Trace.scalar_type() == at::ScalarType::Long && Trace.dim() == 1,
                "Trace must be a contiguous int64 vector");
    TORCH_CHECK(Trace.numel() >= ((query_tiles + 1) / 2) * heads * 32,
                "Trace is too small for qstage2 grid records");
    using C = config_fp4pv_nb128_dvo128_qstage2;
    using G = globals_mxfp4_nb128_dvo128_qstage2<C>;
    G g{reinterpret_cast<const uint8_t *>(Q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const float *>(Q_sg.data_ptr<float>()),
        reinterpret_cast<const float *>(K_sg.data_ptr<float>()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<float *>(LSE.data_ptr<float>()),
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        query_tiles, heads, seqlen, 0.10408340855860843f};
    const int smem = g.dynamic_shared_memory();
    int device = -1, max_smem = 0;
    CUDACHECK(cudaGetDevice(&device));
    CUDACHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    TORCH_CHECK(smem <= max_smem,
                "qstage2 dynamic SMEM exceeds device opt-in limit: ", smem,
                " > ", max_smem);
    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_nb128_dvo128_qstage2<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), smem, at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config,
        kernel_mxfp4_nb128_dvo128_qstage2<C>, g));
}
#endif

void reset_mxfp4_forward_pchain_stamps() {
#if TK_FA4_ENABLE_MXFP4_FWD_PCHAIN_STAMPS
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_forward_pchain_stamps));
    CUDACHECK(cudaMemset(
        ptr, 0, FP4PV_FWD_PCHAIN_STAMP_COUNT * sizeof(unsigned long long)));
#endif
}

std::vector<unsigned long long> read_mxfp4_forward_pchain_stamps() {
#if TK_FA4_ENABLE_MXFP4_FWD_PCHAIN_STAMPS
    std::vector<unsigned long long> out(FP4PV_FWD_PCHAIN_STAMP_COUNT, 0);
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_forward_pchain_stamps));
    CUDACHECK(cudaMemcpy(
        out.data(),
        ptr,
        FP4PV_FWD_PCHAIN_STAMP_COUNT * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    return out;
#else
    return {};
#endif
}

void reset_mxfp4_forward_timeline() {
#if TK_FA4_ENABLE_MXFP4_FWD_TIMELINE
    void *records_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&records_ptr, fp4pv_forward_timeline));
    CUDACHECK(cudaMemset(
        records_ptr,
        0,
        FP4PV_FWD_TIMELINE_MAX_RECORDS *
            FP4PV_FWD_TIMELINE_RECORD_U64 *
            sizeof(unsigned long long)));
#endif
}

std::vector<unsigned long long> read_mxfp4_forward_timeline() {
#if TK_FA4_ENABLE_MXFP4_FWD_TIMELINE
    std::vector<unsigned long long> raw(
        FP4PV_FWD_TIMELINE_MAX_RECORDS * FP4PV_FWD_TIMELINE_RECORD_U64,
        0);
    void *records_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&records_ptr, fp4pv_forward_timeline));
    CUDACHECK(cudaMemcpy(
        raw.data(),
        records_ptr,
        raw.size() * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    unsigned long long count = 0;
    for (int rec = 0; rec < FP4PV_FWD_TIMELINE_MAX_RECORDS; ++rec) {
        const int base = rec * FP4PV_FWD_TIMELINE_RECORD_U64;
        if (raw[base + 1] != 0ULL) {
            ++count;
        }
    }
    std::vector<unsigned long long> out(
        1 + count * FP4PV_FWD_TIMELINE_RECORD_U64,
        0);
    out[0] = count;
    unsigned long long write_rec = 0;
    for (int rec = 0; rec < FP4PV_FWD_TIMELINE_MAX_RECORDS; ++rec) {
        const int base = rec * FP4PV_FWD_TIMELINE_RECORD_U64;
        if (raw[base + 1] == 0ULL) {
            continue;
        }
        const unsigned long long dst_base =
            1ULL + write_rec * FP4PV_FWD_TIMELINE_RECORD_U64;
        for (int word = 0; word < FP4PV_FWD_TIMELINE_RECORD_U64; ++word) {
            out[dst_base + word] = raw[base + word];
        }
        ++write_rec;
    }
    return out;
#else
    return {};
#endif
}

#if TK_FA4_REAL_FWD_TIMELINE_PROBE_BUILD
void reset_mxfp4_real_forward_timeline(int target_bid) {
    TORCH_CHECK(target_bid >= 0, "target_bid must be non-negative");
    void *records_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &records_ptr, fp4pv_real_fwd_timeline_records));
    CUDACHECK(cudaMemset(
        records_ptr,
        0,
        FP4PV_RF_MAX_TILES * FP4PV_RF_EVENT_COUNT *
            FP4PV_RF_RECORD_U64 * sizeof(unsigned long long)));
    CUDACHECK(cudaMemcpyToSymbol(
        fp4pv_real_fwd_timeline_target_bid,
        &target_bid,
        sizeof(target_bid)));
}

std::vector<unsigned long long> read_mxfp4_real_forward_timeline() {
    constexpr int header_words = 5;
    constexpr int raw_words =
        FP4PV_RF_MAX_TILES * FP4PV_RF_EVENT_COUNT * FP4PV_RF_RECORD_U64;
    std::vector<unsigned long long> out(header_words + raw_words, 0);
    void *records_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &records_ptr, fp4pv_real_fwd_timeline_records));
    CUDACHECK(cudaMemcpy(
        out.data() + header_words,
        records_ptr,
        raw_words * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    int target_bid = -1;
    CUDACHECK(cudaMemcpyFromSymbol(
        &target_bid,
        fp4pv_real_fwd_timeline_target_bid,
        sizeof(target_bid)));
    unsigned long long valid_records = 0;
    for (int record = 0;
         record < FP4PV_RF_MAX_TILES * FP4PV_RF_EVENT_COUNT;
         ++record) {
        if (out[header_words + record * FP4PV_RF_RECORD_U64] != 0ULL) {
            ++valid_records;
        }
    }
    out[0] = valid_records;
    out[1] = FP4PV_RF_MAX_TILES;
    out[2] = FP4PV_RF_EVENT_COUNT;
    out[3] = FP4PV_RF_RECORD_U64;
    out[4] = static_cast<unsigned long long>(target_bid);
    return out;
}

void reset_mxfp4_real_forward_readiness(
    int target_bid, int target_task_bid) {
    TORCH_CHECK(target_bid >= 0, "target_bid must be non-negative");
    TORCH_CHECK(
        target_task_bid >= 0,
        "target_task_bid must be non-negative");
    void *record_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &record_ptr, fp4pv_real_fwd_readiness_record));
    CUDACHECK(cudaMemset(
        record_ptr,
        0,
        FP4PV_RF_READINESS_RECORD_U64 * sizeof(unsigned long long)));
    CUDACHECK(cudaMemcpyToSymbol(
        fp4pv_real_fwd_readiness_target_bid,
        &target_bid,
        sizeof(target_bid)));
    CUDACHECK(cudaMemcpyToSymbol(
        fp4pv_real_fwd_readiness_target_task_bid,
        &target_task_bid,
        sizeof(target_task_bid)));
}

std::vector<unsigned long long> read_mxfp4_real_forward_readiness() {
    std::vector<unsigned long long> out(
        FP4PV_RF_READINESS_RECORD_U64, 0);
    void *record_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &record_ptr, fp4pv_real_fwd_readiness_record));
    CUDACHECK(cudaMemcpy(
        out.data(),
        record_ptr,
        FP4PV_RF_READINESS_RECORD_U64 * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    return out;
}

void reset_mxfp4_real_forward_corr_pv_cycles(
    int target_bid, int target_task_bid) {
    TORCH_CHECK(target_bid >= 0, "target_bid must be non-negative");
    TORCH_CHECK(
        target_task_bid >= 0,
        "target_task_bid must be non-negative");
    void *records_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &records_ptr, fp4pv_real_fwd_corr_pv_cycle_records));
    CUDACHECK(cudaMemset(
        records_ptr,
        0,
        FP4PV_RF_CP_SAMPLE_SLOTS * FP4PV_RF_CP_EVENT_COUNT *
            FP4PV_RF_CP_RECORD_U64 * sizeof(unsigned long long)));
    CUDACHECK(cudaMemcpyToSymbol(
        fp4pv_real_fwd_corr_pv_cycle_target_bid,
        &target_bid,
        sizeof(target_bid)));
    CUDACHECK(cudaMemcpyToSymbol(
        fp4pv_real_fwd_corr_pv_cycle_target_task_bid,
        &target_task_bid,
        sizeof(target_task_bid)));
}

std::vector<unsigned long long> read_mxfp4_real_forward_corr_pv_cycles() {
    constexpr int header_words = 7;
    constexpr int raw_words =
        FP4PV_RF_CP_SAMPLE_SLOTS * FP4PV_RF_CP_EVENT_COUNT *
        FP4PV_RF_CP_RECORD_U64;
    std::vector<unsigned long long> out(header_words + raw_words, 0);
    void *records_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &records_ptr, fp4pv_real_fwd_corr_pv_cycle_records));
    CUDACHECK(cudaMemcpy(
        out.data() + header_words,
        records_ptr,
        raw_words * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    int target_bid = -1;
    int target_task_bid = -1;
    CUDACHECK(cudaMemcpyFromSymbol(
        &target_bid,
        fp4pv_real_fwd_corr_pv_cycle_target_bid,
        sizeof(target_bid)));
    CUDACHECK(cudaMemcpyFromSymbol(
        &target_task_bid,
        fp4pv_real_fwd_corr_pv_cycle_target_task_bid,
        sizeof(target_task_bid)));
    unsigned long long valid_records = 0;
    for (int record = 0;
         record < FP4PV_RF_CP_SAMPLE_SLOTS * FP4PV_RF_CP_EVENT_COUNT;
         ++record) {
        if (out[header_words + record * FP4PV_RF_CP_RECORD_U64] != 0ULL) {
            ++valid_records;
        }
    }
    out[0] = valid_records;
    out[1] = FP4PV_RF_CP_SAMPLE_SLOTS;
    out[2] = FP4PV_RF_CP_EVENT_COUNT;
    out[3] = FP4PV_RF_CP_RECORD_U64;
    out[4] = static_cast<unsigned long long>(target_bid);
    out[5] = static_cast<unsigned long long>(target_task_bid);
    out[6] = 1ULL; // schema version
    return out;
}
#endif

void reset_mxfp4_forward_policy126_counters() {
    void *counters_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &counters_ptr,
        fp4pv_forward_policy126_counters));
    CUDACHECK(cudaMemset(
        counters_ptr,
        0,
        FP4PV_FWD_POLICY126_COUNTER_COUNT * sizeof(unsigned long long)));
}

std::vector<unsigned long long> read_mxfp4_forward_policy126_counters() {
    std::vector<unsigned long long> out(
        FP4PV_FWD_POLICY126_COUNTER_COUNT,
        0);
    void *counters_ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(
        &counters_ptr,
        fp4pv_forward_policy126_counters));
    CUDACHECK(cudaMemcpy(
        out.data(),
        counters_ptr,
        out.size() * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    return out;
}

void dispatch_mxfp4_control_plane_diag(at::Tensor Debug, int mode) {
    CHECK_INPUT(Debug);
    TORCH_CHECK(Debug.dtype() == at::ScalarType::Int, "Debug must be int32");
    TORCH_CHECK(Debug.numel() >= 32, "Debug must have at least 32 int32 elements");
    TORCH_CHECK(mode >= 0 && mode <= 18,
                "mode must be 0(one-role sentinel), 1(one-role no-publish timeout), "
                "2(eight-role task), 3(eight-role sentinel/drain), 4(eight-role missing done), "
                "5(eight-role done-only without ack), 6(eight-role partial done), "
                "7(eight-role ack-only without done), 8(early sentinel before ready), "
                "9(missing role ready timeout), 10(missing role drain timeout), "
                "11(late role ready after scheduler timeout), "
                "12(late role ready during drain after scheduler timeout), "
                "13(drain ack-only without done), "
                "14(drain partial-domain completion), "
                "15(early normal task before ready), "
                "16(two-task reuse/reinit), "
                "17(two-task post-reuse missing done), "
                "or 18(two-task post-reuse drain sentinel)");

    kernel_mxfp4_control_plane_diag<<<dim3(1), dim3(128), 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<int32_t *>(Debug.data_ptr()),
        mode);
    CUDACHECK(cudaGetLastError());
}

void dispatch_pv_from_p(at::Tensor P_fp4,
                        at::Tensor P_sc_prepared,
                        at::Tensor V_fp4,
                        at::Tensor V_sc_prepared,
                        at::Tensor O) {
    CHECK_INPUT(O);
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(O.dim() == 4, "O must have shape (batch, seqlen, heads, dvo)");

    const int batch = P_fp4.size(0);
    const int heads = P_fp4.size(1);
    const int seqlen = P_fp4.size(2);

    check_fp4pv_p_tensor(P_fp4, batch, heads, seqlen);
    check_fp4pv_p_scale_tensor(P_sc_prepared, batch, heads, seqlen);
    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_v_scale_tensor(V_sc_prepared, batch, heads, seqlen, 128);
    TORCH_CHECK(seqlen % 256 == 0, "pv_from_p currently requires sequence length divisible by 256");

    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen && O.size(2) == heads && O.size(3) == 128,
                "O must have shape (batch, seqlen, heads, 128)");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_pv_from_p<C>;
    G g{
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc_prepared.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        0,
        0,
        0,
        0,
        0,
        batch,
        heads,
        seqlen
    };
    CUDACHECK(cudaFuncSetAttribute(kernel_pv_from_p<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_pv_from_p<C>, g));
}

void dispatch_mxfp4_pv_from_p_chunked_debug(at::Tensor P_fp4,
                                            at::Tensor P_sc,
                                            at::Tensor V_fp4,
                                            at::Tensor V_sc,
                                            at::Tensor O,
                                            int debug_stage) {
    CHECK_INPUT(P_fp4);
    CHECK_INPUT(P_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(P_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "P_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(P_sc.dtype() == at::ScalarType::Byte, "P_sc must be uint8 E8M0 scales");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte, "V_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(P_fp4.dim() == 4, "P_fp4 must have shape (batch, heads, seqlen, p_cols/2)");
    TORCH_CHECK(P_sc.dim() == 6, "P_sc must have shape (batch, heads, seqlen/128, p_cols/128, 32, 16)");
    TORCH_CHECK(V_fp4.dim() == 4, "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
    TORCH_CHECK(V_sc.dim() == 6, "V_sc must have shape (batch, heads, scale_rows, seqlen/128, 32, 16)");
    TORCH_CHECK(O.dim() == 4, "O must have shape (batch, seqlen, heads, 128)");

    const int batch = P_fp4.size(0);
    const int heads = P_fp4.size(1);
    const int seqlen = P_fp4.size(2);
    const int p_cols = P_fp4.size(3) * 2;
    const int v_scale_rows = V_sc.size(2);

    TORCH_CHECK(seqlen % 256 == 0, "seqlen must be divisible by 256");
    TORCH_CHECK(seqlen / 128 <= 512,
                "mxfp4_pv_from_p_chunked_debug currently supports seqlen <= 65536");
    TORCH_CHECK(p_cols > 0 && p_cols % 128 == 0, "P K chunk width must be a positive multiple of 128");
    TORCH_CHECK(debug_stage >= 0 && debug_stage <= 28, "debug_stage must be in [0, 28]");
    TORCH_CHECK(P_sc.size(0) == batch && P_sc.size(1) == heads &&
                P_sc.size(2) == seqlen / 128 && P_sc.size(3) == p_cols / 128 &&
                P_sc.size(4) == 32 && P_sc.size(5) == 16,
                "P_sc must have shape (batch, heads, seqlen/128, p_cols/128, 32, 16)");
    TORCH_CHECK(V_fp4.size(0) == batch && V_fp4.size(1) == heads &&
                V_fp4.size(2) == 128 && V_fp4.size(3) == seqlen / 2,
                "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
    TORCH_CHECK(v_scale_rows >= 1 && V_sc.size(0) == batch && V_sc.size(1) == heads &&
                V_sc.size(3) == seqlen / 128 && V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "V_sc must have shape (batch, heads, scale_rows>=1, seqlen/128, 32, 16)");
    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen && O.size(2) == heads && O.size(3) == 128,
                "O must have shape (batch, seqlen, heads, 128)");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_mxfp4_pv_from_p_chunked_debug<C>;
    G g{
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        batch,
        heads,
        seqlen,
        p_cols,
        v_scale_rows,
        debug_stage
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_pv_from_p_chunked_debug<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_pv_from_p_chunked_debug<C>, g));
}

#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
void dispatch_mxfp4_pv_dvo64_residency_shell(at::Tensor P_fp4,
                                             at::Tensor P_sc,
                                             at::Tensor V_fp4,
                                             at::Tensor V_sc,
                                             at::Tensor O,
                                             at::Tensor Trace,
                                             int stage,
                                             int min_blocks_per_sm,
                                             int grid_blocks) {
    CHECK_INPUT(P_fp4);
    CHECK_INPUT(P_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(O);
    CHECK_INPUT(Trace);
    TORCH_CHECK(P_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "P_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(P_sc.dtype() == at::ScalarType::Byte,
                "P_sc must be uint8 E8M0 scales");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte,
                "V_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16,
                "O must be bfloat16");
    TORCH_CHECK(Trace.dtype() == at::ScalarType::Long,
                "Trace must be int64");
    TORCH_CHECK(P_fp4.dim() == 4 && P_sc.dim() == 6 &&
                V_fp4.dim() == 4 && V_sc.dim() == 6 && O.dim() == 4,
                "DVO64 residency shell received an invalid tensor rank");

    const int batch = P_fp4.size(0);
    const int heads = P_fp4.size(1);
    const int seqlen = P_fp4.size(2);
    const int p_cols = P_fp4.size(3) * 2;
    const int v_scale_rows = V_sc.size(2);
    TORCH_CHECK(seqlen == 128 && p_cols == 128,
                "DVO64 residency shell requires P shape (...,128,64 packed)");
    TORCH_CHECK(P_sc.size(0) == batch && P_sc.size(1) == heads &&
                P_sc.size(2) == 1 && P_sc.size(3) == 1 &&
                P_sc.size(4) == 32 && P_sc.size(5) == 16,
                "P_sc must have shape (batch,heads,1,1,32,16)");
    TORCH_CHECK(V_fp4.size(0) == batch && V_fp4.size(1) == heads &&
                V_fp4.size(2) == 64 && V_fp4.size(3) == 64,
                "V_fp4 must have shape (batch,heads,64,64 packed)");
    TORCH_CHECK(v_scale_rows >= 1 && V_sc.size(0) == batch &&
                V_sc.size(1) == heads && V_sc.size(3) == 1 &&
                V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "V_sc must have shape (batch,heads,scale_rows>=1,1,32,16)");
    TORCH_CHECK(O.size(0) == batch && O.size(1) == 128 &&
                O.size(2) == heads && O.size(3) == 64,
                "O must have shape (batch,128,heads,64)");
    TORCH_CHECK(stage >= 0 && stage <= 5, "stage must be in [0,5]");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "min_blocks_per_sm must be 1 or 2");
    TORCH_CHECK(grid_blocks > 0 && grid_blocks <= batch * heads,
                "grid_blocks must be in [1,batch*heads]");
    TORCH_CHECK(Trace.numel() >= grid_blocks * 16,
                "Trace must contain at least grid_blocks*16 int64 values");

    using C = config_fp4pv_dvo64_pv_residency_shell;
    using G = globals_mxfp4_pv_dvo64_residency_shell<C>;
    G g{
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        batch,
        heads,
        seqlen,
        p_cols,
        v_scale_rows,
        stage,
        grid_blocks
    };

    if (min_blocks_per_sm == 1) {
        constexpr int ONE_RESIDENT_CONTROL_SMEM = 128 * 1024;
        const int launch_smem = g.dynamic_shared_memory() > ONE_RESIDENT_CONTROL_SMEM
            ? g.dynamic_shared_memory()
            : ONE_RESIDENT_CONTROL_SMEM;
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_pv_dvo64_residency_shell<C, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            launch_smem));
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), launch_smem,
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config, kernel_mxfp4_pv_dvo64_residency_shell<C, 1>, g));
    } else {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_pv_dvo64_residency_shell<C, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.dynamic_shared_memory()));
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), g.dynamic_shared_memory(),
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config, kernel_mxfp4_pv_dvo64_residency_shell<C, 2>, g));
    }
}

void dispatch_mxfp4_nb64_dvo128_residency_shell(at::Tensor Q_fp4,
                                                at::Tensor Q_sc,
                                                at::Tensor K_fp4,
                                                at::Tensor K_sc,
                                                at::Tensor P_fp4,
                                                at::Tensor P_sc,
                                                at::Tensor V_fp4,
                                                at::Tensor V_sc,
                                                at::Tensor O,
                                                at::Tensor Trace,
                                                int stage,
                                                int min_blocks_per_sm,
                                                int grid_blocks) {
    CHECK_INPUT(Q_fp4);
    CHECK_INPUT(Q_sc);
    CHECK_INPUT(K_fp4);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(P_fp4);
    CHECK_INPUT(P_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(O);
    CHECK_INPUT(Trace);
    TORCH_CHECK(Q_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                P_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Q/K/P/V payloads must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Byte &&
                K_sc.dtype() == at::ScalarType::Byte &&
                P_sc.dtype() == at::ScalarType::Byte &&
                V_sc.dtype() == at::ScalarType::Byte,
                "Q/K scales must be raw E4M3 bytes and P/V scales raw E8M0 bytes");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16,
                "O must be bfloat16");
    TORCH_CHECK(Trace.dtype() == at::ScalarType::Long,
                "Trace must be int64");
    TORCH_CHECK(Q_fp4.dim() == 4 && K_fp4.dim() == 4 &&
                P_fp4.dim() == 4 && P_sc.dim() == 6 &&
                V_fp4.dim() == 4 && V_sc.dim() == 6 && O.dim() == 4,
                "Nb64/DVO128 residency shell received an invalid tensor rank");

    const int batch = Q_fp4.size(0);
    const int heads = Q_fp4.size(1);
    constexpr int seqlen = 128;
    constexpr int p_cols = 64;
    const int v_scale_rows = V_sc.size(2);
    TORCH_CHECK(batch == 1 && heads > 0,
                "Nb64/DVO128 residency shell currently requires batch=1");
    TORCH_CHECK(Q_fp4.size(2) == 128 && Q_fp4.size(3) == 96,
                "Q_fp4 must have shape (1,heads,128,96 packed)");
    TORCH_CHECK(K_fp4.size(0) == batch && K_fp4.size(1) == heads &&
                K_fp4.size(2) == 64 && K_fp4.size(3) == 96,
                "K_fp4 must have shape (1,heads,64,96 packed physical)");
    TORCH_CHECK(Q_sc.numel() == heads * 3 * 32 * 16 &&
                K_sc.numel() == heads * 3 * 32 * 16,
                "Q_sc/K_sc must each contain heads*3*32*16 raw bytes");
    TORCH_CHECK(P_fp4.size(0) == batch && P_fp4.size(1) == heads &&
                P_fp4.size(2) == 128 && P_fp4.size(3) == 32,
                "P_fp4 must have shape (1,heads,128,32 packed)");
    TORCH_CHECK(P_sc.size(0) == batch && P_sc.size(1) == heads &&
                P_sc.size(2) == 1 && P_sc.size(3) == 1 &&
                P_sc.size(4) == 32 && P_sc.size(5) == 16,
                "P_sc must have shape (1,heads,1,1,32,16)");
    TORCH_CHECK(V_fp4.size(0) == batch && V_fp4.size(1) == heads &&
                V_fp4.size(2) == 128 && V_fp4.size(3) == 64,
                "V_fp4 must have shape (1,heads,128,64 packed backing)");
    TORCH_CHECK(v_scale_rows >= 1 && V_sc.size(0) == batch &&
                V_sc.size(1) == heads && V_sc.size(3) == 1 &&
                V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "V_sc must have shape (1,heads,scale_rows>=1,1,32,16)");
    TORCH_CHECK(O.size(0) == batch && O.size(1) == 128 &&
                O.size(2) == heads && O.size(3) == 128,
                "O must have shape (1,128,heads,128)");
    TORCH_CHECK(stage >= 0 && stage <= 4, "stage must be in [0,4]");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "min_blocks_per_sm must be 1 or 2");
    TORCH_CHECK(grid_blocks > 0 && grid_blocks <= batch * heads,
                "grid_blocks must be in [1,batch*heads]");
    TORCH_CHECK(Trace.numel() >= grid_blocks * 16,
                "Trace must contain at least grid_blocks*16 int64 values");

    using C = config_fp4pv_nb64_dvo128_residency_shell;
    using G = globals_mxfp4_nb64_dvo128_residency_shell<C>;
    G g{
        reinterpret_cast<const uint8_t *>(Q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        batch,
        heads,
        seqlen,
        p_cols,
        v_scale_rows,
        stage,
        grid_blocks
    };

    if (min_blocks_per_sm == 1) {
        constexpr int ONE_RESIDENT_CONTROL_SMEM = 128 * 1024;
        const int launch_smem = g.dynamic_shared_memory() > ONE_RESIDENT_CONTROL_SMEM
            ? g.dynamic_shared_memory()
            : ONE_RESIDENT_CONTROL_SMEM;
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_nb64_dvo128_residency_shell<C, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            launch_smem));
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), launch_smem,
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config, kernel_mxfp4_nb64_dvo128_residency_shell<C, 1>, g));
    } else {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_nb64_dvo128_residency_shell<C, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.dynamic_shared_memory()));
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), g.dynamic_shared_memory(),
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config, kernel_mxfp4_nb64_dvo128_residency_shell<C, 2>, g));
    }
}

void dispatch_mxfp4_nb64_dual_dvo64_residency_shell(at::Tensor Q_fp4,
                                                     at::Tensor Q_sc,
                                                     at::Tensor K_fp4,
                                                     at::Tensor K_sc,
                                                     at::Tensor P_fp4,
                                                     at::Tensor P_sc,
                                                     at::Tensor V_fp4,
                                                     at::Tensor V_sc,
                                                     at::Tensor O,
                                                     at::Tensor Trace,
                                                     int stage,
                                                     int min_blocks_per_sm,
                                                     int grid_blocks) {
    CHECK_INPUT(Q_fp4);
    CHECK_INPUT(Q_sc);
    CHECK_INPUT(K_fp4);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(P_fp4);
    CHECK_INPUT(P_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(O);
    CHECK_INPUT(Trace);
    TORCH_CHECK(Q_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                P_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Q/K/P/V payloads must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Byte &&
                K_sc.dtype() == at::ScalarType::Byte &&
                P_sc.dtype() == at::ScalarType::Byte &&
                V_sc.dtype() == at::ScalarType::Byte,
                "Q/K/P/V scales must be raw byte records");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16,
                "O must be bfloat16");
    TORCH_CHECK(Trace.dtype() == at::ScalarType::Long,
                "Trace must be int64");

    const int batch = Q_fp4.size(0);
    const int heads = Q_fp4.size(1);
    constexpr int seqlen = 128;
    constexpr int p_cols = 64;
    const int v_scale_rows = V_sc.size(2);
    TORCH_CHECK(batch == 1 && heads > 0,
                "dual-DVO64 residency shell currently requires batch=1");
    TORCH_CHECK(Q_fp4.dim() == 4 && Q_fp4.size(0) == 1 &&
                Q_fp4.size(1) == heads && Q_fp4.size(2) == 128 &&
                Q_fp4.size(3) == 96,
                "Q_fp4 must have shape (1,heads,128,96 packed)");
    TORCH_CHECK(K_fp4.dim() == 4 && K_fp4.size(0) == 1 &&
                K_fp4.size(1) == heads && K_fp4.size(2) == 64 &&
                K_fp4.size(3) == 96,
                "K_fp4 must have shape (1,heads,64,96 packed physical)");
    TORCH_CHECK(Q_sc.numel() == heads * 3 * 32 * 16 &&
                K_sc.numel() == heads * 3 * 32 * 16,
                "Q_sc/K_sc must each contain heads*3*32*16 raw bytes");
    TORCH_CHECK(P_fp4.dim() == 4 && P_fp4.size(0) == 1 &&
                P_fp4.size(1) == heads && P_fp4.size(2) == 128 &&
                P_fp4.size(3) == 32,
                "P_fp4 must have shape (1,heads,128,32 packed)");
    TORCH_CHECK(P_sc.dim() == 6 && P_sc.size(0) == 1 &&
                P_sc.size(1) == heads && P_sc.size(2) == 1 &&
                P_sc.size(3) == 1 && P_sc.size(4) == 32 && P_sc.size(5) == 16,
                "P_sc must have shape (1,heads,1,1,32,16)");
    TORCH_CHECK(V_fp4.dim() == 4 && V_fp4.size(0) == 1 &&
                V_fp4.size(1) == heads && V_fp4.size(2) == 128 &&
                V_fp4.size(3) == 64,
                "V_fp4 must have shape (1,heads,128,64 packed backing)");
    TORCH_CHECK(V_sc.dim() == 6 && V_sc.size(0) == 1 &&
                V_sc.size(1) == heads && V_sc.size(2) == 2 &&
                V_sc.size(3) == 1 && V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "V_sc must have the two-record shape (1,heads,2,1,32,16)");
    TORCH_CHECK(O.dim() == 4 && O.size(0) == 1 && O.size(1) == 128 &&
                O.size(2) == heads && O.size(3) == 128,
                "O must have shape (1,128,heads,128)");
    TORCH_CHECK(stage >= 0 && stage <= 3, "stage must be in [0,3]");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "min_blocks_per_sm must be 1 or 2");
    TORCH_CHECK(grid_blocks > 0 && grid_blocks <= batch * heads,
                "grid_blocks must be in [1,batch*heads]");
    TORCH_CHECK(Trace.numel() >= grid_blocks * 20,
                "Trace must contain at least grid_blocks*20 int64 values");

    using C = config_fp4pv_nb64_dual_dvo64_residency_shell;
    using G = globals_mxfp4_nb64_dvo128_residency_shell<C>;
    G g{
        reinterpret_cast<const uint8_t *>(Q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        batch,
        heads,
        seqlen,
        p_cols,
        v_scale_rows,
        stage,
        grid_blocks
    };

    if (min_blocks_per_sm == 1) {
        constexpr int ONE_RESIDENT_CONTROL_SMEM = 128 * 1024;
        const int launch_smem = g.dynamic_shared_memory() > ONE_RESIDENT_CONTROL_SMEM
            ? g.dynamic_shared_memory()
            : ONE_RESIDENT_CONTROL_SMEM;
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_nb64_dual_dvo64_residency_shell<C, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            launch_smem));
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), launch_smem,
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config,
            kernel_mxfp4_nb64_dual_dvo64_residency_shell<C, 1>, g));
    } else {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_nb64_dual_dvo64_residency_shell<C, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.dynamic_shared_memory()));
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), g.dynamic_shared_memory(),
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config,
            kernel_mxfp4_nb64_dual_dvo64_residency_shell<C, 2>, g));
    }
}

void dispatch_mxfp4_nb128_dual_dvo64_attention_shell(
    at::Tensor Q_fp4,
    at::Tensor Q_sc,
    at::Tensor K_fp4,
    at::Tensor K_sc,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    at::Tensor P_debug,
    at::Tensor P_sc_debug,
    at::Tensor O,
    at::Tensor LSE,
    at::Tensor Row_sum,
    at::Tensor Trace,
    float softmax_scale_log2,
    int stage,
    int min_blocks_per_sm,
    int grid_blocks) {
    CHECK_INPUT(Q_fp4);
    CHECK_INPUT(Q_sc);
    CHECK_INPUT(K_fp4);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(P_debug);
    CHECK_INPUT(P_sc_debug);
    CHECK_INPUT(O);
    CHECK_INPUT(LSE);
    CHECK_INPUT(Row_sum);
    CHECK_INPUT(Trace);
    TORCH_CHECK(Q_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Q/K/V payloads must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Byte &&
                K_sc.dtype() == at::ScalarType::Byte &&
                V_sc.dtype() == at::ScalarType::Byte &&
                P_debug.dtype() == at::ScalarType::Byte &&
                P_sc_debug.dtype() == at::ScalarType::Byte,
                "Q/K/V scales and P debug outputs must be raw bytes");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16,
                "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float &&
                Row_sum.dtype() == at::ScalarType::Float,
                "LSE and Row_sum must be float32");
    TORCH_CHECK(Trace.dtype() == at::ScalarType::Long,
                "Trace must be int64");

    const int batch = Q_fp4.size(0);
    const int heads = Q_fp4.size(1);
    constexpr int seqlen = 128;
    const int v_scale_rows = V_sc.size(2);
    TORCH_CHECK(batch == 1 && heads > 0,
                "Nb128 dual-DVO64 attention shell requires batch=1");
    TORCH_CHECK(Q_fp4.dim() == 4 && Q_fp4.size(0) == 1 &&
                Q_fp4.size(1) == heads && Q_fp4.size(2) == 128 &&
                Q_fp4.size(3) == 96,
                "Q_fp4 must have shape (1,heads,128,96 packed)");
    TORCH_CHECK(K_fp4.dim() == 4 && K_fp4.size(0) == 1 &&
                K_fp4.size(1) == heads && K_fp4.size(2) == 128 &&
                K_fp4.size(3) == 96,
                "K_fp4 must have shape (1,heads,128,96 packed)");
    TORCH_CHECK(Q_sc.numel() == heads * 3 * 32 * 16 &&
                K_sc.numel() == heads * 3 * 32 * 16,
                "Q_sc/K_sc must each contain heads*3*32*16 raw bytes");
    TORCH_CHECK(V_fp4.dim() == 4 && V_fp4.size(0) == 1 &&
                V_fp4.size(1) == heads && V_fp4.size(2) == 128 &&
                V_fp4.size(3) == 64,
                "V_fp4 must have shape (1,heads,128,64 packed backing)");
    TORCH_CHECK(V_sc.dim() == 6 && V_sc.size(0) == 1 &&
                V_sc.size(1) == heads && V_sc.size(2) == 2 &&
                V_sc.size(3) == 1 && V_sc.size(4) == 32 &&
                V_sc.size(5) == 16,
                "V_sc must have two records (1,heads,2,1,32,16)");
    TORCH_CHECK(P_debug.numel() == heads * 128 * 64,
                "P_debug must contain heads*128*64 raw bytes");
    TORCH_CHECK(P_sc_debug.numel() == heads * 32 * 16,
                "P_sc_debug must contain heads*32*16 raw bytes");
    TORCH_CHECK(O.dim() == 4 && O.size(0) == 1 && O.size(1) == 128 &&
                O.size(2) == heads && O.size(3) == 128,
                "O must have shape (1,128,heads,128)");
    TORCH_CHECK(LSE.numel() == heads * 128 &&
                Row_sum.numel() == heads * 128,
                "LSE and Row_sum must each contain heads*128 values");
    TORCH_CHECK(stage >= 0 && stage <= 4, "stage must be in [0,4]");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "min_blocks_per_sm must be 1 or 2");
    TORCH_CHECK(grid_blocks > 0 && grid_blocks <= batch * heads,
                "grid_blocks must be in [1,batch*heads]");
    TORCH_CHECK(Trace.numel() >= grid_blocks * 24,
                "Trace must contain at least grid_blocks*24 int64 values");

    using C = config_fp4pv_nb128_dual_dvo64_attention_shell;
    using G = globals_mxfp4_nb128_dual_dvo64_attention_shell<C>;
    G g{
        reinterpret_cast<const uint8_t *>(Q_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<uint8_t *>(P_debug.data_ptr()),
        reinterpret_cast<uint8_t *>(P_sc_debug.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<float *>(LSE.data_ptr<float>()),
        reinterpret_cast<float *>(Row_sum.data_ptr<float>()),
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        batch,
        heads,
        seqlen,
        v_scale_rows,
        softmax_scale_log2,
        stage,
        grid_blocks
    };

    if (min_blocks_per_sm == 1) {
        constexpr int ONE_RESIDENT_CONTROL_SMEM = 128 * 1024;
        const int launch_smem = g.dynamic_shared_memory() > ONE_RESIDENT_CONTROL_SMEM
            ? g.dynamic_shared_memory()
            : ONE_RESIDENT_CONTROL_SMEM;
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_nb128_dual_dvo64_attention_shell<C, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            launch_smem));
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), launch_smem,
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config,
            kernel_mxfp4_nb128_dual_dvo64_attention_shell<C, 1>, g));
    } else {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_nb128_dual_dvo64_attention_shell<C, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.dynamic_shared_memory()));
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), g.dynamic_shared_memory(),
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config,
            kernel_mxfp4_nb128_dual_dvo64_attention_shell<C, 2>, g));
    }
}

void dispatch_mxfp4_nb128_dual_dvo64_multistep_shell(
    at::Tensor Q_fp4,
    at::Tensor Q_sc,
    at::Tensor K_fp4,
    at::Tensor K_sc,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    at::Tensor P_debug,
    at::Tensor P_sc_debug,
    at::Tensor Acc_scale_debug,
    at::Tensor O,
    at::Tensor LSE,
    at::Tensor Row_sum,
    at::Tensor Trace,
    float softmax_scale_log2,
    int stage,
    int capture_p,
    int min_blocks_per_sm,
    int grid_blocks,
    int production,
    int full_dvo128,
    int shared_accum,
    int register_accum) {
    CHECK_INPUT(Q_fp4);
    CHECK_INPUT(Q_sc);
    CHECK_INPUT(K_fp4);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(P_debug);
    CHECK_INPUT(P_sc_debug);
    CHECK_INPUT(Acc_scale_debug);
    CHECK_INPUT(O);
    CHECK_INPUT(LSE);
    CHECK_INPUT(Row_sum);
    CHECK_INPUT(Trace);
    TORCH_CHECK(Q_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                K_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Q/K/V payloads must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Byte &&
                K_sc.dtype() == at::ScalarType::Byte &&
                V_sc.dtype() == at::ScalarType::Byte &&
                P_debug.dtype() == at::ScalarType::Byte &&
                P_sc_debug.dtype() == at::ScalarType::Byte,
                "Q/K/V scales and P debug outputs must be raw bytes");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16,
                "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float &&
                Row_sum.dtype() == at::ScalarType::Float &&
                Acc_scale_debug.dtype() == at::ScalarType::Float,
                "LSE, Row_sum, and Acc_scale_debug must be float32");
    TORCH_CHECK(Trace.dtype() == at::ScalarType::Long,
                "Trace must be int64");

    TORCH_CHECK(Q_fp4.dim() == 4 && Q_fp4.size(2) == 128 &&
                Q_fp4.size(3) == 96,
                "Q_fp4 must have shape (query_tiles,heads,128,96 packed)");
    const int query_tiles = Q_fp4.size(0);
    const int heads = Q_fp4.size(1);
    TORCH_CHECK(query_tiles > 0 && heads > 0,
                "query_tiles and heads must be positive");
    TORCH_CHECK(K_fp4.dim() == 4 && K_fp4.size(0) == heads &&
                K_fp4.size(2) == 128 && K_fp4.size(3) == 96,
                "K_fp4 must have shape (heads,key_tiles,128,96 packed)");
    const int key_tiles = K_fp4.size(1);
    const int seqlen = query_tiles * 128;
    TORCH_CHECK(key_tiles == query_tiles,
                "K key_tiles must equal query_tiles");
    TORCH_CHECK(Q_sc.numel() == query_tiles * heads * 3 * 32 * 16 &&
                K_sc.numel() == heads * key_tiles * 3 * 32 * 16,
                "Q_sc/K_sc raw-byte sizes do not match Q/K tiles");
    TORCH_CHECK(V_fp4.dim() == 4 && V_fp4.size(0) == 1 &&
                V_fp4.size(1) == heads && V_fp4.size(2) == 128 &&
                V_fp4.size(3) == seqlen / 2,
                "V_fp4 must have shape (1,heads,128,seqlen/2 packed)");
    TORCH_CHECK(
        (full_dvo128 == 0 && V_sc.dim() == 7 && V_sc.size(0) == 1 &&
         V_sc.size(1) == heads && V_sc.size(2) == key_tiles &&
         V_sc.size(3) == 2 && V_sc.size(4) == 1 &&
         V_sc.size(5) == 32 && V_sc.size(6) == 16) ||
        (full_dvo128 == 1 && V_sc.dim() == 5 && V_sc.size(0) == heads &&
         V_sc.size(1) == 1 && V_sc.size(2) == key_tiles &&
         V_sc.size(3) == 32 && V_sc.size(4) == 16),
        "V_sc must be split (1,H,K,2,1,32,16) for DVO64x2 or full (H,1,K,32,16) for DVO128");
    const int64_t p_debug_values =
        static_cast<int64_t>(query_tiles) * heads * key_tiles * 128 * 64;
    const int64_t p_sc_debug_values =
        static_cast<int64_t>(query_tiles) * heads * key_tiles * 32 * 16;
    const int64_t acc_scale_debug_values =
        static_cast<int64_t>(query_tiles) * heads * key_tiles * 128;
    TORCH_CHECK((capture_p == 0 && P_debug.numel() >= 1 &&
                 P_sc_debug.numel() >= 1 &&
                 Acc_scale_debug.numel() >= 1) ||
                (capture_p == 1 && P_debug.numel() == p_debug_values &&
                 P_sc_debug.numel() == p_sc_debug_values &&
                 Acc_scale_debug.numel() == acc_scale_debug_values),
                "P debug buffers must be scalar when disabled or cover every query/key tile pair");
    TORCH_CHECK(O.dim() == 4 && O.size(0) == 1 && O.size(1) == seqlen &&
                O.size(2) == heads && O.size(3) == 128,
                "O must have shape (1,seqlen,heads,128)");
    TORCH_CHECK(LSE.numel() == query_tiles * heads * 128 &&
                Row_sum.numel() == query_tiles * heads * 128,
                "LSE and Row_sum must contain query_tiles*heads*128 values");
    TORCH_CHECK(stage >= 0 && stage <= 4, "stage must be in [0,4]");
    TORCH_CHECK(capture_p == 0 || capture_p == 1,
                "capture_p must be 0 or 1");
    TORCH_CHECK(production == 0 || production == 1,
                "production must be 0 or 1");
    TORCH_CHECK(full_dvo128 == 0 || full_dvo128 == 1,
                "full_dvo128 must be 0 or 1");
    TORCH_CHECK(shared_accum == 0 || shared_accum == 1,
                "shared_accum must be 0 or 1");
    TORCH_CHECK(register_accum == 0 || register_accum == 1,
                "register_accum must be 0 or 1");
    TORCH_CHECK(shared_accum == 0 || full_dvo128 == 1,
                "shared_accum is only enabled for the full DVO128 route");
    TORCH_CHECK(register_accum == 0 || full_dvo128 == 1,
                "register_accum is only enabled for the full DVO128 route");
    TORCH_CHECK(shared_accum == 0 || register_accum == 0,
                "shared and register accumulation are mutually exclusive");
    TORCH_CHECK(production == 0 || (stage == 4 && capture_p == 0),
                "production specialization requires stage=4 and capture_p=0");
    TORCH_CHECK(min_blocks_per_sm == 1 || min_blocks_per_sm == 2,
                "min_blocks_per_sm must be 1 or 2");
    TORCH_CHECK(grid_blocks > 0 &&
                grid_blocks <= query_tiles * heads,
                "grid_blocks must be in [1,query_tiles*heads]");
    TORCH_CHECK(Trace.numel() >= grid_blocks * 32,
                "Trace must contain at least grid_blocks*32 int64 values");

    auto launch = [&]<typename C, bool SHARED_ACCUM>() {
        using G = globals_mxfp4_nb128_dual_dvo64_multistep_shell<C>;
        G g{
            reinterpret_cast<const uint8_t *>(Q_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
            reinterpret_cast<const uint8_t *>(K_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            reinterpret_cast<uint8_t *>(P_debug.data_ptr()),
            reinterpret_cast<uint8_t *>(P_sc_debug.data_ptr()),
            reinterpret_cast<float *>(Acc_scale_debug.data_ptr<float>()),
            reinterpret_cast<bf16 *>(O.data_ptr()),
            reinterpret_cast<float *>(LSE.data_ptr<float>()),
            reinterpret_cast<float *>(Row_sum.data_ptr<float>()),
            reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
            1,
            heads,
            seqlen,
            query_tiles,
            key_tiles,
            softmax_scale_log2,
            stage,
            capture_p,
            grid_blocks
        };

        if (min_blocks_per_sm == 1) {
            constexpr int ONE_RESIDENT_CONTROL_SMEM = 128 * 1024;
            const int launch_smem = g.dynamic_shared_memory() >
                ONE_RESIDENT_CONTROL_SMEM
                ? g.dynamic_shared_memory() : ONE_RESIDENT_CONTROL_SMEM;
            LaunchConfig<true, false> launch_config(
                g.grid(), g.block(), launch_smem,
                at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
            if (production != 0) {
                CUDACHECK(cudaFuncSetAttribute(
                    kernel_mxfp4_nb128_dual_dvo64_multistep_shell<
                        C, 1, false, SHARED_ACCUM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    launch_smem));
                CUDACHECK(cudaLaunchKernelEx(
                    launch_config,
                    kernel_mxfp4_nb128_dual_dvo64_multistep_shell<
                        C, 1, false, SHARED_ACCUM>,
                    g));
            } else {
                CUDACHECK(cudaFuncSetAttribute(
                    kernel_mxfp4_nb128_dual_dvo64_multistep_shell<
                        C, 1, true, SHARED_ACCUM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    launch_smem));
                CUDACHECK(cudaLaunchKernelEx(
                    launch_config,
                    kernel_mxfp4_nb128_dual_dvo64_multistep_shell<
                        C, 1, true, SHARED_ACCUM>,
                    g));
            }
        } else {
            LaunchConfig<true, false> launch_config(
                g.grid(), g.block(), g.dynamic_shared_memory(),
                at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
            if (production != 0) {
                CUDACHECK(cudaFuncSetAttribute(
                    kernel_mxfp4_nb128_dual_dvo64_multistep_shell<
                        C, 2, false, SHARED_ACCUM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    g.dynamic_shared_memory()));
                CUDACHECK(cudaLaunchKernelEx(
                    launch_config,
                    kernel_mxfp4_nb128_dual_dvo64_multistep_shell<
                        C, 2, false, SHARED_ACCUM>,
                    g));
            } else {
                CUDACHECK(cudaFuncSetAttribute(
                    kernel_mxfp4_nb128_dual_dvo64_multistep_shell<
                        C, 2, true, SHARED_ACCUM>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    g.dynamic_shared_memory()));
                CUDACHECK(cudaLaunchKernelEx(
                    launch_config,
                    kernel_mxfp4_nb128_dual_dvo64_multistep_shell<
                        C, 2, true, SHARED_ACCUM>,
                    g));
            }
        }
    };

    if (full_dvo128 != 0 && register_accum != 0) {
        launch.template operator()<
            config_fp4pv_nb128_dvo128_register_multistep_shell, false>();
    } else if (full_dvo128 != 0 && shared_accum != 0) {
        launch.template operator()<
            config_fp4pv_nb128_dvo128_multistep_shell, true>();
    } else if (full_dvo128 != 0) {
        launch.template operator()<
            config_fp4pv_nb128_dvo128_multistep_shell, false>();
    } else {
        launch.template operator()<
            config_fp4pv_nb128_dual_dvo64_attention_shell, false>();
    }
}
#endif

void dispatch_mxfp4_v_tma_tile_check_debug(at::Tensor V_fp4,
                                           at::Tensor Debug,
                                           int b,
                                           int h,
                                           int k_iter,
                                           int mode) {
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(Debug);
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(Debug.dtype() == at::ScalarType::Int,
                "Debug must be int32");
    TORCH_CHECK(V_fp4.dim() == 4,
                "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
    TORCH_CHECK(Debug.numel() >= 12,
                "Debug must have at least 12 int32 elements");

    const int batch = V_fp4.size(0);
    const int heads = V_fp4.size(1);
    const int dvo = V_fp4.size(2);
    const int packed_seqlen = V_fp4.size(3);
    const int seqlen = packed_seqlen * 2;

    TORCH_CHECK(dvo == 128, "V_fp4 Dvo must be 128");
    TORCH_CHECK(seqlen % 128 == 0, "V_fp4 sequence length must be divisible by 128");
    TORCH_CHECK(b >= 0 && b < batch, "b is out of range");
    TORCH_CHECK(h >= 0 && h < heads, "h is out of range");
    TORCH_CHECK(k_iter >= 0 && k_iter < seqlen / 128, "k_iter is out of range");
    TORCH_CHECK(mode >= 0 && mode <= 3, "mode must be 0(compare), 1(manual), 2(tma), or 3(tma-wait)");

    using C = config_fp4pv<128, 128, 192, 128, 1>;
    using G = globals_mxfp4_v_tma_tile_check_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<int32_t *>(Debug.data_ptr()),
        batch,
        heads,
        seqlen,
        b,
        h,
        k_iter,
        mode,
        nullptr,
        0,
        0
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_v_tma_tile_check_debug<C>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(dim3(C::CLUSTER_SIZE), g.block(), g.dynamic_shared_memory(),
                                            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_v_tma_tile_check_debug<C>, g));
}

void dispatch_mxfp4_v_tma_multicast_debug(at::Tensor V_fp4,
                                           at::Tensor Debug,
                                           int b,
                                           int h,
                                           int k_iter,
                                           int mode) {
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(Debug);
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(Debug.dtype() == at::ScalarType::Int,
                "Debug must be int32");
    TORCH_CHECK(V_fp4.dim() == 4,
                "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
    TORCH_CHECK(Debug.numel() >= 64,
                "Debug must have at least 64 int32 elements");

    const int batch = V_fp4.size(0);
    const int heads = V_fp4.size(1);
    const int dvo = V_fp4.size(2);
    const int packed_seqlen = V_fp4.size(3);
    const int seqlen = packed_seqlen * 2;

    TORCH_CHECK(dvo == 128, "V_fp4 Dvo must be 128");
    TORCH_CHECK(seqlen % 128 == 0, "V_fp4 sequence length must be divisible by 128");
    TORCH_CHECK(b >= 0 && b < batch, "b is out of range");
    TORCH_CHECK(h >= 0 && h < heads, "h is out of range");
    TORCH_CHECK(k_iter >= 0 && k_iter < seqlen / 128, "k_iter is out of range");
    TORCH_CHECK(mode >= 0 && mode <= 7,
                "mode must be 0(TK-style multicast), 1(owner mbarrier), 2(per-rank unicast), 3(TK multicast plus rank-local TCGEN consume), 4(per-rank unicast plus rank-local TCGEN consume), 5(two-slot multicast/TCGEN reuse ring), 6(unicast plus TMEM allocate cut), or 7(unicast plus TCGEN issue cut)");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_mxfp4_v_tma_tile_check_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<int32_t *>(Debug.data_ptr()),
        batch,
        heads,
        seqlen,
        b,
        h,
        k_iter,
        mode,
        nullptr,
        0,
        0
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_v_tma_multicast_debug<C>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(dim3(C::CLUSTER_SIZE), g.block(), g.dynamic_shared_memory(),
                                            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_v_tma_multicast_debug<C>, g));
}

void dispatch_mxfp4_v_tma_multicast_vscale_debug(at::Tensor V_fp4,
                                                  at::Tensor V_sc_prepared,
                                                  at::Tensor Debug,
                                                  int b,
                                                  int h,
                                                  int k_iter,
                                                  int mode) {
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc_prepared);
    CHECK_INPUT(Debug);
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(Debug.dtype() == at::ScalarType::Int,
                "Debug must be int32");
    TORCH_CHECK(V_fp4.dim() == 4,
                "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
    TORCH_CHECK(Debug.numel() >= 64,
                "Debug must have at least 64 int32 elements");

    const int batch = V_fp4.size(0);
    const int heads = V_fp4.size(1);
    const int dvo = V_fp4.size(2);
    const int packed_seqlen = V_fp4.size(3);
    const int seqlen = packed_seqlen * 2;

    TORCH_CHECK(dvo == 128, "V_fp4 Dvo must be 128");
    TORCH_CHECK(seqlen % 128 == 0, "V_fp4 sequence length must be divisible by 128");
    TORCH_CHECK(b >= 0 && b < batch, "b is out of range");
    TORCH_CHECK(h >= 0 && h < heads, "h is out of range");
    TORCH_CHECK(k_iter >= 0 && k_iter < seqlen / 128, "k_iter is out of range");
    TORCH_CHECK(mode >= 0 && mode <= 7,
                "mode must be 0(TK-style multicast), 1(owner mbarrier), 2(per-rank unicast), 3(TK multicast plus rank-local TCGEN consume), 4(per-rank unicast plus rank-local TCGEN consume), 5(two-slot multicast/TCGEN reuse ring), 6(unicast plus TMEM allocate cut), or 7(unicast plus TCGEN issue cut)");
    check_fp4pv_mxfp4_v_scale_tensor(V_sc_prepared, batch, heads, seqlen, 128);

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_mxfp4_v_tma_tile_check_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<int32_t *>(Debug.data_ptr()),
        batch,
        heads,
        seqlen,
        b,
        h,
        k_iter,
        mode,
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        (int)V_sc_prepared.size(2),
        1
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_v_tma_multicast_debug<C>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(dim3(C::CLUSTER_SIZE), g.block(), g.dynamic_shared_memory(),
                                            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_v_tma_multicast_debug<C>, g));
}

void dispatch_mxfp4_k_tma_multicast_debug(at::Tensor K,
                                           at::Tensor K_sc,
                                           at::Tensor Debug,
                                           int b,
                                           int h,
                                           int k_iter,
                                           int mode,
                                           bool check_scale) {
    CHECK_INPUT(K);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(Debug);
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Debug.dtype() == at::ScalarType::Int,
                "Debug must be int32");
    TORCH_CHECK(K.dim() == 4,
                "K must have shape (batch, heads, seqlen, packed_dqk)");
    TORCH_CHECK(Debug.numel() >= 64,
                "Debug must have at least 64 int32 elements");

    const int batch = K.size(0);
    const int heads = K.size(1);
    const int seqlen = K.size(2);
    const int packed_dqk = K.size(3);

    TORCH_CHECK(packed_dqk == 192 / 2, "K packed Dqk must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "K sequence length must be divisible by 128");
    TORCH_CHECK(b >= 0 && b < batch, "b is out of range");
    TORCH_CHECK(h >= 0 && h < heads, "h is out of range");
    TORCH_CHECK(k_iter >= 0 && k_iter < seqlen / 128, "k_iter is out of range");
    TORCH_CHECK(mode >= 0 && mode <= 2,
                "mode must be 0(TK-style multicast), 1(owner mbarrier), or 2(per-rank unicast)");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_mxfp4_k_tma_multicast_debug<C>;
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;
    G g{
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        reinterpret_cast<const uint8_t *>(K.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<int32_t *>(Debug.data_ptr()),
        batch,
        heads,
        seqlen,
        b,
        h,
        k_iter,
        mode,
        check_scale ? 1 : 0
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_k_tma_multicast_debug<C>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(dim3(C::CLUSTER_SIZE), g.block(), g.dynamic_shared_memory(),
                                            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_k_tma_multicast_debug<C>, g));
}

void dispatch_mxfp4_kv_tma_multicast_debug(at::Tensor K,
                                            at::Tensor K_sc,
                                            at::Tensor V_fp4,
                                            at::Tensor V_sc_prepared,
                                            at::Tensor Debug,
                                            int b,
                                            int h,
                                            int k_iter,
                                            int mode,
                                            bool check_scale) {
    CHECK_INPUT(K);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc_prepared);
    CHECK_INPUT(Debug);
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "K must be float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(Debug.dtype() == at::ScalarType::Int,
                "Debug must be int32");
    TORCH_CHECK(K.dim() == 4,
                "K must have shape (batch, heads, seqlen, packed_dqk)");
    TORCH_CHECK(V_fp4.dim() == 4,
                "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
    TORCH_CHECK(Debug.numel() >= 128,
                "Debug must have at least 128 int32 elements");

    const int batch = K.size(0);
    const int heads = K.size(1);
    const int seqlen = K.size(2);
    const int packed_dqk = K.size(3);
    TORCH_CHECK(packed_dqk == 192 / 2, "K packed Dqk must be 96");
    TORCH_CHECK(V_fp4.size(0) == batch && V_fp4.size(1) == heads &&
                V_fp4.size(2) == 128 && V_fp4.size(3) == seqlen / 2,
                "V_fp4 must have shape (batch, heads, 128, seqlen/2) matching K");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(b >= 0 && b < batch, "b is out of range");
    TORCH_CHECK(h >= 0 && h < heads, "h is out of range");
    TORCH_CHECK(k_iter >= 0 && k_iter < seqlen / 128, "k_iter is out of range");
    TORCH_CHECK(mode >= 0 && mode <= 3,
                "mode must be 0(expect then K+V multicast), 1(load before expect), 2(per-rank unicast), or 3(serial K then V multicast)");
    if (check_scale) {
        check_fp4pv_mxfp4_v_scale_tensor(V_sc_prepared, batch, heads, seqlen, 128);
    }

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_mxfp4_kv_tma_multicast_debug<C>;
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;
    G g{
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(K.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        reinterpret_cast<int32_t *>(Debug.data_ptr()),
        batch,
        heads,
        seqlen,
        b,
        h,
        k_iter,
        mode,
        check_scale ? 1 : 0,
        check_scale ? (int)V_sc_prepared.size(2) : 0
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_kv_tma_multicast_debug<C>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(dim3(C::CLUSTER_SIZE), g.block(), g.dynamic_shared_memory(),
                                            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_kv_tma_multicast_debug<C>, g));
}

void dispatch_mxfp4_qk_tma_multicast_debug(at::Tensor Q,
                                            at::Tensor Q_sc,
                                            at::Tensor K,
                                            at::Tensor K_sc,
                                            at::Tensor Debug,
                                            int b,
                                            int h,
                                            int q_iter,
                                            int k_iter,
                                            int mode) {
    CHECK_INPUT(Q);
    CHECK_INPUT(Q_sc);
    CHECK_INPUT(K);
    CHECK_INPUT(K_sc);
    CHECK_INPUT(Debug);
    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Debug.dtype() == at::ScalarType::Int,
                "Debug must be int32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4,
                "Q and K must have shape (batch, heads, seqlen, packed_dqk)");
    TORCH_CHECK(Debug.numel() >= 96,
                "Debug must have at least 96 int32 elements");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    const int packed_dqk = Q.size(3);
    TORCH_CHECK(K.size(0) == batch && K.size(1) == heads &&
                K.size(2) == seqlen && K.size(3) == packed_dqk,
                "K must match Q shape");
    TORCH_CHECK(packed_dqk == 192 / 2, "Q/K packed Dqk must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(b >= 0 && b < batch, "b is out of range");
    TORCH_CHECK(h >= 0 && h < heads, "h is out of range");
    TORCH_CHECK(q_iter >= 0 && q_iter < seqlen / 128, "q_iter is out of range");
    TORCH_CHECK(k_iter >= 0 && k_iter < seqlen / 128, "k_iter is out of range");
    TORCH_CHECK(mode >= 0 && mode <= 4,
                "mode must be 0(per-rank unicast QK), 1(rank0 multicast shared-slot QK), 2(multicast shared-slot QK ring), 3(per-rank unicast QK ring), or 4(serial per-rank unicast QK ring)");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_mxfp4_qk_tma_multicast_debug<C>;
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        reinterpret_cast<int32_t *>(Debug.data_ptr()),
        batch,
        heads,
        seqlen,
        b,
        h,
        q_iter,
        k_iter,
        mode
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_qk_tma_multicast_debug<C>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(),
                                            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_qk_tma_multicast_debug<C>, g));
}

void dispatch_mxfp4_v_tma_tile_check_fullg_debug(at::Tensor Q,
                                                 at::Tensor Q_sc,
                                                 at::Tensor Q_sg,
                                                 at::Tensor K,
                                                 at::Tensor K_sc,
                                                 at::Tensor K_sg,
                                                 at::Tensor V_fp4,
                                                 at::Tensor V_sc,
                                                 at::Tensor O,
                                                 at::Tensor LSE,
                                                 at::Tensor Debug,
                                                 int b,
                                                 int h,
                                                 int k_iter,
                                                 int mode) {
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
    CHECK_INPUT(Debug);
    TORCH_CHECK(Debug.dtype() == at::ScalarType::Int, "Debug must be int32");
    TORCH_CHECK(Debug.numel() >= 12, "Debug must have at least 12 int32 elements");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors");
    TORCH_CHECK(O.dim() == 4, "O must be rank-4");
    TORCH_CHECK(LSE.dim() == 4, "LSE must be rank-4");
    TORCH_CHECK(mode >= 0 && mode <= 3, "mode must be 0(compare), 1(manual), 2(tma), or 3(tma-wait)");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(K.size(0) == batch && K.size(1) == heads && K.size(2) == seqlen,
                "K must match Q batch/head/seqlen");
    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen && O.size(2) == heads && O.size(3) == 128,
                "O must have shape (batch, seqlen, heads, 128)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q/K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(b >= 0 && b < batch, "b is out of range");
    TORCH_CHECK(h >= 0 && h < heads, "h is out of range");
    TORCH_CHECK(k_iter >= 0 && k_iter < seqlen / 128, "k_iter is out of range");
    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_mxfp4_v_scale_half_contract(V_sc, batch, heads, seqlen);

    using C = config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>;
    using G = globals_fp4pv_mxfp4_dv<C, true>;
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        1.0f,
        0
    };
    g.disable_lse_store = 1;
    g.v_scale_records_per_head = V_sc.size(2);

    const int dynamic_shared_memory = g.dynamic_shared_memory();
    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_v_tma_tile_check_fullg_debug<C>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   dynamic_shared_memory));
    LaunchConfig<true, false> launch_config(dim3(1), dim3(C::NUM_THREADS), dynamic_shared_memory,
                                            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_mxfp4_v_tma_tile_check_fullg_debug<C>,
        g,
        reinterpret_cast<int32_t *>(Debug.data_ptr()),
        b,
        h,
        k_iter,
        mode));
}

void dispatch_mxfp4_pv_from_p_k256_debug(at::Tensor P_fp4,
                                         at::Tensor P_sc,
                                         at::Tensor V_fp4,
                                         at::Tensor V_sc,
                                         at::Tensor O) {
    CHECK_INPUT(P_fp4);
    CHECK_INPUT(P_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(P_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "P_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(P_sc.dtype() == at::ScalarType::Byte, "P_sc must be uint8 E8M0 scales");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte, "V_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(P_fp4.dim() == 4, "P_fp4 must have shape (batch, heads, seqlen, p_cols/2)");
    TORCH_CHECK(P_sc.dim() == 6, "P_sc must have shape (batch, heads, seqlen/128, p_cols/128, 32, 16)");
    TORCH_CHECK(V_fp4.dim() == 4, "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
    TORCH_CHECK(V_sc.dim() == 6, "V_sc must have shape (batch, heads, scale_rows, seqlen/128, 32, 16)");
    TORCH_CHECK(O.dim() == 4, "O must have shape (batch, seqlen, heads, 128)");

    const int batch = P_fp4.size(0);
    const int heads = P_fp4.size(1);
    const int seqlen = P_fp4.size(2);
    const int p_cols = P_fp4.size(3) * 2;
    const int v_scale_rows = V_sc.size(2);

    TORCH_CHECK(seqlen % 256 == 0, "seqlen must be divisible by 256");
    TORCH_CHECK(seqlen / 128 <= 512,
                "mxfp4_pv_from_p_k256_debug currently supports seqlen <= 65536");
    TORCH_CHECK(p_cols == seqlen, "K256 debug PV currently expects full compact P with p_cols == seqlen");
    TORCH_CHECK(P_sc.size(0) == batch && P_sc.size(1) == heads &&
                P_sc.size(2) == seqlen / 128 && P_sc.size(3) == p_cols / 128 &&
                P_sc.size(4) == 32 && P_sc.size(5) == 16,
                "P_sc must have shape (batch, heads, seqlen/128, p_cols/128, 32, 16)");
    TORCH_CHECK(V_fp4.size(0) == batch && V_fp4.size(1) == heads &&
                V_fp4.size(2) == 128 && V_fp4.size(3) == seqlen / 2,
                "V_fp4 must have shape (batch, heads, 128, seqlen/2)");
    TORCH_CHECK(v_scale_rows >= 1 && V_sc.size(0) == batch && V_sc.size(1) == heads &&
                V_sc.size(3) == seqlen / 128 && V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "V_sc must have shape (batch, heads, scale_rows>=1, seqlen/128, 32, 16)");
    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen && O.size(2) == heads && O.size(3) == 128,
                "O must have shape (batch, seqlen, heads, 128)");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_mxfp4_pv_from_p_k256_debug<C>;
    G g{
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        batch,
        heads,
        seqlen,
        p_cols,
        v_scale_rows
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_pv_from_p_k256_debug<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_pv_from_p_k256_debug<C>, g));
}

template <int KTile, int NTile, int ClusterSize, int ScaleLayout>
static void launch_mxfp4_tcgen_shape_debug(at::Tensor A_fp4,
                                           at::Tensor A_sc,
                                           at::Tensor B_fp4,
                                           at::Tensor B_sc,
                                           at::Tensor O,
                                           int debug_stage) {
    using C = config_mxfp4_tcgen_shape_debug<KTile, NTile, ClusterSize, ScaleLayout>;
    using G = globals_mxfp4_tcgen_shape_debug<C>;
    G g{
        reinterpret_cast<const uint8_t *>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(A_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        (int)A_fp4.size(0),
        (int)A_fp4.size(1),
        (int)B_fp4.size(1),
        (int)A_sc.size(1),
        (int)B_sc.size(1),
        debug_stage
    };
    CUDACHECK(cudaFuncSetAttribute(kernel_mxfp4_tcgen_shape_debug<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_tcgen_shape_debug<C>, g));
}

void dispatch_mxfp4_tcgen_shape_debug(at::Tensor A_fp4,
                                      at::Tensor A_sc,
                                      at::Tensor B_fp4,
                                      at::Tensor B_sc,
                                      at::Tensor O,
                                      int k_tile,
                                      int n_tile,
                                      int ncta,
                                      int scale_layout,
                                      int debug_stage) {
    CHECK_INPUT(A_fp4);
    CHECK_INPUT(A_sc);
    CHECK_INPUT(B_fp4);
    CHECK_INPUT(B_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(A_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(B_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(A_fp4.dim() == 2, "A_fp4 must have shape (rows, packed_k)");
    TORCH_CHECK(B_fp4.dim() == 2, "B_fp4 must have shape (n_tile, packed_k)");
    TORCH_CHECK(A_sc.dim() == 4, "A_sc must have shape (rows/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4, "B_sc must have shape (n_tile/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(O.dim() == 2, "O must have shape (rows, n_tile)");
    TORCH_CHECK(k_tile == 64 || k_tile == 128 || k_tile == 256, "k_tile must be 64, 128, or 256");
    TORCH_CHECK(n_tile == 128 || n_tile == 256 || n_tile == 512, "n_tile must be 128, 256, or 512");
    TORCH_CHECK(ncta == 1 || ncta == 2, "ncta must be 1 or 2");
    TORCH_CHECK(scale_layout == 0 || scale_layout == 1, "scale_layout must be 0(compact) or 1(gemm-like)");
    TORCH_CHECK(debug_stage >= 0 && debug_stage <= 11,
                "debug_stage must be 0(full), 1(preload), 2(payload), 3(scale-smem-published), 4(scale-tmem), 5(tcgen-no-store), 6(scale-smem-copy), 7(scale-smem-publish), 8(a-scale-first-word), 9(b-scale-first-word), 10(a-scale-full), or 11(b-scale-full)");
    const int rows = A_fp4.size(0);
    const int packed_k = A_fp4.size(1);
    const int scale_k_tiles = (k_tile + 127) / 128;
    TORCH_CHECK(rows > 0 && rows % 128 == 0, "A rows must be a positive multiple of 128");
    TORCH_CHECK(ncta == 1 || rows % 256 == 0, "ncta=2 requires rows divisible by 256");
    TORCH_CHECK(packed_k * 2 >= k_tile, "A_fp4 packed_k is smaller than requested k_tile");
    TORCH_CHECK(B_fp4.size(0) == n_tile && B_fp4.size(1) * 2 >= k_tile,
                "B_fp4 must have shape (n_tile, packed_k>=k_tile/2)");
    TORCH_CHECK(A_sc.size(0) == rows / 128 && A_sc.size(1) >= scale_k_tiles &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (rows/128, k_scale_tiles>=ceil(k/128), 32, 16)");
    TORCH_CHECK(B_sc.size(0) == n_tile / 128 && B_sc.size(1) >= scale_k_tiles &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (n_tile/128, k_scale_tiles>=ceil(k/128), 32, 16)");
    TORCH_CHECK(O.size(0) == rows && O.size(1) == n_tile,
                "O must have shape (A rows, n_tile)");

    if (k_tile == 64 && n_tile == 128 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<64, 128, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 64 && n_tile == 128 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<64, 128, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 64 && n_tile == 256 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<64, 256, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 64 && n_tile == 256 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<64, 256, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 64 && n_tile == 512 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<64, 512, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 64 && n_tile == 512 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<64, 512, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 128 && n_tile == 128 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<128, 128, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 128 && n_tile == 128 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<128, 128, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 128 && n_tile == 256 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<128, 256, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 128 && n_tile == 256 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<128, 256, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 128 && n_tile == 512 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<128, 512, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 128 && n_tile == 512 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<128, 512, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 256 && n_tile == 128 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<256, 128, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 256 && n_tile == 128 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<256, 128, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 256 && n_tile == 256 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<256, 256, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 256 && n_tile == 256 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<256, 256, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 256 && n_tile == 512 && ncta == 1 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<256, 512, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 256 && n_tile == 512 && ncta == 2 && scale_layout == 0) {
        launch_mxfp4_tcgen_shape_debug<256, 512, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 128 && n_tile == 128 && ncta == 2 && scale_layout == 1) {
        launch_mxfp4_tcgen_shape_debug<128, 128, 2, 1>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else if (k_tile == 256 && n_tile == 256 && ncta == 2 && scale_layout == 1) {
        launch_mxfp4_tcgen_shape_debug<256, 256, 2, 1>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
    } else {
        TORCH_CHECK(false, "unsupported explicit mxfp4_tcgen_shape_debug combination");
    }
}

template <int QuarterTile, int Quarters, int OptMode, int FixedDebugStage = -1>
static void launch_mxfp4_joint_qk_pv_quarter_debug(at::Tensor A_fp4,
                                                   at::Tensor A_sc,
                                                   at::Tensor B_fp4,
                                                   at::Tensor B_sc,
                                                   at::Tensor O,
                                                   int debug_stage) {
    using C = config_mxfp4_joint_qk_pv_quarter_debug<QuarterTile, Quarters, OptMode>;
    using G = globals_mxfp4_joint_qk_pv_quarter_debug<C>;
    const int kernel_debug_stage = (FixedDebugStage >= 0) ? FixedDebugStage : debug_stage;
    G g{
        reinterpret_cast<const uint8_t *>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(A_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        (int)A_fp4.size(0),
        (int)A_fp4.size(1),
        (int)B_fp4.size(1),
        (int)A_sc.size(1),
        (int)B_sc.size(1),
        kernel_debug_stage
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_joint_qk_pv_quarter_debug<C, FixedDebugStage>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_joint_qk_pv_quarter_debug<C, FixedDebugStage>, g));
}

static void launch_mxfp4_joint_qk_pv_q4_stage0_perf(at::Tensor A_fp4,
                                                    at::Tensor A_sc,
                                                    at::Tensor B_fp4,
                                                    at::Tensor B_sc,
                                                    at::Tensor O) {
    using C = config_mxfp4_joint_qk_pv_q4_stage0_perf;
    using G = globals_mxfp4_joint_qk_pv_q4_stage0_perf<C>;
    G g{
        reinterpret_cast<const uint8_t *>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(A_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        (int)A_fp4.size(0),
        (int)A_fp4.size(1),
        (int)B_fp4.size(1),
        (int)A_sc.size(1),
        (int)B_sc.size(1)
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_joint_qk_pv_q4_stage0_perf<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_joint_qk_pv_q4_stage0_perf<C>, g));
}

void dispatch_mxfp4_joint_qk_pv_q4_stage0_perf(at::Tensor A_fp4,
                                                at::Tensor A_sc,
                                                at::Tensor B_fp4,
                                                at::Tensor B_sc,
                                                at::Tensor O) {
    CHECK_INPUT(A_fp4);
    CHECK_INPUT(A_sc);
    CHECK_INPUT(B_fp4);
    CHECK_INPUT(B_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(A_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(B_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(A_fp4.dim() == 2, "A_fp4 must have shape (rows, packed_k)");
    TORCH_CHECK(B_fp4.dim() == 2, "B_fp4 must have shape (512, packed_k)");
    TORCH_CHECK(A_sc.dim() == 4, "A_sc must have shape (rows/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4, "B_sc must have shape (4, k_scale_tiles, 32, 16)");
    TORCH_CHECK(O.dim() == 2, "O must have shape (rows, 128)");

    const int rows = A_fp4.size(0);
    TORCH_CHECK(rows > 0 && rows % 128 == 0, "A rows must be a positive multiple of 128");
    TORCH_CHECK(A_fp4.size(1) * 2 >= 128, "A_fp4 packed_k is smaller than 128");
    TORCH_CHECK(B_fp4.size(0) == 512 && B_fp4.size(1) * 2 >= 128,
                "B_fp4 must have shape (512, packed_k>=64)");
    TORCH_CHECK(A_sc.size(0) == rows / 128 && A_sc.size(1) >= 1 &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (rows/128, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(B_sc.size(0) == 4 && B_sc.size(1) >= 1 &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (4, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(O.size(0) == rows && O.size(1) == 128, "O must have shape (rows, 128)");

    launch_mxfp4_joint_qk_pv_q4_stage0_perf(A_fp4, A_sc, B_fp4, B_sc, O);
}

static void launch_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf(at::Tensor A_fp4,
                                                             at::Tensor A_sc,
                                                             at::Tensor B_fp4,
                                                             at::Tensor B_sc,
                                                             at::Tensor O) {
    using C = config_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf;
    using G = globals_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf<C>;
    G g{
        reinterpret_cast<const uint8_t *>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(A_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        (int)A_fp4.size(0),
        (int)A_fp4.size(1),
        (int)B_fp4.size(1),
        (int)A_sc.size(1),
        (int)B_sc.size(1)
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf<C>, g));
}

void dispatch_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf(at::Tensor A_fp4,
                                                        at::Tensor A_sc,
                                                        at::Tensor B_fp4,
                                                        at::Tensor B_sc,
                                                        at::Tensor O) {
    CHECK_INPUT(A_fp4);
    CHECK_INPUT(A_sc);
    CHECK_INPUT(B_fp4);
    CHECK_INPUT(B_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(A_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(B_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(A_fp4.dim() == 2, "A_fp4 must have shape (rows, packed_k)");
    TORCH_CHECK(B_fp4.dim() == 2, "B_fp4 must have shape (512, packed_k)");
    TORCH_CHECK(A_sc.dim() == 4, "A_sc must have shape (rows/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4, "B_sc must have shape (4, k_scale_tiles, 32, 16)");
    TORCH_CHECK(O.dim() == 2, "O must have shape (rows, 128)");

    const int rows = A_fp4.size(0);
    TORCH_CHECK(rows > 0 && rows % 128 == 0, "A rows must be a positive multiple of 128");
    TORCH_CHECK(A_fp4.size(1) * 2 >= 128, "A_fp4 packed_k is smaller than 128");
    TORCH_CHECK(B_fp4.size(0) == 512 && B_fp4.size(1) * 2 >= 128,
                "B_fp4 must have shape (512, packed_k>=64)");
    TORCH_CHECK(A_sc.size(0) == rows / 128 && A_sc.size(1) >= 1 &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (rows/128, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(B_sc.size(0) == 4 && B_sc.size(1) >= 1 &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (4, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(O.size(0) == rows && O.size(1) == 128, "O must have shape (rows, 128)");

    launch_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf(A_fp4, A_sc, B_fp4, B_sc, O);
}

static void launch_mxfp4_joint_qk_pv_half_stage0_perf(at::Tensor A_fp4,
                                                      at::Tensor A_sc,
                                                      at::Tensor B_fp4,
                                                      at::Tensor B_sc,
                                                      at::Tensor O) {
    using C = config_mxfp4_joint_qk_pv_half_stage0_perf;
    using G = globals_mxfp4_joint_qk_pv_half_stage0_perf<C>;
    G g{
        reinterpret_cast<const uint8_t *>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(A_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        (int)A_fp4.size(0),
        (int)A_fp4.size(1),
        (int)B_fp4.size(1),
        (int)A_sc.size(1),
        (int)B_sc.size(1)
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_joint_qk_pv_half_stage0_perf<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_joint_qk_pv_half_stage0_perf<C>, g));
}

void dispatch_mxfp4_joint_qk_pv_half_stage0_perf(at::Tensor A_fp4,
                                                 at::Tensor A_sc,
                                                 at::Tensor B_fp4,
                                                 at::Tensor B_sc,
                                                 at::Tensor O) {
    CHECK_INPUT(A_fp4);
    CHECK_INPUT(A_sc);
    CHECK_INPUT(B_fp4);
    CHECK_INPUT(B_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(A_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(B_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(A_fp4.dim() == 2, "A_fp4 must have shape (rows, packed_k)");
    TORCH_CHECK(B_fp4.dim() == 2, "B_fp4 must have shape (512, packed_k)");
    TORCH_CHECK(A_sc.dim() == 4, "A_sc must have shape (rows/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4, "B_sc must have shape (4, k_scale_tiles, 32, 16)");
    TORCH_CHECK(O.dim() == 2, "O must have shape (rows, 128)");

    const int rows = A_fp4.size(0);
    TORCH_CHECK(rows > 0 && rows % 128 == 0, "A rows must be a positive multiple of 128");
    TORCH_CHECK(A_fp4.size(1) * 2 >= 128, "A_fp4 packed_k is smaller than 128");
    TORCH_CHECK(B_fp4.size(0) == 512 && B_fp4.size(1) * 2 >= 128,
                "B_fp4 must have shape (512, packed_k>=64)");
    TORCH_CHECK(A_sc.size(0) == rows / 128 && A_sc.size(1) >= 1 &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (rows/128, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(B_sc.size(0) == 4 && B_sc.size(1) >= 1 &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (4, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(O.size(0) == rows && O.size(1) == 128, "O must have shape (rows, 128)");

    TORCH_CHECK(false,
                "mxfp4_joint_qk_pv_half_stage0_perf is guarded: session6 rejected this "
                "half-tile variant after deterministic q4 mismatch/repeatability failures; "
                "see results/mxfp4_fa4_forward_recover_20260617/session6_half_tile_20260622T091554Z");
    launch_mxfp4_joint_qk_pv_half_stage0_perf(A_fp4, A_sc, B_fp4, B_sc, O);
}

template <typename C>
static void launch_mxfp4_joint_qk_pv_true_q4_stage0_typed(at::Tensor A_fp4,
                                                          at::Tensor A_sc,
                                                          at::Tensor B_fp4,
                                                          at::Tensor B_sc,
                                                          at::Tensor V_fp4,
                                                          at::Tensor V_sc,
                                                          at::Tensor O) {
    using G = globals_mxfp4_joint_qk_pv_true_q4_stage0_ref<C>;
    G g{
        reinterpret_cast<const uint8_t *>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(A_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        (int)A_fp4.size(0),
        (int)A_fp4.size(1),
        (int)B_fp4.size(1),
        (int)A_sc.size(1),
        (int)B_sc.size(1),
        512,
        1,
        (int)V_sc.size(2)
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_joint_qk_pv_true_q4_stage0_ref<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_joint_qk_pv_true_q4_stage0_ref<C>, g));
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_ref(at::Tensor A_fp4,
                                                        at::Tensor A_sc,
                                                        at::Tensor B_fp4,
                                                        at::Tensor B_sc,
                                                        at::Tensor V_fp4,
                                                        at::Tensor V_sc,
                                                        at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_ref>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf(at::Tensor A_fp4,
                                                                   at::Tensor A_sc,
                                                                   at::Tensor B_fp4,
                                                                   at::Tensor B_sc,
                                                                   at::Tensor V_fp4,
                                                                   at::Tensor V_sc,
                                                                   at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_pingpong_perf(at::Tensor A_fp4,
                                                                  at::Tensor A_sc,
                                                                  at::Tensor B_fp4,
                                                                  at::Tensor B_sc,
                                                                  at::Tensor V_fp4,
                                                                  at::Tensor V_sc,
                                                                  at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_pingpong_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_score_tmem_only_perf(at::Tensor A_fp4,
                                                                         at::Tensor A_sc,
                                                                         at::Tensor B_fp4,
                                                                         at::Tensor B_sc,
                                                                         at::Tensor V_fp4,
                                                                         at::Tensor V_sc,
                                                                         at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_score_tmem_only_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_scale_tmem_only_perf(at::Tensor A_fp4,
                                                                         at::Tensor A_sc,
                                                                         at::Tensor B_fp4,
                                                                         at::Tensor B_sc,
                                                                         at::Tensor V_fp4,
                                                                         at::Tensor V_sc,
                                                                         at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_scale_tmem_only_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_vonly_src_perf(at::Tensor A_fp4,
                                                                   at::Tensor A_sc,
                                                                   at::Tensor B_fp4,
                                                                   at::Tensor B_sc,
                                                                   at::Tensor V_fp4,
                                                                   at::Tensor V_sc,
                                                                   at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_vonly_src_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_ponly_src_perf(at::Tensor A_fp4,
                                                                   at::Tensor A_sc,
                                                                   at::Tensor B_fp4,
                                                                   at::Tensor B_sc,
                                                                   at::Tensor V_fp4,
                                                                   at::Tensor V_sc,
                                                                   at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_ponly_src_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_score_scale_tmem_perf(at::Tensor A_fp4,
                                                                          at::Tensor A_sc,
                                                                          at::Tensor B_fp4,
                                                                          at::Tensor B_sc,
                                                                          at::Tensor V_fp4,
                                                                          at::Tensor V_sc,
                                                                          at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_score_scale_tmem_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_score_p_src_perf(at::Tensor A_fp4,
                                                                     at::Tensor A_sc,
                                                                     at::Tensor B_fp4,
                                                                     at::Tensor B_sc,
                                                                     at::Tensor V_fp4,
                                                                     at::Tensor V_sc,
                                                                     at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_score_p_src_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_pv_split_k64_perf(at::Tensor A_fp4,
                                                                      at::Tensor A_sc,
                                                                      at::Tensor B_fp4,
                                                                      at::Tensor B_sc,
                                                                      at::Tensor V_fp4,
                                                                      at::Tensor V_sc,
                                                                      at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_pv_split_k64_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_split_output_accum_perf(at::Tensor A_fp4,
                                                                             at::Tensor A_sc,
                                                                             at::Tensor B_fp4,
                                                                             at::Tensor B_sc,
                                                                             at::Tensor V_fp4,
                                                                            at::Tensor V_sc,
                                                                            at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_split_output_accum_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_q4_stage0_barrier_trim_perf(at::Tensor A_fp4,
                                                                       at::Tensor A_sc,
                                                                       at::Tensor B_fp4,
                                                                       at::Tensor B_sc,
                                                                       at::Tensor V_fp4,
                                                                       at::Tensor V_sc,
                                                                       at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_q4_stage0_typed<
        config_mxfp4_joint_qk_pv_true_q4_stage0_barrier_trim_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(at::Tensor A_fp4,
                                                             at::Tensor A_sc,
                                                             at::Tensor B_fp4,
                                                             at::Tensor B_sc,
                                                             at::Tensor V_fp4,
                                                             at::Tensor V_sc,
                                                             at::Tensor O) {
    CHECK_INPUT(A_fp4);
    CHECK_INPUT(A_sc);
    CHECK_INPUT(B_fp4);
    CHECK_INPUT(B_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(A_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(B_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte, "V_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(A_fp4.dim() == 2, "A_fp4 must have shape (rows, packed_k)");
    TORCH_CHECK(B_fp4.dim() == 2, "B_fp4 must have shape (512, packed_k)");
    TORCH_CHECK(V_fp4.dim() == 4, "V_fp4 must have shape (1, 1, 128, 256)");
    TORCH_CHECK(A_sc.dim() == 4, "A_sc must have shape (rows/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4, "B_sc must have shape (4, k_scale_tiles, 32, 16)");
    TORCH_CHECK(V_sc.dim() == 6, "V_sc must have shape (1, 1, v_scale_rows, 4, 32, 16)");
    TORCH_CHECK(O.dim() == 2, "O must have shape (rows, 128)");

    const int rows = A_fp4.size(0);
    TORCH_CHECK(rows > 0 && rows % 128 == 0, "A rows must be a positive multiple of 128");
    TORCH_CHECK(A_fp4.size(1) * 2 >= 128, "A_fp4 packed_k is smaller than 128");
    TORCH_CHECK(B_fp4.size(0) == 512 && B_fp4.size(1) * 2 >= 128,
                "B_fp4 must have shape (512, packed_k>=64)");
    TORCH_CHECK(V_fp4.size(0) == 1 && V_fp4.size(1) == 1 &&
                V_fp4.size(2) == 128 && V_fp4.size(3) == 256,
                "V_fp4 must have shape (1, 1, 128, 256)");
    TORCH_CHECK(A_sc.size(0) == rows / 128 && A_sc.size(1) >= 1 &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (rows/128, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(B_sc.size(0) == 4 && B_sc.size(1) >= 1 &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (4, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(V_sc.size(0) == 1 && V_sc.size(1) == 1 && V_sc.size(2) >= 1 &&
                V_sc.size(3) == 4 && V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "V_sc must have shape (1, 1, v_scale_rows>=1, 4, 32, 16)");
    TORCH_CHECK(O.size(0) == rows && O.size(1) == 128, "O must have shape (rows, 128)");

}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_ref(at::Tensor A_fp4,
                                                   at::Tensor A_sc,
                                                   at::Tensor B_fp4,
                                                   at::Tensor B_sc,
                                                   at::Tensor V_fp4,
                                                   at::Tensor V_sc,
                                                   at::Tensor O) {
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_ref(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf(at::Tensor A_fp4,
                                                              at::Tensor A_sc,
                                                              at::Tensor B_fp4,
                                                              at::Tensor B_sc,
                                                              at::Tensor V_fp4,
                                                              at::Tensor V_sc,
                                                              at::Tensor O) {
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_pingpong_perf(at::Tensor A_fp4,
                                                             at::Tensor A_sc,
                                                             at::Tensor B_fp4,
                                                             at::Tensor B_sc,
                                                             at::Tensor V_fp4,
                                                             at::Tensor V_sc,
                                                             at::Tensor O) {
    const char *allow_rejected_pingpong = std::getenv("FP4PV_ENABLE_REJECTED_Q4_PINGPONG");
    TORCH_CHECK(allow_rejected_pingpong != nullptr && std::strcmp(allow_rejected_pingpong, "1") == 0,
                "mxfp4_joint_qk_pv_true_q4_stage0_pingpong_perf is guarded: session6 "
                "found it exact/repeatable but slower under NCU than the q4 vprestage winner; "
                "set FP4PV_ENABLE_REJECTED_Q4_PINGPONG=1 only for follow-up experiments. "
                "See results/mxfp4_fa4_forward_recover_20260617/session6_q4_qk_pv_pingpong_20260622T113150Z");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_pingpong_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void require_mxfp4_q4_low_resource_overlap_guard(const char *entrypoint) {
    const char *allow_low_resource = std::getenv("FP4PV_ENABLE_Q4_LOW_RESOURCE_OVERLAP");
    TORCH_CHECK(allow_low_resource != nullptr && std::strcmp(allow_low_resource, "1") == 0,
                entrypoint,
                " is guarded: session6 lower-resource q4 overlap candidate is experimental. "
                "Set FP4PV_ENABLE_Q4_LOW_RESOURCE_OVERLAP=1 only for reproducibility/profiling. "
                "The default remains mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf unless a "
                "candidate is proven correct and faster.");
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_score_tmem_only_perf(at::Tensor A_fp4,
                                                                    at::Tensor A_sc,
                                                                    at::Tensor B_fp4,
                                                                    at::Tensor B_sc,
                                                                    at::Tensor V_fp4,
                                                                    at::Tensor V_sc,
                                                                    at::Tensor O) {
    require_mxfp4_q4_low_resource_overlap_guard("mxfp4_joint_qk_pv_true_q4_stage0_score_tmem_only_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_score_tmem_only_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_scale_tmem_only_perf(at::Tensor A_fp4,
                                                                    at::Tensor A_sc,
                                                                    at::Tensor B_fp4,
                                                                    at::Tensor B_sc,
                                                                    at::Tensor V_fp4,
                                                                    at::Tensor V_sc,
                                                                    at::Tensor O) {
    require_mxfp4_q4_low_resource_overlap_guard("mxfp4_joint_qk_pv_true_q4_stage0_scale_tmem_only_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_scale_tmem_only_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_vonly_src_perf(at::Tensor A_fp4,
                                                              at::Tensor A_sc,
                                                              at::Tensor B_fp4,
                                                              at::Tensor B_sc,
                                                              at::Tensor V_fp4,
                                                              at::Tensor V_sc,
                                                              at::Tensor O) {
    require_mxfp4_q4_low_resource_overlap_guard("mxfp4_joint_qk_pv_true_q4_stage0_vonly_src_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_vonly_src_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_ponly_src_perf(at::Tensor A_fp4,
                                                              at::Tensor A_sc,
                                                              at::Tensor B_fp4,
                                                              at::Tensor B_sc,
                                                              at::Tensor V_fp4,
                                                              at::Tensor V_sc,
                                                              at::Tensor O) {
    require_mxfp4_q4_low_resource_overlap_guard("mxfp4_joint_qk_pv_true_q4_stage0_ponly_src_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_ponly_src_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_score_scale_tmem_perf(at::Tensor A_fp4,
                                                                     at::Tensor A_sc,
                                                                     at::Tensor B_fp4,
                                                                     at::Tensor B_sc,
                                                                     at::Tensor V_fp4,
                                                                     at::Tensor V_sc,
                                                                     at::Tensor O) {
    require_mxfp4_q4_low_resource_overlap_guard("mxfp4_joint_qk_pv_true_q4_stage0_score_scale_tmem_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_score_scale_tmem_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void require_mxfp4_q4_pipeline_gap_guard(const char *entrypoint) {
    const char *allow_pipeline_gap = std::getenv("FP4PV_ENABLE_Q4_PIPELINE_GAP");
    TORCH_CHECK(allow_pipeline_gap != nullptr && std::strcmp(allow_pipeline_gap, "1") == 0,
                entrypoint,
                " is guarded: session6 BF16-vs-MXFP4 pipeline-gap candidate is experimental. "
                "Set FP4PV_ENABLE_Q4_PIPELINE_GAP=1 only for reproducibility/profiling. "
                "The default remains mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf unless a "
                "candidate is exact and clearly faster.");
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_score_p_src_perf(at::Tensor A_fp4,
                                                                at::Tensor A_sc,
                                                                at::Tensor B_fp4,
                                                                at::Tensor B_sc,
                                                                at::Tensor V_fp4,
                                                                at::Tensor V_sc,
                                                                at::Tensor O) {
    require_mxfp4_q4_pipeline_gap_guard("mxfp4_joint_qk_pv_true_q4_stage0_score_p_src_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_score_p_src_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_pv_split_k64_perf(at::Tensor A_fp4,
                                                                 at::Tensor A_sc,
                                                                 at::Tensor B_fp4,
                                                                 at::Tensor B_sc,
                                                                 at::Tensor V_fp4,
                                                                 at::Tensor V_sc,
                                                                 at::Tensor O) {
    require_mxfp4_q4_pipeline_gap_guard("mxfp4_joint_qk_pv_true_q4_stage0_pv_split_k64_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_pv_split_k64_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_split_output_accum_perf(at::Tensor A_fp4,
                                                                       at::Tensor A_sc,
                                                                       at::Tensor B_fp4,
                                                                       at::Tensor B_sc,
                                                                       at::Tensor V_fp4,
                                                                       at::Tensor V_sc,
                                                                       at::Tensor O) {
    require_mxfp4_q4_pipeline_gap_guard("mxfp4_joint_qk_pv_true_q4_stage0_split_output_accum_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_split_output_accum_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void require_mxfp4_barrier_fused_p_guard(const char *entrypoint) {
    const char *allow_barrier_fused_p = std::getenv("FP4PV_ENABLE_BARRIER_FUSED_P");
    TORCH_CHECK(allow_barrier_fused_p != nullptr && std::strcmp(allow_barrier_fused_p, "1") == 0,
                entrypoint,
                " is guarded: session6 barrier/fused-P candidate is experimental. "
                "Set FP4PV_ENABLE_BARRIER_FUSED_P=1 only for reproducibility/profiling. "
                "The default remains q4 vprestage or true-half D unless a candidate is exact "
                "and clearly faster.");
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_barrier_trim_perf(at::Tensor A_fp4,
                                                                 at::Tensor A_sc,
                                                                 at::Tensor B_fp4,
                                                                 at::Tensor B_sc,
                                                                 at::Tensor V_fp4,
                                                                 at::Tensor V_sc,
                                                                 at::Tensor O) {
    require_mxfp4_barrier_fused_p_guard("mxfp4_joint_qk_pv_true_q4_stage0_barrier_trim_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_q4_stage0_barrier_trim_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_q4_stage0_fused_p_handoff_perf(at::Tensor A_fp4,
                                                                    at::Tensor A_sc,
                                                                    at::Tensor B_fp4,
                                                                    at::Tensor B_sc,
                                                                    at::Tensor V_fp4,
                                                                    at::Tensor V_sc,
                                                                    at::Tensor O) {
    require_mxfp4_barrier_fused_p_guard("mxfp4_joint_qk_pv_true_q4_stage0_fused_p_handoff_perf");
    TORCH_CHECK(false,
                "mxfp4_joint_qk_pv_true_q4_stage0_fused_p_handoff_perf is disabled: "
                "session6 real P-handoff semaphore variants were either incorrect or timed out; "
                "the post-fence CTA sync remains required for correctness in this source path.");
}

template <typename C>
static void launch_mxfp4_joint_qk_pv_true_half_stage0_typed(at::Tensor A_fp4,
                                                            at::Tensor A_sc,
                                                            at::Tensor B_fp4,
                                                            at::Tensor B_sc,
                                                            at::Tensor V_fp4,
                                                            at::Tensor V_sc,
                                                            at::Tensor O) {
    using G = globals_mxfp4_joint_qk_pv_true_half_stage0_perf<C>;
    G g{
        reinterpret_cast<const uint8_t *>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(A_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        (int)A_fp4.size(0),
        (int)A_fp4.size(1),
        (int)B_fp4.size(1),
        (int)A_sc.size(1),
        (int)B_sc.size(1),
        512,
        1,
        (int)V_sc.size(2)
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_joint_qk_pv_true_half_stage0_perf<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_joint_qk_pv_true_half_stage0_perf<C>, g));
}

static void launch_mxfp4_joint_qk_pv_true_half_stage0_perf(at::Tensor A_fp4,
                                                           at::Tensor A_sc,
                                                           at::Tensor B_fp4,
                                                           at::Tensor B_sc,
                                                           at::Tensor V_fp4,
                                                           at::Tensor V_sc,
                                                           at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_half_stage0_typed<
        config_mxfp4_joint_qk_pv_true_half_stage0_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_half_stage0_srcwait_pipeline_perf(at::Tensor A_fp4,
                                                                            at::Tensor A_sc,
                                                                            at::Tensor B_fp4,
                                                                            at::Tensor B_sc,
                                                                            at::Tensor V_fp4,
                                                                            at::Tensor V_sc,
                                                                            at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_half_stage0_typed<
        config_mxfp4_joint_qk_pv_true_half_stage0_srcwait_pipeline_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

static void launch_mxfp4_joint_qk_pv_true_half_stage0_barrier_trim_perf(at::Tensor A_fp4,
                                                                        at::Tensor A_sc,
                                                                        at::Tensor B_fp4,
                                                                        at::Tensor B_sc,
                                                                        at::Tensor V_fp4,
                                                                        at::Tensor V_sc,
                                                                        at::Tensor O) {
    launch_mxfp4_joint_qk_pv_true_half_stage0_typed<
        config_mxfp4_joint_qk_pv_true_half_stage0_barrier_trim_perf>(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_half_stage0_perf(at::Tensor A_fp4,
                                                      at::Tensor A_sc,
                                                      at::Tensor B_fp4,
                                                      at::Tensor B_sc,
                                                      at::Tensor V_fp4,
                                                      at::Tensor V_sc,
                                                      at::Tensor O) {
    CHECK_INPUT(A_fp4);
    CHECK_INPUT(A_sc);
    CHECK_INPUT(B_fp4);
    CHECK_INPUT(B_sc);
    CHECK_INPUT(V_fp4);
    CHECK_INPUT(V_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(A_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(B_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte, "V_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(A_fp4.dim() == 2, "A_fp4 must have shape (rows, packed_k)");
    TORCH_CHECK(B_fp4.dim() == 2, "B_fp4 must have shape (512, packed_k)");
    TORCH_CHECK(V_fp4.dim() == 4, "V_fp4 must have shape (1, 1, 128, 256)");
    TORCH_CHECK(A_sc.dim() == 4, "A_sc must have shape (rows/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4, "B_sc must have shape (4, k_scale_tiles, 32, 16)");
    TORCH_CHECK(V_sc.dim() == 6, "V_sc must have shape (1, 1, v_scale_rows, 4, 32, 16)");
    TORCH_CHECK(O.dim() == 2, "O must have shape (rows, 128)");

    const int rows = A_fp4.size(0);
    TORCH_CHECK(rows > 0 && rows % 128 == 0, "A rows must be a positive multiple of 128");
    TORCH_CHECK(A_fp4.size(1) * 2 >= 128, "A_fp4 packed_k is smaller than 128");
    TORCH_CHECK(B_fp4.size(0) == 512 && B_fp4.size(1) * 2 >= 128,
                "B_fp4 must have shape (512, packed_k>=64)");
    TORCH_CHECK(V_fp4.size(0) == 1 && V_fp4.size(1) == 1 &&
                V_fp4.size(2) == 128 && V_fp4.size(3) == 256,
                "V_fp4 must have shape (1, 1, 128, 256)");
    TORCH_CHECK(A_sc.size(0) == rows / 128 && A_sc.size(1) >= 1 &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (rows/128, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(B_sc.size(0) == 4 && B_sc.size(1) >= 1 &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (4, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(V_sc.size(0) == 1 && V_sc.size(1) == 1 && V_sc.size(2) >= 1 &&
                V_sc.size(3) == 4 && V_sc.size(4) == 32 && V_sc.size(5) == 16,
                "V_sc must have shape (1, 1, v_scale_rows>=1, 4, 32, 16)");
    TORCH_CHECK(O.size(0) == rows && O.size(1) == 128, "O must have shape (rows, 128)");

    launch_mxfp4_joint_qk_pv_true_half_stage0_perf(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_half_stage0_srcwait_pipeline_perf(at::Tensor A_fp4,
                                                                       at::Tensor A_sc,
                                                                       at::Tensor B_fp4,
                                                                       at::Tensor B_sc,
                                                                       at::Tensor V_fp4,
                                                                       at::Tensor V_sc,
                                                                       at::Tensor O) {
    const char *allow_true_half_pipeline_gap = std::getenv("FP4PV_ENABLE_TRUE_HALF_PIPELINE_GAP");
    TORCH_CHECK(allow_true_half_pipeline_gap != nullptr && std::strcmp(allow_true_half_pipeline_gap, "1") == 0,
                "mxfp4_joint_qk_pv_true_half_stage0_srcwait_pipeline_perf is guarded: session6 "
                "BF16-vs-MXFP4 true-half pipeline candidate is experimental. Set "
                "FP4PV_ENABLE_TRUE_HALF_PIPELINE_GAP=1 only for reproducibility/profiling.");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_half_stage0_srcwait_pipeline_perf(
        A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_half_stage0_barrier_trim_perf(at::Tensor A_fp4,
                                                                   at::Tensor A_sc,
                                                                   at::Tensor B_fp4,
                                                                   at::Tensor B_sc,
                                                                   at::Tensor V_fp4,
                                                                   at::Tensor V_sc,
                                                                   at::Tensor O) {
    require_mxfp4_barrier_fused_p_guard("mxfp4_joint_qk_pv_true_half_stage0_barrier_trim_perf");
    validate_mxfp4_joint_qk_pv_true_q4_stage0_inputs(A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
    launch_mxfp4_joint_qk_pv_true_half_stage0_barrier_trim_perf(
        A_fp4, A_sc, B_fp4, B_sc, V_fp4, V_sc, O);
}

void dispatch_mxfp4_joint_qk_pv_true_half_stage0_fused_p_handoff_perf(at::Tensor A_fp4,
                                                                      at::Tensor A_sc,
                                                                      at::Tensor B_fp4,
                                                                      at::Tensor B_sc,
                                                                      at::Tensor V_fp4,
                                                                      at::Tensor V_sc,
                                                                      at::Tensor O) {
    require_mxfp4_barrier_fused_p_guard("mxfp4_joint_qk_pv_true_half_stage0_fused_p_handoff_perf");
    TORCH_CHECK(false,
                "mxfp4_joint_qk_pv_true_half_stage0_fused_p_handoff_perf is disabled: "
                "session6 real P-handoff semaphore variants were either incorrect or timed out; "
                "the post-fence CTA sync remains required for correctness in this source path.");
}

void dispatch_mxfp4_joint_qk_pv_quarter_debug(at::Tensor A_fp4,
                                               at::Tensor A_sc,
                                               at::Tensor B_fp4,
                                               at::Tensor B_sc,
                                               at::Tensor O,
                                               int quarter_tile,
                                               int quarters,
                                               int debug_stage,
                                               int opt_mode) {
    CHECK_INPUT(A_fp4);
    CHECK_INPUT(A_sc);
    CHECK_INPUT(B_fp4);
    CHECK_INPUT(B_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(A_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(B_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(A_fp4.dim() == 2, "A_fp4 must have shape (rows, packed_k)");
    TORCH_CHECK(B_fp4.dim() == 2, "B_fp4 must have shape (quarter_tile * quarters, packed_k)");
    TORCH_CHECK(A_sc.dim() == 4, "A_sc must have shape (rows/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4, "B_sc must have shape (quarters, k_scale_tiles, 32, 16)");
    TORCH_CHECK(O.dim() == 2, "O must have shape (rows, 128)");
    TORCH_CHECK(quarter_tile == 128, "quarter_tile must be 128 for this diagnostic");
    TORCH_CHECK(quarters == 1 || quarters == 2 || quarters == 4, "quarters must be 1, 2, or 4");
    TORCH_CHECK(opt_mode == 0 || opt_mode == 2,
                "opt_mode must be 0(baseline) or 2(hoist A scale)");
    TORCH_CHECK(debug_stage >= 0 && debug_stage <= 14,
                "debug_stage must be in [0, 14] for joint-quarter debug; 11/12 vectorize row-major stores and 13/14 are store-only controls");
    const int rows = A_fp4.size(0);
    const int scale_k_tiles = (quarter_tile + 127) / 128;
    TORCH_CHECK(rows > 0 && rows % 128 == 0, "A rows must be a positive multiple of 128");
    TORCH_CHECK(A_fp4.size(1) * 2 >= quarter_tile, "A_fp4 packed_k is smaller than quarter_tile");
    TORCH_CHECK(B_fp4.size(0) == quarter_tile * quarters && B_fp4.size(1) * 2 >= quarter_tile,
                "B_fp4 must have shape (quarter_tile * quarters, packed_k>=quarter_tile/2)");
    TORCH_CHECK(A_sc.size(0) == rows / 128 && A_sc.size(1) >= scale_k_tiles &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (rows/128, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(B_sc.size(0) == quarters && B_sc.size(1) >= scale_k_tiles &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (quarters, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(O.size(0) == rows && O.size(1) == 128, "O must have shape (rows, 128)");

    if (quarters == 1 && opt_mode == 0) {
        if (debug_stage == 0) {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 1, 0, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        } else {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 1, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        }
    } else if (quarters == 2 && opt_mode == 0) {
        if (debug_stage == 0) {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 2, 0, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        } else {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        }
    } else if (quarters == 4 && opt_mode == 0) {
        if (debug_stage == 0) {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 4, 0, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        } else {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 4, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        }
    } else if (quarters == 1 && opt_mode == 2) {
        if (debug_stage == 0) {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 1, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        } else {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 1, 2>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        }
    } else if (quarters == 2 && opt_mode == 2) {
        if (debug_stage == 0) {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 2, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        } else {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 2, 2>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        }
    } else if (quarters == 4 && opt_mode == 2) {
        if (debug_stage == 0) {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 4, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        } else {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 4, 2>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        }
    } else {
        if (debug_stage == 0) {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 4, 2, 0>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        } else {
            launch_mxfp4_joint_qk_pv_quarter_debug<128, 4, 2>(A_fp4, A_sc, B_fp4, B_sc, O, debug_stage);
        }
    }
}

static void launch_mxfp4_joint_qk_pv_quarter_handoff_debug(at::Tensor A_fp4,
                                                           at::Tensor A_sc,
                                                           at::Tensor B_fp4,
                                                           at::Tensor B_sc,
                                                           at::Tensor O,
                                                           int control_mode) {
    using C = config_mxfp4_joint_qk_pv_quarter_debug<128, 4, 2>;
    using G = globals_mxfp4_joint_qk_pv_quarter_handoff_debug<C>;
    G g{
        reinterpret_cast<const uint8_t *>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(A_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(B_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        (int)A_fp4.size(0),
        (int)A_fp4.size(1),
        (int)B_fp4.size(1),
        (int)A_sc.size(1),
        (int)B_sc.size(1),
        control_mode
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_joint_qk_pv_quarter_handoff_debug<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_mxfp4_joint_qk_pv_quarter_handoff_debug<C>, g));
}

void dispatch_mxfp4_joint_qk_pv_quarter_handoff_debug(at::Tensor A_fp4,
                                                       at::Tensor A_sc,
                                                       at::Tensor B_fp4,
                                                       at::Tensor B_sc,
                                                       at::Tensor O,
                                                       int quarter_tile,
                                                       int quarters,
                                                       int control_mode) {
    CHECK_INPUT(A_fp4);
    CHECK_INPUT(A_sc);
    CHECK_INPUT(B_fp4);
    CHECK_INPUT(B_sc);
    CHECK_INPUT(O);
    TORCH_CHECK(A_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "A_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(B_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "B_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(A_sc.dtype() == at::ScalarType::Byte, "A_sc must be uint8 E8M0 scales");
    TORCH_CHECK(B_sc.dtype() == at::ScalarType::Byte, "B_sc must be uint8 E8M0 scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(A_fp4.dim() == 2, "A_fp4 must have shape (rows, packed_k)");
    TORCH_CHECK(B_fp4.dim() == 2, "B_fp4 must have shape (quarter_tile * quarters, packed_k)");
    TORCH_CHECK(A_sc.dim() == 4, "A_sc must have shape (rows/128, k_scale_tiles, 32, 16)");
    TORCH_CHECK(B_sc.dim() == 4, "B_sc must have shape (quarters, k_scale_tiles, 32, 16)");
    TORCH_CHECK(O.dim() == 2, "O must have shape (rows, 128)");
    TORCH_CHECK(quarter_tile == 128, "quarter_tile must be 128 for this handoff diagnostic");
    TORCH_CHECK(quarters == 4, "handoff diagnostic currently requires four quarters");
    TORCH_CHECK(control_mode >= 0 && control_mode <= 4,
                "control_mode must be 0(full CTA), 1(producer fence only), 2(issuer fence source-consumed), 3(issuer+producer fence done-consumed), or 4(issuer+producer fence source-consumed)");
    const int rows = A_fp4.size(0);
    const int scale_k_tiles = (quarter_tile + 127) / 128;
    TORCH_CHECK(rows > 0 && rows % 128 == 0, "A rows must be a positive multiple of 128");
    TORCH_CHECK(A_fp4.size(1) * 2 >= quarter_tile, "A_fp4 packed_k is smaller than quarter_tile");
    TORCH_CHECK(B_fp4.size(0) == quarter_tile * quarters && B_fp4.size(1) * 2 >= quarter_tile,
                "B_fp4 must have shape (quarter_tile * quarters, packed_k>=quarter_tile/2)");
    TORCH_CHECK(A_sc.size(0) == rows / 128 && A_sc.size(1) >= scale_k_tiles &&
                A_sc.size(2) == 32 && A_sc.size(3) == 16,
                "A_sc must have shape (rows/128, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(B_sc.size(0) == quarters && B_sc.size(1) >= scale_k_tiles &&
                B_sc.size(2) == 32 && B_sc.size(3) == 16,
                "B_sc must have shape (quarters, k_scale_tiles>=1, 32, 16)");
    TORCH_CHECK(O.size(0) == rows && O.size(1) == 128, "O must have shape (rows, 128)");

    launch_mxfp4_joint_qk_pv_quarter_handoff_debug(A_fp4, A_sc, B_fp4, B_sc, O, control_mode);
}

void dispatch_two_tile_pv_tmem_debug(
    at::Tensor P_fp4,
    at::Tensor P_sc_prepared,
    at::Tensor V_fp4,
    at::Tensor V_sc_prepared,
    at::Tensor O_after_tile0,
    at::Tensor O_after_tile1,
    bool second_tile_mm2
) {
    CHECK_INPUT(O_after_tile0);
    CHECK_INPUT(O_after_tile1);
    TORCH_CHECK(O_after_tile0.dtype() == at::ScalarType::BFloat16, "O_after_tile0 must be bfloat16");
    TORCH_CHECK(O_after_tile1.dtype() == at::ScalarType::BFloat16, "O_after_tile1 must be bfloat16");
    TORCH_CHECK(O_after_tile0.dim() == 4 && O_after_tile1.dim() == 4, "snapshot outputs must have shape (batch, seqlen, heads, 128)");

    const int batch = P_fp4.size(0);
    const int heads = P_fp4.size(1);
    const int seqlen = P_fp4.size(2);

    check_fp4pv_p_tensor(P_fp4, batch, heads, seqlen);
    check_fp4pv_p_scale_tensor(P_sc_prepared, batch, heads, seqlen);
    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_v_scale_tensor(V_sc_prepared, batch, heads, seqlen, 128);
    TORCH_CHECK(seqlen % 256 == 0, "two_tile_pv_tmem_debug currently requires sequence length divisible by 256");
    TORCH_CHECK(O_after_tile0.sizes() == O_after_tile1.sizes(), "snapshot outputs must have matching shapes");
    TORCH_CHECK(O_after_tile0.size(0) == batch && O_after_tile0.size(1) == seqlen && O_after_tile0.size(2) == heads && O_after_tile0.size(3) == 128,
                "snapshot outputs must have shape (batch, seqlen, heads, 128)");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_two_tile_pv_tmem_debug<C>;
    if (O_after_tile0.numel() > 0) {
        CUDACHECK(cudaMemset(O_after_tile0.data_ptr(), 0, O_after_tile0.nbytes()));
        CUDACHECK(cudaMemset(O_after_tile1.data_ptr(), 0, O_after_tile1.nbytes()));
    }
    G g_tile0{
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc_prepared.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O_after_tile0),
        1,
        batch,
        heads,
        seqlen
    };
    G g_tile1{
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc_prepared.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O_after_tile1),
        2,
        batch,
        heads,
        seqlen
    };
    LaunchConfig<true, false> launch_config_tile0(g_tile0.grid(), g_tile0.block(), g_tile0.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    LaunchConfig<true, false> launch_config_tile1(g_tile1.grid(), g_tile1.block(), g_tile1.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    if (second_tile_mm2) {
        CUDACHECK(cudaFuncSetAttribute(kernel_two_tile_pv_tmem_debug<C, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_tile0.dynamic_shared_memory()));
        CUDACHECK(cudaLaunchKernelEx(launch_config_tile0, kernel_two_tile_pv_tmem_debug<C, true>, g_tile0));
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaLaunchKernelEx(launch_config_tile1, kernel_two_tile_pv_tmem_debug<C, true>, g_tile1));
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
    } else {
        CUDACHECK(cudaFuncSetAttribute(kernel_two_tile_pv_tmem_debug<C, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_tile0.dynamic_shared_memory()));
        CUDACHECK(cudaLaunchKernelEx(launch_config_tile0, kernel_two_tile_pv_tmem_debug<C, false>, g_tile0));
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaLaunchKernelEx(launch_config_tile1, kernel_two_tile_pv_tmem_debug<C, false>, g_tile1));
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
    }
}

void dispatch_two_tile_correction_replay_debug(
    at::Tensor O_pre_correction,
    at::Tensor Correction,
    at::Tensor Row_sum,
    at::Tensor O_corrected
) {
    CHECK_INPUT(O_pre_correction);
    CHECK_INPUT(Correction);
    CHECK_INPUT(Row_sum);
    CHECK_INPUT(O_corrected);
    TORCH_CHECK(O_pre_correction.dtype() == at::ScalarType::Float, "O_pre_correction must be float32");
    TORCH_CHECK(Correction.dtype() == at::ScalarType::Float, "Correction must be float32");
    TORCH_CHECK(Row_sum.dtype() == at::ScalarType::Float, "Row_sum must be float32");
    TORCH_CHECK(O_corrected.dtype() == at::ScalarType::BFloat16, "O_corrected must be bfloat16");
    TORCH_CHECK(O_pre_correction.dim() == 4, "O_pre_correction must have shape (batch, seqlen, heads, 128)");
    TORCH_CHECK(Correction.dim() == 3, "Correction must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Row_sum.dim() == 3, "Row_sum must have shape (batch, heads, seqlen)");

    const int batch = O_pre_correction.size(0);
    const int seqlen = O_pre_correction.size(1);
    const int heads = O_pre_correction.size(2);
    TORCH_CHECK(O_pre_correction.size(3) == 128, "O_pre_correction must have last dim 128");
    TORCH_CHECK(Correction.size(0) == batch && Correction.size(1) == heads && Correction.size(2) == seqlen,
                "Correction must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Row_sum.sizes() == Correction.sizes(), "Row_sum must match Correction shape");
    TORCH_CHECK(O_corrected.sizes() == O_pre_correction.sizes(), "O_corrected must match O_pre_correction shape");
    TORCH_CHECK(seqlen % 128 == 0, "O_pre_correction sequence length must be divisible by 128");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_two_tile_correction_replay_debug<C>;
    G g{
        reinterpret_cast<const float *>(O_pre_correction.data_ptr()),
        reinterpret_cast<const float *>(Correction.data_ptr()),
        reinterpret_cast<const float *>(Row_sum.data_ptr()),
        reinterpret_cast<bf16 *>(O_corrected.data_ptr()),
        batch,
        heads,
        seqlen
    };

    const int row_tiles = seqlen / C::Mb;
    dim3 grid(batch * heads * row_tiles);
    dim3 block(C::Mb);
    kernel_two_tile_correction_replay_debug<C><<<grid, block>>>(g);
    CUDACHECK(cudaGetLastError());
}

void dispatch_two_tile_pv_mainpath_debug(
    at::Tensor P_fp4,
    at::Tensor P_sc_prepared,
    at::Tensor V_fp4,
    at::Tensor V_sc_prepared,
    at::Tensor O,
    int tile_begin,
    int tiles_to_run,
    bool second_tile_mm2,
    int pv_wait_count,
    bool early_tile_arrived
) {
    CHECK_INPUT(O);
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(O.dim() == 4, "O must have shape (batch, seqlen, heads, 128)");

    const int batch = P_fp4.size(0);
    const int heads = P_fp4.size(1);
    const int seqlen = P_fp4.size(2);

    check_fp4pv_p_tensor(P_fp4, batch, heads, seqlen);
    check_fp4pv_p_scale_tensor(P_sc_prepared, batch, heads, seqlen);
    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_v_scale_tensor(V_sc_prepared, batch, heads, seqlen, 128);
    TORCH_CHECK(seqlen % 256 == 0, "two_tile_pv_mainpath_debug currently requires sequence length divisible by 256");
    TORCH_CHECK(tile_begin >= 0, "two_tile_pv_mainpath_debug tile_begin must be non-negative");
    TORCH_CHECK(tiles_to_run == 1 || tiles_to_run == 2, "two_tile_pv_mainpath_debug tiles_to_run must be 1 or 2");
    TORCH_CHECK(tile_begin + tiles_to_run <= seqlen / 128, "two_tile_pv_mainpath_debug tile range must be within sequence tiles");
    TORCH_CHECK(pv_wait_count >= 1 && pv_wait_count <= tiles_to_run, "two_tile_pv_mainpath_debug pv_wait_count must be in [1, tiles_to_run]");
    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen && O.size(2) == heads && O.size(3) == 128,
                "O must have shape (batch, seqlen, heads, 128)");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_two_tile_pv_mainpath_debug<C>;
    if (O.numel() > 0) {
        CUDACHECK(cudaMemset(O.data_ptr(), 0, O.nbytes()));
    }
    G g{
        reinterpret_cast<const uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(P_sc_prepared.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        tile_begin,
        tiles_to_run,
        second_tile_mm2 ? 1 : 0,
        pv_wait_count,
        early_tile_arrived ? 1 : 0,
        batch,
        heads,
        seqlen
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_two_tile_pv_mainpath_debug<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_two_tile_pv_mainpath_debug<C>, g));
    CUDACHECK(cudaGetLastError());
}

template <typename C>
static inline dim3 fullgrid_dim_fp4pv_exp(int batch, int heads, int seqlen) {
    const int tiles_m = seqlen / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
    return dim3(total_bids);
}

template <typename C>
static inline dim3 persistent_dim_fp4pv_exp(int batch, int heads, int seqlen) {
    const int tiles_m = seqlen / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
    constexpr int PERSISTENT_GRID_CAP = fp4pv_persistent_grid_cap<C>();
    int grid_x = total_bids < PERSISTENT_GRID_CAP ? total_bids : PERSISTENT_GRID_CAP;
    if (grid_x < C::CLUSTER_SIZE) {
        grid_x = C::CLUSTER_SIZE;
    } else {
        grid_x = ((grid_x + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
    }
    return dim3(grid_x);
}

#if !defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_STAGEV_PROBE_BUILD) && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
void dispatch_fp4pv_fullgrid_exp(at::Tensor Q, at::Tensor Q_sc, at::Tensor Q_sg,
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
    LaunchConfig<true, false> launch_config(fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_fp4pv<C>, g));
}

void dispatch_forward_streaming_live_localcta(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc_prepared,
    at::Tensor O,
    at::Tensor LSE,
    bool online,
    int consumer_mode,
    int k_iter_begin,
    int k_iter_end,
    bool persistent_launch
) {
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
    const dim3 grid = persistent_launch ? persistent_dim_fp4pv_exp<C>(batch, heads, seqlen) : fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen);
    auto launch_streaming = [&]<bool ONLINE>() {
        using G = globals_fp4pv<C, ONLINE || !FP4PV_DIAG_USE_DIRECT_ROW_UPDATE>;
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

        const int dynamic_shared_memory = g.dynamic_shared_memory();
        LaunchConfig<true, false> launch_config(grid, g.block(), dynamic_shared_memory, at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        if constexpr (ONLINE) {
            CUDACHECK(cudaFuncSetAttribute(kernel_streaming_live_fp4pv<C, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
            CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_streaming_live_fp4pv<C, true>, g, consumer_mode, k_iter_begin, k_iter_end));
        } else {
            const bool use_full_range_fast_path = (k_iter_begin == 0) && (k_iter_end < 0);
            if (use_full_range_fast_path && (consumer_mode == 0)) {
                CUDACHECK(cudaFuncSetAttribute(kernel_streaming_live_fp4pv<C, false, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
                CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_streaming_live_fp4pv<C, false, 0>, g, consumer_mode, k_iter_begin, k_iter_end));
            } else if (use_full_range_fast_path && (consumer_mode == 1)) {
                CUDACHECK(cudaFuncSetAttribute(kernel_streaming_live_fp4pv<C, false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
                CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_streaming_live_fp4pv<C, false, 1>, g, consumer_mode, k_iter_begin, k_iter_end));
            } else if (use_full_range_fast_path && (consumer_mode == 2)) {
                CUDACHECK(cudaFuncSetAttribute(kernel_streaming_live_fp4pv<C, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
                CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_streaming_live_fp4pv<C, false, 2>, g, consumer_mode, k_iter_begin, k_iter_end));
            } else if (use_full_range_fast_path && (consumer_mode == 3)) {
                CUDACHECK(cudaFuncSetAttribute(kernel_streaming_live_fp4pv<C, false, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
                CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_streaming_live_fp4pv<C, false, 3>, g, consumer_mode, k_iter_begin, k_iter_end));
            } else if (use_full_range_fast_path && (consumer_mode == 4)) {
                CUDACHECK(cudaFuncSetAttribute(kernel_streaming_live_fp4pv<C, false, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
                CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_streaming_live_fp4pv<C, false, 4>, g, consumer_mode, k_iter_begin, k_iter_end));
            } else if (use_full_range_fast_path && (consumer_mode == 5)) {
                CUDACHECK(cudaFuncSetAttribute(kernel_streaming_live_fp4pv<C, false, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
                CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_streaming_live_fp4pv<C, false, 5>, g, consumer_mode, k_iter_begin, k_iter_end));
            } else {
                CUDACHECK(cudaFuncSetAttribute(kernel_streaming_live_fp4pv<C, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
                CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_streaming_live_fp4pv<C, false>, g, consumer_mode, k_iter_begin, k_iter_end));
            }
        }
    };

    if (online) {
        launch_streaming.template operator()<true>();
    } else {
        launch_streaming.template operator()<false>();
    }
}
#endif

#if !defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
static void dispatch_forward_streaming_live_mxfp4_impl(
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
    bool store_lse
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(O); CHECK_INPUT(LSE);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2, "p_quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");
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
    check_fp4pv_mxfp4_v_scale_half_contract(V_sc, batch, heads, seqlen);

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto launch_mxfp4 = [&]<typename C>(std::type_identity<C>, bool allow_persistent) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        g.v_scale_records_per_head = V_sc.size(2);

        const int tiles_m = seqlen / C::Mb;
        const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
        const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
        const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
        const int batch_heads = batch * heads;
        const bool persistent_grid_safe =
            fp4pv_online_persistent_outer_clc_multitask_reuse<C>::value ||
            fp4pv_online_persistent_outer_clc_static_stride<C>::value ||
            (!(batch_heads > 4 && seqlen >= 4096) &&
             !(batch_heads <= 4 && seqlen >= 9984));
        const bool use_persistent =
            persistent_grid_safe && allow_persistent && persistent_launch &&
            (fp4pv_force_persistent_grid<C>::value || total_bids <= 2 * C::NUM_SM);
#if defined(TK_FA4_Q4_LARGE_PROBE_BUILD) || \
    defined(TK_FA4_FULLK_WAVE_PROBE_BUILD)
        int q4_diag_controls[FP4PV_Q4_LARGE_DIAG_CONTROLS] = {-1, 0, 1, 0, 0, 0, 0, 0};
        auto q4_diag_env_int = [](const char *name, int fallback) {
            const char *value = std::getenv(name);
            return value == nullptr || std::strcmp(value, "") == 0
                ? fallback
                : std::atoi(value);
        };
        q4_diag_controls[0] = q4_diag_env_int("TK_FA4_Q4_FIXED_M_TILE", -1);
        q4_diag_controls[1] = q4_diag_env_int("TK_FA4_Q4_MAPPING_MODE", 0);
        q4_diag_controls[2] = q4_diag_env_int("TK_FA4_Q4_RECORD_ENABLE", 1);
        q4_diag_controls[3] = q4_diag_env_int("TK_FA4_Q4_LADDER_STAGE", 0);
        CUDACHECK(cudaMemcpyToSymbol(
            fp4pv_q4_large_diag_control,
            q4_diag_controls,
            sizeof(q4_diag_controls)));
        dim3 grid = use_persistent
            ? persistent_dim_fp4pv_exp<C>(batch, heads, seqlen)
            : fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen);
        const int q4_grid_override = q4_diag_env_int("TK_FA4_Q4_GRID_BLOCKS", 0);
        if (q4_grid_override > 0) {
            grid.x = q4_grid_override;
        }
#else
        const dim3 grid = use_persistent
            ? persistent_dim_fp4pv_exp<C>(batch, heads, seqlen)
            : fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen);
        if constexpr (fp4pv_require_one_wave_persistent<C>::value) {
            TORCH_CHECK(allow_persistent && persistent_launch && use_persistent &&
                        grid.x == static_cast<unsigned>(C::NUM_SM),
                        "dedicated persistent route requires one-wave persistent launch");
        }
#endif

#if (defined(TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) && \
     TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) || \
    (defined(TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD) && \
     TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD)
        if constexpr (
            fp4pv_track_b_stage2_cap2_target<C>::value ||
            fp4pv_track_b_stage2_fullgrid_control<C>::value) {
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

            const int logical_tasks = batch * heads * (seqlen / C::Mb);
            constexpr int grid_cap = 2 * C::NUM_SM;
            TORCH_CHECK(
                total_bids == logical_tasks,
                "Track-B cap2 logical-task calculation drifted from one M128 "
                "query tile per task");

            if constexpr (is_target) {
                const int expected_grid =
                    logical_tasks < grid_cap ? logical_tasks : grid_cap;
                TORCH_CHECK(
                    allow_persistent && persistent_launch &&
                        persistent_grid_safe && use_persistent,
                    "Track-B cap2 target requires an actual persistent launch; "
                    "full-grid fallback is forbidden");
                TORCH_CHECK(
                    grid.x == static_cast<unsigned>(expected_grid),
                    "Track-B cap2 target selected an unexpected physical grid");
                if (logical_tasks > grid_cap) {
                    TORCH_CHECK(
                        grid.x == static_cast<unsigned>(grid_cap),
                        "Track-B cap2 saturated launch must use exactly 304 CTAs");
                }
            } else {
                TORCH_CHECK(
                    !allow_persistent && !persistent_launch && !use_persistent,
                    "Track-B full-grid control forbids persistent launch state");
                TORCH_CHECK(
                    grid.x == static_cast<unsigned>(logical_tasks),
                    "Track-B full-grid control must launch one CTA per logical "
                    "M128 task");
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
        }
#endif
        const int dynamic_shared_memory = g.dynamic_shared_memory();
        LaunchConfig<true, false> launch_config(grid, g.block(), dynamic_shared_memory, at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaFuncSetAttribute(
            kernel_streaming_live_fp4pv<C, true, -1, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        int diagnostic_consumer_mode = 0;
#if TK_FA4_ENABLE_MXFP4_FWD_PCHAIN_STAMPS
        diagnostic_consumer_mode = TK_FA4_MXFP4_FWD_PCHAIN_TARGET_IDX;
        if (const char *target = std::getenv("TK_FA4_PCHAIN_RUNTIME_TARGET_IDX")) {
            diagnostic_consumer_mode = std::atoi(target);
        }
#endif
        CUDACHECK(cudaLaunchKernelEx(
            launch_config,
            kernel_streaming_live_fp4pv<C, true, -1, false, true>,
            g,
            diagnostic_consumer_mode,
            0,
            -1));
    };
#if TK_FA4_DVO64_TWO_PASS_BUILD
    auto launch_mxfp4_dvo64_two_pass = [&]<typename C>(std::type_identity<C>) {
        static_assert(C::Dvo == 64, "DVO64 two-pass launch requires Dvo=64");
        static_assert(fp4pv_online_dvo64_two_pass<C>::value,
                      "DVO64 two-pass launch requires explicit config ownership");
        using G = globals_fp4pv_mxfp4_dv<C, true>;

        const dim3 grid = fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen);
#ifdef TK_FA4_Q4_LARGE_PROBE_BUILD
        int q4_diag_controls[FP4PV_Q4_LARGE_DIAG_CONTROLS] = {-1, 0, 1, 0, 0, 0, 0, 0};
        CUDACHECK(cudaMemcpyToSymbol(
            fp4pv_q4_large_diag_control,
            q4_diag_controls,
            sizeof(q4_diag_controls)));
#endif
        for (int dvo_half = 0; dvo_half < 2; ++dvo_half) {
            G g{
                kittens::py::tensor_to_gl<typename G::q_gl>(Q),
                tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
                tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
                kittens::py::tensor_to_gl<typename G::k_gl>(K),
                tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
                tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
                kittens::py::tensor_to_gl<typename G::v_fp4_gl>(
                    V_fp4, batch, heads * 2, C::Dvo, seqlen / 2),
                reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
                reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
                kittens::py::tensor_to_gl<typename G::o_gl>(
                    O, batch, seqlen, heads * 2, C::Dvo),
                kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
                1.0f,
                p_quant_mode
            };
            g.disable_lse_store = (store_lse && dvo_half == 0) ? 0 : 1;
            g.dvo_half = dvo_half;
            g.v_scale_records_per_head = V_sc.size(2);

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
        }
    };

    auto launch_mxfp4_dvo64_lean2wg = [&]<typename C>(
        std::type_identity<C>, bool force_one_resident = false) {
        static_assert(C::Dvo == 64 && C::TOTAL_WGS == 2,
                      "lean DVO64 launch requires a two-WG D64 config");
        using G = globals_mxfp4_dvo64_lean2wg<C>;
        const dim3 grid(batch * heads * (seqlen / C::Mb));
#if TK_FA4_DVO64_LEAN_TRACE_BUILD
        const char *trace_env = std::getenv("TK_FA4_DVO64_LEAN_TRACE_ENABLE");
        const int trace_enabled =
            trace_env != nullptr && std::strcmp(trace_env, "0") != 0;
#endif
        constexpr bool one_walk_register_accum = [] {
            if constexpr (requires {
                C::DVO64_ONE_WALK_REGISTER_ACCUM;
            }) {
                return C::DVO64_ONE_WALK_REGISTER_ACCUM;
            }
            return false;
        }();
        const int output_passes = one_walk_register_accum ? 1 : 2;
        for (int dvo_half = 0; dvo_half < output_passes; ++dvo_half) {
            G g{
                kittens::py::tensor_to_gl<typename G::q_gl>(Q),
                kittens::py::tensor_to_gl<typename G::k_gl>(K),
                kittens::py::tensor_to_gl<typename G::v_fp4_gl>(
                    V_fp4, batch, heads * 2, C::Dvo, seqlen / 2),
                reinterpret_cast<const uint8_t *>(Q.data_ptr()),
                reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
                reinterpret_cast<const float *>(Q_sg.data_ptr()),
                reinterpret_cast<const uint8_t *>(K.data_ptr()),
                reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
                reinterpret_cast<const float *>(K_sg.data_ptr()),
                reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
                reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
                reinterpret_cast<bf16 *>(O.data_ptr()),
                reinterpret_cast<float *>(LSE.data_ptr()),
                batch,
                heads,
                seqlen,
                dvo_half,
                (store_lse && (one_walk_register_accum || dvo_half == 0))
                    ? 0 : 1
#if TK_FA4_DVO64_LEAN_TRACE_BUILD
                , trace_enabled
#endif
            };
            const int dynamic_shared_memory = force_one_resident
                ? 128 * 1024 : g.dynamic_shared_memory();
            LaunchConfig<true, false> launch_config(
                grid, g.block(), dynamic_shared_memory,
                at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
            CUDACHECK(cudaFuncSetAttribute(
                kernel_mxfp4_dvo64_lean2wg<C>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                dynamic_shared_memory));
            CUDACHECK(cudaLaunchKernelEx(
                launch_config,
                kernel_mxfp4_dvo64_lean2wg<C>,
                g));
        }
    };

    auto launch_mxfp4_dvo64_pair2wg = [&]<typename C>(
        std::type_identity<C>) {
        static_assert(C::Dvo == 64 && C::TOTAL_WGS == 2 &&
                      C::CLUSTER_SIZE == 2,
                      "paired DVO64 launch requires two WGs and cluster-2");
        using G = globals_mxfp4_dvo64_lean2wg<C>;
        const dim3 grid(2 * batch * heads * (seqlen / C::Mb));
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(
                V_fp4, batch, heads * 2, C::Dvo, seqlen / 2),
            reinterpret_cast<const uint8_t *>(Q.data_ptr()),
            reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
            reinterpret_cast<const float *>(Q_sg.data_ptr()),
            reinterpret_cast<const uint8_t *>(K.data_ptr()),
            reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
            reinterpret_cast<const float *>(K_sg.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            reinterpret_cast<bf16 *>(O.data_ptr()),
            reinterpret_cast<float *>(LSE.data_ptr()),
            batch,
            heads,
            seqlen,
            0,
            store_lse ? 0 : 1
#if TK_FA4_DVO64_LEAN_TRACE_BUILD
            , 0
#endif
        };
        const int dynamic_shared_memory = g.dynamic_shared_memory();
        LaunchConfig<true, false> launch_config(
            grid, g.block(), dynamic_shared_memory,
            at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaFuncSetAttribute(
            kernel_mxfp4_dvo64_lean2wg<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        CUDACHECK(cudaLaunchKernelEx(
            launch_config,
            kernel_mxfp4_dvo64_lean2wg<C>,
            g));
    };
#endif
    [[maybe_unused]] auto launch_mxfp4_k256_static = [&]<typename C>(std::type_identity<C>, bool allow_persistent) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;

        const int tiles_m = seqlen / C::Mb;
        const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
        const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
        const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
        const int batch_heads = batch * heads;
        const bool persistent_grid_safe =
            !(batch_heads > 4 && seqlen >= 4096) &&
            !(batch_heads <= 4 && seqlen >= 9984);
        const bool use_persistent =
            persistent_grid_safe && allow_persistent && persistent_launch &&
            (fp4pv_force_persistent_grid<C>::value || total_bids <= 2 * C::NUM_SM);
        const dim3 grid = use_persistent ? persistent_dim_fp4pv_exp<C>(batch, heads, seqlen) : fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen);
        const int dynamic_shared_memory = g.dynamic_shared_memory_k256();
        LaunchConfig<true, false> launch_config(grid, g.block(), dynamic_shared_memory, at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        CUDACHECK(cudaFuncSetAttribute(
            kernel_streaming_live_fp4pv<C, true, 178, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        CUDACHECK(cudaLaunchKernelEx(
            launch_config,
            kernel_streaming_live_fp4pv<C, true, 178, false, true>,
            g,
            178,
            0,
            -1));
    };
    [[maybe_unused]] auto launch_mxfp4_dual_query_halfgrid = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        // The adjacent-pair route uses the same packed V-scale contract as
        // launch_mxfp4.  Preserve the per-head record stride explicitly;
        // globals default to one record while this route supplies two.
        g.v_scale_records_per_head = V_sc.size(2);

        const int tiles_m = seqlen / C::Mb;
        const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
        const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
        const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
        const int query_pairs_per_head = (tiles_m + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
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
    [[maybe_unused]] auto launch_mxfp4_dual_query_concurrent = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;

        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory = fp4pv_dual_query_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_dual_query_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_dual_query_concurrent_halfgrid<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_dual_query_two_score_overlay =
        [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory =
            fp4pv_dual_query_two_score_overlay_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_dual_query_two_score_overlay<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_dual_query_two_score_overlay<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_two_n64 = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE), 1.0f, p_quant_mode};
        g.disable_lse_store = store_lse ? 0 : 1;
        const dim3 grid(batch * heads * (((seqlen / C::Mb) + 1) / 2));
        constexpr int dynamic_shared_memory = fp4pv_two_n64_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(kernel_fp4pv_two_n64_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        kernel_fp4pv_two_n64_concurrent_halfgrid<C><<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
            at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_one_n128 = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory = fp4pv_one_n128_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_one_n128_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_one_n128_concurrent_halfgrid<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_one_n128_bf16ret = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory =
            fp4pv_one_n128_bf16ret_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_one_n128_bf16ret_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_one_n128_bf16ret_concurrent_halfgrid<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_one_n128_fp16ret = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory =
            fp4pv_one_n128_fp16ret_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_one_n128_fp16ret_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_one_n128_fp16ret_concurrent_halfgrid<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_one_n128_fp16scaledret = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory =
            fp4pv_one_n128_fp16scaledret_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_one_n128_fp16scaledret_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_one_n128_fp16scaledret_concurrent_halfgrid<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_one_n128_evac = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory =
            fp4pv_one_n128_evac_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_one_n128_evac_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_one_n128_evac_concurrent_halfgrid<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_one_n128_3wg = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
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
    };
    [[maybe_unused]] auto launch_mxfp4_one_n128_smem_evac = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory =
            fp4pv_one_n128_smem_evac_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_one_n128_smem_evac_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_one_n128_smem_evac_concurrent_halfgrid<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    [[maybe_unused]] auto launch_mxfp4_one_n128_no_retention = [&]<typename C>(std::type_identity<C>) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int query_pairs_per_head = ((seqlen / C::Mb) + 1) / 2;
        const dim3 grid(batch * heads * query_pairs_per_head);
        constexpr int dynamic_shared_memory =
            fp4pv_one_n128_no_retention_smem_layout<C>::bytes();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_one_n128_no_retention_concurrent_halfgrid<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_one_n128_no_retention_concurrent_halfgrid<C>
            <<<grid, dim3(C::NUM_THREADS), dynamic_shared_memory,
               at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    auto launch_fused_stream = [&]<typename C, bool DYNAMIC_P_SCALE = false, bool DIRECT_P_SCALE_TMEM = false>(
        std::type_identity<C>,
        std::bool_constant<DYNAMIC_P_SCALE> = {},
        std::bool_constant<DIRECT_P_SCALE_TMEM> = {}
    ) {
        using G = globals_fp4pv_mxfp4_dv<C, true>;
        G g{
            kittens::py::tensor_to_gl<typename G::q_gl>(Q),
            tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
            tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
            kittens::py::tensor_to_gl<typename G::k_gl>(K),
            tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
            tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
            kittens::py::tensor_to_gl<typename G::v_fp4_gl>(V_fp4),
            reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
            kittens::py::tensor_to_gl<typename G::o_gl>(O),
            kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
            1.0f,
            p_quant_mode
        };
        g.disable_lse_store = store_lse ? 0 : 1;
        const int tiles_m = seqlen / C::Mb;
        const dim3 grid(tiles_m * heads * batch, 1, 1);
        const int dynamic_shared_memory = g.dynamic_shared_memory();
        CUDACHECK(cudaFuncSetAttribute(
            kernel_fp4pv_fused_stream<C, DYNAMIC_P_SCALE, DIRECT_P_SCALE_TMEM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
        kernel_fp4pv_fused_stream<C, DYNAMIC_P_SCALE, DIRECT_P_SCALE_TMEM><<<grid, dim3(C::NUM_THREADS, 1, 1), dynamic_shared_memory, at::cuda::getCurrentCUDAStream()>>>(g);
        CUDACHECK(cudaGetLastError());
    };
    const char *fwd_config = std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
#if TK_FA4_HYBRID_QUANT_READER_PROXY_ONLY_BUILD
    TORCH_CHECK(
        fwd_config != nullptr &&
            std::strcmp(
                fwd_config,
                "real_fwd_native_group2_fp4qk_local_group1_fp4pv_quant_reader_proxy") == 0,
        "candidate-only quant-reader-proxy extension requires its exact route");
    TORCH_CHECK(
        !persistent_launch,
        "candidate-only quant-reader-proxy extension requires a full-grid launch");
    const int q_tiles =
        seqlen /
        config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_quant_reader_proxy::Mb;
    TORCH_CHECK(
        (q_tiles & 1) == 0,
        "hybrid quant-reader-proxy route requires an even M128 query-tile count");
    launch_mxfp4(
        std::type_identity<
            config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_quant_reader_proxy>{},
        false);
    return;
#elif TK_FA4_HYBRID_ISSUE_PROXY_ONLY_BUILD
    TORCH_CHECK(
        fwd_config != nullptr &&
            std::strcmp(
                fwd_config,
                "real_fwd_native_group2_fp4qk_local_group1_fp4pv_issue_proxy") == 0,
        "candidate-only issue-proxy extension requires its exact route");
    TORCH_CHECK(
        !persistent_launch,
        "candidate-only issue-proxy extension requires a full-grid launch");
    const int q_tiles =
        seqlen /
        config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_issue_proxy::Mb;
    TORCH_CHECK(
        (q_tiles & 1) == 0,
        "hybrid issue-proxy route requires an even M128 query-tile count");
    launch_mxfp4(
        std::type_identity<
            config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_issue_proxy>{},
        false);
    return;
#else
    const bool fwd_config_auto =
        fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
        std::strcmp(fwd_config, "auto") == 0;
#if TK_FA4_REAL_FWD_TIMELINE_PROBE_BUILD
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_timeline_control") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_timeline_control>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_timeline_trace") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_timeline_trace>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_readiness_probe") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_readiness_probe>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_control") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_control>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_pv_control") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config,
                    "real_fwd_native_group2_fp4qk_fp4pv_lean") == 0) {
        const int q_tiles = seqlen / config_fp4pv_native_group2_fp4qk_fp4pv_lean::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 full-FP4 repro currently requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<config_fp4pv_native_group2_fp4qk_fp4pv_lean>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_score_pair_credit") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 score-pair credit requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_score_pair_credit_diag") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit_diag::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 score-pair diagnostic requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_credit_diag>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_downstream_diag") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_downstream_diag::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 downstream diagnostic requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_downstream_diag>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_split_final_stats") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_split_final_stats::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 split-final-stats route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_split_final_stats>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_exact_lse") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 exact-LSE route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_exact_lse_threshold") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse_threshold::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 thresholded exact-LSE route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_exact_lse_threshold>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_score_load_fence") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_score_load_fence::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 score-load-fence route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_score_load_fence>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_score_pair_count8") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_count8::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 count8 score-credit route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_count8>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_score_pair_wg_rendezvous") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_wg_rendezvous::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 score-rendezvous route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_score_pair_wg_rendezvous>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_q_scale_three_page") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_q_scale_three_page::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 three-page Q-scale route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_q_scale_three_page>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_k_scale_three_page") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 three-page K-scale route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 physical-K-tile K3 route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_local_group1_fp4pv") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_local_group1_fp4pv::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "hybrid group2-QK/local-group1-PV route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_local_group1_fp4pv>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_local_group1_fp4pv_issue_proxy") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_issue_proxy::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "hybrid issue-proxy route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_issue_proxy>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_local_group1_fp4pv_quant_reader_proxy") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_quant_reader_proxy::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "hybrid quant-reader-proxy route requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_local_group1_fp4pv_quant_reader_proxy>{},
            false);
        return;
    }
#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_q_scale_three_page_selected_trace") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_q_scale_three_page_selected_trace::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 selected-row A trace requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_q_scale_three_page_selected_trace>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_k_scale_three_page_selected_trace") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_selected_trace::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 selected-row K3 trace requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_selected_trace>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles_selected_trace") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles_selected_trace::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 physical-K-tile selected-row trace requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_k_scale_three_page_physical_k_tiles_selected_trace>{},
            false);
        return;
    }
#endif
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_native_group2_fp4qk_fp4pv_split_final_stats_diag") == 0) {
        const int q_tiles =
            seqlen /
            config_fp4pv_native_group2_fp4qk_fp4pv_split_final_stats_diag::Mb;
        TORCH_CHECK((q_tiles & 1) == 0,
                    "native group2 split-final-stats diagnostic requires an even M128 query-tile count");
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_native_group2_fp4qk_fp4pv_split_final_stats_diag>{},
            false);
        return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_taskdone6") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v25_g_persistent") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v25_g_persistent>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_skip_reused") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_reused>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_taskdone6_skip_reused") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_skip_reused>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_taskdone6_skip_reused_epoch") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_skip_reused_epoch>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_taskdone6_fullskip") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_fullskip>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_taskdone6_fullskip_timeout") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_taskdone6_fullskip_timeout>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_second_task_diag") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_second_task_diag>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_skip_output_wait_diag") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_diag>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_reinit") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_reinit>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_safe") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_pv_control_persistent_skip_output_wait_nodiag_safe>{}, true); return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_pv_control_adjacent_pair_halfgrid_reinit") == 0) {
        TORCH_CHECK(persistent_launch,
                    "adjacent-pair half-grid route requires persistent_launch=true");
        TORCH_CHECK(seqlen == 4096,
                    "adjacent-pair half-grid route is currently guarded for S4096 only");
        launch_mxfp4_dual_query_halfgrid(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_control_adjacent_pair_halfgrid_reinit>{});
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_high_corr_pv_control_adjacent_pair_halfgrid_safe") == 0) {
        TORCH_CHECK(persistent_launch,
                    "adjacent-pair half-grid safe route requires persistent_launch=true");
        TORCH_CHECK(seqlen == 4096,
                    "adjacent-pair half-grid safe route is currently guarded for S4096 only");
        launch_mxfp4_dual_query_halfgrid(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_control_adjacent_pair_halfgrid_safe>{});
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_corr_pv_control") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_corr_pv_control>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config, "real_fwd_high_corr_mapping_control") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_mapping_control>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config, "real_fwd_long_corr_mapping_control") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_long_corr_mapping_control>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_high_corr_wait_only_lookahead") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_wait_only_lookahead>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "real_fwd_long_corr_wait_only_lookahead") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_long_corr_wait_only_lookahead>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_wait_after_pv") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_high_corr_wait_after_pv>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_corr_wait_after_pv") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_corr_wait_after_pv>{},
            false);
        return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck>{}, false);
        return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck") == 0) {
        launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck>{}, false);
        return;
    }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v15") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v15>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v15") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v15>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v16") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v16>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v16") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v16>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v18") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v18>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v18") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v18>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v19") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v19>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v19") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v19>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v20") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v20>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v20") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v20>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v22_a") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v22_a>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v22_a") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v22_a>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v22_b") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v22_b>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v22_b") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v22_b>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v23_c") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v23_c>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v23_c") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v23_c>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v23_d") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v23_d>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v23_d") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v23_d>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v24_e") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v24_e>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v24_e") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v24_e>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v24_f") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v24_f>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v24_f") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v24_f>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_high_corr_precheck_v25_g") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_high_corr_precheck_v25_g>{}, false); return; }
    if (fwd_config != nullptr && std::strcmp(fwd_config, "real_fwd_long_corr_precheck_v25_g") == 0) { launch_mxfp4(std::type_identity<config_fp4pv_real_fwd_long_corr_precheck_v25_g>{}, false); return; }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_lookahead") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_lookahead>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_corr_lookahead") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_long_corr_lookahead>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_pv_cycle") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_high_corr_pv_cycle>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_corr_pv_cycle") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_corr_pv_cycle>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config, "real_fwd_high_corr_pv_cycle_balanced") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_cycle_balanced>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config, "real_fwd_long_corr_pv_cycle_balanced") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_long_corr_pv_cycle_balanced>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_pv_cycle_q") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_cycle_masked<1>>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_pv_cycle_i") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_cycle_masked<2>>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_pv_cycle_o") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_cycle_masked<4>>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_pv_cycle_qi") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_cycle_masked<3>>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_pv_cycle_qo") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_cycle_masked<5>>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_corr_pv_cycle_io") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_real_fwd_high_corr_pv_cycle_masked<6>>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_store_ceiling") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_high_store_ceiling>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_store_ceiling") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_store_ceiling>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_pscale_ceiling") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_high_pscale_ceiling>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_pscale_ceiling") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_pscale_ceiling>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_pchain") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_high_pchain>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_pchain") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_pchain>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_headlazy") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_high_headlazy>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_headlazy") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_headlazy>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_high_pchain_headlazy") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_high_pchain_headlazy>{},
            false);
        return;
    }
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "real_fwd_long_pchain_headlazy") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_real_fwd_long_pchain_headlazy>{},
            false);
        return;
    }
#endif
    const char *p_bypass_env = std::getenv("TK_FA4_MXFP4_P_BYPASS_MATERIALIZATION");
    const bool p_bypass_materialization =
        p_bypass_env != nullptr &&
        std::strcmp(p_bypass_env, "") != 0 &&
        std::strcmp(p_bypass_env, "0") != 0 &&
        std::strcmp(p_bypass_env, "false") != 0 &&
        std::strcmp(p_bypass_env, "False") != 0 &&
        std::strcmp(p_bypass_env, "no") != 0 &&
        std::strcmp(p_bypass_env, "off") != 0;

#if TK_FA4_DVO64_TWO_PASS_BUILD
    if (fwd_config != nullptr &&
        std::strcmp(fwd_config, "q4_dvo128_parent_ex2e16pc") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_dvo128_parent_fullk_ex2e16pc>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config, "q4_dvo64_two_pass_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_two_pass(
            std::type_identity<config_fp4pv_stage2_ex2_alu_dvo64_two_pass<
                16, 128, 128, 192, 200, 56, 112, 1>>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config, "q4_dvo64_two_pass_ex2e16pc_q128_o48_p80") == 0) {
        launch_mxfp4_dvo64_two_pass(
            std::type_identity<config_fp4pv_stage2_ex2_alu_dvo64_two_pass<
                16, 128, 128, 192, 128, 48, 80, 1>>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config, "q4_dvo64_two_pass_ex2e16pc_q128_o48_p64") == 0) {
        launch_mxfp4_dvo64_two_pass(
            std::type_identity<config_fp4pv_stage2_ex2_alu_dvo64_two_pass<
                16, 128, 128, 192, 128, 48, 64, 1>>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config, "q4_dvo64_two_pass_ex2e16pc_q80_o80_p80") == 0) {
        launch_mxfp4_dvo64_two_pass(
            std::type_identity<config_fp4pv_stage2_ex2_alu_dvo64_two_pass<
                16, 128, 128, 192, 80, 80, 80, 1>>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config, "q4_dvo64_two_pass_ex2e16pc_static80") == 0) {
        launch_mxfp4_dvo64_two_pass(
            std::type_identity<config_fp4pv_stage2_ex2_alu_dvo64_two_pass_static_regs<
                16, 128, 128, 192, 128, 48, 64, 1>>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config, "q4_dvo64_lean2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<config_fp4pv_dvo64_lean2wg_ex2e16pc>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config,
                           "q4_dvo64_lean2wg_ex2e16pc_force1") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<config_fp4pv_dvo64_lean2wg_ex2e16pc>{}, true);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config,
                           "q4_dvo64_pair2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_pair2wg(
            std::type_identity<config_fp4pv_dvo64_pair2wg_ex2e16pc>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config,
                           "q4_dvo64_onewalk_reg2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<
                config_fp4pv_dvo64_onewalk_reg2wg_ex2e16pc>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_dvo64_onewalk_fp16_coldswap_2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<
                config_fp4pv_dvo64_onewalk_fp16_coldswap_2wg_ex2e16pc>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_dvo64_onewalk_bf16_coldswap_2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<
                config_fp4pv_dvo64_onewalk_bf16_coldswap_2wg_ex2e16pc>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_dvo64_onewalk_overlap32_2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<
                config_fp4pv_dvo64_onewalk_overlap32_2wg_ex2e16pc>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_dvo64_onewalk_overlap32_l2lpt_2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<
                config_fp4pv_dvo64_onewalk_overlap32_l2lpt_2wg_ex2e16pc>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_dvo64_onewalk_overlap32_queuepv_2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<
                config_fp4pv_dvo64_onewalk_overlap32_queuepv_2wg_ex2e16pc>{});
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_dvo64_onewalk_deferpv_2wg_ex2e16pc") == 0) {
        launch_mxfp4_dvo64_lean2wg(
            std::type_identity<
                config_fp4pv_dvo64_onewalk_deferpv_2wg_ex2e16pc>{});
        return;
    }
#endif

#if TK_FA4_REGISTER_OUTPUT_PROBE_BUILD
    if (fwd_config != nullptr &&
        std::strcmp(
            fwd_config,
            "q4_fullk_register_output_2score_probe") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_fullk_register_output_4wg_2score>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_register_smem_hybrid_probe") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_fullk_register_smem_hybrid>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_register_output_3score_probe") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_fullk_register_output_4wg_3score>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_register_output_progressive_n64_probe") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_fullk_register_output_progressive_n64>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_register_output_progressive_n32_probe") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_fullk_register_output_progressive_n32>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_register_output_progressive_n64_diag_probe") ==
                   0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_fullk_register_output_progressive_n64_diag>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_register_output_progressive_n64_producer_diag_probe") ==
                   0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_fullk_register_output_progressive_n64_producer_diag>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_register_output_tmem_control_probe") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_fullk_register_output_tmem_control>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_fp32_smem_output_probe") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_fullk_fp32_smem_output>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(fwd_config, "q4_headlazy_fullk_probe") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<
                    16, 128, 128, 192, 128, 200, 56, 112, 1>>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_single_scale_spare_probe") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_single_scale_spare<
                    16, 128, 128, 192, 128, 200, 56, 112, 1>>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_scale_pages_serial_probe") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_fullk_scale_pages_serial>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_scale_pages_overlap_probe") == 0) {
        launch_mxfp4(
            std::type_identity<config_fp4pv_fullk_scale_pages_overlap>{},
            false);
        return;
    } else if (fwd_config != nullptr &&
               std::strcmp(
                   fwd_config,
                   "q4_fullk_scale_pages_lazy_overlap_probe") == 0) {
        launch_mxfp4(
            std::type_identity<
                config_fp4pv_fullk_scale_pages_lazy_overlap>{},
            false);
        return;
    }
#endif

    if (heads == 1) {
#ifdef TK_FA4_STAGEV_PROBE_BUILD
#if TK_FA4_TMEM_COLD_OUTPUT_BUILD
        if (fwd_config != nullptr &&
            (std::strcmp(fwd_config, "q4_headlazy_fullk_probe") == 0 ||
             std::strcmp(fwd_config, "auto") == 0 ||
             std::strcmp(fwd_config, "") == 0)) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (
            fwd_config != nullptr &&
            std::strcmp(fwd_config, "q4_fullk_single_scale_spare_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_single_scale_spare<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (
            fwd_config != nullptr &&
            std::strcmp(fwd_config, "q4_fullk_fp16_cold_output_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_fp16_cold_output<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else {
            TORCH_CHECK(
                false,
                "H=1 cold-output probe build supports only full-K headlazy, single-scale spare, and FP16 cold-output routes");
        }
#elif defined(TK_FA4_DUAL_QUERY_PROBE_BUILD) && TK_FA4_DEEP_JOINT_STAGE >= 2
        if (fwd_config != nullptr &&
            std::strcmp(fwd_config, "deep_joint_fullk_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc>{},
                false);
        } else if (
            fwd_config != nullptr &&
            std::strcmp(
                fwd_config,
                "dualquery_fullk_parent_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_dual_query_parent_fullk_ex2e16pc>{},
                false);
        } else {
            TORCH_CHECK(
                false,
                "H=1 dual-query probe build supports only corrected full-K repaired and parent routes");
        }
#else
        TORCH_CHECK(false, "TK_FA4_STAGEV_PROBE_BUILD only supports multi-head MXFP4 forward probes");
#endif
#else
        if (fwd_config_auto) {
            if (batch == 1 && seqlen >= 16384) {
                launch_mxfp4(std::type_identity<config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split<16, 128, 128, 192, 128, 152, 48, 112, 1>>{}, false);
            } else if (p_bypass_materialization) {
                launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
            } else {
                launch_mxfp4(std::type_identity<config_fp4pv_3wg_async_pscale<128, 128, 192, 128, 2>>{}, true);
            }
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_qkscfix_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_qkscfix_nodirectrescale_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_qkscfix_nodirectrescale_dualaccum_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_rte_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_rte_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p104_o48") == 0) {
            if (seqlen >= 8192) {
                launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_leader_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
            } else {
                launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 144, 48, 120, 1>>{}, true);
            }
        } else if (std::strcmp(fwd_config, "scorederived_ex2e16pc_split2wg_full_pstage2_q152_p112_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split<16, 128, 128, 192, 128, 152, 48, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "scorederived_ex2e16pc_split2wg_k64_pstage2_q152_p112_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split_k64<16, 128, 128, 192, 128, 152, 48, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "shapehalf_k256_scaffold_ex2e16pc_q200_p112_o56") == 0) {
            launch_mxfp4_k256_static(std::type_identity<config_fp4pv_stage2_ex2_alu_true_half_k256<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "shapehalf_k256_pipe2_ex2e16pc_q200_p112_o56") == 0) {
            launch_mxfp4_k256_static(std::type_identity<config_fp4pv_stage2_ex2_alu_true_half_k256_pipe2<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else {
            TORCH_CHECK(false, "unsupported TK_FA4_FP4PV_FWD_CONFIG for H=1 MXFP4 forward experiment: ", fwd_config);
        }
#endif
    } else {
#ifdef TK_FA4_STAGEV_PROBE_BUILD
#ifdef TK_FA4_Q4_HEAD_INTERLEAVED_BUILD
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "q4_head_interleaved") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (
            std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_kvstage3") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_kvstage3<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_head_interleaved_kvstage3") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_kvstage3<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_qk_two_ahead") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_qk_two_ahead<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_head_interleaved_qk_two_ahead") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_qk_two_ahead<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_lazy_output_rescale") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_lazy_output_rescale<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_head_interleaved_lazy_output_rescale") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_Q4_HEAD_INTERLEAVED_BUILD supports only the retained e16pc q4 candidates");
        }
#elif TK_FA4_TMEM_COLD_OUTPUT_BUILD
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "q4_headlazy_fullk_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_single_scale_spare_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_single_scale_spare<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_fp16_cold_output_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_fullk_fp16_cold_output<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
#if TK_FA4_DEEP_JOINT_STAGE >= 2
        } else if (std::strcmp(fwd_config, "deep_joint_fullk_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc>{},
                false);
#endif
        } else {
            throw std::invalid_argument(
                "TK_FA4_TMEM_COLD_OUTPUT_BUILD supports only full-K headlazy, single-scale spare, FP16 cold-output, and corrected deep-joint");
        }
#elif defined(TK_FA4_H32_HEADLAZY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "q4_headlazy_fullk_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "q4_headlazy_unsafe_control") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
#if TK_FA4_DEEP_JOINT_STAGE >= 2
        } else if (std::strcmp(fwd_config, "deep_joint_fullk_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc>{},
                false);
#endif
        } else {
            throw std::invalid_argument(
                "TK_FA4_H32_HEADLAZY_PROBE_BUILD supports only full-K headlazy, unsafe headlazy control, and corrected deep-joint");
        }
#elif defined(TK_FA4_SATURATED_HEAD_GROUP_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "q4_headlazy_fullk_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_group2_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_grouped_lazy_output_rescale_fullk<2, 16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_group4_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_grouped_lazy_output_rescale_fullk<4, 16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_group8_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_grouped_lazy_output_rescale_fullk<8, 16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_group16_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_grouped_lazy_output_rescale_fullk<16, 16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
#if TK_FA4_DEEP_JOINT_STAGE >= 2
        } else if (std::strcmp(fwd_config, "deep_joint_fullk_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc>{},
                false);
#endif
        } else {
            throw std::invalid_argument(
                "TK_FA4_SATURATED_HEAD_GROUP_PROBE_BUILD supports only full-K headlazy baseline/group2/group4/group8/group16 and corrected deep-joint");
        }
#elif defined(TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) && \
    TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD
        if (fwd_config != nullptr &&
            std::strcmp(
                fwd_config,
                "real_fwd_track_b_stage2_fullk_cap2_saturated") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_track_b_stage2_fullk_cap2_saturated>{},
                true);
        } else if (
            fwd_config != nullptr &&
            std::strcmp(
                fwd_config,
                "real_fwd_track_b_stage2_fullk_fullgrid_control") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_track_b_stage2_fullk_fullgrid_control>{},
                false);
        } else {
            throw std::invalid_argument(
                "TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD requires an exact "
                "Track-B cap2 target or full-grid-control route; "
                "auto/default selection is forbidden");
        }
#elif defined(TK_FA4_SATURATED_PERSISTENT_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "q4_headlazy_fullk_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_persistent_cap2_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent<2, 16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_persistent_cap2_late_rescale_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent_late_rescale_wait<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_persistent_cap2_vstage3_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent_vstage3<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_persistent_cap1_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk_persistent<1, 16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
#if TK_FA4_DEEP_JOINT_STAGE >= 2
        } else if (std::strcmp(fwd_config, "deep_joint_fullk_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc>{},
                false);
#endif
        } else {
            throw std::invalid_argument(
                "TK_FA4_SATURATED_PERSISTENT_PROBE_BUILD supports only full-K headlazy baseline/persistent cap1/cap2/cap2-late-rescale/cap2-vstage3 and corrected deep-joint");
        }
#elif defined(TK_FA4_FULLK_WAVE_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "deep_joint_fullk_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "dualquery_fullk_parent_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_dual_query_parent_fullk_ex2e16pc>{},
                false);
        } else {
            throw std::invalid_argument(
                "TK_FA4_FULLK_WAVE_PROBE_BUILD supports only repaired and parent full-K routes");
        }
#elif defined(TK_FA4_Q4_LARGE_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0) {
            const bool use_q4_head_lazy = batch == 1 &&
                ((seqlen == 2048 && (heads == 16 || heads == 32 || heads == 64)) ||
                 (seqlen == 4096 && (heads == 32 || heads == 64)) ||
                 (seqlen == 8192 && (heads == 16 || heads == 32 || heads == 64)));
            if (use_q4_head_lazy) {
                launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
            } else {
                launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
            }
        } else if (std::strcmp(fwd_config, "q4_large_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullgrid_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "q4_headlazy_persistent_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_probe") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
#if TK_FA4_DEEP_JOINT_STAGE >= 2
        } else if (std::strcmp(fwd_config, "deep_joint_fullk_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "deep_joint_fullk_vgrant_shared_ready_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_vgrant_shared_ready_ex2e16pc>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "deep_joint_fullk_vgrant_cp_issued_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_vgrant_cp_issued_ex2e16pc>{},
                false);
#endif
        } else if (std::strcmp(
                       fwd_config,
                       "dualquery_fullk_parent_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_dual_query_parent_fullk_ex2e16pc>{},
                false);
        } else {
            throw std::invalid_argument("TK_FA4_Q4_LARGE_PROBE_BUILD supports auto and the q4 large-shape probe routes");
        }
#elif defined(TK_FA4_DUAL_QUERY_PROBE_BUILD)
#ifdef TK_FA4_DUAL_QUERY_STATIC_PROBE_BUILD
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "dualquery_seq_static_halfgrid_ex2e16pc_q200_p112_o56") == 0) {
            launch_mxfp4_dual_query_halfgrid(std::type_identity<config_fp4pv_stage2_ex2_alu_dual_query_static_halfgrid<16, 128, 128, 192, 128, 200, 56, 112, 1>>{});
        } else {
            throw std::invalid_argument("TK_FA4_DUAL_QUERY_STATIC_PROBE_BUILD only supports the static-stride sequential half-grid candidate");
        }
#else
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "dualquery_seq_halfgrid_ex2e16pc_q200_p112_o56") == 0) {
            launch_mxfp4_dual_query_halfgrid(std::type_identity<config_fp4pv_stage2_ex2_alu_dual_query_seq_halfgrid<16, 128, 128, 192, 128, 200, 56, 112, 1>>{});
        } else if (std::strcmp(fwd_config, "dualquery_seq_3wg_halfgrid_ex2e16pc_q200_p112_o56") == 0) {
            launch_mxfp4_dual_query_halfgrid(std::type_identity<config_fp4pv_stage2_ex2_alu_dual_query_seq_3wg_halfgrid<16, 128, 128, 192, 128, 200, 56, 112, 1>>{});
        } else if (std::strcmp(fwd_config, "dualquery_concurrent_halfgrid_ex2e16pc_q168_i64_p64") == 0) {
            launch_mxfp4_dual_query_concurrent(std::type_identity<config_fp4pv_stage2_ex2_alu_dual_query_concurrent_halfgrid<16, 128, 128, 192, 128, 168, 64, 64, 1>>{});
        } else if (std::strcmp(
                       fwd_config,
                       "dualquery_two_score_overlay_ex2e16pc") == 0) {
            launch_mxfp4_dual_query_two_score_overlay(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_two_score_overlay>{});
        } else if (std::strcmp(fwd_config, "dualquery_two_n64_ex2e16pc") == 0) {
            launch_mxfp4_two_n64(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_two_n64>{});
        } else if (std::strcmp(fwd_config, "dualquery_one_n128_ex2e16pc") == 0) {
            launch_mxfp4_one_n128(std::type_identity<config_fp4pv_stage2_ex2_alu_dual_query_one_n128>{});
        } else if (std::strcmp(fwd_config, "dualquery_one_n128_bf16ret_ex2e16pc") == 0) {
            launch_mxfp4_one_n128_bf16ret(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_one_n128_bf16ret>{});
        } else if (std::strcmp(fwd_config, "dualquery_one_n128_fp16ret_ex2e16pc") == 0) {
            launch_mxfp4_one_n128_fp16ret(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_one_n128_fp16ret>{});
        } else if (std::strcmp(fwd_config, "dualquery_one_n128_fp16scaledret_ex2e16pc") == 0) {
            launch_mxfp4_one_n128_fp16scaledret(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_one_n128_fp16scaledret>{});
        } else if (std::strcmp(fwd_config, "dualquery_one_n128_evac_ex2e16pc") == 0) {
            launch_mxfp4_one_n128_evac(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_one_n128_evac>{});
        } else if (std::strcmp(fwd_config, "dualquery_one_n128_3wg_ex2e16pc") == 0) {
            launch_mxfp4_one_n128_3wg(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_one_n128_3wg>{});
        } else if (std::strcmp(fwd_config, "dualquery_one_n128_smem_evac_ex2e16pc") == 0) {
            launch_mxfp4_one_n128_smem_evac(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_one_n128_smem_evac>{});
        } else if (std::strcmp(fwd_config, "dualquery_one_n128_no_retention_ex2e16pc") == 0) {
            launch_mxfp4_one_n128_no_retention(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_dual_query_one_n128_no_retention>{});
        } else if (std::strcmp(fwd_config, "dualquery_fullk_parent_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<config_fp4pv_dual_query_parent_fullk_ex2e16pc>{},
                false);
        } else if (std::strcmp(fwd_config, "q4_headlazy_fullk_probe") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale_fullk<
                        16, 128, 128, 192, 128, 200, 56, 112, 1>>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_ranklocal_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_ranklocal_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_ranklocal_fullk>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_vmc_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_vmc_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<config_fp4pv_cluster2_vmc_fullk>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<config_fp4pv_cluster2_kmc_fullk>{},
                false);
#if TK_FA4_KMC_TAIL_DRAIN_PROBE_BUILD
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_unmatched_tail_drain_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_unmatched_tail_drain_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_unmatched_tail_drain>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_all_tail_drain_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_all_tail_drain_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_all_tail_drain>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_symmetric_completion_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_symmetric_completion_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_symmetric_completion>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_symmetric_completion_all_tail_drain_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_symmetric_completion_all_tail_drain_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_symmetric_completion_all_tail_drain>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_local_tail_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_local_tail_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_local_tail_symmetric_completion_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_local_tail_symmetric_completion_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_symmetric_completion>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_local_tail_deferred_kscale_completion_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_local_tail_deferred_kscale_completion_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_deferred_kscale_completion>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_local_tail_combined_completion_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_local_tail_combined_completion_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_combined_completion>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_local_tail_payload_owner_arm_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_local_tail_payload_owner_arm_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_owner_arm>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_local_tail_payload_scale_owner_arm_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_local_tail_payload_scale_owner_arm_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_local_tail_payload_scale_owner_arm>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_strict_local_tail_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_strict_local_tail_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_strict_local_tail>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_payload_proxy_acquire_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_payload_proxy_acquire_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_payload_proxy_acquire>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_common_prefix_payload_scale_proxy_acquire_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_common_prefix_payload_scale_proxy_acquire_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_common_prefix_payload_scale_proxy_acquire>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scope_handoff_i_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scope_handoff_i_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scope_handoff_i>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scope_reuse_j_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scope_reuse_j_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scope_reuse_j>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scope_full_k_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scope_full_k_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scope_full_k>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_only_l_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_only_l_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_only_l>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_only_m_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_only_m_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_only_m>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_issue_gate_n_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_issue_gate_n_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_issue_gate_n>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_issue_gate_o_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_issue_gate_o_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_issue_gate_o>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_postreuse_arm_p_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_postreuse_arm_p_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_postreuse_arm_p>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_postreuse_arm_q_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_postreuse_arm_q_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_postreuse_arm_q>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_postreuse_issue_gate_r_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_postreuse_issue_gate_r_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_postreuse_issue_gate_r>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_postreuse_issue_gate_s_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_postreuse_issue_gate_s_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_postreuse_issue_gate_s>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_split_local_remote_reuse_t_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_split_local_remote_reuse_t_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_split_local_remote_reuse_t>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_split_local_remote_reuse_u_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_split_local_remote_reuse_u_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_split_local_remote_reuse_u>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_split_reuse_issue_gate_v_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_split_reuse_issue_gate_v_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_v>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_split_reuse_issue_gate_w_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_split_reuse_issue_gate_w_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_w>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_split_reuse_issue_gate_proxy_x_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_split_reuse_issue_gate_proxy_x_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_proxy_x>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_split_reuse_issue_gate_proxy_y_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_split_reuse_issue_gate_proxy_y_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_proxy_y>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_split_reuse_issue_gate_producer_proxy_z_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_split_reuse_issue_gate_producer_proxy_z_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_split_reuse_issue_gate_producer_proxy_z>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_split_reuse_issue_gate_producer_proxy_aa_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_split_reuse_issue_gate_producer_proxy_aa_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_split_reuse_issue_gate_producer_proxy_aa>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_producer_rendezvous_ab_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_producer_rendezvous_ab_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_producer_rendezvous_ab>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_scale_producer_rendezvous_ac_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_scale_producer_rendezvous_ac_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_scale_producer_rendezvous_ac>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_producer_rendezvous_ad_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_producer_rendezvous_ad_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_producer_rendezvous_ad>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_producer_rendezvous_ae_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_producer_rendezvous_ae_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_producer_rendezvous_ae>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_preissue_rendezvous_af_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_preissue_rendezvous_af_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_preissue_rendezvous_af>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_payload_direct_tma_ready_ag_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_payload_direct_tma_ready_ag_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_payload_direct_tma_ready_ag>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_output_rescale_producer_fence_ah_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_output_rescale_producer_fence_ah_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_output_rescale_producer_fence_ah>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_output_rescale_paired_fence_ai_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_output_rescale_paired_fence_ai_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_output_rescale_paired_fence_ai>{},
                false);
#endif
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_diag_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_diag_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<config_fp4pv_cluster2_kmc_fullk_diag>{},
                false);
#if TK_FA4_KMC_PASSIVE_PROBE_BUILD
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kmc_passive_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kmc_passive_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kmc_fullk_passive>{},
                false);
#endif
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kvmc_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kvmc_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<config_fp4pv_cluster2_kvmc_fullk>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "q4_fullk_cluster2_kvmc_symmetric_probe") == 0) {
            TORCH_CHECK(
                ((seqlen / 128) & 1) == 0,
                "q4_fullk_cluster2_kvmc_symmetric_probe requires an even number of M128 query tiles");
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_cluster2_kvmc_symmetric_fullk>{},
                false);
#if TK_FA4_DEEP_JOINT_STAGE >= 2
        } else if (std::strcmp(fwd_config, "deep_joint_fullk_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_selected_ex2e16pc>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "deep_joint_fullk_vgrant_shared_ready_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_vgrant_shared_ready_ex2e16pc>{},
                false);
        } else if (std::strcmp(
                       fwd_config,
                       "deep_joint_fullk_vgrant_cp_issued_ex2e16pc") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_parent_fullk_deep_joint_vgrant_cp_issued_ex2e16pc>{},
                false);
#endif
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_DUAL_QUERY_PROBE_BUILD only supports dual-query probes and the matched e16pc baseline");
        }
#endif
#elif defined(TK_FA4_SHAPE_ADAPTIVE_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "shapehalf_k256_pipe2_ex2e16pc_q200_p112_o56") == 0) {
            launch_mxfp4_k256_static(std::type_identity<config_fp4pv_stage2_ex2_alu_true_half_k256_pipe2<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "shapehalf_k256_scaffold_ex2e16pc_q200_p112_o56") == 0) {
            launch_mxfp4_k256_static(std::type_identity<config_fp4pv_stage2_ex2_alu_true_half_k256<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else {
            throw std::invalid_argument("TK_FA4_SHAPE_ADAPTIVE_PROBE_BUILD only supports the K256 scaffold");
        }
#elif defined(TK_FA4_FP32PACK_ONLY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_FP32PACK_ONLY_PROBE_BUILD supports auto or fp32pack");
        }
#elif defined(TK_FA4_Q176_ONLY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_Q176_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48");
        }
#elif defined(TK_FA4_NB64_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_nb64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<128, 64, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_NB64_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_nb64");
        }
#elif defined(TK_FA4_SPLIT2WG_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_split2wg_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_early_pready_split2wg_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_SPLIT2WG_PROBE_BUILD supports auto or fp32pack_early_pready_split2wg_q176_o48");
        }
#elif defined(TK_FA4_OUTREG_O40_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o40_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_OUTREG_O40_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o40");
        }
#elif defined(TK_FA4_OUTREG_O56_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o56_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_OUTREG_O56_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o56");
        }
#elif defined(TK_FA4_Q176_STAGEV_ONLY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_Q176_STAGEV_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_stagev");
        }
#elif defined(TK_FA4_DIRECTRESCALE_PRODREG_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "dualaccum_directrescale_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p96") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 64, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_legacy_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_lightpublish_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 4, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_payloadv2_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 4, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_payloadv2_lightpub_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_lightpublish_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 4, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q192_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 192, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pready_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pready_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pready_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pready_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pfirst_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pfirst_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pfirst_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pfirst_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_qk2_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_qk2_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_qk2_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_qk2_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_deferremote_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_deferremote_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_deferremote_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_deferremote_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_cta2_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_cta2_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p80_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 80, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p112_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p96_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 40, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p96_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage3_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage3_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage4_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage4_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage5_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage5_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage6_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage6_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_stagev_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage3_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage3_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage3_stagev_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage3_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_prescaled_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_rte_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_rte_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_latepack_pstage4_q208_p96_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_latepack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q208_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_dualaccum_decoupled_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_qklogn_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_qklogn_dualaccum_decoupled_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_pairqk_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_pairqk_dualaccum_decoupled_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_prescaled_floor_fusedmax_forcespare_pairqk_nox1_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_fusedmax_qkscfix_forcespare_pairqk_dualaccum_decoupled_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "scorepack_prescaled_floor_x1sc_fusedmax_pairqk_singleout_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_pairqk_singleout_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pchainc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_pchain_c<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "scorederived_ex2e16pc_split2wg_full_pstage2_q152_p112_o48") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split<16, 128, 128, 192, 128, 152, 48, 112, 1>>{}, false);
	        } else if (std::strcmp(fwd_config, "scorederived_ex2e16pc_split2wg_k64_pstage2_q152_p112_o48") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split_k64<16, 128, 128, 192, 128, 152, 48, 112, 1>>{}, false);
	        } else if (std::strcmp(fwd_config, "shapehalf_k256_scaffold_ex2e16pc_q200_p112_o56") == 0) {
	            launch_mxfp4_k256_static(std::type_identity<config_fp4pv_stage2_ex2_alu_true_half_k256<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
	        } else if (std::strcmp(fwd_config, "shapehalf_k256_pipe2_ex2e16pc_q200_p112_o56") == 0) {
	            launch_mxfp4_k256_static(std::type_identity<config_fp4pv_stage2_ex2_alu_true_half_k256_pipe2<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pready_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pready_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_lazyorescale_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_lazyorescale_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_k64halfready_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_k64halfready_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_k64halfready_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_k64halfready_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_diag_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_diag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_schedown_taskdiag_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_schedown_taskdiag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_livemirror_taskdiag_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_livemirror_taskdiag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_staticstride_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_staticstride_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_taskdiagnoreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_taskdiagnoreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_taskdonereinit_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
		            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdonereinit_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
			        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
				            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
			        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_outreusefence_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
				            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
			        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_alltasks_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
				            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_alltasks_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_outreusefence_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_alltasks_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_alltasks_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
			        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_taskdonereinit_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
				            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
			        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_vpipereset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
				            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_vpipereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_longspin_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_longspin_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_vpipereset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_vpipereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_outreusefence_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_outreusefence_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_pvoutreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
		            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_pvoutreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_vpipereset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_vpipereset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_skipoutreusewait_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_startsync_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_startsync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_skipoutreusewait_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_startsync_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_startsync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2fullv_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2fullv_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vmc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcpt_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcpt_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptshortdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptshortdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptringdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptringdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptqkdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptqkdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptskiptail_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptskiptail_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vmcdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymcdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymcshortdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcshortdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymcscalediag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcscalediag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymc2slotdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymc2slotdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymcclpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcclpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_idxwait_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_idxwait_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_psrcdone_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_psrcdone_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_pdelay1_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pdelay1_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_alldelay1_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_alldelay1_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_tasknorm_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasknorm_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_tasktrans_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasktrans_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_tasktail_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasktail_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_pvscown_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvscown_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_pvruse_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvruse_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_pvarruse_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvarruse_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_forcespare_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_forcespare_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_corr3_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_corr3_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_pubsync_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pubsync_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_pvscown_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pvscown_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_descwait_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_descwait_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_descdiag_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_descdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pairpsc_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pairpsc_throttle2_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_pairpsc_throttle2_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q192_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 192, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p80_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 80, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p112_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p96_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 40, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p96_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pready_pstage4_stagev_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pready_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage3_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_prescaled_pstage4_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage4_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage5_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage5_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage6_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage6_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_prescaled_pstage6_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage6_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_4wg_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_4wg_pstage3_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_split2wg_q152_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q144_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 144, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q160_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 160, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 96, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p104_o48") == 0) {
	            if (seqlen >= 8192) {
	                launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_leader_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
	            } else {
	                launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 144, 48, 120, 1>>{}, true);
	            }
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vmc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcpt_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcpt_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_mb128_nb64_q152_p104_o48") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 64, 192, 128, 152, 48, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_pstage3_mb128_nb64_q152_p104_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 64, 192, 128, 152, 48, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_rowsplit2wg_mb128_nb64_q152_p104_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_rowsplit2wg_dualaccum_directrescale_pregs_force_persistent<128, 64, 192, 128, 152, 48, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_rowsplit2wg_pstage3_mb128_nb64_q152_p104_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_rowsplit2wg_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 64, 192, 128, 152, 48, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p112_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p128_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p120_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 40, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p120_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 56, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_leader_split2wg_q152_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_leader_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_pstage3_q152_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_split2wg_q160_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent<128, 128, 192, 128, 160, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_split2wg_q168_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent<128, 128, 192, 128, 168, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_rowsplit2wg_q160_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_rowsplit2wg_pregs_force_persistent<128, 128, 192, 128, 160, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_mb64_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<64, 64, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage3_mb64_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<64, 64, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "directscale_mb64_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_pregs_force_persistent<64, 64, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "directscale_pstage3_mb64_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_pstage3_pregs_force_persistent<64, 64, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "directscale_mb128_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_pregs_force_persistent<128, 64, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "directscale_pstage3_mb128_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_pstage3_pregs_force_persistent<128, 64, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_mb64_nb64_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<64, 64, 192, 128, 176, 48, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_mb64_nb64_q192_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<64, 64, 192, 128, 192, 48, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_mb64_nb64_q208_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<64, 64, 192, 128, 208, 48, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_p3_mb64_nb64_q208_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_pstage3_regs_force_persistent<64, 64, 192, 128, 208, 48, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p112") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 64, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p120") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 64, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p136") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 64, 136, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_DIRECTRESCALE_PRODREG_PROBE_BUILD supports direct-rescale probe configs");
        }
#elif defined(TK_FA4_DUALACCUM_REG_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "dualaccum") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_q184") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<128, 128, 192, 128, 184, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_q192") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<128, 128, 192, 128, 192, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_q200") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<128, 128, 192, 128, 200, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_q184_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<128, 128, 192, 128, 184, 48, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_q192_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<128, 128, 192, 128, 192, 48, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_DUALACCUM_REG_PROBE_BUILD supports auto, dualaccum, dualaccum_q184, dualaccum_q192, dualaccum_q200, dualaccum_q208, dualaccum_q184_o48, or dualaccum_q192_o48");
        }
#elif defined(TK_FA4_VLOAD_PROBE_BUILD)
#ifdef TK_FA4_VLOAD_V3_ONLY_PROBE_BUILD
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_v3") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_v3_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_VLOAD_V3_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_v3");
        }
#elif defined(TK_FA4_VLOAD_V1_ONLY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_v1") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_v1_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_VLOAD_V1_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_v1");
        }
#else
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_v1") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_v1_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_v3") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_v3_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_VLOAD_PROBE_BUILD supports auto, fp32pack_early_pready_q176_o48, fp32pack_early_pready_q176_o48_v1, or fp32pack_early_pready_q176_o48_v3");
        }
#endif
#elif defined(TK_FA4_PRODREG_PROBE_BUILD)
#ifdef TK_FA4_PRODREG_P136_ONLY_PROBE_BUILD
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_p136") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p136_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_PRODREG_P136_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_p136");
        }
#elif defined(TK_FA4_PRODREG_P120_ONLY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_p120") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p120_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_PRODREG_P120_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_p120");
        }
#elif defined(TK_FA4_PRODREG_P112_ONLY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_p112") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p112_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_PRODREG_P112_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_p112");
        }
#elif defined(TK_FA4_PRODREG_P104_ONLY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_p104") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p104_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_PRODREG_P104_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_p104");
        }
#elif defined(TK_FA4_PRODREG_P96_ONLY_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_p96") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p96_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_PRODREG_P96_ONLY_PROBE_BUILD supports auto or fp32pack_early_pready_q176_o48_p96");
        }
#else
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_p112") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p112_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_p120") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p120_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_p136") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_p136_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_PRODREG_PROBE_BUILD supports auto, fp32pack_early_pready_q176_o48, fp32pack_early_pready_q176_o48_p112, fp32pack_early_pready_q176_o48_p120, or fp32pack_early_pready_q176_o48_p136");
        }
#endif
#elif defined(TK_FA4_ASYNCV_PROBE_BUILD)
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_stagev_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_dualaccum") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_dualaccum_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_stagev_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_ASYNCV_PROBE_BUILD supports auto, fp32pack_early_pready_q168_o48_stagev, fp32pack_early_pready_q168_o48_asyncv, fp32pack_early_pready_q168_o48_stagev_asyncv, fp32pack_early_pready_q176_o48, fp32pack_early_pready_q176_o48_asyncv, or fp32pack_early_pready_q176_o48_stagev_asyncv");
        }
#else
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
            std::strcmp(fwd_config, "auto") == 0 ||
            std::strcmp(fwd_config, "fp32pack_early_pready") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q160_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q160_o48_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_o48_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_stagev_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o40_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o40_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_dualaccum") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_dualaccum_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_stagev_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else {
            throw std::invalid_argument("TK_FA4_STAGEV_PROBE_BUILD supports auto, fp32pack_early_pready, fp32pack_early_pready_stagev, fp32pack_early_pready_q160_stagev, fp32pack_early_pready_q160_o48_stagev, fp32pack_early_pready_q176_stagev, fp32pack_early_pready_q168_o48_stagev, fp32pack_early_pready_q168_o48_asyncv, fp32pack_early_pready_q168_o48_stagev_asyncv, fp32pack_early_pready_q176_o40_stagev, fp32pack_early_pready_q176_o48, fp32pack_early_pready_q176_o48_asyncv, fp32pack_early_pready_q176_o48_stagev, or fp32pack_early_pready_q176_o48_stagev_asyncv");
        }
#endif
#else
        if (fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 || std::strcmp(fwd_config, "auto") == 0) {
#if defined(TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD) && \
    TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD
            const bool use_track_b_stage2_cap2 =
                persistent_launch && batch == 1 &&
                ((seqlen == 1024 && heads == 64) ||
                 (seqlen == 2048 && (heads == 32 || heads == 64)));
            if (use_track_b_stage2_cap2) {
                launch_mxfp4(
                    std::type_identity<
                        config_fp4pv_track_b_stage2_fullk_cap2_saturated>{},
                    true);
            } else {
#endif
            const bool use_q4_head_lazy = batch == 1 &&
                ((seqlen == 2048 && (heads == 16 || heads == 32 || heads == 64)) ||
                 (seqlen == 4096 && (heads == 32 || heads == 64)) ||
                 (seqlen == 8192 && (heads == 16 || heads == 32 || heads == 64)));
            if (use_q4_head_lazy) {
                launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
            } else {
                launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
            }
#if defined(TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD) && \
    TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD
            }
        } else if (std::strcmp(
                       fwd_config,
                       "real_fwd_track_b_stage2_fullk_cap2_saturated") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_track_b_stage2_fullk_cap2_saturated>{},
                true);
        } else if (std::strcmp(
                       fwd_config,
                       "real_fwd_track_b_stage2_fullk_fullgrid_control") == 0) {
            launch_mxfp4(
                std::type_identity<
                    config_fp4pv_track_b_stage2_fullk_fullgrid_control>{},
                false);
#endif
        } else if (std::strcmp(fwd_config, "fusedstream_v0") == 0) {
            launch_fused_stream(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{});
        } else if (std::strcmp(fwd_config, "fusedstream_dynpscale_v0") == 0) {
            launch_fused_stream(
                std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{},
                std::true_type{});
        } else if (std::strcmp(fwd_config, "fusedstream_dynpscale_direct_tmem_v0") == 0) {
            launch_fused_stream(
                std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{},
                std::true_type{},
                std::true_type{});
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q160") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q160_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q160_o48_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q160_o48_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q168_o48_stagev_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q168_o48_stagev_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_pstage3_q168_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_pstage3_q168_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_earlycorr_q168_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_earlycorr_q168_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_q168_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_q168_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_q168_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q168_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage3_q168_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q168_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o40_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o40_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o40_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_dualaccum") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_dualaccum_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_stagev") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o48_stagev_asyncv") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o48_stagev_asyncv_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q176_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q176_o56_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_q184_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_q184_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_pstage3_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_pstage3_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_streamscore_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_streamscore_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_streamscore_half_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_streamscore_half_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_earlycorr_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_earlycorr_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o40_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o40_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_p104") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_pregs_force_persistent<128, 128, 192, 128, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_p120") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_pregs_force_persistent<128, 128, 192, 128, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48_p104") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48_pregs_force_persistent<128, 128, 192, 128, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48_p120") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48_pregs_force_persistent<128, 128, 192, 128, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_v1") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_vload_force_persistent<128, 128, 192, 128, 1, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_v3") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48_vload_force_persistent<128, 128, 192, 128, 3, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage3_q184_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage3_q184_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_pstage4_q184_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_pstage4_q184_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_splitk64_earlycorr_q176_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_splitk64_earlycorr_q176_o40_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_early_pready_split2wg_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_early_pready_split2wg_q176_o48_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_earlycorr") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_earlycorr_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "fp32pack_warprescale") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_warprescale_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_legacy") == 0 ||
                   std::strcmp(fwd_config, "08380fc_dualaccum") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_legacy_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_force_persistent<128, 128, 192, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_q192") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<128, 128, 192, 128, 192, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q192") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent<128, 128, 192, 128, 192, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p96") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 64, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_legacy_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_lightpublish_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 4, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_payloadv2_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 4, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_payloadv2_lightpub_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_payloadstore_lightpublish_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 4, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q192_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 192, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pready_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pready_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pready_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pready_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pfirst_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pfirst_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pfirst_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pfirst_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_qk2_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_qk2_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_qk2_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_qk2_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_deferremote_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_deferremote_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_deferremote_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_deferremote_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_cta2_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_cta2_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p80_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 80, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p112_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p96_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 40, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_p96_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage3_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage3_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage4_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage4_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage5_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage5_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage6_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage6_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_stagev_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage3_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage3_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage3_stagev_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage3_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_prescaled_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_rte_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_rte_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p96_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_pstage4_q208_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q208_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q208_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pstage4_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage4_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_dualaccum_decoupled_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_qklogn_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_qklogn_dualaccum_decoupled_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_prescaled_floor_x1sc_fusedmax_forcespare_pairqk_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_forcespare_pairqk_dualaccum_decoupled_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_scorepack_prescaled_floor_fusedmax_forcespare_pairqk_nox1_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_fusedmax_qkscfix_forcespare_pairqk_dualaccum_decoupled_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "scorepack_prescaled_floor_x1sc_fusedmax_pairqk_singleout_nb64_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_pairqk_singleout_pstage2_pregs_force_persistent<128, 64, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pchainc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_stage2_pchain_c<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_stage2_ex2_alu_pchain_c<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "scorederived_ex2e16pc_split2wg_full_pstage2_q152_p112_o48") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split<16, 128, 128, 192, 128, 152, 48, 112, 1>>{}, false);
	        } else if (std::strcmp(fwd_config, "scorederived_ex2e16pc_split2wg_k64_pstage2_q152_p112_o48") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split_k64<16, 128, 128, 192, 128, 152, 48, 112, 1>>{}, false);
	        } else if (std::strcmp(fwd_config, "shapehalf_k256_scaffold_ex2e16pc_q200_p112_o56") == 0) {
	            launch_mxfp4_k256_static(std::type_identity<config_fp4pv_stage2_ex2_alu_true_half_k256<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
	        } else if (std::strcmp(fwd_config, "shapehalf_k256_pipe2_ex2e16pc_q200_p112_o56") == 0) {
	            launch_mxfp4_k256_static(std::type_identity<config_fp4pv_stage2_ex2_alu_true_half_k256_pipe2<16, 128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pready_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pready_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_lazyorescale_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_lazyorescale_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_k64halfready_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_k64halfready_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_k64halfready_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_k64halfready_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_diag_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_diag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_schedown_taskdiag_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_schedown_taskdiag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_livemirror_taskdiag_onevpub_fullvsc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_persistouter_clc_schedwg4_slotlife_livemirror_taskdiag_onevpub_fullvsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwarp_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_staticstride_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_staticstride_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, false);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_taskdiagnoreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_skipreused_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_taskdiagnoreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_taskdonereinit_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
		            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdonereinit_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_alltasks_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_alltasks_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_outreusefence_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_pvoutreset_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_alltasks_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_alltasks_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
			        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_taskdonereinit_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
				            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_scorereset_skiptailpcopy_taskdonereinit_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
			        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_vpipereset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
				            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_vpipereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_longspin_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_longspin_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_vpipereset_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_skipoutreusewait_pvoutreset_vpipereset_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_outreusefence_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_taskdone6_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_outreusefence_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
			            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_outreuseafterstore_pvwaitout_outreusefence_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
		        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_pvoutreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
		            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_pvoutreset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_vpipereset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_vpipereset_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_skipoutreusewait_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_corrreset_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
	        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_startsync_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
	            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiagnoreset_startsync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_skipoutreusewait_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_proddonesync_taskdiag_skipoutreusewait_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedonelead_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_startsync_onevpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vsc16_persistouter_clc_schedwg4_persistreuse_carryphase_issuedoneactive_outreuseafterstore_pvwaitout_startsync_onevpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2fullv_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2fullv_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vmc_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmc_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcpt_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcpt_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptshortdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptshortdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptringdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptringdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptqkdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptqkdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2kmcptskiptail_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2kmcptskiptail_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vmcdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vmcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymcdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymcshortdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcshortdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymcscalediag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcscalediag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymc2slotdiag_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymc2slotdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_rowpar2vpaymcclpub_vtma_vstma_pstage2_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_rowpar2vpaymcclpub_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_idxwait_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_idxwait_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_psrcdone_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_psrcdone_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_pdelay1_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pdelay1_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_alldelay1_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_alldelay1_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_tasknorm_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasknorm_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_tasktrans_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasktrans_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_tasktail_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_tasktail_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_pvscown_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvscown_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_pvruse_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvruse_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_statssplit_pvarruse_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_statssplit_pvarruse_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_forcespare_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_forcespare_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_corr3_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_corr3_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_pubsync_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pubsync_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_pvscown_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_pvscown_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_descwait_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_descwait_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_pairpsc_descdiag_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_pairpsc_descdiag_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pairpsc_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_pairpsc_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_pairpsc_throttle2_vtma_vstma_pstage3_q200_p112_o56_qkscfix") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_pairpsc_throttle2_vtma_vstma_dualaccum_directrescale_decoupled_pstage3_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q192_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 192, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q224_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 224, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p80_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 80, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p112_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p96_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 40, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_stagev_q208_p96_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 56, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pready_pstage4_stagev_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pready_stagev_pstage4_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage3_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage3_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage4_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage4_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_prescaled_pstage4_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage4_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage5_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage5_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_pstage6_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_pstage6_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_decoupled_prescaled_pstage6_q208_p96_o48_splitk64") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_decoupled_prescaled_pstage6_splitk64_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_4wg_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_4wg_pstage3_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 128, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_split2wg_q152_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q144_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 144, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q160_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 160, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p104_o48") == 0) {
            if (seqlen >= 8192) {
                launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_leader_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
            } else {
                launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 144, 48, 120, 1>>{}, true);
            }
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_mb128_nb64_q152_p104_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 64, 192, 128, 152, 48, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_pstage3_mb128_nb64_q152_p104_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 64, 192, 128, 152, 48, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_rowsplit2wg_mb128_nb64_q152_p104_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_rowsplit2wg_dualaccum_directrescale_pregs_force_persistent<128, 64, 192, 128, 152, 48, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_rowsplit2wg_pstage3_mb128_nb64_q152_p104_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_rowsplit2wg_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 64, 192, 128, 152, 48, 104, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p112_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 112, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p128_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 128, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p120_o40") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 40, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_q152_p120_o56") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 56, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_leader_split2wg_q152_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_leader_split2wg_dualaccum_directrescale_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_localmax_split2wg_pstage3_q152_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_directpscale_localmax_split2wg_dualaccum_directrescale_pstage3_pregs_force_persistent<128, 128, 192, 128, 152, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_split2wg_q160_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent<128, 128, 192, 128, 160, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_split2wg_q168_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_split2wg_pregs_force_persistent<128, 128, 192, 128, 168, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_rowsplit2wg_q160_p120_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_4wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_rowsplit2wg_pregs_force_persistent<128, 128, 192, 128, 160, 48, 120, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_mb64_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pregs_force_persistent<64, 64, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_pstage3_mb64_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_pstage3_pregs_force_persistent<64, 64, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "directscale_mb64_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_pregs_force_persistent<64, 64, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "directscale_pstage3_mb64_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_pstage3_pregs_force_persistent<64, 64, 192, 128, 208, 48, 96, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "directscale_mb128_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_pregs_force_persistent<128, 64, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "directscale_pstage3_mb128_nb64_q208_p96_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_pstage3_pregs_force_persistent<128, 64, 192, 128, 208, 48, 96, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_mb64_nb64_q176_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<64, 64, 192, 128, 176, 48, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_mb64_nb64_q192_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<64, 64, 192, 128, 192, 48, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_mb64_nb64_q208_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<64, 64, 192, 128, 208, 48, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_p3_mb64_nb64_q208_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_pstage3_regs_force_persistent<64, 64, 192, 128, 208, 48, 2>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_directrescale_q208_o72") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_directrescale_regs_force_persistent<128, 128, 192, 128, 208, 72, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_q192") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<128, 128, 192, 128, 192, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_p3_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_pstage3_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_p3_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_pstage3_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_q192") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 192, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_q240") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 240, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_q208_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 208, 48, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_q208_o72") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 208, 72, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_q224_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 224, 48, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_q224_o72") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 224, 72, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_p3_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_pstage3_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_p3_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_pstage3_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_q192") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 192, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_q240") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 240, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_q208_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 208, 48, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_q208_o72") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 208, 72, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_q224_o48") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 224, 48, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_q224_o72") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 224, 72, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_p3_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_pstage3_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_forcespare_streamscore_half_p3_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_forcespare_streamscore_half_pstage3_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "early_dualaccum_forcespare_streamscore_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "early_dualaccum_forcespare_streamscore_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "early_dualaccum_forcespare_streamscore_half_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "early_dualaccum_forcespare_streamscore_half_q208_o72") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 208, 72, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "early_dualaccum_forcespare_streamscore_half_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_early_pready_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "noearly_dualaccum_forcespare_streamscore_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_noearly_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "noearly_dualaccum_forcespare_streamscore_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_noearly_dualaccum_forcespare_streamscore_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "noearly_dualaccum_forcespare_streamscore_half_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_noearly_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "noearly_dualaccum_forcespare_streamscore_half_q208_o72") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_noearly_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 208, 72, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "noearly_dualaccum_forcespare_streamscore_half_q224") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_noearly_dualaccum_forcespare_streamscore_half_regs_force_persistent<128, 128, 192, 128, 224, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_streamscore_q192") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_streamscore_regs_force_persistent<128, 128, 192, 128, 192, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_streamscore_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_streamscore_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_streamscore_half_q192") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_streamscore_half_regs_force_persistent<128, 128, 192, 128, 192, 64, 1>>{}, true);
        } else if (std::strcmp(fwd_config, "dualaccum_streamscore_half_q208") == 0) {
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_fixed_pscale_fp32pack_dualaccum_streamscore_half_regs_force_persistent<128, 128, 192, 128, 208, 64, 1>>{}, true);
        } else {
	            throw std::invalid_argument("TK_FA4_FP4PV_FWD_CONFIG must be auto, fp32pack, fp32pack_early_pready, fp32pack_early_pready_stagev, fp32pack_early_pready_q160, fp32pack_early_pready_q160_stagev, fp32pack_early_pready_q160_o48_stagev, fp32pack_early_pready_q176, fp32pack_early_pready_q176_stagev, fp32pack_early_pready_q168_o48, fp32pack_early_pready_q168_o48_stagev, fp32pack_early_pready_q168_o48_asyncv, fp32pack_early_pready_q168_o48_stagev_asyncv, fp32pack_early_pready_q176_o40, fp32pack_early_pready_q176_o40_stagev, fp32pack_early_pready_q176_o48, fp32pack_early_pready_q176_o48_asyncv, fp32pack_early_pready_q176_o48_stagev, fp32pack_early_pready_q176_o48_stagev_asyncv, fp32pack_early_pready_q176_o56, fp32pack_early_pready_q184_o48, fp32pack_early_pready_pstage3_q176_o48, fp32pack_early_pready_streamscore_q176_o48, fp32pack_early_pready_streamscore_half_q176_o48, fp32pack_early_pready_earlycorr_q176_o48, fp32pack_early_pready_splitk64_q176_o48, fp32pack_early_pready_splitk64_earlycorr_q176_o48, fp32pack_early_pready_splitk64_earlycorr_pstage3_q176_o48, fp32pack_early_pready_splitk64_earlycorr_pstage4_q176_o48, fp32pack_early_pready_splitk64_earlycorr_q176_o40, fp32pack_early_pready_split2wg_q176_o48, fp32pack_earlycorr, fp32pack_warprescale, dualaccum_legacy, 08380fc_dualaccum, dualaccum, dualaccum_q192, dualaccum_q208, dualaccum_directrescale_q192, dualaccum_directrescale_q208, dualaccum_directrescale_q208_p96, dualaccum_directrescale_q208_p96_o48, dualaccum_directrescale_pstage3_q208_p96_o48, dualaccum_directrescale_pstage3_q208_p96_o48_splitk64, dualaccum_directrescale_decoupled_q208, dualaccum_directrescale_decoupled_pstage3_q208_p96_o48, dualaccum_directrescale_decoupled_pstage3_q208_p96_o48_splitk64, dualaccum_directrescale_q208_o72, dualaccum_forcespare_q192, dualaccum_forcespare_q208, dualaccum_forcespare_streamscore_q192, dualaccum_forcespare_streamscore_q208, dualaccum_forcespare_streamscore_half_q192, dualaccum_forcespare_streamscore_half_q208, dualaccum_streamscore_q192, dualaccum_streamscore_q208, dualaccum_streamscore_half_q192, or dualaccum_streamscore_half_q208");
        }
#endif
    }
#endif
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
    bool persistent_launch
) {
    dispatch_forward_streaming_live_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, p_quant_mode, persistent_launch, true);
}

void dispatch_forward_streaming_live_mxfp4_output_only(
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
    bool persistent_launch
) {
    dispatch_forward_streaming_live_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, p_quant_mode, persistent_launch, false);
}
#endif

// Minimal host boundary for the exact B1/S4096/H64 N64-combine route.

#ifndef TK_FA4_LARGE_I73_N64_COMBINE_ROUTE
#define TK_FA4_LARGE_I73_N64_COMBINE_ROUTE \
    "real_fwd_large_i73_s4096h64_n64_combine"
#endif
#ifndef TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE
#define TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE 0
#endif
#ifndef TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_SHA256
#define TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_SHA256 ""
#endif

struct large_i73_n64_combine_topology_state {
    long long valid = 0;
    long long batch = 0;
    long long heads = 0;
    long long seqlen = 0;
    long long logical_query_ctas = 0;
    long long physical_grid_ctas = 0;
    long long queries_per_cta = 0;
    long long cluster_size = 0;
    long long threads_per_cta = 0;
    long long physical_tmem_cols = 0;
    long long requested_min_blocks_per_sm = 0;
    long long resident_ctas_per_sm = 0;
    long long resident_at_zero = 0;
    long long resident_at_requested = 0;
    long long available_dynamic_for_two = 0;
    long long available_dynamic_for_two_status = 0;
    long long dynamic_shared_bytes = 0;
    long long max_dynamic_shared_bytes = 0;
    long long preferred_shared_carveout = 0;
};

static std::mutex large_i73_n64_combine_topology_mutex;
static large_i73_n64_combine_topology_state
    large_i73_n64_combine_last_topology;

pybind11::dict read_large_i73_n64_combine_topology() {
    std::lock_guard<std::mutex> lock(large_i73_n64_combine_topology_mutex);
    const auto &state = large_i73_n64_combine_last_topology;
    pybind11::dict out;
    out["schema"] = "mxfp4_large_i73_n64_combine_topology_v1";
    out["valid"] = state.valid;
    out["batch"] = state.batch;
    out["heads"] = state.heads;
    out["seqlen"] = state.seqlen;
    out["logical_query_ctas"] = state.logical_query_ctas;
    out["physical_grid_ctas"] = state.physical_grid_ctas;
    out["queries_per_cta"] = state.queries_per_cta;
    out["cluster_size"] = state.cluster_size;
    out["threads_per_cta"] = state.threads_per_cta;
    out["physical_tmem_cols"] = state.physical_tmem_cols;
    out["requested_min_blocks_per_sm"] = state.requested_min_blocks_per_sm;
    out["resident_ctas_per_sm"] = state.resident_ctas_per_sm;
    out["resident_at_zero"] = state.resident_at_zero;
    out["resident_at_requested"] = state.resident_at_requested;
    out["available_dynamic_for_two"] = state.available_dynamic_for_two;
    out["available_dynamic_for_two_status"] =
        state.available_dynamic_for_two_status;
    out["dynamic_shared_bytes"] = state.dynamic_shared_bytes;
    out["max_dynamic_shared_bytes"] = state.max_dynamic_shared_bytes;
    out["preferred_shared_carveout"] = state.preferred_shared_carveout;
    out["api_residency_authoritative"] =
        !static_cast<bool>(TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE);
    out["runtime_overlap_proof_gate"] =
        static_cast<bool>(TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE);
    out["runtime_overlap_proof_sha256"] =
        TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_SHA256;
    out["static_bounds_gate"] =
        static_cast<bool>(TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE);
    return out;
}

void dispatch_large_i73_n64_combine(
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
                    K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "N64-combine requires packed MXFP4 Q/K/V");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Float8_e4m3fn &&
                    K_sc.dtype() == at::ScalarType::Float8_e4m3fn,
                "N64-combine requires E4M3 Q/K block scales");
    TORCH_CHECK(Q_sg.dtype() == at::ScalarType::Float &&
                    K_sg.dtype() == at::ScalarType::Float,
                "N64-combine requires FP32 Q/K global scales");
    TORCH_CHECK(V_sc.dtype() == at::ScalarType::Byte,
                "N64-combine requires uint8 E8M0 V scales");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 &&
                    LSE.dtype() == at::ScalarType::Float,
                "N64-combine requires BF16 O and FP32 LSE");
    TORCH_CHECK(!persistent_launch,
                "N64-combine is an exact nonpersistent full-grid route");
    TORCH_CHECK(p_quant_mode == 0,
                "N64-combine is fail-closed to the validated RTE P encoder");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(batch == 1 && heads == 64 && seqlen == 4096,
                "N64-combine is fail-closed to B1/S4096/H64");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V_fp4.dim() == 4 &&
                    O.dim() == 4 && LSE.dim() == 4,
                "N64-combine tensor rank mismatch");
    TORCH_CHECK(Q.size(3) == 96 && K.sizes() == Q.sizes(),
                "N64-combine Q/K must be [B,H,S,96 packed]");
    TORCH_CHECK(V_fp4.size(0) == batch && V_fp4.size(1) == heads &&
                    V_fp4.size(2) == 128 &&
                    V_fp4.size(3) == seqlen / 2,
                "N64-combine V must be [B,H,128,S/2 packed]");
    TORCH_CHECK(O.size(0) == batch && O.size(1) == seqlen &&
                    O.size(2) == heads && O.size(3) == 128,
                "N64-combine O must be [B,S,H,128]");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads &&
                    LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "N64-combine LSE must be [B,H,1,S]");
    constexpr int QK_CHUNKS = 192 / 64;
    TORCH_CHECK(Q_sc.size(0) == batch &&
                    Q_sc.size(1) == seqlen / 128 &&
                    Q_sc.size(2) == heads * QK_CHUNKS &&
                    Q_sc.size(3) == 512,
                "N64-combine Q_sc shape mismatch");
    TORCH_CHECK(K_sc.size(0) == batch &&
                    K_sc.size(1) == seqlen / 64 &&
                    K_sc.size(2) == heads * QK_CHUNKS &&
                    K_sc.size(3) == 512,
                "N64-combine K_sc shape mismatch");
    TORCH_CHECK(Q_sg.size(0) == batch && Q_sg.size(1) == heads &&
                    K_sg.sizes() == Q_sg.sizes(),
                "N64-combine Q_sg/K_sg shape mismatch");
    check_fp4pv_mxfp4_v_scale_half_contract(
        V_sc, batch, heads, seqlen);

    const char *route = std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    TORCH_CHECK(route != nullptr &&
                    std::strcmp(
                        route,
                        TK_FA4_LARGE_I73_N64_COMBINE_ROUTE) == 0,
                "candidate-only extension requires exact N64-combine route");

    using C = config_fp4pv_large_i73_n64_combine;
    using G = globals_mxfp4_large_i73_n64_combine<C>;
    static_assert(C::NUM_THREADS == 256 && C::CLUSTER_SIZE == 1);
    static_assert(C::Mb == 128 && C::Nb == 64);
    static_assert(C::Dqk == 192 && C::Dvo == 128);
    static_assert(G::DYNAMIC_SHARED_MEMORY == 61440);
#if TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE
    constexpr int PROVEN_RESIDENT_CTAS_PER_SM = 2;
    constexpr int GB200_REGISTERS_PER_SM = 65536;
    constexpr int CONFIGURED_SHARED_BYTES_PER_SM = 135168;
    static_assert(TK_FA4_LARGE_I73_N64_COMBINE_CLUSTER_METADATA_FREE == 1);
    static_assert(TK_FA4_LARGE_I73_N64_COMBINE_REG120 == 1);
    static_assert(tensor_allocator<2, 1>::cols == 256);
    static_assert(C::NUM_THREADS * 120 <= 32768);
    static_assert(G::DYNAMIC_SHARED_MEMORY + 1152 <= 67584);
    static_assert(
        PROVEN_RESIDENT_CTAS_PER_SM * C::NUM_THREADS * 120
        <= GB200_REGISTERS_PER_SM);
    static_assert(
        PROVEN_RESIDENT_CTAS_PER_SM
            * (G::DYNAMIC_SHARED_MEMORY + 1152)
        <= CONFIGURED_SHARED_BYTES_PER_SM);
#endif
    constexpr int LOGICAL_QUERY_CTAS = 64 * (4096 / 128);
    constexpr int GRID_CTAS = 2048;
    static_assert(GRID_CTAS == 2048);
    static_assert(LOGICAL_QUERY_CTAS == GRID_CTAS);

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        kittens::py::tensor_to_gl<typename G::v_gl>(V_fp4),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const float *>(Q_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const float *>(K_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<float *>(LSE.data_ptr()),
        batch,
        heads,
        seqlen};

    // The opt-in dynamic-SMEM attribute is part of the occupancy query's
    // function state.  Publish it before asking CUDA for the resident-block
    // count; querying first undercounts this 64-KiB route.
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_large_i73_n64_combine<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        G::DYNAMIC_SHARED_MEMORY));
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_large_i73_n64_combine<C>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    cudaFuncAttributes attributes{};
    CUDACHECK(cudaFuncGetAttributes(
        &attributes, kernel_mxfp4_large_i73_n64_combine<C>));
    TORCH_CHECK(
        attributes.maxDynamicSharedSizeBytes == G::DYNAMIC_SHARED_MEMORY,
        "N64-combine dynamic-SMEM opt-in was not retained");
    TORCH_CHECK(
        attributes.preferredShmemCarveout == cudaSharedmemCarveoutMaxShared,
        "N64-combine maximum shared-memory carveout was not retained");
    int resident_at_zero = 0;
    int resident_at_requested = 0;
    size_t available_dynamic_for_two = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_at_zero,
        kernel_mxfp4_large_i73_n64_combine<C>,
        C::NUM_THREADS,
        0));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_at_requested,
        kernel_mxfp4_large_i73_n64_combine<C>,
        C::NUM_THREADS,
        G::DYNAMIC_SHARED_MEMORY));
    const cudaError_t availability_status =
        cudaOccupancyAvailableDynamicSMemPerBlock(
        &available_dynamic_for_two,
        kernel_mxfp4_large_i73_n64_combine<C>,
        2,
        C::NUM_THREADS);
    const long long available_dynamic_for_two_value =
        availability_status == cudaSuccess
            ? static_cast<long long>(available_dynamic_for_two)
            : -1;
#if TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE
    TORCH_CHECK(
        std::strcmp(
            TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_SHA256,
            "173d51771205f51d6b89794bd69009f32e175a614d71b4a6d0a0334afd62f399")
            == 0,
        "N64-combine runtime-overlap proof identity mismatch");
    // numRegs/localSizeBytes from cudaFuncGetAttributes are driver-rounded
    // function metadata on this TCGEN route (the exact frozen cubin/SASS gate
    // proves REG120, STACK/LOCAL/LDL/STL zero).  Keep the runtime boundary on
    // properties the driver reports without that rounding: the launch must
    // admit all 256 threads and the real static+dynamic SMEM footprint must
    // remain inside the two-resident bound.
    TORCH_CHECK(
        attributes.maxThreadsPerBlock >= C::NUM_THREADS &&
            attributes.sharedSizeBytes + G::DYNAMIC_SHARED_MEMORY <= 67584,
        "N64-combine runtime-proof launch bounds failed: numRegs=",
        attributes.numRegs, ", localSizeBytes=", attributes.localSizeBytes,
        ", maxThreadsPerBlock=", attributes.maxThreadsPerBlock,
        ", sharedSizeBytes=", attributes.sharedSizeBytes,
        ", dynamicSharedBytes=", G::DYNAMIC_SHARED_MEMORY);
#endif
    const int published_resident_ctas =
        TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE
            ? 2
            : resident_at_requested;
    {
        std::lock_guard<std::mutex> lock(
            large_i73_n64_combine_topology_mutex);
        large_i73_n64_combine_last_topology = {
            1, batch, heads, seqlen, LOGICAL_QUERY_CTAS, GRID_CTAS,
            1, C::CLUSTER_SIZE, C::NUM_THREADS, 256, 2,
            published_resident_ctas, resident_at_zero,
            resident_at_requested,
            available_dynamic_for_two_value,
            static_cast<long long>(availability_status),
            G::DYNAMIC_SHARED_MEMORY,
            attributes.maxDynamicSharedSizeBytes,
            attributes.preferredShmemCarveout};
    }
#if !TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE
    TORCH_CHECK(
        resident_at_requested >= 2,
        "N64-combine failed the two-resident occupancy contract; ",
        "resident_at_zero=", resident_at_zero,
        ", resident_at_requested=", resident_at_requested,
        ", available_dynamic_for_two=", available_dynamic_for_two_value,
        ", available_dynamic_status=",
        cudaGetErrorName(availability_status), "(",
        static_cast<int>(availability_status), ")",
        ", requested_dynamic=", G::DYNAMIC_SHARED_MEMORY);
    TORCH_CHECK(
        availability_status == cudaSuccess,
        "N64-combine two-block dynamic-SMEM availability query failed: ",
        cudaGetErrorName(availability_status), "(",
        static_cast<int>(availability_status), ")");
#endif

#if TK_FA4_LARGE_I73_N64_COMBINE_CLUSTER_METADATA_FREE
    LaunchConfig<false, false> launch_config(
        dim3(GRID_CTAS), g.block(), G::DYNAMIC_SHARED_MEMORY,
        at::cuda::getCurrentCUDAStream());
#else
    LaunchConfig<true, false> launch_config(
        dim3(GRID_CTAS), g.block(), G::DYNAMIC_SHARED_MEMORY,
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
#endif
    CUDACHECK(cudaLaunchKernelEx(
        launch_config, kernel_mxfp4_large_i73_n64_combine<C>, g));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_streaming_live_mxfp4",
          &dispatch_large_i73_n64_combine,
          "Exact S4096/H64 physical-N64 logical-N128 combine route");
    m.def("read_large_i73_n64_combine_topology",
          &read_large_i73_n64_combine_topology,
          "Read the last N64-combine launch topology");
}

void reset_pairpsc_desc_diag() {
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_pairpsc_desc_diag));
    CUDACHECK(cudaMemset(ptr, 0, 16 * sizeof(unsigned long long)));
}

std::vector<unsigned long long> read_pairpsc_desc_diag() {
    std::vector<unsigned long long> out(16, 0);
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_pairpsc_desc_diag));
    CUDACHECK(cudaMemcpy(out.data(), ptr, 16 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return out;
}

void set_pairpsc_desc_diag_slot(int slot, unsigned long long value) {
    TORCH_CHECK(slot >= 0 && slot < 16, "diag slot out of range");
    void *ptr = nullptr;
    CUDACHECK(cudaGetSymbolAddress(&ptr, fp4pv_pairpsc_desc_diag));
    auto *typed = reinterpret_cast<unsigned long long *>(ptr);
    CUDACHECK(cudaMemcpy(typed + slot, &value, sizeof(unsigned long long), cudaMemcpyHostToDevice));
}

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
    check_fp4pv_mxfp4_v_scale_tensor(V_sc, batch, heads, seqlen, 128);

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
    check_fp4pv_mxfp4_v_scale_tensor(V_sc, batch, heads, seqlen, 128);

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
        const dim3 grid = use_persistent ? persistent_dim_fp4pv_exp<C>(batch, heads, seqlen) : fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen);
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
    const bool fwd_config_auto =
        fwd_config == nullptr || std::strcmp(fwd_config, "") == 0 ||
        std::strcmp(fwd_config, "auto") == 0;
    const char *p_bypass_env = std::getenv("TK_FA4_MXFP4_P_BYPASS_MATERIALIZATION");
    const bool p_bypass_materialization =
        p_bypass_env != nullptr &&
        std::strcmp(p_bypass_env, "") != 0 &&
        std::strcmp(p_bypass_env, "0") != 0 &&
        std::strcmp(p_bypass_env, "false") != 0 &&
        std::strcmp(p_bypass_env, "False") != 0 &&
        std::strcmp(p_bypass_env, "no") != 0 &&
        std::strcmp(p_bypass_env, "off") != 0;

    if (heads == 1) {
#ifdef TK_FA4_STAGEV_PROBE_BUILD
        TORCH_CHECK(false, "TK_FA4_STAGEV_PROBE_BUILD only supports multi-head MXFP4 forward probes");
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
        } else {
            TORCH_CHECK(false, "unsupported TK_FA4_FP4PV_FWD_CONFIG for H=1 MXFP4 forward experiment: ", fwd_config);
        }
#endif
    } else {
#ifdef TK_FA4_STAGEV_PROBE_BUILD
#ifdef TK_FA4_FP32PACK_ONLY_PROBE_BUILD
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
            launch_mxfp4(std::type_identity<config_fp4pv_3wg_dual_score_direct_pscale_scorepack_prescaled_floor_x1sc_fusedmax_qkscfix_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_dualaccum_directrescale_decoupled_pstage2_pregs_force_persistent<128, 128, 192, 128, 200, 56, 112, 1>>{}, true);
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

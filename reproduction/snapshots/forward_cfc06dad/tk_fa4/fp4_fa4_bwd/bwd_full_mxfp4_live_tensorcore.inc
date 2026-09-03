#include <c10/cuda/CUDAGuard.h>

// BWD-only full MXFP4 tensorcore orchestration from prequantized artifacts.
// This is the cutover away from the BF16 key-owned scaffold: dQ/dK use the
// live-dP MXFP4 producer/consumer path, and dV uses the MXFP4 PV rewrite.

std::vector<at::Tensor> dispatch_backward_full_mxfp4_tensorcore_prequantized(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor Q_col_fp4,
    at::Tensor Q_col_sc,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor DOut_col_fp4,
    at::Tensor DOut_col_sc,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor DV_K_as_Q_sc,
    at::Tensor DV_Q_as_K_sc,
    at::Tensor LSE,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    at::Tensor DK_pad_out,
    at::Tensor DV_out,
    double p_payload_scale,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool persistent_dv
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(Q_col_fp4); CHECK_INPUT(Q_col_sc);
    CHECK_INPUT(K_col_fp4); CHECK_INPUT(K_col_sc);
    CHECK_INPUT(DOut_col_fp4); CHECK_INPUT(DOut_col_sc);
    CHECK_INPUT(DOut_row_fp4); CHECK_INPUT(DOut_row_sc);
    CHECK_INPUT(V_row_fp4); CHECK_INPUT(V_row_sc);
    CHECK_INPUT(DV_K_as_Q_sc); CHECK_INPUT(DV_Q_as_K_sc);
    CHECK_INPUT(LSE); CHECK_INPUT(Delta);
    CHECK_INPUT(DQ_pad_out); CHECK_INPUT(DK_pad_out); CHECK_INPUT(DV_out);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_col_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q_col_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(K_col_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K_col_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(DOut_col_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "DOut_col_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(DOut_row_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "DOut_row_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(V_row_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_row_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Float8_e4m3fn, "Q_sc must use the B300 float8_e4m3fn scale layout");
    TORCH_CHECK(K_sc.dtype() == at::ScalarType::Float8_e4m3fn, "K_sc must use the B300 float8_e4m3fn scale layout");
    TORCH_CHECK(Q_col_sc.dtype() == at::ScalarType::Byte, "Q_col_sc must be uint8");
    TORCH_CHECK(K_col_sc.dtype() == at::ScalarType::Byte, "K_col_sc must be uint8");
    TORCH_CHECK(DOut_col_sc.dtype() == at::ScalarType::Byte, "DOut_col_sc must be uint8");
    TORCH_CHECK(DOut_row_sc.dtype() == at::ScalarType::Byte, "DOut_row_sc must be uint8");
    TORCH_CHECK(V_row_sc.dtype() == at::ScalarType::Byte, "V_row_sc must be uint8");
    TORCH_CHECK(DV_K_as_Q_sc.dtype() == at::ScalarType::Float8_e4m3fn, "DV_K_as_Q_sc must use the prepared float8_e4m3fn scale layout");
    TORCH_CHECK(DV_Q_as_K_sc.dtype() == at::ScalarType::Float8_e4m3fn, "DV_Q_as_K_sc must use the prepared float8_e4m3fn scale layout");
    TORCH_CHECK(Q_sg.dtype() == at::ScalarType::Float, "Q_sg must be float32");
    TORCH_CHECK(K_sg.dtype() == at::ScalarType::Float, "K_sg must be float32");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(DQ_pad_out.dtype() == at::ScalarType::BFloat16, "DQ_pad_out must be BF16");
    TORCH_CHECK(DK_pad_out.dtype() == at::ScalarType::BFloat16, "DK_pad_out must be BF16");
    TORCH_CHECK(DV_out.dtype() == at::ScalarType::BFloat16, "DV_out must be BF16");

    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q/K must have shape (B,H,S,96)");
    const int batch = static_cast<int>(Q.size(0));
    const int heads = static_cast<int>(Q.size(1));
    const int seqlen = static_cast<int>(Q.size(2));
    TORCH_CHECK(Q.size(3) == 96, "Q packed dim must be 96");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q");
    TORCH_CHECK(seqlen % 128 == 0, "seqlen must be divisible by 128");
    TORCH_CHECK(LSE.sizes() == at::IntArrayRef({batch, heads, 1, seqlen}), "LSE must have shape (B,H,1,S)");
    TORCH_CHECK(Delta.sizes() == at::IntArrayRef({batch, heads, seqlen}), "Delta must have shape (B,H,S)");
    TORCH_CHECK(DQ_pad_out.dim() == 4 && DQ_pad_out.size(0) == batch && DQ_pad_out.size(1) == heads &&
                    DQ_pad_out.size(2) == seqlen && DQ_pad_out.size(3) >= 192,
                "DQ_pad_out must have shape (B,H,S,Dpad>=192)");
    TORCH_CHECK(DK_pad_out.dim() == 4 && DK_pad_out.size(0) == batch && DK_pad_out.size(1) == heads &&
                    DK_pad_out.size(2) == seqlen && DK_pad_out.size(3) == DQ_pad_out.size(3),
                "DK_pad_out must have shape (B,H,S,Dpad) matching DQ_pad_out");
    TORCH_CHECK(DV_out.sizes() == at::IntArrayRef({batch, seqlen, heads, 128}), "DV_out must have shape (B,S,H,128)");
    TORCH_CHECK(std::isfinite(static_cast<float>(p_payload_scale)) && p_payload_scale > 0.0,
                "p_payload_scale must be finite and positive");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2,
                "p_quant_mode must be 0 (rte), 1 (ceil), or 2 (floor)");
    TORCH_CHECK(std::isfinite(static_cast<float>(softmax_scale)), "softmax_scale must be finite");

    bool overlap_dv = true;
    if (const char *overlap_env = std::getenv("TK_FA4_MXFP4_BWD_FULL_PREQUANT_OVERLAP_DV")) {
        overlap_dv = std::atoi(overlap_env) != 0;
    }

    cudaEvent_t dv_ready_event = nullptr;
    cudaEvent_t dv_done_event = nullptr;
    at::cuda::CUDAStream caller_stream = at::cuda::getCurrentCUDAStream(Q.device().index());
    at::cuda::CUDAStream dv_stream = caller_stream;
    if (overlap_dv) {
        dv_stream = at::cuda::getStreamFromPool(false, Q.device().index());
        CUDACHECK(cudaEventCreateWithFlags(&dv_ready_event, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&dv_done_event, cudaEventDisableTiming));
        CUDACHECK(cudaEventRecord(dv_ready_event, caller_stream.stream()));
        CUDACHECK(cudaStreamWaitEvent(dv_stream.stream(), dv_ready_event, 0));
    }

    {
        c10::cuda::CUDAStreamGuard dv_guard(dv_stream);
        dispatch_backward_dv_streaming_live_mxfp4_rewrite(
            K,
            DV_K_as_Q_sc,
            K_sg,
            Q,
            DV_Q_as_K_sc,
            Q_sg,
            DOut_col_fp4,
            DOut_col_sc,
            DV_out,
            LSE,
            p_payload_scale,
            p_quant_mode,
            persistent_dv);
    }
    if (overlap_dv) {
        CUDACHECK(cudaEventRecord(dv_done_event, dv_stream.stream()));
    }

    auto dq_dk = dispatch_backward_dq_dk_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_pair_out(
        Q,
        Q_sc,
        Q_sg,
        K,
        K_sc,
        K_sg,
        K_col_fp4,
        K_col_sc,
        Q_col_fp4,
        Q_col_sc,
        LSE,
        DOut_row_fp4,
        DOut_row_sc,
        V_row_fp4,
        V_row_sc,
        Delta,
        DQ_pad_out,
        DK_pad_out,
        p_quant_mode,
        softmax_scale,
        debug_stage_cut);
    if (overlap_dv) {
        CUDACHECK(cudaStreamWaitEvent(caller_stream.stream(), dv_done_event, 0));
        CUDACHECK(cudaEventDestroy(dv_ready_event));
        CUDACHECK(cudaEventDestroy(dv_done_event));
    }
    TORCH_CHECK(dq_dk.size() == 2, "live-dP dQ/dK pair must return two tensors");

    const int64_t d_pad = DQ_pad_out.size(3);
    const std::vector<int64_t> active_bshd_sizes = {batch, seqlen, heads, 192};
    const std::vector<int64_t> active_bshd_strides = {
        static_cast<int64_t>(heads) * seqlen * d_pad,
        d_pad,
        static_cast<int64_t>(seqlen) * d_pad,
        1,
    };
    return {
        dq_dk[0].as_strided(active_bshd_sizes, active_bshd_strides),
        dq_dk[1].as_strided(active_bshd_sizes, active_bshd_strides),
        DV_out,
    };
}

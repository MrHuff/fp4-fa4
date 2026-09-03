#pragma once

// BWD-only schedule port landing zone.
//
// Target schedule:
//   key-owned tile loop:
//     score = K_tile @ Q_tile^T
//     P     = softmax(score - LSE_q)
//     dP    = dO_q @ V_k^T
//     dS    = P * (dP - Delta_q) * softmax_scale
//     dV_k += P^T  @ dO_q
//     dK_k += dS^T @ Q_q
//
// This file is intentionally under fp4_fa4_bwd so the fused dK/dV rewrite can
// evolve without editing the forward-owned experiment or dispatch files.  The
// current entrypoint separates the ownership boundary explicitly: dQ remains
// query-owned, while dK and dV are launched as key-owned sidecar work.  The
// actual fused dK+dV kernel should replace the two key-owned launches in this
// body.

std::vector<at::Tensor> dispatch_backward_key_owned_dkdv_fused_probe_prequantized(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor LSE,
    at::Tensor Delta,
    at::Tensor DK_pad_out,
    at::Tensor DV_out,
    double softmax_scale
);

std::vector<at::Tensor> dispatch_backward_key_owned_dqdkdv_single_kernel_probe_prequantized(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor LSE,
    at::Tensor Delta,
    at::Tensor DK_pad_out,
    at::Tensor DV_out,
    double softmax_scale
);

std::vector<at::Tensor> dispatch_backward_full_mxfp4_tensorcore_key_owned_schedule_prequantized(
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
    bool use_single_kernel_probe = false;
    if (const char *single_kernel_env = std::getenv("TK_FA4_MXFP4_BWD_SINGLE_KERNEL_PROBE")) {
        use_single_kernel_probe = std::atoi(single_kernel_env) != 0;
    }
    if (use_single_kernel_probe) {
        auto dqdkdv = dispatch_backward_key_owned_dqdkdv_single_kernel_probe_prequantized(
            Q,
            Q_sc,
            Q_sg,
            K,
            K_sc,
            K_sg,
            DOut_row_fp4,
            DOut_row_sc,
            V_row_fp4,
            V_row_sc,
            LSE,
            Delta,
            DK_pad_out,
            DV_out,
            softmax_scale);
        TORCH_CHECK(dqdkdv.size() == 3, "single-kernel probe must return dQ, dK, and dV");

        const int batch = static_cast<int>(Q.size(0));
        const int heads = static_cast<int>(Q.size(1));
        const int seqlen = static_cast<int>(Q.size(2));
        const int64_t dq_d = dqdkdv[0].size(3);
        const int64_t dk_d_pad = DK_pad_out.size(3);
        const std::vector<int64_t> active_bshd_sizes = {batch, seqlen, heads, 192};
        const std::vector<int64_t> dq_bshd_strides = {
            static_cast<int64_t>(heads) * seqlen * dq_d,
            dq_d,
            static_cast<int64_t>(seqlen) * dq_d,
            1,
        };
        const std::vector<int64_t> dk_bshd_strides = {
            static_cast<int64_t>(heads) * seqlen * dk_d_pad,
            dk_d_pad,
            static_cast<int64_t>(seqlen) * dk_d_pad,
            1,
        };
        return {
            dqdkdv[0].as_strided(active_bshd_sizes, dq_bshd_strides),
            dqdkdv[1].as_strided(active_bshd_sizes, dk_bshd_strides),
            dqdkdv[2],
        };
    }

    bool use_fused_dkdv_probe = false;
    if (const char *fused_dkdv_env = std::getenv("TK_FA4_MXFP4_BWD_KEY_OWNED_FUSED_DKDV_PROBE")) {
        use_fused_dkdv_probe = std::atoi(fused_dkdv_env) != 0;
    }
    if (use_fused_dkdv_probe) {
        bool overlap_fused_dkdv = true;
        if (const char *overlap_env = std::getenv("TK_FA4_MXFP4_BWD_KEY_OWNED_FUSED_DKDV_OVERLAP")) {
            overlap_fused_dkdv = std::atoi(overlap_env) != 0;
        }
        if (fp4fa4_bwd_env_enabled("TK_FA4_MXFP4_BWD_PROFILE_COMPONENTS")) {
            overlap_fused_dkdv = false;
        }

        at::Tensor dq;
        std::vector<at::Tensor> dkdv;
        if (overlap_fused_dkdv) {
            at::cuda::CUDAStream caller_stream = at::cuda::getCurrentCUDAStream(Q.device().index());
            at::cuda::CUDAStream dq_stream = at::cuda::getStreamFromPool(false, Q.device().index());
            at::cuda::CUDAStream dkdv_stream = at::cuda::getStreamFromPool(false, Q.device().index());
            cudaEvent_t ready_event = nullptr;
            cudaEvent_t dq_done_event = nullptr;
            cudaEvent_t dkdv_done_event = nullptr;
            CUDACHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
            CUDACHECK(cudaEventCreateWithFlags(&dq_done_event, cudaEventDisableTiming));
            CUDACHECK(cudaEventCreateWithFlags(&dkdv_done_event, cudaEventDisableTiming));
            CUDACHECK(cudaEventRecord(ready_event, caller_stream.stream()));
            CUDACHECK(cudaStreamWaitEvent(dq_stream.stream(), ready_event, 0));
            CUDACHECK(cudaStreamWaitEvent(dkdv_stream.stream(), ready_event, 0));
            {
                c10::cuda::CUDAStreamGuard dq_guard(dq_stream);
                dq = dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_out(
                    Q,
                    Q_sc,
                    Q_sg,
                    K,
                    K_sc,
                    K_sg,
                    K_col_fp4,
                    K_col_sc,
                    LSE,
                    DOut_row_fp4,
                    DOut_row_sc,
                    V_row_fp4,
                    V_row_sc,
                    Delta,
                    DQ_pad_out,
                    p_quant_mode,
                    softmax_scale,
                    debug_stage_cut);
            }
            CUDACHECK(cudaEventRecord(dq_done_event, dq_stream.stream()));
            {
                c10::cuda::CUDAStreamGuard dkdv_guard(dkdv_stream);
                dkdv = dispatch_backward_key_owned_dkdv_fused_probe_prequantized(
                    Q,
                    Q_sc,
                    Q_sg,
                    K,
                    K_sc,
                    K_sg,
                    DOut_row_fp4,
                    DOut_row_sc,
                    V_row_fp4,
                    V_row_sc,
                    LSE,
                    Delta,
                    DK_pad_out,
                    DV_out,
                    softmax_scale);
            }
            CUDACHECK(cudaEventRecord(dkdv_done_event, dkdv_stream.stream()));
            CUDACHECK(cudaStreamWaitEvent(caller_stream.stream(), dq_done_event, 0));
            CUDACHECK(cudaStreamWaitEvent(caller_stream.stream(), dkdv_done_event, 0));
            CUDACHECK(cudaEventDestroy(ready_event));
            CUDACHECK(cudaEventDestroy(dq_done_event));
            CUDACHECK(cudaEventDestroy(dkdv_done_event));
        } else {
            dq = dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_out(
                Q,
                Q_sc,
                Q_sg,
                K,
                K_sc,
                K_sg,
                K_col_fp4,
                K_col_sc,
                LSE,
                DOut_row_fp4,
                DOut_row_sc,
                V_row_fp4,
                V_row_sc,
                Delta,
                DQ_pad_out,
                p_quant_mode,
                softmax_scale,
                debug_stage_cut);
            dkdv = dispatch_backward_key_owned_dkdv_fused_probe_prequantized(
                Q,
                Q_sc,
                Q_sg,
                K,
                K_sc,
                K_sg,
                DOut_row_fp4,
                DOut_row_sc,
                V_row_fp4,
                V_row_sc,
                LSE,
                Delta,
                DK_pad_out,
                DV_out,
                softmax_scale);
        }
        TORCH_CHECK(dkdv.size() == 2, "fused dK+dV probe must return dK and dV");

        const int batch = static_cast<int>(Q.size(0));
        const int heads = static_cast<int>(Q.size(1));
        const int seqlen = static_cast<int>(Q.size(2));
        const int64_t d_pad = DQ_pad_out.size(3);
        const std::vector<int64_t> active_bshd_sizes = {batch, seqlen, heads, 192};
        const std::vector<int64_t> active_bshd_strides = {
            static_cast<int64_t>(heads) * seqlen * d_pad,
            d_pad,
            static_cast<int64_t>(seqlen) * d_pad,
            1,
        };
        return {
            dq.as_strided(active_bshd_sizes, active_bshd_strides),
            dkdv[0].as_strided(active_bshd_sizes, active_bshd_strides),
            dkdv[1],
        };
    }

    int overlap_safe_seqlen = 8192;
    if (const char *overlap_safe_env = std::getenv("TK_FA4_MXFP4_BWD_KEY_OWNED_SCHEDULE_OVERLAP_MAX_SEQLEN")) {
        overlap_safe_seqlen = std::atoi(overlap_safe_env);
    }
    bool overlap_key_owned = overlap_safe_seqlen >= 0 && static_cast<int>(Q.size(2)) <= overlap_safe_seqlen;
    const char *overlap_env = std::getenv("TK_FA4_MXFP4_BWD_KEY_OWNED_SCHEDULE_OVERLAP");
    if (overlap_env != nullptr) {
        overlap_key_owned = std::atoi(overlap_env) != 0;
    }
    // The live-pair schedule already keeps the expensive dQ/dK work local to
    // the fast TMEM path.  Running the dV sidecar on another stream helps at
    // 2K/8K, but can wedge longer contexts in this wrapper, so gate the
    // production default by sequence length.  The env overrides above remain
    // the escape hatches for shape-specific stream experiments.
    if (fp4fa4_bwd_env_enabled("TK_FA4_MXFP4_BWD_PROFILE_COMPONENTS")) {
        overlap_key_owned = false;
    }

    at::cuda::CUDAStream caller_stream = at::cuda::getCurrentCUDAStream(Q.device().index());
    at::cuda::CUDAStream dv_stream = caller_stream;
    cudaEvent_t ready_event = nullptr;
    cudaEvent_t dv_done_event = nullptr;
    if (overlap_key_owned) {
        dv_stream = at::cuda::getStreamFromPool(false, Q.device().index());
        CUDACHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&dv_done_event, cudaEventDisableTiming));
        CUDACHECK(cudaEventRecord(ready_event, caller_stream.stream()));
        CUDACHECK(cudaStreamWaitEvent(dv_stream.stream(), ready_event, 0));
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
    if (overlap_key_owned) {
        CUDACHECK(cudaEventRecord(dv_done_event, dv_stream.stream()));
    }

    // Use the current live-dP MXFP4 tensorcore pair launcher rather than the
    // row-dS staged path.  This is still a launch-level schedule port, but it
    // keeps dS local to the replay kernels and avoids the slower serial dQ/dK
    // order in this wrapper.
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
    TORCH_CHECK(dq_dk.size() == 2, "key-owned schedule pair launcher must return dQ and dK");

    if (overlap_key_owned) {
        CUDACHECK(cudaStreamWaitEvent(caller_stream.stream(), dv_done_event, 0));
        CUDACHECK(cudaEventDestroy(ready_event));
        CUDACHECK(cudaEventDestroy(dv_done_event));
    }

    const int batch = static_cast<int>(Q.size(0));
    const int heads = static_cast<int>(Q.size(1));
    const int seqlen = static_cast<int>(Q.size(2));
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

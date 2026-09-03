// Backward pybind registrations for fp4_fa4_bwd.cu.
// Kept under fp4_fa4_bwd so backward API changes do not touch forward-owned binding files.

#if !defined(TK_FA4_FORWARD_ONLY_BUILD) && !defined(TK_FA4_DQ_ONLY_BUILD)
    m.def("backward_full_mxfp4_tensorcore", &dispatch_backward_full_mxfp4_tensorcore, "Fail-closed contract for true full-MXFP4 tensorcore backward from prequantized saved artifacts");
    m.def("backward_full_mxfp4_tensorcore_prequantized", &dispatch_backward_full_mxfp4_tensorcore_prequantized,
          "Full BWD MXFP4 tensorcore path from prequantized dO/V artifacts; no BF16 Q/K/V bridge and no global row-dS boundary",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("Q_col_fp4"), pybind11::arg("Q_col_sc"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("DOut_col_fp4"), pybind11::arg("DOut_col_sc"),
          pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"),
          pybind11::arg("DV_K_as_Q_sc"), pybind11::arg("DV_Q_as_K_sc"),
          pybind11::arg("LSE"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DK_pad_out"), pybind11::arg("DV_out"),
          pybind11::arg("p_payload_scale"), pybind11::arg("p_quant_mode"),
          pybind11::arg("softmax_scale"), pybind11::arg("debug_stage_cut"),
          pybind11::arg("persistent_dv"));
    m.def("backward_full_mxfp4_tensorcore_key_owned_schedule_prequantized",
          &dispatch_backward_full_mxfp4_tensorcore_key_owned_schedule_prequantized,
          "BWD-only key-owned schedule entrypoint for the fused dK/dV MXFP4 rewrite; currently shares the stable prequantized contract",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("Q_col_fp4"), pybind11::arg("Q_col_sc"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("DOut_col_fp4"), pybind11::arg("DOut_col_sc"),
          pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"),
          pybind11::arg("DV_K_as_Q_sc"), pybind11::arg("DV_Q_as_K_sc"),
          pybind11::arg("LSE"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DK_pad_out"), pybind11::arg("DV_out"),
          pybind11::arg("p_payload_scale"), pybind11::arg("p_quant_mode"),
          pybind11::arg("softmax_scale"), pybind11::arg("debug_stage_cut"),
          pybind11::arg("persistent_dv"));
    m.def("backward_key_owned_dkdv_fused_probe_prequantized",
          &dispatch_backward_key_owned_dkdv_fused_probe_prequantized,
          "Direct BWD-only fused key-owned dK+dV probe from prequantized row dO/V artifacts",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"),
          pybind11::arg("LSE"), pybind11::arg("Delta"),
          pybind11::arg("DK_pad_out"), pybind11::arg("DV_out"),
          pybind11::arg("softmax_scale"));
    m.def("backward_key_owned_dqdkdv_single_kernel_probe_prequantized",
          &dispatch_backward_key_owned_dqdkdv_single_kernel_probe_prequantized,
          "Experimental single BWD compute kernel for key-owned dQ+dK+dV from prequantized row dO/V artifacts",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"),
          pybind11::arg("LSE"), pybind11::arg("Delta"),
          pybind11::arg("DK_pad_out"), pybind11::arg("DV_out"),
          pybind11::arg("softmax_scale"));
    m.def("backward_dv_streaming_live_localcta", &dispatch_backward_dv_streaming_live_localcta, "Streaming live FP4 dV path with external forward LSE (QK fp4, P quantized live, dO fp4)");
#if !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("backward_dv_streaming_live_mxfp4", &dispatch_backward_dv_streaming_live_mxfp4, "Streaming live FP4 dV path with external forward LSE and MXFP4 softmax/PV scales");
#endif
    m.def("backward_dv_streaming_live_mxfp4_rewrite", &dispatch_backward_dv_streaming_live_mxfp4_rewrite, "Production MXFP4 dV rewrite path with saved forward LSE and a hardcoded PV schedule");
#endif
#if !defined(TK_FA4_FORWARD_ONLY_BUILD) && !defined(TK_FA4_DQ_ONLY_BUILD)
    m.def("forward_streaming_live_qk_only", &dispatch_forward_streaming_live_qk_only, "Streaming live FP4 QK+softmax sidecar benchmark returning BF16 P and LSE");
    m.def("forward_streaming_live_qk_only_saved_lse", &dispatch_forward_streaming_live_qk_only_saved_lse, "Streaming live FP4 QK+softmax sidecar benchmark returning BF16 P from saved forward LSE");
    m.def("backward_ds_streaming_live_qk_saved_lse", &dispatch_backward_ds_streaming_live_qk_saved_lse, "Streaming live FP4 QK+saved-LSE sidecar returning BF16 dS without materializing P");
    m.def("backward_ds_streaming_live_qk_saved_lse_row_tile", &dispatch_backward_ds_streaming_live_qk_saved_lse_row_tile, "Streaming live FP4 QK+saved-LSE sidecar returning one BF16 dS row tile");
    m.def("backward_ds_streaming_live_qk_saved_lse_row_cluster", &dispatch_backward_ds_streaming_live_qk_saved_lse_row_cluster, "Streaming live FP4 QK+saved-LSE sidecar returning one BF16 dS row cluster");
    m.def("backward_ds_streaming_live_qk_saved_lse_clustered_full", &dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full, "Streaming live FP4 QK+saved-LSE sidecar assembling full BF16 dS from safe row clusters");
    m.def("backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4", &dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4, "Streaming live FP4 QK+saved-LSE dS emitting direct row-major MXFP4 payload/scales plus BF16 validation tensor");
    m.def("backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4_no_bf16", &dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4_no_bf16, "Streaming live FP4 QK+saved-LSE dS emitting direct row-major MXFP4 payload/scales without BF16 validation tensor");
    m.def("backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4", &dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4, "Streaming live FP4 QK+saved-LSE dS full-grid one-launch row-major MXFP4 payload/scales plus BF16 validation tensor");
    m.def("backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_no_bf16", &dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_no_bf16, "Streaming live FP4 QK+saved-LSE dS full-grid row-major MXFP4 payload/scales without BF16 validation tensor");
    m.def("backward_ds_streaming_live_qk_saved_lse_fullgrid_row_col_mxfp4_no_bf16", &dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_col_mxfp4_no_bf16, "Streaming live FP4 QK+saved-LSE dS full-grid direct row-major and transposed MXFP4 payload/scales without BF16 validation tensor");
    m.def("backward_ds_streaming_live_qk_saved_lse_fullgrid_row_col_mxfp4_window_no_bf16", &dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_col_mxfp4_window_no_bf16, "Streaming live FP4 QK+saved-LSE dS full-grid direct row-major and transposed MXFP4 payload/scales with runtime K-tile window and without BF16 validation tensor");
    m.def("backward_ds_streaming_live_qk_saved_lse_fullgrid_col_mxfp4_window_no_bf16", &dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_col_mxfp4_window_no_bf16, "Streaming live FP4 QK+saved-LSE dS full-grid transposed MXFP4 payload/scales with runtime K-tile window and without row dS materialization");
    m.def("backward_ds_streaming_live_qk_saved_lse_fullgrid_fused_stage_mxfp4_no_global", &dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_fused_stage_mxfp4_no_global, "Streaming live FP4 QK+saved-LSE dS fused local MXFP4 stage probe without global dS payload/scales");
    m.def("backward_ds_streaming_live_qk_saved_lse_fullgrid_fused_col_stage_mxfp4_no_global", &dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_fused_col_stage_mxfp4_no_global, "Streaming live FP4 QK+saved-LSE dS fused local transposed MXFP4 stage probe without global dS payload/scales");
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4, "Streaming live FP4 QK+saved-LSE fused dS-to-dQ using saved MXFP4 K_col artifacts, without global dS row payload/scales");
    m.def("backward_dk_streaming_live_qk_saved_lse_fused_mxfp4", &dispatch_backward_dk_streaming_live_qk_saved_lse_fused_mxfp4, "Streaming live FP4 QK+saved-LSE fused dS^T-to-dK using saved MXFP4 Q_col artifacts, without global dS payload/scales");
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_out, "Streaming live FP4 QK+saved-LSE fused dS-to-dQ into preallocated dQ_pad output");
    m.def("backward_dk_streaming_live_qk_saved_lse_fused_mxfp4_out", &dispatch_backward_dk_streaming_live_qk_saved_lse_fused_mxfp4_out, "Streaming live FP4 QK+saved-LSE fused dS^T-to-dK into preallocated dK_pad output");
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_out,
          "Streaming live FP4 QK+saved-LSE fused dS-to-dQ with live dP=dO@V^T and no global dS outputs",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"));
    m.def("backward_dk_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_out", &dispatch_backward_dk_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_out,
          "Streaming live FP4 QK+saved-LSE fused dS^T-to-dK with live dP=dO@V^T and no global dS outputs",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("Q_col_fp4"), pybind11::arg("Q_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DK_pad_out"), pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"));
    m.def("backward_dq_dk_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_pair_out", &dispatch_backward_dq_dk_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_pair_out,
          "BWD-only live-dP dQ+dK pair entrypoint with preallocated pad outputs; currently wraps the two fused replay specializations",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("Q_col_fp4"), pybind11::arg("Q_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DK_pad_out"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"), pybind11::arg("debug_stage_cut"));
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_dq_bshd_out",
          &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_dq_bshd_out,
          "Streaming live-dP fused dS-to-dQ with direct BSHD dQ output and row-major MXFP4 dS outputs",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DQ_bshd_out"), pybind11::arg("DS_fp4_row_out"), pybind11::arg("DS_sc_row_out"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row,
          "Streaming live FP4 QK+saved-LSE fused dS-to-dQ using saved MXFP4 K_col artifacts while emitting row-major MXFP4 dS for AtBt dK",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DP"), pybind11::arg("Delta"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_out,
          "Streaming live FP4 QK+saved-LSE fused dS-to-dQ with preallocated dQ_pad and row-major MXFP4 dS outputs",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DP"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DS_fp4_row_out"), pybind11::arg("DS_sc_row_out"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_out,
          "Streaming live FP4 QK+saved-LSE fused dS-to-dQ with live dP=dO@V^T and preallocated row-major MXFP4 dS outputs",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DS_fp4_row_out"), pybind11::arg("DS_sc_row_out"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_debug_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_debug_out,
          "Debug live FP4 QK+saved-LSE fused dS-to-dQ with live dP capture into a BF16 DP tensor",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DS_fp4_row_out"), pybind11::arg("DS_sc_row_out"),
          pybind11::arg("DP_live_debug_out"), pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_dq_bshd", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_dq_bshd,
          "Streaming live FP4 QK+saved-LSE fused dS-to-dQ returning final-layout BF16 dQ while emitting row-major MXFP4 dS for AtBt dK",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DP"), pybind11::arg("Delta"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_dq_bshd_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_dq_bshd_out,
          "Streaming live FP4 QK+saved-LSE fused dS-to-dQ with preallocated final-layout BF16 dQ and row-major MXFP4 dS outputs",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DP"), pybind11::arg("Delta"),
          pybind11::arg("DQ_bshd_out"), pybind11::arg("DS_fp4_row_out"), pybind11::arg("DS_sc_row_out"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col, "Streaming live FP4 QK+saved-LSE fused dS-to-dQ using saved MXFP4 K_col artifacts while emitting transposed MXFP4 dS for dK");
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_nozero", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_nozero, "Streaming live FP4 QK+saved-LSE fused dS-to-dQ using saved MXFP4 K_col artifacts while emitting transposed MXFP4 dS for masked dK without prezeroing dS scales");
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_out, "Streaming live FP4 QK+saved-LSE fused dS-to-dQ with preallocated dQ_pad and transposed MXFP4 dS outputs");
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_out_nozero", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_out_nozero, "Streaming live FP4 QK+saved-LSE fused dS-to-dQ with preallocated dQ_pad/transposed dS outputs for masked dK without prezeroing dS scales");
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_live_dp_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_live_dp_out,
          "Streaming live FP4 QK+saved-LSE fused dS-to-dQ with live dP=dO@V^T and preallocated transposed MXFP4 dS outputs",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DS_fp4_col_out"), pybind11::arg("DS_sc_col_out"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_col_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_dq_bshd", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_dq_bshd, "Streaming live FP4 QK+saved-LSE fused dS-to-dQ returning final-layout BF16 dQ while emitting transposed MXFP4 dS for dK");
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_dq_bshd_nozero", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_dq_bshd_nozero, "Streaming live FP4 QK+saved-LSE fused dS-to-dQ returning final-layout BF16 dQ while emitting transposed MXFP4 dS for masked dK without prezeroing dS scales");
    m.def("finalize_dq_dk_pad_bhsd_to_bshd", &dispatch_finalize_dq_dk_pad_bhsd_to_bshd, "Finalize padded dQ/dK tensors from (B,H,S,256) into BF16 gradients shaped (B,S,H,192) in one kernel");
    m.def("finalize_dk_pad_bhsd_to_bshd", &dispatch_finalize_dk_pad_bhsd_to_bshd, "Finalize padded dK tensor from (B,H,S,256) into BF16 gradient shaped (B,S,H,192)");
    m.def("backward_ds_streaming_live_qk_saved_lse_clustered_full_window", &dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_window, "Streaming live FP4 QK+saved-LSE BF16 dS with runtime K-tile window for profiling");
    m.def("backward_ds_streaming_live_qk_saved_lse_row_tile_maxk_debug", &dispatch_backward_ds_streaming_live_qk_saved_lse_row_tile_maxk_debug, "Debug selected-row saved-LSE dS with a capped K-tile count");
    m.def("backward_ds_from_p_bf16", &dispatch_backward_ds_from_p_bf16, "Apply dS = P * (dP - delta) * softmax_scale for saved-LSE MXFP4 backward staging");
    m.def("bf16_row_dot_delta_bshd", &dispatch_bf16_row_dot_delta_bshd, "Compute delta=sum_d Out*dOut for BF16 tensors shaped (B,S,H,128), returning (B,H,S) float32");
    m.def("bf16_transpose_to_mxfp4", &dispatch_bf16_transpose_to_mxfp4, "Quantize a BF16 matrix transpose directly into an MXFP4 GEMM artifact");
    m.def("row_mxfp4_transpose_to_mxfp4", &dispatch_row_mxfp4_transpose_to_mxfp4, "Transpose a row-major MXFP4 matrix artifact into another MXFP4 GEMM artifact");
    m.def("forward_streaming_live_qk_only_lse_only", &dispatch_forward_streaming_live_qk_only_lse_only, "Streaming live FP4 QK+softmax sidecar benchmark returning only LSE");
#if !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("pv_from_p", &dispatch_pv_from_p, "PV-only matmul from prequantized P and V (FP4, B300, causal sidecar)");
    m.def("two_tile_pv_tmem_debug", &dispatch_two_tile_pv_tmem_debug, "Two-tile FP4 PV TMEM snapshot debug path from stored P/V");
    m.def("two_tile_pv_mainpath_debug", &dispatch_two_tile_pv_mainpath_debug, "Two-tile FP4 PV main-path debug from stored P/V");
    m.def("two_tile_correction_replay_debug", &dispatch_two_tile_correction_replay_debug, "Replay fused online correction/materialization from a pre-correction output snapshot");
#endif
    m.def("quantize_p_from_scores_debug", &dispatch_quantize_p_from_scores_debug, "Dump live-quantized P, scales, and LSE from the FP4 QK+softmax path");
    m.def("quantize_p_from_scores_with_live_bf16_ref_debug", &dispatch_quantize_p_from_scores_with_live_bf16_ref_debug, "Dump live-quantized P, prepared scales, live BF16 tilework, and row stats from the FP4 QK+softmax path");
    m.def("quantize_p_from_scores_tile_window_with_live_bf16_ref_debug", &dispatch_quantize_p_from_scores_tile_window_with_live_bf16_ref_debug, "Dump a selected live-quantized P tile window, prepared scales, live BF16 tilework, and row stats from the FP4 QK+softmax path");
    m.def("quantize_p_from_scores_tile_window_live_bf16_only_debug", &dispatch_quantize_p_from_scores_tile_window_live_bf16_only_debug, "Dump a selected live BF16 P tile window and row stats from the FP4 QK+softmax path without the FP4 quantizer consumers");
    m.def("quantize_p_from_scores_tile_window_live_bf16_only_minimal_debug", &dispatch_quantize_p_from_scores_tile_window_live_bf16_only_minimal_debug, "Dump a selected live BF16 P tile window from the FP4 QK+softmax path using the exact-grid minimal probe");
    m.def("probe_loaded_qk_scale_chunks_debug", &dispatch_probe_loaded_qk_scale_chunks_debug, "Probe a selected prepared Q/K scale chunk after the isolated SMEM-to-TMEM load path");
    m.def("quantize_p_from_scores_row_tile_window_live_bf16_only_minimal_debug", &dispatch_quantize_p_from_scores_row_tile_window_live_bf16_only_minimal_debug, "Dump a selected row tile of live BF16 P from the FP4 QK+softmax path using the exact-grid row-cluster probe");
    m.def("quantize_p_from_scores_with_live_bf16_ref_debug_direct", &dispatch_quantize_p_from_scores_with_live_bf16_ref_debug_direct, "Dump direct-style live-quantized P, prepared scales, live BF16 tilework, and row stats from the FP4 QK+softmax path");
    m.def("quantize_p_from_scores_with_live_row_scales_and_bf16_ref_debug", &dispatch_quantize_p_from_scores_with_live_row_scales_and_bf16_ref_debug, "Dump live-quantized P, prepared scales, raw live row-major scales, live BF16 tilework, and row stats from the FP4 QK+softmax path");
#endif
#if defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_FORWARD_ONLY_BUILD)
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_out,
          "Streaming live FP4 QK+saved-LSE fused dS-to-dQ with live dP=dO@V^T and preallocated row-major MXFP4 dS outputs",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DS_fp4_row_out"), pybind11::arg("DS_sc_row_out"),
          pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
    m.def("backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_debug_out", &dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_debug_out,
          "Debug live FP4 QK+saved-LSE fused dS-to-dQ with live dP capture into a BF16 DP tensor",
          pybind11::arg("Q"), pybind11::arg("Q_sc"), pybind11::arg("Q_sg"),
          pybind11::arg("K"), pybind11::arg("K_sc"), pybind11::arg("K_sg"),
          pybind11::arg("K_col_fp4"), pybind11::arg("K_col_sc"),
          pybind11::arg("LSE"), pybind11::arg("DOut_row_fp4"), pybind11::arg("DOut_row_sc"),
          pybind11::arg("V_row_fp4"), pybind11::arg("V_row_sc"), pybind11::arg("Delta"),
          pybind11::arg("DQ_pad_out"), pybind11::arg("DS_fp4_row_out"), pybind11::arg("DS_sc_row_out"),
          pybind11::arg("DP_live_debug_out"), pybind11::arg("p_quant_mode"), pybind11::arg("softmax_scale"),
          pybind11::arg("debug_stage_cut"), pybind11::arg("zero_ds_row_sc") = true);
#endif

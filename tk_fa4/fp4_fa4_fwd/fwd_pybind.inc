// Forward pybind registrations for bf16_b300_mha_causal_fp4_pv_experiments.cu.
// Included from the module wrapper in bf16_b300_mha_causal_fp4_pv_experiments_pybind.inc.

#if TK_FA4_RAW_FP4_FACTOR_PROBE_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("raw_fp4_factor_probe", &dispatch_raw_fp4_factor_probe,
           "Compile-gated raw kind::f8f6f4 FP4 matmul with zero scale TMEM");
    m.def("raw_fp4_isa_extent_probe", &dispatch_raw_fp4_isa_extent_probe,
           "Compile-gated one-instruction raw E2M1 dense-K extent probe");
    m.def("raw_fp4_two_query_probe", &dispatch_raw_fp4_two_query_probe,
          "Compile-gated four-TMEM-tile two-query raw FP4 attention ceiling");
#if defined(TK_FA4_DQ_ONLY_BUILD)
    m.def("mxfp4_tcgen_shape_debug", &dispatch_mxfp4_tcgen_shape_debug,
          "Standalone scaled MXFP4 TCGEN control for the raw FP4 gate");
#endif
#endif

#if TK_FA4_QUARTER_LIFETIME_TWO_QUERY_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_quarter_lifetime_two_query_overlay",
          &dispatch_quarter_lifetime_two_query_overlay,
          "Full-grid production-derived quarter-lifetime paired-query MXFP4 forward");
    m.def("mxfp4_quarter_lifetime_two_query_overlay_pruned",
          &dispatch_quarter_lifetime_two_query_overlay_pruned,
          "Quarter-lifetime paired-query forward with dominated lifecycle events removed");
    m.def("mxfp4_quarter_lifetime_two_query_overlay_pruned_fused",
          &dispatch_quarter_lifetime_two_query_overlay_pruned_fused,
          "Pruned quarter-lifetime forward with correction publication fused into R0");
#endif

#if TK_FA4_RAW_FP4_THROUGHPUT_PROBE_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("raw_fp4_throughput_probe", &dispatch_raw_fp4_throughput_probe,
          "Saturated matched M128xN128xK128 raw/scaled/sparse/BF16 TCGEN throughput probe");
#endif

#if TK_FA4_MXFP4_QK_CTA_GROUP_AB_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_qk_group1_issue_ceiling",
          &dispatch_mxfp4_qk_group1_issue_ceiling,
          "Work-normalized M128xN128xK192 group-1 QK issue ceiling");
    m.def("mxfp4_qk_group1_production_cadence",
          &dispatch_mxfp4_qk_group1_production_cadence,
          "Work-normalized M128xN128xK192 group-1 QK completion cadence");
    m.def("mxfp4_qk_group2_issue_ceiling",
          &dispatch_mxfp4_qk_group2_issue_ceiling,
          "Native M256xN128xK192 group-2 QK issue ceiling");
    m.def("mxfp4_qk_group2_production_cadence",
          &dispatch_mxfp4_qk_group2_production_cadence,
          "Native M256xN128xK192 group-2 QK completion cadence");
#endif

#if TK_FA4_SCALED_K64_TWO_QUERY_CEILING_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_scaled_k64_two_query_ceiling",
          &dispatch_scaled_k64_two_query_ceiling,
          "Trace-free scaled-K64 four-TMEM-tile two-query fake-P ceiling");
    m.def("mxfp4_scaled_k64_one_query_ceiling",
          &dispatch_scaled_k64_one_query_ceiling,
          "Trace-free matched one-query scaled-K64 fake-P ceiling");
    m.def("mxfp4_scaled_k64_two_query_ceiling_audit",
          &dispatch_scaled_k64_two_query_ceiling_audit,
          "Non-authoritative score/count audit clone of the two-query ceiling");
    m.def("mxfp4_scaled_k64_one_query_ceiling_audit",
          &dispatch_scaled_k64_one_query_ceiling_audit,
          "Non-authoritative score/count audit clone of the one-query ceiling");
    m.def("mxfp4_scaled_k64_two_query_ceiling_deferpv0",
          &dispatch_scaled_k64_two_query_ceiling_deferpv0,
          "Trace-free paired ceiling with deferred q0 PV completion wait");
    m.def("mxfp4_scaled_k64_two_query_ceiling_deferpv0_audit",
          &dispatch_scaled_k64_two_query_ceiling_deferpv0_audit,
          "Non-authoritative audit clone of deferred-q0-PV ceiling");
#endif

#if TK_FA4_SCALED_K64_TWO_QUERY_REAL_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_scaled_k64_two_query_real_serial",
          &dispatch_scaled_k64_two_query_real_serial,
          "Default-off four-TMEM adjacent-query real causal-softmax serial control");
    m.def("mxfp4_scaled_k64_two_query_real_overlap",
          &dispatch_scaled_k64_two_query_real_overlap,
          "Default-off four-TMEM adjacent-query real causal-softmax overlapped route");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg",
          &dispatch_scaled_k64_two_query_real_overlap_3wg,
          "Default-off 3WG real-softmax overlap with shared issue/loader producer");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse,
          "Default-off 3WG overlap with production-style retained score registers");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse_cycles",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_cycles,
          "Supporting-only tail-emitted cycle clone of retained-score 3WG overlap");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early,
          "Default-off retained-score 3WG overlap issuing q1 QK before q0 PV wait");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_p4",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_p4,
          "Default-off q1-early overlap with four query-local P payload slots");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_phalf",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_phalf,
          "Default-off q1-early overlap publishing P at native K64 halves");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_scaleahead",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_scaleahead,
          "Default-off q1-early K64 overlap with one ahead-of-pack P-scale copy");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact,
          "Default-off equal-wave q1-early route with production output rescale recurrence");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact_persistent",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_parent_exact_persistent,
          "Fixed 152-worker grid-stride form of the accepted paired parent-exact route");
    m.def("mxfp4_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_cycles",
          &dispatch_scaled_k64_two_query_real_overlap_3wg_score_reuse_q1early_cycles,
          "Supporting-only tail-emitted cycle clone of the q1-early schedule");
#endif

#if TK_FA4_MXFP4_SFID_PROBE_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_sfid_probe", &dispatch_mxfp4_sfid_probe,
          "Compile-gated independent A/B SFID, footprint, and overwrite probe");
    m.def("mxfp4_fine_sfid_probe", &dispatch_mxfp4_fine_sfid_probe,
          "Compile-gated randomized production-layout fine-scale SFID packing probe");
    m.def("mxfp4_production_sfid_probe", &dispatch_mxfp4_production_sfid_probe,
          "Compile-gated Dqk192 QK and K128 PV scale-occupancy probe");
    m.def("mxfp4_scale_page_swap_probe",
          &dispatch_mxfp4_scale_page_swap_probe,
          "Compile-gated serialized/concurrent exact scale-page lifetime probe");
#endif

#if TK_FA4_TMEM_OUTPUT_CHECKPOINT_PROBE_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_output_checkpoint_probe",
          &dispatch_mxfp4_output_checkpoint_probe,
          "Compile-gated FP32-to-BF16/FP16 TMEM checkpoint and MXFP4 continuation probe");
    m.def("mxfp4_output_checkpoint_dvo64_swap_probe",
          &dispatch_mxfp4_output_checkpoint_dvo64_swap_probe,
          "Compile-gated DVO64 active/cold BF16/FP16 TMEM swap probe");
#endif

#if TK_FA4_REGISTER_OUTPUT_PROBE_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_register_output_probe",
          &dispatch_mxfp4_register_output_probe,
          "Compile-gated rolling real-PV register-resident output probe");
#endif

#if TK_FA4_OPTION_AB_PROBE_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_option_b_nb64_dvo128",
          &dispatch_mxfp4_option_b_nb64_dvo128,
          "Compile-gated Option B M128/N64-microepoch persistent-D128 route");
    m.def("mxfp4_option_a_replay",
          &dispatch_mxfp4_option_a_replay,
          "Compile-gated Option A two-query one-hotplate QK replay route");
#endif

#if TK_FA4_CLUSTER_P_PIPELINE_PROBE_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_cluster_p_fake", &dispatch_mxfp4_cluster_p_fake,
          "Compile-gated cluster2 P DSM handoff with rank-local Nb64/D128 PV");
    m.def("mxfp4_cluster_p_local_control",
          &dispatch_mxfp4_cluster_p_local_control,
          "Compile-gated local-P arithmetic control for cluster P handoff");
#endif

#if TK_FA4_LOCAL_P_STAGE_LADDER_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_local_p_stage_a", &dispatch_mxfp4_local_p_stage_a,
          "Trace-free resident scaled-K64 PV ceiling with four-byte sink");
    m.def("mxfp4_local_p_stage_b", &dispatch_mxfp4_local_p_stage_b,
          "A plus exact 128x128 FP32 TMEM-to-global output store");
    m.def("mxfp4_local_p_stage_c", &dispatch_mxfp4_local_p_stage_c,
          "B plus per-epoch packed-P global-to-SMEM publication");
    m.def("mxfp4_local_p_stage_d", &dispatch_mxfp4_local_p_stage_d,
          "C plus per-epoch P-scale global-to-SMEM load");
    m.def("mxfp4_local_p_stage_e", &dispatch_mxfp4_local_p_stage_e,
          "D plus per-epoch P-scale SMEM-to-TMEM completion");
    m.def("mxfp4_local_p_stage_f_hot", &dispatch_mxfp4_local_p_stage_f_hot,
          "E plus explicit per-epoch hot-address V payload movement");
    m.def("mxfp4_local_p_stage_f_perturbed",
          &dispatch_mxfp4_local_p_stage_f_perturbed,
          "E plus block/epoch-unique V payload movement");
    m.def("mxfp4_local_p_stage_g_hot", &dispatch_mxfp4_local_p_stage_g_hot,
          "F-hot plus explicit per-epoch hot-address V-scale completion");
    m.def("mxfp4_local_p_stage_g_perturbed",
          &dispatch_mxfp4_local_p_stage_g_perturbed,
          "F-perturbed plus block/epoch-unique V-scale completion");
    m.def("mxfp4_local_p_stage_b_c_lifecycle_resident",
          &dispatch_mxfp4_local_p_stage_b_c_lifecycle_resident,
          "B operands plus C lifecycle/sync envelope without P bytes/publication");
    m.def("mxfp4_local_p_stage_b_c_lifecycle_publish",
          &dispatch_mxfp4_local_p_stage_b_c_lifecycle_publish,
          "B/C lifecycle control plus per-epoch proxy publication without P bytes");
    m.def("mxfp4_local_p_stage_c_serial",
          &dispatch_mxfp4_local_p_stage_c_serial,
          "C P bytes/publication with immediate per-epoch source/full drain");
    m.def("mxfp4_local_p_stage_h_plus_v_hot",
          &dispatch_mxfp4_local_p_stage_h_plus_v_hot,
          "H-semantics paired ablation with per-epoch hot V payload load");
    m.def("mxfp4_local_p_stage_h_plus_v_perturbed",
          &dispatch_mxfp4_local_p_stage_h_plus_v_perturbed,
          "H-semantics paired ablation with unique V payload load");
    m.def("mxfp4_local_p_stage_h_plus_vscale_hot",
          &dispatch_mxfp4_local_p_stage_h_plus_vscale_hot,
          "H-semantics paired ablation with per-epoch hot V-scale completion");
    m.def("mxfp4_local_p_stage_h_plus_vscale_perturbed",
          &dispatch_mxfp4_local_p_stage_h_plus_vscale_perturbed,
          "H-semantics paired ablation with unique V-scale completion");
    m.def("mxfp4_local_p_stage_f_split_single_hot",
          &dispatch_mxfp4_local_p_stage_f_split_single_hot,
          "F bytes with a single-slot WG1 V producer, hot backing");
    m.def("mxfp4_local_p_stage_f_split_single_perturbed",
          &dispatch_mxfp4_local_p_stage_f_split_single_perturbed,
          "F bytes with a single-slot WG1 V producer, unique backing");
    m.def("mxfp4_local_p_stage_f_split_pipe2_hot",
          &dispatch_mxfp4_local_p_stage_f_split_pipe2_hot,
          "F bytes with a two-slot WG1 V producer pipeline, hot backing");
    m.def("mxfp4_local_p_stage_f_split_pipe2_perturbed",
          &dispatch_mxfp4_local_p_stage_f_split_pipe2_perturbed,
          "F bytes with a two-slot WG1 V producer pipeline, unique backing");
#endif

#if TK_FA4_TMEM224_PROBE_BUILD && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("mxfp4_tmem224_d64_stripmine",
          &dispatch_mxfp4_tmem224_d64_stripmine,
          "Compile-gated paired-query D64 strip-mined full attention");
    m.def("mxfp4_tmem224_serial_qpair",
          &dispatch_mxfp4_tmem224_serial_qpair,
          "Compile-gated serialized paired-query 224-column full attention");
    m.def("mxfp4_tmem224_output_evacuation",
          &dispatch_mxfp4_tmem224_output_evacuation,
          "Compile-gated interleaved paired-query output evacuation routes");
#endif

#if !defined(TK_FA4_STAGEV_PROBE_BUILD) && !defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("forward", &dispatch<false>, "MHA forward (QK fp4, V bf16, B300, causal)");
    m.def("forward_persistent", &dispatch<true>, "MHA forward (QK fp4, V bf16, B300, causal, persistent)");
    m.def("forward_native_group2_fp4qk_bf16pv_group2_b1",
          &dispatch_native_group2_fp4qk_bf16pv_group2_b1,
          "B1 diagnostic (changed BF16-V input contract): native group2 FP4 QK, BF16 P and native group2 BF16 PV");
#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
    m.def("forward_native_group2_fp4qk_bf16pv_group2_b1_selected_trace",
          &dispatch_native_group2_fp4qk_bf16pv_group2_b1_selected_trace,
          "Bounded selected-row trace child of the exact B1 route");
#endif
    m.def("forward_native_group2_fp4qk_fp4v_dequant_bf16pv_group2_b2",
          &dispatch_native_group2_fp4qk_fp4v_dequant_bf16pv_group2_b2,
          "B2 control: production FP4 V/scales reconstructed to BF16 SMEM before native group2 BF16 PV");
#endif
#if !defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_STAGEV_PROBE_BUILD) && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("forward_fp4pv", &dispatch_fp4pv, "MHA forward experimental (QK fp4, PV fp4, B300, causal, persistent)");
    m.def("forward_fp4pv_fullgrid", &dispatch_fp4pv_fullgrid_exp, "MHA forward experimental (QK fp4, PV fp4, B300, causal, full-grid)");
    m.def("forward_streaming_live_localcta", &dispatch_forward_streaming_live_localcta, "Streaming live FP4 attention sidecar microbenchmark (QK fp4, PV fp4, CLUSTER_SIZE=2)");
#endif
#if !defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("forward_streaming_live_mxfp4", &dispatch_forward_streaming_live_mxfp4, "Streaming live FP4 attention forward with MXFP4 P/V scales");
    m.def("forward_streaming_live_mxfp4_output_only", &dispatch_forward_streaming_live_mxfp4_output_only, "Streaming live FP4 attention forward with MXFP4 P/V scales, skipping LSE store");
#if (defined(TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) && \
     TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD) || \
    (defined(TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD) && \
     TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD)
    m.def(
        "read_track_b_stage2_cap2_topology",
        &read_track_b_stage2_cap2_topology,
        "Read the last fail-closed Track-B target/control launch topology");
#endif
    m.def("reset_pairpsc_desc_diag", &reset_pairpsc_desc_diag, "Reset temporary pairpsc descriptor diagnostic buffer");
    m.def("read_pairpsc_desc_diag", &read_pairpsc_desc_diag, "Read temporary pairpsc descriptor diagnostic buffer");
    m.def("read_pairpsc_desc_diag_nonblocking", &read_pairpsc_desc_diag_nonblocking, "Read live pairpsc diagnostics on a nonblocking copy stream");
    m.def("set_pairpsc_desc_diag_slot", &set_pairpsc_desc_diag_slot, "Set one temporary pairpsc diagnostic slot");
#if TK_FA4_NATIVE_GROUP2_RAW_QK_DIAG_BUILD
    m.def("reset_native_group2_raw_qk_diag",
          &reset_native_group2_raw_qk_diag,
          "Reset the selected-row native-group2 raw QK diagnostic");
    m.def("set_native_group2_raw_qk_diag_target",
          &set_native_group2_raw_qk_diag_target,
          "Select batch/head/query row for native-group2 raw QK capture");
    m.def("read_native_group2_raw_qk_diag",
          &read_native_group2_raw_qk_diag,
          "Read selected-row native-group2 raw QK scores and coordinates");
#endif
#if TK_FA4_NATIVE_GROUP2_SELECTED_ROW_TRACE_BUILD
    m.def("reset_native_group2_selected_row_trace",
          &reset_native_group2_selected_row_trace,
          "Reset selected-row trace data and control state");
    m.def("set_native_group2_selected_row_trace_target",
          &set_native_group2_selected_row_trace_target,
          "Select batch/head/query/idx/route for bounded A-vs-B1 trace");
    m.def("read_native_group2_selected_row_trace",
          &read_native_group2_selected_row_trace,
          "Read bounded selected-row summaries and optional raw page");
    m.def("native_group2_selected_row_trace_contract",
          &native_group2_selected_row_trace_contract,
          "Return the immutable selected-row trace layout contract");
#endif
#if TK_FA4_KMC_PASSIVE_PROBE_BUILD
    m.def("reset_kmc_passive_records", &reset_kmc_passive_records,
          "Reset sparse write-only K-multicast lifecycle records");
    m.def("read_kmc_passive_records", &read_kmc_passive_records,
          "Read sparse write-only K-multicast lifecycle records");
#endif
#if TK_FA4_REGISTER_OUTPUT_PROBE_BUILD
    m.def("reset_progressive_diag_records", &reset_progressive_diag_records,
          "Reset compile-gated progressive release diagnostic records");
    m.def("read_progressive_diag_records", &read_progressive_diag_records,
          "Read compile-gated progressive release diagnostic records");
    m.def("read_progressive_diag_records_nonblocking",
          &read_progressive_diag_records_nonblocking,
          "Read live progressive diagnostics on a nonblocking stream");
#endif
#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
    m.def("mxfp4_nb128_dvo128_qstage2", &dispatch_mxfp4_nb128_dvo128_qstage2, "Compile-gated 512-thread qstage2 ownership/resource checkpoint");
    m.def("mxfp4_nb128_dvo128_qstage2_pipeline", &dispatch_mxfp4_nb128_dvo128_qstage2_pipeline, "Default-off two-stage qstage2 pipeline experiment");
    m.def("mxfp4_nb128_dvo128_qstage2_two_n64", &dispatch_mxfp4_nb128_dvo128_qstage2_two_n64, "Isolated M256 two-N64-score qstage2 experiment");
#endif
#if defined(TK_FA4_Q4_LARGE_PROBE_BUILD) || \
    defined(TK_FA4_FULLK_WAVE_PROBE_BUILD) || \
    TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
    m.def("reset_q4_large_diag_records", &reset_q4_large_diag_records, "Reset compile-gated quartered large-H/S CTA records");
    m.def("read_q4_large_diag_records", &read_q4_large_diag_records, "Read compile-gated quartered large-H/S CTA records");
#endif
    m.def("reset_mxfp4_forward_pchain_stamps", &reset_mxfp4_forward_pchain_stamps, "Reset compile-gated Stage2 P-chain clock stamps");
    m.def("read_mxfp4_forward_pchain_stamps", &read_mxfp4_forward_pchain_stamps, "Read compile-gated Stage2 P-chain clock stamps");
    m.def("reset_mxfp4_forward_timeline", &reset_mxfp4_forward_timeline, "Reset compile-gated MXFP4 forward phase timeline buffer");
    m.def("read_mxfp4_forward_timeline", &read_mxfp4_forward_timeline, "Read compile-gated MXFP4 forward phase timeline buffer");
#if TK_FA4_REAL_FWD_TIMELINE_PROBE_BUILD
    m.def("reset_mxfp4_real_forward_timeline",
          &reset_mxfp4_real_forward_timeline,
          "Reset the bounded trace-only real-forward timeline for one block");
    m.def("read_mxfp4_real_forward_timeline",
          &read_mxfp4_real_forward_timeline,
          "Read fixed-index real-forward per-tile timeline records");
    m.def("reset_mxfp4_real_forward_readiness",
          &reset_mxfp4_real_forward_readiness,
          "Reset tail-only real-forward readiness masks for one block");
    m.def("read_mxfp4_real_forward_readiness",
          &read_mxfp4_real_forward_readiness,
          "Read tail-only real-forward readiness masks");
    m.def("reset_mxfp4_real_forward_corr_pv_cycles",
          &reset_mxfp4_real_forward_corr_pv_cycles,
          "Reset sparse correction/PV cycle records for one physical block and logical task");
    m.def("read_mxfp4_real_forward_corr_pv_cycles",
          &read_mxfp4_real_forward_corr_pv_cycles,
          "Read sparse correction/PV cycle-only records");
#endif
    m.def("reset_mxfp4_forward_policy126_counters", &reset_mxfp4_forward_policy126_counters, "Reset compile-gated Policy126 forward counter buffer");
    m.def("read_mxfp4_forward_policy126_counters", &read_mxfp4_forward_policy126_counters, "Read compile-gated Policy126 forward counter buffer");
    m.def("mxfp4_control_plane_diag", &dispatch_mxfp4_control_plane_diag, "Debug isolated scheduler/role control-plane handoff; modes 0/1 one-role publish/timeout, 2 eight-role task, 3 eight-role drain, 4 missing-done timeout, 5 done-only without ack, 6 partial done, 7 ack-only without done, 8 early sentinel before ready, 9 missing role ready timeout, 10 missing role drain timeout, 11 late role ready after scheduler timeout, 12 late role ready during drain after scheduler timeout, 13 drain ack-only without done, 14 drain partial-domain completion, 15 early normal task before ready, 16 two-task reuse/reinit, 17 two-task post-reuse missing done, 18 two-task post-reuse drain sentinel");
    m.def("mxfp4_v_tma_tile_check_debug", &dispatch_mxfp4_v_tma_tile_check_debug, "Debug one MXFP4 V tile loaded manually and by TMA into the same swizzled shared layout; mode 0 compare, 1 manual, 2 TMA, 3 TMA wait");
    m.def("mxfp4_v_tma_multicast_debug", &dispatch_mxfp4_v_tma_multicast_debug, "Debug one MXFP4 V tile loaded by TMA multicast across a cluster-2 CTA group; mode 0 TK-style multicast, 1 owner mbarrier, 2 per-rank unicast, 3 multicast plus rank-local TCGEN consume, 4 unicast plus rank-local TCGEN consume, 6 unicast plus TMEM allocate cut, 7 unicast plus TCGEN issue cut");
    m.def("mxfp4_v_tma_multicast_vscale_debug", &dispatch_mxfp4_v_tma_multicast_vscale_debug, "Debug one MXFP4 V tile plus production MXFP4 V-scale tile loaded by cluster-2 TMA multicast; mode 0 TK-style multicast, 1 owner mbarrier, 2 per-rank unicast, 3 multicast plus rank-local TCGEN consume, 4 unicast plus rank-local TCGEN consume, 6 unicast plus TMEM allocate cut, 7 unicast plus TCGEN issue cut");
    m.def("mxfp4_k_tma_multicast_debug", &dispatch_mxfp4_k_tma_multicast_debug, "Debug one MXFP4 K tile plus optional K-scale tile loaded by cluster-2 TMA multicast; mode 0 TK-style multicast, 1 owner mbarrier, 2 per-rank unicast");
    m.def("mxfp4_kv_tma_multicast_debug", &dispatch_mxfp4_kv_tma_multicast_debug, "Debug one MXFP4 K/K-scale and V/V-scale tile pair loaded by cluster-2 TMA multicast; mode 0 expect then multicast, 1 load before expect, 2 per-rank unicast, 3 serial K then V multicast");
    m.def("mxfp4_qk_tma_multicast_debug", &dispatch_mxfp4_qk_tma_multicast_debug, "Debug MXFP4 QK with K/K-scale TMA movement; mode 0 per-rank unicast QK, 1 rank0 multicast shared-slot QK, 2 rank0 multicast shared-slot QK ring, 3 per-rank unicast QK ring, 4 serial per-rank unicast QK ring");
    m.def("mxfp4_v_tma_tile_check_fullg_debug", &dispatch_mxfp4_v_tma_tile_check_fullg_debug, "Debug one MXFP4 V tile using the production qkscfix/vtma globals; mode 0 compare, 1 manual, 2 TMA, 3 TMA wait");
    m.def("mxfp4_pv_from_p_chunked_debug", &dispatch_mxfp4_pv_from_p_chunked_debug, "Debug MXFP4 PV from compact HBM P/V, accumulating K chunks in one launch");
#if TK_FA4_DVO64_RESIDENCY_PROBE_BUILD
    m.def("mxfp4_pv_dvo64_residency_shell", &dispatch_mxfp4_pv_dvo64_residency_shell, "Two-WG DVO64 MXFP4 P/V scale-copy, PV, TMEM-load residency shell");
    m.def("mxfp4_nb64_dvo128_residency_shell", &dispatch_mxfp4_nb64_dvo128_residency_shell, "Two-WG one-walk Nb64 QK plus DVO128 PV residency shell");
    m.def("mxfp4_nb64_dual_dvo64_residency_shell", &dispatch_mxfp4_nb64_dual_dvo64_residency_shell, "Two-WG Nb64 one-P dual-DVO64 sequential-V residency shell");
    m.def("mxfp4_nb128_dual_dvo64_attention_shell", &dispatch_mxfp4_nb128_dual_dvo64_attention_shell, "Two-WG one-iteration Nb128 QK/softmax/P plus sequential dual-DVO64 attention shell");
    m.def("mxfp4_nb128_dual_dvo64_multistep_shell", &dispatch_mxfp4_nb128_dual_dvo64_multistep_shell, "Two-WG multistep Nb128 QK/softmax/P plus shared-backed dual-DVO64 attention shell");
#endif
    m.def("mxfp4_pv_from_p_k256_debug", &dispatch_mxfp4_pv_from_p_k256_debug, "Debug MXFP4 PV from compact HBM P/V, consuming two K128 chunks per iteration");
    m.def("mxfp4_tcgen_shape_debug", &dispatch_mxfp4_tcgen_shape_debug, "Standalone MXFP4 TCGEN shape diagnostic with real FP4 payloads and E8M0 scales");
    m.def("mxfp4_joint_qk_pv_q4_stage0_perf", &dispatch_mxfp4_joint_qk_pv_q4_stage0_perf, "Standalone MXFP4 joint-quarter q4 opt2 stage0 perf kernel");
    m.def("mxfp4_joint_qk_pv_q4_stage0_scalepp_perf", &dispatch_mxfp4_joint_qk_pv_q4_stage0_scalepp_perf, "Standalone MXFP4 joint-quarter q4 opt2 stage0 scale ping-pong perf kernel");
    m.def("mxfp4_joint_qk_pv_half_stage0_perf", &dispatch_mxfp4_joint_qk_pv_half_stage0_perf, "Standalone MXFP4 joint half-tile opt2 stage0 perf kernel");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_ref", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_ref, "Standalone MXFP4 true-P/V joint-quarter q4 reference kernel");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_vprestage_perf, "Standalone MXFP4 true q4 QK/PV with V and V-scale staging under QK flight");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_pingpong_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_pingpong_perf, "Standalone MXFP4 true q4 QK/PV with score and P/V source ping-pong");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_score_tmem_only_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_score_tmem_only_perf, "Standalone MXFP4 true q4 low-resource overlap with score TMEM ping-pong only");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_scale_tmem_only_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_scale_tmem_only_perf, "Standalone MXFP4 true q4 low-resource overlap with P/V scale TMEM ping-pong only");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_vonly_src_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_vonly_src_perf, "Standalone MXFP4 true q4 low-resource overlap with V source duplication only");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_ponly_src_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_ponly_src_perf, "Standalone MXFP4 true q4 low-resource overlap with P source duplication only");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_score_scale_tmem_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_score_scale_tmem_perf, "Standalone MXFP4 true q4 low-resource overlap with score and scale TMEM ping-pong");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_score_p_src_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_score_p_src_perf, "Standalone MXFP4 true q4 overlap with independent score and P source slots");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_pv_split_k64_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_pv_split_k64_perf, "Standalone MXFP4 true q4 PV issued as two legal K64 chunks");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_split_output_accum_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_split_output_accum_perf, "Standalone MXFP4 true q4 alternating two-output accumulator experiment");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_barrier_trim_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_barrier_trim_perf, "Standalone MXFP4 true q4 vprestage with issuer-local PV sync trimming");
    m.def("mxfp4_joint_qk_pv_true_q4_stage0_fused_p_handoff_perf", &dispatch_mxfp4_joint_qk_pv_true_q4_stage0_fused_p_handoff_perf, "Standalone MXFP4 true q4 vprestage with warpgroup P handoff semaphore");
    m.def("mxfp4_joint_qk_pv_true_half_stage0_perf", &dispatch_mxfp4_joint_qk_pv_true_half_stage0_perf, "Standalone MXFP4 true half QK plus true half PV stage0 perf kernel");
    m.def("mxfp4_joint_qk_pv_true_half_stage0_srcwait_pipeline_perf", &dispatch_mxfp4_joint_qk_pv_true_half_stage0_srcwait_pipeline_perf, "Standalone MXFP4 true half source-wait pipeline experiment");
    m.def("mxfp4_joint_qk_pv_true_half_stage0_barrier_trim_perf", &dispatch_mxfp4_joint_qk_pv_true_half_stage0_barrier_trim_perf, "Standalone MXFP4 true half D with issuer-local PV sync trimming");
    m.def("mxfp4_joint_qk_pv_true_half_stage0_fused_p_handoff_perf", &dispatch_mxfp4_joint_qk_pv_true_half_stage0_fused_p_handoff_perf, "Standalone MXFP4 true half D with warpgroup P handoff semaphore");
    m.def("mxfp4_joint_qk_pv_quarter_debug", &dispatch_mxfp4_joint_qk_pv_quarter_debug, "Standalone MXFP4 QK then placeholder-P PV quarter diagnostic with score/output TMEM reuse");
    m.def("mxfp4_joint_qk_pv_quarter_handoff_debug", &dispatch_mxfp4_joint_qk_pv_quarter_handoff_debug, "Standalone MXFP4 joint-quarter scale/payload producer-consumer handoff diagnostic");
#endif

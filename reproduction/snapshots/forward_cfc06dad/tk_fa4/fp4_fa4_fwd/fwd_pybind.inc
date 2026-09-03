// Forward pybind registrations for bf16_b300_mha_causal_fp4_pv_experiments.cu.
// Included from the module wrapper in bf16_b300_mha_causal_fp4_pv_experiments_pybind.inc.

#if !defined(TK_FA4_STAGEV_PROBE_BUILD) && !defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("forward", &dispatch<false>, "MHA forward (QK fp4, V bf16, B300, causal)");
    m.def("forward_persistent", &dispatch<true>, "MHA forward (QK fp4, V bf16, B300, causal, persistent)");
#endif
#if !defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_STAGEV_PROBE_BUILD) && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("forward_fp4pv", &dispatch_fp4pv, "MHA forward experimental (QK fp4, PV fp4, B300, causal, persistent)");
    m.def("forward_fp4pv_fullgrid", &dispatch_fp4pv_fullgrid_exp, "MHA forward experimental (QK fp4, PV fp4, B300, causal, full-grid)");
    m.def("forward_streaming_live_localcta", &dispatch_forward_streaming_live_localcta, "Streaming live FP4 attention sidecar microbenchmark (QK fp4, PV fp4, CLUSTER_SIZE=2)");
#endif
#if !defined(TK_FA4_DQ_ONLY_BUILD) && !defined(TK_FA4_BACKWARD_ONLY_BUILD)
    m.def("forward_streaming_live_mxfp4", &dispatch_forward_streaming_live_mxfp4, "Streaming live FP4 attention forward with MXFP4 P/V scales");
    m.def("forward_streaming_live_mxfp4_output_only", &dispatch_forward_streaming_live_mxfp4_output_only, "Streaming live FP4 attention forward with MXFP4 P/V scales, skipping LSE store");
    m.def("reset_pairpsc_desc_diag", &reset_pairpsc_desc_diag, "Reset temporary pairpsc descriptor diagnostic buffer");
    m.def("read_pairpsc_desc_diag", &read_pairpsc_desc_diag, "Read temporary pairpsc descriptor diagnostic buffer");
    m.def("set_pairpsc_desc_diag_slot", &set_pairpsc_desc_diag_slot, "Set one temporary pairpsc diagnostic slot");
    m.def("reset_mxfp4_forward_pchain_stamps", &reset_mxfp4_forward_pchain_stamps, "Reset compile-gated Stage2 P-chain clock stamps");
    m.def("read_mxfp4_forward_pchain_stamps", &read_mxfp4_forward_pchain_stamps, "Read compile-gated Stage2 P-chain clock stamps");
    m.def("reset_mxfp4_forward_timeline", &reset_mxfp4_forward_timeline, "Reset compile-gated MXFP4 forward phase timeline buffer");
    m.def("read_mxfp4_forward_timeline", &read_mxfp4_forward_timeline, "Read compile-gated MXFP4 forward phase timeline buffer");
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

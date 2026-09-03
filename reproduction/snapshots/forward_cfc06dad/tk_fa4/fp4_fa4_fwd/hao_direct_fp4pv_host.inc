#pragma once

struct hao_direct_topology_state {
    long long valid = 0;
    long long physical_grid = 0;
    long long logical_jobs = 0;
    long long threads = 0;
    long long kv_stages = 0;
    long long tmem_columns = 0;
    long long dynamic_smem = 0;
};

static std::mutex hao_direct_topology_mutex;
static hao_direct_topology_state hao_direct_last_topology;

pybind11::dict read_hao_direct_topology() {
    std::lock_guard<std::mutex> lock(
        hao_direct_topology_mutex);
    const auto &s = hao_direct_last_topology;
    pybind11::dict out;
    out["schema"] = "tk_hao_direct_pipeline_v1";
    out["valid"] = s.valid;
    out["route"] = TK_HAO_DIRECT_ROUTE_NAME;
    out["batch"] = hao_direct_config::BATCH;
    out["heads"] = hao_direct_config::HEADS;
    out["seqlen"] = hao_direct_config::N_TILES * 128;
    out["dqk"] = hao_direct_config::Dqk;
    out["dvo"] = hao_direct_config::Dvo;
    out["logical_jobs"] = s.logical_jobs;
    out["physical_grid_ctas"] = s.physical_grid;
    out["threads_per_cta"] = s.threads;
    out["query_stages"] = hao_direct_config::QUERY_STAGES;
    out["kv_stages"] = s.kv_stages;
    out["tmem_columns"] = s.tmem_columns;
    out["dynamic_smem_bytes"] = s.dynamic_smem;
    out["persistent"] = true;
    out["p_first_chunks"] =
        hao_direct_config::P_FIRST_CHUNKS;
    out["early_p_k64"] =
        hao_direct_config::HAO_DIRECT_EARLY_P &&
        !hao_direct_config::HAO_DIRECT_SPARSE_PV;
    out["sparse_pv"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV;
    out["sparse_pattern"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV
            ? "paired_4of8"
            : "dense";
    out["sparse_selector_mode"] =
        hao_direct_config::HAO_DIRECT_SPARSE_SELECTOR_MODE;
    out["sparse_fixed_selector"] =
        hao_direct_config::HAO_DIRECT_SPARSE_SELECTOR_MODE;
    out["sparse_split_k128"] =
        hao_direct_config::HAO_DIRECT_SPARSE_SPLIT_K128;
    out["sparse_issue_count"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV
            ? (hao_direct_config::HAO_DIRECT_SPARSE_SPLIT_K128 ? 2 : 1)
            : 0;
    out["sparse_logical_k_per_issue"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV ? 128 : 0;
    out["sparse_early_half"] =
        hao_direct_config::HAO_DIRECT_SPARSE_SPLIT_K128;
    out["mx_disjoint_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_MX_EARLY_P;
    out["mx_direct_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_MX_EARLY_DIRECT_SCALE;
    out["mx_register_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF;
    out["mx_pair_scale_scratch"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_SCRATCH;
    out["mx_async_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_MX_EARLY_ASYNC_SCALE;
    out["mx_shiftless_corr_bypass"] =
        hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS;
    out["mx_defer_job_stats_wait"] =
        hao_direct_config::HAO_DIRECT_MX_DEFER_JOB_STATS_WAIT;
    out["mx_direct_scale_scratch"] =
        hao_direct_config::HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH;
    out["mx_split_stage_pv"] =
        hao_direct_config::HAO_DIRECT_MX_SPLIT_STAGE_PV;
    out["mx_qk_scale_preload"] =
        hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD;
    out["mx_qk_scale_preload_mask"] =
        hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_MASK;
    out["mx_qk_scale_preload_before_p"] =
        hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_BEFORE_P;
    out["nv_qk_folded_k64_scales"] =
        hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES;
    out["nv_qk_folded_k64_scale_mask"] =
        hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK;
    out["nv_qk_compact_folded_scales"] =
        hao_direct_config::HAO_DIRECT_NV_QK_COMPACT_FOLDED_SCALES;
    out["nv_qk_preload_page_mask"] =
        hao_direct_config::HAO_DIRECT_NV_QK_PRELOAD_PAGE_MASK;
    out["mx_q3_corr_wg"] =
        hao_direct_config::HAO_DIRECT_MX_Q3_CORR_WG;
    out["mx_dual_q3_corr_wg"] =
        hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_CORR_WG;
    out["mx_dual_q3_smem_wg"] =
        hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG;
    out["mx_dual_q3_tmem_wg"] =
        hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG;
    out["fused_pv_issue"] =
        hao_direct_config::HAO_DIRECT_FUSED_PV_ISSUE;
    out["log2_p_quant"] =
        hao_direct_config::HAO_DIRECT_LOG2_QUANT;
    out["mx_log2_p_quant"] =
        hao_direct_config::HAO_DIRECT_MX_LOG2_QUANT;
    out["mx_scale_select"] =
        hao_direct_config::HAO_DIRECT_MX_SCALE_SELECT;
    out["mx_direct_log_code"] =
        hao_direct_config::HAO_DIRECT_MX_DIRECT_LOG_CODE;
    out["mx_direct_log_code_mode"] =
        hao_direct_config::HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE;
    out["mx_log_lut_bits"] =
        hao_direct_config::HAO_DIRECT_MX_LOG_LUT_BITS;
    out["mx_log_lut_bit_code_mode"] =
        hao_direct_config::HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE;
    out["mx_quantized_denom"] =
        hao_direct_config::HAO_DIRECT_MX_QUANTIZED_DENOM;
    out["mx_denom_corr_wg"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_CORR_WG;
    out["mx_denom_load_width"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_LOAD_WIDTH;
    out["mx_denom_pipelined_sum"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_PIPELINED_SUM;
    out["mx_denom_pair_overlap"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_PAIR_OVERLAP;
    out["mx_denom_factor_hoist"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_FACTOR_HOIST;
    out["mx_denom_int_scale_fusion"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_INT_SCALE_FUSION;
    out["mx_denom_sum_lanes"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SUM_LANES;
    out["mx_denom_words"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_WORDS;
    out["mx_denom_decode_mode"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_DECODE_MODE;
    out["mx_denom_scale_smem"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SCALE_SMEM;
    out["mx_denom_payload_smem"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM;
    out["mx_denom_magic_i2f"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_MAGIC_I2F;
    out["mx_denom_early_smem"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_EARLY_SMEM;
    out["mx_denom_early_split"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_EARLY_SPLIT;
    out["mx_denom_split_prefetch"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SPLIT_PREFETCH;
    out["mx_global_anchor32"] =
        hao_direct_config::HAO_DIRECT_MX_GLOBAL_ANCHOR32;
    out["mx_global_anchor128"] =
        hao_direct_config::HAO_DIRECT_MX_GLOBAL_ANCHOR128;
    out["mx_global_anchor_bias"] =
        hao_direct_config::HAO_DIRECT_MX_GLOBAL_ANCHOR_BIAS;
    out["mx_global_anchor_margin_log2"] =
        TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2;
    out["mx_anchor_affine_hoist"] =
        hao_direct_config::HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST;
    out["mx_fixed_anchor_log2"] =
        hao_direct_config::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2;
    out["mx_stored_scale_shift_log2"] =
        hao_direct_config::HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2;
    out["debug_lse_stat"] =
        hao_direct_config::HAO_DIRECT_DEBUG_LSE_STAT;
    out["mx_denom_sample_stride"] =
        hao_direct_config::HAO_DIRECT_MX_DENOM_SAMPLE_STRIDE;
    out["mx_max_sample_stride"] =
        hao_direct_config::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE;
    out["mx_max_sample_bias_x2"] =
        hao_direct_config::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2;
    out["mx_q1_max_sample_stride"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_MAX_SAMPLE_STRIDE;
    out["mx_pair_scale_reuse"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_REUSE;
    out["mx_pair_scale_stage_mask"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_STAGE_MASK;
    out["mx_pair_scale_period"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_PERIOD;
    out["mx_pair_scale_delta"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_SCALE_DELTA;
    out["mx_q1_pair_probe"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_PAIR_PROBE;
    out["mx_q1_pair_probe_bias"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_PAIR_PROBE_BIAS;
    out["mx_lagged_q1_scale"] =
        hao_direct_config::HAO_DIRECT_MX_LAGGED_Q1_SCALE;
    out["mx_q1_scale_offload"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_SCALE_OFFLOAD;
    out["mx_q1_self_max"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_SELF_MAX;
    out["mx_scan_fused_p"] =
        hao_direct_config::HAO_DIRECT_MX_SCAN_FUSED_P;
    out["mx_local_native_deferred_scale"] =
        hao_direct_config::HAO_DIRECT_MX_LOCAL_NATIVE_DEFERRED_SCALE;
    out["mx_bit_exp2_mask"] =
        hao_direct_config::HAO_DIRECT_MX_BIT_EXP2_MASK;
    out["mx_pwl_exp2"] =
        hao_direct_config::HAO_DIRECT_MX_PWL_EXP2;
    out["mx_pwl_exp2_mode"] =
        hao_direct_config::HAO_DIRECT_MX_PWL_EXP2_MODE;
    out["mx_mode23_native_density"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_NATIVE_DENSITY;
    out["mx_mode23_native_quarter_mask"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_NATIVE_QUARTER_MASK;
    out["mx_mode23_early_native"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE;
    out["mx_mode23_early_native_stage_mask"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_STAGE_MASK;
    out["mx_mode23_early_native_quarter_mask"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_QUARTER_MASK;
    out["mx_mode23_early_native_lookahead"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_LOOKAHEAD;
    out["mx_mode23_early_native_order"] =
        hao_direct_config::HAO_DIRECT_MX_MODE23_EARLY_NATIVE_ORDER;
    out["mx_pwl_four_sample_denom"] =
        hao_direct_config::HAO_DIRECT_MX_PWL_FOUR_SAMPLE_DENOM;
    out["mx_full_approx_denom"] =
        hao_direct_config::HAO_DIRECT_MX_FULL_APPROX_DENOM;
    out["mx_full_approx_denom_mode"] =
        hao_direct_config::HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE;
    out["mx_affine_a"] =
        hao_direct_config::HAO_DIRECT_MX_AFFINE_A;
    out["mx_affine_b"] =
        hao_direct_config::HAO_DIRECT_MX_AFFINE_B;
    out["mx_stage0_affine_mask"] =
        hao_direct_config::HAO_DIRECT_MX_STAGE0_AFFINE_MASK;
    out["mx_stage1_affine_mask"] =
        hao_direct_config::HAO_DIRECT_MX_STAGE1_AFFINE_MASK;
    out["mx_shiftless_softmax"] =
        hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX;
    out["nv_shiftless_softmax"] =
        hao_direct_config::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX;
    out["nv_shiftless_scale_lift"] =
        hao_direct_config::HAO_DIRECT_NV_SHIFTLESS_SCALE_LIFT;
    out["nv_scale_satfinite"] =
        hao_direct_config::HAO_DIRECT_NV_SCALE_SATFINITE;
    out["nv_v_global_decode"] =
        hao_direct_config::HAO_DIRECT_NV_V_GLOBAL_DECODE;
    out["nv_p_global_log2"] =
        hao_direct_config::HAO_DIRECT_NV_P_GLOBAL_LOG2;
    out["nv_policy_id"] =
        hao_direct_config::HAO_DIRECT_NV_POLICY_ID;
    out["nv_pwl_exp2_mode"] =
        hao_direct_config::HAO_DIRECT_NV_PWL_EXP2_MODE;
    out["nv_stage0_affine_mask"] =
        hao_direct_config::HAO_DIRECT_NV_STAGE0_AFFINE_MASK;
    out["nv_stage1_affine_mask"] =
        hao_direct_config::HAO_DIRECT_NV_STAGE1_AFFINE_MASK;
    out["nv_fast_approx_mode"] =
        hao_direct_config::HAO_DIRECT_NV_FAST_APPROX_MODE;
    out["nv_affine_a"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_AFFINE_A);
    out["nv_affine_b"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_AFFINE_B);
    out["nv_cubic_a"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_CUBIC_A);
    out["nv_cubic_b"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_CUBIC_B);
    out["nv_cubic_c"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_CUBIC_C);
    out["nv_cubic_d"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_CUBIC_D);
    out["nv_refit_cubic_a"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_A);
    out["nv_refit_cubic_b"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_B);
    out["nv_refit_cubic_c"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_C);
    out["nv_refit_cubic_d"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_D);
    out["nv_hinge_m"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_HINGE_M);
    out["nv_hinge_b"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_HINGE_B);
    out["nv_hinge_d"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_HINGE_D);
    out["nv_hinge_t"] =
        static_cast<float>(TK_HAO_DIRECT_FP4PV_NV_HINGE_T);
    out["nv_native_pairs"] =
        hao_direct_config::HAO_DIRECT_NV_NATIVE_PAIRS;
    out["nv_native6_quarter_mask"] =
        hao_direct_config::HAO_DIRECT_NV_NATIVE6_QUARTER_MASK;
    out["nv_relu_pack"] =
        hao_direct_config::HAO_DIRECT_NV_RELU_PACK;
    out["nv_quantized_denom"] =
        hao_direct_config::HAO_DIRECT_NV_QUANTIZED_DENOM;
    out["nv_hybrid_denom"] =
        hao_direct_config::HAO_DIRECT_NV_HYBRID_DENOM;
    out["nv_sampled_denom"] =
        hao_direct_config::HAO_DIRECT_NV_SAMPLED_DENOM;
    out["mx_sampled_stabilizer"] =
        hao_direct_config::HAO_DIRECT_MX_SAMPLED_STABILIZER;
    out["mx_retain_q0"] =
        hao_direct_config::HAO_DIRECT_MX_RETAIN_Q0;
    out["mx_pair_load_scan"] =
        hao_direct_config::HAO_DIRECT_MX_PAIR_LOAD_SCAN;
    out["mx_half_prefetch_q0"] =
        hao_direct_config::HAO_DIRECT_MX_HALF_PREFETCH_Q0;
    out["mx_half_prefetch_q1"] =
        hao_direct_config::HAO_DIRECT_MX_HALF_PREFETCH_Q1;
    out["mx_half_prefetch_q3"] =
        hao_direct_config::HAO_DIRECT_MX_HALF_PREFETCH_Q3;
    out["mx_delayed_half_q2"] =
        hao_direct_config::HAO_DIRECT_MX_DELAYED_HALF_Q2;
    out["mx_delayed_early_q3"] =
        hao_direct_config::HAO_DIRECT_MX_DELAYED_EARLY_Q3;
    out["mx_early_q2_reduce"] =
        hao_direct_config::HAO_DIRECT_MX_EARLY_Q2_REDUCE;
    out["mx_stage1_early_q1"] =
        hao_direct_config::HAO_DIRECT_MX_STAGE1_EARLY_Q1;
    out["mx_q1_max_interleave"] =
        hao_direct_config::HAO_DIRECT_MX_Q1_MAX_INTERLEAVE;
    out["mx_tree_max"] =
        hao_direct_config::HAO_DIRECT_MX_TREE_MAX;
    out["mx_max3_reduce"] =
        hao_direct_config::HAO_DIRECT_MX_MAX3_REDUCE;
    out["mx_max3_wide_reduce"] =
        hao_direct_config::HAO_DIRECT_MX_MAX3_WIDE_REDUCE;
    out["nv_quarter_scale"] =
        hao_direct_config::HAO_DIRECT_NV_QUARTER_SCALE;
    out["nv_scale_encode_mode"] =
        hao_direct_config::HAO_DIRECT_NV_SCALE_ENCODE_MODE;
    out["nv_quarter_schedule_mode"] =
        hao_direct_config::HAO_DIRECT_NV_QUARTER_SCHEDULE_MODE;
    out["nv_denom_source"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_SOURCE;
    out["nv_denom_corr_wg"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_CORR_WG;
    out["nv_denom_smem_handoff"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_SMEM_HANDOFF;
    out["nv_denom_words"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_WORDS;
    out["nv_denom_known_max"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_KNOWN_MAX;
    out["nv_denom_short_exact"] =
        hao_direct_config::HAO_DIRECT_NV_DENOM_SHORT_EXACT;
    out["nv_denom_exact_max_tiles"] =
        TK_HAO_DIRECT_FP4PV_NV_DENOM_EXACT_MAX_TILES;
    out["nv_global_anchor64"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR64;
    out["nv_global_anchor32"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR32;
    out["nv_global_anchor_samples"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR64
            ? 64
            : (hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR32
                   ? 32
                   : 0);
    out["nv_global_anchor_scale"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR_SCALE;
    out["nv_global_anchor_bias"] =
        hao_direct_config::HAO_DIRECT_NV_GLOBAL_ANCHOR_BIAS;
    out["nv_debug_lse_rowmax"] =
        hao_direct_config::HAO_DIRECT_NV_DEBUG_LSE_ROWMAX;
    out["nv_tail_q3_hook_word"] =
        hao_direct_config::HAO_DIRECT_NV_TAIL_Q3_HOOK_WORD;
    out["nv_q01_corr_wg"] =
        hao_direct_config::HAO_DIRECT_NV_Q01_CORR_WG;
    out["nv_tree_max"] =
        hao_direct_config::HAO_DIRECT_NV_TREE_MAX;
    out["nv_max3_reduce"] =
        hao_direct_config::HAO_DIRECT_NV_MAX3_REDUCE;
    out["nv_max_sample_stride"] =
        hao_direct_config::HAO_DIRECT_NVFP4_PV
            ? hao_direct_config::HAO_DIRECT_MX_MAX_SAMPLE_STRIDE
            : 1;
    out["nv_max_sample_bias_x2"] =
        hao_direct_config::HAO_DIRECT_NVFP4_PV
            ? hao_direct_config::HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2
            : 0;
    out["nv_q0_stabilizer"] =
        hao_direct_config::HAO_DIRECT_NV_Q0_STABILIZER;
    out["nv_preview_stabilizer"] =
        hao_direct_config::HAO_DIRECT_NV_PREVIEW_STABILIZER;
    out["nv_preview_width"] =
        hao_direct_config::HAO_DIRECT_NV_PREVIEW_WIDTH;
    out["nv_preview_bias_x8"] =
        hao_direct_config::HAO_DIRECT_NV_PREVIEW_BIAS_X8;
    out["nv_full_row_stabilizer"] =
        hao_direct_config::HAO_DIRECT_NV_FULL_ROW_STABILIZER;
    out["nv_direct_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_NV_EARLY_DIRECT_SCALE;
    out["nv_preload_v_tail_mode"] =
        hao_direct_config::HAO_DIRECT_NV_PRELOAD_V_TAIL_MODE;
    out["nv_issuer_v_scale_prefetch_mask"] =
        hao_direct_config::
            HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK;
    out["nv_register_scale_handoff"] =
        hao_direct_config::HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF;
    out["nv_reuse_v_scale"] =
        hao_direct_config::HAO_DIRECT_NV_REUSE_V_SCALE;
    out["nv_softmax_v_scale_prefetch"] =
        hao_direct_config::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH;
    out["p_ilp_pipeline"] =
        hao_direct_config::HAO_DIRECT_P_ILP_PIPELINE;
    out["fixed_p_ceiling"] =
        hao_direct_config::HAO_DIRECT_FIXED_P_CEILING;
    out["score_pack_ceiling"] =
        hao_direct_config::HAO_DIRECT_SCORE_PACK_CEILING;
    out["rowmax_pack_ceiling"] =
        hao_direct_config::HAO_DIRECT_ROWMAX_PACK_CEILING;
    out["fake_denom_ceiling"] =
        hao_direct_config::HAO_DIRECT_FAKE_DENOM_CEILING;
    out["ex2_emu_mask"] =
        hao_direct_config::ONLINE_STAGE2_EX2_EMU_MASK;
    out["ex2_alu_degree"] =
        hao_direct_config::ONLINE_STAGE2_EX2_ALU_DEGREE;
    out["mx_ex2_q1_mask"] =
        hao_direct_config::HAO_DIRECT_MX_EX2_Q1_MASK;
    out["mx_ex2_q2_mask"] =
        hao_direct_config::HAO_DIRECT_MX_EX2_Q2_MASK;
    out["mx_ex2_q3_mask"] =
        hao_direct_config::HAO_DIRECT_MX_EX2_Q3_MASK;
    out["task_order"] = hao_direct_config::TASK_ORDER;
    out["wait_hint_cycles"] = TK_HAO_DIRECT_WAIT_HINT;
    out["qk_format"] =
        hao_direct_config::HAO_DIRECT_NVFP4_QK
            ? "nvfp4_e4m3_block16"
            : "mxfp4_e8m0_block32";
    out["pv_format"] =
        hao_direct_config::HAO_DIRECT_SPARSE_PV
            ? "sparse_nvfp4_e4m3_block32"
            : (hao_direct_config::HAO_DIRECT_NVFP4_PV
                   ? "nvfp4_e4m3_block16"
                   : "mxfp4_e8m0_block32");
    return out;
}

void dispatch_hao_direct_fp4pv_impl(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    float V_sg,
    at::Tensor O,
    at::Tensor LSE,
    int p_quant_mode,
    bool persistent_launch,
    bool store_lse) {
    using C = hao_direct_config;
    using G = globals_fp4pv_hao_direct<C>;
    using Layout = hao_direct_pipeline_smem_layout<C>;
    const char *route =
        std::getenv("TK_FA4_FP4PV_FWD_CONFIG");
    TORCH_CHECK(
        route != nullptr &&
            std::strcmp(route, TK_HAO_DIRECT_ROUTE_NAME) == 0,
        "TK HAO direct candidate route mismatch");
    TORCH_CHECK(
        persistent_launch,
        "HAO direct port requires persistent_launch=True");
    TORCH_CHECK(
        p_quant_mode == 0,
        "HAO direct full-FP4 port has one fixed P quantizer");
    TORCH_CHECK(
        !C::HAO_DIRECT_NV_V_GLOBAL_DECODE ||
            (std::isfinite(V_sg) && V_sg > 0.0f),
        "V_sg must be finite and positive");
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
    TORCH_CHECK(
        Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            V_sc.dtype() == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(
        O.dtype() == at::ScalarType::BFloat16 &&
            LSE.dtype() == at::ScalarType::Float);
    TORCH_CHECK(
        Q.dim() == 4 &&
            Q.size(0) == C::BATCH &&
            Q.size(1) == C::HEADS &&
            Q.size(2) == C::N_TILES * 128 &&
            Q.size(3) == 64);
    TORCH_CHECK(K.sizes() == Q.sizes());
    TORCH_CHECK(
        V_fp4.dim() == 4 &&
            V_fp4.size(0) == C::BATCH &&
            V_fp4.size(1) == C::HEADS &&
            V_fp4.size(2) == 128 &&
            V_fp4.size(3) == C::N_TILES * 64);
    TORCH_CHECK(
        V_sc.dim() == 4 &&
            V_sc.size(0) == C::BATCH &&
            V_sc.size(1) == C::N_TILES &&
            V_sc.size(2) ==
                C::HEADS * C::PV_SCALE_ROWS &&
            V_sc.size(3) == 512);
    TORCH_CHECK(
        O.dim() == 4 &&
            O.size(0) == C::BATCH &&
            O.size(1) == C::N_TILES * 128 &&
            O.size(2) == C::HEADS &&
            O.size(3) == 128);
    TORCH_CHECK(
        LSE.dim() == 4 &&
            LSE.size(0) == C::BATCH &&
            LSE.size(1) == C::HEADS &&
            LSE.size(2) == 1 &&
            LSE.size(3) == C::N_TILES * 128);

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(
            Q_sc,
            C::BATCH,
            C::N_TILES,
            C::HEADS * C::QK_SCALE_ROWS,
            "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(
            Q_sg, C::BATCH, C::HEADS, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(
            K_sc,
            C::BATCH,
            C::N_TILES * 2,
            C::HEADS * C::QK_SCALE_ROWS,
            "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(
            K_sg, C::BATCH, C::HEADS, "K_sg"),
        kittens::py::tensor_to_gl<typename G::v_fp4_gl>(
            V_fp4),
        tensor_to_fp4_scale_gl<typename G::v_sc_gl>(
            V_sc,
            C::BATCH,
            C::N_TILES,
            C::HEADS * C::PV_SCALE_ROWS,
            "V_sc"),
        V_sg,
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        0};
    g.disable_lse_store = store_lse ? 0 : 1;

    const int physical_grid =
        C::LOGICAL_JOBS < C::PHYSICAL_GRID_CAP
            ? C::LOGICAL_JOBS
            : C::PHYSICAL_GRID_CAP;
    constexpr int DYNAMIC_SMEM = Layout::bytes();
    {
        std::lock_guard<std::mutex> lock(
            hao_direct_topology_mutex);
        hao_direct_last_topology = {
            1,
            physical_grid,
            C::LOGICAL_JOBS,
            C::NUM_THREADS,
            C::KV_STAGES,
            C::TMEM_COLUMNS,
            DYNAMIC_SMEM};
    }

    CUDACHECK(cudaFuncSetAttribute(
        kernel_fp4pv_hao_direct_fp4pv<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        DYNAMIC_SMEM));
    kernel_fp4pv_hao_direct_fp4pv<C>
        <<<dim3(physical_grid),
           dim3(C::NUM_THREADS),
           DYNAMIC_SMEM,
           at::cuda::getCurrentCUDAStream()>>>(g);
    CUDACHECK(cudaGetLastError());
}

void dispatch_hao_direct_fp4pv(
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
    bool store_lse) {
    dispatch_hao_direct_fp4pv_impl(
        Q, Q_sc, Q_sg,
        K, K_sc, K_sg,
        V_fp4, V_sc, 1.0f,
        O, LSE,
        p_quant_mode,
        persistent_launch,
        store_lse);
}

void dispatch_hao_direct_fp4pv_vscale(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    float V_sg,
    at::Tensor O,
    at::Tensor LSE,
    int p_quant_mode,
    bool persistent_launch,
    bool store_lse) {
    dispatch_hao_direct_fp4pv_impl(
        Q, Q_sc, Q_sg,
        K, K_sc, K_sg,
        V_fp4, V_sc, V_sg,
        O, LSE,
        p_quant_mode,
        persistent_launch,
        store_lse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_hao_direct_fp4pv",
        &dispatch_hao_direct_fp4pv);
    m.def(
        "forward_hao_direct_fp4pv_vscale",
        &dispatch_hao_direct_fp4pv_vscale);
    m.def(
        "read_hao_direct_topology",
        &read_hao_direct_topology);
}

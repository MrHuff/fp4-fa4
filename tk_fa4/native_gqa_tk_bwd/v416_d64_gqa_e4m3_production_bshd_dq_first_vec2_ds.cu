#include "v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds.cuh"

#include <array>
#include <cmath>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds;

void check_bshd(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 4 &&
            tensor.size(0) > 0 && tensor.size(0) <= 65535 &&
            tensor.size(1) >= 128 && tensor.size(1) % 128 == 0 &&
            tensor.size(2) == heads &&
            tensor.size(3) == candidate::kDepth,
        name,
        " must be contiguous CUDA [B,S,",
        heads,
        ",64] with B in [1,65535] and S a positive multiple of 128"
    );
}

void check_stats(
    const at::Tensor &tensor,
    const char *name,
    const at::Tensor &q
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat && tensor.dim() == 4 &&
            tensor.size(0) == q.size(0) && tensor.size(1) == q.size(2) &&
            tensor.size(2) == 1 && tensor.size(3) == q.size(1),
        name,
        " must be contiguous CUDA FP32 [B,Hq,1,S]"
    );
}

bool byte_ranges_overlap(
    const at::Tensor &left,
    const at::Tensor &right
) {
    const auto left_begin =
        reinterpret_cast<std::uintptr_t>(left.data_ptr());
    const auto right_begin =
        reinterpret_cast<std::uintptr_t>(right.data_ptr());
    return left_begin < right_begin + right.nbytes() &&
        right_begin < left_begin + left.nbytes();
}

void check_arguments(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lstat,
    const at::Tensor &dstat,
    const at::Tensor &dq,
    const at::Tensor &dk,
    const at::Tensor &dv,
    double softmax_scale
) {
    constexpr auto kE4m3 = at::ScalarType::Float8_e4m3fn;
    check_bshd(q, "q", kE4m3, candidate::kQueryHeads);
    check_bshd(k, "k", kE4m3, candidate::kKvHeads);
    check_bshd(v, "v", kE4m3, candidate::kKvHeads);
    check_bshd(dout, "dout", kE4m3, candidate::kQueryHeads);
    check_stats(lstat, "lstat", q);
    check_stats(dstat, "dstat", q);
    check_bshd(dq, "dq", at::kBFloat16, candidate::kQueryHeads);
    check_bshd(dk, "dk", at::kBFloat16, candidate::kKvHeads);
    check_bshd(dv, "dv", at::kBFloat16, candidate::kKvHeads);
    TORCH_CHECK(
        q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
            q.size(1) == k.size(1) && q.size(1) == v.size(1) &&
            q.sizes() == dout.sizes() && q.sizes() == dq.sizes() &&
            k.sizes() == dk.sizes() && k.sizes() == dv.sizes(),
        "v416 tensors must share batch/sequence and gradient shapes"
    );
    kittens::py::device_check(q, k, v, dout, lstat, dstat, dq, dk, dv);
    const float scale = static_cast<float>(softmax_scale);
    const float beta = scale / 16.0f;
    TORCH_CHECK(
        std::isfinite(softmax_scale) && softmax_scale > 0.0 &&
            std::isfinite(scale) && std::isfinite(beta) && beta > 0.0f,
        "softmax_scale must be finite, positive, and representable in FP32"
    );
    using named_tensor = std::pair<const at::Tensor *, const char *>;
    const std::array<named_tensor, 9> tensors{{
        {&q, "q"}, {&k, "k"}, {&v, "v"}, {&dout, "dout"},
        {&lstat, "lstat"}, {&dstat, "dstat"}, {&dq, "dq"},
        {&dk, "dk"}, {&dv, "dv"},
    }};
    for (int i = 6; i < 9; ++i) {
        for (int j = 0; j < i; ++j) {
            TORCH_CHECK(
                !byte_ranges_overlap(*tensors[i].first, *tensors[j].first),
                tensors[i].second,
                " must not overlap ",
                tensors[j].second
            );
        }
    }
}

void launch_checked(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    double softmax_scale,
    bool clear_outputs
) {
    check_arguments(
        q, k, v, dout, lstat, dstat, dq, dk, dv, softmax_scale
    );
    const c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "v414 production-BSHD D64 GQA dQ-first backward requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (clear_outputs) {
        CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), stream));
        CUDACHECK(cudaMemsetAsync(dk.data_ptr(), 0, dk.nbytes(), stream));
        CUDACHECK(cudaMemsetAsync(dv.data_ptr(), 0, dv.nbytes(), stream));
    }
    candidate::launch(
        q, k, v, dout, lstat, dstat, dq, dk, dv,
        static_cast<float>(softmax_scale), stream
    );
}

void main_e4m3_bshd_precomputed(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor dout,
    at::Tensor lstat, at::Tensor dstat, at::Tensor dq, at::Tensor dk,
    at::Tensor dv, double softmax_scale
) {
    launch_checked(
        q, k, v, dout, lstat, dstat, dq, dk, dv, softmax_scale, false
    );
}

void backward_e4m3_bshd_precomputed_out(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor dout,
    at::Tensor lstat, at::Tensor dstat, at::Tensor dq, at::Tensor dk,
    at::Tensor dv, double softmax_scale
) {
    launch_checked(
        q, k, v, dout, lstat, dstat, dq, dk, dv, softmax_scale, true
    );
}

pybind11::dict metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.native_tk_d64_backward.v1";
    result["backend"] = "thunderkittens_sm100a";
    result["source_identity"] =
        "v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds_v1";
    result["source_file"] = __FILE__;
    result["topology"] =
        "split_qhead_cta_k128_q128_async_owner_aligned_tmem_p_half_"
        "overlap_shared_ds_tma_prelifted_stats_bshd_gradient_store_"
        "pipeline_dq_first";
    result["aggregate_candidate"] = true;
    result["attribution_valid"] = true;
    result["aggregate_anchor"] =
        "v406_d64_gqa_e4m3_owner_aligned_dq_first_v1";
    result["attribution_baseline"] =
        "v406_d64_gqa_e4m3_owner_aligned_dq_first_v1";
    result["production_data_abi_compatible"] = true;
    result["existing_runner_drop_in_compatible"] = false;
    result["existing_runner_incompatibility"] =
        "direct_bf16_outputs_replace_v382_fp32_accumulators_and_partials";
    result["aggregate_components"] = pybind11::make_tuple(
        "v391_clamp_native_ex2_traversal",
        "v392_two_stage_async_gradient_publication",
        "v395_tma_stats_transport",
        "v397_per_half_probability_ready_dv_overlap",
        "v399_owner_aligned_score_dp_p_ds_publication"
    );
    result["schedule_experiment"] = "dq_commit_before_dk_commit_vec2_ds_publication";
    result["gradient_mma_issue_order"] = "dq_then_dk";
    result["dq_tmem_alias"] = "retired_dp_tmem";
    result["dq_alias_reuse_gate"] = "dq_drained_before_next_dp";
    result["dk_tmem_alias"] = "none_disjoint_allocation";
    result["threads"] = candidate::kThreads;
    result["key_tile"] = candidate::kKeyTile;
    result["query_tile"] = candidate::kQueryTile;
    result["postprocess_fragment_columns"] = candidate::kColumnHalf;
    result["score_tmem_stages"] = candidate::kStages;
    result["input_smem_stages"] = candidate::kStages;
    result["probability_smem_stages"] = 0;
    result["probability_tmem_stages"] = candidate::kStages;
    result["probability_tmem_operand"] = "true_A";
    result["probability_fp32_lifetime"] = "half_fragment_through_dp_ds";
    result["probability_fp32_owner"] =
        "lane_mod16_row_lane_div16_contiguous_32_columns";
    result["probability_tmem_store"] = "16x32bx2_x8";
    result["dp_tmem_load"] = "16x32bx2_x32";
    result["ds_shared_store"] = "owner_aligned_v2_b32";
    result["probability_exp"] =
        "native_ex2_clamp_production_log2_pscaled_le_8";
    result["lossy_probability_alu"] = false;
    result["ds_smem_stages"] = candidate::kStages;
    result["probability_ds_alias"] = false;
    result["wait_dv_before_ds_current_phase"] = false;
    result["wait_dv_before_score_tmem_reuse"] = true;
    result["dp_issued_before_probability_ready"] = true;
    result["probability_ready_granularity"] = "per_half_64_columns";
    result["dv_issue_granularity"] = "two_k32_chunks_per_half";
    result["steady_state_cta_barriers"] = 0;
    result["gradient_publication_stages"] =
        candidate::kGradientPublicationStages;
    result["gradient_publication_wait_policy"] =
        "read_wait_1_before_two_stage_reuse_final_wait_0";
    result["stats_transport"] = "loader_owned_tma_prelifted_fp32";
    result["stats_scaling"] =
        "identity_fp32x2_fma_preserving_v406_instruction_shape";
    result["probability_scale_source"] = "lstat_embedded_plus_8";
    result["post_exp2_probability_scale"] = 1.0;
    result["v406_parity_l_aux_transform"] =
        "(production_lstat-8)/(softmax_scale*log2(e))";
    result["v406_parity_delta_transform"] = "production_dstat/-16";
    result["sequence"] = "dynamic_positive_multiple_of_128";
    result["production_sequence"] = 4096;
    result["query_heads"] = candidate::kQueryHeads;
    result["kv_heads"] = candidate::kKvHeads;
    result["head_dim"] = candidate::kDepth;
    result["causal"] = true;
    result["operand_dtype"] = "float8_e4m3fn";
    result["operand_layout"] = "BSHD_contiguous";
    result["encoding_scale"] = candidate::kOperandScale;
    result["lstat_abi"] = "8-LSE*log2(e)";
    result["dstat_abi"] = "-16*sum(O*dO)";
    result["stats_layout"] = "B,Hq,1,S_fp32_contiguous";
    result["public_softmax_scale"] = "natural";
    result["internal_beta_divisor"] = 16.0;
    result["gradient_epilogue_scale"] = 1.0 / 256.0;
    result["output_dtype"] = "bfloat16_additive";
    result["output_layout"] = "BSHD_contiguous";
    result["output_encoding_scale"] = candidate::kOperandScale;
    result["caller_owned_output_api"] = true;
    result["caller_zeros_outputs_for_main"] = true;
    result["backward_out_clears_outputs"] = true;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "main_e4m3_bshd_precomputed",
        &main_e4m3_bshd_precomputed
    );
    module.def(
        "backward_e4m3_bshd_precomputed_out",
        &backward_e4m3_bshd_precomputed_out
    );
    module.def("native_tk_d64_backward_metadata", &metadata);
}

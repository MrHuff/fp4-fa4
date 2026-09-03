#include "v478_d128_gqa_e4m3_b2_s4096_owner4_two_chunk_dq_tmem_release_production_bshd.cuh"

#include <array>
#include <cmath>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v478_d128_gqa_e4m3_b2_s4096_owner4_two_chunk_dq_tmem_release_production_bshd;

void check_bshd(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 4 &&
            (tensor.size(0) == 1 || tensor.size(0) == 2) &&
            tensor.size(1) >= 128 && tensor.size(1) % 128 == 0 &&
            tensor.size(2) == heads &&
            tensor.size(3) == candidate::prior::kDepth,
        name,
        " must be contiguous CUDA [B,S,",
        heads,
        ",128] with B in {1,2} and S a positive multiple of 128"
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
            tensor.size(0) == q.size(0) &&
            tensor.size(1) == q.size(2) && tensor.size(2) == 1 &&
            tensor.size(3) == q.size(1),
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
        "v478 tensors must share batch/sequence and gradient shapes"
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
    for (int output = 6; output < 9; ++output) {
        for (int earlier = 0; earlier < output; ++earlier) {
            TORCH_CHECK(
                !byte_ranges_overlap(
                    *tensors[output].first,
                    *tensors[earlier].first
                ),
                tensors[output].second,
                " must not overlap ",
                tensors[earlier].second
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
        "v478 production-BSHD D128 GQA backward requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (clear_outputs) {
        CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), stream));
        const bool owner4_direct_route =
            q.size(0) == 2 && q.size(1) == candidate::prior::kExactSequence;
        if (!owner4_direct_route) {
            CUDACHECK(cudaMemsetAsync(dk.data_ptr(), 0, dk.nbytes(), stream));
            CUDACHECK(cudaMemsetAsync(dv.data_ptr(), 0, dv.nbytes(), stream));
        }
    }
    candidate::launch(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        static_cast<float>(softmax_scale),
        stream
    );
}

void main_e4m3_bshd_precomputed(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    double softmax_scale
) {
    launch_checked(
        q, k, v, dout, lstat, dstat, dq, dk, dv, softmax_scale, false
    );
}

void backward_e4m3_bshd_precomputed_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    double softmax_scale
) {
    launch_checked(
        q, k, v, dout, lstat, dstat, dq, dk, dv, softmax_scale, true
    );
}

pybind11::dict metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.native_tk_d128_backward.v1";
    result["backend"] = "thunderkittens_sm100a";
    result["source_identity"] =
        "v478_d128_gqa_e4m3_b2_s4096_owner4_two_chunk_dq_tmem_release_production_bshd_v1";
    result["source_file"] = __FILE__;
    result["topology"] =
        "b2_s4096_flat_batch_kv_owner_fast_owner4_exact_runtime_"
        "retained_kv_fp32_dkdv_direct_store_private_additive_dq_"
        "compact_rounded_e4m3_p_register_reuse_first_half_quarter_dp_"
        "split_dv_k32_halves_dv_ready_gated_shared_probability_reuse_"
        "operand_consumed_gated_stats_stage_reuse_"
        "later_dk_commit_gated_operand_stage_reuse_"
        "later_dk_commit_gated_shared_ds_reuse_"
        "head_boundary_score_before_dq_drain_"
        "paired_final_x32_loads_one_wait_split_dq_tmem_release_"
        "before_final_two_chunk_bf16_pack_"
        "shared_dq_publication_ready_retained_"
        "b1_v436_other_b2_v437";
    result["direct_output_entrypoint"] =
        "backward_e4m3_bshd_precomputed_out";
    result["production_data_abi_compatible"] = true;
    result["caller_owned_output_api"] = true;
    result["caller_zeros_outputs_for_main"] = true;
    result["backward_out_clears_outputs"] = true;
    result["backward_out_physical_clear_policy"] =
        "dq_always_dkdv_except_unique_writer_B2_S4096_owner4";
    result["threads"] = candidate::kThreads;
    result["key_tile"] = candidate::kKeyTile;
    result["query_tile"] = candidate::kQueryTile;
    result["query_heads"] = candidate::kQueryHeads;
    result["kv_heads"] = candidate::kKvHeads;
    result["head_ratio"] = candidate::kHeadRatio;
    result["head_dim"] = candidate::prior::kDepth;
    result["batch_values"] = pybind11::make_tuple(1, 2);
    result["sequence"] = "dynamic_positive_multiple_of_128";
    result["owner4_exact_sequence"] = 4096;
    result["gradient_accumulate_control"] =
        "runtime_predicate_work_nonzero_on_first_cta_group1_f8f6f4_mma";
    result["gradient_k32_b_descriptor_correction"] = true;
    result["probability_reuse"] =
        "exact_rounded_e4m3_words_from_dv_publication_to_ds";
    result["probability_fp32_retained"] = false;
    result["probability_shared_for_dv"] = true;
    result["probability_shared_reuse_gate"] =
        "dv_ready_after_all_four_k32_dv_commands_no_probability_barrier";
    result["probability_ds_source"] =
        "retained_per_warp_compact_e4m3_registers";
    result["stats_shared_reuse_gate"] =
        "per_stage_operand_consumed_after_aggregate_ds_ready_and_later_dk_"
        "commit_tracking_prior_dv_completion";
    result["stats_ready_barrier"] = true;
    result["stats_consumed_barrier"] = false;
    result["operand_stage_reuse_gate"] =
        "later_dk_ready_commit_tracks_all_prior_same_thread_tcgen05_"
        "operations_including_both_dv_halves";
    result["shared_ds_reuse_gate"] =
        "later_dk_ready_commit_tracks_prior_same_thread_dq_then_dk_"
        "commands_and_commits";
    result["compute_old_dq_ready_wait"] = false;
    result["dq_ready_barrier_retained"] = true;
    result["dq_ready_reducer_wait_retained"] = true;
    result["dq_drained_reuse_edges_retained"] = true;
    result["dq_tmem_release_barrier"] = "dq_tmem_drained";
    result["dq_tmem_release_point"] =
        "after_paired_final_x32_tmem_loads_complete_before_final_two_"
        "chunk_bf16_pack_and_shared_stores";
    result["dq_tmem_release_control_flow"] =
        "paired_final_two_depth_chunks_one_tensor_load_wait";
    result["dq_shared_publication_ready_barrier"] = "dq_drained";
    result["head_boundary_score_schedule"] =
        "next_query_ready_then_score_issue_before_previous_dq_drained";
    result["head_boundary_dp_reuse_gate"] =
        "previous_dq_tmem_drained_before_aliased_dp_issue";
    result["issuer_explicit_dv_wait"] = false;
    result["dv_ready_barrier_retained"] = true;
    result["probability_publication"] =
        "per_half_after_shared_store_proxy_fence_eight_compute_warps";
    result["compact_probability_words_per_lane"] =
        2 * candidate::kCompactProbabilityWords;
    result["dp_half0_tmem_schedule"] =
        "x16_chunk0_issue_before_previous_dk_wait_only_wait_chunk0_"
        "issue_chunk1_fence_before_thread_sync_consume_chunk0_"
        "wait_chunk1_consume_chunk1";
    result["dp_chunk1_issue_anchor"] =
        "tcgen05_fence_before_thread_sync_only_no_warp_sync";
    result["dp_half0_chunk_mapping"] =
        "owner_aligned_chunk0_cols_0_15_32_47_chunk1_cols_16_31_48_63";
    result["dp_half1_tmem_schedule"] =
        "unchanged_issue_and_wait_at_second_ds";
    result["dv_k32_schedule"] =
        "chunks_0_1_after_probability_half0_chunks_2_3_after_half1";
    result["dv_completion_commit"] =
        "half1_only_dv_ready_commit_with_optional_ordered_score_next_commit";
    result["score_tmem_release"] =
        "after_half1_score_load_before_half1_probability_math";
    result["causal"] = true;
    result["operand_dtype"] = "float8_e4m3fn";
    result["operand_layout"] = "BSHD_contiguous";
    result["encoding_scale"] = candidate::prior::kOperandScale;
    result["lstat_abi"] = "8-LSE*log2(e)";
    result["dstat_abi"] = "-16*sum(O*dO)";
    result["stats_layout"] = "B,Hq,1,S_fp32_contiguous";
    result["public_softmax_scale"] = "natural";
    result["internal_beta_divisor"] = 16.0;
    result["gradient_epilogue_scale"] = 1.0 / 256.0;
    result["output_dtype"] = "bfloat16_additive";
    result["output_layout"] = "BSHD_contiguous";
    result["output_encoding_scale"] = candidate::prior::kOperandScale;
    result["batch_dispatch"] =
        "B2_S4096_owner4_B1_v436_other_B2_v437";
    result["owner4_grid"] = "B_times_8,S_div_128,1";
    result["owner4_heads"] = candidate::kHeadsPerOwner;
    result["owner4_phase_policy"] =
        "monotonic_work_index_across_four_heads";
    result["owner4_dq_policy"] =
        "private_per_head_full_width_additive_tma";
    result["owner4_dkdv_policy"] =
        "fp32_tmem_accumulate_four_heads_then_unique_owner_direct_tma";
    result["owner4_dkdv_unique_writer"] = true;
    result["owner4_dkdv_destination_preclear_required"] = false;
    result["user_shared_storage_bytes"] =
        static_cast<int>(sizeof(candidate::shared_storage));
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
    module.def("native_tk_d128_backward_metadata", &metadata);
}

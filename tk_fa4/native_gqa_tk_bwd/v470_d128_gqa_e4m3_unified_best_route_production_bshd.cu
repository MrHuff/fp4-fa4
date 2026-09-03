#include "v470_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"

#include <array>
#include <cmath>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v470_d128_gqa_e4m3_unified_best_route_production_bshd;

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
            tensor.size(2) == heads && tensor.size(3) == candidate::kDepth,
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
        "v470 tensors must share batch/sequence and gradient shapes"
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
        "v470 production-BSHD D128 GQA backward requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (clear_outputs) {
        CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), stream));
        // Every route publishes additive dK/dV except exact B2/S4096.  That
        // route has one complete writer per destination and directly
        // overwrites dK/dV, so only its additive dQ needs a physical clear.
        if (!candidate::is_b2_exact_direct_route(q.size(0), q.size(1))) {
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
        "v470_d128_gqa_e4m3_unified_best_route_production_bshd_v1";
    result["source_file"] = __FILE__;
    result["topology"] =
        "thin_frozen_dispatch_B1_S4096_v445_owner2_compact_probability_"
        "B2_S4096_v469_owner4_compact_probability_first_half_quarter_dp_"
        "fence_before_thread_sync_split_dv_dv_ready_gated_probability_"
        "operand_consumed_gated_stats_later_dk_commit_gated_operand_stage_"
        "reuse_later_dk_commit_gated_shared_ds_reuse_direct_dkdv_"
        "B1_other_v436_B2_other_v437";
    result["direct_output_entrypoint"] =
        "backward_e4m3_bshd_precomputed_out";
    result["production_data_abi_compatible"] = true;
    result["caller_owned_output_api"] = true;
    result["caller_zeros_outputs_for_main"] = true;
    result["backward_out_clears_outputs"] = true;
    result["backward_out_logical_reset_policy"] =
        "dq_dk_dv_all_routes";
    result["backward_out_physical_clear_policy"] =
        "B2_S4096_memset_dq_only_complete_unique_direct_overwrite_dkdv;"
        "B1_S4096_B1_nonexact_B2_nonexact_memset_dq_dk_dv";
    result["dispatch"] =
        "B1_S4096_v445;B2_S4096_v469;B1_other_v436;B2_other_v437";
    result["selected_exact_kernels"] = pybind11::make_tuple(
        "v445::b1_owner2_exact_s4096_compact_p_reuse_kernel",
        "v469::owner4_kernel"
    );
    result["selected_fallback_kernels"] = pybind11::make_tuple(
        "v436::main_kernel",
        "v437::owner2_kernel"
    );
    result["threads"] = candidate::kThreads;
    result["key_tile"] = candidate::kKeyTile;
    result["query_tile"] = candidate::kQueryTile;
    result["query_heads"] = candidate::kQueryHeads;
    result["kv_heads"] = candidate::kKvHeads;
    result["head_ratio"] = candidate::kHeadRatio;
    result["head_dim"] = candidate::kDepth;
    result["batch_values"] = pybind11::make_tuple(1, 2);
    result["sequence"] = "dynamic_positive_multiple_of_128";
    result["exact_sequence_specialization"] = candidate::kExactSequence;
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
    // Validator ABI: this string describes the public gradient boundary,
    // independent of the route-specific physical reset/overwrite mechanism.
    result["output_dtype"] = "bfloat16_additive";
    result["output_layout"] = "BSHD_contiguous";
    result["output_encoding_scale"] = candidate::kOperandScale;
    result["b1_s4096_route"] =
        "v445_owner2_compact_p_additive_dq_dk_dv";
    result["b2_s4096_route"] =
        "v469_owner4_compact_p_first_half_quarter_dp_"
        "fence_before_thread_sync_split_dv_dv_ready_gated_probability_"
        "operand_consumed_gated_stats_later_dk_commit_gated_operand_stage_"
        "reuse_later_dk_commit_gated_shared_ds_reuse_additive_dq_unique_"
        "direct_dkdv";
    result["b1_nonexact_route"] = "v436_additive_dq_dk_dv";
    result["b2_nonexact_route"] = "v437_additive_dq_dk_dv";
    result["b1_s4096_owner_heads"] =
        candidate::b1_exact::kHeadsPerOwner;
    result["b2_s4096_owner_heads"] =
        candidate::b2_exact_and_fallbacks::kHeadsPerOwner;
    result["b1_s4096_probability_reuse"] =
        "exact_rounded_e4m3_words_from_dv_publication_to_ds";
    result["b2_s4096_probability_reuse"] =
        "exact_rounded_e4m3_words_from_dv_publication_to_ds";
    result["b2_s4096_probability_shared_reuse_gate"] =
        "dv_ready_after_all_four_k32_dv_commands_no_probability_barrier";
    result["b2_s4096_probability_ds_source"] =
        "retained_per_warp_compact_e4m3_registers";
    result["b2_s4096_stats_shared_reuse_gate"] =
        "per_stage_operand_consumed_after_aggregate_ds_ready_and_later_dk_"
        "commit_tracking_prior_dv_completion";
    result["b2_s4096_stats_ready_barrier"] = true;
    result["b2_s4096_stats_consumed_barrier"] = false;
    result["b2_s4096_operand_stage_reuse_gate"] =
        "later_dk_ready_commit_tracks_all_prior_same_thread_tcgen05_"
        "operations_including_both_dv_halves";
    result["b2_s4096_issuer_explicit_dv_wait"] = false;
    result["b2_s4096_dv_ready_barrier_retained"] = true;
    result["b2_s4096_shared_ds_reuse_gate"] =
        "later_dk_ready_commit_tracks_prior_same_thread_dq_then_dk_"
        "commands_and_commits";
    result["b2_s4096_compute_old_dq_ready_wait"] = false;
    result["b2_s4096_dq_ready_barrier_retained"] = true;
    result["b2_s4096_dq_ready_reducer_wait_retained"] = true;
    result["b2_s4096_dq_drained_reuse_edges_retained"] = true;
    result["b2_s4096_dp_half0_tmem_schedule"] =
        "x16_chunk0_issue_before_previous_dk_wait_only_wait_chunk0_"
        "issue_chunk1_fence_before_thread_sync_consume_chunk0_"
        "wait_chunk1_consume_chunk1";
    result["b2_s4096_dp_chunk1_issue_anchor"] =
        "tcgen05_fence_before_thread_sync_only_no_warp_sync";
    result["b2_s4096_dp_half0_chunk_mapping"] =
        "owner_aligned_chunk0_cols_0_15_32_47_chunk1_cols_16_31_48_63";
    result["b2_s4096_dp_half1_tmem_schedule"] =
        "unchanged_issue_and_wait_at_second_ds";
    result["b2_s4096_probability_publication"] =
        "per_half_after_shared_store_proxy_fence_eight_compute_warps";
    result["b2_s4096_dv_k32_schedule"] =
        "chunks_0_1_after_probability_half0_chunks_2_3_after_half1";
    result["b2_s4096_dv_completion_commit"] =
        "half1_only_dv_ready_commit_with_optional_ordered_score_next_commit";
    result["b2_s4096_score_tmem_release"] =
        "after_half1_score_load_before_half1_probability_math";
    result["b2_s4096_dkdv_unique_writer"] = true;
    result["b2_s4096_dkdv_destination_preclear_required"] = false;
    result["b1_s4096_user_shared_storage_bytes"] =
        static_cast<int>(sizeof(candidate::b1_exact::shared_storage));
    result["b2_s4096_user_shared_storage_bytes"] = static_cast<int>(
        sizeof(candidate::b2_exact_and_fallbacks::shared_storage)
    );
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

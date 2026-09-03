#include "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_reuse_production_bshd.cuh"

#include <array>
#include <cmath>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_reuse_production_bshd;

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
        "v445 tensors must share batch/sequence and gradient shapes"
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
        "v445 production-BSHD D128 GQA backward requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (clear_outputs) {
        CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), stream));
        CUDACHECK(cudaMemsetAsync(dk.data_ptr(), 0, dk.nbytes(), stream));
        CUDACHECK(cudaMemsetAsync(dv.data_ptr(), 0, dv.nbytes(), stream));
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
        "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_reuse_production_bshd_v1";
    result["source_file"] = __FILE__;
    result["topology"] =
        "b1_s4096_flat_batch_head_pair_fast_owner2_exact_schedule_"
        "retained_kv_fp32_dkdv_private_dq_runtime_first_accumulate_"
        "compact_rounded_e4m3_p_register_reuse_"
        "all_other_shapes_v438_fallback";
    result["direct_output_entrypoint"] =
        "backward_e4m3_bshd_precomputed_out";
    result["production_data_abi_compatible"] = true;
    result["caller_owned_output_api"] = true;
    result["caller_zeros_outputs_for_main"] = true;
    result["backward_out_clears_outputs"] = true;
    result["threads"] = candidate::kThreads;
    result["key_tile"] = candidate::kKeyTile;
    result["query_tile"] = candidate::kQueryTile;
    result["query_heads"] = candidate::kQueryHeads;
    result["kv_heads"] = candidate::kKvHeads;
    result["head_ratio"] = candidate::kHeadRatio;
    result["head_dim"] = candidate::prior::kDepth;
    result["batch_values"] = pybind11::make_tuple(1, 2);
    result["sequence"] = "dynamic_positive_multiple_of_128";
    result["exact_sequence_specialization"] = candidate::kExactSequence;
    result["non_specialized_fallback"] =
        "v438_b2_owner2_exact_s4096";
    result["gradient_accumulate_control"] =
        "runtime_predicate_on_first_owner_work_cta_group1_f8f6f4_mma";
    result["gradient_k32_b_descriptor_correction"] = true;
    result["probability_reuse"] =
        "exact_rounded_e4m3_words_from_dv_publication_to_ds";
    result["probability_fp32_retained"] = false;
    result["probability_shared_for_dv"] = true;
    result["compact_probability_words_per_lane"] =
        2 * candidate::kCompactProbabilityWords;
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
        "B1_S4096_owner2_exact_all_other_shapes_v438";
    result["owner2_grid"] = "B_times_16,S_div_128,1";
    result["owner2_heads"] = candidate::kHeadsPerOwner;
    result["owner2_phase_policy"] =
        "monotonic_work_index_across_both_heads";
    result["owner2_dq_policy"] = "private_per_head_full_width_tma";
    result["owner2_dkdv_policy"] =
        "fp32_tmem_runtime_accumulate_two_heads_then_pair_additive_tma";
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

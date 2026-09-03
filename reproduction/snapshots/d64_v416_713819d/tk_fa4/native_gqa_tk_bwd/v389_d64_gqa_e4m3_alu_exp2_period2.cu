#include "v389_d64_gqa_e4m3_alu_exp2_period2.cuh"

#include <array>
#include <cmath>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v389_d64_gqa_e4m3_alu_exp2_period2;

void check_bhsd(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 4 &&
            tensor.size(0) > 0 && tensor.size(0) <= 65535 &&
            tensor.size(1) == heads && tensor.size(2) >= 128 &&
            tensor.size(2) % 128 == 0 &&
            tensor.size(3) == candidate::kDepth,
        name,
        " must be contiguous CUDA [B,",
        heads,
        ",S,64] with B in [1,65535] and S a positive multiple of 128"
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
            tensor.size(0) == q.size(0) && tensor.size(1) == q.size(1) &&
            tensor.size(2) == 1 && tensor.size(3) == q.size(2),
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
    const at::Tensor &l_aux,
    const at::Tensor &delta,
    const at::Tensor &dq,
    const at::Tensor &dk,
    const at::Tensor &dv,
    double softmax_scale
) {
    constexpr auto kE4m3 = at::ScalarType::Float8_e4m3fn;
    check_bhsd(q, "q", kE4m3, candidate::kQueryHeads);
    check_bhsd(k, "k", kE4m3, candidate::kKvHeads);
    check_bhsd(v, "v", kE4m3, candidate::kKvHeads);
    check_bhsd(dout, "dout", kE4m3, candidate::kQueryHeads);
    check_stats(l_aux, "l_aux", q);
    check_stats(delta, "delta", q);
    check_bhsd(dq, "dq", at::kBFloat16, candidate::kQueryHeads);
    check_bhsd(dk, "dk", at::kBFloat16, candidate::kKvHeads);
    check_bhsd(dv, "dv", at::kBFloat16, candidate::kKvHeads);
    TORCH_CHECK(
        q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
            q.size(2) == k.size(2) && q.size(2) == v.size(2) &&
            q.sizes() == dout.sizes() && q.sizes() == dq.sizes() &&
            k.sizes() == dk.sizes() && k.sizes() == dv.sizes(),
        "v389 tensors must share batch/sequence and gradient shapes"
    );
    kittens::py::device_check(q, k, v, dout, l_aux, delta, dq, dk, dv);
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
        {&l_aux, "l_aux"}, {&delta, "delta"}, {&dq, "dq"},
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
    at::Tensor l_aux,
    at::Tensor delta,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    double softmax_scale,
    bool clear_outputs
) {
    check_arguments(
        q, k, v, dout, l_aux, delta, dq, dk, dv, softmax_scale
    );
    const c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "v389 D64 GQA asynchronous backward requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (clear_outputs) {
        CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), stream));
        CUDACHECK(cudaMemsetAsync(dk.data_ptr(), 0, dk.nbytes(), stream));
        CUDACHECK(cudaMemsetAsync(dv.data_ptr(), 0, dv.nbytes(), stream));
    }
    candidate::launch(
        q, k, v, dout, l_aux, delta, dq, dk, dv,
        static_cast<float>(softmax_scale), stream
    );
}

void main_e4m3_bhsd(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor dout,
    at::Tensor l_aux, at::Tensor delta, at::Tensor dq, at::Tensor dk,
    at::Tensor dv, double softmax_scale
) {
    launch_checked(
        q, k, v, dout, l_aux, delta, dq, dk, dv, softmax_scale, false
    );
}

void backward_e4m3_bhsd_out(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor dout,
    at::Tensor l_aux, at::Tensor delta, at::Tensor dq, at::Tensor dk,
    at::Tensor dv, double softmax_scale
) {
    launch_checked(
        q, k, v, dout, l_aux, delta, dq, dk, dv, softmax_scale, true
    );
}

pybind11::dict metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.native_tk_d64_backward.v1";
    result["source_identity"] =
        "v389_d64_gqa_e4m3_alu_exp2_period2_v1";
    result["topology"] =
        "single_cta_k128_q128_async_score2_input2_probability2";
    result["exp2_policy"] =
        "degree1_alu_on_even_packed_pairs_native_ex2_on_odd_pairs";
    result["exp2_alu_degree"] = 1;
    result["exp2_alu_period_pairs"] = 2;
    result["l_aux_contract"] =
        "device_lstat_is_unprelifted_negative_lse_log2";
    result["probability_log2_ceiling"] = 0.0;
    result["probability_lift"] = 256.0;
    result["probability_lift_site"] = "post_exp2_before_e4m3_conversion";
    result["probability_lift_applied_once"] = true;
    result["threads"] = candidate::kThreads;
    result["key_tile"] = candidate::kKeyTile;
    result["query_tile"] = candidate::kQueryTile;
    result["postprocess_fragment_columns"] = candidate::kColumnHalf;
    result["score_tmem_stages"] = candidate::kStages;
    result["input_smem_stages"] = candidate::kStages;
    result["probability_smem_stages"] = candidate::kStages;
    result["steady_state_cta_barriers"] = 0;
    result["query_heads"] = candidate::kQueryHeads;
    result["kv_heads"] = candidate::kKvHeads;
    result["head_dim"] = candidate::kDepth;
    result["causal"] = true;
    result["operand_dtype"] = "float8_e4m3fn";
    result["operand_layout"] = "BHSD_contiguous";
    result["operand_value_scale"] = candidate::kOperandScale;
    result["output_dtype"] = "bfloat16_additive";
    result["gradient_value_scale"] = candidate::kOperandScale;
    result["caller_zeros_outputs_for_main"] = true;
    result["backward_out_clears_outputs"] = true;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("main_e4m3_bhsd", &main_e4m3_bhsd);
    module.def("backward_e4m3_bhsd_out", &backward_e4m3_bhsd_out);
    module.def("native_tk_d64_backward_metadata", &metadata);
}

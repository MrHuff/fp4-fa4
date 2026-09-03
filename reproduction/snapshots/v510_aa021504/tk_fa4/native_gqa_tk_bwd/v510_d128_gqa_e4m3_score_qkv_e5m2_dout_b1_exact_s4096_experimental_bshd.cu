#include "v510_d128_gqa_e4m3_score_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cuh"

#include <array>
#include <cmath>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v510_d128_gqa_e4m3_score_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd;

constexpr int kBatch = 1;
constexpr int kSequence = 4096;
constexpr int kDepth = candidate::core::kDepth;

void check_bshd(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 4 &&
            tensor.size(0) == kBatch && tensor.size(1) == kSequence &&
            tensor.size(2) == heads && tensor.size(3) == kDepth,
        name,
        " must be contiguous CUDA [1,4096,",
        heads,
        ",128]"
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
    check_bshd(
        dout,
        "dout_e5m2",
        at::ScalarType::Float8_e5m2,
        candidate::kQueryHeads
    );
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
        "v510 tensors must share exact batch/sequence and gradient shapes"
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
        "v510 dense-score E5M2-dO experiment requires SM100"
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

void main_e4m3_score_qkv_e5m2_dout_bshd_precomputed(
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

void backward_e4m3_score_qkv_e5m2_dout_bshd_precomputed_out(
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
        "v510_dense_e4m3_score_qkv_e5m2_dout_b1_s4096_experimental_v1";
    result["source_file"] = __FILE__;
    result["experimental"] = true;
    result["production_dispatch_connected"] = false;
    result["dispatch"] = "fail_closed_B1_S4096_only_no_fallback";
    result["selected_kernel"] =
        "v510::b1_dense_e4m3_score_qkv_e5m2_dout_exact_s4096_kernel";
    result["score_qk_dtype"] = "float8_e4m3fn_represented_x4";
    result["score_qk_layout"] = "BSHD_contiguous";
    result["score_mma"] = "dense_E4M3_A_times_E4M3_B_transpose";
    result["gradient_qkv_dtype"] = "float8_e4m3fn_represented_x4";
    result["dout_dtype"] = "float8_e5m2_represented_x4";
    result["dout_encode_scale"] = 4.0;
    result["dout_decode_scale"] = 0.25;
    result["mixed_mma_b_format_mask"] = 1024;
    result["score_internal_beta_divisor"] = 16.0;
    result["ds_internal_beta_divisor"] = 16.0;
    result["lstat_abi"] = "8-LSE*log2(e)";
    result["dstat_abi"] = "-16*sum(O*dO)";
    result["dstat_physical_abi"] = "-4*sum(O*raw_E5M2_dO)";
    result["output_dtype"] = "bfloat16_additive";
    result["batch"] = kBatch;
    result["sequence"] = kSequence;
    result["query_heads"] = candidate::kQueryHeads;
    result["kv_heads"] = candidate::kKvHeads;
    result["head_dim"] = kDepth;
    result["threads"] = candidate::kThreads;
    result["user_shared_storage_bytes"] =
        static_cast<int>(sizeof(candidate::shared_storage));
    result["score_schedule"] =
        "dense_E4M3_score_then_mixed_E4M3_E5M2_dP_dV";
    result["caller_owned_output_api"] = true;
    result["main_requires_precleared_outputs"] = true;
    result["backward_out_clears_dq_dk_dv"] = true;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "main_e4m3_score_qkv_e5m2_dout_bshd_precomputed",
        &main_e4m3_score_qkv_e5m2_dout_bshd_precomputed
    );
    module.def(
        "backward_e4m3_score_qkv_e5m2_dout_bshd_precomputed_out",
        &backward_e4m3_score_qkv_e5m2_dout_bshd_precomputed_out
    );
    module.def("native_tk_d128_backward_metadata", &metadata);
}

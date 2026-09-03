#ifndef TKFA4_V509_EXACT_BATCH
#define TKFA4_V509_EXACT_BATCH 1
#endif

#include "v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cuh"

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd;

static_assert(
    TKFA4_V509_EXACT_BATCH == 1 || TKFA4_V509_EXACT_BATCH == 2 ||
        TKFA4_V509_EXACT_BATCH == 4,
    "v509 wrapper supports only separately authenticated B1, B2, or B4 builds"
);
constexpr int kBatch = TKFA4_V509_EXACT_BATCH;
constexpr int kSequence = candidate::kExactSequence;
constexpr int kDepth = candidate::core::kDepth;
constexpr int kDepthChunks = kDepth / 64;

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
        " must be contiguous CUDA [",
        kBatch,
        ",4096,",
        heads,
        ",128]"
    );
}

void check_stats(
    const at::Tensor &tensor,
    const char *name
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat && tensor.dim() == 4 &&
            tensor.size(0) == kBatch &&
            tensor.size(1) == candidate::kQueryHeads &&
            tensor.size(2) == 1 && tensor.size(3) == kSequence,
        name,
        " must be contiguous CUDA FP32 [",
        kBatch,
        ",32,1,4096]"
    );
}

void check_native_payload(
    const at::Tensor &tensor,
    const char *name,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
            tensor.dim() == 4 && tensor.size(0) == kBatch &&
            tensor.size(1) == heads && tensor.size(2) == kSequence &&
            tensor.size(3) == kDepth / 2,
        name,
        " must be contiguous CUDA float4_e2m1fn_x2 [",
        kBatch,
        ",",
        heads,
        ",4096,64] in BHSD packed layout"
    );
}

void check_native_scale(
    const at::Tensor &tensor,
    const char *name,
    int sequence_tiles,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            tensor.dim() == 4 && tensor.size(0) == kBatch &&
            tensor.size(1) == sequence_tiles &&
            tensor.size(2) == heads * kDepthChunks &&
            tensor.size(3) == 512,
        name,
        " must be contiguous CUDA float8_e4m3fn [",
        kBatch,
        ",",
        sequence_tiles,
        ",",
        heads * kDepthChunks,
        ",512]"
    );
}

void check_global_scale(
    const at::Tensor &tensor,
    const char *name,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat && tensor.dim() == 2 &&
            tensor.size(0) == kBatch && tensor.size(1) == heads,
        name,
        " must be contiguous CUDA FP32 [",
        kBatch,
        ",",
        heads,
        "]"
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
    const at::Tensor &q_native,
    const at::Tensor &k_native,
    const at::Tensor &q_native_scale,
    const at::Tensor &k_native_scale,
    const at::Tensor &q_global_scale,
    const at::Tensor &k_global_scale,
    double softmax_scale
) {
    constexpr auto kE4m3 = at::ScalarType::Float8_e4m3fn;
    constexpr auto kE5m2 = at::ScalarType::Float8_e5m2;
    check_bshd(q, "q_e4m3", kE4m3, candidate::kQueryHeads);
    check_bshd(k, "k_e4m3", kE4m3, candidate::kKvHeads);
    check_bshd(v, "v_e4m3", kE4m3, candidate::kKvHeads);
    check_bshd(dout, "dout_e5m2", kE5m2, candidate::kQueryHeads);
    check_stats(lstat, "lstat");
    check_stats(dstat, "dstat");
    check_bshd(dq, "dq", at::kBFloat16, candidate::kQueryHeads);
    check_bshd(dk, "dk", at::kBFloat16, candidate::kKvHeads);
    check_bshd(dv, "dv", at::kBFloat16, candidate::kKvHeads);
    check_native_payload(
        q_native, "q_native", candidate::kQueryHeads
    );
    check_native_payload(
        k_native, "k_native", candidate::kKvHeads
    );
    check_native_scale(
        q_native_scale,
        "q_native_scale",
        kSequence / candidate::kQueryTile,
        candidate::kQueryHeads
    );
    check_native_scale(
        k_native_scale,
        "k_native_scale",
        kSequence / 64,
        candidate::kKvHeads
    );
    check_global_scale(
        q_global_scale, "q_global_scale", candidate::kQueryHeads
    );
    check_global_scale(
        k_global_scale, "k_global_scale", candidate::kKvHeads
    );

    kittens::py::device_check(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        q_native,
        k_native,
        q_native_scale,
        k_native_scale,
        q_global_scale,
        k_global_scale
    );

    const float scale = static_cast<float>(softmax_scale);
    const float beta = scale / 16.0f;
    TORCH_CHECK(
        std::isfinite(softmax_scale) && softmax_scale > 0.0 &&
            std::isfinite(scale) && std::isfinite(beta) && beta > 0.0f,
        "softmax_scale must be finite, positive, and representable in FP32"
    );

    using named_tensor = std::pair<const at::Tensor *, const char *>;
    const std::array<named_tensor, 15> tensors{{
        {&q, "q_e4m3"},
        {&k, "k_e4m3"},
        {&v, "v_e4m3"},
        {&dout, "dout_e5m2"},
        {&lstat, "lstat"},
        {&dstat, "dstat"},
        {&dq, "dq"},
        {&dk, "dk"},
        {&dv, "dv"},
        {&q_native, "q_native"},
        {&k_native, "k_native"},
        {&q_native_scale, "q_native_scale"},
        {&k_native_scale, "k_native_scale"},
        {&q_global_scale, "q_global_scale"},
        {&k_global_scale, "k_global_scale"},
    }};
    for (int output = 6; output <= 8; ++output) {
        for (int other = 0; other < static_cast<int>(tensors.size()); ++other) {
            if (output == other) {
                continue;
            }
            TORCH_CHECK(
                !byte_ranges_overlap(
                    *tensors[output].first,
                    *tensors[other].first
                ),
                tensors[output].second,
                " must not overlap ",
                tensors[other].second
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
    at::Tensor q_native,
    at::Tensor k_native,
    at::Tensor q_native_scale,
    at::Tensor k_native_scale,
    at::Tensor q_global_scale,
    at::Tensor k_global_scale,
    double softmax_scale,
    bool clear_dq,
    bool clear_dkdv
) {
    check_arguments(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        q_native,
        k_native,
        q_native_scale,
        k_native_scale,
        q_global_scale,
        k_global_scale,
        softmax_scale
    );
    const c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "v509 experimental E5M2-dO native-score hybrid requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (clear_dq) {
        CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), stream));
    }
    if (clear_dkdv) {
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
        q_native,
        k_native,
        q_native_scale,
        k_native_scale,
        q_global_scale,
        k_global_scale,
        static_cast<float>(softmax_scale),
        stream
    );
}

void main_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor q_native,
    at::Tensor k_native,
    at::Tensor q_native_scale,
    at::Tensor k_native_scale,
    at::Tensor q_global_scale,
    at::Tensor k_global_scale,
    double softmax_scale
) {
    launch_checked(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        q_native,
        k_native,
        q_native_scale,
        k_native_scale,
        q_global_scale,
        k_global_scale,
        softmax_scale,
        false,
        false
    );
}

void backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor q_native,
    at::Tensor k_native,
    at::Tensor q_native_scale,
    at::Tensor k_native_scale,
    at::Tensor q_global_scale,
    at::Tensor k_global_scale,
    double softmax_scale
) {
    launch_checked(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        q_native,
        k_native,
        q_native_scale,
        k_native_scale,
        q_global_scale,
        k_global_scale,
        softmax_scale,
        true,
        true
    );
}

void backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precleared_dq_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor q_native,
    at::Tensor k_native,
    at::Tensor q_native_scale,
    at::Tensor k_native_scale,
    at::Tensor q_global_scale,
    at::Tensor k_global_scale,
    double softmax_scale
) {
    launch_checked(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        q_native,
        k_native,
        q_native_scale,
        k_native_scale,
        q_global_scale,
        k_global_scale,
        softmax_scale,
        false,
        true
    );
}

pybind11::dict metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.native_tk_d128_backward.v1";
    result["backend"] = "thunderkittens_sm100a";
    result["source_identity"] = kBatch == 1
        ? "v509_native_nvfp4_score_e4m3_qkv_e5m2_dout_b1_s4096_experimental_v1"
        : (kBatch == 2
            ? "v509_native_nvfp4_score_e4m3_qkv_e5m2_dout_b2_s4096_experimental_v1"
            : "v509_native_nvfp4_score_e4m3_qkv_e5m2_dout_b4_s4096_experimental_v1");
    result["source_file"] = __FILE__;
    result["experimental"] = true;
    result["production_dispatch_connected"] = false;
    result["dispatch"] = kBatch == 1
        ? "fail_closed_B1_S4096_only_no_fallback"
        : (kBatch == 2
            ? "fail_closed_B2_S4096_only_no_fallback"
            : "fail_closed_B4_S4096_only_no_fallback");
    // The separately compiled wrappers change the exact batch raster while
    // retaining the original __global__ symbol name for source continuity.
    result["selected_kernel"] =
        "v509::b1_native_nvfp4_score_e4m3_qkv_e5m2_dout_exact_s4096_kernel";
    result["score_qk_dtype"] = "float4_e2m1fn_x2";
    result["score_qk_layout"] = "BHSD_packed";
    result["score_scale_dtype"] = "float8_e4m3fn";
    result["score_scale_layout"] =
        "forward_row_K16_pages_Q_B_S128_Hx2_512_K_B_S64_Hkvx2_512";
    result["score_global_scale"] = "per_head_q_times_k";
    result["score_mma"] =
        "two_K64_mxf4nvf4_block_scale_scale_vec_4X";
    result["gradient_qkv_dtype"] = "float8_e4m3fn_represented_x4";
    result["dout_dtype"] = "float8_e5m2_represented_x4";
    result["dout_encode_scale"] = 4.0;
    result["dout_decode_scale"] = 0.25;
    result["mixed_mma_b_format_mask"] = 1024;
    result["score_internal_beta_divisor"] = 1.0;
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
    result["user_shared_storage_bytes"] = static_cast<int>(
        sizeof(candidate::shared_storage) +
        sizeof(candidate::native_score_shared_storage)
    );
    result["score_scale_tmem_alias"] = "dP_dQ_columns_0_15";
    result["score_schedule"] =
        "wait_dq_tmem_drained_then_native_score_wait_complete_then_dense_dp";
    result["caller_owned_output_api"] = true;
    result["main_requires_precleared_outputs"] = true;
    result["backward_out_clears_dq_dk_dv"] = true;
    result["precleared_dq_out_clears_dk_dv"] = true;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "main_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed",
        &main_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed
    );
    module.def(
        "backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed_out",
        &backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed_out
    );
    module.def(
        "backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precleared_dq_out",
        &backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precleared_dq_out
    );
    module.def("native_tk_d128_backward_metadata", &metadata);
}

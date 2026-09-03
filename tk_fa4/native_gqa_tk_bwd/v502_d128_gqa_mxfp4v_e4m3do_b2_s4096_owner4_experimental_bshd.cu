#include "v502_d128_gqa_mxfp4v_e4m3do_b2_s4096_owner4_experimental_bshd.cuh"

#include <array>
#include <cmath>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v502_d128_gqa_mxfp4v_e4m3do_b2_s4096_owner4_experimental_bshd;

constexpr long long kBatch = 2;
constexpr long long kSequence = 4096;
constexpr long long kScalePages = kSequence / candidate::kKeyTile;

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
            tensor.size(2) == heads &&
            tensor.size(3) == candidate::kDepth,
        name,
        " must be contiguous CUDA [2,4096,",
        heads,
        ",128]"
    );
}

void check_v_mxfp4(
    const at::Tensor &payload,
    const at::Tensor &scales
) {
    CHECK_INPUT(payload);
    CHECK_INPUT(scales);
    TORCH_CHECK(
        payload.scalar_type() == at::kByte && payload.dim() == 4 &&
            payload.size(0) == kBatch && payload.size(1) == kSequence &&
            payload.size(2) == candidate::kKvHeads &&
            payload.size(3) == candidate::kDepth / 2,
        "v_backward_mxfp4 must be contiguous CUDA byte "
        "[2,4096,8,64]"
    );
    TORCH_CHECK(
        scales.scalar_type() == at::kByte && scales.dim() == 4 &&
            scales.size(0) == kBatch && scales.size(1) == kScalePages &&
            scales.size(2) == candidate::kKvHeads &&
            scales.size(3) == 512,
        "v_backward_mxfp4_scales must be contiguous CUDA byte "
        "[2,32,8,512]"
    );
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(payload.data_ptr()) % 16 == 0 &&
            reinterpret_cast<std::uintptr_t>(scales.data_ptr()) % 16 == 0,
        "MXFP4 V payload and scale pages must be 16-byte aligned"
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
        " must be contiguous CUDA FP32 [2,32,1,4096]"
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
    const at::Tensor &v_backward_mxfp4,
    const at::Tensor &v_backward_mxfp4_scales,
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
    check_v_mxfp4(v_backward_mxfp4, v_backward_mxfp4_scales);
    check_bshd(dout, "dout", kE4m3, candidate::kQueryHeads);
    check_stats(lstat, "lstat");
    check_stats(dstat, "dstat");
    check_bshd(dq, "dq", at::kBFloat16, candidate::kQueryHeads);
    check_bshd(dk, "dk", at::kBFloat16, candidate::kKvHeads);
    check_bshd(dv, "dv", at::kBFloat16, candidate::kKvHeads);

    kittens::py::device_check(
        q,
        k,
        v_backward_mxfp4,
        v_backward_mxfp4_scales,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv
    );
    const float scale = static_cast<float>(softmax_scale);
    const float beta = scale / 16.0f;
    TORCH_CHECK(
        std::isfinite(softmax_scale) && softmax_scale > 0.0 &&
            std::isfinite(scale) && std::isfinite(beta) && beta > 0.0f,
        "softmax_scale must be finite, positive, and representable in FP32"
    );

    using named_tensor = std::pair<const at::Tensor *, const char *>;
    const std::array<named_tensor, 10> tensors{{
        {&q, "q"},
        {&k, "k"},
        {&v_backward_mxfp4, "v_backward_mxfp4"},
        {&v_backward_mxfp4_scales, "v_backward_mxfp4_scales"},
        {&dout, "dout"},
        {&lstat, "lstat"},
        {&dstat, "dstat"},
        {&dq, "dq"},
        {&dk, "dk"},
        {&dv, "dv"},
    }};
    for (int output = 7; output < 10; ++output) {
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
    at::Tensor v_backward_mxfp4,
    at::Tensor v_backward_mxfp4_scales,
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
        q,
        k,
        v_backward_mxfp4,
        v_backward_mxfp4_scales,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        softmax_scale
    );
    const c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "v502 experimental MXFP4-V backward requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (clear_outputs) {
        // Exact B2 has additive dQ and unique direct-overwrite dK/dV.
        CUDACHECK(cudaMemsetAsync(dq.data_ptr(), 0, dq.nbytes(), stream));
    }
    candidate::launch(
        q,
        k,
        v_backward_mxfp4,
        v_backward_mxfp4_scales,
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

void main_mxfp4v_e4m3do_bshd_precomputed(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v_backward_mxfp4,
    at::Tensor v_backward_mxfp4_scales,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    double softmax_scale
) {
    launch_checked(
        q,
        k,
        v_backward_mxfp4,
        v_backward_mxfp4_scales,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        softmax_scale,
        false
    );
}

void backward_mxfp4v_e4m3do_bshd_precomputed_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v_backward_mxfp4,
    at::Tensor v_backward_mxfp4_scales,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    double softmax_scale
) {
    launch_checked(
        q,
        k,
        v_backward_mxfp4,
        v_backward_mxfp4_scales,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        softmax_scale,
        true
    );
}

pybind11::dict metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.native_tk_d128_backward.experimental.v1";
    result["backend"] = "thunderkittens_sm100a";
    result["source_identity"] =
        "v502_d128_gqa_mxfp4v_e4m3do_b2_s4096_owner4_experimental_bshd_v3";
    result["source_file"] = __FILE__;
    result["experimental"] = true;
    result["production_data_abi_compatible"] = false;
    result["fail_closed"] = "B2_S4096_D128_GQA_only";
    result["batch_values"] = pybind11::make_tuple(2);
    result["sequence"] = kSequence;
    result["causal"] = true;
    result["threads"] = candidate::kThreads;
    result["query_heads"] = candidate::kQueryHeads;
    result["kv_heads"] = candidate::kKvHeads;
    result["head_dim"] = candidate::kDepth;
    result["q_k_dout_dtype"] = "float8_e4m3fn_x4_encoding";
    result["q_k_dout_layout"] = "BSHD_contiguous";
    result["v_dtype"] = "packed_mxfp4_e2m1_uint8";
    result["v_layout"] = "B,S,Hkv,D_over_2_row_major";
    result["v_shape"] = "[2,4096,8,64]";
    result["v_scale_dtype"] = "e8m0_bytes";
    result["v_scale_layout"] =
        "B,S_over_128,Hkv,physical_32x16_page";
    result["v_scale_shape"] = "[2,32,8,512]";
    result["uses_forward_feature_major_v_payload"] = false;
    result["requires_second_row_major_v_orientation"] = true;
    result["eliminates_backward_e4m3_v_publication"] = true;
    result["eliminates_all_duplicate_v_publication"] = false;
    result["v_shared_restage"] =
        "manual_packed_to_16u4_align16b_8_payload_bytes_then_8_gap";
    result["dp_opcode"] =
        "tcgen05_mma_cta_group_1_kind_mxf8f6f4_block_scale";
    result["dp_a_format"] = "E2M1_encoding_5";
    result["dp_b_format"] = "E4M3_encoding_0";
    result["dp_b_major"] = "K_major_for_ABt";
    result["dp_b_descriptor_chunk_stride"] = 1;
    result["dp_scale_format"] = "E8M0";
    result["dp_reduction_chunk"] = 32;
    result["dp_raw_to_x16"] = 2.0 / 3.0;
    result["dstat_abi"] = "-16*sum(O*dO)";
    result["dout_scale_page"] = "one_E8M0_byte_127";
    result["dv_route"] = "unchanged_v490_probability_times_e4m3_dout";
    result["score_scale_tmem_alias"] =
        "score_columns_384_415_serialized_by_score_consumed_and_dp_ready_"
        "including_head_boundary";
    result["expected_per_key_tile_v_global_bytes"] = 8192;
    result["expected_per_key_tile_v_scale_global_bytes"] = 512;
    result["caller_owned_output_api"] = true;
    result["backward_out_clears_outputs"] = true;
    result["backward_out_physical_clear_policy"] =
        "memset_dq_only_unique_direct_overwrite_dk_dv";
    result["output_dtype"] = "bfloat16";
    result["output_layout"] = "BSHD_contiguous";
    result["lstat_abi"] = "8-LSE*log2(e)";
    result["public_softmax_scale"] = "natural";
    result["internal_beta_divisor"] = 16.0;
    result["gradient_epilogue_scale"] = 1.0 / 256.0;
    result["shared_payload_bytes"] =
        static_cast<int>(sizeof(candidate::shared_storage) +
                         2 * sizeof(candidate::mx_scale_tile));
    result["compiled_static_shared_bytes"] = 168208;
    result["shared_memory_receipt"] =
        "payload_166912B_compiled_total_168208B_including_barriers";
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "main_mxfp4v_e4m3do_bshd_precomputed",
        &main_mxfp4v_e4m3do_bshd_precomputed
    );
    module.def(
        "backward_mxfp4v_e4m3do_bshd_precomputed_out",
        &backward_mxfp4v_e4m3do_bshd_precomputed_out
    );
    module.def("native_tk_d128_backward_metadata", &metadata);
}

#include "v433_d128_gqa_e4m3_head_fast_raster_production_bshd.cuh"

#include <array>
#include <cmath>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v433_d128_gqa_e4m3_head_fast_raster_production_bshd;

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
            tensor.size(3) == candidate::kDepth,
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
        "v433 tensors must share batch/sequence and gradient shapes"
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
        "v433 production-BSHD D128 GQA backward requires SM100"
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
        "v433_d128_gqa_e4m3_head_fast_raster_production_bshd_v1";
    result["source_file"] = __FILE__;
    result["topology"] =
        "split_qhead_cta_k128_q128_single_score_double_qdo_stats_"
        "early_score_release_early_dp_next_shared_p_shared_ds_role_"
        "owner_x32_single_full_bf16_gradient_tile_three_full_width_tma_"
        "head_fast_grid_x_key_tile_grid_y";
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
    result["head_dim"] = candidate::kDepth;
    result["batch_values"] = pybind11::make_tuple(1, 2);
    result["sequence"] = "dynamic_positive_multiple_of_128";
    result["causal"] = true;
    result["operand_dtype"] = "float8_e4m3fn";
    result["operand_layout"] = "BSHD_contiguous";
    result["encoding_scale"] = candidate::kOperandScale;
    result["lstat_abi"] = "8-LSE*log2(e)";
    result["dstat_abi"] = "-16*sum(O*dO)";
    result["stats_layout"] = "B,Hq,1,S_fp32_contiguous";
    result["public_softmax_scale"] = "natural";
    result["internal_beta_divisor"] = 16.0;
    result["probability_exp"] =
        "native_ex2_clamp_production_log2_pscaled_le_8";
    result["probability_payload"] = "shared_e4m3_coordinate_correct_256P";
    result["ds_payload"] = "separate_shared_e4m3_256dS";
    result["lossy_probability_alu"] = false;
    result["score_tmem_stages"] = 1;
    result["score_tmem_reuse_gate"] =
        "eight_warp_collective_after_second_half_tmem_load_before_ex2";
    result["query_smem_stages"] = candidate::kInputStages;
    result["dout_smem_stages"] = candidate::kInputStages;
    result["stats_smem_stages"] = candidate::kInputStages;
    result["user_shared_storage_bytes"] =
        static_cast<int>(sizeof(candidate::shared_storage));
    result["sm100_shared_capacity_bytes"] = 233472;
    result["sm100_optin_shared_per_block_bytes"] = 232448;
    result["authenticated_cute_shared_bytes"] = 150528;
    result["gradient_publication_stages"] = 1;
    result["gradient_publication_wait_policy"] =
        "single_full_tile_read_wait_0_count1_mbarrier_before_every_reuse_"
        "final_store_wait_0";
    result["steady_state_cta_barriers"] = 0;
    result["input_reuse_gate"] =
        "stage_specific_after_score_consumed_ds_ready_dk_dv_readers";
    result["stats_reuse_gate"] =
        "stage_specific_after_eight_probability_ds_consumers";
    result["shared_reuse_gates"] =
        "P_dV_plus_compute_dS_dS_dQ_plus_dK";
    result["dp_dq_alias_gate"] =
        "four_warp_collective_after_full_128x128_bf16_dq_evacuation";
    result["dp_next_schedule"] =
        "immediately_after_full_bf16_dq_evacuation_before_dq_tma_and_"
        "current_dk_dv_release_wait";
    result["tmem_map"] =
        "dp_dq_0_128_dk_128_256_dv_256_384_score_384_512";
    result["gradient_drain"] =
        "four_owner_warps_x32_immediate_bf16_single_full_shared_tile";
    result["gradient_tmem_load_opcode"] =
        "tcgen05.ld.sync.aligned.32x32b.x32.b32";
    result["gradient_tmem_load_wait"] =
        "immediately_after_every_raw_x32_load";
    result["gradient_intermediate_shared_dtype"] = "bfloat16";
    result["gradient_shared_shape"] = "128x128";
    result["gradient_shared_bytes"] =
        static_cast<int>(sizeof(candidate::gradient_full_tile));
    result["gradient_tma_descriptor"] =
        "full_128x128_bfloat16_BSHD_depth_axis";
    result["gradient_full_width_publication_sites"] = 3;
    result["gradient_epilogue_scale"] = 1.0 / 256.0;
    result["output_dtype"] = "bfloat16_additive";
    result["output_layout"] = "BSHD_contiguous";
    result["output_encoding_scale"] = candidate::kOperandScale;
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

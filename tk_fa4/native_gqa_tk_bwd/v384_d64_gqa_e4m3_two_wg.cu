#include "v384_d64_gqa_e4m3_two_wg.cuh"

#include <cmath>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::v384_d64_gqa_e4m3_two_wg;

void check_bhsd(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 4 &&
            tensor.size(0) > 0 && tensor.size(1) == heads &&
            tensor.size(2) >= 128 && tensor.size(2) % 128 == 0 &&
            tensor.size(3) == candidate::kDepth,
        name,
        " must be contiguous CUDA [B,",
        heads,
        ",S,64] with positive B and S a multiple of 128"
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
            tensor.size(1) == q.size(1) && tensor.size(2) == 1 &&
            tensor.size(3) == q.size(2),
        name,
        " must be contiguous CUDA FP32 [B,Hq,1,S]"
    );
}

void main_e4m3_bhsd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor l_aux,
    at::Tensor delta,
    at::Tensor dq_fp32,
    at::Tensor dk_fp32,
    at::Tensor dv_fp32,
    double softmax_scale
) {
    constexpr auto kE4m3 = at::ScalarType::Float8_e4m3fn;
    check_bhsd(q, "q", kE4m3, 32);
    check_bhsd(k, "k", kE4m3, 8);
    check_bhsd(v, "v", kE4m3, 8);
    check_bhsd(dout, "dout", kE4m3, 32);
    check_stats(l_aux, "l_aux", q);
    check_stats(delta, "delta", q);
    check_bhsd(dq_fp32, "dq_fp32", at::kFloat, 32);
    check_bhsd(dk_fp32, "dk_fp32", at::kFloat, 8);
    check_bhsd(dv_fp32, "dv_fp32", at::kFloat, 8);
    TORCH_CHECK(
        q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
            q.size(2) == k.size(2) && q.size(2) == v.size(2) &&
            q.sizes() == dout.sizes() && q.sizes() == dq_fp32.sizes() &&
            k.sizes() == dk_fp32.sizes() && k.sizes() == dv_fp32.sizes(),
        "v384 tensors must share batch/sequence and gradient shapes"
    );
    TORCH_CHECK(
        std::isfinite(softmax_scale) && softmax_scale > 0.0,
        "softmax_scale must be finite and positive"
    );
    kittens::py::device_check(
        q,
        k,
        v,
        dout,
        l_aux,
        delta,
        dq_fp32,
        dk_fp32,
        dv_fp32
    );
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "v384 D64 GQA E4M3 two-WG backward requires SM100"
    );
    const c10::cuda::CUDAGuard device_guard(q.device());
    candidate::launch_e4m3(
        q,
        k,
        v,
        dout,
        l_aux,
        delta,
        dq_fp32,
        dk_fp32,
        dv_fp32,
        static_cast<float>(softmax_scale),
        at::cuda::getCurrentCUDAStream().stream()
    );
}

pybind11::dict metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.native_tk_d64_backward.v1";
    result["source_identity"] = "v384_d64_gqa_e4m3_two_wg_v1";
    result["topology"] = "two_wg_query_parallel_k64x2";
    result["threads"] = candidate::kThreads;
    result["launch_min_blocks_per_sm"] = 2;
    result["tmem_columns_per_cta"] =
        kittens::tensor_allocator<2, 1>::cols;
    result["dynamic_smem_bytes"] = candidate::kDynamicSmemBytes;
    result["operand_dtype"] = "float8_e4m3fn";
    result["operand_layout"] = "BHSD_contiguous";
    result["output_dtype"] = "float32_accumulator";
    result["caller_zeros_accumulators"] = true;
    return result;
}

}  // namespace

PYBIND11_MODULE(_C_b300_gqa_tk_bwd_v384, module) {
    module.def(
        "main_e4m3_bhsd",
        &main_e4m3_bhsd,
        "V384 two-warpgroup D64 GQA E4M3 main backward (BHSD)"
    );
    module.def(
        "native_tk_d64_backward_metadata",
        &metadata,
        "V384 ABI and residency metadata"
    );
}

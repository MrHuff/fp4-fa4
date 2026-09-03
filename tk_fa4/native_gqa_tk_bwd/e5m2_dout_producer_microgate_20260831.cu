#include "e5m2_dout_producer_microgate_20260831.cuh"

#include <array>
#include <cstdint>
#include <limits>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::e5m2_dout_producer_microgate_20260831;

void check_matrix(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 2 &&
            tensor.size(0) > 0 &&
            tensor.size(0) <= std::numeric_limits<int>::max() &&
            tensor.size(1) == candidate::kDepth,
        name,
        " must be contiguous CUDA [positive_rows,128] with exact dtype"
    );
}

void check_vector(
    const at::Tensor &tensor,
    const char *name,
    int64_t rows
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat && tensor.dim() == 1 &&
            tensor.size(0) == rows,
        name,
        " must be contiguous CUDA FP32 [rows]"
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

void produce_out(
    at::Tensor dout_bf16,
    at::Tensor attention_output_bf16,
    at::Tensor dout_e5m2,
    at::Tensor dstat
) {
    check_matrix(dout_bf16, "dout_bf16", at::kBFloat16);
    check_matrix(
        attention_output_bf16,
        "attention_output_bf16",
        at::kBFloat16
    );
    check_matrix(
        dout_e5m2,
        "dout_e5m2",
        at::ScalarType::Float8_e5m2
    );
    check_vector(dstat, "dstat", dout_bf16.size(0));
    TORCH_CHECK(
        attention_output_bf16.sizes() == dout_bf16.sizes() &&
            dout_e5m2.sizes() == dout_bf16.sizes(),
        "all matrix shapes must match"
    );
    kittens::py::device_check(
        dout_bf16,
        attention_output_bf16,
        dout_e5m2,
        dstat
    );
    TORCH_CHECK(
        !byte_ranges_overlap(dout_e5m2, dout_bf16) &&
            !byte_ranges_overlap(dout_e5m2, attention_output_bf16) &&
            !byte_ranges_overlap(dstat, dout_bf16) &&
            !byte_ranges_overlap(dstat, attention_output_bf16) &&
            !byte_ranges_overlap(dstat, dout_e5m2),
        "caller-owned outputs must not overlap inputs or each other"
    );

    const c10::cuda::CUDAGuard device_guard(dout_bf16.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "standalone E5M2 dO producer microgate requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    candidate::launch(
        dout_bf16,
        attention_output_bf16,
        dout_e5m2,
        dstat,
        stream
    );
}

pybind11::dict metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.e5m2_dout_producer_microgate.v1";
    result["source_identity"] =
        "e5m2_dout_producer_microgate_20260831_v1";
    result["source_file"] = __FILE__;
    result["standalone_fail_closed"] = true;
    result["production_dispatch_enabled"] = false;
    result["input_dtype"] = "bfloat16";
    result["payload_dtype"] = "float8_e5m2";
    result["depth"] = candidate::kDepth;
    result["encode"] = "(BF16.float()*4).to(float8_e5m2)";
    result["encode_scale"] = candidate::kEncodeScale;
    result["logical_decode"] = "published_E5_bytes.float()*0.25";
    result["decode_scale"] = candidate::kDecodeScale;
    result["logical_dstat"] =
        "-16*sum(O*(published_E5_bytes.float()*0.25))";
    result["physical_dstat"] =
        "-4*sum(O*published_E5_bytes.float())";
    result["logical_dstat_scale"] = candidate::kLogicalDstatScale;
    result["physical_dstat_scale"] = candidate::kPhysicalDstatScale;
    result["dstat_source"] = "decoded_bytes_actually_published";
    result["rows_per_cta"] = candidate::kWarps;
    result["threads"] = candidate::kThreads;
    result["caller_owned_output_api"] = true;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("produce_out", &produce_out);
    module.def("metadata", &metadata);
}

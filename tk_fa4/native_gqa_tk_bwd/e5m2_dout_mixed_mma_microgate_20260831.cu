#include "e5m2_dout_mixed_mma_microgate_20260831.cuh"

#include <array>
#include <cstdint>

namespace {

namespace candidate =
    tkfa4::native_gqa_tk_bwd::e5m2_dout_mixed_mma_microgate_20260831;

void check_matrix(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 2 &&
            tensor.size(0) == candidate::kRows &&
            tensor.size(1) == candidate::kReduction,
        name,
        " must be contiguous CUDA [128,128] with the exact requested dtype"
    );
}

void check_arguments(
    const at::Tensor &a0,
    const at::Tensor &b0,
    const at::Tensor &a1,
    const at::Tensor &b1,
    const at::Tensor &output
) {
    check_matrix(a0, "a0", at::ScalarType::Float8_e4m3fn);
    check_matrix(b0, "b0", at::ScalarType::Float8_e5m2);
    check_matrix(a1, "a1", at::ScalarType::Float8_e4m3fn);
    check_matrix(b1, "b1", at::ScalarType::Float8_e5m2);
    check_matrix(output, "output", at::kFloat);
    kittens::py::device_check(a0, b0, a1, b1, output);
    TORCH_CHECK(
        output.data_ptr() != a0.data_ptr() &&
            output.data_ptr() != b0.data_ptr() &&
            output.data_ptr() != a1.data_ptr() &&
            output.data_ptr() != b1.data_ptr(),
        "output must not alias an input"
    );
}

template <candidate::operation Operation, bool Accumulate>
void launch_checked(
    at::Tensor a0,
    at::Tensor b0,
    at::Tensor a1,
    at::Tensor b1,
    at::Tensor output
) {
    check_arguments(a0, b0, a1, b1, output);
    const c10::cuda::CUDAGuard device_guard(a0.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "E4M3xE5M2 mixed-MMA microgate requires an SM100-class device"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    candidate::launch<Operation, Accumulate>(
        a0,
        b0,
        a1,
        b1,
        output,
        stream
    );
}

void dp_overwrite_out(
    at::Tensor a,
    at::Tensor b,
    at::Tensor output
) {
    launch_checked<candidate::operation::dp_abt, false>(
        a,
        b,
        a,
        b,
        output
    );
}

void dp_accumulate_out(
    at::Tensor a0,
    at::Tensor b0,
    at::Tensor a1,
    at::Tensor b1,
    at::Tensor output
) {
    launch_checked<candidate::operation::dp_abt, true>(
        a0,
        b0,
        a1,
        b1,
        output
    );
}

void dv_overwrite_out(
    at::Tensor a,
    at::Tensor b,
    at::Tensor output
) {
    launch_checked<candidate::operation::dv_ab, false>(
        a,
        b,
        a,
        b,
        output
    );
}

void dv_accumulate_out(
    at::Tensor a0,
    at::Tensor b0,
    at::Tensor a1,
    at::Tensor b1,
    at::Tensor output
) {
    launch_checked<candidate::operation::dv_ab, true>(
        a0,
        b0,
        a1,
        b1,
        output
    );
}

pybind11::dict metadata() {
    pybind11::dict result;
    result["schema"] = "tkfa4.e5m2_dout_mixed_mma_microgate.v1";
    result["source_identity"] =
        "e5m2_dout_mixed_mma_microgate_20260831_v1";
    result["source_file"] = __FILE__;
    result["production_dispatch_enabled"] = false;
    result["standalone_fail_closed"] = true;
    result["shape"] = pybind11::make_tuple(128, 128, 128);
    result["a_dtype"] = "float8_e4m3fn";
    result["b_dtype"] = "float8_e5m2";
    result["output_dtype"] = "float32";
    result["dp_semantics"] = "A_E4M3_times_B_E5M2_transpose";
    result["dv_semantics"] = "A_E4M3_times_B_E5M2";
    result["accumulator_modes"] = pybind11::make_tuple(
        "overwrite",
        "accumulate_second_product"
    );
    result["descriptor_patch"] = "E4M3xE4M3_descriptor_or_0x400";
    result["e5m2_b_format_mask"] =
        pybind11::int_(candidate::kE5m2BFormatMask);
    result["dp_e4e4_descriptor"] =
        pybind11::int_(candidate::kDpE4m3E4m3Instruction);
    result["dp_e4e5_descriptor"] =
        pybind11::int_(candidate::kDpE4m3E5m2Instruction);
    result["dv_e4e4_descriptor"] =
        pybind11::int_(candidate::kDvE4m3E4m3Instruction);
    result["dv_e4e5_descriptor"] =
        pybind11::int_(candidate::kDvE4m3E5m2Instruction);
    result["threads"] = candidate::kThreads;
    result["shared_storage_bytes"] =
        static_cast<int>(sizeof(candidate::shared_storage));
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("dp_overwrite_out", &dp_overwrite_out);
    module.def("dp_accumulate_out", &dp_accumulate_out);
    module.def("dv_overwrite_out", &dv_overwrite_out);
    module.def("dv_accumulate_out", &dv_accumulate_out);
    module.def("metadata", &metadata);
}

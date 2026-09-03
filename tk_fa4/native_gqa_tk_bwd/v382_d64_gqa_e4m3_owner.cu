#include "v382_d64_gqa_e4m3_owner.cuh"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

namespace owner =
    tkfa4::native_gqa_tk_bwd::v382_d64_e4m3_owner;

void check_operand(
    const at::Tensor &tensor,
    const char *name,
    int heads,
    int64_t batch_size
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            tensor.dim() == 4 &&
            tensor.size(0) == batch_size &&
            tensor.size(1) == owner::kSequence &&
            tensor.size(2) == heads &&
            tensor.size(3) == owner::kDepth,
        name,
        " must be contiguous CUDA float8_e4m3fn [B,4096,",
        heads,
        ",64] (BSHD)"
    );
}

void check_stats(
    const at::Tensor &tensor,
    const char *name,
    int64_t batch_size
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat && tensor.dim() == 4 &&
            tensor.size(0) == batch_size &&
            tensor.size(1) == owner::kQueryHeads &&
            tensor.size(2) == 1 &&
            tensor.size(3) == owner::kSequence,
        name,
        " must be contiguous CUDA FP32 [B,32,1,4096]"
    );
}

void check_bshd_output(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    int heads,
    int64_t batch_size
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 4 &&
            tensor.size(0) == batch_size &&
            tensor.size(1) == owner::kSequence &&
            tensor.size(2) == heads &&
            tensor.size(3) == owner::kDepth,
        name,
        " has the wrong dtype or BSHD shape"
    );
}

void check_common(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lstat,
    const at::Tensor &dstat,
    double softmax_scale
) {
    TORCH_CHECK(q.dim() == 4, "q must be rank four");
    const int64_t batch_size = q.size(0);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(
        batch_size <= 65535,
        "batch exceeds CUDA grid.z capacity"
    );
    check_operand(q, "q", owner::kQueryHeads, batch_size);
    check_operand(k, "k", owner::kKvHeads, batch_size);
    check_operand(v, "v", owner::kKvHeads, batch_size);
    check_operand(dout, "dout", owner::kQueryHeads, batch_size);
    check_stats(lstat, "lstat", batch_size);
    check_stats(dstat, "dstat", batch_size);
    kittens::py::device_check(q, k, v, dout, lstat, dstat);
    TORCH_CHECK(
        std::isfinite(softmax_scale) && softmax_scale > 0.0,
        "softmax_scale must be finite and positive"
    );
}

bool byte_ranges_overlap(
    const at::Tensor &left,
    const at::Tensor &right
) {
    const auto left_begin = reinterpret_cast<std::uintptr_t>(
        left.data_ptr()
    );
    const auto right_begin = reinterpret_cast<std::uintptr_t>(
        right.data_ptr()
    );
    const auto left_end = left_begin + left.nbytes();
    const auto right_end = right_begin + right.nbytes();
    return left_begin < right_end && right_begin < left_end;
}

void check_mutable_buffers_do_not_overlap(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lstat,
    const at::Tensor &dstat,
    const at::Tensor &dq_accum,
    const at::Tensor &dk_partial,
    const at::Tensor &dv_partial,
    const at::Tensor &dq,
    const at::Tensor &dk,
    const at::Tensor &dv
) {
    using named_tensor = std::pair<const at::Tensor *, const char *>;
    const std::array<named_tensor, 12> tensors{{
        {&q, "q"},
        {&k, "k"},
        {&v, "v"},
        {&dout, "dout"},
        {&lstat, "lstat"},
        {&dstat, "dstat"},
        {&dq_accum, "dq_accum"},
        {&dk_partial, "dk_partial"},
        {&dv_partial, "dv_partial"},
        {&dq, "dq"},
        {&dk, "dk"},
        {&dv, "dv"},
    }};
    constexpr int kFirstMutable = 6;
    for (int i = kFirstMutable; i < static_cast<int>(tensors.size()); ++i) {
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

void backward_e4m3_precomputed_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    at::Tensor dq_accum,
    at::Tensor dk_partial,
    at::Tensor dv_partial,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    double softmax_scale
) {
    check_common(q, k, v, dout, lstat, dstat, softmax_scale);
    const int64_t batch_size = q.size(0);
    check_bshd_output(
        dq_accum,
        "dq_accum",
        at::kFloat,
        owner::kQueryHeads,
        batch_size
    );
    check_bshd_output(
        dk_partial,
        "dk_partial",
        at::kFloat,
        owner::kQueryHeads,
        batch_size
    );
    check_bshd_output(
        dv_partial,
        "dv_partial",
        at::kFloat,
        owner::kQueryHeads,
        batch_size
    );
    check_bshd_output(
        dq,
        "dq",
        at::kBFloat16,
        owner::kQueryHeads,
        batch_size
    );
    check_bshd_output(
        dk,
        "dk",
        at::kBFloat16,
        owner::kKvHeads,
        batch_size
    );
    check_bshd_output(
        dv,
        "dv",
        at::kBFloat16,
        owner::kKvHeads,
        batch_size
    );
    kittens::py::device_check(
        q,
        dq_accum,
        dk_partial,
        dv_partial,
        dq,
        dk,
        dv
    );
    check_mutable_buffers_do_not_overlap(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq_accum,
        dk_partial,
        dv_partial,
        dq,
        dk,
        dv
    );

    const c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "V382-style D64 GQA E4M3 owner backward requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CUDACHECK(cudaMemsetAsync(
        dq_accum.data_ptr(),
        0,
        dq_accum.nbytes(),
        stream
    ));

    const float beta = static_cast<float>(softmax_scale / 16.0);
    TORCH_CHECK(
        std::isfinite(beta) && beta > 0.0f,
        "softmax_scale is not representable by the kernel's FP32 beta"
    );
    const owner::main_globals globals{
        kittens::py::tensor_to_gl<owner::main_globals::q_gl>(q),
        kittens::py::tensor_to_gl<owner::main_globals::k_gl>(k),
        kittens::py::tensor_to_gl<owner::main_globals::v_gl>(v),
        kittens::py::tensor_to_gl<owner::main_globals::dout_gl>(dout),
        kittens::py::tensor_to_gl<owner::main_globals::partial_gl>(
            dk_partial
        ),
        kittens::py::tensor_to_gl<owner::main_globals::partial_gl>(
            dv_partial
        ),
        kittens::py::tensor_to_gl<owner::main_globals::dq_gl>(dq_accum),
        reinterpret_cast<const float *>(lstat.data_ptr()),
        reinterpret_cast<const float *>(dstat.data_ptr()),
        beta,
        beta * tkfa4::kLog2E,
    };

    owner::launch_owner_e4m3(
        globals,
        static_cast<int>(batch_size),
        stream
    );
    owner::launch_finalize(
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv,
        static_cast<int>(batch_size),
        stream
    );
}

std::vector<at::Tensor> backward_e4m3_precomputed(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lstat,
    at::Tensor dstat,
    double softmax_scale
) {
    check_common(q, k, v, dout, lstat, dstat, softmax_scale);
    at::Tensor dq_accum = at::empty(
        q.sizes(),
        q.options().dtype(at::kFloat)
    );
    at::Tensor dk_partial = at::empty(
        q.sizes(),
        q.options().dtype(at::kFloat)
    );
    at::Tensor dv_partial = at::empty(
        q.sizes(),
        q.options().dtype(at::kFloat)
    );
    at::Tensor dq = at::empty(
        q.sizes(),
        q.options().dtype(at::kBFloat16)
    );
    at::Tensor dk = at::empty(
        k.sizes(),
        k.options().dtype(at::kBFloat16)
    );
    at::Tensor dv = at::empty(
        v.sizes(),
        v.options().dtype(at::kBFloat16)
    );
    backward_e4m3_precomputed_out(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq_accum,
        dk_partial,
        dv_partial,
        dq,
        dk,
        dv,
        softmax_scale
    );
    return {dq, dk, dv};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "backward_e4m3_precomputed",
        &backward_e4m3_precomputed,
        "V382 D64 GQA E4M3(x4) backward with encoded 4*dX outputs (BSHD)"
    );
    module.def(
        "backward_e4m3_precomputed_out",
        &backward_e4m3_precomputed_out,
        "V382 D64 GQA E4M3(x4) backward into encoded caller-owned outputs"
    );
}

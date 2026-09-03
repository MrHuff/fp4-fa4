#include "v382_d64_gqa_owner.cuh"
#include "../deprecated/fa4_bwd_preprocess.cuh"

#include <cmath>
#include <vector>

namespace {

namespace owner = tkfa4::native_gqa_tk_bwd::v382_d64_owner;

void check_operand(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 4 &&
            tensor.size(0) == owner::kBatch &&
            tensor.size(1) == heads &&
            tensor.size(2) == owner::kSequence &&
            tensor.size(3) == owner::kDepth,
        name,
        " must be contiguous CUDA ",
        dtype,
        " [1,",
        heads,
        ",4096,64]"
    );
}

void check_stats(
    const at::Tensor &tensor,
    const char *name
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat && tensor.dim() == 4 &&
            tensor.size(0) == owner::kBatch &&
            tensor.size(1) == owner::kQueryHeads &&
            tensor.size(2) == 1 &&
            tensor.size(3) == owner::kSequence,
        name,
        " must be contiguous CUDA FP32 [1,32,1,4096]"
    );
}

void check_common(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lse_log2,
    const at::Tensor &delta,
    double softmax_scale
) {
    check_operand(q, "q", at::kBFloat16, owner::kQueryHeads);
    check_operand(k, "k", at::kBFloat16, owner::kKvHeads);
    check_operand(v, "v", at::kBFloat16, owner::kKvHeads);
    check_operand(dout, "dout", at::kBFloat16, owner::kQueryHeads);
    check_stats(lse_log2, "lse_log2");
    check_stats(delta, "delta");
    kittens::py::device_check(q, k, v, dout, lse_log2, delta);
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "V382-style D64 GQA owner control requires SM100"
    );
    TORCH_CHECK(
        std::isfinite(softmax_scale) && softmax_scale > 0.0,
        "softmax_scale must be finite and positive"
    );
}

std::vector<at::Tensor> backward_bf16_precomputed_impl(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lse_log2,
    at::Tensor delta,
    double softmax_scale,
    bool return_partials
) {
    check_common(q, k, v, dout, lse_log2, delta, softmax_scale);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    at::Tensor dq_accum = at::zeros(
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
    at::Tensor dq = at::empty(q.sizes(), q.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());

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
        reinterpret_cast<const float *>(lse_log2.data_ptr()),
        reinterpret_cast<const float *>(delta.data_ptr()),
        static_cast<float>(softmax_scale),
        static_cast<float>(softmax_scale) * tkfa4::kLog2E,
    };

    owner::launch_owner_bf16(globals, stream);
    owner::launch_finalize(
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv,
        stream
    );
    if (return_partials) {
        return {dq, dk, dv, dk_partial, dv_partial};
    }
    return {dq, dk, dv};
}

std::vector<at::Tensor> backward_bf16_precomputed(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lse_log2,
    at::Tensor delta,
    double softmax_scale
) {
    return backward_bf16_precomputed_impl(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        softmax_scale,
        false
    );
}

std::vector<at::Tensor> backward_bf16_precomputed_debug(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lse_log2,
    at::Tensor delta,
    double softmax_scale
) {
    return backward_bf16_precomputed_impl(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        softmax_scale,
        true
    );
}

std::vector<at::Tensor> backward_bf16(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse_log2,
    at::Tensor dout,
    double softmax_scale
) {
    check_operand(out, "out", at::kBFloat16, owner::kQueryHeads);
    TORCH_CHECK(out.sizes() == dout.sizes(), "out and dout must match");
    at::Tensor delta = at::empty(
        {owner::kBatch, owner::kQueryHeads, 1, owner::kSequence},
        q.options().dtype(at::kFloat)
    );
    tkfa4::bwd::launch_preprocess<owner::kDepth>(out, dout, delta);
    return backward_bf16_precomputed(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        softmax_scale
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "backward_bf16",
        &backward_bf16,
        "V382-style cluster-2 D64 GQA BF16 backward control"
    );
    module.def(
        "backward_bf16_precomputed",
        &backward_bf16_precomputed,
        "V382-style D64 GQA BF16 backward with precomputed delta"
    );
    module.def(
        "backward_bf16_precomputed_debug",
        &backward_bf16_precomputed_debug,
        "V382-style D64 GQA BF16 backward with raw Hq partials"
    );
}

#include "v382_d64_gqa_e4m3_hkv2_register_pd.cuh"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

namespace owner =
    tkfa4::native_gqa_tk_bwd::v382_d64_e4m3_hkv2_register_pd;

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

void check_outputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dq_accum,
    const at::Tensor &dk_partial,
    const at::Tensor &dv_partial,
    const at::Tensor &dq,
    const at::Tensor &dk,
    const at::Tensor &dv
) {
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
        owner::kPartialHeads,
        batch_size
    );
    check_bshd_output(
        dv_partial,
        "dv_partial",
        at::kFloat,
        owner::kPartialHeads,
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
    TORCH_CHECK(dq.sizes() == q.sizes(), "dq and q must match");
    TORCH_CHECK(dk.sizes() == k.sizes(), "dk and k must match");
    TORCH_CHECK(dv.sizes() == v.sizes(), "dv and v must match");
    kittens::py::device_check(
        q,
        k,
        v,
        dq_accum,
        dk_partial,
        dv_partial,
        dq,
        dk,
        dv
    );
}

owner::main_globals make_globals(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lstat,
    const at::Tensor &dstat,
    at::Tensor &dq_accum,
    at::Tensor &dk_partial,
    at::Tensor &dv_partial,
    double softmax_scale
) {
    const float beta = static_cast<float>(softmax_scale / 16.0);
    const float beta_log2e = beta * tkfa4::kLog2E;
    TORCH_CHECK(
        std::isfinite(beta) && beta > 0.0f && std::isfinite(beta_log2e),
        "softmax_scale is not representable by the kernel's FP32 beta/log2e"
    );
    return owner::main_globals{
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
        beta_log2e,
    };
}

void launch_precomputed_out_unchecked(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lstat,
    const at::Tensor &dstat,
    at::Tensor &dq_accum,
    at::Tensor &dk_partial,
    at::Tensor &dv_partial,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    double softmax_scale,
    cudaStream_t stream
) {
    CUDACHECK(cudaMemsetAsync(
        dq_accum.data_ptr(),
        0,
        dq_accum.nbytes(),
        stream
    ));
    const owner::main_globals globals = make_globals(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq_accum,
        dk_partial,
        dv_partial,
        softmax_scale
    );
    owner::launch_hkv2_register_pd_e4m3(
        globals,
        static_cast<int>(q.size(0)),
        stream
    );
    owner::launch_finalize(
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv,
        stream
    );
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
    check_outputs(
        q,
        k,
        v,
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
        "V382 D64 GQA E4M3 owner-major two-head register-P/dS "
        "backward requires SM100"
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_precomputed_out_unchecked(
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
        softmax_scale,
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
    const c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "V382 D64 GQA E4M3 owner-major two-head register-P/dS "
        "backward requires SM100"
    );
    at::Tensor dq_accum = at::empty(
        q.sizes(),
        q.options().dtype(at::kFloat)
    );
    const std::vector<int64_t> partial_sizes{
        q.size(0),
        owner::kSequence,
        owner::kPartialHeads,
        owner::kDepth,
    };
    at::Tensor dk_partial = at::empty(
        partial_sizes,
        q.options().dtype(at::kFloat)
    );
    at::Tensor dv_partial = at::empty(
        partial_sizes,
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
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_precomputed_out_unchecked(
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
        softmax_scale,
        stream
    );
    return {dq, dk, dv};
}

pybind11::dict native_tk_d64_backward_metadata() {
    pybind11::dict metadata;
    metadata["schema"] = "tkfa4.native_tk_d64_backward.v1";
    metadata["backend"] = "thunderkittens_sm100a";
    metadata["topology"] = "v382_owner_major_hkv2_register_p_ds";
    metadata["source_identity"] =
        "v382_d64_gqa_e4m3_hkv2_register_pd_v1";
    metadata["source_file"] = __FILE__;
    metadata["sequence"] = owner::kSequence;
    metadata["query_heads"] = owner::kQueryHeads;
    metadata["kv_heads"] = owner::kKvHeads;
    metadata["head_dim"] = owner::kDepth;
    metadata["heads_per_owner"] = owner::kHeadsPerOwner;
    metadata["partial_heads"] = owner::kPartialHeads;
    metadata["operand_dtype"] = "float8_e4m3fn";
    metadata["operand_layout"] = "BSHD_contiguous";
    metadata["encoding_scale"] = 4.0;
    metadata["lstat_abi"] = "8-LSE*log2(e)";
    metadata["dstat_abi"] = "-16*sum(O*dO)";
    metadata["stats_layout"] = "B,Hq,1,S_fp32_contiguous";
    metadata["public_softmax_scale"] = "natural";
    metadata["internal_beta_divisor"] = 16.0;
    metadata["gradient_epilogue_scale"] =
        static_cast<double>(owner::kGradientOutputScale);
    metadata["output_dtype"] = "bfloat16";
    metadata["output_encoding_scale"] = 4.0;
    metadata["caller_owned_output_api"] = true;
    return metadata;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "backward_e4m3_precomputed",
        &backward_e4m3_precomputed,
        "V382 D64 GQA E4M3(x4) Hkv2 register-P/dS backward (BSHD)"
    );
    module.def(
        "backward_e4m3_precomputed_out",
        &backward_e4m3_precomputed_out,
        "Caller-owned E4M3(x4) Hkv2 register-P/dS backward"
    );
    module.def(
        "native_tk_d64_backward_metadata",
        &native_tk_d64_backward_metadata,
        "Read-only ABI and implementation provenance for this backward"
    );
}

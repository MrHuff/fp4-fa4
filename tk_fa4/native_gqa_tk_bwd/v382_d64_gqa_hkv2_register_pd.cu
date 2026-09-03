#include "v382_d64_gqa_hkv2_register_pd.cuh"

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

namespace owner =
    tkfa4::native_gqa_tk_bwd::v382_d64_hkv2_register_pd;

void check_operand_bshd(
    const at::Tensor &tensor,
    const char *name,
    at::ScalarType dtype,
    int heads
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == dtype && tensor.dim() == 4 &&
            tensor.size(0) > 0 && tensor.size(1) == owner::kSequence &&
            tensor.size(2) == heads && tensor.size(3) == owner::kDepth,
        name,
        " must be contiguous CUDA ",
        dtype,
        " [B,4096,",
        heads,
        ",64]"
    );
    TORCH_CHECK(
        tensor.size(0) <= 65535,
        name,
        " batch exceeds CUDA grid.z capacity"
    );
}

void check_stats(
    const at::Tensor &tensor,
    const char *name,
    int64_t batch
) {
    CHECK_INPUT(tensor);
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat && tensor.dim() == 4 &&
            tensor.size(0) == batch &&
            tensor.size(1) == owner::kQueryHeads &&
            tensor.size(2) == 1 &&
            tensor.size(3) == owner::kSequence,
        name,
        " must be contiguous CUDA FP32 [B,32,1,4096]"
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
    TORCH_CHECK(q.dim() == 4, "q must be rank four");
    const int64_t batch_size = q.size(0);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(
        batch_size <= 65535,
        "batch exceeds CUDA grid.z capacity"
    );
    check_operand_bshd(q, "q", at::kBFloat16, owner::kQueryHeads);
    check_operand_bshd(k, "k", at::kBFloat16, owner::kKvHeads);
    check_operand_bshd(v, "v", at::kBFloat16, owner::kKvHeads);
    check_operand_bshd(
        dout,
        "dout",
        at::kBFloat16,
        owner::kQueryHeads
    );
    TORCH_CHECK(
        q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
            q.size(0) == dout.size(0),
        "q/k/v/dout batch dimensions must match"
    );
    check_stats(lse_log2, "lse_log2", q.size(0));
    check_stats(delta, "delta", q.size(0));
    kittens::py::device_check(q, k, v, dout, lse_log2, delta);
    TORCH_CHECK(
        std::isfinite(softmax_scale) && softmax_scale > 0.0,
        "softmax_scale must be finite and positive"
    );
    const float scale_fp32 = static_cast<float>(softmax_scale);
    TORCH_CHECK(
        std::isfinite(scale_fp32) && scale_fp32 > 0.0f &&
            std::isfinite(scale_fp32 * tkfa4::kLog2E),
        "softmax_scale is not representable by the kernel's FP32 scales"
    );
}

void check_sm100_device() {
    TORCH_CHECK(
        tkfa4::is_sm100_device(),
        "V382 D64 GQA owner-major two-head register-P/dP backward requires SM100"
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

__global__ __launch_bounds__(256, 2)
void preprocess_bshd_kernel(
    const __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ dout,
    float *__restrict__ delta,
    int64_t rows
) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 8 + warp;
    if (row >= rows) {
        return;
    }
    const int64_t input_base = row * owner::kDepth;
    float sum =
        __bfloat162float(out[input_base + lane]) *
            __bfloat162float(dout[input_base + lane]) +
        __bfloat162float(out[input_base + lane + 32]) *
            __bfloat162float(dout[input_base + lane + 32]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        const int query_head = static_cast<int>(row % owner::kQueryHeads);
        const int64_t batch_sequence = row / owner::kQueryHeads;
        const int sequence =
            static_cast<int>(batch_sequence % owner::kSequence);
        const int64_t batch = batch_sequence / owner::kSequence;
        const int64_t output_index =
            (batch * owner::kQueryHeads + query_head) * owner::kSequence +
            sequence;
        delta[output_index] = sum;
    }
}

void launch_preprocess_bshd(
    const at::Tensor &out,
    const at::Tensor &dout,
    at::Tensor &delta,
    cudaStream_t stream
) {
    constexpr int kThreads = 256;
    constexpr int kRowsPerBlock = kThreads / 32;
    const int64_t rows =
        out.size(0) * owner::kSequence * owner::kQueryHeads;
    preprocess_bshd_kernel<<<
        static_cast<unsigned int>(
            (rows + kRowsPerBlock - 1) / kRowsPerBlock
        ),
        kThreads,
        0,
        stream
    >>>(
        reinterpret_cast<const __nv_bfloat16 *>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(dout.data_ptr()),
        reinterpret_cast<float *>(delta.data_ptr()),
        rows
    );
    CUDACHECK(cudaGetLastError());
}

void check_outputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dq_accum,
    const at::Tensor &dq,
    const at::Tensor &dk_partial,
    const at::Tensor &dv_partial,
    const at::Tensor &dk,
    const at::Tensor &dv
) {
    CHECK_INPUT(dq_accum);
    TORCH_CHECK(
        dq_accum.scalar_type() == at::kFloat &&
            dq_accum.sizes() == q.sizes(),
        "dq_accum must be contiguous CUDA FP32 with q's shape"
    );
    check_operand_bshd(dq, "dq", at::kBFloat16, owner::kQueryHeads);
    check_operand_bshd(dk, "dk", at::kBFloat16, owner::kKvHeads);
    check_operand_bshd(dv, "dv", at::kBFloat16, owner::kKvHeads);
    TORCH_CHECK(dq.sizes() == q.sizes(), "dq and q must match");
    TORCH_CHECK(dk.sizes() == k.sizes(), "dk and k must match");
    TORCH_CHECK(dv.sizes() == v.sizes(), "dv and v must match");

    const std::vector<int64_t> partial_sizes{
        q.size(0),
        owner::kSequence,
        owner::kPartialHeads,
        owner::kDepth,
    };
    CHECK_INPUT(dk_partial);
    CHECK_INPUT(dv_partial);
    TORCH_CHECK(
        dk_partial.scalar_type() == at::kFloat &&
            dk_partial.sizes() == partial_sizes,
        "dk_partial must be contiguous CUDA FP32 [B,4096,16,64]"
    );
    TORCH_CHECK(
        dv_partial.scalar_type() == at::kFloat &&
            dv_partial.sizes() == partial_sizes,
        "dv_partial must be contiguous CUDA FP32 [B,4096,16,64]"
    );
    kittens::py::device_check(
        q,
        k,
        v,
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv
    );
}

void check_mutable_buffers_do_not_overlap(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lse_log2,
    const at::Tensor &delta,
    const at::Tensor &dq_accum,
    const at::Tensor &dq,
    const at::Tensor &dk_partial,
    const at::Tensor &dv_partial,
    const at::Tensor &dk,
    const at::Tensor &dv
) {
    using named_tensor = std::pair<const at::Tensor *, const char *>;
    const std::array<named_tensor, 12> tensors{{
        {&q, "q"},
        {&k, "k"},
        {&v, "v"},
        {&dout, "dout"},
        {&lse_log2, "lse_log2"},
        {&delta, "delta"},
        {&dq_accum, "dq_accum"},
        {&dq, "dq"},
        {&dk_partial, "dk_partial"},
        {&dv_partial, "dv_partial"},
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

owner::partial_globals make_globals(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lse_log2,
    const at::Tensor &delta,
    at::Tensor &dq_accum,
    at::Tensor &dk_partial,
    at::Tensor &dv_partial,
    double softmax_scale
) {
    return owner::partial_globals{
        kittens::py::tensor_to_gl<owner::partial_globals::q_gl>(q),
        kittens::py::tensor_to_gl<owner::partial_globals::k_gl>(k),
        kittens::py::tensor_to_gl<owner::partial_globals::v_gl>(v),
        kittens::py::tensor_to_gl<owner::partial_globals::dout_gl>(dout),
        kittens::py::tensor_to_gl<owner::partial_globals::partial_gl>(
            dk_partial
        ),
        kittens::py::tensor_to_gl<owner::partial_globals::partial_gl>(
            dv_partial
        ),
        kittens::py::tensor_to_gl<owner::partial_globals::dq_gl>(dq_accum),
        reinterpret_cast<const float *>(lse_log2.data_ptr()),
        reinterpret_cast<const float *>(delta.data_ptr()),
        static_cast<float>(softmax_scale),
        static_cast<float>(softmax_scale) * tkfa4::kLog2E,
        static_cast<int>(q.size(0)),
    };
}

void launch_precomputed_out_unchecked(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dout,
    const at::Tensor &lse_log2,
    const at::Tensor &delta,
    double softmax_scale,
    at::Tensor &dq_accum,
    at::Tensor &dq,
    at::Tensor &dk_partial,
    at::Tensor &dv_partial,
    at::Tensor &dk,
    at::Tensor &dv,
    cudaStream_t stream
) {
    CUDACHECK(cudaMemsetAsync(
        dq_accum.data_ptr(),
        0,
        static_cast<size_t>(dq_accum.numel()) * sizeof(float),
        stream
    ));
    const owner::partial_globals globals = make_globals(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        dq_accum,
        dk_partial,
        dv_partial,
        softmax_scale
    );
    owner::launch_hkv2_register_pd_bf16(globals, stream);
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

void backward_bf16_precomputed_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lse_log2,
    at::Tensor delta,
    double softmax_scale,
    at::Tensor dq_accum,
    at::Tensor dq,
    at::Tensor dk_partial,
    at::Tensor dv_partial,
    at::Tensor dk,
    at::Tensor dv
) {
    check_common(q, k, v, dout, lse_log2, delta, softmax_scale);
    check_outputs(
        q,
        k,
        v,
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv
    );
    check_mutable_buffers_do_not_overlap(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv
    );
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_sm100_device();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_precomputed_out_unchecked(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        softmax_scale,
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv,
        stream
    );
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
    check_common(q, k, v, dout, lse_log2, delta, softmax_scale);
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_sm100_device();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

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
    at::Tensor dq = at::empty(q.sizes(), q.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());

    launch_precomputed_out_unchecked(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        softmax_scale,
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv,
        stream
    );
    return {dq, dk, dv};
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
    check_operand_bshd(
        out,
        "out",
        at::kBFloat16,
        owner::kQueryHeads
    );
    TORCH_CHECK(out.sizes() == q.sizes(), "out and q must match");
    TORCH_CHECK(out.sizes() == dout.sizes(), "out and dout must match");
    kittens::py::device_check(q, out, dout);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    at::Tensor delta = at::empty(
        {q.size(0), owner::kQueryHeads, 1, owner::kSequence},
        q.options().dtype(at::kFloat)
    );
    check_common(q, k, v, dout, lse_log2, delta, softmax_scale);
    check_sm100_device();
    launch_preprocess_bshd(out, dout, delta, stream);
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
        "V382 D64 GQA owner-major two-head register-P/dP backward"
    );
    module.def(
        "backward_bf16_precomputed",
        &backward_bf16_precomputed,
        "Owner-major two-head register-P/dP backward with delta"
    );
    module.def(
        "backward_bf16_precomputed_out",
        &backward_bf16_precomputed_out,
        "Caller-owned owner-major two-head register-P/dP backward"
    );
}

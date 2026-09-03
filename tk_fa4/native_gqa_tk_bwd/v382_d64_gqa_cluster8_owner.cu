#include "v382_d64_gqa_cluster8_owner.cuh"

#include <cmath>
#include <vector>

namespace {

namespace owner =
    tkfa4::native_gqa_tk_bwd::v382_d64_cluster8_owner;

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
        tkfa4::is_sm100_device(),
        "V382 D64 GQA cluster-8 backward requires SM100"
    );
    TORCH_CHECK(
        std::isfinite(softmax_scale) && softmax_scale > 0.0,
        "softmax_scale must be finite and positive"
    );
}

void check_outputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dq_accum,
    const at::Tensor &dq,
    const at::Tensor &dk,
    const at::Tensor &dv
) {
    check_operand_bshd(
        dq_accum,
        "dq_accum",
        at::kFloat,
        owner::kQueryHeads
    );
    check_operand_bshd(dq, "dq", at::kBFloat16, owner::kQueryHeads);
    check_operand_bshd(dk, "dk", at::kBFloat16, owner::kKvHeads);
    check_operand_bshd(dv, "dv", at::kBFloat16, owner::kKvHeads);
    TORCH_CHECK(dq_accum.sizes() == q.sizes(), "dq_accum must match q");
    TORCH_CHECK(dq.sizes() == q.sizes(), "dq must match q");
    TORCH_CHECK(dk.sizes() == k.sizes(), "dk must match k");
    TORCH_CHECK(dv.sizes() == v.sizes(), "dv must match v");
    kittens::py::device_check(q, k, v, dq_accum, dq, dk, dv);
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

void backward_bf16_precomputed_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    at::Tensor lse_log2,
    at::Tensor delta,
    at::Tensor dq_accum,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    double softmax_scale
) {
    check_common(q, k, v, dout, lse_log2, delta, softmax_scale);
    check_outputs(q, k, v, dq_accum, dq, dk, dv);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CUDACHECK(cudaMemsetAsync(
        dq_accum.data_ptr(),
        0,
        dq_accum.nbytes(),
        stream
    ));

    const owner::cluster8_globals globals{
        kittens::py::tensor_to_gl<owner::cluster8_globals::q_gl>(q),
        kittens::py::tensor_to_gl<owner::cluster8_globals::k_gl>(k),
        kittens::py::tensor_to_gl<owner::cluster8_globals::v_gl>(v),
        kittens::py::tensor_to_gl<owner::cluster8_globals::dout_gl>(dout),
        kittens::py::tensor_to_gl<owner::cluster8_globals::hkv_out_gl>(dk),
        kittens::py::tensor_to_gl<owner::cluster8_globals::hkv_out_gl>(dv),
        kittens::py::tensor_to_gl<owner::cluster8_globals::dq_gl>(dq_accum),
        reinterpret_cast<const float *>(lse_log2.data_ptr()),
        reinterpret_cast<const float *>(delta.data_ptr()),
        static_cast<float>(softmax_scale),
        static_cast<float>(softmax_scale) * tkfa4::kLog2E,
        static_cast<int>(q.size(0)),
    };

    owner::launch_cluster8_owner_bf16(globals, stream);
    owner::launch_dq_finalize(dq_accum, dq, stream);
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
    at::Tensor dq_accum = at::empty(
        q.sizes(),
        q.options().dtype(at::kFloat)
    );
    at::Tensor dq = at::empty(q.sizes(), q.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    backward_bf16_precomputed_out(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        dq_accum,
        dq,
        dk,
        dv,
        softmax_scale
    );
    return {dq, dk, dv};
}

void backward_bf16_out(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse_log2,
    at::Tensor dout,
    at::Tensor delta,
    at::Tensor dq_accum,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
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
    check_common(q, k, v, dout, lse_log2, delta, softmax_scale);
    check_outputs(q, k, v, dq_accum, dq, dk, dv);
    kittens::py::device_check(q, out, dout, delta);
    const c10::cuda::CUDAGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_preprocess_bshd(out, dout, delta, stream);
    backward_bf16_precomputed_out(
        q,
        k,
        v,
        dout,
        lse_log2,
        delta,
        dq_accum,
        dq,
        dk,
        dv,
        softmax_scale
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
    at::Tensor delta = at::empty(
        {q.size(0), owner::kQueryHeads, 1, owner::kSequence},
        q.options().dtype(at::kFloat)
    );
    at::Tensor dq_accum = at::empty(
        q.sizes(),
        q.options().dtype(at::kFloat)
    );
    at::Tensor dq = at::empty(q.sizes(), q.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    backward_bf16_out(
        q,
        k,
        v,
        out,
        lse_log2,
        dout,
        delta,
        dq_accum,
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
        "backward_bf16",
        &backward_bf16,
        "V382 D64 GQA BF16 backward, cluster-8 native BSHD"
    );
    module.def(
        "backward_bf16_precomputed",
        &backward_bf16_precomputed,
        "V382 D64 GQA BF16 backward, cluster-8 with delta"
    );
    module.def(
        "backward_bf16_out",
        &backward_bf16_out,
        "Cluster-8 V382 backward with caller-owned outputs and delta scratch"
    );
    module.def(
        "backward_bf16_precomputed_out",
        &backward_bf16_precomputed_out,
        "Cluster-8 V382 backward with caller-owned outputs"
    );
}

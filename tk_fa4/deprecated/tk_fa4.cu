#include "fa4_bwd_unified_sm100.cuh"
#include "fa4_bwd_postprocess.cuh"
#include "fa4_bwd_preprocess.cuh"
#include "fa4_common.cuh"
#include "fa4_fwd_sm100.cuh"

namespace {

void check_common_forward_inputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    float softmax_scale,
    int64_t actual_seq_len
) {
    tkfa4::check_bhsd(q, "q", at::kBFloat16);
    tkfa4::check_bhsd(k, "k", at::kBFloat16);
    tkfa4::check_bhsd(v, "v", at::kBFloat16);
    kittens::py::device_check(q, k, v);
    tkfa4::check_shapes(q, k, v);

    TORCH_CHECK(tkfa4::is_sm100_device(), "tk_fa4 requires GB200 / SM100");
    TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(q.size(2) % tkfa4::kForwardTileM == 0, "padded seqlen must be divisible by 128");
    TORCH_CHECK(actual_seq_len > 0 && actual_seq_len <= q.size(2), "invalid actual_seq_len");
    TORCH_CHECK(softmax_scale > 0.0f, "softmax_scale must be positive");
}

void check_common_backward_inputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &l_aux,
    const at::Tensor &dout,
    float softmax_scale,
    int64_t actual_seq_len
) {
    check_common_forward_inputs(q, k, v, softmax_scale, actual_seq_len);
    tkfa4::check_bhsd(out, "out", at::kBFloat16);
    tkfa4::check_bhsd(dout, "dout", at::kBFloat16);
    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must match q");
    TORCH_CHECK(
        l_aux.dim() == 4 &&
        l_aux.size(0) == q.size(0) &&
        l_aux.size(1) == q.size(1) &&
        l_aux.size(2) == 1 &&
        l_aux.size(3) == q.size(2),
        "l_aux must have shape (batch, heads, 1, seqlen)"
    );
    TORCH_CHECK(l_aux.dtype() == at::kFloat, "l_aux must be float32");
    kittens::py::device_check(q, k, v, out, l_aux, dout);
}

template <int D>
void launch_forward_impl(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &l_aux,
    bool causal,
    float softmax_scale,
    int actual_seq_len
) {
    tkfa4::fwd::launch<D>(q, k, v, out, l_aux, causal, softmax_scale, actual_seq_len);
}

template <int D>
void launch_backward_impl(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &l_aux,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &delta,
    at::Tensor &dq_accum,
    at::Tensor &dq_semaphore,
    bool causal,
    float softmax_scale,
    int actual_seq_len,
    bool deterministic
) {
    tkfa4::bwd::launch_preprocess<D>(out, dout, delta);
    tkfa4::bwd::launch_backward<D>(
        q, k, v, dout, l_aux, delta, dq, dk, dv, dq_accum, dq_semaphore,
        causal, softmax_scale, actual_seq_len, deterministic
    );
    tkfa4::bwd::postprocess_gradients(dq, dk, dv, deterministic);
}

}  // namespace

std::vector<at::Tensor> mha_fwd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len
) {
    check_common_forward_inputs(q, k, v, softmax_scale, actual_seq_len);

    at::Tensor out = at::empty_like(q);
    at::Tensor l_aux = at::empty({q.size(0), q.size(1), 1, q.size(2)}, q.options().dtype(at::kFloat));

    if (q.size(3) == 64) {
        launch_forward_impl<64>(q, k, v, out, l_aux, causal, softmax_scale, static_cast<int>(actual_seq_len));
    } else {
        launch_forward_impl<128>(q, k, v, out, l_aux, causal, softmax_scale, static_cast<int>(actual_seq_len));
    }
    return {out, l_aux};
}

std::vector<at::Tensor> mha_bwd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor l_aux,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    check_common_backward_inputs(q, k, v, out, l_aux, dout, softmax_scale, actual_seq_len);
    const bool use_dense_hot =
        q.size(3) == 128 &&
        tkfa4::bwd::use_dense_hot_backward(
            q, k, causal, static_cast<int>(actual_seq_len), deterministic
        );
    const bool use_wg_hot =
        !use_dense_hot &&
        q.size(3) == 128 &&
        !deterministic &&
        tkfa4::bwd::use_wg_hot_backward(q, k, causal, static_cast<int>(actual_seq_len));

    at::Tensor dq = use_wg_hot
        ? at::zeros({q.size(0), q.size(1), q.size(2), q.size(3)}, l_aux.options())
        : at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, l_aux.options());
    at::Tensor dk = use_wg_hot
        ? at::zeros({k.size(0), k.size(1), k.size(2), k.size(3)}, l_aux.options())
        : at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, l_aux.options());
    at::Tensor dv = use_wg_hot
        ? at::zeros({v.size(0), v.size(1), v.size(2), v.size(3)}, l_aux.options())
        : at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, l_aux.options());
    at::Tensor delta = at::empty({q.size(0), q.size(1), 1, q.size(2)}, l_aux.options());
    at::Tensor dq_accum;
    at::Tensor dq_semaphore;
    if (!use_wg_hot) {
        const int cluster_size = tkfa4::bwd::select_backward_cluster_size(
            q, k, causal, static_cast<int>(actual_seq_len), deterministic
        );
        const int q_tiles = static_cast<int>(q.size(2) / tkfa4::kForwardTileM);
        dq_accum = at::zeros(
            {q.size(0), q.size(1), q_tiles, cluster_size, tkfa4::kForwardTileM, q.size(3)},
            l_aux.options()
        );
        dq_semaphore = at::empty(
            {q.size(0), q.size(1), q_tiles, cluster_size},
            q.options().dtype(at::kInt)
        );
    }

    if (q.size(3) == 64) {
        launch_backward_impl<64>(
            q, k, v, out, l_aux, dout, dq, dk, dv, delta, dq_accum, dq_semaphore,
            causal, softmax_scale, static_cast<int>(actual_seq_len), deterministic
        );
    } else {
        launch_backward_impl<128>(
            q, k, v, out, l_aux, dout, dq, dk, dv, delta, dq_accum, dq_semaphore,
            causal, softmax_scale, static_cast<int>(actual_seq_len), deterministic
        );
    }

    return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_fwd", &mha_fwd, "ThunderKittens GB200 BF16 FlashAttention forward");
    m.def("mha_bwd", &mha_bwd, "ThunderKittens GB200 BF16 FlashAttention backward");
}

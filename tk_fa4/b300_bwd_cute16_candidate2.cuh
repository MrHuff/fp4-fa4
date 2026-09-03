#pragma once

#include "b300_bwd_cute16.cuh"
#include "b300_bwd_cute16_kernel_candidate2.cuh"

namespace tkfa4::bwd_cute16_candidate2 {

template <int _Mb, int _Nb, int _Dqk, int _Dvo, int _ClusterSize>
using config = bwd_cute16_kernel_candidate2::config<_Mb, _Nb, _Dqk, _Dvo, _ClusterSize>;

template <typename C>
inline void launch_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool causal,
    float scale,
    [[maybe_unused]] bool deterministic
) {
    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor dqacc_dummy = at::empty({1}, lse.options());
    bwd_cute16::launch_preprocess<bwd_cute16::preprocess_config<kB300VDim>>(out, dout, lse, dpsum, lse_log2, dq);
    bwd_cute16_kernel_candidate2::launch_backward<C>(
        q,
        k,
        v,
        dout,
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk,
        dv,
        dqacc_dummy,
        dqacc_dummy,
        dqacc_dummy,
        nullptr,
        causal,
        scale,
        deterministic
    );
}

}  // namespace tkfa4::bwd_cute16_candidate2

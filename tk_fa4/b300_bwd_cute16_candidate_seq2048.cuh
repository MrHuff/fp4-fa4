#pragma once

#include <torch/extension.h>

namespace tkfa4::bwd_cute16_candidate_seq2048 {

void launch_backward_cluster1_seq2048(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    at::Tensor &dqacc,
    float scale
);

}  // namespace tkfa4::bwd_cute16_candidate_seq2048

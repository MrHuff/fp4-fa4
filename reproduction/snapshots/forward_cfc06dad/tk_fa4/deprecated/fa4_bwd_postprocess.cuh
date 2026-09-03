#pragma once

#include "fa4_common.cuh"

namespace tkfa4::bwd {

inline void postprocess_gradients(
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    bool deterministic
) {
    (void)dq;
    (void)dk;
    (void)dv;
    (void)deterministic;
}

}  // namespace tkfa4::bwd

// Backward-only MXFP4/FP4 FA4 extension entrypoint.
#ifndef TK_FA4_BACKWARD_ONLY_BUILD
#define TK_FA4_BACKWARD_ONLY_BUILD 1
#endif

#include "bwd_experiment_common.inc"
#include "bwd_experiment_score_quant.inc"
#include "bwd_experiment_streaming_live.inc"
#include "bwd_experiment_transpose_delta.inc"

#include "bwd_shared_host_helpers.inc"
#include "bwd_host_dispatch.inc"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#include "bwd_pybind.inc"
}

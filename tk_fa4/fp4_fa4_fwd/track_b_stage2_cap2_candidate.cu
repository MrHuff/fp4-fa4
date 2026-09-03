// Candidate-only translation unit for the Track-B saturated Stage-2 cap2
// experiment.  This deliberately does not include the monolithic experiment
// TU or either generic host/binding catalog: those files contain direct host
// wrappers which instantiate unrelated kernels even when selector branches
// are preprocessor-disabled.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_TRACK_B_STAGE2_CAP2_ONLY_BUILD 1
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Shared device definitions needed by the one streaming kernel family.  They
// are templates only; the minimal host include below owns the only two launch
// instantiations emitted by this translation unit.
#include "fwd_bf16_baseline.inc"
#include "stage2_ex2_alu_helpers.cuh"
#include "fwd_configs.inc"
#include "fwd_device_helpers.inc"
#include "fwd_option_b_nb64_dvo128.inc"
#include "fwd_cluster_p_pipeline.inc"
#include "fwd_local_p_stage_ladder.inc"
#include "fwd_tmem224_alternatives.inc"
#include "fwd_raw_fp4_factor_probe.inc"
#include "fwd_raw_fp4_throughput_probe.inc"
#include "fwd_mxfp4_qk_cta_group_ab_probe.inc"
#include "fwd_scaled_k64_two_query_ceiling.inc"
#include "fwd_mxfp4_sfid_probe.inc"
#include "fwd_tmem_output_checkpoint_probe.inc"
#include "fwd_register_output_probe.inc"
#include "fwd_raw_fp4_two_query_probe.inc"
#include "fwd_streaming_kernel.inc"

#include "shared_host_helpers.inc"
#include "track_b_stage2_cap2_minimal_host.inc"

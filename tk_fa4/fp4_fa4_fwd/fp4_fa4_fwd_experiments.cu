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

#ifndef TK_FA4_FORWARD_ONLY_BUILD
#define TK_FA4_FORWARD_ONLY_BUILD 1
#endif

// Forward implementation split into focused include files.
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
#include "fwd_dual_query_kernel.inc"
#include "fwd_dual_query_two_score_overlay.inc"
#include "fwd_scaled_k64_two_query_real_softmax.inc"
#include "fwd_quarter_lifetime_two_query_overlay.inc"
#include "fwd_dual_query_two_n64.inc"
#include "fwd_dual_query_one_n128.inc"
#include "fwd_option_a_replay.inc"
#include "fwd_dual_query_one_n128_bf16ret.inc"
#include "fwd_dual_query_one_n128_fp16ret.inc"
#include "fwd_dual_query_one_n128_fp16scaledret.inc"
#include "fwd_dual_query_one_n128_evac.inc"
#include "fwd_dual_query_one_n128_3wg.inc"
#include "fwd_dual_query_one_n128_smem_evac.inc"
#include "fwd_dual_query_one_n128_no_retention.inc"
#include "fwd_fused_stream_kernel.inc"
#include "fwd_host_bindings.inc"

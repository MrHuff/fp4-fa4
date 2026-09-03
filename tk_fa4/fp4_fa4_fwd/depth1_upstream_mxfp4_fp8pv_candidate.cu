// Candidate-only TU.  It intentionally excludes normal dispatch/catalog
// definitions and instantiates one standalone upstream-owned route.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1
#define TK_FA4_UPSTREAM_OWNED_TWO_STAGE_ONLY_BUILD 1
#define TK_FA4_UPSTREAM_FP8PV_TWO_STAGE_ONLY_BUILD 1

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

#include "upstream_mxfp4_fp8pv_bf16_baseline.inc"
#include "stage2_ex2_alu_helpers.cuh"
#include "fwd_configs.inc"
#include "fwd_device_helpers.inc"
#include "depth1_upstream_mxfp4_fp8pv_config.inc"
#include "depth1_upstream_mxfp4_fp8pv_kernel.inc"
#include "shared_host_helpers.inc"
#include "depth1_upstream_mxfp4_fp8pv_minimal_host.inc"

static_assert(
    !upstream_mx_subnormal_contract<tk_hao_native_port_config>::enabled);
static_assert(
    !upstream_mx_subnormal_contract<tk_hao_native_port_config>::scalar_lift);

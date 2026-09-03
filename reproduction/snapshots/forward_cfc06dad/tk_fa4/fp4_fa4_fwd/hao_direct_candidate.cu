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
#define TK_HAO_DIRECT_HELPERS_ONLY 1
#include "depth1_upstream_mxfp4_fp8pv_kernel.inc"
#undef TK_HAO_DIRECT_HELPERS_ONLY
#include "hao_direct_config.inc"
#include "hao_direct_kernel.inc"
#include "shared_host_helpers.inc"
#include "hao_direct_host.inc"

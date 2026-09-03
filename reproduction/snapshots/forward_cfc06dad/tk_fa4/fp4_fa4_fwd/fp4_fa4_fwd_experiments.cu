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
#include "fwd_streaming_kernel.inc"
#include "fwd_fused_stream_kernel.inc"
#include "fwd_host_bindings.inc"

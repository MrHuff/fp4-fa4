// Candidate-only N64-combine route authorized by a pinned runtime proof of
// two-CTA overlap for the exact tensor_allocator<2,1> Option-B donor.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1
#define TK_FA4_LARGE_I73_N64_COMBINE_ONLY_BUILD 1
#define TK_FA4_LARGE_I73_N64_COMBINE_CLUSTER_METADATA_FREE 1
#define TK_FA4_LARGE_I73_N64_COMBINE_REG120 1
#define TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_GATE 1
#define TK_FA4_LARGE_I73_N64_COMBINE_RUNTIME_PROOF_SHA256 \
    "173d51771205f51d6b89794bd69009f32e175a614d71b4a6d0a0334afd62f399"
#define TK_FA4_LARGE_I73_N64_COMBINE_ROUTE \
    "real_fwd_large_i73_s4096h64_n64_combine_runtimeproof_reg120"

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

#include "fwd_bf16_baseline.inc"
#include "stage2_ex2_alu_helpers.cuh"
#include "fwd_configs.inc"

struct config_fp4pv_large_i73_n64_combine_runtimeproof_reg120
    : public config_fp4pv<128, 64, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
};

using config_fp4pv_large_i73_n64_combine =
    config_fp4pv_large_i73_n64_combine_runtimeproof_reg120;

#include "fwd_device_helpers.inc"
#include "fwd_large_i73_n64_combine.inc"
#include "shared_host_helpers.inc"
#include "large_i73_n64_combine_minimal_host.inc"

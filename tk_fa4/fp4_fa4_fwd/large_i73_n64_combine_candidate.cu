// Candidate-only S4096/H64 N64-combine specialization.
//
// The arithmetic shell is derived from the byte-preserved Option-B donor,
// but its two N64 recurrence updates are deliberately replaced by one N128
// update after a single lower-score evacuation.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1
#define TK_FA4_LARGE_I73_N64_COMBINE_ONLY_BUILD 1

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

struct config_fp4pv_large_i73_n64_combine
    : public config_fp4pv<128, 64, 192, 128, 1> {
    static constexpr int TOTAL_WGS = 2;
    static constexpr int NUM_WARPS = TOTAL_WGS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr bool ONLINE_QK_LOGICAL_N = true;
    static constexpr bool ONLINE_QK_PAIR_NB64 = true;
};

#include "fwd_device_helpers.inc"
#include "fwd_large_i73_n64_combine.inc"
#include "shared_host_helpers.inc"
#include "large_i73_n64_combine_minimal_host.inc"

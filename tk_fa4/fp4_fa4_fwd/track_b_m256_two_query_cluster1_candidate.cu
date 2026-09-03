// Candidate-only translation unit for the Track-B M256 two-query cluster-1
// specialization.  It deliberately excludes the generic host/pybind catalogs
// so unrelated streaming kernels cannot be instantiated by this build.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_TRACK_B_M256_TWO_QUERY_CLUSTER1_ONLY_BUILD 1
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

#include "fwd_bf16_baseline.inc"
#include "stage2_ex2_alu_helpers.cuh"
#include "fwd_configs.inc"
#include "fwd_device_helpers.inc"

// A distinct type gives the exact target its own SASS/resource symbol.  The
// arithmetic and lifecycle come from the existing M256 implementation; this
// specialization only makes its cluster/topology contract explicit.
struct config_fp4pv_track_b_m256_two_query_cluster1_scalarized
    : public config_fp4pv_stage2_ex2_alu_dual_query_one_n128_3wg {
    static constexpr bool TRACK_B_M256_TWO_QUERY_CLUSTER1 = true;
};

#include "fwd_dual_query_one_n128_3wg.inc"
#include "shared_host_helpers.inc"
#include "track_b_m256_two_query_cluster1_minimal_host.inc"

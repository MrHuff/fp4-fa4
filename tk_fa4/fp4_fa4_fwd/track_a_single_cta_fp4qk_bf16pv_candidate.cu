#include <pybind11/pybind11.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

// Candidate-only translation unit: instantiate exactly the two Track-A
// group-1 routes and none of the production/native-group2 host dispatches.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1
#define TK_FA4_SINGLE_CTA_FP4QK_BF16PV_ONLY_BUILD 1

#include "fwd_bf16_baseline.inc"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_single_cta_fp4qk_fp4v_dequant_bf16pv",
        &dispatch_single_cta_fp4qk_fp4v_dequant_bf16pv,
        "Track-A production-input single-CTA FP4 QK, BF16 P/PV with FP4 V dequantization");
    m.def(
        "forward_single_cta_fp4qk_bf16v_bf16pv_ceiling",
        &dispatch_single_cta_fp4qk_bf16v_bf16pv_ceiling,
        "Track-A changed-input BF16-V speed ceiling");
}

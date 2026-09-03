#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_MXFP4_QK_CTA_GROUP_AB_BUILD 1

#include <pybind11/pybind11.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "fwd_bf16_baseline.inc"
#include "stage2_ex2_alu_helpers.cuh"
#include "fwd_configs.inc"
#include "fwd_device_helpers.inc"
#include "fwd_mxfp4_qk_cta_group_ab_probe.inc"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "mxfp4_qk_group1_issue_ceiling",
        &dispatch_mxfp4_qk_group1_issue_ceiling);
    m.def(
        "mxfp4_qk_group1_production_cadence",
        &dispatch_mxfp4_qk_group1_production_cadence);
    m.def(
        "mxfp4_qk_group2_issue_ceiling",
        &dispatch_mxfp4_qk_group2_issue_ceiling);
    m.def(
        "mxfp4_qk_group2_production_cadence",
        &dispatch_mxfp4_qk_group2_production_cadence);
}

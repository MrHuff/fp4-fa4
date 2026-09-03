#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_RAW_FP4_THROUGHPUT_PROBE_BUILD 1

#include <pybind11/pybind11.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <type_traits>

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

#include "fwd_raw_fp4_throughput_helpers.inc"
#include "fwd_raw_fp4_throughput_probe.inc"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "raw_fp4_throughput_probe",
        &dispatch_raw_fp4_throughput_probe,
        "Matched M128xN128xK128 TCGEN throughput probe");
}

// Candidate-only occupancy control for the byte-preserved Option-B donor.
//
// This translation unit deliberately exposes no forward/dispatch wrapper and
// never launches the kernel.  Its sole binding configures and queries the
// exact cap-2, trace-disabled specialization used by the historical
// two-resident Option-B experiment.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1
#define TK_FA4_OPTION_AB_PROBE_BUILD 1

#include <pybind11/pybind11.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "fwd_bf16_baseline.inc"
#include "stage2_ex2_alu_helpers.cuh"
#include "fwd_configs.inc"
#include "fwd_device_helpers.inc"
#include "fwd_option_b_nb64_dvo128.inc"

namespace {

constexpr const char *OPTION_B_OCCUPANCY_ROUTE =
    "option_b_nb64_dvo128_cap2_occupancy_control";

pybind11::dict read_option_b_nb64_dvo128_occupancy() {
    using C = config_fp4pv_option_b_nb64_dvo128;
    using G = globals_mxfp4_option_b_nb64_dvo128<C>;

    static_assert(C::NUM_THREADS == 256);
    static_assert(C::TOTAL_WGS == 2);
    static_assert(C::CLUSTER_SIZE == 1);
    static_assert(C::Mb == 128 && C::Nb == 64);
    static_assert(C::Dqk == 192 && C::Dvo == 128);
    static_assert(tensor_allocator<2, 1>::cols == 256);
    static_assert(G::DYNAMIC_SHARED_MEMORY == 33792);

    // Keep this as the one and only exact kernel-specialization reference in
    // the TU so the resulting cubin has an attributable entry-function
    // census.  The donor declaration supplies __launch_bounds__(256, 2) and
    // has no explicit-cluster metadata.
    const auto kernel =
        kernel_mxfp4_option_b_nb64_dvo128<C, 2, false>;

    CUDACHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        G::DYNAMIC_SHARED_MEMORY));
    CUDACHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaFuncAttributes attributes{};
    CUDACHECK(cudaFuncGetAttributes(&attributes, kernel));

    int resident_at_zero = 0;
    int resident_at_requested = 0;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_at_zero,
        kernel,
        C::NUM_THREADS,
        0));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_at_requested,
        kernel,
        C::NUM_THREADS,
        G::DYNAMIC_SHARED_MEMORY));

    size_t available_dynamic_for_two = 0;
    const cudaError_t availability_status =
        cudaOccupancyAvailableDynamicSMemPerBlock(
            &available_dynamic_for_two,
            kernel,
            2,
            C::NUM_THREADS);
    const long long available_dynamic_for_two_value =
        availability_status == cudaSuccess
            ? static_cast<long long>(available_dynamic_for_two)
            : -1;

    pybind11::dict out;
    out["schema"] = "mxfp4_option_b_nb64_dvo128_occupancy_control_v1";
    out["route"] = OPTION_B_OCCUPANCY_ROUTE;
    out["config"] = "config_fp4pv_option_b_nb64_dvo128";
    out["kernel"] =
        "kernel_mxfp4_option_b_nb64_dvo128<C,2,false>";
    out["threads_per_cta"] = C::NUM_THREADS;
    out["requested_min_blocks_per_sm"] = 2;
    out["physical_tmem_cols"] = tensor_allocator<2, 1>::cols;
    out["dynamic_shared_bytes"] = G::DYNAMIC_SHARED_MEMORY;
    out["func_max_threads_per_block"] = attributes.maxThreadsPerBlock;
    out["func_num_regs"] = attributes.numRegs;
    out["func_static_shared_bytes"] =
        static_cast<long long>(attributes.sharedSizeBytes);
    out["func_local_bytes"] =
        static_cast<long long>(attributes.localSizeBytes);
    out["func_const_bytes"] =
        static_cast<long long>(attributes.constSizeBytes);
    out["func_max_dynamic_shared_bytes"] =
        attributes.maxDynamicSharedSizeBytes;
    out["func_preferred_shared_carveout"] =
        attributes.preferredShmemCarveout;
    out["resident_at_zero"] = resident_at_zero;
    out["resident_at_requested"] = resident_at_requested;
    out["available_dynamic_for_two"] =
        available_dynamic_for_two_value;
    out["available_dynamic_for_two_status"] =
        static_cast<int>(availability_status);
    out["available_dynamic_for_two_status_name"] =
        cudaGetErrorName(availability_status);
    return out;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "read_option_b_nb64_dvo128_occupancy",
        &read_option_b_nb64_dvo128_occupancy,
        "Query the exact Option-B cap-2 kernel occupancy without launching");
}

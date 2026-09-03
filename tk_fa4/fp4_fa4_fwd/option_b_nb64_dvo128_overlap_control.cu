// Candidate-only runtime-overlap proof for the byte-preserved Option-B donor.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1
#define TK_FA4_OPTION_AB_PROBE_BUILD 1

#include <pybind11/pybind11.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "fwd_bf16_baseline.inc"
#include "stage2_ex2_alu_helpers.cuh"
#include "fwd_configs.inc"
#include "fwd_device_helpers.inc"
#include "fwd_option_b_nb64_dvo128.inc"
#include "shared_host_helpers.inc"

namespace {

void run_option_b_nb64_dvo128_overlap(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    at::Tensor Trace) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V); CHECK_INPUT(V_sc); CHECK_INPUT(O); CHECK_INPUT(LSE);
    CHECK_INPUT(Trace);

    using C = config_fp4pv_option_b_nb64_dvo128;
    using G = globals_mxfp4_option_b_nb64_dvo128<C>;
    constexpr int BATCH = 1;
    constexpr int HEADS = 32;
    constexpr int SEQLEN = 2048;
    constexpr int TRACE_STRIDE = 16;
    constexpr int GRID_BLOCKS = BATCH * HEADS * (SEQLEN / C::Mb);
    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2);
    static_assert(C::CLUSTER_SIZE == 1 && C::Mb == 128 && C::Nb == 64);
    static_assert(C::Dqk == 192 && C::Dvo == 128);
    static_assert(tensor_allocator<2, 1>::cols == 256);
    static_assert(G::DYNAMIC_SHARED_MEMORY == 33792);
    static_assert(GRID_BLOCKS == 512);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    K.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
                    V.dtype() == at::ScalarType::Float4_e2m1fn_x2,
                "Option-B overlap requires packed MXFP4 Q/K/V");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Float8_e4m3fn &&
                    K_sc.dtype() == at::ScalarType::Float8_e4m3fn &&
                    Q_sg.dtype() == at::ScalarType::Float &&
                    K_sg.dtype() == at::ScalarType::Float &&
                    V_sc.dtype() == at::ScalarType::Byte,
                "Option-B overlap scale dtype mismatch");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16 &&
                    LSE.dtype() == at::ScalarType::Float &&
                    Trace.dtype() == at::ScalarType::Long,
                "Option-B overlap output/trace dtype mismatch");
    TORCH_CHECK(Q.sizes() == at::IntArrayRef({BATCH, HEADS, SEQLEN, 96}) &&
                    K.sizes() == Q.sizes(),
                "Option-B overlap Q/K shape mismatch");
    TORCH_CHECK(V.sizes() == at::IntArrayRef({BATCH, HEADS, 128, SEQLEN / 2}),
                "Option-B overlap V shape mismatch");
    TORCH_CHECK(O.sizes() == at::IntArrayRef({BATCH, SEQLEN, HEADS, 128}) &&
                    LSE.sizes() == at::IntArrayRef({BATCH, HEADS, 1, SEQLEN}),
                "Option-B overlap output/LSE shape mismatch");
    TORCH_CHECK(Trace.numel() == GRID_BLOCKS * TRACE_STRIDE,
                "Option-B overlap Trace must contain 512x16 int64 values");
    TORCH_CHECK(Q_sc.sizes() == at::IntArrayRef({BATCH, SEQLEN / 128,
                    HEADS * C::QK_SCALE_CHUNKS, 512}),
                "Option-B overlap Q scale shape mismatch");
    TORCH_CHECK(K_sc.sizes() == at::IntArrayRef({BATCH, SEQLEN / 64,
                    HEADS * C::QK_SCALE_CHUNKS, 512}),
                "Option-B overlap K scale shape mismatch");
    TORCH_CHECK(Q_sg.sizes() == at::IntArrayRef({BATCH, HEADS}) &&
                    K_sg.sizes() == Q_sg.sizes(),
                "Option-B overlap global-scale shape mismatch");
    check_fp4pv_mxfp4_v_scale_half_contract(
        V_sc, BATCH, HEADS, SEQLEN);

    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        kittens::py::tensor_to_gl<typename G::v_gl>(V),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        reinterpret_cast<const float *>(Q_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        reinterpret_cast<const float *>(K_sg.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        reinterpret_cast<bf16 *>(O.data_ptr()),
        reinterpret_cast<float *>(LSE.data_ptr()),
        reinterpret_cast<unsigned long long *>(Trace.data_ptr<int64_t>()),
        BATCH, HEADS, SEQLEN, TRACE_STRIDE};

    const auto kernel =
        kernel_mxfp4_option_b_nb64_dvo128<C, 2, true>;
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        G::DYNAMIC_SHARED_MEMORY));
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    LaunchConfig<true, false> launch_config(
        dim3(GRID_BLOCKS), dim3(C::NUM_THREADS), G::DYNAMIC_SHARED_MEMORY,
        at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel, g));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_option_b_nb64_dvo128_overlap",
          &run_option_b_nb64_dvo128_overlap,
          "Run fixed S2048/H32 Option-B cap2 trace proof");
}

/*
 * Grouped NVFP4 GEMM for QKV projection using CUTLASS 3 on Blackwell SM100.
 *
 * Pattern: shared A (input activations) × per-group B (Wq, Wk, Wv) → per-group D
 * Each group can have different N dimensions while sharing M and K.
 *
 * Based on CUTLASS examples 72a (NVFP4 types) and the block-scaled ptr-array
 * mainloop in sm100_blockscaled_mma_array_warpspecialized.hpp.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// ===== Type configuration =====
using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4
using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4
using ElementC    = cutlass::bfloat16_t;
using ElementD    = cutlass::bfloat16_t;
using ElementAcc  = float;

// For grouped GEMM: A is RowMajor (M×K), B is ColumnMajor (N×K)
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;  // 128 bits / 4 bits
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// ===== Kernel configuration =====
using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape  = Shape<_128, _256, _256>;
using ClusterShape  = Shape<int32_t, int32_t, _1>;  // Runtime cluster shape

// Ptr-Array NVFP4 schedules for SM100
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

// Use GroupProblemShape for per-group M,N,K
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

// Build epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    Shape<_128, _64>,
    ElementAcc, ElementAcc,
    ElementC, LayoutCTag *, AlignmentC,   // ptr-array C
    ElementD, LayoutCTag *, AlignmentD,   // ptr-array D
    EpilogueSchedule
>::CollectiveOp;

// Build mainloop  
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,   // ptr-array A
    ElementB, LayoutBTag *, AlignmentB,   // ptr-array B
    ElementAcc,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

// Full GEMM kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Get types from the kernel (don't hardcode — let the kernel tell us)
using StrideA = typename CollectiveMainloop::StrideA;
using StrideB = typename CollectiveMainloop::StrideB;
using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
using InternalLayoutSFA = typename CollectiveMainloop::InternalLayoutSFA;
using InternalLayoutSFB = typename CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
using ArrayElementA = typename CollectiveMainloop::ArrayElementA;
using ArrayElementB = typename CollectiveMainloop::ArrayElementB;
using ElementSF = typename CollectiveMainloop::ElementSF;  // kernel decides SF type

// ===== Helpers =====
void fill_random_packed(void* ptr, size_t num_bytes) {
    std::vector<uint8_t> host(num_bytes);
    for (size_t i = 0; i < num_bytes; i++) host[i] = rand() & 0xFF;
    cudaMemcpy(ptr, host.data(), num_bytes, cudaMemcpyHostToDevice);
}

template <typename T>
void fill_random_typed(T* ptr, size_t count, float lo, float hi) {
    std::vector<T> host(count);
    for (size_t i = 0; i < count; i++) host[i] = static_cast<T>(lo + (hi-lo)*rand()/RAND_MAX);
    cudaMemcpy(ptr, host.data(), count * sizeof(T), cudaMemcpyHostToDevice);
}

// ===== Main =====
int main(int argc, char** argv) {
    int M = 2048;
    int K = 4096;
    int num_groups = 3;
    std::vector<int> n_dims = {4096, 1024, 1024};  // Q, K, V

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) {
        num_groups = argc - 3;
        n_dims.resize(num_groups);
        for (int i = 0; i < num_groups; i++) n_dims[i] = atoi(argv[3+i]);
    }

    std::cout << "=== Grouped NVFP4 GEMM Benchmark ===" << std::endl;
    std::cout << "  M=" << M << " K=" << K << " groups=" << num_groups << std::endl;
    for (int g = 0; g < num_groups; g++) 
        std::cout << "  Group " << g << ": N=" << n_dims[g] << std::endl;

    // ----- Problem shapes -----
    std::vector<ProblemShape::UnderlyingProblemShape> problem_shapes_host(num_groups);
    for (int g = 0; g < num_groups; g++)
        problem_shapes_host[g] = {M, n_dims[g], K};

    cutlass::DeviceAllocation<ProblemShape::UnderlyingProblemShape> problem_shapes_device(num_groups);
    problem_shapes_device.copy_from_host(problem_shapes_host.data());

    // ----- Allocate A (shared), B/D per group -----
    // A data: RowMajor (M×K), packed FP4 = M*K/2 bytes
    size_t a_bytes = (size_t)M * K / 2;
    void* d_A_shared;
    cudaMalloc(&d_A_shared, a_bytes);
    fill_random_packed(d_A_shared, a_bytes);

    // SFA layout from Sm1xxBlkScaledConfig (shared)
    auto layout_SFA_ref = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, n_dims[0], K, 1));
    size_t sfa_elems = size(filter_zeros(layout_SFA_ref));
    ElementSF* d_SFA_shared;
    cudaMalloc(&d_SFA_shared, sfa_elems * sizeof(ElementSF));
    fill_random_typed(d_SFA_shared, sfa_elems, 1.0f, 4.0f);

    // Per-group allocations
    std::vector<ArrayElementA const*> h_ptr_A(num_groups);
    std::vector<ArrayElementB const*> h_ptr_B(num_groups);
    std::vector<ElementSF const*> h_ptr_SFA(num_groups);
    std::vector<ElementSF const*> h_ptr_SFB(num_groups);
    std::vector<ElementC const*> h_ptr_C(num_groups);
    std::vector<ElementD*> h_ptr_D(num_groups);

    // Per-group strides and layouts
    std::vector<InternalStrideA> h_strides_A(num_groups);
    std::vector<InternalStrideB> h_strides_B(num_groups);
    std::vector<InternalLayoutSFA> h_layouts_SFA(num_groups);
    std::vector<InternalLayoutSFB> h_layouts_SFB(num_groups);

    // Per-group stride arrays for C/D (epilogue)
    using StrideC = typename CollectiveEpilogue::StrideC;
    using StrideD = typename CollectiveEpilogue::StrideD;
    // For grouped GEMM, StrideC/D are pointers — the internal type is what they point to
    using InternalStrideC = cute::remove_pointer_t<StrideC>;
    using InternalStrideD = cute::remove_pointer_t<StrideD>;
    std::vector<InternalStrideC> h_strides_C(num_groups);
    std::vector<InternalStrideD> h_strides_D(num_groups);

    for (int g = 0; g < num_groups; g++) {
        int N = n_dims[g];

        // A shared across groups
        h_ptr_A[g] = reinterpret_cast<ArrayElementA const*>(d_A_shared);
        h_ptr_SFA[g] = d_SFA_shared;

        // B: ColMajor (N×K)
        size_t b_bytes = (size_t)N * K / 2;
        void* d_B;
        cudaMalloc(&d_B, b_bytes);
        fill_random_packed(d_B, b_bytes);
        h_ptr_B[g] = reinterpret_cast<ArrayElementB const*>(d_B);

        // SFB
        auto layout_SFB_g = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
        size_t sfb_elems = size(filter_zeros(layout_SFB_g));
        ElementSF* d_SFB;
        cudaMalloc(&d_SFB, sfb_elems * sizeof(ElementSF));
        fill_random_typed(d_SFB, sfb_elems, 1.0f, 4.0f);
        h_ptr_SFB[g] = d_SFB;

        // C (void) and D
        h_ptr_C[g] = nullptr;
        ElementD* d_D;
        cudaMalloc(&d_D, (size_t)M * N * sizeof(ElementD));
        h_ptr_D[g] = d_D;

        // Strides for A: RowMajor (M,K,L=1)
        h_strides_A[g] = cutlass::make_cute_packed_stride(InternalStrideA{}, make_shape(M, K, 1));
        // Strides for B: ColMajor (N,K,L=1)
        h_strides_B[g] = cutlass::make_cute_packed_stride(InternalStrideB{}, make_shape(N, K, 1));
        // Layouts for SF
        h_layouts_SFA[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        h_layouts_SFB[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

        // Strides for C/D: RowMajor
        h_strides_C[g] = cutlass::make_cute_packed_stride(InternalStrideC{}, make_shape(M, N, 1));
        h_strides_D[g] = cutlass::make_cute_packed_stride(InternalStrideD{}, make_shape(M, N, 1));
    }

    // Device arrays for pointers
    cutlass::DeviceAllocation<ArrayElementA const*> d_ptr_A(num_groups);
    cutlass::DeviceAllocation<ArrayElementB const*> d_ptr_B(num_groups);
    cutlass::DeviceAllocation<ElementSF const*> d_ptr_SFA(num_groups);
    cutlass::DeviceAllocation<ElementSF const*> d_ptr_SFB(num_groups);
    cutlass::DeviceAllocation<ElementC const*> d_ptr_C(num_groups);
    cutlass::DeviceAllocation<ElementD*> d_ptr_D(num_groups);
    d_ptr_A.copy_from_host(h_ptr_A.data());
    d_ptr_B.copy_from_host(h_ptr_B.data());
    d_ptr_SFA.copy_from_host(h_ptr_SFA.data());
    d_ptr_SFB.copy_from_host(h_ptr_SFB.data());
    d_ptr_C.copy_from_host(h_ptr_C.data());
    d_ptr_D.copy_from_host(h_ptr_D.data());

    // Device arrays for strides/layouts
    cutlass::DeviceAllocation<InternalStrideA> d_strides_A(num_groups);
    cutlass::DeviceAllocation<InternalStrideB> d_strides_B(num_groups);
    cutlass::DeviceAllocation<InternalLayoutSFA> d_layouts_SFA(num_groups);
    cutlass::DeviceAllocation<InternalLayoutSFB> d_layouts_SFB(num_groups);
    cutlass::DeviceAllocation<InternalStrideC> d_strides_C(num_groups);
    cutlass::DeviceAllocation<InternalStrideD> d_strides_D(num_groups);
    d_strides_A.copy_from_host(h_strides_A.data());
    d_strides_B.copy_from_host(h_strides_B.data());
    d_layouts_SFA.copy_from_host(h_layouts_SFA.data());
    d_layouts_SFB.copy_from_host(h_layouts_SFB.data());
    d_strides_C.copy_from_host(h_strides_C.data());
    d_strides_D.copy_from_host(h_strides_D.data());

    // ----- Build arguments -----
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    hw_info.cluster_shape = dim3(2, 1, 1);
    hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {static_cast<int32_t>(num_groups), problem_shapes_device.get(), problem_shapes_host.data()},
        // Mainloop args: ptr_A, dA, ptr_B, dB, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB
        {d_ptr_A.get(), d_strides_A.get(), d_ptr_B.get(), d_strides_B.get(),
         d_ptr_SFA.get(), d_layouts_SFA.get(), d_ptr_SFB.get(), d_layouts_SFB.get()},
        // Epilogue args: alpha/beta, ptr_C, stride_C, ptr_D, stride_D
        {{1.0f, 0.0f}, d_ptr_C.get(), d_strides_C.get(), d_ptr_D.get(), d_strides_D.get()},
        hw_info
    };

    // ----- Initialize and run -----
    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    std::cout << "  Workspace: " << workspace_size << " bytes" << std::endl;

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS cannot implement: " << cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Init failed: " << cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    // Warmup
    status = gemm.run();
    cudaDeviceSynchronize();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Run failed: " << cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    std::cout << "  ✅ CUTLASS grouped NVFP4 GEMM ran successfully!" << std::endl;

    // Timing
    const int warmup = 100, iters = 200;
    for (int i = 0; i < warmup; i++) gemm.run();
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) gemm.run();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    uint64_t total_flops = 0;
    for (int g = 0; g < num_groups; g++) total_flops += 2ULL * M * n_dims[g] * K;
    double tflops = total_flops / (ms * 1e9);

    std::cout << "\n  Timing: " << ms << " ms/iter" << std::endl;
    std::cout << "  TFLOPS: " << tflops << std::endl;

    // Cleanup
    cudaFree(d_A_shared);
    cudaFree(d_SFA_shared);
    for (int g = 0; g < num_groups; g++) {
        cudaFree(const_cast<ArrayElementB*>(h_ptr_B[g]));
        cudaFree(const_cast<ElementSF*>(h_ptr_SFB[g]));
        cudaFree(h_ptr_D[g]);
    }

    return 0;
}

#else
int main() {
    std::cerr << "Requires SM100 (Blackwell)." << std::endl;
    return -1;
}
#endif

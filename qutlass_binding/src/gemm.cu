
#include <iostream>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// SM100/120 FP4 GEMM Kernel Definition
// Based on cutlass/examples/79_blackwell_geforce_gemm

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Alignment

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand (interpreted as KxN ColMajor = NxK RowMajor transposed)
constexpr int AlignmentB  = 32;                                             

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            
using         ElementC    = cutlass::bfloat16_t;                            
using         LayoutCTag  = cutlass::layout::RowMajor;                      
using         LayoutDTag  = cutlass::layout::RowMajor;                      
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    

// Kernel functional config
using ElementAccumulator  = float;                                          
using ArchTag             = cutlass::arch::Sm100;                           // Use Sm100 for GB200 compatibility
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          
using ClusterShape        = Shape<_1,_1,_1>; // RTX 50 series limitation, check for GB200? Using 1x1x1 is safe.

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Check support
// #if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// #error "CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined! Ensure CUDA 12.8+ and correct ARCH."
// #endif

extern "C" void fused_gemm_fp4_bf16_sm100(
    void* D_ptr,
    const void* A_ptr,
    const void* B_ptr,
    const void* ScaleA_ptr,
    const void* ScaleB_ptr,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    // Strides
    // A: MxK RowMajor
    auto stride_A = cutlass::make_cute_packed_stride(Gemm::GemmKernel::StrideA{}, {M, K, 1});
    // B: KxN ColMajor (NxK RowMajor)
    auto stride_B = cutlass::make_cute_packed_stride(Gemm::GemmKernel::StrideB{}, {K, N, 1});
    // C, D: MxN RowMajor
    auto stride_C = cutlass::make_cute_packed_stride(Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(Gemm::GemmKernel::StrideD{}, {M, N, 1});

    // Debug print removed

    // Layouts for Scales
    // Using Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    // Also get Element types for scales ensures matching
    using ElementSFA = typename ElementA::ScaleFactorType; 
    using ElementSFB = typename ElementB::ScaleFactorType;



    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    // Arguments
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
          (typename ElementA::DataType*)A_ptr, stride_A,
          (typename ElementB::DataType*)B_ptr, stride_B,
          (typename ElementA::ScaleFactorType*)ScaleA_ptr, layout_SFA, 
          (typename ElementB::ScaleFactorType*)ScaleB_ptr, layout_SFB
        },
        { // Epilogue arguments
          {alpha, beta},
          nullptr, stride_C, // C (not used if beta=0)
          (ElementD*)D_ptr, stride_D
        }
    };

    Gemm gemm;
    
    // Workspace
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    if(workspace_size > 0) {
        cudaMallocAsync(&workspace, workspace_size, stream);
    }

    // Run
    cutlass::Status status = gemm.can_implement(arguments);
    if(status != cutlass::Status::kSuccess) {
        std::cerr << "Gemm::can_implement failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        // throw?
    }

    status = gemm.initialize(arguments, workspace, stream);
    if(status != cutlass::Status::kSuccess) {
         std::cerr << "Gemm::initialize failed" << std::endl;
    }

    status = gemm.run(stream);
    if(status != cutlass::Status::kSuccess) {
        std::cerr << "Gemm::run failed" << std::endl;
    }

    if(workspace) {
        cudaFreeAsync(workspace, stream);
    }
}

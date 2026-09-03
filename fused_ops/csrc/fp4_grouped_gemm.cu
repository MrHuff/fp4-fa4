/*
 * CUTLASS Grouped NVFP4 GEMM — PyTorch Extension CUDA Kernel
 *
 * Performs: for each group g: D[g] = A @ B[g]^T  (NVFP4 × NVFP4 → BF16)
 * Shared A across groups, per-group B and D.
 *
 * This kernel handles the CUTLASS type system and scale factor layout internally.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

// ===== CUTLASS type configuration (must match bench_grouped_nvfp4_gemm.cu) =====
using CutlassElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using CutlassElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using CutlassElementC    = cutlass::bfloat16_t;
using CutlassElementD    = cutlass::bfloat16_t;
using CutlassElementAcc  = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<CutlassElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<CutlassElementD>::value;

using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape  = Shape<_128, _256, _256>;
using ClusterShape  = Shape<int32_t, int32_t, _1>;

using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    Shape<_128, _64>,
    CutlassElementAcc, CutlassElementAcc,
    CutlassElementC, LayoutCTag *, AlignmentC,
    CutlassElementD, LayoutCTag *, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    CutlassElementA, LayoutATag *, AlignmentA,
    CutlassElementB, LayoutBTag *, AlignmentB,
    CutlassElementAcc,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Kernel-derived types
using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
using InternalLayoutSFA = typename CollectiveMainloop::InternalLayoutSFA;
using InternalLayoutSFB = typename CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
using ArrayElementA = typename CollectiveMainloop::ArrayElementA;
using ArrayElementB = typename CollectiveMainloop::ArrayElementB;
using ElementSF = typename CollectiveMainloop::ElementSF;
using StrideC = typename CollectiveEpilogue::StrideC;
using StrideD = typename CollectiveEpilogue::StrideD;
using InternalStrideC = cute::remove_pointer_t<StrideC>;
using InternalStrideD = cute::remove_pointer_t<StrideD>;

// Persistent workspace for the GEMM kernel
static cutlass::DeviceAllocation<uint8_t>* g_workspace = nullptr;
static size_t g_workspace_size = 0;

static void ensure_workspace(size_t needed) {
    if (g_workspace == nullptr || g_workspace_size < needed) {
        delete g_workspace;
        g_workspace = new cutlass::DeviceAllocation<uint8_t>(needed);
        g_workspace_size = needed;
    }
}

// ===== Scale Factor Layout Conversion =====
// TE stores scale factors in row-major: sf_te[row * sf_stride + col_block]
//   where col_block = col / 16, sf_stride = cols / 16
// CUTLASS expects them in the SfAtom swizzled layout via InternalLayoutSFA.
//
// Instead of hand-deriving the swizzle formula, we use CuTe's own
// InternalLayoutSFA layout object to compute offsets — guaranteed correct.
__global__ void convert_te_sf_to_cutlass_kernel(
    const uint8_t* __restrict__ te_sf,   // TE: row-major (rows, k_blocks) as E4M3 bytes
    uint8_t* __restrict__ cutlass_sf,    // CUTLASS: SfAtom swizzled layout
    InternalLayoutSFA layout_sfa,        // CuTe layout object for SFA
    int rows, int k_blocks, int te_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * k_blocks;
    if (idx >= total) return;

    int row = idx / k_blocks;
    int kb = idx % k_blocks;

    // Read from TE row-major layout
    uint8_t val = te_sf[row * te_stride + kb];

    // Use CuTe layout to compute CUTLASS physical offset.
    // The SFA layout maps (M, K_data, L) coordinates.
    // k_data = kb * SFVecSize(=16) — any element in the block works
    // because SfAtom has stride-0 on the SFVecSize sub-mode.
    int k_data = kb * 16;
    int offset = layout_sfa(row, k_data, 0);

    cutlass_sf[offset] = val;
}

// Host function to convert TE SF to CUTLASS SF layout
torch::Tensor convert_te_sf_to_cutlass(
    torch::Tensor te_sf,   // (rows, k_blocks) uint8 E4M3 row-major
    int64_t rows,
    int64_t cols           // original matrix cols (not k_blocks)
) {
    TORCH_CHECK(te_sf.is_cuda(), "te_sf must be CUDA");
    int k_blocks = cols / 16;  // 16 FP4 elements per scale factor
    int te_stride = te_sf.size(1);  // could be padded

    // Compute CuTe layout for SFA
    // Use a dummy N — SFA layout only depends on M, K, L
    int N_dummy = 128;
    auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        make_shape((int)rows, N_dummy, (int)cols, 1));

    // Compute output size from the CuTe layout (cosize = max offset + 1)
    int total_cutlass_size = cute::cosize(layout_sfa);

    auto cutlass_sf = torch::zeros({(int64_t)total_cutlass_size},
        torch::TensorOptions().dtype(torch::kUInt8).device(te_sf.device()));

    int total_elements = rows * k_blocks;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    convert_te_sf_to_cutlass_kernel<<<blocks, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        te_sf.data_ptr<uint8_t>(),
        cutlass_sf.data_ptr<uint8_t>(),
        layout_sfa,
        rows, k_blocks, te_stride);

    return cutlass_sf;
}

/*
 * fp4_grouped_gemm_forward: optimized with cached GPU buffers.
 *
 * Uses a single pre-allocated GPU buffer for all pointer/stride/layout arrays
 * and cudaMemcpyAsync instead of per-call DeviceAllocation+copy_from_host.
 */

// Max groups we support (3 for QKV is typical)
static constexpr int MAX_GROUPS = 8;

// Persistent GPU buffer for pointer arrays, strides, layouts
struct GpuArrays {
    ArrayElementA const* ptr_A[MAX_GROUPS];
    ArrayElementB const* ptr_B[MAX_GROUPS];
    ElementSF const* ptr_SFA[MAX_GROUPS];
    ElementSF const* ptr_SFB[MAX_GROUPS];
    CutlassElementC const* ptr_C[MAX_GROUPS];
    CutlassElementD* ptr_D[MAX_GROUPS];
    InternalStrideA strides_A[MAX_GROUPS];
    InternalStrideB strides_B[MAX_GROUPS];
    InternalLayoutSFA layouts_SFA[MAX_GROUPS];
    InternalLayoutSFB layouts_SFB[MAX_GROUPS];
    InternalStrideC strides_C[MAX_GROUPS];
    InternalStrideD strides_D[MAX_GROUPS];
    ProblemShape::UnderlyingProblemShape problem_shapes[MAX_GROUPS];
};

static GpuArrays* g_gpu_arrays = nullptr;  // device memory
static GpuArrays g_host_arrays;             // host staging

static void ensure_gpu_arrays() {
    if (g_gpu_arrays == nullptr) {
        cudaMalloc(&g_gpu_arrays, sizeof(GpuArrays));
    }
}

// Cached HW info
static bool g_hw_info_initialized = false;
static cutlass::KernelHardwareInfo g_hw_info;

std::vector<torch::Tensor> fp4_grouped_gemm_forward(
    torch::Tensor A_data,        // (M, K/2) uint8
    torch::Tensor A_sf,          // scale factors for A (CUTLASS layout)
    std::vector<torch::Tensor> B_data_list,  // per-group B data
    std::vector<torch::Tensor> B_sf_list,    // per-group B scale factors (CUTLASS layout)
    std::vector<int64_t> N_dims,  // per-group N
    int64_t M,
    int64_t K,
    double alpha                  // scale correction: 1.0/(S_enc_A * S_enc_B)
) {
    int num_groups = N_dims.size();
    TORCH_CHECK(num_groups <= MAX_GROUPS, "Too many groups (max ", MAX_GROUPS, ")");
    TORCH_CHECK(num_groups == (int)B_data_list.size(), "B list size mismatch");
    TORCH_CHECK(num_groups == (int)B_sf_list.size(), "B SF list size mismatch");
    
    auto device = A_data.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    ensure_gpu_arrays();

    // Initialize HW info once
    if (!g_hw_info_initialized) {
        g_hw_info.device_id = device.index();
        g_hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(g_hw_info.device_id);
        g_hw_info.cluster_shape = dim3(2, 1, 1);
        g_hw_info.cluster_shape_fallback = dim3(1, 1, 1);
        g_hw_info_initialized = true;
    }

    // Allocate outputs
    std::vector<torch::Tensor> outputs;
    for (int g = 0; g < num_groups; g++) {
        outputs.push_back(torch::empty({M, N_dims[g]}, 
            torch::TensorOptions().dtype(torch::kBFloat16).device(device)));
    }

    // Fill host staging buffer
    auto& h = g_host_arrays;
    for (int g = 0; g < num_groups; g++) {
        int N = (int)N_dims[g];

        h.ptr_A[g] = reinterpret_cast<ArrayElementA const*>(A_data.data_ptr());
        h.ptr_B[g] = reinterpret_cast<ArrayElementB const*>(B_data_list[g].data_ptr());
        h.ptr_SFA[g] = reinterpret_cast<ElementSF const*>(A_sf.data_ptr());
        h.ptr_SFB[g] = reinterpret_cast<ElementSF const*>(B_sf_list[g].data_ptr());
        h.ptr_C[g] = nullptr;
        h.ptr_D[g] = reinterpret_cast<CutlassElementD*>(outputs[g].data_ptr());

        h.strides_A[g] = cutlass::make_cute_packed_stride(InternalStrideA{}, make_shape((int)M, (int)K, 1));
        h.strides_B[g] = cutlass::make_cute_packed_stride(InternalStrideB{}, make_shape(N, (int)K, 1));
        h.layouts_SFA[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape((int)M, N, (int)K, 1));
        h.layouts_SFB[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape((int)M, N, (int)K, 1));
        h.strides_C[g] = cutlass::make_cute_packed_stride(InternalStrideC{}, make_shape((int)M, N, 1));
        h.strides_D[g] = cutlass::make_cute_packed_stride(InternalStrideD{}, make_shape((int)M, N, 1));
        h.problem_shapes[g] = {(int)M, N, (int)K};
    }

    // Single async H2D copy for all arrays
    cudaMemcpyAsync(g_gpu_arrays, &g_host_arrays, sizeof(GpuArrays),
                    cudaMemcpyHostToDevice, stream);

    // Build arguments using device-resident arrays
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {static_cast<int32_t>(num_groups), g_gpu_arrays->problem_shapes, h.problem_shapes},
        {g_gpu_arrays->ptr_A, g_gpu_arrays->strides_A,
         g_gpu_arrays->ptr_B, g_gpu_arrays->strides_B,
         g_gpu_arrays->ptr_SFA, g_gpu_arrays->layouts_SFA,
         g_gpu_arrays->ptr_SFB, g_gpu_arrays->layouts_SFB},
        {{static_cast<float>(alpha), 0.0f},
         g_gpu_arrays->ptr_C, g_gpu_arrays->strides_C,
         g_gpu_arrays->ptr_D, g_gpu_arrays->strides_D},
        g_hw_info
    };

    // Run GEMM
    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    ensure_workspace(workspace_size);

    auto status = gemm.initialize(arguments, g_workspace->get(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS init failed: ", cutlassGetStatusString(status));

    status = gemm.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS run failed: ", cutlassGetStatusString(status));

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fp4_grouped_gemm_forward,
          "CUTLASS grouped NVFP4 GEMM forward",
          py::arg("A_data"), py::arg("A_sf"),
          py::arg("B_data_list"), py::arg("B_sf_list"),
          py::arg("N_dims"), py::arg("M"), py::arg("K"),
          py::arg("alpha") = 1.0);
    m.def("convert_te_sf_to_cutlass", &convert_te_sf_to_cutlass,
          "Convert TE row-major E4M3 scale factors to CUTLASS SfAtom swizzled layout",
          py::arg("te_sf"), py::arg("rows"), py::arg("cols"));
}

/*
 * Standalone NVFP4 vs BF16 GEMM Benchmark for Blackwell GB200 (SM100)
 *
 * Adapted from CUTLASS example 72a_blackwell_nvfp4_bf16_gemm.cu
 * Measures raw FP4 GEMM throughput (no runtime quantisation) vs cuBLAS BF16.
 *
 * Key fixes over v1:
 *  1. cuBLAS uses random data (not zeros) for consistent algorithm selection
 *  2. gemm.initialize() called per iteration (SM100 dynamic scheduler needs this)
 *  3. Sweeps valid cluster shapes: 2x2, 2x4, 4x2, 4x4
 *     (SM100 block-scaled ops require cluster M dim even, ≥2)
 *
 * Build:
 *   cd /workspace/fp4_matmul && mkdir -p build && cd build && cmake .. && make bench_nvfp4_gemm
 *
 * Run:
 *   ./build/bench_nvfp4_gemm [--sizes=4096,8192,16384] [--iterations=100] [--warmup=20]
 */

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

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
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

#include <cublas_v2.h>

using namespace cute;

// =====================================================================
// Helpers
// =====================================================================

#define CUDA_CHECK(status) \
  { cudaError_t err = status; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
      exit(1); } }

#define CUTLASS_CHECK(status) \
  { cutlass::Status err = status; \
    if (err != cutlass::Status::kSuccess) { \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(err) << " at line " << __LINE__ << std::endl; \
      exit(1); } }

#define CUBLAS_CHECK(status) \
  { cublasStatus_t err = status; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
      std::cerr << "cuBLAS error: " << err << " at line " << __LINE__ << std::endl; \
      exit(1); } }


// =====================================================================
// NVFP4 GEMM Configuration — templated on ClusterShape
// =====================================================================

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <typename ClusterShape_>
struct NvFp4GemmConfig {
  using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag  = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag  = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD    = cutlass::bfloat16_t;
  using ElementC    = cutlass::bfloat16_t;
  using LayoutCTag  = cutlass::layout::RowMajor;
  using LayoutDTag  = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag            = cutlass::arch::Sm100;
  using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape  = Shape<_256, _256, _256>;
  using ClusterShape   = ClusterShape_;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaTileShape, ClusterShape,
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
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Valid configs: cluster M must be even ≥2 for SM100 block-scaled ops
using Config_2x2 = NvFp4GemmConfig<Shape<_2, _2, _1>>;
using Config_2x4 = NvFp4GemmConfig<Shape<_2, _4, _1>>;
using Config_4x2 = NvFp4GemmConfig<Shape<_4, _2, _1>>;
using Config_4x4 = NvFp4GemmConfig<Shape<_4, _4, _1>>;


// =====================================================================
// Benchmark: NVFP4 GEMM (templated on config)
// =====================================================================
template <typename Config>
double benchmark_nvfp4(int m, int n, int k, int iterations, int warmup, bool reinit = true) {
  using Gemm = typename Config::Gemm;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;
  using ElementD = typename Config::ElementD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using StrideA   = typename Gemm::GemmKernel::StrideA;
  using StrideB   = typename Gemm::GemmKernel::StrideB;
  using StrideC   = typename Gemm::GemmKernel::StrideC;
  using StrideD   = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_A   = make_layout(make_shape(m, k, 1), stride_A);
  auto layout_B   = make_layout(make_shape(n, k, 1), stride_B);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  // Allocate
  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A(cutlass::make_Coord(size(layout_A)));
  cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B(cutlass::make_Coord(size(layout_B)));
  cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C(cutlass::make_Coord(m * n));
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D(cutlass::make_Coord(m * n));

  // Fill with random data for realistic memory traffic
  cutlass::reference::host::TensorFillRandomUniform(block_A.host_view(), 42, 2, -2, 0);
  cutlass::reference::host::TensorFillRandomUniform(block_B.host_view(), 43, 2, -2, 0);
  for (int i = 0; i < block_SFA.capacity(); ++i) block_SFA.host_data()[i] = typename ElementA::ScaleFactorType(1.0f);
  for (int i = 0; i < block_SFB.capacity(); ++i) block_SFB.host_data()[i] = typename ElementB::ScaleFactorType(1.0f);
  cutlass::reference::host::TensorFill(block_C.host_view());
  cutlass::reference::host::TensorFill(block_D.host_view());

  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_D.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    { block_A.device_data(), stride_A,
      block_B.device_data(), stride_B,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB },
    { {1.0f, 0.0f},
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D }
  };

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return -1.0;  // Config can't handle this size
  }

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  if (reinit) {
    // Re-initialize each iteration (measures init + kernel)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    CUDA_CHECK(cudaEventRecord(stop));
  } else {
    // Initialize once, then just run the kernel (measures pure kernel throughput)
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
      CUTLASS_CHECK(gemm.run());
    }
    CUDA_CHECK(cudaEventRecord(stop));
  }
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return double(elapsed_ms) / iterations;
}


// =====================================================================
// Benchmark: BF16 GEMM via cuBLAS (with random data)
// =====================================================================
__global__ void fill_random_bf16(__nv_bfloat16* data, size_t n, unsigned int seed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    unsigned int val = (idx * 1103515245u + 12345u + seed) & 0x7FFF;
    float fval = (float(val) / 16384.0f) - 1.0f;
    data[idx] = __float2bfloat16(fval);
  }
}

double benchmark_bf16_cublas(int m, int n, int k, int iterations, int warmup) {
  __nv_bfloat16 *d_A, *d_B, *d_C;
  size_t count_A = (size_t)m * k;
  size_t count_B = (size_t)k * n;
  size_t count_C = (size_t)m * n;

  CUDA_CHECK(cudaMalloc(&d_A, count_A * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_B, count_B * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_C, count_C * sizeof(__nv_bfloat16)));

  // Random data — zeros cause cuBLAS to pick inconsistent algorithms
  int threads = 256;
  fill_random_bf16<<<(count_A + threads - 1) / threads, threads>>>(d_A, count_A, 42);
  fill_random_bf16<<<(count_B + threads - 1) / threads, threads>>>(d_B, count_B, 43);
  fill_random_bf16<<<(count_C + threads - 1) / threads, threads>>>(d_C, count_C, 44);
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  float alpha = 1.0f, beta = 0.0f;

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    CUBLAS_CHECK(cublasGemmEx(handle,
      CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
      &alpha, d_B, CUDA_R_16BF, n, d_A, CUDA_R_16BF, k,
      &beta, d_C, CUDA_R_16BF, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    CUBLAS_CHECK(cublasGemmEx(handle,
      CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
      &alpha, d_B, CUDA_R_16BF, n, d_A, CUDA_R_16BF, k,
      &beta, d_C, CUDA_R_16BF, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return double(elapsed_ms) / iterations;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED


// =====================================================================
// Main
// =====================================================================
int main(int argc, char const **args) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cerr << "This benchmark requires SM100 (Blackwell) support." << std::endl;
  return 1;
#else

  cudaDeviceProp props;
  int dev;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
  if (props.major != 10) {
    std::cerr << "Requires Blackwell GPU (SM 10.x). Found: SM "
              << props.major << "." << props.minor << std::endl;
    return 1;
  }

  cutlass::CommandLine cmd(argc, args);

  int iterations = 100;
  int warmup = 20;
  std::string sizes_str = "4096,8192,12288,16384";

  cmd.get_cmd_line_argument("iterations", iterations);
  cmd.get_cmd_line_argument("warmup", warmup);
  cmd.get_cmd_line_argument("sizes", sizes_str);

  std::vector<int> sizes;
  std::stringstream ss(sizes_str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    sizes.push_back(std::stoi(item));
  }

  std::cout << "=============================================================" << std::endl;
  std::cout << " NVFP4 vs BF16 GEMM Benchmark (GB200)" << std::endl;
  std::cout << " GPU: " << props.name << " (" << props.multiProcessorCount << " SMs)" << std::endl;
  std::cout << " Iterations: " << iterations << ", Warmup: " << warmup << std::endl;
  std::cout << " Tile: 256x256x256, Cluster: 2x2 (best from sweep)" << std::endl;
  std::cout << " cuBLAS data: random (not zeros)" << std::endl;
  std::cout << "=============================================================" << std::endl;

  std::cout << "\n" << std::setw(8) << "Size"
            << std::setw(12) << "BF16 ms"
            << std::setw(12) << "BF16 TF"
            << std::setw(12) << "FP4+init"
            << std::setw(12) << "FP4 pure"
            << std::setw(12) << "FP4+i TF"
            << std::setw(12) << "FP4p TF"
            << std::setw(10) << "Sp+init"
            << std::setw(10) << "Sp pure"
            << std::setw(12) << "init OH"
            << std::endl;
  std::cout << std::string(110, '-') << std::endl;

  for (int sz : sizes) {
    int m = sz, n = sz, k = sz;
    double flops = 2.0 * m * n * k;

    double bf16_ms = benchmark_bf16_cublas(m, n, k, iterations, warmup);
    double fp4_init_ms = benchmark_nvfp4<Config_2x2>(m, n, k, iterations, warmup, true);   // with init
    double fp4_pure_ms = benchmark_nvfp4<Config_2x2>(m, n, k, iterations, warmup, false);  // pure kernel

    double bf16_tf  = flops / (bf16_ms * 1e-3) / 1e12;
    double fp4i_tf  = flops / (fp4_init_ms * 1e-3) / 1e12;
    double fp4p_tf  = flops / (fp4_pure_ms * 1e-3) / 1e12;
    double sp_init  = bf16_ms / fp4_init_ms;
    double sp_pure  = bf16_ms / fp4_pure_ms;
    double init_oh  = (fp4_init_ms - fp4_pure_ms) / fp4_init_ms * 100.0;

    std::cout << std::setw(8) << sz
              << std::setw(12) << std::fixed << std::setprecision(3) << bf16_ms
              << std::setw(12) << std::setprecision(1) << bf16_tf
              << std::setw(12) << std::setprecision(3) << fp4_init_ms
              << std::setw(12) << std::setprecision(3) << fp4_pure_ms
              << std::setw(12) << std::setprecision(1) << fp4i_tf
              << std::setw(12) << std::setprecision(1) << fp4p_tf
              << std::setw(9)  << std::setprecision(2) << sp_init << "x"
              << std::setw(9)  << std::setprecision(2) << sp_pure << "x"
              << std::setw(11) << std::setprecision(1) << init_oh << "%"
              << std::endl;
  }

  std::cout << "\nColumn legend:" << std::endl;
  std::cout << "  FP4+init = FP4 with gemm.initialize() per iteration (SM100 convention)" << std::endl;
  std::cout << "  FP4 pure = FP4 with single init, kernel-only timing" << std::endl;
  std::cout << "  init OH  = % overhead from per-iteration initialize()" << std::endl;
  std::cout << "  TF = TFLOPS (equivalent BF16: 2*M*N*K/time)" << std::endl;
  std::cout << "\nTheoretical max speedup = 4x. GB200 has " << props.multiProcessorCount << " SMs." << std::endl;

  return 0;
#endif
}


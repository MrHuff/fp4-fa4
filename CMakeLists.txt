cmake_minimum_required(VERSION 3.31)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 100a)
endif()

project(fp4_fa4_auxiliary_gemm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(
  CUTLASS_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/cutlass"
  CACHE PATH
  "Path to the pinned CUTLASS checkout"
)
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(
    FATAL_ERROR
    "CUTLASS was not found at ${CUTLASS_ROOT}. "
    "Initialize submodules or pass -DCUTLASS_ROOT=/path/to/cutlass."
  )
endif()

set(CUTLASS_INCLUDE_DIR "${CUTLASS_ROOT}/include")
set(CUTLASS_TOOLS_INC "${CUTLASS_ROOT}/tools/util/include")
set(CUTLASS_EXAMPLES_COMMON "${CUTLASS_ROOT}/examples/common")
set(CUTLASS_TUTORIAL_BLACKWELL "${CUTLASS_ROOT}/examples/cute/tutorial/blackwell")

set(FP4_CUTLASS_INCLUDE_DIRS
  ${CUTLASS_INCLUDE_DIR}
  ${CUTLASS_TOOLS_INC}
  ${CUTLASS_EXAMPLES_COMMON}
  ${CUTLASS_TUTORIAL_BLACKWELL}
)

set(FP4_CUDA_FLAGS
  -std=c++17
  -O3
  -lineinfo
  --expt-relaxed-constexpr
  --ptxas-options=-v
)

add_executable(bench_nvfp4_gemm benchmarks/bench_nvfp4_gemm.cu)
set_property(TARGET bench_nvfp4_gemm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(bench_nvfp4_gemm PRIVATE ${FP4_CUTLASS_INCLUDE_DIRS})
target_link_libraries(bench_nvfp4_gemm PRIVATE CUDA::cudart CUDA::cublas cudadevrt)
target_compile_options(
  bench_nvfp4_gemm PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:${FP4_CUDA_FLAGS}>"
)

add_executable(bench_grouped_nvfp4_gemm benchmarks/bench_grouped_nvfp4_gemm.cu)
set_property(TARGET bench_grouped_nvfp4_gemm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(bench_grouped_nvfp4_gemm PRIVATE ${FP4_CUTLASS_INCLUDE_DIRS})
target_link_libraries(
  bench_grouped_nvfp4_gemm PRIVATE CUDA::cudart CUDA::cublas cudadevrt
)
target_compile_options(
  bench_grouped_nvfp4_gemm PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:${FP4_CUDA_FLAGS}>"
)

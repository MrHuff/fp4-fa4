#pragma once

#pragma once
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cuda.h>

#include "cutlass/cutlass.h"

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    if (error != cutlass::Status::kSuccess) {       \
      printf("CUTLASS Fail: %s:%d %s\n", __FILE__, __LINE__, cutlassGetStatusString(error)); \
      exit(1); \
    }                                               \
  }

#define CUDA_CHECK(status)                                        \
  {                                                               \
    cudaError_t error = status;                                   \
    if (error != cudaSuccess) {                                   \
      printf("CUDA Fail: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(1); \
    }                                                             \
  }

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

int32_t get_sm_version_num();

/**
 * A wrapper for a kernel that is used to guard against compilation on
 * architectures that will never use the kernel. The purpose of this is to
 * reduce the size of the compiled binary.
 * __CUDA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
 * into code that will be executed on the device where it is defined.
 */
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};
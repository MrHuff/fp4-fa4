#pragma once

// Flattened owner-major schedule for four-Hq-head/Hkv ownership.  This keeps
// the production BSHD V382 math and submits every Hkv head for the longest
// causal K256 owner before moving to shorter owners.  The main kernel writes
// one disjoint FP32 Hkv result, followed only by BF16 conversion.

#include "v382_d64_gqa_hkv2_partial.cuh"

namespace tkfa4::native_gqa_tk_bwd::v382_d64_hkv4_owner_major {

namespace common = v382_d64_hkv2_partial;

using common::kDepth;
using common::kHeadRatio;
using common::kKvHeads;
using common::kOwnerClusters;
using common::kQueryHeads;
using common::kSequence;
using common::kThreads;
using common::partial_globals;

constexpr int kHeadsPerOwner = kHeadRatio;
constexpr int kPartialHeads = kKvHeads;

__global__ __launch_bounds__(256, 2)
void finalize_hkv_kernel(
    const float *__restrict__ dk_accum,
    const float *__restrict__ dv_accum,
    bf16 *__restrict__ dk,
    bf16 *__restrict__ dv,
    int64_t elements
) {
    for (
        int64_t index =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x
    ) {
        dk[index] = __float2bfloat16_rn(dk_accum[index]);
        dv[index] = __float2bfloat16_rn(dv_accum[index]);
    }
}

inline void launch_owner_major_bf16(
    const partial_globals &globals,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(
            kOwnerClusters * kPartialHeads * 2,
            1,
            globals.batch
        ),
        dim3(kThreads, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        common::owner_partial_bf16_kernel<kHeadsPerOwner, true>,
        globals
    ));
}

inline void launch_finalize(
    const at::Tensor &dq_accum,
    at::Tensor &dq,
    const at::Tensor &dk_accum,
    const at::Tensor &dv_accum,
    at::Tensor &dk,
    at::Tensor &dv,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t dq_elements = dq.numel();
    const int64_t kv_elements = dk.numel();
    common::finalize_dq_kernel<<<
        static_cast<unsigned int>((dq_elements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dq_accum.data_ptr()),
        reinterpret_cast<bf16 *>(dq.data_ptr()),
        dq_elements
    );
    CUDACHECK(cudaGetLastError());
    finalize_hkv_kernel<<<
        static_cast<unsigned int>((kv_elements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dk_accum.data_ptr()),
        reinterpret_cast<const float *>(dv_accum.data_ptr()),
        reinterpret_cast<bf16 *>(dk.data_ptr()),
        reinterpret_cast<bf16 *>(dv.data_ptr()),
        kv_elements
    );
    CUDACHECK(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_hkv4_owner_major

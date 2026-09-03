#pragma once

// Production-BSHD Hq-parallel control for the V382 D64/GQA owner map.
// This instantiates the same native dim::DEPTH compute used by the fused
// owner experiments with exactly one Hq head per cluster, then reduces four
// disjoint Hq FP32 dK/dV partials into each Hkv output.

#include "v382_d64_gqa_hkv2_partial.cuh"

namespace tkfa4::native_gqa_tk_bwd::v382_d64_hq_bshd {

namespace common = v382_d64_hkv2_partial;

using common::kDepth;
using common::kHeadRatio;
using common::kKvHeads;
using common::kOwnerClusters;
using common::kQueryHeads;
using common::kSequence;
using common::kThreads;
using common::partial_globals;

constexpr int kHeadsPerOwner = 1;
constexpr int kPartialHeads = kQueryHeads;

__global__ __launch_bounds__(256, 2)
void reduce_hq_partials_kernel(
    const float *__restrict__ dk_partial,
    const float *__restrict__ dv_partial,
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
        const int depth = static_cast<int>(index % kDepth);
        const int64_t row = index / kDepth;
        const int kv_head = static_cast<int>(row % kKvHeads);
        const int64_t batch_sequence = row / kKvHeads;
        const int64_t sequence = batch_sequence % kSequence;
        const int64_t batch = batch_sequence / kSequence;
        const int first_query_head = kv_head * kHeadRatio;
        float dk_sum = 0.0f;
        float dv_sum = 0.0f;
#pragma unroll
        for (int ratio = 0; ratio < kHeadRatio; ++ratio) {
            const int query_head = first_query_head + ratio;
            const int64_t partial_index =
                ((batch * kSequence + sequence) * kPartialHeads +
                    query_head) *
                    kDepth +
                depth;
            dk_sum += dk_partial[partial_index];
            dv_sum += dv_partial[partial_index];
        }
        dk[index] = __float2bfloat16_rn(dk_sum);
        dv[index] = __float2bfloat16_rn(dv_sum);
    }
}

inline void launch_hq_owner_bf16(
    const partial_globals &globals,
    cudaStream_t stream
) {
    kittens::LaunchConfig<true, false> launch_config(
        dim3(kOwnerClusters * 2, kQueryHeads, globals.batch),
        dim3(kThreads, 1, 1),
        0,
        stream,
        dim3(2, 1, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        common::owner_partial_bf16_kernel<kHeadsPerOwner>,
        globals
    ));
}

inline void launch_finalize(
    const at::Tensor &dq_accum,
    at::Tensor &dq,
    const at::Tensor &dk_partial,
    const at::Tensor &dv_partial,
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
    reduce_hq_partials_kernel<<<
        static_cast<unsigned int>((kv_elements + kBlock - 1) / kBlock),
        kBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float *>(dk_partial.data_ptr()),
        reinterpret_cast<const float *>(dv_partial.data_ptr()),
        reinterpret_cast<bf16 *>(dk.data_ptr()),
        reinterpret_cast<bf16 *>(dv.data_ptr()),
        kv_elements
    );
    CUDACHECK(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_hq_bshd

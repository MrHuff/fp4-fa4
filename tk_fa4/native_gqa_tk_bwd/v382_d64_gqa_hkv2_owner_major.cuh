#pragma once

// Flattened owner-major schedule for the parity-green two-head partial path.
// The compute and disjoint group-2 epilogue are unchanged.  Only cluster
// linearization changes: all partial-head owners for causal owner 0 are
// submitted before owner 1, so the longest tasks enter the scheduler first.

#include "v382_d64_gqa_hkv2_partial.cuh"

namespace tkfa4::native_gqa_tk_bwd::v382_d64_hkv2_owner_major {

namespace common = v382_d64_hkv2_partial;

using common::kDepth;
using common::kHeadsPerSplit;
using common::kKvHeads;
using common::kOwnerClusters;
using common::kPartialHeads;
using common::kQueryHeads;
using common::kSequence;
using common::kThreads;
using common::partial_globals;

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
        common::owner_partial_bf16_kernel<kHeadsPerSplit, true>,
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
    common::launch_finalize(
        dq_accum,
        dq,
        dk_partial,
        dv_partial,
        dk,
        dv,
        stream
    );
}

}  // namespace tkfa4::native_gqa_tk_bwd::v382_d64_hkv2_owner_major

#pragma once

#include "v441_d128_gqa_e4m3_b2_s4096_owner4_direct_dkdv_production_bshd.cuh"

// B1/S4096 dispatch experiment reusing the frozen v441 owner-4 kernel.  The
// device kernel is batch-dynamic: blockIdx.x is decoded into (batch, owner),
// and every global-memory coordinate carries that decoded batch.  With one
// owner per KV head, each (batch, KV head, key tile) dK/dV destination still
// has exactly one writer at B1.  All non-B1/S4096 shapes retain v441 dispatch.
namespace tkfa4::native_gqa_tk_bwd::v442_d128_gqa_e4m3_b1_s4096_owner4_direct_dkdv_production_bshd {

namespace fallback =
    tkfa4::native_gqa_tk_bwd::v441_d128_gqa_e4m3_b2_s4096_owner4_direct_dkdv_production_bshd;
namespace prior = fallback::prior;

using fallback::globals;
using fallback::kHeadOwners;
using fallback::kHeadsPerOwner;
using fallback::kHeadRatio;
using fallback::kKeyTile;
using fallback::kKvHeads;
using fallback::kQueryHeads;
using fallback::kQueryTile;
using fallback::kThreads;
using fallback::shared_storage;

inline void launch(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lstat,
    at::Tensor &dstat,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float softmax_scale,
    cudaStream_t stream
) {
    if (
        q.size(0) != 1 ||
        q.size(1) != prior::kExactSequence
    ) {
        fallback::launch(
            q,
            k,
            v,
            dout,
            lstat,
            dstat,
            dq,
            dk,
            dv,
            softmax_scale,
            stream
        );
        return;
    }

    const globals g = fallback::make_globals(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        dq,
        dk,
        dv,
        softmax_scale
    );
    const dim3 grid(
        static_cast<unsigned int>(kHeadOwners * q.size(0)),
        static_cast<unsigned int>(q.size(1) / kKeyTile),
        1
    );
    fallback::owner4_kernel<<<grid, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v442_d128_gqa_e4m3_b1_s4096_owner4_direct_dkdv_production_bshd

#pragma once

#include "v442_d128_gqa_e4m3_b1_s4096_owner4_direct_dkdv_production_bshd.cuh"
#include "v443_d128_gqa_e4m3_b2_s4096_owner4_compact_p_reuse_production_bshd.cuh"

// B1/S4096 dispatch experiment composing v442's validated batch-dynamic
// owner-4/unique-writer route with v443's exact rounded-E4M3 compact-P kernel.
// Every other shape retains v442 dispatch, including v441's non-compact
// B2/S4096 owner-4 route and its established non-exact fallbacks.
namespace tkfa4::native_gqa_tk_bwd::v451_d128_gqa_e4m3_b1_s4096_owner4_compact_p_reuse_direct_dkdv_production_bshd {

namespace fallback =
    tkfa4::native_gqa_tk_bwd::v442_d128_gqa_e4m3_b1_s4096_owner4_direct_dkdv_production_bshd;
namespace compact =
    tkfa4::native_gqa_tk_bwd::v443_d128_gqa_e4m3_b2_s4096_owner4_compact_p_reuse_production_bshd;
namespace prior = compact::prior;

using compact::globals;
using compact::kCompactProbabilityWords;
using compact::kHeadOwners;
using compact::kHeadsPerOwner;
using compact::kHeadRatio;
using compact::kKeyTile;
using compact::kKvHeads;
using compact::kOwnersPerKvHead;
using compact::kQueryHeads;
using compact::kQueryTile;
using compact::kThreads;
using compact::shared_storage;

static_assert(prior::kExactSequence == 4096);
static_assert(kQueryHeads == 32 && kKvHeads == 8);
static_assert(kHeadsPerOwner == 4 && kHeadOwners == 8);
static_assert(kOwnersPerKvHead == 1);

inline bool is_b1_exact_route(long long batch, long long sequence) {
    return batch == 1 && sequence == prior::kExactSequence;
}

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
    if (!is_b1_exact_route(q.size(0), q.size(1))) {
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

    const globals g = compact::make_globals(
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
    compact::owner4_kernel<<<grid, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v451_d128_gqa_e4m3_b1_s4096_owner4_compact_p_reuse_direct_dkdv_production_bshd

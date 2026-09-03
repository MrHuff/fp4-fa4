#pragma once

#include "v439_d128_gqa_e4m3_b1_owner2_exact_s4096_production_bshd.cuh"
#include "v443_d128_gqa_e4m3_b2_s4096_owner4_compact_p_reuse_production_bshd.cuh"

// Frozen best-route dispatcher.  It adds no device implementation: exact
// B1/S4096 launches v439's owner-2 kernel, while exact B2/S4096 and every
// non-exact fallback are delegated to v443.  Consequently v443 preserves the
// frozen v436 B1 fallback and frozen v437 B2 fallback without cloning either
// kernel or changing their launch geometry.
namespace tkfa4::native_gqa_tk_bwd::v444_d128_gqa_e4m3_unified_best_route_production_bshd {

namespace b1_exact =
    tkfa4::native_gqa_tk_bwd::v439_d128_gqa_e4m3_b1_owner2_exact_s4096_production_bshd;
namespace b2_exact_and_fallbacks =
    tkfa4::native_gqa_tk_bwd::v443_d128_gqa_e4m3_b2_s4096_owner4_compact_p_reuse_production_bshd;
namespace b1_fallback =
    tkfa4::native_gqa_tk_bwd::v436_d128_gqa_e4m3_exact_s4096_runtime_accumulate_production_bshd;
namespace b2_fallback =
    tkfa4::native_gqa_tk_bwd::v437_d128_gqa_e4m3_b2_owner2_production_bshd;

using b1_fallback::kDepth;
using b1_fallback::kHeadRatio;
using b1_fallback::kKeyTile;
using b1_fallback::kKvHeads;
using b1_fallback::kOperandScale;
using b1_fallback::kQueryHeads;
using b1_fallback::kQueryTile;
using b1_fallback::kThreads;

constexpr int kExactSequence = b1_fallback::kExactSequence;
static_assert(kExactSequence == 4096);
static_assert(b1_exact::kExactSequence == kExactSequence);
static_assert(b2_exact_and_fallbacks::prior::kExactSequence == kExactSequence);
static_assert(b1_exact::kHeadsPerOwner == 2);
static_assert(b2_exact_and_fallbacks::kHeadsPerOwner == 4);

inline bool is_b1_exact_route(long long batch, long long sequence) {
    return batch == 1 && sequence == kExactSequence;
}

inline bool is_b2_exact_direct_route(long long batch, long long sequence) {
    return batch == 2 && sequence == kExactSequence;
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
    if (is_b1_exact_route(q.size(0), q.size(1))) {
        b1_exact::launch(
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

    // v443 selects its owner-4/direct-dK-dV kernel for B2/S4096,
    // v436 for non-exact B1, and v437 for non-exact B2.
    b2_exact_and_fallbacks::launch(
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
}

}  // namespace tkfa4::native_gqa_tk_bwd::v444_d128_gqa_e4m3_unified_best_route_production_bshd

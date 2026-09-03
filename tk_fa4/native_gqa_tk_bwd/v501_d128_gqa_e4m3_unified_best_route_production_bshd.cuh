#pragma once

#include "v488_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_split_dq_tmem_release_production_bshd.cuh"
#include "v490_d128_gqa_e4m3_b2_s4096_owner4_warp14_gradient_publisher_production_bshd.cuh"

// Thin production dispatcher freezing the independently selected exact-shape
// winners.  B1/S4096 launches v488's owner-2 compact-P route, whose complete
// BF16 dQ capture releases aliased TMEM before shared publication.  B2/S4096
// launches v490's owner-4/direct-dK-dV route, where dedicated warp 14 owns the
// complete dQ/dK/dV TMA publication stream.  Non-exact B1 and B2 continue to
// select frozen v436 and v437, respectively, through v490.  This router adds
// no device implementation and changes neither numerical operations nor
// launch geometry.
namespace tkfa4::native_gqa_tk_bwd::v501_d128_gqa_e4m3_unified_best_route_production_bshd {

namespace b1_exact =
    tkfa4::native_gqa_tk_bwd::v488_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_split_dq_tmem_release_production_bshd;
namespace b2_exact_and_fallbacks =
    tkfa4::native_gqa_tk_bwd::v490_d128_gqa_e4m3_b2_s4096_owner4_warp14_gradient_publisher_production_bshd;
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

    // v490 selects exact B2's owner-4 kernel with its dedicated warp-14
    // gradient publisher.  It retains frozen v436 for non-exact B1 and frozen
    // v437 for non-exact B2.
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

}  // namespace tkfa4::native_gqa_tk_bwd::v501_d128_gqa_e4m3_unified_best_route_production_bshd

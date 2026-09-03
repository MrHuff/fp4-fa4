// Candidate-only B1/S4096/H64 row-parallel K/V multicast discriminator.
// Each rank retains the deployed I73 CTA-local arithmetic and all tensor
// operations remain cta_group::1.  Only K/K-scale and V/V-scale TMA payloads
// cross the two-CTA cluster.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_SUPPRESS_B300_CAUSAL_DISPATCH 1
#define TK_FA4_LARGE_I73_UNIFORM_KV_ONLY_BUILD 1

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "fwd_bf16_baseline.inc"
#include "stage2_ex2_alu_helpers.cuh"
#include "fwd_configs.inc"

using large_i73_uniform_kv_canonical_base =
    config_fp4pv_stage2_ex2_alu_pchain_c_head_interleaved_lazy_output_rescale<
        16, 128, 128, 192, 128, 200, 56, 112, 2>;

struct config_fp4pv_large_i73_uniform_kv_only
    : public large_i73_uniform_kv_canonical_base {
    // The inherited head-interleaved map is cluster-incoherent.  The kernel's
    // exact uniform-pair branch maps both ranks to one head and adjacent M128s.
    static constexpr bool ONLINE_Q4_HEAD_INTERLEAVED_MAPPING = false;
    static constexpr bool ONLINE_ROWPAR2_RANKLOCAL = true;
    static constexpr bool ONLINE_ROWPAR2_SHARED_K_LIFETIME = true;
    static constexpr bool ONLINE_ROWPAR2_V_MULTICAST = true;
    static constexpr bool ONLINE_ROWPAR2_SYMMETRIC_LOCAL_K_COMPLETION = true;
    static constexpr bool ONLINE_ROWPAR2_UNIFORM_PAIR_MAX_ITERS = true;
    static constexpr bool ONLINE_ROWPAR2_CANONICAL_LOCAL_FASTPATH = true;
    static constexpr bool ONLINE_ROWPAR2_PACKED_V_OWNER_PHASES = true;
};

// Ownership-correct successor to the immutable v1 discriminator.  Rank 0 is
// the sole K/K-scale multicast owner.  It acquires the count-2 reuse
// generation (one completed QK use per rank) before it arms either local or
// remote destination for the next transaction.  P/P-scale/PV/output remain
// strictly CTA local.
struct config_fp4pv_large_i73_uniform_kv_owned
    : public config_fp4pv_large_i73_uniform_kv_only {
    static constexpr bool
        ONLINE_ROWPAR2_DEFER_K_SC_SHARED_UNTIL_LOCAL_QK_COMPLETION = true;
    static constexpr bool
        ONLINE_ROWPAR2_COMBINED_SYMMETRIC_DEFERRED_K_COMPLETION = true;
    static constexpr bool
        ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_PAYLOAD_MBAR = true;
    static constexpr bool
        ONLINE_ROWPAR2_OWNER_ARMS_REMOTE_K_SCALE_MBAR = true;
    static constexpr bool ONLINE_ROWPAR2_K_PAYLOAD_ARM_AFTER_REUSE = true;
    static constexpr bool ONLINE_ROWPAR2_K_SCALE_ARM_AFTER_REUSE = true;
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_SCALE_HANDOFF = true;
    static constexpr bool ONLINE_ROWPAR2_CLUSTER_SCOPE_K_REUSE = true;
};

#include "fwd_device_helpers.inc"
#include "fwd_option_b_nb64_dvo128.inc"
#include "fwd_cluster_p_pipeline.inc"
#include "fwd_local_p_stage_ladder.inc"
#include "fwd_tmem224_alternatives.inc"
#include "fwd_raw_fp4_factor_probe.inc"
#include "fwd_raw_fp4_throughput_probe.inc"
#include "fwd_mxfp4_qk_cta_group_ab_probe.inc"
#include "fwd_scaled_k64_two_query_ceiling.inc"
#include "fwd_mxfp4_sfid_probe.inc"
#include "fwd_tmem_output_checkpoint_probe.inc"
#include "fwd_register_output_probe.inc"
#include "fwd_raw_fp4_two_query_probe.inc"
#include "fwd_streaming_kernel.inc"

#include "shared_host_helpers.inc"
#include "large_i73_uniform_kv_minimal_host.inc"

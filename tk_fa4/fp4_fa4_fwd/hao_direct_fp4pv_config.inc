#pragma once

// Direct TK translation of the HAO SM100 forward topology.  Keep arithmetic
// knobs separate from the ownership protocol so PV formats can share one
// scheduler and barrier lifecycle.
#ifndef TK_HAO_DIRECT_HEAD_DIM
#define TK_HAO_DIRECT_HEAD_DIM 128
#endif

#ifndef TK_HAO_DIRECT_NUM_SM
#define TK_HAO_DIRECT_NUM_SM 152
#endif

#ifndef TK_HAO_DIRECT_FP4PV_TWO_RESIDENT_D128
// SM103 can issue scaled FP4 with N96/K96. One N96 score tile, one D128
// output, and two disjoint 16-column scale pages fit in 256 TMEM columns.
#define TK_HAO_DIRECT_FP4PV_TWO_RESIDENT_D128 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_N96_DUAL_QUERY
// SM103 K96-Ultra control that retains the production one-CTA/two-query
// ownership model. Unlike the two-resident experiment, this does not double
// the number of persistent logical jobs.
#define TK_HAO_DIRECT_FP4PV_N96_DUAL_QUERY 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_DOUBLE_SCORE_SINGLE_QUERY
// One M128 query owns two alternating N128 score banks and one shared D128
// output. This is the fixed-P lifecycle gate for true QK/PV ping-pong.
#define TK_HAO_DIRECT_FP4PV_DOUBLE_SCORE_SINGLE_QUERY 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_D64_DETACHED_P
// Each spare 64-column D64 slab holds packed P in [0,16), the V scale page in
// [16,32), and the two P scale pages in [32,64).
#define TK_HAO_DIRECT_FP4PV_D64_DETACHED_P 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_D64_P_BUFFER_MODE
#define TK_HAO_DIRECT_FP4PV_D64_P_BUFFER_MODE 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_GROUP2
// SM103 Ultra route: one M128 query tile is owned by each rank of a two-CTA
// cluster. QK and PV are collective cta_group::2 operations, while score
// processing, P packing, correction, and output storage remain rank-local.
#define TK_HAO_DIRECT_FP4PV_SM103_GROUP2 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_GROUP2_DUAL_QUERY
// Keep two rank-local score/output stages live in each CTA while group2 QK
// and PV span the pair. Q/K scales borrow the opposite score bank exactly as
// in the production one-CTA scheduler, so the layout remains 512 columns.
#define TK_HAO_DIRECT_FP4PV_SM103_GROUP2_DUAL_QUERY 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N256
// SM103 wide-key fixed-P topology. N256 uses K96+K96+K64 PV; the selectable
// N192 gate uses K96+K96 and leaves dedicated TMEM columns for scale pages.
#define TK_HAO_DIRECT_FP4PV_SM103_N256 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_WIDE_KEY_TILE
#define TK_HAO_DIRECT_FP4PV_SM103_WIDE_KEY_TILE 256
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N192_EARLY_K96
// Publish the first three packed P quarters as one complete K96 operand.
// The issuer overlaps that PV operation with construction of quarters 3-5.
#define TK_HAO_DIRECT_FP4PV_SM103_N192_EARLY_K96 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N192_DEDICATED_P0
// Place stage-0 packed P in the otherwise-unused N192 tail columns. QK1 can
// then run before PV0 and expose stage-1 score work while PV0 is in flight.
#define TK_HAO_DIRECT_FP4PV_SM103_N192_DEDICATED_P0 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N256_DUAL_QUERY
// Preserve two query outputs while a wide N256 score consumes the remaining
// TMEM. Folded Q/K scales temporarily borrow a 16-column output prefix.
#define TK_HAO_DIRECT_FP4PV_SM103_N256_DUAL_QUERY 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N256_PREFIX_REGS
// Hold the borrowed output prefix in the stage-1 reader registers instead of
// round-tripping it through shared memory between QK and PV.
#define TK_HAO_DIRECT_FP4PV_SM103_N256_PREFIX_REGS 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N256_DIRECT_PV_HANDOFF
// Publish PV1 completion directly to the prefix reader. This removes the
// issuer-side wait-and-arrive relay from every N256 key-tile transition.
#define TK_HAO_DIRECT_FP4PV_SM103_N256_DIRECT_PV_HANDOFF 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N256_CHAIN_QK1
// Queue QK1 immediately behind PV0, then retire PV0 while stage 1 processes
// its score. The single issuer preserves the overlapping TMEM access order.
#define TK_HAO_DIRECT_FP4PV_SM103_N256_CHAIN_QK1 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N256_SPLIT_OUTPUT_SCALES
// Borrow each output's own 16-column prefix for its QK scale page. The two
// softmax readers preserve their prefixes independently, allowing the next
// QK issue to queue behind PV without a cross-stage completion relay.
#define TK_HAO_DIRECT_FP4PV_SM103_N256_SPLIT_OUTPUT_SCALES 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N256_COMPACT_V
// Store the 256 useful V keys once instead of padding three K96/K96/K64
// segments to K128. Custom descriptors start at packed byte offsets 0, 48,
// and 96 while the existing three scale pages remain unchanged.
#define TK_HAO_DIRECT_FP4PV_SM103_N256_COMPACT_V 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_SM103_N256_ASYNC_PREFIX_RESTORE
// Queue the saved output-prefix restore before P stores and retire both with
// one TMEM-store wait. QK is complete at this point, so its scale page is dead.
#define TK_HAO_DIRECT_FP4PV_SM103_N256_ASYNC_PREFIX_RESTORE 0
#endif

static_assert(
    (TK_HAO_DIRECT_FP4PV_TWO_RESIDENT_D128 +
     TK_HAO_DIRECT_FP4PV_N96_DUAL_QUERY +
     TK_HAO_DIRECT_FP4PV_DOUBLE_SCORE_SINGLE_QUERY +
     TK_HAO_DIRECT_FP4PV_SM103_GROUP2 +
     TK_HAO_DIRECT_FP4PV_SM103_N256) <= 1,
    "D128 topology controls are mutually exclusive");

#ifndef TK_HAO_DIRECT_N96_LIFECYCLE_DEBUG
#define TK_HAO_DIRECT_N96_LIFECYCLE_DEBUG 0
#endif

#ifndef TK_HAO_DIRECT_N96_SPLIT_KV_TMA
// Split payload and scale completion was useful for lifecycle diagnosis, but
// the production path can fold both transactions into one barrier.
#define TK_HAO_DIRECT_N96_SPLIT_KV_TMA 0
#endif

#ifndef TK_HAO_DIRECT_N96_STAGGER_NS
// Delay the issuer in the second resident CTA cohort so its tensor-core work
// can fill the first cohort's P-construction gaps.
#define TK_HAO_DIRECT_N96_STAGGER_NS 0
#endif

struct hao_direct_arithmetic_base
    : public config_fp4pv<
          128,
          TK_HAO_DIRECT_FP4PV_SM103_N256
              ? TK_HAO_DIRECT_FP4PV_SM103_WIDE_KEY_TILE
              : ((TK_HAO_DIRECT_FP4PV_TWO_RESIDENT_D128 ||
                  TK_HAO_DIRECT_FP4PV_N96_DUAL_QUERY) ? 96 : 128),
          TK_HAO_DIRECT_HEAD_DIM,
          TK_HAO_DIRECT_HEAD_DIM,
          TK_HAO_DIRECT_FP4PV_SM103_GROUP2 ? 2 : 1> {
    static constexpr bool ONLINE_FULL_K_TILE = true;
};

#ifndef TK_HAO_DIRECT_SEQ_LEN
#define TK_HAO_DIRECT_SEQ_LEN 4096
#endif
#ifndef TK_HAO_DIRECT_HEADS
#define TK_HAO_DIRECT_HEADS 24
#endif
#ifndef TK_HAO_DIRECT_KV_HEADS
#define TK_HAO_DIRECT_KV_HEADS TK_HAO_DIRECT_HEADS
#endif
#ifndef TK_HAO_DIRECT_BATCH
#define TK_HAO_DIRECT_BATCH 1
#endif
#ifndef TK_HAO_DIRECT_CAUSAL
#define TK_HAO_DIRECT_CAUSAL 0
#endif
#ifndef TK_HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP
#define TK_HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP 0
#endif
#ifndef TK_HAO_DIRECT_CAUSAL_DIAGONAL_MASK
// Causal scheduling excludes future key tiles. Test the tile coordinate once
// and use lane-local coordinates only for the surviving diagonal tile.
#define TK_HAO_DIRECT_CAUSAL_DIAGONAL_MASK 0
#endif
#ifndef TK_HAO_DIRECT_CAUSAL_PAIR_QUARTER_MASK
#define TK_HAO_DIRECT_CAUSAL_PAIR_QUARTER_MASK 0
#endif
#ifndef TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV
#define TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV 0
#endif
#ifndef TK_HAO_DIRECT_KV_STAGES
#define TK_HAO_DIRECT_KV_STAGES 13
#endif
#ifndef TK_HAO_DIRECT_PHYSICAL_GRID_CAP
// Zero selects all SMs; positive values cap the persistent CTA count.
#define TK_HAO_DIRECT_PHYSICAL_GRID_CAP 0
#endif
#ifndef TK_HAO_DIRECT_TASK_ORDER
// 0 keeps the legacy head-fast order; 1 matches HAO's query-block-fast order.
#define TK_HAO_DIRECT_TASK_ORDER 0
#endif
#ifndef TK_HAO_DIRECT_PV_SCALE_MODE
// 0: NVFP4 E4M3 scale per 16 values
// 1: MXFP4 E8M0 scale per 32 values
#define TK_HAO_DIRECT_PV_SCALE_MODE 0
#endif
#ifndef TK_HAO_DIRECT_QK_SCALE_MODE
// 0: NVFP4 E4M3 scale per 16 values
// 1: MXFP4 E8M0 scale per 32 values
#define TK_HAO_DIRECT_QK_SCALE_MODE 0
#endif
#ifndef TK_HAO_DIRECT_WAIT_HINT
// Match the suspend-time hint emitted by HAO's CUTE DSL pipeline on SM100.
#define TK_HAO_DIRECT_WAIT_HINT 10000000
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SM103_LDRED
// SM103 can return a per-row maximum with each x32 TMEM score load.
#define TK_HAO_DIRECT_FP4PV_SM103_LDRED 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SM103_LDRED_MASK
// One bit per score quarter (Q0..Q3). This only takes effect when LDRED is on.
#define TK_HAO_DIRECT_FP4PV_SM103_LDRED_MASK 15
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SM103_LDRED_STAGE0_MASK
#define TK_HAO_DIRECT_FP4PV_SM103_LDRED_STAGE0_MASK \
    TK_HAO_DIRECT_FP4PV_SM103_LDRED_MASK
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SM103_LDRED_STAGE1_MASK
#define TK_HAO_DIRECT_FP4PV_SM103_LDRED_STAGE1_MASK \
    TK_HAO_DIRECT_FP4PV_SM103_LDRED_MASK
#endif
#ifndef TK_HAO_DIRECT_FP4PV_EARLY_P
// The exact HAO K64 handoff is available for profiling, but it regresses
// this nvcc translation because the second scale copy lands on the issue
// warp's critical path.
#define TK_HAO_DIRECT_FP4PV_EARLY_P 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_EARLY_P
// MXFP4 K64 handoff with disjoint direct-TMEM P-scale pages. The low and
// high K64 scales never overwrite one another, so the tail producer does not
// need to wait for completion of the first PV command.
#define TK_HAO_DIRECT_FP4PV_MX_EARLY_P 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE
// 1: the softmax warps write each disjoint scale page directly to TMEM.
// 0: the issuer copies the shared scale page once per K64 command.
#define TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_REGISTER_SCALE_HANDOFF
// Keep each quarter's E8M0 byte in a register and publish adjacent pairs
// directly to TMEM, bypassing the shared-memory scale scratch round trip.
#define TK_HAO_DIRECT_FP4PV_MX_REGISTER_SCALE_HANDOFF 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_SCRATCH
// Spill only the first scale byte of each K64 pair to row-major scratch, then
// combine it with the second byte directly before the TMEM scale store.
#define TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_SCRATCH 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_EARLY_ASYNC_SCALE
// Let the existing TMEM-before-thread-sync fence order direct scale stores
// with the K64 readiness barrier instead of waiting after each scale page.
#define TK_HAO_DIRECT_FP4PV_MX_EARLY_ASYNC_SCALE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS
// Shiftless P has an exact correction of one. Bypass its per-score-tile
// correction rendezvous while retaining final-stat and output reuse ordering.
#define TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DEFER_JOB_STATS_WAIT
// Keep the next job's row statistics in registers while the correction
// warpgroup consumes the preceding job's shared-memory statistics.
#define TK_HAO_DIRECT_FP4PV_MX_DEFER_JOB_STATS_WAIT 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DIRECT_SCALE_SCRATCH
// Use row-major packed scale words for the direct TMEM handoff. This keeps
// shared memory as a dependency break without the hardware-scale swizzle's
// four-way bank conflicts.
#define TK_HAO_DIRECT_FP4PV_MX_DIRECT_SCALE_SCRATCH 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV
// Issue both query stages' first K64 PV commands before either tail. This
// trades the existing inter-stage QK overlap for a deeper P-construction
// window without allocating another score or output tile.
#define TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD
// Copy the next QK scale pages between the first and tail PV K64 commands,
// then issue QK from the preloaded pages after PV releases the score tile.
#define TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_MASK
// Select which query stages preload QK scales: bit 0 is stage 0 and bit 1 is
// stage 1. Keeping this separate from the feature switch makes stage-local
// scheduling costs measurable without duplicating the issuer path.
#define TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_MASK 3
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_BEFORE_P
// Use the P-ready wait window to preload stage-0 QK scales.
#define TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_BEFORE_P 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_OFFLOAD
// Move both query stages' QK scale-page copies to the otherwise-idle fourth
// warp. The tensor-core issuer waits on stage-local completion barriers, so
// scale copies can overlap PV issue without weakening score-bank ownership.
#define TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_OFFLOAD 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DUAL_ISSUER
// Give each query stage an independent tensor-core issuer warp. Both issuers
// release a shared K/V pipeline slot before TMA may overwrite it.
#define TK_HAO_DIRECT_FP4PV_MX_DUAL_ISSUER 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES
// Requantize the two K64 halves with shared NVFP4 scales, then reuse one Q and
// one K TMEM scale page for both MMAs. This halves QK scale-copy traffic at an
// explicit accuracy cost; the benchmark must prepare matching payloads.
#define TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALE_MASK
// Select one-sided scale folding without changing the legacy both-sides flag:
// bit 0 folds Q and bit 1 folds K. The two controls are mutually exclusive.
#define TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALE_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_QK_COMPACT_FOLDED_SCALES
// Store only the one scale page consumed by a fully folded Q/K operand. This
// halves Q/K scale TMA traffic and the scale footprint of each pipeline slot.
#define TK_HAO_DIRECT_FP4PV_NV_QK_COMPACT_FOLDED_SCALES 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_QK_PRELOAD_PAGE_MASK
// For folded Q/K scales, choose which pages are copied before the PV tail:
// bit 0 is Q and bit 1 is K. Missing pages are copied immediately before QK.
#define TK_HAO_DIRECT_FP4PV_NV_QK_PRELOAD_PAGE_MASK 3
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG
// In shiftless mode the correction warpgroup is idle during score iteration.
// Let it own stage-0 Q3 P construction while the stage-0 softmax warpgroup
// builds Q1 and publishes the first K64.
#define TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG
// Offload Q3 for both query stages. Unlike the stage-0-only probe, this
// shortens both softmax critical paths and amortizes the correction-WG
// lifecycle across two useful quarters.
#define TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG
// Copy each Q3 score quarter to compact FP8 shared memory before releasing its
// score TMEM lease. The correction warpgroup builds Q3 from that copy, allowing
// the next QK issue to overlap without participating in the score lease.
#define TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG
// Temporarily checkpoint raw Q3 as FP8 in the not-yet-built P-tail columns.
// The correction warpgroup consumes it before Q2/Q3 replace that storage.
#define TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_FUSED_PV_ISSUE
// HAO keeps both PV chunks and the tail wait in one low-level block. nvcc
// schedules the split form faster, so retain the fused form as an explicit A/B.
#define TK_HAO_DIRECT_FP4PV_FUSED_PV_ISSUE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SPARSE_PV
// Paired-4:8 P compression feeding one sparse K128 NVFP4 PV instruction.
#define TK_HAO_DIRECT_FP4PV_SPARSE_PV 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR
// Sparse selector: 0=branched top-2, 1=fixed ceiling, 2=SIMD top-2,
// 3=branchless scalar top-2, 4=2-bit LUT top-2, 5=OR-score LUT.
#define TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SPARSE_SPLIT_K128
// Restore first-half publication with two sparse K128 commands. Each command
// owns one K64 half and keeps the other half permanently zero-padded.
#define TK_HAO_DIRECT_FP4PV_SPARSE_SPLIT_K128 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_LOG2_QUANT
// Fuse softmax exponentiation and P quantization in the log domain. This is
// faster than HAO's SM100 post-exp quantizer in the nvcc implementation.
#define TK_HAO_DIRECT_FP4PV_LOG2_QUANT 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_POLICY_ID
// 0=custom, 1=fast, 2=balanced, 3=accurate, 4=exact,
// 5=HAO cosine match, 6=HAO relative-L2 match, 7=universal.
#define TK_HAO_DIRECT_FP4PV_NV_POLICY_ID 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX
// Run NVFP4 P quantization in the quarter-local shiftless lifecycle used by
// the fast MXFP4 path, isolating format quality from the row-max scan.
#define TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SCALE_LIFT
// Opt in to a nonzero global NVFP4 P-scale lift without restoring row-max
// stabilization. Callers must choose an exponent-rebased factor whose E4M3
// scale remains finite for their score distribution.
#define TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SCALE_LIFT 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_SCALE_SATFINITE
// Keep native exponentiation, sampled denominators, and the fast log-domain
// E4M3 scale encoder inside finite representable ranges.
#define TK_HAO_DIRECT_FP4PV_NV_SCALE_SATFINITE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_V_GLOBAL_DECODE
// Apply the reciprocal tensor-level NVFP4 V scale in the output epilogue.
// This is opt-in so the production identity-scale kernel remains unchanged.
#define TK_HAO_DIRECT_FP4PV_NV_V_GLOBAL_DECODE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2
// 0: native EX2 for every NVFP4 payload pair; 1: cubic ALU approximation;
// modes 2-4 retain four, eight, or two native pairs per quarter. Mode 5
// matches MXFP4 mode 23: four native pairs and an affine Q0 tail. Mode 6
// additionally folds the nonnegative clamp into the E2M1 conversion, matching
// the fast MXFP4 pack and denominator path. Mode 7 replaces mode 6's unsafe
// unclamped denominator with the exact sum of the packed E2M1 payload. Mode 8
// uses that exact sum only for affine quarters and the cheaper float sum for
// cubic quarters. Mode 9 estimates the denominator from the four native exp2
// pairs already retained by the approximation schedule.
#define TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_STAGE0_AFFINE_MASK
// Select stage-0 NVFP4 score quarters whose non-native exp2 pairs use the
// same one-FMA affine approximation as the fast MXFP4 mode-23 diagnostic.
#define TK_HAO_DIRECT_FP4PV_NV_STAGE0_AFFINE_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_STAGE1_AFFINE_MASK
// Independent stage-1 quarter mask for spending NVFP4's additional E4M3
// scale accuracy on a shorter score-to-P path.
#define TK_HAO_DIRECT_FP4PV_NV_STAGE1_AFFINE_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX
// Approximation used by Q0 and by quarters selected in either affine mask:
// 0 uses the throughput affine; 1 uses the distribution-refitted cubic;
// 2 routes every Q0 pair through native exp2 and keeps selected later
// quarters affine; 3 evaluates the cubic in packed BF16 arithmetic; 4 uses
// a two-slope absolute-value fit with two dependent FMA levels.
#define TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_AFFINE_A
#define TK_HAO_DIRECT_FP4PV_NV_AFFINE_A 1.61131608f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_AFFINE_B
#define TK_HAO_DIRECT_FP4PV_NV_AFFINE_B 0.93574703f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_CUBIC_A
#define TK_HAO_DIRECT_FP4PV_NV_CUBIC_A 0.07839806f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_CUBIC_B
#define TK_HAO_DIRECT_FP4PV_NV_CUBIC_B 0.28625049f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_CUBIC_C
#define TK_HAO_DIRECT_FP4PV_NV_CUBIC_C 0.63145205f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_CUBIC_D
#define TK_HAO_DIRECT_FP4PV_NV_CUBIC_D 0.99202336f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_A
#define TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_A 0.07430709f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_B
#define TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_B 0.28611863f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_C
#define TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_C 0.64670005f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_D
#define TK_HAO_DIRECT_FP4PV_NV_REFIT_CUBIC_D 0.99010784f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_HINGE_M
#define TK_HAO_DIRECT_FP4PV_NV_HINGE_M 1.37630105f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_HINGE_B
#define TK_HAO_DIRECT_FP4PV_NV_HINGE_B 0.28063917f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_HINGE_D
#define TK_HAO_DIRECT_FP4PV_NV_HINGE_D 0.90739089f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_HINGE_T
#define TK_HAO_DIRECT_FP4PV_NV_HINGE_T 0.84563082f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_NATIVE_PAIRS
// Native EX2 pairs per quarter. Six balances SFU and packed-FMA pressure for
// the high-accuracy cubic path; four remains faster for the affine path.
#define TK_HAO_DIRECT_FP4PV_NV_NATIVE_PAIRS 4
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_NATIVE6_QUARTER_MASK
// When six native pairs are selected, choose the score quarters that receive
// the two additional EX2 pairs. The default preserves the all-quarter policy.
#define TK_HAO_DIRECT_FP4PV_NV_NATIVE6_QUARTER_MASK 15
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE
// Duplicate one N32-derived E4M3 scale into both legal NVFP4 block-16 scale
// slots. This removes one max tree while preserving the hardware contract.
#define TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE
// 0: native exp2 plus E4M3 conversion; 1: direct power-of-two E4M3 encoding;
// 2: magic-bias log-grid encoding with all three E4M3 mantissa bits;
// 3: mode 2 with the finite interval folded into one saturating FMA;
// 4: exact log-grid code through saturating float-to-int conversion;
// 5: mode 2 scale bytes with payload classification independent of the
// rounded scale, allowing scale encoding to overlap the payload path.
#define TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE
// 0: two serial block-16 groups; 1: quarter-wide EX2/ALU/pack schedule with
// the existing four-pair denominator sample; 2: use mode 1 with the
// two-pair denominator sample used by the MXFP4 mode-23 schedule.
#define TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE
// 0: retain the selected PWL mode's pre-quantization denominator.
// 1: normalize by the exact represented E2M1 payload and E4M3 scale.
// This changes normalization only: score approximation, payload codes, and
// scale encoding remain identical.
#define TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_DENOM_CORR_WG
// Let the otherwise idle correction warpgroup snapshot packed P and its
// quarter scales from TMEM, release the score slot, then compute the exact
// represented denominator in parallel with the next QK issue.
#define TK_HAO_DIRECT_FP4PV_NV_DENOM_CORR_WG 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_DENOM_SMEM_HANDOFF
// Mirror packed Q0/Q1 in SMEM so correction can release score TMEM after
// loading only the first-half scale and the Q2/Q3 payload.
#define TK_HAO_DIRECT_FP4PV_NV_DENOM_SMEM_HANDOFF 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_DENOM_WORDS
// Number of packed E2M1 words sampled per N32 quarter. Four is exact; one
// and two are stratified represented-P estimators that preserve the actual
// E4M3 quarter scale.
#define TK_HAO_DIRECT_FP4PV_NV_DENOM_WORDS 4
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_DENOM_KNOWN_MAX
// Treat one E2M1=6 value as known from quarter-scale construction, and
// extrapolate only the sampled residual. This lowers estimator variance for
// sparse attention rows.
#define TK_HAO_DIRECT_FP4PV_NV_DENOM_KNOWN_MAX 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_DENOM_EXACT_MAX_TILES
// Use the exact represented-P correction warpgroup for specialized kernels
// at or below this number of N128 tiles. Zero disables length adaptation.
#define TK_HAO_DIRECT_FP4PV_NV_DENOM_EXACT_MAX_TILES 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64
// Use Q0/Q1 of the first physical N128 tile as one global row anchor for
// every key tile. K/V preprocessing must place 64 distributed logical keys
// in those positions. This avoids per-tile online correction.
#define TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32
// Schedule-preserving global anchor. K/V preprocessing places 32 distributed
// keys in physical Q0; the reader reuses Q0's existing max at its legal load
// point and never advances the Q1 read.
#define TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_SCALE_X8
// Calibration applied to the retained global row anchor. Eight is identity.
// This is compile-time so production variants pay only the selected multiply,
// which constant-folds away for the identity setting.
#define TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_SCALE_X8 8
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_BIAS_X8
// Additive raw-score margin applied after global-anchor calibration. This
// compensates for the expected gap between a sampled and full row maximum
// without changing score loads or synchronization.
#define TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_BIAS_X8 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_CORR_WG
// Have the correction warpgroup publish first-tile Q1 maxima. The softmax
// warpgroups combine these with their resident Q0 maxima before packing, so
// the normal shared-memory scale handoff remains unchanged.
#define TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_CORR_WG 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_PREPACK_Q1
// On the first tile, wait for the Q1 load already paired with delayed Q2 and
// establish the Q0/Q1 anchor before packing Q0. Later tiles retain the normal
// overlapped schedule and shared-memory scale handoff.
#define TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_PREPACK_Q1 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_DEBUG_LSE_ROWMAX
// Diagnostic-only: write the retained raw anchor to LSE instead of logsumexp.
#define TK_HAO_DIRECT_FP4PV_NV_DEBUG_LSE_ROWMAX 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD
// Interrupt Q2 after the selected packed word, transform all of the already
// prefetched Q3, then finish Q2. This changes instruction order only.
#define TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG
// In shiftless mode, let the otherwise idle correction warpgroup consume
// stage-0 score quarters Q0/Q1 and publish the first P K64. The stage-0
// reader independently owns Q2/Q3 and the tail K64.
#define TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_TREE_MAX
// Use the balanced 32-value score-max tree for NV quarter-scale encoding.
#define TK_HAO_DIRECT_FP4PV_NV_TREE_MAX 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_MAX3_REDUCE
// Use SM100's three-input FP32 max to form eight independent pair maxima and
// combine them with four more instructions. This is an exact N32 maximum.
#define TK_HAO_DIRECT_FP4PV_NV_MAX3_REDUCE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE
// Reuse the quarter-max sampling layouts for NV block-scale prediction.
#define TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_BIAS_X2
// Conservative log2 margin for sampled NV maxima, in half-exponent units.
#define TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_BIAS_X2 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER
// Use Q0's exact N32 maximum as the online anchor, retain Q0 in registers,
// and consume every score quarter exactly once. The represented-value
// denominator is required so the sampled anchor changes quantization only,
// not probability mass.
#define TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_PREVIEW_STABILIZER
// Estimate the online tile maximum from a staggered preview spanning all four
// score quarters, then read and consume every complete quarter exactly once.
#define TK_HAO_DIRECT_FP4PV_NV_PREVIEW_STABILIZER 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_PREVIEW_WIDTH
// Number of contiguous values sampled from each N32 quarter. Supported
// widths are 8 and 16, for 25% and 50% row coverage respectively.
#define TK_HAO_DIRECT_FP4PV_NV_PREVIEW_WIDTH 8
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_PREVIEW_BIAS_X8
// Conservative additive score-space margin in eighths. This compensates for
// the expected gap between a sampled maximum and the full N128 maximum.
#define TK_HAO_DIRECT_FP4PV_NV_PREVIEW_BIAS_X8 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_FULL_ROW_STABILIZER
// Retain all N128 scores in the softmax warp group's role-local registers.
// This obtains the exact row maximum with one TMEM read per score.
#define TK_HAO_DIRECT_FP4PV_NV_FULL_ROW_STABILIZER 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE
// Write the two NVFP4 K64 P-scale pages directly into retired score columns,
// avoiding shared-to-TMEM copies on the PV issuer's critical path.
#define TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_PRELOAD_V_TAIL
// 0 copies V-tail scales after P-tail is ready; 1 copies them before PV-half-0;
// 2 copies them after PV-half-0 but before waiting for P-tail.
#define TK_HAO_DIRECT_FP4PV_NV_PRELOAD_V_TAIL 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_ISSUER_V_SCALE_PREFETCH
// Bit mask for V-scale copies issued before the first-P wait: bit 0 copies
// the first K64 scale page and bit 1 copies the tail page.
#define TK_HAO_DIRECT_FP4PV_NV_ISSUER_V_SCALE_PREFETCH 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF
// Retain each quarter's two E4M3 scale bytes in registers and combine adjacent
// quarters directly into a K64 TMEM scale page. This bypasses the shared-memory
// scale scratch used to break the Q0/Q1 and Q2/Q3 dependencies.
#define TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE
// Both query stages consume the same V tile. Keep stage 0's two block-16 V
// scale pages resident and let stage 1 reference them instead of copying the
// same scales into the opposite score bank.
#define TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_NV_SOFTMAX_V_SCALE_PREFETCH
// Bit mask: after Q0 is resident in registers, use retired score columns to
// stage both V scale chunks while the remaining P quarters are constructed.
// Bit 0 enables query stage 0; bit 1 enables query stage 1.
#define TK_HAO_DIRECT_FP4PV_NV_SOFTMAX_V_SCALE_PREFETCH 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_SOFTMAX_V_SCALE_PREFETCH
// Bit mask: after Q0 is resident, stage the single MXFP4 V-scale page in its
// retired score columns. The P-ready handoff orders this copy before PV.
// Bit 0 enables query stage 0; bit 1 enables query stage 1.
#define TK_HAO_DIRECT_FP4PV_MX_SOFTMAX_V_SCALE_PREFETCH 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT
// MXFP4 has one E8M0 scale per N32 score quarter. Derive that power-of-two
// scale before exp2 and fold it into the exp2 argument instead of rescanning
// materialized probabilities.
#define TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_SCALE_SELECT
// 0: exact software RTE, 1: shifted-floor nearest power of two,
// 2: native exp2 followed by exponent-bit extraction, 3: magic-bias nearest,
// 4: max-safe ceiling used by the exact-rowmax oracle.
#define TK_HAO_DIRECT_FP4PV_MX_SCALE_SELECT 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_P_EFFECTIVE_MAX
// Four avoids E2M1's coarse 4-to-6 interval. The E8M0 selector targets an
// amplitude 6/4 times larger and the fast policy uses a max-4 affine exp2 fit.
// Six retains the full native payload range.
#define TK_HAO_DIRECT_FP4PV_MX_P_EFFECTIVE_MAX 6
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_SKIP_ZERO_SCALE_MASK
// Experimental: leave packed P unchanged when its E8M0 scale underflows.
// The scale still controls the represented value consumed by PV.
#define TK_HAO_DIRECT_FP4PV_MX_SKIP_ZERO_SCALE_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE
// 0: disabled, 1: exact log-threshold classifier, 2: affine classifier,
// 3: lower-overhead affine classifier with word-pipelined denominator,
// 4: lower-overhead affine classifier with pair-owned denominator weights.
#define TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BITS
// Experimental direct log-to-E2M1 quantizer. Four bits select a 16-bin
// register LUT and form the denominator from the emitted FP4 codes.
#define TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BITS 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BIT_CODE_MODE
// Replace the dynamic 16-bin lookup with magic-bias FP32 mantissa extraction.
// 1 corrects code 1 from raw score bits; 2 also corrects code 2; 3 packs
// both candidates and the code-1 correction into 16-bit SIMD lanes; 4 omits
// low-tail corrections to establish their instruction/quality cost; 5 also
// folds score normalization and magic-bias encoding into one FMA; 7 uses an
// integral 2.0 slope so E8M0 exponent shifts do not perturb the code phase.
#define TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BIT_CODE_MODE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM
// Form the softmax denominator from packed E2M1 codes and their E8M0 scale.
// Direct log coding always enables this; this knob isolates it with native exp2.
#define TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG
// Snapshot packed MXFP4 P and E8M0 scales in the correction warpgroup, then
// compute the exact represented denominator off the score-reader critical path.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_LOAD_WIDTH
// Number of adjacent packed-P TMEM words read by each correction load.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_LOAD_WIDTH 4
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM
// Sum one query stage while the other stage's packed-P snapshot is in flight.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_PAIR_OVERLAP
// Decode query stage 0 while the stage-1 packed-P snapshot is in flight.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_PAIR_OVERLAP 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_LOCAL_DENOM_PIPELINE
// Mode 2 decodes each packed-P word beside its producer and closes each
// two-word half with an independent early DP4A.
#define TK_HAO_DIRECT_FP4PV_MX_LOCAL_DENOM_PIPELINE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DEFER_DENOM_FINALIZE
// Publish packed P before converting its exact integer denominator to FP32.
// Mode 1 finishes Q0/Q1 after first-K64 publication; mode 2 carries all four
// compact {E8M0, qsum} words until tail-K64 publication.
#define TK_HAO_DIRECT_FP4PV_MX_DEFER_DENOM_FINALIZE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_FACTOR_HOIST
// Accumulate 2^e * qsum and apply the common E2M1-x2/E8M0 factor once.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_FACTOR_HOIST 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_INT_SCALE_FUSION
// Align exact quarter sums in the integer domain using their E8M0 exponents,
// then perform one integer-to-float conversion and scale multiply per stage.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_INT_SCALE_FUSION 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_SUM_LANES
// Independent DP4A accumulators used for each packed quarter.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_SUM_LANES 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS
// Packed x8 words sampled per quarter by the correction warpgroup.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS 4
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_KNOWN_MAX
// Add the block's guaranteed E2M1 maximum exactly and estimate only the
// residual mass when fewer than four packed words are sampled.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_KNOWN_MAX 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_LAYOUT
// 0 samples complete packed words. 1 samples one two-value byte from each of
// the four words, covering the full N32 block with the same eight values.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_LAYOUT 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE
// 0 uses serial DP4A accumulation; 1 uses one packed-byte reduction; 2 uses
// two shorter packed-byte reductions to reduce dependency depth and liveness.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_SCALE_SMEM
// Snapshot the four row scales from their existing shared-memory staging word.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_SCALE_SMEM 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_MAGIC_I2F
// Convert the bounded E2M1 integer sum with an exact FP32 magic-bias FMA.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_MAGIC_I2F 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SMEM
// Hand Q0/Q1 payload to the correction warpgroup at first-half publication.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SMEM 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_PAYLOAD_SMEM
// Mirror packed P into a padded shared-memory page so the denominator
// correction warpgroup can release the score/P TMEM overlay without a load.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_PAYLOAD_SMEM 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT
// Snapshot Q0/Q1 from TMEM at first-half publication and Q2/Q3 at the tail.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_SPLIT_PREFETCH
// Start the Q0/Q1 TMEM load at first-half publication, but defer its wait and
// decode until Q2/Q3 is ready so both halves share one completion point.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_SPLIT_PREFETCH 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32
// Use the maximum of 32 globally distributed keys placed in physical Q0 as a
// fixed row shift. E8M0 absorbs anchor error; this only bounds FP32 magnitude.
#define TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128
// Pre-scan the first score tile, populated with 128 globally distributed
// keys, and use its maximum as the fixed row shift.
#define TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_BIAS_X8
// Conservative additive margin for the MX global anchor in score units.
#define TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_BIAS_X8 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2
// Shift the represented probabilities down by an exact power of two. The
// E8M0 exponent absorbs this shift, preserving E2M1 payload codes while
// keeping the FP32 numerator and denominator away from overflow.
#define TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_ANCHOR_AFFINE_HOIST
// Prepare row-invariant affine quantizer terms once when the global anchor is
// established instead of rebuilding them for every score quarter.
#define TK_HAO_DIRECT_FP4PV_MX_ANCHOR_AFFINE_HOIST 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_FIXED_ANCHOR_LOG2
// Subtract an integer log2 shift before P packing. E8M0 absorbs this shift,
// preserving E2M1 codes while bounding the represented FP32 probabilities.
#define TK_HAO_DIRECT_FP4PV_MX_FIXED_ANCHOR_LOG2 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_STORED_SCALE_SHIFT_LOG2
// Translate every stored P E8M0 scale by an exact power of two. PV and the
// represented-denominator path consume the same scale, so normalization
// cancels the translation while keeping both FP32 values in a safe range.
#define TK_HAO_DIRECT_FP4PV_MX_STORED_SCALE_SHIFT_LOG2 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_DEBUG_LSE_STAT
// Diagnostic-only LSE payload: 0=normal LSE, 1=row anchor, 2=denominator.
#define TK_HAO_DIRECT_FP4PV_DEBUG_LSE_STAT 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_STRIDE
// Experimental normalizer estimate. A stride of 2 or 4 sums evenly spaced
// probability pairs and rescales the result; 1 preserves the exact reduction.
#define TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_STRIDE 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE
// Experimental scale estimate. Modes 2, 3, 4, 5, 6, and 7 reduce 16, 24, 28,
// 8, 4, and 2-score subsets respectively; mode 1 preserves the exact maximum.
#define TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_BIAS_X2
// Conservative log2 margin for sampled maxima, in half-exponent units.
#define TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_BIAS_X2 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_SAMPLE_STRIDE
// Override the scale-max sample count for Q1 only. Uses the same modes as
// MX_MAX_SAMPLE_STRIDE while the other three quarters retain their base mode.
#define TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_SAMPLE_STRIDE 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE
// Reuse each even quarter's exact E8M0 scale for the following odd quarter.
// Bit 0 selects Q0/Q1 and bit 1 selects Q2/Q3.
#define TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_STAGE_MASK
// Bit 0 enables scale reuse for softmax stage 0 and bit 1 for stage 1.
#define TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_STAGE_MASK 3
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q3_SCALE_REUSE_STAGE_MASK
// Independently reuse Q2's E8M0 scale for Q3. This leaves the production
// Q0/Q1 pair policy untouched while selecting stage 0, stage 1, or both.
#define TK_HAO_DIRECT_FP4PV_MX_Q3_SCALE_REUSE_STAGE_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_PERIOD
// Reuse Q0's scale on one out of every 1, 2, or 4 score tiles.
#define TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_PERIOD 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_DELTA
// Integer log2 offset applied to an even quarter's reused E8M0 scale.
#define TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_DELTA 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE
// Refine a reused Q0 scale from 4, 8, or 16 fixed Q1 score samples without
// running Q1's full max tree or float-to-integer scale selection.
#define TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE_BIAS_X4
// Sampling correction in quarter-log2 units before the +/-0.5 scale decision.
#define TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE_BIAS_X4 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_LAGGED_Q1_SCALE
// Quantize Q1 with the preceding score tile's exact Q1 scale. Capture the
// current exact scale while packing so it is ready before the next tile.
#define TK_HAO_DIRECT_FP4PV_MX_LAGGED_Q1_SCALE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q1_SCALE_OFFLOAD
// Have the correction warp group snapshot Q1 and publish its exact scale while
// the owning softmax warp group transforms Q0.
#define TK_HAO_DIRECT_FP4PV_MX_Q1_SCALE_OFFLOAD 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q1_SELF_MAX
// While Q1 uses Q0's ready scale, fold Q1's exact max reduction into its own
// mode-23 transform. Mode 1 only diagnoses the reduction cost; mode 2 rebases
// the transformed payload and published scale to the exact Q1 exponent. Mode
// 3 makes stage-0 Q1 all-affine; mode 4 retains its first two native pairs.
#define TK_HAO_DIRECT_FP4PV_MX_Q1_SELF_MAX 0
#endif

#ifndef TK_HAO_DIRECT_FP4PV_MX_SCAN_FUSED_P
// With direct log coding, retain all four score quarters through the max scan
// so P can be generated without reloading the score tile from TMEM.
#define TK_HAO_DIRECT_FP4PV_MX_SCAN_FUSED_P 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_LOCAL_NATIVE_DEFERRED_SCALE
// Consume each score quarter once. Normalize and pack its E2M1 payload around
// the quarter-local maximum, then publish the four E8M0 amplitudes after the
// exact tile maximum is known. This preserves the score/P overlay without the
// direct-log threshold classifier.
#define TK_HAO_DIRECT_FP4PV_MX_LOCAL_NATIVE_DEFERRED_SCALE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_BIT_EXP2_MASK
// Experimental FP4-oriented exp2 approximation. Selected pair positions in
// the middle score quarters construct FP32 exponent bits before E2M1 packing.
#define TK_HAO_DIRECT_FP4PV_MX_BIT_EXP2_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2
// Approximate exp2 with a continuous two-segment linear function tailored to
// E2M1 decision boundaries. Mode 1 operates on materialized log scores; mode
// 2 folds score normalization and hinge subtraction into the two segments;
// mode 3 uses native relu E2M1 packing and a quantized denominator; mode 4
// uses native relu packing and a half-sampled positive denominator; mode 5
// reduces that estimate to one rotating quarter of the values; mode 6 decodes
// one rotating packed FP4 word for the same quarter-sampled estimate; mode 7
// retunes mode 5's segments against normalized attention-weight error; mode 8
// uses that fit with one rotating eighth of the denominator values; mode 9
// replaces the hinged segments with a threshold-fitted cubic; mode 10 uses
// that cubic with four independent, interleaved denominator pair sums; mode
// 11 reduces the cubic denominator estimate to one interleaved pair; mode 12
// routes the two mode-9 denominator pairs through native EX2; mode 13 routes
// one rotating pair from each packed word through native EX2; modes 14 and 15
// increase that split to two pairs per word and all pairs, respectively; mode
// 16 keeps the denominator pairs rotating but issues the other two SFU pairs
// at the start of their packed words; mode 17 also evaluates each rotating
// denominator pair first within its word; mode 18 launches all four native
// pairs before the remaining cubic work. Modes 19-22 retain mode 17's
// ordering and use a normalized-attention-weight-fitted affine mapping on
// selected quarter-0 pairs: every non-native pair, every non-native pair with
// two extra native pairs, packed word 0, or only pair 3, respectively. Mode
// 23 retains mode 19 arithmetic and releases the consumed QK-scale page
// before transforming Q0 so the producer can prepare the next score tile.
#define TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY
// Native EX2 pairs per four-value word in mode 23: 0 disables native samples,
// 1 retains the schedule, 2 adds the opposite pair, 3 adds a third rotated
// pair, and 4 makes every pair native.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK
// Select score quarters that retain mode-23 native EX2 samples. This only
// affects configurations with nonzero native density.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK 15
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY2_QUARTER_MASK
// Select quarters that receive mode-23's second pair of native EX2 samples.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY2_QUARTER_MASK 15
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY3_QUARTER_MASK
// Select quarters that receive mode-23's third pair of native EX2 samples.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY3_QUARTER_MASK 15
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY3_STAGE_MASK
// Select query stages that receive mode-23's third native sample set.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY3_STAGE_MASK 3
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_STAGE_MASK
// Select query stages that retain mode-23 native EX2 samples (bit 0/1).
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_STAGE_MASK 3
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_SELF_STAGE0_NATIVE
// Permit native EX2 samples in stage 0 when its Q0/Q1 pair scale is owned by
// the self-max path. Disabled by default to preserve the GB200 schedule.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_SELF_STAGE0_NATIVE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE
// Launch mode 23's four native EX2 pairs before its affine/pack stream.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_STAGE_MASK
// Select the query stages that use early-native ordering (bit 0/1).
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_STAGE_MASK 3
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_QUARTER_MASK
// Select the score quarters that use early-native ordering (bits 0-3).
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_QUARTER_MASK 15
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_LOOKAHEAD
// Number of native packing groups launched before the affine/pack stream.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_LOOKAHEAD 4
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_ORDER
// Full-lookahead native order: 0=0,1,2,3; 1=0,2,1,3; 2=0,1,3,2.
#define TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_ORDER 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_PWL_FOUR_SAMPLE_DENOM
// Mode 23 already computes four native EX2 pairs per quarter. Include all
// four in its denominator estimate instead of discarding the middle two.
#define TK_HAO_DIRECT_FP4PV_MX_PWL_FOUR_SAMPLE_DENOM 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_FULL_APPROX_DENOM
// Sum all 32 pre-quantization polynomial values already produced by mode 23.
// This avoids the sampled denominator and the represented-P TMEM correction.
#define TK_HAO_DIRECT_FP4PV_MX_FULL_APPROX_DENOM 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_STABLE_LSE_LOGSUM
// Accumulate the represented-P carriers, or the separately requested
// full-approx mass, in a row-relative log domain for saved LSE.  Output
// normalization continues to use the ordinary represented-P float sum.
#define TK_HAO_DIRECT_FP4PV_MX_STABLE_LSE_LOGSUM 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_AFFINE_A
#define TK_HAO_DIRECT_FP4PV_MX_AFFINE_A 1.62330034f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_AFFINE_B
#define TK_HAO_DIRECT_FP4PV_MX_AFFINE_B 0.92083546f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_CUBIC_A
#define TK_HAO_DIRECT_FP4PV_MX_CUBIC_A 0.07839806f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_CUBIC_B
#define TK_HAO_DIRECT_FP4PV_MX_CUBIC_B 0.28625049f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_CUBIC_C
#define TK_HAO_DIRECT_FP4PV_MX_CUBIC_C 0.63145205f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_CUBIC_D
#define TK_HAO_DIRECT_FP4PV_MX_CUBIC_D 0.99202336f
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_STAGE0_AFFINE_MASK
// Diagnostic mask selecting stage-0 score quarters whose non-native mode-23
// exp2 pairs use the cheaper affine fit instead of the cubic fit.
#define TK_HAO_DIRECT_FP4PV_MX_STAGE0_AFFINE_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_STAGE1_AFFINE_MASK
// Diagnostic mask selecting stage-1 score quarters whose non-native mode-23
// exp2 pairs use the cheaper affine fit instead of the cubic fit.
#define TK_HAO_DIRECT_FP4PV_MX_STAGE1_AFFINE_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX
// Experimental MXFP4 path: let each E8M0 P scale carry the absolute exp2
// magnitude and defer normalization to the epilogue, avoiding a row-max pass.
#define TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_SAMPLED_STABILIZER
// Experimental MXFP4 path: use score quarter 2 as the online softmax shift,
// then consume every quarter exactly once with its exact local block maximum.
#define TK_HAO_DIRECT_FP4PV_MX_SAMPLED_STABILIZER 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_RETAIN_Q0
// Keep score quarter 0 live across the tail of the max scan, removing one
// TMEM reload while retaining only one extra 32-register fragment.
#define TK_HAO_DIRECT_FP4PV_MX_RETAIN_Q0 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_PAIR_LOAD_SCAN
// Issue two independent score-quarter TMEM loads before each load wait during
// the max scan. In sampled-stabilizer mode, pair only the disjoint Q3/Q1 tail.
#define TK_HAO_DIRECT_FP4PV_MX_PAIR_LOAD_SCAN 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q0
// In shiftless mode, prefetch half of Q0 while Q2 is reduced and packed, then
// fetch the other half after Q2 retires to limit the live register footprint.
#define TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q0 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q1
// In shiftless mode, pair Q3 with half of Q1, then fetch Q1's second half
// after Q3 retires instead of keeping the full Q1 fragment live.
#define TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q1 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q3
// With Q0 already split, prefetch half of Q3 while Q0 is processed and finish
// Q3 beside the full Q1 load under the existing tail wait.
#define TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q3 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2
// Preserve Q2 in two x16 loads while Q0/Q1 are consumed, publish their first
// K64, then construct Q2 while the first PV command and Q3 load are in flight.
#define TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3
// Start the delayed schedule's Q3 load as soon as Q1's registers are dead,
// overlapping it with first-scale publication and retained-Q2 packing.
#define TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_LOOKAHEAD
// Reuse Q0's retired registers for a full Q3 lookahead.
#define TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_LOOKAHEAD 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_PROGRESSIVE_REUSE
// Refill each half of Q0's register carrier with Q3 as that half retires.
// The fixed extra load issue only amortizes at sequence length 4096 or above.
#define TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_PROGRESSIVE_REUSE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_EARLY_Q2_REDUCE
// Reduce retained Q2 while the asynchronous first-half scale store is in
// flight, trading a later first-P publication for less tail work.
#define TK_HAO_DIRECT_FP4PV_MX_EARLY_Q2_REDUCE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_STAGE1_EARLY_Q1
// Match stage 0's Q1/Q2-half prefetch schedule in softmax stage 1.
#define TK_HAO_DIRECT_FP4PV_MX_STAGE1_EARLY_Q1 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE
// Interleave Q1's exact max tree with Q0 transform/pack instruction groups.
#define TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_STAGE0_Q1_MAX_INTERLEAVE
// Stage 0 normally discovers Q1's exact exponent while transforming Q1.
// Discover it under Q0 instead so Q1 starts directly at its final scale.
#define TK_HAO_DIRECT_FP4PV_MX_STAGE0_Q1_MAX_INTERLEAVE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q2_MAX_INTERLEAVE
// Once Q1's max is known before its transform, use that transform window to
// reduce the already-prefetched Q2 score fragment.
#define TK_HAO_DIRECT_FP4PV_MX_Q2_MAX_INTERLEAVE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_Q3_MAX_INTERLEAVE
// Wait for the long-lived Q3 lookahead before Q2, then reduce Q3 alongside
// Q2's transform instead of at the P-tail dependency point.
#define TK_HAO_DIRECT_FP4PV_MX_Q3_MAX_INTERLEAVE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_TREE_MAX
// Reduce each 32-value score quarter with a balanced max tree instead of one
// serial accumulator dependency.
#define TK_HAO_DIRECT_FP4PV_MX_TREE_MAX 1
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MAX3_REDUCE
// Use SM100's three-input FP32 max for the exact N32 MXFP4 P-scale reduction.
#define TK_HAO_DIRECT_FP4PV_MX_MAX3_REDUCE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_MAX3_WIDE_REDUCE
// Trade two extra MAX3 instructions for a shorter exact reduction chain.
#define TK_HAO_DIRECT_FP4PV_MX_MAX3_WIDE_REDUCE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_P_ILP_PIPELINE
// Keep two probability-construction groups in flight. This preserves the
// arithmetic while interleaving exp2 issue with packing and sum consumption.
#define TK_HAO_DIRECT_FP4PV_P_ILP_PIPELINE 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_FIXED_P_CEILING
// Diagnostic only: preserve the full QK/TMEM/scale/PV lifecycle but replace
// score processing with a fixed P=1 payload and identity block scales.
#define TK_HAO_DIRECT_FP4PV_FIXED_P_CEILING 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_KV_REUSE_CEILING
// Fixed-P diagnostic: initialize each K/V pipeline slot once, then republish
// its resident payload. This bounds the benefit of eliminating repeated TMA
// transfers without changing the production two-query QK/PV lifecycle.
#define TK_HAO_DIRECT_FP4PV_KV_REUSE_CEILING 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SCALE_COPY_CEILING_MASK
// Fixed-P diagnostic: bit 0 suppresses QK scale copies and bit 1 suppresses
// V scale copies. MMA commands still issue against the resident TMEM pages.
#define TK_HAO_DIRECT_FP4PV_SCALE_COPY_CEILING_MASK 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SCORE_PACK_CEILING
// Diagnostic only: read each real score quarter and pack it directly to P,
// preserving the score/P overlay while skipping max, exp2 and normalization.
#define TK_HAO_DIRECT_FP4PV_SCORE_PACK_CEILING 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_ROWMAX_PACK_CEILING
// Diagnostic only: retain the real row-max scan, score reloads and correction
// lifecycle, then pack raw scores with identity block scales.
#define TK_HAO_DIRECT_FP4PV_ROWMAX_PACK_CEILING 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_FAKE_DENOM_CEILING
// Diagnostic only: retain max, exp2, scale selection, FP4 packing and scaled
// PV, but replace each quarter's denominator contribution with a constant.
#define TK_HAO_DIRECT_FP4PV_FAKE_DENOM_CEILING 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_EX2_EMU_MASK
// Split selected exponent pairs between packed ALU emulation and MUFU. The
// measured GB200 winner selects positions 3, 7, 11 and 15 in quarters 1 and 2.
#define TK_HAO_DIRECT_FP4PV_EX2_EMU_MASK 0x8888u
#endif
#ifndef TK_HAO_DIRECT_FP4PV_EX2_ALU_DEGREE
#define TK_HAO_DIRECT_FP4PV_EX2_ALU_DEGREE 3
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_EX2_Q1_MASK
#define TK_HAO_DIRECT_FP4PV_MX_EX2_Q1_MASK \
    TK_HAO_DIRECT_FP4PV_EX2_EMU_MASK
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_EX2_Q2_MASK
#define TK_HAO_DIRECT_FP4PV_MX_EX2_Q2_MASK \
    TK_HAO_DIRECT_FP4PV_EX2_EMU_MASK
#endif
#ifndef TK_HAO_DIRECT_FP4PV_MX_EX2_Q3_MASK
// Q3 was historically all-MUFU. Keep that behavior unless an explicit
// tail-critical ALU/SFU split is requested.
#define TK_HAO_DIRECT_FP4PV_MX_EX2_Q3_MASK 0u
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SOFTMAX_REGS
#define TK_HAO_DIRECT_FP4PV_SOFTMAX_REGS 192
#endif
#ifndef TK_HAO_DIRECT_FP4PV_SOFTMAX_STAGE0_REGS
#define TK_HAO_DIRECT_FP4PV_SOFTMAX_STAGE0_REGS 0
#endif
#ifndef TK_HAO_DIRECT_FP4PV_CORRECTION_REGS
#define TK_HAO_DIRECT_FP4PV_CORRECTION_REGS 80
#endif
#ifndef TK_HAO_DIRECT_FP4PV_PRODUCER_REGS
#define TK_HAO_DIRECT_FP4PV_PRODUCER_REGS 48
#endif
#ifndef TK_HAO_DIRECT_ROUTE_NAME
#if TK_HAO_DIRECT_FP4PV_SM103_N256
#if TK_HAO_DIRECT_FP4PV_SM103_WIDE_KEY_TILE == 192
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_sm103_n192_nvfp4_mxfp4pv"
#else
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_sm103_n256_nvfp4_mxfp4pv"
#endif
#elif TK_HAO_DIRECT_FP4PV_SM103_GROUP2
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_sm103_group2_nvfp4_mxfp4pv"
#elif TK_HAO_DIRECT_FP4PV_SPARSE_PV
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_nvfp4_sparse_nvfp4pv"
#elif TK_HAO_DIRECT_QK_SCALE_MODE == 0 && \
    TK_HAO_DIRECT_PV_SCALE_MODE == 0
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_nvfp4_nvfp4pv"
#elif TK_HAO_DIRECT_QK_SCALE_MODE == 0
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_nvfp4_mxfp4pv"
#elif TK_HAO_DIRECT_PV_SCALE_MODE == 0
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_mxfp4_nvfp4pv"
#else
#define TK_HAO_DIRECT_ROUTE_NAME \
    "real_fwd_tk_hao_direct_mxfp4_mxfp4pv"
#endif
#endif

struct config_fp4pv_hao_direct
    : public hao_direct_arithmetic_base {
    static constexpr bool HAO_DIRECT_PIPELINE = true;
    static constexpr bool HAO_DIRECT_NVFP4_QK =
        TK_HAO_DIRECT_QK_SCALE_MODE == 0;
    static constexpr bool HAO_DIRECT_MXFP4_QK =
        TK_HAO_DIRECT_QK_SCALE_MODE == 1;
    static constexpr bool HAO_DIRECT_NV_QK_COMPACT_FOLDED_SCALES =
        HAO_DIRECT_NVFP4_QK &&
        TK_HAO_DIRECT_FP4PV_NV_QK_COMPACT_FOLDED_SCALES;
    static constexpr int QK_SCALE_ROWS =
        HAO_DIRECT_NV_QK_COMPACT_FOLDED_SCALES
            ? 1
            : (HAO_DIRECT_NVFP4_QK ? QK_SCALE_CHUNKS : 1);
    static constexpr bool HAO_DIRECT_FP8_PV = false;
    static constexpr bool HAO_DIRECT_FP4_PV = true;
    static constexpr bool HAO_DIRECT_NVFP4_PV =
        TK_HAO_DIRECT_PV_SCALE_MODE == 0;
    static constexpr bool HAO_DIRECT_MXFP4_PV =
        TK_HAO_DIRECT_PV_SCALE_MODE == 1;
    static constexpr bool HAO_DIRECT_MX_EARLY_P =
        HAO_DIRECT_MXFP4_PV &&
        TK_HAO_DIRECT_FP4PV_MX_EARLY_P;
    static constexpr bool HAO_DIRECT_MX_EARLY_DIRECT_SCALE =
        HAO_DIRECT_MX_EARLY_P &&
        TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE;
    static constexpr bool HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF =
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_REGISTER_SCALE_HANDOFF;
    static constexpr bool HAO_DIRECT_MX_PAIR_SCALE_SCRATCH =
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_SCRATCH;
    // Both routes expose one legal K64 handoff. NVFP4 copies independent
    // scale chunks; MXFP4 writes the two halves to disjoint retired-score
    // pages and therefore avoids a late overwrite/completion dependency.
    static constexpr bool HAO_DIRECT_EARLY_P =
        (HAO_DIRECT_NVFP4_PV &&
         TK_HAO_DIRECT_FP4PV_EARLY_P) ||
        HAO_DIRECT_MX_EARLY_P;
    static constexpr bool HAO_DIRECT_FUSED_PV_ISSUE =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_FUSED_PV_ISSUE;
    static constexpr bool HAO_DIRECT_SPARSE_PV =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_SPARSE_PV;
    static constexpr int HAO_DIRECT_SPARSE_SELECTOR_MODE =
        HAO_DIRECT_SPARSE_PV
            ? TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR
            : 0;
    static constexpr bool HAO_DIRECT_SPARSE_SPLIT_K128 =
        HAO_DIRECT_SPARSE_PV &&
        TK_HAO_DIRECT_FP4PV_SPARSE_SPLIT_K128;
    static constexpr bool HAO_DIRECT_LOG2_QUANT =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_LOG2_QUANT;
    static constexpr int HAO_DIRECT_NV_POLICY_ID =
        TK_HAO_DIRECT_FP4PV_NV_POLICY_ID;
    static constexpr bool HAO_DIRECT_MX_LOG2_QUANT =
        HAO_DIRECT_MXFP4_PV &&
        TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT;
    static constexpr bool HAO_DIRECT_NV_SHIFTLESS_SOFTMAX =
        HAO_DIRECT_NVFP4_PV &&
        HAO_DIRECT_LOG2_QUANT &&
        TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX;
    static constexpr bool HAO_DIRECT_NV_SHIFTLESS_SCALE_LIFT =
        HAO_DIRECT_NV_SHIFTLESS_SOFTMAX &&
        TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SCALE_LIFT;
    static constexpr bool HAO_DIRECT_NV_SCALE_SATFINITE =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_SCALE_SATFINITE;
    static constexpr bool HAO_DIRECT_NV_V_GLOBAL_DECODE =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_V_GLOBAL_DECODE;
    static constexpr float HAO_DIRECT_NV_P_GLOBAL_LOG2 =
        HAO_DIRECT_NVFP4_PV && HAO_DIRECT_LOG2_QUANT
            ? TK_HAO_DIRECT_FP4PV_NV_P_GLOBAL_LOG2
            : 0.0f;
    static constexpr int HAO_DIRECT_NV_PWL_EXP2_MODE =
        HAO_DIRECT_LOG2_QUANT
            ? TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2
            : 0;
    static constexpr bool HAO_DIRECT_NV_RELU_PACK =
        HAO_DIRECT_NV_PWL_EXP2_MODE == 6 ||
        HAO_DIRECT_NV_PWL_EXP2_MODE == 7 ||
        HAO_DIRECT_NV_PWL_EXP2_MODE == 8 ||
        HAO_DIRECT_NV_PWL_EXP2_MODE == 9;
    static constexpr bool HAO_DIRECT_NV_QUANTIZED_DENOM =
        HAO_DIRECT_NV_PWL_EXP2_MODE == 7;
    static constexpr bool HAO_DIRECT_NV_HYBRID_DENOM =
        HAO_DIRECT_NV_PWL_EXP2_MODE == 8;
    static constexpr bool HAO_DIRECT_NV_SAMPLED_DENOM =
        HAO_DIRECT_NV_PWL_EXP2_MODE == 9;
    static constexpr uint32_t HAO_DIRECT_NV_STAGE0_AFFINE_MASK =
        HAO_DIRECT_NV_PWL_EXP2_MODE != 0
            ? TK_HAO_DIRECT_FP4PV_NV_STAGE0_AFFINE_MASK
            : 0;
    static constexpr uint32_t HAO_DIRECT_NV_STAGE1_AFFINE_MASK =
        HAO_DIRECT_NV_PWL_EXP2_MODE != 0
            ? TK_HAO_DIRECT_FP4PV_NV_STAGE1_AFFINE_MASK
            : 0;
    static constexpr int HAO_DIRECT_NV_FAST_APPROX_MODE =
        HAO_DIRECT_NV_PWL_EXP2_MODE != 0
            ? TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX
            : 0;
    static constexpr int HAO_DIRECT_NV_NATIVE_PAIRS =
        HAO_DIRECT_NV_PWL_EXP2_MODE == 9
            ? TK_HAO_DIRECT_FP4PV_NV_NATIVE_PAIRS
            : 4;
    static constexpr uint32_t HAO_DIRECT_NV_NATIVE6_QUARTER_MASK =
        HAO_DIRECT_NV_NATIVE_PAIRS == 6
            ? TK_HAO_DIRECT_FP4PV_NV_NATIVE6_QUARTER_MASK
            : 0;
    static constexpr bool HAO_DIRECT_NV_QUARTER_SCALE =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE;
    static constexpr int HAO_DIRECT_NV_SCALE_ENCODE_MODE =
        HAO_DIRECT_NV_QUARTER_SCALE
            ? TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE
            : 0;
    static constexpr int HAO_DIRECT_NV_QUARTER_SCHEDULE_MODE =
        HAO_DIRECT_NV_QUARTER_SCALE
            ? TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE
            : 0;
    static constexpr int HAO_DIRECT_NV_DENOM_SOURCE =
        HAO_DIRECT_NVFP4_PV
            ? TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE
            : 0;
    static constexpr bool HAO_DIRECT_NV_DENOM_SHORT_EXACT =
        HAO_DIRECT_NV_DENOM_SOURCE == 1 &&
        TK_HAO_DIRECT_FP4PV_NV_DENOM_EXACT_MAX_TILES > 0 &&
        (TK_HAO_DIRECT_SEQ_LEN / 128) <=
            TK_HAO_DIRECT_FP4PV_NV_DENOM_EXACT_MAX_TILES;
    static constexpr bool HAO_DIRECT_NV_DENOM_CORR_WG =
        HAO_DIRECT_NVFP4_PV &&
        (TK_HAO_DIRECT_FP4PV_NV_DENOM_CORR_WG ||
         HAO_DIRECT_NV_DENOM_SHORT_EXACT);
    static constexpr bool HAO_DIRECT_NV_DENOM_SMEM_HANDOFF =
        HAO_DIRECT_NV_DENOM_CORR_WG
            ? TK_HAO_DIRECT_FP4PV_NV_DENOM_SMEM_HANDOFF != 0
            : false;
    static constexpr int HAO_DIRECT_NV_DENOM_WORDS =
        HAO_DIRECT_NV_DENOM_SOURCE == 1
            ? (HAO_DIRECT_NV_DENOM_SHORT_EXACT
                   ? 4
                   : TK_HAO_DIRECT_FP4PV_NV_DENOM_WORDS)
            : 4;
    static constexpr bool HAO_DIRECT_NV_DENOM_KNOWN_MAX =
        HAO_DIRECT_NV_DENOM_SOURCE == 1 &&
        HAO_DIRECT_NV_DENOM_WORDS < 4 &&
        TK_HAO_DIRECT_FP4PV_NV_DENOM_KNOWN_MAX;
    static constexpr bool HAO_DIRECT_NV_GLOBAL_ANCHOR64 =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64;
    static constexpr bool HAO_DIRECT_NV_GLOBAL_ANCHOR32 =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32;
    static constexpr bool HAO_DIRECT_NV_GLOBAL_ANCHOR =
        HAO_DIRECT_NV_GLOBAL_ANCHOR32 ||
        HAO_DIRECT_NV_GLOBAL_ANCHOR64;
    static constexpr float HAO_DIRECT_NV_GLOBAL_ANCHOR_SCALE =
        static_cast<float>(
            TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_SCALE_X8) *
        0.125f;
    static constexpr float HAO_DIRECT_NV_GLOBAL_ANCHOR_BIAS =
        static_cast<float>(
            TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_BIAS_X8) *
        0.125f;
    static constexpr bool HAO_DIRECT_NV_GLOBAL_ANCHOR_CORR_WG =
        HAO_DIRECT_NV_GLOBAL_ANCHOR64 &&
        TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_CORR_WG;
    static constexpr bool HAO_DIRECT_NV_GLOBAL_ANCHOR_PREPACK_Q1 =
        HAO_DIRECT_NV_GLOBAL_ANCHOR64 &&
        TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_PREPACK_Q1;
    static constexpr bool HAO_DIRECT_NV_DEBUG_LSE_ROWMAX =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_DEBUG_LSE_ROWMAX;
    static constexpr int HAO_DIRECT_NV_TAIL_Q3_HOOK_WORD =
        HAO_DIRECT_NVFP4_PV
            ? TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD
            : 0;
    static constexpr bool HAO_DIRECT_NV_Q01_CORR_WG =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG;
    static constexpr bool HAO_DIRECT_NV_TREE_MAX =
        HAO_DIRECT_NV_QUARTER_SCALE &&
        TK_HAO_DIRECT_FP4PV_NV_TREE_MAX;
    static constexpr bool HAO_DIRECT_NV_MAX3_REDUCE =
        HAO_DIRECT_NV_QUARTER_SCALE &&
        TK_HAO_DIRECT_FP4PV_NV_MAX3_REDUCE;
    static constexpr bool HAO_DIRECT_NV_Q0_STABILIZER =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER;
    static constexpr bool HAO_DIRECT_NV_PREVIEW_STABILIZER =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_PREVIEW_STABILIZER;
    static constexpr int HAO_DIRECT_NV_PREVIEW_WIDTH =
        HAO_DIRECT_NV_PREVIEW_STABILIZER
            ? TK_HAO_DIRECT_FP4PV_NV_PREVIEW_WIDTH
            : 0;
    static constexpr int HAO_DIRECT_NV_PREVIEW_BIAS_X8 =
        HAO_DIRECT_NV_PREVIEW_STABILIZER
            ? TK_HAO_DIRECT_FP4PV_NV_PREVIEW_BIAS_X8
            : 0;
    static constexpr bool HAO_DIRECT_NV_FULL_ROW_STABILIZER =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_FULL_ROW_STABILIZER;
    static constexpr bool HAO_DIRECT_NV_EARLY_DIRECT_SCALE =
        HAO_DIRECT_NVFP4_PV &&
        HAO_DIRECT_EARLY_P &&
        TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE;
    static constexpr int HAO_DIRECT_NV_PRELOAD_V_TAIL_MODE =
        HAO_DIRECT_NVFP4_PV
            ? TK_HAO_DIRECT_FP4PV_NV_PRELOAD_V_TAIL
            : 0;
    static constexpr int
        HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK =
            HAO_DIRECT_NVFP4_PV
                ? TK_HAO_DIRECT_FP4PV_NV_ISSUER_V_SCALE_PREFETCH
                : 0;
    static constexpr bool HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH =
        HAO_DIRECT_NV_ISSUER_V_SCALE_PREFETCH_MASK != 0;
    static constexpr bool HAO_DIRECT_NV_REGISTER_SCALE_HANDOFF =
        HAO_DIRECT_NV_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF;
    static constexpr bool HAO_DIRECT_NV_REUSE_V_SCALE =
        HAO_DIRECT_NVFP4_PV &&
        TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE;
    static constexpr int HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK =
        HAO_DIRECT_NVFP4_PV
            ? TK_HAO_DIRECT_FP4PV_NV_SOFTMAX_V_SCALE_PREFETCH
            : 0;
    static constexpr bool HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH =
        HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK != 0;
    static constexpr int HAO_DIRECT_MX_SOFTMAX_V_SCALE_PREFETCH_MASK =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_SOFTMAX_V_SCALE_PREFETCH
            : 0;
    static constexpr bool HAO_DIRECT_MX_SOFTMAX_V_SCALE_PREFETCH =
        HAO_DIRECT_MX_SOFTMAX_V_SCALE_PREFETCH_MASK != 0;
    static constexpr int HAO_DIRECT_MX_SCALE_SELECT =
        HAO_DIRECT_MX_LOG2_QUANT
            ? TK_HAO_DIRECT_FP4PV_MX_SCALE_SELECT
            : 0;
    static constexpr int HAO_DIRECT_MX_P_EFFECTIVE_MAX =
        HAO_DIRECT_MX_LOG2_QUANT
            ? TK_HAO_DIRECT_FP4PV_MX_P_EFFECTIVE_MAX
            : 6;
    static constexpr bool HAO_DIRECT_MX_SKIP_ZERO_SCALE_MASK =
        HAO_DIRECT_MX_LOG2_QUANT &&
        TK_HAO_DIRECT_FP4PV_MX_SKIP_ZERO_SCALE_MASK;
    static constexpr int HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE =
        HAO_DIRECT_MX_LOG2_QUANT &&
        TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE
            ? TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE
            : 0;
    static constexpr bool HAO_DIRECT_MX_DIRECT_LOG_CODE =
        HAO_DIRECT_MX_DIRECT_LOG_CODE_MODE != 0;
    static constexpr int HAO_DIRECT_MX_LOG_LUT_BITS =
        HAO_DIRECT_MX_LOG2_QUANT
            ? TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BITS
            : 0;
    static constexpr bool HAO_DIRECT_MX_LOG_LUT =
        HAO_DIRECT_MX_LOG_LUT_BITS != 0;
    static constexpr int HAO_DIRECT_MX_LOG_LUT_BIT_CODE_MODE =
        HAO_DIRECT_MX_LOG_LUT
            ? TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BIT_CODE_MODE
            : 0;
    static constexpr bool HAO_DIRECT_MX_QUANTIZED_DENOM =
        HAO_DIRECT_MX_DIRECT_LOG_CODE ||
        HAO_DIRECT_MX_LOG_LUT ||
        (HAO_DIRECT_MX_LOG2_QUANT &&
         TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 3) ||
        (HAO_DIRECT_MX_LOG2_QUANT &&
         TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM);
    static constexpr bool HAO_DIRECT_MX_DENOM_CORR_WG =
        HAO_DIRECT_MXFP4_PV &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG;
    static constexpr int HAO_DIRECT_MX_DENOM_LOAD_WIDTH =
        TK_HAO_DIRECT_FP4PV_MX_DENOM_LOAD_WIDTH;
    static constexpr bool HAO_DIRECT_MX_DENOM_PIPELINED_SUM =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM;
    static constexpr bool HAO_DIRECT_MX_DENOM_PAIR_OVERLAP =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_PAIR_OVERLAP;
    static constexpr int HAO_DIRECT_MX_LOCAL_DENOM_PIPELINE =
        HAO_DIRECT_MX_QUANTIZED_DENOM &&
                !HAO_DIRECT_MX_DIRECT_LOG_CODE &&
                !HAO_DIRECT_MX_LOG_LUT
            ? TK_HAO_DIRECT_FP4PV_MX_LOCAL_DENOM_PIPELINE
            : 0;
    static constexpr int HAO_DIRECT_MX_DEFER_DENOM_FINALIZE =
        HAO_DIRECT_MXFP4_PV &&
                HAO_DIRECT_MX_QUANTIZED_DENOM &&
                HAO_DIRECT_MX_LOCAL_DENOM_PIPELINE != 0
            ? TK_HAO_DIRECT_FP4PV_MX_DEFER_DENOM_FINALIZE
            : 0;
    static_assert(
        HAO_DIRECT_MX_DEFER_DENOM_FINALIZE >= 0 &&
        HAO_DIRECT_MX_DEFER_DENOM_FINALIZE <= 2);
    static constexpr bool HAO_DIRECT_MX_DENOM_FACTOR_HOIST =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_FACTOR_HOIST;
    static constexpr bool HAO_DIRECT_MX_DENOM_INT_SCALE_FUSION =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_INT_SCALE_FUSION;
    static constexpr int HAO_DIRECT_MX_DENOM_SUM_LANES =
        TK_HAO_DIRECT_FP4PV_MX_DENOM_SUM_LANES;
    static constexpr int HAO_DIRECT_MX_DENOM_WORDS =
        TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS;
    static constexpr bool HAO_DIRECT_MX_DENOM_KNOWN_MAX =
        HAO_DIRECT_MX_QUANTIZED_DENOM &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_KNOWN_MAX;
    static constexpr int HAO_DIRECT_MX_DENOM_SAMPLE_LAYOUT =
        TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_LAYOUT;
    static constexpr int HAO_DIRECT_MX_DENOM_DECODE_MODE =
        TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE;
    static constexpr bool HAO_DIRECT_MX_DENOM_SCALE_SMEM =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_SCALE_SMEM;
    static constexpr bool HAO_DIRECT_MX_DENOM_MAGIC_I2F =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_MAGIC_I2F;
    static constexpr bool HAO_DIRECT_MX_DENOM_EARLY_SMEM =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SMEM;
    static constexpr bool HAO_DIRECT_MX_DENOM_PAYLOAD_SMEM =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_PAYLOAD_SMEM;
    static constexpr bool HAO_DIRECT_MX_DENOM_EARLY_SPLIT =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT;
    static constexpr bool HAO_DIRECT_MX_DENOM_SPLIT_PREFETCH =
        HAO_DIRECT_MX_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DENOM_SPLIT_PREFETCH;
    static constexpr bool HAO_DIRECT_MX_GLOBAL_ANCHOR32 =
        HAO_DIRECT_MXFP4_PV &&
        TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32;
    static constexpr bool HAO_DIRECT_MX_GLOBAL_ANCHOR128 =
        HAO_DIRECT_MXFP4_PV &&
        TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128;
    static constexpr int HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2 =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_STORED_SCALE_SHIFT_LOG2
            : 0;
    static constexpr float HAO_DIRECT_MX_GLOBAL_ANCHOR_BIAS =
        static_cast<float>(
            TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_BIAS_X8) *
            0.125f +
        static_cast<float>(
            TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2) *
            0.6931471805599453f;
    static constexpr bool HAO_DIRECT_MX_ANCHOR_AFFINE_HOIST =
        HAO_DIRECT_MXFP4_PV &&
        TK_HAO_DIRECT_FP4PV_MX_ANCHOR_AFFINE_HOIST;
    static constexpr int HAO_DIRECT_MX_FIXED_ANCHOR_LOG2 =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_FIXED_ANCHOR_LOG2
            : 0;
    static constexpr bool HAO_DIRECT_GLOBAL_ANCHOR =
        HAO_DIRECT_NV_GLOBAL_ANCHOR ||
        HAO_DIRECT_MX_GLOBAL_ANCHOR32 ||
        HAO_DIRECT_MX_GLOBAL_ANCHOR128;
    static constexpr bool HAO_DIRECT_DENOM_CORR_WG =
        HAO_DIRECT_NV_DENOM_CORR_WG ||
        HAO_DIRECT_MX_DENOM_CORR_WG;
    static constexpr int HAO_DIRECT_DEBUG_LSE_STAT =
        TK_HAO_DIRECT_FP4PV_DEBUG_LSE_STAT;
    static constexpr int HAO_DIRECT_MX_DENOM_SAMPLE_STRIDE =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_STRIDE
            : 1;
    static constexpr int HAO_DIRECT_MX_MAX_SAMPLE_STRIDE =
        HAO_DIRECT_NVFP4_PV
            ? TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE
            : (HAO_DIRECT_MXFP4_PV
                   ? TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE
                   : 1);
    static constexpr int HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2 =
        HAO_DIRECT_NVFP4_PV
            ? TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_BIAS_X2
            : (HAO_DIRECT_MXFP4_PV
                   ? TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_BIAS_X2
                   : 0);
    static constexpr int HAO_DIRECT_MX_Q1_MAX_SAMPLE_STRIDE =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_SAMPLE_STRIDE
            : 1;
    static constexpr int HAO_DIRECT_MX_PAIR_SCALE_REUSE =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE
            : 0;
    static constexpr int HAO_DIRECT_MX_PAIR_SCALE_STAGE_MASK =
        TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_STAGE_MASK;
    static constexpr int HAO_DIRECT_MX_Q3_SCALE_REUSE_STAGE_MASK =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_Q3_SCALE_REUSE_STAGE_MASK
            : 0;
    static constexpr int HAO_DIRECT_MX_PAIR_SCALE_PERIOD =
        TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_PERIOD;
    static constexpr int HAO_DIRECT_MX_PAIR_SCALE_DELTA =
        TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_DELTA;
    static constexpr int HAO_DIRECT_MX_Q1_PAIR_PROBE =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE
            : 0;
    static constexpr float HAO_DIRECT_MX_Q1_PAIR_PROBE_BIAS =
        0.25f * static_cast<float>(
            TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE_BIAS_X4);
    static constexpr bool HAO_DIRECT_MX_LAGGED_Q1_SCALE =
        HAO_DIRECT_MXFP4_PV &&
        TK_HAO_DIRECT_FP4PV_MX_LAGGED_Q1_SCALE;
    static constexpr bool HAO_DIRECT_MX_Q1_SCALE_OFFLOAD =
        HAO_DIRECT_MXFP4_PV &&
        TK_HAO_DIRECT_FP4PV_MX_Q1_SCALE_OFFLOAD;
    static constexpr int HAO_DIRECT_MX_Q1_SELF_MAX =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_Q1_SELF_MAX
            : 0;
    static constexpr bool HAO_DIRECT_MX_SCAN_FUSED_P =
        HAO_DIRECT_MX_DIRECT_LOG_CODE &&
        TK_HAO_DIRECT_FP4PV_MX_SCAN_FUSED_P;
    static constexpr bool
        HAO_DIRECT_MX_LOCAL_NATIVE_DEFERRED_SCALE =
            HAO_DIRECT_MX_LOG2_QUANT &&
            TK_HAO_DIRECT_FP4PV_MX_LOCAL_NATIVE_DEFERRED_SCALE;
    static constexpr uint32_t HAO_DIRECT_MX_BIT_EXP2_MASK =
        HAO_DIRECT_MX_LOG2_QUANT
            ? TK_HAO_DIRECT_FP4PV_MX_BIT_EXP2_MASK
            : 0;
    static constexpr int HAO_DIRECT_MX_PWL_EXP2_MODE =
        HAO_DIRECT_MX_LOG2_QUANT &&
            TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2
            ? TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2
            : 0;
    static constexpr bool HAO_DIRECT_MX_PWL_EXP2 =
        HAO_DIRECT_MX_PWL_EXP2_MODE != 0;
    static constexpr int HAO_DIRECT_MX_MODE23_NATIVE_DENSITY =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23
            ? TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY
            : 0;
    static constexpr uint32_t HAO_DIRECT_MX_MODE23_NATIVE_QUARTER_MASK =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23
            ? TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK
            : 0;
    static constexpr uint32_t
        HAO_DIRECT_MX_MODE23_NATIVE_DENSITY2_QUARTER_MASK =
            HAO_DIRECT_MX_PWL_EXP2_MODE == 23
                ? TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY2_QUARTER_MASK
                : 0;
    static constexpr uint32_t
        HAO_DIRECT_MX_MODE23_NATIVE_DENSITY3_QUARTER_MASK =
            HAO_DIRECT_MX_PWL_EXP2_MODE == 23
                ? TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY3_QUARTER_MASK
                : 0;
    static constexpr uint32_t HAO_DIRECT_MX_MODE23_NATIVE_DENSITY3_STAGE_MASK =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23
            ? TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY3_STAGE_MASK
            : 0;
    static constexpr uint32_t HAO_DIRECT_MX_MODE23_NATIVE_STAGE_MASK =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23
            ? TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_STAGE_MASK
            : 0;
    static constexpr bool HAO_DIRECT_MX_MODE23_SELF_STAGE0_NATIVE =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23 &&
        TK_HAO_DIRECT_FP4PV_MX_MODE23_SELF_STAGE0_NATIVE;
    static constexpr bool HAO_DIRECT_MX_MODE23_EARLY_NATIVE =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23 &&
        TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE;
    static constexpr int HAO_DIRECT_MX_MODE23_EARLY_NATIVE_STAGE_MASK =
        TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_STAGE_MASK;
    static constexpr int HAO_DIRECT_MX_MODE23_EARLY_NATIVE_QUARTER_MASK =
        TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_QUARTER_MASK;
    static constexpr int HAO_DIRECT_MX_MODE23_EARLY_NATIVE_LOOKAHEAD =
        TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_LOOKAHEAD;
    static constexpr int HAO_DIRECT_MX_MODE23_EARLY_NATIVE_ORDER =
        TK_HAO_DIRECT_FP4PV_MX_MODE23_EARLY_NATIVE_ORDER;
    static constexpr bool HAO_DIRECT_MX_PWL_FOUR_SAMPLE_DENOM =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23 &&
        TK_HAO_DIRECT_FP4PV_MX_PWL_FOUR_SAMPLE_DENOM;
    static constexpr int HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23
            ? TK_HAO_DIRECT_FP4PV_MX_FULL_APPROX_DENOM
            : 0;
    static constexpr bool HAO_DIRECT_MX_FULL_APPROX_DENOM =
        HAO_DIRECT_MX_FULL_APPROX_DENOM_MODE != 0;
    // When the represented MXFP4 denominator remains the output
    // normalization contract, retain the already-computed pre-quantization
    // approximation as a separate saved-LSE statistic.  This keeps O on the
    // fast represented-P path while preventing E2M1 underflow from publishing
    // a zero denominator to backward.
    static constexpr bool HAO_DIRECT_MX_DUAL_LSE_DENOM =
        HAO_DIRECT_MXFP4_PV &&
        HAO_DIRECT_MX_QUANTIZED_DENOM &&
        HAO_DIRECT_MX_FULL_APPROX_DENOM;
    static constexpr bool HAO_DIRECT_MX_STABLE_LSE_LOGSUM =
        HAO_DIRECT_MXFP4_PV &&
        HAO_DIRECT_MX_QUANTIZED_DENOM &&
        TK_HAO_DIRECT_FP4PV_MX_STABLE_LSE_LOGSUM;
    static constexpr bool HAO_DIRECT_MX_ALTERNATE_LSE_STAT =
        HAO_DIRECT_MX_DUAL_LSE_DENOM ||
        HAO_DIRECT_MX_STABLE_LSE_LOGSUM;
    static constexpr float HAO_DIRECT_MX_AFFINE_A =
        TK_HAO_DIRECT_FP4PV_MX_AFFINE_A;
    static constexpr float HAO_DIRECT_MX_AFFINE_B =
        TK_HAO_DIRECT_FP4PV_MX_AFFINE_B;
    static constexpr float HAO_DIRECT_MX_CUBIC_A =
        TK_HAO_DIRECT_FP4PV_MX_CUBIC_A;
    static constexpr float HAO_DIRECT_MX_CUBIC_B =
        TK_HAO_DIRECT_FP4PV_MX_CUBIC_B;
    static constexpr float HAO_DIRECT_MX_CUBIC_C =
        TK_HAO_DIRECT_FP4PV_MX_CUBIC_C;
    static constexpr float HAO_DIRECT_MX_CUBIC_D =
        TK_HAO_DIRECT_FP4PV_MX_CUBIC_D;
    static constexpr uint32_t HAO_DIRECT_MX_STAGE0_AFFINE_MASK =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23
            ? TK_HAO_DIRECT_FP4PV_MX_STAGE0_AFFINE_MASK
            : 0;
    static constexpr uint32_t HAO_DIRECT_MX_STAGE1_AFFINE_MASK =
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23
            ? TK_HAO_DIRECT_FP4PV_MX_STAGE1_AFFINE_MASK
            : 0;
    static constexpr uint32_t HAO_DIRECT_STAGE0_AFFINE_MASK =
        HAO_DIRECT_MX_STAGE0_AFFINE_MASK |
        HAO_DIRECT_NV_STAGE0_AFFINE_MASK;
    static constexpr uint32_t HAO_DIRECT_STAGE1_AFFINE_MASK =
        HAO_DIRECT_MX_STAGE1_AFFINE_MASK |
        HAO_DIRECT_NV_STAGE1_AFFINE_MASK;
    static constexpr bool HAO_DIRECT_MX_PWL_RELU_PACK =
        HAO_DIRECT_MX_PWL_EXP2_MODE >= 3;
    static constexpr bool HAO_DIRECT_MX_SHIFTLESS_SOFTMAX =
        (HAO_DIRECT_MX_LOG2_QUANT &&
         TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX) ||
        HAO_DIRECT_NV_SHIFTLESS_SOFTMAX;
    static constexpr bool HAO_DIRECT_MX_EARLY_ASYNC_SCALE =
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
        TK_HAO_DIRECT_FP4PV_MX_EARLY_ASYNC_SCALE;
    static constexpr bool HAO_DIRECT_EARLY_DIRECT_SCALE =
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE ||
        HAO_DIRECT_NV_EARLY_DIRECT_SCALE;
    static constexpr bool HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS =
        HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
        TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS;
    static constexpr bool HAO_DIRECT_MX_DEFER_JOB_STATS_WAIT =
        HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
        HAO_DIRECT_DENOM_CORR_WG &&
        TK_HAO_DIRECT_FP4PV_MX_DEFER_JOB_STATS_WAIT;
    static constexpr bool HAO_DIRECT_MX_DIRECT_SCALE_SCRATCH =
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_DIRECT_SCALE_SCRATCH;
    static constexpr bool HAO_DIRECT_MX_SPLIT_STAGE_PV =
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV;
    static constexpr bool HAO_DIRECT_MX_QK_SCALE_PRELOAD =
        HAO_DIRECT_NVFP4_QK &&
        HAO_DIRECT_MXFP4_PV &&
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD;
    static constexpr int HAO_DIRECT_MX_QK_SCALE_PRELOAD_MASK =
        TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_MASK;
    static constexpr bool HAO_DIRECT_MX_QK_SCALE_PRELOAD_BEFORE_P =
        HAO_DIRECT_MX_QK_SCALE_PRELOAD &&
        TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_BEFORE_P;
    static constexpr bool HAO_DIRECT_MX_QK_SCALE_OFFLOAD =
        HAO_DIRECT_MX_QK_SCALE_PRELOAD &&
        TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_OFFLOAD;
    static constexpr bool HAO_DIRECT_MX_DUAL_ISSUER =
        TK_HAO_DIRECT_FP4PV_MX_DUAL_ISSUER;
    static constexpr int HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK =
        HAO_DIRECT_NVFP4_QK
            ? (TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES
                   ? 3
                   : TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALE_MASK)
            : 0;
    static constexpr bool HAO_DIRECT_NV_QK_FOLDED_K64_SCALES =
        HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK == 3;
    static constexpr bool HAO_DIRECT_NV_Q_FOLDED_K64_SCALES =
        (HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK & 1) != 0;
    static constexpr bool HAO_DIRECT_NV_K_FOLDED_K64_SCALES =
        (HAO_DIRECT_NV_QK_FOLDED_K64_SCALE_MASK & 2) != 0;
    static constexpr int HAO_DIRECT_NV_QK_PRELOAD_PAGE_MASK =
        TK_HAO_DIRECT_FP4PV_NV_QK_PRELOAD_PAGE_MASK;
    static constexpr bool HAO_DIRECT_MX_Q3_CORR_WG =
        HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG;
    static constexpr bool HAO_DIRECT_MX_DUAL_Q3_CORR_WG =
        HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG;
    static constexpr bool HAO_DIRECT_MX_DUAL_Q3_SMEM_WG =
        HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG;
    static constexpr bool HAO_DIRECT_MX_DUAL_Q3_TMEM_WG =
        HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG;
    static constexpr bool HAO_DIRECT_MX_SAMPLED_STABILIZER =
        HAO_DIRECT_MX_LOG2_QUANT &&
        TK_HAO_DIRECT_FP4PV_MX_SAMPLED_STABILIZER;
    static constexpr bool HAO_DIRECT_MX_RETAIN_Q0 =
        HAO_DIRECT_MXFP4_PV &&
        HAO_DIRECT_MX_LOG2_QUANT &&
        TK_HAO_DIRECT_FP4PV_MX_RETAIN_Q0;
    static constexpr bool HAO_DIRECT_MX_PAIR_LOAD_SCAN =
        HAO_DIRECT_MXFP4_PV &&
        HAO_DIRECT_MX_LOG2_QUANT &&
        TK_HAO_DIRECT_FP4PV_MX_PAIR_LOAD_SCAN;
    static constexpr bool HAO_DIRECT_MX_DELAYED_HALF_Q2 =
        HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
        HAO_DIRECT_EARLY_P &&
        (HAO_DIRECT_MX_PAIR_LOAD_SCAN ||
         HAO_DIRECT_NV_SHIFTLESS_SOFTMAX) &&
        TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2;
    static constexpr bool HAO_DIRECT_MX_DELAYED_EARLY_Q3 =
        HAO_DIRECT_MX_DELAYED_HALF_Q2 &&
        TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3;
    static constexpr int HAO_DIRECT_MX_CAUSAL_Q3_LOOKAHEAD =
        TK_HAO_DIRECT_CAUSAL &&
                HAO_DIRECT_MX_DELAYED_EARLY_Q3
            ? TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_LOOKAHEAD
            : 0;
    static constexpr bool HAO_DIRECT_MX_CAUSAL_Q3_PROGRESSIVE_REUSE =
        HAO_DIRECT_MX_CAUSAL_Q3_LOOKAHEAD == 1 &&
        TK_HAO_DIRECT_SEQ_LEN >= 4096 &&
        TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_PROGRESSIVE_REUSE;
    static constexpr bool HAO_DIRECT_MX_EARLY_Q2_REDUCE =
        HAO_DIRECT_MX_DELAYED_HALF_Q2 &&
        TK_HAO_DIRECT_FP4PV_MX_EARLY_Q2_REDUCE;
    static constexpr bool HAO_DIRECT_MX_STAGE1_EARLY_Q1 =
        TK_HAO_DIRECT_FP4PV_MX_STAGE1_EARLY_Q1;
    static constexpr bool HAO_DIRECT_MX_Q1_MAX_INTERLEAVE =
        TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE;
    static constexpr bool HAO_DIRECT_MX_STAGE0_Q1_MAX_INTERLEAVE =
        TK_HAO_DIRECT_FP4PV_MX_STAGE0_Q1_MAX_INTERLEAVE;
    static constexpr bool HAO_DIRECT_MX_Q2_MAX_INTERLEAVE =
        TK_HAO_DIRECT_FP4PV_MX_Q2_MAX_INTERLEAVE;
    static constexpr bool HAO_DIRECT_MX_Q3_MAX_INTERLEAVE =
        TK_HAO_DIRECT_FP4PV_MX_Q3_MAX_INTERLEAVE;
    static constexpr bool HAO_DIRECT_MX_HALF_PREFETCH_Q0 =
        HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
        !HAO_DIRECT_MX_DELAYED_HALF_Q2 &&
        TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q0;
    static constexpr bool HAO_DIRECT_MX_HALF_PREFETCH_Q1 =
        HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
        HAO_DIRECT_MX_PAIR_LOAD_SCAN &&
        TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q1;
    static constexpr bool HAO_DIRECT_MX_HALF_PREFETCH_Q3 =
        HAO_DIRECT_MX_HALF_PREFETCH_Q0 &&
        HAO_DIRECT_MX_PAIR_LOAD_SCAN &&
        TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q3;
    static constexpr bool HAO_DIRECT_MX_TREE_MAX =
        (HAO_DIRECT_MXFP4_PV &&
         HAO_DIRECT_MX_LOG2_QUANT &&
         TK_HAO_DIRECT_FP4PV_MX_TREE_MAX) ||
        HAO_DIRECT_NV_TREE_MAX;
    static constexpr bool HAO_DIRECT_MX_MAX3_REDUCE =
        HAO_DIRECT_MXFP4_PV &&
        HAO_DIRECT_MX_LOG2_QUANT &&
        TK_HAO_DIRECT_FP4PV_MX_MAX3_REDUCE;
    static constexpr bool HAO_DIRECT_MX_MAX3_WIDE_REDUCE =
        HAO_DIRECT_MX_MAX3_REDUCE &&
        TK_HAO_DIRECT_FP4PV_MX_MAX3_WIDE_REDUCE;
    static constexpr bool HAO_DIRECT_P_MAX3_REDUCE =
        HAO_DIRECT_NV_MAX3_REDUCE ||
        HAO_DIRECT_MX_MAX3_REDUCE;
    static constexpr bool HAO_DIRECT_SM103_LDRED =
        TK_HAO_DIRECT_FP4PV_SM103_LDRED;
    static constexpr uint32_t HAO_DIRECT_SM103_LDRED_MASK =
        TK_HAO_DIRECT_FP4PV_SM103_LDRED_MASK & 0xfu;
    static constexpr uint32_t HAO_DIRECT_SM103_LDRED_STAGE0_MASK =
        TK_HAO_DIRECT_FP4PV_SM103_LDRED_STAGE0_MASK & 0xfu;
    static constexpr uint32_t HAO_DIRECT_SM103_LDRED_STAGE1_MASK =
        TK_HAO_DIRECT_FP4PV_SM103_LDRED_STAGE1_MASK & 0xfu;
    static constexpr bool HAO_DIRECT_P_QUARTER_MAX =
        HAO_DIRECT_MXFP4_PV ||
        HAO_DIRECT_NV_QUARTER_SCALE;
    static constexpr bool HAO_DIRECT_P_ILP_PIPELINE =
        TK_HAO_DIRECT_FP4PV_P_ILP_PIPELINE;
    static constexpr bool HAO_DIRECT_FIXED_P_CEILING =
        TK_HAO_DIRECT_FP4PV_FIXED_P_CEILING;
    static constexpr bool HAO_DIRECT_KV_REUSE_CEILING =
        TK_HAO_DIRECT_FP4PV_KV_REUSE_CEILING;
    static constexpr int HAO_DIRECT_SCALE_COPY_CEILING_MASK =
        TK_HAO_DIRECT_FP4PV_SCALE_COPY_CEILING_MASK;
    static constexpr bool HAO_DIRECT_SCORE_PACK_CEILING =
        TK_HAO_DIRECT_FP4PV_SCORE_PACK_CEILING;
    static constexpr bool HAO_DIRECT_ROWMAX_PACK_CEILING =
        TK_HAO_DIRECT_FP4PV_ROWMAX_PACK_CEILING;
    static constexpr bool HAO_DIRECT_FAKE_DENOM_CEILING =
        TK_HAO_DIRECT_FP4PV_FAKE_DENOM_CEILING;
    static constexpr int PV_SCALE_ROWS =
        HAO_DIRECT_NVFP4_PV ? 2 : 1;
    static constexpr bool HAO_DIRECT_CAUSAL =
        TK_HAO_DIRECT_CAUSAL;
    static constexpr bool HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP =
        HAO_DIRECT_CAUSAL &&
        TK_HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP;
    static constexpr bool HAO_DIRECT_CAUSAL_DIAGONAL_MASK =
        HAO_DIRECT_CAUSAL &&
        TK_HAO_DIRECT_CAUSAL_DIAGONAL_MASK;
    static constexpr bool HAO_DIRECT_CAUSAL_INTERLEAVED_KV =
        HAO_DIRECT_CAUSAL &&
        TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV;
    static constexpr bool HAO_DIRECT_NONCAUSAL =
        !HAO_DIRECT_CAUSAL;
    // This is the exact historical publication guard.  It intentionally
    // excludes approximate-quarter and correction-workgroup variants: the
    // packed Q0..Q3 E8M0 word is a replay contract, not generic telemetry.
    static constexpr bool HAO_DIRECT_MX_D4ALL_P_SCALE_PUBLICATION =
        HAO_DIRECT_CAUSAL &&
        HAO_DIRECT_CAUSAL_INTERLEAVED_KV &&
        Dqk == 64 &&
        Nb == 128 &&
        HAO_DIRECT_MXFP4_PV &&
        HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
        HAO_DIRECT_MX_DELAYED_HALF_Q2 &&
        HAO_DIRECT_EARLY_P &&
        HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
        !HAO_DIRECT_MX_REGISTER_SCALE_HANDOFF &&
        !HAO_DIRECT_MX_PAIR_SCALE_SCRATCH &&
        !HAO_DIRECT_MX_Q3_CORR_WG &&
        !HAO_DIRECT_MX_DUAL_Q3_CORR_WG &&
        !HAO_DIRECT_MX_DUAL_Q3_SMEM_WG &&
        !HAO_DIRECT_MX_DUAL_Q3_TMEM_WG &&
        HAO_DIRECT_MX_PWL_EXP2_MODE == 23 &&
        HAO_DIRECT_MX_SCALE_SELECT == 1 &&
        HAO_DIRECT_MX_P_EFFECTIVE_MAX == 6 &&
        HAO_DIRECT_MX_SKIP_ZERO_SCALE_MASK &&
        HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2 == 16 &&
        HAO_DIRECT_MX_MODE23_NATIVE_DENSITY == 4 &&
        HAO_DIRECT_MX_MODE23_NATIVE_QUARTER_MASK == 0xfu &&
        HAO_DIRECT_MX_MODE23_NATIVE_STAGE_MASK == 0x3u &&
        HAO_DIRECT_MX_MAX_SAMPLE_STRIDE == 1 &&
        HAO_DIRECT_MX_MAX_SAMPLE_BIAS_X2 == 0 &&
        HAO_DIRECT_MX_Q1_MAX_SAMPLE_STRIDE == 1 &&
        HAO_DIRECT_MX_PAIR_SCALE_REUSE == 1 &&
        HAO_DIRECT_MX_PAIR_SCALE_PERIOD == 1 &&
        HAO_DIRECT_MX_PAIR_SCALE_STAGE_MASK == 1 &&
        HAO_DIRECT_MX_PAIR_SCALE_DELTA == 0 &&
        HAO_DIRECT_MX_Q3_SCALE_REUSE_STAGE_MASK == 0 &&
        HAO_DIRECT_MX_Q1_SELF_MAX == 3 &&
        HAO_DIRECT_MX_MODE23_SELF_STAGE0_NATIVE &&
        TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD == 0;
    static constexpr bool HAO_DIRECT_TWO_RESIDENT_D128 =
        TK_HAO_DIRECT_FP4PV_TWO_RESIDENT_D128;
    static constexpr bool HAO_DIRECT_N96_DUAL_QUERY =
        TK_HAO_DIRECT_FP4PV_N96_DUAL_QUERY;
    static constexpr bool HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY =
        TK_HAO_DIRECT_FP4PV_DOUBLE_SCORE_SINGLE_QUERY;
    static constexpr bool HAO_DIRECT_D64_DETACHED_P =
        TK_HAO_DIRECT_FP4PV_D64_DETACHED_P;
    static constexpr int HAO_DIRECT_D64_P_BUFFER_MODE =
        HAO_DIRECT_D64_DETACHED_P
            ? TK_HAO_DIRECT_FP4PV_D64_P_BUFFER_MODE
            : 0;
    static constexpr bool HAO_DIRECT_D64_COMPACT_P =
        HAO_DIRECT_D64_P_BUFFER_MODE != 0;
    static constexpr bool HAO_DIRECT_D64_TRIPLE_P =
        HAO_DIRECT_D64_P_BUFFER_MODE >= 2;
    static constexpr int HAO_DIRECT_D64_TRIPLE_P_STAGE =
        HAO_DIRECT_D64_P_BUFFER_MODE == 5 ||
                HAO_DIRECT_D64_P_BUFFER_MODE == 6 ||
                HAO_DIRECT_D64_P_BUFFER_MODE == 8 ||
                HAO_DIRECT_D64_P_BUFFER_MODE == 10 ||
                HAO_DIRECT_D64_P_BUFFER_MODE == 14
            ? 1
            : 0;
    static constexpr bool HAO_DIRECT_D64_TRIPLE_P_QK_FIRST =
        HAO_DIRECT_D64_P_BUFFER_MODE >= 3;
    static constexpr bool HAO_DIRECT_D64_TRIPLE_P_SPLIT_RELEASE =
        HAO_DIRECT_D64_P_BUFFER_MODE == 4 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 6;
    static constexpr bool HAO_DIRECT_D64_TRIPLE_P_KV_RELEASE =
        HAO_DIRECT_D64_P_BUFFER_MODE == 7 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 8;
    static constexpr bool HAO_DIRECT_D64_TRIPLE_P_NO_RELEASE =
        HAO_DIRECT_D64_P_BUFFER_MODE == 9 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 10 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 12;
    static constexpr bool HAO_DIRECT_D64_TEMPORAL_V1_SCALE =
        HAO_DIRECT_D64_P_BUFFER_MODE == 11 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 12 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 13;
    static constexpr bool HAO_DIRECT_D64_TRIPLE_P_LEADER_WAIT =
        HAO_DIRECT_D64_P_BUFFER_MODE == 13 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 14;
    static constexpr bool HAO_DIRECT_D64_V1_OVERWRITE_P0_SCALE =
        HAO_DIRECT_D64_P_BUFFER_MODE == 15 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 16 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 17 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 18;
    static constexpr bool HAO_DIRECT_D64_V1_OVERWRITE_SYNC =
        HAO_DIRECT_D64_P_BUFFER_MODE == 17 ||
        HAO_DIRECT_D64_P_BUFFER_MODE == 18;
    static constexpr int HAO_DIRECT_D64_V1_OVERWRITE_OFFSET =
        HAO_DIRECT_D64_P_BUFFER_MODE == 16 ||
                HAO_DIRECT_D64_P_BUFFER_MODE == 18
            ? 16
            : 0;
    static constexpr bool HAO_DIRECT_SM103_GROUP2 =
        TK_HAO_DIRECT_FP4PV_SM103_GROUP2;
    static constexpr bool HAO_DIRECT_SM103_GROUP2_DUAL_QUERY =
        HAO_DIRECT_SM103_GROUP2 &&
        TK_HAO_DIRECT_FP4PV_SM103_GROUP2_DUAL_QUERY;
    static constexpr bool HAO_DIRECT_SM103_N256 =
        TK_HAO_DIRECT_FP4PV_SM103_N256;
    static constexpr bool HAO_DIRECT_SM103_N256_DUAL_QUERY =
        HAO_DIRECT_SM103_N256 &&
        TK_HAO_DIRECT_FP4PV_SM103_N256_DUAL_QUERY;
    static constexpr bool HAO_DIRECT_SM103_N192_EARLY_K96 =
        HAO_DIRECT_SM103_N256_DUAL_QUERY &&
        Nb == 192 &&
        TK_HAO_DIRECT_FP4PV_SM103_N192_EARLY_K96;
    static constexpr bool HAO_DIRECT_SM103_N192_DEDICATED_P0 =
        HAO_DIRECT_SM103_N256_DUAL_QUERY &&
        Nb == 192 &&
        TK_HAO_DIRECT_FP4PV_SM103_N192_DEDICATED_P0;
    static constexpr bool HAO_DIRECT_SM103_N256_PREFIX_REGS =
        HAO_DIRECT_SM103_N256_DUAL_QUERY &&
        TK_HAO_DIRECT_FP4PV_SM103_N256_PREFIX_REGS;
    static constexpr bool HAO_DIRECT_SM103_N256_DIRECT_PV_HANDOFF =
        HAO_DIRECT_SM103_N256_DUAL_QUERY &&
        TK_HAO_DIRECT_FP4PV_SM103_N256_DIRECT_PV_HANDOFF;
    static constexpr bool HAO_DIRECT_SM103_N256_CHAIN_QK1 =
        HAO_DIRECT_SM103_N256_DUAL_QUERY &&
        TK_HAO_DIRECT_FP4PV_SM103_N256_CHAIN_QK1;
    static constexpr bool HAO_DIRECT_SM103_N256_SPLIT_OUTPUT_SCALES =
        HAO_DIRECT_SM103_N256_DUAL_QUERY &&
        TK_HAO_DIRECT_FP4PV_SM103_N256_SPLIT_OUTPUT_SCALES;
    static constexpr bool HAO_DIRECT_SM103_N256_COMPACT_V =
        HAO_DIRECT_SM103_N256 &&
        TK_HAO_DIRECT_FP4PV_SM103_N256_COMPACT_V;
    static constexpr int HAO_DIRECT_SM103_WIDE_SEGMENTS =
        HAO_DIRECT_SM103_N256 ? (Nb == 192 ? 2 : 3) : 1;
    static constexpr bool HAO_DIRECT_SM103_N256_ASYNC_PREFIX_RESTORE =
        HAO_DIRECT_SM103_N256_SPLIT_OUTPUT_SCALES &&
        TK_HAO_DIRECT_FP4PV_SM103_N256_ASYNC_PREFIX_RESTORE;
    static constexpr bool HAO_DIRECT_N96 =
        HAO_DIRECT_TWO_RESIDENT_D128 ||
        HAO_DIRECT_N96_DUAL_QUERY;
    static constexpr bool HAO_DIRECT_N96_SPLIT_KV_TMA =
        HAO_DIRECT_TWO_RESIDENT_D128 &&
        TK_HAO_DIRECT_N96_SPLIT_KV_TMA;
    static constexpr int HAO_DIRECT_N96_STAGGER_NS =
        HAO_DIRECT_TWO_RESIDENT_D128
            ? TK_HAO_DIRECT_N96_STAGGER_NS
            : 0;
    static constexpr int TOTAL_WGS = 4;
    static constexpr int NUM_WARPS = 16;
    static constexpr int NUM_THREADS = 512;
    // HAO aliases equal-sized FP4 K and V payloads in one pipeline slot.
    // Thirteen 8-KiB payload stages plus thirteen 1-KiB scale stages fit in
    // the 227-KiB SM100 per-CTA shared-memory budget.
    static constexpr int KV_STAGES = TK_HAO_DIRECT_KV_STAGES;
    static constexpr int LOAD_STAGES = KV_STAGES;
    static constexpr int K_SCALE_SMEM_SLOTS = KV_STAGES;
    static constexpr int QUERY_STAGES =
        (HAO_DIRECT_TWO_RESIDENT_D128 ||
         (HAO_DIRECT_SM103_N256 &&
          !HAO_DIRECT_SM103_N256_DUAL_QUERY) ||
         (HAO_DIRECT_SM103_GROUP2 &&
          !HAO_DIRECT_SM103_GROUP2_DUAL_QUERY))
            ? 1
            : 2;
    static constexpr int QUERY_TILES_PER_JOB =
        HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY
            ? 1
            : (HAO_DIRECT_SM103_GROUP2
                   ? CLUSTER_SIZE * QUERY_STAGES
                   : QUERY_STAGES);
    static constexpr int TMEM_COLUMNS =
        HAO_DIRECT_TWO_RESIDENT_D128 ? 256 : 512;
    static constexpr int P_FIRST_CHUNKS = 1;
    // Keep setmaxnreg requests inside their disjoint warp-role branches.
    // This lets ptxas allocate the same role-local register ranges as CUTE
    // instead of capping every branch at the lowest warp-group request.
    static constexpr int SOFTMAX_REGS =
        TK_HAO_DIRECT_FP4PV_SOFTMAX_REGS;
    static constexpr int SOFTMAX_STAGE0_REGS =
        (HAO_DIRECT_MX_Q3_CORR_WG ||
         HAO_DIRECT_MX_DUAL_Q3_CORR_WG ||
         HAO_DIRECT_MX_DUAL_Q3_SMEM_WG ||
         HAO_DIRECT_MX_DUAL_Q3_TMEM_WG ||
         HAO_DIRECT_NV_Q01_CORR_WG)
            ? 144
            : (TK_HAO_DIRECT_FP4PV_SOFTMAX_STAGE0_REGS > 0
                   ? TK_HAO_DIRECT_FP4PV_SOFTMAX_STAGE0_REGS
                   : SOFTMAX_REGS);
    static constexpr int CORRECTION_REGS =
        TK_HAO_DIRECT_FP4PV_CORRECTION_REGS;
    static constexpr int CORRECTION_ACTIVE_REGS =
        (HAO_DIRECT_MX_Q3_CORR_WG ||
         HAO_DIRECT_MX_DUAL_Q3_CORR_WG ||
         HAO_DIRECT_MX_DUAL_Q3_SMEM_WG ||
         HAO_DIRECT_MX_DUAL_Q3_TMEM_WG ||
         HAO_DIRECT_NV_Q01_CORR_WG)
            ? 128
            : CORRECTION_REGS;
    static constexpr int PRODUCER_REGS =
        TK_HAO_DIRECT_FP4PV_PRODUCER_REGS;
    static constexpr int AUXILIARY_REGS = 48;
    static constexpr int EMPTY_REGS = 24;
    // MXFP4 benefits from splitting selected exponent pairs across ALU and
    // MUFU. NVFP4's two block-16 groups regress with that cadence.
    static constexpr int ONLINE_STAGE2_EX2_EMU_PERIOD = 16;
    static constexpr uint32_t ONLINE_STAGE2_EX2_EMU_MASK =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_EX2_EMU_MASK
            : 0;
    static constexpr int ONLINE_STAGE2_EX2_ALU_DEGREE =
        TK_HAO_DIRECT_FP4PV_EX2_ALU_DEGREE;
    static constexpr uint32_t HAO_DIRECT_MX_EX2_Q1_MASK =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_EX2_Q1_MASK
            : 0;
    static constexpr uint32_t HAO_DIRECT_MX_EX2_Q2_MASK =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_EX2_Q2_MASK
            : 0;
    static constexpr uint32_t HAO_DIRECT_MX_EX2_Q3_MASK =
        HAO_DIRECT_MXFP4_PV
            ? TK_HAO_DIRECT_FP4PV_MX_EX2_Q3_MASK
            : 0;
    static constexpr int SEQ_LEN = TK_HAO_DIRECT_SEQ_LEN;
    static constexpr int QUERY_TILES = SEQ_LEN / 128;
    static constexpr int N_TILES = (SEQ_LEN + Nb - 1) / Nb;
    static constexpr int N96_TAIL_KEYS =
        SEQ_LEN - (N_TILES - 1) * Nb;
    static constexpr int N96_VALID_TAIL_QUARTERS =
        (N96_TAIL_KEYS + 31) / 32;
    static constexpr int HEADS = TK_HAO_DIRECT_HEADS;
    static constexpr int KV_HEADS = TK_HAO_DIRECT_KV_HEADS;
    static constexpr int QUERIES_PER_KV_HEAD = HEADS / KV_HEADS;
    static constexpr int BATCH = TK_HAO_DIRECT_BATCH;
    static constexpr int LOGICAL_JOBS =
        BATCH * HEADS * QUERY_TILES / QUERY_TILES_PER_JOB;
    static constexpr int NUM_SM = TK_HAO_DIRECT_NUM_SM;
    static constexpr int PHYSICAL_GRID_CAP =
        TK_HAO_DIRECT_PHYSICAL_GRID_CAP == 0
            ? (HAO_DIRECT_TWO_RESIDENT_D128 ? 2 * NUM_SM : NUM_SM)
            : TK_HAO_DIRECT_PHYSICAL_GRID_CAP;
    static constexpr int TASK_ORDER = TK_HAO_DIRECT_TASK_ORDER;
};

using hao_direct_config = config_fp4pv_hao_direct;

static_assert(
    hao_direct_config::TASK_ORDER == 0 ||
    hao_direct_config::TASK_ORDER == 1);
static_assert(hao_direct_config::KV_HEADS > 0);
static_assert(
    hao_direct_config::HEADS % hao_direct_config::KV_HEADS == 0,
    "query heads must be an integer multiple of KV heads");

static_assert(hao_direct_config::HAO_DIRECT_PIPELINE);
static_assert(
    hao_direct_config::HAO_DIRECT_NVFP4_QK !=
    hao_direct_config::HAO_DIRECT_MXFP4_QK);
static_assert(
    hao_direct_config::HAO_DIRECT_NVFP4_PV !=
    hao_direct_config::HAO_DIRECT_MXFP4_PV);
static_assert(hao_direct_config::ONLINE_FULL_K_TILE);
static_assert(
    hao_direct_config::CLUSTER_SIZE ==
        (hao_direct_config::HAO_DIRECT_SM103_GROUP2 ? 2 : 1));
static_assert(
    hao_direct_config::Dqk == 64 ||
    hao_direct_config::Dqk == 128);
static_assert(hao_direct_config::Dvo == hao_direct_config::Dqk);
static_assert(
    hao_direct_config::Dqk != 64 ||
    hao_direct_config::HAO_DIRECT_NVFP4_QK);
static_assert(hao_direct_config::N_TILES > 0);
static_assert(
    hao_direct_config::SEQ_LEN % 128 == 0,
    "sequence length must contain complete M128 query tiles");
static_assert(
    !hao_direct_config::HAO_DIRECT_N96 ||
        hao_direct_config::SEQ_LEN % 32 == 0,
    "the N96 route masks tails at complete 32-key quarter boundaries");
static_assert(
    hao_direct_config::N96_TAIL_KEYS > 0 &&
        hao_direct_config::N96_TAIL_KEYS <= hao_direct_config::Nb);
static_assert(
    !hao_direct_config::HAO_DIRECT_N96 ||
        (hao_direct_config::N96_VALID_TAIL_QUARTERS >= 1 &&
         hao_direct_config::N96_VALID_TAIL_QUARTERS <= 3));
static_assert(hao_direct_config::KV_STAGES >= 4);
static_assert(
    hao_direct_config::KV_STAGES <=
    (hao_direct_config::Dqk == 64 ? 32 : 13));
static_assert(hao_direct_config::PHYSICAL_GRID_CAP > 0);
static_assert(
    hao_direct_config::PHYSICAL_GRID_CAP <=
        (hao_direct_config::HAO_DIRECT_TWO_RESIDENT_D128
             ? 2 * hao_direct_config::NUM_SM
             : hao_direct_config::NUM_SM));
static_assert(
    hao_direct_config::QUERY_TILES %
        hao_direct_config::QUERY_TILES_PER_JOB == 0);
static_assert(
    hao_direct_config::QK_SCALE_CHUNKS == 1 ||
    hao_direct_config::QK_SCALE_CHUNKS == 2);
static_assert(hao_direct_config::P_FIRST_CHUNKS == 1);
static_assert(
    hao_direct_config::TMEM_COLUMNS == 256 ||
    hao_direct_config::TMEM_COLUMNS == 512);
static_assert(
    !hao_direct_config::HAO_DIRECT_TWO_RESIDENT_D128 ||
    (hao_direct_config::Dqk == 128 &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
     hao_direct_config::HAO_DIRECT_MX_DELAYED_HALF_Q2 &&
     hao_direct_config::HAO_DIRECT_MX_DELAYED_EARLY_Q3 &&
     hao_direct_config::Nb == 96 &&
     hao_direct_config::QUERY_STAGES == 1 &&
     hao_direct_config::TMEM_COLUMNS == 256));
static_assert(
    !hao_direct_config::HAO_DIRECT_N96_DUAL_QUERY ||
    (hao_direct_config::Dqk == 128 &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
     hao_direct_config::HAO_DIRECT_MX_DELAYED_HALF_Q2 &&
     hao_direct_config::HAO_DIRECT_MX_DELAYED_EARLY_Q3 &&
     hao_direct_config::Nb == 96 &&
     hao_direct_config::QUERY_STAGES == 2 &&
     hao_direct_config::TMEM_COLUMNS == 512));
static_assert(
    !hao_direct_config::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY ||
    (hao_direct_config::Dqk == 128 &&
     hao_direct_config::Dvo == 128 &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     hao_direct_config::HAO_DIRECT_FIXED_P_CEILING &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_P &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
     !hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD &&
     hao_direct_config::QUERY_STAGES == 2 &&
     hao_direct_config::QUERY_TILES_PER_JOB == 1 &&
     hao_direct_config::TMEM_COLUMNS == 512),
    "double-score gate requires fixed-P NV/MX D128 N128");
static_assert(
    TK_HAO_DIRECT_FP4PV_D64_DETACHED_P == 0 ||
    TK_HAO_DIRECT_FP4PV_D64_DETACHED_P == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_D64_P_BUFFER_MODE >= 0 &&
    TK_HAO_DIRECT_FP4PV_D64_P_BUFFER_MODE <= 18);
static_assert(
    !hao_direct_config::HAO_DIRECT_D64_DETACHED_P ||
    (hao_direct_config::Dqk == 64 &&
     hao_direct_config::Dvo == 64 &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     (hao_direct_config::HAO_DIRECT_MXFP4_PV ||
      hao_direct_config::HAO_DIRECT_NVFP4_PV) &&
     !hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD &&
     hao_direct_config::QUERY_STAGES == 2 &&
     hao_direct_config::TMEM_COLUMNS == 512),
    "detached P requires a two-stage NV-QK D64 N128 topology");
static_assert(
    TK_HAO_DIRECT_FP4PV_D64_P_BUFFER_MODE == 0 ||
    hao_direct_config::HAO_DIRECT_D64_DETACHED_P,
    "D64 P-buffer experiments require detached P");
static_assert(
    !hao_direct_config::HAO_DIRECT_SM103_GROUP2 ||
    (hao_direct_config::Dqk == 128 &&
     hao_direct_config::Dvo == 128 &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     !hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES &&
     !hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_P &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
     (hao_direct_config::QUERY_STAGES == 1 ||
      hao_direct_config::HAO_DIRECT_SM103_GROUP2_DUAL_QUERY) &&
     hao_direct_config::TMEM_COLUMNS == 512));
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_GROUP2_DUAL_QUERY ||
    (hao_direct_config::HAO_DIRECT_SM103_GROUP2 &&
     hao_direct_config::QUERY_STAGES == 2),
    "group2 dual-query mode requires two local query stages");
static_assert(
    !hao_direct_config::HAO_DIRECT_SM103_N256 ||
    (hao_direct_config::Dqk == 128 &&
     hao_direct_config::Dvo == 128 &&
     (hao_direct_config::Nb == 192 ||
      hao_direct_config::Nb == 256) &&
     (hao_direct_config::QUERY_STAGES == 1 ||
      hao_direct_config::HAO_DIRECT_SM103_N256_DUAL_QUERY) &&
     hao_direct_config::TMEM_COLUMNS == 512 &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     (hao_direct_config::HAO_DIRECT_FIXED_P_CEILING ||
      (hao_direct_config::Nb == 192 &&
       hao_direct_config::HAO_DIRECT_SM103_N256_DUAL_QUERY &&
       hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
       hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS)) &&
     hao_direct_config::SEQ_LEN % hao_direct_config::Nb == 0),
    "SM103 wide-key mode supports fixed P or dual-query N192 real P");
static_assert(
    !hao_direct_config::HAO_DIRECT_SM103_N256_DUAL_QUERY ||
    (hao_direct_config::QUERY_STAGES == 2 &&
     hao_direct_config::KV_STAGES >= 4 &&
     hao_direct_config::KV_STAGES <=
         (hao_direct_config::Nb == 192
              ? 8
              : (hao_direct_config::HAO_DIRECT_SM103_N256_COMPACT_V
                     ? 7
                     : 6)) &&
     hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES &&
     hao_direct_config::HAO_DIRECT_NV_QK_COMPACT_FOLDED_SCALES),
    "SM103 wide-key dual-query stage depth or folded Q/K scales are invalid");
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_N256_PREFIX_REGS ||
    hao_direct_config::HAO_DIRECT_SM103_N256_DUAL_QUERY,
    "register prefix retention requires SM103 N256 dual-query mode");
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_N256_DIRECT_PV_HANDOFF ||
    hao_direct_config::HAO_DIRECT_SM103_N256_DUAL_QUERY,
    "direct PV handoff requires SM103 N256 dual-query mode");
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_N256_CHAIN_QK1 ||
    hao_direct_config::HAO_DIRECT_SM103_N256_DUAL_QUERY,
    "QK1 chaining requires SM103 N256 dual-query mode");
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_N256_SPLIT_OUTPUT_SCALES ||
    (hao_direct_config::HAO_DIRECT_SM103_N256_PREFIX_REGS &&
     hao_direct_config::HAO_DIRECT_SM103_N256_DIRECT_PV_HANDOFF &&
     hao_direct_config::HAO_DIRECT_SM103_N256_CHAIN_QK1),
    "split output scale pages require register prefixes, direct PV handoff, "
    "and QK1 chaining");
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_N256_COMPACT_V ||
    hao_direct_config::HAO_DIRECT_SM103_N256,
    "compact V storage requires SM103 N256 mode");
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_N256_ASYNC_PREFIX_RESTORE ||
    (hao_direct_config::HAO_DIRECT_SM103_N256_PREFIX_REGS &&
     hao_direct_config::HAO_DIRECT_SM103_N256_SPLIT_OUTPUT_SCALES),
    "async prefix restore requires split register-held output prefixes");
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_N192_EARLY_K96 ||
    (hao_direct_config::HAO_DIRECT_SM103_N192_EARLY_K96 &&
     !hao_direct_config::HAO_DIRECT_FIXED_P_CEILING),
    "early K96 publication requires dual-query N192 real P");
static_assert(
    !TK_HAO_DIRECT_FP4PV_SM103_N192_DEDICATED_P0 ||
    (hao_direct_config::HAO_DIRECT_SM103_N192_DEDICATED_P0 &&
     !hao_direct_config::HAO_DIRECT_FIXED_P_CEILING &&
     !hao_direct_config::HAO_DIRECT_SM103_N256_DIRECT_PV_HANDOFF &&
     !hao_direct_config::HAO_DIRECT_SM103_N256_SPLIT_OUTPUT_SCALES &&
     hao_direct_config::HAO_DIRECT_SM103_N256_CHAIN_QK1),
    "dedicated N192 P0 requires dual-query real P with chained QK1");
static_assert(
    !hao_direct_config::HAO_DIRECT_SM103_N256_DIRECT_PV_HANDOFF ||
    hao_direct_config::N_TILES % 2 == 0,
    "direct N256 PV handoff currently requires an even key-tile count");
static_assert(
    TK_HAO_DIRECT_WAIT_HINT == 0 ||
    TK_HAO_DIRECT_WAIT_HINT == 10000000);
static_assert(hao_direct_config::HAO_DIRECT_N96_STAGGER_NS >= 0);
// Stabilized P uses the native E4M3 conversion and may deliberately probe
// scale underflow. The shiftless magic-bias encoder must stay in its audited
// normal-E4M3 range.
static_assert(
    hao_direct_config::HAO_DIRECT_NV_P_GLOBAL_LOG2 <=
        11.392318f &&
    (hao_direct_config::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX
         ? hao_direct_config::HAO_DIRECT_NV_P_GLOBAL_LOG2 >= -3.0f
         : hao_direct_config::HAO_DIRECT_NV_P_GLOBAL_LOG2 >= -10.0f));
static_assert(
    !hao_direct_config::HAO_DIRECT_NV_SHIFTLESS_SOFTMAX ||
    hao_direct_config::HAO_DIRECT_NV_P_GLOBAL_LOG2 == 0.0f ||
    hao_direct_config::HAO_DIRECT_NV_SHIFTLESS_SCALE_LIFT);
static_assert(
    TK_HAO_DIRECT_FP4PV_EARLY_P == 0 ||
    TK_HAO_DIRECT_FP4PV_EARLY_P == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_REGISTER_SCALE_HANDOFF == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_REGISTER_SCALE_HANDOFF == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_REGISTER_SCALE_HANDOFF ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE &&
     TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_SCRATCH == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_SCRATCH == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_SCRATCH ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE &&
     TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT &&
     !TK_HAO_DIRECT_FP4PV_MX_REGISTER_SCALE_HANDOFF));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_EARLY_ASYNC_SCALE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_ASYNC_SCALE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_EARLY_ASYNC_SCALE ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_EARLY_ASYNC_SCALE ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS ||
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX ||
    TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DIRECT_SCALE_SCRATCH == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DIRECT_SCALE_SCRATCH == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DIRECT_SCALE_SCRATCH ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DIRECT_SCALE_SCRATCH ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_MASK >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_MASK <= 3);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD ||
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_MASK != 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_BEFORE_P == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_BEFORE_P == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_BEFORE_P ||
    (TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD &&
     (TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD_MASK & 1)));
static_assert(
    TK_HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP == 0 ||
    TK_HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP == 1);
static_assert(
    TK_HAO_DIRECT_CAUSAL_DIAGONAL_MASK == 0 ||
    TK_HAO_DIRECT_CAUSAL_DIAGONAL_MASK == 1);
static_assert(
    TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV == 0 ||
    TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV == 1);
static_assert(
    !TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV ||
    (hao_direct_config::HAO_DIRECT_CAUSAL &&
     hao_direct_config::Nb == 128));
static_assert(
    !TK_HAO_DIRECT_CAUSAL_DIAGONAL_MASK ||
    (hao_direct_config::HAO_DIRECT_CAUSAL &&
     hao_direct_config::HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::QUERY_STAGES == 2));
static_assert(
    !hao_direct_config::HAO_DIRECT_CAUSAL_STAGE0_TAIL_SKIP ||
    (hao_direct_config::HAO_DIRECT_CAUSAL &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::QUERY_STAGES == 2 &&
     hao_direct_config::QUERY_TILES_PER_JOB == 2 &&
     !hao_direct_config::HAO_DIRECT_N96 &&
     !hao_direct_config::HAO_DIRECT_SM103_N256 &&
     !hao_direct_config::HAO_DIRECT_SM103_GROUP2 &&
     !hao_direct_config::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY &&
     !hao_direct_config::HAO_DIRECT_MX_DUAL_ISSUER &&
     !hao_direct_config::HAO_DIRECT_MX_SPLIT_STAGE_PV &&
     !hao_direct_config::HAO_DIRECT_DENOM_CORR_WG &&
     !hao_direct_config::HAO_DIRECT_MX_QK_SCALE_OFFLOAD &&
     !hao_direct_config::HAO_DIRECT_MX_Q3_CORR_WG &&
     !hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_CORR_WG &&
     !hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_SMEM_WG &&
     !hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG &&
     !hao_direct_config::HAO_DIRECT_NV_Q01_CORR_WG &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     hao_direct_config::HAO_DIRECT_EARLY_P &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
     (!hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD ||
      ((hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_MASK & 1) &&
       hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_BEFORE_P)) &&
     hao_direct_config::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK == 0 &&
     hao_direct_config::HAO_DIRECT_MX_SOFTMAX_V_SCALE_PREFETCH_MASK == 0));
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD ||
    !TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_OFFLOAD == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_OFFLOAD == 1);
static_assert(
    !hao_direct_config::HAO_DIRECT_MX_QK_SCALE_OFFLOAD ||
    (hao_direct_config::Dqk == 128 &&
     hao_direct_config::Dvo == 128 &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::QUERY_STAGES == 2 &&
     hao_direct_config::QUERY_TILES_PER_JOB == 2 &&
     hao_direct_config::TMEM_COLUMNS == 512 &&
     !hao_direct_config::HAO_DIRECT_N96 &&
     !hao_direct_config::HAO_DIRECT_SM103_N256 &&
     !hao_direct_config::HAO_DIRECT_SM103_GROUP2 &&
     !hao_direct_config::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY &&
     !hao_direct_config::HAO_DIRECT_MX_DUAL_ISSUER &&
     !hao_direct_config::HAO_DIRECT_MX_DUAL_Q3_TMEM_WG &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_P &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
     hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_MASK == 3 &&
     hao_direct_config::HAO_DIRECT_NV_QK_PRELOAD_PAGE_MASK == 3 &&
     !hao_direct_config::HAO_DIRECT_MX_SPLIT_STAGE_PV));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DUAL_ISSUER == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DUAL_ISSUER == 1);
static_assert(
    !hao_direct_config::HAO_DIRECT_MX_DUAL_ISSUER ||
    (hao_direct_config::Dqk == 128 &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::QUERY_STAGES == 2 &&
     !hao_direct_config::HAO_DIRECT_SM103_GROUP2 &&
     !hao_direct_config::HAO_DIRECT_N96 &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_P &&
     hao_direct_config::HAO_DIRECT_MX_EARLY_DIRECT_SCALE &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
     !hao_direct_config::HAO_DIRECT_DENOM_CORR_WG &&
     hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD &&
     hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_MASK == 1 &&
     hao_direct_config::HAO_DIRECT_MX_QK_SCALE_PRELOAD_BEFORE_P &&
     !hao_direct_config::HAO_DIRECT_MX_SPLIT_STAGE_PV));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALE_MASK >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALE_MASK <= 3);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_QK_COMPACT_FOLDED_SCALES == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_QK_COMPACT_FOLDED_SCALES == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_QK_COMPACT_FOLDED_SCALES ||
    TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES ||
    TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALE_MASK == 0);
static_assert(
    (!TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES &&
     TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALE_MASK == 0) ||
    TK_HAO_DIRECT_QK_SCALE_MODE == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_QK_PRELOAD_PAGE_MASK >= 1 &&
    TK_HAO_DIRECT_FP4PV_NV_QK_PRELOAD_PAGE_MASK <= 3);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_QK_PRELOAD_PAGE_MASK == 3 ||
    (TK_HAO_DIRECT_FP4PV_NV_QK_FOLDED_K64_SCALES &&
     TK_HAO_DIRECT_FP4PV_MX_QK_SCALE_PRELOAD));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG ||
    !TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG ||
    (!TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG));
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG ||
    (!TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG));
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG ||
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG ||
    !TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG ||
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG ||
    !TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG ||
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG ||
    !TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG ||
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG ||
    !TK_HAO_DIRECT_FP4PV_MX_SPLIT_STAGE_PV);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_EARLY_P ||
    TK_HAO_DIRECT_PV_SCALE_MODE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_FUSED_PV_ISSUE == 0 ||
    TK_HAO_DIRECT_FP4PV_FUSED_PV_ISSUE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_SPARSE_PV == 0 ||
    TK_HAO_DIRECT_FP4PV_SPARSE_PV == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR >= 0 &&
    TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR <= 5);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_FIXED_SELECTOR ||
    TK_HAO_DIRECT_FP4PV_SPARSE_PV);
static_assert(
    TK_HAO_DIRECT_FP4PV_SPARSE_SPLIT_K128 == 0 ||
    TK_HAO_DIRECT_FP4PV_SPARSE_SPLIT_K128 == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_SPLIT_K128 ||
    TK_HAO_DIRECT_FP4PV_SPARSE_PV);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_SPLIT_K128 ||
    TK_HAO_DIRECT_FP4PV_EARLY_P);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_PV ||
    TK_HAO_DIRECT_PV_SCALE_MODE == 0);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_PV ||
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_PV ||
    TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_PV ||
    !TK_HAO_DIRECT_FP4PV_FUSED_PV_ISSUE);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_PV ||
    TK_HAO_DIRECT_FP4PV_NV_PRELOAD_V_TAIL == 0);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_PV ||
    TK_HAO_DIRECT_FP4PV_NV_ISSUER_V_SCALE_PREFETCH == 0);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_PV ||
    TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE == 0);
static_assert(
    !TK_HAO_DIRECT_FP4PV_SPARSE_PV ||
    TK_HAO_DIRECT_FP4PV_NV_SOFTMAX_V_SCALE_PREFETCH == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_LOG2_QUANT == 0 ||
    TK_HAO_DIRECT_FP4PV_LOG2_QUANT == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_SCALE_SELECT >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_SCALE_SELECT <= 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_P_EFFECTIVE_MAX == 4 ||
    TK_HAO_DIRECT_FP4PV_MX_P_EFFECTIVE_MAX == 6);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE <= 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BITS == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BITS == 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_LOAD_WIDTH == 4 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_LOAD_WIDTH == 8 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_LOAD_WIDTH == 16);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_PAIR_OVERLAP == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_PAIR_OVERLAP == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_PAIR_OVERLAP ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_LOCAL_DENOM_PIPELINE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_LOCAL_DENOM_PIPELINE == 2);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_LOCAL_DENOM_PIPELINE ||
    (TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS == 4 &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE == 1));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_FACTOR_HOIST == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_FACTOR_HOIST == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_FACTOR_HOIST ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_INT_SCALE_FUSION == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_INT_SCALE_FUSION == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_INT_SCALE_FUSION ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_FACTOR_HOIST &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS == 4 &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE == 1 &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_PAIR_OVERLAP));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SUM_LANES == 1 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SUM_LANES == 2 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SUM_LANES == 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS >= 1 &&
    TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS <= 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_KNOWN_MAX == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_KNOWN_MAX == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_KNOWN_MAX ||
    TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_LAYOUT == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_LAYOUT == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_LAYOUT == 0 ||
    (TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_LOCAL_DENOM_PIPELINE == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE == 1 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE == 2);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS == 4 ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS == 3));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SCALE_SMEM == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SCALE_SMEM == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_SCALE_SMEM ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_LOAD_WIDTH == 16));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_MAGIC_I2F == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_MAGIC_I2F == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_MAGIC_I2F ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_FACTOR_HOIST));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SMEM == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SMEM == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SMEM ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_P));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_PAYLOAD_SMEM == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_PAYLOAD_SMEM == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_PAYLOAD_SMEM ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_LOAD_WIDTH == 16 &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_PAIR_OVERLAP &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_P));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SPLIT_PREFETCH == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SPLIT_PREFETCH == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_SPLIT_PREFETCH ||
    (TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_P &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT));
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT &&
     TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_DIRECT_SCALE &&
     !TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM &&
     !TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG &&
     !TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128 == 1);
static_assert(
    !(TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 &&
      TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_BIAS_X8 >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_BIAS_X8 <= 256);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2 >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2 <= 126);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_ANCHOR_AFFINE_HOIST == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_ANCHOR_AFFINE_HOIST == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_ANCHOR_AFFINE_HOIST ||
    (TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 &&
     TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT &&
     TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 23));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_FIXED_ANCHOR_LOG2 >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_FIXED_ANCHOR_LOG2 <= 120);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_FIXED_ANCHOR_LOG2 == 0 ||
    (TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT &&
     !TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 &&
     !TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_STORED_SCALE_SHIFT_LOG2 >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_STORED_SCALE_SHIFT_LOG2 <= 120);
static_assert(
    !(TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 ||
      TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128) ||
    (TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2 +
         TK_HAO_DIRECT_FP4PV_MX_STORED_SCALE_SHIFT_LOG2 <= 126),
    "MX global-anchor and stored-scale shifts must leave at least E8M0 code 1");
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_STORED_SCALE_SHIFT_LOG2 ||
    TK_HAO_DIRECT_PV_SCALE_MODE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_DEBUG_LSE_STAT >= 0 &&
    TK_HAO_DIRECT_FP4PV_DEBUG_LSE_STAT <= 2);
static_assert(
    !(TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 ||
      TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128) ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_P));
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128 ||
    TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3);
static_assert(
    !(TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR32 ||
      TK_HAO_DIRECT_FP4PV_MX_GLOBAL_ANCHOR128) ||
    !(TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32 ||
      TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64));
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BIT_CODE_MODE >= 0 &&
     TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BIT_CODE_MODE <= 5) ||
    TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BIT_CODE_MODE == 7);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BIT_CODE_MODE ||
    TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BITS == 4);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BITS ||
    TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE == 0);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_LOG_LUT_BITS ||
    TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_STRIDE == 1 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_STRIDE == 2 ||
    TK_HAO_DIRECT_FP4PV_MX_DENOM_SAMPLE_STRIDE == 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE == 1 ||
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE == 2 ||
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE == 3 ||
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE == 4 ||
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE == 5 ||
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE == 6 ||
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_STRIDE == 7);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_BIAS_X2 >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_MAX_SAMPLE_BIAS_X2 <= 2);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_SAMPLE_STRIDE >= 1 &&
    TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_SAMPLE_STRIDE <= 7);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE <= 3);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_STAGE_MASK >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_STAGE_MASK <= 3);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q3_SCALE_REUSE_STAGE_MASK >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_Q3_SCALE_REUSE_STAGE_MASK <= 3);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_PERIOD == 1 ||
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_PERIOD == 2 ||
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_PERIOD == 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_DELTA >= -2 &&
    TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_DELTA <= 2);
static_assert(
    !(TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE ||
      TK_HAO_DIRECT_FP4PV_MX_Q3_SCALE_REUSE_STAGE_MASK) ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT &&
     (TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 ||
      (TK_HAO_DIRECT_FP4PV_SM103_N256 &&
       TK_HAO_DIRECT_FP4PV_SM103_WIDE_KEY_TILE == 192))));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE == 4 ||
    TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE == 8 ||
    TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE == 16);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE ||
    (TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE & 1));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE_BIAS_X4 >= -4 &&
    TK_HAO_DIRECT_FP4PV_MX_Q1_PAIR_PROBE_BIAS_X4 <= 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_LAGGED_Q1_SCALE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_LAGGED_Q1_SCALE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_LAGGED_Q1_SCALE ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT &&
     TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 &&
     TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q1_SCALE_OFFLOAD == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_Q1_SCALE_OFFLOAD == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q1_SELF_MAX >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_Q1_SELF_MAX <= 4);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q1_SELF_MAX ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT &&
     TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 23 &&
     (TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE & 1) &&
     !TK_HAO_DIRECT_FP4PV_MX_LAGGED_Q1_SCALE &&
     !TK_HAO_DIRECT_FP4PV_MX_Q1_SCALE_OFFLOAD));
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q1_SCALE_OFFLOAD ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT &&
     TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_CORR_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_SPLIT_PREFETCH &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_EARLY_SPLIT &&
     !TK_HAO_DIRECT_FP4PV_MX_DENOM_PIPELINED_SUM &&
     TK_HAO_DIRECT_FP4PV_MX_PAIR_SCALE_REUSE == 0 &&
     !TK_HAO_DIRECT_FP4PV_MX_LAGGED_Q1_SCALE &&
     !TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG &&
     !TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_SCAN_FUSED_P == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_SCAN_FUSED_P == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_SCAN_FUSED_P ||
    TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_LOCAL_NATIVE_DEFERRED_SCALE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_LOCAL_NATIVE_DEFERRED_SCALE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_LOCAL_NATIVE_DEFERRED_SCALE ||
    TK_HAO_DIRECT_PV_SCALE_MODE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_LOCAL_NATIVE_DEFERRED_SCALE ||
    TK_HAO_DIRECT_FP4PV_MX_LOG2_QUANT);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_BIT_EXP2_MASK & ~0xffffu) == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 <= 23);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK & ~0xfu) == 0);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY2_QUARTER_MASK & ~0xfu) ==
    0);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY3_QUARTER_MASK & ~0xfu) ==
    0);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_DENSITY3_STAGE_MASK & ~0x3u) == 0);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_MODE23_NATIVE_STAGE_MASK & ~0x3u) == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_MODE23_SELF_STAGE0_NATIVE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_MODE23_SELF_STAGE0_NATIVE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_PWL_FOUR_SAMPLE_DENOM == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_PWL_FOUR_SAMPLE_DENOM == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_PWL_FOUR_SAMPLE_DENOM ||
    TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 23);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_FULL_APPROX_DENOM >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_FULL_APPROX_DENOM <= 5);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_FULL_APPROX_DENOM ||
    TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 23);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_FULL_APPROX_DENOM ||
    !TK_HAO_DIRECT_FP4PV_MX_PWL_FOUR_SAMPLE_DENOM);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_STABLE_LSE_LOGSUM == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_STABLE_LSE_LOGSUM == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 ||
    TK_HAO_DIRECT_FP4PV_MX_DIRECT_LOG_CODE == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_EX2_ALU_DEGREE == 1 ||
    TK_HAO_DIRECT_FP4PV_EX2_ALU_DEGREE == 2 ||
    TK_HAO_DIRECT_FP4PV_EX2_ALU_DEGREE == 3);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_EX2_Q1_MASK & ~0xffffu) == 0);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_EX2_Q2_MASK & ~0xffffu) == 0);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_EX2_Q3_MASK & ~0xffffu) == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_POLICY_ID >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_POLICY_ID <= 10);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SCALE_LIFT == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SCALE_LIFT == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 <= 9);
static_assert(
    (TK_HAO_DIRECT_FP4PV_NV_STAGE0_AFFINE_MASK & ~0xfu) == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_STAGE0_AFFINE_MASK == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 != 0);
static_assert(
    (TK_HAO_DIRECT_FP4PV_NV_STAGE1_AFFINE_MASK & ~0xfu) == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_STAGE1_AFFINE_MASK == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 != 0);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_STAGE0_AFFINE_MASK & ~0xfu) == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_STAGE0_AFFINE_MASK == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 23);
static_assert(
    (TK_HAO_DIRECT_FP4PV_MX_STAGE1_AFFINE_MASK & ~0xfu) == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_STAGE1_AFFINE_MASK == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 23);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX == 1 ||
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX == 2 ||
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX == 3 ||
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX == 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 == 7 ||
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 == 9);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX != 2 ||
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 == 9);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX != 3 ||
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 == 9);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_FAST_APPROX != 4 ||
    TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 == 9);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_NATIVE_PAIRS == 4 ||
    TK_HAO_DIRECT_FP4PV_NV_NATIVE_PAIRS == 6 ||
    TK_HAO_DIRECT_FP4PV_NV_NATIVE_PAIRS == 8);
static_assert(
    (TK_HAO_DIRECT_FP4PV_NV_NATIVE6_QUARTER_MASK & ~0xfu) == 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_NATIVE_PAIRS == 4 ||
    (TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 == 9 &&
     (TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE == 1 ||
      TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE == 0)));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE <= 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE == 0 ||
    (TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE == 1 &&
     TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 == 9));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_DENOM_CORR_WG == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_DENOM_CORR_WG == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_DENOM_SMEM_HANDOFF == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_DENOM_SMEM_HANDOFF == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_DENOM_SMEM_HANDOFF ||
    (TK_HAO_DIRECT_FP4PV_NV_DENOM_CORR_WG &&
     TK_HAO_DIRECT_FP4PV_NV_DENOM_WORDS == 4 &&
     TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF));
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_DENOM_CORR_WG ||
    (TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE == 1 &&
     TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32 &&
     TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF &&
     TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS &&
     TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_DENOM_WORDS == 1 ||
    TK_HAO_DIRECT_FP4PV_NV_DENOM_WORDS == 2 ||
    TK_HAO_DIRECT_FP4PV_NV_DENOM_WORDS == 3 ||
    TK_HAO_DIRECT_FP4PV_NV_DENOM_WORDS == 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_DENOM_KNOWN_MAX == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_DENOM_KNOWN_MAX == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_DENOM_EXACT_MAX_TILES >= 0);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64 == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64 == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32 == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32 == 1);
static_assert(
    !(TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32 &&
      TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_SCALE_X8 >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_SCALE_X8 <= 32);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_BIAS_X8 >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_BIAS_X8 <= 32);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_CORR_WG == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_CORR_WG == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_CORR_WG ||
    (TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64 &&
     TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS &&
     TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_PREPACK_Q1 == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_PREPACK_Q1 == 1);
static_assert(
    !(TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_CORR_WG &&
      TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_PREPACK_Q1));
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR_PREPACK_Q1 ||
    (TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64 &&
     TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_DEBUG_LSE_ROWMAX == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_DEBUG_LSE_ROWMAX == 1);
static_assert(
    !(TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR32 ||
      TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64) ||
    (TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE != 0 &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 &&
     TK_HAO_DIRECT_FP4PV_EARLY_P));
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_GLOBAL_ANCHOR64 ||
    TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE == 4);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE <= 5);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE != 3 ||
    TK_HAO_DIRECT_FP4PV_NV_SCALE_SATFINITE);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE != 4 ||
    TK_HAO_DIRECT_FP4PV_NV_SCALE_SATFINITE);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_SCALE_ENCODE != 5 ||
    TK_HAO_DIRECT_FP4PV_NV_SCALE_SATFINITE);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE <= 2);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD <= 3);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD ||
    (TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3 &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_Q2_REDUCE &&
     TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF == 0 &&
     TK_HAO_DIRECT_FP4PV_SPARSE_PV == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_Q01_CORR_WG ||
    (TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE == 1 &&
     TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE &&
     TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_CORR_BYPASS &&
     TK_HAO_DIRECT_FP4PV_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 &&
     TK_HAO_DIRECT_FP4PV_NV_TAIL_Q3_HOOK_WORD == 0 &&
     TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF == 0 &&
     TK_HAO_DIRECT_FP4PV_SPARSE_PV == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_CORR_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_SMEM_WG == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_DUAL_Q3_TMEM_WG == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCHEDULE == 0 ||
    (TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE &&
     TK_HAO_DIRECT_FP4PV_NV_PWL_EXP2 == 9));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_TREE_MAX == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_TREE_MAX == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_TREE_MAX ||
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_MAX3_REDUCE == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_MAX3_REDUCE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_MAX3_REDUCE ||
    TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_MAX3_REDUCE ||
    !TK_HAO_DIRECT_FP4PV_NV_TREE_MAX);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE == 1 ||
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE == 2 ||
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE == 3 ||
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE == 4 ||
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE == 5 ||
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE == 6 ||
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_STRIDE == 7);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_BIAS_X2 >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_MAX_SAMPLE_BIAS_X2 <= 8);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER ||
    (!TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE &&
     TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE == 1 &&
     TK_HAO_DIRECT_FP4PV_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_MX_SAMPLED_STABILIZER == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_PREVIEW_STABILIZER == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_PREVIEW_STABILIZER == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_PREVIEW_WIDTH == 8 ||
    TK_HAO_DIRECT_FP4PV_NV_PREVIEW_WIDTH == 16);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_PREVIEW_BIAS_X8 >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_PREVIEW_BIAS_X8 <= 16);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_PREVIEW_STABILIZER ||
    (!TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE &&
     TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE == 1 &&
     TK_HAO_DIRECT_FP4PV_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_SAMPLED_STABILIZER == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_FULL_ROW_STABILIZER == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_FULL_ROW_STABILIZER == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_FULL_ROW_STABILIZER ||
    (!TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_NV_QUARTER_SCALE &&
     TK_HAO_DIRECT_FP4PV_NV_DENOM_SOURCE == 1 &&
     TK_HAO_DIRECT_FP4PV_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER == 0 &&
     TK_HAO_DIRECT_FP4PV_NV_PREVIEW_STABILIZER == 0 &&
     TK_HAO_DIRECT_FP4PV_MX_SAMPLED_STABILIZER == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_PRELOAD_V_TAIL >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_PRELOAD_V_TAIL <= 2);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_ISSUER_V_SCALE_PREFETCH >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_ISSUER_V_SCALE_PREFETCH <= 3);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_ISSUER_V_SCALE_PREFETCH ||
    (TK_HAO_DIRECT_FP4PV_NV_PRELOAD_V_TAIL == 0 &&
     TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE == 0 &&
     TK_HAO_DIRECT_FP4PV_NV_SOFTMAX_V_SCALE_PREFETCH == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_REGISTER_SCALE_HANDOFF ||
    TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE);
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE == 0 ||
    TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 0 &&
     TK_HAO_DIRECT_FP4PV_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2));
static_assert(
    TK_HAO_DIRECT_FP4PV_NV_SOFTMAX_V_SCALE_PREFETCH >= 0 &&
    TK_HAO_DIRECT_FP4PV_NV_SOFTMAX_V_SCALE_PREFETCH <= 3);
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_SOFTMAX_V_SCALE_PREFETCH ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 0 &&
     TK_HAO_DIRECT_FP4PV_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 &&
     TK_HAO_DIRECT_FP4PV_NV_REUSE_V_SCALE == 0));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_SOFTMAX_V_SCALE_PREFETCH >= 0 &&
    TK_HAO_DIRECT_FP4PV_MX_SOFTMAX_V_SCALE_PREFETCH <= 3);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_SOFTMAX_V_SCALE_PREFETCH ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_P &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 &&
     !TK_HAO_DIRECT_FP4PV_SM103_N256));
static_assert(
    !TK_HAO_DIRECT_FP4PV_NV_EARLY_DIRECT_SCALE ||
    (TK_HAO_DIRECT_PV_SCALE_MODE == 0 &&
     TK_HAO_DIRECT_FP4PV_EARLY_P &&
     (TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 ||
      TK_HAO_DIRECT_FP4PV_NV_Q0_STABILIZER ||
      TK_HAO_DIRECT_FP4PV_NV_PREVIEW_STABILIZER ||
      TK_HAO_DIRECT_FP4PV_NV_FULL_ROW_STABILIZER)));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_SAMPLED_STABILIZER == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_SAMPLED_STABILIZER == 1);
static_assert(
    !hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX ||
    !hao_direct_config::HAO_DIRECT_MX_SAMPLED_STABILIZER);
static_assert(
    int(hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_SOFTMAX) +
        int(hao_direct_config::HAO_DIRECT_MX_SAMPLED_STABILIZER) +
        int(hao_direct_config::HAO_DIRECT_NV_Q0_STABILIZER) +
        int(hao_direct_config::HAO_DIRECT_NV_PREVIEW_STABILIZER) +
        int(hao_direct_config::HAO_DIRECT_NV_FULL_ROW_STABILIZER) +
        int(hao_direct_config::HAO_DIRECT_MX_SCAN_FUSED_P) +
        int(hao_direct_config::
                HAO_DIRECT_MX_LOCAL_NATIVE_DEFERRED_SCALE) <=
    1);
static_assert(
    !hao_direct_config::
        HAO_DIRECT_MX_LOCAL_NATIVE_DEFERRED_SCALE ||
    !hao_direct_config::HAO_DIRECT_EARLY_P);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_RETAIN_Q0 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_RETAIN_Q0 == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_PAIR_LOAD_SCAN == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_PAIR_LOAD_SCAN == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q0 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q0 == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q1 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q1 == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q3 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_HALF_PREFETCH_Q3 == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 ||
    TK_HAO_DIRECT_FP4PV_MX_SHIFTLESS_SOFTMAX ||
    TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_P ||
    (TK_HAO_DIRECT_FP4PV_NV_SHIFTLESS_SOFTMAX &&
     TK_HAO_DIRECT_FP4PV_EARLY_P));
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 ||
    TK_HAO_DIRECT_FP4PV_MX_PAIR_LOAD_SCAN);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2 ||
    !TK_HAO_DIRECT_FP4PV_MX_Q3_CORR_WG);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3 == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3 ||
    TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_LOOKAHEAD == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_LOOKAHEAD == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_LOOKAHEAD ||
    (TK_HAO_DIRECT_CAUSAL &&
     TK_HAO_DIRECT_FP4PV_MX_DELAYED_EARLY_Q3));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_PROGRESSIVE_REUSE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_PROGRESSIVE_REUSE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_PROGRESSIVE_REUSE ||
    (TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_LOOKAHEAD == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_PWL_EXP2 == 23 &&
     TK_HAO_DIRECT_FP4PV_MX_QUANTIZED_DENOM &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_WORDS == 4 &&
     TK_HAO_DIRECT_FP4PV_MX_DENOM_DECODE_MODE == 1 &&
     TK_HAO_DIRECT_FP4PV_MX_LOCAL_DENOM_PIPELINE == 2 &&
     !TK_HAO_DIRECT_FP4PV_MX_PWL_FOUR_SAMPLE_DENOM &&
     !TK_HAO_DIRECT_FP4PV_MX_FULL_APPROX_DENOM));
static_assert(
    !hao_direct_config::HAO_DIRECT_MX_STABLE_LSE_LOGSUM ||
    (hao_direct_config::Dqk == 128 &&
     hao_direct_config::Dvo == 128 &&
     hao_direct_config::HAO_DIRECT_CAUSAL &&
     hao_direct_config::HAO_DIRECT_MX_LOG2_QUANT &&
     hao_direct_config::HAO_DIRECT_MX_GLOBAL_ANCHOR128 &&
     hao_direct_config::HAO_DIRECT_MX_SHIFTLESS_CORR_BYPASS &&
     !hao_direct_config::HAO_DIRECT_DOUBLE_SCORE_SINGLE_QUERY &&
     hao_direct_config::HAO_DIRECT_MX_QUANTIZED_DENOM &&
     hao_direct_config::HAO_DIRECT_MX_DENOM_WORDS == 4 &&
     hao_direct_config::HAO_DIRECT_MX_DENOM_DECODE_MODE == 1 &&
     hao_direct_config::HAO_DIRECT_MX_LOCAL_DENOM_PIPELINE == 2 &&
     hao_direct_config::HAO_DIRECT_MX_DEFER_DENOM_FINALIZE == 2 &&
     hao_direct_config::HAO_DIRECT_MX_DELAYED_HALF_Q2),
    "stable MX saved-LSE logsum is allowlisted only on the causal D128 "
    "four-carrier route");
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_EARLY_Q2_REDUCE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_EARLY_Q2_REDUCE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_EARLY_Q2_REDUCE ||
    TK_HAO_DIRECT_FP4PV_MX_DELAYED_HALF_Q2);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_STAGE1_EARLY_Q1 == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_STAGE1_EARLY_Q1 == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE ||
    TK_HAO_DIRECT_FP4PV_MX_STAGE1_EARLY_Q1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE ||
    hao_direct_config::HAO_DIRECT_MXFP4_PV);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE ||
    hao_direct_config::HAO_DIRECT_P_QUARTER_MAX);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE ||
    hao_direct_config::HAO_DIRECT_MX_PWL_EXP2_MODE >= 8);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_STAGE0_Q1_MAX_INTERLEAVE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_STAGE0_Q1_MAX_INTERLEAVE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_STAGE0_Q1_MAX_INTERLEAVE ||
    (hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     hao_direct_config::HAO_DIRECT_P_QUARTER_MAX &&
     hao_direct_config::HAO_DIRECT_MX_DELAYED_HALF_Q2 &&
     hao_direct_config::HAO_DIRECT_MX_Q1_SELF_MAX >= 2 &&
     hao_direct_config::HAO_DIRECT_MX_PWL_EXP2_MODE >= 8));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q2_MAX_INTERLEAVE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_Q2_MAX_INTERLEAVE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q2_MAX_INTERLEAVE ||
    (TK_HAO_DIRECT_FP4PV_MX_Q1_MAX_INTERLEAVE &&
     TK_HAO_DIRECT_FP4PV_MX_STAGE0_Q1_MAX_INTERLEAVE &&
     TK_HAO_DIRECT_FP4PV_MX_EARLY_Q2_REDUCE &&
     hao_direct_config::HAO_DIRECT_MX_PWL_EXP2_MODE >= 8));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_Q3_MAX_INTERLEAVE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_Q3_MAX_INTERLEAVE == 1);
static_assert(
    !TK_HAO_DIRECT_FP4PV_MX_Q3_MAX_INTERLEAVE ||
    (TK_HAO_DIRECT_FP4PV_MX_Q2_MAX_INTERLEAVE &&
     TK_HAO_DIRECT_FP4PV_MX_CAUSAL_Q3_LOOKAHEAD == 1 &&
     hao_direct_config::HAO_DIRECT_MX_PWL_EXP2_MODE >= 8));
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_TREE_MAX == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_TREE_MAX == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_MAX3_REDUCE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_MAX3_REDUCE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_MX_MAX3_WIDE_REDUCE == 0 ||
    TK_HAO_DIRECT_FP4PV_MX_MAX3_WIDE_REDUCE == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_P_ILP_PIPELINE == 0 ||
    TK_HAO_DIRECT_FP4PV_P_ILP_PIPELINE == 1);
static_assert(
    !hao_direct_config::HAO_DIRECT_MX_RETAIN_Q0 ||
    !hao_direct_config::HAO_DIRECT_MX_PAIR_LOAD_SCAN);
static_assert(
    TK_HAO_DIRECT_FP4PV_FIXED_P_CEILING == 0 ||
    TK_HAO_DIRECT_FP4PV_FIXED_P_CEILING == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_KV_REUSE_CEILING == 0 ||
    TK_HAO_DIRECT_FP4PV_KV_REUSE_CEILING == 1);
static_assert(
    !hao_direct_config::HAO_DIRECT_KV_REUSE_CEILING ||
    (hao_direct_config::HAO_DIRECT_FIXED_P_CEILING &&
     hao_direct_config::Dqk == 128 &&
     hao_direct_config::Dvo == 128 &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::QUERY_STAGES == 2 &&
     !hao_direct_config::HAO_DIRECT_SM103_GROUP2 &&
     !hao_direct_config::HAO_DIRECT_SM103_N256 &&
     !hao_direct_config::HAO_DIRECT_N96),
    "K/V reuse ceiling requires the production fixed-P D128 N128 topology");
static_assert(
    TK_HAO_DIRECT_FP4PV_SCALE_COPY_CEILING_MASK >= 0 &&
    TK_HAO_DIRECT_FP4PV_SCALE_COPY_CEILING_MASK <= 3);
static_assert(
    TK_HAO_DIRECT_FP4PV_SCALE_COPY_CEILING_MASK == 0 ||
    (hao_direct_config::HAO_DIRECT_FIXED_P_CEILING &&
     hao_direct_config::Dqk == 128 &&
     hao_direct_config::Dvo == 128 &&
     hao_direct_config::Nb == 128 &&
     hao_direct_config::HAO_DIRECT_NVFP4_QK &&
     hao_direct_config::HAO_DIRECT_MXFP4_PV &&
     hao_direct_config::HAO_DIRECT_NV_QK_FOLDED_K64_SCALES));
static_assert(
    TK_HAO_DIRECT_FP4PV_SCORE_PACK_CEILING == 0 ||
    TK_HAO_DIRECT_FP4PV_SCORE_PACK_CEILING == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_ROWMAX_PACK_CEILING == 0 ||
    TK_HAO_DIRECT_FP4PV_ROWMAX_PACK_CEILING == 1);
static_assert(
    TK_HAO_DIRECT_FP4PV_FAKE_DENOM_CEILING == 0 ||
    TK_HAO_DIRECT_FP4PV_FAKE_DENOM_CEILING == 1);
static_assert(
    int(hao_direct_config::HAO_DIRECT_FIXED_P_CEILING) +
        int(hao_direct_config::HAO_DIRECT_SCORE_PACK_CEILING) +
        int(hao_direct_config::HAO_DIRECT_ROWMAX_PACK_CEILING) +
        int(hao_direct_config::HAO_DIRECT_FAKE_DENOM_CEILING) <=
    1);
static_assert(
    !hao_direct_config::HAO_DIRECT_FAKE_DENOM_CEILING ||
    (hao_direct_config::HAO_DIRECT_NVFP4_PV
         ? hao_direct_config::HAO_DIRECT_LOG2_QUANT
         : hao_direct_config::HAO_DIRECT_MX_LOG2_QUANT));
static_assert(
    hao_direct_config::SOFTMAX_STAGE0_REGS * 4 +
        hao_direct_config::SOFTMAX_REGS * 4 +
        hao_direct_config::CORRECTION_ACTIVE_REGS * 4 +
        hao_direct_config::PRODUCER_REGS +
        hao_direct_config::AUXILIARY_REGS * 2 +
        (hao_direct_config::HAO_DIRECT_MX_DUAL_ISSUER
             ? hao_direct_config::PRODUCER_REGS
             : hao_direct_config::EMPTY_REGS) <=
    2048);

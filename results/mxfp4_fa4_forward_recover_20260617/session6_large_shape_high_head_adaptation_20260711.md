# Session 6 Task: Large-Sequence And High-Head MXFP4 Adaptation

Date: 2026-07-11

Execute this task continuously. Do not stop at a design memo, compile failure, first
hang, or first slow prototype. Repair implementation and measurement bugs in place.
Stop only after a correct high-load route is benchmarked through the required matrix,
or after the bounded architectures below are implemented/prototyped and rejected by
measured floors with a concrete next requirement.

Keep the Codex session open when finished. Do not commit or push this pass.

## Product Goal And Scope

Adapt MXFP4 FA4 specifically for large H and large S while preserving the current
low/medium-shape routes. The target cells are the measured loss/no-liveness region:

```text
S2048: H32+
S4096: H16+
S8192: H8+
S16384 and S32768: all tested H
batch-factorized equivalents such as B2/S4096/H8 and B4/S4096/H4
```

The current broad matrix and BF16 comparison are authoritative:

- `forward_issue_lane_overlap_bf16_matrix_20260711_report.md`
- `forward_issue_lane_overlap_bf16_matrix_20260711_summary.json`

Do not modify low-shape auto dispatch unless a full regression matrix proves it.

## Starting Facts

- Inputs are already quantized; input quantization is outside timing.
- Large-shape loss profiles show a roughly 4.3k-cycle score/P/softmax chain,
  issue-side P-ready to PV in only 133-136 cycles, eligible warps around 0.41-0.42,
  and TC activity around 14-16%.
- V/output-rescale reordering moved the intended clocks but did not improve wall time.
  Do not retry it.
- Current e16pc uses one quant/P warpgroup and processes a full 128-column score/P
  tile. The source already has guarded two-quant-warpgroup infrastructure, but old
  split routes use fixed/local-max semantics and were not the current score-derived
  e16pc P chain.
- The supported local FP4 TCGEN path can consume K64 chunks; true K32 is unavailable.
- Existing K64-half-ready and K256 diagnostics do not constitute this task's route:
  they did not combine two parallel score-derived e16pc P producers with the current
  pchain behavior.
- S16384+ currently times out in both explicit/fullgrid and persistent/auto probes.
  This is a separate liveness defect and must be localized rather than called a
  performance result.
- Preserve all unrelated dirty files and externally advanced branch history. Do not
  reset, stash, checkout, or revert other-session work.

Read the relevant historical anchors before editing:

- `forward_e16pc_critical_path_optimization_20260711_report.md`
- `forward_stage2_ex2_alu_pack_20260711_report.md`
- `forward_ordered_ledger.md` entries around the 4WG split-P routes, K64-half-ready,
  fake-P ladder, N128/N256/N512 GEMM shape sweep, and rejected head-group prototypes.

## Phase 0: Freeze Baselines And Cheap Decomposition Probe

Use GPU2 and direct preallocation. Check for competing compute processes; do not kill
unrelated jobs.

Reproduce retained FP4 and fastest BF16 p50/min on at least:

- b1/s2048/h32
- b1/s4096/h16, h32, h64
- b1/s8192/h8, h16, h32
- b2/s4096/h8

Record exact route and launch mode. Use 30 samples and rotated order.

Before the source rewrite, perform one bounded host-only head-chunk concurrency probe:

- s4096/h16 as 2 x h8;
- s4096/h32 as 4 x h8 and 2 x h16;
- s8192/h8 as 2 x h4;
- s8192/h16 as 4 x h4.

Prepare contiguous quantized inputs and outputs outside timing. Measure sequential
chunks and genuinely concurrent CUDA-stream launches with correct cross-stream event
timing. Validate concatenated output/LSE. If concurrent chunks robustly beat the
single launch and BF16, retain the evidence and implement a low-overhead host dispatch
candidate. If not, record why and continue immediately to the fused split-P route.

## Phase 1: Localize And Repair S16384+ Liveness

Find the earliest failing sequence for H2 and H4 using a bounded binary/step search
over multiples of 128 between S8192 and S16384. Test Stage2, e16pc, auto, persistent,
and fullgrid independently in fresh processes. Distinguish dispatch rejection, OOM,
host timeout, and device-side semaphore deadlock.

Use the existing target-index sparse profiler and compile-gated timeout/progress
diagnostics. Add only bounded diagnostic state if needed. Identify:

- last completed score index and persistent task;
- warpgroup/role that stops;
- exact semaphore/phase and expected producer;
- whether failure is within one long task or at a task boundary;
- whether phase/counter width, two-slot reuse, causal tail, or grid decomposition is
  responsible.

Fix a local lifecycle bug if proven. Validate the fix at S8192, the first failing S,
S16384, and S32768 with H1/H2/H4, both launch modes where legal, at least 20 bounded
repeated launches, and low-shape regression. Do not hide a deadlock by increasing the
timeout or falling back silently to BF16.

If the root requires the new high-load route, carry the exact invariant into Phase 2
and prove the new route is finite before timing it.

## Phase 2: Modern 4WG Score-Derived Split-P Route

Implement a new explicit/default-off high-load route based on current e16pc semantics,
not the old fixed-scale/local-max split route.

### Roles

Use four warpgroups without the CLC scheduler WG for the first fullgrid prototype:

- existing producer role;
- existing QK/PV/output issue role;
- P quant producer 0: score/P columns 0..63;
- P quant producer 1: score/P columns 64..127.

Use the existing `QUANT_WG0/QUANT_WG1`, split-half score loads, P-stage storage, and
semaphore infrastructure where sound. Add a new narrowly named trait/config rather
than overloading old local-max behavior.

### Required Score-Derived Semantics

Preserve the retained e16pc algorithm:

1. Each quant WG loads its 64 score columns and computes the two corresponding
   32-column block maxima.
2. Publish the four block maxima with warpgroup-scoped readiness. One elected owner
   combines them with `row_max_old` to produce the exact new row max and `acc_scale`.
3. Derive the same four floor-E8M0 P scales, log2 coefficients, reciprocal
   coefficients, and packed scale word as e16pc. Publish these coefficients to both
   P producers.
4. Issue the asynchronous P-scale TMEM store as soon as all four scale bytes are
   known, overlapping it with both P producers.
5. Each P producer applies the retained mixed degree-3 e16 cadence to its own half,
   accumulates an independent denominator partial, performs hardware E2M1 conversion,
   and writes disjoint payload groups.
6. Publish payload visibility with correct proxy ordering. Keep one full-P-ready
   variant and one supported K64-half-ready variant:
   - full-P: PV waits for both producers, then issues the current K128 operation;
   - K64-progressive: each half becomes consumable only after its payload and common
     scale state are legal, and PV issues the existing legal K64 chunks in order.
7. Combine denominator partials and update row sum/correction without changing the
   retained e16pc numerical order more than required. Keep final denominator work off
   the P-ready path where legal.
8. Delay P payload/scale reuse until every consuming PV chunk has committed.

### Synchronization Rules

- No `__syncthreads()` or full-CTA rendezvous inside the per-tile hot path.
- Do not repurpose the fourth WG as both scheduler and quant producer.
- No spinning proxy signal.
- Every semaphore has one documented producer, consumer set, phase owner, and task
  boundary normalization.
- Both quant WGs must agree on logical idx, P slot, scale slot, and persistent task.
- A producer may not overwrite shared P or scale state until both PV halves/full PV
  have committed.
- Keep Stage2/e16pc SASS unchanged when the new route is unselected.

### Resource Gates

The split halves should reduce per-quant-thread score live ranges. Inspect per-role
`setmaxnreg`, selected ptxas, cuobjdump, and SASS.

Reject or repair:

- any stack/spills;
- more than two selected barriers unless each is proven necessary;
- dynamic smem growth that reduces the route below its existing residency class;
- accidental scheduler/control code compiled into the high-load fullgrid route;
- duplicated full-tile exponent/pack work in both P producers.

## Phase 3: Correctness, Lifecycle, And Cycle Proof

Before performance timing, require:

- finite S128/S256/S384/S1024 tail/task cases;
- finite S8192/S16384 high-iteration cases;
- H4/H8/H16/H32 repeated launches;
- exact producer completion counts and one/two legal PV chunk commits per idx;
- no phase drift over at least 100 mixed-shape bounded launches;
- candidate-vs-e16pc output/LSE and candidate-vs-BF16 envelope;
- run-to-run determinism no worse than retained e16pc, with any mismatch investigated
  as a possible ownership race rather than dismissed as approximation.

Extend the sparse profiler with per-half events if needed. At idx0, idx2, a middle
idx, and the final idx, report:

- score load/max/scale rendezvous;
- half0 and half1 exp/pack completion;
- common P-scale issue/completion;
- half/full P-ready;
- K64-0/K64-1 or K128 PV issue;
- P-stage/scale reuse.

The intended success signature is roughly halved exposed exp/pack latency and an
earlier actual first PV issue, not merely simultaneous producer activity.

## Phase 4: High-Load Performance Loop

Time retained e16pc/Stage2/auto, split-full-P, split-K64-progressive, and fastest BF16
in rotated first/reverse order. Use 30 samples, then 60 for finalists.

Required cells:

```text
b1/s2048/h16,h32,h64
b1/s4096/h8,h16,h32,h64
b1/s8192/h4,h8,h16,h32,h64
b1/s16384/h1,h2,h4,h8,h16
b1/s32768/h1,h4,h16 where memory permits
b2/s4096/h8
b4/s4096/h4
```

For each candidate iterate on measured high-load levers only:

- full-P versus K64-progressive handoff;
- one versus two V-load warps if V becomes exposed;
- persistent versus fullgrid only after fullgrid correctness;
- register caps for the two half producers if SASS proves unnecessary live ranges;
- shape-gated use of the fourth WG as quant producer versus the existing scheduler-WG
  route, never both simultaneously.

Do not return to degree-2 EX2, phase masks, issue-lane rescale reorder, generic QK
schedulers, direct nibble classifiers, or cosmetic semaphore movement.

Profile a finalist at h16/s4096, h16/s8192, and h32/s4096. Compare duration, TC/tensor,
issue, eligible, long/short scoreboard, wait, barrier, no-instruction, per-pipe
activity, registers, smem, occupancy, and waves. A route intended for large H/S must
improve steady-state throughput there even if it is slower on small shapes.

## Phase 5: Materialized/Chunked P Fallback Feasibility

Run this phase if either:

- S16384+ fused liveness cannot be repaired locally; or
- the correct split-P route still trails BF16 by more than 10% on the high-load cells.

Large H/S may justify leaving the fully fused decomposition. Use the existing compact
HBM P/V consumer measurements and entrypoints as the consumer bound. Do not build a
full fallback blindly.

First implement/profile a producer-only diagnostic that reuses the exact score-derived
e16pc QK/max/scale/exp/pack logic and writes compact P payload, P scales, and necessary
online-softmax metadata to a documented global layout, with no PV. Measure producer
time and HBM bytes on h16/s4096, h16/s8192, h4/h16 s16384.

Evaluate two legal decompositions:

1. chunked online accumulation: producer writes one large K chunk, existing/persistent
   compact PV consumer produces a partial output, and a device kernel combines partial
   output/LSE state;
2. external-LSE two-pass bound: first pass computes final LSE, second FP4 QK pass
   produces normalized compact P using the existing external-LSE machinery, then the
   optimized compact PV consumer runs once.

For each cell report the sum of measured component times, peak temporary memory, HBM
traffic, and BF16 margin. Build an end-to-end API only if the measured floor can beat
BF16 or if it is the only finite FP4 route at S16384+ and provides a useful liveness
fallback. Correctness must use the same BF16 reference and output/LSE contract.

Do not call a PV-only timing an attention win. Include every producer, reduction,
normalization, and output-layout cost.

## Phase 6: Dispatch And Regression Matrix

Retain a new route only behind a measured shape gate. The host rule may depend on B,
H, and S but must be simple and negligible.

Acceptance hierarchy:

1. finite where current FP4 hangs;
2. correct and lifecycle-stable;
3. repeatably faster than current FP4 on the target region;
4. faster than `BF16_best`, preferably by at least 1.02x.

Re-run the 57-cell matrix from
`forward_issue_lane_overlap_bf16_matrix_20260711_summary.json`, plus all new S16384+
cells. Prove no regression in the prior 37 robust FP4-win cells. Report exact remaining
loss/no-finite cells.

## Deliverables

Write:

`results/mxfp4_fa4_forward_recover_20260617/forward_large_shape_high_head_adaptation_20260711_report.md`

Write machine-readable timing, liveness, sparse-profile, and NCU summaries with prefix:

`forward_large_shape_high_head_adaptation_20260711_`

Append a concise ledger result to:

`results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_ledger.md`

Remove rejected selectors/behavior. Preserve useful diagnostic and benchmark tooling
only when compile-gated or host-only and SASS-neutral when off.

At the end restore all diagnostic/policy defaults to zero, including timeline,
P-chain stamps, target idx, KPIPE, score-reuse, hotplate/selective policies, and
counters. Force a clean default/off build, run finite Stage2 and any retained
high-load smoke, verify empty diagnostics/zero counters, run scoped `git diff --check`,
record external branch movement, and explicitly confirm no commit or push.

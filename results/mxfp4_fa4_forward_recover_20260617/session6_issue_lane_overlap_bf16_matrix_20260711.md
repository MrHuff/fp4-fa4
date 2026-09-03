# Session 6 Task: Issue-Lane Overlap And Broad BF16 Win Matrix

Date: 2026-07-11

Execute this task continuously and end to end. Do not stop after a plan, one smoke,
or the first failed schedule. Fix synchronization, phase, build, benchmark, and
correctness bugs rather than skipping cases. Keep the Codex session open when done.

Do not commit or push this pass.

## Objective

Reorder the current `e16pc` issue lane so independent V/P preparation overlaps the
previous-output rescale wait, leaving that wait immediately before the PV TCGEN
issue. Find the best correct schedule, then measure and optimize the best available
MXFP4 forward route against the fastest local BF16 FA4 implementation across a broad
sequence/head matrix.

The product objective is FP4 faster than BF16 for every supported matrix cell. Do
not claim that objective is met unless every measured cell passes. If it is not met,
identify exact losing cells, their margin, launch mode, and measured bottleneck.

The comparison contract is kernel-only attention forward:

- MXFP4 receives already-quantized Q/K/V, as in the current forward contract.
- BF16 receives BF16 Q/K/V.
- Quantization time is not included.
- Both use causal attention, `Dqk=192`, `Dvo=128`, preallocated outputs where the
  implementation supports them, and the same GPU/load regime.

## Starting State

- Preserve all current unrelated dirty files and externally advanced branch history.
  Do not reset, stash, checkout, or revert any user/other-session changes.
- Preserve the current uncommitted degree/phase and critical-path reports, ledger
  entries, driver fixes, mask trait, and compile-gated sparse profiler.
- Ordinary Stage2 remains the global default.
- Retained explicit/default-off route:

  `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix`

Read first:

- `forward_e16pc_critical_path_optimization_20260711_report.md`
- `forward_stage2_ex2_alu_pack_20260711_report.md`
- `forward_stage2_ex2_degree_phase_20260711_report.md`

The new steady-state profile established:

- h4/s2048 and h8/s4096 producer P-ready is already about 0.88-0.96k cycles ahead
  of issue-side P-ready;
- output-rescale readiness occurs about 3.8-4.0k cycles after issue-loop entry;
- V ready follows roughly 0.4-0.6k cycles later;
- P descriptor/scale observation follows after that;
- actual PV issue is only about 0.13-0.15k cycles after issue-side P-ready.

Therefore do not optimize P production in this pass. The target is the serialized
issue-lane prefix.

## Phase 0: Reproduce Baselines

Use GPU2. Check for competing GPU processes before each timing batch; do not kill
unrelated jobs. Use matched first/reverse timing and direct output/LSE preallocation.

Build with:

```text
MXFP4_FWD_TIMELINE=0
MXFP4_FWD_PCHAIN_STAMPS=0
PCHAIN_TARGET_IDX=0
KPIPE_STAGE=2
SCORE_REUSE_PIPE_STAGE=0
KPIPE_SELECTIVE_POLICY=0
HOTPLATE_SLOT_SCHED=0
HOTPLATE_POLICY=0
POLICY126_COUNTERS=0
```

Reproduce Stage2 and `e16pc` resources/SASS and 30-sample first/reverse p50/min on:

- h4/s2048
- h8/s1024
- h8/s4096
- h16/s4096

Reject a timing batch if both routes enter a visibly shifted load regime. Record all
raw samples and the reason for any rerun.

## Phase 1: Refactor The Exact Rescale Wait

In the active direct-after-rescale issue path, the current non-first PV lane consumes
`direct_rescale_finished[corr_slot]` and toggles
`direct_main_rescale_phase_mask` before V/P staging. Refactor that exact operation
into one force-inlined helper/lambda with a single phase-consumption site per logical
iteration.

Requirements:

- Existing Stage2 and `e16pc` call it at the original location and produce
  byte-identical normalized SASS if possible.
- A candidate trait may move only the call site; it must not change which semaphore,
  corr slot, phase bit, iteration, or output accumulator is used.
- The wait must still complete before `issue_pv*` can issue TCGEN work.
- First iteration behavior remains unchanged.
- QK issue-lane behavior remains unchanged.
- Tail/task-boundary phase state remains unchanged.
- No new full-CTA barrier, spinning loop, global state, TMEM slot, or P-stage slot.
- Prove V scale/payload staging and P descriptor/scale observation do not touch the
  output accumulator protected by `direct_rescale_finished`.

Add compile-gated diagnostic counts or sparse stamps if necessary to prove, for each
run, exactly one rescale wait/phase toggle per required non-first PV iteration and no
PV issue before that wait. Diagnostics must compile out when off.

## Phase 2: Implement Both Overlap Schedules

Create explicit/default-off `e16pc` sibling routes. Do not add runtime branches to
the retained route.

### Schedule V: overlap V staging only

```text
issue-loop entry
  -> wait/stage V scale and payload
  -> wait previous output rescale
  -> acquire/wait P descriptor and scale
  -> issue PV
```

This should hide V preparation while retaining the current P-after-rescale ownership
order.

### Schedule VP: overlap V and P preparation

```text
issue-loop entry
  -> wait/stage V scale and payload
  -> acquire/wait P descriptor and scale
  -> wait previous output rescale
  -> issue PV
```

This maximizes overlap. Holding an observed P slot/scale before output reuse is legal
only if existing P-stage/P-scale lifetime rules remain intact. Audit every reuse
arrival and phase transition; do not infer safety from a finite one-shot run.

If Schedule VP hangs or races, debug the exact ownership transition. A finite V
schedule is not permission to leave VP half-implemented. Conversely, do not add
global synchronization merely to make VP finite.

For each schedule:

1. clean build and selected ptxas;
2. normalized SASS operation counts and ordering around V/P waits, rescale wait, and
   PV TCGEN issue;
3. h4/s1024 finite smoke;
4. repeated same-seed and different-seed determinism;
5. task-boundary/tail cases with 1, 2, 3, and at least 8 score iterations;
6. high-head repeated launches with bounded timeout;
7. sparse idx0 and idx2 profile on h4/s2048, h8/s1024, and h8/s4096.

The expected sparse signature is:

- issue entry -> V ready moves before output-rescale observation;
- output-rescale-ready -> PV issue shrinks;
- producer P-ready -> PV issue does not grow enough to erase the overlap;
- actual PV issue remains after all legal readiness events.

## Phase 3: Candidate Timing And Selection

Benchmark `e16pc`, Schedule V, and Schedule VP in rotated first/reverse order with at
least 30 samples on:

- h4/s2048
- h8/s1024
- h8/s4096
- h16/s4096

Any apparent gain advances to 60-sample isolated first/reverse confirmation. Report
p50, min, p25, p75, and max, not just the best sample.

Correctness must be at least as strong as `e16pc`:

- candidate-vs-e16pc output and LSE should be bit-identical because only legal wait
  ordering changes; any mismatch is presumptive evidence of an ownership race and
  must be debugged;
- candidate-vs-Stage2 and BF16 numerical envelope;
- repeated output/LSE determinism at h8/s4096 and h16/s4096;
- no stack/spills or resource increase;
- no intermittent timeout across at least 100 bounded mixed-shape launches for a
  finalist.

Retain the fastest schedule only if it repeats at least a 1.5% gain on two core
shapes with no material regression. A smaller broad gain may be retained only if
60-sample confidence and sparse timing both support it. Keep a shape-specific route
only with a clear, stable dispatch boundary and at least a 3% gain.

If V wins and VP does not, explain whether the loss is P-slot lifetime pressure,
late producer readiness, or instruction/control overhead using stamps rather than
theory.

Run compact NCU for a real finalist against `e16pc` at h8/s4096 and, if behavior is
shape-dependent, h8/s1024. Record duration, issue active, eligible warps, long/short
scoreboard, wait, barrier, no-instruction, TC/tensor activity, ALU/FMA/XU activity,
and resources.

## Phase 4: Broad FP4 Versus BF16 Matrix

Build a reusable fresh-process matrix driver. It must time the same pre-created inputs
and preallocated outputs repeatedly, with quantization/input preparation outside the
timed region. Rotate implementation order to control clock/load drift.

Primary batch-1 matrix:

```text
sequence = 128, 256, 512, 1024, 2048, 4096, 8192, 16384
heads    = 1, 2, 4, 8, 16, 32
Dqk      = 192
Dvo      = 128
causal   = true
```

Add long/high-load edge cells where memory permits:

```text
(S,H) = (32768,1), (32768,4), (32768,16),
        (4096,64), (8192,64)
```

Add a small batch-factorization check to distinguish head-count behavior from total
task count:

```text
(B,S,H) = (2,1024,8), (2,4096,8), (4,1024,4), (4,4096,4)
```

If a cell is unsupported, record the exact source/dispatch reason; do not silently
drop it. Avoid OOM by running large cells in fresh processes and releasing inputs.

For every cell compare:

1. ordinary Stage2;
2. retained `e16pc`;
3. retained overlap schedule(s);
4. existing FP4 auto-dispatch route;
5. TK BF16 causal FA4 using every legal launch mode:
   - persistent and fullgrid for `S <= 4096` when both are supported;
   - fullgrid for longer sequences;
6. CuTe-DSL/FlashAttention BF16 FA4 where available and finite.

Define `BF16_best` as the minimum repeatable p50 among legal TK BF16 and CuTe BF16
routes for that exact cell. Do not compare FP4 only to a slower BF16 mode. Define
`FP4_best` similarly among legal measured FP4 routes, but retain only routes that pass
correctness and determinism.

Use at least 20 warmups and 30 samples per implementation. Re-run every cell within
2% of parity or with contradictory first/reverse order using 60 samples. Store raw
samples and report:

- p50/min/p25/p75;
- selected FP4 route and launch mode;
- selected BF16 route and launch mode;
- speedup `BF16_best / FP4_best`;
- absolute microsecond margin;
- finite/correct/deterministic status.

Produce a sequence-by-head table/heatmap and explicit lists of winning, parity, and
losing cells. "Faster everywhere" requires every supported cell to have repeatable
speedup greater than 1.0; use a 1.02 engineering margin for a robust win.

## Phase 5: Close Remaining Shape Gaps

If the overlap schedule is correct but the matrix still has BF16-winning cells, do a
bounded measured follow-up rather than immediately ending:

1. Cluster deficits by sequence/head/load regime.
2. For each cluster, test legal FP4 persistent versus fullgrid launch modes and the
   already-correct Stage2/e16pc/V/VP schedules.
3. Permit a shape-gated dispatch rule only when repeated data shows a stable boundary
   and the rule has negligible host overhead.
4. Profile one representative losing cell per distinct cluster with the sparse
   stamps and compact NCU if feasible.
5. Identify whether the remaining gap is launch floor, output-rescale/V issue prefix,
   low grid occupancy, or a different long-sequence limit.

Do not start another generic QK scheduler, direct-nibble, degree-2, K64, extra-TMEM,
or P-production search in this pass. If a cluster needs a deeper kernel rewrite,
state the measured requirement and expected recoverable microseconds/cycles.

## Deliverables And Cleanup

Write:

`results/mxfp4_fa4_forward_recover_20260617/forward_issue_lane_overlap_bf16_matrix_20260711_report.md`

Write machine-readable matrix data and the reusable driver under the same results
directory with the prefix:

`forward_issue_lane_overlap_bf16_matrix_20260711_`

Append a concise result to:

`results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_ledger.md`

Remove rejected route selectors and behavior. Retain a winning explicit route and
shape dispatch only if all gates pass. Retain useful diagnostic/benchmark tooling if
it is compile-gated or host-only and does not alter normal SASS.

At the end restore and verify:

```text
MXFP4_FWD_TIMELINE=0
MXFP4_FWD_PCHAIN_STAMPS=0
PCHAIN_TARGET_IDX=0
KPIPE_STAGE=0
SCORE_REUSE_PIPE_STAGE=0
KPIPE_SELECTIVE_POLICY=0
HOTPLATE_SLOT_SCHED=0
HOTPLATE_POLICY=0
POLICY126_COUNTERS=0
```

Force a clean default/off build; run a finite ordinary Stage2 smoke; verify timeline
and P-chain reads are empty and all policy counters are zero; run scoped
`git diff --check`; report any external branch movement; and explicitly confirm no
commit or push was performed.

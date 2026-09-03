# Session 6 Task: e16pc Critical-Path Profile And Optimization Loop

Date: 2026-07-11

Work continuously through this task. Do not stop after writing a plan or after the
first failed candidate. Fix implementation, build, launch, and measurement bugs in
place. Stop only after a repeatable improvement is retained or the bounded candidate
set below has been measured and rejected with a concrete critical-path explanation.

Do not commit or push this pass. Keep the Codex session open when the pass finishes.

## Starting State

- Checkpoint commit: `1921a8565f405cedba4c21dfa345e2f865d339ad`.
- Preserve the current uncommitted degree/phase report, fit artifacts, driver fixes,
  branch-free mask trait, and ledger entry. Do not reset, stash, or overwrite unrelated
  forward/backward/submodule/generated worktree changes.
- Ordinary Stage2 remains the global default.
- The retained explicit/default-off candidate is:

  `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix`

- On h8/s4096, committed `e16pc` changed Stage2 NCU approximately as follows:
  duration `87.904 -> 82.208 us`, issue `30.57 -> 36.41%`, eligible
  `.40 -> .48`, long scoreboard `3.47 -> 2.75`, wait `1.75 -> 1.26`, and
  TC active `13.52 -> 14.76%`.
- The degree-2 split experiment shortened the local scale-to-exp/pack interval by
  about 47-48 cycles but did not move repeatable wall time. Therefore do not assume
  that the local EX2/pack interval is still the kernel critical path.
- Previous dense Stage2 timeline attribution was useful but inflated absolute spans.
  Source-correlated NCU also failed to provide a usable full source table. Do not
  repeat a broad SourceCounters run that is already known to hang.

Read these reports before editing:

- `forward_stage2_ex2_alu_pack_20260711_report.md`
- `forward_stage2_ex2_degree_phase_20260711_report.md`
- `forward_stage2_bottleneck_followup_20260710_report.md`
- `forward_qk_epilogue_fusion_explore_20260710_report.md`

## Phase 0: Reproduce A Clean Baseline

Use GPU2 and the matched checkpoint build:

```text
MXFP4_FWD_TIMELINE=0
MXFP4_FWD_PCHAIN_STAMPS=0
KPIPE_STAGE=2
SCORE_REUSE_PIPE_STAGE=0
KPIPE_SELECTIVE_POLICY=0
HOTPLATE_SLOT_SCHED=0
HOTPLATE_POLICY=0
POLICY126_COUNTERS=0
```

Before timing, inspect GPU2 for competing processes. Do not kill unrelated jobs.
If the device is busy, wait/retry or record the interference rather than accepting a
drifted baseline. Explain the recent default/off h4/s1024 `.054176 ms` smoke if it is
not reproducible; a smoke number from a different build mode is not a baseline.

Rebuild Stage2 and `e16pc`, record selected ptxas and normalized SASS counts, and run
matched first/reverse timing with direct preallocation and at least 30 samples per
route on:

- h4/s2048
- h8/s1024
- h8/s4096
- h16/s4096

Record p50/min, finite status, run order, and spread. Do not compare candidates from
different build flags or a visibly different GPU load regime.

## Phase 1: Sparse Steady-State Critical-Path Profile

Extend the existing compile-gated P-chain clock stamps into a low-overhead profiler.
This is diagnostic-only and must compile out when
`MXFP4_FWD_PCHAIN_STAMPS=0`.

Requirements:

1. Increase the diagnostic stamp storage/read/reset path if needed; keep one owner
   slot and name every slot in the driver.
2. Add a compile-time target iteration such as
   `TK_FA4_MXFP4_FWD_PCHAIN_TARGET_IDX`, wired through a Makefile variable with a
   default of zero. Profile both idx 0 and a steady-state idx (prefer idx 2; use idx 1
   where the shape has too few iterations).
3. Stamps must be `clock64()` observations from one elected quant lane and one elected
   issue lane. They must not add synchronization, barriers, polling, or ownership
   changes.
4. Capture enough points to derive these spans separately:
   - score TMEM load begin -> load complete;
   - load complete -> P-stage reuse wait complete;
   - reuse complete -> causal mask complete;
   - mask complete -> row/block max complete;
   - max complete -> P-scale derivation complete;
   - P-scale store issue -> TMEM store wait complete;
   - scale derivation -> exp/weight + E2M1 pack complete;
   - exp/pack complete -> payload stores/causal zeroing complete;
   - payload stores -> proxy publication complete;
   - publication/scale completion -> producer P-ready signal;
   - issue-loop entry -> V ready/staged;
   - issue-loop entry -> P descriptor acquired;
   - P descriptor -> P-scale ready observed;
   - producer P-ready -> issue-side P ready observed;
   - issue-side P ready -> actual PV TCGEN issue;
   - producer P-ready -> actual PV TCGEN issue.
5. If producer and issue clocks cannot be compared safely, prove why. They are in the
   same CTA/device clock domain here, but owner matching still must be correct.
6. Reset before every launch. Reject partial/mismatched records rather than folding
   zeros or stale values into medians.

Collect at least 11 valid uncontended fresh-launch records for Stage2 and `e16pc` on
h4/s2048, h8/s1024, and h8/s4096. Use bounded timeouts because repeated high-head
diagnostic launches have previously hung. Report median, p25, and p75 for every span.

The required conclusion is not merely "P is slow." Determine:

- which producer subregion is now longest in `e16pc`;
- whether an earlier producer P-ready produces an earlier PV issue;
- where the 47-cycle degree-2 local saving was absorbed;
- whether PV is instead gated by V staging, output-rescale readiness, P-stage reuse,
  scale-store completion, or another issue-lane dependency.

## Phase 2: Profile-Gated Optimization Candidates

Implement each relevant candidate as a clean explicit/default-off config sibling.
Test it independently against `e16pc` before combining candidates. Do not add runtime
branches to Stage2/e16pc hot paths.

### Candidate A: Denominator Work Off The P-Ready Path

This is the first candidate unless Phase 1 proves the denominator is already fully
hidden.

The current e16pc loop computes each qid's exponent values, accumulates its two
`float2` partial sums, multiplies by `p_q_rcp_coeff`, and serially updates
`tile_sum_scalar_prescaled_dynamic` before payload readiness. `pchainc` only moved the
final `row_sum` recurrence; it did not remove the cross-qid weighted denominator fold
from the readiness chain.

Implement a bounded variant that:

- keeps the exact degree-3 e16pc exponent and hardware E2M1 payload conversion;
- computes one weighted denominator partial per qid without a cross-qid scalar
  recurrence in the payload loop;
- publishes payload and scale readiness as soon as their actual dependencies are
  satisfied;
- performs the four-qid fold and existing `row_sum = row_sum * acc_scale + tile_sum`
  after P-ready, while PV consumes from the other warpgroup;
- preserves the original qid fold order where practical so denominator semantics do
  not change; if compiler scheduling requires a balanced fold, report it as a
  separate numerical candidate;
- does not use the rejected quantized-denominator/direct-nibble F3 approach.

Inspect register count and live ranges. Reject stack/spills or a material resource
increase. The sparse profile must show whether P-ready and PV issue moved, not just a
local source interval.

### Candidate B: Shorter Fused Block-Max Dependency Tree

Attempt this only if `mask -> max` is exposed or Candidate A leaves it dominant.

For each 32-column MX block, replace the single 16-step dependent max chain with a
small number of independent accumulators followed by a balanced reduction. Keep the
same causal masking, `-inf` behavior, block maxima, row maximum, and E8M0 scale
semantics. Confirm in SASS that the longest dependent FMNMX/MAX chain actually
shrinks; source rearrangement alone is insufficient.

Test 2-way and 4-way only if both remain spill-free. Do not combine with Candidate A
until one wins independently.

### Candidate C: Split Payload/Scale Handoff On The Issue Lane

Attempt this only if the profile shows producer P-ready moves but PV issue does not,
or if payload publication and P-scale completion finish at meaningfully different
times.

Use the existing publication/readiness mechanisms where possible
(`p_scale_published_ready`, `p_payload_published_ready`, `p_sc_tmem_ready`). Split the
issue-side sequence so independent V/descriptor/setup work can occur between payload
publication and final P-scale readiness. Preserve proxy ordering and exact slot/phase
ownership. Do not introduce a full-CTA rendezvous, spinning proxy, generic scheduler,
or extra P/score TMEM slot.

If current V-before-P staging already hides the entire separation, document that from
the stamps and reject C without making a cosmetic variant.

### Candidate D: P-Scale Derivation ILP

Attempt this only if `max -> scale derivation` remains exposed. Keep floor E8M0 bits
and `p_q_rcp_coeff` bit-identical. Reduce serial dependencies or redundant live arrays
across the four qids; inspect SASS and register usage. Do not replace it with an
approximate scale, direct nibble classifier, degree-2 EX2, or compare ladder.

## Candidate Gates

For every built candidate:

- selected ptxas: registers, stack, spills, barriers, smem;
- normalized SASS counts and relevant dependency sequence;
- finite h4/s1024 smoke;
- candidate-vs-Stage2 and candidate-vs-e16pc output max abs, relative L2, and LSE max
  abs;
- repeat determinism, including h16/s4096 for any finalist;
- sparse steady-state interval movement;
- matched first/reverse p50/min on h4/s2048, h8/s1024, h8/s4096, h16/s4096.

Keep a candidate only if it produces a repeatable wall-time gain, not merely a shorter
instrumented interval. Prefer at least 1.5% on two shapes with no material regression;
a shape-gated route needs at least 3% repeatable gain and a clear dispatch boundary.
Any apparent win must survive a 60-sample isolated first/reverse confirmation.

Do not retry these exhausted paths:

- degree-2 cadence/phase tuning or coefficient fitting;
- front/even/split degree-3 masks;
- generic interleaved pack schedules;
- direct nibble F1/F2/F3 or quantized denominator;
- QK ownership/hotplate/policy scheduler work;
- K64/K96 or additional score/TMEM slots;
- broad source-correlated NCU that hangs.

## Phase 3: Profile A Real Finalist

Run compact NCU only for a candidate that passes timing and numerical gates. Compare
against matched `e16pc` at h8/s4096; add h4/s2048 only if it explains a shape split.
Record:

- duration;
- issue active and eligible warps/cycle;
- long/short scoreboard, wait, barrier, and no-instruction stalls;
- ALU/FMA/FMA-lite/XU activity and dynamic counts where available;
- TC/tensor activity;
- registers, spills, smem, and occupancy limits.

The expected success signature is an earlier PV issue plus improved eligibility/issue
or lower wait/scoreboard pressure. A local P interval reduction with unchanged PV
issue and wall time is not a finalist.

## Cleanup And Deliverables

Remove rejected config selectors and behavior. Retain diagnostic improvements only if
they remain compile-gated, useful, and zero-cost when off. Preserve `e16pc` as the
fallback and ordinary Stage2 as global default unless a candidate clearly passes all
gates.

Write:

`results/mxfp4_fa4_forward_recover_20260617/forward_e16pc_critical_path_optimization_20260711_report.md`

Append a concise result to:

`results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_ledger.md`

At the end restore and verify:

```text
MXFP4_FWD_TIMELINE=0
MXFP4_FWD_PCHAIN_STAMPS=0
KPIPE_STAGE=0
SCORE_REUSE_PIPE_STAGE=0
KPIPE_SELECTIVE_POLICY=0
HOTPLATE_SLOT_SCHED=0
HOTPLATE_POLICY=0
POLICY126_COUNTERS=0
```

Force a clean default/off build, run a finite ordinary Stage2 smoke, verify timeline
and P-chain reads are empty and all policy counters are zero, run `git diff --check`
on touched files, and report explicitly that no commit or push was performed.

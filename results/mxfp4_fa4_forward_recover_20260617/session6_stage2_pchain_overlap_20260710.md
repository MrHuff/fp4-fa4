# Session 6 Task: Stage2 P-Chain Overlap And Exp/Pack Optimization

Work in `/workspace/codebases/pv/fp4_matmul` and continue from the current default/off restored tree.

## Objective

Improve Stage2 end-to-end forward latency by shortening or overlapping the measured P construction/PV readiness chain. The latest attribution found:

- P exp/weight + payload pack is the largest real-work block.
- P payload publish / P-scale readiness / P-ready adds a substantial tail.
- QK issue-to-commit is already small, so do not move this work into QK.

Do not repeat these exhausted paths:

- K64 half-ready or K64 one-load-wait variants.
- log-domain compare ladders / quantized-denominator ladders.
- four-way sum accumulators.
- QK-tail/native/proxy scheduler policies.
- signal-only early P-ready before the payload and scale are actually visible.

Preserve the current Stage2 baseline as the reference and keep every new behavior behind an explicit compile-time config/flag.

## Step 1: Obtain Low-Perturbation Timing

The dense timeline markers inflate absolute cycle counts. Replace or supplement them with a low-perturbation measurement for these boundaries:

1. P-scale derivation complete.
2. exp/pack complete.
3. payload bytes complete, including causal zeroing.
4. P-scale TMEM store issued.
5. P-scale TMEM wait complete.
6. payload proxy publish complete.
7. P/P-scale ready signal complete.
8. row-sum/correction bookkeeping complete.
9. PV issue begins.

Preferred implementation: sample `clock64()` into registers on one diagnostic lane and write the collected stamps once, after the measured tile, rather than recording a timeline event at every boundary. A set of sparse two-marker builds is also acceptable. Verify that normal timing compiles all diagnostics out and that the diagnostic path does not spill.

Measure h4/s2048, h8/s1024, and h8/s4096 on GPU2. Use this to establish whether the tail is primarily P-scale TMEM store/wait, payload proxy publication, row-sum bookkeeping, or consumer wakeup.

## Step 2: Candidate A - Early Async P-Scale Store, Deferred Wait

The packed P-scale word is available before the exp/pack loop, but the default path stores it to TMEM after payload construction. Implement a route-local Stage2 candidate that:

1. Obtains legal P-scale slot reuse ownership using the existing folded P-stage/P-scale lifetime rules.
2. Issues `fp4pv_store_mxfp4_scale_tmem_32x32b_x1` immediately after P-scale derivation.
3. Does not wait immediately.
4. Runs the existing exp/pack/payload work while the TMEM store is outstanding.
5. Executes `fp4pv_tmem_store_wait()` before publishing `p_sc_tmem_ready` and before any dependent reuse or PV consumption.

The candidate must not signal scale readiness early. Audit proxy and TMEM ordering explicitly. Compare marker deltas, ptxas, determinism, correctness, and wall time with Stage2.

## Step 3: Candidate B - True Early PV Data Readiness

Determine whether PV can legally consume the completed payload and P scale before row-sum/correction bookkeeping finishes.

Implement only if the dependency audit proves that PV needs:

- payload stores and causal zeroing complete,
- payload proxy publication complete,
- P-scale TMEM store wait complete and scale-ready published,
- the existing output-rescale/correction prerequisite,

but does not need the current tile's `row_sum` update to issue PV.

If legal, move the real payload/P-scale-ready publication immediately after those data prerequisites, then finish row-sum/correction bookkeeping while the output/PV owner can proceed. This must be a real data-ready split, not the previously tested `P_READY_BEFORE_RESCALE_WAIT` signal reorder.

Keep the next-tile recurrence serialized on the completed `row_sum`; only PV of the current tile may overlap it. Prove slot lifetime remains correct before timing.

## Step 4: Candidate C - Combine A And B

If A and B are independently finite and either moves its intended cycle interval, test their combination. Reject the combination if it adds waits/barriers, increases long-scoreboard pressure, or regresses both h4/s2048 and h8/s4096.

## Step 5: Exp/Pack Compute Variant

Only after A/B/C are measured, test one bounded compute variant:

- Inspect generated SASS for the Stage2 exp/pack body.
- Try a two-pack software window that issues independent FFMA/`EX2` work before consuming results in sums and `cvt.e2m1` packing. Preserve exact arithmetic and payload bytes.
- Do not retry four-way sum accumulators.
- If the local PTX toolchain supports paired `ex2.approx.f16x2` or `bf16x2`, first prove it in a tiny compile probe. A low-precision route may then be tested as an explicitly approximate candidate, with full output/LSE error and determinism reporting. Do not replace the exact default unless it meets the existing numerical envelope.
- Do not implement compare ladders; prior measurements showed they are far slower than native EX2/F2FP.

## Required Validation

For Stage2 and every candidate, report:

- exact config string and flags,
- ptxas registers, barriers, smem, stack, spill stores/loads,
- finite smoke and determinism,
- output max-abs/relative error and LSE error versus Stage2/reference,
- h4/s2048, h8/s1024, h8/s4096 p50/min with interleaved or repeated ordering,
- low-perturbation cycle deltas for the interval the candidate is intended to move,
- compact NCU only for a candidate that shows a repeatable wall-time gain.

Keep a candidate only if the gain repeats and it does not merely shift latency into another interval. Revert rejected behavior while preserving useful diagnostic infrastructure behind compile-time-off guards.

## Deliverables

Write:

`results/mxfp4_fa4_forward_recover_20260617/forward_stage2_pchain_overlap_20260710_report.md`

Append a concise entry to:

`results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_ledger.md`

Restore default/off at the end, run a finite timeline-off smoke, and leave this Codex session open.

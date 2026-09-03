# Session 6 Task: Continue e16pc Optimization With Degree And Phase Tuning

Work in `/workspace/codebases/pv/fp4_matmul` from committed checkpoint `1921a8565f405cedba4c21dfa345e2f865d339ad` on `tk-fa4-sm100-rewrite`.

## Objective

Improve retained explicit/default-off `e16pc` by tuning the ALU exp2 polynomial degree and static emulation placement, while preserving hardware E2M1 conversion and current softmax denominator semantics.

The global default must remain ordinary Stage2. Keep current `e16pc` as the fallback throughout.

Do not repeat:

- direct E2M1 nibble classifiers F1/F2/F3;
- quantized-denominator semantics;
- log-domain compare ladders;
- K64-half routes or generic scheduler policies;
- the already rejected generic interleaved packing schedules.

## Baselines

Rebuild and record current Stage2 and committed `e16pc` with diagnostics off. Confirm:

- Stage2: 257 MUFU, 0 packed FFMA2, 144 F2FP, 128 E2M1 conversions.
- e16pc: 193 MUFU, 96 packed FFMA2, same F2FP/E2M1 counts.
- both retained kernels remain 168 registers, 1904 B smem, two barriers, zero stack/spills.

Use h4/s2048, h8/s1024, h8/s4096, and h16/s4096 for final comparisons.

## Step 1: Degree-2 Mixed Emulation With Hardware Pack

Add a route-local paired degree-2 ALU exp helper using the local FA4/Sollya coefficients:

```text
c0 = 1.0
c1 = 0.6657850742340087890625
c2 = 0.330107033252716064453125
```

Use the same FTZ clamp, range reduction, packed operations, and exponent reconstruction as the retained degree-3 helper, but only two packed FMAs. Continue feeding the resulting FP32 values to the existing hardware `cvt.rn.satfinite.e2m1x2.f32` pack and existing row-sum accumulation. Do not produce nibbles manually.

Test degree-2 cadences:

- 4 of 16 pairs;
- 4 of 12 pairs;
- 4 of 10 pairs if spill-free;
- 4 of 8 only if the 4-of-10 resource and timing gates justify it.

For each, record SASS MUFU/FFMA2/F2FP/E2M1 counts, ptxas, numerical error, determinism, sparse scale-to-exp/pack interval, and repeated timing. Reject immediately on spills or material numerical degradation.

## Step 2: Static Phase/Placement Sweep

Replace the current period-only selection internally with a compile-time 16-pair mask or equivalent static trait so selection remains convergent and branch-free.

For degree 3, test exactly these four 4-of-16 masks:

```text
tail clustered:  {12,13,14,15}   # committed e16pc reference
front clustered: {0,1,2,3}
evenly spaced:   {3,7,11,15}
split pairs:     {6,7,14,15}
```

For degree 2, test the best two masks from the degree-3 phase sweep, plus the degree-2 tail reference. Do not create a combinatorial sweep.

Confirm static SASS operation counts are identical within a degree and that only instruction ordering changes. Use first-order and reverse-order repeated timing. A phase winner must move the sparse target interval and repeat across at least two shapes.

## Step 3: Row-Sum Scheduling Under The Winner

Only after a degree/phase candidate independently beats committed `e16pc`, retest one bounded row-sum dependency reduction under that exact candidate:

- four independent `float2` accumulators followed by a balanced reduction, or
- an equivalent pairwise reduction that shortens the exposed FADD chain.

This is an interaction retest: the old native-EX2 result was neutral, but e16pc has reduced SFU latency. Keep it only if it now shortens the measured interval and wall time. Report any arithmetic-order numerical delta and high-head determinism.

## Optional Step 4: FP4-Tuned Degree-2 Coefficients

Attempt this only if stock degree 2 is faster or nearly tied but misses the numerical gate.

Collect or reuse representative Stage2 log2 inputs and fit one degree-2 coefficient set minimizing a weighted objective:

- row-sum relative exp error;
- E2M1 payload mismatch after the unchanged hardware conversion;
- extra weight near E2M1 decision thresholds.

Do not optimize coefficients against the timing shapes' outputs directly. Validate on separate seeds/shapes. Compare against stock Sollya degree 2 and retained degree 3.

## Required Validation

For every serious candidate:

- exact config/mask/degree;
- SASS pipe instruction counts;
- registers, smem, barriers, stack/spills;
- finite h4/s1024 smoke;
- candidate-vs-e16pc and candidate-vs-Stage2 output max abs, relative L2, and LSE max abs;
- high-head run-to-run determinism;
- sparse P-chain scale-to-exp/pack and scale-to-ready intervals;
- repeated first/reverse timing on h4/s2048, h8/s1024, h8/s4096;
- h16/s4096 for finalists.

Run compact NCU only for a repeatable finalist, comparing Stage2, committed e16pc, and the finalist:

- duration;
- ALU/FMA/FMA-lite/XU utilization and dynamic pipe counts;
- issue active and eligible warps;
- long scoreboard and wait;
- TC/tensor activity.

## Acceptance And Cleanup

Retain a new explicit candidate only if it is spill-free, deterministic, numerically defensible, shortens the intended interval, and repeats a wall-time gain over committed `e16pc`. Do not replace or remove committed `e16pc` unless the new candidate clearly dominates it.

Remove rejected selectors/config behavior. Keep shared helper extensions only if used by a retained route.

Write:

`results/mxfp4_fa4_forward_recover_20260617/forward_stage2_ex2_degree_phase_20260711_report.md`

Append the concise result to:

`results/mxfp4_fa4_forward_recover_20260617/forward_overlap_loop_20260622_ledger.md`

Restore default/off, run a finite ordinary Stage2 smoke with empty diagnostics, and leave this Codex session open. Do not commit or push this new pass; report first.

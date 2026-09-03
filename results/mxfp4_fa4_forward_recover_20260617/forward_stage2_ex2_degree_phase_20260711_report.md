# Stage2 EX2 Degree And Phase Tuning Report

Date: 2026-07-11

Task: `session6_ex2_degree_phase_loop_20260711.md`

Hardware: NVIDIA GB200, GPU2, SM100a. Experiment timing used the matched
`KPIPE_STAGE=2` checkpoint configuration with timeline, P-chain stamps, selective
policies, hotplate policies, and counters off. Sparse stamps were collected in
separate instrumented builds. The final restored build has every compile gate off.

## Result

No degree or phase candidate repeats a defensible wall-time gain over committed
`e16pc`. The committed route remains the sole explicit/default-off ALU EX2 route,
and ordinary Stage2 remains the global default.

- Stock degree 2 reduces one packed FMA per selected pair, but its LSE error is
  roughly an order of magnitude larger than degree 3 and it does not repeat a
  timing win.
- Moving the four degree-3 pairs changes instruction order while preserving
  operation counts and resources, but no non-tail mask wins both timing orders on
  the long shape.
- Degree-2 split (`0xC0C0`) was the only near-finalist. A larger isolated repeat
  rejected it: five of six shape/order comparisons were slower than `e16pc`, and
  the one gain did not reverse-order repeat.
- The four-way row-sum interaction and NCU finalist profile were not attempted
  because their hard winner gates did not pass.
- A bounded FP4-aware degree-2 fit was attempted. It improves held-out row-sum RMS
  and payload mismatch rates, but worsens maximum relative exp error and remains
  far behind retained degree 3, so it was rejected before kernel integration.

The only retained implementation change is the branch-free compile-time mask
trait. Committed `e16pc` maps to period 16 and mask `0xF000`; its normalized SASS
is byte-identical to the pre-refactor kernel.

## Baseline Reproduction

Static SASS and ptxas reproduced the task baseline exactly:

| Route | MUFU.EX2 | packed FFMA2 | F2FP | E2M1 | Resources |
|---|---:|---:|---:|---:|---|
| Stage2 | 257 | 0 | 144 | 128 | 168 regs, 2 barriers, 1904 B smem, no stack/spills |
| e16pc | 193 | 96 | 144 | 128 | same |

Matched first/reverse p50 timing, ms, with e16pc delta versus Stage2:

| Shape | First Stage2 / e16pc | Reverse Stage2 / e16pc |
|---|---:|---:|
| h4/s2048 | .066240 / .062976 (-4.93%) | .068640 / .065744 (-4.22%) |
| h8/s1024 | .047136 / .047616 (+1.02%) | .048336 / .048080 (-0.53%) |
| h8/s4096 | .105040 / .099552 (-5.22%) | .104704 / .097744 (-6.65%) |
| h16/s4096 | .182752 / .172752 (-5.47%) | .182272 / .174992 (-3.99%) |

The short h8 shape is at the launch-noise boundary; the long and high-head shapes
reproduce the committed e16pc advantage.

## Degree-2 Cadence

The route-local degree-2 helper used the requested coefficients and the same FTZ
clamp, range reduction, exponent reconstruction, hardware E2M1 conversion, and
FP32 denominator accumulation as degree 3.

| Route | MUFU.EX2 | packed FFMA2 | F2FP | E2M1 | ptxas gate |
|---|---:|---:|---:|---:|---|
| d2 4-of-16 | 193 | 64 | 144 | 128 | 168 regs, no stack/spills |
| d2 4-of-12 | 177 | 80 | 144 | 128 | 168 regs, no stack/spills |
| d2 4-of-10 | 161 | 96 | 144 | 128 | 168 regs, no stack/spills |
| d2 4-of-8 | 129 | 128 | 144 | 128 | rejected: 8 B stack, 8 B stores/loads |

`d2e10` was finite and spill-free but reached `4.394531e-3` output max abs on the
h4/s1024 smoke, so it was rejected before the full timing matrix. The spill gate
rejected d2e8 without timing.

First/reverse p50 delta versus matched e16pc:

| Shape | d2 4-of-16 | d2 4-of-12 |
|---|---:|---:|
| h4/s2048 | -0.02% / +0.10% | -1.58% / -0.66% |
| h8/s1024 | +0.97% / +1.46% | +1.15% / +4.74% |
| h8/s4096 | +1.42% / +0.65% | +0.84% / +0.29% |

Numerical error versus Stage2 from first-order runs:

| Shape/route | Output max abs | Output relative L2 | LSE max abs |
|---|---:|---:|---:|
| h4/s2048 d2e16 | 1.464844e-3 | 1.455319e-3 | 4.763603e-4 |
| h4/s2048 d2e12 | 9.765625e-4 | 1.497029e-3 | 6.308556e-4 |
| h8/s1024 d2e16 | 1.220703e-3 | 1.206531e-3 | 4.768372e-4 |
| h8/s1024 d2e12 | 9.765625e-4 | 1.364959e-3 | 6.356239e-4 |
| h8/s4096 d2e16 | 9.765625e-4 | 1.447527e-3 | 4.715919e-4 |
| h8/s4096 d2e12 | 1.953125e-3 | 1.617126e-3 | 9.593964e-4 |

The degree-2 denominator error is material compared with retained degree-3 LSE
errors near `2e-5` on stable runs.

## Degree-3 Phase Sweep

The exact masks were:

- tail: `0xF000`, positions `{12,13,14,15}`;
- front: `0x000F`, positions `{0,1,2,3}`;
- even: `0x8888`, positions `{3,7,11,15}`;
- split: `0xC0C0`, positions `{6,7,14,15}`.

All four kernels have identical `193 MUFU`, `96 packed FFMA2`, `144 F2FP`,
`128 E2M1`, and `263 BRA` counts in the matched build. All use 168 registers,
1904 B smem, two barriers, and zero stack/spills. Tail instructions are identical
to committed e16pc after removing the differing template symbol line.

First/reverse p50 delta versus e16pc:

| Shape | Front | Even | Split |
|---|---:|---:|---:|
| h4/s2048 | +0.76% / -0.22% | +0.73% / +0.15% | +1.51% / -0.15% |
| h8/s1024 | -0.70% / +0.80% | -1.48% / +1.87% | -2.25% / +3.92% |
| h8/s4096 | +2.25% / +1.90% | +2.28% / +1.85% | +1.76% / +2.56% |

Front also reached `3.90625e-3` output max abs in the h4/s1024 smoke. Even and
split were numerically closer to e16pc, but neither repeated a wall-time gain.
Those two masks alone advanced to the bounded degree-2 phase check.

## Degree-2 Phase Sweep

Tail, even, and split all have identical `193 MUFU`, `64 packed FFMA2`,
`144 F2FP`, `128 E2M1`, and `263 BRA` counts, with the same 168-register,
spill-free resource use.

First/reverse p50 delta versus e16pc:

| Shape | Tail | Even | Split |
|---|---:|---:|---:|
| h4/s2048 | +0.25% / +1.64% | +0.23% / +2.35% | -0.48% / +1.72% |
| h8/s1024 | -1.51% / -2.12% | -0.97% / -1.29% | -1.99% / -1.89% |
| h8/s4096 | +1.13% / +0.92% | +1.36% / +1.61% | -0.13% / -0.30% |

The apparent split result triggered a 60-sample isolated confirmation. Delta
versus e16pc was:

| Shape | First | Reverse |
|---|---:|---:|
| h4/s2048 | +0.22% | +2.59% |
| h8/s1024 | +0.28% | +0.65% |
| h8/s4096 | -0.49% | +0.19% |

The only confirmed gain did not reverse-order repeat. Split output max abs versus
Stage2 was `9.765625e-4`, `9.765625e-4`, and `1.953125e-3` on the three shapes;
relative L2 was `1.374e-3`, `1.246e-3`, and `1.489e-3`; LSE max abs was about
`4.77e-4`, `4.88e-4`, and `4.77e-4`. It therefore fails both timing and numerical
acceptance.

## Sparse P-Chain Attribution

Median cycles from 11-run uncontended low-shape stamp passes:

| Shape/route | Scale to exp/pack | Scale to P-ready |
|---|---:|---:|
| h4/s2048 Stage2 | 1352-1356 | 1903-1907 |
| h4/s2048 e16pc | 1284-1290 | 1688-1694 |
| h4/s2048 d3 front/even/split | 1288 / 1291 / 1296 | 1693 / 1693 / 1705 |
| h8/s1024 d3 front/even/split | 1280 / 1281 / 1294 | 1681 / 1690 / 1708 |
| h4/s2048 d2 tail/even/split | 1258 / 1274 / 1237 | 1670 / 1688 / 1644 |
| h8/s1024 d2 tail/even/split | 1261 / 1271 / 1238 | 1670 / 1681 / 1642 |

Degree-3 phase placement moves the target interval by only a few cycles and does
not predict wall time. Degree-2 split genuinely shortens scale-to-exp/pack by
roughly 47-48 cycles and scale-to-ready by 44-51 cycles versus e16pc, but that
local improvement does not survive isolated wall-time confirmation.

Repeated high-head stamp launches can intermittently hang while diagnostics-off
launches remain finite. The issue reproduced with one rejected degree-2 route,
then with a six-route degree-3 batch; fresh-process and full-grid isolation reduced
but did not eliminate it. Low-shape 11-run medians and bounded one-run high-shape
artifacts were retained. No production synchronization was changed for this
diagnostic-only issue.

## Optional Degree-2 Fit

`forward_stage2_ex2_degree2_fit.py` fits on seed 94601 using the prior probe's
representative typical/tail log2 distribution. Its objective includes relative exp
RMS and bias, unchanged E2M1 payload mismatch, extra threshold-adjacent weight,
and maximum relative error. Validation uses held-out seeds 127001 and 20260711
with independent row widths 1024, 2048, and 4096.

The fitted float32 coefficients were:

`c0=1.0, c1=0.6664999723434448 (0x3f2a9fbe), c2=0.3290329873561859 (0x3ea87703)`

Across held-out sets, tuned degree 2 reduced row-sum relative RMS from
`1.21e-4..1.87e-4` to `1.02e-4..1.77e-4` and generally reduced payload mismatch,
but maximum relative exp error worsened from `2.075519e-3` to
`2.233492e-3`. Retained degree 3 remained much better: max relative error
`8.762962e-5`, row-sum RMS `3.58e-6..7.88e-6`, and payload mismatch at most
`1.525879e-5`.

The fit does not dominate stock degree 2, so it was rejected before kernel
integration. This also avoids optimizing coefficients against timing-shape outputs.

## Gated Steps And Cleanup

- Four-way row-sum accumulation was implemented behind the temporary experiment
  trait but not profiled because no degree/phase candidate independently won. The
  hook was removed during cleanup.
- No compact NCU comparison was run because there is no repeatable finalist.
- All temporary degree-2 helpers, tuning configs, dispatch selectors, Makefile
  gates, and row-sum behavior were removed.
- The driver retains only two measurement fixes: explicit candidate-vs-e16pc
  deltas and a launch-mode override used to bisect stamp-only stalls.
- Committed `e16pc` remains selectable and retains hardware E2M1 conversion and
  original denominator semantics.

## Final State

The forced final build succeeded with:

`MXFP4_FWD_TIMELINE=0 MXFP4_FWD_PCHAIN_STAMPS=0 KPIPE_STAGE=0 SCORE_REUSE_PIPE_STAGE=0 KPIPE_SELECTIVE_POLICY=0 HOTPLATE_SLOT_SCHED=0 HOTPLATE_POLICY=0 POLICY126_COUNTERS=0`

Final Stage2/e16pc SASS remains `257/0/144/128` and `193/96/144/128` for
MUFU/packed-FFMA2/F2FP/E2M1. Both are 168 registers, 1904 B smem, two barriers,
and spill-free. The ordinary Stage2 h4/s1024 GPU2 smoke is finite at
`0.054176 ms` p50. Its P-chain read is `[[]]` and every interval count is zero.

No commit or push was performed.

# MXFP4 P-construction speed-of-light study

Date: 2026-07-24

## Scope

All measurements use the HAO-structured TK forward kernel on GB200 with:

- batch 1, sequence 4096, 24 heads, Dqk/Dvo 128
- 152 persistent CTAs, 512 threads per CTA
- two query stages and 512 TMEM columns
- real QK and FP4 PV tensor-core work

The diagnostic ceilings intentionally do not compute attention-correct P. They
retain the production TMEM layout, P-scale SMEM/TMEM copies, synchronization,
two K64 PV MMAs, output ownership, and epilogue.

## Implemented MXFP4 quantizer

The retained path computes each N32 E8M0 P scale in the log domain:

1. Reuse the quarter maximum found during the row-maximum scan.
2. Convert `(quarter_max - row_max) * scale_log2` directly to E8M0.
3. Fold `log2(6) - scale_exponent` into the existing exp2 argument.
4. Pack the normalized values directly from FP32 to E2M1.
5. Construct the float E8M0 scale from exponent bits for the denominator.

This removes the post-exp amax scan, FP32-to-BF16 conversion, reciprocal, and
BF16-to-E2M1 path.

At seed 0, the matched old MX quantizer was `0.304640 ms`. The fused path with
quarter-max reuse is `0.221824 ms` over a 100 ms warmup and 500 ms measurement,
a 27.2% reduction. Accuracy is unchanged within measurement noise:

| Path | Time (ms) | Cosine vs Torch BF16 | RMSE |
|---|---:|---:|---:|
| Old MX quantizer | 0.304640 | 0.970402 | 0.006260 |
| Fused MX quantizer | 0.221824 | 0.970392 | 0.006260 |

The fused path is spill-free at 128 average registers.

## Speed ceilings

Matched short runs on GPU 1:

| Path | What remains in P construction | Time (ms) |
|---|---|---:|
| Fixed P | Constant P payload and identity scales | 0.071648 |
| Score pack | Real score reads and native E2M1 packing, no softmax | 0.084160 |
| Real MX softmax | Max, exp2, E8M0 selection, packing, denominator | 0.221184 |

The real-score read and native FP4 pack add only about `0.0125 ms` over the
fixed-P floor. Max/exp/denominator work adds about `0.1370 ms` over score-pack.

## NCU attribution

One-launch kernel-replay profiles:

| Metric | Fixed P | Score pack | Real MX softmax |
|---|---:|---:|---:|
| NCU duration (us) | 68.288 | 80.608 | 219.552 |
| Tensor instructions | 296,752 | 296,752 | 296,752 |
| Tensor pipe active (%) | 31.58 | 26.56 | 9.67 |
| Memory tensor active (%) | 38.74 | 37.47 | 19.85 |
| Dynamic scheduler instructions | 14,277,144 | 20,709,165 | 82,015,648 |
| Eligible warps / cycle | 0.27 | 0.34 | 0.49 |
| Barrier stall ratio | 0.11 | 0.08 | 0.02 |
| Long-scoreboard stall ratio | 11.88 | 8.95 | 4.18 |

The higher scoreboard ratio in the ceilings reflects short arithmetic paths
waiting on the retained tensor work. The real path is not barrier-bound:
barrier stalls are negligible, while it executes 61.3 million more scheduler
instructions than score-pack before issuing the same tensor work.

Relevant static SASS counts:

| Opcode family | Fixed P | Score pack | Real MX softmax |
|---|---:|---:|---:|
| `MUFU.EX2` | 0 | 0 | 258 |
| `FMNMX*` | 0 | 0 | 136 |
| E2M1 `F2FP*` | 16 | 144 | 144 |
| `FADD*` | 2 | 2 | 168 |

## Rejected experiments

| Experiment | Time (ms) | Accuracy note |
|---|---:|---|
| All values use FP32 bit-constructed exp2 | 0.291168 | cosine 0.969463 |
| Mixed bit-exp mask `0xC0C0` | 0.223552 | cosine 0.970213 |
| Absolute E8M0 magnitude, no row shift | 0.225760 | cosine 0.969368 |

The all-bit exp replaced each MUFU with `FMNMX + FFMA + F2I`; removing the
clamp still replaces one MUFU with two dependent instructions. A selective
split also lost slightly. NCU reports only 0.11 math-pipe-throttle stall ratio,
so GB200 is not currently limited by SFU throughput.

The row-shiftless formulation is mathematically valid for in-range values, but
it retains every N32 maximum, exp2, pack, and denominator reduction. Removing
the global row correction and six static TMEM loads did not shorten the
critical path.

## Format controls

| QK / PV format | Time (ms) | Cosine | RMSE |
|---|---:|---:|---:|
| NVFP4 / NVFP4 | 0.194432 | 0.981251 | 0.004978 |
| NVFP4 / MXFP4 | 0.221824 | 0.970392 | 0.006260 |
| MXFP4 / MXFP4 | 0.219808 | 0.961323 | 0.007342 |

The NV/NV result matches the prior checkpoint, so the MX changes do not
regress that route.

## Conclusion

TMEM capacity still fixes this kernel at one resident CTA and therefore limits
the absolute ceiling. It is not the cause of the current `0.22 ms` MX path:
the same 512-column topology reaches 31.6% tensor-pipe activity and about
`0.072 ms` when P arithmetic is removed.

Native P packing and score movement are also not the main cost. The remaining
target is the serial N32 maximum, exponential, and denominator chain. A useful
next implementation must overlap or eliminate that whole chain. Replacing
MUFU alone with ALU bit arithmetic increases instruction pressure and cannot
close the gap.

The next concrete scheduling experiment is to retain two score quarters and
interleave their max/exp/pack work, so one quarter's independent instructions
cover another quarter's dependencies without allocating another TMEM score
slot.

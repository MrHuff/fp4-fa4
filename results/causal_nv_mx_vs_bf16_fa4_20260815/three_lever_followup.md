# Causal NV/MX three-lever follow-up

Date: 2026-08-15

This follow-up tests three structural changes against the retained causal
NVFP4-QK/MXFP4-PV kernel on GB200. Measurements use
`B1/S4096/Hq32/Hkv8`, a 100 ms warmup, and a 500 ms timing window. Every
functional candidate produced finite output and passed the causal leakage
test bitwise for both the protected output prefix and LSE.

## Clean controls

| D | NV/MX (ms) | HAO BF16 (ms) | Speedup | Registers | Stack/spills |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.156224 | 0.199232 | 1.275x | 128 | 0 / 0 |
| 128 | 0.145408 | 0.210944 | 1.451x | 128 | 0 / 0 |

## 1. Third D64 score bank

D64 leaves columns 384--511 unused after allocating two 128-column score
banks and two 64-column output accumulators. The experiment used that region
as a third score bank, changed score ownership to a three-entry ring, and
issued QK for the next tile before PV for the current tile whenever the bank
lifetime permitted it.

| Variant | Time (ms) | Change from control | Stack/spills |
|---|---:|---:|---:|
| Clean two-bank control | 0.156224 | - | 0 / 0 |
| Three banks, local phase arrays | 0.190432 | +21.9% | 40 B / 0 |
| Three banks, scalar phase state | 0.180240 | +15.4% | 0 / 0 |

Removing the local arrays recovered part of the loss, but not enough. The
third bank does permit earlier QK, yet every softmax reader now pays dynamic
bank selection and a three-bank ownership protocol. That control cost is
larger than the hidden QK interval at D64. The experiment was rejected.

## 2. Specialized D128 diagonal reader

For the diagonal causal tile, reader warp `r` has fully valid quarters below
`r`, one partial quarter at `r`, and fully invalid quarters above `r`. Two
exact implementations specialized the reader around that fact.

| Variant | Time (ms) | Change from control |
|---|---:|---:|
| Clean exact control | 0.145408 | - |
| Specialized mask, retain all score loads | 0.162400 | +11.7% |
| Specialized mask and skip invalid loads | 0.193664 | +33.2% |

Both variants kept 128 registers and had no stack or spills. The mask-only
candidate increased SASS `ISETP` instructions from 460 to 476, `FSEL` from
248 to 258, and branches from 289 to 303. The compiler schedules the original
uniform mask more efficiently than the explicit reader decision tree. Skipped
loads do not shorten the bottom reader warp, which still owns the critical
four-quarter publication path. Both variants were rejected.

## 3. Earlier tail score loads

The retained D128 schedule loads Q2-half0 and Q0, masks and reduces Q0, then
issues Q1 and Q2-half1. Q3 is progressively loaded into Q0 registers as each
half becomes dead. The experiment moved the Q1 and Q2-half1 requests earlier
to overlap them with causal-mask and Q0-max instructions.

| Early request | Time (ms) | Change from control |
|---|---:|---:|
| None, retained schedule | 0.145408 | - |
| Q1 only | 0.148256 | +2.0% |
| Q2-half1 only | 0.147552 | +1.5% |
| Q1 and Q2-half1 after Q2-half0 mask | 0.148544 | +2.2% |
| Q1 and Q2-half1 before both masks | 0.149536 | +2.8% |

All variants retained 128 registers, one barrier, and zero stack/spills. The
existing stagger is already the useful overlap: Q0 mask/max work separates
the tail load issue from its wait while limiting concurrent TMEM requests.
Issuing earlier only increases TMEM queue pressure and extends register live
ranges. No schedule was promoted.

## 4. Final P-chain approximation pass

The final pass targeted the serial Q2/Q3 portion of
`max -> E8M0 scale -> exp2 approximation -> E2M1 pack`. The retained policy
already reuses Q0's scale for Q1 in stage 0. The first experiment also reused
Q2's scale for Q3, either in one stage or both stages. Timings below are paired
against the clean extension in the same process and on identical inputs.
Accuracy is measured against Torch BF16 SDPA.

| Scale policy | Candidate (ms) | Paired control (ms) | Cosine | Relative L2 |
|---|---:|---:|---:|---:|
| Retained exact Q3 scale | 0.145408 | 0.145408 | 0.950341 | 0.316311 |
| Reuse Q2 scale in stage 0 | 0.146496 | 0.145536 | 0.946290 | 0.328550 |
| Reuse Q2 scale in stage 1 | 0.144384 | 0.145696 | 0.945892 | 0.329672 |
| Reuse Q2 scale in both stages | 0.141952 | 0.145408 | 0.941826 | 0.341432 |
| Both stages, add one E8M0 exponent | 0.145408 | 0.145408 | 0.942149 | 0.347487 |

Full reuse exposes a real 2.38% latency ceiling, but it does so by replacing
Q3's data-dependent scale with Q2's scale. The result loses too much range or
precision depending on the score distribution. A fixed exponent correction
restores none of that accuracy and consumes the timing gain.

Sampling 16 of the 32 values used by the initial local max was also rejected:
it measured 0.147776 ms versus a paired 0.145440 ms control, while cosine fell
to 0.935471 and relative L2 rose to 0.355345. The sampled reduction is not on
a schedule where deleting comparisons shortens the critical path, and its
scale estimate is materially worse.

A temporary Q3 predictor then combined stage-1 scale reuse with 4, 8, or 16
fixed Q3 samples. It was one-sided: a probe could enlarge Q3's inherited
range, but could not shrink it. This avoids unsafe underestimation while
retaining much less work than the full max tree.

| Q3 probe samples | Candidate (ms) | Paired control (ms) | Cosine | Relative L2 |
|---:|---:|---:|---:|---:|
| 4 | 0.144672 | 0.145408 | 0.946339 | 0.328454 |
| 8 | 0.143936 | 0.145408 | 0.946689 | 0.327500 |
| 16 | 0.145408 | 0.145408 | 0.947324 | 0.325776 |

Eight samples form a small speed/accuracy Pareto point, about 1.01% faster
than control, but remain clearly less accurate. Sixteen samples consume the
entire speed gain before recovering control accuracy. All predictor builds
used 128 registers, one barrier, 400 bytes of static shared memory, and no
stack or spills. The temporary predictor and its build flags were removed.

## Conclusion

None of the tested changes improves production without a numerical trade-off.
The experimental source gates were removed, and clean rebuilds reproduce the
retained D64 and D128 timings. The remaining gap is not exposed by another
score slot, reader-side causal branching, earlier score requests, sampled
maxima, or local pair-scale prediction in the current ownership model. A
further material gain requires a different P ownership/dataflow design or a
new approximation that jointly predicts the scale and packed values with
better numerical behavior than the local probes tested here.

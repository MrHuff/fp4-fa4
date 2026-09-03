# Wan NV/MX Guard Fix

## Root cause

The 20-step Wan14B failure was not caused by the 128-key QK anchor missing the
row maximum.  The failing layer-39 diagnostics found an exact anchor hit:

- calibration: exact max `677.514221`, anchor-128 miss `0.0`
- held-out: exact max `663.622009`, anchor-128 miss `0.0`

The guard's margin and stored-scale shift can place a P-block scale at E8M0
code 1.  Represented normalization evaluated `(scale / 6) * code_sum`; the
first multiplication is subnormal and can flush to zero.  The mathematically
equivalent `scale * (code_sum / 6)` keeps the intermediate normal because a
winning FP4 block contains the maximum payload code.  The kernel now forces
that operation order, floors translated scales at code 1, and rejects a
combined margin plus stored shift above 126.

## Paired CuTe-BF16 validation

All rows below are cosine / relative L2 against a paired HAO CuTe-DSL BF16
run.  Every listed route completed with finite outputs.

| Model | Prompt | 1 step | 4 steps | 20 steps |
|---|---|---:|---:|---:|
| Wan2.1-14B | calibration | 0.993771 / 0.117878 | 0.933816 / 0.358939 | 0.849586 / 0.533656 |
| Wan2.1-14B | held-out | 0.991538 / 0.153623 | 0.890750 / 0.455639 | 0.823451 / 0.581297 |
| Wan2.1-1.3B | calibration | 0.987025 / 0.180299 | 0.967119 / 0.261059 | 0.913612 / 0.416321 |
| Wan2.1-1.3B | held-out | - | - | 0.882344 / 0.484756 |

The 14B 20-step runs each executed 1,600 self-attention calls.  The 1.3B
20-step runs each executed 1,200.

## Timing

Five-second paired GB200 windows measured:

| Model | Base (ms) | Guard (ms) | Layer-weighted route (ms) |
|---|---:|---:|---:|
| Wan2.1-1.3B | 0.161792 | 0.196960 | 0.165309 |
| Wan2.1-14B | 0.415552 | 0.509984 | 0.424995 |

The production manifests use one global affine map (`A=1.60`, `B=0.95`).
Historical layer-specific maps are disabled because their isolated-layer
teacher proxy did not produce a robust end-to-end improvement.

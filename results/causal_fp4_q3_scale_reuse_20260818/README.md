# Causal NV/MX full-denominator Q3 scale reuse

## Scope

This experiment separates two approximation axes that had previously been
conflated. Every retained run normalizes with all four packed denominator
words, representing all 32 quantized P values in each quarter. Only the Q3
MXFP4 E8M0 scale policy changes.

Measurements use a GB200, `B1/S4096/Hq32/Hkv8`, 100 ms warmup, a 500 ms
timing window, and seeds 0--3. D64 uses interleaved K/V quarters. D128 uses
MSE-selected folded-K64 Q/K scales. Accuracy is measured against Torch BF16
SDPA on the same inputs.

## Multiseed results

| D | Q3 scale policy | Mean time (ms) | Gain vs control | Mean cosine | Mean relative L2 |
|---:|---|---:|---:|---:|---:|
| 64 | Exact Q3 scale | 0.148176 | - | 0.956097 | 0.295628 |
| 64 | Reuse Q2 scale in stage 1 | **0.141304** | **4.86%** | 0.950634 | 0.312981 |
| 64 | Reuse Q2 scale in both stages | 0.139280 | 6.36% | 0.933661 | 0.361469 |
| 128 | Exact Q3 scale | 0.145488 | - | 0.950153 | 0.316660 |
| 128 | Reuse Q2 scale in stage 1 | **0.144040** | **1.01%** | 0.945655 | 0.330119 |
| 128 | Reuse Q2 scale in both stages | 0.142000 | 2.45% | 0.941634 | 0.341717 |

Stage-1 reuse is the practical Pareto point. Both-stage reuse reproduces the
older roughly 0.142 ms / 0.942 cosine speed point, but its additional
accuracy loss is too large to make it the named policy.

## Rejected correction

A one-sided repair tried to increase the inherited scale when Q3 appeared to
need more range. D128 reused the exact Q3 max already computed during Q2
packing. D64 tested 4, 8, and 16 spread Q3 probes.

| D | Repair | Time (ms) | Cosine | Relative L2 |
|---:|---|---:|---:|---:|
| 128 | Existing exact Q3 max | 0.146016 | 0.948031 | 0.323767 |
| 64 | 4 probes | 0.147456 | 0.951826 | 0.309384 |
| 64 | 8 probes | 0.148032 | 0.952279 | 0.308037 |
| 64 | 16 probes | 0.147744 | 0.952970 | 0.306000 |

The repair consumes nearly all of the timing gain while recovering little
accuracy, so its source gates were removed.

## Retained policy

`HAO_FP4PV_MX_POLICY=fast` remains the default and computes an independent
Q3 scale. The opt-in policy is:

```bash
HAO_FP4PV_MX_POLICY=causal-scale-reuse
```

It inherits the production `fast` configuration, retains all four
denominator words, and enables Q2-to-Q3 scale reuse only for softmax stage 1.
On seed 0, clean named-policy builds measured 0.141280 ms at 0.951340 cosine
for D64 and 0.143392 ms at 0.945786 cosine for D128. Both passed the causal
prefix and LSE leakage checks bitwise. The kernels use 128 registers, one
barrier, and no stack or spills.

D192 is not evaluated because the current forward kernel supports only D64
and D128 and has compile-time assertions enforcing those dimensions.

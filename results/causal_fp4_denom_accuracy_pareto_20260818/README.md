# Causal FP4 denominator and 0.97-cosine investigation

## Scope

This experiment tests whether denominator correction can cheaply recover the
accuracy of the causal NVFP4-QK/MXFP4-PV forward kernel. Measurements use a
GB200 at B1/S4096/Hq32/Hkv8 with a BF16 attention output as the reference.
D64 uses interleaved K/V quarters; D128 uses MSE-selected folded-K64 Q/K
scales. Unless marked as a four-seed mean, accuracy values are seed 0.

## Denominator ceiling

A denominator can multiply each output row by one scalar, but it cannot change
the direction of that row's output vector. The benchmark now reports the
best positive least-squares scalar at global, per-head, per-query-tile, and
per-row granularity. The per-row result is an unattainable oracle that uses the
BF16 output and therefore provides a strict denominator-only ceiling.

| Kernel | D | Uncorrected cosine | Per-row oracle | BF16-LSE substitution |
|---|---:|---:|---:|---:|
| fast affine | 64 | 0.956750 | 0.959788 | 0.951965 |
| fast affine | 128 | 0.950080 | 0.954199 | 0.947950 |
| refit cubic, native density 1 | 64 | 0.969350 | 0.971281 | 0.964197 |
| refit cubic, native density 2 in Q0/Q1 | 128 | 0.968787 | 0.970641 | 0.965430 |

The denominator is not the primary error source. Substituting the true BF16
LSE makes every tested kernel worse because the approximate numerator and its
current denominator partially compensate for each other. Cheap fitted
corrections based on query position or the already-computed LSE explain less
than 4% of the per-row oracle variation and do not materially improve cosine.

## Accuracy ladder

| D | P transform | Native samples | Time (ms) | Cosine | Relative L2 | BF16 (ms) | Speedup |
|---:|---|---|---:|---:|---:|---:|---:|
| 64 | affine | none | 0.148080 | 0.956750 | 0.293523 | - | - |
| 64 | refit cubic | density 1, all quarters | 0.187872 | 0.969350 | 0.245691 | 0.199712 | 1.063x |
| 64 | refit cubic | density 2, Q0/Q1 | 0.190784 | 0.970997 | 0.239122 | - | - |
| 64 | refit cubic | density 3 Q0, density 2 Q1 | 0.193536 | 0.972180 | 0.234324 | 0.199760 | 1.032x |
| 128 | refit cubic | density 2, Q0/Q1 | 0.200736 | 0.968787 | 0.247959 | 0.210944 | 1.051x |
| 128 | refit cubic | density 3 Q0, density 2 Q1 | 0.207904 | 0.971873 | 0.235960 | 0.210944 | 1.015x |

The third native sample set is useful only in Q0. Applying it to Q1 instead
measured 0.194592 ms and 0.971618 cosine at D64. Applying it to both Q0 and Q1
measured 0.196928 ms and 0.972803 cosine. Restricting the Q0 samples to only
one query stage was also slower than using them in both stages.

The retained D64 policy adds 16 static `MUFU.EX2` instructions to the density-2
kernel, from 64 to 80, while leaving the 272 packed `FFMA2` instructions
unchanged. It compiles with 128 registers, one barrier, 416 bytes of static
shared memory, and no spills. D128 uses 400 bytes of static shared memory with
the same register, barrier, and spill counts.

## Seed stability

| D | Cosine mean | Cosine minimum | Relative-L2 mean |
|---:|---:|---:|---:|
| 64 | 0.971766 | 0.971078 | 0.236036 |
| 128 | 0.971846 | 0.971582 | 0.236129 |

The four seeds are 0, 1, 2, and 3. All retained measurements exceed 0.97
cosine. This has not yet been established across sequence lengths, head
counts, model distributions, or downstream training.

## Retained implementation

The default `HAO_FP4PV_MX_POLICY` is now `fast`. It sums all four packed words,
or all 32 represented P values, for each MXFP4 block denominator. This is not
the four-sample estimator and does not use the slower floating-point full-sum
path. `causal-accurate` remains an explicit opt-in when 0.97 cosine is more
important than the fast route's latency.

`HAO_FP4PV_MX_POLICY=causal-accurate` inherits the fast causal ownership and
denominator pipeline, then selects:

- refitted cubic coefficients `(0.07430709, 0.28611863, 0.64670005, 0.99010784)`;
- cubic packing in all four quarters and both query stages;
- two rotating native sample sets in Q0 and Q1;
- one additional rotating native sample set in Q0 only, in both stages.

The generic density-2 and density-3 quarter masks remain compile-time knobs,
so future shape-specific tuning does not require changing the transform code.
The benchmark's `--denominator-analysis` flag retains the oracle and cheap
predictor diagnostics.

Example D64 build:

```bash
make -B -f Makefile.hao_direct_fp4pv -j1 \
  GPU=B200 HAO_SEQ_LEN=4096 HAO_HEADS=32 HAO_KV_HEADS=8 \
  HAO_HEAD_DIM=64 HAO_NUM_SM=152 HAO_CAUSAL=1 \
  HAO_CAUSAL_INTERLEAVED_KV=1 \
  HAO_FP4PV_MX_POLICY=causal-accurate \
  OUT=/tmp/_C_causal_accurate_d64.so \
  MODULE=_C_causal_accurate_d64
```

## Conclusion

Denominator work alone cannot move the fast kernel to 0.97 cosine. The error
is predominantly numerator direction error from the approximate score-to-P
transform. A small, targeted increase in exact `EX2` sampling reaches a stable
0.97-cosine operating point for D64 and D128 without new reductions, barriers,
or storage, but it consumes nearly all of the speed advantage over BF16.

# Exact denominator SMEM handoff

Date: 2026-07-30

## Result

The NVFP4 `fast-corrected` path now mirrors the packed Q0/Q1 P payload in
shared memory. The correction warp can consequently snapshot only the
first-half scale and the Q2/Q3 payload from TMEM, signal `nv_denom_loaded`,
and perform the exact represented-denominator arithmetic after QK is free to
reuse the score bank.

This preserves the exact output of the previous represented-P correction. It
adds 8 KiB of dynamic shared memory per CTA, from 209920 to 218112 bytes, but
does not change the one-CTA-per-SM occupancy of this kernel.

```text
softmax warp                         correction warp
------------                         ---------------
pack Q0 -> TMEM
       \-> mirror Q0 in SMEM
pack Q1 -> TMEM
publish first P
       \-> mirror Q1 in SMEM
pack Q2/Q3 -> TMEM
publish P tail --------------------> snapshot:
                                      - Q0/Q1 scale from TMEM
                                      - Q2/Q3 payload + scale from TMEM
                                    release score TMEM to QK
                                    load Q0/Q1 payload from SMEM
                                    compute exact represented denominator
```

## GB200 measurements

All rows are B1, D128, causal, NVFP4 QK + NVFP4 PV, global anchor 32, and
the same `fast-corrected` arithmetic. Times use 200 ms warmup and 1.5-2.5 s
measurement windows.

| S | H | Previous (ms) | Handoff (ms) | Throughput gain |
|---:|---:|---:|---:|---:|
| 1024 | 64 | 0.030720 | 0.030720 | 0.00% |
| 2048 | 64 | 0.086048 | 0.085952 | 0.11% |
| 4096 | 24 | 0.112640 | 0.112640 | 0.00% |
| 4096 | 64 | 0.254208 | 0.251936 | 0.90% |
| 8192 | 24 | 0.409152 | 0.407104 | 0.50% |
| 8192 | 64 | 0.970784 | 0.960656 | 1.05% |

The S4096/H24 production rebuild reports:

- cosine vs BF16: 0.9632462263
- relative L2 vs BF16: 0.2930186689
- RMSE vs BF16: 0.0075190216
- registers: 128
- barriers: 1
- spills: 0

The final S8192/H64 production build reports cosine `0.9634298086`, relative
L2 `0.2924476862`, and RMSE `0.0053295661`, matching the experimental
handoff build and the previous schedule.

## Profile read

At S4096/H24, the handoff improved the full-replay NCU profile even though
wall time remained quantized at 0.112640 ms:

| Metric | Previous | Handoff |
|---|---:|---:|
| NCU replay duration | 190.720 us | about 188.560 us |
| Tensor utilization | 19.36% | about 19.55% |
| Issue active | 53.60% | about 53.85% |
| Eligible warps/cycle | 0.67 | about 0.69 |
| Long scoreboard | 58.03% | about 57.60% |
| Dynamic instructions | 56.292 M | 56.236 M |

The benefit grows under saturated or longer workloads because score-bank
release is repeated more often and QK has useful work ready to issue.

## Rejected variants

The following alternatives were compiled and measured before reducing the
implementation to one boolean:

| Variant | S4096/H24 | Conclusion |
|---|---:|---|
| Retain Q0/Q1 words in registers through `p_tail` | 0.116736 ms | Register lifetime and softmax-role imbalance dominate |
| Mirror both quarters only after first-P publication | about 0.11285 ms | Slightly slower than split stores |
| Mirror Q0 before Q1 and Q1 after first-P publication | 0.112640 ms | Kept |
| Also mirror the first-half scale | 0.112640 ms | No gain; adds 1 KiB |
| Store compact exact quarter sums | 0.114688 ms | Moving integer sum work onto softmax is slower |
| Hide compact-sum work under Q1/PV latency | 0.114688 ms | Same bottleneck |
| Issue Q1 before mirroring Q0 | 0.112640 ms | Equivalent to the kept schedule |

The compact representation reduced the handoff storage from 8 KiB to 1 KiB,
but it moved four `PRMT`/`DP4A` accumulation pairs per quarter onto the
latency-critical softmax warps. Keeping the packed words and leaving all sum
arithmetic on correction is faster.

## Build

`fast-corrected` enables `HAO_FP4PV_NV_DENOM_SMEM_HANDOFF=1`. Other policies
default to zero. The topology dictionary exposes the resolved value as
`nv_denom_smem_handoff`.

# Causal NV/MX versus BF16 FlashAttention-4

Date: 2026-08-15

This comparison measures the causal NVFP4-QK/MXFP4-PV forward kernel against
the causal BF16 FlashAttention-4 implementation from the HAO CuTe-DSL fork.
Both providers run in the same process on the same tensors. Input
quantization is prepared before timing.

## Protocol

- Batch 1, eight K/V heads, causal self-attention, BF16 output.
- TK release configuration at revision `9e67f6d`.
- HAO source revision `9b0abefdbbbe`.
- GB200: 152 SMs, three independent 300 ms `triton.testing.do_bench` windows
  after 100 ms warmup. Each table entry is the median of the three reported
  medians.
- Accuracy is measured against the same BF16 FA4 output.
- The first repeat for every shape changes the latter half of V and verifies
  that the protected causal output prefix and LSE remain bitwise identical.

## GB200

| S | Hq/Hkv | D | NV/MX (ms) | BF16 FA4 (ms) | Speedup | Cosine | Relative L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32/8 | 64 | 0.057600 | 0.081920 | **1.422x** | 0.951057 | 0.316051 |
| 4096 | 32/8 | 64 | 0.148480 | 0.199712 | **1.345x** | 0.950996 | 0.317333 |
| 8192 | 32/8 | 64 | 0.468992 | 0.599824 | **1.279x** | 0.950530 | 0.318134 |
| 4096 | 64/8 | 64 | 0.252512 | 0.324416 | **1.285x** | 0.949899 | 0.320522 |
| 2048 | 32/8 | 128 | 0.057728 | 0.087072 | **1.508x** | 0.951413 | 0.313747 |
| 4096 | 32/8 | 128 | 0.145408 | 0.210944 | **1.451x** | 0.950341 | 0.316311 |
| 8192 | 32/8 | 128 | 0.454656 | 0.629632 | **1.385x** | 0.950187 | 0.316471 |
| 16384 | 32/8 | 128 | 1.610112 | 2.141184 | **1.330x** | 0.949272 | 0.319597 |
| 4096 | 64/8 | 128 | 0.247808 | 0.342768 | **1.383x** | 0.949989 | 0.317118 |
| 8192 | 64/8 | 128 | 0.846848 | 1.120256 | **1.323x** | 0.949431 | 0.319116 |

All outputs are finite. All ten causal leakage checks pass bitwise for both
the protected output prefix and LSE.

An opt-in D128 causal-boundary approximation now pair-rounds Q2 and Q3 with
`HAO_CAUSAL_PAIR_QUARTER_MASK=12`. It removes 64 SASS comparisons and improves
latency by 0.1--1.0% across S2048--S8192 and Hq32--Hq64. It remains strictly
causal and finite, but typically lowers cosine by 0.0003--0.0004 and raises
relative L2 by 0.0011--0.0014, so the exact path remains the default. The full
screen and rejected exact-mask alternatives are in `causal_p_mask_fusion.md`.

## D64 optimization and ceilings

The original D64 policy was missing two causal components already useful in
the D128 route. The retained D64 change derives the diagonal mask from CTA
coordinates and overlaps packed-denominator decoding with P construction.
D64 rebases quarter scales, so it cannot use D128's deferred-denominator
carrier without changing that representation. The larger D128 quarter-max
interleave was tested independently and regressed D64.

Increasing the K/V ring from 12 to 13 stages then produced a smaller but
repeatable 0.2--0.6% gain across S2048, S4096, S8192, and saturated H64. The
combined change reduces latency by 3.3--6.4% versus the original D64 table.
It retains 128 registers, one barrier, zero stack, and zero spills. The S4096
H32 output is bit-identical to the original policy.

A later D64 refinement uses the otherwise-free final 128 TMEM columns for two
detached packed-P/PV-scale slabs. Together with stage-local issue retiming, it
reduces latency by another 3.5--6.3% across the tested sequence/head grid while
remaining bitwise identical to the preceding D64 kernel. The layout, rejected
scale-only controls, fresh BF16 comparison, and NCU read are in
`d64_detached_p_tmem.md`.

An exhaustive follow-up compacted the tail to fit a third 16-column packed-P
payload and tested stage-0/stage-1 ownership, full and split release tokens,
existing-token reuse, retired-score scratch, warpgroup handoff, and temporal
P-scale overwrites. The best correct third-payload schedule was 5.25% slower;
NCU showed lower issue eligibility and tensor-core activity despite unchanged
barrier, register, spill, and static-smem resources. Mode 0 therefore remains
the release default. The full 19-mode matrix is in `d64_detached_p_tmem.md`.

Timing-only ceilings at S4096 H32 separate practical scheduling headroom from
removed-work diagnostics:

| D | Production (ms) | Raw score-pack (ms) | Fixed-P (ms) | Gap to score-pack |
|---:|---:|---:|---:|---:|
| 64 | 0.155984 | 0.126528 | 0.106496 | 18.9% |
| 128 | 0.142695 | 0.117443 | 0.094192 | 17.7% |

The raw score-pack and fixed-P kernels deliberately remove required softmax/P
work and are not usable attention implementations. They show that D64 and
D128 now have similar residual structure: the remaining gap is primarily the
score-load, max/scale, exp2, and FP4-pack dependency chain. K/V depth, reduced
physical grids, alternate task ordering, and six additional ALU/SFU exp2
cadences did not expose another material scheduling win.

## B300 D128

These existing matched measurements use the same provider definitions and
1000 ms timing windows on 148-SM B300. D64 has not yet been tuned on B300.

| S | Hq/Hkv | NV/MX (ms) | BF16 FA4 (ms) | Speedup | Cosine | Relative L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32/8 | 0.037824 | 0.050112 | **1.325x** | 0.951009 | 0.313470 |
| 4096 | 32/8 | 0.091136 | 0.123808 | **1.359x** | 0.950178 | 0.317117 |
| 8192 | 32/8 | 0.279328 | 0.420768 | **1.506x** | 0.949441 | 0.319596 |
| 16384 | 32/8 | 0.980480 | 1.573920 | **1.605x** | 0.948713 | 0.321439 |
| 4096 | 64/8 | 0.154592 | 0.222240 | **1.438x** | 0.950101 | 0.317557 |
| 8192 | 64/8 | 0.517088 | 0.818208 | **1.582x** | 0.950271 | 0.316604 |

## Read

The causal NV/MX kernel beats BF16 FA4 on every measured shape. D128 is the
strong route on GB200, with a 1.32--1.51x speedup. Optimized D64 reaches
1.28--1.42x. Halving D halves much of the QK/PV tensor-core work but does not
halve score-to-P construction, which is why D64 remains slower in absolute
time than D128 at the same S and H. Further D64 progress now requires a
shorter score-to-P dependency graph rather than another launch-policy sweep.

The release-`9e67f6d` JSON, build logs, and resource reports are stored under
`gb200_release_9e67f6d_full`; the reproducible runner is `run_gb200.sh`.
B300 source records remain under
`results/fp4_fa4_b300_tuning_20260802`.

The causal-mask fusion study, opt-in Q2/Q3 pair policy, and timing ceiling are
recorded in `causal_p_mask_fusion.md`.

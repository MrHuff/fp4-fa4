# Saturated causal FA4 model-step bracket (2026-08-27)

This directory records the final local GB200 comparison of BF16 FA4,
NVFP4-QK/FP8-PV, and NVFP4-QK/MXFP4-PV after promoting row-by-K16 Q/K
publication for D128. The result is a short model-step and numerical-stability
gate, not a long-run pretraining curve.

## Result

Changing D128 Q/K publication from fixed-head to row-by-K16 scales removed the
immediate 8B loss and gradient failure in this gate. Replacing only that forward
scale geometry restores BF16-like short-run loss for both PV routes without a
demonstrated material end-to-end penalty. MXFP4-PV and FP8-PV retain the same
E4M3 Q/K/V attention backward and are effectively tied in backward time.

Base timings and throughputs are the arithmetic midpoint of the A/B per-run
p50s or throughputs from the mirrored order BF16, FP8, MX, MX, FP8, BF16;
speedups are ratios of those midpoints.

| 8B/D128 route | Step p50 (ms) | Event speedup | Sustained tok/s | TPS speedup | Decoder (ms) | Backward (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BF16 FA4 | 276.293671 | 1.000000x | 14,738.203 | 1.000000x | 67.420393 | 138.475204 |
| NVFP4-QK + FP8-PV | 258.536064 | 1.068685x | 15,723.099 | 1.066826x | 64.181498 | 123.917330 |
| NVFP4-QK + MXFP4-PV | 257.687294 | 1.072205x | 15,794.051 | 1.071640x | 63.323808 | 124.102711 |

MX versus FP8 is `1.003294x` by event latency and `1.004513x` by sustained
throughput. This two-run bracket does not statistically support an MX speed
claim from that sub-percent gap. The backward-midpoint difference is 0.185 ms,
or 0.15%, in the opposite direction and is consistent with timing noise around
the shared implementation.

The previously completed saturated 1.2B/D64 B16 bracket remains stronger in
absolute speedup and is included for scale context:

| 1.2B/D64 route | Step p50 (ms) | Event speedup | Sustained tok/s | TPS speedup | Final held-out loss |
|---|---:|---:|---:|---:|---:|
| BF16 FA4 | 678.304230 | 1.000000x | 96,314.231 | 1.000000x | 8.049704 |
| NVFP4-QK + FP8-PV | 558.953308 | 1.213526x | 117,056.913 | 1.215365x | 8.126912 |
| NVFP4-QK + MXFP4-PV | 558.834351 | 1.213784x | 117,056.950 | 1.215365x | 7.962223 |

## Short-run numerical gate

Each 8B arm starts from the same checkpoint and packed Dolma stream, runs three
warmup optimizer updates plus 20 measured updates, and evaluates the same held-
out batch before and after. All 138 update records are finite.

| Route | Initial loss | Final A / B | Descriptive midpoint | Peak pre-clip grad A / B |
|---|---:|---:|---:|---:|
| BF16 | 12.571371 | 10.091789 / 9.663280 | 9.877535 | 157 / 157 |
| FP8-PV | 12.585827 | 9.569537 / 9.155010 | 9.362274 | 161 / 161 |
| MXFP4-PV | 12.586738 | 9.035723 / 9.526566 | 9.281144 | 184 / 183 |

This removes the failure seen with fixed-head D128 Q/K scales: the former FP8
and MX loss midpoints were 16.665405 and 12.905418, and their maximum pre-clip
gradient norms were 19,968 and 19,840. The contemporaneous per-block maxima are
161 and 184. Old/new timing is not a strictly paired comparison; FP8 is
unchanged within 0.05 ms, MX costs about 1.17 ms, and the BF16 baseline itself
moved by 2.24 ms between brackets.

The causal evidence is deliberately narrow:

- Fixed-head to per-block changes one backward-contract leaf,
  `projection.per_block_qk_scales`; Q/K/V backward sources remain the
  projection-accumulator E4M3 publications and projection dgrad remains NVFP4.
- The projection boundary validator reduces Q/K publication relative L2 from
  0.1356/0.1208 to 0.0951/0.0951 while keeping all three E4M3 backward tensors
  bitwise equal.
- Replacing NVFP4 projection dgrad with BF16 did not fix the old FP8 failure:
  held-out loss still rose from 12.5831 to 14.9625 and the gradient norm reached
  42,752. NVFP4 dgrad alone was therefore not the cause.

This establishes removal of the immediate B1 instability. It does not prove
long-horizon 8B convergence: initial hidden-state cosine versus BF16 remains
only 0.345 for FP8 and 0.303 for MX, and BF16 A/B itself has material short-run
trajectory variation.

## Why isolated MX speed does not become a large model-step gain

At causal B1/S4096/H32/KV8/D128, the safe MX attention kernel is 0.121696 ms
versus 0.154774 ms for FP8, a `1.2718x` isolated speedup. One attention launch
is a small fraction of a roughly 258 ms model step. The prior Nsight pair
measured 0.737 ms of aggregate MX attention saving over 32 layers, while the
extra MX plus E4M3-V QKV publication cost 0.976 ms. The exact values come from
the earlier fixed-head profile, but they explain the mechanism; the final
per-block bracket has no new Nsight trace.

For 1.2B/B16, the profiler captured 1,698 kernels per route with 98.41% kernel-
union busy time, about 99.31% mean GR active, and about 93.4% mean SM active.
For 8B/B1 the FP8 trace was execution-busy (94.29% kernel-union busy), but the
model used only about 93.35 GiB reserved HBM. It is not an HBM-maximized batch.

## Safe MXFP4 forward policy

The retained D128 MX kernel uses selector 4, anchor32, margin-log2 64, stored
shift 32, folded-K64 disabled, and preload mask 3. On the audited trained
payload it has zero nonfinite output/LSE values, output relative L2 0.01827
against the exact-rowmax MX oracle, and LSE relative L2 0.07394 against exact
logsumexp. The rebuilt production binary is bitwise output/LSE equal to that
candidate.

The faster historical shift16 candidate is rejected: it produced 370,900
nonfinite outputs out of 16,777,216. Its separate timing record is 2.9% lower,
but it uses a different trained payload, so that number is not a paired policy-
cost measurement. The safe policy remains 1.27x faster than isolated FP8 on
its audited payload.

## Protocol and provenance

- Measured 8B source commit: `e4f83793d2b9d33c67d1415e49098c47b6c1a45f`.
- Guard/provenance commit: `0f622498388097e8daeb654527458df6eaf121e8`.
  The later commit only makes the trainer and preflights fail closed on the
  measured per-block policy; it does not change the measured kernel recipe.
- Model: Llama-3.1 8B, 32 layers, B1/S4096/H32/KV8/D128.
- Hardware: one NVIDIA GB200 (SM100, 152 SMs), UUID
  `61ebc9a2-4efb-2335-6ded-8591f9acef8c`.
- Optimizer: fused AdamW, LR `0.00048828125`, betas 0.9/0.95, weight decay
  0.1, pre-update gradient clipping at 1.0.
- Loss: torch-compiled MLCE 25.9.3. Full-logit buffer elision was not proven.
- Corpus SHA256: `860b33924dffd53f4c20b80abbcee96e1bf09c3c313290c15ea3a6ee418269ce`.
- Tokenizer SHA256: `76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa`.
- 8B checkpoint SHA256:
  `a635f06aab6ecb8b28cab759e05961a702cd715c02be5fbcb21703f9494fa907`.
- Projection binary SHA256:
  `70ffe688ae9f91cf717402722b33df3273c3f2b37318dc3b145ca49728d1a84b`.
- FP8/MX forward binary SHA256:
  `275047ff3ac86e1a2e7587b3159f0ee17c55da61ea4a52a6a4a94eee679577a6` /
  `9d34798d2009a939472353fff2714b7c413de4c1394a4c942771848a7e0696d4`.
- All four low-precision backward contracts have canonical SHA256
  `02ddde7f4d9505ab934050ad4113affa402c7bfa39bc5187a06206a9c6924011`.

Each low-precision arm reports 32/32 authenticated QKV dual-weight and output-
weight preparations, generation and same-stream guards, zero in-flight layers,
stable private owner pointers, authenticated runtime topology, and no timed
allocation fallback.

Selected compact values and SHA256s for the listed raw evidence are in
`summary.json`.
The large checkpoints, sample tensors, profiler databases, and raw route JSONs
are intentionally not duplicated in Git.

## Interpretation limits

- These are short, single-GPU, random-initialization gates—not pretraining
  curves or 50B-token results.
- 8B is B1 and execution-busy, but not HBM-maximized. Batch scaling remains an
  explicit follow-up.
- A/B loss midpoints are descriptive. Every low-precision B arm compares its
  samples to BF16-A, not to an independently paired BF16-B reference.
- BF16 uses packed QKV and 227 physical optimizer tensors; low precision keeps
  split canonical masters and 291 tensors. The logical model and AdamW recipe
  match, but this is not an attention-only graph comparison.
- Useful MFU is an analytical BF16-equivalent estimate, not a hardware
  utilization metric. Nsight counters are reported separately.

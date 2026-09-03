# Long-path validation and P-chain ceilings

Date: 2026-07-30

## Scope

This pass tested five possible ways to improve the long-sequence
`fast-adaptive` NVFP4/NVFP4 forward path:

1. real model activations at S1024, S2048, and S4096;
2. rotating the sampled denominator word across tiles, stages, and heads;
3. using two exact packed words in selected P quarters;
4. profiling the retained path against score-pack and fixed-P ceilings;
5. moving early P ownership or publication to the correction warpgroup.

All kernel timings are on GB200, B1, H24, D128, noncausal. The model test is
`nateraw/vit-base-patch16-224-cifar10`: 12 real H64 heads are padded to the
H24/D128 specialization. It is an accuracy test; Q/K/V quantization is not
included in the kernel timings.

## Harness correction

The long-sequence extensions are compiled with a 32-key global anchor. The
first exploratory model runs accidentally omitted the corresponding K/V
permutation. Those results are invalid and are not included below.

`eval_regular_attention.py` now:

- obtains S, H, and D from extension topology instead of assuming S256/H16;
- supports high-resolution ViT inputs with positional interpolation;
- pads tokens to the compiled specialization and masks padded keys;
- rejects a run when its global-anchor preprocessing disagrees with the
  extension topology.

The three image sizes produce:

| Kernel S | Image | Real tokens |
|---:|---:|---:|
| 1024 | 496x496 | 962 |
| 2048 | 720x720 | 2026 |
| 4096 | 1008x1008 | 3970 |

## Real-activation result

Four CIFAR-10 images were propagated through all 12 ViT attention layers.
`Exact` is the existing `fast-corrected` represented-P denominator.

| S | P denominator | Kernel ms | Top-1 agreement | Logit cosine | Logit rel. L2 |
|---:|---|---:|---:|---:|---:|
| 1024 | one-word sampled | 0.018432 | 75% | 0.863323 | 0.515665 |
| 1024 | exact | 0.018784 | 75% | 0.838103 | 0.567965 |
| 2048 | one-word sampled | 0.043296 | 25% | 0.377089 | 0.936442 |
| 2048 | rotated one-word | 0.044800 | 100% | 0.915137 | 0.408917 |
| 2048 | two-word sampled | 0.045088 | 100% | 0.946787 | 0.327413 |
| 2048 | exact | 0.045344 | 100% | 0.972667 | 0.232404 |
| 4096 | one-word sampled | 0.106528 | 50% | 0.813281 | 0.581872 |
| 4096 | exact | 0.112640 | 100% | 0.956989 | 0.290240 |

All outputs were finite. Four samples are enough to reject unsafe estimators,
but not enough to claim task-level quality. The first attention layer is
already the worst layer:

| S/path | Layer-0 cosine | Layer-0 rel. L2 |
|---|---:|---:|
| 2048 sampled | 0.717861 | 0.910445 |
| 2048 exact | 0.921219 | 0.389708 |
| 4096 sampled | 0.782133 | 0.630605 |
| 4096 exact | 0.919703 | 0.392174 |

The sampled denominator remains a useful random-QKV speed policy, but it is
not a model-safe long-sequence default. `fast-corrected` is the supported
choice for real long-sequence activations.

## Denominator experiments

### Rotating one word

The best rule selected

```text
word = (quarter + key_tile + stage + head) mod 4
```

It improved the S2048 model result substantially, but remained less accurate
than exact normalization and added about 4% kernel time. At S4096 it measured
0.110592 ms, 3.46% slower than the fixed one-word sampler.

### More sampled words

Full two-word sampling is nearly as expensive as exact correction:

- S2048: 0.045088 ms versus 0.045344 ms exact;
- S4096: 0.116736 ms versus 0.112640 ms exact.

Sampling two words only in Q0+Q2 or Q0+Q2+Q3 did not preserve the four-image
S2048 model result. The extra arithmetic is also poorly hidden in the P
producer. No two-word policy was promoted.

## Speed ceilings

Four-seed S4096/H24 means:

| Path | Time (ms) | Delta from retained |
|---|---:|---:|
| Retained one-word adaptive | 0.106896 | - |
| Score-pack ceiling | 0.090200 | -0.016696 |
| Fixed-P ceiling | 0.088072 | -0.018824 |

The score-pack ceiling still loads score registers, converts raw score values
to packed FP4, publishes P, and executes PV. It removes the real
max/scale/exp2 transformation. Fixed-P removes score loading and packing too.

Therefore:

- about 16.7 us remains in score-to-P transformation;
- only about 2.1 us remains in score loading, FP4 packing, and publication;
- barriers are not the current wall-time limiter.

For the exact 0.112640 ms path to reach 0.100 ms, approximately 12.6 us must
be removed or hidden. That requires recovering more than half of its gap to
the 0.0902 ms score-pack ceiling.

## NCU attribution

These are instrumented replay metrics, so duration is not the benchmark wall
time.

| Metric | Adaptive | Score-pack | Fixed-P |
|---|---:|---:|---:|
| Replay duration (us) | 181.568 | 149.696 | 142.016 |
| Tensor pipe active | 20.44% | 24.77% | 26.01% |
| Issue active | 50.66% | 21.21% | 15.04% |
| Active warps/cycle | 3.70 | 3.69 | 3.69 |
| Eligible warps/cycle | 0.63 | 0.24 | 0.17 |
| Barrier stall | 0.42% | 0.47% | 0.49% |
| Long-scoreboard stall | 61.53% | 77.39% | 83.06% |

Removing scalar P work raises tensor utilization. The higher scoreboard share
in the ceilings means tensor dependencies dominate once scalar instructions
are removed; it does not mean the ceilings are slower.

## Ownership experiments

The legal Q0/Q1 correction-warpgroup path requires disabling the retained
register scale handoff:

| Matched path | Time (ms) |
|---|---:|
| Register handoff disabled | 0.110624 |
| Q0/Q1 correction-WG ownership | 0.112192 |
| Retained register handoff | 0.106896 |

The ownership transfer adds 1.568 us after already losing 3.728 us by
disabling register handoff.

The anchor-64 publication variants also lost globally:

| Anchor-64 path | Time (ms) |
|---|---:|
| Control | 0.114720 |
| Q1 prepack | 0.116992 |
| Correction-WG anchor | 0.114240 |

Correction-WG anchor work saves 0.480 us inside that schedule, but the
schedule remains 6.9% slower than the retained anchor-32 path.

## Conclusion

None of the scheduling experiments should land in the production kernel.
The validated choices are:

- `fast-adaptive`: fastest isolated long-shape policy, with demonstrated
  model-accuracy risk;
- `fast-corrected`: 2-6% slower across S1024-S4096, but the only tested path
  that preserves all four predictions at S2048 and S4096.

The next credible optimization target is the exact path's score-to-P
max/scale/exp2 transform. More denominator sampling and cross-warpgroup TMEM
ownership are exhausted at this layout.

## Artifacts

- `vit_long_{adaptive,exact}_s{1024,2048,4096}_4sample_anchored.json`
- `vit_long_rot4_s2048_4sample_anchored.json`
- `vit_long_{w2,q0q2,q0tail}_s2048_4sample_anchored.json`
- `ncu_{adaptive,scorepack,fixedp}_s4096.csv`

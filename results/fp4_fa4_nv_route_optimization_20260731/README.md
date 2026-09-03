# NV/MX and NV/NV Route Optimization, 2026-07-31

## Scope

All kernel timings use one GB200 at B1, S4096, H24, D128. Reported long
timings use 100 ms warmup and a 1500 ms measurement window. Downstream
results use 200 CIFAR-10 images through
`nateraw/vit-base-patch16-224-cifar10` at image size 1008. Dynamic Q/K/V
quantization is outside the kernel timing.

## Saturated NV/MX schedule follow-up

The saturated NVFP4-QK/MXFP4-PV route was re-profiled at H64 with its
numerics pinned. Four exact schedule changes were retained:

1. Omit packed-P payload masking when the E8M0 scale is zero. Both the
   represented denominator and TCGEN PV multiply by the encoded zero scale,
   so the payload bits cannot contribute.
2. Use the wider depth-four ternary-max tree for exact quarter-scale maxima.
3. Preload only the Q scale page before PV and copy the K page at QK issue.
4. Use a 16-operation depth-four tree for the interleaved stage-0 Q1 maximum,
   instead of the previous 20-operation tree.

Long paired measurements use a 300 ms warmup and 5000 ms timing window:

| Shape | Pinned original (ms) | Retained (ms) | Latency reduction |
|---|---:|---:|---:|
| B1 S4096 H24 D128 | 0.094224 mean | **0.092320 mean** | 2.02% |
| B1 S4096 H64 D128 | 0.205328 mean | **0.202624 mean** | 1.32% |
| B1 S8192 H64 D128 | 0.774144 | **0.759808** | 1.85% |

The H24 and S4096/H64 means each cover two independent input seeds. Every
paired comparison was bit-identical. Across the two H64 seeds, cosine versus
the Torch BF16 reference was 0.943029-0.943339, relative L2 was
0.337683-0.338420, and RMSE was 0.008703-0.008714. A fresh four-image ViT run
also matched the pinned kernel exactly and retained 100% top-1 agreement,
0.996485 logit cosine, and 0.083908 relative L2 versus BF16.

The retained kernel still uses 128 registers, one barrier, 400 bytes of
static shared memory, and no spills. The pre-Q1-compaction NCU replay reduced
GPC cycles from 389798 to 385591 and dynamically executed instructions by
about 1.08%. Tensor-pipe activity moved only from 40.68% to 40.82%. The
dominant source stall remains the `score_full` wait in
`hao_direct_fp4pv_softmax_reader.inc`; long-scoreboard pressure, not barriers,
is still the structural limit.

## Downstream gate for the retained fast route

The retained route was evaluated in two pretrained models. All runs use the
same folded-Q/K preprocessing as the kernel benchmark. ViT at 1008x1008 has
3970 real tokens and uses the S4096/H24 specialization with an explicit
padding mask. Native ViT and BERT use the matching S256/H16 specialization.
No run produced non-finite attention or model outputs.

| Evaluation | BF16 task metric | Fast NV/MX task metric | Agreement | Logit cosine | Logit relative L2 |
|---|---:|---:|---:|---:|---:|
| ViT, 200 images, 3970 tokens | 88.5% accuracy | **88.5% accuracy** | 95.5% | 0.988674 | 15.086% |
| ViT, 1000 images, 197 tokens | 98.9% accuracy | 98.8% accuracy | 99.3% | 0.998683 | 5.139% |
| BERT, 800 S256 blocks | 61.740% MLM accuracy | 61.311% MLM accuracy | 90.688% | 0.996921 | 7.865% |

BERT MLM loss changes from 2.206614 to 2.238755, a 1.46% increase. Mean
attention-output relative L2 is 8.86% for long ViT, 15.06% for native ViT,
and 22.56% for BERT. The model-level errors are much smaller, showing that
the residual path absorbs much of the structured attention error.

For context, the previous downstream-safe S4096/H24 NV/MX path measured
0.112784 ms on average, 98.5% top-1 agreement, and 4.799% logit relative L2.
The retained route is 18.1% faster at 0.092320 ms. Its 0.5 percentage-point
lower accuracy on this 200-image subset is too small to establish a model
quality regression, but the larger logit drift is a real robustness signal.

The resulting priority is precision recovery under a hard latency budget,
not a return to the slower stabilized path. Exact scheduling improvements can
continue, but further numerical approximation should be gated on retaining
the current task metrics. A useful next target is to halve long-ViT logit
relative L2 while staying at or below 0.095 ms.

The raw evaluation records are:

- `downstream_vit200_current_nvmx_s4096h24.json`
- `downstream_vit1000_current_nvmx_s256h16.json`
- `downstream_bert200_current_nvmx_s256h16.json`
- `downstream_bert800_current_nvmx_s256h16.json`

Two nearby variants were rejected. Preloading the K page instead of Q was a
timing tie. Applying the 16-operation tree to every generic quarter removed
32 more static max instructions, but was neutral at H64 and regressed H24 by
0.21-0.24% because its extra live partials disrupted scheduling.

The follow-up adds these settings to the pinned custom route:

```text
HAO_FP4PV_MX_SKIP_ZERO_SCALE_MASK=1
HAO_FP4PV_MX_MAX3_WIDE_REDUCE=1
HAO_FP4PV_NV_QK_PRELOAD_PAGE_MASK=1
```

The compact interleaved-Q1 tree is selected by the existing
`MX_MAX3_WIDE_REDUCE` setting. The benchmark must continue to use
`--nv-qk-fold-k64-scales both --nv-qk-fold-scale-select mse`.

## Retained results

| Route | Change | Time (ms) | Cosine vs BF16 | Relative L2 | Top-1 agreement | Model accuracy |
|---|---|---:|---:|---:|---:|---:|
| NV/MX previous | retained `(A=1.65, B=0.8)` | 0.116224 | 0.998861 | 0.047994 | 98.5% | 89.0% |
| NV/MX max3 | exact three-input max reduction | 0.112640-0.112928 | 0.998861 | 0.047994 | 98.5% | 89.0% |
| NV/NV fixed G0 | truthful SMEM release | 0.114688 | 0.844774 | 0.536052 | 72.5% | 70.0% |
| NV/NV fixed G2 | G0 fix plus `P_GLOBAL_LOG2=2` | 0.114688 | 0.871437 | 0.490566 | 79.0% | 74.0% |

The NV/MX max3 result averages 0.112784 ms over seeds 0 and 1, a 2.96%
reduction from the previous 0.116224 ms result. It is bit-identical to the
previous NV/MX kernel. The generated kernel uses 128 registers, one barrier,
432 bytes of static shared memory, 208896 bytes of dynamic shared memory, and
has no spills.

The NV/NV G2 rebase is also effectively free. Relative L2 improves by 8.48%
against deterministic G0, but NV/NV remains far less accurate than NV/MX.

## NV/MX speed change

`upstream_reduce_score_quarter` now optionally uses SM100's three-input FP32
maximum instruction for the exact N32 P-scale reduction. Static SASS changes
from 88 three-input plus 80 two-input max instructions to 144 three-input max
instructions. No arithmetic or payload bits change.

An NCU replay measured about 19.5% tensor-pipe activity, 55.1% issue-active,
0.70 eligible warps per cycle, and about 58% long-scoreboard stalls. Barrier
stalls were only about 0.4%. This route is now dependency and data-arrival
limited rather than barrier limited.

The following same-output schedule changes were slower and were rejected:

| Candidate | Time (ms) |
|---|---:|
| retained max3, 13 KV stages, registers 184/96 | 0.112640 |
| 12 KV stages | 0.114208 |
| registers 176/112 | 0.116288 |
| registers 192/80 | 0.115712 |

Refitting the P affine map to `(A=1.8, B=1.2)` preserved 0.112928 ms but
worsened 200-image relative L2 from 0.047994 to 0.049811, so `(1.65, 0.8)`
remains retained.

## NV/MX throughput ceiling

The mode-23 P transform already evaluates four native EX2 pairs in every
quarter, but the sampled denominator previously kept only the first and third
pairs. `MX_PWL_FOUR_SAMPLE_DENOM=1` reuses all four values. It adds no EX2,
TMEM load, or synchronization instruction; it only interleaves the additional
FP32 adds with the existing transform and pack work.

| Shape | Two-sample denominator (ms) | Four-sample denominator (ms) | Change |
|---|---:|---:|---:|
| B1 S4096 H24 D128 | 0.104448 | **0.102400** | -1.96% |
| B1 S4096 H64 D128 | 0.233024 | **0.225536** | -3.21% |

Four independent H24 seeds all measured 0.102400 ms. Their random-input
cosine ranged from 0.950442 to 0.951036 and relative L2 from 0.309680 to
0.311737. A fresh build through the named policy reproduced 0.102688 ms,
0.950378 cosine, and 0.312001 relative L2.

The kernel remains at 128 registers, one barrier, 400 bytes of static shared
memory, and zero spills. Static SASS grows from 3,088 to 3,160 instructions;
the only material class change is FP32 adds, from 36 to 76. The extra
independent adds fill score/TMEM dependency bubbles, so the longer static
program has lower wall time.

This is a throughput ceiling, not a numerically safe attention kernel. On a
20-image ViT smoke test at S4096, the unanchored route had 0% top-1 agreement,
0.04373 cosine, and 1.00153 relative L2. Adding the global anchor slowed the
kernel to 0.107520 ms but recovered only 15% agreement, 0.18348 cosine, and
0.99075 relative L2. The exact represented-denominator route remains the
model-safe option at about 0.1129 ms; its 200-image result is 98.5% agreement,
0.998861 cosine, and 0.047994 relative L2.

Offline replay confirmed that denominator estimation, rather than the fast P
encoder, is the safety boundary. On ViT layer 0, exact represented
normalization reached 0.999342 cosine and 0.036273 relative L2. The best tested
cheap max-control-variate estimate reached only 0.970688 cosine and 0.284701
relative L2, so it was rejected before adding kernel complexity.

## NV/NV synchronization fix

The correction warpgroup copied the Q2/Q3 payload and P scale from TMEM, then
published `nv_denom_loaded` before reading the mirrored Q0/Q1 payload from
shared memory. The issuer interpreted that signal as permission to issue the
next QK tile. The following softmax iteration could therefore overwrite the
shared Q0/Q1 mirror while correction was still reading it.

The symptom was large run-to-run variation for the same binary, seed, and
inputs: 60-image relative L2 ranged from 0.539842 to 0.568143 across four
GB200s. NV/MX was exactly deterministic under the same test.

The retained fix reads the shared Q0/Q1 payload before publishing
`nv_denom_loaded`. Four independent 60-image runs then produced exactly the
same logits and 0.521236 relative L2. This costs about 0.00198 ms versus the
racy schedule because QK reuse waits for the shared loads.

A split-signal design restored the early TMEM release and added a separate
shared-memory ownership barrier. It was correct but slowed to 0.121152 ms, so
it was rejected.

## NV/NV scale sweep

Once the race was fixed, the compile-time P-scale sweep became monotonic and
reproducible. `G` below is the additive base-2 exponent applied to represented
P; PV and its represented denominator consume the same scale.

| G | 60-image relative L2 | 60-image cosine |
|---:|---:|---:|
| 0.0 | 0.521236 | 0.854221 |
| 1.0 | 0.467892 | 0.883814 |
| 1.5 | 0.464440 | 0.885605 |
| 2.0 | 0.463107 | 0.886307 |
| 2.5 | 0.475526 | 0.879702 |
| 3.0 | 0.480947 | 0.876750 |
| 4.0 | 0.493149 | 0.869946 |
| 6.0 | 0.522798 | 0.852468 |

The 200-image gate confirmed G2 as the best tested global setting:

| G | Relative L2 | Cosine | Agreement | Accuracy |
|---:|---:|---:|---:|---:|
| 0.0 | 0.536052 | 0.844774 | 72.5% | 70.0% |
| 1.0 | 0.503574 | 0.864159 | 77.5% | 72.5% |
| 1.5 | 0.497493 | 0.867559 | 79.0% | 73.5% |
| 2.0 | 0.490566 | 0.871437 | 79.0% | 74.0% |
| 2.5 | 0.494298 | 0.869307 | 79.0% | 73.5% |

Layer-specific schedules did not improve global G2 meaningfully. The best
60-image variant changed layers 9-11 to G1.5 and moved relative L2 only from
0.463107 to 0.462920, which does not justify multi-kernel dispatch.

## Remaining limit

Offline decomposition attributes NV/NV's large residual to P-scale range:
with G0, about 82% of sampled quarter scales fall below E4M3's representable
range. A fixed rebase trades low-end underflow against high-end clipping and
G2 is the best tested compromise. MXFP4 P uses an E8M0 scale, so NV/MX avoids
that range conflict and its remaining error is mainly QK and V quantization.

Further NV/NV accuracy requires a row- or tile-adaptive P-scale anchor, a
wider P-scale representation, or stabilization. A different fixed scalar or
affine refit cannot close the gap to NV/MX.

## Retained build additions

```text
HAO_FP4PV_MX_MAX3_REDUCE=1
HAO_FP4PV_NV_P_GLOBAL_LOG2_OVERRIDE=2.0f
```

The NV/MX setting adds `MX_MAX3_REDUCE=1` to the retained build documented in
`results/fp4_fa4_mx_accuracy_recovery_20260730/README.md`. NV/NV G2 uses the
existing `fast-corrected` policy plus the global-log2 override.

Named NV/MX policies are available through the isolated build:

```text
make -f Makefile.hao_direct_fp4pv HAO_FP4PV_MX_POLICY=ceiling \
  HAO_SEQ_LEN=4096 HAO_HEADS=24 OUT=/tmp/nvmx_ceiling.so \
  MODULE=_C_nvmx_ceiling

make -f Makefile.hao_direct_fp4pv HAO_FP4PV_MX_POLICY=accurate \
  HAO_SEQ_LEN=4096 HAO_HEADS=24 OUT=/tmp/nvmx_accurate.so \
  MODULE=_C_nvmx_accurate
```

`ceiling` selects the 192/80 register split, four-sample denominator reuse,
and the refitted affine map. `accurate` selects the correction warpgroup,
global anchor, stored scale shift, and 184/96 split used by the downstream-safe
route. It requires the matching global-anchor32 KV preprocessing. A direct
same-input comparison against the retained max3 binary produced bit-identical
outputs (`max_abs=0`, `relative_l2=0`).

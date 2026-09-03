# Regular-attention downstream evaluation

> Follow-up (2026-07-29): the shiftless scale encoder and denominator are now
> finite-range guarded, and a stabilized `universal` policy has been added.
> A 100-image ViT and 20-block BERT replay reports zero non-finite rows at
> 0.176400 ms for B1/S4096/H24. Results and reproduction commands are in
> [`../fp4_fa4_universal_policy_20260729/README.md`](../fp4_fa4_universal_policy_20260729/README.md).

> Correction (2026-07-29): the offline `attention.p_scale_sweep` fields in
> the JSON records selected E2M1 payload codes with the unrounded block
> scale, then decoded them with the rounded E4M3 scale. Those offline sweep
> fields are invalid. The compiled-kernel model evaluations in this document
> are unaffected. A corrected sweep and new compiled `G=24` evaluations are
> reported in
> [`../fp4_fa4_p_coefficient_recheck_20260729/README.md`](../fp4_fa4_p_coefficient_recheck_20260729/README.md).

This directory evaluates stabilized TK FP4 attention in two pretrained
models with dense, noncausal self-attention. It covers NVFP4 and MXFP4 for
both QK and PV:

- ViT-B/16 fine-tuned on CIFAR-10
- BERT-base masked-language modeling on WikiText-2

The evaluation ran on one NVIDIA GB200. Dynamic Q/K/V quantization is
outside the kernel benchmark and inside the model accuracy path. These are
accuracy evaluations, not end-to-end model latency claims.

## Main result

The random-tensor throughput policy cannot be used as-is downstream. It
sets the softmax row maximum to zero and materializes `exp(score)`. The
first real ViT layer has scaled scores spanning approximately `-69.65` to
`+103.34`, so the shiftless path overflows and produces non-finite output.

Restoring online-softmax stabilization,

```text
P_ij = exp(S_ij - max_j S_ij),
```

makes `0 < P_ij <= 1` and gives a viable full-FP4 model path.

For an E2M1 payload with maximum value 6, an unshifted block scale is

```text
a_b = max(P_b) / 6.
```

The usual NVFP4 tensor factor is `448 * 6 / amax`. Since stabilized P has
`amax <= 1`, the maximal fixed factor is therefore

```text
G = 448 * 6 = 2688,
log2(G) = 11.39231742277876.
```

The implementation stores `E4M3(G * a_b)`, scales the matching softmax
denominator by G, and subtracts `log2(G)` from LSE. G cancels between the
PV numerator and denominator. It does not improve E2M1 payload precision;
it primarily rescues small E4M3 block scales from underflow and changes
E4M3 scale rounding.

Equivalently, in the reciprocal range-scale convention, define

```text
r = 1 / G.
```

Then `r <= 1` expands the represented scale range. For example, G=2048 is
`r=1/2048`. Both conventions describe the same operation; this document
uses G because that is the compile-time kernel parameter.

The option is compile-time and defaults to zero:

```text
HAO_FP4PV_NV_P_GLOBAL_LOG2=11.39231742277876f
```

It is intentionally forbidden with shiftless softmax because shiftless P
is not bounded by 1.

## Scale distributions

The diagnostic uses the real stabilized numerator, not already-normalized
softmax probabilities.

| Model sample | E4M3 scale underflow, G=1 | Underflow, G=2688 |
|---|---:|---:|
| ViT, 10 images x 12 layers | 14.62% | 1.42% |
| BERT, 10 blocks x 12 layers | 33.66% | 3.11% |

No stabilized block-scale overflow occurs at G=2688. In contrast, the
shiftless representation exceeds E4M3 maximum for about 2.9% of sampled
ViT block scales and 5.1% of sampled BERT block scales.

## ViT result

Checkpoint:
[`nateraw/vit-base-patch16-224-cifar10`](https://huggingface.co/nateraw/vit-base-patch16-224-cifar10).
Dataset: the full 10,000-image CIFAR-10 test split.

ViT uses S197/H12/D64. The adapter pads to the existing S256/H16/D128
kernel, multiplies Q by `sqrt(2)` to preserve the D64 attention scale, and
uses one padded head dimension to give padded keys a score of `-8.84`.

| Metric | BF16 | FP4 G=1 | FP4 G=2688 |
|---|---:|---:|---:|
| CIFAR-10 accuracy | 98.54% | 98.38% | **98.48%** |
| Top-1 agreement with BF16 | 100% | 99.64% | **99.70%** |
| Logit cosine vs BF16 | 1.0 | 0.999547 | **0.999642** |
| Relative logit L2 | 0 | 2.988% | **2.653%** |
| Mean KL vs BF16 | 0 | 0.003100 | **0.002243** |
| Maximum logit error | 0 | 3.2500 | **2.6875** |

The scale lift changes 29 predictions relative to G=1: 18 become correct,
8 become incorrect, and 3 remain incorrect with a different class. BF16
disagreements fall from 36 to 30.

Attention-output relative error improves in layers 0 through 8. The
largest reduction is layer 0, from 12.715% to 11.192%. Layers 9 through
11 are nearly neutral.

Local full-split records (excluded from Git because they contain per-sample
logits; SHA-256 digests are retained in `summary.json`):

- `vit_cifar10_stable_global1_full.json`
- `vit_cifar10_stable_global2688_full.json`

## BERT result

Checkpoint: `google-bert/bert-base-uncased`. Dataset: 800 contiguous,
full-attention S256 blocks from the WikiText-2 test split. Standard 15%
MLM masking yields 30,510 evaluated tokens.

| Metric | BF16 | FP4 G=1 | FP4 G=2688 |
|---|---:|---:|---:|
| MLM loss | 2.206614 | 2.224974 | **2.222595** |
| Perplexity | 9.08490 | 9.25324 | **9.23126** |
| Masked-token accuracy | 61.740% | **61.426%** | 61.396% |
| Top-1 agreement with BF16 | 100% | **93.006%** | 92.970% |
| Relative logit L2 | 0 | **5.793%** | 5.815% |
| Mean KL vs BF16 | 0 | **0.029455** | 0.029777 |
| Maximum logit error | 0 | 16.5078 | **16.2578** |

G=2688 improves local attention error in all 12 BERT layers and improves
MLM loss, perplexity, and worst-case logit error. It is slightly worse on
masked-token accuracy, agreement, relative L2, and KL. The factor is
therefore retained as an opt-in policy rather than made the universal
default.

Local full-split records (excluded from Git; SHA-256 digests are retained
in `summary.json`):

- `bert_wikitext_mlm_global1_800.json`
- `bert_wikitext_mlm_global2688_800.json`

## Format combinations

The following pilot uses 1,000 CIFAR-10 examples and 200 WikiText-2 blocks.
NVFP4 P uses G=2048; the E8M0 MXFP4 P scale does not use the E4M3 scale-lift
parameter. These stabilized downstream kernels are separate from the
aggressive shiftless throughput policy in the format benchmark.

| QK / PV | ViT accuracy | ViT cosine | ViT relative L2 | BERT MLM accuracy | BERT cosine | BERT relative L2 |
|---|---:|---:|---:|---:|---:|---:|
| NVFP4 / NVFP4 | 98.7% | **0.999714** | **2.389%** | **60.31%** | **0.998376** | **5.697%** |
| MXFP4 / NVFP4 | 98.2% | 0.999221 | 3.969% | 59.79% | 0.996045 | 8.925% |
| NVFP4 / MXFP4 | **99.0%** | 0.999318 | 3.695% | 60.25% | 0.997603 | 6.919% |
| MXFP4 / MXFP4 | 98.4% | 0.998569 | 5.368% | 59.79% | 0.995434 | 9.585% |

The pilot favors NVFP4 QK numerically. That is expected: its E4M3 block-16
scale is finer than the E8M0 block-32 MXFP4 scale. The mixed-format result
also shows that MXFP4 PV is viable, but this sample is not large enough to
select a universal model policy.

## Scale calibration

The 1,000-example NVFP4/NVFP4 ViT sweep is not monotonic:

| G (`r=1/G`) | Accuracy | Cosine vs BF16 | Relative L2 |
|---:|---:|---:|---:|
| 1 (`1`) | **98.9%** | 0.999643 | 2.668% |
| 256 (`1/256`) | 98.7% | 0.999661 | 2.600% |
| 448 (`1/448`) | 98.6% | 0.999557 | 2.973% |
| 2048 (`1/2048`) | 98.7% | **0.999714** | **2.389%** |
| 2688 (`1/2688`) | 98.8% | 0.999712 | 2.395% |

This supports calibration by layer or model rather than treating 448 or
2688 as a universal constant. FlashAttention-3 uses the related fixed
power-of-two offset G=256 in its Hopper FP8 softmax path
([paper](https://arxiv.org/abs/2407.08608)). The 256-versus-448 precision
tradeoff is analyzed directly in
[*P-Cast Precision in FP8 Attention*](https://arxiv.org/abs/2606.06521).
The current TK option is calibrated at compile time; it is not yet learned.

## Kernel cost

The stabilized S256/H16/D128 kernel retains 128 registers, one barrier,
400 bytes static shared memory, and zero spills for both factors.

Eight interleaved same-input timing trials:

| Variant | Median time |
|---|---:|
| Stabilized G=1 | 12.576 us |
| Stabilized G=2688 | 12.496 us |

The difference is measurement noise; no latency regression is visible.
The current generated SASS has 4,368 static instructions at G=1 and 4,384
at G=2688, so the implementation is not literally instruction-free even
though the extra work is hidden at kernel level.

LSE compensation was checked against BF16: cosine `0.999995` and relative
L2 `0.301%` on the HAO seed-0 S256/H16 test.

The independent headline shiftless kernel still reproduces its archived
output exactly. An interleaved old/new replay measured 10.528 us versus
10.320 us, respectively, so this change does not regress that route.

## Reproduction

Build stabilized controls:

```bash
cd /workspace/codebases/pv/fp4_matmul/tk_fa4/fp4_fa4_fwd

make -B -f Makefile.hao_direct_fp4pv -j1 \
  OUT=/tmp/tk_vit_stable_global1_s256h16.so \
  MODULE=_C_tk_vit_stable_global1_s256h16 \
  HAO_BATCH=1 HAO_SEQ_LEN=256 HAO_HEADS=16 \
  NVCC_SPLIT_COMPILE=2 \
  HAO_FP4PV_NV_P_GLOBAL_LOG2=0.0f

make -B -f Makefile.hao_direct_fp4pv -j1 \
  OUT=/tmp/tk_vit_stable_global2688_s256h16.so \
  MODULE=_C_tk_vit_stable_global2688_s256h16 \
  HAO_BATCH=1 HAO_SEQ_LEN=256 HAO_HEADS=16 \
  NVCC_SPLIT_COMPILE=2 \
  HAO_FP4PV_NV_P_GLOBAL_LOG2=11.39231742277876f
```

Run ViT:

```bash
python eval_regular_attention.py \
  --samples 10000 --scale-sweep-samples 0 \
  --progress-every 1000 --mask-value 10 \
  --extension /tmp/tk_vit_stable_global2688_s256h16.so \
  --extension-module _C_tk_vit_stable_global2688_s256h16 \
  --output ../../results/fp4_fa4_downstream_20260728/vit_replay.json
```

Run BERT:

```bash
python eval_bert_mlm_attention.py \
  --samples 800 --scale-sweep-samples 0 \
  --progress-every 100 \
  --extension /tmp/tk_vit_stable_global2688_s256h16.so \
  --extension-module _C_tk_vit_stable_global2688_s256h16 \
  --output ../../results/fp4_fa4_downstream_20260728/bert_replay.json
```

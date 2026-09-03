# Larger-model downstream evaluation

This directory records a paired downstream check motivated by the model suites
in SageAttention3 and Attn-QAT.  It is deliberately narrower than either
paper's quality evaluation: one prompt and seed are used to validate the model
integration and to measure how low-precision attention error evolves through a
diffusion trajectory.  It is not a VBench result.

## What the papers evaluate

| Work | Inference models | Main quality evaluation | Precision/training regime |
|---|---|---|---|
| SageAttention3 | CogVideoX, HunyuanVideo, Mochi, FLUX, Stable Diffusion 3.5 | Open-Sora video prompts with CLIP/VQA/motion metrics; COCO image generation with FID, sFID, CLIPScore, and ImageReward | Training-free FP4 inference with smoothing and two-level P quantization |
| SageAttention3 training study | Qwen2.5 1.5B/3B, Llama 3.2 1B/3B | GSM8K, DROP, MMLU, HellaSwag, and FineWeb-Edu pretraining | INT8 backward study, not FP4 forward inference |
| Attn-QAT | Wan2.1 1.3B/14B | VBench and blind human evaluation on 99 VBench prompts | FP4 quantization-aware training |
| Attn-QAT | Qwen3-14B, Llama3.1-70B | MMLU(-Redux), IFEval, GPQA-Diamond, MATH-500, GSM8K, WinoGrande, ARC-c, HellaSwag, PIQA, WikiText | Continued pretraining or SFT with FP4 Attn-QAT |

The current TK forward kernel is noncausal, uses equal query/key/value head
counts, and is specialized by `(B, S, H, D)`.  Wan2.1 is therefore the clean
larger-model match: its video self-attention is noncausal MHA with `D=128`.
The LLM experiments are not directly runnable until the kernel supports causal
masking and grouped-query attention.

Primary sources:

- SageAttention3: <https://arxiv.org/abs/2505.11594>
- Attn-QAT: <https://arxiv.org/abs/2603.00040>
- Wan2.1: <https://arxiv.org/abs/2503.20314>

### Published LLM targets for a future causal/GQA kernel

Attn-QAT's continued-training table provides the cleanest larger-language-model
target. Columns are MMLU, WinoGrande, ARC-c, HellaSwag, PIQA, and WikiText
(lower is better only for WikiText).

| Model / attention | MMLU | WinoGrande | ARC-c | HellaSwag | PIQA | WikiText |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-14B BF16 | 0.8044 | 0.7403 | 0.5922 | 0.8140 | 0.8215 | 0.5700 |
| Qwen3-14B plain FP4 | 0.7965 | 0.7214 | 0.5734 | 0.8050 | 0.8052 | 0.5763 |
| Qwen3-14B Attn-QAT | 0.7984 | 0.7585 | 0.6084 | 0.8034 | 0.8188 | 0.5778 |
| Llama3.1-70B BF16 | 0.7881 | 0.8161 | 0.6135 | 0.8575 | 0.8422 | 0.2838 |
| Llama3.1-70B plain FP4 | 0.7577 | 0.7656 | 0.6015 | 0.8463 | 0.8308 | 0.3275 |
| Llama3.1-70B Attn-QAT | 0.7773 | 0.7940 | 0.6153 | 0.8557 | 0.8351 | 0.3076 |

These are trained-checkpoint results, not values that a drop-in forward kernel
should be expected to reproduce. SageAttention3's Qwen/Llama training table is
an INT8 backward experiment, so it is not an FP4 inference baseline.

## Paired Wan setup

- Checkpoints: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and
  `Wan-AI/Wan2.1-T2V-14B-Diffusers`.
- Geometry: 512x768, 17 frames, producing `S=7680` latent tokens.
- 1.3B attention: 30 layers, `H=12`, `D=128`.
- 14B attention: 40 layers, `H=40`, `D=128`.
- Prompt: `A red vintage car drives along a winding coastal road at sunset,
  with ocean waves below and natural camera motion.`
- Seed: `20260805`; guidance: `5.0`.
- Scope: only video self-attention is replaced. Text cross-attention and all
  other model operations remain BF16.
- Every provider receives the same checkpoint, prompt, seed, and scheduler.
- Classifier-free guidance invokes the transformer twice per step, so the 14B
  four-step run exercises 320 low-precision self-attention calls.

Providers:

- **TK calibrated NV/MX**: the fastest retained shiftless/sampled policy with
  model-specific affine E2M1 boundaries. Both models use `A=1.60, B=0.95` as
  the base. Wan1.3B overrides layer 0 with `1.625/0.95` and layer 11 with
  `1.575/1.05`; Wan14B uses `1.575/1.05` in layers
  `1,3,6,8-12,15-17,22-27,30-31,35`. A wider QK probe is used only in the
  model layers with extreme score ranges.
- **TK accurate NV/MX**: represented normalization and the same layer-local
  QK guard. It is slower than fast, but still does not use the exact stable
  softmax fallback.
- **HAO NV/NV**: the native full-FP4 reference path.
- **HAO NV/FP8**: HAO's FP8-PV reference path.

## Why the QK guard is needed

The QK MMA itself remains finite and is not the failed instruction. The
unguarded fast policy fails at the QK-to-P boundary. It deliberately omits a
full row-maximum pass and encodes the absolute exponential magnitude in each
MXFP4 E8M0 P scale. In ordinary layers, maximum logits are generally below
60, so `score * log2(e)` remains within the scale and approximation range. On
the first 14B diffusion step, however, layers 33, 34, 38, and 39 reach sampled
maxima of roughly 760, 560, 1080, and 730. The E8M0 exponent saturates and the
direct exp/pack path receives values far outside its fitted range, producing
non-finite represented P rows before PV. The same four layers fail on a second
prompt and seed. The 1.3B model has the corresponding problem in layers
27--29. Changing only the P payload format therefore cannot fix it.

The guard moves 128 globally distributed K/V rows into the first physical
tile. It reads all four QK quarters of that tile and forms a per-row score
anchor. The later transform uses `(score - anchor) * log2(e)` and applies a
120-binary-exponent safety margin, bringing the direct exp/pack input back
inside its legal range. The same row shift cancels in the represented
denominator. This is not a full row-max pre-scan and it does not reload the
score matrix. Exact attention is invariant to the joint K/V permutation; only
low-precision grouping and summation order change.

The margin is exposed as `--anchor-margin-log2` in the bundle builder and is
recorded in `manifest.json`. It is an integer binary-exponent hyperparameter:
smaller values retain more small probabilities but leave less protection
against an unseen score above the 128-row sample; larger values protect the
high side but can quantize more of the P tail to zero. The coupled common
P-scale translation is exposed as `--stored-scale-shift-log2`. The calibrated
runs use margin `120`, with stored-scale shift `16` for Wan1.3B/H12 and `14`
for Wan14B/H40. Both are compile-time specializations, so changing them adds
no runtime control branch.

E8M0's smallest nonzero exponent is approximately $-126$. Margins 121--126
are therefore legal experiments but leave essentially no lower-range
headroom; a margin above 126 would zero the sampled maximum unless the P-scale
representation is redesigned. It is not sound to obtain larger values merely
by removing the bound.

For inference, the natural deployment is a calibrated per-layer margin. On a
representative calibration set, measure the gap between the true row maximum
and the 128-key anchor, choose a high quantile plus a safety allowance, and
route only sensitive layers. For pre-training or QAT, the same value can be
updated from running layer statistics or optimized through a simulated
FP4/E8M0 quantizer. A free scalar inserted into exact softmax is not learnable
for this purpose: exact softmax cancels a common row shift, so the training
objective must include quantization or an explicit range penalty.

## Results

Each cell is cosine similarity / relative L2 error against the paired BF16
output.  `Latent` compares the final diffusion latent. `Decoded` compares the
full decoded pixel array before video encoding.

| Model | Steps | Output | TK calibrated NV/MX | TK accurate NV/MX | HAO NV/NV | HAO NV/FP8 |
|---|---:|---|---:|---:|---:|---:|
| Wan2.1-1.3B | 1 | Latent | 0.98773 / 0.17474 | 0.98629 / 0.17165 | 0.98772 / 0.17799 | 0.99358 / 0.11825 |
| Wan2.1-1.3B | 4 | Latent | 0.97031 / 0.24959 | 0.96642 / 0.26173 | 0.96857 / 0.25084 | 0.97673 / 0.21577 |
| Wan2.1-1.3B | 20 | Decoded | 0.97999 / 0.19954 | 0.98115 / 0.19920 | 0.98790 / 0.15899 | 0.99054 / 0.14195 |
| Wan2.1-14B | 1 | Latent | 0.99260 / 0.12765 | 0.98809 / 0.15533 | 0.99250 / 0.12236 | 0.98921 / 0.14671 |
| Wan2.1-14B | 4 | Latent | 0.93043 / 0.36886 | 0.92029 / 0.40207 | 0.93050 / 0.36968 | 0.92402 / 0.38488 |
| Wan2.1-14B | 20 | Latent | 0.84687 / 0.53759 | 0.84770 / 0.54359 | 0.85217 / 0.54292 | 0.84481 / 0.55737 |

The final calibrated route is model-specific rather than a copy of the 14B
map. On Wan1.3B, a teacher-forced grid found many local winners, but broad
replacement regressed when every low-precision layer interacted. Retaining
only layers 0 and 11 raises the two-prompt four-step mean cosine from
`0.96764` to `0.96908` and lowers relative L2 from `0.25837` to `0.25293`.
Against the previously reported global route, it also improves the decoded
twenty-step row from `0.97958/0.20355` to `0.97999/0.19954`; the one-step row
is effectively unchanged.

Wan14B retains a regularized 20-layer override. Across four repeated runs on
each prompt, mean cosine/relative L2 moves from `0.93227/0.36562` to
`0.93280/0.36273` on the calibration prompt and from `0.91413/0.40577` to
`0.91712/0.39870` on the held-out prompt. Individual diffusion trajectories
remain noisy, which is why the single-run rows above need not improve
monotonically even when the repeated comparison does.

The base affine retune is a zero-cost Pareto point. It changes two compile-time
constants in the packed FMA code map, not the instruction schedule. The final
layer overrides use the same instruction sequence and resource allocation, so
their routing also has no kernel-level cost.

The fit is deliberately scoped to the Wan bundle. On the Gaussian operator
case it raises cosine but worsens relative L2, so the generic cross-shape
`fast` policy retains `A=1.50, B=1.20`. `build_wan_nv_mx_bundle.py` records
the calibrated constants in `fast_affine_code_map` and exposes
`--fast-affine-a` and `--fast-affine-b` for reproducible retuning.

The sweep also tested replacing selected affine pairs with native `EX2`:

| Base path | H40 base ms | Gaussian rel. L2 | Wan14B step-1 rel. L2 |
|---|---:|---:|---:|
| Old all-affine, `1.50/1.20` | 0.4136 | 0.3392 | 0.2061 |
| Native density 1, all quarters | 0.4385 | 0.3285 | 0.2019 |
| Native density 2, all quarters | 0.4918 | 0.3179 | 0.1740 |
| Calibrated all-affine, `1.60/0.95` | 0.4132 | 0.3487 | 0.1210 |

Native `EX2` improves the Gaussian operator metric but adds 6--19% latency.
The calibrated all-affine map moves in the opposite direction on that test
yet gives the best model trajectory at the original speed. This is why the
two fits remain separate Pareto points rather than one replacing the other.

The H40 fast rows use the calibrated shift `14`. Relative to shift `16`, its
four-step cosine improves from `0.914930` to `0.916649` and relative L2 from
`0.409001` to `0.404455`; at twenty steps the corresponding changes are
`0.831177` to `0.832581` and `0.556096` to `0.554132`. Paired H40 timings are
`0.486864` ms for shift `14` and `0.486912` ms for shift `16`, so this
accuracy gain is effectively free.
Shift `13` remains finite but is less accurate, and shifts `12` and below
fail. H12 requires shift `16`: shift `14` is non-finite and shift `15` worsens
four-step cosine/relative L2 from `0.961564/0.293451` to
`0.957372/0.304603`.

Paired 5-second timing windows confirm that the affine constants are free at
kernel level. In forward order, the old and calibrated H40 bases both measure
`0.413248` ms; in reverse order they measure `0.411488` and `0.411520` ms.
H12 measures `0.160384` versus `0.160480` ms. The generated SASS instruction
count and resource usage are unchanged, and a clean bundle-builder binary is
bit-exact against the exploratory candidate.

All reported providers remain finite in these runs. The 14B trajectory is the
most useful result: all four drop-in low-precision routes are close after one
step and substantially displaced after four and twenty steps.  The ordering at
twenty steps must not be read as a quality ranking; diffusion trajectories are
sensitive, and latent distance can grow even when decoded outputs remain
perceptually useful.  The common drift supports Attn-QAT's central result that
a BF16 checkpoint should be trained or fine-tuned with its deployed attention
quantizer if final quality must match BF16.

The published Attn-QAT result is not contradicted by this table.  Their Wan
weights have been quantization-aware trained and are evaluated over VBench and
99-prompt human preference tests.  These runs intentionally apply each kernel
post-training to the original BF16 checkpoint.

## Kernel performance

End-to-end elapsed times in the JSON files are diagnostic only.  The TK adapter
currently quantizes Q/K/V in Python, while the first HAO call includes CuTe JIT
compilation. Neither is a fair kernel timing. Independently warmed kernel
measurements give the following layer-count-weighted costs:

| Model | Policy | TK base | QK guard | TK routed | TK BF16 | HAO BF16 | HAO NV/NV | HAO NV/FP8 | vs TK BF16 | vs fastest HAO |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan2.1-1.3B | calibrated fast | 0.161888 | 0.188448 | 0.164544 | 0.284672 | 0.288768 | 0.346624 | 0.288768 | 1.73x | 1.75x |
| Wan2.1-1.3B | accurate | 0.196896 | 0.188448 | 0.196051 | 0.284672 | 0.288768 | 0.346624 | 0.288768 | 1.45x | 1.47x |
| Wan2.1-14B | calibrated fast | 0.413728 | 0.487488 | 0.421104 | 0.760160 | 0.888160 | 0.907264 | 0.751648 | 1.81x | 1.78x |
| Wan2.1-14B | accurate | 0.509344 | 0.487488 | 0.507158 | 0.760160 | 0.888160 | 0.907264 | 0.751648 | 1.50x | 1.48x |

The routed cost uses 27 base plus 3 guarded layers for 1.3B and 36 base plus
4 guarded layers for 14B. The guard therefore fixes the QK range failure for
about 1.6--1.8% weighted kernel overhead in the fastest policy. `TK BF16` is
the retained FA4 baseline used throughout this report; `HAO BF16` is HAO's
native CuTe kernel and is listed separately. HAO values are six-order,
independently warmed medians from `wan21_*_hao_kernel_bench.json`. The fastest
HAO route is tied BF16/NV-FP8 at H12 and NVFP4-QK/FP8-PV at H40.

## Artifacts and next step

- `build_wan_nv_mx_bundle.py` builds the calibrated fast, accurate, affine
  override, and QK-guard extensions and writes a manifest that records the
  model-specific layer routing. From
  `tk_fa4/fp4_fa4_fwd`, build and replay the 14B fast policy with:

  ```bash
  python3 build_wan_nv_mx_bundle.py \
    --model 14b \
    --output-dir /tmp/wan_nv_mx_14b_s7680
  CUDA_VISIBLE_DEVICES=0 python3 eval_wan_video.py \
    --model Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --providers bf16,tk \
    --policy-manifest /tmp/wan_nv_mx_14b_s7680/manifest.json \
    --policy fast --steps 4 \
    --output ../../results/fp4_fa4_wan_20260805/replay_14b_fast.json
  ```

- `eval_wan_video.py` is the paired adapter.
- `*_step1.json`, `*_step4.json`, and `*_step20*.json` contain complete run
  metadata and metrics.
- `*fast_affine160_095*` contains the calibrated-fit model and paired timing
  records added by the Pareto sweep.
- `*_preview.png` and `*.mp4` provide the 1.3B decoded visual comparison.
- `wan21_*_hao_kernel_bench.json` contains the exact-shape warmed HAO timing
  samples and provider-order audit.
- Saved `.pt` files permit metric recomputation without rerunning the model.

The statistically meaningful follow-up is the Attn-QAT protocol: 99 fixed
VBench prompts, a normal generation schedule and frame count, decoded video
metrics, and blinded preference evaluation.  Reproducing their headline
quality also requires their QAT checkpoint or training recipe; the present
kernel-only PTQ check cannot substitute for that experiment.

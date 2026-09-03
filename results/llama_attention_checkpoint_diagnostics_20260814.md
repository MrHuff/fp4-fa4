# Llama attention checkpoint diagnosis (2026-08-14)

## Bottom line

The current **causal D192 streaming** MXFP4-forward route is converging worse
primarily because its MXFP4 probability/PV path attenuates the attention
output, not because QK, RoPE, or projection-native NVFP4 QKV is failing.  It
is not the optimized HAO-direct D64/D128 FA4 forward, which is noncausal.  On
the supported
`S=8192, H=16, K=2048, Dqk=192, Dv=128` proxy, the most credible deployment
route today is projection-native NVFP4 QKV/output plus BF16 attention forward
and the retained FP4/FP8 backward.  It is **1.142x faster** than the BF16
training-scope baseline on GPU 0 while its three-seed, 100-step loss under a
shared BF16 evaluator is only **1.60% higher**.

The all-low-precision attention route is **1.074x faster** in the same training
scope, but its attention output has a stable norm ratio near `0.761`.  That
forward bias changes the upstream loss gradient and then the Q/K optimizer
updates.  It must be fixed before stochastic rounding, QKV RHT, or a more
elaborate QKV quantizer is likely to move end-to-end loss materially.

## Scope and model caveat

These are deterministic teacher/student attention-sublayer runs, not language
model pretraining runs.  They train Q/K/V and output-projection weights against
fixed BF16 teacher activations with AdamW for 100 steps.  RMSNorm, residual,
MLP, cross entropy, optimizer-state quantization, and data variation are not in
the experiment.

The local Llama-3 `1B` configuration is 1,235,746,816 parameters with hidden
size 2048, 16 layers, 32 query heads, 8 KV heads, and head dimension 64.  The
current integrated backward path is fixed at QK depth 192, V depth 128, and
equal Q/K/V head counts.  Therefore the measured `S8192/H16/K2048` case is the
closest supported hidden-size proxy; it is not a stock Llama-3.2-1B drop-in.
Native D64 and GQA backward support are required before claiming a 1.2B-model
speedup or convergence result.

## GPU 0 latency

Median of 31 samples after 8 warmups on one NVIDIA GB200:

| Timed scope | BF16 | Full MXFP4 forward | BF16-attention hybrid |
|---|---:|---:|---:|
| Forward + backward activation gradients | 3.017 ms | 2.782 ms, 1.085x | 2.579 ms, 1.170x |
| Plus QKV/output weight gradients | 3.323 ms | 3.096 ms, 1.074x | 2.911 ms, 1.142x |

The corresponding latency reductions in the training scope are 6.85% and
12.40%.  The hybrid wins because the projection-native and fused backward
work is valuable, while this particular MXFP4 forward attention kernel is not:
with identical prepacked operands, BF16 attention takes 0.469 ms, retained
MXFP4 takes 0.612 ms, and the diagnostic QK-FP4/V-BF16 implementation takes
0.666 ms.

The optimized HAO-direct D64/H32 forward measurements reach 1.329x
(NVFP4-QK/MXFP4-PV) and 1.364x (NVFP4-QK/NVFP4-PV) versus the HAO BF16
forward at S8192.  Those kernels hardcode `HAO_DIRECT_NONCAUSAL=true`; their
benchmark uses `causal=False`, equal Q/K/V head counts, and `pack_gqa=False`.
They cannot be substituted into a Llama decoder without adding triangular job
scheduling, diagonal masking before P quantization, and GQA mapping.  The
numbers establish a useful causal-port target, not a measured Llama speedup.

## Three-seed 100-step trajectories

All rows use the same teacher/student construction.  “Shared final” evaluates
both learned weight sets through BF16, separating optimization trajectory from
the low-precision forward's irreducible teacher floor.  Ratios are low
precision divided by BF16; lower is better.  Update and gradient entries are
cosines at step 100.

| Route | Shared final loss | Native final loss | Q update | Native Q gradient | Same-weight Q gradient |
|---|---:|---:|---:|---:|---:|
| MXFP4 forward, gain 1.0, loss scale 2^11 | 1.0532x | 1.8065x | 0.310 | 0.330 | 0.783 |
| MXFP4 forward, gain 1.3, chain rule, scale 2^9 | 1.0234x | 1.6394x | 0.597 | 0.636 | 0.744 |
| Gain 1.3, delta-corrected STE, scale 2^9 | 1.0233x | 1.6379x | 0.565 | 0.581 | 0.929 |
| QK-FP4 + BF16 V/PV diagnostic, scale 2^9 | 1.0159x | 1.4715x | 0.745 | 0.766 | 0.964 |
| BF16 attention on low-precision QKV, scale 2^9 | 1.0160x | 1.4724x | 0.754 | 0.775 | 0.963 |

The 1.3 gain is a localization tool, not yet a general calibration scheme.  Its
least-squares value is stable near 1.302 in this synthetic workload and it
reduces attention-output relative L2 from about 0.263 to 0.127.  A fixed gain
has changed sign across unrelated forward workloads in prior experiments, so
it requires real-model/QAT validation before retention.

The delta-corrected STE demonstrates that the saved-output term in the
softmax derivative can be repaired: same-weight Q-gradient cosine rises from
roughly 0.744 to 0.929.  The short-run shared loss barely changes because the
forward output still changes the loss gradient before attention backward.

## Checkpoint attribution

At the initial checkpoint with gain 1.0:

| Isolated change | Q gradient cosine | Q norm ratio | Q relative L2 |
|---|---:|---:|---:|
| Projection-native QKV + RoPE, then BF16 attention | 0.9805 | 0.9870 | 0.1965 |
| MXFP4 attention added at the same low-precision QKV | 0.5922 | 1.6704 | 1.3461 |
| Backward quantization at the same low-precision state | 0.7540 | 0.5263 | 0.6952 |

Forward checkpoints tell the same story:

- Projection-native Q/K/V plus RoPE has cosine about 0.991 and norm ratio
  approximately 1.0 for each operand.  Q/K/V max-to-RMS is only 4.0--4.4, so
  sparse QKV outliers are not dominant here.
- LSE is effectively exact (`~8e-5` relative L2), excluding score reduction and
  row-max computation as the main source.
- Retained MXFP4 attention at identical low-precision QKV has output cosine
  `0.99198`, norm ratio `0.7616`, and relative L2 `0.2628`.
- Replacing only V/PV with BF16 while keeping FP4 QK gives output cosine
  `0.999976`, norm ratio `0.999981`, and relative L2 `0.00692`.  This localizes
  the dominant bias to probability quantization/PV.
- E8M0 probability-scale rounding modes `rte`, `encode`, and `decode` all leave
  the output norm at `0.7616` and give indistinguishable updates.  Scale
  exponent rounding is not the cause.
- The NVFP4 output projection at a shared attention output is healthy: cosine
  is about 0.991 and norm ratio about 0.998--1.0.

The original `2^20` loss scale also saturated the fixed-scale FP8 dO
publication path.  Moving the raw route to `2^11` raises the mean step-100 Q
update cosine from 0.084 to 0.310 and the retained absolute BF16 loss
improvement from 0.835x to 0.962x.  Gain-corrected and control routes work best
locally around `2^9`; dynamic dO scaling remains preferable to tuning a global
loss scale around a fixed publication scale.

## Quantization decisions

1. **First target: P/PV.** Build an optimized FP8-P/PV rung or change the
   MXFP4 probability representation/normalization so that the row sum and
   output norm are preserved.  The current BF16-V/PV control proves the
   achievable quality, although that unfused diagnostic kernel is too slow.
2. **Then activation-only Four Over Six in projection epilogues.** Existing
   local trained-head replay reduced CE gap by 30.5% and logit RMSE by 21.4%
   for 0.78% latency.  Weight-side Four Over Six regressed accuracy and should
   stay disabled.  On the current route this can address the remaining
   projection-native floor, not the main P/PV error.
3. **RHT only when real checkpoints show tails.** The synthetic QKV tensors do
   not show serious outliers.  RHT may be useful along the projection reduction
   dimension on real Llama activations, but applying it to QKV now adds work to
   a stage whose gradients already have about 0.98 cosine.
4. **Stochastic rounding later.** It is most plausible for long-run weight or
   optimizer-state drift after deterministic forward bias is removed.  It
   cannot correct a stable 24% attention-output norm deficit and is unlikely to
   change these 100-step results materially by itself.

## Next experiment order

1. Make a source-isolated causal port of the optimized HAO-direct D64 forward:
   triangular query/key job bounds, diagonal score masking before P
   quantization, and 32Q/8KV GQA mapping.  Do not modify the read-only forward
   source.  Retain the current causal D192 route only as an integration
   reference.
2. Add native D64/GQA backward and projection contracts; retain BF16 attention
   as the correctness baseline.
3. Optimize an FP8 probability/PV middle rung with row-sum-preserving scaling,
   then rerun these checkpoint splits.  Promote it only if it beats the 2.911 ms
   hybrid training latency while keeping shared-evaluator loss near the 1.016x
   control.
4. Run short real Llama-3.2-1B cross-entropy trajectories with checkpointed
   layerwise activations, dQ/dK/dV, projection gradients, Adam updates, and
   validation logits.  Separate the quantization floor from update divergence
   exactly as in this harness.
5. Only after P/PV is under control, test activation-only Four Over Six, then
   conditional projection RHT, and finally stochastic rounding in longer runs.

## Reproducibility artifacts

- E2E timing: `llama_attention_e2e_s8192h16k2048_gpu0_20260814.json`
- Kernel-only controls: `llama_attention_forward_quality_control_timing_s8192h16_20260814.json`
- Raw three-seed runs: `llama_attention_diagnostic_lossscale2e11_s8192h16_seed*_steps100_20260814.json`
- Gain-corrected runs: `llama_attention_diagnostic_gain1p30_lossscale2e9_s8192h16_seed*_steps100_20260814.json`
- Delta-corrected STE: `llama_attention_diagnostic_gain1p30_scale2e9_delta_corrected_ste_s8192h16_seed*_steps100_20260814.json`
- QK-FP4/V-BF16 control: `llama_attention_qkfp4_vbf16_control_scale2e9_s8192h16_seed*_steps100_20260814.json`
- BF16-attention control: `llama_attention_bf16_on_lowp_qkv_scale2e9_s8192h16_seed*_steps100_20260814.json`
- Stage attribution: `llama_attention_stage_split_gain1p0_scale2e11_s8192h16_seed2026081417_20260814.json`

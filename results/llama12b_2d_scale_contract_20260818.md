# Llama-1.2B 2-D scale-contract audit

Date: 2026-08-18

This audit compares the two retained low-precision attention routes after
putting them on the same scale contract. Learned Q/K/V/O projection weights
use one NVFP4 scale over each 16x16 tile, MXFP4 V uses one scale over each
32x32 sequence-by-depth tile, Q and K forward multipliers are independent,
and the backward Q/K/V field gains default to one. In particular, the MXFP4
route no longer inherits the historical common 0.632 gradient multiplier.

## Representation checks

- Packing a learned weight and independently packing its transpose gives a
  100% transpose match for E2M1 payloads, E4M3 local scales, and global scale.
- The same 32x32 MXFP4 V tile is published in forward- and backward-oriented
  layouts; payloads and repeated E8M0 scales are exact transposes.
- The D128 byte-level validator checks 262,144 V codes and 20,480 replicated
  scale values with zero mismatches. Store-BF16 and no-store specializations
  also produce identical low-precision payloads and metadata.
- Isolated learned-projection cosine is 0.989259 for 2-D weights versus
  0.990916 for legacy 1-D weights. Isolated decoded-V cosine is 0.993031 for
  2-D MXFP4 versus 0.993560 for independently scaled 1-D layouts. The reason
  for retaining 2-D quantization is fprop/bprop operator consistency, not a
  claim that sharing a scale improves one-shot reconstruction error.

## Controlled three-route result

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  tk_fa4/lowp_fa4_bwd/compare_llama12b_mx_fp8pv.py \
  --rounds 24 --training-batches 4 \
  --output /tmp/llama12b_mx_vs_fp8pv_2d_unitgain_20260818.json
```

The benchmark is the synthetic 16-layer, S4096 Llama-1.2B training proxy. It
uses 24 rotated CUDA-event samples and reports the median component times.

| route | forward (ms) | backward (ms) | optimizer (ms) | step (ms) | speedup vs BF16 |
|---|---:|---:|---:|---:|---:|
| BF16 CuTE | 27.718 | 45.596 | 10.726 | 84.050 | 1.000x |
| NVFP4 QK + MXFP4 PV | 22.466 | 44.239 | 10.732 | 77.428 | 1.086x |
| NVFP4 QK + exact FP8 PV | 24.884 | 44.233 | 10.733 | 79.860 | 1.052x |

The two low-precision backward timings are effectively identical. The MXFP4
route's extra 2.43-ms whole-step advantage comes from forward PV.

Initial-state accuracy against the BF16 route:

| route | logits cosine | logits rel. L2 | dWQ cosine / norm | dWK cosine / norm | dWV cosine / norm |
|---|---:|---:|---:|---:|---:|
| NVFP4 QK + MXFP4 PV | 0.69348 | 0.78111 | 0.27691 / 0.94039 | 0.32003 / 0.95347 | 0.39347 / 0.74783 |
| NVFP4 QK + exact FP8 PV | 0.77821 | 0.66776 | 0.44378 / 0.95483 | 0.45383 / 0.95514 | 0.68284 / 0.97812 |

The old MXFP4 comparison had Q/K/V norm ratios near 0.593/0.593/0.425
because only that route inherited the 0.632 multiplier. Removing it fixes the
route-wide Q/K shrink. The remaining MXFP4 weakness is directional error,
especially through V/PV. Although both routes use the same QK policy at layer
zero, their PV outputs alter the residual stream; later layers therefore do
not receive the same hidden states or Q/K tensors.

The 24-step optimization proxy remained finite for all routes. Last-cycle
median loss was 4.281 for BF16, 5.797 for MXFP4 PV, and 5.344 for FP8 PV.
This is a short systems/numerics diagnostic, not a convergence claim.

## Adaptive Q/K and gradient-format decision

The optional per-layer/per-paired-head scale initialization selected roughly
Q=2.13 and K=2.0 in the first layer. In the one-layer audit it moved body
cosine from 0.96253 for fixed Q=2.25/K=2.0 to 0.96275, a small accuracy gain
with no kernel-work reduction. It remains optional rather than the default.

NVFP4 projection dgrad with field-local decode factors folded into E4M3
metadata restored the expected gradient norms, but measured 46.11 ms versus
43.89 ms for the retained BF16 projection-dgrad route. RHT and stochastic
rounding were therefore not added: selected projection dgrad and weight-grad
operands are still BF16, so adding quantize/dequantize work around them would
not address the observed forward/PV error.

## Retained policy

- Use 16x16 NVFP4 scaling for learned projection weights and their transpose.
- Use one 32x32 scale when MXFP4 V must serve forward and backward layouts.
- Keep Q and K representation multipliers independent.
- Decode Q/K/V backward fields independently; infer no correction gain from a
  forward softmax mode.
- Treat MXFP4 PV as the speed endpoint and exact FP8 PV as the better-accuracy
  endpoint until longer convergence runs establish an acceptable default.

## Launch-contract correction

The larger 32x32 epilogue scratch exposed an independent D128 launch failure:
four dynamic load stages plus the output ring left less static shared-memory
headroom than the V publisher requires, causing `cudaFuncSetAttribute` to
reject the kernel. Projection paths that allocate that staging fragment now
use three load stages. The D64 paired-QKV path already used three stages, while
the FP8-only dO producer publishes directly from its dynamic output tile and
retains four. The controlled Llama timing above is therefore unchanged. The
rebuilt D128 unified-QKV validator and D64 dO projection smoke both pass.

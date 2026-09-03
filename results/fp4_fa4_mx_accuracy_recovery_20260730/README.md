# FP4 FA4 Accuracy Recovery, 2026-07-30

## Retained results

All timings are TK forward-kernel timings on GB200 for B1, S4096, H24,
D128. Downstream results use 200 CIFAR-10 images through
`nateraw/vit-base-patch16-224-cifar10` at image size 1008. Dynamic input
quantization is outside the kernel timing.

| Route | QK preparation | P affine `(A, B)` | Time (ms) | Cosine vs BF16 | Relative L2 | Top-1 agreement | Model accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| MX/MX fast | MXFP4 RTE | `(2.0, -0.2)` | 0.115520 | 0.995929 | 0.090427 | 96.5% | 87.0% |
| MX/MX recovered | MXFP4 Q-ceil, K-RTE | `(2.0, -0.2)` | 0.115520 | 0.996875 | 0.079088 | 98.5% | 88.5% |
| NV/MX accuracy | NVFP4 QK | `(1.65, 0.8)` | 0.116224 | 0.998861 | 0.047994 | 98.5% | 89.0% |
| HAO stabilized reference | HAO reference route | n/a | 0.192512 | 0.999098 | 0.042612 | 98.0% | n/a |

The NV-QK/MX-PV accuracy tier costs 0.000704 ms, or 0.61%, relative to
the retained MX/MX kernel. It reduces downstream relative L2 by 46.9%
relative to the MX/MX RTE point and comes within 0.00538 absolute
relative L2 of the HAO stabilized reference.

The attention kernel for the retained NV/MX tier uses 128 registers,
one barrier, 432 bytes of static shared memory, 208896 bytes of dynamic
shared memory, and has no spills.

## What changed

1. The fast MX P transform was made configurable without changing its
   instruction count. It remains one packed FMA followed by the existing
   nonnegative clamp and E2M1 pack.
2. The represented P denominator is accumulated by the correction warp
   group instead of estimated from sparse samples.
3. A 32-key global anchor stabilizes the shiftless P path.
4. The stored P E8M0 exponent is translated down by eight. PV and the
   represented denominator consume the same translated scale, so the
   factor cancels during normalization. This prevents FP32-range
   overflow without changing P payloads or kernel time.
5. Q, K, and V MXFP4 scale-rounding modes are independently selectable
   in the downstream evaluator.
6. NVFP4 QK was rebuilt on the same denominator, anchor, scheduling, and
   P/PV pipeline as MXFP4 QK. This removed nearly all of the latency gap
   seen in the older NV/MX route.

## Error decomposition

Offline replay on representative ViT layers showed:

- QK-only relative L2 is roughly 0.064 to 0.126 after Q-ceil.
- V-only relative L2 is roughly 0.042 to 0.074.
- Replacing the fast P encoder with exact exp2 plus exact MXFP4 packing
  does not reliably improve the combined result and sometimes worsens
  it.

The affine P map is therefore not merely approximating exp2. It also
compensates systematic QK and V quantization error. This is why fitting
the affine map against BF16 P values alone selected the wrong
coefficients.

## Accepted transformations

### Q ceiling for MXFP4

The normal Q quantizer rounds an E8M0 scale down for maxima in the lower
half of an exponent bin. That clips the largest Q value in those
blocks. Always rounding the Q scale upward preserves maxima and improves
the 200-image result from 0.090427 to 0.079088 relative L2. Applying the
same rule to K or V is harmful.

### NVFP4 QK with MXFP4 P and V

NVFP4 block-16 E4M3 scales remove most of the remaining QK error. The
old NV/MX path was slow because it did not contain the current MX P/PV
pipeline. Rebuilding NV QK in the retained structure changes the matched
kernel time from 0.115520 to 0.116224 ms.

The P fit must change with the QK format:

- MXFP4 QK prefers the sharper `(2.0, -0.2)` compensation.
- NVFP4 QK prefers `(1.65, 0.8)`, closer to the original exp fit.

## Rejected transformations

- Per-layer affine dispatch improved 60-image relative L2 by only
  0.00003 and was not worth multiple kernels.
- Restoring a cubic encoder for one quarter cost about 0.006 to
  0.008 ms and did not improve downstream accuracy.
- Intermediate Q E8M0 thresholds (`1/8`, `1/4`, and `3/8`) lost to hard
  Q ceiling on the full 200-image set.
- A per-block L2 oracle lowered local Q reconstruction error but worsened
  downstream relative L2 to 0.082175. Preserving Q maxima matters more
  than minimizing unweighted Q MSE.
- Reciprocal global and per-channel Q/K equalization did not generalize.
- Active-channel spreading looked strong on 60 images but regressed to
  0.085404 relative L2 on 200 images.
- Full Hadamard rotation spread the padding-mask coordinate into every
  scale block and was harmful.
- A mask-isolated 96-channel DCT also regressed, indicating that the
  learned Q/K basis is already preferable to generic rotations here.
- Floor, ceiling, and L2-oracle scale selection for K or V all worsened
  downstream results.

## Retained NV/MX build

The accuracy tier uses:

```text
HAO_QK_SCALE_MODE=0
HAO_PV_SCALE_MODE=1
HAO_FP4PV_MX_PWL_EXP2=23
HAO_FP4PV_MX_AFFINE_A=1.65f
HAO_FP4PV_MX_AFFINE_B=0.8f
HAO_FP4PV_MX_STAGE0_AFFINE_MASK=14
HAO_FP4PV_MX_STAGE1_AFFINE_MASK=14
HAO_FP4PV_MX_SHIFTLESS_SOFTMAX=1
HAO_FP4PV_MX_DENOM_CORR_WG=1
HAO_FP4PV_MX_GLOBAL_ANCHOR32=1
HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2=64
HAO_FP4PV_MX_STORED_SCALE_SHIFT_LOG2=8
HAO_FP4PV_MX_DELAYED_HALF_Q2=1
HAO_FP4PV_MX_DELAYED_EARLY_Q3=1
HAO_FP4PV_MX_EARLY_Q2_REDUCE=1
HAO_FP4PV_MX_EARLY_P=1
HAO_FP4PV_MX_EARLY_ASYNC_SCALE=1
HAO_FP4PV_MX_SHIFTLESS_CORR_BYPASS=1
HAO_FP4PV_SOFTMAX_REGS=184
HAO_FP4PV_CORRECTION_REGS=96
```

## Next validation

1. Run the retained MX/MX and NV/MX tiers on additional regular-attention
   models and sequence/head shapes.
2. Measure quantization plus attention end to end. The latency table
   above intentionally isolates the attention kernel.
3. Add named build profiles for the fast MX/MX and accurate NV/MX
   configurations after cross-model validation.
4. Revisit P only if the NV-QK decomposition shows a consistent
   P-specific residual; generic exp2 accuracy is not the present limit.

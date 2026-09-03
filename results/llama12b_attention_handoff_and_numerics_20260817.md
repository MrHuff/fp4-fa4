# Llama-1.2B attention handoff and numerical localization

Date: 2026-08-17

## Outcome

The backward attention-to-projection handoff no longer materializes a
loss-scaled BF16 `dy` tensor or a sequence-major copy of LSE.  Dynamic NVFP4
normalization makes a uniform positive multiplier cancel from the FP4 payload
and block scales, so the power-of-two loss scale is folded into the operand's
single global decode scalar after packing.  The dO projection epilogue now
also reads the forward kernel's native `[B,H,1,S]` LSE layout.

The replacement is bitwise equal to the old path for the FP4 payload, E4M3
block scales, global decode scale, published FP8 dO, dPsum, and log2-LSE.  It
therefore contributes no new numerical error.

## Performance

The one-layer boundary profile changed from:

| D64 backward handoff | Previous | Fused |
|---|---:|---:|
| BF16 loss-scale + LSE layout copy | 59.52 us | removed |
| NVFP4 dy packing | 31.17 us | 35.14 us |
| Combined | 90.69 us | 35.14 us |

This saves 55.55 us per layer, or about 0.89 ms across 16 layers.  In the
full benchmark, low-precision backward moved from 44.59 ms in the preceding
alternating run to 43.88 ms, a 0.72 ms observed reduction.  Cross-run device
state also made BF16 about 0.90 ms faster, so the honest comparison is the
new same-process alternating result rather than the ratio between runs.

| Complete 1.2B step | BF16 CuTe | Low precision | Ratio |
|---|---:|---:|---:|
| Forward | 28.526 ms | 22.752 ms | 1.254x |
| Backward | 45.150 ms | 43.877 ms | 1.029x |
| Fused AdamW | 10.730 ms | 10.734 ms | 1.000x |
| Full step | 84.346 ms | 77.319 ms | 1.0909x throughput |
| Useful MFU | 17.741% | 19.354% | +1.612 pp |

## Numerical localization

The current D64 path was decomposed on identical first-layer input and
weights.  QKV projection, the attention body, and output projection were
measured separately, followed by a 16-layer hidden-state trace.

| First-layer forward boundary | Cosine | Relative L2 | Norm ratio |
|---|---:|---:|---:|
| NVFP4 Q projection | 0.99097 | 0.13439 | 0.99967 |
| NVFP4 K projection | 0.99098 | 0.13427 | 0.99980 |
| NVFP4 V projection | 0.99097 | 0.13437 | 0.99959 |
| Exact attention: projected QKV vs BF16 QKV | 0.98299 | 0.18385 | 0.99092 |
| FP4 attention body at the same projected QKV | **0.82061** | **0.57248** | **0.78689** |
| NVFP4 O projection at the same attention output | 0.99205 | 0.12587 | 0.98811 |
| Complete attention branch | **0.80618** | **0.59308** | **0.76532** |

The error therefore does not originate primarily in learned-projection
quantization.  The dominant forward loss is the NVFP4-QK/MXFP4-PV attention
body, specifically its represented probability/denominator policy.  The
first bad branch enters the residual stream, then the unchanged MLP amplifies
the input difference.  Layer-0 output cosine is 0.69696, final hidden cosine
is 0.34110, and sampled-logit cosine is 0.50391.

The original same-operand backward decomposition mixed the deployed forward
statistics into the backward comparison and did not decode the common x4 dO
publication scale for dQ/dK.  A corrected exact-statistics experiment is
recorded in `llama12b_backward_accuracy_recovery_20260817.md`.  After changing
the E4M3 dS lift from 256 to 16, using degree-2 exp2, decoding all three
gradient families, and folding the stable probability-mass correction into
the handoff, the deployed backward measures:

| Backward boundary | Cosine | Relative L2 | Norm ratio |
|---|---:|---:|---:|
| NVFP4 dO projection | 0.99099 | 0.13423 | 0.99946 |
| Fixed-scale FP8 dO publication | 0.99965 | 0.02660 | 0.99941 |
| dQ from attention | 0.98804 | 0.15443 | 0.99625 |
| dK from attention | 0.98893 | 0.14847 | 0.99289 |
| dV from attention | 0.99685 | 0.08009 | 1.00818 |

dO preparation and the corrected backward are healthy.  With exact decoded
forward statistics, dQ/dK/dV cosine reaches 0.99704/0.99676/0.99964.  The
remaining deployed difference is driven by rowwise variation in the
forward-LSE/represented-denominator mismatch rather than by the backward dS
format or the projection handoff.

## Policy check

The retained forward artifact selects the `fast` mode-23 shiftless/log2-P
policy.  A matched test against the available non-quantized-denominator build
did not repair quality: on model-scale normalized inputs it reached 0.701
output cosine versus 0.792 for the retained quantized-denominator build and
was slower.  Thus the isolated `mx_quantized_denom` switch is not the culprit.
Both artifacts retain mode-23 PWL exponentiation, log2-P quantization, and
shiftless softmax; the accuracy ladder must step outside that shared policy.

## Next numerical experiments

1. Compile read-only forward variants that preserve NVFP4 QK and MXFP4 PV but
   progressively restore native exponentiation and an exact represented
   denominator.  This isolates probability approximation from FP4 tensor
   quantization.
2. In backward, compare exact exp2 (`period=0`) with degree-2 and the current
   degree-1/period-2 schedule on identical FP8 QKV/dO operands.
3. Check forward/backward probability consistency by using the same
   represented denominator or reusable probability metadata.  The 1.54-2.00x
   gradient norm inflation should be fixed before stochastic rounding or RHT
   experiments.
4. Only after the operator-level gradients have the correct scale and useful
   cosine should stochastic rounding, RHT, or adaptive projection scaling be
   evaluated in training.

## Artifacts

- Full alternating benchmark:
  `llama12b_e2e_interleaved_scaled_operand_native_lse_s4096_20260817.json`
- Boundary profile before fusion:
  `llama12b_e2e_boundaries_post_handoff_s4096_20260817.json`
- Boundary profile after fusion:
  `llama12b_e2e_boundaries_scaled_operand_native_lse_s4096_20260817.json`
- Numerical decomposition:
  `llama12b_numerical_decomposition_s4096_20260817.json`
- Numerical diagnostic:
  `../tk_fa4/lowp_fa4_bwd/diagnose_llama12b_numerics.py`

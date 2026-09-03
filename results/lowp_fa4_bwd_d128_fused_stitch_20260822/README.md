# D128 shared low-precision backward optimization

This result set validates the shared causal FA4 backward used by both
`NVFP4-QK + MXFP4-PV` and `NVFP4-QK + FP8-PV` for the Llama 3.1 8B shape on
one NVIDIA GB200 (SM100). The only route-specific component is the forward
attention kernel. Runtime construction now fails closed unless both routes
retain the same physical backward runner and all of its retained state.

## Current D128 2D-weight and dynamic-Q/K update

Learned QKV and output-projection weights now have a fail-closed D128
contract: they use one shared E4M3 scale for every true 16x16 NVFP4 weight
block. Activations retain row-by-K16 scaling. Independently preparing a
physical learned-weight transpose decodes bit-for-bit as the transpose of the
forward QDQ matrix; this was checked through the compiled CUDA extension at
S=4096. The backward Q/K/V publication remains an independent accumulator-
E4M3 representation and is byte-identical between the MXFP4-PV and FP8-PV
routes.

The causal BF16 reference was also corrected for Llama D128's split-half RoPE
coordinates. The fused projection pair-interleaves Q/K rows, so the reference
must convert split-half features to adjacent pairs before applying the same
rotation. D64 behavior is unchanged.

| Current S=4096 check | Result |
| --- | ---: |
| Learned-weight scale geometry | **true 16x16** |
| Forward QDQ transpose versus independently prepared physical transpose | **bitwise exact** |
| Fused projection versus dense QDQ cosine, Q/K/V | **0.999996 / 0.999996 / 0.999996** |
| Dynamic row-by-K16 publication cosine, Q/K | **0.995472 / 0.995472** |
| Fixed head-scale publication cosine, Q/K | 0.991085 / 0.992752 |
| Corrected causal FP8-PV output cosine versus BF16 | **0.931081** |
| Corrected causal MXFP4-PV output cosine versus BF16 | **0.915439** |
| Isolated causal MXFP4-PV versus FP8-PV | 124.768 vs 149.376 us (**1.197x**) |

A separate dense-QDQ oracle used Dolma tokens at random initialization to
screen projection quantizers. Adaptive 4-to-6 MSE activation scaling raised
the full QKV projection cosine from 0.989263 for static-6/448 to 0.990016;
applying the adaptive search to the already-2D weights changed weight-only
cosine only from 0.993794 to 0.993795. MXFP4 round-to-even activations with
dense 2D MX weights reached 0.986248, versus 0.981872 for ceil/ceil MXFP4.
This is an unfused accuracy oracle, not kernel timing or convergence evidence;
it points to activation-scale selection rather than abandoning 2D weights.

The claim-grade one-layer same-model crossover used 160 forward samples per
route, 20 complement-paired macroblocks, 160 warmups per route, and a separate
80-macroblock blocked full-step test. MXFP4-PV was faster in all 20 forward
macroblocks: 5.5852 versus 5.6107 ms, with a 0.027310 ms paired trimmed saving
and a 90% bootstrap interval of [0.023555, 0.030968] ms. This is a **1.00489x
full-layer forward speedup**. The much larger 1.197x isolated-attention speedup
only removes about 0.025 ms from a roughly 5.6 ms layer forward, so these two
measurements are quantitatively consistent.

The blocked backward timing spread was 0.1003%, inside the 0.75% equality
contract, while the two routes retained one physically shared backward
implementation. The full-step interval crossed zero, so the valid conclusion
is: **MX forward faster, backward equal, no statistically resolved step win**.

## Fresh full-depth Dolma smoke gate

A clean sequential bracket from commit `28ea72f` then exercised all 32 Llama
3.1 8B layers on Dolma tokens at B=1, S=4096. Each route ran in its own process
on one GB200 to keep peak memory near 84 GiB; MX was repeated before and after
the BF16 and FP8 arms. Both low-precision routes used true 16x16 learned-weight
scales, dynamic row-by-K16 Q/K publication, and the same verified accumulator-
E4M3 backward contract. All eight training batches per arm were finite.

| Full-depth route | Forward (ms) | Backward (ms) | Step (ms) | Initial / final validation loss |
| --- | ---: | ---: | ---: | ---: |
| MXFP4-PV A | 69.864 | 136.862 | 252.368 | 12.5457 / 11.2529 |
| MXFP4-PV B | 70.003 | 136.004 | 252.679 | 12.5457 / 11.2987 |
| FP8-PV exact | 70.636 | 135.405 | 252.556 | 12.5452 / 11.2256 |
| BF16 CuTe | 75.708 | 141.186 | 264.630 | 12.5278 / 11.4012 |

The average of the two independent MX medians is 69.933 ms forward, making MX
**1.01005x faster than FP8 forward** and saving 0.703 ms. Its average step is
252.523 ms versus 252.556 ms for FP8, an unresolved 0.033 ms difference. This
small end-to-end delta is expected: the route-specific saving is only about
0.28% of the full training step. Both low-precision routes are about 1.048x
faster than BF16 end to end in this smoke test.

Both low-precision routes still take the same timed projection-publication
allocation fallback. This keeps the comparison matched, but it is common
overhead that dilutes the attention-only percentage gain; preallocating or
fusing those publication buffers remains an end-to-end optimization target.

The MX A/B backward spread is 0.628%, and the FP8 median falls outside that
pair by about 1 ms even though all three executions use the same verified
backward contract. These are independent-process clock samples, so this is not
evidence for a route-specific backward. The same-process claim-grade gate
above remains the valid equality measurement.

The MX A/B final-validation spread is 0.0458, compared with a 0.0502
MX-average-minus-FP8 delta. With only eight training batches and two validation
batches, neither that ordering nor the lower short-run losses versus BF16 is a
convergence claim. The gate establishes dispatch, finite execution, matched
data/state, provenance, and the expected forward performance ordering.

Current compiled artifacts:

- Projection/backward extension SHA-256: `b549d397b5c28576b750289b8183dae5292cb752e191b185b14bb55b010371e2`.
- Non-folded MXFP4-PV forward SHA-256: `2420a5640c694881ff38552b4ccd6a840405d242087c576ea75d5df584703344`.
- Exact FP8-PV forward SHA-256: `f9f67026148c355b3b90026861fc25f3b6b7edccf2d6254703d5ddc4164c3d9e`.

## Earlier shared-backward work

- Fused D128 inverse RoPE, Q/K/V gradient scaling, and BF16
  `[dQ | dK | dV]` publication into one CUDA kernel.
- Removed eager contiguous transpose copies before the D128 QKV and output
  weight-gradient GEMMs; the GEMMs consume transpose views directly.
- Replaced comparator-side backward-object repair with a verifier. A mismatch
  in any of 25 object/allocation identities is now an error.
- Rechecked physical identity at construction, after compilation, and after
  real-token training in the training harness.

## Earlier retained results

All timings below are medians. The model shape is batch 1, sequence 4096,
hidden size 4096, 32 query heads, 8 KV heads, and head dimension 128.

| Test | Result |
| --- | ---: |
| Fused publication versus functional PyTorch reference | 61.888 vs 475.616 us (**7.685x**) |
| Fused publication numerical match | **bitwise exact**, max/mean absolute error 0 |
| Full 32-layer MX forward versus BF16 | 66.594 vs 74.285 ms (**1.115x**) |
| Full 32-layer shared low-precision backward versus BF16 | 136.417 vs 140.548 ms (**1.030x**) |
| Full 32-layer MX step versus BF16 | 249.892 vs 262.134 ms (**1.049x**) |
| One-layer MX versus FP8 attention stage | 117.504 vs 143.840 us (**1.225x**) |
| One-layer MX versus FP8 full forward | 5.593 vs 5.670 ms (**1.014x**) |
| Full 32-layer MX versus FP8 full forward | 65.618 vs 66.804 ms (**1.018x**) |
| Full 32-layer MX versus FP8 observed step | 254.470 vs 257.624 ms (**1.012x**) |

The one-layer matched run measured backward at 10.899 ms for the MX block and
10.882 ms for the FP8 block, a 0.155% spread. The full-depth run measured
132.928 and 134.823 ms, but the FP8 backward improved 1.773% from its first to
last block while its implementation remained physically identical. These are
sequential timing samples of the same backward, not evidence for two backward
paths or a route-specific backward speed difference.

The recorded matched artifacts predate the final five-check unpacked-RoPE
extension and contain all 20 then-current physical-identity checks as true.
Current execution additionally requires the RoPE tuple and its cosine/sine
tensors to share their Python objects and CUDA data pointers. Together, the 25
checks cover the runner, compiled callable, kernel, control module, unpacked
and packed RoPE, gradient scale, workspace, dQ/dK/dV outputs, and dK/dV
partials.

## Artifacts

- `llama31_8b_s4096_projection_boundaries_perblock_qk.json`: real-CUDA
  projection QDQ, true-2D transpose invariance, publication isolation, and
  timing checks.
- `llama31_8b_s4096_causal_forward_perblock_qk_consumer_audited.json`:
  corrected causal dynamic-Q/K matrix, decoded-consumer audits, leakage test,
  and isolated timing.
- `llama31_8b_s4096_causal_forward_fixed_qk_consumer_audited.json`: matched
  fixed-head Q/K ablation.
- `llama31_8b_l1_dynamic_qk_2d_matched_mx_fp8_claim_grade.json`: sufficient
  one-layer same-model forward and blocked full-step crossover.
- `llama31_8b_l0_dolma_projection_qdq_screen.json`: Dolma-token dense-QDQ
  oracle for static-6, adaptive 4-to-6, and MXFP4 projection policies.
- `llama31_8b_l32_dolma_2d_dynamic_sequential_smoke.json`: fresh full-depth
  sequential Dolma bracket with two MX arms, FP8, BF16, exact identities, and
  explicitly limited performance/numerics conclusions.
- `llama31_8b_l32_dolma_2d_dynamic_merged_mx_a.json` and
  `llama31_8b_l32_dolma_2d_dynamic_merged_mx_b.json`: complete matched merged
  records, including per-step timings/losses and dispatch provenance, for both
  sides of the MX bracket.
- `d128_fused_stitch_micro.json`: isolated bitwise and latency validation.
- `llama31_8b_l1_matched_mx_fp8.json`: one-layer ABBA crossover with stage
  profiles and physical-identity evidence.
- `llama31_8b_l32_bf16_mx.json`: full-depth BF16 versus MX performance run.
- `llama31_8b_l32_matched_mx_fp8.json`: full-depth matched MX/FP8 crossover.

The historical compiled backward extension used by the earlier retained GPU
results has SHA-256
`c0e5ce51f69e7c4da3fb29c212fdf19716d5a172d5414695d923c8b83170f514`.

## Scope and unresolved validation

The result set now includes a real-token full-depth Dolma smoke gate, while the
earlier performance runs are synthetic repeated-batch experiments. Together
they verify kernel numerics, true 2D learned-weight scaling, causal forward
advantage, physical backward identity, matched dispatch, and finite eight-step
execution. They do **not** establish loss parity or convergence. The current
one-layer initial-logit cosines versus BF16 are 0.91056 for FP8-PV and 0.89305
for MXFP4-PV; those and the short Dolma losses are diagnostics, not a
training-quality claim. A substantially longer matched Dolma run remains the
next unresolved validation.

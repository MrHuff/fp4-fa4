# Scientific state of the FP4 FlashAttention project

Status date: 2026-09-03

This file is the technical handoff for continuing the research. It separates
measured facts from interpretations and open questions. Internal `v###` names
remain useful for source provenance, but the human-readable names below should
be used in new reports and discussions.

The machine-readable implementation map is `release/routes.json`. Exact source
locations are in `release/KERNEL_MAP.md`; experiment and data boundaries are in
`release/EXPERIMENT_MATRIX.md` and `release/DATA_PROVENANCE.md`.

## Working methods

### Non-causal Direct-P forward

The earlier forward-only study directly quantizes the softmax probabilities
and executes the probability/value product in low precision. Its exact source
epoch is preserved separately under
`reproduction/snapshots/forward_cfc06dad/`; do not silently rerun those paper
rows with the later causal source tree. The study includes NVFP4 and MXFP4
probability routes, HAO's Blackwell implementation, FP8 controls, and
fixed-input downstream evaluations.

### Causal FP8-P/V training candidate

The current training candidate uses:

- NVFP4 Q/K in the attention score product, with row-by-K16 two-dimensional
  scales;
- E4M3 FP8 for the probability/value product;
- NVFP4 learned Q/K/V/O projections; and
- a backward pass that reconstructs scores from saved NVFP4 Q/K, their scales,
  and log-sum-exp, then uses represented E4M3 Q/K/V and represented E5M2
  output gradients.

The row-by-K16 Q/K scale layout replaced a fixed-head D128 scale. The fixed
scale caused immediate 8B loss and gradient failure; the row-K16 change reduced
the observed peak pre-clip norm in the short gate from roughly 20,000 to
161--184 and restored BF16-like short-run behavior. This is short-gate
evidence, not a completed convergence result.

The production source family is internally named v509. The earlier v501
represented-FP8 route established a critical ABI rule:

```text
lstat = 8 - LSE * log2(e)
```

Omitting the `+8` reduces reconstructed probability by 256. A separate
one-over-256 gradient epilogue is also required. These two factors have
different meanings and must not be merged or cancelled casually.

At the exact B2/S4096 shape, the v501 route dispatches the v490 owner-4
schedule. dK and dV are unique-writer outputs; only dQ requires clearing.

### Causal MXFP4-P/V diagnostic

The safe MX route uses E8M0 block-32 scales for the probability/value product.
It shares the same backward binary as FP8-P/V; that binary reconstructs scores
from saved quantized Q/K, scales, and log-sum-exp. A faster shift-16 forward
route remains disabled because one gate produced 370,900 nonfinite outputs.

MXFP4-P/V is useful for diagnosis, not a supported convergence recipe. Its
isolated P/V kernel can be faster than FP8-P/V, but publication and integration
overheads reduce the complete-step difference to measurement noise. More
importantly, both historical projection arms and the recorded B4 trajectory
diverged. The root cause is unresolved; finite outputs alone are not a
promotion criterion.

## Backward design and retained alternatives

The retained backward avoids the known-hanging direct D128 CuTe two-CTA path.
It reconstructs the causal scores inside a native ThunderKittens schedule from
saved NVFP4 Q/K, their row- and column-oriented scales, and log-sum-exp. Its
gradient products consume format-specific tensors published by the projection
and attention forward paths.

The much earlier monolithic backward prototypes from `cfc06dad` are preserved
under `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_bwd/`. They are
source-continuity material, not evidence for the final method: no portable
receipt survives for the family as a whole, and none is connected to the
TorchTitan route.

The retained alternatives answer narrower questions:

- **Represented-FP8 precursor (v501):** established the scale ABI and the
  owner-4 B2 schedule.
- **Common-row MX-V backward (v503):** correct against its represented oracle
  and slightly cheaper in the composed producer/backward slice, but tied
  within noise at complete-step scope.
- **Direct Common-Row MX publication (v506):** numerically correct but measured
  at 237.568 microseconds against a 209-microsecond gate, so it remains off.
- **Four-Anchor Shared-Tile MX-V backward (v507):** correct, but too slow for the
  training route.
- **E4M3-dO backward precursor (v508):** superseded because E4M3 underflowed
  the output gradient too aggressively; v509 uses E5M2.
- **Dense-score E4M3/E5M2 diagnostic (v510):** preserved under
  `reproduction/snapshots/v510_aa021504/`. The implementation was not promoted.
  No portable receipt survives that proves its precise rejection reason, so
  none is asserted here.
- **Direct CuTe FP4-QK:** preserved in the pinned FlashAttention fork but
  fail-closed for D128 because its native two-CTA schedule can hang.

The practical backward is still one-CTA-per-SM and schedule/dependency limited.
Low-precision operands reduce tensor-core work, but they do not automatically
remove synchronization, publication, clearing, or host-launch costs.

## Verified measurements

### Isolated D128 backward and attention module

For B1/S4096/Hq32/Hkv8/D128 on GB200, 101 rotating-order CUDA-event samples
gave these medians:

| Boundary | BF16 or control | Backward from saved Q/K and scales | Ratio |
| --- | ---: | ---: | ---: |
| Backward core | 0.500704 ms | 0.356416 ms | 1.405x |
| Publisher plus backward | 0.500704 ms | 0.507584 ms | 0.986x |
| Projection-inclusive module backward | 1.572192 ms | 1.397440 ms | 1.125x |
| Projection-inclusive module forward + backward | 2.656160 ms | 2.132800 ms | 1.245x |

The isolated BF16 denominator decodes the represented E4M3 Q/K/V and E5M2
dO into BF16. It is shape matched, but does not consume byte-identical saved
score inputs. The low-precision backward outputs were finite and nonzero, and
exact zero-dO produced zero dQ/dK/dV. See
`results/fp4_fa4_technical_report_v2_20260819/receipts/causal_d128_report_boundaries_20260901.json`.

### 8B complete-update timing

The saturated single-GPU B4 comparison includes learned projection
quantization, attention forward and backward, optimizer, and the surrounding
8.03B model update:

| Route | Step median | Paired BF16 median | Speedup |
| --- | ---: | ---: | ---: |
| NVFP4 projections + FP8-P/V | 751.722 ms | 854.516 ms | 1.1367x |
| NVFP4 projections + MXFP4-P/V | 751.597 ms | 857.226 ms | 1.1405x |

FP8-P/V and MXFP4-P/V differ by only 0.125 ms in their direct low-precision
step medians and must be treated as tied. Initial-logit cosine to BF16 was low
in this performance harness (about 0.43 for FP8 and 0.37 for MX), so these are
performance measurements, not convergence evidence. The B1-to-B4 trend shows
why local-batch saturation matters: measured complete-step speedup rose from
about 1.09x at B1 to about 1.14x at B4. See
`results/tk_fa4_8b_batch_scaling_20260901/e2e_batch_scaling_summary.json`.

An older B2 native-v501 bracket measured 489.821 ms for BF16, 434.014 ms for
FP8-P/V plus v501, 435.992 ms for MXFP4-P/V plus v501, and 433.539 ms for MX
plus v503. These values likewise support the conclusion that the complete-step
FP8 and MX routes are tied, not that shared-MX is a throughput win.

### Distributed 8B training snapshot

The matched recipe used a Llama-3.1-style 8.03B model, S4096, local batch 4,
64 GPUs, physical global batch 256, gradient accumulation 4, and effective
global batch 1024. Each update consumed 4,194,304 tokens. It used fused
BF16-stochastic-rounding AdamW, BF16 parameters and moments, no FP32 master
copy, and standard BF16 cross entropy compiled with `torch.compile`.

One BF16 trajectory was observed through update 16,425 and one NVFP4-projection
FP8-P/V trajectory through update 18,150. Both descended stably over the
recorded interval; FP8 retained a measurable loss gap. These are single interim
trajectories, not repeated trials or completed 100-billion-token convergence
runs.

The recorded B4 MX arm used E4M3 learned projections, departed near update 325,
and was stopped at update 2,550 after clear loss and pre-clip-gradient
instability. Historical NVFP4-projection MX arms also diverged. Do not use the
E4M3 B4 diagnostic to claim the exact behavior of a not-yet-run matched
NVFP4-projection MX B4 trajectory, but do retain the cross-arm evidence that MX
is currently unsafe for training.

The clean artifact and TorchTitan configuration path exposes all four
learned-projection/PV combinations: E4M3 or NVFP4 learned Q/K/V/O projections,
crossed with FP8 or MXFP4 P/V. These route manifests reuse the same
format-matched attention binaries but preserve projection format as part of
the authenticated route identity. This makes the historical four-arm study
re-runnable without implying that its diagnostic arms are supported recipes.

The sanitized aggregate receipt is
`results/fp4_fa4_technical_report_v2_20260819/receipts/llama8b_b4_matched_snapshot_20260902T1358Z.json`.
Its raw hosted histories are not committed, so it is a frozen evidence record,
not a source from which the service data can be reacquired.

## Hardware interpretation

Blackwell low-precision matrix throughput is only useful when the schedule can
keep the tensor cores fed. FlashAttention also needs on-chip storage for score,
probability, scale, reduction, and output state. At these shapes, tensor-memory
capacity, tile/CTA choice, publication, and dependencies restrict overlap.
Profiling is consistent with this explanation, but the repository does not
contain a general proof that tensor-memory capacity alone causes every observed
bubble.

D64 and D128 therefore need different schedules. D64 is not simply a smaller
D128: tile size, CTA ownership, and available overlap change. The full D64
kernel lineage, including the v416 production candidate, is preserved. The
schema-v3 D64/B16 builder and TorchTitan route are now wired, but their
clean-clone GPU, distributed save/fresh-resume, and long-horizon convergence
gates remain unrun.

## Unresolved work

The next researcher should start here rather than reopening already rejected
paths without new evidence:

1. Diagnose MXFP4-P/V divergence with matched data, initialization, projection
   format, per-layer activation/gradient statistics, and a checkpoint just
   before departure. Separate probability range/anchor error from backward
   representation and optimizer effects.
2. Validate the current repository from a fresh recursive clone on GB200:
   clean build, B1/B2/B4 numerical gates, exact-zero-dO, liveness, saturated
   timing, DDP smoke, and checkpoint save/fresh-load/resume.
3. Finish a fully matched BF16 versus FP8-P/V long-horizon run on a published,
   immutable token stream before making a convergence claim.
4. Validate the D64 build/manifest/training path from a clean clone, including
   the B16 forward factorial, v416 gates, DDP16 save/fresh-resume, and a matched
   public-data trajectory.
5. Reacquire missing natural-input, B300, Wan, ViT-MAE, and timing captures with
   immutable asset identities and new receipts where the original bytes cannot
   be recovered.
6. Build a redistributable, digest-pinned Blackwell environment. The recorded
   NVIDIA PyTorch and CUTLASS DSL packages are not all obtainable from ordinary
   public indexes.

Do not promote shift-16 MX forward, v506, v507, v510, or direct D128 CuTe
FP4-QK without new evidence that passes their declared correctness, liveness,
and timing gates. Do not advertise v503/shared-MX as a throughput improvement.

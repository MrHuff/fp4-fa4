# Optimized 8B/D128 causal FA4 Dolma bracket (2026-08-22)

This directory preserves the post-optimization Llama-3.1 8B end-to-end
training bracket for causal BF16 FA4, NVFP4-QK/MXFP4-PV, and
NVFP4-QK/FP8-PV. It is a 256-update scale and numerical gate, not a 2K-step
convergence result.

## Result

The optimized low-precision backward now clears the BF16 backward baseline at
8B/D128. MXFP4-PV is also faster than FP8-PV in forward and end-to-end step
time. MX-A and MX-B bookend FP8 in execution order to expose run-order
variation; their median is the central MX estimate.

| Route | Forward (ms) | Backward (ms) | Optimizer (ms) | Step (ms) | Tokens/s | Final validation loss |
|---|---:|---:|---:|---:|---:|---:|
| BF16 FA4 | 73.9851 | 141.0279 | 46.6610 | 261.8312 | 15,643.7 | 7.436397 |
| MXFP4-PV A | 66.3812 | 135.4932 | 45.6673 | 247.9781 | 16,517.6 | 8.044562 |
| FP8-PV | 67.4058 | 135.2809 | 45.6048 | 248.5233 | 16,481.4 | 7.783532 |
| MXFP4-PV B | 66.8188 | 135.5791 | 45.7620 | 248.2227 | 16,501.3 | 8.119128 |
| MX bracket median | 66.6000 | 135.5362 | 45.7147 | 248.1004 | 16,509.5 | n/a |

Normalized comparisons from the MX bracket median:

- MX versus BF16: `1.11089x` forward, `1.04052x` backward, and `1.05534x`
  step speedup.
- FP8 versus BF16: `1.09761x` forward, `1.04248x` backward, and `1.05355x`
  step speedup.
- MX versus FP8: `1.01210x` forward and `1.00170x` step speedup. The observed
  backward ratio is `0.99812x`; the 0.188% difference is timing noise around
  the matched shared implementation, not a distinct MX backward path.
- MX A/B spread is 0.659% forward, 0.063% backward, and 0.099% step.

The companion shared-process full-depth profile makes the backward identity
claim stronger than timing similarity: all 20 runner, compiled callable,
kernel, control, workspace, dQ/dK/dV, partial-buffer, gradient-scale, and
packed-RoPE object/data-pointer checks are true. Across 32 timed samples per
route it measures MX/FP8 forward at 65.618/66.804 ms. Its residual backward
timing variation cannot reflect a distinct route-specific implementation;
forward-dependent state and ordinary measurement drift remain possible.

## Numerical status

All 1,024 training updates are finite. Final eight-sequence validation loss is
4.668% above BF16 for FP8 and 8.178%/9.181% above BF16 for MX-A/MX-B. MX-A and
MX-B have identical initial validation loss (`12.573220`) but finish at
`8.044562` and `8.119128`; at round 191 they are `8.032875` and `8.715221`.
The speed target is therefore verified, while D128 low-precision numerical
reproducibility and convergence remain unresolved.

Validation history at rounds `[-1, 63, 127, 191, 255]`:

| Route | Initial | 63 | 127 | 191 | 255 |
|---|---:|---:|---:|---:|---:|
| BF16 FA4 | 12.561120 | 8.337996 | 8.142539 | 7.821194 | 7.436397 |
| MXFP4-PV A | 12.573220 | 8.327132 | 8.312146 | 8.032875 | 8.044562 |
| FP8-PV | 12.577995 | 8.354431 | 8.291999 | 7.787350 | 7.783532 |
| MXFP4-PV B | 12.573220 | 8.197085 | 8.156860 | 8.715221 | 8.119128 |

## Verified protocol and provenance

- Source commit: `d61663fe6976faecaec587886f2091487c01bb7e`, with
  zero tracked source diff in every route result.
- Model: Llama-3.1 8B, 8,030,261,248 parameters, 32 layers, batch 1,
  sequence 4096, H32/KV8/D128.
- Hardware: one visible NVIDIA GB200 (SM100), UUID
  `61ebc9a2-4efb-2335-6ded-8591f9acef8c`.
- Training: 256 updates per independent process, seed 20260818, learning rate
  1e-4, no gradient clipping, eight validation sequences every 64 updates.
- Execution order: MX-A, BF16, FP8, MX-B. Each arm ran in a fresh process on
  the same local GPU and CPU 0.
- Dataset: first 512 physical rows of Dolma3 Longmino `len-8-16k`, yielding
  446 unique documents after exact-duplicate removal. SHA256
  `860b33924dffd53f4c20b80abbcee96e1bf09c3c313290c15ea3a6ee418269ce`.
- Tokenizer SHA256:
  `76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa`.
- Backward/projection artifact SHA256:
  `c0e5ce51f69e7c4da3fb29c212fdf19716d5a172d5414695d923c8b83170f514`.
- MX forward artifact SHA256:
  `63c43e5cde3af4e9cde82aad1d667221a7ae77dd6271eb653fd202509707b77a`.
- FP8 forward artifact SHA256:
  `f9f67026148c355b3b90026861fc25f3b6b7edccf2d6254703d5ddc4164c3d9e`.

Every route contains exactly 256 finite and timing-eligible records with no
diagnostic timing fallback. Both strict merger outputs report `matched=true`,
including matched source, data, initialization probe, hardware, forward
provenance, and low-precision backward contract.

The first attempted launch exposed a constructor regression before any
training update: shared RoPE binding had accidentally been applied to the
BF16 attention constructor. Commit `d61663f` moves that binding exclusively
to the low-precision runtime and adds a source regression test. The corrected
suite passed 285 tests before this bracket was launched.

## Files

- `raw/{bf16,mx-a,fp8,mx-b}.json`: complete independent-route records.
- `raw/merged-mx-{a,b}.json`: strict BF16/MX/FP8 matched-contract mergers.
- `shared_backward_identity_profile.json`: balanced full-depth shared-process
  MX/FP8 timing and physical backward-identity checks. This profile does not
  embed source or hardware hashes, so its linkage is documentary; the raw
  training routes and strict mergers contain the self-contained provenance.
- `run_8b_dolma_bracket.sh`: exact local launcher used for the completed run;
  its absolute artifact paths are preserved as provenance.
- `SHA256SUMS`: portable checksums for the evidence files.

Verify from the repository root with:

```bash
sha256sum --check results/dolma3_8b_d128_e2e_bracket_20260822/SHA256SUMS
```

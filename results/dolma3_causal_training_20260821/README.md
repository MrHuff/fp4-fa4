# Literal-Dolma3 causal training scale gate (2026-08-21)

This directory preserves the 256-update, sequence-4096 causal FA4 comparison
used to test whether NVFP4-QK/MXFP4-PV gains relative to NVFP4-QK/FP8-PV grow
from a 1.2B/D64 Llama configuration to an 8B/D128 configuration. It is a
short model-scale gate, not a 2K-step convergence result.

## Result

The measurements support the directional scaling hypothesis. MXFP4-PV's
advantage over FP8-PV grows from 0.179% to 1.362% in forward time and from
0.057% to 0.332% in step time. MX and FP8 backward times remain effectively
equal at both model sizes.

The larger-model end-to-end result is not yet a BF16 speedup. At D128 the
shared low-precision backward path is about 8.3% slower than BF16 backward,
which cancels the MX forward saving. The 8B low-precision loss drift is also
material and must be fixed before treating the route as training-ready.

| Model / route | Forward (ms) | Backward (ms) | Step (ms) | Step vs BF16 | Final validation loss |
|---|---:|---:|---:|---:|---:|
| 1.2B/D64 BF16 | 27.7658 | 34.3375 | 69.1126 | 1.0000x | 7.296728 |
| 1.2B/D64 MXFP4-PV | 18.5625 | 32.3164 | 57.5120 | 1.2017x | 7.312903 |
| 1.2B/D64 FP8-PV | 18.5958 | 32.3122 | 57.5450 | 1.2010x | 7.298012 |
| 8B/D128 BF16 | 73.1216 | 138.3854 | 257.9667 | 1.0000x | 7.417654 |
| 8B/D128 MXFP4-PV A | 65.2033 | 149.8347 | 259.0672 | 0.9958x | 8.130195 |
| 8B/D128 FP8-PV | 66.5860 | 149.9079 | 260.4810 | 0.9903x | 7.930487 |
| 8B/D128 MXFP4-PV B | 66.1786 | 149.9197 | 260.1726 | 0.9915x | 8.282399 |

For the 8B central estimate, the MX value is the median of its A/B bracket:
65.6909 ms forward, 149.8772 ms backward, and 259.6199 ms per step. FP8/MX
is 1.01362x forward, 1.00020x backward, and 1.00332x per step. Both bracket
arms beat FP8, although the 1.50% MX forward spread means the exact magnitude
still has run-order uncertainty.

## Verified protocol and provenance

- Source commit: `cd59dda37ebf22e0d77b9c9d6851ec164b86e3af`, with no tracked source diff in every result.
- Hardware: one visible NVIDIA GB200 (SM100), UUID `01355792-ef83-14f6-793b-b31a141c113a`, shared by every arm.
- Dataset: literal Dolma3 Longmino `len-8-16k`, first 512 physical MDS rows without reshuffling. The materialized JSONL is 21,911,537 bytes with SHA256 `860b33924dffd53f4c20b80abbcee96e1bf09c3c313290c15ea3a6ee418269ce`.
- Loader result: 512 source rows, zero empty rows, 66 exact duplicate rows removed, and 446 unique documents. This is one Longmino bucket, not the full ten-stream Dolma mixture.
- Tokenizer SHA256: `76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa`.
- Shared optimization setup: seed 20260818, learning rate 1e-4, no gradient clipping, batch 1, sequence 4096, 256 updates, and validation on eight sequences at rounds `[-1, 63, 127, 191, 255]`.
- Both model sizes consume the same 1,048,832 training tokens. The validation sample is only 32,776 tokens from three consumed documents.
- The 1.2B run has 1,235,814,400 parameters, 16 layers, D64, and interleaves/rotates all three routes in one process.
- The 8B run has 8,030,261,248 parameters, 32 layers, D128, and executes independent processes sequentially as BF16, MX-A, FP8, MX-B. The stored merger artifacts prove matched source, data, initialization, forward dispatch, and MX/FP8 backward contracts.

All route records are finite, contain exactly 256 timing-eligible updates, and
use no diagnostic timing fallback. The raw-file checksums and collector are
the portable verification boundary for this directory.

## Important status distinction

The 8B Kubernetes job completed with exit status 0. The 1.2B training itself
completed all 3 x 256 finite updates and synchronized its complete result, but
the job exited 1 in the post-run assertion because the submitted manifest
expected `corpus_documents=512`. The trainer correctly reports 446 unique
documents after removing 66 exact duplicates. The independent collector
validates the actual `512 -> 446` corpus contract. The corrected, unsubmitted
`20260821b` reproduction manifest fixes that assertion, avoids ListBucket
dependence, and keeps `set +e` confined to the exit trap.

## Interpretation boundary

It is verified that the MX-over-FP8 advantage is larger in the 8B run. It is
an inference—not a causal proof—that model size caused the increase: D64 to
D128, 16 to 32 layers, E4M3 to NVFP4 QKV projection publication, and
shared-process to isolated-process execution all change together.

The 1.2B losses remain close to BF16: final FP8 is +0.0176% and MX is +0.2217%.
At 8B, final FP8 is +6.91% and MX-A/B are +9.61%/+11.66% versus BF16. The two
MX arms have identical initial validation loss but diverge during training,
so D128 low-precision numerical reproducibility is a blocking follow-up, not
ordinary timing noise.

## Reproduction files

- `build_summary.py` validates the packaged raw evidence and regenerates `comparison_summary.json`.
- `raw/1p2b/llama1p2b-matched.json` is the complete shared-process 1.2B result.
- `raw/8b/{bf16,mx-a,fp8,mx-b}.json` are the four isolated 8B arms.
- `raw/8b/merged-mx-{a,b}.json` are strict matched-pair merger outputs.
- `manifests/submitted-fa4-dolma3-1p2b-matched256-cd59dda-20260821a.yaml` is the exact submitted 1.2B manifest, including its faulty final document-count assertion.
- `manifests/fa4-dolma3-1p2b-matched256-cd59dda-20260821b.yaml` is the corrected reproduction manifest; it was not submitted for the preserved result.
- `manifests/fa4-dolma3-8b-isolated256-cd59dda-20260821d.yaml` is the exact successful 8B manifest.

Regenerate and validate from the repository root with:

```bash
python results/dolma3_causal_training_20260821/build_summary.py
sha256sum --check results/dolma3_causal_training_20260821/SHA256SUMS
```

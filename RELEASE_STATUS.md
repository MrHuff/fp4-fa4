# FP4 FlashAttention release status

Status date: 2026-09-03

## Current state

This public export is the continuation and reproduction snapshot. It contains
the complete recovered first-party forward and backward
kernel trees, the historical non-causal source used by the paper, pinned
third-party implementations, result records, and the polished manuscript
snapshot. A portable TorchTitan adapter, fail-closed artifact loader,
clean-build wrapper, config renderer, and measurement planner are present
directly in this tree. The CPU contract suite covers the adapter,
configuration and artifact receipts, exact dataset and tokenizer identities,
checkpoint-aligned prefetch, the recovered SFU-B1 converter chain, and the
fused optimizer state.

The source was written from a single parentless root and audited from a public,
unauthenticated clone created without local object sharing. Inherited
continuous-integration workflows are absent from the exported tree.
On 2026-09-03, the authorized owner confirmed consent to publish the project
source and the paper's fonts and logos. Public identifier and asset hygiene is
complete and documented in `release/PUBLIC_SANITIZATION.md`. Project-specific
source is licensed under Apache-2.0, with
`Copyright (c) 2026 Graphcore Ltd.` recorded in `NOTICE`.

The retained public fresh-clone receipt covers source commit
`5926d20188ec7a8a033e4efc7075f4c40325e3e8`. That clone passed the checked
source inventory, strengthened release verifier, pinned root and nested
gitlink audit, all CPU tests, and complete offline paper reproduction; it
remained clean and reproduced the authenticated 56-page PDF. Both standalone
CUTLASS projection controls also compiled from the clone for `sm_100a`.

The release is **recovered-kernel-source-complete but not fully
GPU-validated**. No claim is made that a fresh clone builds every extension,
launches every historical model profile, or reproduces every GPU number. The
distinction matters: source and offline artifact reproduction are verified for
the audited public commit; the full adapter and Blackwell route matrix still
require clean-checkout validation on target hardware.
These scientific gaps limit validation claims, but they do not block
publication of the source when they remain clearly labelled. The
machine-readable public audit receipt is
`release/audits/public_source_closure_5926d201_20260903.json`.

“Source-complete” here means that the recovered FA4 kernel and experiment
lineages are retained. It does not mean every historical private service can
be recreated: exact old data order, checkpoints, hosted logs, and some raw
captures remain unavailable. `CONTINUATION.md` and
`release/SCIENTIFIC_STATE.md` are the handoff authorities for that boundary.

## Current release checks

The audited public source commit passed 1,669 CPU tests with four optional or
hardware-dependent skips. Its release verifier passed and its checked source
inventory contains 4,138 records. The audit-record child adds one receipt, so
its inventory contains 4,139 records. The current navigation-only descendant
adds three reader-facing indexes, bringing the checked inventory to 4,142
records without changing kernel source. A no-local clean clone rebuilds the
56-page paper and both the arXiv and Overleaf source packages. The LaTeX log
has no undefined references, undefined citations, or overfull boxes; all
embedded fonts are non-Type-3. These CPU and document checks do not replace
the target-hardware GPU gates below.

## Authoritative source pins

| Component | Commit | State here |
| --- | --- | --- |
| TorchTitan base | `20b3de7585696c327bd5aa9f9627f0300abdbf9d` | Materialized and authenticated as the upstream source pin |
| Historical training integration | `e7db209b0c7017c415fdd66e04e85f96ae24f276` | Ported into the TorchTitan experiment boundary |
| Final causal kernels | `4590537f1479e1a7e847f2783e9ab7aa7f11b975` | Present at root |
| Historical non-causal forward | `cfc06dadf684279f657ab66254a3a074be4ee3a9` | Present as a separate source snapshot |
| HAO comparator | `9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247` plus recorded patch | Vendored |
| Technical report and results | `4c394504998f653aa702d030c5f98864dcf34c75` | Present under `results` |
| Final manuscript narrative | `d96ca5599d46ee2ed463adaf40cde974058a0173` | Synced into the credential-scrubbed release snapshot |
| Dense-score diagnostic | `aa02150404418859e33d1ff99fb46543244b9b70` | Preserved as an authenticated, unpromoted snapshot |

Exact tree, patch, and dependency identities are in
`release/SOURCE_PROVENANCE.md` and `release/manifest.json`.

## Experiment coverage

The release scope is every experiment retained by the manuscript, not only the
final 8B training route:

1. non-causal Direct-P operator timing and accuracy, including HAO and FP8
   controls;
2. ViT, BERT, Wan, and ViT-MAE fixed-input evaluations;
3. isolated causal backward timing and correctness;
4. projection-inclusive attention forward/backward timing;
5. 8B B1/B2/B4 complete-update timing; and
6. matched 100-billion-token distributed BF16 and
   NVFP4-projection/FP8-P/V trajectories, plus the separate E4M3/NVFP4
   projection by FP8/MXFP4-P/V diagnostic matrix.

`release/EXPERIMENT_MATRIX.md` maps each family to source, command, evidence,
and remaining external requirements. Some historical paper artifacts can be
regenerated from committed receipts but cannot be reacquired byte-for-byte
because the original raw capture was not preserved. Those cases are labelled
receipt-only.

## Supported training boundaries

The D128 release candidate is causal Llama-style grouped-query attention at
sequence length 4096 and exact local batches 1, 2, or 4 on NVIDIA Blackwell
SM100. The paper's principal end-to-end result uses batch 4.

| Route | Learned projections | Attention forward | Attention backward | Status |
| --- | --- | --- | --- | --- |
| BF16 control | BF16 | BF16 FlashAttention | BF16 FlashAttention | Reference |
| FP8-P/V candidate | NVFP4 Q/K/V/O | NVFP4 Q/K, E4M3 FP8 P/V | Reconstruct P from saved quantized Q/K; E4M3 Q/K/V and E5M2 dO | Candidate |
| MXFP4-P/V diagnostic | NVFP4 Q/K/V/O | NVFP4 Q/K, MXFP4 P/V with E8M0 block scales | Same backward binary | Diagnostic only |
| E4M3-projection FP8-P/V control | E4M3 Q/K/V/O | NVFP4 Q/K, E4M3 FP8 P/V | Same backward binary | Diagnostic only |
| E4M3-projection MXFP4-P/V diagnostic | E4M3 Q/K/V/O | NVFP4 Q/K, MXFP4 P/V with E8M0 block scales | Same backward binary | Diagnostic only |

The matched BF16 and FP8-P/V trajectories both completed the declared
100,000,595,968-token schedule. At the final same-update validation point,
their losses were 2.3048148155 and 2.3948404789, respectively. Across all 874
common post-warmup reports, median throughput was 21,852.6656 and 24,302.9730
tokens/s/GPU, a 1.1121285x ratio of medians. This is one trajectory per arm,
not an estimate of run-to-run uncertainty or evidence of quality equivalence.
Both historical MXFP4-P/V projection arms and the later B4 arm diverged; that
route remains reproducible for diagnosis but is not a recommended training
recipe.

The separate D64 profile fixes a 1.235B model at B16/S4096/Hq32/Hkv8/D64. It
admits a BF16 control and two E4M3-projection routes: NVFP4 Q/K with either FP8
or anchored MXFP4 P/V. Both low-precision routes use the native v416 backward
over represented E4M3 Q/K/V/dO. The source, schema-v3 manifests, renderer, and
CPU contracts are present. Clean-clone GB200 builds, DDP16 save/fresh-resume,
and long-horizon public-data runs remain pending, so this profile is a
reproduction target rather than a validated release recipe. See
`release/D64_REPRODUCTION.md`.

## Safety boundaries

- Direct D128 CuTe FP4-QK backward remains disabled because its SM100 two-CTA
  schedule can hang.
- The fast MXFP4 shift-16 forward remains disabled because it produced
  nonfinite outputs.
- Historical v506, v507, v508, v503/shared-MX, and `mx_exact_replay` paths are
  diagnostic-only and are not selected by the portable training adapter.
- Unsupported shapes and artifact/ABI mismatches must raise an error. They must
  never fall back silently while retaining a low-precision route label.
- Historical shared libraries are not runtime source truth. All extensions are
  rebuilt from pinned source.

## Source-publication state

Publication consent, the Apache-2.0 project license and copyright notice,
public-surface hygiene, publication-asset handling, third-party license
collection, and independent reachability checks for all five root and two
nested dependency pins are complete. The exported project history contains
one parentless commit, no inherited tags or branches, no local object
alternates, and no development-history objects. The final source inventory,
release verifier, CPU suite, paper build, submission-package builds, and
credential scan were run from a separate no-local clone before publication.

## Scientific validation backlog

The following work governs reproducibility and promotion of validated routes.
It does not prevent publication of the clearly labelled source snapshot:

1. Validate the authenticated D64/B16 build and 1.235B renderer from a fresh
   recursive clone on GB200, including DDP16 save/fresh-resume and a
   long-horizon public-data run. A dedicated SFU-B1 profile remains separate
   follow-up work.
2. Publish an installable environment or container for the recorded NVIDIA
   PyTorch and CUTLASS DSL builds. The measured versions and build wrapper are
   present, but the required packages are not all available from ordinary
   public package indexes.
3. Build from a fresh recursive clone on GB200/SM100 and run the D128/B1/B2/B4
   correctness, liveness, zero-dO, and performance gates.
4. Run the generated B1/B4 TorchTitan configs through checkpoint save/resume
   and a distributed training smoke test from that clean build.
5. Record fresh per-batch HBM headroom and saturation diagnostics for B1/B2/B4
   before treating portable end-to-end timings as validated.
6. Generate new credential-free natural-input captures and receipts. Original
   B1/B2/B4 `.pt` captures and several temporary D128 timing records are absent.
7. Publish or replace evidence that depends on the missing B300 raw archive,
   missing Wan inputs, pre-rendered MAE panel, or hosted training histories.
8. Publish the exact SlimPajama preprocessing, ordering, tokenizer revision,
   and shard checksums needed to reproduce the historical trajectory.
9. Publish the raw, credential-free metric histories and an immutable
   SlimPajama order for the completed matched trajectories. Independent seeds
   are still required for uncertainty or quality-equivalence claims.
10. Decide and document the promotion, retirement, or compatibility policy for
    each preserved historical scratch runtime. Preservation alone does not make
    an old harness a supported public entry point.
11. Validate the public TorchTitan input-prefetch and distributed-launch path
    against the historical saturated timing protocol.

## Release-ready definition

For source publication, a fresh clone of the single parentless commit must
verify all source identities, initialize the recorded dependency closure, pass
the CPU/offline checks, regenerate every deterministic paper artifact, and
explain every artifact that requires external data or a new measurement. It
must contain no reachable private history or operational metadata.

A route becomes fully validated only after the same clean clone builds its
supported kernels without machine-local paths, passes the stated GPU gates,
launches the portable TorchTitan recipe, and completes the required checkpoint
and distributed-training checks. Publication makes the source available; it
does not waive or silently satisfy those scientific validation boundaries.

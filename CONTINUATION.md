# Continuing the FP4 FlashAttention research

For a quick orientation, read `PROJECT_MAP.md` first. This document is the
detailed handoff once the current route and task are clear.

This repository is intended to be a research continuation package, not only a
paper artifact. A researcher with a fresh clone and the public experiment
inputs should be able to inspect every retained forward and backward lineage,
build the supported path, rerun its gates, and extend the work without access
to the original development machines or training repository.

That promise has two boundaries:

- The repository must contain the complete FA4 source and development context.
- Large or restricted datasets, model assets, checkpoints, GPU software, and
  historical service logs may be external, but their identities and absence
  must be explicit. An absent external input must never be replaced silently.

`RELEASE_STATUS.md` is the authority for which boundaries have actually been
validated. This public snapshot preserves the recovered source and evidence;
the remaining clean-build, GPU, and distributed-training gates are stated
there rather than implied by publication.

## First hour in a new checkout

Record the checkout before changing it:

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
```

Initialize only the recorded source dependencies:

```bash
git submodule update --init \
  ThunderKittens SageAttention flash-attention qutlass cutlass
git -C flash-attention submodule update --init csrc/cutlass
git -C qutlass submodule update --init third_party/cutlass
```

Then run the CPU-visible source checks from a clean worktree:

```bash
python tools/generate_fa4_source_inventory.py --check
python tools/verify_fa4_release.py
pytest -q
```

These commands authenticate source and exercise CPU contracts. They do not
prove that CUDA builds, numerical gates, performance, or distributed training
work on a new machine. Follow `docs/fa4_build_environment.md` for the Blackwell
build and `docs/development.md` for the development workflow.

Before changing a kernel, read:

1. `release/SCIENTIFIC_STATE.md` for established results and unresolved work;
2. `release/routes.json` for supported, diagnostic, and disabled routes;
3. `release/KERNEL_MAP.md` for source-level dependencies;
4. `release/LEGACY_LINEAGE.md` for intermediate native-backward revisions;
5. `release/EXPERIMENT_MATRIX.md` for measurement entry points; and
6. `release/DATA_PROVENANCE.md` for exact versus missing inputs.

## Source epochs and their roles

Several source states are retained deliberately. They must not be collapsed
into one supposedly universal implementation.

| Source epoch | Role in this release |
| --- | --- |
| TorchTitan `20b3de7585696c327bd5aa9f9627f0300abdbf9d` | Upstream training base and repository history |
| Training integration `e7db209b0c7017c415fdd66e04e85f96ae24f276` | Source of the portable FA4 adapter, optimizer, data, and checkpoint contracts |
| Causal kernel epoch `4590537f1479e1a7e847f2783e9ab7aa7f11b975` | Kernel bytes used by the final causal B1/B2/B4 route |
| Causal development tree `4c394504998f653aa702d030c5f98864dcf34c75` | Later portability, report, and release-preparation state; it changes no CUDA/C++ kernel from `4590537` |
| Historical epoch `cfc06dadf684279f657ab66254a3a074be4ee3a9` | Complete source/development epoch used for the paper's forward-only HAO comparison grid; it also retains the backward prototypes that existed at that point |
| Direct CuTe overlay base `9743edaf3227a25f6afc4fa7be8b5e8498610553` | Base of the once-uncommitted direct-FP4-QK experiment |
| Direct CuTe overlay `b531f67557b8213db339492cd1629e721776f758` | Durable, disabled overlay in the pinned FlashAttention submodule |
| Dense-score diagnostic `aa02150404418859e33d1ff99fb46543244b9b70` | Local-only v510 negative-result branch preserved as a materialized snapshot |

Only the TorchTitan base is necessarily reachable from this repository's Git
history. The other identifiers describe imported source provenance. Their
durable bytes are materialized here and authenticated by manifests; a fresh
clone must not assume `git show <historical-commit>` will work.

## Where the complete FA4 code lives

- `tk_fa4/` contains the complete causal development tree: forward kernels,
  native backward lineages, projection/operand publication, Python runtimes,
  benchmarks, validators, old prototypes, and rejection notes. Internal
  `v###` filenames are retained because they encode source provenance.
- `TK_quantisation/` contains the quantizers and supporting projection code.
  A tracked AArch64 shared library from the source repository is intentionally
  absent; it is a machine-specific build product, not source.
- `fused_ops/` and `qutlass_binding/` preserve the auxiliary quantization,
  projection, and GEMM source used during development. Their source closure is
  retained even where an old standalone benchmark still expects the former
  training package.
- `baseline_kernels/` contains the retained baseline kernel source.
- `flash-attention/` contains the BF16 control and the durable direct-FP4-QK
  CuTe experiment. Public D128 direct-FP4-QK dispatch remains disabled because
  the two-CTA schedule can hang.
- `reproduction/snapshots/forward_cfc06dad/` preserves the earlier source epoch
  separately from the final causal tree. It contains all 126 source/development
  paths from that epoch, including its non-causal forward work and 12-file
  backward prototype family. Twenty of its 24 non-source result paths remain
  byte-identical in the root causal tree; four generated cubins are deliberately
  omitted, with their identities recorded in `release/PUBLIC_SANITIZATION.md`.
- `reproduction/snapshots/v510_aa021504/` preserves the complete 14-file v510
  branch overlay and patch. It must be ported into a disposable branch, not
  copied over the retained v509 route.
- `ThunderKittens/`, `SageAttention/`, `qutlass/`, and `cutlass/` are exact
  source dependencies. The SageAttention pin is retained because several
  historical forward diagnostics reference its Blackwell implementation.
- `torchtitan/experiments/fa4/` is the portable replacement for the original
  training-repository integration. It owns route selection, artifact
  authentication, model conversion, optimizer state, data registration, and
  checkpoint hooks. Its `trainer.py` and `train.py` provide checkpoint-aligned
  pinned-memory CUDA lookahead and bounded numerical diagnostics; generated
  recipes must launch `python -m torchtitan.experiments.fa4.train` to select
  those facilities. Artifact manifests expose the complete E4M3-versus-NVFP4
  learned-projection by FP8-versus-MXFP4 P/V matrix used by the retained 8B
  controls; only the NVFP4-projection/FP8-PV route is a training candidate.
- `tools/` contains clean builders, configuration rendering, experiment
  planning, data checks, and release verification.
- `results/` contains available raw records, normalized receipts, deterministic
  renderers, and both manuscript source and PDF.

The supported D128 builder, measurement planner, and TorchTitan integration
are portable entry points. Some retained scratch benchmarks under
`TK_quantisation/` and `benchmarks/` still encode old paths or import the former
`low_bits_training` package. They are preserved as development provenance, not
advertised as turnkey commands. Before reviving one, replace its path/import
boundary explicitly and add it to the clean build and receipt graph; do not
mistake an archived scratch harness for the supported route.

Likewise, the 80 versioned native-backward Makefiles are preserved even when a
revision never acquired a portable receipt. `release/legacy_backward_makefiles.txt`
is the verifier-enforced inventory; `release/routes.json` describes only the
decision-relevant families. This distinction keeps discarded ideas available
without inventing evidence for them.

Standalone cross-entropy, streaming-GEMM, and other unrelated exploratory
trees from the old multi-project repository are not part of the FA4 runtime.
The retained training method uses standard cross entropy compiled with
`torch.compile`; Cut Cross Entropy is disabled. If a future manuscript makes a
scientific claim about one of those legacy components, import and license its
source as a separate, explicit scope expansion rather than implying it is
already reproduced here.

## Development rules that prevent false results

- Keep a route's format, shape, batch, source identity, and ABI explicit in
  every build manifest and result receipt.
- Never label a fallback with the requested low-precision route name.
  Unsupported shapes and mismatched artifacts must stop.
- Do not substitute the historical non-causal source for the causal source, or
  the disabled direct CuTe experiment for the retained backward that
  reconstructs scores from saved quantized Q/K, scales, and log-sum-exp.
- Treat an isolated-kernel speedup, projection-inclusive attention speedup,
  complete-update speedup, and distributed tokens/s as different claims.
- Preserve exact input order when comparing numerical trajectories. A
  checkpoint or minibatch from one trajectory is not an independent run.
- Keep unsafe and negative lineages in the tree. Promote one only after its
  declared correctness, zero-input, liveness, and timing gates pass.
- Build into a new external directory. Do not overwrite or treat a historical
  `.so` as source truth.
- When changing the release, regenerate `release/source_files.sha256`, run the
  full verifier and tests, and perform a fresh-clone audit before publishing.

## What is verified now

- The retained causal CUDA/C++ kernel closure is complete relative to the
  recovered source epoch; the release includes all forward and backward
  lineages rather than only the advertised route.
- The direct CuTe overlay is durable and its three modified files match the
  recovered hashes.
- The portable training adapter retains the D128 B1/B4 route contracts and the
  fused BF16 stochastic-rounding AdamW state.
- Ordinary synchronous checkpoint save and load are both routed through the
  opt-in world-size Gloo metadata group, avoiding a second NCCL communicator at
  the memory-saturated B4 shape.
- One matched BF16 trajectory and one NVFP4-projection/FP8-P/V trajectory each
  completed the 100,000,595,968-token 8B schedule. Their final held-out losses
  were 2.3048148155 and 2.3948404789, and their median post-warmup throughputs
  were 21,852.6656 and 24,302.9730 tokens/s/GPU. The normalized receipt keeps
  all 954 training, 81 validation, and 874 common throughput rows.
- Available paper tables and plots can be regenerated from their committed
  inputs by the offline artifact graph.

## What is not yet verified or available

- A fresh recursive clone has not yet passed the full GB200 build, numerical,
  liveness, performance, distributed training, and checkpoint-resume gates for
  the current candidate.
- The exact historical SlimPajama MDS shard identities and sample order are
  missing. Portable new runs use an explicitly pinned public dataset source;
  they are replacements, not byte-identical reruns of the historical jobs.
- The completed matched 8B result has one trajectory per arm, and its raw
  hosted metric histories are not public. It therefore supports the recorded
  comparison but not run-to-run uncertainty or statistical-equivalence claims.
- Several raw B300, Wan, ViT-MAE, natural-capture, NUMA-launch, and hosted-run
  artifacts are missing or require redistribution review.
- The D64 source lineages, schema-v3 B16 builder, and authenticated TorchTitan
  route are present. Their clean-clone GB200, DDP16 save/fresh-resume, and
  long-horizon public-data gates remain unrun.
- The custom NVIDIA PyTorch/CUTLASS DSL environment needs a publicly obtainable
  locked container or reconstruction path.
- Publication authorization, Apache-2.0 licensing, and the sanitized parentless
  public root are complete. Future releases must remain ordinary descendants
  of that root; the private recovery history must never be grafted into it.

Those are release blockers, not invitations to guess. A replacement
measurement must carry a new receipt and be described as new evidence.

## Handoff discipline

At the end of a development session, update `release/SCIENTIFIC_STATE.md` and
`release/routes.json` with:

- the exact Git commit and source-tree identity;
- files and route semantics changed;
- checks actually run, with hardware and inputs;
- measured results and their receipt paths;
- failed or unrun gates; and
- the next smallest experiment that could change the decision.

Keep private credentials, scheduler objects, service URLs, and employer-owned
storage locations outside the repository. A saved chat can explain intent,
but the checked source, manifests, receipts, and this handoff must be sufficient
to recover the technical state without trusting chat memory.

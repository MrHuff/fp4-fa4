# Planning fresh FA4 measurements

`tools/plan_fa4_measurements.py` indexes the retained measurement entry points
used by the paper. It validates inputs and prints a dependency-ordered command
graph, including explicit blocked nodes where an exact command or input did
not survive. It does not execute a command, import CUDA, download a model,
submit a job, or contact a scheduler.

The graph covers:

- the non-causal HAO shape grid and FP8 controls;
- ViT, BERT masked-language-model, and SST-2 replays;
- ViT-MAE reconstruction;
- Wan 2.1 policy builds and the blocked paired-model replay protocol;
- reconstruction of the committed B300 aggregate;
- the shared capture for isolated causal backward and projection-inclusive
  forward plus backward;
- the twelve single-GPU 8B B1/B2/B4 brackets covering the E4M3/NVFP4
  learned-projection by FP8/MXFP4 P/V matrix; and
- the BF16 control plus those four 64-GPU TorchTitan arms; and
- the D64/B16 isolated forward, historical v416 backward, saturated 1.235B,
  short real-data, DDP16 50B-token, and save/fresh-resume boundaries.

The matrix in `release/EXPERIMENT_MATRIX.md` remains the authority for whether
an item is a fresh measurement, an offline renderer, or historical evidence.
The planner does not turn a receipt into a rerunnable experiment.

## List and inspect the graph

Listing families needs no environment:

```bash
python tools/plan_fa4_measurements.py list
```

All paths supplied to `check` or `print` must be absolute. A CPU-only check of
the committed B300 aggregate is:

```bash
python tools/plan_fa4_measurements.py check \
  --family b300-aggregate \
  --python /absolute/path/python3 \
  --output-root /absolute/path/new-results
```

Fresh measurement and build roots must be absent or empty, and they may not
overlap. This prevents the historical forward harness from resuming stale
manifest entries or another driver from overwriting a prior capture.

Print shell or machine-readable JSON without running it:

```bash
python tools/plan_fa4_measurements.py print \
  --family noncausal-forward \
  --python /absolute/path/python3 \
  --output-root /absolute/path/new-results \
  --noncausal-build-root /absolute/path/noncausal-build \
  --cuda-home /absolute/path/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/cutlass-dsl/python_packages

python tools/plan_fa4_measurements.py print --format json \
  --family causal-backward \
  --python /absolute/path/python3 \
  --output-root /absolute/path/new-results \
  --artifact-manifest /absolute/path/fa4-build/manifests/nvfp4_qk_fp8_pv_b1_s4096_sm100.json
```

Source-building families require CUDA with `bin/nvcc` and the measured
CUTLASS DSL 4.5.2 package. The planner validates both roots before emitting
their command graph. It also prepends the selected interpreter's directory to
`PATH`, because the historical Makefiles invoke `python3`; the sibling
`python3` must resolve to the explicitly selected interpreter.

Blocked commands are comments in shell output and have `runnable: false` in
JSON. The process exits with status 2 if any selected node is blocked. This is
intentional: a copied historical command is not a reproduction recipe when an
immutable input or authenticated binary is missing.

## Causal kernel manifests

First use `tools/build_fa4.py` to produce clean-build
`fa4_artifact_manifest_v3` files. Pass every manifest explicitly with repeated
`--artifact-manifest`; the planner never searches a directory for a shared
library.

The isolated backward and projection-inclusive measurements share one
`benchmark_v509_report_boundaries` invocation. Its manifest must select the
B1 NVFP4-QK plus FP8-P/V route. The 8B scaling family accepts the four explicit
learned-projection/PV route manifests at B1, B2, and B4. B2 may be
operator-only. The distributed family requires B4 training manifests for the
BF16 control and all four low-precision controls when reproducing the complete
five-arm matrix.

The separate `llama1p2b-d64` family requires schema-v3 B16 manifests for BF16,
E4M3-projection + FP8-P/V, and E4M3-projection + MXFP4-P/V. It emits the
isolated B16 forward factorial, three saturated 1.235B routes, three DDP16
config graphs, and a DDP16 save/fresh-resume smoke. The exact historical v416
acquisition driver, cd57 CuTe control bytes, and an attention-only combined
forward-plus-backward driver are absent, so those nodes remain explicitly
blocked. See `release/D64_REPRODUCTION.md`.

Every generated causal command:

- names the forward, projection publisher, and native backward files exactly;
- carries their module names, byte counts, and SHA256 values where the driver
  accepts them;
- selects the publisher through
  `TK_FA4_LOWP_BWD_EXTENSION_SOURCE`; and
- invokes package-dependent benchmarks with `python -m`, so a fresh checkout
  does not depend on the script directory being placed on `sys.path`.

## External assets

Model and dataset files are declared in an explicit
`fa4_external_assets_v1` JSON file. Each asset has an immutable revision and a
complete list of files that the planner authenticates. Unlisted files and
symlinked directories are rejected, so a loader cannot silently consume bytes
outside the manifest. Logical asset names also enforce their expected kind and
upstream identifier. Nothing is selected by a filename glob.

```json
{
  "schema": "fa4_external_assets_v1",
  "assets": {
    "llama3_8b_assets": {
      "kind": "huggingface_snapshot",
      "identifier": "meta-llama/Llama-3.1-8B",
      "revision": "0123456789abcdef0123456789abcdef01234567",
      "root": "/absolute/path/llama-assets",
      "files": [
        {
          "path": "tokenizer.json",
          "bytes": 123,
          "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        }
      ],
      "tree_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  }
}
```

The example hashes are placeholders. `tree_sha256` is SHA256 over the listed
files sorted by relative path. Each record contributes UTF-8
`path`, NUL, decimal `bytes`, NUL, ASCII `sha256`, newline.

Recognized asset names are:

- `vit_cifar10_model` and `cifar10_dataset`;
- `bert_mlm_model` and `wikitext_dataset`;
- `bert_sst2_model` and `sst2_dataset`;
- `vit_mae_model` and `coco_val_100` (exactly 100 listed images);
- `wan_1_3b_model` and `wan_14b_model`; and
- `llama3_8b_assets` and `slimpajama_dataset`.

The downstream ViT/BERT/SST-2 nodes use authenticated local model and dataset
snapshots and extensions rebuilt by `noncausal.unified`; no temporary build
path is implicit. These are fully pinned replacement measurements. The
historical paper results remain receipt-only because their original upstream
revisions were not recorded.

The TK and HAO ViT-MAE controls use the authenticated model/image assets and
vendored HAO source. Wan policy-bundle builds and paired HAO-BF16/TK model
replays are runnable: `--model` retains the logical identifier checked by the
policy manifest, while `--model-path` selects only the authenticated local
snapshot. Wan HAO low-precision controls remain blocked because the exact
Wan-shape NV/NV build receipt is absent; an NV/MX extension must not be
substituted.

## Distributed launcher manifest

Configuration rendering is portable and credential-free. Actual multi-node
launch flags are site-owned, so the training node remains blocked until an
explicit launcher manifest is supplied:

```json
{
  "schema": "fa4_torchrun_launcher_v1",
  "source_revision": "0123456789abcdef0123456789abcdef01234567",
  "world_size": 64,
  "executable": {
    "path": "/absolute/path/torchrun",
    "bytes": 123,
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  },
  "argv_prefix": [
    "--nnodes=16",
    "--nproc-per-node=4",
    "--rdzv-backend=c10d",
    "--rdzv-endpoint=host:port",
    "--rdzv-id=fa4-paper"
  ],
  "environment": {}
}
```

The planner authenticates the executable, requires world size 64, rejects
credential-shaped environment names, verifies that `nnodes` times
`nproc-per-node` equals 64, and appends
`-m torchtitan.experiments.fa4.train --job.config-file <rendered.toml>`. This
selects the checkpoint-aligned asynchronous data lookahead used by generated
FA4 recipes. Credentials and scheduler submission belong outside this
repository.

The three generated training configs hold local batch 4, global batch 1024,
world size 64, 23,842 updates, 2,000 warmup updates, stochastic-rounding BF16
AdamW, standard compiled cross entropy, and the same validation/checkpoint
cadence. The MXFP4-P/V route is labeled as a divergence diagnostic; the graph
does not promote it to a successful training method.

The single-GPU B1/B2/B4 command previews remain blocked even with all kernel
manifests present. The committed measurement receipt says GPU 0 was paired
with CPU and memory NUMA node 0, but the exact binding launcher was not
preserved. Printing an unbound command as the published timing protocol would
silently change the experiment.

## Known irreducible gaps

The planner reports, rather than guesses around, the following gaps:

- immutable model/data revisions for the historical ViT/BERT/SST-2 runs (new
  fully pinned replacement runs are now supported);
- the full raw B300 cluster archive;
- part of the historical Wan acquisition and calibration record;
- the original ephemeral B1/B2/B4 raw timing files (new measurements can
  replace them), plus an authenticated reconstruction of their NUMA binding;
- the hosted distributed-service histories behind older curves.

Use `tools/reproduce_fa4_paper.py` to rebuild tables, figures, and the PDF from
the committed credential-free receipts. That offline graph is separate from
the fresh measurement graph described here.

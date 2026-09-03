# Developing FP4 FlashAttention from this repository

This repository is designed to replace the original private training workspace
for future FA4 development. Start with `CONTINUATION.md` and
`release/SCIENTIFIC_STATE.md`; neither chat history nor an old machine should
be required to discover the retained implementation state.

## Establish a clean source state

Record the branch, commit, and worktree before changing anything:

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
git submodule status --recursive
```

Initialize the source closure:

```bash
git submodule update --init \
  ThunderKittens SageAttention flash-attention qutlass cutlass
git -C flash-attention submodule update --init csrc/cutlass
git -C qutlass submodule update --init third_party/cutlass
```

Use `make verify-source` for the CPU-visible integrity checks. The release
verifier requires a clean worktree; run `make test` while developing and run
the complete verifier after committing the intended tree.

## Choose the correct source epoch

- Change `tk_fa4/` for the causal training lineages.
- Change `reproduction/snapshots/forward_cfc06dad/` only when intentionally
  revisiting the older source epoch used by the non-causal paper study. Its
  historical backward prototypes are provenance, not the current causal route.
- Treat `reproduction/snapshots/v510_aa021504/` as an overlay to port, not a
  directory to copy over the current causal tree.
- Change `torchtitan/experiments/fa4/` for public training integration. Do not
  reintroduce an import-time dependency on `gc-training`.
- Change the pinned `flash-attention/` fork only on an explicit submodule
  branch. Preserve its exact base and fail-closed direct-FP4-QK behavior.

`release/routes.json` gives every decision-relevant route family a human name,
support state, shape, source, build entry point, evidence, and promotion rule.
Intermediate native-backward revisions remain available through the
verifier-enforced inventory in `release/legacy_backward_makefiles.txt`; see
`release/LEGACY_LINEAGE.md` before reviving one.

For the 8B training controls, learned-projection precision and attention P/V
precision are independent route dimensions. The builder emits the full
E4M3/NVFP4 projection by FP8/MXFP4 P/V manifest matrix. Do not reproduce an
E4M3-projection historical arm by editing an NVFP4 manifest in place.

## Build without contaminating source

Use an absent or empty absolute build directory outside the checkout. The
builder checks the recorded Python, CUDA, driver, package, source, and submodule
identities before compiling:

```bash
python tools/build_fa4.py plan \
  --build-root /absolute/path/new-fa4-build \
  --cuda-home /absolute/path/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/cutlass-dsl/python_packages

python tools/build_fa4.py build \
  --build-root /absolute/path/new-fa4-build \
  --cuda-home /absolute/path/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/cutlass-dsl/python_packages
```

Never edit a historical `.so` or use one as source truth. Every runtime loaded
by TorchTitan must be named in an artifact manifest with its path, byte count,
module name, and SHA256.

## Change a kernel safely

For a new route or source revision:

1. Give the method a descriptive public name; retain an internal revision tag
   only for source provenance.
2. Add a separate build target. Do not repoint an existing route name at
   different semantics.
3. Encode tensor shape, layout, format, scale convention, ownership, and clear
   responsibilities in both the binding and artifact manifest.
4. Add source-contract and runtime-selection tests before enabling dispatch.
5. Run represented-oracle correctness, finite/nonzero, exact-zero-input,
   liveness, and repeated timing gates at every claimed shape.
6. Compare isolated core, producer plus consumer, projection-inclusive module,
   complete model update, and distributed throughput separately.
7. Add a receipt and update `release/routes.json`,
   `release/SCIENTIFIC_STATE.md`, and `release/KERNEL_MAP.md`.

Unsupported shapes and ABI mismatches must raise. A BF16 bridge may be an
explicitly named control, but it must never be reported under a low-precision
route label.

## Extend TorchTitan

The public integration lives in `torchtitan/experiments/fa4/`:

- `artifacts.py` authenticates binaries and source dependencies;
- `exact_lowp_attention.py` owns the route-specific model adapter;
- `fa4_attention.py` owns the generic wrapper and BF16 control;
- `converters.py` installs model conversions;
- `optimizer/` contains fused BF16 stochastic-rounding AdamW and its state;
- `checkpoint.py` owns the opt-in Gloo metadata-group checkpoint path;
- `data.py` registers the revision-pinned public SlimPajama replacement; and
- `train_spec.py` registers the 1.235B/D64 and 8.03B/D128 geometries.

Use `tools/render_fa4_training_config.py` to create a run configuration from an
authenticated build manifest. The renderer writes a version-2 receipt beside
the TOML file. That receipt binds the exact config bytes, artifact manifest,
and required environment. `scripts/fa4/run_torchrun.sh` verifies all three
before it runs either the NCCL preflight or training. Keep comparison arms
identical apart from the named route. Standard BF16 cross entropy compiled by
`torch.compile` is the supported loss; both historical CCE integrations remain
out of scope.

The original production data pipeline used asynchronous pinned-memory
lookahead and checkpoint-aligned iteration. The public TorchTitan replacement
must preserve equivalent alignment whenever prefetch is enabled, and every
performance comparison must use the same setting. See the experiment package
tests before modifying this boundary.

## Run locally or across nodes

For a single node, launch the FA4-specific trainer so the rendered prefetch and
diagnostic fields are active:

```bash
export FA4_DATALOADER_PIN_MEMORY=0
export FA4_DATALOADER_PREFETCH_FACTOR=8
export FA4_TRAIN_DATALOADER_NUM_WORKERS=8
export FA4_VALIDATION_DATALOADER_NUM_WORKERS=1
export LBT_ADAMW_BF16_SR_CHECKPOINT_SCHEMA=v2-fused-stateless
export LBT_ADAMW_BF16_SR_PROVIDER=lbt_fused_stateless_adamw_bf16_sr
export LBT_ADAMW_BF16_SR_PROVIDER_VERSION=1
export LBT_ADAMW_BF16_SR_SEED=0
export LBT_ADAMW_BF16_SR_SOURCE_SHA256=05e9133ac24ac286e059ebaaef4311921c5566f0b57e07367af30ac2f48f4dbd
export LBT_DCP_SYNC_CPU_PROCESS_GROUP=1
export TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC=1
NNODES=1 NPROC_PER_NODE=4 NODE_RANK=0 \
CONFIG_FILE=/absolute/path/run.toml scripts/fa4/run_torchrun.sh
```

Do not edit a rendered TOML or its artifact manifest in place. Render a new
pair instead. The launcher uses `<config>.receipt.json` by default; set
`CONFIG_RECEIPT` only when passing an explicitly relocated receipt/config pair.

For multiple nodes, use `scripts/fa4/run_torchrun.sh` with an explicit
rendezvous endpoint and configuration. Scheduler allocation and credentials
remain outside the repository. Record GPU topology, CPU/NUMA binding, clocks,
software versions, warmup, synchronization, sample count, and timing statistic
in a new measurement receipt.

Do not compare an unbound run with the old bound timing receipt. If the exact
old launcher cannot be recovered, define a new public protocol and label its
measurements as new.

## Data and checkpoints

Remote datasets require an immutable revision; local datasets require a
complete checksum manifest. Create that manifest outside the snapshot root,
then pass it while rendering:

```bash
python tools/fa4_dataset_manifest.py create \
  --root /absolute/path/slimpajama \
  --output /absolute/path/slimpajama.fa4-dataset-manifest.json
python tools/render_fa4_training_config.py \
  ... \
  --dataset-path /absolute/path/slimpajama \
  --dataset-manifest /absolute/path/slimpajama.fa4-dataset-manifest.json
```

The manifest lists every regular file by relative path, byte count, and SHA256.
Rendering and launching both rehash the complete snapshot and reject added,
missing, changed, or symlinked files. The exact historical SlimPajama Mosaic
shard order is unavailable, so public runs are fresh matched experiments, not
byte-for-byte resumes of the old trajectories.

Keep an explicit integer checkpoint `load_step`; use `-1` for latest. A valid
resume must restore the model, optimizer, data-loader state, and stochastic-
rounding phase. Test both same-process load and fresh-process distributed load.

## Preserve a result

Every new result should record:

- repository commit and dirty-state verdict;
- initialized submodule commits;
- generated artifact manifests and hashes;
- hardware and complete software identity;
- data/tokenizer/checkpoint identity;
- exact command and route configuration;
- warmup, synchronization, and raw timing samples;
- numerical and liveness gates; and
- whether it supersedes, extends, or merely diagnoses an earlier result.

Then regenerate the source inventory and paper artifacts as applicable:

```bash
python tools/generate_fa4_source_inventory.py
pytest -q
python tools/reproduce_fa4_paper.py --run --offline all
git diff --check
```

Public publication additionally requires the licensing, asset-rights,
identifier-hygiene, fresh-history, and clean-clone steps in
`release/PUBLIC_EXPORT_POLICY.md`.

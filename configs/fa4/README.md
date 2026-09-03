# Portable FP4 FlashAttention training recipes

The paper's matched 8B and 1.2B recipes are rendered from authenticated kernel
manifests. A generated TOML never searches `PYTHONPATH` for a convenient
binary: it names every shared library and source dependency by absolute path,
byte count, and SHA256. The training adapter rechecks those identities before
loading CUDA code.

The default profile is Llama 3.1-style 8B grouped-query attention at local
batch 4, sequence 4096, 32 query heads, eight key/value heads, and head
dimension 128. Five D128 route names are explicit:

| Route | Learned Q/K/V/O projections | Attention score product | P/V product | Intended use |
| --- | --- | --- | --- | --- |
| `bf16_fa4` | BF16 | BF16 | BF16 | matched control |
| `nvfp4_qk_fp8_pv` | NVFP4 | NVFP4 Q/K | E4M3 FP8 | training candidate |
| `nvfp4_qk_mxfp4_pv` | NVFP4 | NVFP4 Q/K | MXFP4 with E8M0 block-32 scales | divergence diagnostic |
| `e4m3_proj_nvfp4_qk_fp8_pv` | E4M3 | NVFP4 Q/K | E4M3 FP8 | projection-precision control |
| `e4m3_proj_nvfp4_qk_mxfp4_pv` | E4M3 | NVFP4 Q/K | MXFP4 with E8M0 block-32 scales | recorded B4 divergence diagnostic |

All D128 low-precision routes use the same native backward method: represented
E4M3 Q/K/V, E5M2 output gradients, and NVFP4 score replay. The MXFP4 route
does not substitute a different backward kernel.

The `llama1p2b-d64-b16` build profile adds the fixed 1.235B geometry at local
batch 16, sequence 4096, 32 query heads, eight key/value heads, and head
dimension 64. It emits BF16, E4M3-projection + FP8-P/V, and E4M3-projection +
MXFP4-P/V manifests. Both low-precision D64 routes use the same represented-
E4M3 v416 backward. See `release/D64_REPRODUCTION.md` for its precise ABI,
historical-source snapshots, and unavailable-input boundary.

## 1. Build and authenticate the CUDA extensions

Initialize the pinned submodules, activate the recorded Python/CUDA 13.0
environment, then choose a new absolute build directory. CUTLASS DSL means the
Python-package root containing `cutlass/__init__.py`, not this repository's
CUTLASS C++ submodule.

```bash
git submodule update --init \
  ThunderKittens SageAttention flash-attention qutlass cutlass
git -C flash-attention submodule update --init csrc/cutlass
git -C qutlass submodule update --init third_party/cutlass

python tools/build_fa4.py plan \
  --build-root /absolute/path/fa4-build \
  --cuda-home /absolute/path/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/cutlass-dsl/python_packages

python tools/build_fa4.py verify \
  --build-root /absolute/path/fa4-build \
  --cuda-home /absolute/path/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/cutlass-dsl/python_packages

python tools/build_fa4.py build \
  --build-root /absolute/path/fa4-build \
  --cuda-home /absolute/path/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/cutlass-dsl/python_packages
```

`build` refuses a nonempty build directory. A full build emits BF16 and the
complete E4M3/NVFP4 learned-projection by FP8/MXFP4 P/V manifest matrix for
batches 1, 2, and 4 below
`/absolute/path/fa4-build/manifests`. Low-precision B2 manifests are marked
operator-only; only B1 and B4 have the authenticated native-score training
route. Use repeated `--batch` and `--target` options for an explicit subset.

Build the distinct D64 profile in a separate empty directory:

```bash
python tools/build_fa4.py build \
  --profile llama1p2b-d64-b16 \
  --build-root /absolute/path/fa4-d64-build \
  --cuda-home /absolute/path/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/cutlass-dsl/python_packages
```

This emits schema-`fa4_artifact_manifest_v3` B16 manifests for exactly the
three D64 routes above. D64 and D128 artifacts are different profiles; the
renderer rejects cross-profile substitution even if a filename is edited.

## 2. Render one training config

The renderer rehashes the manifest and every referenced binary. By default it
also requires the exact four-file Llama tokenizer identity recorded in
`release/data_manifest.json`, plus either a local SlimPajama snapshot or an
immutable 40-hex Hugging Face dataset revision.

```bash
python tools/render_fa4_training_config.py \
  --artifact-manifest /absolute/path/fa4-build/manifests/nvfp4_qk_fp8_pv_b4_s4096_sm100.json \
  --output /absolute/path/run/fp8-pv.toml \
  --dump-folder /absolute/path/run/output \
  --hf-assets-path /absolute/path/Llama-3.1-8B-assets \
  --dataset-path cerebras/SlimPajama-627B \
  --dataset-revision 0123456789abcdef0123456789abcdef01234567
```

Replace the example revision with the actual immutable dataset commit. Verify
the tokenizer first with `tools/verify_fa4_data.py`. For a local snapshot, pass
its absolute path with `--dataset-path`, omit `--dataset-revision`, and create
an exhaustive manifest outside the snapshot root:

```bash
python tools/fa4_dataset_manifest.py create \
  --root /absolute/path/slimpajama \
  --output /absolute/path/slimpajama.fa4-dataset-manifest.json
```

Then add `--dataset-manifest` with that manifest path to the render command. An
explicit `--allow-nonhistorical-tokenizer` exists for integration smoke tests;
its receipt is marked nonhistorical and it must not be used for a paper
comparison. The default distributed recipe is world size 64, local
batch 4, global batch 1024, 23,842 updates, 2,000 warmup updates, BF16
parameters and moments, fused stochastic-rounding AdamW, gradient clipping at
1.0, and loss-only `torch.compile`. It deliberately contains no CCE, W&B,
object-store, scheduler, or credential integration.

Generated recipes enable the FA4 trainer's depth-one pinned-memory CUDA
lookahead and inexpensive nonfinite loss/gradient-norm guards. The lookahead is
checkpoint-aligned, so a fresh resume replays the pending batch instead of
skipping it. Expensive per-parameter diagnostics remain disabled unless
selected explicitly.

For the 1.2B recipe, add `--profile llama1p2b-d64-50b` and select a B16 D64
manifest. That profile fixes world size 16, global batch 256, gradient
accumulation one, 47,684 updates (50,000,297,984 tokens), learning rate
`0.00048828125`, 954 warmup updates, checkpoints every 954 updates, and
validation every 262 updates at local batch 16. It uses the tied-embedding
`1B` flavor and RoPE scaling factor 32. Like the 8B recipe, it compiles ordinary
dense cross entropy and keeps CCE disabled.

The portable 1.2B profile uses the maintained BF16 stochastic-rounding AdamW
implementation below. Historical 1.2B launch templates used ordinary AdamW,
so a new public run is not optimizer-byte-identical to those trajectories.

The renderer writes both the TOML and `<toml>.receipt.json`; it refuses to
overwrite either. The version-2 receipt binds the exact TOML bytes, artifact
manifest, tokenizer, local dataset tree or remote revision, and required
environment. Select the corresponding B4 manifest to render any of the five
routes. Keep all non-route arguments identical for a matched comparison.

## 3. Launch TorchTitan

Export the non-secret provider contract recorded in the config receipt. The
values below are exact for this source snapshot:

```bash
export FA4_DATALOADER_PIN_MEMORY=0
export FA4_DATALOADER_PREFETCH_FACTOR=8
export FA4_TRAIN_DATALOADER_NUM_WORKERS=8
export FA4_VALIDATION_DATALOADER_NUM_WORKERS=1
export LBT_DCP_SYNC_CPU_PROCESS_GROUP=1
export LBT_ADAMW_BF16_SR_PROVIDER=lbt_fused_stateless_adamw_bf16_sr
export LBT_ADAMW_BF16_SR_PROVIDER_VERSION=1
export LBT_ADAMW_BF16_SR_SOURCE_SHA256=05e9133ac24ac286e059ebaaef4311921c5566f0b57e07367af30ac2f48f4dbd
export LBT_ADAMW_BF16_SR_SEED=0
export LBT_ADAMW_BF16_SR_CHECKPOINT_SCHEMA=v2-fused-stateless
export TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC=1
```

For one node, use the FA4 trainer so the rendered prefetch and diagnostic
settings are active:

```bash
NNODES=1 NPROC_PER_NODE=4 NODE_RANK=0 \
CONFIG_FILE=/absolute/path/run/fp8-pv.toml scripts/fa4/run_torchrun.sh
```

For multiple nodes, invoke the same script once per node with `NNODES`,
`NPROC_PER_NODE`, `NODE_RANK`, and `RDZV_ENDPOINT` set by the site's rendezvous
mechanism. The script verifies the receipt, complete source/artifact closure,
dataset identity, required environment, and world size before either the NCCL
preflight or `torchtitan.experiments.fa4.train` can execute. Cluster allocation,
network transport, and shared checkpoint storage are intentionally site-owned.

The checkpoint config sets `load_step = -1` explicitly and preserves the
AdamW stochastic phase. The Gloo metadata group avoids allocating a second
world-size NCCL communicator at the memory-saturated B4 shape. Resume only
from a complete TorchTitan distributed checkpoint; changing or omitting the
recorded optimizer-provider identity fails closed.

## Evidence boundary

These commands make a fresh, fully receipted run possible. The portable loader
uses a revision-pinned Hugging Face stream; it does not reproduce the private
MosaicML Streaming Dataset shard order used by the historical curves. See
`release/DATA_PROVENANCE.md` and `release/EXPERIMENT_MATRIX.md` for the exact
boundary between new measurements, deterministic offline regeneration, and
receipt-only historical evidence.

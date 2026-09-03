# D64 / 1.2B reproduction boundary

This release has one portable D64 profile:
`llama1p2b-d64-b16`. It fixes local batch 16, sequence length 4096,
32 query heads, eight key/value heads, and head dimension 64. The model flavor
is `1B`: 16 layers, hidden size 2048, feed-forward size 8192, and tied input
and output embeddings, for 1,235,814,400 unique parameters.

## Routes

The profile admits exactly three manifest routes:

| Manifest route | Learned Q/K/V/O projections | Causal forward | Backward | Release role |
| --- | --- | --- | --- | --- |
| `bf16_fa4` | BF16 | BF16 FA4 | BF16 FA4 | matched control |
| `e4m3_proj_nvfp4_qk_fp8_pv` | E4M3 | row-K16 NVFP4 Q/K and E4M3 FP8 P/V | native v416 over represented E4M3 Q/K/V/dO | low-precision candidate |
| `e4m3_proj_nvfp4_qk_mxfp4_pv` | E4M3 | row-K16 NVFP4 Q/K and anchored MXFP4 P/V | the same native v416 backward | numerical ablation |

The two low-precision routes deliberately share the projection publisher and
backward. P/V means the multiplication of softmax probabilities `P` by value
vectors `V`. The MXFP4 P/V payload is a forward choice; v416 consumes the
publisher's represented-E4M3 backward view of `V`, not that MXFP4 payload.

v416 consumes contiguous BSHD E4M3 Q/K/V/dO, where BSHD orders batch,
sequence, head, and head-dimension axes. Its statistics are
`lstat = 8 - LSE*log2(e)` and `dstat = -16*sum(O*dO)`. It returns BF16
gradients. The artifact profile, shape, module name, bytes, and SHA-256 are
checked before dispatch. A D128 v509 image cannot satisfy the D64 contract.

## Build and render

After initializing the pinned dependencies and selecting the recorded CUDA
13.0 / CUTLASS DSL 4.5.2 environment, build into a new absolute directory:

```bash
python tools/build_fa4.py build \
  --profile llama1p2b-d64-b16 \
  --build-root /absolute/path/fa4-d64-build \
  --cuda-home /absolute/path/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/cutlass-dsl/python_packages
```

This emits schema-`fa4_artifact_manifest_v3` manifests for the three routes.
Render one manifest-bound 50B-token recipe with:

```bash
python tools/render_fa4_training_config.py \
  --profile llama1p2b-d64-50b \
  --artifact-manifest /absolute/path/fa4-d64-build/manifests/e4m3_proj_nvfp4_qk_fp8_pv_b16_s4096_sm100.json \
  --output /absolute/path/run/fp8-pv.toml \
  --dump-folder /absolute/path/run/output \
  --hf-assets-path /absolute/path/Llama-3.1-8B-assets \
  --dataset-path /absolute/path/slimpajama \
  --dataset-manifest /absolute/path/slimpajama.fa4-dataset-manifest.json
```

The named profile fixes world size 16, physical and effective global batch
256 (gradient accumulation one), 47,684 updates, and 50,000,297,984 tokens.
It uses learning rate `0.00048828125`, 954 warmup updates, checkpoints every
954 updates, and 16 validation steps every 262 updates at local batch 16.
Parameters and moments are BF16. Ordinary dense cross entropy is compiled
with TorchInductor; cut cross entropy (CCE) is disabled.

The portable recipe uses fused stochastic-rounding BF16 AdamW because that is
the maintained checkpoint-safe optimizer in this release. Historical 1.2B
templates used ordinary AdamW. A fresh portable run therefore tests the same
model and attention routes but is not an optimizer-byte-identical replay of a
historical trajectory.

Use the side-effect-free planner to inspect the entire evidence graph:

```bash
python tools/plan_fa4_measurements.py print \
  --family llama1p2b-d64 \
  --python /absolute/path/python3 \
  --artifact-manifest /absolute/path/fa4-d64-build/manifests/bf16_fa4_b16_s4096_sm100.json \
  --artifact-manifest /absolute/path/fa4-d64-build/manifests/e4m3_proj_nvfp4_qk_fp8_pv_b16_s4096_sm100.json \
  --artifact-manifest /absolute/path/fa4-d64-build/manifests/e4m3_proj_nvfp4_qk_mxfp4_pv_b16_s4096_sm100.json \
  --output-root /absolute/path/measurements
```

The plan includes the B16 isolated forward factorial, three saturated
single-GB200 model routes, three DDP16 long-run configs, and a DDP16
save/fresh-resume smoke. It also keeps two missing historical measurements
visible as blocked nodes instead of silently substituting a different driver.
The planner prints commands; it does not launch them.

## Historical source epochs

Two source-only snapshots close development history that is not represented
by the newer root source alone:

- `reproduction/snapshots/d64_training_cd59dda` preserves all seven public
  source/test paths changed by
  `cd59dda37ebf22e0d77b9c9d6851ec164b86e3af`, including the matched 1.235B
  real-token harness.
- `reproduction/snapshots/d64_v416_713819d` preserves all 33 public
  source/test paths changed by
  `713819d730369ad9e73ded1aedbc301c261f1130`, including the v389--v416
  optimization lineage and selected v416 runtime.

Each directory includes a source-only one-commit patch, a manifest binding
every file to its historical Git blob, byte count, and SHA-256, and a complete
`SHA256SUMS`. Scheduler YAML and result data were excluded deliberately.

## Evidence and unavailable inputs

The isolated v416 receipt and saturated 20-update receipt are retained under
`results/native_tk_d64_ptx_adaptation_20260829/`. The latter reports a
1.098x short full-update speedup over its packed BF16 reference and a final
held-out-loss delta below 0.001. This is a fixed-state synthetic-token smoke,
not long-horizon convergence evidence.

The August 21 256-update Dolma3 comparison is separate historical evidence.
It used the cd59 source and an older CuTe backward control, not v416. Its exact
220,876-byte CuTe source is known only by SHA-256
`cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`;
those source bytes are unavailable. The Llama tokenizer and 512-row Dolma3
input are identified but not redistributed. The historical SlimPajama MDS
index, shard hashes, and row ordering are also unavailable. A public,
manifested SlimPajama snapshot therefore creates new evidence and must not be
described as a byte-identical historical replay.

The exact isolated v416 acquisition driver and an attention-only combined
forward-plus-backward driver were not retained. Their historical receipts may
support the claims they record, but substituting a current control would be a
new measurement. These gaps remain explicit in
`release/EXPERIMENT_MATRIX.md` and the measurement planner.

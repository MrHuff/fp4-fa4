# Paper experiment matrix

The paper contains several different kinds of evidence. This table separates a
fresh measurement from deterministic regeneration of a table or plot, and from
a historical receipt whose original raw acquisition is unavailable.

Status meanings:

- **Runnable**: source and driver are present; external hardware or public
  model/data downloads may still be required.
- **Offline**: the committed inputs are sufficient to regenerate the artifact
  without a GPU or network access.
- **Receipt-only**: the checked receipt regenerates the paper artifact, but an
  original raw capture, service history, or exact external data identity is
  missing. A new run can produce new evidence, not recreate the missing bytes.

| Paper evidence | Measurement entry point | Artifact generator | Status |
| --- | --- | --- | --- |
| Non-causal HAO shape grid and FP8 controls | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_comprehensive_suite.py` | `results/fp4_fa4_hao_table_gb200_20260802/build_summary.py`; `results/fp4_fa4_unified_20260801/{build_summary.py,plot_summary.py}` | Runnable; Offline rendering |
| ViT, BERT MLM, and SST-2 fixed-input evaluation | `tk_fa4/fp4_fa4_fwd/downstream_provider_suite.py` | the same driver with `--summarize-only` | A new evaluation is runnable with authenticated model/dataset snapshots and rebuilt unified extensions; the historical paper measurement is receipt-only because its original revisions are unknown; Offline rendering |
| ViT-MAE reconstruction metrics | `tk_fa4/fp4_fa4_fwd/eval_vit_mae_reconstruction.py` | `results/fp4_fa4_reconstruction_20260805/build_summary.py` | TK and HAO controls runnable with authenticated COCO/model assets and the vendored HAO source; Offline table; historical qualitative PNG receipt-only |
| Wan policy-bundle build | `tk_fa4/fp4_fa4_fwd/build_wan_nv_mx_bundle.py` | none | Runnable with the pinned kernel environment |
| Wan fixed-input model evaluation | `tk_fa4/fp4_fa4_fwd/{eval_wan_video.py,eval_wan_affine_routes.py}` | `results/fp4_fa4_wan_cute_bf16_20260806/build_tables.py` | A paired HAO-BF16/TK evaluation is runnable with an authenticated local model snapshot; HAO low-precision controls remain blocked by their missing exact build receipt; historical table receipt-only |
| B300 forward aggregate | source/configs under `tk_fa4/fp4_fa4_fwd` and `results/fp4_fa4_b300_tuning_20260802` | `build_summary.py --from-summary` | Offline table; raw cluster archive receipt-only |
| Isolated causal backward | `tk_fa4/lowp_fa4_bwd/benchmark_v509_report_boundaries.py` | `results/fp4_fa4_technical_report_v2_20260819/plot_causal_training.py` | Runnable on GB200; Offline plot; historical binary/raw capture receipt-only |
| Projection-inclusive forward/backward | same boundary harness | same plotter | Runnable on GB200; Offline plot; historical binary/raw capture receipt-only |
| 8B local B1/B2/B4 update timing | `tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py` plus the TorchTitan route adapter | same plotter | Kernel and harness source present, but exact reproduction of the paper protocol is blocked because the NUMA binding launcher was not preserved; Offline plot; original temporary raw records absent |
| D64 B16 isolated forward | `tk_fa4/lowp_fa4_bwd/benchmark_b16_forward_factorial.py` | its JSON output | Runnable on one GB200 with the D64 FP8/MX forward manifests; produces a new matched measurement |
| D64 B16 isolated v416 backward | exact historical acquisition driver not retained | `results/native_tk_d64_ptx_adaptation_20260829/v416_production_vec2_ds_receipt_20260829.json` | Receipt-only historical comparison; v416 source/build/runtime are present, but the absent cd57 CuTe control and acquisition driver prevent a byte-identical rerun |
| D64 B16 saturated 1.235B update | `tk_fa4/lowp_fa4_bwd/benchmark_llama12b_saturated.py` | its JSON and sample outputs | Runnable on one GB200 for BF16, FP8-P/V, and MXFP4-P/V with v416; retained 20-update receipt is a short synthetic-token smoke, not convergence evidence |
| D64 256-update Dolma3 numerics | historical `cd59dda` source snapshot | `results/dolma3_causal_training_20260821/raw/1p2b/llama1p2b-matched.json` | Receipt-only: tokenizer, Dolma3 JSONL, historical binaries, and exact cd57 CuTe control bytes are not redistributed; this experiment predates v416 |
| D64 DDP16 50B-token replacement | schema-v3 manifests rendered with `--profile llama1p2b-d64-50b` and launched with `torchtitan.experiments.fa4.train` | config and launch receipts, then a new training curve | Config rendering is runnable with an authenticated public SlimPajama snapshot; clean distributed save/fresh-resume and long-horizon runs remain pending |
| 8B matched distributed training | manifests rendered through `tools/render_fa4_training_config.py` and launched with `torchtitan.experiments.fa4.train` | `llama8b_b4_completed_20260903.json` and `plot_causal_training.py` | The matched BF16 and NVFP4-projection/FP8-P/V trajectories completed 100,000,595,968 tokens; all 954 train, 81 validation, and 874 common post-warmup throughput rows render offline; raw service histories and exact historical MDS order are absent, and a fresh public-data run still requires a site rendezvous and clean distributed save/resume validation |
| MXFP4-P/V divergence diagnostics | the explicit E4M3- or NVFP4-projection MXFP4 manifests | legacy matched snapshot and `plot_causal_training.py` | The MX evidence remains a separate divergent diagnostic; config rendering is runnable, while fresh GPU/distributed validation and historical raw service histories remain unavailable |

## Deterministic paper regeneration

From the repository root:

```bash
make -C results/fp4_fa4_technical_report_v2_20260819 data
make -C results/fp4_fa4_technical_report_v2_20260819
```

The first command rebuilds every table and plot for which committed inputs and
a renderer exist. `data-from-raw` additionally expects the separately
published B300 raw-capture bundle. The manuscript build uses
`SOURCE_DATE_EPOCH=1788357489` through its Makefile.

## Non-causal forward measurement

The historical snapshot is deliberately separate from the final causal source.
A representative measurement is:

```bash
python reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_comprehensive_suite.py \
  --shape 1,4096,24,128 \
  --variant nvmx-fast \
  --warmup-ms 300 --rep-ms 3000 \
  --cooldown-seconds 0.8 \
  --seed 20260814 --gpu 0 \
  --output-dir /path/to/output \
  --build-root /path/to/build
```

Repeat with `nvmx-accurate` and the shape list recorded in the paper's
reproduction appendix. `HAO_FLASH_ATTN_ROOT` may override the vendored HAO
source; leaving it unset uses the pinned snapshot in this repository.

## Causal kernel builds

Use the release build wrapper for the causal kernel closure:

```bash
python tools/build_fa4.py build \
  --build-root /absolute/path/to/new/fa4-build \
  --cuda-home /absolute/path/to/cuda-13.0 \
  --cutlass-dsl-root /absolute/path/to/cutlass-dsl/python_packages
```

The default wrapper profile authenticates the pinned submodules and measured
environment, overrides every historical output directory, refuses a nonempty
build root, and writes hashed D128 manifests for B1, B2, and B4. The separate
`--profile llama1p2b-d64-b16` build emits the fixed D64/B16 BF16, FP8-P/V,
MXFP4-P/V, publisher, and v416 closure. Exact `plan`, `verify`, subset-build,
and operator-only commands are documented in `docs/fa4_build_environment.md`.

## External scientific inputs

The repository does not redistribute restricted or large assets. Fresh
evaluations need:

- CIFAR-10 plus `nateraw/vit-base-patch16-224-cifar10`;
- WikiText-2 plus `google-bert/bert-base-uncased`;
- GLUE/SST-2 plus `textattack/bert-base-uncased-SST-2`;
- COCO 2017 validation images plus `facebook/vit-mae-base`;
- the selected Wan 2.1 Diffusers checkpoints; and
- a documented SlimPajama token stream and Llama tokenizer for pre-training.

Acquisition scripts must pin immutable revisions and write checksums. Tokens,
credentials, scheduler templates, and private storage endpoints are not
scientific inputs and must not enter this repository.

## Known evidence gaps

The current archive does not contain the full B300 raw capture, the original
temporary B1/B2/B4 timing records, all Wan table inputs, the raw hosted service
histories behind distributed curves, or an immutable checksum manifest for the
historical SlimPajama token order. The D64 archive additionally lacks the exact
cd57 CuTe source bytes, isolated v416 acquisition driver, attention-only
forward-plus-backward driver, tokenizer payload, and Dolma3 input payload.
These gaps are explicit release blockers. They must be filled by publishing
the missing artifacts or by replacing the claim with a new, fully receipted
run.

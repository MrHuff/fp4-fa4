# FP4 FlashAttention project map

This file is the shortest path through the repository. It distinguishes the
small current working set from the historical kernels and experiment records
that are retained for provenance. Nothing under the historical headings is a
silent fallback for a current route.

## The two paper studies use different source epochs

| Study | Canonical source | Why it is separate |
| --- | --- | --- |
| Non-causal forward inference | `reproduction/snapshots/forward_cfc06dad/` | This exact snapshot produced the Direct-P forward study and its matched HAO comparisons. |
| Causal forward, backward, and training | Root `tk_fa4/` plus `torchtitan/experiments/fa4/` | This later tree contains the saved-quantized-QK backward and the 8B/D128 training integration. |

Do not rebuild a non-causal paper result from the root causal tree, and do not
substitute the older backward prototypes from the forward snapshot for the
current causal backward.

## Current causal stack

| Layer | Current path | Diagnostic or control |
| --- | --- | --- |
| BF16 attention | `flash-attention/flash_attn/cute/` through `torchtitan/experiments/fa4/fa4_attention.py` | Matched reference control |
| Low-precision forward | `tk_fa4/fp4_fa4_fwd/` with builders in `tk_fa4/lowp_fa4_bwd/` | FP8-P/V is the training candidate; safe MXFP4-P/V is diagnostic |
| Projection and operand publication | `tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu` and `projection_fp4_epilogue.cuh` | Publishes the matrix-oriented representations consumed by forward and backward |
| D128 backward | `tk_fa4/native_gqa_tk_bwd/Makefile.v509*` and the `v509_...` translation unit | `v501`--`v508` are retained development steps, not newer alternatives |
| D64 backward | D64 `v416` source selected by the `llama1p2b-d64-b16` build profile | Separate 1.2B route; it is not interchangeable with D128 `v509` |
| Autograd and route selection | `torchtitan/experiments/fa4/exact_lowp_attention.py` | Enforces shape, artifact, and ABI boundaries |
| Training configuration | `torchtitan/experiments/fa4/` and `tools/render_fa4_training_config.py` | `configs/fa4/README.md` documents the portable inputs |
| Optimizer | `torchtitan/experiments/fa4/optimizer/` | Fused BF16 stochastic-rounding AdamW used by the retained training recipe |

The primary 8B low-precision training candidate is route
`causal_nvfp4_qk_fp8_pv_v509` in `release/routes.json`. The corresponding safe
MXFP4-P/V route uses the same backward but is retained as a divergence
diagnostic, not as a recommended pre-training recipe.

## Where to work

| Task | Start here | Verification map |
| --- | --- | --- |
| Change causal FP8-P/V forward | `tk_fa4/lowp_fa4_bwd/build_causal_gqa_fp8pv_forward.py` | `release/KERNEL_MAP.md`, “Causal D128 forward” |
| Change causal MXFP4-P/V forward | `tk_fa4/lowp_fa4_bwd/build_causal_gqa_d128_mxfp4pv_forward.py` | Same section and the MX publication tests |
| Change D128 backward | `tk_fa4/native_gqa_tk_bwd/Makefile.v509_b4` and the adjacent `v509_...` source | `release/KERNEL_MAP.md`, “Backward from saved quantized operands” |
| Change quantization or projection publication | `tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh` | `release/KERNEL_MAP.md`, “Projection and operand publication” |
| Change TorchTitan integration | `torchtitan/experiments/fa4/exact_lowp_attention.py` and `trainer.py` | `torchtitan/experiments/fa4/README.md` |
| Add or rerun a measurement | `tools/plan_fa4_measurements.py` | `release/EXPERIMENT_MATRIX.md` |
| Rebuild the manuscript | `tools/reproduce_fa4_paper.py` | `results/fp4_fa4_technical_report_v2_20260819/SUBMISSION.md` |
| Resume an unresolved investigation | `release/NEXT_EXPERIMENTS.md` | `release/SCIENTIFIC_STATE.md` |

`tools/build_fa4.py` is the authoritative build selector. Do not choose an
implementation because its filename has the largest version number.

## Historical and disabled source

| Source | Status | Rule |
| --- | --- | --- |
| `reproduction/snapshots/forward_cfc06dad/` | Historical paper source | Use only for the non-causal Direct-P study and its exact contemporaneous prototypes. |
| `reproduction/snapshots/v510_aa021504/` | Preserved, unpromoted diagnostic | Port into a disposable branch before experimentation; never copy it over v509. |
| D128 `v501`, `v503`, `v506`, `v507`, and `v508` | Diagnostic or disabled lineage | Consult `release/routes.json` and `release/LEGACY_LINEAGE.md` before reuse. |
| `flash-attention/flash_attn/cute/fp4_flash_bwd_sm100.py` | Preserved direct-FP4-QK experiment | Public D128 dispatch remains disabled because the two-CTA schedule can hang. |
| `tk_fa4/deprecated/` and `benchmarks/_archive/` | Archived scratch paths | Evidence and development context, not supported entry points. |

The repository intentionally retains these paths. Deleting or moving them
would break authenticated manifests, historical commands, or the evidence
chain without making the active implementation simpler.

## Result archive

The authoritative manuscript is
`results/fp4_fa4_technical_report_v2_20260819/main.tex`. Its direct generated
inputs are indexed in `results/README.md`. The hundreds of other dated files
and directories are experiment evidence, not separate supported FA4 methods.

For a scientific conclusion, read `release/SCIENTIFIC_STATE.md`. For the exact
command and input boundary behind a paper result, read
`release/EXPERIMENT_MATRIX.md`. For machine-readable route status, read
`release/routes.json`.

## Routine commands

```bash
# List the supported measurement families without running them.
make list-measurements

# Validate the committed source and CPU-visible contracts.
make verify-source
make test

# Reproduce every paper artifact supported by committed inputs.
make paper

# Print the clean SM100 build plan. The three roots must be absolute.
FA4_BUILD_ROOT=/absolute/new/build \
CUDA_HOME=/absolute/cuda-13.0 \
CUTLASS_DSL_ROOT=/absolute/cutlass-dsl/python_packages \
make build-plan
```

Full build, dataset, checkpoint, and distributed-training requirements are in
`docs/fa4_build_environment.md`, `release/DATA_PROVENANCE.md`, and
`CONTINUATION.md`.

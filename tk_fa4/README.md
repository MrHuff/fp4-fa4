# `tk_fa4` source guide

This tree contains the current causal kernels and the retained development
lineage. It is intentionally broader than the supported release route.

## Current working set

- `fp4_fa4_fwd/`: base causal forward sources for FP8-P/V and MXFP4-P/V.
- `lowp_fa4_bwd/`: isolated forward builders, projection and operand
  publication, Python runtime adapters, benchmarks, and validators.
- `native_gqa_tk_bwd/`: native grouped-query backward. The current D128
  release route uses `v509`; D64 work uses `v416`.
- `b300_causal_bf16_baseline/`: retained BF16 causal baseline source.

Use `tools/build_fa4.py` from the repository root to select and build a route.
Do not choose a kernel because it has the largest version number.

## Historical and diagnostic paths

- D128 `v501`, `v503`, `v506`, `v507`, and `v508` record specific numerical
  or scheduling experiments. `v510` is preserved separately under
  `reproduction/snapshots/v510_aa021504/` and is not promoted.
- `b300_causal_fp4_experiments/`, `b300_noncausal/`, and `deprecated/` retain
  experiments and earlier implementation strategies.
- Top-level prototypes and dated notes are provenance, not alternate release
  entry points.
- The exact non-causal Direct-P paper source lives under
  `reproduction/snapshots/forward_cfc06dad/`; do not reconstruct it from this
  later causal tree.

See [`PROJECT_MAP.md`](../PROJECT_MAP.md) for the repository-level view,
[`release/KERNEL_MAP.md`](../release/KERNEL_MAP.md) for the transitive source
map, and [`release/routes.json`](../release/routes.json) for the status of every
decision-relevant route.

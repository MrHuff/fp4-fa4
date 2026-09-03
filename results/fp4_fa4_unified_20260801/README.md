# Unified FP4 FA4 speed and accuracy matrix

This directory is the single source for the technical report's kernel-level
comparisons. The published matrix contains six shapes and seven
ThunderKittens variants, plus native HAO NV/NV, HAO NV/FP8, and HAO BF16
controls.

The raw shards also preserve the retired `nvmx-balanced` cases. The
summarizer validates those files for provenance but excludes the policy from
`unified.csv`, generated plots, downstream tables, and the retained frontier.

## Protocol

- NVIDIA GB200, batch 1, D128, noncausal attention
- HAO `create_nvfp4_attention_tensors` input factory
- deterministic seed `20260814`
- 300 ms warmup and 3000 ms timing window
- `triton.testing.do_bench` median
- accuracy against the BF16 output from the same input
- one canonical BF16 timing per shape from a dedicated same-GPU native
  reference process
- native HAO NV/NV, NV/FP8, and BF16 use preallocated outputs and six
  provider-order permutations (50 ms warmup and 500 ms timing each), so
  every provider appears first, middle, and last exactly twice

The tested shapes are `(S,H) = (256,16), (1024,24), (2048,24), (4096,24),
(4096,64), (8192,64)`.

## Outputs

- `summary.json`: normalized records and protocol metadata
- `unified.csv`: flat time/speedup/cosine/relative-L2/RMSE table
- `references/`: dedicated HAO NV/NV, NV/FP8, and BF16 timing/error records,
  including every window sample and provider order
- `p_format_range.json` and `.csv`: isolated P-scale representability test
- `tables/`: generated LaTeX rows and report macros
- `figures/`: generated speed/error plots
- `shard*/cases/`: immutable case-level benchmark records
- `../fp4_fa4_downstream_matrix_20260801/`: six-task TK/HAO downstream
  comparison and shiftless NV/NV failure diagnostics

The per-variant case files also measured BF16, but those samples became
bimodal under the concurrent multi-GPU run. They are retained in
`summary.json` under `reference_timing_audit` and are not used as speedup
denominators.

Regenerate report artifacts with:

```bash
python3 build_summary.py
python3 plot_summary.py
```

`p_format_range_diagnostic.py` is separate because it allocates exact
softmax test tensors on a GPU. It measures format representability, not
kernel latency.

# Hardware-aware FP4 FlashAttention report

This directory contains the v2 structured report for the ThunderKittens FP4
FlashAttention-4 forward and causal-training investigations on NVIDIA
Blackwell.  It preserves the July report layout and adds the final D128 causal
backward, projection-inclusive attention, and end-to-end transformer evidence
collected through September 3, 2026.  The retained training path passes the
forward Q/K quantization, scales, and normalizer into causal backward; the
four-arm 8B diagnostic selects FP8 rather than MXFP4 probabilities and values.

The paper treats HAO AI Lab's two-query TMEM pipeline as the structural
baseline and places the retained contributions on top: NVFP4 QK plus MXFP4
P/V, direct softmax-to-E2M1 approximation, SFU/FMA routing, and represented-P
normalization.

Primary data:

- `../fp4_fa4_hao_table_gb200_20260802/summary.json`: matched fast, accurate,
  native HAO NV/NV, and HAO BF16 results over HAO's published D128 grid;
  `published_hao_results.json` is the checked B200/GB300 NV/FP8 source table.
- `../fp4_fa4_unified_20260801/summary.json`: FP8 controls and additional
  H64 saturation shapes.
- `../fp4_fa4_downstream_matrix_20260801/summary.json`: ViT, BERT MLM, and
  SST-2 fixed-input evaluations.
- `../fp4_fa4_reconstruction_20260805/summary.json`: 100 paired ViT-MAE
  reconstructions on COCO validation images, including native HAO NV/FP8 and
  NV/NV controls, paired confidence intervals, and the qualitative panel.
- `../fp4_fa4_wan_20260805/`: paired Wan2.1-1.3B and Wan2.1-14B diffusion
  trajectories for stable TK NV/MX, HAO NV/NV, HAO NV/FP8, and BF16.
- `../fp4_fa4_b300_tuning_20260802/`: archived Volt B300 job definitions
  plus downloaded compatibility/tuning artifacts, the full 24-shape D64
  NV/MX and NV/NV matrix, final D128 cross-generation records, D128/D64
  NV/FP8 and HAO headline comparisons, and the generated summary.
- `../fp4_fa4_d64_gb200_full_20260802/`: matched 24-shape GB200 D64 NV/MX
  and NV/NV manifests.
- `../native_tk_d64_ptx_adaptation_20260829/`: authenticated D64 native-TK
  backward progression and the saturated Llama-1.2B v416 receipt.
- `../native_tk_d128_gqa_20260829/`: matched D128 native-TK versus generated
  CuTe backward matrices and numerical gates.
- `../tk_fa4_d128_v501_corrected_20260829/`: corrected D128 score-reconstruction
  ABI, saturated 8B BF16/FP8/MX bracket, and straight-MX V experiments.
- `../tk_fa4_d128_shared_tile_mx_20260830/`: one-quantization shared-MX
  producer, represented oracle, composed slice, and saturated 8B A/control/B
  gate.
- A historical companion-training run records the native-TK 8B BF16/FP8/MX
  smoke with fused BF16-SR AdamW, exact runtime hashes, memory, and
  checkpoint-resume evidence.  It is provenance from the separately versioned
  training repository rather than a standalone artifact in this directory.
- `receipts/v509_four_arm_cutoff_20260831T2209Z.json`: frozen redacted cutoff
  values, early matched observation, and first-retained-instability markers for
  the final four-arm numerical diagnostic.  The filename retains the internal
  implementation identifier for provenance; the paper describes the
  forward-to-backward data flow directly.
- `receipts/causal_d128_report_boundaries_20260901.json`: all 101 samples for
  isolated backward and projection-inclusive attention timing, plus numerical
  gates and authenticated runtime identities.
- `../tk_fa4_8b_batch_scaling_20260901/e2e_batch_scaling_summary.json`:
  authenticated single-GB200 B1/B2/B4 complete-update medians for paired
  bfloat16 controls, NVFP4 projections plus FP8 P/V, and NVFP4 projections plus
  MXFP4 P/V. This is explicitly performance-only evidence; the receipt also
  records the poor initial-logit agreement and exact runtime identities.
- `receipts/llama8b_e2e_b1_v509_20260901.json`: the superseded B1-only timing
  bracket, retained with all 31 samples for historical provenance.
- `receipts/llama8b_training_curves_20260901.json`: credential-free,
  token-aligned histories for the two working FP8-P/V arms, the two diverging
  MXFP4-P/V arms, and the historical SFU B1 bfloat16 reference.  The figure
  smoother consumes every unique recovered token coordinate from each current
  arm; one row per 100 current updates is retained for display.  The historical
  reference retains its previously sampled curve and committed smoother.
- `receipts/llama8b_b4_w64_launch_check_20260902.json`: matched W64/B4
  bfloat16 and NVFP4-projection+FP8-P/V observations over updates 300--400,
  the rejected NVFP4-projection+MXFP4-P/V arm, and authenticated update-239
  checkpoint inventories.  This early launch check is used only for route
  validation, not for the aggregate-throughput comparison.
- `receipts/llama8b_b4_completed_20260903.json`: checkpoint-selected,
  credential-free B4/W64 histories for the matched bfloat16 and
  NVFP4-projection+FP8-P/V arms over the completed 100B-token schedule.  It
  contains 954 aligned training reports, 81 aligned validation reports, and
  all 874 common post-warmup throughput observations.  This is one trajectory
  per route and therefore not a repeated-run estimate.
- `receipts/llama8b_b4_matched_snapshot_20260902T1358Z.json`: the earlier
  matched snapshot, retained as the source of the separate failed MXFP4-P/V
  diagnostic.  Its healthy-route prefix is superseded by the completed
  receipt above.

The distributed rows are frozen over the completed schedule.  In the
earlier four-arm diagnostic, both FP8-P/V arms remained non-divergent through their common
55.5B-token horizon, while both MXFP4-P/V arms separated by roughly 0.1B
tokens and diverged.  The report keeps those historical curves and their
failure diagnostic in the causal-training design-history appendix because the
historical bfloat16 trace uses a different topology and sample order.  The main
section now plots the later matched B4 bfloat16 and NVFP4-projection+FP8-P/V
trajectories through the end of the 100B-token schedule, their same-update
validation, and aggregate post-warmup throughput.  It reports the
failed MXFP4-P/V fallback separately and does not include that arm in the
throughput comparison.  The report also separates three local D128
measurements: isolated backward, projection-inclusive attention forward and
backward, and a matched local 8B B1/B2/B4 performance sweep.

Table 6 is the consolidated headline result: retained NV/MX across GB200 and
B300 plus HAO's published NV/FP8 rates on every exact shared B200/GB300 shape.
The former standalone HAO long-context and FP8 headline tables are folded into
this comparison. Classification and MLM results are likewise merged into one
downstream table that connects kernel speed, layer error, and final task
behavior. Historical NV/NV work, the accuracy-matched FP8 control, unsafe
format controls, and rejected directions are kept in the appendices.

Regenerate tables and figures, then build the PDF:

```bash
make data
make
```

`make data` is offline and regenerates the plots from committed receipts.  The
development tree also contains a read-only metric-history capture utility.  It
takes service-side run locations through an uncommitted source map and refuses
to copy those identifiers or credentials into its output.  Public regeneration
consumes the committed credential-free receipt; replacing that receipt still
requires authenticated source access and an explicit paper/provenance update.

The report uses the `graphcore_report.sty` layout and the assets vendored in
this directory.  The build is single-column pdfLaTeX and keeps references
before the appendices.

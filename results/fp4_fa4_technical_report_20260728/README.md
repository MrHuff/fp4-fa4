# Hardware-aware FP4 FlashAttention report

This directory contains the structured report for the ThunderKittens
full-FP4 FlashAttention-4 forward investigation on NVIDIA Blackwell.

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
  SST-2 task-shape replays.
- `../fp4_fa4_reconstruction_20260805/summary.json`: 100 paired ViT-MAE
  reconstructions on COCO validation images, including native HAO NV/FP8 and
  NV/NV controls, paired confidence intervals, and the qualitative panel.
- `../fp4_fa4_wan_20260805/`: paired Wan2.1-1.3B and Wan2.1-14B diffusion
  trajectories for stable TK NV/MX, HAO NV/NV, HAO NV/FP8, and BF16.
- `../fp4_fa4_b300_tuning_20260802/`: reproducible Volt B300 job definitions
  plus downloaded compatibility/tuning artifacts, the full 24-shape D64
  NV/MX and NV/NV matrix, final D128 cross-generation records, D128/D64
  NV/FP8 and HAO headline comparisons, and the generated summary.
- `../fp4_fa4_d64_gb200_full_20260802/`: matched 24-shape GB200 D64 NV/MX
  and NV/NV manifests.

Table 5 is the consolidated headline result: retained NV/MX across GB200 and
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

The report uses the Graphcore technical-report layout and branding from
`/workspace/codebases/low-bits-training-ue5m3_fp4-merge/reports/ue5m3_fp4_training`.
The build is single-column pdfLaTeX and keeps references before the appendices.

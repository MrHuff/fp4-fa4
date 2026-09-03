# Results and manuscript index

This directory is an evidence archive. Dated directory names identify distinct
experiment states; they are not separate supported FA4 implementations. Keep
them in place because paper generators and provenance records use their exact
paths.

## Authoritative paper

- Source and PDF: `fp4_fa4_technical_report_v2_20260819/`
- Paper-specific causal receipts:
  `fp4_fa4_technical_report_v2_20260819/receipts/`
- Submission instructions and packager:
  `fp4_fa4_technical_report_v2_20260819/SUBMISSION.md` and
  `prepare_submission_archives.py`

## Direct inputs to the paper build

| Evidence family | Directory |
| --- | --- |
| Unified forward tables and figures | `fp4_fa4_unified_20260801/` |
| HAO comparison table | `fp4_fa4_hao_table_gb200_20260802/` |
| B300 aggregation | `fp4_fa4_b300_tuning_20260802/` |
| Reconstruction experiment | `fp4_fa4_reconstruction_20260805/` |
| Wan comparison | `fp4_fa4_wan_cute_bf16_20260806/` |
| 8B batch-scaling summary | `tk_fa4_8b_batch_scaling_20260901/` |

The completed matched 8B receipt is
`fp4_fa4_technical_report_v2_20260819/receipts/llama8b_b4_completed_20260903.json`.
The isolated backward, forward-plus-backward, single-GPU update, and divergence
receipts are in the same paper receipt directory.

## Everything else

Other directories retain searches, ablations, negative results, intermediate
kernel measurements, and reconstruction evidence. Use
[`release/SCIENTIFIC_STATE.md`](../release/SCIENTIFIC_STATE.md) to find the
result supporting a conclusion,
[`release/EXPERIMENT_MATRIX.md`](../release/EXPERIMENT_MATRIX.md) to find a
paper experiment, and [`release/routes.json`](../release/routes.json) to
distinguish current, diagnostic, and disabled paths.

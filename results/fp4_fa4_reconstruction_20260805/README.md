# ViT-MAE reconstruction replay

This directory contains paired BF16 and low-precision ViT-MAE Base
reconstructions on 100 COCO 2017 validation images. Every run uses the same
seed, 75% patch mask, model weights, and input order. Only the twelve encoder
self-attention layers change; the decoder remains BF16.

The four measured routes are TK NV/MX `fast`, TK NV/MX `accurate`, native HAO
NV/NV, and native HAO NV/FP8. All model shapes are padded from 50 visible
tokens, 12 heads, and D64 into the S256/H16/D128 kernel specialization.
Reported speedups therefore describe that physical attention dispatch, not
end-to-end MAE latency.

Key artifacts:

- `summary.json` and `summary.csv`: paired aggregate metrics.
- `tables/reconstruction_rows.tex`: generated report table.
- `nvmx_fast_report.png`: four-image qualitative panel with 8x differences.
- `*_100.json`: per-image metrics and per-layer attention error.
- `build_summary.py`: consistency checks, confidence intervals, and table
  generation.

Run `eval_vit_mae_reconstruction.py` from `tk_fa4/fp4_fa4_fwd` once per
provider, then regenerate the summary with:

```bash
python3 results/fp4_fa4_reconstruction_20260805/build_summary.py
```

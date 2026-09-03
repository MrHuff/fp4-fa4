# Wan CuTe-BF16 Reference

This directory replaces the earlier PyTorch-BF16 Wan output reference with
HAO's native CuTe-DSL BF16 FlashAttention-4 kernel. Every FP4 metric is paired
against a CuTe-BF16 run with the same checkpoint, prompt, seed, resolution,
frame count, and diffusion schedule.

`wan1p3b_tk_step*.json` and `wan14b_tk_step*.json` contain the promoted TK
NV/MX fast and accurate routes. `wan1p3b_hao_step*.json` and
`wan14b_hao_step*.json` contain HAO NV/NV and NV/FP8. The affine files compare
the global and layer-calibrated Wan14B routes from fresh model processes.

The latency columns remain independently warmed kernel measurements. They do
not use the end-to-end evaluator wall time because that includes Python-side
quantization for TK and first-call CuTe compilation for HAO. `build_tables.py`
combines the paired quality records with those warmed measurements and writes
the two report tables plus `summary.json`.

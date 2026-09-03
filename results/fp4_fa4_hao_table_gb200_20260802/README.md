# HAO-grid GB200 comparison

This directory contains a matched rerun of the D128 rows in HAO AI Lab's
published FP4 FA4 shape table. It compares TK NV/MX `fast`, TK NV/MX
`accurate`, native HAO NV/NV, and native HAO BF16 on local NVIDIA GB200 GPUs.

Protocol: HAO `create_nvfp4_attention_tensors`, seed `20260814`, noncausal
attention, 300 ms warmup, 3000 ms timing, 0.8 s cooldown, and
`triton.testing.do_bench` median timing. Every case records output cosine,
relative L2, and RMSE against BF16 generated from the same Q/K/V values.

The specialized TK full-FP4 path does not support D64, so the final HAO row
is reported as unsupported rather than inferred from a different kernel.

Regenerate merged JSON, CSV, and LaTeX tables with:

```bash
python3 results/fp4_fa4_hao_table_gb200_20260802/build_summary.py
```

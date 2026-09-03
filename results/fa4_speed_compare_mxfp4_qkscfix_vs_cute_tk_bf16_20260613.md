# MXFP4 qkscfix vs Cute-DSL FA4 BF16 vs TK BF16 Speed Compare - 2026-06-13

Method: MXFP4 and TK BF16 use preallocated output buffers and direct extension calls. Cute-DSL uses `flash_attn_func` wrapper returning fresh outputs. CUDA event timing, 30 warmups, 180 iterations. QKV quantization is excluded from MXFP4 timing.

| Shape | MXFP4 ms | TK BF16 ms | Cute-DSL BF16 ms | MXFP4/TK | MX speed vs TK | MXFP4/Cute | MX speed vs Cute | Launch MX/TK |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H4/S1024 | 0.0337 | 0.0453 | 0.1539 | 0.745x | 1.343x | 0.219x | 4.564x | persistent/persistent |
| H4/S2048 | 0.0535 | 0.0731 | 0.1790 | 0.732x | 1.366x | 0.299x | 3.343x | persistent/persistent |
| H4/S4096 | 0.0883 | 0.1228 | 0.2168 | 0.719x | 1.391x | 0.407x | 2.455x | persistent/fullgrid |
| H4/S8192 | 0.1570 | 0.2236 | 0.3248 | 0.702x | 1.425x | 0.483x | 2.070x | persistent/fullgrid |
| H4/S16384 | 0.5234 | 0.4228 | 0.4803 | 1.238x | 0.808x | 1.090x | 0.917x | fullgrid/fullgrid |
| H8/S2048 | 0.0549 | 0.0744 | 0.1898 | 0.738x | 1.355x | 0.289x | 3.460x | persistent/persistent |
| H8/S4096 | 0.0885 | 0.1248 | 0.2288 | 0.710x | 1.409x | 0.387x | 2.584x | persistent/fullgrid |
| H8/S8192 | 0.2803 | 0.2238 | 0.3292 | 1.252x | 0.798x | 0.852x | 1.174x | fullgrid/fullgrid |
| H16/S1024 | 0.0352 | 0.0460 | 0.1549 | 0.766x | 1.306x | 0.227x | 4.396x | persistent/persistent |
| H16/S2048 | 0.0600 | 0.0753 | 0.2004 | 0.797x | 1.255x | 0.299x | 3.340x | persistent/persistent |
| H16/S4096 | 0.1596 | 0.1401 | 0.2285 | 1.140x | 0.878x | 0.699x | 1.431x | persistent/fullgrid |
| H16/S8192 | 0.5339 | 0.4286 | 0.4750 | 1.246x | 0.803x | 1.124x | 0.890x | fullgrid/fullgrid |
| H16/S16384 | 1.9630 | 1.5030 | 1.3178 | 1.306x | 0.766x | 1.490x | 0.671x | fullgrid/fullgrid |

Notes: `MXFP4/TK` and `MXFP4/Cute` are latency ratios; below 1.0 means MXFP4 is faster. Speed columns are reciprocal; above 1.0 means MXFP4 is faster.

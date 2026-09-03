# GB200 D64 FP4 attention matrix

This directory contains the matched 24-shape NVFP4-QK/MXFP4-PV and
NVFP4-QK/NVFP4-PV matrix on four local GB200 GPUs. Each shape uses HAO input
generation and a same-run HAO BF16 reference, with 100 ms warmup and a
1000 ms median timing window. The kernel policy is from revision `35eeb47`.

| Heads | Sequence | NV/MX (ms) | NV/NV (ms) | Faster route | NV/MX delta |
|---:|---:|---:|---:|---|---:|
| 12 | 1024 | 0.016704 | 0.016384 | NV/NV | +1.95% |
| 12 | 2048 | 0.024160 | 0.022784 | NV/NV | +6.04% |
| 12 | 4096 | 0.067680 | 0.067136 | NV/NV | +0.81% |
| 12 | 8192 | 0.184448 | 0.182272 | NV/NV | +1.19% |
| 12 | 16384 | 0.703264 | 0.696320 | NV/NV | +1.00% |
| 12 | 32768 | 2.541120 | 2.522048 | NV/NV | +0.76% |
| 24 | 1024 | 0.016352 | 0.016416 | tie | -0.39% |
| 24 | 2048 | 0.040960 | 0.038848 | NV/NV | +5.44% |
| 24 | 4096 | 0.097888 | 0.096544 | NV/NV | +1.39% |
| 24 | 8192 | 0.358784 | 0.355904 | NV/NV | +0.81% |
| 24 | 16384 | 1.282032 | 1.270784 | NV/NV | +0.89% |
| 24 | 32768 | 4.863296 | 4.806656 | NV/NV | +1.18% |
| 32 | 1024 | 0.016960 | 0.016480 | NV/NV | +2.91% |
| 32 | 2048 | 0.041248 | 0.039200 | NV/NV | +5.22% |
| 32 | 4096 | 0.129664 | 0.126976 | NV/NV | +2.12% |
| 32 | 8192 | 0.430400 | 0.416032 | NV/NV | +3.45% |
| 32 | 16384 | 1.681728 | 1.625088 | NV/NV | +3.49% |
| 32 | 32768 | 6.472256 | 6.224896 | NV/NV | +3.97% |
| 64 | 1024 | 0.026688 | 0.024864 | NV/NV | +7.34% |
| 64 | 2048 | 0.073728 | 0.069632 | NV/NV | +5.88% |
| 64 | 4096 | 0.221184 | 0.214592 | NV/NV | +3.07% |
| 64 | 8192 | 0.846112 | 0.819200 | NV/NV | +3.29% |
| 64 | 16384 | 3.229184 | 3.111232 | NV/NV | +3.79% |
| 64 | 32768 | 12.865536 | 12.376064 | NV/NV | +3.95% |

For S4096 and above, NV/MX is 1.299x-1.373x faster than matched BF16 and
has cosine 0.9493-0.9520. NV/NV is 1.317x-1.420x faster than BF16 and has
cosine 0.9615-0.9632. NV/NV is therefore the GB200 latency winner, while
NV/MX remains useful when MXFP4's wider probability range matters to the
downstream model.

The original NV/NV H24/S32768 window was invalid because an unrelated process
began using GPU 1 during that measurement. `nvnv_h24_manifest.json` replaces
only that point with an isolated GPU 0 rerun (4.806656 ms) and records the
replacement in `audit_note`.

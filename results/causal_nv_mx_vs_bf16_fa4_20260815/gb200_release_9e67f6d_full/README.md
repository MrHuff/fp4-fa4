# GB200 causal release sweep

Date: 2026-08-15

This is a clean rebuild and rerun of the ten-shape causal NVFP4-QK/MXFP4-PV
suite from release revision `9e67f6d`. Each entry is the median of three
independent 300 ms timing windows after 100 ms warmup. Hkv is 8 throughout.
BF16 FlashAttention-4 runs in the same process on the same input tensors.

| S | Hq | D | NV/MX (ms) | BF16 FA4 (ms) | Speedup | Cosine | Relative L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 64 | 0.057600 | 0.081920 | **1.422x** | 0.951057 | 0.316051 |
| 4096 | 32 | 64 | 0.148480 | 0.199712 | **1.345x** | 0.950996 | 0.317333 |
| 8192 | 32 | 64 | 0.468992 | 0.599824 | **1.279x** | 0.950530 | 0.318134 |
| 4096 | 64 | 64 | 0.252512 | 0.324416 | **1.285x** | 0.949899 | 0.320522 |
| 2048 | 32 | 128 | 0.057728 | 0.087072 | **1.508x** | 0.951413 | 0.313747 |
| 4096 | 32 | 128 | 0.145408 | 0.210944 | **1.451x** | 0.950341 | 0.316311 |
| 8192 | 32 | 128 | 0.454656 | 0.629632 | **1.385x** | 0.950187 | 0.316471 |
| 16384 | 32 | 128 | 1.610112 | 2.141184 | **1.330x** | 0.949272 | 0.319597 |
| 4096 | 64 | 128 | 0.247808 | 0.342768 | **1.383x** | 0.949989 | 0.317118 |
| 8192 | 64 | 128 | 0.846848 | 1.120256 | **1.323x** | 0.949431 | 0.319116 |

All 30 timing records are finite. The first repeat of every shape perturbs
future V values; all ten checks preserve the protected output prefix and LSE
bitwise. Every production build uses 128 registers with zero stack and local
storage.

Relative to the preceding promoted D64 policy, detached P reduces latency by
3.49% at S2048/H32, 4.84% at S4096/H32, 5.83% at S8192/H32, and 5.35% at
S4096/H64. D128 changes by at most 0.14%, confirming that the D64-only guard
does not alter that route.

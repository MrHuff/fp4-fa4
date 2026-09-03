# GB200 D64 full-FP4 forward attention

This directory records the first D64 specialization of the HAO-topology TK
forward kernel for NVFP4 QK with either NVFP4 or MXFP4 PV. All rows use
`B=1`, non-causal attention, seed `20260802`, 100 ms warmup, and a 1000 ms
median timing window. BF16 references were replayed sequentially on one GPU
for three 500 ms windows; their median is shared by both FP4 routes.

| Shape | Route | Time (ms) | BF16 (ms) | Speedup | Cosine | Relative L2 |
|---|---|---:|---:|---:|---:|---:|
| S1024 H24 | NV/MX | 0.016640 | 0.020480 | 1.231x | 0.945537 | 0.325641 |
| S1024 H24 | NV/NV | 0.016640 | 0.020480 | 1.231x | 0.960191 | 0.287052 |
| S4096 H12 | NV/MX | 0.067584 | 0.088384 | 1.308x | 0.949279 | 0.315946 |
| S4096 H12 | NV/NV | 0.066784 | 0.088384 | 1.323x | 0.961483 | 0.280707 |
| S4096 H24 | NV/MX | 0.097888 | 0.129024 | 1.318x | 0.949855 | 0.314270 |
| S4096 H24 | NV/NV | 0.096256 | 0.129024 | 1.340x | 0.961987 | 0.278742 |
| S4096 H64 | NV/MX | 0.221184 | 0.288640 | 1.305x | 0.949866 | 0.314092 |
| S4096 H64 | NV/NV | 0.215040 | 0.288640 | 1.342x | 0.962117 | 0.278776 |
| S32768 H24 | NV/MX | 4.859616 | 6.602048 | 1.359x | 0.950906 | 0.311292 |
| S32768 H24 | NV/NV | 4.792320 | 6.602048 | 1.378x | 0.962866 | 0.275796 |

The D64 NV/MX policy is deliberately different from D128. It uses 12 K/V
stages, native `ex2`, the shiftless affine transform, and no quantized
denominator, sampled max, folded QK-scale preload, or wide max reduction.
This reduced S4096/H24 latency from 0.100832 ms to 0.097888 ms while
restoring a finite, accurate LSE. D128 remains on its existing mixed-SFU/ALU
policy and retains its 0.092160 ms S4096/H24 result.

On these Gaussian inputs NV/NV is the faster and more accurate D64 route.
NV/MX remains useful as the range-oriented option for model distributions
whose small probabilities underflow in NVFP4. Downstream validation is still
required before selecting one route globally.

Raw manifests and build logs are in `nvmx_shard*` and `nvnv_shard*`.
The standalone D64 BF16 replays are in `references/`.

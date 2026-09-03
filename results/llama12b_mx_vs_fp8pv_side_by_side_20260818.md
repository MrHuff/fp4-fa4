# Llama-1.2B MXFP4-PV versus exact FP8-PV

## Scope

This is a controlled single-GB200 comparison of three complete 16-layer
Llama-1.2B-like training routes at batch 1, sequence 4096, QH32/KVH8/D64:

- CuTE BF16 attention.
- Projection-native NVFP4 QK with the retained fast MXFP4 PV forward.
- Projection-native NVFP4 QK with exact-softmax E4M3 FP8 PV forward.

Both low-precision routes use the same retained direct-TMA FP8 attention
backward, fused QKV/RoPE publication, BF16 projection dgrad, and fused AdamW.
The MX route uses its exported/fallback attention-backward branch gain of
0.632; exact FP8 uses 1.0.

The models start from identical weights.  The timing run uses 24 measured
steps over four repeated synthetic batches, with a Latin rotation of route
execution order.  Each batch is therefore visited six times by each route.
An initial logits/gradient audit is taken before any optimizer update.

## Performance

| Route | Forward (ms) | Backward (ms) | Optimizer (ms) | Step (ms) | Speedup vs BF16 | Useful MFU |
|---|---:|---:|---:|---:|---:|---:|
| CuTE BF16 | 27.886 | 45.590 | 10.730 | 84.198 | 1.000x | 17.772% |
| NVFP4 QK + MXFP4 PV | 22.434 | 44.206 | 10.739 | 77.383 | 1.088x | 19.338% |
| NVFP4 QK + exact FP8 PV | 24.913 | 44.212 | 10.735 | 79.861 | 1.054x | 18.738% |

Exact FP8 PV is 3.20% slower per complete step than MXFP4 PV.  The difference
is entirely forward-side: its forward is 11.05% slower, while backward differs
by only 0.013%.  Relative to BF16, MX reduces step time by 8.09% and exact FP8
reduces it by 5.15%.

## Initial-state numerical alignment

| Sample versus BF16 | MXFP4 PV cosine | Exact FP8 PV cosine |
|---|---:|---:|
| Logits | 0.741 | 0.795 |
| Embedding gradient | 0.913 | 0.933 |
| Layer-0 Q weight gradient | 0.296 | 0.501 |
| Layer-0 K weight gradient | 0.307 | 0.498 |
| Layer-0 V weight gradient | 0.482 | 0.416 |
| Layer-0 O weight gradient | 0.472 | 0.693 |
| Layer-0 MLP-down gradient | 0.301 | 0.416 |
| Median sampled gradient cosine | 0.390 | 0.499 |

Exact FP8 wins five of the six sampled gradient groups and materially improves
Q, K, O, MLP, and logits alignment.  MX is better only for the sampled V
gradient.  Logits relative L2 is 0.718 for MX and 0.641 for exact FP8.

## Short optimization/stability proxy

All 24 measured updates are finite for all three routes.

| Route | Last-cycle median loss | Ratio to BF16 |
|---|---:|---:|
| CuTE BF16 | 4.2969 | 1.000 |
| NVFP4 QK + exact FP8 PV | 4.8906 | 1.138 |
| NVFP4 QK + MXFP4 PV | 5.0000 | 1.164 |

Exact FP8's last-cycle median is 2.19% lower than MX.  Per-batch final losses
are mixed but favor exact FP8 on two of four batches, favor MX narrowly on one,
and favor MX more clearly on the first batch:

| Batch | BF16 | MXFP4 PV | Exact FP8 PV |
|---|---:|---:|---:|
| 0 | 1.781 | 2.406 | 2.625 |
| 1 | 4.438 | 4.906 | 4.938 |
| 2 | 4.219 | 5.094 | 4.844 |
| 3 | 4.375 | 5.500 | 5.219 |

This is evidence of short-horizon stability and optimization behavior, not a
claim of corpus-scale convergence.  A multi-seed real-token run is still
required before selecting a training default.

## Reproduction

The comparison harness is
`tk_fa4/lowp_fa4_bwd/compare_llama12b_mx_fp8pv.py`.  The raw result from this
run is `/tmp/llama12b_mx_vs_fp8pv_side_by_side_20260818.json`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  tk_fa4/lowp_fa4_bwd/compare_llama12b_mx_fp8pv.py \
  --rounds 24 --training-batches 4 \
  --output /tmp/llama12b_mx_vs_fp8pv_side_by_side_20260818.json
```

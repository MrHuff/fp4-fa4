# 8B end-to-end batch scaling (2026-09-01)

This gate measures a complete 32-layer Llama 3.1 8B training step at sequence
length 4096 on one NVIDIA GB200.  It compares a packed-QKV BF16 FlashAttention
control with two low-precision routes:

- NVFP4 projections + row-K16 NVFP4 Q/K + E4M3 FP8 probability-value (PV)
- NVFP4 projections + row-K16 NVFP4 Q/K + block-32 MXFP4 PV

Both low-precision routes use the same exact-batch native-TK backward:
represented E4M3 Q/K/V, E5M2 dO, native NVFP4 score reconstruction, and a
fused projection publisher for dO/statistics/dQ clearing.  FP8-PV and
MXFP4-PV have distinct format-specific forward implementations; their
backward binary is identical at a given batch.

## Result

Each process was pinned to GPU 0 and NUMA node 0, used ten warm-up steps and
21 measured steps, and reports the median.  The loss is standard dense
cross-entropy compiled with `torch.compile`; the optimizer is fused AdamW.

| Local batch | Route | Step (ms) | Forward (ms) | Backward (ms) | tok/s/GPU | MFU | BF16-relative |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | BF16 control (FP8 bracket) | 260.313 | 73.136 | 139.442 | 15,735 | 33.74% | 1.000x |
| 1 | NVFP4 + FP8-PV | 239.985 | 69.888 | 123.715 | 17,068 | 36.60% | 1.085x |
| 1 | BF16 control (MX bracket) | 261.133 | 73.741 | 140.091 | 15,685 | 33.64% | 1.000x |
| 1 | NVFP4 + MXFP4-PV | 239.250 | 69.122 | 123.426 | 17,120 | 36.71% | 1.091x |
| 2 | BF16 control (FP8 bracket) | 464.245 | 141.657 | 274.732 | 17,646 | 37.84% | 1.000x |
| 2 | NVFP4 + FP8-PV | 415.532 | 125.796 | 242.195 | 19,715 | 42.28% | 1.117x |
| 2 | BF16 control (MX bracket) | 463.814 | 141.595 | 274.546 | 17,662 | 37.88% | 1.000x |
| 2 | NVFP4 + MXFP4-PV | 415.408 | 125.282 | 242.465 | 19,720 | 42.29% | 1.117x |
| 4 | BF16 control (FP8 bracket) | 854.516 | 277.985 | 527.837 | 19,173 | 41.12% | 1.000x |
| 4 | NVFP4 + FP8-PV | 751.722 | 238.950 | 463.619 | 21,795 | 46.74% | 1.137x |
| 4 | BF16 control (MX bracket) | 857.226 | 278.414 | 529.378 | 19,113 | 40.99% | 1.000x |
| 4 | NVFP4 + MXFP4-PV | 751.597 | 238.514 | 464.045 | 21,799 | 46.75% | 1.141x |

The end-to-end gain grows with saturation: roughly 1.09x at B1, 1.12x at B2,
and 1.14x at B4.  FP8-PV and MXFP4-PV are tied end to end: their direct
low-precision step times differ by at most 0.31%, while independently paired
BF16 anchors vary by up to 0.32%.  MXFP4-PV remains modestly faster in the
timed forward portion, but that sub-millisecond difference is not a material
whole-step win.

These are full-route speedups and therefore include NVFP4 QKV/output
projections, not attention-only speedups.

## What was fixed

- Added separately compiled and authenticated B1/B2/B4 forward routes and
  v509 backward routes instead of reusing a B1 binary for larger batches.
- Removed an artificial B4 memory peak by retaining only the diagnostic logits
  slice and releasing the full `[B,S,V]` logits before backward.  This matches
  the production tensor lifetime.  B4 then fit without activation
  checkpointing and peaked at 176.53 GiB for low precision versus 181.16 GiB
  for BF16.
- Fused E5M2-dO publication now clears dQ in the projection epilogue.  The
  selected backward entrypoint skips the redundant dQ clear and still clears
  dK/dV.
- Runtime receipts now serialize the authenticated post-launch topology, not
  the stale pre-launch topology object.

## Backward isolation evidence

The exact-batch validator used two nontrivial captures at B2 and four at B4,
then repeated them in listed and reversed lane orders.

- dK and dV match independent B1 execution bitwise.
- dQ remains inside an explicit empirical concurrent-store envelope.  The
  worst relative L2 was `4.47e-4` at B2 and `5.16e-4` at B4, below the
  `1e-3` limit.
- Exactly zero dO/dstat produces exactly zero dQ/dK/dV.
- With zero dO/dstat and a nonzero dQ sentinel, the fused-path entrypoint
  preserves every dQ value bitwise while clearing dK/dV.  This directly proves
  that the redundant dQ memset is absent from that wrapper.

This is strong bounded evidence against gross batch-lane raster or address
aliasing.  Because dQ uses concurrent BF16 store-add operations, it is not a
formal bitwise dQ-equivalence proof.  Fused-publisher fidelity is covered by
its separate validation gate.

## Numerical warning

This benchmark is performance evidence, not training-quality or convergence
evidence.  Initial-logit cosine versus BF16 is only 0.416--0.426 for FP8-PV
and 0.373--0.374 for MXFP4-PV; relative L2 is about 1.07 and 1.12,
respectively.  Sampled early attention-gradient cosines are mostly near zero.
Do not use these short synthetic-token steps to claim numerical parity.

The exact machine-readable measurements, artifact identities, raw-receipt
hashes, isolation bounds, and source commits are in
`e2e_batch_scaling_summary.json`.

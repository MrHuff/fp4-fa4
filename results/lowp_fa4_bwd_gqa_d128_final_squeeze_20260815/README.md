# Final D128 GQA backward squeeze and numerical repair

Date: 2026-08-15

This pass completes the projection-native owner experiment, removes BF16
dK/dV materialization from that path, repairs the FP8 probability layout used
by dV, and establishes the remaining performance/accuracy frontier for causal
S4096, Hq32/Hkv8, D128.

The production recommendation is the regular two-stage low-precision backward
followed by the optimized materialized NVFP4 QKV-gradient pack/projection.  It
is 1.33--1.36x faster than the measured BF16 component-sum lower bound and its
event-level advantage is 1.44--1.52x across two clean runs.  The fully
owner-native Q/K/V path is correct and avoids completed BF16 dQ/dK/dV, but is
still 11.5--14.3% slower than the regular materialized path.

## Retained timing

Protocol: one GB200 GPU, B1/S4096/Hq32/Hkv8/D128, causal, exact native exp2,
FP8 P/dO for dV, lifted FP8 P reused for dS, two dO stages, direct statistics,
and a producer-native NVFP4 QKV projection operand.  Medians are paired within
each process; clock variation explains the absolute difference between runs.

| Measurement | Run A (us) | Run B (us) |
|---|---:|---:|
| Low-precision backward with dQ clear | 384.160 | 390.176 |
| BF16 backward with workspace clear | 491.648 | 524.448 |
| Materialized low-precision dO + backward + QKV projection | 516.192 | 560.928 |
| Owner-native dO + backward + QKV projection | 590.016 | 625.632 |
| BF16 projection/backward component lower bound | 783.136 | 807.904 |

The regular low-precision attention kernel is 1.28--1.34x faster than BF16.
The owner path saves the standalone gradient pack and reduces the prepacked
NVFP4 projection to about 53 us, but adds roughly 90--100 us to attention.
The saved pack/projection work does not repay that owner-side cost.

## Owner reduction and projection handoff

For query tile \(q\), causality decomposes dQ as

\[
  dQ_q = \sum_{k=0}^{q}\Delta Q_q^{(k)}.
\]

The diagonal CTA \(k=q\) keeps \(\Delta Q_q^{(q)}\) on chip.  Earlier key
owners reduce into one BF16 prior lane

\[
  P_q = \sum_{k=0}^{q-1}\Delta Q_q^{(k)}, \qquad
  dQ_q = \operatorname{BF16}(P_q + \Delta Q_q^{(q)}).
\]

The owner applies inverse RoPE, computes one delayed global scale and E4M3
block scales, packs E2M1 pairs, and publishes the projection-ready NVFP4 tile.
There is no completed BF16 dQ write or reread and no standalone dQ quantizer;
one global prior-lane read remains because up to 32 key-tile CTAs contribute
to a query tile.

The fused GQA reducer now performs the corresponding K/V publication.  For a
16-value block \(b\), with global decode scale \(G\), it computes

\[
  a_b=\max_{i\in b}|x_i|,\qquad
  s_b=\operatorname{E4M3}\!\left(\frac{a_b}{6G}\right),\qquad
  z_i=\operatorname{E2M1}\!\left(\frac{x_i}{Gs_b}\right).
\]

dK is inverse-RoPE transformed before publication.  dV removes the \(2^8\)
probability lift in the same reducer.  The resulting K/V payload and scale
bytes match the standalone packer exactly, and their BF16 outputs need not be
materialized in the owner path.

Directly issuing projection MMA from every attention owner is not a useful
next rung under the current topology.  The stacked QKV reduction has

\[
  K_{\mathrm{proj}}=(32+8+8)\,128=6144
\]

and therefore 48 independent head contributors to every dX tile:

\[
  dX=\sum_{h=1}^{48} dG_h W_h^T.
\]

Projecting in each owner replaces one compact QKV operand with 48 BF16/FP32 dX
partials or atomics.  Holding the complete dX accumulator across all owners is
outside the available CTA-cluster/TMEM ownership.  A different query-owner or
projection-N multicast topology is required to remove the last prior lane;
adding another barrier to the current key-owner topology cannot do it.

## dV numerical defect and repair

The previous FP8 dV path flattened the score-register fragment into the PdO
TMEM-store fragment.  Those fragments have different lane/value maps for FP8.
The effective query coordinate was

\[
  q' = \operatorname{swapbits}_{4,5}(q),
\]

which swaps rows 16--31 with 32--47 inside each 64-row group.  Constant dO hid
the problem, while random dO reduced dV cosine to about 0.961.

The retained fix publishes the lifted FP8 probability through a
coordinate-preserving 128-bit register-to-shared copy and feeds PdO from that
shared layout.  It changes the represented-probability dV error from a layout
error into normal BF16 accumulation error:

| Check | Cosine | Relative L2 |
|---|---:|---:|
| Low-precision dV vs true BF16 | 0.999325 | 0.036745 |
| Analytic represented FP8 dV vs true BF16 | 0.999333 | 0.036531 |
| Kernel dV vs analytic represented FP8 dV | 0.999997 | 0.002380 |
| Kernel dV partials vs analytic represented partials | 0.999999 | 0.001672 |

Thus essentially all remaining dV error is now the selected FP8
representation, not an indexing or accumulation defect.

## Probability publication ladder

| Variant | S4096 kernel time (us) | Result |
|---|---:|---|
| Old flat FP8 register-to-TMEM map | 280.542 | fast but wrong |
| Scalar coordinate-correct shared publication | 424.823 | correct, rejected |
| 128-bit coordinate-correct shared operand | **303.524** | correct, retained |
| Shared exchange + scalar reload + TMEM operand | 475.752 | rejected |
| Shared exchange + co-tiled vector reload + TMEM operand | 307.248 | coordinate-aware candidate, 1.2% slower |

The co-tiled TMEM route is useful negative evidence: restoring the TMEM PdO
operand recovers nearly all of the scalar-reload loss, but its extra shared
read plus register-to-TMEM store remains slightly slower than consuming the
correct shared tile directly.  Its native PdO shared partition is
coordinate-aware; because it already lost on timing, it was rejected before a
full numerical gate.  The remaining 23 us versus the incorrect path is
therefore the price of the required cross-warp permutation, not an extra
pipeline stage.

Two dO stages are retained.  Three stages measured 494.255 us in the same
compact-dQ configuration, and keeping full FP32 P live for dS measured
477.245 us.  Reusing the already-lifted FP8 P fragment is essential for the
128-register schedule.

## Final gradient quality

Compared with a true BF16 forward and BF16 backward:

| Tensor/path | Cosine | Relative L2 |
|---|---:|---:|
| dQ | 0.997322 | 0.073186 |
| dK | 0.997024 | 0.077133 |
| dV | 0.999325 | 0.036745 |
| NVFP4 QKV projection | 0.990454 | 0.138112 |
| Owner projection vs materialized NVFP4 projection | 1.000000 | 0.000787 |

The remaining end-to-end numerical weakness is the NVFP4 projection, not the
attention dV path or owner publication.  Two projection rungs were measured
on the same real gradients:

| Prepacked projection format | Time (us) | Cosine vs BF16 | Relative L2 |
|---|---:|---:|---:|
| NVFP4 | 52.864--56.000 | 0.990454 | 0.138112 |
| 16-point Hadamard + NVFP4 | 56.000 | 0.990333 | 0.139040 |
| Row-scaled E4M3 FP8 | 109.248 | **0.998645** | **0.052034** |

The orthogonal transform is rejected: it adds work and is slightly less
accurate.  Producer-native row-scaled FP8 is the credible high-accuracy mode.
It costs about 53--56 us over prepacked NVFP4 projection, but remains a useful
training trade-off if its row quantization is fused into the final projection
epilogue/reducer.  Standalone FP8 quantization is not competitive.

## Retained defaults and reproduction

- FP8 dO pipeline depth: 2 for D128.
- P-to-dV: lifted E4M3, coordinate-correct 128-bit shared publication.
- dS: reuse lifted FP8 P; exact/native exp2 in the reported run.
- dQ/dK/dV output boundary: BF16 for the regular path.
- Projection default: materialized NVFP4 QKV pack and GEMM.
- Accuracy option: producer-native row-scaled FP8 QKV projection.
- Owner-native Q/K/V publication remains an opt-in topology experiment.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python3 tk_fa4/lowp_fa4_bwd/profile_gqa_d128_chain.py \
  --forward-extension /tmp/_C_tk_causal_gqa_nvfp4_fp8pv_exact_builder_sm100.cpython-312-aarch64-linux-gnu.so \
  --forward-module _C_tk_causal_gqa_nvfp4_fp8pv_exact_builder_sm100 \
  --true-bf16-reference --owner-only --direct-workspace-stats \
  --exp2-period 0 --reuse-quantized-p --owner-fuse-kv \
  --diagnose-dv --diagnose-projection-formats \
  --warmups 3 --samples 7 --output final_squeeze.json
```

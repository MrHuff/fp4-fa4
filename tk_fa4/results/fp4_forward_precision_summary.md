# FP4 Forward Precision Summary

## Scope

This report summarizes the FP4 forward numerics study in [fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py). It combines:

- A short-sequence, distribution-robust study over `7` families and `5` seeds at `S in {1024, 1536, 2048, 4096}`.
- A BF16-exact `P_mode` study isolating `PV` at `S in {1024, 2048, 4096, 8192}`.
- A long-sequence Gaussian extension to `S = 32768` for `QK-only`, `PV-only`, and full `QK+PV` using `B=1, H=1` so that the exact BF16 reference remains tractable.

Every quantization entry point in the current study now rejects non-finite inputs before quantization, so the rows reported here are finite-by-construction unless explicitly noted otherwise.
The underlying long-sequence rows also tracked `MSE`, `RMSE`, and max relative error; the tables below focus on `MAE`, max absolute error, and `LSE` drift because those were the most readable summary signals.

## Mode Definitions

We study three forward decompositions.

Let

- `S = (Q K^T) / sqrt(d)` be the pre-softmax score matrix.
- `M(S)` be the causal mask.
- `Q_b(·)` be backend `b` quantization.
- `G_b(·, ·)` be backend-matched low-precision GEMM.

### QK-only

Only `QK` is quantized:

`S_q = G_qk(Q_q, K_q)`

`P = softmax(M(S_q))`

`O = P V_bf16`

This isolates the error coming from the `QK` backend.

### PV-only

`Q` and `K` stay BF16 exact, so `S` is exact. Then one of three `P_mode`s is applied.

`stored_p`

`P = softmax(M(S))`

`O = G_pv(Q_pv(P), Q_pv(V))`

`live_direct`

For each tile `t`, with running row max `m` and running row sum `l`:

`m_new = max(m, max(S_t))`

`alpha = exp(m - m_new)`

`E_t = exp(M(S_t) - m_new)`

`O_acc <- alpha * O_acc + G_pv(Q_pv(E_t), Q_pv(V_t))`

`l <- alpha * l + sum(E_t)`

`O = O_acc / l`

This quantizes the unnormalized live softmax tile and divides by the final row sum at the end.

`live_sa3_experimental`

`m_new = max(m, max(S_t))`

`alpha = exp(m - m_new)`

`E_t = exp(M(S_t) - m_new)`

`l_new = alpha * l + sum(E_t)`

`c = (alpha * l) / l_new`

`P_t = E_t / l_new`

`O <- c * O + G_pv(Q_pv(P_t), Q_pv(V_t))`

This is the SageAttention3-like experimental path: it quantizes the normalized per-tile probability and rescales the running output online.

### Full QK+PV

Both sides are quantized:

`S_q = G_qk(Q_qk(Q), Q_qk(K)) / sqrt(d)`

Then `stored_p`, `live_direct`, or `live_sa3_experimental` is applied on top of `S_q` using the chosen `PV` backend.

## Current Quantization Procedures

### QK backends

- `v5`: BF16 rows are padded to `128` multiples, quantized with the real v5 quantizer, and multiplied with the real NVFP4 v5 GEMM path.
- `localcta`: BF16 rows are padded to `128` multiples, quantized with the real localCTA quantizer, and multiplied with the real localCTA NVFP4 GEMM path.
- `mxfp4_v3`: BF16 rows are padded as needed and quantized with the real MXFP4 v3 path, then multiplied with the real TK `mxfp4_gemm` path.

In the orchestrated precision study this happens through `_quantize_rows_2d_forward_backend(...)` and `_backend_gemm_quantized(...)` in [fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py).

### PV backends

- `localcta`: `P` and `V` are quantized with the real localCTA prepared quantizer and multiplied through the localCTA PV path.
- `mxfp4_v3`: `P` and `V` are quantized with the real MXFP4 v3 path and multiplied through the real TK `mxfp4_gemm` path.
- `v5`: available in the shorter stored-`P` study, but the current precision winner for long `PV` is `mxfp4_v3`, so the long extension focuses on the stronger pairings.

### Older live-kernel prepack path

The repo also still contains the older live-kernel `Q/K` prepack adapters (`_pack_live_fp4_qk_v5_from_bf16`, `_pack_live_fp4_qk_localcta_from_bf16`). Those are useful for the fused-kernel interface, but the precision matrix reported here uses the cleaner orchestrated backend path above so that `QK`, `PV`, and `P_mode` can be swept independently.

## Short-Sequence Distribution-Robust Results

These rows use `B=1, H=16`, `7` distribution families, and `5` seeds per family.

| Slice | 1024 winner | 2048 winner | Note |
|---|---|---|---|
| QK-only | `localcta` | `localcta` | `localcta` is the robust default for `QK`; `v5` wins some heavier-tail subcases. |
| PV-only | `live_direct / localcta` | `stored_p / mxfp4_v3` | `live_direct/localcta` wins at `1024`; `stored_p/mxfp4_v3` wins at `2048`. |
| Full QK+PV | `localcta / live_direct / localcta` | median-best `v5 / stored_p / mxfp4_v3` | At `2048`, `stored_p/mxfp4_v3` clearly wins; `localcta` wins more families while `v5` edges the median slightly. |

### Family-level crossover

- At `1024`, `localcta / live_direct / localcta` won `6/7` family distributions.
- At `1536`, the flip already happened: every family preferred `stored_p / mxfp4_v3`.
- At `2048`, `stored_p / mxfp4_v3` still won every family, with `QK=localcta` winning `5/7` families and `QK=v5` winning `2/7` heavier-tail families.
- At `4096`, the same `stored_p / mxfp4_v3` regime remained dominant.

### BF16-exact QK p-mode sweep

| S | Random winner | Random MAE | Zero-QK winner | Zero-QK MAE |
|---|---|---:|---|---:|
| 1024 | `live_direct / localcta` | 0.002536 | `live_direct / localcta` | 0.001718 |
| 2048 | `stored_p / mxfp4_v3` | 0.001997 | `stored_p / mxfp4_v3` | 0.002255 |
| 4096 | `stored_p / mxfp4_v3` | 0.001425 | `stored_p / mxfp4_v3` | 0.001590 |
| 8192 | `stored_p / mxfp4_v3` | 0.001006 | `stored_p / mxfp4_v3` | 0.001134 |

The BF16-exact `QK` sweep says the same thing as the family study: `live_direct/localcta` is best at `1024`, while `stored_p/mxfp4_v3` takes over from `2048` onward.

## Long-Sequence Extension To 32K

These rows use `B=1, H=1`, Gaussian random inputs plus a zero-QK control, and a single seed per case. This keeps the exact BF16 reference tractable at `32K`. These timings are from the Python orchestrator, so they are numerics diagnostics rather than kernel-throughput claims.

### QK-only (BF16-exact PV)

#### Gaussian random

| S | `v5` MAE | `localcta` MAE | `mxfp4_v3` MAE | Best |
|---|---:|---:|---:|---|
| 1024 | 1.647802e-04 | 1.603288e-04 | 1.962737e-04 | `localcta` |
| 2048 | 1.193312e-04 | 1.140097e-04 | 1.404654e-04 | `localcta` |
| 4096 | 8.525315e-05 | 8.213075e-05 | 9.947240e-05 | `localcta` |
| 8192 | 6.041084e-05 | 5.914906e-05 | 7.151438e-05 | `localcta` |
| 16384 | 4.289452e-05 | 4.202718e-05 | 5.049152e-05 | `localcta` |
| 32768 | 3.031128e-05 | 3.043011e-05 | 3.600770e-05 | `v5` |

#### Zero-QK control

| S | `v5` MAE | `localcta` MAE | `mxfp4_v3` MAE | Best |
|---|---:|---:|---:|---|
| 1024 | 2.014405e-05 | 2.014405e-05 | 2.014405e-05 | `v5` |
| 2048 | 1.449950e-05 | 1.449950e-05 | 1.449950e-05 | `v5` |
| 4096 | 9.661345e-06 | 9.661345e-06 | 9.661345e-06 | `v5` |
| 8192 | 7.496098e-06 | 7.496098e-06 | 7.496098e-06 | `v5` |
| 16384 | 5.025996e-06 | 5.025996e-06 | 5.025996e-06 | `v5` |
| 32768 | 3.706699e-06 | 3.706699e-06 | 3.706699e-06 | `v5` |

`localcta` stays best on Gaussian random through `16K`, while `v5` edges it very slightly at `32K`. On the zero-QK control the three `QK` backends are effectively tied, which is the behavior we want.

### PV-only (BF16-exact QK)

#### Gaussian random

| S | `live_direct / localcta` MAE | `stored_p / mxfp4_v3` MAE | `live_sa3_experimental / localcta` MAE | Best |
|---|---:|---:|---:|---|
| 1024 | 0.002623 | 0.002722 | 0.009315 | `live_direct / localcta` |
| 2048 | 0.010536 | 0.002028 | 0.010536 | `stored_p / mxfp4_v3` |
| 4096 | 0.007547 | 0.001421 | 0.007547 | `stored_p / mxfp4_v3` |
| 8192 | 0.005089 | 9.570779e-04 | 0.005089 | `stored_p / mxfp4_v3` |
| 16384 | 0.003634 | 7.042127e-04 | 0.003634 | `stored_p / mxfp4_v3` |
| 32768 | 0.002612 | 4.882013e-04 | 0.002612 | `stored_p / mxfp4_v3` |

#### Zero-QK control

| S | `live_direct / localcta` MAE | `stored_p / mxfp4_v3` MAE | `live_sa3_experimental / localcta` MAE | Best |
|---|---:|---:|---:|---|
| 1024 | 0.001851 | 0.003246 | 0.008771 | `live_direct / localcta` |
| 2048 | 0.010447 | 0.002329 | 0.010447 | `stored_p / mxfp4_v3` |
| 4096 | 0.006950 | 0.001506 | 0.006950 | `stored_p / mxfp4_v3` |
| 8192 | 0.005313 | 0.001149 | 0.005313 | `stored_p / mxfp4_v3` |
| 16384 | 0.003544 | 7.651994e-04 | 0.003544 | `stored_p / mxfp4_v3` |
| 32768 | 0.002593 | 5.626442e-04 | 0.002593 | `stored_p / mxfp4_v3` |

The `PV` story is clean: `live_direct/localcta` is best at `1024`, then `stored_p/mxfp4_v3` takes over and stays best through `32K`. In the long Gaussian extension, `live_sa3_experimental/localcta` numerically collapses onto `live_direct/localcta` rather than beating it.

### Full QK+PV

#### Gaussian random

| S | Best `live_direct/localcta` over QK | Best `stored_p/mxfp4_v3` over QK | Best `live_sa3_experimental/localcta` over QK | Overall best |
|---|---|---|---|---|
| 1024 | `localcta`: 0.002612 | `localcta`: 0.002721 | `mxfp4_v3`: 0.009268 | `localcta / live_direct / localcta` |
| 2048 | `v5`: 0.010536 | `localcta`: 0.002021 | `v5`: 0.010536 | `localcta / stored_p / mxfp4_v3` |
| 4096 | `v5`: 0.007547 | `localcta`: 0.001421 | `v5`: 0.007547 | `localcta / stored_p / mxfp4_v3` |
| 8192 | `localcta`: 0.005089 | `localcta`: 9.553850e-04 | `localcta`: 0.005089 | `localcta / stored_p / mxfp4_v3` |
| 16384 | `localcta`: 0.003634 | `v5`: 7.016947e-04 | `localcta`: 0.003634 | `v5 / stored_p / mxfp4_v3` |
| 32768 | `localcta`: 0.002612 | `localcta`: 4.872253e-04 | `localcta`: 0.002612 | `localcta / stored_p / mxfp4_v3` |

#### Zero-QK control

| S | Best `live_direct/localcta` over QK | Best `stored_p/mxfp4_v3` over QK | Best `live_sa3_experimental/localcta` over QK | Overall best |
|---|---|---|---|---|
| 1024 | `v5`: 0.001851 | `v5`: 0.003246 | `v5`: 0.008771 | `v5 / live_direct / localcta` |
| 2048 | `v5`: 0.010447 | `v5`: 0.002329 | `v5`: 0.010447 | `v5 / stored_p / mxfp4_v3` |
| 4096 | `v5`: 0.006950 | `v5`: 0.001506 | `v5`: 0.006950 | `v5 / stored_p / mxfp4_v3` |
| 8192 | `v5`: 0.005313 | `v5`: 0.001149 | `v5`: 0.005313 | `v5 / stored_p / mxfp4_v3` |
| 16384 | `localcta`: 0.003544 | `v5`: 7.651994e-04 | `localcta`: 0.003544 | `v5 / stored_p / mxfp4_v3` |
| 32768 | `localcta`: 0.002593 | `localcta`: 5.626442e-04 | `localcta`: 0.002593 | `localcta / stored_p / mxfp4_v3` |

The full forward extension agrees with the axis studies:

- `1024`: `localcta / live_direct / localcta` is best on Gaussian random; `live_direct/localcta` is still best on the zero-QK control and the three QK backends are tied there.
- `2048+`: `stored_p / mxfp4_v3` is best end-to-end.
- At `16384` and `32768`, all reported full rows were finite in fresh one-shot processes; the earlier long chained failures were session-poisoning after an `unspecified launch failure`, not intrinsic non-finite numerics for the winning configs.

## Practical Recommendations

- Precision-first default around `S ~= 1024`: `QK=localcta`, `P_mode=live_direct`, `PV=localcta`.
- Precision-first default from `S >= 1536`: `P_mode=stored_p`, `PV=mxfp4_v3`.
- For `QK` in the longer-sequence regime, `localcta` is the safest default, while `v5` is a real alternative and sometimes slightly better on heavier-tail cases or at `32K` in the Gaussian extension.
- `mxfp4_v3` is a real viable `QK` backend now, but in the current results it trails `localcta`/`v5` slightly on pure `QK` accuracy.
- `live_sa3_experimental` is numerically viable with `localcta`, but it is not beating either `live_direct/localcta` at short lengths or `stored_p/mxfp4_v3` at longer lengths.

## Caveats

- The short-sequence distribution study is the robust part of the report: `7` families, `5` seeds, `B=1`, `H=16`.
- The long `32K` extension is intentionally lighter: Gaussian random plus zero-QK control, `B=1`, `H=1`, one seed per case.
- Long-sequence timings come from the Python orchestrator and should not be interpreted as kernel throughput numbers.
- `stored_p / localcta` and `live_sa3_experimental / mxfp4_v3` were already clearly dominated in the shorter PV-only study, so the long extension focuses on the stronger pairings per `P_mode`.

See also the widened long-ladder Cartesian extension in [fp4_forward_precision_full27_extension.md](/workspace/codebases/fp4_matmul/tk_fa4/results/fp4_forward_precision_full27_extension.md).

## Files Used

- Main study implementation: [fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py)
- Report: [fp4_forward_precision_summary.md](/workspace/codebases/fp4_matmul/tk_fa4/results/fp4_forward_precision_summary.md)

# FP4 Forward Precision Full27 Extension

## Scope

This companion report widens the forward study from the focused candidate subset in [fp4_forward_precision_summary.md](/workspace/codebases/fp4_matmul/tk_fa4/results/fp4_forward_precision_summary.md) to the full Cartesian configuration ladder on the lighter long-sequence regime:

- `QK_backend in {v5, localcta, mxfp4_v3}`
- `P_mode in {live_direct, live_sa3_experimental, stored_p}`
- `PV_backend in {v5, localcta, mxfp4_v3}`
- Inputs: Gaussian random plus zero-QK Gaussian control
- `S in {1024, 1536, 2048, 4096, 8192, 16384, 32768}` for full `QK+PV`
- `S in {1024, 2048, 4096, 8192, 16384, 32768}` for `QK-only` and `PV-only`
- `B=1, H=1`, one seed per case

This extension is precision-first and uses the same backend definitions and equations from the main summary. The purpose here is to see what changes once every real backend combination is allowed, not to replace the shorter robust family/seed study.

## Coverage

- `QK-only`: 36 rows, 0 errors, 0 non-finite rows
- `PV-only`: 108 rows, 0 errors, 0 non-finite rows
- Full `QK+PV`: 378 rows, 0 errors, 0 non-finite rows

All rows reported below were finite on both output and `LSE`. Quantization entry points still reject non-finite inputs before quantization, so these rows are finite-by-construction rather than silently repaired after the fact.

## QK-only Winners

| S | Input | Best QK | MAE | Max Abs | MSE | RMSE | Max Rel | LSE Max Abs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1024 | `random_live_fp4` | `localcta` | 1.603288e-04 | 0.008179 | 7.154257e-08 | 2.674744e-04 | 5.845069 | 0.011727 |
| 1024 | `zero_qk_random_v` | `v5` | 2.014405e-05 | 0.001953 | 3.838522e-09 | 6.195581e-05 | 0.007812 | 9.536743e-07 |
| 2048 | `random_live_fp4` | `localcta` | 1.140097e-04 | 0.005859 | 3.461798e-08 | 1.860591e-04 | 2.548218 | 0.007621 |
| 2048 | `zero_qk_random_v` | `v5` | 1.449950e-05 | 0.001953 | 2.271478e-09 | 4.766003e-05 | 0.007812 | 9.536743e-07 |
| 4096 | `random_live_fp4` | `localcta` | 8.213075e-05 | 0.006836 | 2.148102e-08 | 1.465641e-04 | 2.618789 | 0.009404 |
| 4096 | `zero_qk_random_v` | `v5` | 9.661345e-06 | 0.001953 | 1.073686e-09 | 3.276715e-05 | 0.007812 | 9.536743e-07 |
| 8192 | `random_live_fp4` | `localcta` | 5.914906e-05 | 0.004395 | 1.222042e-08 | 1.105460e-04 | 1.895085 | 0.005529 |
| 8192 | `zero_qk_random_v` | `v5` | 7.496098e-06 | 0.003906 | 7.192974e-10 | 2.681972e-05 | 0.007812 | 9.536743e-07 |
| 16384 | `random_live_fp4` | `localcta` | 4.202718e-05 | 0.009766 | 7.134122e-09 | 8.446373e-05 | 2.513885 | 0.013629 |
| 16384 | `zero_qk_random_v` | `v5` | 5.025996e-06 | 0.001953 | 3.499798e-10 | 1.870775e-05 | 0.007812 | 9.536743e-07 |
| 32768 | `random_live_fp4` | `localcta` | 2.966235e-05 | 0.005737 | 3.127267e-09 | 5.592197e-05 | 3.14966 | 0.013921 |
| 32768 | `zero_qk_random_v` | `v5` | 3.706699e-06 | 0.001953 | 2.101748e-10 | 1.449741e-05 | 0.007812 | 9.536743e-07 |

Takeaway: `localcta` is the best `QK` backend on Gaussian random all the way through `32768` in this extension, while the zero-QK control keeps the three `QK` backends effectively tied and lets `v5` win the tiebreaks numerically.

## PV-only Winners

| S | Input | Best `P_mode / PV` | MAE | Max Abs | MSE | RMSE | Max Rel | LSE Max Abs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1024 | `random_live_fp4` | `live_direct / localcta` | 0.002623 | 0.257812 | 2.238029e-05 | 0.004731 | 74.509621 | 1.430511e-06 |
| 1024 | `zero_qk_random_v` | `live_direct / v5` | 0.001836 | 0.148438 | 1.091670e-05 | 0.003304 | 24.780272 | 9.536743e-07 |
| 2048 | `random_live_fp4` | `stored_p / mxfp4_v3` | 0.002028 | 0.168945 | 1.401683e-05 | 0.003744 | 24.963377 | 1.430511e-06 |
| 2048 | `zero_qk_random_v` | `stored_p / mxfp4_v3` | 0.002329 | 0.171875 | 1.663795e-05 | 0.004079 | 26.22032 | 9.536743e-07 |
| 4096 | `random_live_fp4` | `stored_p / mxfp4_v3` | 0.001421 | 0.15625 | 7.115636e-06 | 0.002668 | 31.764982 | 1.907349e-06 |
| 4096 | `zero_qk_random_v` | `stored_p / mxfp4_v3` | 0.001506 | 0.175781 | 8.125334e-06 | 0.00285 | 30.651091 | 9.536743e-07 |
| 8192 | `random_live_fp4` | `stored_p / mxfp4_v3` | 9.570779e-04 | 0.191406 | 3.612523e-06 | 0.001901 | 37.517548 | 1.907349e-06 |
| 8192 | `zero_qk_random_v` | `stored_p / mxfp4_v3` | 0.001149 | 0.21875 | 5.283732e-06 | 0.002299 | 27.368544 | 9.536743e-07 |
| 16384 | `random_live_fp4` | `stored_p / mxfp4_v3` | 7.042105e-04 | 0.152344 | 2.060993e-06 | 0.001436 | 23.742674 | 1.907349e-06 |
| 16384 | `zero_qk_random_v` | `stored_p / mxfp4_v3` | 7.651994e-04 | 0.166016 | 2.519579e-06 | 0.001587 | 28.884886 | 9.536743e-07 |
| 32768 | `random_live_fp4` | `stored_p / mxfp4_v3` | 4.882013e-04 | 0.199219 | 1.068432e-06 | 0.001034 | 59.982296 | 2.861023e-06 |
| 32768 | `zero_qk_random_v` | `stored_p / mxfp4_v3` | 5.626442e-04 | 0.195312 | 1.433664e-06 | 0.001197 | 13.568877 | 9.536743e-07 |

Takeaway: `live_direct / localcta` still wins at `1024`, and the crossover to `stored_p / mxfp4_v3` still happens by `2048`. In the widened ladder, `stored_p / v5` and `stored_p / localcta` never recover once `stored_p / mxfp4_v3` takes over.

## Full QK+PV Winners

| S | Input | Best Full Config | MAE | Max Abs | MSE | RMSE | Max Rel | LSE Max Abs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1024 | `random_live_fp4` | `localcta / live_direct / localcta` | 0.002612 | 0.257812 | 2.209653e-05 | 0.004701 | 74.021339 | 0.011727 |
| 1024 | `zero_qk_random_v` | `v5 / live_direct / v5` | 0.001836 | 0.148438 | 1.091670e-05 | 0.003304 | 24.780272 | 9.536743e-07 |
| 1536 | `random_live_fp4` | `localcta / stored_p / mxfp4_v3` | 0.002253 | 0.214844 | 1.766385e-05 | 0.004203 | 24.072464 | 0.012811 |
| 1536 | `zero_qk_random_v` | `v5 / stored_p / mxfp4_v3` | 0.002526 | 0.234375 | 2.110987e-05 | 0.004595 | 19.107819 | 9.536743e-07 |
| 2048 | `random_live_fp4` | `localcta / stored_p / mxfp4_v3` | 0.002021 | 0.168945 | 1.388299e-05 | 0.003726 | 24.963377 | 0.007621 |
| 2048 | `zero_qk_random_v` | `v5 / stored_p / mxfp4_v3` | 0.002329 | 0.171875 | 1.663795e-05 | 0.004079 | 26.22032 | 9.536743e-07 |
| 4096 | `random_live_fp4` | `localcta / stored_p / mxfp4_v3` | 0.001421 | 0.15625 | 7.135716e-06 | 0.002671 | 31.764982 | 0.009404 |
| 4096 | `zero_qk_random_v` | `v5 / stored_p / mxfp4_v3` | 0.001506 | 0.175781 | 8.125334e-06 | 0.00285 | 30.651091 | 9.536743e-07 |
| 8192 | `random_live_fp4` | `localcta / stored_p / mxfp4_v3` | 9.553850e-04 | 0.191406 | 3.595644e-06 | 0.001896 | 36.219856 | 0.005529 |
| 8192 | `zero_qk_random_v` | `v5 / stored_p / mxfp4_v3` | 0.001149 | 0.21875 | 5.283732e-06 | 0.002299 | 27.368544 | 9.536743e-07 |
| 16384 | `random_live_fp4` | `v5 / stored_p / mxfp4_v3` | 7.016947e-04 | 0.152344 | 2.022321e-06 | 0.001422 | 23.742674 | 0.013065 |
| 16384 | `zero_qk_random_v` | `v5 / stored_p / mxfp4_v3` | 7.651994e-04 | 0.166016 | 2.519579e-06 | 0.001587 | 28.884886 | 9.536743e-07 |
| 32768 | `random_live_fp4` | `localcta / stored_p / mxfp4_v3` | 4.872253e-04 | 0.199219 | 1.066740e-06 | 0.001033 | 59.982296 | 0.013921 |
| 32768 | `zero_qk_random_v` | `v5 / stored_p / mxfp4_v3` | 5.626442e-04 | 0.195312 | 1.433664e-06 | 0.001197 | 13.568877 | 9.536743e-07 |

## Best Config Per `P_mode` On Gaussian Random

| S | Best `live_direct` | MAE | Best `live_sa3_experimental` | MAE | Best `stored_p` | MAE |
|---|---|---:|---|---:|---|---:|
| 1024 | `localcta / localcta` | 0.002612 | `v5 / v5` | 0.009146 | `localcta / mxfp4_v3` | 0.002721 |
| 1536 | `localcta / v5` | 0.011403 | `localcta / v5` | 0.011403 | `localcta / mxfp4_v3` | 0.002253 |
| 2048 | `v5 / v5` | 0.010536 | `v5 / v5` | 0.010536 | `localcta / mxfp4_v3` | 0.002021 |
| 4096 | `v5 / v5` | 0.007547 | `v5 / v5` | 0.007547 | `localcta / mxfp4_v3` | 0.001421 |
| 8192 | `localcta / v5` | 0.005089 | `localcta / v5` | 0.005089 | `localcta / mxfp4_v3` | 9.553850e-04 |
| 16384 | `mxfp4_v3 / v5` | 0.003634 | `mxfp4_v3 / v5` | 0.003634 | `v5 / mxfp4_v3` | 7.016947e-04 |
| 32768 | `localcta / v5` | 0.002612 | `localcta / v5` | 0.002612 | `localcta / mxfp4_v3` | 4.872253e-04 |

This makes the backend split clearer:

- `live_direct` prefers `PV=v5` in this widened Gaussian extension once `PV=v5` is allowed, but it is still dominated by `stored_p / mxfp4_v3` from `1536` onward.
- `live_sa3_experimental` also prefers `PV=v5` here and stays numerically tied with `live_direct` in this orchestrated path, but it remains dominated by `stored_p / mxfp4_v3`.
- `stored_p` strongly prefers `PV=mxfp4_v3` across the whole longer ladder.

## Practical Read

- Around `1024`:
  - Gaussian random: `localcta / live_direct / localcta`
  - Zero-QK control: `v5 / live_direct / v5`
- From `1536` through `4096`: `stored_p / mxfp4_v3` is the clear winner end-to-end.
- From `8192` through `32768`: `stored_p / mxfp4_v3` remains the precision-first choice.
- Within that long `stored_p / mxfp4_v3` regime:
  - `localcta` is the best `QK` default on Gaussian random at `1536`, `2048`, `4096`, `8192`, and `32768`.
  - `v5` slightly wins some longer points, including the zero-QK control and Gaussian random at `16384`.
  - `mxfp4_v3` is real and viable as a `QK` backend, but it stays slightly behind `localcta` and `v5` on the current long Gaussian ladder.

## Relationship To The Main Summary

The main summary remains the decision document for the shorter, more robust family/seed study. This extension answers a narrower question: once we allow every real `QK x P_mode x PV` combination, do any of the previously omitted backend pairings beat the incumbents on the long Gaussian ladder?

Current answer: no. The widened 27-combo sweep reinforces the same transition already visible in the focused study:

- short: `live_direct` is best
- longer: `stored_p / mxfp4_v3` is best
- `QK` choice within the longer regime is still a close `localcta` vs `v5` question, with `localcta` being the safer default and `v5` winning some specific longer/control cases

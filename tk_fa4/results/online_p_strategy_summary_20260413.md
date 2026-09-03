# Online `P` Strategy Summary

## Scope

- Focus: online `P` quantization strategies, not mixed backend-vs-strategy comparisons.
- Trusted comparison mode: `qdq_proxy`.
- Main online strategies compared:
  - `live_direct`
  - `live_sa3_baseline`
  - `live_localcta_cta_amax_experimental`

## Key Results

- Corrected heavy-tail strategy report:
  - [pv_only_online_strategy_distribution_qdq_20260413T000755Z/README.md](./pv_only_online_strategy_distribution_qdq_20260413T000755Z/README.md)
- Longer-sequence addendum at `S=16384`:
  - [pv_only_online_strategy_16384_qdq_20260413T001453Z/README.md](./pv_only_online_strategy_16384_qdq_20260413T001453Z/README.md)
- Kernel-validity / real-kernel-vs-proxy checkpoint:
  - [pv_only_online_dualreport_inproc_20260412T231317Z/README.md](./pv_only_online_dualreport_inproc_20260412T231317Z/README.md)

## Conclusions

- For the canonical online three-way comparison on heavy-tailed and outlier-prone inputs (`heads=8`, `S in {2048,4096,8192}`), the winner is:
  - `localcta_cta_amax` = `live_localcta_cta_amax_experimental / localcta`
- In the canonical report, `localcta_cta_amax` wins `18 / 18` cases.
- `sa3_baseline` is consistently second in the canonical report.
- `mxfp4_online` = `live_direct / mxfp4_v3` is consistently worst in the canonical report.

## Backend-Coupled Read

- On `localcta`, the best online strategy is usually `live_localcta_cta_amax_experimental`.
- On `mxfp4_v3`, `live_sa3_baseline` usually beats the other online strategies.
- So the best online `P` quantizer is backend-coupled:
  - localCTA/NVFP4 consumer: prefer CTA-local `amax`
  - current MXFP4 consumer: prefer SA3-style scaling

## `S=16384` Addendum

- At `S=16384`, `mxfp4_v3` still prefers `live_sa3_baseline` for every tested distribution.
- At `S=16384`, `localcta` prefers:
  - `live_direct` for `gaussian`, `laplace`, `student_t3`, `student_t5`, `signed_lognormal`
  - `live_localcta_cta_amax_experimental` for `gaussian_spikes`
- That suggests CTA-amax remains strong, but the best localCTA-side online strategy may shift with sequence length and distribution family.

## Caveats

- Higher-head real-kernel runs still have launch failures at some longer cases.
- `max_rel_diff_vs_bf16` is noisy on tiny references; use MAE, RMSE, and normalized MAE/RMSE as the ranking signals.

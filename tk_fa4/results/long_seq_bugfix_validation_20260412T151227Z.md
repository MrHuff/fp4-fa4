# Long-Sequence Bugfix Validation

This note records targeted validations after fixing long-sequence precision benchmark instability in `fp4_pv_experiments.py`.

## Fixes Applied

- Clear CUDA allocator state between heads for large combo runs (`seqlen >= 4096`).
- Offload BF16 reference outputs/LSE to CPU in precision subprocesses before running FP4 rows.
- Clear CUDA allocator state between subprocess warmup/timing passes.
- Use host-clock timing instead of CUDA event timing for precision subprocess rows at `seqlen >= 8192`.

## Validated Rows

- `PV-only`, `S=4096`, `random_live_fp4`, full 12-row matched case in one process:
  - all rows completed
  - `stored_p/mxfp4_v3` MAE `0.0014128480106592178`
  - all non-MX winners tied at `0.007401307113468647`

- `PV-only`, `S=8192`, `zero_qk_random_v`, subprocess:
  - `live_direct/v5` completed
  - MAE `0.005154965911060572`
  - LSE max abs diff `9.5367431640625e-07`

- `PV-only`, `S=8192`, `zero_qk_random_v`, subprocess:
  - `stored_p/mxfp4_v3` completed
  - MAE `0.0011234516277909279`
  - LSE max abs diff `9.5367431640625e-07`

- `PV-only`, `S=8192`, `random_live_fp4`, subprocess:
  - `live_direct/v5` completed
  - MAE `0.005258024204522371`
  - LSE max abs diff `1.9073486328125e-06`

- `PV-only`, `S=8192`, `random_live_fp4`, subprocess:
  - `stored_p/mxfp4_v3` completed
  - MAE `0.0009972455445677042`
  - LSE max abs diff `1.9073486328125e-06`

## Interpretation

- The previous `4096/8192` launch failures were primarily benchmark-harness stability issues, not proof that the FP4 rows themselves were invalid.
- At long sequence lengths, the matched rows now run again in the subprocess path for representative live and stored configurations.
- The numerics still point in the same direction as before: long-sequence `stored_p/mxfp4_v3` remains much better than live `P`.

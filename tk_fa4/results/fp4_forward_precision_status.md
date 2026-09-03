# FP4 Forward Precision Status

## Scope

This note is the high-level status page for the current FP4 forward numerics work. It ties together:

- the main precision summary in [fp4_forward_precision_summary.md](/workspace/codebases/fp4_matmul/tk_fa4/results/fp4_forward_precision_summary.md)
- the widened long-ladder Cartesian extension in [fp4_forward_precision_full27_extension.md](/workspace/codebases/fp4_matmul/tk_fa4/results/fp4_forward_precision_full27_extension.md)
- the helper and orchestrator code in [fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py)

The goal of the work so far has been precision, not raw throughput. The current study treats BF16 FA4 as the reference and compares:

- `QK-only`
- `PV-only`
- full `QK+PV`

with real backends:

- `QK_backend in {v5, localcta, mxfp4_v3}`
- `P_mode in {live_direct, live_sa3_experimental, stored_p}`
- `PV_backend in {v5, localcta, mxfp4_v3}`

## Current Bottom Line

- Around `S=1024`, the best precision-first full forward remains `localcta / live_direct / localcta` on Gaussian random inputs.
- On the zero-QK control at `1024`, the widened sweep prefers `v5 / live_direct / v5`.
- From `S=1536` onward, the winner flips and stays `stored_p / mxfp4_v3`.
- In that longer regime, `QK=localcta` is the safest Gaussian-random default, while `QK=v5` wins some control and longer-sequence points.
- `QK=mxfp4_v3` is real and viable, but is still slightly behind `localcta` and `v5` on the current ladders.
- `live_sa3_experimental` is finite and usable in the study, but it is still dominated numerically.

## Relevant Repo Files

### Main study and helpers

- [fp4_pv_experiments.py](/workspace/codebases/fp4_matmul/tk_fa4/fp4_pv_experiments.py)
  This is the main orchestrated precision-study surface.
  It now contains:
  - finite-only quantization guards
  - safe BF16 fallback reference handling
  - real `v5`, `localcta`, and `mxfp4_v3` backend plumbing
  - `benchmark_forward_precision_distribution_study(...)`
  - `benchmark_forward_precision_distribution_study_full27(...)`
  - `benchmark_qk_only_precision_matrix(...)`
  - `benchmark_pv_only_precision_matrix(...)`
  - `benchmark_forward_precision_matrix(...)`

### Reports

- [fp4_forward_precision_summary.md](/workspace/codebases/fp4_matmul/tk_fa4/results/fp4_forward_precision_summary.md)
  The main narrative summary:
  - equations for `QK-only`, `PV-only`, and full `QK+PV`
  - current quantization procedure for `QK` and `PV`
  - short robust family/seed conclusions
  - long Gaussian extension through `32K`

- [fp4_forward_precision_full27_extension.md](/workspace/codebases/fp4_matmul/tk_fa4/results/fp4_forward_precision_full27_extension.md)
  The widened long-ladder extension:
  - full `27`-combo `QK+PV`
  - full `9`-combo `PV-only`
  - `QK-only`
  - `MSE`, `RMSE`, and max relative error
  - finite-only coverage counts

### Earlier kernel-side experiment files

- [bf16_b300_mha_causal_fp4_pv_experiments.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal_fp4_experiments/bf16_b300_mha_causal_fp4_pv_experiments.cu)
  Sidecar experiment kernels for the streaming/live `P @ V` work.

- [bf16_b300_mha_causal_fp4.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal_fp4.cu)
  Production FP4 causal path that the study was repeatedly compared against.

- [bf16_b300_mha_causal.cu](/workspace/codebases/fp4_matmul/tk_fa4/b300_causal/bf16_b300_mha_causal.cu)
  BF16 causal baseline kernel used as the regular FA4 reference.

## Important Commands Used

### Sanity and compile

```bash
python3 -m py_compile tk_fa4/fp4_pv_experiments.py
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

### Widened full `QK+PV` Gaussian long ladder

These were run in split `P_mode` batches on separate GPUs using the temp driver at `/tmp/fp4_full27_sweep.py`.

```bash
python3 -u /tmp/fp4_full27_sweep.py --kind full --device cuda:1 --p-mode live_direct --out /tmp/full27_live_direct.json
python3 -u /tmp/fp4_full27_sweep.py --kind full --device cuda:2 --p-mode stored_p --out /tmp/full27_stored_p.json
python3 -u /tmp/fp4_full27_sweep.py --kind full --device cuda:3 --p-mode live_sa3_experimental --out /tmp/full27_live_sa3.json
```

### Widened `PV-only` Gaussian long ladder

```bash
python3 -u /tmp/fp4_full27_sweep.py --kind pv --device cuda:1 --p-mode live_direct --out /tmp/pvfull_live_direct.json
python3 -u /tmp/fp4_full27_sweep.py --kind pv --device cuda:2 --p-mode stored_p --out /tmp/pvfull_stored_p.json
python3 -u /tmp/fp4_full27_sweep.py --kind pv --device cuda:3 --p-mode live_sa3_experimental --out /tmp/pvfull_live_sa3.json
```

### `QK-only` Gaussian long ladder

```bash
python3 -u /tmp/fp4_full27_sweep.py --kind qk --device cuda:2 --out /tmp/qkfull_long.json
```

## Temporary Artifacts Used

These are not repo-tracked, but they were the immediate sweep outputs used to build the repo reports.

- `/tmp/full27_live_direct.json`
- `/tmp/full27_stored_p.json`
- `/tmp/full27_live_sa3.json`
- `/tmp/pvfull_live_direct.json`
- `/tmp/pvfull_stored_p.json`
- `/tmp/pvfull_live_sa3.json`
- `/tmp/qkfull_long.json`
- `/tmp/fp4_dist_aggregate_report.json`
- `/tmp/fp4_crossover_report.json`
- `/tmp/pmode_sweep_merged.json`
- `/tmp/fp4_postfix_chunked_summary.json`

## Results So Far

### Coverage and finiteness

The widened long Gaussian extension completed with:

- `QK-only`: `36` rows, `0` errors, `0` non-finite rows
- `PV-only`: `108` rows, `0` errors, `0` non-finite rows
- full `QK+PV`: `378` rows, `0` errors, `0` non-finite rows

Quantization now rejects non-finite inputs before quantization, so the study no longer silently passes through bad tensors.

### Best `QK` backend

- On Gaussian random, `localcta` is the best `QK` backend through the current long ladder.
- On the zero-QK control, the `QK` backends are effectively tied and `v5` wins the numeric tiebreaks.
- `mxfp4_v3` is real and competitive, but not yet the best `QK` choice.

### Best `PV` behavior

- `1024`: `live_direct / localcta` still wins.
- `2048+`: `stored_p / mxfp4_v3` clearly wins and stays best through `32768`.
- Once `stored_p / mxfp4_v3` takes over, the other stored-`P` `PV` backends do not recover.

### Best full forward

- `1024`, Gaussian random:
  - `localcta / live_direct / localcta`
- `1024`, zero-QK:
  - `v5 / live_direct / v5`
- `1536` to `32768`:
  - `stored_p / mxfp4_v3`

Inside that longer `stored_p / mxfp4_v3` regime:

- `localcta` is the better `QK` default on most Gaussian-random points
- `v5` wins some specific longer and control points
- `mxfp4_v3` remains a valid but slightly weaker `QK` option

## What Still Needs Fixing

### 1. Full robust short `27`-combo study

We now have helper support for the full `27`-combo matrix, but the fully robust short ladder over:

- `7` families
- `5` seeds
- `S in {1024, 1536, 2048, 4096}`

has not been exhaustively run for all `27` full combinations because it is much more expensive than the focused candidate study. The current robust short report is still candidate-based, while the widened `27`-combo extension is Gaussian random plus zero-QK control.

### 2. `live_sa3_experimental`

`live_sa3_experimental` is finite in the study, but it is still dominated numerically. It needs a real numerics improvement before it should be treated as a serious production candidate.

### 3. `live_direct` at longer sequence lengths

`live_direct` is no longer suffering from the earlier non-finite issues in the study path, but it is still clearly worse than `stored_p / mxfp4_v3` from `1536` onward. That is now a precision gap rather than a liveness bug.

### 4. Throughput interpretation

The orchestrated precision study is not a kernel-throughput benchmark. Timing fields are useful for relative bookkeeping, but not as the final performance story. Any speed claims still need dedicated kernel benchmarking.

### 5. `QK=mxfp4_v3`

The real MXFP4 `QK` path is wired and working, but it still trails `localcta` and `v5` slightly on the current ladders. It is viable, but not the current default recommendation.

## Recommended Next Steps

- Run the full robust short `27`-combo matrix in chunked batches until every family/seed case is covered.
- Keep `stored_p / mxfp4_v3` as the long-sequence precision default unless a better live path emerges.
- Improve `live_sa3_experimental` or explicitly demote it if it continues to stay dominated.
- Do a separate kernel-throughput-focused pass once the precision choice is settled.

## Current Recommendation

- Around `1024`: `localcta / live_direct / localcta`
- From `1536` onward: `stored_p / mxfp4_v3`
- Within the longer regime: default to `QK=localcta`, with `QK=v5` as the main alternative to keep an eye on

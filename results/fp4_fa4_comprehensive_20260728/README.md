# Comprehensive HAO comparison

This directory is the reproducible GB200 benchmark for:

- TK NVFP4 QK + NVFP4 PV, final approximate throughput policy with one
  N32-derived E4M3 P scale duplicated across two block-16 slots
- HAO native NVFP4 QK + NVFP4 PV
- TK NVFP4 QK + optimized FP8 PV control
- HAO native NVFP4 QK + FP8 PV
- HAO native BF16

The protocol uses HAO's seed-0 tensor factory, noncausal attention,
`triton.testing.do_bench`, 10 ms warmup, a 25 ms repetition window, median
timing, and 0.8 seconds of cooldown before each shape/provider process.
Q/K/V quantization is outside the timed forward.

Files:

- `manifest.json`: authoritative protocol, source identity, flags, GPU
  metadata, and per-case results
- `cases/`: one checkpointed JSON record per shape and TK variant
- `build_logs/`: compiler resource reports
- `analysis/summary.csv`: complete flat result table
- `analysis/statistics.json`: aggregate speedups, win counts, and accuracy
- `analysis/*.pdf`: report-ready plots

Output accuracy is measured against the same-shape HAO BF16 output. Relative
error means relative L2,
`||O_tk - O_bf16||_2 / ||O_bf16||_2`, equivalently
`RMSE(O_tk - O_bf16) / RMS(O_bf16)`. The FP8 records omit the reference RMS,
so the analyzer obtains it from the matching full-FP4 record, which uses the
same seed-0 inputs and BF16 reference.

The run is resumable:

```bash
cd tk_fa4/fp4_fa4_fwd
python3 hao_comprehensive_suite.py \
  --shape-set all --variant both \
  --warmup-ms 10 --rep-ms 25 \
  --cooldown-seconds 0.8 --seed 0 --gpu 0 \
  --output-dir \
  /workspace/codebases/pv/fp4_matmul/results/fp4_fa4_comprehensive_20260728 \
  --build-root /tmp/tk_hao_comprehensive_20260728
```

Analysis refuses an incomplete manifest:

```bash
python3 tk_fa4/fp4_fa4_fwd/analyze_hao_comprehensive.py \
  results/fp4_fa4_comprehensive_20260728/manifest.json \
  --output-dir results/fp4_fa4_comprehensive_20260728/analysis
```

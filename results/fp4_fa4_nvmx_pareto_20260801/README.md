# NVFP4/MXFP4 Pareto frontier (2026-08-01)

This directory records the retained NVFP4-QK/MXFP4-PV FlashAttention-4
frontier on one NVIDIA GB200. Kernel timing excludes dynamic Q/K/V
quantization and the optional K/V permutation.

## Headline

Shape: B1/S4096/H24/D128.

| Policy | Time (ms) | Matched HAO BF16 (ms) | Speedup |
|---|---:|---:|---:|
| fast | 0.092448 | 0.166592 | 1.802x |
| balanced | 0.094144 | 0.169536 | 1.801x |
| accurate | 0.110592 | 0.167520 | 1.515x |
| HAO NVFP4/FP8-PV | 0.158240 | 0.169504 | 1.071x |
| HAO NVFP4/NVFP4-PV | 0.192512 | 0.169536 | 0.881x |

`fast` is the latency endpoint. `balanced` adds a fixed 32-row global K/V
anchor layout and recovers substantial long-sequence model fidelity for
about 1.8% more latency. `accurate` retains the anchor and restores
correction-warpgroup normalization.

At saturated H64, balanced measures 0.206944 ms versus 0.404864 ms BF16 at
S4096 (1.956x), and 0.778592 ms versus 1.641504 ms at S8192 (2.108x).

## Numerical gate

The shiftless score approximation remains distribution-sensitive, so the
synthetic and downstream gates must be read together. On the 200-image
S4096 ViT replay, fast, balanced, and accurate have 95.5%, 98.5%, and 98.5%
top-1 agreement with BF16. On the 800-block BERT replay their masked-token
accuracies are 61.311%, 61.272%, and 61.360%, versus 61.740% BF16.

Fast and balanced require matching folded K64 Q/K scales:

```text
--nv-qk-fold-k64-scales both --nv-qk-fold-scale-select mse
```

The comprehensive runner now adds those arguments automatically. Earlier
full-reference files that paired a folded-scale binary with ordinary Q/K
scales remain useful for timing and native-reference latency, but their TK
accuracy fields are invalid. `summary.json` uses the corrected folded-QK
records for synthetic accuracy.

## Approximation result

The production P transform is already the low-degree endpoint: one packed
`FFMA2` evaluates two affine values, then native packed `F2FP` emits E2M1.
A Q0 quadratic increased static `FFMA2` count from 128 to 160, slowed S4096
to 0.097888 ms, and reduced cosine. Direct log-to-code classifiers intended
to remove `F2FP` took 0.143648-0.147744 ms because thresholding, integer
conversion, and packing were more expensive than the hardware conversion.

The remaining credible optimization target is the represented-denominator
and P-publication dependency tail. A correction-warpgroup plus folded-QK
preload experiment was rejected after both variants failed to complete and
held the GPU at 100% utilization.

## Final local pass

A clean rebuild reproduced `fast` at 0.092512 ms versus the retained
0.092448 ms record. Two exact denominator reschedules were rejected:

| Decoder | Time (ms) | Change vs clean baseline | Output |
|---|---:|---:|---|
| aggregate, retained | 0.092512 | baseline | reference |
| pairwise partials | 0.094208 | +1.83% | identical |
| stream after each word | 0.094464 | +2.11% | identical |

The current aggregate decoder is therefore locally optimal among the tested
equivalent schedules. Relative to the retained record, the score-pack,
raw-score-pack, and fixed-P ceilings leave 1.280 us, 2.336 us, and 4.832 us
respectively. The older 12.3 us raw-pack diagnostic belongs to the
superseded 0.1024 ms NV/NV path, not this final NV/MX kernel.

Balanced-mode coefficient retunes improved synthetic random-input error but
regressed the fixed-seed long-ViT gate. The retained `(A,B)=(1.65,0.8)` gives
ViT-logit relative L2 0.037332; `(1.70,1.2)` gives 0.044309 and
`(1.55,1.0)` gives 0.048191. No final-pass kernel change was promoted.

## Reproduction

From `tk_fa4/fp4_fa4_fwd`:

```bash
python3 hao_comprehensive_suite.py \
  --shape-set all \
  --variant nvmx-pareto \
  --warmup-ms 300 \
  --rep-ms 3000 \
  --seed 0 \
  --gpu 0 \
  --output-dir ../../results/fp4_fa4_nvmx_pareto_replay
```

`summary.json` is the compact machine-readable result. `raw/` contains the
selected timing, downstream, native-reference, and rejected-experiment
records used by the report.

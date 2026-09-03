# One-pass NVFP4 row-max stabilization

## Result

The retained noncausal path avoids both a score pre-scan and a score reload by
changing the physical K/V order within each N128 tile:

```text
logical keys:  0  1  2  3  4  5 ... 126 127

physical Q0:   0  4  8 12 ... 124
physical Q1:   1  5  9 13 ... 125
physical Q2:   2  6 10 14 ... 126
physical Q3:   3  7 11 15 ... 127
```

K and V receive the same permutation. Q0 is therefore a distributed 32-key
sample of the complete N128 tile, while the kernel still loads every score
exactly once. Q0 is retained in registers long enough to establish the online
row-max anchor and is then packed normally.

For a permutation matrix `Pi`,

```text
K' = Pi K
V' = Pi V

softmax(Q K'^T) V'
  = softmax(Q K^T Pi^T) Pi V
  = softmax(Q K^T) Pi^T Pi V
  = softmax(Q K^T) V.
```

The reordering is thus exact before quantization. It changes FP4 block
grouping, but not the underlying attention operation.

## S4096 kernel result

Shape: B1/S4096/H24/D128, NVFP4 QK and NVFP4 PV, GB200.

| Path | Time | Cosine vs BF16 | Relative L2 vs BF16 |
|---|---:|---:|---:|
| Q0 one-pass plus interleaved K/V | **0.162048 ms** | 0.979425 | 0.205244 |
| local BF16 | 0.164192 ms | 1.0 | 0.0 |
| stabilized HAO-L2 NVFP4 | 0.188440 ms | 0.981651 | 0.191242 |

The retained path is 1.31% faster than BF16 and 14.0% faster than HAO-L2.
Its random-input error is close to HAO-L2 rather than the old shiftless-fast
endpoint.

## Full downstream replay

The Q0 path uses four native exponent pairs, the refit cubic approximation,
the represented P denominator, direct E4M3 scale encoding, and the
interleaved K/V layout.

| Workload | Metric | BF16 | Q0 interleaved | HAO-L2 |
|---|---|---:|---:|---:|
| ViT, 1,000 images | task accuracy | 98.9% | **99.0%** | 98.6% |
| ViT, 1,000 images | BF16 agreement | 100% | 99.3% | **99.7%** |
| ViT, 1,000 images | logit cosine | 1.0 | 0.998256 | **0.999557** |
| ViT, 1,000 images | logit relative L2 | 0.0 | 0.058996 | **0.029725** |
| BERT, 7,583 masked tokens | task accuracy | 60.583% | **60.425%** | 60.385% |
| BERT, 7,583 masked tokens | BF16 agreement | 100% | 92.127% | **92.246%** |
| BERT, 7,583 masked tokens | logit cosine | 1.0 | **0.998245** | 0.998238 |
| BERT, 7,583 masked tokens | logit relative L2 | 0.0 | **0.059253** | 0.059364 |

All outputs were finite. The new path essentially matches HAO-L2 on BERT
while running faster than BF16. ViT task accuracy is also preserved, although
HAO-L2 remains closer at the logit level.

## Why this avoids reloads

The exact row-max control reads Q0-Q3 to find the maximum, then reads the
quarters again to transform and pack P. The earlier preview control performs
extra loads from every quarter before normal consumption. Both put TMEM loads
and score-register lifetime on the first-P critical path.

The retained schedule is:

```text
load Q0 once -> reduce distributed anchor -> retain Q0
load Q2 once -> transform and pack
transform retained Q0
load Q1 once -> transform and pack -> publish first P half
load Q3 once -> transform and pack -> publish tail
```

No additional score slot, mbarrier, pre-scan, or reload is introduced.

## Rejected variant

For ViT's partial second tile, a follow-up packed 32 real keys into Q0 instead
of retaining the regular stride-4 order. On the 100-image pilot, relative
logit L2 worsened from `0.050784` to `0.059030` and KL rose from `0.004738`
to `0.009192`. This indicates that balanced K/V block composition matters;
the partial-tile specialization is not retained.

## Integration constraints

- The current proof applies to full, noncausal attention. A causal kernel
  would also need to permute mask semantics and restore logical key order.
- K and V must always use the same permutation.
- Returned attention probabilities would need an inverse permutation;
  the context output does not.
- The benchmark assumes prequantized Q/K/V. In a dynamic quantizer, the
  permutation should be fused into K/V destination addressing. A separate
  dense transpose would add bandwidth cost and is not included in timing.

## Reproduction

Kernel benchmark:

```bash
cd /workspace/codebases/pv/fp4_matmul/tk_fa4/fp4_fa4_fwd
PYTHONPATH=/workspace/codebases/flash-attention-fp4:$PYTHONPATH \
python hao_direct_fp4pv_benchmark.py \
  --extension /tmp/tk_q0_n4f1_s4096.so \
  --extension-module _C_tk_q0_n4f1_s4096 \
  --qk-format nvfp4 --pv-format nvfp4 \
  --tk-only --summary-only --interleave-kv-quarters \
  --warmup-ms 10 --rep-ms 100 --seed 0
```

ViT and BERT:

```bash
python eval_regular_attention.py \
  --samples 1000 --scale-sweep-samples 0 --mask-value 20 \
  --extension /tmp/tk_q0_n4f1_s256.so \
  --extension-module _C_tk_q0_n4f1_s256 \
  --interleave-kv-quarters \
  --output ../../results/fp4_fa4_nv_onepass_stabilizer_20260729/vit.json

python eval_bert_mlm_attention.py \
  --samples 200 --scale-sweep-samples 0 \
  --extension /tmp/tk_q0_n4f1_s256.so \
  --extension-module _C_tk_q0_n4f1_s256 \
  --interleave-kv-quarters \
  --output ../../results/fp4_fa4_nv_onepass_stabilizer_20260729/bert.json
```

# Universal finite NVFP4 policy

This follow-up fixes the non-finite outputs produced by the named
`fast`/`accurate` shiftless policies on regular-attention ViT and BERT
inputs. Measurements were collected on one NVIDIA GB200 on 2026-07-29.

## Root cause

The shiftless path materializes a value proportional to `exp(score)` and
encodes its per-quarter NVFP4 scale with a compact magic-bias E4M3 encoder.
That encoder is linear only over normal E4M3 values. Real ViT scores reached
approximately `[-69.65, 103.34]`, so scale bytes could wrap into invalid
encodings. A sampled softmax denominator could also miss a sharp maximum.

The fix has three parts:

1. Clamp the magic-bias scale input to the normal finite E4M3 log2 interval
   `[-6, 8.75]`.
2. For shiftless policies, cap sampled denominator values at E2M1 maximum
   `6` and floor each quarter estimate at `6`.
3. Add policy ID 7, `universal`, which restores online row-max
   stabilization, uses eight native exponential pairs per quarter, applies
   the affine approximation only on the latency-critical first quarter, and
   keeps early K64 P publication.

The first two changes make `fast` and `accurate` finite, but they do not make
shiftless softmax distribution-invariant. `universal` is the general model
dispatch.

## Kernel benchmark

Shape: B1/S4096/H24/D128, NVFP4 QK and NVFP4 PV. Values are means over seeds
0 through 3 with 20 ms warmup and 100 ms repetition windows.

| Policy | Time (ms) | Cosine vs BF16 | Relative L2 vs BF16 |
|---|---:|---:|---:|
| `universal` | **0.176400** | 0.979585 | 0.201632 |
| Previous TK `hao-l2` | 0.188440 | 0.981651 | 0.191242 |
| HAO native NV/NV | 0.192596 | 0.981654 | 0.191603 |
| HAO native BF16 | 0.164192 | 1.000000 | 0 |

`universal` is 1.092x faster than HAO NV/NV and 1.068x faster than the
previous downstream-safe `hao-l2` policy. It remains 7.4% slower than the
HAO BF16 kernel at this shape.

Raw benchmark records are under [`benchmark/`](benchmark/).

## Downstream replay

The retained S256/H16/D128 adapters pad ViT-B/16 and BERT-base into the
kernel shape. Dynamic Q/K/V quantization is part of the accuracy path and is
not included in the kernel timing above.

| Metric | ViT, 100 images | BERT, 20 blocks |
|---|---:|---:|
| Non-finite output rows | **0** | **0** |
| Logit cosine vs BF16 | 0.997157 | 0.991550 |
| Logit relative L2 | 0.075794 | 0.129796 |
| BF16 task accuracy | 99.0% | 59.57% |
| FP4 task accuracy | 99.0% | 56.60% |
| Top-1 agreement | 98.0% | 83.15% |

These runs cover every attention layer for each input. They establish the
regression gate used here; they are not a claim that a 120-input replay
proves model quality universally.

Raw model records are under [`downstream/`](downstream/).

## Reproduction

```bash
cd /workspace/codebases/pv/fp4_matmul/tk_fa4/fp4_fa4_fwd

python hao_nv_policy_suite.py \
  --policy universal \
  --seed 0 --seed 1 --seed 2 --seed 3 \
  --warmup-ms 20 --rep-ms 100 \
  --build-root /tmp/tk_universal_policy_suite \
  --output-dir ../../results/fp4_fa4_universal_policy_20260729/benchmark \
  --rebuild --skip-hao

python hao_nv_policy_downstream_suite.py \
  --policy universal \
  --vit-samples 100 --bert-samples 20 \
  --build-root /tmp/tk_universal_downstream_suite \
  --output-dir ../../results/fp4_fa4_universal_policy_20260729/downstream \
  --rebuild
```

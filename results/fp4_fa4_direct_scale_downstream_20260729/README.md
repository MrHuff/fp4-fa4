# Direct-scale policy downstream replay

This replay tests the current named NVFP4 policies after promoting direct
integer E4M3 scale encoding into `fast`. It resolves the distinction between
finite output and model-level quality.

## Protocol

- Kernel adapter shape: B1/S256/H16/D128
- ViT-B/16 CIFAR-10: 1,000 test images
- BERT-base WikiText-2 MLM: 200 blocks, 7,583 masked tokens
- Baseline: the same pretrained model with BF16 attention
- Q/K/V quantization is part of the accuracy adapter and is not included in
  the S4096 kernel timings quoted below.

The current `fast` and `universal` policies were rebuilt as named policy IDs
1 and 7. The retained `hao-l2` result uses the identical 1,000/200 protocol
from `../fp4_fa4_nv_policy_downstream_20260729/full/summary.json`.

## Matched results

| Policy | S4096 time | ViT accuracy | ViT agreement | BERT accuracy | BERT agreement |
|---|---:|---:|---:|---:|---:|
| BF16 | 0.164192 ms | 98.9% | 100% | 60.583% | 100% |
| `fast` | **0.104800 ms** | 92.0% | 91.7% | 43.492% | 52.314% |
| `universal` | 0.176400 ms | 98.6% | 98.8% | 58.961% | 82.210% |
| `hao-l2` | 0.188440 ms | 98.6% | 99.7% | 60.385% | 92.246% |

The corresponding logit errors are:

| Policy | ViT cosine | ViT relative L2 | BERT cosine | BERT relative L2 |
|---|---:|---:|---:|---:|
| `fast` | 0.938982 | 0.343978 | 0.951274 | 0.312081 |
| `universal` | 0.994882 | 0.101134 | 0.990946 | 0.134263 |
| `hao-l2` | 0.999557 | 0.029725 | 0.998238 | 0.059364 |

## Interpretation

Direct integer scale encoding fixes the non-finite E4M3 wraparound failure
without restoring the floating clamp's latency. It does not make shiftless
softmax distribution-invariant. The no-downstream-degradation hypothesis
for `fast` is therefore rejected.

`fast` is 1.567x faster than the local HAO BF16 kernel and 1.838x faster than
HAO NVFP4/NVFP4 at S4096, but is suitable only as an aggressive calibrated
or training-aware endpoint. `universal` reduces latency by 6.39% relative to
`hao-l2` and preserves the same ViT task accuracy, but loses 1.42 percentage
points of BERT accuracy relative to `hao-l2`. `hao-l2` remains the
high-fidelity endpoint.

The present policy set does not contain a mode that is simultaneously faster
than the local BF16 kernel and quality-neutral on both downstream tasks.

## Artifacts

- `fast/summary.json`
- `fast/vit_fast.json`
- `fast/bert_fast.json`
- `universal/summary.json`
- `universal/vit_universal.json`
- `universal/bert_universal.json`

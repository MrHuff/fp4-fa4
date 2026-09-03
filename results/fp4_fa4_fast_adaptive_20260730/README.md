# Length-adaptive represented-P normalization

Date: 2026-07-30

> **Validation update:** the subsequent
> [long-path model study](../fp4_fa4_long_path_validation_20260730/README.md)
> found that one-word normalization is not robust on real S2048/S4096 ViT
> activations. Treat `fast-adaptive` as an accuracy-risky speed policy and
> use `fast-corrected` for model-safe long-sequence execution.

## Result

The retained `fast-adaptive` policy (policy ID 10) selects one of two
normalization implementations at compile time:

| Specialized sequence length | Normalization |
|---|---|
| S <= 512 | Exact sum of the represented E2M1 payload and E4M3 scales |
| S > 512 | One-word stratified estimate with a known-maximum control variate |

There is no runtime length branch. Short kernels retain the correction
warpgroup used by `fast-corrected`; long kernels compile it out and calculate
the denominator in the P producer.

## Long-path estimator

For one row of one N32 quarter, let the represented nonnegative E2M1 values
before the quarter scale be `c_0, ..., c_31`. Scale construction guarantees
that at least one represented value is the maximum E2M1 value, 6. The kernel
samples one packed word, or eight values, with a different word selected in
successive quarters.

Let `U` be the sum of the eight sampled values and let `h` be one when the
sample contains a 6. The estimated unscaled quarter sum is

```text
             31
C_hat = 6 + ------ (U - 6h),    m = 8.
            m - h
```

The quarter denominator contribution is `scale * C_hat`. The packed-word sum
uses PRMT and DP4A directly on E2M1 nibbles; no FP4-to-float vector unpack is
introduced. Detecting the represented 6 is a short shift/AND sequence.

This is a control variate rather than plain four-times extrapolation. It uses
the maximum that is already known from quarter-scale construction and
estimates only the other 31 values.

## S4096/H24

GB200, B1/S4096/H24/D128, noncausal, NVFP4 QK and NVFP4 PV. Accuracy is
against Torch BF16 on the same input. `fast-adaptive` is the mean over seeds
0-3; timing range was 0.106528-0.106816 ms.

| Path | Time (ms) | Cosine | Relative L2 | RMSE |
|---|---:|---:|---:|---:|
| Invalid sampled-denominator record | 0.104000 | 0.964326 | 0.270026 | - |
| **`fast-adaptive`** | **0.106600** | **0.963239** | **0.298968** | **0.007719** |
| Exact `fast-corrected` | 0.112800 | 0.963629 | 0.291529 | 0.007527 |
| Local BF16 | 0.164192 | 1.000000 | 0 | 0 |

The adaptive long path is 5.50% faster than `fast-corrected`, 1.540x faster
than local BF16, and 2.50% above the invalid 0.104 ms record. Relative to
exact represented-P normalization, cosine changes by -0.000390 and relative
L2 by +0.007439.

## Cutoff check

At S512/H16, enabling the estimator was measurably less accurate, so S512 is
kept on the exact path.

| S512 path | Time (ms) | Cosine | Relative L2 |
|---|---:|---:|---:|
| Diagnostic sampled path | 0.012608 | 0.960084 | 0.316824 |
| **Retained exact path** | **0.014336** | **0.963201** | **0.295022** |

The long estimator is validated at S4096. Intermediate lengths above S512
remain a validation boundary; use `fast-corrected` when exact represented-P
normalization is required independent of length.

## Downstream smoke tests

The S256 specialization uses exactly the four-word correction path. The
following 20-sample runs confirm that the named adaptive policy preserves the
corrected downstream behavior.

| Evaluation | BF16 | `fast-adaptive` | Agreement | Logit cosine | Logit relative L2 |
|---|---:|---:|---:|---:|---:|
| ViT, 20 images | 100.00% | 100.00% | 100.00% | 0.999091 | 0.042649 |
| BERT, 742 masked tokens | 59.569% | 59.030% | 90.162% | 0.997018 | 0.077345 |

All outputs were finite.

## Rejected alternatives

- Changing the residual extrapolation coefficient away from its unbiased
  value worsened BERT error.
- Selecting a packed word that contained the maximum added enough producer
  work to regress S4096 to about 0.115 ms.
- Adding a floor to the estimate hurt BERT accuracy.
- Mirroring packed P into shared memory preserved exactness but increased
  latency to 0.129-0.132 ms because the stores and fence entered the critical
  path.
- Summing the first exact snapshot before loading the second regressed to
  about 0.115 ms.

## Reproduction

```bash
make -f Makefile.hao_direct_fp4pv \
  HAO_FP4PV_NV_POLICY=fast-adaptive \
  HAO_SEQ_LEN=4096 HAO_HEADS=24 \
  NVCC_SPLIT_COMPILE=1 \
  OUT=/tmp/tk_fast_adaptive.so \
  MODULE=_C_tk_fast_adaptive
```

```bash
PYTHONPATH=/workspace/codebases/flash-attention-fp4 \
python3 hao_direct_fp4pv_benchmark.py \
  --extension /tmp/tk_fast_adaptive.so \
  --extension-module _C_tk_fast_adaptive \
  --qk-format nvfp4 --pv-format nvfp4 \
  --tk-only --global-anchor-kv --global-anchor-samples 32
```

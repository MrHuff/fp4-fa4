# Near-record NVFP4 with represented-P normalization

Date: 2026-07-30

## Result

The retained policy is `fast-corrected` (policy ID 9). It repairs the
normalization bug in the approximately 0.104 ms speed-record path while
retaining its one-pass, shiftless structure:

```text
score approximation -> E2M1 payload + E4M3 scale -> exact represented sum
                                                        |
                                                        v
                                            normalize the PV output
```

The old path estimated the denominator from a sparse sample of
pre-quantization exponentials. That estimate was not the sum of the FP4
values consumed by PV, so probability mass varied by input distribution and
caused large downstream failures.

The corrected lifecycle is:

1. The softmax warpgroups approximate and pack P, then publish `p_tail`.
2. The correction warpgroup snapshots the packed E2M1 payload and E4M3
   quarter scales from TMEM.
3. It immediately publishes `nv_denom_loaded`, allowing QK to reuse the score
   bank while the sum is computed.
4. It decodes and sums exactly the represented values that PV consumes.
5. It publishes `nv_denom_done`; the epilogue divides by that row sum.

The build uses a 32-key global anchor, affine coefficients `(1.5, 1.22)`,
stage masks `14/14`, and a 184/96 softmax/correction register split. `ptxas`
reports 128 registers, one barrier, and no spills.

## S4096/H24 result

GB200, B1/S4096/H24/D128, noncausal, NVFP4 QK and NVFP4 PV. Accuracy is
against Torch BF16 on the same HAO-generated input.

| Path | Time (ms) | Cosine | Relative L2 | Status |
|---|---:|---:|---:|---|
| Q2-cubic speed record | 0.104000 | 0.964326 | 0.270026 | Invalid sampled denominator |
| Fake-denominator ceiling | 0.106528 | - | - | Diagnostic only |
| **`fast-corrected`** | **0.112800** | **0.963629** | **0.291529** | Retained |
| Exact denominator + Q2 cubic | 0.114720 | 0.965212 | 0.278109 | Optional accuracy point |
| Previous `fast-accurate` | 0.120864 | 0.970941 | 0.245339 | Slower |
| Local BF16 reference | 0.164192 | 1.000000 | 0 | Reference |

`fast-corrected` is the four-seed mean. Its range was
0.112672-0.112960 ms, mean RMSE was 0.007527, and every output was finite.
It is 8.46% slower than the invalid record, 6.67% faster than
`fast-accurate`, and 1.456x faster than the local BF16 reference.

## Downstream check

The named policy was rebuilt at B1/S256/H16/D128 and evaluated with the
required 32-key global permutation.

| Evaluation | BF16 | `fast-corrected` | Agreement | Logit cosine | Logit relative L2 |
|---|---:|---:|---:|---:|---:|
| ViT, 100 images | 99.00% | 98.00% | 99.00% | 0.995890 | 0.090598 |
| BERT, 3,759 masked tokens | 61.852% | 61.160% | 90.051% | 0.996895 | 0.078792 |

All outputs were finite. The machine-readable rerun is in
`named_policy_downstream/summary.json`.

## What did not help

- One- and two-word represented-denominator estimates were fast but
  numerically unusable. Those diagnostic source modes were removed.
- Six native EX2 pairs cost about 0.010 ms and produced only a small accuracy
  gain.
- A positive anchor bias and moving Q2 cubic to stage 0 worsened downstream
  quality.
- Q2 cubic on both stages improved the random tensor metric but not the
  100-image ViT result.
- One `LDTM.x16` snapshot was bit-identical and removed 64 static SASS
  instructions, but tied at S4096 and was slightly slower at S256. TMEM
  transaction latency, rather than load instruction count, is exposed.

## Boundary

The denominator is now exact for the represented E2M1 payload and E4M3
scales. The row maximum is still estimated from 32 globally distributed
keys. This is a finite, downstream-tested approximation, not exact online
softmax. Restoring a full row-max pass measured about 0.168 ms and gives up
the near-record schedule.

The fake-denominator ceiling leaves roughly 0.0063 ms between the retained
kernel and a free denominator. Closing that gap requires either avoiding the
packed-P TMEM readback or deriving the exact represented sum during packing
without extending the softmax critical path. Wider loads and sampled sums did
not accomplish that.

## Reproduction

```bash
make -B -f Makefile.hao_direct_fp4pv -j1 \
  HAO_FP4PV_NV_POLICY=fast-corrected \
  HAO_SEQ_LEN=4096 HAO_HEADS=24 \
  NVCC_SPLIT_COMPILE=1 \
  OUT=/tmp/tk_fast_corrected.so \
  MODULE=_C_tk_fast_corrected
```

```bash
PYTHONPATH=/workspace/codebases/flash-attention-fp4:$PYTHONPATH \
python3 hao_direct_fp4pv_benchmark.py \
  --extension /tmp/tk_fast_corrected.so \
  --extension-module _C_tk_fast_corrected \
  --qk-format nvfp4 --pv-format nvfp4 \
  --tk-only --global-anchor-kv --global-anchor-samples 32
```

```bash
python3 hao_nv_policy_downstream_suite.py \
  --policy fast-corrected \
  --vit-samples 100 --bert-samples 100 \
  --output-dir /tmp/fast-corrected-downstream
```

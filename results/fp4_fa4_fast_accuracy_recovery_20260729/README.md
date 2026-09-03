# Fast-path accuracy recovery

Date: 2026-07-29

## Result

The production candidate is the `fast-accurate` NVFP4 policy. It keeps the
one-pass global-anchor schedule, evaluates Q1 and Q2 with a cubic, and leaves
the latency-critical Q0 and tail Q3 on one packed affine FMA. Both fits target
the exact denominator formed by the represented E2M1 payload and E4M3 scale.

```text
Q0: refitted affine
Q1: refitted cubic
Q2: refitted cubic
Q3: refitted affine
```

The coefficients are:

```text
affine: y = 1.5 x + 1.22
cubic:  y = 0.06018944 x^3
             + 0.30327865 x^2
             + 0.73266160 x
             + 1.10013866
```

## Kernel measurements

GB200, B1, D128, noncausal, NVFP4 QK and NVFP4 PV. Accuracy is against the
Torch BF16 output on the same HAO-generated input. Timings use
`triton.testing.do_bench`.

### S256, H16

| Policy | Time (ms) | Cosine | Relative L2 |
|---|---:|---:|---:|
| Previous fast affine | 0.01046-0.01088 | 0.962738 | 0.320627 |
| Refitted affine | 0.01072-0.01082 | 0.963528 | 0.293052 |
| Refitted affine + Q2 cubic | 0.01083 | 0.967230 | 0.272150 |
| **`fast-accurate` Q1/Q2 cubic** | **0.01053-0.01101** | **0.971273** | **0.244041** |
| Q1/Q2 cubic + native Q0/Q3 | 0.01184 | 0.972610 | 0.242705 |
| Q1/Q2 plus one-stage Q3 cubic | 0.01181-0.01200 | 0.973134-0.973286 | 0.232835-0.233619 |
| All-quarter cubic | 0.01229 | 0.980002 | 0.201860 |

Matched primary/compare runs put the original and refitted Q1/Q2-cubic
variants in the same timing bucket. Relative to the old all-affine kernel,
`fast-accurate` was 0.6-2.4% slower over seeds 1-3. Its static SASS has 7,888
instructions, 224 `FFMA2`, 68 `MUFU`, 144 `F2FP`, 12 `LDTM`, 128 registers,
one barrier, and no spills.

### S4096, H24

| Policy | Time (ms) | Cosine | Relative L2 |
|---|---:|---:|---:|
| Previous fast affine | **0.116736** | 0.962477 | 0.320125 |
| Q1/Q2 cubic, old coefficients | 0.120864 | 0.970842 | 0.251695 |
| **`fast-accurate`** | **0.120864** | **0.970941** | **0.245339** |
| All-quarter cubic | 0.133152 | 0.979375 | 0.205317 |

At saturation, `fast-accurate` costs 3.5% over the previous fastest path and
is 9.2% faster than all-cubic. It recovers about half of the cosine gap and
65% of the relative-L2 gap between those endpoints.

## Downstream smoke tests

The S256/H16 `fast-accurate` binary was run without score shifting or a
stabilized pre-scan. All outputs were finite.

| Evaluation | BF16 | FP4 | Agreement / error |
|---|---:|---:|---:|
| BERT MLM, 40 samples | 61.35% masked accuracy | 60.75% | 90.25% top-1 agreement |
| BERT logits | - | - | cosine 0.997456, relative L2 0.071295 |
| ViT CIFAR-10, 40 samples | 100% | 100% | 100% top-1 agreement |
| ViT logits | - | - | cosine 0.999525, relative L2 0.030842 |

## Rejected alternatives

- Refitting only the affine is free and reduces relative L2, but one affine
  FMA cannot reproduce the seven nonuniform E2M1 decision boundaries.
- Six native EX2 pairs on every quarter or only on Q2 consume exposed SFU
  latency. Restricting them to the remaining affine quarters improves quality
  slightly but costs about 10% at S256.
- Cubic Q3 is exposed. Applying it to only one query stage already moves the
  kernel to 0.0118-0.0120 ms.
- A positive quadratic fitted to the observed score interval looks better
  offline but turns upward for sufficiently negative scores. It is not
  downstream-safe without a clamp, and the clamp plus second FMA is no longer
  competitive.
- A globally safe concave quadratic does not materially improve on the
  refitted affine.
- Full cubic and two-slope hinge variants retain the desired approximately
  0.98 cosine, but both expose another arithmetic level on the critical path.

## Build and benchmark

```bash
make -f Makefile.hao_direct_fp4pv \
  HAO_FP4PV_NV_POLICY=fast-accurate \
  HAO_SEQ_LEN=256 HAO_HEADS=16 \
  NVCC_SPLIT_COMPILE=2 \
  OUT=/tmp/tk_fast_accurate.so \
  MODULE=_C_tk_fast_accurate
```

The policy requires the matching 64-key permutation:

```bash
PYTHONPATH=/workspace/codebases/flash-attention-fp4:$PYTHONPATH \
python3 hao_direct_fp4pv_benchmark.py \
  --extension /tmp/tk_fast_accurate.so \
  --extension-module _C_tk_fast_accurate \
  --qk-format nvfp4 --pv-format nvfp4 \
  --global-anchor-kv --global-anchor-samples 64
```

## Boundary

Exact all-cubic accuracy was not achieved at the old affine time. Q1 and Q2
can be made cubic because their arithmetic overlaps useful tensor work. Q0 is
the first-P publication path and Q3 closes the tail K64 page; extra dependent
arithmetic in either location delays PV. Reaching the remaining quality gap
without the measured 9-14% all-cubic cost requires a new one-instruction
classifier for Q0/Q3, not another placement of the existing polynomial.

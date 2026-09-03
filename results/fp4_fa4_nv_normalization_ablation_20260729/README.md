# NVFP4 P-normalization ablation

This experiment asks why the fast and accurate shiftless policies can look
reasonable on a random attention-output comparison while failing after several
real model layers.

## Mechanism

The fast path sends a quantized probability payload to PV:

```text
P_pv = scale * E2M1(approx_exp2(score) / scale)
```

but previously divided the output by a denominator estimated from a sparse
sample of the pre-quantized exponentials. Consequently,

```text
denominator_estimate != sum(P_pv)
```

It also omitted the online row-maximum shift. Exact softmax is invariant to an
additive offset in every score, but a finite E2M1/E4M3 quantizer is not. Layer-
and mask-dependent score offsets therefore changed the shiftless result.

## 2x2 downstream ablation

All four cells use the same direct E4M3 scale encoder and fast P packer. The
only changes are online row-max stabilization and whether the denominator is
the exact sum of the represented E2M1 values consumed by PV.

| Row max | Represented denominator | ViT accuracy | ViT agreement | BERT accuracy | BERT agreement |
|---|---|---:|---:|---:|---:|
| no | no | 92.0% | 91.0% | 44.74% | 53.37% |
| no | yes | 87.0% | 86.0% | 48.65% | 59.84% |
| yes | no | 97.0% | 98.0% | 54.04% | 75.34% |
| yes | yes | **98.0%** | **99.0%** | **59.16%** | **91.91%** |
| BF16 | BF16 | 99.0% | 100% | 59.57% | 100% |

This is an interaction. A correct denominator cannot repair the
distribution-dependent shiftless quantizer, and row-max stabilization cannot
repair normalization against values different from those consumed by PV.

The represented-denominator correction also removes the direct signature of
the bug: the arithmetic replay measured a 25-27% mean absolute probability
mass error for sampled-denominator modes and zero mass error for
represented-denominator modes.

## Why random cosine was misleading

The safe accurate-polynomial control and the corrected mode are almost
indistinguishable on one stationary random tensor distribution:

| Mode | Random cosine | Random relative L2 | BERT accuracy | BERT logit relative L2 |
|---|---:|---:|---:|---:|
| accurate polynomial, old normalization | 0.974094 | 0.230252 | 43.67% | 0.318521 |
| fast polynomial, corrected normalization | 0.975859 | 0.227991 | **59.16%** | **0.065124** |

The random aggregate averages away the row-wise probability-mass and
score-offset errors. Repeated model layers do not: they feed the biased output
into the next layer, and BERT's varying score distributions expose the
shiftless path especially strongly. Improving the local exp approximation
therefore does not substitute for fixing normalization.

The original named `accurate` build had a second, independent issue: its
floating E4M3 scale encoder could produce non-finite outputs. Replacing that
encoder with the safe direct integer encoder makes it finite, but the table
above shows that its downstream quality still resembles `fast`.

## Full corrected replay

The compiled row-max plus represented-denominator kernel was replayed over the
full retained protocol:

| Workload | BF16 | Corrected NVFP4 | Agreement | Logit cosine | Logit relative L2 |
|---|---:|---:|---:|---:|---:|
| ViT, 1,000 images | 98.9% | 98.7% | 99.6% | 0.999389 | 0.034896 |
| BERT, 7,583 masked tokens | 60.583% | 60.240% | 91.890% | 0.997921 | 0.064560 |

## Latency at B1/S4096/H24/D128

| Path | Time |
|---|---:|
| shiftless fast | 0.104520 ms |
| shiftless plus represented denominator | 0.110784 ms |
| row max plus sampled denominator | 0.163968 ms |
| row max plus represented denominator | 0.168144 ms |
| local BF16 reference | 0.164192 ms |
| stabilized HAO-L2 NVFP4 | 0.188440 ms |

The final corrected result is the four-seed mean; the two one-factor probes are
seed-0 diagnostics. On matched seed 0, using the exact represented denominator
after stabilization costs only 0.004224 ms, or 2.58%. The large cost is
restoring distribution-invariant online row-max processing and giving up the
shiftless path's aggressive scale handoff schedule. The corrected kernel is
currently 2.41% slower than BF16 but 12.1% faster than the stabilized HAO-L2
path.

## Conclusion

The fast policy does not fail primarily because its polynomial is inaccurate.
It fails because it approximates a different operation from normalized
attention:

1. score offsets alter the shiftless quantization result;
2. the normalization denominator does not match the payload consumed by PV.

Both must be corrected. Further high-fidelity optimization should retain these
two invariants and target the row-max critical path. Arithmetic ceilings show
smaller residual opportunities from block-16 P scales and a better exp
approximation, but neither addresses the original downstream failure.

The detailed arithmetic replay and all downstream JSON records are retained in
this directory. The executable arithmetic harness is
`tk_fa4/fp4_fa4_fwd/hao_nv_p_ablation_suite.py`.

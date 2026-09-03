# NVFP4 approximation coefficient and quarter-placement search

## Scope

This experiment refits the shiftless NVFP4 score-to-E2M1 approximation
against reconstructed attention-output error, then measures where cubic
evaluation is cheap enough to use.

The offline model uses NVFP4-roundtripped Q, K, and V operands with E4M3
block scales. The reference output uses BF16 Q, K, and V. Four seeds, four
heads per seed, 64 rows per head, and 131072 sampled probability values were
used for fitting.

## Coefficient result

The retained affine transform changed from

```text
1.62330034 x + 0.92083546
```

to

```text
1.61131608 x + 0.93574703
```

At S4096/H24, the all-affine kernel improved on seed 0 at unchanged
instruction structure:

| Fit | Time (ms) | Cosine BF16 | Relative L2 | RMSE |
|---|---:|---:|---:|---:|
| Previous | 0.104704 | 0.961779 | 0.279945 | 0.007184 |
| Refit | 0.104832 | 0.961891 | 0.279574 | 0.007174 |

The cubic search found lower-MSE coefficients, but they slightly reduced
end-to-end cosine. The existing cubic remains:

```text
0.07839806 x^3 + 0.28625049 x^2 + 0.63145205 x + 0.99202336
```

## Cubic placement

For the compile-time affine masks, bit 1 is Q1, bit 2 is Q2, and bit 3 is
Q3. A set bit selects affine; a clear bit selects cubic. Q0 remains affine
in fast-approximation mode zero.

At S4096/H24, four-seed means are:

| Stage-0 / stage-1 policy | Time (ms) | Cosine BF16 | Relative L2 |
|---|---:|---:|---:|
| all affine (`14 / 14`) | 0.104356 | 0.962272 | 0.278021 |
| Q2 cubic in stage 1 (`14 / 10`) | **0.104000** | 0.964326 | 0.270026 |
| Q3 cubic in stage 1 (`14 / 6`) | 0.104520 | **0.964330** | **0.270005** |
| Q2 and Q3 cubic in stage 1 (`14 / 2`, seed 0) | 0.106816 | 0.966019 | 0.263332 |
| Q2 and Q3 cubic in both stages (`2 / 2`) | 0.114120 | 0.970550 | 0.244472 |

One cubic quarter is the best throughput/accuracy point. Q2 is preferred:
its accuracy is effectively tied with Q3 and its measured time is lower.

## Shape behavior

Q2 cubic is shape-selective:

| Shape | All-affine (ms) | Q2 cubic (ms) | Cosine gain | Relative-L2 reduction |
|---|---:|---:|---:|---:|
| S1024/H16 | **0.016384** | 0.017920 | +0.002007 | 0.007784 |
| S4096/H24, four-seed mean | 0.104356 | **0.104000** | +0.002054 | 0.007994 |
| S8192/H24, four-seed mean | 0.380928 | **0.378896** | +0.002037 | 0.007973 |

The short shape exposes the extra arithmetic and should retain all-affine.
At S4096 and S8192, Q2 cubic is hidden by the pipeline and is a Pareto
improvement.

## Generated code

Both policies use 128 registers, one barrier, 400 bytes static shared
memory, and no spills. Q2 cubic is not optimized away: it increases static
kernel instructions from 3064 to 3104, including 24 additional `FFMA2`,
five `FFMA`, and seven `FMUL` instructions. The long-shape speed signal is
therefore consistent with latency hiding or a better instruction schedule;
it does not come from reducing instruction count.

Raw offline and compiled results are in `offline_search.json` and
`compiled_results.json`.

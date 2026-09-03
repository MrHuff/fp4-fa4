# FP4 QK/PV format matrix

This matrix compares all four NVFP4/MXFP4 combinations with the same
aggressive softmax-to-FP4 throughput policy on one NVIDIA GB200. QK format
is listed first and PV format second.

## Result

There is no intrinsic MXFP4 throughput tax. The initial matrix accidentally
gave NVFP4 both stage-0 and stage-1 affine masks while MXFP4 had only the
stage-0 mask. MX stage 1 therefore evaluated Q1-Q3 with a three-FMA cubic
transform. Adding the missing `HAO_FP4PV_MX_STAGE1_AFFINE_MASK=14` removes
that software-only asymmetry.

| Shape | NV/NV | MX/NV | NV/MX | MX/MX | Spread |
|---|---:|---:|---:|---:|---:|
| B1 S1024 H16 D128 | 0.016672 ms | 0.016544 ms | **0.016384 ms** | 0.017248 ms | 5.27% |
| B1 S4096 H24 D128 | 0.104448 ms | 0.104032 ms | 0.104000 ms | **0.102720 ms** | 1.68% |
| B1 S8192 H24 D128 | 0.380928 ms | 0.381248 ms | 0.380384 ms | **0.375568 ms** | 1.51% |
| B1 S32768 H24 D128 | 5.144768 ms | 5.172336 ms | 5.138464 ms | **5.088192 ms** | 1.65% |
| B4 S4096 H32 D128 | 0.452896 ms | **0.446496 ms** | 0.449536 ms | 0.448512 ms | 1.43% |

The short S1024 kernels are sensitive to process startup, clocks, and
sub-microsecond fixed costs. A longer 500 ms replay measured NV/NV at
0.016736 ms and MX/MX at 0.015968 ms. The long-context rows, where the
kernel body dominates, stay within 1.7%.

## Ceiling check

At B1/S4096/H24, matched fixed-P kernels measured:

| Route | Time |
|---|---:|
| NVFP4 QK + NVFP4 PV | 0.088064 ms |
| NVFP4 QK + MXFP4 PV | **0.086016 ms** |

Fixed-P bypasses probability generation, so its output is intentionally
invalid. It isolates the tensor operation, scale layout, and publication
floor and confirms that MXFP4 is not inherently slower.

## Accuracy boundary

Equal throughput does not mean equal numerical behavior. At B1/S4096/H24:

| Route | Cosine vs BF16 | Relative L2 |
|---|---:|---:|
| HAO native NV/NV | **0.981415** | **0.192767** |
| NV/NV | **0.961779** | **0.279944** |
| MX/NV | 0.954314 | 0.298980 |
| NV/MX | 0.949646 | 0.314413 |
| MX/MX | 0.939800 | 0.347126 |

The all-affine MX policy is the matched-throughput endpoint. The earlier
mixed cubic MX policy was more accurate, but comparing it with the affine
NV endpoint incorrectly presented an approximation-policy cost as a format
cost.

### Complete aggressive-policy accuracy

Relative L2 is `||candidate - reference||2 / ||reference||2`. The HAO
reference is its native NVFP4 QK + NVFP4 PV result on the same seed-0 input.

| Shape | QK / PV | Cosine vs BF16 | Relative L2 vs BF16 | Cosine vs HAO | Relative L2 vs HAO |
|---|---|---:|---:|---:|---:|
| B1 S1024 H16 | NV / NV | 0.960912 | 0.284190 | 0.971108 | 0.244486 |
|  | MX / NV | 0.954241 | 0.299580 | 0.945337 | 0.327140 |
|  | NV / MX | 0.947013 | 0.321422 | 0.947118 | 0.321143 |
|  | MX / MX | 0.938481 | 0.348555 | 0.921811 | 0.388905 |
| B1 S4096 H24 | NV / NV | 0.961779 | 0.279944 | 0.972513 | 0.237703 |
|  | MX / NV | 0.954314 | 0.298980 | 0.945355 | 0.326624 |
|  | NV / MX | 0.949646 | 0.314413 | 0.949790 | 0.313964 |
|  | MX / MX | 0.939800 | 0.347126 | 0.922542 | 0.388336 |
| B1 S8192 H24 | NV / NV | 0.962806 | 0.275583 | 0.973139 | 0.234173 |
|  | MX / NV | 0.955515 | 0.295041 | 0.946642 | 0.322659 |
|  | NV / MX | 0.951204 | 0.310278 | 0.951358 | 0.309922 |
|  | MX / MX | 0.941677 | 0.342982 | 0.924874 | 0.383595 |
| B1 S32768 H24 | NV / NV | 0.963041 | 0.274687 | 0.973461 | 0.232840 |
|  | MX / NV | 0.955690 | 0.294455 | 0.946765 | 0.322290 |
|  | NV / MX | 0.951811 | 0.308551 | 0.951938 | 0.308227 |
|  | MX / MX | 0.942076 | 0.342085 | 0.925112 | 0.383074 |
| B4 S4096 H32 | NV / NV | 0.962132 | 0.278851 | 0.972656 | 0.236940 |
|  | MX / NV | 0.954760 | 0.297589 | 0.945795 | 0.325316 |
|  | NV / MX | 0.949694 | 0.314453 | 0.949797 | 0.314264 |
|  | MX / MX | 0.940094 | 0.346670 | 0.922943 | 0.387785 |

These are deliberately approximate shiftless-throughput kernels. They are
not the stabilized model-deployment path below.

### Fast shiftless P-scale rebasing

The follow-up sweep varies `G` on the actual fast policy. Large stabilized
factors are divided by a power of two before use: `320 -> 1.25`,
`416 -> 1.625`, and `448 -> 1.75`. This preserves each E4M3 mantissa-grid
phase while avoiding the shiftless path's scale overflow.

| G | Time (ms) | Cosine vs BF16 | Relative L2 vs BF16 | RMSE vs BF16 |
|---:|---:|---:|---:|---:|
| 1 | 0.104000 | **0.961779** | 0.279945 | 0.007184 |
| 1.25 | 0.102720 | 0.961766 | 0.279983 | 0.007185 |
| 1.5 | 0.102848 | 0.961777 | **0.279927** | **0.007183** |
| 1.625 | 0.102752 | 0.961771 | 0.279951 | 0.007184 |
| 1.75 | 0.102720 | 0.961769 | 0.279962 | 0.007184 |
| 4 | 0.103904 | **0.961779** | 0.279945 | 0.007184 |
| 5.5 | 0.102752 | non-finite | non-finite | non-finite |

The small timing spread does not reproduce under reversed run order.
`G=1.5` improves relative L2 by only `0.0000172` on this input, regresses
all four broad-shape checks, and regresses the MX/NV control. The production
default remains `G=1`.

The full 34-factor fast sweep and validation are recorded in
[`../fp4_fa4_shiftless_p_scale_sweep_20260728/README.md`](../fp4_fa4_shiftless_p_scale_sweep_20260728/README.md)
and
[`../fp4_fa4_shiftless_p_scale_validation_20260728/README.md`](../fp4_fa4_shiftless_p_scale_validation_20260728/README.md).
The exact stabilized sweep is retained as an appendix diagnostic in
[`../fp4_fa4_p_scale_sweep_20260728/README.md`](../fp4_fa4_p_scale_sweep_20260728/README.md).

## Protocol

Each standard row uses HAO's tensor factory, seed 0, a 10 ms warmup, and a
100 ms `triton.testing.do_bench` median. Every route compiles at 128
registers, one barrier, 400 bytes of static shared memory, and zero spills.
Raw per-case records are excluded from Git; the compact tracked record is
[`summary.json`](summary.json).

Reproduce the matrix from `tk_fa4/fp4_fa4_fwd`:

```bash
python hao_comprehensive_suite.py \
  --variant fp4-matrix \
  --shape 1,1024,16,128 \
  --shape 1,4096,24,128 \
  --shape 1,8192,24,128 \
  --shape 1,32768,24,128 \
  --shape 4,4096,32,128 \
  --warmup-ms 10 --rep-ms 100 \
  --cooldown-seconds 0.2 --rebuild \
  --output-dir ../../results/fp4_fa4_format_matrix_20260728
```

# Fast shiftless NVFP4 probability-scale sweep

This experiment applies a global NVFP4 probability-scale lift `G` to the
actual fast shiftless kernel, not to the slower stabilized diagnostic. The
shape is B1/S4096/H24/D128 on one NVIDIA GB200, with HAO's seed-0 tensor
factory, a 10 ms warmup, and a 50 ms timing window.

## Direct result

`G` was not responsible for the earlier `0.188416 ms` result. That cost came
from restoring exact row-maximum stabilization and the real softmax path.
The fast shiftless kernel remains around `0.104 ms` with or without a
low-exponent scale lift.

| Route | G | Time (ms) | Cosine vs BF16 | Relative L2 vs BF16 | RMSE vs BF16 |
|---|---:|---:|---:|---:|---:|
| HAO native NV/NV | - | 0.193216 | 0.981415 | 0.192767 | 0.004947 |
| TK shiftless NV/NV | 1 | 0.104000 | **0.961779** | 0.279945 | 0.007184 |
| TK shiftless NV/NV | 1.25 | 0.102720 | 0.961766 | 0.279983 | 0.007185 |
| TK shiftless NV/NV | 1.5 | 0.102848 | 0.961777 | **0.279927** | **0.007183** |
| TK shiftless NV/NV | 1.625 | 0.102752 | 0.961771 | 0.279951 | 0.007184 |
| TK shiftless NV/NV | 1.75 | 0.102720 | 0.961769 | 0.279962 | 0.007184 |
| TK shiftless NV/NV | 4 | 0.103904 | **0.961779** | 0.279945 | 0.007184 |
| TK shiftless NV/NV | 5 | 0.102944 | 0.961766 | 0.279983 | 0.007185 |
| TK shiftless NV/NV | 5.5 | 0.102752 | non-finite | non-finite | non-finite |
| TK shiftless NV/NV | 8 | 0.103424 | non-finite | non-finite | non-finite |

The small timing differences are not a reproducible speedup. A longer
same-process replay measured:

- order `G=1`, then `G=1.5`: `0.104736`, `0.104416 ms`;
- order `G=1.5`, then `G=1`: `0.103072`, `0.104448 ms`.

The result is therefore timing-neutral at this resolution.

## Exponent rebasing

Multiplying `G` by a power of two retains the same E4M3 mantissa-grid phase.
Where neither value underflows nor overflows, it shifts the stored scale
exponent while the common factor cancels between the PV numerator and the
matching denominator. The sweep confirms bit-identical metrics for `G=1`,
`2`, and `4`, and for phase-equivalent pairs such as `1.25`/`2.5` and
`1.5`/`3`.

This lets a large stabilized factor be tested on the shiftless path at a
safe exponent:

| Stabilized phase | Rebased G | Fast-path relative L2 |
|---:|---:|---:|
| 320 | 1.25 | 0.279983 |
| 416 | 1.625 | 0.279951 |
| 448 | 1.75 | 0.279962 |

None beats `G=1`. A denser local search found its best relative L2 at
`G=1.5`, an improvement of only `0.0000172` on this one input. Factors
through `5` remained finite here; `5.5`, `6`, `6.5`, and `8` overflowed.
`G=7` happened to remain finite but was already degraded by encoder wrap, so
it is not a safe operating point.

## Generated code

Both `G=1` and `G=1.5` compile at 128 registers, zero stack bytes, and the
same shared-memory and barrier footprint. The non-power factor adds eight
`FADD` instructions across the eight unrolled scale encoders. No tensor,
MUFU, or branch count changes. The added arithmetic is mostly hidden, but
`G` is not literally one scalar instruction at runtime.

Cross-shape and multi-seed validation is recorded in
[`../fp4_fa4_shiftless_p_scale_validation_20260728/README.md`](../fp4_fa4_shiftless_p_scale_validation_20260728/README.md).
It does not support promoting `G=1.5`; the production default remains
`G=1`.

## Reproduction

From `tk_fa4/fp4_fa4_fwd`:

```bash
python hao_shiftless_p_scale_sweep.py \
  --output-dir \
  ../../results/fp4_fa4_shiftless_p_scale_sweep_20260728
```

The tracked [`summary.json`](summary.json) contains all 34 scale factors.
Raw cases, build logs, and extensions are intentionally untracked.

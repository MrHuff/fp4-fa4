# Hierarchical dQ consumed by projection backward on SM100

Date: 2026-08-15

Device: NVIDIA GB200/B300-class SM100, GPU 2.

Geometry: B=1, S=4096, Hq=32, Hkv=8, D=128, hidden=4096, causal.

## Outcome

The D128 GQA backward now has an exact projection-consumer route that does not
publish a completed standalone BF16 dQ tensor. Attention reduces dQ into two
private head-major BF16 lanes. The NVFP4 projection operand producer folds the
lanes in registers, applies inverse pair-native RoPE to dQ and dK, quantizes
the stacked [dQ | dK | dV] operand, and launches the existing persistent
projection.

The route is correct, but it is not a robust wall-time win. Two same-GPU
nine-sample runs bracketed it between 1.0% faster and 2.8% slower than the
materialized control. The materialized path therefore remains the runtime
default; the hierarchical path is retained as an explicit integration rung.

## Reduction and projection algebra

Let Delta Q_(r,k) be the dQ contribution for query tile r from causal key tile
k. Instead of reducing every contribution into one public tensor, attention
forms two private BF16 reduction lanes:

```text
L0[r] = Reduce_bf16({Delta Q_(r,k) : k is even})
L1[r] = Reduce_bf16({Delta Q_(r,k) : k is odd})
```

The completed dQ value exists only as a register expression in the projection
operand producer:

```text
dQ_reg[r] = bf16(L0[r] + L1[r])
```

With inverse pair-native RoPE R^-1 and delayed-scale NVFP4 quantization Q4,
the projection operand is

```text
G4 = Q4([R^-1(dQ_reg) | R^-1(dK) | dV]; s_G)
W4 = Q4(W_QKV; s_W)                         # cached across steps
dX_hat = Dequant(G4, s_G) @ Dequant(W4, s_W)^T
```

There is no intervening store of

```text
dQ_final = L0 + L1
```

to a caller-visible BF16 output. The two BF16 partial lanes remain in the
backward workspace, so this result eliminates final materialization rather
than all global dQ traffic.

## Correctness

Against the materialized route:

| Quantity | Relative L2 | Cosine | Max abs |
|---|---:|---:|---:|
| Folded hierarchical dQ | 0.003486 | 0.999994 | 1.91e-5 |
| Projected dX | 0.000658 | 0.9999997 | 9.77e-4 |

dK and dV are bit exact between the two attention routes. A separately tiled
projection producer agrees with the full hierarchical producer at effectively
unit cosine; the tiny nonzero delta in the final run comes from the independent
BF16 reduction ordering of its attention launch, not from tile packing.

## Timing

The primary clean run in
`tk_fa4_gqa_d128_hierarchical_projection_20260815.json` measured:

| Path | Median (us) | Relative to materialized |
|---|---:|---:|
| Materialized BF16 dQ -> QKV dgrad | 470.912 | 1.000x |
| Two-lane dQ -> register fold -> QKV dgrad | 466.112 | 1.010x |
| Tile-ready early producer -> QKV dgrad | 495.424 | 0.951x |

A fresh same-GPU repeat measured 451.200, 463.776, and 482.112 us
respectively. The exact hierarchical route was therefore 0.973x on the
repeat. A roughly one-percent result is below the run-to-run clock/load
variation here and must not be presented as a durable speedup.

The primary component medians expose the tradeoff:

| Component | Materialized (us) | Hierarchical (us) |
|---|---:|---:|
| Attention backward including clear | 329.440 | 325.664 |
| NVFP4 QKV dgrad projection | 90.688 | 98.144 |

Splitting the reduction removes about 3.8 us of exposed attention-side
publication/serialization, but consuming the second lane adds about 7.5 us to
the standalone projection. Combined-event overlap can move the balance by a
few microseconds in either direction.

The low-precision projection/backward boundary remains about 1.49x faster than
the favorable BF16 lower-bound chain. This experiment changes only the dQ
handoff and does not change the convergence-gating dV approximation.

## Why final materialization removal does not dominate

For this geometry, one BF16 dQ tensor is 32 MiB. The materialized path clears,
reduces into, and later reads that one tensor. The hierarchical path avoids the
completed tensor but writes and reads two 32 MiB partial lanes. The final add
is cheap and register-local; the extra 32 MiB lane is the physical cost that
largely cancels the attention-side benefit.

Both D128 producer specializations compile without spills. The materialized
control uses 96 registers; the two-lane producer uses 128 registers. Concurrent
TMA staging of both lanes did not reduce wall time because the second-lane
bandwidth remains real.

## Tile-ready experiment

The CuTe reducer can optionally publish one release-stamped owner word per
(query head, key tile) after its TMA reduction stores complete. A one-CTA gate
acquire-waits on an early head prefix, then a bounded producer publishes that
Q slice on a side stream. This is exact and avoids triangular per-query-tile
atomic fanout.

Sweeps over 1, 4, 8, and 16 early heads all lost. Four heads was least bad,
but the stable event was still 495.424 us. Producer admission, the extra pack
launch, and competition with the 16-warp attention CTA cost more than the
overlapped Q-prefix work. The readiness path remains diagnostic and is not the
default schedule.

## Remaining ceiling

The next genuine rung is not another external consumer schedule. It requires
the final dQ owner CTA to quantize or project its completed head tile in the
attention epilogue itself, or a projection-backward fusion that consumes an
on-chip reduction tile. Either removes the second global BF16 lane read. The
current two-lane global handoff has reached its traffic floor.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 \
  tk_fa4/lowp_fa4_bwd/profile_gqa_d128_chain.py \
  --direct-workspace-stats \
  --warmups 3 \
  --samples 9 \
  --tile-ready-early-heads 4 \
  --output results/tk_fa4_gqa_d128_hierarchical_projection_20260815.json
```

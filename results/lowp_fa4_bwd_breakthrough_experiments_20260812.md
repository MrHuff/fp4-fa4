# Low-precision FA4 backward breakthrough experiments — 2026-08-12

## Outcome

The retained adaptive FP4-Q/K + FP8-dP/dV kernel now reduces dQ globally in
BF16.  The existing API widens the completed result to FP32, while a new
projection-ready endpoint can return dQ directly in BF16 and skip that pass.

Retained compile-time state:

- `TK_FA4_BWD_FP8DPDV_BF16_DQ_REDUCTION=1`
- `TK_FA4_BWD_FP8DPDV_TWO_COMMAND_SCORE=0`
- `TK_FA4_BWD_FP8DPDV_HALF_DQ_REDUCTION_CEILING=0`
- `TK_FA4_BWD_FP8DPDV_SKIP_DQ_REDUCTION_CEILING=0`
- `TK_FA4_BWD_FP8DPDV_SKIP_DQ_WIDEN_CEILING=0`

The adaptive Q/K rule remains
`clip=max(0.325*amax, 2.75*RMS)`, with seven FP32 metadata words per
batch/head.  No score-rank approximation is enabled.

## Retained speedups

Changing the global dQ reduction buffer from FP32 to BF16 while preserving the
FP32 API reduced median time as follows:

| Shape | FP32 reduction | BF16 reduction + FP32 return | Improvement |
| --- | ---: | ---: | ---: |
| S4096/H24 | 0.660768 ms | 0.657408 ms | 0.51% |
| S8192/H8 | 0.801312 ms | 0.775392 ms | 3.23% |
| S8192/H24 | 2.029120 ms | 1.962528 ms | 3.28% |
| S8192/H64 | 5.094720 ms | 4.941568 ms | 3.01% |
| S16384/H24 | 6.955840 ms | 6.592192 ms | 5.23% |
| S16384/H64 | 18.302464 ms | 17.426336 ms | 4.79% |

The direct-BF16 endpoint then removes the BF16-to-FP32 epilogue and the full
FP32 dQ output allocation:

| Shape | FP32-return adaptive | Direct BF16 dQ | Direct win | Copied BF16 control | Direct vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| S4096/H24 | 0.657792 ms | 0.624832 ms | 5.01% | 0.704480 ms | 11.31% |
| S8192/H8 | 0.766368 ms | 0.746432 ms | 2.60% | 0.868864 ms | 14.09% |
| S8192/H24 | 1.956064 ms | 1.888768 ms | 3.44% | 2.207200 ms | 14.43% |
| S8192/H64 | 4.948224 ms | 4.761888 ms | 3.77% | 5.560512 ms | 14.36% |
| S16384/H24 | 6.605120 ms | 6.471232 ms | 2.03% | 7.743616 ms | 16.43% |
| S16384/H64 | 17.431936 ms | 17.068480 ms | 2.09% | 20.273727 ms | 15.81% |

The second table is one 5-warmup, 21-sample rotated run on GPU 0.  Its JSON
record is `results/lowp_fa4_bwd_bf16dq_validation_20260812.json`.

## API

The native binding is
`backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native`.  The public helper
selects it with:

```python
dq, dk, dv = b300_mha_bwd_adaptive_lowp(
    q, k, v, out, lse, dout, operands,
    return_bf16_dq=True,
)
```

This is intended for a Q-projection backward consumer that already accepts
BF16.  The default remains `return_bf16_dq=False`, preserving the FP32 dQ
contract.

## Correctness and stability

Across the six shapes above, all direct outputs were finite and BF16.  Against
the copied BF16 control:

- dQ cosine: 0.991435–0.991560
- dK cosine: 0.991379–0.991557
- dV cosine: 0.999345–0.999378

Direct BF16 dQ and the BF16 value underlying the FP32-return path agreed at
cosine 0.999998 or better, with maximum absolute difference at most
`9.536743e-7`.  The small difference is atomic accumulation order, not a new
approximation.  A 100-call S8192/H8 phase-reuse test remained finite; dK/dV
were repeat-exact and maximum dQ drift was `9.536743e-7`.

The retained full-D192 score path also passed a sparse-Q/K-outlier check:

| Shape | dQ cosine | dK cosine | dV cosine |
| --- | ---: | ---: | ---: |
| S4096/H24 | 0.962695 | 0.962139 | 0.999070 |
| S8192/H8 | 0.961676 | 0.963121 | 0.998915 |

Record: `results/lowp_fa4_bwd_adaptive_outlier_check_20260812.json`.

## Fresh profile

A 12-pass SourceCounters/Scheduler/WarpState capture of direct-BF16 dQ at
S8192/H8 produced:

- dense-kernel duration: 691.936 us (preprocessing excluded)
- tensor-pipe active: 21.04%
- issue active: 25.61%
- eligible warps: 0.43 per active cycle
- long-scoreboard stalls: 5.64 per issue-active cycle
- barrier stalls: 1.45 per issue-active cycle
- tensor instructions: 211,200
- DRAM traffic: 94,448,640 bytes
- registers: 128 per thread
- shared memory: 231,484 bytes
- spills/local stack: zero for the hot specialization

The source sample total was 8,220 long-scoreboard and 2,071 barrier not-issued
samples.  The largest long-scoreboard sites were:

| Dependency | Samples | Share of long-scoreboard |
| --- | ---: | ---: |
| compute-side `score_issued` wait | 3,025 | 36.8% |
| Q-source readiness, two sites | 744 | 9.1% |
| statistics-side `score_done` wait | 463 | 5.6% |
| `dp_done` wait | 226 | 2.7% |
| dQ TMA-store drain/publication | 215 | 2.6% |

This confirms that BF16 reduction removed meaningful downstream pressure and
that barriers are no longer the controlling class.  The remaining large
dependency is getting the score tensor work issued early enough for the
consumer, not FP4 arithmetic or dQ atomics.

Artifacts:

- `results/lowp_fa4_bwd_bf16dq_source_profile_20260812.ncu-rep`
- `results/lowp_fa4_bwd_bf16dq_source_profile_20260812_source.csv`
- `results/lowp_fa4_bwd_bf16dq_source_profile_20260812_raw.csv`

## Structural ceilings and rejected rungs

### Four-CTA/super-owner dQ reduction

A timing-only diagnostic suppressed every second dQ-owner publication after
dQ had reached registers.  It measured a gross ceiling of only 3.8% at
S8192/H8 (0.775392 to 0.745760 ms) and 3.0% at S16384/H24 (6.592192 to
6.392896 ms).  A real cluster-of-four implementation would have to spend part
of that budget on DSM coordination and replace several hard-wired rank-0/1
barrier contracts.  It is therefore not retained.

### Two-command D192-to-D128 score

Dropping one D64 score command slightly improved the calibrated timing
(S8192/H8 0.775392 to 0.762720 ms; S16384/H24 6.592192 to 6.545984 ms), but
failed the required accuracy envelope:

- Q/K standard deviation 1.0: dQ/dK cosine approximately 0.825
- 1% sparse outliers: dQ/dK cosine approximately 0.923–0.925
- energy-selected D64 chunks only marginally improved the outlier result
- CountSketch and dense Rademacher D192-to-D128 projections were worse on
  general inputs

This is a rank limitation, not a packing bug: arbitrary D192 inner products
cannot be preserved by a fixed rank-128 score operand.  The experiment remains
disabled.  It becomes valid only if the learned upstream projection itself
produces a genuinely rank-128 Q/K subspace.

### Full dQ-consumer fusion ceiling

Skipping completed dQ publication entirely gave an S8192/H8 timing-only floor
near 0.693 ms.  The native-BF16 API captures the safe, deployable part of that
idea.  Roughly another 7% relative to the direct-BF16 route remains available
only if the upstream Q-projection backward consumes the reduction tile without
materializing a standalone dQ tensor.

## Next breakthrough target

The highest-leverage remaining change is cross-kernel/model integration:

1. make the Q-projection epilogue emit the existing adaptive FP4 layouts and
   seven-word scale record, removing standalone Q/K reduction and packing;
2. consume native BF16 dQ immediately in projection backward;
3. for a larger step, fuse the dQ reduction tile directly into projection
   backward and remove global dQ publication;
4. only revisit two-command score if training constrains Q/K to a real
   rank-128 subspace.

Inside the standalone attention kernel, the remaining score-issue latency is
structural.  More barriers, a different atomic primitive, or another local
TMEM rearrangement is unlikely to produce a breakthrough at the current
128-register and 231-KB (about 226-KiB) shared-memory limits.

## Projection-backward integration

The first deployable integration rung now emits completed BF16 dQ, dK, and dV
directly into one per-head `[Q192, K192, V128]` buffer.  Projection backward
consumes that buffer without allocating three standalone gradients or running
an explicit concatenation.  The dQ zeroing needed by the cross-CTA reduction
runs on the existing auxiliary stream and overlaps mandatory preprocessing.

The public helper prepares a persistent interleaved projection weight once.
The projection consumer then chooses one, two, or four head-contiguous GEMMs
to avoid the poor cuBLAS shape that a single very-wide contraction produces:

| Family | Projection splits |
| --- | ---: |
| H8 | 1 |
| H16 | 2 |
| H24, S4096 | 1 |
| H24, S8192/S16384 | 2 |
| H64 | 4 |

Paired CUDA-event timings compare the old three-output attention plus three
projection GEMMs against the integrated attention plus auto-split projection:

| Shape | Old chain (ms) | Integrated chain (ms) | Saved (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| S4096/H24 | 0.931680 | 0.914048 | 0.017632 | 1.0193x |
| S4096/H64 | 3.276608 | 3.142528 | 0.134080 | 1.0427x |
| S8192/H8 | 0.844256 | 0.827200 | 0.017056 | 1.0206x |
| S8192/H16 | 1.581664 | 1.567712 | 0.013952 | 1.0089x |
| S8192/H24 | 2.407808 | 2.391808 | 0.016000 | 1.0067x |
| S8192/H64 | 8.144192 | 8.063680 | 0.080512 | 1.0100x |
| S16384/H24 | 7.480768 | 7.462944 | 0.017824 | 1.0024x |
| S16384/H64 | 24.055679 | 23.471905 | 0.583775 | 1.0249x |

The combined buffer preserves dK and dV bit-for-bit.  Its dQ differs by at
most `9.536743e-7` from the standalone BF16 route, with cosine approximately
0.999998 to 1.0; this is only atomic accumulation order.  Projection-gradient
cosine is approximately 0.999992 to 0.999996.  The small difference comes
from grouping the BF16 accumulation into one, two, or four GEMMs instead of
the old Q/K/V three-GEMM grouping.  Twenty repeated calls passed for every
shape above.

At S8192/H8 the paired stage medians moved from 0.753664 to 0.752224 ms for
attention and from 0.089760 to 0.075424 ms for projection.  Therefore this
rung removes API materialization and improves projection geometry, but does
not capture the no-publication floor: roughly 59 us remains between the
0.752-ms integrated attention stage and the 0.693-ms timing-only ceiling.

Reaching that floor requires a true reduction-tile handoff.  The last dQ
arriver must make each completed row/tile visible to a projection consumer,
which accumulates the corresponding head contribution before global dQ is
published.  Quantizing a partial dQ tile earlier is not mathematically valid,
because two CTA contributions still have to be reduced.  A tile-ready
producer/consumer pipeline is consequently the next structural experiment.

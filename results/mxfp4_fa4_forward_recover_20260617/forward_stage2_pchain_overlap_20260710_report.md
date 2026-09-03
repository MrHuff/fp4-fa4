# Stage2 P-Chain Overlap And Exp/Pack Optimization Report

Date: 2026-07-10

Task: `session6_stage2_pchain_overlap_20260710.md`

## Outcome

The Stage2 P chain is dominated by exp/weight and payload construction, not by the
P-scale TMEM store, payload proxy publication, or PV wakeup in isolation. The
low-perturbation Stage2 measurements put the P-scale store wait at about 65 cycles,
payload publication at 52 cycles, and ready-to-PV issue at 53-69 cycles. The full
P-scale-derived-to-ready span is about 1.9k cycles.

Four real candidates were implemented and profiled:

- Candidate A issued the P-scale TMEM store early and deferred its wait. It moved
  the store as intended, but its added live range increased the exp/pack span and
  did not produce a repeatable cross-shape win. Rejected and removed.
- Candidate B published true current-tile P/P-scale readiness before the current
  row-sum recurrence update. It exposed 72-82 cycles in the sampled chain, but its
  wall-time gain did not repeat on the supporting shapes. Rejected and removed.
- Candidate C combined A and B. It repeated at `-1.27%` on h4/s2048 and `-1.47%`
  on h8/s4096, and compact NCU confirmed lower duration and scoreboard pressure at
  h8/s4096. It regressed h8/s1024 by `0.98%`, so it is retained only as an explicit,
  default-off, shape-selective experiment. It is not the global default.
- The exact two-pack exp/pack window did not shorten the intended interval and its
  apparent first-pass h8/s4096 gain reversed. Rejected and removed.

Stage2 remains the global default. The useful sparse stamp infrastructure remains
behind `MXFP4_FWD_PCHAIN_STAMPS=0` by default.

## Build And Config Matrix

All performance builds used GPU2 and:

```text
MXFP4_FWD_TIMELINE=0
KPIPE_STAGE=2
SCORE_REUSE_PIPE_STAGE=0
HOTPLATE_SLOT_SCHED=0
HOTPLATE_POLICY=0
KPIPE_SELECTIVE_POLICY=0
POLICY126_COUNTERS=0
```

Attribution builds used `MXFP4_FWD_PCHAIN_STAMPS=1`; wall-time and NCU builds used
`MXFP4_FWD_PCHAIN_STAMPS=0`.

Exact runtime config strings:

| Route | Exact runtime config | Final disposition |
|---|---|---|
| Stage2 | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix` | Global reference |
| A | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pchaina_vtma_vstma_pstage2_q200_p112_o56_qkscfix` | Rejected, selector removed |
| B | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pchainb_vtma_vstma_pstage2_q200_p112_o56_qkscfix` | Rejected, selector removed |
| C | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pchainc_vtma_vstma_pstage2_q200_p112_o56_qkscfix` | Retained explicit/default-off |
| pack2 | `dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_arrivereuse_pscreusefold_skippscarrive_pack2_vtma_vstma_pstage2_q200_p112_o56_qkscfix` | Rejected, implementation removed |

## Implementation And Legality

Low-perturbation attribution:

- `clock64()` stamps are compiled only with `MXFP4_FWD_PCHAIN_STAMPS=1`.
- One quant lane atomically claims one sampled CTA and records the eight producer
  boundaries; the PV owner records the ninth boundary for the matching tile.
- The host reset/read API and benchmark driver collect repeated sparse samples.
- The stamps-off build has no stamp storage or marker writes. Both stamps-on and
  stamps-off selected kernels remained spill-free.

Candidate A:

- It uses the existing folded one-to-one P-stage/P-scale slot lifetime.
- A static assertion requires two folded payload/scale slots and x1 P scales.
- The x1 scale store is issued immediately after packed-scale derivation.
- The original late store is suppressed, but `fp4pv_tmem_store_wait()` remains at
  the original readiness boundary before `p_sc_tmem_ready` is published.
- No P-scale or payload readiness signal is issued early.

Candidate B:

- The existing correction prerequisite remains published at its original point.
- MXFP4 correction is `acc_scale` and does not depend on the updated current
  `row_sum`.
- Payload zeroing/stores, proxy publication, the P-scale TMEM wait, and P/P-scale
  ready publication all complete before the row-sum update is deferred.
- The current `row_sum = row_sum * acc_scale + tile_sum_scalar` executes immediately
  after readiness. The same quant owner cannot begin the next recurrence before it
  completes, so only current-tile PV can overlap it.

Candidate C directly enables both audited traits. After profiling, A and B were
removed as standalone configs, so C is the only behavior route left in the tree.

The pack2 candidate preserved pack order, accumulation order, payload bytes, and
native exact `EX2`/E2M1 conversion. SASS instruction counts were unchanged from
Stage2: 257 `MUFU.EX2`, 144 `F2FP`, and 128 E2M1 conversions. No low-precision
variant was attempted because the exact candidate did not pass the timing gate.

## Ptxas

| Config | Registers | Barriers | Static smem | Stack | Spill stores | Spill loads |
|---|---:|---:|---:|---:|---:|---:|
| Stage2 | 168 | 2 | 1904 B | 0 B | 0 B | 0 B |
| A | 168 | 2 | 1904 B | 0 B | 0 B | 0 B |
| B | 168 | 2 | 1904 B | 0 B | 0 B | 0 B |
| C | 168 | 2 | 1904 B | 0 B | 0 B | 0 B |
| pack2 | 168 | 2 | 1904 B | 0 B | 0 B | 0 B |

The final cleaned/default-off build reports the same footprint for retained C.

## Low-Perturbation Attribution

Median Stage2 cycles, 11 sparse samples per shape:

| Shape | scale->exp/pack | exp/pack->payload bytes | payload->publish | scale store issue->wait | publish->scale wait | scale wait->ready | publish->ready | scale->ready | exp/pack->row/corr | ready->PV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h4/s2048 | 1356 | 227 | 52 | 65 | 169 | 104 | 273 | 1908 | 90 | 65 |
| h8/s1024 | 1349 | 227 | 52 | 65 | 169 | 104 | 273 | 1901 | 91 | 67 |
| h8/s4096 | 1354 | 246 | 52 | 66 | 170 | 104 | 274 | 1922 | 89 | 69 |

This rules out a large isolated TMEM-store, proxy-publication, or consumer-wakeup
tail. The largest local block remains scale-to-exp/pack.

Candidate interval effects:

| Shape | A store issue->wait | A scale->ready | B scale->ready | B ready->row/corr | C scale->ready | C ready->row/corr |
|---|---:|---:|---:|---:|---:|---:|
| h4/s2048 | 1654 | 1916 | 1826 | 54 | 1840 | 54 |
| h8/s1024 | 1653 | 1918 | 1829 | 54 | 1831 | 54 |
| h8/s4096 | 1766 | 2026 | 1845 | 54 | 1925 | 54 |

A did overlap the 65-cycle store wait, but its scale-to-exp/pack interval increased
by 82, 91, and 149 cycles. B moved readiness ahead of row bookkeeping and reduced
scale-to-ready by 82, 72, and 77 cycles. C retained that split, but A's live-range
cost canceled it at h8/s4096 in the sparse sample.

For pack2, scale-to-exp/pack changed by `+4/-5/+16` cycles and scale-to-ready by
`+4/-3/+27` cycles on h4/s2048, h8/s1024, and h8/s4096. It did not move its target.

## Correctness

All candidates passed the finite h4/s1024 gate before larger shapes. The clean
repeats were finite at every requested shape. The maximum candidate-vs-Stage2
differences observed across the reverse-order repeats were:

| Candidate | Output max abs | Output max rel | LSE max abs | LSE max rel |
|---|---:|---:|---:|---:|
| A | 1.220703e-4 | 7.042253e-3 | 9.536743e-7 | 1.683095e-7 |
| B | 6.103516e-5 | 4.807692e-3 | 9.536743e-7 | 1.627524e-7 |
| C | 6.103516e-5 | 4.807692e-3 | 9.536743e-7 | 1.702612e-7 |
| pack2 | 3.051758e-5 | 4.237288e-3 | 9.536743e-7 | 1.702612e-7 |

These tiny deltas coincide with occasional Stage2 run-to-run nondeterminism at
h8/s1024 and h8/s4096. Candidate output was exact against the sampled Stage2 result
in the h4/s1024 gates, and the final cleaned C smoke was exact and deterministic.

Versus BF16 reference, Stage2 and candidates stayed in the same existing envelope:

| Shape | Output max abs | Output mean abs | RMSE | LSE max abs |
|---|---:|---:|---:|---:|
| h4/s2048 | 0.8125 | 0.006693 | 0.013787 | 0.029492 |
| h8/s1024 | 0.898438 | 0.009819 | 0.019236 | 0.029492 |
| h8/s4096 | 0.863281 | 0.005195 | 0.010798 | 0.019137 |

## Matched Wall Time

Reverse-order repeat, 30 samples per config. Negative delta is faster than the
Stage2 sample in the same run.

| Shape | Config | p50 ms | min ms | Delta vs Stage2 |
|---|---|---:|---:|---:|
| h4/s2048 | Stage2 | 0.066512 | 0.061696 | 0.00% |
| h4/s2048 | A | 0.066736 | 0.061440 | +0.34% |
| h4/s2048 | B | 0.066064 | 0.061856 | -0.67% |
| h4/s2048 | C | 0.065664 | 0.062656 | -1.27% |
| h8/s1024 | Stage2 | 0.044112 | 0.041888 | 0.00% |
| h8/s1024 | A | 0.044144 | 0.041824 | +0.07% |
| h8/s1024 | B | 0.044352 | 0.041856 | +0.54% |
| h8/s1024 | C | 0.044544 | 0.041888 | +0.98% |
| h8/s4096 | Stage2 | 0.104688 | 0.102304 | 0.00% |
| h8/s4096 | A | 0.103488 | 0.100928 | -1.15% |
| h8/s4096 | B | 0.104928 | 0.103008 | +0.23% |
| h8/s4096 | C | 0.103152 | 0.100480 | -1.47% |

The first A/B/C order produced C deltas of `-1.43%/-5.47%/-2.80%`. The reversed
run retained the h4/s2048 and h8/s4096 gains but flipped h8/s1024, which is why C is
shape-selective rather than a default replacement.

Pack2 reverse-order repeat:

| Shape | Stage2 p50/min ms | pack2 p50/min ms | Delta |
|---|---:|---:|---:|
| h4/s2048 | 0.067344/0.065248 | 0.066848/0.064512 | -0.74% |
| h8/s1024 | 0.049264/0.045664 | 0.048768/0.046528 | -1.01% |
| h8/s4096 | 0.109104/0.106560 | 0.109344/0.106880 | +0.22% |

Its first h8/s4096 pass was `-1.54%`; that result did not repeat.

## Compact NCU

Only C met the wall-time gate. A minimal three-replay NCU comparison was collected
at h8/s4096 with stamps and timeline disabled:

| Metric | Stage2 | C | Direction |
|---|---:|---:|---:|
| Duration | 88.256 us | 86.336 us | -2.18% |
| SM issue active | 30.44% | 31.05% | +0.61 pp |
| TC pipe active | 13.47% | 13.90% | +0.43 pp |
| Tensor pipe active | 6.28% | 6.43% | +0.15 pp |
| Long scoreboard | 3.47 | 3.40 | -0.07 |
| Wait | 1.75 | 1.72 | -0.03 |
| Barrier | 0.21 | 0.21 | unchanged |
| Eligible warps/cycle | 0.40 | 0.41 | +0.01 |
| Issued/cycle | 0.35 | 0.35 | unchanged |

NCU also confirmed identical `168` registers, `1904 B` static shared memory, and
two barriers. This supports the h8/s4096 C win, but does not override its h8/s1024
regression.

## Final State

Rejected A, B, and pack2 selectors/behavior were removed. Retained source changes:

- compile-gated sparse P-chain stamps and host read/reset API,
- the explicit `pchainc` config containing the combined audited behavior,
- the benchmark/attribution driver.

The final build used:

```text
MXFP4_FWD_TIMELINE=0
MXFP4_FWD_PCHAIN_STAMPS=0
KPIPE_STAGE=0
SCORE_REUSE_PIPE_STAGE=0
HOTPLATE_SLOT_SCHED=0
HOTPLATE_POLICY=0
KPIPE_SELECTIVE_POLICY=0
POLICY126_COUNTERS=0
```

The build succeeded. Final default/off h4/s1024 smoke was finite at p50/min
`0.041840/0.040608 ms`, with timeline raw/decoded `0/0`. Reading the compile-gated
P-chain stamp API returned an empty vector. The cleaned retained-C smoke was finite,
exact versus Stage2, and deterministic.

## Artifacts

- `forward_stage2_pchain_driver.py`
- `forward_stage2_pchain_build_stamps_abc_v3.log`
- `forward_stage2_pchain_build_timing_abc.log`
- `forward_stage2_pchain_build_pack2.log`
- `forward_stage2_pchain_build_stamps_pack2.log`
- `forward_stage2_pchain_build_ncu.log`
- `forward_stage2_pchain_stamps_h4_s2048_gpu2.json`
- `forward_stage2_pchain_stamps_h8_s1024_gpu2.json`
- `forward_stage2_pchain_stamps_h8_s4096_gpu2.json`
- `forward_stage2_pchain_timing_h4_s2048_gpu2.json`
- `forward_stage2_pchain_timing_h8_s1024_gpu2.json`
- `forward_stage2_pchain_timing_h8_s4096_gpu2.json`
- `forward_stage2_pchain_timing_repeat_h4_s2048_gpu2.json`
- `forward_stage2_pchain_timing_repeat_h8_s1024_gpu2.json`
- `forward_stage2_pchain_timing_repeat_h8_s4096_gpu2.json`
- `forward_stage2_pchain_pack2_stamps_h4_s2048_gpu2.json`
- `forward_stage2_pchain_pack2_stamps_h8_s1024_gpu2.json`
- `forward_stage2_pchain_pack2_stamps_h8_s4096_gpu2.json`
- `forward_stage2_pchain_pack2_timing_repeat_h4_s2048_gpu2.json`
- `forward_stage2_pchain_pack2_timing_repeat_h8_s1024_gpu2.json`
- `forward_stage2_pchain_pack2_timing_repeat_h8_s4096_gpu2.json`
- `forward_stage2_pchain_pack2_build_baseline_kernel.sass`
- `forward_stage2_pchain_pack2_kernel.sass`
- `forward_stage2_pchain_ncu_stage2_h8_s4096_gpu2_raw.csv`
- `forward_stage2_pchain_ncu_c_h8_s4096_gpu2_raw.csv`
- `forward_stage2_pchain_restore_default_off_build.log`
- `forward_stage2_pchain_restore_default_off_h4_s1024_gpu2.json`
- `forward_stage2_pchain_restore_default_off_h4_s1024_gpu2_events.csv`
- `forward_stage2_pchain_restore_default_off_h4_s1024_gpu2_summary.json`
- `forward_stage2_pchain_retained_c_final_smoke_h4_s1024_gpu2.json`

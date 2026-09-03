# Issue-Lane Overlap And Broad BF16 Matrix Report

Date: 2026-07-11

Task: `session6_issue_lane_overlap_bf16_matrix_20260711.md`

## Result

The V-only and V+P overlap schedules were implemented, compiled, stress-tested,
profiled, and timed. Both schedules produced the intended local issue-lane
reordering, but neither produced a repeatable wall-time win. Both also failed the
strict reordering-only bit-identical gate on larger shapes, within the same small
nondeterministic envelope already present in Stage2/e16pc. The schedule routes,
traits, host selectors, and compile gate were removed.

The exact direct-rescale consumption was retained as one local helper at
`fwd_streaming_kernel.inc:16751`; ordinary Stage2/e16pc consume it at the original
location before QK/PV issue (`:16781-16785`). Under the matched `KPIPE_STAGE=2`
selection build, normalized Stage2 and e16pc SASS was byte-identical before and
after this helper refactor.

The broad result does **not** meet the product objective:

- 57 requested cells were accounted for.
- 37 are measured robust FP4 wins (`>1.02x`).
- one is a narrow FP4 win: b1/s1024/h32 at `1.0078x` after 60 samples.
- 10 are measured FP4 losses.
- nine S=16384/32768 cells have finite BF16 but no finite FP4 route under bounded
  fresh-process testing.
- 23 cells do not have a strict comparison because a finite FP4 route was not
  available or bitwise determinism failed. Measured comparisons are still reported
  separately and never promoted to strict results.

No new shape dispatch was retained. No commit or push was performed.

## Baseline Reproduction

GPU2 was checked for competing compute processes before timing batches. Inputs and
outputs were pre-created; timing excludes quantization. Each row below is
`p50/min` in milliseconds from 30 samples.

| Shape | Order | Stage2 | e16pc |
|---|---|---:|---:|
| h4/s2048 | first | 0.065568 / 0.063296 | 0.062704 / 0.060448 |
| h4/s2048 | reverse | 0.065504 / 0.063680 | 0.061840 / 0.059104 |
| h8/s1024 | first | 0.047056 / 0.043840 | 0.045232 / 0.042976 |
| h8/s1024 | reverse | 0.046560 / 0.044544 | 0.046128 / 0.044032 |
| h8/s4096 | first | 0.105968 / 0.100416 | 0.096080 / 0.094496 |
| h8/s4096 | reverse | 0.105760 / 0.102944 | 0.100720 / 0.097536 |
| h16/s4096 | first | 0.183776 / 0.180928 | 0.174672 / 0.170144 |
| h16/s4096 | reverse | 0.179824 / 0.177504 | 0.173008 / 0.169696 |

Raw p25/p75/max and samples are in the corresponding
`forward_issue_lane_overlap_bf16_matrix_20260711_baseline_*.json` artifacts.

## Ownership Audit And Implementation

The candidate schedules changed only the location of the exact previous-output
rescale consume:

- Schedule V: V stage -> rescale consume -> P descriptor/scale -> PV.
- Schedule VP: V stage -> P descriptor/scale -> rescale consume -> PV.

The source audit established the candidate ordering was mechanically legal:

- `wait_and_stage_v_sc_for_iter` and `wait_and_stage_p_sc_for_iter` only delegate
  to existing stage helpers (`fwd_streaming_kernel.inc:15457-15470`). They do not
  address the output accumulator protected by `direct_rescale_finished`.
- `issue_pv_for_iter` is the first wrapper that reaches `issue_pv`
  (`:15471-15476`). The TCGEN issue touches the output accumulator in
  `issue_pv_stage` (`:10131-10280`).
- P payload and P-scale reuse are published only after PV TCGEN issue/commit
  (`:10282-10352`). Merely observing the P descriptor/scale does not release the
  slot.
- The helper consumes the same corr slot, phase bit, semaphore, and iteration as
  the original code (`:16751-16779`). First-iteration and QK behavior are unchanged.

No full-CTA barrier, spin loop, global state, P stage, score slot, or TMEM slot was
added.

## SASS And Resources

Matched diagnostics-off `KPIPE_STAGE=2` normalized hashes:

| Route | Normalized hash | Instructions | EX2 | paired FFMA2 | F2FP | E2M1 | BRA |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage2 | `21bf0fcce0c551e7b87a55e0b7895a59173022036cd6b907e61b14f96cb2c031` | 4688 | 257 | 64 | 144 | 128 | 263 |
| e16pc | `03a47d55edbbef7ddeae56d3249db7e7b5b85f8f6da0911854f180e3fbf4ce13` | 4952 | 193 | 160 | 144 | 128 | 263 |
| Schedule V | `513b18d02c698ec7963e4e37c0e4b0494657c54031f112fd03265aa885171291` | 4944 | 193 | 160 | 144 | 128 | 261 |
| Schedule VP | `aaa007fb952bc209f63afb1ed3f6ee26305963826ec13195ed365b6088a68019` | 4944 | 193 | 160 | 144 | 128 | 261 |

Stage2 and e16pc hashes were unchanged by the helper refactor. All four selected
kernels used 168 registers, zero stack, zero local memory, 2928 B shared usage in
`cuobjdump`, and 3724 B constant[0]. Selected ptxas reported no spills. The final
default `KPIPE_STAGE=0` build retains the same 168-register/zero-stack/zero-local
resource result; its SASS differs from the matched KPIPE2 profile as expected from
the global policy macro.

## Correctness And Stress

- h4/s1024 smoke was finite for e16pc, V, and VP.
- Score-iteration tails 1, 2, 3, and 8 (S=128, 256, 384, 1024) were finite and
  candidate output/LSE was exact against e16pc in those runs.
- h32/s4096 completed more than 100 bounded warmup/timed launches per route.
- Sparse idx0/idx2 profiling produced 11/11 valid owner-correlated records for
  every route on h4/s2048, h8/s1024, and h8/s4096.
- At h8/s4096 and h16/s4096, Stage2/e16pc itself is intermittently non-bitwise
  deterministic. Candidate-vs-e16pc maximum differences reached output
  `0.0019226074` and LSE `0.0005817413`, inside that inherited envelope but not
  bit-identical. The task's reordering-only contract therefore rejects both
  candidates independently of timing.
- Paired CuTe checks on isolated matrix cells stayed inside the FP4 numerical
  envelope: output max absolute error `0.8242-1.1875`, LSE max absolute error
  `0.0126-0.0290`. Those checks also reproduced non-bitwise FP4 determinism, so
  they are not reported as strict comparisons.

The TK BF16 kernel is source-unsupported below S=512: `get_tile_idx` computes
`num_block=(S/128)/(2 softmax WGs * 2 CTA cluster)`, which is zero. CuTe BF16 was
used as `BF16_best` on S=128/256 rather than dropping those cells.

## Sparse Schedule Signature

Columns are median cycles: issue entry to output-ready/V-ready/P-descriptor,
producer-P-ready to PV, and output-ready to PV.

| Idx/shape | Route | entry->out | entry->V | entry->Pdesc | producer->PV | out->PV |
|---|---|---:|---:|---:|---:|---:|
| idx0 h4/s2048 | e16pc | 80 | 1311 | 1397 | 499 | 4394 |
| idx0 h4/s2048 | V | 1372 | 1274 | 1463 | 499 | 3040 |
| idx0 h4/s2048 | VP | 4402 | 1289 | 1385 | 590 | 143 |
| idx0 h8/s1024 | e16pc | 79 | 1292 | 1385 | 532 | 4384 |
| idx0 h8/s1024 | V | 1381 | 1281 | 1470 | 490 | 3064 |
| idx0 h8/s1024 | VP | 4351 | 642 | 724 | 607 | 149 |
| idx0 h8/s4096 | e16pc | 77 | 1375 | 1470 | 456 | 4370 |
| idx0 h8/s4096 | V | 1433 | 1350 | 1513 | 453 | 3034 |
| idx0 h8/s4096 | VP | 4481 | 1499 | 1576 | 540 | 134 |
| idx2 h4/s2048 | e16pc | 4048 | 4482 | 4566 | 1024 | 970 |
| idx2 h4/s2048 | V | 4009 | 513 | 4127 | 650 | 593 |
| idx2 h4/s2048 | VP | 4571 | 525 | 608 | 714 | 140 |
| idx2 h8/s1024 | e16pc | 3039 | 3356 | 3433 | 410 | 2031 |
| idx2 h8/s1024 | V | 3147 | 314 | 3224 | 410 | 1870 |
| idx2 h8/s1024 | VP | 5164 | 321 | 414 | 628 | 131 |
| idx2 h8/s4096 | e16pc | 3807 | 4287 | 4366 | 1101 | 1046 |
| idx2 h8/s4096 | V | 4009 | 447 | 4112 | 653 | 614 |
| idx2 h8/s4096 | VP | 4586 | 462 | 544 | 761 | 163 |

The intended overlap is real: V moves before the rescale observation in V/VP, P
moves before it in VP, and `out->PV` drops as low as 131-163 cycles. It does not
translate to wall time. VP increases issue-P-ready to PV from 141 to 366 cycles at
h4/s2048 idx2 and from 132 to 442 cycles at h8/s4096 idx2. That extra issue/control
and held-P pressure consumes the local overlap.

## Candidate Timing And Decision

Each timing cell below is `p50/min/p25/p75/max` ms from 30 samples. `*` marks a
visibly shifted batch; it was recorded but excluded from selection.

| Shape/order | Route | p50/min/p25/p75/max |
|---|---|---|
| h4/s2048 first* | e16pc | 0.072784/0.067328/0.071384/0.074936/0.082752 |
| h4/s2048 first* | V | 0.072336/0.069696/0.071064/0.074304/0.082112 |
| h4/s2048 first* | VP | 0.070240/0.067392/0.068896/0.071920/0.078976 |
| h4/s2048 reverse | e16pc | 0.064224/0.060992/0.063424/0.064696/0.066560 |
| h4/s2048 reverse | V | 0.064656/0.061280/0.063040/0.066312/0.076544 |
| h4/s2048 reverse | VP | 0.064368/0.061728/0.063496/0.065024/0.069568 |
| h8/s1024 first | e16pc | 0.046480/0.044672/0.045592/0.047336/0.053120 |
| h8/s1024 first | V | 0.046176/0.043520/0.045472/0.046968/0.052736 |
| h8/s1024 first | VP | 0.046992/0.043712/0.045624/0.048104/0.055296 |
| h8/s1024 reverse* | e16pc | 0.055200/0.052192/0.053696/0.055688/0.059776 |
| h8/s1024 reverse* | V | 0.055984/0.051424/0.053016/0.058096/0.062688 |
| h8/s1024 reverse* | VP | 0.054864/0.051040/0.053152/0.058408/0.064480 |
| h8/s4096 first | e16pc | 0.100512/0.098880/0.099880/0.101520/0.106272 |
| h8/s4096 first | V | 0.100320/0.097632/0.099376/0.101504/0.112384 |
| h8/s4096 first | VP | 0.100944/0.098944/0.099336/0.102144/0.109056 |
| h8/s4096 reverse | e16pc | 0.099488/0.096992/0.098432/0.100800/0.108512 |
| h8/s4096 reverse | V | 0.099760/0.095968/0.098984/0.101096/0.137664 |
| h8/s4096 reverse | VP | 0.098960/0.095584/0.097368/0.100448/0.110816 |
| h16/s4096 first | e16pc | 0.175072/0.172704/0.173608/0.176016/0.181280 |
| h16/s4096 first | V | 0.175600/0.171776/0.174600/0.177784/0.182592 |
| h16/s4096 first | VP | 0.174336/0.171744/0.173352/0.175856/0.179360 |
| h16/s4096 reverse | e16pc | 0.171024/0.168352/0.169704/0.171640/0.179648 |
| h16/s4096 reverse | V | 0.174080/0.171488/0.172856/0.174976/0.180160 |
| h16/s4096 reverse | VP | 0.173984/0.169856/0.172576/0.176672/0.185472 |

No route repeated a 1.5% gain on two core shapes. The only apparent large result
was VP in the shifted h4/s2048 first batch; reverse order was a 0.22% regression.
No candidate advanced to 60-sample confirmation or finalist NCU. V and VP were
removed.

## Broad Matrix Method

`forward_issue_lane_overlap_bf16_matrix_20260711_driver.py` runs each requested
cell in a fresh process with BF16 source inputs created once, MXFP4 quantization
outside timing, preallocated output/LSE tensors, 20 warmups, and 30 raw event-timed
samples. Route order is rotated and reversed within each worker. Legal TK BF16
persistent/fullgrid and CuTe `_flash_attn_fwd(out=..., lse=...)` are both measured.

Several mixed-route workers hung. They were not discarded: each implementation was
rerun in its own bounded fresh process and recorded in isolated manifests. Paired
short checks supplied CuTe correctness references. This recovered finite S8192
Stage2/auto results and proved the remaining S16384+ behavior is a route-local
timeout, not merely mixed-route contamination.

The within-2% b1/s1024/h32 cell was rerun with 20 warmups and 60 samples. It moved
from `0.994x` in the first matrix pass to `1.00779x`, a `+0.416 us` FP4 margin, so
it is a narrow win but not a robust one.

## Primary Heatmap

Entries are `BF16_best / FP4_best` using retained Stage2/e16pc/auto routes only.
`TO` means BF16 is finite but every bounded FP4 route timed out or is dispatch
unsupported. Robust wins are `>1.02x`.

| S \ H | 1 | 2 | 4 | 8 | 16 | 32 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 3.565 | 3.349 | 3.048 | 3.663 | 3.024 | 3.323 |
| 256 | 2.994 | 2.912 | 2.796 | 3.016 | 2.793 | 3.076 |
| 512 | 1.128 | 1.113 | 1.126 | 1.164 | 1.165 | 1.128 |
| 1024 | 1.180 | 1.201 | 1.240 | 1.226 | 1.166 | 1.008 |
| 2048 | 1.226 | 1.315 | 1.296 | 1.317 | 1.190 | 0.889 |
| 4096 | 1.225 | 1.351 | 1.366 | 1.365 | 0.857 | 0.815 |
| 8192 | 1.259 | 1.440 | 1.452 | 0.804 | 0.742 | 0.628 |
| 16384 | TO | TO | TO | TO | TO | TO |

Primary totals: 35 robust wins, one narrow win, six losses, six no-finite-FP4
cells. The stable measured boundary is H<=16 at S2048, H<=8 at S4096, and H<=4
at S8192.

## Exact Losing Cells

| Cell | FP4 best (mode, ms) | BF16 best (mode, ms) | Speedup | Margin us |
|---|---|---|---:|---:|
| b1/s2048/h32 | auto persistent, 0.118016 | TK fullgrid, 0.104912 | 0.889 | -13.104 |
| b1/s4096/h16 | e16pc persistent, 0.165664 | TK fullgrid, 0.141952 | 0.857 | -23.712 |
| b1/s4096/h32 | e16pc persistent, 0.314496 | CuTe native, 0.256224 | 0.815 | -58.272 |
| b1/s4096/h64 | Stage2 persistent, 0.630976 | CuTe native, 0.437136 | 0.693 | -193.840 |
| b1/s8192/h8 | e16pc fullgrid, 0.287088 | TK fullgrid, 0.230912 | 0.804 | -56.176 |
| b1/s8192/h16 | e16pc fullgrid, 0.533088 | CuTe native, 0.395472 | 0.742 | -137.616 |
| b1/s8192/h32 | Stage2 fullgrid, 1.108048 | CuTe native, 0.696384 | 0.628 | -411.664 |
| b1/s8192/h64 | Stage2 fullgrid, 2.194640 | CuTe native, 1.305632 | 0.595 | -889.008 |
| b2/s4096/h8 | Stage2 persistent, 0.173680 | TK fullgrid, 0.143728 | 0.828 | -29.952 |
| b4/s4096/h4 | e16pc fullgrid, 0.170080 | TK fullgrid, 0.145824 | 0.857 | -24.256 |

Batch factorization confirms total work matters: b2/s1024/h8 and b4/s1024/h4
remain robust wins (`1.175x`, `1.206x`), while their S4096 counterparts lose.
b4/s4096/h4 persistent hangs for Stage2/e16pc/auto; forced fullgrid is finite but
still loses, so a batch-aware fullgrid dispatch would repair liveness without
meeting the performance objective and was not retained here.

No-finite-FP4 cells and finite fastest BF16 p50:

- b1/s16384/h1,h2,h4,h8,h16,h32: CuTe `0.419856, 0.394592, 0.399136,
  0.678176, 1.238944, 2.389040` ms.
- b1/s32768/h1,h4,h16: CuTe `0.710224, 1.255472, 4.550080` ms.

H=1 explicit Stage2/e16pc is dispatch-unsupported; auto timed out. At H>=2,
Stage2, e16pc, and auto each timed out independently after 20 seconds with both
auto/fullgrid and the representative explicit-persistent probe at s16384/h2.

Selected-route p50/min/p25/p75/max for every finite cell, all route statuses, raw
artifact provenance, strict status, winning/parity/losing lists, and the complete
heatmap are in `forward_issue_lane_overlap_bf16_matrix_20260711_summary.json`.
Raw samples remain in the per-cell and isolated route JSON files.

## Bounded Gap Follow-Up

Forced launch-mode results did not close either measured loss cluster:

- b1/s4096/h16 best candidate was VP persistent at `0.165776` ms; e16pc fullgrid
  was `0.166960` ms. BF16 best is `0.141952` ms. Persistent/fullgrid changes are
  much smaller than the `23.7-24.0 us` deficit.
- b1/s4096/h64 e16pc was `0.607312` persistent and `0.608496` fullgrid versus
  `0.437136` ms BF16.
- b1/s8192/h16 VP was `0.533344` persistent and `0.534080` fullgrid versus
  `0.395472` ms BF16.
- b2/s4096/h8 e16pc improved from `0.169648` persistent to `0.165920` fullgrid,
  but BF16 remains `0.143728` ms.
- s16384/h2 timed out in explicit persistent mode as well as fullgrid/auto.

No stable winning dispatch boundary exists in the loss region.

## Losing-Cell Profiles

Stage2 sparse medians on representative losses:

| Idx/shape | Producer chain | entry->out | entry->V | entry->Pdesc | producer->issue-P | issue-P->PV | out->PV |
|---|---:|---:|---:|---:|---:|---:|---:|
| idx0 h16/s4096 | 4248 | 79 | 1379 | 1509 | 288 | 138 | 4628 |
| idx0 h16/s8192 | 4237 | 87 | 1463 | 1543 | 289 | 147 | 4635 |
| idx2 h16/s4096 | 4283 | 3867 | 4282 | 4368 | 485 | 133 | 929 |
| idx2 h16/s8192 | 4304 | 3875 | 4294 | 4379 | 502 | 136 | 939 |

Each row has 11/11 valid owner-correlated records. Steady-state PV follows
issue-side P readiness in only 133-136 cycles. The exposed interval is dominated by
the roughly 4.3k-cycle P chain and late P readiness, not the direct-rescale consume.

Compact five-replay NCU:

| Shape | Duration us | Issue % | TC % | Tensor % | ALU % | FMA % | XU inst | Eligible | long SB | short SB | wait | barrier | no-inst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h16/s4096 | 165.280 | 32.46 | 14.35 | 6.70 | 21.75 | 15.69 | 4,584,448 | 0.41 | 3.46 | 0.52 | 1.75 | 0.20 | 0.49 |
| h16/s8192 | 566.368 | 34.91 | 16.32 | 7.68 | 23.78 | 17.01 | 17,885,184 | 0.42 | 3.51 | 0.48 | 1.74 | 0.19 | 0.40 |

Both use 168 registers, 1904 B static and 100352 B dynamic shared memory; waves
per SM are `3.37` and `6.74`. The signatures are dependency/scoreboard limited,
not barrier limited, and remain ready-warp poor despite adequate grid waves.

The recoverable requirement is therefore much larger than a local wait reorder:
about 24 us at h16/s4096, 58 us at h32/s4096, 138 us at h16/s8192, and 412-889 us
at h32/h64 s8192. Closing these gaps needs a lower-cost P/softmax/PV chain or a
different long-sequence/high-task kernel decomposition. Another issue-lane
schedule cannot supply those margins.

## Artifacts

Core machine-readable artifacts:

- `forward_issue_lane_overlap_bf16_matrix_20260711_driver.py`
- `forward_issue_lane_overlap_bf16_matrix_20260711_summarize.py`
- `forward_issue_lane_overlap_bf16_matrix_20260711_matrix.json`
- `forward_issue_lane_overlap_bf16_matrix_20260711_summary.json`
- `forward_issue_lane_overlap_bf16_matrix_20260711_isolated_*.json`
- `forward_issue_lane_overlap_bf16_matrix_20260711_gap_*.json`
- `forward_issue_lane_overlap_bf16_matrix_20260711_stamps_*.json`
- `forward_issue_lane_overlap_bf16_matrix_20260711_gap_stamps_*.json`
- `forward_issue_lane_overlap_bf16_matrix_20260711_ncu_*.csv`
- `forward_issue_lane_overlap_bf16_matrix_20260711_default_off_smoke.json`

## Final State

Rejected schedule behavior and selectors are absent. The useful host-only matrix
and summary tooling remains, along with the SASS-neutral exact rescale helper and
the previously retained compile-gated sparse profiler.

The forced clean final build succeeded with:

```text
MXFP4_FWD_TIMELINE=0
MXFP4_FWD_PCHAIN_STAMPS=0
PCHAIN_TARGET_IDX=0
KPIPE_STAGE=0
SCORE_REUSE_PIPE_STAGE=0
KPIPE_SELECTIVE_POLICY=0
HOTPLATE_SLOT_SCHED=0
HOTPLATE_POLICY=0
POLICY126_COUNTERS=0
```

Ordinary Stage2 h4/s1024 is finite and deterministic at `0.051472/0.046144` ms
p50/min. Timeline is `[]`, P-chain raw is `[[]]`, and all 64 policy counters are
zero. Scoped `git diff --check` passes.

External branch movement was observed while preserving the dirty worktree: the
thread had previously recorded `d0b2e06beb4fc0f65fb3c2946d1186c065fc8041`, then
`9a66b7524722e5d08b1efa2731b3910174a868a1`, and ended at
`ee3bd37cf76ea324c52c02f218daa67efe1f5c33`; local HEAD and origin match. This
task did not commit or push.

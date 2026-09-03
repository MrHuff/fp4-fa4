# Large-Sequence And High-Head MXFP4 Adaptation

Date: 2026-07-11/12 UTC
GPU: NVIDIA GB200, GPU2
Result: retain split-full only for `B=1, H=1, S>=16384`; reject split-K64 and all
high-head split dispatch; fix batch-factorized persistent-grid liveness; reject the
materialized-P fallback.

## Executive Result

This pass implemented and debugged the requested modern two-producer,
score-derived e16pc split-P path in both full-P and legal K64-progressive forms.
Both are finite and numerically match the retained e16pc error envelope. The route
does not solve the high-head throughput loss: at H16/S4096 split-full is 0.1916 ms
versus e16pc at 0.1721 ms and best BF16 at 0.1436 ms. NCU shows that splitting raises
eligible warps and issue activity but lowers TC utilization, so the extra producer,
rendezvous, and control work dominates.

There is one repeatable useful regime. On 60 samples:

| Cell | old auto | split-full | best BF16 | split vs old auto | split vs BF16 |
|---|---:|---:|---:|---:|---:|
| B1/S16384/H1 | 0.330544 ms | 0.326560 ms | 0.400448 ms | 1.0122x | 1.2263x |
| B1/S32768/H1 | 0.642288 ms | 0.623552 ms | 0.717200 ms | 1.0300x | 1.1502x |

The host therefore selects split-full only when `batch == 1 && heads == 1 &&
seqlen >= 16384`. No low/medium-shape route changed.

The liveness audit also found a separate batch-factorized launch bug. The persistent
safety predicate used `heads` rather than `batch * heads`; B4/S4096/H4 therefore
entered an unsafe persistent decomposition and saturated GPU2 indefinitely. The
predicate now uses `batch_heads`, forcing that cell to finite FP4 fullgrid execution.

## Source Changes

- Added `config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split` and its K64-derived
  config in `fwd_configs.inc:695`, with a dedicated
  `ONLINE_SCORE_DERIVED_SPLIT_P_QUANT_2WG` trait at `fwd_configs.inc:2270`.
- Added split score/max/scale/denominator shared state and narrowly selected
  semaphores in `fwd_streaming_kernel.inc:4453`.
- Implemented the two score-derived producers at
  `fwd_streaming_kernel.inc:21134`. WG1 owns columns 0..63, WG2 owns columns
  64..127, WG0 issues QK/PV/output, and WG3 retains the input producer role.
- Added explicit full-P and K64 route names in `fwd_host_dispatch.inc:2480`,
  `:2675`, and `:3376`.
- Added the measured B1/H1 auto gate at `fwd_host_dispatch.inc:2451`.
- Made both normal and K256 persistent safety checks batch-aware at
  `fwd_host_dispatch.inc:2324` and `:2378`.
- Extended compile-gated P-chain stamps with per-half clocks. The runtime target is
  accepted only when `MXFP4_FWD_PCHAIN_STAMPS=1`; default/off has no stamp SASS or
  storage.
- Removed the temporary runtime stage-0/1/2 split bisection ladder from the selected
  hot path before final timing.

## Liveness Audit

The task's starting S16384 timeout was stale for the current tree. The existing host
guard already forces fullgrid for long low-head jobs. Fresh-process probes found:

- Stage2 finite for H2/H4 at S8192, 9984, 10240, 12288, 14336, and 16384.
- Stage2, e16pc, and auto finite at S16384 H2/H4 and S32768 H1/H2/H4.
- An explicit `persistent` request at these long shapes is safely converted to
  fullgrid; it is not evidence that the persistent kernel itself is legal.
- The new split routes are finite at S128, S256, S384, S1024, S8192, and S16384.
- The 57-cell final matrix has zero non-finite FP4 cells.

The one newly reproduced hang was B4/S4096/H4 under the old safety predicate. Its
equivalent work count is 16 heads, but `heads == 4` incorrectly admitted persistent
execution. After switching the guard to `batch * heads`, the same auto request is
finite and runs through fullgrid FP4. This is a launch-decomposition fix, not a BF16
fallback or timeout increase.

The bounded host head-chunk probe was rejected. Concurrent chunks were slower than
the single launch in every requested case, for example S4096/H16 was 0.189728 ms for
2xH8 versus 0.167840 ms single-launch and 0.144544 ms BF16. Concatenated output also
did not validate within the fused route's envelope, so no chunk dispatch was added.

## Split-P Implementation

The final route preserves retained e16pc semantics:

1. Both quant WGs load disjoint 64-column score halves and compute two 32-column
   block maxima each.
2. WG0 combines all four maxima with `row_max_old`, computes the exact new row max
   and `acc_scale`, and publishes common state.
3. Each producer derives its two floor-E8M0 scale bytes and matching log2/reciprocal
   coefficients. WG0 combines all four bytes and issues the P-scale TMEM store.
4. Each producer runs the retained cadence-16 paired ALU exp2 path, hardware E2M1
   conversion, disjoint payload stores, and an independent denominator partial.
5. Full-P waits for both payload producers before K128 PV. K64 publishes each legal
   half independently and issues the existing K64 chunks in order.
6. A dedicated final rendezvous combines denominator partials. P payload and scale
   reuse remains tied to completed PV consumption.

The main bugs found and repaired during staged bring-up were:

- inherited finalization waited on a stats semaphore that this ownership mode did
  not produce;
- unconditional early correction publication created an idx1 wait cycle;
- WG-level publication could release rows before all four warps had written;
- reusing an immediate two-phase rendezvous at finalization deadlocked slot reuse;
- the first sparse sampler assumed the old issue-lane thread after the role map
  moved the producer/issue WG.

The final synchronization model uses one common-state owner, per-warp arrivals where
all four warps publish row data, and a dedicated final denominator rendezvous. There
are no hot-path CTA barriers or spinning waits.

## Correctness And Lifecycle

- At S512, e16pc, split-full, and split-K64 have identical BF16 comparison metrics:
  output max abs 0.98828125 and LSE max abs 0.0162383.
- At S1024, all three again match: output max abs 0.78125 and LSE max abs 0.0219445.
- Split route run-to-run output is exact in the finite correctness and finalist
  probes. LSE drift is zero or at most 1.91e-6, no worse than retained e16pc.
- The mixed lifecycle stress ran 200 launches across ten S/H shapes and both split
  handoffs. Every launch was finite; max run-to-run output and LSE deltas were zero.
- The required matrix supplies at least 30 launches at H4/H8/H16/H32 and at
  S8192/S16384/S32768. The H1 finalists supply 60 launches each.

Per-half sparse clocks at H16/S4096 were collected for idx0, idx2, idx16, and idx31.
All sampled halves agreed on block/task owner identity. Representative medians:

| idx | route | half0 max->pack | half1 max->pack | pack skew | ready->PV | PV->reuse |
|---:|---|---:|---:|---:|---:|---:|
| 0 | split-full | 2809 cyc | 2751 cyc | 65 cyc | 531 cyc | 2993 cyc |
| 0 | split-K64 | 2798 cyc | 2708 cyc | 75 cyc | 323 cyc | 3623 cyc |
| 2 | split-full | 2664 cyc | 3053 cyc | 93 cyc | 520 cyc | 3097 cyc |
| 2 | split-K64 | 2561 cyc | 3081 cyc | 68 cyc | 292 cyc | 3682 cyc |
| 16 | split-full | 2312 cyc | 2729 cyc | 67 cyc | 503 cyc | 3102 cyc |
| 16 | split-K64 | 2291 cyc | 2755 cyc | 67 cyc | 287 cyc | 3511 cyc |
| 31 | split-full | 1982 cyc | 2435 cyc | 127 cyc | 469 cyc | tail |
| 31 | split-K64 | 1938 cyc | 2408 cyc | 95 cyc | 277 cyc | tail |

K64 does move legal PV issue about 180-240 cycles earlier, and one idx0 sample issued
before both halves were ready as intended. It still increases wall time. The total
idx0 producer chain is about 4851 cycles for split-full and 4813 for split-K64 versus
4130 cycles for e16pc. The intended halved exposed producer chain was therefore not
achieved; the common max/scale rendezvous and duplicated WG control outweigh the
parallel 64-column pack work.

## Required Performance Matrix

All values are kernel-only p50 milliseconds, 30 rotated samples. `n/a` means that the
old explicit Stage2/e16pc names are not exposed by the H1 host branch; old auto was
used for the finalist comparison.

| Cell | Stage2 | e16pc | split-full | split-K64 | BF16 best |
|---|---:|---:|---:|---:|---:|
| B1/S2048/H16 | 0.0666 | 0.0672 | 0.0732 | 0.0726 | 0.0784 |
| B1/S2048/H32 | 0.1156 | 0.1139 | 0.1239 | 0.1255 | 0.1050 |
| B1/S2048/H64 | 0.2074 | 0.2028 | 0.2237 | 0.2264 | 0.1848 |
| B1/S4096/H8 | 0.0963 | 0.0945 | 0.1045 | 0.1073 | 0.1278 |
| B1/S4096/H16 | 0.1742 | 0.1721 | 0.1916 | 0.1941 | 0.1436 |
| B1/S4096/H32 | 0.3249 | 0.3184 | 0.3562 | 0.3629 | 0.2696 |
| B1/S4096/H64 | 0.6292 | 0.6182 | 0.6942 | 0.7081 | 0.4470 |
| B1/S8192/H4 | 0.1632 | 0.1591 | 0.1769 | 0.1802 | 0.2241 |
| B1/S8192/H8 | 0.2976 | 0.2916 | 0.3273 | 0.3338 | 0.2292 |
| B1/S8192/H16 | 0.5608 | 0.5538 | 0.6206 | 0.6366 | 0.3938 |
| B1/S8192/H32 | 1.1054 | 1.0920 | 1.2335 | 1.2652 | 0.7048 |
| B1/S8192/H64 | 2.1956 | 2.1816 | 2.4741 | 2.5348 | 1.3078 |
| B1/S16384/H1 | n/a | n/a | 0.3264 | 0.3335 | 0.3904 |
| B1/S16384/H2 | 0.3070 | 0.2960 | 0.3294 | 0.3362 | 0.4046 |
| B1/S16384/H4 | 0.5470 | 0.5295 | 0.5971 | 0.6104 | 0.3925 |
| B1/S16384/H8 | 1.0391 | 1.0100 | 1.1450 | 1.1734 | 0.6764 |
| B1/S16384/H16 | 2.0527 | 1.9947 | 2.2814 | 2.3298 | 1.2383 |
| B1/S32768/H1 | n/a | n/a | 0.6261 | 0.6397 | 0.7136 |
| B1/S32768/H4 | 1.9922 | 1.9168 | 2.1663 | 2.2155 | 1.2514 |
| B1/S32768/H16 | 7.8358 | 7.5262 | 8.5587 | 8.7378 | 4.5436 |
| B2/S4096/H8 | 0.1735 | 0.1697 | 0.1903 | 0.1931 | 0.1422 |
| B4/S4096/H4 | 0.1739 | 0.1721 | 0.1905 | 0.1937 | 0.1434 |

The split route is consistently 10-14% slower than e16pc in the high-head region.
K64 is another 1-3% slower than full-P. No high-head split dispatch is retained.

## NCU And Resources

Final default/off NCU, one full metric collection per route:

| Cell | route | duration | issue | TC active | tensor | eligible | long SB | wait |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H16/S4096 | e16pc | 158.336 us | 35.88% | 14.41% | 6.80% | 0.46 | 3.07 | 1.22 |
| H16/S4096 | split-full | 178.016 us | 47.77% | 13.37% | 6.19% | 0.85 | 2.61 | 1.07 |
| H16/S8192 | e16pc | 541.664 us | 39.71% | 16.99% | 7.99% | 0.48 | 3.13 | 1.17 |
| H16/S8192 | split-full | 607.936 us | 52.39% | 15.29% | 7.12% | 0.88 | 2.60 | 1.02 |
| H32/S4096 | e16pc | 308.896 us | 37.76% | 15.17% | 7.16% | 0.47 | 3.07 | 1.22 |
| H32/S4096 | split-full | 344.352 us | 49.26% | 13.78% | 6.38% | 0.85 | 2.61 | 1.07 |

The split route improves issue activity, eligible warps, and scoreboard ratios, but
executes longer while TC/tensor utilization falls. It is control/instruction-heavy,
not register- or V-load-limited. TMA activity is only 0.24-0.27%, so the task's
one-vs-two-V-warp lever was not justified.

At H16/S4096, short-scoreboard falls from 0.47 to 0.34, while no-instruction rises
from 0.47 to 0.53. The corresponding pairs are 0.43/0.39 to 0.30/0.40 at H16/S8192
and 0.47/0.44 to 0.34/0.51 at H32/S4096. This again points to instruction/control
footprint rather than a memory dependency as the remaining split-route cost.

Resource usage:

| route | registers | stack | local | static shared | barriers |
|---|---:|---:|---:|---:|---:|
| e16pc | 168 | 0 B | 0 B | 2928 B | 2 |
| split-full | 128 | 0 B | 0 B | 12208 B | 2 |
| split-K64 | 128 | 0 B | 0 B | 12240 B | 2 |

NCU reports 100352 B dynamic shared for all selected split launches. The route remains
in the existing one-CTA residency class. There are no selected spills and no extra
hardware barrier beyond the two-barrier gate.

## Materialized-P Fallback Gate

The compact-P consumer-only floor was measured, but it is not an attention result:

| Cell | BF16 best | full compact-P consumer | producer budget | full temp | min P write+read |
|---|---:|---:|---:|---:|---:|
| H16/S4096 | 0.143568 ms | 0.093408 ms | 0.050160 ms | 136 MiB | 272 MiB |
| H16/S8192 | 0.393808 ms | 0.185856 ms | 0.207952 ms | 544 MiB | 1088 MiB |
| H4/S16384 | 0.392528 ms | 0.163392 ms | 0.229136 ms | 544 MiB | 1088 MiB |
| H16/S16384 | 1.238320 ms | 0.476704 ms | 0.761616 ms | 2176 MiB | 4352 MiB |

The 1 GiB-bounded chunk recommendation at H16/S16384 is K4096, 544 MiB peak ring,
and 0.607104 ms consumer-only. It still omits producer, normalization, and reduction.

The producer gate failed:

- optimistic QK/LSE H1/S4096 took 112 ms on first use;
- the same path at H16/S4096 timed out even with H1 chunking;
- compact live P at H16/S4096 timed out;
- compact live P at H16/S8192 and H16/S16384 returned non-finite LSE;
- H4/S16384 happened to be finite, which does not establish a viable architecture.

Because the exact producer is neither finite nor correct on representative target
cells, summing its cost with the consumer floor is impossible and no materialized-P
API was built. A future fallback requires a new finite score-derived e16pc global-P
producer with documented compact layout before further end-to-end timing.

## Dispatch And Regression

The final 57-cell matrix used 30 samples per route and had zero worker errors:

- robust wins: 40;
- parity: 1;
- losses: 16;
- no finite FP4: 0;
- all 37 historical robust-win cells remain robust wins.

The former nine no-finite large-S cells are all finite. B1/S16384/H1 and
B1/S32768/H1 are robust wins under the retained split gate; B1/S16384/H2 is also a
win on its existing fullgrid route. Remaining losses are H32+ at S2048, H16+ at
S4096, H8+ at S8192, H4+ at S16384/S32768, and the B2/B4 S4096 factorized cells.

## Final State

- Final clean build used timeline=0, P-chain stamps=0, target idx=0, KPIPE=0,
  score-reuse=0, selective policy=0, hotplate scheduler/policy=0, and counters=0.
- Final low-shape Stage2/e16pc/auto smoke, retained H1/S16384 auto smoke, and
  B4/S4096/H4 batch-safety smoke are finite.
- Timeline and P-chain readbacks are empty; all 64 policy counters read zero.
- Scoped `git diff --check` passes.
- HEAD moved externally from `d0e185e045c67497862b9b0732d5004561275913` to
  `b8cc39d782de17464576d5f39ecefa2e035d6d7b` during the pass. Those branch/history
  changes were preserved.
- This task did not commit or push.

Primary machine-readable artifact:
`forward_large_shape_high_head_adaptation_20260711_summary.json`.

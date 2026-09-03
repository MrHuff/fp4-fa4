# Direct HAO SM100 FP4 Attention Port to ThunderKittens

Date: 2026-07-23

## Scope

This is a direct ThunderKittens implementation of the HAO AI Lab SM100
forward-attention topology at commit
`9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247`.

It is separate from the older one-query TK MXFP4 kernel. The port supports:

- NVFP4 QK with FP8 PV
- MXFP4 QK with FP8 PV
- NVFP4 or MXFP4 QK with NVFP4 PV
- NVFP4 or MXFP4 QK with MXFP4 PV

The FP8-PV and block-scaled-PV implementations are isolated extensions so
they can be compared without changing the production dispatch path.

## Ported topology

The port reproduces HAO's high-level ownership model:

- one 512-thread CTA owns two M128 query stages
- WGs 0 and 1 independently read scores, run online softmax, and build P
- WG 2 owns output correction
- warp 12 is the sole QK/PV tensor-command issuer
- warp 13 owns the epilogue
- warp 14 owns TMA loads
- FP8 PV uses a four-stage alternating K/V shared-memory ring
- block-scaled PV uses a thirteen-stage alternating K/V ring
- one persistent grid is capped at the SM count

All 512 TMEM columns are assigned exactly as:

```text
0             128           256           384           512
| score S0    | score S1    | output O0   | output O1   |
| P0 at +64   | P1 at +64   |             |             |
```

Q/K and P/V scale pages temporarily reuse retired score storage. There is no
permanent TMEM scale slab.

## Translation fixes that mattered

Four code-generation differences accounted for most of the initial gap:

1. HAO's FP8-PV exp2 cadence uses ALU emulation for six of sixteen packed
   pairs in each of the first three N32 fragments, then native MUFU for the
   last fragment. The matching local mask is `0x3198`.
2. Retaining score quarter 2 avoids one TMEM reload. Retaining more quarters
   spills heavily or slows the generated nvcc kernel.
3. HAO emits `mbarrier.try_wait` with a 10,000,000-cycle suspend hint. The
   TK generic helper emitted a spin/YIELD loop. A direct-only wait helper now
   reproduces HAO's sync-aware `NANOSLEEP.SYNCS` behavior.
4. Register requests must be placed inside the disjoint warp-role branches.
   The old grouped request made ptxas constrain role code against the wrong
   budget and made HAO's 192/80/48 split appear to spill. Role-local requests
   reproduce that exact split with zero stack and zero spills.

The FP8-PV build has a 64-byte static spill allocation. Matched profiling
showed dynamic spill instructions are about 0.17% of executed instructions,
so removing it is not the current performance priority.

## FP8-PV results

The table uses HAO's D128 benchmark inputs and timer on one GB200. Times are
milliseconds. HAO is the checked-out native NVFP4-QK + FP8-PV kernel. BF16 is
the native HAO BF16 route measured in the same benchmark process.

| Shape | TK NV QK | TK MX QK | HAO NV QK | BF16 |
|---|---:|---:|---:|---:|
| B4 S4096 H16 | 0.362528 | 0.360560 | 0.356544 | 0.392224 |
| B1 S32768 H16 | 5.579712 | 5.473280 | 5.532672 | 5.999648 |
| B4 S4096 H32 | 0.718656 | 0.712672 | 0.704608 | 0.794688 |
| B1 S4096 H12 | 0.110592 | 0.109120 | 0.108544 | 0.108736 |
| B1 S32768 H12 | 4.311072 | 4.246640 | 4.302848 | 4.474464 |
| B1 S4096 H24 | 0.161344 | 0.157696 | 0.159648 | 0.165888 |
| B1 S32768 H24 | 8.316928 | 8.196608 | 8.292352 | 9.006672 |

Focused longer timing at B1/S4096/H24 measured the NV route at 0.159776 ms
versus 0.164864 ms for the prior TK binary and 0.157984 ms for HAO. At
B1/S32768/H24 it measured 8.324096 ms versus 8.321024 ms for HAO.

The full 18-case report is:

`results/hao_direct_suite_hao_wait_20260723_both.json`

All cases passed. Output cosine versus BF16 was approximately 0.9896-0.9904
for NVFP4 QK and 0.9808-0.9823 for MXFP4 QK.

## Block-scaled-PV results

At B1/S4096/H24:

| QK | PV | TK time | Output cosine vs BF16 |
|---|---|---:|---:|
| NVFP4 | NVFP4 | 0.194848 | 0.981251 |
| MXFP4 | NVFP4 | 0.220960 | 0.973015 |
| NVFP4 | MXFP4 | 0.325632 | 0.970402 |
| MXFP4 | MXFP4 | 0.327680 | 0.961330 |

The latest matched 200/1000 ms run measured native HAO
NVFP4-QK/NVFP4-PV at 0.192960 ms and native HAO BF16 at 0.163840 ms. The TK
translation is now only 0.98% behind HAO full FP4. Full FP4 PV therefore still
loses to BF16 in both implementations, but the remaining 18.9% BF16 gap is no
longer explained by a missing HAO topology in TK.

HAO's exact early-K64 P publication was implemented behind
`TK_HAO_DIRECT_FP4PV_EARLY_P=1`. It is numerically correct but regressed TK
from about 0.2210 ms to 0.2273 ms because the delayed second scale copy lands
on the issue warp's critical path. The implementation is retained for
profiling; the faster all-ready path remains the default.

All four block-scaled combinations compile without register spills. The final
NVFP4/NVFP4 build reports 128 registers per thread, zero stack, and zero
spills.

## Route decision

The performance route is MXFP4 QK + FP8 E4M3 PV:

- it preserves HAO's two-query pipeline and TMEM ownership
- it is at or near HAO NVFP4 performance on saturated shapes
- it beats the same-process BF16 route by about 4-10% on the useful
  compute-heavy shapes in this matrix
- it avoids online block-scale construction and P/V scale copies

NVFP4 QK + FP8 PV remains the closer numerical match. NVFP4/NVFP4 is a valid
full-FP4 reference but is not a competitive production default. MXFP4 PV is
not justified by the current speed or accuracy results.

## Files

FP8 PV:

- `tk_fa4/fp4_fa4_fwd/hao_direct_candidate.cu`
- `tk_fa4/fp4_fa4_fwd/hao_direct_config.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_kernel.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_softmax_reader.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_host.inc`
- `tk_fa4/fp4_fa4_fwd/Makefile.hao_direct`

Block-scaled PV:

- `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_candidate.cu`
- `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_config.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_kernel.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_softmax_reader.inc`
- `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_host.inc`
- `tk_fa4/fp4_fa4_fwd/Makefile.hao_direct_fp4pv`

Benchmarking:

- `tk_fa4/fp4_fa4_fwd/hao_direct_benchmark.py`
- `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_benchmark.py`
- `tk_fa4/fp4_fa4_fwd/hao_direct_suite.py`

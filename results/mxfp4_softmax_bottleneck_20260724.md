# MXFP4 softmax bottleneck and promoted scan schedule

Date: 2026-07-24

## Scope

Measurements use GB200 GPU 1 at B1/S4096/H24/D128, noncausal attention,
152 persistent CTAs, and 512 threads per CTA. Long-window timings use the HAO
tensor factory and `triton.testing.do_bench`.

The full-FP4 kernels in this report use microscaled TCGEN instructions:

```text
tcgen05.mma...kind::mxf4nvf4.block_scale.scale_vec::2X/4X
```

They pass packed FP4 payloads plus E4M3/E8M0 scale operands. These are the
four-times-BF16-class block-scaled instructions, not the roughly
two-times-BF16 raw-FP4 path.

QK uses NVFP4 with E4M3 block-16 scales. The selected PV route uses MXFP4
with E8M0 block-32 scales. The local scaled-PV instruction path has K64, not
K32, granularity, so full FP4 can publish one P half at a time. FP8 PV remains
the K32 quarter-ready control.

## Ceiling decomposition

Each diagnostic preserves real QK, scaled PV, TMEM ownership, correction, and
epilogue unless its name explicitly removes arithmetic:

| Path | MXFP4 PV (ms) | NVFP4 PV (ms) |
|---|---:|---:|
| Fixed P | 0.071680 | 0.073728 |
| One real score read + raw pack | 0.083968 | 0.088064 |
| Real row max/reloads + raw pack | 0.133120 | 0.139040 |
| Real exp/scale/pack + fake denominator | 0.173888 | 0.186816 |
| Real softmax before promoted scan | 0.178496 | 0.194560 |

Approximate incremental costs are:

| Increment | MXFP4 (us) | NVFP4 (us) |
|---|---:|---:|
| QK/PV/ownership/epilogue floor | 71.680 | 73.728 |
| First score read and P packing | 12.288 | 14.336 |
| Row max, three reloads, and correction lifecycle | 49.152 | 50.976 |
| Exp2, scale selection, and quantization | 40.768 | 47.776 |
| Denominator reduction | 4.608 | 7.744 |

These deltas are scheduling diagnostics, not perfectly separable cycle
accounts. They are nevertheless decisive: denominator work is not the main
target. Score scanning/reloads and exp-plus-pack dominate.

For context, the FP8-PV control is 0.079872 ms fixed-P, 0.092160 ms
score-pack, and 0.159744 ms real. Its K32 PV granularity and lack of P
block-scale construction explain why it remains faster despite a worse tensor
floor than MXFP4.

## K64 early publication

A legal MXFP4 half-ready route was implemented with two disjoint E8M0 pages in
retired score columns. Both direct softmax-warp scale publication and
issuer-side scale copies compile with 128 registers and no spills.

The route did not improve time:

| Variant | Real MXFP4 (ms) |
|---|---:|
| Control | 0.182272 |
| Direct half-ready scale pages | 0.182272-0.182560 |
| Issuer-copy half-ready pages | 0.182272-0.182592 |

NCU showed no gain in eligible warps or tensor activity. Publication became
earlier, but the issuer still reached the handoff too late for useful overlap,
and the copy variants added instructions.

## Promoted changes

Three changes improved the real MXFP4 route without materially changing its
accuracy. The scan changes are arithmetic-equivalent; the mixed exp2 mask is
an approximation whose measured BF16 error is unchanged at the shown
precision:

1. `0x8888` mixed exp2 cadence: selected pair positions 3, 7, 11, and 15 in
   middle quarters use packed ALU emulation; the rest use native MUFU.
2. Paired max-scan loads: issue Q0/Q1 before one TCGEN load wait, then Q3/Q2
   before the second wait.
3. Balanced quarter max: reduce 32 values with a five-level tree instead of a
   32-operation accumulator dependency.

The exp2 cadence is MXFP4-only. NVFP4 regresses with mixed ALU/MUFU and stays
on native MUFU.

| MXFP4 variant | Time (ms) | Output cosine vs BF16 |
|---|---:|---:|
| Native MUFU, serial scan | 0.182272 | 0.970392 |
| `0xC0C0`, serial scan | 0.1784-0.1789 | 0.970392 |
| `0xC0C0`, paired loads | 0.1767-0.1777 | 0.970392 |
| `0xC0C0`, paired loads + max tree | 0.1764-0.1772 | 0.970392 |
| `0x8888`, paired loads + max tree | **0.1757-0.1761** | 0.970392 |

The clean promoted default measured 0.176128 ms. It uses 128 registers, one
barrier, 400 bytes static shared memory, no stack, and no spills.

## Profile change

Matched NCU kernel-replay profiles compare the `0xC0C0` serial scan with the
promoted `0x8888` paired/tree scan:

| Metric | Serial scan | Promoted |
|---|---:|---:|
| Replay duration (us) | 302.94 | 297.38 |
| Dynamic scheduler instructions | 78.12M | 79.35M |
| Issue active | 45.69% | 47.31% |
| Eligible warps/cycle | 0.56 | 0.60 |
| Long-scoreboard ratio | 4.18 | 3.91 |
| Wait ratio | 1.38 | 1.37 |
| Tensor-pipe active | 12.11% | 12.34% |
| Memory-tensor active | 23.48% | 23.94% |

The max tree exposes more independent instructions, so total instruction count
rises slightly while elapsed time and scoreboard pressure fall.

## Rejected alternatives

| Alternative | Time (ms) | Read |
|---|---:|---|
| Retain Q0 across scan and packing | 0.198912 | Extra live fragment constrains scheduling |
| Shiftless one-pass, serial max | 0.180224 | Quarter max/scale/exp/pack becomes serial |
| Shiftless one-pass + max tree | 0.178176 | Better, but slower and less accurate |
| `0xA0A0` mixed exp2 | 0.178240 | ALU/MUFU placement is poorly balanced |
| Fake denominator | 0.173888 | Only about 4.6 us of recoverable time |

## Current comparison

Matched long-window timings:

| Kernel | Time (ms) |
|---|---:|
| HAO BF16 | **0.165888** |
| TK NVFP4 QK + MXFP4 PV, promoted | **0.176128** |
| HAO native NVFP4 QK + NVFP4 PV | 0.193440 |
| TK NVFP4 QK + NVFP4 PV | about 0.195 |

The promoted full-MXFP4 route is 9.0% faster than HAO full NVFP4 but remains
6.2% slower than BF16 at this shape.

## Next target

Further denominator work has little ceiling. The useful remaining targets are:

1. overlap or batch second-pass score reloads without keeping a fragment live
   through another quarter's exp/pack;
2. fuse selected ALU exp2 results directly into E2M1 codes without the large
   F2I/clamp network used by the rejected direct-log classifier;
3. reduce the row-max/reload lifecycle structurally while preserving the
   two-pass instruction overlap that made shiftless one-pass slower;
4. validate any promoted arithmetic policy across sequence lengths and head
   counts before treating this single-shape gain as general.

## Fused log-code and ILP follow-up

An explicit two-group software pipeline was added behind
`HAO_FP4PV_P_ILP_PIPELINE`. It keeps the arithmetic unchanged and interleaves
probability exponentiation with packing and denominator consumption.

The MXFP4 control and explicit-ILP kernels have byte-identical SASS. The same
is true for the corresponding FP8-PV control and explicit-ILP kernels. Ptxas
already found this local instruction ordering, so source-level reordering has
no remaining gain:

| MXFP4 variant | Time (ms) | Cosine vs BF16 |
|---|---:|---:|
| Promoted control | 0.176064 | 0.970392 |
| Explicit exact ILP | 0.176128 | 0.970392 |

Two direct log-domain classifiers then removed per-element `exp2` and formed
the denominator from the emitted E2M1 codes:

| Direct-log variant | Time (ms) | Cosine vs BF16 | RMSE |
|---|---:|---:|---:|
| Word-pipelined denominator | 0.481568 | 0.970312 | 0.006206 |
| Pair-owned denominator weights | 0.464448 | 0.970312 | 0.006206 |

Pair-owned denominator formation is a real improvement over the late
word-level decode, saving about 17.1 us in this otherwise identical path. The
classifier itself is prohibitive. Static SASS counts explain why:

| Kernel | Instructions | F2I | Float set | Integer min/max |
|---|---:|---:|---:|---:|
| Promoted control | 8,128 | 9 | 14 | 17 |
| Word-pipelined direct log | 12,656 | 265 | 526 | 1,041 |
| Pair-owned direct log | 12,384 | 265 | 526 | 1,041 |

NCU reports 184.3M executed scheduler instructions for the pair-owned path
versus 79.3M for the promoted control. Issue activity falls from 47.31% to
39.08%, eligible warps fall from 0.60 to 0.45 per cycle, and the pair-owned
path has no eligible warp for 60.92% of cycles. Removing MUFU work therefore
trades one latency source for a much larger instruction and dependency graph.

Exponent-bit synthesis was also tested as a lower-instruction approximation:

| Bit-exp placement | Time (ms) | Cosine vs BF16 |
|---|---:|---:|
| Four former ALU-exp positions | 0.179264 | 0.970197 |
| Every pair in middle quarters | 0.184352 | 0.969756 |

Neither beats the mixed `0x8888` ALU/MUFU control. The useful constraint for
future fused-softmax work is now concrete: a replacement must retain the
hardware E2M1 conversion and avoid per-element F2I, threshold trees, and
post-pack integer networks. Structural removal of the score scan/reload pass
has more headroom than another local classifier.

All follow-up variants remain default-off. Rebuilding the promoted defaults
after adding them produces byte-identical kernel SASS, 128 registers, one
barrier, 400 bytes of static shared memory, and no stack or spills.

## Sampled-stabilizer one-pass experiment

`HAO_FP4PV_MX_SAMPLED_STABILIZER=1` uses score quarter 2 as the
running online-softmax shift. It then consumes Q2, Q0, Q3, and Q1 exactly
once, deriving each quarter's exact MXFP4 block maximum immediately before
packing. This cuts the score-quarter traffic from seven loads to four while
retaining the normal correction and output-rescale lifecycle.

Matched B1/S4096/H24/D128 results:

| Variant | Time (ms) | Cosine vs BF16 | RMSE |
|---|---:|---:|---:|
| Promoted full-row max | **0.176128** | **0.970392** | **0.006260** |
| Shiftless one-pass + max tree | 0.178176 | 0.969368 | 0.006403 |
| Q2 stabilizer, serial loads | 0.188416 | 0.969844 | 0.006336 |
| Q2 stabilizer, paired Q3/Q1 tail | **0.180992** | 0.969844 | 0.006336 |
| Q2 stabilizer, paired head and tail | 0.188448 | 0.969844 | 0.006336 |

Pairing only Q3/Q1 recovers 7.424 us because their TMEM loads can share one
wait after the Q0 scale lease has been released. Pairing Q2/Q0 regresses:
Q0's 32-value fragment remains live through Q2 exponentiation and packing,
which constrains instruction scheduling despite preserving the same
128-register, one-barrier, spill-free ptxas envelope.

The best sampled kernel has six fewer static `LDTM` instructions and four
fewer static max instructions than promoted. NCU also reports fewer dynamic
scheduler instructions, 78.79M versus 79.35M. The reduction does not produce
a wall-time win:

| NCU metric | Promoted | Sampled tail pair |
|---|---:|---:|
| Replay duration | 297.38 us | 305.92 us |
| Issue active | 47.31% | 45.61% |
| Eligible warps/cycle | 0.60 | 0.57 |
| Long-scoreboard ratio | 3.91 | 4.11 |
| Wait ratio | 1.37 | 1.49 |
| Memory-tensor active | 23.94% | 21.77% |
| Tensor-pipe active | 12.34% | 11.93% |

Q2 now anchors a strict chain: load Q2, reduce its maximum, update the row
shift and correction, then begin all P exponentiation and scale generation.
The production two-pass path performs more loads but exposes the four-quarter
max scan as independent work before P construction. On GB200, that extra
instruction-level parallelism is worth more than the removed TMEM loads.

The sampled mode remains useful as a structural diagnostic, but is not a
promotion candidate. It is default-off, and the default kernel remains
byte-identical to the prior promoted SASS.

## Shiftless tail pairing and half-Q0 prefetch

Follow-up work on 2026-07-25 found a better one-pass schedule. The earlier
shiftless result serialized every score-quarter load. The revised schedule
issues the Q3 and Q1 TMEM loads together and waits once, so their independent
loads overlap. It also uses a mixed `0xC0C0` ALU/MUFU exp2 cadence.

The final exact candidate additionally splits the Q0 score load into two
`tcgen05.ld.sync.aligned.32x32b.x16.b32` instructions:

1. load and wait for Q2;
2. issue the first half of Q0 without waiting;
3. reduce and pack Q2;
4. issue the second half of Q0 and wait once;
5. reduce and pack Q0;
6. issue full Q3 and Q1 loads together and wait once.

This changes only load timing. Row maxima, exponentials, denominator sums,
E8M0 scales, E2M1 payloads, correction, and PV are unchanged.

Matched B1/S4096/H24/D128 results:

| Variant | Time (ms) | Cosine vs BF16 | RMSE |
|---|---:|---:|---:|
| Promoted two-pass control | 0.176128 | 0.970392 | 0.006260 |
| Shiftless, paired Q3/Q1, `0xC0C0` | 0.169216 | 0.969368 | 0.006403 |
| Above + half-Q0 prefetch | **0.167936-0.168224** | 0.969368 | 0.006403 |
| Above + half-Q1 load | 0.176832 | 0.969368 | 0.006403 |
| Above + half-Q3 prefetch | 0.169184-0.170048 | 0.969368 | 0.006403 |

Half Q1 adds a wait on the publication tail. Half Q3 keeps 16 score registers
live through Q0 processing and restricts ptxas scheduling more than its early
load helps. Half Q0 is the only exact split that hides useful latency without
lengthening the critical tail or increasing the resource envelope. It
compiles with 128 registers, one barrier, 400 bytes static shared memory, no
stack, and no spills.

The candidate is reproducible with:

```text
HAO_QK_SCALE_MODE=0
HAO_PV_SCALE_MODE=1
HAO_FP4PV_MX_SHIFTLESS_SOFTMAX=1
HAO_FP4PV_MX_PAIR_LOAD_SCAN=1
HAO_FP4PV_MX_HALF_PREFETCH_Q0=1
HAO_FP4PV_EX2_EMU_MASK=0xC0C0
```

All new modes remain default-off. A rebuild with promoted defaults measures
0.176128 ms with cosine 0.970392 and RMSE 0.006260.

## BF16 comparison after the shiftless win

The final matched 500-ms HAO benchmark reports:

| Kernel | Time (ms) | Relative to HAO BF16 |
|---|---:|---:|
| HAO BF16 | **0.165440** | 1.000x |
| TK NVFP4 QK + MXFP4 PV | **0.167936** | 0.985x |
| HAO NVFP4 QK + NVFP4 PV | 0.192512 | 0.859x |

The TK candidate is 12.8% faster than HAO full NVFP4 and 1.5% slower than
HAO BF16 at this shape. Seeds 1, 2, and 3 measure 0.167936-0.168288 ms with
cosines 0.96960-0.97001, so the result is not seed-specific.

A matched NCU replay compares the exact paired shiftless path before the
half-Q0 load split against HAO BF16:

| Metric | HAO BF16 | TK full FP4 |
|---|---:|---:|
| Replay duration (us) | 164.064 | 167.296 |
| Dynamic scheduler instructions | 55.742M | 68.232M |
| Issue active | 33.53% | 34.52% |
| Eligible warps/cycle | 0.501 | 0.527 |
| Long-scoreboard ratio | 3.024 | 4.970 |
| Wait ratio | 1.671 | 1.474 |
| Barrier ratio | 2.110 | 0.024 |
| No-instruction ratio | 0.129 | 0.266 |
| Tensor-pipe active | 60.55% | 12.69% |
| Memory-tensor active | 63.88% | 19.25% |

The remaining gap is not a full-CTA barrier problem. FP4 finishes its tensor
work quickly, then executes about 12.5M more dynamic scheduler instructions
to turn scores into scaled E2M1 P. The higher long-scoreboard and
no-instruction ratios show that this score-to-P dependency chain, rather than
PV tensor throughput, supplies the final few microseconds.

## Delayed scaling and approximation diagnostics

Native local delayed scaling was implemented with one integer log2 anchor per
row. Each block emits an absolute E8M0 scale and is rebased to the row anchor
later. It restores the shiftless output exactly, but retains four local
denominator/scale states and measures 0.251424 ms. The arithmetic is valid;
the enlarged live-state and serial correction lifecycle make this form
unsuitable for the current warp ownership.

The exact fake-denominator ceiling for the paired shiftless path is
0.165856 ms. Therefore denominator removal is worth at most roughly 3.6 us
from the pre-half-Q0 candidate, and less from the final candidate.

Sampling confirms where accuracy becomes fragile:

| Approximation | Time (ms) | Cosine vs BF16 | Read |
|---|---:|---:|---|
| Exact denominator | 0.1692 | 0.969368 | Control before half-Q0 split |
| Half denominator samples | 0.1679-0.1680 | 0.96915 | About 1 us, small quality loss |
| Quarter denominator samples | 0.169984 | 0.96874 | Slower and less accurate |
| Max from 28/32 values | 0.168640 | 0.95995 | Rare missed peaks dominate |
| Max from 24/32 values | 0.165536 | 0.94900 | Unacceptable quality loss |
| Max from 16/32 values, biased | 0.163840 | 0.93486-0.94381 | Unacceptable quality loss |

Quantizing the denominator from packed FP4 regresses to 0.201024 ms, and a
balanced quarter-contribution reduction regresses to 0.169984 ms. The current
serial accumulator gives ptxas the best schedule.

The actionable conclusion is narrow: do not spend the next iteration on
another max or denominator approximation. The remaining exact opportunity is
to shorten the score-to-E2M1 dependency graph while retaining hardware E2M1
conversion, or to change warp ownership so local delayed-scale state can be
processed independently instead of retained by one softmax warp.

## Early K64 publication on the shiftless layout

The shiftless half-Q0 winner was extended with a true first-K64 handoff. The
softmax warp now processes Q2, Q0, and Q1, publishes the complete Q0/Q1 P
payload and its scale page, then constructs Q3 while the issuer can begin the
first PV K64. The tail handoff publishes Q2/Q3. The two scale halves use
disjoint retired-score pages at `score + 32` and `score + 48`, so this adds no
TMEM columns.

The conservative implementation was correct but tied or lost: 0.168544 ms
versus the 0.167936-ms control. Its extra publication work added roughly 1.8M
dynamic scheduler instructions. Two ordering changes made the overlap useful:

- The direct TMEM scale route no longer executes a shared-memory publication
  fence that has no consumer.
- The per-half scale-store wait is deferred to the existing
  TMEM-before-thread-sync fence and readiness barrier.

The retained mode is:

```text
HAO_QK_SCALE_MODE=0
HAO_PV_SCALE_MODE=1
HAO_FP4PV_MX_SHIFTLESS_SOFTMAX=1
HAO_FP4PV_MX_PAIR_LOAD_SCAN=1
HAO_FP4PV_MX_HALF_PREFETCH_Q0=1
HAO_FP4PV_EX2_EMU_MASK=0xC0C0
HAO_FP4PV_MX_EARLY_P=1
HAO_FP4PV_MX_EARLY_ASYNC_SCALE=1
```

A 2-second TK-only sample measures **0.165888 ms**, compared with
0.167936 ms for the previous shiftless winner. A matched 500-ms run measures
0.166208 ms for TK, 0.163872 ms for HAO BF16, and 0.192928 ms for HAO full
NVFP4. Output accuracy is unchanged: cosine 0.969368 and RMSE 0.006403 versus
the BF16 reference. Seeds 1-3 pass with cosines 0.96960-0.97001. The kernel
uses 128 registers, one barrier, no spills, and no stack.

| NCU metric | Shiftless control | Early K64, conservative | Early K64, async scale |
|---|---:|---:|---:|
| Replay duration (us) | 287.392 | 288.160 | **285.792** |
| Dynamic scheduler instructions | 68.893M | 70.689M | 70.159M |
| Issue active | 35.20% | 36.04% | 36.06% |
| Eligible warps/cycle | 0.53 | 0.58 | 0.53 |
| Long-scoreboard ratio | 4.84 | 4.60 | 4.81 |
| Wait ratio | 1.48 | 1.45 | **1.37** |
| Tensor-pipe active | 12.82% | 12.79% | 12.90% |
| Memory-tensor active | 19.54% | 19.25% | 19.64% |

An attempted register-direct scale handoff removed the shared scale-byte
stores and reloads, but regressed back to 0.167936 ms in back-to-back
2-second samples. The compiler already hides that shared-memory traffic;
keeping four quarter-scale values live lengthens the dependency graph. That
sub-experiment was removed.

The result validates early K64 publication under this TMEM layout, but the
remaining 1-2% BF16 gap is still score-to-P work. The next useful change must
move independent Q3 construction into the first-PV window without adding
live scale state or another publication barrier.

## Five-way post-early-K64 sweep

Five independent ways to use the remaining Q3/PV window were implemented,
debugged to a correct run, and measured at B1, S4096, H24, D128. All controls
use the early-K64 shiftless configuration above.

| Experiment | Time (ms) | Result |
|---|---:|---|
| Q3 native exp2, control | 0.165888 | Retained arithmetic baseline |
| Q3 ALU mask `0xC0C0` | 0.168320 | Reject |
| Q3 ALU mask `0x8080` | 0.166912 | Reject |
| Q3 ALU mask `0x8000` | 0.168640 | Reject |
| Exact shiftless correction bypass | 0.163392-0.164096 | Retain |
| Linear row-major direct-scale scratch | 0.163840 | Tie; default off |
| Cross-stage split PV, both tails first | 0.225408 | Reject |
| Cross-stage split PV, QK0 before tail1 | 0.215136 | Reject |
| Correction WG owns Q3, per-tile sum handoff | 0.176160 | Reject |
| Correction WG owns Q3, one final sum handoff | 0.174400 | Reject |

The winner is exact, not an approximation. In shiftless mode every online
softmax correction is identically one. The retained bypass therefore removes
the per-score-tile correction store, fence, and correction-WG rendezvous. The
first-P barrier expects four softmax arrivals instead of eight combined
softmax/correction arrivals. A separate job-level output-empty semaphore
protects accumulator reuse; it is overlaid on otherwise-unused correction
scratch and consumes no additional static shared memory.

The final rebuilt candidate uses 128 registers, one barrier, 400 bytes of
static shared memory, no stack, and no spills. Its 2-second TK-only median is
**0.162080 ms**. Seeds 1-3 measure 0.161824-0.162400 ms with output cosines
0.96960-0.97001. A final matched 1-second run is:

| Provider | Time (ms) | Relative to HAO BF16 |
|---|---:|---:|
| TK NVFP4 QK + MXFP4 PV | **0.162432** | **1.0213x** |
| HAO BF16 | 0.165888 | 1.0000x |
| HAO full NVFP4 | 0.193088 | 0.8591x |

The candidate's seed-0 output cosine versus BF16 is 0.969368 and RMSE is
0.006403, unchanged from the early-K64 control.

The correction bypass explains the gain directly:

| NCU metric | Early K64 | Correction bypass |
|---|---:|---:|
| Replay duration (us) | 285.792 | 280.220 |
| Dynamic scheduler instructions | 70.159M | 66.775M |
| Issue active | 36.06% | 34.98% |
| Eligible warps/cycle | 0.53 | 0.50 |
| Long-scoreboard ratio | 4.81 | 5.25 |
| Wait ratio | 1.37 | 1.32 |
| Tensor-pipe active | 12.90% | 13.15% |
| Memory-tensor active | 19.64% | 20.14% |

About 3.38M dynamic instructions disappear while tensor activity rises
slightly. This is a control-lifecycle win, not better exp2 throughput.

The other four avenues fail for distinct reasons:

- Q3 ALU/SFU mask changes lengthen the dependency chain; native Q3 exp2 is
  already the best schedule.
- Linear scale scratch removes measured bank conflicts, but those conflicts
  are hidden and do not improve wall time.
- Cross-stage split PV postpones next-score QK issuance, losing much more
  overlap than first-K64 PV gains.
- Moving Q3 to the correction warpgroup adds a handoff and about 1.40M
  dynamic instructions. Even with one denominator handoff per job, issue and
  tensor activity fall.

All five controls are compile-time flags and remain default-off. The exact
winner is enabled with:

```text
HAO_FP4PV_MX_SHIFTLESS_CORR_BYPASS=1
```

on top of the retained early-K64 configuration.

## Post-winner structural probes

The committed correction-bypass winner was rebaselined before attempting
deeper buffering. Matched local controls measured 0.16179 ms for real
softmax and 0.08806 ms for fixed P. Score-pack and rowmax-pack controls
measured 0.09245 ms and 0.09216 ms respectively. Replacing the exact
denominator with a constant measured 0.16339 ms, confirming that the
denominator reduction is no longer material.

Publishing exact P before finishing the denominator was also implemented.
It retained the score fragment across the handoff and measured 0.16339 ms
against a 0.16179 ms control. Four scalar denominator trees are too small to
repay the longer live range and schedule constraint, so this variant was
removed.

A one-output, three-score-slot fixed-P prototype then tested the structural
value of deeper score buffering before porting real softmax. It used three
complete score-owner warpgroups and a QK/PV round-robin over 512 TMEM
columns. The first build exposed an overcommitted register-redistribution
request; reducing each fixed-P owner from 192 to 80 registers fixed the
lifecycle. Matched long-window results were:

| Fixed-P structure | Logical jobs | Time (ms) |
|---|---:|---:|
| Current two-query/two-output kernel | 384 | **0.08806** |
| One-query/three-score prototype | 768 | 0.16797 |

The extra score slot cannot compensate for doubling logical jobs and losing
shared K/V work across two query tiles. At 1.91x the existing fixed-P floor,
the structure is rejected and its source hook was removed. A viable deeper
pipeline must preserve the current two-query job granularity.

The remaining instruction-level alternative forced each eight-value P word
through exp2, E2M1 packing, denominator accumulation, and an ordered
single-word TMEM store before starting the next word. It preserved the exact
0.969368 output cosine, but matched samples measured 0.16416-0.16454 ms
against 0.16179-0.16208 ms for the four-word store. The extra ordered stores
serialize the path; the experiment was removed.

Finally, the softmax role's dynamic register grant was swept while preserving
the winning instructions:

| Softmax register grant | Time (ms) |
|---:|---:|
| 128 | 0.17069 |
| 144 | 0.16384 |
| 160 | 0.16458 |
| 176 | 0.16384 |
| 192 | **0.16195** |

Although ptxas reports 128 registers and no spills for every build, the
192-register role-local grant remains important at runtime. It is already
the largest aligned grant allowed by the CTA-wide redistribution budget, so
the committed value remains unchanged.

## Cross-shape validation and persistent-grid dispatch

The benchmark harness now derives B/S/H/D from the extension topology instead
of assuming B1/S4096/H24/D128. Native Q/K and MXFP4 V scale conversion is also
shape-derived, which permits the same HAO factory inputs and BF16 baseline to
be used for every compiled TK shape.

Matched HAO-factory results on GB200 are:

| S | H | Logical jobs | TK MXFP4 PV (ms) | HAO BF16 (ms) | HAO NVFP4 PV (ms) | TK / BF16 | Cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4 | 16 | **0.021440** | 0.052384 | 0.070752 | **2.443x** | 0.971083 |
| 1024 | 24 | 96 | **0.023488** | 0.053472 | 0.073920 | **2.277x** | 0.970013 |
| 1024 | 64 | 256 | **0.037152** | 0.056992 | 0.076416 | **1.534x** | 0.969836 |
| 2048 | 24 | 192 | **0.061440** | 0.062048 | 0.074336 | **1.010x** | 0.970019 |
| 4096 | 24 | 384 | **0.162080** | 0.167936 | 0.192512 | **1.036x** | 0.969368 |
| 4096 | 64 | 1024 | **0.371408** | 0.394976 | 0.437072 | **1.063x** | 0.969560 |
| 8192 | 24 | 768 | 0.618528 | **0.593936** | 0.724944 | 0.960x | 0.970057 |
| 8192 | 64 | 2048 | **1.443840** | 1.503248 | 1.684192 | **1.041x** | 0.969601 |

The result is not a generic large-head failure. More heads amortize the
persistent CTA setup and improve the MXFP4 kernel relative to BF16. The one
measured loss is long sequence with only 24 heads, where each CTA performs
too few long jobs to amortize the remaining per-CTA and per-job pipeline cost.

The physical grid was swept on the two shapes most sensitive to partial
waves:

| Shape | Physical CTAs | Time (ms) | Read |
|---|---:|---:|---|
| S8192/H24 | 152 | **0.618528** | One CTA per SM |
| S8192/H24 | 128 | 0.619520 | Exact six jobs per CTA; no gain |
| S2048/H24 | 152 | **0.060992** | One CTA per SM |
| S2048/H24 | 96 | 0.061888 | Exact two jobs per CTA; fewer active SMs lose |
| S2048/H24 | 192 | 0.063488 | Hardware second wave loses persistence reuse |

Both alternatives to 152 CTAs fail. Reducing the grid balances job counts but
leaves SMs unused; increasing it moves the same tail into a second hardware
wave and repeats CTA setup. The default one-CTA-per-SM persistent launch is
therefore retained. The temporary grid-cap diagnostic was removed after the
sweep so it cannot perturb production code generation.

For a performance dispatch today, use this MXFP4 kernel for the measured
winning shapes and fall back to BF16 for S8192/H24. S2048/H24 is effectively
at parity and should use a conservative BF16 fallback unless a deployment
benchmark confirms the approximately 1% TK margin. Closing the S8192/H24 gap
requires reducing the cost of each long logical job; grid reshaping cannot do
it.

### Release code-generation policy

The preserved S4096 winner could not initially be reproduced by the Makefile's
two-way split compilation. Its topology and register count matched, but its
SASS was shorter and hoisted more uniform address state out of repeated
pipeline sections. Rebuilding with one device-code partition recovered that
schedule:

| Shape | Split partitions | Time (ms) | Change |
|---|---:|---:|---:|
| S4096/H24 | 2 | 0.16339-0.16384 | Control |
| S4096/H24 | 1 | **0.16256** | 0.5-0.8% faster |
| S4096/H64 | 4 | 0.37141 | Control |
| S4096/H64 | 1 | **0.36688** | 1.2% faster |
| S8192/H24 | 4 | 0.61853 | Control |
| S8192/H24 | 1 | **0.61235** | 1.0% faster |
| S8192/H64 | 4 | 1.44384 | Control |
| S8192/H64 | 1 | **1.42314** | 1.4% faster |
| S2048/H24 | 2 | **0.06099** | Control |
| S2048/H24 | 1 | 0.06170 | 1.2% slower |

This is shape-dependent compiler optimization, not measurement noise or a
change in arithmetic. The Makefile now defaults to
`NVCC_SPLIT_COMPILE=1`, matching its default S4096 shape. S1024/S2048 release
artifacts must override it with `NVCC_SPLIT_COMPILE=2`. The long-shape
dispatch should use separately compiled split-1 artifacts.

A final matched HAO-factory run at S8192/H24 measured 0.609856 ms for the
split-1 TK artifact, 0.600064 ms for HAO BF16, and 0.722944 ms for HAO full
NVFP4. The optimized TK path is therefore 1.6% behind BF16 rather than the
initial 4.1% gap. It remains a BF16-fallback shape, but is now close enough
that the next iteration should target long-job inner-loop throughput rather
than launch geometry.

## Delayed-half Q2 overlap

The first-PV critical path was shortened without allocating another score
tile. In the previous early-K64 schedule, the reader had to preserve and
finish score quarter Q2 before packing Q0 and Q1. Only then could it publish
the first P K64, so the first PV instruction could not overlap Q2 work.

The new default-off `HAO_FP4PV_MX_DELAYED_HALF_Q2=1` schedule does this:

1. Load Q2-low and Q0 together, then pack Q0.
2. Load Q2-high and Q1 together, then pack Q1.
3. Publish the Q0/Q1 P payload and its first scale page.
4. Start the independent Q3 load.
5. Pack retained Q2 while the first PV K64 and Q3 load are in flight.
6. Pack Q3, publish the tail scale page, and issue the second PV K64.

This uses the existing registers and TMEM overlay. The final kernel still
uses 128 registers, one barrier, 400 bytes of static shared memory, 208896
bytes of dynamic shared memory, and 512 TMEM columns, with no stack or spills.
SASS contains the intended Q2-low/Q0 and Q2-high/Q1 paired loads followed by
the delayed full Q3 load.

A temporary `%globaltimer` trace sampled the same stage in matched builds.
Times below are relative to score-ready:

| Event | Previous schedule (ns) | Delayed-half Q2 (ns) |
|---|---:|---:|
| Q0 packed | 608 | 416 |
| Q1 packed | 896 | 704 |
| First P published | 928 | 768 |
| First PV issued | 1184 | **960** |
| Q2 packed | 384 | 1088 |
| Q3 packed | 1280 | 1504 |

The key causal result is that delayed-half Q2 issues first PV at 960 ns while
Q2 remains active until 1088 ns. The previous schedule completed Q2 at 384 ns
and could not issue first PV until 1184 ns. The instrumentation was removed
after this proof.

Six alternating, same-process pairs on one shared HAO input measured:

| S4096/H24 real-softmax build | Median time (ms) |
|---|---:|
| Previous schedule | 0.161792 |
| Delayed-half Q2 | **0.151584** |

Every pair won by 5.83-6.73%, with a 6.73% aggregate improvement. A cleaned
1000 ms timing run reproduced 0.152576 ms. A full apples-to-apples HAO run
measured 0.152864 ms for TK, 0.166688 ms for HAO BF16, and 0.193408 ms for
HAO full NVFP4. This makes the candidate 1.090x faster than BF16 at this
shape.

NCU attributes the wall-time gain to overlap rather than removed arithmetic:

| NCU metric | Previous | Delayed-half Q2 |
|---|---:|---:|
| Duration (ns) | 160352 | **149984** |
| Memory tensor active | 20.21% | **21.00%** |
| Tensor pipe active | 13.25% | **14.17%** |
| SM throughput | 48.04% | **51.37%** |
| Issue active | 42.59% | **45.15%** |
| Eligible warps/cycle | 0.50 | **0.56** |
| Long-scoreboard ratio | 5.28 | **4.62** |
| Wait ratio | 1.36 | **1.27** |
| Dynamic instructions | 65.72M | 65.62M |

The candidate survived 2001 consecutive launches. Reordering scalar
accumulation changes only final rounding: candidate versus control has a
maximum BF16 output delta of 0.000244 and maximum LSE delta of 9.54e-7.
Quality versus the BF16 reference is unchanged at 0.969948 cosine and
0.006417 RMSE. The candidate remains default-off until cross-shape dispatch
validation is complete.

## Post-delayed scheduling refinement

Two additional default-off ordering controls improve the delayed-half
schedule without changing its resources or arithmetic:

1. `HAO_FP4PV_MX_DELAYED_EARLY_Q3=1` issues the independent Q3 TMEM load
   immediately after Q1 packing, before the first-half scale-page store.
2. `HAO_FP4PV_MX_EARLY_Q2_REDUCE=1` executes retained Q2's exact max tree
   after launching that asynchronous scale-page store but before publishing
   first P. Q2 exp2 and FP4 packing remain after the publication.

On three matched S4096/H24 pairs, moving the Q2 max tree improved the mean
from 0.150272 ms to 0.149365 ms (0.60%). The clean final build measures
0.149696 ms over a 2-second timing window. It still uses 128 registers, one
barrier, 400 bytes of static shared memory, 208896 bytes of dynamic shared
memory, and 512 TMEM columns, with no stack or spills.

The repeated NCU pair confirms a scheduling gain rather than removed work:

| NCU metric | Early Q3 | Early Q3 + Q2 reduce |
|---|---:|---:|
| Replay duration (us) | 254.880 | **253.152** |
| Memory-tensor active | 21.41% | **21.58%** |
| Tensor-pipe active | 14.41% | **14.52%** |
| SM throughput | 52.27% | **52.63%** |
| Issue active | 46.10% | **46.27%** |
| Eligible warps/cycle | 0.56 | **0.57** |
| Long-scoreboard ratio | 4.52 | **4.50** |
| Dynamic instructions | 65.798M | 65.795M |

The gain survives long and saturated shapes while remaining neutral on the
short loop:

| Shape | Early Q3 (ms) | + Q2 reduce (ms) | Change |
|---|---:|---:|---:|
| S2048/H24 | 0.057472 | 0.057488 | -0.03% |
| S4096/H64 | 0.339968 | **0.339664** | +0.09% |
| S8192/H24 | 0.565392 | **0.563200** | +0.39% |
| S8192/H64 | 1.317504 | **1.313792** | +0.28% |

All four candidates have exactly the same output cosine and maximum BF16
error as their controls. The S8192/H24 apples-to-apples HAO-factory run now
measures 0.563200 ms for TK, 0.624992 ms for HAO BF16, and 0.726912 ms for
HAO full NVFP4. S4096/H24 measures 0.149536 ms, 0.165920 ms, and 0.192544 ms
respectively. The retained TK path is therefore 1.110x faster than BF16 at
both measured shapes.

### Ordering constraints established

NCU source sampling separates idle-role waits from critical waits. The
largest sampled wait belongs to the bypassed correction/epilogue role and is
not on the active pipeline. The actionable waits are the stage-0 and stage-1
softmax readers waiting for `score_full`; they arise because the tensor
issuer cannot put the next QK command between the two K64 PV accumulation
commands.

That restriction was tested directly. An aggressive
`first PV -> next QK -> tail PV` sequence traps on an illegal TCGEN
instruction. Progressively adding first-half ownership, tail ownership, and
deferred score publication makes it legal, but every synchronized version
produces NaN output. The two K64 PV commands form one inseparable
accumulation sequence on this path. Next QK may issue only after tail PV.

The remaining local permutations establish a narrow scheduling window:

| Probe | Result | Conclusion |
|---|---:|---|
| Finish Q2 packing before first P | 0.152288 ms | Too much first-P delay |
| Carry pair-max intermediates across first P | 0.150016 ms | Live range loses |
| Prepare Q2 scale before first P | 0.565429 ms at S8192/H24 | 0.39% slower |
| Store first scale before launching Q3 | 0.151509 ms | Q3 load is more urgent |
| Combine Q0/Q1 sums before first P | 0.151552 ms | One extra pre-signal add loses |
| Add Q3 sum after tail publication | 0.150603 ms | Existing fence hides the add |

The useful rule is precise: launch Q3 first, launch the asynchronous first
scale store second, and hide only Q2's max tree under that store. Any longer
pre-publication work delays first PV; carrying intermediate state across a
publication point hurts the compiler schedule; scalar work immediately
before the tail fence is already hidden by the pending TMEM stores.

## FP8-PV, long-context, and S4096 ceiling follow-up

The delayed-Q2 publication strategy was applied to NVFP4-QK + FP8-PV as a
matched structural control. FP8 has no P-scale construction, but copying the
MXFP4 schedule does not provide another win:

| S4096/H24 FP8-PV schedule | Time (ms) | Result |
|---|---:|---|
| Retained control | **0.159744** | Baseline |
| Early full Q3 | 0.167648-0.171072 | Regression |
| `2+1+1` K32 publication | 0.159872 | Neutral |
| Two chunks plus half-Q3 | 0.161792 | Regression |
| HAO native NVFP4-QK + FP8-PV | 0.158016 | Matched |
| HAO BF16 | 0.163168 | FP8 is 1.021x faster |

The K32 publication window is real, but FP8 slice stores, readiness, and
consumer loads cost as much as the overlap they create. Correctness remains
0.989724 cosine and 0.003676 RMSE versus the BF16 reference.

Long-context measurements show that both low-precision routes eventually
amortize the softmax machinery, but MXFP4-PV scales better:

| Shape | TK MXFP4-PV (ms) | TK FP8-PV (ms) | HAO BF16 (ms) |
|---|---:|---:|---:|
| S16384/H24 | **2.020352** | 2.167712 | 2.000448 |
| S32768/H24 | **7.746624** | 8.302624 | 9.145376-9.424928 |

At S16K, MXFP4 is approximately tied with BF16 while FP8 is 0.924x. At S32K,
MXFP4 is 1.181x faster than the matched 9.145 ms BF16 run and about 7% faster
than FP8. FP8 therefore remains a useful quality/control route, not the
preferred long-context implementation.

### S4096 speed-of-light accounting

Matched S4096/H24 builds using the final schedule measure:

| P construction | Time (ms) |
|---|---:|
| Exact retained kernel | **0.149664** |
| Fixed P | 0.085568 |
| Exact row max plus raw pack | 0.088384 |
| Raw score pack | 0.090112 |
| Fake denominator | 0.145408 |

NCU reports 65.79M dynamic warp instructions for exact P versus 17.91M for
row-max/raw-pack. Exact exponentiation and normalization therefore add about
47.9M instructions, while removing the denominator alone saves only 4.4 us.
The exact kernel has 14.46% tensor-pipe activity; row-max/raw-pack reaches
25.78%. Barrier stalls are only 0.28%, while long scoreboard accounts for
56.13% of sampled stalls. The remaining gap is arithmetic and dependent
publication latency, not a missing all-CTA barrier optimization.

Reaching 0.100 ms leaves only 11.6 us above the 0.0884 ms row-max/raw-pack
floor. Approximately 49.7 us, or 81% of the current exact softmax/pack
overhead, must be removed or hidden.

### Rejected arithmetic and ownership probes

The following default-off probes establish the current boundaries:

| Probe | Time (ms) | Quality/result |
|---|---:|---|
| Degree-2 ALU exp, mixed mask | 0.149792 | Exact; neutral |
| Degree-1 ALU exp, mixed mask | 0.149355 avg | Noise-level gain |
| Degree-1, all ALU | 0.160032 | ALU issue regression |
| Sample 16/32 values for max | 0.147008 | 0.919 cosine; reject |
| Sample 24/32 values for max | 0.147040 | 0.949 cosine; reject |
| Sample 28/32 values for max | 0.149536 | 0.960 cosine; no gain |
| Two-segment log-domain `exp2` | 0.201568 | 0.967 cosine; ALU-bound |
| Direct dual-Q3 correction WG | 0.155648 | Exact; score lease loses |
| BF16 shared Q3 staging | 0.204672 | Exact; store/load critical path |
| FP8 shared Q3 staging | 0.198656 | 0.96965 cosine; reject |
| FP8 P-tail TMEM checkpoint | 0.180224 | 0.96965 cosine; reject |
| Preload both P-tail checkpoints | 0.192544 | First-tail regression |

The P-tail checkpoint proves that Q3 can temporarily reuse the future P-tail
columns without allocating another permanent TMEM slot. One TCGEN x8 store
and x8 load are legal and avoid a score-TMEM lease. It still loses because the
owner must encode the checkpoint and wait for consumption before Q2 can
overwrite it; preloading both stages delays the more urgent stage-0 tail.

The final retained rebuild remains 128 registers, 400 bytes static shared
memory, 208896 bytes dynamic shared memory, 512 TMEM columns, and zero spills.
All experimental controls are disabled by default.

### Remaining credible routes to 0.1 ms

Local max, denominator, barrier, and polynomial changes are now too small or
move work onto an already saturated ALU path. A credible 0.1 ms design must
instead provide one of:

1. a second independent score/P construction owner with no score-copy path;
2. a hardware-cheap, quantization-aware exponential that preserves MUFU/ALU
   parallelism rather than replacing MUFU with more dependent ALU work;
3. enough TMEM for a true second score slot so QK can issue while P is
   consumed; or
4. an algorithmic softmax approximation with an explicitly accepted quality
   target, since exact max/exp/normalization cannot be reduced by 81% through
   the tested local transformations.

## Packed log-LUT quantizer

`HAO_FP4PV_MX_LOG_LUT_BITS=4` tests a 16-bin direct log-to-E2M1
quantizer. A packed `float2` affine transform maps each normalized score to a
uniform log2 bin, two immediate register words map that bin to the positive
E2M1 nibble, and the denominator is decoded from the emitted nibbles. The
feature is default-off.

Matched S4096/H24 results:

| Kernel | Time (ms) | Cosine vs BF16 | RMSE |
|---|---:|---:|---:|
| Retained exact kernel | **0.149664** | 0.969947 | 0.006413 |
| 16-bin log LUT | 0.489472 | 0.967907 | 0.006516 |

The approximation is numerically usable but computationally rejected. Ptxas
still reports 128 registers and zero spills, so occupancy is unchanged. The
kernel text grows from 3,632 to 6,224 static SASS instructions. Relative to
the retained kernel, the LUT introduces 264 `F2I`, 528 integer min/max, 256
integer comparisons, 288 selects, 398 right shifts, and 839 `LOP3` operations.
NCU measures about 194.0M executed scheduler instructions versus 65.79M for
the retained exact path. ALU activity reaches 43.76%, tensor activity falls to
4.36%, issue activity is 39.26%, and only 0.44 warps are eligible per cycle.

This closes the register-LUT direction: replacing `exp2` with dynamic
per-element indexing is a roughly 3x instruction expansion, not a latency
shortcut. A future arithmetic replacement must avoid `F2I`, dynamic shifts,
and per-element lookup selection while retaining the native packed E2M1
conversion. A clean rebuild with the LUT disabled measures 0.149440 ms with
the original 0.969947 cosine, confirming that the default production path is
unchanged.

## Bit classifier, packed polynomial, and SFU/FMA split

The direct log classifier was reduced further after the 16-bin LUT result.
The tested variants used FP32 magic-bias conversion, packed halfword
min/max, immediate 64-bit nibble tables, and `SHF.R.U64` extraction. This is
the closest implementation of "use the high bits to select an E2M1 bin" that
the local instruction set supports.

| Direct classifier | S4096/H24 (ms) | Cosine | Result |
|---|---:|---:|---|
| Full 16-bin LUT | 0.489472 | 0.967907 | Reject |
| Magic bits, one correction | 0.360736 | 0.970838 | Reject |
| Magic bits, two corrections | 0.427040 | 0.970310 | Reject |
| SIMD correction | 0.387072 | 0.970825 | Reject |
| Magic bits, no correction | 0.271392 | 0.963902 | Reject |
| Fused magic slope 1.8268 | 0.255872 | 0.959076 | Reject |
| Fused magic slope 2.0 | 0.255840 | 0.959999 | Reject |

An isolated packed-LUT probe still needs approximately 13 hot SASS
instructions per `float2`: permutation, two packed integer clamps, shifts,
two 64-bit funnel shifts, sign extension, and packing. The bin selector is
therefore more expensive than the value it replaces. The important lesson is
that the FP32 exponent bits are useful when the input is already exponential;
the attention score here is a logarithm, so extracting those bits does not
directly produce the nonuniform E2M1 thresholds.

### Threshold-fitted polynomial

The successful replacement leaves final quantization to Blackwell's native
packed instruction:

```
cvt.rn.relu.satfinite.e2m1x2.f32
```

Instead of indexing a LUT, it transforms each normalized log score into a
float that crosses the native E2M1 rounding boundaries at the correct log2
positions. The positive E2M1 values change code at:

```
0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
```

Fitting a cubic through those seven threshold constraints gives:

```
p(x) = 0.07839806*x^3
     + 0.28625049*x^2
     + 0.63145205*x
     + 0.99202336
```

For normalized score `x = m*s + q`, the coefficients are expanded once per
score quarter:

```
A' = A*m^3
B' = m^2*(3*A*q + B)
C' = m*(3*A*q^2 + 2*B*q + C)
D' = A*q^3 + B*q^2 + C*q + D
```

Each pair then needs only three packed `FFMA2` instructions followed by the
native relu E2M1 conversion. There is no per-element `F2I`, comparison tree,
dynamic shift, or LUT load.

The denominator is estimated from two rotating `float2` samples, four of the
32 values in each block. Their work is interleaved with FP4 packing. The
quality/speed progression was:

| Encoder / denominator | S4096/H24 (ms) | Cosine | RMSE |
|---|---:|---:|---:|
| Exact retained path | 0.149536 | 0.969947 | 0.006413 |
| Tuned two-segment PWL, 8 samples | 0.146432 | 0.968760 | 0.006429 |
| Tuned two-segment PWL, 4 samples | 0.143360 | 0.968232 | 0.006484 |
| Cubic, 4 samples | 0.130880 | 0.968439 | 0.006502 |
| Cubic, 8 samples | 0.135168 | 0.969270 | 0.006431 |
| Cubic, 2 samples | 0.131104 | 0.966800 | 0.006647 |
| Cubic, fake denominator | 0.124480 | 0.967653 | 0.016275 |

The two-sample variant removes 24 static instructions but does not improve
wall time. Those denominator clamps were already hidden under packing. The
fake-denominator result also has unacceptable magnitude error. A real
data-dependent denominator remains required.

### Balanced native/cubic encoder

Pure cubic moves pressure from SFU to FMA. A better implementation computes
one rotating pair from each packed word with native `EX2`, while the other
three pairs use the cubic. The two denominator-owned pairs are among the
native pairs, so their exact positive values need no clamp.

The number of native pairs per 16-pair quarter has a clear optimum:

| Native pairs | S4096/H24 (ms) | Cosine | Result |
|---:|---:|---:|---|
| 0 | 0.130880 | 0.968439 | Cubic control |
| 2 | 0.12755-0.12800 | 0.968477 | Win |
| 4 | 0.124928 | 0.968481 | Better |
| 8 | 0.130560 | 0.968484 | SFU regression |
| 16 | 0.151552 | 0.968500 | Reject |

Instruction placement matters at the four-pair point. The retained long-loop
schedule is:

1. Issue the word's native `EX2` pair first.
2. Evaluate its other three pairs with the packed cubic.
3. Pack all eight values with native E2M1 conversion.
4. Repeat for the next word.

Moving the two auxiliary native pairs to the beginning of their words gives
0.12376 ms on average. Also evaluating each rotating denominator pair first
gives 0.123008-0.123296 ms across four seeds. Front-loading all four native
pairs regresses to 0.124832 ms because it bunches SFU work instead of
covering each SFU dependency with local cubic work.

The final long-loop artifact is:

```
/tmp/_C_tk_mxfp4_cubic17_final_20260726.cpython-312-aarch64-linux-gnu.so
```

It uses 128 registers, one barrier, 400 bytes of static shared memory,
208896 bytes of dynamic shared memory, 512 TMEM columns, and zero spills.

### SASS and NCU evidence

| Static SASS | Exact | Scheduled hybrid |
|---|---:|---:|
| Total instructions | 3632 | 3480 |
| `FFMA2` | 176 | 320 |
| `FADD2` | 160 | 8 |
| `FMNMX` | 112 | 80 |
| `FMNMX3` | 88 | 88 |
| `F2FP` | 144 | 144 |
| `MUFU` | 229 | 69 |

Matched NCU replay shows that this is a pipeline improvement, not only an
instruction-count change:

| NCU metric | Exact | Cubic | Hybrid | Scheduled hybrid |
|---|---:|---:|---:|---:|
| Replay duration (ns) | 253536 | 219424 | 210944 | **207200** |
| Dynamic instructions | 65.792M | 59.210M | 57.603M | **57.583M** |
| Memory-tensor active | 21.54% | 24.70% | 25.63% | **25.70%** |
| Tensor-pipe active | 14.49% | 16.75% | 17.43% | **17.74%** |
| Issue active | 46.26% | 48.38% | 49.33% | **49.70%** |
| Eligible warps/cycle | 0.57 | 0.61 | 0.61 | **0.62** |
| Long-scoreboard ratio | 4.51 | 4.64 | 4.52 | **4.50** |
| Wait ratio | 1.24 | 0.66 | 0.67 | **0.67** |

The scheduled hybrid cuts dynamic instructions by 12.5%, increases
tensor-pipe activity by 22.4%, and returns long-scoreboard pressure to the
exact kernel's level.

### Shape dispatch

The per-word SFU-first schedule wins for S4096 and longer. At S2048, its
longer live ranges are not amortized, so the simpler rotating four-pair
schedule is faster.

| Shape | Exact MXFP4 (ms) | Best approximate (ms) | Change | Cosine |
|---|---:|---:|---:|---:|
| S2048/H24 | 0.057488 | **0.049152**, mode 13 | -14.5% | 0.966465 |
| S4096/H24 | 0.149536 | **0.123008**, mode 17 | -17.7% | 0.968481 |
| S4096/H64 | 0.338176 | **0.276480**, mode 17 | -18.2% | 0.968016 |
| S8192/H24 | 0.561472 | **0.454688**, mode 17 | -19.0% | 0.969165 |
| S16384/H24 | 2.020352 | **1.632272**, mode 17 | -19.2% | 0.969223 |

Against matched HAO BF16, S4096/H24 improves from 0.165920 to 0.123008 ms
(1.349x), and S8192/H24 improves from 0.624992 to 0.454688 ms (1.375x).
At S16384/H24, the prior exact MXFP4 path was approximately tied with BF16;
the scheduled hybrid improves on the 2.000448 ms BF16 result by 1.226x. The
gain increases under saturation rather than collapsing at larger head counts
or long context.

All polynomial and hybrid modes remain compile-time default-off because they
are approximate. A final exact rebuild after all source changes measures
0.149536 ms, 0.969947 cosine, and 0.006413 RMSE, matching the pre-experiment
kernel.

## First-quarter affine approximation sweep (2026-07-26)

The next sweep tested whether a cheaper approximation can shorten the first
PV-publication dependency chain. For selected score pairs in quarter 0, it
replaces the three-`FFMA2` cubic with

```
max(0, 1.62330034 * x + 0.92083546)
```

where `x` is the already scaled and shifted log2 softmax score. The
coefficients were fitted against the observed causal-attention distribution,
including MXFP4 block scaling. The measured E2M1 code frequencies in the
simulation reproduced the kernel output frequencies, so this was not a
uniform-input polynomial fit.

The cleaned compile-time policies are:

| Mode | Quarter-0 affine coverage | S4096/H24 median (ms) | Cosine | RMSE |
|---:|---|---:|---:|---:|
| 0 | None, exact path | 0.149760 | 0.969947 | 0.006413 |
| 17 | Scheduled native/cubic baseline | 0.123200 | 0.968481 | 0.006536 |
| 19 | Every non-native pair | **0.116752** | 0.963954 | 0.006960 |
| 20 | Every non-native pair, plus two native pairs | 0.118816 | 0.964701 | 0.006891 |
| 21 | Packed word 0 only | 0.120640 | 0.967360 | 0.006644 |
| 22 | Pair 3 only | 0.122656 | 0.968082 | 0.006574 |

Mode 19 is the aggressive throughput endpoint: 5.2% faster than mode 17 with
a 0.00453 cosine decrease. Mode 22 is the conservative endpoint: 1.9% faster
in its best short run and 0.4% faster by the longer-run median, with only a
0.00040 cosine decrease. Its narrow timing gain is correspondingly noisier
than modes 19-21.

Mode 22 also wins under higher head occupancy and longer sequence length:

| Shape | Mode 17 (ms) | Mode 22 (ms) | Change | Mode 17 / 22 cosine |
|---|---:|---:|---:|---:|
| S4096/H64 | 0.276512 | **0.272384** | -1.49% | 0.968016 / 0.967639 |
| S8192/H24 | 0.454688 | **0.448608** | -1.34% | 0.969165 / 0.968795 |

Static SASS and matched NCU replay identify a latency effect rather than a
large instruction-count effect:

| Metric | Mode 17 | Aggressive Q0 affine | Mode 22 |
|---|---:|---:|---:|
| Static instructions | 4360 | 4296 | 4360 |
| Static `FFMA2` | 320 | 272 | 316 |
| Static `MUFU.EX2` | 64 | 64 | 64 |
| Replay duration (ns) | 208416 | 196960 | 205280 |
| Dynamic instructions | 57.582M | 54.421M | 57.580M |
| Tensor-pipe active | 17.64% | 18.70% | 17.93% |
| Issue active | 49.71% | 49.63% | 50.35% |
| Eligible warps/cycle | 0.62 | 0.62 | 0.63 |

Mode 22 removes only four static packed FMAs but still lowers wall time across
all checked shapes. The useful lever is therefore the latency of the
quarter-0 score-to-P chain and first PV issue, not total arithmetic alone.

Rejected alternatives were more native `EX2` pairs (SFU pressure), quadratic
fits (unacceptable quality), FP16-packed cubic evaluation (conversion cost
and scalar `MUFU.EX2` remained), a rectified square, and affine coverage in
quarters 1-3. Quarter 0 was uniquely valuable because later-quarter work is
better hidden behind already-issued PV work.

Modes 19-22 remain default-off approximations. Mode 0 was rebuilt after the
cleanup and retained its exact-path output and timing.

## Quarter-0 direct E2M1 code experiment (2026-07-26)

The first-quarter result above motivated a stricter test: bypass floating-point
E2M1 values and have the approximation emit packed E2M1 ordinal codes directly.
The positive E2M1 grid is

```
code:   0    1    2    3    4    5    6    7
value:  0   0.5   1   1.5   2    3    4    6
```

An exact seven-threshold oracle in log2 space used thresholds

```
-2, log2(0.75), log2(1.25), log2(1.75),
log2(2.5), log2(3.5), log2(5)
```

and reproduced the native/cubic quality: 0.968483 cosine and 0.006535 RMSE.
This validates direct ordinal classification as a numerical formulation, but
the fourteen comparisons per `float2` made the oracle slow at 0.335728 ms.

The compressed classifier fitted to the observed attention distribution was

```
code = round(1.63745 * x + 2.30780836)
```

where `x` is the scaled and shifted log2 score. A magic-bias carrier placed
lane 0's code in mantissa bits 0-2 and lane 1's code in bits 4-6. One `LOP3`
selected the two nibbles and three `PRMT` instructions assembled four pair
bytes into a packed 32-bit payload word.

Matched S4096/H24 results were:

| Path | Median (ms) | Cosine | RMSE |
|---|---:|---:|---:|
| Mode 17 native/cubic | 0.123200 | 0.968481 | 0.006536 |
| Mode 19 Q0 affine + native pack | **0.116752** | 0.963954 | 0.006960 |
| Fixed Q0 payload, timing-only | 0.119104 | invalid | invalid |
| Unclamped carrier | 0.122944 | 0.654769 | 0.024715 |
| Lower-clamped carrier | 0.127360 | 0.962916 | 0.007091 |
| Fully clamped carrier | 0.133120 | 0.966217 | 0.006789 |

The fixed-payload result is not a useful zero-work wall-time ceiling: removing
all independent Q0 classifier arithmetic exposed more pipeline latency and was
2.0% slower than mode 19.

Kernel-only static SASS explains why the carrier failed:

| Path | Total | `F2FP` | `FFMA2` | `FMNMX` | `LOP3` | `PRMT` |
|---|---:|---:|---:|---:|---:|---:|
| Mode 17 | 3449 | 144 | 320 | 80 | 71 | 2 |
| Mode 19 | **3385** | 144 | 272 | 80 | 71 | 2 |
| Unclamped carrier | 3449 | 112 | 276 | 80 | 135 | 26 |
| Lower-clamped carrier | 3513 | 112 | 276 | 144 | 135 | 26 |
| Fully clamped carrier | 3577 | 112 | 276 | 208 | 135 | 26 |

Even the numerically invalid carrier saves only 32 `F2FP` instructions while
adding 64 `LOP3` and 24 `PRMT` instructions. Clamping adds another 64 or 128
`FMNMX` instructions. Native `F2FP.SATFINITE.RELU.E2M1.F32.PACK_AB_MERGE_C`
already performs classification, saturation, conversion, and dense packing;
the proposed bit carrier cannot match that packing density. The direct-code
carrier modes were therefore rejected and removed. Mode 23 was subsequently
reused for the independent scale-lifetime policy below.

## Post-carrier six-direction sweep (2026-07-26)

The carrier result showed that replacing native FP4 conversion was the wrong
target. Six independent directions were then tested around mode 19, the
aggressive quarter-0 affine policy.

### 1. Refreshed ceilings and timeline

Matched S4096/H24 diagnostic builds measured:

| Diagnostic | Time (ms) |
|---|---:|
| Mode 19 real path | 0.116736-0.117312 |
| Fixed P payload | 0.086016 |
| Score-pack ceiling | 0.090400 |
| Row-max-pack ceiling | 0.088064 |
| Fake denominator | 0.118944 |

About 30.7 us remains between the real path and fixed-P floor. Removing the
denominator is not useful by itself. Global-timer events showed that stage 0
has tail slack, while stage 1 quarter 3 remains on the first-PV critical path.

### 2. Shorter polynomial dependency chains

Estrin evaluation added a packed multiply/FMA and lost in every placement:

| Estrin placement | Time (ms) |
|---|---:|
| All quarters | 0.124928 |
| Q2 and Q3 | 0.122880 |
| Q3 only | 0.120832 |

The serial Horner chain is cheaper on this instruction mix.

### 3. Word lookahead and SFU dephasing

Looking ahead to the next packed word measured 0.117120 ms globally,
0.117056 ms in Q2/Q3, and 0.117760 ms in Q3 only. Changing the rotating native
`EX2` pair offset measured 0.116736-0.116768 ms. Neither changed the limiting
dependency enough to retain.

### 4. Native/affine/quadratic quarter-0 mixes

Reducing native quarter-0 pairs could match timing but degraded quality.
Removing all native pairs also exposed a divergent full-mask shuffle lifetime
and failed under Compute Sanitizer. A refitted affine reached 0.116288 ms but
reduced cosine to 0.962438. A monotone quadratic retained 0.963496 cosine but
cost 0.120672 ms. These variants were removed.

### 5. Fill the quarter-0 window

Prefetching the high half of Q2 before quarter-0 work increased time to
0.118784 ms. Releasing the already-consumed QK-scale page immediately after
its TMEM load, before quarter-0 reduction and packing, was the sole winner.
This is mode 23. It changes no arithmetic and lets the QK producer prepare the
next score tile while the reader processes quarter 0.

Alternating one-second measurements used two compile partitions at S2048 and
the release default of one partition at S4096 and above:

| Shape | Mode 19 (ms) | Mode 23 (ms) | Change | Cosine, both |
|---|---:|---:|---:|---:|
| S2048/H24 | **0.049280** | 0.049440 | +0.32% | 0.961989 |
| S4096/H24 | 0.117440 | **0.116736** | -0.60% | 0.963954 |
| S4096/H64 | 0.262144 | **0.260416** | -0.66% | 0.963476 |
| S8192/H24 | 0.429632 | **0.428320** | -0.31% | 0.964664 |

The S2048 difference lies inside the alternating run-to-run spread and is
treated as neutral. The longer/saturated shapes show a repeatable benefit.
As a separate build audit, forcing two compile partitions at S4096/H24
regressed mode 23 from 0.116736 to 0.123584 ms. Long-shape release and
benchmark builds must retain the Makefile's single-partition default.

Matched S4096/H64 NCU captures support a dependency-lifetime explanation:

| Metric | Mode 19 | Mode 23 |
|---|---:|---:|
| Replay duration (us) | 447.616 | 445.984 |
| Dynamic instructions | 144.265M | 144.534M |
| Issue active | 47.17% | 47.54% |
| Eligible warps/cycle | 0.626 | 0.641 |
| Long-scoreboard stall ratio | 4.532 | 4.410 |
| Wait stall ratio | 0.707 | 0.688 |
| TC-pipe active | 35.98% | 35.62% |

Mode 23 executes 0.19% more instructions, but exposes 2.48% more eligible
warps and shortens replay by 0.36%.

### 6. Deeper ownership splits

The existing split-PV policy measured 0.174080 ms and the existing dual-Q3
owner measured 0.135488 ms. A new stage-1-only Q3 correction-warpgroup owner
was correct and reduced long-scoreboard pressure, but its best register grant
still measured 0.122880 ms. The handoff added 1.61M dynamic instructions and
lost about 5% to mode 19. All ownership experiments were removed.

### Retained result

Mode 23 is the only retained kernel policy from this sweep. It is appropriate
for S4096 and longer loops; mode 19 remains preferable at S2048. The register
budget assertion was also corrected to account for stage-0 and active
correction grants separately.

## Score/max/scale ILP and ALU/XU rebalance (2026-07-27)

Mode 23 was profiled before changing the score/max/scale schedule. It is not
an all-ALU kernel:

| Baseline property, S4096/H24 | Value |
|---|---:|
| Static instructions | 4296 |
| Static `FMNMX` / `FMNMX3` | 80 / 88 |
| Static `FFMA2` | 272 |
| Static `MUFU` | 69 |
| NCU ALU-pipe active | 32.64% |
| NCU FMA-pipe active | 26.96% |
| NCU XU instructions | 3.542M |
| Math-pipe-throttle ratio | 0.156 |

The block maximum already compiles as a wide balanced tree. SASS also
interleaves its independent max instructions with coefficient work and
existing native `EX2` operations.

### Max and cross-quarter ILP

Temporary modes tested a 33-input ternary tree, early reduction of the
already-loaded low Q2 half, and their combination:

| Variant | Time (ms) | Cosine | Static max instructions |
|---|---:|---:|---:|
| Mode 23 | **0.116736** | 0.963228 | 168 |
| Ternary tree | 0.118784 | 0.963228 | 130 |
| Early Q2-half reduction | 0.116736 | 0.963228 | 168 |
| Ternary plus early Q2 half | 0.119072 | 0.963228 | 130 |

The early-Q2 source rewrite compiled to the same kernel SASS hash as mode 23:
the compiler had already performed that cross-quarter scheduling. The ternary
tree removed 38 max instructions and one conceptual dependency level, but
reduced the scheduler's pool of independent operations and lost 1.8-2.0%.

As a more aggressive scale approximation, sampling seven of every eight
values for the block maximum removed 32 additional `FMNMX` instructions.
It still slowed to 0.118784 ms, reduced cosine to 0.954025, and increased max
absolute error from 0.068848 to 0.184692. Deeper sampling was not justified.

### Native EX2 redistribution

Mode 23 already sends four well-spaced pairs per quarter to native `EX2`; the
remaining pairs use packed affine or cubic ALU evaluation. Several targeted
splits moved more work to XU:

| Temporary split | `FFMA2` | `MUFU` | Time (ms) | Cosine |
|---|---:|---:|---:|---:|
| Mode 23, four native/quarter | 272 | 69 | **0.116736** | 0.963228 |
| Six native in Q3 | 264 | 77 | 0.121472 | 0.963230 |
| Six native in Q1/Q2 | 256 | 85 | 0.118784 | 0.963229 |
| Eight native in Q1/Q2 | 248 | 93 | 0.120384 | 0.963230 |
| One extra native pair in Q2 | 268 | 73 | 0.122880 | 0.963229 |

The Q1/Q2 six-native profile confirms that the intended traffic shift
occurred:

| NCU metric | Mode 23 | Q1/Q2 six-native |
|---|---:|---:|
| Replay duration (us) | 197.504 | 202.752 |
| FMA instructions | 22.236M | 21.449M |
| XU instructions | 3.542M | 4.329M |
| Issue active | 40.62% | 39.54% |
| Eligible warps/cycle | 0.633 | 0.623 |
| Long-scoreboard ratio | 4.366 | 4.509 |
| Math-pipe-throttle ratio | 0.156 | 0.152 |

FMA pressure falls, but XU latency reduces eligible work and raises
long-scoreboard stalls. This is not an ALU-throughput bottleneck; extra native
`EX2` cannot be hidden even in Q1/Q2. All temporary modes were removed and
mode 23 remains unchanged.

## Stage-owned ceiling and score-load overlap (2026-07-27)

Commit `f096e07` was pushed before this profiling pass. Its clean
S4096/H24 mode-23 baseline reproduces at 0.116288 ms with 0.963228 cosine,
while the matched HAO BF16 kernel is 0.162336 ms. The 0.100000 ms target
therefore required 16.288 us, or 14.0%, from that checkpoint.

### Stage and quarter ceilings

Diagnostic-only template specializations replaced selected P transforms with
raw E2M1 packing. Their outputs are intentionally invalid; only the timing
dependency is meaningful.

| Raw-pack region | Time (ms) | Gain from 0.116288 ms |
|---|---:|---:|
| Stage 1, all quarters | 0.110592 | 5.696 us |
| Stage 0, Q0 only | 0.110112 | 6.176 us |
| Stage 0, Q3 only | 0.109184 | 7.104 us |
| Stage 0, Q0+Q1 | 0.104448 | 11.840 us |
| Stage 0, Q2+Q3 | 0.104736 | 11.552 us |
| Stage 0, all quarters | **0.094496** | **21.792 us** |

Stage 0 is the leverage point because its first and tail P publications gate
the first `PV -> QK` transition. The four quarter costs are broadly
distributed; no Q3-only or first-K64-only change can reach 0.1 ms.

Matched NCU controls retain exactly 98,304 tensor instructions:

| Metric | Real P | Fixed P | Score pack | Row-max pack |
|---|---:|---:|---:|---:|
| Replay duration (us) | 195.744 | 139.744 | 147.200 | 145.088 |
| Dynamic instructions (M) | 54.557 | 11.858 | 18.015 | 17.993 |
| FMA instructions (M) | 22.235 | 2.624 | 1.669 | 1.764 |
| Tensor-pipe active | 18.81% | 26.55% | 25.08% | 25.56% |
| Eligible warps/cycle | 0.634 | 0.177 | 0.253 | 0.257 |
| Long-scoreboard ratio | 4.363 | 19.052 | 12.450 | 12.489 |

Removing P arithmetic exposes tensor and readiness dependencies, so an
optimization must preserve enough independent work to cover those waits.
Source sampling attributes 1,715 of the real path's 2,669 not-issued samples
to long scoreboards. The largest sites are the output warpgroup waiting for
final softmax statistics, both readers waiting for `score_full`, and the TMA
epilogue waiting for output publication. Scale-byte shared-memory stores
produce excessive wavefronts but almost no sampled stalls and are not the
first-order target.

### Approximation controls

Applying the existing one-`FFMA2` affine mapping to every non-native stage-0
pair reduces FMA instructions from 22.235M to 17.259M and measures
0.107520 ms at 0.956344 cosine. NCU replay falls to 180.320 us, but eligible
warps also fall to 0.613 and the long-scoreboard ratio rises to 4.505.

Further controls separate arithmetic cost from useful scheduling:

| Control | Time (ms) | Cosine | Conclusion |
|---|---:|---:|---|
| Stage-0 affine | 0.107520 | 0.956344 | Real gain, insufficient |
| Stage-0 affine, no native pairs | 0.107072 | 0.949393 | XU removal is not the gap |
| Stage-0 affine, fake denominator | 0.110592 | 0.923699 | Denominator work hides latency |
| Fixed stage-0 E8M0 = 129 | 0.102400 | 0.897987 | Scale/max ceiling, invalid quality |
| Fixed stage-0 E8M0 = 130 | 0.102400 | 0.928587 | Less clipping, still invalid |
| Q0-derived scale reused in Q1-Q3 | 0.108544 | 0.928810 | Cross-quarter dependency loses ILP |
| 128 persistent CTAs | 0.108064 | 0.956344 | Exact 3-job waves do not help |

Fixed scaling proves that the stage-0 max/scale chain contains about 5 us of
headroom, but a constant scale cannot represent per-block score variation.
Reusing a dynamic Q0 scale is slower because it serializes all later quarters.
A future scale predictor must remain quarter-local and avoid an inter-quarter
dependency.

### Retained stage-0 prefetch

The successful exact change keeps stage 1 unchanged. In stage 0 only, it
launches the Q1 score load and the non-overwritten high half of Q2 before the
Q0 transform. Q0 arithmetic covers those TMEM reads; the reader consumes them
after Q0. Prefetching Q3 as well regresses to 0.107008 ms in the affine
control because its register lifetime and TMEM queue occupancy are too large.

The cleaned normal mode-23 kernel measures:

| Shape | Prior mode 23 (ms) | Stage-0 prefetch (ms) | Change |
|---|---:|---:|---:|
| S2048/H24 | 0.049440 | **0.047392** | -4.14% |
| S4096/H24 | 0.116288 | **0.114240** | -1.76% |
| S4096/H64 | 0.260416 | **0.256672** | -1.44% |
| S8192/H24 | 0.428320 | **0.420192** | -1.90% |

All builds use 128 registers, one barrier, 400 bytes static shared memory,
and no spills. S4096/H24 retains 0.963228 cosine and 0.006934 RMSE.
Compute Sanitizer memcheck reports zero errors when unrelated
`cuGetProcAddress` API diagnostics from the imported CUTE stack are disabled.

Combining the retained prefetch with the stage-0 affine diagnostic reaches
0.105056 ms at 0.956344 cosine. It is 5.056 us short of the target and is not
promoted because of its quality loss. The production path is 14.240 us short.
The remaining credible route is a stage-0-specific structural change that
overlaps transform work without adding a full score handoff: either a
quarter-local cheap scale/transform predictor, or a carefully redesigned
correction-warpgroup owner for stage-0 tail work. More native `EX2`, fake
denominators, global prefetching, and cross-quarter scale reuse are ruled out
by the controls above.

## Approximation localization and format comparison (2026-07-27)

The retained mode-23 transform uses four native `EX2` pairs per quarter and
the following cubic for each remaining packed pair:

```text
c(x) = 0.07839806 x^3 + 0.28625049 x^2
     + 0.63145205 x + 0.99202336
```

The stage-0 approximation replaces that cubic with one packed `FFMA2`:

```text
a(x) = 1.62330034 x + 0.92083546
```

This saves two packed FMAs for each non-native pair in stage-0 Q1, Q2, and
Q3. It is fast because it removes most of the serial ALU chain before first
PV. It is inaccurate because E2M1 output depends on crossing discrete
rounding boundaries, not on average function error:

| E2M1 rounding boundary | `log2(y)` | Cubic crossing | Affine crossing |
|---:|---:|---:|---:|
| 0.25 | -2.00000 | -1.99281 | -0.41325 |
| 0.75 | -0.41504 | -0.47081 | -0.10524 |
| 1.25 | 0.32193 | 0.34830 | 0.20277 |
| 1.75 | 0.80735 | 0.82356 | 0.51079 |
| 2.50 | 1.32193 | 1.31743 | 0.97281 |
| 3.50 | 1.80735 | 1.79450 | 1.58884 |
| 5.00 | 2.32193 | 2.32722 | 2.51288 |

The first positive-code boundary moves by `+1.58` in log2 space. Therefore a
broad interval of valid low probabilities is rounded to zero. Direct output
comparison confirms a systematic error rather than a few outliers:

- Only even 128-row query tiles change because they are owned by stage 0;
  stage-1/odd tiles are bit-identical.
- Affine versus exact has 0.986991 global cosine, but only 0.973921 on
  stage-0 tiles.
- Error is uniform across heads, query tiles, and output magnitudes.
- Against BF16, the complete affine path falls from 0.963228 to 0.956344
  cosine.

Quarter controls show the latency contribution is also distributed:

| Stage-0 affine region | Time (ms) | BF16 cosine |
|---|---:|---:|
| Q1 | 0.110720 | 0.960893 |
| Q2 | 0.111680 | 0.960970 |
| Q3 | 0.114688 | 0.960954 |
| Q1 + Q2 | 0.108544 | 0.958624 |
| Q1 + Q2 + Q3 | 0.105056 | 0.956344 |

Q3 has no isolated timing gain but becomes useful after Q1/Q2 are shortened,
which identifies it as a tail dependency rather than independent arithmetic
headroom.

### Quarter-local scale prediction

Sampling E8M0 maxima inside each block was tested without introducing an
inter-quarter dependency:

| Maximum samples per 32 values | Time (ms) | BF16 cosine |
|---:|---:|---:|
| 16 | 0.112640 | 0.914511 |
| 24 | 0.112640 | 0.943393 |
| 28 | 0.112192 | 0.954025 |
| 28 plus half-exponent safety bias | 0.114784 | 0.955762 |

A single missed maximum chooses an undersized power-of-two scale and clips
the whole block. The predictor saves at most about 2 us and loses too much
quality; a conservative bias removes the speed benefit without repairing the
error. This direction is rejected.

### NVFP4 versus MXFP4 P/PV

A matched NVFP4 block-16 E4M3 path was moved onto the same one-pass
quarter-local schedule as MXFP4. Folding normalization into the cubic and
balancing four native exponent pairs against ALU work reduced it from
0.192512 ms to 0.149504 ms.

| S4096/H24 path | Time (ms) | BF16 speedup | BF16 cosine | RMSE |
|---|---:|---:|---:|---:|
| HAO BF16 | 0.163392-0.163840 | 1.00x | 1.000000 | 0 |
| TK MXFP4 mode 23 | **0.114688** | **1.43x** | 0.963228 | 0.006934 |
| TK NVFP4 folded hybrid | 0.149504 | 1.09x | **0.981252** | **0.004970** |
| HAO native NVFP4 | 0.192512 | 0.85x | - | - |

NVFP4 is the clear quality option, but its two E4M3 block-16 scales and
additional conversion path add roughly 35 us over MXFP4. It does not provide
a route to 0.1 ms for this shape.

### Stage-0 Q3 ownership offload

The correction warpgroup was assigned stage-0 Q3 while the normal softmax
warpgroup retained Q0-Q2. A dedicated Q2-ready handoff was required because
the current schedule releases the old QK-scale lease before Q0.

| Q3 owner diagnostic | Apparent time (ms) |
|---|---:|
| Original single owner | **0.114688** |
| Correction owner, original mode-23 mix | 0.121184 |
| Correction owner, 8 native Q3 pairs | 0.119040 |
| 8 native pairs, stage-0 tail publication | 0.117440 |
| Per-row tail-ready semaphores | 0.118784 |

These offload timings are diagnostic only. Racecheck slows the producer and
exposes a scale-page lifetime violation: stage 0 can begin writing the next
tile's Q0-Q2 scale bytes before the correction warpgroup has copied the
current tile's combined Q2/Q3 word. The instrumented offload produces NaNs,
whereas the original path remains numerically stable under the same tool.
Making the offload legal requires either:

1. an acknowledgment from the correction owner before scale-page reuse, or
2. another ping-pong Q3 scale page with a complete empty/full lifecycle.

The second option was implemented with two 512-byte shared-memory scale
pages. The correction owner waits for an empty slot, writes Q3's scale byte,
and publishes the slot. Stage 0 waits for it, merges Q3 with its local Q2
byte, copies the complete tail scale word to TMEM, and returns the slot.

This legal version is stable across five independent inputs, matches exact to
BF16 rounding noise, and passes Compute Sanitizer memcheck. It measures
0.120832 ms, so it is 6.144 us slower than the single-owner path:

| NCU metric | Single owner | Legal Q3 scale ring |
|---|---:|---:|
| Replay duration (us) | 194.240 | 204.480 |
| Dynamic instructions (M) | 54.549 | 57.868 |
| FMA instructions (M) | 22.293 | 23.274 |
| Tensor-pipe active | 19.08% | 18.01% |
| Eligible warps/cycle | 0.656 | 0.745 |
| Math-pipe throttle | 0.162 | 0.309 |
| Barrier-wait ratio | 0.644 | 0.853 |

The offload creates more eligible work, but that work is handoff and
concurrent ALU traffic rather than useful tensor issue. The previously
measured native-`EX2` rebalance can recover only about 2.1 us, which is not
enough to close the 6.1 us loss. The original compile-time prohibition on
combining delayed-Q2 with single-stage Q3 offload has therefore been restored,
and all scale-ring controls were removed.

No approximation, scale predictor, NVFP4 substitution, or stage-0 offload
tested here beats the exact MXFP4 single-owner path. The main actionable
finding is narrower: a future structural win must keep the score/P ownership
local while shortening the cubic threshold path. Moving the same arithmetic
to another warpgroup is not free overlap on GB200 because both groups contend
for the same math and TMEM issue resources.

## 2026-07-27: matched stage-1 NVFP4 approximation

The earlier NVFP4 conclusion compared different numerical policies. NVFP4
already omitted a tensor-wide P amax and global tensor scale, but only stage 0
used the fast affine approximation. Stage 1 retained the cubic approximation
on Q1-Q3, placing that longer dependency chain on every other query tile.
NVFP4 must still publish its hardware-required block-16 E4M3 scales; this
change does not remove or fake those scales.

An independent `TK_HAO_DIRECT_FP4PV_NV_STAGE1_AFFINE_MASK` now controls the
stage-1 policy. The useful localization results at S4096/H24/D128 are:

| Stage-1 affine quarters | Mask | Time (ms) | BF16 cosine | RMSE |
|---|---:|---:|---:|---:|
| none | 0 | 0.108544 | 0.968036 | 0.006544 |
| Q1 + Q2 | 6 | 0.104448 | 0.963849 | 0.006978 |
| Q2 + Q3 | 12 | 0.108544 | 0.963846 | 0.006977 |
| Q1 + Q2 + Q3 | 14 | **0.102752 median** | **0.961779** | **0.007184** |

Q1 is the primary stage-1 latency cut. Q3 becomes useful once Q1/Q2 are
shortened, matching the previously observed stage-0 tail behavior. The
all-quarter policy was retained; rejected early-reduction schedule modes 3-5
were removed.

Five independent 1-second benchmark processes produced
`0.102912, 0.102400, 0.102400, 0.102752, 0.103424 ms`. A same-process
comparison of the cleaned artifact and the pre-cleanup winner measured
`0.102720` and `0.102784 ms`, with bit-identical outputs. The matched
cross-provider result is:

| S4096/H24 path | Time (ms) | Speedup vs HAO BF16 | BF16 cosine |
|---|---:|---:|---:|
| HAO BF16 | 0.162368 | 1.00x | 1.000000 |
| HAO native NVFP4/NVFP4 | 0.192928 | 0.84x | - |
| pinned TK MXFP4 mode-23, 5-run median | 0.104800 | 1.55x | 0.956490 |
| TK NVFP4/NVFP4, stage 0+1 affine | **0.102752** | **1.58x** | **0.961779** |

The NVFP4 winner is therefore 1.95% faster than the pinned MXFP4 result while
also improving cosine by 0.00529. Seeds 0-3 produced cosine
`0.961779-0.962450` and RMSE near `0.00719`.

The profiler confirms that this is a dependency-chain reduction rather than
removal of scale work. Relative to the stage-1-cubic NV control, elapsed
profiled cycles fell from 202,227 to 195,378 and executed instructions from
46.434 million to 41.262 million. The final build uses 128 registers, one
barrier, and no spills. Compute Sanitizer memcheck reports zero errors when
import-time CUDA API diagnostics are disabled.

The final artifact is `/tmp/_C_tk_nvfp4_stage01_affine.so`, module
`_C_tk_nvfp4_stage01_affine`. Its distinguishing build controls are:

```text
HAO_QK_SCALE_MODE=0
HAO_PV_SCALE_MODE=0
HAO_FP4PV_EARLY_P=1
HAO_FP4PV_NV_SHIFTLESS_SOFTMAX=1
HAO_FP4PV_NV_PWL_EXP2=9
HAO_FP4PV_NV_STAGE0_AFFINE_MASK=14
HAO_FP4PV_NV_STAGE1_AFFINE_MASK=14
HAO_FP4PV_NV_QUARTER_SCALE=1
HAO_FP4PV_NV_SCALE_ENCODE=2
HAO_FP4PV_NV_QUARTER_SCHEDULE=1
HAO_FP4PV_NV_EARLY_DIRECT_SCALE=1
HAO_FP4PV_MX_DELAYED_HALF_Q2=1
HAO_FP4PV_MX_DELAYED_EARLY_Q3=1
HAO_FP4PV_MX_EARLY_Q2_REDUCE=1
HAO_FP4PV_MX_SHIFTLESS_CORR_BYPASS=1
```

## 2026-07-27: HAO GB200/GB300 ceiling comparison

The final NVFP4 policy was rebuilt unchanged for S32768/H24 and measured
against HAO's native providers using HAO's published protocol:

- noncausal B1/H24/Dqk128/Dvo128;
- seed 0 and the HAO `create_nvfp4_attention_tensors` factory;
- `triton.testing.do_bench`, `warmup=10 ms`, `rep=25 ms`, median;
- dedicated idle GB200 GPU 0, driver 580.126.09, maximum SM clock 2062 MHz;
- CuTe DSL 4.5.2 and FlashInfer 0.6.15.post1.

The upstream snapshot is commit `9b0abef` from
`hao-ai-lab/flash-attention-fp4:fp4`. Its only runtime source adjustment is
the documented default-zero initialization of the otherwise-unbound
`fp8_pv_p_log2_offset`. The README reports CuTe DSL 4.4.2, so the published
and local HAO columns are retained separately.

Throughput uses HAO's noncausal FLOP convention:

```text
FLOPs = B * H * 2 * Sq * Sk * (Dqk + Dvo)
```

### Local GB200 results

| Shape/provider | Median time | Throughput | BF16 output cosine |
|---|---:|---:|---:|
| S4096 TK NVFP4/NVFP4 | **0.102656 ms** | **2008.2 TFLOPS** | 0.961779 |
| S4096 HAO NVFP4/NVFP4 | 0.192512 ms | 1070.9 TFLOPS | 0.981415 |
| S4096 HAO NVFP4/FP8 | 0.158-0.159 ms | 1299.7-1307.1 TFLOPS | 0.989724 |
| S4096 HAO BF16 | 0.163808 ms | 1258.5 TFLOPS | 1.000000 |
| S32768 TK NVFP4/NVFP4 | **5.104000 ms** | **2585.0 TFLOPS** | 0.963041 |
| S32768 HAO NVFP4/NVFP4 | 9.750528 ms | 1353.2 TFLOPS | 0.981812 |
| S32768 HAO NVFP4/FP8 | 8.295 ms | 1590.6 TFLOPS | 0.989868 |
| S32768 HAO BF16 | 8.792096 ms | 1500.7 TFLOPS | 1.000000 |

The S32768 TK times from three independent processes were `5.104032`,
`5.104000`, and `5.103984 ms`. TK is 1.91x the local HAO full-FP4
throughput and 1.72x the local BF16 throughput at this point.

### Published HAO points

The matching rows in HAO's README are:

| Shape/hardware | NVFP4 + FP8 | NVFP4 + NVFP4 | BF16 |
|---|---:|---:|---:|
| S4096/H24 B200 | 1548 TFLOPS | not reported | 1274 TFLOPS |
| S4096/H24 GB300 | 2046 TFLOPS | 1291 TFLOPS | 1322 TFLOPS |
| S32768/H24 B200 | 2018 TFLOPS | not reported | 1545 TFLOPS |
| S32768/H24 GB300 | 2677 TFLOPS | 1725 TFLOPS | 1533 TFLOPS |

At S32768, TK full FP4 is 49.86% above HAO's published GB300 full-FP4
number and only 3.44% below HAO's best GB300 FP8-PV number. It also exceeds
HAO's published B200 FP8-PV result by 28.10%. This supports the claim that
the present schedule is close to the demonstrated FP4-QK/FP8-PV throughput
ceiling, even though TK also performs P quantization and FP4 PV.

This is not an equal-accuracy claim. HAO's full-FP4 path retains roughly
0.9818 cosine at S32768, while the selected TK affine/sampled path gives
0.9630. The remaining 3.4% to the GB300 FP8-PV headline is therefore a useful
performance reference, not proof that the approximate full-FP4 kernel has
reached the hardware limit for exact softmax.

## 2026-07-27: matched-strategy FP8-PV control

The separate TK FP8-PV route was given the full-FP4 winner's one-pass,
quarter-local score lifecycle. Absolute probabilities are carried directly
in E4M3 because plain FP8 has no P block scale. Four pairs per quarter use
native `EX2`; the remainder use either the selected FP4 cubic/affine mix or
the cubic on all quarters. The latter is required for FP8 because the FP4
affine expects the block scale to shift its input toward the E2M1 grid.

At S4096/H24/D128:

| FP8-PV policy | Time (ms) | TFLOPS | BF16 cosine |
|---|---:|---:|---:|
| FP4 cubic/affine mix, exact denominator | **0.141088** | 1461.2 | 0.855215 |
| Cubic throughout, exact denominator | 0.163840 | 1258.3 | 0.957108 |
| Cubic throughout, sampled denominator | **0.148224** | **1390.9** | **0.956616** |
| Local HAO exact NVFP4-QK/FP8-PV | 0.158272 | 1302.6 | 0.989724 |
| TK full FP4 winner | 0.102656 | 2008.2 | 0.961779 |

The fast affine FP8 result is only a ceiling: without an FP4 block scale, its
first positive interval is wrong and it zeros too much of the softmax tail.
The sampled all-cubic policy is the useful structural control. It is 6.78%
faster than local HAO FP8 but the selected full-FP4 kernel remains 44.39%
higher-throughput and slightly more accurate.

At S32768/H24/D128, the sampled all-cubic FP8 control measures
`7.602336 ms`, `1735.5 TFLOPS`, and `0.957469` cosine. Local HAO FP8 measures
`8.294400 ms` and `1590.7 TFLOPS`, while the TK full-FP4 winner remains
`5.104000 ms`, `2585.0 TFLOPS`, and `0.963041` cosine. The current fused
FP4 strategy is therefore 48.95% higher-throughput than its matched FP8
control at the long-context point.

The experiment is exposed as `HAO_FP8PV_SHIFTLESS_MODE` in
`Makefile.hao_direct`: modes 1/2 select the FP4 cubic/affine mix and modes
3/4 select cubic throughout; even modes use the sampled denominator.
Mode zero preserves the original exact FP8 implementation.

## 2026-07-27: NVFP4 speed/accuracy Pareto search

The stage-0 and stage-1 affine winner above is the throughput endpoint, but
its `0.961779` cosine left a useful accuracy gap. The score-to-P
approximation and denominator sampling were swept independently on
S4096/H24/D128 using the same input and benchmark process.

### Approximation and mask sweep

Replacing selected affine quarters with the original cubic establishes the
first Pareto ladder:

| Stage-0/stage-1 affine masks | Time (ms) | BF16 cosine | RMSE |
|---|---:|---:|---:|
| 14 / 14 | **0.102752** | 0.961779 | 0.007184 |
| 14 / 0 | 0.108640 | 0.968036 | - |
| 0 / 14 | 0.111264 | 0.968024 | - |
| 2 / 2 | 0.112192 | 0.970138 | - |
| 2 / 0 | 0.114656 | 0.972261 | - |
| 0 / 0 | 0.116736 | **0.974384** | 0.005836 |

Q0 is always latency-critical, so its fast approximation was then searched
more aggressively. A one-hinge function, a two-hinge spline, a
distribution-refitted cubic, and a quintic were tested:

| Q0 fast approximation | Time (ms) | BF16 cosine |
|---|---:|---:|
| affine | **0.102400** | 0.961779 |
| one-hinge spline | 0.127264 | 0.975728 |
| refitted cubic | **0.125952** | **0.978694** |
| quintic | 0.139232 | 0.978712 |
| two-hinge spline | 0.172032 | 0.978688 |

The spline families improve fidelity but are instruction-inefficient. The
quintic almost eliminates approximation error at the FP4 bin boundaries but
does not improve end-to-end cosine over the cubic. This shows that
polynomial fit error is no longer the dominant numerical error.

The retained refitted polynomial is:

```text
P(x) = 0.07430709 x^3 + 0.28611863 x^2
     + 0.64670005 x   + 0.99010784
```

It raises sampled E2M1 code agreement from about 97.37% for the original
cubic to 97.98% on the observed score distribution.

### Native EX2 balance

The refitted cubic shifts pressure onto the packed FMA pipe. The number of
native `EX2` pairs used by each 32-value quarter was therefore swept:

| Native pairs | Time (ms) | BF16 cosine |
|---:|---:|---:|
| 4 | 0.125952 | 0.978694 |
| 5 | 0.121504 | 0.978873 |
| 6 | **0.121056** | **0.978990** |
| 7 | 0.122880 | 0.979079 |
| 8 | 0.122592 | 0.979138 |
| 10 | 0.124928 | 0.979230 |
| 12 | 0.135200 | 0.979290 |
| 14 | 0.146848 | 0.979337 |
| 16 | 0.152256 | 0.979371 |
| exact packed denominator | 0.133120 | 0.979313 |

Six pairs are the local throughput optimum. Relative to four pairs, the NCU
replay falls from `214.976` to `204.096 us`, eligible warps rise from
`0.600` to `0.633` per cycle, and tensor-pipe activity rises from `21.20%`
to `21.92%`. Dynamic FMA instructions fall from 27.078 million to
26.146 million. More native pairs continue to improve cosine slightly but
overload the SFU path.

Relative to the affine throughput mode, the high-accuracy mode executes
55.210 million rather than 41.262 million instructions and raises FMA-pipe
activity from `22.55%` to `37.14%`; tensor-pipe activity falls from `25.87%`
to `21.92%`. The extra `18.4 us` is therefore approximation arithmetic, not
additional tensor or TMEM work: both modes execute exactly 98,304 tensor and
1,105,920 TMEM instructions in the profile.

### Retained modes and validation

Only two approximation modes and two native-pair counts remain in the source:

- throughput: `HAO_FP4PV_NV_FAST_APPROX=0`,
  `HAO_FP4PV_NV_NATIVE_PAIRS=4`;
- high accuracy: `HAO_FP4PV_NV_FAST_APPROX=1`,
  `HAO_FP4PV_NV_NATIVE_PAIRS=6`.

Both use quarter schedule 1 and stage masks 14/14. Cleaned builds use 128
registers, one barrier, 400 bytes of static shared memory, and no spills.
They are bit-identical to their corresponding pre-cleanup artifacts.

| Shape/mode | Time (ms) | BF16 cosine | RMSE |
|---|---:|---:|---:|
| S4096 throughput, seed 0 | **0.102400** | 0.961779 | 0.007184 |
| S4096 high accuracy, seed 0 | 0.121120 | 0.978990 | 0.005268 |
| S4096 high accuracy, seeds 1-3 | 0.120832-0.121120 | 0.979223-0.979404 | 0.005267-0.005271 |
| S32768 high accuracy, seed 0 | 6.113632 | 0.979765 | 0.001846 |
| S32768 high accuracy, seeds 1-2 | 6.095888-6.098880 | 0.979384-0.979686 | - |

Using the prior local HAO BF16 medians, high accuracy remains about 1.35x
faster at S4096 and 1.44x faster at S32768. The benchmark now computes error
quantiles from a deterministic strided sample capped at 8 million values;
global cosine and RMSE remain exact. This lets the S32768 validation complete
without changing the reported global accuracy.

## 2026-07-27: fast-mode relative error and refreshed ceilings

The benchmark now reports reference RMS and relative L2 error:

```text
relative_l2 = ||O_fp4 - O_bf16||_2 / ||O_bf16||_2
            = RMSE / RMS(O_bf16)
```

This is preferred to elementwise percentage error because the attention
output contains many values near zero. At S4096/H24/D128 seed 0, reference
RMS is `0.0256605`:

| Mode | Time (ms) | Cosine | RMSE | Relative L2 |
|---|---:|---:|---:|---:|
| throughput affine | **0.102400** | 0.961779 | 0.007184 | **27.99%** |
| refitted-cubic high accuracy | 0.120992 | 0.978990 | 0.005268 | **20.53%** |

The current throughput mode's diagnostic ceilings are:

| Diagnostic | Time (ms) |
|---|---:|
| fixed P | 0.087616 |
| row-max pack | 0.090112 |
| score pack | 0.091168 |

These establish 11-15 us of aggregate score-to-P headroom, but that headroom
cannot be attributed to one removable instruction group. Every isolated
substitution made the balanced schedule equal or slower:

| Experiment | Time (ms) | Relative L2 | Result |
|---|---:|---:|---|
| two-pair denominator schedule, four native pairs | 0.106784 | 28.37% | slower |
| balanced exact max tree | 0.108544 | 27.99% | slower, identical output |
| SM100 three-input max | 0.105472 | 27.99% | slower, identical output |
| only two native EX2 pairs | 0.108768 | 29.40% | slower and less accurate |
| fixed P-scale, no max reduction | 0.102400 | at least 41.70% | no speed gain |
| Q-preconditioned packed add | 0.105184 | 28.00% | slower than packed FMA |

Sampling 8 of 32 maxima ties `0.102400 ms` but raises relative L2 to 51.24%.
Sampling 16-28 values measures `0.104768-0.108096 ms` and is also less
accurate than the exact-max control. The fixed-scale and Q-preconditioning
diagnostics were removed after measurement.

The important scheduling result is that the serial max chain, native `EX2`
pairs, and packed FMAs are covering one another's latency. Deleting any one
source reduces independent work and exposes score/TMEM readiness waits.
Reaching the `0.090 ms` ceiling therefore requires a structural reduction in
the complete scale-transform-pack-publication dependency, not another local
max, denominator, or arithmetic approximation.

The final affine schedule was also retested with the remaining legal NVFP4
scale-publication reorderings. Each candidate retained 128 registers, one
barrier, 400 bytes of static shared memory, and zero spills. Paired runs used
the pinned affine binary in the same process, and every candidate was
bit-identical to it:

| Scale-publication probe | Candidate (ms) | Paired control (ms) | Result |
|---|---:|---:|---|
| Quarter scale bits carried in registers | 0.102400 | 0.102400 | tied |
| Issuer preloads first V-scale half | 0.104896 | 0.103424 | slower |
| Issuer preloads tail V-scale half | 0.104640 | 0.102816 | slower |
| Issuer preloads both V-scale halves | 0.106528 | 0.103424 | slower |
| Stage-1 softmax owner preloads V scales | 0.141664 | 0.103840 | much slower |
| V tail scale loaded before first PV | 0.105184 | 0.103424 | slower |
| V tail scale loaded between first/tail PV | 0.104480 | 0.103424 | slower |

The register handoff proves that the shared scale-byte round trip is hidden.
Issuer prefetch delays useful issue because P is already ready when the issuer
reaches the handoff, while softmax-owner prefetch directly delays score
processing. V-scale movement is therefore not the remaining route to
`0.100 ms`; the first-K64 score-to-P dependency itself must be shortened or
split across an independent owner.

## 2026-07-27: instruction scheduling and tail-ownership follow-up

Four follow-ups tested whether the first-K64 dependency could be shortened
without changing the retained approximation.

### 1. Braided EX2/affine/pack scheduling

Three source orderings were compiled: the retained word-at-a-time order,
starting the next native `EX2` immediately before the current pack, and
starting it before the current affine body.

| Ordering | Time (ms) | Result |
|---|---:|---|
| retained | 0.103936 | control |
| next EX2 before pack | 0.103968 | tied |
| next EX2 before affine | 0.104448 | slower |

The retained and before-pack builds have identical normalized SASS. `ptxas`
already performs the useful braid; forcing an earlier source order only
lengthens live ranges.

### 2. Stage/quarter-specific affine fits

Refitting the affine approximation to the observed score distribution made
both numerical error and runtime worse:

| Fit | Time (ms) | BF16 cosine | RMSE |
|---|---:|---:|---:|
| retained throughput fit | 0.103424 | 0.961779 | 0.007184 |
| one common distribution fit | 0.103552 | 0.960639 | 0.007464 |
| per-stage/per-quarter fit | 0.104736 | 0.960636 | 0.007464 |

The local fit objective did not predict end-to-end attention error, so neither
refit was retained.

### 3. One-FFMA2 affine correction

A monotone quadratic correction was applied to selected quarters. Q3 and the
tail hide the extra arithmetic better than Q0-Q2, but even the best placement
is slower for a very small accuracy return:

| Corrected quarters | Time (ms) | BF16 cosine | RMSE |
|---|---:|---:|---:|
| Q0 | 0.108512 | 0.962099 | 0.007169 |
| Q1 | 0.108544 | 0.962093 | 0.007170 |
| Q2 | 0.108096 | 0.962092 | 0.007170 |
| Q3 | 0.106496 | 0.962090 | 0.007170 |
| Q2+Q3 | 0.106528 | 0.962409 | 0.007157 |
| all | 0.111616 | 0.963071 | 0.007129 |

### 4. Independent first/tail K64 ownership

The correction warpgroup was made responsible for Q2/Q3 in both query stages
while the two score-reader warpgroups built only Q0/Q1. The first correct
implementation measured `0.121600 ms`. Reordering by PV consumption reduced a
legal implementation to `0.110592 ms`, versus `0.102400-0.103968 ms` for the
paired control, with identical BF16 cosine and RMSE.

This experiment exposed a required ownership rule. Allowing the main reader
and tail owner to publish disjoint quarters into the same P TMEM page
concurrently produces NaNs. Waiting for first-half publication before the
tail owner's stores restores exact output, but serializes the publication
that the split was intended to overlap. Pre-capturing Q3 gives the best legal
schedule; moving Q3 behind that handoff regresses to `0.118784 ms`.

The original offload profile also increased executed instructions from
41.27 million to 43.66 million while issue-active fell from 42.87% to 39.06%;
tensor work was unchanged. The extra score captures, handoff barriers, and
single tail owner therefore cost more than the removed reader arithmetic.

All four experimental controls and ownership paths were removed after
measurement. The retained throughput and high-accuracy modes are unchanged.

## 2026-07-27: quarter-local critical-path attribution

A temporary compile-time diagnostic replaced the score-to-P transform with
raw FP4 score packing for selected quarters while preserving score loads,
TMEM stores, scale publication, barriers, and PV issue order. Each diagnostic
used 128 registers, one barrier, 400 bytes of static shared memory, and no
spills. Paired two-second runs used the retained throughput binary in the same
process. The diagnostic output is intentionally not numerically meaningful.

| Raw-packed quarters | Candidate (ms) | Paired control (ms) | Change |
|---|---:|---:|---:|
| Q0 | 0.104736 | 0.103424 | 1.312 us slower |
| Q1 | 0.102400 | 0.102400 | tied |
| Q0 + Q1 | 0.099328 | 0.103392 | **4.064 us faster** |
| Q2 | 0.100352 | 0.103424 | **3.072 us faster** |
| Q3 | 0.101088 | 0.103424 | **2.336 us faster** |
| Q2 + Q3 | 0.096416 | 0.103424 | **7.008 us faster** |
| Q0 + Q1 + Q2 + Q3 | 0.090112 | 0.102400 | **12.288 us faster** |

The non-additive pairs expose two rendezvous. Accelerating only Q0 or Q1
cannot advance first PV because both first-half quarters must publish.
Likewise, Q2 and Q3 individually save 5.408 us in total, but accelerating
both saves 7.008 us because tail PV waits for the slower tail producer.

Matched NCU captures of the first-half and tail substitutions preserve 3.69
active warps per scheduler and all tensor/TMEM work. First-half raw packing
executes 30.186 million instructions in a 162.176 us replay; tail raw packing
executes 29.960 million in 159.648 us. Tensor-pipe activity rises from the
retained profile's 21.26% to 22.73% and 23.04%, respectively. The lower wall
time therefore comes from making P ready earlier, not from changing tensor
work or occupancy.

The scheduling consequence is precise. Q0/Q1 gate the first K64 PV command.
The issuer then waits inside the two-K64 NVFP4 accumulation sequence for
Q2/Q3 before it can issue the tail command and move on to the next QK. The
previous `first PV -> next QK -> tail PV` experiment established that this
TCGEN reordering is illegal, so the 7 us tail exposure cannot be hidden by a
local issuer permutation.

The full raw-pack ceiling leaves 12.288 us of absolute headroom at S4096/H24,
but FP4 conversion itself is not that gap: `F2FP` remains in the raw-pack
ceiling. The removable instruction groups are the quarter max reductions,
packed affine/native-EX2 transform, scale/denominator arithmetic, and the
extra wait-loop iterations caused by their publication latency. The next
credible work is therefore:

1. rebalance exact max work across the first and tail publication windows,
   especially precomputing Q3's max without delaying first P;
2. specialize a cheaper tail-only transform, where 7 us is exposed, rather
   than applying another approximation uniformly;
3. revisit the inseparable K64 pair only through a different accumulator or
   ownership layout, not another local issue-order permutation.

The temporary quarter-selection diagnostic was removed after measurement.

## 2026-07-28: tail-latency optimization probes

The quarter-local attribution motivated exact-schedule and tail-only
arithmetic experiments around the retained S4096/H24 throughput kernel. Every
candidate remained at 128 registers, one barrier, 400 bytes of static shared
memory, and zero spills.

### Early Q3 max

Q3's exact max reduction was moved ahead of first-P publication, after its
asynchronous load and Q2's early max. The output was bit-identical to the
retained kernel, but the candidate measured `0.105120 ms` against a paired
`0.103456 ms` control. Delaying first P and carrying Q3's max across the
publication costs 1.664 us, so the existing placement after first PV is
better.

### Tail-only native EX2 reduction

Reducing Q2/Q3 from four native denominator samples to two initially exposed
the invariant that these `EX2` results also feed the sampled softmax
denominator. Once the two-pair estimator was corrected to use only its two
positive native samples, the candidate was still slower and less accurate:

| Q2/Q3 native pairs | Time (ms) | Paired control (ms) | BF16 cosine | RMSE |
|---:|---:|---:|---:|---:|
| 4, retained | 0.102400 | - | 0.961779 | 0.007184 |
| 2, matched estimator | 0.104000 | 0.102400 | 0.960021 | 0.007365 |
| 0, diagnostic only | 0.104000 | 0.102400 | 0.011861 | 26.370800 |

The native pairs are not dispensable SFU work. They provide the positive
samples required by the cheap denominator estimator and balance the affine
ALU path.

### Tail max sampling and prediction

Sampling fewer maxima only in Q2/Q3 produced one small timing signal, but no
usable Pareto point:

| Tail max policy | Time (ms) | Paired control (ms) | BF16 cosine | Relative L2 |
|---|---:|---:|---:|---:|
| sample 16/32 in Q2 and Q3 | 0.103968 | 0.103968 | 0.937906 | 34.78% |
| sample 8/32 in Q2 and Q3 | **0.102784** | 0.103424 | 0.910861 | 41.27% |
| sample 8/32, +0.5 log2 margin | 0.103424 | 0.103456 | 0.922561 | 38.59% |
| Q2 exact max predicts Q3 + 8 Q3 samples | 0.104448 | 0.103424 | 0.947864 | 32.17% |
| independent 8/32 Q3 max only | 0.104480 | 0.104128 | 0.936331 | 35.25% |
| reuse Q2 max for Q3, no Q3 reduction | 0.106496 | 0.103456 | 0.941982 | 33.76% |

The fixed safety margin reduces clipping error but consumes the entire
0.64 us signal. Reusing Q2's max introduces a cross-quarter dependency and
is slower even when all Q3 max instructions are deleted.

The zero-Q3-reduction NCU profile explains the counterintuitive result:

| Metric | Retained | No Q3 reduction |
|---|---:|---:|
| Replay duration (us) | **173.504** | 176.384 |
| Executed instructions | 41.264M | **38.574M** |
| Issue active | **42.82%** | 39.62% |
| Eligible warps/cycle | **0.52** | 0.47 |
| ALU-pipe active | **29.12%** | 25.58% |
| Tensor-pipe active | 21.26% | 21.00% |
| Long-scoreboard ratio | **5.54** | 6.38 |

Deleting 2.69 million max instructions makes the kernel slower because those
independent instructions cover score/TMEM dependency latency. The 7 us raw
tail-pack ceiling therefore cannot be recovered by deleting one local
arithmetic component. It requires a structural path that shortens the whole
Q2/Q3 max-scale-transform-publication chain while preserving enough
independent work, or an independent owner with substantially cheaper
handoff than the previously rejected full tail offload.

All early-max, tail-native, sampled-max, bias, and Q3-predictor controls were
removed after measurement. The retained kernel is unchanged.

## 2026-07-28: exact Q3-max correction-warpgroup offload

The correction warpgroup was used as a max-only owner: it independently
loaded Q3, computed the exact 32-value row maximum, and handed one scalar per
row to the original score reader. The score reader retained all Q3 payload,
scale, packing, and P-publication ownership, avoiding the concurrent-P-store
failure of the earlier full-tail offload. Existing row-max shared storage and
dual-Q3 semaphores provided the mailbox, so no TMEM or dynamic shared memory
was added. Every candidate was bit-identical to the retained kernel.

The initial two-way ready/reuse mailbox measured `0.107200 ms` against a
paired `0.102912 ms` control. Its profile showed that the independent work did
increase eligible warps from `0.52` to `0.60` per cycle and reduce the
long-scoreboard stall from `5.5` to `4.7` cycles. However, replay duration
increased from `173.57 us` to `178.98 us`, and executed instructions increased
from `41.261M` to `44.530M`.

The reuse acknowledgment was then removed. It is redundant because the next
`score_full` phase cannot be published until the original reader has consumed
the current maximum and published the P tail. This reduced the candidate to
`0.106080 ms`, still slower than the paired `0.102400 ms` control.

Query-stage isolation confirmed where the loss occurs:

| Q3 max owner | Candidate (ms) | Paired control (ms) | Result |
|---|---:|---:|---|
| correction WG, stage 0 only | 0.107200 | 0.103584 | slower |
| correction WG, stage 1 only | 0.103584 | 0.103616 | timing tie |
| correction WG, both stages | 0.106080 | 0.102400 | slower |

The stage-1 timing tie was order-dependent. A matched replay measured
`175.14 us` versus `173.57 us` and `42.666M` versus `41.261M` executed
instructions. Thus even the stage with enough slack to hide most of the
offload does not shorten the kernel.

The structural conclusion is that max arithmetic is useful independent work
in the resident score reader. Moving it to another warpgroup requires a
second 32-value Q3 TMEM load plus mailbox control; those costs exceed the
removed local reduction. All max-only offload controls were removed, and the
retained kernel is unchanged.

## 2026-07-28: structured-sparse FP4 feasibility

CUDA 13 exposes Blackwell sparse FP4 through
`tcgen05.mma.sp.kind::mxf4` and `kind::mxf4nvf4`. The hardware format is not
arbitrary 2:4 over individual FP4 values. FP4 is byte-packed, and one
four-bit metadata code selects two of four packed bytes, so the effective
constraint is four retained FP4 values out of eight in adjacent pairs
("paired 4:8").

For one M128/K128 P tile, the implemented representation uses:

- 4 KiB of compressed P payload in shared memory;
- 1 KiB of metadata, occupying two TMEM columns;
- one logical K128 sparse PV issue instead of two dense K64 issues.

The earlier estimate that this recovers roughly 120 TMEM columns was wrong.
Dense P is not a standalone 128-column TMEM allocation in this kernel: it is
overlaid on score storage and occupies roughly 16 physical columns per live
stage. Sparse P replaces that short-lived overlay with compressed shared
payload plus two metadata columns. It reduces the P overlay footprint and
lifetime, but does not create another full 128-column score or output bank.
All four P quarters must also be ready before the sparse K128 issue. Sparse
NVFP4 remains the natural first target because its logical block-32 scale span
matches one P quarter; sparse MXFP4 has a block-64 span and adds a
cross-quarter scale dependency.

A compile-gated, saturated M128/N128/K128 instruction probe used 1,824 CTAs,
256 logical tiles per CTA, 60 alternating-order samples, and identical
two-CTA-per-SM resource limits:

| TCGEN path | Median kernel time (ms) | Median CTA cycles | Effective logical TFLOP/s |
|---|---:|---:|---:|
| dense MXFP4 | 0.223312 | 64,872 | 8,770.26 |
| BF16 | 0.907568 | 261,476 | 2,157.97 |
| sparse MXFP4 | 0.174080 | 42,540 | 11,250.60 |
| sparse NVFP4 | 0.173904 | 42,544 | 11,261.99 |

Sparse NVFP4 and MXFP4 are indistinguishable at the tensor-core instruction
level. Sparse NVFP4 is 1.284x faster than dense MXFP4 by saturated kernel
time and 1.525x faster by per-CTA cycles. This is real but leaves only about
87 saved cycles per logical K128 tile for pair selection, compression,
metadata publication, and any lost early-K64 overlap.

An ideal software paired-4:8 attention reference retained approximately
73-74% of the softmax mass. Across S512-S4096, cheap selection by the maximum
FP4 value in each packed-byte pair achieved BF16 cosine 0.954-0.958, within
about 0.001 of selection by exact pair probability mass. Selecting directly
from already-packed E2M1 words agreed with the floating score selector on
about 95.3% of pair choices and lost only about 0.001 additional cosine.
This makes an integer selector over the quantizer's existing packed words the
appropriate implementation path.

Ideal-scale sparse NVFP4 reached BF16 cosine 0.946-0.950, versus 0.939-0.943
for sparse MXFP4, because of its finer block-32 scaling. Stochastic E2M1
rounding of P and V reduced single-forward cosine further to roughly
0.925-0.931. It remains potentially useful for unbiased training updates or
multi-sample averaging, but it is rejected for the first inference kernel.

### End-to-end sparse-PV integration

The real forward was extended behind `HAO_FP4PV_SPARSE_PV=1`. It packs P into
the sparse NVFP4 payload, constructs hardware metadata, stores block-32 P
scales, and issues one K128
`tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4` PV command. The production
default remains dense.

At S4096/H24, selector cost dominates the hardware saving:

| Sparse selector | Time (ms) | BF16 cosine | RMSE | Interpretation |
|---|---:|---:|---:|---|
| fixed pairs, tensor-core ceiling | 0.103312 | not meaningful | not meaningful | ties dense despite no value selection |
| 2-bit OR-score LUT | 0.141312 | 0.914269 | 0.010495 | cheapest value-aware implementation |
| exact branchless pair top-2 | 0.241952 | 0.924065 | 0.009917 | better selection, prohibitive control cost |
| dense retained control | 0.102400-0.103136 | 0.961779 | 0.007184 | retained winner |

Renormalizing by the retained quantized P mass fixes the constant-V semantic
check to within 3.125%, but measures `0.143360 ms` and only raises cosine from
`0.914269` to `0.914813`; its relative-L2 error worsens because sparse V
quantization still changes output magnitude.

Matched NCU profiles identify instruction work, not TMEM latency, as the
value-aware selector's blocker:

| Metric | Fixed-pair ceiling | OR/LUT selector | Change |
|---|---:|---:|---:|
| replay duration (us) | 174.496 | 238.880 | +36.9% |
| profiled cycles | 195,430 | 269,190 | +37.7% |
| executed instructions | 43.137M | 63.280M | +46.7% |
| eligible warps/cycle | 0.57 | 0.59 | +0.02 |
| long-scoreboard stall | 60.24% | 55.59% | lower |

The sparse tensor instruction is faster in isolation, but replacing two
dense K64 commands with one sparse K128 command removes the existing
half-ready issue point. The fixed-pair ceiling spends the raw 1.284x tensor
gain waiting for all four quarters; any useful selector then adds repeated
integer classification and packing on the serial publication path. The
dynamic selector is therefore compute-bound rather than TMEM- or
lookup-latency-bound.

This also explains why the paper's epilogue claim does not transfer directly.
Its pruning is attached to a register-resident QK GEMM epilogue before a
global score write. This fused FA4 kernel keeps scores in TMEM and transforms
them directly into P, so sparse selection is additional critical-path work
unless QK ownership and score layout are rewritten to produce sparse P during
the score epilogue.

The less aggressive max-sampling fallback does not rescue this direction.
Sampling 16/32 tail values ties the dense timing at `0.103968 ms` while
reducing cosine to `0.937906`; 8/32 saves only `0.640 us` and falls to
`0.910861`. The sparse candidate and selector ladder remain compile-gated
diagnostics. The retained dense kernel rebuilds with 128 registers, one
barrier, no spills, `0.103136 ms`, cosine `0.961779`, and RMSE `0.007184`.

### Independent sparse-quarter publication

The local PTX ISA description closes an important ambiguity: dense FP4 TCGEN
uses logical K64, while sparse `kind::mxf4`/`kind::mxf4nvf4` has a fixed
logical K128. There is no sparse K32 or K64 instruction. A sparse K128 issue
also owns its payload, metadata, and scale sources until the asynchronous
operation has consumed them, so publishing Q0/Q1 and mutating the same source
with Q2/Q3 after issue is not legal.

The earliest legal approximation to independent quarters is therefore two
sparse K128 commands:

1. issue Q0/Q1 with the logical Q2/Q3 payload permanently zero;
2. while that command runs, finish Q2/Q3;
3. issue Q2/Q3 from disjoint payload, metadata, and scale storage with the
   logical Q0/Q1 payload permanently zero;
4. accumulate the second command into the first command's output.

This route is implemented behind
`HAO_FP4PV_SPARSE_SPLIT_K128=1`. It uses separate first/tail shared payloads,
metadata columns, and P-scale pages. The first sparse command is released
after Q1 rather than waiting for the all-quarter tail publication.

A fresh saturated primitive quantifies the unavoidable instruction cost:

| Primitive | Median time (ms) | Median CTA cycles | Relative to dense |
|---|---:|---:|---:|
| two dense K64 commands | 0.227984 | 64,987.0 | 1.000x |
| one sparse K128 command | 0.170112 | 42,341.0 | 1.340x faster |
| two split sparse K128 commands | 0.321456 | 91,688.5 | 1.410x slower |

The full forward hides most of that raw second-command cost, but does not
turn it into a win:

| Selector | Single sparse K128 (ms) | Split sparse K128 (ms) | Split delta |
|---|---:|---:|---:|
| fixed-pair ceiling | 0.102688 | 0.108832 | +5.98% |
| OR/LUT, four-run alternating mean | 0.143624 | 0.145408 | +1.24% |
| exact branchless, matched-source run | 0.256448 | 0.274464 | +7.02% |

The matched exact outputs are bit-identical (`max_abs=0`, `RMSE=0`,
`cosine=1`) between single and split execution. The OR/LUT outputs also have
identical BF16 metrics (`cosine=0.914813`, `RMSE=0.016197`). Thus the result
is not caused by a metadata, scale-order, synchronization, or numerical bug.

Matched OR/LUT NCU replays show that the intended overlap really occurs:

| Metric | Single K128 | Split K128 |
|---|---:|---:|
| replay duration (us) | 247.392 | 249.280 |
| average active cycles | 227,365.4 | 228,503.3 |
| executed instructions | 67.416M | 68.531M |
| eligible warps/cycle | 0.61 | 0.63 |
| issue active | 48.98% | 49.57% |
| TC-pipe active | 20.89% | 26.68% |
| tensor-pipe active | 11.14% | 14.79% |
| long-scoreboard per issue | 4.15 | 4.14 |
| barrier per issue | 0.02 | 0.02 |

Independent publication raises eligibility and issue activity without adding
barrier or scoreboard pressure. It loses because the hardware performs a
second full sparse K128 tensor operation, visible in the higher TC/tensor
activity. The extra tensor work consumes the overlap recovered by releasing
Q0/Q1 early. A profitable version needs native sparse K64/K32, or a different
algorithm that gives both sparse K128 commands useful nonzero work; scheduling
alone cannot remove this floor.

The retained dense regression remains unchanged: `0.102720 ms`, cosine
`0.961779`, RMSE `0.007184`, 128 registers, one barrier, and no spills. The
split sparse mode remains default-off as an ISA diagnostic. Stochastic E2M1
rounding also remains a training-only experiment; it is not enabled in the
inference forward.

## Four-way post-sparsity search (2026-07-28)

The retained S4096/H24 NVFP4/NVFP4 kernel was rebuilt before this search:

| Configuration | Time (ms) | BF16 cosine | RMSE | Relative L2 |
|---|---:|---:|---:|---:|
| retained full-max control | 0.103104 | 0.961779 | 0.007184 | 0.279945 |

All structural experiments below use the same approximation, scale encoding,
and early-P configuration as that control. New experimental paths are
compile-time gated and default off.

### 1. Interleave the Q3 tail into Q2

`HAO_FP4PV_NV_TAIL_Q3_HOOK_WORD` moves the Q3 transform into the Q2 packed
word sequence without adding a barrier or changing ownership:

| Q3 insertion point | Time (ms) | Delta from control |
|---|---:|---:|
| disabled/control | about 0.1030 | - |
| after Q2 word 1 | 0.104448 | about +1.4 us |
| after Q2 word 2 | 0.106208 | about +3.2 us |
| after Q2 word 3 | 0.104448 | about +1.4 us |

The existing contiguous Q2 then Q3 sequence schedules better than manual
instruction interleaving. Reordering alone does not shorten the critical
path and was not promoted.

### 2. Produce Q0/Q1 next to the QK epilogue

`HAO_FP4PV_NV_Q01_CORR_WG` gives the otherwise idle correction warpgroup
ownership of the first-half score load, max, scale, P pack, and first K64
publication. The normal stage-0 reader retains Q2/Q3.

The first version allowed the two warpgroups to write disjoint quarters of
one P TMEM page concurrently. It ran at `0.106784 ms` but produced NaNs. This
confirms the hardware ownership constraint: disjoint columns do not make
concurrent cross-warpgroup writes to one TMEM page legal.

The corrected version uses a single-writer transfer. The normal reader loads
Q2/Q3 and computes their maxima early, then waits for Q0/Q1 publication
before storing the tail. It is numerically exact relative to the retained
kernel (`cosine=0.99999988`, `RMSE=3.65e-7`) but measures `0.104704 ms`
against a paired `0.103200 ms` control. The mandatory TMEM ownership handoff
costs about `1.5 us`, so this ownership split was not promoted.

### 3. Shape-specific launch and stage schedules

The kernel now accepts `HAO_KV_STAGES=4..13` and
`HAO_PHYSICAL_GRID_CAP`. A zero grid cap preserves the original all-SM
launch. The useful comparisons were:

| Shape `(S,H)` | Candidate | Candidate (ms) | Control (ms) | Result |
|---|---|---:|---:|---|
| (4096,24) | grid 128 vs 152 | 0.102400 | 0.102400 | tie |
| (1024,64) | grid 128 vs 152 | 0.026624 | 0.026624 | tie |
| (1024,64) | KV8 vs KV13, grid 128 | 0.026912 | 0.026912 | tie |
| (2048,32) | grid 128 vs 152 | 0.043008 | 0.043008 | order-neutral tie |
| (8192,8) | grid 128 vs 152 | 0.133536 | 0.133792 | grid 128 wins 0.2% |
| (8192,8) | quarter schedule 2 vs 1 | 0.136192 | 0.134144 | schedule 2 loses |

KV8 reduces dynamic shared memory from `209920` to `163840` bytes, but
512 TMEM columns still restrict the kernel to one resident CTA per SM, so
latency does not improve. Only the long-context `(8192,8)` shape has a small
repeatable grid-128 gain. This is suitable for shape dispatch, not a new
general schedule.

### 4. Aggressive max-estimator accuracy/speed frontier

The full 32-value max was replaced with progressively smaller fixed sample
sets. Timings are paired with the rebuilt full-max control:

| Max samples | Bias (log2) | Time (ms) | Control (ms) | BF16 cosine | RMSE | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 32 | 0.0 | 0.103104 | - | 0.961779 | 0.007184 | retained |
| 8 | 0.0 | 0.101952 | 0.102400 | 0.862155 | 0.013148 | accuracy collapse |
| 8 | +0.5 | 0.104448 | 0.103456 | 0.884913 | 0.012033 | slower and inaccurate |
| 4 | 0.0 | 0.102720 | 0.103200 | 0.801002 | 0.015754 | accuracy collapse |
| 2 | 0.0 | 0.104480 | 0.102400 | NaN | NaN | invalid and slower |

The eight-sample estimator saves less than `0.5 us` in the matched run while
losing almost ten cosine points. Four samples buy no meaningful additional
latency, and two samples destabilize the recurrence. A static conservative
bias adds work/codegen pressure but cannot compensate for the missed maxima.
Max sampling is therefore not a useful path to `0.1 ms`.

### Consolidated conclusion

None of the four searches changes the retained S4096/H24 production kernel.
The results narrow the remaining problem:

1. Q2/Q3 arithmetic order is already locally well scheduled.
2. Moving P work to another warpgroup cannot overlap TMEM writes; legal
   ownership transfer costs more than it hides.
3. Grid tuning provides a small long-context-only gain, while KV stage depth
   cannot change the one-CTA-per-SM TMEM limit.
4. The exact max reduction is not expensive enough to approximate: removing
   most of it saves under one microsecond and severely damages numerics.

Further material gains require a different score/P storage or ownership
model, not another local ordering, sampled-max, or barrier tweak.

The final default-off rebuild uses 128 registers, one barrier, 400 bytes of
static shared memory, and no spills. A 200 ms paired regression measured
`0.103424 ms` versus `0.103584 ms` for the pinned artifact and produced
bit-identical output (`max_abs=0`, `RMSE=0`).

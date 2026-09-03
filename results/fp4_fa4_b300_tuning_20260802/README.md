# B300 compatibility and tuning

This directory records the compatibility work and architecture-specific
tuning for the NVFP4-QK/MXFP4-PV and NVFP4-QK/NVFP4-PV kernels on B300
(`sm_103a`). It includes the original fixed-policy D128 port, D64 kernel
specializations, SM103 reduction experiments, and matched HAO BF16
references.

## Causal D128 GQA pipeline

- Discovery job: `e61vydxupba8`
- K/V depth job: `p78qmpnm51pi`
- Cross-shape and HAO BF16 job: `xz398n4qedei`
- TK source revision measured: `7ef04d7`
- Hardware: NVIDIA B300 SXM6 AC, 148 visible SMs

The GB200 causal score pipeline also wins on B300, but its final denominator
handoff does not. At B1/S4096/Hq32/Hkv8/D128, the legacy B300 schedule takes
0.092954 ms, the complete pipeline with in-place denominator finalization
takes 0.087072 ms, and mode-2 deferred finalization takes 0.089674 ms. A
separate depth sweep selects 12 K/V stages at 0.086213 ms. All variants retain
128 registers, one barrier, and zero spills.

The promoted B300 policy therefore uses causal tail skipping, diagonal-only
masking, progressive Q3 register reuse, quarter-max interleaving, task order
zero, 12 K/V stages, and no deferred denominator finalization. Tiny diagnostic
wins from a 144-CTA grid at S4096/H32 and KV8 at S16384/H32 were not promoted:
they were only 0.17% and 0.018%, respectively. Quarter-3 native `exp2` was
0.41% slower than the all-ALU schedule.

| S | Hq/Hkv | NV/MX (ms) | HAO BF16 (ms) | Speedup | Cosine | Relative L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32/8 | 0.037824 | 0.050112 | 1.325x | 0.951009 | 0.313470 |
| 4096 | 32/8 | 0.091136 | 0.123808 | 1.359x | 0.950178 | 0.317117 |
| 8192 | 32/8 | 0.279328 | 0.420768 | 1.506x | 0.949441 | 0.319596 |
| 16384 | 32/8 | 0.980480 | 1.573920 | 1.605x | 0.948713 | 0.321439 |
| 4096 | 64/8 | 0.154592 | 0.222240 | 1.438x | 0.950101 | 0.317557 |
| 8192 | 64/8 | 0.517088 | 0.818208 | 1.582x | 0.950271 | 0.316604 |

Every selected output is finite. The causal leakage test leaves the earlier
output prefix and LSE bitwise unchanged after replacing the latter half of V.
Jobs `e61vydxupba8` and `xz398n4qedei` completed all measurements before a
final report-only parser failure; their per-run artifacts are complete, and
the checked-in YAML files contain the corrected aggregators.

## D128 wide-key geometry ceiling

- B300 N192/N256 job: `zfdeyjgbdzd7`
- B300 N192 validation job: `sqhvletxn33r`
- TK revision: `777d597`
- Hardware: NVIDIA B300 SXM6 AC, 148 visible SMs
- Protocol: 1776 blocks, 256 repeated attention lifecycles per block,
  12 warmups, and two independent 60-sample timing windows

SM103 exposes 512 allocatable TMEM columns to one CTA, with power-of-two
allocation sizes. It therefore does not provide a usable 288-column CTA
allocation that would allow two resident attention CTAs while retaining the
current score, output, and scale state. Narrowing the matrix shapes enough to
fit 256 columns is also a poor trade: the N96/N128 split routes lose wide-MMA
efficiency on both GB200 and B300.

The productive SM103 direction is instead a wider key tile. The raw N192
route computes QK over M128xN192xK128, then consumes P in two native K96
SM103 Ultra PV instructions. The N256 route computes QK over
M128xN256xK128, then consumes P with K96+K96+K64 PV instructions. Its TMEM
layout uses 256 score columns, 128 output columns, four 16-column scale pages,
and leaves 64 columns unused.

| Raw attention lifecycle | Repeat A (PFLOP/s) | Repeat B (PFLOP/s) | Median CTA cycles B |
|---|---:|---:|---:|
| N96, one CTA | 2.831 | 2.833 | 156111 |
| N128 split, one CTA | 2.405 | 2.405 | 246724 |
| N128 split, two CTA | 3.560 | 3.560 | 293087 |
| N192 Ultra, one CTA | 5.021 | 5.021 | 173981 |
| N256 Ultra, one CTA | 5.683 | 5.681 | 205923 |

N256 is 59.6% faster than the best two-CTA N128 split and 13.1% faster than
N192. The improvement comes from halving the number of key-tile lifecycle
boundaries, retaining a wide QK instruction, and replacing four K64 PV
issues across two N128 tiles with K96+K96+K64 across one N256 tile. A
standalone SM103 K96 Ultra probe reaches about 9.19 PFLOP/s, so the remaining
gap in this ceiling is the intentionally serial QK-to-P-to-PV lifecycle, not
the native K96 instruction itself.

These are timing ceilings, not correct forward kernels: P and scale pages are
synthetic and the V payload is repeated. The next gate is a production-shaped
N256 fixed-P route containing the real persistent scheduler, TMA traffic, and
output store. It must materially beat the existing approximately 0.1461 ms
fixed-P D128 floor before the real P path is expanded from four to eight
quarters. A successful real port then publishes PV after quarters 0-2,
quarters 3-5, and quarters 6-7, matching the K96+K96+K64 issue sequence.

## Saturated D128 N128 closure

- Shape: B1/S6144/H64/D128
- Double-score fixed-P job: `nze7iyaucg5s`
- Prefetch/SFU sweep: `oen3rbbipbtf`
- TK revision: `f3a12b7`
- Hardware: NVIDIA B300 SXM6 AC, 148 visible SMs
- Kernel resources: 128 registers, one barrier, no spills

The production N128 topology assigns two M128 query tiles to each persistent
CTA, so both queries reuse every K/V transfer. A true two-score ping-pong
prototype instead assigned one query to each CTA and alternated two N128 score
banks around one D128 output. It was functionally valid in the fixed-P gate,
but it doubled the logical jobs from 1536 to 3072 and therefore doubled K/V
pipeline work. Its median time was 0.502564 ms versus 0.328978 ms for the
production fixed-P topology, a 52.8% regression. The additional QK/PV overlap
did not repay the lost two-query reuse, so this route is rejected for real P.

The production real-P path was then swept across K/V prefetch depth and native
`exp2` share. Three independent 1000-iteration timing windows gave:

| Variant | Median (ms) | Delta vs best | Cosine | Relative L2 |
|---|---:|---:|---:|---:|
| K/V stages 4, all ALU | 0.426715 | +2.06% | - | - |
| K/V stages 6, all ALU | **0.418107** | - | 0.943199 | 0.338089 |
| K/V stages 8, all ALU | 0.426889 | +2.10% | - | - |
| K/V stages 10, all ALU | 0.428226 | +2.42% | - | - |
| K/V stages 12, all ALU | 0.425992 | +1.89% | - | - |
| Native `exp2` in quarter 0 | 0.426814 | +2.08% | 0.943978 | 0.335428 |
| Native `exp2` in quarter 3 | 0.418983 | +0.21% | 0.943981 | 0.335418 |
| One native pair in all quarters | 0.424794 | +1.60% | 0.946301 | 0.327474 |
| Two native pairs in all quarters | 0.439913 | +5.22% | 0.949374 | 0.316939 |

Six K/V stages and the all-ALU mode remain the latency optimum. Deeper
prefetching cannot hide the remaining dependency chain. Quarter-3 native
`exp2` is an optional near-zero-cost accuracy point, but broader SFU use is an
explicit speed/accuracy trade rather than a throughput win. The best B300 row
is 7.7% faster than the matched 0.452848 ms GB200 row at this saturated shape.

### Rejected reader-owned Q-scale copy

A scale-copy ceiling suggested that moving the stage-0 Q-scale page off the
tensor-core issuer might shorten its critical path. Three implementations
separate the copy cost from the required ownership synchronization:

| GB200 B1/S6144/H64/D128 variant | Time (ms) | Correctness |
|---|---:|---|
| Production issuer copy | 0.448768 | reference |
| Four reader-local TMEM stores | 0.460736 | bit-exact |
| Reader multicast before all score loads retire | invalid | corrupts lagging reader rows |
| Reader multicast after all score loads retire | 0.578240 | bit-exact |

The `tcgen05.cp.32x128b.warpx4` mapping itself is not the problem. A
standalone lane-layout probe reproduced the expected 512-byte source mapping
and four-way row multicast exactly. The unsafe version failed because reader
warp 0 overwrote all four row groups of the S1 score bank while reader warps
1-3 could still be loading their Q1/Q3 scores. The corruption consequently
appeared only in an odd query tile and usually in its final 32 rows.

Waiting until all four readers retire their final score load fixes every
tested seed bit-for-bit, but it requires a warpgroup rendezvous and a separate
copy-completion token. Stage 0 is already waiting by this point, so the copy
becomes serial rather than overlapped and regresses latency by 28.8%. Direct
row-local stores avoid the cross-warp overwrite race, but four stores plus
their completion wait still cost 2.7%. Reader ownership is therefore rejected
for the production kernel. A useful replacement must either retain issuer
ordering or provide a genuinely independent scale destination; changing only
the owner cannot create overlap within the current S0/S1 lifecycle.

### Rejected split K-scale placement

Keeping the stage-0 Q-scale copy before P while moving only its folded
K-scale copy between the first and tail PV K64 issues is bit-exact and keeps
the production resource footprint. It nevertheless regresses the matched
GB200 B1/S6144/H64/D128 kernel from a median 0.488979 ms to 0.491034 ms
(0.42%) across four alternating 1000-iteration windows. Both binaries contain
the same 12 `UTCCP` and 16 `UTCOMMA` instructions.

The move does not create independent execution: PV and the scale copy enter
the same tensor-core command stream. It instead breaks apart the compact
copy-plus-QK issue sequence without hiding the copy. This closes the remaining
unmeasured combination from the earlier placement sweep: Q before P and K
between PV-first and PV-tail.

## Confirmed 3 PFLOP/s D128 points

- Discovery job: `ibwp4shzwfmi`
- Independent confirmation: `d9e3yhudpu93`
- Kernel revision: `f3a12b7`
- Route: NVFP4 QK, MXFP4 P/V, six K/V stages, quarter-3 native EX2
- Resources: 128 registers, one barrier, no spills, 148 persistent CTAs

The discovery sweep tested K/V depths 5--7, all-ALU versus quarter-3 native
EX2, a 147-CTA control, and three saturated shapes. It measured 3118 TFLOP/s
at the standard B1/S8192/H64/D128 shape and 3164 TFLOP/s at S9472/H64. The
S9472 shape has 2368 logical query-pair jobs, exactly 16 per visible SM. A
separate job rebuilt only these two candidates, used two correctness seeds,
and alternated five 2000-iteration timing windows:

| Shape | Median (ms) | TFLOP/s | Window span | Cosine range | Relative L2 range |
|---|---:|---:|---:|---:|---:|
| S8192/H64/D128 | 0.705684 | 3116 | 0.064% | 0.944063--0.944113 | 0.334931--0.335152 |
| S9472/H64/D128 | 0.930724 | 3159 | 0.392% | 0.943960--0.944056 | 0.335198--0.335473 |

The standard S8192 row is 6.94% lower latency than the matched 0.758336 ms
GB200 result. The gain is not only a favorable nonstandard shape. Stage 6
remains the depth optimum; stage 5 is slower, and stage 7 ranges from slightly
slower to below 3 PFLOP/s. Native EX2 is useful only in quarter 3: broader SFU
routing remains slower because it lengthens the serial P publication path.

## Tuned D64 result

- Full NV/MX job: `f8ajtqmbvd8p`
- Full NV/NV job: `ig2tvyqoxtlz`
- Reduction-mask sweep: `jltd2e1msdao`
- Short-shape grid sweep: `q6nb0jj73xik`
- NV/NV non-finite diagnostic: `y7iu73u97dc4`
- Bounded NV/NV validation: `szpe9052wipe`
- Promoted-policy verification: `2bz33uyyvvo8`
- Hardware: NVIDIA B300 SXM6 AC, 148 visible SMs
- Protocol: 100 ms warmup, 1000 ms median timing window, one second cooldown
- Kernel resources: 128 registers, one barrier, 102400 bytes of dynamic
  shared memory, all 512 TMEM columns, and no spills

The B300 NV/MX specialization combines two SM103-specific changes. First,
`tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max` loads and reduces each score
quarter directly from TMEM. Second, mode 23 sends eight of each quarter's 16
score pairs through native `exp2` and handles the other eight with the affine
ALU approximation. GB200 keeps four native pairs because the eight-pair mix
slows its representative D64 point from about 0.0979 ms to 0.104704 ms.

| H | S | NV/MX (ms) | NV/NV (ms) | BF16 speedup | NV/MX cosine | NV/MX rel. L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1024 | 0.017376 | 0.017376 | 1.002x | 0.951242 | 0.308901 |
| 12 | 2048 | 0.023520 | 0.025536 | 1.088x | 0.954197 | 0.300150 |
| 12 | 4096 | 0.066528 | 0.070464 | 1.124x | 0.955818 | 0.295719 |
| 12 | 8192 | 0.175072 | 0.185248 | 1.163x | 0.955544 | 0.297119 |
| 12 | 16384 | 0.668512 | 0.702432 | 1.196x | 0.958042 | 0.289235 |
| 12 | 32768 | 2.414528 | 2.540640 | 1.252x | 0.956991 | 0.292417 |
| 24 | 1024 | 0.017376 | 0.017376 | 1.007x | 0.950979 | 0.309532 |
| 24 | 2048 | 0.039872 | 0.039904 | 1.104x | 0.954513 | 0.299560 |
| 24 | 4096 | 0.093248 | 0.099296 | 1.153x | 0.955551 | 0.296653 |
| 24 | 8192 | 0.341056 | 0.359424 | 1.180x | 0.957003 | 0.292264 |
| 24 | 16384 | 1.217376 | 1.280192 | 1.249x | 0.956494 | 0.294114 |
| 24 | 32768 | 4.601808 | 4.842496 | 1.280x | 0.956089 | 0.295010 |
| 32 | 1024 | 0.017376 | 0.017376 | 1.120x | 0.951423 | 0.308101 |
| 32 | 2048 | 0.039872 | 0.039968 | 1.104x | 0.954553 | 0.299511 |
| 32 | 4096 | 0.121984 | 0.129952 | 1.150x | 0.956886 | 0.292484 |
| 32 | 8192 | 0.396352 | 0.419808 | 1.248x | 0.957006 | 0.292608 |
| 32 | 16384 | 1.544128 | 1.632224 | 1.273x | 0.956796 | 0.292861 |
| 32 | 32768 | 6.115360 | 6.465520* | 1.284x | 0.957303 | 0.291325 |
| 64 | 1024 | 0.025568 | 0.025632 | 1.159x | 0.952035 | 0.306387 |
| 64 | 2048 | 0.068576 | 0.072576 | 1.150x | 0.954919 | 0.297864 |
| 64 | 4096 | 0.207840 | 0.218176 | 1.208x | 0.955838 | 0.295956 |
| 64 | 8192 | 0.784224 | 0.828480 | 1.266x | 0.956713 | 0.293189 |
| 64 | 16384 | 3.073088 | 3.252352 | 1.284x | 0.957443 | 0.290835 |
| 64 | 32768 | 12.223360 | 12.919808 | 1.284x | 0.957248 | 0.291501 |

For every S4096-and-longer point, NV/MX is 1.124x-1.284x faster than the
matched B300 BF16 kernel and 4.97%-6.53% faster than NV/NV. It is also
1.73%-8.91% faster than the corresponding GB200 NV/MX run. Density two wins
or ties density one on 23 of 24 shapes. The apparent exception at H32/S2048
is a launch-geometry effect: capping the persistent grid at 136 CTAs restores
0.039872 ms without reverting to the less accurate density-one arithmetic.
H64/S1024 similarly prefers a 128-CTA grid and reaches 0.025568 ms. Other
shapes retain the full 148-CTA grid.

The starred unbounded NV/NV timing is not a valid production result. With
seed 20260802 it produces 64 non-finite output values while BF16 and NV/MX
remain finite. A second seed is finite, proving the failure is
distribution-dependent. Saturating the E4M3 scale encoder (mode 4) fixes the
same long case at 6.670336 ms, cosine 0.962921, and relative L2 0.275742. The
bounded path costs about 2%-3% and is exposed as `nv-nv-bounded`; raw
`nv-nv` remains an explicitly unsafe diagnostic.

The mask sweep establishes that all-quarter fused reduction (mask 15) is the
best schedule under the density-two arithmetic mix. It measures 0.095008 ms
in the ordered 16-mask sweep, followed by mask 13 at 0.095232 ms. An
independent run records 0.093216 ms for mask 15. Every mask produces the same
cosine and relative L2, so this is a scheduling result rather than a numerical
tradeoff.

All full-matrix NV/MX outputs are finite. Every build uses 128 registers, one
barrier, and no local-memory spills. The promoted defaults are density two,
all-quarter fused reduction, and the two short-shape grid caps above; GB200
keeps density one and no cap.

Job `2bz33uyyvvo8` validates revision `7380158` without any density or grid
override. It reproduces 0.039872 ms at H32/S2048 with grid 136, 0.025792 ms
at H64/S1024 with grid 128, and 0.093280 ms at H24/S4096 with grid 148. All
three outputs are finite and all three builds retain 128 registers, one
barrier, and zero spills.

## Completed long-context and NV/FP8 comparison

- NV/MX plus exact NV/FP8 job: `xfby78ic4nql`
- Optimized D128/D64 NV/FP8 job: `mlqf9rjlss7h`
- Shape: B1/S32768/H24, D128 and D64
- Hardware: NVIDIA B300 SXM6 AC, 148 CUDA-visible SMs, 2032 MHz reported
  maximum SM clock
- Timing: five independent `triton.testing.do_bench` median windows,
  10 ms warmup, 25 ms repetition, and 0.8 s cooldown

| D | Provider | Time (ms) | TFLOP/s | Cosine BF16 | Relative L2 BF16 |
|---:|---|---:|---:|---:|---:|
| 128 | TK NV/MX fast | 4.480736 | 2945 | 0.942874 | 0.338905 |
| 128 | TK NV/FP8 optimized | 7.583744 | 1740 | 0.957334 | 0.291279 |
| 128 | TK NV/FP8 exact | 8.853520 | 1490 | 0.989723 | 0.143296 |
| 64 | TK NV/MX fast | 4.601808 | 1434 | 0.956089 | 0.295010 |
| 64 | TK NV/FP8 optimized | 7.436288 | 887 | 0.956353 | 0.294779 |

The optimized NV/FP8 binaries use 128 registers, one barrier, and no spills.
The exact D128 binary spills 48 bytes in each direction. Removing that spill
and using shiftless mode 4 improves throughput by 16.7%, but it reduces
cosine. HAO publishes 2677 TFLOP/s for D128 NV/FP8 and 1203 TFLOP/s for D64
NV/FP8 on GB300, so HAO is 1.539x and 1.356x faster than the current TK
NV/FP8 translation. HAO's corresponding BF16 rows are 1533 and 1221
TFLOP/s. These are published cross-run references, not measurements from our
Volt process.

The D64 NV/FP8 translation is valid only for NVFP4 QK plus plain FP8 PV.
HAO's block-scaled NVFP4 and MXFP8 PV atoms require K128 and are unsupported
at D64. The TK NV/MX and NV/NV specializations above use separate D64 atoms
and do support full FP4 PV. Halving D does not halve latency because the
score/softmax tile and job count are unchanged; it mostly removes QK/PV
matrix work from a path whose P processing is already dominant.

The D128 NV/MX row is 1.8% below the matched local GB200 result of 2998
TFLOP/s. The tested B300 has 148 SMs at 2032 MHz, while the local GB200 has
152 SMs at 2062 MHz. Normalized by that SM-clock envelope, B300 is 2.35%
faster. Applying the measured per-SM-clock rate to the GB200 envelope gives
3069 TFLOP/s; this explains the sub-3000 aggregate result without treating
3069 as an observed B300 measurement.

Job `sxrb3y8vspqc` compiled both optimized binaries successfully but failed
before launch because the benchmark module directory was absent from
`PYTHONPATH`. The successful job makes that path explicit.

## Completed visible-SM sweep

- Volt job: `ndbwsa3zg6j9`
- Result: succeeded
- Config: `volt_b300_density_sweep.yaml`
- Runtime-visible SMs: 148
- Swept control: native-EX2 density 0, 1, 2, and 4
- Artifact: `artifacts/ndbwsa3zg6j9/workers/0/fp4-fa4-b300-visible-sm-density/b1_s4096_h24_d128_visible_sm_density_sweep.json`

The persistent scheduler was compiled for the 148 SMs reported by CUDA.
Every variant uses 128 registers, one barrier, 400 bytes of static shared
memory, and no spills.

| Native EX2 density | Time (ms) | TFLOP/s | Cosine | Relative L2 | RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.093216 | 2212 | 0.943964 | 0.335825 | 0.008716 |
| 1 | 0.097248 | 2120 | 0.947037 | 0.325258 | 0.008442 |
| 2 | 0.097280 | 2119 | 0.950069 | 0.314812 | 0.008171 |
| 4 | 0.111584 | 1848 | 0.956047 | 0.294037 | 0.007632 |

Density 0 is the latency winner. Density 2 dominates density 1 within this
timing resolution because it has nearly identical latency and lower error.
The matched GB200 density-0 row is 0.092160 ms, so the corrected B300 port is
1.15% slower on this single shape rather than the 34% regression suggested
by the compatibility launch.

## Oversubscribed-grid diagnostic

- Volt job: `kf2sxae8ugxs`
- Result: succeeded
- Compile-time persistent-grid width: 160
- Runtime-visible SMs: 148
- Artifact: `artifacts/kf2sxae8ugxs/workers/0/fp4-fa4-b300-density/b1_s4096_h24_d128_density_sweep.json`

The 160-worker run is retained as an audit control and excluded from the
main generated table. Density 0 takes 0.123872 ms. Because the kernel uses
all 512 TMEM columns, one CTA occupies each SM; the extra persistent workers
form a small trailing wave. This isolates launch geometry as the cause of
the apparent B300 slowdown.

## Completed compatibility job

- Volt job: `28n9zhmijb6p`
- Submitted: 2026-08-02 01:55 UTC
- Result: succeeded
- TK commit: `cfc06dadf684279f657ab66254a3a074be4ee3a9`
- HAO commit: `9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247`
- Config: `volt_b300_smoke.yaml`
- Artifact: `artifacts/28n9zhmijb6p/workers/0/fp4-fa4-b300-smoke/b1_s4096_h24_d128_fast.json`

The unmodified GB200 `fast` policy compiles for `sm_103a` with 128 registers,
one barrier, and no spills. At B1/S4096/H24/D128 it records 0.123872 ms and
1664 TFLOP/s, with cosine 0.943964, relative L2 0.335825, and RMSE 0.008716
against BF16. The matching GB200 row is 0.092160 ms. Numerics transfer almost
exactly; the untuned schedule does not.

The port inherited a hard-coded 152-SM persistent-grid width, while CUDA
reports 148 SMs on this B300 SKU. Four persistent workers therefore formed a
trailing wave. Matching the compile-time width to 148 improves this point by
1.329x without changing its output.

## Previous job

- Volt job: `8rgxprbjqrm1`
- Submitted: 2026-08-02 01:49 UTC
- Result: failed before the attention launch because the architecture-specific
  `mxfp4_quant_v3` input-packing extension was absent
- TK commit: `cfc06dadf684279f657ab66254a3a074be4ee3a9`
- HAO commit: `9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247`
- Config: `volt_b300_smoke.yaml`

The earlier queued request `mcg6ny69yqc7` was cancelled before allocation
because its remote fetch used an abbreviated TK commit ID.

Job `gvf7gfjiqlz6` established that the kernel compiles for `sm_103a` with
no spills, then failed before launch because the environment installed the
CUTLASS base libraries without the CUDA 13 Python package. Job
`dgh03uipjptq` added HAO's `nvidia-cutlass-dsl[cu13]` dependency, but the
wheel's nested Python directory was still absent from `sys.path` in the
container. Job `lt70yy57x6a4` exposed a second package-layout mismatch in
`cutlass.cute.runtime`. Job `8rgxprbjqrm1` removed HAO's CuTe
frontend from the TK-only runtime path: it uses the committed local
quantizers, HAO-compatible seeded inputs, FlashInfer scale packing, and a
PyTorch BF16 SDPA reference. It compiled the attention kernel for `sm_103a`
with no spills, then exposed the missing MXFP4 V-packing build. The current
job recipe now builds that extension for `sm_103a` as part of the job. This
keeps the kernel measurement independent of the external Python compiler
environment.

The earlier D128 density conclusion does not transfer to D64. D128 favors
density zero for latency, while the D64 path needs native samples to seed its
shiftless denominator and reaches its best valid speed/accuracy point at
density two. Density zero D64 runs are invalid numerical ceilings rather than
production candidates.

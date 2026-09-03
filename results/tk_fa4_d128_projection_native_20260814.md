# Projection-native D128 GQA operand production

Date: 2026-08-14

Device: GPU 0 (GB200 / SM100)

Production geometry unless noted: B=1, S=4096, Hq=32, Hkv=8, D=128,
hidden width 4096, causal attention.  This is the Llama-8B attention geometry.

## Retained producer

The NVFP4 QKV projection now accepts asymmetric GQA widths and applies
pair-native RoPE in its epilogue.  The same BF16-rounded accumulator fragment
publishes, without standalone packing kernels:

- compact NVFP4 Q and K plus their forward scale pages and global scales;
- sequence-major MXFP4 V and its forward E8M0 scale page;
- fixed-scale E4M3 Q, K, and V for the retained hybrid backward;
- optional BF16 Q/K/V for diagnostics only.

The production specialization disables BF16 publication and does not allocate
the legacy aligned Q/K layouts.  Its two compact Q/K payloads are shared by the
forward and backward-facing views rather than copied.

Nine exact optimizations are retained:

1. Each N256 projection tile contains two D128 heads.  Epilogue slices are
   visited in head-paired order, and packed cosine/sine values loaded for the
   first head are retained in registers for the second head.
2. FP4 codes and BF16 pairs used by the FP8 conversion are staged in one
   fragment traversal.  Separate shared pages permit both consumers to publish
   after one synchronization.
3. The D128 projection uses a three-stage input pipeline.  This funds the
   second epilogue page while remaining faster than the four-stage producer.
4. Static BF16 cosine/sine tables are packed once into one 32-bit word per
   rotary pair.  The epilogue loads that word directly instead of issuing two
   independent 16-bit global loads.  The packed and split-table paths are
   bitwise identical; split tables remain available as a compatibility path.
5. The row block's batch index and sequence-relative base are derived once and
   reused by Q/K FP4, Q/K FP8, V MXFP4, V FP8, adaptive-scale, and forward-scale
   publication.  The previous hot path repeated the same runtime division in
   each publisher.
6. Packed-RoPE D128 launches use 68 resident clusters at S4096 and 72 at S8192.
   Interleaved same-process sweeps show that these caps shorten the uneven final
   task wave while retaining enough SM parallelism.  Other sequence lengths
   remain uncapped.
7. Each CTA cooperatively stages its 128-row by 64-pair packed RoPE tile in a
   32 KiB shared page before waiting for projection output.  A 68-word row
   stride makes the warp's eight-row by four-pair reads bank-disjoint.  The page
   aliases publication scratch after RoPE consumption, so it adds no lifetime
   to the epilogue pipeline.  The uncached packed-table specialization remains
   available as a diagnostic control.
8. A D128 head's adaptive Q/K scale is loaded once and retained across its four
   N32 epilogue slices.  This is approximately wall-time neutral by itself, but
   simplifies the long-context production kernel from 206 to 168 registers.
   The original per-slice specialization remains the default below S2048,
   where the longer scalar lifetime does not amortize.
9. The BF16 publication fragment uses row-major shared storage with a 23-word
   stride.  Its producer lane map is eight rows by four rotary pairs, whereas
   FP8 and MXFP4 consumers are row-wise; the former pair-major layout therefore
   created severe staging conflicts.  The new layout reduces the analytically
   unavoidable staging collisions from 21 to 3 banks per warp instruction and
   leaves row-wise consumers conflict-free.

The final long-context kernel uses 168 registers per thread with zero spills, 183,296
bytes of dynamic shared memory, 34,928 bytes of static shared memory, and two
barriers.  It retains one resident block per SM; shared memory, rather than
registers, remains the residency limit.

## Projection timing

Warm-cache CUDA-event medians include RoPE and all forward/backward operand
publication, but not the one-time preparation of static projection weights.

| Sequence | Packed-RoPE checkpoint (us) | Retained (us) | Improvement |
| ---: | ---: | ---: | ---: |
| 512 | 71.104 | 68.576 | 3.69% |
| 1024 | 95.456 | 87.072 | 9.63% |
| 2048 | 127.328 | 113.856 | 11.83% |
| 4096 | 206.816 | 185.632 | 11.41% |
| 8192 | 352.864 | 307.168 | 14.88% |

At S4096, removing RoPE gives a 162.240 us diagnostic floor.  The retained
producer progression was approximately 304.8 us with all BF16/aligned
publications, 229.5 us after compact-only publication, 217.8 us after D128
head-pair RoPE caching, 215.3 us after fused FP4/FP8 staging, and 206.8 us
after direct packed-RoPE loads.  Hoisting invariant row mapping and balancing
the long-context launch lower the retained S4096 producer to 185.6 us.

The scheduling contribution was isolated from clock and process-order drift by
randomly interleaving caps on identical operands in one process.  At S4096,
cap 68 changes 185.024 to 184.384 us (0.35%); at S8192, cap 72 changes 316.192
to 310.816 us (1.70%).  Cap 0 and cap 76 agree within noise, as expected for
the same physical grid.  The larger overall improvement therefore comes from
eliminating repeated row-index divisions, not from reducing the grid.

The shared-cache contribution was also measured by randomly interleaving the
cached and uncached specializations on identical operands in one process:

| Sequence | Uncached (us) | Shared cache (us) | Improvement |
| ---: | ---: | ---: | ---: |
| 512 | 68.512 | 68.160 | 0.52% |
| 1024 | 91.936 | 89.632 | 2.57% |
| 2048 | 115.936 | 111.872 | 3.63% |
| 4096 | 188.960 | 182.656 | 3.45% |
| 8192 | 316.896 | 301.600 | 5.07% |

The increasing long-context gain is consistent with removing repeated global
RoPE dependencies from every projection tile rather than changing launch
overhead.

The BF16 scratch transpose was compared against the immediately preceding
binary in three alternating GPU-2 processes.  Each sample amortized 32
projection launches so the unrelated resident workload could not dominate a
single event.  Baseline round medians were 92.688, 92.615, and 92.749 us;
transposed round medians were 89.821, 89.828, and 89.703 us.  The
median-of-medians improvement is **3.19%**.  These steady-state absolute values
are intentionally not mixed with the earlier GPU-0 event table; the controlled
ratio is the result.

## Chained forward result

The production projection output was passed directly to the existing causal
NVFP4-QK/MXFP4-PV forward artifact.  No Q/K/V repacking or fake quantization
launch occurs between the two kernels.

| B=1, S=4096, Hq/Hkv=32/8, D=128 | Time (us) |
| --- | ---: |
| Prior BF16 projection + BF16 forward chain | 456.896 |
| Projection-native low-precision chain before shared cache | 355.840 |
| Projection-native low-precision chain at shared-cache checkpoint | 350.464 |

The cached chain is 1.53% faster than its interleaved uncached control.  Relative
to the prior BF16 chain it is **1.304x faster** and reduces this portion of
training time by **23.29%**.  Adding the independently measured retained
backward gives a broader attention-side aggregate of 1,108.531 us for BF16
(456.896 + 651.635) and 847.119 us for low precision (350.464 + 496.655), or
**1.309x / 23.58% less time**.

The subsequent BF16-scratch transpose improves a controlled, 16-launch
steady-state projection-plus-forward chain from 189.640 to 187.504 us, another
**1.14%**.  One of three candidate rounds was interrupted by the resident GPU
workload and is excluded naturally by the median-of-rounds statistic.  The
earlier BF16 aggregate is therefore a conservative checkpoint rather than a
fresh end-to-end absolute measurement.

The latter is still a matched component aggregate, not a complete transformer
block benchmark.  It excludes output projection backward, MLP, normalization,
optimizer work, and communication.  The backward timing also needs an external
operand entry point before the exact projection-produced E4M3 tensors can be
timed as one chained graph.

## Numerical checks

- Fused Q/K RoPE agrees with independently rotating the unrotated BF16
  projection at cosine 0.9999979 / 0.9999980; maximum absolute difference is
  one BF16 step (0.0009765625).
- All eight forward operands and Q/K/V E4M3 tensors are bitwise identical
  between the diagnostic BF16-publication and compact production variants.
- Q/K/V E4M3 publication is bitwise identical to converting the BF16 fragment
  after the required factor-of-four exponent shift.
- Packed and split cosine/sine inputs produce bitwise-identical BF16 Q/K/V and
  all fourteen published low-precision/metadata tensors.
- A B=2 sequence-boundary test after row-index hoisting is bitwise identical
  across all nineteen tensor fields, including Q/K/V E4M3 publication.  Capped
  and uncapped launches are bitwise identical across all sixteen production
  tensor fields at S4096.
- The cached and uncached packed-RoPE specializations are bitwise identical
  across all nineteen diagnostic tensor fields in the same B=2 boundary test.
- Separate pre- and post-transpose extension binaries produce matching SHA-256
  payload hashes for all nineteen B=2 diagnostic tensor fields.
- Feeding the projected operands to the low-precision causal forward gives
  output cosine 0.991047 and relative L2 0.13503 against BF16 SDPA; output and
  LSE are finite.

## Profile attribution

Head-pair RoPE caching changed the full-profile projection duration from
208.448 to 198.112 us.  Long-scoreboard stall fell from 4.15 to 3.31
issue-equivalents (-20.2%); LG throttle fell from 0.25 to 0.10.  The final
three-stage/fused-publication kernel measures 195.456 us under the reduced
profile set, with long scoreboard 3.29, wait 0.83, short scoreboard 0.61,
barrier 0.33, and LG throttle 0.09.  Direct packed-RoPE loads reduce the profile
duration again to 186.752 us.  Long scoreboard falls to 3.08, wait to 0.80,
barrier to 0.31, and LG throttle to zero; no-instruction and short-scoreboard
remain 2.65 and 0.61, respectively.

The retained invariant-row specialization profiles at 166.528 us, another
10.83% reduction.  Compute throughput rises from 23.43% to 26.07%, memory
throughput from 28.70% to 33.59%, achieved occupancy from 9.3% to 9.5%, and
issued warps per scheduler from 0.17 to 0.19.  No-eligible cycles fall from
82.95% to 81.17%.  Most importantly, no-instruction stall falls from 2.65 to
1.90 issue-equivalents; long-scoreboard is 3.02, wait 0.75, short-scoreboard
0.77, and barrier 0.36.  PC sampling falls from 6,379 to 5,391 total samples:
no-instruction 2,096 to 1,453, long-scoreboard 1,970 to 1,743, barrier 310 to
247, while short-scoreboard is essentially unchanged at 486 versus 482.

The shared-RoPE specialization profiles at 157.344 us, 5.52% below the
166.528 us uncached profile.  Long-scoreboard falls from 3.02 to 2.57
issue-equivalents and its PC samples fall from 1,743 to 1,468.  Total PC samples
fall from 5,391 to 5,144.  The cooperative load raises barrier samples from 247
to 325, but the removed global dependency more than pays for it: memory
throughput falls from 33.59% to 30.75%, compute throughput rises from 26.07% to
27.66%, and no-eligible cycles improve from 81.17% to 80.13%.  Achieved
occupancy remains 9.47% and issued warps per scheduler rise slightly to 0.20.

The transposed BF16 scratch profiles at 156.70 us versus 159.04 us for its
immediate control.  More importantly, excessive shared wavefronts fall from
1,179,648 to 393,216 (-66.7%).  Shared-store bank-conflict wavefronts fall from
2,328,248 to 1,350,921 (-42.0%), and their average conflict degree drops from
4.7-way to 3.5-way.  No-eligible cycles improve from 80.10% to 79.63% and warp
cycles per issued instruction from 7.67 to 7.53.  Registers remain 168, static
and dynamic shared memory are unchanged, and outputs remain bitwise exact.

Row-local persistent work assignment was also measured and rejected: it was
bitwise exact but regressed S4096 production timing from 215.328 to 221.728 us
(+3.0%) because the reduced block-level load balance outweighed cache reuse.
The shared cache resolves the packed-RoPE global-load dependency: the first
cosine consumer drops from 361/336 to 52/28 source-correlated stall samples.
Scale reuse and the transposed publication page then remove the next two local
dependencies.  The largest measured exact target is now E4M3 publication:
global stores still waste 1,241,088 sectors, 40% of their theoretical total,
because one lane owns a sequence row and issues two half-sector stores.  A
two-lanes-per-row mapping can coalesce the two 16-byte halves without changing
the output layout or conversion count.
